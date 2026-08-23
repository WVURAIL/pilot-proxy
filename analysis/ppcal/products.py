# coding=utf-8
"""Loading, geometry and per-unit reduction for the per-pilot survey products.

Conventions locked against the pilot-proxy source:

* The archived coarse power ratio (``ARCHIVED_COARSE_POWER_RATIO``) is the
  statistic F = 2*P_target/(P_ref_lo + P_ref_hi), summed over all 2048 input
  streams before the ratio; its exact null mean is the stored ``mu0`` =
  2*target_norm_sq/reference_norm_sum_sq.
* The stored spectra live in the *receiver* spectral frame, which is inverted
  with respect to RF when ``sense == -1``.  RF offset from the CHIME channel
  centre is therefore ``sense * f_stored``; DC (``f_stored == 0``) is the
  channel centre in either frame.
* The archived fine power ratio (``ARCHIVED_FINE_POWER_RATIO``) is the
  time-coherent window-axis FFT: 256 padded bins of (390625/128)/256 =
  11.9209 Hz, centred convention ``((b+128) % 256) - 128``, again
  sense-flipped with respect to RF.
* Frame health uses the released ``pilot_proxy.archive_health`` v1 gate.
"""
from __future__ import annotations

import datetime as dt
import glob
import os
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from pilot_proxy.archive_health import (
    evaluate_frame_health,
    health_correct_integrated_spectra,
)
from pilot_proxy.archived_product_keys import (
    ARCHIVED_COARSE_POWER_RATIO,
    ARCHIVED_FINE_POWER_RATIO,
)
from pilot_proxy.detector_geometry import predicted_pilot_fine_bin

SAMPLE_RATE_HZ = 390625.0            # one CHIME frequency channel
COARSE_HZ = SAMPLE_RATE_HZ / 128     # 3051.7578 Hz detector bin
FINE_HZ = COARSE_HZ / 256            # 11.9209 Hz envelope bin
FRAME_SECONDS = 16384 * 2.56e-6      # 41.94 ms
REFERENCE_OFFSET_HZ = 2 * COARSE_HZ  # +/- 2 coarse bins to the guard references

# survey month grid, 2018-12 .. 2026-07 inclusive
M0 = 2018 * 12 + 11
M1 = 2026 * 12 + 6
NMONTHS = M1 - M0 + 1


def month_index(ctime) -> np.ndarray:
    """Calendar-month index into the survey grid for UTC epoch seconds."""
    out = np.empty(np.size(ctime), dtype=np.int64)
    for i, x in enumerate(np.atleast_1d(ctime)):
        d = dt.datetime.fromtimestamp(float(x), dt.timezone.utc)
        out[i] = d.year * 12 + (d.month - 1) - M0
    return out


def month_label(i) -> str:
    m = M0 + int(i)
    return "%04d-%02d" % (m // 12, m % 12 + 1)


def product_paths(directory):
    return sorted(glob.glob(os.path.join(directory, "*.npz")),
                  key=lambda p: int(os.path.basename(p)[:-4]))


@dataclass
class Channel:
    """One physical DTV channel product, reduced and health-gated."""

    path: str

    def __post_init__(self):
        self._z = np.load(self.path, allow_pickle=True)
        z = self._z
        self.fid = int(z["freq_id"][0])
        self.ch = int(z["physical_channel"][0])
        self.mu0 = float(z["mu0"][0])
        self.sense = int(z["sense"])
        self.nfft = int(z["nfft"])
        self.window = int(z["detector_window_samples"])
        self.pad = int(z["fine_pad_factor"])
        self.pilot_hz = float(z["pilot_frequency_hz"][0])
        self.center_hz = float(z["chime_frequency_hz"][0])
        self.pilot_offset_hz = self.pilot_hz - self.center_hz
        health = evaluate_frame_health(z)
        self.health_include = health.include
        self.health_reasons = dict(health.reason_counts)
        self._health = health
        self.n_frames_raw = int(self.health_include.size)
        self.n_units_raw = int(z["unit_keys"].size)

    # ---- frame level -----------------------------------------------------
    @cached_property
    def fstat(self):
        """Healthy-frame coarse statistic F."""
        return self._z[ARCHIVED_COARSE_POWER_RATIO][self.health_include, 0]

    @cached_property
    def rho(self):
        """Fractional excess statistic rho-hat = F/mu0 - 1, healthy frames."""
        return self.fstat / self.mu0 - 1.0

    @cached_property
    def frame_unit(self):
        return self._z["frame_unit_index"][self.health_include]

    @cached_property
    def frame_ctime(self):
        return self._z["unit_time0_ctime"][self.frame_unit]

    @cached_property
    def frame_month(self):
        return month_index(self.frame_ctime)

    @cached_property
    def fine(self):
        """Healthy-frame fine statistic, shape (n_frames, 256)."""
        return self._z[ARCHIVED_FINE_POWER_RATIO][self.health_include]

    # ---- unit level ------------------------------------------------------
    @cached_property
    def unit_ctime(self):
        return self._z["unit_time0_ctime"]

    @cached_property
    def units(self):
        """(ctime, mean-F, max-F, n_frames) for units with >=1 healthy frame."""
        n = self.unit_ctime.size
        s = np.zeros(n)
        c = np.zeros(n, dtype=np.int64)
        mx = np.full(n, -np.inf)
        np.add.at(s, self.frame_unit, self.fstat)
        np.add.at(c, self.frame_unit, 1)
        np.maximum.at(mx, self.frame_unit, self.fstat)
        keep = c > 0
        return self.unit_ctime[keep], s[keep] / c[keep], mx[keep], c[keep]

    @cached_property
    def unit_level_db(self):
        """10*log10(mean F / mu0) per surviving unit."""
        _, mean_f, _, _ = self.units
        return 10.0 * np.log10(np.maximum(mean_f, 1e-9) / self.mu0)

    @cached_property
    def unit_month(self):
        return month_index(self.units[0])

    def monthly_level_db(self):
        """(month index, median level dB, n_units) over observed months."""
        mi = self.unit_month
        lvl = self.unit_level_db
        months, med, cnt = [], [], []
        for m in np.unique(mi):
            sel = mi == m
            months.append(int(m))
            med.append(float(np.median(lvl[sel])))
            cnt.append(int(sel.sum()))
        return np.array(months), np.array(med), np.array(cnt, dtype=np.int64)

    # ---- geometry --------------------------------------------------------
    @property
    def sense_token(self):
        return "inverted" if self.sense < 0 else "normal"

    @cached_property
    def predicted_fine_bin(self):
        return predicted_pilot_fine_bin(
            pilot_rf_hz=self.pilot_hz, coarse_center_hz=self.center_hz,
            sample_rate_hz=SAMPLE_RATE_HZ, detector_window_samples=self.window,
            nfft=self.nfft, spectral_sense=self.sense_token, pad_factor=self.pad)

    @staticmethod
    def centered(b):
        """Centered fine-bin index, ((b + 128) % 256) - 128."""
        return ((np.asarray(b) + 128) % 256) - 128

    @cached_property
    def fine_rf_hz(self):
        """RF offset (Hz) of each fine bin from the target coarse-bin centre."""
        return self.sense * self.centered(np.arange(256)) * FINE_HZ

    # ---- integrated spectrum --------------------------------------------
    def _spectrum(self, which):
        corrected = health_correct_integrated_spectra(self._z, self._health)
        if not corrected.exact:
            raise RuntimeError("%s: %s" % (self.path, corrected.unavailable_reason))
        s = np.asarray(getattr(corrected, which), dtype=float)
        f_stored = np.fft.fftfreq(self.nfft, d=1.0 / SAMPLE_RATE_HZ)
        rf = self.sense * f_stored
        o = np.argsort(rf)
        return rf[o], s[o]

    @cached_property
    def spectrum(self):
        """(rf offset Hz, power), health-corrected before-mask, RF-ascending."""
        return self._spectrum("before")

    @cached_property
    def spectrum_after(self):
        """(rf offset Hz, power), health-corrected after-mask, RF-ascending."""
        return self._spectrum("after")


def load_all(directory):
    return [Channel(p) for p in product_paths(directory)]
