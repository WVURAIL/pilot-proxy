# coding=utf-8
"""Per-channel null calibration and the threshold ladder.

Vocabulary, fixed by how the survey was actually run:

``F > 1``
    The provisional collection-time rule.  The archived ``mu0`` is the
    analytic weight-norm constant, which sits within 1.2% of unity on every
    channel, and the products' stored reject mask is that rule.  It is the
    strictest threshold available and it was never meant to be the final
    one -- it exists so that a threshold could be exercised while the archive
    was still being built.

``F > mu``
    The calibrated rule.  ``mu`` is *measured* per channel, on the frames
    the survey actually collected, and it is the real null centre of the
    statistic for that channel in that era.  This module produces it.

``F > eta * mu``
    The historical report rule. ``eta`` is not a false-alarm parameter; it
    comes from the science tolerance, which prices how much residual shelf
    power the analysis can absorb.

Estimating ``mu``.  The distribution of F is a very narrow null core with a
heavy transient tail on the right, so no moment-based estimator is usable.
The centre is taken as the half-sample mode -- the densest region of the
sample, which is the null core wherever a null exists and the carrier lobe
where the channel is fully occupied, and that distinction is itself the
evidence for the disposition.  The scale is then estimated from frames at or
below the centre, which a carrier can only ever add to from above, using the
released ``rfisher.residual.NULL_SCALE_PROBES`` convention: the p-th
percentile of a lower-half sample is the (p/2)-th percentile of the full
null.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from statistics import NormalDist

import numpy as np

# Released left-side probes and their full-null deviates.
NULL_SCALE_PROBES = ((32.0, 1.0000), (5.0, 1.9600), (0.3, 2.9677))

REPORT_ETA_LADDER = (1.0, 1.1, 1.2, 1.4, 2.0, 5.0)

# Compatibility for older report scripts.
ETA_LADDER = REPORT_ETA_LADDER
NULL_TOLERANCE_DB = 0.20
MIN_LOWER_TAIL_FRAMES = 20
DETECTION_FLOOR_GAUSSIAN_TAIL_PROBABILITY = 1.0e-3
DETECTION_FLOOR_Z = NormalDist().inv_cdf(
    1.0 - DETECTION_FLOOR_GAUSSIAN_TAIL_PROBABILITY)
KEPT_TAIL_PERCENTILE = 99.0


def half_sample_mode(x):
    """Robust mode: recursively keep the shortest half of the sample."""
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n == 0:
        return float("nan")
    while n > 3:
        h = (n + 1) // 2
        widths = x[h - 1:] - x[:n - h + 1]
        j = int(np.argmin(widths))
        x = x[j:j + h]
        n = x.size
    if n == 3:
        d1, d2 = x[1] - x[0], x[2] - x[1]
        if d1 < d2:
            return float(0.5 * (x[0] + x[1]))
        if d2 < d1:
            return float(0.5 * (x[1] + x[2]))
        return float(x[1])
    return float(np.mean(x))


def null_scale_about(f, centre):
    """(sigma, spread) from the sample at or below ``centre``.

    ``spread`` is the ratio of the largest to the smallest single-probe
    estimate: near 1 the left tail is Gaussian and well sampled, large means
    the lower half is too thin or too skewed to characterise a null and the
    scale should be read as indicative only.
    """
    lower = f[f <= centre]
    if lower.size < MIN_LOWER_TAIL_FRAMES:
        return float("nan"), float("nan")
    ests = [(centre - float(np.percentile(lower, p))) / z
            for p, z in NULL_SCALE_PROBES]
    ests = [e for e in ests if e > 0.0]
    if not ests:
        return float("nan"), float("nan")
    return float(np.median(ests)), float(max(ests) / min(ests))


@dataclass
class Calibration:
    """Calibrated null and threshold ladder for one channel-era."""

    ch: int
    fid: int
    era_label: str
    n_frames: int
    n_units: int
    mu0_provisional: float       # archived analytic constant, the F > 1 rule
    mu: float                    # calibrated null centre, this era
    mu_over_mu0: float
    mu_shift_db: float           # 10 log10(mu / mu0_provisional)
    sigma: float                 # null scale about mu
    sigma_over_mu: float
    sigma_spread: float
    null_available: bool
    occupancy: dict              # fraction of frames above eta * mu
    occupancy_provisional: float  # fraction above mu0 -- the F > 1 rule
    detection_floor_eta: float   # smallest resolvable eta, mu + 3.09 sigma
    detection_floor_db: float
    leakage_p99_db: dict         # kept-frame excess above mu, per eta

    def row(self):
        d = asdict(self)
        occ = d.pop("occupancy")
        leak = d.pop("leakage_p99_db")
        for k, v in occ.items():
            d["occ_eta_%s" % k] = v
        for k, v in leak.items():
            d["leak_p99_db_eta_%s" % k] = v
        return d


def calibrate(channel, frame_mask, era_label, n_units,
              eta_ladder=REPORT_ETA_LADDER):
    """Calibrate ``mu`` and the ladder on the frames selected by the mask."""
    f = channel.fstat[frame_mask]
    mu0 = channel.mu0
    n = int(f.size)
    if n == 0:
        raise ValueError("channel %d: empty era" % channel.ch)

    mu = half_sample_mode(f)
    sigma, spread = null_scale_about(f, mu)
    mu_shift_db = 10.0 * np.log10(mu / mu0)
    # a null is "available" when the calibrated centre still sits at the
    # analytic constant: a channel whose densest population is a carrier
    # lobe has no null in this era at all.
    null_available = bool(abs(mu_shift_db) <= NULL_TOLERANCE_DB)

    occupancy, leakage = {}, {}
    for eta in eta_ladder:
        key = "%g" % eta
        thr = eta * mu
        occupancy[key] = float(np.mean(f > thr))
        kept = f[f <= thr]
        leakage[key] = (float(10.0 * np.log10(
            np.percentile(kept, KEPT_TAIL_PERCENTILE) / mu))
                        if kept.size else float("nan"))

    floor_eta = ((mu + DETECTION_FLOOR_Z * sigma) / mu
                 if np.isfinite(sigma) else float("nan"))

    return Calibration(
        ch=channel.ch, fid=channel.fid, era_label=era_label, n_frames=n,
        n_units=int(n_units), mu0_provisional=mu0, mu=mu, mu_over_mu0=mu / mu0,
        mu_shift_db=mu_shift_db, sigma=sigma,
        sigma_over_mu=sigma / mu if np.isfinite(sigma) else float("nan"),
        sigma_spread=spread, null_available=null_available,
        occupancy=occupancy,
        occupancy_provisional=float(np.mean(f > mu0)),
        detection_floor_eta=floor_eta,
        detection_floor_db=(10.0 * np.log10(floor_eta)
                            if np.isfinite(floor_eta) else float("nan")),
        leakage_p99_db=leakage)
