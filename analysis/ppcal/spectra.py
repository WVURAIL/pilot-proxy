# coding=utf-8
"""Spectral views: transmitter census, era-averaged spectra, spectrograms.

Two spectral products survive in the archive, and they answer different
questions:

* ``integrated_spectrum_before_mask`` -- 16384 bins of 23.84 Hz spanning the
  whole 390.625 kHz CHIME channel.  Wide enough to see every carrier in the
  channel and where each sits relative to the detector's guard references,
  but integrated over the *entire* archive: it cannot be split by era.
* the archived fine power ratio -- 256 bins of 11.92 Hz spanning +/-1.53 kHz
  about the target coarse bin, stored **per frame**.  Narrow, but the only
  time-resolved spectral product, so it is what an era-resolved spectrum or a
  spectrogram has to be built from.

Both are reported in the RF (transmitted) sense; the receiver frame is
inverted when ``sense == -1``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .products import COARSE_HZ, REFERENCE_OFFSET_HZ

DC_HALF_WIDTH_HZ = 60.0        # channel-centre artefact, ~2 spectrum bins
PILOT_SEARCH_HZ = 5.0e3        # repo convention for refining the pilot peak
PEAK_MIN_DB = 3.0
PEAK_MIN_SEPARATION_HZ = 3.0e3


@dataclass(frozen=True)
class Peak:
    """One spectral feature in the channel-wide integrated spectrum."""

    rf_offset_hz: float          # from the CHIME channel centre
    offset_from_pilot_hz: float  # from the synthesized pilot position
    db_rel_median: float
    kind: str                    # 'pilot' | 'channel-centre artefact' | 'secondary'


def target_coarse_offset_hz(channel):
    """RF offset of the detector's target coarse bin from the channel centre."""
    eff = channel.sense * channel.pilot_offset_hz
    return channel.sense * round(eff / COARSE_HZ) * COARSE_HZ


def guard_reference_offsets_hz(channel):
    """RF offsets of the two +/-2-bin guard references."""
    t = target_coarse_offset_hz(channel)
    return t - REFERENCE_OFFSET_HZ, t + REFERENCE_OFFSET_HZ


def spectrum_db(channel, which="before"):
    """(rf offset Hz, dB relative to the channel's median spectral power)."""
    rf, s = channel.spectrum if which == "before" else channel.spectrum_after
    pos = s[s > 0]
    floor = pos.min() if pos.size else 1.0
    return rf, 10.0 * np.log10(np.maximum(s, floor) / np.median(pos))


def peak_census(channel, min_db=PEAK_MIN_DB,
                min_separation_hz=PEAK_MIN_SEPARATION_HZ, max_peaks=8):
    """Isolated features in the channel-wide spectrum, classified.

    The bin at the channel centre is an instrumental artefact of the CHIME
    channelisation -- it is present at 18-24 dB before the health repair in
    every one of the 23 channels, including channels with no transmitter
    anywhere near it -- so it is labelled rather than counted as a carrier.
    The detector's own contract names the same bin a forbidden tone.
    """
    rf, db = spectrum_db(channel)
    poff = channel.pilot_offset_hz
    work = db.copy()
    picked = []
    order = np.argsort(work)[::-1]
    for i in order:
        if work[i] < min_db:
            break
        if any(abs(rf[i] - rf[j]) < min_separation_hz for j in picked):
            continue
        picked.append(i)
        if len(picked) >= max_peaks:
            break

    out = []
    for i in sorted(picked, key=lambda j: rf[j]):
        d = float(rf[i] - poff)
        if abs(float(rf[i])) <= DC_HALF_WIDTH_HZ:
            kind = "channel-centre artefact"
        elif abs(d) <= PILOT_SEARCH_HZ:
            kind = "pilot"
        else:
            kind = "secondary"
        out.append(Peak(rf_offset_hz=float(rf[i]), offset_from_pilot_hz=d,
                        db_rel_median=float(db[i]), kind=kind))
    return out


def measured_pilot_offset_hz(channel):
    """Refined carrier position from the wide spectrum, RF sense.

    Returns ``(offset_from_pilot_hz, db_rel_median)`` for the strongest bin
    within +/-``PILOT_SEARCH_HZ`` of the synthesized pilot position.
    """
    rf, db = spectrum_db(channel)
    sel = np.abs(rf - channel.pilot_offset_hz) <= PILOT_SEARCH_HZ
    if not np.any(sel):
        return float("nan"), float("nan")
    idx = np.flatnonzero(sel)
    j = idx[int(np.argmax(db[idx]))]
    return float(rf[j] - channel.pilot_offset_hz), float(db[j])


# ---------------------------------------------------------------------------
# fine-band, time resolved
# ---------------------------------------------------------------------------

def fine_axis_hz(channel):
    """RF offset (Hz) of each fine bin, ascending, with the sort order."""
    rf = channel.fine_rf_hz
    o = np.argsort(rf)
    return rf[o], o


def era_fine_spectrum(channel, frame_mask):
    """(rf Hz, median dB, p90 dB) of the fine statistic over selected frames.

    Referenced to ``mu0``, the analytic null of the same ratio statistic, so
    0 dB is "no excess in this bin".  The median trace shows persistent
    carriers; the p90 trace exposes carriers that are only on part of the
    time.
    """
    rf, o = fine_axis_hz(channel)
    f = channel.fine[frame_mask][:, o]
    if f.size == 0:
        nan = np.full(rf.size, np.nan)
        return rf, nan, nan
    med = np.median(f, axis=0)
    p90 = np.percentile(f, 90, axis=0)

    def to_db(a):
        return 10.0 * np.log10(np.maximum(a, 1e-6) / channel.mu0)

    return rf, to_db(med), to_db(p90)


def era_fine_spectrum_masked(channel, frame_mask, threshold):
    """Latest-era fine spectrum before and after the threshold is applied.

    ``threshold`` is the absolute cut on the coarse statistic (eta * mu), so a
    frame survives when F <= threshold. Both traces are **means** over their
    frames rather than medians: mean power is what integrates into a map, so
    it is the statistic that says what masking actually removes. The median
    is the right choice for the spectrogram, where the question is persistence
    rather than integrated power, and the two part company on a bursty
    channel.

    Returns ``(rf_hz, before_db, after_db, stats)`` in dB relative to ``mu0``,
    with ``stats`` carrying the kept fraction, the suppression at the peak bin
    and the band-integrated suppression.
    """
    rf, o = fine_axis_hz(channel)
    f = channel.fstat[frame_mask]
    block = channel.fine[frame_mask][:, o]
    keep = f <= threshold

    def to_db(a):
        return 10.0 * np.log10(np.maximum(a, 1e-12) / channel.mu0)

    before = block.mean(axis=0)
    stats = {"n_frames": int(block.shape[0]), "n_kept": int(keep.sum()),
             "kept_fraction": float(keep.mean()) if keep.size else float("nan")}
    if not keep.any():
        nan = np.full(rf.size, np.nan)
        stats.update(peak_offset_hz=float("nan"),
                     peak_suppression_db=float("nan"),
                     band_suppression_db=float("nan"))
        return rf, to_db(before), nan, stats

    after = block[keep].mean(axis=0)
    j = int(np.argmax(before))
    stats.update(
        peak_offset_hz=float(rf[j]),
        peak_suppression_db=float(
            10.0 * np.log10(before[j] / max(after[j], 1e-12))),
        band_suppression_db=float(
            10.0 * np.log10(before.mean() / max(after.mean(), 1e-12))))
    return rf, to_db(before), to_db(after), stats


def wide_pair_is_era_resolved(eras):
    """Whether the stored wide before/after pair describes the latest era.

    The channel-wide integrated spectra are accumulated over the whole archive
    and are not retained per frame, so they can be read as a latest-era
    quantity only on a channel that has exactly one era. Where a transition
    exists they blend populations the transition made physically distinct, and
    no post-hoc split is possible from what the products keep. Splitting them
    needs a per-era accumulator at trawl time -- cheap to add, impossible to
    recover afterwards.
    """
    return len(eras) == 1


def fine_spectrogram(channel, months_grid=None, statistic="median"):
    """Monthly fine-band spectrogram.

    Returns ``(month_indices, rf_hz, image)`` where ``image`` has shape
    (n_fine_bins, n_months) in dB relative to ``mu0``, with ``nan`` in months
    that hold no healthy frame.  The monthly median is the default because it
    is the same robust statistic the era segmentation runs on, so era
    boundaries drawn on the spectrogram mean exactly what they mean in the
    occupancy history.
    """
    rf, o = fine_axis_hz(channel)
    fm = channel.frame_month
    if months_grid is None:
        months_grid = np.arange(int(fm.min()), int(fm.max()) + 1)
    img = np.full((rf.size, months_grid.size), np.nan)
    fine = channel.fine
    for j, m in enumerate(months_grid):
        sel = fm == m
        if not np.any(sel):
            continue
        block = fine[sel][:, o]
        if statistic == "median":
            v = np.median(block, axis=0)
        elif statistic == "p90":
            v = np.percentile(block, 90, axis=0)
        else:
            v = block.mean(axis=0)
        img[:, j] = 10.0 * np.log10(np.maximum(v, 1e-6) / channel.mu0)
    return months_grid, rf, img


def fine_line_census(channel, frame_mask, min_db=1.5, min_prominence_db=1.5,
                     min_separation_hz=48.0, max_lines=6):
    """Resolved lines in the era-averaged fine spectrum.

    A line is a local maximum that stands ``min_prominence_db`` clear of the
    saddle joining it to any taller neighbour, so the shoulders of one
    broadened carrier are not counted twice.  The 2x zero padding correlates
    adjacent bins, so the minimum separation is held at four padded bins
    (two independent samples).

    A resolved line is **not** by itself evidence of a separate emitter.
    Co-channel ATSC transmitters are offset from one another by tens to
    hundreds of Hz, which this grid resolves; so are the sidebands of a
    single modulated carrier.  The discriminator is symmetry -- sideband
    pairs sit at matched offsets either side of the dominant line, and
    ch22's six lines are exactly that (roughly +/-96 and +/-200 Hz about
    one carrier).  Treat the count as lines resolved, and read the
    geometry before calling any of them a transmitter.
    """
    from scipy.signal import find_peaks

    rf, med, p90 = era_fine_spectrum(channel, frame_mask)
    if not np.isfinite(med).any():
        return []
    base = float(np.median(med))
    excess = med - base
    distance = max(1, int(round(min_separation_hz / abs(rf[1] - rf[0]))))
    idx, props = find_peaks(excess, height=min_db,
                            prominence=min_prominence_db, distance=distance)
    if idx.size == 0:
        return []
    order = np.argsort(props["prominences"])[::-1][:max_lines]
    keep = np.sort(idx[order])
    return [(float(rf[i]), float(excess[i]), float(p90[i] - base))
            for i in keep]
