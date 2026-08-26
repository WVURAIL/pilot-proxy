# coding=utf-8
"""Activity-era segmentation for one channel's occupancy history.

A channel's pilot occupancy is piecewise stationary: a transmitter signs on,
signs off, or steps power, and the level between such events is flat apart
from propagation scatter.  This module recovers those change points from the
data rather than from a hand-maintained table, so that "characterise the
latest era" has an auditable definition.

The latest era is the one carried forward because it is the *forecast*: what
the band will do on the telescope next depends on the configuration the
transmitters are in now, not on their history.  A channel that signed on is
therefore allowed to look worse under era resolution, not better.

Method (pre-registered here, applied identically to all channels):

1. Reduce to one point per observed calendar month: the median of the
   per-unit level 10*log10(mean F / mu0).  Monthly medians are robust to the
   heavy transient tail that dominates the frame-level distribution, and the
   month grid keeps a long quiet stretch from being outvoted by a densely
   sampled one.
2. Recursive binary segmentation.  At every admissible split the two sides
   are compared with a Mann-Whitney U z-score (rank based, so no Gaussian
   assumption) and by the difference of their medians.
3. A split is accepted only if all of:
   * both sides contain at least ``min_months`` observed months,
   * both sides span at least ``min_days`` of wall-clock time,
   * the median step is at least ``min_step_db``,
   * |z| >= ``z_crit``.
   The strongest admissible split is taken first, then each side recurses,
   to at most ``max_eras`` segments.

The gate is deliberately conservative: it is meant to find transmitter
transitions, not to track seasonal propagation.  Channels with no accepted
split are reported as single-era, which is the expected outcome for most.
"""
from __future__ import annotations

from dataclasses import dataclass
import operator

import numpy as np

from .products import month_label

MIN_MONTHS = 6
MIN_DAYS = 270.0
MIN_STEP_DB = 2.0
Z_CRIT = 4.0
MAX_ERAS = 5
SECONDS_PER_DAY = 86400.0
ERA_LEVEL_UPPER_PERCENTILE = 90.0


@dataclass(frozen=True)
class Era:
    """One activity era, closed on both ends in the observed month grid."""

    index: int
    month_start: int
    month_end: int
    n_units: int
    level_median_db: float
    level_p90_db: float

    @property
    def label(self):
        return "%s..%s" % (month_label(self.month_start),
                           month_label(self.month_end))


def _mann_whitney_z(a, b):
    """Normal-approximation z for the rank-sum of ``a`` against ``b``."""
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return 0.0
    both = np.concatenate([a, b])
    order = np.argsort(both, kind="mergesort")
    ranks = np.empty(both.size, dtype=float)
    ranks[order] = np.arange(1, both.size + 1, dtype=float)
    # average ranks within ties
    s = both[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    u = ranks[:na].sum() - na * (na + 1) / 2.0
    mu = na * nb / 2.0
    # tie-corrected variance
    _, counts = np.unique(both, return_counts=True)
    n = both.size
    tie = float(np.sum(counts ** 3 - counts))
    var = na * nb / 12.0 * ((n + 1) - tie / (n * (n - 1))) if n > 1 else 0.0
    if var <= 0:
        return 0.0
    return float((u - mu) / np.sqrt(var))


def _best_split(months, med, first_time, last_time, lo, hi, cfg):
    """Strongest admissible split index in [lo, hi), or None."""
    best = None
    for k in range(lo + cfg["min_months"], hi - cfg["min_months"] + 1):
        left, right = med[lo:k], med[k:hi]
        if left.size < cfg["min_months"] or right.size < cfg["min_months"]:
            continue
        span_l = (last_time[k - 1] - first_time[lo]) / SECONDS_PER_DAY
        span_r = (last_time[hi - 1] - first_time[k]) / SECONDS_PER_DAY
        if span_l < cfg["min_days"] or span_r < cfg["min_days"]:
            continue
        step = abs(float(np.median(right)) - float(np.median(left)))
        if step < cfg["min_step_db"]:
            continue
        z = abs(_mann_whitney_z(left, right))
        if z < cfg["z_crit"]:
            continue
        score = z * step
        if best is None or score > best[0]:
            best = (score, k, step, z)
    return best


def _segment(months, med, first_time, last_time, lo, hi, cfg, out):
    if len(out) >= cfg["max_eras"] - 1:
        return
    best = _best_split(months, med, first_time, last_time, lo, hi, cfg)
    if best is None:
        return
    _, k, _, _ = best
    out.append(k)
    _segment(months, med, first_time, last_time, lo, k, cfg, out)
    _segment(months, med, first_time, last_time, k, hi, cfg, out)


def segment(channel, min_months=MIN_MONTHS, min_days=MIN_DAYS,
            min_step_db=MIN_STEP_DB, z_crit=Z_CRIT, max_eras=MAX_ERAS):
    """Segment one :class:`~ppcal.products.Channel` into activity eras."""
    min_months = operator.index(min_months)
    max_eras = operator.index(max_eras)
    if min_months < 1:
        raise ValueError("min_months must be positive")
    if max_eras < 1:
        raise ValueError("max_eras must be positive")
    for name, value in (("min_days", min_days),
                        ("min_step_db", min_step_db),
                        ("z_crit", z_crit)):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("%s must be finite and non-negative" % name)
    months, med, _ = channel.monthly_level_db()
    if not len(months):
        raise ValueError("channel has no retained acquisitions")
    unit_time = channel.units[0]
    unit_month = channel.unit_month
    first_time = np.array([unit_time[unit_month == month].min()
                           for month in months])
    last_time = np.array([unit_time[unit_month == month].max()
                          for month in months])
    cfg = dict(min_months=min_months, min_days=min_days,
               min_step_db=min_step_db, z_crit=z_crit, max_eras=max_eras)
    cuts = []
    _segment(months, med, first_time, last_time, 0, len(months), cfg, cuts)
    bounds = [0] + sorted(cuts) + [len(months)]

    lvl = channel.unit_level_db
    umonth = channel.unit_month
    eras = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        m0, m1 = int(months[a]), int(months[b - 1])
        sel = (umonth >= m0) & (umonth <= m1)
        eras.append(Era(index=i, month_start=m0, month_end=m1,
                        n_units=int(sel.sum()),
                        level_median_db=float(np.median(lvl[sel])),
                        level_p90_db=float(np.percentile(
                            lvl[sel], ERA_LEVEL_UPPER_PERCENTILE))))
    return eras


def final_era_frame_mask(channel, eras):
    """Boolean mask over healthy frames selecting the latest era."""
    last = eras[-1]
    fm = channel.frame_month
    return (fm >= last.month_start) & (fm <= last.month_end)


def final_era_unit_mask(channel, eras):
    last = eras[-1]
    um = channel.unit_month
    return (um >= last.month_start) & (um <= last.month_end)
