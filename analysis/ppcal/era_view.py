# coding=utf-8
"""Select a measured sensitivity floor from the quietest era."""
from __future__ import annotations

import operator

import numpy as np

from pilot_proxy.archived_product_keys import ARCHIVED_DATA_SHELF_SNR_DB

QUIET_ERA_MAX_LEVEL_DB = 1.0
QUIET_FLOOR_PERCENTILE = 90.0
MIN_QUIET_FLOOR_FRAMES = 30


def quiet_era_floor_db(
        channel,
        eras,
        level_threshold_db=QUIET_ERA_MAX_LEVEL_DB,
        percentile=QUIET_FLOOR_PERCENTILE,
        minimum_frames=MIN_QUIET_FLOOR_FRAMES):
    """(floor dB, era label, n frames) from the quietest era with a level.

    A transmitter-off level can only be bounded by frames the transmitter was
    off for.  Returns ``(nan, None, 0)`` when the channel has no such era,
    which is the case for every always-on carrier and is where the caller has
    to fall back on the sigma-implied substitute.
    """
    minimum_frames = operator.index(minimum_frames)
    if minimum_frames <= 0:
        raise ValueError("minimum_frames must be positive")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    z = channel._z
    shelf = z[ARCHIVED_DATA_SHELF_SNR_DB][:, 0][channel.health_include]
    fm = channel.frame_month
    best = None
    for era in eras:
        if era.level_median_db > level_threshold_db:
            continue
        sel = (fm >= era.month_start) & (fm <= era.month_end)
        vals = shelf[sel]
        vals = vals[np.isfinite(vals)]
        if vals.size < minimum_frames:
            continue
        db = float(np.percentile(vals, percentile))
        if best is None or era.level_median_db < best[2]:
            best = (db, era.label, era.level_median_db, int(vals.size))
    if best is None:
        return float("nan"), None, 0
    return best[0], best[1], best[3]
