# coding=utf-8
"""Temporary single-era views of a survey product.

``rfisher.residual.threshold_sweep`` characterises a transmitter-*on* epoch
and takes its sensitivity floor from the *off* epoch.  That is the right
shape for a sign-on channel, whose latest era is the loud one, and the wrong
shape for a sign-off channel, whose latest era is the quiet one: naming the
quiet era as ``off`` would then sweep the era that no longer exists, and
naming it as ``on`` would draw the floor from the loud era and inflate every
residual by the contamination the sign-off removed.

So the era is applied to the frames instead of to the epoch arguments.  This
module writes a product view holding only the frames of one era, leaving the
unit axis whole so ``frame_unit_index`` stays valid, and the caller passes
the floor explicitly -- measured on the quietest era the channel has, which
is the only population that can bound a transmitter-off level.
"""
from __future__ import annotations

import contextlib
import operator
import os
import tempfile

import numpy as np

from pilot_proxy.archived_product_keys import ARCHIVED_DATA_SHELF_SNR_DB

QUIET_ERA_MAX_LEVEL_DB = 1.0
QUIET_FLOOR_PERCENTILE = 90.0
MIN_QUIET_FLOOR_FRAMES = 30


@contextlib.contextmanager
def era_product_view(channel, frame_mask, tmpdir=None):
    """Write an npz holding only ``frame_mask`` frames; yield its path.

    Frame-indexed arrays are subset and ``frame_index`` renumbered; unit-axis
    arrays and scalars are copied unchanged.
    """
    z = channel._z
    n_frames = channel.n_frames_raw
    sel = np.zeros(n_frames, dtype=bool)
    sel[np.flatnonzero(channel.health_include)[frame_mask]] = True

    out = {}
    for key in z.files:
        a = z[key]
        if a.ndim >= 1 and a.shape[0] == n_frames:
            out[key] = a[sel]
        else:
            out[key] = a
    out["frame_index"] = np.arange(int(sel.sum()), dtype=np.int64)

    fd, path = tempfile.mkstemp(suffix=".npz", prefix="era_ch%02d_" % channel.ch,
                                dir=tmpdir)
    os.close(fd)
    try:
        np.savez(path, **out)
        yield path
    finally:
        with contextlib.suppress(OSError):
            os.remove(path)


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
