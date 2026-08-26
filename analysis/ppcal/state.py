# coding=utf-8
"""Assemble the report-only per-channel calibration state.

One pass builds eras, the calibrated ``mu``, the threshold ladder on that
``mu``, the science-priced threshold from the supplied table when available,
and the historical report disposition. This module does not export an
operational threshold. Both figure scripts and the table writer consume this
state so no two of them can disagree.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np

from . import eras as E
from .calib import calibrate
from .products import load_all

# A median frame carrying this much excess is carrier-dominated: as much
# pilot power as noise in the target bin.  The survey separates cleanly --
# every channel is either below 1.0 dB or above 5.3 dB -- so the cut sits in
# an empty gap rather than on top of the population it divides.
CARRIER_DOMINATED_LEVEL_DB = 3.0
EXCISE_MASKED_FRACTION = 0.50
LIGHT_MASKING_FRACTION = 0.10
FALLBACK_ETA = 1.4
MAX_THRESHOLD_BRACKET_RATIO = 1.10

# published complete-23 policy, for cross-checking only
PUBLISHED_EXCISED = {17, 22, 24, 30, 31, 35}
PUBLISHED_INCLUSIVE_KEEP = {14, 15, 36}
COLLECTION_CEASED = {30: "September 2023"}
LOCKED_EPOCH = {35: "sign-on Nov 2021", 19: "sign-off Dec 2024",
                26: "sign-off Apr 2023", 20: "step down Sep 2022",
                27: "sign-off in 2021-22 archive gap",
                32: "sign-off in 2021-22 archive gap"}


@dataclass
class ChannelState:
    c: object
    segs: list
    fmask: np.ndarray
    umask: np.ndarray
    cal: object
    blind: object
    thresholds: dict
    threshold_status: str
    decision_scope: str
    carrier_dominated: bool
    verdict: str
    disposition: str
    reason: str

    @property
    def ch(self):
        return self.c.ch

    @property
    def eta_channel(self):
        """The channel threshold, or the historical report fallback."""
        value = _threshold_eta(self.thresholds, "eta_cost_cap")
        return value if value is not None else FALLBACK_ETA

    @property
    def eta_thermal(self):
        """The same cost optimum evaluated at the optimistic bracket end."""
        value = _threshold_eta(self.thresholds, "eta_cost_thermal")
        return value if value is not None else float("nan")

    @property
    def eta_bracket_ratio(self):
        """How far eta moves between the two ends of the coherence bracket.

        This is the quantity that decides whether a per-channel threshold is
        identified at all. Where the correlation time was refused, eta is a
        function of a bound rather than a measurement, and the two ends of
        the bracket disagree -- by up to an order of magnitude on this
        archive. Where tau was measured the bracket collapses and the ratio
        is exactly 1, which is the clearest statement of what measuring tau
        actually buys.
        """
        import numpy as _np
        a, b = self.eta_channel, self.eta_thermal
        if not _np.isfinite(b) or a <= 0:
            return float("nan")
        return float(b / a)

    @property
    def eta_is_identified(self):
        """True when the bracket does not move the threshold materially."""
        import numpy as _np
        r = self.eta_bracket_ratio
        return bool(_np.isfinite(r) and r < MAX_THRESHOLD_BRACKET_RATIO)

    @property
    def eta_is_per_channel(self):
        return _threshold_eta(self.thresholds, "eta_cost_cap") is not None

    def occ_at(self, eta):
        """Fraction of latest-era frames above ``eta * mu``."""
        f = self.c.fstat[self.fmask]
        return float(np.mean(f > eta * self.cal.mu))

    @property
    def occ_working(self):
        """Masked fraction at this channel's own threshold."""
        return self.occ_at(self.eta_channel)

    @property
    def occ_global(self):
        """Masked fraction at the single global rule, for comparison."""
        return self.cal.occupancy["%g" % FALLBACK_ETA]

    @property
    def agrees_with_published(self):
        pub = "excise" if self.ch in PUBLISHED_EXCISED else "keep"
        return self.verdict == pub


def _read_thresholds(path):
    if not path or not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            num = {}
            for k, v in row.items():
                if v in ("", None):
                    num[k] = None
                elif k in ("tau_measured", "dilation_tol_published"):
                    num[k] = v == "True"
                elif k in ("ch",):
                    num[k] = int(v)
                elif k in ("era", "note", "residual_basis"):
                    num[k] = v
                else:
                    try:
                        num[k] = float(v)
                    except ValueError:
                        num[k] = v
            channel = int(row["ch"])
            for field in ("eta_cost_cap", "eta_cost_thermal"):
                value = num.get(field)
                if value is None:
                    continue
                if (not isinstance(value, float) or not np.isfinite(value)
                        or value <= 0.0):
                    raise ValueError(
                        "channel %d has invalid %s" % (channel, field))
            out[channel] = num
    return out


def _threshold_eta(thresholds, field):
    if not thresholds:
        return None
    value = thresholds.get(field)
    if isinstance(value, float) and np.isfinite(value) and value > 0.0:
        return float(value)
    return None


def decide(cal, thresholds, eta, occ):
    """Historical report label from the null and residual bound.

    The residual is carried on the bounded basis: where the coherence time was
    refused the chain is evaluated at the sidereal-day cap, which is the
    physical ceiling, so the reported residual is an upper limit rather than
    an estimate. A residual above tolerance therefore means the tolerance is
    *not certified*, not that it is exceeded -- a measured coherence time can
    only move the residual down. No report label below depends on it: excision
    is decided on carrier dominance and occupancy, both of which are
    coherence free.
    """
    carrier = cal.mu_shift_db > CARRIER_DOMINATED_LEVEL_DB
    bracket = ""
    if (thresholds and thresholds.get("r_cost_cap")
            and thresholds.get("r_tol_dilation")):
        mult = thresholds["r_cost_cap"] / thresholds["r_tol_dilation"]
        at_cap = not bool(thresholds.get("tau_measured"))
        bracket = ("; the residual is bounded at %.3g times its dilation "
                   "tolerance%s, so that tolerance is uncertified rather than "
                   "shown to be exceeded"
                   % (mult, " with the coherence time at the sidereal cap"
                      if at_cap else " on a measured coherence time"))
    if carrier:
        return ("excise", "excise",
                "the median frame carries %+.1f dB of excess -- the densest "
                "population in this era is the carrier itself, so no null "
                "exists to threshold against%s" % (cal.mu_shift_db, bracket))
    if occ >= EXCISE_MASKED_FRACTION:
        return ("excise", "excise",
                "%.1f%% of latest-era frames sit above F > %.3f mu%s"
                % (100 * occ, eta, bracket))
    tier = "light" if occ <= LIGHT_MASKING_FRACTION else "heavy"
    return ("keep", "keep, %s masking" % tier,
            "null calibrated to mu = %.4f (%+.3f dB off the provisional "
            "constant); %.1f%% of latest-era frames masked at F > %.3f mu%s"
            % (cal.mu, cal.mu_shift_db, 100 * occ, eta, bracket))


def build(products, threshold_csv=None, *, bao_csv=None):
    """Every channel's report state, ordered by physical channel.

    ``bao_csv`` is retained only as a keyword compatibility alias.
    """
    if threshold_csv is not None and bao_csv is not None:
        raise ValueError("use threshold_csv or the legacy bao_csv alias, not both")
    threshold_csv = threshold_csv if threshold_csv is not None else bao_csv
    threshold_all = _read_thresholds(threshold_csv)
    out = []
    for c in sorted(load_all(products), key=lambda c: c.ch):
        segs = E.segment(c)
        fmask = E.final_era_frame_mask(c, segs)
        umask = E.final_era_unit_mask(c, segs)
        cal = calibrate(c, fmask, segs[-1].label, int(umask.sum()))
        blind = calibrate(c, np.ones(c.fstat.size, bool), "full archive",
                          c.n_units_raw)
        thresholds = threshold_all.get(c.ch, {})
        selected = _threshold_eta(thresholds, "eta_cost_cap") is not None
        s = ChannelState(
            c=c, segs=segs, fmask=fmask, umask=umask, cal=cal, blind=blind,
            thresholds=thresholds,
            threshold_status=("supplied" if selected
                              else "historical_fallback"),
            decision_scope="report_only",
            carrier_dominated=(
                cal.mu_shift_db > CARRIER_DOMINATED_LEVEL_DB),
            verdict="", disposition="", reason="")
        eta = s.eta_channel
        s.verdict, s.disposition, s.reason = decide(cal, thresholds, eta,
                                                    s.occ_at(eta))
        if selected:
            note = "this disposition is descriptive, not an operational export"
        else:
            note = ("the historical eta = %.1f fallback was used, so this "
                    "disposition is report-only" % FALLBACK_ETA)
        s.reason = "%s; %s" % (s.reason, note)
        out.append(s)
    return out
