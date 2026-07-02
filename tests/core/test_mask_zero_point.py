# coding=utf-8
"""Weight-norm zero-point regression tests for the positive-excess mask.

int4 quantization of the steering vectors leaves the three weight-term norms
unequal, so under a flat noise floor E[F] = mu0 = 2*target_norm_sq/
ref_norm_sum_sq differs from 1 per channel (~0.985..1.011 across the shipped
ATSC 14-36 bank). The legacy ``F > 1`` mask therefore pinned the H0 mask
fraction toward 0 or 1 per channel. These tests pin the corrected behaviour:

* ``weight_term_norms_sq`` matches a brute-force unpack exactly;
* the corrected rule reduces to the legacy rule when norms are 1:2, and is
  exact at the integer cross-multiplication boundary;
* Monte Carlo through the shipped ROM's real int4 weights shows E[F] tracks
  mu0 (not 1) and the corrected mask restores an ~50% H0 mask fraction on the
  two most-biased channels;
* exported runtime bundles declare per-channel half-threshold rationals
  ``nt : (nl+nu)`` that the bundle validator cross-checks against the weights.
"""
from __future__ import annotations

import numpy as np
import pytest

from pilot_proxy.detector_contract import (
    LEGACY_POSITIVE_EXCESS_MASK_RULE,
    POSITIVE_EXCESS_MASK_RULE,
    norm_corrected_mu0,
    norm_corrected_positive_excess,
    weight_term_norms_sq,
)
from pilot_proxy.detector_reference import (
    quantize_complex_numpy,
    unpack_packed_complex,
)
from pilot_proxy.detector_weights import DetectorWeightBank
from pilot_proxy.paths import DEFAULT_WEIGHTS_PATH

# 4096 rows x 200 trials keeps the Monte Carlo a few seconds while giving
# sigma(mean F) small enough to separate mu0 from 1 by >5 sigma on the two
# most-biased shipped channels (|mu0 - 1| ~ 0.011..0.015 vs SEM ~ 0.0014).
_MC_ROWS = 4096
_MC_TRIALS = 200
_MC_SEED = 20260701
_MOST_BIASED_HIGH_CHANNEL = 18  # largest shipped mu0
_MOST_BIASED_LOW_CHANNEL = 20   # smallest shipped mu0


def test_weight_term_norms_sq_matches_bruteforce() -> None:
    rng = np.random.default_rng(1234)
    for bits in (4, 8):
        packed_dtype = np.int8 if bits == 4 else np.int16
        limit = np.iinfo(packed_dtype)
        packed = rng.integers(
            limit.min, limit.max + 1, size=(3, 128), dtype=packed_dtype
        )
        got = weight_term_norms_sq(packed, bits_per_component=bits)
        w = unpack_packed_complex(packed, bits)
        expected = (np.abs(w) ** 2).sum(axis=1)
        assert got == tuple(int(round(v)) for v in expected)
        assert all(isinstance(v, int) for v in got)


def test_norm_corrected_rule_reduces_to_legacy_for_1_2_norms() -> None:
    # target_norm_sq : ref_norm_sum_sq = 1 : 2 is exactly the legacy rule.
    for p_target in range(0, 60):
        for p_ref_sum in (0, 1, 19, 20, 21, 40):
            legacy = int(p_ref_sum != 0 and 2 * p_target > p_ref_sum)
            corrected = norm_corrected_positive_excess(
                p_target, p_ref_sum, target_norm_sq=100, ref_norm_sum_sq=200
            )
            assert corrected == legacy


def test_norm_corrected_rule_exact_boundary() -> None:
    # nt=5, nrs=9 -> mask iff p_target*9 > 5*p_ref_sum, strictly.
    assert norm_corrected_positive_excess(11, 20, target_norm_sq=5, ref_norm_sum_sq=9) == 0
    assert norm_corrected_positive_excess(12, 20, target_norm_sq=5, ref_norm_sum_sq=9) == 1
    # equality is not an excess
    assert norm_corrected_positive_excess(10, 18, target_norm_sq=5, ref_norm_sum_sq=9) == 0
    # invalid reference floor
    assert norm_corrected_positive_excess(10, 0, target_norm_sq=5, ref_norm_sum_sq=9) == 0
    # exactness beyond float precision: (2**60 + 1) * 2 > 2 * 2**60 must mask
    big = 2**60
    assert norm_corrected_positive_excess(
        big + 1, 2 * big, target_norm_sq=1, ref_norm_sum_sq=2
    ) == 1
    assert norm_corrected_positive_excess(
        big, 2 * big, target_norm_sq=1, ref_norm_sum_sq=2
    ) == 0


def test_mask_rule_strings_are_distinct_and_stable() -> None:
    assert POSITIVE_EXCESS_MASK_RULE != LEGACY_POSITIVE_EXCESS_MASK_RULE
    assert "target_norm_sq" in POSITIVE_EXCESS_MASK_RULE
    assert LEGACY_POSITIVE_EXCESS_MASK_RULE == "valid && (p_target > (p_ref_sum >> 1))"


def _mc_h0_frames(weights_packed: np.ndarray, *, rows: int, trials: int,
                  seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """White noise through the real int4 path: per-frame (F, p_target, p_ref)."""
    rng = np.random.default_rng(seed)
    conj_w = np.conj(unpack_packed_complex(weights_packed, 4)).T.astype(np.complex64)
    fstats = np.empty(trials, dtype=np.float64)
    p_targets = np.empty(trials, dtype=np.int64)
    p_refs = np.empty(trials, dtype=np.int64)
    k = weights_packed.shape[1]
    for trial in range(trials):
        x = (
            rng.standard_normal((rows, k)) + 1j * rng.standard_normal((rows, k))
        ).astype(np.complex64)
        xq = unpack_packed_complex(
            quantize_complex_numpy(x, 4, 2.0), 4
        ).astype(np.complex64)
        powers = (np.abs(xq @ conj_w) ** 2).sum(axis=0)
        p_targets[trial] = int(round(float(powers[0])))
        p_refs[trial] = int(round(float(powers[1] + powers[2])))
        fstats[trial] = 2.0 * powers[0] / (powers[1] + powers[2])
    return fstats, p_targets, p_refs


@pytest.mark.parametrize(
    "channel", [_MOST_BIASED_HIGH_CHANNEL, _MOST_BIASED_LOW_CHANNEL]
)
def test_shipped_rom_h0_zero_point_and_corrected_mask(channel: int) -> None:
    bank = DetectorWeightBank(explicit_path=DEFAULT_WEIGHTS_PATH)
    weights, valid = bank.get_weights_for_physical_channel(channel)
    assert valid and weights is not None
    nt, nl, nu = weight_term_norms_sq(weights)
    nrs = int(nl + nu)
    mu0 = norm_corrected_mu0(nt, nrs)
    # The shipped bank's quantized norms are unequal on these channels; that
    # inequality is the entire point of the correction, so guard it.
    assert abs(mu0 - 1.0) > 5e-3, (
        f"channel {channel}: shipped mu0={mu0!r} is ~1; pick a channel with a "
        "larger norm imbalance for this regression test"
    )

    fstats, p_targets, p_refs = _mc_h0_frames(
        np.asarray(weights, dtype=np.int8),
        rows=_MC_ROWS,
        trials=_MC_TRIALS,
        seed=_MC_SEED + channel,
    )
    sem = fstats.std(ddof=1) / np.sqrt(len(fstats))
    # E[F] under H0 is mu0, not 1: the bias is detected at >5 sigma and the
    # measured mean agrees with mu0 within 6 sigma.
    assert abs(fstats.mean() - mu0) < 6.0 * sem
    assert abs(fstats.mean() - 1.0) > 5.0 * sem

    corrected_mask = np.asarray(
        [
            norm_corrected_positive_excess(
                int(p_t), int(p_r), target_norm_sq=nt, ref_norm_sum_sq=nrs
            )
            for p_t, p_r in zip(p_targets, p_refs)
        ]
    )
    fraction = float(corrected_mask.mean())
    # P(F > mu0 | H0) ~ 0.5; 200 trials -> binomial sigma 0.035, band = +-4 sigma.
    assert 0.36 <= fraction <= 0.64, (
        f"channel {channel}: corrected H0 mask fraction {fraction:.3f} is not ~0.5"
    )


def test_runtime_bundle_declares_norm_thresholds(tmp_path) -> None:
    import json

    from pilot_proxy.integration.defaults import (
        DEFAULT_CHIME_DTV_RECEIVER_PROFILE,
        DEFAULT_DETECTOR_CORE_PROFILE,
    )
    from pilot_proxy.runtime_bundle import (
        export_runtime_weight_bundle,
        validate_runtime_weight_bundle,
    )

    bundle_dir = tmp_path / "bundle"
    export_runtime_weight_bundle(
        receiver_profile_path=DEFAULT_CHIME_DTV_RECEIVER_PROFILE,
        detector_core_profile_path=DEFAULT_DETECTOR_CORE_PROFILE,
        physical_channels=[_MOST_BIASED_HIGH_CHANNEL],
        weight_coordinate_system="post_spectral_sense_normalization",
        output_dir=bundle_dir,
    )
    pilots = json.loads((bundle_dir / "pilot_profiles.json").read_text())
    row = pilots["profiles"][0]
    assert int(row["positive_excess_half_threshold_num"]) == int(row["target_norm_sq"])
    assert int(row["positive_excess_half_threshold_den"]) == int(row["ref_norm_sum_sq"])
    assert row["ref_norm_sum_sq"] > 0
    assert row["mu0"] == pytest.approx(
        2.0 * row["target_norm_sq"] / row["ref_norm_sum_sq"]
    )
    report = validate_runtime_weight_bundle(bundle_dir=bundle_dir)
    assert report["valid"] is True, report["errors"]
