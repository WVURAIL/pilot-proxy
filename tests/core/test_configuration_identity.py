# coding=utf-8
from __future__ import annotations

import json
from pathlib import Path

from pilot_proxy.detector_contract import (
    ALL_ROWS_DETECTOR_POWER_RATIO_DEFINITION,
    DETECTOR_POWER_RATIO_DEFINITION,
    WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
    build_chime_detector_contract,
)
from pilot_proxy.detector_weights import DetectorWeightBank
from pilot_proxy.integration.defaults import DEFAULT_DETECTOR_CORE_PROFILE
from pilot_proxy.integration.detector_core import load_detector_core_profile
from pilot_proxy.integration.receiver_profile import load_receiver_profile
from pilot_proxy.integration.schemas import (
    DETECTOR_CORE_ID_PILOT_PROXY_CUDA_LOCAL_REFERENCE_POWER_RATIO,
)
from pilot_proxy.integration.stream_layout import load_stream_map

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
CORE_ID = "pilotproxy_cuda_local_reference_power_ratio"


def test_canonical_detector_core_identity_and_filename() -> None:
    old = CONFIGS / "detector_core" / (
        "pilotproxy_cuda_" + "fstat_v1.json"
    )
    new = (
        CONFIGS
        / "detector_core"
        / "pilotproxy_cuda_local_reference_power_ratio.json"
    )
    assert not old.exists()
    assert new.is_file()
    assert DEFAULT_DETECTOR_CORE_PROFILE.resolve() == new.resolve()
    core = load_detector_core_profile(new)
    assert core.detector_core_id == CORE_ID
    assert (
        DETECTOR_CORE_ID_PILOT_PROXY_CUDA_LOCAL_REFERENCE_POWER_RATIO
        == CORE_ID
    )
    contract = core.to_dict()["kernel_contract"]
    assert "statistic" not in contract
    assert "pilot_excess" not in contract
    assert contract["coarse_power_ratio_definition"] == (
        "R_coarse = 2 * P_target / (P_ref_lower + P_ref_upper)"
    )
    assert contract["raw_pilot_excess_definition"] == (
        "rho_raw = R_coarse - 1 (diagnostic only)"
    )


def test_public_detector_contract_uses_power_ratio_keys() -> None:
    contract = build_chime_detector_contract(
        detector_window_samples=128,
        skipped_guard_bins=1,
        reference_offset_bins=2,
        num_weight_terms=3,
        weight_coordinate_system=WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
    )
    assert "statistic" not in contract
    assert ("all_rows_" + "statistic") not in contract
    assert (
        contract["coarse_power_ratio_definition"]
        == DETECTOR_POWER_RATIO_DEFINITION
    )
    assert (
        contract["all_rows_coarse_power_ratio_definition"]
        == ALL_ROWS_DETECTOR_POWER_RATIO_DEFINITION
    )


def test_receiver_and_stream_map_ids_are_chronology_free() -> None:
    profile_ids = {
        "reference_800mhz_pfb.json": "reference_800mhz_pfb",
        "chime_dtv_fengine.json": "chime_dtv_fengine",
        "chord_dtv_fengine.json": "chord_dtv_fengine",
        "chord_pathfinder_dtv_fengine.json": "chord_pathfinder_dtv_fengine",
    }
    for filename, expected in profile_ids.items():
        profile = load_receiver_profile(CONFIGS / "receiver_profiles" / filename)
        assert profile.receiver_profile_id == expected
        assert profile.compatible_detector_core_id == CORE_ID

    stream_ids = {
        "chime_feed_pol_example.json": "chime_dtv_fengine",
        "chord_dish_pol_example.json": "chord_dtv_fengine",
        "chord_pathfinder_dish_pol_example.json": (
            "chord_pathfinder_dtv_fengine"
        ),
    }
    for filename, expected in stream_ids.items():
        stream_map = load_stream_map(CONFIGS / "stream_maps" / filename)
        assert stream_map.receiver_profile_id == expected


def test_active_weight_headers_and_manifests_use_canonical_ids() -> None:
    expected = {
        "chime_dtv_weights_k128.bin": "chime_dtv_fengine",
        "chord_dtv_weights_k64.bin": "chord_dtv_fengine",
    }
    for filename, profile_id in expected.items():
        path = ROOT / "weights" / filename
        bank = DetectorWeightBank(explicit_path=path)
        assert bank.header.profile_name == profile_id
        manifest = json.loads(
            path.with_suffix(path.suffix + ".manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert manifest["receiver_profile"]["receiver_profile_id"] == profile_id
        assert (
            manifest["receiver_profile"]["detector_adapter"]
            ["compatible_detector_core_id"]
            == CORE_ID
        )
        assert all(
            row["reference_selection_method"]
            == "adaptive_circular_reference_placement"
            for row in manifest["target_reference_layout"]
        )
