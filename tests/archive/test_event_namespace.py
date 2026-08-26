# coding=utf-8
from __future__ import annotations

import numpy as np
import pytest

from pilot_proxy.archive import combine as combine_module
from pilot_proxy.archive.chime_coarse import source_event_key
from pilot_proxy.archive.combine import _align_frames, combine_detector_products


def _product(channel: int, freq_id: int, event: str, *, time0: float) -> dict:
    return {
        "physical_channel": np.asarray([channel], dtype=np.int32),
        "freq_id": np.asarray([freq_id], dtype=np.int64),
        "frame_index": np.asarray([0], dtype=np.int64),
        "source_event_keys": np.asarray([event]),
        "frame_unit_index": np.asarray([0], dtype=np.int32),
        "frame_in_unit": np.asarray([0], dtype=np.int32),
        "unit_event_id": np.asarray([123], dtype=np.int64),
        "unit_time0_ctime": np.asarray([time0], dtype=np.float64),
        "unit_time0_fpga": np.asarray([456], dtype=np.uint64),
        "unit_delta_time": np.asarray([1.0 / 390_625.0], dtype=np.float64),
        "coarse_power_ratio": np.asarray([[1.0]], dtype=np.float64),
    }


def test_source_event_key_preserves_campaign_namespace() -> None:
    assert source_event_key("/campaign-a/baseband_100_844.h5", 844) == (
        "/campaign-a/baseband_100.h5"
    )
    assert source_event_key("/campaign-b/baseband_100_829.h5", 829) == (
        "/campaign-b/baseband_100.h5"
    )
    assert source_event_key("cadc:ARCHIVE/baseband_100_844.h5", 844) == (
        "cadc:ARCHIVE/baseband_100.h5"
    )


def test_same_basename_in_different_namespaces_does_not_align() -> None:
    first = _product(14, 844, "/campaign-a/baseband_100.h5", time0=10.0)
    second = _product(15, 829, "/campaign-b/baseband_100.h5", time0=10.0)

    with pytest.raises(ValueError, match="share no common"):
        _align_frames([first, second])


def test_path_matched_event_with_different_start_time_is_rejected() -> None:
    event = "/campaign-a/baseband_100.h5"
    first = _product(14, 844, event, time0=10.0)
    second = _product(15, 829, event, time0=20.0)

    with pytest.raises(ValueError, match="disagrees on unit_time0_ctime"):
        _align_frames([first, second])


@pytest.mark.parametrize(
    "field",
    [
        "unit_scope",
        "archive_version",
        "unit_git_version_tag",
        "unit_input_map_sha256",
    ],
)
def test_path_matched_event_with_different_receiver_state_is_rejected(field) -> None:
    event = "/campaign-a/baseband_100.h5"
    first = _product(14, 844, event, time0=10.0)
    second = _product(15, 829, event, time0=10.0)
    first[field] = np.asarray(["first"], dtype=str)
    second[field] = np.asarray(["second"], dtype=str)

    with pytest.raises(ValueError, match=rf"disagrees on {field}"):
        _align_frames([first, second])


def test_nearly_one_sample_start_time_shift_is_rejected() -> None:
    event = "/campaign-a/baseband_100.h5"
    sample_period = 1.0 / 390_625.0
    first = _product(14, 844, event, time0=10.0)
    second = _product(
        15,
        829,
        event,
        time0=10.0 + 0.9 * sample_period,
    )

    with pytest.raises(ValueError, match="disagrees on unit_time0_ctime"):
        _align_frames([first, second])


def test_start_time_shift_below_half_sample_is_accepted() -> None:
    event = "/campaign-a/baseband_100.h5"
    sample_period = 1.0 / 390_625.0
    first = _product(14, 844, event, time0=10.0)
    second = _product(
        15,
        829,
        event,
        time0=10.0 + 0.49 * sample_period,
    )

    aligned, _, _ = _align_frames([first, second])

    assert len(aligned) == 2


def test_identical_start_times_are_accepted_below_timestamp_resolution() -> None:
    event = "/campaign-a/baseband_100.h5"
    first = _product(14, 844, event, time0=1_700_000_000.0)
    second = _product(15, 829, event, time0=1_700_000_000.0)
    first["unit_delta_time"] = np.asarray([1.0e-9], dtype=np.float64)
    second["unit_delta_time"] = np.asarray([1.0e-9], dtype=np.float64)

    aligned, _, _ = _align_frames([first, second])

    assert len(aligned) == 2


def test_exact_half_sample_start_time_shift_is_rejected() -> None:
    event = "/campaign-a/baseband_100.h5"
    sample_period = 1.0 / 390_625.0
    first = _product(14, 844, event, time0=10.0)
    second = _product(
        15,
        829,
        event,
        time0=10.0 + 0.5 * sample_period,
    )

    with pytest.raises(ValueError, match="disagrees on unit_time0_ctime"):
        _align_frames([first, second])


def test_failed_combined_build_does_not_publish_partial_set(
    tmp_path, monkeypatch
) -> None:
    destination = tmp_path / "run"
    destination.mkdir()
    canonical = destination / "chime_detector_outputs.npz"
    canonical.write_bytes(b"previous-complete-set")

    def fail_build(product_paths, run_dir, **kwargs):
        del product_paths, kwargs
        (run_dir / "chime_detector_outputs.npz").write_bytes(b"partial-new-set")
        raise RuntimeError("simulated terminal combine failure")

    monkeypatch.setattr(combine_module, "_combine_detector_products", fail_build)

    with pytest.raises(RuntimeError, match="terminal combine failure"):
        combine_detector_products([], destination)

    assert canonical.read_bytes() == b"previous-complete-set"
    assert not list(tmp_path.glob(".run.combine.*"))
