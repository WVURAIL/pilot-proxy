# coding=utf-8
from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from pilot_proxy.chime.hdf5_input import discover_chime_pilot_datasets, read_complex_window
from pilot_proxy.chime.segmented_input import available_frames, iter_frame_chunks


def _write_segment(path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["freq"] = 470.3125
        h5.attrs["freq_id"] = 844
        ds = h5.create_dataset("baseband", data=data)
        ds.attrs["axis"] = np.asarray(["time", "input"], dtype=object)


def test_multiple_segment_files_concatenate_in_sorted_file_order(tmp_path) -> None:
    first = np.asarray(
        [
            [10, 20],
            [11, 21],
            [12, 22],
            [13, 23],
        ],
        dtype=np.uint8,
    )
    second = np.asarray(
        [
            [30, 40],
            [31, 41],
            [32, 42],
            [33, 43],
        ],
        dtype=np.uint8,
    )
    _write_segment(tmp_path / "ch0844" / "002.h5", second)
    _write_segment(tmp_path / "ch0844" / "001.h5", first)

    dataset = discover_chime_pilot_datasets(tmp_path, dataset_path=None)[14]
    block = read_complex_window(dataset, start_sample=2, stop_sample=6)

    assert dataset.total_time_samples == 8
    assert block.shape == (2, 1, 4)
    np.testing.assert_array_equal(block[0, 0], [12, 13, 30, 31])
    np.testing.assert_array_equal(block[1, 0], [22, 23, 40, 41])


def test_absolute_time_is_not_required_for_frame_chunks(tmp_path) -> None:
    _write_segment(tmp_path / "ch0844" / "001.h5", np.zeros((10, 2), dtype=np.uint8))
    dataset = discover_chime_pilot_datasets(tmp_path, dataset_path=None)[14]

    assert available_frames(dataset, frame_size_samples=4) == 2
    chunks = list(
        iter_frame_chunks(
            dataset,
            frame_size_samples=4,
            frames_per_chunk=1,
        )
    )

    assert [chunk.start_sample for chunk in chunks] == [0, 4]
    assert [chunk.stop_sample for chunk in chunks] == [4, 8]
