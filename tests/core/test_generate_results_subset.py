import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[2] / "scripts" / "generate_results.py"
SPEC = importlib.util.spec_from_file_location("generate_results", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
generate_results = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generate_results)


def test_exact_subset_search_finds_unobserved_intersection():
    event_sets = {
        10: {"a", "b"},
        20: {"a", "b"},
        30: {"a"},
        40: {"b"},
    }

    result = generate_results.exact_subset_search(event_sets, min_channels=2)

    assert result["selected"] == {
        "k": 2,
        "common_events": 2,
        "channels": [10, 20],
        "excluded": [30, 40],
    }
    assert [row["common_events"] for row in result["by_k"]] == [0, 1, 2]


def test_exact_subset_search_matches_historical_exhaustive_result():
    path = (
        Path(__file__).parents[2]
        / "data"
        / "provenance"
        / "combine_subset_20260714"
        / "event_presence_signatures.npz"
    )
    with np.load(path, allow_pickle=False) as archive:
        fids = archive["freq_ids"].astype(int).tolist()
        signatures = archive["signature"].astype(np.int64)
        counts = archive["count"].astype(np.int64)

    event_sets = {fid: set() for fid in fids}
    for signature, count in zip(signatures.tolist(), counts.tolist()):
        for occurrence in range(count):
            event = (signature, occurrence)
            for index, fid in enumerate(fids):
                if (signature >> index) & 1:
                    event_sets[fid].add(event)

    result = generate_results.exact_subset_search(event_sets, min_channels=16)

    assert [row["common_events"] for row in result["by_k"]] == [
        0, 12, 13, 824, 1547, 1587, 1789, 1829
    ]
    assert result["selected"]["channels"] == [
        506, 537, 552, 614, 629, 644, 675, 706,
        721, 736, 752, 767, 783, 798, 813, 829,
    ]
