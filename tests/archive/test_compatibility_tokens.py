import importlib

from pilot_proxy.archive import pipeline
from pilot_proxy.archive.inventory import (
    INVENTORY_META_SCHEMA_KEY,
    INVENTORY_META_SCHEMA_VERSION,
)
from pilot_proxy.archive.sources.cadc_inventory import logical_unit_key


def test_persisted_archive_tokens_stay_compatible():
    assert pipeline._OUTPUT_LOCK_SUFFIX == ".datatrawl.lock"
    assert pipeline._QUARANTINE_LOCK_SUFFIX == ".datatrawl.lock"
    assert INVENTORY_META_SCHEMA_KEY == "datatrawl_inventory"
    assert INVENTORY_META_SCHEMA_VERSION == 1
    assert logical_unit_key("scope", "event", "name.h5") == (
        'cadc-datatrail:["scope","event","name.h5"]'
    )


def test_previous_module_paths_forward_to_archive_modules():
    names = {
        "_chime_coarse": "chime_coarse",
        "combine": "combine",
        "control": "control",
        "detector": "detector",
        "packed_reader": "packed_reader",
        "scan": "scan",
        "stream_kinds": "stream_kinds",
    }
    for previous_name, current_name in names.items():
        previous = importlib.import_module(
            f"pilot_proxy.datatrawl_plugins.{previous_name}"
        )
        current = importlib.import_module(f"pilot_proxy.archive.{current_name}")
        assert previous is current
