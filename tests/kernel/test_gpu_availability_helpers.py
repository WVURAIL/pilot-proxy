# coding=utf-8
"""GPU availability regression tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


GPU_TEST_MODULES = tuple(
    path
    for path in sorted(Path(__file__).parent.glob("test_*_gpu.py"))
    if "def _import_cupy_or_skip" in path.read_text(encoding="utf-8")
)
if len(GPU_TEST_MODULES) != 5:
    raise RuntimeError(
        "expected exactly five GPU test modules with availability helpers"
    )


@pytest.mark.parametrize("module_path", GPU_TEST_MODULES, ids=lambda path: path.stem)
def test_gpu_test_helper_skips_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    module_path: Path,
) -> None:
    module_name = f"_pilot_proxy_{module_path.stem}_availability_probe"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)

    monkeypatch.setitem(sys.modules, "cupy", ModuleType("cupy"))
    monkeypatch.setattr(
        module,
        "cuda_available",
        lambda: (False, "sentinel CUDA-unavailable reason"),
    )

    with pytest.raises(pytest.skip.Exception, match="sentinel CUDA-unavailable reason"):
        module._import_cupy_or_skip()
