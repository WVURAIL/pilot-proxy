# coding=utf-8
from __future__ import annotations

from types import SimpleNamespace

import pytest

from pilot_proxy import kernel as kernel_module
from pilot_proxy.kernel import FStatKernel


class _Function:
    def __init__(self, callback=lambda *args: None):
        self.callback = callback
        self.argtypes = None
        self.restype = object()

    def __call__(self, *args):
        return self.callback(*args)


def test_void_kernel_call_surfaces_last_error() -> None:
    error = b""

    def fail(*_args):
        nonlocal error
        error = b"FStat API error: handle is null."

    lib = SimpleNamespace(
        FStat_Compute_Powers_U64=_Function(fail),
        FStat_LastError=_Function(lambda: error),
    )
    wrapper = object.__new__(FStatKernel)
    wrapper._lib = lib
    wrapper._has_last_error = True
    wrapper._has_powers_u64 = True

    with pytest.raises(RuntimeError, match="handle is null"):
        wrapper.compute_powers_u64(None, 1, 2)


def test_setup_signatures_marks_void_functions_as_void(monkeypatch) -> None:
    names = {
        "FStat_GetSpecs",
        "FStat_GetVersion",
        "FStat_LastError",
        "FStat_Create",
        "FStat_Compute_DiagnosticFloat",
        "FStat_Destroy",
        "FStat_Compute_Powers_U64",
    }
    lib = SimpleNamespace(**{name: _Function() for name in names})
    wrapper = object.__new__(FStatKernel)
    wrapper._lib = lib
    monkeypatch.setattr(
        kernel_module,
        "_has_symbol",
        lambda _lib, name: name in names,
    )

    wrapper._setup_signatures()

    for name in (
        "FStat_GetSpecs",
        "FStat_GetVersion",
        "FStat_Compute_DiagnosticFloat",
        "FStat_Destroy",
        "FStat_Compute_Powers_U64",
    ):
        assert getattr(lib, name).restype is None
