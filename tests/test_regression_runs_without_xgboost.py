"""Regression and the rest of spacr.ml load when XGBoost cannot.

2026-10-20 macOS acceptance: ``spacr.ml`` imported XGBoost at the top, and
XGBoost's native library failed against Homebrew's libomp 18.1.8. Every
function in the module went with it, and Regression said it was
"interactive-only in this Qt build".
"""
import builtins
import importlib
import sys

import pytest


def test_spacr_ml_and_the_regression_entry_load_without_xgboost(monkeypatch):
    real_import = builtins.__import__

    def no_xgboost(name, *args, **kwargs):
        if name == "xgboost" or name.startswith("xgboost."):
            raise OSError("\\ndlopen(libxgboost.dylib): Symbol not found: ___kmpc_dispatch_deinit")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "spacr.ml", raising=False)
    monkeypatch.setattr(builtins, "__import__", no_xgboost)
    ml = importlib.import_module("spacr.ml")
    assert callable(ml.perform_regression)
    with pytest.raises(RuntimeError, match="Symbol not found.*brew install libomp"):
        ml._xgboost_classifier()


def test_run_shows_the_import_error_instead_of_interactive_only(monkeypatch):
    from types import SimpleNamespace

    from spacr.qt import bridge
    from spacr.qt.screens import app_screen

    real_import = builtins.__import__

    def broken_ml(name, *args, **kwargs):
        if name == "spacr.ml":
            raise OSError("dlopen(libomp.dylib): image not found\nsecond line")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "spacr.ml", raising=False)
    monkeypatch.setattr(builtins, "__import__", broken_ml)
    assert bridge.resolve_pipeline_entry("regression") is None
    monkeypatch.setattr(builtins, "__import__", real_import)
    assert bridge._entry_import_error("regression") == (
        "dlopen(libomp.dylib): image not found")

    shown = []
    monkeypatch.setattr(app_screen, "resolve_pipeline_entry", lambda key: None)
    monkeypatch.setattr(app_screen.QMessageBox, "warning",
                        lambda parent, title, text: shown.append(("warning", text)))
    monkeypatch.setattr(app_screen.QMessageBox, "information",
                        lambda parent, title, text: shown.append(("information", text)))
    screen = SimpleNamespace(app_key="regression",
                             _wait_for_the_entry_import=lambda override: False)
    app_screen.AppScreen._on_run(screen)
    assert len(shown) == 1 and shown[0][0] == "warning"
    assert "image not found" in shown[0][1] and "interactive-only" not in shown[0][1]

    bridge._ENTRY_IMPORT_ERRORS.pop("regression", None)
    shown.clear()
    app_screen.AppScreen._on_run(screen)
    assert shown and shown[0][0] == "information"
