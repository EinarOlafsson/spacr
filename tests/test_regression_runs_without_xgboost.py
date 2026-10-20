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
