"""Guard: spaCR must never import TensorFlow / stardist / csbdeep.

Per the project's no-TensorFlow rule, none of spaCR's code paths may
pull in TF-backed libraries. TF (via stardist/csbdeep) is heavy, prints
the noisy cpu_feature_guard banner, and is an off-main-thread segfault
vector when imported during a GUI run.

These tests import spaCR's core modules with tensorflow/stardist/csbdeep
import-blocked, and assert every module still imports. If someone adds
a ``import stardist`` somewhere, this fails loudly.
"""
from __future__ import annotations

import builtins
import importlib

import pytest

BLOCKED_ROOTS = ("tensorflow", "stardist", "csbdeep")

SPACR_MODULES = [
    "spacr.core", "spacr.measure", "spacr.io", "spacr.plot", "spacr.ml",
    "spacr.deep_spacr", "spacr.submodules", "spacr.sequencing",
    "spacr.utils", "spacr.object", "spacr.toxo", "spacr.spacr_cellpose",
    "spacr.timelapse", "spacr.settings",
]


@pytest.fixture
def _block_tf(monkeypatch):
    """Make importing tensorflow/stardist/csbdeep raise ImportError."""
    real_import = builtins.__import__

    def _guarded(name, *args, **kwargs):
        if name.split(".")[0] in BLOCKED_ROOTS:
            raise ImportError(f"{name} is blocked by the no-TF guard test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded)
    yield


@pytest.mark.parametrize("mod", SPACR_MODULES)
def test_module_imports_without_tensorflow(mod, _block_tf):
    """Each spaCR module must import with TF/stardist/csbdeep blocked."""
    # Force a fresh import so the guard is exercised even if the module
    # was already cached by an earlier test.
    import sys
    sys.modules.pop(mod, None)
    try:
        importlib.import_module(mod)
    except ImportError as e:
        if "blocked by the no-TF guard" in str(e):
            pytest.fail(
                f"{mod} imports a TF-backed library "
                f"(tensorflow/stardist/csbdeep): {e}")
        raise


def test_qt_app_imports_without_tensorflow(_block_tf):
    import sys
    sys.modules.pop("spacr.qt.app", None)
    try:
        importlib.import_module("spacr.qt.app")
    except ImportError as e:
        if "blocked by the no-TF guard" in str(e):
            pytest.fail(f"spacr.qt.app pulls in a TF-backed library: {e}")
        raise


def test_no_tf_import_string_in_source():
    """Belt-and-braces: grep spaCR source for direct TF/stardist imports
    (excluding comments + the logging_util level-setter which only names
    'tensorflow' as a string to silence its logger)."""
    import re
    from pathlib import Path
    import spacr
    root = Path(spacr.__file__).parent
    offenders = []
    pat = re.compile(r"^\s*(import|from)\s+(tensorflow|stardist|csbdeep)\b")
    for py in root.rglob("*.py"):
        for i, line in enumerate(py.read_text(errors="ignore").splitlines(), 1):
            if pat.match(line):
                offenders.append(f"{py.relative_to(root)}:{i}: {line.strip()}")
    assert not offenders, (
        "spaCR source contains direct TF-backed imports:\n"
        + "\n".join(offenders))
