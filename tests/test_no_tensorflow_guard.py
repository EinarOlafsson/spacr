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
from contextlib import contextmanager

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


@contextmanager
def _reimported(mod):
    """Import ``mod`` fresh, then put the original module object back.

    The fresh import is the whole point of these tests — a module already
    in ``sys.modules`` would not execute its imports again and the guard
    would prove nothing. But leaving the *new* module object in
    ``sys.modules`` poisons the rest of the session: re-executing a module
    rebinds every class it defines, so a class captured earlier is no
    longer the class a later lazy ``from .io import X`` resolves to, and
    ``except X`` / ``pytest.raises(X)`` stop matching.

    That is not hypothetical. ``spacr.ml._assign_prcfo_parts`` imports
    ``TimelapseKeyMismatch`` from ``spacr.io`` at call time, while
    ``tests/test_timelapse_prcfo_split.py`` imports it at module-import
    time; with ``spacr.io`` left re-imported by this file, the two are
    different classes and
    ``test_a_frame_mixing_both_key_shapes_is_reported`` fails with the
    exception it was asserting on escaping uncaught. It passed alone and
    failed in a full run, which is the worst way for a suite to fail.
    """
    import sys

    _missing = object()
    parent_name, _, child = mod.rpartition(".")
    parent = sys.modules.get(parent_name) if parent_name else None

    original = sys.modules.get(mod)
    # importlib.import_module also rebinds the child attribute on the parent
    # package, so restoring sys.modules alone is not enough: ``spacr.io``
    # would still be the fresh module even though ``sys.modules['spacr.io']``
    # was the original one.
    original_attr = getattr(parent, child, _missing) if parent else _missing

    sys.modules.pop(mod, None)
    try:
        yield
    finally:
        if original is not None:
            sys.modules[mod] = original
        else:
            sys.modules.pop(mod, None)
        if parent is not None:
            if original_attr is _missing:
                if hasattr(parent, child):
                    delattr(parent, child)
            else:
                setattr(parent, child, original_attr)


@pytest.mark.parametrize("mod", SPACR_MODULES)
def test_module_imports_without_tensorflow(mod, _block_tf):
    """Each spaCR module must import with TF/stardist/csbdeep blocked."""
    with _reimported(mod):
        try:
            importlib.import_module(mod)
        except ImportError as e:
            if "blocked by the no-TF guard" in str(e):
                pytest.fail(
                    f"{mod} imports a TF-backed library "
                    f"(tensorflow/stardist/csbdeep): {e}")
            raise


def test_the_guard_leaves_the_imported_modules_as_it_found_them():
    """The fresh import must not outlive the test that needed it.

    Every module here defines classes other modules catch by identity.
    """
    import sys

    before = {mod: sys.modules[mod] for mod in SPACR_MODULES
              if mod in sys.modules}
    assert before, "nothing imported yet; this test proves nothing"
    for mod in before:
        with _reimported(mod):
            importlib.import_module(mod)
    for mod, module in before.items():
        assert sys.modules[mod] is module, mod

    from spacr.io import TimelapseKeyMismatch
    from spacr.ml import _assign_prcfo_parts  # noqa: F401 - imports .io lazily
    import spacr.io
    assert spacr.io.TimelapseKeyMismatch is TimelapseKeyMismatch


def test_qt_app_imports_without_tensorflow(_block_tf):
    with _reimported("spacr.qt.app"):
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
