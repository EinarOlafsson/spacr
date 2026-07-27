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


# ---------------------------------------------------------------------------
# Did TensorFlow actually load?
#
# Everything above proves spaCR still imports when TF is *blocked*. That is a
# weaker claim than it looks, and it missed a real regression: `spacr.utils`
# did `import umap.umap_ as umap`, umap's package __init__ imports
# umap.parametric_umap -> tensorflow, and umap catches the ImportError and
# substitutes a stub ParametricUMAP. So with the guard fixture active the
# import succeeded and the test passed — while a normal `import spacr.utils`
# on a machine that happens to have TF installed loaded TensorFlow AND Keras,
# 2.6 s of it, plus the cpu_feature_guard banner.
#
# The only claim that catches that is the direct one: after importing spaCR,
# tensorflow is not in sys.modules. It has to be checked in a FRESH
# interpreter — inside the pytest session another test may already have
# imported TF for its own reasons, and a subprocess also keeps this file's
# hard-won sys.modules hygiene (see _reimported) intact by never touching
# sys.modules at all.
# ---------------------------------------------------------------------------

#: What a user or a worker process actually imports.
ENTRY_POINTS = [
    "spacr", "spacr.utils", "spacr.core", "spacr.settings", "spacr.io",
    "spacr.measure", "spacr.object", "spacr.plot", "spacr.ml",
    "spacr.deep_spacr", "spacr.submodules", "spacr.timelapse",
    "spacr.spacr_cellpose", "spacr.sequencing", "spacr.toxo",
    "spacr.hyperparam",
]

_TF_ROOTS = ("tensorflow", "keras", "tf_keras")


def _run_in_fresh_interpreter(body, timeout=600):
    """Run ``body`` in a subprocess and return its completed process.

    :param body: Python source to execute.
    :param timeout: seconds before the run is killed.
    :returns: the ``subprocess.CompletedProcess``.
    """
    import os
    import subprocess
    import sys

    # Hand the child the same import path this session is using, so it
    # imports the spaCR under test rather than whatever the cwd happens to
    # make importable.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("QT_QPA_PLATFORM", "offscreen")

    return subprocess.run(
        [sys.executable, "-c", body], env=env,
        capture_output=True, text=True, timeout=timeout,
    )


#: Imports each name in turn and prints the first one that drags TF in.
_IMPORT_AND_REPORT = """
import sys
roots = {roots!r}
for name in {names!r}:
    __import__(name)
    loaded = [r for r in roots if r in sys.modules]
    if loaded:
        print("TFLOADED", name, ",".join(loaded))
        break
else:
    print("CLEAN")
"""


def test_importing_spacr_does_not_load_tensorflow():
    """No spaCR entry point may put TensorFlow or Keras into sys.modules.

    TF is not a declared dependency (setup.py:105 has it commented out); it is
    only ever present because something else in the environment installed it.
    Importing it anyway costs seconds per process — paid once per worker in a
    spawn/forkserver pool — prints TF's banner over the run log, and is an
    off-main-thread segfault vector in a GUI run.
    """
    proc = _run_in_fresh_interpreter(
        _IMPORT_AND_REPORT.format(roots=_TF_ROOTS, names=ENTRY_POINTS))
    assert proc.returncode == 0, (
        f"importing spaCR's entry points failed:\n{proc.stdout}\n{proc.stderr}")
    out = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    assert out == "CLEAN", (
        f"a spaCR entry point loaded a TF-backed library: {out}\n"
        f"Find it with: python -X importtime -c 'import <that module>' "
        f"and read the nesting.")


def test_importing_the_qt_app_does_not_load_tensorflow():
    """The GUI process is where a stray TF import does the most damage."""
    proc = _run_in_fresh_interpreter(
        _IMPORT_AND_REPORT.format(roots=_TF_ROOTS, names=["spacr.qt.app"]))
    if proc.returncode != 0 and "PySide6" in proc.stderr:
        pytest.skip("PySide6 not importable in this environment")
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert proc.stdout.strip().splitlines()[-1] == "CLEAN", proc.stdout


def test_using_umap_does_not_load_tensorflow():
    """Deferring the umap import postpones TF; it does not prevent it.

    ``spacr.utils.umap`` is lazy, so ``import spacr.utils`` is clean either
    way — but the moment anything reads ``umap.UMAP`` the package __init__
    runs and, unblocked, imports TensorFlow. This asserts the block, not just
    the deferral, and checks that umap landed on its own documented no-TF
    path (a stub ParametricUMAP) rather than failing to import at all.
    """
    body = """
import sys
from spacr.utils import umap
assert "tensorflow" not in sys.modules, "lazy import already loaded TF"
reducer = umap.UMAP(n_neighbors=3, n_components=2, random_state=0)
assert type(reducer).__module__ == "umap.umap_", type(reducer).__module__
# umap's own TF-less fallback: __init__ catches the ImportError and defines a
# stub ParametricUMAP in the package namespace. spaCR never uses it.
assert sys.modules["umap"].ParametricUMAP.__module__ == "umap"
loaded = [r for r in ("tensorflow", "keras") if r in sys.modules]
print("TFLOADED" if loaded else "CLEAN", ",".join(loaded))
"""
    proc = _run_in_fresh_interpreter(body)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert proc.stdout.strip().splitlines()[-1] == "CLEAN", (
        f"running UMAP loaded a TF-backed library: {proc.stdout}")


def test_the_tf_blocker_is_removed_after_the_import_it_guarded():
    """The block is scoped. It must not linger on sys.meta_path.

    A permanent blocker would be a global decision spaCR has no business
    making — anything else in the process is free to want TensorFlow.
    """
    import sys as _sys
    from spacr.utils import _BlockTensorFlowFinder, umap

    umap.UMAP  # force the guarded import (no-op if already done)
    assert not any(isinstance(f, _BlockTensorFlowFinder)
                   for f in _sys.meta_path), _sys.meta_path


def test_the_tf_blocker_refuses_only_tf_backed_roots():
    """It must let every other import through by returning None."""
    from spacr.utils import _BlockTensorFlowFinder, _TensorFlowIsNotADependency

    finder = _BlockTensorFlowFinder()
    assert finder.find_spec("numpy") is None
    assert finder.find_spec("umap.parametric_umap") is None
    assert finder.find_spec("kerasplotlib_not_a_root") is None
    for name in ("tensorflow", "tensorflow.python", "keras", "keras.src",
                 "tf_keras"):
        with pytest.raises(_TensorFlowIsNotADependency):
            finder.find_spec(name)


def test_no_module_imports_umap_directly():
    """A bare ``import umap`` re-opens the hole.

    umap's package __init__ imports TensorFlow, so spaCR reaches umap only
    through ``spacr.utils.umap``, which blocks TF for that import. This is the
    grep that keeps a future ``import umap`` from quietly undoing it.
    """
    import re
    from pathlib import Path
    import spacr

    root = Path(spacr.__file__).parent
    pat = re.compile(r"^\s*(?:import\s+umap|from\s+umap(\.|\s))")
    offenders = []
    for py in root.rglob("*.py"):
        for i, line in enumerate(py.read_text(errors="ignore").splitlines(), 1):
            if pat.match(line):
                offenders.append(f"{py.relative_to(root)}:{i}: {line.strip()}")
    assert not offenders, (
        "import umap through spacr.utils instead — a bare import pulls in "
        "umap.parametric_umap -> tensorflow:\n" + "\n".join(offenders))


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
