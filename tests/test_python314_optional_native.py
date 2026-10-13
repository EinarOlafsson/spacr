"""Contracts that keep optional native features out of core imports."""

from __future__ import annotations

import builtins
import subprocess
import sys

import numpy as np
import pytest


# The guard the child interpreter runs, plus a readback of what it actually
# imported. Shared by the real run and the poisoned contrast run below, so the
# two differ only in the setup that precedes them.
_CHILD_GUARD = """
assert "pylibCZIrw" not in sys.modules
assert "btrack" not in sys.modules
print("MODULES:" + ",".join(sorted(m for m in sys.modules
                                   if m.startswith("spacr."))))
"""


def test_core_modules_do_not_eagerly_import_optional_native_features():
    code = """
import sys
import numpy
import spacr.io
import spacr.timelapse
""" + _CHILD_GUARD
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == "", f"child wrote to stderr:\n{proc.stderr}"

    # The child really did import the modules under test — without this the
    # two asserts inside it would be vacuously true in a bare interpreter.
    assert "MODULES:" in proc.stdout
    imported = proc.stdout.split("MODULES:", 1)[1].strip().split(",")
    assert "spacr.io" in imported
    assert "spacr.timelapse" in imported

    # Contrast case: the identical guard, in a child where one optional
    # native IS present in sys.modules, must fail. Proves the child's asserts
    # are live and that a child assertion really reaches us as a non-zero
    # return code rather than being swallowed.
    poisoned = """
import sys, types
sys.modules["pylibCZIrw"] = types.ModuleType("pylibCZIrw")
import numpy
import spacr.io
import spacr.timelapse
""" + _CHILD_GUARD
    bad = subprocess.run([sys.executable, "-c", poisoned],
                         capture_output=True, text=True)
    assert bad.returncode != 0
    assert "AssertionError" in bad.stderr


def _block_import(monkeypatch, package):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == package or name.startswith(package + "."):
            raise ImportError(f"blocked optional dependency: {package}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def test_missing_pylibczirw_error_names_the_czi_extra(monkeypatch):
    from spacr import io

    _block_import(monkeypatch, "pylibCZIrw")
    with pytest.raises(ImportError, match=r"spacr\[czi\]"):
        io._load_pylibczi()


def test_missing_btrack_error_names_the_btrack_extra(monkeypatch, tmp_path):
    from spacr import timelapse

    _block_import(monkeypatch, "btrack")
    with pytest.raises(ImportError, match=r"spacr\[btrack\]"):
        timelapse._btrack_track_cells(
            src=str(tmp_path),
            name="synthetic",
            batch_filenames=[],
            object_type="cell",
            plot=False,
            save=False,
            masks_3D=np.zeros((1, 2, 2), dtype=np.uint16),
            mode="btrack",
            timelapse_remove_transient=False,
        )
