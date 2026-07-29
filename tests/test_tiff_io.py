"""Tests for spaCR's canonical scientific TIFF writer."""

from __future__ import annotations

import ast
import warnings
from pathlib import Path

import numpy as np
import tifffile

from spacr.tiff_io import write_tiff


def test_scientific_stack_has_explicit_non_rgb_layout(tmp_path):
    path = tmp_path / "stack.tif"
    stack = np.arange(3 * 8 * 8, dtype=np.uint16).reshape(3, 8, 8)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        write_tiff(path, stack, metadata={"axes": "ZYX"})

    with tifffile.TiffFile(path) as tif:
        assert tif.series[0].axes == "ZYX"
        assert tif.pages[0].photometric.name == "MINISBLACK"
    assert np.array_equal(tifffile.imread(path), stack)


def test_explicit_rgb_options_override_scientific_defaults(
        tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        tifffile,
        "imwrite",
        lambda path, array, **kwargs: calls.append(kwargs),
    )

    write_tiff(
        tmp_path / "rgb.tif",
        np.zeros((8, 8, 3), dtype=np.uint8),
        photometric="rgb",
        planarconfig="separate",
    )

    assert calls == [{
        "photometric": "rgb",
        "planarconfig": "separate",
    }]


def test_production_tifffile_writes_use_the_canonical_helper():
    package = Path(__file__).parents[1] / "spacr"
    violations = []

    for path in package.rglob("*.py"):
        if path.name in {"tiff_io.py", "gui_elements.py"}:
            # tiff_io owns the one allowed direct call. gui_elements is the
            # retired Tk application and is intentionally out of scope.
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if not isinstance(function, ast.Attribute):
                continue
            owner = function.value
            if (
                function.attr in {"imwrite", "imsave"}
                and isinstance(owner, ast.Name)
                and owner.id in {"tifffile", "tiff"}
            ):
                violations.append(f"{path.relative_to(package)}:{node.lineno}")

    assert violations == []
