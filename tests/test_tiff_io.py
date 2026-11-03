"""Tests for spaCR's canonical scientific TIFF writer."""

from __future__ import annotations

import ast
import io
import warnings
import zipfile
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


def test_image_bundle_preserves_three_plane_scientific_stack(tmp_path):
    """A three-plane stack in an image zip is data, not inferred RGB."""
    from matplotlib.figure import Figure

    from spacr.figures.bundle import _register_figure_data, _save_zip

    stack = np.arange(3 * 8 * 8, dtype=np.uint16).reshape(3, 8, 8)
    figure = Figure()
    _register_figure_data(figure, stack, kind="image")

    output = _save_zip(figure, str(tmp_path / "stack.zip"), formats=["png"])
    with zipfile.ZipFile(output) as archive:
        with tifffile.TiffFile(io.BytesIO(archive.read("image_0.tif"))) as tif:
            assert tif.pages[0].photometric.name == "MINISBLACK"
            np.testing.assert_array_equal(tif.asarray(), stack)


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


def test_an_undeclared_channel_last_array_is_split_into_its_channels():
    """(Y, X, C) with C <= 4 and shorter than Y and X: one plane per channel."""
    from spacr.tiff_io import _yokogawa_planes

    images = np.arange(6 * 7 * 3, dtype=np.uint16).reshape(6, 7, 3)
    planes = _yokogawa_planes(images, '', 'rgb_like.tif')
    assert [(t, c) for t, c, _ in planes] == [(0, 0), (0, 1), (0, 2)]
    for _, c, plane in planes:
        assert plane.flags['C_CONTIGUOUS']
        np.testing.assert_array_equal(plane, images[:, :, c])


def test_an_undeclared_deep_stack_is_projected_over_depth():
    from spacr.tiff_io import _yokogawa_planes

    images = np.zeros((6, 8, 9), dtype=np.uint16)
    images[3, 2, 2] = 50
    [(t, c, plane)] = _yokogawa_planes(images, '', 'stack.tif')
    assert (t, c) == (0, 0) and plane.shape == (8, 9) and plane[2, 2] == 50


def test_a_4d_array_declared_zt_projects_each_timepoint_over_depth(capsys):
    """ZT puts time second; TZ needs no warning either."""
    from spacr.tiff_io import _undeclared_4d_planes

    images = np.zeros((3, 2, 4, 4), dtype=np.uint16)
    images[1, 0, 0, 0] = 9
    images[2, 1, 1, 1] = 7
    planes = _undeclared_4d_planes(images, 'ZTYX', 'zt.tif')
    assert [(t, c) for t, c, _ in planes] == [(0, 0), (1, 0)]
    assert planes[0][2][0, 0] == 9 and planes[1][2][1, 1] == 7
    tz = _undeclared_4d_planes(images, 'TZYX', 'tz.tif')
    assert len(tz) == 3 and tz[2][2][1, 1] == 7
    assert "WARNING" not in capsys.readouterr().out
