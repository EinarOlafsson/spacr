"""Item 545 in Make Masks: the field on screen or the whole queue, out and back.

Driven offscreen through the screen's own methods (the file dialogs are the
only part not exercised). The button is an ALPHA feature: shown only while
Preferences -> "Show alpha features" is on.
"""
from __future__ import annotations

import json

import imageio.v2 as imageio
import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr import mask_io  # noqa: E402


def _mask(seed: int) -> np.ndarray:
    mask = np.zeros((48, 48), np.uint16)
    mask[4:14, 4:14] = 3 + seed
    mask[6:10, 6:10] = 0
    mask[4:14, 14:24] = 8
    mask[30:34, 30:34] = 11
    mask[40:44, 5:9] = 11
    return mask


@pytest.fixture
def screen(qtbot, qt_theme_applied, tmp_path):
    from spacr.qt.screens.make_masks import MakeMasksScreen

    folder = tmp_path / "field"
    (folder / "cell_mask_stack").mkdir(parents=True)
    rng = np.random.default_rng(3)
    for i in range(3):
        imageio.imwrite(folder / f"f{i}.tif",
                        rng.integers(0, 65535, (48, 48), dtype=np.uint16))
    imageio.imwrite(folder / "cell_mask_stack" / "f2.tif", _mask(2))
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    widget._open_folder(str(folder), masks_dir=str(folder / "cell_mask_stack"))
    widget._canvas.mask = _mask(0)
    return widget


def test_the_button_offers_every_export_and_import(screen):
    assert screen._btn_rois.menu() is not None
    assert set(screen._roi_actions) == {
        "export_field_geojson", "export_field_imagej", "export_field_coco",
        "export_all_geojson", "export_all_imagej", "export_all_coco",
        "import_field", "import_all_geojson", "import_all_imagej",
        "import_all_coco"}
    assert screen._roi_object_type() == "cell"


def test_the_button_is_hidden_unless_alpha_features_are_shown(
        qtbot, qt_theme_applied, monkeypatch):
    from spacr.qt import preferences
    from spacr.qt.screens.make_masks import MakeMasksScreen

    for shown in (False, True):
        monkeypatch.setattr(preferences, "get_show_alpha", lambda s=shown: s)
        widget = MakeMasksScreen()
        qtbot.addWidget(widget)
        assert widget._btn_rois.isVisibleTo(widget) is shown


@pytest.mark.parametrize("fmt,suffix", (("geojson", ".geojson"),
                                        ("imagej", ".zip"),
                                        ("coco", ".json")))
def test_the_field_on_screen_round_trips(screen, tmp_path, fmt, suffix):
    if fmt == "imagej":
        pytest.importorskip("roifile")
    path = screen.export_field_rois(str(tmp_path / f"out{suffix}"), fmt)
    back = mask_io.import_rois(path, (48, 48), fmt, file_name="f0.tif")
    assert np.array_equal(back["cell"], _mask(0))
    screen._canvas.mask = np.zeros((48, 48), np.uint16)
    assert screen.import_field_rois(path, fmt) == 3
    assert np.array_equal(screen._canvas.mask, _mask(0))
    assert screen._history.can_undo()
    screen._on_undo()
    assert not np.any(screen._canvas.mask)


def test_every_field_exports_and_imports(screen, tmp_path):
    """A saved field goes through Make Masks' own save, which splits a label
    lying in two pieces (one id, one object), so it is compared with that."""
    import tifffile

    from spacr.qt import mask_engine as engine

    out = tmp_path / "geo"
    written = screen.export_all_rois(str(out), "geojson")
    assert sorted(p.split("/")[-1] for p in written) == ["f0.geojson",
                                                       "f2.geojson"]
    coco = tmp_path / "all.json"
    screen.export_all_rois(str(coco), "coco")
    data = json.loads(coco.read_text())
    assert [i["file_name"] for i in data["images"]] == ["f0.tif", "f2.tif"]
    assert {a["object_type"] for a in data["annotations"]} == {"cell"}

    edited = _mask(5)
    other = tmp_path / "other"
    mask_io.export_rois(edited, other / "f1.geojson", object_type="cell")
    mask_io.export_rois(edited, other / "f2.geojson", object_type="cell")
    done = screen.import_all_rois(str(other), "geojson")
    assert done == ["f1.tif", "f2.tif"]
    saved = tifffile.imread(str(screen._masks_dir) + "/f1.tif")
    assert np.array_equal(saved, engine.canonical_labels(edited))
    assert np.array_equal(screen._canvas.mask, _mask(0))

    screen._canvas.mask = np.zeros((48, 48), np.uint16)
    assert screen.import_all_rois(str(coco), "coco") == ["f0.tif", "f2.tif"]
    assert np.array_equal(screen._canvas.mask, _mask(0))
    assert np.array_equal(
        tifffile.imread(str(screen._masks_dir) + "/f2.tif"),
        engine.canonical_labels(_mask(2)))
