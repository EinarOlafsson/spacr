"""Mask layers in Make Masks: the screen half.

Any number of mask folders are drawn over the images as coloured outlines;
one editable curated overlay starts as a copy of the base layer, and the
Pick tool removes, restores and adds objects through the ordinary undo
history. Keep saves the overlay to the save folder (a new folder by
default, a layer folder only after an in-page confirmation) with a
decisions log, and progress goes to that folder's ``curate_status.csv``.
With no layers the editor is the ordinary one and saves the same bytes.
"""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from spacr import cli_make_masks
from spacr.curation_queue import STATUS_FILENAME, build_queue, read_status
from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm
from spacr.qt.screens.make_masks import MakeMasksScreen


def _labels(*boxes):
    out = np.zeros((32, 32), dtype=np.uint16)
    for label, (r0, r1, c0, c1) in boxes:
        out[r0:r1, c0:c1] = label
    return out


BASE = _labels((1, (2, 8, 2, 8)), (2, (12, 18, 12, 18)))
MODEL_A = _labels((1, (22, 28, 2, 8)), (2, (12, 18, 12, 18)))
MODEL_B = _labels((3, (22, 28, 22, 28)))


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    root = tmp_path / "plaques"
    for name in ("masks", "model_a", "model_b"):
        (root / name).mkdir(parents=True)
    rng = np.random.default_rng(2)
    for index in range(3):
        imageio.imwrite(root / f"f_{index}.tif",
                        rng.integers(0, 4000, (32, 32), dtype=np.uint16))
        imageio.imwrite(root / "masks" / f"f_{index}.tif", BASE)
        imageio.imwrite(root / "model_a" / f"f_{index}.tif", MODEL_A)
        imageio.imwrite(root / "model_b" / f"f_{index}.tif", MODEL_B)
    return root


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    cli_make_masks.hand_over(None)
    cli_make_masks._hand_over_layers(None)
    made = MakeMasksScreen()
    qtbot.addWidget(made)
    yield made
    cli_make_masks.hand_over(None)
    cli_make_masks._hand_over_layers(None)


def _settle(qtbot, screen):
    """Wait for queued mask writes and for the field being loaded."""
    assert mm._wait_for_mask_writes(timeout=30)
    qtbot.waitUntil(lambda: not screen._loading and screen._canvas.mask is not None,
                    timeout=30000)


def _open(screen, folder, target=None):
    cli_make_masks._hand_over_layers({
        "layers": [str(folder / "masks"), str(folder / "model_a"),
                   str(folder / "model_b")],
        "save_to": str(target) if target else None, "root": str(folder)})
    assert screen.open_queue(build_queue(folder, order="name",
                                         include_reviewed=True))


def test_layers_load_with_the_overlay_a_copy_of_the_base(screen, folder):
    _open(screen, folder)
    assert len(screen._layers) == 3
    assert len({layer["colour"] for layer in screen._layers}) == 3
    assert np.array_equal(screen._canvas.mask, BASE)
    assert screen._canvas.mode == mm.MODE_PICK
    assert not screen._mode_buttons[mm.MODE_PICK].isHidden()
    assert not screen._layer_section.isHidden()
    assert screen._canvas.layer_overlay is not None
    assert Path(screen._layer_target).name.startswith("masks_curated_")
    assert (folder / "curate_layers.json").is_file()


def test_remove_restore_and_undo_go_through_the_history(screen, folder):
    _open(screen, folder)
    screen._on_layer_pick(3, 3)
    assert 1 not in screen._canvas.mask
    screen._on_undo()
    assert np.array_equal(screen._canvas.mask, BASE)
    screen._on_redo()
    assert 1 not in screen._canvas.mask
    screen._on_layer_pick(3, 3)
    assert np.array_equal(screen._canvas.mask, BASE), "a second click did not restore"
    kinds = [edit.kind for edit in screen._log.edits]
    assert "layer_remove" in kinds and "layer_restore" in kinds


def test_add_and_unadd_another_layers_object(screen, folder):
    _open(screen, folder)
    screen._on_layer_pick(24, 24)
    added = int(screen._canvas.mask[24, 24])
    assert added == 3, "an added object did not get a new unique id"
    assert screen._layer_added[added]["layer"].endswith("model_b")
    screen._on_layer_pick(24, 24)
    assert added not in screen._canvas.mask
    screen._on_undo()
    assert int(screen._canvas.mask[24, 24]) == added
    screen._on_layer_pick(4, 24)
    assert int(screen._canvas.mask[24, 4]) == 4, "an id was reused"


def test_a_hidden_layer_is_neither_drawn_nor_picked(screen, folder):
    _open(screen, folder)
    screen._on_layer_visible(2, False)
    before = screen._canvas.mask.copy()
    screen._on_layer_pick(24, 24)
    assert np.array_equal(screen._canvas.mask, before)
    stored = json.loads((folder / "curate_layers.json").read_text())
    assert stored["layers"][2]["visible"] is False


def test_a_chosen_colour_is_kept_for_the_session(screen, folder):
    _open(screen, folder)
    screen._on_layer_colour(1, "#123456")
    stored = json.loads((folder / "curate_layers.json").read_text())
    assert stored["layers"][1]["colour"] == "#123456"
    again = MakeMasksScreen()
    try:
        _open(again, folder)
        assert again._layers[1]["colour"] == "#123456"
    finally:
        again.deleteLater()


def test_a_click_on_the_canvas_reaches_the_pick_tool(screen, folder, qtbot):
    _open(screen, folder)
    screen.resize(1200, 900)
    screen.show()
    qtbot.waitExposed(screen)
    screen._canvas.refresh()
    point = screen._canvas._image_to_canvas(4.5, 4.5)
    QTest.mouseClick(screen._canvas, Qt.LeftButton, Qt.NoModifier, QPoint(point))
    assert 1 not in screen._canvas.mask


def test_keep_saves_the_overlay_its_decisions_and_progress(screen, folder, qtbot):
    _open(screen, folder)
    original = (folder / "masks" / "f_0.tif").read_bytes()
    screen._on_layer_pick(3, 3)
    screen._on_layer_pick(24, 24)
    screen._on_curate(True)
    _settle(qtbot, screen)
    target = Path(screen._layer_target)
    saved = imageio.imread(target / "f_0.tif")
    assert 1 not in saved and int(saved[24, 24]) == 3 and int(saved[14, 14]) == 2
    log = json.loads((target / "f_0.decisions.json").read_text())
    assert log["removed"] == [1]
    assert log["added"] == [{"id": 3, "layer": str(folder / "model_b"), "source_id": 3}]
    assert (target / "f_0.tif.curation.json").is_file() or list(
        target.glob("f_0*.curation.json"))
    assert read_status(target)["f_0"].state == "done"
    assert not (folder / STATUS_FILENAME).exists()
    assert (folder / "masks" / "f_0.tif").read_bytes() == original
    assert screen._current_index == 1


def test_a_saved_field_resumes_from_the_save_folder(screen, folder, tmp_path,
                                                   qtbot):
    target = tmp_path / "review"
    _open(screen, folder, target)
    screen._on_layer_pick(3, 3)
    screen._on_curate(True)
    _settle(qtbot, screen)
    again = MakeMasksScreen()
    try:
        _open(again, folder, target)
        _settle(qtbot, again)
        assert 1 not in again._canvas.mask
        assert int(again._canvas.mask[14, 14]) == 2, "a lone object lost its id"
        assert again._loaded_from_save_path is True
    finally:
        again.deleteLater()


def test_a_layer_folder_needs_an_in_page_confirmation(screen, folder, qtbot):
    _open(screen, folder)
    first = screen._layer_target
    index = screen._layer_target_combo.findData(str(folder / "masks"))
    screen._layer_target_combo.setCurrentIndex(index)
    assert not screen._layer_confirm.isHidden()
    assert screen._layer_target == first, "the target changed before confirming"
    screen._layer_confirm_no.click()
    assert screen._layer_target == first
    assert screen._layer_target_combo.currentIndex() == 0
    screen._layer_target_combo.setCurrentIndex(index)
    screen._layer_confirm_yes.click()
    assert screen._layer_target == str(folder / "masks")
    screen._on_layer_pick(3, 3)
    screen._on_curate(True)
    _settle(qtbot, screen)
    assert 1 not in imageio.imread(folder / "masks" / "f_0.tif")


def test_a_named_new_folder_becomes_the_save_folder(screen, folder):
    _open(screen, folder)
    screen._layer_name_edit.setText("masks_review_v2")
    screen._on_layer_folder_named()
    assert screen._layer_target == str(folder / "masks_review_v2")


def test_layers_added_from_the_gui(screen, folder):
    assert screen._open_folder(str(folder))
    assert screen._mode_buttons[mm.MODE_PICK].isHidden()
    assert screen._on_add_mask_layer(str(folder / "masks"))
    assert screen._on_add_mask_layer(str(folder / "model_a"))
    assert [layer["name"] for layer in screen._layers] == ["masks", "model_a"]
    assert np.array_equal(screen._canvas.mask, BASE)
    screen._open_folder(str(folder / "model_a"))
    assert screen._layers == [], "layers outlived their folder"


def test_no_layers_is_the_ordinary_editor_and_saves_the_same_bytes(
        screen, folder, tmp_path):
    assert screen.open_queue(build_queue(folder, order="name"))
    assert screen._layers == []
    assert screen._layout_kwargs() == {}
    assert screen._mode_buttons[mm.MODE_PICK].isHidden()
    assert screen._layer_section.isHidden()
    assert screen._canvas.layer_overlay is None
    edited = BASE.copy()
    edited[edited == 1] = 0
    screen._canvas.mask = edited
    path = screen._save_mask()
    assert path == str(folder / "masks" / "f_0.tif")
    reference = tmp_path / "reference"
    (reference / "masks").mkdir(parents=True)
    (reference / "f_0.tif").write_bytes((folder / "f_0.tif").read_bytes())
    expected = engine.save_mask(str(reference), "f_0.tif", edited)
    assert Path(path).read_bytes() == Path(expected).read_bytes()
    assert not list(folder.glob("masks_curated_*"))
    assert not (folder / "curate_layers.json").exists()
    assert not list((folder / "masks").glob("*.decisions.json"))
    assert read_status(folder)["f_0"].state == "done"


def _canvas_array(canvas):
    from PySide6.QtGui import QImage

    image = canvas.grab().toImage().convertToFormat(QImage.Format_RGB32)
    width, height = image.width(), image.height()
    data = np.frombuffer(image.constBits(), dtype=np.uint8,
                         count=image.bytesPerLine() * height)
    return data.reshape(height, image.bytesPerLine())[:, :4 * width].reshape(
        height, width, 4)[..., :3].astype(int)


def _outline_ink(screen, thickness):
    screen._on_outline_thickness_changed(thickness)
    drawn = _canvas_array(screen._canvas)
    overlay, screen._canvas.layer_overlay = screen._canvas.layer_overlay, None
    try:
        bare = _canvas_array(screen._canvas)
    finally:
        screen._canvas.layer_overlay = overlay
    return np.abs(drawn - bare).sum()


def test_thinner_outlines_draw_less_on_the_canvas(screen, folder, qtbot):
    _open(screen, folder)
    screen.resize(1200, 900)
    screen.show()
    qtbot.waitExposed(screen)
    screen._canvas.refresh()
    ink = {t: _outline_ink(screen, t) for t in (0.25, 0.5, 1.0, 2.0)}
    assert 0 < ink[0.25] < ink[0.5] < ink[1.0] < ink[2.0]
    assert screen._layer_outline_thickness.value() == 2.0, "the two boxes disagree"


def test_the_default_width_draws_the_overlay_as_before(screen, folder, qtbot):
    _open(screen, folder)
    assert screen._outline_thickness.value() == engine._OUTLINE_THICKNESS_DEFAULT
    assert screen._canvas.outline_thickness == 1.0
    screen.resize(1200, 900)
    screen.show()
    qtbot.waitExposed(screen)
    screen._canvas.refresh()
    default = _canvas_array(screen._canvas)
    screen._canvas.layer_source = None
    assert np.array_equal(default, _canvas_array(screen._canvas)), (
        "width 1 no longer draws the image-resolution overlay")


def test_the_width_is_kept_in_the_preferences_and_the_layer_file(screen, folder):
    from spacr.qt.prefs import _s

    _open(screen, folder)
    screen._layer_outline_thickness.setValue(0.5)
    assert screen._outline_thickness.value() == 0.5
    assert float(_s().value(mm.OUTLINE_THICKNESS_KEY)) == 0.5
    stored = json.loads((folder / "curate_layers.json").read_text())
    assert stored["outline_thickness"] == 0.5
    stored["outline_thickness"] = 0.25
    (folder / "curate_layers.json").write_text(json.dumps(stored))
    _s().setValue(mm.OUTLINE_THICKNESS_KEY, 3.0)
    again = MakeMasksScreen()
    try:
        assert again._outline_thickness.value() == 3.0
        _open(again, folder)
        assert again._outline_thickness.value() == 0.25
        assert again._canvas.outline_thickness == 0.25
    finally:
        again.deleteLater()
        _s().remove(mm.OUTLINE_THICKNESS_KEY)


def test_without_layers_the_width_changes_nothing_drawn(screen, folder, qtbot):
    assert screen.open_queue(build_queue(folder, order="name"))
    _settle(qtbot, screen)
    screen.resize(1200, 900)
    screen.show()
    qtbot.waitExposed(screen)
    screen._canvas.refresh()
    before = _canvas_array(screen._canvas)
    screen._outline_thickness.setValue(0.25)
    assert np.array_equal(before, _canvas_array(screen._canvas))
    assert not (folder / "curate_layers.json").exists()
    screen._outline_thickness.setValue(1.0)
