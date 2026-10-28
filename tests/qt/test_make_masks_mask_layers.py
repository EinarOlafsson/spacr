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


@pytest.mark.parametrize('choice', ['cancel', 'not-folder'])
def test_rejected_add_layer_keeps_the_current_masks_and_layer_configuration(
        screen, folder, monkeypatch, choice):
    _open(screen, folder)
    before = screen._canvas.mask.copy()
    history = screen._history.head().copy()
    layers = [dict(layer) for layer in screen._layers]
    source = {path: path.read_bytes() for path in folder.rglob('*.tif')}
    if choice == 'cancel':
        monkeypatch.setattr(mm.QFileDialog, 'getExistingDirectory', lambda *a, **k: '')
        assert not screen._on_add_mask_layer()
    else:
        assert not screen._on_add_mask_layer(str(folder / 'f_0.tif'))
        assert 'not a folder' in screen._status_label.text()
    np.testing.assert_array_equal(screen._canvas.mask, before)
    np.testing.assert_array_equal(screen._history.head(), history)
    assert screen._layers == layers
    assert {path: path.read_bytes() for path in source} == source


def test_unsaved_layer_edits_survive_base_selection_and_target_cancellation(
        screen, folder, qtbot):
    _open(screen, folder)
    screen._on_layer_pick(3, 3)
    edited = screen._canvas.mask.copy()
    history = screen._history.head().copy()
    source = {path: path.read_bytes() for path in folder.rglob('*.tif')}
    assert screen._edits_unsaved()
    assert not screen._on_add_mask_layer(str(folder / 'model_a'))
    screen._on_layer_base_changed(1)
    assert screen._layer_base == 1
    assert 'keeps its edits' in screen._status_label.text()
    target = screen._layer_target
    index = screen._layer_target_combo.findData(str(folder / 'model_b'))
    assert index >= 0
    screen._layer_target_combo.setCurrentIndex(index)
    assert screen._layer_pending_target == str(folder / 'model_b')
    qtbot.mouseClick(screen._layer_confirm_no, Qt.LeftButton)
    assert screen._layer_pending_target is None and screen._layer_target == target
    np.testing.assert_array_equal(screen._canvas.mask, edited)
    np.testing.assert_array_equal(screen._history.head(), history)
    assert engine._read_layer_settings(str(folder))['base'] == str(folder / 'model_a')
    assert {path: path.read_bytes() for path in source} == source


def test_corrupt_optional_layer_is_skipped_without_replacing_the_base_mask(
        screen, folder):
    _open(screen, folder)
    corrupt = folder / 'corrupt-layer'
    corrupt.mkdir()
    path = corrupt / 'f_0.tif'
    path.write_bytes(b'not a TIFF mask')
    source = {p: p.read_bytes() for p in folder.rglob('*.tif')}
    before = screen._canvas.mask.copy()
    assert screen._on_add_mask_layer(str(corrupt))
    assert screen._layers[-1]['path'] == str(corrupt)
    assert screen._layer_masks[-1] is None
    assert not engine._layer_has_objects(str(corrupt), 'f_0.tif')
    np.testing.assert_array_equal(screen._canvas.mask, before)
    assert {p: p.read_bytes() for p in source} == source


@pytest.mark.parametrize('accepted', [False, True])
def test_layer_colour_picker_changes_only_the_chosen_outline_and_persists_it(
        screen, folder, qtbot, monkeypatch, accepted):
    from PySide6.QtGui import QColor
    from spacr.qt.widgets import colour_picker

    _open(screen, folder)
    original = [dict(layer) for layer in screen._layers]
    mask = screen._canvas.mask.copy()
    sources = {path: path.read_bytes() for path in folder.rglob('*.tif')}
    chosen = QColor('#3456ab') if accepted else QColor()
    monkeypatch.setattr(colour_picker, 'pick_colour', lambda *args: chosen)
    screen._on_layer_colour(1)
    assert screen._layers[0] == original[0] and screen._layers[2] == original[2]
    assert screen._layers[1]['colour'] == ('#3456ab' if accepted else original[1]['colour'])
    assert np.array_equal(screen._canvas.mask, mask)
    saved = json.loads((folder / 'curate_layers.json').read_text())
    assert saved['layers'][1]['colour'] == screen._layers[1]['colour']
    assert all(path.read_bytes() == value for path, value in sources.items())


@pytest.mark.parametrize('stale_index', [-1, 99])
def test_stale_layer_controls_cannot_mutate_a_reconfigured_review(screen, folder, stale_index):
    _open(screen, folder)
    config = (folder / 'curate_layers.json').read_bytes()
    original = [dict(layer) for layer in screen._layers]
    mask = screen._canvas.mask.copy()
    screen._on_layer_colour(stale_index, '#102030')
    screen._on_layer_visible(stale_index, False)
    screen._on_layer_flagged(stale_index, True)
    screen._on_layer_required(stale_index, True)
    screen._on_layer_base_changed(stale_index)
    assert screen._layers == original
    assert (folder / 'curate_layers.json').read_bytes() == config
    assert np.array_equal(screen._canvas.mask, mask)


@pytest.mark.parametrize('mode', ['any', 'all'])
def test_required_layers_skip_only_fields_without_the_required_objects_and_restore_them(
        screen, folder, qtbot, mode):
    imageio.imwrite(folder / 'model_a' / 'f_0.tif', np.zeros_like(BASE))
    imageio.imwrite(folder / 'model_b' / 'f_1.tif', np.zeros_like(BASE))
    _open(screen, folder)
    snapshots = {path: path.read_bytes() for path in folder.rglob('*.tif')}
    screen._on_layer_required(1, True)
    screen._on_layer_required(2, True)
    index = screen._layer_require_combo.findData(mode)
    assert index >= 0
    screen._on_layer_require_mode(index)
    qtbot.waitUntil(lambda: all((str(folder / layer), filename) in screen._layer_object_cache
                               for layer in ('model_a', 'model_b')
                               for filename in screen._image_files), timeout=10000)
    assert [screen._field_matches_layers(i) for i in range(3)] == (
        [True, True, True] if mode == 'any' else [False, False, True])
    screen._on_layer_required(1, False)
    screen._on_layer_required(2, False)
    assert all(screen._field_matches_layers(i) for i in range(3))
    assert all(path.read_bytes() == value for path, value in snapshots.items())
    assert not (folder / STATUS_FILENAME).exists()


@pytest.mark.parametrize('absolute', [False, True])
def test_new_curated_folder_name_changes_the_target_without_overwriting_a_layer(
        screen, folder, absolute):
    _open(screen, folder)
    snapshots = {path: path.read_bytes() for path in folder.rglob('*.tif')}
    target = folder / 'a_new_review'
    screen._layer_name_edit.setText(str(target) if absolute else target.name)
    screen._on_layer_folder_named()
    assert screen._layer_target == str(target)
    assert screen._layer_pending_target is None
    assert not target.exists()
    screen._on_layer_target_chosen(0)
    assert screen._layer_target == str(target)
    assert all(path.read_bytes() == value for path, value in snapshots.items())


@pytest.mark.parametrize('action', ['flag', 'visible', 'clear'])
def test_layer_review_display_changes_preserve_the_editable_mask_and_source_labels(
        screen, folder, action):
    _open(screen, folder)
    mask = screen._canvas.mask.copy()
    sources = {path: path.read_bytes() for path in folder.rglob('*.tif')}
    if action == 'flag':
        screen._on_layer_flagged(1, True)
        assert screen._layer_only_flagged and screen._layers_flagged() == [1]
        screen._on_layer_flagged(1, False)
        assert not screen._layers_flagged()
    elif action == 'visible':
        screen._on_layer_visible(2, False)
        assert not screen._layers[2]['visible']
        screen._on_layer_visible(2, True)
        assert screen._layers[2]['visible']
    else:
        screen._clear_layers()
        assert screen._canvas.mode == mm.MODE_BRUSH
        assert not screen._layers and screen._canvas.layer_overlay is None
        assert screen._layer_section.isHidden()
    assert np.array_equal(screen._canvas.mask, mask)
    assert all(path.read_bytes() == value for path, value in sources.items())


@pytest.mark.parametrize('failure', ['missing', 'wrong-shape', 'occupied', 'cancelled'])
def test_paired_copy_refusal_preserves_original_images_and_leaves_no_partial_output(
        tmp_path, failure):
    root = tmp_path / 'source'
    root.mkdir()
    image, mask = root / 'field.tif', root / 'field_mask.tif'
    imageio.imwrite(image, np.arange(64, dtype=np.uint16).reshape(8, 8))
    imageio.imwrite(mask, np.ones((7, 7) if failure == 'wrong-shape' else (8, 8), dtype=np.uint16))
    destination = tmp_path / 'prepared'
    if failure == 'occupied':
        destination.mkdir()
        (destination / 'keep.txt').write_bytes(b'existing review')
    snapshots = {path: path.read_bytes() for path in root.iterdir()}
    candidate = root / 'absent.tif' if failure == 'missing' else mask
    with pytest.raises((ValueError, FileExistsError, InterruptedError)):
        mm._copy_paired_layout([(str(image), str(candidate))], str(destination),
                               cancelled=lambda: failure == 'cancelled')
    assert all(path.read_bytes() == value for path, value in snapshots.items())
    if failure == 'occupied':
        assert (destination / 'keep.txt').read_bytes() == b'existing review'
        assert list(destination.iterdir()) == [destination / 'keep.txt']
    else:
        assert not destination.exists()
    assert not list(tmp_path.glob('.spacr-paired-*'))


def test_flagged_restore_never_overwrites_an_object_added_after_the_removal():
    original = np.zeros((5, 5), dtype=np.uint16)
    flagged = np.zeros_like(original)
    flagged[1:4, 1:4] = 7
    original[0, 0] = 12
    snapshots = original.copy(), flagged.copy()
    removed = {(0, 7): {4: np.array([0])}}
    result, action = engine._flagged_pick(original, [(0, None), (0, flagged)], 1, 1, 20, removed)
    assert result[0, 0] == 12
    assert action['kind'] == 'add' and action['id'] == 20
    assert np.all(result[(flagged > 0) & (original == 0)] == 20)
    assert np.array_equal(original, snapshots[0]) and np.array_equal(flagged, snapshots[1])
    unchanged, no_action = engine._flagged_pick(original, [(0, flagged)], 0, 0, 21, removed)
    assert unchanged is None and no_action is None


def test_responsive_shortcut_fold_respects_a_manual_choice_until_the_window_is_wide(screen, qtbot):
    screen.resize(1000, 760)
    screen.show()
    qtbot.waitExposed(screen)
    screen._fit_the_side_column()
    pane = screen._view_pane
    assert pane.is_collapsed('Shortcuts') and screen._side_folded_for_width
    pane.set_collapsed('Shortcuts', False, by_user=True)
    assert screen._side_held_by_user and not screen._side_folded_for_width
    screen._fit_the_side_column()
    assert not pane.is_collapsed('Shortcuts')
    screen.resize(mm._COMPACT_WIDTH + 100, 900)
    screen._fit_the_side_column()
    assert not screen._side_held_by_user and not pane.is_collapsed('Shortcuts')
    screen.resize(1000, 760)
    screen._fit_the_side_column()
    assert pane.is_collapsed('Shortcuts') and screen._side_folded_for_width


@pytest.mark.parametrize('review', ['duplicates', 'legacy-colours', 'no-layers'])
def test_layer_session_configuration_recovers_without_duplicate_layers_or_losing_the_field(
        screen, folder, review):
    _open(screen, folder)
    before = screen._canvas.mask.copy()
    original = {path: path.read_bytes() for path in folder.rglob('*.tif')}
    paths = [str(folder / 'masks'), str(folder / 'model_a'), str(folder / 'model_b')]
    if review == 'duplicates':
        paths += paths
    elif review == 'legacy-colours':
        (folder / 'curate_layers.json').write_text(json.dumps({
            'layers': [{'path': paths[0], 'colour': 'not-a-colour', 'flagged': 'old'},
                       {'path': paths[1], 'colour': '#123456', 'visible': False}],
            'base': paths[1], 'require_mode': 'obsolete', 'outline_thickness': .5}))
    else:
        paths = []
    screen._set_layers(paths, root=str(folder), settings_root=str(folder), require=[])
    assert np.array_equal(screen._canvas.mask, before)
    assert all(path.read_bytes() == value for path, value in original.items())
    if review == 'no-layers':
        assert not screen._layers and screen._canvas.mode == mm.MODE_BRUSH
        assert screen._canvas.layer_overlay is None
    else:
        assert len(screen._layers) == len({row['path'] for row in screen._layers}) == 3
        assert not screen._layers_required()
        if review == 'legacy-colours':
            assert screen._layer_base == 1 and screen._layer_require_mode == 'any'
            assert screen._layers[1]['colour'] == '#123456' and not screen._layers[1]['visible']
            assert screen._canvas.outline_thickness == .5
            assert all(len(row['colour']) == 7 and row['colour'].startswith('#') for row in screen._layers)


def test_no_matching_required_layer_keeps_the_current_field_and_disabling_it_restores_navigation(
        screen, folder, qtbot):
    for path in (folder / 'model_a').glob('*.tif'):
        imageio.imwrite(path, np.zeros_like(BASE))
    _open(screen, folder)
    field = screen._current_index
    mask = screen._canvas.mask.copy()
    screen._on_layer_required(1, True)
    qtbot.waitUntil(lambda: all((str(folder / 'model_a'), name) in screen._layer_object_cache
                               for name in screen._image_files), timeout=10000)
    assert not any(screen._field_matches_layers(i) for i in range(3))
    assert screen._current_index == field and np.array_equal(screen._canvas.mask, mask)
    screen._on_layer_required(1, False)
    assert all(screen._field_matches_layers(i) for i in range(3))
    assert screen._step_target(1) == 1
    assert not (folder / STATUS_FILENAME).exists()


@pytest.mark.parametrize('thickness', [.5, 2.0])
def test_real_layer_outline_render_is_cached_at_the_requested_width_without_editing_labels(
        screen, folder, thickness):
    _open(screen, folder)
    canvas = screen._canvas
    before = canvas.mask.copy()
    source = {path: path.read_bytes() for path in folder.rglob('*.tif')}
    screen._on_outline_thickness_changed(thickness)
    assert screen._outline_thickness.value() == thickness
    assert screen._layer_outline_thickness.value() == thickness
    picture = canvas._outline_picture((0, 0, 32, 32), (256, 256), thickness)
    assert picture is not None and not picture.isNull()
    same = canvas._outline_picture((0, 0, 32, 32), (256, 256), thickness)
    assert same is picture
    alpha = np.frombuffer(picture.bits(), dtype=np.uint8).reshape(256, picture.bytesPerLine())[:, 3::4]
    assert np.any(alpha > 0) and np.any(alpha == 0)
    assert np.array_equal(canvas.mask, before)
    assert all(path.read_bytes() == value for path, value in source.items())
    stored = json.loads((folder / 'curate_layers.json').read_text())
    assert stored['outline_thickness'] == thickness


@pytest.mark.parametrize('cancel_at', ['confirmation', 'parent-folder'])
def test_canceling_real_paired_layout_setup_never_launches_a_copy_or_changes_the_open_field(
        screen, folder, monkeypatch, cancel_at):
    from PySide6.QtWidgets import QFileDialog

    _open(screen, folder)
    before = screen._canvas.mask.copy()
    paths = {path: path.read_bytes() for path in folder.rglob('*.tif')}
    monkeypatch.setattr(screen, '_confirm', lambda *args: cancel_at != 'confirmation')
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', lambda *args: '')
    assert not screen._open_paired_layout({str(folder / 'f_0.tif'): str(folder / 'masks/f_0.tif')})
    assert not screen._folder_job_running()
    assert np.array_equal(screen._canvas.mask, before)
    assert all(path.read_bytes() == value for path, value in paths.items())
    assert not list(folder.parent.glob('make_masks*'))


def test_unsuccessful_mask_save_cannot_mark_a_field_as_kept_or_advance_the_queue(
        screen, folder, monkeypatch):
    _open(screen, folder)
    target = Path(screen._layer_target)
    before = screen._canvas.mask.copy()
    monkeypatch.setattr(screen, '_save_mask', lambda: None)
    assert screen._on_curate(True) is None
    assert screen._current_index == 0
    assert not (target / STATUS_FILENAME).exists()
    assert np.array_equal(screen._canvas.mask, before)
    assert not (folder / STATUS_FILENAME).exists()


def test_decision_log_write_failure_retains_the_previous_ledger_and_source_masks(
        screen, folder, monkeypatch):
    _open(screen, folder)
    filename = screen._image_files[0]
    screen._on_layer_pick(24, 24)
    first = screen._write_layer_decisions(filename)
    assert first is not None
    path = Path(first)
    stored = path.read_bytes()
    previous = dict(screen._layer_previous)
    sources = {p: p.read_bytes() for p in folder.rglob('*.tif')}

    def fail(*args):
        raise OSError('ledger destination is read-only')

    monkeypatch.setattr(engine, '_write_layer_decisions', fail)
    assert screen._write_layer_decisions(filename) is None
    assert path.read_bytes() == stored and screen._layer_previous == previous
    assert all(p.read_bytes() == value for p, value in sources.items())


def test_curated_added_objects_resume_with_their_source_identity_and_do_not_reload_the_new_base(
        screen, folder, tmp_path, qtbot):
    target = tmp_path / 'review'
    _open(screen, folder, target)
    screen._on_layer_pick(24, 24)
    expected = screen._canvas.mask.copy()
    screen._on_curate(True)
    _settle(qtbot, screen)
    screen._go_to_index(0)
    _settle(qtbot, screen)
    assert screen._loaded_from_save_path
    assert screen._layer_added == {3: {'layer': str(folder / 'model_b'), 'source_id': 3}}
    screen._on_layer_base_changed(1)
    assert screen._layer_base == 1 and screen._loaded_from_save_path
    np.testing.assert_array_equal(screen._canvas.mask, expected)
    assert screen._layer_added[3]['source_id'] == 3
    assert int(imageio.imread(target / 'f_0.tif')[24, 24]) == 3


@pytest.mark.parametrize('pairs', [[], [('one-path',)]])
def test_invalid_pairing_request_cannot_create_a_destination(tmp_path, pairs):
    destination = tmp_path / 'review'
    with pytest.raises(ValueError):
        mm._copy_paired_layout(pairs, str(destination))
    assert not destination.exists() and not list(tmp_path.glob('.spacr-paired-*'))


@pytest.mark.parametrize('thickness', [.5, 2.0])
def test_visible_layer_outlines_change_real_canvas_pixels_without_changing_the_mask(
        screen, folder, qtbot, thickness):
    from PySide6.QtGui import QImage

    _open(screen, folder)
    screen.resize(1280, 900)
    screen.show()
    qtbot.waitExposed(screen)
    canvas = screen._canvas
    screen._on_outline_thickness_changed(thickness)
    before = canvas.mask.copy()
    shown = canvas.grab().toImage().convertToFormat(QImage.Format_RGBA8888)
    for index in range(len(screen._layers)):
        screen._on_layer_visible(index, False)
    hidden = canvas.grab().toImage().convertToFormat(QImage.Format_RGBA8888)
    assert shown.size() == hidden.size() and not shown.isNull()
    first = np.frombuffer(shown.bits(), dtype=np.uint8).copy()
    second = np.frombuffer(hidden.bits(), dtype=np.uint8)
    assert np.count_nonzero(first != second) > 0
    np.testing.assert_array_equal(canvas.mask, before)
    assert not screen._edits_unsaved()


def test_source_mutation_during_a_real_paired_copy_refuses_publication_and_cleans_its_stage(tmp_path):
    import os

    root = tmp_path / 'source'
    root.mkdir()
    image, mask = root / 'field.tif', root / 'field_mask.tif'
    imageio.imwrite(image, np.arange(64, dtype=np.uint16).reshape(8, 8))
    imageio.imwrite(mask, np.ones((8, 8), dtype=np.uint16))
    original = image.read_bytes(), mask.read_bytes()
    destination = tmp_path / 'prepared'
    mutated = []

    def observe_copy():
        if not mutated and any((stage / 'images/field.tif').exists()
                               for stage in tmp_path.glob('.spacr-paired-*')):
            stamp = image.stat()
            os.utime(image, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000_000))
            mutated.append(True)
        return False

    with pytest.raises(ValueError, match='source changed'):
        mm._copy_paired_layout([(str(image), str(mask))], str(destination), cancelled=observe_copy)
    assert mutated == [True]
    assert image.read_bytes() == original[0] and mask.read_bytes() == original[1]
    assert not destination.exists() and not list(tmp_path.glob('.spacr-paired-*'))


def test_empty_curated_folder_entry_chooses_a_safe_new_target_without_overwriting_a_layer(screen, folder):
    _open(screen, folder)
    original = {path: path.read_bytes() for path in folder.rglob('*.tif')}
    screen._layer_name_edit.clear()
    screen._on_layer_folder_named()
    screen._on_layer_target_chosen(0)
    target = Path(screen._layer_target)
    assert target.parent == folder and target.name.startswith('masks_curated_')
    assert str(target) not in {row['path'] for row in screen._layers}
    assert not target.exists() and screen._layer_pending_target is None
    assert all(path.read_bytes() == value for path, value in original.items())


@pytest.mark.parametrize('shown_ids', [[], [1], [1, 2]])
def test_clearing_filtered_layers_twice_restores_all_ids_without_editing_or_saving(
        screen, folder, shown_ids):
    _open(screen, folder)
    before = screen._canvas.mask.copy()
    history = screen._history.head().copy()
    originals = {path: path.read_bytes() for path in folder.rglob('*.tif')}
    screen._show_curated_ids(shown_ids)
    np.testing.assert_array_equal(screen._canvas.display_ids, shown_ids)
    screen._clear_layers()
    screen._clear_layers()
    assert screen._canvas.display_ids is None
    assert screen._canvas.layer_overlay is None and screen._canvas.layer_source is None
    assert not screen._layers_active() and screen._layer_section.isHidden()
    assert screen._canvas.mode == mm.MODE_BRUSH
    np.testing.assert_array_equal(screen._canvas.mask, before)
    np.testing.assert_array_equal(screen._history.head(), history)
    assert all(path.read_bytes() == data for path, data in originals.items())


@pytest.mark.parametrize('stage', ['unopened', 'cleared'])
def test_late_layer_actions_cannot_edit_an_ordinary_or_unopened_editor(
        screen, folder, monkeypatch, stage):
    if stage == 'cleared':
        _open(screen, folder)
        screen._clear_layers()
    before = None if screen._canvas.mask is None else screen._canvas.mask.copy()
    originals = {path: path.read_bytes() for path in folder.rglob('*.tif')}
    if stage == 'unopened':
        assert not screen._edits_unsaved()
        monkeypatch.setattr(mm.QFileDialog, 'getExistingDirectory',
                            lambda *a, **k: pytest.fail('Unopened layer action prompted'))
        assert not screen._on_add_mask_layer()
        assert 'Open one folder' in screen._status_label.text()
    screen._on_layer_pick(3, 3)
    assert not screen._on_merge_layers()
    assert not screen._layers_active()
    if before is None:
        assert screen._canvas.mask is None
    else:
        np.testing.assert_array_equal(screen._canvas.mask, before)
    assert all(path.read_bytes() == data for path, data in originals.items())


def test_changing_the_base_of_an_unedited_field_loads_the_selected_source_exactly(screen, folder):
    _open(screen, folder)
    originals = {path: path.read_bytes() for path in folder.rglob('*.tif')}
    assert not screen._edits_unsaved() and not screen._loaded_from_save_path
    screen._on_layer_base_changed(1)
    assert screen._layer_base == 1
    np.testing.assert_array_equal(screen._canvas.mask, MODEL_A)
    np.testing.assert_array_equal(screen._history.head(), MODEL_A)
    assert all(path.read_bytes() == data for path, data in originals.items())


def test_reselecting_a_confirmed_layer_target_and_empty_confirmation_are_idempotent(screen, folder):
    _open(screen, folder)
    index = screen._layer_target_combo.findData(str(folder / 'model_a'))
    assert index > 0
    screen._on_layer_target_chosen(index)
    assert screen._layer_pending_target == str(folder / 'model_a')
    screen._confirm_layer_target()
    assert screen._layer_target == str(folder / 'model_a')
    mask = screen._canvas.mask.copy()
    before = {path: path.read_bytes() for path in folder.rglob('*') if path.is_file()}
    screen._on_layer_target_chosen(index)
    screen._confirm_layer_target()
    assert screen._layer_pending_target is None and screen._layer_confirm.isHidden()
    assert screen._layer_target == str(folder / 'model_a')
    np.testing.assert_array_equal(screen._canvas.mask, mask)
    assert all(path.read_bytes() == data for path, data in before.items())


@pytest.mark.parametrize('matching', [False, True])
def test_reopening_a_required_layer_review_selects_the_first_available_field_without_writing(
        screen, folder, matching):
    for index in range(3):
        imageio.imwrite(folder / 'model_a' / f'f_{index}.tif',
                        MODEL_A if matching and index == 1 else np.zeros_like(BASE))
    originals = {path: path.read_bytes() for path in folder.rglob('*.tif')}
    screen._set_layers([str(folder / 'masks'), str(folder / 'model_a')],
                       root=str(folder), settings_root=str(folder),
                       require=[str(folder / 'model_a')])
    assert screen._open_folder(str(folder), files=[f'f_{index}.tif' for index in range(3)])
    assert screen._current_index == (1 if matching else 0)
    np.testing.assert_array_equal(screen._canvas.mask, BASE)
    assert all(path.read_bytes() == data for path, data in originals.items())
    assert not (folder / STATUS_FILENAME).exists()
