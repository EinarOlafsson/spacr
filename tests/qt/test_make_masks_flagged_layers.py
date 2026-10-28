"""Flagged and required mask layers in Make Masks.

A layer can be flagged as holding problems to review: with Show only
flagged on, only flagged layers are drawn and only the curated objects that
overlap a flagged object (or were edited on the field) are filled, and Pick
works on flagged objects even with the base layer hidden. A layer can also be
required: Previous, Next and Keep then visit only images where any (or
every) required layer has an object. Nothing changes when no layer is
flagged or required.
"""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from spacr import cli_make_masks
from spacr.curation_queue import build_queue
from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm
from spacr.qt.screens.make_masks import MakeMasksScreen


def _labels(*boxes):
    out = np.zeros((32, 32), dtype=np.uint16)
    for label, (r0, r1, c0, c1) in boxes:
        out[r0:r1, c0:c1] = label
    return out


BASE = _labels((1, (2, 8, 2, 8)), (2, (12, 18, 12, 18)), (3, (2, 8, 22, 28)))
SUSPECT = _labels((1, (2, 8, 2, 8)))
WRONG = _labels((5, (22, 28, 22, 28)))
EMPTY = _labels()
SUSPECT_FIELDS = {0, 2}
WRONG_FIELDS = {0, 1}


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    root = tmp_path / "plaques"
    for name in ("masks", "masks_suspect", "masks_model_wrong", "masks_model_right"):
        (root / name).mkdir(parents=True)
    rng = np.random.default_rng(3)
    for index in range(4):
        imageio.imwrite(root / f"f_{index}.tif",
                        rng.integers(0, 4000, (32, 32), dtype=np.uint16))
        imageio.imwrite(root / "masks" / f"f_{index}.tif", BASE)
        imageio.imwrite(root / "masks_suspect" / f"f_{index}.tif",
                        SUSPECT if index in SUSPECT_FIELDS else EMPTY)
        imageio.imwrite(root / "masks_model_wrong" / f"f_{index}.tif",
                        WRONG if index in WRONG_FIELDS else EMPTY)
        imageio.imwrite(root / "masks_model_right" / f"f_{index}.tif", BASE)
    return root


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    cli_make_masks.hand_over(None)
    cli_make_masks._hand_over_layers(None)
    made = MakeMasksScreen()
    qtbot.addWidget(made)
    yield made
    made._layer_scan_generation += 1
    cli_make_masks.hand_over(None)
    cli_make_masks._hand_over_layers(None)


NAMES = ("masks", "masks_suspect", "masks_model_wrong", "masks_model_right")


def _open(qtbot, screen, folder, require=None, mode=None):
    handed = {"layers": [str(folder / name) for name in NAMES],
              "save_to": None, "root": str(folder)}
    if require is not None:
        handed["require"] = [str(folder / name) for name in require]
    if mode is not None:
        handed["require_mode"] = mode
    cli_make_masks._hand_over_layers(handed)
    assert screen.open_queue(build_queue(folder, order="name",
                                         include_reviewed=True))
    qtbot.waitUntil(lambda: not screen._loading and screen._reader_worker is None
                    and screen._load_worker is None)


def _settle(qtbot, screen):
    assert mm._wait_for_mask_writes(timeout=30)
    qtbot.waitUntil(lambda: not screen._loading and screen._canvas.mask is not None,
                    timeout=30000)


def _visits(screen, qtbot):
    seen = [screen._current_index]
    while True:
        target = screen._step_target(1)
        if target is None:
            return seen
        screen._go_to_index(target)
        _settle(qtbot, screen)
        seen.append(screen._current_index)


def test_engine_flag_defaults_and_matching():
    assert engine._layer_flagged_by_name("/q/masks_suspect")
    assert engine._layer_flagged_by_name("/q/masks_model_WRONG/")
    assert not engine._layer_flagged_by_name("/q/masks_model_right")
    assert engine._layers_match([], "all") and engine._layers_match([], "any")
    assert engine._layers_match([True, False], "any")
    assert not engine._layers_match([True, False], "all")
    assert engine._layers_match([True, True], "all")
    assert list(engine._flagged_curated_ids(BASE, [SUSPECT, None])) == [1]
    changed = np.zeros(BASE.shape, dtype=bool)
    changed[14, 14] = True
    assert list(engine._flagged_curated_ids(BASE, [SUSPECT], changed)) == [1, 2]


def test_engine_flagged_pick_removes_restores_and_adds():
    removed = {}
    out, action = engine._flagged_pick(BASE, [(1, SUSPECT)], 3, 3, 9, removed)
    assert action["kind"] == "remove" and action["ids"] == [1]
    assert 1 not in out
    removed[(1, 1)] = action["pixels"]
    back, action = engine._flagged_pick(out, [(1, SUSPECT)], 3, 3, 9, removed)
    assert action["kind"] == "restore" and np.array_equal(back, BASE)
    added, action = engine._flagged_pick(BASE, [(2, WRONG)], 24, 24, 9, {})
    assert action["kind"] == "add" and int(added[24, 24]) == 9
    assert engine._flagged_pick(BASE, [(2, WRONG)], 14, 14, 9, {}) == (None, None)


def test_suspect_and_wrong_layers_are_flagged_and_reviewed_by_default(qtbot, screen, folder):
    _open(qtbot, screen, folder)
    flags = [layer["flagged"] for layer in screen._layers]
    assert flags == [False, True, True, False]
    assert screen._layer_only_flagged is True
    assert screen._layer_only_flagged_box.isChecked()
    assert list(screen._canvas.display_ids) == [1]
    assert screen._canvas.layer_source[1][0][2] is False, "the base was drawn"
    assert screen._canvas.layer_source[1][3][2] is False, "an unflagged layer was drawn"
    assert screen._canvas.layer_source[1][1][2] is True
    assert screen._layers[0]["visible"] is True, "a visibility box changed"
    screen._layer_only_flagged_box.setChecked(False)
    assert screen._canvas.display_ids is None
    assert screen._canvas.layer_source[1][0][2] is True


def test_pick_on_a_flagged_object_works_with_the_base_hidden(qtbot, screen, folder):
    _open(qtbot, screen, folder)
    screen._on_layer_visible(0, False)
    screen._on_layer_pick(3, 3)
    assert 1 not in screen._canvas.mask
    screen._on_layer_pick(3, 3)
    assert np.array_equal(screen._canvas.mask, BASE), "a second click did not restore"
    screen._on_layer_pick(3, 3)
    screen._on_undo()
    assert np.array_equal(screen._canvas.mask, BASE)
    screen._on_redo()
    assert 1 not in screen._canvas.mask
    screen._on_layer_pick(24, 24)
    added = int(screen._canvas.mask[24, 24])
    assert added == 4
    assert screen._layer_added[added]["layer"].endswith("masks_model_wrong")
    assert added in screen._canvas.display_ids, "an added object is not shown"
    before = screen._canvas.mask.copy()
    screen._on_layer_pick(14, 14)
    assert np.array_equal(screen._canvas.mask, before), "a hidden object was picked"
    kinds = [edit.kind for edit in screen._log.edits]
    assert "layer_remove" in kinds and "layer_restore" in kinds and "layer_add" in kinds


def test_hidden_objects_are_saved_unchanged(screen, folder, qtbot):
    _open(qtbot, screen, folder)
    screen._on_layer_pick(3, 3)
    screen._on_curate(True)
    _settle(qtbot, screen)
    target = Path(screen._layer_target)
    saved = imageio.imread(target / "f_0.tif")
    assert 1 not in saved
    assert int(saved[14, 14]) == 2 and int(saved[4, 24]) == 3
    log = json.loads((target / "f_0.decisions.json").read_text())
    assert log["removed"] == [1]


@pytest.mark.parametrize(("require", "mode", "expected"), [
    ((), None, [0, 1, 2, 3]),
    (("masks_suspect",), None, [0, 2]),
    (("masks_model_wrong",), None, [0, 1]),
    (("masks_suspect", "masks_model_wrong"), "any", [0, 1, 2]),
    (("masks_suspect", "masks_model_wrong"), "all", [0]),
])
def test_required_layers_choose_the_images(screen, folder, qtbot, require, mode,
                                           expected):
    _open(qtbot, screen, folder, require=require, mode=mode)
    _settle(qtbot, screen)
    if require:
        qtbot.waitUntil(lambda: "match" in screen._layer_match_label.text(),
                        timeout=30000)
        assert screen._layer_match_label.text() == (
            f"{len(expected)} of 4 images match")
    else:
        assert screen._layer_match_label.text() == "All 4 images are shown."
    assert _visits(screen, qtbot) == expected


def test_the_filter_starts_on_a_matching_image_and_changes_in_place(
        screen, folder, qtbot):
    _open(qtbot, screen, folder, require=("masks_suspect",))
    _settle(qtbot, screen)
    screen._go_to_index(2)
    _settle(qtbot, screen)
    screen._on_layer_required(1, False)
    screen._on_layer_required(2, True)
    _settle(qtbot, screen)
    assert screen._current_index == 0, "the filter left a non-matching field on screen"
    qtbot.waitUntil(lambda: screen._layer_match_label.text() == "2 of 4 images match",
                    timeout=30000)
    screen._layer_require_combo.setCurrentIndex(
        screen._layer_require_combo.findData("all"))
    screen._on_layer_required(1, True)
    qtbot.waitUntil(lambda: screen._layer_match_label.text() == "1 of 4 images match",
                    timeout=30000)


def test_flags_requirements_and_mode_are_kept(screen, folder, qtbot):
    _open(qtbot, screen, folder)
    stored = json.loads((folder / "curate_layers.json").read_text())
    assert all("flagged" not in row and "required" not in row
               for row in stored["layers"])
    assert stored["show_only_flagged"] is True and "require_mode" not in stored
    screen._on_layer_flagged(2, False)
    screen._on_layer_flagged(3, True)
    screen._on_layer_required(1, True)
    screen._layer_require_combo.setCurrentIndex(
        screen._layer_require_combo.findData("all"))
    screen._layer_only_flagged_box.setChecked(False)
    _settle(qtbot, screen)
    stored = json.loads((folder / "curate_layers.json").read_text())
    rows = {Path(row["path"]).name: row for row in stored["layers"]}
    assert rows["masks_model_wrong"]["flagged"] is False
    assert rows["masks_model_right"]["flagged"] is True
    assert rows["masks_suspect"]["required"] is True
    assert stored["require_mode"] == "all" and stored["show_only_flagged"] is False
    again = MakeMasksScreen()
    try:
        _open(qtbot, again, folder)
        _settle(qtbot, again)
        assert [layer["flagged"] for layer in again._layers] == [False, True, False, True]
        assert [layer["required"] for layer in again._layers] == [False, True, False, False]
        assert again._layer_require_mode == "all"
        assert again._layer_only_flagged is False and again._canvas.display_ids is None
    finally:
        again._layer_scan_generation += 1
        again.deleteLater()


def test_unflagged_layers_change_nothing(screen, tmp_path, qtbot):
    root = tmp_path / "plain"
    for name in ("masks", "model_a"):
        (root / name).mkdir(parents=True)
    imageio.imwrite(root / "f_0.tif", np.ones((32, 32), dtype=np.uint16))
    imageio.imwrite(root / "masks" / "f_0.tif", BASE)
    imageio.imwrite(root / "model_a" / "f_0.tif", WRONG)
    cli_make_masks._hand_over_layers({
        "layers": [str(root / "masks"), str(root / "model_a")],
        "save_to": None, "root": str(root)})
    assert screen.open_queue(build_queue(root, order="name", include_reviewed=True))
    qtbot.waitUntil(lambda: not screen._loading and screen._reader_worker is None
                    and screen._load_worker is None)
    assert screen._canvas.display_ids is None
    assert not screen._layer_only_flagged_box.isEnabled()
    stored = json.loads((root / "curate_layers.json").read_text())
    assert set(stored) == {"layers", "base", "target", "outline_thickness"}
    assert all(set(row) == {"path", "colour", "visible"} for row in stored["layers"])


def test_the_cli_hands_over_required_layers(folder, monkeypatch, tmp_path):
    caught = []
    monkeypatch.setattr(cli_make_masks, "open_editor",
                        lambda queue: caught.append(cli_make_masks._take_layers()) or 0)
    monkeypatch.chdir(tmp_path)
    code = cli_make_masks.main([
        "--folder", str(folder), "--order", "name", "--layers", *NAMES,
        "--require-layers", "masks_suspect", "masks_model_wrong",
        "--require-mode", "all"])
    assert code == 0
    assert [Path(p).name for p in caught[0]["require"]] == [
        "masks_suspect", "masks_model_wrong"]
    assert caught[0]["require_mode"] == "all"


def test_the_cli_refuses_required_layers_it_was_not_given(folder, capsys):
    code = cli_make_masks.main([
        "--folder", str(folder), "--layers", "masks", "masks_suspect",
        "--require-layers", "masks_model_wrong"])
    assert code == cli_make_masks.EXIT_USAGE
    assert "not among --layers" in capsys.readouterr().err
    assert cli_make_masks.main([
        "--folder", str(folder), "--require-mode", "all"]) == cli_make_masks.EXIT_USAGE


def test_engine_merge_fills_only_background_with_new_ids():
    other = _labels((7, (4, 10, 4, 10)), (8, (22, 26, 2, 6)))
    out, added = engine._layer_merge(BASE, [(1, other), (2, WRONG), (3, None)], 10)
    assert [(row["id"], row["layer"], row["source_id"]) for row in added] == [
        (10, 1, 7), (11, 1, 8), (12, 2, 5)]
    assert int(out[4, 4]) == 1, "an added object took the overlay's pixels"
    assert int(out[9, 9]) == 10 and int(out[24, 24]) == 12
    assert engine._layer_merge(BASE, [(1, BASE)], 10) == (None, [])


def test_merge_visible_layers_is_one_step_and_is_logged(screen, folder, qtbot):
    _open(qtbot, screen, folder)
    screen._on_layer_visible(2, False)
    assert screen._on_merge_layers() is False, "a hidden layer was merged"
    screen._on_layer_visible(2, True)
    assert screen._on_merge_layers() is True
    merged = int(screen._canvas.mask[24, 24])
    assert merged == 4 and merged in screen._canvas.display_ids
    screen._on_undo()
    assert int(screen._canvas.mask[24, 24]) == 0
    screen._on_redo()
    assert int(screen._canvas.mask[24, 24]) == merged
    assert [edit.kind for edit in screen._log.edits].count("layer_merge") == 1
    screen._on_curate(True)
    _settle(qtbot, screen)
    log = json.loads((Path(screen._layer_target) / "f_0.decisions.json").read_text())
    assert log["merged"] == [{"layer": str(folder / "masks_model_wrong"),
                              "ids": [merged], "source_ids": [5]}]


def test_only_flagged_layers_are_merged_while_showing_only_flagged(qtbot, screen, folder):
    extra = _labels((9, (26, 30, 10, 14)))
    for index in range(4):
        imageio.imwrite(folder / "masks_model_right" / f"f_{index}.tif", extra)
    _open(qtbot, screen, folder)
    screen._on_layer_visible(2, False)
    assert screen._on_merge_layers() is False
    screen._layer_only_flagged_box.setChecked(False)
    assert screen._on_merge_layers() is True
    assert int(screen._canvas.mask[27, 11]) == 4


def _edit_and(screen, path, qtbot):
    if path == "previous":
        screen._go_to_index(2)
        _settle(qtbot, screen)
    edited = screen._image_files[screen._current_index]
    screen._on_layer_pick(3, 3)
    assert 1 not in screen._canvas.mask
    if path == "keep":
        screen._on_curate(True)
    elif path == "save":
        screen._on_save()
    elif path == "next":
        screen._btn_save_on_navigation.setChecked(True)
        screen._on_next()
    else:
        screen._btn_save_on_navigation.setChecked(True)
        screen._on_prev()
    _settle(qtbot, screen)
    return edited


@pytest.mark.parametrize("path", ["keep", "save", "next", "previous"])
@pytest.mark.parametrize("overwrite", [False, True])
def test_every_save_path_follows_the_save_target(screen, folder, qtbot, path,
                                                 overwrite):
    _open(qtbot, screen, folder)
    _settle(qtbot, screen)
    originals = {p.name: p.read_bytes() for p in (folder / "masks").glob("*.tif")}
    if overwrite:
        index = screen._layer_target_combo.findData(str(folder / "masks"))
        screen._layer_target_combo.setCurrentIndex(index)
        screen._layer_confirm_yes.click()
        assert screen._layer_target == str(folder / "masks")
    edited = _edit_and(screen, path, qtbot)
    original = folder / "masks" / edited
    if overwrite:
        assert 1 not in imageio.imread(original)
        assert int(imageio.imread(original)[14, 14]) == 2
        assert not list(folder.glob("masks_curated_*/*.tif"))
    else:
        saved = Path(screen._layer_target) / edited
        assert 1 not in imageio.imread(saved)
        assert Path(screen._layer_target).name.startswith("masks_curated_")
        assert {p.name: p.read_bytes()
                for p in (folder / "masks").glob("*.tif")} == originals


def test_a_new_layer_does_not_take_a_stored_layers_colour(qtbot, screen, folder):
    stored = {"layers": [
        {"path": str(folder / "masks"), "colour": "#00e5ff", "visible": True},
        {"path": str(folder / "masks_model_right"), "colour": "#ffd60a", "visible": True},
        {"path": str(folder / "masks_model_wrong"), "colour": "#ff4d6d", "visible": True}]}
    (folder / "curate_layers.json").write_text(json.dumps(stored))
    _open(qtbot, screen, folder)
    colours = [layer["colour"] for layer in screen._layers]
    assert len(set(colours)) == 4, colours
    assert colours[1] == "#7cff6b"
