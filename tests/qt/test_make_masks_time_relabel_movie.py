"""Time mode: relabel a clicked cell across the movie, and paint a label while scrolling.

Real Make Masks screen on the synthetic four-frame movie of
``tests/qt/test_708_make_masks_label_navigation``: two square objects,
pixel labels 7 and 11, in every frame; the frames' TIFFs are checked after
the mask writer has rewritten them.
"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm

from .test_708_make_masks_label_navigation import (
    click_pixel, enable_time, screen, wait_idle, wheel)

__all__ = ["screen"]


def _written(qtbot, screen):
    qtbot.waitUntil(lambda: not getattr(screen, "_mask_write_requests", []), timeout=10000)


def _saved(screen, index, point):
    path = engine.mask_save_path(screen._folder, f"frame_{index}.tif")
    return int(imageio.imread(path)[point[1], point[0]])


def _two_tracks(qtbot, screen):
    """Trace cell A on (10, 10) forward and cell B on (30, 30) back; end on frame 0."""
    enable_time(screen)
    wheel(screen, 3, point=(10, 10))
    wait_idle(qtbot, screen)
    first = screen._selected_cell
    screen._add_lineage_cell()
    second = screen._selected_cell
    wheel(screen, -3, point=(30, 30))
    wait_idle(qtbot, screen)
    assert screen._current_index == 0
    cells = screen._lineage["cells"]
    assert (cells[first]["label"], cells[second]["label"]) == (1, 2)
    return first, second


def _edit_label_mode(screen, monkeypatch, answer):
    shown = []

    def ask(*args, **kwargs):
        shown.append(kwargs["text"])
        return answer, True

    monkeypatch.setattr(mm.QInputDialog, "getText", ask)
    screen._label_tool_button.click()
    assert screen._canvas.mode == mm.MODE_EDIT_LABEL and screen._time_label_enabled
    return shown


def test_a_clicked_cell_takes_the_proposed_next_label_in_every_frame_with_undo(qtbot, screen, monkeypatch):
    first, _second = _two_tracks(qtbot, screen)
    _written(qtbot, screen)
    shown = _edit_label_mode(screen, monkeypatch, "3")
    click_pixel(screen, (10, 10))
    assert shown == ["3"]
    assert int(screen._canvas.mask[10, 10]) == 3
    assert screen._lineage["cells"][first]["label"] == 3
    assert all(mapping.get("3") == first for mapping in screen._lineage["frames"].values())
    _written(qtbot, screen)
    assert [_saved(screen, index, (10, 10)) for index in (1, 2, 3)] == [3, 3, 3]
    screen._on_undo()
    assert int(screen._canvas.mask[10, 10]) == 1
    assert all(mapping.get("1") == first for mapping in screen._lineage["frames"].values())
    _written(qtbot, screen)
    assert [_saved(screen, index, (10, 10)) for index in (1, 2, 3)] == [1, 1, 1]


def test_a_manual_label_is_taken_as_typed(qtbot, screen, monkeypatch):
    _two_tracks(qtbot, screen)
    _edit_label_mode(screen, monkeypatch, "40")
    click_pixel(screen, (10, 10))
    assert int(screen._canvas.mask[10, 10]) == 40
    _written(qtbot, screen)
    assert [_saved(screen, index, (10, 10)) for index in (1, 2, 3)] == [40, 40, 40]


def test_a_label_another_cell_carries_moves_that_cell_after_yes(qtbot, screen, monkeypatch):
    first, second = _two_tracks(qtbot, screen)
    asked = []
    monkeypatch.setattr(screen, "_confirm", lambda title, text: asked.append(text) or True)
    _edit_label_mode(screen, monkeypatch, "2")
    click_pixel(screen, (10, 10))
    assert "Label 2 already belongs to another cell" in asked[0]
    cells = screen._lineage["cells"]
    assert cells[first]["label"] == 2 and cells[second]["label"] == 3
    assert int(screen._canvas.mask[10, 10]) == 2 and int(screen._canvas.mask[30, 30]) == 3
    _written(qtbot, screen)
    for index in (1, 2, 3):
        assert (_saved(screen, index, (10, 10)), _saved(screen, index, (30, 30))) == (2, 3)
    screen._on_undo()
    assert int(screen._canvas.mask[10, 10]) == 1 and int(screen._canvas.mask[30, 30]) == 2
    _written(qtbot, screen)
    for index in (1, 2, 3):
        assert (_saved(screen, index, (10, 10)), _saved(screen, index, (30, 30))) == (1, 2)


def test_no_to_the_prompt_changes_nothing(qtbot, screen, monkeypatch):
    _two_tracks(qtbot, screen)
    monkeypatch.setattr(screen, "_confirm", lambda *args: False)
    before, lineage = screen._canvas.mask.copy(), screen._lineage
    _edit_label_mode(screen, monkeypatch, "2")
    click_pixel(screen, (10, 10))
    np.testing.assert_array_equal(screen._canvas.mask, before)
    assert screen._lineage == lineage


def test_an_unmapped_object_on_the_label_moves_aside_in_this_frame(qtbot, screen, monkeypatch):
    enable_time(screen)
    _edit_label_mode(screen, monkeypatch, "11")
    click_pixel(screen, (10, 10))
    assert int(screen._canvas.mask[10, 10]) == 11
    assert int(screen._canvas.mask[30, 30]) not in (0, 7, 11)


def test_relabelling_reports_a_lineage_refusal(qtbot, screen, monkeypatch):
    _two_tracks(qtbot, screen)

    def refuse(*args):
        raise ValueError("synthetic lineage refusal")

    monkeypatch.setattr(engine, "_relabel_lineage_cell", refuse)
    before = screen._canvas.mask.copy()
    _edit_label_mode(screen, monkeypatch, "40")
    click_pixel(screen, (10, 10))
    assert "synthetic lineage refusal" in screen._status_label.text()
    np.testing.assert_array_equal(screen._canvas.mask, before)
    assert screen._relabel_time_object(0, 0, 5) is False
    assert screen._relabel_time_object(10, 10, 1) is True


def test_the_engine_moves_pixels_after_their_cells_and_clears_blocked_labels():
    mask = np.zeros((6, 6), np.uint16)
    mask[0:2, 0:2], mask[3:5, 3:5], mask[0:2, 4:6] = 1, 2, 3
    moved = engine._follow_lineage_labels(mask, {"1": "a", "2": "b"}, {"3": "a", "2": "b"})
    assert int(moved[0, 0]) == 3 and int(moved[3, 3]) == 2
    assert int(moved[0, 4]) not in (0, 1, 2, 3)
    assert engine._follow_lineage_labels(mask, {"1": "a"}, {"1": "a"}) is None


def _paint(screen, label):
    screen._label_value.setText(str(label))
    screen._paint_label_action.setChecked(True)
    assert screen._paint_label_enabled
    assert f"label {label} in each frame" in screen._status_label.text()


def test_the_time_button_offers_painting_from_its_menu(screen):
    menu = screen._btn_label_time.split_menu()
    assert menu.objectName() == "MakeMasksLabelTimeMenu"
    assert screen._paint_label_action in menu.actions()
    assert screen._paint_label_action.isCheckable()


def test_painting_forward_and_back_gives_each_hovered_object_the_label(qtbot, screen):
    enable_time(screen)
    _paint(screen, 5)
    wheel(screen, 2, point=(30, 30))
    wait_idle(qtbot, screen)
    assert screen._current_index == 2 and int(screen._canvas.mask[30, 30]) == 5
    cell = screen._paint_cell
    frames = {Path(name).name: mapping for name, mapping in screen._lineage["frames"].items()}
    assert [frames[f"frame_{index}.tif"].get("5") for index in range(3)] == [cell] * 3
    wheel(screen, -2, point=(10, 10))
    wait_idle(qtbot, screen)
    assert screen._current_index == 0 and int(screen._canvas.mask[10, 10]) == 5
    assert int(screen._canvas.mask[30, 30]) not in (0, 5)
    assert screen._lineage["frames"][screen._lineage_frame()]["5"] == cell
    _written(qtbot, screen)
    assert _saved(screen, 1, (10, 10)) == 5 and _saved(screen, 1, (30, 30)) not in (0, 5)
    screen._paint_label_action.setChecked(False)
    assert not screen._paint_label_enabled


def test_painting_a_label_two_cells_carry_moves_the_other_cell_everywhere(qtbot, screen):
    first, second = _two_tracks(qtbot, screen)
    state, stray = engine.new_lineage_cell(screen._lineage, "stray", 2, screen._lineage_frame())
    screen._lineage = state
    _paint(screen, 2)
    wheel(screen, 1, point=(10, 10))
    wait_idle(qtbot, screen)
    assert screen._current_index == 1
    assert screen._paint_cell == second
    assert engine._cells_with_label(screen._lineage, 2) == [second]
    assert screen._lineage["cells"][stray]["label"] not in (1, 2)
    assert int(screen._canvas.mask[10, 10]) == 2
    assert int(screen._canvas.mask[30, 30]) not in (0, 1, 2)


def test_painting_onto_an_unmapped_label_and_undo_restore_the_frame(qtbot, screen):
    enable_time(screen)
    _paint(screen, 11)
    before = screen._canvas.mask.copy()
    assert screen._canvas.assign_label_at((10, 10), 11)
    assert int(screen._canvas.mask[10, 10]) == 11
    assert int(screen._canvas.mask[30, 30]) not in (0, 11)
    screen._on_undo()
    np.testing.assert_array_equal(screen._canvas.mask, before)


def test_the_label_dialog_falls_back_to_the_old_label_when_none_is_free(screen, monkeypatch):
    enable_time(screen)

    def none_free(*args):
        raise ValueError("No unused 32-bit label remains.")

    monkeypatch.setattr(engine, "_unused_movie_label", none_free)
    monkeypatch.setattr(engine, "MAX_MASK_LABEL", 11)
    shown = _edit_label_mode(screen, monkeypatch, "7")
    click_pixel(screen, (10, 10))
    assert shown == ["7"]


def test_the_engine_label_helpers_refuse_what_they_cannot_do(tmp_path, monkeypatch):
    frame, other = str(tmp_path / "a.tif"), str(tmp_path / "b.tif")
    state, first = engine.new_lineage_cell(None, "first", 1, frame)
    state, second = engine.new_lineage_cell(state, "second", 5, frame)
    state = engine.assign_lineage_label(state, frame, 1, first)
    state = engine.assign_lineage_label(state, other, 1, first)
    state = engine.assign_lineage_label(state, other, 2, second)
    state = engine.assign_lineage_label(state, frame, 2, second)
    assert engine._cells_with_label(state, 2) == [second]
    state["cells"][second]["label"] = 5
    assert engine._cells_with_label(state, 2) == [second]
    with pytest.raises(ValueError, match="existing lineage cell"):
        engine._relabel_lineage_cell(state, "missing", 3)
    with pytest.raises(ValueError, match="nonzero uint32"):
        engine._relabel_lineage_cell(state, first, 0)
    with pytest.raises(ValueError, match="already belongs to another lineage cell"):
        engine._relabel_lineage_cell(state, first, 2)
    monkeypatch.setattr(engine, "MAX_MASK_LABEL", 2)
    with pytest.raises(ValueError, match="No unused 32-bit label"):
        engine._unused_movie_label(state)


def test_removing_a_division_keeps_the_first_daughter_own_division_type(tmp_path):
    frames = [str(tmp_path / f"t{index}.tif") for index in range(3)]
    state, mother = engine.new_lineage_cell(None, "m", 1, frames[0])
    state = engine.assign_lineage_label(state, frames[0], 1, mother)
    state, (first, second) = engine.divide_lineage_cell(state, mother, frames[1], ["a", "b"], [2, 3])
    state, _ = engine.divide_lineage_cell(state, first, frames[2], ["c", "d"], [4, 5])
    state["cells"][first]["division_type"] = "mitosis"
    state, kept, alone = engine._remove_lineage_division(state, mother)
    assert (kept, alone) == (first, second)
    assert state["cells"][mother]["division_type"] == "mitosis"
    assert len(state["cells"][mother]["children"]) == 2


def test_the_label_dialog_proposes_the_lowest_free_label_past_the_top(screen, monkeypatch):
    enable_time(screen)
    monkeypatch.setattr(engine, "MAX_MASK_LABEL", 11)
    shown = _edit_label_mode(screen, monkeypatch, "7")
    click_pixel(screen, (10, 10))
    assert shown == ["1"]
