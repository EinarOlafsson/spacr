"""Time mode: changing an object's label works, with the lineage kept in step.

Real Make Masks screen on the synthetic four-frame movie of
``tests/qt/test_708_make_masks_label_navigation``: two square objects,
pixel labels 7 and 11, in every frame.
"""
from __future__ import annotations

import numpy as np

from .test_708_make_masks_label_navigation import enable_time, screen, wait_idle, wheel

__all__ = ["screen"]


def _type_label(screen, text):
    screen._label_value.setText(text)
    screen._label_value.editingFinished.emit()


def test_typing_a_new_number_relabels_the_traced_object_after_scrolling(qtbot, screen):
    enable_time(screen)
    wheel(screen, 1, point=(10, 10))
    wait_idle(qtbot, screen)
    assert screen._current_index == 1
    identity, frame = screen._selected_cell, screen._lineage_frame()
    assert screen._lineage["frames"][frame] == {"1": identity}
    before = screen._canvas.mask.copy()
    _type_label(screen, "5")
    assert int(screen._canvas.mask[10, 10]) == 5
    assert int(screen._canvas.mask[30, 30]) == 11
    assert screen._lineage["frames"][frame] == {"5": identity}
    assert screen._selected_cell == identity
    assert screen._label_value.text() == "5"
    screen._on_undo()
    np.testing.assert_array_equal(screen._canvas.mask, before)
    assert screen._lineage["frames"][frame] == {"1": identity}


def test_a_number_held_by_another_object_is_reported_and_changes_nothing(qtbot, screen):
    enable_time(screen)
    wheel(screen, 1, point=(10, 10))
    wait_idle(qtbot, screen)
    before, lineage = screen._canvas.mask.copy(), screen._lineage
    _type_label(screen, "11")
    assert "Label 11 already belongs to another object in this frame" in \
        screen._status_label.text()
    np.testing.assert_array_equal(screen._canvas.mask, before)
    assert screen._lineage == lineage


def test_the_same_number_or_an_unplaced_cell_changes_no_pixels(qtbot, screen):
    enable_time(screen)
    before = screen._canvas.mask.copy()
    _type_label(screen, "1")
    np.testing.assert_array_equal(screen._canvas.mask, before)
    screen._add_lineage_cell()
    _type_label(screen, "9")
    np.testing.assert_array_equal(screen._canvas.mask, before)
    assert screen._relabel_selected_object(0) is False


def test_a_mapping_whose_object_is_gone_relabels_nothing(qtbot, screen, monkeypatch):
    enable_time(screen)
    wheel(screen, 1, point=(10, 10))
    wait_idle(qtbot, screen)
    monkeypatch.setattr(screen._canvas, "mask", np.zeros_like(screen._canvas.mask))
    assert screen._relabel_selected_object(5) is False
