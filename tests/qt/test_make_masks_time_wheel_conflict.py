"""Time mode: a wheel step over another cell's object while a cell is selected."""
from __future__ import annotations

import pytest

from spacr.qt import mask_engine as engine

from .test_708_make_masks_label_navigation import screen, wait_idle, wheel
from .test_make_masks_time_relabel_movie import _two_tracks

__all__ = ["screen"]


def _frame_cells(screen):
    mapping = screen._lineage["frames"][screen._lineage_frame()]
    return {int(local): identity for local, identity in mapping.items()}


def test_wheel_over_another_cells_object_swaps_the_two_identities_from_here(qtbot, screen):
    first, second = _two_tracks(qtbot, screen)
    screen._selected_cell = first
    wheel(screen, 1, point=(30, 30))
    wait_idle(qtbot, screen)
    assert "already labels another object" not in screen._status_label.text()
    assert screen._current_index == 1
    now = _frame_cells(screen)
    assert now[int(screen._canvas.mask[30, 30])] == first
    assert now[int(screen._canvas.mask[10, 10])] == second
    screen._on_undo()
    wait_idle(qtbot, screen)
    now = _frame_cells(screen)
    assert now[int(screen._canvas.mask[30, 30])] == second
    assert now[int(screen._canvas.mask[10, 10])] == first


def test_the_swap_helper_refuses_an_object_without_a_cell(qtbot, screen):
    first, _second = _two_tracks(qtbot, screen)
    with pytest.raises(ValueError, match="before they can swap"):
        engine._swap_frame_identities(screen._lineage, screen._lineage_frame(), 1, 99)
