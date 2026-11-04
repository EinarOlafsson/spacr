"""Time mode: the division type names the daughters, is remembered, and a division can be removed.

Real Make Masks screen on the synthetic four-frame movie of
``tests/qt/test_708_make_masks_label_navigation``.
"""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QDialog, QLineEdit

from spacr.qt.screens import make_masks as mm

from .test_708_make_masks_label_navigation import (
    click_pixel, enable_time, screen, select, wait_idle, wheel)

__all__ = ["screen"]


def _divide(screen, monkeypatch, kind=None):
    seen = {}

    def answer(dialog):
        combo = dialog.findChild(QComboBox, "MakeMasksDivisionType")
        seen["default"] = combo.currentData()
        if kind is not None:
            combo.setCurrentIndex(combo.findData(kind))
        seen["fields"] = [field.text() for field in dialog.findChildren(QLineEdit)]
        return QDialog.Accepted

    monkeypatch.setattr(QDialog, "exec", answer)
    screen._divide_lineage_cell()
    return seen


def _founder(screen):
    enable_time(screen)
    click_pixel(screen, (10, 10))
    return screen._selected_cell


def test_budding_keeps_the_mother_name_and_mitosis_names_two_new_daughters(screen, monkeypatch):
    mother = _founder(screen)
    budding = _divide(screen, monkeypatch)
    assert budding["default"] == "budding"
    assert budding["fields"] == ["1", "1", "1a", "2"]
    cells = screen._lineage["cells"]
    assert cells[mother]["division_type"] == "budding"
    assert [cells[child]["name"] for child in cells[mother]["children"]] == ["1", "1a"]
    screen._on_undo()
    assert not screen._lineage["cells"][mother]["children"]
    mitosis = _divide(screen, monkeypatch, "mitosis")
    assert mitosis["fields"] == ["1a", "2", "1b", "3"]
    cells = screen._lineage["cells"]
    assert cells[mother]["division_type"] == "mitosis"
    assert [cells[child]["name"] for child in cells[mother]["children"]] == ["1a", "1b"]


def test_the_division_type_is_remembered_for_the_movie(qtbot, screen, monkeypatch):
    _founder(screen)
    _divide(screen, monkeypatch, "fission")
    screen._add_lineage_cell()
    assert _divide(screen, monkeypatch)["default"] == "fission"
    assert screen._division_type() == "fission"
    assert screen._save_mask() is not None
    fresh = mm.MakeMasksScreen()
    qtbot.addWidget(fresh)
    assert fresh._open_folder(screen._folder)
    qtbot.waitUntil(lambda: not fresh._loading and fresh._reader_worker is None
                    and fresh._load_worker is None)
    assert fresh._division_type() == "fission"
    fresh.close()


def test_removing_a_division_restores_the_mother_track_with_undo(qtbot, screen, monkeypatch):
    mother = _founder(screen)
    wheel(screen, 1, point=(10, 10))
    wait_idle(qtbot, screen)
    _divide(screen, monkeypatch, "mitosis")
    first, second = screen._lineage["cells"][mother]["children"]
    click_pixel(screen, (10, 10))
    select(screen, second)
    click_pixel(screen, (30, 30))
    frame = screen._lineage_frame()
    assert screen._lineage["frames"][frame] == {"2": first, "3": second}
    select(screen, first)
    screen._remove_lineage_division()
    cells = screen._lineage["cells"]
    assert first not in cells and not cells[mother]["children"]
    assert cells[second]["parent_uuid"] is None
    assert "division_type" not in cells[mother]
    assert screen._lineage["frames"][frame] == {"2": mother, "3": second}
    assert "is now a separate cell" in screen._status_label.text()
    assert screen._selected_cell == mother
    screen._on_undo()
    cells = screen._lineage["cells"]
    assert cells[mother]["children"] == [first, second]
    assert screen._lineage["frames"][frame] == {"2": first, "3": second}


def test_removing_needs_a_division(screen):
    screen._remove_lineage_division()
    _founder(screen)
    lineage = screen._lineage
    screen._remove_lineage_division()
    assert "Select a cell that divided" in screen._status_label.text()
    assert screen._lineage == lineage
