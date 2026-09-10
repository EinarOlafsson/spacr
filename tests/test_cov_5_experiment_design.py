"""Dragging a rectangle of wells, editing the condition table, exporting.

The plate map is a grid the user drags across, so the gesture is the feature:
press anchors, move previews, release commits — and the pressed widget keeps
the mouse grab, so the well under the pointer has to be found by asking rather
than by waiting to be entered. The rest is what the screen does with rows and
exports that are not usable yet.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt              # noqa: E402
from PySide6.QtGui import QMouseEvent                               # noqa: E402
from PySide6.QtWidgets import QTableWidgetItem, QWidget             # noqa: E402

from spacr.qt.screens import experiment_design as ED                # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture()
def screen(qtbot):
    widget = ED.ExperimentDesignScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.show()
    return widget


def _mouse(kind, widget, point=(2, 2), modifiers=Qt.NoModifier):
    return QMouseEvent(kind, QPointF(*point), widget.mapToGlobal(QPointF(*point)),
                       Qt.LeftButton, Qt.LeftButton, modifiers)


def _well(screen, row, column):
    return next(label for label in screen._well_labels
                if (label.row, label.column) == (row, column))


# ---------------------------------------------------------------------------
# The drag
# ---------------------------------------------------------------------------

def test_pressing_a_well_anchors_a_selection_on_it(screen):
    label = _well(screen, 1, 1)

    label.mousePressEvent(_mouse(QEvent.MouseButtonPress, label))

    assert screen.selected_wells() == {(1, 1)}
    assert screen._well_anchor == (1, 1)

    label.mouseReleaseEvent(_mouse(QEvent.MouseButtonRelease, label))
    assert screen._well_anchor is None
    assert screen.selected_wells() == {(1, 1)}, "release must not clear it"


def test_dragging_across_the_plate_previews_the_rectangle(screen):
    start, end = _well(screen, 1, 1), _well(screen, 3, 2)
    start.mousePressEvent(_mouse(QEvent.MouseButtonPress, start))

    # The pressed widget keeps the grab, so the move arrives at `start` with
    # a position over `end`.
    over_end = start.mapFromGlobal(end.mapToGlobal(QPointF(2, 2)))
    start.mouseMoveEvent(QMouseEvent(
        QEvent.MouseMove, QPointF(over_end),
        end.mapToGlobal(QPointF(2, 2)),
        Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))

    assert screen.selected_wells() == {
        (r, c) for r in (1, 2, 3) for c in (1, 2)}

    start.mouseReleaseEvent(_mouse(QEvent.MouseButtonRelease, start))
    assert screen._well_last is None


def test_a_move_with_no_press_selects_nothing(screen):
    """A pointer crossing the plate is not a gesture."""
    label = _well(screen, 2, 2)
    before = screen.selected_wells()

    screen.drag_wells_to(label.mapToGlobal(QPoint(2, 2)))

    assert screen.selected_wells() == before


def test_a_point_off_the_plate_names_no_well(screen):
    assert screen.well_at(QPoint(-5000, -5000)) is None


def test_a_well_with_no_screen_above_it_does_not_raise(qtbot):
    """A well built outside the screen must not throw from a click."""
    holder = QWidget()
    qtbot.addWidget(holder)
    orphan = ED._Well(0, 0, holder)

    orphan.mousePressEvent(_mouse(QEvent.MouseButtonPress, orphan))
    orphan.mouseMoveEvent(_mouse(QEvent.MouseMove, orphan))
    orphan.mouseReleaseEvent(_mouse(QEvent.MouseButtonRelease, orphan))

    assert orphan._screen() is None


# ---------------------------------------------------------------------------
# The condition table
# ---------------------------------------------------------------------------

def test_removing_the_selected_rows_redraws_the_plate(screen):
    before = len(screen.conditions())
    screen._table.selectRow(0)

    screen._remove_row()

    assert len(screen.conditions()) == before - 1
    assert "negative" not in [c.name for c in screen.conditions()]
    assert "usable wells assigned" in screen.status_text()


def test_a_row_asking_for_no_replicates_is_not_a_condition(screen):
    """Zero replicates is a row being typed, not a condition to lay out."""
    screen._table.setItem(0, 1, QTableWidgetItem("0"))

    names = [c.name for c in screen.conditions()]

    assert "negative" not in names
    assert "positive" in names


def test_a_row_whose_count_is_not_a_number_is_skipped(screen):
    screen._table.setItem(0, 1, QTableWidgetItem("six"))

    assert "negative" not in [c.name for c in screen.conditions()]


def test_adding_a_row_gives_it_a_usable_default(screen):
    before = len(screen.conditions())

    screen._add_row()

    assert len(screen.conditions()) == before + 1
    assert screen.conditions()[-1].replicates == 3


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def test_browsing_to_a_folder_writes_the_plate_map(screen, tmp_path,
                                                    monkeypatch):
    monkeypatch.setattr(ED.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp_path)))

    screen._on_export()

    assert (tmp_path / "plate_map.csv").is_file()
    assert "plate_map.csv joins to a measurements table" in screen.status_text()


def test_cancelling_the_folder_dialog_writes_nothing(screen, tmp_path,
                                                      monkeypatch):
    monkeypatch.setattr(ED.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: ""))

    screen._on_export()

    assert list(tmp_path.iterdir()) == []


def test_an_export_that_wrote_nothing_leaves_the_status_alone(screen):
    """A job that produced no paths has nothing to name."""
    screen._set_status("untouched")

    screen._on_exported({})

    assert screen.status_text() == "untouched"


def test_an_export_that_failed_says_so_in_the_error_colour(screen):
    """Silence after pressing Export reads as an export that worked."""
    screen._on_job_failed("permission denied on /mnt/plate")

    assert "Export failed: permission denied on /mnt/plate" in (
        screen.status_text())


def test_exporting_a_plate_with_no_conditions_refuses_out_loud(screen):
    screen._table.setRowCount(0)

    assert screen.export_to("/tmp") is False
    assert "no conditions" in screen.status_text()
