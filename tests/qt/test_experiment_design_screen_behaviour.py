"""The experiment design screen: the drag gesture, the condition rows, the status line, export.

The plate map is a grid the user drags across (press anchors, move previews,
release commits; the pressed well keeps the grab, so the well under the pointer
is found by asking), the condition table skips rows that are half typed, the
status line names the assignment or the reason it failed (and keeps the last
line for a failure with no words), and an export names what it wrote, refuses
an empty plate, and reports a worker failure in the error style.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt              # noqa: E402
from PySide6.QtGui import QMouseEvent                               # noqa: E402
from PySide6.QtTest import QTest                                    # noqa: E402
from PySide6.QtWidgets import QApplication, QTableWidgetItem        # noqa: E402

from spacr.qt.screens import experiment_design as ED                # noqa: E402
from spacr.qt.widgets.plate_layout import ROLE_TREATMENT, Condition  # noqa: E402

pytestmark = pytest.mark.qt

DEFAULT = [("negative", 6, "negative_control"),
           ("positive", 6, "positive_control"),
           ("treatment_a", 12, "treatment")]
NOT_WHOLE_ROWS = (
    " does not occupy whole rows or whole columns, and annotate_conditions "
    "can only address wells by row or column id. Use the exported well table "
    "(plate_map.csv) and join on (plateID, rowID, columnID) instead -- it "
    "carries the layout exactly.")
TOO_MANY = ("500 well(s) requested but only 96 usable on a 96-well plate. "
            "Reduce replicates, use a bigger plate, or split across plates.")
WROTE = ("Wrote plate_map.csv, plate_map.json, plate_map_settings.json to {}. "
         "plate_map.csv joins to a measurements table on "
         "(plateID, rowID, columnID).")


@pytest.fixture
def make(qtbot):
    """Build a shown design screen (unthreaded unless asked)."""
    def _build(threaded=False):
        widget = ED.ExperimentDesignScreen(threaded=threaded)
        qtbot.addWidget(widget)
        widget.resize(1200, 800)
        widget.show()
        QApplication.processEvents()
        return widget
    return _build


def _well(screen, row, column):
    return next(label for label in screen._well_labels
                if (label.row, label.column) == (row, column))


def _mouse(kind, widget, point=QPointF(2, 2), global_point=None,
           modifiers=Qt.NoModifier):
    return QMouseEvent(kind, QPointF(point),
                       global_point if global_point is not None
                       else widget.mapToGlobal(QPointF(point)),
                       Qt.LeftButton, Qt.LeftButton, modifiers)


def _block(rows, columns):
    return {(r, c) for r in rows for c in columns}


def _chosen_flags(screen):
    return {(label.row, label.column) for label in screen._well_labels
            if label.property("spacrWellChosen") == "true"}


def test_the_well_drag_selects_the_rectangle_from_its_anchor(make, subtests):
    """Press anchors one well, a move previews the block, release keeps it."""
    with subtests.test(msg="a real mouse click selects one square"):
        screen = make()
        well = _well(screen, 2, 3)
        QTest.mousePress(well, Qt.LeftButton, Qt.NoModifier, well.rect().center())
        QTest.mouseRelease(well, Qt.LeftButton, Qt.NoModifier,
                           well.rect().center())
        assert screen.selected_wells() == {(2, 3)}
        assert screen.selected_well_names() == ["B03"]
        assert screen._well_anchor is None

    with subtests.test(msg="a real mouse drag selects the block"):
        screen = make()
        start, end = _well(screen, 2, 3), _well(screen, 4, 5)
        towards = start.mapFromGlobal(end.mapToGlobal(end.rect().center()))
        QTest.mousePress(start, Qt.LeftButton, Qt.NoModifier,
                         start.rect().center())
        QTest.mouseMove(start, towards)
        QApplication.processEvents()
        QTest.mouseRelease(start, Qt.LeftButton, Qt.NoModifier, towards)
        assert screen.selected_wells() == _block((2, 3, 4), (3, 4, 5))
        assert screen.selected_well_names() == [
            "B03", "B04", "B05", "C03", "C04", "C05", "D03", "D04", "D05"]

    with subtests.test(msg="press anchors, release ends the gesture only"):
        screen = make()
        label = _well(screen, 1, 1)
        label.mousePressEvent(_mouse(QEvent.MouseButtonPress, label))
        assert screen.selected_wells() == {(1, 1)}
        assert screen._well_anchor == (1, 1)
        assert screen._well_adding is False
        label.mouseReleaseEvent(_mouse(QEvent.MouseButtonRelease, label))
        assert screen._well_anchor is None
        assert screen._well_last is None
        assert screen.selected_wells() == {(1, 1)}

    with subtests.test(msg="the grabbed well asks which well the pointer is over"):
        screen = make()
        start, end = _well(screen, 3, 2), _well(screen, 1, 1)
        start.mousePressEvent(_mouse(QEvent.MouseButtonPress, start))
        over_end = end.mapToGlobal(QPointF(2, 2))
        start.mouseMoveEvent(_mouse(QEvent.MouseMove, start,
                                    start.mapFromGlobal(over_end), over_end))
        assert screen.selected_wells() == _block((1, 2, 3), (1, 2))
        assert _chosen_flags(screen) == _block((1, 2, 3), (1, 2))
        assert screen._well_last == (1, 1)
        # A second move over the same well changes nothing.
        screen._select_wells({(8, 12)})
        screen.drag_wells_to(over_end.toPoint())
        assert screen.selected_wells() == {(8, 12)}
        start.mouseReleaseEvent(_mouse(QEvent.MouseButtonRelease, start))
        assert screen._well_last is None

    with subtests.test(msg="ctrl adds the block to the wells already chosen"):
        screen = make()
        screen.begin_well_drag(8, 12)
        screen.finish_well_drag()
        screen.begin_well_drag(1, 1, Qt.ControlModifier)
        assert screen._well_adding is True
        screen.drag_wells_to(_well(screen, 2, 2).mapToGlobal(QPoint(2, 2)))
        assert screen.selected_wells() == _block((1, 2), (1, 2)) | {(8, 12)}

    with subtests.test(msg="a move with no press behind it selects nothing"):
        screen = make()
        screen.begin_well_drag(5, 5)
        screen.finish_well_drag()
        screen.drag_wells_to(_well(screen, 7, 9).mapToGlobal(QPoint(2, 2)))
        assert screen.selected_wells() == {(5, 5)}
        fresh = make()
        fresh.drag_wells_to(_well(fresh, 2, 2).mapToGlobal(QPoint(2, 2)))
        assert fresh.selected_wells() == set()

    with subtests.test(msg="a move off the plate keeps the preview"):
        screen = make()
        screen.begin_well_drag(4, 4)
        screen.drag_wells_to(QPoint(-5000, -5000))
        assert screen.selected_wells() == {(4, 4)}
        assert getattr(screen, "_well_last", None) is None


def test_conditions_skip_the_rows_that_are_not_usable_yet(make, subtests):
    """Zero, negative, non-numeric counts and blank names are left off the map."""
    cases = [
        # (case id, (row, column, text) edits, expected names)
        ("zero replicates", [(0, 1, "0")], ["positive", "treatment_a"]),
        ("negative replicates", [(1, 1, "-4")], ["negative", "treatment_a"]),
        ("both at once", [(0, 1, "0"), (1, 1, "-4")], ["treatment_a"]),
        ("a count that is not a number", [(0, 1, "six")],
         ["positive", "treatment_a"]),
        ("a blank name", [(2, 0, "   ")], ["negative", "positive"]),
        ("a name is stripped", [(2, 0, "  dmso ")],
         ["negative", "positive", "dmso"]),
    ]
    for case_id, edits, names in cases:
        with subtests.test(msg=case_id):
            screen = make()
            assert [(c.name, c.replicates, c.role)
                    for c in screen.conditions()] == DEFAULT
            for row, column, text in edits:
                screen._table.setItem(row, column, QTableWidgetItem(text))
            assert [c.name for c in screen.conditions()] == names
            assert [c.name for c in screen.design().conditions] == names
            assert screen._table.rowCount() == 3

    with subtests.test(msg="removing the selected row redraws the plate"):
        screen = make()
        screen._table.selectRow(0)
        screen._remove_row()
        assert [(c.name, c.replicates, c.role)
                for c in screen.conditions()] == DEFAULT[1:]
        assert screen.status_text() == (
            "18 of 96 usable wells assigned. 'positive'" + NOT_WHOLE_ROWS)
        assert screen._table.rowCount() == 2

    with subtests.test(msg="adding a row gives it a usable default"):
        screen = make()
        screen._add_row()
        assert [(c.name, c.replicates, c.role) for c in screen.conditions()] \
            == DEFAULT + [("condition_4", 3, ROLE_TREATMENT)]
        assert screen.status_text() == (
            "27 of 96 usable wells assigned. 'negative'" + NOT_WHOLE_ROWS)


def test_the_status_line_reports_the_assignment_or_why_it_failed(make,
                                                                 monkeypatch,
                                                                 subtests):
    """A message failure is shown as an error; a wordless one keeps the last line."""
    screen = make()

    with subtests.test(msg="the default plate"):
        assert screen.status_text() == (
            "24 of 96 usable wells assigned. 'negative'" + NOT_WHOLE_ROWS)
        assert screen._status.property("spacrError") == "false"

    with subtests.test(msg="a design that does not fit"):
        screen._set_conditions([Condition("treatment_a", 500, ROLE_TREATMENT)])
        screen.refresh()
        assert screen.status_text() == TOO_MANY
        assert screen._status.property("spacrError") == "true"
        assert {label.property("spacrWellRole")
                for label in screen._well_labels} == {"empty"}

    with subtests.test(msg="a failure with no message leaves the last line"):
        def wordless(_design):
            raise ValueError("")

        with monkeypatch.context() as patch:
            patch.setattr(ED, "assign_wells", wordless)
            screen.refresh()
        assert screen.status_text() == TOO_MANY
        assert len(screen._well_labels) == 96
        assert {label.toolTip() for label in screen._well_labels} == {
            "unassigned"}

    with subtests.test(msg="a design that fits again recovers"):
        screen._set_conditions([Condition("treatment_a", 4, ROLE_TREATMENT)])
        screen.refresh()
        assert screen.status_text() == (
            "4 of 96 usable wells assigned. This layout can also be exported "
            "as treatment_plate_metadata.")
        assert screen._status.property("spacrError") == "false"
        assert sum(label.property("spacrWellRole") == ROLE_TREATMENT
                   for label in screen._well_labels) == 4

    with subtests.test(msg="a label with no style still takes text and flag"):
        screen._status.style = lambda: None
        try:
            screen._set_status("too many replicates", is_error=True)
        finally:
            del screen._status.style
        assert screen.status_text() == "too many replicates"
        assert screen._status.property("spacrError") == "true"
        screen._set_status("all good")
        assert screen.status_text() == "all good"
        assert screen._status.property("spacrError") == "false"


def test_export_names_what_it_wrote_or_says_why_not(make, tmp_path,
                                                    monkeypatch, qtbot,
                                                    subtests):
    """The folder chosen gets the three files; failures speak in the error style."""
    for threaded in (False, True):
        with subtests.test(msg=f"browsing to a folder writes (threaded={threaded})"):
            screen = make(threaded=threaded)
            folder = tmp_path / f"out_{threaded}"
            folder.mkdir()
            monkeypatch.setattr(ED.QFileDialog, "getExistingDirectory",
                                staticmethod(lambda *a, **k: str(folder)))
            screen._on_export()
            qtbot.waitUntil(lambda: not screen.is_busy(), timeout=5000)
            QApplication.processEvents()
            qtbot.waitUntil(lambda: screen.status_text().startswith("Wrote"),
                            timeout=5000)
            assert sorted(p.name for p in folder.iterdir()) == [
                "plate_map.csv", "plate_map.json", "plate_map_settings.json"]
            assert screen.status_text() == WROTE.format(folder)
            assert screen._status.property("spacrError") == "false"

    with subtests.test(msg="an export that wrote nothing changes nothing"):
        screen = make()
        screen._set_status("untouched")
        for empty in ({}, None):
            screen._on_exported(empty)
            assert screen.status_text() == "untouched"

    with subtests.test(msg="a failed export is an error with its message"):
        screen = make()
        screen._on_job_failed("permission denied on /mnt/plate")
        assert screen.status_text() == (
            "Export failed: permission denied on /mnt/plate")
        assert screen._status.property("spacrError") == "true"

    with subtests.test(msg="a plate with no conditions refuses out loud"):
        screen = make()
        screen._table.setRowCount(0)
        target = tmp_path / "refused"
        target.mkdir()
        assert screen.export_to(str(target)) is False
        assert screen.status_text() == (
            "Nothing to export: the plate has no conditions.")
        assert screen._status.property("spacrError") == "true"
        assert list(target.iterdir()) == []
