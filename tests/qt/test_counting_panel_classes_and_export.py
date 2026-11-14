"""The counting panel: which clicks and keys count, how classes are listed, what is exported.

Only left (count) and right (uncount) change the tally; only digits choose a
class; layer housekeeping never re-emits the tally; out-of-range and cleared
rows are not selections; and the two exports write markers or per-class rows
at the dialog's path, writing nothing when it is cancelled.
"""
from __future__ import annotations

import csv
import os

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest

from spacr.layers import LayerStack, Spacing
from spacr.qt import counting_tool as ct
from spacr.qt import layer_viewer as lv

pytestmark = pytest.mark.qt


def _stack():
    stack = LayerStack()
    stack.add_image(np.zeros((64, 64), np.uint16), name="image",
                    spacing=Spacing.isotropic(2, 1.0, units="px"))
    return stack


def _panel(qtbot, stack=None, **kwargs):
    canvas = lv.LayerCanvas(stack if stack is not None else _stack())
    qtbot.addWidget(canvas)
    canvas.resize(200, 200)
    canvas._ensure_canvas()
    panel = ct.CountingPanel(canvas, **kwargs)
    qtbot.addWidget(panel)
    return canvas, panel


def _key_event(key, text):
    return QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier, text)


def _click(canvas, button, x, y=45):
    QTest.mouseClick(canvas, button, Qt.NoModifier, QPoint(x, y))


def _rows(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_counting_panel_counts_left_uncounts_right_and_exports(
        subtests, qtbot, qt_theme_applied, tmp_path, monkeypatch):
    """Side buttons change nothing; exports go through the dialog and the writer."""
    with subtests.test(msg="left counts, side buttons do not, right uncounts"):
        canvas, panel = _panel(qtbot)
        panel.start_counting()
        _click(canvas, Qt.LeftButton, 60)
        assert panel.session.counts() == {"infected": 1, "uninfected": 0}
        _click(canvas, Qt.BackButton, 90, 90)
        _click(canvas, Qt.BackButton, 60)
        _click(canvas, Qt.ForwardButton, 60)
        assert panel.session.total == 1
        _click(canvas, Qt.RightButton, 60)
        assert panel.session.total == 0
        assert panel.total.text() == "nothing counted yet"

    def _export_with(answer, asked):
        def dialog(parent, caption, suggested, filter_):
            asked.append((caption, os.path.basename(suggested), filter_))
            return answer, filter_
        monkeypatch.setattr(ct.QFileDialog, "getSaveFileName",
                            staticmethod(dialog))

    for case_id, method, name, expected in [
            ("clicks export: one row per marker", "export_points",
             "counts.csv", ["infected"] * 3),
            ("tally export: one row per class", "export_summary",
             "counts_summary.csv", ["infected", "uninfected"])]:
        with subtests.test(msg=case_id):
            canvas, panel = _panel(qtbot)
            panel.start_counting()
            for x in (30, 60, 90):
                _click(canvas, Qt.LeftButton, x)
            exported = []
            panel.exported.connect(exported.append)
            target = tmp_path / f"{method}.csv"
            asked = []
            _export_with(str(target), asked)
            assert getattr(panel, method)() == str(target)
            assert asked == [("Save the count", name, "CSV (*.csv)")]
            assert exported == [str(target)]
            rows = _rows(target)
            assert [row["class"] for row in rows] == expected
            if method == "export_summary":
                assert [(r["count"], r["fraction"], r["total"])
                        for r in rows] == [("3", "1.0", "3"),
                                           ("0", "0.0", "3")]
            else:
                assert {r["units"] for r in rows} == {"px"}
                xs = [float(r["x"]) for r in rows]
                assert xs == sorted(xs) and len(set(xs)) == 3

    with subtests.test(msg="a cancelled dialog writes nothing"):
        canvas, panel = _panel(qtbot)
        panel.start_counting()
        _click(canvas, Qt.LeftButton, 60)
        empty = tmp_path / "cancel"
        empty.mkdir()
        monkeypatch.chdir(empty)
        asked = []
        _export_with("", asked)
        assert panel.export_points() is None
        assert panel.export_summary() is None
        assert [a[1] for a in asked] == ["counts.csv", "counts_summary.csv"]
        assert list(empty.iterdir()) == []


def test_counting_tool_claims_only_digit_keys(subtests, qtbot,
                                              qt_theme_applied):
    """A digit selects its class; letters, arrows and modifiers pass through."""
    canvas, panel = _panel(qtbot)
    tool = panel.start_counting()
    cases = [
        ("digit 2 selects the second class", Qt.Key_2, "2", True,
         "uninfected"),
        ("a letter", Qt.Key_A, "a", False, "uninfected"),
        ("an arrow with no text", Qt.Key_Left, "", False, "uninfected"),
        ("a modifier", Qt.Key_Shift, "", False, "uninfected"),
        ("digit 1 selects the first class", Qt.Key_1, "1", True, "infected"),
    ]
    for case_id, key, text, claimed, active in cases:
        with subtests.test(msg=case_id):
            assert tool.key(canvas, _key_event(key, text)) is claimed
            assert panel.session.active == active

    with subtests.test(msg="through the real widget"):
        QTest.keyClick(canvas, Qt.Key_2)
        assert panel.session.active == "uninfected"
        QTest.keyClick(canvas, Qt.Key_A)
        assert panel.session.active == "uninfected"
        assert panel.session.total == 0
        QTest.keyClick(canvas, Qt.Key_1)
        assert panel.session.active == "infected"


def test_counting_panel_class_list_follows_only_real_classes(
        subtests, qtbot, qt_theme_applied):
    """Housekeeping, unbacked rows and cleared lists never move the count or the class."""
    with subtests.test(msg="rename, select and move do not re-emit the tally"):
        stack = _stack()
        _canvas, panel = _panel(qtbot, stack)
        seen = []
        panel.counts_changed.connect(seen.append)
        image = stack["image"]
        assert stack.rename(image, "field 3") == "field 3"
        stack.select(image)
        stack.move(image, len(list(stack)) - 1)
        assert seen == []
        panel.session.add({"y": 10.0, "x": 10.0})
        assert seen == [{"infected": 1, "uninfected": 0}]
        assert panel.total.text() == panel.session.describe()

    with subtests.test(msg="a row with no class behind it is not a selection"):
        _canvas, panel = _panel(qtbot)
        assert panel.session.class_names == ("infected", "uninfected")
        assert panel.class_list.count() == 2
        assert panel.session.active == "infected"
        panel.class_list.addItem("a row nothing backs")
        panel.class_list.setCurrentRow(2)
        assert panel.class_list.currentRow() == 2
        assert panel.session.active == "infected"
        panel.class_list.setCurrentRow(1)
        assert panel.session.active == "uninfected"
        panel.session.add({"y": 12.0, "x": 12.0})
        assert panel.session.counts() == {"infected": 0, "uninfected": 1}

    with subtests.test(msg="row -1 from a cleared list is not a selection"):
        _canvas, panel = _panel(qtbot)
        panel.session.active = "uninfected"
        panel.refresh()
        assert panel.class_list.currentRow() == 1
        panel.class_list.setCurrentRow(-1)
        assert panel.class_list.currentRow() == -1
        assert panel.session.active == "uninfected"
        panel.session.add({"y": 20.0, "x": 20.0})
        panel.refresh()
        assert panel.session.active == "uninfected"
        assert panel.class_list.currentRow() == 1
        assert panel.session.counts() == {"infected": 0, "uninfected": 1}

    with subtests.test(msg="no classes yet: an empty tally with no highlight"):
        _canvas, panel = _panel(qtbot, classes=[])
        assert panel.session.class_names == ()
        assert panel.session.active == ""
        assert panel.class_list.count() == 0
        assert panel.class_list.currentRow() == -1
        assert panel.total.text() == "nothing counted yet"
        seen = []
        panel.counts_changed.connect(seen.append)
        assert panel.add_class("infected") == "infected"
        assert panel.session.active == "infected"
        assert panel.class_list.count() == 1
        assert panel.class_list.currentRow() == 0
        assert seen[-1] == {"infected": 0}
        panel.session.add({"y": 30.0, "x": 30.0})
        item = panel.class_list.item(panel.class_list.currentRow())
        assert item.data(Qt.UserRole) == "infected"
        assert "1" in item.text()
        assert panel.total.text() == "infected 1 (100%) · 1 total"

    with subtests.test(msg="an unnamed class is numbered"):
        _canvas, panel = _panel(qtbot, classes=[])
        assert panel.add_class(False) == "class 1"
        assert panel.add_class("   ") == "class 2"
        assert panel.add_class("mitotic") == "mitotic"
        assert panel.add_class(None) == "class 4"
        assert panel.session.class_names == (
            "class 1", "class 2", "mitotic", "class 4")
        assert [panel.class_list.item(i).data(Qt.UserRole)
                for i in range(panel.class_list.count())] == [
            "class 1", "class 2", "mitotic", "class 4"]
