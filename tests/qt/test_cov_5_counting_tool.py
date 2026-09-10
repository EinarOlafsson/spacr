"""The count is only changed by the two buttons that mean something.

A manual count is thousands of clicks and the tally is the result. Two
things have to hold: a mouse button that is not left or right must leave the
count alone, and the export buttons must go through the same writer the
dialog-free seam uses -- so that what is saved is what is on screen, and a
cancelled dialog saves nothing at all.
"""
from __future__ import annotations

import csv
import os

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from spacr.layers import LayerStack, Spacing
from spacr.qt import counting_tool as ct
from spacr.qt import layer_viewer as lv


def _panel(qtbot):
    stack = LayerStack()
    stack.add_image(np.zeros((64, 64), np.uint16), name="image",
                    spacing=Spacing.isotropic(2, 1.0, units="px"))
    canvas = lv.LayerCanvas(stack)
    qtbot.addWidget(canvas)
    canvas.resize(200, 200)
    canvas._ensure_canvas()
    panel = ct.CountingPanel(canvas)
    qtbot.addWidget(panel)
    return canvas, panel


def test_a_side_button_neither_counts_nor_uncounts(qtbot, qt_theme_applied):
    """A mouse button that is not left or right leaves the tally alone.

    Thumb buttons are bound to back/forward on most mice and get pressed by
    accident over the image. Treating an unrecognised button as a click
    would add a marker the counter never meant, and the tally is the
    published number.
    """
    canvas, panel = _panel(qtbot)
    panel.start_counting()

    QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, QPoint(60, 45))
    assert panel.session.total == 1

    QTest.mouseClick(canvas, Qt.BackButton, Qt.NoModifier, QPoint(90, 90))
    QTest.mouseClick(canvas, Qt.BackButton, Qt.NoModifier, QPoint(60, 45))
    assert panel.session.total == 1


def test_exporting_the_clicks_writes_one_row_per_marker(
        qtbot, qt_theme_applied, tmp_path, monkeypatch):
    """The clicks button saves the markers, at the path the dialog returned.

    The dialog is the only difference between this and the tested seam, so
    what is pinned here is that the button reaches the seam at all and that
    ``summary=False`` is what it asks for -- a button wired to the tally
    would write four rows where the analysis needs four hundred.
    """
    canvas, panel = _panel(qtbot)
    panel.start_counting()
    for x in (30, 60, 90):
        QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, QPoint(x, 45))

    target = tmp_path / "clicks.csv"
    asked = {}

    def _dialog(parent, caption, suggested, filter_):
        asked["suggested"] = suggested
        return str(target), filter_

    monkeypatch.setattr(ct.QFileDialog, "getSaveFileName", staticmethod(_dialog))

    written = panel.export_points()

    assert written == str(target)
    assert os.path.basename(asked["suggested"]) == "counts.csv"
    rows = list(csv.DictReader(target.read_text().splitlines()))
    assert len(rows) == 3


def test_exporting_the_tally_writes_one_row_per_class(
        qtbot, qt_theme_applied, tmp_path, monkeypatch):
    """The tally button saves per-class counts under the summary name.

    The suggested filename is what tells the two exports apart in a folder;
    both offering ``counts.csv`` is how a summary silently replaces the
    clicks it was meant to sit beside.
    """
    canvas, panel = _panel(qtbot)
    panel.start_counting()
    for x in (30, 60, 90):
        QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, QPoint(x, 45))

    target = tmp_path / "tally.csv"
    asked = {}

    def _dialog(parent, caption, suggested, filter_):
        asked["suggested"] = suggested
        return str(target), filter_

    monkeypatch.setattr(ct.QFileDialog, "getSaveFileName", staticmethod(_dialog))

    written = panel.export_summary()

    assert written == str(target)
    assert os.path.basename(asked["suggested"]) == "counts_summary.csv"
    rows = list(csv.DictReader(target.read_text().splitlines()))
    assert len(rows) == len(panel.session.classes)


def test_a_cancelled_save_writes_nothing(
        qtbot, qt_theme_applied, tmp_path, monkeypatch):
    """Dismissing the dialog leaves no file and reports no path.

    An empty path from the dialog is the user saying no; passing it on would
    make the writer either raise or create a file called nothing in the
    working directory.
    """
    canvas, panel = _panel(qtbot)
    panel.start_counting()
    QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, QPoint(60, 45))

    monkeypatch.setattr(ct.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    monkeypatch.chdir(tmp_path)

    assert panel.export_points() is None
    assert panel.export_summary() is None
    assert list(tmp_path.iterdir()) == []
