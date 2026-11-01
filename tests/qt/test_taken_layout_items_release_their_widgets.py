"""A widget moved out of a layout and back follows its own size again.

``takeAt`` called from Python hands back a layout item nobody owns, so the
item stayed alive and kept its widget's size cache. The item ``addWidget``
then made for the widget never owned that cache: its height-for-width was
never cleared, and the Overview kept its status line 52 px tall after the
font went back to 13 px (test_action_font_override_restores_selected_style_
without_feedback failed whenever the status line was re-added while its old
item lived). ``spacr.qt.widgets.flow._take_every_item`` deletes the items it
takes.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QLabel, QSplitter, QVBoxLayout, QWidget

from spacr.qt.widgets.flow import _take_every_item

pytestmark = pytest.mark.qt


def _column(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    column = QVBoxLayout(host)
    splitter = QSplitter()
    label = QLabel("Each well keeps its source images and metadata. " * 12)
    label.setWordWrap(True)
    column.addWidget(splitter, 1)
    column.addSpacing(5)
    column.addWidget(label)
    host.resize(800, 600)
    host.show()
    QApplication.processEvents()
    return host, column, splitter, label


def test_a_label_taken_out_and_put_back_follows_its_font(qtbot):
    host, column, splitter, label = _column(qtbot)
    held = _take_every_item(column)
    assert [widget for widget, _nested in held] == [splitter, None, label]
    assert column.count() == 0
    column.addWidget(splitter, 1)
    column.addWidget(label)
    label.setStyleSheet("font-size: 52px;")
    QApplication.processEvents()
    tall = column.totalHeightForWidth(800)
    label.setStyleSheet("font-size: 13px;")
    QApplication.processEvents()
    short = column.totalHeightForWidth(800)
    assert short < tall
    assert short == column.totalHeightForWidth(800)


def test_a_nested_layout_is_handed_back_not_deleted(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    column = QVBoxLayout(host)
    row = QVBoxLayout()
    column.addLayout(row)
    inner = QLabel("inner")
    row.addWidget(inner)
    (widget, nested), = _take_every_item(column)
    assert widget is None and nested is row
    assert row.indexOf(inner) == 0
