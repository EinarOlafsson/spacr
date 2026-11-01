"""A ColumnAlignedRow's items die with their C++ objects, whoever else holds them.

The same defect FlowLayout had: ``addWidget`` handed the Python layout a
``QWidgetItem`` Qt created, PySide's layout glue gave that wrapper to the
outer box as soon as Python asked the box for an item, and once Qt deleted
the item (its button moved away) the wrapper stayed alive and registered at
the freed address for the next allocation to be handed.
"""
from __future__ import annotations

import gc

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QTableWidget, QVBoxLayout, QWidget
from shiboken6 import isValid

from spacr.qt.widgets.column_aligned_row import ColumnAlignedRow

pytestmark = pytest.mark.qt


def test_a_deleted_item_is_never_left_alive_behind_a_foreign_parent(qtbot):
    table = QTableWidget(2, 3)
    qtbot.addWidget(table)
    strip = QWidget()
    qtbot.addWidget(strip)
    outer = QHBoxLayout(strip)
    row = ColumnAlignedRow(table.horizontalHeader(), None)
    outer.addLayout(row)
    buttons = [QPushButton(f"Button {n}") for n in range(3)]
    for button in buttons:
        row.addWidget(button)
    assert outer.itemAt(0) is not None
    first = row.itemAt(0)
    assert first.widget() is buttons[0]

    elsewhere = QWidget()
    qtbot.addWidget(elsewhere)
    QVBoxLayout(elsewhere).addWidget(buttons[0])
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    gc.collect()

    assert row.count() == 2
    assert not isValid(first)
    assert [row.itemAt(index).widget() for index in range(row.count())] == buttons[1:]
    assert all(isValid(row.itemAt(index)) for index in range(row.count()))
