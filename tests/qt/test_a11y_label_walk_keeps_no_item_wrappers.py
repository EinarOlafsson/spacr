"""Finding a field's caption leaves no layout-item wrappers behind.

``_a11y_label_beside`` walked layouts with ``itemAt``. PySide's layout glue
makes the asked layout the Python parent of each item wrapper it returns,
so every walk left Qt-created ``QWidgetItem`` wrappers alive under the
layout. Once Qt deleted such an item (its widget moved or went), the
wrapper stayed registered at the freed address and the next object
allocated there was handed to an event filter as a ``QWidgetItem``.
"""
from __future__ import annotations

import gc

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import (QFormLayout, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                               QVBoxLayout, QWidget, QWidgetItem)

from spacr.qt.i18n import _a11y_label_beside

pytestmark = pytest.mark.qt


def _item_wrappers():
    gc.collect()
    return sum(1 for obj in gc.get_objects() if isinstance(obj, QWidgetItem))


def test_box_and_grid_captions_are_found_without_keeping_item_wrappers(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    column = QVBoxLayout(host)
    row = QHBoxLayout()
    column.addLayout(row)
    box_label, box_field = QLabel("Objects"), QLineEdit()
    row.addWidget(box_label)
    row.addSpacing(6)
    row.addWidget(box_field)
    grid = QGridLayout()
    column.addLayout(grid)
    grid_label, grid_field = QLabel("Channel"), QLineEdit()
    grid.addWidget(grid_label, 0, 0)
    grid.addWidget(QLabel("unit"), 1, 1)
    grid.addWidget(grid_field, 0, 2)
    form = QFormLayout()
    column.addLayout(form)
    form_field = QLineEdit()
    form.addRow("Name", form_field)
    before = _item_wrappers()

    assert _a11y_label_beside(box_field) is box_label
    assert _a11y_label_beside(grid_field) is grid_label
    assert _a11y_label_beside(form_field) is not None
    assert _a11y_label_beside(box_label) is None

    assert _item_wrappers() == before
