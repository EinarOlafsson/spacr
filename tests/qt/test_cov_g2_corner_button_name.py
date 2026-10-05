"""A table's corner button is named "Select all" whatever Qt calls it."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QAbstractButton, QTableWidget  # noqa: E402

from spacr.qt import i18n  # noqa: E402


@pytest.mark.parametrize("object_name", ["", "qt_tableview_cornerbutton"])
def test_the_corner_button_is_select_all(qtbot, object_name):
    table = QTableWidget(2, 2)
    qtbot.addWidget(table)
    corner = table.findChild(QAbstractButton)
    corner.setObjectName(object_name)
    name, _description = i18n._a11y_source(corner, {})
    assert name == i18n.tr("Select all")
