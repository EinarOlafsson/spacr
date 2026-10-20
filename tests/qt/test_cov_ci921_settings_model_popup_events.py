"""Popup-viewport events of the regression backend combo that are not hover.

Covers ``_RegressionBackendField.eventFilter``: an Enter event on the combo
popup's viewport is neither a mouse move nor a leave, so it hovers no row,
releases nothing, and the base filter's answer is returned.
"""
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent

from spacr.qt.screens import settings_model as SM

pytestmark = pytest.mark.qt


def test_other_popup_events_neither_hover_nor_release(qtbot):
    field = SM._RegressionBackendField(regression_type="ols")
    qtbot.addWidget(field)
    hovered, released = [], []
    field._hover_popup_row = lambda *a: hovered.append(a)
    field._release_popup = lambda *a: released.append(a)
    viewport = field.combo.view().viewport()

    result = field.eventFilter(viewport, QEvent(QEvent.Enter))

    assert result is False
    assert hovered == [] and released == []
