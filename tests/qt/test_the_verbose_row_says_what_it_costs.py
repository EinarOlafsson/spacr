"""The Diagnostics row in Preferences says what verbose logging costs.

The sentence a user reads is the one the hint strip shows when the pointer
is on the row's label. `explain_every_row` moves the switch's tooltip there,
so a check on the source string alone would pass while the strip said
something else. This one opens the dialog and reads the strip.

The figures come from the whole-application benchmark of 2026-09-19. Home
was ready in 4.00-4.18 s from a cold start, and the slowest module opened in
7.02-7.54 s, with verbose on and with it off. The row used to quote 3 s
against 65 s. That was the function tracer's cost, and the preference
stopped installing the tracer on 2026-08-30.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF
from PySide6.QtGui import QEnterEvent
from PySide6.QtWidgets import QApplication, QFormLayout, QLabel

from spacr.qt.preferences import PreferencesDialog
from spacr.qt.widgets.hint_bar import HintBar


@pytest.fixture
def dialog(qtbot, qt_theme_applied, tmp_path, monkeypatch):
    """A Preferences dialog built against an empty preference store."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    built = PreferencesDialog()
    qtbot.addWidget(built)
    built.show()
    qtbot.waitExposed(built)
    return built


def _diagnostics_label(dialog) -> QLabel:
    """The label of the row that holds the verbose switch."""
    for form in dialog.findChildren(QFormLayout):
        for index in range(form.rowCount()):
            item = form.itemAt(index, QFormLayout.LabelRole)
            label = item.widget() if item is not None else None
            if isinstance(label, QLabel) and label.text() == "Diagnostics":
                return label
    raise AssertionError("Preferences has no Diagnostics row")


def test_hovering_the_row_shows_the_measured_cost(dialog):
    """Point at the label, as a user does, and read the strip."""
    label = _diagnostics_label(dialog)
    bar = dialog.findChild(HintBar)
    assert bar is not None, "Preferences built no hint strip"

    QApplication.sendEvent(
        label, QEnterEvent(QPointF(4, 4), QPointF(4, 4), QPointF(4, 4)))
    shown = bar.text()

    assert "about 4 seconds" in shown, shown
    assert "about 7 seconds" in shown, shown
    assert "does not trace every function call" in shown, shown
    assert "65 seconds" not in shown, (
        "the row still quotes the function tracer's cost")
