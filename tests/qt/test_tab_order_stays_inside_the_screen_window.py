"""The settings-first Tab order never links widgets from another window.

Opening the plaque assay logged "QWidget::setTabOrder: 'first' and 'second'
must be in the same window" every time the screen was shown: the runtime
column held a child that is its own window, and its widgets were chained
behind the main window's settings form.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLineEdit, QVBoxLayout, QWidget

from spacr.qt.screens.app_screen import AppScreen


class _Screen:
    """The attributes the Tab-order pass reads from a screen."""

    def __init__(self, window, settings, side):
        self._window = window
        self._settings_panel = settings
        self._runtime_wrap = side

    def window(self):
        return self._window


def test_a_separate_window_inside_the_runtime_column_keeps_its_own_tab_chain(qtbot, monkeypatch):
    window = QWidget()
    qtbot.addWidget(window)
    column = QVBoxLayout(window)
    side = QWidget(window)
    console = QLineEdit(side)
    QVBoxLayout(side).addWidget(console)
    settings = QWidget(window)
    field = QLineEdit(settings)
    QVBoxLayout(settings).addWidget(field)
    column.addWidget(side)
    column.addWidget(settings)
    floating = QWidget(side, Qt.Window)
    preview = QLineEdit(floating)
    QVBoxLayout(floating).addWidget(preview)
    window.show()
    floating.show()
    qtbot.waitExposed(window)

    pairs = []
    original = QWidget.setTabOrder

    def record(first, second):
        pairs.append((first, second))
        original(first, second)

    monkeypatch.setattr(QWidget, "setTabOrder", staticmethod(record))
    AppScreen._tab_through_the_form_first(_Screen(window, settings, side))

    assert pairs
    assert all(first.window() == second.window() == window for first, second in pairs)
    assert console in [second for _, second in pairs]
    assert preview not in [second for _, second in pairs]
    floating.close()
