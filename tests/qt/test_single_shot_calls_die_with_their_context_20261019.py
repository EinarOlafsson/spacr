from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from types import SimpleNamespace

import shiboken6

from PySide6.QtWidgets import QApplication


def _drain(qapp):
    for _ in range(3):
        qapp.processEvents()


def test_console_panel_scroll_is_dropped_when_the_panel_dies(qapp, monkeypatch):
    from spacr.qt.widgets.console_panel import ConsolePanel

    calls = []
    monkeypatch.setattr(
        ConsolePanel, "_scroll_widget_to_top",
        lambda self, bar: calls.append(bar))
    monkeypatch.setattr(ConsolePanel, "section_body", lambda self, bar: [])
    panel = ConsolePanel()
    bar = SimpleNamespace(set_expanded=lambda value: None)
    panel.raise_section(bar)
    shiboken6.delete(panel)
    _drain(qapp)
    assert calls == []


def test_figure_grid_scroll_is_dropped_when_the_view_dies(qapp, monkeypatch):
    from spacr.qt.widgets.figure_grid_view import FigureGridView

    calls = []
    monkeypatch.setattr(
        FigureGridView, "_scroll_section_to_top",
        lambda self, key: calls.append(key))
    monkeypatch.setattr(FigureGridView, "_relayout", lambda self: None)
    view = FigureGridView()
    header = SimpleNamespace(section_key="a")
    view._collapsed.add("a")
    assert view.toggle_section(header) is True
    shiboken6.delete(view)
    _drain(qapp)
    assert calls == []
