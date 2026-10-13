"""Item 288: the display and pixel ratio lookups when the obvious source is gone.

``spacr.qt.hidpi`` answers "which display" and "how many real pixels per
logical pixel" for every picture spaCR draws. The answers below are the
fall-backs, each pinned by value:

* with no GUI application there is no display;
* a widget whose C++ half is gone is placed on the primary display rather
  than raising;
* a widget that reports no ratio of its own takes its display's;
* a widget that cannot take an event filter gets no ratio watcher.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject  # noqa: E402
from PySide6.QtWidgets import QWidget  # noqa: E402

from spacr.qt import hidpi  # noqa: E402


def test_without_an_application_there_is_no_display(monkeypatch):
    monkeypatch.setattr(hidpi, "QGuiApplication",
                        SimpleNamespace(instance=lambda: None))
    assert hidpi.screen_for_widget(QObject()) is None


def test_a_deleted_widget_is_placed_on_the_primary_display(qapp):
    import shiboken6

    widget = QWidget()
    shiboken6.delete(widget)
    assert hidpi.screen_for_widget(widget) is qapp.primaryScreen()


class _Unrendered(QWidget):
    """A widget that cannot yet say its own ratio."""

    def devicePixelRatioF(self):  # noqa: N802
        return 0.0

    def devicePixelRatio(self):  # noqa: N802
        return 0


def test_a_widget_without_a_ratio_takes_its_displays(qtbot, qapp):
    widget = _Unrendered()
    qtbot.addWidget(widget)
    expected = hidpi._ratio_of(hidpi.screen_for_widget(widget))
    assert expected > 0
    assert hidpi.device_ratio(widget) == expected


class _NoFilters(QObject):
    def installEventFilter(self, _watcher):  # noqa: N802
        raise RuntimeError("this object takes no event filters")


def test_an_object_that_takes_no_filter_gets_no_watcher(qapp):
    assert hidpi.follow_device_ratio(_NoFilters(), lambda: None) is None
