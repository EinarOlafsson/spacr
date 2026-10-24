"""News title observation stays inert until its native construction finishes."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget

from spacr.qt.widgets import home

pytestmark = pytest.mark.qt


def test_title_observation_defers_reentrant_construction_then_fits(qtbot, qapp):
    panel = QWidget()
    qtbot.addWidget(panel)
    observed = []

    class ReentrantNotes(QScrollArea):
        def viewport(self):
            fitter = panel.findChild(home._NewsTitleFit)
            if fitter is not None and not observed:
                observed.append(fitter)
                panel.installEventFilter(fitter)
                QCoreApplication.sendEvent(panel, QEvent(QEvent.FontChange))
            return super().viewport()

    panel._notes = ReentrantNotes(panel)
    panel._notes.setWidgetResizable(True)
    body = QWidget()
    title = QLabel('A release heading with enough words to wrap in a narrow viewport')
    title.setWordWrap(True)
    QVBoxLayout(body).addWidget(title)
    panel._notes.setWidget(body)
    QVBoxLayout(panel).addWidget(panel._notes)
    panel._headline_labels = [title]
    with qtbot.captureExceptions() as exceptions:
        fitter = home._NewsTitleFit(panel)
    assert observed == [fitter]
    assert exceptions == []
    panel.resize(300, 200)
    panel.show()
    qtbot.waitExposed(panel)
    viewport = panel._notes.viewport()
    qtbot.waitUntil(lambda: title.minimumWidth() == viewport.width()
                   and title.maximumWidth() == viewport.width()
                   and title.minimumHeight() >= title.fontMetrics().height()
                   and not fitter._timer.isActive())
    old_width = title.minimumWidth()
    panel.resize(230, 200)
    qtbot.waitUntil(lambda: title.minimumWidth() == viewport.width() < old_width
                   and title.maximumWidth() == viewport.width()
                   and not fitter._timer.isActive())
    assert title.text() == 'A release heading with enough words to wrap in a narrow viewport'
    assert title.minimumHeight() > title.fontMetrics().height()
