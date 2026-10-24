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


def test_news_viewport_destruction_cancels_pending_and_later_fitting(qtbot, qapp):
    import shiboken6

    panel = QWidget()
    qtbot.addWidget(panel)
    panel._notes = QScrollArea(panel)
    panel._notes.setWidgetResizable(True)
    body = QWidget()
    title = QLabel('A live title before the viewport is replaced')
    title.setWordWrap(True)
    QVBoxLayout(body).addWidget(title)
    panel._notes.setWidget(body)
    QVBoxLayout(panel).addWidget(panel._notes)
    panel._headline_labels = [title]
    fitter = home._NewsTitleFit(panel)
    panel.resize(300, 180)
    panel.show()
    qtbot.waitExposed(panel)
    viewport = panel._notes.viewport()
    qtbot.waitUntil(lambda: title.minimumWidth() == viewport.width()
                   and title.minimumHeight() > 0 and not fitter._timer.isActive())
    fitter._timer.start(0)
    with qtbot.captureExceptions() as exceptions:
        panel._notes.setViewport(QWidget())
        assert not shiboken6.isValid(viewport)
        assert shiboken6.isValid(fitter) and not fitter._timer.isActive()
        QCoreApplication.sendEvent(panel, QEvent(QEvent.FontChange))
        qapp.processEvents()
    assert exceptions == []
    assert not fitter._timer.isActive() and not fitter._busy
    assert panel.isVisible() and panel._notes.viewport() is not viewport


def test_body_viewport_destruction_cancels_pending_and_later_reflow(qtbot, qapp):
    import shiboken6
    from PySide6.QtWidgets import QTabWidget

    scroll = QScrollArea()
    qtbot.addWidget(scroll)
    scroll.setWidgetResizable(True)
    body = QWidget()
    hero = QLabel('Home masthead', body)
    hero.setObjectName('Hero')
    tabs = QTabWidget(body)
    page = QScrollArea()
    page_body = QWidget()
    QVBoxLayout(page_body)
    page.setWidget(page_body)
    tabs.addTab(page, 'Applications')
    layout = QVBoxLayout(body)
    layout.addWidget(hero)
    layout.addWidget(tabs)
    scroll.setWidget(body)
    reflows = []
    fitter = home._HomeBodyFit(body, scroll, tabs, lambda: reflows.append('reflow'))
    scroll.resize(400, 300)
    scroll.show()
    qtbot.waitExposed(scroll)
    viewport = scroll.viewport()
    qtbot.waitUntil(lambda: bool(reflows) and not fitter._timer.isActive()
                   and hero.maximumHeight() == viewport.height())
    retained_body = scroll.takeWidget()
    assert retained_body is body
    qtbot.addWidget(body)
    before = list(reflows)
    fitter._timer.start(0)
    with qtbot.captureExceptions() as exceptions:
        scroll.setViewport(QWidget())
        assert not shiboken6.isValid(viewport)
        assert shiboken6.isValid(fitter) and not fitter._timer.isActive()
        QCoreApplication.sendEvent(body, QEvent(QEvent.FontChange))
        qapp.processEvents()
    assert exceptions == []
    assert reflows == before
    assert not fitter._timer.isActive() and not fitter._busy
    assert shiboken6.isValid(hero) and shiboken6.isValid(tabs)
