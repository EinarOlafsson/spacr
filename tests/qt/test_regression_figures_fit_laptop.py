"""Regression results shrink without covering the shell or clipping controls."""
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QPoint
from PySide6.QtWidgets import QScrollArea


def _wait_for_runtime_geometry(qtbot, screen):
    """Observe settled panes and scroll ranges before measuring reachability.

    :param qtbot: the event-loop runner for this test.
    :param screen: the real regression screen whose layout must settle.
    :returns: None after consecutive unchanged readings, within one second.
    """
    from spacr.qt.widgets.collapsible_splitter import CollapsibleSplitter

    card = screen._figures_card
    scroll = card.body
    bar = scroll.verticalScrollBar()
    panes = [card, screen._console_wrap, screen._usage_card,
             screen._actions_section]
    splitters = screen.findChildren(CollapsibleSplitter)
    observed = panes + [screen.window(), screen._runtime_viewport,
                        screen._runtime_viewport.widget(), scroll.viewport(),
                        scroll.widget()]
    previous = None

    def settled():
        nonlocal previous
        # qWait can return with a layout request posted during its final sleep.
        QCoreApplication.sendPostedEvents(None, QEvent.LayoutRequest)
        geometry = (tuple(widget.geometry().getRect() for widget in observed),
                    bar.minimum(), bar.maximum(), bar.pageStep())
        pending = screen._runtime_viewport._floor_timer.isActive() or any(
            splitter._rebalance_queued
            or getattr(splitter, '_refit_queued', False)
            for splitter in splitters)
        unchanged = not pending and geometry == previous
        previous = None if pending else geometry
        return unchanged

    qtbot.waitUntil(settled, timeout=1000)


@pytest.mark.parametrize('height', [600, 768])
def test_results_scroll_inside_the_card_and_shell_headings_remain_accessible(
        height, qtbot, qt_theme_applied):
    from spacr.qt.app import MainWindow
    from spacr.qt.screens.app_screen import _REGRESSION_RESULTS
    from spacr.qt.walkthrough import mark_seen

    mark_seen('regression')
    window = MainWindow()
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.resize(1366, height)
    window.show()
    window._on_nav_selected('regression')
    screen = window._screens['regression']
    assert screen._part_is_owed(_REGRESSION_RESULTS)
    card = screen._figures_card
    card.show()
    assert not screen._part_is_owed(_REGRESSION_RESULTS)
    assert screen._if_built('_results_panel') is not None
    scroll = card.body
    assert isinstance(scroll, QScrollArea)
    _wait_for_runtime_geometry(qtbot, screen)

    def check_shell():
        panes = [card, screen._console_wrap, screen._usage_card, screen._actions_section]
        for upper, lower in zip(panes, panes[1:]):
            assert upper.geometry().bottom() < lower.geometry().top()
        assert panes[-1].geometry().bottom() < screen._runtime_splitter.height()

    def check_scrolling():
        assert scroll.verticalScrollBar().maximum() > 0
        scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
        _wait_for_runtime_geometry(qtbot, screen)
        content = scroll.widget()
        bottom = content.mapTo(scroll.viewport(), content.rect().bottomLeft()).y()
        assert 0 <= bottom < scroll.viewport().height()
        scroll.verticalScrollBar().setValue(0)
        assert content.mapTo(scroll.viewport(), QPoint()).y() == 0

    check_shell()
    check_scrolling()

    screen._console_folder.toggle()
    _wait_for_runtime_geometry(qtbot, screen)
    check_shell()
    card.folder.toggle()
    _wait_for_runtime_geometry(qtbot, screen)
    assert not scroll.isVisibleTo(window)
    card.folder.toggle()
    _wait_for_runtime_geometry(qtbot, screen)
    assert scroll.isVisibleTo(window)
    check_shell()
    check_scrolling()
