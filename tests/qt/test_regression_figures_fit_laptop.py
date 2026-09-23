"""Regression results shrink without covering the shell or clipping controls."""
import pytest
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QScrollArea


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
    qtbot.wait(50)
    assert not screen._part_is_owed(_REGRESSION_RESULTS)
    assert screen._if_built('_results_panel') is not None
    scroll = card.body
    assert isinstance(scroll, QScrollArea)

    def check_shell():
        panes = [card, screen._console_wrap, screen._usage_card, screen._actions_section]
        for upper, lower in zip(panes, panes[1:]):
            assert upper.geometry().bottom() < lower.geometry().top()
        assert panes[-1].geometry().bottom() < screen._runtime_splitter.height()

    check_shell()
    assert scroll.verticalScrollBar().maximum() > 0
    scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
    qtbot.wait(10)
    content = scroll.widget()
    bottom = content.mapTo(scroll.viewport(), content.rect().bottomLeft()).y()
    assert 0 <= bottom < scroll.viewport().height()
    scroll.verticalScrollBar().setValue(0)
    assert content.mapTo(scroll.viewport(), QPoint()).y() == 0

    screen._console_folder.toggle()
    qtbot.wait(20)
    check_shell()
    card.folder.toggle()
    qtbot.wait(20)
    assert not scroll.isVisibleTo(window)
    card.folder.toggle()
    qtbot.wait(20)
    assert scroll.isVisibleTo(window)
    check_shell()
