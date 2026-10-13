"""Controls remain readable at the font sizes used for workflow recordings."""
import pytest

from PySide6.QtCore import QPoint, QRect
from PySide6.QtWidgets import QPushButton, QStackedWidget

from .test_the_text_fits_sweep import at_font_scale, settle


def _assert_condition_controls_fit(table):
    header = table.horizontalHeader()
    for column in (1, 2):
        assert table.columnWidth(column) >= header.sectionSizeHint(column)
    for row in range(table.rowCount()):
        role = table.cellWidget(row, 2)
        assert role.width() >= role.sizeHint().width()
        assert role.height() >= role.sizeHint().height()
    assert table.columnWidth(0) > table.columnWidth(1)
    assert table.columnWidth(0) > table.columnWidth(2)


@pytest.mark.parametrize("scale", [1.0, 1.5, 2.0])
def test_condition_headers_and_role_choices_fit_without_manual_resize(qtbot, at_font_scale, scale):
    from spacr.qt.screens.experiment_design import ExperimentDesignScreen

    at_font_scale(scale)
    host = QStackedWidget()
    qtbot.addWidget(host)
    screen = ExperimentDesignScreen(threaded=False)
    host.addWidget(screen)
    host.setMinimumSize(1, 1)
    host.resize(1280, 900)
    host.show()
    qtbot.waitExposed(host)
    settle(qtbot, screen)
    _assert_condition_controls_fit(screen._table)


def test_existing_and_added_conditions_follow_live_font_changes(qtbot, at_font_scale):
    from spacr.qt.screens.experiment_design import ExperimentDesignScreen

    at_font_scale(1.0)
    screen = ExperimentDesignScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.resize(1280, 900)
    screen.show()
    qtbot.waitExposed(screen)
    settle(qtbot, screen)
    original = screen.conditions()
    for scale in (2.0, 1.0):
        at_font_scale(scale)
        settle(qtbot, screen)
        _assert_condition_controls_fit(screen._table)
        assert screen.conditions() == original
    screen._add_row()
    settle(qtbot, screen)
    _assert_condition_controls_fit(screen._table)
    assert screen.conditions()[:len(original)] == original


def test_folded_ops_action_buttons_do_not_overlap_at_capture_scale(qtbot, at_font_scale):
    from spacr.qt.app import MainWindow
    from spacr.qt.screens.mask import ops_page

    at_font_scale(1.5)
    window = MainWindow()
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.resize(3840, 2160)
    window.show()
    window.open_module("ops")
    page = ops_page(window._screens["mask"]).page
    qtbot.waitUntil(page.isVisible)
    settle(qtbot, page)
    row = page._actions_row
    buttons = [button for button in row.findChildren(QPushButton) if button.isVisible()]
    assert page._btn_run in buttons and page._btn_remote in buttons
    rectangles = []
    for button in buttons:
        assert button.width() >= button.sizeHint().width(), button.text()
        rect = QRect(button.mapTo(row, QPoint()), button.size())
        assert row.rect().contains(rect), (button.text(), rect, row.rect())
        assert all(not rect.intersects(other) for other in rectangles), button.text()
        rectangles.append(rect)
