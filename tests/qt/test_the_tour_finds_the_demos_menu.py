"""Tutorial menu lookup respects the live menu bar and missing destinations."""
import logging

from PySide6.QtWidgets import QMainWindow
from spacr.qt.tutorial import scripts


def test_the_menu_on_the_bar_is_found_without_its_mnemonic(qtbot):
    window = QMainWindow()
    qtbot.addWidget(window)
    reference = window.menuBar().addMenu("&Reference")
    assert scripts._find_menu(window, "Reference") is reference


def test_an_absent_menu_has_no_cursor_target(qtbot, caplog):
    window = QMainWindow()
    qtbot.addWidget(window)
    window.menuBar().addMenu("&File")
    with caplog.at_level(logging.WARNING, logger=scripts.LOG.name):
        bar, point = scripts._menu_target(window, "Reference")
    assert bar is window.menuBar() and point is None
    assert "has no 'Reference' menu" in caplog.text
