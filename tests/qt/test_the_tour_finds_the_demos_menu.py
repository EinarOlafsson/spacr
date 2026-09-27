"""The tour's Demos beat finds the Demos menu on a window that has one, and
says so in the log when it does not.

The beat only resolves the menu (popping it up would grab input for the
rest of the render), so what it returns is the whole of its behaviour: the
real QMenu when the bar carries "&Demos", None and a warning when it does not.
"""
import logging

from PySide6.QtWidgets import QMainWindow

from spacr.qt.tutorial import scripts


def test_the_demos_menu_on_the_bar_is_found(qtbot):
    window = QMainWindow()
    qtbot.addWidget(window)
    window.menuBar().addMenu("&File")
    demos = window.menuBar().addMenu("&Demos")

    assert scripts._open_demos_menu(window) is demos


def test_a_bar_without_demos_warns(qtbot, caplog):
    window = QMainWindow()
    qtbot.addWidget(window)
    window.menuBar().addMenu("&File")

    with caplog.at_level(logging.WARNING, logger=scripts.LOG.name):
        assert scripts._open_demos_menu(window) is None
    assert "no Demos menu" in caplog.text
