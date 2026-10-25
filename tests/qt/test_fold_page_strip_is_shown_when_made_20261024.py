"""A host's page strip made on a screen already on show is visible at once.

On Windows the overview wizard opened inside a hidden ``FoldPages`` strip:
it is made lazily, as a new child of a visible screen, and was left waiting
for the show Qt queues when a layout adopts a widget.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from spacr.qt.screens.map_barcodes import PAGES_NAME, host_pages


def _screen(qtbot):
    screen = QWidget()
    layout = QVBoxLayout(screen)
    layout.addWidget(QLabel("masthead"))
    body = QWidget()
    layout.addWidget(body, 1)
    qtbot.addWidget(screen)
    return screen, body


def test_a_strip_made_on_a_shown_screen_is_visible_without_an_event_pass(qtbot):
    screen, body = _screen(qtbot)
    screen.show()
    qtbot.waitExposed(screen)
    pages = host_pages(screen)
    assert pages.objectName() == PAGES_NAME
    assert not pages.isHidden() and pages.isVisible()
    assert body.isVisible() and pages.currentWidget() is body


def test_a_strip_made_before_the_screen_shows_follows_it(qtbot):
    screen, body = _screen(qtbot)
    pages = host_pages(screen)
    screen.show()
    qtbot.waitExposed(screen)
    assert pages.isVisible() and body.isVisible()
