"""Re-describing the card already described keeps the reader's caret and links.

The lower description panel is refilled whenever the pointer enters a card.
Entering the card it already describes asked for the same text again, and
the panel replaced its document: the caret, a selection and the position of
the link under the pointer all jumped. On a real display the late Enter event
from a warped pointer did exactly that between finding a link and clicking it.
"""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QMainWindow

from spacr.qt.widgets import home

pytestmark = pytest.mark.qt


@pytest.fixture
def details(qtbot):
    window = QMainWindow()
    qtbot.addWidget(window)
    browser = home._NavigatorDetailsBrowser(window)
    window.setCentralWidget(browser)
    window.resize(900, 240)
    window.show()
    qtbot.waitExposed(window)
    return browser


def _place_caret_in(details, text):
    cursor = details.document().find(text)
    assert not cursor.isNull()
    cursor.setPosition(cursor.selectionStart() + 2)
    details.setTextCursor(cursor)
    return cursor.position()


def test_the_same_description_again_keeps_the_caret_and_the_link(details):
    html = ('<p><b>Images</b><br/>Inspect images and edit masks.</p>'
            + home._navigator_learning_links('make_masks'))
    details.setHtml(html)
    revision = details.document().revision()
    position = _place_caret_in(details, 'Open Dock')
    point = details.cursorRect().center()
    assert details.anchorAt(point) == 'spacr://sidebar'

    details.setHtml(html)

    assert details.document().revision() == revision
    assert details.textCursor().position() == position
    assert details.anchorAt(details.cursorRect().center()) == 'spacr://sidebar'


def test_a_different_description_or_an_edit_still_redraws(details):
    first = '<p>First card.</p>' + home._navigator_learning_links('make_masks')
    second = '<p>Second card.</p>' + home._navigator_learning_links('mask')
    details.setHtml(first)
    details.setHtml(second)
    assert 'Second card.' in details.toPlainText()
    details.setHtml(first)
    assert 'First card.' in details.toPlainText()
    details.document().setPlainText('changed elsewhere')
    details.setHtml(first)
    assert 'First card.' in details.toPlainText()
