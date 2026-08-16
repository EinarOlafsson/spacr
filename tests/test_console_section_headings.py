"""Click a console section heading to bring it to the top and expand it.

Instruction 110.

The assertion that matters most is
:func:`test_new_output_does_not_drag_the_view_off_a_raised_section`. Raising a
section is a statement that the user is reading THERE; a live log that scrolls
away from what you are reading cannot be read at all, and the console appends
continuously while a pipeline runs.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, QEventLoop, QPoint, Qt, QTimer
from PySide6.QtGui import QKeyEvent, QMouseEvent

pytestmark = pytest.mark.qt


def _settle(ms: int = 120) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


@pytest.fixture
def panel(qtbot):
    from spacr.qt.widgets.console_panel import ConsolePanel

    widget = ConsolePanel()
    qtbot.addWidget(widget)
    widget.resize(700, 300)
    widget.show()
    for index in range(10):
        widget.begin_topic(f"Section {index}")
        widget.append_stdout(f"line {index}\n" * 25)
    _settle()
    return widget


def _bars(panel):
    from spacr.qt.widgets.console_panel import _TopicBar

    return panel.findChildren(_TopicBar)


def _click(bar):
    """A real press and release on the heading, as a user's mouse sends."""
    from PySide6.QtCore import QPointF
    from PySide6.QtWidgets import QApplication

    centre = QPointF(bar.rect().center())
    for kind in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease):
        QApplication.sendEvent(
            bar, QMouseEvent(kind, centre, Qt.LeftButton, Qt.LeftButton,
                             Qt.NoModifier))


def test_one_click_on_a_heading_brings_its_section_to_the_top(panel):
    """The gesture that was asked for, on a section in the state it is in.

    "Click on a console section heading to bring it to the top of the console
    and expand it." Sections are created EXPANDED, so a plain
    expanded/collapsed toggle spent the user's first click hiding the section
    they were trying to reach, and the viewport never moved at all. Reaching
    a section is the first meaning of clicking its heading.
    """
    bar = _bars(panel)[6]
    assert bar.is_expanded(), "the fixture's sections start expanded"
    scrollbar = panel._scroll.verticalScrollBar()

    _click(bar)
    _settle()

    top = bar.mapTo(panel._holder, bar.rect().topLeft()).y()
    assert abs(scrollbar.value() - min(top, scrollbar.maximum())) <= 4, (
        "one click on a heading must bring its section to the top")
    assert bar.is_expanded(), "and leave it expanded, not fold it away"


def test_clicking_the_heading_again_folds_the_section_away(panel):
    """The second click is the toggle.

    A heading already sitting at the top of the viewport has nowhere left to
    navigate to, so collapsing is the only thing the gesture can still mean --
    and it is exactly where the user's hand already is.
    """
    bar = _bars(panel)[6]
    _click(bar)
    _settle()
    assert bar.is_expanded()

    _click(bar)
    _settle()
    assert not bar.is_expanded()
    assert all(not w.isVisible() for w in panel.section_body(bar))


def test_a_heading_is_an_interactive_control(panel):
    bar = _bars(panel)[0]
    assert bar.cursor().shape() == Qt.PointingHandCursor
    assert bar.focusPolicy() == Qt.StrongFocus
    # A toggle with no indicator is a control found by accident.
    assert bar._chevron.text() in ("▾", "▸")


def test_clicking_a_heading_scrolls_its_section_to_the_top(panel):
    bar = _bars(panel)[6]
    panel.raise_section(bar)
    _settle()

    scrollbar = panel._scroll.verticalScrollBar()
    top = bar.mapTo(panel._holder, bar.rect().topLeft()).y()
    assert abs(scrollbar.value() - min(top, scrollbar.maximum())) <= 2


def test_the_order_of_sections_is_never_changed(panel):
    """A transcript's order is the one property a log has: scroll, never
    reorder."""
    before = [b.text() for b in _bars(panel)]
    panel.raise_section(_bars(panel)[6])
    _settle()
    assert [b.text() for b in _bars(panel)] == before


def test_collapsing_hides_the_body_and_turns_the_chevron(panel):
    bar = _bars(panel)[3]
    body = panel.section_body(bar)
    assert body, "the fixture should give every section a body"

    panel.collapse_section(bar)
    assert not bar.is_expanded()
    assert bar._chevron.text() == "▸"
    assert all(not w.isVisible() for w in body)

    panel.raise_section(bar)
    _settle()
    assert bar.is_expanded()
    assert bar._chevron.text() == "▾"
    assert all(w.isVisible() for w in body)


def test_new_output_does_not_drag_the_view_off_a_raised_section(panel):
    """The whole point. Raising a section suspends following."""
    bar = _bars(panel)[6]
    panel.raise_section(bar)
    _settle()
    resting = panel._scroll.verticalScrollBar().value()
    assert panel._follow_output is False

    panel.append_stdout("MORE OUTPUT\n" * 30)
    _settle()
    assert panel._scroll.verticalScrollBar().value() == resting


def test_scrolling_back_to_the_bottom_resumes_following(panel):
    bar = _bars(panel)[6]
    panel.raise_section(bar)
    _settle()
    assert panel._follow_output is False

    scrollbar = panel._scroll.verticalScrollBar()
    scrollbar.setValue(scrollbar.maximum())
    _settle(60)
    assert panel._follow_output is True


def test_the_keyboard_activates_a_heading(panel):
    """Enter and Space do what the mouse does: reach the section, then fold
    it. A control only a mouse can reach is one some users cannot reach."""
    bar = _bars(panel)[3]
    scrollbar = panel._scroll.verticalScrollBar()
    assert bar.is_expanded()

    bar.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Return, Qt.NoModifier))
    _settle()
    top = bar.mapTo(panel._holder, bar.rect().topLeft()).y()
    assert abs(scrollbar.value() - min(top, scrollbar.maximum())) <= 4
    assert bar.is_expanded()

    bar.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Space, Qt.NoModifier))
    _settle(30)
    assert not bar.is_expanded()


def test_the_copy_button_does_not_move_the_viewport(panel):
    """The heading's children handle their own clicks. Copying a section is
    not a request to navigate to it."""
    bar = _bars(panel)[6]
    panel.raise_section(bar)
    _settle()
    resting = panel._scroll.verticalScrollBar().value()
    expanded = bar.is_expanded()

    bar._copy_btn.click()
    _settle(30)
    assert panel._scroll.verticalScrollBar().value() == resting
    assert bar.is_expanded() is expanded


def test_section_body_agrees_with_section_text(panel):
    """Raise, collapse and copy must not disagree about where a section
    ends."""
    bar = _bars(panel)[4]
    body = panel.section_body(bar)
    text = panel.section_text(bar)
    for widget in body:
        if hasattr(widget, "toPlainText"):
            snippet = widget.toPlainText().strip().splitlines()
            if snippet:
                assert snippet[0] in text


def test_dragging_off_the_heading_cancels(panel):
    bar = _bars(panel)[3]
    was = bar.is_expanded()
    bar.mouseReleaseEvent(
        QMouseEvent(QEvent.MouseButtonRelease, QPoint(-40, -40),
                    Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
    _settle(30)
    assert bar.is_expanded() is was
