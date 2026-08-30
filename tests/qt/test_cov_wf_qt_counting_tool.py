"""``B13`` — the four ways the counting panel is asked to do nothing.

A manual count is thousands of keystrokes and clicks, and most of what the
keyboard and the layer model send this panel is *not* a count. The branches
covered here are the ones that decide to stay out of the way: a letter key
that belongs to whatever else is listening, a layer event that renamed
something rather than moving a marker, a selected row the session has no
class for, and a session with no active class at all. Each of them is a
"leave it alone" decision, and each is tested here beside the input that
*does* make the panel act, so an assertion about nothing happening is never
the only thing being measured.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest

from spacr.layers import LayerStack, Spacing
from spacr.qt import counting_tool as ct
from spacr.qt import layer_viewer as lv


def _stack():
    stack = LayerStack()
    stack.add_image(np.zeros((64, 64), np.uint16), name="image",
                    spacing=Spacing.isotropic(2, 1.0, units="px"))
    return stack


def _canvas(qtbot, stack=None):
    canvas = lv.LayerCanvas(stack if stack is not None else _stack())
    qtbot.addWidget(canvas)
    canvas.resize(200, 200)
    canvas._ensure_canvas()
    return canvas


def _panel(qtbot, canvas, **kwargs):
    panel = ct.CountingPanel(canvas, **kwargs)
    qtbot.addWidget(panel)
    return panel


def _key_event(key, text):
    return QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier, text)


# ---------------------------------------------------------------------------
# The tool: a key that is not a digit belongs to somebody else
# ---------------------------------------------------------------------------

def test_a_letter_key_is_left_for_whatever_else_is_listening(
        qtbot, qt_theme_applied):
    """Only the number row selects a class; every other key passes through.

    The counting tool sits on the canvas for the whole session, so it sees
    every keystroke the canvas gets — including the ones that belong to the
    rest of the viewer. If the tool swallowed a letter or an arrow key by
    claiming to have handled it, the canvas would stop panning with the
    arrows and every single-letter shortcut in the window would go dead the
    moment counting was switched on, with nothing on screen to explain why.
    The tool must answer "not mine" for anything that is not a digit.
    """
    canvas = _canvas(qtbot)
    panel = _panel(qtbot, canvas)
    tool = panel.start_counting()

    # A digit IS the tool's: it claims the key and moves the active class.
    assert tool.key(canvas, _key_event(Qt.Key_2, "2")) is True
    assert panel.session.active == "uninfected"

    # A letter is not, and neither is a key that carries no text at all.
    assert tool.key(canvas, _key_event(Qt.Key_A, "a")) is False
    assert tool.key(canvas, _key_event(Qt.Key_Left, "")) is False
    assert tool.key(canvas, _key_event(Qt.Key_Shift, "")) is False

    # Through the real widget too: the class and the tally are untouched.
    QTest.keyClick(canvas, Qt.Key_A)
    assert panel.session.active == "uninfected"
    assert panel.session.total == 0

    # ...and the number row still works after all that.
    QTest.keyClick(canvas, Qt.Key_1)
    assert panel.session.active == "infected"


# ---------------------------------------------------------------------------
# The model: only the events that change the markers rebuild the tally
# ---------------------------------------------------------------------------

def test_renaming_or_selecting_a_layer_does_not_rebuild_the_tally(
        qtbot, qt_theme_applied):
    """A rename or a selection is not a count, so the tally must not redraw.

    The panel listens to the whole layer stack, which also carries the
    housekeeping of the layer list: renames, reordering, and the highlight
    moving from one row to another. Rebuilding the class list on those would
    re-emit ``counts_changed`` — the signal a screen uses to write the count
    into a table — for events where no marker moved, so a downstream table
    would record a "new" identical count every time somebody clicked around
    the layer list. Worse, the rebuild reselects the active class row, which
    would fight the user's own selection. Only ``data``, ``inserted`` and
    ``removed`` are counts.
    """
    stack = _stack()
    canvas = _canvas(qtbot, stack)
    panel = _panel(qtbot, canvas)

    seen = []
    panel.counts_changed.connect(seen.append)

    image = stack["image"]
    assert stack.rename(image, "field 3") == "field 3"
    stack.select(image)
    stack.move(image, len(list(stack)) - 1)

    assert seen == [], f"housekeeping redrew the tally: {seen}"

    # The same listener, driven by an event that IS a count.
    panel.session.add({"y": 10.0, "x": 10.0})

    assert len(seen) == 1
    assert seen[-1] == {"infected": 1, "uninfected": 0}
    assert panel.total.text() == panel.session.describe()


# ---------------------------------------------------------------------------
# The class list: a row past the last class changes nothing
# ---------------------------------------------------------------------------

def test_a_row_the_session_has_no_class_for_leaves_the_active_class_alone(
        qtbot, qt_theme_applied):
    """Selecting a row beyond the session's classes must not raise or switch.

    ``class_list`` is a public widget on the panel and the selection signal
    fires for whatever row lands in it, including a row put there by a
    screen that has got ahead of the model. Indexing the class names with
    that row would raise straight out of a Qt slot — which in this build
    means a traceback on the console and a selection the model never
    followed — and taking the nearest class instead would silently start
    scoring clicks as the wrong thing. Out of range means "not a class",
    and the active class stays where the counter put it.
    """
    canvas = _canvas(qtbot)
    panel = _panel(qtbot, canvas)

    names = panel.session.class_names
    assert panel.class_list.count() == len(names) == 2
    assert panel.session.active == "infected"

    # A row the session has no class for: nothing changes.
    panel.class_list.addItem("a row nothing backs")
    panel.class_list.setCurrentRow(2)

    assert panel.class_list.currentRow() == 2
    assert panel.session.active == "infected", (
        "an unbacked row was allowed to change what clicks are scored as")

    # The same signal, on a row that IS a class, does move the active class.
    panel.class_list.setCurrentRow(1)
    assert panel.session.active == "uninfected"

    # And a click now scores as the class the real row chose.
    panel.session.add({"y": 12.0, "x": 12.0})
    assert panel.session.counts() == {"infected": 0, "uninfected": 1}


def test_a_negative_row_from_a_cleared_list_is_not_a_selection(
        qtbot, qt_theme_applied):
    """Clearing the list emits row ``-1``, which must not touch the model.

    Every refresh clears ``class_list`` before it refills it, and Qt reports
    that as the current row changing to ``-1``. Treating ``-1`` as a real
    selection would index the class names from the end and quietly make the
    LAST class active on every redraw — so the counter's chosen class would
    jump the moment anything in the stack changed, and the markers already
    placed would be joined by ones scored as something else.
    """
    canvas = _canvas(qtbot)
    panel = _panel(qtbot, canvas)
    panel.session.active = "infected"

    panel.class_list.setCurrentRow(-1)
    assert panel.class_list.currentRow() == -1
    assert panel.session.active == "infected"

    # A redraw (which clears the list) also leaves the class where it was.
    panel.session.add({"y": 20.0, "x": 20.0})
    panel.refresh()

    assert panel.session.active == "infected"
    assert panel.class_list.currentRow() == 0
    assert panel.session.counts() == {"infected": 1, "uninfected": 0}


# ---------------------------------------------------------------------------
# The tally: a session with no classes yet
# ---------------------------------------------------------------------------

def test_a_panel_with_no_classes_yet_draws_an_empty_tally(
        qtbot, qt_theme_applied):
    """A panel opened with nothing to count must still draw, and stay empty.

    ``classes=[]`` is how a screen offers a blank counting session for the
    user to name their own classes in. Such a session has no active class —
    the active name is the empty string — and the redraw has to notice that
    there is no row to highlight instead of selecting one. If it selected
    something anyway the list would show a highlight for a class that does
    not exist, and the first ``+ Class`` press would look like it had
    re-selected rather than created.
    """
    canvas = _canvas(qtbot)
    panel = _panel(qtbot, canvas, classes=[])

    assert panel.session.class_names == ()
    assert panel.session.active == ""
    assert panel.class_list.count() == 0
    assert panel.class_list.currentRow() == -1
    assert panel.total.text() == "nothing counted yet"

    seen = []
    panel.counts_changed.connect(seen.append)

    # Now give it something to count: the same redraw path highlights it.
    created = panel.add_class("infected")

    assert created == "infected"
    assert panel.session.active == "infected"
    assert panel.class_list.count() == 1
    assert panel.class_list.currentRow() == 0
    assert seen[-1] == {"infected": 0}

    # ...and the highlighted row is the one the tally is about.
    panel.session.add({"y": 30.0, "x": 30.0})
    item = panel.class_list.item(panel.class_list.currentRow())
    assert item.data(Qt.UserRole) == "infected"
    assert "1" in item.text()
    assert panel.total.text() == "infected 1 (100%) · 1 total"


def test_an_unnamed_class_is_numbered_by_how_many_there_are(
        qtbot, qt_theme_applied):
    """The ``+ Class`` button names the class when the caller does not.

    The button is connected straight to ``add_class``, and a Qt ``clicked``
    signal hands the slot a ``False`` for the checked state — not a name.
    Passing that through would ask the session to count something called
    ``False``, or blow up on a blank name. The panel has to recognise a
    non-string (or an all-whitespace string) as "no name given" and number
    the class instead, which is what makes the button usable at all.
    """
    canvas = _canvas(qtbot)
    panel = _panel(qtbot, canvas, classes=[])

    # Exactly what the button's clicked(bool) signal delivers.
    assert panel.add_class(False) == "class 1"
    assert panel.add_class("   ") == "class 2"
    # A real name is kept.
    assert panel.add_class("mitotic") == "mitotic"

    assert panel.session.class_names == ("class 1", "class 2", "mitotic")
    assert panel.class_list.count() == 3
    assert [panel.class_list.item(i).data(Qt.UserRole) for i in range(3)] == [
        "class 1", "class 2", "mitotic"]
