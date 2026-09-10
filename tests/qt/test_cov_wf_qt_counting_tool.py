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

from spacr.counting import CountingSession
from spacr.layers import LayerEvent, LayerStack, Spacing
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


# ---------------------------------------------------------------------------
# The same "do nothing" decisions, reached without the drawing canvas
# ---------------------------------------------------------------------------
#
# Everything above drives a real :class:`~spacr.qt.layer_viewer.LayerCanvas`
# and real Qt events. That is the honest end-to-end route, but it means a
# machine that cannot paint — a headless build, a session where the canvas
# fails to come up — takes these decisions with it and nobody notices the
# counting panel has stopped refusing the input it must refuse. The tests
# below reach the same refusals through the seams the panel actually
# publishes: a tool that is handed an event object, a listener that is handed
# a :class:`~spacr.layers.LayerEvent`, and a slot that is handed a row.


class _EventStub:
    """The two things :meth:`CountingTool.key` asks a key event for."""

    def __init__(self, key, text):
        self._key = key
        self._text = text

    def key(self):
        return self._key

    def text(self):
        return self._text


class _CanvasStub:
    """The three things :class:`CountingPanel` asks its canvas for.

    A counting panel needs a layer stack to count into and somewhere to hand
    the tool; it does not need pixels. Standing in for the canvas keeps the
    panel's own decisions testable on a machine that cannot draw.
    """

    def __init__(self, stack):
        self.stack = stack
        self.tool = None

    def set_tool(self, tool):
        self.tool = tool


def test_the_tool_only_claims_a_key_that_names_a_class():
    """A key with no digit in it must be handed back, not swallowed.

    The counting tool stays on the canvas for the whole session, so every
    keystroke the viewer receives passes through it first. Its answer is what
    decides whether the key travels on to the rest of the window. If it
    claimed a key it had no use for — a modifier press that carries no text,
    a letter, a punctuation mark — then switching counting on would silently
    kill every other single-key shortcut in the viewer, and the counter would
    have no way to tell that the tool was the thing eating them. Only a digit
    that actually names one of this session's classes is the tool's business.
    """
    session = CountingSession(_stack())
    session.active = "uninfected"
    tool = ct.CountingTool(session)

    # A digit that names a class is claimed, and it moves the active class.
    assert tool.key(None, _EventStub(Qt.Key_1, "1")) is True
    assert session.active == "infected"

    # Anything that is not a digit is not the tool's, and changes nothing.
    for key, text in ((Qt.Key_Shift, ""), (Qt.Key_Left, ""), (Qt.Key_A, "a"),
                      (Qt.Key_Plus, "+"), (Qt.Key_Space, " ")):
        assert tool.key(None, _EventStub(key, text)) is False, text
    # ...and a digit with no class behind it is not claimed either.
    assert session.class_for_shortcut("7") is None
    assert tool.key(None, _EventStub(Qt.Key_7, "7")) is False

    assert session.active == "infected", (
        "a key the tool should have passed on moved the active class")
    assert session.total == 0


def test_layer_housekeeping_does_not_re_emit_the_tally(qtbot,
                                                       qt_theme_applied):
    """Only a marker change may re-announce the count downstream.

    ``counts_changed`` is how a screen writes a count into a table, and the
    panel is subscribed to the *whole* layer stack — which also carries
    renames, reordering, selection and display-property changes such as a
    layer's colour or opacity. Announcing a "new" count for those would put a
    duplicate row in the table every time somebody clicked around the layer
    list or dragged an opacity slider, and the rebuild would also drag the
    highlighted class back to the active one under the counter's fingers.
    The listener has to recognise the events that cannot have changed a
    marker and return without redrawing.
    """
    stack = _stack()
    panel = ct.CountingPanel(_CanvasStub(stack))
    qtbot.addWidget(panel)

    seen = []
    panel.counts_changed.connect(seen.append)

    image = stack["image"]
    for kind in ("renamed", "moved", "selected", "changed"):
        panel._on_layers_changed(LayerEvent(kind=kind, layer=image, index=0))

    assert seen == [], f"housekeeping re-announced the tally: {seen}"

    # The same listener, given an event that IS a marker change.
    panel._on_layers_changed(LayerEvent(kind="data", layer=image, index=0))

    assert seen == [{"infected": 0, "uninfected": 0}]
    assert panel.total.text() == "nothing counted yet"


def test_a_row_number_the_session_cannot_name_is_not_a_selection(
        qtbot, qt_theme_applied):
    """A row past the last class must not raise or re-score the clicks.

    ``class_list`` is public, and the row it reports is only as fresh as
    whatever last wrote to it: a screen that adds a row, or a selection left
    over from a session with more classes, both deliver a number the model
    has no class for. Indexing the names with it would raise straight out of
    a Qt slot, and clamping to the nearest class instead would quietly start
    scoring every following click as the wrong thing — the worst outcome for
    a hand count, because the tally still looks plausible. Out of range has
    to mean "not a class".
    """
    stack = _stack()
    panel = ct.CountingPanel(_CanvasStub(stack))
    qtbot.addWidget(panel)

    names = panel.session.class_names
    assert names == ("infected", "uninfected")
    assert panel.session.active == "infected"

    for row in (len(names), len(names) + 3):
        panel._on_class_selected(row)
        assert panel.session.active == "infected", (
            f"row {row}, which names no class, changed what clicks score as")

    # The same slot on a row that IS a class does move the active class...
    panel._on_class_selected(1)
    assert panel.session.active == "uninfected"

    # ...and the next click is scored as the class that row named.
    panel.session.add({"y": 8.0, "x": 8.0})
    assert panel.session.counts() == {"infected": 0, "uninfected": 1}


def test_a_prepared_session_with_nothing_in_it_yet_draws_no_highlight(
        qtbot, qt_theme_applied):
    """A panel handed an empty session must draw, and highlight nothing.

    A screen can build the counting session itself — to point it at a
    particular field, or to hand the same session to two views — and pass it
    in. Such a session can arrive with no classes at all, which means no
    active class: the name is the empty string. The redraw has to notice
    there is no row to highlight rather than reaching for one, because a
    highlight on a class that does not exist tells the counter their clicks
    are being scored as something, when in fact the first click would fail.
    """
    stack = _stack()
    session = CountingSession(stack, classes=[])
    panel = ct.CountingPanel(_CanvasStub(stack), session=session)
    qtbot.addWidget(panel)

    assert panel.session is session
    assert session.active == ""
    assert panel.class_list.count() == 0
    assert panel.class_list.currentRow() == -1
    assert panel.total.text() == "nothing counted yet"

    # A redraw of the same empty session still highlights nothing.
    panel.refresh()
    assert panel.class_list.currentRow() == -1

    # Give it something to count and the highlight appears on that class.
    panel.add_class("mitotic")

    assert session.active == "mitotic"
    assert panel.class_list.count() == 1
    assert panel.class_list.currentRow() == 0
    assert panel.class_list.item(0).data(Qt.UserRole) == "mitotic"
