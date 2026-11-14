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

from PySide6.QtCore import Qt

from spacr.counting import CountingSession
from spacr.layers import LayerEvent, LayerStack, Spacing
from spacr.qt import counting_tool as ct


def _stack():
    stack = LayerStack()
    stack.add_image(np.zeros((64, 64), np.uint16), name="image",
                    spacing=Spacing.isotropic(2, 1.0, units="px"))
    return stack


# ---------------------------------------------------------------------------
# The tool: a key that is not a digit belongs to somebody else
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The model: only the events that change the markers rebuild the tally
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The class list: a row past the last class changes nothing
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The tally: a session with no classes yet
# ---------------------------------------------------------------------------


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
