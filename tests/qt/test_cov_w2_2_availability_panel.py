"""The panel's edges: a dead anchor, a screen corner, and a link nobody wired.

An interactive tooltip is a set of lifetime problems wearing a label. The
anchor can be destroyed while the panel is still up. The pointer can be
somewhere the offscreen platform has no answer for. The widget that had focus
before the panel opened can be gone by the time focus is handed back. Each of
those is a `RuntimeError` from PySide6 in the middle of an event handler,
which is exactly where an exception is least survivable.

The anchors here are real widgets, really destroyed through shiboken, so the
`RuntimeError` the code catches is the one Qt actually raises. The install
flow is driven through its injected callables, which is what the function
documents them as being for.
"""

import time

import pytest
import shiboken6
from PySide6.QtCore import QEvent, QPoint, QRect, Qt
from PySide6.QtGui import QGuiApplication, QKeyEvent
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

from spacr.qt.widgets import availability_panel as ap
from spacr.qt.widgets.availability_panel import (AvailabilityPanel, explain,
                                                 run_install_offer)


class _Offer:
    """The `InstallOffer` surface this module reads."""

    def __init__(self, action="installable", requirement="cuml-cu12",
                 command=("pip", "install", "cuml-cu12"), message="",
                 title="GPU acceleration"):
        self.action = action
        self.requirement = requirement
        self.command = list(command)
        self.message = message
        self.title = title

    def as_text(self):
        return f"{self.title}: {self.action}"


def _entry(title="cuML", reason="cuml is not installed", url="", offer=None):
    return {"title": title, "reason": reason, "url": url,
            "offer": offer if offer is not None else _Offer()}


@pytest.fixture
def panel(qapp):
    """A panel of this test's own, dismissed however the test ends."""
    made = AvailabilityPanel()
    yield made
    made.dismiss()
    made.deleteLater()


@pytest.fixture
def no_singleton_left_behind():
    """`instance()` and `explain()` build a process-wide panel; clear it."""
    yield
    existing = AvailabilityPanel._INSTANCE
    if existing is not None:
        existing.dismiss()
        existing.deleteLater()
    AvailabilityPanel._INSTANCE = None


# ---------------------------------------------------------------------------
# what is on screen
# ---------------------------------------------------------------------------

def test_an_empty_panel_has_no_entry_and_no_offer(panel):
    """Asked before it is shown, it says so rather than indexing nothing."""
    assert panel.entries() == []
    assert panel.current_entry() is None
    assert panel.current_offer() is None
    panel.show_entry(3)                       # must not raise
    assert panel.current_entry() is None


def test_the_entries_it_hands_back_are_a_copy(panel, qapp):
    """Mutating what a caller was given must not change what is on screen."""
    anchor = QWidget()
    panel.show_for(anchor, [_entry("cuML"), _entry("Torch")])

    handed = panel.entries()
    assert [e["title"] for e in handed] == ["cuML", "Torch"]
    handed.clear()
    assert len(panel.entries()) == 2

    assert panel.current_entry()["title"] == "cuML"
    assert isinstance(panel.current_offer(), _Offer)


def test_up_and_down_move_between_the_entries_and_wrap(panel, qapp):
    """A pinned panel is cyclable, so the last entry's Down is the first."""
    anchor = QWidget()
    panel.show_for(anchor, [_entry("cuML"), _entry("Torch")], pinned=True)

    def press(key):
        event = QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier)
        panel.keyPressEvent(event)
        return event

    assert press(Qt.Key_Down).isAccepted()
    assert panel.current_entry()["title"] == "Torch"
    press(Qt.Key_Down)
    assert panel.current_entry()["title"] == "cuML"
    press(Qt.Key_Up)
    assert panel.current_entry()["title"] == "Torch"


def test_a_key_the_panel_has_no_use_for_is_passed_on(panel, qapp):
    """Only Escape and the arrows are the panel's; the rest are not eaten."""
    anchor = QWidget()
    panel.show_for(anchor, [_entry()])
    event = QKeyEvent(QEvent.KeyPress, Qt.Key_A, Qt.NoModifier)
    event.ignore()
    panel.keyPressEvent(event)
    assert event.isAccepted() is False


def test_escape_closes_the_panel_and_says_it_closed(panel, qapp):
    """`dismissed` is what a caller re-enables its own control on."""
    closed = []
    panel.dismissed.connect(lambda: closed.append(1))
    anchor = QWidget()
    panel.show_for(anchor, [_entry()])
    assert panel.isVisible()

    event = QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier)
    panel.keyPressEvent(event)

    assert panel.isVisible() is False
    assert closed == [1]
    assert event.isAccepted()

    # a second dismiss is silent: it was already closed
    panel.dismiss()
    assert closed == [1]


# ---------------------------------------------------------------------------
# an anchor that is not there any more
# ---------------------------------------------------------------------------

def test_a_panel_with_no_anchor_stays_where_it_is(panel, qapp):
    """Nothing to dock under is a no-op, not a move to the origin."""
    assert panel._anchor_global_rect() is None
    before = panel.pos()
    panel._position()
    assert panel.pos() == before


def test_an_anchor_destroyed_under_the_panel_is_forgotten(panel, qapp):
    """Qt raises `RuntimeError` for a destroyed widget; the panel drops it.

    A screen closing while its tooltip is up is an ordinary event, not an
    exceptional one.
    """
    anchor = QWidget()
    panel.show_for(anchor, [_entry()])
    assert panel._anchor is anchor

    shiboken6.delete(anchor)
    assert panel._anchor_global_rect() is None
    assert panel._anchor is None


def test_the_corridor_falls_back_to_the_panel_itself(panel, qapp):
    """With no anchor the corridor is the panel, and nothing when hidden."""
    assert panel.corridor() is None

    anchor = QWidget()
    panel.show_for(anchor, [_entry()])
    panel._anchor = None
    panel._anchor_rect = None
    assert panel.corridor() == panel.geometry()


def test_an_explicit_rectangle_docks_the_panel_without_the_widget(panel,
                                                                  qapp):
    """One row of an open combo popup is smaller than the widget it is in."""
    anchor = QWidget()
    rect = QRect(40, 60, 120, 20)
    panel.show_for(anchor, [_entry()], anchor_rect=rect)
    assert panel._anchor_global_rect() == rect
    assert panel.corridor().contains(rect)


def test_a_panel_at_the_bottom_of_the_screen_opens_upwards(panel, qapp):
    """Docking below would put it off the screen, so it flips above."""
    screen = QGuiApplication.primaryScreen()
    available = screen.availableGeometry()
    anchor = QWidget()
    rect = QRect(available.left() + 10, available.bottom() - 4, 100, 4)

    panel.show_for(anchor, [_entry()], anchor_rect=rect)
    assert panel.y() + panel.sizeHint().height() <= available.bottom() + 1
    assert panel.y() < rect.top()


def test_a_panel_anchored_off_every_screen_still_lands_somewhere(panel,
                                                                 qapp):
    """`screenAt` answers None for a point on no screen; the primary is used."""
    anchor = QWidget()
    panel.show_for(anchor, [_entry()],
                   anchor_rect=QRect(-100000, -100000, 10, 10))
    available = QGuiApplication.primaryScreen().availableGeometry()
    assert panel.x() >= available.left()


# ---------------------------------------------------------------------------
# staying alive across the gap
# ---------------------------------------------------------------------------

def test_the_pointer_travelling_through_the_gap_re_arms_the_hide(panel,
                                                                 qapp,
                                                                 monkeypatch):
    """A naive leave-event dismissal makes the Install link unreachable."""
    anchor = QWidget()
    rect = QRect(200, 200, 100, 20)
    panel.show_for(anchor, [_entry()], anchor_rect=rect)
    panel.start_hide(1)
    corridor = panel.corridor()

    monkeypatch.setattr(panel, "_cursor_pos", lambda: corridor.center())
    panel._maybe_hide()
    assert panel.isVisible(), "the panel closed while the pointer crossed it"
    assert panel._hide_timer.isActive()


def test_a_pointer_well_clear_of_the_corridor_closes_the_panel(panel, qapp,
                                                               monkeypatch):
    """Moving away is a dismissal; the corridor is a grace, not a lock."""
    anchor = QWidget()
    panel.show_for(anchor, [_entry()], anchor_rect=QRect(200, 200, 100, 20))
    panel.start_hide(1)

    monkeypatch.setattr(panel, "_cursor_pos", lambda: QPoint(-5000, -5000))
    panel._maybe_hide()
    assert panel.isVisible() is False


def test_the_grace_period_runs_out_even_inside_the_corridor(panel, qapp,
                                                            monkeypatch):
    """A pointer parked in the gap does not hold the panel open forever."""
    anchor = QWidget()
    panel.show_for(anchor, [_entry()], anchor_rect=QRect(200, 200, 100, 20))
    panel.start_hide(1)
    corridor = panel.corridor()
    monkeypatch.setattr(panel, "_cursor_pos", lambda: corridor.center())
    panel._hide_since = time.monotonic() - (panel.CORRIDOR_GRACE_MS / 1000.0
                                            + 1.0)

    panel._maybe_hide()
    assert panel.isVisible() is False


def test_a_pinned_panel_ignores_every_hover_timer(panel, qapp):
    """A reader who is not holding the mouse must not lose the panel."""
    anchor = QWidget()
    panel.show_for(anchor, [_entry()], pinned=True)
    assert panel.is_pinned() is True

    panel.start_hide(1)
    assert panel._hide_timer.isActive() is False
    panel._maybe_hide()
    assert panel.isVisible() is True


def test_the_panel_under_the_pointer_is_not_hidden(panel, qapp, monkeypatch):
    """The pointer arriving on the panel is the opposite of leaving."""
    anchor = QWidget()
    panel.show_for(anchor, [_entry()])
    monkeypatch.setattr(panel, "underMouse", lambda: True)
    panel._maybe_hide()
    assert panel.isVisible() is True


def test_a_destroyed_anchor_does_not_break_the_hide_check(panel, qapp,
                                                          monkeypatch):
    """Asking a deleted widget whether it is under the mouse raises."""
    anchor = QWidget()
    panel.show_for(anchor, [_entry()], anchor_rect=QRect(200, 200, 100, 20))
    panel.start_hide(1)
    shiboken6.delete(anchor)
    monkeypatch.setattr(panel, "_cursor_pos", lambda: QPoint(-5000, -5000))

    panel._maybe_hide()                       # must not raise
    assert panel._anchor is None
    assert panel.isVisible() is False


def test_the_pointer_arriving_and_leaving_arms_and_cancels_the_hide(panel,
                                                                    qapp):
    """The two events the whole corridor rule hangs off."""
    anchor = QWidget()
    panel.show_for(anchor, [_entry()])
    panel.start_hide(5000)
    assert panel._hide_timer.isActive()

    QApplication.sendEvent(panel, QEvent(QEvent.Enter))
    assert panel._hide_timer.isActive() is False

    QApplication.sendEvent(panel, QEvent(QEvent.Leave))
    assert panel._hide_timer.isActive() is True


def test_focus_is_handed_back_even_when_it_has_nowhere_to_go(panel, qapp):
    """The widget that had focus can be destroyed while the panel is up."""
    holder = QWidget()
    holder.show()
    panel._return_focus = holder
    shiboken6.delete(holder)

    panel.dismiss()                           # must not raise
    assert panel._return_focus is None


# ---------------------------------------------------------------------------
# the one Install receiver
# ---------------------------------------------------------------------------

def test_replacing_the_install_handler_does_not_accumulate_receivers(panel,
                                                                     qapp):
    """Two callers connected at once would both answer one press."""
    first, second = [], []
    panel.set_install_handler(first.append)
    panel.set_install_handler(second.append)

    offer = _Offer()
    panel.install_requested.emit(offer)
    assert first == []
    assert second == [offer]


def test_clearing_the_handler_of_a_destroyed_panel_is_quiet(qapp):
    """The singleton can be taken away while a caller still holds it.

    A plain `disconnect()` warns when nothing is connected, and a signal on a
    destroyed QObject raises; neither is worth reporting on the way out.
    """
    doomed = AvailabilityPanel()
    doomed.set_install_handler(lambda _offer: None)
    shiboken6.delete(doomed)

    doomed.set_install_handler(None)          # must not raise
    assert doomed._install_handler is None


def test_the_handler_can_be_cleared(panel, qapp):
    """`None` leaves the signal with no receiver at all."""
    seen = []
    panel.set_install_handler(seen.append)
    panel.set_install_handler(None)
    panel.install_requested.emit(_Offer())
    assert seen == []


# ---------------------------------------------------------------------------
# what pressing the word does
# ---------------------------------------------------------------------------

def _flow():
    """Recorders for the four injected side effects."""
    return {"informed": [], "confirmed": [], "installed": []}


def test_an_offer_with_no_requirement_is_explained_and_nothing_runs(qapp):
    """There is nothing to install, so the recipe is shown instead."""
    calls = _flow()
    outcome = run_install_offer(
        None, _Offer(action="installable", requirement=""),
        inform=lambda t, x: calls["informed"].append((t, x)),
        confirm=lambda t, x: calls["confirmed"].append(t) or True,
        dry_run=lambda r: pytest.fail("the resolver ran with no requirement"),
        install=lambda c: pytest.fail("pip ran with no requirement"))

    assert outcome == "explained"
    assert calls["informed"]
    assert calls["confirmed"] == []


def test_an_offer_that_is_already_available_says_so(qapp):
    """'ready' runs nothing and reports the offer's own message."""
    informed = []
    outcome = run_install_offer(
        None, _Offer(action="ready", message="cuML is already here"),
        inform=lambda t, x: informed.append(x),
        install=lambda c: pytest.fail("pip ran for a ready offer"))
    assert outcome == "ready"
    assert informed == ["cuML is already here"]


def test_an_offer_that_belongs_elsewhere_runs_nothing(qapp):
    """A prompt that runs pip here either fails or breaks the install."""
    for action in ("elsewhere", "impossible"):
        informed = []
        outcome = run_install_offer(
            None, _Offer(action=action),
            inform=lambda t, x: informed.append(x),
            install=lambda c: pytest.fail("pip ran for an %s offer" % action))
        assert outcome == "explained"
        assert informed


def test_the_default_dialogs_are_a_question_and_an_information(qapp,
                                                               monkeypatch):
    """The default confirm's default button is No, and inform just tells."""
    asked = {}

    def question(parent, title, text, buttons, default):
        asked["question"] = (title, text, default)
        return QMessageBox.Yes

    def information(parent, title, text):
        asked["information"] = (title, text)

    monkeypatch.setattr(QMessageBox, "question", staticmethod(question))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(information))

    confirm = ap._default_confirm(None)
    assert confirm("Install cuml?", "the plan") is True
    assert asked["question"][0] == "Install cuml?"
    assert asked["question"][2] == QMessageBox.No, \
        "the default button was not No"

    inform = ap._default_inform(None)
    assert inform("done", "restart spaCR") is None
    assert asked["information"] == ("done", "restart spaCR")

    # and a No really is a refusal
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.No))
    assert ap._default_confirm(None)("Install?", "plan") is False


# ---------------------------------------------------------------------------
# the two-line form
# ---------------------------------------------------------------------------

def test_explaining_nothing_opens_nothing(qapp, no_singleton_left_behind):
    """No entries means there is nothing unavailable to explain."""
    assert explain(QWidget(), []) is None
    assert explain(QWidget(), None) is None
    assert AvailabilityPanel._INSTANCE is None


def test_explaining_wires_install_to_the_caller_s_own_re_probe(
        qapp, no_singleton_left_behind, monkeypatch):
    """A successful install tells the caller, so it can ask again."""
    reprobed = []
    outcomes = ["installed"]

    def fake_run(parent, offer, **kwargs):
        return outcomes[0]

    monkeypatch.setattr(ap, "run_install_offer", fake_run)

    offer = _Offer()
    panel = explain(QWidget(), [_entry(offer=offer)],
                    on_installed=reprobed.append)
    assert panel is AvailabilityPanel.instance()
    assert panel.isVisible()

    panel.install_requested.emit(offer)
    assert reprobed == [offer]

    # an install that did not happen does not trigger the re-probe
    outcomes[0] = "declined"
    panel.install_requested.emit(offer)
    assert reprobed == [offer]


def test_the_keyboard_route_opens_the_panel_pinned(qapp,
                                                   no_singleton_left_behind):
    """A disabled row cannot be tabbed to, so this is the explicit way in."""
    anchor = QWidget()
    anchor.show()
    panel = explain(anchor, [_entry(), _entry("Torch")], pinned=True)
    assert panel.is_pinned() is True
    assert panel.isVisible()
    assert panel.api_link() is not None
    assert panel.install_link() is not None
    assert panel.body_label() is not None


def test_the_panel_asks_the_real_cursor_where_it_is(panel, qapp):
    """A method rather than a bare call, so the corridor can be driven.

    The offscreen platform has no pointer to move, and the gap-crossing rule
    is the one that has to be checked rather than assumed -- but the seam
    still has to return the real position when nobody has replaced it.
    """
    from PySide6.QtGui import QCursor

    where = panel._cursor_pos()
    assert isinstance(where, QPoint)
    assert where == QCursor.pos()


def test_the_pointer_still_on_the_anchor_keeps_the_panel_open(panel, qapp,
                                                              monkeypatch):
    """Leaving the panel back onto the row it belongs to is not leaving."""
    anchor = QWidget()
    panel.show_for(anchor, [_entry()], anchor_rect=QRect(200, 200, 100, 20))
    panel.start_hide(1)
    monkeypatch.setattr(panel, "underMouse", lambda: False)
    monkeypatch.setattr(anchor, "underMouse", lambda: True)
    monkeypatch.setattr(panel, "_cursor_pos", lambda: QPoint(-5000, -5000))

    panel._maybe_hide()
    assert panel.isVisible() is True
