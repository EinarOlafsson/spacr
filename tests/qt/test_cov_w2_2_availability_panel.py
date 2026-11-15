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


import pytest
import shiboken6
from PySide6.QtCore import QEvent, QPoint, QRect, Qt
from PySide6.QtGui import QGuiApplication, QKeyEvent
from PySide6.QtWidgets import QMessageBox, QWidget

from spacr.qt.widgets import availability_panel as ap
from spacr.qt.widgets.availability_panel import (AvailabilityPanel, explain)


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


def _clear_the_process_wide_panel():
    """Dismiss and forget the singleton, if this process has one."""
    existing = AvailabilityPanel._INSTANCE
    if existing is not None:
        existing.dismiss()
        existing.deleteLater()
    AvailabilityPanel._INSTANCE = None


@pytest.fixture
def no_singleton_left_behind():
    """`instance()` and `explain()` build a process-wide panel; clear it.

    CLEARED BEFORE AS WELL AS AFTER, and the "before" half is the one that
    was missing. Leaving nothing behind makes this file safe for its
    NEIGHBOURS; it does nothing about a panel a neighbour left for US, and
    `test_explaining_nothing_opens_nothing` asserts the singleton is None --
    a statement about the whole process, not about this test.

    Six other Qt test files reach this panel, and
    `spacr/qt/screens/settings_model.py` builds one through `instance()` at
    four sites, so any settings test that opens a greyed row can leave one
    standing. That is why the failure is CI-only and order-dependent: run the
    file alone and there is no neighbour to inherit from.
    """
    _clear_the_process_wide_panel()
    yield
    _clear_the_process_wide_panel()


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
    panel.show_for(anchor, [_entry("cuML"), _entry("Torch")], immediate=True)

    handed = panel.entries()
    assert [e["title"] for e in handed] == ["cuML", "Torch"]
    handed.clear()
    assert len(panel.entries()) == 2

    assert panel.current_entry()["title"] == "cuML"
    assert isinstance(panel.current_offer(), _Offer)


def test_up_and_down_move_between_the_entries_and_wrap(panel, qapp):
    """A pinned panel is cyclable, so the last entry's Down is the first."""
    anchor = QWidget()
    panel.show_for(anchor, [_entry("cuML"), _entry("Torch")], pinned=True, immediate=True)

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


# ---------------------------------------------------------------------------
# an anchor that is not there any more
# ---------------------------------------------------------------------------

def test_a_panel_with_no_anchor_stays_where_it_is(panel, qapp):
    """Nothing to dock under is a no-op, not a move to the origin."""
    assert panel._anchor_global_rect() is None
    before = panel.pos()
    panel._position()
    assert panel.pos() == before


def test_the_corridor_falls_back_to_the_panel_itself(panel, qapp):
    """With no anchor the corridor is the panel, and nothing when hidden."""
    assert panel.corridor() is None

    anchor = QWidget()
    panel.show_for(anchor, [_entry()], immediate=True)
    panel._anchor = None
    panel._anchor_rect = None
    assert panel.corridor() == panel.geometry()


def test_a_panel_anchored_far_above_every_screen_is_clamped_vertically(panel,
                                                                       qapp):
    """The dock is clamped to the screen on y as well as on x."""
    anchor = QWidget()
    panel.show_for(anchor, [_entry()],
                   anchor_rect=QRect(-100000, -100000, 10, 10), immediate=True)
    available = QGuiApplication.primaryScreen().availableGeometry()
    assert panel.y() >= available.top()
    assert panel.y() + panel.sizeHint().height() <= available.bottom() + 1


# ---------------------------------------------------------------------------
# staying alive across the gap
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# what pressing the word does
# ---------------------------------------------------------------------------


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


