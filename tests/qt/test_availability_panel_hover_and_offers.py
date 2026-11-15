"""The availability panel's lifetime, hover corridor and install offers.

:class:`spacr.qt.widgets.availability_panel.AvailabilityPanel` is an
interactive tooltip: its anchor can be destroyed while it is up, the pointer
crosses a gap to reach its Install link, and it docks against screen edges.
:func:`run_install_offer` turns an offer into an inform/confirm/install flow
whose side effects are injected.  Each test walks a case table and asserts the
exact visibility, timer state, geometry or flow outcome.
"""
from __future__ import annotations

import time

import pytest

pytest.importorskip("PySide6")

import shiboken6                                                      # noqa: E402
from PySide6.QtCore import QEvent, QPoint, QRect, Qt                  # noqa: E402
from PySide6.QtGui import QGuiApplication, QKeyEvent                  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget                   # noqa: E402

from spacr.qt.widgets.availability_panel import (                    # noqa: E402
    AvailabilityPanel, run_install_offer)

pytestmark = pytest.mark.qt

FAR_AWAY = QPoint(-5000, -5000)
DOCK = QRect(200, 200, 100, 20)


class _Offer:
    """The ``InstallOffer`` surface this module reads."""

    def __init__(self, action="installable", requirement="cuml-cu12",
                 command=("pip", "install", "cuml-cu12"), message="",
                 title="GPU acceleration"):
        self.action = action
        self.requirement = requirement
        self.command = list(command)
        self.message = message
        self.title = title

    def as_text(self):
        """The recipe text the panel shows for the offer."""
        return f"{self.title}: {self.action}"


def _entry():
    """One greyed-out feature with an install offer."""
    return {"title": "cuML", "reason": "cuml is not installed", "url": "",
            "offer": _Offer()}


@pytest.fixture
def make_panel(qapp):
    """Panels of this test's own, dismissed however the test ends."""
    made = []

    def make(anchor_rect=None, **kwargs):
        panel = AvailabilityPanel()
        anchor = QWidget()
        made.append((panel, anchor))
        if anchor_rect is not None:
            kwargs["anchor_rect"] = anchor_rect
        panel.show_for(anchor, [_entry()], immediate=True, **kwargs)
        return panel, anchor

    yield make
    for panel, _anchor in made:
        if shiboken6.isValid(panel):
            panel.dismiss()
            panel.deleteLater()


def test_the_hide_check_keeps_the_panel_while_the_pointer_is_on_its_way(
        make_panel, monkeypatch, subtests):
    """``_maybe_hide``: the panel, the anchor and the corridor hold it open
    (the corridor only within its grace); anything else closes it."""
    cases = [
        # case, under panel, under anchor, cursor, expired grace, pinned, visible
        ("pointer on the panel", True, None, None, False, False, True),
        ("pointer back on the anchor", False, True, FAR_AWAY, False, False,
         True),
        ("pointer crossing the corridor", None, None, "corridor", False, False,
         True),
        ("pointer parked past the grace", None, None, "corridor", True, False,
         False),
        ("pointer well clear", None, None, FAR_AWAY, False, False, False),
        ("a pinned panel ignores the timer", None, None, FAR_AWAY, False, True,
         True),
    ]
    for case, on_panel, on_anchor, cursor, expired, pinned, visible in cases:
        with subtests.test(msg=case):
            panel, anchor = make_panel(DOCK, pinned=pinned)
            assert panel.is_pinned() is pinned
            panel.start_hide(1)
            assert panel._hide_timer.isActive() is (not pinned)
            if on_panel is not None:
                monkeypatch.setattr(panel, "underMouse", lambda v=on_panel: v)
            if on_anchor is not None:
                monkeypatch.setattr(anchor, "underMouse", lambda v=on_anchor: v)
            if cursor == "corridor":
                cursor = panel.corridor().center()
            if cursor is not None:
                monkeypatch.setattr(panel, "_cursor_pos", lambda c=cursor: c)
            if expired:
                panel._hide_since = time.monotonic() - (
                    panel.CORRIDOR_GRACE_MS / 1000.0 + 1.0)
            panel._maybe_hide()
            assert panel.isVisible() is visible
            if case == "pointer crossing the corridor":
                assert panel._hide_timer.isActive()
            monkeypatch.undo()

    with subtests.test(msg="enter cancels the hide, leave arms it"):
        panel, _anchor = make_panel()
        panel.start_hide(5000)
        assert panel._hide_timer.isActive()
        QApplication.sendEvent(panel, QEvent(QEvent.Enter))
        assert panel._hide_timer.isActive() is False
        QApplication.sendEvent(panel, QEvent(QEvent.Leave))
        assert panel._hide_timer.isActive() is True


def test_the_panel_docks_on_screen_and_forgets_a_destroyed_anchor(
        make_panel, monkeypatch, subtests):
    """``show_for`` geometry and the anchor's lifetime."""
    available = QGuiApplication.primaryScreen().availableGeometry()

    with subtests.test(msg="an explicit rectangle docks the panel"):
        rect = QRect(40, 60, 120, 20)
        panel, _anchor = make_panel(rect)
        assert panel._anchor_global_rect() == rect
        assert panel.corridor().contains(rect)

    with subtests.test(msg="the bottom of the screen opens upwards"):
        rect = QRect(available.left() + 10, available.bottom() - 4, 100, 4)
        panel, _anchor = make_panel(rect)
        assert panel.y() + panel.sizeHint().height() <= available.bottom() + 1
        assert panel.y() < rect.top()

    with subtests.test(msg="off every screen lands on the primary"):
        panel, _anchor = make_panel(QRect(-100000, -100000, 10, 10))
        assert panel.x() == available.left()
        assert panel.y() >= available.top()

    with subtests.test(msg="an anchor destroyed under the panel"):
        panel, anchor = make_panel()
        assert panel._anchor is anchor
        shiboken6.delete(anchor)
        assert panel._anchor_global_rect() is None
        assert panel._anchor is None

    with subtests.test(msg="a destroyed anchor during the hide check"):
        panel, anchor = make_panel(DOCK)
        panel.start_hide(1)
        shiboken6.delete(anchor)
        monkeypatch.setattr(panel, "_cursor_pos", lambda: FAR_AWAY)
        panel._maybe_hide()
        assert panel._anchor is None
        assert panel.isVisible() is False


def test_keys_and_the_install_handler(make_panel, qapp, subtests):
    """Escape closes and says so once, other keys pass on, and the install
    handler is replaced or cleared rather than stacked."""
    with subtests.test(msg="escape closes and announces once"):
        panel, _anchor = make_panel()
        closed = []
        panel.dismissed.connect(lambda: closed.append(1))
        event = QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier)
        panel.keyPressEvent(event)
        assert panel.isVisible() is False
        assert closed == [1]
        assert event.isAccepted()
        panel.dismiss()
        assert closed == [1]

    with subtests.test(msg="a key the panel has no use for"):
        panel, _anchor = make_panel()
        event = QKeyEvent(QEvent.KeyPress, Qt.Key_A, Qt.NoModifier)
        event.ignore()
        panel.keyPressEvent(event)
        assert event.isAccepted() is False

    for case, handlers, receivers in [
            ("a replaced handler", ["first", "second"], {"first": 0, "second": 1}),
            ("a cleared handler", ["first", None], {"first": 0}),
    ]:
        with subtests.test(msg=case):
            panel, _anchor = make_panel()
            seen = {name: [] for name in handlers if name}
            for name in handlers:
                panel.set_install_handler(seen[name].append if name else None)
            offer = _Offer()
            panel.install_requested.emit(offer)
            assert {name: len(got) for name, got in seen.items()} == receivers

    with subtests.test(msg="clearing the handler of a destroyed panel"):
        doomed = AvailabilityPanel()
        doomed.set_install_handler(lambda _offer: None)
        shiboken6.delete(doomed)
        doomed.set_install_handler(None)
        assert doomed._install_handler is None


def test_an_offer_that_cannot_be_installed_here_is_only_explained(subtests):
    """``run_install_offer``: ready, elsewhere, impossible and requirement-less
    offers inform the user and never confirm, resolve or install."""
    def never(what):
        return lambda *_a: pytest.fail(f"{what} ran")

    cases = [
        ("ready", _Offer(action="ready", message="cuML is already here"),
         "ready", [("GPU acceleration", "cuML is already here")]),
        ("elsewhere", _Offer(action="elsewhere"), "explained",
         [("GPU acceleration", "GPU acceleration: elsewhere")]),
        ("impossible", _Offer(action="impossible"), "explained",
         [("GPU acceleration", "GPU acceleration: impossible")]),
        ("no requirement", _Offer(action="installable", requirement=""),
         "explained", [("GPU acceleration", "GPU acceleration: installable")]),
    ]
    for case, offer, outcome, informed in cases:
        with subtests.test(msg=case):
            seen = []
            assert run_install_offer(
                None, offer, inform=lambda t, x: seen.append((t, x)),
                confirm=never("confirm"), dry_run=never("the resolver"),
                install=never("pip")) == outcome
            assert seen == informed

    with subtests.test(msg="the default resolver is built but never asked"):
        seen = []
        assert run_install_offer(
            None, _Offer(action="elsewhere"),
            inform=lambda t, x: seen.append((t, x)),
            confirm=never("confirm"), install=never("pip")) == "explained"
        assert seen == [("GPU acceleration", "GPU acceleration: elsewhere")]
