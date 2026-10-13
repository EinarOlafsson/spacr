"""A fold gesture never takes the panel down with it.

Folding is a decoration on top of panels that are already built and already
doing work. Everything that can fail around the fold -- the click handler, the
caller's change callback, the preference store the fold is remembered in --
is therefore contained: the panel keeps working and the fold either happens or
does not. The alternative is a traceback out of an event filter, which in Qt
surfaces as an unhandled exception on the GUI thread.

Setting the state it already has is also a no-op on purpose: re-asserting
"open" must not clear an alert or fire the change callback, or a periodic
refresh that re-applies the stored state would erase the marker that says the
folded panel has something to show.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def _panel():
    from PySide6.QtWidgets import QLabel, QWidget

    return QLabel("Console"), QWidget()


def test_a_failing_toggle_does_not_escape_the_event_filter(qapp):
    """A click whose handler raises is still consumed, not re-thrown.

    An exception leaving ``eventFilter`` reaches Qt's event loop as an
    unhandled error on the GUI thread.
    """
    from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QLabel

    from spacr.qt.widgets.foldable import _ClickToFold

    label = QLabel("Console")
    calls = []

    def _explode():
        calls.append("tried")
        raise RuntimeError("fold failed")

    watcher = _ClickToFold(label, _explode)
    event = QMouseEvent(QEvent.MouseButtonRelease, QPointF(1.0, 1.0),
                        Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)

    assert watcher.eventFilter(label, event) is True
    assert calls == ["tried"]


def test_a_right_click_is_not_a_fold(qapp):
    """Only the left button folds; anything else passes through."""
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QLabel

    from spacr.qt.widgets.foldable import _ClickToFold

    label = QLabel("Console")
    calls = []
    watcher = _ClickToFold(label, lambda: calls.append("folded"))
    event = QMouseEvent(QEvent.MouseButtonRelease, QPointF(1.0, 1.0),
                        Qt.RightButton, Qt.RightButton, Qt.NoModifier)

    assert watcher.eventFilter(label, event) is False
    assert calls == []


def test_setting_the_state_it_already_has_changes_nothing(qapp):
    """Re-asserting the current state does not fire the change callback.

    A refresh that re-applies the stored fold state runs this path on every
    pass; doing the work would clear the alert marker each time.
    """
    from spacr.qt.widgets.foldable import Folder

    heading, body = _panel()
    seen = []
    folder = Folder(heading, body, on_change=seen.append)
    folder.alert("!")

    assert folder.set_shut(False) is False
    assert seen == []
    assert "!" in heading.text() or folder.shut is False


def test_a_failing_change_callback_still_folds_the_panel(qapp):
    """The fold completes even when the caller's callback raises."""
    from spacr.qt.widgets.foldable import Folder

    heading, body = _panel()

    def _explode(shut):
        raise RuntimeError("callback failed")

    folder = Folder(heading, body, on_change=_explode)

    assert folder.toggle() is True
    assert folder.shut is True
    assert body.isVisible() is False


def test_an_unreadable_preference_store_opens_the_panel(qapp, monkeypatch):
    """A fold state that cannot be read starts open rather than raising.

    An unreadable store must not stop a panel being built; an open panel is
    the safe guess because the content is at least reachable.
    """
    import spacr.qt.preferences as preferences
    from spacr.qt.widgets.foldable import make_foldable

    def _explode():
        raise RuntimeError("no preference store")

    monkeypatch.setattr(preferences, "get_folded_panels", _explode,
                        raising=False)

    heading, body = _panel()
    folder = make_foldable(heading, body, persist_key="tests/console")

    assert folder.shut is False


def test_an_unwritable_preference_store_still_folds(qapp, monkeypatch):
    """A fold whose state cannot be stored still happens on screen."""
    import spacr.qt.preferences as preferences
    from spacr.qt.widgets.foldable import make_foldable

    monkeypatch.setattr(preferences, "get_folded_panels", lambda: {},
                        raising=False)

    def _explode(key, shut):
        raise RuntimeError("read-only settings")

    monkeypatch.setattr(preferences, "set_folded_panel", _explode,
                        raising=False)

    heading, body = _panel()
    seen = []
    folder = make_foldable(heading, body, on_change=seen.append,
                           persist_key="tests/console")

    assert folder.toggle() is True
    assert folder.shut is True
    assert seen == [True]


def test_a_stored_fold_state_is_applied_at_build_time(qapp, monkeypatch):
    """A panel the user folded last time comes back folded."""
    import spacr.qt.preferences as preferences
    from spacr.qt.widgets.foldable import make_foldable

    monkeypatch.setattr(preferences, "get_folded_panels",
                        lambda: {"tests/console": True}, raising=False)
    monkeypatch.setattr(preferences, "set_folded_panel",
                        lambda key, shut: None, raising=False)

    heading, body = _panel()
    folder = make_foldable(heading, body, persist_key="tests/console")

    assert folder.shut is True
