"""Item 288: the application event hub when there is nothing safe to talk to.

`spacr.qt.gil_priority` puts every application-wide event filter spaCR
installs behind one hub on ``QApplication`` (item 284). The contract the
hub keeps for live watchers is pinned in
``test_the_theme_does_not_lag_while_a_run_is_going.py``. What is pinned
here is the other half: no application, no watcher, a hub or an application
whose C++ half is gone, Qt's bindings not importable, a hub that raises
while being asked. In each case the answer is "not watching" -- ``None``,
``False`` or an empty tuple -- and never a call into a freed object.
"""
from __future__ import annotations

import sys

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QObject                    # noqa: E402

from spacr.qt import gil_priority                             # noqa: E402
from spacr.qt.gil_priority import (                           # noqa: E402
    _HUB_ATTRIBUTE, _application_event_hub, _application_event_hub_class,
    _application_watchers, _stop_watching_application_events,
    _watch_application_events)


def test_the_hub_class_is_made_once_and_reused(qapp):
    first = _application_event_hub_class()
    assert _application_event_hub_class() is first
    assert issubclass(first, QObject)


def test_an_event_that_cannot_say_its_type_is_passed_on(qapp):
    hub = _application_event_hub_class()()
    try:
        calls = []

        class _Recorder(QObject):
            def eventFilter(self, watched, event):            # noqa: N802
                calls.append(event)
                return True

        watcher = _Recorder()
        hub.add(watcher, (QEvent.Type.User,))

        class _Typeless:
            def type(self):
                raise RuntimeError("the event was already freed")

        assert hub.eventFilter(None, _Typeless()) is False
        assert calls == []
        assert hub.eventFilter(None, QEvent(QEvent.Type.User)) is True
        assert len(calls) == 1
    finally:
        hub.deleteLater()


def test_no_application_has_no_hub():
    assert _application_event_hub(None) is None
    assert _application_watchers(None) == ()


def test_without_shiboken_there_is_no_hub(qapp, monkeypatch):
    monkeypatch.setitem(sys.modules, "shiboken6", None)
    assert _application_event_hub(qapp) is None


def test_a_hub_whose_object_was_destroyed_is_not_used(qapp, monkeypatch):
    from shiboken6 import delete

    dead = QObject()
    delete(dead)
    monkeypatch.setattr(qapp, _HUB_ATTRIBUTE, dead, raising=False)
    assert _application_event_hub(qapp, create=False) is None
    assert _application_watchers(qapp) == ()


def test_an_application_being_torn_down_gets_no_new_hub(qapp, monkeypatch):
    import shiboken6

    monkeypatch.setattr(qapp, _HUB_ATTRIBUTE, None, raising=False)
    real_is_valid = shiboken6.isValid
    monkeypatch.setattr(shiboken6, "isValid",
                        lambda obj: obj is not qapp and real_is_valid(obj))
    assert _application_event_hub(qapp) is None
    assert getattr(qapp, _HUB_ATTRIBUTE) is None


def test_nothing_to_watch_or_nothing_to_watch_with_is_refused(qapp):
    watcher = QObject()
    kinds = (QEvent.Type.User,)
    assert _watch_application_events(None, watcher, kinds) is False
    assert _watch_application_events(qapp, None, kinds) is False
    assert _stop_watching_application_events(None, watcher) is False
    assert _stop_watching_application_events(qapp, None) is False
    assert watcher not in _application_watchers(qapp)


class _FailingHub:
    """A hub whose C++ half went away between being found and being asked."""

    def discard(self, _watcher):
        raise RuntimeError("Internal C++ object already deleted.")

    def watchers(self):
        raise RuntimeError("Internal C++ object already deleted.")


def test_a_hub_that_dies_while_being_asked_reports_not_watching(
        qapp, monkeypatch):
    monkeypatch.setattr(gil_priority, "_application_event_hub",
                        lambda app, create=True: _FailingHub())
    assert _stop_watching_application_events(qapp, QObject()) is False
    assert _application_watchers(qapp) == ()
