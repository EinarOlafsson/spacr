"""The busy indicator when the things it watches are not there.

The spinner reads a preference for its delay and the process-wide run
registry for its state, and it installs itself next to a screen's *Clear
console* button. None of those is guaranteed: a preference store can refuse,
the registry can be absent in a stripped process, and a screen can be rebuilt
under the spinner already attached to it. Every one of those has to leave a
screen that opens with no spinner rather than a screen that will not open.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


from spacr.qt.widgets import activity_spinner as AS      # noqa: E402
from spacr.qt.widgets.activity_spinner import (          # noqa: E402
    ActivitySpinner)


def test_a_preference_that_cannot_be_read_gives_the_shipped_delay(
        monkeypatch):
    """A spinner that cannot find its setting is a spinner on the default,
    never a screen that refuses to open."""
    from spacr.qt import preferences

    def _explode():
        raise RuntimeError("no preference store here")

    monkeypatch.setattr(preferences, "get_spinner_delay", _explode)
    assert AS._preferred_delay_ms() == 2000


def test_a_spinner_that_cannot_find_the_registry_stops_watching(
        qtbot, monkeypatch):
    """Without the registry there is no state to follow. The spinner turns
    itself off rather than raising from the constructor of every screen that
    carries one."""
    from spacr.qt import bridge

    def _explode():
        raise RuntimeError("no registry in this process")

    monkeypatch.setattr(bridge, "registry", _explode)
    spinner = ActivitySpinner(auto=True, delay_ms=0)
    qtbot.addWidget(spinner)
    assert spinner._auto is False
    assert spinner.is_busy() is False


def test_a_registry_that_fails_mid_session_reports_no_running_work(
        qtbot, monkeypatch):
    """The registry is read on every state change. One that starts failing
    must read as "nothing running", which stops the spinner, rather than
    propagating out of a Qt slot."""
    spinner = ActivitySpinner(auto=True, delay_ms=0)
    qtbot.addWidget(spinner)
    from spacr.qt import bridge

    def _explode():
        raise RuntimeError("registry went away")

    monkeypatch.setattr(bridge, "registry", _explode)
    assert spinner._running_handles() == []
    assert spinner.is_busy() is False


def test_the_delay_can_be_changed_and_is_never_negative(qtbot):
    """The preference is in seconds and arrives as a float; a negative value
    would arm a timer that fires immediately and defeat the delay."""
    spinner = ActivitySpinner(auto=False, delay_ms=0)
    qtbot.addWidget(spinner)
    spinner.set_delay_ms(1500)
    assert spinner.delay_ms() == 1500
    spinner.set_delay_ms(-5)
    assert spinner.delay_ms() == 0


