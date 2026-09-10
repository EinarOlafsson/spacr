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

from PySide6.QtWidgets import (                          # noqa: E402
    QGridLayout, QHBoxLayout, QPushButton, QWidget)

from spacr.qt.widgets import activity_spinner as AS      # noqa: E402
from spacr.qt.widgets.activity_spinner import (          # noqa: E402
    ActivitySpinner, attach_activity_spinner)


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


def _row_with_clear_button(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    layout = QHBoxLayout(host)
    button = QPushButton("Clear console", host)
    host._btn_clear = button
    layout.addWidget(button)
    return host, button


def test_attaching_twice_returns_the_spinner_already_there(qtbot):
    """``showEvent`` runs every time a screen is shown. A second spinner in
    the row would be two indicators disagreeing about the same work."""
    host, _button = _row_with_clear_button(qtbot)
    first = attach_activity_spinner(host)
    assert isinstance(first, ActivitySpinner)
    assert attach_activity_spinner(host) is first


def test_a_descendant_finds_the_spinner_its_ancestor_already_owns(qtbot):
    """Called from a child of the screen, the walk up must stop at the
    spinner that is already installed instead of continuing past it and
    adding a second one."""
    host, _button = _row_with_clear_button(qtbot)
    spinner = attach_activity_spinner(host)
    child = QWidget()
    qtbot.addWidget(child)
    child.setParent(host)
    del host._btn_clear          # the button is gone; the spinner is not
    assert attach_activity_spinner(child) is spinner


def test_a_spinner_whose_c_half_is_gone_is_replaced(qtbot):
    """A rebuilt screen leaves a Python reference to a deleted widget. Using
    it would raise on the next call; a fresh one is installed instead."""
    import shiboken6

    host, _button = _row_with_clear_button(qtbot)
    stale = attach_activity_spinner(host)
    shiboken6.delete(stale)
    fresh = attach_activity_spinner(host)
    assert isinstance(fresh, ActivitySpinner)
    assert fresh is not stale


def test_a_button_in_no_layout_gets_no_spinner(qtbot):
    """There is nowhere to put it. Returning None is what lets the screen
    open exactly as it did before."""
    host = QWidget()
    qtbot.addWidget(host)
    row = QWidget(host)
    button = QPushButton("Clear console", row)
    host._btn_clear = button
    assert attach_activity_spinner(host) is None


def test_a_row_that_is_not_a_box_layout_gets_no_spinner(qtbot):
    """``insertWidget`` is a box-layout method. A grid has no "next to this
    one" position, so nothing sensible can be inserted and the caller gets
    None with no stray widget left parented to the row."""
    host = QWidget()
    qtbot.addWidget(host)
    row = QWidget(host)
    layout = QGridLayout(row)
    button = QPushButton("Clear console", row)
    layout.addWidget(button, 0, 0)
    host._btn_clear = button

    assert attach_activity_spinner(host) is None
    assert not row.findChildren(ActivitySpinner)
