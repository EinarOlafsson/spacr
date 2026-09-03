"""The bottom strip holds the LAST setting, so its API link can be reached.

Instruction 371, part 3, and the request explains its own reason: "for the
user to be able to press the botom tooltip API link (which should also just
say API) the last setting the mouse hovered over should be shown, not only
when the mouse hovers the setting. this way the user can hover then move the
mouse to the link and click it, which is otherwise not possible. the tooltip
should be up for 10 seconds before it disapears (or if another setting is
hovered.)"

BLANKING ON LEAVE MADE THE LINK UNREACHABLE BY CONSTRUCTION. It appeared only
while the pointer was on the setting, and moving toward it removed it. That
is not a timing bug to be tuned; it is a contradiction, and this file exists
to stop it being reintroduced by anything that looks like tidying up.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent                              # noqa: E402


def _screen(qtbot, key="mask"):
    from spacr.qt.screens.app_screen import AppScreen

    scr = AppScreen(app_key=key)
    qtbot.addWidget(scr)
    return scr


def _a_setting_widget(scr):
    """The first widget on the form that names a setting."""
    from PySide6.QtWidgets import QWidget

    for child in scr.findChildren(QWidget):
        if child.property("settingKey"):
            return child
    pytest.skip("this screen exposes no setting widgets")


def test_the_strip_still_holds_the_setting_after_the_pointer_leaves(
        qtbot, qt_theme_applied):
    scr = _screen(qtbot)
    widget = _a_setting_widget(scr)

    scr.eventFilter(widget, QEvent(QEvent.Type.Enter))
    held = scr._hint_strip.text()
    assert held and held != scr._default_hint(), (
        "hovering a setting did not write it to the strip")

    scr.eventFilter(widget, QEvent(QEvent.Type.Leave))

    assert scr._hint_strip.text() == held, (
        "the strip was blanked when the pointer left, so the API link in it "
        "can never be reached -- moving toward the link removes it")


def test_the_hold_is_ten_seconds_and_is_running(qtbot, qt_theme_applied):
    scr = _screen(qtbot)
    widget = _a_setting_widget(scr)

    scr.eventFilter(widget, QEvent(QEvent.Type.Enter))
    timer = scr._hint_hold_timer

    assert timer.isActive(), "no hold was started, so nothing will clear it"
    assert scr.HINT_HOLD_MS == 10_000
    assert timer.interval() == 10_000


def test_the_hold_restarts_on_the_next_setting(qtbot, qt_theme_applied):
    """Reading down a form must not be a race against a clock started by the
    first row."""
    scr = _screen(qtbot)
    widget = _a_setting_widget(scr)

    scr.eventFilter(widget, QEvent(QEvent.Type.Enter))
    scr._hint_hold_timer.setInterval(9_000)          # pretend time passed
    scr.eventFilter(widget, QEvent(QEvent.Type.Enter))

    assert scr._hint_hold_timer.interval() == 10_000


def test_the_strip_comes_back_to_its_prompt_when_the_hold_runs_out(
        qtbot, qt_theme_applied):
    scr = _screen(qtbot)
    widget = _a_setting_widget(scr)

    scr.eventFilter(widget, QEvent(QEvent.Type.Enter))
    scr._release_the_hint()

    assert scr._hint_strip.text() == scr._default_hint()


def test_releasing_does_not_restart_its_own_hold(qtbot, qt_theme_applied):
    """`_release_the_hint` writes the default prompt, and the prompt is not
    empty. Inferring "hold" from "the text is not empty" would make the strip
    restart its own timer forever."""
    scr = _screen(qtbot)
    widget = _a_setting_widget(scr)

    scr.eventFilter(widget, QEvent(QEvent.Type.Enter))
    scr._release_the_hint()

    assert not scr._hint_hold_timer.isActive()


def test_the_link_says_api(qtbot, qt_theme_applied):
    """"which should also just say API". The long form repeated on every
    setting and the strip has four lines to spend."""
    scr = _screen(qtbot)
    scr._write_hint("what this setting does", "https://example.test/page")

    text = scr._hint_strip.text()
    assert ">API<" in text, f"the link is not the word API: {text!r}"
    assert "Open spaCR API documentation" not in text


def test_the_bottom_strip_obeys_its_switch(qtbot, qt_theme_applied,
                                           monkeypatch):
    """Cleared, a hovered setting writes nothing to the strip."""
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "get_tooltips_bottom_enabled",
                        lambda: False)
    scr = _screen(qtbot)
    widget = _a_setting_widget(scr)
    before = scr._hint_strip.text()

    scr.eventFilter(widget, QEvent(QEvent.Type.Enter))

    assert scr._hint_strip.text() == before
