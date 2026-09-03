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


@pytest.fixture(autouse=True)
def _put_the_singleton_back(qapp):
    """Hide the shared popup after every test in this file.

    `HoverTooltip.instance()` is a SINGLETON, and several tests here drive a
    hover with the bottom strip switched off -- which is precisely the case
    that opens the popup. Left showing, it leaks into whatever file runs
    next: `test_no_information_dots.py` counts visible popups and saw three
    failures that were this file's fault and looked like its own.
    """
    yield
    from spacr.qt.widgets.hover_tooltip import HoverTooltip

    popup = HoverTooltip.instance()
    popup.hide()
    qapp.processEvents()


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


def test_the_strip_offers_an_animation_word_only_when_there_is_one(
        qtbot, qt_theme_applied):
    """371 asks for the links on BOTH surfaces: "an API link and Annimation
    link text ... same for the botom tooltips".

    Only when the setting has one. 141 do, and a word that visibly does
    nothing is worse than no word -- the rule the popup's own footer already
    follows.
    """
    scr = _screen(qtbot)

    scr._write_hint("what it does", "https://example.test/x", animated=True)
    assert ">Animation<" in scr._hint_strip.text()

    scr._write_hint("what it does", "https://example.test/x", animated=False)
    assert ">Animation<" not in scr._hint_strip.text()
    assert ">API<" in scr._hint_strip.text()


def test_the_animation_word_is_not_a_url_a_browser_could_open(
        qtbot, qt_theme_applied):
    """The strip has `setOpenExternalLinks(True)` so its API link works, and
    anything resembling a real address would be handed to a browser."""
    from spacr.qt.screens.app_screen import _HINT_ANIMATION_HREF

    scr = _screen(qtbot)
    scr._write_hint("x", "https://example.test/x", animated=True)

    assert _HINT_ANIMATION_HREF in scr._hint_strip.text()
    assert not _HINT_ANIMATION_HREF.startswith(("http", "file", "//"))


def test_pressing_the_animation_word_opens_the_box_for_that_setting(
        qtbot, qt_theme_applied, monkeypatch):
    """The strip hands off rather than duplicating: it is four lines at the
    bottom of the window and the popup already has a column sized for the
    square."""
    from spacr.qt.screens.app_screen import _HINT_ANIMATION_HREF
    from spacr.qt.widgets import hover_tooltip

    scr = _screen(qtbot)
    widget = _a_setting_widget(scr)
    shown, toggled = [], []

    # PATCHED ON THE CLASS, NOT THE INSTANCE, and that is not a style choice.
    # `HoverTooltip.instance()` is a process-wide singleton, and
    # `monkeypatch.setattr(popup, "show_for", ...)` reads the BOUND METHOD
    # off the class and restores it as an INSTANCE attribute -- which then
    # shadows the class for the rest of the session. `test_no_information_
    # dots.py` replaces `HoverTooltip.show_for` on the class and counts the
    # calls; with an instance attribute in the way its replacement never
    # ran, and three of its tests failed in a file that had already finished.
    monkeypatch.setattr(hover_tooltip.HoverTooltip, "show_for",
                        lambda self, anchor, html, *a, **k: shown.append(anchor))
    monkeypatch.setattr(hover_tooltip.HoverTooltip, "toggle_animation",
                        lambda self: toggled.append(True))

    scr._hinted_widget = widget
    scr._hinted_html = "<b>x</b>"
    scr._on_hint_link(_HINT_ANIMATION_HREF)

    assert shown == [widget], "the box was not opened for the hovered setting"
    assert toggled == [True], "the animation was not revealed"


def test_an_unknown_link_target_does_nothing(qtbot, qt_theme_applied):
    """The API link goes through Qt's external handler; only the private
    scheme is ours, and anything else must be ignored rather than guessed
    at."""
    scr = _screen(qtbot)
    scr._hinted_widget = _a_setting_widget(scr)
    scr._hinted_html = "<b>x</b>"
    scr._on_hint_link("https://example.test/somewhere")     # must not raise
