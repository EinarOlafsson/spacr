"""The Preferences window's own controls, and the way out of a wedged app.

Three things here are only reachable through the built dialog: a console log
switch that must not survive its file switch being turned off, a Reset that
walks every control back through the real getters, and the Quit button that
exists for the case `closeEvent` cannot handle -- a worker wedged in a C
extension, which leaves the window refusing to close with no way out from
inside the application.
"""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtGui import QColor, QPalette                      # noqa: E402
from PySide6.QtWidgets import QComboBox, QPushButton            # noqa: E402

from spacr.qt import preferences as prefs                       # noqa: E402
from spacr.qt import shutdown                                   # noqa: E402
from spacr.qt.widgets.toggle import Toggle                      # noqa: E402


@pytest.fixture()
def dialog(qapp, qtbot):
    dlg = prefs.PreferencesDialog()
    qtbot.addWidget(dlg)
    yield dlg


def _toggle(dialog, name) -> Toggle:
    found = dialog.findChild(Toggle, name)
    assert found is not None, f"no {name} toggle on the Preferences dialog"
    return found


# ---------------------------------------------------------------------------
# Log levels
# ---------------------------------------------------------------------------

def test_a_console_level_cannot_outlive_the_file_level_it_depends_on(dialog):
    """The console stream is filtered out of what reaches the log file, so a
    console switch left on while its file switch is off would promise a
    level that nothing ever emits."""
    file_toggle = _toggle(dialog, "LogFileLevelInfo")
    console_toggle = _toggle(dialog, "LogConsoleLevelInfo")

    file_toggle.setChecked(True)
    console_toggle.setChecked(True)
    assert console_toggle.isEnabled() is True

    file_toggle.setChecked(False)

    assert console_toggle.isEnabled() is False
    assert console_toggle.isChecked() is False, (
        "the console level stayed on after its file level was switched off")


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def test_resetting_a_control_whose_default_is_unknown_leaves_it_alone(
        dialog, monkeypatch):
    """A getter with no answer must not point the combo at index -1, which
    is a blank entry the user cannot get out of without knowing what it
    used to say."""
    monkeypatch.setattr(prefs, "get_language", lambda: None)
    reset = next(button for button in dialog.findChildren(QPushButton)
                 if button.objectName() == "PreferencesReset")
    combo = dialog.findChild(QComboBox, "LanguagePreference")
    before = combo.currentIndex()

    reset.click()

    assert combo.currentIndex() == before >= 0


# ---------------------------------------------------------------------------
# Quitting
# ---------------------------------------------------------------------------

def _quit_button(dialog) -> QPushButton:
    button = dialog.findChild(QPushButton, "QuitSpacrButton")
    assert button is not None, "the Preferences dialog offers no way to quit"
    return button


def test_declining_the_quit_dialog_leaves_the_window_open(dialog,
                                                          monkeypatch):
    """Cancel has to mean cancel: nothing is asked to stop and the window
    the user was working in is still there."""
    forced = []
    monkeypatch.setattr(shutdown, "ask_how_to_quit",
                        lambda *a, **k: shutdown.CANCEL)
    monkeypatch.setattr(shutdown, "force_quit_now",
                        lambda *a, **k: forced.append(True))
    dialog.show()

    _quit_button(dialog).click()

    assert forced == []
    assert dialog.isVisible() is True


def test_choosing_to_stop_immediately_leaves_without_waiting(dialog,
                                                             monkeypatch):
    """The point of this button is the wedged run that will never see a
    cancel flag, so the immediate path must not go through the watcher."""
    forced = []
    monkeypatch.setattr(shutdown, "ask_how_to_quit",
                        lambda *a, **k: shutdown.FORCE)
    monkeypatch.setattr(shutdown, "force_quit_now",
                        lambda *a, **k: forced.append(True))
    dialog.show()

    _quit_button(dialog).click()

    assert forced == [True]


def test_choosing_to_finish_the_step_closes_the_window_and_watches(
        dialog, monkeypatch):
    """The graceful path runs the same shutdown hooks a normal close does,
    and leaves a watcher behind so a run that never stops can be asked
    again rather than hanging silently."""
    started = []
    forced = []
    cancelled = []

    class Watcher:
        def __init__(self, *args, **kwargs):
            self.describe = kwargs.get("describe")

        def start(self):
            started.append(True)

    class Registry:
        def active(self):
            return ["mask on plate1"]

        def cancel_all(self, reason=""):
            cancelled.append(reason)

    dialog._runs = Registry()
    monkeypatch.setattr(shutdown, "ask_how_to_quit",
                        lambda *a, **k: shutdown.GRACEFUL)
    monkeypatch.setattr(shutdown, "force_quit_now",
                        lambda *a, **k: forced.append(True))
    monkeypatch.setattr(shutdown, "GracefulQuitWatcher", Watcher)
    dialog.show()

    _quit_button(dialog).click()

    assert started == [True]
    assert forced == []
    assert cancelled and "quit" in cancelled[0].lower()
    assert dialog.isVisible() is False


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------

def test_a_card_that_will_not_reread_does_not_stop_the_others(qapp, qtbot):
    """Every card on screen has to take a rim change, so one card that
    throws must be counted out rather than ending the pass."""
    from spacr.qt.widgets.setup_card import SetupCard

    healthy, broken = SetupCard(), SetupCard()
    qtbot.addWidget(healthy)
    qtbot.addWidget(broken)

    def refuse():
        raise RuntimeError("this card will not reread")

    broken.reread_the_preferences = refuse

    told = prefs._tell_the_cards_the_rim_changed()

    assert told >= 1, "the healthy card was never told"


# ---------------------------------------------------------------------------
# Following the system theme
# ---------------------------------------------------------------------------

@pytest.fixture()
def light_application_palette(qapp):
    """Put the application into a light palette and put it back after."""
    original = qapp.palette()
    palette = QPalette(original)
    palette.setColor(QPalette.ColorRole.Window, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#000000"))
    qapp.setPalette(palette)
    try:
        yield qapp
    finally:
        qapp.setPalette(original)


@pytest.mark.xfail(strict=True, reason=(
    "resolve_effective_theme reads QPalette.Window off the palette INSTANCE, "
    "which raises AttributeError on PySide6 6.11; the luminance test is "
    "never reached and 'Follow system' always resolves to dark"))
def test_following_the_system_theme_on_a_light_desktop_resolves_to_light(
        light_application_palette):
    """'Follow system' is the setting a user picks so spaCR matches their
    desktop. Resolving it to dark on a light desktop is the one outcome
    that makes the setting useless, and it fails silently because the
    palette poll is wrapped in a bare except."""
    prefs.set_theme("system")

    assert prefs.resolve_effective_theme() == "light"


def test_following_the_system_theme_still_produces_a_palette_name(
        light_application_palette):
    """Whatever it resolves to, it has to be a palette the stylesheet can
    be built from -- 'system' itself is not one."""
    prefs.set_theme("system")

    assert prefs.resolve_effective_theme() in prefs.PALETTE_THEMES


def test_the_level_of_a_stored_theme_that_is_not_a_theme_is_the_default():
    """A store written by a newer spaCR can name a theme this build cannot
    render; it falls back rather than handing the stylesheet an unknown."""
    prefs._settings().setValue(prefs._KEY_THEME, "aurora")

    assert prefs.get_theme() == prefs.DEFAULT_THEME
