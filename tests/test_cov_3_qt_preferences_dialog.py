"""The Preferences window's own controls, and the way out of a wedged app.

Three things here are only reachable through the built dialog: a console log
switch that must not survive its file switch being turned off, a Reset that
walks every control back through the real getters, and the Quit button that
exists for the case `closeEvent` cannot handle -- a worker wedged in a C
extension, which leaves the window refusing to close with no way out from
inside the application.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")


from spacr.qt import preferences as prefs                       # noqa: E402
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


# ---------------------------------------------------------------------------
# Quitting
# ---------------------------------------------------------------------------

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

