"""What is left of a retired theme, and the three guarantees the dialog leans on.

Every branch this file is about is one of the module's own defensive
re-checks, and each one is only defensive because something a few lines
earlier already settled the question. This file drives the settled side and
asserts what it settles, so the "cannot happen" side is a claim with a test
behind it rather than an assumption:

* ``prefs/theme`` may still say ``space`` -- the theme was retired and its
  key was not -- and :func:`get_theme` answers with a theme this build has,
  which is why :func:`get_theme_choice` can never take its ``space`` arm;
* the dialog's button box is built with Save and Cancel, so asking it for
  them answers with buttons;
* the button box is added to the dialog's own layout, so looking for its row
  in that layout finds one -- and the hint strip goes above it;
* the Quit button's handler is only ever handed the dialog, and a QWidget
  always has a window, so neither of its ``is not None`` guards can fail.

The proofs for the four dead arms are written beside the tests that pin
them, in the file that would have to change first if any of them came back.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings

from spacr.qt import preferences as prefs

pytestmark = pytest.mark.qt


@pytest.fixture
def store(qapp, tmp_path, monkeypatch):
    """A throwaway INI store, so no test touches the real preferences."""
    settings = QSettings(str(tmp_path / "spacr-qt.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: settings)
    monkeypatch.setattr(prefs, "_SAFE_MODE", False)
    assert str(tmp_path) in prefs._settings().fileName()
    return settings


def _dialog(qtbot):
    """A built Preferences dialog, registered for teardown."""
    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    return dialog


# ---------------------------------------------------------------------------
# The Space theme, which was taken out from under a key that is still there
# ---------------------------------------------------------------------------

def test_a_store_still_naming_the_retired_space_theme_reads_as_a_theme_we_have(
        store):
    """``prefs/theme`` is a string an existing install already holds, and one
    of the values it holds is ``space`` -- the theme is gone, the settings
    file it was written into is not. `get_theme` answers such a store with
    :data:`DEFAULT_THEME`, so the composite token the dialog reads is one of
    this build's own, and Preferences opens on a row the user can see.

    That fallback is also the PROOF for two arms of this module that no
    input can reach any more, and the reason they are pinned here rather
    than contorted into coverage:

    * `get_theme_choice`'s ``if theme == "space"`` cannot be true, because
      ``space`` is not in :data:`VALID_THEMES` and `get_theme` returns
      ``raw if raw in VALID_THEMES else DEFAULT_THEME`` -- there is no
      stored value, hand-edited or migrated, that gets ``"space"`` past it.
    * `set_theme_choice`'s ``if choice.startswith("space:")`` cannot be
      reached, because the line above it refuses anything `theme_choices`
      does not offer, and `theme_choices` offers dark, light, glass, system
      and the ``cell:`` variants -- no ``space:`` token at all.

    Both are dead code left by the retirement, and the module's own comment
    at :data:`VALID_THEMES` -- "an existing install has ``prefs/theme`` set
    to one of dark/light/system/space; those keep resolving exactly as
    before" -- is no longer true of the fourth one: it resolves to the
    default now, which is what this test says out loud.
    """
    store.setValue(prefs._KEY_THEME, "space")

    assert "space" not in prefs.VALID_THEMES
    assert prefs.get_theme() == prefs.DEFAULT_THEME
    assert prefs.get_theme_choice() == prefs.DEFAULT_THEME

    # A theme this build DOES have still composes its variant into the
    # token, which is what says the answer above is about ``space`` and not
    # about `get_theme_choice` having stopped composing anything.
    store.setValue(prefs._KEY_THEME, "cell")
    assert prefs.get_theme_choice() == f"cell:{prefs.get_cell_variant()}"

    tokens = [token for _label, token in prefs.theme_choices()]
    assert not [token for token in tokens if token.startswith("space:")]
    with pytest.raises(ValueError, match="unknown theme choice 'space:"):
        prefs.set_theme_choice(f"space:{prefs.get_space_variant()}")

    # And a token that IS offered is taken, so the refusal above is the
    # missing choice and not a setter that refuses everything.
    cell_token = next(token for token in tokens
                      if token.startswith("cell:"))
    prefs.set_theme_choice(cell_token)
    assert prefs.get_theme() == "cell"
    assert prefs.get_theme_choice() == cell_token


# ---------------------------------------------------------------------------
# The foot of the dialog: the strip, then the buttons
# ---------------------------------------------------------------------------

def test_the_hint_strip_sits_directly_above_the_row_of_buttons(store, qtbot):
    """Asked for on 2026-08-28: the explanation goes ABOVE the buttons.
    Appended, it reads as a footnote to Save and Cancel rather than as the
    answer to the control the pointer is on, and it sits furthest from the
    tabs it describes.

    The insertion is by row -- ``layout.indexOf(buttons)`` -- and the
    fallback beside it (append when that row is not found) cannot run: the
    button box is added to ``outer``, which is the layout ``QVBoxLayout(dlg)``
    installed on the dialog, and ``dlg.layout()`` is that same object. A
    widget in a layout always has an index in it, so the append arm is dead
    for as long as those two are the same layout -- which is what the last
    two assertions here check.
    """
    from PySide6.QtWidgets import QDialogButtonBox, QPushButton

    from spacr.qt.widgets.hint_bar import HintBar

    dialog = _dialog(qtbot)
    layout = dialog.layout()
    buttons = dialog.findChild(QDialogButtonBox)
    assert buttons is not None

    row = layout.indexOf(buttons)
    assert row > 0, "the buttons are in the dialog's own layout"
    assert layout.itemAt(layout.count() - 1).widget() is buttons
    above = layout.itemAt(row - 1).widget()
    assert isinstance(above, HintBar)

    # The strip is the dialog's own, and it is the bar the finished dialog
    # was swept into: nothing is left holding a tooltip, so a second sweep
    # moves nothing. A control added afterwards is moved into that same
    # strip, which is what says the zero above is an emptied dialog and not
    # a sweep that has stopped finding anything.
    assert above.parent() is dialog
    assert prefs._everything_explains_itself_in_the_strip(dialog, above) == 0

    latecomer = QPushButton("Added after the sweep", dialog)
    latecomer.setToolTip("What this button would do.")
    assert prefs._everything_explains_itself_in_the_strip(dialog, above) == 1
    assert latecomer.toolTip() == ""


def test_the_button_box_answers_for_the_two_buttons_it_was_built_with(
        store, qtbot):
    """The box is constructed ``Save | Cancel``, so ``buttons.button(...)``
    returns a button for each: the ``is not None`` guard around the two
    ``setText`` calls can never take its other arm. Rather than leave that
    as an assumption, this asks the finished box for both and reads the
    words off them -- the words this build chose, through ``tr``, not Qt's.

    Reset is checked beside them because it is the one button whose
    PLACEMENT is the point: ``ResetRole`` is what puts it on the left, away
    from the two that close the dialog, which is what stops it being hit by
    muscle memory aimed at Cancel.
    """
    from PySide6.QtWidgets import QDialogButtonBox, QPushButton

    dialog = _dialog(qtbot)
    buttons = dialog.findChild(QDialogButtonBox)
    save = buttons.button(QDialogButtonBox.Save)
    cancel = buttons.button(QDialogButtonBox.Cancel)

    assert save is not None and cancel is not None
    assert save.text() == "Save" and cancel.text() == "Cancel"

    reset = dialog.findChild(QPushButton, "PreferencesReset")
    assert buttons.buttonRole(reset) == QDialogButtonBox.ResetRole
    assert buttons.buttonRole(save) == QDialogButtonBox.AcceptRole
    assert buttons.buttonRole(cancel) == QDialogButtonBox.RejectRole


# ---------------------------------------------------------------------------
# Quitting, which is handed the dialog and nothing else
# ---------------------------------------------------------------------------

def test_quitting_cancels_the_running_work_and_closes_the_window_it_is_on(
        store, qtbot, monkeypatch):
    """The graceful path cancels whatever the window has running, then
    closes it, so a quit from Preferences still runs every shutdown hook.
    This drives it with a registry that HAS work, which is the case that
    distinguishes the graceful path from simply closing the dialog.

    It also settles the two ``is not None`` guards at the end of that
    handler. The Quit button's only connection is
    ``lambda: _quit_spacr(dlg)`` -- ``dlg`` is the QDialog built a few
    hundred lines above and is never None -- and ``parent.window()`` is
    QWidget's own, which walks up to a top-level widget and answers with the
    widget itself when there is none above it. So neither ``parent`` nor
    ``window`` can be None here, and the two skips are dead.
    """
    from PySide6.QtWidgets import QDialog, QPushButton

    from spacr.qt import shutdown

    class _Handle:
        app_key = "segmentation"

        def elapsed(self):
            return 180.0

    class _Registry:
        def __init__(self):
            self.cancelled = []
            self.running = [_Handle()]

        def active(self):
            return list(self.running)

        def cancel_all(self, reason=""):
            self.cancelled.append(reason)
            self.running = []

    asked = []
    monkeypatch.setattr(shutdown, "ask_how_to_quit",
                        lambda *args, **kwargs: asked.append(kwargs)
                        or shutdown.GRACEFUL)
    monkeypatch.setattr(shutdown, "force_quit_now",
                        lambda: pytest.fail("the graceful path force-quit"))

    dialog = _dialog(qtbot)
    window = dialog.window()
    assert window is not None, "a QWidget always has a window"
    registry = _Registry()
    window._runs = registry

    dialog.show()
    dialog.findChild(QPushButton, "QuitSpacrButton").click()

    # What was running is named in the question, cancelled with the reason
    # this caller gives, and the window is gone afterwards.
    assert len(asked) == 1
    assert "segmentation — running for 3 min" in asked[0]["detail"]
    assert registry.cancelled == ["quit from Preferences"]
    assert dialog.result() == QDialog.Accepted
    assert not window.isVisible()
