"""Measure's action row fits its captions in every locale, at a real width.

Instruction 350's last open finding. The sweep in
``test_the_text_fits_sweep.py`` recorded it as six clipped captions on
Measure in German and five in Icelandic, and this file exists because that
sweep -- which builds a screen as a TOP-LEVEL widget at 1200x850 -- is the
wrong shape to hold this particular defect down. Two reasons, both measured:

* A TOP-LEVEL WIDGET GROWS OUT OF THE PROBLEM. ``QLayout::activate`` forces a
  window's minimum size up to its layout's total minimum unless the window
  has set one of its own, so a screen whose action row demanded 1092 px of
  minimum width did not clip: it became 1193 px wide when asked for 1000, and
  every caption then fitted. The screens in the application are not windows,
  they are pages in a ``QStackedWidget`` inside one, and a page gets whatever
  the window has -- less than its minimum included. That is when Qt shrinks
  every child below its hint and every caption loses its ending.
* AND AT 1200 PX THE DAMAGE WAS NOT IN THE CAPTIONS AT ALL. The row's
  minimum simply took the width from its neighbour: the settings column,
  which the screen's own ``setSizes([400, 800])`` asks to be a third of the
  body, was 251 px in English, 91 in Icelandic and 67 in German.

WHAT THE OLD LAYOUT DID, measured on this file's own fixture before the fix,
settled, in a 1000x850 window that could not grow:

    en   1 caption cut off   ("Submit remote…", 134 px against a 143 hint)
    de   4                   ("Einstellungen importieren…" 106 against 192)
    is   4                   ("Flytja inn stillingar…" 96 against 144)

and at 1200x850 the settings column measured 251 / 67 / 91 px in en / de / is
while the action row took 908 / 1092 / 1068.

WHAT IT DOES NOW: 0 / 0 / 0 clipped at 1000 px, the window is the width it
was asked for in all three locales, and the settings column is 389 px in all
three at 1200 -- the split the screen asked for in the first place.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import (QHBoxLayout, QPushButton,       # noqa: E402
                               QStackedWidget, QWidget)

from spacr.qt import i18n as I                                 # noqa: E402

from .test_the_text_fits_sweep import _fits, settle            # noqa: E402

#: Source, longest compounds, and the maintainer's own -- the same three the
#: sweep uses, for the same reasons, so the two files' numbers compare.
LOCALES = ("en", "de", "is")

#: A window a user really can have. The developers' windows are wide, which
#: is the whole reason this survived to be reported.
NARROW = 1000


def _screen_in_a_window(qtbot, app_key, width, height=850):
    """Build ``app_key`` the way the application holds it: as a PAGE.

    The screen goes into a ``QStackedWidget`` -- what ``MainWindow`` uses --
    and that container is given an EXPLICIT minimum size. Without the
    explicit minimum this proves nothing: ``QLayout::activate`` raises a
    window's minimum to its layout's total minimum, so the container would
    grow to 1193 px rather than let the screen inside it be squeezed, and
    the measurement would come back clean for a screen that cannot be shown
    at the size it was asked for.
    """
    from spacr.qt.screens.app_screen import AppScreen

    host = QStackedWidget()
    qtbot.addWidget(host)
    screen = AppScreen(app_key=app_key)
    host.addWidget(screen)
    host.setCurrentWidget(screen)
    host.setMinimumSize(1, 1)
    host.resize(width, height)
    host.show()
    qtbot.waitExposed(host)
    settle(qtbot, screen)
    return host, screen


def _action_buttons(screen):
    """The visible push buttons of the action row, in layout order."""
    return [button for button in screen._actions_row.findChildren(QPushButton)
            if button.isVisible()]


# ---------------------------------------------------------------------------
# The failure a user meets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("locale", LOCALES)
def test_no_action_button_is_cut_off_in_a_narrow_window(locale, qtbot,
                                                        qt_theme_applied,
                                                        monkeypatch):
    """Every caption in the row is drawn whole at 1000 px, in every locale.

    Built with ``qt_theme_applied`` deliberately: WITHOUT the application
    stylesheet the same screen reports every button at exactly its hint and
    nothing is clipped at all, which is how this defect stayed invisible to
    anyone reproducing it by hand.
    """
    monkeypatch.setenv(I.ENV_LANGUAGE, locale)
    _host, screen = _screen_in_a_window(qtbot, "measure", NARROW)

    cut = [f"{button.text()!r}: {_fits(button)}"
           for button in _action_buttons(screen) if _fits(button)]
    assert not cut, (
        f"measure in {locale} at {NARROW} px: {len(cut)} action captions cut "
        f"off -- {'; '.join(cut)}")


@pytest.mark.parametrize("locale", LOCALES)
def test_the_screen_fits_the_window_it_was_given(locale, qtbot,
                                                 qt_theme_applied,
                                                 monkeypatch):
    """A SECOND MEASUREMENT OF THE SAME THING, differently shaped.

    Instruction 350 records two confident wrong headlines that one
    measurement each made look plausible, and the rule it drew from them is
    that every number should be doubted until a differently-shaped
    measurement agrees. So: rather than ask whether captions fit, ask whether
    the screen can be the size it was asked for at all. A row whose minimum
    width exceeds the window is the CAUSE of the clipping in the test above,
    and it shows up here as a window that refuses to be narrow -- 1193 px in
    German and 1169 in Icelandic when 1000 was asked for, and 1009 even in
    English.

    This one uses the screen as a top-level widget on purpose: that is the
    form in which Qt reports the minimum by growing rather than by clipping.
    """
    monkeypatch.setenv(I.ENV_LANGUAGE, locale)
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(app_key="measure")
    qtbot.addWidget(screen)
    screen.resize(NARROW, 850)
    screen.show()
    qtbot.waitExposed(screen)
    settle(qtbot, screen)

    assert screen.width() <= NARROW, (
        f"measure in {locale} forced itself to {screen.width()} px when asked "
        f"for {NARROW}: the action row's minimum width is "
        f"{screen._actions_row.minimumSizeHint().width()}")


@pytest.mark.parametrize("locale", LOCALES)
def test_the_settings_column_is_not_starved_by_the_row(locale, qtbot,
                                                       qt_theme_applied,
                                                       monkeypatch):
    """The row must not take the settings column's width to fit itself.

    The threshold is the screen's own request, not a number picked here:
    ``_build_ui`` asks the body splitter for ``setSizes([400, 800])``, a
    third of the body for the settings. A QUARTER is that with slack for the
    margins and the handle, and it is the line between "the splitter got
    roughly what it asked for" and "the action row overruled it" -- which is
    what 67 px of settings column in German was.
    """
    monkeypatch.setenv(I.ENV_LANGUAGE, locale)
    _host, screen = _screen_in_a_window(qtbot, "measure", 1200)

    column = screen._settings_panel.width()
    assert column >= screen.width() // 4, (
        f"measure in {locale}: the settings column is {column} px of a "
        f"{screen.width()} px screen because the action row took "
        f"{screen._actions_row.width()}")


# ---------------------------------------------------------------------------
# The mechanism, not just the absence of the symptom
# ---------------------------------------------------------------------------

def test_the_strip_wraps_rather_than_shrinking_its_buttons(qtbot,
                                                           qt_theme_applied,
                                                           monkeypatch):
    """German at 1000 px: more than one line, and every button full width.

    Asserting only "nothing is clipped" would also pass if somebody made the
    captions shorter, or gave the buttons an ellipsis. The row was asked to
    WRAP -- squeezing is the one option that loses text, and eliding a
    six-character verb loses it just as thoroughly -- so this test says which
    of the three answers it got.
    """
    monkeypatch.setenv(I.ENV_LANGUAGE, "de")
    _host, screen = _screen_in_a_window(qtbot, "measure", NARROW)

    buttons = _action_buttons(screen)
    lines = {button.y() for button in buttons}
    assert len(lines) > 1, (
        f"the strip stayed on one line of {len(buttons)} buttons at "
        f"{NARROW} px in German, so it is not wrapping")
    for button in buttons:
        assert button.width() >= button.sizeHint().width(), (
            f"{button.text()!r} wrapped and was STILL squeezed: "
            f"{button.width()} px against a {button.sizeHint().width()} hint")


def test_the_check_still_notices_a_squeezed_row(qtbot, qt_theme_applied):
    """Proof the assertions above have teeth, which this item keeps needing.

    Three confident wrong headlines are on the record for instruction 350,
    and one of them was a sweep that failed twenty-five combinations without
    measuring anything. A test that cannot fail and a test that passes look
    identical from the summary line, so here is the old shape of the row --
    the same German captions in a plain ``QHBoxLayout`` in a box too small
    for them -- proving that ``_fits`` reports it.
    """
    captions = ["Ausführen", "Stopp", "Einstellungen importieren…",
                "Remote übermitteln…", "Konsole leeren", "Konsole kopieren"]
    # HELD BY SOMETHING THAT WILL NOT GROW, and the first draft of this test
    # forgot: shown as a top-level widget the box went straight to the 850 px
    # its layout asked for, six buttons all at their hints and nothing
    # flagged -- the same escape the real screen makes at 1193 px, on six
    # buttons instead of a whole module.
    host = QStackedWidget()
    qtbot.addWidget(host)
    box = QWidget()
    row = QHBoxLayout(box)
    row.setContentsMargins(0, 0, 0, 0)
    for caption in captions:
        row.addWidget(QPushButton(caption, box))
    host.addWidget(box)
    host.setCurrentWidget(box)
    host.setMinimumSize(1, 1)
    host.resize(400, 44)                    # the row wants roughly 850
    host.show()
    qtbot.waitExposed(host)
    settle(qtbot, box)

    squeezed = [button for button in box.findChildren(QPushButton)
                if _fits(button)]
    assert len(squeezed) >= 4, (
        "a QHBoxLayout of six German captions crushed into 400 px was not "
        f"reported as clipped ({len(squeezed)} of 6 flagged)")


# ---------------------------------------------------------------------------
# What the split had to leave alone
# ---------------------------------------------------------------------------

#: Everything outside this file that reaches into the action row by name.
#: `spacr.qt.chaining`, `spacr.qt.prerun` and `spacr.qt.preview_registry` use
#: `_actions_row` as the anchor they insert above; six test files name the
#: buttons; `spacr.qt.walkthrough` looks for `_btn_run`; `spacr.qt.theme`'s
#: surface sweep is handed `_actions_row` by `_clear_page_surfaces`.
ROW_ATTRIBUTES = ("_actions_row", "_btn_run", "_btn_stop", "_btn_import",
                  "_btn_remote", "_btn_clear", "_btn_copy_console",
                  "_btn_preferences", "_btn_file_issue", "_progress")

#: The two the wrap must never separate, and which therefore share a parent
#: of their own inside the strip. See the comment beside `copy_and_gear`.
PAIRED = ("_btn_copy_console", "_btn_preferences")


def test_the_split_left_every_reachable_name_where_it_was(qtbot,
                                                          qt_theme_applied):
    """The buttons moved into a sub-layout, NOT into a widget of their own.

    That distinction is the whole reason this shape was chosen over the
    obvious one: a widget added to a sub-layout is parented to the widget
    that owns the TOP-level layout, so every button is still a Qt child of
    ``_actions_row`` and every reach into this row still lands. A nested
    ``FlowHost`` would have reparented all eight and broken
    ``test_chaining_gui.py``'s ``screen._btn_run.parent() is
    screen._actions_row`` -- and it would have put a second anonymous
    container between the backdrop and the eye for
    ``_clear_page_surfaces`` to have to find.
    """
    _host, screen = _screen_in_a_window(qtbot, "measure", 1200)

    for name in ROW_ATTRIBUTES:
        assert getattr(screen, name, None) is not None, f"{name} is gone"

    row = screen._actions_row
    for name in ROW_ATTRIBUTES[1:]:
        widget = getattr(screen, name)
        if name in PAIRED:
            # Copy console and the gear are welded into one flow item so a
            # wrap cannot separate them, so their parent is that pair rather
            # than the row. Still found from the row, which is what the reach
            # into this row actually needs, and
            # `tests/qt/test_preferences_gear.py` asserts the two share a
            # parent -- which they do.
            assert widget.parent().parent() is row, (
                f"{name} is no longer inside the actions row")
        else:
            assert widget.parent() is row, (
                f"{name} is parented to {type(widget.parent()).__name__} "
                f"rather than the actions row")
        assert widget in row.findChildren(type(widget))


def test_the_activity_spinner_is_still_beside_clear_console(qtbot,
                                                            qt_theme_applied):
    """It is built with the row now, and the lazy path must find that one.

    ``attach_activity_spinner`` installs the spinner by asking
    ``_btn_clear.parentWidget().layout().indexOf(_btn_clear)`` and inserting
    at ``index + 1``. ``QLayout.indexOf`` does not descend into a sub-layout,
    so after the split that lookup answers -1 and the helper returns None --
    silently, and in the application only, because every test of that helper
    builds its own flat row. The row therefore builds its own spinner and
    publishes it under the name the helper checks first.
    """
    from spacr.qt.widgets.activity_spinner import (ActivitySpinner,
                                                   attach_activity_spinner)

    _host, screen = _screen_in_a_window(qtbot, "measure", 1200)

    spinner = getattr(screen, "_activity_spinner", None)
    assert isinstance(spinner, ActivitySpinner)
    assert attach_activity_spinner(screen) is spinner, (
        "the lazy path did not recognise the spinner the row built, so a "
        "second one would be installed")
    assert len(screen.findChildren(ActivitySpinner)) == 1

    strip = screen._actions_row.layout().itemAt(0).layout()
    order = [strip.itemAt(i).widget() for i in range(strip.count())]
    assert order.index(spinner) == order.index(screen._btn_clear) + 1
