"""Figure colours are stored as TOKENS and resolved against the live theme.

Instruction 152 section A. Two reports, one cause:

    "in white mode lots of graphs still have white text and axes color"

``preferences.get_figure_colors()` was always right — it stored "auto" and
resolved it per theme. What was wrong is that the Figure settings dialog
seeded itself from the RESOLVED pair and wrote that pair back on OK, so
opening it once on a dark theme and pressing OK without changing anything
replaced "auto" with a hard "#ffffff". After that no theme switch could move
it: white text on a white page, for good. The maintainer's own store was in
exactly that state (``figure_bg=none``, ``figure_fg=#ffffff``), which is why
they could see the bug while the code that produced it looked correct.

So this file asserts three separate things, and each of them failed before
instruction 152:

1. the store keeps the token — pressing OK with nothing touched is a no-op;
2. a store already frozen is migrated back to "auto" once, loudly;
3. THE PIXELS. A light theme with nothing chosen renders dark ink. Sampled
   off a real PNG rather than read off the Figure object, because "the
   artists have the right colour set" is what was true all along.
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# preference isolation
# ---------------------------------------------------------------------------

@pytest.fixture
def prefs(monkeypatch, tmp_path):
    """The real preference module, writing to a throwaway ini file.

    The getters and setters ARE the item, so they are not stubbed; only the
    single accessor they funnel through is redirected, which is what keeps
    the developer's own store out of it.
    """
    from PySide6.QtCore import QSettings
    from spacr.qt import preferences as preferences_module

    store = tmp_path / "prefs.ini"
    monkeypatch.setattr(
        preferences_module, "_settings",
        lambda: QSettings(str(store), QSettings.Format.IniFormat))
    return preferences_module


def _theme(monkeypatch, prefs, name):
    """Pin the effective theme. Light is the only light one."""
    monkeypatch.setattr(prefs, "resolve_effective_theme", lambda: name)


def _raw(prefs, key):
    """Read a key straight out of the store, with no resolution at all."""
    return prefs._settings().value(key, None)


# ---------------------------------------------------------------------------
# 1. the token survives
# ---------------------------------------------------------------------------

def test_a_fresh_store_holds_the_token_not_a_colour(prefs, monkeypatch):
    _theme(monkeypatch, prefs, "dark")
    assert prefs.get_figure_color_tokens() == ("auto", "auto")
    # …and resolving is a separate call, so nothing can confuse the two.
    assert prefs.get_figure_colors() == ("none", "#ffffff")


def test_auto_follows_the_theme_in_both_directions(prefs, monkeypatch):
    _theme(monkeypatch, prefs, "dark")
    assert prefs.get_figure_colors()[1] == "#ffffff"
    _theme(monkeypatch, prefs, "light")
    assert prefs.get_figure_colors()[1] == "#000000"
    # Space is a DARK theme, and a `== "dark"` test would have handed it
    # white figures. Kept here so the resolver's one subtlety is asserted
    # beside the tokens rather than only in the theme suite.
    _theme(monkeypatch, prefs, "space")
    assert prefs.get_figure_colors()[1] == "#ffffff"


def test_an_explicit_colour_is_honoured_on_every_theme(prefs, monkeypatch):
    prefs.set_figure_colors("auto", "#ff0000")
    for theme in ("dark", "light", "space"):
        _theme(monkeypatch, prefs, theme)
        assert prefs.get_figure_colors()[1] == "#ff0000"


def test_set_figure_colors_auto_writes_the_token(prefs, monkeypatch):
    _theme(monkeypatch, prefs, "dark")
    prefs.set_figure_colors("#123456", "#ff0000")
    prefs.set_figure_colors_auto()
    assert _raw(prefs, prefs._KEY_FIG_FG) == "auto"
    _theme(monkeypatch, prefs, "light")
    assert prefs.get_figure_colors()[1] == "#000000"


# ---------------------------------------------------------------------------
# 2. the migration — the half that actually fixes the report
# ---------------------------------------------------------------------------

def _freeze(prefs, bg, fg):
    """Put the store into the state the old dialog left it in."""
    settings = prefs._settings()
    settings.setValue(prefs._KEY_FIG_BG, bg)
    settings.setValue(prefs._KEY_FIG_FG, fg)
    settings.remove(prefs._KEY_FIG_COLOR_SCALE)
    settings.sync()


def test_the_maintainers_frozen_store_is_migrated_back_to_auto(
        prefs, monkeypatch, caplog):
    """bg 'none' + fg '#ffffff' on a dark theme IS the resolution, verbatim.

    Nothing can tell it from a deliberate choice, so the only repair that
    helps the people who never chose anything is to assume the bug.
    """
    _freeze(prefs, "none", "#ffffff")
    _theme(monkeypatch, prefs, "light")
    with caplog.at_level(logging.INFO, logger="spacr.qt.preferences"):
        assert prefs.get_figure_colors() == ("none", "#000000")
    assert prefs.get_figure_color_tokens() == ("auto", "auto")
    assert _raw(prefs, prefs._KEY_FIG_FG) == "auto"
    said = " ".join(record.getMessage() for record in caplog.records)
    assert "follow the theme" in said, said
    assert "#ffffff" in said, "the migration must name what it changed"


def test_a_colour_auto_never_produces_is_left_alone(prefs, monkeypatch):
    _freeze(prefs, "#ff0000", "#00ff00")
    _theme(monkeypatch, prefs, "light")
    assert prefs.get_figure_colors() == ("#ff0000", "#00ff00")
    assert prefs.get_figure_color_tokens() == ("#ff0000", "#00ff00")


def test_the_migration_runs_once_and_then_gets_out_of_the_way(
        prefs, monkeypatch):
    """A colour chosen AFTER the migration is a decision, not a resolution."""
    _freeze(prefs, "none", "#ffffff")
    _theme(monkeypatch, prefs, "dark")
    assert prefs.get_figure_color_tokens() == ("auto", "auto")
    prefs.set_figure_colors("auto", "#ffffff")
    # Same value, opposite meaning — and the store must keep it now.
    assert prefs.get_figure_color_tokens() == ("auto", "#ffffff")
    _theme(monkeypatch, prefs, "light")
    assert prefs.get_figure_colors()[1] == "#ffffff"


def test_setting_a_colour_before_any_read_is_never_second_guessed(prefs):
    """`set` marks the store migrated, so an explicit write always sticks.

    Without this a caller that writes before it reads — several tests, and
    any first-run wizard — would have its own value taken away on the next
    read.
    """
    prefs.set_figure_colors("#ffffff", "#000000")
    assert prefs.get_figure_colors() == ("#ffffff", "#000000")


def test_the_background_is_unfrozen_with_the_text(prefs, monkeypatch):
    """Both halves, together.

    A store frozen on a LIGHT theme holds white/black. Migrating the text
    and keeping the white background would give white-on-white again the
    moment the theme went dark — the same bug wearing the other colour.
    """
    _freeze(prefs, "#ffffff", "#000000")
    _theme(monkeypatch, prefs, "dark")
    assert prefs.get_figure_color_tokens() == ("auto", "auto")
    assert prefs.get_figure_colors() == ("none", "#ffffff")


# ---------------------------------------------------------------------------
# 3. the dialog — the regression for the bug itself
# ---------------------------------------------------------------------------

def _dialog(qtbot):
    from spacr.qt.widgets.figure_queue import _FigureSettingsDialog
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 1, 2], [0, 1, 0])
    dialog = _FigureSettingsDialog(fig)
    qtbot.addWidget(dialog)
    return dialog


def test_the_dialog_seeds_from_the_token(prefs, monkeypatch, qtbot):
    _theme(monkeypatch, prefs, "dark")
    dialog = _dialog(qtbot)
    assert dialog._bg == "auto" and dialog._fg == "auto", (
        "seeded from the resolved pair again — that is the whole bug")


def test_ok_without_touching_anything_leaves_auto_alone(
        prefs, monkeypatch, qtbot):
    """THE REGRESSION. This failed before instruction 152.

    Open the dialog on a dark theme, press OK, change nothing, switch to the
    light theme — and the figures must still follow it.
    """
    _theme(monkeypatch, prefs, "dark")
    dialog = _dialog(qtbot)
    dialog._apply_and_accept()

    assert prefs.get_figure_color_tokens() == ("auto", "auto"), (
        "pressing OK persisted the RESOLVED default")
    _theme(monkeypatch, prefs, "light")
    assert prefs.get_figure_colors() == ("none", "#000000")


def test_the_button_says_when_a_colour_is_automatic(prefs, monkeypatch,
                                                    qtbot):
    """The preview is labelled, so "white" and "white because dark" differ.

    A dialog that cannot SHOW the difference cannot be trusted to store it.
    """
    _theme(monkeypatch, prefs, "dark")
    dialog = _dialog(qtbot)
    assert dialog._fg_btn.text() == "Automatic (#ffffff)"
    assert dialog._bg_btn.text() == "Automatic (transparent)"
    _theme(monkeypatch, prefs, "light")
    dialog._paint_colour_buttons()
    assert dialog._fg_btn.text() == "Automatic (#000000)"
    # And the way back is greyed when it would do nothing (instruction 106),
    # which doubles as the readout for "am I frozen?".
    assert not dialog._auto_btn.isEnabled()


def test_picking_a_colour_stores_it_explicitly(prefs, monkeypatch, qtbot):
    from PySide6.QtGui import QColor

    _theme(monkeypatch, prefs, "dark")
    dialog = _dialog(qtbot)
    monkeypatch.setattr("spacr.qt.widgets.colour_picker.pick_colour",
                        lambda *a, **k: QColor("#ff0000"))
    dialog._pick("_fg", dialog._fg_btn)
    assert dialog._fg == "#ff0000"
    assert dialog._fg_btn.text() == "#ff0000"
    dialog._apply_and_accept()
    assert prefs.get_figure_color_tokens() == ("auto", "#ff0000")
    _theme(monkeypatch, prefs, "light")
    assert prefs.get_figure_colors()[1] == "#ff0000", (
        "an explicit choice must NOT follow the theme")


def test_follow_the_theme_is_the_way_back(prefs, monkeypatch, qtbot):
    """A user frozen by their own click has a route out."""
    _theme(monkeypatch, prefs, "dark")
    prefs.set_figure_colors("#101010", "#ffffff")
    dialog = _dialog(qtbot)
    assert dialog._fg == "#ffffff"
    assert dialog._auto_btn.isEnabled(), "the way out must be reachable"
    dialog._auto_btn.click()
    assert (dialog._bg, dialog._fg) == ("auto", "auto")
    dialog._apply_and_accept()
    _theme(monkeypatch, prefs, "light")
    assert prefs.get_figure_colors() == ("none", "#000000")


# ---------------------------------------------------------------------------
# 4. the pixels
# ---------------------------------------------------------------------------

def _ink_luminance(png_path):
    """Median luminance of the OPAQUE pixels of a rendered figure.

    "auto" resolves the background to transparent, so everything with alpha
    is ink: spines, ticks, tick labels, the title, the data line. That is
    exactly the set of marks the report is about, and it needs no knowledge
    of where on the page any of them landed.
    """
    image = plt.imread(str(png_path))
    assert image.shape[2] == 4, "expected RGBA — the page should be transparent"
    ink = image[image[:, :, 3] > 0.9]
    assert len(ink) > 200, f"only {len(ink)} opaque pixels; nothing was drawn"
    return float(np.median(0.299 * ink[:, 0]
                           + 0.587 * ink[:, 1]
                           + 0.114 * ink[:, 2]))


def _render(prefs, tmp_path, name):
    from spacr.qt.widgets.figure_queue import render_figure_to_png

    prefs.set_figure_format("png")
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1, 2, 3], [1, 3, 2, 4])
    ax.set_title("title")
    ax.set_xlabel("x")
    out = tmp_path / name
    assert render_figure_to_png(fig, str(out))
    plt.close(fig)
    return _ink_luminance(out)


def test_a_light_theme_renders_dark_ink(prefs, monkeypatch, tmp_path):
    _theme(monkeypatch, prefs, "light")
    assert _render(prefs, tmp_path, "light.png") < 0.25


def test_a_dark_theme_renders_light_ink(prefs, monkeypatch, tmp_path):
    _theme(monkeypatch, prefs, "dark")
    assert _render(prefs, tmp_path, "dark.png") > 0.75


def test_pressing_ok_then_going_light_still_renders_dark_ink(
        prefs, monkeypatch, qtbot, tmp_path):
    """The report, end to end, measured in pixels.

    Dark theme → open Figure settings → OK → switch to light → render. Before
    instruction 152 the ink came out at ~1.0 (white on a white page).
    """
    _theme(monkeypatch, prefs, "dark")
    _dialog(qtbot)._apply_and_accept()
    _theme(monkeypatch, prefs, "light")
    assert _render(prefs, tmp_path, "after_ok.png") < 0.25
