"""The AI toggle and the console must ink in the theme that is on screen.

Both widgets built their own ``setStyleSheet`` strings out of
``spacr.qt.theme.PALETTE``. That name is the **dark** palette and nothing
updates it, so the colours they inlined never changed when the theme did.
Two measured symptoms, both on the light theme:

* :class:`~spacr.qt.widgets.ai_toggle_label.AiToggleLabel` inked its OFF
  state ``#ffffff`` onto the light page's ``#fafafa`` — **1.04:1**. The
  "AI" and "Live" switches in the bottom-right of every AppScreen were
  invisible until you clicked one.
* :class:`~spacr.qt.widgets.console_panel._Bubble` inked its text
  ``#ffffff`` on a bubble the app stylesheet fills with the *light*
  theme's colours — **1.00:1** on ``#ffffff``. Not low contrast: none.

:mod:`tests.qt.test_theme_blind_widgets` covers the same failure for the
Home page and keeps the shrinking list of modules that still read the
frozen palette. This file is the console half, and it asserts against
what the widgets actually put in their stylesheets rather than against
their structure — the structure was never wrong.
"""
from __future__ import annotations

import ast
import pathlib
import re

import pytest

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QWidget

from spacr.qt import preferences, theme
from spacr.qt.widgets import console_panel as cp
from spacr.qt.widgets.ai_toggle_label import AiToggleLabel

#: AA for body text (WCAG 1.4.3).
AA_BODY = 4.5

QT_ROOT = pathlib.Path(theme.__file__).resolve().parent

_HEX = re.compile(r"#[0-9a-fA-F]{6}\b")
_COLOR_PROP = re.compile(r"(?<!-)\bcolor\s*:\s*(#[0-9a-fA-F]{6})")


@pytest.fixture
def as_theme(monkeypatch):
    """Put a named theme in force for the duration of one test."""
    def _use(theme_name: str) -> dict:
        monkeypatch.setattr(preferences, "get_theme", lambda: theme_name)
        return theme.palette_for(theme_name)
    return _use


def _ink(widget) -> str:
    """The foreground colour the widget wrote into its own stylesheet."""
    found = _COLOR_PROP.search(widget.styleSheet() or "")
    assert found, f"{type(widget).__name__} inlines no colour: " \
                  f"{widget.styleSheet()!r}"
    return found.group(1).lower()


def _rendered_fill(widget: QWidget) -> str:
    """The colour actually painted behind ``widget``, off a real render."""
    image = widget.grab().toImage()
    assert image.width() > 4 and image.height() > 4, "widget never laid out"
    return QColor(image.pixel(image.width() // 2, 2)).name().lower()


# ===========================================================================
# AiToggleLabel
# ===========================================================================

@pytest.mark.parametrize("theme_name", theme.THEMES)
def test_ai_toggle_off_is_legible_on_its_own_page(qtbot, as_theme,
                                                  theme_name):
    """OFF inks the theme's own ``fg``, not the dark palette's white.

    Measured on the version this replaces: 1.04:1 on light
    (``#ffffff`` on ``#fafafa``). Dark/space/cell were fine, which is
    exactly why it survived — three of four themes agreed with the
    frozen palette.
    """
    palette = as_theme(theme_name)
    label = AiToggleLabel()
    qtbot.addWidget(label)

    ratio = theme.contrast_ratio(_ink(label), palette["bg"])
    assert ratio >= AA_BODY, (
        f"{theme_name}: the OFF 'AI' label is {ratio:.2f}:1 — "
        f"{_ink(label)} on a {palette['bg']} page")


@pytest.mark.parametrize("theme_name", theme.THEMES)
def test_ai_toggle_on_is_the_theme_invariant_accent(qtbot, as_theme,
                                                    theme_name):
    """ON is deliberately the same blue everywhere — that is the signal."""
    as_theme(theme_name)
    label = AiToggleLabel()
    qtbot.addWidget(label)
    label.setChecked(True)
    assert _ink(label) == theme.CONSTANT_ROLES["button_accent"].lower()


def test_ai_toggle_restyles_when_the_theme_changes(qtbot, monkeypatch):
    """The colour is resolved per call, not captured at import.

    This is the difference between the fix and a fix-shaped change: an
    import-time constant would give the same hex for both themes here.
    """
    monkeypatch.setattr(preferences, "get_theme", lambda: "dark")
    label = AiToggleLabel()
    qtbot.addWidget(label)
    on_dark = _ink(label)

    monkeypatch.setattr(preferences, "get_theme", lambda: "light")
    label._refresh_style()
    on_light = _ink(label)

    assert on_dark == theme.DARK_PALETTE["fg"].lower()
    assert on_light == theme.LIGHT_PALETTE["fg"].lower()
    assert on_dark != on_light


def test_ai_toggle_falls_back_to_dark_when_preferences_explode(qtbot,
                                                               monkeypatch):
    """Headless / corrupt settings must still produce a styled label."""
    def boom():
        raise RuntimeError("no settings backend")
    monkeypatch.setattr(preferences, "resolve_effective_theme", boom)
    label = AiToggleLabel()
    qtbot.addWidget(label)
    assert _ink(label) == theme.DARK_PALETTE["fg"].lower()


# ===========================================================================
# ConsolePanel
# ===========================================================================

@pytest.mark.parametrize("theme_name", theme.THEMES)
def test_chat_bubble_text_is_legible_on_the_bubble_it_sits_in(
        qtbot, qapp, as_theme, theme_name):
    """The 1.00:1 case, off a real render.

    The bubble's fill comes from the app stylesheet (theme-correct); its
    text colour came from the frozen dark palette. On light that put
    ``#ffffff`` on ``#ffffff``.
    """
    as_theme(theme_name)
    bubble = cp._Bubble("assistant", "spaCR AI says something useful")
    qtbot.addWidget(bubble)
    bubble.setStyleSheet(theme.stylesheet(theme_name))
    bubble.resize(420, 80)
    bubble.show()
    qtbot.waitExposed(bubble)
    qapp.processEvents()

    ink = _ink(bubble._label)
    fill = _rendered_fill(bubble)
    ratio = theme.contrast_ratio(ink, fill)
    assert ratio >= AA_BODY, (
        f"{theme_name}: bubble text is {ratio:.2f}:1 — {ink} on {fill}")


@pytest.mark.parametrize("theme_name", theme.THEMES)
def test_console_entries_inline_no_colour_from_another_theme(
        qtbot, as_theme, theme_name):
    """The cause, not the symptom.

    Drive the panel through every entry type it can produce, then sweep
    every stylesheet in the tree for a hex that belongs to the dark
    palette and *not* to the theme on screen. Constant roles
    (``button_accent`` and friends) and the provider-branded AI colours
    are outside the palettes, so they are not flagged — this is about
    importing the wrong palette.
    """
    palette = as_theme(theme_name)
    panel = cp.ConsolePanel("mask")
    qtbot.addWidget(panel)
    panel.append_stdout("segmenting field 3\n")
    panel.append_error("Traceback (most recent call last):\n")
    panel._input.setPlainText("why did that fail?")
    panel._on_submit()

    live = {value.lower() for value in palette.values()}
    dark_only = {value.lower() for value in theme.DARK_PALETTE.values()
                 if value.lower() not in live}
    role_of = {value.lower(): role
               for role, value in theme.DARK_PALETTE.items()}

    offenders = []
    for widget in [panel] + panel.findChildren(QWidget):
        sheet = widget.styleSheet()
        if not sheet:
            continue
        for found in {m.group(0).lower() for m in _HEX.finditer(sheet)}:
            if found in dark_only:
                offenders.append(
                    f"{widget.objectName() or type(widget).__name__} "
                    f"inlines {found} (dark {role_of[found]})")
    assert not offenders, f"{theme_name}: {sorted(set(offenders))}"


@pytest.mark.parametrize("theme_name", theme.THEMES)
def test_stdout_block_inherits_the_console_material(qtbot, as_theme,
                                                     theme_name):
    """Text stays transparent so the parent ConsoleBox material is continuous.

    The ConsoleBox already owns the readable surface. Repainting each output
    block made Glass a stack of opaque strips and previously made Light use a
    frozen dark fill.
    """
    as_theme(theme_name)
    block = cp._StdoutBlock("hello")
    qtbot.addWidget(block)
    assert "background-color: transparent" in block.styleSheet()


@pytest.mark.parametrize("theme_name", theme.THEMES)
def test_the_three_text_colours_follow_the_theme(as_theme, theme_name):
    """``color_*()`` and the legacy ``COLOR_*`` names resolve live.

    ``from ...console_panel import COLOR_USER`` used to freeze the dark
    green into the importing module. Module ``__getattr__`` now serves
    the three names against the current theme, so the old spelling is
    correct rather than merely still working.
    """
    palette = as_theme(theme_name)
    assert cp.color_output() == palette["accent"]
    assert cp.color_user() == palette["success"]
    assert cp.color_error() == palette["error"]
    assert cp.COLOR_OUTPUT == palette["accent"]
    assert cp.COLOR_USER == palette["success"]
    assert cp.COLOR_ERROR == palette["error"]


def test_console_panel_module_still_rejects_unknown_attributes():
    """``__getattr__`` serves three names and nothing else."""
    with pytest.raises(AttributeError, match="COLOR_MAUVE"):
        cp.COLOR_MAUVE


@pytest.mark.parametrize("theme_name", theme.THEMES)
def test_a_user_block_is_green_in_every_theme(qtbot, as_theme, theme_name):
    """Green still means "you typed this" — it is just this theme's green."""
    palette = as_theme(theme_name)
    panel = cp.ConsolePanel()
    qtbot.addWidget(panel)
    panel.set_ai_active(False)
    panel._input.setPlainText("a note to self")
    panel._on_submit()

    blocks = [w for w in panel.findChildren(cp._StdoutBlock)]
    assert blocks, "the user note produced no block"
    assert any(palette["success"].lower() in (b.styleSheet() or "").lower()
               for b in blocks)


# ===========================================================================
# The sweep, for the two modules this file owns
# ===========================================================================

def _reads_the_dark_palette(relative: str) -> bool:
    """True when ``relative`` names ``PALETTE`` in code (not in prose)."""
    path = QT_ROOT / relative
    tree = ast.parse(path.read_text(), str(path))
    for node in ast.walk(tree):
        imported = (isinstance(node, ast.ImportFrom)
                    and (node.module or "").endswith("theme")
                    and any(a.name == "PALETTE" for a in node.names))
        attribute = (isinstance(node, ast.Attribute)
                     and node.attr == "PALETTE")
        if imported or attribute:
            return True
    return False


@pytest.mark.parametrize("relative", ["widgets/ai_toggle_label.py",
                                      "widgets/console_panel.py"])
def test_these_modules_no_longer_read_the_frozen_dark_palette(relative):
    """Both were on ``STILL_READS_THE_DARK_PALETTE``. Neither is now.

    That list is asserted as an upper bound, so it cannot catch a
    regression in a module that has already been fixed. This can.
    """
    assert not _reads_the_dark_palette(relative), (
        f"{relative} imports spacr.qt.theme.PALETTE again — that is the "
        "DARK palette and nothing updates it. Use theme.active_palette().")
