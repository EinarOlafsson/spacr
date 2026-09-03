"""Provide palettes, geometry tokens, and QSS for the spaCR Qt interface.

Use :func:`active_palette` for colors shown by a live widget and
:func:`stylesheet` for the application stylesheet. :data:`THEMES` contains
the selectable palettes ``"dark"``, ``"light"``, ``"cell"``, and ``"glass"``;
the ``"system"`` preference resolves to dark or light before palette lookup.
A legacy ``"space"`` palette can still be read from persisted settings but is
not selectable.

.. warning::

   ``theme.PALETTE`` is a deprecated, read-only alias for the dark palette and
   does not follow runtime theme changes. Use :func:`active_palette`, or
   :data:`DARK_PALETTE` only when dark colors are explicitly required.

Cell and Glass are :data:`IMAGE_THEMES`. Their panels use translucent scrims
so the backdrop remains visible without sacrificing text contrast.
:func:`contrast_failures` validates roles painted on scrims, while
:func:`image_contrast_failures` validates roles painted directly over image
content. :func:`solve_scrim_alpha` balances those contrast constraints against
:data:`MIN_PICTURE_CONTRAST`, and :func:`scrim_report` exposes the result.

:func:`enable_spaceout` dresses the process in the rainbow palette the
``spaceout`` entry point launches into. It re-hues whichever theme is
resolved rather than adding a fifth one, so :data:`THEMES` and the light/dark
handling are untouched; it is process state and is never persisted.
"""
from __future__ import annotations

import logging
import math
import warnings
from contextlib import contextmanager
from functools import lru_cache
from types import MappingProxyType
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dark palette — kept aligned with the Tk gui_elements apply_theme dict so
# switching between the two GUIs feels visually consistent.
# ---------------------------------------------------------------------------
# Named DARK_PALETTE, not PALETTE. The old name read like "the palette"
# and was imported as one by two dozen widgets, which is how the light
# theme ended up drawing dark panels. See the module docstring and
# :func:`active_palette`.
DARK_PALETTE = {
    # Surfaces — pure black bg, subtle depth via layered near-blacks
    "bg":          "#000000",   # main window background
    "page":        "#23252a",   # the page BEHIND the panels — see `page`
    "surface":     "#0d0e10",   # sidebar / panels — barely lifted from bg
    "surface_alt": "#161719",   # cards / grouped sections — one more step
    "surface_hi":  "#1f2124",   # hovered surfaces
    "border":      "#2a2d33",   # visible dividers
    "border_soft": "#1c1e22",   # hairline card borders
    # Text
    "fg":          "#ffffff",
    "fg_muted":    "#a1a6ad",   # secondary text (was #9ba0a6)
    "fg_dim":      "#6b6f76",   # disabled / hints
    # Accent
    "accent":      "#4A9EFF",   # primary interactive
    "accent_hi":   "#66B2FF",   # hover
    "accent_lo":   "#2F80D9",   # pressed
    "accent_soft": "#1e3550",   # accent-tinted surface (chips, highlights)
    # Status
    "success":     "#3fb950",
    # The class/value bubbles of the Classify selector. Semantic roles, not
    # literals at the call site, so they follow the theme -- and named for
    # what they MEAN rather than for the colour, so a retheme can move them.
    "chip_class":  "#1fb6ad",   # teal  -- the class name
    "chip_value":  "#3fb950",   # green -- its value
    "warning":     "#d29922",
    "error":       "#f85149",
    "info":        "#4A9EFF",
}


# ---------------------------------------------------------------------------
# Light palette — mirrors the dark set. Used when the user picks the
# light theme, or when the OS colour scheme is light and theme=system.
# Kept close in structure to DARK_PALETTE so a caller can swap between
# them by key without touching consumers.
# ---------------------------------------------------------------------------
# Every value below was picked (or corrected) against
# :data:`CONTRAST_RULES`. The originals failed AA in several places —
# `accent` at 4.10:1 on `surface_hi`, `accent_hi` at 2.81:1 on
# `accent_soft`, `warning` at 3.82:1, `fg_dim` at 2.54:1 — because in a
# light theme "hover" has to go *darker*, not brighter, and the first
# cut of this palette mirrored the dark one literally.
#
# Every key here also exists in :data:`DARK_PALETTE` — see
# ``test_theme.py`` — so a caller can swap palettes without touching
# consumers. That is what makes :func:`active_palette` a drop-in for the
# old module-level import.
LIGHT_PALETTE = {
    "bg":          "#fafafa",
    "page":        "#e1e4e6",   # see `page` — a light theme goes the other way
    "surface":     "#ffffff",
    "surface_alt": "#f2f4f7",
    "surface_hi":  "#e6e9ee",
    "border":      "#d5d9df",
    "border_soft": "#e5e8ec",
    "fg":          "#0d0e10",
    "fg_muted":    "#4b5460",
    "fg_dim":      "#68707e",
    "accent":      "#0a63c4",
    "accent_hi":   "#0851a3",   # hover = darker in a light theme
    "accent_lo":   "#063d7a",
    "accent_soft": "#dbe8fb",
    "success":     "#0f7030",
    # The class/value bubbles of the Classify selector. Semantic roles, not
    # literals at the call site, so they follow the theme -- and named for
    # what they MEAN rather than for the colour, so a retheme can move them.
    "chip_class":  "#0a6f6a",   # teal  -- the class name
    "chip_value":  "#0f7030",   # green -- its value
    "warning":     "#8f4e00",
    "error":       "#b81d1a",
    "info":        "#0a63c4",
}


# ---------------------------------------------------------------------------
# Space palette — a dark theme layered over a generated deep-space image.
# ---------------------------------------------------------------------------
# The surface colours here are *scrim* colours: they are painted at the
# alpha in :data:`SCRIM_ALPHA` over whatever the background image
# happens to be. They are therefore chosen so that even composited over
# a pure white star they stay dark enough for `fg`, `fg_muted` and
# `accent` to clear AA — see :func:`effective_surface`.
SPACE_PALETTE = {
    "bg":          "#04060d",   # fallback flat sky when no image is cached
    "page":        "#232b3d",   # see `page`
    "surface":     "#080d18",
    "surface_alt": "#0d1524",
    "surface_hi":  "#182338",
    "border":      "#33405c",
    "border_soft": "#1e293d",
    "fg":          "#ffffff",
    "fg_muted":    "#c2ccdd",
    "fg_dim":      "#93a0b6",
    "accent":      "#6cb6ff",   # brighter than the dark theme's accent:
    "accent_hi":   "#9bcdff",   # it has to clear AA on a translucent
    "accent_lo":   "#3d8ddb",   # scrim, not on solid #161719
    "accent_soft": "#16304f",
    "success":     "#5fd97a",
    # The class/value bubbles of the Classify selector. Semantic roles, not
    # literals at the call site, so they follow the theme -- and named for
    # what they MEAN rather than for the colour, so a retheme can move them.
    "chip_class":  "#4fd6ce",   # teal  -- the class name
    "chip_value":  "#5fd97a",   # green -- its value
    "warning":     "#f0c14b",
    "error":       "#ff7b72",
    "info":        "#6cb6ff",
}


# ---------------------------------------------------------------------------
# Cell palette — a dark theme layered over the user's own micrographs.
# ---------------------------------------------------------------------------
# Same construction as Space (scrim colours, judged composited over a
# white worst case), re-hued to the imagery: the microtubule network is
# cyan and the filopodia are cyan-to-green, so a blue accent would fight
# the picture and a warm one would clash with it.
#
# `accent`, `fg_muted` and `error` are *brighter* than their Space
# counterparts on purpose. They are the roles that get painted straight
# onto the wallpaper, so their luminance is what
# :func:`max_background_luma` turns into the exposure the imagery is
# allowed — a dim accent here would mean a needlessly murky background
# everywhere. As set, the limit is 0.109 rather than Space's 0.059,
# which is the difference between the microtubule frame reading as a
# photograph and reading as a stain.
CELL_PALETTE = {
    "bg":          "#02080b",   # fallback flat field when no image is cached
    "page":        "#24363f",   # see `page`
    "surface":     "#061218",
    "surface_alt": "#0b1c23",
    "surface_hi":  "#152d37",
    "border":      "#33596a",
    "border_soft": "#1d3a46",
    "fg":          "#ffffff",
    "fg_muted":    "#cfe0e6",
    "fg_dim":      "#a3bbc4",
    "accent":      "#8fe3f7",
    "accent_hi":   "#b9f0ff",
    "accent_lo":   "#4fb3cf",
    "accent_soft": "#123742",
    "success":     "#6fe39a",
    # The class/value bubbles of the Classify selector. Semantic roles, not
    # literals at the call site, so they follow the theme -- and named for
    # what they MEAN rather than for the colour, so a retheme can move them.
    "chip_class":  "#5fe0d8",   # teal  -- the class name
    "chip_value":  "#6fe39a",   # green -- its value
    "warning":     "#f2ca5c",
    "error":       "#ff8f86",
    "info":        "#8fe3f7",
}


# ---------------------------------------------------------------------------
# Glass palette — neutral translucent material over a built-in light field.
# ---------------------------------------------------------------------------
# Qt stylesheets do not expose the compositor's native macOS/iOS backdrop
# blur or Liquid Glass lensing. The cross-platform approximation is a layered
# material: a bounded neutral light field behind low-alpha charcoal surfaces,
# brighter top-edge highlights, soft lower shading, generous concentric
# corners, and selective tint only for actions. This avoids the old result,
# which was simply opaque navy when the global Page opacity was 100%.
GLASS_PALETTE = {
    "bg":          "#0b0d11",
    # The one palette whose `bg` was already a page: its surfaces are
    # charcoal, not near-black, so `bg` clears them by 12.0 L* / 1.301:1
    # without moving. Stated rather than defaulted so `page` is never
    # "the key dark forgot" — see `page`.
    "page":        "#0b0d11",
    "surface":     "#25272c",
    "surface_alt": "#2d3036",
    "surface_hi":  "#3a3e45",
    "border":      "#aeb2ba",
    "border_soft": "#747982",
    "fg":          "#fafafa",
    "fg_muted":    "#d5d6d9",
    "fg_dim":      "#a8abb1",
    # Tint is deliberately reserved for actions and state. The material
    # itself stays neutral, matching Apple's guidance not to tint everything.
    "accent":      "#8cc8ff",
    "accent_hi":   "#b8dcff",
    "accent_lo":   "#579fe0",
    "accent_soft": "#263746",
    "success":     "#78dfa3",
    # The class/value bubbles of the Classify selector. Semantic roles, not
    # literals at the call site, so they follow the theme -- and named for
    # what they MEAN rather than for the colour, so a retheme can move them.
    "chip_class":  "#6fe0d8",   # teal  -- the class name
    "chip_value":  "#78dfa3",   # green -- its value
    "warning":     "#f5cf72",
    "error":       "#ff9a95",
    "info":        "#8cc8ff",
}


#: The themes with a palette of their own. ``"system"`` is a
#: *preference* value that resolves to one of these, not an entry here.
#: "space" was retired: the generated skies were a lot of machinery for a
#: backdrop nobody chose, and the Cell wallpapers do the same job with the
#: lab's own images. A persisted "space" falls back to dark — see
#: `preferences.get_theme`.
THEMES = ("dark", "light", "cell", "glass")

_PALETTES = {
    "dark": DARK_PALETTE,
    "light": LIGHT_PALETTE,
    # Kept, though "space" is no longer offered: `palette_for` is called with
    # whatever is persisted, and a settings file written by an older spaCR can
    # still say "space". Resolving it beats raising; it simply cannot be
    # chosen any more, and `THEMES` no longer lists it so nothing iterates it.
    "space": SPACE_PALETTE,
    "cell": CELL_PALETTE,
    "glass": GLASS_PALETTE,
}

#: Themes whose window background is an image or depth gradient rather than
#: a flat colour. They share one treatment — a transparent ``QWidget``
#: default, translucent scrims on panels, and opaque popups — so the QSS
#: branches on membership here rather than on a theme name. The public name
#: is retained because Space and Cell predate the generated Glass backdrop.
IMAGE_THEMES = ("cell", "glass")


# ---------------------------------------------------------------------------
# Two colours that are a *function of the theme*, not entries in it
# ---------------------------------------------------------------------------
# Both used to be inlined at the call site, which is the exact mistake
# `PALETTE` exists as a warning about: a hex typed into a widget is a
# hex that stays dark on the light theme. Neither is a palette role
# because neither is a colour you would ever choose independently —
# each is derived from one that is already there.

def rim_colour(theme: str = "dark") -> str:
    """The meaningful hairline on outlined tiles — the theme's own ink.

    White in the dark themes, near-black in the light one, because that
    is what ``fg`` already is. Horizontal cards and interactive hover states
    still use it; resting Home module tiles deliberately do not. Deriving it
    from ``fg`` rather than writing ``#ffffff`` means every palette gets the
    right ink without a raw colour drifting out of sync.
    """
    return palette_for(theme)["fg"]


def selection_ink(theme: str = "dark") -> str:
    """Text colour for a row selected with the accent behind it.

    Derived, not chosen. `accent` is a mid blue on both themes, and the ink
    that reads on it flips: measured, black is 7.63:1 on the dark accent and
    white is 5.84:1 on the light one, while `button_accent_ink` -- the
    obvious-looking role -- is 6.96:1 on dark and **3.28:1** on light, which
    is below the 4.5 minimum and was shipped by picking a role instead of
    measuring.

    So it picks whichever of the theme's own two extremes contrasts better,
    which keeps the colour inside the palette rather than reaching for a raw
    #ffffff that belongs to no theme.
    """
    palette = palette_for(theme)
    accent = palette["accent"]
    return max((palette["fg"], palette["bg"]),
               key=lambda ink: contrast_ratio(ink, accent))


def dock_colour(theme: str = "dark") -> str:
    """The left dock's background. **Never translucent, in any theme.**

    The dock used to paint ``surface``, which the image themes re-render
    through :func:`scrim_alpha` — so on Space and Cell the app list was a
    ghost with a galaxy behind every row. A navigation column is chrome:
    it is the thing you look at when you have lost your place, and it has
    to be a solid edge for the page to end at.

    White under the light theme (its ``surface``), a dark grey everywhere
    else (``surface_alt``, one step up from the near-black window so the
    dock reads as a separate plane rather than as more page).
    """
    base = palette_for(theme)
    return base["surface"] if theme == "light" else base["surface_alt"]


# ---------------------------------------------------------------------------
# Maturity — how finished a module is, drawn as a colour
# ---------------------------------------------------------------------------
# The app→stage table lives in :data:`spacr.qt.app.APP_STAGE`; this is
# only what each stage LOOKS like. Both the tile hover and the legend
# under the Home aside read these, so a stage cannot be one colour in
# the legend and another on the tile.
#
# The three hues were chosen by the user. They are deliberately not
# palette accents: the point is that "this module is alpha" is not a
# theme decision, and the same green means the same thing on Space as on
# Light.

STAGE_HOVER = {
    "stable": "#3B82F6",   # blue
    "beta":   "#FF00FF",   # magenta
    "alpha":  "#00CEC8",   # green-cyan
}

#: What the legend writes next to each swatch.
STAGE_LABEL = {
    "stable": "Stable",
    "beta":   "Beta",
    "alpha":  "Alpha",
}

#: The legend's tooltip per stage — the sentence the colour stands in for.
STAGE_NOTE = {
    "stable": "Signed off and in normal use.",
    "beta":   ("Further along than alpha and in regular use, but not "
               "signed off yet."),
    "alpha":  ("Built and reachable, not yet trusted end to end. Expect "
               "rough edges, and check the numbers before you rely on "
               "them."),
}


def stage_hover(stage: str) -> str:
    """Hover colour for ``stage``; unknown stages read as stable."""
    return STAGE_HOVER.get(stage, STAGE_HOVER["stable"])


# ---------------------------------------------------------------------------
# The module tile's geometry — here, because the QSS needs it too
# ---------------------------------------------------------------------------
# These would live happily in :class:`spacr.qt.widgets.home.HomePage`
# were it not for `min-height`, and `min-height` is not optional.
#
# The app stylesheet carries `QPushButton { min-height: 22px }`, and
# QStyleSheetStyle turns that into a real `setMinimumHeight(22)` on every
# button when it polishes it. `qSmartMinSize` lets an EXPLICIT minimum
# override the widget's own `minimumSizeHint()`, so a tile that answers
# "124 px" through its hints is still, as far as the layout is concerned,
# collapsible to 22 — and on a page that does not fit, it collapses. The
# symptom is not a clipped tile: it is the name label drawn on top of the
# icon, with no warning and no scrollbar.
#
# The fix is for the QSS to state the floor itself, which means the QSS
# has to know the number. So the number lives here, the widget reads it
# from here, and there is one of it.
TILE_W = 172
TILE_MAX_W = 260
TILE_H = 124
TILE_ICON_PX = 48


# ---------------------------------------------------------------------------
# Scrims — per-theme opacity of each surface role
# ---------------------------------------------------------------------------
# Only the image themes are translucent. Everything else resolves to 1.0
# and the QSS emits plain hex, byte-identical to what it emitted before
# scrims existed.
#
# The alphas are **solved, not chosen** — see :func:`solve_scrim_alpha`.
# The first cut of them was a hand-picked 0.86/0.88/0.90/0.93, which
# passes every contrast rule and hides the wallpaper: a 0.90 scrim
# transmits a 1.10:1 range of the picture, i.e. a ghost. Users reported
# the image themes as "not implemented — I can't see the cells", which
# was a fair reading of a 10 % image.
#
# `elevated` (menus, tooltips, combo popups) is deliberately opaque even
# there: those are separate top-level windows, and a translucent popup
# without a compositor shows the desktop, not the app.

#: Worst-case pixel that a scrim can be composited over when nothing
#: bounds the wallpaper's brightness. A star core is pure white, so that
#: is what the contrast check assumes sits behind every panel —
#: anything dimmer only helps.
WORST_CASE_UNDER = "#ffffff"

#: How much of the wallpaper a scrim must still let through, as the WCAG
#: contrast ratio between a panel sitting over the brightest background
#: its theme can present and the same panel over black. It is the
#: dynamic range of the picture as seen *through* the panel.
#:
#: 1.5:1 is well above the ~1.1:1 at which a large-area luminance step
#: becomes visible at all — so the image is unambiguously present rather
#: than a ghost — and well below the 3:1 that WCAG 1.4.11 asks of a
#: meaningful UI boundary, so a card still reads as a card and not as a
#: hole. At the old 0.90 the same number was 1.10:1: right at the
#: threshold of visible, which is what the bug reports were about.
MIN_PICTURE_CONTRAST = 1.5

#: Multiplier applied to every WCAG minimum when solving an alpha, so a
#: solved scrim is not sitting exactly on the line. Qt composites
#: ``rgba()`` in 8-bit, and the solver's arithmetic is exact, so the
#: drift is a fraction of a level — but a rule that passes at 4.50:1 and
#: fails at 4.49:1 should not be the thing standing between the user and
#: a readable panel.
SCRIM_HEADROOM = 1.05

#: Themes whose wallpaper is *guaranteed* to stay under
#: :func:`max_background_luma` — every frame they can show has been
#: through :func:`spacr.qt.imagery.render`, which exposure-solves the
#: shipped masters and a user's own drop-in alike. Only these may have
#: their scrims judged against that ceiling instead of against white;
#: see :func:`scrim_under` for why Space is not one of them.
#:
#: A theme joins this set by having its wallpaper solved, not by being
#: added here. Adding one whose picture is not bounded would silently
#: thin its panels past what its own background can survive.
EXPOSURE_BOUNDED_THEMES = ("cell", "glass")

# Brightest stop in Glass's built-in `_window_block` light field. Unlike Space,
# Glass cannot accept an arbitrary photograph, so this is a hard rendering
# contract rather than a hopeful estimate.
GLASS_BACKDROP_UNDER = "#454950"


def _grey_for_luminance(luminance: float) -> str:
    """The neutral grey whose WCAG relative luminance is ``luminance``.

    A grey has equal linear channels, and the luminance weights sum to
    1, so its relative luminance *is* its linear channel value — the
    inverse is just the sRGB transfer function.
    """
    value = max(0.0, min(1.0, float(luminance)))
    srgb = (value * 12.92 if value <= 0.0031308
            else 1.055 * value ** (1 / 2.4) - 0.055)
    level = max(0, min(255, int(round(srgb * 255.0))))
    return "#%02x%02x%02x" % (level, level, level)


def scrim_under(theme: str) -> str:
    """Brightest colour ``theme``'s wallpaper can put behind a panel.

    This is the whole reason the two image themes do not end up with the
    same alphas, and it is a property of the *pipeline that produces the
    wallpaper*, not of the palette:

    * **Cell** wallpapers always come out of :func:`spacr.qt.imagery.render`,
      which exposure-solves every frame it returns — the shipped masters
      and the user's own drop-in alike. No text-line-sized region of a
      Cell background can therefore exceed :func:`max_background_luma`,
      and that ceiling, not white, is the worst case a Cell panel has to
      survive. Measured: the shipped ``microtubules`` master peaks at
      0.098 against a 0.109 limit, ``filopodia`` at 0.073.

    * **Space** can be the procedurally generated sky, whose exposure is
      anchored on the 40th percentile *precisely so a sun stays
      white-hot* (:data:`spacr.qt.space.TARGET_SKY_PERCENTILE`). That is
      a deliberate look, and it means the sky really does present a
      near-white region the size of a line of text: the 1440x900 galaxy
      sky measures 0.49 over a text window, colour ``#bab9b9``. Only a
      strong scrim saves text over that, so Space is judged against
      white and its panels stay much more opaque than Cell's.

    Anything that is not an image theme gets white; its alphas are 1.0
    and the answer is never used.
    """
    if theme == "glass":
        return GLASS_BACKDROP_UNDER
    if theme not in EXPOSURE_BOUNDED_THEMES:
        return WORST_CASE_UNDER
    return _grey_for_luminance(max_background_luma(theme))


def _scrim_rules(role: str) -> Tuple[Tuple[str, float], ...]:
    """``(foreground role, minimum ratio)`` for text on surface ``role``."""
    return tuple((fg, required)
                 for fg, surface, required in CONTRAST_RULES
                 if surface == role)


def legible_scrim_floor(theme: str, role: str,
                        colour_role: Optional[str] = None,
                        under: Optional[str] = None) -> float:
    """Thinnest scrim for ``role`` that text is still readable over.

    Every rule in :data:`CONTRAST_RULES` that paints text on this
    surface must clear its WCAG minimum (times :data:`SCRIM_HEADROOM`)
    with the surface composited over :func:`scrim_under` — the brightest
    thing the theme's wallpaper pipeline can put behind it. Below this
    number the panel stops being readable; it is a hard lower bound.

    :param theme: name of the palette whose surface is being evaluated.
    :param role: palette surface role on which the text is painted.
    :param colour_role: palette entry the surface is painted with, when
        it differs from ``role`` — ``tile`` is painted with ``surface``.
    :param under: what is actually behind the surface, when it is not
        the wallpaper. :func:`pane_alpha_floor` passes the flat window
        colour for the opaque themes; judging a dark panel over a *dark*
        window against a white worst case would report a 0.92 floor for
        a surface that is legible at any alpha at all.
    """
    palette = palette_for(theme)
    base = palette[colour_role or role]
    if under is None:
        under = scrim_under(theme)
    rules = _scrim_rules(colour_role or role)
    for step in range(0, 1001):
        alpha = step / 1000.0
        over_worst = composite(base, alpha, under)
        if all(contrast_ratio(palette[fg], over_worst)
               >= required * SCRIM_HEADROOM for fg, required in rules):
            return alpha
    return 1.0


def picture_contrast(theme: str, role: str, alpha: float,
                     colour_role: Optional[str] = None) -> float:
    """How much of the wallpaper survives ``role`` at ``alpha``.

    The WCAG ratio between the panel sitting over the brightest thing
    the theme can put behind it and the same panel over black: the
    dynamic range of the picture as seen *through* the panel. 1.0 is an
    opaque panel — no picture at all.
    """
    base = palette_for(theme)[colour_role or role]
    return contrast_ratio(composite(base, alpha, scrim_under(theme)),
                          composite(base, alpha, "#000000"))


def present_scrim_ceiling(theme: str, role: str,
                          colour_role: Optional[str] = None) -> float:
    """Thickest scrim for ``role`` that the picture still reads through.

    The largest alpha whose :func:`picture_contrast` is still at least
    :data:`MIN_PICTURE_CONTRAST`. Above this number the wallpaper is a
    ghost — which is the bug this whole solver exists to close.
    """
    for step in range(1000, -1, -1):
        alpha = step / 1000.0
        if picture_contrast(theme, role, alpha, colour_role) \
                >= MIN_PICTURE_CONTRAST:
            return alpha
    return 0.0


def solve_scrim_alpha(theme: str, role: str,
                      colour_role: Optional[str] = None) -> float:
    """The opacity ``role`` should be painted at in ``theme``.

    Two bounds, pulling opposite ways:

    * :func:`legible_scrim_floor` is a **lower** bound — thinner than
      that and text on the panel stops clearing AA over the worst thing
      the wallpaper can present.
    * :func:`present_scrim_ceiling` is an **upper** bound — thicker than
      that and the picture stops reading through the panel.

    The answer is the ceiling, clamped up to the floor: as solid a panel
    as the picture can afford, and never thinner than legibility allows.
    Every alpha in that window satisfies both constraints, so the choice
    within it is which one to spend the slack on, and it goes to the
    panel: the settings form sits on this surface, and preserving its grey
    category structure is more important than exposing additional wallpaper.
    Taking the floor instead would show *more* picture — Cell's floor is 0.05,
    a nearly transparent panel — at the cost of the form dissolving into the
    wallpaper.

    When the floor lands *above* the ceiling the theme cannot do both,
    legibility wins, and the shortfall is visible in
    :func:`scrim_report`.

    :param theme: name of the palette whose surface is being solved.
    :param role: palette surface role whose opacity is being chosen.
    :param colour_role: palette entry the surface is painted with, when
        it differs from ``role`` — ``tile`` is painted with ``surface``.
    """
    return max(legible_scrim_floor(theme, role, colour_role),
               present_scrim_ceiling(theme, role, colour_role))


#: The roles :func:`_solve_scrims` solves, and the palette entry each
#: one is painted with. ``tile`` is the odd one out: the home-screen
#: tiles are painted with the ``surface`` colour, so they are judged
#: against the ``surface`` rules.
SCRIM_ROLES: Dict[str, str] = {
    "surface":     "surface",
    "surface_alt": "surface_alt",
    "surface_hi":  "surface_hi",
    "tile":        "surface",
}


def scrim_report(theme: str) -> List[dict]:
    """Both bounds, the solved alpha and what it buys, for every role.

    Each entry is ``{"role", "colour_role", "alpha", "floor", "ceiling",
    "picture", "worst_fg", "worst_ratio", "required", "legible",
    "shows_picture"}``. This is the audit trail for
    :data:`SCRIM_ALPHA` — the numbers a reviewer would otherwise have to
    re-derive to check that a solved alpha is the right one.
    """
    palette = palette_for(theme)
    out: List[dict] = []
    for role, colour_role in SCRIM_ROLES.items():
        alpha = scrim_alpha(theme, role)
        over_worst = composite(palette[colour_role], alpha,
                               scrim_under(theme))
        worst = min(((contrast_ratio(palette[fg], over_worst) / required,
                      fg, required)
                     for fg, required in _scrim_rules(colour_role)),
                    default=(float("inf"), "", 0.0))
        picture = picture_contrast(theme, role, alpha, colour_role)
        out.append({
            "role": role, "colour_role": colour_role, "alpha": alpha,
            "floor": legible_scrim_floor(theme, role, colour_role),
            "ceiling": present_scrim_ceiling(theme, role, colour_role),
            "surface_color": over_worst,
            "picture": picture,
            "worst_fg": worst[1],
            "worst_ratio": worst[0] * worst[2],
            "required": worst[2],
            "legible": worst[0] >= 1.0,
            "shows_picture": picture >= MIN_PICTURE_CONTRAST,
        })
    return out


def scrim_failures(theme: str) -> List[str]:
    """Every role of ``theme`` that cannot be both legible and see-through.

    Empty when the theme manages both. A non-empty result is not a
    crash — legibility wins and the entry says by how much the picture
    misses — but it means the wallpaper is a ghost under that role and
    something upstream (the palette, or the exposure the imagery is
    solved to) has to give.
    """
    return [
        f"{theme}.{row['role']}: alpha {row['alpha']:.3f} shows the picture "
        f"at {row['picture']:.2f}:1 < {MIN_PICTURE_CONTRAST:.2f}:1 "
        f"(legibility floor {row['floor']:.3f} is above the "
        f"see-through ceiling {row['ceiling']:.3f})"
        for row in scrim_report(theme)
        if not row["shows_picture"]
    ]


def _solve_scrims() -> Dict[str, Dict[str, float]]:
    """Solve every translucent role of every image theme, once, at import.

    Pure colour arithmetic over a thousand-step sweep of four roles and
    three themes: a few milliseconds, no Qt, no I/O. Solved rather than
    tabulated so that re-hueing a palette moves its scrims with it
    instead of silently invalidating a comment.
    """
    out: Dict[str, Dict[str, float]] = {}
    for name in IMAGE_THEMES:
        solved = {role: solve_scrim_alpha(name, role, colour_role)
                  for role, colour_role in SCRIM_ROLES.items()}
        # Popups are separate top-level windows. Translucency there
        # shows the desktop, not the wallpaper.
        solved["elevated"] = 1.00
        out[name] = solved
    return out


SCRIM_ALPHA: Dict[str, Dict[str, float]] = {}


def scrim_alpha(theme: str, role: str) -> float:
    """Opacity of surface ``role`` in ``theme``. 1.0 unless translucent."""
    return SCRIM_ALPHA.get(theme, {}).get(role, 1.0)


# ---------------------------------------------------------------------------
# The page pane — the one surface whose opacity the USER sets
# ---------------------------------------------------------------------------
# Everything above solves an alpha. This does not: the rounded box
# behind Home's tiles is the one surface the user asked to be able to
# see through, and the whole point of a preference is that the answer is
# theirs.
#
# What the solver still owns is the FLOOR. `legible_scrim_floor` is the
# thinnest this pane can be painted and still have the tile names on it
# clear WCAG AA over the worst pixel that theme's wallpaper can present,
# so the preference is clamped UP to it and text can never be dragged
# into illegibility.
#
# The solver's other bound, `present_scrim_ceiling`, is deliberately NOT
# applied. That one exists to guarantee the wallpaper is still visible
# through a panel — and a user who drags this slider to 100 % is
# explicitly asking for the wallpaper to be hidden behind the app they
# are trying to use. A floor protects them from a mistake; a ceiling
# would only overrule a choice.

#: What the preference means at each end. 100 % is a solid panel in the
#: conventional themes. In Glass it is full *material strength*, whose own
#: designed alpha remains translucent.
PANE_OPACITY_MIN = 0.0
PANE_OPACITY_MAX = 1.0
DEFAULT_PANE_OPACITY = 1.0


def pane_alpha_floor(theme: str) -> float:
    """Thinnest the Home pane may be painted and stay readable.

    An image theme is judged against :func:`scrim_under`, the brightest
    thing its wallpaper can put behind the panel. Everything else is
    judged against its own flat window colour, which is where the answer
    goes to (near) zero: a dark panel fading into a dark window cannot
    make white text harder to read, so those themes let the user take
    the box away entirely.
    """
    under = (scrim_under(theme) if theme in IMAGE_THEMES
             else palette_for(theme)["bg"])
    return legible_scrim_floor(theme, "surface", under=under)


def pane_alpha(theme: str, opacity: Optional[float] = None) -> float:
    """The alpha a user-controlled page surface is actually painted at.

    The user's ``opacity`` (0..1), clamped up to :func:`pane_alpha_floor`.
    ``None`` means :data:`DEFAULT_PANE_OPACITY`.
    """
    if opacity is None:
        opacity = DEFAULT_PANE_OPACITY
    wanted = max(PANE_OPACITY_MIN, min(PANE_OPACITY_MAX, float(opacity)))
    if theme == "glass":
        # Glass has material translucency of its own. Page opacity controls
        # how strongly that material is present; 100% means the designed
        # glass, not an opaque navy panel. This keeps the preference and the
        # theme as two genuinely separate features.
        wanted *= scrim_alpha("glass", "surface")
    return max(wanted, pane_alpha_floor(theme))


def pane_surface(role: str = "surface_alt",
                 theme: Optional[str] = None,
                 opacity: Optional[float] = None) -> str:
    """A page-surface colour, already carrying the user's page opacity.

    The single accessor every container should use, including the ones styled
    inline rather than through :func:`stylesheet`. Those were the gap: Home's
    aside panels, the dock and the tile boxes all read
    :func:`active_palette` directly, which returns **raw hex**, so they stayed
    fully opaque no matter what the preference said and the setting looked
    broken from the page the user lands on.

    Reads the live preference when ``opacity`` is not given, so a caller does
    not have to plumb it through — and falls back to the theme's designed
    scrim if preferences cannot be read at all, which is what a first run
    mid-generation gets.

    :param role: a palette key, normally ``surface``/``surface_alt``/``tile``.
    :param theme: theme name; ``None`` resolves the effective one.
    :param opacity: 0..1 override; ``None`` reads the preference.
    :returns: a QSS colour — plain hex when opaque, ``rgba()`` when not.
    """
    if theme is None:
        try:
            from .preferences import resolve_effective_theme
            theme = resolve_effective_theme()
        except Exception:
            theme = "dark"
    if opacity is None:
        try:
            from .preferences import get_pane_opacity
            opacity = get_pane_opacity()
        except Exception:
            opacity = None
    base = palette_for(theme)
    colour_role = SCRIM_ROLES.get(role, role)
    return css_color(base.get(colour_role, base["surface_alt"]),
                     panel_alpha(theme, role, opacity))


def block_surface(role: str = "surface_alt",
                  theme: Optional[str] = None,
                  opacity: Optional[float] = None) -> str:
    """:func:`pane_surface` for a registered QSS block: ``None`` IS the scrim.

    The two differ in exactly one place, and it matters only there. A block
    registered with :func:`register_widget_qss` is handed the ``opacity``
    :func:`stylesheet` was called with, and that is ``None`` when the caller
    asked for the theme's *designed* scrim rather than any user's setting —
    the documented meaning of ``surface_opacity=None``, and what every
    built-in rule in this file honours through :func:`panel_alpha`.

    :func:`pane_surface` cannot honour it. Its ``None`` means "nobody told me,
    go and look", so it reads the live page-opacity preference. That is right
    for the inline and paint-time callers it was written for — Home's aside
    panels have no ``stylesheet`` argument to plumb through — and wrong inside
    a block, where ``None`` was already an answer. The consequence was that
    ``stylesheet("dark")`` emitted ``rgba(22, 23, 25, 0.600)`` for every block
    that passed it through: a function of its arguments quietly depending on a
    QSettings value, which made the assertion that the opaque themes emit
    plain hex pass or fail on module import order.

    No user-visible change. The live path calls :func:`stylesheet` with the
    preference already in hand, so ``opacity`` is a number there and the two
    functions return the same string; the emitted sheet was compared byte for
    byte across every theme at 30 %, 60 % and 100 %.

    :param role: a palette key, normally ``surface``/``surface_alt``/``tile``.
    :param theme: theme name; ``None`` resolves the effective one.
    :param opacity: 0..1, straight from the block's own argument. ``None``
        means the theme's designed scrim and is NOT looked up anywhere.
    :returns: a QSS colour — plain hex when opaque, ``rgba()`` when not.
    """
    if theme is None:
        try:
            from .preferences import resolve_effective_theme
            theme = resolve_effective_theme()
        except Exception:
            theme = "dark"
    base = palette_for(theme)
    colour_role = SCRIM_ROLES.get(role, role)
    return css_color(base.get(colour_role, base["surface_alt"]),
                     panel_alpha(theme, role, opacity))


def panel_alpha(theme: str, role: str,
                opacity: Optional[float] = None) -> float:
    """Apply the page-opacity preference to a shared UI surface role.

    ``None`` preserves the theme's designed scrim, which keeps
    :func:`stylesheet` useful to callers that have no preferences store.
    A numeric value is the user's requested alpha and is honoured for every
    card, settings section, console and preview surface, clamped only where
    going thinner would make that role's text illegible. Glass treats it as
    relative material strength, because making 100% an opaque fill would
    remove the defining property of the theme.

    Popups stay opaque because they are separate native windows; making those
    translucent reveals the desktop rather than the spaCR backdrop.
    """
    if role == "elevated":
        return 1.0
    if opacity is None:
        return scrim_alpha(theme, role)
    wanted = max(PANE_OPACITY_MIN,
                 min(PANE_OPACITY_MAX, float(opacity)))
    colour_role = SCRIM_ROLES.get(role, role)
    under = (scrim_under(theme) if theme in IMAGE_THEMES
             else palette_for(theme)["bg"])
    floor = legible_scrim_floor(
        theme, role, colour_role=colour_role, under=under)
    if theme == "glass":
        # Relative material strength: even the 100% setting retains the
        # role's designed translucency. Other themes keep literal 0..100%
        # surface opacity, so this does not change their established control.
        wanted *= scrim_alpha("glass", role)
    return max(wanted, floor)


# ---------------------------------------------------------------------------
# The field fade — the one surface the page-opacity preference does NOT own
# ---------------------------------------------------------------------------
# Everything above answers "how solid is this panel?" with a single number.
# An input field answers it with a ramp instead, and it is deliberately
# *exempt* from :func:`panel_alpha`:
#
#   "the fields should not be subject to the occupacy setting. the fields
#    could gradually become fully transparent (not the text in the field
#    but the container) with the transparency growing faster towards the
#    right. outlines should also be subject to the same effect."
#
# So a field is painted fully opaque at its left edge — whatever the page
# slider says — and dissolves to nothing at its right edge. Two properties
# fall out of that and both matter:
#
# * The page-opacity slider cannot make a field harder to read. Its left
#   edge, where the value starts, is always a solid surface.
# * The ramp has to be *convex* in transparency, not linear. A linear ramp
#   is already 50 % gone at the midpoint, which is where the text still is.
#
# The curve is therefore a cubic ease-in on TRANSPARENCY:
#
#     transparency(t) = t ** FIELD_FADE_EXPONENT
#     alpha(t)        = 1 - t ** FIELD_FADE_EXPONENT
#
# with ``t`` the fraction of the way across the field, 0 at the left edge
# and 1 at the right. At the midpoint the container is still 87.5 % opaque;
# it is 58 % at three-quarters, 27 % at nine-tenths, and gone at the edge.
# That is "faster towards the right" spelled as a number: d(transparency)/dt
# is 0 at the left edge and 3 at the right.

#: Exponent of the ease-in applied to TRANSPARENCY across a field's width.
#: 1.0 would be a linear fade, which loses the middle of the field where
#: the value is. 3.0 (cubic) keeps the left half essentially solid and
#: spends the whole fade on the trailing third.
FIELD_FADE_EXPONENT = 3.0

#: How many colour stops the cubic is sampled at when it is handed to a
#: ``QLinearGradient``, which only interpolates linearly between stops.
#: 17 evenly-spaced stops hold the piecewise-linear error under
#: 6*(1/16)**2/8 = 0.003 alpha — below one 8-bit level, so the rendered
#: ramp is the cubic to the last representable bit.
FIELD_FADE_STOPS = 17


def field_fade_alpha(t: float) -> float:
    """Alpha of a field's *container* at fraction ``t`` across its width.

    ``t`` is clamped to [0, 1]. ``field_fade_alpha(0.0)`` is 1.0 — a field
    is fully opaque where its value begins, no matter what the page-opacity
    preference is set to — and ``field_fade_alpha(1.0)`` is 0.0.

    This is the container and its outline only. The text is drawn *after*
    the ramp, at full alpha, and never passes through this function.
    """
    t = max(0.0, min(1.0, float(t)))
    return 1.0 - t ** FIELD_FADE_EXPONENT


def field_fade_profile(stops: int = FIELD_FADE_STOPS):
    """The sampled ramp as ``((t, alpha), ...)``, left edge first.

    What a ``QLinearGradient`` is built from, and what a test asserts the
    shape of without needing a QApplication.
    """
    stops = max(2, int(stops))
    return tuple((i / (stops - 1), field_fade_alpha(i / (stops - 1)))
                 for i in range(stops))


def field_chrome(theme: str = "dark") -> Dict[str, object]:
    """Return theme colors and geometry for faded field containers.

    Parameters
    ----------
    theme : str, default="dark"
        Theme name accepted by :func:`palette_for`.

    Returns
    -------
    dict
        Radius, fill, border, focus, and disabled-state tokens. Each color is
        represented as ``(hex_color, alpha)``.

    Notes
    -----
    The fade multiplies each token's alpha so translucent themes retain their
    material. Field chrome is independent of the page-opacity preference.
    """
    base = palette_for(theme)
    glass = theme == "glass"
    return {
        # Glass rounds its inputs to 10px; everything else uses RADIUS.sm.
        "radius": 10.0 if glass else float(RADIUS["sm"]),
        "fill": (base["surface_alt"], 1.0),
        "fill_disabled": (base["surface"], 1.0),
        "border": (("#ffffff", 0.16) if glass else (base["border"], 1.0)),
        "border_focus": (base["accent"], 1.0),
        "border_disabled": (("#ffffff", 0.10) if glass
                            else (base["border_soft"], 1.0)),
    }


def _splash_roles(palette: dict) -> dict:
    """The loading screen's colours, derived from the theme's own surface.

    The splash used to be ``#003737``, sampled from the installer icon. It
    reads as teal because it is teal -- a very dark cyan-green at hue 180 --
    and it was the one full-window surface in the application with a colour
    cast, shown before anything else was on screen.

    It now takes the window's OWN background and foreground, which buys two
    things at once. The colour is black on the dark theme, as asked. And the
    splash matches the window that replaces it exactly, so the handover has
    nothing to flash: a splash one shade off the first painted window is
    visible precisely at the moment the loading screen exists to hide.

    The ink follows for the same reason it must -- white on the light
    theme's near-white surface would be invisible -- and inherits the
    contrast the theme already guarantees for body text.

    The installer ICON keeps its own colour. This is the splash only.
    """
    ink = str(palette.get("fg", "#ffffff"))
    bg = str(palette.get("bg", "#000000"))
    # The dim weight is SOLVED, not fixed, for the same reason the scrims
    # are. A fixed alpha does not mean a fixed contrast: dark ink fading
    # toward a light surface loses contrast faster than white ink fading
    # toward black gains it, so the 110 that read at 3.04:1 on the dark
    # theme read at 2.31:1 on the light one -- under the floor, on the
    # screen that is up while the user has nothing else to look at.
    dim = splash_dim_alpha(ink, bg)
    # COMPOSITED TO OPAQUE HEX, not stored as `rgba(...)`. Every palette
    # value is #rrggbb -- `test_palette_values_are_hex` is the contract --
    # and these three are only ever painted ON the splash background, so
    # flattening them against it is exact rather than an approximation. The
    # widget then draws opaque colours and does no alpha maths at all.
    return {
        "splash_bg": bg,
        "splash_ink": ink,
        "splash_ink_dim": _composite(ink, bg, dim),
        "splash_track": _composite(ink, bg, 45),
        "splash_fill": _composite(ink, bg, 200),
    }


def splash_dim_alpha(ink: str, bg: str, *, target: float = 3.0,
                     floor: int = 110) -> int:
    """The lowest alpha at which ``ink`` still reads ``target`` over ``bg``.

    Unlit phases are meant to look unreached, so this searches UP from
    ``floor`` rather than starting bright: dim enough to read as pending,
    legible enough to read at all.
    """
    for alpha in range(int(floor), 256):
        if _contrast(_composite(ink, bg, alpha), bg) >= target:
            return alpha
    return 255


def _composite(fg: str, bg: str, alpha: int) -> str:
    """``fg`` painted at ``alpha`` over ``bg``, as the painter blends it."""
    fr, fg_, fb = _channels(fg, _UNREADABLE)
    br, bg_, bb = _channels(bg, _UNREADABLE)
    a = max(0, min(255, int(alpha))) / 255.0
    return "#%02x%02x%02x" % tuple(
        int(round(f * a + b * (1 - a)))
        for f, b in ((fr, br), (fg_, bg_), (fb, bb)))


def _relative_luminance(hex_colour: str) -> float:
    def channel(value: int) -> float:
        v = value / 255.0
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (channel(c) for c in _channels(hex_colour, _UNREADABLE))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a: str, b: str) -> float:
    la, lb = _relative_luminance(a), _relative_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


# ---------------------------------------------------------------------------
# spaceout — the same application, wearing something else
# ---------------------------------------------------------------------------
# `spaceout` is a console entry point beside `spacr` and `spacr-qt`
# (:mod:`spacr.qt.spaceout`). It starts the same application — the same
# screens, the same modules, the same settings — and changes only the
# dressing: the palette goes rainbow, and the ambient backdrop draws moving
# fractals instead of drifting blobs.
#
# THE CHOICE IS MADE ONCE, AT LAUNCH, AND IS STORED NOWHERE. It lives here
# as process state because this module is the funnel every colour in the
# application already passes through: `palette_for` is what `stylesheet`,
# `apply_qpalette`, `active_palette`, `page_colour`, every contrast check and
# every widget that paints its own pixels end up calling, so nothing else has
# to learn that the mode exists. Handing it to `spacr.qt.preferences` would
# make it survive a restart and leak the dressing into an ordinary `spacr`
# start, which is the one thing the request rules out — so it is written
# nowhere, no Preferences control offers it, and the entry point is the only
# way in.
#
# THE THEME CONTRACT DOES NOT CHANGE. `resolve_effective_theme` still answers
# one of `THEMES`, `THEMES` is still those four, and the light/dark handling
# every screen reads goes on working. spaceout re-hues whichever theme was
# resolved; it does not become a fifth one, and a light start stays light.
#
# READABILITY IS THE CONSTRUCTION HERE, not a table of colours somebody
# eyeballed afterwards. Every role moves in HUE ONLY and keeps its own WCAG
# relative luminance, and three checks fall out of that identity:
#
#   * `contrast_ratio` is a function of relative luminance and of nothing
#     else, so every rule in `CONTRAST_RULES` measures what it measured on
#     the theme being re-hued;
#   * `lightness` (CIE L*) is a function of relative luminance too, so
#     `page_separation_report` — can you see the panel — is preserved with
#     it;
#   * `max_background_luma` is a minimum over luminances, so the exposure
#     the imagery is solved down to does not move either.
#
# The identity is exact in the reals and within 8-bit rounding on screen;
# `_hue_shift` picks the closest representable colour on the hue line rather
# than the first one that fits, which keeps the drift to a few thousandths of
# a luminance level.
#
# The one place it does not carry on its own is a TRANSLUCENT surface. An
# image theme composites its panels over the wallpaper channel by channel in
# sRGB, and two colours of equal luminance but different hue do not
# composite to equal luminance. That is why `SCRIM_ALPHA` is re-solved when
# the mode is enabled — `_solve_scrims` was written as a solve rather than a
# table for exactly this case, and it says so.

#: Where each palette role lands on the spectrum, in degrees of hue.
#:
#: The surfaces sweep it — violet window, magenta page, blue and cyan and
#: green panels — because those are the large areas, and they are what makes
#: the application read as rainbow rather than as a blue application with
#: coloured buttons.
#:
#: The STATUS roles deliberately do not sweep. `error` stays at the red end,
#: `warning` in the ambers and `success` in the greens, because their whole
#: job is to be recognised before they are read; a dressing that makes a
#: failure look like a success is a broken theme, not a trippy one. They are
#: re-hued — five degrees, fifty, a hundred and thirty — so they belong to
#: the same spectrum as everything else, but they stay in their own
#: neighbourhood of it.
#:
#: `info` tracks `accent` and `chip_value` tracks `success`, the way they
#: already do in every shipped palette.
SPACEOUT_HUES: Dict[str, float] = {
    # Surfaces, sweeping violet -> magenta -> blue -> cyan -> green.
    "bg":          285.0,
    "page":        300.0,
    "surface":     250.0,
    "surface_alt": 205.0,
    "surface_hi":  165.0,
    "border":      320.0,
    "border_soft": 265.0,
    # Text. `fg` is the extreme of its theme's range — white on the dark
    # themes, near-black on the light one — and a hue cannot move a colour
    # that is already at the top or the bottom of the luminance scale, so
    # this entry mostly documents where the light theme's ink goes.
    "fg":          210.0,
    "fg_muted":     45.0,
    "fg_dim":       25.0,
    # Interactive.
    "accent":      190.0,
    "accent_hi":   175.0,
    "accent_lo":   215.0,
    "accent_soft": 275.0,
    "info":        190.0,
    # Status — see the note above.
    "success":     130.0,
    "warning":      50.0,
    "error":         5.0,
    "chip_class":  165.0,
    "chip_value":  115.0,
    # The theme-invariant button roles. They are re-hued by the same table
    # for every theme, so they stay invariant across themes inside the
    # dressing exactly as they are outside it.
    "button_accent":     305.0,
    "button_accent_hi":  320.0,
    "button_accent_lo":  290.0,
    "button_accent_ink": 340.0,
}

#: How far toward its hue a role is taken. Everything not listed goes all
#: the way.
#:
#: The two that are held back are the two the ANIMATION IS PAINTED ONTO, and
#: the number is measured rather than judged. On a dark page the ambient
#: backdrop composites ADDITIVELY, so what reaches the eye is the page's own
#: channels plus the animation's. At full saturation the dressed page is
#: ``#480048`` — 72 of red and 72 of blue and none of green — and the
#: fractal's green at its peak alpha adds 67. The green never wins, and the
#: measured result was a rainbow palette rendering as four neighbouring
#: hues: blue through magenta to red, and nothing else, whatever the
#: animation was drawing underneath.
#:
#: Damped to a third, the same luminance is spread across all three channels
#: instead of piled into two, and every colour the animation draws clears
#: it. The page is still unmistakably not the ordinary grey — it is a plum —
#: and the rainbow it was hiding is now visible. See
#: ``tests/qt/test_spaceout_fractals_move_and_stay_in_budget.py``, which
#: counts the hue families in a real painted frame.
SPACEOUT_SATURATION: Dict[str, float] = {
    "bg":   0.35,
    "page": 0.35,
}


def _hue_rgb(hue: float, saturation: float = 1.0) -> Tuple[float, float,
                                                           float]:
    """sRGB for ``hue`` in degrees at ``saturation``, as 0..1 per channel.

    The HSV ``V=1`` plane, written out rather than imported so this module
    keeps its short import list. ``saturation`` 1.0 is the pure hue; 0.0 is
    white, and every value between mixes the two, which is what
    :data:`SPACEOUT_SATURATION` asks for.
    """
    position = (float(hue) % 360.0) / 60.0
    ramp = 1.0 - abs(position % 2.0 - 1.0)
    pure = ((1.0, ramp, 0.0), (ramp, 1.0, 0.0), (0.0, 1.0, ramp),
            (0.0, ramp, 1.0), (ramp, 0.0, 1.0), (1.0, 0.0, ramp))[
                int(position) % 6]
    weight = max(0.0, min(1.0, float(saturation)))
    return tuple(1.0 - weight * (1.0 - channel) for channel in pure)


#: The sRGB transfer function, tabulated for all 256 levels.
#:
#: :func:`_hue_shift` scores 512 candidate colours per ``(colour, hue)``
#: pair, and doing that through :func:`_relative_luminance` would mean
#: formatting each one to hex and parsing it straight back — a hundredfold
#: on the only part of this that is not free.
_LINEAR_CHANNEL: Tuple[float, ...] = tuple(
    (level / 255.0) / 12.92 if level / 255.0 <= 0.03928
    else ((level / 255.0 + 0.055) / 1.055) ** 2.4
    for level in range(256))


def _rgb_luminance(rgb: Tuple[int, int, int]) -> float:
    """WCAG relative luminance of an 8-bit ``(r, g, b)`` triple."""
    return (0.2126 * _LINEAR_CHANNEL[rgb[0]]
            + 0.7152 * _LINEAR_CHANNEL[rgb[1]]
            + 0.0722 * _LINEAR_CHANNEL[rgb[2]])


#: Levels either side of the crossing :func:`_hue_shift` measures.
#:
#: The two ramps are monotone but only weakly: 8-bit rounding leaves short
#: plateaus where two neighbouring levels land on the same luminance, and a
#: linear scan keeping the first strict improvement returns the LOWEST index
#: on such a plateau. Eight is far wider than any plateau either ramp can
#: produce — a fully saturated hue moves at least one channel on every
#: level — so the window contains the whole tie and the lowest index in it
#: still wins.
_HUE_WINDOW = 8


@lru_cache(maxsize=None)
def _hue_shift(colour: str, hue: float, saturation: float = 1.0) -> str:
    """``colour`` moved to ``hue``, at the relative luminance it already had.

    Two ramps are searched, and both are needed because a saturated hue can
    only reach part of the luminance scale — fully saturated blue tops out
    at 0.0722, and the light theme's ``surface`` is 1.0:

    * **value**, at full saturation: the hue's own colour, darkened. This is
      the arm that answers for the surfaces and for most of the ink, and it
      is the one that makes the result *look* like a rainbow.
    * **saturation**, at full value: the hue mixed toward white, for the
      roles that need more light than the hue itself carries. This is what
      keeps a white ``fg`` white instead of substituting a violet nobody
      could read a settings form in.

    The closest of all 512 candidates wins, so the 8-bit rounding error is
    minimised rather than merely bounded. Cached because ``palette_for`` is
    on the path of every stylesheet build and every widget that paints.

    THE 512 ARE NOT ALL VISITED. Both ramps are monotone in luminance —
    every channel of the value ramp rises with the level and every channel
    of the saturation ramp falls with the step — so the closest entry is
    found by crossing rather than by scanning, and only a window either side
    of the crossing is measured. That is 20-odd candidates instead of 512
    and it is not an approximation: ``test_spaceout_looks_alive.py`` asserts
    the answer is identical to the full scan for every colour, hue and
    saturation the dressing can produce. It matters because the drift asks
    for the whole palette at :data:`SPACEOUT_DRIFT_STEPS` offsets rather
    than once, and the full scan spent a second of the launcher's startup
    doing it.
    """
    target = _relative_luminance(colour)
    base = _hue_rgb(hue, saturation)

    def value(level: int) -> Tuple[int, int, int]:
        return (int(round(base[0] * level)),
                int(round(base[1] * level)),
                int(round(base[2] * level)))

    def tint(step: int) -> Tuple[int, int, int]:
        weight = step / 255.0
        return (int(round(255.0 * (1.0 - weight + weight * base[0]))),
                int(round(255.0 * (1.0 - weight + weight * base[1]))),
                int(round(255.0 * (1.0 - weight + weight * base[2]))))

    best, error = value(0), abs(_rgb_luminance(value(0)) - target)
    for ramp, rising in ((value, True), (tint, False)):
        low, high = 0, 255
        while low < high:
            mid = (low + high) // 2
            here = _rgb_luminance(ramp(mid))
            if (here < target) if rising else (here > target):
                low = mid + 1
            else:
                high = mid
        for index in range(max(0, low - _HUE_WINDOW),
                           min(256, low + _HUE_WINDOW + 1)):
            candidate = ramp(index)
            miss = abs(_rgb_luminance(candidate) - target)
            if miss < error:
                best, error = candidate, miss
    return "#%02x%02x%02x" % best


def spaceout_palette(palette: dict, drift: float = 0.0,
                     theme: Optional[str] = None) -> dict:
    """Re-hue a theme palette while preserving accessible luminance.

    Roles absent from :data:`SPACEOUT_HUES` are returned unchanged. Named ink
    roles are constrained to contrast-safe luminance bands for ``theme``.

    :param palette: Mapping from theme roles to colour values.
    :param drift: Hue rotation in degrees applied to all mapped roles.
    :param theme: Theme used to resolve contrast-safe ink bands, or ``None``
        for a direct hue shift.
    :returns: A new role-to-colour mapping.
    """
    bands = _INK_BANDS.get(theme, {}) if theme else {}
    damping = _PAGE_DAMPING.get(theme, {}).get(float(drift), 1.0) \
        if theme else 1.0
    out = {}
    for role, colour in palette.items():
        seat = SPACEOUT_HUES.get(role)
        if seat is None:
            out[role] = colour
            continue
        hue = (seat + float(drift)) % 360.0
        band = bands.get(role)
        if band:
            out[role] = _hue_ink(hue, band[0], band[1], colour)
            continue
        saturation = SPACEOUT_SATURATION.get(role, 1.0)
        if role in SPACEOUT_DAMPED_ROLES:
            saturation *= damping
        out[role] = _hue_shift(colour, hue, saturation)
    return out


# ---------------------------------------------------------------------------
# The drift — the spectrum turns, and the readability turns with it
# ---------------------------------------------------------------------------
# The dressing above is a *table*: every role lands on one hue and stays
# there. This turns the whole table, slowly and without a loop a watcher can
# learn, so the application is not one rainbow but a rainbow that moves.
#
# WHY THIS COSTS NOTHING IN READABILITY, and why that is a property of the
# construction rather than a claim: `_hue_shift` moves a role in hue and
# leaves its WCAG relative luminance where it was, and `contrast_ratio` is a
# function of relative luminance and of NOTHING ELSE. Adding a constant to
# every hue therefore leaves every ratio in `CONTRAST_RULES` exactly where it
# was — at every point on the drift, not only at the one it started from.
#
# TWO THINGS DO NOT CARRY ON THEIR OWN, and both are solved rather than
# hoped for:
#
#   * a TRANSLUCENT surface. An image theme composites its panels over the
#     wallpaper channel by channel in sRGB, and two colours of equal
#     luminance and different hue do not composite to equal luminance. The
#     alphas were already re-solved when the dressing went on; they are now
#     solved for the WORST POINT ON THE DRIFT instead of for one palette.
#     Measured: keeping the alphas solved at the starting hue, the wallpaper
#     stops reading through the panels at 74 of the 360 one-degree offsets,
#     down to 1.37:1 against a 1.50:1 rule.
#   * the INK, which is the other half of the request — see
#     :data:`SPACEOUT_INK_ROLES`.
#
# THE CLOCK IS DRIVEN, NOT READ. `advance_spaceout_drift` is called by the
# things that are already painting frames — the ambient backdrop's tick and
# the setup card's — rather than by `palette_for` reading a wall clock. Two
# reasons, and the second is the load-bearing one: a backdrop the user
# turned off should not be quietly replaced by a palette animating instead,
# and `palette_for` is on the path of every stylesheet build and every
# widget that paints, so a wall clock inside it would make the palette a
# different value on two calls in the same frame.

#: Seconds for the spectrum to travel once round, before the wander.
#:
#: Nine minutes. A backdrop must never look like it is *moving*, only like
#: it has moved when you look back at it — the same figure the blob drift
#: and the fractal's own spin are set by.
SPACEOUT_DRIFT_TURN = 540.0

#: The wander, as ``(share of a turn, period in seconds, phase in turns)``.
#:
#: WHAT MAKES IT NOT A LOOP. On its own the term above is a metronome: the
#: hue advances by the same amount every second and a watcher learns the
#: cycle. These three add a wander whose periods are mutually incommensurate
#: with each other and with the turn, so the *sequence* of hues — fast here,
#: backing up there, dwelling somewhere else — does not repeat on any period
#: short enough to be learned. They are amplitudes on the ANGLE, so the
#: drift can slow, stall and briefly reverse without ever jumping.
SPACEOUT_DRIFT_WANDER: Tuple[Tuple[float, float, float], ...] = (
    (0.070, 149.0, 0.137),
    (0.041, 76.3, 0.611),
    (0.023, 31.7, 0.283),
)

#: How many hue offsets the *palette* is allowed to take.
#:
#: The drift itself is continuous — :func:`spaceout_drift` — and the things
#: that repaint every frame use it that way. The palette is quantised onto
#: this grid instead, for one reason: every offset on it has to be SOLVED,
#: because the scrim alphas and the ink bands below are worst cases over the
#: offsets the palette can actually reach. A continuous palette would be a
#: continuum of solves.
#:
#: Sixty is six degrees a step and one step every nine seconds. Six degrees
#: moves a saturated surface by two or three 8-bit levels, which is under
#: the step the eye resolves on a large flat area, and the QSS chrome only
#: re-reads the palette when the stylesheet is rebuilt anyway.
SPACEOUT_DRIFT_STEPS = 60

#: The roles whose colour is solved for CHROMA rather than carried over.
#:
#: "the color of the text is pretty good but could be more rainbow like."
#: The reason it was not is the identity the rest of the dressing rests on:
#: hue moves, luminance does not — and a role already at the top of the
#: luminance scale cannot carry a hue at all. Dark's `fg` is ``#ffffff``, so
#: re-hueing it returns ``#ffffff``, and the body text of the application is
#: the one thing in it that was not in the rainbow.
#:
#: So for these three the luminance is allowed to MOVE, inside a band solved
#: from :data:`CONTRAST_RULES` — see :func:`_ink_band` — and the most
#: chromatic colour on the hue line inside that band is taken. On dark that
#: turns ``#ffffff`` into a fully saturated ``#00d5ff`` at the hue the table
#: gives `fg`, and it still clears 4.5:1 on every surface with
#: :data:`SPACEOUT_INK_HEADROOM` to spare.
#:
#: THE CHECK IS WHAT DECIDES HOW FAR IT GOES. Where the band is narrow the
#: answer is a pale tint, and that is the right answer: a trippy theme that
#: cannot be read is a broken theme.
SPACEOUT_INK_ROLES: Tuple[str, ...] = ("fg", "fg_muted", "fg_dim")

#: Multiplier on every WCAG minimum when solving an ink band, so a solved
#: ink is not sitting exactly on the line — the same reason
#: :data:`SCRIM_HEADROOM` exists, and much larger than it, for two reasons
#: that both come from what is UNDER the text.
#:
#: The scrims are re-solved AFTER the ink and must not be able to push it
#: under, which is the small half. The large half is that
#: :data:`CONTRAST_RULES` judges ink against a surface role, and some panels
#: in the application are painted translucent by the WIDGET rather than by
#: the theme — ``SetupCard`` lays its body down at alpha 216 so the backdrop
#: shows through it, and under ``spaceout`` that backdrop is a bright
#: fractal. Measured on the rendered first-run card over a real frame: at
#: 1.12 the heading came out at 4.56:1 against a 4.5:1 rule, which is inside
#: the rule and outside any comfort. At 1.30 the same measurement is 5.6:1
#: and ``fg`` is still a saturated blue rather than the white it was.
SPACEOUT_INK_HEADROOM = 1.30

#: The saturations tried when the drift breaks the page separation, in
#: order. The first one that clears the rule wins, so a drift offset that
#: never had a problem keeps the full colour.
#:
#: WHY THIS EXISTS, and it is the same reason the scrims are re-solved. The
#: contrast rules survive a re-hue by construction, because a ratio is a
#: function of relative luminance alone — but `page_separation_report` asks
#: whether you can SEE the panel, and half of its rows composite the panel
#: over the page at :data:`PAGE_FADED_OPACITY`. That composite happens
#: channel by channel in sRGB, and two colours of equal luminance and
#: different hue do not composite to equal luminance. Measured over the
#: sixty offsets the palette can take: the light theme's faded
#: ``surface_alt`` drops to 1.069:1 against a 1.08:1 rule at four of them.
#:
#: So at those offsets — and only at those — the page and the panels are
#: mixed back toward white until the panel separates again. It costs
#: saturation on four sixtieths of the drift and it buys a page you can
#: still see the panels on, which is the trade the request names outright.
SPACEOUT_DAMPING_STEPS: Tuple[float, ...] = (1.0, 0.75, 0.55, 0.40, 0.28,
                                             0.20, 0.12)

#: Extra saturation damping for the page and its panels, as
#: ``{theme: {drift offset: multiplier}}``. An offset that is not in the
#: table needs no damping, which is nearly all of them.
_PAGE_DAMPING: Dict[str, Dict[float, float]] = {}

#: Solved damping, keyed by whether the dressing is on — the twin of
#: :data:`_SOLVED_SCRIMS`.
_SOLVED_DAMPING: Dict[bool, Dict[str, Dict[float, float]]] = {}

#: The roles the damping reaches: the page itself and the panels that have
#: to stay visible on it.
#:
#: Written out rather than built from :data:`PAGE_PANEL_ROLES`, which is
#: declared further down the module; ``test_spaceout_looks_alive.py`` asserts
#: the two agree so the pair cannot drift apart.
SPACEOUT_DAMPED_ROLES: Tuple[str, ...] = ("page", "surface", "surface_alt")

#: Elapsed animation seconds the drift is at. Advanced by the widgets that
#: are already painting frames; never read off a wall clock.
_DRIFT_SECONDS = 0.0

#: Solved ink bands for the current dressing, ``{theme: {role: (lo, hi)}}``.
#: Empty when the dressing is off, which is what makes every ink role fall
#: back to the plain hue shift.
_INK_BANDS: Dict[str, Dict[str, Tuple[float, float]]] = {}

#: Solved ink bands, keyed by whether the dressing is on — the twin of
#: :data:`_SOLVED_SCRIMS`, and cached for the same reason.
_SOLVED_INK: Dict[bool, Dict[str, Dict[str, Tuple[float, float]]]] = {}

#: While a solve is running: the hue offset to dress at, and whether the ink
#: treatment is applied. `palette_for` consults both, which is what lets the
#: solvers call the ordinary public helpers — `effective_surface`,
#: `scrim_under`, `max_background_luma` — instead of restating them, and
#: what stops the ink solve recursing into the palette it is solving.
_SOLVE_DRIFT: Optional[float] = None
_SOLVE_INK = True


@contextmanager
def _dressed_at(drift: float, ink: bool = True):
    """Resolve palettes at hue offset ``drift`` for the duration.

    ``ink`` False leaves the ink roles on the plain hue shift, which is what
    :func:`_ink_band` needs: it is solving the band the ink will be chosen
    from, and it reads the surfaces through :func:`effective_surface`, which
    goes back through :func:`palette_for`.
    """
    global _SOLVE_DRIFT, _SOLVE_INK
    was = (_SOLVE_DRIFT, _SOLVE_INK)
    _SOLVE_DRIFT, _SOLVE_INK = float(drift), bool(ink)
    try:
        yield
    finally:
        _SOLVE_DRIFT, _SOLVE_INK = was


def _drift_grid() -> Tuple[float, ...]:
    """The hue offsets the palette can take, in degrees."""
    return tuple(index * 360.0 / SPACEOUT_DRIFT_STEPS
                 for index in range(SPACEOUT_DRIFT_STEPS))


def spaceout_drift(at: Optional[float] = None) -> float:
    """Return the continuous spaceout hue rotation in degrees.

    :param at: Elapsed animation time in seconds. ``None`` uses the current
        drift clock.
    :returns: Hue rotation in ``[0, 360)``, or zero when spaceout is disabled.
    """
    if not _SPACEOUT:
        return 0.0
    elapsed = _DRIFT_SECONDS if at is None else float(at)
    turns = elapsed / SPACEOUT_DRIFT_TURN
    for share, period, phase in SPACEOUT_DRIFT_WANDER:
        turns += share * math.sin(2.0 * math.pi * (elapsed / period + phase))
    return (turns * 360.0) % 360.0


def spaceout_drift_step(at: Optional[float] = None) -> float:
    """Return :func:`spaceout_drift` quantised to its solved palette grid."""
    if not _SPACEOUT:
        return 0.0
    step = 360.0 / SPACEOUT_DRIFT_STEPS
    return (round(spaceout_drift(at) / step) % SPACEOUT_DRIFT_STEPS) * step


def spaceout_drift_seconds() -> float:
    """Return the elapsed spaceout animation time in seconds."""
    return _DRIFT_SECONDS


def advance_spaceout_drift(dt: float) -> float:
    """Advance the spaceout clock and return the resulting hue rotation.

    Non-positive intervals and calls made while spaceout is disabled do not
    modify the clock.
    """
    global _DRIFT_SECONDS
    if _SPACEOUT and dt > 0:
        _DRIFT_SECONDS += float(dt)
    return spaceout_drift()


def set_spaceout_drift_seconds(seconds: float) -> None:
    """Set the spaceout animation clock, clamped to zero or greater."""
    global _DRIFT_SECONDS
    _DRIFT_SECONDS = max(0.0, float(seconds))


@lru_cache(maxsize=None)
def _hue_ink(hue: float, low: float, high: float, fallback: str) -> str:
    """The most chromatic colour on ``hue`` whose luminance is in the band.

    The same 512 candidates :func:`_hue_shift` scores — the value ramp at
    full saturation and the saturation ramp at full value — judged on a
    different question. :func:`_hue_shift` asks which one is closest to a
    luminance it must keep; this asks which one is the most COLOURED of
    those the readability band allows, because for the ink the luminance is
    the constraint and the colour is the point.

    Falls back to the plain hue shift when the band admits nothing, which is
    what a role whose rules leave it no room gets: unchanged and readable.
    """
    base = _hue_rgb(hue)
    best: Optional[Tuple[int, int, int]] = None
    chroma = -1
    for level in range(256):
        candidate = (int(round(base[0] * level)),
                     int(round(base[1] * level)),
                     int(round(base[2] * level)))
        if low <= _rgb_luminance(candidate) <= high:
            spread = max(candidate) - min(candidate)
            if spread > chroma:
                best, chroma = candidate, spread
    for step in range(256):
        weight = step / 255.0
        candidate = (
            int(round(255.0 * (1.0 - weight + weight * base[0]))),
            int(round(255.0 * (1.0 - weight + weight * base[1]))),
            int(round(255.0 * (1.0 - weight + weight * base[2]))))
        if low <= _rgb_luminance(candidate) <= high:
            spread = max(candidate) - min(candidate)
            if spread > chroma:
                best, chroma = candidate, spread
    if best is None:
        return _hue_shift(fallback, hue)
    return "#%02x%02x%02x" % best


def _ink_band(theme: str, role: str) -> Optional[Tuple[float, float]]:
    """The luminances ``role`` may take in ``theme`` and still be read.

    Closed form per rule rather than a search. For an ink of luminance
    ``L`` on a surface of luminance ``Ls``, WCAG asks
    ``(hi + 0.05) / (lo + 0.05) >= r``; an ink that is the LIGHTER of the
    pair is therefore bounded below by ``(Ls + 0.05) * r - 0.05`` and a
    darker one bounded above by ``(Ls + 0.05) / r - 0.05``. The band is the
    tightest of those over every rule in :data:`CONTRAST_RULES` that names
    the role, at every offset the drift can reach — the surfaces keep their
    luminance under the dressing, but an image theme's surfaces are
    composited over the wallpaper and those do move with hue.

    One more bound, and it is what keeps the rest of the module honest:
    :func:`max_background_luma` is a minimum over ink luminances, and the
    imagery pipeline exposure-solves every wallpaper down to it. Letting the
    ink darken would silently darken every photograph in the application, so
    each role that feeds that minimum is held at or above the luminance it
    needs to leave the ceiling where it was.

    ``None`` when the role sits between its surfaces, or when the bounds
    cross — both of which mean there is no room to spend and the plain hue
    shift is the right answer.
    """
    rules = tuple((surface, required)
                  for fg, surface, required in CONTRAST_RULES if fg == role)
    if not rules:
        return None
    plain = dict(_PALETTES.get(theme, DARK_PALETTE))
    plain.update(CONSTANT_ROLES)
    ink = relative_luminance(plain[role])
    low, high = 0.0, 1.0
    for drift in _drift_grid():
        with _dressed_at(drift, ink=False):
            surfaces = [relative_luminance(effective_surface(theme, surface))
                        for surface, _ in rules]
        if ink >= max(surfaces):
            for (_surface, required), luma in zip(rules, surfaces):
                low = max(low, (luma + 0.05) * required
                          * SPACEOUT_INK_HEADROOM - 0.05)
        elif ink <= min(surfaces):
            for (_surface, required), luma in zip(rules, surfaces):
                high = min(high, (luma + 0.05) / (required
                                                  * SPACEOUT_INK_HEADROOM)
                           - 0.05)
        else:
            return None
    with _dressed_at(0.0, ink=False):
        keep = max_background_luma(theme)
    for named, required in BARE_IMAGE_RULES:
        if named == role:
            low = max(low, (keep + 0.05) * required - 0.05)
    low, high = max(0.0, low), min(1.0, high)
    return (low, high) if high - low > 1e-6 else None


def _solve_page_damping() -> Dict[str, Dict[float, float]]:
    """How much colour each theme has to give up, at each drift offset, for
    its panels to stay visible on its page.

    Solved by *trying* rather than by arithmetic, because the rule it is
    solving against — :func:`page_separation_failures` — is two
    measurements, one of them in CIE L*, and reading them backwards to a
    saturation would be a second implementation of the thing it has to
    agree with. Seven candidates over sixty offsets is 130 ms once.

    The candidate under test is written straight into :data:`_PAGE_DAMPING`
    so :func:`page_separation_failures` sees it through the palette, which
    is what makes this the published rule judging the published colours
    rather than a copy of either.
    """
    _PAGE_DAMPING.clear()
    for name in THEMES:
        rows: Dict[float, float] = {}
        _PAGE_DAMPING[name] = rows
        for drift in _drift_grid():
            for damping in SPACEOUT_DAMPING_STEPS:
                rows[drift] = damping
                with _dressed_at(drift):
                    if not page_separation_failures(name):
                        break
            if rows[drift] >= 1.0:
                del rows[drift]
    return {name: dict(rows) for name, rows in _PAGE_DAMPING.items()}


def _solve_ink_bands() -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Every ink band of every theme. Solved once per dressing."""
    out: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for name in THEMES:
        rows = {}
        for role in SPACEOUT_INK_ROLES:
            band = _ink_band(name, role)
            if band is not None:
                rows[role] = band
        out[name] = rows
    return out


def _scrim_bounds(palette: dict, role: str, colour_role: str,
                  under: Tuple[int, int, int]) -> Tuple[float, float]:
    """:func:`legible_scrim_floor` and :func:`present_scrim_ceiling`, in one
    pass over 8-bit channels rather than over hex strings.

    Exactly the two published solvers and it has to stay exactly them —
    ``tests/qt/test_spaceout_looks_alive.py`` asserts the answers match for
    every role of every image theme. What it is not is their cost: those
    format a colour to hex and parse it straight back once per step of a
    thousand-step sweep, and the drift asks for the pair at every offset in
    :func:`_drift_grid` rather than once.

    The ceiling is found coarse-to-fine. :func:`picture_contrast` falls as
    the panel thickens, so the first coarse step that still shows the
    picture puts the answer inside the block above it, and the block is then
    walked from the top. Not a bisection: the fall is monotone in the reals
    but 8-bit rounding makes it wobble by two or three thousandths, and a
    bisection lands on the wrong side of the wobble.
    """
    # `colour_role or role`, matching the two published solvers exactly --
    # which this docstring insists on and, until 310 A7, did not do. Both
    # `picture_contrast` and `present_scrim_ceiling` take `colour_role` as
    # Optional and fall back to `role`; here `role` was accepted and never
    # read, so a caller passing the documented `colour_role=None` got
    # `KeyError: None` out of the drift solver at import rather than that
    # role's own bounds. Not reachable today -- every SCRIM_ROLES value is a
    # non-None string -- but the dead parameter also hid the divergence from
    # anyone comparing the three implementations, which is the comparison the
    # docstring asks them to make.
    base = _channels(palette[colour_role or role])
    inks = tuple((_rgb_luminance(_channels(palette[fg])),
                  required * SCRIM_HEADROOM)
                 for fg, required in _scrim_rules(colour_role or role))

    def over(alpha: float, beneath: Tuple[int, int, int]) -> float:
        rest = 1.0 - alpha
        return _rgb_luminance((
            int(round(alpha * base[0] + rest * beneath[0])),
            int(round(alpha * base[1] + rest * beneath[1])),
            int(round(alpha * base[2] + rest * beneath[2]))))

    floor = 1.0
    for step in range(0, 1001):
        panel = over(step / 1000.0, under)
        if all((max(luma, panel) + 0.05) / (min(luma, panel) + 0.05) >= need
               for luma, need in inks):
            floor = step / 1000.0
            break

    def shows(step: int) -> bool:
        alpha = step / 1000.0
        lit, dark = over(alpha, under), over(alpha, (0, 0, 0))
        return ((max(lit, dark) + 0.05) / (min(lit, dark) + 0.05)
                >= MIN_PICTURE_CONTRAST)

    coarse = 1000
    while coarse > 0 and not shows(coarse):
        coarse -= 25
    ceiling = 0.0
    for step in range(min(1000, coarse + 25), max(-1, coarse - 1), -1):
        if shows(step):
            ceiling = step / 1000.0
            break
    return floor, ceiling


def _solve_scrims_over_drift() -> Dict[str, Dict[str, float]]:
    """Scrim alphas that hold at every offset the drift can reach.

    The bounds :func:`solve_scrim_alpha` weighs are the same two, taken as a
    worst case instead of at one palette: the HIGHEST legibility floor and
    the LOWEST see-through ceiling over :func:`_drift_grid`. The answer is
    still the ceiling clamped up to the floor, so where the drift makes the
    two cross, legibility takes it — which is the same way round this module
    has always resolved that pair.
    """
    out: Dict[str, Dict[str, float]] = {}
    for name in IMAGE_THEMES:
        floors: Dict[str, float] = {}
        ceilings: Dict[str, float] = {}
        for drift in _drift_grid():
            with _dressed_at(drift):
                palette = palette_for(name)
                under = _channels(scrim_under(name))
            for role, colour_role in SCRIM_ROLES.items():
                floor, ceiling = _scrim_bounds(palette, role, colour_role,
                                               under)
                floors[role] = max(floors.get(role, 0.0), floor)
                ceilings[role] = min(ceilings.get(role, 1.0), ceiling)
        solved = {role: max(ceilings[role], floors[role])
                  for role in SCRIM_ROLES}
        # Popups are separate top-level windows. Translucency there shows
        # the desktop, not the wallpaper.
        solved["elevated"] = 1.00
        out[name] = solved
    return out


#: Whether this process is wearing the spaceout dressing. Process state,
#: never a stored preference — see the block above.
_SPACEOUT = False

#: Solved scrim alphas, keyed by whether the dressing is on. Populated at
#: import for ``False`` and on the first :func:`enable_spaceout` for
#: ``True``, so flipping the mode costs one solve and never more.
_SOLVED_SCRIMS: Dict[bool, Dict[str, Dict[str, float]]] = {}


def spaceout_enabled() -> bool:
    """Return whether spaceout rendering is enabled for this process."""
    return _SPACEOUT


def _apply_dressing() -> None:
    """Point :data:`SCRIM_ALPHA` and :data:`_INK_BANDS` at the current
    dressing's solved values.

    The image themes paint translucent panels, so their alphas are a
    function of the palette — re-hueing one moves the colour a panel is
    composited from and therefore what it takes for text to stay readable
    over the wallpaper. Under the dressing the palette also *drifts*, so the
    answer is a worst case over the offsets it can reach rather than a
    single solve.

    TWO PASSES, and the order is forced. The ink band is read off the
    surfaces, and an image theme's surfaces are its scrims composited over
    the wallpaper — so the scrims have to exist before the ink can be
    solved. The ink then changes what those panels have to carry, so the
    scrims are solved again against it. :data:`SPACEOUT_INK_HEADROOM` is
    what stops the second pass from invalidating the first: the ink is
    solved with 12 % in hand, and the scrims only ever thicken.

    Solved once per dressing and cached, so taking it off and putting it
    back costs one dict copy.
    """
    solved = _SOLVED_SCRIMS.get(_SPACEOUT)
    bands = _SOLVED_INK.get(_SPACEOUT)
    damping = _SOLVED_DAMPING.get(_SPACEOUT)
    if solved is None or bands is None or damping is None:
        if not _SPACEOUT:
            solved, bands, damping = _solve_scrims(), {}, {}
        else:
            _INK_BANDS.clear()
            # The damping first: it moves the panel colours, and the scrims
            # are solved from those.
            damping = _solve_page_damping()
            SCRIM_ALPHA.clear()
            SCRIM_ALPHA.update(_solve_scrims_over_drift())
            bands = _solve_ink_bands()
            _INK_BANDS.update(bands)
            solved = _solve_scrims_over_drift()
        _SOLVED_SCRIMS[_SPACEOUT] = solved
        _SOLVED_INK[_SPACEOUT] = bands
        _SOLVED_DAMPING[_SPACEOUT] = damping
    SCRIM_ALPHA.clear()
    SCRIM_ALPHA.update(solved)
    _INK_BANDS.clear()
    _INK_BANDS.update(bands)
    _PAGE_DAMPING.clear()
    _PAGE_DAMPING.update({name: dict(rows) for name, rows in damping.items()})


def enable_spaceout() -> None:
    """Enable process-local spaceout rendering.

    This operation is idempotent and does not modify saved preferences.
    """
    global _SPACEOUT, _DRIFT_SECONDS
    if _SPACEOUT:
        return
    _SPACEOUT = True
    _DRIFT_SECONDS = 0.0
    _apply_dressing()


def disable_spaceout() -> None:
    """Disable process-local spaceout rendering and reset its clock."""
    global _SPACEOUT, _DRIFT_SECONDS
    if not _SPACEOUT:
        return
    _SPACEOUT = False
    _DRIFT_SECONDS = 0.0
    _apply_dressing()


def palette_for(theme: str = "dark") -> dict:
    """Return the palette dict for ``theme``.

    ``theme`` is one of :data:`THEMES`; anything else (including
    ``"system"``, which the caller is expected to have resolved) falls
    back to the dark palette. The returned dict always carries every
    theme-invariant key from :data:`CONSTANT_ROLES` so callers can hit
    e.g. ``palette_for(t)["button_accent"]`` and know the value is the
    same across themes.

    Under the ``spaceout`` dressing (:func:`spaceout_enabled`) the result is
    re-hued onto the spectrum on the way out, at whatever offset the drift
    has reached. The keys and the count are unchanged, and so is every
    surface role's relative luminance, so callers, contrast rules and the
    light/dark distinction all go on working. The three ink roles are the
    exception and are solved rather than carried — see
    :data:`SPACEOUT_INK_ROLES`.
    """
    base = _PALETTES.get(theme, DARK_PALETTE)
    out = dict(base)
    out.update(CONSTANT_ROLES)
    if _SPACEOUT:
        # Same keys, same surface luminances, different hues — see the
        # spaceout block above. Applied before the splash roles so those are
        # derived from the colours the window will actually be painted with.
        drift = (spaceout_drift_step() if _SOLVE_DRIFT is None
                 else _SOLVE_DRIFT)
        out = spaceout_palette(out, drift, theme if _SOLVE_INK else None)
    out.update(_splash_roles(out))
    return out


def active_palette() -> dict:
    """The palette for the theme that is **on screen right now**.

    This is what a widget wants. ``DARK_PALETTE`` is a constant; the
    theme is a preference, and it changes while the process is running.
    A widget that inlines colours — anything that builds its own
    ``setStyleSheet`` string, and anything that paints in a
    ``paintEvent`` — must resolve them through here, per instance, at
    construction or paint time.

    Screens are rebuilt on a theme change
    (``MainWindow._rebuild_startup_page`` for Home, and the stylesheet is
    re-applied to everything else), so one call per widget build is
    enough; there is no need to cache the result across constructions.

    Falls back to dark if preferences cannot be read — headless, no
    ``QApplication``, a corrupt settings file — because that is what the
    app looked like before this function existed.
    """
    try:
        from .preferences import resolve_effective_theme
        return palette_for(resolve_effective_theme())
    except Exception:
        return palette_for("dark")


# ---------------------------------------------------------------------------
# `page` — the colour of the surface the panels float ON
# ---------------------------------------------------------------------------
# `bg` is the WINDOW colour. It is `QPalette.Window`, it is the ink on a
# filled accent button (`QPalette.HighlightedText`), it is the blanket
# `QWidget { background-color: bg }` in `_window_block`, and in the dark
# theme it is literally `#000000`. Thirty-six uses, most of which are not
# "the page".
#
# For a long time it was the page anyway, by omission. A module screen
# clears its layout containers (`clear_container_surfaces`) so the
# backdrop shows between the settings cards — and when the ambient
# animation is switched off there IS no backdrop, so what showed through
# was the blanket window fill. Pure black. Measured on a real AppScreen
# with `ambient_enabled=False`: 40 of 74 samples down the settings column
# were exactly (0,0,0), and (0,0,0) was the single most common colour on
# the whole screen (17,621 samples, against 11,420 for `surface`). Users
# reported it three times as "a black box behind the settings
# categories"; two fixes swept more containers transparent, which is the
# right sweep and made the hole bigger, because the thing behind the
# containers had no colour of its own.
#
# So the page gets one. `page` is a separate role from `bg` precisely so
# that giving the page a colour cannot change selected-text rendering,
# `QPalette.Window`, or the ink on a pressed button.
#
# The values are solved, not chosen. A page has to satisfy, at once:
#
# * **Separation from the resting panels.** `surface` and `surface_alt`
#   are what a settings category is painted with, and they have to read
#   as panels sitting on something. Judged in CIE L*, because near black
#   the WCAG ratio saturates and stops discriminating — on the dark
#   theme `bg`-to-`surface` is 1.087:1 whatever you do, and `#000000`
#   against `#0d0e10` is 3.95 L* on a scale where the palette's own
#   deliberate step is 3.75. The bar is one full palette step (>=3.5 L*)
#   and >=1.15:1 from each. See `page_separation_report`.
# * **Survival at 60 % page opacity**, where a panel is composited
#   0.6 * panel + 0.4 * page and the separation shrinks by ~40 %. The bar
#   there is >=2.0 L* and >=1.08:1.
# * **Text still lands on it.** Every rule in :data:`CONTRAST_RULES` is
#   enforced against `page` as a surface, so `fg`/`fg_muted`/`accent`
#   clear 4.5:1 and `fg_dim` and the status hues clear 3.0:1 — hint text
#   and disabled controls do sit straight on the page.
#
# On the dark theme those three close to a band two values wide.
# `surface_alt` separation pushes from below and `fg_dim` at 3.0:1 caps
# from above; `#23252a` sits in it at 1.170:1 / 6.96 L* from
# `surface_alt` and 1.259:1 / 10.72 L* from `surface`, with `fg_dim` at
# 3.04:1. The light theme is the same solve mirrored — the page goes
# *down* so white cards lift off it — and closes just as tightly,
# between `surface_alt` from below and `accent` at 4.5:1 from above.
#
# The consequence is that on the dark theme the page is now *lighter*
# than the cards, which is not the usual dark-UI layering and is
# deliberate: it is what the app already looks like with the animation
# on. The ambient backdrop is brighter than `#0d0e10` nearly everywhere
# (2 of 74 samples pure black with it enabled, against 40 with it off),
# so dark cards floating on a lighter field is the established look and
# `page` is the still frame of it, not a new idea.
#
# `surface_hi` is deliberately NOT in the solve. It is the *hover*
# colour: converging toward the page as a card lifts is what hover is
# for, and requiring a step from it as well pushes the page past
# `fg_dim`'s AA ceiling with no band left at all.

def page_colour(theme: str = "dark") -> str:
    """The flat colour the page is painted with under ``theme``.

    Prefer this over ``palette_for(theme)["bg"]`` anywhere the question
    is "what is behind the panels". ``bg`` answers a different question
    — see the block above — and on the dark theme it answers it
    ``#000000``.

    Falls back to ``bg`` for a palette that has no ``page``, so an older
    or third-party palette dict still resolves rather than raising.
    """
    palette = palette_for(theme)
    return palette.get("page") or palette["bg"]


def active_page_colour() -> str:
    """:func:`page_colour` for the theme that is on screen right now.

    Falls back to dark, like :func:`active_palette`, and for the same
    reason: a backdrop must never be the thing that stops a screen
    opening.
    """
    try:
        from .preferences import resolve_effective_theme
        return page_colour(resolve_effective_theme())
    except Exception:
        return page_colour("dark")


# ---------------------------------------------------------------------------
# The name that was a trap
# ---------------------------------------------------------------------------
# `PALETTE` is served by module ``__getattr__`` rather than bound as a
# global, for three reasons:
#
# 1. It cannot be assigned to. `theme.PALETTE = ...` and
#    `PALETTE["fg"] = ...` both fail now, so nobody can "fix" a theme by
#    mutating it and have the change silently apply to every module that
#    already imported it — or, worse, not apply, because half of them
#    copied the value into an f-string at import time.
# 2. `grep -n 'PALETTE ='` over this file no longer finds a definition
#    for it. The dark dict is spelled `DARK_PALETTE` at its definition,
#    where a reader decides whether it is the right thing to import.
# 3. Importing it warns, naming the exact failure, so the remaining
#    call sites are discoverable with `-W error::DeprecationWarning`
#    instead of by eye.
#
# It is kept (rather than deleted) only because the migration is not
# finished: two dozen screens and widgets still import it. Each one is a
# light-theme rendering bug until it moves to `active_palette()`.
_FROZEN_DARK = MappingProxyType(DARK_PALETTE)

_PALETTE_DEPRECATION = (
    "spacr.qt.theme.PALETTE is the DARK palette and nothing updates it, "
    "so inlining it renders dark chrome on the light theme (measured: "
    "1.08:1 ink-on-panel, i.e. black on black). Use "
    "spacr.qt.theme.active_palette() for the theme on screen, or "
    "spacr.qt.theme.DARK_PALETTE if you really do mean the dark colours."
)


def __getattr__(name: str):
    """Serve the deprecated ``PALETTE`` alias (PEP 562)."""
    if name == "PALETTE":
        warnings.warn(_PALETTE_DEPRECATION, DeprecationWarning, stacklevel=2)
        return _FROZEN_DARK
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Colour maths — WCAG contrast
# ---------------------------------------------------------------------------

#: What an unreadable colour becomes for a caller that cannot afford to
#: raise. White is the safe answer for paint: it keeps type and scrims
#: visible instead of blanking them.
_UNREADABLE = (255, 255, 255)


def _channels(color: str,
              fallback: Optional[Tuple[int, int, int]] = None
              ) -> Tuple[int, int, int]:
    """Split ``#rgb`` or ``#rrggbb`` into its three 0-255 channels.

    Parsing is strict by default, so a palette entry that is not a colour is
    reported where someone can fix it. Callers that run inside a paint pass
    ``fallback`` instead: a swatch that comes out the wrong colour is
    cosmetic, an exception raised out of a repaint is not.
    """
    text = str(color).strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    try:
        if len(text) != 6:
            raise ValueError
        return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
    except ValueError:
        if fallback is not None:
            return fallback
        raise ValueError(f"not a #rrggbb colour: {color!r}") from None


def _linear(value: int) -> float:
    c = value / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(color: str) -> float:
    """WCAG relative luminance of a ``#rrggbb`` colour, in [0, 1]."""
    r, g, b = _channels(color)
    return 0.2126 * _linear(r) + 0.7152 * _linear(g) + 0.0722 * _linear(b)


def contrast_ratio(a: str, b: str) -> float:
    """WCAG contrast ratio between two colours — 1.0 (same) to 21.0."""
    la, lb = relative_luminance(a), relative_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def composite(top: str, alpha: float, under: str = WORST_CASE_UNDER) -> str:
    """Alpha-composite ``top`` at ``alpha`` over ``under``, as hex."""
    alpha = max(0.0, min(1.0, float(alpha)))
    tr, tg, tb = _channels(top)
    ur, ug, ub = _channels(under)
    out = tuple(int(round(alpha * t + (1.0 - alpha) * u))
                for t, u in ((tr, ur), (tg, ug), (tb, ub)))
    return "#%02x%02x%02x" % out


#: FULLY OPAQUE. Tried at 0.94 first, on the reasoning that a solid bar
#: would read as a slab pasted over the backdrop. Seen on a real screen
#: that was still wrong -- "remove the transparency for the bar and it
#: will be perfect" -- because this bar is the frameless window's TITLE
#: bar: the backdrop moving behind its two labels is motion under text
#: the eye is trying to read, and no amount of it is an improvement.
#:
#: Kept as a named constant rather than inlined, because the corner
#: chrome and the bar must agree and this is the single thing they agree
#: on. At 1.0 `css_color` returns plain hex, which is also what the flat
#: themes are required to emit.
MENU_BAR_ALPHA = 1.0


def menu_bar_background(theme: Optional[str] = None) -> str:
    """The QSS colour the menu bar and its corner chrome both paint.

    ONE FUNCTION FOR BOTH so they cannot drift. The bar is styled from
    the generated stylesheet and the window chrome is styled in
    ``spacr.qt.app`` with a stylesheet of its own; two hard-coded colours
    that have to match is one of them going stale.

    :param theme: theme name; the active theme when omitted.
    :returns: a QSS colour string.
    """
    if theme is None:
        # The theme ON SCREEN, resolved the way `active_palette` does --
        # the chrome is restyled on a theme change like everything else,
        # so reading a constant here would leave the corner painting the
        # dark bar's colour under the light one.
        try:
            from .preferences import resolve_effective_theme
            theme = resolve_effective_theme()
        except Exception:                                   # noqa: BLE001
            theme = "dark"
    return css_color(palette_for(theme)["surface"], MENU_BAR_ALPHA)


def css_color(color: str, alpha: float = 1.0) -> str:
    """Render a colour for QSS — plain hex, or ``rgba()`` when translucent."""
    if alpha >= 1.0:
        return color
    r, g, b = _channels(color)
    return f"rgba({r}, {g}, {b}, {alpha:.3f})"


def _mix_color(a: str, b: str, amount: float) -> str:
    """Mix two hex colours in sRGB space for small material highlights."""
    amount = max(0.0, min(1.0, float(amount)))
    ac = _channels(a)
    bc = _channels(b)
    out = tuple(int(round(x * (1.0 - amount) + y * amount))
                for x, y in zip(ac, bc))
    return "#%02x%02x%02x" % out


def glass_material(color: str, alpha: float) -> str:
    """Return a neutral, layered QSS brush that suggests optical depth.

    QSS cannot sample and refract pixels behind a widget. A thin bright upper
    layer, translucent neutral body, and slightly denser lower edge provide
    the stable cross-platform cues of glass without pretending opacity alone
    is a material.
    """
    alpha = max(0.0, min(1.0, float(alpha)))
    highlight = _mix_color(color, "#ffffff", 0.16)
    shade = _mix_color(color, "#000000", 0.18)
    return (
        "qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, "
        f"stop: 0 {css_color(highlight, min(1.0, alpha + 0.10))}, "
        f"stop: 0.10 {css_color(color, min(1.0, alpha + 0.035))}, "
        f"stop: 0.72 {css_color(color, alpha)}, "
        f"stop: 1 {css_color(shade, min(1.0, alpha + 0.065))})"
    )


def effective_surface(theme: str, role: str,
                      under: Optional[str] = None) -> str:
    """The colour a surface role *actually* presents to the eye.

    For opaque themes that is just the palette entry — ``under`` cannot
    reach through an alpha of 1.0. For an image theme it is the scrim
    composited over ``under``, which defaults to :func:`scrim_under`:
    the brightest thing *that theme's* wallpaper pipeline can put behind
    a panel. White for Space, whose sky blows its sun out on purpose;
    the exposure ceiling for Cell, whose every wallpaper is solved down
    to it.
    """
    palette = palette_for(theme)
    if under is None:
        under = scrim_under(theme)
    return composite(palette[role], scrim_alpha(theme, role), under)


#: ``(foreground role, surface role, minimum ratio)``.
#:
#: 4.5:1 is AA for body text. 3.0:1 is AA for large text and for
#: non-text UI components (WCAG 1.4.11) — which is the right tier for
#: `fg_dim`, whose only jobs are disabled controls and hint text, both
#: explicitly exempt from 1.4.3, and for the status hues that mostly
#: paint progress-bar chunks.
#: `page` is in the surface list because text really does land on it: a
#: section blurb, a hint under a field, an empty-state line all sit
#: straight on the page between the cards. It is also what caps how far
#: `page` may travel from the panels — see the `page` block above, where
#: `fg_dim` at 3.0:1 is the ceiling the dark value is solved against.
PAGE_SURFACES: Tuple[str, ...] = ("bg", "page", "surface", "surface_alt",
                                  "surface_hi")

CONTRAST_RULES: Tuple[Tuple[str, str, float], ...] = tuple(
    [(fg, surf, 4.5)
     for fg in ("fg", "fg_muted", "accent")
     for surf in PAGE_SURFACES]
    + [("accent", "accent_soft", 4.5),
       ("accent_hi", "accent_soft", 4.5),
       # `bg` is the ink on filled accent/danger surfaces: the selected
       # menu row, a pressed button, DangerButton on hover.
       ("bg", "accent", 4.5),
       ("bg", "accent_lo", 4.5),
       ("bg", "error", 4.5),
       ("button_accent_ink", "button_accent", 4.5),
       ("button_accent_ink", "button_accent_hi", 4.5),
       ("button_accent_ink", "button_accent_lo", 4.5)]
    + [(fg, surf, 3.0)
       for fg in ("fg_dim", "success", "warning", "error")
       for surf in PAGE_SURFACES]
)


def contrast_report(theme: str) -> List[dict]:
    """Measured contrast for every rule in :data:`CONTRAST_RULES`.

    Each entry is ``{"fg", "bg", "fg_color", "bg_color", "ratio",
    "required", "passes"}``. Surfaces are resolved through
    :func:`effective_surface`, so Space is judged on the composited
    scrim rather than on a colour the user never actually sees.
    """
    palette = palette_for(theme)
    out: List[dict] = []
    for fg_role, bg_role, required in CONTRAST_RULES:
        fg_color = palette[fg_role]
        bg_color = effective_surface(theme, bg_role)
        ratio = contrast_ratio(fg_color, bg_color)
        out.append({
            "fg": fg_role, "bg": bg_role,
            "fg_color": fg_color, "bg_color": bg_color,
            "ratio": ratio, "required": required,
            "passes": ratio >= required,
        })
    return out


def contrast_failures(theme: str) -> List[str]:
    """Human-readable description of every rule ``theme`` fails."""
    return _describe(contrast_report(theme))


def _describe(report: List[dict]) -> List[str]:
    return [
        f"{r['fg']} ({r['fg_color']}) on {r['bg']} ({r['bg_color']}): "
        f"{r['ratio']:.2f}:1 < {r['required']:.1f}:1"
        for r in report if not r["passes"]
    ]


# ---------------------------------------------------------------------------
# Does the page separate from the panels?
# ---------------------------------------------------------------------------
# :data:`CONTRAST_RULES` asks "can you read the text on it". This asks the
# other question, the one nobody was asking when the dark page was
# ``#000000``: can you see the *panel*. They are different measurements,
# and the second one has no WCAG tier to borrow, because WCAG has nothing
# to say about two surfaces neither of which is text.
#
# So it is measured twice. The ratio is kept because it is the number the
# rest of this module speaks in; CIE L* is the one that decides, because
# down at the black end the ratio stops discriminating — every pair of
# near-blacks is "about 1.1:1" — while L* stays linear in perceived
# lightness all the way to zero.

#: The panel roles that rest ON the page. `surface_hi` is excluded: it is
#: the hover colour, and converging toward the page is what it is for.
PAGE_PANEL_ROLES: Tuple[str, ...] = ("surface", "surface_alt")

#: One full palette step. The dark palette's own smallest deliberate
#: surface step is `surface` -> `surface_alt` at 3.75 L*, so this is that,
#: rounded down.
PAGE_MIN_LSTAR = 3.5

#: Roughly double the 1.087:1 that `#000000` against `#0d0e10` produced —
#: which is to say, enough that the ratio agrees with L* rather than
#: merely failing to contradict it.
PAGE_MIN_RATIO = 1.15

#: The page opacity a panel is judged at as well as at 1.0. Deliberately
#: not the default (that is :data:`DEFAULT_PANE_OPACITY`, 1.0): a solve
#: that only holds while panels are fully opaque is a solve that breaks
#: for everyone who moved the slider.
PAGE_FADED_OPACITY = 0.6

#: At 60 % a panel composites 0.6*panel + 0.4*page, so a little over half
#: the separation survives. These are that fraction of the bars above.
PAGE_MIN_LSTAR_FADED = 2.0
PAGE_MIN_RATIO_FADED = 1.08


def lightness(color: str) -> float:
    """CIE L* of ``color``, 0 (black) to 100 (white).

    Perceptually uniform, which :func:`relative_luminance` is not and
    :func:`contrast_ratio` is not: a ratio of 1.09:1 means something very
    different between two near-blacks and between two near-whites, and
    the page/panel question lives at both ends.
    """
    y = relative_luminance(color)
    return 903.3 * y if y <= 0.008856 else 116.0 * (y ** (1.0 / 3.0)) - 16.0


def page_separation_report(theme: str) -> List[dict]:
    """How far each resting panel role sits from the page in ``theme``.

    One entry per role in :data:`PAGE_PANEL_ROLES` per opacity in
    ``(1.0, PAGE_FADED_OPACITY)``, carrying ``{"role", "opacity", "page",
    "panel", "ratio", "delta_lstar", "min_ratio", "min_delta_lstar",
    "passes"}``.

    The faded rows composite the panel over the *page* rather than over
    anything else, because the page is what is behind it — that is the
    whole subject.
    """
    page = page_colour(theme)
    palette = palette_for(theme)
    out: List[dict] = []
    for role in PAGE_PANEL_ROLES:
        for opacity, min_ratio, min_dl in (
                (1.0, PAGE_MIN_RATIO, PAGE_MIN_LSTAR),
                (PAGE_FADED_OPACITY, PAGE_MIN_RATIO_FADED,
                 PAGE_MIN_LSTAR_FADED)):
            panel = (palette[role] if opacity >= 1.0
                     else composite(palette[role], opacity, page))
            ratio = contrast_ratio(page, panel)
            delta = abs(lightness(page) - lightness(panel))
            out.append({
                "role": role, "opacity": opacity,
                "page": page, "panel": panel,
                "ratio": ratio, "delta_lstar": delta,
                "min_ratio": min_ratio, "min_delta_lstar": min_dl,
                "passes": ratio >= min_ratio and delta >= min_dl,
            })
    return out


def page_separation_failures(theme: str) -> List[str]:
    """Human-readable description of every separation ``theme`` fails."""
    return [
        f"{theme}: page ({r['page']}) vs {r['role']} ({r['panel']}) at "
        f"{r['opacity']:.0%}: {r['ratio']:.3f}:1 / {r['delta_lstar']:.2f} L* "
        f"< {r['min_ratio']:.2f}:1 / {r['min_delta_lstar']:.2f} L*"
        for r in page_separation_report(theme) if not r["passes"]
    ]


# ---------------------------------------------------------------------------
# Contrast against a *background image*
# ---------------------------------------------------------------------------
# :func:`contrast_report` resolves the `bg` role to the palette's flat
# fallback colour, which is right for the opaque themes and for the
# gradient an image theme falls back to — but it is not what an image
# theme actually shows. There, `bg` is the photograph, and text painted
# with no surface under it (a hero subtitle, a tile caption, a ghost
# button) lands on whatever pixels happen to be there.
#
# So the rules that name `bg` are the ones a wallpaper has to satisfy,
# and they are read straight out of CONTRAST_RULES rather than restated
# — a role added there is automatically enforced against the imagery.

def _bare_image_rules() -> Tuple[Tuple[str, float], ...]:
    return tuple((fg, required)
                 for fg, surface, required in CONTRAST_RULES
                 if surface == "bg")


#: ``(foreground role, minimum ratio)`` for every role that can end up
#: painted directly on the window background.
BARE_IMAGE_RULES: Tuple[Tuple[str, float], ...] = _bare_image_rules()


def max_background_luma(theme: str) -> float:
    """Brightest a background image may be before ``theme`` fails AA.

    Closed form, not a search: for a foreground of relative luminance
    ``Lf`` and a required ratio ``r``, WCAG allows a background up to
    ``(Lf + 0.05) / r - 0.05``. The answer is the tightest of those over
    :data:`BARE_IMAGE_RULES`, and it is what
    :func:`spacr.qt.imagery.solve_dim` expects as its target.
    """
    palette = palette_for(theme)
    return min((relative_luminance(palette[role]) + 0.05) / required - 0.05
               for role, required in BARE_IMAGE_RULES)


def image_contrast_report(theme: str, under: str) -> List[dict]:
    """Measured contrast for every rule, judged over a real image colour.

    ``under`` is a colour sampled from the wallpaper — in practice the
    mean of its brightest text-line-sized region, which is what
    :func:`spacr.qt.imagery.brightest_window` returns. Rules naming the
    ``bg`` surface are judged against it directly, because in an image
    theme nothing is painted between the photograph and the text.
    Everything else is judged against its scrim composited over it.
    """
    palette = palette_for(theme)
    out: List[dict] = []
    for fg_role, bg_role, required in CONTRAST_RULES:
        fg_color = palette[fg_role]
        bg_color = (under if bg_role == "bg"
                    else effective_surface(theme, bg_role, under))
        ratio = contrast_ratio(fg_color, bg_color)
        out.append({
            "fg": fg_role, "bg": bg_role,
            "fg_color": fg_color, "bg_color": bg_color,
            "ratio": ratio, "required": required,
            "passes": ratio >= required,
        })
    return out


def image_contrast_failures(theme: str, under: str) -> List[str]:
    """Every rule ``theme`` fails over a wallpaper colour ``under``."""
    return _describe(image_contrast_report(theme, under))


# ---------------------------------------------------------------------------
# Theme-invariant colour roles
# ---------------------------------------------------------------------------
# The user should be able to recognise interactive controls by colour
# regardless of theme. If the AI/Live toggle went accent-blue in dark and
# a different accent-blue in light, that recognition breaks. These keys
# resolve to the SAME value in both DARK_PALETTE and LIGHT_PALETTE so
# button / toggle styling can rely on them.
#
# `button_accent`     — primary button + toggle "on" colour
# `button_accent_hi`  — hover
# `button_accent_lo`  — pressed
# `button_accent_ink` — text drawn ON those fills
# Chosen to read well on both surface_alt colours (near-black + near-white).
#
# The ink is near-black, not white. White on #4A9EFF measures 2.75:1 —
# well under AA — so the "Run" button had unreadable small text in every
# theme until this was measured. Near-black on the same fill is 6.96:1,
# and still 4.75:1 on the darker pressed shade.
CONSTANT_ROLES = {
    "button_accent":    "#4A9EFF",
    "button_accent_hi": "#66B2FF",
    "button_accent_lo": "#2F80D9",
    "button_accent_ink": "#04101c",
    # NOTE: the loading screen's background and ink are NOT here. They
    # follow the theme's own `bg` and `fg` -- see `palette_for`. Putting
    # them here as one fixed colour is what instruction 78 undid: a splash
    # one shade off the window behind it makes the handover flash, and the
    # point of the loading screen is that the transition is invisible.
}


# Every ingredient the scrim solver needs — the palettes (including
# these constant roles, which `palette_for` folds in), the contrast
# rules, the colour maths and the exposure ceiling — exists by this
# point, so the alphas can be solved. Done at import so that
# `scrim_alpha` stays a dict lookup on the hot path (the QSS asks for it
# once per role per theme change) and so a palette edit that makes a
# theme unsolvable fails loudly here rather than three screens later.
_SOLVED_SCRIMS[False] = _solve_scrims()
SCRIM_ALPHA.update(_SOLVED_SCRIMS[False])


# ---------------------------------------------------------------------------
# Spacing / radius scale — 4/8-based, matches Tk gui_elements.
# ---------------------------------------------------------------------------
SPACING = {
    "xs": 4,
    "sm": 8,
    "md": 12,
    "lg": 16,
    "xl": 24,
    "xxl": 32,
}

RADIUS = {
    "sm": 4,
    "md": 8,
    "lg": 12,
    "pill": 999,
}

FONT_SIZE = {
    "xs":      11,   # inline metadata, table cell suffixes
    "small":   12,   # captions, muted secondary text, form hints
    "body":    13,   # default body text
    "label":   13,   # form field labels
    "header":  15,   # card / section titles
    "subtitle":17,   # dialog headings, secondary display
    "title":   22,   # screen-level headings
    "display": 30,   # startup screen brand title
    "hero":    42,   # empty-state hero numerals
}

def font_px(role_or_px, scale: Optional[float] = None) -> int:
    """Return a font size in px with the user's Zoom preference applied.

    The application stylesheet scales :data:`FONT_SIZE` itself (see
    :func:`stylesheet`), so anything styled by it already tracks Zoom.
    What does *not* is a widget that sets its own sheet — a per-widget
    ``setStyleSheet`` beats the application sheet whatever the selector
    says — or one that paints text with a ``QPainter``. Those surfaces
    hard-coded a pixel number and so stayed 13 px at 150 %: the tab
    strips, the Home aside, the hover tooltip, the Live/AI toggles.

    Route every such number through here instead of writing a literal.

    :param role_or_px: a :data:`FONT_SIZE` key (``"body"``, ``"small"``
        …) or a raw base pixel size.
    :param scale: override the preference — used by :func:`stylesheet`,
        which is generating a sheet for a scale that may not be the
        saved one yet. ``None`` reads the preference.
    :returns: at least 6 px, so a tiny scale cannot collapse text.
    """
    base = FONT_SIZE.get(role_or_px) if isinstance(role_or_px, str) else None
    if base is None:
        try:
            base = float(role_or_px)
        except (TypeError, ValueError):
            base = FONT_SIZE["body"]
    if scale is None:
        # Lazy: `preferences` imports this module, so a module-level
        # import would be circular. Degrade to 1.0 rather than raising —
        # an unscaled label is cosmetic, an exception in a paint is not.
        try:
            from .preferences import get_font_scale
            scale = get_font_scale()
        except Exception:
            scale = 1.0
    return max(6, int(round(float(base) * float(scale))))


# Typography roles — pair size with weight + tracking + line-height
TYPOGRAPHY = {
    "display":   {"size": FONT_SIZE["display"],  "weight": 300, "tracking": "-0.4px", "line_height": "1.15"},
    "title":     {"size": FONT_SIZE["title"],    "weight": 500, "tracking": "-0.2px", "line_height": "1.2"},
    "subtitle":  {"size": FONT_SIZE["subtitle"], "weight": 500, "tracking": "-0.1px", "line_height": "1.25"},
    "header":    {"size": FONT_SIZE["header"],   "weight": 600, "tracking": "0px",    "line_height": "1.3"},
    "body":      {"size": FONT_SIZE["body"],     "weight": 400, "tracking": "0px",    "line_height": "1.45"},
    "small":     {"size": FONT_SIZE["small"],    "weight": 400, "tracking": "0px",    "line_height": "1.4"},
    "caption":   {"size": FONT_SIZE["xs"],       "weight": 500, "tracking": "0.6px",  "line_height": "1.4"},
    "hero":      {"size": FONT_SIZE["hero"],     "weight": 200, "tracking": "-0.5px", "line_height": "1.1"},
}


def apply_qpalette(app: QApplication, theme: str = "dark") -> None:
    """Apply the palette to the QApplication so native controls (menu
    bars, tooltips, dialogs) match the QSS-styled widgets.

    :param app: the running QApplication.
    :param theme: one of :data:`THEMES`; unknown values fall back to dark.
    """
    P = palette_for(theme)
    p = app.palette()
    p.setColor(QPalette.Window,          QColor(P["bg"]))
    p.setColor(QPalette.WindowText,      QColor(P["fg"]))
    p.setColor(QPalette.Base,            QColor(P["surface"]))
    p.setColor(QPalette.AlternateBase,   QColor(P["surface_alt"]))
    p.setColor(QPalette.ToolTipBase,     QColor(P["surface_alt"]))
    p.setColor(QPalette.ToolTipText,     QColor(P["fg"]))
    p.setColor(QPalette.Text,            QColor(P["fg"]))
    p.setColor(QPalette.Button,          QColor(P["surface"]))
    p.setColor(QPalette.ButtonText,      QColor(P["fg"]))
    p.setColor(QPalette.BrightText,      QColor(P["error"]))
    p.setColor(QPalette.Highlight,       QColor(P["accent"]))
    p.setColor(QPalette.HighlightedText, QColor(P["bg"]))
    p.setColor(QPalette.Link,            QColor(P["accent"]))
    p.setColor(QPalette.LinkVisited,     QColor(P["accent_lo"]))
    p.setColor(QPalette.PlaceholderText, QColor(P["fg_dim"]))
    p.setColor(QPalette.Mid,             QColor(P["border"]))
    p.setColor(QPalette.Midlight,        QColor(P["border_soft"]))
    p.setColor(QPalette.Dark,            QColor(P["surface_alt"]))
    p.setColor(QPalette.Shadow,          QColor("#000000"))
    app.setPalette(p)


#: Dynamic property that marks a widget as a *page surface*: something
#: that lays other widgets out but must not paint anything itself.
TRANSPARENT_PROPERTY = "spacrTransparent"

#: Dynamic property that marks a widget as *being* the page surface rather
#: than sitting on one — the exact opposite of :data:`TRANSPARENT_PROPERTY`,
#: and the one opt-out from :func:`clear_container_surfaces`. See
#: :func:`mark_surface`, which is how a screen sets it.
SURFACE_PROPERTY = "spacrSurface"


def mark_surface(*widgets) -> None:
    """Declare that ``widgets`` ARE the page surface, not passengers on one.

    :func:`clear_container_surfaces` tags every ``QAbstractScrollArea`` by
    type, and ``QAbstractItemView`` and ``QPlainTextEdit`` are both one. So
    the shipped ``QTableView/QTreeView {{ background-color: surface_alt }}``
    rule never landed on any view in the application: the attribute selector
    for :data:`TRANSPARENT_PROPERTY` outranks a bare type selector, and the
    view painted nothing at all.

    Where a view sits on a tab pane or inside a card that is right, and it is
    right by accident — the container behind supplies the surface and the view
    shows it through. Where a view sits straight on the page there is nothing
    behind it, and the backdrop arrives untouched: a measured 1.000
    transmission, which over a near-black window colour reads as a black box
    with the text floating on it.

    The sweep cannot be narrowed to exact types. Doing that flips
    **every** view in the application at once, and the ones already sitting on
    a pane would then stack two translucent greys and read about 0.49 — a
    shade no position of the page-opacity slider can produce. Nor can a type
    test make the distinction: Hit List's ``QTreeWidget`` is the page and
    Control Chart's ``QListWidget`` is a passenger, and the pair after them is
    the other way round. Only the screen that built the layout knows which it
    is, so the screen is asked, once, per view.

    Opt-**in** rather than opt-out on purpose. Today every view in the
    application is swept, so opting in changes nothing except where a screen
    says so, and a view nobody has looked at keeps the behaviour it was
    written against.

    Two screens said this before the mechanism existed, by giving the view an
    object name and registering a whole QSS block for it — Model Compare's
    result tables, Model Zoo's listing and provenance box. An ID selector
    outranks the transparent tag the same way. This is that without the block:
    one call, and either the shipped table rule or the
    ``*[spacrSurface="true"]`` rule supplies the fill at the user's page
    opacity. The second is not redundant: nothing in the sheet covers a bare
    ``QListWidget``, which would otherwise fall through to the blanket
    ``QWidget`` rule and paint the WINDOW colour, which is not a surface.

    Safe to call before or after the sweep, and safe to call twice: the
    transparent tag is cleared as well as the surface tag set, and the style
    is re-polished so a visible widget changes immediately.

    One Qt rule has to be paid on the way. A **subclass** of ``QWidget``
    ignores a QSS background entirely unless ``WA_StyledBackground`` is set,
    which is why Power's caveat panel still measured the backdrop untouched
    with a matching rule sitting in the sheet. The attribute is set here for
    those, and deliberately NOT for a ``QAbstractScrollArea``: a view already
    paints its background through its viewport, and a second styled fill on
    top of that is the two-surfaces-stacked fault again. Measured both ways
    rather than reasoned about.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QAbstractScrollArea
    for widget in widgets:
        if widget is None:
            continue
        targets = [widget]
        if isinstance(widget, QAbstractScrollArea):
            targets.append(widget.viewport())
        else:
            widget.setAttribute(Qt.WA_StyledBackground, True)
        for target in targets:
            if target is None:
                continue
            target.setProperty(SURFACE_PROPERTY, True)
            target.setProperty(TRANSPARENT_PROPERTY, False)
            style = target.style()
            if style is not None:
                style.unpolish(target)
                style.polish(target)


def is_surface(widget) -> bool:
    """Whether ``widget`` was declared a page surface by :func:`mark_surface`."""
    if widget is None:
        return False
    return bool(widget.property(SURFACE_PROPERTY))


def clear_container_surfaces(root) -> int:
    """Tag every layout container under ``root`` so it paints nothing.

    Most spaCR screens are plain ``QWidget`` trees. A ``QWidget`` with no QSS
    rule of its own inherits the blanket ``QWidget {{ background-color: bg }}``
    and paints the WINDOW colour — which is not a surface, so no value of the
    page-opacity preference can reach it. That is why a screen could sit as a
    black slab over the animated background no matter what the slider said.

    The rule, and it is a heuristic worth stating plainly:

    * an **anonymous** ``QWidget`` (no ``objectName``) is scaffolding — it
      exists to hold a layout, so it should show whatever is behind it;
    * a **named** widget is something the designer styled on purpose — a
      ``Card``, a ``Section``, a ``ConsoleBox`` — and keeps its fill, at the
      page opacity.

    Scroll areas, their viewports and splitters are always containers whatever
    they are called, so they are tagged by type — **unless** the screen has
    declared one a surface with :func:`mark_surface`. That is the one opt-out,
    and it exists because the type test cannot tell a table that sits ON a
    pane from a table that IS the page. See :func:`mark_surface`.

    :param root: the screen (or any subtree) to sweep.
    :returns: how many widgets were tagged, which is what a test asserts on.
    """
    from PySide6.QtWidgets import (QAbstractScrollArea, QSplitter,
                                   QStackedWidget, QWidget)

    targets = []
    for area in root.findChildren(QAbstractScrollArea):
        if is_surface(area):
            # The screen says this view IS the page. Tagging it — or its
            # viewport, which is the half that actually paints — would put
            # the backdrop straight underneath the text.
            continue
        targets.append(area)
        viewport = area.viewport()
        if viewport is not None:
            targets.append(viewport)
    targets.extend(root.findChildren(QSplitter))
    targets.extend(root.findChildren(QStackedWidget))

    for widget in root.findChildren(QWidget):
        # `type(widget) is QWidget` on purpose, not isinstance: a subclass is a
        # component someone wrote and may well paint deliberately. Only the
        # bare scaffolding qualifies.
        if type(widget) is QWidget and not widget.objectName():
            targets.append(widget)

    targets = [w for w in targets if not is_surface(w)]
    make_transparent(*targets)
    return len(targets)


def make_transparent(*widgets) -> None:
    """Stop ``widgets`` painting a background of their own.

    A backdrop — the theme's wallpaper, or the DNA rain on the
    sequencing screen — is behind the *page*, and in the opaque themes
    every container between it and the eye is an opaque ``bg`` by
    virtue of the blanket ``QWidget`` rule. One container is enough to
    bury it: a screen's header widget, its splitter, a scroll area and
    that scroll area's viewport are each a QWidget, and each one used to
    paint solid black over the animation the screen had just installed.

    Tag the layout containers with this and the backdrop reaches the
    eye; leave the cards, panels and inputs alone and they stay the
    readable surface on top of it. Safe to call on a widget that is
    already visible — the style is re-polished so the change takes
    effect immediately rather than at the next theme switch.

    A ``QScrollArea``'s ``viewport()`` is tagged automatically along
    with it: they are two widgets, the viewport is the one that
    actually paints, and forgetting it is the obvious way to get this
    wrong.
    """
    from PySide6.QtWidgets import QAbstractScrollArea
    for widget in widgets:
        if widget is None:
            continue
        targets = [widget]
        if isinstance(widget, QAbstractScrollArea):
            targets.append(widget.viewport())
        for target in targets:
            if target is None:
                continue
            target.setProperty(TRANSPARENT_PROPERTY, True)
            style = target.style()
            if style is not None:
                style.unpolish(target)
                style.polish(target)


def panel_qcolor(role: str = "surface",
                 theme: Optional[str] = None,
                 opacity: Optional[float] = None) -> QColor:
    """:func:`pane_surface` as a ``QColor``, alpha included.

    The QSS accessor is no use to a widget that draws itself: a
    custom-painted canvas has no stylesheet to put ``rgba(...)`` in, and
    the obvious ``QColor(active_palette()["surface"])`` it reaches for
    instead is **raw hex** — fully opaque, whatever the page-opacity
    preference says. That is how a screen ends up with one flat black
    rectangle in the middle of a page of translucent panels.

    :param role: palette key, normally ``surface``/``surface_alt``.
    :param theme: theme name; ``None`` resolves the effective one.
    :param opacity: 0..1 override; ``None`` reads the preference.
    """
    if theme is None:
        try:
            from .preferences import resolve_effective_theme
            theme = resolve_effective_theme()
        except Exception:
            theme = "dark"
    if opacity is None:
        try:
            from .preferences import get_pane_opacity
            opacity = get_pane_opacity()
        except Exception:
            opacity = None
    base = palette_for(theme)
    colour = QColor(base.get(SCRIM_ROLES.get(role, role),
                             base["surface_alt"]))
    colour.setAlphaF(max(0.0, min(1.0, panel_alpha(theme, role, opacity))))
    return colour


def paint_panel(painter, widget, *, role: str = "surface",
                radius: Optional[int] = None,
                border: bool = True,
                inset: float = 0.0,
                theme: Optional[str] = None,
                opacity: Optional[float] = None) -> None:
    """Draw a rounded, translucent panel filling ``widget``.

    The ``paintEvent`` counterpart of the QSS panel rules, for the
    regions QSS cannot reach. Call it first in a ``paintEvent``, in
    place of ``painter.fillRect(self.rect(), QColor(palette["surface"]))``
    — that call is opaque by construction and is what makes a
    custom-painted canvas read as a bare dark hole punched through the
    page.

    Composites rather than replaces: the widget must be
    ``WA_TranslucentBackground`` or otherwise unfilled for the backdrop
    to reach this, which is what tagging the parents with
    :func:`make_transparent` arranges.

    :param painter: an active ``QPainter`` on ``widget``.
    :param widget: the widget being painted; its ``rect()`` is the panel.
    :param role: palette key for the fill.
    :param radius: corner radius in px; ``None`` uses ``RADIUS["md"]``.
    :param border: draw the theme's soft hairline around the panel.
    :param inset: shrink the panel by this many px on every side, so a
        1 px border lands inside the widget instead of being clipped.
    """
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QPainter, QPen

    if theme is None:
        try:
            from .preferences import resolve_effective_theme
            theme = resolve_effective_theme()
        except Exception:
            theme = "dark"
    corner = float(RADIUS["md"] if radius is None else radius)
    rect = QRectF(widget.rect()).adjusted(inset, inset, -inset, -inset)

    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setPen(Qt.NoPen)
    painter.setBrush(panel_qcolor(role, theme, opacity))
    painter.drawRoundedRect(rect, corner, corner)
    if border:
        base = palette_for(theme)
        pen = QPen(QColor(base["border_soft"]))
        pen.setWidthF(1.0)
        painter.setPen(pen)
        painter.setBrush(QColor(0, 0, 0, 0))
        painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5),
                                corner, corner)
    painter.restore()


def _qss_url(path) -> str:
    """Quote a filesystem path for a QSS ``url(...)``.

    QSS wants forward slashes on every platform — a Windows backslash
    path silently fails to load and you get no background at all.
    """
    text = str(path).replace("\\", "/").replace('"', '\\"')
    return f'url("{text}")'


def _window_block(theme: str, P: dict, background, body_px: int) -> str:
    """The base + top-level-window rules, which image themes rewrite.

    An image theme needs three things the opaque themes do not: a
    background image on the window, ``QWidget`` transparent so the image
    is not covered by every child, and each top-level window type
    explicitly re-opaqued so a stray plain ``QWidget`` window does not
    render as a hole.
    """
    if theme not in IMAGE_THEMES:
        return f"""QWidget {{
    background-color: {P["bg"]};
    color: {P["fg"]};
    font-family: "Open Sans", "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: {body_px}px;
    outline: none;
}}
QMainWindow, QDialog {{
    background-color: {P["bg"]};
}}"""

    if background is not None:
        sky = (f'background-color: {P["bg"]};\n'
               f'    background-image: {_qss_url(background)};\n'
               '    background-position: center center;\n'
               '    background-repeat: no-repeat;')
    elif theme == "glass":
        # A neutral off-axis light field gives translucent surfaces something
        # to optically respond to. It is intentionally not blue: colour belongs
        # to content and selected actions, not to every piece of chrome.
        sky = (
            "background-color: qradialgradient(\n"
            "        cx: 0.18, cy: 0.12, radius: 1.08,\n"
            "        fx: 0.14, fy: 0.08,\n"
            "        stop: 0 #454950, stop: 0.18 #292d33,\n"
            "        stop: 0.52 #16191e, stop: 0.82 #0e1115,\n"
            "        stop: 1 #080a0d);")
    else:
        # No cached image (first run mid-generation, unwritable home,
        # a source build with the masters stripped, tests): a gradient
        # in the theme's own hues. Dimmer than the real thing but every
        # scrim, border and text colour still lands correctly, so the
        # theme degrades to "plain dark" rather than to broken.
        sky = ("background-color: qlineargradient(\n"
               "        x1: 0, y1: 0, x2: 1, y2: 1,\n"
               f'        stop: 0 {P["surface"]}, stop: 0.55 {P["bg"]},\n'
               f'        stop: 1 {P["accent_soft"]});')

    return f"""/* Image theme: the window paints the picture and every child
   is transparent by default, so panels are the only opaque things and
   the imagery shows through the gaps. */
QWidget {{
    background-color: transparent;
    color: {P["fg"]};
    font-family: "Open Sans", "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: {body_px}px;
    outline: none;
}}
QMainWindow, QDialog {{
    {sky}
}}
/* Popups are separate top-level windows: they must stay opaque or a
   compositor-less desktop shows through them. */
QMenu, QToolTip, QMessageBox, QComboBox QAbstractItemView {{
    background-color: {P["surface_alt"]};
}}"""


def _glass_material_layer(base: dict,
                          surface_opacity: Optional[float]) -> str:
    """Final QSS overrides that turn neutral transparency into a material.

    Kept as a last layer so every existing selector still has a conservative
    fallback. Only Glass receives these rules; Dark, Light, Space, and Cell
    remain byte-for-byte on their existing paths.
    """
    surface = glass_material(
        base["surface"], panel_alpha("glass", "surface", surface_opacity))
    alt = glass_material(
        base["surface_alt"],
        panel_alpha("glass", "surface_alt", surface_opacity))
    high = glass_material(
        base["surface_hi"],
        panel_alpha("glass", "surface_hi", surface_opacity))
    tile = glass_material(
        base["surface"], panel_alpha("glass", "tile", surface_opacity))
    rim = css_color("#ffffff", 0.27)
    rim_soft = css_color("#ffffff", 0.16)
    return f"""
/* -----------------------------------------------------------------
 *  Glass material layer
 *
 *  Neutral translucent body + a brighter upper stop suggest lensing;
 *  white rims provide a specular silhouette; larger concentric radii
 *  make controls float. Accent colour remains reserved for actions.
 * ----------------------------------------------------------------- */
QFrame#Card, QFrame#ConsoleBox {{
    background: {alt};
    border: 1px solid {rim};
    border-radius: 14px;
}}
/* The masthead is type on the page, not a card — no glass body, no rim. */
QFrame#Hero {{
    background: transparent;
    border: none;
}}
QFrame#SectionCard {{
    background: {surface};
    border: 1px solid {rim_soft};
    border-radius: 14px;
}}
QFrame#ConsoleTopicBar {{
    background: {high};
    border-top: 1px solid {rim_soft};
    border-bottom: 1px solid {css_color("#000000", 0.24)};
}}
QPlainTextEdit#ConsoleStdoutBlock,
QPlainTextEdit#ConsoleStdoutBlockError {{
    background: transparent;
}}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox,
QPlainTextEdit, QTextEdit {{
    background: {alt};
    border: 1px solid {rim_soft};
    border-radius: 10px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QComboBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{
    background: {high};
    border: 1px solid {base["accent"]};
}}
QPlainTextEdit#ConsoleChatInput, QTextEdit#ConsoleChatInput {{
    background: {alt};
    border: 1px solid {rim};
    border-radius: 14px;
}}
QPushButton {{
    background: {alt};
    border: 1px solid {rim_soft};
    border-radius: 10px;
}}
QPushButton:hover {{
    background: {high};
    border: 1px solid {rim};
}}
QPushButton#Tile, QPushButton#HTile {{
    background: {tile};
    border: 1px solid {rim};
    border-radius: 16px;
}}
/* Module launchers sit directly on the Home pane. A resting glass rim made
   each row read as a ruled table; only an interactive state earns an edge. */
QPushButton#AppTile {{
    background: {tile};
    border: none;
    border-radius: 16px;
}}
QPushButton#AppTile:focus {{
    border: 1px solid {base["accent"]};
}}
QGroupBox {{
    border: 1px solid {rim_soft};
    border-radius: 12px;
}}
"""


# ---------------------------------------------------------------------------
# The widget QSS seam — how a new widget styles itself without editing this
# ---------------------------------------------------------------------------
# `stylesheet()` is one 1100-line f-string, and every widget added to the
# app used to need a block inside it. That makes this file the bottleneck
# for anyone building a widget, and it makes two people building two
# widgets a merge conflict in a literal nobody can review line by line.
#
# A widget registers its own block instead, at import time, from its own
# module:
#
#     from spacr.qt.theme import register_widget_qss, pane_surface
#
#     def _gate_editor_qss(palette, opacity):
#         return f'''
#     QFrame#GateEditor {{
#         background: {pane_surface("surface_alt", palette["theme"], opacity)};
#         border: 1px solid {palette["border_soft"]};
#     }}'''
#
#     register_widget_qss("GateEditor", _gate_editor_qss)
#
# Registered blocks are appended LAST, after the glass layer, so a widget
# can override a general rule that would otherwise win on specificity —
# and so the built-in look is unchanged while nothing is registered.

#: registration name → ``fn(palette, opacity) -> str``. Insertion-ordered:
#: QSS is order-sensitive between rules of equal specificity, so "first
#: registered wins ties" is the contract, and it is the import order of
#: the widgets themselves.
_WIDGET_QSS: Dict[str, object] = {}


#: Every module that registers a widget QSS block **at import time**.
#:
#: This is the exhaustive/static stylesheet inventory.  The public
#: :func:`stylesheet` default imports it so documentation, screenshots and
#: callers that request one complete sheet retain that contract.  Production
#: startup skips the imports: when a screen is opened, its root receives the
#: registered blocks that are absent from the application sheet before that
#: root is shown.  Scoping the late rules to the new screen avoids asking Qt
#: to re-polish every widget already alive merely because one module arrived.
#: Order follows registration order, with later rules winning ties.
WIDGET_QSS_MODULES: Tuple[str, ...] = (
    "spacr.qt.settings_search",
    "spacr.qt.screens.annotate",
    "spacr.qt.screens.app_screen",
    "spacr.qt.screens.settings_model",
    "spacr.qt.shortcuts",
    "spacr.qt.recipes",
    "spacr.qt.comparison_grid",
    "spacr.qt.counting_tool",
    "spacr.qt.curation_tool",
    "spacr.qt.layer_viewer",
    "spacr.qt.ortho_view",
    "spacr.qt.roi_tool",
    # The lightweight Classify footer registers the theme-native box without
    # importing FlowView itself.  Its renderer remains lazy until expansion.
    "spacr.qt.screens.classify",
    "spacr.qt.screens.classifier_evaluation",
    "spacr.qt.screens.control_chart",
    "spacr.qt.screens.data_manager",
    "spacr.qt.screens.experiment_design",
    # The Gate Editor's Filter/Search tab strip. It tried to register its own
    # block from the screen's __init__, against a `theme.register_qss` that
    # has never existed, so the strip has been falling through to the blanket
    # `QWidget { background-color: bg }` since it was written.
    "spacr.qt.screens.gate_editor",
    "spacr.qt.screens.hit_list",
    "spacr.qt.screens.image_scatter",
    # The fold page strip every module screen grows when a fold is opened.
    # A page can be opened long after launch, so its import-time registration
    # is applied to its host screen then; exhaustive sheets include it up
    # front too.
    "spacr.qt.screens.map_barcodes",
    "spacr.qt.screens.methods_export",
    "spacr.qt.screens.model_compare",
    "spacr.qt.screens.model_zoo",
    "spacr.qt.screens.outliers",
    "spacr.qt.screens.pipeline_graph",
    "spacr.qt.screens.power",
    "spacr.qt.screens.profiler",
    "spacr.qt.screens.qc_dashboard",
    # Not a screen: the Measure segmentation-QC banner and the Mask diameter
    # panel. Its block used to be registered only from `prerun.register()`,
    # which runs after the stylesheet has been built and applied, so the
    # banner fell through to the blanket QWidget rule and the verdict text
    # sat on a solid black slab.
    "spacr.qt.prerun",
    "spacr.qt.screens.run_compare",
    "spacr.qt.screens.run_history",
    # Registers at import, through `ensure_field_fade_qss()` at module scope,
    # but nothing imported it while the sheet was being built -- so the block
    # was absent at launch. Found by the strengthened registrar check, not by
    # a bug report.
    # The gate list had no QSS block at all, so its rows fell back to Qt's
    # default black text on the theme's surface -- unreadable on dark.
    "spacr.qt.widgets.gate_editor",
    "spacr.qt.widgets.table_chip",
    "spacr.qt.widgets.gate_console",
    "spacr.qt.widgets.class_editor",
    "spacr.qt.widgets.field_fade",
    "spacr.qt.widgets.formula_editor",
    "spacr.qt.widgets.graph_builder",
    "spacr.qt.widgets.pca_view",
    "spacr.qt.widgets.pivot_builder",
)

_QSS_REGISTRARS_LOADED = False


def load_widget_qss_registrars() -> Tuple[str, ...]:
    """Import :data:`WIDGET_QSS_MODULES` so their blocks are registered.

    Called by the exhaustive/default :func:`stylesheet` path before it
    composes anything.  The production preference path opts out so unopened
    data screens do not import their scientific dependencies merely to
    contribute decoration.

    Idempotent, and the flag is set BEFORE the imports rather than after:
    several of these modules call :func:`stylesheet` while being imported,
    and without that ordering the first one would recurse.

    One module's failure costs that module's rules and nothing else. A
    widget QSS block is decoration; it must never be the thing that stops
    the GUI from starting.

    :returns: the module names that imported cleanly.
    """
    global _QSS_REGISTRARS_LOADED
    if _QSS_REGISTRARS_LOADED:
        return ()
    _QSS_REGISTRARS_LOADED = True

    import importlib

    loaded = []
    for name in WIDGET_QSS_MODULES:
        try:
            importlib.import_module(name)
        except Exception:
            LOG.debug("could not load the widget QSS in %s", name,
                      exc_info=True)
        else:
            loaded.append(name)
    return tuple(loaded)


def register_widget_qss(name: str, fn, *, replace: bool = False):
    """Register a QSS block appended to every generated stylesheet.

    Registration itself never re-applies the ``QApplication`` stylesheet.
    Qt re-polishes every live widget on a global ``setStyleSheet`` call; doing
    that once for every screen imported on demand made later module opens
    progressively slower.  :func:`ensure_widget_qss_applied` installs the
    missing blocks on the new screen's root before it can paint instead.

    :param name: stable identifier, normally the widget's ``objectName``.
        It is what the block is reported and unregistered by; it does not
        appear in the QSS.
    :param fn: ``fn(palette, opacity) -> str``, called once per
        :func:`stylesheet` call.

        ``palette`` is the theme's palette with the three surface roles
        (``surface``, ``surface_alt``, ``surface_hi``) already rendered
        through the user's page opacity — the same values the built-in
        rules interpolate, so a registered block matches the app without
        doing the alpha maths. Two reserved non-colour keys ride along:
        ``theme`` (the theme name, which :func:`pane_surface`,
        :func:`pane_alpha` and :func:`palette_for` all want) and
        ``font_scale``.

        ``opacity`` is the user's page-opacity preference, or ``None``
        for "use the theme's designed scrim". Pass it straight through
        to :func:`block_surface` / :func:`pane_alpha` / :func:`panel_alpha`
        rather than interpreting it: ``None`` is not 1.0, and the
        legibility floor is theirs to apply.

        :func:`block_surface` and **not** :func:`pane_surface`, which is
        the near-identical accessor for inline and paint-time callers. Its
        ``None`` means "nobody told me" and reads the live preference, so a
        block using it turns ``stylesheet(theme)`` into a function of a
        QSettings value rather than of its arguments.
    :param replace: allow re-registering ``name``. Off by default so two
        widgets cannot quietly claim one name.
    :raises ValueError: on a duplicate name without ``replace``.
    :raises TypeError: if ``fn`` is not callable.
    """
    name = str(name)
    if not name:
        raise ValueError("a widget QSS block needs a name")
    if not callable(fn):
        raise TypeError(f"widget QSS {name!r} is not callable: {fn!r}")
    if name in _WIDGET_QSS and not replace:
        raise ValueError(
            f"widget QSS {name!r} is already registered; pass replace=True "
            "if that is really what you mean")
    _WIDGET_QSS[name] = fn
    return fn


def unregister_widget_qss(name: str) -> bool:
    """Drop a registered block. ``True`` if there was one."""
    return _WIDGET_QSS.pop(str(name), None) is not None


def widget_qss_names() -> Tuple[str, ...]:
    """Every registered block name, in registration order."""
    return tuple(_WIDGET_QSS)


#: The marker :func:`registered_widget_qss` writes above every block. It is
#: what tells a live stylesheet apart from one generated before a screen
#: module was imported.
_WIDGET_QSS_MARKER = "/* --- registered widget QSS: {name} --- */"

# A late block lives on the root of the screen that needed it.  The outer
# markers make that suffix replaceable as more modules are imported and
# removable before the next whole-application preference rebuild.  The
# attribute holds the exact suffix, including its leading newline, so a
# screen's own stylesheet is restored byte-for-byte rather than reparsed.
_LOCAL_WIDGET_QSS_START = "/* --- local registered widget QSS: start --- */"
_LOCAL_WIDGET_QSS_END = "/* --- local registered widget QSS: end --- */"
_LOCAL_WIDGET_QSS_ATTRIBUTE = "_spacr_local_widget_qss_suffix"
_WIDGET_QSS_CONTEXT_ATTRIBUTE = "_spacr_widget_qss_context"


def set_widget_qss_context(app, theme: str, font_scale: float,
                           surface_opacity: Optional[float]) -> None:
    """Record the exact live preference inputs for late screen blocks."""
    if app is not None:
        setattr(app, _WIDGET_QSS_CONTEXT_ATTRIBUTE,
                (str(theme), float(font_scale), surface_opacity))


def _live_widget_qss_context(app) -> Tuple[str, float, Optional[float]]:
    """Return the preference inputs used by the live application sheet."""
    context = getattr(app, _WIDGET_QSS_CONTEXT_ATTRIBUTE, None)
    if (isinstance(context, tuple) and len(context) == 3):
        return context
    try:
        from .preferences import (
            get_font_scale,
            get_pane_opacity,
            resolve_effective_theme,
        )
        return (resolve_effective_theme(), get_font_scale(),
                get_pane_opacity())
    except Exception:
        return "dark", 1.0, None


def _widget_qss_palette(theme: str, font_scale: float,
                        surface_opacity: Optional[float]) -> dict:
    """Build the callback palette shared by global and screen-local QSS."""
    base = palette_for(theme)
    palette = dict(base)
    for role in ("surface", "surface_alt", "surface_hi"):
        palette[role] = css_color(
            base[role], panel_alpha(theme, role, surface_opacity))
    palette["theme"] = theme
    palette["font_scale"] = font_scale
    return palette


def clear_widget_qss_overlays(app=None) -> int:
    """Remove screen-local late-QSS suffixes before a global theme rebuild.

    The rebuilt application sheet contains every block registered so far,
    with the new theme, opacity and font scale.  Leaving an older local copy
    in place would give it precedence and strand the screen on the previous
    preference values.

    :returns: number of screen roots whose owned suffix was removed.
    """
    app = app or QApplication.instance()
    if app is None:
        return 0
    cleared = 0
    for widget in list(app.allWidgets()):
        suffix = getattr(widget, _LOCAL_WIDGET_QSS_ATTRIBUTE, "")
        if not suffix:
            continue
        try:
            current = widget.styleSheet()
            setattr(widget, _LOCAL_WIDGET_QSS_ATTRIBUTE, "")
            if current.endswith(suffix):
                widget.setStyleSheet(current[:-len(suffix)])
            cleared += 1
        except RuntimeError:
            # The C++ widget was deleted while Qt was draining its queue.
            pass
    return cleared


def preserve_widget_qss_overlay(root, stylesheet: str) -> str:
    """Return ``stylesheet`` with ``root``'s owned late-QSS suffix intact.

    A screen may legitimately replace its own base stylesheet after its late
    widget blocks were installed.  Folding the suffix into that existing
    assignment avoids a second ``setStyleSheet`` call (and its palette-change
    cascade) while keeping the blocks available for the next paint.
    """
    return str(stylesheet) + getattr(root, _LOCAL_WIDGET_QSS_ATTRIBUTE, "")


def ensure_widget_qss_applied(*names: str, root=None) -> bool:
    """Install late registered blocks on ``root`` without restyling the app.

    The production application stylesheet is composed before unopened screen
    modules are imported.

    Replacing that whole sheet for each import closes
    the first-paint race, but it also makes Qt parse the sheet and re-polish
    every widget accumulated in every cached screen.

    A screen root is a QSS
    scope: rules installed there reach that screen and its descendants, and
    applying them before the root is shown preserves the same first-paint
    contract without touching Home or any previously opened module.

    The suffix contains every registered block absent from the application
    sheet, in registry order, rather than only ``names``.  This keeps blocks
    imported by a screen's dependencies together and means a later call can
    replace one complete suffix instead of stacking fragments with different
    preference values.  ``names`` remains the caller's documentation of the
    blocks it requires; omitting it is the MainWindow screen-host path.

    It is a no-op with no ``root``, no ``QApplication``, or no application
    stylesheet.  A caller that never opted into spaCR styling is not opted in
    merely by constructing one of its widgets.

    :returns: ``True`` only when ``root.setStyleSheet`` was called.
    """
    if root is None:
        return False
    app = QApplication.instance()
    if app is None:
        return False
    app_sheet = app.styleSheet()
    if not app_sheet:
        return False
    wanted = tuple(
        name for name in _WIDGET_QSS
        if _WIDGET_QSS_MARKER.format(name=name) not in app_sheet
    )
    theme, font_scale, opacity = _live_widget_qss_context(app)
    palette = _widget_qss_palette(theme, font_scale, opacity)
    fragment = registered_widget_qss(palette, opacity, names=wanted)
    if fragment:
        body_px = max(6, int(round(FONT_SIZE["body"] * font_scale)))
        fragment += close_mark_rules(theme, body_px)
    suffix = (
        f"\n{_LOCAL_WIDGET_QSS_START}\n{fragment}"
        f"{_LOCAL_WIDGET_QSS_END}"
        if fragment else ""
    )
    try:
        current = root.styleSheet()
        previous = getattr(root, _LOCAL_WIDGET_QSS_ATTRIBUTE, "")
        base = current[:-len(previous)] if (
            previous and current.endswith(previous)) else current
        desired = base + suffix
        setattr(root, _LOCAL_WIDGET_QSS_ATTRIBUTE, suffix)
        if desired == current:
            return False
        root.setStyleSheet(desired)
    except (AttributeError, RuntimeError):
        return False
    return True


def page_tabs_qss(object_name: str, palette: dict, opacity=None) -> str:
    """Home's tab treatment, for a tab strip that IS the page.

    The shipped ``QTabBar``/``QTabWidget::pane`` rules paint
    ``P["surface"]`` and ``P["surface_alt"]`` — **raw hex**, so a tab
    strip that is the main content of a screen (Classifier Evaluation,
    Run History) sat there as a flat opaque slab while the cards beside
    it thinned with the slider.

    This is the same shape Home uses, at the page opacity: rounded top
    corners, a dark-grey tab by default, the accent blue under the
    pointer, and a rounded translucent pane below it.

    Register it per screen rather than making it a blanket rule — a tab
    strip *inside* a card is on a surface already and must keep the
    shipped look, or it double-fills.

    :param object_name: ``objectName`` of the ``QTabWidget``.
    :param palette: the palette handed to a registered block, including
        the reserved ``theme`` key.
    :param opacity: the page-opacity preference, passed straight
        through — ``None`` here means the theme's designed scrim, which is
        why this reads :func:`block_surface` and not :func:`pane_surface`.
    """
    theme = palette.get("theme", "dark")
    scale = palette.get("font_scale")
    pane = block_surface("surface_alt", theme, opacity)
    tab = block_surface("surface", theme, opacity)
    return f"""
QTabWidget#{object_name}::pane {{
    background: {pane};
    border: 1px solid {palette["border_soft"]};
    border-radius: {RADIUS["md"]}px;
    top: -1px;
}}
/* The bar itself, not the tabs on it. Qt builds `qt_tabwidget_tabbar`
   and with no rule of its own it takes the blanket window fill. */
QTabWidget#{object_name} > QTabBar {{
    background: transparent;
}}
QTabWidget#{object_name} > QTabBar::tab {{
    background: {tab};
    color: {palette["fg_muted"]};
    border: 1px solid {palette["border_soft"]};
    border-bottom: none;
    border-top-left-radius: {RADIUS["md"]}px;
    border-top-right-radius: {RADIUS["md"]}px;
    padding: 7px 14px;
    margin-right: 2px;
    font-size: {font_px("body", scale)}px;
}}
QTabWidget#{object_name} > QTabBar::tab:hover {{
    background: {palette["accent"]};
    color: {palette["bg"]};
}}
QTabWidget#{object_name} > QTabBar::tab:selected {{
    background: {pane};
    color: {palette["accent"]};
    border-bottom-color: {pane};
}}
/* The container below the tabs. Qt builds the page stack itself, and a
   read-only detail view is a *display*, not a field — left as one it
   takes the shipped input fill and paints an opaque rectangle over the
   pane that was just made translucent, which is the bare dark area
   again one layer down. The pane is the panel; everything sitting on it
   shows it through. */
QTabWidget#{object_name} > QStackedWidget {{
    background: transparent;
}}
QTabWidget#{object_name} QPlainTextEdit[readOnly="true"],
QTabWidget#{object_name} QTextEdit[readOnly="true"] {{
    background: transparent;
    border: none;
}}
"""


def registered_widget_qss(palette: dict,
                          opacity: Optional[float] = None, *,
                          names=None) -> str:
    """Render every registered block into one QSS fragment.

    Empty (not even a newline) while nothing is registered, which is what
    keeps the shipped stylesheet byte-identical to the one that had no
    seam at all.

    A block that raises, or returns something that is not a string, is
    dropped with a logged traceback rather than taking the stylesheet
    down: an unstyled widget is a cosmetic fault, and an exception here
    would leave the whole application unstyled — black text on a black
    window — because one contributed widget had a typo.
    """
    wanted = None if names is None else {str(name) for name in names}
    parts = []
    for name, fn in list(_WIDGET_QSS.items()):
        if wanted is not None and name not in wanted:
            continue
        try:
            block = fn(palette, opacity)
        except Exception:
            LOG.exception("Widget QSS %s failed to render", name)
            continue
        if not isinstance(block, str):
            LOG.error("Widget QSS %s returned %s, expected str",
                      name, type(block).__name__)
            continue
        if block.strip():
            parts.append(f"\n/* --- registered widget QSS: {name} --- */\n"
                         f"{block.strip()}\n")
    return "".join(parts)


def stylesheet(theme: str = "dark", font_scale: float = 1.0,
               background: Optional[str] = None,
               surface_opacity: Optional[float] = None, *,
               load_widget_registrars: bool = True) -> str:
    """Return the QSS string that styles every custom widget in the app.

    Blocks registered with :func:`register_widget_qss` are appended after
    everything below, so a widget's own rules win a specificity tie
    against the general ones.

    :param theme: one of :data:`THEMES`; unknown values fall back to dark.
    :param font_scale: multiplier applied to every font size in
        :data:`FONT_SIZE`. 1.0 = 100 %.
    :param background: path to a background image. Only the themes in
        :data:`IMAGE_THEMES` use it; ``None`` (the default, and what a
        first run mid-generation gets) falls back to a flat gradient.
    :param surface_opacity: optional user-requested alpha for all shared
        module surfaces. ``None`` uses the theme's designed scrims.
    :param load_widget_registrars: import every module that contributes a
        widget block before composing.  This remains the public default for
        exhaustive callers and tests.  Application startup passes ``False``
        so an unopened data screen cannot pull the scientific stack into the
        first frame; ``MainWindow`` scopes late blocks to a new screen before
        inserting that screen into the visible stack.
    """
    base = palette_for(theme)
    # Exhaustive callers get every known block in one static sheet. The live
    # application opts out; MainWindow installs blocks registered by a lazy
    # module on that module's screen root before it is shown.
    if load_widget_registrars:
        load_widget_qss_registrars()

    S = SPACING
    R = RADIUS
    # Surface roles are re-rendered through the theme's scrim alpha.
    # For dark and light every alpha is 1.0 and this is a no-op that
    # emits the same hex it always did; for Space each one becomes an
    # ``rgba()`` so the background image reads through the panel.
    P = _widget_qss_palette(theme, font_scale, surface_opacity)
    # Opaque variants for the places translucency would be wrong.
    ELEVATED = css_color(
        base["surface_alt"], panel_alpha(theme, "elevated", surface_opacity))
    #: The menu bar, and everything drawn onto it: its items, and the
    #: window chrome in its corner. Slightly translucent so the bar does
    #: not read as a separate slab, but nowhere near transparent -- see
    #: the QMenuBar rules for what fully-transparent cost on macOS.
    BAR_BG = css_color(base["surface"], 0.94)
    over_image = theme in IMAGE_THEMES
    # Tiles take page opacity on every theme. Over an image they always did;
    # on the flat themes they were `transparent`, which looked identical to
    # the window and meant the tile itself could not be dialled — the user
    # asked for the tiles AND the boxes they sit in to follow the setting.
    # `transparent` is still the right answer at 100%, because a fully opaque
    # tile over an identical window colour is what the flat themes looked like
    # before, so the alpha is applied to the surface colour rather than
    # replacing it with a hard block.
    TILE_BG = css_color(base["surface"],
                        panel_alpha(theme, "tile", surface_opacity))
    # Scrollbar troughs paint over the window; over a photograph they must not
    # be an opaque black block. Group-box titles are transparent below.
    #
    # `page`, not `bg`, on the flat themes. A trough runs the full height of
    # the settings column and it took the WINDOW colour, so on the dark theme
    # it was a black stripe down the page — the last 168 pure-black samples in
    # that column after the page itself had a colour, and the only ones left.
    # It is a groove IN the page, so the page is what it should be.
    TROUGH = "transparent" if over_image else page_colour(theme)
    # The console honours page opacity on EVERY theme, not just the image
    # ones. `#0a0b0d` was a hard-coded near-black, so on dark and light the
    # console stayed a solid slab no matter where the slider was — one of the
    # containers the preference visibly failed to reach.
    CONSOLE_BG = (P["surface_alt"] if over_image else css_color(
        "#0a0b0d", panel_alpha(theme, "surface_alt", surface_opacity)))
    # The dock takes page opacity on the FLAT themes and stays opaque over an
    # image. Two user instructions meet here and both are kept:
    #
    #   #16j — "the dock to the left should never have a transparent
    #   background, either dark gray or white". A navigation column is chrome:
    #   it is what you look at when you have lost your place. It used to paint
    #   `surface`, which the image themes re-render through `scrim_alpha`, so
    #   on Space the app list was a ghost with a galaxy behind every row.
    #
    #   Later — page opacity should reach "the dock" as well.
    #
    # On dark and light there is no wallpaper to show through, only the
    # ambient animation, so thinning the dock does exactly what was asked and
    # nothing #16j was protecting against. Over Space or Cell the picture is
    # behind it and the old complaint applies verbatim — and the legibility
    # floor does NOT save it there (Cell floors at 0.047), so the split is
    # explicit rather than left to the solver.
    DOCK_BG = (dock_colour(theme) if over_image else css_color(
        dock_colour(theme),
        panel_alpha(theme, "surface_alt", surface_opacity)))
    #: What the dock actually paints. See the Sidebar block below: 369 takes
    #: the container off, #16j says it may never be transparent over a
    #: picture, and `over_image` is the seam that was already carrying that
    #: distinction.
    DOCK_FILL = DOCK_BG if over_image else "transparent"
    # The theme ink used by outlined horizontal tiles, and the three maturity
    # hues a module tile switches to on hover. Resting AppTiles are rimless;
    # the hover fill is the stage colour at a low alpha so the tile lights UP
    # rather than being replaced by a block of magenta.
    RIM = rim_colour(theme)
    SELECTION_INK = selection_ink(theme)
    STAGE_RULES = "\n".join(
        f"""QPushButton#AppTile[stage="{stage}"]:hover {{
    background-color: {css_color(hue, 0.22)};
    border: 1px solid {hue};
}}
QPushButton#AppTile[stage="{stage}"]:pressed {{
    background-color: {css_color(hue, 0.40)};
    border: 1px solid {hue};
}}"""
        for stage, hue in STAGE_HOVER.items())
    # THE FOLDED MODULES LIGHT UP LIKE THE TILES THEY REPLACED.
    #
    # A module folded into a host screen is a button on that host's
    # masthead rather than a tile on Home, and the maturity it promises
    # did not change when it moved. Same hue, same two states, read from
    # the same STAGE_HOVER table -- so signing a module off recolours its
    # button and its tile together, and neither can drift from the other.
    FOLD_STAGE_RULES = "\n".join(
        f"""QPushButton#FoldButton[stage="{stage}"]:hover {{
    background-color: {css_color(hue, 0.22)};
    border: 1px solid {hue};
}}
QPushButton#FoldButton[stage="{stage}"]:pressed {{
    background-color: {css_color(hue, 0.40)};
    border: 1px solid {hue};
}}"""
        for stage, hue in STAGE_HOVER.items())
    SECTION_STAGE_RULES = "\n".join(
        f"""QFrame#SectionCard[maturity="{stage}"] {{
    border: 1px solid {css_color(hue, 0.72)};
    border-left: 4px solid {hue};
}}
QToolButton#SectionHeader[maturity="{stage}"]:hover,
QToolButton#SectionHeader[maturity="{stage}"]:checked {{
    background-color: {css_color(hue, 0.14)};
}}"""
        for stage, hue in STAGE_HOVER.items())
    # Scaled with the font, like every other Python-set size: a 150 %
    # font makes the name taller, and a tile that did not grow with it
    # would clip the thing it exists to show.
    TILE_MIN_H = max(1, int(round(TILE_H * font_scale)))
    TILE_MIN_W_PX = max(1, int(round(TILE_W * font_scale)))
    # Scaled font sizes so the "Font scale" preference actually
    # resizes the whole app, not just the base body text.
    F = {k: max(6, int(round(v * font_scale)))
         for k, v in FONT_SIZE.items()}
    GLASS_LAYER = (
        _glass_material_layer(base, surface_opacity)
        if theme == "glass" else ""
    )
    # Contributed blocks, rendered against the same surfaces the rules
    # below use. `theme` and `font_scale` ride along because a widget
    # that wants `pane_surface`/`pane_alpha` needs the theme name, and
    # this callback is the only place it would otherwise have to guess it
    # (guessing means reading the preference again, which is wrong while
    # a stylesheet is being generated for a theme that is not yet live).
    WIDGET_QSS = registered_widget_qss(P, surface_opacity)
    # LAST, after the contributed blocks. The close mark is the one glyph
    # the whole application shares, so a widget block that grows a rule for
    # its own X loses the tie instead of quietly winning it. See
    # `close_mark_rules`.
    CLOSE_MARK_RULES = close_mark_rules(theme, F["body"])
    return f"""
/* -----------------------------------------------------------------
 *  Base
 * ----------------------------------------------------------------- */
{_window_block(theme, base, background, F["body"])}
/* Page surfaces — see `make_transparent`. A widget carrying this
 * property paints nothing at all, so whatever sits behind the page
 * shows through it: the wallpaper in an image theme, the DNA rain on
 * the sequencing screen. Cards, panels and inputs are NOT tagged, so
 * they keep their surface and stay the readable thing on top.
 *
 * An attribute selector outranks the bare `QWidget` type selector in
 * QSS specificity, so this wins in every theme whatever the rule order
 * — which matters for dark and light, where `QWidget` is an opaque
 * `bg` and used to bury the rain under the first container it met. */
*[{TRANSPARENT_PROPERTY}="true"] {{
    background: transparent;
}}
/* The opposite declaration — see `mark_surface`. A view that IS the page,
 * rather than a passenger on a pane, needs a surface of its own, and most
 * of them would not get one even if the sweep left them alone: the shipped
 * rule covers QTableView/QTreeView but nothing covers a bare QListWidget,
 * which then falls through to the blanket `QWidget` fill and paints the
 * WINDOW colour, which is not a surface.
 *
 * Background only. Borders, radii and item padding stay with whatever type
 * rule the widget already matched, so a marked table still looks like a
 * table; QSS cascades per property, and only this one is being decided
 * here. `P["surface_alt"]` already carries the user's page opacity. */
*[{SURFACE_PROPERTY}="true"] {{
    background-color: {P["surface_alt"]};
}}
/* Every QLabel is transparent by default so it inherits the bg of
 * whatever container it lives in (surface, surface_alt, hero card,
 * etc). Individual labels can override with their own object name. */
QLabel {{
    background: transparent;
}}
/* Settings labels are wrapped with a layout-only QWidget that right-aligns
 * the text against its field. A bare QWidget inherits the window canvas
 * colour; without this rule that wrapper paints a black rectangle on the
 * section's dark-gray surface even though the QLabel itself is transparent.
 * These wrappers are structural and must show their actual container
 * through. */
QWidget#SettingLabelWithInfo,
QWidget#SettingControlWithInfo,
QWidget#SettingLinkStack {{
    background: transparent;
}}
/* Grey out text on disabled widgets (e.g. a live-preview compartment panel
   that isn't the chosen object) so the label reads as inactive too, not just
   the field. */
QLabel:disabled, QCheckBox:disabled, QGroupBox:disabled,
QGroupBox::title:disabled, QRadioButton:disabled {{
    color: {P["fg_dim"]};
}}

/* -----------------------------------------------------------------
 *  Menu bar + menus
 * ----------------------------------------------------------------- */
QMenuBar {{
    /* ONE FLAT, MOSTLY-OPAQUE COLOUR. This bar is the frameless window's
       title bar, so it sits over the animated backdrop -- and read
       through a fully translucent bar that backdrop is a moving gradient
       behind the only two words on it. Reported from macOS: "the bar is
       transparent and has a gradient so it is hard to see the spaCR and
       Help". A little translucency keeps it from looking pasted on; the
       rest is what makes the labels legible over anything. */
    background-color: {BAR_BG};
    color: {P["fg_muted"]};
    padding: {S["xs"]}px {S["sm"]}px;
    border-bottom: 1px solid {P["border_soft"]};
    font-size: {F["small"]}px;
}}
QMenuBar::item {{
    /* THE BAR'S OWN COLOUR, NEVER `transparent`. `transparent` means
       "paint nothing", and what is behind this bar is the WINDOW, whose
       palette Window role is the splash colour -- pure black. On Linux
       the bar's own fill covers that and nothing shows; on macOS the
       hover repaint clears to the window first, and the black came
       through as a box behind each label. Painting the bar's colour
       here is indistinguishable from transparent wherever transparent
       worked, and correct where it did not. */
    background: {BAR_BG};
    padding: {S["xs"]}px {S["sm"]}px;
    border-radius: {R["sm"]}px;
}}
QMenuBar::item:selected, QMenuBar::item:pressed {{
    /* THE WORD LIGHTS, not a plate behind it: the same accent the dock's
       open section header takes, so pointing at spaCR or Help reads the
       same way as pointing at a category. The background repeats the
       bar's colour rather than being `transparent` for the reason
       above -- this is the exact state the black box appeared in. */
    background: {BAR_BG};
    color: {P["accent"]};
}}
QMenu {{
    background-color: {ELEVATED};
    color: {P["fg"]};
    border: 1px solid {P["border"]};
    border-radius: {R["md"]}px;
    padding: {S["xs"]}px;
}}
QMenu::item {{
    padding: {S["xs"]}px {S["md"]}px;
    border-radius: {R["sm"]}px;
    background: transparent;
}}
QMenu::item:selected {{
    background: {P["accent"]};
    color: {P["bg"]};
}}
QMenu::separator {{
    height: 1px;
    background: {P["border"]};
    margin: {S["xs"]}px {S["sm"]}px;
}}

/* -----------------------------------------------------------------
 *  Sidebar (main window navigation)
 * ----------------------------------------------------------------- */
/* THE TRAY GOES ON THE FLAT THEMES AND STAYS OVER A PICTURE, AND THAT SPLIT
   IS TWO MAINTAINER REQUESTS THAT DISAGREE.

   2026-09-02, instruction 369: "the background dark gray container can be
   removed, the hover highlight should stay."
   Earlier, #16j: "the dock to the left should never have a transparent
   background, either dark gray or white" -- filed because on Space the app
   list was a ghost with a galaxy behind every row.

   Both are real. Taken literally the second forbids the first. The split
   already in this file resolves it, and 369 is applied along the SAME seam
   rather than a new one: on `dark` and `light` there is no wallpaper behind
   the dock -- only the ambient animation -- so the container comes off and
   that is exactly what was asked for. Over `space` and `cell` there IS a
   picture, #16j's complaint applies verbatim, and the legibility floor does
   not rescue it (Cell floors at 0.047), so the dock stays opaque there.

   THE EDGE IS KEPT EITHER WAY: `#Sidebar` still draws its right border, so
   the page still ends at a line rather than bleeding into the dock.

   THE HOVER HIGHLIGHT SURVIVES BECAUSE IT WAS NEVER THE TRAY. It is drawn
   by the row itself, in `_DockRow._paint_plate`, and translucently on
   purpose: removing the plane behind it changes what it sits ON, not
   whether it is drawn. (It is NOT the `QPushButton#SidebarItem:hover`
   rule below, which this comment used to claim and which reaches no dock
   row -- see the note on that rule.)

   IF THE MAINTAINER WANTS IT GONE OVER THE PICTURES TOO, this is one line:
   drop the `over_image` arm of `DOCK_FILL` below. */
#EdgeDrawer, #Sidebar, #SidebarScroll, #SidebarInner {{
    background-color: {DOCK_FILL};
}}
/* NO RIGHT BORDER. The dock is a rounded slab painted by `Sidebar.
   paintEvent` (2026-09-03), and a full-height 1 px rule down its right edge
   cuts straight across the two corners it just rounded. The slab draws its
   own hairline edge, all the way round, which is what separates the dock
   from the page now. */
#Sidebar {{
    border: none;
}}
#SidebarTitle {{
    color: {P["accent"]};
    font-family: "Open Sans", "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: 24px;
    font-weight: 300;                 /* Light */
    letter-spacing: -0.5px;
    padding: {S["lg"]}px {S["md"]}px {S["md"]}px;
    background: {DOCK_FILL};
}}
#SidebarSection {{
    color: {P["fg_dim"]};
    font-size: {F["xs"]}px;
    font-weight: 600;
    padding: {S["md"]}px {S["md"]}px {S["xs"]}px;
    text-transform: uppercase;
    letter-spacing: 1px;
    background: {DOCK_FILL};
}}
/* AN OPEN SECTION IS BLUE, and so is one under the pointer. The header is
   the control that opens it, and a control that looks identical whether it
   is on or off is a control nobody learns. */
#SidebarSection[open="true"], #SidebarSection[hovered="true"] {{
    color: {P["accent"]};
}}
/* THE ROW'S PLATE IS NOT DRAWN FROM HERE. Look in
   `_DockRow._paint_plate` (spacr/qt/app.py) -- the translucent rounded box
   behind each icon, its hover step and the accent bar on the open module
   are all painted there, from the live palette.

   That is not a style preference, it is what was measured on 2026-09-03: a
   plain `QPushButton` carrying this object name renders the background
   below, and a `_DockRow` renders the dock's own fill instead, because a
   `paintEvent` that goes straight to `drawControl(CE_PushButton)` skips the
   pass in which QStyleSheetStyle fills a widget's background. So no
   `background` written here has reached a dock row since 348 gave the rows
   their own painting, INCLUDING the `:hover` arm below -- which is why the
   dock had no per-row box at all and every icon sat flat on the dark dock.

   The rules are kept, at their pre-348 values, for the `color` they set --
   which QStyleSheetStyle does apply, through the palette it hands the
   row -- and so that any other widget given this object name still looks
   like a dock row. Edit the plate's colours in `_DockRow`, not here. */
QPushButton#SidebarItem {{
    text-align: left;
    background: transparent;
    color: {P["fg_muted"]};
    padding: {S["sm"]}px {S["md"]}px;
    border: none;
    border-left: 3px solid transparent;
    border-radius: 0px;
    font-size: {F["body"]}px;
}}
QPushButton#SidebarItem:hover {{
    background: {P["surface_hi"]};
    color: {P["fg"]};
}}
QPushButton#SidebarItem:checked, QPushButton#SidebarItem[selected="true"] {{
    background: {P["surface_hi"]};
    color: {P["accent"]};
    border-left: 3px solid {P["accent"]};
}}

/* -----------------------------------------------------------------
 *  Cards / grouped sections
 * ----------------------------------------------------------------- */
QFrame#Card {{
    /* A clear dark-gray rounded box sitting on the black app background —
       same surface as the console box so System/Figures/console read as one
       consistent family of boxes. */
    background-color: {P["surface_alt"]};
    border: 1px solid {P["border_soft"]};
    border-radius: {R["md"]}px;
}}
/* The masthead — logo, spaCR wordmark, end-to-end subtitle. No fill and no
   rim: it is type on the page, not a card. It used to paint a diagonal
   gradient with a border, which read as a black box drawn around the brand. */
QFrame#Hero {{
    background: transparent;
    border: none;
}}
QPushButton#Tile:hover {{
    border: 1px solid {P["accent"]};
    background-color: {P["surface_hi"]};
}}
QLabel#CardTitle {{
    color: {P["fg"]};
    font-size: {F["header"]}px;
    font-weight: 600;
    padding: 0px;
    background: transparent;
}}
QLabel#CardSubtitle {{
    color: {P["fg_muted"]};
    font-size: {F["small"]}px;
    background: transparent;
}}
QFrame#Divider {{
    background: {P["border"]};
    max-height: 1px;
    min-height: 1px;
    border: none;
}}

/* -----------------------------------------------------------------
 *  Startup tiles
 * ----------------------------------------------------------------- */
QPushButton#Tile {{
    background-color: {P["surface"]};
    color: {P["fg"]};
    border: 1px solid {P["border_soft"]};
    border-radius: {R["lg"]}px;
    padding: {S["md"]}px;
    font-size: {F["body"]}px;
    text-align: center;
    min-width: 96px;
    min-height: 96px;
}}
QPushButton#Tile:hover {{
    border: 1px solid {P["accent"]};
    background-color: {P["surface_alt"]};
    color: {P["accent"]};
}}
QPushButton#Tile:pressed {{
    background-color: {P["accent_lo"]};
    color: {P["bg"]};
}}
QLabel#TileCaption {{
    color: {P["fg"]};
    font-size: {F["body"]}px;
    font-weight: 500;
    background: transparent;
    padding-top: 4px;
}}

/* -----------------------------------------------------------------
 *  Horizontal tiles (HTile) — icons-left cards on the home screen
 * ----------------------------------------------------------------- */
/* Every tile carries a hairline rim in the theme's ink — white on the
   dark themes, near-black on the light one. It is not decoration: with
   the descriptions gone the tiles are icon + name on the pane's own
   colour, and the rim is the only thing that says where one button ends
   and the next begins. `border: 1px solid transparent` (what this used
   to be) drew nothing at all until you hovered. */
QPushButton#HTile {{
    background-color: {TILE_BG};
    color: {P["fg"]};
    border: 1px solid {css_color(RIM, 0.35)};
    border-radius: {R["lg"]}px;
    padding: 12px 14px 12px 20px;
    text-align: left;
    font-family: "Open Sans", "Segoe UI", "Helvetica Neue", sans-serif;
}}
QPushButton#HTile:hover {{
    background-color: {css_color(STAGE_HOVER["stable"], 0.22)};
    border: 1px solid {STAGE_HOVER["stable"]};
}}
QPushButton#HTile:pressed {{
    background-color: {css_color(STAGE_HOVER["stable"], 0.40)};
    border: 1px solid {STAGE_HOVER["stable"]};
}}

/* -----------------------------------------------------------------
 *  Module tiles (AppTile) — icon over name, one size, every Home tab
 * ----------------------------------------------------------------- */
/* `min-height` and `min-width` are the point of this rule existing
   separately from `#HTile`. See TILE_H in theme.py: without them the
   blanket 22 px QPushButton minimum lets a full page squash every tile
   and draw the name over the icon. Zero padding, because the tile's own
   QVBoxLayout owns its margins — inheriting HTile's left-weighted
   padding would push a centred icon off centre. */
QPushButton#AppTile {{
    background-color: {TILE_BG};
    color: {P["fg"]};
    border: none;
    border-radius: {R["lg"]}px;
    padding: 0px;
    min-height: {TILE_MIN_H}px;
    min-width: {TILE_MIN_W_PX}px;
    text-align: center;
    font-family: "Open Sans", "Segoe UI", "Helvetica Neue", sans-serif;
}}
/* Fallback hover, for a tile whose `stage` property was never set. The
   three stage rules below win over it wherever it was. */
QPushButton#AppTile:hover {{
    background-color: {css_color(STAGE_HOVER["stable"], 0.22)};
    border: 1px solid {STAGE_HOVER["stable"]};
}}
QPushButton#AppTile:pressed {{
    background-color: {css_color(STAGE_HOVER["stable"], 0.40)};
    border: 1px solid {STAGE_HOVER["stable"]};
}}
/* Maturity, as colour. See `STAGE_HOVER` and the legend under the Home
   aside — a tile that lights magenta is a beta module, and the legend
   beside it is what says so. */
{STAGE_RULES}
/* A resting rim is decoration; a keyboard-focus ring carries state. Keep it
   after the maturity rules so focus remains visible while a tile is hovered. */
QPushButton#AppTile:focus {{
    border: 1px solid {P["accent"]};
}}
QLabel#HTileName {{
    color: {P["fg"]};
    font-family: "Open Sans", "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: {F["subtitle"]}px;
    font-weight: 400;                 /* Open Sans Regular */
    background: transparent;
    letter-spacing: -0.1px;
}}
QLabel#HTileDesc {{
    color: {P["fg_muted"]};
    font-family: "Open Sans", "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: {F["small"]}px;
    font-weight: 300;                 /* Open Sans Light */
    background: transparent;
}}
/* Home hero subtitle ("End-to-end microscopy → …"): styled here (not inline)
   so it scales with the font-size preference and reads in the primary
   (white on dark) text colour. Sized at "subtitle" rather than "body" so it
   carries the masthead beside the enlarged wordmark instead of trailing it
   as ordinary paragraph text; it still moves with the font preference. */
QLabel#HeroSubtitle {{
    color: {P["fg"]};
    font-family: "Open Sans", "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: {F["subtitle"]}px;
    font-weight: 300;
    background: transparent;
    padding-left: 8px;
}}
/* Sticky bottom hint bar ("Hover a tile to see what it does."): also styled
   here so it scales with the font preference and uses primary text colour. */
QLabel#HintBar {{
    background-color: {P["surface_alt"]};
    border-top: 1px solid {P["border_soft"]};
    color: {P["fg"]};
    font-family: "Open Sans", "Segoe UI", "Helvetica Neue", sans-serif;
    font-weight: 300;
    font-size: {F["body"]}px;
    padding: 8px 12px;
}}

/* -----------------------------------------------------------------
 *  Sliders — blue (accent) handle + filled track, not the default
 *  dark-gray handle.
 * ----------------------------------------------------------------- */
/* THE WIDGET'S OWN BACKGROUND, which no rule had claimed. The groove and
   handle were styled and the QSlider behind them was not, so it painted the
   palette's window colour as an opaque rectangle on a container that is a
   translucent SURFACE -- reported as "there is a figure size slider that has
   a black background in regression module, it should be same color as
   container". A figure control is not a window (INVARIANTS 2); the same
   omission is what made the tab overflow arrows black boxes. */
QSlider {{
    background: transparent;
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: {P["border"]};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {P["accent_lo"]};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {P["accent"]};
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}}
QSlider::handle:horizontal:hover {{
    background: {P["accent_hi"]};
}}
QSlider::groove:vertical {{
    width: 4px;
    background: {P["border"]};
    border-radius: 2px;
}}
QSlider::handle:vertical {{
    background: {P["accent"]};
    width: 16px;
    height: 16px;
    margin: 0 -6px;
    border-radius: 8px;
}}
QSlider::handle:vertical:hover {{
    background: {P["accent_hi"]};
}}

/* -----------------------------------------------------------------
 *  Typography helpers — pair each role with weight + tracking
 * ----------------------------------------------------------------- */
QLabel#Hero {{
    color: {P["fg"]};
    font-size: {F["hero"]}px;
    font-weight: 200;
    letter-spacing: -0.5px;
    background: transparent;
}}
QLabel#DisplayHeading {{
    color: {P["fg"]};
    font-size: {F["display"]}px;
    font-weight: 300;
    letter-spacing: -0.4px;
    background: transparent;
}}
QLabel#TitleHeading {{
    color: {P["fg"]};
    font-size: {F["title"]}px;
    font-weight: 500;
    letter-spacing: -0.2px;
    background: transparent;
}}
QLabel#Subtitle {{
    color: {P["fg_muted"]};
    font-size: {F["subtitle"]}px;
    font-weight: 400;
    background: transparent;
}}
QLabel#SubtitleSmall, QLabel#Muted {{
    color: {P["fg_muted"]};
    font-size: {F["small"]}px;
    background: transparent;
}}
QLabel#Caption {{
    color: {P["fg_dim"]};
    font-size: {F["xs"]}px;
    font-weight: 500;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    background: transparent;
}}
QLabel#SectionHeading {{
    color: {P["fg"]};
    font-size: {F["header"]}px;
    font-weight: 600;
    background: transparent;
}}

/* -----------------------------------------------------------------
 *  Buttons
 * ----------------------------------------------------------------- */
QPushButton {{
    background-color: {P["surface_alt"]};
    color: {P["fg"]};
    border: 1px solid {P["border_soft"]};
    border-radius: {R["sm"]}px;
    padding: {S["sm"]}px {S["md"]}px;
    min-height: 22px;
    font-weight: 500;
}}
QPushButton:hover {{
    background-color: {P["surface_hi"]};
    border-color: {P["border"]};
    color: {P["fg"]};
}}
QPushButton:pressed {{
    background-color: {P["accent_lo"]};
    border-color: {P["accent_lo"]};
    color: {P["bg"]};
}}
QPushButton:checked {{
    background-color: {P["accent_soft"]};
    border-color: {P["accent"]};
    color: {P["accent"]};
}}
QPushButton:checked:hover {{
    background-color: {P["accent_soft"]};
    border-color: {P["accent_hi"]};
    color: {P["accent_hi"]};
}}
QPushButton:disabled {{
    color: {P["fg_dim"]};
    border-color: {P["border_soft"]};
    background-color: {P["surface"]};
}}
/* Semantic action buttons: outlined at rest, softly tinted on hover, and
 * solid while pressed or while an asynchronous action remains active.
 * buttonActionRole is assigned centrally by button_roles.py. */
QPushButton#PrimaryButton,
QPushButton[buttonActionRole="positive"] {{
    background-color: transparent;
    color: {P["button_accent"]};
    border: 1px solid {P["button_accent"]};
    font-weight: 600;
    padding: {S["sm"]}px {S["lg"]}px;
}}
QPushButton#PrimaryButton:hover,
QPushButton[buttonActionRole="positive"]:hover {{
    background-color: {css_color(P["button_accent"], 0.18)};
    color: {P["button_accent"]};
    border-color: {P["button_accent"]};
}}
QPushButton#PrimaryButton:pressed,
QPushButton[buttonActionRole="positive"]:pressed,
QPushButton#PrimaryButton[buttonActionBusy="true"],
QPushButton[buttonActionRole="positive"][buttonActionBusy="true"] {{
    background-color: {P["button_accent"]};
    color: {P["button_accent_ink"]};
    border-color: {P["button_accent"]};
}}
QPushButton#DangerButton,
QPushButton[buttonActionRole="negative"] {{
    background-color: transparent;
    color: {P["error"]};
    border: 1px solid {P["error"]};
    font-weight: 600;
    padding: {S["sm"]}px {S["lg"]}px;
}}
QPushButton#DangerButton:hover,
QPushButton[buttonActionRole="negative"]:hover {{
    background-color: {css_color(P["error"], 0.18)};
    color: {P["error"]};
    border-color: {P["error"]};
}}
QPushButton#DangerButton:pressed,
QPushButton[buttonActionRole="negative"]:pressed,
QPushButton#DangerButton[buttonActionBusy="true"],
QPushButton[buttonActionRole="negative"][buttonActionBusy="true"] {{
    background-color: {P["error"]};
    color: {P["bg"]};
    border-color: {P["error"]};
}}
QPushButton[buttonActionRole="positive"]:disabled,
QPushButton[buttonActionRole="negative"]:disabled {{
    background-color: transparent;
}}
QPushButton[buttonActionRole="positive"][buttonActionBusy="true"]:disabled {{
    background-color: {P["button_accent"]};
    color: {P["button_accent_ink"]};
    border-color: {P["button_accent"]};
}}
QPushButton[buttonActionRole="negative"][buttonActionBusy="true"]:disabled {{
    background-color: {P["error"]};
    color: {P["bg"]};
    border-color: {P["error"]};
}}
QPushButton#GhostButton {{
    background-color: transparent;
    color: {P["fg_muted"]};
    border: none;
}}
QPushButton#GhostButton:hover {{
    color: {P["accent"]};
    background: transparent;
}}
QPushButton#IconButton {{
    background-color: transparent;
    border: none;
    padding: {S["xs"]}px;
    min-height: 0;
    color: {P["fg_muted"]};
}}
QPushButton#IconButton:hover {{
    color: {P["accent"]};
    background: {P["surface_alt"]};
    border-radius: {R["sm"]}px;
}}
/* Provider picker beside the AI text toggle. It is a QToolButton, so the
   QPushButton rules above do not reach it. Qt's native dark primitive filled
   368 of its 525 rendered pixels with pure black: one visible square in the
   Regression action row. The row supplies the resting surface; only hover and
   open states need a fill of their own. */
QToolButton#AiProviderMenuButton {{
    background: transparent;
    color: {P["fg_muted"]};
    border: none;
    border-radius: {R["sm"]}px;
    min-width: 20px;
    min-height: 20px;
    padding: 2px;
}}
QToolButton#AiProviderMenuButton:hover {{
    background: {P["surface_alt"]};
    color: {P["fg"]};
}}
QToolButton#AiProviderMenuButton:pressed,
QToolButton#AiProviderMenuButton:open {{
    background: {P["accent_soft"]};
    color: {P["accent"]};
}}
/* SQL column pickers are QToolButtons, so they do not inherit the normal
   QPushButton treatment.  Keep them visually part of the settings card:
   a dark card-coloured face, a light neutral rim and white text. */
QWidget#ColumnPickerRow {{
    background: transparent;
}}
QToolButton#ColumnPickerButton {{
    background-color: {P["surface"]};
    color: {P["fg"]};
    border: 1px solid {P["fg_muted"]};
    border-radius: {R["sm"]}px;
    padding: {S["xs"]}px {S["sm"]}px;
    min-height: 22px;
    font-weight: 500;
}}
QToolButton#ColumnPickerButton:hover {{
    background-color: {P["accent_soft"]};
    color: {P["fg"]};
    border-color: {P["accent"]};
}}
QToolButton#ColumnPickerButton:pressed {{
    background-color: {P["accent_lo"]};
    color: {P["fg"]};
    border-color: {P["accent_hi"]};
}}
QToolButton#ColumnPickerButton:disabled {{
    background-color: {P["surface"]};
    color: {P["fg_dim"]};
    border-color: {P["border"]};
}}

/* -----------------------------------------------------------------
 *  Inputs (QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox)
 * ----------------------------------------------------------------- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background-color: {P["surface_alt"]};
    color: {P["fg"]};
    border: 1px solid {P["border"]};
    border-radius: {R["sm"]}px;
    padding: {S["xs"]}px {S["sm"]}px;
    selection-background-color: {P["accent"]};
    selection-color: {P["bg"]};
}}
QPlainTextEdit#Console {{
    background-color: {CONSOLE_BG};
    color: #d4d7dc;
    border: 1px solid {P["border_soft"]};
    font-family: "Open Sans", "Segoe UI", "Helvetica Neue", sans-serif;
    font-weight: 300;
    font-size: {F["small"]}px;
    padding: {S["sm"]}px;
    selection-background-color: {P["accent_lo"]};
}}
QFrame#ConsoleSectionResizeHandle {{
    background: transparent;
    border: none;
    border-bottom: 1px solid {P["border_soft"]};
}}
QFrame#ConsoleSectionResizeHandle:hover {{
    border-bottom: 2px solid {P["accent"]};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QComboBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{
    border: 1px solid {P["accent"]};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
QComboBox:disabled {{
    color: {P["fg_dim"]};
    background-color: {P["surface"]};
    border-color: {P["border_soft"]};
}}
QLineEdit::placeholder {{
    color: {P["fg_dim"]};
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background: transparent;
    border: none;
    width: 16px;
}}
QSpinBox::up-arrow, QSpinBox::down-arrow,
QDoubleSpinBox::up-arrow, QDoubleSpinBox::down-arrow {{
    width: 8px; height: 8px;
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 24px;
    border: none;
}}
QComboBox::down-arrow {{
    image: none;
    border: 4px solid transparent;
    border-top-color: {P["fg_muted"]};
    margin-top: 4px;
    width: 0;
    height: 0;
}}
QComboBox QAbstractItemView {{
    background-color: {ELEVATED};
    color: {P["fg"]};
    border: 1px solid {P["border"]};
    border-radius: {R["sm"]}px;
    padding: {S["xs"]}px;
    selection-background-color: {P["accent"]};
    selection-color: {P["bg"]};
}}
/* UMAP search inputs sit directly on a raised gray card. Use the adjacent
   theme surface instead of a native/base palette, which rendered these three
   controls black on some Linux Qt styles. */
QWidget#UmapHyperparamControls QLineEdit,
QWidget#UmapHyperparamControls QSpinBox,
QWidget#UmapHyperparamControls QDoubleSpinBox,
QWidget#UmapHyperparamControls QComboBox {{
    background-color: {P["surface_hi"]};
    color: {P["fg"]};
}}

/* -----------------------------------------------------------------
 *  Checkboxes + toggles
 * ----------------------------------------------------------------- */
QCheckBox {{
    color: {P["fg"]};
    background: transparent;
    spacing: {S["sm"]}px;
    padding: 2px 0px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {P["border"]};
    border-radius: {R["sm"]}px;
    background: {P["surface_alt"]};
}}
QCheckBox::indicator:hover {{
    border-color: {P["accent"]};
}}
QCheckBox::indicator:checked {{
    background: {P["accent"]};
    border-color: {P["accent"]};
    image: none;
}}
QCheckBox::indicator:disabled {{
    background: {P["surface"]};
    border-color: {P["border_soft"]};
}}
QRadioButton {{
    color: {P["fg"]};
    background: transparent;
    spacing: {S["sm"]}px;
}}
QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {P["border"]};
    border-radius: 8px;
    background: {P["surface_alt"]};
}}
QRadioButton::indicator:checked {{
    background: {P["accent"]};
    border-color: {P["accent"]};
}}

/* -----------------------------------------------------------------
 *  Scrollbars
 * ----------------------------------------------------------------- */
QScrollBar:vertical {{
    background: {TROUGH};
    width: 10px;
    margin: 0px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: {P["border"]};
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {P["accent"]};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    background: transparent; height: 0px;
}}
QScrollBar:horizontal {{
    background: {TROUGH};
    height: 10px;
    margin: 0px;
    border: none;
}}
QScrollBar::handle:horizontal {{
    background: {P["border"]};
    border-radius: 5px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {P["accent"]};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    background: transparent; width: 0px;
}}

/* -----------------------------------------------------------------
 *  Progress bar
 * ----------------------------------------------------------------- */
QProgressBar {{
    background: {P["surface"]};
    border: none;
    border-radius: 4px;
    text-align: center;
    color: {P["fg_muted"]};
    height: 8px;
    max-height: 8px;
}}
QProgressBar::chunk {{
    background-color: {P["accent"]};
    border-radius: 4px;
}}
QProgressBar#UsageBar {{
    /* The System card already paints surface_alt. A second translucent fill
       here compounds the page opacity and makes each track a darker slab. */
    background: transparent;
    height: 6px;
    max-height: 6px;
}}
QProgressBar#UsageBarWarn, QProgressBar#UsageBarError {{
    background: transparent;
    height: 6px;
    max-height: 6px;
}}
QProgressBar#UsageBar::chunk {{
    background: {P["accent"]};
    border-radius: 3px;
}}
QProgressBar#UsageBarWarn::chunk {{
    background: {P["warning"]};
}}
QProgressBar#UsageBarError::chunk {{
    background: {P["error"]};
}}

/* -----------------------------------------------------------------
 *  Splitter handle
 * ----------------------------------------------------------------- */
QSplitter::handle {{
    background: {P["border_soft"]};
}}
QSplitter::handle:horizontal {{
    width: 1px;
}}
QSplitter::handle:vertical {{
    height: 1px;
}}
QSplitter::handle:hover {{
    background: {P["accent"]};
}}

/* -----------------------------------------------------------------
 *  Tooltip
 * ----------------------------------------------------------------- */
QToolTip {{
    background-color: {ELEVATED};
    color: {P["fg"]};
    border: 1px solid {P["border"]};
    border-radius: {R["sm"]}px;
    padding: {S["xs"]}px {S["sm"]}px;
    font-size: {F["small"]}px;
}}

/* -----------------------------------------------------------------
 *  AI Console chat bubbles (legacy standalone panel)
 * ----------------------------------------------------------------- */
QLabel#ChatBubbleUser {{
    background-color: {P["accent_soft"]};
    color: {P["fg"]};
    border: 1px solid {P["accent_lo"]};
    border-radius: {R["md"]}px;
    padding: {S["sm"]}px {S["md"]}px;
    font-size: {F["body"]}px;
}}
QLabel#ChatBubbleAssistant {{
    background-color: {P["surface_alt"]};
    color: {P["fg"]};
    border: 1px solid {P["border_soft"]};
    border-radius: {R["md"]}px;
    padding: {S["sm"]}px {S["md"]}px;
    font-size: {F["body"]}px;
}}

/* -----------------------------------------------------------------
 *  Tabs — Classifier Evaluation, Run History, and anything else
 * -----------------------------------------------------------------
 *  There was no generic tab styling, so Qt's own drew a flat opaque
 *  strip with a large dark pane under it. These follow the Home tabs:
 *  rounded top corners, a dark surface at the page opacity, and the
 *  accent blue on hover so the tab under the pointer is unambiguous.
 *
 *  Scoped by :not() on the Home tab widget, which keeps its own rules —
 *  it is the one place the pane is deliberately empty because the tiles
 *  carry the fill.
 */
QTabWidget:!hover {{ }}
QTabWidget::pane {{
    background-color: {P["surface_alt"]};
    border: 1px solid {P["border_soft"]};
    border-radius: {R["md"]}px;
    top: -1px;
}}
QTabBar {{
    background: transparent;
}}
/* THE OVERFLOW ARROWS ARE NOT OURS AND LOOK IT. Qt draws a tab bar that does
   not fit with two QToolButton scrollers, and no rule in this sheet claimed
   them -- so they came out as opaque boxes with white arrows on every theme.
   Reported 2026-08-19: "two arrows that are visable black boxes with white
   arrows. these are ugly and can be removed."
   Styled rather than hidden with a width of 0: a bar that genuinely
   overflows still needs a way along it, and `setUsesScrollButtons` is where
   a screen decides that. This makes them belong to the theme. */
QTabBar::scroller {{
    width: {S["lg"]}px;
}}
QTabBar QToolButton {{
    background: transparent;
    border: none;
    color: {P["fg_muted"]};
}}
QTabBar QToolButton:hover {{
    background: {P["surface"]};
    border-radius: {R["sm"]}px;
    color: {P["fg"]};
}}
QTabBar QToolButton:disabled {{
    color: {P["border_soft"]};
}}
QTabBar::tab {{
    background-color: {P["surface"]};
    color: {P["fg_muted"]};
    border: 1px solid {P["border_soft"]};
    border-bottom: none;
    border-top-left-radius: {R["sm"]}px;
    border-top-right-radius: {R["sm"]}px;
    padding: {S["xs"]}px {S["md"]}px;
    margin-right: 2px;
    /* Stated, not inherited. A tab strip that acquires a sheet of its
       own later (Home's does) loses the blanket QWidget font-size, so
       Zoom stops reaching the tab text unless the size is on the rule
       that styles it. */
    font-size: {F["body"]}px;
}}
QTabBar::tab:hover {{
    background-color: {P["accent"]};
    color: {P["bg"]};
}}
QTabBar::tab:selected {{
    background-color: {P["surface_alt"]};
    color: {P["accent"]};
    border-bottom-color: {P["surface_alt"]};
}}

/* -----------------------------------------------------------------
 *  Tables — every module that shows one
 * -----------------------------------------------------------------
 *  There was no table styling at all, so Qt's own took over: white
 *  header text on a flat black bar that ignored the page opacity
 *  because it was never a spaCR surface to begin with.
 *
 *  Each header cell is now its own rounded dark chip with a gap beside
 *  it, the body carries the page opacity like every other panel, the
 *  grid is a light hairline, and hovering a row turns it the accent
 *  blue so the cell under the pointer is unambiguous.
 */
QTableView, QTableWidget, QTreeView, QTreeWidget {{
    background-color: {P["surface_alt"]};
    alternate-background-color: {P["surface"]};
    gridline-color: {P["border_soft"]};
    border: 1px solid {P["border_soft"]};
    border-radius: {R["md"]}px;
    selection-background-color: {P["accent"]};
    selection-color: {P["bg"]};
}}
QHeaderView {{
    background: transparent;
    border: none;
}}
QHeaderView::section {{
    background-color: {P["surface_hi"]};
    color: {P["fg"]};
    border: none;
    /* The gap that separates one chip from the next. A right/bottom
       margin rather than a border, so the surface behind shows between
       them instead of a drawn line. */
    margin: 0px 2px 2px 0px;
    padding: {S["xs"]}px {S["sm"]}px;
    border-radius: {R["sm"]}px;
    font-weight: 600;
}}
QHeaderView::section:hover {{
    background-color: {P["accent"]};
    color: {P["bg"]};
}}
QTableView::item, QTableWidget::item,
QTreeView::item, QTreeWidget::item {{
    padding: {S["xs"]}px {S["sm"]}px;
    border: none;
}}
QTableView::item:hover, QTableWidget::item:hover,
QTreeView::item:hover, QTreeWidget::item:hover {{
    background-color: {P["accent"]};
    color: {P["bg"]};
}}
/* SELECTION. There was a `:hover` rule and no `:selected` one, so every
   multi-select view in the app fell through to Qt's own selection colours
   -- which assume a light background and paint BLACK text. On the dark
   theme the chosen rows were the only unreadable thing on screen, and in
   the SQL column picker the selection IS the state of the dialog: invisible
   selection means no way to tell what you are about to query.
   `QListWidget` is named explicitly because it is not a `QTableView` and
   nothing above covers it. */
QListView::item:selected, QListWidget::item:selected,
QTableView::item:selected, QTableWidget::item:selected,
QTreeView::item:selected, QTreeWidget::item:selected {{
    background-color: {P["accent"]};
    color: {SELECTION_INK};
}}
/* Kept readable when the view loses focus. Qt dims the selection to a grey
   that is close enough to the surface on the dark themes to read as
   unselected, which is how a picked column disappears the moment the user
   clicks the OK button. */
QListView::item:selected:!active, QListWidget::item:selected:!active,
QTableView::item:selected:!active, QTableWidget::item:selected:!active,
QTreeView::item:selected:!active, QTreeWidget::item:selected:!active {{
    background-color: {P["accent"]};
    color: {SELECTION_INK};
}}
/* The empty square where the two headers meet. Left unstyled it is the
   one opaque corner in an otherwise translucent table. */
QTableCornerButton::section {{
    background-color: {P["surface_hi"]};
    border: none;
    border-radius: {R["sm"]}px;
}}

/* -----------------------------------------------------------------
 *  Merged Console (pipeline stdout + AI chat)
 * ----------------------------------------------------------------- */
/* The panel is just a transparent container: the rounded box is the
   ConsoleBox frame (wrapping the scroll), and the AI chat input sits UNDER it
   as its own edge-aligned row. */
QWidget#ConsolePanel {{
    background-color: transparent;
    border: none;
}}
/* The console box KEEPS its dark surface, at the page opacity. Making it
   transparent (tried, reverted) left a rounded outline floating on the opaque
   container behind it — the fill is what makes it read as a console. What has
   to go is that container, which `_clear_page_surfaces` now tags. */
QFrame#ConsoleBox {{
    background-color: {P["surface_alt"]};
    border: 1px solid {P["border_soft"]};
    border-radius: {R["md"]}px;
}}
/* The AI chat text box under the console — its own rounded field, edges flush
   with the console + system boxes. */
QPlainTextEdit#ConsoleChatInput, QTextEdit#ConsoleChatInput {{
    background-color: {P["surface_alt"]};
    border: 1px solid {P["border_soft"]};
    border-radius: {R["md"]}px;
    padding: {S["sm"]}px {S["md"]}px;
    color: {P["fg"]};
}}
/* Transparent so the box's rounded surface shows through at the corners
   (a solid child background would square them off). */
QWidget#ConsoleHolder {{
    background-color: transparent;
}}
QScrollArea#ConsoleScroll {{
    background-color: transparent;
    border: none;
}}
QFrame#ConsoleTopicBar {{
    background-color: {P["surface_hi"]};
    border-top: 1px solid {P["border_soft"]};
    border-bottom: 1px solid {P["border_soft"]};
}}
QLabel#ConsoleTopicLabel {{
    color: {P["fg_muted"]};
    font-size: {F["small"]}px;
    font-weight: 600;
    letter-spacing: 0.4px;
    background: transparent;
}}
QPlainTextEdit#ConsoleStdoutBlock {{
    color: {P["fg"]};
    background-color: {P["surface_alt"]};
    border: none;
    font-family: "Open Sans", "Segoe UI", "Helvetica Neue", sans-serif;
    font-weight: 300;
    font-size: {F["small"]}px;
    padding: {S["sm"]}px {S["md"]}px;
}}
QPlainTextEdit#ConsoleStdoutBlockError {{
    color: {P["error"]};
    background-color: {P["surface_alt"]};
    border: none;
    font-family: "Open Sans", "Segoe UI", "Helvetica Neue", sans-serif;
    font-weight: 300;
    font-size: {F["small"]}px;
    padding: {S["sm"]}px {S["md"]}px;
}}
QFrame#ConsoleBubbleUser {{
    background-color: #163b28;                 /* dark green */
    border: none;
    border-top: 1px solid #2a6a48;
    border-bottom: 1px solid #2a6a48;
    border-radius: 0px;
}}
QFrame#ConsoleBubbleAI {{
    background-color: {P["accent_soft"]};      /* dark blue */
    border: none;
    border-top: 1px solid {P["accent_lo"]};
    border-bottom: 1px solid {P["accent_lo"]};
    border-radius: 0px;
}}
QFrame#ConsoleInputBar {{
    background-color: {P["surface"]};
    border-top: 1px solid {P["border_soft"]};
}}

/* -----------------------------------------------------------------
 *  Section — collapsible dropdown (custom widget)
 * ----------------------------------------------------------------- */
QFrame#SectionCard {{
    background-color: {P["surface"]};
    border: 1px solid {P["border_soft"]};
    border-radius: {R["md"]}px;
    margin-bottom: {S["sm"]}px;
}}
/* THE RESTING HEADING IS THE FOREGROUND (198). It was `fg_muted` at rest
   and `fg` only on hover or when open -- backwards, because the
   unhighlighted state is the one a user READS: on a screen with sixteen
   folded categories at most one is open and the rest are what they are
   scanning to decide where to go. Dimming them says "secondary" about the
   only thing on the page that is not.

   The highlight is still visible: hover and checked keep `surface_alt`
   behind them, and checked keeps its underline. What stopped distinguishing
   the states is the text going away. */
QToolButton#SectionHeader {{
    background: transparent;
    color: {P["fg"]};
    border: none;
    border-radius: {R["md"]}px;
    padding: {S["sm"]}px {S["md"]}px;
    text-align: left;
    font-size: {F["small"]}px;
    font-weight: 600;
    letter-spacing: 0.6px;
}}
QToolButton#SectionHeader:hover {{
    color: {P["fg"]};
    background: {P["surface_alt"]};
}}
QToolButton#SectionHeader:checked {{
    color: {P["fg"]};
    background: {P["surface_alt"]};
    border-bottom: 1px solid {P["border_soft"]};
    border-bottom-left-radius: 0px;
    border-bottom-right-radius: 0px;
}}
QWidget#SectionBody {{
    background-color: transparent;
    border-bottom-left-radius: {R["md"]}px;
    border-bottom-right-radius: {R["md"]}px;
}}
/* The same maturity colours used by Home, carried into every module's
   settings. Labels keep the theme's readable text ink; the coloured rule is
   the maturity signal, so alpha cyan remains legible on the light theme. */
{SECTION_STAGE_RULES}
{FOLD_STAGE_RULES}
QPushButton#FoldButton {{
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 0px;
}}

/* -----------------------------------------------------------------
 *  Group box (used by settings sections)
 * ----------------------------------------------------------------- */
QGroupBox {{
    background: transparent;
    border: 1px solid {P["border_soft"]};
    border-radius: {R["md"]}px;
    margin-top: {S["md"]}px;
    padding: {S["md"]}px {S["sm"]}px {S["sm"]}px;
    color: {P["fg_muted"]};
    font-weight: 600;
    font-size: {F["small"]}px;
    text-transform: uppercase;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: {S["md"]}px;
    top: -{S["xs"]}px;
    padding: 0px {S["xs"]}px;
    /* The title is text on its owning container, not a separate black notch.
       Transparency keeps it matched when the same group box is placed on a
       section card, popup canvas or preview surface. */
    background: transparent;
}}

/* -----------------------------------------------------------------
 *  Status bar
 * ----------------------------------------------------------------- */
QStatusBar {{
    background: {P["surface"]};
    color: {P["fg_muted"]};
    border-top: 1px solid {P["border_soft"]};
    font-size: {F["small"]}px;
    padding: 0px {S["sm"]}px;
}}
{GLASS_LAYER}{WIDGET_QSS}{CLOSE_MARK_RULES}
"""


# ---------------------------------------------------------------------------
# Shared marks — one glyph per gesture, drawn the same way everywhere
# ---------------------------------------------------------------------------
# A close mark restyled at each site drifts at the next one. The montage's
# well tabs drew their own `×` at the tab bar's font, the gate-editor chip
# drew one in the chip's ink, the folded pages got whatever small pixmap the
# platform style felt like, and the value chips coloured theirs `fg_muted`.
# There is ONE mark now: the glyph, its size, its hit target and its two
# colours live here, and every site asks for it instead of describing it
# again.

#: The close mark. U+2715 MULTIPLICATION X is a full-height stroked X.
#: `×` (U+00D7 MULTIPLICATION SIGN) is a *maths operator* drawn at x-height,
#: which is why the marks it replaces read as small however large the font
#: was set.
CLOSE_MARK = "✕"

#: How much larger than body text the mark is drawn. "A large X" was the
#: ask, and body text is what the tab title beside it uses -- so the mark
#: reads as larger than the title it sits next to, and as a great deal
#: larger than the 16 px pixmap Qt draws on a closable tab.
CLOSE_MARK_SCALE = 1.15

#: Breathing room around the glyph inside its square, in px. Also what
#: keeps the mark from touching the tab title on its left.
CLOSE_MARK_PAD_PX = 8

#: The smallest square the mark stays clickable inside. Qt's own tab close
#: button is 16 px, so this is also the floor that keeps a *larger glyph*
#: from arriving with a *smaller target*.
CLOSE_MARK_HIT_PX = 22

#: Dynamic property that puts a widget under the shared close-mark rules.
#: Set it through :func:`apply_close_mark`; :func:`close_mark_rules` keys on
#: it, and a sweep for close marks keys on it too.
CLOSE_MARK_PROPERTY = "spacrCloseMark"


def close_mark_colours(theme: str = "dark") -> Dict[str, str]:
    """Return normal, hover, and disabled close-mark colours for a theme."""
    P = palette_for(theme)
    return {"rest": P["fg"], "hover": P["error"], "disabled": P["fg_dim"]}


def close_mark_font_px(body_px: Optional[int] = None) -> int:
    """Return the close-mark font size in pixels.

    :param body_px: Resolved body-text size. ``None`` uses :func:`font_px`.
    """
    base = font_px("body") if body_px is None else int(body_px)
    return max(12, int(round(base * CLOSE_MARK_SCALE)))


def close_mark_rules(theme: str = "dark",
                     body_px: Optional[int] = None) -> str:
    """Return the shared Qt style-sheet rules for close marks."""
    ink = close_mark_colours(theme)
    size = close_mark_font_px(body_px)
    prop = CLOSE_MARK_PROPERTY
    return f"""
/* -----------------------------------------------------------------
 *  The one close mark
 * -----------------------------------------------------------------
 *  Theme ink at rest, red under the pointer, everywhere in the app.
 *  Keyed on a property rather than an object name so a new closable
 *  thing joins by asking for the mark, not by editing this sheet.
 */
*[{prop}="true"] {{
    color: {ink["rest"]};
    background: transparent;
    border: none;
    padding: 0px;
    font-size: {size}px;
    font-weight: 400;
}}
*[{prop}="true"]:hover {{
    color: {ink["hover"]};
    background: transparent;
    border: none;
}}
*[{prop}="true"]:pressed {{
    color: {ink["hover"]};
    background: transparent;
    border: none;
}}
*[{prop}="true"]:disabled {{
    color: {ink["disabled"]};
    background: transparent;
    border: none;
}}
"""


def repolish(widget) -> None:
    """Reapply Qt styling after a widget property changes."""
    style = widget.style()
    if style is not None:
        style.unpolish(widget)
        style.polish(widget)
    widget.update()


def close_mark_side(widget=None, body_px: Optional[int] = None) -> int:
    """Return the required side length for a close-mark hit target.

    The result accounts for the rendered glyph, current interface scale, and
    :data:`CLOSE_MARK_HIT_PX` minimum.
    """
    from PySide6.QtGui import QFont, QFontMetrics

    font = QFont(widget.font()) if widget is not None else QFont()
    font.setPixelSize(max(font.pixelSize(), close_mark_font_px(body_px)))
    metrics = QFontMetrics(font)
    # The glyph's own box, not the font's line box. A line box carries
    # ascent, descent and leading for text that is not there, and sizing the
    # square from it made a mark half again as tall as the tab holding it.
    ink = metrics.tightBoundingRect(CLOSE_MARK)
    return max(CLOSE_MARK_HIT_PX,
               ink.width() + CLOSE_MARK_PAD_PX,
               ink.height() + CLOSE_MARK_PAD_PX)


def apply_close_mark(button, *, tooltip: Optional[str] = None,
                     body_px: Optional[int] = None):
    """Apply the shared close-mark glyph, styling, and hit-target size.

    :param button: Qt button to configure.
    :param tooltip: Replacement tooltip. ``None`` preserves the existing
        tooltip.
    :param body_px: Optional resolved body-text size.
    :returns: The configured button.
    """
    button.setText(CLOSE_MARK)
    button.setProperty(CLOSE_MARK_PROPERTY, True)
    # Polished BEFORE it is measured: the style resolves the sheet's
    # font-size onto the widget, so both the glyph below and the control's
    # own size hint describe what will actually be painted.
    repolish(button)
    size_close_mark(button, body_px)
    if getattr(button, "_spacr_close_mark_resizer", None) is None:
        resizer = _CloseMarkResizer(button, body_px)
        button._spacr_close_mark_resizer = resizer
        button.installEventFilter(resizer)
    button.setCursor(Qt.PointingHandCursor)
    set_auto_raise = getattr(button, "setAutoRaise", None)
    if callable(set_auto_raise):
        set_auto_raise(True)
    set_flat = getattr(button, "setFlat", None)
    if callable(set_flat):
        set_flat(True)
    if tooltip is not None:
        button.setToolTip(tooltip)
    return button


def size_close_mark(button, body_px: Optional[int] = None) -> None:
    """Resize a close-mark button for its current font and interface scale."""
    side = close_mark_side(button, body_px)
    hint = button.sizeHint()
    height = max(side, button.minimumHeight(), hint.height())
    width = max(side, button.minimumWidth(), min(hint.width(), height))
    # Unconditional, and deliberately so. This once read
    # `if button.size() != (width, height):`, which compares a QSize against a
    # tuple and is therefore true for every size there is -- so the box has
    # always been re-fixed on every FontChange and StyleChange the style
    # delivered. Making that guard real would change WHEN setFixedSize
    # re-applies its minimum and maximum, so the skip is left out rather than
    # introduced here.
    button.setFixedSize(width, height)


class _CloseMarkResizer(QObject):
    """Re-measure a close mark when the style hands it a new font.

    The box is FIXED so a close mark cannot balloon into the row beside it,
    which means a live Zoom change -- the sheet is rebuilt, the glyph grows,
    the box does not -- would clip the X. Qt sends ``FontChange`` when the
    sheet's font-size reaches the widget; that is the moment to re-measure.
    """

    def __init__(self, button, body_px: Optional[int] = None):
        super().__init__(button)
        self._body_px = body_px

    def eventFilter(self, obj, event):
        """Re-fix the box whenever the style or the font under it moves."""
        if event.type() in (QEvent.FontChange, QEvent.StyleChange):
            try:
                size_close_mark(obj, self._body_px)
            except RuntimeError:
                # The button went away under the event. Nothing to size.
                pass
        return False


def close_mark_button(parent=None, *, tooltip: Optional[str] = None,
                      body_px: Optional[int] = None):
    """Create a standalone close mark as a flat ``QToolButton``."""
    from PySide6.QtWidgets import QToolButton

    return apply_close_mark(QToolButton(parent), tooltip=tooltip,
                            body_px=body_px)


def is_close_mark(widget) -> bool:
    """Return whether a widget uses the shared close-mark styling."""
    return bool(widget is not None and widget.property(CLOSE_MARK_PROPERTY))


class _CloseMarkWatcher(QObject):
    """Re-mark a tab bar whenever Qt hands it a new close button."""

    def __init__(self, bar, tooltip: Optional[str] = None):
        super().__init__(bar)
        self._bar = bar
        self._tooltip = tooltip
        self._pending = False

    def eventFilter(self, obj, event):
        """Schedule a re-mark for the child Qt has just added."""
        if (obj is self._bar and event.type() == QEvent.ChildAdded
                and not self._pending):
            self._pending = True
            QTimer.singleShot(0, self._sweep)
        return False

    def _sweep(self) -> None:
        """Mark whatever arrived, once the bar has finished wiring it up."""
        self._pending = False
        try:
            mark_tab_bar(self._bar, self._tooltip)
        except RuntimeError:
            # The bar went away between the event and this turn of the
            # loop. Nothing left to mark.
            pass


def _request_tab_close(bar, mark, side) -> None:
    """Ask ``bar`` to close whichever tab ``mark`` is sitting on right now.

    The mark is looked up rather than remembered: closing an earlier tab
    renumbers every later one, and a remembered index would then close the
    wrong page.
    """
    for index in range(bar.count()):
        if bar.tabButton(index, side) is mark:
            bar.tabCloseRequested.emit(index)
            return


def mark_tab_bar(bar, tooltip: Optional[str] = None) -> int:
    """Replace existing tab close buttons with the shared close mark.

    Tabs without a close button remain unchanged, and hidden buttons remain
    hidden.

    :returns: Number of close marks installed.
    """
    from PySide6.QtWidgets import QTabBar, QToolButton

    if getattr(bar, "_spacr_marking_tabs", False):
        return 0
    bar._spacr_marking_tabs = True
    replaced = 0
    try:
        for index in range(bar.count()):
            for side in (QTabBar.RightSide, QTabBar.LeftSide):
                existing = bar.tabButton(index, side)
                if existing is None or is_close_mark(existing):
                    continue
                mark = QToolButton(bar)
                text = tooltip if tooltip is not None else existing.toolTip()
                apply_close_mark(mark, tooltip=text or None)
                mark.clicked.connect(
                    lambda *_a, b=bar, m=mark, s=side:
                    _request_tab_close(b, m, s))
                hidden = existing.isHidden()
                bar.setTabButton(index, side, mark)
                # AFTER, not before. `setTabButton` shows whatever it is
                # given, so a mark hidden on the way in comes back visible
                # and puts an X on the page that must not close.
                if hidden:
                    mark.hide()
                replaced += 1
    finally:
        bar._spacr_marking_tabs = False
    return replaced


def install_close_marks(root, *, tooltip: Optional[str] = None) -> int:
    """Install shared close marks on closable tabs below ``root``.

    ``root`` may be a tab widget, tab bar, or containing widget. Event
    filters also style close buttons added later. Repeated calls are
    idempotent.

    :returns: Number of close marks installed during this call.
    """
    from PySide6.QtWidgets import QTabBar, QTabWidget

    if isinstance(root, QTabWidget):
        bars = [root.tabBar()]
    elif isinstance(root, QTabBar):
        bars = [root]
    else:
        bars = [widget.tabBar() for widget in root.findChildren(QTabWidget)]
        bars.extend(root.findChildren(QTabBar))

    installed = 0
    for bar in {id(bar): bar for bar in bars}.values():
        if getattr(bar, "_spacr_close_mark_watcher", None) is None:
            watcher = _CloseMarkWatcher(bar, tooltip)
            bar._spacr_close_mark_watcher = watcher
            bar.installEventFilter(watcher)
        installed += mark_tab_bar(bar, tooltip)
    return installed


# ---------------------------------------------------------------------------
# Tab-bar overflow controls
# ---------------------------------------------------------------------------

def take_the_scroll_arrows_off(root) -> int:
    """Disable overflow buttons for every tab bar below ``root``.

    This changes only the visibility of the scroll buttons. Qt's keyboard and
    mouse-wheel tab navigation remain available.

    Parameters
    ----------
    root : PySide6.QtWidgets.QWidget
        Widget, tab widget, or tab bar to inspect recursively.

    Returns
    -------
    int
        Number of distinct tab bars found.
    """
    from PySide6.QtWidgets import QTabBar, QTabWidget

    bars = []
    if isinstance(root, QTabWidget):
        bars.append(root.tabBar())
    elif isinstance(root, QTabBar):
        bars.append(root)
    for widget in root.findChildren(QTabWidget):
        bars.append(widget.tabBar())
    bars.extend(root.findChildren(QTabBar))

    unique = {id(bar): bar for bar in bars}
    for bar in unique.values():
        bar.setUsesScrollButtons(False)
    return len(unique)
