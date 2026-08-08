"""
Themes (palettes + QSS stylesheet) for the spacr Qt GUI.

Single source of truth for every color, radius, and font size used by the
custom widgets and screens. Call :func:`active_palette` for the colours
that are on screen right now and :func:`stylesheet` for the Qt StyleSheet
string to hand to `QApplication.setStyleSheet`.

.. warning::

   There is deliberately **no module-level ``PALETTE``**. The name used
   to exist, held the *dark* palette, and nothing ever updated it — so
   ``from .theme import PALETTE`` followed by
   ``widget.setStyleSheet(f"background: {PALETTE['surface_alt']}")``
   painted a near-black panel on the light theme's near-white page, and
   any text the app stylesheet inked landed on it at 1.08:1. Black on
   black, measured. The dark palette is now called
   :data:`DARK_PALETTE`, which says what it is; ``theme.PALETTE`` still
   resolves (read-only, with a ``DeprecationWarning``) so the modules
   that have not been migrated yet keep working.

Four themes ship: ``"dark"``, ``"light"``, ``"cell"`` and
``"glass"``.
(Preferences also offers ``"system"``, which resolves to dark or light
at runtime — it is not a palette of its own.) They are *themes*, not
"modes": "dark mode" stopped being accurate the moment a third one
existed.

Space, Cell and Glass are :data:`IMAGE_THEMES`: dark themes with a visual
backdrop — a generated deep-space render or downloaded photograph for
Space (see :mod:`spacr.qt.space`), one of the user's own micrographs for
Cell (see :mod:`spacr.qt.imagery`), and a built-in neutral light field for
Glass. Panels, cards and inputs are drawn as translucent scrims so
text always lands on a readable surface while the backdrop shows through
the chrome and empty areas.

Legibility over a picture is checked two ways, because the two failure
modes are different:

* :func:`contrast_failures` judges every scrim against the worst case
  *that theme's wallpaper pipeline can actually produce* — see
  :func:`scrim_under`. Space's procedural sky keeps its sun blown out on
  purpose, so Space is judged against a pure white pixel; every Cell
  wallpaper goes through :func:`spacr.qt.imagery.render`, which
  exposure-solves it, so Cell is judged against that ceiling.
* :func:`image_contrast_failures` judges the roles that are painted
  with **nothing** under them against a colour measured from the real
  wallpaper. That is the case a scrim cannot help with, and
  :func:`max_background_luma` is what the imagery pipeline dims to.

The scrim opacities themselves are **solved from those two facts plus
one more** — :data:`MIN_PICTURE_CONTRAST`, how much of the picture a
panel must still transmit — rather than picked by eye. See
:func:`solve_scrim_alpha`, and :func:`scrim_report` for the audit
trail. Picking them by eye is what produced a set of panels that passed
every contrast rule and showed 10 % of the photograph underneath, which
users read, reasonably, as the image themes not working.
"""
from __future__ import annotations

import logging
import warnings
from types import MappingProxyType
from typing import Dict, List, Optional, Tuple

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
    """The hairline that outlines every tile — the theme's own ink.

    White in the dark themes, near-black in the light one, because that
    is what ``fg`` already is. Asked for as "a thin white rim, black in
    white mode"; deriving it from ``fg`` rather than writing ``#ffffff``
    means Space and Cell get it for free and no theme can be added that
    silently draws an invisible rim.
    """
    return palette_for(theme)["fg"]


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
    panel: the settings form sits on this surface and the user asked for
    the grey categories to stay grey categories. Taking the floor
    instead would show *more* picture — Cell's floor is 0.05, a panel
    that is not there — at the cost of the form dissolving into the
    wallpaper.

    When the floor lands *above* the ceiling the theme cannot do both,
    legibility wins, and the shortfall is visible in
    :func:`scrim_report`.

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
    """Colours and geometry the field fade paints a field's container with.

    One place so the painter and the QSS that gets out of its way cannot
    drift apart, and so a theme that restyles its inputs restyles the fade
    with them. Every colour is ``(hex, alpha)``: the ramp is applied as a
    **multiplier** on that alpha, so a theme whose border is intrinsically
    translucent (Glass paints a white rim at 16 %) keeps its own material
    and still reaches zero at the right edge, while the flat themes start
    from a genuinely solid 1.0 exactly as the request asks.

    Note what is *not* here: :func:`panel_alpha`. Fields are exempt from
    the page-opacity preference — see :func:`field_fade_alpha`.
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


def palette_for(theme: str = "dark") -> dict:
    """Return the palette dict for ``theme``.

    ``theme`` is one of :data:`THEMES`; anything else (including
    ``"system"``, which the caller is expected to have resolved) falls
    back to the dark palette. The returned dict always carries every
    theme-invariant key from :data:`CONSTANT_ROLES` so callers can hit
    e.g. ``palette_for(t)["button_accent"]`` and know the value is the
    same across themes.
    """
    base = _PALETTES.get(theme, DARK_PALETTE)
    out = dict(base)
    out.update(CONSTANT_ROLES)
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

def _channels(color: str) -> Tuple[int, int, int]:
    text = color.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6:
        raise ValueError(f"not a #rrggbb colour: {color!r}")
    return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))


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
}


# Every ingredient the scrim solver needs — the palettes (including
# these constant roles, which `palette_for` folds in), the contrast
# rules, the colour maths and the exposure ceiling — exists by this
# point, so the alphas can be solved. Done at import so that
# `scrim_alpha` stays a dict lookup on the hot path (the QSS asks for it
# once per role per theme change) and so a palette edit that makes a
# theme unsolvable fails loudly here rather than three screens later.
SCRIM_ALPHA.update(_solve_scrims())


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

    So the sweep cannot simply be narrowed to exact types. Doing that flips
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
QPushButton#Tile, QPushButton#HTile, QPushButton#AppTile {{
    background: {tile};
    border: 1px solid {rim};
    border-radius: 16px;
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
#: This list exists because a rule that is not registered when the
#: stylesheet is built simply is not in it, and the widget it was meant for
#: then falls through to the blanket ``QWidget {{ background-color: bg }}``.
#: ``bg`` is the WINDOW colour -- ``#000000`` on the dark theme -- so an
#: unstyled container is not "slightly off", it is a solid black rectangle.
#:
#: That is the black box behind the settings categories, reported over and
#: over across Mask, Measure, Timelapse, Motility, both Classify screens,
#: Map Barcodes, Regression, External Masks, Illumination, Train Cellpose,
#: Cellpose Masks, Image UMAP, Activation, Barcode QC, Replication,
#: Invasion, Recruitment and Plaque. ``settings_search`` owns
#: ``SettingsSearchPane``, the wrapper around the search strip AND the
#: settings scroll area -- i.e. the entire left column -- and it registers
#: its rules when it is imported, which is when the first module screen is
#: built. The application stylesheet is composed and applied before that.
#: So the first screen of a session opened onto a black column, and any
#: later rebuild of the stylesheet -- switching theme, switching animation,
#: opening enough screens that something re-applied it -- silently fixed
#: it. Every one of those is exactly what was reported.
#:
#: Ordering, not styling: the rules were right the whole time and were not
#: in the sheet yet. Fixing it by styling one more container would have
#: fixed one screen and left the next.
WIDGET_QSS_MODULES: Tuple[str, ...] = (
    "spacr.qt.settings_search",
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
    "spacr.qt.screens.classifier_evaluation",
    "spacr.qt.screens.control_chart",
    "spacr.qt.screens.data_manager",
    "spacr.qt.screens.experiment_design",
    "spacr.qt.screens.hit_list",
    "spacr.qt.screens.image_scatter",
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

    Called from :func:`stylesheet` before it composes anything, so the very
    first sheet of a session carries every rule rather than acquiring them
    as screens happen to be opened.

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


def ensure_widget_qss_applied(*names: str) -> bool:
    """Re-apply the application stylesheet if ``names`` are missing from it.

    The registration seam has one seam of its own, and it is silent. A block
    is registered at its module's **import**, and the application stylesheet
    is generated **once at launch**, before ``MainWindow`` exists. A screen
    that ``app.py`` imports lazily — inside the ``if key == …`` branch that
    builds it — therefore registers its block minutes after the only
    stylesheet that would have carried it, and the screen opens unstyled.

    That is not hypothetical: Model Compare's panels were given a page
    surface, the test that measures them passed (a test imports the module
    before it applies the stylesheet), and the panels were still bare in the
    running app, because ``spacr.qt.screens.model_compare`` is not in
    ``sys.modules`` when the stylesheet is built. The screens listed in
    ``spacr.qt.SELF_REGISTERING_MODULES`` are imported at launch and never
    had the problem, which is why it went unnoticed for as long as it did.

    Call this from the constructor of a screen whose module registers a
    block. It is a no-op in every case except the one it exists for: no
    ``QApplication``, no stylesheet yet, or the block already present.

    :param names: registered block names the caller needs to be live.
    :returns: ``True`` if the stylesheet was regenerated.
    """
    try:
        from PySide6.QtWidgets import QApplication
    except Exception:  # pragma: no cover - PySide6 is a hard dependency here
        return False
    app = QApplication.instance()
    if app is None:
        return False
    sheet = app.styleSheet()
    if not sheet:
        # Nothing has styled the application, so there is nothing to be
        # missing from and re-applying would install a stylesheet the caller
        # never asked for.
        return False
    if all(_WIDGET_QSS_MARKER.format(name=name) in sheet for name in names):
        return False
    try:
        from .preferences import apply_preferences_to_app
    except Exception:
        return False
    try:
        apply_preferences_to_app(app)
    except Exception:
        LOG.exception("Could not re-apply the stylesheet for %s", names)
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
                          opacity: Optional[float] = None) -> str:
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
    parts = []
    for name, fn in list(_WIDGET_QSS.items()):
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
               surface_opacity: Optional[float] = None) -> str:
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
    """
    base = palette_for(theme)
    # Before anything is composed: a block that is not registered yet is
    # not in the sheet, and its widget falls through to the blanket
    # `QWidget { background-color: bg }` -- black on the dark theme. See
    # `WIDGET_QSS_MODULES`.
    load_widget_qss_registrars()

    S = SPACING
    R = RADIUS
    # Surface roles are re-rendered through the theme's scrim alpha.
    # For dark and light every alpha is 1.0 and this is a no-op that
    # emits the same hex it always did; for Space each one becomes an
    # ``rgba()`` so the background image reads through the panel.
    P = dict(base)
    for role in ("surface", "surface_alt", "surface_hi"):
        P[role] = css_color(
            base[role], panel_alpha(theme, role, surface_opacity))
    # Opaque variants for the places translucency would be wrong.
    ELEVATED = css_color(
        base["surface_alt"], panel_alpha(theme, "elevated", surface_opacity))
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
    # The hairline every tile carries, and the three maturity hues its
    # hover switches to. `RIM` is the theme's ink; the hover fill is the
    # stage colour at a low alpha so the tile lights UP rather than being
    # replaced by a block of magenta.
    RIM = rim_colour(theme)
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
    WIDGET_QSS = registered_widget_qss(
        dict(P, theme=theme, font_scale=font_scale), surface_opacity)
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
/* Settings labels are wrapped with a layout-only QWidget so the teal API dot
 * can sit immediately beside the text. A bare QWidget inherits the window
 * canvas colour; without this rule that wrapper paints a black rectangle on
 * the section's dark-gray surface even though the QLabel itself is transparent.
 * Both wrappers are structural and must show their actual container through. */
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
    background-color: {P["surface"]};
    color: {P["fg_muted"]};
    padding: {S["xs"]}px {S["sm"]}px;
    border-bottom: 1px solid {P["border_soft"]};
    font-size: {F["small"]}px;
}}
QMenuBar::item {{
    background: transparent;
    padding: {S["xs"]}px {S["sm"]}px;
    border-radius: {R["sm"]}px;
}}
QMenuBar::item:selected {{
    background: {P["surface"]};
    color: {P["fg"]};
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
/* NEVER translucent, in any theme — see `dock_colour`. Every widget
   between the dock's edge and its rows is named here, because the
   image themes make `QWidget` transparent by default and one unnamed
   container is enough to put the galaxy back behind the app list. */
#EdgeDrawer, #Sidebar, #SidebarScroll, #SidebarInner {{
    background-color: {DOCK_BG};
}}
#Sidebar {{
    border-right: 1px solid {P["border_soft"]};
}}
#SidebarTitle {{
    color: {P["accent"]};
    font-family: "Open Sans", "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: 24px;
    font-weight: 300;                 /* Light */
    letter-spacing: -0.5px;
    padding: {S["lg"]}px {S["md"]}px {S["md"]}px;
    background: {DOCK_BG};
}}
#SidebarSection {{
    color: {P["fg_dim"]};
    font-size: {F["xs"]}px;
    font-weight: 600;
    padding: {S["md"]}px {S["md"]}px {S["xs"]}px;
    text-transform: uppercase;
    letter-spacing: 1px;
    background: {DOCK_BG};
}}
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
/* `surface_hi`, not `surface_alt`: the dock IS `surface_alt` now, and a
   hover the same colour as the thing under it is no hover at all. */
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
    border: 1px solid {css_color(RIM, 0.35)};
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
    /* Match the System card body (surface_alt) so the RAM/GPU/CPU/VRAM track
       blends into the box it sits in (only the filled chunk stands out)
       instead of reading as a separate black bar. */
    background: {P["surface_alt"]};
    height: 6px;
    max-height: 6px;
}}
QProgressBar#UsageBarWarn, QProgressBar#UsageBarError {{
    background: {P["surface_alt"]};
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
QToolButton#SectionHeader {{
    background: transparent;
    color: {P["fg_muted"]};
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
{GLASS_LAYER}{WIDGET_QSS}
"""
