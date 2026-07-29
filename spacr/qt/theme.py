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

Five themes ship: ``"dark"``, ``"light"``, ``"space"``, ``"cell"`` and
``"glass"``.
(Preferences also offers ``"system"``, which resolves to dark or light
at runtime — it is not a palette of its own.) They are *themes*, not
"modes": "dark mode" stopped being accurate the moment a third one
existed.

Space, Cell and Glass are :data:`IMAGE_THEMES`: dark themes with a visual
backdrop — a generated deep-space render or downloaded photograph for
Space (see :mod:`spacr.qt.space`), one of the user's own micrographs for
Cell (see :mod:`spacr.qt.imagery`), and a built-in blue depth gradient for
Glass. Panels, cards and inputs are drawn as translucent dark scrims so
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

import warnings
from types import MappingProxyType
from typing import Dict, List, Optional, Tuple

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


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
# Glass palette — translucent blue-grey material over a built-in gradient.
# ---------------------------------------------------------------------------
# Qt stylesheets do not expose the compositor's native macOS/iOS backdrop
# blur. The reliable cross-platform equivalent is a layered, translucent
# material: a bounded blue gradient behind low-alpha slate surfaces, with
# cool light rims and high-contrast text. Because the backdrop is generated
# by `_window_block` from this same palette, its maximum brightness is known
# and the ordinary scrim solver can make every grey/black module box genuinely
# see-through without guessing at legibility.
GLASS_PALETTE = {
    "bg":          "#07101e",
    "surface":     "#111b2b",
    "surface_alt": "#18253a",
    "surface_hi":  "#243550",
    "border":      "#7187a5",
    "border_soft": "#526881",
    "fg":          "#f7fbff",
    "fg_muted":    "#c9d3e0",
    "fg_dim":      "#97a8ba",
    "accent":      "#78b9ff",
    "accent_hi":   "#a8d3ff",
    "accent_lo":   "#3f8fdc",
    "accent_soft": "#173551",
    "success":     "#66d995",
    "warning":     "#f2ca66",
    "error":       "#ff8f8a",
    "info":        "#78b9ff",
}


#: The themes with a palette of their own. ``"system"`` is a
#: *preference* value that resolves to one of these, not an entry here.
THEMES = ("dark", "light", "space", "cell", "glass")

_PALETTES = {
    "dark": DARK_PALETTE,
    "light": LIGHT_PALETTE,
    "space": SPACE_PALETTE,
    "cell": CELL_PALETTE,
    "glass": GLASS_PALETTE,
}

#: Themes whose window background is an image or depth gradient rather than
#: a flat colour. They share one treatment — a transparent ``QWidget``
#: default, translucent scrims on panels, and opaque popups — so the QSS
#: branches on membership here rather than on a theme name. The public name
#: is retained because Space and Cell predate the generated Glass backdrop.
IMAGE_THEMES = ("space", "cell", "glass")


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

# Brightest stop in Glass's built-in `_window_block` gradient. Unlike Space,
# Glass cannot accept an arbitrary photograph, so this is a hard rendering
# contract rather than a hopeful estimate.
GLASS_BACKDROP_UNDER = "#173551"


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

#: What the preference means at each end. 100 % is a solid panel — the
#: default, and what every theme drew before the slider existed.
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
    """The alpha the Home pane is actually painted at.

    The user's ``opacity`` (0..1), clamped up to :func:`pane_alpha_floor`.
    ``None`` means :data:`DEFAULT_PANE_OPACITY`.
    """
    if opacity is None:
        opacity = DEFAULT_PANE_OPACITY
    wanted = max(PANE_OPACITY_MIN, min(PANE_OPACITY_MAX, float(opacity)))
    return max(wanted, pane_alpha_floor(theme))


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
CONTRAST_RULES: Tuple[Tuple[str, str, float], ...] = tuple(
    [(fg, surf, 4.5)
     for fg in ("fg", "fg_muted", "accent")
     for surf in ("bg", "surface", "surface_alt", "surface_hi")]
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
       for surf in ("bg", "surface", "surface_alt", "surface_hi")]
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
# regardless of theme. If the AI/LP toggle went accent-blue in dark and
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


def stylesheet(theme: str = "dark", font_scale: float = 1.0,
               background: Optional[str] = None) -> str:
    """Return the QSS string that styles every custom widget in the app.

    :param theme: one of :data:`THEMES`; unknown values fall back to dark.
    :param font_scale: multiplier applied to every font size in
        :data:`FONT_SIZE`. 1.0 = 100 %.
    :param background: path to a background image. Only the themes in
        :data:`IMAGE_THEMES` use it; ``None`` (the default, and what a
        first run mid-generation gets) falls back to a flat gradient.
    """
    base = palette_for(theme)
    S = SPACING
    R = RADIUS
    # Surface roles are re-rendered through the theme's scrim alpha.
    # For dark and light every alpha is 1.0 and this is a no-op that
    # emits the same hex it always did; for Space each one becomes an
    # ``rgba()`` so the background image reads through the panel.
    P = dict(base)
    for role in ("surface", "surface_alt", "surface_hi"):
        P[role] = css_color(base[role], scrim_alpha(theme, role))
    # Opaque variants for the places translucency would be wrong.
    ELEVATED = css_color(base["surface_alt"], scrim_alpha(theme, "elevated"))
    over_image = theme in IMAGE_THEMES
    TILE_BG = (css_color(base["surface"], scrim_alpha(theme, "tile"))
               if over_image else "transparent")
    # Scrollbar troughs and the group-box title notch paint over the
    # window; over a photograph they must not be an opaque black block.
    TROUGH = "transparent" if over_image else base["bg"]
    NOTCH = P["surface_alt"] if over_image else base["bg"]
    CONSOLE_BG = (P["surface_alt"] if over_image else "#0a0b0d")
    # The dock never goes through a scrim — see `dock_colour`.
    DOCK_BG = dock_colour(theme)
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
}}
QLabel[settingMaturity="{stage}"] {{
    border: none;
    border-left: 2px solid {hue};
    padding-left: 6px;
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
/* Every QLabel is transparent by default so it inherits the bg of
 * whatever container it lives in (surface, surface_alt, hero card,
 * etc). Individual labels can override with their own object name. */
QLabel {{
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
QFrame#Hero {{
    background-color: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 1,
        stop: 0 {P["surface_alt"]}, stop: 1 {P["surface"]}
    );
    border: 1px solid {P["border"]};
    border-radius: {R["lg"]}px;
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
   (white on dark) text colour. */
QLabel#HeroSubtitle {{
    color: {P["fg"]};
    font-family: "Open Sans", "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: {F["body"]}px;
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
/* PrimaryButton stays the same colour in dark AND light — users
 * should recognise "Run" (and any other primary action) by hue
 * without their eye having to relearn per theme. The text foreground
 * is white in both themes so contrast holds against #4A9EFF. */
QPushButton#PrimaryButton {{
    background-color: {P["button_accent"]};
    color: {P["button_accent_ink"]};
    border: none;
    font-weight: 600;
    padding: {S["sm"]}px {S["lg"]}px;
}}
QPushButton#PrimaryButton:hover {{
    background-color: {P["button_accent_hi"]};
}}
QPushButton#PrimaryButton:pressed {{
    background-color: {P["button_accent_lo"]};
}}
QPushButton#DangerButton {{
    background-color: transparent;
    color: {P["error"]};
    border: 1px solid {P["error"]};
    font-weight: 600;
    padding: {S["sm"]}px {S["lg"]}px;
}}
QPushButton#DangerButton:hover {{
    background-color: {P["error"]};
    color: {P["bg"]};
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
    font-family: "JetBrains Mono", "Menlo", "Consolas", monospace;
    font-size: {F["small"]}px;
    padding: {S["sm"]}px;
    selection-background-color: {P["accent_lo"]};
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
 *  Merged Console (pipeline stdout + AI chat)
 * ----------------------------------------------------------------- */
/* The panel is just a transparent container: the rounded box is the
   ConsoleBox frame (wrapping the scroll), and the AI chat input sits UNDER it
   as its own edge-aligned row. */
QWidget#ConsolePanel {{
    background-color: transparent;
    border: none;
}}
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
QLabel#ConsoleStdoutBlock {{
    color: {P["fg"]};
    background-color: {P["surface_alt"]};
    font-family: "JetBrains Mono", "Menlo", "Consolas", monospace;
    font-size: {F["small"]}px;
    padding: {S["sm"]}px {S["md"]}px;
}}
QLabel#ConsoleStdoutBlockError {{
    color: {P["error"]};
    background-color: {P["surface_alt"]};
    font-family: "JetBrains Mono", "Menlo", "Consolas", monospace;
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
    background: {NOTCH};
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
"""
