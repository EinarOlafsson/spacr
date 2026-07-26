"""
Themes (palettes + QSS stylesheet) for the spacr Qt GUI.

Single source of truth for every color, radius, and font size used by the
custom widgets and screens. Import `PALETTE` for programmatic access and
`stylesheet()` for the Qt StyleSheet string to hand to
`QApplication.setStyleSheet`.

Four themes ship: ``"dark"``, ``"light"``, ``"space"`` and ``"cell"``.
(Preferences also offers ``"system"``, which resolves to dark or light
at runtime — it is not a palette of its own.) They are *themes*, not
"modes": "dark mode" stopped being accurate the moment a third one
existed.

Space and Cell are :data:`IMAGE_THEMES`: dark themes whose window
background is a picture rather than a colour — a generated deep-space
render or a downloaded photograph for Space (see :mod:`spacr.qt.space`),
one of the user's own micrographs for Cell (see
:mod:`spacr.qt.imagery`). Panels, cards, inputs and dialogs are drawn as
translucent dark scrims so text always lands on a readable surface while
the imagery shows through the chrome and the empty areas.

Legibility over a picture is checked two ways, because the two failure
modes are different:

* :func:`contrast_failures` judges every scrim against the *worst case*
  a background can present — a pure white pixel directly behind the
  panel. Passing that means no image can ever make a panel unreadable.
* :func:`image_contrast_failures` judges the roles that are painted
  with **nothing** under them against a colour measured from the real
  wallpaper. That is the case a scrim cannot help with, and
  :func:`max_background_luma` is what the imagery pipeline dims to.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


# ---------------------------------------------------------------------------
# Palette — kept aligned with the Tk gui_elements apply_theme dict so
# switching between the two GUIs feels visually consistent.
# ---------------------------------------------------------------------------
PALETTE = {
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
# Kept close in structure to PALETTE so a caller can swap between them
# by key without touching consumers.
# ---------------------------------------------------------------------------
# Every value below was picked (or corrected) against
# :data:`CONTRAST_RULES`. The originals failed AA in several places —
# `accent` at 4.10:1 on `surface_hi`, `accent_hi` at 2.81:1 on
# `accent_soft`, `warning` at 3.82:1, `fg_dim` at 2.54:1 — because in a
# light theme "hover" has to go *darker*, not brighter, and the first
# cut of this palette mirrored the dark one literally.
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


#: The themes with a palette of their own. ``"system"`` is a
#: *preference* value that resolves to one of these, not an entry here.
THEMES = ("dark", "light", "space", "cell")

_PALETTES = {
    "dark": PALETTE,
    "light": LIGHT_PALETTE,
    "space": SPACE_PALETTE,
    "cell": CELL_PALETTE,
}

#: Themes whose window background is an image rather than a flat colour.
#: They share one treatment — a transparent ``QWidget`` default so the
#: picture is not covered by every child, translucent scrims on the
#: panels, and opaque popups — so the QSS branches on membership here
#: rather than on a theme name.
IMAGE_THEMES = ("space", "cell")


# ---------------------------------------------------------------------------
# Scrims — per-theme opacity of each surface role
# ---------------------------------------------------------------------------
# Only the image themes are translucent. Everything else resolves to 1.0
# and the QSS emits plain hex, byte-identical to what it emitted before
# scrims existed.
#
# `elevated` (menus, tooltips, combo popups) is deliberately opaque even
# there: those are separate top-level windows, and a translucent popup
# without a compositor shows the desktop, not the app.
SCRIM_ALPHA: Dict[str, Dict[str, float]] = {
    "space": {
        "surface":     0.88,
        "surface_alt": 0.90,
        "surface_hi":  0.93,
        "tile":        0.86,
        "elevated":    1.00,
    },
    # Cell runs the same opacities. They were re-checked against
    # CELL_PALETTE rather than assumed: `contrast_failures("cell")` is
    # empty at these values, over a pure white worst case.
    "cell": {
        "surface":     0.88,
        "surface_alt": 0.90,
        "surface_hi":  0.93,
        "tile":        0.86,
        "elevated":    1.00,
    },
}

#: Worst-case pixel that a scrim can be composited over. A star core is
#: pure white, so that is what the contrast check assumes sits behind
#: every panel — anything dimmer only helps.
WORST_CASE_UNDER = "#ffffff"


def scrim_alpha(theme: str, role: str) -> float:
    """Opacity of surface ``role`` in ``theme``. 1.0 unless translucent."""
    return SCRIM_ALPHA.get(theme, {}).get(role, 1.0)


def palette_for(theme: str = "dark") -> dict:
    """Return the palette dict for ``theme``.

    ``theme`` is one of :data:`THEMES`; anything else (including
    ``"system"``, which the caller is expected to have resolved) falls
    back to the dark palette. The returned dict always carries every
    theme-invariant key from :data:`CONSTANT_ROLES` so callers can hit
    e.g. ``palette_for(t)["button_accent"]`` and know the value is the
    same across themes.
    """
    base = _PALETTES.get(theme, PALETTE)
    out = dict(base)
    out.update(CONSTANT_ROLES)
    return out


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
                      under: str = WORST_CASE_UNDER) -> str:
    """The colour a surface role *actually* presents to the eye.

    For opaque themes that is just the palette entry. For Space it is
    the scrim composited over ``under`` — by default a white star, the
    worst case the background image can put behind a panel.
    """
    palette = palette_for(theme)
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
# resolve to the SAME value in both PALETTE and LIGHT_PALETTE so button
# / toggle styling can rely on them.
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
    # Scaled font sizes so the "Font scale" preference actually
    # resizes the whole app, not just the base body text.
    F = {k: max(6, int(round(v * font_scale)))
         for k, v in FONT_SIZE.items()}
    return f"""
/* -----------------------------------------------------------------
 *  Base
 * ----------------------------------------------------------------- */
{_window_block(theme, base, background, F["body"])}
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
#Sidebar {{
    background-color: {P["surface"]};
    border-right: 1px solid {P["border_soft"]};
}}
#SidebarTitle {{
    color: {P["accent"]};
    font-family: "Open Sans", "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: 24px;
    font-weight: 300;                 /* Light */
    letter-spacing: -0.5px;
    padding: {S["lg"]}px {S["md"]}px {S["md"]}px;
    background: {P["surface"]};
}}
#SidebarSection {{
    color: {P["fg_dim"]};
    font-size: {F["xs"]}px;
    font-weight: 600;
    padding: {S["md"]}px {S["md"]}px {S["xs"]}px;
    text-transform: uppercase;
    letter-spacing: 1px;
    background: {P["surface"]};
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
QPushButton#SidebarItem:hover {{
    background: {P["surface_alt"]};
    color: {P["fg"]};
}}
QPushButton#SidebarItem:checked, QPushButton#SidebarItem[selected="true"] {{
    background: {P["surface_alt"]};
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
QPushButton#HTile {{
    background-color: {TILE_BG};
    color: {P["fg"]};
    border: 1px solid transparent;
    border-radius: {R["lg"]}px;
    padding: 12px 14px 12px 20px;
    text-align: left;
    font-family: "Open Sans", "Segoe UI", "Helvetica Neue", sans-serif;
}}
QPushButton#HTile:hover {{
    background-color: {P["surface_alt"]};
    border: 1px solid {P["border_soft"]};
}}
QPushButton#HTile:pressed {{
    background-color: {P["accent_lo"]};
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
