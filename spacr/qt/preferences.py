"""
User-facing preferences — language, theme, font scale and accessibility.

Persistent settings backed by :class:`PySide6.QtCore.QSettings`, so
they survive app restarts. New knobs can slot in alongside the existing
ones without changing consumers thanks to the small typed API
(``get_theme()`` / ``set_theme(...)`` etc.).

Wire-up:

* :func:`apply_preferences_to_app` — call once at startup and again
  whenever a setting changes; reapplies the stylesheet with the
  current theme + font scale.
* :class:`PreferencesDialog` — the modal Settings dialog opened by
  Ctrl+, (see :mod:`spacr.qt.shortcuts`).

Public API::

    from spacr.qt.preferences import (
        get_theme, set_theme, get_theme_choice, set_theme_choice,
        get_language, set_language,
        get_space_variant, set_space_variant,
        get_cell_variant, set_cell_variant,
        get_space_seed, set_space_seed,
        space_background_path, cell_background_path,
        theme_background_path,
        get_ambient_enabled, set_ambient_enabled,
        get_ambient_theme, set_ambient_theme,
        get_ambient_palette, set_ambient_palette,
        get_ambient_blur, set_ambient_blur,
        get_ambient_speed, set_ambient_speed,
        get_ambient_size, set_ambient_size,
        ambient_default_palette, apply_ambient_preferences,
        get_setting_animations_enabled, set_setting_animations_enabled,
        get_font_scale, set_font_scale,
        get_color_blind_mode, set_color_blind_mode,
        get_db_browser_editable, set_db_browser_editable,
        get_dock_mode, set_dock_mode,
        get_pane_opacity, set_pane_opacity, effective_pane_alpha,
        get_field_fade_enabled, set_field_fade_enabled,
        get_show_alpha, set_show_alpha,
        get_show_beta, set_show_beta, maturity_is_visible,
        apply_preferences_to_app,
        PreferencesDialog,
    )

Values:

* ``theme``: ``"dark"`` | ``"light"`` | ``"space"`` | ``"cell"`` |
  ``"glass"`` | ``"system"`` (default ``"dark"``). ``"system"`` follows
  the reader's OS colour scheme. ``"space"`` is a dark theme over a
  generated deep-space background or a deep-field photograph; ``"cell"``
  uses one of the bundled fluorescence micrographs; ``"glass"`` uses
  neutral layered materials over a built-in light field.
  Space and Cell variants appear directly in the single Theme dropdown;
  their existing persisted variant keys remain backward compatible.
* ``space_seed``: int; the generated sky is deterministic in this.
* ``font_scale``: float, 1.0 = 100 %. Clamped to [0.75, 2.0].
* ``color_blind_mode``: ``"off"`` | ``"deuteranopia"`` | ``"protanopia"``
  | ``"tritanopia"`` (default ``"off"``). Swaps matplotlib rainbow /
  red-green palettes for perceptually-uniform + colour-blind-safe
  alternatives (viridis for continuous, Okabe-Ito for categorical).
* ``db_browser_editable``: bool, default ``False``. Permits the
  Database Browser to open a read-write connection at all; see
  :func:`get_db_browser_editable`.
* ``dock_mode``: ``"auto"`` | ``"locked"`` | ``"hidden"`` (default
  ``"locked"``). Whether the left app dock reveals on hover, is pinned
  open as a permanent column, or is not there at all.
* ``pane_opacity``: int percent, default ``60``. How solid shared surfaces
  are, or the relative material strength in Glass. Clamped up to
  :func:`spacr.qt.theme.pane_alpha_floor` at paint time — the
  preference is a request, legibility is not negotiable.
* ``field_fade``: bool, default ``True``. Whether an input field's
  container and outline ramp from solid on the left to fully transparent
  on the right. Fields are **exempt from ``pane_opacity``** while this is
  on — see :func:`get_field_fade_enabled` and
  :mod:`spacr.qt.widgets.field_fade`.
* ``show_alpha`` / ``show_beta``: bool, both default ``True``. Control
  whether modules and settings at that maturity are shown. Stable features
  are always visible.
* ``ambient_enabled``: bool, default ``True``. Whether module screens
  paint the animated background at all. Turning it off is a first-class
  choice — see :func:`get_ambient_enabled`.
* ``ambient_theme`` / ``ambient_palette``: which animation, and in which
  colours. Validated against
  :data:`spacr.qt.widgets.ambient.AMBIENT_THEMES` and
  :func:`spacr.qt.widgets.ambient.palettes_for` respectively; palettes
  are *per theme*, so see :func:`get_ambient_palette` for how the two
  keys stay consistent with each other.
* ``ambient_blur`` / ``ambient_speed`` / ``ambient_size``: floats, all
  default ``1.0``, all *multipliers* on what the chosen animation already
  does — how soft its shapes are, how fast it moves, and how large its
  elements are. 1.0 is the shipped animation in every theme, exactly, so a
  user who never touches them sees no change. Clamped on read and on write
  to the ranges the engines declare
  (:data:`spacr.qt.widgets.ambient.BLUR_RANGE` and friends).
* ``setting_animations``: bool, default ``True``. Whether a setting's
  hover tooltip plays its explanatory animation beside the text. Off
  leaves the text tooltip exactly as it was, and leaves the purple
  animation dot working — see
  :func:`get_setting_animations_enabled`.
* ``language``: one of the bundled language codes from
  :mod:`spacr.qt.i18n`; defaults to English and falls back safely when a
  persisted value is invalid.
"""
from __future__ import annotations

import logging

from PySide6.QtCore import QSettings

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------

_ORG = "spacr"
_APP = "qt"

_KEY_THEME       = "prefs/theme"
_KEY_LANGUAGE    = "prefs/language"
_KEY_FONT_SCALE  = "prefs/font_scale"
_KEY_CB_MODE     = "prefs/color_blind_mode"
_KEY_VERBOSE_LOG = "prefs/verbose_logging"
_KEY_DB_EDIT     = "prefs/db_browser_editable"
_KEY_DOCK_MODE   = "prefs/dock_mode"
_KEY_PANE_OPACITY = "prefs/pane_opacity"
_KEY_FIELD_FADE = "prefs/field_fade"
_KEY_SHOW_ALPHA = "prefs/show_alpha"
_KEY_SHOW_BETA = "prefs/show_beta"
_KEY_AMBIENT_ENABLED = "prefs/ambient_enabled"
_KEY_AMBIENT_THEME   = "prefs/ambient_theme"
_KEY_AMBIENT_PALETTE = "prefs/ambient_palette"
_KEY_AMBIENT_BLUR    = "prefs/ambient_blur"
_KEY_AMBIENT_SPEED   = "prefs/ambient_speed"
_KEY_AMBIENT_SIZE    = "prefs/ambient_size"
_KEY_SETTING_ANIMATIONS = "prefs/setting_animations"

#: Themes with a palette of their own — mirrors
#: :data:`spacr.qt.theme.THEMES`, restated here so importing this module
#: does not pull in QtGui/QtWidgets.
PALETTE_THEMES = ("dark", "light", "cell", "glass")

#: Persisted values. An existing install has ``prefs/theme`` set to one
#: of dark/light/system/space; those keep resolving exactly as before,
#: and an unrecognised value (hand-edited INI, a downgrade from a build
#: with more themes) falls back to :data:`DEFAULT_THEME` rather than
#: raising.
VALID_THEMES = PALETTE_THEMES + ("system",)
#: Follow the reader's OS colour scheme unless they choose otherwise. A
#: desktop app that ignores the system setting looks broken on a light desktop.
DEFAULT_THEME = "system"

_KEY_SPACE_VARIANT = "prefs/space_variant"
_KEY_SPACE_SEED    = "prefs/space_seed"
_KEY_CELL_VARIANT  = "prefs/cell_variant"

FONT_SCALE_MIN = 0.75
FONT_SCALE_MAX = 2.00
#: Presented as "Zoom" rather than "Font scale", because it scales the whole
#: interface — spacing, tiles, dots and icons move with the type, so calling it
#: a font setting undersells what the control does.
#:
#: 150% is the default: spaCR's natural size was laid out on a 1080p display,
#: and on the 4K panels it is now used on everything reads small.
DEFAULT_FONT_SCALE = 1.5

VALID_CB_MODES = ("off", "deuteranopia", "protanopia", "tritanopia")
DEFAULT_CB_MODE = "off"

# Figure rendering
_KEY_FIG_FORMAT = "prefs/figure_format"
_KEY_FIG_PNG_DPI = "prefs/figure_png_dpi"
VALID_FIG_FORMATS = ("png", "pdf")
DEFAULT_FIG_FORMAT = "pdf"
VALID_PNG_DPIS = (100, 200, 300, 600, 1200)
DEFAULT_PNG_DPI = 300


def _settings() -> QSettings:
    return QSettings(_ORG, _APP)


# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------

def get_language() -> str:
    """Return the persisted UI language code, falling back to English."""
    from .i18n import DEFAULT_LANGUAGE, normalize_language
    raw = _settings().value(_KEY_LANGUAGE, DEFAULT_LANGUAGE)
    return normalize_language(raw)


def set_language(language: str) -> None:
    """Persist one of the bundled UI languages.

    :raises ValueError: if ``language`` is not a supported language code.
    """
    from .i18n import VALID_LANGUAGE_CODES
    code = str(language or "").strip().replace("-", "_")
    if code not in VALID_LANGUAGE_CODES:
        raise ValueError(
            f"unknown language {language!r}. "
            f"Choose from {VALID_LANGUAGE_CODES}."
        )
    _settings().setValue(_KEY_LANGUAGE, code)


# ---------------------------------------------------------------------------
# Figures — display format (png / pdf) + png resolution
# ---------------------------------------------------------------------------

def get_figure_format() -> str:
    """Return the saved figure format, falling back to ``pdf``."""
    raw = str(_settings().value(_KEY_FIG_FORMAT, DEFAULT_FIG_FORMAT)).lower()
    return raw if raw in VALID_FIG_FORMATS else DEFAULT_FIG_FORMAT


def set_figure_format(fmt: str) -> None:
    """Persist a supported figure format.

    :raises ValueError: if ``fmt`` is not ``png`` or ``pdf``.
    """
    if fmt not in VALID_FIG_FORMATS:
        raise ValueError(f"unknown figure format {fmt!r}. "
                          f"Choose from {VALID_FIG_FORMATS}.")
    _settings().setValue(_KEY_FIG_FORMAT, fmt)


def get_figure_png_dpi() -> int:
    """Return the saved PNG resolution, or the 300-DPI default."""
    try:
        raw = int(_settings().value(_KEY_FIG_PNG_DPI, DEFAULT_PNG_DPI))
    except (TypeError, ValueError):
        raw = DEFAULT_PNG_DPI
    return raw if raw in VALID_PNG_DPIS else DEFAULT_PNG_DPI


def set_figure_png_dpi(dpi: int) -> None:
    """Persist one of :data:`VALID_PNG_DPIS`.

    :raises ValueError: if ``dpi`` is not a supported resolution.
    """
    dpi = int(dpi)
    if dpi not in VALID_PNG_DPIS:
        raise ValueError(
            f"unknown PNG resolution {dpi!r}. Choose from {VALID_PNG_DPIS}."
        )
    _settings().setValue(_KEY_FIG_PNG_DPI, dpi)


# Figure colours. Stored as hex strings; "auto" (the default) follows the app
# theme — dark → black background + white text, light → white + black.
_KEY_FIG_BG = "prefs/figure_bg"
_KEY_FIG_FG = "prefs/figure_fg"
_KEY_FIG_TEXT_SIZE = "prefs/figure_text_size"


def get_figure_colors() -> tuple:
    """Return ``(background, text)`` hex colours for rendered figures,
    resolving "auto" against the current theme."""
    bg = str(_settings().value(_KEY_FIG_BG, "auto"))
    fg = str(_settings().value(_KEY_FIG_FG, "auto"))
    if bg == "auto" or fg == "auto":
        # Light is the only light theme; Space is a dark one, so a
        # `== "dark"` test here would have handed it white figures.
        dark = resolve_effective_theme() != "light"
        auto_bg, auto_fg = ("#000000", "#ffffff") if dark else ("#ffffff", "#000000")
        if bg == "auto":
            bg = auto_bg
        if fg == "auto":
            fg = auto_fg
    return bg, fg


def set_figure_colors(bg: str, fg: str) -> None:
    """Persist background and text colour tokens for generated figures."""
    _settings().setValue(_KEY_FIG_BG, bg)
    _settings().setValue(_KEY_FIG_FG, fg)


def get_figure_text_size() -> int:
    """Return the saved figure font size; zero leaves Matplotlib unchanged."""
    try:
        return int(_settings().value(_KEY_FIG_TEXT_SIZE, 0))
    except (TypeError, ValueError):
        return 0   # 0 = leave matplotlib's own sizes alone


def set_figure_text_size(size: int) -> None:
    """Persist a figure font size; zero delegates sizing to Matplotlib."""
    _settings().setValue(_KEY_FIG_TEXT_SIZE, int(size))


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

def get_theme() -> str:
    """Return the saved application theme, or the default when invalid."""
    raw = str(_settings().value(_KEY_THEME, DEFAULT_THEME))
    return raw if raw in VALID_THEMES else DEFAULT_THEME


def set_theme(theme: str) -> None:
    """Persist a supported application theme.

    :raises ValueError: if ``theme`` is not in :data:`VALID_THEMES`.
    """
    if theme not in VALID_THEMES:
        raise ValueError(f"unknown theme {theme!r}. "
                          f"Choose from {VALID_THEMES}.")
    _settings().setValue(_KEY_THEME, theme)


def theme_choices() -> tuple:
    """Return ``(label, token)`` choices for the single Theme control.

    Image variants are represented as composite tokens in the UI while the
    persisted keys remain backward compatible.
    """
    from .imagery import CELL_VARIANTS, title_for

    choices = [
        ("Dark", "dark"),
        ("Light", "light"),
        ("Glass", "glass"),
        ("Follow system", "system"),
    ]
    # Space is gone; its variants are no longer offered. The wallpapers are
    # named for what they show rather than prefixed with a theme the user
    # never has to think about.
    choices.extend(
        (title_for(key), f"cell:{key}")
        for key in CELL_VARIANTS
    )
    return tuple(choices)


def get_theme_choice() -> str:
    """Return the composite token representing the current visual theme."""
    theme = get_theme()
    if theme == "space":
        return f"space:{get_space_variant()}"
    if theme == "cell":
        return f"cell:{get_cell_variant()}"
    return theme


def set_theme_choice(choice: str) -> None:
    """Persist one token from :func:`theme_choices`."""
    valid = {token for _label, token in theme_choices()}
    if choice not in valid:
        raise ValueError(
            f"unknown theme choice {choice!r}. Choose from {sorted(valid)}.")
    if choice.startswith("space:"):
        set_space_variant(choice.split(":", 1)[1])
        set_theme("space")
    elif choice.startswith("cell:"):
        set_cell_variant(choice.split(":", 1)[1])
        set_theme("cell")
    else:
        set_theme(choice)


def space_variants() -> tuple:
    """Every background the Space theme offers.

    The three procedural skies from :mod:`spacr.qt.space` plus the
    photographic ones from :mod:`spacr.qt.imagery`. The photo keys are
    kept out of ``space.VARIANTS`` because those three index
    ``space._VARIANT_MIX`` and a photograph has no mix.
    """
    from .imagery import SPACE_PHOTO_VARIANTS
    from .space import VARIANTS
    return tuple(VARIANTS) + tuple(SPACE_PHOTO_VARIANTS)


def get_space_variant() -> str:
    """Which background the Space theme uses."""
    from .space import DEFAULT_VARIANT
    raw = str(_settings().value(_KEY_SPACE_VARIANT, DEFAULT_VARIANT))
    return raw if raw in space_variants() else DEFAULT_VARIANT


def set_space_variant(variant: str) -> None:
    """Persist a supported procedural or photographic Space variant."""
    valid = space_variants()
    if variant not in valid:
        raise ValueError(f"unknown space variant {variant!r}. "
                          f"Choose from {valid}.")
    _settings().setValue(_KEY_SPACE_VARIANT, variant)


def get_cell_variant() -> str:
    """Which of the user's micrographs the Cell theme uses."""
    from .imagery import CELL_VARIANTS, DEFAULT_CELL_VARIANT
    raw = str(_settings().value(_KEY_CELL_VARIANT, DEFAULT_CELL_VARIANT))
    return raw if raw in CELL_VARIANTS else DEFAULT_CELL_VARIANT


def set_cell_variant(variant: str) -> None:
    """Persist one of the bundled Cell-theme microscopy variants."""
    from .imagery import CELL_VARIANTS
    if variant not in CELL_VARIANTS:
        raise ValueError(f"unknown cell variant {variant!r}. "
                          f"Choose from {CELL_VARIANTS}.")
    _settings().setValue(_KEY_CELL_VARIANT, variant)


def get_space_seed() -> int:
    """Seed for the procedural sky. Same seed → same pixels, forever."""
    from .space import DEFAULT_SEED
    try:
        return int(_settings().value(_KEY_SPACE_SEED, DEFAULT_SEED))
    except (TypeError, ValueError):
        return DEFAULT_SEED


def set_space_seed(seed: int) -> None:
    """Persist the deterministic seed used for procedural backgrounds."""
    _settings().setValue(_KEY_SPACE_SEED, int(seed))


def space_background_path(width: int = 0, height: int = 0):
    """Path of the background image for the Space theme, or ``None``.

    The selected photographic variant is used when its master is installed;
    otherwise the selected procedural sky is generated and cached. Returns
    ``None`` only when neither can be produced, at which point the stylesheet
    paints a flat gradient.

    A missing photo master is not an error and not a dead end — it
    simply falls through to the generated sky, which needs no assets and
    no network. **Never raises and never touches the network.**
    """
    try:
        from . import space
        variant = get_space_variant()
        from .imagery import SPACE_PHOTO_VARIANTS, background_path
        if variant in SPACE_PHOTO_VARIANTS:
            photo = background_path(variant, width, height)
            if photo is not None:
                return photo
        if width <= 0 or height <= 0:
            width, height = space.screen_size()
        return space.background_path(width, height,
                                     variant=variant,
                                     seed=get_space_seed())
    except Exception:
        return None


def cell_background_path(width: int = 0, height: int = 0):
    """Path of the background image for the Cell theme, or ``None``.

    ``None`` when the masters were stripped from the build; the
    stylesheet then paints the Cell gradient, which is a dark teal wash
    rather than anything broken.
    """
    try:
        from .imagery import background_path
        return background_path(get_cell_variant(), width, height)
    except Exception:
        return None


def theme_background_path(theme: str, width: int = 0, height: int = 0):
    """Background image for ``theme``, or ``None`` if it does not use one.

    One place for the "which theme wants which picture" question, so
    :func:`apply_preferences_to_app` and anything else that re-applies
    the stylesheet cannot drift apart.
    """
    if theme == "space":
        return space_background_path(width, height)
    if theme == "cell":
        return cell_background_path(width, height)
    return None


def resolve_effective_theme() -> str:
    """Return the theme to render — one of :data:`PALETTE_THEMES`.

    Resolves ``"system"`` to the OS colour scheme, defaulting to dark
    when Qt can't tell. Every other value passes through, so callers
    that only understand light/dark should compare against ``"light"``
    and treat everything else as dark (Space and Cell are dark themes).
    """
    theme = get_theme()
    if theme in PALETTE_THEMES:
        return theme
    # system — poll Qt's palette hint
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            bg = app.palette().color(app.palette().Window)
            # crude luminance test — < 128 → dark scheme
            lum = (0.299 * bg.red() + 0.587 * bg.green()
                   + 0.114 * bg.blue())
            return "dark" if lum < 128 else "light"
    except Exception:
        pass
    return "dark"


# ---------------------------------------------------------------------------
# Animated background
# ---------------------------------------------------------------------------

#: The animation is on out of the box. It costs nothing while a screen is
#: hidden (:class:`spacr.qt.widgets.ambient.AmbientWidget` stops its timer
#: on ``hideEvent``), so leaving it on does not tax a machine that is busy
#: segmenting on the GPU behind a different tab.
DEFAULT_AMBIENT_ENABLED = True


def get_ambient_enabled() -> bool:
    """Whether module screens paint the animated background.

    Default ``True``. When this is ``False`` no ambient widget should be
    installed at all — and any already-installed one is hidden and
    stopped by :func:`apply_ambient_preferences`, so the toggle takes
    effect the moment Preferences is saved rather than at the next launch.
    """
    return _as_bool(_settings().value(_KEY_AMBIENT_ENABLED,
                                      DEFAULT_AMBIENT_ENABLED),
                    DEFAULT_AMBIENT_ENABLED)


def set_ambient_enabled(on: bool) -> None:
    """Turn the animated background on or off.

    Flushed immediately: module screens re-read this key when they are
    built, and a stale read right after the user cleared the checkbox
    would put the animation back on the very next screen they open.
    """
    settings = _settings()
    settings.setValue(_KEY_AMBIENT_ENABLED, bool(on))
    settings.sync()


def get_ambient_theme() -> str:
    """Which animation module screens paint — see ``AMBIENT_THEMES``.

    Validated on read: a value written by a newer spaCR (or by hand)
    that this build does not know about falls back to the default theme
    rather than propagating an unpaintable name into the widget.
    """
    # Aliased on import: this module's own DEFAULT_THEME is the *app*
    # theme (dark), and shadowing it here would be a trap for the next
    # reader.
    from .widgets.ambient import (
        AMBIENT_THEMES, DEFAULT_THEME as DEFAULT_AMBIENT_THEME,
    )
    raw = str(_settings().value(_KEY_AMBIENT_THEME, DEFAULT_AMBIENT_THEME))
    return raw if raw in AMBIENT_THEMES else DEFAULT_AMBIENT_THEME


def set_ambient_theme(name: str) -> None:
    """Persist one of :data:`spacr.qt.widgets.ambient.AMBIENT_THEMES`.

    Palettes belong to a theme, so switching themes can strand the
    stored palette. Rather than raise — the user picked a theme, not a
    broken pair — the stored palette is repaired in the same write: it
    is kept if the new theme also offers it, and otherwise replaced with
    that theme's default (see :func:`ambient_default_palette`).

    :raises ValueError: if ``name`` is not a known ambient theme.
    """
    from .widgets.ambient import AMBIENT_THEMES, palettes_for
    if name not in AMBIENT_THEMES:
        raise ValueError(f"unknown ambient theme {name!r}. "
                         f"Choose from {AMBIENT_THEMES}.")
    settings = _settings()
    settings.setValue(_KEY_AMBIENT_THEME, name)
    # Repair the companion key against the theme we just stored.
    stored = str(settings.value(_KEY_AMBIENT_PALETTE, ""))
    if stored not in palettes_for(name):
        settings.setValue(_KEY_AMBIENT_PALETTE, ambient_default_palette(name))
    settings.sync()


def ambient_default_palette(theme: str) -> str:
    """The palette a theme falls back to.

    :data:`spacr.qt.widgets.ambient.DEFAULT_PALETTE` when that theme
    offers it (spaCR's own brand colours are the intended default
    everywhere they exist), otherwise the theme's first palette. Never
    raises for an unknown theme — it simply reports the global default.
    """
    from .widgets.ambient import DEFAULT_PALETTE, palettes_for
    valid = palettes_for(theme)
    if DEFAULT_PALETTE in valid:
        return DEFAULT_PALETTE
    return valid[0] if valid else DEFAULT_PALETTE


def get_ambient_palette() -> str:
    """Which colours the current ambient theme is painted in.

    Validated against ``palettes_for(get_ambient_theme())``, so this can
    never hand a widget a palette its theme does not have — not after a
    downgrade, not after a hand-edited INI, and not after a theme change
    that stranded the old palette. Falls back to
    :func:`ambient_default_palette` for the current theme.
    """
    theme = get_ambient_theme()
    fallback = ambient_default_palette(theme)
    from .widgets.ambient import palettes_for
    raw = str(_settings().value(_KEY_AMBIENT_PALETTE, fallback))
    return raw if raw in palettes_for(theme) else fallback


def set_ambient_palette(name: str) -> None:
    """Persist a palette offered by the *current* ambient theme.

    :raises ValueError: if ``name`` is not one of
        ``palettes_for(get_ambient_theme())``. Set the theme first: a
        palette is only meaningful next to the theme that draws it.
    """
    from .widgets.ambient import palettes_for
    valid = palettes_for(get_ambient_theme())
    if name not in valid:
        raise ValueError(f"unknown ambient palette {name!r} for theme "
                         f"{get_ambient_theme()!r}. Choose from {valid}.")
    settings = _settings()
    settings.setValue(_KEY_AMBIENT_PALETTE, name)
    settings.sync()


# --- blur, speed and size --------------------------------------------------
# Three multipliers on whatever the chosen animation already does, rather
# than absolute pixels or seconds. 1.0 means "as shipped" in every theme, so
# a user who never opens these controls sees the animation that was designed;
# the ranges and the clamping live in the widget module next to the engines
# that honour them, because a number this module accepted and the engine
# then rejected would be a preference that silently does nothing.

def _ambient_ranges():
    """``(blur, speed, size)`` ranges and defaults, from the widget module.

    Imported lazily and defended, like every other ambient read here: this
    module is imported headless (no QtGui) in places, and a decorative
    setting is never a reason to fail.
    """
    try:
        from .widgets.ambient import (BLUR_RANGE, DEFAULT_BLUR, DEFAULT_SIZE,
                                      DEFAULT_SPEED, SIZE_RANGE, SPEED_RANGE)
        return ((BLUR_RANGE, DEFAULT_BLUR), (SPEED_RANGE, DEFAULT_SPEED),
                (SIZE_RANGE, DEFAULT_SIZE))
    except Exception:
        return (((0.25, 3.0), 1.0), ((0.1, 4.0), 1.0), ((0.25, 2.5), 1.0))


def _ambient_multiplier(key: str, index: int) -> float:
    (low, high), default = _ambient_ranges()[index]
    try:
        value = float(_settings().value(key, default))
    except (TypeError, ValueError):
        return default
    if value != value:            # NaN — a hand-edited INI can hold one
        return default
    return max(low, min(high, value))


def _set_ambient_multiplier(key: str, index: int, value: float) -> None:
    (low, high), default = _ambient_ranges()[index]
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    if value != value:
        value = default
    settings = _settings()
    settings.setValue(key, max(low, min(high, value)))
    settings.sync()


def get_ambient_blur() -> float:
    """How soft the animated background's shapes are. 1.0 is as designed.

    Clamped to ``spacr.qt.widgets.ambient.BLUR_RANGE`` on read, so a value
    from a newer build or a hand-edited file cannot ask for a blur the
    engines will not paint.
    """
    return _ambient_multiplier(_KEY_AMBIENT_BLUR, 0)


def set_ambient_blur(value: float) -> None:
    """Set the softness multiplier. Out-of-range values are clamped, not
    refused: this is a slider, and there is no user error to report."""
    _set_ambient_multiplier(_KEY_AMBIENT_BLUR, 0, value)


def get_ambient_speed() -> float:
    """How fast the animated background moves, as a multiplier on each
    theme's own motion. 1.0 is as designed."""
    return _ambient_multiplier(_KEY_AMBIENT_SPEED, 1)


def set_ambient_speed(value: float) -> None:
    """Set the motion multiplier. Clamped."""
    _set_ambient_multiplier(_KEY_AMBIENT_SPEED, 1, value)


def get_ambient_size() -> float:
    """How large the animated background's elements are, as a multiplier on
    each theme's own size range. 1.0 is as designed."""
    return _ambient_multiplier(_KEY_AMBIENT_SIZE, 2)


def set_ambient_size(value: float) -> None:
    """Set the element-size multiplier. Clamped."""
    _set_ambient_multiplier(_KEY_AMBIENT_SIZE, 2, value)


def apply_ambient_preferences(app=None) -> None:
    """Push the ambient preferences onto every live ambient widget.

    The user's explicit ask was a toggle that works *now*, so this walks
    the running widget tree the same way
    :func:`spacr.qt.button_roles.install_button_roles` does and updates
    the widgets in place instead of waiting for the screens to be
    rebuilt. Hiding one also stops its timer (the widget stops animating
    whenever it is not visible), so "off" really is zero frames.

    Turning it back *on* only resumes the widgets that are actually on
    screen. Every module screen keeps its ambient widget alive while the
    user is on some other tab, and un-pausing those would spend frames
    on pixels nobody can see — which is the one thing this animation is
    not allowed to do. Their own ``showEvent`` restarts them when the
    tab comes back.

    Never raises. A widget whose C++ half is already gone, or an ambient
    module that could not be imported, is a cosmetic problem — not a
    reason to fail a preferences save.
    """
    try:
        from PySide6.QtWidgets import QApplication
        from .widgets.ambient import AmbientWidget
    except Exception:
        return
    app = app or QApplication.instance()
    if app is None:
        return
    try:
        widgets = list(app.allWidgets())
    except Exception:
        return
    enabled = get_ambient_enabled()
    # Only read — and only repaint into — what is going to be shown.
    theme = get_ambient_theme() if enabled else None
    palette = get_ambient_palette() if enabled else None
    blur = get_ambient_blur() if enabled else None
    speed = get_ambient_speed() if enabled else None
    size = get_ambient_size() if enabled else None
    for widget in widgets:
        try:
            if not isinstance(widget, AmbientWidget):
                continue
            if not enabled:
                # Stop first, then hide: no last frame on the way out.
                widget.set_animating(False)
                widget.setVisible(False)
                continue
            widget.setVisible(True)
            # Unconditionally True, NOT `widget.isVisible()`. A module screen
            # the user is not currently looking at has an invisible backdrop,
            # so the isVisible() form latched `_animating = False` onto every
            # background tab the moment Preferences was saved — and because
            # `showEvent` honours that flag (which is what makes a pause a
            # pause), those screens never animated again for the rest of the
            # session. Un-pausing an off-screen widget costs nothing:
            # `AmbientWidget._should_run` already refuses to tick while
            # hidden, and the widget's own showEvent starts it when the tab
            # comes back — which is what this function's docstring already
            # says is supposed to happen.
            widget.set_animating(True)
            # Everything cosmetic goes *after* the run state, and in its own
            # guard. The bug described above cost a session's worth of
            # animation because one step of this loop threw and skipped the
            # rest; a backdrop that comes back in last week's palette is a
            # smaller failure than a backdrop that never moves again.
            try:
                widget.set_theme(theme)
                widget.set_palette(palette)
                # After the theme: a theme change rebuilds the engine, and
                # these three ride on it.
                widget.set_blur(blur)
                widget.set_speed(speed)
                widget.set_size_scale(size)
            except Exception:
                LOG.debug("could not restyle an ambient backdrop",
                          exc_info=True)
        except Exception:
            continue


# ---------------------------------------------------------------------------
# Setting animations in tooltips
# ---------------------------------------------------------------------------

#: Shown by default. The animation is the fastest way to understand what a
#: geometric setting like ``cell_diameter`` or ``merge_pathogens`` actually
#: does, and it costs nothing until a tooltip is on screen.
DEFAULT_SETTING_ANIMATIONS = True


def get_setting_animations_enabled() -> bool:
    """Whether setting tooltips play their animation beside the text.

    Default ``True``. When ``False`` the tooltip is text only: no GIF is
    decoded, no frames are cached and no timer runs. The purple animation
    dot beside the setting keeps working either way — that is a click the
    user asked for, whereas this preference is about what a *hover* does.

    Read on every tooltip, not once at startup:
    :class:`spacr.qt.widgets.hover_tooltip.HoverTooltip` is a process-wide
    singleton that outlives the Preferences dialog, so caching this would
    keep animating until the app was restarted.
    """
    return _as_bool(_settings().value(_KEY_SETTING_ANIMATIONS,
                                      DEFAULT_SETTING_ANIMATIONS),
                    DEFAULT_SETTING_ANIMATIONS)


def set_setting_animations_enabled(on: bool) -> None:
    """Turn the animation inside setting tooltips on or off.

    Flushed immediately so the very next hover honours it — see
    :func:`get_setting_animations_enabled` for why nothing caches it.
    """
    settings = _settings()
    settings.setValue(_KEY_SETTING_ANIMATIONS, bool(on))
    settings.sync()


# ---------------------------------------------------------------------------
# Font scale
# ---------------------------------------------------------------------------

def get_font_scale() -> float:
    """Return the saved UI font scale, clamped to supported bounds."""
    try:
        raw = float(_settings().value(_KEY_FONT_SCALE,
                                        DEFAULT_FONT_SCALE))
    except (TypeError, ValueError):
        raw = DEFAULT_FONT_SCALE
    return max(FONT_SCALE_MIN, min(FONT_SCALE_MAX, raw))


def set_font_scale(scale: float) -> None:
    """Persist a UI font scale after clamping it to supported bounds."""
    scale = float(scale)
    scale = max(FONT_SCALE_MIN, min(FONT_SCALE_MAX, scale))
    _settings().setValue(_KEY_FONT_SCALE, scale)


def scaled_px(base_px: int) -> int:
    """Return ``base_px`` scaled by the current user font scale.

    Widget sizes set from Python (``setMinimumWidth`` etc.) don't grow
    when the stylesheet's font size grows, so any control tuned to
    match a text width goes wrong at large font scales. Route those
    calls through this helper so they track the preference.

    Rounds to the nearest int; caps to at least 1 px so a very small
    scale doesn't collapse things to zero.
    """
    return max(1, int(round(base_px * get_font_scale())))


# ---------------------------------------------------------------------------
# The left dock — revealed, pinned, or gone
# ---------------------------------------------------------------------------

#: ``"auto"``    the 6 px hot strip on the left edge reveals the app list
#:               on dwell and hides it again.
#: ``"locked"``  the app list is a real column in the window layout: it
#:               never slides, never covers the page, and never has to be
#:               summoned. This is the default; users with narrower screens
#:               can switch to hover reveal or hide it completely.
#: ``"hidden"``  no strip, no reveal, no column. Apps stay reachable from
#:               the spaCR menu, Ctrl+1..9 and the command palette — a
#:               dock you cannot summon must not be a dead end.
VALID_DOCK_MODES = ("auto", "locked", "hidden")
DEFAULT_DOCK_MODE = "locked"


def get_dock_mode() -> str:
    """How the left app dock behaves — one of :data:`VALID_DOCK_MODES`."""
    raw = str(_settings().value(_KEY_DOCK_MODE, DEFAULT_DOCK_MODE))
    return raw if raw in VALID_DOCK_MODES else DEFAULT_DOCK_MODE


def set_dock_mode(mode: str) -> None:
    """Persist a valid left-navigation dock mode."""
    if mode not in VALID_DOCK_MODES:
        raise ValueError(f"unknown dock mode {mode!r}. "
                          f"Choose from {VALID_DOCK_MODES}.")
    _settings().setValue(_KEY_DOCK_MODE, mode)


# ---------------------------------------------------------------------------
# Page opacity — how solid shared page and module surfaces are
# ---------------------------------------------------------------------------

#: 0 = the box is not painted at all, 100 = solid. Stored as a percent
#: because that is what the slider shows and what a user reading the INI
#: would expect to find.
#:
#: **This is a request, not the final alpha.** It is clamped up to
#: :func:`spacr.qt.theme.pane_alpha_floor` before anything is painted, so
#: dragging it to zero on the Space theme thins the panel to the point
#: where the tile names still clear WCAG AA over the brightest star the
#: sky can put behind them, and no further. See
#: :func:`spacr.qt.theme.pane_alpha` for why the solver's *upper* bound
#: is deliberately not applied to it.
DEFAULT_PANE_OPACITY_PCT = 60


def get_pane_opacity() -> float:
    """The user's requested page-panel opacity, 0.0-1.0.

    Un-clamped: the floor belongs to the theme, which is the only thing
    that knows what is behind the panel. Callers want
    :func:`spacr.qt.theme.pane_alpha`, which applies it.
    """
    try:
        raw = int(_settings().value(_KEY_PANE_OPACITY,
                                    DEFAULT_PANE_OPACITY_PCT))
    except (TypeError, ValueError):
        raw = DEFAULT_PANE_OPACITY_PCT
    return max(0, min(100, raw)) / 100.0


def set_pane_opacity(fraction: float) -> None:
    """Store the requested opacity. Accepts 0.0-1.0; clamped, then rounded."""
    try:
        value = float(fraction)
    except (TypeError, ValueError):
        value = DEFAULT_PANE_OPACITY_PCT / 100.0
    _settings().setValue(_KEY_PANE_OPACITY,
                         int(round(max(0.0, min(1.0, value)) * 100)))


def effective_pane_alpha() -> float:
    """The opacity a user-controlled page surface is painted at.

    The user's request put through :func:`spacr.qt.theme.pane_alpha`.
    One call so the Home page and any test asking "what will it look
    like" get the same number.
    """
    from .theme import pane_alpha
    return pane_alpha(resolve_effective_theme(), get_pane_opacity())


# ---------------------------------------------------------------------------
# The field fade — the exception to the setting above
# ---------------------------------------------------------------------------

#: On by default. The fade is what makes a form of value boxes read as
#: values rather than as a wall of boxes, and it is the shipped look;
#: the preference exists because an effect that touches every input in
#: the app has to be refusable.
DEFAULT_FIELD_FADE = True


def get_field_fade_enabled() -> bool:
    """Whether input fields dissolve towards their right edge.

    ``True`` (the default) means every line edit, combo box and spin box
    paints its container and outline through
    :func:`spacr.qt.theme.field_fade_alpha` — solid where the value
    starts, gone at the right edge — and is **exempt** from
    ``pane_opacity``. The text inside is never faded.

    ``False`` restores the flat opaque input styling exactly:
    :func:`spacr.qt.widgets.field_fade.field_fade_qss` emits nothing, so
    the built-in rules in :func:`spacr.qt.theme.stylesheet` are the only
    thing that styles a field, and the paint hook returns immediately.
    """
    return _as_bool(_settings().value(_KEY_FIELD_FADE, DEFAULT_FIELD_FADE),
                    DEFAULT_FIELD_FADE)


def set_field_fade_enabled(on: bool) -> None:
    """Turn the field fade on or off.

    Flushed immediately and the paint hook's cache dropped, so the very
    next repaint honours it. Re-applying the stylesheet
    (:func:`apply_preferences_to_app`) is what makes it land on fields
    that are already on screen.
    """
    settings = _settings()
    settings.setValue(_KEY_FIELD_FADE, bool(on))
    settings.sync()
    try:
        from .widgets.field_fade import invalidate_field_fade
        invalidate_field_fade()
    except Exception:
        # Headless, or PySide6 not importable: the cache cannot be stale
        # if it was never built.
        pass


# ---------------------------------------------------------------------------
# Colour-blind mode
# ---------------------------------------------------------------------------

def get_color_blind_mode() -> str:
    """Return the active colour-vision mode, falling back to ``off``."""
    raw = str(_settings().value(_KEY_CB_MODE, DEFAULT_CB_MODE))
    return raw if raw in VALID_CB_MODES else DEFAULT_CB_MODE


def set_color_blind_mode(mode: str) -> None:
    """Persist a supported colour-vision mode."""
    if mode not in VALID_CB_MODES:
        raise ValueError(f"unknown CB mode {mode!r}. "
                          f"Choose from {VALID_CB_MODES}.")
    _settings().setValue(_KEY_CB_MODE, mode)


def color_blind_categorical_palette() -> list:
    """Return a list of hex colours safe for the active CB mode.

    Uses the Okabe-Ito palette for all three deficiencies (empirically
    the most robust choice for categorical distinctions across
    common types of colour-blindness).
    """
    if get_color_blind_mode() == "off":
        # Default spaCR categorical palette — matches theme accents
        return ["#4A9EFF", "#3fb950", "#f0883e", "#a78bfa",
                "#f85149", "#e879f9", "#22d3ee", "#facc15"]
    # Okabe-Ito — see https://jfly.uni-koeln.de/color/
    return ["#0072B2", "#E69F00", "#009E73", "#F0E442",
            "#56B4E9", "#D55E00", "#CC79A7", "#000000"]


def get_verbose_logging() -> bool:
    """Return True when the user has opted into the verbose diagnostic
    logger. Toggled via the Preferences dialog; consulted at startup
    by :func:`apply_preferences_to_app`."""
    raw = _settings().value(_KEY_VERBOSE_LOG, False)
    if isinstance(raw, str):
        return raw.lower() in ("true", "1", "yes", "on")
    return bool(raw)


def set_verbose_logging(on: bool) -> None:
    """Persist whether package-wide diagnostic tracing is enabled."""
    _settings().setValue(_KEY_VERBOSE_LOG, bool(on))


# ---------------------------------------------------------------------------
# Database Browser — editing is opt-in
# ---------------------------------------------------------------------------

#: The Database Browser opens ``measurements.db`` read-only. Editing is a
#: separate, deliberate opt-in because an UPDATE against a measurements
#: database is unrecoverable — there is no undo and no backup.
DEFAULT_DB_BROWSER_EDITABLE = False


def _as_bool(raw, default: bool) -> bool:
    """Coerce a QSettings value to bool.

    The INI backend hands strings back ("true"), the native backends hand
    real bools back, and a hand-edited file can hold anything at all —
    which must fall back to ``default`` rather than turn editing on.
    """
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    text = str(raw).strip().lower()
    if text in ("true", "1", "yes", "on"):
        return True
    if text in ("false", "0", "no", "off", ""):
        return False
    return default


def get_db_browser_editable() -> bool:
    """True when the user has allowed the Database Browser to write.

    Default ``False``: the browser opens every database with
    ``mode=ro``. Turning this on only *permits* edit mode — the user
    still has to arm it per session, per database, in the browser
    itself.
    """
    return _as_bool(_settings().value(_KEY_DB_EDIT, DEFAULT_DB_BROWSER_EDITABLE),
                    DEFAULT_DB_BROWSER_EDITABLE)


def set_db_browser_editable(on: bool) -> None:
    """Allow (or forbid) edit mode in the Database Browser.

    Flushed immediately. QSettings writes back lazily, and the Database
    Browser re-reads this key on every UI refresh — a stale read right
    after the user ticked the box would tell them editing is still off.
    One tiny INI write is worth not having to explain that.
    """
    settings = _settings()
    settings.setValue(_KEY_DB_EDIT, bool(on))
    settings.sync()


# ---------------------------------------------------------------------------
# Module visibility — whether unfinished modules and settings are shown
# ---------------------------------------------------------------------------

DEFAULT_SHOW_ALPHA = True
DEFAULT_SHOW_BETA = True


def get_show_alpha() -> bool:
    """Whether Alpha modules and settings are visible (default: ``True``)."""
    return _as_bool(_settings().value(_KEY_SHOW_ALPHA, DEFAULT_SHOW_ALPHA),
                    DEFAULT_SHOW_ALPHA)


def set_show_alpha(on: bool) -> None:
    """Show or hide modules and settings classified as Alpha."""
    _settings().setValue(_KEY_SHOW_ALPHA, bool(on))


def get_show_beta() -> bool:
    """Whether Beta modules and settings are visible (default: ``True``)."""
    return _as_bool(_settings().value(_KEY_SHOW_BETA, DEFAULT_SHOW_BETA),
                    DEFAULT_SHOW_BETA)


def set_show_beta(on: bool) -> None:
    """Show or hide modules and settings classified as Beta."""
    _settings().setValue(_KEY_SHOW_BETA, bool(on))


def maturity_is_visible(stage: str) -> bool:
    """Return whether a maturity stage should be present in the UI.

    Unknown stages are treated as stable. Stable features cannot be hidden;
    the two preferences are deliberately scoped to unfinished features.
    """
    normalized = str(stage or "stable").strip().lower()
    if normalized == "alpha":
        return get_show_alpha()
    if normalized == "beta":
        return get_show_beta()
    return True


def color_blind_continuous_cmap() -> str:
    """Return a matplotlib colormap name safe for the active CB mode.

    * Off → the current default (``"viridis"`` is already CB-safe but
      keeping the app's default until the user asks otherwise).
    * Any CB mode → ``"cividis"`` (viridis's cousin, tuned for
      protanopia + deuteranopia + tritanopia).
    """
    return "cividis" if get_color_blind_mode() != "off" else "viridis"


# ---------------------------------------------------------------------------
# Wire prefs into the running QApplication
# ---------------------------------------------------------------------------

def apply_preferences_to_app(app=None) -> None:
    """Re-apply language, theme and font scale to ``QApplication``.

    Called at startup from :func:`spacr.qt.app.launch`, and again
    whenever the user changes a preference (via :class:`PreferencesDialog`
    ``accepted`` signal).

    :param app: optional QApplication. Falls back to
        ``QApplication.instance()``.
    """
    from PySide6.QtWidgets import QApplication
    from .theme import apply_qpalette, stylesheet

    app = app or QApplication.instance()
    if app is None:
        return

    app.setProperty("spacrLanguage", get_language())

    theme = resolve_effective_theme()
    scale = get_font_scale()

    # Only the image themes want a picture, and only they pay for
    # producing one. Everything here degrades to None on any failure,
    # and the stylesheet renders a gradient in that case.
    #
    # This is also the ONLY call site that can decode a master. It runs
    # at startup and on a preferences save — never from a resize, never
    # from a paint. See :func:`spacr.qt.imagery.decode_count`.
    background = theme_background_path(theme)

    apply_qpalette(app, theme=theme)

    # Fields before the stylesheet, not after: importing the module is what
    # registers its QSS block, and dropping the cached preference is what
    # lets that block agree with the painter about whether the effect is on.
    # Do it the other way round and the first save after a toggle emits the
    # previous state's stylesheet.
    try:
        from .widgets.field_fade import (install_field_fade,
                                         invalidate_field_fade)
        invalidate_field_fade()
        install_field_fade(app)
    except Exception:
        LOG.exception("Could not install the field fade")

    app.setStyleSheet(stylesheet(
        theme=theme, font_scale=scale, background=background,
        surface_opacity=get_pane_opacity()))

    # A field whose QSS did not change still has to redraw: turning the
    # effect off while its block was already empty changes only what the
    # paint hook does.
    try:
        from .widgets.field_fade import repaint_fields
        repaint_fields(app)
    except Exception:
        pass
    # Run/Propagate and Stop/Close-style buttons are tagged centrally,
    # including QDialogButtonBox buttons created after startup.
    from .button_roles import install_button_roles
    install_button_roles(app)

    # The animated background follows the same rule as the theme: it is
    # re-applied here, so toggling it (or switching palette) lands on the
    # screens that are already open instead of at the next launch.
    apply_ambient_preferences(app)

    # The console and AI chat paint their own entries with an explicit point
    # size, so the stylesheet's font scale never reaches them. Push Zoom into
    # every open one here, or changing it would only affect consoles opened
    # afterwards.
    try:
        from .widgets.console_panel import ConsolePanel
        for widget in app.allWidgets():
            if isinstance(widget, ConsolePanel):
                widget.apply_zoom()
    except Exception:
        pass

    # Apply the verbose-logger preference too — cheap to re-apply, and
    # this is the one place that runs on every prefs save. Also
    # attaches the rotating file handler if it isn't already, so every
    # spaCR launch drops a trail into ~/.spacr/logs/ regardless of
    # whether the user turned verbose logging on.
    try:
        from .verbose_logger import (
            apply_verbose_logging, _ensure_file_handler,
        )
        _ensure_file_handler()
        apply_verbose_logging(get_verbose_logging())
    except Exception:
        # Logger module is optional at import time — never let its
        # absence prevent the app from theming itself.
        pass


# ---------------------------------------------------------------------------
# Preferences dialog
# ---------------------------------------------------------------------------

class PreferencesDialog:
    """Wrapper that builds the modal Preferences dialog on demand.

    Kept as a factory (not a real class subclass) so this module can
    be imported headless without pulling in QtWidgets. The real
    :class:`QDialog` is returned by ``PreferencesDialog(parent)``.
    """

    def __new__(cls, parent=None):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QComboBox, QDialog, QDialogButtonBox, QFormLayout,
            QLabel, QSlider, QVBoxLayout,
        )
        from .i18n import language_choices, tr
        from .widgets.toggle import Toggle

        dlg = QDialog(parent)
        dlg.setWindowTitle(tr("spaCR — Preferences"))
        dlg.setMinimumWidth(420)
        outer = QVBoxLayout(dlg)

        form = QFormLayout()

        # Language is first so it remains discoverable even on a small screen.
        language_combo = QComboBox()
        language_combo.setObjectName("LanguagePreference")
        for label, key in language_choices():
            language_combo.addItem(label, key)
        current_language = get_language()
        for i in range(language_combo.count()):
            if language_combo.itemData(i) == current_language:
                language_combo.setCurrentIndex(i)
                break
        language_combo.setToolTip(
            "Choose the language used by spaCR navigation, Preferences, "
            "common actions and settings terminology. Untranslated "
            "scientific terms safely remain in English."
        )
        form.addRow(tr("Language"), language_combo)

        # Theme
        theme_combo = QComboBox()
        for label, key in theme_choices():
            theme_combo.addItem(tr(label), key)
        current = get_theme_choice()
        for i in range(theme_combo.count()):
            if theme_combo.itemData(i) == current:
                theme_combo.setCurrentIndex(i); break
        form.addRow(tr("Theme"), theme_combo)

        # Animated background — the drifting shapes behind every module
        # page. Sits directly under Theme because it is the same kind of
        # decision, and the off switch comes first: a user who finds the
        # motion distracting must not have to read past two dropdowns to
        # find the way out. Applied on Save, without a restart.
        from .widgets.ambient import (
            AMBIENT_THEMES, palette_label, palettes_for, theme_label,
        )
        try:
            # Purely descriptive, and resolved through the same import as
            # everything else above so the dialog cannot end up reading
            # two different ambient modules in one function.
            from .widgets.ambient import theme_note
        except ImportError:
            theme_note = None

        ambient_check = Toggle(tr("Animate module backgrounds"))
        ambient_check.setObjectName("AmbientEnabled")
        ambient_check.setToolTip(
            "Slow, out-of-focus shapes drifting behind the module pages. "
            "They stop completely whenever their page is not on screen, "
            "so they cost nothing while a pipeline runs on another tab. "
            "Clear this to have no animation anywhere; the Sequencing "
            "page keeps its own DNA rain either way."
        )
        ambient_check.setChecked(get_ambient_enabled())
        form.addRow(tr("Animated background"), ambient_check)

        ambient_theme_combo = QComboBox()
        ambient_theme_combo.setObjectName("AmbientTheme")
        for key in AMBIENT_THEMES:
            ambient_theme_combo.addItem(tr(theme_label(key)), key)
        current_ambient = get_ambient_theme()
        for i in range(ambient_theme_combo.count()):
            if ambient_theme_combo.itemData(i) == current_ambient:
                ambient_theme_combo.setCurrentIndex(i); break
        form.addRow(tr("Animation"), ambient_theme_combo)

        ambient_palette_combo = QComboBox()
        ambient_palette_combo.setObjectName("AmbientPalette")

        def _reload_ambient_palettes(preferred=None):
            """Refill the palette list for the selected animation.

            Palettes are per theme, so the two controls cannot be filled
            independently. The current choice is carried across when the
            new theme also offers it; otherwise that theme's default is
            selected, which is exactly what the stored keys do.
            """
            theme_key = ambient_theme_combo.currentData()
            valid = palettes_for(theme_key)
            wanted = (preferred if preferred in valid
                      else ambient_default_palette(theme_key))
            blocked = ambient_palette_combo.blockSignals(True)
            try:
                ambient_palette_combo.clear()
                for key in valid:
                    ambient_palette_combo.addItem(
                        tr(palette_label(theme_key, key)), key)
                for index in range(ambient_palette_combo.count()):
                    if ambient_palette_combo.itemData(index) == wanted:
                        ambient_palette_combo.setCurrentIndex(index); break
            finally:
                ambient_palette_combo.blockSignals(blocked)
            # Say what the selected animation actually looks like — the
            # names alone ("Ripples", "Starfield") do not tell a user
            # what they are about to put behind their work.
            ambient_theme_combo.setToolTip(
                tr(theme_note(theme_key)) if theme_note is not None else "")

        ambient_theme_combo.currentIndexChanged.connect(
            lambda _index: _reload_ambient_palettes(
                ambient_palette_combo.currentData()))
        _reload_ambient_palettes(get_ambient_palette())
        ambient_palette_combo.setToolTip(
            "Which colours the animation uses. \"spaCR\" is built from "
            "the app's own blue, magenta and green-cyan."
        )
        form.addRow(tr("Animation palette"), ambient_palette_combo)

        # The three shape-of-the-motion controls, beside the animation they
        # shape. Each is a percentage of what the chosen animation already
        # does, so 100 % is the designed look in every theme and every one of
        # them starts there. Percentages rather than pixels or seconds
        # because "40 px" means nothing to a starfield and "6 seconds" means
        # nothing to a blob.
        (blur_lo, blur_hi) = _ambient_ranges()[0][0]
        (speed_lo, speed_hi) = _ambient_ranges()[1][0]
        (size_lo, size_hi) = _ambient_ranges()[2][0]

        def _percent_row(name, label_text, low, high, current, tip):
            slider = QSlider(Qt.Horizontal)
            slider.setObjectName(name)
            slider.setRange(int(round(low * 100)), int(round(high * 100)))
            slider.setSingleStep(5)
            slider.setPageStep(25)
            slider.setTickInterval(50)
            slider.setValue(int(round(current * 100)))
            slider.setToolTip(tip)
            value = QLabel()

            def _update(v):
                # Say when it is the designed value, because "100%" alone
                # does not tell a reader that it is the one to come back to.
                value.setText(f"{v}% — as designed" if v == 100
                              else f"{v}%")

            slider.valueChanged.connect(_update)
            _update(slider.value())
            column = QVBoxLayout()
            column.setContentsMargins(0, 0, 0, 0)
            column.addWidget(slider)
            column.addWidget(value)
            form.addRow(tr(label_text), _hbox_wrap(column))
            return slider

        blur_slider = _percent_row(
            "AmbientBlur", "Animation blur",
            blur_lo, blur_hi, get_ambient_blur(),
            "How out of focus the animation is. Above 100 % it is softer "
            "and, because softening it means drawing it smaller and "
            "stretching it further, also cheaper; below 100 % it sharpens "
            "and costs more.")
        speed_slider = _percent_row(
            "AmbientSpeed", "Animation speed",
            speed_lo, speed_hi, get_ambient_speed(),
            "How fast the animation moves, against the speed each one was "
            "designed at. It applies to every kind of motion in the chosen "
            "animation at once, and changing it never makes what is already "
            "on screen jump.")
        size_slider = _percent_row(
            "AmbientSize", "Animation size",
            size_lo, size_hi, get_ambient_size(),
            "How large the moving elements are: blob width, curtain height, "
            "the spacing between ripples, star size. Scaled against each "
            "animation's own range, so one setting means the same thing in "
            "all four.")

        def _sync_ambient_enabled(on):
            """Grey out the pickers when there is nothing to paint."""
            ambient_theme_combo.setEnabled(bool(on))
            ambient_palette_combo.setEnabled(bool(on))
            blur_slider.setEnabled(bool(on))
            speed_slider.setEnabled(bool(on))
            size_slider.setEnabled(bool(on))

        ambient_check.toggled.connect(_sync_ambient_enabled)
        _sync_ambient_enabled(ambient_check.isChecked())

        setting_anim_check = Toggle(tr("Animate setting tooltips"))
        setting_anim_check.setObjectName("SettingAnimationsEnabled")
        setting_anim_check.setToolTip(
            "Hovering a setting shows a short animation of what it does, "
            "beside the explanation. Clear this for text-only tooltips; "
            "the purple dot beside each setting still opens the same "
            "animation on demand."
        )
        setting_anim_check.setChecked(get_setting_animations_enabled())
        form.addRow(tr("Setting animations"), setting_anim_check)

        # Font scale
        scale_slider = QSlider(Qt.Horizontal)
        scale_slider.setRange(int(FONT_SCALE_MIN * 100),
                                int(FONT_SCALE_MAX * 100))
        scale_slider.setSingleStep(5)
        scale_slider.setPageStep(25)
        scale_slider.setTickInterval(25)
        scale_slider.setValue(int(get_font_scale() * 100))
        scale_value = QLabel(f"{int(get_font_scale() * 100)}%")

        def _update_scale_lbl(v):
            scale_value.setText(f"{v}%")
        scale_slider.valueChanged.connect(_update_scale_lbl)

        scale_row = QVBoxLayout()
        scale_row.addWidget(scale_slider)
        scale_row.addWidget(scale_value)
        _wrap = _hbox_wrap(scale_row)
        form.addRow(tr("Font scale"), _wrap)

        # The left dock — revealed on hover, pinned open, or gone.
        dock_combo = QComboBox()
        for label, key in (
            ("Reveal on hover", "auto"),
            ("Locked open",     "locked"),
            ("Hidden",          "hidden"),
        ):
            dock_combo.addItem(tr(label), key)
        current_dock = get_dock_mode()
        for i in range(dock_combo.count()):
            if dock_combo.itemData(i) == current_dock:
                dock_combo.setCurrentIndex(i); break
        dock_combo.setToolTip(
            "Reveal on hover: the app list slides in when you rest the "
            "pointer against the left edge, and slides out again.\n"
            "Locked open: it is a permanent column instead — it never "
            "covers the page, and costs its own width.\n"
            "Hidden: no edge strip and no column. Apps stay reachable "
            "from the spaCR menu, Ctrl+1..9 and Ctrl+K."
        )
        form.addRow(tr("App dock"), dock_combo)

        # Page opacity — shared by Home and every module surface.
        opacity_slider = QSlider(Qt.Horizontal)
        opacity_slider.setRange(0, 100)
        opacity_slider.setSingleStep(5)
        opacity_slider.setPageStep(10)
        opacity_slider.setTickInterval(25)
        opacity_slider.setValue(int(round(get_pane_opacity() * 100)))
        opacity_value = QLabel()

        def _update_opacity_lbl(v):
            """Show what was asked for, and what the theme will allow.

            The floor is the whole design of this control: on Space it
            is 78 %, so "20 %" would otherwise be a number the user set
            and the app quietly ignored.
            """
            from .theme import pane_alpha
            if resolve_effective_theme() == "glass":
                opacity_value.setText(
                    f"{v}% material strength — Glass stays translucent "
                    "by design")
                return
            actual = pane_alpha(resolve_effective_theme(), v / 100.0)
            shown = int(round(actual * 100))
            opacity_value.setText(
                f"{v}%" if shown == v else
                f"{v}% — held at {shown}% so the text stays readable "
                "over the background")

        opacity_slider.valueChanged.connect(_update_opacity_lbl)
        _update_opacity_lbl(opacity_slider.value())
        opacity_value.setWordWrap(True)
        opacity_slider.setToolTip(
            "How solid cards, settings sections, consoles, previews and the "
            "rounded Home panel are. This applies in every theme, including "
            "Glass, Space, Cell, Dark and Light. In Glass this controls "
            "material strength while preserving its designed translucency; "
            "in other themes it is literal surface opacity. A surface will not go "
            "thinner than the point where its text stops clearing WCAG AA "
            "over the background."
        )
        opacity_col = QVBoxLayout()
        opacity_col.addWidget(opacity_slider)
        opacity_col.addWidget(opacity_value)
        form.addRow(tr("Page opacity"), _hbox_wrap(opacity_col))

        # The one surface Page opacity does not reach, and why it sits
        # directly under the slider: this is the exception to the row above.
        field_fade_check = Toggle(tr("Fade fields towards the right"))
        field_fade_check.setObjectName("FieldFadeEnabled")
        field_fade_check.setToolTip(
            "Input fields ignore Page opacity and dissolve instead: solid "
            "where the value starts, fully transparent at their right edge, "
            "fading faster the further right it goes. The outline fades with "
            "the box; the text inside never fades. Clear this for plain "
            "opaque fields."
        )
        field_fade_check.setChecked(get_field_fade_enabled())
        form.addRow(tr("Field fade"), field_fade_check)

        # Colour-blind mode
        cb_combo = QComboBox()
        for label, key in (
            ("Off",                     "off"),
            ("Deuteranopia (red-green)", "deuteranopia"),
            ("Protanopia (red-green)",   "protanopia"),
            ("Tritanopia (blue-yellow)", "tritanopia"),
        ):
            cb_combo.addItem(tr(label), key)
        current_cb = get_color_blind_mode()
        for i in range(cb_combo.count()):
            if cb_combo.itemData(i) == current_cb:
                cb_combo.setCurrentIndex(i); break
        form.addRow(tr("Colour-blind mode"), cb_combo)

        # Verbose logging — one toggle, wired at Save time. When on,
        # spaCR + third-party libs (cellpose, torch, PIL, matplotlib)
        # dial their loggers to DEBUG/INFO and every record echoes into
        # the active ConsolePanel. Aimed at bug reports.
        verbose_check = Toggle(tr("Enable verbose logging"))
        verbose_check.setToolTip(
            "When on, every spaCR log record — plus INFO-level chatter "
            "from cellpose, torch, PIL and matplotlib — echoes into "
            "the active app's Console. Very chatty; leave off unless "
            "you're triaging a bug."
        )
        verbose_check.setChecked(get_verbose_logging())
        form.addRow(tr("Diagnostics"), verbose_check)

        # Database Browser — off by default. The browser opens
        # measurements.db with mode=ro; this is the only switch that lets
        # it open a read-write connection at all, and even then the user
        # has to arm edit mode per session and confirm it.
        db_edit_check = Toggle(tr("Allow editing in the Database Browser"))
        db_edit_check.setToolTip(
            "Off by default. The Database Browser opens measurements.db "
            "read-only (mode=ro). With this on you can still only edit "
            "after arming 'Edit mode' for a database you chose yourself, "
            "and every change is one UPDATE scoped to one row. There is "
            "no undo — spaCR writes straight into your measurements file."
        )
        db_edit_check.setChecked(get_db_browser_editable())
        form.addRow(tr("Database Browser"), db_edit_check)

        # Module visibility. Both are opt-out: existing users and fresh
        # installs continue to see every feature until they choose a quieter,
        # stable-only interface.
        alpha_check = Toggle(tr("Show Alpha modules and settings"))
        alpha_check.setObjectName("ShowAlphaFeatures")
        alpha_check.setToolTip(
            "Hide modules and settings that are built but not yet trusted "
            "end to end. Stable and Beta features are unaffected."
        )
        alpha_check.setChecked(get_show_alpha())

        beta_check = Toggle(tr("Show Beta modules and settings"))
        beta_check.setObjectName("ShowBetaFeatures")
        beta_check.setToolTip(
            "Hide modules and settings that are in regular use but not yet "
            "signed off. Stable and Alpha features are unaffected."
        )
        beta_check.setChecked(get_show_beta())
        maturity_col = QVBoxLayout()
        maturity_col.setContentsMargins(0, 0, 0, 0)
        maturity_col.addWidget(alpha_check)
        maturity_col.addWidget(beta_check)
        form.addRow(tr("Module visibility"), _hbox_wrap(maturity_col))

        # Figures — display format (png = lighter / faster, pdf = vector +
        # editable via the figure-settings button) and the PNG resolution.
        fig_format_combo = QComboBox()
        fig_format_combo.addItem("PNG (raster, lighter)", "png")
        fig_format_combo.addItem("PDF (vector, editable)", "pdf")
        cur_fmt = get_figure_format()
        for i in range(fig_format_combo.count()):
            if fig_format_combo.itemData(i) == cur_fmt:
                fig_format_combo.setCurrentIndex(i); break
        form.addRow(tr("Figure format"), fig_format_combo)

        png_dpi_combo = QComboBox()
        for dpi in VALID_PNG_DPIS:
            png_dpi_combo.addItem(f"{dpi} dpi", dpi)
        cur_dpi = get_figure_png_dpi()
        for i in range(png_dpi_combo.count()):
            if png_dpi_combo.itemData(i) == cur_dpi:
                png_dpi_combo.setCurrentIndex(i); break
        form.addRow(tr("PNG resolution"), png_dpi_combo)

        outer.addLayout(form)

        preview = QLabel(
            "<span style='color:gray;'>Theme, font scale and the "
            "animated background apply instantly on Save. Colour-blind "
            "mode affects plot colours the next time a figure is "
            "generated.</span>"
        )
        preview.setTextFormat(Qt.RichText)
        preview.setWordWrap(True)
        outer.addWidget(preview)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        save_button = buttons.button(QDialogButtonBox.Save)
        cancel_button = buttons.button(QDialogButtonBox.Cancel)
        if save_button is not None:
            save_button.setText(tr("Save"))
        if cancel_button is not None:
            cancel_button.setText(tr("Cancel"))
        outer.addWidget(buttons)

        def _save():
            set_language(language_combo.currentData())
            set_theme_choice(theme_combo.currentData())
            set_ambient_enabled(ambient_check.isChecked())
            # Theme first: it repairs a palette the new theme cannot
            # draw, so the following call always validates against the
            # theme the user just chose.
            set_ambient_theme(ambient_theme_combo.currentData())
            palette_choice = ambient_palette_combo.currentData()
            if palette_choice is not None:
                # An animation that offers no palette leaves the combo
                # empty. The theme write above already stored a usable
                # value, so there is nothing to save here — and a
                # decorative background must never be the reason the
                # whole Preferences dialog refuses to close.
                set_ambient_palette(palette_choice)
            set_ambient_blur(blur_slider.value() / 100.0)
            set_ambient_speed(speed_slider.value() / 100.0)
            set_ambient_size(size_slider.value() / 100.0)
            set_setting_animations_enabled(setting_anim_check.isChecked())
            set_font_scale(scale_slider.value() / 100.0)
            set_dock_mode(dock_combo.currentData())
            set_pane_opacity(opacity_slider.value() / 100.0)
            set_field_fade_enabled(field_fade_check.isChecked())
            set_color_blind_mode(cb_combo.currentData())
            set_verbose_logging(verbose_check.isChecked())
            set_db_browser_editable(db_edit_check.isChecked())
            set_show_alpha(alpha_check.isChecked())
            set_show_beta(beta_check.isChecked())
            set_figure_format(fig_format_combo.currentData())
            set_figure_png_dpi(png_dpi_combo.currentData())
            apply_preferences_to_app()
            _refresh_owner_window(parent)
            dlg.accept()

        buttons.accepted.connect(_save)
        buttons.rejected.connect(dlg.reject)
        return dlg


def _refresh_owner_window(parent) -> None:
    """Ask the window that opened Preferences to rebuild itself.

    A QIcon bakes its pixmap when it is built, so re-applying the
    stylesheet leaves every existing icon in the *old* theme's ink —
    switch to the light theme and the sidebar keeps its white glyphs, on
    white. Only the dialog's own window is touched: walking
    ``QApplication.topLevelWidgets()`` instead reaches leftover windows
    whose C++ side is already being torn down, and rebuilding one of
    those segfaults rather than raising.

    Never raises: a window that cannot rebuild is a cosmetic problem,
    not a reason to fail the Save.
    """
    if parent is None:
        return
    try:
        window = parent.window()
    except Exception:
        return
    refresh = getattr(window, "refresh_theme", None)
    if callable(refresh):
        try:
            refresh()
        except Exception:
            pass


def _hbox_wrap(layout):
    from PySide6.QtWidgets import QWidget
    w = QWidget()
    w.setLayout(layout)
    return w
