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
        get_ambient_animation, set_ambient_animation,
        get_ambient_theme, set_ambient_theme,
        get_spacr_mode, set_spacr_mode, mode_label, mode_note, mode_warning,
        confirm_resource_action, run_resource_action,
        get_ambient_palette, set_ambient_palette,
        get_ambient_blur, set_ambient_blur,
        get_ambient_speed, set_ambient_speed,
        get_ambient_size, set_ambient_size,
        get_ambient_resolution, set_ambient_resolution,
        get_ambient_density, set_ambient_density,
        get_ambient_drift_direction, set_ambient_drift_direction,
        get_spinner_delay, set_spinner_delay,
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
  choice — see :func:`get_ambient_enabled`. The user-facing control is the
  ``None`` entry in the Animation list rather than a second switch: one
  row, one meaning. Choosing an animation turns it back on.
* ``spacr_mode``: ``"extra_performance"`` | ``"performance"`` |
  ``"balanced"`` (default ``"balanced"``). How hard spaCR tries to stay out
  of the machine's way — when it frees its own caches, and whether it
  overrides the visual settings. Balanced does neither. See
  :func:`set_spacr_mode` and :mod:`spacr.qt.resource_cleanup`, which owns
  what a cleanup is allowed to touch (spaCR's own memory, and nothing
  else — no other process, ever).
* ``ambient_theme`` / ``ambient_palette``: which animation, and in which
  colours. ``ambient_theme`` also holds
  :data:`spacr.qt.widgets.ambient.NO_ANIMATION` — read it with
  :func:`get_ambient_animation`, which can answer ``"none"``, or with
  :func:`get_ambient_theme`, whose answer is always something paintable.
  Validated against
  :data:`spacr.qt.widgets.ambient.AMBIENT_THEMES` and
  :func:`spacr.qt.widgets.ambient.palettes_for` respectively; palettes
  are *per theme*, so see :func:`get_ambient_palette` for how the two
  keys stay consistent with each other.
* ``ambient_speed`` / ``ambient_size`` / ``ambient_resolution`` /
  ``ambient_density``: floats, all default ``1.0``, all *multipliers* on
  what the chosen animation already does — how fast it moves, how large its
  elements are, how much detail it is drawn with and how many elements
  there are. 1.0 is the shipped animation in every theme, exactly, so a
  user who never touches them sees no change. Clamped on read and on write
  to the ranges the engines declare
  (:data:`spacr.qt.widgets.ambient.SPEED_RANGE` and friends).
* ``ambient_blur``: float, default ``0.0`` — how much the finished picture
  is softened, in units of eight screen pixels. **Its meaning changed**: it
  used to run 0.25–3.0 with 1.0 as the shipped look and sharpened the
  picture by enlarging the shading buffer, which made "sharp" and "not
  blocky" the same slider. Sharpening is now ``ambient_resolution``'s job
  and this one only softens. A value stored under the old scale is
  translated once, on read — see :func:`_migrate_ambient_motion`.
* ``ambient_drift_direction``: ``"up"`` | ``"down"`` | ``"random"``
  (default ``"up"``). Which way the Starfield animation travels. A
  preference rather than three entries in the animation menu; see
  :data:`spacr.qt.widgets.ambient.DRIFT_DIRECTIONS` for why.
* ``spinner_delay``: float seconds, default ``2.0``. How long background
  work has to run before the activity spinner appears at all — see
  :func:`get_spinner_delay`.
* ``setting_animations``: bool, default ``False``. Whether a setting's
  hover tooltip plays its animation WITHOUT being asked. Off — the
  default — leaves every hover text only until the reader presses the
  **Animation** word in that tooltip's footer, which speaks for that one
  setting; see :func:`get_setting_animations_enabled`.
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
# Stored as level names ("INFO,WARNING,ERROR") rather than numbers: QSettings
# round-trips strings predictably across platforms, and a settings file a
# human might open says what it means.
_KEY_LOG_FILE_LEVELS = "prefs/log_file_levels"
_KEY_LOG_CONSOLE_LEVELS = "prefs/log_console_levels"
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
_KEY_AMBIENT_RESOLUTION = "prefs/ambient_resolution"
_KEY_AMBIENT_DENSITY = "prefs/ambient_density"
_KEY_AMBIENT_DRIFT_DIR = "prefs/ambient_drift_direction"
#: Which generation of the motion keys the store was last written by. Only
#: ``ambient_blur`` has ever changed meaning, and this is how a value written
#: under the old one is recognised — see :func:`_migrate_ambient_motion`.
_KEY_AMBIENT_SCALE   = "prefs/ambient_motion_scale"
AMBIENT_MOTION_SCALE = 2
_KEY_SPINNER_DELAY   = "prefs/spinner_delay"
_KEY_SETTING_ANIMATIONS = "prefs/setting_animations"
_KEY_SPACR_MODE = "prefs/spacr_mode"
#: Where the visual settings Extra Performance overrode are kept, so
#: leaving that mode gives the user back exactly what they had.
_KEY_MODE_VISUAL_STASH = "prefs/mode_visual_stash"

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
    """Return the saved figure format, falling back to ``pdf``.

    Read by :func:`spacr.qt.widgets.figure_queue.render_figure_to_png`, which
    is the single consumer. ``pdf`` makes it write a vector page beside the
    display raster; the queue then rasterises that page for a crisper view and
    the user has a file that opens as editable art. Scope is worth stating,
    since the name suggests otherwise: this is the format of the figures spaCR
    renders **for its own Figures panel**. Figures a pipeline writes into a
    results directory are saved by ``savefig`` calls in :mod:`spacr.plot`,
    :mod:`spacr.submodules`, :mod:`spacr.ml` and friends, each of which
    hard-codes its own format and never reads this preference.
    """
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
    """Return the saved PNG resolution, or the 300-DPI default.

    Two consumers, and they treat it differently on purpose.
    :func:`spacr.qt.widgets.figure_queue.render_figure_to_png` clamps it for
    the on-screen raster — a 16x12" figure at 300 DPI is a 4800 px PNG that
    costs more to decode than any screen can show — so a large figure is
    displayed at a lower DPI than the one chosen here.
    :func:`spacr.qt.widgets.figure_queue._export_vector_pdf` uses the value
    unclamped, because the PDF is a file rather than a screenful and its
    embedded rasters really do need the resolution the user asked for.
    """
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

    **Two keys answer this one question, and both are honoured.** The
    Animation preference gained a ``None`` entry (see
    :data:`spacr.qt.widgets.ambient.NO_ANIMATION`), and "no animation" has
    to mean *nothing is constructed* rather than "an engine that paints an
    empty frame sixty times a second". The three install sites all read this
    function before they build anything, so answering ``False`` for None
    here is what makes the guarantee true everywhere at once, without a
    second condition in three other modules that could drift apart.

    The separate on/off key stays because it is the programmatic switch —
    :mod:`spacr.qt.resource_cleanup` uses it, and so does any caller that
    wants the animation back exactly as the user had it.
    """
    if _raw_ambient_animation() == _no_animation_key():
        return False
    return _as_bool(_settings().value(_KEY_AMBIENT_ENABLED,
                                      DEFAULT_AMBIENT_ENABLED),
                    DEFAULT_AMBIENT_ENABLED)


def _no_animation_key() -> str:
    """``NO_ANIMATION``, defended: this module is imported headless."""
    try:
        from .widgets.ambient import NO_ANIMATION
        return NO_ANIMATION
    except Exception:
        return "none"


def _animation_choices() -> tuple:
    """Everything the Animation preference may hold, in menu order.

    Falls back to "none plus whatever themes exist" rather than to a bare
    default, so a stored animation stays readable against an ambient module
    that predates this list — including the test doubles that stand in for
    it.
    """
    try:
        from .widgets.ambient import ANIMATION_CHOICES
        return tuple(ANIMATION_CHOICES)
    except Exception:
        pass
    try:
        from .widgets.ambient import AMBIENT_THEMES
        return (_no_animation_key(),) + tuple(AMBIENT_THEMES)
    except Exception:
        return (_no_animation_key(),)


def _raw_ambient_animation() -> str:
    """The stored animation choice, validated, ``None`` included.

    :func:`get_ambient_theme` cannot do this job: it promises a *paintable*
    theme, and half the callers hand what it returns straight to
    ``make_engine``.
    """
    try:
        from .widgets.ambient import DEFAULT_THEME as _default
    except Exception:
        _default = "blobs"
    raw = str(_settings().value(_KEY_AMBIENT_THEME, _default))
    return raw if raw in _animation_choices() else _default


def get_ambient_animation() -> str:
    """Which animation the user chose, or :data:`NO_ANIMATION` for none.

    The value the Preferences dropdown shows. Use :func:`get_ambient_theme`
    when you are about to paint something — it never returns ``"none"``.
    """
    return _raw_ambient_animation()


def set_ambient_animation(name: str) -> None:
    """Persist an entry of ``ANIMATION_CHOICES``, ``"none"`` included.

    Choosing an animation turns the backdrop on, and choosing None turns it
    off, so the dropdown is the whole control: a user who picks Blobs after
    something switched the backdrop off gets Blobs, not silence.

    Picking None does **not** disturb the stored theme's palette, so
    switching back later restores exactly the animation that was there.
    """
    choices = _animation_choices()
    if name not in choices:
        raise ValueError(f"unknown animation {name!r}. Choose from {choices}.")
    if name == _no_animation_key():
        settings = _settings()
        settings.setValue(_KEY_AMBIENT_THEME, name)
        settings.setValue(_KEY_AMBIENT_ENABLED, False)
        settings.sync()
        return
    set_ambient_theme(name)
    set_ambient_enabled(True)


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
    # "none" lands on the default here rather than propagating: this
    # getter's whole contract is that its answer can be painted, and the
    # callers that must not paint at all are gated on
    # `get_ambient_enabled()`, which is already False in that case.
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
    try:
        valid = palettes_for(theme)
    except Exception:
        # "Never raises for an unknown theme" was a claim this function did
        # not keep: the real `palettes_for` raises ValueError on a name it
        # has no engine for, which "none" now is, and the exception escaped
        # a Qt slot in the middle of refilling the palette picker.
        return DEFAULT_PALETTE
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
    """``(blur, speed, size, resolution, density)`` ranges and defaults, from
    the widget module.

    Imported lazily and defended, like every other ambient read here: this
    module is imported headless (no QtGui) in places, and a decorative
    setting is never a reason to fail.
    """
    try:
        from .widgets.ambient import (BLUR_RANGE, DEFAULT_BLUR,
                                      DEFAULT_DENSITY, DEFAULT_RESOLUTION,
                                      DEFAULT_SIZE, DEFAULT_SPEED,
                                      DENSITY_RANGE, RESOLUTION_RANGE,
                                      SIZE_RANGE, SPEED_RANGE)
        return ((BLUR_RANGE, DEFAULT_BLUR), (SPEED_RANGE, DEFAULT_SPEED),
                (SIZE_RANGE, DEFAULT_SIZE),
                (RESOLUTION_RANGE, DEFAULT_RESOLUTION),
                (DENSITY_RANGE, DEFAULT_DENSITY))
    except Exception:
        return (((0.0, 3.0), 0.0), ((0.1, 4.0), 1.0), ((0.25, 2.5), 1.0),
                ((0.25, 2.0), 1.0), ((0.25, 3.0), 1.0))


def _migrate_ambient_motion() -> None:
    """Bring a store written under the old blur scale up to the current one.

    ``ambient_blur`` used to be a *buffer resolution* divisor: 0.25 meant a
    four-times-larger buffer (sharp, dear), 1.0 the shipped one, 3.0 a
    third of it (soft, cheap). One slider therefore answered two questions,
    and the sharp end of it is now a separate ``ambient_resolution``. A
    stored value is translated rather than reinterpreted, because
    reinterpreting it would silently invert what half the range meant:

        resolution <- 1 / old        (0.25 asked for four times the pixels)
        blur       <- max(0, old-1)  (only the soft half was ever a blur)

    Recognised by the absence of :data:`AMBIENT_MOTION_SCALE` rather than by
    guessing from the value, because 1.0 is a legal reading on both scales.
    Runs once; writes the marker even when there was nothing to migrate, so
    a default store is not re-examined on every read.

    Never raises. A preference that cannot be migrated is a preference that
    stays at its default, which is a cosmetic loss.
    """
    settings = _settings()
    try:
        if int(settings.value(_KEY_AMBIENT_SCALE, 0) or 0) >= \
                AMBIENT_MOTION_SCALE:
            return
    except (TypeError, ValueError):
        pass
    try:
        raw = settings.value(_KEY_AMBIENT_BLUR, None)
        if raw is not None:
            old = float(raw)
            if old == old and old > 0:      # not NaN
                (res_low, res_high), _ = _ambient_ranges()[3]
                (blur_low, blur_high), _ = _ambient_ranges()[0]
                settings.setValue(
                    _KEY_AMBIENT_RESOLUTION,
                    max(res_low, min(res_high, 1.0 / old)))
                settings.setValue(
                    _KEY_AMBIENT_BLUR,
                    max(blur_low, min(blur_high, max(0.0, old - 1.0))))
        settings.setValue(_KEY_AMBIENT_SCALE, AMBIENT_MOTION_SCALE)
        settings.sync()
    except Exception:
        LOG.debug("could not migrate the ambient motion keys", exc_info=True)


def _ambient_multiplier(key: str, index: int) -> float:
    _migrate_ambient_motion()
    (low, high), default = _ambient_ranges()[index]
    try:
        value = float(_settings().value(key, default))
    except (TypeError, ValueError):
        return default
    if value != value:            # NaN — a hand-edited INI can hold one
        return default
    return max(low, min(high, value))


def _set_ambient_multiplier(key: str, index: int, value: float) -> None:
    _migrate_ambient_motion()
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
    """How much the animated background is softened. 0.0 is as designed.

    In units of eight screen pixels of area averaging, which is exactly the
    softness the buffered animations shipped with — so 1.0 asks for the old
    look at whatever detail :func:`get_ambient_resolution` is set to.

    Clamped to ``spacr.qt.widgets.ambient.BLUR_RANGE`` on read, so a value
    from a newer build or a hand-edited file cannot ask for a blur the
    engines will not paint.
    """
    return _ambient_multiplier(_KEY_AMBIENT_BLUR, 0)


def set_ambient_blur(value: float) -> None:
    """Set the softening. Out-of-range values are clamped, not refused:
    this is a slider, and there is no user error to report."""
    _set_ambient_multiplier(_KEY_AMBIENT_BLUR, 0, value)


def get_ambient_resolution() -> float:
    """How much detail the animated background is drawn with, as a
    multiplier on each animation's own shading buffer. 1.0 is as designed.

    Separate from :func:`get_ambient_blur` on purpose: this decides how much
    of the geometry is computed, blur decides how much of it is then thrown
    away, and having one control do both was the reason the aurora could
    only be soft *and* blocky. Costs quadratically — 2.0 is four times the
    pixels — which is why the range stops where it does.
    """
    return _ambient_multiplier(_KEY_AMBIENT_RESOLUTION, 3)


def set_ambient_resolution(value: float) -> None:
    """Set the detail multiplier. Clamped."""
    _set_ambient_multiplier(_KEY_AMBIENT_RESOLUTION, 3, value)


def get_ambient_density() -> float:
    """How many elements the animated background draws — blobs, curtains,
    ripple sources, stars, discs, cells — as a multiplier on each
    animation's own count. 1.0 is as designed.

    Density and resolution share one cost budget in the engines
    (:data:`spacr.qt.widgets.ambient.WORK_BUDGET`), so asking for the top of
    both ranges at once gets a trimmed density rather than a stalled frame.
    """
    return _ambient_multiplier(_KEY_AMBIENT_DENSITY, 4)


def set_ambient_density(value: float) -> None:
    """Set the element-count multiplier. Clamped."""
    _set_ambient_multiplier(_KEY_AMBIENT_DENSITY, 4, value)


def get_ambient_drift_direction() -> str:
    """Which way the Starfield animation travels.

    Validated on read against
    :data:`spacr.qt.widgets.ambient.DRIFT_DIRECTIONS`, so a value from a
    newer build or a hand-edited file falls back to the default rather than
    reaching an engine that cannot honour it.
    """
    try:
        from .widgets.ambient import (DEFAULT_DRIFT_DIRECTION,
                                      DRIFT_DIRECTIONS)
    except Exception:
        DEFAULT_DRIFT_DIRECTION, DRIFT_DIRECTIONS = "up", ("up", "down",
                                                           "random")
    raw = str(_settings().value(_KEY_AMBIENT_DRIFT_DIR,
                                DEFAULT_DRIFT_DIRECTION))
    return raw if raw in DRIFT_DIRECTIONS else DEFAULT_DRIFT_DIRECTION


def set_ambient_drift_direction(name: str) -> None:
    """Persist one of :data:`spacr.qt.widgets.ambient.DRIFT_DIRECTIONS`.

    :raises ValueError: if ``name`` is not one of them.
    """
    try:
        from .widgets.ambient import DRIFT_DIRECTIONS
    except Exception:
        DRIFT_DIRECTIONS = ("up", "down", "random")
    if name not in DRIFT_DIRECTIONS:
        raise ValueError(f"unknown starfield direction {name!r}. "
                         f"Choose from {DRIFT_DIRECTIONS}.")
    settings = _settings()
    settings.setValue(_KEY_AMBIENT_DRIFT_DIR, name)
    settings.sync()


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
    resolution = get_ambient_resolution() if enabled else None
    density = get_ambient_density() if enabled else None
    direction = get_ambient_drift_direction() if enabled else None
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
                # all of these ride on it.
                widget.set_blur(blur)
                widget.set_speed(speed)
                widget.set_size_scale(size)
                widget.set_resolution(resolution)
                widget.set_density(density)
                widget.set_direction(direction)
            except Exception:
                LOG.debug("could not restyle an ambient backdrop",
                          exc_info=True)
        except Exception:
            continue


# ---------------------------------------------------------------------------
# spaCR mode — how hard the app tries to stay out of the machine's way
# ---------------------------------------------------------------------------

#: The three modes, most aggressive first (the order the dropdown lists
#: them, so the default is at the bottom where a reader lands last).
SPACR_MODES = ("extra_performance", "performance", "balanced")

#: Balanced. A tool that starts by taking things away from you has made a
#: decision you did not ask for; the other two are opt-in and both warn.
DEFAULT_SPACR_MODE = "balanced"

MODE_LABELS = {
    "extra_performance": "Extra Performance",
    "performance": "Performance",
    "balanced": "Balanced",
}

MODE_NOTES = {
    "extra_performance": (
        "Frees as much as is safe: spaCR drops its own caches, returns its "
        "unused GPU blocks and retires its idle threads at launch AND "
        "before every module run, and every visual setting goes to its "
        "minimum — no animated backdrop, no field fade, no setting "
        "animations."),
    "performance": (
        "Frees spaCR's own caches and unused GPU blocks once, at launch, "
        "and whenever you press one of the four buttons below. Visual "
        "settings are left alone."),
    "balanced": (
        "The default. Nothing is freed at launch or before a run, and your "
        "visual settings stay exactly as you set them. The four buttons "
        "below still work whenever you press them."),
}

#: Shown when the mode is *selected*, before it is saved. Both performance
#: modes cost something, and the cost is named rather than implied.
MODE_WARNINGS = {
    "extra_performance": (
        "Extra Performance overwrites your visual settings with their "
        "minimums — the animated backdrop is switched off, field fade and "
        "setting animations are cleared. They are remembered and put back "
        "when you leave this mode.\n\n"
        "spaCR will also drop its caches before every run, so the first "
        "preview after a run redraws from disk. It never touches another "
        "program's memory or processes."),
    "performance": (
        "Performance drops spaCR's own caches once at launch, so the first "
        "screen you open redraws from disk instead of from memory. Your "
        "visual settings are not changed.\n\n"
        "It never touches another program's memory or processes."),
    "balanced": "",
}

#: The visual settings Extra Performance overrides, and the value it
#: overrides them with. Names are read back through this module's own
#: setters, so a stashed value is validated on the way home like any other.
_MODE_MINIMISED_VISUALS = (
    "ambient_animation", "ambient_resolution", "ambient_density",
    "setting_animations", "field_fade",
)


def get_spacr_mode() -> str:
    """Which resource posture spaCR is in — one of :data:`SPACR_MODES`."""
    raw = str(_settings().value(_KEY_SPACR_MODE, DEFAULT_SPACR_MODE))
    return raw if raw in SPACR_MODES else DEFAULT_SPACR_MODE


def set_spacr_mode(mode: str) -> None:
    """Persist the mode, and move the visual settings with it.

    Entering Extra Performance stashes the five visual settings it
    overrides and writes their minimums; leaving it puts the stashed values
    back. Nothing else about a mode change is retroactive — the launch
    cleanup has already happened or not happened by the time anyone can
    reach this dialog.

    :raises ValueError: on an unknown mode.
    """
    if mode not in SPACR_MODES:
        raise ValueError(f"unknown spaCR mode {mode!r}. "
                         f"Choose from {SPACR_MODES}.")
    previous = get_spacr_mode()
    settings = _settings()
    settings.setValue(_KEY_SPACR_MODE, mode)
    settings.sync()
    if mode == "extra_performance" and previous != "extra_performance":
        _stash_visuals()
        _minimise_visuals()
    elif previous == "extra_performance" and mode != "extra_performance":
        _restore_visuals()


def mode_label(mode: str) -> str:
    """The name the dropdown shows for ``mode`` — e.g. ``"Extra Performance"``.

    :param mode: One of :data:`SPACR_MODES`.
    :returns: The human-readable label from :data:`MODE_LABELS`, falling back
        to ``mode`` itself when the name is not one this module knows, so a
        stored value from a newer build still renders as text rather than as
        a blank row.
    """
    return MODE_LABELS.get(mode, str(mode))


def mode_note(mode: str) -> str:
    """What ``mode`` does, as the standing description under the dropdown.

    The note says what is freed, when it is freed, and whether the visual
    settings are touched. It is not the warning — :func:`mode_warning` carries
    what *switching* to the mode costs you, and is shown on selection.

    :param mode: One of :data:`SPACR_MODES`.
    :returns: The prose from :data:`MODE_NOTES`, or ``""`` for a mode this
        module does not know, so the caller can render it unconditionally.
    """
    return MODE_NOTES.get(mode, "")


def mode_warning(mode: str) -> str:
    """What choosing ``mode`` will cost, or ``""`` when it costs nothing."""
    return MODE_WARNINGS.get(mode, "")


def _visual_snapshot() -> dict:
    """The five settings Extra Performance overrides, as they are now."""
    return {
        "ambient_animation": get_ambient_animation(),
        "ambient_resolution": get_ambient_resolution(),
        "ambient_density": get_ambient_density(),
        "setting_animations": get_setting_animations_enabled(),
        "field_fade": get_field_fade_enabled(),
    }


def _stash_visuals() -> None:
    import json
    settings = _settings()
    settings.setValue(_KEY_MODE_VISUAL_STASH,
                      json.dumps(_visual_snapshot()))
    settings.sync()


def _minimise_visuals() -> None:
    """Every overridden visual to its cheapest setting.

    "Minimum" means the cheapest value the control offers, not zero for its
    own sake: the animation goes to None (no widget, no timer at all — see
    :func:`get_ambient_enabled`), detail and density to the bottom of the
    ranges the engines declare, and the two per-paint effects off.
    """
    ranges = _ambient_ranges()
    try:
        set_ambient_animation(_no_animation_key())
    except Exception:
        LOG.debug("could not switch the animation off", exc_info=True)
    set_ambient_resolution(ranges[3][0][0])
    set_ambient_density(ranges[4][0][0])
    set_setting_animations_enabled(False)
    set_field_fade_enabled(False)


def _restore_visuals() -> bool:
    """Put back what :func:`_stash_visuals` recorded. ``True`` if it did.

    A stash that cannot be read is discarded rather than guessed at: the
    user keeps the minimums they can see and change, which is better than
    being handed somebody's idea of a default.
    """
    import json
    settings = _settings()
    raw = settings.value(_KEY_MODE_VISUAL_STASH, "")
    settings.remove(_KEY_MODE_VISUAL_STASH)
    settings.sync()
    try:
        stashed = json.loads(str(raw)) if raw else None
    except Exception:
        stashed = None
    if not isinstance(stashed, dict):
        return False
    try:
        if "ambient_animation" in stashed:
            set_ambient_animation(str(stashed["ambient_animation"]))
        if "ambient_resolution" in stashed:
            set_ambient_resolution(float(stashed["ambient_resolution"]))
        if "ambient_density" in stashed:
            set_ambient_density(float(stashed["ambient_density"]))
        if "setting_animations" in stashed:
            set_setting_animations_enabled(bool(stashed["setting_animations"]))
        if "field_fade" in stashed:
            set_field_fade_enabled(bool(stashed["field_fade"]))
    except Exception:
        LOG.debug("could not restore the stashed visuals", exc_info=True)
        return False
    return True


# ---------------------------------------------------------------------------
# The activity spinner
# ---------------------------------------------------------------------------

#: How long work has to run before the spinner appears, in seconds.
#:
#: Two, because the spinner exists to say "this is going to take a moment",
#: and the great majority of what goes through ``make_thread`` — reading a
#: measurement table, listing a plate, loading a settings file — is done
#: inside one. A spinner that appears and vanishes inside a second is not
#: information, it is a flicker in the corner of the eye, and it trains the
#: reader to stop looking at the one place the app says it is busy.
DEFAULT_SPINNER_DELAY = 2.0

#: Nought is a real setting: it means "always show it", which is what
#: somebody debugging a hang wants. The top is chosen so a mistyped value
#: cannot hide the indicator for the length of a real job.
SPINNER_DELAY_MIN = 0.0
SPINNER_DELAY_MAX = 10.0


def get_spinner_delay() -> float:
    """How long background work must run before the activity spinner shows.

    In seconds, default :data:`DEFAULT_SPINNER_DELAY`. This is a *delay
    before showing*, not a prediction: the widget starts a single-shot timer
    when work begins and only becomes visible if the work is still running
    when it fires, so a job that finishes at 1.9 s never puts a spinner on
    screen at all. See :class:`spacr.qt.widgets.activity_spinner
    .ActivitySpinner`.

    Clamped on read: a hand-edited file must not be able to hide the
    indicator for the length of a real job.
    """
    try:
        value = float(_settings().value(_KEY_SPINNER_DELAY,
                                        DEFAULT_SPINNER_DELAY))
    except (TypeError, ValueError):
        return DEFAULT_SPINNER_DELAY
    if value != value:                     # NaN
        return DEFAULT_SPINNER_DELAY
    return max(SPINNER_DELAY_MIN, min(SPINNER_DELAY_MAX, value))


def set_spinner_delay(seconds: float) -> None:
    """Set the spinner's appearance delay, in seconds. Clamped, not
    refused."""
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        value = DEFAULT_SPINNER_DELAY
    if value != value:
        value = DEFAULT_SPINNER_DELAY
    settings = _settings()
    settings.setValue(_KEY_SPINNER_DELAY,
                      max(SPINNER_DELAY_MIN, min(SPINNER_DELAY_MAX, value)))
    settings.sync()


# ---------------------------------------------------------------------------
# Setting animations in tooltips
# ---------------------------------------------------------------------------

#: Off by default. Every hover is text only until the reader presses the
#: **Animation** word in that tooltip's footer, and pressing it speaks for
#: that setting alone: 141 settings have an animation and each one is a
#: ~73 ms decoded movie, so neither a hover that only wanted the sentence nor
#: the 140 hovers after a press should pay for one. This preference is the
#: escape hatch for the reader who never wants to be asked.
DEFAULT_SETTING_ANIMATIONS = False


def get_setting_animations_enabled() -> bool:
    """Whether setting tooltips show their animation WITHOUT being asked.

    Default ``False``: a hover is text only, no GIF is decoded, no frames are
    cached and no timer runs, and the teal **Animation** word in the footer is
    the invitation to see one — for that setting, once. Turning this on starts
    every tooltip revealed instead, and the word then folds the one in front
    of the reader away. The meaning is "stop asking me", not "allow
    animations".

    The two cannot disagree and neither needs to defer to the other, because
    they are scoped differently: a press names exactly one setting, so it can
    never stop this preference reaching the rest. See
    :meth:`spacr.qt.widgets.hover_tooltip.HoverTooltip.animations_shown`.

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


def _level_names(levels) -> str:
    return ",".join(logging.getLevelName(level) for level in sorted(levels))


def _parse_levels(raw, fallback) -> frozenset:
    """Read a stored ``"INFO,WARNING"`` string back into level numbers."""
    from ..logging_util import normalise_levels
    if raw is None or raw == "":
        return frozenset(fallback)
    if isinstance(raw, (list, tuple)):
        text = ",".join(str(item) for item in raw)
    else:
        text = str(raw)
    found = set()
    for token in text.split(","):
        token = token.strip().upper()
        if not token:
            continue
        value = logging.getLevelName(token)
        if isinstance(value, int):
            found.add(value)
    return normalise_levels(found) or frozenset(fallback)


def get_log_file_levels() -> frozenset:
    """Levels written to the log files. The master switch of the pair."""
    from ..logging_util import DEFAULT_FILE_LEVELS
    return _parse_levels(_settings().value(_KEY_LOG_FILE_LEVELS, None),
                         DEFAULT_FILE_LEVELS)


def get_log_console_levels() -> frozenset:
    """Levels echoed to the in-app console, always a subset of the files.

    Clamped on read as well as on write: the stored value can predate a
    change to the file switches made by a different code path, and a
    console line with no matching entry in the log file is exactly what
    the subset rule exists to prevent.
    """
    from ..logging_util import DEFAULT_CONSOLE_LEVELS, clamp_console_to_file
    stored = _parse_levels(_settings().value(_KEY_LOG_CONSOLE_LEVELS, None),
                           DEFAULT_CONSOLE_LEVELS)
    return clamp_console_to_file(stored, get_log_file_levels())


def set_log_levels(file_levels, console_levels) -> tuple:
    """Persist both switch sets, then apply them to the live handlers.

    :returns: ``(file_levels, console_levels)`` as actually stored, which
        is not necessarily what was asked for -- a console level whose file
        level is off is dropped rather than saved and silently ignored.
    """
    from ..logging_util import (apply_level_policy, clamp_console_to_file,
                                normalise_levels)
    files = normalise_levels(file_levels)
    console = clamp_console_to_file(console_levels, files)
    settings = _settings()
    settings.setValue(_KEY_LOG_FILE_LEVELS, _level_names(files))
    settings.setValue(_KEY_LOG_CONSOLE_LEVELS, _level_names(console))
    apply_level_policy(files, console)
    return files, console


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
        # AFTER apply_verbose_logging, which still sets a blanket threshold
        # on the attached loggers. The per-level switches are the finer
        # statement and have to be the one that lands last.
        from ..logging_util import apply_level_policy
        apply_level_policy(get_log_file_levels(), get_log_console_levels())
    except Exception:
        # Logger module is optional at import time — never let its
        # absence prevent the app from theming itself.
        pass


# ---------------------------------------------------------------------------
# The four resource buttons
# ---------------------------------------------------------------------------

def confirm_resource_action(action: str, parent=None) -> bool:
    """Ask before doing ``action``, by saying what it will do.

    The dialog states the steps in the order they will happen and what the
    action cannot do (see
    :func:`spacr.qt.resource_cleanup.confirmation_text`), and its accept
    button is labelled with the action rather than "OK". "Are you sure?"
    is not a question anybody can answer: a user cannot consent to an
    unnamed action, and a button called OK does not name one.

    Cancel is the default, so a stray Return key does nothing.

    :returns: ``True`` only if the user explicitly accepted.
    """
    from PySide6.QtWidgets import QMessageBox
    from . import resource_cleanup
    from .i18n import tr

    title = resource_cleanup.confirmation_title(action)
    box = QMessageBox(parent)
    box.setObjectName("ResourceActionConfirm")
    box.setIcon(QMessageBox.Question)
    box.setWindowTitle(tr(title))
    box.setText(tr(title))
    box.setInformativeText(tr(resource_cleanup.confirmation_text(action)))
    proceed = box.addButton(tr(title), QMessageBox.AcceptRole)
    cancel = box.addButton(tr("Cancel"), QMessageBox.RejectRole)
    box.setDefaultButton(cancel)
    box.exec()
    return box.clickedButton() is proceed


def _show_resource_result(action: str, result, parent=None) -> None:
    """Report what actually happened. Split out so a test can silence it."""
    from PySide6.QtWidgets import QMessageBox
    from . import resource_cleanup
    from .i18n import tr

    box = QMessageBox(parent)
    box.setObjectName("ResourceActionResult")
    box.setIcon(QMessageBox.Information)
    box.setWindowTitle(tr(resource_cleanup.confirmation_title(action)))
    box.setText(result.summary())
    details = getattr(result, "details", ())
    if details:
        box.setDetailedText("\n".join(details))
    box.exec()


def run_resource_action(action: str, parent=None):
    """Confirm ``action``, run it, and report the measured result.

    :returns: the :class:`~spacr.qt.resource_cleanup.Reclaim` or
        :class:`~spacr.qt.resource_cleanup.DiskReport`, or ``None`` when the
        user declined — in which case **nothing ran**. The confirmation is
        asked before any work is started, not after, which is the whole
        point of asking.
    """
    from . import resource_cleanup
    if not confirm_resource_action(action, parent):
        return None
    runner = {
        "ram": lambda: resource_cleanup.clear_ram(aggressive=True),
        "vram": resource_cleanup.clear_vram,
        "cpu": resource_cleanup.clear_cpu,
        "disk": resource_cleanup.disk_report,
    }[action]
    result = runner()
    _show_resource_result(action, result, parent)
    return result


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
            QComboBox, QDialog, QDialogButtonBox, QFormLayout, QFrame,
            QHBoxLayout, QLabel, QPushButton, QScrollArea, QSlider,
            QTabWidget, QVBoxLayout, QWidget,
        )
        from .i18n import language_choices, tr
        from .widgets.toggle import Toggle

        dlg = QDialog(parent)
        dlg.setWindowTitle(tr("spaCR — Preferences"))
        dlg.setMinimumWidth(460)
        outer = QVBoxLayout(dlg)

        # One scrollable column had grown to thirty controls, which is a
        # column nobody reads to the bottom of: Module visibility and the
        # figure format sat below five animation sliders, and the only way
        # to find out whether a setting existed was to scroll past
        # everything else. The tabs are by WHAT A SETTING IS ABOUT rather
        # than by how often it is touched — a reader looking for "how much
        # of my machine does this use" has one place to go, and so does a
        # reader looking for "why is the text so small".
        tabs = QTabWidget()
        tabs.setObjectName("PreferencesTabs")

        def _page(title: str, object_name: str) -> "QFormLayout":
            """Add a tab and return the form to fill it with.

            Each page scrolls on its own so that a small screen shortens
            the tallest tab instead of the whole dialog, and every tab is
            still reachable at any window height.
            """
            page = QWidget()
            page.setObjectName(object_name)
            column = QVBoxLayout(page)
            column.setContentsMargins(4, 8, 4, 8)
            page_form = QFormLayout()
            page_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            column.addLayout(page_form)
            column.addStretch(1)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setWidget(page)
            tabs.addTab(scroll, tr(title))
            return page_form

        # General first because Language is in it, and a reader who cannot
        # read the interface has to be able to find that one without
        # understanding any of the others.
        form = _page("General", "PreferencesTabGeneral")
        appearance = _page("Appearance", "PreferencesTabAppearance")
        performance = _page("Performance", "PreferencesTabPerformance")
        modules = _page("Modules", "PreferencesTabModules")
        figures = _page("Figures", "PreferencesTabFigures")
        logging_form = _page("Logging", "PreferencesTabLogging")

        # Two independent switches per level rather than one severity
        # threshold. A threshold cannot express "record DEBUG but not INFO",
        # which is the shape of most triage: one chatty subsystem is wanted
        # and the routine progress chatter is not.
        #
        # The file column gates the console column. Both are fed by the same
        # records, so a line shown in the console but absent from the log
        # file would be a line the user can see and then cannot produce when
        # asked for the log -- the console switch is disabled whenever its
        # file switch is off, and unticking a file switch takes its console
        # switch with it.
        log_level_toggles = {}
        _log_header = QLabel(tr(
            "Each level is written to its own file, plus a master log "
            "containing everything. The console can only show a level the "
            "log file is keeping."))
        _log_header.setWordWrap(True)
        _log_header.setObjectName("LoggingTabHelp")
        logging_form.addRow(_log_header)

        _file_levels_now = set(get_log_file_levels())
        _console_levels_now = set(get_log_console_levels())

        def _sync_console_enabled(level_value) -> None:
            """A console switch is only live while its file switch is."""
            file_toggle, console_toggle = log_level_toggles[level_value]
            allowed = file_toggle.isChecked()
            console_toggle.setEnabled(allowed)
            if not allowed and console_toggle.isChecked():
                console_toggle.setChecked(False)

        for _level in (logging.DEBUG, logging.INFO, logging.WARNING,
                       logging.ERROR, logging.CRITICAL):
            _name = logging.getLevelName(_level)
            _row = QWidget()
            _row_layout = QHBoxLayout(_row)
            _row_layout.setContentsMargins(0, 0, 0, 0)
            _row_layout.setSpacing(12)

            _file_toggle = Toggle()
            _file_toggle.setObjectName(f"LogFileLevel{_name.title()}")
            _file_toggle.setChecked(_level in _file_levels_now)
            _console_toggle = Toggle()
            _console_toggle.setObjectName(f"LogConsoleLevel{_name.title()}")
            _console_toggle.setChecked(_level in _console_levels_now)

            _row_layout.addWidget(QLabel(tr("Log file")))
            _row_layout.addWidget(_file_toggle)
            _row_layout.addSpacing(16)
            _row_layout.addWidget(QLabel(tr("Console")))
            _row_layout.addWidget(_console_toggle)
            _row_layout.addStretch(1)

            log_level_toggles[_level] = (_file_toggle, _console_toggle)
            _file_toggle.toggled.connect(
                lambda _checked, value=_level: _sync_console_enabled(value))
            logging_form.addRow(_name.title(), _row)

        for _level in log_level_toggles:
            _sync_console_enabled(_level)

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
        # page. The first entry is None, and it is an animation choice
        # rather than a separate switch: a user who finds the motion
        # distracting reads one row, not a checkbox and a dropdown that can
        # disagree with each other. Applied on Save, without a restart.
        from .widgets.ambient import palette_label, palettes_for, theme_label
        try:
            from .widgets.ambient import (ANIMATION_CHOICES, NO_ANIMATION,
                                          animation_label)
        except ImportError:
            # A build (or a test double) whose ambient module predates the
            # None entry still gets a working dialog with its six
            # animations, rather than a Preferences window that will not
            # open because of a decorative setting.
            from .widgets.ambient import AMBIENT_THEMES
            ANIMATION_CHOICES = tuple(AMBIENT_THEMES)
            NO_ANIMATION = _no_animation_key()
            animation_label = theme_label
        try:
            # Purely descriptive, and resolved through the same import as
            # everything else above so the dialog cannot end up reading
            # two different ambient modules in one function.
            from .widgets.ambient import animation_note
        except ImportError:
            try:
                from .widgets.ambient import theme_note as animation_note
            except ImportError:
                animation_note = None

        ambient_theme_combo = QComboBox()
        ambient_theme_combo.setObjectName("AmbientTheme")
        for key in ANIMATION_CHOICES:
            ambient_theme_combo.addItem(tr(animation_label(key)), key)
        current_ambient = get_ambient_animation()
        for i in range(ambient_theme_combo.count()):
            if ambient_theme_combo.itemData(i) == current_ambient:
                ambient_theme_combo.setCurrentIndex(i); break
        appearance.addRow(tr("Animation"), ambient_theme_combo)

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
            # None has no palette to offer and no engine to ask, so the
            # list is emptied rather than filled with the last theme's
            # colours — an enabled-looking picker for a backdrop that is
            # not being drawn is a control that lies.
            valid = () if theme_key == NO_ANIMATION else palettes_for(theme_key)
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
                tr(animation_note(theme_key))
                if animation_note is not None else "")

        ambient_theme_combo.currentIndexChanged.connect(
            lambda _index: _reload_ambient_palettes(
                ambient_palette_combo.currentData()))
        _reload_ambient_palettes(get_ambient_palette())
        ambient_palette_combo.setToolTip(
            "Which colours the animation uses. \"spaCR\" is built from "
            "the app's own blue, magenta and green-cyan."
        )
        appearance.addRow(tr("Animation palette"), ambient_palette_combo)

        # Which way the starfield goes. Only meaningful for that one
        # animation, so it is shown only when that animation is chosen
        # rather than sitting greyed out under five others.
        ambient_dir_combo = QComboBox()
        ambient_dir_combo.setObjectName("AmbientDriftDirection")
        try:
            from .widgets.ambient import (DRIFT_DIRECTIONS,
                                          drift_direction_label,
                                          drift_direction_note)
        except ImportError:      # pragma: no cover - ambient always imports
            DRIFT_DIRECTIONS = ()
            drift_direction_label = drift_direction_note = None
        for key in DRIFT_DIRECTIONS:
            ambient_dir_combo.addItem(tr(drift_direction_label(key)), key)
        current_dir = get_ambient_drift_direction()
        for i in range(ambient_dir_combo.count()):
            if ambient_dir_combo.itemData(i) == current_dir:
                ambient_dir_combo.setCurrentIndex(i); break
        dir_label = QLabel(tr("Starfield direction"))
        appearance.addRow(dir_label, ambient_dir_combo)

        def _sync_direction_row(*_args):
            wanted = ambient_theme_combo.currentData() == "drift"
            dir_label.setVisible(wanted)
            ambient_dir_combo.setVisible(wanted)
            key = ambient_dir_combo.currentData()
            if key is not None and drift_direction_note is not None:
                ambient_dir_combo.setToolTip(tr(drift_direction_note(key)))

        ambient_theme_combo.currentIndexChanged.connect(_sync_direction_row)
        ambient_dir_combo.currentIndexChanged.connect(_sync_direction_row)
        _sync_direction_row()

        # The shape-of-the-motion controls, beside the animation they shape.
        # Each is a percentage of what the chosen animation already does, so
        # 100 % is the designed look in every theme and every one of them
        # starts there — except blur, whose designed value is 0 %, because
        # the animation ships unsoftened and the softening is what this one
        # adds. Percentages rather than pixels or seconds because "40 px"
        # means nothing to a starfield and "6 seconds" means nothing to a
        # blob.
        (blur_lo, blur_hi) = _ambient_ranges()[0][0]
        (speed_lo, speed_hi) = _ambient_ranges()[1][0]
        (size_lo, size_hi) = _ambient_ranges()[2][0]
        (res_lo, res_hi) = _ambient_ranges()[3][0]
        (den_lo, den_hi) = _ambient_ranges()[4][0]

        def _percent_row(name, label_text, low, high, current, tip,
                         designed=1.0, target=None):
            slider = QSlider(Qt.Horizontal)
            slider.setObjectName(name)
            slider.setRange(int(round(low * 100)), int(round(high * 100)))
            slider.setSingleStep(5)
            slider.setPageStep(25)
            slider.setTickInterval(50)
            slider.setValue(int(round(current * 100)))
            slider.setToolTip(tip)
            value = QLabel()
            mark = int(round(designed * 100))

            def _update(v):
                # Say when it is the designed value, because "100%" alone
                # does not tell a reader that it is the one to come back to.
                value.setText(f"{v}% — as designed" if v == mark
                              else f"{v}%")

            slider.valueChanged.connect(_update)
            _update(slider.value())
            column = QVBoxLayout()
            column.setContentsMargins(0, 0, 0, 0)
            column.addWidget(slider)
            column.addWidget(value)
            (appearance if target is None else target).addRow(
                tr(label_text), _hbox_wrap(column))
            return slider

        resolution_slider = _percent_row(
            "AmbientResolution", "Animation detail",
            res_lo, res_hi, get_ambient_resolution(),
            "How much detail the animation is drawn with. This is the one "
            "that decides whether it looks pixelated: it sets how many "
            "pixels the picture is worked out in before it is stretched to "
            "fill the page. Costs roughly the square of what it says — "
            "200 % is four times the work — so turn it down on a machine "
            "that is busy.")
        blur_slider = _percent_row(
            "AmbientBlur", "Animation blur",
            blur_lo, blur_hi, get_ambient_blur(),
            "How out of focus the animation is, on top of whatever detail "
            "it was drawn with. 0 % leaves it as sharp as the detail "
            "setting allows; 100 % is the softness the animations used to "
            "ship with. Unlike detail, this one is nearly free — and the "
            "two together are what let the backdrop be soft without being "
            "blocky.", designed=0.0)
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
            "the spacing between ripples, star size, cell diameter. Scaled "
            "against each animation's own range, so one setting means the "
            "same thing in all of them.")
        density_slider = _percent_row(
            "AmbientDensity", "Animation density",
            den_lo, den_hi, get_ambient_density(),
            "How many things there are: blobs, aurora curtains, ripple "
            "sources, stars, bokeh discs, cells. Density and detail share "
            "one cost budget, so asking for the most of both trims the "
            "density rather than dropping frames.")

        def _sync_ambient_enabled(*_args):
            """Grey out the shaping controls when there is nothing to paint.

            Driven by the Animation row itself now that None lives in it.
            The controls stay *visible* rather than disappearing, so the
            reader can see what choosing an animation would give them back;
            they are simply not settings that mean anything while nothing
            is being drawn.
            """
            on = ambient_theme_combo.currentData() != NO_ANIMATION
            ambient_palette_combo.setEnabled(on)
            ambient_dir_combo.setEnabled(on)
            resolution_slider.setEnabled(on)
            blur_slider.setEnabled(on)
            speed_slider.setEnabled(on)
            size_slider.setEnabled(on)
            density_slider.setEnabled(on)

        ambient_theme_combo.currentIndexChanged.connect(_sync_ambient_enabled)
        _sync_ambient_enabled()

        setting_anim_check = Toggle(tr("Animate setting tooltips"))
        setting_anim_check.setObjectName("SettingAnimationsEnabled")
        setting_anim_check.setToolTip(
            "Hovering a setting shows a short animation of what it does, "
            "beside the explanation, without being asked. Cleared — the "
            "default — every tooltip is text only until you press the "
            "Animation word in its footer, and pressing it shows that one "
            "setting's animation only."
        )
        setting_anim_check.setChecked(get_setting_animations_enabled())
        appearance.addRow(tr("Setting animations"), setting_anim_check)

        # How long work has to run before the busy indicator appears.
        # Seconds, not a percentage: this one is a real duration and the
        # reader is entitled to see it as one.
        spinner_slider = QSlider(Qt.Horizontal)
        spinner_slider.setObjectName("SpinnerDelay")
        spinner_slider.setRange(int(SPINNER_DELAY_MIN * 10),
                                int(SPINNER_DELAY_MAX * 10))
        spinner_slider.setSingleStep(1)
        spinner_slider.setPageStep(5)
        spinner_slider.setTickInterval(10)
        spinner_slider.setValue(int(round(get_spinner_delay() * 10)))
        spinner_slider.setToolTip(
            "How long a background job has to run before the spinner beside "
            "Clear console appears. Short jobs never show it at all — the "
            "timer starts when the work does and the spinner only appears "
            "if the work is still going when it fires, so nothing flashes. "
            "Set it to 0 to see every job."
        )
        spinner_value = QLabel()

        def _update_spinner_lbl(v):
            spinner_value.setText(
                tr("show immediately") if v == 0 else f"{v / 10:.1f} s")

        spinner_slider.valueChanged.connect(_update_spinner_lbl)
        _update_spinner_lbl(spinner_slider.value())
        spinner_column = QVBoxLayout()
        spinner_column.setContentsMargins(0, 0, 0, 0)
        spinner_column.addWidget(spinner_slider)
        spinner_column.addWidget(spinner_value)
        appearance.addRow(tr("Show busy spinner after"),
                          _hbox_wrap(spinner_column))

        # Font scale
        scale_slider = QSlider(Qt.Horizontal)
        scale_slider.setObjectName("FontScale")
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
        # Named because (0, 100) stopped identifying it: the spinner-delay
        # slider is (0, SPINNER_DELAY_MAX * 10) = (0, 100) too, and anything
        # picking this control out of findChildren by its range silently got
        # that one instead.
        opacity_slider.setObjectName("PaneOpacity")
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
        appearance.addRow(tr("Page opacity"), _hbox_wrap(opacity_col))

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
        appearance.addRow(tr("Field fade"), field_fade_check)

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
        modules.addRow(tr("Diagnostics"), verbose_check)

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
        modules.addRow(tr("Database Browser"), db_edit_check)

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
        modules.addRow(tr("Module visibility"), _hbox_wrap(maturity_col))

        # Figures — display format (png = lighter / faster, pdf = vector +
        # editable via the figure-settings button) and the PNG resolution.
        #
        # Both tooltips say plainly what these two settings reach, because
        # their labels invite a bigger reading than the truth. They govern the
        # figures the app renders into the Figures panel; the figures a
        # pipeline saves into its own results directory are written by
        # ``savefig`` calls inside spacr.plot / spacr.submodules / spacr.ml,
        # which choose their own format and DPI and never read preferences.
        fig_format_combo = QComboBox()
        fig_format_combo.addItem("PNG (raster, lighter)", "png")
        fig_format_combo.addItem("PDF (vector, editable)", "pdf")
        fig_format_combo.setToolTip(
            "How figures are rendered into the Figures panel. PDF also writes "
            "a vector page with TrueType-embedded text — sharper on screen "
            "when zoomed, and editable in Illustrator or Inkscape. Figures a "
            "pipeline saves to its results folder keep the format that "
            "pipeline chose and are not affected."
        )
        cur_fmt = get_figure_format()
        for i in range(fig_format_combo.count()):
            if fig_format_combo.itemData(i) == cur_fmt:
                fig_format_combo.setCurrentIndex(i); break
        figures.addRow(tr("Figure format"), fig_format_combo)

        png_dpi_combo = QComboBox()
        for dpi in VALID_PNG_DPIS:
            png_dpi_combo.addItem(f"{dpi} dpi", dpi)
        png_dpi_combo.setToolTip(
            "Resolution of the raster spaCR renders for the Figures panel, "
            "and of any image embedded inside a vector PDF page. Very large "
            "figures are rendered at a lower DPI for the screen so they stay "
            "quick to draw; the PDF page is written at the full resolution "
            "chosen here."
        )
        cur_dpi = get_figure_png_dpi()
        for i in range(png_dpi_combo.count()):
            if png_dpi_combo.itemData(i) == cur_dpi:
                png_dpi_combo.setCurrentIndex(i); break
        figures.addRow(tr("PNG resolution"), png_dpi_combo)

        # -- Performance ---------------------------------------------------
        # The mode, then the four things the two performance modes press on
        # your behalf. They are in the same tab deliberately: a mode that
        # says "cleanup runs at launch" should be read next to the buttons
        # that say exactly what a cleanup is, or "cleanup" is a word the
        # user has to take on trust.
        mode_combo = QComboBox()
        mode_combo.setObjectName("SpacrMode")
        for key in SPACR_MODES:
            mode_combo.addItem(tr(mode_label(key)), key)
        current_mode = get_spacr_mode()
        for i in range(mode_combo.count()):
            if mode_combo.itemData(i) == current_mode:
                mode_combo.setCurrentIndex(i); break
        performance.addRow(tr("spaCR mode"), mode_combo)

        mode_note_label = QLabel()
        mode_note_label.setObjectName("SpacrModeNote")
        mode_note_label.setWordWrap(True)
        performance.addRow("", mode_note_label)

        def _sync_mode_note(*_args):
            key = mode_combo.currentData()
            mode_combo.setToolTip(tr(mode_note(key)))
            text = tr(mode_note(key))
            warning = mode_warning(key)
            if warning:
                # Warn on SELECTION, not on Save: a warning that arrives
                # after the dialog has closed is a report, not a choice.
                text = f"{text}\n\n⚠ {tr(warning)}"
            mode_note_label.setText(text)

        mode_combo.currentIndexChanged.connect(_sync_mode_note)
        _sync_mode_note()

        # The four buttons. Each one is confirmed by a dialog that NAMES
        # what will happen — "are you sure?" is not something a user can
        # consent to — and each reports what was actually freed, measured
        # before and after, including when that is nothing.
        def _resource_button(action, label_text, row_label):
            button = QPushButton(tr(label_text))
            button.setObjectName({
                "ram": "ClearRamButton", "vram": "ClearVramButton",
                "cpu": "ClearCpuButton", "disk": "CheckDiskButton",
            }[action])
            from . import resource_cleanup
            button.setToolTip(resource_cleanup.confirmation_text(action))
            button.clicked.connect(lambda: run_resource_action(action, dlg))
            performance.addRow(tr(row_label), button)
            return button

        _resource_button("ram", "Clear RAM", "Memory")
        _resource_button("vram", "Clear VRAM", "GPU memory")
        _resource_button("cpu", "Clear CPU", "Threads")
        _resource_button("disk", "Check disk space", "Disk")

        outer.addWidget(tabs)

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
        # `ResetRole` is what puts it on the LEFT, away from Save and
        # Cancel: every Qt style groups the destructive-ish button apart
        # from the two that close the dialog, which is what stops it being
        # clicked by muscle memory aimed at Cancel.
        reset_button = buttons.addButton(
            tr("Reset to defaults"), QDialogButtonBox.ResetRole)
        reset_button.setObjectName("PreferencesReset")
        reset_button.setToolTip(tr(
            "Put every preference back to the value a fresh install has. "
            "Nothing is written until you press Save, so Cancel still "
            "undoes it."))
        outer.addWidget(buttons)

        def _select(combo, value) -> None:
            """Point ``combo`` at the entry whose data is ``value``."""
            if value is None:
                return
            index = combo.findData(value)
            if index >= 0:
                combo.setCurrentIndex(index)

        def _reset_to_defaults() -> None:
            """Put every control back to what a fresh install would show.

            Read through the real getters against an EMPTY store rather
            than from a second copy of the default values. A hand-written
            table here would be a second place to update every time a
            preference gains a default, and the failure mode of getting it
            wrong is silent: a Reset that quietly sets something to a value
            no code path ever chose.

            Only the controls change. Nothing is persisted until Save, so
            Cancel still walks away from a reset the user did not mean --
            which is why this does not write the empty store back.
            """
            import os
            import tempfile

            from PySide6.QtCore import QSettings

            global _settings
            original = _settings
            empty = os.path.join(
                tempfile.mkdtemp(prefix="spacr-defaults-"), "defaults.ini")
            _settings = lambda: QSettings(empty, QSettings.IniFormat)
            try:
                _select(language_combo, get_language())
                _select(theme_combo, get_theme_choice())
                _select(ambient_theme_combo, get_ambient_animation())
                _select(ambient_palette_combo, get_ambient_palette())
                _select(ambient_dir_combo, get_ambient_drift_direction())
                _select(dock_combo, get_dock_mode())
                _select(cb_combo, get_color_blind_mode())
                _select(fig_format_combo, get_figure_format())
                _select(png_dpi_combo, get_figure_png_dpi())
                _select(mode_combo, get_spacr_mode())

                resolution_slider.setValue(
                    int(round(get_ambient_resolution() * 100)))
                blur_slider.setValue(int(round(get_ambient_blur() * 100)))
                speed_slider.setValue(int(round(get_ambient_speed() * 100)))
                size_slider.setValue(int(round(get_ambient_size() * 100)))
                density_slider.setValue(
                    int(round(get_ambient_density() * 100)))
                spinner_slider.setValue(
                    int(round(get_spinner_delay() * 10)))
                scale_slider.setValue(int(round(get_font_scale() * 100)))
                opacity_slider.setValue(
                    int(round(get_pane_opacity() * 100)))

                setting_anim_check.setChecked(
                    get_setting_animations_enabled())
                field_fade_check.setChecked(get_field_fade_enabled())
                verbose_check.setChecked(get_verbose_logging())
                db_edit_check.setChecked(get_db_browser_editable())
                alpha_check.setChecked(get_show_alpha())
                beta_check.setChecked(get_show_beta())
            finally:
                _settings = original

        reset_button.clicked.connect(_reset_to_defaults)

        def _save():
            set_language(language_combo.currentData())
            set_theme_choice(theme_combo.currentData())
            # One write for the whole Animation row: it stores the choice,
            # repairs a palette the new animation cannot draw, and turns the
            # backdrop off for None (which is what makes "no timer" true —
            # every install site reads `get_ambient_enabled` first).
            set_ambient_animation(ambient_theme_combo.currentData())
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
            set_ambient_resolution(resolution_slider.value() / 100.0)
            set_ambient_density(density_slider.value() / 100.0)
            direction_choice = ambient_dir_combo.currentData()
            if direction_choice is not None:
                set_ambient_drift_direction(direction_choice)
            set_spinner_delay(spinner_slider.value() / 10.0)
            set_setting_animations_enabled(setting_anim_check.isChecked())
            set_font_scale(scale_slider.value() / 100.0)
            set_dock_mode(dock_combo.currentData())
            set_pane_opacity(opacity_slider.value() / 100.0)
            set_field_fade_enabled(field_fade_check.isChecked())
            set_color_blind_mode(cb_combo.currentData())
            set_verbose_logging(verbose_check.isChecked())
            # set_log_levels re-clamps rather than trusting the dialog: the
            # console switch is disabled when its file switch is off, but a
            # disabled QCheckBox still reports whatever it was last set to.
            set_log_levels(
                [level for level, (file_t, _c) in log_level_toggles.items()
                 if file_t.isChecked()],
                [level for level, (_f, console_t) in log_level_toggles.items()
                 if console_t.isChecked()],
            )
            set_db_browser_editable(db_edit_check.isChecked())
            set_show_alpha(alpha_check.isChecked())
            set_show_beta(beta_check.isChecked())
            set_figure_format(fig_format_combo.currentData())
            set_figure_png_dpi(png_dpi_combo.currentData())
            # LAST of the writes, and deliberately: entering Extra
            # Performance overrides five of the settings written above with
            # their minimums, and leaving it puts back what it stashed. Do
            # it earlier and the dialog's own values would land on top,
            # which would mean the mode silently did not take effect.
            set_spacr_mode(mode_combo.currentData())
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
