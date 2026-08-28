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

* ``theme``: ``"dark"`` | ``"light"`` | ``"cell"`` | ``"glass"`` |
  ``"system"`` (default ``"system"``). ``"system"`` follows the operating
  system color scheme. ``"cell"`` uses fluorescence imagery and ``"glass"``
  uses neutral layered materials over a built-in light field. Legacy Space
  accessors remain for old settings, but Space is not a selectable theme.
* ``space_seed``: int; retained for deterministic legacy Space backgrounds.
* ``font_scale``: float, 1.0 = 100 % (the default). Clamped to [0.75, 2.0].
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
  is softened, in units of eight screen pixels. Image detail is controlled
  separately by ``ambient_resolution``. Values stored under the legacy blur
  scale are translated once on read by :func:`_migrate_ambient_motion`.
* ``ambient_drift_direction``: ``"up"`` | ``"down"`` | ``"random"``
  (default ``"up"``). Which way the Starfield animation travels. A
  preference rather than three entries in the animation menu; see
  :data:`spacr.qt.widgets.ambient.DRIFT_DIRECTIONS` for why.
* ``spinner_delay``: float seconds, default ``2.0``. How long background
  work has to run before the activity spinner appears at all — see
  :func:`get_spinner_delay`.
* ``setting_animations``: bool, default ``False``. Whether setting tooltips
  play their animations automatically. When disabled, hover remains text-only
  until the user activates **Animation** in that tooltip's footer; see
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
_KEY_SHARE_DIAGNOSTICS = "privacy/share_diagnostic_logs"
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
_KEY_LAPTOP_MODE = "prefs/laptop_mode"
_KEY_FONT_WEIGHT = "prefs/interface_font_weight"
_KEY_PRELOAD = "prefs/preload_policy"
_KEY_FRACTAL_PATTERN = "spaceout/fractal_pattern"
_KEY_FRACTAL_BACKEND = "spaceout/fractal_backend"
_KEY_FRACTAL_QUALITY = "spaceout/fractal_quality"
_KEY_FRACTAL_SCALE = "spaceout/fractal_scale"
_KEY_FRACTAL_SPEED = "spaceout/fractal_speed"
_KEY_FRACTAL_DREAM = "spaceout/fractal_dream"
_KEY_FRACTAL_VARIABLE_SPEED = "spaceout/fractal_variable_speed"
_KEY_FRACTAL_SPEED_MIN = "spaceout/fractal_speed_min"
_KEY_FRACTAL_SPEED_MAX = "spaceout/fractal_speed_max"
_KEY_FRACTAL_SPEED_PERIOD = "spaceout/fractal_speed_period"
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
#: 100% IS THE DEFAULT, and the reason it was 150% is worth keeping: spaCR's
#: natural size was laid out on a 1080p display, and on a 4K panel driven at
#: 1x everything read small.
#:
#: But this scales the whole interface, and on a HiDPI display the operating
#: system is ALREADY scaling -- macOS reports a 2x device pixel ratio and
#: draws accordingly. Applying 1.5 on top of that is 3x linear and NINE
#: TIMES the pixels of a 100% layout, which is a laptop rendering nine
#: screens' worth of work to show one. Reported as spaCR being "extremely
#: slow" on a machine measurably faster than the workstation it runs well
#: on.
#:
#: The 4K case is a preference a user on that display sets once. The laptop
#: case was everybody, silently, by default.
DEFAULT_FONT_SCALE = 1.0

VALID_CB_MODES = ("off", "deuteranopia", "protanopia", "tritanopia")
DEFAULT_CB_MODE = "off"

# Figure rendering
_KEY_FIG_FORMAT = "prefs/figure_format"
_KEY_FIG_PNG_DPI = "prefs/figure_png_dpi"
VALID_FIG_FORMATS = ("png", "pdf")
DEFAULT_FIG_FORMAT = "pdf"
VALID_PNG_DPIS = (100, 200, 300, 600, 1200)
DEFAULT_PNG_DPI = 300

# How many of the most recent figures keep their LIVE matplotlib Figure, and
# what happens to the ones past that.
#
# A live Figure is what makes a figure restylable: it still has a legend to
# toggle, an axis to set log, series to recolour. A pixmap has none of those
# — it is a picture of a figure. Keeping every Figure forever is not an
# option either, since each holds its own data arrays.
_KEY_FIG_LIVE_CACHE = "prefs/figure_live_cache"
_KEY_FIG_DYNAMIC = "prefs/figure_dynamic"
DEFAULT_FIG_LIVE_CACHE = 20
#: Bounds, not a menu: any number in range is legal.
MIN_FIG_LIVE_CACHE = 1
MAX_FIG_LIVE_CACHE = 500
DEFAULT_FIG_DYNAMIC = True


#: Set by :func:`enable_safe_mode` before anything reads a preference.
#: Process-local and never persisted: safe mode is a way IN, not a state to
#: get stuck in, so an ordinary `spacr` start can never inherit it.
_SAFE_MODE = False


#: What safe mode turns OFF outright, rather than leaving to a default.
#:
#: DEFAULTS ARE NOT SAFE BY THEMSELVES. The animated backdrop is on by
#: default, and the backdrop and the GL path it can take are exactly what
#: the crash log points at -- so a safe mode that merely ignored the stored
#: preferences would start the very thing it exists to avoid. Verbose
#: logging is here for the same reason: it traces per-frame paint calls and
#: writes megabytes a minute, which is its own way of making the interface
#: unusable.
_SAFE_OVERRIDES = {}


class _DefaultsForReadingRealForWriting:
    """Reads answer with the caller's default; writes reach the real store.

    THE POINT OF SAFE MODE IS TO ESCAPE A SAVED VALUE. When a preference is
    what makes spaCR die on launch, a safe mode that reads that same
    preference inherits the fault it exists to escape -- so here every read
    returns the fallback the caller passed, exactly as a first-ever launch
    would see it, without consulting the stored value at all.

    Writes are NOT shadowed. The user opened safe mode to change a setting
    and save it, and a write that went to a scratch file would leave the
    broken value in place and the next ordinary start would die again.

    :param real: the ``QSettings`` writes are forwarded to.
    """

    def __init__(self, real: QSettings):
        self._real = real

    def value(self, key, default=None, type=None):
        """A forced-safe value where there is one, else the caller's default.

        :returns: the entry in :data:`_SAFE_OVERRIDES` for ``key``, or
            ``default`` -- so every other getter falls back to its own
            documented default without a branch of its own.
        """
        if key in _SAFE_OVERRIDES:
            return _SAFE_OVERRIDES[key]
        return default

    def setValue(self, key, value) -> None:
        """Write through to the real store."""
        self._real.setValue(key, value)

    def remove(self, key) -> None:
        """Remove from the real store."""
        self._real.remove(key)

    def sync(self) -> None:
        """Flush the real store."""
        self._real.sync()


def _fill_safe_overrides() -> None:
    """Populate :data:`_SAFE_OVERRIDES` once the key names exist."""
    _SAFE_OVERRIDES.update({
        _KEY_AMBIENT_ENABLED: False,
        _KEY_SETTING_ANIMATIONS: False,
        _KEY_VERBOSE_LOG: False,
        _KEY_PRELOAD: "on_demand",
    })


def enable_safe_mode() -> None:
    """Read preferences as defaults for the rest of this process.

    Called by the ``safespacr`` entry point before any preference is read.
    Idempotent.
    """
    global _SAFE_MODE
    _fill_safe_overrides()
    _SAFE_MODE = True


def in_safe_mode() -> bool:
    """Whether this process is running in safe mode.

    :returns: ``True`` after :func:`enable_safe_mode`.
    """
    return _SAFE_MODE


def _settings():
    """The preference store: the real one, or safe mode's read shadow."""
    real = QSettings(_ORG, _APP)
    return _DefaultsForReadingRealForWriting(real) if _SAFE_MODE else real


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

#: Where the general figure style lives in QSettings.
_KEY_FIG_STYLE = "figures/style_general"
#: Where the per-graph overrides live, as one JSON blob keyed by graph kind.
_KEY_FIG_STYLE_PER_GRAPH = "figures/style_per_graph"


#: Name of the preferred AI provider used when the console opens.
_KEY_AI_PROVIDER = "ai/preferred_provider"


def get_preferred_provider() -> str:
    """Return the preferred AI provider name.

    An empty string allows the console to select an available provider.
    """
    return str(_settings().value(_KEY_AI_PROVIDER, "") or "")


def set_preferred_provider(name: str) -> None:
    """Store the preferred AI provider name.

    Parameters
    ----------
    name : str
        Provider name. An empty string clears the preference.
    """
    _settings().setValue(_KEY_AI_PROVIDER, str(name or ""))


#: QSettings key for the mapping of panel identifiers to folded state. One
#: mapping accommodates newly added panels without introducing new preference
#: keys or requiring callers to discover them individually.
_KEY_FOLDED = "ui/folded_panels"


def get_folded_panels() -> dict:
    """Which bottom panels the user left folded, ``{key: True}``.

    Keyed by ``"<module>/<panel>"`` so folding the console on Mask does not
    fold it on Sequencing -- the same rule the console/chat splitter already
    follows, and for the same reason: the modules are used for different
    work and want different amounts of room.
    """
    import json

    raw = _settings().value(_KEY_FOLDED, "")
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        return {str(k): bool(v) for k, v in value.items()} \
            if isinstance(value, dict) else {}
    except (TypeError, ValueError, AttributeError):
        return {}


def set_folded_panel(key: str, shut: bool) -> None:
    """Remember that ``key`` is folded, or is not.

    A PANEL THAT IS OPEN IS REMOVED rather than stored as False. The default
    is open, so storing it would grow the dict by one entry for every panel
    the user has ever touched and never shrink it.
    """
    import json

    key = str(key or "").strip()
    if not key:
        return
    state = get_folded_panels()
    if shut:
        state[key] = True
    else:
        state.pop(key, None)
    _settings().setValue(_KEY_FOLDED, json.dumps(state))


def get_figure_style() -> dict:
    """The user's GENERAL figure settings, or an empty dict.

    Empty rather than the defaults: :func:`spacr.figure_style.resolve` layers
    the defaults underneath, so storing them here as well would freeze today's
    defaults into every user's settings and make improving them impossible.
    """
    import json

    raw = _settings().value(_KEY_FIG_STYLE, "")
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except (TypeError, ValueError):
        return {}


def set_figure_style(style: dict) -> None:
    """Store the general figure settings."""
    import json

    _settings().setValue(_KEY_FIG_STYLE, json.dumps(dict(style or {})))


def get_figure_style_per_graph() -> dict:
    """Per-graph overrides, ``{kind: {setting: value}}``."""
    import json

    raw = _settings().value(_KEY_FIG_STYLE_PER_GRAPH, "")
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        return {k: v for k, v in value.items() if isinstance(v, dict)} \
            if isinstance(value, dict) else {}
    except (TypeError, ValueError):
        return {}


def set_figure_style_per_graph(overrides: dict) -> None:
    """Store the per-graph overrides."""
    import json

    clean = {k: dict(v) for k, v in (overrides or {}).items()
             if isinstance(v, dict) and v}
    _settings().setValue(_KEY_FIG_STYLE_PER_GRAPH, json.dumps(clean))


#: Where a SAVED STYLE OBJECT's per-project default lives.
#:
#: NOT the same store as `_KEY_FIG_STYLE_PER_GRAPH`, and the difference is
#: worth stating because the two look alike from a distance. That one holds
#: `spacr.figure_style`'s own vocabulary -- font, palette, marker size -- which
#: `figure_style.resolve` merges into rcParams for every figure spaCR draws.
#: THIS one holds a verbatim snapshot of one interactive plot's own style
#: DATACLASS (`volcano_style.VolcanoStyle` and whatever joins it), keyed by the
#: kind of style it is. Merging the two vocabularies would put `label_top_n`
#: into `rcParams.update`, which raises rather than being ignored.
_KEY_FIG_STYLE_DEFAULTS = "figures/style_defaults"


def get_figure_style_defaults() -> dict:
    """Every saved per-project style default, as ``{kind: {field: value}}``."""
    raw = _settings().value(_KEY_FIG_STYLE_DEFAULTS, None)
    if isinstance(raw, dict):
        return {str(kind): dict(values) for kind, values in raw.items()
                if isinstance(values, dict)}
    if isinstance(raw, str) and raw.strip():
        import json
        try:
            loaded = json.loads(raw)
        except ValueError:
            return {}
        if isinstance(loaded, dict):
            return {str(kind): dict(values) for kind, values in loaded.items()
                    if isinstance(values, dict)}
    return {}


def get_figure_style_default(kind: str) -> dict:
    """The saved default for one kind of style, or ``{}``.

    Empty rather than "today's defaults", for the reason the figure colour
    section states at length: a stored resolution is a preference that has
    stopped tracking. A style with no saved default is drawn from the
    dataclass's own defaults, which move when the package does.
    """
    return dict(get_figure_style_defaults().get(str(kind), {}))


def set_figure_style_default(kind: str, values) -> None:
    """Make ``values`` the default for every future figure of ``kind``.

    The design: "a per-project default so a lab's house style is
    applied to every figure of that type without re-setting it each time".
    """
    import json

    stored = get_figure_style_defaults()
    stored[str(kind)] = dict(values or {})
    settings = _settings()
    # JSON rather than a nested QVariant map: QSettings' INI writer flattens a
    # dict of dicts into keys containing the field names, and a style field
    # called `x_label` would then be indistinguishable from a group.
    settings.setValue(_KEY_FIG_STYLE_DEFAULTS, json.dumps(stored))
    settings.sync()


def clear_figure_style_default(kind: str) -> bool:
    """Forget the default for ``kind``. True if there was one.

    The way back, and it is not optional: a default that can only be set is
    the same trap as a colour that can only be set.
    """
    import json

    stored = get_figure_style_defaults()
    if str(kind) not in stored:
        return False
    stored.pop(str(kind))
    settings = _settings()
    settings.setValue(_KEY_FIG_STYLE_DEFAULTS, json.dumps(stored))
    settings.sync()
    return True


def apply_figure_style(kind: str | None = None) -> dict:
    """Push the user's style for ``kind`` into matplotlib. Returns it.

    The one call a plotting function needs: it reads the preferences, layers
    them over the defaults and this graph kind's own, and applies the result.
    """
    from ..figure_style import apply

    return apply(kind, get_figure_style(), get_figure_style_per_graph())


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


def get_figure_live_cache() -> int:
    """How many of the most recent figures keep their live matplotlib Figure.

    The Figures panel used to hold a pixmap per figure and a Figure for every
    one of them, unbounded. The pixmap is what it displayed, so nothing could
    be restyled from a picture; the Figures were retained but never capped, so
    a long run accumulated all of them.

    This bounds the live set. Figures past it keep their rendered page and
    stay viewable -- see :func:`get_figure_dynamic` for what happens when the
    user navigates back to one.

    Larger is more restylable and more memory; a figure with a big ``imshow``
    panel can hold tens of megabytes.
    """
    try:
        value = int(_settings().value(_KEY_FIG_LIVE_CACHE,
                                      DEFAULT_FIG_LIVE_CACHE))
    except (TypeError, ValueError):
        return DEFAULT_FIG_LIVE_CACHE
    return max(MIN_FIG_LIVE_CACHE, min(value, MAX_FIG_LIVE_CACHE))


def set_figure_live_cache(count: int) -> None:
    """Persist how many figures keep their live Figure.

    :raises ValueError: outside ``MIN_FIG_LIVE_CACHE..MAX_FIG_LIVE_CACHE``.
    """
    count = int(count)
    if not MIN_FIG_LIVE_CACHE <= count <= MAX_FIG_LIVE_CACHE:
        raise ValueError(
            f"figure live cache must be between {MIN_FIG_LIVE_CACHE} and "
            f"{MAX_FIG_LIVE_CACHE}; got {count}.")
    _settings().setValue(_KEY_FIG_LIVE_CACHE, count)


def get_figure_dynamic() -> bool:
    """Whether an evicted figure is reloaded from its vector page on demand.

    With this on, navigating back past the live-cache window and selecting a
    figure loads its PDF if one exists, so an old figure is shown from the
    vector page rather than from the display-capped raster and stays sharp at
    any zoom. Off, it shows the raster it already has, which is faster and
    touches no disk.

    It cannot make an old figure restylable again -- a PDF is a finished page,
    with no legend to toggle. It makes it *legible*.
    """
    raw = _settings().value(_KEY_FIG_DYNAMIC, DEFAULT_FIG_DYNAMIC)
    if isinstance(raw, str):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return bool(raw)


def set_figure_dynamic(enabled: bool) -> None:
    """Persist whether evicted figures reload from their vector page."""
    _settings().setValue(_KEY_FIG_DYNAMIC, bool(enabled))


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


# Figure colours. Stored as TOKENS, never as answers: either an explicit
# colour the user picked, or "auto" (the default), which is resolved against
# the live theme on every read.
#
# NEVER PERSIST A RESOLVED DEFAULT.
# ---------------------------------
# This is the rule the whole section exists to enforce, and it is written
# here because this is where the next person will be standing when they are
# about to break it. Writing back what "auto" happened to resolve to turns a
# preference that TRACKS into a preference that FREEZES, and the damage
# outlives the session that caused it: a store holding "#ffffff" cannot be
# told apart from a user who chose white, so nothing downstream can ever undo
# it. That is not hypothetical -- `_FigureSettingsDialog` seeded itself from
# `get_figure_colors()` (resolved) and wrote the same pair back on OK, so
# opening the dialog once on a dark theme and pressing OK without touching
# anything froze every future figure white, including on a light theme. See
# `_migrate_frozen_figure_colors` for the clean-up that costs.
#
# The same reasoning is already recorded one screen up for `get_figure_style`,
# which stores {} rather than today's defaults for exactly this reason.
#
# So: anything that can WRITE this preference back seeds itself from
# `get_figure_color_tokens()`, shows `auto_figure_colors()` as a labelled
# PREVIEW, and passes "auto" to `set_figure_colors` unless the user picked.
_KEY_FIG_BG = "prefs/figure_bg"
_KEY_FIG_FG = "prefs/figure_fg"
#: Preference key for line colour, including axis spines and tick marks.
#: Text and tick-label colour remains under :data:`_KEY_FIG_FG`.
#: `_KEY_FIG_FG` is the font half and predates the split, which is why it is
#: still spelled `fg` -- renaming the key would silently discard the colour of
#: every store that already holds one.
_KEY_FIG_LINE = "prefs/figure_line"
_KEY_FIG_TEXT_SIZE = "prefs/figure_text_size"
#: Marker recording which generation of the un-freeze migration a store has
#: been through — see :func:`_migrate_frozen_figure_colors`.
_KEY_FIG_COLOR_SCALE = "prefs/figure_color_scale"
#: Distinguishes a current explicit choice from the indistinguishable colour
#: pair written by the retired dialog. Older stores do not carry this marker.
_KEY_FIG_COLORS_EXPLICIT = "prefs/figure_colors_explicit"

#: Bump when a *new* family of frozen values needs unfreezing; every store
#: below this number is examined once and then marked.
FIGURE_COLOR_SCALE = 1

#: The token meaning "ask the theme, every time". Not a colour.
AUTO_FIGURE_COLOR = "auto"


#: What "no background at all" is spelled as, in the one place that decides
#: it. matplotlib understands "none" for a facecolor; savefig needs
#: `transparent=True` as well, which is why callers test against this
#: constant rather than comparing strings of their own.
TRANSPARENT_FIGURE_BG = "none"


def figure_bg_is_transparent(bg: str) -> bool:
    """Whether ``bg`` means "let whatever is behind show through"."""
    return str(bg).strip().lower() in {"none", "transparent", ""}


def figure_color_is_auto(token) -> bool:
    """Whether ``token`` is the "follow the theme" token rather than a colour.

    Matching is case- and space-insensitive because tokens can come from a
    hand-edited INI file or the dialog.
    """
    return str(token).strip().lower() == AUTO_FIGURE_COLOR


def auto_figure_colors() -> tuple:
    """What :data:`AUTO_FIGURE_COLOR` resolves to *right now*, as
    ``(background, text)``.

    Public because a control that offers "Follow the theme" has to SHOW what
    that currently means without storing it. Storing what this returns is the
    bug the section header describes; previewing it is the fix.

    TRANSPARENT, not the theme's window colour. "auto" used to resolve to
    #000000 on a dark theme, which is where the black slab behind every plot
    came from: an opaque black rectangle sitting on a container that is a
    translucent SURFACE. ``bg`` is the window colour and a figure is not a
    window (INVARIANTS 2).

    Transparent also means the page-opacity preference reaches the plot for
    free, and one value is right for both themes — baking in a grey would
    freeze one opacity into every figure while everything around it kept
    following the preference.
    """
    # Light is the only light theme; Space is a dark one, so a `== "dark"`
    # test here would have handed it white figures.
    dark = resolve_effective_theme() != "light"
    return TRANSPARENT_FIGURE_BG, ("#ffffff" if dark else "#000000")


#: Background values that a historical ``"auto"`` preference could persist:
#: the current transparent value and the former opaque light/dark values. A
#: stored value equal to one of these is
#: indistinguishable from a resolution that was written back, which is why
#: the migration below cannot be cleverer than "assume the bug".
_FROZEN_BG_VALUES = frozenset({TRANSPARENT_FIGURE_BG, "#000000", "#ffffff"})
#: The same, for the text colour. "auto" has only ever produced black or
#: white, so any black or white in the store is suspect.
_FROZEN_FG_VALUES = frozenset({"#000000", "#ffffff"})


def _migrate_frozen_figure_colors() -> None:
    """Restore persisted theme-derived figure colors to ``"auto"`` once.

    Older dialogs could store the resolved theme colors as explicit values.
    Values that match a known automatic background, text, or line color are
    therefore returned to automatic mode. A scale marker prevents repeated
    migration, and preference-access failures are ignored because they must
    not interrupt figure rendering.
    """
    settings = _settings()
    try:
        if int(settings.value(_KEY_FIG_COLOR_SCALE, 0) or 0) >= \
                FIGURE_COLOR_SCALE:
            return
    except (TypeError, ValueError):
        pass
    try:
        changed = []
        # The line key is examined on the same pass rather than behind a
        # scale bump, and that is safe rather than lucky: it did not exist
        # before this migration shipped, so a store already marked cannot
        # hold a frozen one. A key that CAN predate the marker needs the
        # bump; this one cannot.
        for key, frozen, label in (
                (_KEY_FIG_BG, _FROZEN_BG_VALUES, "background"),
                (_KEY_FIG_FG, _FROZEN_FG_VALUES, "text colour"),
                (_KEY_FIG_LINE, _FROZEN_FG_VALUES, "line colour")):
            raw = settings.value(key, None)
            if raw is None:
                continue
            token = str(raw).strip().lower()
            if token != AUTO_FIGURE_COLOR and token in frozen:
                settings.setValue(key, AUTO_FIGURE_COLOR)
                changed.append(f"{label} {str(raw).strip()!r}")
        settings.setValue(_KEY_FIG_COLOR_SCALE, FIGURE_COLOR_SCALE)
        settings.sync()
        if changed:
            # WARNING AND NOT INFO, AND THE LEVEL IS THE MESSAGE'S JOB. This
            # line exists so that changing a stored preference underneath the
            # user is not silent -- the docstring above says "with a line in
            # the console". It was not reaching one: `spacr.qt` is pinned by
            # the app's level policy, so an INFO logged from
            # `spacr.qt.preferences` is dropped at the source before any
            # handler sees it. Measured after pressing Save in Preferences:
            # `spacr.qt` at level 30, `spacr.qt.preferences` at NOTSET, so
            # the effective level for this module is WARNING.
            #
            # It is warning-grade on its own merits too: something the user
            # did not ask for happened to their settings.
            LOG.warning(
                "Figure colours: %s had been saved as a fixed colour that is "
                "exactly what \"follow the theme\" produces, which is how an "
                "older Figure settings dialog left them. They now follow the "
                "theme again. Pick a colour in Figure settings… to set one "
                "deliberately.", " and ".join(changed))
    except Exception:
        LOG.debug("could not migrate the figure colour keys", exc_info=True)


def get_figure_color_tokens() -> tuple:
    """The STORED ``(background, text)`` tokens, *unresolved*.

    Either half may be :data:`AUTO_FIGURE_COLOR`. Anything that will write the
    preference back must seed itself from here rather than from
    :func:`get_figure_colors`, because a resolved pair has already lost the
    one bit that matters: whether the user chose it.
    """
    _migrate_frozen_figure_colors()
    _unfreeze_figure_colors_that_fight_the_theme()
    settings = _settings()
    return (str(settings.value(_KEY_FIG_BG, AUTO_FIGURE_COLOR)),
            str(settings.value(_KEY_FIG_FG, AUTO_FIGURE_COLOR)))


def _unfreeze_figure_colors_that_fight_the_theme() -> None:
    """Restore an implicit frozen color pair when it conflicts with the theme.

    The repair applies only when neither color was explicitly selected, both
    values are known automatic resolutions, and the pair differs from the
    current theme. Explicit or custom colors remain unchanged. Preference
    access failures are ignored so a cosmetic repair cannot stop rendering.
    """
    try:
        settings = _settings()
        if _as_bool(settings.value(_KEY_FIG_COLORS_EXPLICIT, False), False):
            return                      # chosen through the current dialog/API
        bg = str(settings.value(_KEY_FIG_BG, AUTO_FIGURE_COLOR))
        fg = str(settings.value(_KEY_FIG_FG, AUTO_FIGURE_COLOR))
        if figure_color_is_auto(bg) or figure_color_is_auto(fg):
            return                      # nothing frozen to hand back
        if bg.lower() not in _FROZEN_BG_VALUES:
            return                      # a colour "auto" never produced
        if fg.lower() not in _FROZEN_FG_VALUES:
            return
        if (bg.lower(), fg.lower()) == tuple(
                str(v).lower() for v in auto_figure_colors()):
            return                      # frozen, but at today's answer anyway
        settings.setValue(_KEY_FIG_BG, AUTO_FIGURE_COLOR)
        settings.setValue(_KEY_FIG_FG, AUTO_FIGURE_COLOR)
        print(f"Figure colours were pinned to {bg} / {fg}, which is what "
              f"'follow the theme' resolved to on a different theme. They "
              f"have been handed back to the theme; set them explicitly in "
              f"Preferences > Figures if that was deliberate.")
    except Exception:                                        # noqa: BLE001
        return


def get_figure_line_token() -> str:
    """The STORED line colour token, *unresolved*.

    Seed a control that will write the preference back from HERE, never from
    :func:`get_figure_line_colour` -- the section header says why, and the
    line half is new enough that it has not yet been frozen by anybody.
    """
    _migrate_frozen_figure_colors()
    return str(_settings().value(_KEY_FIG_LINE, AUTO_FIGURE_COLOR))


def get_figure_line_colour() -> str:
    """The colour a figure's LINES are drawn in, "auto" resolved.

    Automatic means the same ink as the text, which is what every figure did
    before there were two controls -- so a store that has never been touched
    renders exactly as it did, and the split costs nobody a changed figure
    until they choose one.

    WHAT THIS REACHES AND WHAT IT DOES NOT. It is the colour of the figure's
    CHROME: the axis spines and the tick marks. It is deliberately not pushed
    over the data's own lines on every render, because a preference that
    repainted every series in one ink would flatten every multi-series figure
    in the package the first time a theme was read. The control that DOES
    reach the data's lines is the per-figure one
    (:func:`spacr.qt.widgets.figure_settings.apply_line_colour`), which
    is a user asking for it about one figure -- the same division as the
    pyqtgraph side, where the theme sets `_foreground` and `set_line_colour`
    is a menu entry.

    GRIDLINES ARE LEFT ALONE, and that is the one exclusion. A grid repainted
    in the ink is a cage over the data; `spacr.figure_style.PRINT_GRID`
    already states it for the save path and this agrees with it.
    """
    token = get_figure_line_token()
    if not figure_color_is_auto(token):
        return token
    return get_figure_colors()[1]


def set_figure_line_colour(token: str) -> None:
    """Persist the line colour TOKEN. Pass :data:`AUTO_FIGURE_COLOR` for
    "follow the text", never what it resolved to."""
    settings = _settings()
    settings.setValue(_KEY_FIG_LINE, token)
    settings.setValue(_KEY_FIG_COLOR_SCALE, FIGURE_COLOR_SCALE)
    settings.sync()


def get_figure_colors() -> tuple:
    """Return ``(background, text)`` hex colours for rendered figures,
    resolving "auto" against the current theme.

    For DRAWING. A caller that will later write the preference back wants
    :func:`get_figure_color_tokens`; see the section header for why.
    """
    bg, fg = get_figure_color_tokens()
    if figure_color_is_auto(bg) or figure_color_is_auto(fg):
        auto_bg, auto_fg = auto_figure_colors()
        # An EXPLICIT colour the user has chosen is still honoured; only the
        # "auto" halves are substituted.
        if figure_color_is_auto(bg):
            bg = auto_bg
        if figure_color_is_auto(fg):
            fg = auto_fg
    return bg, fg


def set_figure_colors(bg: str, fg: str) -> None:
    """Persist background and text colour TOKENS for generated figures.

    Pass :data:`AUTO_FIGURE_COLOR` for a half the user has not chosen — NEVER
    what :func:`auto_figure_colors` returned for it. See the section header.

    Writing also marks the store as migrated: a value set here is a decision
    taken under the current scheme, so :func:`_migrate_frozen_figure_colors`
    must not second-guess it afterwards.
    """
    settings = _settings()
    settings.setValue(_KEY_FIG_BG, bg)
    settings.setValue(_KEY_FIG_FG, fg)
    settings.setValue(_KEY_FIG_COLOR_SCALE, FIGURE_COLOR_SCALE)
    settings.setValue(_KEY_FIG_COLORS_EXPLICIT, True)
    settings.sync()


def set_figure_colors_auto() -> None:
    """Put both halves back to "follow the theme".

    The explicit way out. A user who has been frozen — by the old dialog or
    by their own click — otherwise has no route back to automatic at all,
    and a preference you can only ever set is a trap.
    """
    set_figure_colors(AUTO_FIGURE_COLOR, AUTO_FIGURE_COLOR)
    # All THREE, because "follow the theme" that left one of them frozen
    # would be the trap this function exists to be the way out of.
    set_figure_line_colour(AUTO_FIGURE_COLOR)


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

    A missing photo master is not an error and not a dead end — the theme
    falls back to the generated sky, which needs no assets and
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
        from PySide6.QtGui import QPalette
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            # THE ENUM, NOT THE INSTANCE. `palette.Window` was removed in
            # PySide6 6.x, and the bare except below would swallow the
            # AttributeError and hand every desktop the dark theme.
            bg = app.palette().color(QPalette.ColorRole.Window)
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
    raises for an unknown theme — it reports the global default.
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


#: What the laptop-mode preference may be set to. ``"automatic"`` leaves the
#: decision to the measurement, which is what an unset preference has always
#: meant; the other two override it in either direction.
LAPTOP_MODE_CHOICES = ("automatic", "on", "off")

#: What the dialog calls each one.
LAPTOP_MODE_LABELS = {
    "automatic": "Automatic (decide from this machine)",
    "on": "On (turn the animation and blur down)",
    "off": "Off (keep everything on)",
}


#: The spaceout fractal's settings. SPACEOUT ONLY -- the rows are not built
#: in an ordinary launch, and these functions are the only readers, so a
#: normal session neither shows them nor is affected by them.
#:
#: The defaults are the maintainer's two command lines. `auto` picks the GPU
#: when vispy is importable and the CPU otherwise, which is what lets one set
#: of numbers serve both.
FRACTAL_PATTERNS = ("orbit", "cascade")
FRACTAL_BACKENDS = ("auto", "gpu", "cpu")
FRACTAL_QUALITIES = ("auto", "balanced", "high")


def get_fractal_settings() -> dict:
    """Every spaceout fractal setting, ready for `Settings`/`RuntimeControls`.

    Read through one function so the dialog and the backdrop cannot disagree
    about a default. Out-of-range stored values are clamped rather than
    refused -- a backdrop must not stop the application from starting.
    """
    from .fractal_defaults import (
        DEFAULT_BACKEND, DEFAULT_DREAM, DEFAULT_PATTERN, DEFAULT_QUALITY,
        DEFAULT_SCALE, DEFAULT_SPEED, DEFAULT_SPEED_MAX, DEFAULT_SPEED_MIN,
        DEFAULT_SPEED_PERIOD, DEFAULT_VARIABLE_SPEED, clamp,
    )

    settings = _settings()

    def _text(key, default, allowed):
        raw = str(settings.value(key, default))
        return raw if raw in allowed else default

    def _number(key, default, low, high):
        try:
            return clamp(float(settings.value(key, default)), low, high)
        except (TypeError, ValueError):
            return default

    raw_variable = settings.value(_KEY_FRACTAL_VARIABLE_SPEED,
                                  DEFAULT_VARIABLE_SPEED)
    if isinstance(raw_variable, str):
        variable = raw_variable.strip().lower() in ("1", "true", "yes", "on")
    else:
        variable = bool(raw_variable)

    return {
        "pattern": _text(_KEY_FRACTAL_PATTERN, DEFAULT_PATTERN,
                         FRACTAL_PATTERNS),
        "backend": _text(_KEY_FRACTAL_BACKEND, DEFAULT_BACKEND,
                         FRACTAL_BACKENDS),
        "quality": _text(_KEY_FRACTAL_QUALITY, DEFAULT_QUALITY,
                         FRACTAL_QUALITIES),
        "scale": _number(_KEY_FRACTAL_SCALE, DEFAULT_SCALE, 0.25, 2.0),
        "speed": _number(_KEY_FRACTAL_SPEED, DEFAULT_SPEED, 0.15, 8.0),
        "dream": _number(_KEY_FRACTAL_DREAM, DEFAULT_DREAM, 0.0, 1.5),
        "variable_speed": variable,
        "speed_min": _number(_KEY_FRACTAL_SPEED_MIN, DEFAULT_SPEED_MIN,
                             0.15, 8.0),
        "speed_max": _number(_KEY_FRACTAL_SPEED_MAX, DEFAULT_SPEED_MAX,
                             0.15, 8.0),
        "speed_period": _number(_KEY_FRACTAL_SPEED_PERIOD,
                                DEFAULT_SPEED_PERIOD, 5.0, 300.0),
    }


def set_fractal_settings(**values) -> None:
    """Persist any subset of the fractal settings.

    :raises ValueError: on an unknown name, or a backend/quality outside its
        set. A number outside its range is clamped, because a slider cannot
        produce one and a hand-edited file should still start.
    """
    from .fractal_defaults import clamp

    keys = {
        "pattern": (_KEY_FRACTAL_PATTERN, None),
        "backend": (_KEY_FRACTAL_BACKEND, None),
        "quality": (_KEY_FRACTAL_QUALITY, None),
        "scale": (_KEY_FRACTAL_SCALE, (0.25, 2.0)),
        "speed": (_KEY_FRACTAL_SPEED, (0.15, 8.0)),
        "dream": (_KEY_FRACTAL_DREAM, (0.0, 1.5)),
        "variable_speed": (_KEY_FRACTAL_VARIABLE_SPEED, None),
        "speed_min": (_KEY_FRACTAL_SPEED_MIN, (0.15, 8.0)),
        "speed_max": (_KEY_FRACTAL_SPEED_MAX, (0.15, 8.0)),
        "speed_period": (_KEY_FRACTAL_SPEED_PERIOD, (5.0, 300.0)),
    }
    store = _settings()
    for name, value in values.items():
        if name not in keys:
            raise ValueError(f"unknown fractal setting {name!r}; "
                             f"expected one of {sorted(keys)}")
        key, bounds = keys[name]
        if name == "pattern" and value not in FRACTAL_PATTERNS:
            raise ValueError(f"unknown fractal pattern {value!r}")
        if name == "backend" and value not in FRACTAL_BACKENDS:
            raise ValueError(f"unknown fractal backend {value!r}")
        if name == "quality" and value not in FRACTAL_QUALITIES:
            raise ValueError(f"unknown fractal quality {value!r}")
        if bounds is not None:
            value = clamp(float(value), *bounds)
        if name == "variable_speed":
            value = bool(value)
        store.setValue(key, value)
    store.sync()


#: The two weights the interface is drawn in. Bold and SemiBold stay
#: registered for a stylesheet that asks for emphasis; this is what
#: everything else defaults to.
INTERFACE_FONT_WEIGHTS = ("regular", "light")


#: When the heavy pipeline modules are imported.
#:
#: ``'on_demand'`` -- when the operation that needs them is called. The
#: default, and what instruction 282 asked for.
#: ``'eager'`` -- at startup, on a worker thread. Only worth it on a machine
#: that will certainly run a pipeline and would rather wait once at the
#: beginning; on the maintainer's own machine that wait was TWENTY SECONDS.
PRELOAD_POLICIES = ("on_demand", "eager")


def get_preload_policy() -> str:
    """When to import torch and the rest. 'on_demand' or 'eager'."""
    raw = str(_settings().value(_KEY_PRELOAD, "on_demand")).strip().lower()
    return raw if raw in PRELOAD_POLICIES else "on_demand"


def set_preload_policy(policy: str) -> None:
    """Persist it. Takes effect at the next launch, and says so.

    :raises ValueError: on anything but the two policies.
    """
    text = str(policy).strip().lower()
    if text not in PRELOAD_POLICIES:
        raise ValueError(f"unknown preload policy {policy!r}; expected one "
                         f"of {list(PRELOAD_POLICIES)}")
    _settings().setValue(_KEY_PRELOAD, text)
    _settings().sync()


#: Body text is Light. Asked for 2026-08-28: "light for text and regular for
#: titles". Only the APPLICATION font is set from this -- the headings,
#: buttons and section titles carry their own `font-weight` in the
#: stylesheet (400 and above), so making the default body weight lighter
#: does not thin the titles with it.
DEFAULT_INTERFACE_FONT_WEIGHT = "light"


def get_interface_font_weight() -> str:
    """Which Open Sans weight the interface's body text uses.

    :returns: ``'light'`` or ``'regular'``, defaulting to
        :data:`DEFAULT_INTERFACE_FONT_WEIGHT`.
    """
    raw = str(_settings().value(
        _KEY_FONT_WEIGHT, DEFAULT_INTERFACE_FONT_WEIGHT)).strip().lower()
    return raw if raw in INTERFACE_FONT_WEIGHTS \
        else DEFAULT_INTERFACE_FONT_WEIGHT


def set_interface_font_weight(weight: str) -> None:
    """Persist the weight and apply it to the running application.

    :raises ValueError: on anything but 'regular' or 'light'.
    """
    text = str(weight).strip().lower()
    if text not in INTERFACE_FONT_WEIGHTS:
        raise ValueError(f"unknown interface font weight {weight!r}; "
                         f"expected one of {list(INTERFACE_FONT_WEIGHTS)}")
    _settings().setValue(_KEY_FONT_WEIGHT, text)
    _settings().sync()
    try:
        from PySide6.QtWidgets import QApplication

        from .app import _use_open_sans

        instance = QApplication.instance()
        if instance is not None:
            _use_open_sans(instance, text)
    except Exception:                                        # noqa: BLE001
        pass


def get_laptop_mode() -> str:
    """Whether laptop mode is forced on, forced off, or measured.

    The stored value is one of :data:`LAPTOP_MODE_CHOICES`. Anything else --
    an old file, a hand-edited one -- reads as ``"automatic"``, because a
    setting nobody can interpret should behave as though it were never set.
    """
    raw = str(_settings().value(_KEY_LAPTOP_MODE, "automatic"))
    return raw if raw in LAPTOP_MODE_CHOICES else "automatic"


def set_laptop_mode(choice: str) -> None:
    """Persist the laptop-mode preference and apply it now.

    :raises ValueError: on an unknown choice.

    Applied immediately rather than at the next launch, because the two
    things it changes -- the ambient animation and the backdrop blur -- are
    both visible in the window behind the dialog. A performance setting
    that needs a restart to show its effect cannot be judged by the person
    setting it.
    """
    if choice not in LAPTOP_MODE_CHOICES:
        raise ValueError(f"unknown laptop mode {choice!r}; "
                         f"expected one of {list(LAPTOP_MODE_CHOICES)}")
    _settings().setValue(_KEY_LAPTOP_MODE, choice)
    _settings().sync()
    from .laptop_mode import apply as _apply, wanted
    _apply(None if choice == "automatic" else choice == "on")
    return None


def laptop_mode_note(choice: str) -> str:
    """What the chosen setting will do on THIS machine, said before saving.

    Automatic is the case that needs saying: the label cannot state the
    outcome, because the outcome depends on the machine reading it.
    """
    from .laptop_mode import measure, wanted, what_it_turns_down

    turns_down = ", ".join(what for what, _cost in what_it_turns_down())
    if choice == "automatic":
        # Asked with the override cleared, because what the note has to
        # report is what the MEASUREMENT says -- an environment variable
        # set for one launch would otherwise be read back as the machine's
        # own answer.
        _on, why = wanted({**measure(), "override": None})
        return why
    if choice == "on":
        return (f"Turns down {turns_down}. Only the drawing changes: a run "
                f"computes exactly the same answer either way.")
    return "Keeps the animation and the blur on, whatever this machine is."


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

#: How the auto-issue reporter behaves when something goes wrong.
#:
#: Three states, not two. "prompt me" and "never prompt me" leave out the
#: user who wants a report filed and does not want to be asked, and the
#: moment someone wants this off is the moment it has just interrupted them.
ISSUE_PROMPT_ASK = "ask"
ISSUE_PROMPT_NEVER = "never"
ISSUE_PROMPT_ALWAYS = "always"
ISSUE_PROMPT_MODES = (ISSUE_PROMPT_ASK, ISSUE_PROMPT_NEVER,
                      ISSUE_PROMPT_ALWAYS)
_KEY_ISSUE_PROMPT = "ai/issue_prompt"


#: Whether the AI assistant is on when spaCR opens (221).
#:
#: OFF BY DEFAULT, and that is the default rather than the preference. An
#: assistant that is on before anybody asked for it sends what it is looking
#: at somewhere, and the first run is exactly when the user has not yet
#: decided whether that is acceptable. The setup screen asks; until it is
#: answered the answer is no.
#:
#: A stored value that is not recognised reads as OFF for the same reason a
#: bad `issue_prompt` reads as 'ask': the failure has to fall on the quiet
#: side.
_KEY_AI_DEFAULT_ON = "ai/on_by_default"


#: Off until the setup screen records the user's choice. Enabling an external
#: assistant before consent would send context before the user has decided
#: whether that is acceptable.
DEFAULT_AI_ON_AT_LAUNCH = False


def get_ai_on_by_default() -> bool:
    """Is the assistant on when spaCR opens?

    :returns: :data:`DEFAULT_AI_ON_AT_LAUNCH` unless the user has said
        otherwise. An explicit choice is always written, so an opt-out
        survives a change to the default rather than being overwritten
        by it.
    """
    raw = _settings().value(_KEY_AI_DEFAULT_ON, DEFAULT_AI_ON_AT_LAUNCH)
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def set_ai_on_by_default(enabled: bool) -> None:
    """Persist whether the assistant starts enabled.

    :param enabled: True to have it on at launch.
    """
    _settings().setValue(_KEY_AI_DEFAULT_ON, bool(enabled))


def get_issue_prompt_mode() -> str:
    """How to behave when a report could be filed.

    :returns: one of :data:`ISSUE_PROMPT_MODES`; ``'ask'`` by default, and
        for any stored value that is not recognised -- a preference file
        written by a newer build must not silence the reporter on an older
        one.
    """
    value = str(_settings().value(_KEY_ISSUE_PROMPT, ISSUE_PROMPT_ASK) or "")
    return value if value in ISSUE_PROMPT_MODES else ISSUE_PROMPT_ASK


def set_issue_prompt_mode(mode: str) -> None:
    """Persist the auto-issue behaviour.

    :param mode: one of :data:`ISSUE_PROMPT_MODES`.
    :raises ValueError: for anything else. Silently storing an unknown mode
        would read back as 'ask' and look like the setting was ignored.
    """
    mode = str(mode)
    if mode not in ISSUE_PROMPT_MODES:
        raise ValueError(
            f"issue prompt mode {mode!r} is not one of "
            f"{list(ISSUE_PROMPT_MODES)}.")
    _settings().setValue(_KEY_ISSUE_PROMPT, mode)



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
#: On, and it is not free. Hashing every file under every path-valued
#: setting is proportional to the DATA and not to the run: on a plate of raw
#: images it is minutes of reading before the first mask is made, and it
#: happens whether or not anybody ever compares the digests. It ships on
#: anyway, because a result that cannot be traced back to its inputs is
#: worth less than the minutes, and the cost is refusable in one click.
#: The manifest is written either way and SAYS which it was, so the record
#: is never ambiguous.
DEFAULT_HASH_INPUTS = True
_KEY_HASH_INPUTS = "prefs/hash_inputs"


def get_hash_inputs() -> bool:
    """Whether a run hashes its inputs and outputs for the manifest."""
    return _as_bool(_settings().value(_KEY_HASH_INPUTS, DEFAULT_HASH_INPUTS),
                    DEFAULT_HASH_INPUTS)


def set_hash_inputs(on: bool) -> None:
    """Persist the input-hashing choice."""
    _settings().setValue(_KEY_HASH_INPUTS, bool(on))


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


#: The stored colour-vision mode -> the display-primaries mode
#: :func:`spacr.crops.apply_display_primaries` takes.
#:
#: TWO VOCABULARIES, ON PURPOSE. This preference names a CONDITION, because
#: that is what a user knows about themselves and what they pick in
#: Preferences: "I have deuteranopia". :data:`spacr.crops.DISPLAY_PRIMARIES`
#: names a RENDERING: "draw this for a deuteranope". They are one fact seen
#: from its two ends, and renaming either would break stored settings for no
#: gain, so the bridge is written down once here rather than guessed at every
#: call site.
#:
#: ``cmy`` is deliberately absent. It is a PUBLISHING convention, not an
#: accessibility mode -- measured against a deuteranope simulation it is
#: WORSE than plain RGB -- so it must never be reached by having a
#: deficiency. It is chosen per view, by somebody making a figure.
_CB_MODE_TO_PRIMARIES = {
    "off": "rgb",
    "deuteranopia": "deuteranope",
    "protanopia": "protanope",
    "tritanopia": "tritanope",
}


def image_display_primaries() -> str:
    """How images should be drawn for this user, everywhere.

    The global half of the colour-blind mode. A user who needs the
    substitution needs it in Annotate, in every live view and in every crop
    grid, in every session -- not as a toggle they re-find on each screen.
    A view may still override it, because a figure being prepared for
    publication wants ``cmy`` whatever the author's vision, but this is
    what every view starts from.

    :returns: one of :data:`spacr.crops.DISPLAY_PRIMARIES`.
    """
    return _CB_MODE_TO_PRIMARIES.get(get_color_blind_mode(), "rgb")


def color_blind_categorical_palette() -> list:
    """Return a list of hex colours safe for the active CB mode.

    Uses the Okabe-Ito categorical palette whenever colour-blind mode is
    enabled.
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


#: Verbose diagnostic logging is ON unless the user turns it off.
#:
#: A bug report is worth far more with a trail behind it, and the trail has
#: to already exist when the thing goes wrong -- asking a user to turn
#: logging on and reproduce it is asking for the one run nobody captured.
#:
#: THIS WAS ONLY SAFE ONCE VERBOSE WAS CHEAP. With the animation traced it
#: wrote three 5 MB files a minute and the interface stopped responding;
#: `spacr.logging_util._TRACE_SKIP_MODULES` is what makes the default
#: defensible, and the two must not be separated.
DEFAULT_VERBOSE_LOGGING = True


def get_verbose_logging() -> bool:
    """Whether package-wide diagnostic tracing is on.

    :returns: the stored choice, defaulting to
        :data:`DEFAULT_VERBOSE_LOGGING`. Consulted at startup by
        :func:`apply_preferences_to_app` and toggled in Preferences.
    """
    raw = _settings().value(_KEY_VERBOSE_LOG, DEFAULT_VERBOSE_LOGGING)
    if isinstance(raw, str):
        return raw.lower() in ("true", "1", "yes", "on")
    return bool(raw)


def set_verbose_logging(on: bool) -> None:
    """Persist whether package-wide diagnostic tracing is enabled."""
    _settings().setValue(_KEY_VERBOSE_LOG, bool(on))


#: On, and only safe to be. A report no longer carries log lines in its
#: body: the bundle is written to a local path and the body NAMES that
#: path, so a public issue can never carry a user's logs. The preference
#: governs whether that bundle is prepared at all, every report still
#: stops at an editable preview, and nothing is sent without its own Send.
DEFAULT_SHARE_DIAGNOSTIC_LOGS = True


def get_share_diagnostic_logs() -> bool:
    """Whether report previews may include a redacted recent-log excerpt.

    This never authorises background submission. Every report still stops at
    the editable preview and needs its own Send click.
    """
    return _as_bool(_settings().value(_KEY_SHARE_DIAGNOSTICS,
                                      DEFAULT_SHARE_DIAGNOSTIC_LOGS),
                    DEFAULT_SHARE_DIAGNOSTIC_LOGS)


def set_share_diagnostic_logs(on: bool) -> None:
    """Persist the revocable diagnostic-log preview opt-in."""
    _settings().setValue(_KEY_SHARE_DIAGNOSTICS, bool(on))


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

    # Instruction 180. Pushed on every preferences save and not only at
    # startup: the run journal reads a module-level default it can see
    # without Qt, and a user who changed the setting mid-session would
    # otherwise not see it take effect until the next launch.
    try:
        apply_workspace_preference()
    except Exception:                                       # noqa: BLE001
        LOG.debug("could not push the workspace preference", exc_info=True)

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

#: What each Preferences row means, keyed by the label it carries.
#:
#: ON THE LABEL, NOT THE FIELD, and that is the house rule everywhere in
#: spaCR: the words are what a reader points at when they want to know
#: what something is, and a tooltip on the control is one they find only
#: after reaching for it. :func:`explain_every_row` moves any that were
#: put on a field, so a row explained either way ends up explained the
#: same way.
PREFERENCE_TIPS = {
    # -- logging -------------------------------------------------------
    "Debug": "Record all diagnostic messages, including internal run steps. This produces large logs and is recommended when preparing a bug report.",
    "Info": "Record one message for each major run step. Recommended for routine use.",
    "Warning": "Record conditions that may require review or intervention.",
    "Error": "Record failed operations.",
    "Critical": "Record failures that stop the run.",
    # -- figures: type -------------------------------------------------
    "Font family": "Typeface used for text in saved figures.",
    "Font size": "Base font size for saved figures. Title, label and tick sizes are scaled relative to this value.",
    "Title size": "Figure-title size relative to the base font size.",
    "Label size": "Axis-label size relative to the base font size.",
    "Tick size": "Axis-tick label size relative to the base font size.",
    # -- figures: colour ----------------------------------------------
    "Palette": "Sequence of colours used when a figure contains multiple series.",
    "Background": "Figure background colour. A transparent background uses the colour of the destination document or interface.",
    "Foreground": "Axis lines, ticks and text.",
    "Chrome colour": "Colour used for the figure frame, axis ticks and spines.",
    "Mark colouring": "Assign point colours from the palette or from a selected data column.",
    # -- figures: grid and frame ---------------------------------------
    "Grid": "Draw grid lines behind the data.",
    "Grid colour": "Colour used for grid lines.",
    "Grid width": "Grid-line width in points.",
    "Grid style": "Solid, dashed or dotted.",
    "Spines": "Select which of the four plot-frame edges are drawn.",
    "Spine width": "Plot-frame edge width in points.",
    # -- figures: marks ------------------------------------------------
    "Marker size": "Plotted-point area in points squared.",
    "Marker style": "Shape used for plotted points.",
    "Line width": "Plotted-line width in points.",
    "Jitter width": "Horizontal displacement applied to overlapping points.",
    "Point alpha": "Point opacity. Values below 1 make overlapping points appear darker.",
    "Bar alpha": "Bar opacity.",
    "Fill colour": "Interior colour of bars and histogram bins.",
    "Edge colour": "Outline colour of bars and histogram bins.",
    "Edge width": "Outline width in points.",
    # -- figures: statistics drawn on the plot -------------------------
    "Error bars": "Statistic represented by error bars: standard deviation, standard error or confidence interval.",
    "Reference style": "Line style used for reference or baseline values.",
    "Reference colour": "Reference-line colour.",
    "Trend line": "Fit and draw a trend line through the points.",
    "Trend colour": "Trend-line colour.",
    "Threshold style": "Line style used for significance or effect-size thresholds.",
    "Threshold colour": "Threshold-line colour.",
    "Threshold width": "Threshold-line width in points.",
    "Bins": "Number of intervals used to partition a histogram's range.",
    "Log y": "Use a logarithmic y-axis scale.",
    "Split axis": "Omit an intermediate axis interval when one group is separated substantially from the others.",
    "Centred": "Centre a diverging colour scale on zero so colour indicates the sign of each value.",
    "Colormap": "Colour scale used to represent continuous values.",
    # -- figures: labels and layout ------------------------------------
    "Annotate": "Display values on the plot.",
    "Annotate cells": "Display each cell's value in a heatmap.",
    "Label top n": "Number of highest-ranked points labelled by name.",
    "Legend": "Display a legend at the selected location.",
    "Per row": "Number of panels in each row of a grid figure.",
    # KEYED BY THE LABEL THE ROW SHOWS, which for the `aspect` setting comes
    # from `style_setting_label`. The setting itself offers 'equal' or
    # 'auto', so it is the axis-scale lock -- one y unit drawn the same
    # length as one x unit -- and not the shape of the figure. The shape is
    # 'Page shape' below, and the explanation says so rather than describing
    # shapes this control cannot take.
    "Lock axis scales": "Whether one y unit is drawn the same length as one x "
                   "unit ('equal', which is what keeps a plate's wells "
                   "square), or the panel is filled instead ('auto'). This "
                   "locks the axis scales, which is a statement about the "
                   "data; the proportions of the figure are 'Page shape'.",
    "Page shape": "Aspect of the saved page.",
    "Dpi": "Resolution of the saved image in pixels per inch. A value of 300 is commonly required for print.",
    "Format": "File format used when saving figures.",
    "Tight layout": "Adjust margins to fit labels within the saved figure.",
    # -- the application -----------------------------------------------
    "Theme": "Application colour scheme. 'Follow system' uses the desktop colour scheme.",
    "Font scale": "Scale interface text independently of saved-figure font sizes.",
    "Colour-blind mode": "Use interface and figure colours designed to remain distinguishable for common colour-vision deficiencies.",
    "Module visibility": "Select the module maturity levels shown in navigation: stable only, or stable with beta and alpha modules.",
    "Show busy spinner after": "Delay before displaying the busy indicator for a running task.",
    "Page opacity": "Page opacity relative to the animated background.",
    # -- the backdrop and the rim --------------------------------------
    "Animation detail": "Backdrop rendering detail. Reduce this value if animation affects interface performance.",
    "Pattern": "Which fractal spaceout draws. Orbit fold is an orbit-fold map antialiased across four frames; fold-inversion cascade is a Kaliset-like fold and sphere inversion coloured by three orbit traps, travelling through two overlapping scale windows so it never resets. The cascade takes four samples of one instant per pixel, so it costs about four times as much and runs at a lower frame rate by design.",
    "Backend": "Which renderer draws the fractal. GPU is a shader and is far cheaper; it needs vispy and a real display, and falls back to the CPU renderer when either is missing. Automatic picks the GPU when it can and says below which one this machine will get.",
    "Quality": "How much detail the fractal is asked for. Balanced costs less per frame; high adds an iteration to the fractal and raises the internal resolution. Automatic chooses from the number of cores on the CPU renderer and uses balanced on the GPU.",
    "Scale": "A resource multiplier for the CPU renderer's internal resolution, applied before it adapts. Below 1.0 draws fewer pixels and scales them up; above 1.0 draws more. It does not change what the fractal looks like, only how finely it is sampled. The GPU renderer ignores it.",
    "Speed": "How fast the view travels inward. It scales the depth the fractal is sampled at, so a higher number moves through the structure sooner; it does not change the frame rate or the cost of a frame.",
    "Dream": "How much the pattern warps, drifts and shears as it travels. 0.0 is a still camera moving straight in; 1.5 is the maximum and is the default. It costs nothing extra to raise.",
    "Slowest": "The slowest the travel goes when variable speed is on. Ignored when it is off. If this is above Fastest the two are simply used the other way round -- a swapped pair is not an empty range.",
    "Fastest": "The fastest the travel goes when variable speed is on. Ignored when it is off.",
    "Sweep time": "How long one full sweep takes, slowest to fastest and back. This is how GRADUAL the change is, not how fast the fractal goes: a larger number means the speed drifts more slowly between the two bounds. Below about ten seconds it stops reading as drift and starts reading as a pulse.",
    "Variable speed": "Let the travel speed breathe instead of holding one value. It modulates the speed above rather than replacing it, so the number you set is still the middle of the range.",
    "Interface font": "The weight the interface is drawn in. spaCR ships Open Sans and uses it everywhere, so the application looks the same whatever fonts the machine has. Light is thinner and suits a large high-resolution display; Regular is easier to read on a small or low-resolution one. Bold stays available to anything that asks for emphasis.",
    "Laptop mode": "Turns the ambient animation and the backdrop blur down on a small machine. Automatic decides from the cores and memory it finds. Only drawing is affected: a run computes the same answer either way.",
    "Animation blur": "Blur applied to background shapes.",
    "Animation speed": "Background-animation speed.",
    "Animation size": "Size of background shapes.",
    "Animation density": "Number of background shapes.",
    "Rim length": "Fraction of a card border covered by the moving highlight.",
    "Rim chase": "Responsiveness of the border highlight to pointer movement.",
    "Rim cycle": "Duration of one border-highlight cycle.",
}


def explain_every_row(dialog) -> int:
    """Put a tooltip on every Preferences LABEL. Returns how many it set.

    Two jobs, and the second is why this walks the finished dialog rather
    than being written at each call site: it fills in from
    :data:`PREFERENCE_TIPS`, and it MOVES a tooltip that was put on the
    control to the label beside it. A row explained either way ends up
    explained the same way, and a row added later without a tooltip is
    reported by the test rather than passing unnoticed.
    """
    from PySide6.QtWidgets import (QFormLayout, QLabel, QPushButton,
                                   QToolButton)

    from .widgets.hint_bar import explain_through_the_bar

    from .i18n import tr

    explained = 0
    for form in dialog.findChildren(QFormLayout):
        for index in range(form.rowCount()):
            label_item = form.itemAt(index, QFormLayout.LabelRole)
            field_item = form.itemAt(index, QFormLayout.FieldRole)
            if label_item is None or field_item is None:
                continue
            label = label_item.widget()
            field = field_item.widget()
            if not isinstance(label, QLabel) or field is None:
                continue
            text = (label.text() or "").replace("&", "").strip()
            tip = PREFERENCE_TIPS.get(text, "")
            # A BUTTON IS NOT A SETTING, and its tooltip is not a row's
            # explanation -- it says what pressing it DOES, which is often
            # the confirmation the press will ask for. Moving that onto the
            # label and clearing the button leaves the user hovering the
            # thing they are about to press and being told nothing.
            #
            # So a button row gets its label explained and KEEPS its own
            # tooltip. Every other kind of field hands its explanation over,
            # which is the rule: on the setting's text, never on its field.
            is_action = isinstance(field, (QPushButton, QToolButton))
            if not tip:
                # WHATEVER THE ROW ALREADY SAID, moved rather than
                # duplicated: a tooltip on both reads as two answers.
                tip = (field.toolTip() or "").strip()
            if not tip:
                continue
            label.setToolTip(tr(tip))
            if is_action:
                # A BUTTON SAYS WHAT PRESSING IT DOES, AND IT SAYS IT AT
                # THE FOOT OF THE WINDOW. A tooltip appears over the button
                # -- where the pointer already is, and where the user is
                # about to click -- so the sentence covers the thing it
                # describes. The bar is out of the way, holds a long
                # sentence without hiding anything, and does not flicker as
                # the pointer crosses a row of buttons. This is what the
                # module tiles on Home already do.
                #
                # With no bar in the window the tooltip STAYS: a control
                # that explains itself nowhere is worse than one that
                # explains itself awkwardly.
                explain_through_the_bar(field)
            else:
                field.setToolTip("")
            explained += 1
    return explained


class PreferencesDialog:
    """Wrapper that builds the modal Preferences dialog on demand.

    Kept as a factory (not a real class subclass) so this module can
    be imported headless without pulling in QtWidgets. The real
    :class:`QDialog` is returned by ``PreferencesDialog(parent)``.
    """

    def __new__(cls, parent=None):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QCheckBox, QComboBox, QDialog, QDialogButtonBox,
            QDoubleSpinBox, QFormLayout,
            QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSlider,
            QSpinBox, QTabWidget, QVBoxLayout, QWidget,
        )
        from .i18n import language_choices, tr
        from .theme import spaceout_enabled
        from .widgets.toggle import Toggle

        dlg = QDialog(parent)
        # Detached from the parent for the window manager's purposes, so the
        # user can put it where they like. It is still parented, still modal
        # and still exec()s: only the window TYPE changes. See
        # spacr.qt.dialogs for what a WM does with an attached modal dialog.
        from .dialogs import detach_from_window_manager
        detach_from_window_manager(dlg)
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

        # ---- The travelling rim -------------------------------------
        #
        # THREE SETTINGS, NOT THREE CONSTANTS. How long the light is, how
        # hard it chases and whether it sits centred on the pointer are
        # matters of taste, and taste is what a preference is for.
        rim_length_slider = QSlider(Qt.Horizontal)
        rim_length_slider.setObjectName("RimLength")
        rim_length_slider.setRange(*RIM_LENGTH_RANGE)
        rim_length_slider.setSingleStep(10)
        rim_length_slider.setPageStep(40)
        rim_length_slider.setValue(get_rim_length())
        rim_length_slider.setToolTip(
            "How far the accent runs along the edge of a settings card, in "
            "pixels. Short reads as a dash sitting on one edge; past about "
            "half the perimeter it stops being a highlight and becomes a "
            "border.")
        rim_length_value = QLabel()

        def _rim_length_says(px):
            rim_length_value.setText(tr("%d px") % int(px))

        _rim_length_says(rim_length_slider.value())
        rim_length_slider.valueChanged.connect(_rim_length_says)
        rim_length_row = QHBoxLayout()
        rim_length_row.setContentsMargins(0, 0, 0, 0)
        rim_length_row.addWidget(rim_length_slider, 1)
        rim_length_row.addWidget(rim_length_value)
        appearance.addRow(tr("Rim length"), _hbox_wrap(rim_length_row))

        rim_lag_slider = QSlider(Qt.Horizontal)
        rim_lag_slider.setObjectName("RimLag")
        # Stored as a fraction; shown as a percentage, because a slider
        # from 0.02 to 1.0 is a slider with no readable numbers on it.
        rim_lag_slider.setRange(int(RIM_LAG_RANGE[0] * 100),
                                int(RIM_LAG_RANGE[1] * 100))
        rim_lag_slider.setSingleStep(1)
        rim_lag_slider.setPageStep(5)
        rim_lag_slider.setValue(int(round(get_rim_lag() * 100)))
        rim_lag_slider.setToolTip(
            "How much of the distance to the pointer the accent covers each "
            "frame. Lower is a longer, lazier trail; at 100% it is under "
            "the pointer with no travel at all, and the travel is the whole "
            "effect.")
        rim_lag_value = QLabel()

        def _rim_lag_says(percent):
            rim_lag_value.setText(tr("%d%%") % int(percent))

        _rim_lag_says(rim_lag_slider.value())
        rim_lag_slider.valueChanged.connect(_rim_lag_says)
        rim_lag_row = QHBoxLayout()
        rim_lag_row.setContentsMargins(0, 0, 0, 0)
        rim_lag_row.addWidget(rim_lag_slider, 1)
        rim_lag_row.addWidget(rim_lag_value)
        appearance.addRow(tr("Rim chase"), _hbox_wrap(rim_lag_row))

        rim_align_combo = QComboBox()
        rim_align_combo.setObjectName("RimAlignment")
        for label, key in (("Centred on the pointer", "centre"),
                           ("Trailing behind it", "head")):
            rim_align_combo.addItem(tr(label), key)
        index = rim_align_combo.findData(get_rim_alignment())
        rim_align_combo.setCurrentIndex(index if index >= 0 else 0)
        rim_align_combo.setToolTip(
            "Centred puts the middle of the lit run under the pointer; "
            "trailing puts its leading end there and drags the rest of the "
            "light behind.")
        appearance.addRow(tr("Rim alignment"), rim_align_combo)

        rim_mode_combo = QComboBox()
        rim_mode_combo.setObjectName("RimMode")
        for label, key in (("Glow", "glow"), ("Rainbow", "rainbow"),
                           ("Beat", "beat")):
            rim_mode_combo.addItem(tr(label), key)
        index = rim_mode_combo.findData(get_rim_mode())
        rim_mode_combo.setCurrentIndex(index if index >= 0 else 0)
        rim_mode_combo.setToolTip(
            "Glow is the theme's accent with a fading tail. Rainbow walks "
            "the hue along the light and turns it over time. Beat keeps the "
            "accent and pulses it. Rainbow and Beat repaint every frame; "
            "Glow only repaints when the light moves.")
        appearance.addRow(tr("Rim mode"), rim_mode_combo)

        rim_period_slider = QSlider(Qt.Horizontal)
        rim_period_slider.setObjectName("RimPeriod")
        # Stored in seconds, shown in tenths, because a slider from 0.4 to
        # 12.0 has no readable integer positions.
        rim_period_slider.setRange(int(RIM_PERIOD_RANGE[0] * 10),
                                   int(RIM_PERIOD_RANGE[1] * 10))
        rim_period_slider.setSingleStep(1)
        rim_period_slider.setPageStep(5)
        rim_period_slider.setValue(int(round(get_rim_period() * 10)))
        rim_period_slider.setToolTip(
            "How long one pulse of Beat takes, or one full turn of "
            "Rainbow's hue. Ignored by Glow, which does not animate.")
        rim_period_value = QLabel()

        def _rim_period_says(tenths):
            rim_period_value.setText(tr("%.1f s") % (int(tenths) / 10.0))

        _rim_period_says(rim_period_slider.value())
        rim_period_slider.valueChanged.connect(_rim_period_says)
        rim_period_row = QHBoxLayout()
        rim_period_row.setContentsMargins(0, 0, 0, 0)
        rim_period_row.addWidget(rim_period_slider, 1)
        rim_period_row.addWidget(rim_period_value)
        appearance.addRow(tr("Rim cycle"), _hbox_wrap(rim_period_row))

        popup_backdrop_combo = QComboBox()
        popup_backdrop_combo.setObjectName("PopupBackdrop")
        for key in POPUP_BACKDROPS:
            popup_backdrop_combo.addItem(
                tr("None") if key == "off" else tr(key.capitalize()), key)
        index = popup_backdrop_combo.findData(get_popup_backdrop())
        popup_backdrop_combo.setCurrentIndex(index if index >= 0 else 0)
        popup_backdrop_combo.setToolTip(
            "Which animation drifts behind a settings window. Separate from "
            "the module screens' own backdrop above: what belongs behind a "
            "screen of figures is not necessarily what belongs behind a form "
            "you are reading. None keeps the card and the rim and drops only "
            "the movement.")
        appearance.addRow(tr("Settings backdrop"), popup_backdrop_combo)

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

        share_diagnostics_check = Toggle(
            tr("Include redacted log excerpts in issue previews")
        )
        share_diagnostics_check.setToolTip(
            "Off by default. When enabled, the editable public-GitHub report "
            "preview includes recent log lines after paths and credentials "
            "are redacted. Nothing is submitted until you press Send on that "
            "specific report."
        )
        share_diagnostics_check.setChecked(get_share_diagnostic_logs())
        modules.addRow(tr("Report logs"), share_diagnostics_check)

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

        # How many figures stay EDITABLE, and what happens to the rest. The
        # panel keeps a live matplotlib Figure for the most recent N: those
        # can be restyled, because they still have a legend to toggle and
        # axes to rescale. Older ones keep only their rendered page.
        live_cache_spin = QSpinBox()
        live_cache_spin.setRange(MIN_FIG_LIVE_CACHE, MAX_FIG_LIVE_CACHE)
        live_cache_spin.setValue(get_figure_live_cache())
        live_cache_spin.setToolTip(
            "How many of the most recent figures keep their live figure, and "
            "so stay restylable rather than being a picture of a figure. "
            "Older ones are still shown and still on disk. Higher costs "
            "memory: a figure with a large image panel can hold tens of "
            "megabytes."
        )
        figures.addRow(tr("Editable figures kept"), live_cache_spin)

        # CELLS PER ROW IN A MONTAGE, decided rather than measured. The well
        # tab used to divide the panel width by a fixed cell size, so the
        # montage changed shape whenever the window did and two wells looked
        # at side by side were not laid out the same way.
        montage_columns_spin = QSpinBox()
        montage_columns_spin.setRange(*MONTAGE_COLUMNS_RANGE)
        montage_columns_spin.setValue(get_montage_columns())
        montage_columns_spin.setToolTip(
            "How many cells a well's montage puts on a row. The count stays "
            "the same whatever size the window is — a wider panel draws the "
            "same cells larger, up to their natural size, rather than "
            "fitting more of them in, so two wells are always laid out "
            "alike. How many ROWS fit is still measured, because that is "
            "what decides the page."
        )
        figures.addRow(tr("Cells per montage row"), montage_columns_spin)

        dynamic_check = QCheckBox()
        dynamic_check.setChecked(get_figure_dynamic())
        dynamic_check.setToolTip(
            "When you go back past the number above and select a figure, "
            "load its PDF page if one exists, so an old figure stays sharp "
            "at any zoom instead of being an enlarged screen raster. It "
            "cannot make an old figure editable again — a PDF page has no "
            "legend to toggle — but it does make it legible."
        )
        figures.addRow(tr("Dynamic figures"), dynamic_check)

        # -- HOW THE GRAPHS LOOK (instruction 118) -------------------------
        #
        # Everything above this line is about the FILE: its format, its
        # resolution, how many stay editable. Nothing above it is about how a
        # plot LOOKS -- no font, no palette, no marker size, no grid default,
        # and nothing specific to any one kind of graph -- which is what "the
        # graphs look pretty ugly" means in practice: every plot inherited
        # matplotlib's defaults.
        #
        # The panel builds itself from `spacr.figure_style`'s own tables, so a
        # style key added there gains a control here without this file being
        # touched. It stores DELTAS only; see its docstring for why that is
        # not an optimisation.
        from .widgets.figure_settings import FigureStylePreferences

        style_panel = FigureStylePreferences(get_figure_style(),
                                             get_figure_style_per_graph())
        style_heading = QLabel(tr(
            "<b>Graph style</b> — how every figure is drawn, and per graph "
            "type where they differ."))
        style_heading.setWordWrap(True)
        figures.addRow(style_heading)
        figures.addRow(style_panel)

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

        # LAPTOP MODE, which until now could only be set through an
        # environment variable -- which is to say it could not be found.
        # It sits under the mode because it answers the same question at a
        # smaller scale: how much of this machine should the WINDOW use.
        laptop_combo = QComboBox()
        laptop_combo.setObjectName("LaptopMode")
        for _key in LAPTOP_MODE_CHOICES:
            laptop_combo.addItem(tr(LAPTOP_MODE_LABELS[_key]), _key)
        _current_laptop = get_laptop_mode()
        for _i in range(laptop_combo.count()):
            if laptop_combo.itemData(_i) == _current_laptop:
                laptop_combo.setCurrentIndex(_i)
                break
        font_weight = QComboBox()
        font_weight.setObjectName("InterfaceFontWeight")
        for _key, _label in (("regular", "Regular"), ("light", "Light")):
            font_weight.addItem(tr(_label), _key)
        font_weight.setCurrentIndex(
            max(0, font_weight.findData(get_interface_font_weight())))
        appearance.addRow(tr("Interface font"), font_weight)

        performance.addRow(tr("Laptop mode"), laptop_combo)
        # THE SPACEOUT FRACTAL, and ONLY under spaceout. An ordinary launch
        # builds none of these rows, so the hidden mode stays hidden: a
        # settings page advertising it would be the giveaway.
        if spaceout_enabled():
            fractal = _page("Fractal", "PreferencesTabFractal")
            _fractal_values = get_fractal_settings()

            fractal_pattern = QComboBox()
            fractal_pattern.setObjectName("FractalPattern")
            from .widgets.fractal_travel import PATTERN_LABELS
            for _key in FRACTAL_PATTERNS:
                fractal_pattern.addItem(tr(PATTERN_LABELS.get(_key, _key)),
                                        _key)
            fractal_pattern.setCurrentIndex(
                max(0, fractal_pattern.findData(_fractal_values["pattern"])))
            # FIRST in the tab: it decides which fractal the rows below it
            # are describing, and the two have different costs.
            fractal.addRow(tr("Pattern"), fractal_pattern)

            fractal_backend = QComboBox()
            fractal_backend.setObjectName("FractalBackend")
            for _key in FRACTAL_BACKENDS:
                fractal_backend.addItem(tr(_key), _key)
            fractal_backend.setCurrentIndex(
                max(0, fractal_backend.findData(_fractal_values["backend"])))
            fractal.addRow(tr("Backend"), fractal_backend)

            fractal_note = QLabel()
            fractal_note.setObjectName("FractalBackendNote")
            fractal_note.setWordWrap(True)
            fractal.addRow("", fractal_note)

            def _sync_fractal_note(*_args):
                """Say which renderer 'auto' will actually pick HERE.

                The label cannot state it: it depends on whether vispy is
                importable on this machine, and the honest answer is the one
                the user will get.
                """
                from .widgets.fractal_travel import (
                    gpu_is_available, resolve_backend)

                chosen = fractal_backend.currentData()
                actual = resolve_backend(chosen)
                if chosen == "auto":
                    text = (f"Automatic: this machine will use the "
                            f"{actual.upper()} renderer.")
                elif chosen == "gpu" and not gpu_is_available():
                    text = ("The GPU renderer needs vispy, which is not "
                            "installed here, so the CPU renderer runs "
                            "instead. Nothing else changes when it is.")
                else:
                    text = f"The {actual.upper()} renderer."
                fractal_note.setText(text)

            fractal_backend.currentIndexChanged.connect(_sync_fractal_note)
            _sync_fractal_note()

            fractal_quality = QComboBox()
            fractal_quality.setObjectName("FractalQuality")
            for _key in FRACTAL_QUALITIES:
                fractal_quality.addItem(tr(_key), _key)
            fractal_quality.setCurrentIndex(
                max(0, fractal_quality.findData(_fractal_values["quality"])))
            fractal.addRow(tr("Quality"), fractal_quality)

            def _tenths(name, value, low, high):
                """A slider in tenths -- QSlider is integer-only."""
                box = QDoubleSpinBox()
                box.setObjectName(name)
                box.setRange(low, high)
                box.setSingleStep(0.05)
                box.setDecimals(2)
                box.setValue(float(value))
                return box

            fractal_scale = _tenths("FractalScale",
                                    _fractal_values["scale"], 0.25, 2.0)
            fractal.addRow(tr("Scale"), fractal_scale)
            fractal_speed = _tenths("FractalSpeed",
                                    _fractal_values["speed"], 0.15, 8.0)
            fractal.addRow(tr("Speed"), fractal_speed)
            fractal_dream = _tenths("FractalDream",
                                    _fractal_values["dream"], 0.0, 1.5)
            fractal.addRow(tr("Dream"), fractal_dream)

            fractal_variable = Toggle()
            fractal_variable.setObjectName("FractalVariableSpeed")
            fractal_variable.setChecked(bool(_fractal_values["variable_speed"]))
            fractal.addRow(tr("Variable speed"), fractal_variable)

            fractal_speed_min = _tenths("FractalSpeedMin",
                                        _fractal_values["speed_min"],
                                        0.15, 8.0)
            fractal.addRow(tr("Slowest"), fractal_speed_min)
            fractal_speed_max = _tenths("FractalSpeedMax",
                                        _fractal_values["speed_max"],
                                        0.15, 8.0)
            fractal.addRow(tr("Fastest"), fractal_speed_max)

            fractal_speed_period = QDoubleSpinBox()
            fractal_speed_period.setObjectName("FractalSpeedPeriod")
            fractal_speed_period.setRange(5.0, 300.0)
            fractal_speed_period.setSingleStep(5.0)
            fractal_speed_period.setDecimals(0)
            fractal_speed_period.setSuffix(tr(" s"))
            fractal_speed_period.setValue(
                float(_fractal_values["speed_period"]))
            fractal.addRow(tr("Sweep time"), fractal_speed_period)

            def _sync_variable_rows(*_args):
                """Grey the three bounds when the sweep is off.

                They are still SHOWN, so a user can see what turning it on
                would do; a hidden row is a setting nobody discovers.
                """
                on = fractal_variable.isChecked()
                for box in (fractal_speed_min, fractal_speed_max,
                            fractal_speed_period):
                    box.setEnabled(on)

            fractal_variable.toggled.connect(_sync_variable_rows)
            _sync_variable_rows()


        laptop_note_label = QLabel()
        laptop_note_label.setObjectName("LaptopModeNote")
        laptop_note_label.setWordWrap(True)
        performance.addRow("", laptop_note_label)

        def _sync_laptop_note(*_args):
            """Say what the choice will do HERE, before it is saved.

            Automatic is the case that needs it: the label cannot state the
            outcome, because the outcome depends on the machine reading it.
            """
            text = laptop_mode_note(laptop_combo.currentData())
            laptop_combo.setToolTip(text)
            laptop_note_label.setText(text)

        laptop_combo.currentIndexChanged.connect(_sync_laptop_note)
        _sync_laptop_note()
        _sync_mode_note()

        def _quit_spacr(parent) -> None:
            """Ask how, then either stop cooperatively or leave outright.

            The graceful path is the same one `closeEvent` takes --
            `cancel_all` with a short budget -- and then closes the window,
            so a normal quit still runs every shutdown hook. What this adds
            is the five-minute re-prompt, for the case `closeEvent` cannot
            handle: a worker wedged in a C extension that will never see
            the cancel flag, which leaves the window refusing to close with
            no way out from inside the application.
            """
            from .shutdown import (CANCEL, FORCE, GracefulQuitWatcher,
                                   ask_how_to_quit, describe_active,
                                   force_quit_now)

            window = parent.window() if parent is not None else None
            registry = getattr(window, "_runs", None)
            active = list(registry.active()) if registry is not None else []

            choice = ask_how_to_quit(parent, what="spaCR",
                                     detail=describe_active(active))
            if choice == CANCEL:
                return
            if choice == FORCE:
                force_quit_now()
                return

            if registry is not None:
                registry.cancel_all(reason="quit from Preferences")
            watcher = GracefulQuitWatcher(
                window,
                lambda: bool(registry is not None and registry.active()),
                what="spaCR",
                describe=lambda: describe_active(
                    list(registry.active()) if registry is not None else []),
            )
            # Parented to the window, not to the dialog: the dialog is
            # about to close and a timer that dies with it would ask
            # nothing.
            watcher.start()
            if parent is not None:
                parent.accept()
            if window is not None:
                window.close()

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
            # THE SHORT FORM ON HOVER. The confirmation still shows the full
            # bulleted promise when the button is pressed; a hint bar that
            # grew to eight lines made the dialog jump as the pointer moved
            # between two buttons.
            button.setToolTip(resource_cleanup.summary_text(action))
            button.clicked.connect(lambda: run_resource_action(action, dlg))
            performance.addRow(tr(row_label), button)
            return button

        # Hashing is a COST setting, which is why it lives here beside the
        # four that free resources rather than under Appearance. It is off
        # by default: hashing every file under every path-valued setting is
        # proportional to the data, not the run, and on a plate of raw
        # images it is minutes of reading before the first mask is made.
        hash_check = Toggle(tr("Hash inputs for the run manifest"))
        hash_check.setObjectName("HashInputsEnabled")
        hash_check.setToolTip(
            "Record a SHA-256 of every input and output file in the run "
            "manifest, so a result can be traced to the exact data and "
            "weights that produced it. Costs minutes on a large plate. The "
            "manifest is written either way and says which it was, so a run "
            "without hashes is never mistaken for one whose hashes matched."
        )
        hash_check.setChecked(get_hash_inputs())
        performance.addRow(tr("Reproducibility"), hash_check)

        # Instruction 180. On this tab and beside the hashing switch because
        # it is the same kind of decision -- how much of the machine a saved
        # run is allowed to use -- and the two are read together: a user who
        # wants a run they can hand to somebody else wants both.
        workspace_combo = QComboBox()
        workspace_combo.setObjectName("SaveWorkspaceMode")
        for value, label in (
            ("off", tr("Nothing — settings and manifest only")),
            ("reference", tr("What was open, and where its files are")),
            ("copy", tr("What was open, and copies of its files")),
        ):
            workspace_combo.addItem(label, value)
        workspace_combo.setToolTip(tr(
            "What a finished run records about the workspace around it — the "
            "databases attached, the montage, and the view built on every "
            "figure.\n\n"
            "Where its files are: the paths, sizes and checksums, so a "
            "restore can say a database moved instead of failing obscurely. "
            "Kilobytes.\n\n"
            "Copies of its files: the databases and tables as well, up to the "
            "per-file limit below. A source folder of images is tens to "
            "hundreds of gigabytes, so anything over the limit is named in "
            "the run's own record rather than copied — nothing is skipped "
            "silently.\n\n"
            "Figures the session generated are copied either way: they exist "
            "nowhere else."))
        index = workspace_combo.findData(get_save_workspace())
        workspace_combo.setCurrentIndex(max(0, index))
        performance.addRow(tr("Saved runs carry"), workspace_combo)

        workspace_limit = QSpinBox()
        workspace_limit.setObjectName("WorkspaceCopyLimitMb")
        workspace_limit.setRange(0, 1024 * 1024)
        workspace_limit.setSingleStep(64)
        workspace_limit.setSuffix(" MB")
        workspace_limit.setValue(int(get_workspace_copy_limit_mb()))
        workspace_limit.setToolTip(tr(
            "The largest single file a saved run copies in. Files over it are "
            "recorded with their size and the limit that excluded them."))
        performance.addRow(tr("Copy files up to"), workspace_limit)

        def _workspace_copying(mode: str) -> None:
            """The limit only means anything when files are being copied."""
            copying = mode == "copy"
            workspace_limit.setEnabled(copying)
            # Instruction 106: disabled and SAYING WHY, never inert.
            workspace_limit.setToolTip(workspace_limit.toolTip() if copying else tr(
                "Only used when saved runs carry copies of their files."))

        workspace_combo.currentIndexChanged.connect(
            lambda _i: _workspace_copying(str(workspace_combo.currentData())))
        _workspace_copying(str(workspace_combo.currentData()))

        _resource_button("ram", "Clear RAM", "Memory")
        _resource_button("vram", "Clear VRAM", "GPU memory")
        _resource_button("cpu", "Clear CPU", "Threads")
        _resource_button("disk", "Check disk space", "Disk")

        # Quitting belongs on this tab and not with Save/Cancel: it is the
        # last of the "this machine is not behaving" tools, next to the
        # four that free what a wedged run is holding. It is the one to
        # reach for when freeing memory was not enough.
        #
        # Deliberately NOT wired to `dlg.accept()` first. A user reaching
        # for this has a window that will not close; making them save
        # preferences on the way out would be one more thing between them
        # and leaving.
        quit_button = QPushButton(tr("Quit spaCR…"))
        quit_button.setObjectName("QuitSpacrButton")
        quit_button.setToolTip(tr(
            "Stop spaCR. You are asked whether to let running work finish "
            "the step it is on, or to stop immediately. Immediately leaves "
            "anything being written half-written."))
        from .shutdown import style_as_danger
        style_as_danger(quit_button)
        quit_button.clicked.connect(lambda: _quit_spacr(dlg))
        performance.addRow(tr("Application"), quit_button)

        outer.addWidget(tabs)

        # NO STANDING SENTENCES UNDER THE TABS. Two of them sat here on
        # every visit -- what applies instantly on Save, and when
        # colour-blind mode reaches a figure -- and a paragraph that is
        # always true of the whole dialog is not read after the first time.
        # Whatever a particular control does belongs to that control, and
        # the hint bar below says it on hover.

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
                live_cache_spin.setValue(get_figure_live_cache())
                dynamic_check.setChecked(get_figure_dynamic())
                # Told directly rather than re-read: this panel holds its
                # controls, not its store, so the throwaway-settings trick
                # every getter above uses does not reach it.
                style_panel.reset()
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
                hash_check.setChecked(get_hash_inputs())
                verbose_check.setChecked(get_verbose_logging())
                share_diagnostics_check.setChecked(
                    get_share_diagnostic_logs())
                db_edit_check.setChecked(get_db_browser_editable())
                alpha_check.setChecked(get_show_alpha())
                beta_check.setChecked(get_show_beta())
            finally:
                _settings = original

        reset_button.clicked.connect(_reset_to_defaults)

        def _save():
            # The rim first: every open card rereads these, and doing it
            # before the theme work means one repaint rather than two.
            set_rim_length(rim_length_slider.value())
            set_rim_lag(rim_lag_slider.value() / 100.0)
            set_rim_alignment(rim_align_combo.currentData())
            set_rim_mode(rim_mode_combo.currentData())
            set_rim_period(rim_period_slider.value() / 10.0)
            set_popup_backdrop(popup_backdrop_combo.currentData())
            _tell_the_cards_the_rim_changed()
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
            set_hash_inputs(hash_check.isChecked())
            # The limit FIRST: `set_save_workspace` pushes both down to
            # spacr.workspace together, so setting the mode against a stale
            # limit would leave the journal copying to the old ceiling until
            # something else happened to push again.
            set_workspace_copy_limit_mb(workspace_limit.value())
            set_save_workspace(workspace_combo.currentData())
            set_color_blind_mode(cb_combo.currentData())
            set_verbose_logging(verbose_check.isChecked())
            set_share_diagnostic_logs(share_diagnostics_check.isChecked())
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
            set_figure_live_cache(live_cache_spin.value())
            set_montage_columns(montage_columns_spin.value())
            set_figure_dynamic(dynamic_check.isChecked())
            style_general, style_per_graph = style_panel.values()
            set_figure_style(style_general)
            set_figure_style_per_graph(style_per_graph)
            # LAST of the writes, and deliberately: entering Extra
            # Performance overrides five of the settings written above with
            # their minimums, and leaving it puts back what it stashed. Do
            # it earlier and the dialog's own values would land on top,
            # which would mean the mode silently did not take effect.
            if spaceout_enabled():
                set_fractal_settings(
                    pattern=fractal_pattern.currentData(),
                    backend=fractal_backend.currentData(),
                    quality=fractal_quality.currentData(),
                    scale=fractal_scale.value(),
                    speed=fractal_speed.value(),
                    dream=fractal_dream.value(),
                    variable_speed=fractal_variable.isChecked(),
                    speed_min=fractal_speed_min.value(),
                    speed_max=fractal_speed_max.value(),
                    speed_period=fractal_speed_period.value(),
                )
            set_interface_font_weight(font_weight.currentData())
            set_laptop_mode(laptop_combo.currentData())
            set_spacr_mode(mode_combo.currentData())
            apply_preferences_to_app()
            _refresh_owner_window(parent)
            dlg.accept()

        buttons.accepted.connect(_save)
        buttons.rejected.connect(dlg.reject)
        # A LINE AT THE FOOT, THE WAY THE HOME SCREEN DOES IT. Added before
        # the rows are explained, because `explain_every_row` hands an
        # action button's sentence to whatever bar its window has -- so the
        # bar must exist by then or the button keeps a tooltip nobody
        # asked for.
        from .widgets.hint_bar import HintBar
        hints = HintBar(parent=dlg)
        # ABOVE THE BUTTONS, not under them. Asked for 2026-08-28. Appending
        # put the explanation below Defaults/Close/Open, which reads as a
        # footnote to the buttons rather than as the answer to the control
        # the pointer is on -- and puts it furthest from the tabs it
        # describes.
        layout = dlg.layout()
        row_of_buttons = layout.indexOf(buttons)
        if row_of_buttons >= 0:
            layout.insertWidget(row_of_buttons, hints)
        else:
            layout.addWidget(hints)
        # EVERY ROW EXPLAINED, ON ITS LABEL. Done here, over the finished
        # dialog, so a row added anywhere above is covered without the
        # author having to remember the rule.
        explain_every_row(dlg)
        _everything_explains_itself_in_the_strip(dlg, hints)
        return dlg


def _everything_explains_itself_in_the_strip(dialog, bar) -> int:
    """Move every remaining tooltip in ``dialog`` into ``bar``.

    :param dialog: the finished Preferences dialog.
    :param bar: its :class:`~spacr.qt.widgets.hint_bar.HintBar`.
    :returns: how many were moved, so a test can assert a number.

    THE STRIP IS THE ANSWER, NOT A SECOND ONE. Asked for 2026-08-28: "in the
    preference pannel where it says hover a controll to see what it does,
    this is where the tooltip should be, not in a tooltip window." A control
    that both writes to the strip and pops a window answers twice, and the
    window covers the strip it is duplicating.

    `explain_every_row` pairs a row's label with its field, which reached 5
    of this dialog's controls; the other 125 are labels and buttons that are
    not settings rows -- log levels, figure options, the resource actions --
    and each kept a tooltip of its own. Sweeping the finished dialog cannot
    miss a shape, including one added later.

    The strip's own label is skipped: it is the thing being written to.
    """
    from PySide6.QtWidgets import QWidget

    from .widgets.hint_bar import HintBar

    moved = 0
    for widget in dialog.findChildren(QWidget):
        if isinstance(widget, HintBar) or widget is bar:
            continue
        if not (widget.toolTip() or "").strip():
            continue
        try:
            # An empty `text` takes the widget's own tooltip and clears it,
            # so the sentence MOVES rather than being said in two places.
            if bar.explain(widget):
                moved += 1
        except Exception:                                    # noqa: BLE001
            # Help that will not move is a blemish, never a reason for
            # Preferences not to open.
            continue
    return moved


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


def _tell_the_cards_the_rim_changed() -> int:
    """Make every card on screen take the new rim settings. Returns how many.

    A PREFERENCE THE USER CANNOT SEE TAKE EFFECT is a preference they will
    set twice. The cards read length, chase and alignment when they draw,
    so all this has to do is tell them to draw -- and re-read the length,
    which is the one they cache.
    """
    try:
        from PySide6.QtWidgets import QApplication

        from .widgets.setup_card import SetupCard
    except Exception:                                        # noqa: BLE001
        return 0
    application = QApplication.instance()
    if application is None:
        return 0
    told = 0
    for widget in application.allWidgets():
        if isinstance(widget, SetupCard):
            try:
                widget.reread_the_preferences()
                told += 1
            except Exception:                                # noqa: BLE001
                LOG.debug("a card would not reread the rim", exc_info=True)
    return told


def _hbox_wrap(layout):
    from PySide6.QtWidgets import QWidget
    w = QWidget()
    w.setLayout(layout)
    return w


#: Where a panel's folded sections and divider positions live, as one JSON
#: blob keyed by panel name.
_KEY_SECTION_LAYOUT = "panels/section_layout"


def get_section_layout(panel: str) -> dict:
    """What ``panel`` looked like when it was last used.

    Divider sizes and collapsed sections are remembered per category so the
    next session restores the user's working layout.

    :returns: ``{"folded": [title, ...], "sizes": [int, ...]}``, or an empty
        dict when the panel has never been arranged. EMPTY, not a default
        layout -- the panel's own first-run arrangement is the right one, and
        freezing today's into every user's settings would make improving it
        impossible. Same reasoning as :func:`get_figure_style`.
    """
    import json

    raw = _settings().value(_KEY_SECTION_LAYOUT, "")
    if not raw:
        return {}
    try:
        stored = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    if not isinstance(stored, dict):
        return {}
    layout = stored.get(str(panel))
    return layout if isinstance(layout, dict) else {}


def set_section_layout(panel: str, folded=(), sizes=()) -> None:
    """Remember which sections of ``panel`` are folded, and the divider sizes.

    :param folded: the titles that are folded away.
    :param sizes: the splitter's sizes, in its own order.
    """
    import json

    raw = _settings().value(_KEY_SECTION_LAYOUT, "")
    try:
        stored = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        stored = {}
    if not isinstance(stored, dict):
        stored = {}
    stored[str(panel)] = {
        "folded": [str(title) for title in (folded or ())],
        "sizes": [int(size) for size in (sizes or ())],
    }
    _settings().setValue(_KEY_SECTION_LAYOUT, json.dumps(stored))


#: How wide one figure tile is drawn in the grid.
_KEY_FIGURE_GRID_SIZE = "figures/grid_cell_px"


def get_figure_grid_size() -> int:
    """The tile width the user last chose, or the grid's own default.

    A READING preference, not a property of a run: someone who wants big
    figures wants them on the next run too.
    """
    from .widgets.figure_grid_view import (MAX_CELL_PX, MIN_CELL_PX,
                                           TARGET_CELL_PX)

    raw = _settings().value(_KEY_FIGURE_GRID_SIZE, TARGET_CELL_PX)
    try:
        pixels = int(raw)
    except (TypeError, ValueError):
        return TARGET_CELL_PX
    return max(MIN_CELL_PX, min(pixels, MAX_CELL_PX))


def set_figure_grid_size(pixels: int) -> None:
    """Remember the tile width, clamped to what the grid will accept."""
    from .widgets.figure_grid_view import MAX_CELL_PX, MIN_CELL_PX

    _settings().setValue(_KEY_FIGURE_GRID_SIZE,
                         max(MIN_CELL_PX, min(int(pixels), MAX_CELL_PX)))


# ---------------------------------------------------------------------------
# Workspace content retained with a saved run
# ---------------------------------------------------------------------------

_KEY_SAVE_WORKSPACE = "runs/save_workspace"
_KEY_WORKSPACE_COPY_LIMIT = "runs/workspace_copy_limit_mb"


def get_save_workspace() -> str:
    """Return how a completed run records its open workspace.

    The result is ``"off"``, ``"reference"``, or ``"copy"``; see
    :mod:`spacr.workspace`. This application preference applies to every run
    until changed.
    """
    from ..workspace import resolve_mode

    return resolve_mode(_settings().value(_KEY_SAVE_WORKSPACE, None))


def set_save_workspace(mode) -> str:
    """Store the workspace mode and update the process-wide default.

    Updating both values makes the change available immediately to pipeline
    code that cannot read Qt settings directly.
    """
    from ..workspace import resolve_mode, set_default_mode

    resolved = resolve_mode(mode)
    _settings().setValue(_KEY_SAVE_WORKSPACE, resolved)
    set_default_mode(resolved, get_workspace_copy_limit_mb())
    return resolved


def get_workspace_copy_limit_mb() -> float:
    """The per-file ceiling on what ``copy`` mode brings in, in megabytes."""
    from ..workspace import DEFAULT_COPY_LIMIT_MB

    raw = _settings().value(_KEY_WORKSPACE_COPY_LIMIT, DEFAULT_COPY_LIMIT_MB)
    try:
        limit = float(raw)
    except (TypeError, ValueError):
        return float(DEFAULT_COPY_LIMIT_MB)
    return limit if limit >= 0 else float(DEFAULT_COPY_LIMIT_MB)


def set_workspace_copy_limit_mb(limit) -> float:
    """Remember the per-file copy limit, and push it down with the mode."""
    from ..workspace import DEFAULT_COPY_LIMIT_MB, set_default_mode

    try:
        value = float(limit)
    except (TypeError, ValueError):
        value = float(DEFAULT_COPY_LIMIT_MB)
    value = max(0.0, value)
    _settings().setValue(_KEY_WORKSPACE_COPY_LIMIT, value)
    set_default_mode(get_save_workspace(), value)
    return value


def apply_workspace_preference() -> str:
    """Push the stored preference into :mod:`spacr.workspace`. Call at startup.

    Without this the journal writes the module default on the first run of
    every session, whatever the user chose last time.
    """
    from ..workspace import set_default_mode

    return set_default_mode(get_save_workspace(), get_workspace_copy_limit_mb())


# ---------------------------------------------------------------------------
# The travelling rim
# ---------------------------------------------------------------------------
#
# The accent that runs round a settings card follows the pointer, and how it
# does that is a matter of taste rather than of correctness -- so it is three
# settings rather than three constants.

_KEY_RIM_LENGTH = "rim/length_px"
#: How many cells the montage puts on a row, per well.
_KEY_MONTAGE_COLUMNS = "montage/columns"

#: The default number of cells per row in a well's montage tab.
#:
#: A DECIDED NUMBER, not one that falls out of the window. The tab used to
#: compute `viewport_width // cell_px`, so a narrow panel showed three cells
#: per well and widening it showed more -- "the cell tab shows 3 cells per
#: well and then more if i change the size of the container". The count is
#: now the same whatever the window does, and the THUMBNAILS take up the
#: slack instead, which is the half of the geometry it makes sense to let a
#: container drive.
DEFAULT_MONTAGE_COLUMNS = 6

#: Sensible bounds. One column is a list; past a dozen the thumbnails are
#: smaller than the objects in them on any ordinary screen.
MONTAGE_COLUMNS_RANGE = (1, 12)


def get_montage_columns() -> int:
    """Cells per row in a well's montage tab."""
    low, high = MONTAGE_COLUMNS_RANGE
    try:
        value = int(_settings().value(_KEY_MONTAGE_COLUMNS,
                                      DEFAULT_MONTAGE_COLUMNS))
    except (TypeError, ValueError):
        return DEFAULT_MONTAGE_COLUMNS
    return max(low, min(high, value))


def set_montage_columns(columns) -> int:
    """Store the cells-per-row count. Returns the value actually stored."""
    low, high = MONTAGE_COLUMNS_RANGE
    try:
        value = max(low, min(high, int(columns)))
    except (TypeError, ValueError):
        value = DEFAULT_MONTAGE_COLUMNS
    _settings().setValue(_KEY_MONTAGE_COLUMNS, value)
    return value


_KEY_RIM_LAG = "rim/lag"
_KEY_RIM_ALIGNMENT = "rim/alignment"

#: How far the lit run reaches along the rim, in pixels.
DEFAULT_RIM_LENGTH = 280

#: How hard the accent chases the pointer, per frame. Smaller is slower.
DEFAULT_RIM_LAG = 0.16

#: Where the run sits relative to the pointer.
RIM_ALIGNMENTS = ("centre", "head")
DEFAULT_RIM_ALIGNMENT = "centre"

#: Bounds the settings panel and the reader both honour.
RIM_LENGTH_RANGE = (60, 900)
RIM_LAG_RANGE = (0.02, 1.0)


def get_rim_length() -> int:
    """Pixels of rim the accent lights up.

    Clamped on READ as well as on write: the stored value can come from a
    settings file written by hand or by an older build, and a rim longer
    than its own perimeter is a border rather than a highlight.
    """
    low, high = RIM_LENGTH_RANGE
    try:
        value = int(_settings().value(_KEY_RIM_LENGTH, DEFAULT_RIM_LENGTH))
    except (TypeError, ValueError):
        return DEFAULT_RIM_LENGTH
    return max(low, min(high, value))


def set_rim_length(pixels) -> int:
    """Store the rim length. Returns the value actually stored."""
    low, high = RIM_LENGTH_RANGE
    try:
        value = max(low, min(high, int(pixels)))
    except (TypeError, ValueError):
        value = DEFAULT_RIM_LENGTH
    settings = _settings()
    settings.setValue(_KEY_RIM_LENGTH, value)
    settings.sync()
    return value


def get_rim_lag() -> float:
    """How far the accent closes the gap to the pointer each frame.

    SMALLER IS SLOWER, and the name is the user's: what they see is the lag
    between the pointer arriving and the light catching up. 1.0 would put
    the light under the pointer with no travel at all, and the travel is
    the whole effect -- so that is the top of the range, not past it.
    """
    low, high = RIM_LAG_RANGE
    try:
        value = float(_settings().value(_KEY_RIM_LAG, DEFAULT_RIM_LAG))
    except (TypeError, ValueError):
        return DEFAULT_RIM_LAG
    return max(low, min(high, value))


def set_rim_lag(fraction) -> float:
    """Store the chase fraction. Returns the value actually stored."""
    low, high = RIM_LAG_RANGE
    try:
        value = max(low, min(high, float(fraction)))
    except (TypeError, ValueError):
        value = DEFAULT_RIM_LAG
    settings = _settings()
    settings.setValue(_KEY_RIM_LAG, value)
    settings.sync()
    return value


def get_rim_alignment() -> str:
    """Where the lit run sits relative to the pointer.

    ``centre`` puts the MIDDLE of the run under the pointer, ``head`` puts
    its leading end there and trails the rest behind.
    """
    value = str(_settings().value(_KEY_RIM_ALIGNMENT,
                                  DEFAULT_RIM_ALIGNMENT) or "").strip().lower()
    return value if value in RIM_ALIGNMENTS else DEFAULT_RIM_ALIGNMENT


def set_rim_alignment(name: str) -> str:
    """Store the alignment. An unknown name stores the default instead."""
    value = str(name or "").strip().lower()
    if value not in RIM_ALIGNMENTS:
        value = DEFAULT_RIM_ALIGNMENT
    settings = _settings()
    settings.setValue(_KEY_RIM_ALIGNMENT, value)
    settings.sync()
    return value


#: How the lit run of rim is coloured.
#:
#: `glow` is the accent colour with a fading tail. `rainbow` walks the hue
#: along the run so the light carries a spectrum. `beat` keeps the accent
#: colour and PULSES it, brightening and dimming on a steady cycle.
_KEY_RIM_MODE = "rim/mode"
RIM_MODES = ("glow", "rainbow", "beat")
DEFAULT_RIM_MODE = "glow"

#: Seconds for one full pulse of `beat`, or one full hue turn of `rainbow`.
_KEY_RIM_PERIOD = "rim/period_s"
DEFAULT_RIM_PERIOD = 2.4
RIM_PERIOD_RANGE = (0.4, 12.0)


def get_rim_mode() -> str:
    """Which way the rim is coloured -- glow, rainbow or beat."""
    value = str(_settings().value(_KEY_RIM_MODE,
                                  DEFAULT_RIM_MODE) or "").strip().lower()
    return value if value in RIM_MODES else DEFAULT_RIM_MODE


def set_rim_mode(name: str) -> str:
    """Store the rim mode. An unknown name stores the default instead."""
    value = str(name or "").strip().lower()
    if value not in RIM_MODES:
        value = DEFAULT_RIM_MODE
    settings = _settings()
    settings.setValue(_KEY_RIM_MODE, value)
    settings.sync()
    return value


def get_rim_period() -> float:
    """Seconds for one pulse of `beat` or one hue turn of `rainbow`."""
    low, high = RIM_PERIOD_RANGE
    try:
        value = float(_settings().value(_KEY_RIM_PERIOD, DEFAULT_RIM_PERIOD))
    except (TypeError, ValueError):
        return DEFAULT_RIM_PERIOD
    return max(low, min(high, value))


def set_rim_period(seconds) -> float:
    """Store the pulse period. Returns the value actually stored."""
    low, high = RIM_PERIOD_RANGE
    try:
        value = max(low, min(high, float(seconds)))
    except (TypeError, ValueError):
        value = DEFAULT_RIM_PERIOD
    settings = _settings()
    settings.setValue(_KEY_RIM_PERIOD, value)
    settings.sync()
    return value


#: Which animation drifts behind a settings popup.
#:
#: SEPARATE FROM THE MODULE SCREENS' OWN. A backdrop that is right behind a
#: full screen of figures is not necessarily the one somebody wants behind a
#: form they are reading; `off` keeps the card and the rim and drops only the
#: movement.
_KEY_POPUP_BACKDROP = "rim/popup_backdrop"
POPUP_BACKDROPS = ("off", "aurora", "blobs", "bokeh", "cells", "drift",
                   "ripple")
#: NO MOVING BACKDROP BEHIND A SETTINGS WINDOW unless the user asks for
#: one. The card and the rim stay either way -- 'off' drops only the
#: movement, which is what is distracting behind a form you are reading
#: rather than behind a screen of figures.
DEFAULT_POPUP_BACKDROP = "off"


def get_popup_backdrop() -> str:
    """Which ambient theme drifts behind a settings popup, or ``'off'``."""
    value = str(_settings().value(_KEY_POPUP_BACKDROP,
                                  DEFAULT_POPUP_BACKDROP) or "").strip().lower()
    return value if value in POPUP_BACKDROPS else DEFAULT_POPUP_BACKDROP


def set_popup_backdrop(name: str) -> str:
    """Store the popup backdrop. An unknown name stores the default."""
    value = str(name or "").strip().lower()
    if value not in POPUP_BACKDROPS:
        value = DEFAULT_POPUP_BACKDROP
    settings = _settings()
    settings.setValue(_KEY_POPUP_BACKDROP, value)
    settings.sync()
    return value
