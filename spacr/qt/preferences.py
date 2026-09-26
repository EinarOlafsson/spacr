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
  Ctrl+P (see :mod:`spacr.qt.shortcuts`).

Public API::

    from spacr.qt.preferences import (
        get_theme, set_theme, get_theme_choice, set_theme_choice,
        get_language, set_language,
        get_cell_variant, set_cell_variant,
        cell_background_path,
        theme_background_path,
        get_sound_music_file, set_sound_music_file,
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
        get_tooltips_enabled, set_tooltips_enabled,
        get_font_scale, set_font_scale,
        get_gui_scale, set_gui_scale,
        get_figure_save_mode, set_figure_save_mode,
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

* ``theme``: ``"dark"`` | ``"light"`` | ``"cell"`` | ``"glass"`` | one of
  the ten night themes in :data:`spacr.qt.night_themes.NIGHT_THEME_KEYS` |
  ``"system"`` (default ``"dark"``). ``"system"`` follows the operating
  system color scheme, and only once somebody has picked it: a stored
  ``"system"`` written before dark became the default (2026-09-21) was
  the old default, not a choice, and reads as ``"dark"``; see
  :func:`get_theme`. ``"cell"`` uses fluorescence imagery and ``"glass"``
  uses neutral layered materials over a built-in light field. A night
  theme also carries a backdrop and a sound set, written by
  :func:`apply_night_theme` when it is chosen. Space is not a selectable
  theme. The retired ``space_variant`` and ``space_seed`` values are
  removed from an older store the first time the theme is read; see
  :func:`get_theme`.
* ``font_scale``: float, 1.0 = 100 % (the default). Clamped to [0.10, 2.0].
* ``gui_scale``: float, 1.0 = 100 % (the default). Clamped to [0.10, 2.0].
  Scales every size of the interface, not only text, and applies live.
  See :mod:`spacr.qt.gui_scale`.
* ``figure_save_mode``: ``"print"`` | ``"screen"`` | ``"transparent"``
  (default ``"print"``). Controls the page and figure-element colours used
  for saved figures. ``SPACR_FIGURE_SAVE_MODE`` remains a process-local
  override; see :func:`spacr.figure_style.figure_save_mode`.
* ``color_blind_mode``: ``"off"`` | ``"deuteranopia"`` | ``"protanopia"``
  | ``"tritanopia"`` (default ``"off"``). Swaps matplotlib rainbow /
  red-green palettes for perceptually-uniform + colour-blind-safe
  alternatives (viridis for continuous, Okabe-Ito for categorical).
* ``performance_logging``: ``"off"`` | ``"summary"`` | ``"detailed"``
  (default ``"summary"``). Records process-tree resource use independently
  of verbose call tracing; see :func:`get_performance_logging` and
  :mod:`spacr.resource_log`.
* ``db_browser_editable``: bool, default ``False``. Permits the
  Database Browser to open a read-write connection at all; see
  :func:`get_db_browser_editable`.
* ``dock_mode``: ``"locked"`` | ``"hidden"`` (default
  ``"locked"``). Whether the left app dock reveals on hover, is pinned
  open as a permanent column, or is not there at all.
* ``pane_opacity``: int percent, default ``60``. How solid shared surfaces
  are, or the relative material strength in Glass. Clamped up to
  :func:`spacr.qt.theme.pane_alpha_floor` at paint time — the
  preference is a request, legibility is not negotiable.
* ``field_fade``: bool, default ``True``. Whether an input field's
  container and outline ramp from solid on the left to fully transparent
  on the right. Fields are **exempt from** ``pane_opacity`` while this is
  on — see :func:`get_field_fade_enabled` and
  :mod:`spacr.qt.widgets.field_fade`.
* ``show_alpha`` / ``show_beta``: bool, both default ``True``. Control
  whether modules and settings at that maturity are shown. Stable features
  are always visible.
* ``sound/music_file``: str, default ``""``. A WAV of the user's own that
  the music bed plays instead of the synthesized one, and that the
  Resonance backdrop is driven by. See :func:`get_sound_music_file`.
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

from .night_themes import NIGHT_THEME_KEYS, is_night_theme, theme_for

LOG = logging.getLogger(__name__)


_ORG = "spacr"
_APP = "qt"

_KEY_THEME       = "prefs/theme"
_KEY_THEME_FOLLOW_SYSTEM_CHOSEN = "prefs/theme_follow_system_chosen"
_KEY_LANGUAGE    = "prefs/language"
_KEY_FONT_SCALE  = "prefs/font_scale"
_KEY_GUI_SCALE   = "prefs/gui_scale"
_KEY_CB_MODE     = "prefs/color_blind_mode"
_KEY_VERBOSE_LOG = "prefs/verbose_logging"
_KEY_PERFORMANCE_LOG = "prefs/performance_logging"
_KEY_SHARE_DIAGNOSTICS = "privacy/share_diagnostic_logs"
_KEY_REFRESH_NEWS = "privacy/refresh_news_from_github"
_KEY_LOG_FILE_LEVELS = "prefs/log_file_levels"
_KEY_LOG_CONSOLE_LEVELS = "prefs/log_console_levels"
_KEY_DB_EDIT     = "prefs/db_browser_editable"
_KEY_DOCK_MODE   = "prefs/dock_mode"
_KEY_DOCK_WIDTH  = "prefs/dock_width"
_KEY_RUNTIME_TEXT_SCALE = "prefs/runtime_text_scale"
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
#: The two tooltip surfaces. INDEPENDENT: both on,
#: both off, or either alone are all legal, which is why they are two
#: booleans and not a three-way choice wearing two checkboxes.
_KEY_TOOLTIPS_BOX = "prefs/tooltips_box"
_KEY_OBJECT_GRID = "prefs/object_settings_grid"
_KEY_TOOLTIPS_BOTTOM = "prefs/tooltips_bottom"
#: The master switch over every ordinary Qt tooltip in the application --
#: buttons, toolbars, table headers, the lot. It is not one of the two
#: SETTINGS surfaces above: those answer "what is this setting", this one
#: answers "do small labels pop up at all".
_KEY_TOOLTIPS_ENABLED = "prefs/tooltips_enabled"
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

#: What a fractal number must satisfy to be usable, as
#: ``name -> (floor, ceiling, why)``. ``None`` for a bound means there is
#: none.
#:
#: These are validation bounds, not display-field caps. A field accepts the
#: typed value and :func:`explain_a_fractal_number` says plainly when it
#: cannot be used and why; it never silently changes 8000 to 8.
#:
#: A bound is here only where a value outside it CANNOT WORK -- a
#: supersampling of 0 takes no samples, a scale of 0 renders nothing, a
#: negative iteration count is not a count. Values that are merely
#: extravagant are the user's business.
def _mandelbrot_defaults() -> dict:
    """What the fractal settings start at, before anyone chooses a level.

    THE SHIPPED DEFAULTS ARE THE LIGHT ONES, keeping first launch responsive
    on modest hardware.

    So the numbers that decide COST -- supersampling, render scale and the
    iteration budget -- come from the `balanced` preset rather than from the
    Mandelbrot renderer's published set, which is a `high` profile. The
    published numbers are still what the Mandelbrot pattern documents and
    what choosing High restores; they are not what a user who has chosen
    nothing gets.

    Everything that does not cost anything -- the steering behaviour, the
    starting scale, the precision the reference orbit is built to -- keeps
    the published value, because making those timid would change what the
    pattern IS rather than how hard it works.
    """
    try:
        from .widgets.fractal_mandelbrot import DEFAULTS

        return dict(DEFAULTS)
    except Exception:                                        # noqa: BLE001
        return {"supersampling": 2, "seconds_per_decade": 24.0,
                "base_iterations": 300, "iterations_per_decade": 55.0,
                "max_iterations": 2200, "precision_digits": 320,
                "initial_scale": 1.25, "zoom_rate": 1.0,
                "render_scale": 1.0, "steering_strength": 0.09,
                "steering_interval_decades": 0.40,
                "steering_duration": 3.8, "candidate_count": 24,
                "max_depth": 21.0}


class _LazyDefaults(dict):
    """The Mandelbrot defaults, resolved on first use.

    NOT AT IMPORT. `_mandelbrot_defaults` reaches into
    `spacr.qt.widgets.fractal_mandelbrot`, and importing that package pulls
    QtWidgets in -- which `test_preferences_imports_without_touching_the_
    ambient_widget` exists to prevent, because this module is imported by
    headless paths that must never build a widget toolkit.

    A dict subclass rather than a function, so every existing
    `_MANDEL_DEFAULTS[name]` and `.get(name)` reads the same as before.
    """

    _loaded = False

    def _load(self) -> None:
        """Fill the defaults once, on first read.

        The flag is set BEFORE the work so a failure part-way through does not
        leave every subsequent read retrying an import that already failed.
        """
        if self._loaded:
            return
        self._loaded = True
        try:
            self.update(_mandelbrot_defaults())
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not read the fractal defaults", exc_info=True)

    def __getitem__(self, key):
        """Force the load, then read as a normal dict."""
        self._load()
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        """Read one default, loading the table on first use.

        :param key: the setting name.
        :param default: what to return when it has none.
        :returns: the default value.
        """
        self._load()
        return dict.get(self, key, default)

    def __contains__(self, key) -> bool:
        """Force the load, then answer as a normal dict.

        Every access point forces it, not just ``__getitem__``: a caller testing
        ``in`` before reading would otherwise see an empty mapping and conclude
        the key does not exist.
        """
        self._load()
        return dict.__contains__(self, key)

    def keys(self):
        """Return the setting names, loading the table on first use."""
        self._load()
        return dict.keys(self)

    def items(self):
        """Return the name/default pairs, loading the table on first use."""
        self._load()
        return dict.items(self)


_MANDEL_DEFAULTS = _LazyDefaults()

FRACTAL_LIMITS = {
    "scale": (0.01, None,
              "a scale of zero or less renders nothing at all"),
    "speed": (0.0, None, "speed cannot run backwards"),
    "dream": (0.0, None, "dream is an amount, and cannot be negative"),
    "speed_min": (0.0, None, "speed cannot run backwards"),
    "speed_max": (0.0, None, "speed cannot run backwards"),
    "speed_period": (0.1, None,
                     "a period of zero would change speed infinitely fast"),
    "pointer_size": (0.0, None, "a reach cannot be negative"),
    "pointer_strength": (0.0, None, "a strength cannot be negative"),
    "supersampling": (1, None,
                      "fewer than one sample a pixel draws nothing"),
    "seconds_per_decade": (0.1, None,
                           "a decade cannot take no time at all"),
    "base_iterations": (1, None, "a frame needs at least one iteration"),
    "iterations_per_decade": (0.0, None,
                              "iterations cannot be taken away as you "
                              "descend; the picture would go solid"),
    "max_iterations": (1, 4096,
                       "the shader's loop is bounded at 4096, so a larger "
                       "number would be silently ignored"),
    "precision_digits": (16, None,
                         "below about sixteen digits the reference orbit "
                         "is no better than the float it is meant to "
                         "rescue"),
    "initial_scale": (0.0000001, None,
                      "a starting scale of zero has nothing to zoom out of"),
    "tile_rows": (1, None, "a tile needs at least one row"),
    "render_scale": (0.05, None,
                     "below about a twentieth there are not enough pixels "
                     "to see"),
    "fps": (1, None, "a frame rate of zero never draws"),
    "zoom_rate": (0.0, None, "the zoom cannot run backwards"),
    "steering_strength": (0.0, None, "a strength cannot be negative"),
    "steering_interval_decades": (0.01, None,
                                  "steering every zero decades would never "
                                  "stop choosing a new target"),
    "steering_duration": (0.1, None,
                          "a move that takes no time is a jump"),
    "candidate_count": (1, None,
                        "choosing between no candidates chooses nothing"),
    "steering": (0.0, 1.0,
                 "steering is an amount between none and restless"),
    "max_depth": (0.1, 23.0,
                  "the reference orbit is carried as three float32s and "
                  "reproduces Z to about 4.2e-24, so past roughly "
                  "twenty-three decades the perturbation is measuring noise "
                  "and the picture turns to mush"),
}


#: Every setting that is really about SPEED, and what one Speed of 1.0
#: means for each.
#:
#: One control drives the related timing values, preventing separate fields
#: that answer the same question from contradicting one another.
#:
#: Speed multiplies the first four and DIVIDES seconds-per-decade, because
#: that one is a duration: a bigger number there is a slower dive, and a
#: control called Speed that made things slower as it rose would be a trap.
SPEED_GROUP = {
    "speed": 1.0,
    "zoom_rate": 1.0,
    "speed_min": 0.55,
    "speed_max": 1.65,
    "speed_period": 41.0,
}

#: The duration that Speed divides rather than multiplies.
SPEED_SECONDS_PER_DECADE: float = 24.0

#: Every setting that is really about SCALE -- how much is drawn, and how
#: finely -- and what one Scale of 1.0 means for each.
#:
#: Supersampling is a whole number of samples a side, so it steps rather
#: than scaling smoothly: below 1.0 it is 1, and it reaches 2 and 3 as the
#: control rises. The cost is its square, which is why it is the last thing
#: to go up.
SCALE_GROUP = {
    "scale": 1.0,
    "render_scale": 1.0,
}


def speed_group_values(speed: float) -> dict:
    """What one Speed means for every setting that follows it.

    :param speed: the single user-facing number.
    :returns: ``{setting: value}`` for the whole group.
    """
    amount = max(0.01, float(speed))
    values = {name: round(base * amount, 4)
              for name, base in SPEED_GROUP.items()}
    values["seconds_per_decade"] = round(
        SPEED_SECONDS_PER_DECADE / amount, 4)
    return values


def scale_group_values(scale: float) -> dict:
    """What one Scale means for every setting that follows it.

    :param scale: the single user-facing number.
    :returns: ``{setting: value}`` for the whole group.
    """
    amount = max(0.05, float(scale))
    values = {name: round(base * amount, 4)
              for name, base in SCALE_GROUP.items()}
    values["supersampling"] = 1 if amount < 1.5 else (2 if amount < 2.5
                                                      else 3)
    return values


def explain_a_fractal_number(name: str, value) -> str:
    """Why ``value`` cannot be used for ``name``, or ``""`` if it can.

    :param name: a fractal setting name.
    :param value: whatever the field holds.
    :returns: a sentence for the user, empty when the value is fine.

    THE FIELD TAKES ANYTHING; this is what decides whether it WORKS. The
    message names the setting, the value and the reason, because "invalid
    input" tells a user only that the software disagrees with them.
    """
    floor, ceiling, why = FRACTAL_LIMITS.get(name, (None, None, ""))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return f"{name}: {value!r} is not a number."
    if number != number:
        return f"{name}: not a number."
    if floor is not None and number < floor:
        return (f"{name}: {value} is too small (needs at least {floor}) "
                f"— {why}.")
    if ceiling is not None and number > ceiling:
        return (f"{name}: {value} is too large (at most {ceiling}) "
                f"— {why}.")
    return ""


MAX_FRACTAL_SPEED: float = 1_000_000.0

#: Whether the pointer pulls the backdrop about at all.
_KEY_FRACTAL_POINTER = "spaceout/fractal_pointer_gravity"
#: How far that pull reaches, and how hard it pulls.
_KEY_FRACTAL_POINTER_SIZE = "spaceout/fractal_pointer_size"
_KEY_FRACTAL_POINTER_STRENGTH = "spaceout/fractal_pointer_strength"

#: Supersampling, and the Mandelbrot renderer's own numbers.
_KEY_FRACTAL_SUPERSAMPLING = "spaceout/fractal_supersampling"
_KEY_FRACTAL_SECONDS_PER_DECADE = "spaceout/fractal_seconds_per_decade"
_KEY_FRACTAL_BASE_ITERATIONS = "spaceout/fractal_base_iterations"
_KEY_FRACTAL_ITERATIONS_PER_DECADE = "spaceout/fractal_iterations_per_decade"
_KEY_FRACTAL_MAX_ITERATIONS = "spaceout/fractal_max_iterations"
_KEY_FRACTAL_PRECISION_DIGITS = "spaceout/fractal_precision_digits"
_KEY_FRACTAL_INITIAL_SCALE = "spaceout/fractal_initial_scale"
_KEY_FRACTAL_ZOOM_RATE = "spaceout/fractal_zoom_rate"
_KEY_FRACTAL_RENDER_SCALE = "spaceout/fractal_render_scale"
_KEY_FRACTAL_STEERING_STRENGTH = "spaceout/fractal_steering_strength"
_KEY_FRACTAL_STEERING_INTERVAL_DECADES = "spaceout/fractal_steering_interval_decades"
_KEY_FRACTAL_STEERING_DURATION = "spaceout/fractal_steering_duration"
_KEY_FRACTAL_CANDIDATE_COUNT = "spaceout/fractal_candidate_count"
_KEY_FRACTAL_MAX_DEPTH = "spaceout/fractal_max_depth"
#: The one user-facing steering control, 0..1.
_KEY_FRACTAL_STEERING = "spaceout/fractal_steering"
#: "fixed" descends to one point; "guided" searches as it goes; "tour"
#: floats between the twenty mapped regions.
_KEY_FRACTAL_PATH = "spaceout/fractal_path"

#: The memory budget: how long an unused thing may sit, how much may be
#: held, and how much of the machine must stay free for everything else.
_KEY_IDLE_MINUTES = "prefs/cache_idle_minutes"
_KEY_CACHE_CEILING = "prefs/cache_ceiling_mb"
_KEY_HEADROOM = "prefs/headroom_mb"
_KEY_FRACTAL_SPEED_PERIOD = "spaceout/fractal_speed_period"
#: Where the visual settings Extra Performance overrode are kept, so
#: leaving that mode gives the user back exactly what they had.
_KEY_MODE_VISUAL_STASH = "prefs/mode_visual_stash"

#: Themes with a palette of their own — mirrors
#: :data:`spacr.qt.theme.THEMES`, restated here so importing this module
#: does not pull in QtGui/QtWidgets.
#:
#: The ten night themes are appended from :mod:`spacr.qt.night_themes`
#: rather than written out again, because that module is Qt-free for
#: exactly this reason: it can be imported here without QtGui, and it
#: cannot then drift from what :data:`spacr.qt.theme.THEMES` holds.
PALETTE_THEMES = ("dark", "light", "cell", "glass") + NIGHT_THEME_KEYS

#: Persisted values. An existing install has ``prefs/theme`` set to one
#: of dark/light/system/space; those keep resolving exactly as before,
#: and an unrecognised value (hand-edited INI, a downgrade from a build
#: with more themes) falls back to :data:`DEFAULT_THEME` rather than
#: raising.
VALID_THEMES = PALETTE_THEMES + ("system",)
#: Dark on every platform until somebody chooses otherwise (maintainer,
#: 2026-09-21: start spaCR dark by default, and the setup screen too). The
#: operating system's own light or dark setting does not override it; only
#: an explicit "Follow system" choice does.
DEFAULT_THEME = "dark"

#: RETIRED 2026-09-09. Kept as names only so `_forget_the_space_theme_keys`
#: can remove a stored value; nothing reads them. See the note above
#: `theme_background_path`.
_KEY_SPACE_VARIANT = "prefs/space_variant"
_KEY_SPACE_SEED    = "prefs/space_seed"
#: Store files already cleared of the two keys above in this process.
_SPACE_KEYS_CLEARED: set = set()
_KEY_CELL_VARIANT  = "prefs/cell_variant"

FONT_SCALE_MIN = 0.10
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

#: The whole-GUI scale's bounds and default (item 471): the same 10 % to
#: 200 % as Zoom, so the two sliders read alike. Applied live by the
#: scaling layer in :mod:`spacr.qt.gui_scale`, which says how.
GUI_SCALE_MIN = 0.10
GUI_SCALE_MAX = 2.00
DEFAULT_GUI_SCALE = 1.0

#: What the GUI scale row says it is, including the way back.
GUI_SCALE_TIP = (
    "Scale every part of the interface -- widget sizes, spacing, icons, "
    "figures and text -- from 10 % to 200 %. Lower it to fit more on a "
    "small or low-resolution screen. It applies straight away and then "
    "asks whether to keep it; with no answer it goes back by itself after "
    "15 seconds. Font scale applies on top: 50 % GUI at 200 % font gives "
    "half-size controls with text the usual size. Ctrl+Alt+0 puts GUI "
    "scale and font scale back to 100 % from anywhere.")

VALID_CB_MODES = ("off", "deuteranopia", "protanopia", "tritanopia")
DEFAULT_CB_MODE = "off"

_KEY_FIG_FORMAT = "prefs/figure_format"
_KEY_FIG_PNG_DPI = "prefs/figure_png_dpi"
_KEY_FIG_SAVE_MODE = "prefs/figure_save_mode"
VALID_FIG_FORMATS = ("png", "pdf")
DEFAULT_FIG_FORMAT = "pdf"
VALID_PNG_DPIS = (100, 200, 300, 600, 1200)
DEFAULT_PNG_DPI = 300
VALID_FIG_SAVE_MODES = ("print", "screen", "transparent")
DEFAULT_FIG_SAVE_MODE = "print"

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
        """Hold the real store, which is where WRITES still go."""
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



def get_language() -> str:
    """Return the persisted UI language code, falling back to English."""
    from .i18n import DEFAULT_LANGUAGE, normalize_language
    raw = _settings().value(_KEY_LANGUAGE, DEFAULT_LANGUAGE)
    return normalize_language(raw)


def set_language(language: str) -> None:
    """Persist one of the bundled UI languages.

    :param language: a UI language code from
        :data:`spacr.qt.i18n.VALID_LANGUAGE_CODES`; stripped, with ``-`` read
        as ``_``.
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
    """Which panels the user left folded, ``{key: True}``, or opened, ``{key: False}``.

    False is stored only for a panel that starts folded; see
    :func:`set_folded_panel`.

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


def set_folded_panel(key: str, shut: bool, *, default_shut: bool = False) -> None:
    """Remember that ``key`` is folded, or is not.

    A PANEL IN ITS DEFAULT STATE IS REMOVED rather than stored. Most panels
    default to open, so storing that would grow the dict by one entry for
    every panel the user has ever touched and never shrink it. A panel that
    starts folded (item 509: the advanced PSF, restoration and CLAHE rows)
    passes ``default_shut=True``, so opening it is what gets stored, as
    False, and folding it again forgets it.

    :param key: the panel key, ``"<module>/<panel>"``; stripped, and an empty
        key does nothing.
    :param shut: true to record the panel as folded, false as open.
    :param default_shut: the panel's state when nothing is stored.
    """
    import json

    key = str(key or "").strip()
    if not key:
        return
    state = get_folded_panels()
    if bool(shut) == bool(default_shut):
        state.pop(key, None)
    else:
        state[key] = bool(shut)
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
    """Store the general figure settings.

    :param style: the general figure settings, ``{setting: value}``; stored as
        JSON, and ``None`` stores an empty dict.
    """
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
    """Store the per-graph overrides.

    :param overrides: per-graph overrides, ``{kind: {setting: value}}``;
        entries whose value is not a non-empty dict are dropped before storing.
    """
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
#: Which graph is drawn FIRST, per data shape. ``{shape: graph_type}``.
#:
#: Regression lets the user right-click to change a drawn graph; this mapping
#: also chooses what is drawn before the first right-click. Stored per shape
#: rather than as one value
#: because "Bar" is not an answer for two continuous axes -- a bar needs
#: groups to summarise, and there are none -- so a single setting would be
#: ignored by most graphs and look broken.
_KEY_DEFAULT_GRAPH_TYPES = "figures/default_graph_types"


def get_default_graph_types() -> dict:
    """Every saved default graph type, as ``{shape: graph_type}``.

    :returns: the saved mapping, empty when nothing has been chosen.
    """
    import json

    raw = _settings().value(_KEY_DEFAULT_GRAPH_TYPES, "")
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items() if v}
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        loaded = json.loads(raw)
    except ValueError:
        return {}
    if not isinstance(loaded, dict):
        return {}
    return {str(k): str(v) for k, v in loaded.items() if v}


def get_default_graph_type(shape: str) -> str:
    """The graph type the user wants drawn first for ``shape``.

    :param shape: a `spacr.graph_types` data shape.
    :returns: the saved graph type, or ``""`` when none is saved.

    Empty rather than the table's default, so `graph_types.default_for` can
    tell "the user chose this" from "nothing was chosen" -- the table moves
    when the package does, and a stored copy of it is a preference that has
    stopped tracking.
    """
    return str(get_default_graph_types().get(str(shape), ""))


def set_default_graph_type(shape: str, graph_type: str) -> None:
    """Persist which graph is drawn first for ``shape``.

    :param shape: a `spacr.graph_types` data shape.
    :param graph_type: a graph type, or ``""`` to go back to the default.
    """
    import json

    saved = get_default_graph_types()
    if graph_type:
        saved[str(shape)] = str(graph_type)
    else:
        saved.pop(str(shape), None)
    settings = _settings()
    settings.setValue(_KEY_DEFAULT_GRAPH_TYPES, json.dumps(saved))
    settings.sync()


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

    :param kind: the figure kind, e.g. ``"volcano"``, as
        :func:`spacr.style_base.style_kind` derives it; converted to ``str``.
    """
    return dict(get_figure_style_defaults().get(str(kind), {}))


def set_figure_style_default(kind: str, values) -> None:
    """Make ``values`` the default for every future figure of ``kind``.

    The design: "a per-project default so a lab's house style is
    applied to every figure of that type without re-setting it each time".

    :param kind: the figure kind, e.g. ``"volcano"``, as
        :func:`spacr.style_base.style_kind` derives it; converted to ``str``.
    :param values: the ``{field: value}`` style to save for that kind,
        replacing any previous default; ``None`` saves an empty dict.
    """
    import json

    stored = get_figure_style_defaults()
    stored[str(kind)] = dict(values or {})
    settings = _settings()
    settings.setValue(_KEY_FIG_STYLE_DEFAULTS, json.dumps(stored))
    settings.sync()


def clear_figure_style_default(kind: str) -> bool:
    """Forget the default for ``kind``. True if there was one.

    The way back, and it is not optional: a default that can only be set is
    the same trap as a colour that can only be set.

    :param kind: the figure kind, e.g. ``"volcano"``, as
        :func:`spacr.style_base.style_kind` derives it; converted to ``str``.
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

    :param fmt: the figure file format, one of :data:`VALID_FIG_FORMATS`.
    :raises ValueError: if ``fmt`` is not ``png`` or ``pdf``.
    """
    if fmt not in VALID_FIG_FORMATS:
        raise ValueError(f"unknown figure format {fmt!r}. "
                          f"Choose from {VALID_FIG_FORMATS}.")
    _settings().setValue(_KEY_FIG_FORMAT, fmt)


def get_figure_save_mode() -> str:
    """Return the saved figure appearance mode, defaulting to ``print``.

    The environment override belongs to
    :func:`spacr.figure_style.figure_save_mode`, not here. Keeping this getter
    store-only lets the Preferences dialog show what it will persist even
    while a command-line or notebook process temporarily overrides it.

    Returns
    -------
    {'print', 'screen', 'transparent'}
        Persisted mode, or ``print`` when the stored value is invalid.
    """
    raw = str(_settings().value(
        _KEY_FIG_SAVE_MODE, DEFAULT_FIG_SAVE_MODE)).strip().lower()
    return (raw if raw in VALID_FIG_SAVE_MODES
            else DEFAULT_FIG_SAVE_MODE)


def set_figure_save_mode(mode: str) -> None:
    """Persist the saved-figure appearance mode.

    Parameters
    ----------
    mode : {'print', 'screen', 'transparent'}
        ``print`` writes a light page with dark figure elements, ``screen``
        preserves the displayed appearance, and ``transparent`` removes the
        page background.

    Raises
    ------
    ValueError
        If *mode* is not a supported figure save mode.
    """
    normalized = str(mode).strip().lower()
    if normalized not in VALID_FIG_SAVE_MODES:
        raise ValueError(
            f"unknown figure save mode {mode!r}. "
            f"Choose from {VALID_FIG_SAVE_MODES}.")
    _settings().setValue(_KEY_FIG_SAVE_MODE, normalized)


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

    :param count: how many of the most recent figures keep their live Figure;
        converted to ``int``.
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
    """Persist whether evicted figures reload from their vector page.

    :param enabled: true to reload an evicted figure from its vector page,
        false to keep showing its raster; stored as a ``bool``.
    """
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

    :param dpi: the PNG resolution in dots per inch; converted to ``int`` and
        checked against :data:`VALID_PNG_DPIS`.
    :raises ValueError: if ``dpi`` is not a supported resolution.
    """
    dpi = int(dpi)
    if dpi not in VALID_PNG_DPIS:
        raise ValueError(
            f"unknown PNG resolution {dpi!r}. Choose from {VALID_PNG_DPIS}."
        )
    _settings().setValue(_KEY_FIG_PNG_DPI, dpi)


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
    """Whether ``bg`` means "let whatever is behind show through".

    :param bg: a background colour token; ``"none"``, ``"transparent"`` and the
        empty string (after stripping and lower-casing) mean transparent.
    """
    return str(bg).strip().lower() in {"none", "transparent", ""}


def figure_color_is_auto(token) -> bool:
    """Whether ``token`` is the "follow the theme" token rather than a colour.

    Matching is case- and space-insensitive because tokens can come from a
    hand-edited INI file or the dialog.

    :param token: a stored colour token or colour string; converted to ``str``
        before comparing with :data:`AUTO_FIGURE_COLOR`.
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
            return
        bg = str(settings.value(_KEY_FIG_BG, AUTO_FIGURE_COLOR))
        fg = str(settings.value(_KEY_FIG_FG, AUTO_FIGURE_COLOR))
        if figure_color_is_auto(bg) or figure_color_is_auto(fg):
            return
        if bg.lower() not in _FROZEN_BG_VALUES:
            return
        if fg.lower() not in _FROZEN_FG_VALUES:
            return
        if (bg.lower(), fg.lower()) == tuple(
                str(v).lower() for v in auto_figure_colors()):
            return
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
    "follow the text", never what it resolved to.

    :param token: the line colour token, or :data:`AUTO_FIGURE_COLOR` to follow
        the text colour.
    """
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

    :param bg: the background colour token, or :data:`AUTO_FIGURE_COLOR`.
    :param fg: the text colour token, or :data:`AUTO_FIGURE_COLOR`.
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
    set_figure_line_colour(AUTO_FIGURE_COLOR)


def get_figure_text_size() -> int:
    """Return the saved figure font size; zero leaves Matplotlib unchanged."""
    try:
        return int(_settings().value(_KEY_FIG_TEXT_SIZE, 0))
    except (TypeError, ValueError):
        return 0


def set_figure_text_size(size: int) -> None:
    """Persist a figure font size; zero delegates sizing to Matplotlib.

    :param size: the figure font size; converted to ``int``, and 0 leaves
        sizing to Matplotlib.
    """
    _settings().setValue(_KEY_FIG_TEXT_SIZE, int(size))



def _forget_the_space_theme_keys(store) -> None:
    """Remove the retired Space theme's two keys from ``store``, once.

    ``prefs/space_variant`` and ``prefs/space_seed`` chose between skies for
    a theme :data:`VALID_THEMES` does not offer, and nothing has read either
    since their accessors were retired on 2026-09-09. A store written before
    then still holds them, so they are removed here and the file stops
    carrying values that look live and are not.

    Runs once per store file per process, and writes only when there was
    something to remove. Skipped in safe mode, which reads nothing it was
    given; the next ordinary start removes them. Never raises: a key that
    cannot be removed is a stale line in the store and changes nothing.

    :param store: the preference store :func:`get_theme` is reading.
    :returns: None; ``store`` is edited in place.
    """
    if _SAFE_MODE or not isinstance(store, QSettings):
        return
    try:
        name = str(store.fileName())
        if name in _SPACE_KEYS_CLEARED:
            return
        gone = [key for key in (_KEY_SPACE_VARIANT, _KEY_SPACE_SEED)
                if store.contains(key)]
        for key in gone:
            store.remove(key)
        if gone:
            store.sync()
            LOG.info("removed %s from the preferences: they belonged to the "
                     "retired Space theme and nothing reads them.",
                     " and ".join(gone))
        _SPACE_KEYS_CLEARED.add(name)
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not remove the retired Space theme keys",
                  exc_info=True)


def _follow_system_was_chosen(store) -> bool:
    """Whether a stored ``"system"`` theme was somebody's choice.

    Until 2026-09-21 ``"system"`` was the default, and both the setup
    screen and Preferences write the value their Theme control shows when
    they are saved. So a store holding ``"system"`` without this flag was
    most likely written by a user who never touched the control, so the
    default is dark unless this flag records a choice. :func:`set_theme`
    sets the flag whenever ``"system"`` is chosen from now on.

    :param store: the preference store being read.
    :returns: ``True`` only when the flag is present and true.
    """
    return _as_bool(store.value(_KEY_THEME_FOLLOW_SYSTEM_CHOSEN, False),
                    False)


def get_theme() -> str:
    """Return the saved application theme, or the default when invalid.

    A stored ``"system"`` counts only when it was chosen (see
    :func:`_follow_system_was_chosen`); otherwise it reads as
    :data:`DEFAULT_THEME`, which is dark. An explicit Light, or any other
    stored theme, is returned as it is.

    The first read of a store also removes the retired Space theme's
    ``space_variant`` and ``space_seed`` values from it; see
    :func:`_forget_the_space_theme_keys`.
    """
    store = _settings()
    _forget_the_space_theme_keys(store)
    raw = str(store.value(_KEY_THEME, DEFAULT_THEME))
    if raw == "system" and not _follow_system_was_chosen(store):
        raw = DEFAULT_THEME
    return raw if raw in VALID_THEMES else DEFAULT_THEME


def set_theme(theme: str) -> None:
    """Persist a supported application theme.

    Choosing ``"system"`` also records that it was chosen, so
    :func:`get_theme` honours it instead of reading it as the default.

    :param theme: one of :data:`VALID_THEMES`.
    :raises ValueError: if ``theme`` is not in :data:`VALID_THEMES`.
    """
    if theme not in VALID_THEMES:
        raise ValueError(f"unknown theme {theme!r}. "
                          f"Choose from {VALID_THEMES}.")
    store = _settings()
    store.setValue(_KEY_THEME, theme)
    if theme == "system":
        store.setValue(_KEY_THEME_FOLLOW_SYSTEM_CHOSEN, True)
    else:
        store.remove(_KEY_THEME_FOLLOW_SYSTEM_CHOSEN)


def theme_choices() -> tuple:
    """Return ``(label, token)`` choices for the single Theme control.

    Image variants are represented as composite tokens in the UI while the
    persisted keys remain backward compatible.

    The ten night themes come last, in
    :data:`spacr.qt.night_themes.NIGHT_THEMES` order, so the four the
    application has always had stay where a returning user looks for them
    and the new family reads as one block down the bottom of the list.
    Their tokens are their plain keys: a night theme has no variant, so
    there is nothing to compose into the token the way Cell does.
    """
    from .imagery import CELL_VARIANTS, title_for
    from .night_themes import NIGHT_THEMES

    choices = [
        ("Dark", "dark"),
        ("Light", "light"),
        ("Glass", "glass"),
        ("Follow system", "system"),
    ]
    choices.extend(
        (title_for(key), f"cell:{key}")
        for key in CELL_VARIANTS
    )
    choices.extend((theme.label, key) for key, theme in NIGHT_THEMES.items())
    return tuple(choices)


def theme_description(token: str) -> str:
    """Return the one-sentence explanation of a :func:`theme_choices` token.

    The ten night themes each carry a sentence saying what colours,
    backdrop and sound set come with them; that sentence is what the
    Theme control shows as the entry's tooltip, the way the Sound set
    control shows :attr:`spacr.qt.sound_synth.SoundTheme.description`.

    :param token: a token from :func:`theme_choices`.
    :returns: the theme's sentence, or ``""`` for the four themes that
        predate the family and for any token without one. The caller
        passes the result through :func:`tr` and sets no tooltip when it
        is empty.
    """
    from .night_themes import NIGHT_THEMES

    theme = NIGHT_THEMES.get(token)
    return theme.description if theme is not None else ""


def get_theme_choice() -> str:
    """Return the composite token representing the current visual theme."""
    theme = get_theme()
    if theme == "cell":
        return f"cell:{get_cell_variant()}"
    return theme


def set_theme_choice(choice: str) -> None:
    """Persist one token from :func:`theme_choices`.

    Choosing one of the ten night themes also writes that theme's
    backdrop and its sound set — see :func:`apply_night_theme`, which is
    where the reasoning for doing so lives.

    :param choice: a token from :func:`theme_choices`; a ``"cell:<variant>"``
        token sets the Cell theme and that variant. Any other value raises
        :class:`ValueError`.
    """
    valid = {token for _label, token in theme_choices()}
    if choice not in valid:
        raise ValueError(
            f"unknown theme choice {choice!r}. Choose from {sorted(valid)}.")
    if choice.startswith("cell:"):
        set_cell_variant(choice.split(":", 1)[1])
        set_theme("cell")
    else:
        set_theme(choice)
        if is_night_theme(choice):
            apply_night_theme(choice)


def apply_night_theme(name: str) -> None:
    """Write the backdrop and the sound set a night theme comes with.

    A night theme is one choice that moves three things: the colours, the
    animation behind them and the set of sounds spaCR would play. So this
    writes the ambient animation, the ambient palette and the sound set
    that go with the colours.

    IT DOES NOT SWITCH ANYTHING ON. The sound master stays exactly where
    the user left it, which on a fresh install and on every install that
    has never opened the Sound tab is off; all this decides is *which*
    set would play if it were ever switched on. It leaves the animation
    master alone in the same way: if the backdrop is off, the stored
    animation is what comes back when it is switched on again, and
    :func:`set_ambient_animation` is not used here for exactly that
    reason -- that setter also switches the backdrop on.

    IT IS A PRESET, NOT AN OVERRIDE. The three values are written once,
    at the moment the theme is chosen, into the same keys the Animation
    and Sound controls read and write. Nothing re-imposes them, so a user
    who picks Nocturne and then changes the animation to Bokeh keeps
    Bokeh.

    AND IT NEVER SWITCHES THE BACKDROP BACK ON. "No animation" is stored
    as the animation NAME (:data:`spacr.qt.widgets.ambient.NO_ANIMATION`),
    which :func:`get_ambient_enabled` reads, so writing an animation over
    it would hand a moving backdrop to a user who had turned motion off
    -- through the Animation control, or through Extra Performance, which
    turns it off the same way. :func:`backdrop_is_switched_off` is
    therefore asked first, and when it says yes the two ambient keys are
    left exactly as they are. The theme still changes the colours and the
    sound set; it just does not start anything moving. What that costs is
    small and worth saying: a user who later switches the backdrop on
    gets the animation they had before, not the one this theme would have
    brought, and they can pick it on the same control they just used.

    :param name: one of :data:`spacr.qt.night_themes.NIGHT_THEME_KEYS`.
    :raises KeyError: if ``name`` is not one of the ten.
    """
    theme = theme_for(name)
    settings = _settings()
    if not backdrop_is_switched_off():
        settings.setValue(_KEY_AMBIENT_THEME, theme.ambient)
        settings.setValue(_KEY_AMBIENT_PALETTE, theme.ambient_palette)
    settings.setValue(_KEY_SOUND_THEME, theme.sound)
    settings.sync()


def backdrop_is_switched_off() -> bool:
    """Whether the user has turned the animated backdrop off and left it.

    The STORED choice, not the live answer:
    :func:`get_ambient_enabled` also reports ``False`` for
    ``SPACR_NO_BACKDROP``, which :mod:`spacr.qt.crash_recovery` sets for
    one process after two failed launches. That is a suppression and not
    a preference, and treating it as one would silently strip the
    backdrop out of a theme the user chose during that one run.

    :returns: ``True`` when the Animation control reads "None", or when
        the separate on/off key is off.
    """
    if _raw_ambient_animation() == _no_animation_key():
        return True
    return not _as_bool(_settings().value(_KEY_AMBIENT_ENABLED,
                                          DEFAULT_AMBIENT_ENABLED),
                        DEFAULT_AMBIENT_ENABLED)


def get_cell_variant() -> str:
    """Which of the user's micrographs the Cell theme uses."""
    from .imagery import CELL_VARIANTS, DEFAULT_CELL_VARIANT
    raw = str(_settings().value(_KEY_CELL_VARIANT, DEFAULT_CELL_VARIANT))
    return raw if raw in CELL_VARIANTS else DEFAULT_CELL_VARIANT


def set_cell_variant(variant: str) -> None:
    """Persist one of the bundled Cell-theme microscopy variants.

    :param variant: one of :data:`spacr.qt.imagery.CELL_VARIANTS`; any other
        value raises :class:`ValueError`.
    """
    from .imagery import CELL_VARIANTS
    if variant not in CELL_VARIANTS:
        raise ValueError(f"unknown cell variant {variant!r}. "
                          f"Choose from {CELL_VARIANTS}.")
    _settings().setValue(_KEY_CELL_VARIANT, variant)


#: THE SPACE THEME'S KEYS WERE RETIRED, 2026-09-09.
#:
#: `space_variants`, `get_space_variant`, `set_space_variant`,
#: `get_space_seed`, `set_space_seed` and `space_background_path` lived here
#: and only ever called each other. `space_background_path` had exactly one
#: caller -- the `theme == "space"` branch of `theme_background_path` -- and
#: `"space"` is not in `VALID_THEMES`, so `set_theme` refuses it,
#: `theme_choices()` offers no space token, and `get_theme()` maps anything
#: unrecognised to `DEFAULT_THEME`. The branch could not be entered by any
#: route through this module.
#:
#: The comment in `get_theme_choice` above already made half this argument --
#: "a branch for it could not be reached by any route through this module" --
#: and then kept the accessors on the grounds that "spaceout still draws it".
#: That half was wrong. `spaceout` is the fractal dressing, and the "space"
#: fractal PATTERN is `widgets/fractal_space.py`, a starfield shader that
#: reads neither key. `set_space_variant` and `set_space_seed` were called
#: from nowhere at all.
#:
#: The Space ARTWORK is untouched; what is gone is the accessor pair for a
#: theme name nothing can select.

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

    :param theme: an application theme name; only ``"cell"`` has a background
        image.
    :param width: the wanted image width in pixels; 0 or less means the screen
        size.
    :param height: the wanted image height in pixels; 0 or less means the
        screen size.
    """
    if theme == "cell":
        return cell_background_path(width, height)
    return None


def resolve_effective_theme() -> str:
    """Return the theme to render — one of :data:`PALETTE_THEMES`.

    Resolves an explicitly chosen ``"system"`` to the operating system's
    colour scheme as Qt reports it (``QStyleHints.colorScheme``), and to
    dark when Qt can't tell. It does not read the application palette:
    that is spaCR's own once a theme has been applied, so it would answer
    with whatever was applied last. Every other value passes through, so
    callers that only understand light/dark should compare against
    ``"light"`` and treat everything else as dark (Space and Cell are dark
    themes).
    """
    theme = get_theme()
    if theme in PALETTE_THEMES:
        return theme
    try:
        from .theme import system_colour_scheme
        return system_colour_scheme() or "dark"
    except Exception:
        LOG.debug("could not read the system colour scheme", exc_info=True)
    return "dark"



#: The animation is on out of the box. It costs nothing while a screen is
#: hidden (:class:`spacr.qt.widgets.ambient.AmbientWidget` stops its timer
#: on ``hideEvent``), so leaving it on does not tax a machine that is busy
#: segmenting on the GPU behind a different tab.
DEFAULT_AMBIENT_ENABLED = True


def get_ambient_enabled() -> bool:
    """Whether module screens paint the animated background.

    Answers ``False`` outright when ``SPACR_NO_BACKDROP`` is set, whatever
    is stored. `spacr.qt.crash_recovery` sets it after spaCR has died on
    launch twice running: the backdrop is the only thing spaCR asks a
    driver to do at startup, and the setting that would turn it off is
    behind the window that never appears. Process-local and never saved, so
    the next clean run brings it back with nothing for the user to undo.

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
    import os

    if os.environ.get("SPACR_NO_BACKDROP"):
        return False
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

    :param name: an entry of ``ANIMATION_CHOICES`` from
        :mod:`spacr.qt.widgets.ambient`; its ``NO_ANIMATION`` entry
        (``"none"``) switches the backdrop off, and any other value raises
        :class:`ValueError`.
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

    :param on: true to turn it on, false to turn it off; stored as a ``bool``.
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

    :param name: an ambient theme name from ``AMBIENT_THEMES``.
    :raises ValueError: if ``name`` is not a known ambient theme.
    """
    from .widgets.ambient import AMBIENT_THEMES, palettes_for
    if name not in AMBIENT_THEMES:
        raise ValueError(f"unknown ambient theme {name!r}. "
                         f"Choose from {AMBIENT_THEMES}.")
    settings = _settings()
    settings.setValue(_KEY_AMBIENT_THEME, name)
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

    :param theme: an ambient theme name, as offered by
        :data:`spacr.qt.widgets.ambient.AMBIENT_THEMES`; an unknown name gives
        the global default palette.
    """
    from .widgets.ambient import DEFAULT_PALETTE, palettes_for
    try:
        valid = palettes_for(theme)
    except Exception:
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

    :param name: a palette name offered by the current ambient theme.
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
            if old == old and old > 0:
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
    """Read one ambient-motion multiplier, clamped to its range.

    A hand-edited INI can hold ``nan``, which compares false against every
    bound and would pass a range check written as two comparisons -- so it
    is tested for explicitly and falls back to the default.

    :param key: the settings key.
    :param index: which multiplier, indexing the range table.
    :returns: the value, within range.
    """
    _migrate_ambient_motion()
    (low, high), default = _ambient_ranges()[index]
    try:
        value = float(_settings().value(key, default))
    except (TypeError, ValueError):
        return default
    if value != value:
        return default
    return max(low, min(high, value))


def _set_ambient_multiplier(key: str, index: int, value: float) -> None:
    """Write one ambient-motion multiplier, clamped to its range.

    :param key: the settings key.
    :param index: which multiplier, indexing the range table.
    :param value: the value to store; anything unparseable, or ``nan``,
        stores the default instead.
    """
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
    this is a slider, and there is no user error to report.

    :param value: the blur, in units of eight screen pixels of area averaging
        (0.0 is as designed); clamped to ``BLUR_RANGE`` from
        :mod:`spacr.qt.widgets.ambient`, and an unparseable value or NaN stores
        ``DEFAULT_BLUR``.
    """
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
    """Set the detail multiplier. Clamped.

    :param value: the multiplier on each animation's own shading buffer (1.0 is
        as designed); clamped to ``RESOLUTION_RANGE`` from
        :mod:`spacr.qt.widgets.ambient`, and an unparseable value or NaN stores
        ``DEFAULT_RESOLUTION``.
    """
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
    """Set the element-count multiplier. Clamped.

    :param value: the multiplier on each animation's own element count (1.0 is
        as designed); clamped to ``DENSITY_RANGE`` from
        :mod:`spacr.qt.widgets.ambient`, and an unparseable value or NaN stores
        ``DEFAULT_DENSITY``.
    """
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

    :param name: the starfield drift direction, one of ``DRIFT_DIRECTIONS``
        (``"up"``, ``"down"`` or ``"random"`` when the widget module cannot be
        imported).
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
    """Set the motion multiplier. Clamped.

    :param value: the multiplier on each theme's own motion (1.0 is as
        designed); clamped to ``SPEED_RANGE`` from
        :mod:`spacr.qt.widgets.ambient`, and an unparseable value or NaN stores
        ``DEFAULT_SPEED``.
    """
    _set_ambient_multiplier(_KEY_AMBIENT_SPEED, 1, value)


def get_ambient_size() -> float:
    """How large the animated background's elements are, as a multiplier on
    each theme's own size range. 1.0 is as designed."""
    return _ambient_multiplier(_KEY_AMBIENT_SIZE, 2)


def set_ambient_size(value: float) -> None:
    """Set the element-size multiplier. Clamped.

    :param value: the multiplier on each animation's own element size (1.0 is
        as designed); clamped to ``SIZE_RANGE`` from
        :mod:`spacr.qt.widgets.ambient`, and an unparseable value or NaN stores
        ``DEFAULT_SIZE``.
    """
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
    import sys

    if (sys.modules.get(f"{__package__}.widgets.ambient") is None
            and not get_ambient_enabled()):
        return
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
                widget.set_animating(False)
                widget.setVisible(False)
                continue
            widget.setVisible(True)
            widget.set_animating(True)
            try:
                widget.set_theme(theme)
                widget.set_palette(palette)
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



#: The three modes, most aggressive first (the order the dropdown lists
#: them, so the default is at the bottom where a reader lands last).
SPACR_MODES = ("extra_performance", "performance", "balanced")

#: THE ONE PERFORMANCE SETTING, ordered by how much of the machine spaCR
#: keeps for itself: least first.
#:
#: Laptop was a second control that quietly overrode this one, so a user who
#: chose Workstation-like behaviour here could have it undone by a setting on
#: another row -- two answers to one question. It is a LEVEL, not an
#: independent axis: the most constrained end of the same scale.
#:
#: Scientific computation and results are identical at every level. Only
#: scheduling, caching, memory retention and interface decoration differ.
PERFORMANCE_LEVELS = ("laptop", "extra_performance", "performance",
                      "balanced", "workstation")

#: What the dialog calls each level.
PERFORMANCE_LABELS = {
    "laptop": "Laptop",
    "extra_performance": "Extra Performance",
    "performance": "Performance",
    "balanced": "Balanced",
    "workstation": "Workstation",
}

#: The hardware each level is for, and what it trades. Shown as the level's
#: tooltip, so the choice can be made without guessing.
#
#: 286: EVERY CLAIM HERE IS ONE THE CODE KEEPS. The minutes and megabytes
#: are `memory_budget.RECOMMENDED` for the level, which an untouched budget
#: follows, and a test holds each note to them. The old wording promised
#: "one worker", "dropped as soon as a run finishes" and "nothing is dropped
#: until you ask", none of which any code did.
PERFORMANCE_NOTES = {
    "laptop": (
        "For a machine with 8 GB of memory or less, or one running on "
        "battery. spaCR keeps the least: no animated backdrop, the fewest "
        "editable figures, and by default cached data is dropped after 2 "
        "minutes unused or above 256 MB. Caches are also cleared at launch "
        "and before every run, so going back to a figure is slower."),
    "extra_performance": (
        "For a shared machine you do not want spaCR to crowd. spaCR clears "
        "its caches, unused GPU memory and idle threads at launch and before "
        "every run, turns every visual setting to its minimum, and by "
        "default drops cached data after 5 minutes unused or above 512 MB."),
    "performance": (
        "For a machine with other work on it. spaCR clears its caches and "
        "unused GPU memory once, at launch, leaves your visual settings "
        "alone, and by default drops cached data after 10 minutes unused or "
        "above 1024 MB."),
    "balanced": (
        "For an ordinary desktop with 16 GB or more. Nothing is cleared at "
        "launch or before a run, recent figures and images stay ready so "
        "going back is instant, and by default cached data is dropped after "
        "15 minutes unused or above 2048 MB."),
    "workstation": (
        "For a machine with 64 GB or more that is yours alone. spaCR keeps "
        "the most: the most editable figures, and by default cached data "
        "for 60 minutes unused and up to 16384 MB. Nothing is cleared at "
        "launch or before a run. Uses the most memory of any level, by "
        "design."),
}

#: The level a machine gets when nothing has been chosen.
DEFAULT_PERFORMANCE_LEVEL = "balanced"

_KEY_PERFORMANCE_LEVEL = "prefs/performance_level"

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
#: The published defaults use ``auto`` so one set of numbers selects the GPU
#: when vispy is importable and the CPU otherwise.
from .fractal_defaults import PATTERNS as FRACTAL_PATTERNS  # noqa: E402
FRACTAL_BACKENDS = ("auto", "gpu", "cpu")
#: The quality levels, least demanding first.
#:
#: A quality level governs the related render numbers. A level that only
#: nudged an internal detail count while supersampling,
#: render scale and the iteration budget sat at whatever they were is a
#: label rather than a setting.
#:
#: `auto` stays first because it is not a level but a refusal to choose one:
#: it asks the machine.
FRACTAL_QUALITIES = ("auto", "balanced", "high", "ultra")

#: What each level sets, as ``quality -> {setting: value}``.
#:
#: A LEVEL IS A SET OF NUMBERS, not an adjective. These are applied when the
#: level is chosen, and every one of them remains a field the user can then
#: change -- picking a level is a starting point, not a lock.
QUALITY_PRESETS = {
    "balanced": {
        "scale": 0.75,
        "base_iterations": 200,
        "iterations_per_decade": 40.0,
        "max_iterations": 1200,
    },
    "high": {
        "scale": 1.75,
        "base_iterations": 300,
        "iterations_per_decade": 55.0,
        "max_iterations": 2200,
    },
    "ultra": {
        "scale": 2.5,
        "base_iterations": 500,
        "iterations_per_decade": 90.0,
        "max_iterations": 2200,
    },
}


def apply_quality_preset(quality: str) -> dict:
    """Set the numbers a quality level implies, and return them.

    :param quality: one of :data:`FRACTAL_QUALITIES`.
    :returns: what was applied; empty for ``auto`` or an unknown level.

    ``auto`` applies nothing on purpose: it means "decide from the machine"
    and the renderer does that per backend, so writing numbers here would
    turn a decision that follows the hardware into one frozen at the moment
    somebody opened Preferences.
    """
    preset = QUALITY_PRESETS.get(str(quality))
    if not preset:
        return {}
    set_fractal_settings(**preset)
    return dict(preset)


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
        DEFAULT_FOLLOW_POINTER, DEFAULT_POINTER_SIZE,
        DEFAULT_POINTER_STRENGTH,
    )

    settings = _settings()

    def _text(key, default, allowed):
        """One stored string, or the default when it is not an allowed value.

        Falls back rather than raising: a stale preference naming a theme that no
        longer exists must not stop the settings loading.
        """
        raw = str(settings.value(key, default))
        return raw if raw in allowed else default

    def _number(key, default, low, high):
        """A stored number, with only the bounds that are real.

        ``None`` for a bound means there is none: the settings are FIELDS,
        and a value the user typed is not quietly reduced on the way back
        out. Only a value that cannot work at all is refused, and
        `explain_a_fractal_number` is what says so, in words, at the point
        it is entered.
        """
        try:
            value = float(settings.value(key, default))
        except (TypeError, ValueError):
            return default
        if value != value:
            return default
        if low is not None and value < low:
            return low
        if high is not None and value > high:
            return high
        return value

    def _truth(key, default):
        """A stored boolean, however QSettings gave it back.

        An INI file hands every value back as a string, so `bool("false")`
        is True and a switch the user turned off comes back on.
        """
        raw = settings.value(key, default)
        if isinstance(raw, str):
            return raw.strip().lower() in ("1", "true", "yes", "on")
        return bool(raw)

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
        "scale": _number(_KEY_FRACTAL_SCALE, DEFAULT_SCALE, 0.01,
                         None),
        "speed": _number(_KEY_FRACTAL_SPEED, DEFAULT_SPEED, 0.0, None),
        "dream": _number(_KEY_FRACTAL_DREAM, DEFAULT_DREAM, 0.0, None),
        "variable_speed": variable,
        "speed_min": _number(_KEY_FRACTAL_SPEED_MIN, DEFAULT_SPEED_MIN,
                             0.0, None),
        "speed_max": _number(_KEY_FRACTAL_SPEED_MAX, DEFAULT_SPEED_MAX,
                             0.0, None),
        "speed_period": _number(_KEY_FRACTAL_SPEED_PERIOD,
                                DEFAULT_SPEED_PERIOD, 0.1, None),
        "pointer_gravity": _truth(_KEY_FRACTAL_POINTER,
                                  DEFAULT_FOLLOW_POINTER),
        "pointer_size": _number(_KEY_FRACTAL_POINTER_SIZE,
                                DEFAULT_POINTER_SIZE, 0.0, None),
        "pointer_strength": _number(_KEY_FRACTAL_POINTER_STRENGTH,
                                    DEFAULT_POINTER_STRENGTH, 0.0, None),
        "supersampling": int(_number(_KEY_FRACTAL_SUPERSAMPLING,
                          _MANDEL_DEFAULTS["supersampling"],
                          FRACTAL_LIMITS['supersampling'][0], None)),
        "seconds_per_decade": _number(_KEY_FRACTAL_SECONDS_PER_DECADE,
                     _MANDEL_DEFAULTS["seconds_per_decade"],
                     FRACTAL_LIMITS['seconds_per_decade'][0], None),
        "base_iterations": int(_number(_KEY_FRACTAL_BASE_ITERATIONS,
                          _MANDEL_DEFAULTS["base_iterations"],
                          FRACTAL_LIMITS['base_iterations'][0], None)),
        "iterations_per_decade": _number(_KEY_FRACTAL_ITERATIONS_PER_DECADE,
                     _MANDEL_DEFAULTS["iterations_per_decade"],
                     FRACTAL_LIMITS['iterations_per_decade'][0], None),
        "max_iterations": int(_number(_KEY_FRACTAL_MAX_ITERATIONS,
                          _MANDEL_DEFAULTS["max_iterations"],
                          FRACTAL_LIMITS['max_iterations'][0], None)),
        "precision_digits": int(_number(_KEY_FRACTAL_PRECISION_DIGITS,
                          _MANDEL_DEFAULTS["precision_digits"],
                          FRACTAL_LIMITS['precision_digits'][0], None)),
        "initial_scale": _number(_KEY_FRACTAL_INITIAL_SCALE,
                     _MANDEL_DEFAULTS["initial_scale"],
                     FRACTAL_LIMITS['initial_scale'][0], None),
        "zoom_rate": _number(_KEY_FRACTAL_ZOOM_RATE,
                     _MANDEL_DEFAULTS["zoom_rate"],
                     FRACTAL_LIMITS['zoom_rate'][0], None),
        "render_scale": _number(_KEY_FRACTAL_RENDER_SCALE,
                     _MANDEL_DEFAULTS["render_scale"],
                     FRACTAL_LIMITS['render_scale'][0], None),
        "steering_strength": _number(_KEY_FRACTAL_STEERING_STRENGTH,
                     _MANDEL_DEFAULTS["steering_strength"],
                     FRACTAL_LIMITS['steering_strength'][0], None),
        "steering_interval_decades": _number(_KEY_FRACTAL_STEERING_INTERVAL_DECADES,
                     _MANDEL_DEFAULTS["steering_interval_decades"],
                     FRACTAL_LIMITS['steering_interval_decades'][0], None),
        "steering_duration": _number(_KEY_FRACTAL_STEERING_DURATION,
                     _MANDEL_DEFAULTS["steering_duration"],
                     FRACTAL_LIMITS['steering_duration'][0], None),
        "candidate_count": int(_number(_KEY_FRACTAL_CANDIDATE_COUNT,
                          _MANDEL_DEFAULTS["candidate_count"],
                          FRACTAL_LIMITS['candidate_count'][0], None)),
        "path": _text(_KEY_FRACTAL_PATH,
                      _MANDEL_DEFAULTS.get("path", "tour"),
                      ("fixed", "guided", "tour")),
        "steering": _number(_KEY_FRACTAL_STEERING,
                            _MANDEL_DEFAULTS.get("steering", 0.35),
                            0.0, 1.0),
        "max_depth": _number(_KEY_FRACTAL_MAX_DEPTH,
                             _MANDEL_DEFAULTS["max_depth"],
                             *FRACTAL_LIMITS["max_depth"][:2]),
    }


def set_fractal_settings(**values) -> None:
    """Persist any subset of the fractal settings.

    :raises ValueError: on an unknown name, or a backend/quality outside its
        set. A number is stored as given; only one that cannot work at all
        is moved, and `explain_a_fractal_number` says so in words before it
        reaches here. What follows describes the old behaviour, kept because
        the reasoning about a slider still applies to the two sliders left
        produce one and a hand-edited file should still start.
    """
    from .fractal_defaults import clamp

    keys = {
        "pattern": (_KEY_FRACTAL_PATTERN, None),
        "backend": (_KEY_FRACTAL_BACKEND, None),
        "quality": (_KEY_FRACTAL_QUALITY, None),
        "scale": (_KEY_FRACTAL_SCALE, (0.01, None)),
        "speed": (_KEY_FRACTAL_SPEED, (0.0, None)),
        "dream": (_KEY_FRACTAL_DREAM, (0.0, None)),
        "variable_speed": (_KEY_FRACTAL_VARIABLE_SPEED, None),
        "speed_min": (_KEY_FRACTAL_SPEED_MIN, (0.0, None)),
        "speed_max": (_KEY_FRACTAL_SPEED_MAX, (0.0, None)),
        "speed_period": (_KEY_FRACTAL_SPEED_PERIOD, (0.1, None)),
        "pointer_gravity": (_KEY_FRACTAL_POINTER, None),
        "pointer_size": (_KEY_FRACTAL_POINTER_SIZE, (0.0, None)),
        "pointer_strength": (_KEY_FRACTAL_POINTER_STRENGTH, (0.0, None)),
        "supersampling": (_KEY_FRACTAL_SUPERSAMPLING,
                (FRACTAL_LIMITS['supersampling'][0], FRACTAL_LIMITS['supersampling'][1])),
        "seconds_per_decade": (_KEY_FRACTAL_SECONDS_PER_DECADE,
                (FRACTAL_LIMITS['seconds_per_decade'][0], FRACTAL_LIMITS['seconds_per_decade'][1])),
        "base_iterations": (_KEY_FRACTAL_BASE_ITERATIONS,
                (FRACTAL_LIMITS['base_iterations'][0], FRACTAL_LIMITS['base_iterations'][1])),
        "iterations_per_decade": (_KEY_FRACTAL_ITERATIONS_PER_DECADE,
                (FRACTAL_LIMITS['iterations_per_decade'][0], FRACTAL_LIMITS['iterations_per_decade'][1])),
        "max_iterations": (_KEY_FRACTAL_MAX_ITERATIONS,
                (FRACTAL_LIMITS['max_iterations'][0], FRACTAL_LIMITS['max_iterations'][1])),
        "precision_digits": (_KEY_FRACTAL_PRECISION_DIGITS,
                (FRACTAL_LIMITS['precision_digits'][0], FRACTAL_LIMITS['precision_digits'][1])),
        "initial_scale": (_KEY_FRACTAL_INITIAL_SCALE,
                (FRACTAL_LIMITS['initial_scale'][0], FRACTAL_LIMITS['initial_scale'][1])),
        "zoom_rate": (_KEY_FRACTAL_ZOOM_RATE,
                (FRACTAL_LIMITS['zoom_rate'][0], FRACTAL_LIMITS['zoom_rate'][1])),
        "render_scale": (_KEY_FRACTAL_RENDER_SCALE,
                (FRACTAL_LIMITS['render_scale'][0], FRACTAL_LIMITS['render_scale'][1])),
        "steering_strength": (_KEY_FRACTAL_STEERING_STRENGTH,
                (FRACTAL_LIMITS['steering_strength'][0], FRACTAL_LIMITS['steering_strength'][1])),
        "steering_interval_decades": (_KEY_FRACTAL_STEERING_INTERVAL_DECADES,
                (FRACTAL_LIMITS['steering_interval_decades'][0], FRACTAL_LIMITS['steering_interval_decades'][1])),
        "steering_duration": (_KEY_FRACTAL_STEERING_DURATION,
                (FRACTAL_LIMITS['steering_duration'][0], FRACTAL_LIMITS['steering_duration'][1])),
        "candidate_count": (_KEY_FRACTAL_CANDIDATE_COUNT,
                (FRACTAL_LIMITS['candidate_count'][0], FRACTAL_LIMITS['candidate_count'][1])),
        "path": (_KEY_FRACTAL_PATH, None),
        "steering": (_KEY_FRACTAL_STEERING, (0.0, 1.0)),
        "max_depth": (_KEY_FRACTAL_MAX_DEPTH,
                      (FRACTAL_LIMITS["max_depth"][0],
                       FRACTAL_LIMITS["max_depth"][1])),
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
        if name in ("speed", "scale"):
            derived = (speed_group_values(value) if name == "speed"
                       else scale_group_values(value))
            store.setValue(key, float(value))
            for _name, _value in derived.items():
                if _name == name:
                    continue
                _key, _ = keys[_name]
                store.setValue(_key, _value)
            continue
        if name == "steering":
            from .widgets.fractal_mandelbrot import steering_from_one_number

            store.setValue(key, float(value))
            derived = steering_from_one_number(
                float(value),
                float(get_fractal_settings().get("seconds_per_decade", 24.0)))
            for _name, _value in derived.items():
                _key, _ = keys[_name]
                store.setValue(_key, _value)
            continue
        if bounds is not None:
            low, high = bounds
            value = float(value)
            if low is not None and value < low:
                value = low
            if high is not None and value > high:
                value = high
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
#: ``'on_demand'`` -- when the operation that needs them is called; this is
#: the default.
#: ``'eager'`` -- at startup, on a worker thread. It is useful only on a
#: machine that will certainly run a pipeline and prefers to pay a potentially
#: tens-of-seconds import cost at the beginning.
PRELOAD_POLICIES = ("on_demand", "eager")


def get_preload_policy() -> str:
    """When to import torch and the rest. 'on_demand' or 'eager'."""
    raw = str(_settings().value(_KEY_PRELOAD, "on_demand")).strip().lower()
    return raw if raw in PRELOAD_POLICIES else "on_demand"


def set_preload_policy(policy: str) -> None:
    """Persist it. Takes effect at the next launch, and says so.

    :param policy: one of :data:`PRELOAD_POLICIES`, matched after stripping and
        lower-casing.
    :raises ValueError: on anything but the two policies.
    """
    text = str(policy).strip().lower()
    if text not in PRELOAD_POLICIES:
        raise ValueError(f"unknown preload policy {policy!r}; expected one "
                         f"of {list(PRELOAD_POLICIES)}")
    _settings().setValue(_KEY_PRELOAD, text)
    _settings().sync()


#: Body text is Light while titles are Regular. Only the application font is
#: set from this -- the headings,
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

    :param weight: one of :data:`INTERFACE_FONT_WEIGHTS`, matched after
        stripping and lower-casing.
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
    """Whether laptop constraints apply.

    :returns: ``"on"`` at the Laptop level, otherwise ``"off"``.

    DERIVED, NOT STORED. This was a second control that quietly overrode
    the mode selector, so a user could choose one posture on one row and
    have another row undo it -- two answers to one question. Laptop is now
    the most constrained LEVEL of the single selector, and this answers
    from it so callers that still ask in these words agree with it.

    There is no ``"automatic"`` any more: it meant "measure the machine and
    decide", which is a guess presented as a setting. The five levels say
    which hardware each is for and let the user pick.
    """
    return "on" if get_performance_level() == "laptop" else "off"


def set_laptop_mode(choice: str) -> None:
    """Persist the laptop-mode preference and apply it now.

    :param choice: one of :data:`LAPTOP_MODE_CHOICES`: ``"on"`` selects the
        Laptop performance level, ``"off"`` moves a Laptop level back to the
        default level, and ``"automatic"`` changes nothing.
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
    # 286: THE OLD WORDS WRITE THE ONE VALUE. This used to store the key the
    # migration removes and apply a hardware measurement for "automatic",
    # so a caller could set "on" and read "off" back from `get_laptop_mode`,
    # or have a two-core reading override Workstation. "on" is the Laptop
    # level; "off" leaves Laptop for the default level; "automatic" states no
    # choice and leaves the level alone. This run's backdrop follows through
    # `set_performance_level`.
    if choice == "on":
        set_performance_level("laptop")
    elif choice == "off" and get_performance_level() == "laptop":
        set_performance_level(DEFAULT_PERFORMANCE_LEVEL)
    return None


def laptop_mode_note(choice: str) -> str:
    """What the chosen setting will do on THIS machine, said before saving.

    Automatic is the case that needs saying: the label cannot state the
    outcome, because the outcome depends on the machine reading it.

    :param choice: one of :data:`LAPTOP_MODE_CHOICES`: ``"automatic"`` reports
        what this machine's measurement decides, ``"on"`` lists what is turned
        down, and anything else is described as off.
    """
    from .laptop_mode import measure, wanted, what_it_turns_down

    turns_down = ", ".join(what for what, _cost in what_it_turns_down())
    if choice == "automatic":
        _on, why = wanted({**measure(), "override": None})
        return why
    if choice == "on":
        return (f"Turns down {turns_down}. Only the drawing changes: a run "
                f"computes exactly the same answer either way.")
    return "Keeps the animation and the blur on, whatever this machine is."


def get_idle_minutes() -> float:
    """How long an unused cache entry may sit before it is dropped.

    :returns: minutes; 0 means "as soon as nothing is using it".
    """
    from .memory_budget import MAX_IDLE_MINUTES, MIN_IDLE_MINUTES
    # 286: an untouched budget follows the performance level, so the sweep
    # enforces what the level promises; a number the user set is kept.
    fallback = float(_level_budget()[0])
    raw = _settings().value(_KEY_IDLE_MINUTES, None)
    if raw is None or raw == "":
        return max(MIN_IDLE_MINUTES, min(MAX_IDLE_MINUTES, fallback))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return fallback
    return max(MIN_IDLE_MINUTES, min(MAX_IDLE_MINUTES, value))


def set_idle_minutes(minutes: float) -> None:
    """Persist the idle timeout.

    :param minutes: how long an unused cache entry may sit before it is
        dropped, in minutes; 0 drops it as soon as nothing uses it. Stored as a
        ``float``.
    """
    settings = _settings()
    settings.setValue(_KEY_IDLE_MINUTES, float(minutes))
    settings.sync()


def get_cache_ceiling_mb() -> int:
    """How much cache spaCR may hold at once, in megabytes."""
    from .memory_budget import MAX_CACHE_CEILING_MB, MIN_CACHE_CEILING_MB
    # Follows the level while untouched; see `get_idle_minutes`.
    fallback = int(_level_budget()[1])
    raw = _settings().value(_KEY_CACHE_CEILING, None)
    if raw is None or raw == "":
        return max(MIN_CACHE_CEILING_MB, min(MAX_CACHE_CEILING_MB, fallback))
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return fallback
    return max(MIN_CACHE_CEILING_MB, min(MAX_CACHE_CEILING_MB, value))


def set_cache_ceiling_mb(megabytes: int) -> None:
    """Persist the cache ceiling.

    :param megabytes: the most cache spaCR may hold at once, in megabytes;
        stored as an ``int``.
    """
    settings = _settings()
    settings.setValue(_KEY_CACHE_CEILING, int(megabytes))
    settings.sync()


def get_headroom_mb() -> int:
    """How much memory must stay free for everything else on the machine.

    THE FIRST OF THE THREE. The idle timeout and the ceiling say what may be
    kept; this says when keeping it stops being acceptable, and without it
    neither of the others has anything to answer to.
    """
    from .memory_budget import MAX_HEADROOM_MB, MIN_HEADROOM_MB
    # Follows the level while untouched; see `get_idle_minutes`.
    fallback = int(_level_budget()[2])
    raw = _settings().value(_KEY_HEADROOM, None)
    if raw is None or raw == "":
        return max(MIN_HEADROOM_MB, min(MAX_HEADROOM_MB, fallback))
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return fallback
    return max(MIN_HEADROOM_MB, min(MAX_HEADROOM_MB, value))


def set_headroom_mb(megabytes: int) -> None:
    """Persist the headroom floor.

    :param megabytes: the memory that must stay free for everything else on the
        machine, in megabytes; stored as an ``int``.
    """
    settings = _settings()
    settings.setValue(_KEY_HEADROOM, int(megabytes))
    settings.sync()


def _level_budget():
    """The budget the current performance level recommends.

    :returns: ``(idle minutes, cache MB, headroom MB)`` for the stored
        level, or the shipped defaults when the level cannot be read.

    The getters use this as the value of a budget nobody has typed, so an
    untouched budget follows the level (286). It never raises: a background
    budget sweep with no user in front of it reads these numbers, and an
    unreadable store must cost that sweep a default rather than an exception.
    """
    from .memory_budget import (DEFAULT_CACHE_CEILING_MB,
                                DEFAULT_HEADROOM_MB, DEFAULT_IDLE_MINUTES,
                                recommended_for)
    try:
        return recommended_for(get_performance_level())
    except Exception:                                        # noqa: BLE001
        return (DEFAULT_IDLE_MINUTES, DEFAULT_CACHE_CEILING_MB,
                DEFAULT_HEADROOM_MB)


def _save_budget_for_level(level, idle_minutes, cache_mb, headroom_mb):
    """Store the three budget numbers Preferences' Save was given (286).

    :param level: the performance level being saved beside them.
    :param idle_minutes: the idle timeout shown in the dialog.
    :param cache_mb: the cache ceiling shown in the dialog.
    :param headroom_mb: the free-memory floor shown in the dialog.
    :returns: None; the store is written and synced.

    A number equal to ``level``'s own recommendation is stored as "follow
    the level" -- its key removed -- so a later level change still moves
    it; any other number is the user's and is kept at every level. Writing
    all three unconditionally, as Save once did, froze the budget at
    whatever level happened to be showing on the first Save.
    """
    from .memory_budget import recommended_for

    own = recommended_for(level)
    settings = _settings()
    for key, value, level_value in (
            (_KEY_IDLE_MINUTES, float(idle_minutes), float(own[0])),
            (_KEY_CACHE_CEILING, int(cache_mb), int(own[1])),
            (_KEY_HEADROOM, int(headroom_mb), int(own[2]))):
        if value == level_value:
            settings.remove(key)
        else:
            settings.setValue(key, value)
    settings.sync()


def _budget_follows_level(mode_combo, last_level, spins) -> None:
    """Move each untouched budget spin box to the newly chosen level's value.

    :param mode_combo: the Preferences dialog's Performance combo; its
        current data is the level just chosen.
    :param last_level: a one-item list holding the level the spin boxes were
        last aligned to. It is updated in place, so the next change compares
        against the level this one chose.
    :param spins: the ``(idle minutes, cache MB, headroom MB)`` spin boxes,
        in the order of :data:`memory_budget.RECOMMENDED`'s tuples.
    :returns: None; the spin boxes are changed in place.

    286: a number still equal to the previous level's recommendation was
    never chosen by the user, so it moves with the level; a number the user
    typed differs from it and stays where they put it. That keeps the dialog
    agreeing with :func:`_save_budget_for_level`, which stores a number equal
    to the level's own as "follow the level". An unknown level on either
    side moves nothing, because there is no recommendation to compare with.
    """
    from .memory_budget import RECOMMENDED

    new, old = mode_combo.currentData(), last_level[0]
    last_level[0] = new
    if new == old or new not in RECOMMENDED or old not in RECOMMENDED:
        return
    for index, spin in enumerate(spins):
        if spin.value() == RECOMMENDED[old][index]:
            spin.setValue(RECOMMENDED[new][index])


def get_performance_level() -> str:
    """The single performance level, one of :data:`PERFORMANCE_LEVELS`.

    :returns: the stored level, migrating an older pair of settings on
        first read.

    MIGRATION HAPPENS HERE rather than in a startup step, because every
    reader of the old settings comes through this function and a migration
    that only ran at launch would be skipped by a headless run, a test, or
    a second process. It is idempotent: once a level is stored the old
    values are never consulted again.

    An explicit Laptop mode of ``on`` becomes ``Laptop`` -- that user asked
    for the most constrained profile and still gets it. With Laptop off or
    automatic the previous mode is kept as-is, so ``balanced`` stays
    ``Balanced``, and nobody's choice is silently changed.
    """
    settings = _settings()
    stored = str(settings.value(_KEY_PERFORMANCE_LEVEL, "") or "")
    if stored in PERFORMANCE_LEVELS:
        return stored

    laptop = str(settings.value(_KEY_LAPTOP_MODE, "automatic") or "automatic")
    previous = str(settings.value(_KEY_SPACR_MODE, DEFAULT_SPACR_MODE) or "")
    if laptop == "on":
        level = "laptop"
    elif previous in SPACR_MODES:
        level = previous
    else:
        level = DEFAULT_PERFORMANCE_LEVEL

    if _SAFE_MODE:
        # Safe mode answers every read with a default and sends every write
        # to the real store, so migrating here would "migrate" defaults and
        # write Balanced over the user's real level. Answer and store
        # nothing; the next ordinary start migrates the real values.
        return level

    try:
        settings.setValue(_KEY_PERFORMANCE_LEVEL, level)
        settings.sync()
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not store the migrated performance level",
                  exc_info=True)
        return level
    if _level_is_durable(settings, level):
        # The obsolete answers go only once the level has reached the store:
        # until then they are the only record of what the user chose (286).
        try:
            settings.remove(_KEY_LAPTOP_MODE)
            settings.remove(_KEY_SPACR_MODE)
            settings.sync()
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not remove the obsolete performance keys",
                      exc_info=True)
    return level


def _level_is_durable(settings, level: str) -> bool:
    """Whether a migrated performance level really reached the store.

    :param settings: the QSettings (or stand-in) the level was written to.
    :param level: the level that was written.
    :returns: True only if the store reports no error after the sync AND
        reads ``level`` back; False on any doubt, including an exception.

    The migration in :func:`get_performance_level` deletes the obsolete
    Laptop-mode and spaCR-mode answers only when this is True, because until
    the level is durable those answers are the only record of the user's
    choice. Both checks are needed: QSettings keeps a written value in
    memory even when its file cannot be written, so a read-back alone would
    pass a store whose disk write failed, and ``status()`` is what reports
    that.
    """
    status = getattr(settings, "status", None)
    if callable(status):
        try:
            if status() != QSettings.Status.NoError:
                return False
        except Exception:                                    # noqa: BLE001
            return False
    try:
        return str(settings.value(_KEY_PERFORMANCE_LEVEL, "") or "") == level
    except Exception:                                        # noqa: BLE001
        return False


def set_performance_level(level: str) -> None:
    """Persist the performance level.

    :param level: one of :data:`PERFORMANCE_LEVELS`.
    :raises ValueError: on an unknown level.
    """
    if level not in PERFORMANCE_LEVELS:
        raise ValueError(f"unknown performance level {level!r}. "
                         f"Choose from {PERFORMANCE_LEVELS}.")
    posture = spacr_mode_for_level(level)
    set_spacr_mode(posture)
    settings = _settings()
    settings.setValue(_KEY_PERFORMANCE_LEVEL, level)
    settings.sync()
    _backdrop_follows_the_level(level)


def _backdrop_follows_the_level(level: str) -> None:
    """Apply a newly set level's laptop constraint to this running process.

    :param level: the performance level just stored.
    :returns: None; a failure is logged at debug level and swallowed.

    286: the level, not a second switch, decides this run's laptop
    constraint -- the same answer ``launch`` gives at startup. Laptop
    suppresses the animated backdrop for this process; any other level lifts
    only a suppression THIS process made (``laptop_mode._suppressed_here``),
    so crash recovery's variable and the user's stored answer are never
    touched. Nothing here is persisted, and a failure must not stop the
    level itself from being saved.
    """
    try:
        from .laptop_mode import apply as _apply

        _apply(level == "laptop")
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not bring this run's backdrop in line with the "
                  "performance level", exc_info=True)


def spacr_mode_for_level(level: str) -> str:
    """The resource posture a level implies, in the old three-mode words.

    :param level: one of :data:`PERFORMANCE_LEVELS`.
    :returns: one of :data:`SPACR_MODES`.

    Laptop is more constrained than Extra Performance and Workstation is
    less constrained than Balanced, but neither has its own posture in the
    cleanup code -- they differ in what is RETAINED, not in how launch
    cleanup runs. Mapping them onto the nearest existing posture keeps one
    answer to "how hard does spaCR try to stay out of the way".
    """
    return {
        "laptop": "extra_performance",
        "extra_performance": "extra_performance",
        "performance": "performance",
        "balanced": "balanced",
        "workstation": "balanced",
    }.get(str(level), DEFAULT_SPACR_MODE)


#: What each level keeps, as multiples of the Balanced allowance.
#:
#: MONOTONIC BY CONSTRUCTION, and asserted by a test: the order of
#: :data:`PERFORMANCE_LEVELS` is meant to be a resource scale, and a table
#: that broke that order would make the selector a list of unrelated words.
#: Laptop keeps the least, Workstation the most.
PERFORMANCE_RETENTION = {
    "laptop": 0.25,
    "extra_performance": 0.5,
    "performance": 0.75,
    "balanced": 1.0,
    "workstation": 2.5,
}


def retention_scale(level: str = "") -> float:
    """How much reusable state this level allows, relative to Balanced.

    :param level: a performance level; the current one when omitted.
    :returns: a positive multiplier.
    """
    level = str(level or get_performance_level())
    return float(PERFORMANCE_RETENTION.get(level, 1.0))


def live_figure_allowance(level: str = "") -> int:
    """How many figures stay editable at this level.

    :param level: a performance level; the current one when omitted.
    :returns: a count of at least one.

    A live Figure is what makes a figure restylable -- it still has a
    legend to toggle and series to recolour -- and each holds its own data
    arrays, so this is the clearest thing the level scales. At least one,
    always: a level that kept none would make the right-click menu useless
    rather than cheap.
    """
    base = get_figure_live_cache()
    return max(1, int(round(base * retention_scale(level))))


def get_spacr_mode() -> str:
    """Which resource posture spaCR is in — one of :data:`SPACR_MODES`.

    DERIVED FROM THE PERFORMANCE LEVEL, which is the one stored setting.
    Kept because the cleanup code speaks in these three words.
    """
    return spacr_mode_for_level(get_performance_level())


def set_spacr_mode(mode: str) -> None:
    """Persist the mode, and move the visual settings with it.

    Entering Extra Performance stashes the five visual settings it
    overrides and writes their minimums; leaving it puts the stashed values
    back. Nothing else about a mode change is retroactive — the launch
    cleanup has already happened or not happened by the time anyone can
    reach this dialog.

    :param mode: one of :data:`SPACR_MODES`.
    :raises ValueError: on an unknown mode.
    """
    if mode not in SPACR_MODES:
        raise ValueError(f"unknown spaCR mode {mode!r}. "
                         f"Choose from {SPACR_MODES}.")
    previous = get_spacr_mode()
    settings = _settings()
    # ONE STORED VALUE (286). The posture is derived from the level, so the
    # level is all that is written; the old `prefs/spacr_mode` copy was a
    # second answer that only the migration ever read.
    settings.setValue(_KEY_PERFORMANCE_LEVEL, mode)
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
    """What choosing ``mode`` will cost, or ``""`` when it costs nothing.

    :param mode: one of :data:`SPACR_MODES`; a mode with no entry in
        :data:`MODE_WARNINGS` gives ``""``.
    """
    return MODE_WARNINGS.get(mode, "")


def _visual_snapshot() -> dict:
    """The five settings Extra Performance overrides, as they are now."""
    # "ambient_enabled" is the STORED switch, read past SPACR_NO_BACKDROP.
    # Restoring the animation goes through `set_ambient_animation`, which
    # turns the backdrop on, so without it a user who had switched the
    # backdrop off got it back by passing through Extra Performance or Laptop
    # (286). The raw key and not `get_ambient_enabled()`, which answers False
    # for a process-local suppression that must never be saved as a choice.
    return {
        "ambient_animation": get_ambient_animation(),
        "ambient_enabled": _as_bool(
            _settings().value(_KEY_AMBIENT_ENABLED, DEFAULT_AMBIENT_ENABLED),
            DEFAULT_AMBIENT_ENABLED),
        "ambient_resolution": get_ambient_resolution(),
        "ambient_density": get_ambient_density(),
        "setting_animations": get_setting_animations_enabled(),
        "field_fade": get_field_fade_enabled(),
    }


def _stash_visuals() -> None:
    """Save the current visual settings, so a mode change can restore them.

    Written and synced immediately: the stash exists to survive a crash
    during the mode switch it is protecting.
    """
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
        # Last, because `set_ambient_animation` above switches the backdrop
        # on. A stash written before 286 has no such entry and keeps the old
        # behaviour.
        if "ambient_enabled" in stashed:
            set_ambient_enabled(_as_bool(stashed["ambient_enabled"], True))
    except Exception:
        LOG.debug("could not restore the stashed visuals", exc_info=True)
        return False
    return True



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
    if value != value:
        return DEFAULT_SPINNER_DELAY
    return max(SPINNER_DELAY_MIN, min(SPINNER_DELAY_MAX, value))


def set_spinner_delay(seconds: float) -> None:
    """Set the spinner's appearance delay, in seconds. Clamped, not
    refused.

    :param seconds: how long background work must run before the spinner shows;
        clamped between :data:`SPINNER_DELAY_MIN` and
        :data:`SPINNER_DELAY_MAX`, and an unparseable value or NaN stores
        :data:`DEFAULT_SPINNER_DELAY`.
    """
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



#: Off by default. Every hover is text only until the reader presses the
#: **Animation** word in that tooltip's footer, and pressing it speaks for
#: that setting alone: 141 settings have an animation and each one is a
#: ~73 ms decoded movie, so neither a hover that only wanted the sentence nor
#: the 140 hovers after a press should pay for one. This preference is the
#: escape hatch for the reader who never wants to be asked.
DEFAULT_SETTING_ANIMATIONS = False

#: The two tooltip surfaces, BOTH ON by default.
#:
#: The box is today's behaviour -- `spacr.qt.widgets.hover_tooltip` has been
#: the setting tooltip for some time -- and the bottom strip is the nearest
#: thing to it, since the same strip already carries CATEGORY help on hover.
#: Defaulting either to off would take away help nobody asked to lose.
#:
#: BOTH OFF IS A LEGAL STATE AND IT COSTS SOMETHING. On several forms the
#: API link inside a setting's tooltip is the only route from that setting to
#: its documentation, so a reader who clears both has no way from the control
#: to the page describing it. The Preferences rows say so; see the tooltips
#: on the two switches.
#:
#: NEITHER TOUCHES THE CATEGORY STRIP, which answers a different question --
#: "what is this whole group of settings for".
#: THE BOX DEFAULTS OFF, AND THAT IS NOT A JUDGEMENT ABOUT THE BOX. The
#: popup box is redundant when the tooltip is shown at the bottom of the
#: window, so it appears by default only on screens with no strip. Both
#: surfaces are CHOOSABLE, and the switch does not reverse that default:
#: defaulting the box on would hand back a popup a user who reads the strip
#: does not need, and they would have to find a checkbox to undo it.
#:
#: So the box is one click away.
#: `tests/qt/test_setting_tooltip_footer.py::
#: test_hovering_a_real_setting_shows_no_tooltip_box` is the guard for the
#: default.
DEFAULT_TOOLTIPS_BOX = False
DEFAULT_TOOLTIPS_BOTTOM = True

#: Tooltips are ON by default, at the maintainer's instruction (2026-09-24).
#: They are how spaCR explains a button without spending a line of the
#: window on it; the switch exists for the person who already knows.
DEFAULT_TOOLTIPS_ENABLED = True

#: OFF until someone chooses it. The grid is a different way to read
#: the most-used screen in the application, so it arrives as an offer
#: rather than as a change to what everyone already knows.
DEFAULT_OBJECT_GRID = False


def get_tooltips_box_enabled() -> bool:
    """Whether hovering a setting's title opens the tooltip box.

    The box is `spacr.qt.widgets.hover_tooltip.HoverTooltip`: a QFrame the
    pointer can move INTO, which is what lets its API and Animation links be
    clicked at all. Cleared, no box opens and the bottom strip -- if it is on
    -- is the only place a setting explains itself.
    """
    return _as_bool(_settings().value(_KEY_TOOLTIPS_BOX,
                                      DEFAULT_TOOLTIPS_BOX),
                    DEFAULT_TOOLTIPS_BOX)


def set_tooltips_box_enabled(on: bool) -> None:
    """Turn the hover tooltip box on or off, effective at the next hover.

    :param on: true to turn it on, false to turn it off; stored as a ``bool``.
    """
    _settings().setValue(_KEY_TOOLTIPS_BOX, bool(on))
    _settings().sync()


def get_object_grid_enabled() -> bool:
    """Whether the per-object settings are shown as one table.

    78 of Mask's 201 settings are the same twenty-odd questions asked once
    per object type, so a form that lists them flat asks 203 questions before
    anything is segmented. Set, those rows are hidden and a grid takes their
    place -- one row per question, one column per object.

    THE STORED KEYS DO NOT CHANGE either way. The grid edits the same widgets
    the flat rows do, so a settings file written with this on is the same file
    written with it off.
    """
    return _as_bool(_settings().value(_KEY_OBJECT_GRID, DEFAULT_OBJECT_GRID),
                    DEFAULT_OBJECT_GRID)


def set_object_grid_enabled(on: bool) -> None:
    """Turn the per-object grid on or off, effective at the next form build.

    :param on: true to show the per-object settings as one table, false to list
        them flat; stored as a ``bool``.
    """
    _settings().setValue(_KEY_OBJECT_GRID, bool(on))
    _settings().sync()


def get_tooltips_bottom_enabled() -> bool:
    """Whether a hovered setting's help also appears in the bottom strip.

    The strip already shows CATEGORY help on hover; this puts SETTING help
    there too, and holds it for ten seconds after the pointer leaves so its
    API link can be reached. Without that hold the link is unreachable: it
    appears only while the pointer is on the setting, and moving toward it
    removes it.
    """
    return _as_bool(_settings().value(_KEY_TOOLTIPS_BOTTOM,
                                      DEFAULT_TOOLTIPS_BOTTOM),
                    DEFAULT_TOOLTIPS_BOTTOM)


def set_tooltips_bottom_enabled(on: bool) -> None:
    """Turn the bottom tooltip strip on or off, effective at the next hover.

    :param on: true to turn it on, false to turn it off; stored as a ``bool``.
    """
    _settings().setValue(_KEY_TOOLTIPS_BOTTOM, bool(on))
    _settings().sync()


def get_tooltips_enabled() -> bool:
    """Whether ordinary tooltips appear anywhere in spaCR. Default ``True``.

    The master switch read by :mod:`spacr.qt.tooltip_policy`, the one event
    filter on ``QApplication`` that decides when every tooltip appears and
    goes. Cleared, no tooltip is shown at all; the two settings surfaces --
    :func:`get_tooltips_box_enabled` and
    :func:`get_tooltips_bottom_enabled` -- are separate and unaffected.
    """
    return _as_bool(_settings().value(_KEY_TOOLTIPS_ENABLED,
                                      DEFAULT_TOOLTIPS_ENABLED),
                    DEFAULT_TOOLTIPS_ENABLED)


def set_tooltips_enabled(on: bool) -> None:
    """Turn every tooltip on or off, effective immediately.

    Drops :mod:`spacr.qt.tooltip_policy`'s cached answer and takes down any
    tooltip already on screen, so clearing the switch is not followed by one
    last popup nobody asked for.

    :param on: true to turn it on, false to turn it off; stored as a ``bool``.
    """
    _settings().setValue(_KEY_TOOLTIPS_ENABLED, bool(on))
    _settings().sync()
    try:
        from .tooltip_policy import invalidate_tooltip_policy
        invalidate_tooltip_policy()
    except Exception:                                       # noqa: BLE001
        LOG.debug("could not refresh the tooltip policy", exc_info=True)



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

    :param on: true to turn it on, false to turn it off; stored as a ``bool``.
    """
    settings = _settings()
    settings.setValue(_KEY_SETTING_ANIMATIONS, bool(on))
    settings.sync()



#: Where the first-run layout decision is recorded.
#:
#: PRIVATE, deliberately, though the rest of this module's accessors are
#: public. Each has exactly one caller -- `app._open_at_the_measured_width`
#: -- and a public pair would put two more symbols on the documented API
#: surface, which means another nine-language catalog pass for a record
#: nothing outside this module reads.
#:
#: RESOLVED ONCE AND RECORDED WITH ITS EVIDENCE, rather than recomputed on
#: every launch. The value is a
#: JSON object holding the width chosen AND the measurements that justified
#: it, so a later launch can tell whether the answer still applies rather
#: than re-deriving it and hoping it matches.
_KEY_LAYOUT_DECISION = "layout/first_run_decision"


def _get_layout_decision() -> dict:
    """The recorded first-run layout decision, or ``{}``.

    :returns: the stored object, or an empty dict when nothing has been
        recorded or what is stored cannot be read.

    NEVER RAISES AND NEVER RETURNS A PARTIAL RECORD. A decision that cannot
    be parsed is the same as no decision: the caller re-derives one. A
    half-read record is worse than none, because it would be compared
    against current metrics and could match by accident.
    """
    import json

    try:
        raw = _settings().value(_KEY_LAYOUT_DECISION, "")
        if not raw:
            return {}
        found = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    if not isinstance(found, dict):
        return {}
    required = {"width", "available", "font_scale"}
    return found if required <= set(found) else {}


def _set_layout_decision(record: dict) -> None:
    """Record the first-run layout decision and the metrics behind it.

    :param record: what was chosen and what it was chosen from.

    Storing is best-effort: a launch is not worth failing over a
    preference that only makes the NEXT launch cheaper.
    """
    import json

    try:
        _settings().setValue(_KEY_LAYOUT_DECISION, json.dumps(record))
    except (TypeError, ValueError):
        return


def get_font_scale() -> float:
    """Return the saved UI font scale, clamped to supported bounds."""
    try:
        raw = float(_settings().value(_KEY_FONT_SCALE,
                                        DEFAULT_FONT_SCALE))
    except (TypeError, ValueError):
        raw = DEFAULT_FONT_SCALE
    return max(FONT_SCALE_MIN, min(FONT_SCALE_MAX, raw))


def set_font_scale(scale: float) -> None:
    """Persist a UI font scale after clamping it to supported bounds.

    :param scale: the UI font scale, 1.0 for the designed size; converted to
        ``float`` and clamped between :data:`FONT_SCALE_MIN` and
        :data:`FONT_SCALE_MAX`.
    """
    scale = float(scale)
    scale = max(FONT_SCALE_MIN, min(FONT_SCALE_MAX, scale))
    _settings().setValue(_KEY_FONT_SCALE, scale)


#: The text size of a module screen's right-hand column (item 529), as a
#: multiple of the size the rest of the interface has. Ctrl + wheel over the
#: column moves it; nothing outside the column follows it.
RUNTIME_TEXT_SCALE_MIN = 0.60
RUNTIME_TEXT_SCALE_MAX = 2.00
DEFAULT_RUNTIME_TEXT_SCALE = 1.0


def get_runtime_text_scale() -> float:
    """The right-hand column's text size, clamped to its bounds.

    One value for every module screen, so the console reads the same size
    wherever the user goes; see :mod:`spacr.qt.live_zoom`.
    """
    try:
        raw = float(_settings().value(_KEY_RUNTIME_TEXT_SCALE,
                                      DEFAULT_RUNTIME_TEXT_SCALE))
    except (TypeError, ValueError):
        raw = DEFAULT_RUNTIME_TEXT_SCALE
    return max(RUNTIME_TEXT_SCALE_MIN, min(RUNTIME_TEXT_SCALE_MAX, raw))


def set_runtime_text_scale(scale: float) -> float:
    """Persist the right-hand column's text size, clamped to its bounds.

    :param scale: 1.0 for the size the rest of the interface has.
    :returns: the value stored.
    """
    scale = round(max(RUNTIME_TEXT_SCALE_MIN,
                      min(RUNTIME_TEXT_SCALE_MAX, float(scale))), 4)
    _settings().setValue(_KEY_RUNTIME_TEXT_SCALE, scale)
    return scale


def get_dock_width() -> int:
    """The width the user dragged the dock to, or 0 for its fitting width.

    Item 529. Stored in logical pixels; the dock clamps it to its drag
    bounds when it applies it, see :meth:`spacr.qt.widgets.dock.Dock.column_width`.
    """
    try:
        return max(0, int(float(_settings().value(_KEY_DOCK_WIDTH, 0) or 0)))
    except (TypeError, ValueError):
        return 0


def set_dock_width(width: int) -> None:
    """Persist the dock's dragged width; 0 goes back to the fitting width.

    :param width: logical pixels.
    """
    try:
        width = max(0, int(width))
    except (TypeError, ValueError):
        width = 0
    _settings().setValue(_KEY_DOCK_WIDTH, width)


def get_gui_scale() -> float:
    """Return the saved whole-GUI scale, clamped to supported bounds.

    Applied at startup by :func:`spacr.qt.gui_scale.apply_saved_gui_scale`
    and live by :func:`spacr.qt.gui_scale.set_gui_scale_live`.
    """
    try:
        raw = float(_settings().value(_KEY_GUI_SCALE, DEFAULT_GUI_SCALE))
    except (TypeError, ValueError):
        raw = DEFAULT_GUI_SCALE
    return max(GUI_SCALE_MIN, min(GUI_SCALE_MAX, raw))


def set_gui_scale(scale: float) -> None:
    """Persist a whole-GUI scale after clamping it to supported bounds.

    :param scale: the factor, 1.0 = 100 %. Only stored; drawing it is
        :func:`spacr.qt.gui_scale.set_gui_scale_live`'s job.
    """
    scale = max(GUI_SCALE_MIN, min(GUI_SCALE_MAX, float(scale)))
    _settings().setValue(_KEY_GUI_SCALE, scale)

#: How the auto-issue reporter behaves when something goes wrong.
#:
#: Three states, not two. "prompt me" and "never prompt me" leave out the
#: user who wants a report filed and does not want to be asked, and the
#: moment someone wants this off is the moment it has just interrupted them.
#:
#: 'always' files a redacted report to the public tracker as soon as a run
#: fails, with no preview. 'ask' opens the report in a preview and sends it
#: only on Send. 'never' files nothing.
ISSUE_PROMPT_ASK = "ask"
ISSUE_PROMPT_NEVER = "never"
ISSUE_PROMPT_ALWAYS = "always"
ISSUE_PROMPT_MODES = (ISSUE_PROMPT_ASK, ISSUE_PROMPT_NEVER,
                      ISSUE_PROMPT_ALWAYS)
_KEY_ISSUE_PROMPT = "ai/issue_prompt"

#: Written beside the mode by :func:`set_issue_prompt_mode`, so a value
#: stored by a build that knew the three modes apart can be told from one an
#: older build wrote on the user's behalf.
#:
#: IT IS NOT DECORATION, AND WITHOUT IT THE DECISION BELOW REACHED ALMOST
#: NOBODY. `SetupSlides.accept()` and `reject()` both call
#: `setup_screen.apply(self.answers())`, and `issue_prompt` has been one of
#: those answers since 6c57da8d6 (2026-08-21, shipped in 1.5.0.5). So every
#: profile that ever opened first-run setup — including one dismissed at the
#: first slide — has `'ask'` written into it, chosen by the default of the
#: day rather than by the user. Read as "an explicit earlier choice", the
#: new default would have applied to brand-new profiles only, and the
#: reporter of issue #117 would still have been on 'ask' after upgrading.
_KEY_ISSUE_PROMPT_CHOSEN = "ai/issue_prompt_chosen"

#: The default before 2026-09-19, and so the value an unmarked profile
#: holds when nobody chose it. 'never' and 'always' are never written by
#: accident: both took an answer, so both are kept as they stand.
_SUPERSEDED_ISSUE_PROMPT_MODE = ISSUE_PROMPT_ASK

#: The mode of a profile that has never chosen one.
#:
#: Automatic filing is the default, and a user agrees to it in the user
#: agreement when the mode is 'always'. The agreement is Section 5.6 of
#: :data:`spacr.qt.terms.TERMS`, which every profile is asked to accept
#: again (4.1 -> 4.2) and which nothing is filed without. A stored choice is
#: kept.
DEFAULT_ISSUE_PROMPT_MODE = ISSUE_PROMPT_ALWAYS


#: Whether the AI assistant is on when spaCR opens (248).
#:
#: The setup card's three launch toggles ship ON together. The card still
#: records an explicit opt-out, so a user who turns the assistant off stays
#: opted out when the shipped default changes.
#:
#: A stored value that is not recognised reads as OFF for the same reason a
#: bad `issue_prompt` reads as 'ask': the failure has to fall on the quiet
#: side.
_KEY_AI_DEFAULT_ON = "ai/on_by_default"


#: The untouched-profile value required by the three-toggle setup contract.
DEFAULT_AI_ON_AT_LAUNCH = True


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

    :returns: one of :data:`ISSUE_PROMPT_MODES`.
        :data:`DEFAULT_ISSUE_PROMPT_MODE` (``'always'``) when nothing is
        stored, and also when the only thing stored is the superseded
        default ``'ask'`` written by a build that did not mark what the user
        had chosen (:data:`_KEY_ISSUE_PROMPT_CHOSEN`). A choice this build or
        a later one wrote is returned as it stands, 'ask' included. A stored
        value that is not recognised reads as ``'ask'``: it was somebody's
        choice, even if this build cannot read it, so it must neither
        silence the reporter nor start publishing without a preview.
    """
    store = _settings()
    if not store.contains(_KEY_ISSUE_PROMPT):
        return DEFAULT_ISSUE_PROMPT_MODE
    value = str(store.value(_KEY_ISSUE_PROMPT, ISSUE_PROMPT_ASK) or "")
    if value not in ISSUE_PROMPT_MODES:
        return ISSUE_PROMPT_ASK
    if (value == _SUPERSEDED_ISSUE_PROMPT_MODE
            and not store.contains(_KEY_ISSUE_PROMPT_CHOSEN)):
        return DEFAULT_ISSUE_PROMPT_MODE
    return value


def set_issue_prompt_mode(mode: str) -> None:
    """Persist the auto-issue behaviour, and that it was chosen.

    The marker is what makes a later 'ask' stick: from here on, 'ask' in the
    store is an answer somebody gave, not the default of the day written
    into every profile that opened first-run setup. Every writer goes
    through this function — the setup slides, the Preferences dialog, the
    AI Console and the installer's consent page — so all four count as
    choosing.

    :param mode: one of :data:`ISSUE_PROMPT_MODES`.
    :raises ValueError: for anything else. Silently storing an unknown mode
        would read back as 'ask' and look like the setting was ignored.
    """
    mode = str(mode)
    if mode not in ISSUE_PROMPT_MODES:
        raise ValueError(
            f"issue prompt mode {mode!r} is not one of "
            f"{list(ISSUE_PROMPT_MODES)}.")
    store = _settings()
    store.setValue(_KEY_ISSUE_PROMPT, mode)
    store.setValue(_KEY_ISSUE_PROMPT_CHOSEN, True)



def scaled_px(base_px: int) -> int:
    """Return ``base_px`` scaled by the current user font scale.

    Widget sizes set from Python (``setMinimumWidth`` etc.) don't grow
    when the stylesheet's font size grows, so any control tuned to
    match a text width goes wrong at large font scales. Route those
    calls through this helper so they track the preference.

    Rounds to the nearest int; caps to at least 1 px so a very small
    scale doesn't collapse things to zero.

    :param base_px: a size in pixels as designed for a font scale of 1.0.
    """
    return max(1, int(round(base_px * get_font_scale())))


#: Dynamic property carrying an icon's width at 100 %, in logical pixels.
#:
#: A Qt property rather than a Python attribute for two reasons: it lives on
#: the C++ side, so it survives the wrapper being collected and rebuilt, and
#: a plain ``QPushButton`` can carry it without anyone subclassing Qt to
#: give it somewhere to put the number.
_KEY_ICON_BASE_W = "spacrIconBaseWidth"

#: The same for the icon's height. Two integer properties rather than one
#: pair, because Qt stores an int natively while a tuple crosses the
#: boundary as an opaque Python object that only PySide can read back.
_KEY_ICON_BASE_H = "spacrIconBaseHeight"

#: The method a widget may implement to re-derive its own icon geometry.
#:
#: Found by duck-typing, the way ``refresh_theme`` is. It exists for the
#: widgets whose icon size is not the only thing that has to move with it --
#: a tile whose hover animation has a resting size to return to, a square
#: button whose frame is drawn around the mark -- and it is handed the scale
#: so it never has to ask twice and get a different answer.
_ICON_SCALE_HOOK = "_apply_icon_scale"

#: Qt's own small-icon default, in logical pixels before the scale.
#:
#: A button that is given an icon and no size gets this from the style, and
#: the style's copy of it does NOT follow the font scale -- which is how a
#: handful of ghost buttons stayed 16 px wide while the interface around
#: them doubled. Naming it lets those buttons be registered at the size
#: they already draw at, so they start tracking the scale without changing
#: what they look like at 100 %.
_SMALL_ICON_PX = 16


def _scaled_side(base_px: int, scale: float) -> int:
    """Return ``base_px`` at ``scale``, by the rule :func:`scaled_px` uses.

    Split out so a caller that already knows the scale -- the sweep below
    resizes hundreds of widgets from one reading -- does not re-read the
    setting once per widget, and so the two can never round differently.

    :param base_px: the size at 100 %.
    :param scale: the multiplier to apply.
    :returns: the scaled size, never below 1 px.
    """
    return max(1, int(round(base_px * scale)))


def _set_scaled_icon_size(widget, base_w: int, base_h=None, scale=None):
    """Size a widget's icon from a base, and remember that base.

    THE BASE IS WHAT MAKES THE GESTURE REVERSIBLE. An icon resized from
    the size it is already wearing compounds its rounding: twenty notches
    of ``round(px * 1.05)`` and twenty back do not return a 20 px icon to
    20 px. Every size this sets is computed from the number stored here,
    which is the size at 100 % and never changes, so the scale alone
    decides the answer and the round trip is exact by construction.

    Idempotent, and cheap to call again: assigning an icon size a widget
    already has still invalidates its layout, so the assignment is skipped
    when nothing would move.

    :param widget: anything with ``setIconSize`` -- a button, a list view.
    :param base_w: the icon's width at 100 %, in logical pixels.
    :param base_h: its height at 100 %; square when omitted.
    :param scale: the scale to apply; the stored preference when omitted.
    :returns: the :class:`~PySide6.QtCore.QSize` now on the widget.
    """
    from PySide6.QtCore import QSize

    base_w = max(1, int(base_w))
    base_h = base_w if base_h is None else max(1, int(base_h))
    if scale is None:
        scale = get_font_scale()
    widget.setProperty(_KEY_ICON_BASE_W, base_w)
    widget.setProperty(_KEY_ICON_BASE_H, base_h)
    size = QSize(_scaled_side(base_w, scale), _scaled_side(base_h, scale))
    if widget.iconSize() != size:
        widget.setIconSize(size)
    return size


def _rescale_icon_sizes(app=None) -> int:
    """Re-derive every remembered icon size at the scale now in force.

    WHY THIS EXISTS. An icon size is a widget PROPERTY, set once when the
    widget is built. Nothing in a stylesheet reaches it, so a font scale
    that grew every caption left every glyph beside those captions exactly
    where it was: the text followed the wheel of the hold-Z zoom and the
    marks beside it did not.

    WHERE IT RUNS, AND WHY THERE. In
    :func:`apply_preferences_to_app`, which is the one step both routes to
    a new scale already take: the Preferences slider on its way out of the
    dialog, and :meth:`spacr.qt.live_zoom.LiveZoomFilter.settle` when the
    wheel goes quiet. That is the deliberate half of the gesture -- the
    one that rebuilds the stylesheet -- so the icons catch up with the
    spacing, in the same step, rather than stuttering alongside the text.

    Widgets built AFTER a scale change need none of this: they size
    themselves through :func:`scaled_px` and :func:`_set_scaled_icon_size`,
    both of which read the scale at construction.

    :param app: optional QApplication; falls back to the running instance.
    :returns: how many widgets had their icon geometry re-derived.
    """
    from PySide6.QtCore import QSize
    from PySide6.QtWidgets import QApplication

    app = app or QApplication.instance()
    if app is None:
        return 0
    scale = get_font_scale()
    moved = 0
    for widget in app.allWidgets():
        try:
            hook = getattr(widget, _ICON_SCALE_HOOK, None)
            if callable(hook):
                hook(scale)
                moved += 1
                continue
            base_w = widget.property(_KEY_ICON_BASE_W)
            if base_w is None:
                continue
            base_h = widget.property(_KEY_ICON_BASE_H) or base_w
            size = QSize(_scaled_side(int(base_w), scale),
                         _scaled_side(int(base_h), scale))
            if widget.iconSize() != size:
                widget.setIconSize(size)
                moved += 1
        except (RuntimeError, AttributeError, TypeError, ValueError):
            continue
    return moved



#: ``"auto"``    the 6 px hot strip on the left edge reveals the app list
#:               on dwell and hides it again.
#: ``"locked"``  the app list is a real column in the window layout: it
#:               never slides, never covers the page, and never has to be
#:               summoned. This is the default; users with narrower screens
#:               can switch to hover reveal or hide it completely.
#: ``"hidden"``  no strip, no reveal, no column. Apps stay reachable from
#:               the spaCR menu, Ctrl+1..9 and the command palette — a
#:               dock you cannot summon must not be a dead end.
VALID_DOCK_MODES = ("locked", "hidden")

#: HIDDEN UNTIL ASKED FOR. The dock is a permanent 220 px column, and every
#: app in it is already reachable from the spaCR menu, Ctrl+1..9 and Ctrl+K --
#: so a first run spends that width on navigation nobody has asked for yet.
#: The "All apps" action carries a tooltip saying where to turn it on.
DEFAULT_DOCK_MODE = "hidden"

#: Withdrawn: the dock used to slide in over the page when the pointer rested
#: against the left edge. It overlaid the home screen -- the module tiles sat
#: underneath it and did not move aside -- and it drew a second container
#: behind the dock's own panel. A stored ``auto`` reads as ``locked`` rather
#: than being refused, so an existing settings file keeps working and gets the
#: column it was already half-asking for.
#:
#: NOT the new default. ``auto`` was somebody CHOOSING to have a dock, and the
#: default changing underneath them is not a reason to take theirs away.
RETIRED_DOCK_MODES = {"auto": "locked"}


def get_dock_mode() -> str:
    """How the left app dock behaves — one of :data:`VALID_DOCK_MODES`.

    A withdrawn mode is MIGRATED rather than rejected; see
    :data:`RETIRED_DOCK_MODES`.
    """
    raw = str(_settings().value(_KEY_DOCK_MODE, DEFAULT_DOCK_MODE))
    raw = RETIRED_DOCK_MODES.get(raw, raw)
    return raw if raw in VALID_DOCK_MODES else DEFAULT_DOCK_MODE


def set_dock_mode(mode: str) -> None:
    """Persist a valid left-navigation dock mode.

    A withdrawn mode is accepted and stored as its replacement, so code that
    still names one is migrated rather than made to raise.

    :param mode: one of :data:`VALID_DOCK_MODES`, or a retired mode from
        :data:`RETIRED_DOCK_MODES`, which is stored as its replacement;
        anything else raises :class:`ValueError`.
    """
    mode = RETIRED_DOCK_MODES.get(mode, mode)
    if mode not in VALID_DOCK_MODES:
        raise ValueError(f"unknown dock mode {mode!r}. "
                          f"Choose from {VALID_DOCK_MODES}.")
    _settings().setValue(_KEY_DOCK_MODE, mode)



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
    """Store the requested opacity. Accepts 0.0-1.0; clamped, then rounded.

    :param fraction: the opacity from 0.0 to 1.0, stored as a whole percentage;
        an unparseable value stores :data:`DEFAULT_PANE_OPACITY_PCT`.
    """
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
    """Persist the input-hashing choice.

    :param on: true to hash a run's inputs and outputs for the manifest, false
        not to; stored as a ``bool``.
    """
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

    :param on: true to turn it on, false to turn it off; stored as a ``bool``.
    """
    settings = _settings()
    settings.setValue(_KEY_FIELD_FADE, bool(on))
    settings.sync()
    try:
        from .widgets.field_fade import invalidate_field_fade
        invalidate_field_fade()
    except Exception:
        pass



def get_color_blind_mode() -> str:
    """Return the active colour-vision mode, falling back to ``off``."""
    raw = str(_settings().value(_KEY_CB_MODE, DEFAULT_CB_MODE))
    return raw if raw in VALID_CB_MODES else DEFAULT_CB_MODE


def set_color_blind_mode(mode: str) -> None:
    """Persist a supported colour-vision mode.

    :param mode: one of :data:`VALID_CB_MODES`; any other value raises
        :class:`ValueError`.
    """
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
        return ["#4A9EFF", "#3fb950", "#f0883e", "#a78bfa",
                "#f85149", "#e879f9", "#22d3ee", "#facc15"]
    return ["#0072B2", "#E69F00", "#009E73", "#F0E442",
            "#56B4E9", "#D55E00", "#CC79A7", "#000000"]


def _level_names(levels) -> str:
    """Render a set of logging levels as their names.

    :param levels: the level numbers.
    :returns: a sorted, comma-separated list -- sorted so the same set
        always writes the same string, which is what makes it comparable.
    """
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
    """Levels written to the log files. The master switch of the pair.

    :returns: the levels the file handler admits.

    VERBOSE LOGGING ADDS DEBUG, because otherwise the two settings
    contradict each other and the one the user did not touch wins.
    With verbose on, spaCR's loggers emit DEBUG records. Omitting DEBUG from
    the file handler would build every one of those records and then discard
    it.

    Whatever verbose means, it cannot mean "do the work and write none of
    it". It is not stored into the level preference: the user's own choice
    of levels is left exactly as they set it, and DEBUG goes away again
    when they turn verbose off.
    """
    return _with_verbose_debug(_chosen_log_file_levels())


def _chosen_log_file_levels() -> frozenset:
    """The file levels the user switched on, without verbose's DEBUG.

    :returns: the stored switch set, or the defaults when none is stored.
    """
    from ..logging_util import DEFAULT_FILE_LEVELS
    return frozenset(_parse_levels(
        _settings().value(_KEY_LOG_FILE_LEVELS, None), DEFAULT_FILE_LEVELS))


def _with_verbose_debug(levels) -> frozenset:
    """Add DEBUG to ``levels`` while verbose logging is on.

    :param levels: a set of file levels as the user chose them.
    :returns: the levels the log files actually keep.
    """
    if get_verbose_logging():
        return frozenset(levels) | {logging.DEBUG}
    return frozenset(levels)


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

    :param file_levels: the ``logging`` level numbers the log files record;
        anything other than DEBUG, INFO, WARNING, ERROR and CRITICAL is
        discarded.
    :param console_levels: the ``logging`` level numbers shown on the console;
        a level the log files do not keep is dropped.
    :returns: ``(file_levels, console_levels)`` as actually stored, which
        is not necessarily what was asked for -- a console level whose file
        level is off is dropped rather than saved and silently ignored.

    ``file_levels`` is the user's own choice, and it is stored as given.
    While verbose logging is on, the log files keep DEBUG as well, so a
    console DEBUG switch is kept and the live handlers are given DEBUG. The
    DEBUG that verbose adds is not written into the stored file levels.
    """
    from ..logging_util import (apply_level_policy, clamp_console_to_file,
                                normalise_levels)
    files = normalise_levels(file_levels)
    kept = _with_verbose_debug(files)
    console = clamp_console_to_file(console_levels, kept)
    settings = _settings()
    settings.setValue(_KEY_LOG_FILE_LEVELS, _level_names(files))
    settings.setValue(_KEY_LOG_CONSOLE_LEVELS, _level_names(console))
    apply_level_policy(kept, console)
    return files, console


#: Verbose diagnostic logging is ON unless the user turns it off.
#:
#: It used to be off because the preference installed the interpreter-wide
#: function tracer: offscreen, a usable Home took 3.05 s with verbose off and
#: 65.28 s with it on. The preference no longer installs the tracer; it only
#: raises log levels and adds no work to an ordinary call.
#:
#: Benchmarked with ``tools/spacr_startup_benchmark.py``'s workers,
#: offscreen. Every one of the 45 registered modules was opened, in
#: a cold and a warm process per arm, and the arms were run off, on, on, off:
#:
#:     Home, cold       on 4.00 / 4.07 s     off 4.04 / 4.18 s    budget 5 s
#:     Home, warm       on 2.07 / 1.95 s     off 2.01 / 1.92 s
#:     slowest module   on 7.08-7.26 s       off 7.02-7.54 s      budget 10 s
#:     peak RSS         on 1,165-1,177 MB    off 1,164-1,174 MB
#:
#: Every difference is inside the spread between two runs of the same arm.
#: The 500 ms event-loop stall ceiling is breached by both arms alike, and
#: verbose logging does not move it.
#:
#: The tracer is still there for developers, as
#: :func:`spacr.logging_util.enable_function_trace`, and it is not cheap:
#: with it installed, Home took 7.4 s and the next screen did not open
#: within the benchmark's 10 s hang guard.
DEFAULT_VERBOSE_LOGGING = True

#: Process-tree accounting is cheap enough to leave on.  It samples once a
#: second and installs no Python profile hook.
PERFORMANCE_LOGGING_LEVELS = ("off", "summary", "detailed")
DEFAULT_PERFORMANCE_LOGGING = "summary"


def get_verbose_logging() -> bool:
    """Return True when the user has opted into the verbose diagnostic
    logger. Toggled via the Preferences dialog; consulted at startup by
    :func:`apply_preferences_to_app`.

    The wording of that first paragraph is deliberate: a reviewed Korean
    translation of it is held in `docs/i18n/reviewed/api`, and the
    localisation audit refuses a source block that no longer matches what
    was reviewed. Changing it discards a human translation, so it is left
    exactly as it was and anything new goes below.

    Defaults to :data:`DEFAULT_VERBOSE_LOGGING`, which is on. Opening every
    module took the same time with it on as with it off, measured.
    """
    raw = _settings().value(_KEY_VERBOSE_LOG, DEFAULT_VERBOSE_LOGGING)
    if isinstance(raw, str):
        return raw.lower() in ("true", "1", "yes", "on")
    return bool(raw)


def set_verbose_logging(on: bool) -> None:
    """Persist whether package-wide diagnostic tracing is enabled.

    :param on: true to turn it on, false to turn it off; stored as a ``bool``.
    """
    _settings().setValue(_KEY_VERBOSE_LOG, bool(on))


def get_performance_logging() -> str:
    """Return the independent process-tree resource logging level.

    ``summary`` is the default: it records whole-run totals and peaks at
    roughly one sample per second. ``detailed`` additionally retains the
    bounded process/thread series. This setting never enables verbose
    logging or installs a profile hook.

    :returns: one of :data:`PERFORMANCE_LOGGING_LEVELS`.
    """
    raw = str(_settings().value(
        _KEY_PERFORMANCE_LOG, DEFAULT_PERFORMANCE_LOGGING)).strip().lower()
    return (raw if raw in PERFORMANCE_LOGGING_LEVELS
            else DEFAULT_PERFORMANCE_LOGGING)


def set_performance_logging(level: str) -> None:
    """Persist the process-tree resource logging level.

    :param level: ``off``, ``summary`` or ``detailed``.
    :raises ValueError: when ``level`` names no supported mode.
    """
    named = str(level).strip().lower()
    if named not in PERFORMANCE_LOGGING_LEVELS:
        raise ValueError(
            f"Unknown performance-logging level {level!r}. "
            f"Choose from {PERFORMANCE_LOGGING_LEVELS}.")
    _settings().setValue(_KEY_PERFORMANCE_LOG, named)


#: On, and only safe to be. A report no longer carries log lines in its
#: body: the bundle is written to a local path and the body NAMES that
#: path, so a public issue can never carry a user's logs. The preference
#: governs whether that bundle is prepared at all, every report still
#: stops at an editable preview, and nothing is sent without its own Send.
DEFAULT_SHARE_DIAGNOSTIC_LOGS = True


def get_share_diagnostic_logs() -> bool:
    """Whether an error report saves a redacted copy of the recent log.

    The copy is written to a file on this computer and the report names
    that file. The log is never posted to GitHub. Whether a report is sent
    at all is :func:`get_issue_prompt_mode`.
    """
    return _as_bool(_settings().value(_KEY_SHARE_DIAGNOSTICS,
                                      DEFAULT_SHARE_DIAGNOSTIC_LOGS),
                    DEFAULT_SHARE_DIAGNOSTIC_LOGS)


def set_share_diagnostic_logs(on: bool) -> None:
    """Persist the revocable diagnostic-log preview opt-in.

    :param on: true to let an error report save a redacted copy of the recent
        log, false not to; stored as a ``bool``.
    """
    _settings().setValue(_KEY_SHARE_DIAGNOSTICS, bool(on))


#: On, because "the news section should always automatically reflect the
#: latest spaCR release news" (2026-09-24) and a wheel's bundled notes stop
#: at the release before its own. Nothing is sent: the request is a GET of
#: the public releases list, unauthenticated, at most once a day, on a
#: worker thread, after Home has been drawn. Off leaves the panel exactly
#: as it was -- the bundled resource, and no socket.
DEFAULT_REFRESH_NEWS = True


def get_refresh_news() -> bool:
    """Whether Home's News panel may ask GitHub for newer releases.

    Read by :meth:`spacr.qt.app.MainWindow._refresh_news`, which is the one
    place that starts the fetch. The bundled
    ``spacr/resources/release_notes.json`` is drawn either way.
    """
    return _as_bool(_settings().value(_KEY_REFRESH_NEWS,
                                      DEFAULT_REFRESH_NEWS),
                    DEFAULT_REFRESH_NEWS)


def set_refresh_news(on: bool) -> None:
    """Persist the News panel's release-refresh opt-out.

    :param on: true to let the News panel ask GitHub for newer releases, false
        to opt out; stored as a ``bool``.
    """
    _settings().setValue(_KEY_REFRESH_NEWS, bool(on))


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

    :param on: true to allow edit mode, false to forbid it; stored as a
        ``bool``.
    """
    settings = _settings()
    settings.setValue(_KEY_DB_EDIT, bool(on))
    settings.sync()



DEFAULT_SHOW_ALPHA = True
DEFAULT_SHOW_BETA = True


def get_show_alpha() -> bool:
    """Whether Alpha modules and settings are visible (default: ``True``)."""
    return _as_bool(_settings().value(_KEY_SHOW_ALPHA, DEFAULT_SHOW_ALPHA),
                    DEFAULT_SHOW_ALPHA)


def set_show_alpha(on: bool) -> None:
    """Show or hide modules and settings classified as Alpha.

    :param on: true to show Alpha modules and settings, false to hide them;
        stored as a ``bool``.
    """
    _settings().setValue(_KEY_SHOW_ALPHA, bool(on))


def get_show_beta() -> bool:
    """Whether Beta modules and settings are visible (default: ``True``)."""
    return _as_bool(_settings().value(_KEY_SHOW_BETA, DEFAULT_SHOW_BETA),
                    DEFAULT_SHOW_BETA)


def set_show_beta(on: bool) -> None:
    """Show or hide modules and settings classified as Beta.

    :param on: true to show Beta modules and settings, false to hide them;
        stored as a ``bool``.
    """
    _settings().setValue(_KEY_SHOW_BETA, bool(on))


def maturity_is_visible(stage: str) -> bool:
    """Return whether a maturity stage should be present in the UI.

    Unknown stages are treated as stable. Stable features cannot be hidden;
    the two preferences are deliberately scoped to unfinished features.

    :param stage: the maturity stage, e.g. ``"alpha"`` or ``"beta"``; matched
        case-insensitively, and an empty value or any other stage counts as
        stable.
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



def apply_preferences_to_app(app=None) -> None:
    """Re-apply language, theme and font scale to ``QApplication``.

    Called at startup from :func:`spacr.qt.app.launch`, and again
    whenever the user changes a preference (via :class:`PreferencesDialog`
    ``accepted`` signal).

    :param app: optional QApplication. Falls back to
        ``QApplication.instance()``.
    """
    from PySide6.QtWidgets import QApplication

    from .theme import (
        apply_qpalette,
        apply_stylesheet_per_window,
        clear_widget_qss_overlays,
        set_widget_qss_context,
        stylesheet,
        widget_qss_names,
        window_stylesheet,
    )

    app = app or QApplication.instance()
    if app is None:
        return

    app.setProperty("spacrLanguage", get_language())

    try:
        apply_workspace_preference()
    except Exception:                                       # noqa: BLE001
        LOG.debug("could not push the workspace preference", exc_info=True)

    theme = resolve_effective_theme()
    scale = get_font_scale()
    pane_opacity = get_pane_opacity()

    background = theme_background_path(theme)

    try:
        from .widgets.field_fade import (install_field_fade,
                                         invalidate_field_fade)
        invalidate_field_fade()
        install_field_fade(app)
    except Exception:
        LOG.exception("Could not install the field fade")

    try:
        from .tooltip_policy import (install_tooltip_policy,
                                     invalidate_tooltip_policy)
        invalidate_tooltip_policy()
        install_tooltip_policy(app)
    except Exception:
        LOG.exception("Could not install the tooltip policy")

    style_signature = (
        str(theme),
        float(scale),
        pane_opacity,
        str(background or ""),
        bool(get_field_fade_enabled()),
        widget_qss_names(),
    )
    style_changed = (
        getattr(app, "_spacr_preferences_style_signature", None)
        != style_signature
        or window_stylesheet(app)
        != getattr(app, "_spacr_preferences_stylesheet", None)
        or bool(app.styleSheet())
    )
    if style_changed:
        set_widget_qss_context(app, theme, scale, pane_opacity)
        apply_qpalette(app, theme=theme,
                       follow_system=get_theme() == "system")
        sheet = stylesheet(
            theme=theme, font_scale=scale, background=background,
            surface_opacity=pane_opacity, load_widget_registrars=False)
        clear_widget_qss_overlays(app)
        apply_stylesheet_per_window(app, sheet)
        setattr(app, "_spacr_preferences_style_signature", style_signature)
        setattr(app, "_spacr_preferences_stylesheet", sheet)

        try:
            from .widgets.field_fade import repaint_fields
            repaint_fields(app)
        except Exception:
            pass
        try:
            import sys as _sys
            if _sys.modules.get(__package__ + ".widgets.preview_scale"):
                from .widgets.preview_scale import refresh_all_preview_scales
                refresh_all_preview_scales()
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not re-scale the previews", exc_info=True)
    from .button_roles import install_button_roles
    install_button_roles(app)

    apply_ambient_preferences(app)

    try:
        import sys as _sys
        if (_sys.modules.get(__package__ + ".sound") is not None
                or get_sound_enabled()):
            from .sound import apply_sound_preferences
            apply_sound_preferences(app)
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not apply the sound preferences", exc_info=True)

    try:
        _rescale_icon_sizes(app)
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not re-derive the icon sizes", exc_info=True)

    try:
        from .widgets.console_panel import ConsolePanel
        for widget in app.allWidgets():
            if isinstance(widget, ConsolePanel):
                widget.apply_zoom()
    except Exception:
        pass

    try:
        from .verbose_logger import (
            apply_verbose_logging, _ensure_file_handler,
        )
        _ensure_file_handler()
        apply_verbose_logging(get_verbose_logging())
        from ..logging_util import apply_level_policy
        apply_level_policy(get_log_file_levels(), get_log_console_levels())
    except Exception:
        pass



def confirm_resource_action(action: str, parent=None) -> bool:
    """Ask before doing ``action``, by saying what it will do.

    The dialog states the steps in the order they will happen and what the
    action cannot do (see
    :func:`spacr.qt.resource_cleanup.confirmation_text`), and its accept
    button is labelled with the action rather than "OK". "Are you sure?"
    is not a question anybody can answer: a user cannot consent to an
    unnamed action, and a button called OK does not name one.

    Cancel is the default, so a stray Return key does nothing.

    :param action: which clean-up to confirm: ``"ram"``, ``"vram"``, ``"cpu"``
        or ``"disk"``.
    :param parent: the widget the message box is parented to, or ``None``.
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


#: The one worker the disk readout uses, for every caller.
#:
#: MODULE-SCOPED ON PURPOSE, AND NOT PARENTED TO THE DIALOG. A JobRunner
#: collected while its QThread is still running ABORTS the process (see
#: :mod:`spacr.qt.job_runner`), and a runner parented to the Preferences
#: dialog is held alive by nothing else: closing Preferences during a slow
#: read releases the dialog, its Python attributes, the runner and finally
#: the QThread wrapper -- while the stat is still in the kernel. Twenty
#: seconds on a sleeping automount is exactly the window in which a user
#: gives up and closes the dialog, so that is the likely case, not the
#: unlikely one. Held here instead, the runner outlives every dialog and the
#: report is dropped by :func:`_still_asking` rather than by a crash.
_DISK_RUNNER = None
#: Whether :data:`_DISK_RUNNER` was built to use a thread.
_DISK_RUNNER_THREADED = False
#: Runners replaced by :func:`_disk_report_runner`, kept forever. See there.
_RETIRED_DISK_RUNNERS = []


def _disk_report_runner():
    """The worker that reads the disk, made once for the whole module.

    ``user_visible=False``: the run banner on Home is for module runs, and
    ``spacr/qt/widgets/home.py`` filters on exactly this flag. This runner
    carries the disk readout and nothing else -- it is not shared with any
    user-started job that a banner would then hide -- and the readout is a
    handful of stat calls behind a modal dialog that covers Home anyway. The
    activity spinner still turns, so something IS visibly running.

    Unthreaded when there is no ``QApplication``, because then there is no
    event loop to deliver the callback on -- and no GUI thread to protect
    either, which is the only reason the thread was wanted.
    """
    global _DISK_RUNNER, _DISK_RUNNER_THREADED
    from PySide6.QtWidgets import QApplication
    from .job_runner import JobRunner

    threaded = QApplication.instance() is not None
    if _DISK_RUNNER is None or _DISK_RUNNER_THREADED != threaded:
        if _DISK_RUNNER is not None:
            try:
                _DISK_RUNNER.cancel()
            except RuntimeError:
                pass
            _RETIRED_DISK_RUNNERS.append(_DISK_RUNNER)
        _DISK_RUNNER = JobRunner(None, threaded=threaded,
                                 app_key="disk report", user_visible=False)
        _DISK_RUNNER_THREADED = threaded
    return _DISK_RUNNER


def _disk_button(parent=None):
    """The "Check disk space" button inside ``parent``, if it is there."""
    if parent is None:
        return None
    try:
        from PySide6.QtWidgets import QPushButton
        return parent.findChild(QPushButton, "CheckDiskButton")
    except (AttributeError, RuntimeError):
        return None


def _still_asking(parent) -> bool:
    """True while ``parent`` is still on screen to be answered.

    A report that lands after the user closed Preferences has no one left to
    show it to: the question was abandoned, and a message box arriving out
    of a dialog that is gone is not the answer to anything. ``None`` -- the
    caller that owns no dialog -- is always answered.
    """
    if parent is None:
        return True
    try:
        return bool(parent.isVisible())
    except RuntimeError:
        return False
    except AttributeError:
        return True


def _start_disk_report(parent=None) -> None:
    """Read the disk on a worker thread and report it when it lands.

    WHY THIS IS NOT A PLAIN CALL, which is what it was until 2026-09-04.
    :func:`spacr.qt.resource_cleanup.disk_report` asks
    :func:`~spacr.qt.resource_cleanup.project_paths` for every folder the
    project touches — the source folders every module remembers, read back
    out of QSettings — and then does ``os.stat`` and ``shutil.disk_usage`` on
    each one. Those are paths the USER chose. Measured on one workstation that
    day: one of them was under ``/nas_mnt``, an ``autofs`` mount
    whose share was asleep, and a single stat on it had NOT RETURNED AFTER
    TWENTY SECONDS — the stat is what triggers the automount.

    Run from the button's ``clicked`` slot, that is the whole interface
    frozen between the confirmation box and the result box, with no traceback
    to show for it, because a stalled event loop is not a crash.

    :mod:`spacr.qt.path_probe` is the wrong tool here and deliberately not
    used: it answers a cheap yes/no optimistically from a cache, and a disk
    readout needs real device ids and real byte counts. Work that must
    genuinely touch the disk belongs on a worker, not behind a cache.

    ``DiskReport`` and ``DiskEntry`` are frozen dataclasses of plain numbers,
    so the result crosses the thread boundary safely and the message box is
    still opened on the GUI thread, by the callback.

    THE FAILURE PATH IS THE CALLBACK PATH. ``JobRunner`` calls ``on_done``
    only for a job that succeeded, so a worker that raised would leave the
    button disabled and reading "Reading the disk…" for the rest of the
    session — the one failure a user cannot recover from, because the button
    that would retry is the one that is stuck. The read is therefore wrapped
    so the worker returns its exception instead of raising it, and the
    callback always runs: it gives the button back first and decides what to
    show second.
    """
    from . import resource_cleanup
    from .i18n import tr

    button = _disk_button(parent)
    resting_tip = None
    if button is not None:
        try:
            resting_tip = button.toolTip()
            button.setEnabled(False)
            button.setToolTip(tr("Reading the disk…"))
        except RuntimeError:
            button = None

    def read():
        """On the worker thread. Returns the report, or the exception.

        Returned rather than raised so the callback below is reached either
        way; see the failure paragraph above. ``disk_report`` is looked up
        here, not captured, so a caller that replaces it still gets its own.
        """
        try:
            return resource_cleanup.disk_report()
        except Exception as exc:  # noqa: BLE001 — returned, not swallowed
            return exc

    def restore() -> None:
        """Give the button back. The C++ half may be gone; that is fine."""
        if button is None:
            return
        try:
            button.setEnabled(True)
            button.setToolTip(resting_tip or "")
        except RuntimeError:
            pass

    def done(report) -> None:
        """On the GUI thread, with whatever the worker came back with."""
        restore()
        if isinstance(report, BaseException):
            LOG.warning("the disk could not be read", exc_info=report)
            return
        if not _still_asking(parent):
            LOG.debug("the disk report outlived the dialog that asked for it")
            return
        try:
            _show_resource_result("disk", report, parent)
        except RuntimeError:
            LOG.debug("the disk report outlived its dialog", exc_info=True)

    if not _disk_report_runner().submit(read, done):
        restore()


def run_resource_action(action: str, parent=None):
    """Confirm ``action``, run it, and report the measured result.

    :param action: which clean-up to run: ``"ram"``, ``"vram"``, ``"cpu"`` or
        ``"disk"``.
    :param parent: the widget the confirmation and result dialogs are parented
        to, or ``None``.
    :returns: the :class:`~spacr.qt.resource_cleanup.Reclaim` for "ram",
        "vram" and "cpu", or ``None`` when the user declined — in which case
        **nothing ran**. The confirmation is asked before any work is
        started, not after, which is the whole point of asking.

        ``None`` for "disk" as well, and that one is not a refusal: the disk
        readout stats folders the user chose, so it goes to a worker thread
        (:func:`_start_disk_report`) and its result arrives in the same
        message box a moment later rather than in this return value. The
        other three free memory and threads and touch no path, so they stay
        inline where their before/after measurements are taken.
    """
    from . import resource_cleanup
    if not confirm_resource_action(action, parent):
        return None
    if action == "disk":
        _start_disk_report(parent)
        return None
    result = {
        "ram": lambda: resource_cleanup.clear_ram(aggressive=True),
        "vram": resource_cleanup.clear_vram,
        "cpu": resource_cleanup.clear_cpu,
    }[action]()
    _show_resource_result(action, result, parent)
    return result



#: What each Preferences row means, keyed by the label it carries.
#:
#: ON THE LABEL, NOT THE FIELD, and that is the house rule everywhere in
#: spaCR: the words are what a reader points at when they want to know
#: what something is, and a tooltip on the control is one they find only
#: after reaching for it. :func:`explain_every_row` moves any that were
#: put on a field, so a row explained either way ends up explained the
#: same way.
PREFERENCE_TIPS = {
    "Debug": "Record all diagnostic messages, including internal run steps. This produces large logs and is recommended when preparing a bug report.",
    "Info": "Record one message for each major run step. Recommended for routine use.",
    "Warning": "Record conditions that may require review or intervention.",
    "Error": "Record failed operations.",
    "Critical": "Record failures that stop the run.",
    "Font family": "Typeface used for text in saved figures.",
    "Font size": "Base font size for saved figures. Title, label and tick sizes are scaled relative to this value.",
    "Title size": "Figure-title size relative to the base font size.",
    "Label size": "Axis-label size relative to the base font size.",
    "Tick size": "Axis-tick label size relative to the base font size.",
    "Palette": "Sequence of colours used when a figure contains multiple series.",
    "Background": "Figure background colour. A transparent background uses the colour of the destination document or interface.",
    "Foreground": "Axis lines, ticks and text.",
    "Chrome colour": "Colour used for the figure frame, axis ticks and spines.",
    "Mark colouring": "Assign point colours from the palette or from a selected data column.",
    "Grid": "Draw grid lines behind the data.",
    "Grid colour": "Colour used for grid lines.",
    "Grid width": "Grid-line width in points.",
    "Grid style": "Solid, dashed or dotted.",
    "Spines": "Select which of the four plot-frame edges are drawn.",
    "Spine width": "Plot-frame edge width in points.",
    "Marker size": "Plotted-point area in points squared.",
    "Marker style": "Shape used for plotted points.",
    "Line width": "Plotted-line width in points.",
    "Jitter width": "Horizontal displacement applied to overlapping points.",
    "Point alpha": "Point opacity. Values below 1 make overlapping points appear darker.",
    "Bar alpha": "Bar opacity.",
    "Fill colour": "Interior colour of bars and histogram bins.",
    "Edge colour": "Outline colour of bars and histogram bins.",
    "Edge width": "Outline width in points.",
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
    "Annotate": "Display values on the plot.",
    "Annotate cells": "Display each cell's value in a heatmap.",
    "Label top n": "Number of highest-ranked points labelled by name.",
    "Legend": "Display a legend at the selected location.",
    "Per row": "Number of panels in each row of a grid figure.",
    "Lock axis scales": "Whether one y unit is drawn the same length as one x "
                   "unit ('equal', which is what keeps a plate's wells "
                   "square), or the panel is filled instead ('auto'). This "
                   "locks the axis scales, which is a statement about the "
                   "data; the proportions of the figure are 'Page shape'.",
    "Page shape": "Aspect of the saved page.",
    "Dpi": "Resolution of the saved image in pixels per inch. A value of 300 is commonly required for print.",
    "Format": "File format used when saving figures.",
    "Tight layout": "Adjust margins to fit labels within the saved figure.",
    "Theme": "Application colour scheme. 'Follow system' uses the desktop colour scheme.",
    "Font scale": "Scale interface text independently of saved-figure font sizes.",
    "GUI scale": GUI_SCALE_TIP,
    "Colour-blind mode": "Use interface and figure colours designed to remain distinguishable for common colour-vision deficiencies.",
    "Module visibility": "Select the module maturity levels shown in navigation: stable only, or stable with beta and alpha modules.",
    "Show busy spinner after": "Delay before displaying the busy indicator for a running task.",
    "Page opacity": "Page opacity relative to the animated background.",
    "Animation detail": "Backdrop rendering detail. Reduce this value if animation affects interface performance.",
    "Pattern": "Which fractal spaceout draws. Orbit fold is an orbit-fold map antialiased across four frames; fold-inversion cascade is a Kaliset-like fold and sphere inversion coloured by three orbit traps, travelling through two overlapping scale windows so it never resets. The cascade takes four samples of one instant per pixel, so it costs about four times as much and runs at a lower frame rate by design. Space is forward flight through a dark star field with six parallax layers and three object slots that pass by -- mostly stars, occasionally a lit planet or a bright sun. It is mostly empty sky, so it is the cheapest option and the one that competes least with what you are reading. Mandelbrot is a continuous deep zoom into one point on the set's boundary, rendered by perturbation around a high-precision reference orbit -- which is what lets it keep descending past the depth a float can address, hundreds of decades in, still finding structure. GPU only: it needs a texture of the reference orbit.",
    "Backend": "Which renderer draws the fractal. GPU is a shader and is far cheaper; it needs vispy and a real display, and falls back to the CPU renderer when either is missing. Automatic picks the GPU when it can and says below which one this machine will get.",
    "Quality": "How much detail the fractal is asked for. Balanced costs less per frame; high adds an iteration to the fractal and raises the internal resolution. Automatic chooses from the number of cores on the CPU renderer and uses balanced on the GPU.",
    "Scale": "A resource multiplier for the CPU renderer's internal resolution, applied before it adapts. Below 1.0 draws fewer pixels and scales them up; above 1.0 draws more. It does not change what the fractal looks like, only how finely it is sampled. The GPU renderer ignores it.",
    "Speed": "How fast the view travels inward. It scales the depth the fractal is sampled at, so a higher number moves through the structure sooner; it does not change the frame rate or the cost of a frame.",
    "Dream": "How much the pattern warps, drifts and shears as it travels. 0.0 is a still camera moving straight in; 1.5 is the default, and higher values exaggerate the motion without a fixed ceiling. It costs nothing extra to raise.",
    "Variable speed": "Let the travel speed breathe instead of holding one value. It modulates the speed above rather than replacing it, so the number you set is still the middle of the range.",
    "Interface font": "The weight the interface is drawn in. spaCR ships Open Sans and uses it everywhere, so the application looks the same whatever fonts the machine has. Light is thinner and suits a large high-resolution display; Regular is easier to read on a small or low-resolution one. Bold stays available to anything that asks for emphasis.",
    "Animation blur": "Blur applied to background shapes.",
    "Animation speed": "Background-animation speed.",
    "Animation size": "Size of background shapes.",
    "Animation density": "Number of background shapes.",
    "Rim length": "Fraction of a card border covered by the moving highlight.",
    "Rim chase": "Responsiveness of the border highlight to pointer movement.",
    "Rim cycle": "Duration of one border-highlight cycle.",
}


def _widget_is_alive(widget) -> bool:
    """Whether ``widget``'s C++ half still exists.

    The same check :func:`spacr.qt.screens.settings_model._widget_is_alive`
    and ``live_zoom._alive`` make, for the same reason: a Python wrapper
    outlives the object it wraps, and reading through it is undefined rather
    than an exception.

    :param widget: any Qt object, or None.
    :returns: whether it is safe to touch.
    """
    if widget is None:
        return False
    try:
        from shiboken6 import isValid
        return bool(isValid(widget))
    except Exception:                                        # noqa: BLE001
        try:
            widget.objectName()
            return True
        except RuntimeError:
            return False


def explain_every_row(dialog) -> int:
    """Put a tooltip on every Preferences LABEL. Returns how many it set.

    Two jobs, and the second is why this walks the finished dialog rather
    than being written at each call site: it fills in from
    :data:`PREFERENCE_TIPS`, and it MOVES a tooltip that was put on the
    control to the label beside it. A row explained either way ends up
    explained the same way, and a row added later without a tooltip is
    reported by the test rather than passing unnoticed.

    :param dialog: the finished Preferences dialog; every ``QFormLayout``
        inside it is walked, and each row whose label is a ``QLabel`` is given
        a tooltip when one is known.
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
            if not _widget_is_alive(label) or not _widget_is_alive(field):
                continue
            if not isinstance(label, QLabel) or field is None:
                continue
            text = (label.text() or "").replace("&", "").strip()
            tip = PREFERENCE_TIPS.get(text, "")
            is_action = isinstance(field, (QPushButton, QToolButton))
            if not tip:
                tip = (field.toolTip() or "").strip()
            if not tip:
                continue
            label.setToolTip(tr(tip))
            if is_action:
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
        """Build the dialog with the UI language resolved once.

        The scope is the whole reason this wrapper exists. Building this
        dialog was measured asking the preference store what language the
        interface is in 346 times, through 415 ``QSettings`` reads, and none
        of those answers could differ: nothing runs between them.

        :param parent: parent widget, or ``None``.
        :returns: the dialog, ready to ``exec``.
        """
        from .i18n import ui_language_resolved_once
        global _settings
        shadowed = _settings
        if _SAFE_MODE:
            _settings = lambda: QSettings(_ORG, _APP)
        try:
            with ui_language_resolved_once():
                return cls._build_the_dialog(parent)
        finally:
            _settings = shadowed

    @classmethod
    def _build_the_dialog(cls, parent=None):
        """Build and return the preferences dialog.

        A ``__new__`` returning a plain ``QDialog`` rather than an ``__init__``
        on a subclass: everything Qt is imported inside the call, so importing
        this module costs nothing until a dialog is actually asked for.

        The window is detached from the window manager's point of view, so the
        user can put it where they like -- it is still parented, still modal and
        still calls ``exec``, and only the window TYPE changes.

        :param parent: parent widget, or ``None``.
        :returns: the dialog, ready to ``exec``.
        """
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
        from .dialogs import detach_from_window_manager
        detach_from_window_manager(dlg)
        dlg.setWindowTitle(tr("spaCR — Preferences"))
        dlg.setMinimumWidth(scaled_px(460))
        outer = QVBoxLayout(dlg)

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

        form = _page("General", "PreferencesTabGeneral")
        appearance = _page("Appearance", "PreferencesTabAppearance")
        theme_tab = _page("Theme", "PreferencesTabTheme")
        animation = _page("Animation", "PreferencesTabAnimation")
        performance = _page("Performance", "PreferencesTabPerformance")
        modules = _page("Modules", "PreferencesTabModules")
        figures = _page("Figures", "PreferencesTabFigures")
        logging_form = _page("Logging", "PreferencesTabLogging")
        ai_form = _page("AI", "PreferencesTabAI")

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

        _debug_file_toggle = log_level_toggles[logging.DEBUG][0]
        _debug_file_toggle.setToolTip(
            "While verbose logging is on (Modules tab), DEBUG is always "
            "written to the log files, so this switch stays on. Turn "
            "verbose logging off to choose it yourself. Your own choice "
            "is kept for when you do.")
        _chosen_debug = [logging.DEBUG in _chosen_log_file_levels()]

        def _remember_the_debug_choice(checked) -> None:
            """Record the DEBUG file switch only while the user holds it."""
            if _debug_file_toggle.isEnabled():
                _chosen_debug[0] = bool(checked)

        _debug_file_toggle.toggled.connect(_remember_the_debug_choice)

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

        theme_combo = QComboBox()
        for label, key in theme_choices():
            theme_combo.addItem(tr(label), key)
            blurb = theme_description(key)
            if blurb:
                theme_combo.setItemData(theme_combo.count() - 1, tr(blurb),
                                        Qt.ItemDataRole.ToolTipRole)
        current = get_theme_choice()
        for i in range(theme_combo.count()):
            if theme_combo.itemData(i) == current:
                theme_combo.setCurrentIndex(i); break
        theme_tab.addRow(tr("Theme"), theme_combo)

        from .widgets.ambient import palette_label, palettes_for, theme_label
        try:
            from .widgets.ambient import (ANIMATION_CHOICES, NO_ANIMATION,
                                          animation_label)
        except ImportError:
            from .widgets.ambient import AMBIENT_THEMES
            ANIMATION_CHOICES = tuple(AMBIENT_THEMES)
            NO_ANIMATION = _no_animation_key()
            animation_label = theme_label
        try:
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
        animation.addRow(tr("Animation"), ambient_theme_combo)

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
        animation.addRow(tr("Animation palette"), ambient_palette_combo)

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
        animation.addRow(dir_label, ambient_dir_combo)

        def _sync_direction_row(*_args):
            """Show the drift direction only for the theme that travels."""
            wanted = ambient_theme_combo.currentData() == "drift"
            dir_label.setVisible(wanted)
            ambient_dir_combo.setVisible(wanted)
            key = ambient_dir_combo.currentData()
            if key is not None and drift_direction_note is not None:
                ambient_dir_combo.setToolTip(tr(drift_direction_note(key)))

        ambient_theme_combo.currentIndexChanged.connect(_sync_direction_row)
        ambient_dir_combo.currentIndexChanged.connect(_sync_direction_row)
        _sync_direction_row()

        (blur_lo, blur_hi) = _ambient_ranges()[0][0]
        (speed_lo, speed_hi) = _ambient_ranges()[1][0]
        (size_lo, size_hi) = _ambient_ranges()[2][0]
        (res_lo, res_hi) = _ambient_ranges()[3][0]
        (den_lo, den_hi) = _ambient_ranges()[4][0]

        def _percent_row(name, label_text, low, high, current, tip,
                         designed=1.0, target=None):
            """Build one labelled percentage slider and return its parts."""
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
                """Show the percentage, saying when it is the designed value.

                "100%" alone does not tell a reader that it is the one to come back to.
                """
                value.setText(f"{v}% — as designed" if v == mark
                              else f"{v}%")

            slider.valueChanged.connect(_update)
            _update(slider.value())
            column = QVBoxLayout()
            column.setContentsMargins(0, 0, 0, 0)
            column.addWidget(slider)
            column.addWidget(value)
            (animation if target is None else target).addRow(
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
        animation.addRow(tr("Setting animations"), setting_anim_check)

        tooltips_all_check = Toggle(tr("Show tooltips"))
        tooltips_all_check.setObjectName("TooltipsEnabled")
        tooltips_all_check.setToolTip(
            "Resting the pointer on a button, a field or a column header "
            "for two seconds shows a small label saying what it is. The "
            "label stays while the pointer is on it and leaves a second "
            "after the pointer goes. Cleared, no tooltip appears anywhere "
            "in spaCR."
        )
        tooltips_all_check.setChecked(get_tooltips_enabled())
        appearance.addRow(tr("Tooltips"), tooltips_all_check)

        tooltips_box_check = Toggle(tr("Tooltips box"))
        tooltips_box_check.setObjectName("TooltipsBox")
        tooltips_box_check.setToolTip(
            "Hovering a setting's title opens a box beside it with the "
            "explanation, an API link and an Animation link. The box stays "
            "while the pointer moves into it, which is what makes those two "
            "links clickable at all."
        )
        tooltips_box_check.setChecked(get_tooltips_box_enabled())
        appearance.addRow(tr("Tooltips box"), tooltips_box_check)

        tooltips_bottom_check = Toggle(tr("Tooltips bottom"))
        tooltips_bottom_check.setObjectName("TooltipsBottom")
        tooltips_bottom_check.setToolTip(
            "The same explanation appears along the bottom of the window, "
            "where a category's help already appears. It holds the LAST "
            "setting you hovered for ten seconds, so you can move the "
            "pointer down to its API link and press it."
        )
        tooltips_bottom_check.setChecked(get_tooltips_bottom_enabled())
        appearance.addRow(tr("Tooltips bottom"), tooltips_bottom_check)

        object_grid_check = Toggle(tr("Per-object settings as a table"))
        object_grid_check.setObjectName("ObjectSettingsGrid")
        object_grid_check.setToolTip(
            "78 of Mask's 201 settings are the same twenty-odd questions "
            "asked once per object type. Set, those rows are replaced by one "
            "table: a row per question, a column per object. The stored "
            "settings are identical either way, so a file written with this "
            "on is the same file written with it off. Takes effect the next "
            "time a module's form is built."
        )
        object_grid_check.setChecked(get_object_grid_enabled())
        appearance.addRow(tr("Per-object settings as a table"),
                          object_grid_check)

        def _warn_when_both_are_off() -> None:
            """Say what turning both off costs, on the rows themselves.

            Not a refusal: both off is legal and the request says so. But on
            several forms the API link inside a setting's tooltip is the only
            route from that control to its documentation, so a reader who
            clears both loses that route with nothing on screen to say why.
            The warning is on the switches because that is where the choice
            is made.
            """
            silent = not (tooltips_box_check.isChecked()
                          or tooltips_bottom_check.isChecked())
            for widget in (tooltips_box_check, tooltips_bottom_check):
                widget.setProperty("warns", "true" if silent else "false")
                widget.setToolTip(widget.toolTip().split("\n\nBOTH OFF")[0]
                                  + ("\n\nBOTH OFF: no setting will explain "
                                     "itself anywhere, and the API link in a "
                                     "setting's tooltip is the only route "
                                     "from some controls to their "
                                     "documentation. Category help is "
                                     "unaffected." if silent else ""))

        tooltips_box_check.toggled.connect(_warn_when_both_are_off)
        tooltips_bottom_check.toggled.connect(_warn_when_both_are_off)
        _warn_when_both_are_off()

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
            """Show the spinner delay in seconds, or "show immediately" at zero."""
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
            """Show the font scale as a percentage."""
            scale_value.setText(f"{v}%")
        scale_slider.valueChanged.connect(_update_scale_lbl)

        scale_row = QVBoxLayout()
        scale_row.addWidget(scale_slider)
        scale_row.addWidget(scale_value)
        _wrap = _hbox_wrap(scale_row)
        form.addRow(tr("Font scale"), _wrap)

        from PySide6.QtCore import QTimer

        from .gui_scale import change_scales

        gui_scale_slider = QSlider(Qt.Horizontal)
        gui_scale_slider.setObjectName("GuiScale")
        gui_scale_slider.setRange(int(round(GUI_SCALE_MIN * 100)),
                                  int(round(GUI_SCALE_MAX * 100)))
        gui_scale_slider.setSingleStep(5)
        gui_scale_slider.setPageStep(25)
        gui_scale_slider.setTickInterval(25)
        gui_scale_slider.setValue(int(round(get_gui_scale() * 100)))
        gui_scale_slider.setToolTip(tr(GUI_SCALE_TIP))
        gui_scale_value = QLabel(f"{gui_scale_slider.value()}%")
        gui_scale_value.setObjectName("GuiScaleValue")
        gui_scale_slider.valueChanged.connect(
            lambda v: gui_scale_value.setText(f"{v}%"))
        gui_scale_row = QVBoxLayout()
        gui_scale_row.addWidget(gui_scale_slider)
        gui_scale_row.addWidget(gui_scale_value)
        form.addRow(tr("GUI scale"), _hbox_wrap(gui_scale_row))

        scale_settle = QTimer(dlg)
        scale_settle.setObjectName("ScaleSettle")
        scale_settle.setSingleShot(True)
        scale_settle.setInterval(450)

        def _put_the_sliders_back(kept: bool) -> None:
            """After a Revert, show the values that are in force again."""
            if kept:
                return
            for slider, value in ((scale_slider, get_font_scale()),
                                  (gui_scale_slider, get_gui_scale())):
                try:
                    slider.blockSignals(True)
                    slider.setValue(int(round(value * 100)))
                    slider.blockSignals(False)
                    scale_value.setText(f"{scale_slider.value()}%")
                    gui_scale_value.setText(f"{gui_scale_slider.value()}%")
                except RuntimeError:
                    return

        def _apply_the_scales_live() -> None:
            """Apply the two scales now and ask whether to keep them."""
            gui = gui_scale_slider.value() / 100.0
            font = scale_slider.value() / 100.0
            if (abs(gui - get_gui_scale()) < 1e-9
                    and abs(font - get_font_scale()) < 1e-9):
                return
            owner = parent if parent is not None else dlg
            change_scales(owner, gui=gui, font=font,
                          on_done=_put_the_sliders_back)

        scale_settle.timeout.connect(_apply_the_scales_live)

        def _settle_unless_dragging(slider) -> None:
            """A drag applies on release; a click or key applies at once."""
            if not slider.isSliderDown():
                scale_settle.start()

        for slider in (scale_slider, gui_scale_slider):
            slider.sliderReleased.connect(scale_settle.start)
            slider.valueChanged.connect(
                lambda _v, s=slider: _settle_unless_dragging(s))

        dock_combo = QComboBox()
        for label, key in (
            ("Locked open",     "locked"),
            ("Hidden",          "hidden"),
        ):
            dock_combo.addItem(tr(label), key)
        current_dock = get_dock_mode()
        for i in range(dock_combo.count()):
            if dock_combo.itemData(i) == current_dock:
                dock_combo.setCurrentIndex(i); break
        dock_combo.setToolTip(
            "Locked open: the app list is a permanent column to the left "
            "of the page. It never covers what you are working on, and "
            "costs its own width.\n"
            "Hidden: no column. Apps stay reachable from the spaCR menu, "
            "Ctrl+1..9 and Ctrl+K."
        )
        form.addRow(tr("App dock"), dock_combo)

        opacity_slider = QSlider(Qt.Horizontal)
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
        theme_tab.addRow(tr("Page opacity"), _hbox_wrap(opacity_col))

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
        theme_tab.addRow(tr("Field fade"), field_fade_check)

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
            """Show the rim length in pixels."""
            rim_length_value.setText(tr("%d px") % int(px))

        _rim_length_says(rim_length_slider.value())
        rim_length_slider.valueChanged.connect(_rim_length_says)
        rim_length_row = QHBoxLayout()
        rim_length_row.setContentsMargins(0, 0, 0, 0)
        rim_length_row.addWidget(rim_length_slider, 1)
        rim_length_row.addWidget(rim_length_value)
        theme_tab.addRow(tr("Rim length"), _hbox_wrap(rim_length_row))

        rim_lag_slider = QSlider(Qt.Horizontal)
        rim_lag_slider.setObjectName("RimLag")
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
            """Show the rim chase as a percentage."""
            rim_lag_value.setText(tr("%d%%") % int(percent))

        _rim_lag_says(rim_lag_slider.value())
        rim_lag_slider.valueChanged.connect(_rim_lag_says)
        rim_lag_row = QHBoxLayout()
        rim_lag_row.setContentsMargins(0, 0, 0, 0)
        rim_lag_row.addWidget(rim_lag_slider, 1)
        rim_lag_row.addWidget(rim_lag_value)
        theme_tab.addRow(tr("Rim chase"), _hbox_wrap(rim_lag_row))

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
        theme_tab.addRow(tr("Rim alignment"), rim_align_combo)

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
        theme_tab.addRow(tr("Rim mode"), rim_mode_combo)

        rim_period_slider = QSlider(Qt.Horizontal)
        rim_period_slider.setObjectName("RimPeriod")
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
            """Show the rim period in seconds."""
            rim_period_value.setText(tr("%.1f s") % (int(tenths) / 10.0))

        _rim_period_says(rim_period_slider.value())
        rim_period_slider.valueChanged.connect(_rim_period_says)
        rim_period_row = QHBoxLayout()
        rim_period_row.setContentsMargins(0, 0, 0, 0)
        rim_period_row.addWidget(rim_period_slider, 1)
        rim_period_row.addWidget(rim_period_value)
        theme_tab.addRow(tr("Rim cycle"), _hbox_wrap(rim_period_row))

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
        animation.addRow(tr("Settings backdrop"), popup_backdrop_combo)

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

        verbose_check = Toggle(tr("Enable verbose logging"))
        verbose_check.setToolTip(
            "Adds spaCR's DEBUG messages to the log files in ~/.spacr/logs. "
            "It also lets cellpose report which model it loaded, and it "
            "records which buttons you pressed. That trail is what makes a "
            "bug report worth reading.\n\n"
            "Starting spaCR and opening its screens took no longer with "
            "this on. This was measured on one "
            "workstation by opening every module, once with this on and "
            "once with it off. Home was ready in "
            "about 4 seconds after a cold start both ways. "
            "The slowest module opened in "
            "about 7 seconds both ways. Pipeline runs were not timed.\n\n"
            "The Console still shows only the levels you switch on for it "
            "on the Logging tab. This switch "
            "does not trace every function call. "
            "That tracer is a separate tool for developers, and nothing "
            "here turns it on."
        )
        verbose_check.setChecked(get_verbose_logging())
        modules.addRow(tr("Diagnostics"), verbose_check)

        def _debug_follows_verbose(on) -> None:
            """Hold the DEBUG file switch on while verbose is on.

            Verbose adds DEBUG to the log files whatever the switch says,
            so the switch shows that and cannot be changed. When verbose
            goes off, the switch is given back with the user's own choice.
            """
            if on:
                _debug_file_toggle.setEnabled(False)
                _debug_file_toggle.setChecked(True)
            elif not _debug_file_toggle.isEnabled():
                _debug_file_toggle.setEnabled(True)
                _debug_file_toggle.setChecked(_chosen_debug[0])
            _sync_console_enabled(logging.DEBUG)

        verbose_check.toggled.connect(_debug_follows_verbose)
        _debug_follows_verbose(verbose_check.isChecked())

        performance_log_combo = QComboBox()
        performance_log_combo.setObjectName("PerformanceLogging")
        for label, key in (
            ("Off", "off"),
            ("Summary (recommended)", "summary"),
            ("Detailed", "detailed"),
        ):
            performance_log_combo.addItem(tr(label), key)
        performance_index = performance_log_combo.findData(
            get_performance_logging())
        performance_log_combo.setCurrentIndex(
            performance_index if performance_index >= 0 else 1)
        performance_log_combo.setToolTip(tr(
            "Records what spaCR's process tree costs without tracing every "
            "function call. Summary samples about once a second and keeps "
            "whole-run totals and peaks. Detailed also retains a bounded "
            "per-process and per-thread time series. Off starts no sampler "
            "thread. This setting is independent of verbose logging."
        ))
        modules.addRow(tr("Performance log"), performance_log_combo)

        share_diagnostics_check = Toggle(
            tr("Include redacted log excerpts in issue previews")
        )
        share_diagnostics_check.setToolTip(
            "On by default. When on, an error report saves recent log lines, "
            "with paths and credentials redacted, to a file on this computer "
            "and names that file in the report. The log itself is never "
            "posted to GitHub."
        )
        share_diagnostics_check.setChecked(get_share_diagnostic_logs())
        modules.addRow(tr("Report logs"), share_diagnostics_check)

        refresh_news_check = Toggle(
            tr("Show newer releases than this build in Home's News panel")
        )
        refresh_news_check.setObjectName("RefreshNews")
        refresh_news_check.setToolTip(
            "On by default. A build's bundled release notes stop at the "
            "release before its own, so with this on spaCR reads the public "
            "list of releases from GitHub once a day, on a background "
            "thread, after Home is drawn, and shows anything newer at the "
            "top of News. Nothing is sent and nothing is signed in. Off "
            "shows only the notes bundled in this build."
        )
        refresh_news_check.setChecked(get_refresh_news())
        modules.addRow(tr("Release news"), refresh_news_check)

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

        ai_provider_combo = QComboBox()
        ai_provider_combo.setObjectName("AiProvider")
        ai_provider_combo.addItem(tr("Automatic (first available)"), "")
        try:
            from . import ai as _ai_module

            for _p in _ai_module.configured_providers():
                ai_provider_combo.addItem(_p.label, _p.name)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not list AI providers", exc_info=True)
        _wanted = get_preferred_provider()
        _at = ai_provider_combo.findData(_wanted)
        ai_provider_combo.setCurrentIndex(_at if _at >= 0 else 0)
        ai_provider_combo.setToolTip(tr(
            "Which assistant the AI switch routes through. Automatic picks "
            "the first vendor CLI that is installed and logged in."
        ))
        ai_form.addRow(tr("Provider"), ai_provider_combo)

        ai_providers_btn = QPushButton(tr("Providers…"))
        ai_providers_btn.setObjectName("AiProvidersButton")
        ai_providers_btn.setToolTip(tr(
            "Install a vendor CLI, or sign in to one you have."
        ))

        def _open_providers():
            """Open the install/login dialog, then re-list the providers."""
            from PySide6.QtWidgets import QDialog

            from .widgets.ai_chat_panel import _ProvidersDialog

            if _ProvidersDialog(dlg).exec() != QDialog.Accepted:
                return
            chosen = ai_provider_combo.currentData()
            ai_provider_combo.clear()
            ai_provider_combo.addItem(
                tr("Automatic (first available)"), "")
            try:
                from . import ai as _ai

                for _q in _ai.configured_providers():
                    ai_provider_combo.addItem(_q.label, _q.name)
            except Exception:                                # noqa: BLE001
                LOG.debug("could not re-list AI providers", exc_info=True)
            back = ai_provider_combo.findData(chosen)
            ai_provider_combo.setCurrentIndex(back if back >= 0 else 0)

        ai_providers_btn.clicked.connect(_open_providers)
        ai_form.addRow("", ai_providers_btn)

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

        figure_save_mode_combo = QComboBox()
        figure_save_mode_combo.setObjectName("FigureSaveMode")
        figure_save_mode_combo.addItem(tr("Print (light page)"), "print")
        figure_save_mode_combo.addItem(tr("Screen (as shown)"), "screen")
        figure_save_mode_combo.addItem(
            tr("Transparent (no page)"), "transparent")
        figure_save_mode_combo.setToolTip(tr(
            "How saved figures handle their background, text and lines. "
            "Print uses a light page with dark figure elements; Screen "
            "keeps the colours shown in spaCR; Transparent removes the "
            "page and chooses figure-element colours for the current theme. "
            "SPACR_FIGURE_SAVE_MODE can temporarily override this setting "
            "for command-line and notebook runs."
        ))
        save_mode_index = figure_save_mode_combo.findData(
            get_figure_save_mode())
        figure_save_mode_combo.setCurrentIndex(
            save_mode_index if save_mode_index >= 0 else 0)
        figures.addRow(tr("Figure save mode"), figure_save_mode_combo)

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

        from ..graph_types import (DATA_SHAPES, GRAPH_NAMES, DEFAULTS,
                                   types_for)

        default_graph_combos = {}
        for shape, shape_caption in DATA_SHAPES:
            combo = QComboBox()
            combo.setObjectName(f"DefaultGraphType_{shape}")
            table_default = DEFAULTS.get(shape, "")
            combo.addItem(
                tr("Recommended — {name}").format(
                    name=tr(GRAPH_NAMES.get(table_default, table_default))),
                "")
            for kind in types_for(shape):
                combo.addItem(tr(GRAPH_NAMES.get(kind, kind)), kind)
            saved = get_default_graph_type(shape)
            index = combo.findData(saved) if saved else 0
            combo.setCurrentIndex(index if index >= 0 else 0)
            combo.setToolTip(tr(
                "Which graph is drawn first for {shape}. Right-click any "
                "graph to change that one; this chooses where they all "
                "start. Leave it on Recommended to follow spaCR's own "
                "choice, which moves as the package does."
            ).format(shape=tr(shape_caption)))
            figures.addRow(tr("First graph — {shape}").format(
                shape=tr(shape_caption)), combo)
            default_graph_combos[shape] = combo


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

        from .widgets.figure_settings import FigureStylePreferences

        style_panel = FigureStylePreferences(get_figure_style(),
                                             get_figure_style_per_graph())
        style_heading = QLabel(tr(
            "<b>Graph style</b> — how every figure is drawn, and per graph "
            "type where they differ."))
        style_heading.setWordWrap(True)
        figures.addRow(style_heading)
        figures.addRow(style_panel)

        mode_combo = QComboBox()
        mode_combo.setObjectName("PerformanceLevel")
        for key in PERFORMANCE_LEVELS:
            mode_combo.addItem(tr(PERFORMANCE_LABELS[key]), key)
        current_mode = get_performance_level()
        for i in range(mode_combo.count()):
            if mode_combo.itemData(i) == current_mode:
                mode_combo.setCurrentIndex(i); break
        performance.addRow(tr("Performance"), mode_combo)

        mode_note_label = QLabel()
        mode_note_label.setObjectName("PerformanceLevelNote")
        mode_note_label.setWordWrap(True)
        performance.addRow("", mode_note_label)

        def _sync_mode_note(*_args):
            """Say which hardware the selected mode is for.

            A selector whose levels do not name their hardware makes the user guess
            which one their machine is.
            """
            key = mode_combo.currentData()
            said = PERFORMANCE_NOTES.get(key) or mode_note(
                spacr_mode_for_level(key))
            mode_combo.setToolTip(tr(said))
            text = tr(said)
            warning = mode_warning(spacr_mode_for_level(key))
            if warning:
                text = f"{text}\n\n⚠ {tr(warning)}"
            mode_note_label.setText(text)

        mode_combo.currentIndexChanged.connect(_sync_mode_note)

        from .memory_budget import (DEFAULT_CACHE_CEILING_MB,
                                    DEFAULT_HEADROOM_MB,
                                    DEFAULT_IDLE_MINUTES, HARDWARE_NOTES,
                                    MAX_CACHE_CEILING_MB, MAX_HEADROOM_MB,
                                    MAX_IDLE_MINUTES, MIN_CACHE_CEILING_MB,
                                    MIN_HEADROOM_MB, MIN_IDLE_MINUTES,
                                    RECOMMENDED)

        def _suggestions(index: int) -> str:
            """What each level suggests for this row, as one sentence."""
            parts = []
            for level in PERFORMANCE_LEVELS:
                value = RECOMMENDED[level][index]
                shown = (f"{value:g} min" if index == 0
                         else f"{value} MB")
                parts.append(
                    f"{tr(PERFORMANCE_LABELS[level])} "
                    f"({tr(HARDWARE_NOTES[level])}): "
                    f"{shown}")
            return "\n".join(parts)

        headroom_spin = QSpinBox()
        headroom_spin.setObjectName("HeadroomMb")
        headroom_spin.setRange(MIN_HEADROOM_MB, MAX_HEADROOM_MB)
        headroom_spin.setSingleStep(256)
        headroom_spin.setSuffix(tr(" MB"))
        headroom_spin.setValue(get_headroom_mb())
        headroom_spin.setToolTip(tr(
            "How much memory must stay free for everything else on this "
            "machine. When free memory falls below it, spaCR drops what it "
            "is holding rather than competing for the last of it.\n\n"
            "Suggested:\n{levels}").format(levels=_suggestions(2)))
        performance.addRow(tr("Keep free"), headroom_spin)

        idle_spin = QDoubleSpinBox()
        idle_spin.setObjectName("CacheIdleMinutes")
        idle_spin.setRange(MIN_IDLE_MINUTES, MAX_IDLE_MINUTES)
        idle_spin.setDecimals(1)
        idle_spin.setSingleStep(1.0)
        idle_spin.setSuffix(tr(" min"))
        idle_spin.setValue(get_idle_minutes())
        idle_spin.setToolTip(tr(
            "How long something spaCR is holding — a merged frame, a loaded "
            "image, a model's weights — may sit unused before it is "
            "dropped. Zero drops it as soon as nothing needs it, which "
            "costs time when you go back to it.\n\n"
            "This does NOT unload a library: a Python C extension cannot be "
            "unloaded, and nothing here pretends otherwise. What it "
            "governs is what spaCR chose to keep.\n\n"
            "Suggested:\n{levels}").format(levels=_suggestions(0)))
        performance.addRow(tr("Drop unused after"), idle_spin)

        cache_spin = QSpinBox()
        cache_spin.setObjectName("CacheCeilingMb")
        cache_spin.setRange(MIN_CACHE_CEILING_MB, MAX_CACHE_CEILING_MB)
        cache_spin.setSingleStep(256)
        cache_spin.setSuffix(tr(" MB"))
        cache_spin.setValue(get_cache_ceiling_mb())
        cache_spin.setToolTip(tr(
            "The most spaCR will hold in caches at once. Over it, the least "
            "recently used goes first — the one least likely to be wanted "
            "next — until what remains fits.\n\n"
            "Suggested:\n{levels}").format(levels=_suggestions(1)))
        performance.addRow(tr("Cache ceiling"), cache_spin)

        # 286: a budget number still at the previous level's value moves with
        # the level; a number the user typed stays where they put it.
        _budget_level = [mode_combo.currentData()]
        _budget_spins = (idle_spin, cache_spin, headroom_spin)
        mode_combo.currentIndexChanged.connect(
            lambda *_args: _budget_follows_level(
                mode_combo, _budget_level, _budget_spins))

        font_weight = QComboBox()
        font_weight.setObjectName("InterfaceFontWeight")
        for _key, _label in (("regular", "Regular"), ("light", "Light")):
            font_weight.addItem(tr(_label), _key)
        font_weight.setCurrentIndex(
            max(0, font_weight.findData(get_interface_font_weight())))
        appearance.addRow(tr("Interface font"), font_weight)

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
                    gpu_is_available, platform_can_do_opengl, resolve_backend)

                chosen = fractal_backend.currentData()
                actual = resolve_backend(chosen)
                wants_gpu = chosen in ("auto", "gpu")
                if wants_gpu and not platform_can_do_opengl():
                    text = tr(
                        "No usable display/OpenGL context is available in "
                        "this session, so the CPU renderer runs instead. "
                        "Installing VisPy cannot enable GPU rendering in a "
                        "headless session."
                    )
                elif wants_gpu and not gpu_is_available():
                    text = tr(
                        "VisPy is not installed, so the CPU renderer runs "
                        "instead. Install the GPU renderer with pip install "
                        "\"spacr[fractal]\"."
                    )
                elif chosen == "auto":
                    text = tr(
                        "Automatic: this machine will use the {renderer} "
                        "renderer."
                    ).format(renderer=actual.upper())
                else:
                    text = tr("The {renderer} renderer.").format(
                        renderer=actual.upper())
                fractal_note.setText(text)

            fractal_backend.currentIndexChanged.connect(_sync_fractal_note)
            _sync_fractal_note()

            fractal_quality = QComboBox()
            fractal_quality.setObjectName("FractalQuality")
            for _key in FRACTAL_QUALITIES:
                fractal_quality.addItem(tr(_key), _key)
            fractal_quality.setCurrentIndex(
                max(0, fractal_quality.findData(_fractal_values["quality"])))
            fractal_quality.setToolTip(tr(
                "A level is a set of numbers, not an adjective: choosing "
                "one fills in supersampling, render scale, the iteration "
                "budget and the scale below. Every one of them stays a "
                "field you can then change — a level is a starting point, "
                "not a lock.\n\n"
                "Auto asks the machine instead, per backend, so it keeps "
                "following the hardware rather than freezing an answer."))
            fractal.addRow(tr("Quality"), fractal_quality)

            def _tenths(name, value, low=None, high=None):
                """A NUMBER FIELD, not a capped slider.

                A spin box whose maximum is 2 turns a typed 40 into 2 without
                explaining the change. These settings accept the typed number
                and let validation report clearly when it cannot be used.

                The range is opened to the widest a QDoubleSpinBox has, so
                the widget refuses nothing; `explain_a_fractal_number` is
                what decides whether a value can be used, and says why when
                it cannot.
                """
                box = QDoubleSpinBox()
                box.setObjectName(name)
                box.setRange(-1e12, 1e12)
                box.setSingleStep(0.05)
                box.setDecimals(4)
                box.setKeyboardTracking(False)
                box.setValue(float(value))
                return box

            def _whole(name, value):
                """A whole-number field, equally uncapped."""
                box = QSpinBox()
                box.setObjectName(name)
                box.setRange(-2_000_000_000, 2_000_000_000)
                box.setKeyboardTracking(False)
                box.setValue(int(value))
                return box

            fractal_scale = _tenths("FractalScale",
                                    _fractal_values["scale"], 0.25, 2.0)
            fractal.addRow(tr("Scale"), fractal_scale)
            fractal_speed = _tenths("FractalSpeed",
                                    _fractal_values["speed"], 0.15,
                                    MAX_FRACTAL_SPEED)
            fractal.addRow(tr("Speed"), fractal_speed)
            fractal_dream = _tenths("FractalDream",
                                    _fractal_values["dream"], 0.0, 1.5)
            fractal.addRow(tr("Dream"), fractal_dream)

            fractal_variable = Toggle()
            fractal_variable.setObjectName("FractalVariableSpeed")
            fractal_variable.setChecked(bool(_fractal_values["variable_speed"]))
            fractal.addRow(tr("Variable speed"), fractal_variable)

            fractal_speed_min = None
            fractal_speed_max = None

            fractal_ss = _whole(
                "FractalSupersampling", _fractal_values["supersampling"])
            fractal_ss.setToolTip(tr(
                "Samples per pixel along each axis. 1 is none, 2 is four "
                "samples a pixel and is what the published defaults use, 3 "
                "is nine. The cost is the square of this number, so it is "
                "the first thing to turn down on a slow machine and the "
                "first to turn up on a fast one."))
            fractal.addRow(tr("Supersampling"), fractal_ss)

            fractal_path = QComboBox()
            fractal_path.setObjectName("FractalPath")
            for _key, _label in (("fixed", "Straight down"),
                                 ("guided", "Search as it goes"),
                                 ("tour", "Tour the interesting places")):
                fractal_path.addItem(tr(_label), _key)
            fractal_path.setCurrentIndex(
                max(0, fractal_path.findData(_fractal_values["path"])))
            fractal_path.setToolTip(tr(
                "Straight down descends to one point on the boundary and "
                "stays pointed at it — the steadiest picture, and what the "
                "published settings use.\n\n"
                "Search as it goes looks for somewhere more interesting "
                "every so often and moves the camera onto it. It finds more "
                "variety, and moving the camera is visible: the Steering "
                "control below sets how much.\n\n"
                "Tour the interesting places floats between twenty "
                "coordinates chosen in advance for keeping their detail "
                "over four decades of zoom, easing out of one and into "
                "the next. Dragging the view stops the tour; Ctrl+R hands "
                "the camera back to it."))
            fractal.addRow(tr("Path"), fractal_path)

            fractal_steering = _tenths(
                "FractalSteering", _fractal_values["steering"], 0.0, 1.0)
            fractal_steering.setRange(0.0, 1.0)
            fractal_steering.setSingleStep(0.05)
            fractal_steering.setToolTip(tr(
                "How much the view wanders while it descends. 0 goes "
                "straight down; 1 keeps looking for somewhere more "
                "interesting.\n\n"
                "It sets how far each move reaches, how often one happens "
                "and how long it takes, together — a move never takes more "
                "than half the gap before the next, at any setting, so it "
                "always settles rather than being caught mid-course."))
            fractal.addRow(tr("Steering"), fractal_steering)

            fractal_depth = _tenths(
                "FractalMaxDepth", _fractal_values["max_depth"], 0.1, 23.0)
            fractal_depth.setDecimals(1)
            fractal_depth.setSingleStep(1.0)
            fractal_depth.setToolTip(tr(
                "How many factors of ten the zoom descends before it starts "
                "again. At the default speed each decade takes 24 seconds, "
                "so 21 is about eight and a half minutes — and slower "
                "Speed makes it longer.\n\n"
                "It stops rather than running for ever because the "
                "reference orbit is carried as three 32-bit floats, which "
                "reproduce it to about 4.2e-24: past roughly 23 decades the "
                "perturbation is measuring its own error and the picture "
                "turns to mush. Going deeper needs more precision, not a "
                "larger number here — which is why this one refuses to go "
                "above it."))
            fractal.addRow(tr("Depth (decades)"), fractal_depth)

            def _steering_only_matters_when_guided(*_args):
                """Grey Steering on the fixed path, where it does nothing.

                Shown rather than hidden, so it is clear that choosing the
                other path is what makes it live -- a control that vanishes
                is one the user has to rediscover.
                """
                fractal_steering.setEnabled(
                    fractal_path.currentData() == "guided")

            fractal_path.currentIndexChanged.connect(
                _steering_only_matters_when_guided)
            _steering_only_matters_when_guided()

            fractal_mandel = {}

            fractal_pointer = Toggle(tr("Mouse gravity"))
            fractal_pointer.setObjectName("FractalPointerGravity")
            fractal_pointer.setChecked(
                bool(_fractal_values["pointer_gravity"]))
            fractal_pointer.setToolTip(tr(
                "Let the backdrop follow the pointer: it drifts toward the "
                "cursor and is shoved away by a click. The backdrop never "
                "receives the click itself — it reads where the mouse is "
                "rather than taking events, so nothing on top of it loses a "
                "press."))
            fractal.addRow(tr("Pointer"), fractal_pointer)

            fractal_pointer_size = _tenths(
                "FractalPointerSize", _fractal_values["pointer_size"],
                0.0, 3.0)
            fractal_pointer_size.setToolTip(tr(
                "How far the pointer reaches, as a share of the window's "
                "short edge. 1.0 pulls the pattern within roughly one short "
                "edge of the cursor; 0 turns the reach off without turning "
                "the pointer off, so a click still shoves. The pull fades "
                "toward the edge of the reach rather than stopping at it."))
            fractal.addRow(tr("Pointer reach"), fractal_pointer_size)
            fractal_pointer_size.setEnabled(fractal_pointer.isChecked())
            fractal_pointer.toggled.connect(fractal_pointer_size.setEnabled)

            fractal_pointer_strength = None

            fractal_speed_period = None




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
            watcher.start()
            if parent is not None:
                parent.accept()
            if window is not None:
                window.close()

        def _resource_button(action, label_text, row_label):
            """Build one labelled action button for the resources row."""
            button = QPushButton(tr(label_text))
            button.setObjectName({
                "ram": "ClearRamButton", "vram": "ClearVramButton",
                "cpu": "ClearCpuButton", "disk": "CheckDiskButton",
            }[action])
            from . import resource_cleanup
            button.setToolTip(resource_cleanup.summary_text(action))
            button.clicked.connect(lambda: run_resource_action(action, dlg))
            performance.addRow(tr(row_label), button)
            return button

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
            workspace_limit.setToolTip(workspace_limit.toolTip() if copying else tr(
                "Only used when saved runs carry copies of their files."))

        workspace_combo.currentIndexChanged.connect(
            lambda _i: _workspace_copying(str(workspace_combo.currentData())))
        _workspace_copying(str(workspace_combo.currentData()))

        _resource_button("ram", "Clear RAM", "Memory")
        _resource_button("vram", "Clear VRAM", "GPU memory")
        _resource_button("cpu", "Clear CPU", "Threads")
        _resource_button("disk", "Check disk space", "Disk")

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

        sound_page = None
        if sound_is_offered():
            from .sound_preferences import SoundPage
            sound_page = SoundPage(_page("Sound", "PreferencesTabSound"),
                                   dlg)

        def _the_theme_brings_its_backdrop_and_its_sound(_index=0) -> None:
            """Move the other three controls when a night theme is picked.

            THE BINDING HAS TO HAPPEN HERE AND NOT ONLY IN
            `set_theme_choice`. Save writes the Theme control and then
            writes the Animation, palette and Sound set controls straight
            after it, so a preset applied inside `set_theme_choice` would
            be overwritten three lines later by whatever the untouched
            combos still held. Moving the controls instead means the two
            paths agree and, more to the point, that the user SEES what
            the theme brought with it and can put any of it back before
            pressing Save.

            The four themes that are not night themes change nothing else,
            which is why this returns early rather than reaching for a
            default: Dark has never carried an opinion about the backdrop
            and is not being given one now.

            AND IT LEAVES THE ANIMATION ALONE WHEN THE CONTROL SAYS NONE,
            for the reason :func:`apply_night_theme` gives: a theme must
            not start something moving for a user who has turned motion
            off. The Sound set still moves, because that control decides
            WHICH sounds would play and not WHETHER any do.
            """
            choice = theme_combo.currentData()
            if not is_night_theme(choice):
                return
            night = theme_for(choice)
            if ambient_theme_combo.currentData() == NO_ANIMATION:
                if sound_page is not None:
                    sound_page.select_theme(night.sound)
                return
            for index in range(ambient_theme_combo.count()):
                if ambient_theme_combo.itemData(index) == night.ambient:
                    ambient_theme_combo.setCurrentIndex(index)
                    break
            _reload_ambient_palettes(night.ambient_palette)
            if sound_page is not None:
                sound_page.select_theme(night.sound)

        theme_combo.currentIndexChanged.connect(
            _the_theme_brings_its_backdrop_and_its_sound)

        outer.addWidget(tabs)


        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        save_button = buttons.button(QDialogButtonBox.Save)
        cancel_button = buttons.button(QDialogButtonBox.Cancel)
        if save_button is not None:
            save_button.setText(tr("Save"))
        if cancel_button is not None:
            cancel_button.setText(tr("Cancel"))
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
                _select(figure_save_mode_combo, get_figure_save_mode())
                _select(fig_format_combo, get_figure_format())
                _select(png_dpi_combo, get_figure_png_dpi())
                live_cache_spin.setValue(get_figure_live_cache())
                dynamic_check.setChecked(get_figure_dynamic())
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
                gui_scale_slider.setValue(int(round(get_gui_scale() * 100)))
                opacity_slider.setValue(
                    int(round(get_pane_opacity() * 100)))

                setting_anim_check.setChecked(
                    get_setting_animations_enabled())
                tooltips_all_check.setChecked(get_tooltips_enabled())
                field_fade_check.setChecked(get_field_fade_enabled())
                hash_check.setChecked(get_hash_inputs())
                verbose_check.setChecked(get_verbose_logging())
                _select(performance_log_combo, get_performance_logging())
                share_diagnostics_check.setChecked(
                    get_share_diagnostic_logs())
                refresh_news_check.setChecked(get_refresh_news())
                db_edit_check.setChecked(get_db_browser_editable())
                alpha_check.setChecked(get_show_alpha())
                beta_check.setChecked(get_show_beta())
                if sound_page is not None:
                    sound_page.reset()
            finally:
                _settings = original

        reset_button.clicked.connect(_reset_to_defaults)

        def _save():
            """Write every preference this dialog owns, rim first.

            THE RIM GOES FIRST because every open card rereads it: doing it before
            the theme work means one repaint rather than two.
            """
            set_rim_length(rim_length_slider.value())
            set_rim_lag(rim_lag_slider.value() / 100.0)
            set_rim_alignment(rim_align_combo.currentData())
            set_rim_mode(rim_mode_combo.currentData())
            set_rim_period(rim_period_slider.value() / 10.0)
            set_popup_backdrop(popup_backdrop_combo.currentData())
            _tell_the_cards_the_rim_changed()
            set_language(language_combo.currentData())
            set_theme_choice(theme_combo.currentData())
            set_ambient_animation(ambient_theme_combo.currentData())
            palette_choice = ambient_palette_combo.currentData()
            if palette_choice is not None:
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
            set_tooltips_enabled(tooltips_all_check.isChecked())
            set_tooltips_box_enabled(tooltips_box_check.isChecked())
            set_tooltips_bottom_enabled(
                tooltips_bottom_check.isChecked())
            set_object_grid_enabled(object_grid_check.isChecked())
            set_preferred_provider(ai_provider_combo.currentData() or "")
            _tell_the_screens_the_object_grid_changed()
            scale_settle.stop()
            set_font_scale(scale_slider.value() / 100.0)
            set_gui_scale(gui_scale_slider.value() / 100.0)
            try:
                from .gui_scale import set_gui_scale_live
                set_gui_scale_live(get_gui_scale())
            except Exception:                                # noqa: BLE001
                LOG.debug("could not apply the GUI scale", exc_info=True)
            set_dock_mode(dock_combo.currentData())
            set_pane_opacity(opacity_slider.value() / 100.0)
            set_field_fade_enabled(field_fade_check.isChecked())
            set_hash_inputs(hash_check.isChecked())
            set_workspace_copy_limit_mb(workspace_limit.value())
            set_save_workspace(workspace_combo.currentData())
            set_color_blind_mode(cb_combo.currentData())
            set_verbose_logging(verbose_check.isChecked())
            set_performance_logging(performance_log_combo.currentData())
            set_share_diagnostic_logs(share_diagnostics_check.isChecked())
            set_refresh_news(refresh_news_check.isChecked())
            verbose_holds_debug = verbose_check.isChecked()
            set_log_levels(
                [level for level, (file_t, _c) in log_level_toggles.items()
                 if (_chosen_debug[0]
                     if file_t is _debug_file_toggle and verbose_holds_debug
                     else file_t.isChecked())],
                [level for level, (_f, console_t) in log_level_toggles.items()
                 if console_t.isChecked()],
            )
            set_db_browser_editable(db_edit_check.isChecked())
            set_show_alpha(alpha_check.isChecked())
            set_show_beta(beta_check.isChecked())
            set_figure_save_mode(figure_save_mode_combo.currentData())
            set_figure_format(fig_format_combo.currentData())
            for shape, combo in default_graph_combos.items():
                set_default_graph_type(shape, combo.currentData() or "")
            set_figure_png_dpi(png_dpi_combo.currentData())
            set_figure_live_cache(live_cache_spin.value())
            set_montage_columns(montage_columns_spin.value())
            set_figure_dynamic(dynamic_check.isChecked())
            style_general, style_per_graph = style_panel.values()
            set_figure_style(style_general)
            set_figure_style_per_graph(style_per_graph)
            if spaceout_enabled():
                set_fractal_settings(
                    pattern=fractal_pattern.currentData(),
                    backend=fractal_backend.currentData(),
                    quality=fractal_quality.currentData(),
                    scale=fractal_scale.value(),
                    speed=fractal_speed.value(),
                    dream=fractal_dream.value(),
                    variable_speed=fractal_variable.isChecked(),
                    speed_min=fractal_speed.value() * 0.55,
                    speed_max=fractal_speed.value() * 1.65,
                    pointer_gravity=fractal_pointer.isChecked(),
                    pointer_size=(fractal_pointer_size.value()
                                  if fractal_pointer_size is not None
                                  else 1.0),
                    pointer_strength=(1.0 if fractal_pointer.isChecked()
                                      else 0.0),
                    supersampling=int(fractal_ss.value()),
                    path=fractal_path.currentData(),
                    steering=fractal_steering.value(),
                    max_depth=fractal_depth.value(),
                    **{name: box.value()
                       for name, box in fractal_mandel.items()},
                )
                complaints = [
                    explain_a_fractal_number(name, box.value())
                    for name, box in
                    list(fractal_mandel.items())
                    + [("max_depth", fractal_depth),
                       ("supersampling", fractal_ss),
                       ("scale", fractal_scale),
                       ("speed", fractal_speed)]
                ]
                complaints = [text for text in complaints if text]
                try:
                    from .widgets.fractal_travel import (
                        apply_saved_controls, restart_the_dive)

                    apply_saved_controls()
                    restart_the_dive()
                except Exception:                            # noqa: BLE001
                    LOG.debug("could not restart the dive", exc_info=True)
                try:
                    from .widgets.ambient import (
                        rebuild_the_spaceout_backdrops)

                    rebuild_the_spaceout_backdrops()
                except Exception:                            # noqa: BLE001
                    LOG.debug("could not rebuild the backdrop",
                              exc_info=True)
                if complaints:
                    from PySide6.QtWidgets import QMessageBox

                    QMessageBox.warning(
                        dlg, tr("Some numbers cannot be used"),
                        tr("These were saved as the nearest value that "
                           "works:") + "\n\n" + "\n".join(complaints))
            set_interface_font_weight(font_weight.currentData())
            set_performance_level(mode_combo.currentData())
            _save_budget_for_level(mode_combo.currentData(),
                                   idle_spin.value(), cache_spin.value(),
                                   headroom_spin.value())
            if sound_page is not None:
                sound_page.save()
            apply_preferences_to_app()
            _refresh_owner_window(parent)
            dlg.accept()

        buttons.accepted.connect(_save)
        buttons.rejected.connect(dlg.reject)
        from .widgets.hint_bar import HintBar
        hints = HintBar(parent=dlg)
        layout = dlg.layout()
        row_of_buttons = layout.indexOf(buttons)
        if row_of_buttons >= 0:
            layout.insertWidget(row_of_buttons, hints)
        else:
            layout.addWidget(hints)
        explain_every_row(dlg)
        _everything_explains_itself_in_the_strip(dlg, hints)
        return dlg


def _everything_explains_itself_in_the_strip(dialog, bar) -> int:
    """Move every remaining tooltip in ``dialog`` into ``bar``.

    :param dialog: the finished Preferences dialog.
    :param bar: its :class:`~spacr.qt.widgets.hint_bar.HintBar`.
    :returns: how many were moved, so a test can assert a number.

    THE STRIP IS THE ANSWER, NOT A SECOND ONE. A control that both writes to
    the explanatory strip and pops a tooltip window answers twice, and the
    window can cover the strip it duplicates.

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
            if bar.explain(widget):
                moved += 1
        except Exception:                                    # noqa: BLE001
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


def _tell_the_screens_the_object_grid_changed() -> int:
    """Mount or unmount the per-object table on every open module.

    A PREFERENCE THE USER CANNOT SEE TAKE EFFECT is a preference they will
    set twice. Every screen decides for itself -- a module with too few
    shared questions still declines the table -- so this only has to ask.

    :returns: how many screens changed.
    """
    try:
        from PySide6.QtWidgets import QApplication

        from .screens.app_screen import AppScreen
    except Exception:                                        # noqa: BLE001
        return 0
    application = QApplication.instance()
    if application is None:
        return 0
    changed = 0
    for widget in application.allWidgets():
        if not isinstance(widget, AppScreen):
            continue
        try:
            changed += bool(widget.apply_object_grid_preference())
        except Exception:                                    # noqa: BLE001
            LOG.debug("a screen would not retake the grid switch",
                      exc_info=True)
    return changed


def _hbox_wrap(layout):
    """Wrap a layout in a widget so it can be placed where a widget is wanted.

    :param layout: the layout to wrap.
    :returns: the widget owning it.
    """
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

    :param panel: the stable category or panel key the layout was saved under
        with :func:`set_section_layout`; converted to ``str``.
    :returns: ``{"folded": [title, ...], "sizes": [int, ...]}``, plus
        ``"steps"`` and ``"boxes"`` for a panel whose nested sections fold or
        whose boxes are draggable -- see :func:`set_section_layout`. An empty
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


def set_section_layout(panel: str, folded=(), sizes=(), steps=None,
                       boxes=None) -> None:
    """Remember which sections of ``panel`` are folded, and the divider sizes.

    :param panel: stable category or panel name under which this layout is
        stored, independently of every other panel's arrangement.
    :param folded: the titles that are folded away.
    :param sizes: the splitter's sizes, in its own order.
    :param steps: the SUB-subsections -- ``{"1": False}`` for a numbered
        workflow step folded away. A panel's nested sections
        collapse too, and a collapse that is forgotten on the way out of the
        module is a collapse the user does again every visit.
    :param boxes: dragged heights, ``{name: px at 100 % font scale}``. STORED
        UNSCALED on purpose: a user who drags the merge report to eleven
        lines and then doubles the font wants eleven lines, not half of them,
        so the number that comes back is re-scaled rather than replayed.

    Both new mappings are written only when they hold something, so a panel
    that has neither goes on producing exactly the record it always did.
    """
    import json

    raw = _settings().value(_KEY_SECTION_LAYOUT, "")
    try:
        stored = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        stored = {}
    if not isinstance(stored, dict):
        stored = {}
    record = {
        "folded": [str(title) for title in (folded or ())],
        "sizes": [int(size) for size in (sizes or ())],
    }
    if steps:
        record["steps"] = {str(key): bool(value)
                           for key, value in dict(steps).items()}
    if boxes:
        record["boxes"] = {str(key): int(value)
                           for key, value in dict(boxes).items()}
    stored[str(panel)] = record
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
    """Remember the tile width, clamped to what the grid will accept.

    :param pixels: the tile width in pixels; converted to ``int`` and clamped
        between ``MIN_CELL_PX`` and ``MAX_CELL_PX`` of
        :mod:`spacr.qt.widgets.figure_grid_view`.
    """
    from .widgets.figure_grid_view import MAX_CELL_PX, MIN_CELL_PX

    _settings().setValue(_KEY_FIGURE_GRID_SIZE,
                         max(MIN_CELL_PX, min(int(pixels), MAX_CELL_PX)))



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

    :param mode: ``"off"``, ``"reference"`` or ``"copy"``, or anything
        :func:`spacr.workspace.resolve_mode` accepts (booleans and yes/no
        aliases); unrecognised values select the default mode.
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
    """Remember the per-file copy limit, and push it down with the mode.

    :param limit: the largest file ``copy`` mode brings in, in megabytes;
        negative values store 0.0, and an unparseable value stores the
        workspace default.
    """
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
    """Store the cells-per-row count. Returns the value actually stored.

    :param columns: cells per row; converted to ``int`` and clamped to
        :data:`MONTAGE_COLUMNS_RANGE`, and an unparseable value stores
        :data:`DEFAULT_MONTAGE_COLUMNS`.
    """
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
    """Store the rim length. Returns the value actually stored.

    :param pixels: how far the lit run reaches along the rim, in pixels;
        clamped to :data:`RIM_LENGTH_RANGE`, and an unparseable value stores
        :data:`DEFAULT_RIM_LENGTH`.
    """
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
    """Store the chase fraction. Returns the value actually stored.

    :param fraction: how far the accent closes the gap to the pointer each
        frame; clamped to :data:`RIM_LAG_RANGE`, and an unparseable value
        stores :data:`DEFAULT_RIM_LAG`.
    """
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
    """Store the alignment. An unknown name stores the default instead.

    :param name: one of :data:`RIM_ALIGNMENTS`, matched after stripping and
        lower-casing.
    """
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
    """Store the rim mode. An unknown name stores the default instead.

    :param name: one of :data:`RIM_MODES`, matched after stripping and
        lower-casing.
    """
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
    """Store the pulse period. Returns the value actually stored.

    :param seconds: seconds for one pulse of ``beat`` or one hue turn of
        ``rainbow``; clamped to :data:`RIM_PERIOD_RANGE`, and an unparseable
        value stores :data:`DEFAULT_RIM_PERIOD`.
    """
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
#:
#: SEPARATE CHOICE, NOT A SHORTER LIST. What is curated here is which
#: *setting* a popup follows, not which animations exist: every name in
#: :data:`spacr.qt.widgets.ambient.AMBIENT_THEMES` is offered, alphabetically,
#: behind `off`. An animation that appeared in one of the two menus and not
#: the other would be a difference nobody decided on, so a new theme is added
#: here at the same time as there.
_KEY_POPUP_BACKDROP = "rim/popup_backdrop"
POPUP_BACKDROPS = ("off", "aurora", "blobs", "bokeh", "cells", "drift",
                   "resonance", "ripple")
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
    """Store the popup backdrop. An unknown name stores the default.

    :param name: one of :data:`POPUP_BACKDROPS`, matched after stripping and
        lower-casing.
    """
    value = str(name or "").strip().lower()
    if value not in POPUP_BACKDROPS:
        value = DEFAULT_POPUP_BACKDROP
    settings = _settings()
    settings.setValue(_KEY_POPUP_BACKDROP, value)
    settings.sync()
    return value



#: When the user last pressed **Clear** on Home's Recent runs, and **Reset**
#: on Totals, as a UTC ISO-8601 string. Empty means never.
_KEY_RUNS_CLEARED = "home/runs_cleared_utc"
_KEY_TOTALS_RESET = "home/totals_reset_utc"

#: The two watermarks, by the name the panels ask for them under.
DASHBOARD_WATERMARKS = {"runs": _KEY_RUNS_CLEARED,
                        "totals": _KEY_TOTALS_RESET}


def get_dashboard_watermark(which: str) -> str:
    """When Home's ``which`` panel was last cleared, as a UTC ISO string.

    A WATERMARK, NOT A DELETION, and that is the whole design. **Clear** on
    Recent runs and **Reset** on Totals sit beside the queue's Clear, but
    the queue holds plates waiting to start while
    these two read the run journal -- which is the record of what this
    installation has actually done, is what the Run History screen searches,
    and is what a run's `manifest.json` is for. Emptying a dashboard panel
    must not delete that.

    So the panels remember a time instead and show only what happened after
    it. The journal is untouched, Run History still has everything, and a
    user who clears by accident loses a view rather than a history.

    :param which: ``runs`` or ``totals``.
    :returns: the stored ISO string, or ``""`` for never cleared.
    """
    key = DASHBOARD_WATERMARKS.get(which)
    if key is None:
        return ""
    return str(_settings().value(key, "") or "").strip()


def set_dashboard_watermark(which: str, when: str = "") -> str:
    """Move ``which``'s watermark to ``when``, or to now when empty.

    :param which: ``runs`` or ``totals``. An unknown name is ignored.
    :param when: a UTC ISO-8601 string. Empty means "now".
    :returns: what was stored, or ``""`` when nothing was.
    """
    key = DASHBOARD_WATERMARKS.get(which)
    if key is None:
        return ""
    if not when:
        from datetime import datetime, timezone
        when = datetime.now(timezone.utc).isoformat()
    settings = _settings()
    settings.setValue(key, when)
    settings.sync()
    return when


def clear_dashboard_watermark(which: str) -> None:
    """Forget ``which``'s watermark, so its panel shows everything again.

    :param which: the Home panel, ``"runs"`` or ``"totals"`` (the keys of
        :data:`DASHBOARD_WATERMARKS`); any other name does nothing.
    """
    key = DASHBOARD_WATERMARKS.get(which)
    if key is None:
        return
    settings = _settings()
    settings.remove(key)
    settings.sync()


#: How tall the reader dragged Home's News list, in px at 100 % font scale.
#: 0 means "never dragged" and the panel uses its own default.
_KEY_NEWS_HEIGHT = "home/news_height_px"


def get_news_height() -> int:
    """The remembered height of Home's release-notes list, or 0.

    Stored in FONT-SCALE-INDEPENDENT px, so a reader who drags the box tall
    and then raises the interface zoom gets a box that is still the same
    size relative to the text in it, rather than one that keeps the pixel
    count and loses two of its four visible lines.
    """
    try:
        return max(0, int(_settings().value(_KEY_NEWS_HEIGHT, 0) or 0))
    except (TypeError, ValueError):
        return 0


def set_news_height(px: int) -> int:
    """Remember how tall Home's release-notes list was dragged.

    :param px: the list height in font-scale-independent pixels; negative
        values store 0, and an unparseable value stores nothing and returns 0.
    """
    try:
        value = max(0, int(px))
    except (TypeError, ValueError):
        return 0
    settings = _settings()
    settings.setValue(_KEY_NEWS_HEIGHT, value)
    settings.sync()
    return value


#: Sound (427). Every key is off, or quiet, on a fresh install: a scientific
#: tool that makes noise the first time it is opened, in a shared office or
#: during a talk, is a tool people learn to distrust.
_KEY_SOUND_ENABLED = "sound/enabled"
_KEY_SOUND_VOLUME = "sound/volume"
_KEY_SOUND_THEME = "sound/theme"
_KEY_SOUND_EVENT = "sound/event/{}"
_KEY_SOUND_MUSIC = "sound/music_file"

#: The master switch. Nothing is imported, constructed or played while off.
DEFAULT_SOUND_ENABLED = False

#: Master volume as a fraction of the slider, 0 to 1.
DEFAULT_SOUND_VOLUME = 0.5

#: Each event's own switch, as a fresh install has it. They only matter once
#: the master switch is on; hover and the music bed stay off even then,
#: because each is the one a user should have to ask for by name.
SOUND_EVENT_DEFAULTS = {
    "click": True,
    "hover": False,
    "run_finished": True,
    "run_failed": True,
    "bed": False,
}

#: Performance levels at which the music bed rests. They are the two that
#: switch the animated backdrop off, and a loop playing for hours is the
#: audio equivalent of one.
SOUND_BED_RESTS_AT = ("laptop", "extra_performance")


def sound_is_offered() -> bool:
    """Whether this process offers sound at all: only in spaceout mode.

    Ordinary spaCR leaves sound off and hides its preferences tab. Spaceout is
    process-local (:func:`spacr.qt.theme.enable_spaceout`, called only by
    the ``spaceout`` launcher), so this is read live and never stored.

    Does not import :mod:`spacr.qt.theme`: a process that has not imported
    it cannot have enabled spaceout, and this is asked on paths that must
    stay free of QtGui.

    :returns: ``True`` in spaceout mode, ``False`` in ordinary spaCR.
    """
    import sys as _sys

    theme = _sys.modules.get(__package__ + ".theme")
    if theme is None:
        return False
    try:
        return bool(theme.spaceout_enabled())
    except Exception:                                        # noqa: BLE001
        return False


def get_saved_sound_enabled() -> bool:
    """The stored master sound switch, whatever the mode.

    Ordinary spaCR ignores it (see :func:`get_sound_enabled`) but never
    erases it, so a user who switched sound on in spaceout finds it on the
    next time spaceout starts.

    :returns: the stored switch, default ``False``.
    """
    return _as_bool(_settings().value(_KEY_SOUND_ENABLED,
                                      DEFAULT_SOUND_ENABLED),
                    DEFAULT_SOUND_ENABLED)


def get_sound_enabled() -> bool:
    """Whether spaCR plays any sound at all. Default ``False``.

    Always ``False`` outside spaceout mode (:func:`sound_is_offered`),
    whatever is stored: ordinary spaCR has no Sound tab, so a switch the
    user cannot see must not be able to make a noise.

    :returns: the stored master switch in spaceout mode, else ``False``.
    """
    return sound_is_offered() and get_saved_sound_enabled()


def set_sound_enabled(on: bool) -> None:
    """Persist the master sound switch.

    :param on: play sounds when True.
    """
    settings = _settings()
    settings.setValue(_KEY_SOUND_ENABLED, bool(on))
    settings.sync()


def get_sound_volume() -> float:
    """The master volume, 0 to 1, clamped on read.

    :returns: the stored fraction, or :data:`DEFAULT_SOUND_VOLUME` when the
        store holds something that is not a number.
    """
    try:
        value = float(_settings().value(_KEY_SOUND_VOLUME,
                                        DEFAULT_SOUND_VOLUME))
    except (TypeError, ValueError):
        return DEFAULT_SOUND_VOLUME
    if value != value:
        return DEFAULT_SOUND_VOLUME
    return min(1.0, max(0.0, value))


def set_sound_volume(fraction: float) -> float:
    """Persist the master volume.

    :param fraction: 0 to 1; values outside are clamped.
    :returns: the value stored.
    """
    try:
        value = min(1.0, max(0.0, float(fraction)))
    except (TypeError, ValueError):
        value = DEFAULT_SOUND_VOLUME
    settings = _settings()
    settings.setValue(_KEY_SOUND_VOLUME, value)
    settings.sync()
    return value


def _sound_theme_keys() -> tuple:
    """Every sound set that can be chosen, by key."""
    from .sound_synth import SOUND_THEMES
    return tuple(SOUND_THEMES)


def get_sound_theme() -> str:
    """Which sound set plays, validated against the sets that exist.

    :returns: a key of :data:`spacr.qt.sound_synth.SOUND_THEMES`; a stored
        key that no longer exists reads as the default set.
    """
    from .sound_synth import DEFAULT_THEME
    raw = str(_settings().value(_KEY_SOUND_THEME, DEFAULT_THEME) or "")
    return raw if raw in _sound_theme_keys() else DEFAULT_THEME


def set_sound_theme(key: str) -> None:
    """Persist the chosen sound set.

    :param key: a key of :data:`spacr.qt.sound_synth.SOUND_THEMES`.
    :raises ValueError: for a key no sound set has.
    """
    if key not in _sound_theme_keys():
        raise ValueError(f"unknown sound set {key!r}. "
                         f"Choose from {_sound_theme_keys()}.")
    settings = _settings()
    settings.setValue(_KEY_SOUND_THEME, str(key))
    settings.sync()


def get_sound_event_enabled(event: str) -> bool:
    """Whether one event's sound is switched on, apart from the master.

    :param event: a key of :data:`SOUND_EVENT_DEFAULTS`.
    :returns: the stored switch.
    :raises KeyError: for an event spaCR has no sound for.
    """
    default = SOUND_EVENT_DEFAULTS[event]
    return _as_bool(_settings().value(_KEY_SOUND_EVENT.format(event), default),
                    default)


def set_sound_event_enabled(event: str, on: bool) -> None:
    """Persist one event's switch.

    :param event: a key of :data:`SOUND_EVENT_DEFAULTS`.
    :param on: play that event's sound when the master switch is on.
    :raises KeyError: for an event spaCR has no sound for.
    """
    if event not in SOUND_EVENT_DEFAULTS:
        raise KeyError(event)
    settings = _settings()
    settings.setValue(_KEY_SOUND_EVENT.format(event), bool(on))
    settings.sync()


def get_sound_music_file() -> str:
    """A WAV of the user's own to play as the music bed, or ``""``.

    Empty -- the default -- means spaCR's own synthesized bed. The file is
    NOT checked here: this is called on the GUI thread on every settings
    read, and ``spacr.qt.sound`` looks for the file on its audio thread,
    where a network home directory costs nobody a frame.

    :returns: the stored path, or ``""``.
    """
    return str(_settings().value(_KEY_SOUND_MUSIC, "") or "").strip()


def set_sound_music_file(path) -> str:
    """Persist the music file the bed plays.

    :param path: a path, or anything empty for spaCR's own music.
    :returns: the value stored.
    """
    value = str(path or "").strip()
    settings = _settings()
    settings.setValue(_KEY_SOUND_MUSIC, value)
    settings.sync()
    return value


def sound_bed_rests() -> bool:
    """Whether the current performance level silences the music bed.

    :returns: True at the levels named in :data:`SOUND_BED_RESTS_AT`.
    """
    try:
        return get_performance_level() in SOUND_BED_RESTS_AT
    except Exception:                                        # noqa: BLE001
        return False
