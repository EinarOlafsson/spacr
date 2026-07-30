"""
User-facing preferences — theme, font scale, colour-blind mode.

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
        get_space_variant, set_space_variant,
        get_cell_variant, set_cell_variant,
        get_space_seed, set_space_seed,
        space_background_path, cell_background_path,
        theme_background_path,
        get_font_scale, set_font_scale,
        get_color_blind_mode, set_color_blind_mode,
        get_db_browser_editable, set_db_browser_editable,
        get_dock_mode, set_dock_mode,
        get_pane_opacity, set_pane_opacity, effective_pane_alpha,
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
* ``pane_opacity``: int percent, default ``100``. How solid shared surfaces
  are, or the relative material strength in Glass. Clamped up to
  :func:`spacr.qt.theme.pane_alpha_floor` at paint time — the
  preference is a request, legibility is not negotiable.
* ``show_alpha`` / ``show_beta``: bool, both default ``True``. Control
  whether modules and settings at that maturity are shown. Stable features
  are always visible.
"""
from __future__ import annotations

from PySide6.QtCore import QSettings

# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------

_ORG = "spacr"
_APP = "qt"

_KEY_THEME       = "prefs/theme"
_KEY_FONT_SCALE  = "prefs/font_scale"
_KEY_CB_MODE     = "prefs/color_blind_mode"
_KEY_VERBOSE_LOG = "prefs/verbose_logging"
_KEY_DB_EDIT     = "prefs/db_browser_editable"
_KEY_DOCK_MODE   = "prefs/dock_mode"
_KEY_PANE_OPACITY = "prefs/pane_opacity"
_KEY_SHOW_ALPHA = "prefs/show_alpha"
_KEY_SHOW_BETA = "prefs/show_beta"

#: Themes with a palette of their own — mirrors
#: :data:`spacr.qt.theme.THEMES`, restated here so importing this module
#: does not pull in QtGui/QtWidgets.
PALETTE_THEMES = ("dark", "light", "space", "cell", "glass")

#: Persisted values. An existing install has ``prefs/theme`` set to one
#: of dark/light/system/space; those keep resolving exactly as before,
#: and an unrecognised value (hand-edited INI, a downgrade from a build
#: with more themes) falls back to :data:`DEFAULT_THEME` rather than
#: raising.
VALID_THEMES = PALETTE_THEMES + ("system",)
DEFAULT_THEME = "dark"

_KEY_SPACE_VARIANT = "prefs/space_variant"
_KEY_SPACE_SEED    = "prefs/space_seed"
_KEY_CELL_VARIANT  = "prefs/cell_variant"

FONT_SCALE_MIN = 0.75
FONT_SCALE_MAX = 2.00
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


def _settings() -> QSettings:
    return QSettings(_ORG, _APP)


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
    space_labels = {
        "galaxy": "Spiral galaxy",
        "sun": "Star and corona",
        "stars": "Deep starfield",
        "deep_field": "Galaxy deep field",
    }
    choices.extend(
        (f"Space — {space_labels.get(key, key.replace('_', ' ').title())}",
         f"space:{key}")
        for key in space_variants()
    )
    choices.extend(
        (f"Cell — {title_for(key)}", f"cell:{key}")
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
DEFAULT_PANE_OPACITY_PCT = 100


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
# Feature maturity — visibility of unfinished modules and settings
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
    """Re-apply the theme + font scale to a running ``QApplication``.

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
    app.setStyleSheet(stylesheet(
        theme=theme, font_scale=scale, background=background,
        surface_opacity=get_pane_opacity()))
    # Run/Propagate and Stop/Close-style buttons are tagged centrally,
    # including QDialogButtonBox buttons created after startup.
    from .button_roles import install_button_roles
    install_button_roles(app)

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
        from .widgets.toggle import Toggle

        dlg = QDialog(parent)
        dlg.setWindowTitle("spaCR — Preferences")
        dlg.setMinimumWidth(420)
        outer = QVBoxLayout(dlg)

        form = QFormLayout()

        # Theme
        theme_combo = QComboBox()
        for label, key in theme_choices():
            theme_combo.addItem(label, key)
        current = get_theme_choice()
        for i in range(theme_combo.count()):
            if theme_combo.itemData(i) == current:
                theme_combo.setCurrentIndex(i); break
        form.addRow("Theme", theme_combo)

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
        form.addRow("Font scale", _wrap)

        # The left dock — revealed on hover, pinned open, or gone.
        dock_combo = QComboBox()
        for label, key in (
            ("Reveal on hover", "auto"),
            ("Locked open",     "locked"),
            ("Hidden",          "hidden"),
        ):
            dock_combo.addItem(label, key)
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
        form.addRow("App dock", dock_combo)

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
        form.addRow("Page opacity", _hbox_wrap(opacity_col))

        # Colour-blind mode
        cb_combo = QComboBox()
        for label, key in (
            ("Off",                     "off"),
            ("Deuteranopia (red-green)", "deuteranopia"),
            ("Protanopia (red-green)",   "protanopia"),
            ("Tritanopia (blue-yellow)", "tritanopia"),
        ):
            cb_combo.addItem(label, key)
        current_cb = get_color_blind_mode()
        for i in range(cb_combo.count()):
            if cb_combo.itemData(i) == current_cb:
                cb_combo.setCurrentIndex(i); break
        form.addRow("Colour-blind mode", cb_combo)

        # Verbose logging — one toggle, wired at Save time. When on,
        # spaCR + third-party libs (cellpose, torch, PIL, matplotlib)
        # dial their loggers to DEBUG/INFO and every record echoes into
        # the active ConsolePanel. Aimed at bug reports.
        verbose_check = Toggle("Enable verbose logging")
        verbose_check.setToolTip(
            "When on, every spaCR log record — plus INFO-level chatter "
            "from cellpose, torch, PIL and matplotlib — echoes into "
            "the active app's Console. Very chatty; leave off unless "
            "you're triaging a bug."
        )
        verbose_check.setChecked(get_verbose_logging())
        form.addRow("Diagnostics", verbose_check)

        # Database Browser — off by default. The browser opens
        # measurements.db with mode=ro; this is the only switch that lets
        # it open a read-write connection at all, and even then the user
        # has to arm edit mode per session and confirm it.
        db_edit_check = Toggle("Allow editing in the Database Browser")
        db_edit_check.setToolTip(
            "Off by default. The Database Browser opens measurements.db "
            "read-only (mode=ro). With this on you can still only edit "
            "after arming 'Edit mode' for a database you chose yourself, "
            "and every change is one UPDATE scoped to one row. There is "
            "no undo — spaCR writes straight into your measurements file."
        )
        db_edit_check.setChecked(get_db_browser_editable())
        form.addRow("Database Browser", db_edit_check)

        # Feature maturity. Both are opt-out: existing users and fresh
        # installs continue to see every feature until they choose a quieter,
        # stable-only interface.
        alpha_check = Toggle("Show Alpha modules and settings")
        alpha_check.setObjectName("ShowAlphaFeatures")
        alpha_check.setToolTip(
            "Hide modules and settings that are built but not yet trusted "
            "end to end. Stable and Beta features are unaffected."
        )
        alpha_check.setChecked(get_show_alpha())

        beta_check = Toggle("Show Beta modules and settings")
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
        form.addRow("Feature maturity", _hbox_wrap(maturity_col))

        # Figures — display format (png = lighter / faster, pdf = vector +
        # editable via the figure-settings button) and the PNG resolution.
        fig_format_combo = QComboBox()
        fig_format_combo.addItem("PNG (raster, lighter)", "png")
        fig_format_combo.addItem("PDF (vector, editable)", "pdf")
        cur_fmt = get_figure_format()
        for i in range(fig_format_combo.count()):
            if fig_format_combo.itemData(i) == cur_fmt:
                fig_format_combo.setCurrentIndex(i); break
        form.addRow("Figure format", fig_format_combo)

        png_dpi_combo = QComboBox()
        for dpi in VALID_PNG_DPIS:
            png_dpi_combo.addItem(f"{dpi} dpi", dpi)
        cur_dpi = get_figure_png_dpi()
        for i in range(png_dpi_combo.count()):
            if png_dpi_combo.itemData(i) == cur_dpi:
                png_dpi_combo.setCurrentIndex(i); break
        form.addRow("PNG resolution", png_dpi_combo)

        outer.addLayout(form)

        preview = QLabel(
            "<span style='color:gray;'>Theme + font scale apply "
            "instantly on Save. Colour-blind mode affects plot colours "
            "the next time a figure is generated.</span>"
        )
        preview.setTextFormat(Qt.RichText)
        preview.setWordWrap(True)
        outer.addWidget(preview)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        outer.addWidget(buttons)

        def _save():
            set_theme_choice(theme_combo.currentData())
            set_font_scale(scale_slider.value() / 100.0)
            set_dock_mode(dock_combo.currentData())
            set_pane_opacity(opacity_slider.value() / 100.0)
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
