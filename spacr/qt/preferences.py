"""
User-facing preferences — theme, font scale, colour-blind mode.

Persistent settings backed by :class:`PySide6.QtCore.QSettings`, so
they survive app restarts. Three knobs today; more can slot in
alongside without changing consumers thanks to the small typed API
(``get_theme()`` / ``set_theme(...)`` etc.).

Wire-up:

* :func:`apply_preferences_to_app` — call once at startup and again
  whenever a setting changes; reapplies the stylesheet with the
  current theme + font scale.
* :class:`PreferencesDialog` — the modal Settings dialog opened by
  Ctrl+, (see :mod:`spacr.qt.shortcuts`).

Public API::

    from spacr.qt.preferences import (
        get_theme, set_theme,
        get_font_scale, set_font_scale,
        get_color_blind_mode, set_color_blind_mode,
        apply_preferences_to_app,
        PreferencesDialog,
    )

Values:

* ``theme``: ``"dark"`` | ``"light"`` | ``"system"`` (default ``"dark"``).
  ``"system"`` follows the reader's OS colour scheme.
* ``font_scale``: float, 1.0 = 100 %. Clamped to [0.75, 2.0].
* ``color_blind_mode``: ``"off"`` | ``"deuteranopia"`` | ``"protanopia"``
  | ``"tritanopia"`` (default ``"off"``). Swaps matplotlib rainbow /
  red-green palettes for perceptually-uniform + colour-blind-safe
  alternatives (viridis for continuous, Okabe-Ito for categorical).
"""
from __future__ import annotations

from typing import Optional

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

VALID_THEMES = ("dark", "light", "system")
DEFAULT_THEME = "dark"

FONT_SCALE_MIN = 0.75
FONT_SCALE_MAX = 2.00
DEFAULT_FONT_SCALE = 1.0

VALID_CB_MODES = ("off", "deuteranopia", "protanopia", "tritanopia")
DEFAULT_CB_MODE = "off"


def _settings() -> QSettings:
    return QSettings(_ORG, _APP)


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

def get_theme() -> str:
    raw = str(_settings().value(_KEY_THEME, DEFAULT_THEME))
    return raw if raw in VALID_THEMES else DEFAULT_THEME


def set_theme(theme: str) -> None:
    if theme not in VALID_THEMES:
        raise ValueError(f"unknown theme {theme!r}. "
                          f"Choose from {VALID_THEMES}.")
    _settings().setValue(_KEY_THEME, theme)


def resolve_effective_theme() -> str:
    """Return ``"dark"`` or ``"light"`` — resolves ``"system"`` to the
    OS colour scheme, defaulting to dark when Qt can't tell."""
    theme = get_theme()
    if theme in ("dark", "light"):
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
    try:
        raw = float(_settings().value(_KEY_FONT_SCALE,
                                        DEFAULT_FONT_SCALE))
    except (TypeError, ValueError):
        raw = DEFAULT_FONT_SCALE
    return max(FONT_SCALE_MIN, min(FONT_SCALE_MAX, raw))


def set_font_scale(scale: float) -> None:
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
# Colour-blind mode
# ---------------------------------------------------------------------------

def get_color_blind_mode() -> str:
    raw = str(_settings().value(_KEY_CB_MODE, DEFAULT_CB_MODE))
    return raw if raw in VALID_CB_MODES else DEFAULT_CB_MODE


def set_color_blind_mode(mode: str) -> None:
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
    _settings().setValue(_KEY_VERBOSE_LOG, bool(on))


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

    apply_qpalette(app, theme=theme)
    app.setStyleSheet(stylesheet(theme=theme, font_scale=scale))

    # Apply the verbose-logger preference too — cheap to re-apply, and
    # this is the one place that runs on every prefs save.
    try:
        from .verbose_logger import apply_verbose_logging
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
            QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
            QLabel, QSlider, QVBoxLayout,
        )

        dlg = QDialog(parent)
        dlg.setWindowTitle("spaCR — Preferences")
        dlg.setMinimumWidth(420)
        outer = QVBoxLayout(dlg)

        form = QFormLayout()

        # Theme
        theme_combo = QComboBox()
        for label, key in (
            ("Dark", "dark"),
            ("Light", "light"),
            ("Follow system", "system"),
        ):
            theme_combo.addItem(label, key)
        current = get_theme()
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
        verbose_check = QCheckBox("Enable verbose logging")
        verbose_check.setToolTip(
            "When on, every spaCR log record — plus INFO-level chatter "
            "from cellpose, torch, PIL and matplotlib — echoes into "
            "the active app's Console. Very chatty; leave off unless "
            "you're triaging a bug."
        )
        verbose_check.setChecked(get_verbose_logging())
        form.addRow("Diagnostics", verbose_check)

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
            set_theme(theme_combo.currentData())
            set_font_scale(scale_slider.value() / 100.0)
            set_color_blind_mode(cb_combo.currentData())
            set_verbose_logging(verbose_check.isChecked())
            apply_preferences_to_app()
            dlg.accept()

        buttons.accepted.connect(_save)
        buttons.rejected.connect(dlg.reject)
        return dlg


def _hbox_wrap(layout):
    from PySide6.QtWidgets import QWidget
    w = QWidget()
    w.setLayout(layout)
    return w
