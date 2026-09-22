"""The whole-GUI scale: one factor for every pixel spaCR draws (item 471).

WHAT IT IS, IN ONE LINE. A startup-time Qt scale factor. The "GUI scale"
preference (:func:`spacr.qt.preferences.get_gui_scale`, 10 % to 200 %,
default 100 %) is written into ``QT_SCALE_FACTOR`` before the
``QApplication`` exists, so Qt itself multiplies every logical pixel --
widget sizes, fixed sizes, margins, spacing, paddings and borders in the
style sheet, icons, pictures and text -- by the same number. Changing it
takes a restart, and Preferences offers one ("Restart now").

WHY NOT LIVE. The alternative was to route every hard-coded size through one
scaled-size helper so a change could apply on the spot. Counted on nightly
2026-09-22 across the 328 modules under ``spacr/qt``:

==============================  ======  =====
call                             sites  files
==============================  ======  =====
``setContentsMargins(``            470    144
``setSpacing(``                    489    136
``setMinimumWidth/Height(``        179     75
``setMaximumWidth/Height(``         67     46
``setFixedWidth/Height/Size(``      74     34
``setMinimumSize(``                 30     24
``QSize(``                          56     27
``resize(``                         64     37
``setIconSize(``                     6      4
``px`` in style-sheet text         486     64
  of which in ``theme.py``         177      1
==============================  ======  =====

About 1,420 geometry calls in 181 files, plus 486 pixel literals in QSS
text in 64 files. Routing all of them through a helper is a change to
nearly every screen, and every size added afterwards would have to
remember it; one missed call is a control that stays full size while its
neighbours shrink. Qt's scale factor has no such gap: it applies below all of that code, so a
size nobody routed is scaled anyway.

HOW IT COMPOSES WITH ZOOM. The existing Zoom ("Font scale",
:func:`spacr.qt.preferences.get_font_scale`) stays live and is applied on top,
inside the scaled coordinate system. The two multiply: GUI 50 % at Zoom 200 %
draws text at 13 px x 2 x 0.5 = 13 device pixels in widgets half their usual
size, which is exactly the "fit more on a laptop without shrinking the words"
combination. Neither undoes the other.

A USER'S OWN ``QT_SCALE_FACTOR`` IS KEPT. It is remembered once, in
``SPACR_GUI_SCALE_BASE``, and the preference multiplies it; a restart
inherits the remembered base, so restarting does not compound the factor.

HOW TO GET BACK FROM 10 %. Three ways, none of which needs to read the tiny
screen: ``Ctrl+Alt+0`` (see :mod:`spacr.qt.shortcuts`) puts GUI scale, Zoom
and every preview scale back to 100 % and asks to restart with Enter as the
answer; ``safespacr`` reads every preference as its default, so it starts
at 100 %; and ``SPACR_GUI_SCALE=1`` in the environment overrides the stored
value for one launch.
"""
from __future__ import annotations

import logging
import os
from typing import MutableMapping, Optional

LOG = logging.getLogger("spacr.qt.gui_scale")

#: The environment variable Qt reads for its global scale factor.
QT_ENV = "QT_SCALE_FACTOR"

#: Where the user's own ``QT_SCALE_FACTOR`` (or ``"none"``) is remembered,
#: so a restarted process multiplies the same base rather than its parent's
#: already-multiplied value.
BASE_ENV = "SPACR_GUI_SCALE_BASE"

#: A one-launch override of the stored preference, e.g. ``SPACR_GUI_SCALE=1``.
OVERRIDE_ENV = "SPACR_GUI_SCALE"

#: What this process started with, once :func:`apply_gui_scale_to_environment`
#: has run. ``None`` before that, which :func:`running_gui_scale` reads as 1.
_RUNNING: Optional[float] = None


def _parse(text) -> Optional[float]:
    """A positive float from ``text``, or ``None``."""
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def requested_gui_scale(environ: Optional[MutableMapping] = None) -> float:
    """The GUI scale this launch should use: the override, else the preference.

    :param environ: the environment to read the override from; ``os.environ``
        by default.
    :returns: a factor clamped to the preference's bounds.
    """
    from .preferences import GUI_SCALE_MAX, GUI_SCALE_MIN, get_gui_scale

    environ = os.environ if environ is None else environ
    override = _parse(environ.get(OVERRIDE_ENV, ""))
    if override is not None:
        return max(GUI_SCALE_MIN, min(GUI_SCALE_MAX, override))
    return get_gui_scale()


def apply_gui_scale_to_environment(
        environ: Optional[MutableMapping] = None) -> float:
    """Write the GUI scale into ``QT_SCALE_FACTOR``. Call before QApplication.

    At 100 % with no user factor nothing is written at all, so a default
    launch runs exactly as it did before the preference existed.

    :param environ: the environment to change; ``os.environ`` by default.
    :returns: the GUI scale applied (1.0 when none was).
    """
    global _RUNNING

    environ = os.environ if environ is None else environ
    try:
        scale = float(requested_gui_scale(environ))
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not read the GUI scale; using 100 %", exc_info=True)
        scale = 1.0

    if BASE_ENV not in environ:
        environ[BASE_ENV] = str(environ.get(QT_ENV, "") or "none")
    base = _parse(environ.get(BASE_ENV, ""))

    if abs(scale - 1.0) < 1e-6:
        if base is None:
            environ.pop(QT_ENV, None)
        else:
            environ[QT_ENV] = f"{base:g}"
    else:
        environ[QT_ENV] = f"{(base or 1.0) * scale:.4g}"
        LOG.info("GUI scale %d %% (QT_SCALE_FACTOR=%s)",
                 round(scale * 100), environ[QT_ENV])
    if environ is os.environ:
        _RUNNING = scale
    return scale


def running_gui_scale() -> float:
    """The GUI scale this process started with (1.0 if it never applied one)."""
    return 1.0 if _RUNNING is None else float(_RUNNING)


def restart_pending() -> bool:
    """Whether the stored GUI scale differs from the one on screen."""
    try:
        return abs(requested_gui_scale() - running_gui_scale()) > 1e-6
    except Exception:                                        # noqa: BLE001
        return False


def fit_window_to_gui_scale(window, scale: Optional[float] = None) -> bool:
    """Keep the window the same size on the screen at any GUI scale.

    Qt measures a window in scaled pixels, so a window asked for at 1200 px
    at 50 % would cover 600 real pixels -- the laptop would gain room inside
    a window that had halved. Dividing by the scale keeps its physical size,
    clamped to what the screen offers, and the gained room goes to the
    content.

    :param window: the main window, before or after it is shown.
    :param scale: the scale to fit for; the running one by default.
    :returns: ``True`` if the window was resized.
    """
    scale = running_gui_scale() if scale is None else float(scale)
    if abs(scale - 1.0) < 1e-6 or scale <= 0:
        return False
    try:
        from PySide6.QtWidgets import QApplication

        handle = window.screen() or QApplication.primaryScreen()
        width = round(window.width() / scale)
        height = round(window.height() / scale)
        if handle is not None:
            available = handle.availableGeometry()
            width = min(width, available.width())
            height = min(height, available.height())
        window.resize(max(1, width), max(1, height))
        return True
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not fit the window to the GUI scale", exc_info=True)
        return False


def _main_window(widget=None):
    """The spaCR main window: ``widget``'s own, else the first one open."""
    from PySide6.QtWidgets import QApplication

    candidates = []
    if widget is not None:
        try:
            candidates.append(widget.window())
            parent = widget.parentWidget()
            if parent is not None:
                candidates.append(parent.window())
        except (AttributeError, RuntimeError):
            pass
    app = QApplication.instance()
    if app is not None:
        candidates.extend(app.topLevelWidgets())
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "_stack"):
            return candidate
    return None


def restart_to_apply(parent=None, *, launcher=None, exiter=None) -> bool:
    """Restart spaCR so a new GUI scale takes effect.

    The current module and its settings come back, through the same record
    a Force restart writes (:func:`spacr.qt.screens.app_screen.AppScreen.force_restart`).
    A run in progress does not, so when one is going the user is asked
    first and Cancel is the default.

    :param parent: a widget in the window to restart.
    :param launcher: optional process-launch hook, for tests.
    :param exiter: optional exit hook, for tests.
    :returns: ``True`` when the new process was started.
    """
    from PySide6.QtCore import QSettings

    from . import preferences
    from .i18n import tr

    try:
        QSettings(preferences._ORG, preferences._APP).sync()
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not flush the preferences", exc_info=True)

    window = _main_window(parent)
    screen = None
    try:
        screen = window._stack.currentWidget() if window is not None else None
    except (AttributeError, RuntimeError):
        screen = None

    running = []
    try:
        running = list(screen.running_modules()) if hasattr(
            screen, "running_modules") else []
    except Exception:                                        # noqa: BLE001
        running = []
    if running:
        from PySide6.QtWidgets import QMessageBox

        answer = QMessageBox.question(
            parent or window, tr("Restart spaCR?"),
            tr("A run is still going. Restarting stops it; the module and "
               "its settings come back, the run does not."),
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
        if answer != QMessageBox.Yes:
            return False

    try:
        from .crash_recovery import note_a_clean_shutdown

        note_a_clean_shutdown()
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not mark the shutdown clean", exc_info=True)

    if screen is not None and hasattr(screen, "force_restart"):
        return bool(screen.force_restart(launcher=launcher, exiter=exiter))
    from .shutdown import restart_spacr

    return restart_spacr("", None, launcher=launcher, exiter=exiter)


def reset_every_scale(parent=None, *, ask: bool = True) -> bool:
    """Put GUI scale, Zoom and every preview scale back to 100 %.

    The escape hatch for a scale too small to read. Zoom and the preview
    scales apply at once; the GUI scale needs a restart, so when the one on
    screen is not 100 % the user is asked, with **Restart now** as the
    default button -- Enter answers it without reading the dialog.

    :param parent: a widget in the window to reset.
    :param ask: ``False`` skips the restart question (tests).
    :returns: ``True`` if a restart was started.
    """
    from . import preferences

    preferences.set_gui_scale(1.0)
    preferences.set_font_scale(1.0)
    try:
        from .widgets.preview_scale import reset_all_preview_scales

        reset_all_preview_scales()
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not reset the preview scales", exc_info=True)
    try:
        preferences.apply_preferences_to_app()
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not re-apply Zoom", exc_info=True)

    if not ask or abs(running_gui_scale() - 1.0) < 1e-6:
        return False
    from PySide6.QtWidgets import QMessageBox

    from .i18n import tr

    box = QMessageBox(parent)
    box.setObjectName("GuiScaleResetRestart")
    box.setIcon(QMessageBox.Question)
    box.setWindowTitle(tr("GUI scale reset"))
    box.setText(tr("GUI scale is back to 100 %. It takes effect when spaCR "
                   "restarts."))
    restart = box.addButton(tr("Restart now"), QMessageBox.AcceptRole)
    box.addButton(tr("Later"), QMessageBox.RejectRole)
    box.setDefaultButton(restart)
    box.exec()
    if box.clickedButton() is restart:
        return restart_to_apply(parent)
    return False


def mend_matplotlib_icons() -> bool:
    """Keep matplotlib's toolbar icons full size below a device ratio of 1.

    Matplotlib's toolbar icon engine multiplies the size Qt asks for by the
    device pixel ratio, and Qt has already done so; above 1 the two cancel in
    the drawing, below 1 they compound. Measured offscreen at GUI scale
    50 %: the icons drew 3 px tall instead of 11, a row of dots where the
    pan and zoom buttons should be. Holding the engine's ratio at 1 or more
    draws them 11 px tall, as at 100 % halved, and changes nothing at a
    ratio of 1 or 2 (measured: identical rows at both).

    Only installed when this process runs below 100 %, so a default launch
    never touches matplotlib. Idempotent.

    :returns: ``True`` if the mend is in place.
    """
    if running_gui_scale() >= 1.0:
        return False
    try:
        import matplotlib.backends.backend_qt as backend
    except Exception:                                        # noqa: BLE001
        return False
    engine = getattr(backend, "_IconEngine", None)
    original = getattr(engine, "_devicePixelRatio", None)
    if engine is None or original is None:
        return False
    if getattr(engine, "_spacr_mended", False):
        return True

    def _at_least_one(self):
        """The toolbar's device ratio, never below 1."""
        return max(1.0, float(original(self)))

    engine._devicePixelRatio = _at_least_one
    engine._spacr_mended = True
    return True
