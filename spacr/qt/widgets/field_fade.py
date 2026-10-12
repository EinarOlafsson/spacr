"""Fade form-field chrome to the right while keeping text fully opaque.

The fill and outline follow a cubic transparency ramp defined by
:func:`spacr.qt.theme.field_fade_alpha`: the left edge uses the theme token's
own alpha, the midpoint remains 87.5% opaque, and the right edge is fully
transparent. Field chrome is independent of the page-opacity preference and
the effect is enabled by default through the field-fade preference.

A registered stylesheet makes each supported editor's background and border
transparent while reserving its one-pixel border geometry. An application-
wide paint-event filter then draws the ramped fill and outline before Qt draws
the editor's text, selection, and cursor. Installing one filter on
``QApplication`` covers fields created or rebuilt after startup without
per-screen registration. Non-paint events pass through without repaint work.

Embedded line editors, item-view cell editors, multiline text widgets, and
widgets carrying :data:`OPT_OUT_PROPERTY` are excluded. Higher-specificity
ID-based styles can intentionally keep an opaque fill and hide the ramp.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

from PySide6.QtCore import QEvent, QObject, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QLinearGradient, QPainter, QPen,
                           QPixmap)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QLineEdit,
)

from ..gil_priority import (_stop_watching_application_events,
                            _watch_application_events)
from ..theme import (
    FIELD_FADE_STOPS,
    field_chrome,
    field_fade_alpha,
    _forget_painting_theme,
    _painting_theme,
    register_widget_qss,
)

LOG = logging.getLogger(__name__)

#: The widget types that count as "a field". Single-line value editors,
#: which is what a settings form is made of. ``QAbstractSpinBox`` covers
#: ``QSpinBox``, ``QDoubleSpinBox`` and the date/time editors.
#:
#: ``QPlainTextEdit``/``QTextEdit`` are pointedly NOT here: the console,
#: the AI chat transcript and the log panes are those types, and a
#: multi-line body of text that dissolves mid-paragraph is a different
#: (and unasked-for) thing from a value box that trails off past its end.
FIELD_TYPES = (QLineEdit, QComboBox, QAbstractSpinBox)

#: Dynamic property a widget can set to ``True`` to keep the plain look
#: even while the preference is on. Nothing in spaCR sets it yet; it
#: exists so a widget that paints its own background has a way out that
#: is not "edit this module".
OPT_OUT_PROPERTY = "spacrNoFieldFade"

#: Width of the painted outline, in logical pixels. Matches the ``1px``
#: the stylesheet reserves, so turning the fade on or off never reflows
#: a form.
FIELD_BORDER_PX = 1.0

_filter: Optional["_FieldFadeFilter"] = None
_enabled: Optional[bool] = None



def field_fade_enabled() -> bool:
    """Whether fields fade. Cached — this is read on every paint event.

    Building a ``QSettings`` per paint would put a file-format lookup in
    the middle of the render loop. The cache is dropped by
    :func:`invalidate_field_fade`, which
    :func:`spacr.qt.preferences.set_field_fade_enabled` and
    :func:`spacr.qt.preferences.apply_preferences_to_app` both call, so
    the two can never disagree about what is on screen.
    """
    global _enabled
    if _enabled is None:
        try:
            from ..preferences import get_field_fade_enabled
            _enabled = bool(get_field_fade_enabled())
        except Exception:
            _enabled = True
    return _enabled


def invalidate_field_fade() -> None:
    """Forget the cached preference so the next read hits ``QSettings``."""
    global _enabled
    _enabled = None
    _forget_painting_theme()
    _CHROME.clear()


def fades(widget) -> bool:
    """Whether ``widget`` is a field this effect should paint.

    Two exclusions, both of them about what the widget *is* rather than
    what class it belongs to:

    * The ``QLineEdit`` a spin box or an editable combo box embeds. That
      inner editor is a field by type but not by appearance: its
      container already ramps, and a second ramp inside the first would
      put a seam down the middle of one control.
    * An item view's in-place cell editor. It is a temporary widget laid
      over a row of data, not a form field with space to its right, so a
      transparent trailing half would show the cell it is covering and
      read as a rendering fault rather than as a design.

    :param widget: the widget to test; only the field types in ``FIELD_TYPES``
        that have not set the opt-out property can qualify.
    """
    if not isinstance(widget, FIELD_TYPES):
        return False
    if widget.property(OPT_OUT_PROPERTY):
        return False
    parent = widget.parentWidget()
    if parent is None:
        return True
    if isinstance(widget, QLineEdit) and isinstance(
            parent, (QAbstractSpinBox, QComboBox)):
        return False
    grandparent = parent.parentWidget()
    if (isinstance(grandparent, QAbstractItemView)
            and grandparent.viewport() is parent):
        return False
    return True



#: Built gradients, keyed by ``(colour, alpha, left, right)``. A form of
#: thirty settings repaints two gradients of seventeen stops per field,
#: and the answer only ever depends on those four numbers. Capped and
#: dropped wholesale rather than evicted one at a time: the working set
#: is two colours per state per field width, so it is small or it is
#: pathological, and there is nothing in between worth an LRU.
_GRADIENTS: dict = {}
_GRADIENT_CAP = 64


def _ramped(colour: str, alpha: float, left: float, right: float
            ) -> QLinearGradient:
    """A left-to-right gradient of ``colour`` following the fade curve.

    ``alpha`` is the colour's own opacity and the curve multiplies it, so
    a theme with a translucent rim keeps its material and still reaches
    zero on the right.
    """
    key = (colour, round(alpha, 4), round(left, 2), round(right, 2))
    cached = _GRADIENTS.get(key)
    if cached is not None:
        return cached
    gradient = QLinearGradient(left, 0.0, right, 0.0)
    last = FIELD_FADE_STOPS - 1
    for i in range(FIELD_FADE_STOPS):
        t = i / last
        stop = QColor(colour)
        stop.setAlphaF(max(0.0, min(1.0, alpha * field_fade_alpha(t))))
        gradient.setColorAt(t, stop)
    if len(_GRADIENTS) >= _GRADIENT_CAP:
        _GRADIENTS.clear()
    _GRADIENTS[key] = gradient
    return gradient


#: Rasterised chrome, keyed by size, pixel ratio, radius and the two
#: state colours. A field repaints whenever the backdrop under it moves,
#: sixty times a second, and two antialiased rounded rectangles with
#: seventeen-stop gradients cost more than the field's own text; the
#: pixels only ever depend on this key. Dropped wholesale at the cap.
_CHROME: dict = {}
_CHROME_CAP = 256


def _chrome_pixmap(width: int, height: int, ratio: float, radius: float,
                   fill: tuple, line: tuple) -> QPixmap:
    """The ramped container and outline of a ``width`` x ``height`` field.

    Painted exactly as a direct paint would draw it, into a transparent
    pixmap at the painter's pixel ratio, so drawing the pixmap over the
    field leaves the same pixels.
    """
    key = (width, height, round(ratio, 4), radius, fill, line)
    cached = _CHROME.get(key)
    if cached is not None:
        return cached
    pixmap = QPixmap(max(1, math.ceil(width * ratio)),
                     max(1, math.ceil(height * ratio)))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.transparent)
    inset = FIELD_BORDER_PX / 2.0
    rect = QRectF(0.0, 0.0, width, height).adjusted(inset, inset,
                                                     -inset, -inset)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(_ramped(fill[0], fill[1],
                                        rect.left(), rect.right())))
        painter.drawRoundedRect(rect, radius, radius)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QBrush(_ramped(line[0], line[1],
                                           rect.left(), rect.right())),
                            FIELD_BORDER_PX))
        painter.drawRoundedRect(rect, radius, radius)
    finally:
        painter.end()
    if len(_CHROME) >= _CHROME_CAP:
        _CHROME.clear()
    _CHROME[key] = pixmap
    return pixmap


def paint_field_fade(widget, painter: QPainter, theme: Optional[str] = None
                     ) -> None:
    """Draw ``widget``'s ramped container and outline with ``painter``.

    Separated from the event filter so a test can drive it against a
    plain image, and so a widget that wants the look inside its own
    ``paintEvent`` can call it directly.

    :param widget: the field. Only its ``rect()`` and its focus/enabled
        state are read.
    :param painter: an active painter whose coordinates are the widget's.
    :param theme: theme name; ``None`` resolves the effective one.
    """
    if theme is None:
        theme = _painting_theme()
    chrome = field_chrome(theme)
    radius = float(chrome["radius"])

    inset = FIELD_BORDER_PX / 2.0
    rect = QRectF(widget.rect()).adjusted(inset, inset, -inset, -inset)
    if rect.width() <= 0.0 or rect.height() <= 0.0:
        return

    enabled = widget.isEnabled()
    if not enabled:
        fill_key, border_key = "fill_disabled", "border_disabled"
    elif widget.hasFocus():
        fill_key, border_key = "fill", "border_focus"
    else:
        fill_key, border_key = "fill", "border"
    size = widget.size()
    painter.drawPixmap(0, 0, _chrome_pixmap(
        size.width(), size.height(), painter.device().devicePixelRatioF(),
        radius, tuple(chrome[fill_key]), tuple(chrome[border_key])))


class _FieldFadeFilter(QObject):
    """Paints the ramp under every field, just before the field paints.

    The application tells it about each widget once, when the widget is
    polished, and it filters the fields among them directly. It is not
    asked about every paint event in the application: over an animated
    backdrop every widget on screen repaints on every frame, and asking a
    Python filter about each of them cost more than painting the ramps.
    """

    def eventFilter(self, obj, event):  # noqa: N802 - Qt contract
        """Watch a field as it is polished; paint its ramp before it paints.

        :param obj: the field, or on a polish any widget.
        :param event: the event.
        :returns: ``False`` -- observed, never consumed.
        """
        kind = event.type()
        if kind == QEvent.Type.Polish:
            if isinstance(obj, FIELD_TYPES):
                obj.installEventFilter(self)
            return False
        if kind != QEvent.Type.Paint:
            return False
        if not field_fade_enabled() or not fades(obj):
            return False
        paint_owner = obj.window()
        painter = None
        try:
            painter = QPainter(obj)
            if painter.isActive():
                paint_field_fade(obj, painter)
        except Exception:
            LOG.exception("Field fade could not paint %s",
                          type(obj).__name__)
        finally:
            if painter is not None and painter.isActive():
                painter.end()
            del paint_owner
        return False


def _watch_fields(app) -> None:
    """Hear every widget's polish, and filter the fields that already exist."""
    _watch_application_events(app, _filter, (QEvent.Type.Polish,))
    for widget in app.allWidgets():
        if isinstance(widget, FIELD_TYPES):
            widget.installEventFilter(_filter)


def install_field_fade(app=None) -> bool:
    """Install the application-wide paint hook. Idempotent.

    :param app: the QApplication; defaults to the running instance.
    :returns: ``True`` if a filter was installed by this call.
    """
    global _filter
    ensure_field_fade_qss()
    app = app or QApplication.instance()
    if app is None:
        return False
    if _filter is not None:
        _watch_fields(app)
        return False
    _filter = _FieldFadeFilter()
    _watch_fields(app)
    return True


def uninstall_field_fade(app=None) -> bool:
    """Remove the paint hook. ``True`` if there was one."""
    global _filter
    if _filter is None:
        return False
    app = app or QApplication.instance()
    if app is not None:
        _stop_watching_application_events(app, _filter)
        for widget in app.allWidgets():
            widget.removeEventFilter(_filter)
    _filter = None
    return True


def repaint_fields(app=None) -> int:
    """Schedule a repaint of every live field. Returns how many.

    The stylesheet swap that follows a preference change already forces a
    repolish, but a field whose look changed without its *style* changing
    — turning the effect off while the QSS block was already empty — has
    nothing else to trigger it.
    """
    app = app or QApplication.instance()
    if app is None:
        return 0
    count = 0
    for widget in app.allWidgets():
        if fades(widget):
            widget.update()
            count += 1
    return count



#: Every selector the effect has to neutralise. Listed once, because a
#: state whose rule is missed here paints an opaque box over the ramp and
#: the fade silently stops working in that state only.
_FIELD_SELECTORS = (
    "QLineEdit", "QComboBox", "QAbstractSpinBox",
    "QSpinBox", "QDoubleSpinBox",
)

_FIELD_STATES = ("", ":focus", ":disabled", ":hover", ":read-only")


def field_fade_qss(palette: dict, opacity: Optional[float] = None) -> str:
    """The registered QSS block. Empty when the preference is off.

    Empty is load-bearing: with nothing emitted, the built-in input rules
    are untouched and a field looks exactly as it did before this module
    existed, which is what "turn it off" has to mean.

    Signature is :func:`spacr.qt.theme.register_widget_qss`'s contract;
    neither argument is used, and that is the point — a field is exempt
    from ``opacity``, and its colours come from
    :func:`spacr.qt.theme.field_chrome` at paint time so they survive a
    theme switch without the stylesheet having baked them in.

    :param palette: the theme palette passed by
        :func:`spacr.qt.theme.register_widget_qss`; unused.
    """
    if not field_fade_enabled():
        return ""
    selectors = ",\n".join(
        f"{name}{state}"
        for state in _FIELD_STATES
        for name in _FIELD_SELECTORS
    )
    return f"""
/* The container and outline are painted by
   spacr.qt.widgets.field_fade so they can ramp to transparent; the
   widget keeps drawing its text, at full alpha, on top. The border is
   still declared 1px so nothing reflows when the effect is toggled. */
{selectors} {{
    background: transparent;
    background-color: transparent;
    border: {FIELD_BORDER_PX:.0f}px solid transparent;
}}"""


def ensure_field_fade_qss() -> None:
    """(Re)register the QSS block. Idempotent, and called by
    :func:`install_field_fade` as well as at import.

    Both, because importing a module happens once per process while the
    registry is a mutable global: a test that snapshots and restores it
    would otherwise switch the effect off for the rest of the session
    with no way to get it back.
    """
    register_widget_qss("FieldFade", field_fade_qss, replace=True)


ensure_field_fade_qss()
