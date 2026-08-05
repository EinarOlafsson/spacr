"""Input fields dissolve to the right — container and outline, never the text.

The request this implements, verbatim::

    the fields should not be subject to the occupacy setting. the fields
    could gradually become fully transparent (not the text in the field
    but the container (0% to the left, 100% to the right)) with the
    transparency growing faster towards the right. outlines should also
    be subject to the same effect. this should be on by default but there
    should be a preference to turn it off.

Four separate things, and each one decides part of the design:

1. **Exempt from page opacity.** A field's left edge is painted at its
   colour's own alpha — solid on the flat themes — whatever the page
   slider says. The slider still moves the section card *behind* the
   field; it no longer moves the field.
2. **A ramp, accelerating right.** :func:`spacr.qt.theme.field_fade_alpha`
   owns the curve: a cubic ease-in on transparency, 87.5 % opaque at the
   midpoint and gone at the right edge.
3. **Container and outline, not the text.** This is the constraint that
   picks the rendering approach, below.
4. **On by default, with a preference.**
   :func:`spacr.qt.preferences.get_field_fade_enabled`.

Why a paint hook and not QSS
----------------------------
QSS can express a horizontal ``qlineargradient`` fill, so requirement 2
alone would be a one-line stylesheet change. It cannot express
requirement 3. A QSS gradient paints the widget's *background*, and the
widget then draws its text on top of it — but a QSS ``border`` takes a
single colour, so the outline cannot ramp, and the two would visibly
disagree at the right edge. The obvious alternatives fail the same test
from the other side: a ``QGraphicsOpacityEffect`` with a gradient
``opacityMask`` ramps the whole widget *including its text*, and an
overlay child can only add ink, never subtract the fill underneath it.

So the chrome is painted here, and the text is left to the widget:

* A registered QSS block (see :func:`field_fade_qss`) makes every field's
  own background and border ``transparent`` — the border keeps its 1 px
  so no content shifts — leaving the widget painting **only its text,
  selection and cursor**, at full alpha.
* An application-wide event filter catches each field's ``QEvent.Paint``
  *before* the widget handles it, draws the two ramped rounded rects, and
  returns ``False`` so the widget then paints its text over them.

Painting from an event filter is legal precisely here: Qt sets
``WA_WState_InPaintEvent`` on the widget before it sends the paint event,
and event filters run inside that send, so a ``QPainter`` opened on the
widget is clipped to the update region like any other paint-event
painter.

The filter is installed on the ``QApplication``, not on individual
widgets, for one reason worth stating: settings panels build their inputs
deep inside :mod:`spacr.qt.screens.settings_model` and rebuild them
whenever a module is re-selected. A sweep would have to be re-run from
every one of those call sites and would silently miss the next one. One
application filter covers every field that will ever exist, now and in
whatever screen is written next.

The price is an application-wide filter, so it was measured rather than
assumed: 1.1 µs per event on top of Qt's own ~2.1 µs of delivery, for
events that are not paints — one enum comparison and a return. A desktop
session generating a few thousand events a second spends single-digit
milliseconds a second on it.

Known and deliberate: three inputs styled by an ID selector
(``QWidget#UmapHyperparamControls QLineEdit`` and friends) outrank the
blanket rules below on CSS specificity, so they keep their opaque fill
and the ramp painted behind them never shows. They are a raised card's
controls rather than settings fields, and leaving them alone is the
conservative answer.
"""
from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QEvent, QObject, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import (QAbstractItemView, QAbstractSpinBox,
                               QApplication, QComboBox, QLineEdit)

from ..theme import (FIELD_FADE_STOPS, field_chrome, field_fade_alpha,
                     register_widget_qss)

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


# ---------------------------------------------------------------------------
# The preference, read once and cached
# ---------------------------------------------------------------------------

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
            # An unreadable settings store falls back to the shipped
            # look, not to "off" — the same rule every other preference
            # in this app follows.
            _enabled = True
    return _enabled


def invalidate_field_fade() -> None:
    """Forget the cached preference so the next read hits ``QSettings``."""
    global _enabled
    _enabled = None


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
    # Editors are parented to the view's VIEWPORT, not to the view.
    grandparent = parent.parentWidget()
    if (isinstance(grandparent, QAbstractItemView)
            and grandparent.viewport() is parent):
        return False
    return True


# ---------------------------------------------------------------------------
# The paint
# ---------------------------------------------------------------------------

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
        try:
            from ..preferences import resolve_effective_theme
            theme = resolve_effective_theme()
        except Exception:
            theme = "dark"
    chrome = field_chrome(theme)
    radius = float(chrome["radius"])

    # Half-pixel inset so the 1px outline lands ON the widget's edge
    # pixels rather than straddling them, which is what keeps the left
    # end reading as a hard edge rather than a 50 % smear.
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
    fill_colour, fill_alpha = chrome[fill_key]
    line_colour, line_alpha = chrome[border_key]

    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(_ramped(fill_colour, fill_alpha,
                                    rect.left(), rect.right())))
    painter.drawRoundedRect(rect, radius, radius)
    painter.setBrush(Qt.NoBrush)
    painter.setPen(QPen(QBrush(_ramped(line_colour, line_alpha,
                                       rect.left(), rect.right())),
                        FIELD_BORDER_PX))
    painter.drawRoundedRect(rect, radius, radius)
    painter.restore()


class _FieldFadeFilter(QObject):
    """Paints the ramp under every field, just before the field paints."""

    def eventFilter(self, obj, event):  # noqa: N802 - Qt contract
        # First line of a filter that sees every event in the process:
        # one enum compare, then out.
        if event.type() != QEvent.Type.Paint:
            return False
        if not field_fade_enabled() or not fades(obj):
            return False
        painter = None
        try:
            painter = QPainter(obj)
            if painter.isActive():
                paint_field_fade(obj, painter)
        except Exception:
            # A cosmetic effect must never be the reason a screen fails to
            # draw. Logged rather than swallowed, so a broken palette is
            # discoverable instead of merely invisible.
            LOG.exception("Field fade could not paint %s",
                          type(obj).__name__)
        finally:
            # Explicit, not left to refcounting: a painter still active on
            # this widget would break the widget's own paint two lines
            # later, which is a blank field rather than an unstyled one.
            if painter is not None and painter.isActive():
                painter.end()
        # False, always: the widget still has to draw its text on top.
        return False


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
        # Re-install on the (possibly new) app. Qt ignores a duplicate
        # install of the same filter on the same object.
        app.installEventFilter(_filter)
        return False
    _filter = _FieldFadeFilter()
    app.installEventFilter(_filter)
    return True


def uninstall_field_fade(app=None) -> bool:
    """Remove the paint hook. ``True`` if there was one."""
    global _filter
    if _filter is None:
        return False
    app = app or QApplication.instance()
    if app is not None:
        app.removeEventFilter(_filter)
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


# ---------------------------------------------------------------------------
# The QSS that gets out of the painter's way
# ---------------------------------------------------------------------------

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
