"""Apply one GUI scale factor to every size spaCR sets, while spaCR runs.

"GUI scale" in Preferences (10 % to 200 %, default 100 %) scales widget sizes,
fixed sizes, margins, spacing, style-sheet sizes (fonts and paddings
included), icons, splitter sizes and figure dpi -- and it does so while spaCR
runs, with no restart.

The scale is applied without editing the call sites. Counted on nightly
2026-09-22 across the 328 modules under ``spacr/qt``: about 1,420 hard-coded
geometry calls in 181 files (470 ``setContentsMargins``, 489 ``setSpacing``,
179 ``setMinimumWidth/Height``, 67 ``setMaximumWidth/Height``, 74
``setFixed*``, 30 ``setMinimumSize``, 56 ``QSize``, 64 ``resize``, 6
``setIconSize``) and 486 ``px`` literals in style-sheet text in 64 files.
Routing each by hand is a change to every screen, and every size added later
would have to remember it. Instead :func:`install_scaling_layer` -- called
once from :func:`spacr.qt.app.launch` before the first widget exists --
replaces the setters themselves on the Qt classes (``QWidget``'s size setters
and ``setStyleSheet``, the layouts' margin and spacing setters, every
``setIconSize``, ``QSplitter.setSizes``). Each replacement

* remembers, on the object, the value the code asked for -- its size at
  100 %;
* passes the value times the current scale on to Qt;
* and answers the matching getter with the remembered value, so code that
  reads a size back and sets it again (the theme compares its own sheet by
  digest, a splitter saves ``sizes()``) keeps working in 100 % units and
  never compounds.

A change of scale (:func:`set_gui_scale_live`) walks every live widget and
layout, puts every remembered value back at the new scale, and asks the
windows to lay out again. A widget built later is scaled as it is built.
A splitter's panes keep the room they have on screen -- what shrinks is
what is inside them -- and ``sizes()`` answers in 100 % units, so saved
splitter sizes survive a change of scale.

Font scale is applied separately: it writes the font sizes into the theme's
style sheet. That sheet goes through the ``setStyleSheet`` replacement like
any other, so a font size ends up as base x font scale x GUI scale. 50 % GUI
at 200 % font is half-size widgets with text the usual size on screen.

A user's own ``QT_SCALE_FACTOR`` is not changed and multiplies the result,
as it does in any Qt application.

Every change of GUI scale or font scale applies at once and then asks "Keep
these settings?" (:class:`KeepOrRevertDialog`), counting down 15 s; the
countdown, Esc and closing the dialog all put the old values back. The dialog
is exempt from the scale and sets its own text size, so it is readable at 10 %
GUI and 10 % font alike. ``Ctrl+Alt+0`` (:func:`reset_every_scale`) also
resets every scale to 100 %.

Sizes that do not pass through a setter are not rescaled live: text and shapes
a widget paints itself at fixed pixel coordinates, pixmaps a widget scales to
a number it computed, ``setFont`` with an explicit size,
``move``/``setGeometry`` positions, header section sizes, a Python
``sizeHint`` override that returns a constant, the Fusion style's own metrics
that the theme sheet does not restate, and pyqtgraph's axis text. Sizes
computed from font metrics are already scaled once by the font and are scaled
again by the setter -- the one way the layer can over-shrink. The main
window's own size is left alone, so the gained room goes to the content.
"""
from __future__ import annotations

import logging
import math
import re
import sys
import threading
import weakref
from typing import Callable, List, Optional

LOG = logging.getLogger("spacr.qt.gui_scale")

#: Qt's "no maximum" size, which is never scaled.
QWIDGETSIZE_MAX = 16777215

#: A window carrying this property is drawn at 100 % whatever the scale --
#: the Keep or Revert question is the one that has to stay readable.
EXEMPT = "spacrGuiScaleExempt"

#: Style-sheet properties whose pixel values follow the scale. Borders are
#: left alone: a hairline that rounds to nothing disappears.
_QSS_SIZE = re.compile(
    r"(?<![\w-])(font-size|font|padding(?:-(?:top|right|bottom|left))?"
    r"|margin(?:-(?:top|right|bottom|left))?|(?:min|max)-(?:width|height)"
    r"|width|height|spacing|border(?:-(?:top|bottom)-(?:left|right))?-radius)"
    r"(\s*:\s*)([^;{}]*)", re.I)
_PX = re.compile(r"(-?\d+(?:\.\d+)?)px")

_SCALE = 1.0
_INSTALLED = False
_ORIGINAL = {}
_SHEET_CACHE: dict = {}
_LISTENERS: List = []


def current_scale() -> float:
    """The GUI scale in force now (1.0 = 100 %)."""
    return _SCALE


def installed() -> bool:
    """Whether :func:`install_scaling_layer` has run in this process."""
    return _INSTALLED


def scale_int(value, factor: Optional[float] = None) -> int:
    """``value`` at ``factor``, keeping 0, negatives and Qt's maximum as they are.

    :param value: a size in 100 % pixels.
    :param factor: the scale; the current one by default.
    """
    value = int(value)
    factor = _SCALE if factor is None else float(factor)
    if value <= 0 or value >= QWIDGETSIZE_MAX or factor == 1.0:
        return value
    return max(1, int(round(value * factor)))


def _scaled_px(match, factor: float, prop: str = "") -> str:
    """One ``Npx`` at ``factor``, never rounding a non-zero size to nothing.

    A radius rounds down, not to nearest. Qt drops every corner of a box
    whose two radii add up to more than its side, so a circle or a pill
    whose radius is half its size turns square the moment the radius rounds
    up past half of the rounded side: 8px of 16px at 70 % is 6 of 11.

    :param match: the ``Npx`` match.
    :param factor: the scale.
    :param prop: the property the value belongs to.
    """
    value = float(match.group(1))
    if value == 0:
        return match.group(0)
    size = abs(value) * factor
    if prop.lower().endswith("radius"):
        scaled = int(math.floor(size + 1e-9)) or 1
    else:
        scaled = int(round(size)) or 1
    return f"{-scaled if value < 0 else scaled}px"


def scale_qss_text(text: str, factor: Optional[float] = None) -> str:
    """Scale the sizes in a style sheet, leaving every other byte as it was.

    Only the values of size properties (font sizes, paddings, margins,
    min/max sizes, widths, heights, spacing, radii) change; selectors,
    colours, borders and ``url(...)`` data are untouched.

    :param text: the style sheet.
    :param factor: the scale; the current one by default.
    """
    factor = _SCALE if factor is None else float(factor)
    text = "" if text is None else str(text)
    if factor == 1.0 or "px" not in text:
        return text
    key = (text, factor)
    hit = _SHEET_CACHE.get(key)
    if hit is not None:
        return hit
    scaled = _QSS_SIZE.sub(
        lambda m: m.group(1) + m.group(2)
        + _PX.sub(lambda p: _scaled_px(p, factor, m.group(1)), m.group(3)),
        text)
    if len(_SHEET_CACHE) > 256:
        _SHEET_CACHE.clear()
    _SHEET_CACHE[key] = scaled
    return scaled


def _exempt(widget) -> bool:
    """Is ``widget`` inside a window drawn at 100 % regardless?"""
    try:
        window = widget.window()
        return bool(window is not None and window.property(EXEMPT))
    except (AttributeError, RuntimeError, TypeError):
        return False


def _layout_exempt(layout) -> bool:
    """Is ``layout``'s widget inside an exempt window?"""
    try:
        owner = layout.parentWidget()
    except (AttributeError, RuntimeError):
        return False
    return owner is not None and _exempt(owner)


def _factor_for(widget) -> float:
    """The scale ``widget`` is drawn at: the current one, or 1 if exempt."""
    if _SCALE == 1.0:
        return 1.0
    return 1.0 if _exempt(widget) else _SCALE


def _store(obj, name: str) -> dict:
    """The per-object record of what the code asked for."""
    record = getattr(obj, name, None)
    if record is None:
        record = {}
        try:
            setattr(obj, name, record)
        except (AttributeError, TypeError):
            return {}
    return record


def _size_args(args):
    """``(w, h)`` from ``(QSize)`` or ``(w, h)``."""
    if len(args) == 1:
        size = args[0]
        return int(size.width()), int(size.height())
    return int(args[0]), int(args[1])


def _margin_args(args):
    """``(l, t, r, b)`` from ``(QMargins)`` or four ints."""
    if len(args) == 1:
        m = args[0]
        return int(m.left()), int(m.top()), int(m.right()), int(m.bottom())
    return tuple(int(v) for v in args[:4])


_DIMS = {
    "minw": ("setMinimumWidth", "minimumWidth"),
    "minh": ("setMinimumHeight", "minimumHeight"),
    "maxw": ("setMaximumWidth", "maximumWidth"),
    "maxh": ("setMaximumHeight", "maximumHeight"),
}


def _is_our_own_value(base, value, factor: float) -> bool:
    """Is ``value`` this layer's own scaled ``base``, measured again?

    A widget that re-derives its size from the font it is drawn in -- the
    window's close mark re-measures its glyph whenever the sheet changes --
    would otherwise hand the shrunken size back as a NEW base, and going
    back to 100 % would leave it shrunken. Measured on a 100 -> 50 -> 100
    round trip before this guard: the window's close marks came back half
    size. A value within 6 % of what this layer itself applied is taken to
    be that same size, so the base stands.

    :param base: the remembered 100 % value.
    :param value: what the caller is setting now.
    :param factor: the scale in force.
    """
    if base is None or factor == 1.0:
        return False
    wanted = scale_int(base, factor)
    return abs(int(value) - wanted) <= max(1, int(round(wanted * 0.06)))


def _bounded_main_window_minimum(widget, key, value, screen=None):
    """Bound an opted-in floor by its desktop baseline and actual client room.

    :param widget: window; unmarked and embedded widgets pass through.
    :param key: ``minw`` or ``minh``; other dimensions pass through.
    :param value: already scaled floor in actual pixels.
    :param screen: known screen, or ``None`` to use the window's screen.
    :returns: floor bounded by its recorded100% value and native work area
        minus frame margins, or unchanged when screen geometry is unavailable.
        Smaller GUI scales still lower the floor; larger controls do not
        turn the desktop minimum into a screen-filling resize lock.
    """
    if (key not in ("minw", "minh")
            or not getattr(widget, "_gs_main_window_floor", False)):
        return value
    try:
        if not widget.isWindow():
            return value
        screen = screen or widget.screen()
        if screen is None:
            return value
        work = screen.availableGeometry()
        if work.isEmpty():
            return value
        width, height = work.width(), work.height()
        handle = widget.windowHandle()
        if handle is not None:
            margins = handle.frameMargins()
            width -= margins.left() + margins.right()
            height -= margins.top() + margins.bottom()
        baseline = getattr(widget, "_gs_geo", {}).get(key, value)
        return min(value, baseline, max(1, width if key == "minw" else height))
    except (AttributeError, RuntimeError):
        return value


def _fit_main_window_minimum(widget, screen=None):
    """Reapply logical floors after native decoration or work-area changes.

    :param widget: opted-in window with remembered logical minimum dimensions.
    :param screen: known screen, or ``None`` to use the window's screen.
    :returns: whether stored floors were applied through original Qt setters;
        ``False`` leaves callers' existing unmarked-window fallback intact.
    """
    from PySide6.QtWidgets import QWidget

    record = getattr(widget, "_gs_geo", None)
    if (not getattr(widget, "_gs_main_window_floor", False)
            or not record or not all(key in record for key in ("minw", "minh"))
            or not all((QWidget, _DIMS[key][0]) in _ORIGINAL for key in ("minw", "minh"))):
        return False
    applied = _store(widget, "_gs_geo_set")
    for key in ("minw", "minh"):
        wanted = _bounded_main_window_minimum(widget, key, scale_int(record[key], _factor_for(widget)), screen)
        applied[key] = wanted
        _ORIGINAL[(QWidget, _DIMS[key][0])](widget, wanted)
    return True


def _set_dims(widget, **dims) -> None:
    """Remember and apply minimum/maximum dimensions of ``widget``."""
    from PySide6.QtWidgets import QWidget

    record = _store(widget, "_gs_geo")
    applied = _store(widget, "_gs_geo_set")
    factor = _factor_for(widget)
    for key, value in dims.items():
        if not _is_our_own_value(record.get(key), value, factor):
            record[key] = int(value)
        wanted = _bounded_main_window_minimum(widget, key, scale_int(record[key], factor))
        applied[key] = wanted
        _ORIGINAL[(QWidget, _DIMS[key][0])](widget, wanted)


def _get_dim(widget, key: str) -> int:
    """A dimension in 100 % units when it is still the one the layer set."""
    from PySide6.QtWidgets import QWidget

    actual = _ORIGINAL[(QWidget, _DIMS[key][1])](widget)
    record = getattr(widget, "_gs_geo", None)
    applied = getattr(widget, "_gs_geo_set", None)
    if record and key in record and applied and applied.get(key) == actual:
        return record[key]
    return actual


def _actual_minimum_height(widget) -> int:
    """``widget``'s minimum height in the pixels it is drawn at now.

    The patched :meth:`QWidget.minimumHeight` answers in 100 % units, which
    is right for a value code wrote and wrong for one measured from a laid
    out widget, such as a ``minimumSizeHint`` or a ``heightForWidth``.

    :param widget: the widget to read.
    :returns: the minimum height Qt is enforcing.
    """
    from PySide6.QtWidgets import QWidget

    getter = _ORIGINAL.get((QWidget, "minimumHeight"))
    return int(getter(widget) if getter is not None else widget.minimumHeight())


def _actual_minimum_size(widget):
    """``widget``'s minimum size in the pixels it is drawn at now.

    :param widget: the widget to read.
    :returns: the ``QSize`` Qt is enforcing, not the 100 % record.
    """
    from PySide6.QtCore import QSize
    from PySide6.QtWidgets import QWidget

    getter = _ORIGINAL.get((QWidget, "minimumWidth"))
    width = getter(widget) if getter is not None else widget.minimumWidth()
    return QSize(int(width), _actual_minimum_height(widget))


def _set_actual_minimum_height(widget, height: int) -> None:
    """Hold ``widget`` at least ``height`` pixels tall at the current scale.

    For a height MEASURED from widgets already drawn at the GUI scale.
    Passing it to :meth:`QWidget.setMinimumHeight` would scale it a second
    time: at 75 % a pane that needs 196 px got a floor of 147, and the
    Actions buttons were cut to half their height (the macOS distribution
    smoke at 1440 x 870, 2026-10-20). The 100 % record is kept, rounded up,
    so a later change of scale starts from a floor that still fits.

    :param widget: the widget to hold.
    :param height: the minimum, in the pixels it is drawn at now.
    """
    _set_actual_dim(widget, "minh", height)


def _set_actual_dim(widget, key: str, value: int) -> None:
    """Set one minimum or maximum of ``widget`` in the pixels it is drawn at.

    The counterpart of :func:`_set_actual_minimum_height` for every
    dimension the layer scales. A collapsed pane's heading is measured from
    its font at the scale in force, and handing that height to
    :meth:`QWidget.setMaximumHeight` scaled it once more: at 75 % the Mask
    module's folded Actions and System headings were held at three
    quarters of their own height, their titles cut to a few pixels, with
    no way left to click them open.

    :param widget: the widget to hold.
    :param key: ``"minh"``, ``"minw"``, ``"maxh"`` or ``"maxw"``.
    :param value: the size, in the pixels it is drawn at now. Qt's "no
        maximum" is kept as it is.
    """
    from PySide6.QtWidgets import QWidget

    value = max(0, min(QWIDGETSIZE_MAX, int(value)))
    setter = _ORIGINAL.get((QWidget, _DIMS[key][0]))
    if setter is None:
        getattr(widget, _DIMS[key][0])(value)
        return
    factor = _factor_for(widget)
    if factor == 1.0 or value >= QWIDGETSIZE_MAX:
        base = value
    else:
        base = int(math.ceil(value / factor))
    _store(widget, "_gs_geo")[key] = base
    _store(widget, "_gs_geo_set")[key] = value
    setter(widget, value)


def _resize_actual(widget, size) -> None:
    """Resize ``widget`` to ``size``, given in the pixels it is drawn at.

    For a size measured from the laid-out screen, such as the room a scroll
    area has. :meth:`QWidget.resize` would scale it again: at 50 % a
    cramped module screen's content filled half its scroll area.

    :param widget: the widget to resize.
    :param size: a ``QSize`` in drawn pixels.
    """
    from PySide6.QtWidgets import QWidget

    resize = _ORIGINAL.get((QWidget, "resize"))
    if resize is None:
        widget.resize(size)
        return
    resize(widget, int(size.width()), int(size.height()))


def _actual_sizes(splitter) -> list:
    """``splitter``'s sizes in the pixels they are drawn at.

    :param splitter: a ``QSplitter``.
    :returns: one size per child.
    """
    from PySide6.QtWidgets import QSplitter

    getter = _ORIGINAL.get((QSplitter, "sizes"), QSplitter.sizes)
    return [int(v) for v in getter(splitter)]


def _set_actual_sizes(splitter, sizes) -> None:
    """Lay ``splitter`` out at ``sizes``, given in the pixels it is drawn at.

    The 100 % record that a later change of scale re-applies is kept in
    step, as :meth:`QSplitter.setSizes` keeps it.

    :param splitter: a ``QSplitter``.
    :param sizes: one size per child, in drawn pixels.
    """
    from PySide6.QtWidgets import QSplitter

    actual = [max(0, int(v)) for v in sizes]
    setter = _ORIGINAL.get((QSplitter, "setSizes"))
    if setter is None:
        splitter.setSizes(actual)
        return
    factor = _factor_for(splitter)
    try:
        splitter._gs_sizes = ([int(round(v / factor)) if v > 0 else v
                               for v in actual], list(actual))
    except (AttributeError, TypeError):
        pass
    setter(splitter, actual)


def _install_widget_setters() -> None:
    """Replace ``QWidget``'s size setters and getters."""
    from PySide6.QtCore import QSize
    from PySide6.QtWidgets import QMainWindow, QWidget

    for name in ("setMinimumWidth", "setMinimumHeight", "setMaximumWidth",
                 "setMaximumHeight", "minimumWidth", "minimumHeight",
                 "maximumWidth", "maximumHeight", "resize", "setStyleSheet",
                 "styleSheet"):
        _ORIGINAL[(QWidget, name)] = getattr(QWidget, name)

    def setMinimumWidth(self, w):                      # noqa: N802
        """Record ``w`` at 100 % and apply it at the GUI scale."""
        _set_dims(self, minw=w)

    def setMinimumHeight(self, h):                     # noqa: N802
        """Record ``h`` at 100 % and apply it at the GUI scale."""
        _set_dims(self, minh=h)

    def setMaximumWidth(self, w):                      # noqa: N802
        """Record ``w`` at 100 % and apply it at the GUI scale."""
        _set_dims(self, maxw=w)

    def setMaximumHeight(self, h):                     # noqa: N802
        """Record ``h`` at 100 % and apply it at the GUI scale."""
        _set_dims(self, maxh=h)

    def setMinimumSize(self, *args):                   # noqa: N802
        """Record the size at 100 % and apply it at the GUI scale."""
        w, h = _size_args(args)
        _set_dims(self, minw=w, minh=h)

    def setMaximumSize(self, *args):                   # noqa: N802
        """Record the size at 100 % and apply it at the GUI scale."""
        w, h = _size_args(args)
        _set_dims(self, maxw=w, maxh=h)

    def setFixedWidth(self, w):                        # noqa: N802
        """Record ``w`` at 100 % and apply it at the GUI scale."""
        _set_dims(self, minw=w, maxw=w)

    def setFixedHeight(self, h):                       # noqa: N802
        """Record ``h`` at 100 % and apply it at the GUI scale."""
        _set_dims(self, minh=h, maxh=h)

    def setFixedSize(self, *args):                     # noqa: N802
        """Record the size at 100 % and apply it at the GUI scale."""
        w, h = _size_args(args)
        _set_dims(self, minw=w, maxw=w, minh=h, maxh=h)

    def minimumWidth(self):                            # noqa: N802
        """The minimum width in 100 % units."""
        return _get_dim(self, "minw")

    def minimumHeight(self):                           # noqa: N802
        """The minimum height in 100 % units."""
        return _get_dim(self, "minh")

    def maximumWidth(self):                            # noqa: N802
        """The maximum width in 100 % units."""
        return _get_dim(self, "maxw")

    def maximumHeight(self):                           # noqa: N802
        """The maximum height in 100 % units."""
        return _get_dim(self, "maxh")

    def minimumSize(self):                             # noqa: N802
        """The minimum size in 100 % units."""
        return QSize(_get_dim(self, "minw"), _get_dim(self, "minh"))

    def maximumSize(self):                             # noqa: N802
        """The maximum size in 100 % units."""
        return QSize(_get_dim(self, "maxw"), _get_dim(self, "maxh"))

    original_resize = _ORIGINAL[(QWidget, "resize")]

    def resize(self, *args):
        """Resize at the GUI scale; the main window keeps its size."""
        w, h = _size_args(args)
        factor = 1.0 if isinstance(self, QMainWindow) else _factor_for(self)
        original_resize(self, scale_int(w, factor), scale_int(h, factor))

    original_set_sheet = _ORIGINAL[(QWidget, "setStyleSheet")]
    original_sheet = _ORIGINAL[(QWidget, "styleSheet")]

    def setStyleSheet(self, text):                     # noqa: N802
        """Record the sheet at 100 % and apply its sizes at the GUI scale."""
        text = "" if text is None else str(text)
        record = getattr(self, "_gs_sheet", None)
        if record is not None and text == record[1] != record[0]:
            text = record[0]
        scaled = scale_qss_text(text, _factor_for(self))
        try:
            self._gs_sheet = (text, scaled)
        except (AttributeError, TypeError):
            pass
        original_set_sheet(self, scaled)

    def styleSheet(self):                              # noqa: N802
        """The sheet as the code set it, at 100 %."""
        actual = original_sheet(self)
        record = getattr(self, "_gs_sheet", None)
        if record is not None and record[1] == actual:
            return record[0]
        return actual

    for fn in (setMinimumWidth, setMinimumHeight, setMaximumWidth,
               setMaximumHeight, setMinimumSize, setMaximumSize,
               setFixedWidth, setFixedHeight, setFixedSize, minimumWidth,
               minimumHeight, maximumWidth, maximumHeight, minimumSize,
               maximumSize, resize, setStyleSheet, styleSheet):
        setattr(QWidget, fn.__name__, fn)


def _qt_subclasses(base):
    """Every class in QtWidgets that derives from ``base``, ``base`` included.

    Read through ``dir`` rather than ``vars``: PySide6 fills a module's
    dictionary lazily, so ``vars`` on a fresh import lists three classes and
    the patch missed ``QAbstractButton`` -- which is every button's icon.
    """
    from PySide6 import QtWidgets

    found = []
    for name in dir(QtWidgets):
        value = getattr(QtWidgets, name, None)
        if isinstance(value, type) and issubclass(value, base):
            found.append(value)
    return found


def _patch_everywhere(base, name: str, make: Callable) -> None:
    """Patch ``name`` on ``base`` and on every Qt subclass binding its own."""
    wrappers = set()
    for cls in [base] + [c for c in _qt_subclasses(base) if c is not base]:
        current = getattr(cls, name, None)
        if current is None or current in wrappers:
            continue
        if cls is not base and current is getattr(base, name):
            continue
        _ORIGINAL[(cls, name)] = current
        wrapper = make(current)
        wrapper.__name__ = name
        setattr(cls, name, wrapper)
        wrappers.add(wrapper)


def _install_layout_setters() -> None:
    """Replace the layouts' margin and spacing setters and getters."""
    from PySide6.QtCore import QMargins
    from PySide6.QtWidgets import (QBoxLayout, QLayout, QSizePolicy,
                                   QSpacerItem)

    def margins_setter(original):
        """The wrapper that records a layout's margins and scales them."""
        def setContentsMargins(self, *args):          # noqa: N802
            """Record the margins at 100 % and apply them at the GUI scale."""
            base = _margin_args(args)
            factor = 1.0 if _layout_exempt(self) else _SCALE
            wanted = tuple(scale_int(v, factor) for v in base)
            try:
                self._gs_margins = (base, wanted)
            except (AttributeError, TypeError):
                pass
            original(self, *wanted)
        return setContentsMargins

    def margins_getter(original):
        """The wrapper that answers a layout's margins at 100 %."""
        def contentsMargins(self):                     # noqa: N802
            """The margins in 100 % units."""
            actual = original(self)
            record = getattr(self, "_gs_margins", None)
            if record is not None and _margin_args((actual,)) == record[1]:
                return QMargins(*record[0])
            return actual
        return contentsMargins

    _patch_everywhere(QLayout, "setContentsMargins", margins_setter)
    _patch_everywhere(QLayout, "contentsMargins", margins_getter)

    def spacing_setter(key):
        """The factory for one spacing setter, by its record key."""
        def make(original):
            """The wrapper that records a spacing and scales it."""
            def setter(self, value):
                """Record the spacing at 100 % and apply it at the GUI scale."""
                factor = 1.0 if _layout_exempt(self) else _SCALE
                wanted = scale_int(value, factor)
                record = _store(self, "_gs_spacing")
                record[key] = (int(value), wanted)
                original(self, wanted)
            return setter
        return make

    def spacing_getter(key):
        """The factory for one spacing getter, by its record key."""
        def make(original):
            """The wrapper that answers a spacing at 100 %."""
            def getter(self):
                """The spacing in 100 % units."""
                actual = original(self)
                record = getattr(self, "_gs_spacing", None)
                if record and key in record and record[key][1] == actual:
                    return record[key][0]
                return actual
            return getter
        return make

    from PySide6.QtWidgets import QFormLayout, QGridLayout

    for base, setter, getter, key in (
            (QLayout, "setSpacing", "spacing", "sp"),
            (QGridLayout, "setHorizontalSpacing", "horizontalSpacing", "hs"),
            (QGridLayout, "setVerticalSpacing", "verticalSpacing", "vs"),
            (QFormLayout, "setHorizontalSpacing", "horizontalSpacing", "hs"),
            (QFormLayout, "setVerticalSpacing", "verticalSpacing", "vs")):
        _patch_everywhere(base, setter, spacing_setter(key))
        _patch_everywhere(base, getter, spacing_getter(key))

    def spacer(self, index, size):
        """A fixed spacer item along the box's direction."""
        horizontal = self.direction() in (QBoxLayout.LeftToRight,
                                          QBoxLayout.RightToLeft)
        factor = 1.0 if _layout_exempt(self) else _SCALE
        wanted = scale_int(size, factor)
        if horizontal:
            item = QSpacerItem(wanted, 0, QSizePolicy.Fixed,
                               QSizePolicy.Minimum)
        else:
            item = QSpacerItem(0, wanted, QSizePolicy.Minimum,
                               QSizePolicy.Fixed)
        spacers = getattr(self, "_gs_spacers", None)
        if spacers is None:
            spacers = []
            try:
                self._gs_spacers = spacers
            except (AttributeError, TypeError):
                pass
        spacers.append((item, int(size), horizontal))
        if index is None:
            self.addItem(item)
        else:
            self.insertItem(index, item)

    def addSpacing(self, size):                        # noqa: N802
        """Add a spacer recorded at 100 % and drawn at the GUI scale."""
        spacer(self, None, size)

    def insertSpacing(self, index, size):              # noqa: N802
        """Insert a spacer recorded at 100 % and drawn at the GUI scale."""
        spacer(self, index, size)

    _ORIGINAL[(QBoxLayout, "addSpacing")] = QBoxLayout.addSpacing
    _ORIGINAL[(QBoxLayout, "insertSpacing")] = QBoxLayout.insertSpacing
    QBoxLayout.addSpacing = addSpacing
    QBoxLayout.insertSpacing = insertSpacing


def _install_icon_setters() -> None:
    """Replace every ``setIconSize`` and ``iconSize`` in QtWidgets."""
    from PySide6.QtCore import QSize
    from PySide6.QtWidgets import QWidget

    def setter(original):
        """The wrapper that records an icon size and scales it."""
        def setIconSize(self, size):                   # noqa: N802
            """Record the icon size at 100 % and apply it at the GUI scale."""
            base = _size_args((size,))
            factor = _factor_for(self)
            previous = getattr(self, "_gs_icon", None)
            if previous is not None and all(
                    _is_our_own_value(old, new, factor)
                    for old, new in zip(previous[0], base)):
                base = previous[0]
            wanted = tuple(scale_int(v, factor) for v in base)
            try:
                self._gs_icon = (base, wanted)
            except (AttributeError, TypeError):
                pass
            original(self, QSize(*wanted))
        return setIconSize

    def getter(original):
        """The wrapper that answers an icon size at 100 %."""
        def iconSize(self):                            # noqa: N802
            """The icon size in 100 % units."""
            actual = original(self)
            record = getattr(self, "_gs_icon", None)
            if record is not None and _size_args((actual,)) == record[1]:
                return QSize(*record[0])
            return actual
        return iconSize

    roots = [cls for cls in _qt_subclasses(QWidget)
             if getattr(cls, "setIconSize", None) is not None
             and not any(getattr(parent, "setIconSize", None) is not None
                         for parent in cls.__mro__[1:])]
    for root in roots:
        _patch_everywhere(root, "setIconSize", setter)
        _patch_everywhere(root, "iconSize", getter)


def _install_splitter_setters() -> None:
    """Keep splitter sizes in 100 % units, so saved sizes survive a change."""
    from PySide6.QtWidgets import QSplitter

    original_set = QSplitter.setSizes
    original_get = QSplitter.sizes
    _ORIGINAL[(QSplitter, "setSizes")] = original_set
    _ORIGINAL[(QSplitter, "sizes")] = original_get

    def setSizes(self, sizes):                         # noqa: N802
        """Apply 100 % sizes at the GUI scale, and remember them."""
        factor = _factor_for(self)
        wanted = [scale_int(v, factor) for v in sizes]
        try:
            self._gs_sizes = ([int(v) for v in sizes], list(wanted))
        except (AttributeError, TypeError):
            pass
        original_set(self, wanted)

    def sizes(self):
        """The sizes in 100 % units."""
        factor = _factor_for(self)
        actual = original_get(self)
        if factor == 1.0:
            return actual
        return [int(round(v / factor)) if v > 0 else v for v in actual]

    QSplitter.setSizes = setSizes
    QSplitter.sizes = sizes


def _install_application_sheet() -> None:
    """The application-wide sheet goes through the same size rewrite."""
    from PySide6.QtWidgets import QApplication

    original_set = QApplication.setStyleSheet
    original_get = QApplication.styleSheet
    _ORIGINAL[(QApplication, "setStyleSheet")] = original_set
    _ORIGINAL[(QApplication, "styleSheet")] = original_get
    state = {}

    def setStyleSheet(self, text):                     # noqa: N802
        """Record the sheet at 100 % and apply its sizes at the GUI scale."""
        text = "" if text is None else str(text)
        scaled = scale_qss_text(text)
        state["sheet"] = (text, scaled)
        original_set(self, scaled)

    def styleSheet(self):                              # noqa: N802
        """The sheet as the code set it, at 100 %."""
        actual = original_get(self)
        record = state.get("sheet")
        if record is not None and record[1] == actual:
            return record[0]
        return actual

    QApplication.setStyleSheet = setStyleSheet
    QApplication.styleSheet = styleSheet
    _ORIGINAL["app_state"] = state


_SHEET_GUARD = {"installed": False, "depth": 0, "posted": False,
                "kinds": None, "watcher": None, "opening": None}

_PENDING_SHEETS: dict = {}
_PENDING_LOCK = threading.RLock()

_TORN_DOWN: dict = {}

_HANDLER_FRAME_DEPTH = 14

_PAINT_OR_POLISH_FRAMES = frozenset({"paintEvent", "polish", "unpolish"})

_EVENT_HANDLER_FRAMES = frozenset({"changeEvent", "event", "eventFilter",
                                   "customEvent"})


def _delicate_event_types() -> frozenset:
    """The event types during which Qt must not be handed a new sheet.

    Every one is delivered from inside Qt's own styling, painting,
    reparenting or deletion code, which holds pointers into the style and
    the widget it is working on. ``PolishRequest`` is not among them: it is
    only Qt's posted request to call ``ensurePolished``, and the ``Polish``
    that call sends is. ``QWidget::setStyleSheet`` replaces the
    widget's style proxy and sends a ``StyleChange`` of its own, so a sheet
    put on from such a handler frees what the outer call is still using.

    Font and palette changes are not among them either. Qt posts the
    application-wide ones, and sends a widget's own ``FontChange`` or
    ``PaletteChange`` after it has propagated the change to the children;
    one sent by a style sheet's polish arrives inside a guarded call and
    waits through the guard's depth anyway. Deferring them broke a widget
    that resizes from a child's font sheet in its ``FontChange``: the
    Preferences help strip measured four rows before the sheet that sets
    their size had landed (main 1ea138ca7, Qt shard 0).

    :returns: the ``QEvent.Type`` values, read once.
    """
    kinds = _SHEET_GUARD["kinds"]
    if kinds is None:
        from PySide6.QtCore import QEvent

        names = ("Polish", "StyleChange", "Paint", "UpdateRequest",
                 "StyleAnimationUpdate", "ThemeChange", "DeferredDelete",
                 "ParentAboutToChange", "ParentChange")
        kinds = frozenset(getattr(QEvent.Type, name) for name in names
                          if hasattr(QEvent.Type, name))
        _SHEET_GUARD["kinds"] = kinds
    return kinds


def _polish_not_yet_begun(target, arguments) -> bool:
    """Is this the ``Polish`` of ``target`` itself, heard before Qt acts on it?

    An application filter hears a window's ``Polish`` before
    ``QWidget::event`` polishes it: nothing of Qt's styling of that window
    has started, and its children are polished after it, so a sheet put on
    the window there is the one moment that is before the polish rather
    than inside it -- the theme's documented moment to sheet a new window.
    Once the window carries ``WA_WState_Polished`` the polish is under way
    and a sheet waits like any other.

    :param target: the object the sheet is for.
    :param arguments: the handler's argument values.
    :returns: ``True`` only for that one moment.
    """
    from PySide6.QtCore import Qt

    if target is None or not any(value is target for value in arguments):
        return False
    try:
        return bool(target.isWindow()
                    and not target.testAttribute(Qt.WA_WState_Polished))
    except (AttributeError, RuntimeError, TypeError):
        return False


def _frame_handles_a_delicate_event(frame, kinds, target=None) -> bool:
    """Is ``frame`` an event handler working on one of ``kinds``?

    :param frame: a Python frame named like an event handler.
    :param kinds: the delicate event types.
    :param target: the object the sheet is for, which
        :func:`_polish_not_yet_begun` may clear.
    :returns: ``True`` when one of its arguments is such an event.
    """
    from PySide6.QtCore import QEvent

    code = frame.f_code
    try:
        values = frame.f_locals
    except Exception:
        return False
    arguments = [values.get(name)
                 for name in code.co_varnames[:code.co_argcount]]
    for value in arguments:
        if isinstance(value, QEvent):
            try:
                kind = value.type()
            except (RuntimeError, TypeError):
                return True
            if kind not in kinds:
                continue
            if (kind == QEvent.Type.Polish and code.co_name == "eventFilter"
                    and _polish_not_yet_begun(target, arguments)):
                continue
            return True
    return False


def _inside_a_delicate_moment(target=None) -> bool:
    """Is Qt in the middle of styling, painting, reparenting or deleting?

    True while a guarded ``setStyleSheet`` is already running -- its
    ``StyleChange`` and ``Polish`` reach Python handlers that restyle -- and
    while a Python ``paintEvent``, style ``polish``, or an event handler for
    one of :func:`_delicate_event_types` is on the stack.

    :param target: the object the sheet is for.
    :returns: ``True`` when a new sheet has to wait for the event loop.
    """
    if _SHEET_GUARD["depth"] > 0:
        return True
    kinds = _delicate_event_types()
    frame = sys._getframe(1)
    for _ in range(_HANDLER_FRAME_DEPTH):
        if frame is None:
            break
        name = frame.f_code.co_name
        if name in _PAINT_OR_POLISH_FRAMES:
            return True
        if (name in _EVENT_HANDLER_FRAMES
                and _frame_handles_a_delicate_event(frame, kinds, target)):
            return True
        frame = frame.f_back
    return False


def _is_alive(obj) -> bool:
    """Does ``obj`` still have its C++ object?

    :param obj: a Qt object wrapper.
    :returns: ``False`` once Qt or Python has deleted it.
    """
    if obj is None:
        return False
    try:
        from shiboken6 import isValid
    except ImportError:
        return True
    try:
        return bool(isValid(obj))
    except Exception:
        return False


def _address_of(obj):
    """The C++ address behind ``obj``, which outlives a changing wrapper.

    :param obj: a Qt object wrapper.
    :returns: the address, or ``None`` when it cannot be read.
    """
    try:
        from shiboken6 import getCppPointer

        return int(getCppPointer(obj)[0])
    except Exception:
        return None


def _prune_torn_down() -> int:
    """Forget every torn-down mark whose object Qt has since destroyed.

    A mark holds the object's wrapper, and a wrapper answers
    ``shiboken6.isValid`` false once the C++ object is gone, so a new
    object at the same address is never mistaken for the old one.

    :returns: how many marks are still standing.
    """
    for address, wrapper in list(_TORN_DOWN.items()):
        if not _is_alive(wrapper):
            _TORN_DOWN.pop(address, None)
    return len(_TORN_DOWN)


def _prune_torn_down_on_collection(phase, _info) -> None:
    """Let go of torn-down marks whose objects are gone when Python collects.

    A widget is marked while Qt delivers its ``DeferredDelete`` and its C++
    object is destroyed straight after, but the mark kept its wrapper, and
    with it everything the widget's Python side held, until the next sheet
    asked about teardown: a closed screen stayed alive in Python. Dropping a
    wrapper whose C++ object is gone frees nothing on the Qt side, and it is
    done only on the GUI thread so nothing the widget held is freed on a
    worker.

    :param phase: ``start`` or ``stop``, as ``gc.callbacks`` passes it.
    :param _info: the collection's details, unused.
    """
    if phase != "start" or not _TORN_DOWN:
        return
    if threading.current_thread() is not threading.main_thread():
        return
    _prune_torn_down()


def _mark_torn_down(obj) -> None:
    """Never put a sheet on ``obj`` or anything inside it again.

    For a widget whose deletion is under way or whose window has accepted
    its last close. Nothing is connected to the object: a slot attached to
    ``destroyed`` while its ``DeferredDelete`` is being delivered runs
    inside ``~QObject`` and crashed there (main 1ea138ca7, Qt shard 3).
    The mark is pruned once the object is gone, see
    :func:`_prune_torn_down`.

    :param obj: the object being torn down.
    """
    address = _address_of(obj)
    if address is None:
        return
    _prune_torn_down()
    if address in _TORN_DOWN:
        return
    _TORN_DOWN[address] = obj
    _drop_pending_sheets_inside(obj)


def _is_torn_down(obj) -> bool:
    """Is ``obj``, or a widget it sits inside, being destroyed?

    :param obj: a widget, or the application.
    :returns: ``True`` while the application is closing down or a parent
        chain reaches an object :func:`_mark_torn_down` recorded.
    """
    try:
        from PySide6.QtCore import QCoreApplication

        if QCoreApplication.closingDown():
            return True
    except Exception:
        pass
    if not _TORN_DOWN or not _prune_torn_down():
        return False
    node = obj
    for _ in range(256):
        if node is None:
            return False
        if _address_of(node) in _TORN_DOWN:
            return True
        parent = getattr(node, "parentWidget", None)
        if not callable(parent):
            return False
        try:
            node = parent()
        except RuntimeError:
            return True
    return False


def _on_the_gui_thread() -> bool:
    """Is this the thread the widgets live on?

    :returns: ``True`` on the main thread, the only one spaCR builds
        widgets on.
    """
    return threading.current_thread() is threading.main_thread()


def _top_level_of(obj):
    """The window ``obj`` is drawn in, for batching deferred sheets.

    :param obj: a widget, or the application.
    :returns: the top-level widget, or ``obj`` itself.
    """
    window = getattr(obj, "window", None)
    if not callable(window):
        return obj
    try:
        return window() or obj
    except RuntimeError:
        return obj


_QT_POPUP_CONTAINERS = frozenset({"QComboBoxPrivateContainer"})


def _inside_a_hidden_popup(obj) -> bool:
    """Is ``obj`` a hidden Qt popup, or inside one?

    A combo box keeps its list in a parentless, hidden popup frame
    (``QComboBoxPrivateContainer``) from the moment the list is first
    asked for. Every theme pass and every change of scale used to restyle
    each of them while hidden, 1,603 times in one Windows run, and that
    restyle crashed intermittently with an access violation (nightly
    4f723c7d8). Nothing can see a hidden popup, so its sheet waits for its
    ``Show``, where the theme puts it on before the popup paints.

    :param obj: a widget, or the application.
    :returns: ``True`` for a hidden popup window or a widget inside one.
    """
    from PySide6.QtCore import Qt

    top = _top_level_of(obj)
    opening = _SHEET_GUARD.get("opening")
    if opening is not None and top is opening:
        return False
    try:
        if not top.isWindow() or top.isVisible():
            return False
        if top.windowType() == Qt.WindowType.Popup:
            return True
        return top.metaObject().className() in _QT_POPUP_CONTAINERS
    except (AttributeError, RuntimeError, TypeError):
        return False


def _sheet_a_popup_about_to_open(popup, apply) -> None:
    """Run ``apply`` with ``popup`` counted as opening, not hidden.

    A menu's ``aboutToShow`` is emitted by ``popup()`` and ``exec()``
    before Qt measures the menu, and it is the moment the theme sheets a
    menu at, so its first frame has its styled size. The menu is still
    hidden then, and :func:`_inside_a_hidden_popup` held its sheet for the
    ``Show``, after the measuring: Windows and Linux both found a menu
    unsheeted at ``aboutToShow`` (2026-10-24). Every other rule of the
    guard -- dead objects, teardown, other threads, delicate moments --
    still applies.

    :param popup: the popup window about to open.
    :param apply: what puts its sheet on.
    """
    before = _SHEET_GUARD.get("opening")
    _SHEET_GUARD["opening"] = popup
    try:
        apply()
    finally:
        _SHEET_GUARD["opening"] = before


def _hold_for_the_show(obj, text: str, raw_set) -> None:
    """Keep ``text`` for ``obj`` until its popup window is shown.

    Unlike :func:`_defer_sheet` nothing is posted: the event loop must not
    apply it while the popup is still hidden. The theme's ``Show`` handler
    applies it through :func:`_flush_pending_sheets`.

    :param obj: a widget in a hidden popup.
    :param text: the sheet text Qt is to be given.
    :param raw_set: the setter below the guard.
    """
    with _PENDING_LOCK:
        _PENDING_SHEETS[id(obj)] = (obj, text, raw_set)


def _pending_sheet_for(obj):
    """The sheet ``obj`` is waiting for, so a getter answers what was set.

    :param obj: a widget, or the application.
    :returns: the text, or ``None`` when nothing is waiting.
    """
    if not _PENDING_SHEETS:
        return None
    with _PENDING_LOCK:
        entry = _PENDING_SHEETS.get(id(obj))
    if entry is None or entry[0] is not obj:
        return None
    return entry[1]


def _drop_pending_sheets_inside(root) -> int:
    """Forget every sheet waiting for ``root`` or a widget inside it.

    :param root: the object being torn down.
    :returns: how many were forgotten.
    """
    dropped = 0
    with _PENDING_LOCK:
        for key, entry in list(_PENDING_SHEETS.items()):
            target = entry[0]
            inside = target is root
            if not inside and _is_alive(target) and _is_alive(root):
                is_ancestor = getattr(root, "isAncestorOf", None)
                try:
                    inside = bool(callable(is_ancestor) and is_ancestor(target))
                except (RuntimeError, TypeError):
                    inside = False
            if inside or not _is_alive(target):
                _PENDING_SHEETS.pop(key, None)
                dropped += 1
    return dropped


def _post_sheet_flush() -> None:
    """Have the event loop apply the waiting sheets once, soon."""
    if _SHEET_GUARD["posted"]:
        return
    try:
        from PySide6.QtCore import QCoreApplication, QTimer
    except ImportError:
        return
    if QCoreApplication.instance() is None:
        return
    _SHEET_GUARD["posted"] = True
    if _on_the_gui_thread():
        QTimer.singleShot(0, _flush_pending_sheets)
        return
    watcher = _SHEET_GUARD["watcher"]
    if watcher is None or not _is_alive(watcher):
        _SHEET_GUARD["posted"] = False
        return
    watcher.flush_requested.emit()


def _defer_sheet(obj, text: str, raw_set) -> None:
    """Keep ``text`` for ``obj`` and apply it from the event loop.

    A later request for the same object replaces this one, so a burst of
    restyles costs Qt one repolish.

    :param obj: a widget, or the application.
    :param text: the sheet text Qt is to be given.
    :param raw_set: the setter below the guard.
    """
    with _PENDING_LOCK:
        _PENDING_SHEETS[id(obj)] = (obj, text, raw_set)
    _post_sheet_flush()


def _widgets_a_restyle_walks(obj) -> list:
    """Every widget Qt will visit while it restyles ``obj``.

    ``QApplication::setStyle`` (which an application sheet goes through)
    walks a COPY of ``allWidgets()``, and
    ``QWidget::setStyleSheet`` walks copies of each child list. Each step
    sends a ``StyleChange`` into Python, and a
    handler that lets go of a parentless widget -- or allocates enough for
    the cyclic collector to free one -- deletes a widget the copy still
    lists; Qt then reads it and the process dies in ``QApplication::setStyle``
    (reproduced; the hosted Qt shard 0 core of 2026-10-24 and
    ``tests/qt/test_stylesheets_reach_only_live_widgets_between_events_n709``).

    :param obj: a widget, or the application.
    :returns: the widgets, as wrappers, so none can be freed meanwhile.
    """
    from PySide6.QtWidgets import QApplication, QWidget

    try:
        if isinstance(obj, QApplication) or obj is None:
            app = QApplication.instance()
            return list(app.allWidgets()) if app is not None else []
        return [obj] + list(obj.findChildren(QWidget))
    except (RuntimeError, TypeError):
        return []


def _restyle_holding_every_widget(obj, call, *args):
    """Run ``call(*args)``, a Qt restyle, with nothing it walks freeable.

    The widgets it will visit are held by Python references and automatic
    cyclic collection is paused for its length, so no Python handler it
    reaches can delete one; they are let go afterwards, outside Qt's walk.

    :param obj: the widget, or the application, being restyled.
    :param call: the Qt setter below the guard.
    :returns: what ``call`` returns.
    """
    from .gc_policy import _automatic_collection_paused

    held = _widgets_a_restyle_walks(obj)
    try:
        with _automatic_collection_paused():
            return call(*args)
    finally:
        del held


def _apply_now(obj, text: str, raw_set) -> None:
    """Hand ``text`` to Qt, counting the call as a guarded application.

    :param obj: a live widget, or the application.
    :param text: the sheet text.
    :param raw_set: the setter below the guard.
    """
    if _the_sheet_is_already_on(obj, text, raw_set):
        return
    _SHEET_GUARD["depth"] += 1
    try:
        _restyle_holding_every_widget(obj, raw_set, obj, text)
    finally:
        _SHEET_GUARD["depth"] -= 1


def _the_sheet_is_already_on(obj, text: str, raw_set) -> bool:
    """Does Qt already hold exactly ``text`` for ``obj``?

    Qt repolishes the whole subtree on every ``setStyleSheet``, even for the
    text it already has, and each repolish relays every height-for-width
    layout inside it: the 0.3-1 s GUI-thread stalls of lessons 70 and 85
    opening a module were mostly restyles to an unchanged sheet. The text
    compared is the scaled one, so a change of scale is never skipped.

    :param obj: a widget, or the application.
    :param text: the sheet text about to be handed to Qt.
    :param raw_set: the setter below the guard.
    :returns: ``True`` when the call would change nothing.
    """
    if raw_set is _ORIGINAL.get("guard_raw_widget_set"):
        getter = _ORIGINAL.get("guard_raw_widget_get")
    elif raw_set is _ORIGINAL.get("guard_raw_app_set"):
        getter = _ORIGINAL.get("guard_raw_app_get")
    else:
        return False
    try:
        return getter is not None and getter(obj) == text
    except (RuntimeError, TypeError):
        return False


def _set_sheet_guarded(obj, text: str, raw_set) -> bool:
    """Put ``text`` on ``obj`` only at a moment Qt can take it.

    The four rules, each one a way a stylesheet crashes Qt natively:

    * a dead object (``shiboken6.isValid`` is false) is left alone;
    * an object being destroyed, or inside a window that is, is left alone
      and anything it was waiting for is forgotten;
    * from another thread, or from inside a polish, a style change, a
      paint, a reparent or a deletion -- including the ``StyleChange`` a
      guarded call itself sends -- the sheet waits for the event loop;
    * the waiting sheets are applied one top-level window at a time.

    :param obj: a widget, or the application.
    :param text: the sheet text Qt is to be given.
    :param raw_set: the setter below the guard.
    :returns: ``True`` when Qt was given the sheet now.
    """
    if not _is_alive(obj):
        return False
    if not _on_the_gui_thread():
        _defer_sheet(obj, text, raw_set)
        return False
    if _is_torn_down(obj):
        with _PENDING_LOCK:
            _PENDING_SHEETS.pop(id(obj), None)
        return False
    if _inside_a_hidden_popup(obj):
        _hold_for_the_show(obj, text, raw_set)
        return False
    if _inside_a_delicate_moment(obj):
        _defer_sheet(obj, text, raw_set)
        return False
    if _PENDING_SHEETS:
        with _PENDING_LOCK:
            _PENDING_SHEETS.pop(id(obj), None)
    _apply_now(obj, text, raw_set)
    return True


def _flush_pending_sheets(window=None) -> int:
    """Apply the waiting sheets, grouped by their top-level window.

    Run from the event loop, where no Qt styling is under way. A sheet for
    an object that has died or is being torn down since it was asked for is
    dropped. A flush asked for at a moment that is itself delicate is
    posted again instead.

    :param window: only the sheets drawn in this top-level window, for a
        window about to be shown; ``None`` applies them all.
    :returns: how many sheets Qt was given.
    """
    if window is None:
        _SHEET_GUARD["posted"] = False
    if not _PENDING_SHEETS:
        return 0
    if not _on_the_gui_thread():
        return 0
    if _inside_a_delicate_moment():
        _post_sheet_flush()
        return 0
    batches: dict = {}
    with _PENDING_LOCK:
        for key, entry in list(_PENDING_SHEETS.items()):
            target = entry[0]
            if not _is_alive(target):
                _PENDING_SHEETS.pop(key, None)
                continue
            top = _top_level_of(target)
            if window is not None and top is not window:
                continue
            if window is None and _inside_a_hidden_popup(target):
                continue
            _PENDING_SHEETS.pop(key, None)
            batches.setdefault(id(top), (top, []))[1].append(entry)
    applied = 0
    for top, entries in batches.values():
        if not _is_alive(top) or _is_torn_down(top):
            continue
        for target, text, raw_set in entries:
            if not _is_alive(target) or _is_torn_down(target):
                continue
            _apply_now(target, text, raw_set)
            applied += 1
    return applied


def _note_sheet_lifecycle(watched, kind) -> None:
    """Record a widget whose deletion Qt is delivering now.

    Only widgets carry sheets, so no other object's deletion is touched.

    :param watched: the object the event is for.
    :param kind: the event type.
    """
    from PySide6.QtCore import QEvent

    if kind == QEvent.Type.DeferredDelete:
        from PySide6.QtWidgets import QWidget

        if isinstance(watched, QWidget):
            _mark_torn_down(watched)


def _sheet_guard_watcher_init(watcher, parent=None) -> None:
    """Start the guard's watcher; cross-thread flushes reach the GUI thread.

    :param watcher: the watcher.
    :param parent: its Qt parent, normally none.
    """
    from PySide6.QtCore import QObject, Qt

    QObject.__init__(watcher, parent)
    watcher.flush_requested.connect(watcher._flush, Qt.ConnectionType.QueuedConnection)


def _sheet_guard_watcher_flush(watcher) -> None:
    """Apply the waiting sheets on the GUI thread.

    :param watcher: the watcher.
    """
    _flush_pending_sheets()


def _sheet_guard_watcher_event_filter(watcher, watched, event):
    """Note a deletion the hub delivers; never consume the event.

    :param watcher: the watcher.
    :param watched: the object the event is for.
    :param event: the event.
    :returns: ``False`` always.
    """
    try:
        _note_sheet_lifecycle(watched, event.type())
    except (AttributeError, RuntimeError, TypeError):
        pass
    return False


def _sheet_guard_watcher_class():
    """The guard's application watcher class, built on first use.

    It hears ``DeferredDelete`` through the application event hub, so an
    object about to be deleted is never restyled, and it carries a sheet
    asked for on another thread across to the GUI thread. Built with
    ``type`` from the module-level functions above so that importing this
    module stays free of Qt.

    :returns: the class.
    """
    cached = _SHEET_GUARD.get("watcher_class")
    if cached is not None:
        return cached
    from PySide6.QtCore import QObject, Signal

    cached = type("_SheetGuardWatcher", (QObject,), {
        "__doc__": "Marks objects whose deletion is being delivered.",
        "flush_requested": Signal(),
        "__init__": _sheet_guard_watcher_init,
        "_flush": _sheet_guard_watcher_flush,
        "eventFilter": _sheet_guard_watcher_event_filter,
    })
    _SHEET_GUARD["watcher_class"] = cached
    return cached


def _watch_for_teardown() -> bool:
    """Start the guard's watcher on the application, once.

    :returns: ``True`` when it is watching.
    """
    from PySide6.QtCore import QCoreApplication, QEvent

    app = QCoreApplication.instance()
    if app is None or not _on_the_gui_thread():
        return False
    watcher = _SHEET_GUARD["watcher"]
    if watcher is not None and _is_alive(watcher):
        return True
    watcher = _sheet_guard_watcher_class()()
    _SHEET_GUARD["watcher"] = watcher
    from .gil_priority import _watch_application_events

    return _watch_application_events(app, watcher,
                                     {QEvent.Type.DeferredDelete})


def _guarded_widget_set_style_sheet(widget, text):
    """``QWidget.setStyleSheet`` once the guard is on: apply when Qt can.

    :param widget: the widget.
    :param text: the sheet text; ``None`` is the empty sheet.
    """
    _set_sheet_guarded(widget, "" if text is None else str(text),
                       _ORIGINAL["guard_raw_widget_set"])


def _guarded_widget_style_sheet(widget):
    """``QWidget.styleSheet`` once the guard is on.

    :param widget: the widget.
    :returns: the sheet last asked for, applied or still waiting.
    """
    waiting = _pending_sheet_for(widget)
    if waiting is None:
        return _ORIGINAL["guard_raw_widget_get"](widget)
    return waiting


def _guarded_app_set_style_sheet(app, text):
    """``QApplication.setStyleSheet`` once the guard is on.

    :param app: the application.
    :param text: the sheet text; ``None`` is the empty sheet.
    """
    _set_sheet_guarded(app, "" if text is None else str(text),
                       _ORIGINAL["guard_raw_app_set"])


def _guarded_app_style_sheet(app):
    """``QApplication.styleSheet`` once the guard is on.

    :param app: the application.
    :returns: the application sheet last asked for.
    """
    waiting = _pending_sheet_for(app)
    if waiting is None:
        return _ORIGINAL["guard_raw_app_get"](app)
    return waiting


def _held_app_set_style(*args):
    """``QApplication.setStyle`` with every widget it walks held alive.

    ``QApplication.setStyle`` walks a copy of ``allWidgets()`` and calls
    into Python for each widget, exactly as an application sheet does; see
    :func:`_widgets_a_restyle_walks`. ``setPalette`` and ``setFont`` are not
    wrapped: Qt posts their per-widget change events instead of sending
    them from inside a walk.

    :param args: as for ``QApplication.setStyle``.
    :returns: what Qt returns.
    """
    return _restyle_holding_every_widget(None, _ORIGINAL["guard_raw_app_style"],
                                         *args)


def _install_sheet_guard() -> bool:
    """Route every ``setStyleSheet`` in the process through the guard.

    Installed below the scaling layer, so the scaled setters and every
    caller that reaches past them through :data:`_ORIGINAL` are guarded
    too. ``styleSheet`` answers a sheet that is still waiting, so code that
    compares its own sheet sees what it asked for. Idempotent.

    :returns: ``True`` the first time.
    """
    import gc

    from PySide6.QtWidgets import QApplication, QWidget

    _watch_for_teardown()
    if _prune_torn_down_on_collection not in gc.callbacks:
        gc.callbacks.append(_prune_torn_down_on_collection)
    if _SHEET_GUARD["installed"]:
        return False
    _ORIGINAL["guard_raw_widget_set"] = QWidget.setStyleSheet
    _ORIGINAL["guard_raw_widget_get"] = QWidget.styleSheet
    _ORIGINAL["guard_raw_app_set"] = QApplication.setStyleSheet
    _ORIGINAL["guard_raw_app_get"] = QApplication.styleSheet
    _ORIGINAL["guard_raw_app_style"] = QApplication.setStyle
    _ORIGINAL[(QWidget, "raw_setStyleSheet")] = QWidget.setStyleSheet
    _ORIGINAL[(QApplication, "raw_setStyleSheet")] = QApplication.setStyleSheet
    QWidget.setStyleSheet = _guarded_widget_set_style_sheet
    QWidget.styleSheet = _guarded_widget_style_sheet
    QApplication.setStyleSheet = _guarded_app_set_style_sheet
    QApplication.styleSheet = _guarded_app_style_sheet
    QApplication.setStyle = staticmethod(_held_app_set_style)
    _SHEET_GUARD["installed"] = True
    return True


def _close_without_restyling(window, closing: bool = True) -> None:
    """Stop, or resume, every restyle of a main window that is closing.

    From the moment the window has passed every check that could keep it
    open, its panels shut down and its workers are released; a sheet put on
    any widget inside it then can only reach a widget on its way out.

    :param window: the main window.
    :param closing: ``False`` puts restyling back when the close was not
        accepted after all.
    """
    if closing:
        _mark_torn_down(window)
        return
    address = _address_of(window)
    if address is not None:
        _TORN_DOWN.pop(address, None)


def install_scaling_layer() -> bool:
    """Put the scaling layer on the Qt classes. Idempotent.

    Called from :func:`spacr.qt.app.launch` before the first widget, so every
    size spaCR sets is remembered at 100 %. At 100 % every replacement hands
    Qt exactly the value it was given, which is why a default session draws
    the pixels it drew before the layer existed.

    :returns: ``True`` the first time, ``False`` if it was already in place.
    """
    global _INSTALLED
    if _INSTALLED:
        return False
    _install_sheet_guard()
    _install_widget_setters()
    _install_layout_setters()
    _install_icon_setters()
    _install_splitter_setters()
    _install_application_sheet()
    _INSTALLED = True
    return True


def add_listener(callback: Callable[[float], None]) -> None:
    """Call ``callback(scale)`` after every change of GUI scale.

    Held weakly when it is a bound method, so a listener does not keep its
    widget alive.

    :param callback: called with the new scale.
    """
    try:
        ref = weakref.WeakMethod(callback)
    except TypeError:
        ref = (lambda cb=callback: cb)
    _LISTENERS.append(ref)


def _reapply_widget(widget, factor: float, previous: float = 1.0) -> None:
    """Put a widget's remembered sizes, sheet and icon back at ``factor``.

    :param widget: the widget to re-scale.
    :param factor: the scale it is drawn at now.
    :param previous: the scale it was drawn at, which is how an icon size
        nobody ever set -- matplotlib's toolbar takes the style's 24 px --
        is read back into 100 % units the first time the scale moves.
    """
    from PySide6.QtCore import QSize
    from PySide6.QtWidgets import QWidget

    record = getattr(widget, "_gs_geo", None)
    if record:
        applied = _store(widget, "_gs_geo_set")
        for key, value in record.items():
            wanted = _bounded_main_window_minimum(widget, key, scale_int(value, factor))
            if applied.get(key) != wanted:
                applied[key] = wanted
                _ORIGINAL[(QWidget, _DIMS[key][0])](widget, wanted)
    sheet = getattr(widget, "_gs_sheet", None)
    if sheet is not None:
        scaled = scale_qss_text(sheet[0], factor)
        if scaled != sheet[1]:
            widget._gs_sheet = (sheet[0], scaled)
            _ORIGINAL[(QWidget, "setStyleSheet")](widget, scaled)
    icon = getattr(widget, "_gs_icon", None)
    if icon is None and factor != previous:
        getter = _original_for(widget, "iconSize")
        if getter is not None:
            try:
                actual = getter(widget)
                base = (int(round(actual.width() / previous)),
                        int(round(actual.height() / previous)))
            except (RuntimeError, ZeroDivisionError):
                base = None
            if base and base[0] > 0 and base[1] > 0:
                icon = (base, (0, 0))
                try:
                    widget._gs_icon = icon
                except (AttributeError, TypeError):
                    icon = None
    if icon is not None:
        wanted = tuple(scale_int(v, factor) for v in icon[0])
        if wanted != icon[1]:
            setter = _original_for(widget, "setIconSize")
            if setter is not None:
                widget._gs_icon = (icon[0], wanted)
                setter(widget, QSize(*wanted))


def _original_for(obj, name: str):
    """The Qt method ``name`` of ``obj``'s class, before the layer."""
    for cls in type(obj).__mro__:
        found = _ORIGINAL.get((cls, name))
        if found is not None:
            return found
    return None


def _reapply_layout(layout, factor: float) -> None:
    """Put a layout's remembered margins, spacing and spacers back."""
    margins = getattr(layout, "_gs_margins", None)
    if margins is not None:
        wanted = tuple(scale_int(v, factor) for v in margins[0])
        if wanted != margins[1]:
            layout._gs_margins = (margins[0], wanted)
            setter = _original_for(layout, "setContentsMargins")
            if setter is not None:
                setter(layout, *wanted)
    spacing = getattr(layout, "_gs_spacing", None)
    if spacing:
        names = {"sp": "setSpacing", "hs": "setHorizontalSpacing",
                 "vs": "setVerticalSpacing"}
        for key, (base, applied) in list(spacing.items()):
            wanted = scale_int(base, factor)
            if wanted != applied:
                setter = _original_for(layout, names[key])
                if setter is not None:
                    setter(layout, wanted)
                    spacing[key] = (base, wanted)
    for item, base, horizontal in getattr(layout, "_gs_spacers", None) or ():
        from PySide6.QtWidgets import QSizePolicy

        wanted = scale_int(base, factor)
        try:
            if layout.indexOf(item) < 0:
                continue
            if horizontal:
                item.changeSize(wanted, 0, QSizePolicy.Fixed,
                                QSizePolicy.Minimum)
            else:
                item.changeSize(0, wanted, QSizePolicy.Minimum,
                                QSizePolicy.Fixed)
        except RuntimeError:
            continue
    from PySide6.QtWidgets import QLayout

    if not isinstance(layout, QLayout):
        return
    layout.invalidate()
    for child in _child_layouts(layout):
        _reapply_layout(child, factor)


def _child_layouts(layout) -> list:
    """The layouts nested directly in ``layout``, without touching its items.

    NEVER BY ``itemAt``. Asking a layout for its items gives each one a
    Python wrapper that the layout's wrapper then keeps, and a layout item
    is not a QObject, so nothing tells the wrapper when Qt deletes the item.
    Qt does that all the time: a ``QDialogButtonBox`` throws its items away
    and makes new ones whenever its buttons change, and a widget deleted out
    of a layout takes its item with it. The wrapper stays registered at the
    freed address, and the next object Qt puts there -- a message box's
    layout, a widget -- comes back to Python as that ``QWidgetItem``. That
    is how a live scale change, which used to wrap every item of every
    layout in the application, took a Qt shard down with a segfault and a
    ``free(): invalid pointer`` (item 43). A nested layout is a QObject child
    of the layout it sits in, so ``findChildren`` finds it without wrapping
    anything that is not a QObject.

    :param layout: the layout to look inside.
    :returns: its directly nested layouts.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLayout

    try:
        found = layout.findChildren(QLayout, "",
                                    Qt.FindChildOption.FindDirectChildrenOnly)
    except (RuntimeError, TypeError):
        return []
    return [child for child in found
            if isinstance(child, QLayout) and child is not layout]


def _rescale_canvases(widgets, factor: float) -> int:
    """Set every matplotlib canvas's dpi for the new scale."""
    count = 0
    for widget in widgets:
        if hasattr(widget, "figure") and hasattr(widget, "draw_idle"):
            if apply_canvas_dpi(widget, factor):
                count += 1
    return count


def apply_canvas_dpi(canvas, factor: Optional[float] = None) -> bool:
    """Draw a matplotlib canvas at base dpi x GUI scale x its preview scale.

    A figure draws in points, so its dpi is what makes a 9 pt label take
    more or fewer pixels; scaling the dpi scales every label, dot and line
    while the widget keeps the size its layout gives it.

    :param canvas: a ``FigureCanvasQTAgg``.
    :param factor: the GUI scale; the current one by default.
    :returns: ``True`` if the canvas's dpi was set.
    """
    figure = getattr(canvas, "figure", None)
    if figure is None:
        return False
    factor = _SCALE if factor is None else float(factor)
    if _exempt(canvas):
        factor = 1.0
    try:
        base = getattr(figure, "_spacr_base_dpi", None)
        if base is None:
            base = float(getattr(figure, "_original_dpi", figure.dpi))
            figure._spacr_base_dpi = base
        preview = float(getattr(canvas, "_spacr_preview_scale", 1.0) or 1.0)
        wanted = base * factor * preview
        if abs(float(getattr(figure, "_original_dpi", 0)) - wanted) < 1e-9:
            return True
        ratio = float(getattr(canvas, "device_pixel_ratio", 1.0) or 1.0)
        figure._original_dpi = wanted
        figure._set_dpi(wanted * ratio, forward=False)
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtGui import QResizeEvent

        QCoreApplication.sendEvent(
            canvas, QResizeEvent(canvas.size(), canvas.size()))
        canvas.draw_idle()
        return True
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not rescale a figure canvas", exc_info=True)
        return False


def set_gui_scale_live(scale: float) -> float:
    """Draw spaCR at ``scale`` now: every live widget, and every one built later.

    Does not save the preference; see :func:`spacr.qt.preferences.set_gui_scale`.

    :param scale: the factor, 1.0 = 100 %; clamped to 10-200 %.
    :returns: the scale applied.
    """
    global _SCALE
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication, QSplitter

    from .preferences import GUI_SCALE_MAX, GUI_SCALE_MIN

    install_scaling_layer()
    scale = max(GUI_SCALE_MIN, min(GUI_SCALE_MAX, float(scale)))
    old = _SCALE
    if abs(scale - old) < 1e-9:
        return old
    _SCALE = scale
    app = QApplication.instance()
    if app is None:
        return scale

    state = _ORIGINAL.get("app_state") or {}
    record = state.get("sheet")
    if record is not None and record[0]:
        scaled = scale_qss_text(record[0], scale)
        state["sheet"] = (record[0], scaled)
        _ORIGINAL[(QApplication, "setStyleSheet")](app, scaled)

    from PySide6.QtWidgets import QLayout, QWidget

    widgets = [widget for widget in app.allWidgets()
               if isinstance(widget, QWidget)]
    for widget in widgets:
        try:
            factor = 1.0 if _exempt(widget) else scale
            _reapply_widget(widget, factor, 1.0 if _exempt(widget) else old)
            layout = widget.layout()
            if isinstance(layout, QLayout):
                _reapply_layout(layout, factor)
            if isinstance(widget, QSplitter):
                record = getattr(widget, "_gs_sizes", None)
                if record is not None:
                    actual = _ORIGINAL[(QSplitter, "sizes")](widget)
                    widget._gs_sizes = (
                        [int(round(v / factor)) if v > 0 else v
                         for v in actual], list(actual))
        except RuntimeError:
            continue
    _rescale_canvases(widgets, scale)
    for top in app.topLevelWidgets():
        try:
            QCoreApplication.postEvent(top, QEvent(QEvent.LayoutRequest))
            top.update()
        except RuntimeError:
            continue
    for ref in list(_LISTENERS):
        callback = ref()
        if callback is None:
            _LISTENERS.remove(ref)
            continue
        try:
            callback(scale)
        except Exception:                                    # noqa: BLE001
            LOG.debug("a GUI scale listener failed", exc_info=True)
    LOG.info("GUI scale %d %%", round(scale * 100))
    return scale


def apply_saved_gui_scale() -> float:
    """Start at the saved GUI scale; called once the application exists.

    :returns: the scale in force.
    """
    from .preferences import get_gui_scale

    return set_gui_scale_live(get_gui_scale())


KEEP_SECONDS = 15


def _dialog_classes():
    """Build :class:`KeepOrRevertDialog` on first use (keeps import light)."""
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QPushButton,
                                   QVBoxLayout)

    from .i18n import tr
    from .theme import font_px

    class KeepOrRevertDialog(QDialog):
        """"Keep these settings?" with a countdown that reverts on its own.

        Exempt from the GUI scale and sized in its own sheet, so it can be
        read at 10 % GUI and 10 % font. Keep accepts; Revert, Esc, closing
        the window and the end of the countdown all reject.

        :param parent: the window it is centred on.
        :param seconds: how long before it reverts by itself.
        :param what: the line under the question naming what changed.
        """

        def __init__(self, parent=None, seconds: int = KEEP_SECONDS,
                     what: str = ""):
            """Build the question and start the countdown."""
            super().__init__(parent)
            self.setProperty(EXEMPT, True)
            self.setObjectName("SpacrKeepOrRevert")
            self.setWindowTitle(tr("Keep these settings?"))
            self._left = max(1, int(seconds))
            column = QVBoxLayout(self)
            column.setContentsMargins(20, 18, 20, 16)
            column.setSpacing(10)
            self.question = QLabel(tr("Keep these settings?"), self)
            self.question.setObjectName("SpacrKeepQuestion")
            column.addWidget(self.question)
            if what:
                detail = QLabel(what, self)
                detail.setObjectName("SpacrKeepDetail")
                detail.setWordWrap(True)
                column.addWidget(detail)
            self.countdown = QLabel(self)
            self.countdown.setObjectName("SpacrKeepCountdown")
            column.addWidget(self.countdown)
            row = QHBoxLayout()
            row.addStretch(1)
            self.revert_button = QPushButton(tr("Revert"), self)
            self.keep_button = QPushButton(tr("Keep"), self)
            self.revert_button.clicked.connect(self.reject)
            self.keep_button.clicked.connect(self.accept)
            row.addWidget(self.revert_button)
            row.addWidget(self.keep_button)
            column.addLayout(row)
            self.keep_button.setDefault(True)
            self.keep_button.setFocus(Qt.OtherFocusReason)
            title_px = font_px(17, scale=1.0)
            body_px = font_px(13, scale=1.0)
            self._own_rule = (
                "QDialog#SpacrKeepOrRevert QLabel#SpacrKeepQuestion "
                f"{{ font-size: {title_px}px; font-weight: 600; }}"
                "QDialog#SpacrKeepOrRevert QLabel#SpacrKeepDetail, "
                "QDialog#SpacrKeepOrRevert QLabel#SpacrKeepCountdown "
                f"{{ font-size: {body_px}px; }}"
                "QDialog#SpacrKeepOrRevert QPushButton "
                f"{{ font-size: {body_px}px; min-width: 80px; min-height: 26px; "
                "padding: 4px 14px; }")
            try:
                from .theme import set_a_sheeted_widgets_own_rule
                set_a_sheeted_widgets_own_rule(self, self._own_rule)
            except Exception:                                # noqa: BLE001
                self.setStyleSheet(self._own_rule)
            self.setMinimumWidth(360)
            self._timer = QTimer(self)
            self._timer.setInterval(1000)
            self._timer.timeout.connect(self._tick)
            self._show_left()
            self._timer.start()

        def _show_left(self) -> None:
            """Say how long is left."""
            self.countdown.setText(tr("Reverting in {seconds} s",
                                      seconds=self._left))

        def _tick(self) -> None:
            """One second less; at zero, revert."""
            self._left -= 1
            if self._left <= 0:
                self._timer.stop()
                self.reject()
                return
            self._show_left()

        def done(self, result):
            """Stop the countdown on any answer.

            :param result: the dialog result.
            """
            self._timer.stop()
            super().done(result)

    return KeepOrRevertDialog


_DIALOG_CLASS = None


def keep_or_revert_dialog(parent=None, seconds: int = KEEP_SECONDS,
                          what: str = ""):
    """A new :class:`KeepOrRevertDialog` (see :func:`_dialog_classes`).

    :param parent: the window it is centred on.
    :param seconds: how long before it reverts by itself.
    :param what: the line naming what changed.
    """
    global _DIALOG_CLASS
    if _DIALOG_CLASS is None:
        _DIALOG_CLASS = _dialog_classes()
    return _DIALOG_CLASS(parent, seconds=seconds, what=what)


def _apply_scales(gui: float, font: float) -> None:
    """Save and draw ``gui`` and ``font`` scale now."""
    from . import preferences

    preferences.set_gui_scale(gui)
    preferences.set_font_scale(font)
    set_gui_scale_live(gui)
    try:
        preferences.apply_preferences_to_app()
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not re-apply the font scale", exc_info=True)
    refresh_the_windows()


def change_scales(parent=None, *, gui: Optional[float] = None,
                  font: Optional[float] = None, ask: bool = True,
                  seconds: int = KEEP_SECONDS,
                  previous: Optional[tuple] = None,
                  require_parent: bool = True,
                  on_done: Optional[Callable[[bool], None]] = None):
    """Apply a new GUI and/or font scale now, then ask whether to keep it.

    The question is shown without blocking (``open``, not ``exec``): the
    rest of spaCR keeps running under it, and the countdown reverts by
    itself if nobody answers.

    :param parent: the window the question is centred on.
    :param gui: the new GUI scale; unchanged when ``None``.
    :param font: the new font scale; unchanged when ``None``.
    :param ask: ``False`` keeps without asking.
    :param seconds: the countdown before it reverts by itself.
    :param previous: ``(gui, font)`` to revert to when the caller has
        already applied and saved the new values (the Z + wheel gesture);
        nothing is applied again then.
    :param require_parent: ask only when there is a window to centre the
        question on. ``False`` asks anyway, which is what a test does.
    :param on_done: called with ``True`` when kept, ``False`` when reverted.
    :returns: the question dialog, or ``None`` when nothing was asked.
    """
    from . import preferences
    from .i18n import tr

    if previous is None:
        old_gui, old_font = (preferences.get_gui_scale(),
                             preferences.get_font_scale())
        new_gui = old_gui if gui is None else float(gui)
        new_font = old_font if font is None else float(font)
        if abs(new_gui - old_gui) < 1e-9 and abs(new_font - old_font) < 1e-9:
            return None
        _apply_scales(new_gui, new_font)
    else:
        old_gui, old_font = previous
        new_gui = preferences.get_gui_scale() if gui is None else float(gui)
        new_font = preferences.get_font_scale() if font is None else float(
            font)
    if ask and parent is None:
        from PySide6.QtWidgets import QApplication

        parent = QApplication.activeWindow()
        ask = parent is not None or not require_parent
    if not ask:
        if on_done is not None:
            on_done(True)
        return None
    what = tr("GUI scale {gui}%, font scale {font}%",
              gui=int(round(new_gui * 100)), font=int(round(new_font * 100)))
    dialog = keep_or_revert_dialog(parent, seconds=seconds, what=what)

    def _answered(result) -> None:
        """Keep, or put the old values back."""
        from PySide6.QtWidgets import QDialog

        kept = result == QDialog.Accepted
        if not kept:
            _apply_scales(old_gui, old_font)
        if on_done is not None:
            try:
                on_done(kept)
            except Exception:                                # noqa: BLE001
                LOG.debug("a scale answer callback failed", exc_info=True)
        dialog.deleteLater()

    dialog.finished.connect(_answered)
    dialog.open()
    return dialog


def refresh_the_windows() -> int:
    """Rebuild what a style sheet cannot reach: icons, tiles, window chrome.

    The same step the Z + wheel gesture ends with. ``MainWindow.refresh_theme``
    repaints the marks in the window corner and the Home tiles, which paint
    their own pixmaps at a size they compute -- and a pixmap is not a size
    the scaling layer ever sees.

    :returns: how many windows rebuilt.
    """
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        return 0
    done = 0
    for window in list(app.topLevelWidgets()):
        refresh = getattr(window, "refresh_theme", None)
        if not callable(refresh):
            continue
        try:
            refresh()
            done += 1
        except Exception:                                    # noqa: BLE001
            LOG.debug("a window would not rebuild after a scale change",
                      exc_info=True)
    return done


def reset_every_scale(parent=None) -> bool:
    """Restore the 90 % GUI and font defaults and request preview-scale resets.

    The backup way out of a scale too small to read: ``Ctrl+Alt+0``. It
    needs no reading and no answer.

    :param parent: unused; kept so a shortcut can pass its window.
    :returns: ``True``.
    """
    from .preferences import DEFAULT_FONT_SCALE, DEFAULT_GUI_SCALE

    _apply_scales(DEFAULT_GUI_SCALE, DEFAULT_FONT_SCALE)
    try:
        from .widgets.preview_scale import reset_all_preview_scales

        reset_all_preview_scales()
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not reset the preview scales", exc_info=True)
    return True


def mend_matplotlib_icons() -> bool:
    """Keep matplotlib's toolbar icons full size below a device ratio of 1.

    Matplotlib's toolbar icon engine multiplies the size Qt asks for by the
    device pixel ratio, and Qt has already done so; above 1 the two cancel,
    below 1 (a user's ``QT_SCALE_FACTOR`` under 1) they compound. Measured
    offscreen at a ratio of 0.5: the icons drew 3 px tall instead of 11.
    Holding the engine's ratio at 1 or more draws them 11 px tall and
    changes nothing at a ratio of 1 or 2. Idempotent.

    :returns: ``True`` if the mend is in place.
    """
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


def follow_canvas(canvas) -> bool:
    """Draw a new matplotlib canvas at the current GUI scale.

    :param canvas: a ``FigureCanvasQTAgg`` just built.
    :returns: ``True`` if its dpi was set.
    """
    if _SCALE == 1.0 and float(getattr(canvas, "_spacr_preview_scale",
                                       1.0) or 1.0) == 1.0:
        return False
    return apply_canvas_dpi(canvas)
