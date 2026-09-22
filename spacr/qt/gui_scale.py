"""The whole-GUI scale, applied live: one factor for every size spaCR sets (471).

WHAT IT IS. "GUI scale" in Preferences (10 % to 200 %, default 100 %) scales
widget sizes, fixed sizes, margins, spacing, style-sheet sizes (fonts and
paddings included), icons, splitter sizes and figure dpi -- and it does so
while spaCR runs, with no restart.

HOW, WITHOUT TOUCHING 1,420 CALL SITES. Counted on nightly 2026-09-22 across
the 328 modules under ``spacr/qt``: about 1,420 hard-coded geometry calls in
181 files (470 ``setContentsMargins``, 489 ``setSpacing``, 179
``setMinimumWidth/Height``, 67 ``setMaximumWidth/Height``, 74 ``setFixed*``,
30 ``setMinimumSize``, 56 ``QSize``, 64 ``resize``, 6 ``setIconSize``) and
486 ``px`` literals in style-sheet text in 64 files. Routing each by hand is
a change to every screen, and every size added later would have to remember
it. Instead :func:`install_scaling_layer` -- called once from
:func:`spacr.qt.app.launch` before the first widget exists -- replaces the
setters themselves on the Qt classes (``QWidget``'s size setters and
``setStyleSheet``, the layouts' margin and spacing setters, every
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
what is inside them -- and ``sizes()`` answers in 100 % units, so the sizes
slice B persists survive a change of scale.

HOW IT COMPOSES WITH FONT SCALE. Font scale stays where it is: it writes the
font sizes into the theme's style sheet. That sheet goes through the
``setStyleSheet`` replacement like any other, so a font size ends up as
base x font scale x GUI scale. 50 % GUI at 200 % font is half-size widgets
with text the usual size on screen.

A USER'S OWN ``QT_SCALE_FACTOR`` is not touched and multiplies on the
outside, as Qt always has.

KEEP OR REVERT. Every change of GUI scale or font scale applies at once and
then asks "Keep these settings?" (:class:`KeepOrRevertDialog`), counting down
15 s; the countdown, Esc and closing the dialog all put the old values back.
The dialog is exempt from the scale and sets its own text size, so it is
readable at 10 % GUI and 10 % font alike. ``Ctrl+Alt+0``
(:func:`reset_every_scale`) remains the backup.

WHAT DOES NOT FOLLOW LIVE. Sizes the scale cannot see because they do not
pass through a setter: text and shapes a widget paints itself at fixed pixel
coordinates, pixmaps a widget scales to a number it computed, ``setFont``
with an explicit size, ``move``/``setGeometry`` positions, header section
sizes, a Python ``sizeHint`` override that returns a constant, the Fusion
style's own metrics that the theme sheet does not restate, and pyqtgraph's
axis text. Sizes computed from font metrics are already scaled once by the
font and are scaled again by the setter -- the one way the layer can
over-shrink. The main window's own size is left alone, so the gained room
goes to the content.
"""
from __future__ import annotations

import logging
import re
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


def _scaled_px(match, factor: float) -> str:
    """One ``Npx`` at ``factor``, never rounding a non-zero size to nothing."""
    value = float(match.group(1))
    if value == 0:
        return match.group(0)
    scaled = int(round(abs(value) * factor)) or 1
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
        + _PX.sub(lambda p: _scaled_px(p, factor), m.group(3)), text)
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


def _set_dims(widget, **dims) -> None:
    """Remember and apply minimum/maximum dimensions of ``widget``."""
    from PySide6.QtWidgets import QWidget

    record = _store(widget, "_gs_geo")
    applied = _store(widget, "_gs_geo_set")
    factor = _factor_for(widget)
    for key, value in dims.items():
        if not _is_our_own_value(record.get(key), value, factor):
            record[key] = int(value)
        wanted = scale_int(record[key], factor)
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
            wanted = scale_int(value, factor)
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
    for index in range(layout.count()):
        item = layout.itemAt(index)
        child = item.layout() if item is not None else None
        if isinstance(child, QLayout) and child is not layout:
            _reapply_layout(child, factor)


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

    widgets = list(app.allWidgets())
    for widget in widgets:
        try:
            factor = 1.0 if _exempt(widget) else scale
            _reapply_widget(widget, factor, 1.0 if _exempt(widget) else old)
            layout = widget.layout()
            if layout is not None:
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
            self._own_rule = (
                "QDialog#SpacrKeepOrRevert QLabel#SpacrKeepQuestion "
                "{ font-size: 17px; font-weight: 600; }"
                "QDialog#SpacrKeepOrRevert QLabel#SpacrKeepDetail, "
                "QDialog#SpacrKeepOrRevert QLabel#SpacrKeepCountdown "
                "{ font-size: 13px; }"
                "QDialog#SpacrKeepOrRevert QPushButton "
                "{ font-size: 13px; min-width: 80px; min-height: 26px; "
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
    """Put GUI scale, font scale and every preview scale back to 100 %, now.

    The backup way out of a scale too small to read: ``Ctrl+Alt+0``. It
    needs no reading and no answer.

    :param parent: unused; kept so a shortcut can pass its window.
    :returns: ``True``.
    """
    _apply_scales(1.0, 1.0)
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
