"""One scale slider per live preview, scaling that preview alone (item 471).

The whole-GUI scale (:mod:`spacr.qt.gui_scale`) is fixed at startup and
applies everywhere. A preview wants its own: a Mask preview on a laptop
may need to shrink its controls to give the images room, while the settings
beside it stay readable. Every preview -- Mask, Measure, Plaque, Timelapse,
Motility and the image-UMAP views -- installs the same control through
:func:`install_preview_scale`, so there is one implementation of it.

WHAT IT SCALES, LIVE, WITHOUT A RESTART. Everything inside the preview that
spaCR sized in pixels:

* style-sheet sizes -- font sizes, paddings, margins, minimum and maximum
  widths and heights, radii -- both the ones the preview inherits from the
  application sheet and the ones its widgets set on themselves. Border
  widths are left alone, so a 1 px rule stays a rule;
* minimum, maximum and fixed widget sizes set in code;
* layout margins and spacing;
* button icon sizes;
* and, through :meth:`PreviewScaler.add_hook`, what only the panel knows
  how to redraw: Measure's crop thumbnails, a matplotlib figure's dots and
  labels (:func:`scale_figure_canvas`).

Each value's own base is remembered on the widget, so scaling is always
from the size the code asked for and never compounds. A size the code sets
again later becomes the new base. At 100 % nothing is touched at all -- no
sheet, no size, no property -- so a preview at its default scale is the
preview it was before this module existed.

The slider itself is exempt: at 10 % the thing that brings the preview back
must still be there to grab, and double-clicking its value returns to 100 %.
``Ctrl+Alt+0`` resets every preview (:func:`reset_all_preview_scales`).

A WINDOW OPENED FROM THE PREVIEW IS NOT THE PREVIEW (item 522). A dialog
parented to the panel is a window of its own, yet Qt cascades the panel's
scaled sheet into it, and the walk found it among the panel's children: at
150 % a settings dialog's buttons were 59 px tall, not the 40 px of every
other dialog. Such a window and everything in it are kept at 100 %, and the
window's own sheet re-states, at 100 %, the sizes it would otherwise inherit
scaled -- set as it is polished, so before it is first laid out.

Each preview remembers its own scale, under ``prefs/preview_scale/<name>``.
"""
from __future__ import annotations

import logging
import re
import weakref
from typing import Callable, List, Optional

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtWidgets import (QAbstractButton, QFormLayout, QGridLayout,
                               QHBoxLayout, QLabel, QLayout, QSlider,
                               QWidget)

LOG = logging.getLogger("spacr.qt.preview_scale")

PREVIEW_SCALE_MIN = 0.10
PREVIEW_SCALE_MAX = 2.00
DEFAULT_PREVIEW_SCALE = 1.0

_KEY = "prefs/preview_scale/{name}"

#: Qt's "no maximum", which is never scaled.
_QWIDGETSIZE_MAX = 16777215

#: Style-sheet properties whose pixel values follow the scale. Border widths
#: are deliberately absent: a hairline that rounds to zero disappears.
SCALED_QSS_PROPERTIES = frozenset({
    "font-size", "font",
    "padding", "padding-top", "padding-right", "padding-bottom",
    "padding-left",
    "margin", "margin-top", "margin-right", "margin-bottom", "margin-left",
    "min-width", "min-height", "max-width", "max-height",
    "width", "height", "spacing", "border-radius",
    "border-top-left-radius", "border-top-right-radius",
    "border-bottom-left-radius", "border-bottom-right-radius",
})

PREVIEW_SCALE_TIP = (
    "Scale this preview's controls, text and pictures, from 10 % to 200 %, "
    "without changing the rest of spaCR. It applies straight away and is "
    "remembered for this preview. Double-click the percentage to go back "
    "to 100 %; Ctrl+Alt+0 resets every scale.")

_EXEMPT = "spacrPreviewScaleExempt"
_P_SHEET_BASE = "spacrPsSheetBase"
_P_SHEET_SET = "spacrPsSheetSet"
_P_MIN_BASE = "spacrPsMinBase"
_P_MIN_SET = "spacrPsMinSet"
_P_MAX_BASE = "spacrPsMaxBase"
_P_MAX_SET = "spacrPsMaxSet"
_P_ICON_BASE = "spacrPsIconBase"
_P_ICON_SET = "spacrPsIconSet"
_P_MARGIN_BASE = "spacrPsMarginBase"
_P_MARGIN_SET = "spacrPsMarginSet"
_P_SPACING_BASE = "spacrPsSpacingBase"
_P_SPACING_SET = "spacrPsSpacingSet"

_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_RULE = re.compile(r"([^{}]+)\{([^{}]*)\}")
_PX = re.compile(r"(-?\d+(?:\.\d+)?)px")

_SCALERS: "weakref.WeakSet[PreviewScaler]" = weakref.WeakSet()


def clamp_preview_scale(value) -> float:
    """``value`` as a float inside the slider's bounds (1.0 if unreadable).

    :param value: anything a stored preference may hold.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return DEFAULT_PREVIEW_SCALE
    return max(PREVIEW_SCALE_MIN, min(PREVIEW_SCALE_MAX, value))


def get_preview_scale(name: str) -> float:
    """The saved scale of the preview called ``name``.

    :param name: the preview's key, e.g. ``"mask"``.
    """
    from ..preferences import _settings

    try:
        raw = _settings().value(_KEY.format(name=name),
                                DEFAULT_PREVIEW_SCALE)
    except Exception:                                        # noqa: BLE001
        raw = DEFAULT_PREVIEW_SCALE
    return clamp_preview_scale(raw)


def set_preview_scale(name: str, scale: float) -> None:
    """Remember ``scale`` for the preview called ``name``.

    :param name: the preview's key, e.g. ``"mask"``.
    :param scale: the factor, 1.0 = 100 %; clamped to 10-200 %.
    """
    from ..preferences import _settings

    _settings().setValue(_KEY.format(name=name), clamp_preview_scale(scale))


def _scaled_px(match, factor: float, prop: str = "") -> str:
    """One ``Npx`` scaled, never rounding a non-zero size to nothing.

    A radius rounds down, as in :func:`spacr.qt.gui_scale._scaled_px`, so a
    circle's radius never outgrows half of its rounded side.
    """
    from ..gui_scale import _scaled_px as scaled_px

    return scaled_px(match, factor, prop)


def scale_qss(text: str, factor: float, *, sizes_only: bool = False) -> str:
    """Scale the pixel sizes in a style sheet by ``factor``.

    :param text: the QSS.
    :param factor: the scale; 1.0 returns ``text`` unchanged (unless
        ``sizes_only``).
    :param sizes_only: keep only the declarations that were scaled, in rules
        that had any. That is what a preview's own sheet is built from: the
        sizes of every rule it inherits, re-stated at its scale, and nothing
        about colour -- so a theme change needs no rebuild of it.
    :returns: the scaled QSS.
    """
    text = str(text or "")
    if not sizes_only and abs(factor - 1.0) < 1e-9:
        return text
    text = _COMMENT.sub("", text)
    out: List[str] = []
    for match in _RULE.finditer(text):
        selector, body = match.group(1).strip(), match.group(2)
        kept = []
        changed = False
        for declaration in body.split(";"):
            if ":" not in declaration:
                if declaration.strip() and not sizes_only:
                    kept.append(declaration.strip())
                continue
            name, value = declaration.split(":", 1)
            prop = name.strip().lower()
            if prop in SCALED_QSS_PROPERTIES and "px" in value:
                value = _PX.sub(lambda m: _scaled_px(m, factor, prop), value)
                changed = True
                kept.append(f"{name.strip()}: {value.strip()}")
            elif not sizes_only:
                kept.append(f"{name.strip()}: {value.strip()}")
        if sizes_only and not changed:
            continue
        if not selector:
            out.append("; ".join(kept))
        else:
            out.append(f"{selector} {{ {'; '.join(kept)} }}")
    if not out and not _RULE.search(text) and not sizes_only:
        return _scale_bare_declarations(text, factor)
    return "\n".join(out)


def _scale_bare_declarations(text: str, factor: float) -> str:
    """Scale bare declarations such as a font size and a text color."""
    parts = []
    for declaration in text.split(";"):
        if ":" not in declaration:
            if declaration.strip():
                parts.append(declaration.strip())
            continue
        name, value = declaration.split(":", 1)
        prop = name.strip().lower()
        if prop in SCALED_QSS_PROPERTIES:
            value = _PX.sub(lambda m: _scaled_px(m, factor, prop), value)
        parts.append(f"{name.strip()}: {value.strip()}")
    return "; ".join(parts)


def _pair(text) -> Optional[tuple]:
    """``"w,h"`` from a property back into ints."""
    if text is None:
        return None
    try:
        return tuple(int(v) for v in str(text).split(","))
    except ValueError:
        return None


def _text(values) -> str:
    """Ints into the ``"a,b"`` form kept on a property."""
    return ",".join(str(int(v)) for v in values)


def _scale_int(value: int, factor: float) -> int:
    """A size scaled, keeping zero at zero and a non-zero size above it."""
    if value <= 0:
        return value
    return max(1, int(round(value * factor)))


def _base_of(obj, base_key: str, set_key: str, current) -> tuple:
    """The value the code asked for: remembered, unless it was set again."""
    base = _pair(obj.property(base_key))
    applied = _pair(obj.property(set_key))
    current = tuple(current)
    if base is None or (applied is not None and current != applied):
        base = current
        obj.setProperty(base_key, _text(base))
    return base


def _forget(obj, *keys) -> None:
    """Drop the remembered bases, so a widget at 100 % carries nothing."""
    for key in keys:
        if obj.property(key) is not None:
            obj.setProperty(key, None)


class PreviewScaler(QObject):
    """Scales one preview's subtree, and keeps it scaled as it grows.

    :param root: the preview panel.
    :param name: the key its scale is saved under.
    :param scale: the starting scale; the saved one by default.
    """

    def __init__(self, root: QWidget, name: str,
                 scale: Optional[float] = None):
        """Remember the preview and apply its saved scale once it is built."""
        super().__init__(root)
        self._root = root
        self._name = str(name)
        self._scale = clamp_preview_scale(
            get_preview_scale(self._name) if scale is None else scale)
        self._hooks: List[Callable[[float], None]] = []
        self._applying = False
        self._root_sheet_key = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self.apply)
        root.installEventFilter(self)
        _SCALERS.add(self)
        if abs(self._scale - 1.0) > 1e-9:
            QTimer.singleShot(0, self.apply)

    @property
    def name(self) -> str:
        """The key this preview's scale is saved under."""
        return self._name

    def scale(self) -> float:
        """The preview's current scale."""
        return self._scale

    def add_hook(self, hook: Callable[[float], None]) -> None:
        """Call ``hook(scale)`` after every change, and once now if scaled.

        For what a panel draws itself -- a thumbnail size, a figure's dpi --
        which no style sheet or size constraint reaches.

        :param hook: called with the new scale; an exception it raises is
            logged, never passed on.
        """
        self._hooks.append(hook)
        if abs(self._scale - 1.0) > 1e-9:
            self._run_hook(hook)

    def _run_hook(self, hook) -> None:
        """Run one hook, never letting it take the preview down."""
        try:
            hook(self._scale)
        except Exception:                                    # noqa: BLE001
            LOG.debug("a preview scale hook failed", exc_info=True)

    def set_scale(self, scale: float, *, persist: bool = True) -> float:
        """Scale the preview to ``scale`` now, and remember it.

        :param scale: the factor, 1.0 = 100 %.
        :param persist: save it for this preview's next opening.
        :returns: the scale applied, after clamping.
        """
        scale = clamp_preview_scale(scale)
        changed = abs(scale - self._scale) > 1e-9
        self._scale = scale
        if persist:
            try:
                set_preview_scale(self._name, scale)
            except Exception:                                # noqa: BLE001
                LOG.debug("could not save the preview scale", exc_info=True)
        self.apply()
        if changed:
            for hook in list(self._hooks):
                self._run_hook(hook)
        return scale

    def refresh(self) -> None:
        """Rebuild after the application sheet changed (Zoom, theme)."""
        self._root_sheet_key = None
        self.apply()

    def eventFilter(self, watched, event):             # noqa: N802
        """Scale what joins the preview later, and sheets set on it later.

        Read through ``getattr``: a garbage collection that breaks a cycle
        through this object clears its attributes while Qt still delivers
        it events, and a filter that raised then would put an error in the
        event loop for a preview that is already going away.

        :param watched: the widget the event is for.
        :param event: the event; never consumed.
        """
        scale = getattr(self, "_scale", 1.0)
        timer = getattr(self, "_timer", None)
        if (timer is not None and not getattr(self, "_applying", True)
                and abs(scale - 1.0) > 1e-9):
            kind = event.type()
            if kind == QEvent.ChildPolished:
                self._window_polished(event.child(), scale)
            elif kind in (QEvent.ChildAdded, QEvent.StyleChange,
                        QEvent.Show):
                try:
                    timer.start()
                except RuntimeError:
                    pass
        return False

    def _exempt(self, widget) -> bool:
        """Is ``widget`` the scale control, or inside it?"""
        while widget is not None and widget is not self._root:
            try:
                if widget.property(_EXEMPT):
                    return True
                widget = widget.parentWidget()
            except RuntimeError:
                return False
        return False

    def _window_of(self, widget):
        """The outermost window between ``widget`` and the preview, if any.

        A dialog parented to the panel, or to anything in it, is its own
        window: it and its contents are not the preview's to scale.
        """
        window = None
        while widget is not None and widget is not self._root:
            try:
                if widget.isWindow():
                    window = widget
                widget = widget.parentWidget()
            except RuntimeError:
                return window
        return window

    def _window_polished(self, child, scale: float) -> None:
        """Re-state a new window's sizes before it is first laid out.

        Its parent reports it polished from inside ``show()``, before the
        window takes its size; waiting for the next :meth:`apply` would lay
        it out once at the preview's scale.
        """
        try:
            if (not isinstance(child, QWidget) or not child.isWindow()
                    or self._window_of(child) is not child):
                return
        except RuntimeError:
            return
        self._applying = True
        try:
            self._unscale_window(child, scale)
            child.installEventFilter(self)
        except RuntimeError:
            pass
        finally:
            self._applying = False

    def _unscale_window(self, window, factor: float) -> None:
        """Give a window under the preview the sizes of every other window.

        Its own sheet is the preview's cascade re-stated at 100 %, then the
        sheet the window set on itself, which still wins where they differ.
        At 100 % the window's own sheet is given back untouched.
        """
        current = window.styleSheet() or ""
        base = window.property(_P_SHEET_BASE)
        applied = window.property(_P_SHEET_SET)
        if base is None or (applied is not None and current != applied):
            base = current
        if abs(factor - 1.0) < 1e-9:
            if applied is not None and current != base:
                window.setStyleSheet(base)
            _forget(window, _P_SHEET_BASE, _P_SHEET_SET)
            return
        chain = []
        widget = window.parentWidget()
        while widget is not None:
            chain.append(widget)
            if widget is self._root:
                break
            widget = widget.parentWidget()
        parts = [scale_qss(self._inherited_sheet(), 1.0, sizes_only=True)]
        for widget in reversed(chain):
            own = widget.property(_P_SHEET_BASE)
            if own is None:
                own = widget.styleSheet() or ""
            if own:
                parts.append(scale_qss(own, 1.0, sizes_only=True))
        if base:
            parts.append(base)
        wanted = "\n".join(part for part in parts if part)
        window.setProperty(_P_SHEET_BASE, base)
        window.setProperty(_P_SHEET_SET, wanted)
        if current != wanted:
            window.setStyleSheet(wanted)

    def apply(self) -> None:
        """Put the current scale on every widget of the preview."""
        if self._applying:
            return
        try:
            root = self._root
            root.objectName()
        except RuntimeError:
            return
        self._applying = True
        factor = self._scale
        try:
            widgets = [root] + list(root.findChildren(QWidget))
            windows = []
            for widget in widgets:
                if widget is not root and self._exempt(widget):
                    continue
                window = self._window_of(widget)
                own = factor if window is None else 1.0
                if widget is window:
                    windows.append(widget)
                elif widget is not root:
                    self._scale_own_sheet(widget, own)
                self._scale_geometry(widget, own)
                layout = widget.layout()
                if layout is not None:
                    self._scale_layout(layout, own)
                if widget is root:
                    continue
                if (abs(factor - 1.0) > 1e-9
                        and (window is None or widget is window)):
                    widget.installEventFilter(self)
                else:
                    widget.removeEventFilter(self)
            self._scale_root_sheet(factor)
            for window in windows:
                self._unscale_window(window, factor)
            self._pin_the_control_text(factor)
        except RuntimeError:
            pass
        finally:
            self._applying = False

    def _controls(self):
        """The scale controls that sit inside this preview's own subtree."""
        return self._root.findChildren(PreviewScaleControl)

    def _pin_the_control_text(self, factor: float) -> None:
        """Keep the slider itself at its unscaled size whatever the scale.

        The preview's own sheet reaches every child, the slider included, so
        the slider is given a sheet of its own that re-states the sizes it
        inherits at 100 %. A widget's own sheet beats its parent's.
        """
        controls = self._controls()
        if not controls:
            return
        restated = ""
        if abs(factor - 1.0) > 1e-9:
            restated = scale_qss(self._inherited_sheet(), 1.0,
                                 sizes_only=True)
        for control in controls:
            if (control.styleSheet() or "") != restated:
                control.setStyleSheet(restated)

    def _scale_own_sheet(self, widget, factor: float) -> None:
        """Scale the sizes in a widget's own ``setStyleSheet``."""
        current = widget.styleSheet() or ""
        base = widget.property(_P_SHEET_BASE)
        applied = widget.property(_P_SHEET_SET)
        if base is None or (applied is not None and current != applied):
            if not current:
                if base is not None:
                    _forget(widget, _P_SHEET_BASE, _P_SHEET_SET)
                return
            base = current
        if abs(factor - 1.0) < 1e-9:
            if current != base:
                widget.setStyleSheet(base)
            _forget(widget, _P_SHEET_BASE, _P_SHEET_SET)
            return
        widget.setProperty(_P_SHEET_BASE, base)
        scaled = scale_qss(base, factor)
        widget.setProperty(_P_SHEET_SET, scaled)
        if current != scaled:
            widget.setStyleSheet(scaled)

    def _inherited_sheet(self) -> str:
        """Every sheet the preview inherits, outermost first."""
        from PySide6.QtWidgets import QApplication

        from ..theme import (_WINDOW_SHEET_BASE_LEN, _LOCAL_WIDGET_QSS_ATTRIBUTE,
                             window_stylesheet)

        chain = []
        widget = self._root.parentWidget()
        while widget is not None:
            text = widget.styleSheet() or ""
            base_len = widget.property(_WINDOW_SHEET_BASE_LEN)
            if isinstance(base_len, int) and 0 < base_len <= len(text):
                text = text[base_len:]
            if text:
                chain.append(text)
            widget = widget.parentWidget()
        app = QApplication.instance()
        window_sheet = window_stylesheet(app) or (
            app.styleSheet() if app is not None else "")
        suffix = ""
        for ancestor in _ancestors(self._root):
            suffix = getattr(ancestor, _LOCAL_WIDGET_QSS_ATTRIBUTE, "") or suffix
        return "\n".join([window_sheet or ""] + list(reversed(chain))
                         + ([suffix] if suffix else []))

    def _scale_root_sheet(self, factor: float) -> None:
        """Re-state the inherited sizes at this scale, on the preview itself."""
        root = self._root
        current = root.styleSheet() or ""
        base = root.property(_P_SHEET_BASE)
        applied = root.property(_P_SHEET_SET)
        if base is None or (applied is not None and current != applied):
            base = current
        if abs(factor - 1.0) < 1e-9:
            if root.property(_P_SHEET_SET) is not None and current != base:
                root.setStyleSheet(base)
                if not base:
                    for widget in [root] + root.findChildren(QWidget):
                        style = widget.style()
                        style.unpolish(widget)
                        style.polish(widget)
                        QWidget.updateGeometry(widget)
                        QWidget.update(widget)
            _forget(root, _P_SHEET_BASE, _P_SHEET_SET)
            self._root_sheet_key = None
            return
        inherited = self._inherited_sheet()
        key = (hash(inherited), factor, base)
        if key == self._root_sheet_key and current == applied:
            return
        scaled = (scale_qss(inherited, factor, sizes_only=True) + "\n"
                  + scale_qss(base, factor))
        root.setProperty(_P_SHEET_BASE, base)
        root.setProperty(_P_SHEET_SET, scaled)
        self._root_sheet_key = key
        if current != scaled:
            root.setStyleSheet(scaled)

    def _scale_geometry(self, widget, factor: float) -> None:
        """Scale the minimum, maximum and icon sizes the code set."""
        size = widget.minimumSize()
        current = (size.width(), size.height())
        if current != (0, 0) or widget.property(_P_MIN_BASE) is not None:
            base = _base_of(widget, _P_MIN_BASE, _P_MIN_SET, current)
            if abs(factor - 1.0) < 1e-9:
                if current != base:
                    widget.setMinimumSize(*base)
                _forget(widget, _P_MIN_BASE, _P_MIN_SET)
            else:
                wanted = tuple(_scale_int(v, factor) for v in base)
                widget.setProperty(_P_MIN_SET, _text(wanted))
                if current != wanted:
                    widget.setMinimumSize(*wanted)
        size = widget.maximumSize()
        current = (size.width(), size.height())
        if (current != (_QWIDGETSIZE_MAX, _QWIDGETSIZE_MAX)
                or widget.property(_P_MAX_BASE) is not None):
            base = _base_of(widget, _P_MAX_BASE, _P_MAX_SET, current)
            if abs(factor - 1.0) < 1e-9:
                if current != base:
                    widget.setMaximumSize(*base)
                _forget(widget, _P_MAX_BASE, _P_MAX_SET)
            else:
                wanted = tuple(v if v >= _QWIDGETSIZE_MAX
                               else _scale_int(v, factor) for v in base)
                widget.setProperty(_P_MAX_SET, _text(wanted))
                if current != wanted:
                    widget.setMaximumSize(*wanted)
        if isinstance(widget, QAbstractButton) and not widget.icon().isNull():
            size = widget.iconSize()
            current = (size.width(), size.height())
            base = _base_of(widget, _P_ICON_BASE, _P_ICON_SET, current)
            if abs(factor - 1.0) < 1e-9:
                if current != base:
                    widget.setIconSize(_qsize(base))
                _forget(widget, _P_ICON_BASE, _P_ICON_SET)
            else:
                wanted = tuple(_scale_int(v, factor) for v in base)
                widget.setProperty(_P_ICON_SET, _text(wanted))
                if current != wanted:
                    widget.setIconSize(_qsize(wanted))

    def _scale_layout(self, layout: QLayout, factor: float) -> None:
        """Scale a layout's margins and spacing, and its nested layouts'."""
        margins = layout.contentsMargins()
        current = (margins.left(), margins.top(), margins.right(),
                   margins.bottom())
        if current != (0, 0, 0, 0) or layout.property(_P_MARGIN_BASE):
            base = _base_of(layout, _P_MARGIN_BASE, _P_MARGIN_SET, current)
            if abs(factor - 1.0) < 1e-9:
                if current != base:
                    layout.setContentsMargins(*base)
                _forget(layout, _P_MARGIN_BASE, _P_MARGIN_SET)
            else:
                wanted = tuple(_scale_int(v, factor) for v in base)
                layout.setProperty(_P_MARGIN_SET, _text(wanted))
                if current != wanted:
                    layout.setContentsMargins(*wanted)
        if isinstance(layout, (QGridLayout, QFormLayout)):
            current = (layout.horizontalSpacing(), layout.verticalSpacing())
        else:
            current = (layout.spacing(), layout.spacing())
        if (min(current) >= 0 and current != (0, 0)) or layout.property(
                _P_SPACING_BASE):
            base = _base_of(layout, _P_SPACING_BASE, _P_SPACING_SET, current)
            wanted = base if abs(factor - 1.0) < 1e-9 else tuple(
                _scale_int(v, factor) for v in base)
            if abs(factor - 1.0) < 1e-9:
                _forget(layout, _P_SPACING_BASE, _P_SPACING_SET)
            else:
                layout.setProperty(_P_SPACING_SET, _text(wanted))
            if current != wanted and min(wanted) >= 0:
                if isinstance(layout, (QGridLayout, QFormLayout)):
                    layout.setHorizontalSpacing(wanted[0])
                    layout.setVerticalSpacing(wanted[1])
                else:
                    layout.setSpacing(wanted[0])
        for index in range(layout.count()):
            item = layout.itemAt(index)
            child = item.layout() if item is not None else None
            if child is not None:
                self._scale_layout(child, factor)


def _qsize(pair):
    """A ``QSize`` from a pair of ints."""
    from PySide6.QtCore import QSize

    return QSize(int(pair[0]), int(pair[1]))


def _ancestors(widget):
    """``widget``'s parents, nearest first."""
    parent = widget.parentWidget()
    while parent is not None:
        yield parent
        parent = parent.parentWidget()


class PreviewScaleControl(QWidget):
    """The slider and its percentage: compact, and exempt from its own scale.

    :param scaler: the preview's scaler.
    :param parent: owning widget.
    """

    def __init__(self, scaler: PreviewScaler, parent=None):
        """Build the slider at the preview's saved scale."""
        super().__init__(parent)
        from ..i18n import tr

        self._scaler = scaler
        self.setObjectName("PreviewScaleControl")
        self.setProperty(_EXEMPT, True)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setObjectName("PreviewScaleSlider")
        self.slider.setRange(int(round(PREVIEW_SCALE_MIN * 100)),
                             int(round(PREVIEW_SCALE_MAX * 100)))
        self.slider.setSingleStep(5)
        self.slider.setPageStep(25)
        self.slider.setFixedWidth(72)
        self.slider.setValue(int(round(scaler.scale() * 100)))
        self.value = QLabel(self)
        self.value.setObjectName("PreviewScaleValue")
        self.value.setProperty("i18nSkipText", True)
        self._show(self.slider.value())
        tip = tr(PREVIEW_SCALE_TIP)
        for widget in (self, self.slider, self.value):
            widget.setToolTip(tip)
        row.addWidget(self.slider)
        row.addWidget(self.value)
        self.slider.valueChanged.connect(self._show)
        self.slider.sliderReleased.connect(self._commit)
        self.slider.valueChanged.connect(self._commit_unless_dragging)
        self.value.installEventFilter(self)

    def _show(self, value: int) -> None:
        """Show the slider's value as a percentage."""
        self.value.setText(f"{int(value)}%")

    def _commit_unless_dragging(self, _value: int) -> None:
        """Apply keyboard and click steps at once; a drag applies on release."""
        if not self.slider.isSliderDown():
            self._commit()

    def _commit(self) -> None:
        """Scale the preview to what the slider says."""
        self._scaler.set_scale(self.slider.value() / 100.0)

    def set_percent(self, percent: int) -> None:
        """Move the slider (and so the preview) to ``percent``.

        :param percent: the scale in percent, 10 to 200.
        """
        self.slider.setValue(int(percent))
        self._commit()

    def eventFilter(self, watched, event):             # noqa: N802
        """Double-clicking the percentage goes back to 100 %.

        :param watched: the widget the event is for.
        :param event: the event; a double-click on the value is consumed.
        """
        if (watched is getattr(self, "value", None)
                and event.type() == QEvent.MouseButtonDblClick):
            self.set_percent(100)
            return True
        return False


def _card_of(panel):
    """The preview card around ``panel``: the nearest ancestor with a title row."""
    for ancestor in _ancestors(panel):
        if callable(getattr(ancestor, "add_title_action", None)):
            return ancestor
    return None


class _PlacesTheControl(QObject):
    """Puts the control in its row the first time the panel is shown.

    WHY AT THE SHOW. A preview's toolbar row is already as full as the
    narrowest laptop allows, and a slider added to it raises the whole
    column's minimum width -- measured, the settings column lost 68 px at
    100 %, which is exactly what "100 % looks as it does today" forbids. The
    preview card's title row, beside Refresh, has the room; but a panel is
    built before it is put in its card, so where the control goes can only
    be decided once the panel is on screen.
    """

    def __init__(self, panel, control, layout, index, stretch_before,
                 prefer_card):
        """Remember where the control could go, and wait for the show."""
        super().__init__(panel)
        self._panel = panel
        self._control = control
        self._layout = layout
        self._index = index
        self._stretch_before = stretch_before
        self._prefer_card = prefer_card
        panel.installEventFilter(self)

    def eventFilter(self, watched, event):             # noqa: N802
        """Place the control on the panel's first show.

        :param watched: the widget the event is for.
        :param event: the event; never consumed.
        """
        panel = getattr(self, "_panel", None)
        if panel is not None and watched is panel and (
                event.type() == QEvent.Show):
            panel.removeEventFilter(self)
            self.place()
        return False

    def place(self) -> str:
        """Put the control in the card's title row, else show it in its row.

        It already sits, hidden, in the row it was installed into -- a
        hidden widget takes no room and adds nothing to the row's minimum --
        so without a card showing it is all there is to do.

        :returns: ``"card"``, ``"row"`` or ``""`` for where it went.
        """
        control = self._control
        card = _card_of(self._panel) if self._prefer_card else None
        where = ""
        if card is not None:
            if self._layout is not None:
                self._layout.removeWidget(control)
            row = getattr(card, "_title_row", None)
            refresh = getattr(card, "_refresh_button", None)
            if row is not None and refresh is not None and row.indexOf(
                    refresh) >= 0:
                control.setParent(card)
                row.insertWidget(row.indexOf(refresh), control)
            else:
                card.add_title_action(control)
            where = "card"
        elif self._layout is not None:
            where = "row"
        if where:
            control.show()
            host = card if card is not None else self._panel
            top = host.layout() if host is not None else None
            if top is not None:
                top.activate()
        return where


def install_preview_scale(panel: QWidget, name: str, layout=None, *,
                          index: Optional[int] = None,
                          stretch_before: bool = False,
                          prefer_card: bool = True
                          ) -> PreviewScaleControl:
    """Give ``panel`` its own scale slider, saved under ``name``.

    :param panel: the preview panel whose contents the slider scales.
    :param name: the key its scale is saved under, e.g. ``"mask"``.
    :param layout: the row to put the slider in when the panel is not in a
        preview card; ``None`` leaves it to the caller.
    :param index: where in ``layout`` to insert it; the end by default.
    :param stretch_before: put a stretch in front of it, for a row whose
        last item does not already push it right.
    :param prefer_card: put it in the enclosing card's title row, beside
        Refresh, when there is one (see :class:`_PlacesTheControl`).
    :returns: the control. Its scaler is ``control.scaler`` and on
        ``panel.preview_scaler``; ``control.placer.place()`` places it now.
    """
    scaler = PreviewScaler(panel, name)
    control = PreviewScaleControl(scaler, panel)
    control.hide()
    control.scaler = scaler
    scaler.control = control
    panel.preview_scaler = scaler
    if layout is not None:
        if index is None:
            if stretch_before:
                layout.addStretch(1)
            layout.addWidget(control)
        else:
            layout.insertWidget(index, control)
    control.hide()
    control.placer = _PlacesTheControl(panel, control, layout, index,
                                       stretch_before, prefer_card)
    return control


def refresh_all_preview_scales() -> int:
    """Rebuild every scaled preview; called after Preferences re-styles.

    :returns: how many previews were scaled and so rebuilt.
    """
    count = 0
    for scaler in list(_SCALERS):
        try:
            if abs(scaler.scale() - 1.0) > 1e-9:
                scaler.refresh()
                count += 1
        except RuntimeError:
            continue
    return count


def reset_all_preview_scales() -> int:
    """Put every live preview back to 100 %, and forget the saved scales.

    :returns: how many live previews were reset.
    """
    count = 0
    for scaler in list(_SCALERS):
        try:
            scaler.set_scale(1.0)
            count += 1
        except RuntimeError:
            continue
    try:
        from ..preferences import _settings

        store = _settings()
        real = getattr(store, "_real", store)
        real.beginGroup("prefs/preview_scale")
        keys = list(real.childKeys())
        real.endGroup()
        for key in keys:
            store.setValue(f"prefs/preview_scale/{key}", 1.0)
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not reset the saved preview scales", exc_info=True)
    for scaler in list(_SCALERS):
        control = getattr(scaler, "control", None)
        if control is None:
            continue
        try:
            control.slider.blockSignals(True)
            control.slider.setValue(100)
            control.slider.blockSignals(False)
            control._show(100)
        except RuntimeError:
            continue
    return count


def scale_figure_canvas(canvas, scale: float) -> bool:
    """Scale a matplotlib canvas's text, dots and lines with its preview.

    A figure draws in points, so its dpi is what makes a 9 pt label take
    more or fewer pixels. Scaling the dpi the canvas was built with -- and
    keeping the widget's size -- scales everything the figure draws, at the
    cost of nothing but a redraw. It composes with the GUI scale:
    :func:`spacr.qt.gui_scale.apply_canvas_dpi` draws base dpi x GUI scale
    x this preview's scale.

    :param canvas: a ``FigureCanvasQTAgg``.
    :param scale: the preview's scale.
    :returns: ``True`` if the canvas was rescaled.
    """
    if getattr(canvas, "figure", None) is None:
        return False
    canvas._spacr_preview_scale = float(scale)
    from ..gui_scale import apply_canvas_dpi

    return apply_canvas_dpi(canvas)
