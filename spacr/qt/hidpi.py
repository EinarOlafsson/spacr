"""Pictures rendered for the panel they land on.

Why a module for two lines of arithmetic
----------------------------------------
``QPixmap.scaled(w, h, ...)`` takes DEVICE pixels but every caller in a GUI
means LOGICAL ones -- the size the picture should occupy on screen. On a
display with a device pixel ratio of 2, which is every retina Mac and plenty
of Windows and Linux machines, Qt then stretches that logical bitmap across
twice as many real pixels in each direction: a 200 px render blown up to
400 px, from a 3334 px source. The picture is not low-resolution; the
scaling threw the resolution away.

The whole fix is to render at device pixels and then say so::

    ratio = widget.devicePixelRatioF()
    picture = source.scaled(round(px * ratio), round(px * ratio), ...)
    picture.setDevicePixelRatio(ratio)

**Missing the second line is worse than missing both.** The picture comes
out correct in pixels and twice the intended SIZE, because Qt has no way to
know those pixels are dense ones.

A rule applied by hand at each site holds until the next site is written, so
what ships is :func:`scaled_for` -- source, widget, logical size -- and every
call site asking it. The next picture is then right without its author
knowing this problem exists.

At a device pixel ratio of 1 :func:`scaled_for` is byte-for-byte what
``.scaled()`` did before: the same pixel dimensions, and a device pixel ratio
of 1 is what a pixmap already carries. No ordinary display changes.

Reading a picture back
----------------------
``QPixmap.width()`` counts device pixels, so a picture rendered through this
module reports twice its on-screen width on a 2x display. Anything that
compares a pixmap against widget coordinates -- centring it, mapping a mouse
position onto it, fitting a label round it -- wants :func:`logical_size`,
which is the size it OCCUPIES whatever it was rendered at.

Moving a window between screens
-------------------------------
The ratio belongs to the screen, not the application, and a window can be
dragged from a retina laptop onto an ordinary external monitor. A picture
rendered at 2x and moved to a 1x screen is merely wasteful; the reverse is
blurry. :func:`follow_device_ratio` subscribes a widget to the change Qt
already sends, for the pictures whose source is still in memory and whose
re-render is a single call. Pictures scaled inside ``paintEvent`` need
nothing: they ask for the ratio again on the next frame.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple, Union

from PySide6.QtCore import QEvent, QObject, QSize, Qt
from PySide6.QtGui import QGuiApplication

#: Ratios outside this are a broken or hostile environment variable, not a
#: display. Rendering 40x is an out-of-memory crash, not a sharper logo.
MAX_RATIO = 8.0

SizeLike = Union[int, float, QSize, Tuple[int, int]]

__all__ = [
    "MAX_RATIO",
    "device_ratio",
    "scaled_for",
    "logical_size",
    "follow_device_ratio",
]


def _ratio_of(source: Any) -> float:
    """``devicePixelRatioF``/``devicePixelRatio`` off ``source``, or 0.0.

    0.0 rather than 1.0 so the caller can tell "this object has no ratio"
    from "this object is on an ordinary display" and keep looking.
    """
    if source is None:
        return 0.0
    for name in ("devicePixelRatioF", "devicePixelRatio"):
        getter = getattr(source, name, None)
        if getter is None:
            continue
        try:
            value = float(getter())
        except Exception:                                # noqa: BLE001
            continue
        if value > 0.0 and value == value and value != float("inf"):
            return min(value, MAX_RATIO)
    return 0.0


def device_ratio(target: Any = None) -> float:
    """How many real pixels ``target`` gets per logical pixel.

    Asks the widget, then the screen the widget is on, then the primary
    screen, and answers 1.0 when nothing will say -- which is a plain
    display and the behaviour every call site had before.

    ``target`` may be any object with a ``devicePixelRatio`` accessor: a
    widget, a window, a screen, a paint device, or a stand-in in a test.
    """
    ratio = _ratio_of(target)
    if ratio:
        return ratio
    for name in ("screen", "window"):
        getter = getattr(target, name, None)
        if getter is None:
            continue
        try:
            ratio = _ratio_of(getter())
        except Exception:                                # noqa: BLE001
            continue
        if ratio:
            return ratio
    try:
        app = QGuiApplication.instance()
        if app is not None:
            ratio = _ratio_of(app.primaryScreen())
            if ratio:
                return ratio
    except Exception:                                    # noqa: BLE001
        pass
    return 1.0


def _as_size(width: SizeLike, height: Optional[SizeLike]) -> QSize:
    """``(width, height)``, a ``QSize`` or a 2-tuple, as a ``QSize``."""
    if height is not None:
        return QSize(int(width), int(height))            # type: ignore[arg-type]
    if isinstance(width, QSize):
        return QSize(width)
    if isinstance(width, (tuple, list)) and len(width) == 2:
        return QSize(int(width[0]), int(width[1]))
    side = int(width)                                    # type: ignore[arg-type]
    return QSize(side, side)


def _device_dim(value: int, ratio: float) -> int:
    """One logical edge in device pixels, never rounding a picture away."""
    if ratio == 1.0:
        return int(value)
    scaled = int(round(value * ratio))
    return max(1, scaled) if value >= 1 else scaled


def scaled_for(source: Any, target: Any, width: SizeLike,
               height: Optional[SizeLike] = None, *,
               aspect: Qt.AspectRatioMode = Qt.KeepAspectRatio,
               mode: Qt.TransformationMode = Qt.SmoothTransformation) -> Any:
    """``source`` at ``width`` x ``height`` LOGICAL pixels on ``target``.

    The one call every picture in the application should make. ``source``
    is a ``QPixmap`` or a ``QImage``; ``target`` is the widget the picture
    will be shown on (anything :func:`device_ratio` understands); the size
    is a side length, a ``(w, h)`` pair, a ``QSize``, or two arguments.

    The result is rendered at ``size * ratio`` real pixels and carries that
    ratio, so it lays out at ``size`` and draws at full panel resolution.

    A null or missing source comes back untouched -- a caller that already
    checks ``isNull()`` keeps its own answer, and one that does not is no
    worse off than it was with a bare ``.scaled()``.
    """
    if source is None:
        return source
    is_null = getattr(source, "isNull", None)
    if is_null is not None and is_null():
        return source
    size = _as_size(width, height)
    ratio = device_ratio(target)
    picture = source.scaled(_device_dim(size.width(), ratio),
                            _device_dim(size.height(), ratio),
                            aspect, mode)
    setter = getattr(picture, "setDevicePixelRatio", None)
    if setter is not None and not picture.isNull():
        setter(ratio)
    return picture


def logical_size(picture: Any) -> QSize:
    """The size ``picture`` OCCUPIES, whatever it was rendered at.

    ``QPixmap.width()`` counts device pixels; this counts the widget
    coordinates the picture covers, which is what centring, hit-testing
    and layout are all measured in. A null or missing picture is 0 x 0.
    """
    if picture is None:
        return QSize(0, 0)
    is_null = getattr(picture, "isNull", None)
    if is_null is not None and is_null():
        return QSize(0, 0)
    independent = getattr(picture, "deviceIndependentSize", None)
    if independent is not None:
        try:
            size = independent()
            return QSize(int(round(size.width())), int(round(size.height())))
        except Exception:                                # noqa: BLE001
            pass
    ratio = _ratio_of(picture) or 1.0
    return QSize(int(round(picture.width() / ratio)),
                 int(round(picture.height() / ratio)))


class _RatioWatcher(QObject):
    """Calls back when its widget's device pixel ratio changes.

    Parented to the widget, so it dies with it and never outlives the
    thing it would redraw.

    :param widget: the widget whose ratio is watched, and the QObject parent
        -- which is what the note above means.
    :param redraw: called with no arguments when the ratio actually CHANGES.
        Not on every DevicePixelRatioChange: the old ratio is remembered and
        compared, so dragging a window between two identical monitors
        redraws nothing.
    """

    def __init__(self, widget: Any, redraw: Callable[[], None]) -> None:
        super().__init__(widget)
        self._redraw = redraw
        self._ratio = device_ratio(widget)

    def eventFilter(self, watched: QObject,                # noqa: N802 - Qt
                    event: QEvent) -> bool:
        if event.type() == QEvent.Type.DevicePixelRatioChange:
            ratio = device_ratio(watched)
            if ratio != self._ratio:
                self._ratio = ratio
                try:
                    self._redraw()
                except Exception:                        # noqa: BLE001
                    import logging

                    logging.getLogger(__name__).debug(
                        "redraw after a device pixel ratio change failed",
                        exc_info=True)
        return False

    def ratio(self) -> float:
        """The ratio this watcher last saw."""
        return self._ratio


def follow_device_ratio(widget: Any,
                        redraw: Callable[[], None]) -> Optional[_RatioWatcher]:
    """Re-render ``widget``'s picture when it moves to a different screen.

    Qt sends ``DevicePixelRatioChange`` when a window is dragged from a
    retina laptop onto an ordinary monitor or back. ``redraw`` is the
    widget's own "put the picture on again" call -- it must scale from the
    source it kept, not from what is currently on the label, or each move
    re-scales an already-scaled picture.

    Returns the watcher (for tests to drive) or ``None`` if the widget
    cannot take an event filter.
    """
    try:
        watcher = _RatioWatcher(widget, redraw)
        widget.installEventFilter(watcher)
    except Exception:                                    # noqa: BLE001
        return None
    return watcher
