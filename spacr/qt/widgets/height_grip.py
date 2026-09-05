"""A drag handle that makes ONE widget in a column taller or shorter.

Qt has no vertical-resize handle for "this one widget inside a column", and
the two things that look like one do not fit: a ``QSplitter`` needs two panes
to divide, and ``QSizeGrip`` resizes the WINDOW. Home grew its own for the
release-notes panel; the nested containers of Regression's Measurements tab
need the same affordance, so the class lives here and Home imports it rather
than the tool carrying two of them.

WHAT WAS ADDED WHEN IT MOVED, and why each addition is not decoration:

* an accessible name per instance -- "Resize the release notes" is the wrong
  thing for a screen reader to say about the merge report;
* keyboard resizing, because a pointer-only handle is not an accessible
  height-resize affordance. Up and Down move by one step, Page Up/Down by
  five, Home/End go to the bounds;
* bounds expressed in BASE pixels and re-scaled on demand. At a 200 % font
  scale a ceiling of 400 physical px is half a box, so every bound goes
  through :func:`spacr.qt.preferences.scaled_px` and
  :meth:`HeightGrip.rescale` re-reads it when the font preference changes.
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QWidget

from ..theme import active_palette, make_transparent


class HeightGrip(QWidget):
    """A thin bar that drags the widget above it taller or shorter.

    Drawn as three short lines rather than a plain strip because a strip that
    happens to be draggable is a strip nobody drags. It brightens under the
    pointer and takes focus for the same reason.

    :param target: the widget whose fixed height this drags.
    :param minimum: floor in px at 100 % font scale.
    :param maximum: ceiling in px at 100 % font scale.
    :param parent: parent widget; ownership only.
    :param name: accessible name, said aloud in place of "resize handle".
    """

    #: Emitted with the new height in device px, on release rather than on
    #: every mouse move: this is what gets written to QSettings.
    height_changed = Signal(int)

    #: How tall the grip itself is, in px at 100 % font scale.
    BAR_H = 9

    #: How far one arrow key moves the border, in px at 100 % font scale.
    #: Big enough to be visible in one press, small enough to aim with.
    STEP = 12

    def __init__(self, target: QWidget, minimum: int, maximum: int,
                 parent=None, name: str = "Resize this panel"):
        """Build the grip, bounded so a drag cannot collapse or run away."""
        super().__init__(parent)

        self._target = target
        self._base_min = int(minimum)
        self._base_max = int(maximum)
        self._from_y = None
        self._from_h = 0
        self._hovered = False
        self._min = None
        self._max = None
        #: The font scale the bounds below were computed at. SET BEFORE ANY
        #: QWidget call, because `setCursor` and friends deliver a
        #: `StyleChange` synchronously and `changeEvent` reads this -- which
        #: is how the first draft raised AttributeError out of every Home
        #: test that built the release-notes panel.
        self._scale = 0.0
        #: Whether this grip has ever set the target's height. Until it has,
        #: `rescale` leaves the target alone: the constructor must not pin a
        #: height on a widget whose owner is about to restore a stored one.
        self._sized = False
        #: The height this handle first set, in base px. A dragged border
        #: must never be able to make a panel permanently unreachable, so
        #: there is always a way back to what it opened at.
        self._base_default = 0
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._offer_reset)
        # THE HANDLE IS A TAB STOP. Everything this widget offers is otherwise
        # reachable only with a pointer, and a nested container a keyboard
        # user cannot enlarge is the one that stays too small to read.
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.SizeVerCursor)
        self.setToolTip("Drag to resize, or focus it and use the arrow keys")
        self.setAccessibleName(name)
        make_transparent(self)
        self.rescale()

    # ------------------------------------------------------------------ size
    def rescale(self) -> None:
        """Re-read the font scale and re-apply the bounds.

        Called from the constructor and again after a font-preference change:
        the bounds are stored in base px precisely so that the answer can be
        recomputed rather than baked in at construction time.
        """
        from ..preferences import get_font_scale, scaled_px

        self._scale = get_font_scale()
        self._min = scaled_px(self._base_min)
        self._max = scaled_px(max(self._base_max, self._base_min))
        self.setFixedHeight(scaled_px(self.BAR_H))
        if self._sized:
            self.resize_target(self._target.height())

    def target_height(self) -> int:
        """How tall the widget under this handle currently is, in device px."""
        return int(self._target.height())

    def changeEvent(self, event):               # noqa: N802 - Qt naming
        """Follow a font change rather than keep a height chosen for the old one.

        The layout is recomputed after a font, locale, theme or display
        change; first launch is not the only moment the geometry moves. spaCR
        applies its font scale through the application stylesheet, so what
        arrives here is a style or font change rather than an explicit call --
        which is why this listens for the event instead of waiting to be told.
        """
        if (event.type() in (QEvent.FontChange, QEvent.ApplicationFontChange,
                             QEvent.StyleChange)
                and getattr(self, "_min", None) is not None):
            self._follow_the_font()
        super().changeEvent(event)

    def _follow_the_font(self) -> None:
        """Re-scale the bounds, and the box, by how much the font moved."""
        from ..preferences import get_font_scale, scaled_px

        scale = get_font_scale()
        if scale == self._scale:
            return
        was = self._scale
        # IN BASE PX, taken BEFORE `rescale` moves the clamp: converting after
        # would divide a height by the new scale and keep the old pixels.
        base = (int(round(self.target_height() / max(was, 0.01)))
                if self._sized else 0)
        self.rescale()
        if base:
            self.resize_target(scaled_px(base))

    def bounds(self) -> tuple:
        """The floor and ceiling in device px, as currently scaled."""
        return (self._min, self._max)

    def resize_target(self, height: int) -> int:
        """Set the target's height, clamped. Returns what it became.

        Public because it is the whole behaviour, and a test that drives it
        directly is testing the clamp rather than Qt's event delivery.
        """
        from ..preferences import get_font_scale

        wanted = max(self._min, min(self._max, int(height)))
        self._target.setFixedHeight(wanted)
        if not self._sized:
            self._base_default = int(round(
                wanted / max(get_font_scale(), 0.01)))
        self._sized = True
        return wanted

    def reset(self) -> int:
        """Put the box back to the height it opened at. Returns that height.

        Reached from the handle's own context menu, because that is where a
        user who has dragged one too far is already pointing.
        """
        from ..preferences import scaled_px

        height = self.resize_target(scaled_px(self._base_default)
                                    if self._base_default else self._min)
        self.height_changed.emit(height)
        return height

    def _offer_reset(self, where) -> None:
        """Right-click: one item, and it is the way back."""
        from PySide6.QtWidgets import QMenu

        menu = QMenu(self)
        menu.addAction("Reset height", self.reset)
        menu.exec(self.mapToGlobal(where))

    def nudge(self, delta: int) -> int:
        """Move the border by ``delta`` device px. Returns the new height."""
        height = self.resize_target(self._target.height() + int(delta))
        self.height_changed.emit(height)
        return height

    # --------------------------------------------------------------- drawing
    def paintEvent(self, event):                # noqa: N802 - Qt naming
        """Three short lines, centred, brighter under the pointer or focus."""
        P = active_palette()
        painter = QPainter(self)
        colour = QColor(P["fg"])
        lit = self._hovered or self.hasFocus()
        colour.setAlphaF(0.34 if lit else 0.16)
        painter.setPen(Qt.NoPen)
        painter.setBrush(colour)
        width = max(12, self.width() // 6)
        left = (self.width() - width) // 2
        mid = self.height() // 2
        for offset in (-3, 0, 3):
            painter.drawRect(left, mid + offset, width, 1)

    def enterEvent(self, event):                # noqa: N802 - Qt naming
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):                # noqa: N802 - Qt naming
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def focusInEvent(self, event):              # noqa: N802 - Qt naming
        self.update()
        super().focusInEvent(event)

    def focusOutEvent(self, event):             # noqa: N802 - Qt naming
        self.update()
        super().focusOutEvent(event)

    # ---------------------------------------------------------------- events
    def mousePressEvent(self, event):           # noqa: N802 - Qt naming
        """Remember where the drag started, in GLOBAL coordinates.

        Global, because this widget MOVES while the drag is happening -- it
        sits under the widget being resized, so growing the target by 40 px
        slides the grip 40 px down and a local y would double every step.
        """
        if event.button() != Qt.LeftButton:
            return super().mousePressEvent(event)
        self._from_y = event.globalPosition().y()
        self._from_h = self._target.height()
        event.accept()

    def mouseMoveEvent(self, event):            # noqa: N802 - Qt naming
        if self._from_y is None:
            return super().mouseMoveEvent(event)
        delta = event.globalPosition().y() - self._from_y
        self.resize_target(self._from_h + int(delta))
        event.accept()

    def mouseReleaseEvent(self, event):         # noqa: N802 - Qt naming
        if self._from_y is None:
            return super().mouseReleaseEvent(event)
        self._from_y = None
        self.height_changed.emit(self._target.height())
        event.accept()

    def keyPressEvent(self, event):             # noqa: N802 - Qt naming
        """Arrow keys resize; everything else is passed on.

        Passed on matters as much as handled: swallowing Tab would trap focus
        on the one control whose whole purpose is to be reachable.
        """
        from ..preferences import scaled_px

        step = scaled_px(self.STEP)
        key = event.key()
        if key in (Qt.Key_Down, Qt.Key_Plus, Qt.Key_Equal):
            self.nudge(step)
        elif key in (Qt.Key_Up, Qt.Key_Minus):
            self.nudge(-step)
        elif key == Qt.Key_PageDown:
            self.nudge(step * 5)
        elif key == Qt.Key_PageUp:
            self.nudge(-step * 5)
        elif key == Qt.Key_End:
            self.nudge(self._max)
        elif key == Qt.Key_Home:
            self.nudge(-self._max)
        else:
            return super().keyPressEvent(event)
        event.accept()
