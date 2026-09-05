"""Toggle — QCheckBox styled as an iOS-style switch."""
from __future__ import annotations

from PySide6.QtCore import Property, QPropertyAnimation, QRect, QSize, Qt
from PySide6.QtGui import QBrush, QColor, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QCheckBox

from ..theme import active_palette


class Toggle(QCheckBox):
    """A compact switch that can be clicked or dragged between states."""

    def __init__(self, text: str = "", parent=None):
        """Initialize the switch with an optional trailing label.

        :param text: the label drawn after the switch. Empty leaves the
            switch alone, which is what a settings row wants -- the caption
            beside it is the form's, not the control's.
        :param parent: parent widget.
        """
        super().__init__(text, parent)
        # Approximately 75% of the original 40 x 22 px switch.
        # Leave two physical pixels before the track: a track starting at x=0
        # clips half of its antialiased 1.5 px outline.
        self._track_x = 2
        self._track_w = 30
        self._track_h = 17
        self._knob_d = 12
        self._label_gap = 9
        self._knob_pos = float(self._minimum_knob_x())
        self._mouse_pressed = False
        self._dragging = False
        self._press_x = 0.0
        self._anim = QPropertyAnimation(self, b"knobPos", self)
        self._anim.setDuration(140)
        self.stateChanged.connect(self._start_anim)
        self.setMinimumHeight(self._track_h + 2)

    def sizeHint(self) -> "QSize":
        """Return the default checkbox hint widened to fit the switch track."""
        base = super().sizeHint()
        base.setWidth(
            self._track_x + self._track_w + self._label_gap + base.width())
        return base

    def _minimum_knob_x(self) -> int:
        """Return the knob's left edge in the unchecked position."""
        return self._track_x + (self._track_h - self._knob_d) // 2

    def _maximum_knob_x(self) -> int:
        """Return the knob's left edge in the checked position."""
        inset = (self._track_h - self._knob_d) // 2
        return self._track_x + self._track_w - self._knob_d - inset

    # Custom paint — QCheckBox default indicator is hidden via QSS
    # (we override paintEvent so we don't render it at all).
    def paintEvent(self, event):
        """Paint the switch track, knob, and (optional) trailing label."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        palette = active_palette()
        # Track
        checked = self.isChecked()
        on_color = palette.get("button_accent", palette["accent"])
        off_color = palette.get(
            "fg_dim", palette.get("border", palette["fg"]))
        state_color = QColor(
            on_color if checked else off_color)
        if not self.isEnabled():
            state_color.setAlpha(110)
        on_fill = palette.get("accent_soft", palette["accent"])
        track_fill = QColor(
            on_fill if checked else palette["surface_alt"])
        if not self.isEnabled():
            track_fill.setAlpha(90)
        painter.setBrush(QBrush(track_fill))
        painter.setPen(QPen(state_color, 1.5))
        track_rect = QRect(self._track_x,
                           (self.height() - self._track_h) // 2,
                            self._track_w, self._track_h)
        painter.drawRoundedRect(track_rect, self._track_h // 2, self._track_h // 2)
        # Knob
        knob_x = int(self._knob_pos)
        knob_y = (self.height() - self._knob_d) // 2
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(state_color))
        painter.drawEllipse(QRect(knob_x, knob_y, self._knob_d, self._knob_d))
        # Label
        if self.text():
            painter.setPen(QColor(palette["fg"]))
            painter.drawText(
                self._track_x + self._track_w + self._label_gap,
                (self.height() + painter.fontMetrics().ascent()) // 2 - 2,
                self.text(),
            )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Begin a click or drag without delegating a second toggle to Qt."""
        if event.button() != Qt.LeftButton or not self.isEnabled():
            super().mousePressEvent(event)
            return
        self._anim.stop()
        self._mouse_pressed = True
        self._dragging = False
        self._press_x = float(event.position().x())
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Move the knob with the pointer while the left button is held."""
        if not self._mouse_pressed or not (event.buttons() & Qt.LeftButton):
            super().mouseMoveEvent(event)
            return

        x = float(event.position().x())
        if abs(x - self._press_x) >= 3.0:
            self._dragging = True
        if self._dragging:
            left = x - self._knob_d / 2.0
            left = max(self._minimum_knob_x(),
                       min(self._maximum_knob_x(), left))
            self._set_knob_pos(left)
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Toggle on a tap, or select the side where a drag was released."""
        if event.button() != Qt.LeftButton or not self._mouse_pressed:
            super().mouseReleaseEvent(event)
            return

        self._mouse_pressed = False
        if self._dragging:
            target = (
                self._knob_pos + self._knob_d / 2.0
                >= self._track_x + self._track_w / 2.0
            )
        else:
            target = not self.isChecked()
        self._dragging = False

        if target == self.isChecked():
            self._start_anim(self.checkState())
        else:
            self.setChecked(target)
        self.clicked.emit()
        event.accept()

    def _start_anim(self, _state):
        """Slide the knob to whichever end the new state calls for.

        A toggle that is not on screen jumps rather than animating: there is
        nothing to see, and an animation running for a hidden widget costs
        frames for nobody.

        :param _state: the new check state; the switch is re-read, so it is not
            used.
        """
        end_x = float(
            self._maximum_knob_x()
            if self.isChecked()
            else self._minimum_knob_x()
        )
        if not self.isVisible():
            self._anim.stop()
            self._set_knob_pos(end_x)
            return
        self._anim.stop()
        self._anim.setStartValue(self._knob_pos)
        self._anim.setEndValue(end_x)
        self._anim.start()

    def _get_knob_pos(self) -> float:
        """Return the knob's current horizontal position.

        :returns: the position in pixels. This is the property the animation
            drives.
        """
        return self._knob_pos

    def _set_knob_pos(self, v: float) -> None:
        """Move the knob and repaint.

        :param v: the new horizontal position in pixels.
        """
        self._knob_pos = float(v)
        self.update()

    knobPos = Property(float, _get_knob_pos, _set_knob_pos)
