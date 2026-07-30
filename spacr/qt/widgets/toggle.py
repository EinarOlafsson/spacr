"""Toggle — QCheckBox styled as an iOS-style switch."""
from __future__ import annotations

from PySide6.QtCore import Property, QPropertyAnimation, QRect, QSize, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import QCheckBox

from ..theme import active_palette


class Toggle(QCheckBox):
    """A modern toggle switch. Emits `stateChanged` on user interaction."""

    def __init__(self, text: str = "", parent=None):
        """Initialize the switch with an optional trailing label."""
        super().__init__(text, parent)
        # Track geometry
        self._track_w = 40
        self._track_h = 22
        self._knob_d = 16
        self._knob_pos = 3.0
        self._anim = QPropertyAnimation(self, b"knobPos", self)
        self._anim.setDuration(140)
        self.stateChanged.connect(self._start_anim)
        self.setMinimumHeight(self._track_h + 2)

    def sizeHint(self) -> "QSize":
        """Return the default checkbox hint widened to fit the switch track."""
        base = super().sizeHint()
        base.setWidth(self._track_w + 12 + base.width())
        return base

    # Custom paint — QCheckBox default indicator is hidden via QSS
    # (we override paintEvent so we don't render it at all).
    def paintEvent(self, event):
        """Paint the switch track, knob, and (optional) trailing label."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        palette = active_palette()
        # Track
        checked = self.isChecked()
        state_color = QColor(
            palette["button_accent"] if checked else palette["fg_dim"])
        if not self.isEnabled():
            state_color.setAlpha(110)
        track_fill = QColor(
            palette["accent_soft"] if checked else palette["surface_alt"])
        if not self.isEnabled():
            track_fill.setAlpha(90)
        painter.setBrush(QBrush(track_fill))
        painter.setPen(QPen(state_color, 2))
        track_rect = QRect(0, (self.height() - self._track_h) // 2,
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
                self._track_w + 12,
                (self.height() + painter.fontMetrics().ascent()) // 2 - 2,
                self.text(),
            )

    def _start_anim(self, _state):
        pad = (self._track_h - self._knob_d) // 2
        end_x = float(self._track_w - self._knob_d - pad if self.isChecked() else pad)
        if not self.isVisible():
            self._anim.stop()
            self._set_knob_pos(end_x)
            return
        self._anim.stop()
        self._anim.setStartValue(self._knob_pos)
        self._anim.setEndValue(end_x)
        self._anim.start()

    def _get_knob_pos(self) -> float:
        return self._knob_pos

    def _set_knob_pos(self, v: float) -> None:
        self._knob_pos = float(v)
        self.update()

    knobPos = Property(float, _get_knob_pos, _set_knob_pos)
