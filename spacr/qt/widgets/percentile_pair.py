"""Two numeric fields for the low and high ends of a percentile window.

A PERCENTILE WINDOW IS TWO NUMBERS, so it is asked for as two numbers. Typed
into one box it is a small parsing problem handed to the user: ``[1, 99]``
and ``[1 99]`` are the same intent and only one of them used to survive the
trip to the renderer, and the user found out by getting a picture that was
not the one they asked for rather than by being told.

The STORED value does not change. :meth:`PercentilePair.value` answers the
``[low, high]`` list every settings file already holds, and
:func:`spacr.picture_settings.percentile_pair` reads both that and every text
spelling that reached disk before this control existed -- so an old settings
CSV opens on the pair it recorded instead of being refused.
"""
from __future__ import annotations

from typing import Any, List, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDoubleSpinBox, QHBoxLayout, QLabel, QWidget

from ...picture_settings import DEFAULT_PERCENTILES, percentile_pair

__all__ = ["PercentilePair"]

#: What each field is called on screen. Short, because the setting's own
#: label already says these are percentiles.
LOW_LABEL = "low %"
HIGH_LABEL = "high %"

#: Decimal places each field carries.
#:
#: PERCENTILES ARE NOT ALWAYS WHOLE. spaCR's own normalisation walks 98,
#: 99, 99.9, 99.99 and 99.999, so a control that rounded to integers could
#: not express the top half of that ladder at all.
#:
#: SIX, NOT THREE, and matching :data:`spacr.qt.screens.make_masks.
#: PERCENTILE_DECIMALS` rather than being a second opinion about the same
#: quantity. Three stops at 99.999, and the reason the ladder does not stop
#: there is arithmetic: on a 2048x2048 field 99.999 still keeps forty pixels,
#: and it is the last four or five -- a cosmic ray, a saturated bead, one hot
#: sensor pixel -- that pin the display range and make every real object look
#: black. 99.9999 clips those four. A control that cannot express the step
#: that fixes the image is a control that cannot fix the image, which is what
#: was reported.
DECIMALS = 6


def _tidy(number: float):
    """``2.0`` back to ``2``, so a whole percentile is stored as one.

    The annotator ships ``[2, 98]`` and every settings file on disk holds
    integers; answering ``[2.0, 98.0]`` would rewrite all of them on the
    first save for no change in meaning.
    """
    value = float(number)
    return int(value) if value == int(value) else value


class PercentilePair(QWidget):
    """The low and high ends of a percentile window, as two spin boxes.

    The two fields constrain each other rather than validating after the
    fact: the high field cannot go below the low one and the low field
    cannot go above the high one, so an inverted window is not a value the
    panel can be left holding.
    """

    changed = Signal(object)

    def __init__(self, value: Any = None, parent: Optional[QWidget] = None):
        """Build the pair.

        :param value: the starting window, in any spelling
            :func:`spacr.picture_settings.percentile_pair` accepts.
        :param parent: the owning widget.
        """
        super().__init__(parent)
        low, high = percentile_pair(value, DEFAULT_PERCENTILES)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._low = self._field(low, LOW_LABEL,
                                "The percentile mapped to black. Raise it to "
                                "crush more background; 0 keeps the dimmest "
                                "pixel.")
        self._high = self._field(high, HIGH_LABEL,
                                 "The percentile mapped to white. Lower it "
                                 "for more contrast at the cost of clipping "
                                 "the brightest pixels; 100 clips none.")
        for caption, field in ((LOW_LABEL, self._low),
                               (HIGH_LABEL, self._high)):
            tag = QLabel(caption, self)
            tag.setObjectName("Muted")
            tag.setToolTip(field.toolTip())
            layout.addWidget(tag)
            layout.addWidget(field)
        layout.addStretch(1)

        # THE ORDER IS ENFORCED BY THE CONTROLS, not checked on the way out.
        # A window whose low end is above its high end has no meaning, and a
        # panel that lets one be entered has to decide later what the user
        # meant.
        self._low.setMaximum(self._high.value())
        self._high.setMinimum(self._low.value())
        self._low.valueChanged.connect(self._on_low)
        self._high.valueChanged.connect(self._on_high)
        # HOVER HELP BELONGS TO THE SETTING'S NAME, never to the box
        # you type in. Built here on the field, it is moved onto the
        # label as the last step, so every panel in the application
        # explains itself the same way.
        from ..screens.settings_model import retarget_field_tooltips
        retarget_field_tooltips(self)

    def _field(self, start: float, name: str, why: str) -> QDoubleSpinBox:
        """One numeric field, ranged to what a percentile can be."""
        spin = QDoubleSpinBox(self)
        spin.setDecimals(DECIMALS)
        spin.setRange(0.0, 100.0)
        # A STEP THAT MATCHES THE NUMBERS. Left at Qt's default of 1.0 a
        # wheel tick on 99.5 lands on 100.5, which the range then clamps --
        # so the control appears to ignore the gesture.
        spin.setSingleStep(0.5)
        spin.setValue(float(start))
        spin.setAccessibleName(name)
        spin.setToolTip(why)
        return spin

    # ------------------------------------------------------------- keeping
    #  the two ends in order

    def _on_low(self, value: float) -> None:
        """Raise the high box's floor to the new low, and announce the window.

        The two boxes bound each other rather than being validated afterwards,
        so an inverted window cannot be typed in the first place.

        :param value: the new low percentile.
        """
        self._high.setMinimum(float(value))
        self.changed.emit(self.value())

    def _on_high(self, value: float) -> None:
        """Lower the low box's ceiling to the new high, and announce the window.

        :param value: the new high percentile.
        """
        self._low.setMaximum(float(value))
        self.changed.emit(self.value())

    # -------------------------------------------------------------- value

    def value(self) -> List[Any]:
        """The window as ``[low, high]``, in the form settings files hold."""
        return [_tidy(self._low.value()), _tidy(self._high.value())]

    def set_value(self, value: Any) -> None:
        """Apply a window and announce it once.

        :param value: any spelling
            :func:`spacr.picture_settings.percentile_pair` accepts, including
            the bracketed text a settings file written before this control
            existed still holds.
        """
        low, high = percentile_pair(value, DEFAULT_PERCENTILES)
        for field in (self._low, self._high):
            field.blockSignals(True)
        # THE BOUNDS ARE OPENED BEFORE THE VALUES ARE SET. Each field's
        # range is pinned to the other's value, so setting a whole new
        # window in place clamps whichever end moves first.
        self._low.setMaximum(100.0)
        self._high.setMinimum(0.0)
        self._low.setValue(float(low))
        self._high.setValue(float(high))
        self._low.setMaximum(float(high))
        self._high.setMinimum(float(low))
        for field in (self._low, self._high):
            field.blockSignals(False)
        self.changed.emit(self.value())

    def low(self) -> QDoubleSpinBox:
        """The field holding the low end."""
        return self._low

    def high(self) -> QDoubleSpinBox:
        """The field holding the high end."""
        return self._high

    # The picture dialog reads unfamiliar editors through `text()`/`setText`,
    # so the pair answers those too rather than needing a special case in
    # every reader.
    def text(self) -> str:
        """The window as ``"low, high"``, for readers that want text."""
        low, high = self.value()
        return f"{low}, {high}"

    def setText(self, value: Any) -> None:      # noqa: N802 - Qt naming
        """Set both percentiles from one settings string.

        Named for QLineEdit's method for the same reason as
        :meth:`ChannelPicker.setText`.

        :param value: the pair, as the settings dict spells it.
        """
        self.set_value(value)
