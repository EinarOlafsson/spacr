"""Compact RGB channel controls for image-view settings.

The widget presents R, G, and B as independent toggles while preserving the
canonical comma-separated value used by existing settings files. This keeps
the active channels visible without accepting ambiguous free-form spellings.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QWidget

#: The three channels, in the order they are stored and displayed.
CHANNELS = ("r", "g", "b")

#: What each one is called on screen.
LABELS = {"r": "R", "g": "G", "b": "B"}

#: The colour each toggle is tinted, so the control says which channel it
#: means without reading the letter. Muted rather than saturated: this is a
#: label, not a mark on a figure.
TINTS = {"r": "#C0504D", "g": "#4F8162", "b": "#4472C4"}


def parse(value: Any) -> tuple:
    """Return the recognized channels in canonical RGB order.

    Parameters
    ----------
    value : Any
        A comma-separated string, an iterable of names, or an empty value.

    Returns
    -------
    tuple of str
        Recognized channel names in ``('r', 'g', 'b')`` order. Unknown names
        are omitted rather than inferred.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        parts: Iterable = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        parts = value
    else:
        return ()
    wanted = {str(part).strip().lower() for part in parts}
    return tuple(name for name in CHANNELS if name in wanted)


def to_text(names: Iterable[str]) -> str:
    """Serialize channel names as a canonical comma-separated string."""
    chosen = {str(name).strip().lower() for name in names}
    return ",".join(name for name in CHANNELS if name in chosen)


class ChannelPicker(QWidget):
    """Select RGB channels with three checkboxes.

    The public value is a canonical comma-separated string such as
    ``"r,g,b"``. Clearing every channel can be allowed for normalization
    controls or refused for displays that must retain a visible channel.
    """

    changed = Signal(str)

    def __init__(self, value: Any = "", parent: Optional[QWidget] = None,
                 *, allow_none: bool = True):
        """Initialize the picker.

        Parameters
        ----------
        value : Any, default=''
            Initial channel selection.
        parent : QWidget, optional
            Parent widget.
        allow_none : bool, default=True
            Allow every channel to be cleared. Image displays normally set
            this to ``False``; normalization controls may use an empty value
            to mean that no channel is normalized.
        """
        super().__init__(parent)
        self._allow_none = bool(allow_none)
        self._boxes = {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        chosen = set(parse(value))
        for name in CHANNELS:
            box = QCheckBox(LABELS[name], self)
            box.setChecked(name in chosen)
            box.setStyleSheet(f"QCheckBox {{ color: {TINTS[name]}; }}")
            box.setToolTip(f"Show the {LABELS[name]} channel.")
            box.toggled.connect(self._on_toggled)
            self._boxes[name] = box
            layout.addWidget(box)
        layout.addStretch(1)

    def _on_toggled(self, _checked: bool) -> None:
        if not self._allow_none and not self.value():
            # PUT THE LAST ONE BACK rather than let the picture go blank.
            # Blocked so this correction does not re-enter and does not
            # announce a value the user never chose.
            box = self.sender()
            if isinstance(box, QCheckBox):
                box.blockSignals(True)
                box.setChecked(True)
                box.blockSignals(False)
                return
        self.changed.emit(self.value())

    def value(self) -> str:
        """Return the selected channels in canonical stored form."""
        return to_text(name for name, box in self._boxes.items()
                       if box.isChecked())

    def set_value(self, value: Any) -> None:
        """Apply a selection and emit one consolidated change signal."""
        chosen = set(parse(value))
        for name, box in self._boxes.items():
            box.blockSignals(True)
            box.setChecked(name in chosen)
            box.blockSignals(False)
        self.changed.emit(self.value())

    # The panel reads editors through duck-typed accessors; these are the two
    # it looks for, so this widget drops into `_editor` without a special case.
    text = value

    def setText(self, value: Any) -> None:      # noqa: N802 - Qt naming
        self.set_value(value)
