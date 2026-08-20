"""R, G and B as three toggles -- the control the annotation app taught.

Instruction 188 B, asked for 2026-08-20: "for the channels and normalize
channels i liked the r,g,b system better, instead of the dropdown you made. i
want to you to bring back the r,g,b system, it works great in the annotation
app."

WHAT THE DROPDOWN COST. It offered the eight combinations as eight rows --
"all three", "red only", "red and green" -- so reading which channels were on
meant opening it, and the one thing a user does constantly here (turn one
channel off to see what is under it) took two clicks and a search through a
list whose order means nothing.

WHY NOT THE ANNOTATION APP'S CONTROL EXACTLY. That one is a QLineEdit holding
``r,g,b``, and it is what the maintainer is asking for -- but a text box
accepts ``rgb``, ``R,G,B``, ``red``, and a trailing comma, and instruction
176 ("one channel vocabulary") exists because those spellings reached the run
and meant different things. Three checkboxes ARE the r,g,b system: same
letters, same stored value, same directness, and no spelling to get wrong.

THE STORED VALUE IS UNCHANGED -- a comma string like ``"r,g,b"`` -- so every
settings file already written goes on meaning what it meant, and the crop
layer's `filter_channels_pil` reads exactly what it read before.
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
    """Which channels ``value`` names, in canonical order.

    Accepts what the settings files and the panels actually hold: a comma
    string, a list, ``None`` and ``""``. Anything unrecognised is DROPPED
    rather than guessed -- a channel nobody asked for is worse than one
    missing, because it changes what the picture shows without saying so.
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
    """The stored form: the canonical order, comma separated."""
    chosen = {str(name).strip().lower() for name in names}
    return ",".join(name for name in CHANNELS if name in chosen)


class ChannelPicker(QWidget):
    """Three checkboxes whose value is the comma string spaCR already stores."""

    changed = Signal(str)

    def __init__(self, value: Any = "", parent: Optional[QWidget] = None,
                 *, allow_none: bool = True):
        """
        :param allow_none: whether clearing every box is a legal value.
            `normalize_channels` says "" for "normalise nothing", which is a
            real answer; `channels` showing no channel at all would be a
            blank picture, so it is refused by putting the last one back.
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
        """The stored form, e.g. ``"r,g"``."""
        return to_text(name for name, box in self._boxes.items()
                       if box.isChecked())

    def set_value(self, value: Any) -> None:
        """Show ``value``, without emitting for each box on the way."""
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
