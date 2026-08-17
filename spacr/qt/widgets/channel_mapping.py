"""Three-field editor for ``png_channel_mapping``: which source channel is
red, which is green, which is blue.

Replaces a text box holding ``{'r': 2, 'g': 1, 'b': 0}``. The dict form is
still what the setting *is* -- this only stops the user having to type a
Python literal to say something as simple as "555 is my red".

Why three labelled fields rather than one ordered list: the list form
(``png_dims``) never said which colour it meant. Position 0 was blue because
of how cv2 interprets an array it is handed, which is not something the
settings panel could show and not something a user could infer. It got read
backwards for eleven days and every crop written in that window has its
nuclear stain in the red channel. A field labelled "R" cannot be read
backwards.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QWidget,
)

#: Highest source channel index the spin boxes offer. Well past any real
#: stack; the cost of a too-high ceiling is nothing, and the cost of a too-low
#: one is a user who cannot enter their own data.
MAX_SOURCE_CHANNEL = 31

#: The value that means "leave this colour empty". QSpinBox needs a real
#: number for its `specialValueText` slot, so the empty plane lives one below
#: the first legal channel rather than in a separate control.
_EMPTY = -1

#: Label and tooltip per colour slot, in file order.
_SLOTS = (
    ("r", "R", "Source channel shown as RED (conventionally 555 / 647)"),
    ("g", "G", "Source channel shown as GREEN (conventionally 488)"),
    ("b", "B", "Source channel shown as BLUE (conventionally 405 / DAPI)"),
)


class ChannelMappingWidget(QWidget):
    """Editor for a ``{'r': int, 'g': int, 'b': int}`` channel mapping."""

    valueChanged = Signal(dict)

    def __init__(self, value: Any = None, parent: Optional[QWidget] = None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._boxes: Dict[str, QSpinBox] = {}
        for key, label_text, tip in _SLOTS:
            label = QLabel(label_text, self)
            label.setObjectName(f"ChannelMappingLabel{label_text}")
            # The whole help, on the name (instruction 113). The spin box
            # used to carry a longer variant of this text, so hovering the
            # field the user was about to type in covered it with a tooltip
            # they had already read on the label beside it.
            label.setToolTip(tip + ". “—” leaves this colour empty.")
            layout.addWidget(label)

            box = QSpinBox(self)
            box.setObjectName(f"ChannelMappingSpin{label_text}")
            box.setRange(_EMPTY, MAX_SOURCE_CHANNEL)
            box.setSpecialValueText("—")     # shown when the value is _EMPTY
            box.valueChanged.connect(self._emit)
            layout.addWidget(box)
            self._boxes[key] = box
        layout.addStretch(1)

        # A plain QWidget used as a layout container inherits the blanket
        # `QWidget { background-color: bg }` rule and paints the window colour
        # over whatever is behind it (INVARIANTS §1/§3). This widget registers
        # no QSS of its own precisely so there is no new rule to forget to add
        # to theme.WIDGET_QSS_MODULES -- the children are styled by the
        # existing QSpinBox/QLabel rules, and the container paints nothing.
        try:
            from ..theme import make_transparent
            make_transparent(self)
        except Exception:
            # Decoration must never be load-bearing (INVARIANTS §10): if the
            # theme cannot be reached the field still works, it just sits on
            # the window colour.
            pass

        self.set_value(value)
        # Hover help belongs on a setting's NAME, not on the field the user
        # is about to type into (instruction 113). One post-pass rather than
        # a convention every hand-built row has to remember.
        from ..screens.settings_model import retarget_field_tooltips
        retarget_field_tooltips(self)

    # -- value -------------------------------------------------------------

    def get_value(self) -> Dict[str, Optional[int]]:
        """Return the mapping, with ``None`` for any colour left empty."""
        out: Dict[str, Optional[int]] = {}
        for key, box in self._boxes.items():
            raw = box.value()
            out[key] = None if raw == _EMPTY else int(raw)
        return out

    def set_value(self, value: Any) -> None:
        """Load a mapping dict, a legacy ``png_dims`` list, or nothing.

        The list form is accepted because a settings CSV written by an older
        build holds one, and the panel has to be able to show what that file
        actually asked for. It is translated the same way the pipeline
        translates it -- entry 0 is blue -- so the fields show the colours the
        run will produce, not a rearrangement of them.
        """
        mapping = self._coerce(value)
        for key, box in self._boxes.items():
            idx = mapping.get(key)
            box.blockSignals(True)
            box.setValue(_EMPTY if idx is None else int(idx))
            box.blockSignals(False)
        self._emit()

    @staticmethod
    def _coerce(value: Any) -> Dict[str, Optional[int]]:
        from ...crops import (
            DEFAULT_PNG_CHANNEL_MAPPING,
            png_dims_to_channel_mapping,
            resolve_png_channel_mapping,
        )
        if value is None:
            return dict(DEFAULT_PNG_CHANNEL_MAPPING)
        if isinstance(value, str):
            import ast
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return dict(DEFAULT_PNG_CHANNEL_MAPPING)
        try:
            if isinstance(value, (list, tuple)):
                return png_dims_to_channel_mapping(value)
            if isinstance(value, dict):
                return resolve_png_channel_mapping(
                    {"png_channel_mapping": value})
        except Exception:
            pass
        return dict(DEFAULT_PNG_CHANNEL_MAPPING)

    def _emit(self, *_args) -> None:
        self.valueChanged.emit(self.get_value())
