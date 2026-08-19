"""The annotator's control over the picture, offered wherever cells are shown.

Instruction 170: "a settings button that spawns a settings window like
annotation aplication and gives the user the same controll over how to show
the images like the annotation application. settings that do not apply for
the chosen method are grayed out."

THE DEFAULTS COME FROM `set_annotate_default_settings`, not from a list here.
That is the whole reason this dialog is thin: the annotator already decides
what these settings are called, what they default to and what type they are,
and a second declaration would be a second answer. Instruction 145.

WHAT APPLIES TO WHICH MODE IS `spacr.picture_settings`, for the same reason
one level up: the greying rule has to hold when a settings CSV reaches the
code without passing a panel, so the panel asks the table rather than being
the table.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                               QLabel, QLineEdit, QSpinBox, QVBoxLayout,
                               QWidget)

from ...picture_settings import ALL_KEYS, applies_to, why_not

__all__ = ["PictureSettingsDialog", "picture_defaults"]


def picture_defaults() -> Dict[str, Any]:
    """The annotator's defaults for the keys this dialog offers."""
    from ...settings import set_annotate_default_settings

    try:
        filled = set_annotate_default_settings({})
    except Exception:                                        # noqa: BLE001
        filled = {}
    if not isinstance(filled, dict):
        filled = {}
    from ...picture_settings import OWN_DEFAULTS

    return {key: filled.get(key, OWN_DEFAULTS.get(key)) for key in ALL_KEYS}


def _editor(value: Any, parent: Optional[QWidget] = None,
            choices: Any = ()) -> QWidget:
    """A control suited to ``value``'s type.

    Deliberately small: a float gets a step that follows its magnitude, for
    the reason the settings panel had to be taught the same thing -- a spin
    box left at Qt's default step of 1.0 turns 0.05 into -0.95 on one wheel
    tick.
    """
    if choices:
        # BUILT FROM THE SCREEN, not typed. Offering `object_array` as free
        # text asks the user to remember what their own screen contains and
        # to spell it the way `measure` did -- and every other chooser in
        # spaCR is built from the data.
        combo = QComboBox(parent)
        for option in choices:
            combo.addItem(str(option), option)
        current = combo.findData(value)
        if current < 0:
            current = combo.findText(str(value))
        combo.setCurrentIndex(max(current, 0))
        return combo
    if isinstance(value, bool):
        box = QCheckBox(parent)
        box.setChecked(value)
        return box
    if isinstance(value, int):
        spin = QSpinBox(parent)
        spin.setRange(0, 1_000_000)
        spin.setValue(int(value))
        return spin
    if isinstance(value, float):
        spin = QDoubleSpinBox(parent)
        spin.setDecimals(4)
        spin.setRange(-1e6, 1e6)
        spin.setSingleStep(0.01 if abs(value) < 1 else 0.1)
        spin.setValue(float(value))
        return spin
    edit = QLineEdit(parent)
    edit.setText("" if value is None else str(value))
    return edit


def _value_of(widget: QWidget) -> Any:
    if isinstance(widget, QComboBox):
        data = widget.currentData()
        return widget.currentText() if data is None else data
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        return widget.value()
    return widget.text()


class PictureSettingsDialog(QDialog):
    """How the cells are drawn, with what the mode cannot use greyed out."""

    def __init__(self, values: Optional[Dict[str, Any]] = None,
                 mode: str = "png", parent: Optional[QWidget] = None, *,
                 source: Any = None, objects: Any = None):
        super().__init__(parent)
        self.setWindowTitle("Picture settings")
        self._mode = str(mode or "png")
        self._editors: Dict[str, QWidget] = {}
        self._labels: Dict[str, QLabel] = {}

        start = dict(picture_defaults())
        start.update({k: v for k, v in (values or {}).items() if k in ALL_KEYS})

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        from ...picture_settings import offered_values

        for key in ALL_KEYS:
            editor = _editor(start.get(key), self,
                             choices=offered_values(key, source=source,
                                                    frame=objects))
            label = QLabel(key.replace("_", " "), self)
            # THE TOOLTIP IS ON THE LABEL, not the field: a tooltip on the
            # control fires while the user is editing it, which is the one
            # moment they did not ask for it.
            self._editors[key] = editor
            self._labels[key] = label
            form.addRow(label, editor)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
                                   parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.set_mode(self._mode)

    # ------------------------------------------------------------------ mode

    def set_mode(self, mode: str) -> None:
        """Grey what ``mode`` cannot use, and say why on each.

        GREYED, NEVER HIDDEN (INVARIANTS 6). A control that vanishes cannot
        tell the user why their mode does not offer it.
        """
        self._mode = str(mode or "png")
        from ...settings import tooltips

        for key, editor in self._editors.items():
            usable = applies_to(key, self._mode)
            editor.setEnabled(usable)
            label = self._labels[key]
            label.setEnabled(usable)
            explain = why_not(key, self._mode) if not usable else (
                str(tooltips.get(key, "") or ""))
            label.setToolTip(explain)

    def mode(self) -> str:
        return self._mode

    # ---------------------------------------------------------------- values

    def values(self) -> Dict[str, Any]:
        """Every setting, including the greyed ones.

        THE GREYED ONES ARE RETURNED UNCHANGED rather than dropped: a user who
        set `object_array`, switched to load images and switched back must
        find it where they left it, and a dialog that forgot it would be
        indistinguishable from one that reset it.
        """
        return {key: _value_of(editor)
                for key, editor in self._editors.items()}

    def applied_values(self) -> Dict[str, Any]:
        """Only the settings the current mode actually uses."""
        return {key: value for key, value in self.values().items()
                if applies_to(key, self._mode)}
