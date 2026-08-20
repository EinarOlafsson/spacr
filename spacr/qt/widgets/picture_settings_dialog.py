"""Picture-rendering controls shared by cell and image views.

Defaults come from :func:`spacr.settings.set_annotate_default_settings`, and
mode applicability comes from :mod:`spacr.picture_settings`. The dialog thus
uses the same values and availability rules as non-GUI callers.
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
    """Return typed defaults for every setting offered by the dialog."""
    from ...settings import set_annotate_default_settings

    try:
        filled = set_annotate_default_settings({})
    except Exception:                                        # noqa: BLE001
        filled = {}
    if not isinstance(filled, dict):
        filled = {}
    from ...picture_settings import OWN_DEFAULTS

    # OWN_DEFAULTS WINS WHERE IT SPEAKS, and that is not a preference for
    # our own table: it is where a shipped default is the wrong TYPE for a
    # control. `set_annotate_default_settings` ships the STRING 'False' for
    # `edge_image`, and a non-empty string is TRUE -- so the flag read as on
    # everywhere it was used as one, and this dialog drew a text box
    # containing the word False instead of a checkbox. The annotator's value
    # is still what fills every key OWN_DEFAULTS does not name.
    out = {}
    for key in ALL_KEYS:
        if key in OWN_DEFAULTS:
            out[key] = OWN_DEFAULTS[key]
        else:
            out[key] = filled.get(key)
    return out


#: The two settings that name channels rather than choose from a list.
CHANNEL_KEYS = ("channels", "normalize_channels")


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
            # A chooser may offer (value, label) or a bare value. The STORED
            # value is always the first, so a label can be renamed without
            # changing what any settings file already on disk means.
            #
            # `stored`, NOT `value`: the first version of this loop unpacked
            # into `value` and so clobbered the parameter it was about to
            # search for -- every dropdown then opened on its LAST entry,
            # whatever the setting actually was.
            if isinstance(option, tuple) and len(option) == 2:
                stored, label = option
            else:
                stored = label = option
            combo.addItem(str(label), stored)
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
    """Edit picture settings while retaining mode-inapplicable values."""

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

        # THE R,G,B SYSTEM FOR THE TWO CHANNEL SETTINGS (188 B). A dropdown
        # of the eight combinations made "which channels are on" a question
        # you had to open a list to answer, and turning one channel off --
        # the thing a user does constantly here -- two clicks.
        from .channel_picker import ChannelPicker

        for key in ALL_KEYS:
            if key in CHANNEL_KEYS:
                editor = ChannelPicker(
                    start.get(key), self,
                    # `channels` with nothing on is a blank picture;
                    # `normalize_channels` with nothing on means "normalise
                    # nothing", which is a real answer.
                    allow_none=(key != "channels"))
            else:
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

        # THE SAME API HELP THE SETTINGS PANEL GIVES, not a plainer copy.
        # Reported: "there are still no tooltips with api guides". These
        # labels carried the description string and nothing else -- no
        # `settingKey`, no rendered `apiTooltipHtml`, so no link to the API
        # page and none of the typed metadata every other reader keys on.
        #
        # `api_dots=False` for the reason the Annotate settings dialog turns
        # them off: twenty-five settings is twenty-five teal dots, which
        # reads as a column of dots rather than a column of settings, and the
        # link is in the hover text either way.
        try:
            from ..screens.settings_model import install_api_tooltips

            install_api_tooltips(
                self, "annotate",
                {editor: key for key, editor in self._editors.items()},
                api_dots=False)
        except Exception:                                    # noqa: BLE001
            # A dialog that cannot decorate its help is still a dialog that
            # sets the picture. The plain descriptions installed below remain.
            pass

        self.set_mode(self._mode)

    # ------------------------------------------------------------------ mode

    def set_mode(self, mode: str) -> None:
        """Update control availability for an image-source mode.

        Inapplicable controls remain visible and explain why they are disabled,
        so switching modes does not hide or discard configured values.
        """
        self._mode = str(mode or "png")
        from ...settings import tooltips

        for key, editor in self._editors.items():
            usable = applies_to(key, self._mode)
            editor.setEnabled(usable)
            label = self._labels[key]
            label.setEnabled(usable)
            if not usable:
                # THE REASON BEATS THE DESCRIPTION when the control is
                # greyed: what the user is asking at that moment is why they
                # cannot touch it, not what it would have done.
                label.setToolTip(why_not(key, self._mode))
                continue
            # THE RICH API HELP IF IT WAS INSTALLED, and the plain
            # description otherwise. This line used to write the plain text
            # unconditionally, straight over the HTML `install_api_tooltips`
            # had just rendered -- so every label carried the API metadata
            # and showed none of it. Reported as "there are still no
            # tooltips with api guides".
            rich = str(label.property("apiTooltipHtml") or "")
            label.setToolTip(rich or str(tooltips.get(key, "") or ""))

    def mode(self) -> str:
        return self._mode

    # ---------------------------------------------------------------- values

    def values(self) -> Dict[str, Any]:
        """Return every configured value, including disabled controls.

        Values for the current mode's disabled controls are preserved so that
        switching away from a mode and back restores the prior configuration.
        """
        return {key: _value_of(editor)
                for key, editor in self._editors.items()}

    def applied_values(self) -> Dict[str, Any]:
        """Only the settings the current mode actually uses."""
        return {key: value for key, value in self.values().items()
                if applies_to(key, self._mode)}
