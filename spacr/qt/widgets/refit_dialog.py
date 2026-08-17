"""Pick another model for the screen already on screen.

Asked for on 2026-08-16: "when all the analasees are done id like to be able
to right click on the regression plot and choose a different regression and
the other related settings as well as FDR etc."

THIS DIALOG STARTS A FIT. Everything else reachable by right-clicking a plot
changes how it looks, so this one has to read differently: it names the model
the table on screen was fitted with, it says WHERE the new run will write, and
it says in advance which settings it is about to reset -- all before the OK
button does anything.

The reset sentence is the one that matters. Switching from a penalised model
to an unpenalised one drops the penalty weight, because the fit refuses to
accept a number it cannot read (:func:`spacr.ml._reject_unused_settings`),
and a user who reads "alpha" on the settings panel afterwards is entitled to
have been told rather than to work it out from a folder name.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QLabel,
                               QVBoxLayout)


class RefitDialog(QDialog):
    """Choose a regression type, a correction and a penalty weight."""

    def __init__(self, settings: dict, parent=None):
        """
        :param settings: the settings the run on screen used. Read only; the
            new dict is built by :func:`spacr.refit.refit_settings`.
        """
        super().__init__(parent)
        # From the spec, not from ml: importing ml here would pull torch,
        # cv2 and IPython onto the GUI thread the moment the user opens this
        # dialog. The tables are the same objects; ml re-exports them.
        from ...regression_spec import (REGRESSION_SETTINGS_USED,
                                        REGRESSION_TYPES)
        from ...multiple_testing import METHODS
        from ...refit import CORRECTION_ALPHA_KEY, CORRECTION_KEY

        self.setWindowTitle("Re-fit the screen")
        self._settings = dict(settings or {})
        self._used = REGRESSION_SETTINGS_USED

        layout = QVBoxLayout(self)
        current = self._settings.get("regression_type")
        layout.addWidget(QLabel(
            f"The table on screen was fitted with <b>{current or 'auto'}</b>. "
            f"Re-fitting runs the same data through another model; the run "
            f"you are looking at is not touched."))

        form = QFormLayout()
        self._type = QComboBox()
        # "As before" first, because changing ONLY the correction is a real
        # request -- comparing thirteen corrections on one fit is what the
        # results-folder rule was written for.
        self._type.addItem("as before", None)
        for name in REGRESSION_TYPES:
            self._type.addItem(name, name)
        index = self._type.findData(current)
        self._type.setCurrentIndex(index if index >= 0 else 0)
        self._type.currentIndexChanged.connect(self._refresh)
        form.addRow("Regression", self._type)

        self._correction = QComboBox()
        self._correction.addItem("as before", None)
        for name, spec in METHODS.items():
            self._correction.addItem(
                f"{name} — {getattr(spec, 'label', name)}"
                if getattr(spec, "label", None) else name, name)
        chosen = self._settings.get(CORRECTION_KEY)
        index = self._correction.findData(chosen)
        self._correction.setCurrentIndex(index if index >= 0 else 0)
        self._correction.currentIndexChanged.connect(self._refresh)
        form.addRow("Correction", self._correction)

        # THE SIGNIFICANCE LEVEL, which is not the penalty weight below it
        # despite `alpha` being the name of both in the settings. This one
        # cuts the hit list; that one changes the model.
        self._fdr_alpha = QDoubleSpinBox()
        self._fdr_alpha.setDecimals(4)
        self._fdr_alpha.setRange(0.0001, 0.5)
        self._fdr_alpha.setSingleStep(0.01)
        level = self._settings.get(CORRECTION_ALPHA_KEY)
        self._fdr_alpha.setValue(float(level)
                                 if isinstance(level, (int, float)) else 0.05)
        self._fdr_alpha.valueChanged.connect(self._refresh)
        form.addRow("Significance level", self._fdr_alpha)

        self._alpha = QDoubleSpinBox()
        self._alpha.setDecimals(4)
        self._alpha.setRange(0.0001, 1000.0)
        alpha = self._settings.get("alpha")
        self._alpha.setValue(float(alpha) if isinstance(alpha, (int, float))
                             else 1.0)
        self._alpha.valueChanged.connect(self._refresh)
        self._alpha.setToolTip(
            "The penalty weight of a penalised fit — NOT the significance "
            "level above. Greyed out for a model "
            "that has no penalty, because the number would change nothing "
            "there and the run refuses it rather than ignoring it.")
        form.addRow("Penalty weight", self._alpha)
        layout.addLayout(form)

        self._notice = QLabel("")
        self._notice.setWordWrap(True)
        layout.addWidget(self._notice)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Re-fit")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._refresh()

    # ------------------------------------------------------------------ state

    def chosen(self) -> dict:
        """``{regression_type, correction_method, alpha}`` as picked.

        ``None`` for anything left at "as before", which is what
        :func:`spacr.refit.refit_settings` reads as "leave it alone".
        """
        regression_type = self._type.currentData()
        alpha = None
        if self._alpha.isEnabled():
            effective = regression_type or self._settings.get("regression_type")
            if "alpha" in self._used.get(effective, ()):
                alpha = self._alpha.value()
        return {"regression_type": regression_type,
                "correction_method": self._correction.currentData(),
                "fdr_alpha": self._fdr_alpha.value(),
                "alpha": alpha}

    def settings(self):
        """``(settings, notes)`` for the new run, or raises ValueError."""
        from ...refit import refit_settings

        return refit_settings(self._settings, **self.chosen())

    # ------------------------------------------------------------------- view

    def _refresh(self, *_args) -> None:
        """Say what will change, before anything does."""
        from ...refit import destination, refit_settings

        chosen = self._type.currentData() or self._settings.get(
            "regression_type")
        # A penalty weight on an unpenalised model is not ignored, it is
        # REFUSED -- so the box is disabled rather than left to produce a
        # number the run will reject.
        self._alpha.setEnabled("alpha" in self._used.get(chosen, ()))

        try:
            settings, notes = refit_settings(self._settings, **self.chosen())
        except ValueError as error:
            self._notice.setText(str(error))
            self._ok(False)
            return

        where = destination(settings)
        lines = list(notes)
        if where:
            lines.append(f"Writes to {where}.")
        self._notice.setText(" ".join(lines) if lines else
                             "Nothing to change: re-fitting these settings "
                             "would repeat the run you are looking at.")
        self._ok(True)

    def _ok(self, enabled: bool) -> None:
        box = self.findChild(QDialogButtonBox)
        if box is not None:
            button = box.button(QDialogButtonBox.Ok)
            if button is not None:
                button.setEnabled(enabled)


def ask_refit(settings: dict, parent=None) -> Optional[tuple]:
    """Show the dialog; return ``(settings, notes)`` or None if cancelled."""
    dialog = RefitDialog(settings, parent)
    if dialog.exec() != QDialog.Accepted:
        return None
    return dialog.settings()


__all__ = ["RefitDialog", "ask_refit"]
