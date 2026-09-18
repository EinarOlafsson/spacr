"""The form that collects a shared model's scorecard before it is uploaded."""
from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout, QLabel,
                               QLineEdit, QVBoxLayout, QWidget)

from .model_share import SHARE_FIELDS


class ShareDialog(QDialog):
    """One line per scorecard column, in the order the card prints them.

    Blank is allowed everywhere: a card that says "not recorded" is honest,
    and a form that refused to submit without a number would be answered with
    a made-up one.
    """

    def __init__(self, filename: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(f"Share {filename}")
        outer = QVBoxLayout(self)
        note = QLabel(
            "These numbers become the model's table on Hugging Face. Leave "
            "anything you did not measure blank — it will read “not recorded”, "
            "which is better than a guess.", self)
        note.setWordWrap(True)
        outer.addWidget(note)

        form = QFormLayout()
        self._edits: Dict[str, QLineEdit] = {}
        for key, label, default in SHARE_FIELDS:
            edit = QLineEdit(default, self)
            self._edits[key] = edit
            form.addRow(label, edit)
        outer.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def values(self) -> Dict[str, str]:
        """What the user typed, by field key."""
        return {key: edit.text().strip() for key, edit in self._edits.items()}
