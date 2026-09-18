"""The form that collects a shared model's scorecard before it is uploaded."""
from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QVBoxLayout,
                               QWidget)

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

        # Optional training data. A model whose data came with it can be
        # retrained and checked; one without it can only be believed.
        self._train_dir = ""
        train_row = QHBoxLayout()
        self._train_label = QLabel("No training data attached (optional)", self)
        self._train_label.setWordWrap(True)
        train_button = QPushButton("Train data", self)
        train_button.setToolTip(
            "Attach the folder this model was trained on. It is packed into a "
            "single tar and uploaded beside the checkpoint. Large folders take "
            "a long time to pack and upload; the limit is about 25 GB.")
        train_button.clicked.connect(self._choose_train_data)
        train_row.addWidget(self._train_label, 1)
        train_row.addWidget(train_button)
        outer.addLayout(train_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def _choose_train_data(self) -> None:
        """Pick the folder the model was trained on."""
        folder = QFileDialog.getExistingDirectory(
            self, "Folder this model was trained on", self._train_dir or "")
        if not folder:
            return
        self._train_dir = folder
        self._train_label.setText(f"Training data: {folder}")

    def train_data_dir(self) -> str:
        """The chosen training-data folder, or an empty string."""
        return self._train_dir

    def values(self) -> Dict[str, str]:
        """What the user typed, by field key."""
        out = {key: edit.text().strip() for key, edit in self._edits.items()}
        out["train_data_dir"] = self._train_dir
        return out
