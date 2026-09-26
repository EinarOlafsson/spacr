"""The form that collects a shared model's scorecard before it is uploaded.

Also the consent step and the upload shared by every community training-data
contribution (item 523), so each screen that contributes asks the same
question once and sends the same way.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox,
                               QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QVBoxLayout,
                               QWidget)

from ..i18n import tr
from .model_share import COMMUNITY_CONSENT_KEY, COMMUNITY_LICENCE, SHARE_FIELDS

LOG = logging.getLogger("spacr.qt.model_share_dialog")


class ShareDialog(QDialog):
    """One line per scorecard column, in the order the card prints them.

    Blank is allowed everywhere: a card that says "not recorded" is honest,
    and a form that refused to submit without a number would be answered with
    a made-up one.
    """

    def __init__(self, filename: str, parent: Optional[QWidget] = None):
        """Build the form for sharing ``filename``.

        :param filename: the checkpoint being shared, named in the title.
        :param parent: the dialog's parent widget.
        """
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


class ConsentDialog(QDialog):
    """Asked once, before the first contribution leaves the machine.

    :param parent: the owning widget.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """Build the two statements and the buttons.

        :param parent: the owning widget.
        """
        super().__init__(parent)
        self.setWindowTitle(tr("Before your first contribution"))
        layout = QVBoxLayout(self)
        note = QLabel(tr(
            "What you contribute is published openly on Hugging Face, so "
            "that anyone can train and check a model on it. It is reviewed "
            "before it is used."))
        note.setWordWrap(True)
        layout.addWidget(note)
        self.rights = QCheckBox(tr(
            "I have the right to share these images: I made them, or their "
            "source (for a paper figure, the paper's licence) allows it."))
        self.licence = QCheckBox(tr(
            "I agree that they and my annotations are shared under {licence}.",
            licence=COMMUNITY_LICENCE))
        for box in (self.rights, self.licence):
            layout.addWidget(box)
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)
        for box in (self.rights, self.licence):
            box.toggled.connect(self._sync)
        self._sync()

    def _sync(self, *_args: Any) -> None:
        """OK only when both are ticked."""
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(
            self.rights.isChecked() and self.licence.isChecked())


def _preferences():
    """The preferences store; see :func:`spacr.qt.preferences._settings`."""
    from ..preferences import _settings

    return _settings()


def community_consented() -> bool:
    """Whether the contributor already agreed to the community licence."""
    try:
        return str(_preferences().value(COMMUNITY_CONSENT_KEY, "")) \
            == COMMUNITY_LICENCE
    except Exception:
        return False


def ask_community_consent(parent: Optional[QWidget] = None, *,
                          ask: Optional[Callable[[], bool]] = None) -> bool:
    """Ask once, before the first community contribution; remember a yes.

    :param parent: the window the question belongs to.
    :param ask: ``fn() -> bool`` in place of :class:`ConsentDialog` (tests).
    :returns: True when the contributor has agreed, now or before.
    """
    if community_consented():
        return True
    agreed = ask() if ask is not None else \
        ConsentDialog(parent).exec() == QDialog.Accepted
    if agreed:
        try:
            _preferences().setValue(COMMUNITY_CONSENT_KEY, COMMUNITY_LICENCE)
        except Exception:
            LOG.debug("could not store the contribution consent", exc_info=True)
    return bool(agreed)


def upload_with_own_login(folder: Any, target: str) -> str:
    """Send a contribution with the contributor's own Hugging Face login.

    :param folder: a folder :func:`~spacr.qt.widgets.model_share.write_contribution` made.
    :param target: the community collection; see
        :func:`~spacr.qt.widgets.model_share.community_repo`.
    :returns: the pull request's URL.
    """
    from . import model_share

    token = model_share.find_token()
    if not token:
        raise RuntimeError(tr(
            "Log in to Hugging Face first (a free account: run "
            "'huggingface-cli login', or set HF_TOKEN), then press Upload "
            "again. Your annotations are kept while this window is open."))
    return model_share.contribute(folder, target, token)
