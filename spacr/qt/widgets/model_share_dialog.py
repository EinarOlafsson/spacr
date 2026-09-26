"""The form that collects a shared model's scorecard before it is uploaded.

Also the consent step and the upload shared by every community training-data
contribution (item 523), so each screen that contributes asks the same
question once and sends the same way, and Make Masks' dialog for sending
images with their masks to a community dataset the user names.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QDialog,
                               QDialogButtonBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit,
                               QPlainTextEdit, QPushButton, QRadioButton,
                               QVBoxLayout, QWidget)

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
    """Send a contribution, with the contributor's own Hugging Face login if any.

    With a login, the pull request is opened with it (see
    :func:`~spacr.qt.widgets.model_share.contribute`). Without one, the
    folder goes through spaCR's upload Space, which opens the same pull
    request with its own token, so no account is needed.

    :param folder: a folder :func:`~spacr.qt.widgets.model_share.write_contribution` made.
    :param target: the community collection; see
        :func:`~spacr.qt.widgets.model_share.community_repo`.
    :returns: the pull request's URL.
    :raises RuntimeError: with no login, when the upload Space cannot take
        it; the message says why and how to send it with a login instead.
    """
    from . import model_share

    token = model_share.find_token()
    if token:
        return model_share.contribute(folder, target, token)
    try:
        return model_share.central_contribute(folder, target)
    except Exception as exc:
        raise RuntimeError(tr(
            "spaCR's upload service could not take it ({why}). You can send "
            "it with a free Hugging Face login instead: run 'huggingface-cli "
            "login', or set HF_TOKEN, then press Upload again. Your "
            "annotations are kept while this window is open.",
            why=exc)) from exc


LAST_MASKS_DATASET_KEY = "community_training_data/last_masks_dataset"
CURRENT_SOURCE = "current"
CURATED_SOURCE = "curated"
FOLDERS_SOURCE = "folders"


def contribute_masks_tooltip() -> str:
    """What "Contribute images and masks…" does: its tooltip and the dialog's intro."""
    return tr(
        "If you can't train a model yourself, upload your images and masks "
        "here and we'll train one for you. Send the image on screen with its "
        "mask, every image you have curated here, or an images folder with a "
        "masks folder holding masks of the same names. Say what they show "
        "(organism, object type, stain) and they go to their own community "
        "dataset on Hugging Face as a pull request, which the maintainer "
        "reviews before any model is trained on it. They are shared under "
        "{licence}, so send only images you have the right to share. A "
        "careful mask makes a good model: every object, hugging its edge.",
        licence=COMMUNITY_LICENCE)


def _names(names: List[str], limit: int = 12) -> str:
    """``names`` joined, cut short with a count when there are many."""
    shown = ", ".join(names[:limit])
    if len(names) > limit:
        shown += " " + tr("and {n} more", n=len(names) - limit)
    return shown


def _has_objects(labels: Any) -> bool:
    """Whether a label mask holds at least one object."""
    import numpy as np

    return labels is not None and bool(np.asarray(labels).any())


class ContributeMasksDialog(QDialog):
    """Send images and their masks to a community dataset the user names.

    Make Masks' way into the community training datasets. Three
    sources, whichever are on offer: the image on screen with the mask
    being edited, every image of the open folder with a saved mask, or an
    images folder and a masks folder matched by file name. The user names
    the dataset (organism, object type, stain), which becomes
    ``einarolafsson/community_<name>``, and may add notes. Nothing is sent
    without a mask for every image, the consent asked once for every
    community contribution, and the conscience text beside Upload.

    :param current: the image on screen as one ``{name, source, labels}``
        item, or None when no image is open.
    :param curated: the open folder's images with saved masks, as
        ``{name, source, mask}`` items (``mask`` a file read at upload) or
        ``{name, source, labels}``; None when no folder is open.
    :param images_dir: the images folder to start with.
    :param masks_dir: the masks folder to start with.
    :param name: the dataset name to start with; the last one used when
        empty.
    :param upload: ``fn(folder, target) -> url`` in place of
        :func:`upload_with_own_login` (tests).
    :param threaded: send on a worker thread; False sends inline (tests).
    :param parent: the owning widget.
    """

    uploaded = Signal(str)

    def __init__(self, *, current: Optional[Dict[str, Any]] = None,
                 curated: Optional[List[Dict[str, Any]]] = None,
                 images_dir: str = "", masks_dir: str = "", name: str = "",
                 upload: Optional[Callable[[Any, str], str]] = None,
                 threaded: bool = True, parent: Optional[QWidget] = None):
        """Build the sources, the two folder fields, the name and the notes.

        :param current: see the class docstring.
        :param curated: see the class docstring.
        :param images_dir: see the class docstring.
        :param masks_dir: see the class docstring.
        :param name: see the class docstring.
        :param upload: see the class docstring.
        :param threaded: see the class docstring.
        :param parent: see the class docstring.
        """
        from ..job_runner import JobRunner
        from .model_share import conscience_for

        super().__init__(parent)
        self.setWindowTitle(tr("Contribute images and masks"))
        self._current = current
        self._curated = list(curated) if curated is not None else None
        self._upload = upload or upload_with_own_login
        self.ask_consent: Optional[Callable[[], bool]] = None
        self.contribution_folder = None
        self.report: Dict[str, Any] = {}
        self._uploading = False
        self._jobs = JobRunner(self, threaded=threaded,
                               app_key="mask contribution", user_visible=False)
        self._jobs.job_failed.connect(self._on_failed)

        outer = QVBoxLayout(self)
        intro = QLabel(contribute_masks_tooltip(), self)
        intro.setWordWrap(True)
        outer.addWidget(intro)

        self._sources = QButtonGroup(self)
        self.source_buttons: Dict[str, QRadioButton] = {}
        offered = []
        if current is not None:
            offered.append((CURRENT_SOURCE, tr(
                "The image on screen and its mask: {name}",
                name=str(current.get("name") or ""))))
        if curated is not None:
            offered.append((CURATED_SOURCE, tr(
                "Every curated image here with its saved mask ({n})",
                n=len(self._curated))))
        offered.append((FOLDERS_SOURCE, tr(
            "An images folder and a masks folder")))
        for key, text in offered:
            button = QRadioButton(text, self)
            self._sources.addButton(button)
            self.source_buttons[key] = button
            outer.addWidget(button)
        if len(offered) == 1:
            for button in self.source_buttons.values():
                button.hide()

        self.folders = QWidget(self)
        form = QFormLayout(self.folders)
        form.setContentsMargins(0, 0, 0, 0)
        self.images_edit = QLineEdit(str(images_dir or ""), self.folders)
        self.masks_edit = QLineEdit(str(masks_dir or ""), self.folders)
        for label, edit, title in (
                (tr("Images folder"), self.images_edit,
                 tr("Choose the images folder")),
                (tr("Masks folder"), self.masks_edit,
                 tr("Choose the masks folder"))):
            row = QHBoxLayout()
            row.addWidget(edit, 1)
            browse = QPushButton(tr("Browse…"), self.folders)
            browse.clicked.connect(
                lambda _c=False, e=edit, t=title: self._browse(e, t))
            row.addWidget(browse)
            form.addRow(label, row)
        outer.addWidget(self.folders)

        self.check = QLabel(self)
        self.check.setWordWrap(True)
        self.check.setTextInteractionFlags(Qt.TextSelectableByMouse)
        outer.addWidget(self.check)

        about = QFormLayout()
        self.name_edit = QLineEdit(str(name or self._last_name()), self)
        self.name_edit.setPlaceholderText(tr(
            "Organism, object type, stain: e.g. toxoplasma vacuoles GFP"))
        about.addRow(tr("What is it?"), self.name_edit)
        self.target_label = QLabel(self)
        self.target_label.setWordWrap(True)
        about.addRow("", self.target_label)
        self.notes_edit = QPlainTextEdit(self)
        self.notes_edit.setPlaceholderText(tr(
            "Anything that helps: microscope, magnification, cell line, "
            "what counts as one object."))
        self.notes_edit.setMaximumHeight(80)
        about.addRow(tr("Notes"), self.notes_edit)
        outer.addLayout(about)

        conscience = QLabel(tr(conscience_for("masks")), self)
        conscience.setWordWrap(True)
        conscience.setObjectName("ContributionConscience")
        self.conscience = conscience
        outer.addWidget(conscience)
        self.status = QLabel(self)
        self.status.setWordWrap(True)
        self.status.setOpenExternalLinks(True)
        outer.addWidget(self.status)
        buttons = QDialogButtonBox(self)
        self.upload_button = buttons.addButton(tr("Upload"),
                                               QDialogButtonBox.AcceptRole)
        self.upload_button.setToolTip(tr(
            "Send these images and masks for review. Enabled when the "
            "dataset has a name and every image has a mask."))
        buttons.addButton(QDialogButtonBox.Close)
        buttons.accepted.connect(self.upload)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        for button in self.source_buttons.values():
            button.toggled.connect(self._refresh)
        for edit in (self.images_edit, self.masks_edit, self.name_edit):
            edit.textChanged.connect(self._refresh)
        first = CURRENT_SOURCE if current is not None else (
            CURATED_SOURCE if curated else FOLDERS_SOURCE)
        self.use(first)

    @staticmethod
    def _last_name() -> str:
        """The dataset name the last contribution went to, or ``""``."""
        try:
            return str(_preferences().value(LAST_MASKS_DATASET_KEY, "") or "")
        except Exception:
            return ""

    def _browse(self, edit: QLineEdit, title: str) -> None:
        """Pick a folder into ``edit``."""
        folder = QFileDialog.getExistingDirectory(self, title, edit.text())
        if folder:
            edit.setText(folder)

    def use(self, source: str) -> None:
        """Switch to one of the sources on offer.

        :param source: ``"current"``, ``"curated"`` or ``"folders"``.
        """
        self.source_buttons[source].setChecked(True)
        self._refresh()

    def source(self) -> str:
        """The source chosen: ``"current"``, ``"curated"`` or ``"folders"``."""
        for key, button in self.source_buttons.items():
            if button.isChecked():
                return key
        return FOLDERS_SOURCE

    def target(self) -> str:
        """The community target the typed name maps to, or ``""``."""
        from .model_share import masks_dataset_target

        try:
            return masks_dataset_target(self.name_edit.text())
        except ValueError:
            return ""

    def _check_folders(self) -> str:
        """Match the two folders, list what does not match, say why not."""
        from .model_share import pair_images_and_masks, pairs_match

        report = pair_images_and_masks(self.images_edit.text(),
                                       self.masks_edit.text())
        self.report = report
        if report["problem"]:
            why = tr("Not ready: {why}.", why=report["problem"])
            self.check.setText(why)
            return why
        problems = []
        if report["no_mask"]:
            problems.append(tr("No mask with the same name for: {names}",
                               names=_names(report["no_mask"])))
        if report["no_image"]:
            problems.append(tr("No image with the same name for: {names}",
                               names=_names(report["no_image"])))
        if report["duplicates"]:
            problems.append(tr("More than one file with the same name: {names}",
                               names=_names(report["duplicates"])))
        if not problems and not pairs_match(report):
            problems.append(tr("There are no image and mask pairs to send."))
        counts = tr("{images} images, {masks} masks.",
                    images=report["images"], masks=report["masks"])
        if problems:
            self.check.setText("\n".join([counts] + problems))
            return " ".join(problems)
        self.check.setText(counts + " " + tr("Every image has its mask. Ready."))
        return ""

    def _not_ready(self) -> str:
        """Why Upload cannot be pressed yet, or ``""`` when it can."""
        from .model_share import community_repo

        source = self.source()
        self.folders.setVisible(source == FOLDERS_SOURCE)
        why = ""
        if source == CURRENT_SOURCE:
            if not _has_objects((self._current or {}).get("labels")):
                why = tr("The image on screen has no mask yet: segment or "
                         "paint it first. No mask, no upload.")
                self.check.setText(why)
            else:
                self.check.setText(tr("Ready: {name} and its mask.",
                                      name=self._current.get("name", "")))
        elif source == CURATED_SOURCE:
            if not self._curated:
                why = tr("No image here has a saved mask yet.")
                self.check.setText(why)
            else:
                self.check.setText(tr(
                    "Ready: {n} images, each with its saved mask.",
                    n=len(self._curated)))
        else:
            why = self._check_folders()
        target = self.target()
        if target:
            self.target_label.setText(tr(
                "Goes to huggingface.co/datasets/{repo} (made if it is new).",
                repo=community_repo(target)))
        else:
            self.target_label.setText(tr(
                "Name the dataset: what the images show, and how."))
            why = why or tr("Name the dataset first.")
        return why

    def _refresh(self, *_args: Any) -> None:
        """Enable Upload only when there is something whole to send."""
        why = self._not_ready()
        self.upload_button.setEnabled(not why and not self._uploading)

    def _items(self) -> List[Dict[str, Any]]:
        """The contribution as :func:`~spacr.qt.widgets.model_share.write_contribution` takes it."""
        from .model_share import read_label_mask

        source = self.source()
        if source == CURRENT_SOURCE:
            raw = [dict(self._current or {})]
        elif source == CURATED_SOURCE:
            raw = [dict(item) for item in self._curated or ()]
        else:
            raw = [{"name": image.name, "source": str(image), "mask": mask}
                   for image, mask in self.report["pairs"]]
        items = []
        for item in raw:
            mask = item.pop("mask", None)
            if item.get("labels") is None and mask is not None:
                item["labels"] = read_label_mask(mask)
            extra = dict(item.get("extra") or {})
            extra["dataset"] = self.name_edit.text().strip()
            if mask is not None:
                extra["mask_file"] = os.path.basename(str(mask))
            item["extra"] = extra
            items.append(item)
        return items

    def upload(self, *_args: Any) -> bool:
        """Check, ask for consent once, write the contribution and send it.

        :returns: True when an upload was started.
        """
        import tempfile

        from .model_share import write_contribution

        why = self._not_ready()
        if why or self._uploading:
            self.status.setText(tr("Not sent: {why}", why=why or tr(
                "an upload is already running")))
            self._refresh()
            return False
        if not ask_community_consent(self, ask=self.ask_consent):
            self.status.setText(tr("Not sent: the licence was not agreed to."))
            return False
        consent = {"rights_to_share": True, "licence": COMMUNITY_LICENCE}
        target = self.target()
        try:
            folder = write_contribution(
                target, self._items(),
                tempfile.mkdtemp(prefix="spacr-contribution-"),
                consent=consent, notes=self.notes_edit.toPlainText())
        except (OSError, ValueError) as exc:
            self.status.setText(tr("Not sent: {why}", why=str(exc)))
            return False
        try:
            _preferences().setValue(LAST_MASKS_DATASET_KEY,
                                    self.name_edit.text().strip())
        except Exception:
            LOG.debug("could not remember the dataset name", exc_info=True)
        self.contribution_folder = folder
        self._uploading = True
        self._refresh()
        self.status.setText(tr("Uploading…"))
        upload = self._upload
        self._jobs.submit(lambda: upload(folder, target), self._on_uploaded)
        return True

    def _on_uploaded(self, url: Any) -> None:
        """The upload finished."""
        self._uploading = False
        url = str(url or "")
        self.status.setText(tr(
            "Thank you. Your contribution is waiting for review: "
            "<a href=\"{url}\">{url}</a>", url=url))
        self._refresh()
        self.uploaded.emit(url)

    def _on_failed(self, message: str) -> None:
        """The upload raised."""
        self._uploading = False
        self.status.setText(tr("Upload failed: {why}", why=message))
        self._refresh()
