"""Browse the model zoo, download a model, and hand back its path.

A model setting takes a filesystem path, which is exact and unhelpful: the
user has to know a model exists, find where it lives, and type it correctly
before anything happens. This dialog is the other way in -- the list of models
spaCR knows about, what each was trained on, whether it is already on this
machine, and a button that downloads one and returns its path to the field
that opened it.

WHY THE DOWNLOAD LOCATION IS A CONTROL RATHER THAN A CONSTANT. Checkpoints are
large -- the Toxoplasma models are 1.2 GB each -- and on a shared workstation
or a laptop with a small root volume the default is often the wrong disk. A
lab that keeps models on a NAS wants them there once, not once per user. So
the folder is on screen, remembered between openings, and shown before anything
is fetched rather than discovered afterwards in a full-disk error.

NOTHING IS DOWNLOADED WITHOUT BEING ASKED FOR. Opening the dialog lists; the
download happens on the button. That matters because the list is useful on its
own -- seeing that a model exists and what it was trained on is often the whole
question -- and because a dialog that starts a gigabyte transfer on open is one
users learn not to open.
"""
from __future__ import annotations

import os
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QDialog, QDialogButtonBox,
                               QFileDialog, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QMessageBox, QProgressBar,
                               QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

#: Where checkpoints land unless the user says otherwise.
DEFAULT_MODEL_DIR = os.path.join(os.path.expanduser("~"), ".spacr", "models")

#: QSettings key remembering the chosen folder.
_DIR_SETTING = "model_zoo/download_dir"

_COLUMNS = ("Model", "Kind", "Trained on", "Status")


def remembered_model_dir() -> str:
    """The folder the user last downloaded into, or the default.

    Reading through QSettings rather than holding it on the dialog: the next
    model is usually wanted in the same place as the last one, and that is
    true across sessions, not only within one.
    """
    try:
        from PySide6.QtCore import QSettings

        stored = str(QSettings().value(_DIR_SETTING, "") or "")
        if stored:
            return stored
    except Exception:                                       # noqa: BLE001
        pass
    return DEFAULT_MODEL_DIR


def _remember_model_dir(folder: str) -> None:
    """Persist the download folder, quietly."""
    try:
        from PySide6.QtCore import QSettings

        QSettings().setValue(_DIR_SETTING, str(folder))
    except Exception:                                       # noqa: BLE001
        pass


class ModelZooPicker(QDialog):
    """Pick a model from the zoo; returns a local path.

    :param kinds: restrict the list to these :data:`spacr.model_zoo.KINDS`.
        A pathogen-model field wants ``("cellpose",)`` -- offering a detector
        there would be offering something that cannot be loaded.
    :param parent: the widget that opened this.
    """

    #: Emitted with the local path when the user accepts a model.
    model_chosen = Signal(str)

    def __init__(self, kinds: Optional[tuple] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Model zoo")
        self.setMinimumWidth(720)
        self._kinds = tuple(kinds) if kinds else None
        self._entries: List = []
        self._chosen_path: Optional[str] = None

        layout = QVBoxLayout(self)

        blurb = QLabel(
            "Models spaCR knows about. A model already on this machine can be "
            "used straight away; the rest are downloaded when you ask for one.")
        blurb.setWordWrap(True)
        layout.addWidget(blurb)

        self.table = QTableWidget(0, len(_COLUMNS), self)
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        layout.addWidget(self.table, 1)

        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel("Save to:"))
        self.folder_edit = QLineEdit(remembered_model_dir(), self)
        self.folder_edit.setToolTip(
            "Where downloaded checkpoints are written. These are large — the "
            "Toxoplasma models are about 1.2 GB each — so a shared drive is "
            "often the right answer on a workstation.")
        folder_row.addWidget(self.folder_edit, 1)
        browse = QPushButton("Browse…", self)
        browse.clicked.connect(self._browse)
        folder_row.addWidget(browse)
        layout.addLayout(folder_row)

        self.progress = QProgressBar(self)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status = QLabel("", self)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        buttons = QDialogButtonBox(self)
        self.download_button = buttons.addButton(
            "Download", QDialogButtonBox.ActionRole)
        self.download_button.clicked.connect(self._download_selected)
        self.use_button = buttons.addButton("Use this model",
                                            QDialogButtonBox.AcceptRole)
        buttons.addButton(QDialogButtonBox.Cancel)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self._accept_selected)
        layout.addWidget(buttons)

        self.refresh()

    # -- data ------------------------------------------------------------

    def refresh(self) -> None:
        """Reload the catalogue and redraw the table."""
        from ... import model_zoo

        try:
            entries = list(model_zoo.catalogue(remote=True))
        except Exception as exc:                            # noqa: BLE001
            # A zoo that cannot be listed must not be a dialog that cannot be
            # opened: the user may already have the model and only need to
            # find it on disk.
            self.status.setText(f"Could not read the model list: {exc}")
            entries = []
        if self._kinds:
            entries = [e for e in entries if e.kind in self._kinds]
        self._entries = entries

        self.table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            local = self._local_path(entry)
            cells = (
                getattr(entry, "key", "") or entry.name,
                entry.kind,
                (entry.trained_on or "")[:160],
                "on this machine" if local else "not downloaded",
            )
            for column, text in enumerate(cells):
                item = QTableWidgetItem(str(text))
                if column == 3 and local:
                    item.setToolTip(local)
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Stretch)
        self._selection_changed()

    def _local_path(self, entry) -> Optional[str]:
        """Where this entry already is on disk, if it is.

        Checks the entry's own recorded path first -- a locally discovered
        model has one -- then the chosen download folder.
        """
        recorded = getattr(entry, "path", "") or ""
        if recorded and os.path.isfile(recorded):
            return recorded
        candidate = os.path.join(self.folder_edit.text().strip()
                                 or DEFAULT_MODEL_DIR, entry.name)
        return candidate if os.path.isfile(candidate) else None

    def selected_entry(self):
        """The catalogue entry on the highlighted row, or ``None``."""
        rows = {i.row() for i in self.table.selectedIndexes()}
        if len(rows) != 1:
            return None
        row = rows.pop()
        return self._entries[row] if 0 <= row < len(self._entries) else None

    # -- actions ---------------------------------------------------------

    def _selection_changed(self) -> None:
        entry = self.selected_entry()
        local = self._local_path(entry) if entry else None
        self.use_button.setEnabled(bool(local))
        self.download_button.setEnabled(bool(entry) and not local)
        if entry is None:
            self.status.setText("")
        elif local:
            self.status.setText(f"Ready: {local}")
        else:
            note = "; ".join(getattr(entry, "notes", ()) or ())
            self.status.setText(note or "Not downloaded yet.")

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Where should models be saved?",
            self.folder_edit.text().strip() or DEFAULT_MODEL_DIR)
        if folder:
            self.folder_edit.setText(folder)
            _remember_model_dir(folder)
            self.refresh()

    def _download_selected(self) -> None:
        """Fetch the highlighted model into the chosen folder."""
        from ... import model_zoo

        entry = self.selected_entry()
        if entry is None:
            return
        folder = self.folder_edit.text().strip() or DEFAULT_MODEL_DIR
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError as exc:
            QMessageBox.warning(self, "Model zoo",
                                f"Cannot write to {folder}:\n{exc}")
            return

        self.progress.setVisible(True)
        self.progress.setRange(0, 0)          # indeterminate; size is unknown
        self.status.setText(f"Downloading {entry.name}…")
        self.download_button.setEnabled(False)
        outcome = ""
        try:
            path = model_zoo.fetch(entry, folder)
        except Exception as exc:                            # noqa: BLE001
            # NAMED, not swallowed. fetch refuses an entry whose checksum does
            # not match, and that refusal is the single most important message
            # this dialog can carry: it means the bytes are not the model.
            QMessageBox.warning(
                self, "Model zoo",
                f"Could not download {entry.name}:\n{exc}")
            outcome = f"Download failed: {exc}"
        else:
            _remember_model_dir(folder)
            outcome = f"Downloaded to {path}"
        finally:
            self.progress.setVisible(False)
            # REFRESH FIRST, THEN SAY WHAT HAPPENED. refresh() re-runs
            # _selection_changed, which rewrites the status line from the
            # selected entry -- so a message set before it is overwritten by
            # the entry's own notes, and the failure the user most needs to
            # see is the one that disappears. Caught by the test below.
            self.refresh()
            if outcome:
                self.status.setText(outcome)

    def _accept_selected(self) -> None:
        entry = self.selected_entry()
        local = self._local_path(entry) if entry else None
        if not local:
            return
        self._chosen_path = local
        self.model_chosen.emit(local)
        self.accept()

    def chosen_path(self) -> Optional[str]:
        """The path the user accepted, or ``None`` if they cancelled."""
        return self._chosen_path


def choose_model(parent: Optional[QWidget] = None,
                 kinds: Optional[tuple] = None) -> Optional[str]:
    """Open the picker and return the chosen path, or ``None``.

    The one-call form for a settings row's trailing button::

        path = choose_model(self, kinds=("cellpose",))
        if path:
            field.setText(path)

    :param parent: the widget opening the dialog.
    :param kinds: restrict to these model kinds.
    :returns: a local filesystem path, or ``None`` when cancelled.
    """
    dialog = ModelZooPicker(kinds=kinds, parent=parent)
    if dialog.exec() == QDialog.Accepted:
        return dialog.chosen_path()
    return None
