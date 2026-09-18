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
from types import SimpleNamespace
from typing import List, Optional

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (QAbstractItemView, QDialog, QDialogButtonBox,
                               QFileDialog, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QMessageBox, QProgressBar,
                               QComboBox, QPushButton, QTableWidget,
                               QTextBrowser,
                               QVBoxLayout, QWidget)

from .sortable_table import install_sorting, table_item

#: Where checkpoints land unless the user says otherwise.
DEFAULT_MODEL_DIR = os.path.join(os.path.expanduser("~"), ".spacr", "models")

#: QSettings key remembering the chosen folder.
_DIR_SETTING = "model_zoo/download_dir"

_COLUMNS = ("Model", "Kind", "Trained on", "Status", "Version")


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


class _DownloadWorker(QObject):
    """Fetch one model off the GUI thread.

    WHY A THREAD AT ALL. ``model_zoo.fetch`` streams a file that is 1.2 GB for
    the Toxoplasma models. Called from a button handler it blocks the event
    loop for minutes: the dialog stops repainting, the progress bar cannot
    move, and the compositor offers to force-quit spaCR -- the same failure
    a slow screen build causes, arriving through a dialog instead.

    The worker owns nothing Qt-visual. It emits numbers; the dialog draws.

    :param entry: the model-zoo entry to fetch.
    :param folder: where to install it.
    :param unverified: skip the checksum requirement. ONLY EVER TRUE FOR AN
        ENTRY THAT PUBLISHES NO CHECKSUM, and only after the dialog has told
        the user that spaCR cannot then tell a truncated or substituted file
        from the real one and they have said to go ahead. It is not a retry
        knob for a checksum that failed.
    """

    progressed = Signal(int, int)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, entry, folder: str, *, unverified: bool = False):
        """Hold what to fetch, where to put it, and whether to skip the checksum."""
        super().__init__()
        self._entry = entry
        self._folder = folder
        self._unverified = bool(unverified)

    def run(self) -> None:
        """Do the fetch, reporting as it goes."""
        from ... import model_zoo

        try:
            path = model_zoo.fetch(
                self._entry, self._folder,
                require_checksum=not self._unverified,
                progress=lambda done, total: self.progressed.emit(
                    int(done), int(total or 0)))
        except Exception as exc:                            # noqa: BLE001
            self.failed.emit(str(exc))
        else:
            self.finished.emit(str(path))


def _human_bytes_local(size: float) -> str:
    """A byte count a person can read.

    model_zoo has its own ``_human_bytes``; this does not import it, because
    that module pulls the whole zoo -- and its network paths -- onto the GUI
    thread to format a number during a repaint that happens five times a
    second.
    """
    value = float(size)
    for unit in ("B", "kB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} GB"


def _human_rate(bytes_per_second: float) -> str:
    """A transfer rate a person can read."""
    for unit in ("B/s", "kB/s", "MB/s", "GB/s"):
        if bytes_per_second < 1024 or unit == "GB/s":
            return f"{bytes_per_second:.1f} {unit}"
        bytes_per_second /= 1024.0
    return f"{bytes_per_second:.1f} GB/s"


def _human_eta(seconds: float) -> str:
    """A remaining time a person can read.

    Says "estimating…" rather than a number until there is enough of a
    transfer to divide by: an ETA computed from the first chunk is wrong by
    minutes and reads as a promise.
    """
    if seconds <= 0 or seconds != seconds or seconds > 86400:
        return "estimating…"
    if seconds < 60:
        return f"{int(seconds)}s left"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s left"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m left"


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
        """Build the model zoo dialog.

        :param kinds: restrict the listing to these model kinds; ``None`` lists
            everything spaCR knows about.
        :param parent: parent widget, or ``None``.
        """
        super().__init__(parent)
        self.setWindowTitle("Model zoo")
        from ..preferences import scaled_px
        
        self.setMinimumWidth(scaled_px(720))
        self._kinds = tuple(kinds) if kinds else None
        self._entries: List = []
        self._chosen_path: Optional[str] = None

        layout = QVBoxLayout(self)

        blurb = QLabel(
            "Models spaCR knows about. A model already on this machine can be "
            "used straight away; the rest are downloaded when you ask for one.")
        blurb.setWordWrap(True)
        layout.addWidget(blurb)

        self._groups: list = []
        self._chosen: dict = {}
        self._rebuilding = False
        self.table = QTableWidget(0, len(_COLUMNS), self)
        install_sorting(self.table)
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        layout.addWidget(self.table, 1)

        # The scorecard sits between the list and the controls, at a fixed
        # height: a box that grew and shrank with the selected model would
        # move the Download button under the pointer between clicks.
        self.card = QTextBrowser(self)
        self.card.setOpenExternalLinks(True)
        self.card.setFixedHeight(200)
        layout.addWidget(self.card)

        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel("Save to:"))
        self.folder_edit = QLineEdit(remembered_model_dir(), self)
        folder_row.addWidget(self.folder_edit, 1)
        browse = QPushButton("Browse…", self)
        browse.clicked.connect(self._browse)
        folder_row.addWidget(browse)
        add = QPushButton("Add…", self)
        add.setToolTip("List a model file you already have, and optionally "
                       "share it on Hugging Face so others can use it.")
        add.clicked.connect(self._add_model)
        folder_row.addWidget(add)
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
        self._warm_the_community_catalogue()
        from ..screens.settings_model import retarget_field_tooltips
        retarget_field_tooltips(self)


    #: The stock Cellpose model, offered as a zoo row.
    #:
    #: It is not a download and has no checkpoint: choosing it writes the
    #: literal string "cpsam" into the field, which is what Cellpose 4 loads
    #: by default and what `_resolve_cellpose_pretrained` passes through
    #: untouched. Offered here because the picker is where a user goes to
    #: CHANGE a model, and "put it back to the standard one" is the commonest
    #: thing they want -- without this row the only way back is to remember
    #: the spelling and type it.
    STOCK_MODEL = SimpleNamespace(
        key="cpsam_v2",
        name="cpsam",
        kind="cellpose",
        path="cpsam",
        sha256="stock",
        uri="",
        source="stock",
        trained_on=("Cellpose 4's own general model. No download: choosing "
                    "this writes 'cpsam' into the field."),
        trained_by="Cellpose",
        notes=(),
    )

    def _warm_the_community_catalogue(self) -> None:
        """Fetch the community rows off the GUI thread, then redraw.

        WHY THIS EXISTS. :func:`spacr.model_zoo.shared_catalogue` refuses to
        wait for the network when it is called on the GUI thread -- measured:
        an unreachable catalogue host froze a module open for
        32.2 s and produced the desktop's "force quit" dialog. :meth:`refresh`
        therefore comes back with whatever is cached, which on the first
        picker of a session is nothing.

        So the fetch happens here instead, on a worker, and the table is
        rebuilt when it lands. Nothing is lost and nobody waits.
        """
        from ... import model_zoo

        if not model_zoo.shared_catalogue_is_stale():
            return
        try:
            from ..job_runner import JobRunner
        except Exception:                                    # noqa: BLE001
            return
        self._catalogue_job = JobRunner(self, app_key="model zoo",
                                        user_visible=False)
        self._catalogue_job.submit(
            lambda: model_zoo.shared_catalogue(block=True),
            lambda _entries: self.refresh())

    def refresh(self) -> None:
        """Reload the catalogue and redraw the table.

        Answers from the shared catalogue's cache rather than the network --
        see :meth:`_warm_the_community_catalogue`.
        """
        from ... import model_zoo

        try:
            entries = [self.STOCK_MODEL]
            # The catalogue lists the Cellpose stock models too. Duplicates
            # are collapsed per version label by group_entries, which catches
            # the stock row whose key and name disagree -- name "cpsam", key
            # "cpsam_v2" -- where a name comparison here did not.
            entries += list(model_zoo.catalogue(remote=True, block=False))
        except Exception as exc:                            # noqa: BLE001
            self.status.setText(f"Could not read the model list: {exc}")
            entries = [self.STOCK_MODEL]
        if self._kinds:
            entries = [e for e in entries if e.kind in self._kinds]
        self._entries = entries

        self._rebuild(entries)

    def _rebuild(self, entries) -> None:
        """Redraw the table from this list of entries.

        Split out of :meth:`refresh` so a caller -- a test, mainly -- can list
        entries of its own choosing without going through the catalogue.
        """
        from ..screens.model_zoo import group_entries

        self._entries = list(entries)

        # Tear the old rows down FIRST. A combo box from the previous refresh
        # is still wired to _version_picked, and setRowCount destroying it can
        # emit currentIndexChanged against groups that no longer exist.
        self._rebuilding = True
        self.table.clearContents()
        self.table.setRowCount(0)
        self._groups = group_entries(entries)
        self._chosen = {stem: 0 for stem, _ in self._groups}

        self.table.setRowCount(len(self._groups))
        for row, (stem, pairs) in enumerate(self._groups):
            combo = QComboBox(self.table)
            combo.addItems([label for label, _ in pairs])
            combo.setCurrentIndex(0)
            combo.currentIndexChanged.connect(
                lambda index, r=row: self._version_picked(r, index))
            self.table.setCellWidget(row, 4, combo)
            self._fill_row(row)
        self._rebuilding = False
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Stretch)
        self._selection_changed()

    def _fill_row(self, row: int) -> None:
        """Write a row's cells for the version it currently shows."""
        stem, pairs = self._groups[row]
        entry = pairs[self._chosen[stem]][1]
        local = self._local_path(entry)
        cells = (
            stem,
            entry.kind,
            (entry.trained_on or "")[:160],
            "on this machine" if local else "not downloaded",
        )
        from ... import model_zoo

        tip = model_zoo.scorecard_html(entry)
        for column, text in enumerate(cells):
            item = table_item(str(text))
            if tip:
                item.setToolTip(tip)
            elif column == 3 and local:
                item.setToolTip(local)
            self.table.setItem(row, column, item)

    def _version_picked(self, row: int, index: int) -> None:
        """A different version was chosen: this row now means another model.

        The status cell has to be rewritten too -- v1 may be on this machine
        while v2 is not, and a stale "on this machine" would send the user to
        a file that is not there.
        """
        if self._rebuilding or not (0 <= row < len(self._groups)):
            return
        stem, pairs = self._groups[row]
        self._chosen[stem] = max(0, min(int(index), len(pairs) - 1))
        self._fill_row(row)
        self._selection_changed()

    def _add_model(self) -> None:
        """List a checkpoint the user already has, then offer to share it.

        The model is usable immediately whether or not it is ever uploaded --
        listing and sharing are separate steps, because most users adding a
        model want to USE it, not publish it.
        """
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        from ... import model_zoo

        path, _ = QFileDialog.getOpenFileName(
            self, "Add a model", self.folder_edit.text().strip() or "",
            "Models (*.pth *.pt *.CP_model *.safetensors);;All files (*)")
        if not path:
            return
        try:
            entry = model_zoo.entry_from_file(path)
        except Exception as exc:                            # noqa: BLE001
            self.status.setText(f"Not a model this can read: {exc}")
            return
        self._rebuild(list(self._entries) + [entry])
        self.status.setText(f"Listed {os.path.basename(path)}. It is usable now.")

        if QMessageBox.question(
                self, "Share this model?",
                "Share this model on Hugging Face so other spaCR users can "
                "download it?\n\nYou will be asked for the numbers that go in "
                "its scorecard. Uploading uses YOUR Hugging Face login.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No) != QMessageBox.Yes:
            return
        self._share_model(path)

    def _share_model(self, path: str) -> None:
        """Collect the scorecard and upload, using the uploader's own token."""
        from PySide6.QtWidgets import QMessageBox

        from . import model_share
        from .model_share_dialog import ShareDialog

        token = model_share.find_token()
        if not token:
            QMessageBox.information(
                self, "Hugging Face login needed",
                "No Hugging Face token was found, so nothing was uploaded.\n\n"
                "Run `huggingface-cli login`, or set HF_TOKEN, and press Add "
                "again. spaCR does not ship a token of its own: one that could "
                "upload could also delete, and it would be readable by anyone "
                "who installs spaCR.")
            return
        dialog = ShareDialog(os.path.basename(path), self)
        if not dialog.exec():
            return
        self.status.setText("Uploading to Hugging Face…")
        try:
            url = model_share.share(path, dialog.values(), token)
        except Exception as exc:                            # noqa: BLE001
            QMessageBox.warning(
                self, "Upload failed",
                f"{exc}\n\nIf this says you may not write there, ask the owner "
                f"of {model_share.SHARE_REPO} for write access, or it will be "
                "published under your own account instead.")
            self.status.setText("Upload failed.")
            return
        self.status.setText(f"Shared: {url}")
        QMessageBox.information(self, "Shared", f"Uploaded to\n{url}")

    def _local_path(self, entry) -> Optional[str]:
        """Where this entry already is, or the name Cellpose resolves itself."""
        if getattr(entry, "source", "") == "stock":
            return str(entry.path)
        return self._local_path_on_disk(entry)

    def _local_path_on_disk(self, entry) -> Optional[str]:
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
        if not (0 <= row < len(self._groups)):
            return None
        stem, pairs = self._groups[row]
        return pairs[self._chosen[stem]][1]


    def _selection_changed(self) -> None:
        """Describe the selected model and enable the action that applies to it.

        An entry publishing no checksum says so *before* the click: the fetch
        refuses what it cannot verify, so without this the button is enabled,
        pressing it fails, and the message explains a policy the user had no way
        to see. Accepting it is still possible -- but as a choice, made
        knowingly.
        """
        entry = self.selected_entry()
        local = self._local_path(entry) if entry else None
        self.use_button.setEnabled(bool(local))
        self.download_button.setEnabled(bool(entry) and not local)
        self._show_card(entry)
        if entry is not None and not getattr(entry, "sha256", ""):
            self.status.setText(
                "This model publishes no checksum, so a truncated or "
                "substituted file could not be told from the real one. "
                "Downloading it will ask you to accept that.")
        else:
            self.status.setText("")

    def _show_card(self, entry) -> None:
        """The selected model's scorecard, and a link to its full page.

        The table replaces the sentence that used to sit here. A sentence
        cannot be compared between two models; a table can, and it is the same
        table the Hugging Face card prints.
        """
        from ... import model_zoo

        if entry is None:
            self.card.setHtml("")
            return
        html = model_zoo.scorecard_html(entry)
        if not html:
            # The stock model is a SimpleNamespace, not a ModelEntry, so it has
            # no describe(); fall back to what any entry-shaped object has.
            describe = getattr(entry, "describe", None)
            if callable(describe):
                html = f"<p>{describe()}</p>"
            else:
                name = getattr(entry, "display_name", "") or getattr(entry, "name", "")
                html = (f"<p><b>{name}</b></p>"
                        f"<p>{getattr(entry, 'trained_on', '') or ''}</p>")
        url = getattr(entry, "model_card_url", "")
        if url:
            html += f'<p><a href="{url}">{url}</a></p>'
        self.card.setHtml(html)

    def _browse(self) -> None:
        """Ask where downloaded checkpoints should live, and remember the answer."""
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

        import time

        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.progress.setFormat("%p%")
        self.status.setText(f"Downloading {entry.name}…")
        self.download_button.setEnabled(False)
        self.use_button.setEnabled(False)
        unverified = not getattr(entry, "sha256", "")
        if unverified:
            answer = QMessageBox.question(
                self, "Model zoo",
                f"{entry.name} publishes no checksum.\n\n"
                "spaCR cannot tell a truncated or substituted file from the "
                "real one, so it normally refuses to install it. Download it "
                "anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                self.progress.setVisible(False)
                self.download_button.setEnabled(True)
                self.status.setText("Download cancelled.")
                return
        self._folder_for_download = folder
        self._started_at = time.monotonic()
        self._last_emit = 0.0

        self._thread = QThread(self)
        self._worker = _DownloadWorker(entry, folder,
                                       unverified=unverified)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progressed.connect(self._on_progress)
        self._worker.finished.connect(self._on_download_finished)
        self._worker.failed.connect(self._on_download_failed)
        self._thread.start()

    def _on_progress(self, done: int, total: int) -> None:
        """Draw percent, speed and time remaining.

        Throttled to about five updates a second. A progress signal per 64 kB
        chunk on a gigabyte file is sixteen thousand repaints, which costs more
        than the download and makes the bar juddery rather than smooth.
        """
        import time

        now = time.monotonic()
        if total and now - self._last_emit < 0.2 and done < total:
            return
        self._last_emit = now
        elapsed = max(now - self._started_at, 1e-6)
        rate = done / elapsed

        if total > 0:
            self.progress.setRange(0, 100)
            self.progress.setValue(int(done * 100 / total))
            remaining = (total - done) / rate if rate > 0 else -1
            self.status.setText(
                f"{_human_bytes_local(done)} of {_human_bytes_local(total)}  ·  "
                f"{_human_rate(rate)}  ·  {_human_eta(remaining)}")
        else:
            self.progress.setRange(0, 0)
            self.status.setText(
                f"{_human_bytes_local(done)}  ·  {_human_rate(rate)}  ·  "
                f"size unknown")

    def _finish_download(self, outcome: str) -> None:
        """Common teardown for both download outcomes."""
        self.progress.setVisible(False)
        thread = getattr(self, "_thread", None)
        if thread is not None:
            thread.quit()
            thread.wait(5000)
            self._thread = None
            self._worker = None
        self.refresh()
        if outcome:
            self.status.setText(outcome)

    def _on_download_finished(self, path: str) -> None:
        """Remember the folder used and report the finished download.

        :param path: where the checkpoint landed.
        """
        _remember_model_dir(getattr(self, "_folder_for_download", "") or path)
        self._finish_download(f"Downloaded to {path}")

    def _on_download_failed(self, message: str) -> None:
        """Report a failed download in a dialog as well as on the status line.

        Named rather than swallowed: the fetch refuses an entry whose checksum
        does not match, and that refusal is the most important message this
        dialog can carry -- it means the bytes are not the model.

        :param message: the failure text.
        """
        QMessageBox.warning(self, "Model zoo",
                            f"Could not download:\n{message}")
        self._finish_download(f"Download failed: {message}")

    def _accept_selected(self) -> None:
        """Announce the selected model's local path and close.

        A model that is not on this machine yet does nothing: there is no path
        to hand back.
        """
        entry = self.selected_entry()
        local = self._local_path(entry) if entry else None
        if not local:
            return
        self._chosen_path = local
        self.model_chosen.emit(local)
        self.accept()

    def _stop_any_download(self) -> None:
        """Stop and join a running download thread.

        A QThread destroyed while it is still running takes the process with
        it -- Qt aborts rather than unwinding. So closing this dialog during a
        1.2 GB download, which is exactly when a user would close it, has to
        wait for the worker rather than let Python drop the last reference to
        a live thread. This crashed the test suite before it could crash a
        user, which is the only reason it was found here.
        """
        thread = getattr(self, "_thread", None)
        if thread is None:
            return
        try:
            if thread.isRunning():
                thread.quit()
                thread.wait(10000)
        except RuntimeError:
            pass
        self._thread = None
        self._worker = None

    def closeEvent(self, event):                            # noqa: N802
        """Join the download before the dialog goes away."""
        self._stop_any_download()
        super().closeEvent(event)

    def reject(self) -> None:
        """Cancel closes the dialog; it must not leave a thread behind."""
        self._stop_any_download()
        super().reject()

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
