"""
Model Zoo — every model this machine can segment or classify with, in one list.

The question "which models do I have, where did they come from, and does this
one work on my images" is currently answered with ``find``, memory and a full
plate run. This screen answers it in a list, a provenance card and three
fields.

Layout::

    ┌──────────────────────────────────────────────────────────────────────┐
    │ /data/screen1                       [Choose folder…] [Scan]          │
    ├──────────────────────────────────────────────────────────────────────┤
    │ model            kind      source  v  size    checksum  trained on   │
    │ toxo_plaque…     cellpose  bundled 1  25 MB   none      /nas_mnt/…   │
    │ maxvit_t_epoch…  classifier local  1  310 MB  none      /data/…      │
    │ hela_60x.CP…     cellpose  remote  1  26 MB   published unknown      │
    ├──────────────────────────────────────────────────────────────────────┤
    │ hela_60x_confluent.CP_model  [cellpose · remote · v1]                │
    │   uri        https://…                                               │
    │   sha256     9f86d0…  (published)                                    │
    │   trained on unknown                                                 │
    │   ! this model does not say what it was trained on…                  │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Download to [~/.spacr/models    ] [Choose…] [Download] [Cancel]      │
    │ [====================              ] 12.4 MB / 26.5 MB               │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Test on [3] fields from [/data/screen1/plate1/1] [Choose…]  [Test]   │
    │ field    objects  seg_qc  flags                    ┌───────────────┐ │
    │ A01_f01  212      ok      -                        │  [ masks ]    │ │
    │ A01_f02  198      warn    high_border_fraction     └───────────────┘ │
    └──────────────────────────────────────────────────────────────────────┘

Design notes:

* **Provenance is a column, not a tooltip.** "What was this trained on" is what
  decides whether a model applies to your images at all, so it is in the table
  and spelled ``unknown`` where it is unknown — a blank cell reads as "no
  constraints", which is the opposite of what it means.
* **Downloads are off the GUI thread, atomic, and cancellable.** The bytes go
  to a temp file inside the destination and are renamed only after the checksum
  passes (:func:`spacr.model_zoo.fetch`), so Cancel leaves the destination
  exactly as it was — no half-written file at a name that looks like a model.
* **``worker.finished`` is relayed through a signal into a bound method.**
  PySide6 delivers a plain closure connected to a worker's signal as a *direct*
  call on the worker thread, and the completion handler here fills a
  QPlainTextEdit and a QTableWidget. Building QTextDocument children off the
  GUI thread is undefined behaviour, so ``finished`` chains through
  :attr:`ModelZooScreen._job_settled` — a widget-affine bound method — exactly
  as ``PlateViewScreen`` does.
* **No modal dialogs on any error path.** A folder with no models, a checksum
  mismatch, a corrupt checkpoint, a cancelled download: all inline. A
  QMessageBox hangs a headless run.
* **Two selected models hand off to Model Compare.** Benchmarking each one
  separately and reading the two tables is not a comparison — the A/B harness
  in :mod:`spacr.model_compare` is, and it is the one that says out loud that
  neither model is ground truth. This screen builds a configured
  ``ModelCompareScreen`` and emits it rather than growing a second, weaker
  comparison of its own.
* **Benchmarks are never sorted across field sets.** The results table belongs
  to one model on one set of fields, and the summary line names the field set,
  because a score on your three fields says nothing about a score on anybody
  else's.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from ..widgets.toggle import Toggle

from ... import model_zoo as zoo
from ..bridge import make_thread
from ..theme import SPACING, active_palette
from ..widgets import Divider

__all__ = ["ModelZooScreen", "DEFAULT_DOWNLOAD_DIR", "FIELD_RANGE",
           "PREVIEW_PX", "compose_labels"]


#: Fields the "test on N fields" run will take. Three by default because three
#: is what a human looks at; the ceiling stops an interactive question turning
#: into an overnight run by accident.
FIELD_RANGE = (1, 25)

#: Edge of the mask preview, in pixels.
PREVIEW_PX = 320

#: RGB for a segmented object in the preview. One colour, deliberately: a
#: single model's labels have no correspondence to colour by, and a random
#: per-label palette would invite comparing label ids between two runs.
COLOUR_OBJECT = (46, 196, 182)

#: Where a downloaded model goes unless the user says otherwise.
DEFAULT_DOWNLOAD_DIR = os.path.join(os.path.expanduser("~"), ".spacr", "models")

_ZOO_HEADERS = ("model", "kind", "source", "v", "size", "checksum",
                "trained on", "trained by")

_BENCH_HEADERS = ("field", "objects", "seg_qc", "flags")


def _cell(text: str) -> QTableWidgetItem:
    """A read-only table cell."""
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    return item


def compose_labels(image: Optional[np.ndarray], mask: Any,
                   alpha: float = 0.45) -> Optional[np.ndarray]:
    """Blend one model's mask over its field, in one colour.

    The greyscale backdrop comes from
    :func:`spacr.qt.screens.model_compare.to_display_gray` — the same 1-99.9
    percentile stretch the comparison screen uses, so the same field looks the
    same in both places.

    One colour rather than a palette: a single segmentation has no
    correspondence to encode, and colouring by label id across two screens
    invites reading label 3 here as label 3 there.

    :param image: the field, or None for a black backdrop.
    :param mask: a 2-D label image.
    :param alpha: overlay strength.
    :returns: a uint8 ``(H, W, 3)`` array, or None when the mask is not a 2-D
        label image.
    """
    from .model_compare import to_display_gray

    labels = np.squeeze(np.asarray(mask))
    if labels.ndim != 2 or labels.size == 0:
        return None
    labels = np.rint(labels).astype(np.int64)
    if labels.min() < 0:
        return None

    gray = to_display_gray(image, labels.shape)
    rgb = np.stack((gray,) * 3, axis=-1).astype(np.float32)
    colour = np.array(COLOUR_OBJECT, dtype=np.float32)
    foreground = labels > 0
    out = np.where(foreground[..., None],
                   rgb * (1.0 - alpha) + colour * alpha, rgb)
    return np.clip(out, 0, 255).astype(np.uint8)


class ModelZooScreen(QWidget):
    """Browse, verify, download and bench the models spaCR can run.

    :param parent: Qt parent.
    :param threaded: run scans, downloads and benchmarks on worker threads (the
        default). Tests pass ``False`` to run them inline; both paths emit the
        same signals in the same order.
    :ivar last_error: text of the most recent failure, ``''`` after a success.
        Errors go here and to the inline status label — never to a dialog.
    """

    #: emitted with the number of models after every scan
    models_listed = Signal(int)
    #: emitted after every job settles (scan, download or benchmark)
    job_finished = Signal(bool)
    #: emitted with ``(ok, path)`` when a download settles; ``path`` is ``''``
    #: on failure or cancellation, and nothing was written in that case
    download_finished = Signal(bool, str)
    #: emitted with ``{'model_a', 'model_b', 'folder', 'n_fields'}`` when the
    #: user asks to compare two selected models. The host opens Model Compare;
    #: :meth:`build_comparison_screen` builds the configured screen for it.
    compare_requested = Signal(dict)

    #: private. Re-emitted from ``PipelineWorker.finished`` purely to hop the
    #: completion handler onto the GUI thread — see the module docstring.
    _job_settled = Signal(bool)
    #: private. Progress from the worker thread, likewise.
    _progress_ticked = Signal(int, int)
    _progress_said = Signal(str)

    def __init__(self, parent=None, threaded: bool = True):
        super().__init__(parent)
        self._threaded = bool(threaded)
        self._entries: List[zoo.ModelEntry] = []
        self._result: Optional[zoo.BenchmarkResult] = None
        self._images: List[np.ndarray] = []
        self._field_names: List[str] = []
        self._fields_folder: str = ""
        self._segment_fn: Optional[Callable] = None
        self._opener: Optional[Callable] = None
        self._busy = False
        self._cancel = {"stop": False}
        # Strong references to in-flight (QThread, worker) pairs: a QThread
        # garbage-collected while still running takes the process down.
        self._jobs: List[tuple] = []
        self._pending: List[tuple] = []
        self._error_handler: Optional[Callable[[str], None]] = None
        self.last_error: str = ""

        self._build_ui()
        from ..dnd import install_dropzone
        from ..dnd_handlers import get_handler
        install_dropzone(self, get_handler("model_zoo"), self)
        self._job_settled.connect(self._on_job_settled)
        self._progress_ticked.connect(self._on_progress)
        self._progress_said.connect(self._on_progress_text)
        self._set_status(
            "Scan a folder for checkpoints, or pick a catalogue entry to "
            "download. Every model shows what it was trained on — 'unknown' "
            "means nobody recorded it, not that it fits anything.")
        self._update_controls()

    # -- construction ------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                 SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["md"])

        title = QLabel("Model Zoo")
        title.setObjectName("DisplayHeading")
        outer.addWidget(title)

        subtitle = QLabel(
            "Every Cellpose and classifier checkpoint this machine can reach, "
            "with what it was trained on, whether its bytes check out, and "
            "what it does to three of your fields.")
        subtitle.setObjectName("Muted")
        subtitle.setWordWrap(True)
        outer.addWidget(subtitle)
        outer.addWidget(Divider())

        # ── scan row ──────────────────────────────────────────────────
        scan_row = QHBoxLayout()
        scan_row.setSpacing(SPACING["sm"])
        self._scan_edit = QLineEdit(self)
        self._scan_edit.setPlaceholderText(
            "…/screen1  — a folder to scan for .CP_model / .pth checkpoints")
        self._scan_edit.setClearButtonEnabled(True)
        self._scan_edit.returnPressed.connect(self._on_scan_typed)
        self._btn_pick_scan = QPushButton("Choose folder…", self)
        self._btn_pick_scan.clicked.connect(self._pick_scan_folder)
        self._btn_scan = QPushButton("Scan", self)
        self._btn_scan.clicked.connect(self._on_scan_typed)
        scan_row.addWidget(self._scan_edit, 1)
        scan_row.addWidget(self._btn_pick_scan)
        scan_row.addWidget(self._btn_scan)
        outer.addLayout(scan_row)

        # ── the listing ───────────────────────────────────────────────
        self._table = QTableWidget(0, len(_ZOO_HEADERS), self)
        self._table.setHorizontalHeaderLabels(list(_ZOO_HEADERS))
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Interactive)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        outer.addWidget(self._table, 1)

        # ── provenance card ───────────────────────────────────────────
        self._detail = QPlainTextEdit(self)
        self._detail.setReadOnly(True)
        self._detail.setMaximumHeight(150)
        self._detail.setPlaceholderText(
            "Select a model to see where it came from and what it was trained "
            "on.")
        outer.addWidget(self._detail)

        # ── download ──────────────────────────────────────────────────
        download = QGroupBox("Download", self)
        dl = QVBoxLayout(download)
        dl.setSpacing(SPACING["xs"])
        row = QHBoxLayout()
        row.setSpacing(SPACING["sm"])
        self._dest_edit = QLineEdit(DEFAULT_DOWNLOAD_DIR, download)
        self._dest_edit.setToolTip(
            "(str) Where a downloaded model is written. Never overwritten: a "
            "second copy of the same name lands as name_v2.")
        self._btn_pick_dest = QPushButton("Choose…", download)
        self._btn_pick_dest.clicked.connect(self._pick_dest_folder)
        self._btn_download = QPushButton("Download", download)
        self._btn_download.setObjectName("PrimaryButton")
        self._btn_download.clicked.connect(self.download_selected)
        self._btn_cancel = QPushButton("Cancel", download)
        self._btn_cancel.clicked.connect(self.cancel_download)
        row.addWidget(QLabel("to", download))
        row.addWidget(self._dest_edit, 1)
        row.addWidget(self._btn_pick_dest)
        row.addWidget(self._btn_download)
        row.addWidget(self._btn_cancel)
        dl.addLayout(row)

        self._allow_unverified = Toggle(
            "Accept a model with no published checksum", download)
        self._allow_unverified.setToolTip(
            "(bool) Off by default. A download nobody can check against a "
            "published hash could be truncated or substituted, and a wrong "
            "checkpoint still loads and still produces masks — just not the "
            "author's.")
        dl.addWidget(self._allow_unverified)

        self._progress = QProgressBar(download)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        dl.addWidget(self._progress)
        outer.addWidget(download)

        # ── benchmark ─────────────────────────────────────────────────
        test = QGroupBox("Test on fields", self)
        tl = QVBoxLayout(test)
        tl.setSpacing(SPACING["xs"])
        row = QHBoxLayout()
        row.setSpacing(SPACING["sm"])
        self._fields_box = QSpinBox(test)
        self._fields_box.setRange(*FIELD_RANGE)
        self._fields_box.setValue(zoo.DEFAULT_N_FIELDS)
        self._fields_box.setToolTip(
            "(int) How many fields to segment with the selected model.")
        self._fields_box.valueChanged.connect(lambda *_: self._reload_fields())
        self._fields_edit = QLineEdit(test)
        self._fields_edit.setPlaceholderText(
            "…/plate1/1  — a folder of .tif / .png / .npy / .npz fields")
        self._fields_edit.setClearButtonEnabled(True)
        self._fields_edit.returnPressed.connect(self._on_fields_typed)
        self._btn_pick_fields = QPushButton("Choose…", test)
        self._btn_pick_fields.clicked.connect(self._pick_fields_folder)
        self._btn_test = QPushButton("Test on 3 fields", test)
        self._btn_test.clicked.connect(self.run_benchmark)
        self._btn_compare = QPushButton("Compare the two selected", test)
        self._btn_compare.clicked.connect(self.compare_selected)
        row.addWidget(QLabel("fields", test))
        row.addWidget(self._fields_box)
        row.addWidget(self._fields_edit, 1)
        row.addWidget(self._btn_pick_fields)
        row.addWidget(self._btn_test)
        row.addWidget(self._btn_compare)
        tl.addLayout(row)

        split = QSplitter(Qt.Horizontal, test)
        self._bench_table = QTableWidget(0, len(_BENCH_HEADERS), split)
        self._bench_table.setHorizontalHeaderLabels(list(_BENCH_HEADERS))
        self._bench_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._bench_table.verticalHeader().setVisible(False)
        self._bench_table.horizontalHeader().setStretchLastSection(True)
        self._bench_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._bench_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._bench_table.currentCellChanged.connect(
            lambda row, *_: self.select_field(row))
        split.addWidget(self._bench_table)

        self._preview = QLabel("", split)
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setMinimumSize(PREVIEW_PX, PREVIEW_PX)
        self._preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._preview.setStyleSheet(
            f"background: {active_palette()['bg']};")
        split.addWidget(self._preview)
        split.setSizes([700, 400])
        tl.addWidget(split, 1)

        self._summary = QLabel("", test)
        self._summary.setWordWrap(True)
        self._summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        tl.addWidget(self._summary)
        outer.addWidget(test, 1)

        self._status = QLabel("", self)
        self._status.setObjectName("Muted")
        self._status.setWordWrap(True)
        self._status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        outer.addWidget(self._status)

    # -- status ------------------------------------------------------------

    def _set_status(self, text: str, error: bool = False) -> None:
        """Report inline. Never a QMessageBox — a modal hangs a headless run."""
        self.last_error = text if error else ""
        palette = active_palette()
        colour = palette["error"] if error else palette["fg_muted"]
        self._status.setStyleSheet(f"color: {colour};")
        self._status.setText(text)

    def status_text(self) -> str:
        """The inline status message (test/introspection helper)."""
        return self._status.text()

    def summary_text(self) -> str:
        """The benchmark summary line, or ``''``."""
        return self._summary.text()

    def detail_text(self) -> str:
        """The provenance card for the selected model, or ``''``."""
        return self._detail.toPlainText()

    # -- listing -----------------------------------------------------------

    def entries(self) -> List[zoo.ModelEntry]:
        """Everything currently listed, in table order."""
        return list(self._entries)

    def set_entries(self, entries) -> None:
        """Replace the listing (used by the scan, and directly by tests)."""
        self._entries = list(entries)
        table = self._table
        table.blockSignals(True)
        table.setRowCount(len(self._entries))
        for r, entry in enumerate(self._entries):
            cells = (
                entry.name,
                entry.kind,
                entry.source,
                entry.version,
                zoo._human_bytes(entry.size_bytes),
                entry.checksum_state,
                entry.trained_on,
                entry.trained_by,
            )
            for c, text in enumerate(cells):
                item = _cell(str(text))
                if c == 6 and not entry.provenance_known:
                    # Never blank, and never quiet: 'unknown' in the warning
                    # colour, because a model with no provenance is the one you
                    # are most likely to misapply.
                    item.setForeground(_brush(active_palette()["warning"]))
                if c == 5 and entry.checksum_state == "none":
                    item.setForeground(_brush(active_palette()["warning"]))
                item.setToolTip(entry.describe())
                table.setItem(r, c, item)
        table.blockSignals(False)
        table.resizeColumnsToContents()
        self.models_listed.emit(len(self._entries))
        self._update_controls()

    def rows(self) -> List[List[str]]:
        """The listing as plain strings."""
        return [[(self._table.item(r, c).text() if self._table.item(r, c)
                  else "")
                 for c in range(self._table.columnCount())]
                for r in range(self._table.rowCount())]

    def scan(self, folder: Optional[str] = None,
             include_catalogue: bool = True) -> bool:
        """List the catalogue plus every checkpoint under ``folder``.

        Never raises and never opens a dialog: a mistyped path, a folder with
        no models, an unreadable tree all land in the status label.

        :param folder: folder to scan; None uses whatever is in the box.
        :param include_catalogue: also list the bundled and declared entries.
        :returns: for the synchronous path, whether anything was listed; for
            the threaded path, True once the job started.
        """
        target = folder if folder is not None else self._scan_edit.text()
        target = str(target or "").strip()
        if folder is not None:
            self._scan_edit.setText(target)
        if target and not os.path.isdir(target):
            self._set_status(f"No such folder: {target}", error=True)
            self._update_controls()
            return False

        def _job() -> List[zoo.ModelEntry]:
            found = list(zoo.catalogue()) if include_catalogue else []
            have = {e.path for e in found if e.path}
            if target:
                for entry in zoo.discover_local(target):
                    if entry.path not in have:
                        found.append(entry)
            return found

        self._set_status(
            f"Scanning {target}…" if target else "Reading the catalogue…")
        return self._run_job(_job, self._apply_scan)

    def _apply_scan(self, entries: List[zoo.ModelEntry]) -> None:
        self.set_entries(entries)
        unknown = sum(1 for e in entries if not e.provenance_known)
        self._set_status(
            f"{len(entries)} model(s): "
            f"{sum(1 for e in entries if e.kind == 'cellpose')} Cellpose, "
            f"{sum(1 for e in entries if e.kind == 'classifier')} classifier. "
            + (f"{unknown} do not record what they were trained on."
               if unknown else
               "Every one records what it was trained on."))

    def _on_scan_typed(self) -> None:
        self.scan(self._scan_edit.text())

    def _pick_scan_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Choose a folder to scan for models", os.getcwd())
        if path:
            self.scan(path)

    def _pick_dest_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Choose where downloaded models go",
            self._dest_edit.text() or DEFAULT_DOWNLOAD_DIR)
        if path:
            self._dest_edit.setText(path)

    # -- selection ---------------------------------------------------------

    def selected_rows(self) -> List[int]:
        """Indices of the selected rows, in order."""
        return sorted({i.row() for i in self._table.selectedIndexes()})

    def selected_entries(self) -> List[zoo.ModelEntry]:
        """The selected models, in table order."""
        return [self._entries[r] for r in self.selected_rows()
                if 0 <= r < len(self._entries)]

    def select(self, *rows: int) -> None:
        """Select these rows (test/programmatic helper).

        Goes through the selection model rather than ``selectRow`` because the
        latter clears the selection first in ExtendedSelection mode, so
        ``select(0, 1)`` would leave exactly one row selected — and "two
        selected models" is the whole input to the compare hand-off.
        """
        from PySide6.QtCore import QItemSelectionModel

        selection = self._table.selectionModel()
        self._table.clearSelection()
        for row in rows:
            if 0 <= row < self._table.rowCount():
                selection.select(
                    self._table.model().index(row, 0),
                    QItemSelectionModel.Select | QItemSelectionModel.Rows)
        self._on_selection_changed()

    def _on_selection_changed(self) -> None:
        chosen = self.selected_entries()
        if len(chosen) == 1:
            self._detail.setPlainText(chosen[0].describe())
        elif len(chosen) > 1:
            self._detail.setPlainText(
                "\n\n".join(e.describe() for e in chosen[:2]))
        else:
            self._detail.setPlainText("")
        self._update_controls()

    # -- download ----------------------------------------------------------

    def set_opener(self, opener: Optional[Callable]) -> None:
        """Override how bytes are fetched.

        ``fn(uri) -> chunks`` or ``fn(uri) -> (chunks, total)``; None restores
        :func:`spacr.model_zoo.open_uri`. This is the seam the tests use, and
        it is why no test in this suite touches the network.
        """
        self._opener = opener

    def download_selected(self) -> bool:
        """Download the selected model into the destination folder.

        Off the GUI thread, atomic, checksummed and versioned — see
        :func:`spacr.model_zoo.fetch`. A cancel leaves the destination exactly
        as it was.

        :returns: for the synchronous path, whether a file was installed; for
            the threaded path, True once the job started.
        """
        chosen = self.selected_entries()
        if len(chosen) != 1:
            self._set_status(
                "Select exactly one model to download." if chosen
                else "Select a model to download.", error=True)
            return False
        entry = chosen[0]
        if entry.exists:
            self._set_status(
                f"{entry.name} is already here: {entry.path}", error=True)
            return False
        if not entry.uri:
            self._set_status(
                f"{entry.name} has no download URI — it is a {entry.source} "
                f"entry.", error=True)
            return False

        dest = str(self._dest_edit.text() or DEFAULT_DOWNLOAD_DIR).strip()
        require = not self._allow_unverified.isChecked()
        opener = self._opener
        self._cancel = {"stop": False}
        cancel = self._cancel
        tick = self._progress_ticked.emit

        def _job() -> zoo.ModelEntry:
            return zoo.install(entry, dest, require_checksum=require,
                               opener=opener, progress=tick,
                               cancel=lambda: bool(cancel["stop"]))

        self._progress.setValue(0)
        self._set_status(f"Downloading {entry.name} → {dest}…")
        return self._run_job(_job, self._apply_download,
                             on_error=self._on_download_failed)

    def _apply_download(self, entry: zoo.ModelEntry) -> None:
        self._progress.setValue(100)
        # The catalogue row this came from is replaced by the local file, not
        # listed beside it: one model, one row, and the row now points at bytes
        # that are here.
        listing = [e for e in self._entries
                   if not (e.uri and e.uri == entry.uri and not e.exists)]
        listing.append(entry)
        self.set_entries(listing)
        self.select(len(listing) - 1)
        state = ("checksum verified against the published digest"
                 if entry.verified else
                 "no published checksum to verify against — the recorded hash "
                 "is of the bytes that arrived and proves nothing about where "
                 "they came from")
        self._set_status(f"Installed {entry.name} → {entry.path} ({state}).")
        self.download_finished.emit(True, entry.path)

    def _on_download_failed(self, message: str) -> None:
        self._progress.setValue(0)
        self._set_status(message, error=True)
        self.download_finished.emit(False, "")

    def cancel_download(self) -> bool:
        """Ask the running download to stop. Nothing is left behind.

        The flag is a plain dict read by the worker between chunks; the fetch
        deletes its temporary file and raises, so the destination folder never
        sees a partial model.

        :returns: True when there was something to cancel.
        """
        if not self._busy:
            self._set_status("Nothing is downloading.")
            return False
        self._cancel["stop"] = True
        self._set_status("Cancelling…")
        return True

    def _on_progress(self, done: int, total: int) -> None:
        """Progress, on the GUI thread (relayed via :attr:`_progress_ticked`)."""
        if total > 0:
            self._progress.setRange(0, 100)
            self._progress.setValue(int(100 * done / total))
            self._progress.setFormat(
                f"{zoo._human_bytes(done)} / {zoo._human_bytes(total)} (%p%)")
        else:
            self._progress.setRange(0, 0)
            self._progress.setFormat(zoo._human_bytes(done))

    def _on_progress_text(self, message: str) -> None:
        self._set_status(message)

    def download_progress(self) -> int:
        """The progress bar's current value (test helper)."""
        return int(self._progress.value())

    # -- benchmark ---------------------------------------------------------

    def set_segment_fn(self, fn: Optional[Callable]) -> None:
        """Override the segmentation backend.

        ``fn(images, config) -> masks``; None restores
        :func:`spacr.model_compare.segment_with_cellpose`. Every test injects
        one, which is why nothing here loads Cellpose.
        """
        self._segment_fn = fn

    def set_fields_source(self, folder: str) -> bool:
        """Load benchmark fields without blocking the GUI thread.

        :param folder: a folder of ``.tif`` / ``.png`` / ``.npy`` / ``.npz``.
        :returns: with ``threaded=False``, True when at least one field loaded;
            otherwise True once the load starts. On failure the reason is in
            the status label.
        """
        from ... import model_compare as mc

        if self._busy:
            self._set_status("Another Model Zoo job is already running.",
                             error=True)
            return False
        self._images = []
        self._field_names = []
        self._fields_folder = ""
        source = os.fspath(folder)
        n_fields = int(self._fields_box.value())

        def _job():
            names, images = mc.load_fields(source, n_fields=n_fields)
            return source, names, images

        self._set_status(f"Loading up to {n_fields} field(s) from {source}…")
        return self._run_job(
            _job, self._apply_loaded_fields,
            on_error=self._on_fields_failed)

    def _apply_loaded_fields(self, result) -> None:
        """Install loaded benchmark fields on the GUI thread."""
        source, names, images = result
        self._fields_folder = source
        self._field_names = names
        self._images = images
        self._fields_edit.setText(self._fields_folder)
        self._set_status(
            f"Loaded {len(images)} field(s) from {self._fields_folder}: "
            f"{', '.join(names)}.")
        self._update_controls()

    def _on_fields_failed(self, message: str) -> None:
        """Report a field-loading failure inline."""
        self._set_status(
            f"Could not load benchmark fields: {message}", error=True)
        self._update_controls()

    def fields_folder(self) -> str:
        """The loaded field folder, or ``''``."""
        return self._fields_folder

    def field_names(self) -> List[str]:
        """The loaded field names, in order."""
        return list(self._field_names)

    def _on_fields_typed(self) -> None:
        self.set_fields_source(self._fields_edit.text())

    def _pick_fields_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Choose a folder of fields", self._fields_folder or os.getcwd())
        if path:
            self.set_fields_source(path)

    def _reload_fields(self) -> None:
        self._btn_test.setText(f"Test on {self._fields_box.value()} fields")
        if self._fields_folder:
            self.set_fields_source(self._fields_folder)

    def run_benchmark(self) -> bool:
        """Segment the loaded fields with the selected model and score them.

        :returns: for the synchronous path, whether a result was produced; for
            the threaded path, True once the job started.
        """
        chosen = self.selected_entries()
        if len(chosen) != 1:
            self._set_status(
                "Select exactly one model to test." if chosen
                else "Select a model to test.", error=True)
            return False
        if not self._images:
            self._set_status(
                "Choose a folder of fields to test on first.", error=True)
            return False
        entry = chosen[0]
        if not entry.exists:
            self._set_status(
                f"{entry.name} is not on this machine yet — download it "
                f"first.", error=True)
            return False

        images = list(self._images)
        names = list(self._field_names)
        folder = self._fields_folder
        segment_fn = self._segment_fn
        say = self._progress_said.emit

        def _job() -> zoo.BenchmarkResult:
            return zoo.benchmark(
                entry, images=images, field_names=names, source=folder,
                segment_fn=segment_fn,
                progress=lambda message, done, total: say(message))

        self._set_status(f"Segmenting {len(images)} field(s) with "
                         f"{entry.name}…")
        return self._run_job(_job, self._apply_benchmark)

    def _apply_benchmark(self, result: zoo.BenchmarkResult) -> None:
        self._result = result
        table = self._bench_table
        table.blockSignals(True)
        table.setRowCount(len(result.rows))
        for r, row in enumerate(result.rows):
            cells = (row.field, str(row.n_objects), row.severity,
                     ", ".join(row.flags) if row.flags else "-")
            for c, text in enumerate(cells):
                item = _cell(text)
                item.setToolTip(row.note)
                if row.severity == "fail":
                    item.setForeground(_brush(active_palette()["error"]))
                elif row.severity == "warn":
                    item.setForeground(_brush(active_palette()["warning"]))
                table.setItem(r, c, item)
        table.blockSignals(False)
        table.resizeColumnsToContents()

        # The field set is named in the summary on purpose: this number is only
        # meaningful next to another number from the same three fields.
        self._summary.setText(
            f"{result.summary}  Field set {result.fieldset} — "
            f"{result.fieldset_label}. seg_qc is a quality-control verdict on "
            f"this model's own masks, not an accuracy: there is no ground "
            f"truth here.")
        if result.rows:
            table.setCurrentCell(0, 0)
            self.select_field(0)
        self._set_status(result.summary)

    def result(self) -> Optional[zoo.BenchmarkResult]:
        """The most recent :class:`spacr.model_zoo.BenchmarkResult`."""
        return self._result

    def benchmark_rows(self) -> List[List[str]]:
        """The benchmark table as plain strings."""
        return [[(self._bench_table.item(r, c).text()
                  if self._bench_table.item(r, c) else "")
                 for c in range(self._bench_table.columnCount())]
                for r in range(self._bench_table.rowCount())]

    def select_field(self, row: int) -> bool:
        """Draw field ``row``'s mask over its image.

        :param row: index into the benchmark table.
        :returns: True when something was drawn.
        """
        result = self._result
        if result is None or not (0 <= row < len(result.rows)):
            return False
        if row >= len(result.masks):
            self._preview.setText("Masks were not kept for this run.")
            return False
        image = result.images[row] if row < len(result.images) else None
        composed = compose_labels(image, result.masks[row])
        if composed is None:
            self._preview.setPixmap(QPixmap())
            self._preview.setText("Nothing to draw for this field.")
            return False
        height, width = composed.shape[:2]
        qimage = QImage(composed.tobytes(), width, height, 3 * width,
                        QImage.Format_RGB888).copy()
        self._preview.setText("")
        self._preview.setPixmap(QPixmap.fromImage(qimage).scaled(
            max(PREVIEW_PX, self._preview.width()),
            max(PREVIEW_PX, self._preview.height()),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))
        return True

    def preview_size(self):
        """``(w, h)`` of the preview pixmap, ``(0, 0)`` when there is none."""
        pixmap = self._preview.pixmap()
        if pixmap is None or pixmap.isNull():
            return (0, 0)
        return (pixmap.width(), pixmap.height())

    # -- hand-off to Model Compare -----------------------------------------

    def compare_selected(self) -> bool:
        """Hand two selected models to the A/B comparison.

        Two separate benchmarks are not a comparison: they have no shared
        object matching, no split/merge attribution and no statement about
        which differences are real. :mod:`spacr.model_compare` has all three,
        so this emits :attr:`compare_requested` for the host to open that
        screen rather than growing a weaker comparison here.

        :returns: True when the request was emitted.
        """
        chosen = self.selected_entries()
        if len(chosen) != 2:
            self._set_status(
                f"Select exactly two models to compare "
                f"({len(chosen)} selected).", error=True)
            return False
        missing = [e.name for e in chosen if not e.exists]
        if missing:
            self._set_status(
                f"Download {' and '.join(missing)} before comparing.",
                error=True)
            return False
        request = {
            "model_a": chosen[0].path or chosen[0].name,
            "model_b": chosen[1].path or chosen[1].name,
            "name_a": chosen[0].name,
            "name_b": chosen[1].name,
            "folder": self._fields_folder,
            "n_fields": int(self._fields_box.value()),
        }
        self._set_status(
            f"Comparing {chosen[0].name} against {chosen[1].name} on "
            f"{request['n_fields']} field(s) — neither is treated as ground "
            f"truth.")
        self.compare_requested.emit(request)
        return True

    def build_comparison_screen(self, threaded: bool = True):
        """A :class:`~spacr.qt.screens.model_compare.ModelCompareScreen`, configured.

        The host can show this directly instead of wiring
        :attr:`compare_requested` itself.

        :param threaded: passed through to the comparison screen.
        :returns: the configured screen, or None when two models with local
            files are not selected.
        """
        chosen = self.selected_entries()
        if len(chosen) != 2 or any(not e.exists for e in chosen):
            return None
        from .model_compare import ModelCompareScreen

        screen = ModelCompareScreen(threaded=threaded)
        screen._panel_a.model_edit.setText(chosen[0].path)
        screen._panel_b.model_edit.setText(chosen[1].path)
        screen._fields_box.setValue(int(self._fields_box.value()))
        if self._segment_fn is not None:
            screen.set_segment_fn(self._segment_fn)
        if self._fields_folder:
            screen.set_source(self._fields_folder)
        return screen

    # -- job plumbing ------------------------------------------------------

    def _run_job(self, fn: Callable[[], Any],
                 on_done: Callable[[Any], None],
                 on_error: Optional[Callable[[str], None]] = None) -> bool:
        """Run ``fn`` off the GUI thread and hand its result to ``on_done``.

        The same idiom as ``PlateViewScreen._run_job``, and for the same
        reason: ``PipelineWorker.finished`` is emitted *in the worker thread*,
        and PySide6 invokes a plain closure connected to it directly, on that
        thread. This screen's completion handlers fill a QPlainTextEdit and two
        QTableWidgets, and building a QTextDocument's children off the GUI
        thread is undefined behaviour ("Cannot create children for a parent
        that is in a different thread"). So ``finished`` chains through
        :attr:`_job_settled` into a *bound method* of this widget, which has
        GUI-thread affinity; Qt then queues the call.

        ``threaded=False`` runs inline and emits the same signals, so both
        paths behave identically from outside.
        """
        self._error_handler = on_error
        if not self._threaded:
            ok = True
            try:
                on_done(fn())
            except Exception as e:
                self._report_failure(e)
                ok = False
            self._busy = False
            self._update_controls()
            self.job_finished.emit(ok)
            return ok

        box: Dict[str, Any] = {}

        def _job(payload: Dict[str, Any]) -> None:
            payload["result"] = fn()

        thread, worker = make_thread(_job, box)
        # Strong references: PySide6 will not keep the worker alive through the
        # started→run connection alone, and a QThread garbage-collected while
        # still running takes the process down with it.
        self._jobs.append((thread, worker))
        self._pending.append((box, on_done, on_error))
        worker.error.connect(self._on_worker_error_text)
        worker.finished.connect(self._job_settled)
        thread.finished.connect(self._retire_finished_jobs)
        self._busy = True
        self._update_controls()
        thread.start()
        return True

    def _on_job_settled(self, ok: bool) -> None:
        """Finish the oldest in-flight job. Always on the GUI thread."""
        self._busy = False
        box, on_done, on_error = (self._pending.pop(0) if self._pending
                                  else ({}, None, None))
        ok = bool(ok)
        if ok and on_done is not None:
            try:
                on_done(box.get("result"))
            except Exception as e:
                self._error_handler = on_error
                self._report_failure(e)
                ok = False
        self._update_controls()
        self.job_finished.emit(ok)

    def _retire_finished_jobs(self) -> None:
        """Retire every job whose QThread has stopped. GUI thread only.

        A BOUND METHOD, not a closure — the rule ``make_thread`` states and
        then relies on for its own ``handle.retire``. With a closure PySide6
        makes the QThread itself the receiver, and ``make_thread`` connects
        ``thread.finished -> thread.deleteLater`` FIRST; slots run in
        connection order, so the DeferredDelete is posted ahead of the
        closure's metacall and Qt discards queued events for a destroyed
        receiver. The job was then never retired, ``active_jobs()`` never
        returned to zero, and every ``waitUntil(active_jobs() == 0)`` sat
        there until it timed out with the QThread's C++ half already gone.

        It sweeps rather than naming a sender for the same reason: by the
        time this runs, the emitter may be exactly what is gone, and
        ``QObject.sender()`` is null for a queued call whose emitter was
        destroyed.
        """
        from ..bridge import thread_has_stopped

        for thread, _worker in list(self._jobs):
            if thread_has_stopped(thread):
                self._retire_job(thread)

    def _retire_job(self, thread) -> None:
        """Release *this* job's refs once its own event loop has exited."""
        self._jobs = [(t, w) for (t, w) in self._jobs if t is not thread]

    def active_jobs(self) -> int:
        """How many worker threads are still winding down."""
        return len(self._jobs)

    def is_busy(self) -> bool:
        """True while a scan, download or benchmark is in flight."""
        return self._busy

    def _report_failure(self, exc: Exception) -> None:
        message = str(exc) or exc.__class__.__name__
        handler = getattr(self, "_error_handler", None)
        self._busy = False
        if handler is not None:
            handler(message)
        else:
            self._set_status(message, error=True)

    def _on_worker_error_text(self, tb: str) -> None:
        """Turn a worker traceback into the message its exception carried.

        Not "the last line": a :class:`spacr.model_zoo.ChecksumMismatch`
        message is five lines long and the last of them on its own says
        nothing. The exception line is found instead, and everything from it to
        the end is kept, minus the class name.
        """
        import re as _re

        lines = [l for l in str(tb).strip().splitlines() if l.strip()]
        start = 0
        for i, candidate in enumerate(lines):
            if _re.match(r"^[A-Za-z_][\w.]*\s*:", candidate) and \
                    not candidate.startswith(" "):
                start = i
        message = "\n".join(lines[start:]).strip()
        message = _re.sub(r"^[A-Za-z_][\w.]*\s*:\s*", "", message, count=1)
        line = message or (lines[-1] if lines else "unknown error")
        self._busy = False
        pending = self._pending[0] if self._pending else None
        handler = pending[2] if pending else None
        if handler is not None:
            handler(line)
        else:
            self._set_status(line, error=True)

    # -- enablement --------------------------------------------------------

    def _update_controls(self) -> None:
        chosen = self.selected_entries()
        one = len(chosen) == 1
        busy = self._busy
        self._btn_scan.setEnabled(not busy)
        self._btn_pick_scan.setEnabled(not busy)
        self._btn_pick_dest.setEnabled(not busy)
        self._btn_pick_fields.setEnabled(not busy)
        self._fields_box.setEnabled(not busy)
        self._btn_download.setEnabled(
            not busy and one and bool(chosen[0].uri) and not chosen[0].exists)
        self._btn_cancel.setEnabled(busy)
        self._btn_test.setEnabled(
            not busy and one and bool(self._images) and chosen[0].exists)
        self._btn_compare.setEnabled(not busy and len(chosen) == 2)

    # -- shutdown ----------------------------------------------------------

    def closeEvent(self, event):  # noqa: N802
        """Let every in-flight job finish before the widget dies.

        A QThread collected while still running aborts the process, so the
        widget waits rather than dropping its references and hoping.
        """
        self._cancel["stop"] = True
        for thread, _worker in list(self._jobs):
            try:
                if thread.isRunning():
                    thread.quit()
                    thread.wait(5000)
            except RuntimeError:
                pass
        super().closeEvent(event)


def _brush(colour: str):
    """A QBrush for a palette colour string."""
    from PySide6.QtGui import QBrush, QColor

    return QBrush(QColor(colour))
