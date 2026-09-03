"""Choose which published sequencing runs to fetch, and how much of each.

Map Barcodes offers the paper's own reads, and lets the user say how many
sequencing lines to take from each file.

THE LIMIT IS THE FEATURE, not a convenience. The four runs are 20.4 GB in
full; a hundred thousand reads from each is about 30 MB, because ENA serves
gzipped FASTQ that :func:`spacr.sra.fetch_reads` reads as a stream and
abandons once it has enough. The estimate is shown before anything starts, and
it is recomputed as the limit changes, so the size is a decision rather than a
surprise.
"""
from __future__ import annotations

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ...sra import estimated_bytes, fetch_reads, runs_for
from ..i18n import tr

#: Reads per file when the dialog opens. Enough to exercise barcode mapping
#: end to end -- the screen's own guides are found in the first few thousand --
#: while costing about 30 MB across the four runs rather than 20.4 GB.
DEFAULT_READS = 100_000


class _FetchWorker(QThread):
    """Download the chosen files off the GUI thread."""

    progress = Signal(str, int, int)
    finished_all = Signal(list, str)

    def __init__(self, files, destination, max_reads, parent=None):
        super().__init__(parent)
        self._files = list(files)
        self._destination = destination
        self._max_reads = max_reads
        self._stop = False

    def cancel(self) -> None:
        self._stop = True

    def run(self) -> None:                        # noqa: D102 - Qt entry point
        written, error = [], ""
        try:
            for one in self._files:
                if self._stop:
                    break
                written.append(str(fetch_reads(
                    one, self._destination, max_reads=self._max_reads,
                    should_stop=lambda: self._stop,
                    progress=lambda reads, byts, run=one.run:
                        self.progress.emit(run, reads, byts))))
        except InterruptedError:
            error = "cancelled"
        except Exception as exc:                  # noqa: BLE001
            error = str(exc)
        self.finished_all.emit(written, error)


class SraPicker(QDialog):
    """List the published runs, take a read limit, and fetch what is ticked."""

    #: The width this dialog opens at.
    #:
    #: Left to Qt it took the width of its widest ROW -- the read-limit spin
    #: box and its two labels -- and wrapped both the blurb and the run names
    #: into a column narrower than either reads well in. A run accession and
    #: its size on one line is what this list is for.
    DIALOG_WIDTH = 520

    def __init__(self, destination, parent=None, *, files=None):
        """Offer the published runs and download the ticked ones.

        :param destination: the folder downloads are written to.
        :param parent: parent widget.
        :param files: the runs to offer. ``None`` asks the index for them,
            so a test can supply a list without reaching the network.
        """
        super().__init__(parent)
        self.setWindowTitle(tr("Load test data"))
        self.setMinimumWidth(self.DIALOG_WIDTH)
        self._destination = destination
        self._worker = None
        self.written: list[str] = []

        layout = QVBoxLayout(self)
        self._blurb = QLabel(tr(
            "Raw reads from the published screen, NCBI BioProject "
            "PRJNA1261935. Tick the runs you want and set how many reads to "
            "take from each — only that much is downloaded."), self)
        self._blurb.setWordWrap(True)
        layout.addWidget(self._blurb)

        self._list = QListWidget(self)
        self._list.setSelectionMode(QAbstractItemView.NoSelection)
        layout.addWidget(self._list, 1)

        limit_row = QHBoxLayout()
        limit_row.addWidget(QLabel(tr("Reads from each file:"), self))
        self._reads = QSpinBox(self)
        # A MILLION IS NOT THE CEILING the data has; it is the ceiling this
        # control offers, because past it the download stops being a sample
        # and the "whole file" tick is the honest way to ask for everything.
        self._reads.setRange(1_000, 100_000_000)
        self._reads.setSingleStep(10_000)
        self._reads.setGroupSeparatorShown(True)
        self._reads.setValue(DEFAULT_READS)
        limit_row.addWidget(self._reads)
        self._whole = QCheckBox(tr("Whole file"), self)
        self._whole.setToolTip(tr(
            "Download every read. The four runs are about 20 GB in total."))
        limit_row.addWidget(self._whole)
        limit_row.addStretch(1)
        layout.addLayout(limit_row)

        self._estimate = QLabel("", self)
        layout.addWidget(self._estimate)

        self._progress = QProgressBar(self)
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self._download = QPushButton(tr("Download"), self)
        self._download.clicked.connect(self._start)
        buttons.addWidget(self._download)
        self._close = QPushButton(tr("Close"), self)
        self._close.clicked.connect(self.reject)
        buttons.addWidget(self._close)
        layout.addLayout(buttons)

        self._reads.valueChanged.connect(lambda _v: self._refresh_estimate())
        self._whole.toggled.connect(self._on_whole_toggled)
        self._list.itemChanged.connect(lambda _i: self._refresh_estimate())

        self._files = tuple(files) if files is not None else ()
        if files is None:
            self._load_the_listing()
        else:
            self._show(self._files)

        # AS SMALL AS THE CONTENT NEEDS, once the list has been filled --
        # after, because the runs are what decide how tall it wants to be.
        # Asked for on 2026-09-02 about the sibling dialog and applied here
        # for the same reason: a window that opens taller than its contents
        # is a window the user has to fix before reading it.
        self.adjustSize()

    # -- listing -------------------------------------------------------
    def _load_the_listing(self) -> None:
        """Ask ENA what the project holds. A failure is said, not raised."""
        try:
            self._files = runs_for()
        except Exception as exc:                  # noqa: BLE001
            self._files = ()
            self._blurb.setText(tr(
                "Could not reach the sequence archive: {detail}",
                detail=str(exc)))
            self._download.setEnabled(False)
            return
        self._show(self._files)

    def _show(self, files) -> None:
        # SIGNALS OFF WHILE POPULATING. setCheckState emits itemChanged, which
        # recomputes the estimate -- and it fires while the item being built
        # has no RunFile attached yet, so the estimate reads a None. Set the
        # data first as well: belt and braces, because the order inside a Qt
        # item constructor is not this file's to guarantee.
        self._list.blockSignals(True)
        try:
            self._list.clear()
            for one in files:
                item = QListWidgetItem(one.label(), self._list)
                item.setData(Qt.UserRole, one)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
        finally:
            self._list.blockSignals(False)
        self._refresh_estimate()

    def chosen_files(self) -> list:
        """Every ticked run file."""
        out = []
        for row in range(self._list.count()):
            item = self._list.item(row)
            payload = item.data(Qt.UserRole)
            if payload is not None and item.checkState() == Qt.Checked:
                out.append(payload)
        return out

    def max_reads(self):
        """The limit, or ``None`` for the whole file."""
        return None if self._whole.isChecked() else int(self._reads.value())

    # -- estimate ------------------------------------------------------
    def _on_whole_toggled(self, on: bool) -> None:
        self._reads.setEnabled(not on)
        self._refresh_estimate()

    def _refresh_estimate(self) -> None:
        chosen = self.chosen_files()
        if not chosen:
            self._estimate.setText(tr("Nothing selected."))
            self._download.setEnabled(False)
            return
        self._download.setEnabled(self._worker is None)
        size = estimated_bytes(chosen, self.max_reads())
        self._estimate.setText(tr(
            "{count} files, about {size} to download.",
            count=len(chosen), size=_human(size)))

    # -- fetching ------------------------------------------------------
    def _start(self) -> None:
        chosen = self.chosen_files()
        if not chosen:
            return
        self._worker = _FetchWorker(chosen, self._destination,
                                    self.max_reads(), self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_all.connect(self._on_done)
        self._download.setEnabled(False)
        self._progress.setVisible(True)
        self._worker.start()

    def _on_progress(self, run: str, reads: int, byts: int) -> None:
        self._estimate.setText(tr(
            "{run}: {reads} reads ({size})",
            run=run, reads=f"{reads:,}", size=_human(byts)))

    def _on_done(self, written, error) -> None:
        self.written = list(written)
        self._progress.setVisible(False)
        self._worker = None
        if error:
            self._estimate.setText(tr("Stopped: {detail}", detail=error))
            self._download.setEnabled(True)
            return
        self.accept()

    def reject(self) -> None:                     # noqa: D102 - Qt override
        if self._worker is not None:
            self._worker.cancel()
            self._worker.wait(3000)
        super().reject()


def _human(size: int) -> str:
    """Bytes as the unit a person would say."""
    if size >= 1e9:
        return f"{size / 1e9:.1f} GB"
    if size >= 1e6:
        return f"{size / 1e6:.0f} MB"
    return f"{size / 1e3:.0f} kB"


__all__ = ["SraPicker", "DEFAULT_READS"]
