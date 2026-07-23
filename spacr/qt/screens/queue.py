"""
Queue screen — dashboard for the plate queue.

Layout:

    ┌───────────────────────────────────────────────────────────────┐
    │ [Add plate] [Import CSV…] [Clear finished]  [Run] [Stop]      │
    ├───────────────────────────────────────────────────────────────┤
    │ ID   │ App   │ Label        │ Status   │ Elapsed │            │
    │ ...  │ mask  │ /data/plateA │ success  │ 42.1 s  │  [Remove]  │
    │ ...  │ mask  │ /data/plateB │ running  │ 08.5 s  │            │
    │ ...  │ mask  │ /data/plateC │ queued   │         │  [Remove]  │
    └───────────────────────────────────────────────────────────────┘

Each row reflects a :class:`spacr.qt.plate_queue.QueueItem`. The
screen owns a :class:`spacr.qt.plate_queue.PlateQueue` and a single
worker QThread that walks items sequentially.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractItemView, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
    QMessageBox, QPushButton, QSizePolicy, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from ..plate_queue import (
    PlateQueue, QueueItem, Status, import_plates_from_csv,
)

LOG = logging.getLogger("spacr.qt.queue_screen")


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class _QueueRunner(QThread):
    """Walks the queue sequentially in a background thread.

    Emits :attr:`item_state_changed` on every status transition so
    the table can refresh a single row without a full rebuild.
    """

    item_state_changed = Signal(str)   # item id
    queue_finished     = Signal()

    def __init__(self, queue: PlateQueue, parent=None):
        super().__init__(parent)
        self._queue = queue
        self._stop = False

    def stop(self) -> None:
        """Ask the runner to exit after the current item completes."""
        self._stop = True

    def run(self) -> None:
        from ..bridge import resolve_pipeline_entry
        while not self._stop:
            item = self._queue.next_queued()
            if item is None:
                break
            self._queue.update(item.id, status=Status.RUNNING,
                                  start_ts=time.time())
            self.item_state_changed.emit(item.id)
            try:
                fn = resolve_pipeline_entry(item.app_key)
                if fn is None:
                    raise RuntimeError(
                        f"no pipeline for app_key={item.app_key!r}")
                fn(item.settings)
            except Exception as e:
                LOG.warning("queue item %s failed: %s", item.id, e,
                              exc_info=True)
                self._queue.update(item.id, status=Status.FAILED,
                                      end_ts=time.time(), error=str(e))
                self.item_state_changed.emit(item.id)
                continue
            self._queue.update(item.id, status=Status.SUCCESS,
                                  end_ts=time.time())
            self.item_state_changed.emit(item.id)
        self.queue_finished.emit()


# ---------------------------------------------------------------------------
# Screen
# ---------------------------------------------------------------------------

_COLUMNS = ("ID", "App", "Label", "Status", "Elapsed", "")


class QueueScreen(QWidget):
    """Main widget rendering the plate queue."""

    # Emitted whenever the queue changes size (add / remove / clear).
    # MainWindow can use this to update the Home-tile badge count.
    queue_size_changed = Signal(int)

    def __init__(self, queue: Optional[PlateQueue] = None, parent=None):
        super().__init__(parent)
        self._queue = queue if queue is not None else PlateQueue()
        self._runner: Optional[_QueueRunner] = None

        self._build_ui()
        self._refresh_table()

        # Poll for elapsed-time updates while the runner is going
        self._tick = QTimer(self)
        self._tick.setInterval(1000)
        self._tick.timeout.connect(self._refresh_elapsed_only)
        self._tick.start()

    # -- construction ------------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.setSpacing(12)

        header = QLabel("Plate queue")
        header.setObjectName("DisplayHeading")
        outer.addWidget(header)

        subtitle = QLabel(
            "Chain multiple plates through the same pipeline. "
            "The runner processes one plate at a time and picks up "
            "where it left off if you close the app.")
        subtitle.setObjectName("Muted")
        outer.addWidget(subtitle)

        # Toolbar
        bar = QHBoxLayout()
        self._btn_add     = QPushButton("Add current plate", self)
        self._btn_import  = QPushButton("Import CSV…", self)
        self._btn_clear   = QPushButton("Clear finished", self)
        self._btn_run     = QPushButton("Run queue", self)
        self._btn_stop    = QPushButton("Stop", self)
        self._btn_stop.setEnabled(False)
        for b in (self._btn_add, self._btn_import, self._btn_clear,
                    self._btn_run, self._btn_stop):
            bar.addWidget(b)
        bar.addStretch(1)
        outer.addLayout(bar)

        self._btn_import.clicked.connect(self._on_import)
        self._btn_clear.clicked.connect(self._on_clear_finished)
        self._btn_run.clicked.connect(self.start_runner)
        self._btn_stop.clicked.connect(self.stop_runner)
        # `_btn_add` isn't wired here — MainWindow connects it to the
        # active app screen's settings snapshot. See wire_add_current.

        # Table
        self._table = QTableWidget(self)
        self._table.setColumnCount(len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(_COLUMNS)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        outer.addWidget(self._table, 1)

    # -- public API --------------------------------------------------------

    def wire_add_current(self, callback):
        """Route the "Add current plate" button through ``callback``.

        MainWindow supplies a callback that snapshots the currently
        active AppScreen's settings dict and returns
        ``(app_key, settings_dict)``. This screen calls
        :meth:`add_item` with that pair.
        """
        def _on_click():
            try:
                app_key, settings = callback()
            except Exception as e:
                QMessageBox.warning(self, "Queue",
                                     f"Couldn't add current plate: {e}")
                return
            if not settings.get("src"):
                QMessageBox.information(self, "Queue",
                    "The active app has no `src` set — nothing to enqueue.")
                return
            self.add_item(app_key, settings)
        self._btn_add.clicked.connect(_on_click)

    def add_item(self, app_key: str, settings: dict) -> QueueItem:
        item = QueueItem.build(app_key, settings)
        self._queue.add(item)
        self._refresh_table()
        self.queue_size_changed.emit(len(self._queue))
        return item

    def queue(self) -> PlateQueue:
        return self._queue

    # -- runner control ----------------------------------------------------

    def start_runner(self):
        if self._runner is not None and self._runner.isRunning():
            return
        if self._queue.next_queued() is None:
            QMessageBox.information(self, "Queue",
                "Nothing to run — every item is already finished.")
            return
        self._runner = _QueueRunner(self._queue, self)
        self._runner.item_state_changed.connect(self._on_item_changed)
        self._runner.queue_finished.connect(self._on_runner_done)
        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._runner.start()

    def stop_runner(self):
        if self._runner is not None and self._runner.isRunning():
            self._runner.stop()

    def _on_runner_done(self):
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._refresh_table()

    def _on_item_changed(self, item_id: str):
        self._refresh_table()

    # -- toolbar handlers --------------------------------------------------

    def _on_import(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import plates from CSV", "", "CSV files (*.csv)")
        if not path:
            return
        try:
            items = import_plates_from_csv(path, base_settings={},
                                             app_key="mask")
        except Exception as e:
            QMessageBox.warning(self, "Queue import",
                                 f"Couldn't import {path}:\n{e}")
            return
        for it in items:
            self._queue.add(it)
        self._refresh_table()
        self.queue_size_changed.emit(len(self._queue))
        QMessageBox.information(self, "Queue import",
                                  f"Added {len(items)} plate(s) from {path}.")

    def _on_clear_finished(self):
        n = self._queue.clear_finished()
        self._refresh_table()
        self.queue_size_changed.emit(len(self._queue))
        if n:
            self._table.selectRow(-1)

    # -- table plumbing ----------------------------------------------------

    def _refresh_table(self):
        items = self._queue.items()
        self._table.setRowCount(len(items))
        for row, item in enumerate(items):
            self._table.setItem(row, 0, QTableWidgetItem(item.id))
            self._table.setItem(row, 1, QTableWidgetItem(item.app_key))
            self._table.setItem(row, 2, QTableWidgetItem(item.label))
            status_item = QTableWidgetItem(item.status.value)
            self._set_status_color(status_item, item.status)
            self._table.setItem(row, 3, status_item)
            elapsed = item.elapsed_s
            self._table.setItem(row, 4, QTableWidgetItem(
                "" if elapsed is None else f"{elapsed:.1f} s"))
            btn = QPushButton("Remove", self)
            btn.setEnabled(item.status != Status.RUNNING)
            btn.clicked.connect(
                lambda _c=False, iid=item.id: self._on_remove(iid))
            self._table.setCellWidget(row, 5, btn)

    def _refresh_elapsed_only(self):
        # Only touch the elapsed column so we don't churn the whole
        # table (and lose selection state) every second.
        items = self._queue.items()
        for row, item in enumerate(items):
            if item.status != Status.RUNNING or row >= self._table.rowCount():
                continue
            e = item.elapsed_s
            if e is not None:
                self._table.setItem(row, 4, QTableWidgetItem(f"{e:.1f} s"))

    def _on_remove(self, item_id: str):
        item = self._queue.find(item_id)
        if item is not None and item.status == Status.RUNNING:
            QMessageBox.warning(self, "Queue",
                "Can't remove a plate while it's running — stop the queue first.")
            return
        self._queue.remove(item_id)
        self._refresh_table()
        self.queue_size_changed.emit(len(self._queue))

    @staticmethod
    def _set_status_color(item: QTableWidgetItem, status: Status):
        colors = {
            Status.QUEUED:  Qt.darkGray,
            Status.RUNNING: Qt.blue,
            Status.SUCCESS: Qt.darkGreen,
            Status.FAILED:  Qt.darkRed,
            Status.SKIPPED: Qt.gray,
        }
        c = colors.get(status, Qt.black)
        item.setForeground(c)
