"""Searchable run-history dashboard for every journalled spaCR job.

The screen is read-only. Enumerating hundreds of manifests and settings files
runs in a worker thread; filtering and rendering the already-loaded records is
instant and stays on the GUI thread.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QBrush, QColor, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...run_journal import search_runs
from ..bridge import make_thread
from ..iconset import icon
from ..theme import (SPACING, active_palette, page_tabs_qss,
                     register_widget_qss)
from ..widgets import Divider

LOG = logging.getLogger(__name__)

__all__ = [
    "RunHistoryScreen",
    "APP_KEY",
    "APP_NAME",
    "APP_SECTION",
    "APP_INTRO",
]

APP_KEY = "run_history"
APP_NAME = "Run History"
APP_SECTION = "Results & QC"
APP_INTRO = (
    "Search every recorded job and inspect its settings, inputs, outputs, "
    "warnings, failure traceback, versions, seeds, and performance in one place."
)

_COLUMNS = (
    "Started", "Module", "Status", "Duration", "CPU", "Inputs", "Outputs",
    "Warnings",
)


def _readonly_item(text: Any) -> QTableWidgetItem:
    """Return a non-editable table item."""
    item = QTableWidgetItem("" if text is None else str(text))
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    return item


def _seconds(value: Any) -> str:
    """Format seconds compactly for the history table."""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "—"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {remaining:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _bytes(value: Any) -> str:
    """Format a byte count with binary units."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if abs(number) < 1024.0 or unit == units[-1]:
            return f"{number:.0f} {unit}" if unit == "B" else f"{number:.1f} {unit}"
        number /= 1024.0
    return "—"


def _json_text(value: Any) -> str:
    """Pretty-print a possibly non-JSON-native value."""
    return json.dumps(value, indent=2, sort_keys=True, default=str)


#: ``objectName`` of the tab strip, and the name its QSS block is
#: registered under. The tabs ARE the page on this screen, so they take
#: Home's treatment — rounded top corners, a dark-grey tab, the accent
#: blue on hover — at the page opacity, instead of the shipped rules'
#: opaque `surface`/`surface_alt` hex.
TABS_NAME = "RunHistoryTabs"


def _tabs_qss(palette: dict, opacity) -> str:
    """QSS for the tab strip, registered through the theme seam."""
    return page_tabs_qss(TABS_NAME, palette, opacity)


# ``replace=True``: this module owns the name, so a reimport re-registers
# rather than raising and leaving the tabs unstyled.
register_widget_qss(TABS_NAME, _tabs_qss, replace=True)


class RunHistoryScreen(QWidget):
    """Search and inspect all run-journal records.

    :param parent: Qt parent.
    :param threaded: enumerate the journal on a worker thread. Tests use
        ``False`` for deterministic execution.
    :ivar last_error: most recent refresh failure, or ``""``.
    :ivar records: all records loaded by the last successful refresh.
    """

    settings_requested = Signal(str, dict)
    history_refreshed = Signal(int)

    def __init__(self, parent=None, threaded: bool = True):
        super().__init__(parent)
        self._threaded = bool(threaded)
        self.records: List[Dict[str, Any]] = []
        self.last_error = ""
        self._busy = False
        self._loaded_once = False
        self._jobs: List[tuple] = []
        self._pending_result: Optional[List[Dict[str, Any]]] = None
        self._pending_error = ""
        self._record_by_id: Dict[str, Dict[str, Any]] = {}
        self._build_ui()
        self._set_status("Open this module to load the run journal.")
        # Drop anywhere on this screen: the path is resolved through spaCR's
        # project layout, so the plate folder finds what this screen reads.
        from ..dnd import install_for
        install_for(self, "run_history")

    def _build_ui(self) -> None:
        """Construct filters, run table, and tabbed detail inspector."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(
            SPACING["lg"], SPACING["lg"], SPACING["lg"], SPACING["lg"],
        )
        outer.setSpacing(SPACING["md"])

        title = QLabel(APP_NAME, self)
        title.setObjectName("DisplayHeading")
        outer.addWidget(title)
        intro = QLabel(APP_INTRO, self)
        intro.setObjectName("Muted")
        intro.setWordWrap(True)
        outer.addWidget(intro)
        outer.addWidget(Divider())

        filters = QHBoxLayout()
        filters.setSpacing(SPACING["sm"])
        self._search = QLineEdit(self)
        self._search.setPlaceholderText(
            "Search module, setting, path, warning, failure, version…"
        )
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._apply_filters)
        self._module = QComboBox(self)
        self._module.addItem("All modules", "")
        self._module.currentIndexChanged.connect(self._apply_filters)
        self._status_filter = QComboBox(self)
        for label, value in (
            ("All statuses", ""),
            ("Success", "success"),
            ("Failed", "failed"),
            ("Running", "running"),
            ("Corrupt", "corrupt"),
        ):
            self._status_filter.addItem(label, value)
        self._status_filter.currentIndexChanged.connect(self._apply_filters)
        self._refresh = QPushButton("Refresh", self)
        self._refresh.setIcon(icon("redo"))
        self._refresh.clicked.connect(self.refresh)
        filters.addWidget(self._search, 1)
        filters.addWidget(self._module)
        filters.addWidget(self._status_filter)
        filters.addWidget(self._refresh)
        outer.addLayout(filters)

        splitter = QSplitter(Qt.Vertical, self)
        self._table = QTableWidget(0, len(_COLUMNS), splitter)
        self._table.setHorizontalHeaderLabels(_COLUMNS)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.itemSelectionChanged.connect(self._show_selection)
        splitter.addWidget(self._table)

        detail = QWidget(splitter)
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(SPACING["sm"])
        action_row = QHBoxLayout()
        self._selection_label = QLabel("Select a run to inspect it.", detail)
        self._selection_label.setObjectName("Muted")
        self._open_folder = QPushButton("Open run folder", detail)
        self._open_folder.setIcon(icon("folder"))
        self._open_folder.clicked.connect(self._open_selected_folder)
        self._copy_path = QPushButton("Copy path", detail)
        self._copy_path.clicked.connect(self._copy_selected_path)
        self._load_settings = QPushButton("Load settings in module", detail)
        self._load_settings.setObjectName("PrimaryButton")
        self._load_settings.clicked.connect(self._load_selected_settings)
        action_row.addWidget(self._selection_label, 1)
        action_row.addWidget(self._open_folder)
        action_row.addWidget(self._copy_path)
        action_row.addWidget(self._load_settings)
        detail_layout.addLayout(action_row)

        self._tabs = QTabWidget(detail)
        self._tabs.setObjectName(TABS_NAME)
        self._overview = self._text_tab("Run summary and performance")
        self._settings = self._text_tab("Exact resolved settings")
        self._outputs = self._text_tab("Input, output, and model hashes")
        self._problems = self._text_tab("Warnings and failure traceback")
        self._environment = self._text_tab("Versions, seeds, and manifest")
        for label, widget in (
            ("Overview", self._overview),
            ("Settings", self._settings),
            ("Files & models", self._outputs),
            ("Warnings & failure", self._problems),
            ("Environment", self._environment),
        ):
            self._tabs.addTab(widget, label)
        detail_layout.addWidget(self._tabs, 1)
        splitter.addWidget(detail)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        outer.addWidget(splitter, 1)

        self._status = QLabel("", self)
        self._status.setObjectName("Muted")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)
        self._clear_details()

    def _text_tab(self, accessible_name: str) -> QPlainTextEdit:
        """Return a read-only detail view with an accessible name."""
        widget = QPlainTextEdit(self)
        widget.setReadOnly(True)
        widget.setLineWrapMode(QPlainTextEdit.NoWrap)
        widget.setAccessibleName(accessible_name)
        return widget

    def showEvent(self, event) -> None:
        """Load history on first display, not during application startup."""
        super().showEvent(event)
        if not self._loaded_once and not self._busy:
            self.refresh()

    def refresh(self) -> None:
        """Reload every run record without blocking the GUI thread."""
        if self._busy:
            return
        self._busy = True
        self.last_error = ""
        self._refresh.setEnabled(False)
        self._set_status("Loading run manifests and settings…")
        if not self._threaded:
            try:
                self.records = search_runs()
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.records = []
            self._finish_refresh(not bool(self.last_error))
            return

        self._pending_result = None
        self._pending_error = ""

        def _load(_settings):
            try:
                self._pending_result = search_runs()
            except Exception as exc:
                self._pending_error = f"{type(exc).__name__}: {exc}"
                LOG.exception("Run-history refresh failed")

        thread, worker = make_thread(
            _load, {}, app_key="run_history_refresh", journal=False,
        )
        pair = (thread, worker)
        self._jobs.append(pair)
        worker.finished.connect(self._on_refresh_finished)
        thread.finished.connect(self._retire_jobs)
        thread.start()

    def _on_refresh_finished(self, ok: bool) -> None:
        """Apply the worker result on the GUI thread."""
        if ok and not self._pending_error:
            self.records = list(self._pending_result or [])
        else:
            self.records = []
            self.last_error = self._pending_error or "History worker failed."
        self._finish_refresh(ok and not bool(self.last_error))

    def _retire_jobs(self) -> None:
        """Release ownership pairs whose QThread has stopped.

        A bare ``[p for p in self._jobs if p[0].isRunning()]`` leaked every
        pair here: by the time this queued slot runs, ``thread.finished ->
        deleteLater`` has reaped the QThread's C++ half and ``isRunning()``
        raises ``RuntimeError`` out of the slot, so the assignment never
        happens. See :func:`spacr.qt.bridge.prune_job_pairs`.
        """
        from ..bridge import prune_job_pairs

        self._jobs = prune_job_pairs(self._jobs, self.sender())

    def closeEvent(self, event) -> None:
        """Drain the history worker before Qt destroys this screen.

        Without this the screen could be destroyed with its refresh still
        running. The job survives — ``bridge.make_thread`` registers it
        process-wide — but it survives *ownerless*, and an ownerless job in
        the run registry is what ``MainWindow.closeEvent`` reads when it
        decides whether the application may quit.
        """
        from ..bridge import drain_thread

        for thread, worker in list(self._jobs):
            if worker is not None:
                try:
                    worker.request_cancel("run-history screen closed")
                except Exception:
                    pass
            drain_thread(thread, worker, timeout_ms=3000)
        self._jobs.clear()
        super().closeEvent(event)

    def _finish_refresh(self, ok: bool) -> None:
        """Rebuild filters/table after synchronous or threaded loading."""
        self._busy = False
        self._loaded_once = True
        self._refresh.setEnabled(True)
        self._record_by_id = {
            str(record["run_id"]): record for record in self.records
        }
        selected_module = self._module.currentData()
        modules = sorted({
            str(record.get("app_key") or "unknown") for record in self.records
        })
        self._module.blockSignals(True)
        self._module.clear()
        self._module.addItem("All modules", "")
        for module in modules:
            self._module.addItem(module, module)
        index = self._module.findData(selected_module)
        self._module.setCurrentIndex(max(0, index))
        self._module.blockSignals(False)
        self._apply_filters()
        if ok:
            self._set_status(f"Loaded {len(self.records)} recorded run(s).")
            self.history_refreshed.emit(len(self.records))
        else:
            self._set_status(f"Could not load run history: {self.last_error}",
                             error=True)

    def _matches_filters(self, record: Dict[str, Any]) -> bool:
        """Return whether ``record`` matches current module/status/text."""
        module = str(self._module.currentData() or "")
        status = str(self._status_filter.currentData() or "")
        if module and record.get("app_key") != module:
            return False
        if status and record.get("status") != status:
            return False
        terms = [
            term.casefold() for term in self._search.text().split() if term
        ]
        if not terms:
            return True
        haystack = _json_text({
            "run_id": record.get("run_id"),
            "app_key": record.get("app_key"),
            "status": record.get("status"),
            "settings": record.get("settings"),
            "inputs": list((record.get("inputs") or {}).keys()),
            "outputs": list((record.get("outputs") or {}).keys()),
            "warnings": record.get("warnings"),
            "failure": record.get("failure"),
            "environment": record.get("environment"),
        }).casefold()
        return all(term in haystack for term in terms)

    def _apply_filters(self, *_args) -> None:
        """Render records matching the current controls."""
        visible = [
            record for record in self.records if self._matches_filters(record)
        ]
        self._table.setSortingEnabled(False)
        self._table.setRowCount(0)
        palette = active_palette()
        status_colours = {
            "success": palette["success"],
            "failed": palette["error"],
            "running": palette["warning"],
            "corrupt": palette["error"],
        }
        for record in visible:
            row = self._table.rowCount()
            self._table.insertRow(row)
            performance = record.get("performance") or {}
            started = str(record.get("start_utc") or "—")
            if "T" in started:
                started = started.replace("T", " ", 1).replace("+00:00", "Z")
            values = (
                started,
                record.get("app_key") or "unknown",
                record.get("status") or "unknown",
                _seconds(performance.get("wall_s")),
                _seconds(performance.get("process_cpu_s")),
                f"{performance.get('input_files', 0)} / "
                f"{_bytes(performance.get('input_bytes', 0))}",
                f"{performance.get('output_files', 0)} / "
                f"{_bytes(performance.get('output_bytes', 0))}",
                len(record.get("warnings") or []),
            )
            for column, value in enumerate(values):
                item = _readonly_item(value)
                if column == 0:
                    item.setData(Qt.UserRole, record["run_id"])
                if column == 2:
                    colour = status_colours.get(str(value))
                    if colour:
                        item.setForeground(QBrush(QColor(colour)))
                self._table.setItem(row, column, item)
        self._table.setSortingEnabled(True)
        if visible:
            self._table.selectRow(0)
        else:
            self._clear_details()
        if self._loaded_once and not self._busy:
            self._set_status(
                f"Showing {len(visible)} of {len(self.records)} recorded run(s)."
            )

    def _selected_record(self) -> Optional[Dict[str, Any]]:
        """Return the record belonging to the selected table row."""
        row = self._table.currentRow()
        item = self._table.item(row, 0) if row >= 0 else None
        run_id = item.data(Qt.UserRole) if item is not None else None
        return self._record_by_id.get(str(run_id)) if run_id else None

    def _show_selection(self) -> None:
        """Populate every detail tab from the selected run."""
        record = self._selected_record()
        if record is None:
            self._clear_details()
            return
        perf = record.get("performance") or {}
        summary = {
            "run_id": record.get("run_id"),
            "module": record.get("app_key"),
            "status": record.get("status"),
            "started_utc": record.get("start_utc"),
            "ended_utc": record.get("end_utc"),
            "duration": _seconds(perf.get("wall_s")),
            "process_cpu": _seconds(perf.get("process_cpu_s")),
            "input_files": perf.get("input_files"),
            "input_bytes": perf.get("input_bytes"),
            "input_size": _bytes(perf.get("input_bytes")),
            "output_files": perf.get("output_files"),
            "output_bytes": perf.get("output_bytes"),
            "output_size": _bytes(perf.get("output_bytes")),
            "run_folder": str(record.get("dir")),
        }
        self._selection_label.setText(
            f"{record.get('app_key')} · {record.get('status')} · "
            f"{record.get('run_id')}"
        )
        self._overview.setPlainText(_json_text(summary))
        self._settings.setPlainText(_json_text(record.get("settings") or {}))
        self._outputs.setPlainText(_json_text({
            "inputs": record.get("inputs") or {},
            "outputs": record.get("outputs") or {},
            "models": record.get("models") or {},
        }))
        warnings = record.get("warnings") or []
        failure = record.get("failure") or ""
        problem_text = (
            "WARNINGS\n"
            + ("\n\n".join(str(value) for value in warnings)
               if warnings else "None recorded.")
            + "\n\nFAILURE TRACEBACK\n"
            + (str(failure) if failure else "None.")
        )
        self._problems.setPlainText(problem_text)
        manifest = record.get("manifest") or {}
        self._environment.setPlainText(_json_text({
            "environment": record.get("environment") or {},
            "seeds": manifest.get("seeds") or {},
            "settings_sha256": manifest.get("settings_sha256"),
            "input_tree_sha256": manifest.get("input_tree_sha256"),
            "output_tree_sha256": manifest.get("output_tree_sha256"),
            "manifest_schema": manifest.get("schema_version"),
        }))
        enabled = Path(record["dir"]).is_dir()
        self._open_folder.setEnabled(enabled)
        self._copy_path.setEnabled(enabled)
        self._load_settings.setEnabled(
            bool(record.get("settings"))
            and str(record.get("app_key") or "") not in ("", "unknown")
        )

    def _clear_details(self) -> None:
        """Reset detail panes and actions when nothing is selected."""
        self._selection_label.setText("Select a run to inspect it.")
        for widget in (
            self._overview, self._settings, self._outputs, self._problems,
            self._environment,
        ):
            widget.clear()
        self._open_folder.setEnabled(False)
        self._copy_path.setEnabled(False)
        self._load_settings.setEnabled(False)

    def _open_selected_folder(self) -> None:
        """Open the selected run folder in the platform file manager."""
        record = self._selected_record()
        if record is not None:
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(Path(record["dir"]).resolve()))
            )

    def _copy_selected_path(self) -> None:
        """Copy the selected run folder path to the clipboard."""
        record = self._selected_record()
        if record is not None:
            QApplication.clipboard().setText(str(record["dir"]))
            self._set_status("Run-folder path copied.")

    def _load_selected_settings(self) -> None:
        """Ask MainWindow to open the run's module with its exact settings."""
        record = self._selected_record()
        if record is not None and record.get("settings"):
            self.settings_requested.emit(
                str(record.get("app_key") or ""),
                dict(record["settings"]),
            )

    def select_run(self, run_dir: Any) -> bool:
        """Select ``run_dir`` when present in the loaded table.

        :param run_dir: run folder path or folder name.
        :returns: whether a matching visible row was selected.
        """
        wanted = Path(run_dir).name
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item is not None and item.data(Qt.UserRole) == wanted:
                self._table.selectRow(row)
                return True
        return False

    def _set_status(self, text: str, *, error: bool = False) -> None:
        """Set the inline status message with theme-aware severity."""
        palette = active_palette()
        self._status.setText(text)
        self._status.setStyleSheet(
            f"color: {palette['error' if error else 'fg_muted']}; "
            "background: transparent;"
        )
