"""
Annotator Agreement — how much do two annotation passes actually agree?

The Annotate app writes one ``INTEGER`` column per pass into
``png_list``. Two people labelling the same crops therefore leave two
columns side by side, and the only question that matters before either
set is used as ground truth is: *do they agree, and where don't they?*

Layout::

    ┌───────────────────────────────────────────────────────────────────┐
    │ /data/plate1/measurements/measurements.db  [DB…] [Run folder…]    │
    │ Read-only (mode=ro) — nothing here can modify your annotations.   │
    ├──────────────┬────────────────────────────────────────────────────┤
    │ Annotators   │ a       b      n   abstain   raw    κ   band       │
    │ ☑ alice      │ alice   bob  410      92   88.0% +0.61 substantial │
    │ ☑ bob        ├────────────────────────────────────────────────────┤
    │ ☐ carol      │ Confusion  [alice vs bob ▾]     1     2            │
    │              │                              1  310    22          │
    │ [Compute]    │                              2   27    51          │
    ├──────────────┴────────────────────────────────────────────────────┤
    │ Overall Cohen's κ +0.61 (substantial) · raw 88.0% · 49 to review   │
    ├───────────────────────────────────┬───────────────────────────────┤
    │ Disagreements  png_path  alice bob│      [crop preview]           │
    │  …/A01_1_cell_3.png        1    2 │                               │
    └───────────────────────────────────┴───────────────────────────────┘

Design notes:

* **Read-only, structurally.** Every connection goes through
  :mod:`spacr.agreement`, which opens the file with ``file:…?mode=ro``
  and ``PRAGMA query_only = ON`` — the same approach as the Database
  Browser. Adjudicating a disagreement is a job for the Annotate app;
  this screen only ever looks.
* **Off the GUI thread.** The report reads the whole ``png_list``
  annotation block, so it runs through :func:`spacr.qt.bridge.make_thread`
  like every other spaCR job. Tests pass ``threaded=False``.
* **No modal dialogs on any error path.** "You only picked one column",
  "that file isn't a database", "this table has no annotation columns" —
  all of it lands in an inline status label. A QMessageBox would hang a
  headless run.
* **κ is never quoted alone.** Raw percent agreement, the number of
  compared rows, the number of abstentions and the reason κ is undefined
  (when it is) are all on screen next to it. See :mod:`spacr.agreement`
  for why that matters on a screen where 98 % of cells are negative.
"""
from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ... import agreement as agree
from ..bridge import make_thread
from ..theme import SPACING, active_palette
from ..widgets import Divider
from .db_browser import resolve_db_path

__all__ = ["AgreementScreen", "DEFAULT_REVIEW_LIMIT", "format_kappa"]


#: Rows of the disagreement review list fetched by default. The list is a
#: work queue for a human, not a data dump — 500 is already a long day.
DEFAULT_REVIEW_LIMIT = 500
REVIEW_LIMIT_RANGE = (10, 100000)

#: Thumbnail edge for the crop preview, in pixels.
PREVIEW_PX = 260

_KAPPA_HEADERS = ("annotator A", "annotator B", "n compared", "abstentions",
                  "raw agreement", "κ", "interpretation")


def format_kappa(value: Any) -> str:
    """Render a κ for display — ``"undefined"`` rather than a fake number.

    :param value: a κ, possibly ``nan``.
    :returns: signed 3-decimal string, or ``"undefined"``.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "undefined"
    return "undefined" if math.isnan(v) else f"{v:+.3f}"


def _format_pct(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if math.isnan(v) else f"{v:.1%}"


def _cell(text: str) -> QTableWidgetItem:
    """A read-only table cell."""
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    return item


class AgreementScreen(QWidget):
    """Inter-annotator agreement over the annotation columns of a database.

    :param threaded: compute the report on a worker thread (the default).
        Tests pass ``False`` for deterministic, synchronous behaviour.
    :ivar last_error: text of the most recent failure, ``""`` when the
        last operation succeeded. Errors are *only* ever reported here
        and in the inline status label — never in a modal dialog.
    """

    #: emitted with the resolved path whenever a database opens
    database_opened = Signal(str)
    #: emitted after every compute job settles (ok or not)
    job_finished = Signal(bool)
    #: internal relay that hops a worker completion back onto the GUI thread
    _job_settled = Signal(bool)

    def __init__(self, parent=None, threaded: bool = True):
        super().__init__(parent)
        self._threaded = bool(threaded)
        self._db_path: str = ""
        self._candidates: List[str] = []
        self._report: Optional[agree.AgreementReport] = None
        self._disagreements = None      # pd.DataFrame | None
        self._busy = False
        # Ownership list for in-flight (QThread, worker) pairs — a QThread
        # collected while still running takes the process down with it.
        # Same idiom as DbBrowserScreen._jobs.
        self._jobs: List[tuple] = []
        self._pending: List[tuple] = []
        self._thread = None
        self._worker = None
        self.last_error: str = ""
        self._job_settled.connect(self._on_job_settled)

        self._build_ui()
        from ..dnd import install_dropzone
        from ..dnd_handlers import get_handler
        install_dropzone(self, get_handler("agreement"), self)
        self._set_status(
            "Choose a measurements.db (or a run folder), tick two or more "
            "annotation columns, then Compute agreement.")
        self._update_controls()

    # -- construction ------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                 SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["md"])

        title = QLabel("Annotator Agreement")
        title.setObjectName("DisplayHeading")
        outer.addWidget(title)

        subtitle = QLabel(
            "Cohen's κ between annotation columns (Fleiss' κ for three or "
            "more), plus the rows they disagree on. Read-only: the database "
            "is opened with mode=ro, so nothing here can change your "
            "annotations.")
        subtitle.setObjectName("Muted")
        subtitle.setWordWrap(True)
        outer.addWidget(subtitle)

        outer.addWidget(Divider())

        # ── Source row ────────────────────────────────────────────────
        src_row = QHBoxLayout()
        src_row.setSpacing(SPACING["sm"])
        self._path_edit = QLineEdit(self)
        self._path_edit.setPlaceholderText(
            "…/measurements/measurements.db  — or a run folder")
        self._path_edit.setClearButtonEnabled(True)
        self._path_edit.returnPressed.connect(self._on_open_typed_path)
        self._btn_pick_db = QPushButton("Choose database…", self)
        self._btn_pick_db.clicked.connect(self._pick_database)
        self._btn_pick_src = QPushButton("Choose run folder…", self)
        self._btn_pick_src.clicked.connect(self._pick_run_folder)
        self._btn_open = QPushButton("Open", self)
        self._btn_open.clicked.connect(self._on_open_typed_path)
        src_row.addWidget(self._path_edit, 1)
        src_row.addWidget(self._btn_pick_db)
        src_row.addWidget(self._btn_pick_src)
        src_row.addWidget(self._btn_open)
        outer.addLayout(src_row)

        # ── Annotators | results ──────────────────────────────────────
        split = QSplitter(Qt.Horizontal, self)

        left = QWidget(split)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(SPACING["xs"])
        left_layout.addWidget(QLabel("Annotation columns"))
        hint = QLabel("Tick two or more — one annotator cannot disagree "
                      "with anybody.")
        hint.setObjectName("Caption")
        hint.setWordWrap(True)
        left_layout.addWidget(hint)
        self._columns_list = QListWidget(left)
        self._columns_list.setSelectionMode(QAbstractItemView.NoSelection)
        self._columns_list.itemChanged.connect(lambda *_: self._update_controls())
        left_layout.addWidget(self._columns_list, 1)
        self._btn_compute = QPushButton("Compute agreement", left)
        self._btn_compute.clicked.connect(self.compute)
        left_layout.addWidget(self._btn_compute)
        split.addWidget(left)

        right = QWidget(split)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(SPACING["xs"])

        right_layout.addWidget(QLabel("Pairwise agreement"))
        self._kappa_table = QTableWidget(0, len(_KAPPA_HEADERS), right)
        self._kappa_table.setHorizontalHeaderLabels(list(_KAPPA_HEADERS))
        self._prepare_table(self._kappa_table)
        self._kappa_table.currentCellChanged.connect(
            lambda row, *_: self._on_pair_row_changed(row))
        right_layout.addWidget(self._kappa_table, 1)

        conf_row = QHBoxLayout()
        conf_row.setSpacing(SPACING["sm"])
        conf_row.addWidget(QLabel("Confusion matrix"))
        self._pair_combo = QComboBox(right)
        self._pair_combo.setMinimumWidth(220)
        self._pair_combo.currentIndexChanged.connect(
            lambda *_: self._show_confusion(self._pair_combo.currentIndex()))
        conf_row.addWidget(self._pair_combo)
        conf_row.addStretch(1)
        right_layout.addLayout(conf_row)

        self._confusion_table = QTableWidget(0, 0, right)
        self._prepare_table(self._confusion_table)
        right_layout.addWidget(self._confusion_table, 1)

        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([260, 860])
        outer.addWidget(split, 1)

        # ── Summary ───────────────────────────────────────────────────
        self._summary = QLabel("", self)
        self._summary.setWordWrap(True)
        self._summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        outer.addWidget(self._summary)

        outer.addWidget(Divider())

        # ── Disagreement review ───────────────────────────────────────
        review_row = QHBoxLayout()
        review_row.setSpacing(SPACING["sm"])
        self._review_label = QLabel("Disagreement review", self)
        review_row.addWidget(self._review_label)
        review_row.addStretch(1)
        review_row.addWidget(QLabel("Max rows", self))
        self._limit_box = QSpinBox(self)
        self._limit_box.setRange(*REVIEW_LIMIT_RANGE)
        self._limit_box.setSingleStep(50)
        self._limit_box.setValue(DEFAULT_REVIEW_LIMIT)
        self._limit_box.setToolTip(
            "(int) How many disagreeing rows to pull into the review list.")
        review_row.addWidget(self._limit_box)
        outer.addLayout(review_row)

        review_split = QSplitter(Qt.Horizontal, self)
        self._review_table = QTableWidget(0, 0, review_split)
        self._prepare_table(self._review_table)
        self._review_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._review_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._review_table.currentCellChanged.connect(
            lambda row, *_: self.select_disagreement(row))
        review_split.addWidget(self._review_table)

        preview = QWidget(review_split)
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(SPACING["xs"])
        self._crop_label = QLabel("", preview)
        self._crop_label.setAlignment(Qt.AlignCenter)
        self._crop_label.setMinimumSize(PREVIEW_PX, PREVIEW_PX)
        self._crop_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        preview_layout.addWidget(self._crop_label, 1)
        self._crop_caption = QLabel("", preview)
        self._crop_caption.setObjectName("Caption")
        self._crop_caption.setWordWrap(True)
        self._crop_caption.setTextInteractionFlags(Qt.TextSelectableByMouse)
        preview_layout.addWidget(self._crop_caption)
        review_split.addWidget(preview)
        review_split.setStretchFactor(0, 1)
        review_split.setStretchFactor(1, 0)
        review_split.setSizes([760, 320])
        outer.addWidget(review_split, 1)

        self._status = QLabel("", self)
        self._status.setObjectName("Muted")
        self._status.setWordWrap(True)
        self._status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        outer.addWidget(self._status)

    @staticmethod
    def _prepare_table(table: QTableWidget) -> None:
        """Common read-only look for every result table."""
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        table.horizontalHeader().setStretchLastSection(True)

    # -- status ------------------------------------------------------------

    def _set_status(self, text: str, error: bool = False) -> None:
        """Report inline. Deliberately never a QMessageBox — a modal dialog
        would hang a headless run (and did, in MakeMasksScreen)."""
        self.last_error = text if error else ""
        palette = active_palette()
        colour = palette["error"] if error else palette["fg_muted"]
        self._status.setStyleSheet(f"color: {colour};")
        self._status.setText(text)

    def status_text(self) -> str:
        """Current inline status message (test/introspection helper)."""
        return self._status.text()

    def summary_text(self) -> str:
        """The overall-κ summary line, or ``''`` before a report exists."""
        return self._summary.text()

    # -- database ----------------------------------------------------------

    def _pick_database(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open measurements database", "",
            "SQLite databases (*.db *.sqlite *.sqlite3);;All files (*)")
        if path:
            self.set_database(path)

    def _pick_run_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose a run folder", "")
        if path:
            self.set_database(path)

    def _on_open_typed_path(self) -> None:
        self.set_database(self._path_edit.text())

    def set_database(self, path: str) -> bool:
        """Open ``path`` read-only and list its annotation columns.

        Accepts the database file or a run ``src`` folder. Any problem
        (missing file, not a database, no ``png_list``) is reported in the
        status label and returns ``False`` — this never raises.

        :returns: True when a database with at least one annotation
            column was opened.
        """
        self._clear_results()
        self._columns_list.clear()
        self._candidates = []
        self._db_path = ""
        try:
            resolved = resolve_db_path(path)
            candidates = agree.annotation_columns(resolved)
        except Exception as e:
            self._set_status(str(e) or e.__class__.__name__, error=True)
            self._update_controls()
            return False

        self._db_path = resolved
        self._candidates = list(candidates)
        self._path_edit.setText(resolved)
        self._columns_list.blockSignals(True)
        for name in candidates:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self._columns_list.addItem(item)
        self._columns_list.blockSignals(False)
        self.database_opened.emit(resolved)

        if not candidates:
            self._set_status(
                f"Opened {resolved} — but png_list has no annotation "
                f"columns yet. Label some crops in the Annotate app first; "
                f"each pass adds one column.", error=True)
            self._update_controls()
            return False
        if len(candidates) == 1:
            # Not a crash and not a dialog: one column is a legitimate
            # state, it just cannot produce an agreement number.
            self._set_status(
                f"Opened {resolved} — only one annotation column "
                f"({candidates[0]}). Agreement needs at least two: have a "
                f"second annotator label the same crops into a new column "
                f"in the Annotate app.", error=True)
            self._columns_list.item(0).setCheckState(Qt.Checked)
            self._update_controls()
            return True

        # Two candidates is the common case — tick them so Compute works
        # on the first click.
        for i in range(min(2, self._columns_list.count())):
            self._columns_list.item(i).setCheckState(Qt.Checked)
        self._set_status(
            f"Opened {resolved} (read-only) — {len(candidates)} annotation "
            f"column{'s' if len(candidates) != 1 else ''}: "
            f"{', '.join(candidates)}.")
        self._update_controls()
        return True

    def database_path(self) -> str:
        """Path of the open database, or ``''``."""
        return self._db_path

    def available_columns(self) -> List[str]:
        """Annotation columns offered for comparison."""
        return list(self._candidates)

    def selected_columns(self) -> List[str]:
        """Ticked annotation columns, in list order."""
        return [self._columns_list.item(i).text()
                for i in range(self._columns_list.count())
                if self._columns_list.item(i).checkState() == Qt.Checked]

    def select_columns(self, names) -> bool:
        """Tick exactly ``names``; report inline for anything unknown.

        :returns: True when every requested column was found.
        """
        wanted = {str(n) for n in names}
        unknown = wanted - set(self._candidates)
        self._columns_list.blockSignals(True)
        for i in range(self._columns_list.count()):
            item = self._columns_list.item(i)
            item.setCheckState(Qt.Checked if item.text() in wanted
                               else Qt.Unchecked)
        self._columns_list.blockSignals(False)
        self._update_controls()
        if unknown:
            self._set_status(
                f"Unknown annotation column(s): "
                f"{', '.join(sorted(unknown))}. Available: "
                f"{', '.join(self._candidates) or 'none'}.", error=True)
            return False
        return True

    # -- compute -----------------------------------------------------------

    def compute(self) -> bool:
        """Build the agreement report for the ticked columns.

        Everything that can go wrong here is a normal state — no database,
        one column ticked, a column that turns out to be empty — so every
        failure is inline text and a ``False`` return.

        :returns: for the synchronous path, whether the report was built;
            for the threaded path, ``True`` once the job has started.
        """
        if not self._db_path:
            self._set_status("Open a measurements database first.", error=True)
            return False
        if self._busy:
            self._set_status("A computation is already running…", error=True)
            return False
        columns = self.selected_columns()
        if len(columns) < 2:
            self._clear_results()
            self._set_status(
                f"Tick at least two annotation columns — agreement needs two "
                f"annotators (ticked: {', '.join(columns) or 'none'}). "
                f"Available: {', '.join(self._candidates) or 'none'}.",
                error=True)
            return False

        db_path = self._db_path
        limit = int(self._limit_box.value())

        def _job() -> Dict[str, Any]:
            report = agree.agreement_report(db_path, columns)
            rows = agree.disagreements(db_path, columns, limit=limit)
            return {"report": report, "disagreements": rows}

        return self._run_job(_job, self._apply_result)

    def _apply_result(self, result: Dict[str, Any]) -> None:
        self._report = result["report"]
        self._disagreements = result["disagreements"]
        self._fill_kappa_table(self._report)
        self._fill_pair_combo(self._report)
        self._fill_summary(self._report)
        self._fill_review_table(self._disagreements, self._report)
        r = self._report
        self._set_status(
            f"{r.overall_method} {format_kappa(r.overall_kappa)} "
            f"({r.interpretation}) over {r.n_complete} row(s) labelled by "
            f"all {r.n_annotators} annotators · raw agreement "
            f"{_format_pct(r.percent_agreement)} · {r.n_disagreements} "
            f"disagreement(s) · {r.n_partial} abstention row(s).")

    def report(self) -> Optional[agree.AgreementReport]:
        """The most recent :class:`~spacr.agreement.AgreementReport`, or None."""
        return self._report

    # -- result rendering --------------------------------------------------

    def _clear_results(self) -> None:
        self._report = None
        self._disagreements = None
        self._kappa_table.setRowCount(0)
        self._confusion_table.setRowCount(0)
        self._confusion_table.setColumnCount(0)
        self._pair_combo.blockSignals(True)
        self._pair_combo.clear()
        self._pair_combo.blockSignals(False)
        self._review_table.setRowCount(0)
        self._review_table.setColumnCount(0)
        self._review_label.setText("Disagreement review")
        self._summary.setText("")
        self._crop_label.setPixmap(QPixmap())
        self._crop_label.setText("")
        self._crop_caption.setText("")

    def _fill_kappa_table(self, report: agree.AgreementReport) -> None:
        table = self._kappa_table
        table.blockSignals(True)
        table.setRowCount(len(report.pairs))
        for r, pair in enumerate(report.pairs):
            cells = (
                pair.column_a,
                pair.column_b,
                f"{pair.n_compared:,}",
                f"{pair.n_abstained:,}",
                _format_pct(pair.percent_agreement),
                format_kappa(pair.kappa),
                pair.interpretation,
            )
            for c, text in enumerate(cells):
                item = _cell(text)
                if pair.note:
                    item.setToolTip(pair.note)
                table.setItem(r, c, item)
        table.blockSignals(False)
        table.resizeColumnsToContents()

    def kappa_rows(self) -> List[List[str]]:
        """The κ table as plain strings — one list per pair."""
        return [[self._kappa_table.item(r, c).text()
                 for c in range(self._kappa_table.columnCount())]
                for r in range(self._kappa_table.rowCount())]

    def _fill_pair_combo(self, report: agree.AgreementReport) -> None:
        self._pair_combo.blockSignals(True)
        self._pair_combo.clear()
        for pair in report.pairs:
            self._pair_combo.addItem(f"{pair.column_a} vs {pair.column_b}")
        self._pair_combo.blockSignals(False)
        if report.pairs:
            self._pair_combo.setCurrentIndex(0)
            self._show_confusion(0)

    def _on_pair_row_changed(self, row: int) -> None:
        """Clicking a κ row shows that pair's confusion matrix."""
        if self._report is None or not (0 <= row < len(self._report.pairs)):
            return
        if self._pair_combo.currentIndex() != row:
            self._pair_combo.setCurrentIndex(row)   # fires _show_confusion
        else:
            self._show_confusion(row)

    def _show_confusion(self, index: int) -> None:
        """Render pair ``index``'s confusion matrix into the grid."""
        table = self._confusion_table
        if self._report is None or not (0 <= index < len(self._report.pairs)):
            table.setRowCount(0)
            table.setColumnCount(0)
            return
        pair = self._report.pairs[index]
        conf = pair.confusion
        labels = [str(l) for l in conf.index]
        table.clear()
        table.setRowCount(len(labels))
        table.setColumnCount(len(labels) + 1)
        table.setHorizontalHeaderLabels(
            [f"{pair.column_a} ↓ / {pair.column_b} →"] +
            [str(l) for l in conf.columns])
        for r, label in enumerate(labels):
            table.setItem(r, 0, _cell(label))
            for c in range(len(conf.columns)):
                value = int(conf.iloc[r, c])
                item = _cell(f"{value:,}")
                if r == c:
                    item.setToolTip("agreement")
                table.setItem(r, c + 1, item)
        table.resizeColumnsToContents()

    def confusion_rows(self) -> List[List[str]]:
        """The confusion grid as plain strings, row label first."""
        return [[self._confusion_table.item(r, c).text()
                 for c in range(self._confusion_table.columnCount())]
                for r in range(self._confusion_table.rowCount())]

    def _fill_summary(self, report: agree.AgreementReport) -> None:
        bits = [
            f"<b>Overall {report.overall_method}: "
            f"{format_kappa(report.overall_kappa)}</b> "
            f"({report.interpretation}) &nbsp;·&nbsp; raw agreement "
            f"{_format_pct(report.percent_agreement)} on "
            f"{report.n_complete:,} row(s) labelled by all "
            f"{report.n_annotators} annotators &nbsp;·&nbsp; "
            f"{report.n_partial:,} abstention row(s) &nbsp;·&nbsp; "
            f"{report.n_disagreements:,} disagreement(s)."
        ]
        if report.overall_note:
            bits.append(report.overall_note)
        bits.extend(report.warnings)
        bits.append(f"<i>{report.convention}</i>")
        self._summary.setText("<br>".join(bits))

    def _fill_review_table(self, rows, report: agree.AgreementReport) -> None:
        table = self._review_table
        columns = [report.key] + list(report.columns)
        table.blockSignals(True)
        table.clear()
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.setRowCount(len(rows))
        for r in range(len(rows)):
            for c, col in enumerate(columns):
                value = rows[col].iloc[r]
                text = "—" if value is None else str(value)
                item = _cell(text)
                if c == 0:
                    item.setToolTip(str(value))
                table.setItem(r, c, item)
        table.blockSignals(False)
        table.resizeColumnsToContents()
        shown, total = len(rows), report.n_disagreements
        suffix = (f" — showing the first {shown:,} of {total:,}"
                  if shown < total else "")
        self._review_label.setText(
            f"Disagreement review — {total:,} row(s){suffix}")
        if shown:
            table.setCurrentCell(0, 0)
        else:
            self._crop_label.setText("No disagreements to review.")
            self._crop_caption.setText("")

    def disagreement_rows(self) -> List[List[str]]:
        """The review list as plain strings — key first, then each label."""
        return [[self._review_table.item(r, c).text()
                 for c in range(self._review_table.columnCount())]
                for r in range(self._review_table.rowCount())]

    def disagreement_paths(self) -> List[str]:
        """The ``png_path`` of every row in the review list."""
        return [row[0] for row in self.disagreement_rows()]

    # -- crop preview ------------------------------------------------------

    def select_disagreement(self, row: int) -> bool:
        """Show the crop for review row ``row``.

        A missing PNG is a fact about the dataset (crops get moved, or the
        database was copied without them), not an error worth a dialog —
        the preview says so and the row stays selected.

        :returns: True when an image was actually rendered.
        """
        self._crop_label.setPixmap(QPixmap())
        if self._disagreements is None or not (0 <= row < len(self._disagreements)):
            self._crop_label.setText("")
            self._crop_caption.setText("")
            return False
        key = self._report.key if self._report is not None else agree.PNG_KEY
        raw = self._disagreements[key].iloc[row]
        path = self._resolve_crop(raw)
        labels = ", ".join(
            f"{col}={'—' if self._disagreements[col].iloc[row] is None else self._disagreements[col].iloc[row]}"
            for col in (self._report.columns if self._report else []))
        self._crop_caption.setText(f"{raw}\n{labels}")
        if path is None:
            self._crop_label.setText(
                f"Crop not found on disk:\n{raw}\n\n"
                "The labels above still stand — only the picture is missing.")
            return False
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self._crop_label.setText(f"Could not read image:\n{path}")
            return False
        self._crop_label.setText("")
        self._crop_label.setPixmap(pixmap.scaled(
            PREVIEW_PX, PREVIEW_PX, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        return True

    def _resolve_crop(self, png_path: Any) -> Optional[str]:
        """Find the crop on disk, or None.

        ``png_list`` stores absolute paths from the machine that ran
        Measure. When a dataset has been copied elsewhere those break, so
        a relative path is also tried against the run folder — the
        database's grandparent, ``<src>/measurements/measurements.db``.
        """
        if png_path is None:
            return None
        p = str(png_path)
        if os.path.isfile(p):
            return p
        if self._db_path:
            src = os.path.dirname(os.path.dirname(os.path.abspath(self._db_path)))
            candidate = os.path.join(src, p.lstrip("/\\"))
            if os.path.isfile(candidate):
                return candidate
        return None

    def current_crop_path(self) -> str:
        """Text of the crop-preview caption's first line (test helper)."""
        return self._crop_caption.text().splitlines()[0] if \
            self._crop_caption.text() else ""

    def crop_message(self) -> str:
        """Text shown instead of a crop when there is no image."""
        return self._crop_label.text()

    # -- job plumbing ------------------------------------------------------

    def _run_job(self, fn: Callable[[], Any],
                 on_done: Callable[[Any], None]) -> bool:
        """Run ``fn`` off the GUI thread and hand its result to ``on_done``.

        Mirrors ``DbBrowserScreen._run_job``: one threading idiom for the
        whole Qt layer, and ``threaded=False`` runs inline while firing
        the same signals so both paths behave identically from outside.

        One detail this cannot copy from the Database Browser.
        ``PipelineWorker.finished`` is emitted *in the worker thread*, and
        PySide6 hands a plain closure connected to it a direct call — so
        the completion handler, and every widget it touches, would run off
        the GUI thread. It is chained through :attr:`_job_settled` into a
        *bound method* of this widget instead. The widget lives on the GUI
        thread, so Qt queues the call and the handler runs where every
        other widget call runs.
        """
        if not self._threaded:
            ok = True
            try:
                on_done(fn())
            except Exception as e:
                self._on_job_error(e)
                ok = False
            self._update_controls()
            self.job_finished.emit(ok)
            return ok

        box: Dict[str, Any] = {}

        def _job(payload: Dict[str, Any]) -> None:
            payload["result"] = fn()

        thread, worker = make_thread(_job, box)
        self._jobs.append((thread, worker))
        self._thread, self._worker = thread, worker
        self._pending.append((box, on_done))
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
        box, on_done = self._pending.pop(0) if self._pending else ({}, None)
        ok = bool(ok)
        if ok and on_done is not None:
            try:
                on_done(box.get("result"))
            except Exception as e:
                self._on_job_error(e)
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
        if self._thread is thread:
            self._thread = None
            self._worker = None

    def active_jobs(self) -> int:
        """How many compute threads are still winding down."""
        return len(self._jobs)

    def is_busy(self) -> bool:
        return self._busy

    def _on_worker_error_text(self, tb: str) -> None:
        """Turn a worker traceback into one inline line (never a dialog)."""
        line = ""
        for candidate in reversed(str(tb).strip().splitlines()):
            if candidate.strip():
                line = candidate.strip()
                break
        self._clear_results()
        self._set_status(f"Agreement failed: {line}", error=True)

    def _on_job_error(self, exc: Exception) -> None:
        self._clear_results()
        self._set_status(f"Agreement failed: {exc}", error=True)

    # -- enablement --------------------------------------------------------

    def _update_controls(self) -> None:
        has_db = bool(self._db_path)
        enough = len(self.selected_columns()) >= 2
        self._btn_compute.setEnabled(has_db and enough and not self._busy)
        self._columns_list.setEnabled(has_db and not self._busy)
        self._limit_box.setEnabled(has_db and not self._busy)

    # -- shutdown ----------------------------------------------------------

    def closeEvent(self, event):  # noqa: N802
        """Let every in-flight compute thread finish before the widget dies."""
        for thread, _worker in list(self._jobs):
            try:
                if thread.isRunning():
                    thread.quit()
                    thread.wait(5000)
            except RuntimeError:
                pass
        super().closeEvent(event)
