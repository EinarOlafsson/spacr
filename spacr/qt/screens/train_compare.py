"""
Training Runs — several runs' curves on one axis, with the settings diffed.

The question this screen exists to answer is "why is run B better than run A",
which today is answered by opening two folders of PDFs in one window, two
settings CSVs in another, and holding the difference in your head.

Layout::

    ┌──────────────────────────────────────────────────────────────────────┐
    │ /data/screen1/model                     [Choose folder…]  [Scan]     │
    ├──────────────────────┬───────────────────────────────────────────────┤
    │ Runs found (4)       │            ┌───────────────────────────┐      │
    │ ☑ maxvit_t/…/ep_25   │            │   accuracy, 5 series      │      │
    │   25 epochs · tr+val │            │   ╱‾‾‾‾  B val            │      │
    │ ☑ maxvit_t/…/ep_10   │            │  ╱ ─ ─   A val            │      │
    │ ☐ resnet50/…/ep_8    │            └───────────────────────────┘      │
    │   8 ep x 3 folds     ├───────────────────────────────────────────────┤
    │ ☐ maxvit_t/…/ep_3 !  │ 2 settings changed · 1 env drift · 0 drift    │
    │                      │ bucket   setting        A        B            │
    │ Metric [accuracy ▾]  │ changed  learning_rate  1e-4     1e-3         │
    │ Folds  [per fold ▾]  │ changed  batch_size     64       32           │
    │ [Overlay selected]   │ env      n_jobs         30       8            │
    ├──────────────────────┴───────────────────────────────────────────────┤
    │ ! maxvit_t/…/epochs_3: no per-epoch curves in this folder            │
    │ Clicked: maxvit_t/…/epochs_25 · val — best 0.87 @ 18, last 0.85 @ 25 │
    └──────────────────────────────────────────────────────────────────────┘

Design notes:

* **Discovery runs off the GUI thread.** A scan walks a model tree and parses
  every progress CSV under it, which on a real screen is hundreds of files. It
  goes through :func:`spacr.qt.bridge.make_thread` like every other spaCR job;
  tests pass ``threaded=False``, which runs the same code inline. Drawing stays
  on the GUI thread — by then the data is already in memory.
* **No modal dialogs on any error path.** A folder with no runs, a run with no
  curves, a metric nothing logged — all of it lands in the inline status and
  problem labels. A QMessageBox hangs a headless run.
* **Broken runs are listed, not hidden.** A folder holding checkpoints but no
  ``train.csv`` still appears, marked, with its note in the problem line. A
  scan that silently drops the folder you were looking for is worse than one
  that says what is wrong with it.
* **The diff is bucketed, never flat.** It renders
  :func:`spacr.train_compare.diff_settings`, which reuses the provenance
  bucketing from :mod:`spacr.run_journal`: environment drift (paths, hosts,
  worker counts) is shown in its own bucket instead of being counted as
  something the user changed, and schema drift is summarised. When two runs
  match, the table says "no differences" in words rather than going blank.
* **Every series says run · split · fold.** Clicking a line names its run,
  folder and both its best and last epoch, because a legend that only carries
  the run id invites reading a train curve as a held-out result.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
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
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ... import train_compare as tc
from ..bridge import make_thread
from ..theme import PALETTE, SPACING, palette_for
from ..widgets import Divider

__all__ = ["TrainCompareScreen", "APP_KEY", "APP_NAME", "APP_SECTION",
           "APP_INTRO", "FOLD_MODE_LABELS"]

#: Registration constants. ``spacr.qt.app.APPS`` and the title/intro tables in
#: ``spacr.qt.screens.app_screen`` are the registry; these are kept here so the
#: screen and the registry can be checked against each other.
APP_KEY = "train_compare"
APP_NAME = "Training Runs"
#: The SUBJECT category: comparing training runs is reading a result.
#: #16i staged this app into Alpha modules — which is where
#: ``spacr.qt.app.APPS`` files it and where Home lists it — but staging
#: says how finished an app is and this constant says what it does, so
#: it did not move. ``spacr.qt.app.subject_section`` is what the suite
#: checks it against.
APP_SECTION = "Results & QC"
APP_INTRO = (
    "Overlay the loss and accuracy curves of several training runs on one "
    "axis and see, beside them, exactly which settings differed — with "
    "environment drift bucketed away from the knobs you actually turned.")

#: Combo label -> ``spacr.train_compare.compare_runs(folds=…)`` value.
FOLD_MODE_LABELS = (
    ("per fold", "per_fold"),
    ("mean ± sd", "mean"),
    ("both", "both"),
)

_DIFF_BUCKETS = ("changed", "env", "drift")


def _cell(text: str) -> QTableWidgetItem:
    """A read-only table cell."""
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    return item


def _active_palette() -> dict:
    """The palette the plot should be drawn in, defaulting to dark."""
    theme = "dark"
    try:
        from ..preferences import get_theme
        theme = get_theme()
    except Exception:
        pass
    return palette_for("light" if theme == "light" else "dark")


class TrainCompareScreen(QWidget):
    """Compare training runs: overlaid curves plus the bucketed settings diff.

    :param threaded: discover and load runs on a worker thread (the default).
        Tests pass ``False`` for deterministic, synchronous behaviour.
    :ivar last_error: text of the most recent failure, ``""`` when the last
        operation succeeded. Errors are only ever reported here and in the
        inline status label — never in a modal dialog.
    """

    #: emitted with the number of runs found after every scan
    runs_discovered = Signal(int)
    #: emitted with the series label whenever a curve is clicked
    series_clicked = Signal(str)
    #: emitted after every job settles (ok or not)
    job_finished = Signal(bool)

    def __init__(self, parent=None, threaded: bool = True):
        super().__init__(parent)
        self._threaded = bool(threaded)
        self._root: str = ""
        self._runs: List[tc.TrainingRun] = []
        self._comparison: Optional[tc.Comparison] = None
        self._busy = False
        # Ownership list for in-flight (QThread, worker) pairs — a QThread
        # collected while still running takes the process down with it.
        self._jobs: List[tuple] = []
        self.last_error: str = ""

        self._build_ui()
        self._set_status(
            "Choose the folder your models were trained into (a dataset's "
            "model/ folder, or anything above it), then Scan.")
        self._update_controls()

    # -- construction ------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                 SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["md"])

        title = QLabel(APP_NAME)
        title.setObjectName("DisplayHeading")
        outer.addWidget(title)

        subtitle = QLabel(APP_INTRO)
        subtitle.setObjectName("Muted")
        subtitle.setWordWrap(True)
        outer.addWidget(subtitle)

        outer.addWidget(Divider())

        # ── Source row ────────────────────────────────────────────────
        src_row = QHBoxLayout()
        src_row.setSpacing(SPACING["sm"])
        self._path_edit = QLineEdit(self)
        self._path_edit.setPlaceholderText(
            "…/my_dataset/model  — or any folder above your run folders")
        self._path_edit.setClearButtonEnabled(True)
        self._path_edit.returnPressed.connect(self._on_scan_typed_path)
        self._path_edit.textChanged.connect(lambda *_: self._update_controls())
        self._btn_pick = QPushButton("Choose folder…", self)
        self._btn_pick.clicked.connect(self._pick_folder)
        self._btn_scan = QPushButton("Scan", self)
        self._btn_scan.clicked.connect(self._on_scan_typed_path)
        src_row.addWidget(self._path_edit, 1)
        src_row.addWidget(self._btn_pick)
        src_row.addWidget(self._btn_scan)
        outer.addLayout(src_row)

        # ── Runs | plot + diff ────────────────────────────────────────
        split = QSplitter(Qt.Horizontal, self)

        left = QWidget(split)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(SPACING["xs"])
        self._runs_header = QLabel("Runs found", left)
        left_layout.addWidget(self._runs_header)
        hint = QLabel("Tick two or more to overlay them. Runs marked ! are "
                      "missing curves or settings — they still diff.", left)
        hint.setObjectName("Caption")
        hint.setWordWrap(True)
        left_layout.addWidget(hint)
        self._runs_list = QListWidget(left)
        self._runs_list.setSelectionMode(QAbstractItemView.NoSelection)
        self._runs_list.itemChanged.connect(lambda *_: self._update_controls())
        left_layout.addWidget(self._runs_list, 1)

        metric_row = QHBoxLayout()
        metric_row.setSpacing(SPACING["sm"])
        metric_row.addWidget(QLabel("Metric", left))
        self._metric_combo = QComboBox(left)
        self._metric_combo.setMinimumWidth(140)
        self._metric_combo.currentIndexChanged.connect(self._on_metric_changed)
        metric_row.addWidget(self._metric_combo, 1)
        left_layout.addLayout(metric_row)

        fold_row = QHBoxLayout()
        fold_row.setSpacing(SPACING["sm"])
        fold_row.addWidget(QLabel("Folds", left))
        self._fold_combo = QComboBox(left)
        for label, _value in FOLD_MODE_LABELS:
            self._fold_combo.addItem(label)
        self._fold_combo.setToolTip(
            "How to draw a cross-validated run: every fold, the fold mean "
            "with a ±1 sd band, or both. A mean drawn as if it were one run "
            "hides the fold-to-fold spread k-fold exists to show.")
        self._fold_combo.currentIndexChanged.connect(self._on_fold_changed)
        fold_row.addWidget(self._fold_combo, 1)
        left_layout.addLayout(fold_row)

        self._btn_overlay = QPushButton("Overlay selected", left)
        self._btn_overlay.clicked.connect(self.overlay)
        left_layout.addWidget(self._btn_overlay)
        split.addWidget(left)

        right = QSplitter(Qt.Vertical, split)

        # Matplotlib canvas — created here so the same figure is reused for
        # every overlay rather than leaking one per click.
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        self._figure = Figure(figsize=(7.0, 4.2), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setMinimumHeight(240)
        self._canvas.mpl_connect("pick_event", self._on_pick)
        right.addWidget(self._canvas)

        diff_panel = QWidget(right)
        diff_layout = QVBoxLayout(diff_panel)
        diff_layout.setContentsMargins(0, 0, 0, 0)
        diff_layout.setSpacing(SPACING["xs"])
        self._diff_summary = QLabel("", diff_panel)
        self._diff_summary.setWordWrap(True)
        self._diff_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        diff_layout.addWidget(self._diff_summary)
        self._diff_table = QTableWidget(0, 0, diff_panel)
        self._diff_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._diff_table.setAlternatingRowColors(True)
        self._diff_table.verticalHeader().setVisible(False)
        self._diff_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Interactive)
        self._diff_table.horizontalHeader().setStretchLastSection(True)
        diff_layout.addWidget(self._diff_table, 1)
        right.addWidget(diff_panel)
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 2)

        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([300, 820])
        outer.addWidget(split, 1)

        self._problems = QLabel("", self)
        self._problems.setWordWrap(True)
        self._problems.setTextInteractionFlags(Qt.TextSelectableByMouse)
        outer.addWidget(self._problems)

        self._picked = QLabel("", self)
        self._picked.setObjectName("Caption")
        self._picked.setWordWrap(True)
        self._picked.setTextInteractionFlags(Qt.TextSelectableByMouse)
        outer.addWidget(self._picked)

        self._status = QLabel("", self)
        self._status.setObjectName("Muted")
        self._status.setWordWrap(True)
        self._status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        outer.addWidget(self._status)

    # -- status ------------------------------------------------------------

    def _set_status(self, text: str, error: bool = False) -> None:
        """Report inline. Never a QMessageBox — a modal hangs a headless run."""
        self.last_error = text if error else ""
        colour = PALETTE["error"] if error else PALETTE["fg_muted"]
        self._status.setStyleSheet(f"color: {colour};")
        self._status.setText(text)

    def status_text(self) -> str:
        return self._status.text()

    def summary_text(self) -> str:
        """The line above the diff table."""
        return self._diff_summary.text()

    def problem_text(self) -> str:
        """Every note from every discovered run, one per line."""
        return self._problems.text()

    def picked_text(self) -> str:
        """Description of the last clicked series."""
        return self._picked.text()

    # -- scanning ----------------------------------------------------------

    def _pick_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Choose a folder to scan for training runs", "")
        if path:
            self.scan(path)

    def _on_scan_typed_path(self) -> None:
        self.scan(self._path_edit.text().strip())

    def scan(self, root: Any) -> bool:
        """Discover training runs under ``root`` and list them.

        Runs off the GUI thread unless the screen was built with
        ``threaded=False``. Every failure is reported inline.

        :param root: folder to walk.
        :returns: True when the scan started (or, unthreaded, succeeded).
        """
        path = str(root or "").strip()
        if not path:
            self._set_status("Type or choose a folder to scan.", error=True)
            return False
        path = os.path.expanduser(path)
        if not os.path.isdir(path):
            self._set_status(f"Not a folder: {path}", error=True)
            return False

        self._root = path
        self._path_edit.setText(path)
        self._set_status(f"Scanning {path} …")

        def _job():
            return tc.find_runs(path)

        return self._run_job(_job, self._apply_runs)

    def _apply_runs(self, runs: Any) -> None:
        self._runs = list(runs or [])
        self._comparison = None
        self._fill_runs_list()
        self._fill_problems()
        self._clear_diff()
        self._clear_plot()
        n = len(self._runs)
        if n == 0:
            self._set_status(
                f"No training runs under {self._root}. A run folder is the "
                f"one holding train.csv / validation.csv (or fold_1/, fold_2/ "
                f"for a cross-validated run) — that is <dataset>/model/"
                f"<model_type>/<channels>/epochs_<N>.")
        else:
            broken = sum(1 for r in self._runs if not r.has_curves)
            extra = f" ({broken} with no curves)" if broken else ""
            self._set_status(
                f"Found {n} run{'s' if n != 1 else ''}{extra}. Tick two or "
                f"more and press Overlay selected.")
        self.runs_discovered.emit(n)

    def _fill_runs_list(self) -> None:
        self._runs_list.blockSignals(True)
        self._runs_list.clear()
        for run in self._runs:
            item = QListWidgetItem(run.summary_line())
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            item.setData(Qt.UserRole, run.run_id)
            if run.notes:
                item.setToolTip("\n".join(run.notes))
            if not run.has_curves:
                item.setForeground(QBrush(QColor(PALETTE["fg_dim"])))
            self._runs_list.addItem(item)
        self._runs_list.blockSignals(False)
        self._runs_header.setText(f"Runs found ({len(self._runs)})")
        self._fill_metric_combo(tc.available_metrics(self._runs))
        self._update_controls()

    def _fill_metric_combo(self, metrics: Sequence[str]) -> None:
        previous = self._metric_combo.currentText()
        self._metric_combo.blockSignals(True)
        self._metric_combo.clear()
        self._metric_combo.addItems(list(metrics))
        if previous in metrics:
            self._metric_combo.setCurrentText(previous)
        self._metric_combo.blockSignals(False)

    def _fill_problems(self) -> None:
        lines = [f"! {r.run_id}: {n}" for r in self._runs for n in r.notes]
        if lines:
            self._problems.setStyleSheet(f"color: {PALETTE['warning']};")
            self._problems.setText("\n".join(lines))
        else:
            self._problems.setStyleSheet("")
            self._problems.setText("")

    # -- introspection helpers (used by tests and by callers) -------------

    def root(self) -> str:
        return self._root

    def runs(self) -> List[tc.TrainingRun]:
        return list(self._runs)

    def run_ids(self) -> List[str]:
        return [r.run_id for r in self._runs]

    def run_rows(self) -> List[str]:
        return [self._runs_list.item(i).text()
                for i in range(self._runs_list.count())]

    def available_metrics(self) -> List[str]:
        return [self._metric_combo.itemText(i)
                for i in range(self._metric_combo.count())]

    def metric(self) -> str:
        return self._metric_combo.currentText()

    def set_metric(self, name: str) -> bool:
        """Pick the metric to draw; re-draws when a comparison exists."""
        if name not in self.available_metrics():
            self._set_status(f"No run logged '{name}'.", error=True)
            return False
        self._metric_combo.setCurrentText(name)
        return True

    def fold_mode(self) -> str:
        idx = max(0, self._fold_combo.currentIndex())
        return FOLD_MODE_LABELS[idx][1]

    def set_fold_mode(self, mode: str) -> bool:
        for i, (_label, value) in enumerate(FOLD_MODE_LABELS):
            if value == mode:
                self._fold_combo.setCurrentIndex(i)
                return True
        self._set_status(f"Unknown fold mode {mode!r}.", error=True)
        return False

    def selected_run_ids(self) -> List[str]:
        out = []
        for i in range(self._runs_list.count()):
            item = self._runs_list.item(i)
            if item.checkState() == Qt.Checked:
                out.append(item.data(Qt.UserRole))
        return out

    def select_runs(self, run_ids: Sequence[str]) -> bool:
        """Tick exactly these run ids. Unknown ids are reported inline."""
        wanted = set(run_ids or ())
        known = set(self.run_ids())
        missing = sorted(wanted - known)
        self._runs_list.blockSignals(True)
        for i in range(self._runs_list.count()):
            item = self._runs_list.item(i)
            item.setCheckState(Qt.Checked
                               if item.data(Qt.UserRole) in wanted
                               else Qt.Unchecked)
        self._runs_list.blockSignals(False)
        self._update_controls()
        if missing:
            self._set_status(f"No such run: {', '.join(missing)}", error=True)
            return False
        return True

    def comparison(self) -> Optional[tc.Comparison]:
        return self._comparison

    def figure(self):
        """The live matplotlib figure (one per screen, reused every overlay)."""
        return self._figure

    def series_labels(self) -> List[str]:
        """Labels of the currently drawn series, in draw order."""
        mapping = getattr(self._figure, "spacr_series_by_label", {}) or {}
        return list(mapping)

    # -- overlay -----------------------------------------------------------

    def _on_metric_changed(self, *_a) -> None:
        if self._comparison is not None:
            self._draw()

    def _on_fold_changed(self, *_a) -> None:
        if self._comparison is not None:
            self.overlay()

    def overlay(self) -> bool:
        """Compare the ticked runs: draw their curves and fill the diff table.

        Everything the drawing needs is already in memory after
        :meth:`scan`, so this runs on the GUI thread.

        :returns: True when a comparison was produced.
        """
        chosen = set(self.selected_run_ids())
        runs = [r for r in self._runs if r.run_id in chosen]
        if not runs:
            self._set_status("Tick at least one run to overlay.", error=True)
            return False

        try:
            comparison = tc.compare_runs(runs, folds=self.fold_mode())
        except Exception as e:
            self._on_job_error(e)
            return False

        self._comparison = comparison
        self._fill_metric_combo(comparison.metrics or self.available_metrics())
        self._fill_diff(comparison)
        self._draw()

        plotted = len(self.series_labels())
        no_curves = [r.run_id for r in runs if not r.has_curves]
        bits = [f"{plotted} series from {len(runs)} run"
                f"{'s' if len(runs) != 1 else ''}"]
        if comparison.lengths_differ():
            bits.append("different epoch counts — each curve is drawn to its "
                        "own length")
        if no_curves:
            bits.append(f"no curves in {', '.join(no_curves)} — settings only")
        self._set_status(" · ".join(bits))
        return True

    def _clear_plot(self) -> None:
        self._figure.clear()
        self._figure.spacr_series_by_label = {}
        self._canvas.draw_idle()
        self._picked.setText("")

    def _draw(self) -> None:
        """Redraw the curves for the current metric into the shared figure."""
        if self._comparison is None:
            return
        pal = _active_palette()
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        metric = self.metric() or (self._comparison.metrics[0]
                                   if self._comparison.metrics else "accuracy")
        tc.plot_curves(self._comparison, metric, ax=ax)
        self._style_axes(ax, pal)
        self._canvas.draw_idle()
        self._picked.setText("")

    @staticmethod
    def _style_axes(ax, pal: dict) -> None:
        """Match the plot to the app palette so it doesn't glare."""
        fig = ax.figure
        fig.set_facecolor(pal["surface"])
        ax.set_facecolor(pal["surface"])
        for spine in ax.spines.values():
            spine.set_color(pal["border"])
        ax.tick_params(colors=pal["fg_muted"])
        ax.xaxis.label.set_color(pal["fg_muted"])
        ax.yaxis.label.set_color(pal["fg_muted"])
        ax.title.set_color(pal["fg"])
        for text in ax.texts:
            text.set_color(pal["fg_muted"])
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(pal["surface_alt"])
            legend.get_frame().set_edgecolor(pal["border"])
            for text in legend.get_texts():
                text.set_color(pal["fg"])

    # -- diff table --------------------------------------------------------

    def _clear_diff(self) -> None:
        self._diff_table.clear()
        self._diff_table.setRowCount(0)
        self._diff_table.setColumnCount(0)
        self._diff_summary.setText("")

    def _fill_diff(self, comparison: tc.Comparison) -> None:
        diff = comparison.settings_diff
        ids = list(diff.get("run_ids") or [])
        changed = list(diff.get("changed") or [])
        env = list(diff.get("env") or []) + list(diff.get("env_manifest") or [])
        drift = list(diff.get("drift") or [])

        if len(ids) < 2:
            which = ", ".join(diff.get("no_settings") or []) or "these runs"
            self._diff_summary.setText(
                f"Settings not comparable — fewer than two selected runs have "
                f"a settings snapshot ({which} had none).")
            self._set_table(("setting", "difference"),
                            [["—", "no settings to compare"]])
            return

        if diff.get("identical"):
            # An empty table reads as a failure; say it in words instead.
            self._diff_summary.setText(
                f"No differences — all {len(ids)} selected runs ran with "
                f"identical settings.")
            self._set_table(
                ("setting", "difference"),
                [["—", f"No differences — all {len(ids)} selected runs ran "
                       f"with identical settings ({diff.get('shared', 0)} "
                       f"keys compared, none differ)."]])
            return

        self._diff_summary.setText(
            f"{len(changed)} setting(s) changed of {diff.get('shared', 0)} "
            f"shared · {len(env)} environment drift · {len(drift)} schema "
            f"drift. Environment drift (paths, hosts, worker counts, "
            f"versions) is bucketed separately — none of it is a modelling "
            f"decision.")

        headers = ("bucket", "setting", *ids)
        rows: List[List[str]] = []
        for entry in changed:
            rows.append(["changed", entry["key"]]
                        + [tc.render_setting_value(entry["values"].get(rid), 40)
                           for rid in ids])
        for entry in env:
            rows.append(["env", entry["key"]]
                        + [tc.render_setting_value(entry["values"].get(rid), 40)
                           for rid in ids])
        for entry in drift:
            present = set(entry.get("present") or ())
            rows.append(["drift", entry["key"]]
                        + ["recorded" if rid in present else "not recorded"
                           for rid in ids])
        self._set_table(headers, rows)

    def _set_table(self, headers: Sequence[str],
                   rows: Sequence[Sequence[str]]) -> None:
        self._diff_table.clear()
        self._diff_table.setColumnCount(len(headers))
        self._diff_table.setHorizontalHeaderLabels(list(headers))
        self._diff_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                self._diff_table.setItem(r, c, _cell(str(value)))
        self._diff_table.resizeColumnsToContents()

    def diff_headers(self) -> List[str]:
        return [self._diff_table.horizontalHeaderItem(c).text()
                if self._diff_table.horizontalHeaderItem(c) else ""
                for c in range(self._diff_table.columnCount())]

    def diff_rows(self) -> List[List[str]]:
        out = []
        for r in range(self._diff_table.rowCount()):
            row = []
            for c in range(self._diff_table.columnCount()):
                item = self._diff_table.item(r, c)
                row.append(item.text() if item else "")
            out.append(row)
        return out

    # -- picking -----------------------------------------------------------

    def _on_pick(self, event) -> None:
        artist = getattr(event, "artist", None)
        getter = getattr(artist, "get_label", None)
        label = getter() if callable(getter) else ""
        self.identify_series(str(label))

    def identify_series(self, label: str) -> str:
        """Name the run behind a series label and report it inline.

        :returns: the description shown, or ``''`` when the label is unknown.
        """
        mapping = getattr(self._figure, "spacr_series_by_label", {}) or {}
        series = mapping.get(label)
        if series is None:
            self._picked.setText("")
            return ""
        run = next((r for r in self._runs if r.run_id == series.run_id), None)
        metric = self.metric()
        lo, hi = series.epoch_range()
        bits = [f"{series.label}",
                f"epochs {lo}–{hi}" if lo != hi else f"epoch {hi}"]
        best = series.best(metric)
        last = series.last(metric)
        if last is not None:
            bits.append(f"last {metric} {last['value']:.4f} @ {last['epoch']}")
        if best is not None:
            bits.append(f"best {metric} {best['value']:.4f} @ {best['epoch']} "
                        f"(chosen on this same curve, so optimistic)")
        if run is not None:
            bits.append(str(run.path))
        text = " · ".join(bits)
        self._picked.setText(text)
        self.series_clicked.emit(series.label)
        return text

    # -- job plumbing ------------------------------------------------------

    def _run_job(self, fn: Callable[[], Any],
                 on_done: Callable[[Any], None]) -> bool:
        """Run ``fn`` off the GUI thread and hand its result to ``on_done``.

        Mirrors ``AgreementScreen._run_job`` / ``ModelCompareScreen._run_job``:
        one threading idiom for the whole Qt layer, and ``threaded=False``
        runs inline while firing the same signals.
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
        worker.error.connect(self._on_worker_error_text)

        def _finished(ok: bool) -> None:
            self._busy = False
            if ok:
                try:
                    on_done(box.get("result"))
                except Exception as e:
                    self._on_job_error(e)
                    ok = False
            self._update_controls()
            self.job_finished.emit(ok)

        worker.finished.connect(_finished)
        thread.finished.connect(lambda t=thread: self._retire_job(t))
        self._busy = True
        self._update_controls()
        thread.start()
        return True

    def _retire_job(self, thread) -> None:
        self._jobs = [(t, w) for (t, w) in self._jobs if t is not thread]

    def active_jobs(self) -> int:
        return len(self._jobs)

    def is_busy(self) -> bool:
        return self._busy

    def _on_worker_error_text(self, tb: str) -> None:
        last = [ln for ln in str(tb).strip().splitlines() if ln.strip()]
        self._set_status(last[-1] if last else "Scan failed.", error=True)

    def _on_job_error(self, exc: Exception) -> None:
        self._set_status(f"{type(exc).__name__}: {exc}", error=True)

    def _update_controls(self) -> None:
        has_root = bool(self._path_edit.text().strip())
        self._btn_scan.setEnabled(has_root and not self._busy)
        self._btn_overlay.setEnabled(
            bool(self.selected_run_ids()) and not self._busy)

    def closeEvent(self, event):  # noqa: N802 — Qt naming
        for thread, _worker in list(self._jobs):
            try:
                thread.quit()
                thread.wait(2000)
            except Exception:
                pass
        self._jobs.clear()
        super().closeEvent(event)
