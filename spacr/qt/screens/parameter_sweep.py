"""Parameter Sweep — run the regression under many settings and compare.

A pooled screen has no single correct analysis. It has a model family, an
aggregation rule, a unit of analysis, nuisance structure, a multiple-testing
correction and several filtration cutoffs, and the honest question is not
"what did my settings say" but "which findings survive the settings I could
defensibly have chosen".

This screen asks that question. Tick the axes to vary, pin the rest, press
Start: every trial runs into its own folder and one tidy row per trial lands
in the table -- settings, whether it ran, how many hits it called, and where
the named controls ended up.

Three things it is careful about, each learned the hard way:

* **It cannot take the machine down.** The worker count is sized from FREE
  MEMORY, not the core count, and is clamped by
  :func:`spacr.parameter_sweep.recommended_workers`. Trials run niced, at
  idle I/O priority, and the pool stops growing when free memory falls below
  a floor. A sweep sized from 32 cores exhausted memory and killed the
  editor -- twice -- which is why none of this is optional.
* **A failed trial is a result.** Many combinations are illegal by
  construction; those are recorded with their reason and the sweep continues.
* **It is resumable.** The results table is rewritten after every trial, so a
  sweep that is stopped, or whose machine dies, still leaves everything it
  learned.
"""
from __future__ import annotations

import os

APP_KEY = "parameter_sweep"
APP_NAME = "Parameter Sweep"
APP_DESCRIPTION = (
    "Run the regression under many settings combinations and compare what "
    "each one concludes")
APP_INTRO = (
    "Point this at the same score and count CSVs the regression uses, tick "
    "the settings you want varied, and start. Each trial runs in its own "
    "folder and the table fills in as they finish: model family, correction, "
    "analysis unit, how many hits were called, and the rank of your positive "
    "control in each. The spread across settings is the result — a screen "
    "whose hit count swings from 2 to 400 depending on the correction has "
    "not been analysed, it has been chosen. Workers are sized from free "
    "memory and run at low priority, so a sweep will not slow the machine "
    "down or compete with anything else you are doing.")
APP_TRANSLATIONS = (
    "Parametersvep", "Parameter-Sweep", "Barrido de parámetros",
    "参数扫描", "Varredura de parâmetros", "पैरामीटर स्वीप",
    "매개변수 스윕", "Færibreytusveip", "Balayage de paramètres")

__all__ = ["APP_KEY", "APP_NAME", "APP_DESCRIPTION", "APP_INTRO", "register"]


def _make_screen(app_key=None, host=None):
    """Build the screen lazily -- it pulls in the sweep engine and pandas."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
        QLineEdit, QMessageBox, QProgressBar, QPushButton, QScrollArea,
        QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout,
        QWidget,
    )

    from ..job_runner import JobRunner
    from ..widgets.file_list import FilePathListWidget
    from ...parameter_sweep import (
        DEFAULT_SWEEP_SPACE, SweepSpace, build_trials, recommended_workers,
    )

    class ParameterSweepScreen(QWidget):
        def __init__(self, host=None):
            # `host` is the main window the registry passes for navigation,
            # not a Qt parent.
            super().__init__()
            self.host = host
            self._results = None
            self._runner = JobRunner(self, app_key=APP_KEY)

            outer = QVBoxLayout(self)
            splitter = QSplitter(Qt.Horizontal, self)
            outer.addWidget(splitter)

            # ---- left: what to sweep --------------------------------------
            left = QScrollArea(self)
            left.setWidgetResizable(True)
            left.setMinimumWidth(420)
            panel = QWidget()
            form = QVBoxLayout(panel)

            inputs = QGroupBox("Inputs", panel)
            inputs_form = QFormLayout(inputs)
            self.score_data = FilePathListWidget(
                kind="table", title="Choose per-object score CSVs")
            self.count_data = FilePathListWidget(
                kind="table", title="Choose gRNA count CSVs (one per plate)")
            self.destination = QLineEdit(panel)
            self.destination.setPlaceholderText(
                "Folder for the trial folders and the results table")
            self.dependent_variable = QLineEdit("pred", panel)
            inputs_form.addRow("Score CSVs", self.score_data)
            inputs_form.addRow("Count CSVs", self.count_data)
            inputs_form.addRow("Response column", self.dependent_variable)
            inputs_form.addRow("Output folder", self.destination)
            form.addWidget(inputs)

            axes = QGroupBox("Settings to sweep", panel)
            axes_layout = QVBoxLayout(axes)
            axes_layout.addWidget(QLabel(
                "Tick a setting to vary it. Unticked settings keep the value "
                "on the right, applied to every trial.", panel))
            self._axis_rows = {}
            grid = QFormLayout()
            for key, values in DEFAULT_SWEEP_SPACE.items():
                row = QWidget(panel)
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                include = QCheckBox(row)
                # Default on for the axes the comparison is actually about;
                # the filtration cutoffs are usually pinned by the user.
                include.setChecked(key not in (
                    "fraction_threshold", "min_cell_count", "fdr_alpha",
                    "min_n", "outlier_detection", "threshold_method"))
                editor = QLineEdit(
                    ", ".join("None" if v is None else str(v) for v in values),
                    row)
                editor.setToolTip(
                    f"Comma-separated values for {key}. When the box on the "
                    f"left is unticked, the FIRST value is used for every "
                    f"trial.")
                row_layout.addWidget(include)
                row_layout.addWidget(editor, 1)
                grid.addRow(key, row)
                self._axis_rows[key] = (include, editor)
            axes_layout.addLayout(grid)
            form.addWidget(axes)

            budget = QGroupBox("Budget", panel)
            budget_form = QFormLayout(budget)
            self.max_trials = QSpinBox(panel)
            self.max_trials.setRange(1, 100000)
            self.max_trials.setValue(500)
            self.mode = QComboBox(panel)
            self.mode.addItems(["random", "grid"])
            self.mode.setToolTip(
                "random samples the space evenly; grid walks its product in "
                "order and truncates.")
            self.seed = QSpinBox(panel)
            self.seed.setRange(0, 2 ** 31 - 1)
            self.seed.setValue(20260815)
            self.workers = QSpinBox(panel)
            self.workers.setRange(1, 32)
            suggested, reason = recommended_workers()
            self.workers.setValue(suggested)
            self.workers.setToolTip(
                "A request, not a command: it is clamped to what free memory "
                "allows so a sweep cannot starve the machine.")
            self.worker_note = QLabel(reason, panel)
            self.worker_note.setWordWrap(True)
            budget_form.addRow("Maximum trials", self.max_trials)
            budget_form.addRow("Sampling", self.mode)
            budget_form.addRow("Seed", self.seed)
            budget_form.addRow("Workers", self.workers)
            budget_form.addRow("", self.worker_note)
            form.addWidget(budget)

            buttons = QHBoxLayout()
            self.estimate_button = QPushButton("Estimate", panel)
            self.estimate_button.setToolTip(
                "Count the legal trials and show what the sweep would cost, "
                "without running anything")
            self.estimate_button.clicked.connect(self.estimate)
            self.start_button = QPushButton("Start sweep", panel)
            self.start_button.clicked.connect(self.start)
            buttons.addWidget(self.estimate_button)
            buttons.addWidget(self.start_button)
            buttons.addStretch(1)
            form.addLayout(buttons)
            form.addStretch(1)
            left.setWidget(panel)
            splitter.addWidget(left)

            # ---- right: what came back ------------------------------------
            right = QWidget(self)
            right_layout = QVBoxLayout(right)
            self.status = QLabel("Nothing running.", right)
            self.status.setWordWrap(True)
            right_layout.addWidget(self.status)
            self.progress = QProgressBar(right)
            self.progress.setVisible(False)
            right_layout.addWidget(self.progress)
            self.table = QTableWidget(0, 0, right)
            right_layout.addWidget(self.table, 1)
            self.refresh_button = QPushButton("Refresh results", right)
            self.refresh_button.clicked.connect(self.load_results)
            right_layout.addWidget(self.refresh_button)
            splitter.addWidget(right)
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 2)

        # ------------------------------------------------------------ space

        def space(self):
            """Build the sweep space from the ticked axes.

            An unticked axis is not dropped -- it is PINNED to its first
            value and applied to every trial, so the settings a sweep ran
            under are always fully recorded rather than left to defaults that
            might change.
            """
            import ast

            def parse(text):
                values = []
                for chunk in str(text).split(","):
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    try:
                        values.append(ast.literal_eval(chunk))
                    except (ValueError, SyntaxError):
                        values.append(chunk)
                return values

            axes, fixed = {}, {}
            for key, (include, editor) in self._axis_rows.items():
                values = parse(editor.text())
                if not values:
                    continue
                if include.isChecked() and len(values) > 1:
                    axes[key] = values
                else:
                    fixed[key] = values[0]
            return SweepSpace(axes=axes, fixed=fixed)

        def base_settings(self):
            return {
                "score_data": self.score_data.get_value(),
                "count_data": self.count_data.get_value(),
                "dependent_variable": self.dependent_variable.text().strip(),
                "toxo": False,
                "verbose": False,
            }

        # ---------------------------------------------------------- actions

        def estimate(self):
            space = self.space()
            trials = build_trials(
                space, mode=self.mode.currentText(),
                max_trials=int(self.max_trials.value()),
                seed=int(self.seed.value()))
            workers, reason = recommended_workers(
                requested=int(self.workers.value()))
            self.worker_note.setText(reason)
            # 60s is what one trial took on a real four-plate screen.
            minutes = len(trials) * 60 / max(workers, 1) / 60
            self.status.setText(
                f"{space.size():,} legal combinations; {len(trials)} trials "
                f"would run on {workers} worker(s) — roughly "
                f"{minutes:.0f} minutes. Illegal combinations are rejected "
                f"before they cost a run.")
            return len(trials)

        def start(self):
            base = self.base_settings()
            if not base["score_data"] or not base["count_data"]:
                QMessageBox.warning(self, "Nothing to sweep",
                                    "Choose at least one score CSV and one "
                                    "count CSV.")
                return
            destination = self.destination.text().strip()
            if not destination:
                QMessageBox.warning(self, "No output folder",
                                    "Choose a folder for the trial folders "
                                    "and the results table.")
                return
            space = self.space()
            mode = self.mode.currentText()
            max_trials = int(self.max_trials.value())
            seed = int(self.seed.value())
            workers = int(self.workers.value())
            self.start_button.setEnabled(False)
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)
            self.status.setText("Sweeping…")

            def job():
                from ...parameter_sweep import run_sweep_parallel
                return run_sweep_parallel(
                    base, destination, space, mode=mode,
                    max_trials=max_trials, seed=seed, n_jobs=workers,
                    controls={"positive": str(base.get("positive_control",
                                                       "239740"))})

            # Bound method, so the handler runs on the GUI thread.
            self._runner.submit(job, self._sweep_finished)

        def _sweep_finished(self, results):
            self.progress.setVisible(False)
            self.start_button.setEnabled(True)
            self._results = results
            if results is None or not len(results):
                self.status.setText("The sweep produced no trials.")
                return
            ok = int((results["status"] == "ok").sum())
            self.status.setText(
                f"{len(results)} trials, {ok} succeeded. "
                f"Results in {self.destination.text().strip()}")
            self._show(results)

        def load_results(self):
            """Read the table from disk, so a running sweep can be watched."""
            import pandas as pd
            path = os.path.join(self.destination.text().strip(),
                                "sweep_results.csv")
            if not os.path.exists(path):
                self.status.setText(f"No results table at {path} yet.")
                return
            frame = pd.read_csv(path)
            self._results = frame
            self.status.setText(f"{len(frame)} trials recorded so far.")
            self._show(frame)

        def _show(self, frame):
            # The columns worth reading first, when they exist. The rest are
            # still in the CSV; this is a view, not a filter.
            preferred = [c for c in (
                "trial_id", "status", "regression_type", "inference",
                "analysis_unit", "agg_type", "transform",
                "multiple_testing_method", "n_below_alpha", "positive_rank",
                "seconds", "error_type") if c in frame.columns]
            columns = preferred or list(frame.columns)[:12]
            self.table.setColumnCount(len(columns))
            self.table.setHorizontalHeaderLabels(columns)
            self.table.setRowCount(min(len(frame), 2000))
            for row in range(self.table.rowCount()):
                for column, name in enumerate(columns):
                    value = frame.iloc[row][name]
                    self.table.setItem(row, column,
                                       QTableWidgetItem(str(value)))

    return ParameterSweepScreen(host=host)


def register() -> bool:
    """Add the module through spaCR's single application-registration seam."""
    from ..app import APPS, SECTION_RESULTS, STAGE_ALPHA, register_app
    if any(row[0] == APP_KEY for row in APPS):
        return False
    register_app(
        APP_KEY, APP_NAME, APP_DESCRIPTION, SECTION_RESULTS,
        factory=_make_screen, stage=STAGE_ALPHA, title=APP_NAME,
        intro=APP_INTRO, translations=APP_TRANSLATIONS)
    return True


register()
