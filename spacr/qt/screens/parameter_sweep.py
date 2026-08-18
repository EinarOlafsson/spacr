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
#: Why there is no ``spacr-run parameter_sweep``. Reaches
#: :data:`spacr.cli.INTERACTIVE_ONLY`, which is what the CLI prints instead of
#: "unknown module".
#:
#: The SWEEP is fully headless -- :mod:`spacr.parameter_sweep` is the engine
#: this screen drives, and a cluster is where a few hundred trials belong. It
#: is the WORKBENCH that cannot be batched: the table, and double-clicking a
#: row to get that exact regression back with its figures. So the note points
#: at the engine rather than claiming the feature has no headless path.
APP_CLI_NOTE = (
    "Parameter Sweep is the interactive workbench for a sweep -- the table of "
    "trials, and double-clicking a row to get that exact regression back with "
    "its figures. The sweep itself is headless: from spacr.parameter_sweep "
    "import run_sweep, summarise_sweep; run_sweep(base_settings, destination) "
    "writes the same results table this screen reads, and summarise_sweep / "
    "rank_trials give the comparison it draws.")

__all__ = ["APP_KEY", "APP_NAME", "APP_DESCRIPTION", "APP_INTRO",
           "APP_CLI_NOTE", "register"]


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
            # CLICK A ROW TO GET THAT REGRESSION BACK.
            #
            # A sweep row carries every setting its trial was given, so it is
            # enough to reproduce the trial exactly. Running it here rather
            # than opening the saved page matters: these come back as live
            # matplotlib Figures, so they land in the figure queue below and
            # can be restyled -- thresholds, colours, legend, axis limits --
            # which is the whole reason for looking at a condition again.
            self.table.setSelectionBehavior(QTableWidget.SelectRows)
            self.table.setEditTriggers(QTableWidget.NoEditTriggers)
            self.table.setSortingEnabled(True)
            self.table.doubleClicked.connect(self._on_row_activated)
            self.table.setToolTip(
                "Double-click a row to re-run that trial and draw its "
                "figures below. They are live figures: right-click one to "
                "restyle it.")
            right_layout.addWidget(self.table, 1)

            row_buttons = QHBoxLayout()
            self.refresh_button = QPushButton("Refresh results", right)
            self.refresh_button.clicked.connect(self.load_results)
            row_buttons.addWidget(self.refresh_button)
            self.show_button = QPushButton("Show figures for selected row",
                                           right)
            self.show_button.clicked.connect(self._on_row_activated)
            self.show_button.setEnabled(False)
            row_buttons.addWidget(self.show_button, 1)
            right_layout.addLayout(row_buttons)
            self.table.itemSelectionChanged.connect(
                lambda: self.show_button.setEnabled(
                    self.table.currentRow() >= 0))

            self.trial_status = QLabel("", right)
            self.trial_status.setWordWrap(True)
            right_layout.addWidget(self.trial_status)

            # THE WHOLE SET OF GRAPHS FOR THE CLICKED ROW.
            #
            # Re-running was the expensive half and it already worked, but
            # showing one figure at a time in a queue means the user still
            # cannot put a run's residual plot beside its volcano -- which is
            # the comparison that decides whether a configuration is any good.
            from ..widgets.regression_results import RegressionResultsPanel
            self.results = RegressionResultsPanel(right)
            self.results.setMinimumHeight(320)
            right_layout.addWidget(self.results, 1)

            from ..widgets.figure_queue import FigureQueue
            self.figures = FigureQueue(parent=right)
            self.figures.setMinimumHeight(200)
            self.figures.hide()      # shown only when a re-run makes figures
            right_layout.addWidget(self.figures, 1)
            splitter.addWidget(right)
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 2)

            # Every screen that reads a path takes a drop, and a sweep reads
            # more paths than anything else in spaCR -- a plate per pair. The
            # policy is SweepInputsDropHandler, which sorts each CSV into the
            # score or the count list from its header. install_for never
            # raises: a Qt build without drag-and-drop loses the convenience,
            # not the screen.
            from ..dnd import install_for
            install_for(self, APP_KEY, self)

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

        def apply_settings(self, settings):
            """Seed the sweep from the module's settings panel.

            Opening the sweep should not mean retyping the inputs that are
            already on screen. The score/count CSVs, the response column and
            the output folder come straight across; every swept axis is
            additionally PINNED to the value the user currently has, so an
            unticked axis reproduces their run rather than a default.
            """
            if not isinstance(settings, dict):
                return
            for key, widget in (("score_data", self.score_data),
                                ("count_data", self.count_data)):
                value = settings.get(key)
                if value and hasattr(widget, "set_value"):
                    try:
                        widget.set_value(list(value))
                    except Exception:
                        pass
            response = settings.get("dependent_variable")
            if response:
                self.dependent_variable.setText(str(response))
            source = settings.get("src")
            if source and not self.destination.text().strip():
                # Beside the data rather than inside it: a sweep writes
                # thousands of folders and they should not be mixed in with
                # the user's inputs.
                self.destination.setText(os.path.join(str(source), "sweep"))
            for key, (include, editor) in self._axis_rows.items():
                if key in settings and settings[key] is not None:
                    current = settings[key]
                    text = "None" if current is None else str(current)
                    if not include.isChecked():
                        editor.setText(text)
                    elif text not in editor.text():
                        # Keep the user's value in the swept range, so their
                        # own condition is one of the trials that gets run.
                        editor.setText(f"{text}, {editor.text()}")

        def base_settings(self):
            return {
                "score_data": self.score_data.get_value(),
                "count_data": self.count_data.get_value(),
                "dependent_variable": self.dependent_variable.text().strip(),
                "Toxoplasma": False,
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

        def _on_row_activated(self, *_args):
            """Re-run the selected trial and show its figures, editable.

            Off the GUI thread: this is a full regression, not a lookup. The
            row keeps its own settings, so what comes back is that trial and
            not a fresh one built from whatever the controls happen to say
            now -- which is the point of being able to compare conditions.
            """
            if self._results is None or not len(self._results):
                return
            row_index = self.table.currentRow()
            if row_index < 0:
                return
            # The table may be sorted, so trust the trial_id in the row
            # rather than the table's row number.
            key_item = self.table.item(row_index, 0)
            frame = self._results
            record = None
            if key_item is not None and "trial_id" in frame.columns:
                try:
                    match = frame[frame["trial_id"].astype(str)
                                  == key_item.text()]
                    if len(match):
                        record = match.iloc[0].to_dict()
                except Exception:
                    record = None
            if record is None and row_index < len(frame):
                record = frame.iloc[row_index].to_dict()
            if record is None:
                return
            if str(record.get("status", "ok")) != "ok":
                QMessageBox.information(
                    self, "That trial failed",
                    "This trial did not produce a regression:\n\n"
                    f"{record.get('error_type', '')}: "
                    f"{record.get('error', 'no reason recorded')}")
                return

            # A SAVED RUN IS INSTANT; A RE-FIT IS A MINUTE.
            #
            # The trial wrote its results when the sweep ran, so prefer them.
            # Re-fitting to see something already on disk is a minute of
            # waiting for an identical answer.
            folder = record.get("folder")
            if folder and self.results.load(folder):
                self.trial_status.setText(
                    f"Trial {record.get('trial_id', '?')} loaded from disk "
                    f"({folder}). Nothing was re-fitted.")
                return

            base = self.base_settings()
            self.show_button.setEnabled(False)
            self.trial_status.setText(
                f"Trial {record.get('trial_id', '?')} has no saved results; "
                f"re-fitting it to draw them…")

            def job():
                from ...parameter_sweep import rerun_trial
                return rerun_trial(base, record)

            self._runner.submit(job, self._trial_figures_ready)

        def _trial_figures_ready(self, payload):
            """Put a re-run trial's figures on screen. On the GUI thread."""
            self.show_button.setEnabled(self.table.currentRow() >= 0)
            if not isinstance(payload, dict):
                self.trial_status.setText(
                    "That trial did not come back. See the console.")
                return
            figures = payload.get("figures") or []
            for figure in figures:
                try:
                    self.figures.add_figure(figure)
                except Exception:
                    pass
            if figures:
                self.figures.show()
            # The re-fit wrote its results too, so the full panel can show
            # them exactly as it would for a saved trial.
            output = payload.get("output") or {}
            settings = payload.get("settings") or {}
            try:
                results = output.get("results")
                if results is not None and len(results):
                    self.results.set_frame(
                        results, source=str(settings.get("src", "")))
            except Exception:
                pass
            settings = payload.get("settings") or {}
            described = ", ".join(
                f"{key}={settings.get(key)!r}" for key in (
                    "regression_type", "inference", "analysis_unit",
                    "multiple_testing_method", "min_cell_count",
                    "fraction_threshold") if key in settings)
            self.trial_status.setText(
                f"{len(figures)} figure(s) from {described or 'that trial'}. "
                f"Right-click a figure to restyle it."
                if figures else
                "That trial produced no figures.")

        def _show(self, frame):
            # The columns worth reading first, when they exist. The rest are
            # still in the CSV; this is a view, not a filter.
            # Settings first, then WHAT WENT IN, then what came out. A hit
            # count means little without the size of the design it came from:
            # two trials differing only by a filtration cutoff can fit
            # completely different data.
            preferred = [c for c in (
                "trial_id", "status", "regression_type", "inference",
                "analysis_unit", "agg_type", "transform",
                "multiple_testing_method", "fdr_alpha",
                "fraction_threshold", "min_cell_count",
                "n_wells", "n_guides", "n_cells", "n_rows_fitted",
                "n_results", "n_below_alpha", "positive_rank",
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
        intro=APP_INTRO, cli_note=APP_CLI_NOTE,
        translations=APP_TRANSLATIONS)
    return True


register()


#: Text on the toggle that reveals the card, and its hover help. Mirrors
#: spacr.qt.screens.hyperparam so the two searches read as the same feature in
#: two modules rather than as two different ones.
SWEEP_TOGGLE_TEXT = "Parameter sweep"
SWEEP_TOGGLE_TOOLTIP = (
    "Toggle the parameter sweep. When ON (blue), set a range for any "
    "regression setting -- model family, correction method, filtration "
    "cutoffs, wells vs cells -- run every legal combination, and double-click "
    "a row in the results to get that exact regression back with its figures."
)


def build_parameter_sweep_card(host):
    """Build the ``Parameter sweep`` card + panel pair.

    Mirrors :func:`spacr.qt.screens.hyperparam.build_hyperparam_card`: returns
    the pair without adding it to a layout, so the host puts it where it likes
    and starts it hidden behind the toggle.

    :param host: the ``AppScreen`` asking for the card.
    :returns: ``(panel, card)``.
    """
    from ..widgets.card import Card
    card = Card(title="Parameter sweep")
    panel = _make_screen(host=host)
    card.body_layout.addWidget(panel)
    card.setMinimumHeight(320)
    return panel, card


def sweepable(app_key: str) -> bool:
    """Whether a parameter sweep exists for ``app_key``.

    Only the regression module for now: the sweep axes, the legality filters
    and the row-to-regression round trip are all specific to it.
    """
    return app_key in ("regression", "ml_analyze_regression")
