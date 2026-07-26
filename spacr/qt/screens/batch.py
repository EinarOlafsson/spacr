"""Batch Runner — the Tools module that spends the night for you.

The Plate Queue chains *plates* through *one* pipeline. This screen is the
other axis: arbitrary ``(module, settings)`` jobs in any order —
``Mask → Measure → Classify (CV) → Classify (ML)``, then the same four again
with a different diameter, then a fifth plate's Mask — built, validated, saved
to a file and run unattended.

The screen is deliberately thin. Everything it knows about a queue it learns
from :mod:`spacr.batch`, which is headless, torch-free and tested without Qt.
This file is only the part that has to be a GUI: pick a module and a settings
file, duplicate a job and edit it, reorder, validate, save/load, run, and watch
per-job status, progress and log.

Three decisions are visible to the user:

* **Jobs are validated when they are added, not when they run.** A job whose
  settings file is missing, whose ``src`` is misspelled or whose module is
  GUI-only is refused at the Add button, with the reason inline. Finding that
  out at 3 a.m. from job 9 of 12 is what this whole module exists to prevent.
* **No modal dialogs, ever.** Every failure lands in the inline problems pane
  and in :attr:`BatchScreen.last_error`. A ``QMessageBox`` hangs a headless
  run (it did, in ``MakeMasksScreen``), and this screen is exercised headlessly.
* **The run happens off the GUI thread, and its completion handler comes back
  onto it.** ``PipelineWorker.finished`` is emitted *in the worker thread* and
  PySide6 invokes a plain closure connected to it directly, on that thread —
  so it is chained through :attr:`BatchScreen._queue_settled` into a bound
  method of this widget, which has GUI-thread affinity. Same idiom as
  :class:`spacr.qt.screens.plate_view.PlateViewScreen`.

Jobs run one at a time — they compete for one GPU — and each one is its own
``spacr-run`` process, so a segfault in cellpose kills that job rather than
this window.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ... import batch as bt
from ...cli import MODULES
from ..bridge import make_thread
from ..theme import PALETTE, SPACING
from ..widgets import Divider

__all__ = ["BatchScreen", "COLUMNS", "ON_ERROR_LABELS", "STATUS_COLOURS"]


#: Table columns, in order.
COLUMNS: Tuple[str, ...] = (
    "#", "Job", "Module", "Label", "After", "Status", "Time", "Log",
)

#: What the on-error combo offers, mapping label to ``run_queue(on_error=...)``.
ON_ERROR_LABELS: Tuple[Tuple[str, str], ...] = (
    ("Continue with the jobs that don't depend on it", "continue"),
    ("Stop the queue at the first failure", "stop"),
)

#: Per-status colours. ``skipped`` is deliberately not the failure colour: it
#: is not a failure, it is a job that was correctly never run.
STATUS_COLOURS: Dict[str, str] = {
    bt.STATUS_PENDING: "fg_muted",
    bt.STATUS_RUNNING: "info",
    bt.STATUS_SUCCESS: "success",
    bt.STATUS_FAILED: "error",
    bt.STATUS_SKIPPED: "warning",
    bt.STATUS_NOT_RUN: "fg_dim",
}

_LOG_TAIL_BYTES = 200_000


class BatchScreen(QWidget):
    """Build a queue of module+settings jobs, validate it, and run it overnight.

    :param parent: parent widget.
    :param threaded: run the queue on a worker thread (the default). Tests
        pass ``False`` for deterministic, synchronous behaviour.
    :param runner: ``(job, settings_path, log_path) -> exit_code``, forwarded
        to :func:`spacr.batch.run_queue`. Defaults to
        :func:`spacr.batch.subprocess_runner` — one child process per job.
        Tests inject a fake so nothing real is ever segmented.
    :ivar last_error: text of the most recent failure, ``""`` when the last
        operation succeeded. Errors are *only* ever reported here and in the
        inline panes — never in a modal dialog.
    :ivar settled_thread: the thread the completion handler last ran on.
        Exists so a test can prove it was the GUI thread.
    """

    #: emitted with the job count whenever the queue is edited
    queue_changed = Signal(int)
    #: emitted with (job_id, status) on every per-job transition
    job_status_changed = Signal(str, str)
    #: emitted when the whole queue finishes; True when nothing failed,
    #: nothing was skipped and nothing came back partial
    queue_finished = Signal(bool)

    #: private. Re-emitted from the worker thread purely to hop onto the GUI
    #: thread — see :meth:`_on_progress`.
    _progress_relayed = Signal(object)
    #: private. Re-emitted from ``PipelineWorker.finished`` for the same
    #: reason — see :meth:`_on_queue_settled`.
    _queue_settled = Signal(bool)

    def __init__(self, parent=None, threaded: bool = True,
                 runner: Optional[Callable[[Any, str, str], int]] = None):
        super().__init__(parent)
        self._threaded = bool(threaded)
        self._runner = runner
        self._queue = bt.Queue(name="queue")
        self._path: str = ""
        self._busy = False
        self._stop_requested = False
        self._result: Optional[bt.QueueResult] = None
        self._problems: List[bt.Problem] = []
        self._jobs: List[tuple] = []
        self._pending: List[Dict[str, Any]] = []
        self._thread = None
        self._worker = None
        self.last_error: str = ""
        self.settled_thread: Optional[QThread] = None

        self._progress_relayed.connect(self._on_progress)
        self._queue_settled.connect(self._on_queue_settled)

        self._build_ui()
        self._refresh_table()
        self._update_controls()
        self._set_status("Add a job: pick a module and the settings file you "
                         "would run it with.")

        # Elapsed time for the running job. Cheap, and only ever touches one
        # cell, so it does not churn the table or the selection.
        self._tick = QTimer(self)
        self._tick.setInterval(1000)
        self._tick.timeout.connect(self._refresh_running_row)
        self._tick.start()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                 SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["md"])

        title = QLabel("Batch Runner")
        title.setObjectName("DisplayHeading")
        outer.addWidget(title)

        subtitle = QLabel(
            "Stack any modules, in any order, with any settings, and run them "
            "overnight: Mask → Measure → Classify, then the same again with "
            "different settings. Every job is checked when you add it, the "
            "queue file survives a reboot, and a job whose input failed is "
            "skipped instead of run on a half-written result. One job at a "
            "time — they share one GPU.")
        subtitle.setObjectName("Muted")
        subtitle.setWordWrap(True)
        outer.addWidget(subtitle)

        outer.addWidget(Divider())

        # ── Job editor ────────────────────────────────────────────────
        edit = QHBoxLayout()
        edit.setSpacing(SPACING["sm"])
        edit.addWidget(QLabel("Module", self))
        self._module_combo = QComboBox(self)
        for key in sorted(MODULES):
            self._module_combo.addItem(key, key)
            self._module_combo.setItemData(self._module_combo.count() - 1,
                                           MODULES[key].summary, Qt.ToolTipRole)
        self._module_combo.setMinimumWidth(150)
        edit.addWidget(self._module_combo)

        self._settings_edit = QLineEdit(self)
        self._settings_edit.setPlaceholderText(
            "…/settings/gen_mask_settings.csv — the settings file this job runs with")
        self._settings_edit.setClearButtonEnabled(True)
        self._settings_edit.returnPressed.connect(self._on_add_clicked)
        edit.addWidget(self._settings_edit, 1)

        self._btn_pick = QPushButton("Choose settings…", self)
        self._btn_pick.clicked.connect(self._pick_settings_file)
        edit.addWidget(self._btn_pick)
        outer.addLayout(edit)

        edit2 = QHBoxLayout()
        edit2.setSpacing(SPACING["sm"])
        self._label_edit = QLineEdit(self)
        self._label_edit.setPlaceholderText("Label (optional) — e.g. 'plate 3 mask, 30 px'")
        edit2.addWidget(self._label_edit, 1)
        self._depends_edit = QLineEdit(self)
        self._depends_edit.setPlaceholderText(
            "After (optional) — job ids, comma separated; this job is skipped if they fail")
        edit2.addWidget(self._depends_edit, 1)
        self._overrides_edit = QLineEdit(self)
        self._overrides_edit.setPlaceholderText(
            "Overrides (optional) — key=value, comma separated")
        edit2.addWidget(self._overrides_edit, 1)
        self._btn_add = QPushButton("Add job", self)
        self._btn_add.clicked.connect(self._on_add_clicked)
        edit2.addWidget(self._btn_add)
        outer.addLayout(edit2)

        # ── Queue toolbar ─────────────────────────────────────────────
        bar = QHBoxLayout()
        bar.setSpacing(SPACING["sm"])
        self._btn_dup = QPushButton("Duplicate", self)
        self._btn_dup.setToolTip("Copy the selected job so you can edit the copy — "
                                 "how a twelve-job night is usually built.")
        self._btn_dup.clicked.connect(self.duplicate_selected)
        self._btn_remove = QPushButton("Remove", self)
        self._btn_remove.clicked.connect(self.remove_selected)
        self._btn_up = QPushButton("Move up", self)
        self._btn_up.clicked.connect(lambda: self.move_selected(-1))
        self._btn_down = QPushButton("Move down", self)
        self._btn_down.clicked.connect(lambda: self.move_selected(1))
        self._btn_validate = QPushButton("Validate", self)
        self._btn_validate.clicked.connect(self.validate_now)
        self._btn_load = QPushButton("Load queue…", self)
        self._btn_load.clicked.connect(self._pick_queue_to_load)
        self._btn_save = QPushButton("Save queue…", self)
        self._btn_save.clicked.connect(self._pick_queue_to_save)
        for button in (self._btn_dup, self._btn_remove, self._btn_up,
                       self._btn_down, self._btn_validate, self._btn_load,
                       self._btn_save):
            bar.addWidget(button)
        bar.addStretch(1)
        outer.addLayout(bar)

        # ── Run controls ──────────────────────────────────────────────
        run_row = QHBoxLayout()
        run_row.setSpacing(SPACING["sm"])
        run_row.addWidget(QLabel("On failure", self))
        self._on_error_combo = QComboBox(self)
        for label, value in ON_ERROR_LABELS:
            self._on_error_combo.addItem(label, value)
        run_row.addWidget(self._on_error_combo)
        run_row.addWidget(QLabel("Stop after", self))
        self._threshold_spin = QSpinBox(self)
        self._threshold_spin.setRange(0, 99)
        self._threshold_spin.setValue(3)
        self._threshold_spin.setSuffix(" failures in a row")
        self._threshold_spin.setToolTip(
            "Three jobs failing the same way is a systematic problem, not three "
            "accidents — the rest of the queue would fail too. 0 disables the check.")
        run_row.addWidget(self._threshold_spin)
        run_row.addStretch(1)
        self._btn_run = QPushButton("Run queue", self)
        self._btn_run.clicked.connect(self.run)
        self._btn_stop = QPushButton("Stop", self)
        self._btn_stop.clicked.connect(self.stop)
        run_row.addWidget(self._btn_run)
        run_row.addWidget(self._btn_stop)
        outer.addLayout(run_row)

        self._progress = QProgressBar(self)
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setFormat("%v / %m jobs")
        outer.addWidget(self._progress)

        # ── Table + panes ─────────────────────────────────────────────
        split = QSplitter(Qt.Vertical, self)

        self._table = QTableWidget(self)
        self._table.setColumnCount(len(COLUMNS))
        self._table.setHorizontalHeaderLabels(list(COLUMNS))
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(
            COLUMNS.index("Label"), QHeaderView.Stretch)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        split.addWidget(self._table)

        panes = QWidget(self)
        pane_layout = QVBoxLayout(panes)
        pane_layout.setContentsMargins(0, 0, 0, 0)
        pane_layout.setSpacing(SPACING["xs"])
        self._problems_view = QPlainTextEdit(panes)
        self._problems_view.setReadOnly(True)
        self._problems_view.setPlaceholderText(
            "Validation problems appear here — all of them at once, so the queue "
            "can be fixed in one pass.")
        self._problems_view.setMaximumHeight(140)
        pane_layout.addWidget(self._problems_view)
        self._log_view = QPlainTextEdit(panes)
        self._log_view.setReadOnly(True)
        self._log_view.setPlaceholderText(
            "Select a job to read its own log. Every job writes its own file — a "
            "single interleaved log from an overnight run is unreadable.")
        pane_layout.addWidget(self._log_view, 1)
        split.addWidget(panes)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        outer.addWidget(split, 1)

        self._status = QLabel("", self)
        self._status.setObjectName("Muted")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

    # ------------------------------------------------------------------
    # queue editing
    # ------------------------------------------------------------------

    def queue(self) -> bt.Queue:
        """The live :class:`spacr.batch.Queue` this screen is editing."""
        return self._queue

    def result(self) -> Optional[bt.QueueResult]:
        """The :class:`spacr.batch.QueueResult` of the last run, or None."""
        return self._result

    def add_job(self, module: str = "", settings: str = "", label: str = "",
                depends_on: Sequence[str] = (), overrides: Sequence[str] = ()) -> bool:
        """Add one job, validating it now. Problems land inline, never modally.

        :param module: module key; the combo's current value when empty.
        :param settings: settings file path; the settings box when empty.
        :param label: display label; derived from the module and src when empty.
        :param depends_on: ids of jobs that must succeed first.
        :param overrides: ``key=value`` strings, exactly like ``--set``.
        :returns: True when the job was added.
        """
        job = bt.Job(
            module=module or self._module_combo.currentData() or "",
            settings=settings,
            label=label,
            depends_on=[d for d in depends_on if d],
            overrides=[o for o in overrides if o],
        )
        try:
            self._queue.add(job)
        except bt.QueueError as exc:
            self._show_problems_text(str(exc))
            self._set_status(f"{job.module}: not added — see the problems above.",
                             error=True)
            return False
        self._after_edit(f"Added {job.id} ({job.label}).")
        return True

    def duplicate_selected(self) -> bool:
        """Copy the selected job, never-run, and select the copy.

        Building a night's work means one job and eleven variations of it, so
        this is the button that actually gets used.
        """
        job = self.selected_job()
        if job is None:
            self._set_status("Select a job to duplicate.", error=True)
            return False
        clone = job.copy()
        clone.label = f"{job.label} (copy)"
        try:
            self._queue.add(clone)
        except bt.QueueError as exc:
            self._show_problems_text(str(exc))
            self._set_status("Could not duplicate — see the problems above.",
                             error=True)
            return False
        self._after_edit(f"Duplicated {job.id} as {clone.id}. Edit the copy's "
                         f"settings or overrides, then run.")
        self.select_job(clone.id)
        return True

    def remove_selected(self) -> bool:
        """Remove the selected job, and any dependency other jobs had on it."""
        job = self.selected_job()
        if job is None:
            self._set_status("Select a job to remove.", error=True)
            return False
        if self._busy and job.status == bt.STATUS_RUNNING:
            self._set_status("That job is running — stop the queue first.",
                             error=True)
            return False
        self._queue.remove(job.id)
        self._after_edit(f"Removed {job.id}.")
        return True

    def move_selected(self, offset: int) -> bool:
        """Move the selected job ``offset`` places (negative is earlier)."""
        job = self.selected_job()
        if job is None:
            self._set_status("Select a job to reorder.", error=True)
            return False
        if self._busy:
            self._set_status("The queue is running — stop it before reordering.",
                             error=True)
            return False
        self._queue.move(job.id, offset)
        self._after_edit(f"Moved {job.id}.")
        self.select_job(job.id)
        return True

    def _after_edit(self, message: str) -> None:
        """Refresh everything an edit touches, then re-validate quietly."""
        self._refresh_table()
        self._update_controls()
        self.queue_changed.emit(len(self._queue))
        self._set_status(message)
        self.validate_now(quiet=True)

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    def validate_now(self, quiet: bool = False) -> List[bt.Problem]:
        """Validate the whole queue and show every problem inline at once.

        :param quiet: do not overwrite the status line when the queue is clean.
        :returns: the problems, errors and warnings mixed.
        """
        problems = bt.validate_queue(self._queue)
        self._problems = problems
        errors = [p for p in problems if p.is_error]
        if problems:
            self._problems_view.setPlainText(bt.format_problems(problems))
        else:
            self._problems_view.setPlainText("")
        if errors:
            self._set_status(
                f"{len(errors)} job(s) cannot run — the queue will not start "
                f"until they are fixed.", error=True)
        elif not quiet:
            warnings = len(problems)
            self._set_status(
                f"{len(self._queue)} job(s) ready"
                + (f", {warnings} warning(s)." if warnings else ", no problems found."))
        self._update_controls()
        return problems

    def problems_text(self) -> str:
        """The inline problems pane (test/introspection helper)."""
        return self._problems_view.toPlainText()

    def has_errors(self) -> bool:
        """True when the last validation found something that blocks the run."""
        return any(p.is_error for p in self._problems)

    # ------------------------------------------------------------------
    # the queue file
    # ------------------------------------------------------------------

    def save_queue_to(self, path: str) -> bool:
        """Write the queue to ``path`` atomically. Errors land inline."""
        try:
            bt.save_queue(self._queue, path)
        except (OSError, bt.QueueError) as exc:
            self._set_status(f"Could not save the queue: {exc}", error=True)
            return False
        self._path = str(path)
        self._set_status(f"Saved {len(self._queue)} job(s) to {path}. The run "
                         f"keeps this file up to date, so it can be resumed.")
        return True

    def load_queue_from(self, path: str) -> bool:
        """Replace the queue with the one in ``path``. Errors land inline."""
        try:
            queue = bt.load_queue(path)
        except bt.QueueError as exc:
            self._show_problems_text(str(exc))
            self._set_status(f"Could not load {path} — see above.", error=True)
            return False
        self._queue = queue
        self._path = str(path)
        self._result = None
        self._refresh_table()
        self.queue_changed.emit(len(self._queue))
        self._set_status(f"Loaded {len(self._queue)} job(s) from {path}.")
        self.validate_now(quiet=True)
        return True

    def queue_path(self) -> str:
        """The queue file this screen saves to and resumes from, or ``''``."""
        return self._path

    # ------------------------------------------------------------------
    # running
    # ------------------------------------------------------------------

    def set_runner(self, runner: Optional[Callable[[Any, str, str], int]]) -> None:
        """Replace the per-job runner. None restores the subprocess default."""
        self._runner = runner

    def is_busy(self) -> bool:
        """True while the queue is running."""
        return self._busy

    def active_jobs(self) -> int:
        """How many worker threads are still winding down."""
        return len(self._jobs)

    def run(self) -> bool:
        """Validate, then run the queue off the GUI thread.

        :returns: True when the run started (or, unthreaded, completed).
        """
        if self._busy:
            return False
        if not self._queue.jobs:
            self._set_status("Nothing to run — add a job first.", error=True)
            return False
        problems = self.validate_now(quiet=True)
        if any(p.is_error for p in problems):
            self._set_status(
                "The queue will not start: some jobs cannot run. Every problem "
                "is listed above — fixing them now is the whole point of "
                "checking before the night, not during it.", error=True)
            return False

        self._stop_requested = False
        self._result = None
        self._busy = True
        self._progress.setRange(0, len(self._queue))
        self._progress.setValue(0)
        self._update_controls()
        self._set_status(f"Running {len(self._queue)} job(s), one at a time…")

        box: Dict[str, Any] = {}

        def _job(payload: Dict[str, Any]) -> None:
            payload["result"] = bt.run_queue(
                self._queue,
                path=self._path or None,
                on_error=self._on_error_combo.currentData() or "continue",
                max_consecutive_failures=self._threshold_spin.value(),
                on_progress=self._relay_progress,
                runner=self._runner,
                stop_flag=lambda: self._stop_requested,
                echo=False,
            )

        if not self._threaded:
            ok = True
            try:
                _job(box)
            except Exception as exc:  # noqa: BLE001 - reported inline, never raised
                self._on_run_error(exc)
                ok = False
            self._pending.append(box)
            self._on_queue_settled(ok)
            return ok

        thread, worker = make_thread(_job, box)
        # Strong references: PySide6 will not keep the worker alive through the
        # started→run connection alone, and a QThread garbage-collected while
        # still running takes the whole process down with it.
        self._jobs.append((thread, worker))
        self._thread, self._worker = thread, worker
        self._pending.append(box)
        worker.error.connect(self._on_worker_error_text)
        worker.finished.connect(self._queue_settled)
        thread.finished.connect(lambda t=thread: self._retire_job(t))
        thread.start()
        return True

    def stop(self) -> bool:
        """Ask the queue to stop after the job that is currently running.

        Killing a job mid-write would leave exactly the half-written artifact
        the skip rules exist to avoid, so the running job is always allowed to
        finish.
        """
        if not self._busy:
            return False
        self._stop_requested = True
        self._set_status("Stopping after the current job finishes — killing it "
                         "would leave a half-written result.")
        return True

    # -- progress, on the worker thread ---------------------------------

    def _relay_progress(self, progress: "bt.Progress") -> None:
        """Called ON THE WORKER THREAD by :func:`spacr.batch.run_queue`.

        Touches no widget: it only emits a signal, which Qt queues onto the
        GUI thread where :meth:`_on_progress` does the widget work.
        """
        self._progress_relayed.emit(progress)

    def _on_progress(self, progress: "bt.Progress") -> None:
        """Handle one progress report. Always on the GUI thread."""
        self._refresh_table()
        if progress.total:
            self._progress.setRange(0, progress.total)
        done = sum(1 for job in self._queue.jobs
                   if job.status in (bt.STATUS_SUCCESS, bt.STATUS_FAILED,
                                     bt.STATUS_SKIPPED))
        self._progress.setValue(done)
        if progress.job_id:
            job = self._queue.find(progress.job_id)
            if job is not None:
                self.job_status_changed.emit(job.id, job.status)
        if progress.message:
            self._set_status(progress.message,
                             error=progress.status in (bt.STATUS_FAILED,
                                                       bt.STATUS_SKIPPED))
        if progress.event == "job_started" and progress.job_id:
            self.select_job(progress.job_id)
        # Keep the log pane live: the selected job's file is being written
        # right now, and a pane that only updates on selection would show the
        # previous job's output for the next seven hours.
        selected = self.selected_job()
        if selected is not None:
            self._load_log(selected)

    # -- completion, back on the GUI thread ------------------------------

    def _on_queue_settled(self, ok: bool) -> None:
        """Finish the run. Always on the GUI thread — see the module docstring."""
        self.settled_thread = QThread.currentThread()
        self._busy = False
        box = self._pending.pop(0) if self._pending else {}
        result = box.get("result")
        if isinstance(result, bt.QueueResult):
            self._result = result
            self._problems_view.setPlainText(result.summary())
            counts = self._queue.counts()
            head = (f"{counts[bt.STATUS_SUCCESS]} ok, {counts[bt.STATUS_FAILED]} failed, "
                    f"{counts[bt.STATUS_SKIPPED]} skipped")
            if result.partial:
                head += f", {len(result.partial)} PARTIAL"
            self._set_status(
                f"Queue finished — {head}. {result.stopped_reason}".strip(),
                error=bool(result.failed or result.partial or result.stopped_reason))
            ok = bool(ok) and result.ok
        self._refresh_table()
        self._progress.setValue(self._progress.maximum())
        self._update_controls()
        self.queue_finished.emit(bool(ok))

    def _retire_job(self, thread) -> None:
        """Release this job's refs once its own event loop has exited."""
        self._jobs = [(t, w) for (t, w) in self._jobs if t is not thread]
        if self._thread is thread:
            self._thread = None
            self._worker = None

    def _on_run_error(self, exc: BaseException) -> None:
        self._busy = False
        self._set_status(f"The queue runner failed: {exc}", error=True)

    def _on_worker_error_text(self, text: str) -> None:
        line = (text or "").strip().splitlines()[-1] if text else "unknown error"
        self._busy = False
        self._set_status(f"The queue runner failed: {line}", error=True)

    # ------------------------------------------------------------------
    # table
    # ------------------------------------------------------------------

    def selected_job(self) -> Optional[bt.Job]:
        """The :class:`spacr.batch.Job` for the selected row, or None."""
        row = self._table.currentRow()
        if row < 0 or row >= len(self._queue.jobs):
            return None
        return self._queue.jobs[row]

    def select_job(self, job_id: str) -> bool:
        """Select the row for ``job_id``."""
        index = self._queue.index(job_id)
        if index < 0:
            return False
        self._table.selectRow(index)
        return True

    def row_values(self, row: int) -> List[str]:
        """Text of every cell in ``row`` (test/introspection helper)."""
        return [(self._table.item(row, col).text()
                 if self._table.item(row, col) is not None else "")
                for col in range(len(COLUMNS))]

    def row_status(self, row: int) -> str:
        """Status text shown in ``row`` (test/introspection helper)."""
        item = self._table.item(row, COLUMNS.index("Status"))
        return item.text() if item is not None else ""

    def _status_text(self, job: bt.Job) -> str:
        """What the Status cell says — partial is not success."""
        if job.status == bt.STATUS_SUCCESS and job.is_partial:
            return "success (partial)"
        if job.status == bt.STATUS_NOT_RUN:
            return "not run"
        return job.status

    def _refresh_table(self) -> None:
        jobs = self._queue.jobs
        self._table.setRowCount(len(jobs))
        for row, job in enumerate(jobs):
            cells = (
                str(row + 1),
                job.id,
                job.module,
                job.label,
                ", ".join(job.depends_on),
                self._status_text(job),
                bt.fmt_duration(job.elapsed_s),
                os.path.basename(job.log_path),
            )
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col == COLUMNS.index("Status"):
                    colour = PALETTE.get(STATUS_COLOURS.get(job.status, "fg"))
                    if job.status == bt.STATUS_SUCCESS and job.is_partial:
                        colour = PALETTE["warning"]
                    if colour:
                        item.setForeground(_brush(colour))
                if col in (COLUMNS.index("Label"), COLUMNS.index("Status")) and job.error:
                    item.setToolTip(job.error)
                self._table.setItem(row, col, item)

    def _refresh_running_row(self) -> None:
        """Tick the elapsed cell of the running job, and only that cell."""
        for row, job in enumerate(self._queue.jobs):
            if job.status != bt.STATUS_RUNNING or row >= self._table.rowCount():
                continue
            self._table.setItem(row, COLUMNS.index("Time"),
                                QTableWidgetItem(bt.fmt_duration(job.elapsed_s)))
            if self._table.currentRow() == row:
                self._load_log(job)

    def _on_selection_changed(self) -> None:
        job = self.selected_job()
        if job is None:
            return
        if not self._busy:
            self._settings_edit.setText(job.settings_path)
            self._label_edit.setText(job.label)
            self._depends_edit.setText(", ".join(job.depends_on))
            self._overrides_edit.setText(", ".join(job.override_args))
            index = self._module_combo.findData(job.module)
            if index >= 0:
                self._module_combo.setCurrentIndex(index)
        self._load_log(job)
        self._update_controls()

    def _load_log(self, job: bt.Job) -> None:
        """Show this job's own log file — never a queue-wide interleaved one."""
        if not job.log_path or not os.path.isfile(job.log_path):
            self._log_view.setPlainText(
                "" if not job.log_path else f"(no log yet at {job.log_path})")
            return
        try:
            with open(job.log_path, "r", encoding="utf-8", errors="replace") as handle:
                text = handle.read(_LOG_TAIL_BYTES)
        except OSError as exc:
            text = f"(could not read {job.log_path}: {exc})"
        if self._log_view.toPlainText() != text:
            self._log_view.setPlainText(text)
            bar = self._log_view.verticalScrollBar()
            bar.setValue(bar.maximum())

    def log_text(self) -> str:
        """The log pane (test/introspection helper)."""
        return self._log_view.toPlainText()

    # ------------------------------------------------------------------
    # small helpers
    # ------------------------------------------------------------------

    def _on_add_clicked(self) -> None:
        self.add_job(
            settings=self._settings_edit.text().strip(),
            label=self._label_edit.text().strip(),
            depends_on=_split_list(self._depends_edit.text()),
            overrides=_split_list(self._overrides_edit.text()),
        )

    def _pick_settings_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a settings file", "",
            "Settings (*.csv *.json);;All files (*)")
        if path:
            self._settings_edit.setText(path)

    def _pick_queue_to_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a queue", "", "Queue files (*.json);;All files (*)")
        if path:
            self.load_queue_from(path)

    def _pick_queue_to_save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the queue", self._path or "queue.json",
            "Queue files (*.json);;All files (*)")
        if path:
            self.save_queue_to(path)

    def _show_problems_text(self, text: str) -> None:
        self._problems_view.setPlainText(text)

    def _set_status(self, text: str, error: bool = False) -> None:
        """Report inline. Deliberately never a QMessageBox — a modal dialog
        would hang a headless run (and did, in MakeMasksScreen)."""
        self.last_error = text if error else ""
        colour = PALETTE["error"] if error else PALETTE["fg_muted"]
        self._status.setStyleSheet(f"color: {colour};")
        self._status.setText(text)

    def status_text(self) -> str:
        """Current inline status message (test/introspection helper)."""
        return self._status.text()

    def _update_controls(self) -> None:
        has_jobs = bool(self._queue.jobs)
        has_selection = self.selected_job() is not None
        for button in (self._btn_add, self._btn_load):
            button.setEnabled(not self._busy)
        for button in (self._btn_dup, self._btn_remove, self._btn_up, self._btn_down):
            button.setEnabled(has_selection and not self._busy)
        self._btn_validate.setEnabled(has_jobs)
        self._btn_save.setEnabled(has_jobs)
        self._btn_run.setEnabled(has_jobs and not self._busy and not self.has_errors())
        self._btn_stop.setEnabled(self._busy)


def _split_list(text: str) -> List[str]:
    """Split a comma-separated input box into stripped, non-empty parts."""
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def _brush(colour: str):
    """A QBrush for a palette colour string."""
    from PySide6.QtGui import QBrush, QColor

    return QBrush(QColor(colour))
