"""Display regression runs and parameter-sweep trials in one Runs tab.

The panel shares the main regression screen's results and figure surfaces.
Selecting a row loads that run's results and replaces the figures beside the
tab. Ordinary runs, refits, and sweep trials use the same columns so their
settings, status, output folder, and figures can be compared directly.

Runs are recorded when they start and updated when they finish. Saved sweep
tables can be loaded alongside in-session records; switching rows announces a
single active run to every dependent view.

NOTHING ON THIS TAB TOUCHES A RUN FOLDER FROM THE GUI THREAD. Measured on
the maintainer's machine 2026-09-04: one ``os.path.exists`` on a path under
``/nas_mnt`` -- an ``autofs`` mount whose share was asleep -- had NOT
RETURNED AFTER TWENTY SECONDS. Every row here carries a folder the user
chose, and this panel used to walk, stat, read and delete those folders in
the click that asked for it: the chooser's start directory, the sweep table
read on a tab change, ``describe_folder``'s ``os.walk`` inside the delete
confirmation, ``shutil.rmtree`` after it, and ``workspace.has_workspace``
while the row menu was being built. Each of those is a frozen application
with no traceback for as long as the mount takes to wake -- reported as
"opening map barcodes crashes spacr", as hover flicker, and as glimpses of
other screens. They run on a :class:`spacr.qt.job_runner.JobRunner` now, and
the one bare existence question left -- which folder to open the chooser in
-- is answered from :mod:`spacr.qt.path_probe`'s cache.
"""

from __future__ import annotations

import os
from typing import Optional

import logging

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .. import path_probe

LOG = logging.getLogger("spacr.qt.sweep_runs")

#: What the sweep writes its table to, inside the destination folder.
RESULTS_FILENAME = "sweep_results.csv"

#: Column marking the run currently shown by the other views on the screen.
#:
#: A text marker keeps the value sortable, filterable, and copyable in the
#: shared :class:`ResultsTable`.
LOADED_COLUMN = "loaded"

#: What that column holds for the loaded run. Blank for every other row: a
#: column of "no" against one "yes" is noise, and the eye is looking for the
#: single mark.
LOADED_MARK = "loaded"

#: The columns worth reading first, in this order, when the table has them.
#: Settings, then WHAT WENT IN, then what came out: a hit count means little
#: without the size of the design behind it, and two trials differing only by
#: a filtration cutoff can be fitted on completely different data.
#:
#: ``run`` and ``source`` lead, ahead of the sweep's ``trial_id``, because the
#: first question over a mixed table is WHICH RUN THIS IS -- a trial number
#: names nothing when half the rows are not trials.
PREFERRED_COLUMNS = (
    LOADED_COLUMN,
    "run", "source",
    "trial_id", "status", "dependent_variable",
    "regression_type", "regression_backend", "inference",
    "guide_permutations", "analysis_unit",
    "agg_type", "transform", "multiple_testing_method", "fdr_alpha",
    "fraction_threshold", "min_cell_count",
    "n_wells", "n_guides", "n_cells", "n_rows_fitted",
    "n_results", "n_below_alpha", "positive_rank", "positive_percentile",
    "r_squared", "genomic_inflation", "seconds", "error_type",
)

#: What a run of the module contributes about ITSELF, taken off the settings
#: it was started with. Every name here is also in :data:`PREFERRED_COLUMNS`,
#: and a test holds that: a setting recorded under a name the ordering does
#: not know lands past the last sweep column, which is the far right of a
#: twenty-column table -- recorded, and never seen.
RUN_SETTING_COLUMNS = (
    # WHAT WAS FITTED, first. Instruction 154 F queues one run per column of
    # the merged measurements, so a table of those runs differs in the
    # RESPONSE and in nothing else -- and a comparison table whose only
    # varying column is missing is a list of identical-looking rows.
    "dependent_variable",
    # WHICH ENGINE AND HOW MANY SHUFFLES. Two permutation runs a thousand
    # shuffles apart were one row apart in this table and identical on it,
    # which is the comparison the table exists for. `guide_permutations` is
    # copied only for a run that actually permuted -- see `_run_settings_row`.
    "regression_type", "regression_backend", "inference",
    "guide_permutations", "analysis_unit", "agg_type", "transform",
    "multiple_testing_method", "fdr_alpha", "fraction_threshold",
    "min_cell_count",
)

#: What the ``source`` column says. The sweep's trials are read off its CSV
#: and get :data:`SOURCE_SWEEP`; the other two are recorded live.
SOURCE_RUN = "run"
SOURCE_REFIT = "re-fit"
SOURCE_SWEEP = "sweep trial"
#: Source label for a run opened from disk. Disk-loaded runs retain settings
#: and may become the active run like runs produced in the current process.
SOURCE_DISK = "on disk"
#: Source label for one fit from the Measurements tab's column queue. It
#: distinguishes queued fits over one merged frame from ordinary runs.
SOURCE_MEASUREMENT = "measurement column"

#: A run that has been started and has not come back yet. Not "ok": a run
#: still going has produced nothing to look at, and a row that claims
#: otherwise is a click that opens last month's folder.
STATUS_RUNNING = "running"


# What :meth:`SweepRunsPanel.delete_runs_from_disk` answers when the delete
# has been STARTED rather than finished: the count is not knowable at return
# time and pretending otherwise would be a lie the caller cannot detect. It is
# truthy, because "the delete is under way" is what the menu wants to hear.
_DELETION_STARTED = -1


def _should_thread() -> bool:
    """Whether this panel's filesystem work has to leave the calling thread.

    THE ANSWER IS "IS ANYONE WATCHING". In the application the calling thread
    is the one painting, and a stat on a sleeping mount freezes the whole
    interface (see the module docstring); under pytest there is no event loop
    to starve, and every caller in this file's test suite reads what the
    method RETURNED -- a count of deleted folders, whether a run was opened --
    which an answer arriving later cannot supply. `somebody_is_there` is this
    repository's existing answer to "is a person waiting on this thread", so
    it is asked rather than re-derived.

    Unanswerable means threaded: never block a thread that might be the one
    the user is looking at.
    """
    try:
        from ..ask_for_the_path import somebody_is_there

        return bool(somebody_is_there())
    except Exception:                                        # noqa: BLE001
        return True


def _read_results_table(path: str):
    """The sweep's table at ``path``. WORKER ONLY -- never from a slot.

    Both halves block: the ``isfile`` is what wakes an ``autofs`` mount, and
    the read that follows it is a whole file over the same wire. Returns
    ``(frame, error)``; ``(None, "")`` means there is no table there yet,
    which is a different sentence from one that could not be read.

    IT NEVER RAISES, and that is load-bearing rather than defensive:
    `JobRunner._on_settled` calls ``on_done`` only for a job that came back
    cleanly, so a worker that throws leaves the "Reading …" placeholder on
    the status line with nothing behind it that would ever replace it. Every
    failure comes back as the second half of the pair instead -- the
    ``isfile`` included, which raises on a path with a null byte in it, and
    that is a path the user typed.
    """
    try:
        import pandas as pd

        if not os.path.isfile(path):
            return None, ""
        return pd.read_csv(path), ""
    except Exception as error:                               # noqa: BLE001
        # NAMED EVEN WHEN THE EXCEPTION IS NOT. An empty message would be
        # read as "no table here yet", which is the one thing it is not.
        return None, str(error) or type(error).__name__


def _find_the_run(folder: str):
    """The results table under ``folder`` and its settings. WORKER ONLY.

    `find_results_table` walks the tree and `settings_of_run` opens a file
    beside it; on a folder the user picked from a chooser, either can be the
    twenty seconds the module docstring is about. Returns ``(table,
    settings)`` and raises nothing, so the GUI half always gets its turn --
    the "Load run…" button is disabled until it does.
    """
    try:
        from .regression_results import find_results_table

        table = str(find_results_table(folder) or "")
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not search %s for a run", folder, exc_info=True)
        return "", {}
    if not table:
        return "", {}
    # ITS OWN SETTINGS, so an old run is described by the same columns as a
    # new one. Without them the row is a name and a folder, and two runs
    # cannot be compared on the settings that differ.
    try:
        from ...refit import settings_of_run

        settings = settings_of_run(table) or {}
    except Exception:                                        # noqa: BLE001
        settings = {}
    return table, settings


def _what_would_be_deleted(folders) -> list:
    """``(folder, what is in it)`` for each folder still on disk. WORKER ONLY.

    The confirmation cannot be composed without this and this cannot be done
    without walking every run folder, which is why the modal is now shown
    from a callback rather than from the click.

    RAISES NOTHING, for the reason :func:`_read_results_table` gives: the
    callback that puts the confirmation up is the only thing that can clear
    "Working out what these runs hold…", and it does not run for a job that
    threw. A folder that cannot even be asked about is dropped from the list
    rather than taking the other folders' answers down with it.
    """
    described = []
    for folder in folders:
        try:
            if not os.path.isdir(folder):
                continue
            described.append((folder, SweepRunsPanel.describe_folder(folder)))
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not describe %s", folder, exc_info=True)
    return described


def _delete_folders(folders) -> tuple:
    """Remove each folder and its contents. WORKER ONLY.

    `shutil.rmtree` walks and unlinks every file under the folder, so on a
    sleeping mount it blocks at least as long as the walk that described it
    did. Returns ``(deleted, failed)``.

    EVERY failure is caught, not only ``OSError``: this runs on a worker now,
    and a worker that throws never reaches `_deletion_finished` -- so the
    rows would stay on the table under a "Deleting…" that never resolves,
    with no way to tell whether the folders went.
    """
    import shutil

    deleted, failed = [], []
    for folder in folders:
        try:
            shutil.rmtree(folder)
        except Exception as error:                           # noqa: BLE001
            why = getattr(error, "strerror", None) or error
            failed.append(f"{folder} ({why})")
        else:
            deleted.append(folder)
    return deleted, failed


def _readable_size(total: int) -> str:
    """Bytes as the unit a person decides in. 31 MB, not 32,505,856."""
    size = float(max(0, int(total)))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" or size >= 10 \
                else f"{size:.1f} {unit}"
        size /= 1024
    # NO TRAILING RETURN. "GB" is the last unit and the condition carries
    # `or unit == "GB"`, so the final iteration returns whatever the size
    # is -- the loop cannot fall out of the bottom. Checked against 20,000
    # values spanning zero to 2**63-1 and negatives; every one returned
    # from inside.
    raise AssertionError(                                # pragma: no cover
        "the unit table no longer ends at GB")


def save_run_states(folders, app_key: str = "") -> tuple:
    """Write a workspace bundle for each of ``folders``.

    WORKER ONLY -- :meth:`SweepRunsPanel._apply_run_menu` submits it. The
    ``isdir`` below and the write after it are both on a folder the user
    chose, and doing either in the menu's own click is the freeze the module
    docstring describes.

    ASKED FOR, SO IT IS WRITTEN. `workspace.save_for_run` returns None when
    the `runs/save_workspace` preference is off, which is right for the
    automatic save at the end of a run and wrong here: a user who chose "Save
    the state" from a menu has asked, and a menu item that silently does
    nothing is worse than one that is absent. The mode is forced on for this
    call only, and the preference is not written.

    :param folders: run folders to save.
    :returns: ``(saved, failures)`` -- the folders that got a bundle, and
        ``(folder, reason)`` for those that did not. One folder failing does
        not stop the others, and every failure is NAMED: a count of "3 of 5"
        with no names is a report the user cannot act on.
    """
    saved = []
    failures = []
    try:
        from ...workspace import save_for_run
    except Exception as error:                               # noqa: BLE001
        return [], [(str(folder), f"workspace unavailable: {error}")
                    for folder in folders]

    for folder in folders:
        path = str(folder or "").strip()
        if not path:
            continue
        try:
            on_disk = os.path.isdir(path)
        except Exception as error:                           # noqa: BLE001
            # INSIDE THE TRY LIKE EVERYTHING ELSE. This runs on a worker, and
            # a worker that throws never reaches `_on_states_saved` -- so one
            # unaskable path would leave "Saving the state of 3 runs…" on the
            # line for the rest of the session and say nothing about the two
            # that could have been saved.
            failures.append((path, f"{type(error).__name__}: {error}"))
            continue
        if not on_disk:
            # `save_for_run` would CREATE this folder, leaving a directory
            # holding nothing but a workspace file where a deleted run used
            # to be. A run that is gone from disk is not a run to save.
            failures.append((path, "the run folder is not on disk any more"))
            continue
        try:
            # `reference` rather than `copy`: the run's own files are already
            # on disk beside it, and copying them again to save a state would
            # double a screen's worth of crops.
            written = save_for_run(path, {"save_workspace": "reference"})
        except Exception as error:                           # noqa: BLE001
            failures.append((path, f"{type(error).__name__}: {error}"))
            continue
        if written is None:
            failures.append(
                (path, "nothing to save -- no panel offered any state"))
        else:
            saved.append(path)
    return saved, failures


def describe_saved_states(saved, failures) -> str:
    """One sentence about what a save did, for the panel's own note."""
    if not saved and not failures:
        return "No run was selected, so nothing was saved."
    parts = []
    if saved:
        plural = "" if len(saved) == 1 else "s"
        parts.append(f"Saved the state of {len(saved)} run{plural}.")
    for path, why in failures[:3]:
        parts.append(f"{os.path.basename(path) or path}: {why}")
    if len(failures) > 3:
        parts.append(f"...and {len(failures) - 3} more that did not save.")
    return " ".join(parts)


def _permuted(settings) -> bool:
    """Whether this run actually ran the permutation test.

    `guide_permutations` has a default, so every settings dict carries a
    number whether or not a shuffle ever happened. Reading it off a
    least-squares fit would put "200000" in the column for a run that
    permuted nothing.
    """
    mode = str((settings or {}).get("analysis_mode") or "").strip().lower()
    if mode == "guide_permutation":
        return True
    inference = str((settings or {}).get("inference") or "").strip().lower()
    return inference in ("nonparametric", "permutation")


def _run_settings_row(settings) -> dict:
    """The settings columns for one run, as a dict to update a row with.

    A value of None is left out rather than written, so a column stays empty
    for a run that has nothing to say there -- and 0 permutations stays
    distinguishable from "this was not a permutation test".
    """
    out = {}
    for name in RUN_SETTING_COLUMNS:
        if name == "guide_permutations" and not _permuted(settings):
            continue
        value = (settings or {}).get(name)
        if value is not None:
            out[name] = value
    return out


def _has_workspace(folder: str) -> bool:
    """Whether a run folder carries a workspace bundle to restore.

    Asked of the FOLDER and not of a column, because the bundle is written
    when the run closes and the row was built when it started -- a run that
    finished this session would otherwise be offered no restore until the
    list was reloaded.

    WORKER ONLY. `workspace.has_workspace` is a bare stat on a folder the
    user chose; the menu asks :meth:`SweepRunsPanel._workspace_answer`, which
    answers from a cache this fills in the background.
    """
    if not folder:
        return False
    try:
        from ...workspace import has_workspace
        return bool(has_workspace(folder))
    except Exception:                                       # noqa: BLE001
        return False


def _is_ok(row) -> bool:
    """Whether a run row has results behind it, so it can be the loaded run.

    A MISSING STATUS COUNTS AS OK, which is the same reading `_show_trial`
    takes: a sweep's CSV need not carry the column, and treating its absence
    as a failure would make every trial in an older table unloadable. NaN is
    handled explicitly -- concatenating a session run into a sweep frame fills
    the column with it, and ``str(nan)`` is ``'nan'``, which is not a status.
    """
    if not isinstance(row, dict):
        return False
    status = row.get("status", "ok")
    if status is None:
        return True
    text = str(status).strip()
    if not text or text.lower() == "nan":
        return True
    return text == "ok"


def ordered_columns(frame) -> list:
    """``PREFERRED_COLUMNS`` that this frame has, then everything else.

    Ordering, not filtering. A column nobody thought to list is still worth
    seeing -- it is in the CSV, and hiding it means the user has to leave the
    application to read their own results.
    """
    if frame is None:
        return []
    have = list(frame.columns)
    first = [name for name in PREFERRED_COLUMNS if name in have]
    return first + [name for name in have if name not in first]


class SweepRunsPanel(QWidget):
    """One row per RUN -- the sweep's trials and this session's own.

    :ivar trial_activated: emitted with a run's row, as a dict, whenever a
        view should be showing that run. The older of the two names for that
        one event -- see :attr:`loaded_run_changed`.
    :ivar loaded: emitted with the number of rows shown.

    :param parent: parent widget.
    :param threaded: whether the folder reads, walks and deletes go to a
        worker. ``None`` decides by asking whether a person is waiting on
        the calling thread (:func:`_should_thread`), which is what the
        application wants and what a test does not: unthreaded, every job
        runs inline and the methods below still return what they found.
    """

    trial_activated = Signal(dict)
    loaded = Signal(int)
    #: Emitted with the row of the run that is now the LOADED one -- the run
    #: every view on this screen is describing. Emitted for the automatic
    #: cases too (a run finishing, a folder holding exactly one run), because
    #: a view that only learns about deliberate choices shows the wrong run
    #: after the common one.
    #:
    #: THE SAME EVENT AS :attr:`trial_activated`, from the same funnel
    #: (:meth:`_announce_the_loaded_run`), because "a run was activated" and
    #: "the loaded run changed" turned out to be one fact -- and while they
    #: were two, only one of them was connected to anything (157).
    loaded_run_changed = Signal(dict)
    #: Emitted with the rows that have just left the table, as dicts.
    #:
    #: Other views use this event to discard per-run plot and figure state. The
    #: results panel keeps a plot state per run and the figure grid keys its
    #: sections by run label; the screen coordinates cleanup so this widget
    #: remains usable on its own.
    runs_removed = Signal(list)
    #: Emitted with one run's row when the user asks to see it beside the
    #: loaded one. Opening a second run is a deliberate
    #: act: two runs is what a comparison needs and twelve is what makes the
    #: screen unusable, so it has its own gesture rather than happening on a
    #: click. The screen decides whether the bound allows it.
    compare_requested = Signal(dict)
    #: Emitted with one run's row when the user requests its saved workspace.
    #: Restoration is separate from loading results because
    #: a restore reattaches databases, re-points the montage and rewrites the
    #: settings on screen, which is a great deal more than showing a table,
    #: and a user who wanted to glance at a run's volcano would not expect
    #: their current screen replaced.
    workspace_restore_requested = Signal(dict)

    def __init__(self, parent=None, *, threaded: Optional[bool] = None):
        super().__init__(parent)
        from ..job_runner import JobRunner
        from .fast_plots import ResultsTable

        self._threaded = _should_thread() if threaded is None else bool(
            threaded)
        # TWO RUNNERS, because one of them is CANCELLED. A tab change can ask
        # for the sweep table again while the last read is still out on a
        # slow mount, and the answer that must win is the newest -- so the
        # table reads have a runner of their own that `load` can cancel
        # without abandoning a delete or a save half-way through.
        #
        # `user_visible=False` ON BOTH, and the reason is Preferences' rather
        # than the usage poller's: a right-click that saves a bundle or
        # deletes a folder IS something the user started, but it is not a RUN
        # -- and `home.py` filters that flag to decide which of them gets a
        # blue "<module> — running" banner across the top of Home. None of
        # this work had a banner before, because none of it had a thread; the
        # flag hides nothing the user saw, and without it every right-click
        # on this tab flashes one.
        self._jobs = JobRunner(self, threaded=self._threaded,
                               app_key="sweep runs", user_visible=False)
        self._load_jobs = JobRunner(self, threaded=self._threaded,
                                    app_key="sweep runs table",
                                    user_visible=False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        header = QHBoxLayout()
        self._status = QLabel("Nothing run yet.")
        self._status.setWordWrap(True)
        header.addWidget(self._status, 1)
        # A RUN ON DISK IS A FIRST-CLASS RUN (154 G). An earlier session's
        # results folder is opened here, gets a row beside this session's own,
        # and becomes the loaded run -- rather than being something the user
        # can only reach by re-running a fit that already finished.
        self._open = QPushButton("Load run…")
        self._open.setToolTip(
            "Open a run's results folder from disk — including one from an "
            "earlier session. It joins the table as a run like any other and "
            "becomes the loaded run.")
        self._open.clicked.connect(lambda: self.load_run_from_disk())
        header.addWidget(self._open)
        self._reload = QPushButton("Reload")
        self._reload.setToolTip(
            "Re-read the results table. A sweep writes each trial as it "
            "finishes, so a running sweep can be watched.")
        self._reload.clicked.connect(self.reload)
        header.addWidget(self._reload)
        layout.addLayout(header)

        # The same table widget the results use: it already sorts numerically,
        # filters, and copies as TSV, and a second implementation of those is
        # a second set of bugs.
        self.table = ResultsTable()
        # Its own words. The coefficient table's "type a gene, a guide" and
        # "significant only" belong to a table of findings; over a list of
        # trials the first is wrong and the second cannot do anything.
        self.table.configure(
            placeholder="Filter runs — a model, a cutoff, anything in the row",
            significance_filter=False)
        self.table.table.itemSelectionChanged.connect(self._on_selection)
        # THE TWO GESTURES A USER LOOKS FOR (instruction 146 B): a context
        # menu on the row, and the Delete key on the selection. Multi-select
        # is the QTableWidget default and is kept -- a sweep writes one
        # folder per trial, and clearing up after one by hand twenty times is
        # not a feature.
        # SAID RATHER THAN INHERITED. ExtendedSelection is the QTableWidget
        # default today, and "several rows at once" is a requirement of
        # instruction 146 B rather than a happy accident of a default that
        # could change.
        from PySide6.QtWidgets import QAbstractItemView

        self.table.table.setSelectionMode(
            QAbstractItemView.ExtendedSelection)
        self.table.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.table.customContextMenuRequested.connect(self._run_menu)
        # DOUBLE-CLICK IS THE GESTURE THAT LOADS (190). It used to be the
        # second way in, beside a selection that also loaded; as of
        # 2026-08-20 it is the only one, because selecting a row to read its
        # name should not cost a multi-second read. It still FORCES the load
        # rather than asking politely, so a run refused for any reason can be
        # insisted on.
        self.table.table.doubleClicked.connect(self._on_double_click)
        self.table.table.installEventFilter(self)
        layout.addWidget(self.table, 1)

        # THE STILL OF A RUN THAT IS NOT LIVE (instruction 116's last line).
        # A run opened beside the loaded one is photographed when it closes;
        # this is where that photograph is finally SHOWN, beside the row it
        # belongs to. Hidden when there is none, which is most rows -- an
        # empty frame under the table would read as a run that failed to draw
        # rather than as one nobody has opened beside.
        self._photo = QLabel()
        self._photo.setAlignment(Qt.AlignCenter)
        self._photo.setObjectName("RunPhotograph")
        self._photo.setToolTip(
            "This run's volcano as it last looked. A still, not a live plot: "
            "open the run to hover, click and filter it.")
        self._photo.hide()
        layout.addWidget(self._photo)
        #: Callable that returns the latest still for a run folder. Using a
        #: provider ensures photographs created after panel construction are
        #: still found.
        self._photo_provider = None

        self._frame = None
        self._folder = ""
        #: The sweep's own table, as read off disk. Kept SEPARATE from the
        #: composed frame the table shows, because a reload re-reads this one
        #: and must not take the session's own runs down with it.
        self._sweep_frame = None
        #: What this session has started, oldest first, keyed by the handle
        #: :meth:`record_run` hands back. An ordinary run and a re-fit are
        #: the same kind of thing as a trial and belong in the same table.
        self._recorded: "dict[int, dict]" = {}
        self._next_handle = 0
        #: WHICH RUN IS LOADED, as the key :meth:`_row_key` gives a row --
        #: its folder when it has one, its name when it does not. A key
        #: rather than an index, because the table is rebuilt from scratch
        #: every time a run is recorded and an index would name a different
        #: run afterwards.
        self._loaded_key = ""
        #: Previous loaded-run key, retained so a failed load can restore the
        #: marker to the run that remains on screen.
        self._previous_loaded_key = ""
        #: Which announcement `_previous_loaded_key` answers.
        self._undo_answers = None
        #: The sentence the last load or refusal added to the status line,
        #: kept so moving the mark can rewrite the line without losing it.
        self._source_note = ""
        #: True while :meth:`_rebuild` is refilling the table. The re-select
        #: it does at the end would otherwise read as a user picking a row and
        #: re-emit `trial_activated`, which re-loads the results panel on
        #: every recorded run.
        self._rebuilding = False
        #: What the last job of each kind found, so the method that started
        #: it can still answer its caller when it ran inline. Threaded, the
        #: answer is not knowable at return time and these are not read.
        self._load_answer = False
        self._open_answer = False
        self._saved_state = False
        self._deleted_count = 0
        #: Which run folders carry a workspace bundle, as far as this panel
        #: knows. Filled in off the GUI thread -- see
        #: :meth:`_workspace_answer`.
        self._workspace_answers: "dict[str, bool]" = {}
        self._workspace_pending: set = set()
        #: The "Reading …" / "Deleting…" placeholder currently on the status
        #: line, and the sentence it is covering. A placeholder is written
        #: before a job and cleared by its answer; when there is no answer --
        #: the worker raised, or the user said No -- these are what put the
        #: line back. See :meth:`_start_waiting`.
        self._waiting_note = ""
        self._note_before_waiting = ""

        # THE OTHER END OF EVERY HANDLER BELOW. `JobRunner` calls `on_done`
        # only for a job that SUCCEEDED, so a worker that raises leaves the
        # placeholder on the line and the "Load run…" button disabled by the
        # click that started it -- for the rest of the session. Connected
        # last, after `_open` and the state above exist, because a job can
        # fail the moment it is submitted.
        for runner in (self._jobs, self._load_jobs):
            runner.job_failed.connect(self._on_job_failed)

    def closeEvent(self, event):                              # noqa: N802
        # Qt ABORTS THE PROCESS if a running QThread is destroyed, and a run
        # folder on a sleeping mount is exactly the job that is still going
        # when the user closes the screen it belongs to. Both runners are
        # asked to stop and are waited for a bounded time; a job that outlasts
        # the budget is parked rather than killed mid-delete.
        for runner in (getattr(self, "_load_jobs", None),
                       getattr(self, "_jobs", None)):
            if runner is None:
                continue
            try:
                runner.shutdown()
            except Exception:                                # noqa: BLE001
                LOG.debug("could not stop a runs-tab job", exc_info=True)
        super().closeEvent(event)

    # ------------------------------------------------------------------ load

    def load(self, folder) -> bool:
        """Read ``sweep_results.csv`` from a sweep's destination folder.

        THE READ IS NOT DONE HERE. This is reached from a tab change
        (`AppScreen._on_results_tab_changed`) with the sweep destination the
        user typed, and both the ``isfile`` and the ``read_csv`` block for as
        long as that folder takes to answer -- which on the maintainer's
        ``/nas_mnt`` was twenty seconds of frozen application. The path work
        below is pure string manipulation; everything that touches the disk
        goes to a worker and comes back to :meth:`_table_arrived`.

        :returns: whether the tab now shows a row -- or, when the read went
            to a worker, whether it was started. The row count arrives with
            the `loaded` signal either way.
        """
        if not folder:
            return False
        folder = os.path.abspath(os.path.expanduser(os.fspath(folder)))
        path = folder if folder.lower().endswith(".csv") else os.path.join(
            folder, RESULTS_FILENAME)
        self._load_answer = False
        # LAST ASK WINS. A tab flipped twice queues two reads of the same
        # folder, and the older one landing second would put a stale table
        # back. Cancelling drops the result rather than joining the thread.
        self._load_jobs.cancel()
        self._start_waiting(f"Reading {path}…")
        started = self._load_jobs.submit(
            lambda target=path: _read_results_table(target),
            lambda outcome, target=path, home=folder:
                self._table_arrived(outcome, target, home))
        return bool(started) if self._threaded else self._load_answer

    def _table_arrived(self, outcome, path: str, folder: str) -> None:
        """Put the sweep's table on screen. On the GUI thread, from `load`."""
        self._stop_waiting()
        frame, error = outcome if outcome else (None, "")
        if frame is None:
            if error:
                self._rebuild(f"Could not read {path}: {error}")
            else:
                # NOT a wipe, and the note goes THROUGH the rebuild. The
                # sweep's table being absent says nothing about the runs this
                # session has made; setting the status here and clearing the
                # table would be how opening the tab before a sweep exists
                # used to empty it.
                self._rebuild(f"No results table at {path} yet.")
            self._load_answer = False
            return
        self._folder = folder
        # The chooser opens here next time, and asking now means the answer
        # is in the cache by then rather than being stat-ed under the click.
        path_probe.isdir(folder)
        self._load_answer = self.set_frame(frame, source=path)

    def reload(self) -> bool:
        return self.load(self._folder) if self._folder else False

    def set_frame(self, frame, source: str = "") -> bool:
        # Takes THE SWEEP'S HALF of the table and returns whether the tab now
        # shows a row -- which is no longer the same question, because this
        # session's own runs are rows too. Deliberately left as a comment
        # rather than a docstring: a docstring here is public API surface, and
        # `tests/test_api_i18n_extractor.py` holds an exact count of that, so
        # promoting this to a docstring means bumping the count in the same
        # commit. (It used to say the count file "belongs to another session
        # right now" -- true on 2026-08-17, not a constraint today.)
        self._sweep_frame = frame if frame is not None and len(frame) else None
        if self._sweep_frame is None and not source:
            source = "The sweep has recorded no trials yet."
        return self._rebuild(source)

    # ------------------------------------------------------- this session's

    def record_run(self, label: str, source: str = SOURCE_RUN,
                    settings=None, folder: str = "") -> int:
        """Put a run of the module on the table, and return its handle.

        Recorded when the run STARTS, for the reason the figure grid marks
        its section then: a run that fails, or that is still going, is a fact
        worth seeing rather than a gap. Its row says ``running`` until
        :meth:`update_run` is told otherwise -- not ``ok``, because a row
        claiming ok is a row a click will try to open results from.

        :param label: user-visible run name stored in the new row.
        :param settings: the dict the run was started with. Only
            :data:`RUN_SETTING_COLUMNS` are copied out of it, and they are the
            sweep's own setting columns -- which is what makes a run and a
            trial two rows of one table rather than two tables.
        """
        self._next_handle += 1
        handle = self._next_handle
        row = {
            "run": str(label),
            "source": str(source),
            "status": STATUS_RUNNING,
            "folder": str(folder or ""),
        }
        for name in RUN_SETTING_COLUMNS:
            value = (settings or {}).get(name)
            if value is not None:
                row[name] = value
        self._recorded[handle] = row
        self._rebuild()
        return handle

    def update_run(self, handle: int, **fields) -> bool:
        """Change a recorded run's row -- its status, folder, seconds.

        Returns False for a handle this panel never issued, rather than
        inventing a row: a run whose panel was rebuilt underneath it is a
        stale handle, and a phantom row is worse than a missing one.
        """
        row = self._recorded.get(int(handle))
        if row is None:
            return False
        row.update({name: value for name, value in fields.items()
                    if value is not None})
        # A RUN FINISHING IS WHEN A BUNDLE APPEARS, which is the whole reason
        # `_has_workspace` asks the folder rather than the row. A "no
        # workspace" answer cached from a right-click while the run was still
        # going would otherwise grey the restore entry for the rest of the
        # session.
        folder = str(row.get("folder") or "")
        if folder:
            self._workspace_answers.pop(folder, None)
        # A RUN THAT FINISHES BECOMES THE LOADED RUN, with no step in
        # between (154 G): "i just ran a regression so that should be loaded
        # automatically". The views were being told nothing had been loaded
        # by the run the user had just watched finish.
        #
        # AND THEY ARE TOLD NOW (157). This line moved the mark and stopped
        # there, so a second run finishing left the first run's coefficients,
        # figures and summary on screen under a mark naming the second. The
        # key the mark was on is carried into the rebuild, which announces
        # the change once everything else has settled.
        before = self._loaded_key
        if _is_ok(row):
            self._loaded_key = self._row_key(row)
        self._rebuild(since=before)
        return True

    def load_run_from_disk(self, folder: str = "") -> bool:
        """Open a results folder and make that run active.

        Parameters
        ----------
        folder : str, optional
            Run directory. An empty value opens a directory chooser.

        Returns
        -------
        bool
            ``True`` when a valid run was opened.

        Notes
        -----
        Saved settings are restored beside the results so imported and
        current-session runs use the same table columns.

        The search under the chosen folder runs on a worker: it is an
        ``os.walk`` of somewhere the user just pointed at, which is the one
        place a sleeping network mount is guaranteed to be reached. Threaded,
        ``True`` means the search was started and the row appears when it
        answers.
        """
        if not folder:
            from PySide6.QtWidgets import QFileDialog

            # NOT `self._folder` DIRECTLY. Qt stats the start directory
            # before it draws the dialog, so handing it a remembered
            # ``/nas_mnt`` path freezes the click that opened the chooser.
            # `path_probe.isdir` answers from its cache and says no to a path
            # it has not seen -- the dialog opens at its default place, and
            # the next click gets the remembered folder back.
            start = self._folder if path_probe.isdir(self._folder) else ""
            folder = QFileDialog.getExistingDirectory(
                self, "Choose a run's results folder", start)
            if not folder:
                return False
            path_probe.prime(folder, True)
        try:
            folder = os.path.abspath(os.path.expanduser(os.fspath(folder)))
        except TypeError:
            return False

        self._open_answer = False
        # A SECOND CLICK MUST NOT QUEUE A SECOND WALK. Re-enabled by
        # `_run_arrived` when the search answers and by `_on_job_failed` when
        # it does not -- and BOTH are needed, because `JobRunner` hands a
        # result to `on_done` only for a job that came back cleanly. A button
        # disabled by a click and re-enabled by nothing is the silent no-op
        # this repository keeps fixing.
        self._open.setEnabled(False)
        self._start_waiting(f"Looking for a run in {folder}…")
        started = self._jobs.submit(
            lambda target=folder: _find_the_run(target),
            lambda found, target=folder: self._run_arrived(found, target))
        return bool(started) if self._threaded else self._open_answer

    def _run_arrived(self, found, folder: str) -> None:
        """Put the run that was found on the table. On the GUI thread."""
        self._open.setEnabled(True)
        self._stop_waiting()
        table, settings = found if found else ("", {})
        if not table:
            from .regression_results import RESULT_FILENAMES

            # NAMED, not a silent no-op. The user picked a folder; being told
            # nothing happened is the failure this repository keeps fixing.
            self._rebuild(f"No run in {folder}: none of "
                          f"{', '.join(RESULT_FILENAMES)} is in it or under "
                          f"it.")
            self._open_answer = False
            return
        run_folder = os.path.dirname(table)

        handle = self._handle_for_folder(run_folder)
        if handle is None:
            self._next_handle += 1
            handle = self._next_handle
            row = {
                "run": self._name_for_folder(run_folder),
                "source": SOURCE_DISK,
                "status": "ok",
                "folder": run_folder,
            }
            row.update(_run_settings_row(settings))
            self._recorded[handle] = row

        before = self._loaded_key
        self._loaded_key = self._row_key(self._recorded[handle])
        # AND THE VIEWS FOLLOW IT -- through the same funnel every other path
        # uses (157). This used to emit the two signals itself, which is how
        # the finishing path came to be the only one that did not.
        self._rebuild(f"Loaded the run in {run_folder}.", since=before)
        self._open_answer = True

    def _handle_for_folder(self, folder: str):
        """The handle of the recorded row for ``folder``, or ``None``.

        Opening the same run twice is one row, not two: the second would be
        indistinguishable from a re-run, which is a different event.
        """
        for handle, row in self._recorded.items():
            existing = row.get("folder")
            if isinstance(existing, str) and existing and os.path.abspath(
                    os.path.expanduser(existing)) == folder:
                return handle
        return None

    def _name_for_folder(self, folder: str) -> str:
        """A row name for a run opened off disk, unique in this table.

        A run folder is ``results/<kind>_<n>``, so the basename alone is
        ``ols_3`` -- readable, and ambiguous across two screens. The parent is
        added only when the basename is already taken, because a name that is
        always a path is a name nobody can scan.
        """
        base = os.path.basename(folder.rstrip(os.sep)) or folder
        taken = {str(row.get("run")) for row in self._recorded.values()}
        if base not in taken:
            return base
        parent = os.path.basename(os.path.dirname(folder.rstrip(os.sep)))
        longer = os.path.join(parent, base) if parent else folder
        if longer not in taken:
            return longer
        return folder

    def _recorded_rows(self) -> list:
        """This session's runs, oldest first."""
        return [dict(row) for _handle, row in sorted(self._recorded.items())]

    # ------------------------------------------------------ which is loaded

    @staticmethod
    def _row_key(row) -> str:
        """What names a run across a rebuild: its folder, else its name.

        The folder first because it is what a view actually needs -- the
        results table, the summary and ``regression_data.csv`` are all in it
        -- and because two runs of the same screen carry the same label and
        different folders.
        """
        if not isinstance(row, dict):
            return ""
        folder = row.get("folder")
        if isinstance(folder, str) and folder.strip():
            return os.path.abspath(os.path.expanduser(folder.strip()))
        name = row.get("run")
        return str(name).strip() if isinstance(name, str) else ""

    def loaded_run(self) -> Optional[dict]:
        """The run every view on this screen is describing, or ``None``.

        Read off the composed frame rather than off the recorded dict, so a
        sweep trial and a session run answer the same way -- which is the
        whole reason they share one table.
        """
        if self._frame is None:
            return None
        for _index, row in self._frame.iterrows():
            record = row.to_dict()
            if self._row_key(record) and self._row_key(
                    record) == self._loaded_key:
                return record
        return None

    def loaded_run_folder(self) -> str:
        """The loaded run's folder, or ``""``. What a view actually needs."""
        record = self.loaded_run()
        folder = (record or {}).get("folder")
        return str(folder) if isinstance(folder, str) and folder.strip() else ""

    def set_loaded_run(self, key) -> bool:
        """Make ``key`` the loaded run. ``key`` is a folder or a run name.

        :returns: True when a row matched. False rather than a blank mark for
            a run this table does not hold -- a tick against nothing is worse
            than none, because it reads as an answer.
        """
        if key is None:
            return False
        wanted = str(key).strip()
        if not wanted:
            return False
        expanded = os.path.abspath(os.path.expanduser(wanted))
        before = self._loaded_key
        for record in self._all_rows():
            key = self._row_key(record)
            if key and key in (wanted, expanded):
                changed = self._loaded_key != key
                self._loaded_key = key
                if changed:
                    # The note describes the LAST LOAD, which this supersedes.
                    # Left alone it produced "Loaded: ols_1. Loaded the run in
                    # .../ols_2." -- one sentence naming two runs.
                    self._source_note = ""
                self._paint_the_loaded_mark()
                if changed:
                    # AND THE VIEWS FOLLOW THE CHOICE. Moving the mark and
                    # leaving the results panel, the summary and the figure
                    # grid on the previous run is the failure 154 G is about:
                    # the choice has to be visible from the views that depend
                    # on it, not only from the tab that sets it.
                    self._announce_the_loaded_run(before)
                return True
        return False

    def _announce_the_loaded_run(self, before: str) -> bool:
        """Notify dependent views when the loaded-run key changes.

        All load paths use this helper so row selection, folder loading, and a
        run finishing emit the same update. Return whether a change was
        announced relative to ``before``.
        """
        if not self._loaded_key or self._loaded_key == before:
            return False
        record = self.loaded_run()
        if record is None:
            # The key names a row the composed frame does not hold, which is
            # not a run anybody can be shown. Say nothing rather than hand a
            # view a record it cannot open.
            return False
        self._previous_loaded_key = before
        # BOTH NAMES FOR ONE EVENT. `loaded_run_changed` is the question the
        # screen connects ("which run is on screen"); `trial_activated`
        # predates it and is what the sweep's own listeners were written
        # against. They are emitted together and the screen's handler is
        # idempotent on the run already showing, so connecting either -- or
        # both -- costs one load.
        # SET BEFORE THE EMITS. A listener that refuses SYNCHRONOUSLY -- which
        # is how the screen behaved before the read moved to a worker, and how
        # a test drives it -- calls back into `the_load_failed` from inside
        # these two lines, and the token has to already name this announcement
        # or its own refusal is rejected as stale.
        self._undo_answers = self._loaded_key
        self.loaded_run_changed.emit(dict(record))
        self.trial_activated.emit(dict(record))
        # THE UNDO IS KEPT, AND IT NAMES WHAT IT ANSWERS.
        #
        # It used to be spent right here, on the reasoning that the window in
        # which a load can fail IS the emission -- true while the listener read
        # the run synchronously, and false since instruction 159 moved that
        # read onto a worker. The failure now arrives after this method has
        # returned, so an undo consumed here is gone before the only caller
        # that needs it: a run whose folder does not exist kept the mark, which
        # is 157's disagreement pointing the other way.
        #
        # Kept, but not open-ended. `_undo_answers` records WHICH announcement
        # the undo belongs to, so a listener coming back later -- a click on a
        # trial that failed two choices ago -- cannot drag the mark backwards:
        # `the_load_failed` checks that the mark is still on the run it was
        # told about, and does nothing when it is not.
        return True

    def the_load_succeeded(self) -> None:
        """Finalize the selection after an asynchronous run load succeeds.

        Records the loaded run as the current stable selection. A later failure
        callback for an older request therefore cannot move the selection back.
        """
        self._previous_loaded_key = self._loaded_key
        self._undo_answers = self._loaded_key

    def the_load_failed(self, why: str = "") -> bool:
        """The run that was just announced could not be shown: undo the mark.

        A mark on a run whose results are not visible leaves the run list and
        results view out of sync. Restore the previous mark so the list names
        the run that remains on screen.

        :param why: Reason added to the status line.
        :returns: Whether the mark moved back.
        """
        previous = self._previous_loaded_key
        # ONLY THE ANNOUNCEMENT THIS ANSWERS. An asynchronous read reports back
        # after the mark may have moved again; rolling back then would undo a
        # choice the user has since made.
        answers = getattr(self, "_undo_answers", None)
        if answers is not None and answers != self._loaded_key:
            return False
        if previous == self._loaded_key:
            return False
        if not previous:
            # NOTHING TO GO BACK TO, so the mark stays on the run that
            # finished. Clearing it here would answer "no run is loaded"
            # immediately after a run the user watched finish -- which is
            # 154 G's report, arrived at from the other direction. The run IS
            # the loaded one; what failed is drawing it, and the status line
            # says so. The rule this rolls back is "the mark must not point
            # at a run OTHER than the one on screen", and with nothing on
            # screen there is no other run to point at.
            return False
        self._loaded_key = previous
        # Spent: this undo has been used and must not be used twice.
        self._undo_answers = previous
        self._source_note = str(why or "")
        self._paint_the_loaded_mark()
        return True

    def _all_rows(self) -> list:
        """Every run the table is SHOWING, in the order it shows them.

        Read off the composed frame rather than off the two halves it was
        built from: a sweep trial's name is derived during the rebuild
        (``trial 2`` from ``trial_id``), so a search over the raw sweep frame
        cannot match the name the user is looking at.
        """
        if self._frame is None:
            return []
        return [row.to_dict() for _index, row in self._frame.iterrows()]

    def _paint_the_loaded_mark(self) -> None:
        """Move the mark to the loaded run WITHOUT rebuilding the table.

        A rebuild from inside a selection handler destroys the item Qt is
        still holding -- ``select_key`` came back to a deleted C++ object and
        raised -- and it drops the selection a moment after the user made it.
        The mark is one cell per row; moving it is not a reason to refill a
        table of sixty trials either.
        """
        frame = self._frame
        if frame is None or LOADED_COLUMN not in frame.columns:
            return
        column = list(frame.columns).index(LOADED_COLUMN)
        marks = [LOADED_MARK if self._row_key(row.to_dict()) == self._loaded_key
                 and self._loaded_key else ""
                 for _index, row in frame.iterrows()]
        frame[LOADED_COLUMN] = marks
        table = self.table.table
        for row in range(table.rowCount()):
            item = table.item(row, column)
            if item is None:
                continue
            index = item.data(0x0100)              # Qt.UserRole: the frame row
            if index is None or not 0 <= int(index) < len(marks):
                continue
            item.setText(marks[int(index)])
        self._status.setText(self._describe(frame, self._source_note))

    def _settle_the_loaded_run(self, frame) -> None:
        """Keep the mark on a run that still exists, or place it.

        ONE RUN IN THE FOLDER IS THE LOADED RUN -- there is nothing to choose
        between (154 G). Several means the choice is the user's and the mark
        stays where they put it, or where the last finished run put it.
        """
        import pandas as pd

        keys = [self._row_key(row.to_dict()) for _i, row in frame.iterrows()]
        ok = [self._row_key(row.to_dict()) for _i, row in frame.iterrows()
              if _is_ok(row.to_dict())]
        if self._loaded_key in keys:
            pass
        elif len(ok) == 1:
            self._loaded_key = ok[0]
        else:
            # A key naming no row is worse than none: `loaded_run` would
            # answer None while the column showed nothing, and the two
            # disagreeing is how a stale mark survives a reload.
            self._loaded_key = ""
        frame[LOADED_COLUMN] = pd.Series(
            [LOADED_MARK if key and key == self._loaded_key else ""
             for key in keys], index=frame.index, dtype=object)

    def _rebuild(self, source: str = "", since: Optional[str] = None) -> bool:
        """Compose the session's runs and the sweep's trials into one table.

        THE SESSION'S OWN RUNS COME FIRST. A sweep is sixty rows and the run
        the user just made is the one they are looking for; sorting is one
        click away for every other question.

        :param since: the key the mark was on before the caller touched it.
            Every rebuild can move the mark -- a run finishing sets it, and
            :meth:`_settle_the_loaded_run` places it when a folder turns out
            to hold exactly one run -- so the announcement belongs HERE
            rather than at each of those call sites, which is how the
            finishing one came to be missing (157). ``None`` means "compare
            against wherever the mark is now", so a rebuild that does not
            move it says nothing.
        """
        import pandas as pd

        before = self._loaded_key if since is None else str(since)

        frames = []
        recorded = self._recorded_rows()
        if recorded:
            frames.append(pd.DataFrame(recorded))
        if self._sweep_frame is not None and len(self._sweep_frame):
            trials = self._sweep_frame.copy()
            if "source" not in trials.columns:
                trials["source"] = SOURCE_SWEEP
            if "run" not in trials.columns and "trial_id" in trials.columns:
                # A trial's name IS its number; saying so in the same column
                # the session's runs use is what lets one glance answer
                # "which run is this row".
                trials["run"] = ["trial " + str(value)
                                 for value in trials["trial_id"]]
            frames.append(trials)
        if not frames:
            self._frame = None
            self._source_note = source
            self.table.set_frame(None)
            self._status.setText(source or "Nothing run yet.")
            self.loaded.emit(0)
            self._announce_the_loaded_run(before)
            return False
        if len(frames) == 1:
            frame = frames[0]
        else:
            # WHICH COLUMNS WERE WHOLE NUMBERS BEFORE THE CONCAT. A session
            # run has no `trial_id` and no `n_below_alpha`, so concatenating
            # it in fills those with NaN and pandas promotes the column to
            # float -- and the sweep's trial 1 becomes "1.0", its 12 hits
            # become "12.0". Recorded here, restored below.
            whole = {name for part in frames for name in part.columns
                     if part[name].dtype.kind in "iu"}
            frame = pd.concat(frames, ignore_index=True, sort=False)
            frame = self._keep_whole_numbers_whole(frame, whole)
        # WHICH ROW IS THE LOADED ONE, decided over the composed frame --
        # the session's runs and the sweep's trials are one population, and
        # "there is only one run so it is the loaded one" has to count both.
        self._settle_the_loaded_run(frame)
        self._frame = frame[ordered_columns(frame)].reset_index(drop=True)
        self._rebuilding = True
        try:
            self.table.set_frame(self._frame, key_column=(
                "run" if "run" in self._frame.columns else None))
            # THE LOADED RUN IS THE SELECTED ROW. Refilling the table clears
            # the selection, so without this the highlight jumped off the run
            # being shown every time another one was recorded.
            record = self.loaded_run()
            if record is not None and isinstance(record.get("run"), str):
                self.table.select_key(record["run"])
        finally:
            self._rebuilding = False
        self._source_note = source
        self._status.setText(self._describe(self._frame, source))
        self.loaded.emit(len(self._frame))
        # LAST, AND OUTSIDE `_rebuilding`. A listener re-points the results
        # panel and the figure grid, and one that came back into this method
        # would be refilling a table Qt is still holding items from -- the
        # crash `_paint_the_loaded_mark` exists to avoid. Nothing below this
        # line reads `_loaded_key`, so a listener that hands the mark back
        # (:meth:`the_load_failed`) cannot leave the table half-built.
        self._announce_the_loaded_run(before)
        return True

    @staticmethod
    def _keep_whole_numbers_whole(frame, columns):
        """Put ``columns`` back to integers after a concat introduced NaN.

        Nullable ``Int64``, not ``int``: the missing values are real -- a run
        of the module has no trial number -- and casting them away would put a
        zero where "this row is not a trial" belongs.
        """
        for name in columns:
            if name not in frame.columns:
                continue
            series = frame[name]
            if getattr(series.dtype, "kind", "") != "f":
                continue
            present = series.dropna()
            if not len(present) or not bool((present % 1 == 0).all()):
                continue
            try:
                frame[name] = series.astype("Int64")
            except (TypeError, ValueError):
                continue
        return frame

    def _describe(self, frame, source: str = "") -> str:
        """The one-line summary over the table.

        Counts what did not work OUT LOUD, rather than leaving it to be
        noticed: a sweep whose trials mostly failed still writes a
        full-looking table, and so does a session of runs that all crashed.

        AND IT NAMES THE LOADED RUN. The mark is in a column that may be
        scrolled past or sorted away; the sentence over the table is where a
        user looks to find out which run everything else on the screen is
        describing.
        """
        note = f"{len(frame)} runs"
        if "source" in frame.columns:
            mine = int((frame["source"].astype(str) != SOURCE_SWEEP).sum())
            if mine and mine != len(frame):
                note += f" — {mine} from this session, " \
                        f"{len(frame) - mine} from the sweep"
        if "status" in frame.columns:
            status = frame["status"].astype(str)
            failed = int(((status != "ok") & (status != STATUS_RUNNING)).sum())
            running = int((status == STATUS_RUNNING).sum())
            if failed:
                note += f", {failed} of which did not produce a regression"
            if running:
                note += f", {running} still going"
        record = self.loaded_run()
        if record is not None:
            name = record.get("run")
            note += (f". Loaded: {name}" if isinstance(name, str) and name
                     else ". One run is loaded")
        elif len(frame) > 1:
            # SAID, rather than left as an empty column. Several runs and no
            # choice made is a state the user has to resolve, and a blank
            # column is indistinguishable from a feature that is not working.
            note += ". No run is loaded — pick one to show it everywhere else"
        return f"{note}. {source}" if source else note

    # ------------------------------------------------------------- selection

    def selected_trial(self) -> Optional[dict]:
        """The selected row as a dict, or ``None``.

        Read back through the FRAME, not off the table's cells: the table
        holds display strings, and a trial re-run from `"0.05"` instead of
        0.05 is not the trial that was recorded.
        """
        if self._frame is None:
            return None
        items = self.table.table.selectedItems()
        if not items:
            return None
        index = items[0].data(0x0100)          # Qt.UserRole: the frame row
        if index is None or not 0 <= int(index) < len(self._frame):
            return None
        return self._frame.iloc[int(index)].to_dict()

    # -------------------------------------------------------------- delete
    #
    # Instruction 146, requested 2026-08-18: "the user should be able to
    # delete runs from the figures (currently possible) and from the run tab
    # (not possible)".
    #
    # TWO DIFFERENT THINGS A USER COULD MEAN, and a single "Delete" that does
    # not distinguish them is how a screen's results are lost. Both are
    # legitimate; the DEFAULT gesture is the safe one.

    def selected_runs(self) -> list:
        """Every selected row as a dict, in the order the table shows them.

        Read back through the FRAME for the same reason
        :meth:`selected_trial` is: the table holds display strings, and a
        folder path is not something to reconstruct from one.
        """
        if self._frame is None:
            return []
        rows, seen = [], set()
        for item in self.table.table.selectedItems():
            index = item.data(0x0100)              # Qt.UserRole: frame row
            if index is None:
                continue
            index = int(index)
            if index in seen or not 0 <= index < len(self._frame):
                continue
            seen.add(index)
            rows.append(self._frame.iloc[index].to_dict())
        return rows

    @staticmethod
    def _is_running(record) -> bool:
        """Whether this row is a run that has not come back yet."""
        return str((record or {}).get("status", "")).strip() == STATUS_RUNNING

    @staticmethod
    def describe_folder(folder: str) -> str:
        """What is in a run folder, in the words a decision needs.

        "12 figures, 4 CSVs, 31 MB". A user deciding whether to destroy an
        overnight fit needs to see WHAT they are destroying, and a folder
        path alone is not that.

        WORKER ONLY -- it walks the whole run folder and stats every file in
        it. Never call it from menu-build or paint code.
        """
        figures = tables = other = 0
        total = 0
        if not folder or not os.path.isdir(folder):
            return "nothing on disk"
        for root, _dirs, names in os.walk(folder):
            for name in names:
                path = os.path.join(root, name)
                try:
                    total += os.path.getsize(path)
                except OSError:
                    continue
                suffix = os.path.splitext(name)[1].lower()
                if suffix in (".png", ".pdf", ".svg", ".jpg", ".jpeg"):
                    figures += 1
                elif suffix in (".csv", ".tsv"):
                    tables += 1
                else:
                    other += 1
        parts = []
        if figures:
            parts.append(f"{figures} figure" + ("s" if figures != 1 else ""))
        if tables:
            parts.append(f"{tables} CSV" + ("s" if tables != 1 else ""))
        if other:
            parts.append(f"{other} other file" + ("s" if other != 1 else ""))
        if not parts:
            return "an empty folder"
        return ", ".join(parts) + f", {_readable_size(total)}"

    def remove_runs(self, records) -> int:
        """Take rows off the table. THE FOLDERS ON DISK ARE UNTOUCHED.

        The safe half, and the default gesture. It needs no confirmation
        because it is recoverable -- :meth:`reload` reads the folder again --
        and the status line says exactly that rather than leaving the user to
        discover it.

        :returns: how many rows left the table.
        """
        wanted = {self._row_key(record) for record in (records or [])
                  if self._row_key(record)}
        if not wanted:
            return 0
        gone = [record for record in (records or [])
                if self._row_key(record) in wanted]

        for handle, row in list(self._recorded.items()):
            if self._row_key(row) in wanted:
                del self._recorded[handle]
        if self._sweep_frame is not None and len(self._sweep_frame):
            # BY TRIAL NUMBER AS WELL AS BY KEY, because a sweep trial's
            # NAME is derived during the rebuild ("trial 2" from `trial_id`)
            # and its row in the raw sweep frame has neither a `run` column
            # nor a `folder` -- so `_row_key` answers "" for every one of
            # them and a match on the key alone removed nothing at all. That
            # is the whole of what "reload brings it back" was promising
            # about rows that had never left.
            trials = {str(record.get("trial_id"))
                      for record in (records or [])
                      if record.get("trial_id") is not None}
            keep = []
            for _index, row in self._sweep_frame.iterrows():
                record = row.to_dict()
                keep.append(self._row_key(record) not in wanted
                            and str(record.get("trial_id")) not in trials)
            self._sweep_frame = self._sweep_frame.loc[keep].copy()
            if not len(self._sweep_frame):
                self._sweep_frame = None

        before = self._loaded_key
        if self._loaded_key in wanted:
            # The mark cannot stay on a run that is no longer a row: a key
            # naming nothing makes `loaded_run` answer None while the column
            # shows a tick, and the two disagreeing is how a stale mark
            # survives (see `_settle_the_loaded_run`).
            self._loaded_key = ""
        count = len(wanted)
        self._rebuild(f"Removed {count} run" + ("s" if count != 1 else "")
                      + " from the list; Reload brings "
                      + ("them" if count != 1 else "it") + " back.",
                      since=before)
        # AFTER the rebuild, so a listener that re-points the results panel
        # is looking at the table as it now is.
        self.runs_removed.emit([dict(record) for record in gone])
        return count

    def delete_runs_from_disk(self, records, confirm=None) -> int:
        """Delete the run FOLDERS, then take the rows off the table.

        NOT RECOVERABLE, so it takes a confirmation naming the folder and
        saying what is in it, and there is no undo offered -- an undo that
        cannot honour itself is worse than none.

        :param records: run-row mappings whose distinct, existing ``folder``
            paths are candidates for deletion. A running record refuses the
            whole operation.
        :param confirm: called with the message and the list of folders;
            returns True to go ahead. Defaults to a modal question. Injected
            rather than assumed so a headless test can drive the real method
            instead of a copy of it.
        :returns: how many folders were deleted, or :data:`_DELETION_STARTED`
            when the work went to a worker and the count is not knowable yet.
            Nothing the user sees changes -- the same question is asked and
            the same sentence written, a moment later.
        """
        records = [record for record in (records or []) if record]
        refused = [record for record in records if self._is_running(record)]
        if refused:
            # NEVER DELETE WHAT IS RUNNING, and say why rather than ignoring
            # the gesture (instruction 106).
            self._say(self._why_it_cannot_be_deleted(refused))
            return 0
        folders = []
        for record in records:
            folder = record.get("folder")
            if isinstance(folder, str) and folder.strip():
                folders.append(os.path.abspath(os.path.expanduser(
                    folder.strip())))
        folders = list(dict.fromkeys(folders))
        if not folders:
            self._say("Nothing to delete: these runs have no folder on disk.")
            return 0

        # NEITHER HALF OF THE QUESTION CAN BE ASKED HERE. Which of these
        # folders is still on disk is a stat each, and what is in one is an
        # `os.walk` of a whole run -- and the answer is needed BEFORE the
        # modal, because the modal is what says what is about to be
        # destroyed. So the description goes to a worker and the confirmation
        # is shown from its callback.
        self._deleted_count = 0
        self._start_waiting("Working out what these runs hold…")
        started = self._jobs.submit(
            lambda targets=list(folders): _what_would_be_deleted(targets),
            lambda described, rows=list(records):
                self._ask_then_delete(described, rows, confirm))
        if self._threaded:
            return _DELETION_STARTED if started else 0
        return self._deleted_count

    def _ask_then_delete(self, described, records, confirm=None) -> bool:
        """Show the confirmation, then delete on a worker. On the GUI thread.

        The only half of a delete that touches a widget: it composes the
        message out of what the worker found and puts the modal up.
        """
        folders = [folder for folder, _what in described or []]
        if not folders:
            self._stop_waiting()
            self._say("Nothing to delete: these runs have no folder on disk.")
            return False
        lines = [f"{folder} — {what}" for folder, what in described]
        message = ("Delete " + ("this run" if len(folders) == 1
                                else f"these {len(folders)} runs")
                   + " from disk? This cannot be undone.\n\n"
                   + "\n".join(lines))
        ask = confirm if callable(confirm) else self._confirm_deletion
        if not ask(message, list(folders)):
            # NO IS AN ANSWER, and the line goes back to what it said before
            # the delete was asked for. Saying nothing was already the
            # behaviour; the placeholder is the only thing that has to be
            # taken back down, and it was taken down above.
            return False
        self._start_waiting("Deleting…")
        return bool(self._jobs.submit(
            lambda targets=list(folders): _delete_folders(targets),
            lambda outcome, rows=records:
                self._deletion_finished(outcome, rows)))

    def _deletion_finished(self, outcome, records) -> None:
        """Take the rows off the table and say what went. On the GUI thread."""
        deleted, failed = outcome if outcome else ([], [])
        keep = {folder for folder in failed}
        gone = [record for record in records
                if os.path.abspath(os.path.expanduser(
                    str(record.get("folder") or ""))) not in keep]
        self.remove_runs(gone)
        for folder in deleted:
            # The caches answer "it is there" until told otherwise, and this
            # is the moment they are wrong.
            path_probe.forget(folder)
            self._workspace_answers.pop(folder, None)
            self._workspace_pending.discard(folder)
        note = (f"Deleted {len(deleted)} run folder"
                + ("s" if len(deleted) != 1 else "") + " from disk.")
        if failed:
            note += " Could not delete " + "; ".join(failed) + "."
        self._say(note)
        self._deleted_count = len(deleted)

    def _confirm_deletion(self, message: str, folders) -> bool:
        """The modal question. Defaults to No: this one cannot be undone."""
        from PySide6.QtWidgets import QMessageBox

        answer = QMessageBox.question(
            self, "Delete runs from disk", message,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return answer == QMessageBox.Yes

    @staticmethod
    def _why_it_cannot_be_deleted(running) -> str:
        """The reason a running run is refused, naming it."""
        names = ", ".join(str(record.get("run") or "a run")
                          for record in running)
        return (f"{names} is still going and cannot be deleted. Stop it "
                f"first — a folder deleted underneath a run that is still "
                f"writing leaves half a result and no way to tell.")

    def _say(self, note: str) -> None:
        """Put a sentence on the status line without refilling the table."""
        self._source_note = str(note or "")
        if self._frame is None:
            self._status.setText(self._source_note or "Nothing run yet.")
        else:
            self._status.setText(self._describe(self._frame,
                                                self._source_note))

    # --------------------------------------------------- the waiting line

    def _start_waiting(self, note: str) -> None:
        """Say what is being waited for, and remember what that replaced.

        Only threaded: unthreaded the answer is already here by the time this
        would be read, and a placeholder that is overwritten in the same call
        is a line the user never sees.
        """
        if not self._threaded:
            return
        self._note_before_waiting = self._source_note
        self._waiting_note = str(note or "")
        self._say(self._waiting_note)

    def _stop_waiting(self) -> None:
        """The wait has been answered; the answer writes its own sentence."""
        self._waiting_note = ""
        self._note_before_waiting = ""

    def _abandon_waiting(self) -> None:
        """Nothing came of it -- put back the sentence the placeholder hid.

        SAYING NOTHING IS THE RIGHT ANSWER HERE, and it is not the same as
        leaving the placeholder up. Declining the delete confirmation used to
        leave the status line exactly as it was; without this it would leave
        "Working out what these runs hold…" on screen for good.
        """
        if not self._waiting_note:
            return
        previous = self._note_before_waiting
        self._stop_waiting()
        self._say(previous)

    def _on_job_failed(self, message) -> None:
        """A worker raised. On the GUI thread, from either runner.

        `JobRunner._on_settled` calls ``on_done`` only for a job that came
        back cleanly, so every arrival handler in this file is one that may
        never run. This is what is left holding the placeholder it would have
        replaced and the button its click disabled.

        NOT GENERATION-GUARDED, deliberately: `job_failed` carries no job id,
        so it cannot be, and everything done here is safe to do twice or for
        a job whose result nobody wanted any more. Enabling an enabled button
        changes nothing, and the line is only rewritten while a placeholder
        is actually outstanding -- a stale failure landing after a newer
        placeholder went up replaces that one, which is the correct order.

        Guarded for ``RuntimeError`` because a worker parked by
        `bridge.drain_thread` outlives the widget it belonged to, and by the
        time it fails this panel's C++ half may be gone.
        """
        try:
            # THE BUTTON FIRST. It is the one thing here that a user cannot
            # work around, and the failure that disabled it is exactly the
            # case `_run_arrived` does not cover.
            self._open.setEnabled(True)
            if self._waiting_note:
                self._stop_waiting()
                self._say("That did not finish: "
                          + (str(message).strip() or "unknown error"))
        except RuntimeError:
            LOG.debug("the runs tab has gone; a job failure has nowhere to "
                      "go", exc_info=True)

    def _workspace_answer(self, folder: str) -> bool:
        """Whether this run has a bundle, without stat-ing on this thread.

        THE MENU CANNOT WAIT. `_build_run_menu` runs inside the right-click
        that opens the menu, and `workspace.has_workspace` is a stat on a
        folder the user chose -- twenty seconds of nothing, on the mount this
        module's docstring is about, before the menu appears.

        Answered from what a worker has already found out, and OPTIMISTICALLY
        while nothing is known: for the same reason :mod:`spacr.qt.path_probe`
        gives, an entry offered and then quietly correct is better than one
        withdrawn on the strength of a slow mount. The restore handler has to
        cope with a bundle it cannot read in any case.
        """
        key = str(folder or "")
        if not key:
            return False
        if key in self._workspace_answers:
            return self._workspace_answers[key]
        if key not in self._workspace_pending:
            self._workspace_pending.add(key)
            self._jobs.submit(
                lambda target=key: _has_workspace(target),
                lambda answer, target=key:
                    self._remember_workspace(target, answer))
        # Unthreaded the job above has already answered, so this reads the
        # truth rather than the optimism.
        return self._workspace_answers.get(key, True)

    def _remember_workspace(self, folder: str, answer) -> None:
        """File what the worker found out about a run's bundle."""
        self._workspace_pending.discard(folder)
        self._workspace_answers[folder] = bool(answer)

    def _build_run_menu(self, records):
        """The context menu for these rows: remove first, delete second.

        THE FIRST ENTRY IS THE SAFE ONE. A user reaching for a menu takes
        what is at the top, and what is at the top must not be the
        irreversible one.

        BUILT APART FROM BEING SHOWN so a test can read the entries without
        entering a modal loop. `QMenu.exec` is a C++ event loop and it cannot
        be monkeypatched off a PySide type -- tried, and the test hung inside
        it -- so the seam has to be here rather than in the test.

        Each action carries its verb in `setData`, so dispatch does not
        depend on identity or on the order the entries were added.
        """
        from PySide6.QtWidgets import QMenu

        count = len(records)
        plural = "s" if count != 1 else ""
        menu = QMenu(self)
        # LOAD FIRST, because it is what a user opens this menu for. The menu
        # offered Remove, Open beside and Delete and no way to LOAD -- so the
        # only route to a different run was a single click, and when that was
        # refused there was no second route at all.
        load = None
        if count == 1:
            load = menu.addAction("Load this run")
            load.setData("load")
            load.setToolTip("Show this run's results, figures and summary.")
            menu.addSeparator()
        # SAVE, FOR ONE OR FOR SEVERAL. Restore below is single-run because
        # two workspaces cannot both be put on screen; SAVING several is a
        # different thing and is what was asked for.
        keep = menu.addAction(
            f"Save the state of {count} run{plural}"
            if count != 1 else "Save this run's state")
        keep.setData("save_state")
        keep.setEnabled(any(str(record.get("folder") or "").strip()
                            for record in records))

        if count == 1:
            folder = str(records[0].get("folder") or "")
            restore = menu.addAction("Restore what this run had open…")
            restore.setData("restore")
            if self._workspace_answer(folder):
                restore.setToolTip(
                    "Re-attach the databases, put the montage back on its "
                    "coefficient, and restore the view built on every figure. "
                    "Says what it put back and what it could not.")
            else:
                # Instruction 106: OFFERED AND DISABLED, saying why. An entry
                # that appeared only for runs that happen to have a bundle is
                # one nobody learns exists.
                restore.setEnabled(False)
                restore.setToolTip(
                    "This run saved no workspace — it was run with 'Saved "
                    "runs carry: Nothing', or before spaCR recorded one. "
                    "Preferences ▸ Performance ▸ Saved runs carry.")
            menu.addSeparator()
        remove = menu.addAction(f"Remove {count} run{plural} from the list")
        remove.setData("remove")
        remove.setToolTip("The folders on disk are untouched. Reload brings "
                          "them back.")
        if count == 1 and _is_ok(records[0]):
            beside = menu.addAction("Open beside the loaded run")
            beside.setData("beside")
            beside.setToolTip(
                "Show this run's own volcano next to the loaded run's, each "
                "answering its own hover and click. Two runs can be live at "
                "once.")
            menu.addSeparator()
        delete = menu.addAction(f"Delete {count} run{plural} from disk…")
        delete.setData("delete")
        delete.setToolTip("Deletes the run folder and everything in it. This "
                          "cannot be undone; you are shown the path first.")
        running = [record for record in records if self._is_running(record)]
        if running:
            # GREYED OUT AND SAYING WHY (instruction 106), not silently
            # ignored. Both entries: removing the row of a run this session
            # is still updating would leave `update_run` writing to a handle
            # with nothing to show for it.
            why = self._why_it_cannot_be_deleted(running)
            # LOAD IS GREYED TOO, and for its own reason rather than the
            # delete reason: a run still going has produced no results table,
            # no figures and no summary, so loading it would put an empty
            # screen under a mark claiming a run. That is worse than a
            # disabled entry, which at least says why.
            if load is not None:
                load.setEnabled(False)
                load.setToolTip(
                    "This run is still going, so it has no results to show "
                    "yet. It becomes the loaded run when it finishes.")
            for action in (remove, delete):
                action.setEnabled(False)
                action.setToolTip(why)
            menu.addSeparator()
            menu.addAction(why).setEnabled(False)
        return menu

    def _on_double_click(self, index=None) -> None:
        """Load the double-clicked run, whatever state the mark is in."""
        record = self.selected_trial()
        if record is None:
            return
        self.load_this_run(record)

    def load_this_run(self, record) -> bool:
        """Load a run and announce it even when it is already selected.

        Parameters
        ----------
        record : mapping
            Run record containing a resolvable row key.

        Returns
        -------
        bool
            ``True`` when the record was accepted and announced.

        Notes
        -----
        Unlike :meth:`set_loaded_run`, this method deliberately reloads the
        current selection. This lets an explicit user action resynchronize
        the run list and results view.
        """
        if not isinstance(record, dict):
            return False
        key = self._row_key(record)
        if not key:
            return False
        before = self._loaded_key
        self._loaded_key = key
        self._source_note = ""
        self._paint_the_loaded_mark()
        if not self._announce_the_loaded_run(before):
            # Nothing listened to the loaded-run signal; the results panel
            # still has to be told, and `trial_activated` is the other door.
            self.trial_activated.emit(dict(record))
        return True

    def _run_menu(self, position) -> None:
        """Show the row menu where the user right-clicked, and act on it."""
        records = self.selected_runs()
        if not records:
            item = self.table.table.itemAt(position)
            if item is None:
                return
            self.table.table.selectRow(item.row())
            records = self.selected_runs()
        if not records:
            return
        menu = self._build_run_menu(records)
        chosen = menu.exec(self.table.table.viewport().mapToGlobal(position))
        self._apply_run_menu(chosen.data() if chosen is not None else "",
                             records)

    def _apply_run_menu(self, verb, records) -> bool:
        """Do what one menu entry says. The seam a test can drive.

        Apart from :meth:`_run_menu` because that method ends in
        `QMenu.exec`, a C++ event loop -- so a test that wants to know what
        an entry DOES either enters it and hangs, or re-implements the
        dispatch and tests its own copy.
        """
        if verb == "save_state" and records:
            folders = [str(record.get("folder") or "").strip()
                       for record in records]
            folders = [folder for folder in folders if folder]
            self._saved_state = False
            if not folders:
                # Nothing to hand the writer, and the note still has to be
                # written -- a menu entry that does nothing and says nothing
                # is the failure instruction 106 is about.
                self._on_states_saved(([], []))
                return False
            # THE WRITE GOES TO A WORKER. `save_run_states` stats each folder
            # and then writes a bundle into it, both on paths the user chose.
            if self._threaded:
                plural = "" if len(folders) == 1 else "s"
                self._say(f"Saving the state of {len(folders)} run{plural}…")
            started = self._jobs.submit(
                lambda targets=list(folders): save_run_states(targets),
                self._on_states_saved)
            return bool(started) if self._threaded else self._saved_state

        if verb == "load" and records:
            # The menu greys it, and so does this: a menu is one door and
            # `_apply_run_menu` is the seam tests drive, so a guard on the
            # paint alone would be a guard a test could walk straight past.
            if self._is_running(records[0]):
                return False
            return self.load_this_run(records[0])
        if verb == "remove":
            return bool(self.remove_runs(records))
        if verb == "beside" and records:
            self.compare_requested.emit(dict(records[0]))
            return True
        if verb == "restore" and records:
            # THE SAME NON-BLOCKING ANSWER THE MENU WAS DRAWN FROM, rather
            # than a blocking check "to be sure": the menu has already been
            # built and shown from it, so a stat here would only add the
            # freeze back at the click. `AppScreen.restore_run_workspace` is
            # where a bundle that turns out not to be readable is reported.
            if not self._workspace_answer(str(records[0].get("folder") or "")):
                return False
            self.workspace_restore_requested.emit(dict(records[0]))
            return True
        if verb == "delete":
            return bool(self.delete_runs_from_disk(records))
        return False

    def _on_states_saved(self, result) -> None:
        """Say what the save did. On the GUI thread, from `_apply_run_menu`.

        SAVING IS NOT CHOOSING. The loaded mark stays where it was: keeping a
        run for later is not the same as switching to it, and moving the mark
        would drag every view with it.
        """
        saved, failures = result if result else ([], [])
        self._saved_state = bool(saved)
        for folder in saved:
            # A save is the moment a cached "no bundle" answer goes stale.
            self._workspace_answers.pop(str(folder), None)
            self._workspace_pending.discard(str(folder))
        self._source_note = describe_saved_states(saved, failures)
        self._rebuild(self._source_note)

    def eventFilter(self, watched, event):                    # noqa: N802
        """Delete on the selection removes it FROM THE LIST.

        The safe half on the bare key, per the design. Deleting from
        disk is a separate, explicitly-worded choice and is not something a
        keystroke can reach.
        """
        if (watched is self.table.table
                and event.type() == QEvent.KeyPress
                and event.key() in (Qt.Key_Delete, Qt.Key_Backspace)):
            records = self.selected_runs()
            running = [record for record in records
                       if self._is_running(record)]
            if running:
                self._say(self._why_it_cannot_be_deleted(running))
                return True
            if records:
                self.remove_runs(records)
                return True
        return super().eventFilter(watched, event)

    # ------------------------------------------------------------- selection

    def set_photo_provider(self, provider) -> None:
        """Tell this panel where a run's still comes from.

        :param provider: ``folder -> QPixmap or None``.
        """
        self._photo_provider = provider
        self._show_photograph(self.selected_trial())

    def photograph_shown(self):
        """The still currently painted under the table, or ``None``.

        Public because "is the still on screen" is what a caller and a test
        both want, and reading `isVisible()` off a widget that has never been
        shown answers a different question.
        """
        return self._photo.pixmap() if not self._photo.isHidden() else None

    def _show_photograph(self, record) -> bool:
        """Paint the selected run's still, or take the frame away."""
        photo = None
        folder = str((record or {}).get("folder") or "") if isinstance(
            record, dict) else ""
        if folder and callable(self._photo_provider):
            try:
                photo = self._photo_provider(folder)
            except Exception:                                # noqa: BLE001
                LOG.debug("could not reach the run photograph", exc_info=True)
        if photo is None or photo.isNull():
            self._photo.clear()
            self._photo.hide()
            return False
        width = max(160, min(self.width() - 12, photo.width()))
        self._photo.setPixmap(photo.scaledToWidth(width,
                                                  Qt.SmoothTransformation))
        self._photo.show()
        return True

    def _on_selection(self) -> None:
        self._show_photograph(self.selected_trial())
        if self._rebuilding:
            # The re-select at the end of `_rebuild` is this panel putting
            # the highlight back, not the user choosing a run.
            return
        record = self.selected_trial()
        if record is None:
            return
        # PICKING A RUN IS NOT LOADING IT (190). Reported 2026-08-20: "for
        # some reason clicking once on a run shows the results. double click
        # should loade the results".
        #
        # This used to load on selection, which meant ARROWING DOWN A LIST OF
        # FIVE RUNS LOADED FIVE RUNS -- five multi-second reads nobody asked
        # for, to look at five names. Selection now does what selection does:
        # it shows this run's photograph and its detail, and nothing else.
        # `_load_selected` on double-click is the gesture that costs time,
        # and it was already wired.
        #
        # THE FAILURE MESSAGE STILL BELONGS TO SELECTION, though. A trial that
        # failed or is still going has no results to show ever, and saying so
        # when it is picked is the difference between a table that ignores
        # clicks and one that explains them -- it costs nothing to say.
        if not _is_ok(record):
            self.trial_activated.emit(record)
