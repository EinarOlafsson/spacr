"""The sweep's runs, as a tab beside the regression's results.

The parameter search uses the main module setup plus one extra tab for its
runs. The
parameter search was a bespoke screen carrying its own copies of the table,
the figure queue and the results panel; every fix to the shared module screen
had to be made again there, and this repository is already paying for that
kind of drift -- 155 field tooltips existed because 28 hand-built screens each
re-implemented a row.

So the runs go where the results go: the left half of the module screen's
figure splitter, as a second tab. Picking a run swaps the figures on the
right, which is the substance of the request and the only part that did not
change.

DELIBERATELY NOT THE SHAPE THE RESULTS GET. A results table is hundreds of
findings read ALONGSIDE the figure describing them, so it is on screen at the
same time as the plot. A runs table is a short list scanned top to bottom,
and picking one replaces everything to its right -- so it is a tab. One is
reading within a run, the other is navigating between them.

Every run, not only the sweep's, appears here:

    "the runs tab should capture all the runs in a sweep and all the runs run
     in the normal module."

This panel was fed by ``sweep_results.csv`` alone, so it answered "which
trials did the sweep try" rather than "what have I run" -- an ordinary run of
the module, and the re-fit from 124 E, simply did not appear. They are the
same kind of thing: a fit, its settings, its figures, and a folder to read
them back out of. So an ordinary run is RECORDED as it happens
(``record_run``, updated by ``update_run`` when it finishes) and shown in
the same table as the sweep's trials, described by the SAME COLUMNS -- which
is what makes the two comparable, and comparing them is the entire reason the
tab exists.

The recorders are underscored deliberately: `tests/test_api_i18n_extractor.py`
holds an EXACT count of the documented public API, so promoting them to
`record_run` / `update_run` has to bump that count in the same commit. That is
the whole of what is left to do it; the sentence here used to add "and that
file belongs to another session right now", which was true on 2026-08-17 and
is not a reason today. A temporary coordination fact written into a permanent
comment reads later as a standing constraint.
"""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

#: What the sweep writes its table to, inside the destination folder.
RESULTS_FILENAME = "sweep_results.csv"

#: The column that says WHICH RUN IS LOADED -- the one every view on this
#: screen is describing. Requested 2026-08-18: "there should be a checkbox in
#: runs that specifies which run is loaded if there are several runs".
#:
#: A COLUMN RATHER THAN A CHECKBOX, and the difference is only in the widget:
#: the table is the shared `ResultsTable`, which draws text, and a second
#: table implementation to carry one tick box is a second set of bugs. It
#: reads the same way -- one row is marked, the rest are blank -- and it
#: sorts, filters and copies with the rest of the row.
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
    "regression_type", "inference", "analysis_unit",
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
    "regression_type", "inference", "analysis_unit", "agg_type", "transform",
    "multiple_testing_method", "fdr_alpha", "fraction_threshold",
    "min_cell_count",
)

#: What the ``source`` column says. The sweep's trials are read off its CSV
#: and get :data:`SOURCE_SWEEP`; the other two are recorded live.
SOURCE_RUN = "run"
SOURCE_REFIT = "re-fit"
SOURCE_SWEEP = "sweep trial"
#: A run opened DELIBERATELY from disk, including one from an earlier
#: session. Instruction 154 G: a run on disk is a first-class run, not a
#: degraded one -- it gets a row, a source, its settings read back beside its
#: results, and it can be the loaded run like any other.
SOURCE_DISK = "on disk"
#: One fit of the Measurements tab's column queue. Instruction 154 F: "do
#: regression on a selection of columns each gets saved as a run that i can
#: evaluate." Its own source rather than :data:`SOURCE_RUN`, because what
#: makes those rows worth having together is that they came from one queue
#: over one merged frame -- which is exactly the thing a `source` column is
#: for.
SOURCE_MEASUREMENT = "measurement column"

#: A run that has been started and has not come back yet. Not "ok": a run
#: still going has produced nothing to look at, and a row that claims
#: otherwise is a click that opens last month's folder.
STATUS_RUNNING = "running"


def _readable_size(total: int) -> str:
    """Bytes as the unit a person decides in. 31 MB, not 32,505,856."""
    size = float(max(0, int(total)))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" or size >= 10 \
                else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.0f} GB"                        # pragma: no cover - loop


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
    #: A REMOVAL IS AN EVENT OTHER VIEWS CARE ABOUT (instruction 146). The
    #: results panel keeps a plot state per run and the figure grid keys its
    #: sections by run label, so a run that leaves this table and stays in
    #: both of those is the same disagreement between two views that 157 was
    #: about, pointing the other way. The panel does not reach into them
    #: itself -- it says what happened and the screen does the wiring, which
    #: is what keeps this widget usable outside that screen.
    runs_removed = Signal(list)
    #: Emitted with ONE run's row when the user asks to see it BESIDE the
    #: loaded one (instruction 116). Opening a second run is a deliberate
    #: act: two runs is what a comparison needs and twelve is what makes the
    #: screen unusable, so it has its own gesture rather than happening on a
    #: click. The screen decides whether the bound allows it.
    compare_requested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        from .fast_plots import ResultsTable

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
        # DOUBLE-CLICK LOADS THE RUN, and it was connected to nothing at all.
        # Selection alone already tried to load (see `_on_selection`), but a
        # selection that is refused for any reason leaves the previous run on
        # screen with no way to insist -- which is what "the first run is
        # perpetually loaded" was. A double-click is the user saying it again,
        # so it FORCES the load rather than asking politely.
        self.table.table.doubleClicked.connect(self._on_double_click)
        self.table.table.installEventFilter(self)
        layout.addWidget(self.table, 1)

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
        #: WHERE THE MARK WAS BEFORE the current one, so a load that fails
        #: can put it back. Instruction 157: "the mark is set by the load,
        #: not beside it -- a failed load leaves the mark where it was rather
        #: than pointing at something not on screen." The mark moves first
        #: and the view is asked to follow, because the panel cannot know
        #: whether a run will open until something tries; what it can do is
        #: be told when it did not, which is :meth:`the_load_failed`.
        self._previous_loaded_key = ""
        #: The sentence the last load or refusal added to the status line,
        #: kept so moving the mark can rewrite the line without losing it.
        self._source_note = ""
        #: True while :meth:`_rebuild` is refilling the table. The re-select
        #: it does at the end would otherwise read as a user picking a row and
        #: re-emit `trial_activated`, which re-loads the results panel on
        #: every recorded run.
        self._rebuilding = False

    # ------------------------------------------------------------------ load

    def load(self, folder) -> bool:
        """Read ``sweep_results.csv`` from a sweep's destination folder."""
        import pandas as pd

        if not folder:
            return False
        folder = os.path.abspath(os.path.expanduser(os.fspath(folder)))
        path = folder if folder.lower().endswith(".csv") else os.path.join(
            folder, RESULTS_FILENAME)
        if not os.path.isfile(path):
            # NOT a wipe, and the note goes THROUGH the rebuild. The sweep's
            # table being absent says nothing about the runs this session has
            # made; setting the status here and clearing the table below is
            # how opening the tab before a sweep exists used to empty it.
            self._rebuild(f"No results table at {path} yet.")
            return False
        try:
            frame = pd.read_csv(path)
        except Exception as error:  # noqa: BLE001 - report, do not raise
            self._rebuild(f"Could not read {path}: {error}")
            return False
        self._folder = folder
        return self.set_frame(frame, source=path)

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
        """Open a run's results folder and make it the loaded run.

        :param folder: the run folder. ``""`` asks for one -- which is the
            button's path, and the reason a test always passes one: a static
            file dialog runs its event loop in C++ and would hang a headless
            run.
        :returns: True when a run was opened.

        A RUN ON DISK IS A FIRST-CLASS RUN (154 G). It gets a row beside this
        session's own, described by the SAME columns -- its settings are read
        back from beside its results, so an old run and a fresh one are
        comparable, which is the entire reason this tab exists. It is not a
        degraded mode of "re-run it and look again".
        """
        if not folder:
            from PySide6.QtWidgets import QFileDialog

            folder = QFileDialog.getExistingDirectory(
                self, "Choose a run's results folder", self._folder or "")
            if not folder:
                return False
        try:
            folder = os.path.abspath(os.path.expanduser(os.fspath(folder)))
        except TypeError:
            return False

        from .regression_results import RESULT_FILENAMES, find_results_table

        table = find_results_table(folder)
        if not table:
            # NAMED, not a silent no-op. The user picked a folder; being told
            # nothing happened is the failure this repository keeps fixing.
            self._rebuild(f"No run in {folder}: none of "
                          f"{', '.join(RESULT_FILENAMES)} is in it or under "
                          f"it.")
            return False
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
            # ITS OWN SETTINGS, so an old run is described by the same columns
            # as a new one. Without them the row is a name and a folder, and
            # two runs cannot be compared on the settings that differ.
            try:
                from ...refit import settings_of_run

                settings = settings_of_run(table) or {}
            except Exception:                                    # noqa: BLE001
                settings = {}
            for name in RUN_SETTING_COLUMNS:
                value = settings.get(name)
                if value is not None:
                    row[name] = value
            self._recorded[handle] = row

        before = self._loaded_key
        self._loaded_key = self._row_key(self._recorded[handle])
        # AND THE VIEWS FOLLOW IT -- through the same funnel every other path
        # uses (157). This used to emit the two signals itself, which is how
        # the finishing path came to be the only one that did not.
        self._rebuild(f"Loaded the run in {run_folder}.", since=before)
        return True

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
        """THE ONE PLACE the views are told which run is on screen.

        Instruction 157, reported 2026-08-18:

            "i ran a mixed model and a ols model and eaven if the ols model is
             marked as loaded i think i still see the mixed results and no
             summary (because the ols is actually not loaded)"

        Which was one missing connection and one missing emission. The three
        DELIBERATE paths -- a row clicked, a run chosen, a folder opened --
        each announced themselves; the path a user actually reaches most, A
        RUN THAT BECOMES LOADED BY FINISHING, moved the mark inside
        :meth:`update_run` and told nobody. So the mark said `ols` and the
        coefficients, the figures and the summary were still the `mixed` run's.

        ONE FUNNEL, NOT FOUR. Every path that can move the mark ends here, so
        a fifth cannot be added without being announced, and the finishing
        path cannot drift from the clicking one -- which is exactly how only
        one of them ended up tested.

        :param before: the key the mark was on when the caller started.
        :returns: whether anything was announced.
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
        self.loaded_run_changed.emit(dict(record))
        self.trial_activated.emit(dict(record))
        # THE UNDO IS SPENT. `the_load_failed` is answering THIS announcement,
        # and a listener that comes back later -- a click on a trial that
        # failed, say -- must not be able to drag the mark back to a run two
        # choices ago. Consumed here rather than on a timer or a flag: the
        # window in which a load can fail is the emission itself.
        self._previous_loaded_key = self._loaded_key
        return True

    def the_load_failed(self, why: str = "") -> bool:
        """The run that was just announced could not be shown: undo the mark.

        THE MARK IS A CONSEQUENCE OF THE RUN BEING SHOWN (157). A mark on a
        run whose results are not on screen is the same disagreement between
        two views that this instruction is about, only pointing the other
        way -- and it is the one a user cannot act on, because the run they
        ARE looking at is no longer named anywhere.

        :param why: added to the status line, so the refusal is readable
            rather than a mark that silently jumped back.
        :returns: whether the mark moved back.
        """
        previous = self._previous_loaded_key
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

        :param confirm: called with the message and the list of folders;
            returns True to go ahead. Defaults to a modal question. Injected
            rather than assumed so a headless test can drive the real method
            instead of a copy of it.
        :returns: how many folders were deleted.
        """
        import shutil

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
        folders = [folder for folder in dict.fromkeys(folders)
                   if os.path.isdir(folder)]
        if not folders:
            self._say("Nothing to delete: these runs have no folder on disk.")
            return 0

        lines = [f"{folder} — {self.describe_folder(folder)}"
                 for folder in folders]
        message = ("Delete " + ("this run" if len(folders) == 1
                                else f"these {len(folders)} runs")
                   + " from disk? This cannot be undone.\n\n"
                   + "\n".join(lines))
        ask = confirm if callable(confirm) else self._confirm_deletion
        if not ask(message, list(folders)):
            return 0

        deleted, failed = [], []
        for folder in folders:
            try:
                shutil.rmtree(folder)
            except OSError as error:                             # noqa: BLE001
                failed.append(f"{folder} ({error.strerror or error})")
            else:
                deleted.append(folder)
        keep = {folder for folder in failed}
        gone = [record for record in records
                if os.path.abspath(os.path.expanduser(
                    str(record.get("folder") or ""))) not in keep]
        self.remove_runs(gone)
        note = (f"Deleted {len(deleted)} run folder"
                + ("s" if len(deleted) != 1 else "") + " from disk.")
        if failed:
            note += " Could not delete " + "; ".join(failed) + "."
        self._say(note)
        return len(deleted)

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
        """Load ``record`` NOW, even if the mark already claims it.

        THE DIFFERENCE FROM `set_loaded_run`, and the reason both exist: that
        one is idempotent -- it returns early when the key has not changed,
        which is right for a run announcing itself and wrong for a user asking
        twice. If the mark and the screen have drifted apart for any reason,
        an idempotent load is exactly the one that cannot repair it.

        So this always announces. Asking for the run already on screen costs a
        redundant reload and gets a user out of a stuck state; refusing them
        leaves them with a table that ignores clicks.
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
        if verb == "delete":
            return bool(self.delete_runs_from_disk(records))
        return False

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

    def _on_selection(self) -> None:
        if self._rebuilding:
            # The re-select at the end of `_rebuild` is this panel putting
            # the highlight back, not the user choosing a run.
            return
        record = self.selected_trial()
        if record is None:
            return
        # PICKING A RUN IS LOADING IT. The selection already re-points the
        # results panel and the figure grid through `trial_activated`, so a
        # mark that did not follow it would be a second answer to "which run
        # is loaded" -- and instruction 145's rule is one vocabulary.
        key = self._row_key(record)
        before = self._loaded_key
        if key and key != before and _is_ok(record):
            self._loaded_key = key
            self._source_note = ""
            self._paint_the_loaded_mark()
            if self._announce_the_loaded_run(before):
                # The funnel emitted `trial_activated` with the row read back
                # off the composed frame. Emitting again here is the same
                # click twice, and the screen would load the run twice.
                return
        # A ROW THAT CANNOT BECOME THE LOADED RUN IS STILL A CLICK. A trial
        # that failed or is still going has no results to show, and the
        # screen answers with the reason -- which is the whole difference
        # between a table that ignores clicks and one that explains them.
        self.trial_activated.emit(record)
