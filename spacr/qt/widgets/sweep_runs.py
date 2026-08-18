"""The sweep's runs, as a tab beside the regression's results.

Instruction 116, as corrected by the maintainer on 2026-08-16:

    the parameter search is the main module setup plus one extra tab for the
    runs (the maintainer's exact words are in instruction 116).

Which is a smaller job than the plan it superseded, and a better one. The
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

EVERY RUN, NOT ONLY THE SWEEP'S (instruction 125 C):

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

from PySide6.QtCore import Signal
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
    "trial_id", "status", "regression_type", "inference", "analysis_unit",
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

#: A run that has been started and has not come back yet. Not "ok": a run
#: still going has produced nothing to look at, and a row that claims
#: otherwise is a click that opens last month's folder.
STATUS_RUNNING = "running"


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

    :ivar trial_activated: emitted with the selected run's row, as a dict.
    :ivar loaded: emitted with the number of rows shown.
    """

    trial_activated = Signal(dict)
    loaded = Signal(int)
    #: Emitted with the row of the run that is now the LOADED one -- the run
    #: every view on this screen is describing. Emitted for the automatic
    #: cases too (a run finishing, a folder holding exactly one run), because
    #: a view that only learns about deliberate choices shows the wrong run
    #: after the common one.
    loaded_run_changed = Signal(dict)

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
        if _is_ok(row):
            self._loaded_key = self._row_key(row)
        self._rebuild()
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

        self._loaded_key = self._row_key(self._recorded[handle])
        self._rebuild(f"Loaded the run in {run_folder}.")
        record = self.loaded_run() or dict(self._recorded[handle])
        self.loaded_run_changed.emit(record)
        # AND THE VIEWS FOLLOW IT. `trial_activated` is what re-points the
        # results panel, the summary and the figure grid; a run that is
        # marked loaded and shown nowhere would be the same broken click as
        # a row that does nothing.
        self.trial_activated.emit(record)
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
                    record = self.loaded_run() or record
                    self.loaded_run_changed.emit(record)
                    # AND THE VIEWS FOLLOW THE CHOICE. Moving the mark and
                    # leaving the results panel, the summary and the figure
                    # grid on the previous run is the failure 154 G is about:
                    # the choice has to be visible from the views that depend
                    # on it, not only from the tab that sets it.
                    self.trial_activated.emit(record)
                return True
        return False

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

    def _rebuild(self, source: str = "") -> bool:
        """Compose the session's runs and the sweep's trials into one table.

        THE SESSION'S OWN RUNS COME FIRST. A sweep is sixty rows and the run
        the user just made is the one they are looking for; sorting is one
        click away for every other question.
        """
        import pandas as pd

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
        if key and key != self._loaded_key and _is_ok(record):
            self._loaded_key = key
            self._source_note = ""
            self._paint_the_loaded_mark()
            self.loaded_run_changed.emit(record)
        self.trial_activated.emit(record)
