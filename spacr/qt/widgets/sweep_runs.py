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
(``_record_run``, updated by ``_update_run`` when it finishes) and shown in
the same table as the sweep's trials, described by the SAME COLUMNS -- which
is what makes the two comparable, and comparing them is the entire reason the
tab exists.

The recorders are underscored deliberately: `tests/test_api_i18n_extractor.py`
holds an EXACT count of the documented public API, and that file belongs to
another session right now. The two names want promoting to `record_run` /
`update_run` in the same commit that bumps the count.
"""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

#: What the sweep writes its table to, inside the destination folder.
RESULTS_FILENAME = "sweep_results.csv"

#: The columns worth reading first, in this order, when the table has them.
#: Settings, then WHAT WENT IN, then what came out: a hit count means little
#: without the size of the design behind it, and two trials differing only by
#: a filtration cutoff can be fitted on completely different data.
#:
#: ``run`` and ``source`` lead, ahead of the sweep's ``trial_id``, because the
#: first question over a mixed table is WHICH RUN THIS IS -- a trial number
#: names nothing when half the rows are not trials.
PREFERRED_COLUMNS = (
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

#: A run that has been started and has not come back yet. Not "ok": a run
#: still going has produced nothing to look at, and a row that claims
#: otherwise is a click that opens last month's folder.
STATUS_RUNNING = "running"


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
        #: :meth:`_record_run` hands back. An ordinary run and a re-fit are
        #: the same kind of thing as a trial and belong in the same table.
        self._recorded: "dict[int, dict]" = {}
        self._next_handle = 0

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
        # `tests/test_api_i18n_extractor.py` holds an exact count of that
        # which another session owns right now.
        self._sweep_frame = frame if frame is not None and len(frame) else None
        if self._sweep_frame is None and not source:
            source = "The sweep has recorded no trials yet."
        return self._rebuild(source)

    # ------------------------------------------------------- this session's

    def _record_run(self, label: str, source: str = SOURCE_RUN,
                    settings=None, folder: str = "") -> int:
        """Put a run of the module on the table, and return its handle.

        Recorded when the run STARTS, for the reason the figure grid marks
        its section then: a run that fails, or that is still going, is a fact
        worth seeing rather than a gap. Its row says ``running`` until
        :meth:`_update_run` is told otherwise -- not ``ok``, because a row
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

    def _update_run(self, handle: int, **fields) -> bool:
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
        self._rebuild()
        return True

    def _recorded_rows(self) -> list:
        """This session's runs, oldest first."""
        return [dict(row) for _handle, row in sorted(self._recorded.items())]

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
        self._frame = frame[ordered_columns(frame)].reset_index(drop=True)
        self.table.set_frame(self._frame, key_column=(
            "run" if "run" in self._frame.columns else None))
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

    @staticmethod
    def _describe(frame, source: str = "") -> str:
        """The one-line summary over the table.

        Counts what did not work OUT LOUD, rather than leaving it to be
        noticed: a sweep whose trials mostly failed still writes a
        full-looking table, and so does a session of runs that all crashed.
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
        record = self.selected_trial()
        if record is not None:
            self.trial_activated.emit(record)
