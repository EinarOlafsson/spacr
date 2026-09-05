"""The set of measurement databases a screen is working on.

Where a workflow accepts one database, this widget lets you add, remove, and
inspect several. It uses
:class:`spacr.qt.widgets.table_chip.TableChip`, already used for the Gate
Editor's table working set. Databases get the same control rather than a
second one for the same idea.

WHAT THIS WIDGET IS FOR, beyond holding a list of paths:

    THE ANSWER HAS TO ARRIVE BEFORE THE USER COMMITS.

The column set produced by a merge defines the analysis about to run, and
finding out afterwards that half the
measurements were dropped is finding out too late. So every time the set
changes this asks :func:`spacr.multi_database.describe_merge` -- which reads
only sqlite metadata and the distinct plate ids -- and puts the answer on
screen: how many rows, how many columns are common, which measurements are in
only some, and whether two databases claim the same plate.

AND IT ASKS OFF THE GUI THREAD. That read is cheap in the VOLUME it reads and
it is not cheap in LATENCY, and this file used to hold the first as a reason
to believe the second: it stats every database and then opens each one with
sqlite five times over, on paths that are folders the user chose. Measured on
the maintainer's machine 2026-09-04, a single ``os.path.exists`` under an
``autofs`` mount whose share was asleep had not returned after TWENTY SECONDS.
This widget is built while a settings panel is laid out and refreshed on every
drop, so that was the whole application frozen, and it left no traceback
because a stalled event loop is not a crash. Everything the summary needs is
now read by :func:`_read_the_merge` on a worker thread
(:mod:`spacr.qt.job_runner`) and painted when it lands: every line this widget
used to print, it still prints, a moment later.

A COLLISION IS SHOWN, NOT RESOLVED. Two databases that each hold a ``plate1``
are two experiments, and the widget says so and names them. It deliberately
does NOT offer ``on_collision='qualify'``: rewriting ``plate1`` to
``runA-plate1`` makes the keys unique and hides which experiment a plate came
from inside its own id, where nothing can block on it, test for it or colour
by it. The resolutions offered are the ones that keep the experiment
analysable -- remove one of the databases, or rename the plates.
"""
from __future__ import annotations

import os
from functools import partial
from typing import Callable, List, Optional, Sequence

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
    QWidget,
)

from ..theme import SPACING
from .table_chip import TableChip

__all__ = ["DatabaseSetWidget", "database_for_source"]

#: What the Image UMAP and every other ``src``-taking module means by a source
#: root: the plate folder Measure wrote into.
#:
#: The same join :func:`spacr.utils.get_db_paths` performs, spelled out here
#: rather than imported: ``spacr.utils`` pulls torch, and a settings panel must
#: not pay seconds of import to redraw a summary line when a chip is added.
#: ``test_the_folder_join_matches_get_db_paths`` is what keeps the two equal.
MEASUREMENTS_SUFFIX = ("measurements", "measurements.db")


def database_for_source(source: str, mode: str = "database") -> str:
    """The database a chosen source names.

    :param source: what the user picked -- a database file in ``'database'``
        mode, a spaCR project/plate folder in ``'folder'`` mode.
    :param mode: ``'database'`` or ``'folder'``.
    :returns: the path to open. Nothing is checked for existence; a source
        whose database is missing is reported by the summary rather than
        removed from the set behind the user's back.
    """
    if mode == "folder":
        return os.path.join(str(source).rstrip(os.sep), *MEASUREMENTS_SUFFIX)
    return str(source)


def _read_the_merge(paths: Sequence[str], table: str):
    """Everything the summary line says, read OFF the GUI thread.

    Returns one of ``('plan', MergePlan)``, ``('missing', (paths, total))`` or
    ``('error', (total, message))`` -- data rather than painted text, because
    a worker thread may not touch a widget, and a returned failure rather than
    a raised one because "that file is not a database" is one of the three
    answers this summary legitimately has, not a crash.

    NEITHER HALF CAN COME FROM :mod:`spacr.qt.path_probe`. The stat is what
    gates the sqlite opens, and the probe answers a path it has not seen
    OPTIMISTICALLY -- so believing it here would move the twenty-second wait
    off ``os.path.isfile`` and onto the ``sqlite3.connect`` one line later,
    which is the same freeze with a longer call chain.

    TOTAL: every path out of here is one of those three answers, and the
    import and the stat loop are inside the guard along with the read. A
    worker that raises delivers NOTHING -- ``JobRunner`` hands a result only
    to a job that succeeded -- and the summary would then keep the "reading
    …" placeholder it was given before the submit, which reads as "still
    working" rather than "this failed". A failure the user can see beats a
    traceback in a log they never open.
    """
    paths = list(paths)
    try:
        from ...multi_database import describe_merge

        missing = [path for path in paths if not os.path.isfile(path)]
        if missing:
            return ("missing", (missing, len(paths)))
        return ("plan", describe_merge(paths, table))
    except Exception as exc:                                     # noqa: BLE001
        return ("error", (len(paths), str(exc)))


def _the_ones_still_there(sources: Sequence[str]) -> List[str]:
    """Which of ``sources`` the filesystem still has. Off the GUI thread."""
    return [path for path in sources if os.path.exists(path)]


def _threaded_by_default() -> bool:
    """Whether a new widget reads its databases on a worker thread.

    True in the application, where that read is what froze it. False inside a
    pytest session, where ``JobRunner(threaded=False)`` runs the same job
    inline and emits the same signals in the same order -- the seam that class
    documents for exactly this -- so a test can press a control and read the
    summary on the next line instead of pumping an event loop. A test that
    wants the threaded behaviour, and the regression test for this freeze
    does, asks for it with ``threaded=True``.

    The default is where the decision has to live: the application builds this
    widget inside ``settings_model._widget_for``, which has no such flag of
    its own to thread through. ``SPACR_PYTEST_SESSION`` is set by the root
    conftest before collection; ``PYTEST_CURRENT_TEST`` covers a widget built
    by a test that reached this module some other way.
    """
    return not (os.environ.get("SPACR_PYTEST_SESSION") == "1"
                or "PYTEST_CURRENT_TEST" in os.environ)


class DatabaseSetWidget(QWidget):
    """Add, remove and see the databases a screen will merge.

    :param value: the initial sources. A bare string is accepted because
        every module's ``src`` has historically been one, and a settings CSV
        written before this widget existed still holds one.
    :param mode: ``'database'`` (the user picks .db files -- the Gate Editor)
        or ``'folder'`` (the user picks plate folders -- Image UMAP and every
        other module whose ``src`` is a project root).
    :param table: the table the merge is described on. ``'cell'`` is the
        anchor every object join is built on.
    :param min_items: how many sources may never be removed. The Gate Editor
        passes 1, because a gate editor with no table is a screen with
        nothing on it; a settings panel passes 0, because clearing the field
        is a legitimate thing to want.
    :param on_colour_by: called with :data:`spacr.multi_database.SOURCE_COLUMN`
        or ``None`` when the "colour by source" box is toggled. Given by a
        settings panel that owns a ``color_by`` field; omitted elsewhere, and
        the box is then not shown.
    :param parent: parent widget; ownership only.
    :param threaded: whether the databases are read on a worker thread.
        ``None`` decides -- see :func:`_threaded_by_default`. Only a test has
        any reason to say.
    :param title: the caption on the file dialog this opens. ``None`` picks
        one from ``mode``, so it only needs giving when the dialog is asking
        for something more specific than "a database" or "a folder" -- the
        dialog's title is the only prompt it has.
    """

    value_changed = Signal()

    def __init__(self, value=None, parent=None, *, mode: str = "database",
                 table: str = "cell", title: Optional[str] = None,
                 min_items: int = 0,
                 on_colour_by: Optional[Callable[[Optional[str]], None]] = None,
                 threaded: Optional[bool] = None):
        """Build the multi-database chooser.

        The reads leave the GUI thread deliberately: the summary stats every
        database and then opens each with sqlite, over paths that are the user's
        own -- one of them an autofs mount that had not answered a single stat
        after twenty seconds -- and doing that while a settings panel lays out
        froze the application. Two runners rather than one, because
        ``job_finished`` carries a bool and no job identity: a shared runner
        could not tell whose failure it was being told about, so the restore
        check failing would clear the summary read's in-flight flag and paint
        its error over a read still running.

        :param value: the sources to start with.
        :param parent: parent widget, or ``None``.
        :param mode: ``"folder"`` for spaCR project folders, anything else for
            measurement databases.
        :param table: which table the merge summary is computed over.
        :param title: the file-dialog caption; defaults by mode.
        :param min_items: how many sources must remain; chips below this are
            not removable.
        :param on_colour_by: called with the source column when the user asks
            for the map to be coloured by database, and with ``None`` when they
            stop.
        :param threaded: run the reads on a worker thread; ``None`` follows the
            process default.
        """
        super().__init__(parent)
        self.setObjectName("DatabaseSetWidget")
        self._mode = "folder" if mode == "folder" else "database"
        self._table = table
        self._min_items = max(0, int(min_items))
        self._on_colour_by = on_colour_by
        self._sources: List[str] = []
        self._plan = None
        #: THE READS LEAVE THE GUI THREAD. `_read_the_merge` stats every
        #: database and then opens each of them with sqlite, and the paths are
        #: the user's own -- one of the maintainer's is an `autofs` mount that
        #: had not answered a single stat after twenty seconds. Doing that
        #: while a settings panel is being laid out froze the application.
        #:
        #: `user_visible=False`: nothing here is a run the user started, so it
        #: must never claim a run banner on Home.
        from ..job_runner import JobRunner

        threads = (_threaded_by_default() if threaded is None
                   else bool(threaded))
        #: TWO RUNNERS, NOT ONE, and the reason is the same one
        #: :class:`spacr.qt.widgets.row_exclusion.RowExclusionEditor` has two:
        #: ``job_finished`` carries a bool and no job identity, so one runner
        #: shared by the summary read and the restore check cannot tell whose
        #: failure it is being told about. The restore check failing would
        #: then let go of the summary read's in-flight flag and paint its
        #: error over the read still running. ``cancel()`` is likewise all or
        #: nothing per runner.
        self._jobs = JobRunner(
            self, threaded=threads,
            app_key="database set", user_visible=False)
        self._presence_jobs = JobRunner(
            self, threaded=threads,
            app_key="database set restore", user_visible=False)
        # A JobRunner hands the result only to a job that SUCCEEDED, so a read
        # that dies some other way would leave `_reading` true for the life of
        # the widget -- every later change to the set would then coalesce into
        # a read that is never going to run, freezing the summary on whatever
        # it happened to say -- and would leave the "reading …" placeholder up
        # for ever. `_read_settled` is what clears both; `job_failed` arrives
        # first and carries the one line worth showing.
        self._jobs.job_failed.connect(self._read_failed)
        self._jobs.job_finished.connect(self._read_settled)
        #: Bumped every time the set changes, so an answer about the set as it
        #: WAS cannot land on top of the set as it is.
        self._summary_token = 0
        #: One read in flight at a time. Dropping three plate folders is three
        #: calls to `_rebuild`, and they all ask the same question of the same
        #: files, so a read arriving mid-flight is REMEMBERED AND RUN ONCE
        #: rather than queued as three more worker threads.
        self._reading = False
        self._read_again = False
        #: How many databases the read in flight is about, and the last thing
        #: `job_failed` said about one -- both only so that a read which never
        #: delivers can still be reported in the words the summary uses.
        self._reading_count = 0
        self._last_read_error = ""
        self._title = title or (
            "Choose one or more spaCR project folders"
            if self._mode == "folder" else
            "Choose one or more measurement databases")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(SPACING["xs"])

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING["xs"])
        self.add_button = QPushButton(
            "Add project folders…" if self._mode == "folder"
            else "Add databases…", self)
        self.add_button.setObjectName("DatabaseSetAdd")
        self.add_button.setToolTip(
            "Adds to the set rather than replacing it, so three plates can "
            "be gathered in three trips. Every row of the merged frame "
            "carries the database it came from, so the map can be coloured "
            "by source.")
        self.add_button.clicked.connect(self.choose_sources)
        row.addWidget(self.add_button)
        row.addStretch(1)
        outer.addLayout(row)

        self._chips = QHBoxLayout()
        self._chips.setContentsMargins(0, 0, 0, 0)
        self._chips.setSpacing(SPACING["xs"])
        self._chips.addStretch(1)
        outer.addLayout(self._chips)

        #: What the merge WOULD cost, before it is performed.
        self.summary = QLabel("", self)
        self.summary.setObjectName("DatabaseSetSummary")
        self.summary.setWordWrap(True)
        self.summary.setProperty("role", "hint")
        outer.addWidget(self.summary)

        self.colour_by_source = QCheckBox(
            "Colour the map by source database", self)
        self.colour_by_source.setObjectName("DatabaseSetColourBySource")
        self.colour_by_source.setToolTip(
            "A merged embedding whose clusters turn out to be the plates "
            "rather than biology is the most important thing a multi-database "
            "map can show — and it can only show it if the points are "
            "coloured by where they came from.")
        self.colour_by_source.toggled.connect(self._on_colour_toggled)
        self.colour_by_source.setVisible(False)
        outer.addWidget(self.colour_by_source)

        self.set_value(value)

    # -- the value ---------------------------------------------------------
    def get_value(self):
        """The chosen sources: a bare string for one, a list for several.

        NOT always a list, deliberately. ``src`` has been a string for every
        module since spaCR had modules, and it is written to the settings
        CSV, read by the CLI, replayed by the run journal and joined onto by
        anything that does ``os.path.join(src, ...)``. Returning
        ``['/data/plate1']`` where a string was returned before would change
        what every one of those sees for a user who chose ONE folder and
        wanted nothing to do with this feature -- exactly the regression the
        one-element list caused for ``column_csv`` (see
        ``settings_model.PATH_LIST_SINGLE_KEYS``).

        Several sources produce a list, which is the shape
        :func:`spacr.core.generate_image_umap` builds on the first line it
        touches ``src`` anyway.
        """
        if len(self._sources) == 1:
            return self._sources[0]
        return list(self._sources)

    def set_value(self, value) -> None:
        """Replace the set. Accepts a list, a bare string, or ``None``."""
        self._sources = self._clean(value)
        self._rebuild()

    def sources(self) -> List[str]:
        """The chosen sources."""
        return list(self._sources)

    def database_paths(self) -> List[str]:
        """The database each source names -- what a merge would open."""
        return [database_for_source(source, self._mode)
                for source in self._sources]

    def plan(self):
        """The last :class:`spacr.multi_database.MergePlan`, or ``None``."""
        return self._plan

    # -- instruction 180: what this widget contributes to a saved run -------

    def workspace_state(self) -> dict:
        """The attached set, IN ORDER, and the databases behind it.

        Order is state, not presentation: a merge resolves a column
        collision in favour of the first source that has it, so the same two
        databases in the other order are a different merged table.

        Both the sources and the databases they resolve to are written down.
        A source is a folder the user chose; the database is the file the
        merge opens, and it is the one whose absence or edit a restore has to
        be able to report.
        """
        return {
            "mode": str(self._mode),
            "sources": self.sources(),
            "databases": self.database_paths(),
        }

    def apply_workspace_state(self, state) -> bool:
        """Re-attach the set. Returns whether anything was attached.

        A source that is no longer there is left out and the rest are still
        attached -- one moved plate must not cost the user the other three.
        The workspace document names every missing one in its own report, so
        nothing dropped here is dropped silently.

        WHICH ONES ARE STILL THERE IS ASKED OFF THE GUI THREAD. This was an
        ``os.path.exists`` per source, inline, and these are the paths of a
        workspace saved days ago -- precisely the ones most likely to sit on a
        mount that has since gone to sleep, and reopening a saved session is
        not a moment to freeze the application in. So the whole set is
        attached now and the ones that turn out to be gone are dropped when
        the answer lands. Attaching a moved plate and removing it a moment
        later costs one redraw; waiting twenty seconds for its stat costs the
        session.

        WHAT THE RETURN VALUE MEANS NOW, because it is the one thing the move
        changed: it answers "was there anything to attach", not "was any of it
        still on disk". :func:`spacr.workspace.restore_workspace` prints the
        section as restored either way, where a set whose every source had
        moved used to be printed as declined. Nothing is hidden from the user
        by that -- the ones that turn out to be gone are dropped when the
        answer lands, and the summary line names every source with no database
        behind it -- but the report line is a moment earlier than the truth,
        and the alternative is blocking the restore on the stat, which is the
        freeze. Unthreaded (a test, and every other test file here) the check
        has already run by the time this returns, so the old answer stands.
        """
        if not isinstance(state, dict):
            return False
        sources = [str(p) for p in (state.get("sources") or []) if p]
        if not sources:
            return False
        self.set_value(sources)
        self._presence_jobs.submit(
            partial(_the_ones_still_there, tuple(sources)),
            partial(self._present_sources_arrived, tuple(sources)))
        return bool(self._sources)

    def _present_sources_arrived(self, checked, present) -> None:
        """Drop the restored sources that are not there. GUI thread only.

        Only the ones this answer was ABOUT are dropped -- a source the user
        added while the check was in flight was never asked about, and
        removing it because it is missing from an answer that predates it
        would be the widget quietly undoing what the user just did. A check
        abandoned by :meth:`spacr.qt.job_runner.JobRunner.cancel` never
        reaches here at all: that generation guard is the runner's.
        """
        gone = set(checked) - set(present or ())
        if not gone:
            return
        keep = [path for path in self._sources if path not in gone]
        if keep == self._sources:
            return
        try:
            self.set_value(keep)
            # SAID OUT LOUD, because the panel FOLLOWS the set: `settings_
            # model` rebuilds the fields that offer columns and rows from
            # these databases on `value_changed`. While this check was inline
            # the pruning happened before the panel ever saw the set; now it
            # happens after, so a silent prune would leave those fields
            # offering the columns of a plate that has moved.
            self.value_changed.emit()
        except RuntimeError:
            pass            # the widget went while the check was in flight

    # -- letting go --------------------------------------------------------
    def shutdown(self) -> None:
        """Stop reading; let no worker outlive the widget. Idempotent.

        Qt ABORTS THE PROCESS when a running QThread is destroyed, and this
        widget now starts them. Public because it is a CHILD -- a settings
        panel builds it inside a form, and Qt delivers a close event to the
        window rather than to every widget in it, so a host that wants the
        reads stopped when the user navigates away has to say so.

        Not the only line of defence: the QThreads are unparented and retire
        themselves, and ``JobRunner._relay`` catches the ``RuntimeError``
        PySide6 raises when a worker settles after its runner's C++ half has
        gone.
        """
        # Cleared here as well as shut down, because `JobRunner.cancel`
        # abandons what is pending WITHOUT emitting `job_finished` -- nothing
        # would otherwise let go of the in-flight flag.
        self._reading = False
        self._read_again = False
        self._jobs.shutdown()
        self._presence_jobs.shutdown()

    def closeEvent(self, event):        # noqa: N802 - Qt override
        """Closing mid-read must not leave a worker behind."""
        self.shutdown()
        super().closeEvent(event)

    # -- editing -----------------------------------------------------------
    def choose_sources(self) -> None:
        """Open the picker and ADD what comes back.

        The dialog is opened at ``""`` -- no remembered root. That is the one
        reason this stays on the GUI thread while the rest of the file's
        filesystem work moved off it: there is no path of the user's to stat
        before the dialog exists, and listing the directory it lands in is
        the thing the user asked for, in a window that is its own progress
        indicator. A remembered start directory would put a possibly-sleeping
        mount back in front of the event loop; do not add one here.
        """
        if self._mode == "folder":
            folder = QFileDialog.getExistingDirectory(self, self._title, "")
            chosen = [folder] if folder else []
        else:
            chosen, _ = QFileDialog.getOpenFileNames(
                self, self._title, "",
                "Measurements (*.db *.sqlite *.sqlite3);;All files (*)")
        self.add_sources(chosen)

    def add_sources(self, paths: Sequence[str]) -> int:
        """Add sources, ignoring the ones already in the set.

        :returns: how many were actually added.
        """
        added = 0
        for path in paths or []:
            text = str(path).strip()
            if text and text not in self._sources:
                self._sources.append(text)
                added += 1
        if added:
            self._rebuild()
            self.value_changed.emit()
        return added

    def remove_source(self, name: str) -> bool:
        """Drop one member of the set, by its chip's label or its path.

        The chip carries the LABEL -- ``plate1``, not
        ``/data/plate1/measurements/measurements.db`` -- because that is what
        a legend and the provenance column say. Both are accepted so a caller
        with the path does not have to work out the label first.
        """
        target = None
        if name in self._sources:
            target = name
        else:
            for source, label in zip(self._sources, self._labels()):
                if label == name:
                    target = source
                    break
        if target is None or len(self._sources) <= self._min_items:
            return False
        self._sources.remove(target)
        self._rebuild()
        self.value_changed.emit()
        return True

    def clear(self) -> None:
        """Empty the set."""
        if not self._sources:
            return
        self._sources = []
        self._rebuild()
        self.value_changed.emit()

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _clean(value) -> List[str]:
        """Normalise a stored value into a list of source paths.

        The settings placeholders for an unchosen ``src`` are dropped rather
        than kept: rendering ``path`` as a chip would offer to merge a database
        called "path".

        :param value: ``None``, one path, or an iterable of them.
        :returns: the paths, de-duplicated and in order.
        """
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            # 'path' is what spacr.settings ships as the "not chosen yet"
            # placeholder for src. Rendering it as a chip would offer to merge
            # a database called path.
            return [] if text in ("", "path", "/path", "/path/to/src") else [text]
        out: List[str] = []
        for item in value:
            text = str(item).strip()
            if text and text not in out:
                out.append(text)
        return out

    def _labels(self) -> List[str]:
        """The name each source will carry in the provenance column.

        Asked of :mod:`spacr.multi_database` rather than computed here, so a
        chip and the ``source_database`` value it stands for cannot disagree.
        """
        from ...multi_database import source_labels

        paths = self.database_paths()
        if not paths:
            return []
        try:
            return list(source_labels(paths))
        except Exception:
            return [os.path.basename(str(p).rstrip(os.sep)) or str(p)
                    for p in self._sources]

    def _rebuild(self) -> None:
        """Redraw the chips, re-read the summary and re-gate the colour switch.

        The switch appears only with a callback to answer and more than one
        source -- colouring by database says nothing about a set of one.
        """
        self._rebuild_chips()
        self._refresh_summary()
        self.colour_by_source.setVisible(
            self._on_colour_by is not None and len(self._sources) > 1)

    def _rebuild_chips(self) -> None:
        """Rebuild the source chips, keeping the trailing stretch.

        Chips stop being removable at ``min_items``, so the set cannot be
        emptied below what the module requires.
        """
        while self._chips.count() > 1:
            item = self._chips.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        removable = len(self._sources) > self._min_items
        for index, (source, label) in enumerate(
                zip(self._sources, self._labels())):
            chip = TableChip(label, self, removable=removable)
            chip.setToolTip(source)
            chip.removed.connect(self.remove_source)
            self._chips.insertWidget(index, chip)

    def _refresh_summary(self) -> None:
        """Ask what the merge would cost; paint the answer when it lands.

        SPLIT IN TWO, and the split is the fix for a frozen application. This
        half touches widgets only. The half in :func:`_read_the_merge` stats
        every database and opens each one with sqlite, and it runs on a
        worker, because those paths are folders the user chose -- see the
        module docstring for the twenty seconds one of them took to answer.

        Nothing the user was shown has gone: all three answers -- the cost of
        the merge, the sources with no database yet, the file that could not
        be read -- are still printed, from :meth:`_summary_arrived`.
        """
        self._plan = None
        if not self._sources:
            self._summary_token += 1
            self._read_again = False
            self.summary.setText("")
            return
        if self._reading:
            # Coalesced, not queued -- and the token goes up so that the
            # answer already in flight, which is about a different set of
            # files, is discarded rather than painted for a moment.
            self._summary_token += 1
            self._read_again = True
            return
        paths = self.database_paths()
        self._summary_token += 1
        self._reading = True
        self._reading_count = len(paths)
        self._last_read_error = ""
        self.summary.setText(f"reading {len(paths)} database(s)…")
        token = self._summary_token
        started = self._jobs.submit(
            partial(_read_the_merge, tuple(paths), self._table),
            partial(self._summary_arrived, token))
        if not started and self._summary_token == token:
            # Nothing is in flight, so the flag must not say there is. An
            # unthreaded runner answers False when the read itself raised,
            # having already reported it through `job_failed` -- and through
            # `_read_settled`, which may by then have run the coalesced read
            # this one was holding up. The token says whether that happened:
            # clearing the flag unconditionally would abandon THAT read.
            self._reading = False

    def _read_failed(self, message: str) -> None:
        """Remember why a read died, so :meth:`_read_settled` can say it."""
        self._last_read_error = str(message or "")

    def _read_settled(self, ok: bool) -> None:
        """Clear the in-flight flag, AND the placeholder, for a read that
        never delivered.

        :meth:`spacr.qt.job_runner.JobRunner.submit` hands the result only to
        a job that succeeded, so a read that dies some other way -- the
        handler raising, a worker killed on the way out -- reaches
        :meth:`_summary_arrived` never. Two things would then be left behind:
        ``_reading``, which makes every later change to the set coalesce into
        a read that is never going to run, and the "reading N database(s)…"
        placeholder, which tells the user the answer is still coming when it
        is not. Both are let go here.

        Only ``self._jobs`` is connected to this. The restore check has its
        own runner precisely so that its failure cannot be mistaken for this
        one's -- ``job_finished`` carries no job identity.
        """
        if ok or not self._reading:
            return
        self._reading = False
        again, self._read_again = self._read_again, False
        try:
            if self._sources:
                detail = self._last_read_error or "the read did not finish"
                self.summary.setText(
                    f"could not read {self._reading_count} database(s) as "
                    f"'{self._table}': {detail}")
        except RuntimeError:
            return          # the widget's C++ half went with the read
        finally:
            self._last_read_error = ""
        if again:
            self._refresh_summary()

    def _summary_arrived(self, token: int, answer) -> None:
        """Paint one finished read. GUI thread only."""
        self._reading = False
        self._last_read_error = ""
        again, self._read_again = self._read_again, False
        alive = True
        try:
            if answer is not None and token == self._summary_token:
                kind, payload = answer
                if kind == "plan":
                    self._plan = payload
                    self.summary.setText(self._summary_text(payload))
                elif kind == "missing":
                    missing, total = payload
                    # Named, not swallowed. In folder mode the user picked a
                    # plate folder and the database is two levels below it, so
                    # "nothing happened" would be indistinguishable from "that
                    # plate was never measured".
                    self.summary.setText(
                        f"{len(missing)} of {total} sources have no "
                        f"measurements database yet: "
                        + ", ".join(os.path.dirname(p) for p in missing[:3])
                        + (" …" if len(missing) > 3 else ""))
                else:
                    total, message = payload
                    self.summary.setText(
                        f"could not read {total} database(s) as "
                        f"'{self._table}': {message}")
        except RuntimeError:
            # The widget's C++ half went while the read was in flight. There
            # is nothing left to paint on, and raising here would surface as
            # an unhandled exception inside the Qt event loop.
            alive = False
        finally:
            # In a `finally`, because the coalesced read is the set the user
            # is actually looking at. Losing it to a bad answer for the
            # PREVIOUS set would freeze the summary on the placeholder.
            if alive and again:
                self._refresh_summary()

    def _summary_text(self, plan) -> str:
        """What the merge would cost, in the order a user needs it."""
        lines = [
            f"{len(plan.sources)} database(s) · {plan.total_rows:,} rows · "
            f"{len(plan.common_columns)} columns in all of them"
        ]
        dropped = plan.dropped_columns
        if dropped:
            shown = ", ".join(dropped[:6]) + (" …" if len(dropped) > 6 else "")
            lines.append(
                f"{len(dropped)} measurement(s) are in only some of them and "
                f"would be dropped: {shown}")
        if plan.colliding_plates:
            detail = "; ".join(
                f"{plate} in {', '.join(labels)}"
                for plate, labels in sorted(plan.colliding_plates.items()))
            lines.append(
                f"THE SAME PLATE ID IS IN MORE THAN ONE DATABASE: {detail}. "
                "Merging would compute every per-well number over two "
                "experiments at once. Remove one of them, or rename the "
                "plates.")
        return "\n".join(lines)

    def _on_colour_toggled(self, on: bool) -> None:
        """Ask the owning panel to colour by source, or to stop.

        :param on: the switch's new state. A merged embedding whose clusters
            turn out to be the plates rather than biology is the most important
            thing a multi-database map can show, and it can only show it when
            the points are coloured by where they came from.
        """
        if self._on_colour_by is None:
            return
        from ...multi_database import SOURCE_COLUMN

        self._on_colour_by(SOURCE_COLUMN if on else None)
