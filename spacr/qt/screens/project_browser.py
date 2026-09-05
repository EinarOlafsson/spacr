"""``N4`` — the Project Browser: every project on disk, in one table.

Navigating by folder is how spaCR has always been used, and it is why nobody
can answer "which plates have been measured?", "which one is the 400 GB?" or
"which results no longer match the masks under them?" without opening six
windows. This screen is that answer as a list.

It computes nothing. Every column comes from :mod:`spacr.projects`, which in
turn assembles :func:`spacr.data_manager.scan_project` (the size and the
unaccounted-for bytes), :func:`spacr.ports.declared_outputs` (the stage),
:meth:`spacr.artifacts.Registry.is_stale` (what is out of date) and
:func:`spacr.chaining.next_steps` (what could run next). A browser with its
own opinion about any of those is a browser that disagrees with the screen
the user opens next.

Two things about it are load-bearing rather than cosmetic.

**A project the registry has never seen is still listed.** Projects are found
by walking the disk, so a folder copied from a colleague appears the moment
the browser is pointed at its parent — with its stage, its size and the date
its files were last written. What it does *not* show is "0 stale", which
would read as *clean*: with no provenance there is nothing to compare
against, so the state column says "unknown — nothing recorded" and the note
says why.

**The scan never blocks the window.** Walking a plate folder is tens of
thousands of ``stat`` calls, and doing it on the GUI thread is a frozen
application for however long the filesystem takes. Everything goes through
:class:`spacr.qt.job_runner.JobRunner`, whose completion handlers here are
**bound methods** — read that module's docstring for what a closure connected
to ``thread.finished`` costs. The background activity spinner follows the run
registry on its own, so nothing here has to drive it.

**And neither does opening the screen, nor opening its folder chooser.**
Two things here take a path the user typed on some *other* screen: the search
folders :func:`make_project_browser_screen` seeds from the recent-source list,
and the folder the "Add folder…" dialog starts in. Both used to be settled
with an ``os.path.isdir`` on the GUI thread. Measured 2026-09-04 on the
maintainer's machine, one such folder was under an ``autofs`` mount whose
share was asleep and a single ``isdir`` on it had not returned after TWENTY
SECONDS — which is not a slow screen, it is the whole application frozen with
no traceback, and it was reported as "opening the project browser crashes
spacr". Both now ask :mod:`spacr.qt.path_probe`, which answers from a cache
and stats in the background. Those two are the whole inventory: the walk
runs on a worker, and every column, note and detail line the screen draws is
rendered from the frozen :class:`spacr.projects.ProjectSummary` that walk
already returned, so filling the table and the detail pane reads nothing.

:func:`register` is **not** called at import; read its docstring.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
    QListWidget, QListWidgetItem, QPlainTextEdit, QPushButton, QSpinBox,
    QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..job_runner import JobRunner
from .. import path_probe
from ..theme import SPACING, mark_surface
from .app_screen import ModuleHeader
from ..widgets.sortable_table import install_sorting, table_item
from ..app_catalog import declared_app, register_declared

LOG = logging.getLogger("spacr.qt.screens.project_browser")

__all__ = ["ProjectBrowserScreen", "make_project_browser_screen", "register",
           "APP_KEY", "APP_NAME", "APP_DESCRIPTION", "APP_INTRO",
           "APP_CLI_NOTE", "APP_NAME_TRANSLATIONS", "COLUMNS", "MAX_PROJECTS"]

#: The registry key. Chosen once and never renamed — saved user state, the
#: command palette and ``spacr-qt project_browser`` all key off it.
APP_KEY = "project_browser"

#: The table, left to right. "State" is staleness and "Note" is the one thing
#: about the project worth saying; both come from :class:`ProjectSummary`
#: properties rather than being assembled here.
COLUMNS = ("Project", "Stage", "Size", "Files", "Last run", "State", "Note")

#: How many projects one scan will list. A user who points the browser at
#: their home directory must get a table rather than a filesystem walk.
MAX_PROJECTS = 300

#: How deep below a chosen folder to look, and the range the spin box offers.
#: The default is the shape people have: a folder of experiments, each
#: holding plates.
DEFAULT_DEPTH = 2
MAX_DEPTH = 6


class ProjectBrowserScreen(QWidget):
    """The browser: a roots picker, a table of projects, and a detail pane.

    :param threaded: ``False`` runs the scan inline through the same
        :class:`~spacr.qt.job_runner.JobRunner` code path, so a test drives
        the real thing synchronously.
    :param roots: folders to search on the first scan.
    :param parent: parent widget; ownership only.
    """

    #: A scan finished; carries how many projects it listed.
    scanned = Signal(int)
    #: Something went wrong, in one line fit for a status bar.
    failed = Signal(str)
    #: The user chose a project. Carries its absolute root, for a host that
    #: wants to seed another screen with it.
    project_chosen = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None, *,
                 threaded: bool = True,
                 roots: Tuple[str, ...] = ()) -> None:
        super().__init__(parent)
        self.setObjectName("ProjectBrowser")
        self._summaries: Tuple = ()
        self._roots: List[str] = [str(r) for r in roots if r]

        self._jobs = JobRunner(self, threaded=threaded, app_key=APP_KEY)
        self._jobs.job_failed.connect(self._on_job_failed)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["md"], SPACING["md"],
                                 SPACING["md"], SPACING["md"])
        outer.setSpacing(SPACING["sm"])

        header = ModuleHeader(
            APP_NAME,
            description=APP_DESCRIPTION,
            instruction="Add the folder your projects live in, then pick "
                        "one to see what it holds.",
        )
        self._header = header
        outer.addWidget(header)

        controls = QHBoxLayout()
        controls.setSpacing(SPACING["sm"])
        self._add = QPushButton("Add folder…")
        self._add.setToolTip("Search a folder for spaCR projects")
        self._add.clicked.connect(self.choose_root)
        controls.addWidget(self._add)
        self._forget = QPushButton("Remove")
        self._forget.setToolTip("Stop searching the selected folder")
        self._forget.clicked.connect(self.forget_selected_root)
        controls.addWidget(self._forget)
        controls.addWidget(QLabel("Depth"))
        self._depth = QSpinBox()
        self._depth.setRange(0, MAX_DEPTH)
        self._depth.setValue(DEFAULT_DEPTH)
        self._depth.setToolTip(
            "How many folder levels below each search folder to look. "
            "Descent stops at a project, so a project's merged/ is never "
            "listed as a project of its own.")
        controls.addWidget(self._depth)
        self._rescan = QPushButton("Scan")
        self._rescan.setToolTip("Walk the search folders again")
        self._rescan.clicked.connect(self.rescan)
        controls.addWidget(self._rescan)
        controls.addStretch(1)
        self._status = QLabel("Add a folder to search for projects.")
        self._status.setObjectName("ProjectBrowserStatus")
        self._status.setWordWrap(True)
        controls.addWidget(self._status, 2)
        outer.addLayout(controls)

        self._root_list = QListWidget()
        self._root_list.setObjectName("ProjectBrowserRoots")
        self._root_list.setMaximumHeight(70)
        self._root_list.setToolTip("Folders searched for projects")
        outer.addWidget(self._root_list)

        split = QSplitter(Qt.Horizontal)
        self._table = QTableWidget(0, len(COLUMNS))
        install_sorting(self._table)
        self._table.setHorizontalHeaderLabels(list(COLUMNS))
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSortingEnabled(True)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._table.itemDoubleClicked.connect(self._on_double_clicked)
        split.addWidget(self._table)

        self._detail = QPlainTextEdit()
        self._detail.setObjectName("ProjectBrowserDetail")
        self._detail.setReadOnly(True)
        self._detail.setPlaceholderText(
            "Pick a project to see its stages, what is stale and why, and "
            "what could run next.")
        split.addWidget(self._detail)
        # All three regions of this screen ARE the page: there is no card and
        # no tab pane behind any of them, so without this the sweep leaves
        # them showing the backdrop straight through the text.
        mark_surface(self._root_list, self._table, self._detail)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        outer.addWidget(split, 1)

        self._refresh_root_list()
        if self._roots:
            self.rescan()
        # Drop anywhere on this screen: the path is resolved through spaCR's
        # project layout, so the plate folder finds what this screen reads.
        from ..dnd import install_for
        install_for(self, "project_browser")
        # Hover help belongs on a setting's NAME, not on the field the user
        # is about to type into (instruction 113). One post-pass rather than
        # a convention every hand-built row has to remember.
        from .settings_model import retarget_field_tooltips
        retarget_field_tooltips(self)

    # -- the search folders -------------------------------------------------
    def roots(self) -> Tuple[str, ...]:
        """The folders that will be searched."""
        return tuple(self._roots)

    def add_root(self, path: str, *, scan: bool = True) -> bool:
        """Add a folder to search. ``True`` when it was not already there."""
        path = os.path.abspath(os.path.expanduser(str(path or "")))
        if not path or path in self._roots:
            return False
        self._roots.append(path)
        # Queue a background check on it, and throw the answer away: what is
        # wanted is the CACHE ENTRY, so that `_start_directory` can offer this
        # folder to the chooser on the next click -- it refuses a folder
        # `path_probe` has never been asked about. `_start_directory` does ask
        # for itself as well, but only once the user has clicked, which is one
        # click too late to answer that click. Asking here is what makes a
        # root added this session usable by the chooser at all.
        path_probe.isdir(path)
        self._refresh_root_list()
        try:
            from ..prefs import push_recent_source
            push_recent_source(APP_KEY, path)
        except Exception:
            # Remembering the folder is a convenience; failing to remember it
            # must not cost the scan the user asked for.
            LOG.debug("could not record the recent folder", exc_info=True)
        if scan:
            self.rescan()
        return True

    def choose_root(self) -> None:
        """Ask for a folder and add it.

        The dialog opens on a folder the probe cache has already confirmed
        rather than simply on the last one searched; :meth:`_start_directory`
        says why that distinction is the difference between a chooser and a
        frozen window.
        """
        path = QFileDialog.getExistingDirectory(
            self, "Search this folder for spaCR projects",
            self._start_directory())
        if path:
            # The dialog only ever returns a folder it has just listed, so
            # this is a fact already in hand rather than an excuse for another
            # stat -- which is exactly what `path_probe.prime` is for. Primed
            # under the SAME spelling `add_root` and `push_recent_source`
            # store, because the cache is keyed on the string: priming
            # `/data/plate/` would leave the `/data/plate` every other screen
            # asks about still unknown, and the prime would buy nothing.
            #
            # `prime` records the "does it exist?" answer only; the "is it a
            # directory?" answer that :meth:`_start_directory` reads has no
            # priming entry point, and `add_root` queues a real probe for it
            # one line down. That probe is cheap by construction -- the dialog
            # has just listed the folder, so the kernel answers from cache.
            path = os.path.abspath(os.path.expanduser(str(path)))
            path_probe.prime(path, True)
            self.add_root(path)

    def _start_directory(self) -> str:
        """Where the folder chooser opens. Stats nothing, ever.

        NOT ``self._roots[-1]``, which is what this was, and which is the
        same twenty-second freeze as the seeding one button along: Qt stats
        and then LISTS the start directory before it draws the dialog, so
        handing it a remembered ``/nas_mnt`` root hangs the very click that
        asked for the chooser.

        `path_probe.isdir` answers from its cache and says *no* to a folder it
        has not probed yet. That is the pessimistic direction and the right
        one here, because the two costs are not comparable: skipping a root
        costs a dialog that opens at the home directory, and stating one costs
        the application. It also needs no `path_probe.probes.answered`
        subscription to recover -- unlike a gate in front of something the
        screen PAINTS, this one leaves nothing on screen to be wrong. The
        answer lands in the cache moments later and the next click uses it,
        and in a real session it is there already: the factory queued a probe
        for every seeded root, and :meth:`add_root` queues one for every root
        added since.

        The roots are tried newest-first rather than only the newest being
        considered, so one root still waiting on its probe does not cost the
        user the several that have already answered.

        One residue is left, and it is the smallest one available:
        `path_probe._stat_with_timeout` reports a stat that never came back as
        ``True``, so a mount that is not merely asleep but dead can still be
        cached as a directory and handed to the dialog. Narrowing that further
        is `path_probe`'s job, not this screen's. What this gate removes is
        the far commoner case -- a remembered root nobody has asked about at
        all, which is every root on the first open of every session.

        The home directory is the fallback because it is where Qt was
        already opening this dialog before any root had been remembered, and
        it is where the user's shell and desktop have it mounted anyway. It
        is not probed: a network home can be slow too, but a probe cannot
        help — there is nowhere further to fall back to, and refusing to open
        a chooser at all is worse than any wait.
        """
        for root in reversed(self._roots):
            # Asking is also how an unprobed root gets probed: `isdir` queues
            # the check it cannot answer, so a "no" here is what makes the
            # NEXT click a "yes". That is the whole recovery this gate needs.
            if path_probe.isdir(root):
                return root
        return os.path.expanduser("~")

    def forget_selected_root(self) -> None:
        """Drop the selected search folder and scan again."""
        row = self._root_list.currentRow()
        if 0 <= row < len(self._roots):
            self._roots.pop(row)
            self._refresh_root_list()
            self.rescan()

    def _refresh_root_list(self) -> None:
        self._root_list.clear()
        for path in self._roots:
            self._root_list.addItem(QListWidgetItem(path))

    # -- scanning -----------------------------------------------------------
    def rescan(self) -> None:
        """Walk the search folders again, off the GUI thread."""
        roots = list(self._roots)
        if not roots:
            self._summaries = ()
            self._table.setRowCount(0)
            self._detail.setPlainText("")
            self._status.setText("Add a folder to search for projects.")
            self.scanned.emit(0)
            return
        depth = int(self._depth.value())
        # A second scan supersedes the first, so clicking Scan twice does not
        # deliver two tables in whatever order the walks happen to finish.
        self._jobs.cancel()
        self._rescan.setEnabled(False)
        self._status.setText(
            f"scanning {len(roots)} folder(s), {depth} level(s) deep…")
        self._jobs.submit(lambda r=roots, d=depth: _scan(r, d),
                          self._on_walk_done)

    def _on_walk_done(self, outcome) -> None:
        """One walk landed: show its table, or say why there is none.

        GUI thread only — a bound method, always. A FAILURE ARRIVES HERE,
        through the completion handler, rather than on the runner's
        ``job_failed`` signal; :func:`_scan` says why that matters.
        """
        ok, payload = outcome
        if ok:
            self._on_scanned(payload)
        else:
            self._on_job_failed(str(payload))

    def _on_scanned(self, summaries) -> None:
        """Show a finished scan. GUI thread only — a bound method, always."""
        self._rescan.setEnabled(True)
        self._summaries = tuple(summaries or ())
        self._fill_table()
        unknown = sum(1 for s in self._summaries if not s.known)
        note = f", {unknown} not in the registry" if unknown else ""
        self._status.setText(
            f"{len(self._summaries)} project(s){note}."
            if self._summaries else
            "No projects found — try a parent folder, or a greater depth.")
        self.scanned.emit(len(self._summaries))

    def _on_job_failed(self, message: str) -> None:
        """Something failed. Say so without a modal, and re-enable the button.

        Never a dialog: a browser that pops one up per unreadable folder is
        unusable on a machine with a stale network mount.

        The walk's own failures reach this through :meth:`_on_walk_done`. It
        stays connected to ``JobRunner.job_failed`` as well, for the failures
        that are not the walk's — a completion handler that raised, or trouble
        in the runner itself — because those must still be said out loud.
        """
        self._rescan.setEnabled(True)
        LOG.info("project browser: %s", message)
        self._status.setText(message)
        self.failed.emit(message)

    # -- the table ----------------------------------------------------------
    def _fill_table(self) -> None:
        from ...data_manager import human_bytes

        # Sorting is switched off while rows are inserted: a sorted table
        # re-orders on every setItem, and the row index the next call writes
        # into is then not the row it just wrote.
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(self._summaries))
        for row, summary in enumerate(self._summaries):
            cells = (
                summary.name,
                summary.stage_label,
                human_bytes(summary.size_bytes),
                f"{summary.n_files:,}",
                (summary.last_run_utc or "never").replace("+00:00", ""),
                summary.staleness_note(),
                summary.note(),
            )
            for column, text in enumerate(cells):
                item = table_item(str(text))
                if column == 0:
                    item.setToolTip(summary.root)
                    # The root travels with the row so a re-sorted table still
                    # selects the project the user clicked rather than the one
                    # that happens to be at that index now.
                    item.setData(Qt.UserRole, summary.root)
                elif column == 2:
                    item.setData(Qt.UserRole, int(summary.size_bytes))
                self._table.setItem(row, column, item)
        self._table.setSortingEnabled(True)
        self._detail.setPlainText("")

    def summaries(self) -> Tuple:
        """Every :class:`spacr.projects.ProjectSummary` currently listed."""
        return self._summaries

    def selected_root(self) -> str:
        """The selected project's root, or ``""``."""
        items = self._table.selectedItems()
        if not items:
            return ""
        item = self._table.item(items[0].row(), 0)
        return str(item.data(Qt.UserRole) or "") if item is not None else ""

    def summary_for(self, root: str):
        """The listed summary for one root, or ``None``."""
        for summary in self._summaries:
            if summary.root == root:
                return summary
        return None

    def _on_selection_changed(self) -> None:
        self.show_detail(self.selected_root())

    def _on_double_clicked(self, _item) -> None:
        root = self.selected_root()
        if root:
            self.project_chosen.emit(root)

    def show_detail(self, root: str) -> str:
        """Draw the detail pane for one project. Returns what it drew."""
        summary = self.summary_for(root)
        if summary is None:
            self._detail.setPlainText("")
            return ""
        from ...projects import format_project

        lines = [format_project(summary), "", "Stages"]
        for state in summary.modules:
            lines.append(f"  {state.describe()}")
        if summary.stale or summary.missing:
            lines.append("")
            lines.append("Out of date")
            for entry in summary.stale:
                lines.append(f"  {entry.describe()}")
                explanation = entry.explain()
                if explanation:
                    lines.append(f"    {explanation}")
            for entry in summary.missing:
                lines.append(f"  {entry.describe()}")
        elif not summary.staleness_known:
            lines.append("")
            lines.append(
                "Nothing here has a run record, so no result can be checked "
                "against what produced it. Run a module on this project and "
                "spaCR starts keeping one.")
        if summary.next_steps:
            lines.append("")
            lines.append("What could run next")
            for module, blocked in summary.next_steps:
                lines.append(f"  {module}"
                             + (f" — blocked: {blocked}" if blocked else ""))
        text = "\n".join(lines)
        self._detail.setPlainText(text)
        return text

    # -- lifecycle ----------------------------------------------------------
    def is_busy(self) -> bool:
        """True while a scan is in flight."""
        return self._jobs.is_busy()

    def active_jobs(self) -> int:
        """How many worker threads are still winding down."""
        return self._jobs.active_jobs()

    def closeEvent(self, event):  # noqa: N802 - Qt name
        # Abandon in-flight work rather than let it outlive the screen: Qt
        # aborts the process if a running QThread is destroyed, and a worker
        # delivering into a closed widget is a use-after-free.
        """Stop background work and unlink before going away.

        :param event: the Qt close event.
        """
        self._jobs.shutdown()
        super().closeEvent(event)


def _browse(roots: List[str], depth: int):
    """Do the walk. Runs on a worker thread and touches no widget.

    Imported here rather than at module scope so opening any other screen
    does not pay for :mod:`spacr.projects` and the registry machinery under
    it.
    """
    from ...projects import browse

    return browse(roots, depth=depth, limit=MAX_PROJECTS)


def _scan(roots: List[str], depth: int):
    """Walk, and hand a failure back as a value instead of raising it.

    Runs on a worker thread and touches no widget.

    :returns: ``(True, summaries)`` when the walk finished, ``(False,
        message)`` when it did not.

    WHY IT DOES NOT SIMPLY RAISE. ``JobRunner.job_failed`` is **not**
    generation-guarded: :meth:`~spacr.qt.job_runner.JobRunner.cancel` drops
    the *results* of the jobs it abandons, and their failures are emitted
    anyway. Two scans in quick succession — one extra click on "Scan", or a
    second folder dropped on the table — could therefore have the first
    walk's error land after the second walk's finished table, replace the
    project count with a stale message about a folder no longer being
    searched, and re-enable the Scan button underneath a walk still in
    flight. A failure returned as a *value* travels back through ``on_done``,
    which is generation-guarded, so a superseded walk's failure is discarded
    with everything else the cancel abandoned.

    The message is spelled the way the threaded path already spelled it —
    ``TypeName: text``, which is the last line of the traceback
    ``JobRunner._on_worker_error_text`` used to hand over — so the status bar
    of a real, threaded browser reads exactly as it did before. The
    ``threaded=False`` path a test drives synchronously used to get the bare
    ``str(exc)`` from ``JobRunner._fail`` and now gets the same prefixed line
    as everything else; the two spellings agreeing is the point of putting
    the wording here rather than in two places.
    """
    try:
        return True, _browse(roots, depth)
    except Exception as exc:                                     # noqa: BLE001
        detail = str(exc).strip()
        return False, (f"{type(exc).__name__}: {detail}" if detail
                       else type(exc).__name__)


def make_project_browser_screen(app_key: Optional[str] = None) -> QWidget:
    """Factory handed to :func:`spacr.qt.app.register_app`.

    Seeds the search folders from the ones the user last pointed any
    source-taking screen at, so the browser is useful on first open instead
    of empty.
    """
    roots: Tuple[str, ...] = ()
    try:
        from ..prefs import get_recent_sources
        remembered = [os.path.dirname(p.rstrip(os.sep)) or p
                      for p in get_recent_sources(APP_KEY, limit=4)]
        remembered += get_recent_sources("mask", limit=2)
        # This factory runs on the GUI thread -- `MainWindow._build_screen`
        # calls it inline, because Qt forbids building widgets anywhere else
        # -- and every path here is one the user typed at some other screen.
        # `os.path.isdir` on the maintainer's remembered `/nas_mnt` root had
        # not returned after TWENTY SECONDS on 2026-09-04, so this filter
        # froze the whole window for as long as the automount slept, and
        # opening the browser was reported as a crash with no traceback.
        # `exists(..., want_dir=True)` answers from the probe cache instead,
        # optimistically: `isdir()` would default to False and open the
        # browser empty on the first run of every session, which is the one
        # thing this seeding exists to prevent. Optimism costs nothing here
        # because the seeded roots go straight to `rescan`, whose walk runs
        # on a JobRunner worker -- a root that turns out to be gone is
        # discovered off the GUI thread and simply lists no projects.
        roots = tuple(dict.fromkeys(
            p for p in remembered
            if p and path_probe.exists(p, default=True, want_dir=True)))
    except Exception:
        LOG.debug("could not read the recent folders", exc_info=True)
    return ProjectBrowserScreen(roots=roots)


# The row this screen puts in the registry is declared in
# `spacr.qt.app_catalog`, which is what lets the app be registered without
# importing this module -- the launch reads the table, not the screen. These
# read the same row back rather than restating it, so the name, the blurb and
# the nine translations have one spelling and no second copy to drift from.
_ROW = declared_app(APP_KEY)
APP_NAME = _ROW.name
APP_DESCRIPTION = _ROW.desc
APP_INTRO = _ROW.intro
APP_CLI_NOTE = _ROW.cli_note
APP_NAME_TRANSLATIONS = _ROW.translations


def register() -> bool:
    """Put the Project Browser in the app registry. Idempotent.

    Called from :data:`spacr.qt.SELF_REGISTERING_MODULES`, which
    :func:`spacr.qt.run` runs after ``spacr.qt.app`` is fully executed and
    before ``MainWindow.__init__`` reads the registry.

    The row itself -- the key, the name, the blurb, the section, the "no
    headless run" sentence, the API doc link and the nine translations of the
    display name -- is declared in :mod:`spacr.qt.app_catalog`.
    :func:`spacr.qt.app.register_app` distributes those into the four tables
    each used to need a hand-edit in, and this function's whole job is to name
    which row. That is what lets the app be registered without importing this
    module at all: the launch reads the table, and the screen is imported when
    somebody opens it.

    :returns: ``True`` if this call is what registered it. Safe to call
        again: a module imported twice, or a test that re-imports it, must
        not raise on the duplicate key.
    """
    return register_declared(__name__) is not None
