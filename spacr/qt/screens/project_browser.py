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
from ..theme import SPACING, mark_surface
from .app_screen import ModuleHeader

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
        """Ask for a folder and add it."""
        start = self._roots[-1] if self._roots else os.path.expanduser("~")
        path = QFileDialog.getExistingDirectory(
            self, "Search this folder for spaCR projects", start)
        if path:
            self.add_root(path)

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
        self._jobs.submit(lambda r=roots, d=depth: _browse(r, d),
                          self._on_scanned)

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
        """A worker raised. Say so without a modal, and re-enable the button.

        Never a dialog: a browser that pops one up per unreadable folder is
        unusable on a machine with a stale network mount.
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
                item = QTableWidgetItem(str(text))
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
        roots = tuple(dict.fromkeys(p for p in remembered
                                    if p and os.path.isdir(p)))
    except Exception:
        LOG.debug("could not read the recent folders", exc_info=True)
    return ProjectBrowserScreen(roots=roots)


APP_NAME = "Project Browser"
APP_DESCRIPTION = "Every project on disk: stage, size, last run and what is stale"
APP_INTRO = (
    "Point it at the folder your experiments live in and it lists every "
    "spaCR project under it: how far each one got, what it costs on disk, "
    "when it last produced anything, and which of its results no longer "
    "match the data underneath them. A project spaCR has never recorded — a "
    "plate folder copied from a colleague this morning — is listed too, with "
    "everything the filesystem can answer; what it will not do is call it "
    "current, because with no run record there is nothing to compare it "
    "against. Nothing here is computed twice: the stage is which declared "
    "outputs exist, the size is the Data Manager's own walk, the staleness "
    "is the artifact registry's verdict, and the next step is the offer that "
    "module's own screen makes.")
APP_CLI_NOTE = (
    "Project Browser is a table you read and sort. Run it in the GUI "
    "(spacr-qt). Headless, spacr.projects.browse([root]) returns the same "
    "summaries and spacr.projects.format_projects prints the same table, so "
    "a nightly job can mail you which projects went stale.")
#: The display name in the nine non-English UI languages, in
#: `spacr.qt.i18n.LANGUAGES` order (sv, de, es, zh_CN, pt, hi, ko, is, fr).
APP_NAME_TRANSLATIONS = (
    "Projektbläddrare", "Projektbrowser", "Explorador de proyectos",
    "项目浏览器", "Navegador de projetos", "प्रोजेक्ट ब्राउज़र", "프로젝트 브라우저",
    "Verkefnavafri", "Navigateur de projets")


def register() -> bool:
    """Put the Project Browser in the app registry. Idempotent.

    Called from :data:`spacr.qt.SELF_REGISTERING_MODULES`, which
    :func:`spacr.qt.run` runs after ``spacr.qt.app`` is fully executed and
    before ``MainWindow.__init__`` reads the registry.

    Everything after ``SECTION_DATA`` is a table this key would otherwise
    need a hand-edit in: the screen header and blurb, the "no headless run"
    sentence, the API doc link and the nine translations of the display name.
    :func:`spacr.qt.app.register_app` distributes them from this one call.

    :returns: ``True`` if this call is what registered it. Safe to call
        again: a module imported twice, or a test that re-imports it, must
        not raise on the duplicate key.
    """
    from ..app import APPS, SECTION_DATA, STAGE_ALPHA, register_app
    if any(row[0] == APP_KEY for row in APPS):
        return False
    register_app(APP_KEY, APP_NAME, APP_DESCRIPTION, SECTION_DATA,
                 factory=make_project_browser_screen, stage=STAGE_ALPHA,
                 intro=APP_INTRO, cli_note=APP_CLI_NOTE,
                 api_module="qt/screens/project_browser",
                 translations=APP_NAME_TRANSLATIONS)
    return True
