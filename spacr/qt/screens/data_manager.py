"""The Data Manager screen — what the project costs, and how to get it back.

Three surfaces over :mod:`spacr.data_manager`, in the order a user needs
them:

**Usage** — one row per artifact kind, measured by walking the project, with
the unregistered bytes called out. The row that is always the surprise is
"unregistered": bytes spaCR did not put there and cannot account for, which
is also why none of them is ever offered for deletion.

**Prune** — the plan, in full, *before* anything happens. Every candidate
with its size and the module that would make it again; every kept item with
the rule that kept it, because a user who expected 300 GB back and was
offered 12 needs to read why rather than guess. Deleting requires the
confirmation dialog, and the dialog shows the file list and the total one
more time.

**Archive** — move the project, or part of it, somewhere else, and leave the
record that says where it went.

The screen never deletes anything itself: it holds a
:class:`spacr.data_manager.PrunePlan` and hands it back with its own token.
A plan that has gone stale — because a run wrote into the project while the
screen was open — is refused by the module, not by this file, which is where
that check belongs.

Registered through :func:`spacr.qt.app.register_app` and
:func:`spacr.qt.theme.register_widget_qss` rather than by editing ``app.py``
or ``theme.py``.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QDialogButtonBox, QFileDialog,
    QFrame, QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QTabWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from ... import data_manager as dm
from ...ports import ALL_KINDS
from .. import path_probe
from ..theme import (SPACING, block_surface, font_px,
                     register_widget_qss)
from .app_screen import ModuleHeader
from ..widgets.toggle import Toggle
from ..widgets.sortable_table import install_sorting, table_item
from ..app_catalog import register_declared

LOG = logging.getLogger("spacr.qt.screens.data_manager")

__all__ = ["DataManagerScreen", "ConfirmDeleteDialog", "APP_KEY",
           "make_data_manager_screen", "register"]

#: The app key. Chosen once and never renamed — saved user state, the command
#: palette and ``spacr-qt data_manager`` all key off it.
APP_KEY = "data_manager"

_USAGE_COLUMNS = ("Kind", "Size", "Files", "Registered", "Unregistered",
                  "Note")
_PRUNE_COLUMNS = ("Free", "Kind", "Where", "Files", "Get it back by")
_KEPT_COLUMNS = ("Size", "Where", "Why it is kept")
_ARCHIVE_COLUMNS = ("Size", "Kind", "From", "To")

#: Kinds the prune tab offers a checkbox for. Originals are absent because
#: there is no code path that deletes them, and a checkbox that cannot do
#: anything is worse than no checkbox.
_OFFERED_KINDS = tuple(k for k in ALL_KINDS if k not in dm.ORIGINAL_KINDS)


def _data_manager_qss(palette: dict, opacity) -> str:
    """QSS for this screen, registered through the theme seam.

    Only two things are styled: the totals strip, which has to read as a
    summary rather than as another row of the table under it, and the delete
    button, which is the one control on this screen that destroys data and
    must not look like the others. Everything else is deliberately left to
    the shipped stylesheet — a screen that restyles tables is a screen that
    stops following the theme.
    """
    surface = block_surface("surface_alt", palette["theme"], opacity)
    return f"""
QFrame#DataManagerTotals {{
    background: {surface};
    border: 1px solid {palette["border_soft"]};
    border-radius: 8px;
}}
QLabel#DataManagerTotal {{
    font-weight: 600;
    font-size: {font_px(15)}px;
}}
QLabel#DataManagerNote[warn="true"] {{
    color: {palette["error"]};
}}
QPushButton#DataManagerDelete {{
    border: 1px solid {palette["error"]};
    color: {palette["error"]};
    font-weight: 600;
}}
QPushButton#DataManagerDelete:disabled {{
    border: 1px solid {palette["border_soft"]};
    color: {palette["fg_muted"]};
    font-weight: 400;
}}
"""


# ``replace=True`` because this module owns the name: a reimport must
# re-register the same block rather than raise and leave the screen unstyled.
register_widget_qss("DataManager", _data_manager_qss, replace=True)


# ---------------------------------------------------------------------------
# The confirmation
# ---------------------------------------------------------------------------

class ConfirmDeleteDialog(QDialog):
    """The last thing between a plan and an irreversible deletion.

    It shows the total, the item list and every file that would go — the
    same list :meth:`spacr.data_manager.PrunePlan.file_list` returns, not a
    summary of it — and its accept button stays disabled until the user
    ticks the box that says they have read it. The box is deliberately not
    pre-ticked and deliberately not a plain OK: this data is somebody's
    experiment and there is no undo.

    IT FILLS IN TWO STAGES, and the reason is the whole point of this
    exercise. The file list is not stored on the plan: every call to
    ``file_list`` runs ``os.walk`` over every candidate directory, because a
    plan for a project with millions of crops must not carry millions of
    strings. Doing that in ``__init__`` put a full recursive walk of the
    user's project on the GUI thread, at the one moment they are waiting for
    a window to appear — and on a share that is asleep it is the walk's FIRST
    stat that spends twenty seconds waking the mount, not the walk. So the
    dialog opens with the half that needs nothing from the disk (the totals,
    every candidate, every kept item and its reason) and the file list
    arrives behind it. Nothing is dropped: the user still reads every path
    before anything is deleted, which is why the acknowledgement is held
    disabled until the list is on screen — or, when the walk fails, until
    :meth:`_on_listing_failed` has said so in the list's place. The one
    thing that never happens is that box being armed over a placeholder.

    :param plan: the plan to confirm.
    :param parent: Qt parent.
    :param threaded: enumerate the files off the GUI thread. ``False`` walks
        inline, so a test has the finished dialog when the constructor
        returns; both paths produce the same text and the same enabled
        states.
    """

    #: What stands in for the file list until the walk comes back. It says
    #: why the acknowledgement below it will not move yet, because a control
    #: that is greyed for an unstated reason reads as a broken one.
    READING = ("  Reading the disk — the list of every file this would "
               "delete is being gathered.\n"
               "  Delete stays locked until it is here.")

    #: How long :meth:`_stop_the_file_list` gives the walk to stop before it
    #: lets go. Short enough that Cancel never reads as a hang, long enough
    #: that a walk which has already finished is reaped normally rather than
    #: parked. It is not a deadline for the walk — nothing can interrupt an
    #: ``os.walk`` — only for how long this thread is willing to watch it.
    TEARDOWN_GRACE_MS = 100

    def __init__(self, plan: "dm.PrunePlan", parent=None, *,
                 threaded: bool = True) -> None:
        """Ask the user to confirm a deletion, in words they must read.

        :param plan: what is about to be deleted.
        :param parent: parent widget.
        :param threaded: whether the deletion runs on a worker.
        """
        super().__init__(parent)
        self.plan = plan
        self._runner = None
        self.setObjectName("DataManagerConfirm")
        self.setWindowTitle("Delete regenerable data")
        self.setModal(True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["md"], SPACING["md"],
                                 SPACING["md"], SPACING["md"])
        outer.setSpacing(SPACING["sm"])

        headline = QLabel(
            f"This deletes {len(plan.candidates)} item(s), "
            f"{plan.total_files:,} files, and frees "
            f"{dm.human_bytes(plan.total_bytes)}.", self)
        headline.setObjectName("DataManagerTotal")
        headline.setWordWrap(True)
        outer.addWidget(headline)

        warning = QLabel(
            "It cannot be undone. Everything listed can be produced again by "
            "re-running the module named beside it; nothing spaCR cannot "
            "account for is in this list.", self)
        warning.setWordWrap(True)
        outer.addWidget(warning)

        self.listing = QPlainTextEdit(self)
        self.listing.setObjectName("DataManagerFileList")
        self.listing.setReadOnly(True)
        # The heading only. `describe()` walks the project, and this is the
        # GUI thread.
        self.listing.setPlainText(
            f"{self._heading()}\n\nFiles:\n{self.READING}")
        outer.addWidget(self.listing, 1)

        self.acknowledged = Toggle(
            "I have read the list above and want these files deleted", self)
        self.acknowledged.setObjectName("DataManagerAcknowledge")
        # Nobody can have read a list that is not on screen yet.
        self.acknowledged.setEnabled(False)
        outer.addWidget(self.acknowledged)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.button(QDialogButtonBox.Ok).setText("Delete")
        self.buttons.button(QDialogButtonBox.Ok).setObjectName(
            "DataManagerDelete")
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.acknowledged.toggled.connect(self._on_acknowledged)
        outer.addWidget(self.buttons)
        self.resize(760, 560)
        # Last, because its completion handler touches every widget above
        # and, unthreaded, it runs before this line returns.
        self._start_the_file_list(threaded)

    def describe(self) -> str:
        """The whole text — the plan, then every file — walked right now.

        WALKS THE PROJECT, AND THE DIALOG NO LONGER CALLS IT.
        :meth:`spacr.data_manager.PrunePlan.file_list` runs ``os.walk`` over
        every candidate each time it is asked, which is the twenty seconds
        this exercise removed, so nothing on the GUI thread may call this.
        The dialog builds the same string in two pieces instead:
        :meth:`_heading` at once, then :meth:`_with_the_files` on the walk
        that :meth:`_start_the_file_list` sent to a worker.

        It stays public, and blocking, because the whole text in one call is
        what a caller outside Qt wants — a test, or a CLI that has a thread
        to spare. Read :attr:`listing` for what is actually on screen.
        """
        return self._with_the_files(*self.plan.file_list())

    def _heading(self) -> str:
        """The part of the text that needs nothing from the disk.

        :func:`spacr.data_manager.format_prune_plan` reads the plan the
        screen is already holding — the totals, every candidate with the
        module that would make it again, and every kept item with the rule
        that kept it. All of it can be on screen before the walk starts.
        """
        return dm.format_prune_plan(self.plan)

    def _with_the_files(self, files, truncated: bool) -> str:
        """The heading and then the file list, as the dialog shows them.

        :param files: the paths the walk found, in plan order.
        :param truncated: True when the plan holds more files than
            :data:`spacr.data_manager.MAX_RECORDED_FILES` and the list was
            cut short.
        """
        lines = [self._heading(), "", "Files:"]
        lines.extend(f"  {path}" for path in files)
        if truncated:
            lines.append(f"  … and more; over {dm.MAX_RECORDED_FILES:,} "
                         f"files, the list is cut short. The totals above "
                         f"cover all of them.")
        return "\n".join(lines)

    # -- the file list, off the GUI thread ---------------------------------

    def _start_the_file_list(self, threaded: bool) -> None:
        """Enumerate the files behind the dialog instead of in front of it.

        Its own :class:`spacr.qt.job_runner.JobRunner` rather than the
        screen's ``_run``: that one carries the scan, the plan, the prune and
        the archive, refuses a second job while one is in flight, and marks
        the screen busy — a listing that greyed the screen behind its own
        modal dialog would be a new defect, not a fixed one.

        ``user_visible=False`` because this runner carries nothing else. The
        walk is housekeeping for one dialog, and Home filters its run banners
        on exactly that flag; without it opening a confirmation flashes
        "data_manager — running" at a user who started no run.

        :param threaded: False to walk inline, for tests.
        """
        if not self.plan.candidates:
            # Nothing to walk. `file_list` returns an empty tuple without
            # touching the disk, and a thread costs more than the answer.
            self._show_the_files(((), False))
            return

        from ..job_runner import JobRunner

        self._runner = JobRunner(self, threaded=bool(threaded),
                                 app_key=APP_KEY, user_visible=False)
        self._runner.job_failed.connect(self._on_listing_failed)
        self._runner.submit(self.plan.file_list, self._show_the_files)

    def _show_the_files(self, found) -> None:
        """Replace the placeholder with the list the walk came back with.

        :param found: the ``(paths, truncated)`` pair
            :meth:`spacr.data_manager.PrunePlan.file_list` returns.
        """
        files, truncated = found
        self.listing.setPlainText(self._with_the_files(files, truncated))
        self.acknowledged.setEnabled(True)

    def _on_listing_failed(self, message: str) -> None:
        """Say the walk failed rather than leave "Reading the disk" up.

        A runner hands a result to its handler only for a job that
        succeeded, so a placeholder cleared only there stays on screen for
        good when the worker raises. The plan itself is still readable and
        still true — it is what the totals were computed from — so the
        acknowledgement is released rather than the deletion refused, and
        the line says exactly which half is missing.

        :param message: the worker's one-line message.
        """
        if self._runner is None:
            # The dialog has already let go of the walk; this is a failure
            # arriving after the shutdown that abandoned it.
            return
        self.listing.setPlainText(
            f"{self._heading()}\n\nFiles:\n"
            f"  The file list could not be read: {message}\n"
            f"  Everything above comes from the plan and still holds.")
        self.acknowledged.setEnabled(True)

    def _stop_the_file_list(self) -> None:
        """Retire the walk's thread. Safe to call more than once.

        WITHOUT WAITING FOR THE WALK, and that is the point. The default
        :meth:`spacr.qt.job_runner.JobRunner.shutdown` budget is three
        seconds, and spending it here put a fresh freeze on Cancel: pressing
        it while the walk was out held the GUI thread for the full three
        seconds before the dialog would close.

        The wait cannot even succeed. :func:`spacr.qt.bridge.drain_thread`
        stops a thread by asking its EVENT LOOP to quit, and this worker is
        not in an event loop — it is inside ``os.walk``, which has no
        interruption point and will not return until the filesystem answers.
        So the three seconds always elapsed in full and always ended in the
        same place: `drain_thread` parking the thread, which is what makes
        letting go of it safe, and which it does just as well at once.

        Parking is safe here for the reason Qt aborts otherwise — the
        process-wide park list keeps a strong reference, so nothing drops
        the last reference to a running QThread. Nor is the dialog being
        destroyed at this point: ``QDialog.done`` hides it and it stays
        parented to the screen, so there is no destruction deadline to beat.
        """
        runner = self._runner
        self._runner = None
        if runner is not None:
            try:
                runner.shutdown(self.TEARDOWN_GRACE_MS)
            except RuntimeError:
                # The runner's C++ half has gone with the dialog. The
                # threads are still drained by `job_runner.shutdown_all`
                # on the way out of the application.
                pass

    def done(self, result: int) -> None:
        """Close the dialog, having first stopped the walk.

        Both Delete and Cancel come through here and NEITHER sends a close
        event — ``QDialog.done`` hides the widget — so this, not
        ``closeEvent``, is where a modal dialog's teardown has to live. Qt
        aborts the process outright if a running ``QThread`` is destroyed
        with its owner, and the walk can still be out when the user cancels.

        :param result: the dialog code to finish with.
        """
        self._stop_the_file_list()
        super().done(result)

    def closeEvent(self, event):        # noqa: N802 - Qt name
        """The same teardown for the window's own close button.

        :param event: the close event.
        """
        self._stop_the_file_list()
        super().closeEvent(event)

    def _on_acknowledged(self, checked: bool) -> None:
        """Enable the delete button only once the warning is acknowledged.

        DELETION IS NOT UNDOABLE HERE, so the confirmation is a deliberate
        second action rather than a default-focused OK button.

        :param checked: True when the box is ticked.
        """
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(bool(checked))


# ---------------------------------------------------------------------------
# The screen
# ---------------------------------------------------------------------------

class DataManagerScreen(QWidget):
    """Disk usage, pruning and archiving for one project.

    :param parent: Qt parent.
    :param project: open straight onto this project root, skipping the
        folder picker.
    :param threaded: run the scan and the plan on a worker thread. Tests
        pass False so a scan is finished when the call returns; both paths
        run the same code and emit the same signals.
    """

    #: Emitted after a scan, a plan, a prune or an archive settles.
    #: ``True`` when it succeeded. Tests wait on this; nothing else needs it.
    job_finished = Signal(bool)

    def __init__(self, parent=None, *, project: str = "",
                 threaded: bool = True) -> None:
        """Build the manager's three tabs and its totals row.

        :param parent: parent widget.
        """
        super().__init__(parent)
        self.setObjectName("DataManagerScreen")
        self._threaded = bool(threaded)
        self._root = str(project or "")
        self._usage: Optional[dm.ProjectUsage] = None
        self._plan: Optional[dm.PrunePlan] = None
        self._archive_plan: Optional[dm.ArchivePlan] = None
        self._destination = ""
        #: Folders a file dialog on this screen has handed back. They are
        #: the one class of path this screen may believe in without a stat
        #: of its own -- the dialog reached them to return them -- and
        #: `_dialog_start` needs that because `path_probe.prime` cannot
        #: record an `isdir` answer. Bounded by how many times a user
        #: presses the two Choose buttons.
        self._picked_dirs: set = set()
        self._jobs: List[Any] = []
        self._pending: Any = ({}, None)
        self._busy = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["md"], SPACING["md"],
                                 SPACING["md"], SPACING["md"])
        outer.setSpacing(SPACING["sm"])

        outer.addLayout(self._build_head())
        outer.addWidget(self._build_totals())

        self.tabs = QTabWidget(self)
        self.tabs.setObjectName("DataManagerTabs")
        self.tabs.addTab(self._build_usage_tab(), "Usage")
        self.tabs.addTab(self._build_prune_tab(), "Prune")
        self.tabs.addTab(self._build_archive_tab(), "Archive")
        outer.addWidget(self.tabs, 1)

        self._follow_path_probes()
        self._update_controls()
        if self._root:
            self.scan()
        # Drop anywhere on this screen: the path is resolved through spaCR's
        # project layout, so the plate folder finds what this screen reads.
        from ..dnd import install_for
        install_for(self, "data_manager")

    # -- construction -----------------------------------------------------

    def _build_head(self) -> QHBoxLayout:
        """Build the project row and the refresh control."""
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(SPACING["sm"])

        header = ModuleHeader(
            "Data Manager",
            description="What is on disk for this project, and what it cost",
            instruction="Choose a project, then rescan to remeasure it.",
        )
        self._header = header
        head.addWidget(header)

        self.project_label = QLabel(self._root or "no project chosen", self)
        self.project_label.setObjectName("DataManagerProject")
        self.project_label.setToolTip("The project being measured")
        head.addWidget(self.project_label, 1)

        choose = QPushButton("Choose project…", self)
        choose.setToolTip("The plate folder — the one holding merged/ and "
                          "measurements/")
        choose.clicked.connect(self.choose_project)
        head.addWidget(choose)

        self.rescan_button = QPushButton("Rescan", self)
        self.rescan_button.setObjectName("PrimaryButton")
        self.rescan_button.setToolTip("Walk the project again and remeasure")
        self.rescan_button.clicked.connect(self.scan)
        head.addWidget(self.rescan_button)
        return head

    def _build_totals(self) -> QFrame:
        """Build the totals strip over the tabs."""
        frame = QFrame(self)
        frame.setObjectName("DataManagerTotals")
        row = QHBoxLayout(frame)
        row.setContentsMargins(SPACING["sm"], SPACING["sm"],
                               SPACING["sm"], SPACING["sm"])
        row.setSpacing(SPACING["md"])

        self.total_label = QLabel("—", frame)
        self.total_label.setObjectName("DataManagerTotal")
        row.addWidget(self.total_label)

        self.note_label = QLabel("", frame)
        self.note_label.setObjectName("DataManagerNote")
        self.note_label.setWordWrap(True)
        row.addWidget(self.note_label, 1)

        self.progress = QProgressBar(frame)
        self.progress.setObjectName("DataManagerProgress")
        self.progress.setRange(0, 0)
        self.progress.setMaximumWidth(140)
        self.progress.setVisible(False)
        row.addWidget(self.progress)
        return frame

    @staticmethod
    def _table(name: str, columns) -> QTableWidget:
        """One configured results table.

        :param name: the table's name.
        :param columns: the column headings.
        :returns: the table widget.
        """
        table = QTableWidget(0, len(columns))
        install_sorting(table)
        table.setObjectName(name)
        table.setHorizontalHeaderLabels(list(columns))
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        return table

    def _build_usage_tab(self) -> QWidget:
        """Build the tab showing what the project is using."""
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, SPACING["sm"], 0, 0)
        layout.setSpacing(SPACING["sm"])

        self.usage_table = self._table("DataManagerUsage", _USAGE_COLUMNS)
        layout.addWidget(self.usage_table, 1)

        self.usage_note = QLabel(
            "Unregistered bytes are files spaCR has no record of producing. "
            "They are measured, and they are never deleted.", page)
        self.usage_note.setWordWrap(True)
        layout.addWidget(self.usage_note)
        return page

    def _build_prune_tab(self) -> QWidget:
        """Build the tab that proposes what can be deleted."""
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, SPACING["sm"], 0, 0)
        layout.setSpacing(SPACING["sm"])

        picker = QHBoxLayout()
        picker.setContentsMargins(0, 0, 0, 0)
        picker.setSpacing(SPACING["sm"])
        picker.addWidget(QLabel("Consider:", page))
        self.kind_boxes: Dict[str] = {}
        for kind in _OFFERED_KINDS:
            box = Toggle(dm.KIND_LABELS.get(kind, kind), page)
            box.setObjectName(f"DataManagerKind_{kind}")
            box.setChecked(kind in dm.DEFAULT_PRUNABLE_KINDS)
            if kind in dm.PROTECTED_KINDS:
                box.setToolTip(
                    "Kept by default. Regenerable in principle, but "
                    "expensive or irreplaceable in practice — tick it only "
                    "if you mean it.")
            box.toggled.connect(self._on_kinds_changed)
            self.kind_boxes[kind] = box
            picker.addWidget(box)
        picker.addStretch(1)
        layout.addLayout(picker)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(SPACING["sm"])
        self.plan_button = QPushButton("Show what can be deleted", page)
        self.plan_button.setObjectName("PrimaryButton")
        self.plan_button.clicked.connect(self.plan_prune)
        buttons.addWidget(self.plan_button)

        self.freed_label = QLabel("No plan yet.", page)
        self.freed_label.setObjectName("DataManagerFreed")
        buttons.addWidget(self.freed_label, 1)

        self.delete_button = QPushButton("Delete…", page)
        self.delete_button.setObjectName("DataManagerDelete")
        self.delete_button.setToolTip(
            "Shows the full list and asks again. There is no undo.")
        self.delete_button.setEnabled(False)
        self.delete_button.clicked.connect(self.confirm_and_prune)
        buttons.addWidget(self.delete_button)
        layout.addLayout(buttons)

        self.prune_table = self._table("DataManagerPrune", _PRUNE_COLUMNS)
        layout.addWidget(self.prune_table, 2)

        layout.addWidget(QLabel("Kept, and why:", page))
        self.kept_table = self._table("DataManagerKept", _KEPT_COLUMNS)
        layout.addWidget(self.kept_table, 1)
        return page

    def _build_archive_tab(self) -> QWidget:
        """Build the tab that proposes what can be archived."""
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, SPACING["sm"], 0, 0)
        layout.setSpacing(SPACING["sm"])

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING["sm"])
        self.destination_label = QLabel("no destination chosen", page)
        self.destination_label.setObjectName("DataManagerDestination")
        row.addWidget(self.destination_label, 1)

        choose = QPushButton("Choose destination…", page)
        choose.clicked.connect(self.choose_destination)
        row.addWidget(choose)

        self.archive_plan_button = QPushButton("Show what would move", page)
        self.archive_plan_button.setObjectName("PrimaryButton")
        self.archive_plan_button.clicked.connect(self.plan_archive)
        row.addWidget(self.archive_plan_button)

        self.archive_button = QPushButton("Move…", page)
        self.archive_button.setEnabled(False)
        self.archive_button.clicked.connect(self.confirm_and_archive)
        row.addWidget(self.archive_button)
        layout.addLayout(row)

        self.archive_table = self._table("DataManagerArchive",
                                         _ARCHIVE_COLUMNS)
        layout.addWidget(self.archive_table, 1)

        note = QLabel(
            "Archiving moves the files and leaves a record: a manifest at "
            "the destination, a log at the origin, and registry rows at the "
            "destination carrying the provenance the artifacts arrived with. "
            "Nothing is overwritten.", page)
        note.setWordWrap(True)
        layout.addWidget(note)
        return page

    # -- project ----------------------------------------------------------

    @property
    def project(self) -> str:
        """The project root this screen is showing."""
        return self._root

    @property
    def usage(self) -> Optional["dm.ProjectUsage"]:
        """The last scan, or None."""
        return self._usage

    @property
    def plan(self) -> Optional["dm.PrunePlan"]:
        """The last prune plan, or None."""
        return self._plan

    def _dialog_start(self, remembered: str) -> str:
        """Where a folder picker should open, without stat-ing to find out.

        The remembered path is only a convenience, and that convenience is
        not worth a stat on the GUI thread: ``os.path.isdir`` on a path
        behind a sleeping ``autofs`` mount can take twenty seconds to
        return, because the stat is what triggers the automount, and handing
        that path to :class:`QFileDialog` spends the same twenty seconds
        inside the dialog instead. So the question goes to
        :mod:`spacr.qt.path_probe`, which answers from cache and never waits.

        THE DEFAULT IS THE WHOLE DESIGN HERE. ``path_probe`` has to be told
        what to say while a path is still being probed, and the two wrong
        answers cost very different amounts:

        * A path nobody has vouched for — a root restored from last session,
          say — defaults to MISSING, so the picker opens at home. Being
          wrong costs the user one click; being wrong the other way parks
          the application on a mount that is asleep.
        * A path a picker has just handed back defaults to PRESENT, because
          the dialog only returns directories it reached. Nothing is being
          taken on trust: the file dialog is the stat, and it already ran.

        The second case is not a refinement, it is the bug this method was
        extracted to fix. :func:`spacr.qt.path_probe.prime` records the plain
        "does it exist" answer, under the cache key ``(path, False)``, and
        the question a start directory asks is ``isdir`` — key
        ``(path, True)``. So priming did not answer the question that gets
        asked, and the SECOND press of "Choose destination…" reopened at home
        having forgotten the folder the first press landed on. ``_root`` hid
        the same fault behind a race, because :meth:`_update_controls` probes
        it with ``want_dir=True`` anyway and usually wins; ``_destination``
        is asked about nowhere else, so it never recovered at all.

        :param remembered: the path last used for this picker, if any.
        :returns: ``remembered`` when it is safe to open there, else the
            user's home directory.
        """
        if not remembered:
            return os.path.expanduser("~")
        # `exists(want_dir=True)`, not `isdir()`, only so the default can be
        # chosen per path; the question and the cache key are identical.
        if path_probe.exists(remembered, want_dir=True,
                             default=remembered in self._picked_dirs):
            return remembered
        return os.path.expanduser("~")

    def _remember_picked(self, path: str) -> None:
        """Record a folder a file dialog just returned, so it can be reopened.

        Two records, because they answer two different questions and
        ``path_probe`` keys them separately: :func:`spacr.qt.path_probe.prime`
        for "does it exist", which is what every other widget asks, and
        :attr:`_picked_dirs` for "is it a directory", which is what
        :meth:`_dialog_start` asks and which ``prime`` cannot express.

        :param path: the folder the dialog returned.
        """
        path_probe.prime(path, True)
        self._picked_dirs.add(str(path))

    def choose_project(self) -> None:
        """Ask for a project folder and scan it."""
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose a spaCR project", self._dialog_start(self._root))
        if chosen:
            # The dialog has just proved this folder is there, so record it
            # rather than have the cache learn it again.
            self._remember_picked(chosen)
            self.set_project(chosen)

    def set_project(self, root: str) -> None:
        """Point the screen at ``root`` and scan it."""
        self._root = str(root)
        self._usage = None
        self._plan = None
        self._archive_plan = None
        self.project_label.setText(self._root)
        self._clear_tables()
        self.scan()

    def choose_destination(self) -> None:
        """Ask where an archive should go."""
        # Same reasoning as choose_project, and the same two calls: the last
        # destination is a hint about where to open, never a reason to wait
        # on a filesystem.
        chosen = QFileDialog.getExistingDirectory(
            self, "Archive this project into",
            self._dialog_start(self._destination))
        if chosen:
            self._remember_picked(chosen)
            self.set_destination(chosen)

    def set_destination(self, path: str) -> None:
        """Set the archive destination without touching anything."""
        self._destination = str(path)
        self.destination_label.setText(self._destination)
        self._archive_plan = None
        self.archive_table.setRowCount(0)
        self._update_controls()

    def _clear_tables(self) -> None:
        """Empty every table, so a stale plan is not read as a current one."""
        for table in (self.usage_table, self.prune_table, self.kept_table,
                      self.archive_table):
            table.setRowCount(0)
        self.total_label.setText("—")
        self.freed_label.setText("No plan yet.")

    # -- jobs -------------------------------------------------------------

    def _run(self, fn, on_done) -> bool:
        """Run ``fn`` off the GUI thread and hand the result to ``on_done``.

        ``PipelineWorker.finished`` is emitted *in the worker thread*, and
        PySide6 invokes a plain closure connected to it on that same thread.
        This screen's completion handlers fill QTableWidgets, and building
        widget children off the GUI thread is undefined behaviour — so
        ``finished`` is chained through :attr:`job_settled`, a *bound method*
        of this widget, which has GUI-thread affinity. Qt then queues the
        call and the handler runs where every other widget call runs.

        With ``threaded=False`` the call runs inline and the same signals
        fire, so both paths behave identically from outside.
        """
        if self._busy:
            return False
        if not self._threaded:
            box: Dict[str, Any] = {}
            ok = True
            self._set_busy(True)
            try:
                box["result"] = fn()
            except Exception as exc:              # noqa: BLE001 - reported
                self._set_busy(False)
                self._on_job_error(exc)
                self.job_finished.emit(False)
                return False
            self._settle(box, on_done, ok)
            return True

        from ..bridge import make_thread

        box = {}

        def _job(payload: Dict[str, Any]) -> None:
            """Call the wrapped function, stashing its result in the payload.

            The payload is how a value crosses back from the worker: a return would
            be swallowed by the runner.
            """
            payload["result"] = fn()

        thread, worker = make_thread(_job, box, journal=False)
        # Strong references: PySide6 will not keep the worker alive through
        # the started→run connection alone, and a QThread garbage-collected
        # while still running takes the process down with it.
        self._jobs.append((thread, worker))
        self._pending = (box, on_done)
        worker.error.connect(self._on_worker_error_text)
        worker.finished.connect(self._job_settled)
        thread.finished.connect(self._retire_finished_job)
        self._set_busy(True)
        thread.start()
        return True

    def _job_settled(self, ok: bool) -> None:
        """Finish the in-flight job. Always on the GUI thread."""
        box, on_done = self._pending
        self._pending = ({}, None)
        self._settle(box, on_done, bool(ok))

    def _settle(self, box: Dict[str, Any], on_done, ok: bool) -> None:
        """Hand a finished job's result to its handler and announce it.

        Busy is cleared *before* the handler runs, not after: a handler that
        starts the next job — a prune re-scans the project it just changed —
        would otherwise find the screen still marked busy and its call would
        be dropped. Clearing afterwards would also un-busy the job the
        handler just started. Both paths, threaded and inline, go through
        here so they cannot drift apart.
        """
        self._set_busy(False)
        if ok and on_done is not None:
            try:
                on_done(box.get("result"))
            except Exception as exc:              # noqa: BLE001 - reported
                self._on_job_error(exc)
                ok = False
        self.job_finished.emit(ok)

    def _retire_finished_job(self) -> None:
        """Release this job's references, on this widget's GUI thread."""
        thread = self.sender()
        self._jobs = [(t, w) for (t, w) in self._jobs if t is not thread]

    def _on_worker_error_text(self, text: str) -> None:
        """Show a worker's error text without closing the screen.

        :param text: what went wrong.
        """
        line = (text or "").strip().splitlines()[-1] if text else "unknown error"
        self._note(line, warn=True)

    def _on_job_error(self, exc: Exception) -> None:
        """Report a failed background scan.

        :param exc: what went wrong.
        """
        LOG.info("data manager job failed", exc_info=True)
        self._note(str(exc) or exc.__class__.__name__, warn=True)

    def _set_busy(self, busy: bool) -> None:
        """Disable the controls while a scan or a deletion is running.

        :param busy: True while work is outstanding.
        """
        self._busy = bool(busy)
        self.progress.setVisible(self._busy)
        self._update_controls()

    def _note(self, text: str, *, warn: bool = False) -> None:
        """Put one line in the status area.

        :param text: the line.
        """
        self.note_label.setText(text)
        self.note_label.setProperty("warn", "true" if warn else "false")
        style = self.note_label.style()
        if style is not None:
            style.unpolish(self.note_label)
            style.polish(self.note_label)

    def _update_controls(self) -> None:
        # self._root is whatever folder the user picked or dropped, and this
        # runs on construction, on every checkbox change and on every job
        # settle. A bare os.path.isdir here was therefore a stat on the GUI
        # thread with a user-supplied path: measured 2026-09-04, one on a
        # sleeping /nas_mnt autofs share had not returned after twenty
        # seconds, and a stalled event loop is a freeze with no traceback.
        # Optimistic while the probe is out -- everything these controls
        # start hands the real question to a worker, which reports a bad root
        # through _on_job_error -- and _follow_path_probes greys them once the
        # answer lands.
        """Enable the actions for whatever root is set.

        The root is a user-supplied path and this runs on construction, on every
        tick box and on every job settling -- so it asks the cached probe rather
        than stat-ing directly. One bare stat on a sleeping autofs share had not
        returned after twenty seconds, and a stalled event loop is a freeze with
        no traceback. It is optimistic while the probe is out, because
        everything these controls start hands the real question to a worker
        which reports a bad root itself.
        """
        has_project = bool(self._root) and path_probe.exists(
            self._root, want_dir=True, default=True)
        self.rescan_button.setEnabled(has_project and not self._busy)
        self.plan_button.setEnabled(has_project and not self._busy)
        self.delete_button.setEnabled(
            bool(self._plan and self._plan.candidates) and not self._busy)
        self.archive_plan_button.setEnabled(
            has_project and bool(self._destination) and not self._busy)
        self.archive_button.setEnabled(
            bool(self._archive_plan and self._archive_plan.items)
            and not self._busy)

    def _follow_path_probes(self) -> None:
        """Grey the controls when a background path check finally answers.

        `path_probe` reports a path it has not seen as PRESENT so that asking
        never blocks, which means this screen opens with Rescan and Plan live
        on a root that may have gone since the last session. This is the half
        that corrects it; without it the buttons stay lit until something
        else happens to refresh them.

        `probes.answered` is process-wide and outlives any one screen, so the
        slot swallows the RuntimeError raised when the Python wrapper is
        still here and the C++ widget is not. `closeEvent` disconnects it;
        the guard covers the window before that.

        IT IS ALSO PROCESS-WIDE IN THE OTHER SENSE: the signal fires for
        every path anything in spaCR probes, and `file_list.py` alone probes
        every remembered path in the application at start-up. The only path
        `_update_controls` reads through `path_probe` is `self._root`, so an
        answer about anything else is re-running the enable pass to reach the
        identical conclusion. Answering only for this screen's own root is
        not a dropped refresh -- there is nothing in it to drop.
        """
        def redraw(path: str, _answer: bool) -> None:
            """Re-run the enable pass now that this root's state is known.

            :param path: the path whose probe just answered.
            :param _answer: what it answered; unused, because
                `_update_controls` reads it back from the cache along with
                everything else it depends on.
            """
            # `getattr`, not `self._root`, and for the reason spelled out in
            # `spacr.qt.dnd._DropzoneFilter.eventFilter`: PySide6 CLEARS the
            # Python wrapper's __dict__ when the C++ widget goes, so a plain
            # attribute read on a dead screen raises AttributeError rather
            # than the RuntimeError this guard is named for -- and it raises
            # it inside the Qt event loop, where no caller can catch it.
            try:
                if getattr(self, "_root", None) != path:
                    return
                self._update_controls()
            except RuntimeError:
                # The screen has gone; the signal outlived it. The enable
                # pass touches widgets, so this is where that lands.
                pass

        # Held on the instance because the connection alone does not keep a
        # plain closure alive.
        self._path_probe_redraw = redraw
        path_probe.probes.answered.connect(redraw)

    # -- scanning ---------------------------------------------------------

    def selected_kinds(self) -> List[str]:
        """The kinds the prune tab is currently asking about."""
        return [kind for kind, box in self.kind_boxes.items()
                if box.isChecked()]

    def _on_kinds_changed(self, _checked: bool) -> None:
        """A changed selection invalidates the plan it was not made with."""
        self._plan = None
        self.prune_table.setRowCount(0)
        self.kept_table.setRowCount(0)
        self.freed_label.setText("Kinds changed — plan again.")
        self._update_controls()

    def scan(self) -> bool:
        """Measure the project. Off the GUI thread unless ``threaded=False``."""
        root = self._root
        # A guard against no project at all, not an authority on this one:
        # dm.scan_project runs in the worker and is what genuinely fails on a
        # root that is not there. exists(want_dir=True) rather than
        # path_probe.isdir because isdir answers False for a path nobody has
        # probed yet, which would refuse the very first scan of a folder the
        # user just chose.
        if not root or not path_probe.exists(root, want_dir=True,
                                             default=True):
            self._note("Choose a project folder first.", warn=True)
            return False
        self._note("")

        def measure():
            """Measure ``root``, having first made sure it is a folder.

            The isdir the guard above can no longer do lives here, where the
            thread waiting on it is a worker and waiting is free. The sentence
            it raises is the one the guard has always said, because a root
            that is not a folder has to read the same to the user however
            spaCR found out.
            """
            if not os.path.isdir(root):
                raise NotADirectoryError("Choose a project folder first.")
            return dm.scan_project(root)

        return self._run(measure, self._show_usage)

    def _show_usage(self, usage: "dm.ProjectUsage") -> None:
        """Fill the usage tab from a finished scan.

        :param usage: the scan's result.
        """
        self._usage = usage
        self.total_label.setText(
            f"{dm.human_bytes(usage.total_bytes)} in "
            f"{usage.total_files:,} files")
        notes = []
        if usage.unregistered_bytes:
            notes.append(f"{dm.human_bytes(usage.unregistered_bytes)} in "
                         f"{usage.unregistered_files:,} files has no registry "
                         f"record and is never deleted")
        if usage.missing:
            notes.append(f"{len(usage.missing)} registered artifact(s) are no "
                         f"longer on disk")
        if usage.symlinks:
            notes.append(f"{len(usage.symlinks)} symlink(s), not followed")
        self._note("; ".join(notes))

        self.usage_table.setRowCount(0)
        for row in usage.kinds:
            if not row.size_bytes and not row.n_artifacts:
                continue
            note = ""
            if not row.size_bytes and row.shared_paths:
                note = (f"lives in {row.shared_paths} file(s) counted under "
                        f"another kind")
            elif row.drifted:
                note = ("what is on disk is not the size the registry "
                        "recorded")
            self._append(self.usage_table, (
                row.label, dm.human_bytes(row.size_bytes), f"{row.n_files:,}",
                dm.human_bytes(row.registered_bytes),
                dm.human_bytes(row.unregistered_bytes), note))

    # -- pruning ----------------------------------------------------------

    def plan_prune(self) -> bool:
        """Work out what could be deleted. Deletes nothing."""
        root = self._root
        kinds = self.selected_kinds()
        usage = self._usage
        # Same guard, same reasoning as scan(): never a stat on this thread.
        if not root or not path_probe.exists(root, want_dir=True,
                                             default=True):
            self._note("Choose a project folder first.", warn=True)
            return False
        def plan():
            """The same worker-side isdir as scan(), for the same reason."""
            if not os.path.isdir(root):
                raise NotADirectoryError("Choose a project folder first.")
            return dm.plan_prune(root, kinds=kinds, usage=usage)

        return self._run(plan, self._show_plan)

    def _show_plan(self, plan: "dm.PrunePlan") -> None:
        """Fill the prune tab with what a deletion WOULD remove.

        A PLAN BEFORE AN ACTION: the point of this screen is that a user sees
        the list before anything is deleted, not a progress bar afterwards.

        :param plan: the proposed deletions.
        """
        self._plan = plan
        self.prune_table.setRowCount(0)
        for candidate in plan.candidates:
            self._append(self.prune_table, (
                dm.human_bytes(candidate.size_bytes), candidate.label,
                os.path.relpath(candidate.path, plan.root),
                f"{candidate.n_files:,}", candidate.regenerate_with))
        self.kept_table.setRowCount(0)
        for skip in plan.kept:
            self._append(self.kept_table, (
                dm.human_bytes(skip.size_bytes),
                os.path.relpath(skip.path, plan.root), skip.reason))
        if plan.candidates:
            self.freed_label.setText(
                f"{dm.human_bytes(plan.total_bytes)} in "
                f"{plan.total_files:,} files can be deleted and made again.")
        else:
            self.freed_label.setText(
                "Nothing here can be deleted safely — see the reasons below.")
        self._update_controls()

    def confirm_and_prune(self) -> bool:
        """Show the confirmation dialog and, if accepted, delete."""
        plan = self._plan
        if plan is None or not plan.candidates:
            return False
        # The screen's own threading, so a test that drives this screen
        # synchronously gets a dialog whose file list is already filled in
        # rather than one that is still reading the disk.
        dialog = ConfirmDeleteDialog(plan, self, threaded=self._threaded)
        if dialog.exec() != QDialog.Accepted:
            self._note("Nothing was deleted.")
            return False
        return self.execute_prune(plan)

    def execute_prune(self, plan: "dm.PrunePlan") -> bool:
        """Carry out ``plan``. The screen's only destructive call.

        The token comes from the plan the screen is holding, so a plan that
        has gone stale — a run wrote into the project while this was open —
        is refused by :func:`spacr.data_manager.prune`, which re-checks the
        tree, rather than by a check duplicated here that could answer
        differently.
        """
        return self._run(lambda: dm.prune(plan, confirm=plan.token),
                         self._after_prune)

    def _after_prune(self, result: "dm.PruneResult") -> None:
        """Report what a completed deletion actually removed.

        :param result: what was deleted.
        """
        self._plan = None
        self.prune_table.setRowCount(0)
        self.freed_label.setText(
            f"Freed {dm.human_bytes(result.freed_bytes)} in "
            f"{result.n_files:,} files.")
        self._note(f"Deleted {len(result.removed_paths)} item(s). The "
                   f"registry still records what produced them, so they can "
                   f"be made again.")
        self.scan()

    # -- archiving --------------------------------------------------------

    def plan_archive(self) -> bool:
        """Work out what an archive would move. Moves nothing."""
        root, destination = self._root, self._destination
        if not root or not destination:
            self._note("Choose a project and a destination first.", warn=True)
            return False
        return self._run(
            lambda: dm.plan_archive(root, destination, usage=self._usage),
            self._show_archive_plan)

    def _show_archive_plan(self, plan: "dm.ArchivePlan") -> None:
        """Fill the archive tab with what WOULD be archived.

        :param plan: the proposed archive.
        """
        self._archive_plan = plan
        self.archive_table.setRowCount(0)
        for item in plan.items:
            self._append(self.archive_table, (
                dm.human_bytes(item.size_bytes),
                dm.KIND_LABELS.get(item.kind, item.kind),
                os.path.relpath(item.source, plan.root), item.destination))
        self._note(f"{dm.human_bytes(plan.total_bytes)} in "
                   f"{plan.total_files:,} files would move to "
                   f"{plan.destination}.")
        self._update_controls()

    def confirm_and_archive(self) -> bool:
        """Ask once, then move."""
        plan = self._archive_plan
        if plan is None or not plan.items:
            return False
        answer = QMessageBox.question(
            self, "Archive this project",
            f"Move {dm.human_bytes(plan.total_bytes)} in "
            f"{plan.total_files:,} files from\n{plan.root}\nto\n"
            f"{plan.destination}?\n\n"
            f"A manifest is written at the destination and a log is left at "
            f"the origin, so the registry still knows where it went.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            self._note("Nothing was moved.")
            return False
        return self._run(lambda: dm.archive(plan, confirm=plan.token),
                         self._after_archive)

    def _after_archive(self, result: "dm.ArchiveResult") -> None:
        """Report what a completed archive actually moved.

        :param result: what was archived.
        """
        self._archive_plan = None
        self.archive_table.setRowCount(0)
        self._note(f"Moved {dm.human_bytes(result.total_bytes)} to "
                   f"{result.destination}. Record left at "
                   f"{result.ledger_path}.")
        self.scan()

    # -- tables -----------------------------------------------------------

    @staticmethod
    def _append(table: QTableWidget, values) -> int:
        """Append one row of strings; returns its index."""
        row = table.rowCount()
        table.insertRow(row)
        for column, value in enumerate(values):
            item = table_item(str(value))
            item.setToolTip(str(value))
            if column == 0:
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            table.setItem(row, column, item)
        return row

    def closeEvent(self, event):        # noqa: N802 - Qt name
        """Stop background work and unlink before going away.

        :param event: the Qt close event.
        """
        redraw = getattr(self, "_path_probe_redraw", None)
        if redraw is not None:
            try:
                path_probe.probes.answered.disconnect(redraw)
            except (RuntimeError, TypeError):
                # Already gone. A screen that refused to close over its own
                # housekeeping would be the worse defect.
                pass
            self._path_probe_redraw = None
        for thread, _worker in list(self._jobs):
            try:
                thread.quit()
                thread.wait(2000)
            except RuntimeError:
                # The thread's C++ half has already gone. A close handler
                # that let this out would leave the screen half-closed,
                # and the job list below is cleared either way.
                pass
        self._jobs.clear()
        super().closeEvent(event)


def make_data_manager_screen(app_key: Optional[str] = None) -> QWidget:
    """Factory handed to :func:`spacr.qt.app.register_app`."""
    return DataManagerScreen()


def register() -> bool:
    """Put the Data Manager in the app registry. Idempotent.

    Called at import time, so importing this module is all it takes for the
    app to exist. Returns rather than raises on a duplicate key, so a
    re-import is a no-op instead of taking the import down.

    :returns: True when this call is what registered it.
    """
    return register_declared(__name__) is not None


register()
