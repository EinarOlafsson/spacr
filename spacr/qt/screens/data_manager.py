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
    QAbstractItemView, QCheckBox, QDialog, QDialogButtonBox, QFileDialog,
    QFrame, QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QTabWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from ... import data_manager as dm
from ...ports import ALL_KINDS
from ..theme import SPACING, font_px, pane_surface, register_widget_qss

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
    surface = pane_surface("surface_alt", palette["theme"], opacity)
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

    :param plan: the plan to confirm.
    :param parent: Qt parent.
    """

    def __init__(self, plan: "dm.PrunePlan", parent=None) -> None:
        super().__init__(parent)
        self.plan = plan
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

        listing = QPlainTextEdit(self)
        listing.setObjectName("DataManagerFileList")
        listing.setReadOnly(True)
        listing.setPlainText(self.describe())
        outer.addWidget(listing, 1)

        self.acknowledged = QCheckBox(
            "I have read the list above and want these files deleted", self)
        self.acknowledged.setObjectName("DataManagerAcknowledge")
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

    def describe(self) -> str:
        """The text shown in the dialog: the plan, then every file."""
        files, truncated = self.plan.file_list()
        lines = [dm.format_prune_plan(self.plan), "", "Files:"]
        lines.extend(f"  {path}" for path in files)
        if truncated:
            lines.append(f"  … and more; over {dm.MAX_RECORDED_FILES:,} "
                         f"files, the list is cut short. The totals above "
                         f"cover all of them.")
        return "\n".join(lines)

    def _on_acknowledged(self, checked: bool) -> None:
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
        super().__init__(parent)
        self.setObjectName("DataManagerScreen")
        self._threaded = bool(threaded)
        self._root = str(project or "")
        self._usage: Optional[dm.ProjectUsage] = None
        self._plan: Optional[dm.PrunePlan] = None
        self._archive_plan: Optional[dm.ArchivePlan] = None
        self._destination = ""
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

        self._update_controls()
        if self._root:
            self.scan()
        # Drop anywhere on this screen: the path is resolved through spaCR's
        # project layout, so the plate folder finds what this screen reads.
        from ..dnd import install_for
        install_for(self, "data_manager")

    # -- construction -----------------------------------------------------

    def _build_head(self) -> QHBoxLayout:
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(SPACING["sm"])

        title = QLabel("Data Manager", self)
        title.setObjectName("ScreenTitle")
        head.addWidget(title)

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
        table = QTableWidget(0, len(columns))
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
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, SPACING["sm"], 0, 0)
        layout.setSpacing(SPACING["sm"])

        picker = QHBoxLayout()
        picker.setContentsMargins(0, 0, 0, 0)
        picker.setSpacing(SPACING["sm"])
        picker.addWidget(QLabel("Consider:", page))
        self.kind_boxes: Dict[str, QCheckBox] = {}
        for kind in _OFFERED_KINDS:
            box = QCheckBox(dm.KIND_LABELS.get(kind, kind), page)
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

    def choose_project(self) -> None:
        """Ask for a project folder and scan it."""
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose a spaCR project", self._root or os.path.expanduser("~"))
        if chosen:
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
        chosen = QFileDialog.getExistingDirectory(
            self, "Archive this project into",
            self._destination or os.path.expanduser("~"))
        if chosen:
            self.set_destination(chosen)

    def set_destination(self, path: str) -> None:
        """Set the archive destination without touching anything."""
        self._destination = str(path)
        self.destination_label.setText(self._destination)
        self._archive_plan = None
        self.archive_table.setRowCount(0)
        self._update_controls()

    def _clear_tables(self) -> None:
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
        line = (text or "").strip().splitlines()[-1] if text else "unknown error"
        self._note(line, warn=True)

    def _on_job_error(self, exc: Exception) -> None:
        LOG.info("data manager job failed", exc_info=True)
        self._note(str(exc) or exc.__class__.__name__, warn=True)

    def _set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        self.progress.setVisible(self._busy)
        self._update_controls()

    def _note(self, text: str, *, warn: bool = False) -> None:
        self.note_label.setText(text)
        self.note_label.setProperty("warn", "true" if warn else "false")
        style = self.note_label.style()
        if style is not None:
            style.unpolish(self.note_label)
            style.polish(self.note_label)

    def _update_controls(self) -> None:
        has_project = bool(self._root) and os.path.isdir(self._root)
        self.rescan_button.setEnabled(has_project and not self._busy)
        self.plan_button.setEnabled(has_project and not self._busy)
        self.delete_button.setEnabled(
            bool(self._plan and self._plan.candidates) and not self._busy)
        self.archive_plan_button.setEnabled(
            has_project and bool(self._destination) and not self._busy)
        self.archive_button.setEnabled(
            bool(self._archive_plan and self._archive_plan.items)
            and not self._busy)

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
        if not root or not os.path.isdir(root):
            self._note("Choose a project folder first.", warn=True)
            return False
        self._note("")
        return self._run(lambda: dm.scan_project(root), self._show_usage)

    def _show_usage(self, usage: "dm.ProjectUsage") -> None:
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
        if not root or not os.path.isdir(root):
            self._note("Choose a project folder first.", warn=True)
            return False
        return self._run(
            lambda: dm.plan_prune(root, kinds=kinds, usage=usage),
            self._show_plan)

    def _show_plan(self, plan: "dm.PrunePlan") -> None:
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
        dialog = ConfirmDeleteDialog(plan, self)
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
            item = QTableWidgetItem(str(value))
            item.setToolTip(str(value))
            if column == 0:
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            table.setItem(row, column, item)
        return row

    def closeEvent(self, event):        # noqa: N802 - Qt name
        for thread, _worker in list(self._jobs):
            try:
                thread.quit()
                thread.wait(2000)
            except RuntimeError:        # pragma: no cover - already gone
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
    import inspect

    from ..app import APPS, SECTION_DATA, STAGE_ALPHA, register_app
    if any(row[0] == APP_KEY for row in APPS):
        return False
    # The api_module link points the ⓘ at the module that does the work,
    # through the metadata seam, rather than through a hand-edit of the table
    # in ``settings_model.py``. The ``spacr.cli.INTERACTIVE_ONLY`` entry is
    # written by hand instead of pushed from here on purpose: that push only
    # lands when ``spacr.qt.app`` is imported, and ``spacr-run`` must answer
    # "why can I not run this headless?" without PySide6.
    #
    # It is passed only when register_app accepts it. The metadata keywords
    # are being added to that seam by a separate change, and this module
    # shipped ahead of them -- so `spacr` died at launch with
    # `TypeError: register_app() got an unexpected keyword argument
    # 'api_module'`: a screen committed against a signature that had not
    # landed. Asking the live signature costs one inspect call at import and
    # makes this module correct against both, in either merge order.
    extras = {"api_module": "data_manager"}
    accepted = inspect.signature(register_app).parameters
    register_app(
        APP_KEY, "Data Manager",
        "See what a project costs in disk, and reclaim it without touching "
        "the originals",
        SECTION_DATA, factory=make_data_manager_screen, stage=STAGE_ALPHA,
        **{k: v for k, v in extras.items() if k in accepted})
    return True


register()
