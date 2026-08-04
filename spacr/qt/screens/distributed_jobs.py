"""Configure, submit and monitor remote/distributed spaCR jobs.

Every network and scheduler call runs through :func:`make_thread`; the GUI
thread only updates controls and renders already-returned records.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, List, Optional

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...remote_execution import (
    ACTIVE_STATES,
    ExecutionProfile,
    RemoteExecutionError,
    RemoteJob,
    RemoteJobManager,
)
from ..bridge import make_thread
from ..i18n import tr
from ..iconset import icon
from ..theme import SPACING, active_palette
from ..widgets import Card, Divider, InfoLink
from .settings_model import api_docs_url, attach_api_tooltip

LOG = logging.getLogger(__name__)

APP_KEY = "distributed_jobs"
APP_NAME = "Distributed Jobs"
APP_SECTION = "Data"
APP_INTRO = (
    "Submit a normal spaCR settings file to another workstation, Slurm, or "
    "a cloud/HPC command; monitor logs and cancel it without blocking spaCR."
)

_COLUMNS = (
    "Job", "Status", "Module", "Profile", "Remote ID", "Submitted", "Updated",
)


def _item(value) -> QTableWidgetItem:
    """Return a non-editable table item."""
    item = QTableWidgetItem("" if value is None else str(value))
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    return item


class ExecutionProfileDialog(QDialog):
    """Create or edit one distributed execution profile."""

    def __init__(
        self,
        parent=None,
        profile: Optional[ExecutionProfile] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(tr("Execution profile"))
        self.setMinimumWidth(650)
        self._original_name = profile.name if profile else ""
        self._build_ui()
        if profile is not None:
            self._load(profile)
        self._sync_backend()

    def _build_ui(self) -> None:
        """Build the backend-independent profile editor."""
        outer = QVBoxLayout(self)
        intro = QLabel(
            tr(
                "Profiles store connection commands, never passwords. Configure "
                "SSH keys or your cloud CLI outside spaCR."
            ),
            self,
        )
        intro.setWordWrap(True)
        intro.setObjectName("Muted")
        outer.addWidget(intro)

        form = QFormLayout()
        self._name = QLineEdit(self)
        self._name.setObjectName("ProfileName")
        self._name.setToolTip(
            "A local display name. API: spacr.remote_execution.ExecutionProfile"
        )
        self._add_profile_row(
            form, "Profile name", self._name, "profile_name"
        )

        self._backend = QComboBox(self)
        self._backend.setObjectName("ProfileBackend")
        self._backend.addItem(tr("SSH workstation"), "ssh")
        self._backend.addItem(tr("Slurm cluster"), "slurm")
        self._backend.addItem(tr("Cloud / custom command"), "command")
        self._backend.setToolTip(
            "SSH runs a durable background process; Slurm uses sbatch/squeue/"
            "sacct; custom executes argument templates without shell=True."
        )
        self._backend.currentIndexChanged.connect(self._sync_backend)
        self._add_profile_row(form, "Backend", self._backend, "backend")

        self._host = QLineEdit(self)
        self._host.setObjectName("ProfileHost")
        self._host.setPlaceholderText("user@workstation or cluster-login")
        self._host.setToolTip(
            "OpenSSH target. Leave blank only for Slurm commands installed "
            "on this computer. Configure key authentication outside spaCR."
        )
        self._add_profile_row(form, "SSH host", self._host, "host")

        self._workdir = QLineEdit(self)
        self._workdir.setObjectName("ProfileWorkdir")
        self._workdir.setPlaceholderText("/shared/project")
        self._workdir.setToolTip(
            "Absolute directory on the execution host. Small settings and log "
            "files are stored below <workdir>/spacr-jobs/."
        )
        self._add_profile_row(
            form, "Remote work directory", self._workdir, "workdir"
        )

        self._local_root = QLineEdit(self)
        self._local_root.setObjectName("ProfileLocalRoot")
        self._local_root.setPlaceholderText("/local/mount/project")
        self._local_root.setToolTip(
            "Optional local prefix for shared/mirrored input data. spaCR maps "
            "paths but deliberately does not copy entire image datasets."
        )
        self._add_profile_row(
            form, "Local dataset root", self._local_root, "local_root"
        )

        self._remote_root = QLineEdit(self)
        self._remote_root.setObjectName("ProfileRemoteRoot")
        self._remote_root.setPlaceholderText("/cluster/mount/project")
        self._remote_root.setToolTip(
            "Remote prefix representing the same data as Local dataset root."
        )
        self._add_profile_row(
            form, "Remote dataset root", self._remote_root, "remote_root"
        )

        self._runner = QLineEdit(self)
        self._runner.setObjectName("ProfileRunner")
        self._runner.setText("spacr-run")
        self._runner.setToolTip(
            "Installed headless spaCR executable on the remote target."
        )
        self._add_profile_row(
            form, "spaCR runner", self._runner, "runner"
        )

        self._slurm = QLineEdit(self)
        self._slurm.setObjectName("ProfileSlurmOptions")
        self._slurm.setPlaceholderText("--partition=gpu --gres=gpu:1 --time=12:00:00")
        self._slurm.setToolTip(
            "Additional sbatch arguments. They are parsed as arguments, not "
            "executed as a shell command."
        )
        self._add_profile_row(
            form, "Slurm options", self._slurm, "scheduler_options"
        )

        self._submit_command = QLineEdit(self)
        self._submit_command.setObjectName("ProfileSubmitCommand")
        self._submit_command.setPlaceholderText(
            "cloud-submit --module {module} --settings {settings}"
        )
        self._submit_command.setToolTip(
            "Argument template. Must print a job ID and contain {settings}. "
            "Also supports {job_id}, {module} and {profile}."
        )
        self._add_profile_row(
            form, "Submit command", self._submit_command, "submit_command"
        )

        self._status_command = QLineEdit(self)
        self._status_command.setObjectName("ProfileStatusCommand")
        self._status_command.setPlaceholderText(
            "cloud-status {external_id}"
        )
        self._status_command.setToolTip(
            "Must print a conventional state such as PENDING, RUNNING, "
            "SUCCEEDED, FAILED or CANCELLED."
        )
        self._add_profile_row(
            form, "Status command", self._status_command, "status_command"
        )

        self._cancel_command = QLineEdit(self)
        self._cancel_command.setObjectName("ProfileCancelCommand")
        self._cancel_command.setPlaceholderText(
            "cloud-cancel {external_id}"
        )
        self._cancel_command.setToolTip(
            "Argument template used to request cancellation. Must contain "
            "{external_id}."
        )
        self._add_profile_row(
            form, "Cancel command", self._cancel_command, "cancel_command"
        )

        self._log_command = QLineEdit(self)
        self._log_command.setObjectName("ProfileLogCommand")
        self._log_command.setPlaceholderText("cloud-logs {external_id}")
        self._log_command.setToolTip(
            "Optional argument template that prints the job log. Use "
            "{external_id}; credentials stay in the external CLI."
        )
        self._add_profile_row(
            form, "Log command (optional)", self._log_command, "log_command"
        )

        self._job_pattern = QLineEdit(self)
        self._job_pattern.setObjectName("ProfileJobPattern")
        self._job_pattern.setPlaceholderText(
            r'"jobId":\s*"(?P<id>[A-Za-z0-9-]+)"'
        )
        self._job_pattern.setToolTip(
            "Optional regular expression extracting a job ID from submit "
            "output. Use a named group (?P<id>...)."
        )
        self._add_profile_row(
            form, "Job-ID pattern (optional)", self._job_pattern,
            "job_id_pattern",
        )

        self._poll = QSpinBox(self)
        self._poll.setObjectName("ProfilePollSeconds")
        self._poll.setRange(2, 3600)
        self._poll.setValue(10)
        self._poll.setSuffix(tr(" seconds"))
        self._poll.setToolTip(
            "Suggested local polling interval. Manual refresh is always "
            "available."
        )
        self._add_profile_row(
            form, "Poll interval", self._poll, "poll_seconds"
        )
        outer.addLayout(form)

        self._error = QLabel("", self)
        self._error.setObjectName("InlineError")
        self._error.setWordWrap(True)
        outer.addWidget(self._error)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel, parent=self
        )
        save = buttons.button(QDialogButtonBox.Save)
        if save is not None:
            save.setObjectName("PrimaryButton")
        cancel = buttons.button(QDialogButtonBox.Cancel)
        if cancel is not None:
            cancel.setObjectName("DangerButton")
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def _add_profile_row(
        self,
        form: QFormLayout,
        source_label: str,
        field: QWidget,
        api_key: str,
    ) -> None:
        """Add a hover-help label and teal API dot beside one profile field."""
        help_text = field.toolTip() or (
            f"Controls {source_label.casefold()} for distributed execution."
        )
        label = QLabel(tr(source_label), self)
        label.setObjectName("SettingsLabel")
        attach_api_tooltip(
            label, help_text, "distributed_jobs", api_key
        )
        info = InfoLink(
            api_docs_url("distributed_jobs", api_key),
            tooltip=(
                f"Open spaCR API documentation for {source_label.casefold()}."
            ),
            parent=self,
        )
        wrapper = QWidget(self)
        row = QHBoxLayout(wrapper)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING["xs"])
        row.addWidget(label)
        row.addWidget(info)
        row.addStretch(1)
        form.addRow(wrapper, field)

    def _sync_backend(self, *_args) -> None:
        """Show only controls meaningful to the selected backend."""
        backend = self._backend.currentData()
        ssh_or_slurm = backend in {"ssh", "slurm"}
        for widget in (
            self._host, self._workdir, self._local_root, self._remote_root,
            self._runner,
        ):
            widget.setEnabled(ssh_or_slurm)
        self._slurm.setEnabled(backend == "slurm")
        for widget in (
            self._submit_command, self._status_command, self._cancel_command,
            self._log_command, self._job_pattern,
        ):
            widget.setEnabled(backend == "command")

    def _load(self, profile: ExecutionProfile) -> None:
        """Populate controls from an existing profile."""
        self._name.setText(profile.name)
        index = self._backend.findData(profile.backend)
        self._backend.setCurrentIndex(max(0, index))
        self._host.setText(profile.host)
        self._workdir.setText(profile.workdir)
        self._local_root.setText(profile.local_root)
        self._remote_root.setText(profile.remote_root)
        self._runner.setText(profile.runner)
        self._slurm.setText(profile.scheduler_options)
        self._submit_command.setText(profile.submit_command)
        self._status_command.setText(profile.status_command)
        self._cancel_command.setText(profile.cancel_command)
        self._log_command.setText(profile.log_command)
        self._job_pattern.setText(profile.job_id_pattern)
        self._poll.setValue(profile.poll_seconds)

    def profile(self) -> ExecutionProfile:
        """Return the currently entered, validated profile."""
        return ExecutionProfile(
            name=self._name.text(),
            backend=str(self._backend.currentData()),
            host=self._host.text(),
            workdir=self._workdir.text(),
            local_root=self._local_root.text(),
            remote_root=self._remote_root.text(),
            runner=self._runner.text(),
            scheduler_options=self._slurm.text(),
            submit_command=self._submit_command.text(),
            status_command=self._status_command.text(),
            cancel_command=self._cancel_command.text(),
            log_command=self._log_command.text(),
            job_id_pattern=self._job_pattern.text(),
            poll_seconds=self._poll.value(),
        ).validate()

    def _validate_and_accept(self) -> None:
        """Keep the dialog open and show validation errors inline."""
        try:
            self.profile()
        except RemoteExecutionError as exc:
            self._error.setText(str(exc))
            return
        self._error.clear()
        self.accept()


class DistributedJobsScreen(QWidget):
    """Submit and monitor distributed spaCR jobs without blocking Qt."""

    def __init__(
        self,
        parent=None,
        *,
        manager: Optional[RemoteJobManager] = None,
        threaded: bool = True,
        auto_poll: bool = True,
    ):
        super().__init__(parent)
        self.manager = manager or RemoteJobManager()
        self._threaded = bool(threaded)
        self._auto_poll = bool(auto_poll)
        self._jobs: List[RemoteJob] = []
        self._workers: List[tuple] = []
        self._busy = False
        self._pending_result = None
        self._pending_error = ""
        self._pending_callback: Optional[Callable] = None
        self._settings_snapshot: Optional[dict] = None
        self.setAcceptDrops(True)
        self._build_ui()
        self._reload_profiles()
        try:
            initial_jobs = self.manager.jobs.list()
        except Exception as exc:
            initial_jobs = []
            self._set_status(
                f"Could not load distributed job records: "
                f"{type(exc).__name__}: {exc}",
                error=True,
            )
        self._render_jobs(initial_jobs)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._update_poll_interval()

    def _build_ui(self) -> None:
        """Construct profile, submission, table, and detail controls."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(
            SPACING["lg"], SPACING["lg"], SPACING["lg"], SPACING["lg"]
        )
        outer.setSpacing(SPACING["md"])
        title = QLabel(APP_NAME, self)
        title.setObjectName("DisplayHeading")
        outer.addWidget(title)
        intro = QLabel(APP_INTRO, self)
        intro.setObjectName("Muted")
        intro.setWordWrap(True)
        outer.addWidget(intro)
        outer.addWidget(Divider())

        submit_card = Card(title="Submission")
        profile_row = QHBoxLayout()
        self._profile = QComboBox(self)
        self._profile.setObjectName("ExecutionProfileChoice")
        self._profile.setProperty("i18nSkipItems", True)
        self._profile.setToolTip(
            "Saved target from spacr.remote_execution.ProfileStore."
        )
        self._new_profile = QPushButton("New profile…", self)
        self._new_profile.clicked.connect(self._create_profile)
        self._edit_profile = QPushButton("Edit profile…", self)
        self._edit_profile.clicked.connect(self._edit_selected_profile)
        self._delete_profile = QPushButton("Delete profile", self)
        self._delete_profile.setObjectName("DangerButton")
        self._delete_profile.clicked.connect(self._delete_selected_profile)
        profile_row.addWidget(QLabel("Execution target", self))
        profile_row.addWidget(self._profile, 1)
        profile_row.addWidget(self._new_profile)
        profile_row.addWidget(self._edit_profile)
        profile_row.addWidget(self._delete_profile)
        submit_card.body_layout.addLayout(profile_row)

        job_row = QHBoxLayout()
        self._module = QComboBox(self)
        self._module.setObjectName("DistributedModule")
        self._module.setProperty("i18nSkipItems", True)
        from ...cli import MODULES
        for key in sorted(MODULES):
            self._module.addItem(key, key)
        self._settings_path = QLineEdit(self)
        self._settings_path.setObjectName("DistributedSettingsPath")
        self._settings_path.setPlaceholderText(
            "Drop or choose a spaCR settings CSV/JSON"
        )
        self._settings_path.setToolTip(
            "A settings CSV exported by a module or settings.json from a run "
            "manifest. It is fully resolved before submission."
        )
        self._settings_path.textEdited.connect(self._clear_settings_snapshot)
        browse = QPushButton("Browse…", self)
        browse.clicked.connect(self._browse_settings)
        self._submit = QPushButton("Submit", self)
        self._submit.setObjectName("PrimaryButton")
        self._submit.clicked.connect(self.submit)
        job_row.addWidget(QLabel("Module", self))
        job_row.addWidget(self._module)
        job_row.addWidget(self._settings_path, 1)
        job_row.addWidget(browse)
        job_row.addWidget(self._submit)
        submit_card.body_layout.addLayout(job_row)
        outer.addWidget(submit_card)

        actions = QHBoxLayout()
        self._refresh = QPushButton("Refresh", self)
        self._refresh.setObjectName("PrimaryButton")
        self._refresh.setIcon(icon("redo"))
        self._refresh.clicked.connect(self.refresh)
        self._cancel = QPushButton("Cancel job", self)
        self._cancel.setObjectName("DangerButton")
        self._cancel.clicked.connect(self.cancel_selected)
        self._logs = QPushButton("Refresh log", self)
        self._logs.clicked.connect(self.refresh_log)
        self._open_local = QPushButton("Open local record", self)
        self._open_local.setIcon(icon("folder"))
        self._open_local.clicked.connect(self._open_local_record)
        actions.addWidget(self._refresh)
        actions.addWidget(self._cancel)
        actions.addWidget(self._logs)
        actions.addWidget(self._open_local)
        actions.addStretch(1)
        outer.addLayout(actions)

        splitter = QSplitter(Qt.Vertical, self)
        self._table = QTableWidget(0, len(_COLUMNS), splitter)
        self._table.setHorizontalHeaderLabels(list(_COLUMNS))
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        self._table.itemSelectionChanged.connect(self._show_selection)
        splitter.addWidget(self._table)

        self._detail = QPlainTextEdit(splitter)
        self._detail.setReadOnly(True)
        self._detail.setLineWrapMode(QPlainTextEdit.NoWrap)
        self._detail.setAccessibleName("Distributed job details and log")
        splitter.addWidget(self._detail)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter, 1)

        self._status = QLabel("", self)
        self._status.setObjectName("Muted")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

    def configure_submission(self, module: str, settings: dict) -> None:
        """Preload an immutable settings snapshot handed off by an AppScreen."""
        index = self._module.findData(str(module))
        if index >= 0:
            self._module.setCurrentIndex(index)
        self._settings_snapshot = dict(settings)
        self._settings_path.setText(
            tr("[current {module} settings snapshot]").format(module=module)
        )
        self._set_status(
            tr("Choose an execution profile, then Submit.")
        )

    def _clear_settings_snapshot(self, _text: str = "") -> None:
        """Switch back to file mode when the user edits the path field."""
        self._settings_snapshot = None

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Accept one local settings CSV/JSON."""
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if any(
            url.isLocalFile()
            and Path(url.toLocalFile()).suffix.casefold() in {".csv", ".json"}
            for url in urls
        ):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Use the first compatible dropped settings file."""
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if url.isLocalFile() and path.suffix.casefold() in {".csv", ".json"}:
                self._settings_path.setText(str(path))
                event.acceptProposedAction()
                return

    def _reload_profiles(self, selected: str = "") -> None:
        """Reload the profile combo from persistent storage."""
        current = selected or str(self._profile.currentData() or "")
        try:
            profiles = self.manager.profiles.list()
        except Exception as exc:
            profiles = []
            self._set_status(
                f"Could not load execution profiles: "
                f"{type(exc).__name__}: {exc}",
                error=True,
            )
        self._profile.clear()
        for profile in profiles:
            self._profile.addItem(profile.name, profile.name)
        index = self._profile.findData(current)
        self._profile.setCurrentIndex(index if index >= 0 else 0)
        available = bool(profiles)
        self._edit_profile.setEnabled(available)
        self._delete_profile.setEnabled(available)
        self._submit.setEnabled(available and not self._busy)
        self._update_poll_interval(profiles)

    def _update_poll_interval(
        self, profiles: Optional[List[ExecutionProfile]] = None
    ) -> None:
        """Use the fastest configured profile interval for local monitoring."""
        timer = getattr(self, "_timer", None)
        if timer is None:
            return
        profiles = profiles if profiles is not None else self.manager.profiles.list()
        seconds = min(
            (profile.poll_seconds for profile in profiles), default=10
        )
        timer.setInterval(max(2, int(seconds)) * 1000)

    def _create_profile(self) -> None:
        """Open an empty profile editor and persist accepted values."""
        dialog = ExecutionProfileDialog(self)
        if dialog.exec() == QDialog.Accepted:
            profile = dialog.profile()
            try:
                self.manager.profiles.save(profile)
            except Exception as exc:
                LOG.exception("Could not save execution profile")
                self._set_status(
                    f"Could not save profile: {type(exc).__name__}: {exc}",
                    error=True,
                )
                return
            self._reload_profiles(profile.name)

    def _edit_selected_profile(self) -> None:
        """Edit the selected profile snapshot."""
        name = str(self._profile.currentData() or "")
        if not name:
            return
        try:
            existing = self.manager.profiles.get(name)
        except RemoteExecutionError as exc:
            self._set_status(str(exc), error=True)
            return
        dialog = ExecutionProfileDialog(self, existing)
        if dialog.exec() == QDialog.Accepted:
            profile = dialog.profile()
            # Persist the replacement first: a disk error must not erase the
            # only usable profile merely because the user renamed it.
            try:
                self.manager.profiles.save(profile)
                if profile.name.casefold() != existing.name.casefold():
                    self.manager.profiles.delete(existing.name)
            except Exception as exc:
                LOG.exception("Could not update execution profile")
                self._set_status(
                    f"Could not update profile: {type(exc).__name__}: {exc}",
                    error=True,
                )
                return
            self._reload_profiles(profile.name)

    def _delete_selected_profile(self) -> None:
        """Delete a profile after explicit confirmation."""
        name = str(self._profile.currentData() or "")
        if not name:
            return
        answer = QMessageBox.question(
            self,
            tr("Delete profile"),
            tr("Delete execution profile '{name}'?").format(name=name),
        )
        if answer == QMessageBox.Yes:
            try:
                self.manager.profiles.delete(name)
            except Exception as exc:
                LOG.exception("Could not delete execution profile")
                self._set_status(
                    f"Could not delete profile: {type(exc).__name__}: {exc}",
                    error=True,
                )
                return
            self._reload_profiles()

    def _browse_settings(self) -> None:
        """Choose a normal spaCR settings export."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            tr("Choose spaCR settings"),
            "",
            "spaCR settings (*.csv *.json);;All files (*)",
        )
        if path:
            self._settings_path.setText(path)

    def _start_task(self, label: str, operation: Callable, callback: Callable) -> None:
        """Run one blocking manager operation off the GUI thread."""
        if self._busy:
            self._set_status(tr("Another distributed-job operation is running."))
            return
        self._busy = True
        self._pending_result = None
        self._pending_error = ""
        self._pending_callback = callback
        self._set_busy(True)
        self._set_status(label)

        def _work(_settings):
            try:
                self._pending_result = operation()
            except Exception as exc:
                LOG.exception("Distributed job operation failed")
                self._pending_error = f"{type(exc).__name__}: {exc}"

        if not self._threaded:
            _work({})
            self._finish_task(not bool(self._pending_error))
            return
        thread, worker = make_thread(
            _work, {}, app_key="distributed_jobs_io", journal=False
        )
        self._workers.append((thread, worker))
        worker.finished.connect(self._finish_task)
        thread.finished.connect(self._retire_workers)
        thread.start()

    def _finish_task(self, ok: bool) -> None:
        """Apply one worker result on the GUI thread."""
        self._busy = False
        self._set_busy(False)
        if not ok or self._pending_error:
            self._set_status(
                self._pending_error or tr("Distributed operation failed."),
                error=True,
            )
            return
        callback = self._pending_callback
        if callback is not None:
            callback(self._pending_result)

    def _retire_workers(self) -> None:
        """Release ownership pairs whose QThread has stopped.

        A bare ``isRunning()`` filter leaked every pair: by the time this
        queued slot runs, ``thread.finished -> deleteLater`` has reaped the
        QThread's C++ half and ``isRunning()`` raises ``RuntimeError`` out
        of the slot, so the assignment never happens. See
        :func:`spacr.qt.bridge.prune_job_pairs`.
        """
        from ..bridge import prune_job_pairs

        self._workers = prune_job_pairs(self._workers, self.sender())

    def _set_busy(self, busy: bool) -> None:
        """Update semantic action states while a network call is active."""
        for button in (self._submit, self._refresh, self._cancel, self._logs):
            button.setEnabled(not busy)
        for button in (self._submit, self._refresh):
            button.setProperty("buttonActionBusy", bool(busy))
            button.style().unpolish(button)
            button.style().polish(button)
        if not busy:
            self._reload_profiles()

    def submit(self) -> None:
        """Resolve and submit the selected settings file."""
        module_name = str(self._module.currentData() or "")
        profile_name = str(self._profile.currentData() or "")
        settings_path = self._settings_path.text().strip()
        if not profile_name:
            self._set_status(tr("Create an execution profile first."), error=True)
            return
        if (
            self._settings_snapshot is None
            and (not settings_path or not Path(settings_path).is_file())
        ):
            self._set_status(tr("Choose an existing settings CSV/JSON."), error=True)
            return
        settings_snapshot = (
            dict(self._settings_snapshot)
            if self._settings_snapshot is not None else None
        )

        def _submit():
            from ...cli import resolve_module, resolve_settings
            module = resolve_module(module_name)
            if module is None:
                raise RemoteExecutionError(
                    f"Unknown spaCR module: {module_name}"
                )
            settings = (
                settings_snapshot
                if settings_snapshot is not None
                else resolve_settings(module, settings_path, [])
            )
            return self.manager.submit(module.key, settings, profile_name)

        def _done(job: RemoteJob):
            self._render_jobs(self.manager.jobs.list(), select=job.job_id)
            self._set_status(
                tr("Submitted {module} as {job}.").format(
                    module=job.module, job=job.job_id
                )
            )

        self._start_task(tr("Submitting distributed job…"), _submit, _done)

    def refresh(self) -> None:
        """Poll all non-terminal jobs in a worker thread."""
        def _done(jobs):
            self._render_jobs(jobs)
            active = sum(
                job.status in ACTIVE_STATES or job.status == "unknown"
                for job in jobs
            )
            self._set_status(
                tr("Loaded {count} jobs; {active} active.").format(
                    count=len(jobs), active=active
                )
            )

        self._start_task(
            tr("Polling remote jobs…"),
            lambda: self.manager.refresh_all(include_logs=False),
            _done,
        )

    def _selected_job(self) -> Optional[RemoteJob]:
        """Return the job attached to the selected table row."""
        row = self._table.currentRow()
        if row < 0:
            return None
        item = self._table.item(row, 0)
        job_id = item.data(Qt.UserRole) if item is not None else ""
        return next((job for job in self._jobs if job.job_id == job_id), None)

    def cancel_selected(self) -> None:
        """Request cancellation for the selected active job."""
        job = self._selected_job()
        if job is None:
            self._set_status(tr("Select a job first."), error=True)
            return
        if job.status not in ACTIVE_STATES and job.status != "unknown":
            self._set_status(
                tr("That job is already {status}.").format(status=job.status)
            )
            return
        answer = QMessageBox.question(
            self,
            tr("Cancel job"),
            tr("Request cancellation for job {job}?").format(
                job=job.job_id[:12]
            ),
        )
        if answer != QMessageBox.Yes:
            return

        def _done(updated):
            self._render_jobs(self.manager.jobs.list(), select=updated.job_id)
            self._set_status(
                tr("Cancellation requested for {job}.").format(
                    job=updated.job_id
                )
            )

        self._start_task(
            tr("Cancelling remote job…"),
            lambda: self.manager.cancel(job.job_id),
            _done,
        )

    def refresh_log(self) -> None:
        """Fetch the selected job's remote log tail."""
        job = self._selected_job()
        if job is None:
            self._set_status(tr("Select a job first."), error=True)
            return

        def _done(text):
            updated = self.manager.jobs.get(job.job_id)
            self._render_jobs(self.manager.jobs.list(), select=updated.job_id)
            self._detail.setPlainText(self._job_detail(updated))
            self._set_status(tr("Log refreshed."))

        self._start_task(
            tr("Retrieving remote log…"),
            lambda: self.manager.logs(job.job_id, 500),
            _done,
        )

    def _render_jobs(
        self,
        jobs: List[RemoteJob],
        *,
        select: str = "",
    ) -> None:
        """Render newest-first persistent jobs and restore selection."""
        previous = select
        current = self._selected_job()
        if not previous and current is not None:
            previous = current.job_id
        self._jobs = list(jobs)
        self._table.setRowCount(0)
        palette = active_palette()
        colours = {
            "success": palette["success"],
            "failed": palette["error"],
            "cancelled": palette["error"],
            "running": palette["accent_hi"],
            "queued": palette["warning"],
            "pending": palette["warning"],
            "unknown": palette["fg_dim"],
        }
        selected_row = -1
        for job in self._jobs:
            row = self._table.rowCount()
            self._table.insertRow(row)
            values = (
                job.job_id[:12], job.status, job.module, job.profile_name,
                job.external_id or "—", job.created_utc, job.updated_utc,
            )
            for column, value in enumerate(values):
                cell = _item(value)
                if column == 0:
                    cell.setData(Qt.UserRole, job.job_id)
                if column == 1:
                    cell.setForeground(
                        QColor(colours.get(job.status, palette["fg_dim"]))
                    )
                self._table.setItem(row, column, cell)
            if job.job_id == previous:
                selected_row = row
        if selected_row >= 0:
            self._table.selectRow(selected_row)
        elif self._jobs:
            self._table.selectRow(0)
        else:
            self._detail.setPlainText(
                tr("No distributed jobs have been submitted.")
            )
        self._show_selection()

    @staticmethod
    def _job_detail(job: RemoteJob) -> str:
        """Format a complete, copyable job record plus retained log."""
        record = job.to_dict()
        log = record.pop("log_tail", "")
        profile = dict(record.get("profile") or {})
        # Profiles never contain credentials, but avoid encouraging users to
        # paste arbitrary custom command lines into public bug reports.
        for key in (
            "submit_command", "status_command", "cancel_command", "log_command",
        ):
            if profile.get(key):
                profile[key] = "<configured command>"
        record["profile"] = profile
        text = json.dumps(record, indent=2, sort_keys=True, default=str)
        if log:
            text += "\n\n--- remote log tail ---\n" + log
        return text

    def _show_selection(self) -> None:
        """Render selected details and update action availability."""
        job = self._selected_job()
        if job is None:
            self._cancel.setEnabled(False)
            self._logs.setEnabled(False)
            self._open_local.setEnabled(False)
            return
        self._detail.setPlainText(self._job_detail(job))
        self._cancel.setEnabled(
            not self._busy
            and (job.status in ACTIVE_STATES or job.status == "unknown")
        )
        self._logs.setEnabled(not self._busy)
        self._open_local.setEnabled(True)

    def _open_local_record(self) -> None:
        """Open the selected local job record directory."""
        job = self._selected_job()
        if job is None:
            return
        QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(Path(job.settings_path).parent))
        )

    def _set_status(self, text: str, error: bool = False) -> None:
        """Set the status line with theme-aware severity."""
        self._status.setText(str(text))
        palette = active_palette()
        self._status.setStyleSheet(
            f"color: {palette['error' if error else 'fg_dim']};"
        )

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Poll only while this screen is visible."""
        super().showEvent(event)
        if self._auto_poll and not self._timer.isActive():
            self._timer.start()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Stop background polling when another module is open."""
        self._timer.stop()
        super().hideEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Stop polling and drain live workers before the screen goes away.

        ``requestInterruption`` on its own was decorative — ``_work`` never
        polls it — and left the screen's REST calls running with nobody
        owning them. An ownerless job stays in the process-wide run
        registry, which is what ``MainWindow.closeEvent`` consults when it
        decides whether the application may quit.
        """
        from ..bridge import drain_thread

        self._timer.stop()
        for thread, worker in list(self._workers):
            if worker is not None:
                try:
                    worker.request_cancel("distributed-jobs screen closed")
                except Exception:
                    pass
            try:
                thread.requestInterruption()
            except Exception:
                pass
            drain_thread(thread, worker, timeout_ms=3000)
        self._workers.clear()
        super().closeEvent(event)
