"""Configure, submit and monitor remote/distributed spaCR jobs.

Every network and scheduler call runs through :func:`make_thread`; the GUI
thread only updates controls and renders already-returned records.
"""
from __future__ import annotations

import functools
import json
import logging
import os
import re
import threading
from dataclasses import replace
from pathlib import Path
from typing import Callable, List, Optional

from PySide6.QtCore import QObject, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFontDatabase
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...remote_execution import (
    _ENV_PRESETS,
    _JOB_SCRIPT_PLACEHOLDERS,
    _PROFILE_PRESETS,
    _SCRIPT_PRESETS,
    _SSH_SESSIONS,
    ACTIVE_STATES,
    TERMINAL_STATES,
    ExecutionProfile,
    RemoteExecutionError,
    RemoteJob,
    RemoteJobManager,
    _allocation_shell_command,
    _auth_mode,
    _discover_ssh_settings,
    _keychain_account,
    _keychain_available,
    _keychain_password,
    _load_cluster_cache,
    _log_chunk,
    _preview_job_script,
    _probe_cluster,
    _remote_argv,
    _resource_problems,
    _run_command,
    _save_cluster_cache,
    _signin_command,
    _start_allocation,
    _submit_job,
)
from ..bridge import make_thread
from ..i18n import tr
from ..iconset import icon
from ..theme import SPACING, active_palette
from ..widgets import Card, Divider
from ..widgets.collapsible_splitter import CollapsibleSplitter
from ..widgets.sortable_table import (
    _retire_table_cells, _spare_table_item, install_sorting, table_item,
)
from .settings_model import attach_api_tooltip

LOG = logging.getLogger(__name__)

APP_KEY = "distributed_jobs"
APP_NAME = "Distributed Jobs"
APP_SECTION = "Data"
APP_INTRO = (
    "Submit a normal spaCR settings file to another workstation, a Slurm, "
    "PBS or LSF cluster, or a cloud/HPC command; follow its log in the "
    "console and cancel it without blocking spaCR."
)

_COLUMNS = (
    "Job", "Status", "Module", "Profile", "Remote ID", "Submitted", "Updated",
)

_CONNECTION_PROFILE_FIELDS = (
    'host', 'workdir', 'local_root', 'remote_root', 'ssh_options',
    'ssh_interactive', 'ssh_keychain', 'auth_mode', 'ssh_command',
    'login_hosts', 'poll_seconds',
)


def _check_profile_transfer_secrets(value):
    """Refuse credential values while retaining paths and environment references."""
    refused = tr("Profile files may not contain passwords, verification codes, tokens or private-key contents. Use non-secret references.")
    names = r'(?:[A-Za-z0-9]+_)*(?:password|passwd|passphrase|otp|mfa|mfa_reply|verification_code|one_time_password|api_key|access_token|secret_key|token|secret|private_key)'
    names = names.replace('_', '[_-]')
    secret = re.compile(names, re.I)
    reference = re.compile(r'(?:\$\{[A-Za-z_][A-Za-z0-9_]*\}|\$[A-Za-z_][A-Za-z0-9_]*|%[A-Za-z_][A-Za-z0-9_]*%)')
    if isinstance(value, dict):
        for key, item in value.items():
            if secret.fullmatch(str(key).replace('-', '_')) and item is not None and item != '' and item is not False:
                if not isinstance(item, str) or not reference.fullmatch(item.strip()):
                    raise RemoteExecutionError(refused)
            _check_profile_transfer_secrets(item)
    elif isinstance(value, list):
        for item in value:
            _check_profile_transfer_secrets(item)
    elif isinstance(value, str):
        if re.search(r'-----BEGIN [A-Z ]*PRIVATE KEY-----|\bsshpass\b', value, re.I):
            raise RemoteExecutionError(refused)
        assignments = r'(?<![A-Za-z0-9_])(?:' + names + r'\s*[:=]\s*|--' + names + r'(?:=|\s+))'
        for match in re.finditer(assignments + r'("[^"]*"|\'[^\']*\'|[^\s;]+)', value, re.I):
            if not reference.fullmatch(match.group(1).strip('\"\'')):
                raise RemoteExecutionError(refused)


def _validate_profile_transfer(value, scope):
    """Validate a bounded transfer document before any form or store changes."""
    from ...cli import MODULES

    _check_profile_transfer_secrets(value)
    invalid = tr("Remote profile contains unsupported or incompatible fields.")
    if (scope not in ('job', 'connection') or not isinstance(value, dict) or value.get('format') != 'spacr-remote-profile'
            or type(value.get('version')) is not int or value.get('version') != 1 or value.get('scope') != scope
            or set(value) - {'format', 'version', 'scope', 'profile', 'job'}):
        raise RemoteExecutionError(invalid)
    profile = value.get('profile')
    allowed = set(ExecutionProfile.__dataclass_fields__) if scope == 'job' else {'name', *_CONNECTION_PROFILE_FIELDS}
    if not isinstance(profile, dict) or set(profile) - allowed:
        raise RemoteExecutionError(invalid)
    defaults = ExecutionProfile('imported', 'ssh').to_dict()
    for key, item in profile.items():
        if key not in defaults or type(item) is not type(defaults[key]):
            raise RemoteExecutionError(invalid)
    if 'resources' in profile and any(not isinstance(item, str) for item in profile['resources'].values()):
        raise RemoteExecutionError(invalid)
    if scope == 'job':
        ExecutionProfile.from_dict(profile)
        job = value.get('job')
        if (not isinstance(job, dict)
                or set(job) != {'module', 'settings_path', 'settings_snapshot', 'allocation_id', 'allocated_gpus'}
                or not isinstance(job.get('module'), str) or job.get('module') not in MODULES
                or not isinstance(job.get('settings_path'), str)
                or '\0' in job.get('settings_path', '')
                or not isinstance(job.get('allocation_id'), str)
                or type(job.get('allocated_gpus')) is not bool
                or (job.get('settings_snapshot') is not None and not isinstance(job['settings_snapshot'], dict))
                or (job.get('settings_snapshot') is None and not job.get('settings_path', '').strip())):
            raise RemoteExecutionError(invalid)
    elif 'job' in value or set(profile) != {'name', *_CONNECTION_PROFILE_FIELDS}:
        raise RemoteExecutionError(invalid)
    encoded = json.dumps(value, allow_nan=False)
    if len(encoded.encode('utf-8')) > 2 * 1024 * 1024:
        raise RemoteExecutionError(tr("Remote profile is too large."))
    return json.loads(encoded)
_EXAMPLES = {
    "scheduler": "--constraint=v100",
    "login_hosts": "gl-login*, *login*",
    "env_setup": "module load mamba\nmamba activate spacr",
    "mail_user": "name@example.org",
    "ssh_command": "ssh -J bastion.example.org {host} {command}",
    "mem": "90G",
    "time": "48:00:00",
}
_MAIL_TYPES = ("", "END,FAIL", "ALL", "BEGIN,END,FAIL", "NONE")
_ENV_CHOICES = (
    ("module load mamba; mamba activate spacr", "module_mamba"),
    ("conda activate spacr", "conda"),
    ("source $HOME/spacr-venv/bin/activate", "venv"),
    ("pip install spacr", "pip_install"),
    ("pip install --upgrade spacr", "pip_upgrade"),
)
_SYSTEM_COLUMNS = (
    "Partition / queue", "Nodes", "CPUs per node", "Memory per node",
    "GPUs per node", "GPU types", "Time limit", "State",
)


def _item(value) -> QTableWidgetItem:
    """Return a non-editable table item."""
    item = table_item("" if value is None else str(value))
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    return item


class _SSHAuthBroker(QObject):
    """Marshal ephemeral SSH challenges onto the application GUI thread."""

    requested = Signal(object)

    def __init__(self, app):
        """Hold pending SSH challenges and connect GUI delivery and shutdown."""
        super().__init__(app)
        self._notices = []
        self._workers = []
        self._pending = []
        self.requested.connect(self._show, Qt.QueuedConnection)
        app.aboutToQuit.connect(self._shutdown)

    def _request(self, title, instructions, prompts):
        """Wait only in the SSH worker; Qt retains control of the window."""
        if QThread.currentThread() == self.thread():
            raise RemoteExecutionError(tr("SSH authentication must run in the background."))
        request = {"title": title, "instructions": instructions, "prompts": prompts,
                   "event": threading.Event(), "answers": None, "cancelled": False}
        self._pending.append(request)
        self.requested.emit(request)
        try:
            if not request["event"].wait(185):
                request["cancelled"] = True
                return None
            return request["answers"]
        finally:
            self._pending.remove(request)

    def _shutdown(self):
        """Cancel challenges before closing transports off the GUI thread."""
        for request in list(self._pending):
            request["cancelled"] = True
            request["event"].set()
        threading.Thread(target=_SSH_SESSIONS.close, daemon=True).start()

    def _show(self, request):
        """Show server text verbatim, masking prompts whose echo flag is false."""
        if request["cancelled"]:
            return
        answers = []
        for notice in self._notices:
            notice.close()
        self._notices.clear()
        if not request["prompts"] and request["instructions"]:
            notice = QMessageBox(QApplication.activeWindow())
            notice.setWindowTitle(str(request["title"]) or tr("SSH authentication"))
            notice.setTextFormat(Qt.PlainText)
            notice.setText(str(request["instructions"]))
            notice.show()
            self._notices.append(notice)
        for prompt, echo in request["prompts"]:
            dialog = QInputDialog(QApplication.activeWindow())
            dialog.setWindowTitle(str(request["title"]) or tr("SSH authentication"))
            text = str(request["instructions"])
            dialog.setLabelText((text + "\n\n" if text else "") + str(prompt))
            for label in dialog.findChildren(QLabel):
                label.setTextFormat(Qt.PlainText)
            dialog.setTextEchoMode(QLineEdit.Normal if echo else QLineEdit.Password)
            timer = QTimer(dialog)
            timer.setSingleShot(True)
            timer.timeout.connect(dialog.reject)
            timer.start(180000)
            accepted = dialog.exec() == QDialog.Accepted
            timer.stop()
            if not accepted or request["cancelled"]:
                answers.clear()
                request["event"].set()
                return
            answers.append(dialog.textValue())
            dialog.setTextValue("")
        if not request["cancelled"]:
            request["answers"] = answers
        request["event"].set()


def _ensure_ssh_broker():
    """Keep the authentication bridge alive after a profile editor closes."""
    app = QApplication.instance()
    broker = getattr(app, "_spacr_ssh_auth_broker", None)
    if broker is None:
        broker = _SSHAuthBroker(app)
        app._spacr_ssh_auth_broker = broker
    _SSH_SESSIONS.prompt = broker._request
    return broker




class _ScriptPreviewDialog(QDialog):
    """Show the rendered job script before it is sent to the remote host.

    :param parent: parent widget.
    :param script: the script exactly as it will be sent.
    :param confirm: True asks the user to submit or cancel; False only shows
        the script with a Close button.
    """

    def __init__(self, parent=None, script: str = "", confirm: bool = True):
        """Build the read-only script view and its buttons."""
        super().__init__(parent)
        self.setWindowTitle(tr("Job script preview"))
        from ..preferences import scaled_px
        self.resize(scaled_px(760), scaled_px(560))
        layout = QVBoxLayout(self)
        note = QLabel(tr("This is the script the remote host will run. spaCR adds the login-node check after the scheduler lines; everything else comes from the job script and the profile."), self)
        note.setWordWrap(True)
        note.setObjectName("Muted")
        layout.addWidget(note)
        self.text = QPlainTextEdit(self)
        self.text.setObjectName("JobScriptPreview")
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.text.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.text.setPlainText(script)
        layout.addWidget(self.text, 1)
        buttons = QDialogButtonBox(self)
        if confirm:
            submit = buttons.addButton(tr("Submit job"), QDialogButtonBox.AcceptRole)
            submit.setObjectName("PrimaryButton")
            buttons.addButton(QDialogButtonBox.Cancel)
        else:
            buttons.addButton(QDialogButtonBox.Close)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class ExecutionProfileDialog(QDialog):
    """Create or edit one distributed execution profile.

    :param parent: parent widget.
    :param profile: the profile to edit. ``None`` creates one -- the same
        dialog handles both cases.
    """

    def __init__(
        self,
        parent=None,
        profile: Optional[ExecutionProfile] = None,
    ):
        """Build the execution-profile editor.

        :param parent: parent widget, or ``None``.
        :param profile: the profile to edit; ``None`` starts a new one. Its
            original name is kept so a rename can be told from a new profile.
        """
        super().__init__(parent)
        self.setWindowTitle(tr("Execution profile"))
        from ..preferences import scaled_px
        self.setMinimumWidth(scaled_px(650))
        self._original_name = profile.name if profile else ""
        self._ssh_options = {}
        self._carried = {"resources": dict(profile.resources) if profile else {},
                         "env_setup": profile.env_setup if profile else "",
                         "job_script": profile.job_script if profile else ""}
        self._loaded_keychain = (profile.host, dict(profile.ssh_options)) if profile is not None and profile.ssh_keychain else None
        self._connection_worker = None
        _ensure_ssh_broker()
        self._build_ui()
        if profile is not None:
            self._load(profile)
        self._sync_backend()
        from .settings_model import retarget_field_tooltips
        retarget_field_tooltips(self)

    def _build_ui(self) -> None:
        """Build the backend-independent profile editor."""
        outer = QVBoxLayout(self)
        intro = QLabel(
            tr(
                "Profiles store connection settings, never passwords or OTPs. SSH prompts stay in this session."
            ),
            self,
        )
        intro.setWordWrap(True)
        intro.setObjectName("Muted")
        outer.addWidget(intro)

        form = QFormLayout()
        self._preset = QComboBox(self)
        self._preset.setObjectName("ProfilePreset")
        self._preset.addItem(tr("Start from a preset…"), "")
        self._preset.addItem(tr("Slurm with module load mamba and mamba activate spacr (Great Lakes style)"), "slurm_module_mamba")
        self._preset.addItem(tr("Slurm with conda activate spacr"), "slurm_conda")
        self._preset.addItem(tr("PBS / Torque with conda activate spacr"), "pbs_conda")
        self._preset.addItem(tr("LSF with conda activate spacr"), "lsf_conda")
        self._preset.addItem(tr("Workstation over ssh with a Python venv (no scheduler)"), "workstation_venv")
        self._preset.setToolTip(tr("Fills the backend, the environment setup, suggested resources and the login-node patterns. Review every value for your own cluster and account."))
        self._preset.activated.connect(self._apply_preset)
        form.addRow(tr("Preset"), self._preset)

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
        self._backend.addItem(tr("PBS / Torque cluster"), "pbs")
        self._backend.addItem(tr("LSF cluster"), "lsf")
        self._backend.addItem(tr("Cloud / custom command"), "command")
        self._backend.setToolTip(
            "SSH runs a durable background process; Slurm uses sbatch/squeue/"
            "sacct, PBS qsub/qstat and LSF bsub/bjobs; custom executes argument "
            "templates without shell=True."
        )
        self._backend.currentIndexChanged.connect(self._sync_backend)
        self._add_profile_row(form, "Backend", self._backend, "backend")

        self._host = QLineEdit(self)
        self._host.setObjectName("ProfileHost")
        self._host.setPlaceholderText("user@workstation or cluster-login")
        self._host.setToolTip(
            "OpenSSH target or Host alias from your SSH config. Leave blank only "
            "for scheduler commands installed on this computer."
        )
        self._add_profile_row(form, "SSH host", self._host, "host")

        discovery = QHBoxLayout()
        self._find_ssh_button = QPushButton(tr("Find my SSH settings"), self)
        self._find_ssh_button.setObjectName("FindSSHSettings")
        self._find_ssh_button.setToolTip(tr("Reads your SSH config file and the files it includes, the system-wide SSH config, the .ssh folder in your Windows user profile, and known_hosts. Read-only: spaCR never changes SSH files, never reads private keys and never runs ssh -G."))
        self._find_ssh_button.clicked.connect(lambda: self._find_ssh())
        self._choose_ssh_button = QPushButton(tr("Choose folder…"), self)
        self._choose_ssh_button.setObjectName("ChooseSSHFolder")
        self._choose_ssh_button.setToolTip(tr("Choose another folder that holds an SSH config or known_hosts file."))
        self._choose_ssh_button.clicked.connect(self._choose_ssh_folder)
        discovery.addWidget(self._find_ssh_button)
        discovery.addWidget(self._choose_ssh_button)
        form.addRow(discovery)
        self._ssh_aliases = QComboBox(self)
        self._ssh_aliases.setObjectName("SSHDiscoveredAliases")
        self._ssh_aliases.currentIndexChanged.connect(self._select_ssh)
        form.addRow(tr("SSH settings"), self._ssh_aliases)
        self._ssh_hostname = QLineEdit(self)
        self._ssh_user = QLineEdit(self)
        self._ssh_port = QSpinBox(self)
        self._ssh_port.setRange(0, 65535)
        self._ssh_port.setSpecialValueText(tr("From SSH settings"))
        self._ssh_key = QLineEdit(self)
        self._ssh_jump = QLineEdit(self)
        for label, widget, key in (("SSH host name", self._ssh_hostname, "ssh_hostname"), ("SSH user", self._ssh_user, "ssh_user"), ("SSH port", self._ssh_port, "ssh_port"),
                                   ("SSH identity file", self._ssh_key, "ssh_identity_file"), ("SSH jump host", self._ssh_jump, "ssh_jump_host")):
            widget.setObjectName(key)
            widget.setToolTip(tr("Non-secret SSH settings. Review imported paths and routing before connecting."))
            self._add_profile_row(form, label, widget, key)
        self._auth_mode = QComboBox(self)
        self._auth_mode.setObjectName("SSHSignInMethod")
        self._auth_mode.addItem(tr("OpenSSH: SSH config, keys, agent and ControlMaster"), "openssh")
        self._auth_mode.addItem(tr("Sign in in spaCR: password, Duo, Okta and other prompts"), "interactive")
        self._auth_mode.addItem(tr("Kerberos (GSSAPI) ticket from kinit"), "gssapi")
        self._auth_mode.addItem(tr("Custom SSH command"), "custom")
        self._auth_mode.setToolTip(tr("How spaCR signs in. OpenSSH uses your SSH config, keys, agent and an open ControlMaster connection. Sign in in spaCR shows the server's prompts verbatim, so a password plus Duo or Okta works; Great Lakes, for example, asks for a Kerberos password and Okta verification. Kerberos uses a ticket from kinit. Custom runs your own SSH command. spaCR stores no password or one-time code in the profile."))
        self._auth_mode.currentIndexChanged.connect(self._sync_backend)
        self._auth_label = QLabel(tr("Sign-in method"), self)
        form.addRow(self._auth_label, self._auth_mode)
        self._ssh_command = QLineEdit(self)
        self._ssh_command.setObjectName("SSHCustomCommand")
        self._ssh_command.setPlaceholderText(_EXAMPLES["ssh_command"])
        self._ssh_command.setToolTip(tr("Your own SSH command for the Custom sign-in method. {host} is replaced by the SSH host and {command} by the remote command, which is added at the end when {command} is missing. Passwords are refused here."))
        form.addRow(tr("Custom SSH command"), self._ssh_command)
        self._ssh_keychain = QCheckBox(tr("Remember my password in the system keychain"), self)
        self._ssh_keychain.setObjectName("SSHKeychainPassword")
        self._ssh_keychain.setChecked(False)
        self._ssh_keychain.setToolTip(
            tr("Off by default. Stores only the login password in the operating-system keychain, never one-time codes or push answers.")
            if _keychain_available() else tr("Install the keyring package to store the password in the system keychain."))
        form.addRow(self._ssh_keychain)
        connection = QHBoxLayout()
        self._test_ssh = QPushButton(tr("Test connection"), self)
        self._test_ssh.setObjectName("SSHTestConnection")
        self._test_ssh.clicked.connect(self._test_connection)
        self._retry_ssh = QPushButton(tr("Retry sign-in"), self)
        self._retry_ssh.setObjectName("SSHRetryConnection")
        self._retry_ssh.clicked.connect(self._test_connection)
        self._retry_ssh.setVisible(False)
        self._copy_signin = QPushButton(tr("Copy terminal sign-in command"), self)
        self._copy_signin.setObjectName("SSHCopySignIn")
        self._copy_signin.setToolTip(tr("Copies an ssh command that signs in once in a terminal, with your password and second factor, and leaves an OpenSSH ControlMaster open for four hours; spaCR's commands then reuse it without asking again."))
        self._copy_signin.clicked.connect(self._copy_signin_command)
        connection.addWidget(self._test_ssh)
        connection.addWidget(self._retry_ssh)
        connection.addWidget(self._copy_signin)
        form.addRow(connection)

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
            "Headless spaCR command on the remote target. The job script "
            "decides how it is called; the default script runs it with "
            "--settings after the environment setup lines."
        )
        self._add_profile_row(
            form, "spaCR runner", self._runner, "runner"
        )

        self._slurm = QLineEdit(self)
        self._slurm.setObjectName("ProfileSlurmOptions")
        self._slurm.setPlaceholderText(_EXAMPLES["scheduler"])
        self._slurm.setToolTip(
            "Additional sbatch, qsub or bsub arguments. They are parsed as "
            "arguments, not executed as a shell command. GPUs, CPUs, memory "
            "and time belong in Resources."
        )
        self._add_profile_row(
            form, "Slurm options", self._slurm, "scheduler_options"
        )

        self._login_hosts = QLineEdit(self)
        self._login_hosts.setObjectName("ProfileLoginHosts")
        self._login_hosts.setPlaceholderText(_EXAMPLES["login_hosts"])
        self._login_hosts.setToolTip(tr("Host-name patterns of this cluster's login nodes, separated by commas. spaCR refuses to run compute on a matching host, and on any host with a batch scheduler, unless the command runs inside a scheduler job. *login* is always included."))
        form.addRow(tr("Login hosts"), self._login_hosts)
        self._direct_run = QCheckBox(tr("Run directly over ssh (no scheduler)"), self)
        self._direct_run.setObjectName("ProfileDirectRun")
        self._direct_run.setToolTip(tr("Only for a private workstation or a cluster with no scheduler. It turns off the login-node refusal; running compute on a shared login node can get an account banned."))
        self._direct_run.toggled.connect(self._warn_direct_run)
        form.addRow(self._direct_run)

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
        """Put this field's hover help, API link and all, on its label.

        NOTHING IS DRAWN BESIDE THE LABEL. A teal API dot used to be, and
        the settings forms had already dropped theirs: a column of dots
        reads as texture rather than as one affordance per setting. The
        link is the last line of the help, where the reader was already
        finding it.
        """
        help_text = field.toolTip() or (
            f"Controls {source_label.casefold()} for distributed execution."
        )
        label = QLabel(tr(source_label), self)
        label.setObjectName("SettingsLabel")
        attach_api_tooltip(
            label, "distributed_jobs", api_key, help_text
        )
        field.setToolTip("")
        label.setCursor(Qt.WhatsThisCursor)
        label.setProperty("settingHelpLabel", True)
        self._install_help_filter(label)
        form.addRow(label, field)

    def _install_help_filter(self, label: QWidget) -> None:
        """Give ``label`` the clickable sticky popup the settings forms use.

        A native Qt tooltip cannot be walked into, so its API link cannot be
        clicked -- the pointer leaving the label takes the link with it.
        One filter per dialog: Qt keeps a LIST of event filters, so a second
        object would show two popups for one hover.
        """
        filter_ = getattr(self, "_api_tooltip_filter", None)
        if filter_ is None:
            from .settings_model import _ApiTooltipFilter
            filter_ = _ApiTooltipFilter(self)
            self._api_tooltip_filter = filter_
        filter_.watch(label)

    def _sync_backend(self, *_args) -> None:
        """Show only controls meaningful to the selected backend and sign-in method."""
        backend = self._backend.currentData()
        remote = backend in {"ssh", "slurm", "pbs", "lsf"}
        mode = self._auth_mode.currentData()
        for widget in (
            self._host, self._workdir, self._local_root, self._remote_root,
            self._runner, self._ssh_hostname, self._choose_ssh_button, self._ssh_user, self._ssh_port, self._ssh_key, self._ssh_jump,
            self._ssh_aliases, self._find_ssh_button, self._auth_mode, self._test_ssh, self._login_hosts,
        ):
            widget.setEnabled(remote)
        self._ssh_keychain.setEnabled(remote and mode == "interactive" and _keychain_available())
        self._ssh_command.setEnabled(remote and mode == "custom")
        self._copy_signin.setEnabled(remote and mode in {"openssh", "gssapi"})
        self._direct_run.setEnabled(backend == "ssh")
        self._slurm.setEnabled(backend in {"slurm", "pbs", "lsf"})
        for widget in (
            self._submit_command, self._status_command, self._cancel_command,
            self._log_command, self._job_pattern,
        ):
            widget.setEnabled(backend == "command")

    def _warn_direct_run(self, checked: bool) -> None:
        """Warn, in the dialog, the moment a direct run is switched on.

        :param checked: the new state of the box.
        """
        warning = tr("Warning: Run directly over ssh turns off the login-node refusal. Use it only for a private workstation or a cluster with no scheduler; compute on a shared login node can get an account banned.")
        if checked:
            self._error.setText(warning)
        elif self._error.text() == warning:
            self._error.clear()

    def _apply_preset(self, index: int) -> None:
        """Fill the backend, login patterns, environment and resources from a preset.

        :param index: the chosen preset row; the first row does nothing.
        """
        key = self._preset.itemData(index)
        preset = _PROFILE_PRESETS.get(key or "")
        if not preset:
            return
        self._backend.setCurrentIndex(max(0, self._backend.findData(preset["backend"])))
        self._login_hosts.setText(preset["login_hosts"])
        self._carried["env_setup"] = preset["env_setup"]
        self._carried["resources"] = dict(preset["resources"])
        self._carried["job_script"] = ""
        self._error.setText(tr("Preset applied: review the account, partition and resources for your cluster in the Resources panel after saving."))

    def _load(self, profile: ExecutionProfile) -> None:
        """Populate controls from an existing profile."""
        self._name.setText(profile.name)
        index = self._backend.findData(profile.backend)
        self._backend.setCurrentIndex(max(0, index))
        self._host.setText(profile.host)
        self._ssh_hostname.setText(profile.ssh_options.get("hostname", ""))
        self._ssh_options = dict(profile.ssh_options)
        self._ssh_user.setText(profile.ssh_options.get("user", ""))
        self._ssh_port.setValue(int(profile.ssh_options.get("port", "0")))
        self._ssh_key.setText(profile.ssh_options.get("identityfile", ""))
        self._ssh_jump.setText(profile.ssh_options.get("proxyjump", ""))
        self._auth_mode.setCurrentIndex(max(0, self._auth_mode.findData(_auth_mode(profile))))
        self._ssh_command.setText(profile.ssh_command)
        self._ssh_keychain.setChecked(profile.ssh_keychain)
        self._workdir.setText(profile.workdir)
        self._local_root.setText(profile.local_root)
        self._remote_root.setText(profile.remote_root)
        self._runner.setText(profile.runner)
        self._slurm.setText(profile.scheduler_options)
        self._login_hosts.setText(profile.login_hosts)
        self._direct_run.blockSignals(True)
        self._direct_run.setChecked(profile.direct_run)
        self._direct_run.blockSignals(False)
        self._submit_command.setText(profile.submit_command)
        self._status_command.setText(profile.status_command)
        self._cancel_command.setText(profile.cancel_command)
        self._log_command.setText(profile.log_command)
        self._job_pattern.setText(profile.job_id_pattern)
        self._poll.setValue(profile.poll_seconds)

    def profile(self) -> ExecutionProfile:
        """Return the currently entered, validated profile."""
        mode = str(self._auth_mode.currentData() or "openssh")
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
            ssh_options=self._connection_options(),
            ssh_interactive=mode == "interactive",
            ssh_keychain=mode == "interactive" and self._ssh_keychain.isChecked(),
            auth_mode=mode,
            ssh_command=self._ssh_command.text().strip() if mode == "custom" else "",
            resources=dict(self._carried["resources"]),
            env_setup=self._carried["env_setup"],
            job_script=self._carried["job_script"],
            login_hosts=self._login_hosts.text().strip(),
            direct_run=self._direct_run.isChecked() and self._backend.currentData() == "ssh",
        ).validate()

    def _find_ssh(self, folder=None):
        """Populate aliases only after an explicit read-only discovery action."""
        try:
            rows = _discover_ssh_settings(folder)
        except (OSError, ValueError, RemoteExecutionError):
            self._error.setText(tr("SSH settings could not be read. Choose a folder containing a readable config file."))
            return
        self._ssh_aliases.clear()
        for row in rows:
            self._ssh_aliases.addItem(f'{row["alias"]} → {row["hostname"]}', row)
        if not rows:
            self._error.setText(tr("No SSH aliases or public host names were found."))
        else:
            self._error.setText(tr("SSH settings imported. Conditional Match blocks are not evaluated; review the connection before testing."))
            self._select_ssh(0)

    def _choose_ssh_folder(self):
        """Read a user-selected configuration folder without copying keys."""
        folder = QFileDialog.getExistingDirectory(self, tr("Choose SSH settings folder"))
        if folder:
            self._find_ssh(folder)

    def _select_ssh(self, index):
        """Apply one alias and its explicit non-secret connection settings."""
        row = self._ssh_aliases.itemData(index)
        if not row:
            return
        try:
            port = int(row.get("port", "22"))
            if not 1 <= port <= 65535:
                raise ValueError
        except ValueError:
            self._error.setText(tr("SSH port must be between 1 and 65535."))
            return
        self._host.setText(row["alias"])
        self._ssh_options = {key: value for key, value in row.items() if key != "alias"}
        self._ssh_hostname.setText(row.get("hostname", ""))
        self._ssh_user.setText(row.get("user", ""))
        self._ssh_port.setValue(port)
        self._ssh_key.setText(row.get("identityfile", ""))
        self._ssh_jump.setText(row.get("proxyjump", ""))
        self._auth_mode.setCurrentIndex(self._auth_mode.findData("interactive"))

    def _connection_options(self):
        """Collect paths and routing fields, never authentication responses."""
        options = dict(self._ssh_options)
        for key, value in (("hostname", self._ssh_hostname.text()), ("user", self._ssh_user.text()), ("port", str(self._ssh_port.value()) if self._ssh_port.value() else ""),
                           ("identityfile", self._ssh_key.text()), ("proxyjump", self._ssh_jump.text())):
            if value.strip():
                options[key] = value.strip()
            else:
                options.pop(key, None)
        if options.get("proxyjump"):
            options.pop("proxycommand", None)
        return options

    def _connection_profile(self) -> ExecutionProfile:
        """Build a throwaway profile holding only the connection fields.

        :returns: the validated profile.
        :raises RemoteExecutionError: if a connection field is invalid.
        """
        mode = str(self._auth_mode.currentData() or "openssh")
        backend = self._backend.currentData()
        return ExecutionProfile(
            name="connection-test", backend=backend if backend in {"slurm", "pbs", "lsf"} else "ssh",
            host=self._host.text(), workdir="/tmp", ssh_options=self._connection_options(),
            ssh_interactive=mode == "interactive", auth_mode=mode,
            ssh_command=self._ssh_command.text().strip() if mode == "custom" else "",
            ssh_keychain=mode == "interactive" and self._ssh_keychain.isChecked()).validate()

    def _copy_signin_command(self) -> None:
        """Copy the terminal command that opens a reusable OpenSSH connection."""
        try:
            command = _signin_command(self._connection_profile())
        except RemoteExecutionError as exc:
            self._error.setText(str(exc))
            return
        QApplication.clipboard().setText(command)
        self._error.setText(tr("Copied. Run it in a terminal and complete the sign-in; spaCR then reuses the connection: {command}").format(command=command))

    def _test_connection(self):
        """Run hostname, then a scheduler query, on a background SSH channel."""
        if self._connection_worker is not None:
            return
        try:
            profile = self._connection_profile()
        except RemoteExecutionError as exc:
            self._error.setText(str(exc))
            return
        self._connection_error = ""
        stages = [("hostname", ["hostname"])]
        query = {"slurm": ("sinfo", ["sinfo", "--noheader"]), "pbs": ("qstat", ["qstat", "-Q"]),
                 "lsf": ("bqueues", ["bqueues"])}.get(self._backend.currentData())
        if query:
            stages.append(query)
        self._test_ssh.setEnabled(False)
        self._retry_ssh.setVisible(False)
        self._error.setText(tr("Testing SSH connection…"))
        thread, worker = make_thread(functools.partial(self._run_connection_stages, profile, stages), {},
                                     app_key="ssh_connection_test", journal=False, capture_figures=False)
        self._connection_worker = (thread, worker)
        _ensure_ssh_broker()._workers.append((self, thread, worker))
        worker.finished.connect(self._connection_finished)
        thread.finished.connect(self._connection_retired)
        thread.start()

    def _run_connection_stages(self, profile, stages, _settings) -> None:
        """Run each connection stage in turn on the worker thread, stopping at the first failure.

        :param profile: the connection profile.
        :param stages: ``(name, argv)`` pairs.
        :param _settings: the worker's settings mapping, unused.
        """
        for stage, argv in stages:
            try:
                result = _run_command(_remote_argv(profile, argv), timeout=60)
                if result.returncode:
                    raise RemoteExecutionError(tr("Remote command returned a non-zero exit status."))
            except RemoteExecutionError as exc:
                self._connection_error = tr("{stage} failed: {error}").format(stage=stage, error=exc)
                break

    def _connection_finished(self, ok):
        """Report the precise failed stage without retaining secret responses."""
        self._test_ssh.setEnabled(True)
        failed = bool(self._connection_error) or not ok
        self._retry_ssh.setVisible(failed)
        self._error.setText(self._connection_error or (tr("SSH connection test passed.") if ok else tr("SSH connection test failed.")))

    def _connection_retired(self):
        """Release a finished worker while keeping a running worker owned."""
        self._connection_worker = None
        broker = _ensure_ssh_broker()
        broker._workers[:] = [entry for entry in broker._workers if entry[0] is not self]

    def _validate_and_accept(self) -> None:
        """Keep the dialog open and show validation errors inline."""
        try:
            self.profile()
        except RemoteExecutionError as exc:
            self._error.setText(str(exc))
            return
        self._error.clear()
        if self._loaded_keychain and not self._ssh_keychain.isChecked():
            self._forget_password(*self._loaded_keychain)
        self.accept()

    def _forget_password(self, host, options):
        """Remove a remembered password off the GUI thread once storage is unticked."""
        try:
            account = _keychain_account(host, options)
        except (OSError, RemoteExecutionError):
            return
        threading.Thread(target=_keychain_password, args=("delete", account), daemon=True).start()


def _short_time(stamp: str) -> str:
    """Show an ISO timestamp as local ``YYYY-MM-DD HH:MM``.

    :param stamp: the stored UTC timestamp.
    :returns: the short local form, or the text unchanged when it is not a
        timestamp.
    """
    from datetime import datetime

    try:
        return datetime.fromisoformat(str(stamp)).astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return str(stamp or "")


def _mono(widget: QPlainTextEdit, height: int) -> QPlainTextEdit:
    """Give a code editor the fixed-width font, no wrapping and a starting height.

    :param widget: the editor.
    :param height: its minimum height in unscaled pixels.
    :returns: the editor.
    """
    from ..preferences import scaled_px
    widget.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
    widget.setLineWrapMode(QPlainTextEdit.NoWrap)
    widget.setMinimumHeight(scaled_px(height))
    return widget


class DistributedJobsScreen(QWidget):
    """Submit and monitor distributed spaCR jobs without blocking Qt.

    :param parent: parent widget.
    :param manager: the job manager to submit through. One is created when
        none is given, so a test can supply a manager that reaches no cluster.
    :param threaded: whether submissions and polls run off the GUI thread.
        False runs them inline, which is what makes a test deterministic.
    :param auto_poll: whether the screen starts its own refresh timer. False
        leaves the job list to be refreshed by hand, so a test is not racing
        a timer it did not start.
    :param preview: show the rendered job script for confirmation before a
        submission is sent; by default only when ``threaded``.
    """

    def __init__(
        self,
        parent=None,
        *,
        manager: Optional[RemoteJobManager] = None,
        threaded: bool = True,
        auto_poll: bool = True,
        preview: Optional[bool] = None,
    ):
        """Build the screen and arm its drop zone and poll timer.

        A store that cannot be read reports itself on the status line and leaves
        the screen usable with no jobs listed, rather than failing to build --
        the profiles half still works and is how the store gets fixed.

        :param parent: parent widget, or ``None``.
        :param manager: the remote job manager; ``None`` opens the default one.
        :param threaded: run submissions and polls on a worker thread.
        :param auto_poll: start the refresh timer. Off in tests, which drive
            ``refresh`` themselves.
        :param preview: confirm the rendered script before submitting.
        """
        super().__init__(parent)
        self.manager = manager or RemoteJobManager()
        self._threaded = bool(threaded)
        self._auto_poll = bool(auto_poll)
        self._preview = self._threaded if preview is None else bool(preview)
        self._jobs: List[RemoteJob] = []
        self._workers: List[tuple] = []
        self._busy = False
        self._pending_result = None
        self._pending_error = ""
        self._pending_callback: Optional[Callable] = None
        self._settings_snapshot: Optional[dict] = None
        self._snapshot_module = ""
        self._follow: dict = {}
        self._last_status: dict = {}
        self._shown_job = ""
        self._cluster: dict = {}
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
        self._last_status = {job.job_id: job.status for job in initial_jobs}
        self._render_jobs(initial_jobs)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._update_poll_interval()
        from ..dnd import install_for
        install_for(self, "distributed_jobs")
        from .settings_model import retarget_field_tooltips
        retarget_field_tooltips(self)

    def _build_ui(self) -> None:
        """Construct the submission, resources and script column beside jobs, console and System."""
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

        columns = QSplitter(Qt.Horizontal, self)
        columns.setChildrenCollapsible(False)
        self._columns = columns
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, SPACING["sm"], 0)
        left_layout.setSpacing(SPACING["md"])
        left_layout.addWidget(self._build_submission_card())
        left_layout.addWidget(self._build_resources_card())
        left_layout.addWidget(self._build_script_card())
        left_layout.addStretch(1)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(left)
        columns.addWidget(scroll)

        splitter = CollapsibleSplitter(Qt.Vertical, self,
                                       persist_key="distributed_jobs::body")
        splitter.add_section(self._build_jobs_card(), "Jobs",
                             persist_key="distributed_jobs/Jobs", stretch=2)
        from ..widgets import ConsolePanel
        console_card = Card()
        self._console_card = console_card
        self._console = ConsolePanel(active_app_label=APP_NAME,
                                     persist_key=APP_KEY, follow_log=False)
        self._console.setMinimumHeight(180)
        console_card.body_layout.addWidget(self._console, 1)
        splitter.add_section(console_card, "Console",
                             persist_key="distributed_jobs/Console", stretch=3)
        self._read_cluster_button = QPushButton("Read cluster", self)
        self._read_cluster_button.setObjectName("DistributedReadCluster")
        self._read_cluster_button.setToolTip(tr("Asks the cluster, with read-only scheduler queries (sinfo, sacctmgr, qstat or bqueues, nproc, free and nvidia-smi), which partitions, nodes, GPUs, cores, memory, accounts and QoS this profile can use."))
        self._read_cluster_button.clicked.connect(self._read_cluster)
        splitter.add_section(self._build_system_card(), "System",
                             persist_key="distributed_jobs/System", stretch=1)
        columns.addWidget(splitter)
        columns.setStretchFactor(0, 2)
        columns.setStretchFactor(1, 3)
        outer.addWidget(columns, 1)
        self._body_splitter = splitter

        self._status = QLabel("", self)
        self._status.setObjectName("Muted")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

    def _build_submission_card(self) -> Card:
        """Build the profile, module, settings and allocation controls."""
        submit_card = Card(title="Submission")
        profile_row = QHBoxLayout()
        self._profile = QComboBox(self)
        self._profile.setObjectName("ExecutionProfileChoice")
        self._profile.setProperty("i18nSkipItems", True)
        self._profile.setToolTip(
            "Saved target from spacr.remote_execution.ProfileStore."
        )
        self._profile.currentIndexChanged.connect(self._load_profile_editors)
        self._new_profile = QPushButton("New profile…", self)
        self._new_profile.clicked.connect(self._create_profile)
        self._edit_profile = QPushButton("Edit profile…", self)
        self._edit_profile.clicked.connect(self._edit_selected_profile)
        self._delete_profile = QPushButton("Delete profile", self)
        self._delete_profile.setObjectName("DangerButton")
        self._delete_profile.clicked.connect(self._delete_selected_profile)
        profile_row.addWidget(QLabel("Execution target", self))
        profile_row.addWidget(self._profile, 1)
        submit_card.body_layout.addLayout(profile_row)
        self._profile_transfer_buttons = []
        for label, name, scope, saving in (
                (tr('Save job profile…'), 'SaveJobProfile', 'job', True),
                (tr('Import job profile…'), 'ImportJobProfile', 'job', False),
                (tr('Save connection profile…'), 'SaveConnectionProfile', 'connection', True),
                (tr('Import connection profile…'), 'ImportConnectionProfile', 'connection', False)):
            if saving:
                transfer = QHBoxLayout()
                submit_card.body_layout.addLayout(transfer)
            button = QPushButton(label, self)
            button.setObjectName(name)
            button.clicked.connect(functools.partial(
                self._save_profile_transfer if saving else self._import_profile_transfer, scope))
            transfer.addWidget(button)
            self._profile_transfer_buttons.append(button)
        profile_buttons = QHBoxLayout()
        profile_buttons.addWidget(self._new_profile)
        profile_buttons.addWidget(self._edit_profile)
        profile_buttons.addWidget(self._delete_profile)
        profile_buttons.addStretch(1)
        submit_card.body_layout.addLayout(profile_buttons)

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
        job_row.addWidget(QLabel("Module", self))
        job_row.addWidget(self._module)
        job_row.addWidget(self._settings_path, 1)
        job_row.addWidget(browse)
        submit_card.body_layout.addLayout(job_row)

        from ..i18n import set_translatable_text
        submit_row = QHBoxLayout()
        self._run_inside = QComboBox(self)
        self._run_inside.setObjectName("DistributedRunInside")
        self._run_inside.setProperty("i18nSkipItems", True)
        self._run_inside.setToolTip(tr("New batch job submits through the scheduler. Choosing a running allocation runs the job inside it, as an srun step on its compute node."))
        self._alloc_button = QPushButton("Start allocation", self)
        self._alloc_button.setObjectName("DistributedStartAllocation")
        self._alloc_button.setToolTip(tr("Requests a Slurm allocation with the Resources below. It waits in the queue like any job; when it is running, choose it under Run inside and submit, so spaCR runs on the compute node and never on the login node."))
        self._alloc_button.clicked.connect(self._request_allocation)
        self._copy_shell = QPushButton("Copy shell command", self)
        self._copy_shell.setObjectName("DistributedCopyShell")
        self._copy_shell.setToolTip(tr("Copies an ssh command that opens an interactive shell inside the chosen allocation, on its compute node, for work in a terminal."))
        self._copy_shell.clicked.connect(self._copy_allocation_shell)
        submit_row.addWidget(QLabel("Run inside", self))
        submit_row.addWidget(self._run_inside, 1)
        submit_row.addWidget(self._alloc_button)
        submit_row.addWidget(self._copy_shell)
        submit_card.body_layout.addLayout(submit_row)

        action_row = QHBoxLayout()
        self._save_snapshot = QPushButton(self)
        set_translatable_text(self._save_snapshot, "Save settings CSV…")
        self._save_snapshot.setObjectName("DistributedSaveSnapshot")
        self._save_snapshot.setToolTip(tr(
            "Save the settings snapshot sent by Submit remote… as a settings "
            "CSV, to copy to a cluster and run with spacr-run --settings."))
        self._save_snapshot.setEnabled(False)
        self._save_snapshot.clicked.connect(self._save_snapshot_csv)
        self._submit = QPushButton("Submit", self)
        self._submit.setObjectName("PrimaryButton")
        self._submit.clicked.connect(self.submit)
        action_row.addWidget(self._save_snapshot)
        action_row.addStretch(1)
        action_row.addWidget(self._submit)
        submit_card.body_layout.addLayout(action_row)
        self._allocated_gpus = QCheckBox(
            tr("Mask generation: segment batches on every GPU allocated to the job"),
            self)
        self._allocated_gpus.setObjectName("DistributedAllocatedGpus")
        self._allocated_gpus.setToolTip(tr(
            "Submits mask_parallel on with a blank mask_gpu_indices, so the "
            "job uses exactly the GPUs its scheduler allocates, one model per "
            "GPU. Request them under Resources, for example 2 GPUs. A job "
            "given one GPU runs on that one."))
        submit_card.body_layout.addWidget(self._allocated_gpus)
        self._module.currentIndexChanged.connect(self._sync_allocated_gpus)
        self._sync_allocated_gpus()
        return submit_card

    def _build_resources_card(self) -> Card:
        """Build the per-profile scheduler resources form."""
        card = Card(title="Resources")
        self._resources_card = card
        form = QFormLayout()
        self._res_partition = QComboBox(self)
        self._res_account = QComboBox(self)
        self._res_qos = QComboBox(self)
        self._res_gpu_type = QComboBox(self)
        self._res_mail_type = QComboBox(self)
        for combo, name in ((self._res_partition, "ResourcePartition"), (self._res_account, "ResourceAccount"),
                            (self._res_qos, "ResourceQos"), (self._res_gpu_type, "ResourceGpuType"),
                            (self._res_mail_type, "ResourceMailType")):
            combo.setEditable(True)
            combo.setObjectName(name)
            combo.setProperty("i18nSkipItems", True)
        for value in _MAIL_TYPES:
            self._res_mail_type.addItem(value)
        self._res_gpus = QSpinBox(self)
        self._res_gpus.setRange(0, 64)
        self._res_gpus.setObjectName("ResourceGpus")
        self._res_cpus = QSpinBox(self)
        self._res_cpus.setRange(0, 4096)
        self._res_cpus.setSpecialValueText(tr("Scheduler default"))
        self._res_cpus.setObjectName("ResourceCpus")
        self._res_nodes = QSpinBox(self)
        self._res_nodes.setRange(0, 4096)
        self._res_nodes.setSpecialValueText(tr("Scheduler default"))
        self._res_nodes.setObjectName("ResourceNodes")
        self._res_mem = QLineEdit(self)
        self._res_mem.setPlaceholderText(_EXAMPLES["mem"])
        self._res_mem.setObjectName("ResourceMemory")
        self._res_time = QLineEdit(self)
        self._res_time.setPlaceholderText(_EXAMPLES["time"])
        self._res_time.setObjectName("ResourceTime")
        self._res_mail_user = QLineEdit(self)
        self._res_mail_user.setPlaceholderText(_EXAMPLES["mail_user"])
        self._res_mail_user.setObjectName("ResourceMailUser")
        self._res_extra = _mono(QPlainTextEdit(self), 56)
        from ..preferences import scaled_px
        self._res_extra.setMaximumHeight(scaled_px(96))
        self._res_extra.setObjectName("ResourceExtra")
        self._res_extra.setPlaceholderText(_EXAMPLES["scheduler"])
        rows = (
            (QLabel("Partition / queue", self), self._res_partition, tr("Slurm partition, PBS or LSF queue. Read cluster lists the ones you can use.")),
            (QLabel("Account", self), self._res_account, tr("The allocation account jobs are charged to. Read cluster lists your accounts.")),
            (QLabel("QoS", self), self._res_qos, tr("Slurm quality of service, if your cluster uses one.")),
            (QLabel("Number of GPUs", self), self._res_gpus, tr("GPUs per job. 0 asks for none.")),
            (QLabel("GPU type", self), self._res_gpu_type, tr("Optional GPU model, for example v100 or a100, as the scheduler names it.")),
            (QLabel("CPUs per task", self), self._res_cpus, tr("CPU cores for the spaCR process.")),
            (QLabel("Memory", self), self._res_mem, tr("Memory per node, for example 90G or 4000M.")),
            (QLabel("Wall time", self), self._res_time, tr("Longest run time, as HH:MM:SS or D-HH:MM:SS. The job is stopped when it runs out.")),
            (QLabel("Nodes", self), self._res_nodes, tr("Nodes for the job; spaCR runs on one.")),
            (QLabel("Mail type", self), self._res_mail_type, tr("When the scheduler emails you, for example END,FAIL.")),
            (QLabel("Mail address", self), self._res_mail_user, tr("Where the scheduler sends job emails.")),
            (QLabel("Extra scheduler lines", self), self._res_extra, tr("One option per line, added after the resource lines, for example --constraint=v100. A line starting with # is written as is.")),
        )
        for label, widget, tip in rows:
            label.setToolTip(tip)
            widget.setToolTip(tip)
            form.addRow(label, widget)
        card.body_layout.addLayout(form)
        buttons = QHBoxLayout()
        self._save_resources_button = QPushButton("Save resources", self)
        self._save_resources_button.setObjectName("PrimaryButton")
        self._save_resources_button.setToolTip(tr("Saves these resources in the selected profile; every later job of that profile asks for them."))
        self._save_resources_button.clicked.connect(self._save_resources)
        self._check_resources_button = QPushButton("Check against cluster", self)
        self._check_resources_button.setToolTip(tr("Compares these resources with the partitions, node sizes, time limits and accounts that Read cluster found, and lists what does not fit."))
        self._check_resources_button.clicked.connect(self._check_resources)
        buttons.addWidget(self._save_resources_button)
        buttons.addWidget(self._check_resources_button)
        buttons.addStretch(1)
        card.body_layout.addLayout(buttons)
        return card

    def _build_script_card(self) -> Card:
        """Build the environment-setup and job-script editors."""
        card = Card(title="Job script")
        self._script_card = card
        note = QLabel(tr("The script the remote host runs. Placeholders such as {settings_path} and {run_command} are filled at submission; nothing is appended to your commands."), self)
        note.setWordWrap(True)
        note.setObjectName("Muted")
        card.body_layout.addWidget(note)
        env_row = QHBoxLayout()
        self._env_preset = QComboBox(self)
        self._env_preset.setObjectName("EnvironmentPreset")
        self._env_preset.setProperty("i18nSkipItems", True)
        for text, key in _ENV_CHOICES:
            self._env_preset.addItem(text, key)
        self._env_preset.setToolTip(tr("Common lines that make spacr-run available on the remote host. Add appends them to the environment setup."))
        add_env = QPushButton("Add", self)
        add_env.setObjectName("EnvironmentAdd")
        add_env.setToolTip(tr("Appends the chosen lines to the environment setup."))
        add_env.clicked.connect(self._add_env_lines)
        env_row.addWidget(QLabel("Environment setup", self))
        env_row.addWidget(self._env_preset, 1)
        env_row.addWidget(add_env)
        card.body_layout.addLayout(env_row)
        self._env_edit = _mono(QPlainTextEdit(self), 64)
        from ..preferences import scaled_px
        self._env_edit.setMaximumHeight(scaled_px(120))
        self._env_edit.setObjectName("EnvironmentSetup")
        self._env_edit.setPlaceholderText(_EXAMPLES["env_setup"])
        self._env_edit.setToolTip(tr("Shell lines run before spaCR starts; the job script places them at {env_setup}. Without them spacr-run may not be found, which ends the job with exit code 127."))
        card.body_layout.addWidget(self._env_edit)
        script_row = QHBoxLayout()
        self._script_preset = QComboBox(self)
        self._script_preset.setObjectName("JobScriptPreset")
        self._script_preset.addItem(tr("Slurm sbatch script"), "slurm")
        self._script_preset.addItem(tr("PBS / Torque qsub script"), "pbs")
        self._script_preset.addItem(tr("LSF bsub script"), "lsf")
        self._script_preset.addItem(tr("Plain ssh script (no scheduler)"), "ssh")
        self._script_preset.setToolTip(tr("Starting points for the job script. Use preset replaces the editor's text."))
        use_preset = QPushButton("Use preset", self)
        use_preset.setObjectName("JobScriptUsePreset")
        use_preset.setToolTip(tr("Replaces the job script with the chosen preset."))
        use_preset.clicked.connect(self._use_script_preset)
        self._placeholder = QComboBox(self)
        self._placeholder.setObjectName("JobScriptPlaceholder")
        self._placeholder.setProperty("i18nSkipItems", True)
        for name in _JOB_SCRIPT_PLACEHOLDERS:
            self._placeholder.addItem("{" + name + "}", name)
        self._placeholder.setToolTip(tr("Values spaCR fills in at submission. Paths and resources are quoted for the shell; {env_setup}, {scheduler_directives}, {run_command}, {runner_check} and {runner} are inserted as shell code."))
        insert = QPushButton("Insert", self)
        insert.setObjectName("JobScriptInsert")
        insert.setToolTip(tr("Inserts the chosen placeholder at the cursor."))
        insert.clicked.connect(self._insert_placeholder)
        script_row.addWidget(self._script_preset, 1)
        script_row.addWidget(use_preset)
        script_row.addWidget(self._placeholder)
        script_row.addWidget(insert)
        card.body_layout.addLayout(script_row)
        self._script_edit = _mono(QPlainTextEdit(self), 220)
        self._script_edit.setObjectName("JobScriptEditor")
        card.body_layout.addWidget(self._script_edit)
        buttons = QHBoxLayout()
        self._preview_script_button = QPushButton("Preview", self)
        self._preview_script_button.setObjectName("JobScriptPreviewButton")
        self._preview_script_button.setToolTip(tr("Shows the script exactly as the next submission would send it, with every placeholder filled, and prints it to the console."))
        self._preview_script_button.clicked.connect(self._preview_script)
        self._save_script_button = QPushButton("Save script", self)
        self._save_script_button.setObjectName("PrimaryButton")
        self._save_script_button.setToolTip(tr("Saves the environment setup and job script in the selected profile."))
        self._save_script_button.clicked.connect(self._save_script)
        buttons.addWidget(self._preview_script_button)
        buttons.addWidget(self._save_script_button)
        buttons.addStretch(1)
        card.body_layout.addLayout(buttons)
        return card

    def _build_jobs_card(self) -> Card:
        """Build the jobs list inside a rounded, translucent glass card."""
        card = Card()
        self._jobs_card = card
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
        card.body_layout.addLayout(actions)
        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setObjectName("DistributedJobsTable")
        install_sorting(self._table)
        self._table.setHorizontalHeaderLabels(list(_COLUMNS))
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)
        self._table.itemSelectionChanged.connect(self._show_selection)
        card.body_layout.addWidget(self._table, 1)
        self._empty = QLabel(tr("No distributed jobs have been submitted."), self)
        self._empty.setObjectName("Muted")
        self._empty.setAlignment(Qt.AlignCenter)
        card.body_layout.addWidget(self._empty)
        return card

    def _build_system_card(self) -> Card:
        """Build the System panel: what this profile can reach on the cluster."""
        card = Card()
        self._system_card = card
        row = QHBoxLayout()
        row.addWidget(self._read_cluster_button)
        row.addStretch(1)
        card.body_layout.addLayout(row)
        self._system_summary = QLabel(tr("Not read yet. Read cluster asks the scheduler, read-only, what this profile can reach."), self)
        self._system_summary.setObjectName("Muted")
        self._system_summary.setWordWrap(True)
        card.body_layout.addWidget(self._system_summary)
        self._system_table = QTableWidget(0, len(_SYSTEM_COLUMNS))
        self._system_table.setObjectName("DistributedSystemTable")
        self._system_table.setHorizontalHeaderLabels(list(_SYSTEM_COLUMNS))
        self._system_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._system_table.verticalHeader().setVisible(False)
        self._system_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        install_sorting(self._system_table)
        card.body_layout.addWidget(self._system_table, 1)
        self._system_accounts = QLabel("", self)
        self._system_accounts.setObjectName("Muted")
        self._system_accounts.setWordWrap(True)
        self._system_accounts.setTextInteractionFlags(Qt.TextSelectableByMouse)
        card.body_layout.addWidget(self._system_accounts)
        return card

    def configure_submission(self, module: str, settings: dict) -> None:
        """Preload an immutable settings snapshot handed off by an AppScreen.

        :param module: the ``spacr.cli.MODULES`` key to select in the module
            box; an unknown key leaves the selection unchanged.
        :param settings: settings to submit. A deep copy with exact model
            identities is retained until the settings path is edited.
        """
        from ...model_zoo import _capture_model_metadata

        snapshot = _capture_model_metadata(module, settings)
        index = self._module.findData(str(module))
        if index >= 0:
            self._module.setCurrentIndex(index)
        self._settings_snapshot = snapshot
        self._snapshot_module = str(module)
        self._save_snapshot.setEnabled(True)
        self._settings_path.setText(
            tr("[current {module} settings snapshot]").format(module=module)
        )
        self._set_status(
            tr("Choose an execution profile, then Submit.")
        )

    def _sync_allocated_gpus(self, *_args) -> None:
        """Offer allocated-GPU mask batches only for the mask module."""
        self._allocated_gpus.setEnabled(
            str(self._module.currentData() or "") == "mask")

    def _clear_settings_snapshot(self, _text: str = "") -> None:
        """Switch back to file mode when the user edits the path field."""
        self._settings_snapshot = None
        self._save_snapshot.setEnabled(False)

    def _save_snapshot_csv(self) -> None:
        """Save the settings snapshot as a CSV that ``spacr-run`` reads.

        Submitting is unchanged: the snapshot stays what Submit sends. The
        file is the one Export settings… writes on the module's own screen.
        """
        if self._settings_snapshot is None:
            return
        from .app_screen import default_settings_csv_path, export_settings_csv

        module = self._snapshot_module
        default = default_settings_csv_path(
            module, self._settings_snapshot.get("src"))
        try:
            written = export_settings_csv(self, self._settings_snapshot,
                                          default)
        except Exception as exc:
            self._set_status(
                tr("Could not save the settings: {error}").format(error=exc),
                error=True)
            return
        if written:
            self._set_status(
                tr("Saved the settings to {path}. On the cluster run: "
                   "spacr-run {module} --settings {name}").format(
                       path=written[0], module=module,
                       name=os.path.basename(written[0])))

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
        self._profile.blockSignals(True)
        self._profile.clear()
        for profile in profiles:
            self._profile.addItem(profile.name, profile.name)
        index = self._profile.findData(current)
        self._profile.setCurrentIndex(index if index >= 0 else 0)
        self._profile.blockSignals(False)
        available = bool(profiles)
        self._edit_profile.setEnabled(available)
        self._delete_profile.setEnabled(available)
        self._submit.setEnabled(available and not self._busy)
        self._update_poll_interval(profiles)
        if str(self._profile.currentData() or "") != getattr(self, "_editor_profile", None):
            self._load_profile_editors()

    def _current_profile(self) -> Optional[ExecutionProfile]:
        """Return the selected saved profile, or None when there is none."""
        name = str(self._profile.currentData() or "")
        if not name:
            return None
        try:
            return self.manager.profiles.get(name)
        except RemoteExecutionError:
            return None

    def _profile_transfer_document(self, scope):
        """Capture edited profile and job choices without saving or submitting."""
        profile = self._current_profile() if scope == 'connection' else self._profile_from_editors()
        if profile is None:
            raise RemoteExecutionError(tr('Create an execution profile first.'))
        record = profile.to_dict()
        if scope == 'connection':
            record = {key: record[key] for key in ('name', *_CONNECTION_PROFILE_FIELDS)}
        document = dict(format='spacr-remote-profile', version=1, scope=scope, profile=record)
        if scope == 'job':
            document['job'] = dict(module=str(self._module.currentData() or ''),
                                   settings_path=self._settings_path.text(),
                                   settings_snapshot=self._settings_snapshot,
                                   allocation_id=str(self._run_inside.currentData() or ''),
                                   allocated_gpus=self._allocated_gpus.isChecked())
        return _validate_profile_transfer(document, scope)

    def _save_profile_transfer(self, scope, *_args):
        """Save an explicit portable profile using the existing atomic JSON writer."""
        from ...remote_execution import _write_json_atomic

        if self._busy:
            return
        try:
            document = self._profile_transfer_document(scope)
            path, _ = QFileDialog.getSaveFileName(self, tr('Save profile'), '', 'JSON (*.json)')
            if not path:
                return
            _write_json_atomic(Path(path), document)
            self._set_status(tr('Remote profile saved: {path}').format(path=path))
        except Exception as exc:
            self._set_status(tr('Remote profile operation failed: {error}').format(error=str(exc)), error=True)

    def _prepare_profile_transfer(self, document, scope):
        """Resolve compatible target and allocation choices before the preview."""
        document = _validate_profile_transfer(document, scope)
        if scope == 'connection':
            current = self._profile_from_editors()
            if current is None:
                raise RemoteExecutionError(tr('Create an execution profile first.'))
            candidate = replace(current, **{key: document['profile'][key] for key in _CONNECTION_PROFILE_FIELDS}).validate()
            allocation = str(self._run_inside.currentData() or '')
        else:
            candidate = ExecutionProfile.from_dict(document['profile'])
            allocation = document['job']['allocation_id']
        for spin, key in ((self._res_gpus, 'gpus'), (self._res_cpus, 'cpus'), (self._res_nodes, 'nodes')):
            amount = candidate.resources.get(key, '')
            if amount and (not str(amount).isdigit() or not spin.minimum() <= int(amount) <= spin.maximum()):
                raise RemoteExecutionError(tr('Remote profile contains unsupported or incompatible fields.'))
        profiles = self.manager.profiles.list()
        if allocation:
            try:
                holder = self.manager.jobs.get(allocation)
                stored = self.manager.profiles.get(holder.profile_name)
                original = ExecutionProfile.from_dict(holder.profile) if holder.profile else stored
                left, right, captured = candidate.to_dict(), stored.to_dict(), original.to_dict()
                left.pop('name')
                right.pop('name')
                captured.pop('name')
                if holder.module != 'allocation' or holder.status != 'running' or left != right or left != captured:
                    raise ValueError('allocation mismatch')
            except Exception:
                raise RemoteExecutionError(tr('Selected allocation is unavailable or incompatible with this profile.')) from None
            candidate = stored
        elif not any(profile.to_dict() == candidate.to_dict() for profile in profiles):
            names = {profile.name.casefold() for profile in profiles}
            base, counter = candidate.name[:48], 1
            name = candidate.name
            while name.casefold() in names:
                name = f'{base} imported {counter}'
                counter += 1
            candidate = replace(candidate, name=name)
        return document, candidate, allocation

    def _confirm_profile_transfer(self, document, candidate):
        """Preview the exact validated values and require Apply before mutation."""
        dialog = QDialog(self)
        dialog.setWindowTitle(tr('Remote profile preview'))
        layout = QVBoxLayout(dialog)
        note = (tr('Import all saved job settings.')
                if document['scope'] == 'job' else
                tr('Import connection settings and keep the current job settings.'))
        label = QLabel(note, dialog)
        label.setWordWrap(True)
        layout.addWidget(label)
        preview = QPlainTextEdit(dialog)
        preview.setObjectName('RemoteProfileImportPreview')
        preview.setReadOnly(True)
        shown = dict(document, profile=candidate.to_dict())
        preview.setPlainText(json.dumps(shown, indent=2, ensure_ascii=False))
        layout.addWidget(preview)
        buttons = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Cancel, parent=dialog)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        buttons.button(QDialogButtonBox.Apply).setAutoDefault(False)
        buttons.button(QDialogButtonBox.Cancel).setDefault(True)
        layout.addWidget(buttons)
        dialog.resize(720, 540)
        return dialog.exec() == QDialog.Accepted

    def _import_profile_transfer(self, scope, *_args):
        """Validate and preview a local file, then fill the form after explicit Apply."""
        if self._busy:
            return
        try:
            path, _ = QFileDialog.getOpenFileName(self, tr('Import profile'), '', 'JSON (*.json)')
            if not path:
                return
            with open(path, 'rb') as handle:
                raw = handle.read(2 * 1024 * 1024 + 1)
            if len(raw) > 2 * 1024 * 1024:
                raise RemoteExecutionError(tr('Remote profile is too large.'))
            document, candidate, allocation = self._prepare_profile_transfer(json.loads(raw), scope)
            if not self._confirm_profile_transfer(document, candidate):
                return
            jobs = None
            if allocation:
                document, candidate, allocation = self._prepare_profile_transfer(document, scope)
                jobs = self.manager.jobs.list()
                if not any(job.job_id == allocation and job.status == 'running'
                           and job.profile_name.casefold() == candidate.name.casefold() for job in jobs):
                    raise RemoteExecutionError(tr('Selected allocation is unavailable or incompatible with this profile.'))
            self.manager.profiles.save(candidate)
            if jobs is not None:
                self._jobs = jobs
            self._editor_profile = None
            self._reload_profiles(candidate.name)
            if scope == 'job':
                job = document['job']
                self._module.setCurrentIndex(self._module.findData(job['module']))
                self._settings_snapshot = job['settings_snapshot']
                self._snapshot_module = job['module'] if self._settings_snapshot is not None else ''
                self._settings_path.setText(job['settings_path'])
                self._save_snapshot.setEnabled(self._settings_snapshot is not None)
                self._allocated_gpus.setChecked(job['allocated_gpus'])
            self._run_inside.setCurrentIndex(self._run_inside.findData(allocation))
            self._set_status(tr('Remote profile imported.'))
        except Exception as exc:
            self._set_status(tr('Remote profile operation failed: {error}').format(error=str(exc)), error=True)

    def _load_profile_editors(self, *_args) -> None:
        """Show the selected profile's resources, script and remembered cluster."""
        profile = self._current_profile()
        self._editor_profile = profile.name if profile else ""
        scheduled = profile is not None and profile.backend in {"slurm", "pbs", "lsf"}
        scripted = profile is not None and profile.backend != "command"
        self._resources_card.setEnabled(scheduled)
        self._script_card.setEnabled(scripted)
        self._alloc_button.setEnabled(profile is not None and profile.backend == "slurm")
        self._read_cluster_button.setEnabled(scripted)
        resources = dict(profile.resources) if profile else {}
        for combo, key in ((self._res_partition, "partition"), (self._res_account, "account"),
                           (self._res_qos, "qos"), (self._res_gpu_type, "gpu_type"),
                           (self._res_mail_type, "mail_type")):
            combo.setEditText(resources.get(key, ""))
        for spin, key in ((self._res_gpus, "gpus"), (self._res_cpus, "cpus"), (self._res_nodes, "nodes")):
            value = resources.get(key, "")
            spin.setValue(int(value) if str(value).isdigit() else 0)
        self._res_mem.setText(resources.get("mem", ""))
        self._res_time.setText(resources.get("time", ""))
        self._res_mail_user.setText(resources.get("mail_user", ""))
        self._res_extra.setPlainText(resources.get("extra", ""))
        self._env_edit.setPlainText(profile.env_setup if profile else "")
        if profile is not None:
            self._script_preset.setCurrentIndex(max(0, self._script_preset.findData(
                profile.backend if profile.backend in _SCRIPT_PRESETS else "ssh")))
        self._script_edit.setPlainText(
            (profile.job_script or _SCRIPT_PRESETS.get(profile.backend, _SCRIPT_PRESETS["ssh"])) if scripted else "")
        self._cluster = _load_cluster_cache(profile.name) if profile else {}
        self._render_cluster(self._cluster)
        self._reload_allocations()

    def _resources_from_form(self) -> dict:
        """Collect the Resources form into a profile resources mapping."""
        values = {
            "partition": self._res_partition.currentText(), "account": self._res_account.currentText(),
            "qos": self._res_qos.currentText(), "gpu_type": self._res_gpu_type.currentText(),
            "mail_type": self._res_mail_type.currentText(), "mem": self._res_mem.text(),
            "time": self._res_time.text(), "mail_user": self._res_mail_user.text(),
            "gpus": str(self._res_gpus.value()) if self._res_gpus.value() else "",
            "cpus": str(self._res_cpus.value()) if self._res_cpus.value() else "",
            "nodes": str(self._res_nodes.value()) if self._res_nodes.value() else "",
            "extra": self._res_extra.toPlainText().strip(),
        }
        return {key: value.strip() for key, value in values.items() if value.strip()}

    def _profile_from_editors(self) -> Optional[ExecutionProfile]:
        """Return the selected profile with the editors' resources and script applied.

        :returns: the validated profile, or None when no profile is selected.
        :raises RemoteExecutionError: if an edited value is invalid.
        """
        profile = self._current_profile()
        if profile is None or profile.backend == "command":
            return profile
        script = self._script_edit.toPlainText()
        if script.strip() == _SCRIPT_PRESETS.get(profile.backend, _SCRIPT_PRESETS["ssh"]).strip():
            script = ""
        resources = self._resources_from_form() if profile.backend in {"slurm", "pbs", "lsf"} else dict(profile.resources)
        return replace(profile, resources=resources, env_setup=self._env_edit.toPlainText().strip(),
                       job_script=script).validate()

    def _store_editors(self, part: str) -> Optional[ExecutionProfile]:
        """Save the editors into the selected profile when they changed.

        :param part: ``resources``, ``script`` or ``both``; names what the
            console notice says was saved.
        :returns: the saved profile, or None when there was an error.
        """
        try:
            edited = self._profile_from_editors()
            if edited is None:
                self._warn(tr("Create an execution profile first."))
                return None
            if edited != self._current_profile():
                self.manager.profiles.save(edited)
                if part == "resources":
                    self._console.append_notice("Saved the resources in profile {profile}.\n", profile=edited.name)
                elif part == "script":
                    self._console.append_notice("Saved the job script in profile {profile}.\n", profile=edited.name)
                else:
                    self._console.append_notice("Saved the resources and job script in profile {profile}.\n",
                                                profile=edited.name)
        except RemoteExecutionError as exc:
            self._warn(str(exc))
            return None
        return edited

    def _save_resources(self) -> None:
        """Save the Resources form in the selected profile and check it against the cluster."""
        profile = self._store_editors("resources")
        if profile is not None:
            self._set_status(tr("Resources saved in {profile}.").format(profile=profile.name))
            self._check_resources()

    def _check_resources(self) -> List[str]:
        """List the requested resources the remembered cluster cannot meet.

        :returns: the problems; each is also printed to the console as a
            warning.
        """
        if not self._cluster:
            self._set_status(tr("Read cluster first, so spaCR knows this cluster's limits."))
            return []
        problems = _resource_problems(self._resources_from_form(), self._cluster)
        for problem in problems:
            self._console.append_warning(problem)
        if problems:
            self._set_status(tr("{count} resource requests do not fit this cluster; see the console.").format(count=len(problems)), error=True)
        else:
            self._set_status(tr("The resources fit what the cluster reported."))
        return problems

    def _save_script(self) -> None:
        """Save the environment setup and job script in the selected profile."""
        profile = self._store_editors("script")
        if profile is not None:
            self._set_status(tr("Job script saved in {profile}.").format(profile=profile.name))

    def _preview_script(self) -> str:
        """Render the next submission's script, print it and show it.

        :returns: the rendered script, or an empty string on an error.
        """
        try:
            profile = self._profile_from_editors()
            if profile is None or profile.backend == "command":
                self._warn(tr("Choose an SSH, Slurm, PBS or LSF profile to preview its job script."))
                return ""
            text = _preview_job_script(profile, str(self._module.currentData() or ""))
        except RemoteExecutionError as exc:
            self._warn(str(exc))
            return ""
        self._console.append_stdout(tr("Job script preview for {profile}:").format(profile=profile.name) + "\n" + text.rstrip("\n") + "\n")
        if self._preview:
            _ScriptPreviewDialog(self, text, confirm=False).exec()
        return text

    def _add_env_lines(self) -> None:
        """Append the chosen environment preset's lines to the setup editor."""
        lines = _ENV_PRESETS.get(str(self._env_preset.currentData() or ""), "")
        current = self._env_edit.toPlainText().rstrip()
        self._env_edit.setPlainText((current + "\n" if current else "") + lines)

    def _use_script_preset(self) -> None:
        """Replace the job script with the chosen preset."""
        self._script_edit.setPlainText(_SCRIPT_PRESETS[str(self._script_preset.currentData() or "ssh")])

    def _insert_placeholder(self) -> None:
        """Insert the chosen placeholder at the script editor's cursor."""
        self._script_edit.insertPlainText("{" + str(self._placeholder.currentData()) + "}")
        self._script_edit.setFocus()

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
            self._editor_profile = None
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
        if not self._threaded:
            self._run_operation(operation, {})
            self._finish_task(not bool(self._pending_error))
            return
        thread, worker = make_thread(
            functools.partial(self._run_operation, operation), {},
            app_key="distributed_jobs_io", journal=False
        )
        self._workers.append((thread, worker))
        worker.finished.connect(self._finish_task)
        thread.finished.connect(self._retire_workers)
        thread.start()

    def _run_operation(self, operation: Callable, _settings) -> None:
        """Run the operation on the job runner's thread and keep its result or error.

        :param operation: the blocking call.
        :param _settings: the worker's settings mapping, unused.
        """
        try:
            self._pending_result = operation()
        except Exception as exc:
            LOG.exception("Distributed job operation failed")
            self._pending_error = f"{type(exc).__name__}: {exc}"

    def _finish_task(self, ok: bool) -> None:
        """Apply one worker result on the GUI thread."""
        self._busy = False
        self._set_busy(False)
        if not ok or self._pending_error:
            message = self._pending_error or tr("Distributed operation failed.")
            self._set_status(message, error=True)
            self._console._display_error(message)
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
        for button in (self._submit, self._refresh, self._cancel, self._logs,
                       self._read_cluster_button, self._alloc_button, *self._profile_transfer_buttons):
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
        profile = self._store_editors("both")
        if profile is None:
            return
        if profile.backend != "command" and self._preview:
            try:
                text = _preview_job_script(profile, module_name)
            except RemoteExecutionError as exc:
                self._warn(str(exc))
                return
            if _ScriptPreviewDialog(self, text).exec() != QDialog.Accepted:
                self._set_status(tr("Submission cancelled; nothing was sent."))
                return
        from copy import deepcopy

        settings_snapshot = (
            deepcopy(self._settings_snapshot)
            if self._settings_snapshot is not None else None
        )
        use_allocated_gpus = (module_name == "mask"
                              and self._allocated_gpus.isChecked())
        allocation = str(self._run_inside.currentData() or "")
        self._start_task(tr("Submitting distributed job…"), functools.partial(
            self._submit_operation, module_name, profile_name, settings_path,
            settings_snapshot, use_allocated_gpus, allocation), self._submitted)

    def _submit_operation(self, module_name, profile_name, settings_path, settings_snapshot,
                          use_allocated_gpus, allocation):
        """Resolve the module and its settings, then submit. Off-thread.

        :param module_name: the spaCR module key.
        :param profile_name: the execution profile.
        :param settings_path: the settings file, used when there is no snapshot.
        :param settings_snapshot: settings handed over by a module screen, or None.
        :param use_allocated_gpus: run mask batches on every allocated GPU.
        :param allocation: local id of the allocation to run inside, or blank.
        :returns: the submitted job.
        """
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
        if "model_metadata" in settings:
            from ...model_zoo import _model_input_kinds, _validate_model_metadata

            _validate_model_metadata(
                settings, model_kinds=_model_input_kinds(module, settings))
        if use_allocated_gpus:
            settings = dict(settings, mask_parallel=True,
                            mask_gpu_indices="")
        if allocation:
            return _submit_job(self.manager, module.key, settings, profile_name, allocation)
        return self.manager.submit(module.key, settings, profile_name)

    def _submitted(self, job: RemoteJob) -> None:
        """Redraw the job list with the new job selected and start following its log.

        :param job: the submitted job.
        """
        self._last_status[job.job_id] = job.status
        if job.log_reference.startswith("/") or job.backend == "command":
            self._follow[job.job_id] = 0
        self._shown_job = job.job_id
        self._render_jobs(self.manager.jobs.list(), select=job.job_id)
        self._console.append_notice("Submitted {module} as {job} through {profile}; scheduler id {external}.\n",
                                    module=job.module, job=job.job_id[:12], profile=job.profile_name,
                                    external=job.external_id or "-")
        if job.job_script:
            self._console.append_stdout(job.job_script.rstrip("\n") + "\n")
        self._set_status(
            tr("Submitted {module} as {job}.").format(
                module=job.module, job=job.job_id
            )
        )

    def _request_allocation(self) -> None:
        """Request a Slurm allocation with the selected profile's resources."""
        profile = self._store_editors("resources")
        if profile is None:
            return
        self._start_task(tr("Requesting an allocation…"),
                         functools.partial(_start_allocation, self.manager, profile.name),
                         self._allocation_requested)

    def _allocation_requested(self, job: RemoteJob) -> None:
        """Report a requested allocation and list it.

        :param job: the allocation's job record.
        """
        self._last_status[job.job_id] = job.status
        self._follow[job.job_id] = 0
        self._render_jobs(self.manager.jobs.list(), select=job.job_id)
        self._console.append_notice("Requested allocation {job} (Slurm {external}); it can run jobs once it is running.\n",
                                    job=job.job_id[:12], external=job.external_id)
        self._set_status(tr("Allocation requested; it waits in the queue like any job."))

    def _reload_allocations(self) -> None:
        """List the selected profile's running allocations under Run inside."""
        current = str(self._run_inside.currentData() or "")
        name = str(self._profile.currentData() or "").casefold()
        self._run_inside.clear()
        self._run_inside.addItem(tr("New batch job"), "")
        for job in self._jobs:
            if job.module == "allocation" and job.status == "running" and job.profile_name.casefold() == name:
                self._run_inside.addItem(tr("Allocation {job} (Slurm {external})").format(
                    job=job.job_id[:12], external=job.external_id), job.job_id)
        self._run_inside.setCurrentIndex(max(0, self._run_inside.findData(current)))
        self._copy_shell.setEnabled(self._run_inside.count() > 1)

    def _copy_allocation_shell(self) -> None:
        """Copy the command that opens a shell inside the chosen allocation."""
        job_id = str(self._run_inside.currentData() or "")
        holder = next((job for job in self._jobs if job.job_id == job_id), None)
        if holder is None:
            self._warn(tr("Choose a running allocation under Run inside first."))
            return
        command = _allocation_shell_command(ExecutionProfile.from_dict(holder.profile), holder.external_id)
        QApplication.clipboard().setText(command)
        self._console.append_stdout(command + "\n")
        self._set_status(tr("Copied the shell command; paste it in a terminal."))

    def refresh(self) -> None:
        """Poll all non-terminal jobs, and follow their logs, in a worker thread."""
        self._start_task(
            tr("Polling remote jobs…"),
            functools.partial(self._poll, dict(self._follow)),
            self._polled,
        )

    def _poll(self, follow: dict):
        """Poll every active job, then fetch the new part of each followed log. Off-thread.

        :param follow: job id to bytes of log already shown.
        :returns: ``(jobs, {job id: (text, offset, error)})``.
        """
        jobs = self.manager.refresh_all(include_logs=False)
        chunks = {}
        for job_id, offset in follow.items():
            try:
                text, new_offset = _log_chunk(self.manager, job_id, offset)
                chunks[job_id] = (text, new_offset, "")
            except Exception as exc:
                chunks[job_id] = ("", offset, f"{type(exc).__name__}: {exc}")
        return jobs, chunks

    def _polled(self, result) -> None:
        """Print new log text and status changes, then redraw the jobs.

        :param result: what :meth:`_poll` returned.
        """
        jobs, chunks = result
        by_id = {job.job_id: job for job in jobs}
        for job_id, (text, offset, error) in chunks.items():
            job = by_id.get(job_id)
            previous_offset = self._follow.get(job_id)
            if text:
                self._console.append_stdout(f"[{job_id[:12]} {job.module if job else ''}]\n{text.rstrip(chr(10))}\n")
            self._follow[job_id] = offset
            if error:
                self._console.append_warning(tr("Could not read the log of {job}: {error}").format(job=job_id[:12], error=error))
            elif (job is not None and job.status in TERMINAL_STATES
                  and not text and offset == previous_offset):
                self._follow.pop(job_id, None)
        for job in jobs:
            self._report_transition(job)
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

    def _report_transition(self, job: RemoteJob) -> None:
        """Print a job's status change to the console; a failure goes through spaCR AI.

        :param job: the freshly polled job.
        """
        before = self._last_status.get(job.job_id)
        self._last_status[job.job_id] = job.status
        if before is None or before == job.status:
            return
        if job.status == "failed":
            lines = [tr("Remote job {job} ({module}, profile {profile}) failed with exit code {code}.").format(
                job=job.job_id[:12], module=job.module, profile=job.profile_name,
                code="?" if job.exit_code is None else job.exit_code)]
            if job.error:
                lines.append(job.error)
            lines.append(self._job_detail(job))
            self._console.append_error("\n".join(lines))
        else:
            self._console.append_notice("Job {job} ({module}) is now {status}.\n", job=job.job_id[:12],
                                        module=job.module, status=job.status)

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
        self._start_task(
            tr("Cancelling remote job…"),
            functools.partial(self.manager.cancel, job.job_id),
            self._cancelled,
        )

    def _cancelled(self, updated: RemoteJob) -> None:
        """Redraw with the cancelled job still selected.

        The selection is kept so the list does not jump to the top after
        a cancel and the user's row stays selected.

        :param updated: the cancelled job.
        """
        self._report_transition(updated)
        self._render_jobs(self.manager.jobs.list(), select=updated.job_id)
        self._set_status(
            tr("Cancellation requested for {job}.").format(
                job=updated.job_id
            )
        )

    def refresh_log(self) -> None:
        """Fetch the selected job's remote log tail into the console."""
        job = self._selected_job()
        if job is None:
            self._set_status(tr("Select a job first."), error=True)
            return
        self._start_task(
            tr("Retrieving remote log…"),
            functools.partial(self.manager.logs, job.job_id, 500),
            functools.partial(self._log_fetched, job.job_id),
        )

    def _log_fetched(self, job_id: str, text: str) -> None:
        """Print the fetched log tail under the job's header.

        :param job_id: the job.
        :param text: the tail.
        """
        updated = self.manager.jobs.get(job_id)
        self._render_jobs(self.manager.jobs.list(), select=updated.job_id)
        self._console.append_stdout(
            f"--- remote log tail {updated.job_id[:12]} ({updated.module}) ---\n{text.rstrip(chr(10))}\n")
        self._set_status(tr("Log refreshed."))

    def _read_cluster(self) -> None:
        """Ask the cluster, read-only, what the selected profile can reach."""
        profile = self._current_profile()
        if profile is None or profile.backend == "command":
            self._warn(tr("Choose an SSH, Slurm, PBS or LSF profile to read its cluster."))
            return
        self._start_task(tr("Reading cluster resources…"),
                         functools.partial(_probe_cluster, profile, self.manager.runner),
                         functools.partial(self._cluster_read, profile.name))

    def _cluster_read(self, name: str, cluster: dict) -> None:
        """Show, remember and check a fresh cluster probe.

        :param name: the profile name.
        :param cluster: the parsed probe.
        """
        try:
            _save_cluster_cache(name, cluster)
        except RemoteExecutionError as exc:
            self._console.append_warning(str(exc))
        if str(self._profile.currentData() or "").casefold() != name.casefold():
            self._set_status(tr("Cluster resources read for {profile}.").format(profile=name))
            return
        self._cluster = cluster
        self._render_cluster(cluster)
        self._console.append_stdout(self._system_summary.text() + "\n" + self._system_accounts.text() + "\n")
        problems = _resource_problems(self._resources_from_form(), cluster)
        for problem in problems:
            self._console.append_warning(problem)
        self._set_status(tr("Cluster resources read for {profile}.").format(profile=name))

    def _render_cluster(self, cluster: dict) -> None:
        """Draw the System panel and offer the cluster's values as suggestions.

        :param cluster: a parsed probe, or an empty mapping.
        """
        _retire_table_cells(self._system_table)
        self._system_table.setRowCount(0)
        if not cluster:
            self._system_summary.setText(tr("Not read yet. Read cluster asks the scheduler, read-only, what this profile can reach."))
            self._system_accounts.setText("")
            return
        scheduler = cluster.get("scheduler") or tr("no scheduler")
        gpus = ", ".join(cluster.get("gpus") or []) or tr("none visible")
        self._system_summary.setText(tr("{host} as {user}; scheduler: {scheduler}. This host: {cpus} CPUs, {memory} GB memory, GPUs: {gpus}.").format(
            host=cluster.get("host") or "?", user=cluster.get("user") or "?", scheduler=scheduler,
            cpus=cluster.get("cpus") or "?", memory=round((cluster.get("mem_mb") or 0) / 1024) or "?", gpus=gpus))
        rows = [(row["partition"] + (" *" if row["default"] else ""), row["nodes"], row["cpus"],
                 f"{round(row['mem_mb'] / 1024)} GB" if row["mem_mb"] else "", row["gpus"] or "",
                 ", ".join(row["gpu_types"]), row["time_limit"], row["available"])
                for row in cluster.get("partitions") or []]
        rows += [(queue, "", "", "", "", "", "", "PBS") for queue in cluster.get("pbs_queues") or []]
        rows += [(queue, "", "", "", "", "", "", "LSF") for queue in cluster.get("lsf_queues") or []]
        self._system_table.setSortingEnabled(False)
        for values in rows:
            row = self._system_table.rowCount()
            self._system_table.insertRow(row)
            for column, value in enumerate(values):
                self._system_table.setItem(row, column, _item(value))
        self._system_table.setSortingEnabled(True)
        accounts = cluster.get("accounts") or []
        self._system_accounts.setText(tr("Your accounts: {accounts}").format(accounts="; ".join(
            f"{row['account']}" + (f" ({row['partition']})" if row["partition"] else "")
            + (f" QoS {', '.join(row['qos'])}" if row["qos"] else "") for row in accounts)) if accounts else "")
        for combo, values in (
                (self._res_partition, [row["partition"] for row in cluster.get("partitions") or []]
                 + list(cluster.get("pbs_queues") or []) + list(cluster.get("lsf_queues") or [])),
                (self._res_account, [row["account"] for row in accounts]),
                (self._res_qos, [qos for row in accounts for qos in row["qos"]]),
                (self._res_gpu_type, [model for row in cluster.get("partitions") or [] for model in row["gpu_types"]])):
            text = combo.currentText()
            combo.clear()
            combo.addItems([""] + list(dict.fromkeys(values)))
            combo.setEditText(text)

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
        _retire_table_cells(self._table)
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
        self._table.blockSignals(True)
        for job in self._jobs:
            row = self._table.rowCount()
            self._table.insertRow(row)
            values = (
                job.job_id[:12], job.status, job.module, job.profile_name,
                job.external_id or "—", _short_time(job.created_utc), _short_time(job.updated_utc),
            )
            for column, value in enumerate(values):
                cell = _spare_table_item(self._table, "" if value is None else str(value), make=_item)
                cell.setFlags(cell.flags() & ~Qt.ItemIsEditable)
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
        self._table.blockSignals(False)
        self._empty.setVisible(not self._jobs)
        self._reload_allocations()
        self._show_selection()

    @staticmethod
    def _job_summary(job: RemoteJob) -> str:
        """Format the few facts of a job that matter when it is selected.

        :param job: the job.
        :returns: one block: id, module, state, scheduler id, profile,
            remote folder and log, plus the last error.
        """
        lines = [f"job {job.job_id}  {job.module}  {job.status}"
                 + (f"  exit {job.exit_code}" if job.exit_code is not None else ""),
                 f"profile {job.profile_name} ({job.backend})  scheduler id {job.external_id or '-'}"
                 + (f"  inside allocation {job.allocation_id}" if job.allocation_id else "")]
        if job.remote_job_dir:
            lines.append(f"remote folder {job.remote_job_dir}")
        if job.log_reference:
            lines.append(f"log {job.log_reference}")
        if job.error:
            lines.append(job.error)
        return "\n".join(lines)

    @staticmethod
    def _job_detail(job: RemoteJob) -> str:
        """Format a complete, copyable job record plus retained log."""
        record = job.to_dict()
        log = record.pop("log_tail", "")
        record.pop("job_script", None)
        profile = dict(record.get("profile") or {})
        for key in (
            "submit_command", "status_command", "cancel_command", "log_command",
            "ssh_command",
        ):
            if profile.get(key):
                profile[key] = "<configured command>"
        record["profile"] = profile
        text = json.dumps(record, indent=2, sort_keys=True, default=str)
        if log:
            text += "\n\n--- remote log tail ---\n" + log
        return text

    def _show_selection(self) -> None:
        """Print a newly selected job's record to the console and update the actions."""
        job = self._selected_job()
        if job is None:
            self._cancel.setEnabled(False)
            self._logs.setEnabled(False)
            self._open_local.setEnabled(False)
            return
        if job.job_id != self._shown_job:
            self._shown_job = job.job_id
            self._console.append_stdout(self._job_summary(job) + "\n")
        if (job.status in ACTIVE_STATES and job.job_id not in self._follow
                and job.log_reference.startswith("/")):
            self._follow[job.job_id] = 0
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
            QUrl.fromLocalFile(str(Path(job.settings_path or self.manager.jobs.path).parent))
        )

    def _set_status(self, text: str, error: bool = False) -> None:
        """Set the status line with theme-aware severity."""
        self._status.setText(str(text))
        palette = active_palette()
        self._status.setStyleSheet(
            f"color: {palette['error' if error else 'fg_dim']};"
        )

    def _warn(self, text: str) -> None:
        """Show a problem the user can fix on the status line and as a console warning.

        :param text: the message.
        """
        self._set_status(text, error=True)
        self._console.append_warning(text)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Poll only while this screen is visible.

        :param event: the show event; passed on unchanged to the base class
            before polling restarts (when auto-polling is on).
        """
        super().showEvent(event)
        if self._auto_poll and not self._timer.isActive():
            self._timer.start()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Stop background polling when another module is open.

        :param event: the hide event; passed on unchanged to the base class.
        """
        self._timer.stop()
        super().hideEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Stop polling and drain live workers before the screen goes away.

        ``requestInterruption`` alone has no effect — ``_run_operation`` never
        polls it — and would leave the screen's REST calls running with no
        owner. An ownerless job stays in the process-wide run registry, which
        is what ``MainWindow.closeEvent`` consults when it decides whether the
        application may quit.

        :param event: the close event; it is passed on unchanged to the
            base-class handler after the workers are drained.
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
