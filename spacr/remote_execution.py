"""Persistent remote and distributed execution for spaCR pipelines.

The local GUI and :mod:`spacr.cli` already agree on one headless contract::

    spacr-run MODULE --settings SETTINGS.json

This module transports that contract to a workstation over SSH, submits it to
Slurm, or hands it to a user-configured cloud/HPC command.  Submitted jobs are
recorded locally and can be polled or cancelled after spaCR itself has closed.

No command uses ``shell=True``.  SSH necessarily invokes a remote login shell;
all user-controlled values interpolated into its small fixed scripts are
quoted with :func:`shlex.quote`, and host names are validated separately.
Custom command profiles are parsed into argument vectors and substitute each
placeholder inside one argument, so shell operators have no special meaning.
"""
from __future__ import annotations

import contextlib
import base64
import codecs
import fnmatch
import functools
import hashlib
import json
import os
import posixpath
import re
import shlex
import subprocess
import threading
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .qt.i18n import tr

__all__ = [
    "ACTIVE_STATES",
    "TERMINAL_STATES",
    "BACKENDS",
    "CommandResult",
    "ExecutionProfile",
    "RemoteJob",
    "ProfileStore",
    "JobStore",
    "RemoteExecutionError",
    "RemoteJobManager",
    "map_settings_paths",
    "state_directory",
]

BACKENDS = ("ssh", "slurm", "pbs", "lsf", "command")
_AUTH_MODES = ("openssh", "interactive", "gssapi", "custom")
_SCHEDULERS = ("slurm", "pbs", "lsf")
_RESOURCE_KEYS = ("partition", "account", "qos", "gpus", "gpu_type", "cpus",
                  "mem", "time", "nodes", "mail_type", "mail_user", "extra")
_RESOURCE_PATTERNS = {
    "partition": r"[A-Za-z0-9_.,-]+", "account": r"[A-Za-z0-9_.-]+",
    "qos": r"[A-Za-z0-9_.-]+", "gpus": r"[0-9]{1,3}", "gpu_type": r"[A-Za-z0-9_.-]+",
    "cpus": r"[0-9]{1,4}", "mem": r"[0-9]+(\.[0-9]+)?([KMGT]B?|[kmgt]b?)?",
    "time": r"([0-9]+-)?[0-9]{1,4}(:[0-9]{1,2}){0,2}", "nodes": r"[0-9]{1,4}",
    "mail_type": r"[A-Za-z_,]+", "mail_user": r"[^\s'\"\\;&|<>`$]+",
}
_LOGIN_PATTERN_RE = re.compile(r"^[A-Za-z0-9*?._\[\]-]+$")
_DEFAULT_LOGIN_PATTERNS = ("*login*",)
_LOGIN_REFUSED_EXIT = 97
_SCRIPT_VALUE_KEYS = ("settings_path", "job_dir", "module", "job_name", "log_dir",
                      "log_path", "workdir", "job_id", "gpus", "cpus", "mem", "time",
                      "partition", "account", "qos")
_SCRIPT_CODE_KEYS = ("env_setup", "scheduler_directives", "run_command",
                     "runner_check", "runner")
_JOB_SCRIPT_PLACEHOLDERS = _SCRIPT_VALUE_KEYS + _SCRIPT_CODE_KEYS
_PLACEHOLDER_RE = re.compile(r"(?<![$\w{])\{([A-Za-z_][A-Za-z0-9_]*)\}(?!\})")
_SCRIPT_BODY = (
    "{env_setup}\n\n"
    "cd {workdir}\n"
    "echo \"spaCR job {job_name} on $(hostname)\"\n"
    "export PYTHONUNBUFFERED=1\n"
    "{runner_check}\n"
    "exec {run_command}\n"
)
_SCRIPT_PRESETS = {
    "slurm": "#!/bin/bash -l\n{scheduler_directives}\n\n" + _SCRIPT_BODY.replace(
        "on $(hostname)", "on $(hostname), ${SLURM_CPUS_PER_TASK:-1} CPUs, GPUs ${CUDA_VISIBLE_DEVICES:-none}"),
    "pbs": "#!/bin/bash -l\n{scheduler_directives}\n\nexec >> {log_path} 2>&1\n" + _SCRIPT_BODY,
    "lsf": "#!/bin/bash -l\n{scheduler_directives}\n\nexec >> {log_path} 2>&1\n" + _SCRIPT_BODY,
    "ssh": "#!/bin/bash -l\n" + _SCRIPT_BODY,
}
_ENV_PRESETS = {
    "module_mamba": "module load mamba\nmamba activate spacr",
    "conda": "eval \"$(conda shell.bash hook)\"\nconda activate spacr",
    "venv": "source \"$HOME/spacr-venv/bin/activate\"",
    "pip_install": "pip install spacr",
    "pip_upgrade": "pip install --upgrade spacr",
}
_PROFILE_PRESETS = {
    "slurm_module_mamba": {
        "backend": "slurm", "env_setup": _ENV_PRESETS["module_mamba"],
        "resources": {"partition": "gpu", "gpus": "1", "cpus": "20", "mem": "90G",
                      "time": "48:00:00", "mail_type": "END,FAIL"},
        "login_hosts": "*login*",
    },
    "slurm_conda": {
        "backend": "slurm", "env_setup": _ENV_PRESETS["conda"],
        "resources": {"cpus": "8", "mem": "32G", "time": "12:00:00"},
        "login_hosts": "*login*",
    },
    "pbs_conda": {
        "backend": "pbs", "env_setup": _ENV_PRESETS["conda"],
        "resources": {"cpus": "8", "mem": "32G", "time": "12:00:00"},
        "login_hosts": "*login*",
    },
    "lsf_conda": {
        "backend": "lsf", "env_setup": _ENV_PRESETS["conda"],
        "resources": {"cpus": "8", "mem": "32G", "time": "12:00:00"},
        "login_hosts": "*login*",
    },
    "workstation_venv": {
        "backend": "ssh", "env_setup": _ENV_PRESETS["venv"], "resources": {},
        "login_hosts": "",
    },
}
ACTIVE_STATES = frozenset({"submitting", "queued", "pending", "running"})
TERMINAL_STATES = frozenset({"success", "failed", "cancelled"})
_ALL_STATES = ACTIVE_STATES | TERMINAL_STATES | {"unknown"}
_HOST_RE = re.compile(r"^[A-Za-z0-9_.@:-]+$")
_PROFILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_. -]{0,63}$")
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_.:+-]+$")
_LOCK = threading.RLock()


class RemoteExecutionError(RuntimeError):
    """A profile, submission, polling, or cancellation error.

    These errors are safe to show directly in the GUI: passwords and
    environment variables are never included in command rendering.
    """


@dataclass(frozen=True)
class CommandResult:
    """Result returned by the injectable command runner.

    :param returncode: process exit status, where zero denotes success.
    :param stdout: captured standard-output text.
    :param stderr: captured standard-error text.
    """

    returncode: int
    stdout: str = ""
    stderr: str = ""


def _run_command(
    argv: Sequence[str],
    *,
    input_text: Optional[str] = None,
    timeout: float = 60.0,
) -> CommandResult:
    """Run one argument vector without a shell and capture UTF-8 output.

    ``timeout`` is floored at one second: a sub-second budget is shorter than
    process startup on any real machine, so honouring it literally would kill
    every command. The floored value is what is WAITED and therefore what the
    timeout message must name -- reporting the raw request would print a
    number the user cannot reconcile with the wall clock.
    """
    if argv and argv[0] == "__spacr_interactive_ssh__":
        return _SSH_SESSIONS.run(json.loads(argv[1]), argv[2], input_text=input_text, timeout=max(1.0, float(timeout)))
    waited = max(1.0, float(timeout))
    try:
        result = subprocess.run(
            list(argv),
            input=input_text,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=waited,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RemoteExecutionError(
            f"Command not found: '{argv[0]}'. Install it or update the "
            "execution profile."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RemoteExecutionError(
            f"Command timed out after {waited:g}s: {argv[0]}"
        ) from exc
    except OSError as exc:
        raise RemoteExecutionError(
            f"Could not start '{argv[0]}': {type(exc).__name__}: {exc}"
        ) from exc
    return CommandResult(result.returncode, result.stdout, result.stderr)


CommandRunner = Callable[..., CommandResult]


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def state_directory() -> Path:
    """Return the persistent directory for profiles, jobs, settings and logs.

    ``SPACR_REMOTE_STATE_DIR`` is intentionally supported for tests, portable
    deployments, and managed lab installations.  Otherwise XDG state storage
    is used on Linux and a conventional per-user directory elsewhere.
    """
    override = os.environ.get("SPACR_REMOTE_STATE_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_STATE_HOME", "").strip()
    if xdg:
        return Path(xdg).expanduser() / "spacr" / "remote"
    return Path.home() / ".local" / "state" / "spacr" / "remote"


def _safe_text(value: Any, label: str, *, allow_empty: bool = False) -> str:
    """Validate one field that will be interpolated into a remote command.

    :param value: the field.
    :param label: its name, used in the error.
    :param allow_empty: accept an empty value.
    :returns: the text, stripped.
    :raises RemoteExecutionError: if it is empty when it must not be, or if
        it contains a newline or a NUL -- either would let one field become
        two arguments, or terminate a shell line early, on the far side of
        the connection.
    """
    text = str(value or "").strip()
    if not text and not allow_empty:
        raise RemoteExecutionError(f"{label} is required.")
    if "\x00" in text or "\n" in text or "\r" in text:
        raise RemoteExecutionError(f"{label} may not contain newlines or NUL.")
    return text


def _path_text(value: Any, label: str, *, allow_empty: bool = False) -> str:
    """Validate one field that must be an absolute POSIX path.

    :param value: the path.
    :param label: its name, used in the error.
    :param allow_empty: accept an empty value.
    :returns: the path with any trailing slash removed, except that ``/``
        stays ``/``.
    :raises RemoteExecutionError: if it fails the text checks, or if it is
        relative -- a relative path on the remote resolves against whatever
        the login shell happens to leave as the working directory.
    """
    text = _safe_text(value, label, allow_empty=allow_empty)
    if text and not text.startswith("/"):
        raise RemoteExecutionError(f"{label} must be an absolute POSIX path.")
    return text.rstrip("/") or ("/" if text else "")


@dataclass(frozen=True)
class ExecutionProfile:
    """Connection and scheduler settings for one execution target.

    ``local_root`` and ``remote_root`` describe the same shared or mirrored
    dataset.  Every absolute string nested in the settings is rewritten when
    it lies below ``local_root``.  Image datasets are deliberately not copied:
    accidental recursive transfer of a multi-terabyte plate is worse than a
    clear pre-flight error.

    For ``command`` profiles, command strings are tokenized with
    :func:`shlex.split` and support ``{job_id}``, ``{module}``, ``{settings}``
    and ``{external_id}`` placeholders.  They are argument templates, not
    shell scripts.

    :param name: human-readable profile name used to select this target.
    :param backend: execution mechanism, one of :data:`BACKENDS`.
    :param host: OpenSSH target for SSH and remote Slurm; blank permits a
        Slurm scheduler running on the local host.
    :param workdir: absolute execution directory where ``spacr-jobs``
        artifacts are created.
    :param local_root: local dataset prefix paired with ``remote_root`` for
        nested absolute-path rewriting.
    :param remote_root: equivalent dataset prefix visible to the target host.
    :param runner: headless executable invoked as ``runner module --settings
        path`` by SSH and Slurm backends.
    :param scheduler_options: additional shell-free arguments passed to
        ``sbatch``.
    :param submit_command: custom-backend argument template containing
        ``{settings}`` and printing a job identifier.
    :param status_command: custom status template containing ``{external_id}``
        and returning a state spaCR can normalize.
    :param cancel_command: custom cancellation template containing
        ``{external_id}``.
    :param log_command: optional custom template for fetching job output.
    :param job_id_pattern: optional regular expression extracting a named
        ``id``, first capture, or full match from submission output.
    :param poll_seconds: recommended client polling interval, from 2 through
        3600 seconds.
    :param ssh_options: non-secret SSH configuration overrides, such as user,
        port, identity-file path and jump host; passwords and OTPs are forbidden.
    :param ssh_interactive: use a verified persistent SSH session with GUI
        keyboard-interactive prompts instead of the external OpenSSH command.
    :param ssh_keychain: remember the login password, never one-time codes or
        push answers, in the operating-system keychain; off by default.
    :param auth_mode: how spaCR signs in: ``openssh`` (SSH config, keys,
        agent and ControlMaster reuse), ``interactive`` (password and
        second-factor prompts shown in spaCR), ``gssapi`` (Kerberos ticket)
        or ``custom`` (the user's own SSH command). Blank follows
        ``ssh_interactive``.
    :param ssh_command: custom SSH command for ``custom`` sign-in; ``{host}``
        is the target and ``{command}`` the remote command, appended when
        absent. It may not carry a password.
    :param resources: scheduler resources: partition, account, qos, gpus,
        gpu_type, cpus, mem, time, nodes, mail_type, mail_user and extra
        scheduler lines.
    :param env_setup: shell lines that prepare the remote environment before
        spaCR runs, such as ``module load mamba`` and ``mamba activate spacr``.
    :param job_script: editable job-script template with placeholders such as
        ``{settings_path}`` and ``{run_command}``; blank uses the backend's
        default script.
    :param login_hosts: comma-separated host-name patterns of login nodes on
        which spaCR must never run compute.
    :param direct_run: explicit opt-in to run directly over SSH on a host with
        no scheduler, skipping the login-node refusal.
    """

    name: str
    backend: str
    host: str = ""
    workdir: str = ""
    local_root: str = ""
    remote_root: str = ""
    runner: str = "spacr-run"
    scheduler_options: str = ""
    submit_command: str = ""
    status_command: str = ""
    cancel_command: str = ""
    log_command: str = ""
    job_id_pattern: str = ""
    poll_seconds: int = 10
    ssh_options: Dict[str, str] = field(default_factory=dict)
    ssh_interactive: bool = False
    ssh_keychain: bool = False
    auth_mode: str = ""
    ssh_command: str = ""
    resources: Dict[str, str] = field(default_factory=dict)
    env_setup: str = ""
    job_script: str = ""
    login_hosts: str = ""
    direct_run: bool = False

    def validate(self) -> "ExecutionProfile":
        """Validate the profile and return ``self`` for fluent callers."""
        name = _safe_text(self.name, "Profile name")
        if not _PROFILE_RE.fullmatch(name):
            raise RemoteExecutionError(
                "Profile name must begin with a letter or number and contain "
                "only letters, numbers, spaces, '.', '_' or '-'."
            )
        if self.backend not in BACKENDS:
            raise RemoteExecutionError(
                f"Unknown backend {self.backend!r}; choose one of "
                f"{', '.join(BACKENDS)}."
            )
        if not 2 <= int(self.poll_seconds) <= 3600:
            raise RemoteExecutionError("Poll interval must be 2–3600 seconds.")

        if not isinstance(self.ssh_options, dict) or any(
                key not in {"config", "hostname", "user", "port", "identityfile", "proxyjump", "proxycommand",
                            "controlmaster", "controlpersist", "controlpath", "userknownhostsfile"}
                or not isinstance(value, str) or "\n" in value or "\r" in value or "\0" in value
                for key, value in self.ssh_options.items()):
            raise RemoteExecutionError(tr("SSH options must contain only supported non-secret connection fields."))
        if "port" in self.ssh_options:
            try:
                valid_port = 1 <= int(self.ssh_options["port"]) <= 65535
            except ValueError:
                valid_port = False
            if not valid_port:
                raise RemoteExecutionError(tr("SSH port must be between 1 and 65535."))
        if self.ssh_options.get("proxyjump") and self.ssh_options.get("proxycommand"):
            raise RemoteExecutionError(tr("Choose ProxyJump or ProxyCommand, not both."))
        if not isinstance(self.ssh_interactive, bool):
            raise RemoteExecutionError(tr("SSH interactive authentication must be a boolean."))
        if not isinstance(self.ssh_keychain, bool):
            raise RemoteExecutionError(tr("SSH keychain storage must be a boolean."))
        _validate_connection_fields(self)
        if self.backend in {"ssh", *_SCHEDULERS}:
            if self.host:
                host = _safe_text(self.host, "SSH host")
                if host.startswith("-") or not _HOST_RE.fullmatch(host):
                    raise RemoteExecutionError(
                        "SSH host contains unsupported characters."
                    )
            elif self.backend == "ssh":
                raise RemoteExecutionError("An SSH workstation needs a host.")
            _path_text(self.workdir, "Remote work directory")
            _safe_text(self.runner, "spaCR runner")
            if bool(self.local_root) != bool(self.remote_root):
                raise RemoteExecutionError(
                    "Set both local and remote dataset roots, or leave both "
                    "blank when their absolute paths are identical."
                )
            if self.local_root:
                local = os.path.expanduser(self.local_root)
                if not os.path.isabs(local):
                    raise RemoteExecutionError(
                        "Local dataset root must be absolute."
                    )
                _path_text(self.remote_root, "Remote dataset root")
            if self.scheduler_options:
                _split_template(
                    self.scheduler_options, "Slurm options",
                    require_program=False, posix=True,
                )
            _validate_script_fields(self)

        if self.backend == "command":
            for label, command in (
                ("Submit command", self.submit_command),
                ("Status command", self.status_command),
                ("Cancel command", self.cancel_command),
            ):
                _split_template(_safe_text(command, label), label)
            if "{settings}" not in self.submit_command:
                raise RemoteExecutionError(
                    "Cloud/custom submit command must contain {settings}."
                )
            if "{external_id}" not in self.status_command:
                raise RemoteExecutionError(
                    "Cloud/custom status command must contain {external_id}."
                )
            if "{external_id}" not in self.cancel_command:
                raise RemoteExecutionError(
                    "Cloud/custom cancel command must contain {external_id}."
                )
            if self.log_command:
                _split_template(self.log_command, "Log command")
            if self.job_id_pattern:
                try:
                    re.compile(self.job_id_pattern)
                except re.error as exc:
                    raise RemoteExecutionError(
                        f"Job-ID regular expression is invalid: {exc}"
                    ) from exc
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionProfile":
        """Construct and validate a profile from JSON-compatible data.

        :param value: stored profile fields keyed by their dataclass names.
        """
        fields = cls.__dataclass_fields__
        profile = cls(**{key: value[key] for key in fields if key in value})
        return profile.validate()


@dataclass
class RemoteJob:
    """Persistent local record of one submitted job.

    :param job_id: locally generated stable identifier for the job.
    :param module: resolved spaCR command-line module key to execute.
    :param profile_name: name of the execution profile used for submission.
    :param backend: backend recorded at submission time.
    :param status: compact lifecycle state, initially ``"submitting"``.
    :param external_id: validated process, scheduler, or provider identifier.
    :param created_utc: ISO-8601 construction timestamp.
    :param updated_utc: timestamp replaced whenever :class:`JobStore` persists
        the record.
    :param settings_path: retained local path of the mapped settings JSON.
    :param settings_sha256: SHA-256 digest of the exact serialized settings.
    :param remote_settings_path: uploaded settings path for SSH and Slurm.
    :param remote_job_dir: target-side directory containing job artifacts.
    :param log_reference: backend log location or custom-backend marker.
    :param log_tail: most recently fetched output or availability message.
    :param exit_code: normalized process exit status, or ``None`` until known.
    :param error: latest submission, polling, or cancellation failure; cleared
        after a successful refresh or cancellation.
    :param profile: submission-time profile snapshot, retained so the job
        remains operable after profile edits or deletion.
    :param job_script: the rendered job script exactly as it was sent.
    :param allocation_id: the scheduler allocation the job runs inside, or
        blank for a batch job.
    """

    job_id: str
    module: str
    profile_name: str
    backend: str
    status: str = "submitting"
    external_id: str = ""
    created_utc: str = field(default_factory=_utc_now)
    updated_utc: str = field(default_factory=_utc_now)
    settings_path: str = ""
    settings_sha256: str = ""
    remote_settings_path: str = ""
    remote_job_dir: str = ""
    log_reference: str = ""
    log_tail: str = ""
    exit_code: Optional[int] = None
    error: str = ""
    profile: Dict[str, Any] = field(default_factory=dict)
    job_script: str = ""
    allocation_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RemoteJob":
        """Construct a job from a stored mapping, tolerating future fields.

        :param value: stored job fields keyed by their dataclass names.
        """
        fields = cls.__dataclass_fields__
        job = cls(**{key: value[key] for key in fields if key in value})
        if job.status not in _ALL_STATES:
            job.status = "unknown"
        return job


@contextlib.contextmanager
def _file_lock(path: Path):
    """Take a small cross-process advisory lock where the platform supports it."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+", encoding="utf-8")
    except OSError as exc:
        raise RemoteExecutionError(
            f"Could not open state lock {path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    try:
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        handle.close()


def _read_json(path: Path, default: Any) -> Any:
    """Read a JSON store, returning a default when it does not exist.

    :param path: the file.
    :param default: what to return when the file is absent -- absent is a
        first run, not a fault.
    :returns: the parsed value.
    :raises RemoteExecutionError: if the file exists but cannot be read or
        parsed, naming the file. A corrupt store is not an empty one, and
        silently returning the default would lose every recorded job.
    """
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RemoteExecutionError(
            f"Could not read {path}: {type(exc).__name__}: {exc}"
        ) from exc


def _write_json_atomic(path: Path, value: Any) -> None:
    """Write a JSON store atomically.

    Written to a temporary file in the same directory, flushed and fsynced,
    then renamed over the target -- so a crash mid-write cannot leave a
    truncated store that parses as a shorter job list. The temporary file is
    removed on any path out. It is opened once with ``O_EXCL`` rather than
    through ``tempfile.mkstemp``, which on Windows takes a refused folder for
    a name collision and retries ten thousand times before it fails.

    :param path: the file to write.
    :param value: the value to serialise.
    :raises RemoteExecutionError: if the write fails, naming the file.
    """
    temporary = ""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            value, indent=2, sort_keys=True, default=str
        ) + "\n"
        candidate = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
        flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
                 | getattr(os, "O_NOINHERIT", 0) | getattr(os, "O_CLOEXEC", 0))
        fd = os.open(candidate, flags, 0o600)
        temporary = str(candidate)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise RemoteExecutionError(
            f"Could not write {path}: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if temporary and os.path.exists(temporary):
            try:
                os.unlink(temporary)
            except OSError:
                pass


class ProfileStore:
    """Atomic persistent store for execution profiles.

    :param path: optional profiles JSON path; by default
        ``state_directory()/profiles.json``. The advisory lock uses a sibling
        path with ``.lock`` appended.
    """

    def __init__(self, path: Optional[os.PathLike] = None):
        """Open the execution-profile store.

        :param path: where the profiles live; ``None`` uses the state directory.
            A sibling lock file is derived from it, so two processes writing
            profiles cannot interleave.
        """
        self.path = Path(path) if path is not None else (
            state_directory() / "profiles.json"
        )
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def list(self) -> List[ExecutionProfile]:
        """Return profiles sorted by case-insensitive name."""
        with _LOCK, _file_lock(self.lock_path):
            raw = _read_json(self.path, {"profiles": []})
        profiles = [
            ExecutionProfile.from_dict(row)
            for row in raw.get("profiles", [])
        ]
        return sorted(profiles, key=lambda profile: profile.name.casefold())

    def get(self, name: str) -> ExecutionProfile:
        """Return a named profile or raise a user-facing error.

        :param name: profile name to match case-insensitively.
        """
        wanted = str(name).casefold()
        for profile in self.list():
            if profile.name.casefold() == wanted:
                return profile
        raise RemoteExecutionError(
            f"Execution profile {name!r} does not exist."
        )

    def save(self, profile: ExecutionProfile) -> None:
        """Insert or replace one profile atomically.

        :param profile: validated execution profile to persist.
        """
        profile.validate()
        with _LOCK, _file_lock(self.lock_path):
            raw = _read_json(self.path, {"profiles": []})
            rows = list(raw.get("profiles", []))
            rows = [
                row for row in rows
                if str(row.get("name", "")).casefold()
                != profile.name.casefold()
            ]
            rows.append(profile.to_dict())
            _write_json_atomic(self.path, {"version": 1, "profiles": rows})

    def delete(self, name: str) -> bool:
        """Delete one profile; return whether it existed.

        :param name: profile name to delete case-insensitively.
        """
        wanted = str(name).casefold()
        with _LOCK, _file_lock(self.lock_path):
            raw = _read_json(self.path, {"profiles": []})
            rows = list(raw.get("profiles", []))
            kept = [
                row for row in rows
                if str(row.get("name", "")).casefold() != wanted
            ]
            changed = len(kept) != len(rows)
            if changed:
                _write_json_atomic(
                    self.path, {"version": 1, "profiles": kept}
                )
        return changed


class JobStore:
    """Atomic persistent store for remote job metadata.

    :param path: optional jobs JSON path; by default
        ``state_directory()/jobs.json``. The advisory lock uses a sibling path
        with ``.lock`` appended.
    """

    def __init__(self, path: Optional[os.PathLike] = None):
        """Open the remote-job store.

        :param path: where the jobs live; ``None`` uses the state directory. A
            sibling lock file is derived from it, so two processes writing jobs
            cannot interleave.
        """
        self.path = Path(path) if path is not None else (
            state_directory() / "jobs.json"
        )
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def list(self) -> List[RemoteJob]:
        """Return newest jobs first."""
        with _LOCK, _file_lock(self.lock_path):
            raw = _read_json(self.path, {"jobs": []})
        jobs = [RemoteJob.from_dict(row) for row in raw.get("jobs", [])]
        return sorted(jobs, key=lambda job: job.created_utc, reverse=True)

    def get(self, job_id: str) -> RemoteJob:
        """Return a job by full ID or unambiguous prefix.

        :param job_id: complete job identifier or an unambiguous prefix.
        """
        wanted = str(job_id).strip()
        matches = [
            job for job in self.list() if job.job_id.startswith(wanted)
        ]
        if not matches:
            raise RemoteExecutionError(f"Remote job {job_id!r} was not found.")
        if len(matches) > 1:
            raise RemoteExecutionError(
                f"Remote job prefix {job_id!r} is ambiguous."
            )
        return matches[0]

    def save(self, job: RemoteJob) -> None:
        """Insert or replace one job atomically.

        :param job: remote-job record to persist.
        """
        job.updated_utc = _utc_now()
        with _LOCK, _file_lock(self.lock_path):
            raw = _read_json(self.path, {"jobs": []})
            rows = [
                row for row in raw.get("jobs", [])
                if row.get("job_id") != job.job_id
            ]
            rows.append(job.to_dict())
            _write_json_atomic(self.path, {"version": 1, "jobs": rows})


def _map_path_string(value: str, local_root: str, remote_root: str) -> str:
    """Rewrite a local path into its place under the remote root.

    A path outside the local root is returned unchanged: it is not part of
    the mapped tree, and guessing a remote location for it would send the
    job somewhere nobody asked for. Paths on different drives raise inside
    ``commonpath`` and take the same route.

    :param value: the local path.
    :param local_root: the local tree's root; empty disables mapping.
    :param remote_root: the corresponding root on the remote.
    :returns: the mapped path, or the original.
    """
    if not local_root:
        return value
    expanded = os.path.abspath(os.path.expanduser(value))
    root = os.path.abspath(os.path.expanduser(local_root))
    try:
        common = os.path.commonpath((expanded, root))
    except ValueError:
        return value
    if common != root:
        return value
    relative = os.path.relpath(expanded, root)
    if relative == ".":
        return remote_root
    return posixpath.join(remote_root, *Path(relative).parts)


def map_settings_paths(
    value: Any,
    local_root: str,
    remote_root: str,
) -> Any:
    """Recursively map absolute paths below one local root to a remote root.

    Non-path strings and paths outside the configured root are unchanged.
    Mapping keys are intentionally preserved: setting names are not paths.

    :param value: nested settings value whose absolute path strings are mapped.
    :param local_root: local dataset root that mapped paths must lie below.
    :param remote_root: remote dataset root that replaces ``local_root``.
    """
    if isinstance(value, dict):
        return {
            key: map_settings_paths(item, local_root, remote_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            map_settings_paths(item, local_root, remote_root)
            for item in value
        ]
    if isinstance(value, tuple):
        return [
            map_settings_paths(item, local_root, remote_root)
            for item in value
        ]
    if isinstance(value, str) and os.path.isabs(os.path.expanduser(value)):
        return _map_path_string(value, local_root, remote_root)
    return value


def _split_template(
    command: str,
    label: str,
    *,
    require_program: bool = True,
    posix: Optional[bool] = None,
) -> List[str]:
    """Split a command template into argv, the way a shell would.

    Windows paths keep their backslashes, so a local template is split
    without POSIX escapes there; quotes around a token such as
    ``"C:\\Program Files\\gcloud\\gcloud.cmd"`` are then removed, as a
    shell would.

    :param command: the template.
    :param label: its name, used in the errors.
    :param require_program: reject a template whose first token is an option
        rather than a program.
    :param posix: POSIX rules; by default only off Windows. Options read by
        a remote POSIX program pass ``True``.
    :returns: the argument vector.
    :raises RemoteExecutionError: if the template cannot be parsed, is
        empty, or begins with an option.
    """
    if posix is None:
        posix = os.name != "nt"
    try:
        argv = shlex.split(command, posix=posix)
    except ValueError as exc:
        raise RemoteExecutionError(f"{label} cannot be parsed: {exc}") from exc
    if not posix:
        argv = [token[1:-1] if len(token) > 1 and token[0] == token[-1]
                and token[0] in "\"'" else token for token in argv]
    if not argv:
        raise RemoteExecutionError(f"{label} is empty.")
    if require_program and argv[0].startswith("-"):
        raise RemoteExecutionError(f"{label} must begin with a program name.")
    return argv


def _render_template(
    command: str,
    context: Mapping[str, str],
    label: str,
) -> List[str]:
    """Split a command template and substitute its placeholders.

    Substituted PER TOKEN, after splitting: a value containing a space
    therefore stays one argument rather than becoming two, which is the
    whole reason the template is split first.

    :param command: the template.
    :param context: the placeholder values.
    :param label: the template's name, used in the errors.
    :returns: the rendered argument vector.
    :raises RemoteExecutionError: naming the placeholder, if the template
        uses one the context does not have or its syntax is invalid.
    """
    argv = _split_template(command, label)
    rendered: List[str] = []
    for token in argv:
        try:
            rendered.append(token.format_map(context))
        except KeyError as exc:
            raise RemoteExecutionError(
                f"{label} uses unknown placeholder {{{exc.args[0]}}}."
            ) from exc
        except ValueError as exc:
            raise RemoteExecutionError(
                f"{label} has invalid placeholder syntax: {exc}"
            ) from exc
    return rendered


def _auth_mode(profile) -> str:
    """Return the sign-in method a profile uses.

    :param profile: the execution profile.
    :returns: one of :data:`_AUTH_MODES`. A blank ``auth_mode`` follows the
        older ``ssh_interactive`` switch, so profiles saved before sign-in
        methods existed keep working unchanged.
    """
    if profile.auth_mode:
        return profile.auth_mode
    return "interactive" if profile.ssh_interactive else "openssh"


def _login_patterns(text) -> List[str]:
    """Split and validate a profile's login-node host patterns.

    :param text: comma- or space-separated shell-style host patterns.
    :returns: the patterns.
    :raises RemoteExecutionError: if a pattern holds characters that a shell
        ``case`` statement would treat as code.
    """
    patterns = [item for item in re.split(r"[\s,]+", str(text or "")) if item]
    for pattern in patterns:
        if not _LOGIN_PATTERN_RE.fullmatch(pattern):
            raise RemoteExecutionError(tr("Login host pattern {pattern} may contain only letters, numbers, '.', '-', '_' and the wildcards * ? [ ].").format(pattern=pattern))
    return patterns


def _validate_resources(resources) -> None:
    """Check scheduler resources before they reach a job script.

    :param resources: the profile's resource mapping.
    :raises RemoteExecutionError: naming the first unknown key or malformed
        value. Values are written into scheduler directives, so anything a
        scheduler or shell could read as more than one value is refused.
    """
    if not isinstance(resources, dict):
        raise RemoteExecutionError(tr("Resources must be a mapping of names to values."))
    for key, value in resources.items():
        if key not in _RESOURCE_KEYS or not isinstance(value, str) or "\x00" in value:
            raise RemoteExecutionError(tr("Resources may contain only {keys}.").format(keys=", ".join(_RESOURCE_KEYS)))
        if key == "extra":
            if len(value) > 16384:
                raise RemoteExecutionError(tr("Extra scheduler lines are too long."))
            continue
        if value.strip() and not re.fullmatch(_RESOURCE_PATTERNS[key], value.strip()):
            raise RemoteExecutionError(tr("Resource {key} has an invalid value: {value}").format(key=key, value=value))


def _validate_connection_fields(profile) -> None:
    """Validate the sign-in method, custom SSH command and safety switches.

    :param profile: the execution profile.
    :raises RemoteExecutionError: on an unknown method, a custom command that
        cannot be parsed or that carries a password, or a malformed field.
    """
    if profile.auth_mode not in ("", *_AUTH_MODES):
        raise RemoteExecutionError(tr("Unknown sign-in method {mode}; choose openssh, interactive, gssapi or custom.").format(mode=profile.auth_mode))
    if not isinstance(profile.direct_run, bool):
        raise RemoteExecutionError(tr("Run directly over ssh must be a boolean."))
    if _auth_mode(profile) == "custom":
        command = _safe_text(profile.ssh_command, tr("Custom SSH command"))
        argv = _split_template(command, tr("Custom SSH command"), posix=os.name != "nt")
        if (os.path.basename(argv[0]).casefold().startswith("sshpass")
                or any(re.match(r"(?i)(?:--?password(?:=|$)|password=)", token)
                       for token in argv)):
            raise RemoteExecutionError(tr("A custom SSH command may not contain a password. Use sign-in in spaCR, which can keep the password in the system keychain."))
    _login_patterns(profile.login_hosts)
    _validate_resources(profile.resources)
    for label, text in ((tr("Environment setup"), profile.env_setup), (tr("Job script"), profile.job_script)):
        if not isinstance(text, str) or "\x00" in text or len(text) > 65536:
            raise RemoteExecutionError(tr("{label} must be text shorter than 64 KB.").format(label=label))


def _validate_script_fields(profile) -> None:
    """Refuse a job-script template that names a placeholder spaCR lacks.

    :param profile: the execution profile.
    :raises RemoteExecutionError: naming the unknown placeholder, so a typo
        such as ``{setings_path}`` fails when the profile is saved rather than
        on the cluster.
    """
    for name in _PLACEHOLDER_RE.findall(profile.job_script or ""):
        if name not in _JOB_SCRIPT_PLACEHOLDERS:
            raise RemoteExecutionError(tr("Job script uses an unknown placeholder: {name}. Known placeholders: {known}.").format(
                name="{" + name + "}", known=", ".join("{" + key + "}" for key in _JOB_SCRIPT_PLACEHOLDERS)))


def _placeholder_value(context, label, match) -> str:
    """Return the replacement for one job-script placeholder.

    :param context: placeholder values.
    :param label: the template's name, used in the error.
    :param match: the regular-expression match of ``{name}``.
    :returns: the value.
    :raises RemoteExecutionError: if the name is unknown.
    """
    name = match.group(1)
    if name not in context:
        raise RemoteExecutionError(tr("Job script uses an unknown placeholder: {name}. Known placeholders: {known}.").format(
            name="{" + name + "}", known=", ".join("{" + key + "}" for key in context)))
    return context[name]


def _fill_script(template: str, context: Mapping[str, str], label: str) -> str:
    """Substitute placeholders in one pass, leaving shell ``${VAR}`` alone.

    :param template: the job-script template.
    :param context: placeholder values.
    :param label: the template's name, used in errors.
    :returns: the rendered text. Substituted values are never rescanned.
    """
    return _PLACEHOLDER_RE.sub(functools.partial(_placeholder_value, context, label), template)


def _minutes(value: str) -> int:
    """Convert a Slurm-style time limit to whole minutes.

    :param value: ``MM``, ``MM:SS``, ``HH:MM:SS`` or ``D-HH[:MM[:SS]]``.
    :returns: minutes, rounding seconds up; 0 when blank.
    """
    text = str(value or "").strip()
    if not text:
        return 0
    days = 0
    if "-" in text:
        head, text = text.split("-", 1)
        days = int(head)
        parts = [int(part) for part in text.split(":")] + [0, 0]
        hours, minutes, seconds = parts[0], parts[1], parts[2]
    else:
        parts = [int(part) for part in text.split(":")]
        if len(parts) == 1:
            hours, minutes, seconds = 0, parts[0], 0
        elif len(parts) == 2:
            hours, minutes, seconds = 0, parts[0], parts[1]
        else:
            hours, minutes, seconds = parts[0], parts[1], parts[2]
    return days * 1440 + hours * 60 + minutes + (1 if seconds else 0)


def _mem_mb(value: str) -> int:
    """Convert a memory request such as ``90G`` to megabytes.

    :param value: a number with an optional K, M, G or T unit; no unit is MB,
        as in Slurm.
    :returns: megabytes; 0 when blank or unreadable.
    """
    match = re.fullmatch(r"([0-9.]+)\s*([KMGTkmgt])?[Bb]?", str(value or "").strip())
    if not match:
        return 0
    scale = {"k": 1 / 1024, "m": 1, "g": 1024, "t": 1024 * 1024}[(match.group(2) or "m").lower()]
    return int(float(match.group(1)) * scale)


def _scheduler_output_path(backend: str, path: str) -> str:
    """Quote one output path using the scheduler's directive token grammar.

    PBS copies quoted contents literally, unlike Slurm's backslash-aware
    scanner. LSF accepts double-quoted option arguments. Plain paths retain
    their original representation.
    """
    if not any(character.isspace() or character in "\"'\\#" for character in path):
        return path
    if backend == "pbs":
        return shlex.quote(path)
    return '"' + path.replace('\\', '\\\\').replace('"', '\\"') + '"'


def _scheduler_directives(profile, job_name: str, log_path: str) -> str:
    """Render the profile's resources as scheduler directive lines.

    :param profile: the execution profile; its backend picks ``#SBATCH``,
        ``#PBS`` or ``#BSUB``.
    :param job_name: the scheduler job name.
    :param log_path: where the scheduler writes the job's output.
    :returns: the directive block, or an empty string for a backend without
        a scheduler.
    """
    resources = {key: str(value).strip() for key, value in profile.resources.items()
                 if key != "extra" and str(value).strip()}
    gpus = resources.get("gpus", "0") or "0"
    output_path = _scheduler_output_path(profile.backend, log_path)
    lines = []
    if profile.backend == "slurm":
        prefix = "#SBATCH "
        lines += [f"--job-name={job_name}", f"--output={output_path}"]
        for key, flag in (("partition", "--partition"), ("account", "--account"), ("qos", "--qos"),
                          ("nodes", "--nodes"), ("cpus", "--cpus-per-task"), ("mem", "--mem"),
                          ("time", "--time"), ("mail_type", "--mail-type"), ("mail_user", "--mail-user")):
            if key in resources:
                lines.append(f"{flag}={resources[key]}")
        if int(gpus):
            gpu_type = resources.get("gpu_type")
            lines.append("--gpus=" + (f"{gpu_type}:" if gpu_type else "") + gpus)
    elif profile.backend == "pbs":
        prefix = "#PBS "
        lines += [f"-N {job_name[:15]}", f"-o {output_path}", "-j oe"]
        if "partition" in resources:
            lines.append(f"-q {resources['partition']}")
        if "account" in resources:
            lines.append(f"-A {resources['account']}")
        select = f"select={resources.get('nodes', '1')}:ncpus={resources.get('cpus', '1')}"
        if int(gpus):
            select += f":ngpus={gpus}"
        if "mem" in resources:
            select += f":mem={_mem_mb(resources['mem'])}mb"
        lines.append(f"-l {select}")
        if "time" in resources:
            total = _minutes(resources["time"])
            lines.append(f"-l walltime={total // 60}:{total % 60:02d}:00")
        mail = resources.get("mail_type", "").upper()
        letters = "".join(letter for word, letter in (("BEGIN", "b"), ("END", "e"), ("FAIL", "a")) if word in mail or "ALL" in mail)
        if letters:
            lines.append(f"-m {letters}")
        if "mail_user" in resources:
            lines.append(f"-M {resources['mail_user']}")
    elif profile.backend == "lsf":
        prefix = "#BSUB "
        lines += [f"-J {job_name}", f"-o {output_path}"]
        if "partition" in resources:
            lines.append(f"-q {resources['partition']}")
        if "account" in resources:
            lines.append(f"-P {resources['account']}")
        if "cpus" in resources:
            lines += [f"-n {resources['cpus']}", '-R "span[hosts=1]"']
        if int(gpus):
            gpu_type = resources.get("gpu_type")
            lines.append(f'-gpu "num={gpus}' + (f":gmodel={gpu_type}" if gpu_type else "") + '"')
        if "mem" in resources:
            lines.append(f"-M {_mem_mb(resources['mem'])}MB")
        if "time" in resources:
            total = _minutes(resources["time"])
            lines.append(f"-W {total // 60}:{total % 60:02d}")
        mail = resources.get("mail_type", "").upper()
        if "BEGIN" in mail or "ALL" in mail:
            lines.append("-B")
        if "END" in mail or "FAIL" in mail or "ALL" in mail:
            lines.append("-N")
        if "mail_user" in resources:
            lines.append(f"-u {resources['mail_user']}")
    else:
        return ""
    block = [prefix + line for line in lines]
    for extra in str(profile.resources.get("extra", "")).splitlines():
        extra = extra.strip()
        if extra:
            block.append(extra if extra.startswith("#") else prefix + extra)
    return "\n".join(block)


def _runner_argv(profile) -> List[str]:
    """Split the profile's spaCR runner into words.

    :param profile: the execution profile.
    :returns: the runner's argument vector, such as ``["spacr-run"]``.
    """
    return _split_template(profile.runner, "spaCR runner", posix=True)


def _script_context(profile, job, job_dir: str, settings_path: str, log_path: str) -> Dict[str, str]:
    """Build every value a job-script template may name.

    Value placeholders are shell-quoted when they need it; code placeholders
    (environment setup, directives, the run line and the runner check) are
    inserted as written, because they ARE shell code.

    :param profile: the execution profile.
    :param job: the job being submitted.
    :param job_dir: the remote job directory.
    :param settings_path: the uploaded settings file.
    :param log_path: the job's log file.
    :returns: placeholder name to text.
    """
    runner = _runner_argv(profile)
    job_name = f"spacr-{job.module}-{job.job_id[:8]}"
    resources = profile.resources
    values = {
        "settings_path": settings_path, "job_dir": job_dir, "module": job.module,
        "job_name": job_name, "log_dir": posixpath.dirname(log_path), "log_path": log_path,
        "workdir": profile.workdir, "job_id": job.job_id,
        **{key: str(resources.get(key, "")).strip() for key in ("gpus", "cpus", "mem", "time", "partition", "account", "qos")},
    }
    context = {key: shlex.quote(value) if value else "" for key, value in values.items()}
    message = (f"spaCR: {runner[0]} was not found on PATH (exit 127). Add environment setup lines "
               "that provide it to the job script, for example: module load mamba; mamba activate spacr, "
               "or conda activate spacr, or pip install spacr.")
    context.update({
        "runner": shlex.join(runner),
        "run_command": shlex.join([*runner, job.module, "--settings", settings_path]),
        "env_setup": str(profile.env_setup or "").strip(),
        "scheduler_directives": _scheduler_directives(profile, job_name, log_path),
        "runner_check": f"command -v {shlex.quote(runner[0])} >/dev/null 2>&1 || "
                        f"{{ echo {shlex.quote(message)} >&2; exit 127; }}",
    })
    return context


def _login_guard(profile) -> str:
    """Shell lines that refuse to start outside a scheduler job on a login node.

    The check runs ON the remote host: a host name matching a login pattern,
    or any installed batch scheduler, with no scheduler job id in the
    environment, means the command would run on a shared login node.

    :param profile: the execution profile; its ``login_hosts`` add patterns.
    :returns: the guard, ending in a newline.
    """
    patterns = "|".join(dict.fromkeys([*_DEFAULT_LOGIN_PATTERNS, *_login_patterns(profile.login_hosts)]))
    return (
        "spacr_host=$(hostname 2>/dev/null || uname -n)\n"
        'if [ -z "${SLURM_JOB_ID:-}${PBS_JOBID:-}${LSB_JOBID:-}" ]; then\n'
        f'  case "$spacr_host" in {patterns}) echo "spaCR refused to start on $spacr_host: it is a cluster '
        'login node. Submit through the scheduler or start an allocation." >&2; '
        f"exit {_LOGIN_REFUSED_EXIT};; esac\n"
        "  for spacr_scheduler in sbatch qsub bsub; do\n"
        '    if command -v "$spacr_scheduler" >/dev/null 2>&1; then echo "spaCR refused to start on '
        '$spacr_host: $spacr_scheduler is installed, so this is a cluster node outside a job. Submit through '
        'the scheduler or start an allocation." >&2; '
        f"exit {_LOGIN_REFUSED_EXIT}; fi\n"
        "  done\n"
        "fi\n"
    )


def _with_login_guard(script: str, profile) -> str:
    """Insert the login-node guard after the shebang and directive lines.

    :param script: the rendered job script.
    :param profile: the execution profile; ``direct_run`` leaves the script
        unguarded, as the user explicitly chose.
    :returns: the guarded script.
    """
    if profile.direct_run:
        return script
    lines = script.splitlines()
    index = 0
    while index < len(lines) and (not lines[index].strip() or lines[index].lstrip().startswith("#")):
        index += 1
    return "\n".join([*lines[:index], _login_guard(profile).rstrip("\n"), *lines[index:]]) + "\n"


def _job_script_text(profile, job, job_dir: str, settings_path: str, log_path: str) -> str:
    """Render the job script the remote host will run.

    :param profile: the execution profile; a blank ``job_script`` uses the
        backend's default from :data:`_SCRIPT_PRESETS`.
    :param job: the job.
    :param job_dir: the remote job directory.
    :param settings_path: the uploaded settings file.
    :param log_path: the job's log file.
    :returns: the script, guarded against login nodes unless the profile
        opted into running directly.
    """
    template = profile.job_script or _SCRIPT_PRESETS.get(profile.backend, _SCRIPT_PRESETS["ssh"])
    text = _fill_script(template, _script_context(profile, job, job_dir, settings_path, log_path), tr("Job script"))
    return _with_login_guard(text, profile)


def _preview_job_script(profile, module: str) -> str:
    """Render the script a submission would send, for review before sending.

    :param profile: the execution profile.
    :param module: the spaCR module that would run.
    :returns: the script, with ``preview`` standing in for the job id.
    """
    job = RemoteJob("preview", module, profile.name, profile.backend)
    job_dir, settings_path, log_path = _remote_paths(profile, job)
    return _job_script_text(profile, job, job_dir, settings_path, log_path)


def _interpreter(script: str) -> List[str]:
    """Return the interpreter the script's ``#!`` line names.

    Running the interpreter on the file, rather than the file itself, works on
    file systems mounted without execute permission.

    :param script: the job script.
    :returns: the interpreter's argument vector; ``["/bin/sh"]`` without one.
    """
    first = script.split("\n", 1)[0]
    if first.startswith("#!"):
        words = first[2:].split()
        if words:
            return words
    return ["/bin/sh"]


def _login_refusal(host: str) -> str:
    """Explain why spaCR will not run compute on a login node.

    :param host: the host name.
    :returns: the message.
    """
    return tr("spaCR will not run compute on {host}: it is a cluster login node, and running jobs there can get an account banned. Submit through the scheduler (a Slurm, PBS or LSF profile) or start an allocation and run inside it. If {host} is a private workstation with no scheduler, tick Run directly over ssh in the profile.").format(host=host)


def _refuse_login_node(profile) -> None:
    """Refuse a direct ssh run whose target matches a login-node pattern.

    :param profile: the execution profile.
    :raises RemoteExecutionError: before anything is sent, when an ``ssh``
        profile without the ``direct_run`` opt-in names a login host.
    """
    if profile.backend != "ssh" or profile.direct_run:
        return
    patterns = [*_DEFAULT_LOGIN_PATTERNS, *_login_patterns(profile.login_hosts)]
    for name in (profile.host.rsplit("@", 1)[-1], profile.ssh_options.get("hostname", "")):
        if name and any(fnmatch.fnmatchcase(name.casefold(), pattern.casefold()) for pattern in patterns):
            raise RemoteExecutionError(_login_refusal(name))


def _exit_hint(code, profile) -> str:
    """Explain an exit status the user can fix, or return an empty string.

    :param code: the job's exit status.
    :param profile: the execution profile.
    :returns: the explanation.
    """
    if code == 127:
        return tr("Exit code 127 means a command was not found: the job's environment did not provide {runner}. Add environment setup lines to the job script, for example module load mamba and mamba activate spacr, conda activate spacr, or pip install spacr.").format(runner=profile.runner)
    if code == _LOGIN_REFUSED_EXIT:
        return tr("spaCR refused to start because the job was on a cluster login node, outside a scheduler job. Submit through the scheduler or start an allocation and run inside it.")
    return ""


_CLUSTER_PROBE = (
    "echo '##host'; hostname 2>/dev/null\n"
    "echo '##user'; id -un 2>/dev/null\n"
    "echo '##cpus'; nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null\n"
    "echo '##mem_mb'; free -m 2>/dev/null | awk '/^Mem:/{print $2}'\n"
    "echo '##gpu'; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null\n"
    "echo '##sinfo'; command -v sinfo >/dev/null 2>&1 && sinfo -h -o '%P|%a|%l|%D|%c|%m|%G' 2>/dev/null\n"
    "echo '##assoc'; command -v sacctmgr >/dev/null 2>&1 && "
    "sacctmgr -n -P show assoc user=\"$(id -un)\" format=Account,Partition,QOS 2>/dev/null\n"
    "echo '##pbs'; command -v pbsnodes >/dev/null 2>&1 && qstat -Q 2>/dev/null\n"
    "echo '##lsf'; command -v bqueues >/dev/null 2>&1 && bqueues -w 2>/dev/null\n"
    "echo '##end'\n"
)


def _parse_gres(text: str) -> tuple:
    """Read the GPU count and models from a Slurm GRES string.

    :param text: such as ``gpu:v100:2(S:0-1),gpu:a100:4`` or ``(null)``.
    :returns: ``(largest GPU count per node, sorted GPU models)``.
    """
    count, models = 0, set()
    for piece in str(text or "").split(","):
        piece = re.sub(r"\(.*?\)", "", piece).strip()
        parts = piece.split(":")
        if not parts or parts[0] != "gpu":
            continue
        number = parts[-1] if parts[-1].isdigit() else "1"
        if len(parts) == 3 or (len(parts) == 2 and not parts[1].isdigit()):
            models.add(parts[1])
        count = max(count, int(number))
    return count, sorted(models)


def _parse_cluster_probe(text: str) -> Dict[str, Any]:
    """Parse the read-only cluster probe into what the System panel shows.

    :param text: the probe's output, in ``##section`` blocks.
    :returns: host, user, CPUs, memory and GPUs of the host reached; the
        Slurm partitions with nodes, CPUs, memory, GPUs and time limit per
        partition; the user's accounts with partitions and QoS; PBS or LSF
        queues; and which scheduler was found.
    """
    sections: Dict[str, List[str]] = {}
    current = ""
    for line in str(text or "").splitlines():
        if line.startswith("##"):
            current = line[2:].strip()
            sections.setdefault(current, [])
        elif current and line.strip():
            sections[current].append(line.strip())
    partitions = []
    for row in sections.get("sinfo", []):
        fields = row.split("|")
        if len(fields) < 7:
            continue
        name, available, limit, nodes, cpus, memory, gres = fields[:7]
        gpus, models = _parse_gres(gres)
        partitions.append({
            "partition": name.rstrip("*"), "default": name.endswith("*"), "available": available,
            "time_limit": limit, "nodes": int(nodes) if nodes.isdigit() else 0,
            "cpus": int(cpus.rstrip("+")) if cpus.rstrip("+").isdigit() else 0,
            "mem_mb": int(memory.rstrip("+")) if memory.rstrip("+").isdigit() else 0,
            "gpus": gpus, "gpu_types": models,
        })
    accounts = []
    for row in sections.get("assoc", []):
        fields = (row.split("|") + ["", "", ""])[:3]
        if fields[0]:
            accounts.append({"account": fields[0], "partition": fields[1],
                             "qos": [item for item in fields[2].split(",") if item]})
    queues = {}
    for key, header in (("pbs", "Queue"), ("lsf", "QUEUE_NAME")):
        rows = [row.split()[0] for row in sections.get(key, [])
                if row.split() and not row.startswith(header) and not set(row) <= set("- ")]
        queues[key] = rows
    host = (sections.get("host") or [""])[0]
    gpus = [row for row in sections.get("gpu", [])]
    scheduler = "slurm" if partitions else "pbs" if queues["pbs"] else "lsf" if queues["lsf"] else ""
    first = (sections.get("cpus") or [""])[0]
    memory = (sections.get("mem_mb") or [""])[0]
    return {"host": host, "user": (sections.get("user") or [""])[0],
            "cpus": int(first) if first.isdigit() else 0,
            "mem_mb": int(memory) if memory.isdigit() else 0, "gpus": gpus,
            "partitions": partitions, "accounts": accounts,
            "pbs_queues": queues["pbs"], "lsf_queues": queues["lsf"], "scheduler": scheduler}


def _probe_cluster(profile, runner: CommandRunner = _run_command) -> Dict[str, Any]:
    """Ask the cluster what the user can reach, using read-only queries only.

    :param profile: the execution profile.
    :param runner: how the command is run.
    :returns: the parsed probe; see :func:`_parse_cluster_probe`.
    :raises RemoteExecutionError: if the probe cannot run.
    """
    result = runner(_remote_script(profile, _CLUSTER_PROBE), timeout=120.0)
    _require_ok(result, "Cluster resource query")
    return _parse_cluster_probe(result.stdout)


def _resource_problems(resources: Mapping[str, str], cluster: Mapping[str, Any]) -> List[str]:
    """Compare requested resources with what the cluster reported.

    :param resources: the profile's resources.
    :param cluster: a parsed probe from :func:`_parse_cluster_probe`.
    :returns: one message per request the cluster cannot meet; empty when
        everything fits or nothing is known.
    """
    problems = []
    partitions = list(cluster.get("partitions") or [])
    names = [row["partition"] for row in partitions]
    wanted = [item for item in str(resources.get("partition", "")).split(",") if item]
    unknown = [name for name in wanted if names and name not in names]
    if unknown:
        problems.append(tr("Partition {partition} is not on this cluster; it has {names}.").format(
            partition=", ".join(unknown), names=", ".join(dict.fromkeys(names))))
    rows = ([row for row in partitions if row["partition"] in wanted] if wanted
            else [row for row in partitions if row.get("default")])
    label = ", ".join(wanted) or ", ".join(dict.fromkeys(row["partition"] for row in rows))
    if rows:
        checks = (
            ("cpus", int(resources.get("cpus") or 0), max(row["cpus"] for row in rows),
             tr("{count} CPUs per task were requested, but nodes in {partition} have at most {limit}.")),
            ("gpus", int(resources.get("gpus") or 0), max(row["gpus"] for row in rows),
             tr("{count} GPUs were requested, but nodes in {partition} have at most {limit}.")),
            ("mem", _mem_mb(resources.get("mem", "")), max(row["mem_mb"] for row in rows),
             tr("{count} MB of memory was requested, but nodes in {partition} have at most {limit} MB.")),
        )
        for _key, count, limit, message in checks:
            if count and limit and count > limit:
                problems.append(message.format(count=count, partition=label, limit=limit))
        limits = [row["time_limit"] for row in rows if re.fullmatch(r"([0-9]+-)?[0-9:]+", row["time_limit"])]
        requested = _minutes(resources.get("time", ""))
        if requested and limits and len(limits) == len(rows) and requested > max(_minutes(limit) for limit in limits):
            problems.append(tr("A wall time of {time} was requested, but {partition} allows at most {limit}.").format(
                time=resources.get("time"), partition=label, limit=max(limits, key=_minutes)))
        models = sorted({model for row in rows for model in row["gpu_types"]})
        if resources.get("gpu_type") and models and resources["gpu_type"] not in models:
            problems.append(tr("GPU type {gpu_type} is not in {partition}; it has {models}.").format(
                gpu_type=resources["gpu_type"], partition=label, models=", ".join(models)))
    accounts = cluster.get("accounts") or []
    if resources.get("account") and accounts and resources["account"] not in {row["account"] for row in accounts}:
        problems.append(tr("Account {account} is not one of your accounts: {accounts}.").format(
            account=resources["account"], accounts=", ".join(dict.fromkeys(row["account"] for row in accounts))))
    qos = {item for row in accounts for item in row["qos"]}
    if resources.get("qos") and qos and resources["qos"] not in qos:
        problems.append(tr("QoS {qos} is not available to you: {choices}.").format(
            qos=resources["qos"], choices=", ".join(sorted(qos))))
    return problems


def _cluster_cache_path() -> Path:
    """Return the file that remembers each profile's last cluster probe."""
    return state_directory() / "clusters.json"


def _save_cluster_cache(name: str, cluster: Mapping[str, Any]) -> None:
    """Remember one profile's cluster probe for later sessions.

    :param name: the profile name.
    :param cluster: the parsed probe.
    """
    path = _cluster_cache_path()
    with _LOCK, _file_lock(path.with_suffix(".json.lock")):
        data = _read_json(path, {})
        data[str(name).casefold()] = {**cluster, "read_utc": _utc_now()}
        _write_json_atomic(path, data)


def _load_cluster_cache(name: str) -> Dict[str, Any]:
    """Return one profile's remembered cluster probe, or an empty mapping.

    :param name: the profile name.
    :returns: the probe; a cache that cannot be read counts as empty, since
        it only offers suggestions.
    """
    try:
        return dict(_read_json(_cluster_cache_path(), {}).get(str(name).casefold(), {}))
    except (RemoteExecutionError, AttributeError):
        return {}


def _signin_command(profile) -> str:
    """Return the terminal command that opens a reusable OpenSSH connection.

    Signing in once in a terminal, with password and second factor, leaves an
    OpenSSH ControlMaster that spaCR's own commands then reuse without
    another prompt.

    :param profile: the execution profile.
    :returns: the command line.
    """
    options = dict(profile.ssh_options)
    options.setdefault("controlmaster", "auto")
    options.setdefault("controlpersist", "4h")
    sample = replace(profile, ssh_options=options)
    return shlex.join(["ssh", *_ssh_overrides(sample), "-fN", profile.host])


def _allocation_shell_command(profile, allocation: str) -> str:
    """Return the terminal command for an interactive shell inside an allocation.

    :param profile: the execution profile.
    :param allocation: the Slurm job id of the allocation.
    :returns: the command line; the shell runs on the compute node.
    """
    return shlex.join(["ssh", "-t", *_ssh_overrides(profile), profile.host,
                       f"srun --jobid={allocation} --overlap --pty bash -l"])


def _ssh_config_directives(paths):
    """Read bounded static SSH directives; never execute Match or local commands."""
    import glob

    directives = []
    count = 0

    def visit(path, base, stack):
        """Read one bounded config file and expand its static Include directives."""
        nonlocal count
        path = Path(path).expanduser()
        canonical = path.resolve()
        if canonical in stack or len(stack) >= 16 or count >= 128:
            raise RemoteExecutionError(tr('SSH Include cycle or file limit exceeded.'))
        if not path.is_file():
            return
        if path.name in {"id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "id_ecdsa_sk", "id_ed25519_sk"}:
            return
        count += 1
        if path.stat().st_size > 1024 * 1024:
            raise RemoteExecutionError(tr('SSH configuration file is too large.'))
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if "PRIVATE KEY-----" in line:
                    raise RemoteExecutionError(tr('A private-key file cannot be used as SSH configuration.'))
                lexer = shlex.shlex(line, posix=True)
                lexer.whitespace_split = True
                lexer.commenters = "#"
                lexer.escape = ""
                parts = list(lexer)
                if not parts:
                    continue
                key, separator, first = parts[0].partition("=")
                values = ([first] if separator and first else []) + parts[1:]
                if values and values[0].startswith("="):
                    first_value = values[0][1:]
                    values = ([first_value] if first_value else []) + values[1:]
                keyword = key.casefold()
                if keyword == "include":
                    for pattern in values:
                        expanded = os.path.expanduser(os.path.expandvars(pattern))
                        if not os.path.isabs(expanded):
                            expanded = str(base / expanded)
                        for included in sorted(glob.glob(expanded)):
                            directives.append((None, "include_start"))
                            visit(included, base, stack | {canonical})
                            directives.append((None, "include_end"))
                else:
                    directives.append((keyword, values))
    for candidate in paths:
        path = Path(candidate)
        directives.append((None, "top"))
        visit(path, path.parent, set())
    return directives


def _ssh_host_matches(host, patterns):
    """Use SSH's positive and negated wildcard host selection."""
    import fnmatch

    positive = False
    for pattern in patterns:
        negative = pattern.startswith("!")
        token = pattern[1:] if negative else pattern
        if fnmatch.fnmatchcase(host.casefold(), token.casefold()):
            if negative:
                return False
            positive = True
    return positive


def _ssh_lookup(host, directives, *, home=None, default_port="22"):
    """Apply first-value precedence without evaluating Match conditions."""
    selected = {}
    active = True
    include_active = []
    allowed = {"hostname", "user", "port", "identityfile", "proxyjump", "proxycommand",
               "controlmaster", "controlpersist", "controlpath", "userknownhostsfile"}
    for keyword, values in directives:
        if keyword is None:
            if values == "top":
                active = True
            elif values == "include_start":
                include_active.append(active)
            elif values == "include_end":
                active = include_active.pop()
        elif keyword == "host":
            active = all(include_active) and _ssh_host_matches(host, values)
        elif keyword == "match":
            active = False
        elif active and keyword in allowed and values:
            if keyword in {"proxyjump", "proxycommand"} and ({"proxyjump", "proxycommand"} & selected.keys()):
                continue
            if keyword not in selected:
                selected[keyword] = " ".join(values)
    selected.setdefault("hostname", host)
    selected["hostname"] = re.sub(
        r"%[%h]", lambda match: "%" if match.group(0) == "%%" else host,
        selected["hostname"])
    selected.setdefault("port", str(default_port))
    user_home = str(home or Path.home())
    for key in ("identityfile", "userknownhostsfile"):
        if key in selected:
            selected[key] = selected[key].replace("%d", user_home)
            if selected[key].startswith("~/") or selected[key].startswith("~\\"):
                selected[key] = os.path.join(user_home, selected[key][2:].replace("/", os.sep))
    return selected


def _system_ssh_config(environ=None, *, windows=None):
    """Name the OpenSSH client's system-wide config: %PROGRAMDATA%\\ssh\\ssh_config on Windows."""
    environment = os.environ if environ is None else environ
    if (os.name == "nt") if windows is None else windows:
        return Path(environment.get("PROGRAMDATA") or r"C:\ProgramData") / "ssh" / "ssh_config"
    return Path("/etc/ssh/ssh_config")


def _discover_ssh_settings(folder=None, *, home=None, environ=None, system_config=None):
    """Discover non-secret aliases from configuration and public host names."""
    environment = os.environ if environ is None else environ
    home = Path(home or Path.home())
    folders = [Path(folder)] if folder is not None else [home / ".ssh"]
    if folder is None and environment.get("USERPROFILE"):
        candidate = Path(environment["USERPROFILE"]) / ".ssh"
        if candidate not in folders:
            folders.append(candidate)
    paths = [path / "config" for path in folders]
    system = Path(system_config) if system_config is not None else _system_ssh_config(environment)
    if folder is None:
        paths.append(system)
    directives = _ssh_config_directives(paths)
    hosts = set()
    known_ports = {}
    for key, values in directives:
        if key == "host":
            hosts.update(value for value in values if not any(mark in value for mark in "!*?[]")
                         and _HOST_RE.fullmatch(value) and not value.startswith("-"))
    for directory in folders:
        known = directory / "known_hosts"
        if not known.is_file():
            continue
        if known.stat().st_size > 4 * 1024 * 1024:
            raise RemoteExecutionError(tr('SSH known-hosts file is too large.'))
        for line in known.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split()
            if fields and fields[0].startswith("@"):
                fields = fields[1:]
            if not fields or fields[0].startswith(("#", "|")):
                continue
            for token in fields[0].split(","):
                host = token
                if host.startswith("[") and "]:" in host:
                    host, _, port = host[1:].partition("]:")
                    if port.isdigit() and 1 <= int(port) <= 65535:
                        known_ports.setdefault(host, port)
                if _HOST_RE.fullmatch(host) and not host.startswith("-"):
                    hosts.add(host)
    source = next((str(path) for path in paths if path.is_file()), "")
    return [{"alias": host, "config": source, **_ssh_lookup(host, directives, home=home, default_port=known_ports.get(host, "22"))}
            for host in sorted(hosts, key=str.casefold)]


def _ssh_overrides(profile):
    """Render validated connection overrides as separate OpenSSH arguments."""
    names = {"hostname": "HostName", "user": "User", "port": "Port", "identityfile": "IdentityFile",
             "proxyjump": "ProxyJump", "proxycommand": "ProxyCommand", "controlmaster": "ControlMaster",
             "controlpersist": "ControlPersist", "controlpath": "ControlPath",
             "userknownhostsfile": "UserKnownHostsFile"}
    arguments = []
    if profile.ssh_options.get("config"):
        arguments.extend(["-F", profile.ssh_options["config"]])
    for key, name in names.items():
        if profile.ssh_options.get(key):
            value = profile.ssh_options[key]
            if key in {"identityfile", "controlpath"} and any(character.isspace() or character == '"' for character in value):
                value = '"' + value.replace('"', '\\"') + '"'
            arguments.extend(["-o", name + "=" + value])
    master = profile.ssh_options.get("controlmaster", "").casefold()
    if master and master != "no" and not profile.ssh_options.get("controlpath"):
        arguments.extend(["-o", "ControlPath=" + _default_control_path()])
    if _auth_mode(profile) == "gssapi":
        arguments.extend(["-o", "GSSAPIAuthentication=yes", "-o", "GSSAPIDelegateCredentials=yes"])
    return arguments


def _default_control_path() -> str:
    """Return a private ControlPath for a profile that asks for ControlMaster.

    OpenSSH ignores ControlMaster without a ControlPath, so a profile that
    sets only ``ControlMaster auto`` would sign in again on every command.
    The socket lives in spaCR's state directory, or in the temporary
    directory when that path is too long for a Unix socket.

    :returns: the path, with OpenSSH's ``%C`` connection hash.
    """
    import tempfile

    folder = state_directory() / "ssh"
    if len(str(folder)) > 60:
        folder = Path(tempfile.gettempdir()) / f"spacr-ssh-{os.getuid() if hasattr(os, 'getuid') else 'user'}"
    try:
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError:
        pass
    return str(folder / "cm-%C")


_KEYCHAIN_SERVICE = "spaCR SSH"


def _keychain_available():
    """Report whether the optional keyring package can reach an OS keychain."""
    import importlib.util

    return importlib.util.find_spec("keyring") is not None


def _keychain_password(action, account, secret=None):
    """Get, set or delete one stored login password; failures store nothing."""
    try:
        import keyring

        if action == "get":
            return keyring.get_password(_KEYCHAIN_SERVICE, account)
        if action == "set":
            keyring.set_password(_KEYCHAIN_SERVICE, account, secret)
        elif action == "delete":
            keyring.delete_password(_KEYCHAIN_SERVICE, account)
    except Exception:
        return None
    return None


def _ssh_resolve(host, options):
    """Resolve one target from static SSH files plus explicit profile overrides."""
    import getpass

    alias = host.rsplit("@", 1)[-1]
    config = options.get("config")
    paths = [Path(config)] if config else [Path.home() / ".ssh/config", _system_ssh_config()]
    resolved = _ssh_lookup(alias, _ssh_config_directives(paths))
    resolved.update({key: value for key, value in options.items() if value})
    if "@" in host:
        resolved["user"] = host.rsplit("@", 1)[0]
    resolved.setdefault("user", getpass.getuser())
    return resolved


def _keychain_account(host, options):
    """Name the keychain entry by user, host name and port, never by secret."""
    resolved = _ssh_resolve(host, options)
    return f'{resolved["user"]}@{resolved.get("hostname", host)}:{resolved.get("port", "22")}'


def _is_password_prompt(text, echo):
    """Recognise a hidden password prompt; passcodes and push choices never match."""
    text = str(text)
    return not echo and ("password" in text.casefold() or text == tr("Password:"))


class _SSHSessionPool:
    """Keep verified SSH transports for this application session only."""

    def __init__(self):
        """Initialize the prompt hook, verified-session cache and reentrant lock."""
        self.prompt = None
        self._sessions = {}
        self._lock = threading.RLock()

    def _answers(self, title, instructions, prompts):
        """Obtain ephemeral responses without logging or persistence."""
        if self.prompt is None:
            raise RemoteExecutionError(tr('SSH authentication needs an open spaCR window.'))
        responses = self.prompt(title, instructions, prompts)
        if responses is None or len(responses) != len(prompts):
            raise RemoteExecutionError(tr('SSH authentication was cancelled or timed out.'))
        return responses

    def _respond(self, login, title, instructions, prompts):
        """Answer one server challenge, using a remembered password at most once."""
        if login["stored"] and len(prompts) == 1 and _is_password_prompt(*prompts[0]):
            answer = [login["stored"]]
            login["stored"] = None
            login["used_stored"] = True
            return answer
        responses = self._answers(title, instructions, prompts)
        if login["keychain"]:
            entered = [answer for (text, echo), answer in zip(prompts, responses) if _is_password_prompt(text, echo)]
            login["entered"] = entered[:1] or login["entered"]
        return responses

    def _connect(self, host, options, timeout, chain=(), upstream=None, keychain=False):
        """Verify each host key before key or keyboard-interactive authentication."""
        import functools
        import socket

        import paramiko

        if host in chain or len(chain) >= 8:
            raise RemoteExecutionError(tr('SSH jump-host cycle or hop limit exceeded.'))
        alias = host.rsplit("@", 1)[-1]
        config = options.get("config")
        resolved = _ssh_resolve(host, options)
        hostname = resolved.get("hostname", alias)
        username = resolved["user"]
        port = int(resolved.get("port", "22"))
        account = f"{username}@{hostname}:{port}"
        login = {"keychain": keychain, "stored": _keychain_password("get", account) if keychain else None,
                 "used_stored": False, "entered": []}
        respond = functools.partial(self._respond, login)
        sock = None
        transport = None
        parents = []
        try:
            jump = resolved.get("proxyjump", "none")
            proxy = resolved.get("proxycommand", "none")
            if upstream is not None:
                sock = upstream.open_channel("direct-tcpip", (hostname, port), ("127.0.0.1", 0), timeout=timeout)
            elif jump.casefold() != "none":
                hops = jump.split(",")
                if len(hops) + len(chain) >= 8:
                    raise RemoteExecutionError(tr('SSH jump-host cycle or hop limit exceeded.'))
                previous = None
                visited = [*chain, host]
                for hop in hops:
                    jump_options = {"config": config} if config else {}
                    if ":" in hop and not hop.endswith("]"):
                        hop, jump_port = hop.rsplit(":", 1)
                        jump_options["port"] = jump_port
                    parent, ancestors = self._connect(hop, jump_options, timeout, tuple(visited), upstream=previous)
                    parents.extend([parent, *ancestors])
                    previous = parent
                    visited.append(hop)
                sock = previous.open_channel("direct-tcpip", (hostname, port), ("127.0.0.1", 0), timeout=timeout)
            elif proxy.casefold() != "none":
                substitutions = {"h": hostname, "p": str(port), "r": username, "n": alias, "%": "%"}
                command = re.sub(r"%([hprn%])", lambda match: substitutions[match[1]], proxy)
                sock = paramiko.ProxyCommand(command)
            else:
                sock = socket.create_connection((hostname, port), timeout=timeout)
            transport = paramiko.Transport(sock)
            transport.banner_timeout = timeout
            transport.auth_timeout = max(timeout, 120.0)
            transport.start_client(timeout=timeout)
            key = transport.get_remote_server_key()
            known = paramiko.HostKeys()
            lexer = shlex.shlex(resolved.get("userknownhostsfile", ""), posix=True)
            lexer.whitespace_split = True
            lexer.escape = ""
            known_paths = list(lexer) or [
                str(Path.home() / ".ssh/known_hosts"), "/etc/ssh/ssh_known_hosts"]
            for path in known_paths:
                if Path(path).is_file():
                    known.load(path)
            key_host = hostname if port == 22 else f"[{hostname}]:{port}"
            if not known.check(key_host, key):
                raise RemoteExecutionError(tr('SSH host key is unknown or changed; verify it with your administrator before connecting.'))
            agent = paramiko.Agent()
            try:
                keys = list(agent.get_keys())
                identity = resolved.get("identityfile")
                if identity and identity.casefold() != "none":
                    try:
                        keys.insert(0, paramiko.PKey.from_path(Path(identity)))
                    except paramiko.PasswordRequiredException:
                        answer = self._answers(tr("SSH authentication"), "", [(tr("Private-key passphrase:"), False)])
                        try:
                            keys.insert(0, paramiko.PKey.from_path(Path(identity), passphrase=answer[0]))
                        finally:
                            answer.clear()
                for candidate in keys:
                    try:
                        remaining = transport.auth_publickey(username, candidate)
                    except paramiko.AuthenticationException:
                        continue
                    if transport.is_authenticated() or remaining:
                        break
            finally:
                agent.close()
            if not transport.is_authenticated():
                try:
                    try:
                        transport.auth_interactive(username, respond)
                    except paramiko.BadAuthenticationType as exc:
                        if "password" not in exc.allowed_types:
                            raise
                        answer = respond(tr("SSH authentication"), "", [(tr("Password:"), False)])
                        try:
                            remaining = transport.auth_password(username, answer[0], fallback=False)
                        finally:
                            answer.clear()
                        if not transport.is_authenticated() and "keyboard-interactive" in (remaining or ()):
                            transport.auth_interactive(username, respond)
                except paramiko.AuthenticationException as exc:
                    if "timeout" in str(exc).casefold():
                        raise RemoteExecutionError(_auth_timeout_message()) from None
                    raise RemoteExecutionError(_auth_denied_message()) from None
            if not transport.is_authenticated():
                raise RemoteExecutionError(_auth_denied_message())
            if keychain and login["entered"]:
                _keychain_password("set", account, login["entered"][0])
            login["entered"] = []
            return transport, parents
        except BaseException:
            login["entered"] = []
            if login["used_stored"]:
                _keychain_password("delete", account)
            if transport is not None:
                transport.close()
            elif sock is not None:
                sock.close()
            for parent in parents:
                parent.close()
            raise

    def run(self, target, command, *, input_text=None, timeout=60.0):
        """Execute once on a separate command channel, never the MFA channel."""
        import time

        identifier = json.dumps(target, sort_keys=True)
        with self._lock:
            session = self._sessions.get(identifier)
            if session is not None and not session[0].is_active():
                self._discard(identifier)
                session = None
            channel = None
            try:
                if session is None:
                    session = self._connect(target["host"], target["options"], timeout,
                                            keychain=bool(target.get("keychain")))
                    self._sessions[identifier] = session
                channel = session[0].open_session(timeout=timeout)
                channel.settimeout(timeout)
                channel.exec_command(command)
                if input_text is not None:
                    channel.sendall(input_text.encode("utf-8"))
                channel.shutdown_write()
                output, errors = bytearray(), bytearray()
                deadline = time.monotonic() + timeout
                while True:
                    progressed = False
                    for ready, receive, buffer in (
                        (channel.recv_ready, channel.recv, output),
                        (channel.recv_stderr_ready, channel.recv_stderr, errors),
                    ):
                        if ready():
                            buffer.extend(receive(65536))
                            progressed = True
                    if len(output) + len(errors) > 4 * 1024 * 1024:
                        raise RemoteExecutionError(tr('SSH command output exceeded the capture limit.'))
                    if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
                        return CommandResult(channel.recv_exit_status(), output.decode("utf-8", "replace"),
                                             errors.decode("utf-8", "replace"))
                    if time.monotonic() >= deadline:
                        raise RemoteExecutionError(tr('SSH command timed out; it was not submitted again.'))
                    if not progressed:
                        time.sleep(0.01)
            except RemoteExecutionError:
                self._discard(identifier)
                raise
            except Exception as exc:
                self._discard(identifier)
                raise RemoteExecutionError(tr("SSH connection or authentication failed ({error}); test the connection to retry.").format(error=type(exc).__name__)) from None
            finally:
                if channel is not None:
                    channel.close()

    def _discard(self, identifier):
        """Close a failed session and every jump transport it owns."""
        session = self._sessions.pop(identifier, None)
        if session is not None:
            for transport in [session[0], *session[1]]:
                transport.close()

    def close(self):
        """Release all session resources when the application exits."""
        with self._lock:
            for identifier in list(self._sessions):
                self._discard(identifier)


_SSH_SESSIONS = _SSHSessionPool()


def _auth_denied_message():
    """Explain a rejected password, code or push without retrying anything."""
    return tr("The login or second-factor request was denied (for example, a rejected push). Nothing was run; choose Retry sign-in or Test connection to try again.")


def _auth_timeout_message():
    """Explain an unanswered push or code without retrying anything."""
    return tr("The login or second-factor request timed out (for example, an unanswered push). Nothing was run; choose Retry sign-in or Test connection to try again.")


def _interactive_ssh(profile, command):
    """Render a private runner envelope containing non-secret fields only."""
    target = {"host": profile.host, "options": profile.ssh_options}
    if profile.ssh_keychain:
        target["keychain"] = True
    return ["__spacr_interactive_ssh__", json.dumps(target), command]


def _remote_argv(profile: ExecutionProfile, argv: Sequence[str]) -> List[str]:
    """Wrap an argument vector in an ssh invocation, if the profile has a host.

    The vector is joined with shell quoting before being handed to ssh,
    because ssh concatenates its arguments and re-parses them with the
    remote shell -- an unquoted path with a space would arrive as two.

    :param profile: the execution profile.
    :param argv: the command to run.
    :returns: the vector to execute locally.
    """
    if not profile.host:
        return list(argv)
    return _ssh_wrap(profile, shlex.join([str(item) for item in argv]))


def _remote_script(profile: ExecutionProfile, script: str) -> List[str]:
    """Wrap a shell script in an ssh invocation, if the profile has a host.

    :param profile: the execution profile.
    :param script: the script text.
    :returns: the vector to execute locally, with the script quoted as one
        argument.
    """
    if not profile.host:
        return ["sh", "-c", script]
    return _ssh_wrap(profile, "sh -c " + shlex.quote(script))


def _ssh_wrap(profile: ExecutionProfile, command: str) -> List[str]:
    """Wrap one remote command line for the profile's sign-in method.

    :param profile: the execution profile.
    :param command: the remote command line, already shell-quoted.
    :returns: the local argument vector: OpenSSH (with GSSAPI options for
        Kerberos), the in-spaCR session for interactive sign-in, or the
        user's custom SSH command with ``{host}`` and ``{command}`` filled.
    """
    mode = _auth_mode(profile)
    if mode == "interactive":
        return _interactive_ssh(profile, command)
    if mode == "custom":
        argv = _split_template(profile.ssh_command, tr("Custom SSH command"), posix=os.name != "nt")
        placed = any("{command}" in token for token in argv)
        rendered = [token.replace("{host}", profile.host).replace("{command}", command) for token in argv]
        return rendered if placed else [*rendered, command]
    return ["ssh", *_ssh_overrides(profile), profile.host, command]


def _require_ok(result: CommandResult, operation: str) -> str:
    """Return a command's output, or raise with the tail of its error.

    :param result: the finished command.
    :param operation: what was being attempted, used to open the message.
    :returns: the trimmed stdout.
    :raises RemoteExecutionError: on a non-zero exit, carrying the LAST 1200
        characters of stderr -- the useful part of a remote failure is at
        the end, and the whole of it can be a log file.
    """
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        if len(detail) > 1200:
            detail = detail[-1200:]
        raise RemoteExecutionError(
            f"{operation} failed with exit code {result.returncode}"
            + (f": {detail}" if detail else ".")
        )
    return result.stdout.strip()


def _safe_external_id(
    value: str, *, allow_slurm_cluster_suffix: bool = False
) -> str:
    """Validate a job id a scheduler handed back.

    This id is interpolated into later commands -- cancel, poll, log tail --
    so it is checked against a strict pattern before it is stored. An id
    beginning with ``-`` is refused separately: it would be read as an
    option by whatever it is passed to.

    :param value: the scheduler's output.
    :param allow_slurm_cluster_suffix: accept Slurm's ``id;cluster`` form
        and validate the cluster name too.
    :returns: the id alone.
    :raises RemoteExecutionError: if the id, or the cluster name, is empty
        or does not match.
    """
    raw = str(value).strip()
    if allow_slurm_cluster_suffix and ";" in raw:
        external_id, cluster = raw.split(";", 1)
        if cluster and not _JOB_ID_RE.fullmatch(cluster):
            raise RemoteExecutionError(
                f"Scheduler returned an unsafe cluster name: {value!r}."
            )
    else:
        external_id = raw
    if (
        not external_id
        or external_id.startswith("-")
        or not _JOB_ID_RE.fullmatch(external_id)
    ):
        raise RemoteExecutionError(
            f"Scheduler returned an unsafe or empty job ID: {value!r}."
        )
    return external_id


def _remote_paths(
    profile: ExecutionProfile, job: RemoteJob
) -> tuple[str, str, str]:
    """Return the remote job directory, settings path and log path.

    :param profile: the execution profile.
    :param job: the job.
    :returns: the three paths, all under a per-job directory so two jobs
        cannot overwrite each other's settings or log.
    """
    base = profile.workdir.rstrip("/") or "/"
    job_dir = posixpath.join(base, "spacr-jobs",
                             job.job_id)
    return (
        job_dir,
        posixpath.join(job_dir, "settings.json"),
        posixpath.join(job_dir, "job.log"),
    )


def _upload_settings(
    profile: ExecutionProfile,
    job: RemoteJob,
    payload: str,
    runner: CommandRunner,
) -> tuple[str, str, str]:
    """Create the remote job directory and write the settings into it.

    ``umask 077`` first: the settings can carry paths and credentials for
    the user's own data, and a job directory readable by every account on a
    shared cluster is not what submitting a job asks for.

    The payload is piped on stdin rather than interpolated into the script,
    so its content is never parsed by a shell.

    :param profile: the execution profile.
    :param job: the job being submitted.
    :param payload: the settings, as text.
    :param runner: how to execute the command.
    :returns: the job directory, settings path and log path.
    :raises RemoteExecutionError: if the upload fails.
    """
    job_dir, settings_path, log_path = _remote_paths(profile, job)
    script = (
        f"umask 077; mkdir -p -- {shlex.quote(job_dir)} && "
        f"cat > {shlex.quote(settings_path)}"
    )
    result = runner(
        _remote_script(profile, script),
        input_text=payload,
        timeout=120.0,
    )
    _require_ok(result, "Settings upload")
    return job_dir, settings_path, log_path


def _normalise_state(text: str) -> tuple[str, Optional[int]]:
    """Map common scheduler/cloud states to spaCR's compact state model."""
    token = str(text or "").strip().splitlines()
    token = token[0].strip().upper() if token else ""
    token = token.split()[0].rstrip("+") if token else ""
    if token.startswith("EXIT:"):
        try:
            code = int(token.split(":", 1)[1])
        except ValueError:
            return "failed", None
        return ("success" if code == 0 else "failed"), code
    if token in {"SUCCESS", "SUCCEEDED", "COMPLETED", "COMPLETE", "DONE"}:
        return "success", 0
    if token in {
        "FAILED", "FAILURE", "TIMEOUT", "OUT_OF_MEMORY", "OOM",
        "NODE_FAIL", "BOOT_FAIL", "PREEMPTED", "DEADLINE_EXCEEDED",
    }:
        return "failed", None
    if token in {"CANCELLED", "CANCELED", "STOPPED", "TERMINATED"}:
        return "cancelled", None
    if token in {
        "PENDING", "PEND", "QUEUED", "CONFIGURING", "SUBMITTED",
        "SCHEDULED",
    }:
        return "pending", None
    if token in {"RUNNING", "RUN", "STARTED", "EXECUTING"}:
        return "running", None
    return "unknown", None


class _Backend:
    """Base for the SSH, Slurm and command backends.

    :param runner: how commands are executed. THE ONLY INJECTION POINT the
        backends have -- every subclass reaches the outside world through
        it and nowhere else, which is what lets a test drive a real backend
        against a recorded transcript instead of a host.
    """

    def __init__(self, runner: CommandRunner):
        """Hold the runner every command goes through."""
        self.runner = runner

    def submit(
        self, profile: ExecutionProfile, job: RemoteJob, payload: str
    ) -> None:
        """Submit a job.

        :param profile: the execution profile.
        :param job: the job; updated in place with its external id and paths.
        :param payload: the settings to upload.
        :raises NotImplementedError: always -- every backend must define this.
        """
        raise NotImplementedError

    def refresh(self, profile: ExecutionProfile, job: RemoteJob) -> None:
        """Update a job's status from the remote.

        :param profile: the execution profile.
        :param job: the job, updated in place.
        :raises NotImplementedError: always.
        """
        raise NotImplementedError

    def cancel(self, profile: ExecutionProfile, job: RemoteJob) -> None:
        """Cancel a running job.

        :param profile: the execution profile.
        :param job: the job, updated in place.
        :raises NotImplementedError: always.
        """
        raise NotImplementedError

    def logs(
        self, profile: ExecutionProfile, job: RemoteJob, lines: int
    ) -> str:
        """Return a job's log tail.

        :param profile: the execution profile.
        :param job: the job.
        :param lines: how many lines to return.
        :returns: the cached tail. Overridden by backends that can fetch it
            live; the base returns what was last recorded, so a caller always
            gets something rather than an error.
        """
        return job.log_tail


class _SSHBackend(_Backend):
    """Runs the job on a workstation over ssh, detached, tracked by PID.

    THE ONE BACKEND WITH NO SCHEDULER BEHIND IT, which is what makes it
    the awkward one: nothing else knows the job exists, so this has to
    detach it from the ssh session that started it and remember the PID
    itself. A job left attached dies when the connection drops, which on
    a laptop lid is every time.

    Liveness is therefore a signal-0 probe rather than a queue query, and
    a recycled PID is the known limit of that -- see :meth:`poll`.
    """

    def submit(
        self, profile: ExecutionProfile, job: RemoteJob, payload: str
    ) -> None:
        """Start the job on a workstation over ssh, detached, and record its PID.

        The command is backgrounded with its output redirected to the log and
        stdin closed, so the ssh connection can drop without killing it -- which
        is the whole point of a remote workstation run. Its exit code is written
        to a temporary file and RENAMED into place, so a reader can never see a
        half-written code and take it as the result.

        :param profile: the execution profile.
        :param job: the job, updated in place.
        :param payload: the settings to upload.
        :raises RemoteExecutionError: if the upload or the launch fails, or if
            what came back is not a plain process id.
        """
        job_dir, settings_path, log_path = _upload_settings(
            profile, job, payload, self.runner
        )
        script = _job_script_text(profile, job, job_dir, settings_path, log_path)
        output = _require_ok(
            self.runner(
                _remote_script(profile, _detached_launch(
                    profile, job_dir, log_path, script,
                    guard="" if profile.direct_run else _login_guard(profile))),
                input_text=script, timeout=60.0),
            "SSH submission",
        )
        job.job_script = script
        job.external_id = _safe_external_id(output.splitlines()[-1])
        if not job.external_id.isdecimal():
            raise RemoteExecutionError(
                f"SSH workstation returned a non-numeric process ID: "
                f"{job.external_id!r}."
            )
        job.remote_job_dir = job_dir
        job.remote_settings_path = settings_path
        job.log_reference = log_path
        job.status = "running"

    def refresh(self, profile: ExecutionProfile, job: RemoteJob) -> None:
        """Update the job's status on a workstation over ssh.

        :param profile: the execution profile.
        :param job: the job to act on; updated in place.
        """
        exit_path = posixpath.join(job.remote_job_dir, "exit-code")
        script = (
            f"if test -f {shlex.quote(exit_path)}; then "
            f"printf 'EXIT:'; cat {shlex.quote(exit_path)}; "
            f"elif kill -0 {shlex.quote(job.external_id)} 2>/dev/null; then "
            "printf 'RUNNING\\n'; else printf 'UNKNOWN\\n'; fi"
        )
        output = _require_ok(
            self.runner(_remote_script(profile, script), timeout=30.0),
            "SSH status check",
        )
        job.status, job.exit_code = _normalise_state(output)

    def cancel(self, profile: ExecutionProfile, job: RemoteJob) -> None:
        """Kill the job on a workstation over ssh.

        :param profile: the execution profile.
        :param job: the job to act on; updated in place.
        """
        result = self.runner(
            _remote_argv(profile, ["kill", "-TERM", job.external_id]),
            timeout=30.0,
        )
        if result.returncode and "No such process" not in result.stderr:
            _require_ok(result, "SSH cancellation")
        job.status = "cancelled"

    def logs(
        self, profile: ExecutionProfile, job: RemoteJob, lines: int
    ) -> str:
        """Fetch the tail of the job's log over ssh.

        :param profile: the execution profile.
        :param job: the job.
        :param lines: how many lines to fetch.
        :returns: the tail, or the cached one when the fetch fails -- a log that
            cannot be reached is not a reason to lose the last one seen.
        """
        result = self.runner(
            _remote_argv(
                profile,
                ["tail", "-n", str(max(1, min(int(lines), 10000))),
                 "--", job.log_reference],
            ),
            timeout=30.0,
        )
        _require_ok(result, "SSH log retrieval")
        return result.stdout


def _detached_launch(profile, job_dir: str, log_path: str, script: str, *,
                     guard: str = "", wrapper: Sequence[str] = (), session: bool = False) -> str:
    """Return the shell that stores a job script and starts it detached.

    The script arrives on stdin, is written to ``job.sh`` and run by the
    interpreter its ``#!`` line names, in the background with output to the
    log, so the connection can drop without killing it. The exit status is
    written to a temporary file and renamed into place, so a reader never
    sees half a code.

    :param profile: the execution profile.
    :param job_dir: the remote job directory.
    :param log_path: the job's log file.
    :param script: the job script, used to choose its interpreter.
    :param guard: shell lines run first, before anything starts; the
        login-node guard exits there.
    :param wrapper: words put before the interpreter, such as an ``srun``
        line that runs the script inside an allocation.
    :param session: isolate an allocation launcher and its children in a
        recorded process group so cancellation also stops a waiting step.
    :returns: the launch script; it prints the background process id.
    """
    job_sh = posixpath.join(job_dir, "job.sh")
    exit_path = posixpath.join(job_dir, "exit-code")
    exit_tmp = posixpath.join(job_dir, ".exit-code.tmp")
    command = shlex.join([*wrapper, *_interpreter(script), job_sh])
    body = (f"{command}; code=$?; printf '%s\\n' \"$code\" > {shlex.quote(exit_tmp)}; "
            f"mv -f -- {shlex.quote(exit_tmp)} {shlex.quote(exit_path)}")
    marker = shlex.quote(posixpath.join(job_dir, "launcher-group"))
    if session:
        body = f"printf '%s\\n' \"$$\" > {marker} || exit 1\n" + body
    launch = shlex.join(["setsid", "sh", "-c", body]) if session else f"( {body} )"
    preparation = ("command -v setsid >/dev/null 2>&1 || "
                   "{ echo 'Allocation launch requires setsid.' >&2; exit 1; }\n") if session else ""
    recorded = (
        "for spacr_attempt in 1 2 3 4 5 6 7 8 9 10; do\n"
        f"  test -s {marker} && break\n"
        "  kill -0 \"$spacr_launcher\" 2>/dev/null || exit 1\n"
        "  sleep 0.1\n"
        "done\n"
        f"test -s {marker} || {{ kill -s TERM \"$spacr_launcher\"; exit 1; }}\n"
        f"spacr_launcher=$(cat {marker})\n"
    ) if session else ""
    return (
        "umask 077\n"
        f"cat > {shlex.quote(job_sh)} || exit 1\n"
        f"{guard}"
        f"cd {shlex.quote(profile.workdir)} || exit 1\n"
        f"{preparation}{launch} > {shlex.quote(log_path)} 2>&1 < /dev/null &\n"
        "spacr_launcher=$!\n"
        f"{recorded}"
        "printf '%s\\n' \"$spacr_launcher\"\n"
    )


def _allocation_cancel_script(job) -> str:
    """Stop an owned allocation launcher group and confirm its named step is gone."""
    identifier = str(job.external_id)
    if not identifier.isdecimal():
        raise RemoteExecutionError(tr("The allocation launcher has no valid process identity."))
    marker = shlex.quote(posixpath.join(job.remote_job_dir, "launcher-group"))
    job_sh = shlex.quote(posixpath.join(job.remote_job_dir, "job.sh"))
    name = shlex.quote(f"spacr-{job.module}-{job.job_id[:8]}")
    status = (
        "spacr_processes=$(ps -ww -eo pgid=,stat=,args=) || exit 1\n"
        f"spacr_group=$(printf '%s\\n' \"$spacr_processes\" | awk -v group={identifier} -v path={job_sh} "
        "'$1 == group && $2 !~ /^Z/ { live=1; if (index($0,path)) owned=1 } "
        "END { print live+0, owned+0 }')\n"
    )
    return (
        f"spacr_marker={marker}\n"
        "spacr_owned=0\n"
        "if test -f \"$spacr_marker\"; then\n"
        f"  test \"$(cat \"$spacr_marker\")\" = {shlex.quote(identifier)} || exit 1\n"
        f"  {status}"
        "  case \"$spacr_group\" in '1 0') echo 'Launcher ownership changed; cancellation refused.' >&2; exit 1;; esac\n"
        f"  if [ \"$spacr_group\" = '1 1' ]; then env kill -s TERM -- -{identifier} || exit 1; fi\n"
        "  for spacr_attempt in 1 2 3 4 5 6 7 8 9 10; do\n"
        f"    {status}"
        "    case \"$spacr_group\" in '0 0') break;; '1 0') exit 1;; esac\n"
        f"    if [ \"$spacr_attempt\" = 5 ]; then env kill -s KILL -- -{identifier} || exit 1; fi\n"
        "    sleep 0.1\n"
        "  done\n"
        f"  {status}"
        "  test \"$spacr_group\" = '0 0' || exit 1\n"
        "  spacr_owned=1\n"
        "fi\n"
        f"spacr_steps=$(squeue -h -s -j {shlex.quote(job.allocation_id)} -o '%i %j') || exit 1\n"
        f"spacr_matching=$(printf '%s\\n' \"$spacr_steps\" | awk -v name={name} '$2 == name {{print $1}}')\n"
        "if [ \"$spacr_owned\" = 0 ] && [ -z \"$spacr_matching\" ]; then "
        "echo 'No owned launcher or scheduler step could be cancelled.' >&2; exit 1; fi\n"
        "for spacr_step in $spacr_matching; do scancel \"$spacr_step\" || exit 1; done\n"
        f"spacr_steps=$(squeue -h -s -j {shlex.quote(job.allocation_id)} -o '%i %j') || exit 1\n"
        f"if printf '%s\\n' \"$spacr_steps\" | awk -v name={name} '$2 == name {{found=1}} END {{exit !found}}'; "
        "then echo 'The scheduler step is still present; cancellation is not confirmed.' >&2; exit 1; fi\n"
    )


def _tail_log(runner: CommandRunner, profile, job, lines: int, label: str, missing: str) -> str:
    """Fetch the last lines of a job's log file.

    :param runner: how the command is run.
    :param profile: the execution profile.
    :param job: the job.
    :param lines: how many lines, clamped to 1 through 10000.
    :param label: the operation name used in errors.
    :param missing: what to return while the scheduler has not created the
        log yet.
    :returns: the tail.
    """
    result = runner(
        _remote_argv(profile, ["tail", "-n", str(max(1, min(int(lines), 10000))), "--", job.log_reference]),
        timeout=30.0,
    )
    if result.returncode and "No such file" in result.stderr:
        return missing
    _require_ok(result, label)
    return result.stdout


class _SlurmBackend(_Backend):
    """Submits through ``sbatch`` and asks Slurm what happened.

    The easy backend: the scheduler owns the job's identity and its
    state, so nothing here has to be remembered between calls. Its ids
    are opaque strings rather than integers, because a Slurm id can carry
    an array suffix and parsing it to an int loses the task.

    A job with an ``allocation_id`` runs inside an allocation the user
    started, as an ``srun`` step, rather than as a batch job of its own.
    """

    def submit(
        self, profile: ExecutionProfile, job: RemoteJob, payload: str
    ) -> None:
        """Submit the job with ``sbatch`` and record the id it returns.

        :param profile: the execution profile.
        :param job: the job, updated in place.
        :param payload: the settings to upload.
        :raises RemoteExecutionError: if the upload or the submission fails, or
            if the id Slurm returned does not validate.
        """
        job_dir, settings_path, log_path = _upload_settings(
            profile, job, payload, self.runner
        )
        script = _job_script_text(profile, job, job_dir, settings_path, log_path)
        job.job_script = script
        job.remote_job_dir = job_dir
        job.remote_settings_path = settings_path
        job.log_reference = log_path
        if job.allocation_id:
            self._submit_in_allocation(profile, job, script)
            return
        argv = ["sbatch", "--parsable", "--job-name",
                f"spacr-{job.module}-{job.job_id[:8]}",
                "--output", log_path]
        if profile.scheduler_options:
            argv.extend(_split_template(
                profile.scheduler_options, "Slurm options",
                require_program=False, posix=True,
            ))
        output = _require_ok(
            self.runner(
                _remote_argv(profile, argv),
                input_text=script,
                timeout=60.0,
            ),
            "Slurm submission",
        )
        job.external_id = _safe_external_id(
            output.splitlines()[-1], allow_slurm_cluster_suffix=True
        )
        job.status = "queued"

    def _submit_in_allocation(self, profile: ExecutionProfile, job: RemoteJob, script: str) -> None:
        """Start the job as an ``srun`` step inside a running allocation.

        :param profile: the execution profile.
        :param job: the job, with ``allocation_id`` set; updated in place.
        :param script: the rendered job script.
        """
        resources = profile.resources
        wrapper = ["srun", f"--jobid={job.allocation_id}", "--overlap", "--ntasks=1",
                   f"--job-name=spacr-{job.module}-{job.job_id[:8]}"]
        if str(resources.get("cpus", "")).strip():
            wrapper.append(f"--cpus-per-task={resources['cpus'].strip()}")
        if str(resources.get("gpus", "")).strip() not in ("", "0"):
            wrapper.append(f"--gpus={resources['gpus'].strip()}")
        output = _require_ok(
            self.runner(_remote_script(profile, _detached_launch(
                profile, job.remote_job_dir, job.log_reference, script, wrapper=wrapper, session=True)),
                input_text=script, timeout=60.0),
            "Allocation launch",
        )
        job.external_id = _safe_external_id(output.splitlines()[-1])
        if not job.external_id.isdecimal():
            raise RemoteExecutionError(
                f"Allocation launch returned a non-numeric process ID: {job.external_id!r}."
            )
        job.status = "running"

    def refresh(self, profile: ExecutionProfile, job: RemoteJob) -> None:
        """Update the job's status from the scheduler on Slurm.

        :param profile: the execution profile.
        :param job: the job to act on; updated in place.
        """
        if job.allocation_id:
            exit_path = shlex.quote(posixpath.join(job.remote_job_dir, "exit-code"))
            script = (
                f"if test -f {exit_path}; then printf 'EXIT:'; cat {exit_path}; "
                f"elif squeue -h -j {shlex.quote(job.allocation_id)} -o %T 2>/dev/null | grep -q RUNNING; "
                "then printf 'RUNNING\\n'; else printf 'UNKNOWN\\n'; fi"
            )
            output = _require_ok(self.runner(_remote_script(profile, script), timeout=30.0),
                                 "Allocation status check")
            job.status, job.exit_code = _normalise_state(output)
            if job.status == "unknown":
                holder = replace(job, allocation_id="", external_id=job.allocation_id)
                self.refresh(profile, holder)
                if holder.status in TERMINAL_STATES:
                    job.status = "cancelled" if holder.status == "cancelled" else "failed"
                    job.exit_code = holder.exit_code if holder.exit_code else None
            return
        result = self.runner(
            _remote_argv(
                profile,
                ["squeue", "-h", "-j", job.external_id, "-o", "%T"],
            ),
            timeout=30.0,
        )
        output = result.stdout.strip()
        if result.returncode == 0 and output:
            job.status, job.exit_code = _normalise_state(output)
            return
        accounting = self.runner(
            _remote_argv(
                profile,
                ["sacct", "-n", "-X", "-j", job.external_id,
                 "--format=State,ExitCode", "--parsable2"],
            ),
            timeout=30.0,
        )
        text = _require_ok(accounting, "Slurm accounting check")
        line = next((row for row in text.splitlines() if row.strip()), "")
        columns = line.split("|")
        job.status, job.exit_code = _normalise_state(columns[0] if columns else "")
        if len(columns) > 1 and ":" in columns[1]:
            try:
                job.exit_code = int(columns[1].split(":", 1)[0])
            except ValueError:
                pass

    def cancel(self, profile: ExecutionProfile, job: RemoteJob) -> None:
        """Cancel the job with ``scancel`` on Slurm.

        A job inside an allocation cancels only its own step, by name, and
        leaves the allocation for the next run.

        :param profile: the execution profile.
        :param job: the job to act on; updated in place.
        """
        if job.allocation_id:
            script = _allocation_cancel_script(job)
            _require_ok(self.runner(_remote_script(profile, script), timeout=30.0), "Slurm cancellation")
            job.status = "cancelled"
            return
        _require_ok(
            self.runner(
                _remote_argv(profile, ["scancel", job.external_id]),
                timeout=30.0,
            ),
            "Slurm cancellation",
        )
        job.status = "cancelled"

    def logs(
        self, profile: ExecutionProfile, job: RemoteJob, lines: int
    ) -> str:
        """Fetch the tail of the job's log.

        :param profile: the execution profile.
        :param job: the job.
        :param lines: how many lines to fetch.
        :returns: the tail, or the cached one when it cannot be reached.
        """
        return _tail_log(self.runner, profile, job, lines, "Slurm log retrieval",
                         "The Slurm log has not been created yet.")


def _parse_pbs_status(text: str) -> tuple:
    """Read a job's state and exit status from ``qstat -f`` output.

    :param text: the output.
    :returns: ``(state, exit code or None)`` in spaCR's state model.
    """
    state = re.search(r"job_state\s*=\s*(\w)", text)
    code = re.search(r"[Ee]xit_status\s*=\s*(-?\d+)", text)
    exit_code = int(code.group(1)) if code else None
    letter = state.group(1).upper() if state else ""
    if letter in ("Q", "H", "W", "T", "S", "B"):
        return "pending", None
    if letter in ("R", "E"):
        return "running", None
    if letter in ("C", "F", "X"):
        if exit_code is None:
            return "unknown", None
        return ("success" if exit_code == 0 else "failed"), exit_code
    return "unknown", None


class _PBSBackend(_Backend):
    """Submits through ``qsub`` (PBS Pro, OpenPBS or Torque)."""

    def submit(self, profile: ExecutionProfile, job: RemoteJob, payload: str) -> None:
        """Submit the job script to ``qsub`` on stdin and record its id.

        :param profile: the execution profile.
        :param job: the job, updated in place.
        :param payload: the settings to upload.
        """
        job_dir, settings_path, log_path = _upload_settings(profile, job, payload, self.runner)
        script = _job_script_text(profile, job, job_dir, settings_path, log_path)
        argv = ["qsub"]
        if profile.scheduler_options:
            argv.extend(_split_template(profile.scheduler_options, "PBS options",
                                        require_program=False, posix=True))
        output = _require_ok(self.runner(_remote_argv(profile, argv), input_text=script, timeout=60.0),
                             "PBS submission")
        job.external_id = _safe_external_id(output.splitlines()[-1])
        job.job_script = script
        job.remote_job_dir = job_dir
        job.remote_settings_path = settings_path
        job.log_reference = log_path
        job.status = "queued"

    def refresh(self, profile: ExecutionProfile, job: RemoteJob) -> None:
        """Update the job's state from ``qstat -f``, including finished jobs.

        :param profile: the execution profile.
        :param job: the job, updated in place.
        """
        identifier = shlex.quote(job.external_id)
        script = f"qstat -x -f {identifier} 2>/dev/null || qstat -f {identifier}"
        output = _require_ok(self.runner(_remote_script(profile, script), timeout=30.0), "PBS status check")
        job.status, job.exit_code = _parse_pbs_status(output)

    def cancel(self, profile: ExecutionProfile, job: RemoteJob) -> None:
        """Cancel the job with ``qdel``.

        :param profile: the execution profile.
        :param job: the job, updated in place.
        """
        _require_ok(self.runner(_remote_argv(profile, ["qdel", job.external_id]), timeout=30.0),
                    "PBS cancellation")
        job.status = "cancelled"

    def logs(self, profile: ExecutionProfile, job: RemoteJob, lines: int) -> str:
        """Fetch the tail of the job's log.

        :param profile: the execution profile.
        :param job: the job.
        :param lines: how many lines to fetch.
        :returns: the tail.
        """
        return _tail_log(self.runner, profile, job, lines, "PBS log retrieval",
                         "The PBS log has not been created yet.")


def _parse_lsf_status(text: str) -> tuple:
    """Read a job's state and exit code from ``bjobs -o "stat exit_code"``.

    :param text: the output.
    :returns: ``(state, exit code or None)`` in spaCR's state model.
    """
    words = str(text or "").split()
    state = words[0].upper() if words else ""
    code = int(words[1]) if len(words) > 1 and words[1].lstrip("-").isdigit() else None
    if state == "DONE":
        return "success", 0
    if state == "EXIT":
        return "failed", code
    if state == "RUN":
        return "running", None
    if state in ("PEND", "PSUSP", "USUSP", "SSUSP", "WAIT", "PROV"):
        return "pending", None
    return "unknown", None


class _LSFBackend(_Backend):
    """Submits through ``bsub`` (IBM Spectrum LSF)."""

    def submit(self, profile: ExecutionProfile, job: RemoteJob, payload: str) -> None:
        """Submit the job script to ``bsub`` on stdin and record its id.

        :param profile: the execution profile.
        :param job: the job, updated in place.
        :param payload: the settings to upload.
        """
        job_dir, settings_path, log_path = _upload_settings(profile, job, payload, self.runner)
        script = _job_script_text(profile, job, job_dir, settings_path, log_path)
        argv = ["bsub"]
        if profile.scheduler_options:
            argv.extend(_split_template(profile.scheduler_options, "LSF options",
                                        require_program=False, posix=True))
        output = _require_ok(self.runner(_remote_argv(profile, argv), input_text=script, timeout=60.0),
                             "LSF submission")
        match = re.search(r"Job <([^>]+)>", output)
        job.external_id = _safe_external_id(match.group(1) if match else output)
        job.job_script = script
        job.remote_job_dir = job_dir
        job.remote_settings_path = settings_path
        job.log_reference = log_path
        job.status = "queued"

    def refresh(self, profile: ExecutionProfile, job: RemoteJob) -> None:
        """Update the job's state from ``bjobs``.

        :param profile: the execution profile.
        :param job: the job, updated in place.
        """
        output = _require_ok(self.runner(_remote_argv(
            profile, ["bjobs", "-noheader", "-o", "stat exit_code", job.external_id]), timeout=30.0),
            "LSF status check")
        job.status, job.exit_code = _parse_lsf_status(output)

    def cancel(self, profile: ExecutionProfile, job: RemoteJob) -> None:
        """Cancel the job with ``bkill``.

        :param profile: the execution profile.
        :param job: the job, updated in place.
        """
        _require_ok(self.runner(_remote_argv(profile, ["bkill", job.external_id]), timeout=30.0),
                    "LSF cancellation")
        job.status = "cancelled"

    def logs(self, profile: ExecutionProfile, job: RemoteJob, lines: int) -> str:
        """Fetch the tail of the job's log.

        :param profile: the execution profile.
        :param job: the job.
        :param lines: how many lines to fetch.
        :returns: the tail.
        """
        return _tail_log(self.runner, profile, job, lines, "LSF log retrieval",
                         "The LSF log has not been created yet.")


class _CommandBackend(_Backend):
    """Runs whatever the profile says, for a queue spaCR does not model.

    THE ESCAPE HATCH, and deliberately the least helpful backend. A site
    with LSF, PBS, a cloud CLI or a wrapper script gives its own submit,
    poll and cancel commands with placeholders, and spaCR substitutes and
    runs them. It cannot interpret the output beyond an id and an exit
    code, so it reports less than the other two -- that is the trade for
    not needing spaCR to know the scheduler.

    Placeholders are substituted from :meth:`_context`, and a template
    naming one that does not exist fails at submit rather than silently
    running a command with a literal brace in it.
    """

    @staticmethod
    def _context(
        profile: ExecutionProfile, job: RemoteJob
    ) -> Dict[str, str]:
        """The substitution values a user's command template may use."""
        return {
            "job_id": job.job_id,
            "module": job.module,
            "settings": job.settings_path,
            "external_id": job.external_id,
            "profile": profile.name,
        }

    def submit(
        self, profile: ExecutionProfile, job: RemoteJob, payload: str
    ) -> None:
        """Start the job through the profile's own submit template.

        :param profile: the execution profile.
        :param job: the job, updated in place.
        :param payload: the settings to upload.
        :raises RemoteExecutionError: if the template cannot be rendered, the
            submission fails, or the id it returned does not validate.
        """
        del payload
        output = _require_ok(
            self.runner(
                _render_template(
                    profile.submit_command, self._context(profile, job),
                    "Submit command",
                ),
                timeout=120.0,
            ),
            "Cloud/custom submission",
        )
        identifier = ""
        if profile.job_id_pattern:
            match = re.search(profile.job_id_pattern, output)
            if match:
                identifier = (
                    match.groupdict().get("id")
                    if match.groupdict() else match.group(1)
                    if match.groups() else match.group(0)
                )
        else:
            identifier = next(
                (line.strip() for line in output.splitlines() if line.strip()),
                "",
            )
        job.external_id = _safe_external_id(identifier)
        job.log_reference = "custom command"
        job.status = "queued"

    def refresh(self, profile: ExecutionProfile, job: RemoteJob) -> None:
        """Update the job's status through the profile's status template on the configured scheduler.

        :param profile: the execution profile.
        :param job: the job to act on; updated in place.
        """
        output = _require_ok(
            self.runner(
                _render_template(
                    profile.status_command, self._context(profile, job),
                    "Status command",
                ),
                timeout=60.0,
            ),
            "Cloud/custom status check",
        )
        job.status, job.exit_code = _normalise_state(output)

    def cancel(self, profile: ExecutionProfile, job: RemoteJob) -> None:
        """Cancel the job through the profile's cancel template on the configured scheduler.

        :param profile: the execution profile.
        :param job: the job to act on; updated in place.
        """
        _require_ok(
            self.runner(
                _render_template(
                    profile.cancel_command, self._context(profile, job),
                    "Cancel command",
                ),
                timeout=60.0,
            ),
            "Cloud/custom cancellation",
        )
        job.status = "cancelled"

    def logs(
        self, profile: ExecutionProfile, job: RemoteJob, lines: int
    ) -> str:
        """Fetch the job's log through the profile's own log template.

        :param profile: the execution profile.
        :param job: the job.
        :param lines: how many lines to fetch.
        :returns: the tail, or the cached one when it cannot be reached.
        """
        del lines
        if not profile.log_command:
            return (
                "This command profile has no log command. Add one with "
                "{external_id} in the execution profile."
            )
        result = self.runner(
            _render_template(
                profile.log_command, self._context(profile, job),
                "Log command",
            ),
            timeout=60.0,
        )
        _require_ok(result, "Cloud/custom log retrieval")
        return result.stdout


def _backend(profile: ExecutionProfile, runner: CommandRunner) -> _Backend:
    """Return the backend a profile names.

    :param profile: the execution profile.
    :param runner: how the backend executes commands.
    :returns: the backend.
    :raises RemoteExecutionError: if the profile names one this build does
        not have.
    """
    if profile.backend == "ssh":
        return _SSHBackend(runner)
    if profile.backend == "slurm":
        return _SlurmBackend(runner)
    if profile.backend == "pbs":
        return _PBSBackend(runner)
    if profile.backend == "lsf":
        return _LSFBackend(runner)
    if profile.backend == "command":
        return _CommandBackend(runner)
    raise RemoteExecutionError(f"Unsupported backend: {profile.backend}")


class RemoteJobManager:
    """Submit, monitor, cancel and inspect persistent remote jobs.

    All methods are synchronous and may perform network I/O.  GUI callers must
    invoke them on a worker thread; the shipped Distributed Jobs screen does.

    :param profile_store: injectable persistent profile store; omitted creates
        the default :class:`ProfileStore`.
    :param job_store: injectable persistent job store; omitted creates the
        default :class:`JobStore`.
    :param runner: shell-free callable accepting an argument vector, timeout,
        and optional input and returning :class:`CommandResult`.
    """

    def __init__(
        self,
        profile_store: Optional[ProfileStore] = None,
        job_store: Optional[JobStore] = None,
        runner: CommandRunner = _run_command,
    ):
        """Create the manager over its two stores and a command runner.

        :param profile_store: where execution profiles are read from; ``None``
            opens the default store.
        :param job_store: where submitted jobs are recorded; ``None`` opens the
            default store.
        :param runner: how commands are executed. Injectable so a test can drive
            the manager without running anything.
        """
        self.profiles = profile_store or ProfileStore()
        self.jobs = job_store or JobStore()
        self.runner = runner

    def submit(
        self,
        module: str,
        settings: Mapping[str, Any],
        profile_name: str,
    ) -> RemoteJob:
        """Submit resolved settings through a named execution profile.

        :param module: headless spaCR module name accepted by ``spacr-run``.
        :param settings: resolved module settings to serialize for the job.
        :param profile_name: name of the execution profile to use.
        """
        return _submit_job(self, module, settings, profile_name)

    def refresh(self, job_id: str, *, include_logs: bool = True) -> RemoteJob:
        """Poll one non-terminal job and optionally retain its latest log tail.

        :param job_id: complete local job identifier or unambiguous prefix.
        """
        job = self.jobs.get(job_id)
        if job.status in TERMINAL_STATES:
            return job
        profile = ExecutionProfile.from_dict(job.profile)
        backend = _backend(profile, self.runner)
        try:
            backend.refresh(profile, job)
            if include_logs and job.log_reference:
                try:
                    job.log_tail = backend.logs(profile, job, 200)
                except RemoteExecutionError as exc:
                    job.log_tail = f"Log not available yet: {exc}"
            job.error = _exit_hint(job.exit_code, profile) if job.status == "failed" else ""
        except Exception as exc:
            job.error = f"{type(exc).__name__}: {exc}"
        self.jobs.save(job)
        return job

    def refresh_all(self, *, include_logs: bool = False) -> List[RemoteJob]:
        """Poll every active job and return the complete newest-first list."""
        for job in self.jobs.list():
            if job.status in ACTIVE_STATES or job.status == "unknown":
                self.refresh(job.job_id, include_logs=include_logs)
        return self.jobs.list()

    def cancel(self, job_id: str) -> RemoteJob:
        """Request cancellation and persist the result.

        :param job_id: complete local job identifier or unambiguous prefix.
        """
        job = self.jobs.get(job_id)
        if job.status in TERMINAL_STATES:
            return job
        profile = ExecutionProfile.from_dict(job.profile)
        try:
            _backend(profile, self.runner).cancel(profile, job)
            job.error = ""
        except Exception as exc:
            job.error = f"{type(exc).__name__}: {exc}"
            self.jobs.save(job)
            if isinstance(exc, RemoteExecutionError):
                raise
            raise RemoteExecutionError(job.error) from exc
        self.jobs.save(job)
        return job

    def logs(self, job_id: str, lines: int = 200) -> str:
        """Retrieve and persist the tail of one remote job's log.

        :param job_id: complete local job identifier or unambiguous prefix.
        """
        job = self.jobs.get(job_id)
        profile = ExecutionProfile.from_dict(job.profile)
        text = _backend(profile, self.runner).logs(profile, job, lines)
        job.log_tail = text
        self.jobs.save(job)
        return text


def _submit_job(manager: RemoteJobManager, module: str, settings: Mapping[str, Any],
                profile_name: str, allocation: str = "") -> RemoteJob:
    """Submit resolved settings through a named profile, optionally into an allocation.

    :param manager: the job manager.
    :param module: headless spaCR module name accepted by ``spacr-run``.
    :param settings: resolved module settings to serialize for the job.
    :param profile_name: name of the execution profile to use.
    :param allocation: local id (or unambiguous prefix) of a running Slurm
        allocation started by :func:`_start_allocation`; blank submits a
        batch job.
    :returns: the recorded job.
    :raises RemoteExecutionError: for an unknown module, a direct run aimed
        at a login node, an allocation that is not running, or any failure
        of the submission itself.
    """
    from .cli import resolve_module

    module_record = resolve_module(module)
    if module_record is None:
        raise RemoteExecutionError(
            f"{module!r} is not a headless spaCR module. Run "
            "`spacr-run --list` to see valid modules."
        )
    profile = manager.profiles.get(profile_name).validate()
    _refuse_login_node(profile)
    allocation_id = ""
    if allocation:
        holder = manager.jobs.get(allocation)
        if holder.module != "allocation" or holder.profile_name.casefold() != profile.name.casefold():
            raise RemoteExecutionError(tr("Job {job} is not an allocation of profile {profile}.").format(
                job=holder.job_id[:12], profile=profile.name))
        saved = ExecutionProfile.from_dict(holder.profile)
        connection_fields = ("backend", "host", "ssh_options", "ssh_command")
        if (_auth_mode(saved) != _auth_mode(profile)
                or any(getattr(saved, key) != getattr(profile, key) for key in connection_fields)):
            raise RemoteExecutionError(tr("Allocation {job} belongs to the original connection for profile {profile}. Restore that connection or start a new allocation before submitting.").format(
                job=holder.job_id[:12], profile=profile.name))
        if holder.status != "running":
            raise RemoteExecutionError(tr("Allocation {job} is {status}; wait until it is running, then submit.").format(
                job=holder.job_id[:12], status=holder.status))
        allocation_id = holder.external_id
    job_id = uuid.uuid4().hex
    job_dir = manager.jobs.path.parent / "jobs" / job_id
    mapped = map_settings_paths(
        dict(settings), profile.local_root, profile.remote_root
    )
    settings_path = job_dir / "settings.json"
    payload = json.dumps(mapped, indent=2, sort_keys=True, default=str) + "\n"
    try:
        job_dir.mkdir(parents=True, exist_ok=False)
        settings_path.write_text(payload, encoding="utf-8")
    except OSError as exc:
        raise RemoteExecutionError(
            f"Could not prepare local job record {job_dir}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    job = RemoteJob(
        job_id=job_id,
        module=module_record.key,
        profile_name=profile.name,
        backend=profile.backend,
        settings_path=str(settings_path),
        settings_sha256=digest,
        profile=profile.to_dict(),
        allocation_id=allocation_id,
    )
    manager.jobs.save(job)
    try:
        _backend(profile, manager.runner).submit(profile, job, payload)
        if job.job_script:
            (job_dir / "job.sh").write_text(job.job_script, encoding="utf-8")
        manager.jobs.save(job)
    except Exception as exc:
        job.status = "failed"
        job.error = f"{type(exc).__name__}: {exc}"
        manager.jobs.save(job)
        if isinstance(exc, RemoteExecutionError):
            raise
        raise RemoteExecutionError(job.error) from exc
    return job


def _start_allocation(manager: RemoteJobManager, profile_name: str) -> RemoteJob:
    """Ask Slurm for an allocation with the profile's resources.

    The allocation is a batch job that holds its nodes and runs nothing, so
    requesting it never blocks spaCR while the queue is busy, it is listed
    and cancelled like any job, and spaCR jobs then run inside it as
    ``srun`` steps on the compute node -- never on the login node.

    :param manager: the job manager.
    :param profile_name: the Slurm profile to allocate with.
    :returns: the allocation's job record (module ``allocation``).
    :raises RemoteExecutionError: for a profile that is not Slurm, or if the
        request fails.
    """
    profile = manager.profiles.get(profile_name).validate()
    if profile.backend != "slurm":
        raise RemoteExecutionError(tr("Allocations need a Slurm profile. On PBS use qsub -I, and on LSF bsub -Is, in a terminal."))
    job = RemoteJob(uuid.uuid4().hex, "allocation", profile.name, "slurm", profile=profile.to_dict())
    manager.jobs.save(job)
    job_dir, _settings_path, log_path = _remote_paths(profile, job)
    name = f"spacr-allocation-{job.job_id[:8]}"
    holder = ("#!/bin/bash\n" + _scheduler_directives(profile, name, log_path) + "\n"
              "echo \"spaCR allocation $SLURM_JOB_ID ready on $(hostname)\"\n"
              "exec sleep infinity\n")
    argv = ["sbatch", "--parsable", "--job-name", name, "--output", log_path]
    if profile.scheduler_options:
        argv.extend(_split_template(profile.scheduler_options, "Slurm options", require_program=False, posix=True))
    try:
        _require_ok(manager.runner(_remote_script(profile, f"umask 077; mkdir -p -- {shlex.quote(job_dir)}"),
                                   timeout=60.0), "Allocation folder")
        output = _require_ok(manager.runner(_remote_argv(profile, argv), input_text=holder, timeout=60.0),
                             "Slurm allocation")
        job.external_id = _safe_external_id(output.splitlines()[-1], allow_slurm_cluster_suffix=True)
        job.remote_job_dir = job_dir
        job.log_reference = log_path
        job.job_script = holder
        job.status = "queued"
        manager.jobs.save(job)
    except Exception as exc:
        job.status = "failed"
        job.error = f"{type(exc).__name__}: {exc}"
        manager.jobs.save(job)
        if isinstance(exc, RemoteExecutionError):
            raise
        raise RemoteExecutionError(job.error) from exc
    return job


def _log_chunk(manager: RemoteJobManager, job_id: str, offset: int, limit: int = 262144) -> tuple:
    """Fetch the part of a job's log written since ``offset``.

    Polled at the profile's interval over the same connection -- an OpenSSH
    ControlMaster or the in-spaCR session -- so following a log costs no new
    sign-in. The remote side reports the new offset itself, so bytes are
    never repeated or skipped even if the log grows during the read; a log
    that shrank (rotated or rewritten) is read again from the start.

    :param manager: the job manager.
    :param job_id: the local job id or an unambiguous prefix.
    :param offset: bytes already shown.
    :param limit: requested bytes per read, bounded to 4–262144 so one
        complete UTF-8 character always fits. Incomplete trailing characters
        stay unread until a later poll; a terminal short read finalizes an
        invalid trailing sequence with a replacement character.
    :returns: ``(new text, new offset)``.
    """
    if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset <= 2 ** 63 - 1:
        raise RemoteExecutionError(tr("A log offset must be a nonnegative integer."))
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise RemoteExecutionError(tr("A log read limit must be a positive integer."))
    limit = max(4, min(limit, 262144))
    job = manager.jobs.get(job_id)
    profile = ExecutionProfile.from_dict(job.profile)
    if job.backend == "command" or not job.log_reference.startswith("/"):
        if job.backend != "command" or not profile.log_command:
            return "", offset
        text = _backend(profile, manager.runner).logs(profile, job, 10000)
        start = offset if offset <= len(text) else 0
        return text[start:], len(text)
    path = shlex.quote(job.log_reference)
    script = (
        f"f={path}\n"
        f"test -f \"$f\" || {{ echo {int(offset)}; exit 0; }}\n"
        "size=$(wc -c < \"$f\" | tr -d ' ')\n"
        f"start={int(offset)}\n"
        "if [ \"$size\" -lt \"$start\" ]; then start=0; fi\n"
        "n=$((size - start))\n"
        f"if [ \"$n\" -gt {int(limit)} ]; then n={int(limit)}; fi\n"
        "echo \"$start\"\n"
        "if [ \"$n\" -gt 0 ]; then tail -c +$((start + 1)) -- \"$f\" | head -c \"$n\" | base64; fi\n"
    )
    result = manager.runner(_remote_script(profile, script), timeout=60.0)
    _require_ok(result, "Log retrieval")
    first, _, rest = result.stdout.partition("\n")
    try:
        start = int(first.strip())
        if start < 0 or start not in (0, offset):
            raise ValueError("Unexpected log offset")
        payload = base64.b64decode("".join(rest.split()), validate=True)
        if len(payload) > limit:
            raise ValueError("Oversized log response")
    except (ValueError, UnicodeError) as exc:
        raise RemoteExecutionError(tr("The remote log response was invalid.")) from exc
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    text = decoder.decode(payload, final=job.status in TERMINAL_STATES and len(payload) < limit)
    pending, _state = decoder.getstate()
    return text, start + len(payload) - len(pending)
