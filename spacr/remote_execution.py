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
import hashlib
import json
import os
import posixpath
import re
import shlex
import subprocess
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

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

BACKENDS = ("ssh", "slurm", "command")
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
    """Result returned by the injectable command runner."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


def _run_command(
    argv: Sequence[str],
    *,
    input_text: Optional[str] = None,
    timeout: float = 60.0,
) -> CommandResult:
    """Run one argument vector without a shell and capture UTF-8 output."""
    try:
        result = subprocess.run(
            list(argv),
            input=input_text,
            text=True,
            capture_output=True,
            timeout=max(1.0, float(timeout)),
            check=False,
        )
    except FileNotFoundError as exc:
        raise RemoteExecutionError(
            f"Command not found: {argv[0]!r}. Install it or update the "
            "execution profile."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RemoteExecutionError(
            f"Command timed out after {timeout:g}s: {argv[0]}"
        ) from exc
    except OSError as exc:
        raise RemoteExecutionError(
            f"Could not start {argv[0]!r}: {type(exc).__name__}: {exc}"
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
    text = str(value or "").strip()
    if not text and not allow_empty:
        raise RemoteExecutionError(f"{label} is required.")
    if "\x00" in text or "\n" in text or "\r" in text:
        raise RemoteExecutionError(f"{label} may not contain newlines or NUL.")
    return text


def _path_text(value: Any, label: str, *, allow_empty: bool = False) -> str:
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

        if self.backend in {"ssh", "slurm"}:
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
                    require_program=False,
                )

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
        """Construct and validate a profile from JSON-compatible data."""
        fields = cls.__dataclass_fields__
        profile = cls(**{key: value[key] for key in fields if key in value})
        return profile.validate()


@dataclass
class RemoteJob:
    """Persistent local record of one submitted job."""

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

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RemoteJob":
        """Construct a job from a stored mapping, tolerating future fields."""
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
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        yield
    finally:
        try:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        handle.close()


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RemoteExecutionError(
            f"Could not read {path}: {type(exc).__name__}: {exc}"
        ) from exc


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = ""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            value, indent=2, sort_keys=True, default=str
        ) + "\n"
        fd, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
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
    """Atomic persistent store for execution profiles."""

    def __init__(self, path: Optional[os.PathLike] = None):
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
        """Return a named profile or raise a user-facing error."""
        wanted = str(name).casefold()
        for profile in self.list():
            if profile.name.casefold() == wanted:
                return profile
        raise RemoteExecutionError(
            f"Execution profile {name!r} does not exist."
        )

    def save(self, profile: ExecutionProfile) -> None:
        """Insert or replace one profile atomically."""
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
        """Delete one profile; return whether it existed."""
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
    """Atomic persistent store for remote job metadata."""

    def __init__(self, path: Optional[os.PathLike] = None):
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
        """Return a job by full ID or unambiguous prefix."""
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
        """Insert or replace one job atomically."""
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
) -> List[str]:
    try:
        argv = shlex.split(command, posix=os.name != "nt")
    except ValueError as exc:
        raise RemoteExecutionError(f"{label} cannot be parsed: {exc}") from exc
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


def _remote_argv(profile: ExecutionProfile, argv: Sequence[str]) -> List[str]:
    if not profile.host:
        return list(argv)
    return ["ssh", profile.host, shlex.join([str(item) for item in argv])]


def _remote_script(profile: ExecutionProfile, script: str) -> List[str]:
    if not profile.host:
        return ["sh", "-c", script]
    return ["ssh", profile.host, "sh -c " + shlex.quote(script)]


def _require_ok(result: CommandResult, operation: str) -> str:
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
    def __init__(self, runner: CommandRunner):
        self.runner = runner

    def submit(
        self, profile: ExecutionProfile, job: RemoteJob, payload: str
    ) -> None:
        raise NotImplementedError

    def refresh(self, profile: ExecutionProfile, job: RemoteJob) -> None:
        raise NotImplementedError

    def cancel(self, profile: ExecutionProfile, job: RemoteJob) -> None:
        raise NotImplementedError

    def logs(
        self, profile: ExecutionProfile, job: RemoteJob, lines: int
    ) -> str:
        return job.log_tail


class _SSHBackend(_Backend):
    def submit(
        self, profile: ExecutionProfile, job: RemoteJob, payload: str
    ) -> None:
        job_dir, settings_path, log_path = _upload_settings(
            profile, job, payload, self.runner
        )
        exit_path = posixpath.join(job_dir, "exit-code")
        exit_tmp = posixpath.join(job_dir, ".exit-code.tmp")
        command = shlex.join([
            profile.runner, job.module, "--settings", settings_path,
        ])
        script = (
            f"cd {shlex.quote(profile.workdir)} && "
            f"( {command}; code=$?; "
            f"printf '%s\\n' \"$code\" > {shlex.quote(exit_tmp)}; "
            f"mv -f -- {shlex.quote(exit_tmp)} {shlex.quote(exit_path)} "
            f") > {shlex.quote(log_path)} 2>&1 < /dev/null & "
            "printf '%s\\n' \"$!\""
        )
        output = _require_ok(
            self.runner(_remote_script(profile, script), timeout=60.0),
            "SSH submission",
        )
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


class _SlurmBackend(_Backend):
    def submit(
        self, profile: ExecutionProfile, job: RemoteJob, payload: str
    ) -> None:
        job_dir, settings_path, log_path = _upload_settings(
            profile, job, payload, self.runner
        )
        script = (
            "#!/bin/sh\n"
            "set -eu\n"
            f"cd {shlex.quote(profile.workdir)}\n"
            f"exec {shlex.join([profile.runner, job.module, '--settings', settings_path])}\n"
        )
        argv = ["sbatch", "--parsable", "--job-name",
                f"spacr-{job.module}-{job.job_id[:8]}",
                "--output", log_path]
        if profile.scheduler_options:
            argv.extend(_split_template(
                profile.scheduler_options, "Slurm options",
                require_program=False,
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
        job.remote_job_dir = job_dir
        job.remote_settings_path = settings_path
        job.log_reference = log_path
        job.status = "queued"

    def refresh(self, profile: ExecutionProfile, job: RemoteJob) -> None:
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
        result = self.runner(
            _remote_argv(
                profile,
                ["tail", "-n", str(max(1, min(int(lines), 10000))),
                 "--", job.log_reference],
            ),
            timeout=30.0,
        )
        if result.returncode and "No such file" in result.stderr:
            return "The Slurm log has not been created yet."
        _require_ok(result, "Slurm log retrieval")
        return result.stdout


class _CommandBackend(_Backend):
    @staticmethod
    def _context(
        profile: ExecutionProfile, job: RemoteJob
    ) -> Dict[str, str]:
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
    if profile.backend == "ssh":
        return _SSHBackend(runner)
    if profile.backend == "slurm":
        return _SlurmBackend(runner)
    if profile.backend == "command":
        return _CommandBackend(runner)
    raise RemoteExecutionError(f"Unsupported backend: {profile.backend}")


class RemoteJobManager:
    """Submit, monitor, cancel and inspect persistent remote jobs.

    All methods are synchronous and may perform network I/O.  GUI callers must
    invoke them on a worker thread; the shipped Distributed Jobs screen does.
    """

    def __init__(
        self,
        profile_store: Optional[ProfileStore] = None,
        job_store: Optional[JobStore] = None,
        runner: CommandRunner = _run_command,
    ):
        self.profiles = profile_store or ProfileStore()
        self.jobs = job_store or JobStore()
        self.runner = runner

    def submit(
        self,
        module: str,
        settings: Mapping[str, Any],
        profile_name: str,
    ) -> RemoteJob:
        """Submit resolved settings through a named execution profile."""
        from .cli import resolve_module

        module_record = resolve_module(module)
        if module_record is None:
            raise RemoteExecutionError(
                f"{module!r} is not a headless spaCR module. Run "
                "`spacr-run --list` to see valid modules."
            )
        profile = self.profiles.get(profile_name).validate()
        job_id = uuid.uuid4().hex
        # Keep settings beside the selected JobStore.  Besides making custom
        # installations coherent, this ensures a portable/test store never
        # leaks files into the user's normal state directory.
        job_dir = self.jobs.path.parent / "jobs" / job_id
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
        )
        self.jobs.save(job)
        try:
            _backend(profile, self.runner).submit(profile, job, payload)
            self.jobs.save(job)
        except Exception as exc:
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            self.jobs.save(job)
            if isinstance(exc, RemoteExecutionError):
                raise
            raise RemoteExecutionError(job.error) from exc
        return job

    def refresh(self, job_id: str, *, include_logs: bool = True) -> RemoteJob:
        """Poll one non-terminal job and optionally retain its latest log tail."""
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
            job.error = ""
        except Exception as exc:
            # A transient SSH/cloud outage must not turn a still-running remote
            # job into a permanent failure.  Preserve its prior state and make
            # the polling error visible.
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
        """Request cancellation and persist the result."""
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
        """Retrieve and persist the tail of one remote job's log."""
        job = self.jobs.get(job_id)
        profile = ExecutionProfile.from_dict(job.profile)
        text = _backend(profile, self.runner).logs(profile, job, lines)
        job.log_tail = text
        self.jobs.save(job)
        return text
