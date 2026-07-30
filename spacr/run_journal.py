"""
Run journal — reproducibility record for every pipeline invocation.

Every time a spaCR pipeline runs (mask / measure / classify / …),
:func:`open_run` writes a timestamped folder under ``~/.spacr/runs/``
containing everything a reviewer needs to reproduce the result:

::

    ~/.spacr/runs/2026-07-23_143507_ab12cd34/
        settings.csv          # exact settings dict, Key,Value CSV
        settings.json         # same, JSON (source of truth for machines)
        manifest.json         # spaCR version, git hash, python, packages,
                              # torch / cuda / cellpose, start time,
                              # end time, elapsed, exit status, model hashes
        log.txt               # tail of ~/.spacr/logs/spacr.log for the run
        stdout.txt            # captured pipeline stdout (if opened via
                              # :func:`capture_stdout`)
        outputs/              # optional — any pipeline-emitted artifacts
                              # (masks, DBs, CSVs, plots) copied in

Public API::

    from spacr.run_journal import open_run

    with open_run("mask", settings) as run:
        preprocess_generate_masks(settings)
        run.attach_output(Path("/path/to/mask.tif"))
        run.set_status("success")

The context manager records start / end timestamps, catches
exceptions, writes a ``FAILED`` marker when something raises, and
returns the run folder path on ``__exit__`` so callers can log it.

Consumers of the journal:

* ``spacr repro <run-folder>`` — replays the run (see
  :mod:`spacr.cli_repro`).
* AI Console → "File as issue" — includes the last run's manifest
  when present so bug reports are self-contained.
* Home screen "Recent runs" list — enumerated from
  :func:`recent_runs` newest first.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

LOG = logging.getLogger("spacr.run_journal")

MANIFEST_SCHEMA_VERSION = 2
"""Current on-disk reproducibility-manifest schema."""

_HASH_ALGORITHM = "sha256"
_SEED_KEY_PARTS = ("seed", "random_state", "random_seed")
_OUTPUT_KEY_PARTS = (
    "dst", "dest", "output", "export", "save_path", "report_path",
    "checkpoint_path", "tar_path",
)
_PATH_KEY_PARTS = (
    "src", "path", "file", "folder", "dir", "model", "database", "db",
    "csv", "json", "plate", "project", "checkpoint", "weights", "tar",
    *_OUTPUT_KEY_PARTS,
)
_IGNORED_TREE_NAMES = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".spacr", ".ipynb_checkpoints",
})


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def runs_root() -> Path:
    """Return ``~/.spacr/runs``; created on first access."""
    p = Path.home() / ".spacr" / "runs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _new_run_dir(app_key: str) -> Path:
    """Return a fresh ``<UTC-timestamp>_<short-uuid>__<app>`` folder."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    tag = uuid.uuid4().hex[:8]
    safe_app = re.sub(r"[^A-Za-z0-9_.-]+", "_", app_key or "unknown").strip(
        "._"
    ) or "unknown"
    d = runs_root() / f"{ts}_{tag}__{safe_app}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "outputs").mkdir(exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Environment + version snapshot
# ---------------------------------------------------------------------------

def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version as _v
        return _v(name)
    except Exception:
        return "not installed"


def _git_hash() -> Optional[str]:
    """If spaCR is installed as an editable checkout, return the current
    commit hash + a dirty-tree marker; else None."""
    try:
        import spacr
        pkg_dir = Path(spacr.__file__).resolve().parent.parent
        head = subprocess.run(
            ["git", "-C", str(pkg_dir), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3,
        )
        if head.returncode != 0:
            return None
        sha = head.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(pkg_dir), "status", "--porcelain"],
            capture_output=True, text=True, timeout=3,
        )
        if dirty.stdout.strip():
            sha += "+dirty"
        return sha
    except Exception:
        return None


@lru_cache(maxsize=1)
def _installed_packages() -> Dict[str, str]:
    """Return all installed Python distributions as ``{name: version}``.

    Distribution metadata is read without importing packages, so recording a
    run does not initialize CUDA, Qt, or another expensive optional runtime.
    Duplicate normalized names are collapsed deterministically.
    """
    packages: Dict[str, str] = {}
    try:
        for dist in importlib.metadata.distributions():
            name = str(dist.metadata.get("Name") or "").strip()
            if name:
                packages[name.lower().replace("_", "-")] = str(
                    dist.version or "unknown"
                )
    except Exception as exc:
        LOG.warning("Could not enumerate installed packages: %s", exc)
    return dict(sorted(packages.items()))


def _env_snapshot() -> Dict[str, Any]:
    """Capture host and complete package versions for reproduction."""
    return {
        "spacr":         _pkg_version("spacr"),
        "spacr_git":     _git_hash(),
        "python":        sys.version.split()[0],
        "platform":      platform.platform(),
        "torch":         _pkg_version("torch"),
        "torchvision":   _pkg_version("torchvision"),
        "cellpose":      _pkg_version("cellpose"),
        "pyside6":       _pkg_version("PySide6"),
        "numpy":         _pkg_version("numpy"),
        "scipy":         _pkg_version("scipy"),
        "pandas":        _pkg_version("pandas"),
        "scikit_image":  _pkg_version("scikit-image"),
        "scikit_learn":  _pkg_version("scikit-learn"),
        "packages":       _installed_packages(),
    }


def hash_file(
    path: Path,
    chunk_size: int = 1 << 20,
    *,
    full: bool = False,
) -> Optional[str]:
    """Return a SHA-256 digest of a file, or ``None`` on error.

    :param path: regular file to hash.
    :param chunk_size: bytes read per iteration.
    :param full: return all 64 hexadecimal characters. The default preserves
        the historic 16-character public result; reproducibility manifests use
        the full digest.
    """
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        digest = h.hexdigest()
        return digest if full else digest[:16]
    except Exception as exc:
        LOG.warning("Could not hash %s: %s", path, exc)
        return None


def _json_digest(value: Any) -> str:
    """Return a full SHA-256 of a JSON-compatible value."""
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace ``path`` with UTF-8 ``text``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _is_output_key(key: str) -> bool:
    """Return whether a setting key conventionally denotes an output path."""
    lowered = str(key).lower()
    tokens = set(filter(None, re.split(r"[^a-z0-9]+", lowered)))
    return (
        bool(tokens.intersection({"dst", "dest", "output", "export"}))
        or any(
            lowered == part
            or lowered.endswith(f"_{part}")
            or lowered.startswith(f"{part}_")
            for part in _OUTPUT_KEY_PARTS
        )
    )


def _is_path_key(key: str) -> bool:
    """Return whether a setting key conventionally contains filesystem data."""
    lowered = str(key).lower()
    tokens = set(filter(None, re.split(r"[^a-z0-9]+", lowered)))
    return (
        bool(tokens.intersection(_PATH_KEY_PARTS))
        or any(
            lowered == part
            or lowered.endswith(f"_{part}")
            or lowered.startswith(f"{part}_")
            for part in _PATH_KEY_PARTS
        )
    )


def _walk_setting_values(
    value: Any,
    key: str = "",
    *,
    _seen: Optional[Set[int]] = None,
) -> Iterator[Tuple[str, Any]]:
    """Yield scalar setting values with their dotted/list-qualified key."""
    if _seen is None:
        _seen = set()
    if isinstance(value, dict):
        marker = id(value)
        if marker in _seen:
            return
        _seen.add(marker)
        for child_key, child in value.items():
            name = f"{key}.{child_key}" if key else str(child_key)
            yield from _walk_setting_values(child, name, _seen=_seen)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        marker = id(value)
        if marker in _seen:
            return
        _seen.add(marker)
        for index, child in enumerate(value):
            yield from _walk_setting_values(
                child, f"{key}[{index}]", _seen=_seen,
            )
        return
    yield key, value


def extract_seeds(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Capture declared seeds plus already-loaded RNG state fingerprints.

    The state fingerprints do not import NumPy or Torch. When those libraries
    are already loaded, their state is captured; otherwise the manifest says
    so explicitly instead of changing application startup behavior.
    """
    declared: Dict[str, Any] = {}
    for key, value in _walk_setting_values(settings):
        lowered = key.lower()
        if any(part in lowered for part in _SEED_KEY_PARTS):
            declared[key] = value

    result: Dict[str, Any] = {
        "declared": declared,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "python_random_state_sha256": hashlib.sha256(
            repr(random.getstate()).encode("utf-8")
        ).hexdigest(),
    }
    numpy = sys.modules.get("numpy")
    try:
        if numpy is not None:
            result["numpy_random_state_sha256"] = hashlib.sha256(
                repr(numpy.random.get_state()).encode("utf-8")
            ).hexdigest()
        else:
            result["numpy_random_state_sha256"] = None
    except Exception as exc:
        result["numpy_random_state_error"] = str(exc)
    torch = sys.modules.get("torch")
    try:
        result["torch_initial_seed"] = (
            int(torch.initial_seed()) if torch is not None else None
        )
    except Exception as exc:
        result["torch_seed_error"] = str(exc)
    return result


def _setting_path_candidates(
    settings: Dict[str, Any],
) -> List[Tuple[str, Path, bool]]:
    """Return unique plausible paths found recursively in ``settings``.

    Existing strings are paths regardless of their setting name. Non-existing
    strings are retained only for output-looking keys, allowing a new output
    file or directory to be discovered when the run closes.
    """
    candidates: List[Tuple[str, Path, bool]] = []
    seen: Set[Tuple[str, str]] = set()
    for key, value in _walk_setting_values(settings):
        if not isinstance(value, (str, os.PathLike)):
            continue
        raw = os.fspath(value).strip()
        if not raw or "\x00" in raw or "\n" in raw:
            continue
        try:
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
            path = path.resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            continue
        output_only = _is_output_key(key)
        if not _is_path_key(key) and not output_only:
            continue
        if not path.exists() and not output_only:
            continue
        token = (key, str(path))
        if token not in seen:
            seen.add(token)
            candidates.append((key, path, output_only))
    return candidates


def _iter_files(path: Path, excluded_roots: Iterable[Path]) -> Iterator[Path]:
    """Yield regular files below ``path`` in deterministic order."""
    try:
        resolved_excludes = tuple(
            root.resolve(strict=False) for root in excluded_roots
        )
    except Exception:
        resolved_excludes = tuple(excluded_roots)

    def excluded(candidate: Path) -> bool:
        try:
            resolved = candidate.resolve(strict=False)
            return any(
                resolved == root or root in resolved.parents
                for root in resolved_excludes
            )
        except Exception:
            return False

    if excluded(path):
        return
    if path.is_file() and not path.is_symlink():
        yield path
        return
    if not path.is_dir():
        return
    for root, dirnames, filenames in os.walk(path, followlinks=False):
        root_path = Path(root)
        dirnames[:] = sorted(
            name for name in dirnames
            if name not in _IGNORED_TREE_NAMES
            and not (root_path / name).is_symlink()
            and not excluded(root_path / name)
        )
        for name in sorted(filenames):
            candidate = root_path / name
            if not candidate.is_symlink() and not excluded(candidate):
                yield candidate


def _file_record(path: Path) -> Optional[Dict[str, Any]]:
    """Return a complete immutable provenance record for one file."""
    try:
        stat = path.stat()
        digest = hash_file(path, full=True)
        if digest is None:
            return None
        return {
            "sha256": digest,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    except OSError as exc:
        LOG.warning("Could not inspect %s: %s", path, exc)
        return None


def _inventory_signature(path: Path) -> Optional[Tuple[int, int]]:
    """Return ``(size, mtime_ns)`` for output-change detection."""
    try:
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Run object
# ---------------------------------------------------------------------------

@dataclass
class Run:
    """A single pipeline invocation's on-disk record.

    Instances are produced by :func:`open_run`. Users don't construct
    them directly.

    :ivar app_key: id of the pipeline app that opened the run.
    :ivar settings: settings dict originally passed to the pipeline.
    :ivar dir: run folder path (``~/.spacr/runs/<ts>_<uuid>__<app>``).
    :ivar start_ts: unix epoch seconds when the run opened.
    :ivar end_ts: unix epoch seconds when the run closed (set by
        :func:`open_run` on exit).
    :ivar status: ``"running"`` / ``"success"`` / ``"failed"``.
    :ivar model_hashes: dict of ``{human-name → sha256-16}``. Populated
        by callers via :meth:`record_model`.
    :ivar model_files: full SHA-256, size, and path records for models.
    :ivar input_hashes: per-file full SHA-256 input provenance.
    :ivar output_hashes: per-file full SHA-256 output provenance.
    :ivar seeds: declared seeds and runtime RNG-state identifiers.
    :ivar provenance_warnings: non-fatal path/hash failures retained in the
        manifest instead of being silently discarded.
    :ivar run_warnings: distinct warning lines emitted by the pipeline.
    :ivar environment: host, spaCR, Git, and installed-package versions.
    """
    app_key: str
    settings: Dict[str, Any]
    dir: Path
    start_ts: float = field(default_factory=time.time)
    end_ts: Optional[float] = None
    status: str = "running"
    model_hashes: Dict[str, str] = field(default_factory=dict)
    model_files: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    input_hashes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    output_hashes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    seeds: Dict[str, Any] = field(default_factory=dict)
    provenance_warnings: List[str] = field(default_factory=list)
    run_warnings: List[str] = field(default_factory=list)
    environment: Dict[str, Any] = field(default_factory=dict)
    stdout_path: Optional[Path] = None
    error_traceback: str = ""
    _path_candidates: List[Tuple[str, Path, bool]] = field(
        default_factory=list, repr=False,
    )
    _baseline: Dict[str, Tuple[int, int]] = field(
        default_factory=dict, repr=False,
    )
    _start_cpu_s: float = field(default_factory=time.process_time, repr=False)

    # -- external mutations ------------------------------------------------
    def record_model(self, name: str, checkpoint_path: Any) -> None:
        """Fingerprint ``checkpoint_path`` and remember it under ``name``.

        Silently no-ops if the file is unreadable — model logging must
        never itself fail a run.
        """
        try:
            p = Path(checkpoint_path)
            digest = hash_file(p)
            if digest:
                self.model_hashes[name] = f"{p.name}:{digest}"
            record = _file_record(p)
            if record:
                record["path"] = str(p.resolve(strict=False))
                self.model_files[name] = record
        except Exception as exc:
            warning = f"model {name!r} could not be recorded: {exc}"
            self.provenance_warnings.append(warning)
            LOG.warning(warning)

    def record_input(self, path: Any, *, setting_key: str = "") -> None:
        """Hash a file or directory as an explicit run input.

        Directories are recorded as one full SHA-256 record per regular file.
        Unreadable files are reported in ``provenance_warnings`` and logs.

        :param path: input file or directory.
        :param setting_key: optional setting that referred to ``path``.
        """
        self._record_tree(
            Path(path), self.input_hashes, setting_key=setting_key,
        )

    def record_output(self, path: Any, *, setting_key: str = "") -> None:
        """Hash a file or directory as an explicit run output."""
        self._record_tree(
            Path(path), self.output_hashes, setting_key=setting_key,
        )

    def attach_output(self, src_path: Any) -> Optional[Path]:
        """Copy ``src_path`` into the run's ``outputs/`` folder.

        :param src_path: path to a file (or folder) worth preserving
            for reproducibility.
        :returns: destination path in the run folder, or ``None`` on
            error.
        """
        try:
            src = Path(src_path)
            dst = self.dir / "outputs" / src.name
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
            self.record_output(src, setting_key="attach_output")
            return dst
        except Exception as exc:
            warning = f"output {src_path!r} could not be attached: {exc}"
            self.provenance_warnings.append(warning)
            LOG.warning(warning)
            return None

    def set_status(self, status: str) -> None:
        """Explicitly stamp ``status`` (``success`` / ``failed`` / …)."""
        self.status = status

    def record_warning(self, message: Any) -> None:
        """Retain a distinct warning for the run-history dashboard.

        :param message: warning text captured from pipeline stdout/stderr or
            supplied directly by pipeline code.
        """
        text = str(message or "").strip()
        if text and text not in self.run_warnings:
            # Bound the manifest if a library repeats the same warning with
            # field-specific text thousands of times.
            if len(self.run_warnings) < 500:
                self.run_warnings.append(text)

    # -- private -----------------------------------------------------------
    def _record_tree(
        self,
        path: Path,
        destination: Dict[str, Dict[str, Any]],
        *,
        setting_key: str = "",
    ) -> None:
        """Hash ``path`` into ``destination`` without raising."""
        try:
            path = path.expanduser().resolve(strict=False)
            found = False
            for file_path in _iter_files(path, (self.dir, runs_root())):
                found = True
                record = _file_record(file_path)
                if record is None:
                    warning = f"could not hash provenance file {file_path}"
                    self.provenance_warnings.append(warning)
                    continue
                if setting_key:
                    prior = destination.get(str(file_path), {})
                    keys = set(prior.get("setting_keys") or [])
                    keys.add(setting_key)
                    record["setting_keys"] = sorted(keys)
                destination[str(file_path)] = record
            if not found and path.exists():
                warning = f"no regular provenance files found in {path}"
                self.provenance_warnings.append(warning)
                LOG.warning(warning)
        except Exception as exc:
            warning = f"could not record provenance path {path}: {exc}"
            self.provenance_warnings.append(warning)
            LOG.warning(warning)

    def _capture_initial_provenance(self) -> None:
        """Discover path-valued inputs and retain a before-run inventory."""
        self.seeds = extract_seeds(self.settings)
        self._path_candidates = _setting_path_candidates(self.settings)
        seen_files: Set[str] = set()
        for key, path, output_only in self._path_candidates:
            if path.exists() and not output_only:
                self.record_input(path, setting_key=key)
            root = path if path.is_dir() else path.parent
            if not root.exists():
                continue
            for file_path in _iter_files(root, (self.dir, runs_root())):
                file_key = str(file_path)
                if file_key in seen_files:
                    continue
                seen_files.add(file_key)
                signature = _inventory_signature(file_path)
                if signature is not None:
                    self._baseline[file_key] = signature

    def _capture_final_provenance(self) -> None:
        """Hash files created or modified under setting-derived roots."""
        seen_files: Set[str] = set()
        for key, path, _output_only in self._path_candidates:
            root = path if path.is_dir() else path.parent
            if not root.exists():
                continue
            for file_path in _iter_files(root, (self.dir, runs_root())):
                file_key = str(file_path)
                if file_key in seen_files:
                    continue
                seen_files.add(file_key)
                signature = _inventory_signature(file_path)
                if (
                    signature is not None
                    and self._baseline.get(file_key) != signature
                ):
                    record = _file_record(file_path)
                    if record is None:
                        warning = (
                            f"could not hash changed output file {file_path}"
                        )
                        self.provenance_warnings.append(warning)
                        continue
                    record["setting_keys"] = [key]
                    self.output_hashes[file_key] = record

    def _write_manifest(self) -> None:
        """Atomically write the current versioned ``manifest.json``."""
        elapsed = None
        if self.end_ts is not None:
            elapsed = round(self.end_ts - self.start_ts, 3)
        settings_sha256 = (
            hash_file(self.dir / "settings.json", full=True)
            or _json_digest(self.settings)
        )
        wall_s = elapsed
        performance = {
            "wall_s": wall_s,
            "process_cpu_s": round(
                max(0.0, time.process_time() - self._start_cpu_s), 3,
            ),
            "input_files": len(self.input_hashes),
            "input_bytes": sum(
                int(record.get("size_bytes", 0) or 0)
                for record in self.input_hashes.values()
            ),
            "output_files": len(self.output_hashes),
            "output_bytes": sum(
                int(record.get("size_bytes", 0) or 0)
                for record in self.output_hashes.values()
            ),
        }
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "hash_algorithm": _HASH_ALGORITHM,
            "app_key":       self.app_key,
            "start_utc":     datetime.fromtimestamp(
                self.start_ts, tz=timezone.utc).isoformat(),
            "end_utc":       (datetime.fromtimestamp(
                self.end_ts, tz=timezone.utc).isoformat()
                if self.end_ts else None),
            "elapsed_s":     elapsed,
            "status":        self.status,
            "env":           self.environment,
            "model_hashes":  self.model_hashes,
            "model_files":   self.model_files,
            "settings_file": "settings.json",
            "settings_sha256": settings_sha256,
            "seeds":         self.seeds,
            "input_hashes":  self.input_hashes,
            "input_tree_sha256": _json_digest(self.input_hashes),
            "output_hashes": self.output_hashes,
            "output_tree_sha256": _json_digest(self.output_hashes),
            "provenance_warnings": self.provenance_warnings,
            "warnings":       self.run_warnings,
            "performance":    performance,
            "n_settings":    len(self.settings),
            "traceback":     self.error_traceback or None,
        }
        _atomic_write_text(
            self.dir / "manifest.json",
            json.dumps(manifest, indent=2, default=str, sort_keys=True),
        )

    def _write_settings(self) -> None:
        """Write exact machine- and human-readable settings snapshots."""
        # Machine-friendly JSON (source of truth)
        _atomic_write_text(
            self.dir / "settings.json",
            json.dumps(self.settings, indent=2, default=str, sort_keys=True),
        )
        # Human-friendly CSV (Key,Value — spacr.utils.load_settings compatible)
        with open(self.dir / "settings.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Key", "Value"])
            for k, v in self.settings.items():
                w.writerow([k, "" if v is None else str(v)])

    def _snapshot_log_tail(self, n: int = 200) -> None:
        """Copy the last ``n`` application-log lines into this run folder."""
        try:
            from .logging_util import log_path
            src = log_path()
            if not src.exists():
                return
            with open(src, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            (self.dir / "log.txt").write_text("".join(lines[-n:]))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

_RUN_LOCAL = threading.local()


def current_run() -> Optional["Run"]:
    """Return the :class:`Run` currently open in this process, or None.

    Useful for pipeline internals (Cellpose model loaders, etc.) that
    want to record a model checkpoint hash without every caller
    needing to plumb the :class:`Run` object through.

    State is thread-local because the batch runner and database browser can
    execute independent workers concurrently.
    """
    return getattr(_RUN_LOCAL, "active", None)


@contextmanager
def open_run(app_key: str, settings: Dict[str, Any]) -> Iterator[Run]:
    """Open a fresh run journal folder around a pipeline invocation.

    Example::

        from spacr.run_journal import open_run

        with open_run("mask", settings) as run:
            run.record_model("cellpose_cyto", ckpt_path)
            preprocess_generate_masks(settings)
            run.set_status("success")

    :param app_key: pipeline id (``"mask"``, ``"measure"``, …).
    :param settings: settings dict handed to the pipeline. Written to
        the run folder as both JSON and CSV.
    :yields: the :class:`Run` object.
    """
    run = Run(app_key=app_key, settings=dict(settings or {}),
                dir=_new_run_dir(app_key))
    run.environment = _env_snapshot()
    run._write_settings()
    run._capture_initial_provenance()
    # A running manifest makes an interrupted process visible and auditable.
    run._write_manifest()
    LOG.info("run opened → %s", run.dir)
    prev_active = current_run()
    _RUN_LOCAL.active = run
    try:
        yield run
        if run.status == "running":
            run.status = "success"
    except BaseException as e:
        import traceback as _tb
        run.status = "failed"
        run.error_traceback = "".join(
            _tb.format_exception(type(e), e, e.__traceback__)
        )
        raise
    finally:
        _RUN_LOCAL.active = prev_active
        run.end_ts = time.time()
        try:
            run._capture_final_provenance()
        except Exception as exc:
            warning = f"final output provenance failed: {exc}"
            run.provenance_warnings.append(warning)
            LOG.exception(warning)
        try:
            run._snapshot_log_tail()
            run._write_manifest()
        except Exception:
            # A manifest failure is never silent, but it also must not mask the
            # original pipeline exception during context-manager unwinding.
            LOG.exception("Could not finalize run manifest in %s", run.dir)
        LOG.info("run closed [%s] in %.1fs → %s",
                  run.status, run.end_ts - run.start_ts, run.dir)


# ---------------------------------------------------------------------------
# Listing + lookup
# ---------------------------------------------------------------------------

def recent_runs(limit: int = 10) -> List[Dict[str, Any]]:
    """Return the ``limit`` most-recent runs newest-first.

    Ordered by the manifest's ``start_utc`` timestamp (parsed as
    :class:`datetime.datetime`), so runs opened in the same wall-
    clock second still sort correctly — folder names alone truncate
    to seconds and would produce ties. Corrupt / partial run
    folders are silently skipped.

    Each entry is a dict with keys ``dir`` (Path), ``app_key`` (str),
    ``status`` (str), ``start_utc`` (ISO str), ``elapsed_s`` (float),
    and the raw ``manifest`` (dict, best-effort).
    """
    all_entries: List[Dict[str, Any]] = []
    root = runs_root()
    for d in root.iterdir():
        if not d.is_dir():
            continue
        manifest_path = d / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            m = json.loads(manifest_path.read_text())
        except Exception:
            continue
        all_entries.append({
            "dir":       d,
            "app_key":   m.get("app_key", "?"),
            "status":    m.get("status", "?"),
            "start_utc": m.get("start_utc", ""),
            "elapsed_s": m.get("elapsed_s"),
            "manifest":  m,
        })
    # Sort by parsed timestamp (with folder-mtime as tiebreaker for
    # any manifests missing / mangled start_utc).
    def _sort_key(e):
        s = e.get("start_utc") or ""
        try:
            return (datetime.fromisoformat(s), e["dir"].stat().st_mtime)
        except Exception:
            return (datetime.fromtimestamp(0, tz=timezone.utc),
                     e["dir"].stat().st_mtime)
    all_entries.sort(key=_sort_key, reverse=True)
    return all_entries[:limit]


def search_runs(
    query: str = "",
    *,
    app_key: str = "",
    status: str = "",
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return searchable, dashboard-ready records for all journalled runs.

    Search covers the run id, module, status, settings keys and values, input
    and output paths, warnings, failure traceback, and environment versions.
    Corrupt or interrupted run folders remain visible with an explicit
    ``"corrupt"``/``"running"`` status and diagnostic warnings.

    :param query: whitespace-separated case-insensitive terms; every term must
        occur somewhere in the record.
    :param app_key: optional exact module filter.
    :param status: optional exact status filter.
    :param limit: maximum returned records after newest-first sorting.
    :returns: JSON-friendly record dictionaries. ``dir`` remains a
        :class:`~pathlib.Path` for convenient GUI use.
    """
    records: List[Dict[str, Any]] = []
    root = runs_root()
    try:
        directories = [path for path in root.iterdir() if path.is_dir()]
    except OSError as exc:
        LOG.warning("Could not enumerate run history at %s: %s", root, exc)
        return []

    wanted_app = str(app_key or "").strip().lower()
    wanted_status = str(status or "").strip().lower()
    terms = [term.casefold() for term in str(query or "").split() if term]

    for directory in directories:
        rec = _read_run_record(directory)
        manifest = rec["manifest"] or {}
        settings = rec["settings"] or {}
        current_app = str(manifest.get("app_key") or "unknown")
        current_status = str(manifest.get("status") or "").lower()
        if not current_status:
            current_status = (
                "running"
                if "no manifest.json (run may still be in flight)" in rec["errors"]
                else "corrupt"
            )
        if wanted_app and current_app.lower() != wanted_app:
            continue
        if wanted_status and current_status != wanted_status:
            continue

        warnings_list: List[str] = []
        for key in ("warnings", "provenance_warnings"):
            values = manifest.get(key) or []
            if isinstance(values, (list, tuple)):
                warnings_list.extend(str(value) for value in values if value)
            elif values:
                warnings_list.append(str(values))
        warnings_list.extend(str(error) for error in rec["errors"])

        # Legacy manifests did not structure warnings. Their bounded log tail
        # is still useful, so surface warning-looking lines without failing a
        # history scan on encoding or permissions.
        if "warnings" not in manifest:
            log_path = directory / "log.txt"
            try:
                if log_path.exists():
                    for line in log_path.read_text(
                        encoding="utf-8", errors="replace",
                    ).splitlines():
                        if re.search(
                            r"\b(?:warning|warn)\b", line, re.IGNORECASE,
                        ):
                            warnings_list.append(line.strip())
            except OSError as exc:
                warnings_list.append(
                    f"log.txt unreadable ({type(exc).__name__})"
                )
        warnings_list = list(dict.fromkeys(warnings_list))

        inputs = manifest.get("input_hashes")
        outputs = manifest.get("output_hashes")
        inputs = inputs if isinstance(inputs, dict) else {}
        outputs = outputs if isinstance(outputs, dict) else {}
        performance = manifest.get("performance")
        if not isinstance(performance, dict):
            performance = {
                "wall_s": manifest.get("elapsed_s"),
                "process_cpu_s": None,
                "input_files": len(inputs),
                "input_bytes": sum(
                    int(value.get("size_bytes", 0) or 0)
                    for value in inputs.values() if isinstance(value, dict)
                ),
                "output_files": len(outputs),
                "output_bytes": sum(
                    int(value.get("size_bytes", 0) or 0)
                    for value in outputs.values() if isinstance(value, dict)
                ),
            }
        failure = str(manifest.get("traceback") or "")
        record: Dict[str, Any] = {
            "dir": directory,
            "run_id": directory.name,
            "app_key": current_app,
            "status": current_status,
            "start_utc": str(manifest.get("start_utc") or ""),
            "end_utc": str(manifest.get("end_utc") or ""),
            "elapsed_s": manifest.get("elapsed_s"),
            "performance": performance,
            "settings": settings,
            "inputs": inputs,
            "outputs": outputs,
            "models": manifest.get("model_files")
                      or manifest.get("model_hashes") or {},
            "warnings": warnings_list,
            "failure": failure,
            "environment": (
                manifest.get("env")
                if isinstance(manifest.get("env"), dict) else {}
            ),
            "manifest": manifest,
        }
        if terms:
            haystack = json.dumps(
                {
                    "run_id": record["run_id"],
                    "app_key": current_app,
                    "status": current_status,
                    "settings": settings,
                    "inputs": list(inputs),
                    "outputs": list(outputs),
                    "warnings": warnings_list,
                    "failure": failure,
                    "environment": record["environment"],
                },
                default=str,
                sort_keys=True,
            ).casefold()
            if not all(term in haystack for term in terms):
                continue
        records.append(record)

    def _history_sort_key(record: Dict[str, Any]) -> Tuple[datetime, float]:
        try:
            started = datetime.fromisoformat(record["start_utc"])
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            started = datetime.fromtimestamp(0, tz=timezone.utc)
        try:
            mtime = record["dir"].stat().st_mtime
        except OSError:
            mtime = 0.0
        return started, mtime

    records.sort(key=_history_sort_key, reverse=True)
    if limit is not None:
        return records[:max(0, int(limit))]
    return records


def journal_totals() -> Dict[str, int]:
    """Return aggregate counts across every stored run.

    Powers the Home-screen insights dashboard: ``total_runs`` (all
    manifests seen), ``mask_runs`` / ``measure_runs`` / ``classify_runs``
    (per-app tallies), and ``models_recorded`` (distinct model hashes
    ever recorded across all mask runs). Returns zeros when no journal
    exists yet.

    Cheap enough to call on Home-screen construction — one iterdir + a
    file read per run folder. Callers that need more should cache.
    """
    totals = {"total_runs": 0, "mask_runs": 0, "measure_runs": 0,
                "classify_runs": 0, "models_recorded": 0}
    seen_models: set = set()
    root = runs_root()
    if not root.exists():
        return totals
    for d in root.iterdir():
        if not d.is_dir():
            continue
        manifest_path = d / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            m = json.loads(manifest_path.read_text())
        except Exception:
            continue
        totals["total_runs"] += 1
        app_key = m.get("app_key", "")
        if app_key in ("mask", "measure", "classify"):
            totals[f"{app_key}_runs"] += 1
        # Per-run model record. The manifest stores these under
        # ``model_hashes`` as a {name: "filename:digest"} dict (see
        # Run._write_manifest) — NOT a ``models`` list, which never
        # matched and left models_recorded stuck at 0.
        hashes = m.get("model_hashes") or {}
        if isinstance(hashes, dict):
            for digest in hashes.values():
                if digest:
                    seen_models.add(digest)
        # Back-compat: also honour a legacy ``models`` list of dicts.
        for model in m.get("models", []) or []:
            sha = model.get("sha256") if isinstance(model, dict) else None
            if sha:
                seen_models.add(sha)
    totals["models_recorded"] = len(seen_models)
    return totals


def load_run_settings(run_dir: Path) -> Dict[str, Any]:
    """Read a run's ``settings.json`` (falling back to settings.csv)."""
    run_dir = Path(run_dir)
    j = run_dir / "settings.json"
    if j.exists():
        return json.loads(j.read_text())
    c = run_dir / "settings.csv"
    if not c.exists():
        raise FileNotFoundError(f"no settings in {run_dir}")
    return _read_settings_csv(c)


def _read_settings_csv(path: Path) -> Dict[str, Any]:
    """Parse a ``Key,Value`` settings CSV into a plain dict."""
    out: Dict[str, Any] = {}
    with open(path) as f:
        for row in csv.reader(f):
            if any("\x00" in cell for cell in row):
                raise csv.Error("embedded NUL byte in settings CSV")
            if row and row[0] and row[0] != "Key":
                out[row[0]] = row[1] if len(row) > 1 else ""
    return out


# ---------------------------------------------------------------------------
# Provenance diff — "what actually changed between run A and run B?"
# ---------------------------------------------------------------------------
#
# Why this is not a plain key-by-key dict diff: spaCR's settings schema
# moves between releases. Diffing a run recorded on 1.4.3.7 (204 keys)
# against one recorded on 1.4.8.7 (38 keys) turns up ~196 "differences",
# of which *zero* are decisions the user made — they are keys that simply
# did not exist on one side. The signal (a knob the user turned) drowns
# in schema drift. So the diff buckets keys by presence first and only
# calls a key "changed" when it exists in BOTH runs.

# Strings that stand in for "unset" once a value has round-tripped
# through CSV (``None`` is written as an empty cell) or through
# ``json.dumps(..., default=str)``.
_NULLISH_STRINGS = frozenset({"", "none", "null"})

_LITERAL_LEAD = "([{'\"-+.0123456789"


def _normalize_value(v: Any, _depth: int = 0) -> Any:
    """Canonicalise a settings value so equal *meanings* compare equal.

    Settings reach the journal by several routes (a live Python dict, a
    JSON round-trip with ``default=str``, a ``Key,Value`` CSV where every
    cell is a string), so the same setting can be recorded as ``[0, 1, 2]``
    in one run and ``"[0, 1, 2]"`` in another. Comparing by ``repr`` would
    flag that as a change; it is not one.

    Normalisation applied, in order:

    * ``None`` stays ``None``; ``bool`` stays ``bool``.
    * ``float('nan')`` → the sentinel ``"<nan>"`` (so NaN == NaN, since
      IEEE NaN compares unequal to itself and would report a phantom
      change on every diff).
    * anything with ``.tolist()`` (numpy arrays / scalars) is converted
      to plain Python first.
    * :class:`~pathlib.Path` → ``str``.
    * ``str`` → stripped; ``""`` / ``"none"`` / ``"null"``
      (case-insensitive) → ``None``; ``"true"`` / ``"false"`` → ``bool``;
      otherwise, if it looks like a Python literal, it is parsed with
      :func:`ast.literal_eval` and normalised recursively, so
      ``"[0, 1, 2]" == [0, 1, 2]`` and ``"3" == 3``. Un-parseable strings
      are returned as-is.
    * ``list`` / ``tuple`` → tuple of normalised elements (so a list and
      a tuple of the same contents compare equal).
    * ``dict`` → tuple of ``(str(key), normalised value)`` sorted by key.
    * ``set`` / ``frozenset`` → frozenset of normalised elements.

    Recursion is capped at eight levels; deeper structures fall back to
    ``repr`` so a self-referential settings value cannot hang the diff.
    """
    if _depth > 8:
        return repr(v)
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, float):
        return "<nan>" if v != v else v
    if isinstance(v, str):
        return _normalize_str(v, _depth)
    if isinstance(v, Path):
        return str(v)
    tolist = getattr(v, "tolist", None)
    if callable(tolist):
        try:
            return _normalize_value(tolist(), _depth + 1)
        except Exception:
            return repr(v)
    if isinstance(v, dict):
        try:
            return tuple(sorted(
                (str(k), _normalize_value(val, _depth + 1))
                for k, val in v.items()
            ))
        except Exception:
            return tuple(
                (str(k), _normalize_value(val, _depth + 1))
                for k, val in v.items()
            )
    if isinstance(v, (list, tuple)):
        return tuple(_normalize_value(x, _depth + 1) for x in v)
    if isinstance(v, (set, frozenset)):
        try:
            return frozenset(_normalize_value(x, _depth + 1) for x in v)
        except Exception:
            return repr(v)
    return v


def _normalize_str(s: str, depth: int = 0) -> Any:
    """String half of :func:`_normalize_value` (see its docstring)."""
    t = s.strip()
    low = t.lower()
    if low in _NULLISH_STRINGS:
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    if len(t) <= 4096 and t[0] in _LITERAL_LEAD:
        try:
            import ast
            return _normalize_value(ast.literal_eval(t), depth + 1)
        except Exception:
            pass
    return t


def values_equal(a: Any, b: Any) -> bool:
    """True when ``a`` and ``b`` mean the same thing.

    Compares :func:`_normalize_value` output structurally, falling back
    to a ``repr`` comparison for exotic values whose ``__eq__`` refuses
    to produce a bool (numpy-style elementwise comparison, etc.) — and
    to "not equal" if even that blows up. A settings comparison must
    never be the thing that raises.
    """
    try:
        return bool(_normalize_value(a) == _normalize_value(b))
    except Exception:
        pass
    try:
        return repr(a) == repr(b)
    except Exception:
        return False


def resolve_run_dir(ref: Any) -> Path:
    """Turn a run reference into a run-folder :class:`~pathlib.Path`.

    Accepts, in order of preference:

    * a :class:`Run` object (uses its ``dir``),
    * a path (``str`` / ``Path``) to an existing run folder,
    * a run-id — the folder basename, e.g.
      ``"2026-07-23_214737_b66bae6b__mask"`` — resolved under
      :func:`runs_root`,
    * an unambiguous *prefix* of a run-id (``"2026-07-23_2147"``), handy
      from a shell.

    :raises FileNotFoundError: when nothing matches, or when a prefix
        matches more than one run.
    """
    if isinstance(ref, Run):
        return Path(ref.dir)
    if ref is not None:
        try:
            p = Path(ref)
        except TypeError:
            p = None
        if p is not None and p.is_dir():
            return p
        name = str(ref).strip().rstrip("/")
        if name and os.sep not in name:
            root = runs_root()
            cand = root / name
            if cand.is_dir():
                return cand
            try:
                matches = sorted(
                    d for d in root.iterdir()
                    if d.is_dir() and d.name.startswith(name)
                )
            except Exception:
                matches = []
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise FileNotFoundError(
                    f"run id {name!r} is ambiguous — matches {len(matches)} "
                    f"runs ({', '.join(m.name for m in matches[:3])}…)"
                )
    raise FileNotFoundError(f"no such run: {ref!r}")


def _read_run_record(ref: Any) -> Dict[str, Any]:
    """Best-effort read of one run folder.

    Never raises for a *malformed* run — a half-written folder left by a
    crashed pipeline (``status="running"``, no manifest yet) is a real,
    common case and must still diff. Problems are collected into
    ``["errors"]`` instead. A run folder that does not exist at all is a
    caller mistake, and :func:`resolve_run_dir` still raises for it.
    """
    d = resolve_run_dir(ref)
    rec: Dict[str, Any] = {
        "dir": d, "settings": {}, "manifest": {}, "errors": [],
    }

    # -- settings ----------------------------------------------------------
    try:
        rec["settings"] = load_run_settings(d) or {}
    except FileNotFoundError:
        rec["errors"].append("no settings.json / settings.csv in run folder")
    except Exception as e:
        # settings.json exists but is unreadable — try the CSV twin
        # before giving up; they are written together.
        rec["errors"].append(f"settings.json unreadable ({e.__class__.__name__})")
        csv_path = d / "settings.csv"
        if csv_path.exists():
            try:
                rec["settings"] = _read_settings_csv(csv_path) or {}
                rec["errors"].append("fell back to settings.csv")
            except Exception as e2:
                rec["errors"].append(
                    f"settings.csv unreadable ({e2.__class__.__name__})")
    if not isinstance(rec["settings"], dict):
        rec["errors"].append(
            f"settings is {type(rec['settings']).__name__}, not a dict")
        rec["settings"] = {}

    # -- manifest ----------------------------------------------------------
    mp = d / "manifest.json"
    if not mp.exists():
        rec["errors"].append("no manifest.json (run may still be in flight)")
    else:
        try:
            m = json.loads(mp.read_text())
            rec["manifest"] = m if isinstance(m, dict) else {}
            if not isinstance(m, dict):
                rec["errors"].append(
                    f"manifest.json is {type(m).__name__}, not an object")
        except Exception as e:
            rec["errors"].append(
                f"manifest.json unreadable ({e.__class__.__name__})")
    return rec


def _run_meta(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Summarise one run for the diff's ``meta`` block."""
    m = rec["manifest"] or {}
    env = m.get("env") if isinstance(m.get("env"), dict) else {}
    return {
        "run_id":         rec["dir"].name,
        "dir":            str(rec["dir"]),
        "app_key":        m.get("app_key"),
        "status":         m.get("status"),
        "start_utc":      m.get("start_utc"),
        "elapsed_s":      m.get("elapsed_s"),
        "n_settings":     len(rec["settings"]),
        "spacr_version":  env.get("spacr"),
        "errors":         list(rec["errors"]),
    }


def diff_runs(run_a: Any, run_b: Any) -> Dict[str, Any]:
    """Compare two journalled runs and report exactly what changed.

    ``run_a`` / ``run_b`` may each be a run folder (``str`` / ``Path``),
    a run-id or unambiguous id prefix (resolved under :func:`runs_root`),
    or a :class:`Run` object — see :func:`resolve_run_dir`.

    Results are bucketed by *presence* before value, because the settings
    schema drifts between releases and a flat diff of an old run against
    a new one is almost entirely schema noise::

        {
          "changed":   [{"key": k, "a": av, "b": bv}, …],  # THE signal
          "only_in_a": ["key", …],      # schema drift / dropped options
          "only_in_b": ["key", …],      # schema drift / new options
          "same":      int,             # count only, to keep this small
          "env":       [{"key": k, "a": av, "b": bv}, …],  # from manifests
          "meta":      {"a": {...}, "b": {...}, "app_key_differs": bool},
        }

    ``changed`` holds only keys present in *both* runs whose values
    actually differ, sorted by key — those are the knobs someone turned.
    Values are compared structurally, not by ``repr``
    (see :func:`_normalize_value`): ``[1, 2] == [1, 2]``, ``"[1, 2]"``
    (as CSV round-trips it) ``== [1, 2]``, and ``"None" == None``.

    ``env`` diffs ``manifest.json``'s ``env`` snapshot — spaCR version,
    git hash, python, platform, torch / cellpose / numpy versions — which
    is usually where an unexplained behaviour change actually lives.

    Comparing two runs of *different* ``app_key`` (mask vs measure) is
    allowed — sometimes that is exactly the question — but flagged via
    ``meta["app_key_differs"]`` so the caller can warn that the two
    schemas were never meant to line up.

    A missing or corrupt ``settings.json`` / ``manifest.json`` never
    raises: whatever could be read is diffed and the problem is listed in
    ``meta[side]["errors"]``.

    :param run_a: baseline run reference.
    :param run_b: comparison run reference.
    :returns: the diff dict described above (JSON-serialisable as long as
        the settings themselves are).
    :raises FileNotFoundError: only when a run reference resolves to no
        run folder at all.
    """
    ra = _read_run_record(run_a)
    rb = _read_run_record(run_b)
    sa, sb = ra["settings"], rb["settings"]

    changed: List[Dict[str, Any]] = []
    same = 0
    for k in sorted(set(sa) & set(sb)):
        if values_equal(sa[k], sb[k]):
            same += 1
        else:
            changed.append({"key": k, "a": sa[k], "b": sb[k]})

    meta_a, meta_b = _run_meta(ra), _run_meta(rb)
    return {
        "changed":   changed,
        "only_in_a": sorted(set(sa) - set(sb)),
        "only_in_b": sorted(set(sb) - set(sa)),
        "same":      same,
        "env":       _diff_env(ra["manifest"], rb["manifest"]),
        "meta": {
            "a": meta_a,
            "b": meta_b,
            "app_key_differs": (
                meta_a["app_key"] != meta_b["app_key"]
                and meta_a["app_key"] is not None
                and meta_b["app_key"] is not None
            ),
        },
    }


def _diff_env(man_a: Dict[str, Any], man_b: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Diff the ``env`` snapshots of two manifests, sorted by key.

    Keys absent from one manifest are reported with ``None`` on that
    side — an env key that only one run recorded is itself a difference
    worth seeing (a package that wasn't tracked yet).

    But when one side has *no* env snapshot at all — missing or corrupt
    manifest — this returns nothing rather than declaring every package
    on the other side "changed to None". That would be a dozen invented
    differences from one unreadable file; the unreadable file itself is
    already reported in ``meta[side]["errors"]``.
    """
    ea = man_a.get("env") if isinstance(man_a, dict) else None
    eb = man_b.get("env") if isinstance(man_b, dict) else None
    ea = ea if isinstance(ea, dict) else {}
    eb = eb if isinstance(eb, dict) else {}
    if not ea or not eb:
        return []
    out: List[Dict[str, Any]] = []
    for k in sorted(set(ea) | set(eb)):
        av, bv = ea.get(k), eb.get(k)
        if not values_equal(av, bv):
            out.append({"key": k, "a": av, "b": bv})
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_value(v: Any, width: int = 46) -> str:
    """One-line, length-capped rendering of a settings value."""
    s = "—" if v is None else (v if isinstance(v, str) else repr(v))
    s = " ".join(str(s).split())
    if width and len(s) > width:
        s = s[: width - 1] + "…"
    return s


def _render_change_pair(av: Any, bv: Any, width: int = 46) -> tuple:
    """Render both sides of a change so the *difference* stays visible.

    Two long values usually share a long head — ``src`` is the classic
    case, where both runs point deep into the same tree and differ in
    one path component. Truncating each side independently then prints
    the same 46 characters twice and tells the reader nothing. So when
    either side overflows, the common prefix is elided instead.
    """
    sa, sb = _render_value(av, 0), _render_value(bv, 0)
    if len(sa) > width or len(sb) > width:
        n = len(os.path.commonprefix([sa, sb]))
        if n > 8:
            sa, sb = "…" + sa[n:], "…" + sb[n:]
    return _render_value(sa, width), _render_value(sb, width)


def _render_elapsed(v: Any) -> str:
    try:
        return f"{float(v):.1f}s"
    except (TypeError, ValueError):
        return "—"


def _drift_names(keys: List[str], limit: int) -> str:
    """``"a, b, c, … (+37 more)"`` — never the whole list."""
    if not keys:
        return ""
    limit = max(int(limit), 0)
    head = ", ".join(keys[:limit])
    rest = len(keys) - limit
    if not head:
        return f"({rest} keys, not shown)"
    return f"{head}, … (+{rest} more)" if rest > 0 else head


def format_run_diff(diff: Dict[str, Any], max_drift_names: int = 6) -> str:
    """Render :func:`diff_runs` output as a readable console report.

    Ordering is deliberate: changed settings first (the signal), then the
    environment, then schema drift reduced to a **one-line summary** plus
    a handful of names. The drifted keys are never dumped in full — on a
    real cross-release pair that is ~200 lines of noise that buries the
    six settings the user actually changed.

    :param diff: the dict returned by :func:`diff_runs`.
    :param max_drift_names: how many drifted key names to name before
        collapsing the rest into ``(+N more)``.
    :returns: a multi-line report (no trailing newline).
    """
    meta = diff.get("meta") or {}
    a, b = meta.get("a") or {}, meta.get("b") or {}
    lines: List[str] = ["Run diff"]
    for tag, m in (("A", a), ("B", b)):
        lines.append(f"  {tag}  {m.get('run_id', '?')}")
        lines.append(
            f"     {m.get('app_key') or '?'} · {m.get('status') or '?'}"
            f" · {m.get('start_utc') or '?'}"
            f" · {_render_elapsed(m.get('elapsed_s'))}"
            f" · {m.get('n_settings', 0)} settings"
            + (f" · spacr {m['spacr_version']}" if m.get("spacr_version") else "")
        )
        for err in m.get("errors") or []:
            lines.append(f"     ! {err}")
    if meta.get("app_key_differs"):
        lines.append(
            f"  ! different pipelines ({a.get('app_key')} vs {b.get('app_key')})"
            " — their settings schemas were never meant to line up"
        )

    # -- the signal --------------------------------------------------------
    changed = diff.get("changed") or []
    shared = len(changed) + int(diff.get("same") or 0)
    lines.append("")
    if changed:
        lines.append(f"Settings changed ({len(changed)} of {shared} shared keys)")
        width = min(max((len(c["key"]) for c in changed), default=0), 34)
        for c in changed:
            av, bv = _render_change_pair(c["a"], c["b"])
            lines.append(f"  {c['key']:<{width}}  {av}  →  {bv}")
    else:
        lines.append(f"Settings changed (0 of {shared} shared keys) — identical")

    # -- environment -------------------------------------------------------
    env = diff.get("env") or []
    lines.append("")
    if env:
        lines.append(f"Environment changed ({len(env)})")
        width = min(max(len(e["key"]) for e in env), 34)
        for e in env:
            lines.append(
                f"  {e['key']:<{width}}  {_render_value(e['a'], 28)}"
                f"  →  {_render_value(e['b'], 28)}"
            )
    else:
        lines.append("Environment changed (0) — same versions on both runs")

    # -- schema drift, summarised (never enumerated) -----------------------
    only_a = diff.get("only_in_a") or []
    only_b = diff.get("only_in_b") or []
    lines.append("")
    if only_a or only_b:
        since = a.get("spacr_version")
        vb = b.get("spacr_version")
        ver = ""
        if since and vb and since != vb:
            ver = f" (spacr {since} → {vb})"
        elif since:
            ver = f" since {since}"
        lines.append(
            f"Schema drift: +{len(only_b)} keys added, "
            f"-{len(only_a)} removed{ver}"
        )
        if only_b:
            lines.append(f"  added:   {_drift_names(only_b, max_drift_names)}")
        if only_a:
            lines.append(f"  removed: {_drift_names(only_a, max_drift_names)}")
    else:
        lines.append("Schema drift: none — both runs share the same keys")
    return "\n".join(lines)
