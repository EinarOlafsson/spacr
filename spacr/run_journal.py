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
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

LOG = logging.getLogger("spacr.run_journal")


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
    d = runs_root() / f"{ts}_{tag}__{app_key or 'unknown'}"
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


def _env_snapshot() -> Dict[str, Any]:
    """Capture host + package versions worth reproducing against."""
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
    }


def hash_file(path: Path, chunk_size: int = 1 << 20) -> Optional[str]:
    """Return the sha256 (first 16 hex chars) of a file, or None on error.

    Used to fingerprint model checkpoints so a mask can be traced
    back to the exact weights that produced it.
    """
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except Exception:
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
    """
    app_key: str
    settings: Dict[str, Any]
    dir: Path
    start_ts: float = field(default_factory=time.time)
    end_ts: Optional[float] = None
    status: str = "running"
    model_hashes: Dict[str, str] = field(default_factory=dict)
    stdout_path: Optional[Path] = None
    error_traceback: str = ""

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
        except Exception:
            pass

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
            return dst
        except Exception:
            return None

    def set_status(self, status: str) -> None:
        """Explicitly stamp ``status`` (``success`` / ``failed`` / …)."""
        self.status = status

    # -- private -----------------------------------------------------------
    def _write_manifest(self) -> None:
        elapsed = None
        if self.end_ts is not None:
            elapsed = round(self.end_ts - self.start_ts, 3)
        manifest = {
            "app_key":       self.app_key,
            "start_utc":     datetime.fromtimestamp(
                self.start_ts, tz=timezone.utc).isoformat(),
            "end_utc":       (datetime.fromtimestamp(
                self.end_ts, tz=timezone.utc).isoformat()
                if self.end_ts else None),
            "elapsed_s":     elapsed,
            "status":        self.status,
            "env":           _env_snapshot(),
            "model_hashes":  self.model_hashes,
            "n_settings":    len(self.settings),
            "traceback":     self.error_traceback or None,
        }
        (self.dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str)
        )

    def _write_settings(self) -> None:
        # Machine-friendly JSON (source of truth)
        (self.dir / "settings.json").write_text(
            json.dumps(self.settings, indent=2, default=str)
        )
        # Human-friendly CSV (Key,Value — spacr.utils.load_settings compatible)
        with open(self.dir / "settings.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Key", "Value"])
            for k, v in self.settings.items():
                w.writerow([k, "" if v is None else str(v)])

    def _snapshot_log_tail(self, n: int = 200) -> None:
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

_ACTIVE_RUN: Optional["Run"] = None


def current_run() -> Optional["Run"]:
    """Return the :class:`Run` currently open in this process, or None.

    Useful for pipeline internals (Cellpose model loaders, etc.) that
    want to record a model checkpoint hash without every caller
    needing to plumb the :class:`Run` object through.

    Not thread-local — spaCR pipelines run one at a time; if that
    changes we'll switch to :mod:`threading.local`.
    """
    return _ACTIVE_RUN


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
    global _ACTIVE_RUN
    run = Run(app_key=app_key, settings=dict(settings or {}),
                dir=_new_run_dir(app_key))
    run._write_settings()
    LOG.info("run opened → %s", run.dir)
    prev_active = _ACTIVE_RUN
    _ACTIVE_RUN = run
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
        _ACTIVE_RUN = prev_active
        run.end_ts = time.time()
        run._snapshot_log_tail()
        run._write_manifest()
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


def load_run_settings(run_dir: Path) -> Dict[str, Any]:
    """Read a run's ``settings.json`` (falling back to settings.csv)."""
    run_dir = Path(run_dir)
    j = run_dir / "settings.json"
    if j.exists():
        return json.loads(j.read_text())
    c = run_dir / "settings.csv"
    if not c.exists():
        raise FileNotFoundError(f"no settings in {run_dir}")
    out: Dict[str, Any] = {}
    with open(c) as f:
        for row in csv.reader(f):
            if row and row[0] and row[0] != "Key":
                out[row[0]] = row[1] if len(row) > 1 else ""
    return out
