"""
Run journal — reproducibility record for every pipeline invocation.

Every time a spaCR pipeline runs (mask / measure / classify / …),
:func:`open_run` writes a timestamped folder under ``~/.spacr/runs/``
containing everything a reviewer needs to reproduce the result:

::

    ~/.spacr/runs/2026-07-23_143507_ab12cd34__mask/
        settings.csv          # exact settings dict, Key,Value CSV
        settings.json         # same, JSON (source of truth for machines)
        manifest.json         # spaCR version, git hash, python, packages,
                              # torch / cuda / cellpose, start time,
                              # end time, elapsed, exit status, model hashes
        log.txt               # tail of ~/.spacr/logs/spacr.log for the run
        outputs/              # optional — any pipeline-emitted artifacts
                              # (masks, DBs, CSVs, plots) copied in

Public API::

    from spacr.run_journal import open_run

    with open_run("mask", settings) as run:
        preprocess_generate_masks(settings)
        run.attach_output(Path("/path/to/mask.tif"))
        run.set_status("success")

The context manager records start / end timestamps and yields the
:class:`Run` object, whose ``dir`` attribute is the run folder. When
something raises it stamps ``"status": "failed"`` (plus the traceback)
into ``manifest.json`` and re-raises; no marker file is written.

Consumers of the journal:

* ``spacr repro <run-folder>`` — replays the run (see
  :mod:`spacr.cli_repro`).
* AI Console → "File as issue" — includes the last run's manifest
  when present so bug reports are self-contained.
* Home screen "Recent runs" list — enumerated from
  :func:`recent_runs` newest first.

Beside the runs, the journal keeps two records for blinded work.
:func:`start_blinding` writes a blinding key (coded names and a shuffled
order) to ``~/.spacr/blinding`` and :func:`unblind` logs who opened it and
when. :func:`lock_analysis` freezes an analysis plan, hashed and timestamped,
in ``~/.spacr/analysis_locks`` (settings of one or more pipelines, Gate
Editor gating strategies, models and files); every later run of a locked
pipeline on its ``src`` is checked against it by
:func:`check_analysis_lock`, every model the run records is checked as it is
recorded, and the
verdict is written into that run's ``manifest.json`` under
``analysis_lock``, with any difference also listed in
``provenance_warnings``.
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

from .macro import begin_recording, finish_recording
from .logging_util import _spacr_home

LOG = logging.getLogger("spacr.run_journal")

MANIFEST_SCHEMA_VERSION = 3
"""Current on-disk reproducibility-manifest schema."""

#: Ceiling on files inventoried under any one setting-derived root. A
#: source folder can hold a million PNG crops, and the baseline is taken on
#: EVERY run whether or not hashing was asked for. Hitting it is recorded in
#: `provenance_warnings` rather than passed over -- a truncated inventory
#: that says nothing is a manifest claiming completeness it does not have.
INVENTORY_BUDGET = 200_000

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



def runs_root() -> Path:
    """Return ``~/.spacr/runs``; created on first access."""
    p = _spacr_home() / "runs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def delete_runs(directories: Iterable[Any]) -> Tuple[int, List[str]]:
    """Delete journalled run folders. Returns ``(deleted, refused)``.

    THIS REMOVES FILES, so every guard below is load-bearing rather than
    defensive habit:

    * A path is deleted only if, once resolved, it is strictly INSIDE
      :func:`runs_root`. A record's ``dir`` is read from a manifest on
      disk, and a manifest is a file a user can edit; ``../../..`` in one
      must not reach anything. Resolving first is what makes the check
      real -- comparing unresolved strings passes for a symlink that
      points anywhere.
    * ``runs_root()`` ITSELF is refused. "Delete everything" is a loop
      over children, never a removal of the root, so a caller that
      computes an empty selection cannot take the journal with it.
    * The run OPEN ON THIS THREAD is refused: deleting the folder a run
      is still writing into leaves it failing on its next write with an
      error that names nothing.
    * A path that is not a directory is refused rather than unlinked.

    Refusals are RETURNED, not raised. Deleting fifty runs where one is
    live should delete forty-nine and say which one it kept -- an
    exception at that point has already deleted an unknown number and
    tells the caller nothing about which.

    :param directories: run folder paths, as in a record's ``"dir"``.
    :returns: how many were removed, and a message per refusal.
    """
    import shutil

    root = runs_root().resolve()
    live = ""
    try:
        run = current_run()
        if run is not None and getattr(run, "dir", None):
            live = str(Path(run.dir).resolve())
    except Exception:                                       # noqa: BLE001
        LOG.debug("Could not identify the running run", exc_info=True)

    deleted, refused = 0, []
    for raw in directories:
        try:
            target = Path(str(raw)).resolve()
        except Exception:                                   # noqa: BLE001
            refused.append(f"{raw}: not a usable path")
            continue
        if target == root:
            refused.append(f"{target}: that is the run journal itself")
            continue
        if root not in target.parents:
            refused.append(f"{target}: outside {root}")
            continue
        if not target.is_dir():
            refused.append(f"{target}: not a run folder")
            continue
        if live and str(target) == live:
            refused.append(f"{target.name}: still running")
            continue
        try:
            shutil.rmtree(target)
            deleted += 1
        except Exception as error:                          # noqa: BLE001
            refused.append(f"{target.name}: {type(error).__name__}: {error}")
    return deleted, refused


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



@lru_cache(maxsize=None)
def _pkg_version(name: str) -> str:
    """Return an installed distribution version or ``"not installed"``.

    Memoised for the life of the process, like :func:`_installed_packages`:
    every run records the same dozen versions, and each lookup reads the
    distribution's metadata from disk.
    """
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
    Each distribution's metadata file is parsed once for both its name and
    its version (``dist.version`` would parse it a second time); a
    metadata without a ``Version`` falls back to ``dist.version``.
    Duplicate normalized names are collapsed deterministically.
    """
    packages: Dict[str, str] = {}
    try:
        for dist in importlib.metadata.distributions():
            metadata = dist.metadata
            name = str(metadata.get("Name") or "").strip()
            if name:
                version = metadata.get("Version") or dist.version
                packages[name.lower().replace("_", "-")] = str(
                    version or "unknown"
                )
    except Exception as exc:
        LOG.warning("Could not enumerate installed packages: %s", exc)
    return dict(sorted(packages.items()))


_ENV_VERSIONS = (
    ("torch", "torch"), ("torchvision", "torchvision"),
    ("cellpose", "cellpose"), ("pyside6", "PySide6"), ("numpy", "numpy"),
    ("scipy", "scipy"), ("pandas", "pandas"),
    ("scikit_image", "scikit-image"), ("scikit_learn", "scikit-learn"),
)


def _env_snapshot() -> Dict[str, Any]:
    """Capture host and complete package versions for reproduction."""
    snapshot: Dict[str, Any] = {
        "spacr":         _pkg_version("spacr"),
        "spacr_git":     _git_hash(),
        "python":        sys.version.split()[0],
        "platform":      platform.platform(),
    }
    for key, distribution in _ENV_VERSIONS:
        snapshot[key] = _pkg_version(distribution)
    snapshot["packages"] = _installed_packages()
    return snapshot


_ENV_LOCK_FOLDER = "environment"
_PIP_LOCK_NAME = "requirements-lock.txt"
_CONDA_LOCK_NAME = "conda-explicit.txt"
_ENV_LOCK_TIMEOUT_S = 120


def _environment_digest(packages: Dict[str, str]) -> str:
    """Return the SHA-256 that names one Python environment's lockfiles."""
    return _json_digest({
        "python": sys.version,
        "executable": os.path.abspath(sys.executable),
        "prefix": os.path.abspath(sys.prefix),
        "packages": packages,
    })


def _run_quiet(command: List[str]) -> Optional[str]:
    """Run ``command`` and return its standard output, or ``None`` on failure."""
    try:
        done = subprocess.run(
            command, capture_output=True, text=True,
            timeout=_ENV_LOCK_TIMEOUT_S, check=False,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        LOG.debug("environment lock command %s failed: %s", command, exc)
        return None
    if done.returncode != 0 or not done.stdout.strip():
        LOG.debug("environment lock command %s exited %s",
                  command, done.returncode)
        return None
    return done.stdout


def _pip_lock_text(packages: Dict[str, str]) -> str:
    """Return a ``pip freeze`` of this interpreter's environment.

    Falls back to ``name==version`` lines read from the installed
    distributions' metadata when pip cannot be run, so a lockfile is always
    written.
    """
    frozen = _run_quiet([sys.executable, "-m", "pip", "freeze",
                         "--all", "--disable-pip-version-check"])
    if frozen:
        return frozen if frozen.endswith("\n") else frozen + "\n"
    lines = ["# pip freeze was unavailable; read from package metadata."]
    lines += [f"{name}=={version}" for name, version in packages.items()]
    return "\n".join(lines) + "\n"


def _conda_lock_text() -> Optional[str]:
    """Return ``conda list --explicit`` for this environment, inside conda.

    :returns: the explicit spec list, or ``None`` outside a conda
        environment or when conda cannot be run.
    """
    prefix = os.path.abspath(sys.prefix)
    if not os.path.isdir(os.path.join(prefix, "conda-meta")):
        return None
    conda = (os.environ.get("CONDA_EXE", "").strip()
             or shutil.which("conda") or shutil.which("mamba")
             or shutil.which("micromamba"))
    if not conda:
        return None
    return _run_quiet([conda, "list", "--explicit", "--prefix", prefix])


_ENV_LOCK_MEMO: Dict[str, Dict[str, Optional[str]]] = {}
_ENV_LOCK_MEMO_GUARD = threading.Lock()


def _environment_lock_texts(digest: str,
                            packages: Dict[str, str]) -> Dict[str, Optional[str]]:
    """Return the pip and conda lockfile texts for one environment digest.

    Built at most once per digest: first from this process's memory, then
    from the shared store beside the run journal (``env_locks/<digest>``),
    and only when neither has it by running pip and conda, whose output is
    then saved to that store for every later run and process.
    """
    with _ENV_LOCK_MEMO_GUARD:
        cached = _ENV_LOCK_MEMO.get(digest)
        if cached is not None:
            return cached
        store = runs_root().parent / "env_locks" / digest[:32]
        pip_file = store / _PIP_LOCK_NAME
        conda_file = store / _CONDA_LOCK_NAME
        texts: Dict[str, Optional[str]]
        if pip_file.is_file():
            texts = {
                "pip": pip_file.read_text(encoding="utf-8"),
                "conda": (conda_file.read_text(encoding="utf-8")
                          if conda_file.is_file() else None),
            }
        else:
            texts = {"pip": _pip_lock_text(packages),
                     "conda": _conda_lock_text()}
            try:
                store.mkdir(parents=True, exist_ok=True)
                if texts["conda"]:
                    _atomic_write_text(conda_file, texts["conda"])
                _atomic_write_text(pip_file, texts["pip"] or "")
            except OSError as exc:
                LOG.warning("Could not save the environment lock in %s: %s",
                            store, exc)
        _ENV_LOCK_MEMO[digest] = texts
        return texts


def _write_environment_lock(run_dir: Path,
                            packages: Dict[str, str]) -> Dict[str, Any]:
    """Write this environment's lockfiles into ``run_dir/environment``.

    The pip freeze, and the conda explicit list inside a conda environment,
    are produced once per environment digest (see
    :func:`_environment_lock_texts`); each run only writes the small text
    files into its own folder.

    :returns: the manifest's ``environment_lock`` record: ``sha256`` (the
        environment digest), ``pip`` and ``conda`` (paths relative to the run
        folder, ``conda`` ``None`` outside conda).
    """
    digest = _environment_digest(packages)
    texts = _environment_lock_texts(digest, packages)
    folder = run_dir / _ENV_LOCK_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    record: Dict[str, Any] = {"sha256": digest, "pip": None, "conda": None}
    for key, name in (("pip", _PIP_LOCK_NAME), ("conda", _CONDA_LOCK_NAME)):
        text = texts.get(key)
        if text:
            _atomic_write_text(folder / name, text)
            record[key] = f"{_ENV_LOCK_FOLDER}/{name}"
    return record


def _warm_env_snapshot() -> None:
    """Read the package versions a run records, so the first run need not.

    Enumerating every installed distribution reads hundreds of metadata
    files, the largest part of opening a run's manifest. Both readers are
    memoised, so calling this from a background thread once the window is
    up moves that cost off the first run. The environment's pip freeze and
    conda explicit list are prepared here too, for the same reason.
    """
    _pkg_version("spacr")
    for _key, distribution in _ENV_VERSIONS:
        _pkg_version(distribution)
    packages = _installed_packages()
    try:
        _environment_lock_texts(_environment_digest(packages), packages)
    except Exception as exc:
        LOG.debug("could not prepare the environment lock: %s", exc)


def hash_file(
    path: Path,
    chunk_size: int = 1 << 20,
    *,
    full: bool = False,
) -> Optional[str]:
    """Return a file's SHA-256 — 16 hex chars, 64 if ``full`` — or ``None``.

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

    :param settings: settings dict; every nested key whose name contains
        ``seed``, ``random_state`` or ``random_seed`` is collected.
    :returns: a dict with ``declared`` (the collected seed settings),
        ``python_hash_seed`` (``PYTHONHASHSEED``, or ``None`` when unset),
        ``python_random_state_sha256``, ``numpy_random_state_sha256`` and
        ``torch_initial_seed`` — the last two ``None`` when the library is
        not already imported. A probe that raises contributes
        ``numpy_random_state_error`` / ``torch_seed_error`` instead of its
        value key.
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
        """Return whether ``candidate`` resolves to or below an excluded root.

        A path that cannot be resolved is retained so one hostile entry does
        not abort or silently empty the rest of the file walk.
        """
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
    :ivar status: ``"running"`` / ``"success"`` / ``"failed"`` /
        ``"cancelled"``. The last is a run the user stopped, which is not a
        run that broke, and the two want different things done next.
    :ivar model_hashes: dict of ``{human-name: "filename:sha256-16"}``.
        Populated by callers via :meth:`record_model`.
    :ivar model_files: full SHA-256, size, and path records for models.
    :ivar input_hashes: per-file full SHA-256 input provenance.
    :ivar output_hashes: per-file full SHA-256 output provenance.
    :ivar seeds: declared seeds and runtime RNG-state identifiers.
    :ivar provenance_warnings: non-fatal path/hash failures retained in the
        manifest instead of being silently discarded.
    :ivar run_warnings: distinct warning lines emitted by the pipeline.
    :ivar environment: host, spaCR, Git, and installed-package versions.
    :ivar stages: consolidated FlowView lifecycle records in execution order.
    :ivar stdout_path: captured standard-output log path, when one is attached.
    :ivar error_traceback: formatted exception traceback for failed or
        cancelled runs; empty for successful runs.
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
    environment_lock: Dict[str, Any] = field(default_factory=dict, init=False)
    stages: List[Dict[str, Any]] = field(default_factory=list)
    stdout_path: Optional[Path] = None
    error_traceback: str = ""
    _path_candidates: List[Tuple[str, Path, bool]] = field(
        default_factory=list, repr=False,
    )
    _baseline: Dict[str, Tuple[int, int]] = field(
        default_factory=dict, repr=False,
    )
    _start_cpu_s: float = field(default_factory=time.process_time, repr=False)
    _ledgers: List[Dict[str, Any]] = field(default_factory=list, repr=False)

    def _note_ledger(self, ledger: Any) -> None:
        """Remember a finished item ledger's counts for the run's summary.

        Called when a :class:`spacr.errors.RunLedger` is finalized while this
        run is open; the counts feed the run-finished notification. Never
        raises: a summary line must not replace a result.

        :param ledger: the finalized ledger.
        """
        try:
            self._ledgers.append({
                "name": str(getattr(ledger, "name", "run")),
                "attempted": int(ledger.n_attempted),
                "succeeded": int(ledger.n_succeeded),
                "failed": int(ledger.n_failed),
            })
        except Exception:
            LOG.debug("could not note an item ledger", exc_info=True)

    def record_model(self, name: str, checkpoint_path: Any) -> None:
        """Fingerprint ``checkpoint_path`` and remember it under ``name``.

        :param name: human-readable key under which to record the model.
        :param checkpoint_path: model checkpoint file to fingerprint.

        Records ``"<filename>:<digest>"`` in ``model_hashes``. An unreadable
        checkpoint is only logged and leaves no entry at all; any other
        failure is also appended to ``provenance_warnings``, so it reaches
        ``manifest.json`` — model logging must never itself fail a run.
        When an analysis lock applies to the run, the model is also checked
        against it (see :func:`lock_analysis`), and the run's verdict is
        updated.
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
            _check_model_for_run(self, name, p)
        except Exception as exc:
            warning = f"model {name!r} could not be recorded: {exc}"
            self.provenance_warnings.append(warning)
            LOG.warning(warning)

    def record_input(self, path: Any, *, setting_key: str = "") -> None:
        """Hash a file or directory as an explicit run input.

        A no-op unless :meth:`hashing_enabled` is true, i.e. unless the
        settings enable ``hash_inputs`` — a default run records nothing here.
        Directories are recorded as one full SHA-256 record per regular file.
        Unreadable files are reported in ``provenance_warnings`` and logs.

        :param path: input file or directory.
        :param setting_key: optional setting that referred to ``path``.
        """
        if not self.hashing_enabled():
            return
        self._record_tree(
            Path(path), self.input_hashes, setting_key=setting_key,
        )

    def record_output(self, path: Any, *, setting_key: str = "") -> None:
        """Hash a file or directory as an explicit run output.

        Like :meth:`record_input`, a no-op unless :meth:`hashing_enabled` is
        true, i.e. unless the settings enable ``hash_inputs``.

        :param path: output file or directory.
        :param setting_key: optional setting that referred to ``path``.
        """
        if not self.hashing_enabled():
            return
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
        """Explicitly stamp ``status`` (``success`` / ``failed`` / …).

        :param status: lifecycle state to store on the run.
        """
        self.status = status

    def record_warning(self, message: Any) -> None:
        """Retain a distinct warning for the run-history dashboard.

        :param message: warning text captured from pipeline stdout/stderr or
            supplied directly by pipeline code.
        """
        text = str(message or "").strip()
        if text and text not in self.run_warnings:
            if len(self.run_warnings) < 500:
                self.run_warnings.append(text)

    def _record_stage(
        self,
        stage_id: Any,
        *,
        label: Any = None,
        state: Any = None,
        started_at: Any = None,
        ended_at: Any = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Merge one pipeline-stage observation into the run record.

        Instrumentation calls this at the same boundary used for its live
        graph, so ``manifest.json`` and the graph share exact timestamps
        rather than trying to reconcile two clocks after the run. Repeated
        calls update the existing stage in place and preserve first-seen
        order. Invalid diagnostics are ignored: provenance must never replace
        a scientific result or exception.

        :param stage_id: stable stage identifier.
        :param label: optional human-readable stage label.
        :param state: optional lifecycle state such as ``running`` or ``done``.
        :param started_at: optional Unix epoch start timestamp.
        :param ended_at: optional Unix epoch terminal timestamp.
        :param metrics: scalar counts or measurements to merge.
        """
        try:
            identifier = str(stage_id).strip()
            if not identifier:
                return
            stage = next(
                (item for item in self.stages if item.get("id") == identifier),
                None,
            )
            if stage is None:
                stage = {
                    "id": identifier,
                    "label": str(label) if label is not None else identifier,
                    "state": "pending",
                    "started_at": None,
                    "ended_at": None,
                    "duration_s": None,
                    "metrics": {},
                }
                self.stages.append(stage)
            elif label is not None:
                stage["label"] = str(label)
            if state is not None:
                stage["state"] = str(state)
            if started_at is not None:
                stage["started_at"] = float(started_at)
            if ended_at is not None:
                stage["ended_at"] = float(ended_at)
            if metrics:
                stage_metrics = stage.setdefault("metrics", {})
                for name, value in metrics.items():
                    stage_metrics[str(name)] = value
            start = stage.get("started_at")
            end = stage.get("ended_at")
            stage["duration_s"] = (
                float(end) - float(start)
                if start is not None and end is not None
                else None
            )
        except BaseException:
            try:
                LOG.debug("could not record pipeline stage evidence", exc_info=True)
            except BaseException:
                pass

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

    def hashing_enabled(self) -> bool:
        """Whether to hash inputs and outputs for this run.

        Off unless the settings say otherwise. Hashing every file under
        every path-valued setting is proportional to the DATA, not to the
        run: on a plate of raw images it is minutes of reading before the
        first mask is made, and it happens whether or not anybody will ever
        compare the digests.

        Read from the settings dict, never from QSettings. A
        `from PySide6 import` in a pipeline module makes the package
        unimportable on a cluster (see the architecture notes), so the GUI
        reads the preference and passes it down as an ordinary setting, and
        a headless caller sets the same key.
        """
        return bool(self.settings.get("hash_inputs", False))

    @staticmethod
    def _inventory_root(key: str, path: Path, output_only: bool) -> Optional[Path]:
        """Which directory this candidate's inventory should cover.

        THE BASELINE EXISTS TO ANSWER ONE QUESTION -- which files did this
        run CREATE -- so the inventory only needs to cover somewhere the run
        might write.

        ``root = path if path.is_dir() else path.parent`` answered that too
        generously and cost a directory walk per input file. A setting
        naming ONE model checkpoint inventoried the whole folder it sits in:
        measured, ``model_path`` pointing at a file among 20,000 others
        stat'ed 20,001 paths, of which 20,000 could not possibly change,
        because ``model_path`` is an input and the run writes nothing beside
        it. On the machine this was found on, a file candidate under ``/tmp``
        walked 475,250 paths and took 44 s cold.

        AND IT IS PAID BY EVERY RUN, not only ones that opted into hashing --
        see the caller's docstring.

        :returns: the directory to walk, or ``None`` when the candidate is a
            file the run cannot write beside, in which case that one file is
            the whole inventory.
        """
        if path.is_dir():
            return path
        if output_only or _is_output_key(key):
            return path.parent
        return None

    def _capture_initial_provenance(self) -> None:
        """Discover path-valued inputs and retain a before-run inventory.

        The inventory is taken whether or not hashing is on: it is a cheap
        stat() per file and it is what lets the final pass know which files
        the run CREATED. Only the hashing is skipped. That is exactly why
        :meth:`_inventory_root` matters -- a walk here is charged to every
        run there is.
        """
        self.seeds = extract_seeds(self.settings)
        self._path_candidates = _setting_path_candidates(self.settings)
        seen_files: Set[str] = set()
        for key, path, output_only in self._path_candidates:
            if path.exists() and not output_only:
                self.record_input(path, setting_key=key)
            root = self._inventory_root(key, path, output_only)
            if root is None:
                signature = _inventory_signature(path)
                if signature is not None:
                    self._baseline[str(path)] = signature
                    seen_files.add(str(path))
                continue
            if not root.exists():
                continue
            for file_path in self._bounded_walk(root):
                file_key = str(file_path)
                if file_key in seen_files:
                    continue
                seen_files.add(file_key)
                signature = _inventory_signature(file_path)
                if signature is not None:
                    self._baseline[file_key] = signature

    def _bounded_walk(self, root: Path) -> Iterator[Path]:
        """``_iter_files`` with a ceiling, and a warning when it is hit.

        The backstop for the case :meth:`_inventory_root` cannot rule out: a
        genuine source folder holding millions of crops. Truncating in
        silence would make the manifest claim a complete inventory it does
        not have, so the ceiling is recorded where the manifest carries it.
        """
        for count, file_path in enumerate(
                _iter_files(root, (self.dir, runs_root()))):
            if count >= INVENTORY_BUDGET:
                warning = (f"provenance inventory of {root} stopped at "
                           f"{INVENTORY_BUDGET} files; it is not complete")
                if warning not in self.provenance_warnings:
                    self.provenance_warnings.append(warning)
                    LOG.warning(warning)
                return
            yield file_path

    def _capture_final_provenance(self) -> None:
        """Hash files created or modified under setting-derived roots."""
        if not self.hashing_enabled():
            return
        seen_files: Set[str] = set()
        for key, path, _output_only in self._path_candidates:
            root = self._inventory_root(key, path, _output_only)
            if root is None:
                signature = _inventory_signature(path)
                if (signature is not None
                        and self._baseline.get(str(path)) != signature):
                    record = _file_record(path)
                    if record is not None:
                        record["setting_keys"] = [key]
                        self.output_hashes[str(path)] = record
                continue
            if not root.exists():
                continue
            for file_path in self._bounded_walk(root):
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
            "input_hashing": "on" if self.hashing_enabled() else "skipped",
            "app_key":       self.app_key,
            "start_utc":     datetime.fromtimestamp(
                self.start_ts, tz=timezone.utc).isoformat(),
            "end_utc":       (datetime.fromtimestamp(
                self.end_ts, tz=timezone.utc).isoformat()
                if self.end_ts else None),
            "elapsed_s":     elapsed,
            "status":        self.status,
            "env":           self.environment,
            "environment_lock": self.environment_lock or None,
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
            "stages":         self.stages,
            "performance":    performance,
            "n_settings":    len(self.settings),
            "traceback":     self.error_traceback or None,
        }
        lock = getattr(self, "_analysis_lock", None)
        if lock:
            manifest["analysis_lock"] = lock
        _atomic_write_text(
            self.dir / "manifest.json",
            json.dumps(manifest, indent=2, default=str, sort_keys=True),
        )

    def _write_settings(self) -> None:
        """Write exact machine- and human-readable settings snapshots."""
        _atomic_write_text(
            self.dir / "settings.json",
            json.dumps(self.settings, indent=2, default=str, sort_keys=True),
        )
        with open(self.dir / "settings.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Key", "Value"])
            for k, v in self.settings.items():
                w.writerow([k, "" if v is None else str(v)])

    def _snapshot_log_tail(self, n: int = 200) -> None:
        """Copy the application tail and append this run's stage evidence."""
        lines: List[str] = []
        try:
            from .logging_util import log_path
            src = log_path()
            if src.exists():
                with open(src, encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
        except Exception as exc:
            LOG.warning("could not copy the last %d log lines into %s (%s)",
                        n, self.dir, exc)
        try:
            stage_lines = []
            for stage in self.stages:
                metrics = json.dumps(
                    stage.get("metrics") or {},
                    default=str,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                stage_lines.append(
                    "FlowView stage "
                    f"{stage.get('id')} state={stage.get('state')} "
                    f"started_at={stage.get('started_at')} "
                    f"ended_at={stage.get('ended_at')} "
                    f"duration_s={stage.get('duration_s')} metrics={metrics}\n"
                )
            if not lines and not stage_lines:
                return
            content = "".join(lines[-n:])
            if content and not content.endswith("\n"):
                content += "\n"
            (self.dir / "log.txt").write_text(
                content + "".join(stage_lines), encoding="utf-8",
            )
        except Exception as exc:
            LOG.warning(
                "could not write run log snapshot into %s (%s)", self.dir, exc
            )



_RUN_LOCAL = threading.local()


def current_run() -> Optional["Run"]:
    """Return the :class:`Run` currently open on this thread, or ``None``.

    Useful for pipeline internals (Cellpose model loaders, etc.) that
    want to record a model checkpoint hash without every caller
    needing to plumb the :class:`Run` object through.

    State is thread-local because the batch runner and database browser can
    execute independent workers concurrently.
    """
    return getattr(_RUN_LOCAL, "active", None)


def _raise_if_incomplete(run: Run) -> None:
    """Refuse success after a finalized item ledger reported partial output."""
    if run.status not in ("running", "success"):
        return
    failures = [ledger for ledger in run._ledgers if ledger["failed"] > 0]
    if failures:
        from .errors import PartialRunError

        summary = "; ".join(
            f"{ledger['name']}: {ledger['failed']} of "
            f"{ledger['attempted']} items failed" for ledger in failures)
        raise PartialRunError(
            f"Run incomplete: {summary}. Partial artifacts were retained.")


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
    try:
        run.environment_lock = _write_environment_lock(
            run.dir, run.environment.get("packages") or {})
    except Exception as exc:
        warning = f"environment lockfile not written: {exc}"
        run.provenance_warnings.append(warning)
        LOG.warning(warning)
    _check_lock_for_run(run)
    run._write_settings()
    run._capture_initial_provenance()
    run._write_manifest()
    LOG.info("run opened → %s", run.dir)
    macro = begin_recording(app_key, run.settings, run_dir=run.dir)
    prev_active = current_run()
    _RUN_LOCAL.active = run
    try:
        yield run
        _raise_if_incomplete(run)
        if run.status == "running":
            run.status = "success"
    except BaseException as e:
        import traceback as _tb
        run.status = ("cancelled" if type(e).__name__ == "PipelineCancelled"
                      else "failed")
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
            LOG.exception("Could not finalize run manifest in %s", run.dir)
        try:
            from .workspace import save_for_run
            save_for_run(run.dir, run.settings, app_key=run.app_key)
        except Exception:
            LOG.exception("Could not save the workspace for %s", run.dir)
        finish_recording(macro, status=run.status, settings=run.settings)
        LOG.info("run closed [%s] in %.1fs → %s",
                  run.status, run.end_ts - run.start_ts, run.dir)
        _notify_run_finished(run)



_NOTIFY_KEYRING_SERVICE = "spacr-notifications"
"""Service name the run-finished notification secrets use in the OS keyring."""

_NOTIFY_SECRET_NAMES = ("smtp_password", "slack_webhook", "ntfy_topic",
                        "ntfy_token", "teams_webhook", "webhook_url",
                        "webhook_token")
"""The notification settings that are secrets and never leave the store."""

_NOTIFY_TIMEOUT_S = 10.0
"""Seconds any one notification channel may take before it is abandoned."""

_DESKTOP_NOTIFIER: List[Any] = [None]
"""The desktop sender the Qt app installs: ``fn(title, body, failed)``."""


def _notify_secrets_path() -> Path:
    """Return ``~/.spacr/notification_secrets.json``, the keyring fallback."""
    return _spacr_home() / "notification_secrets.json"


def _notify_keyring() -> Any:
    """The ``keyring`` module when a working OS keyring backs it, else None.

    The fail and null backends, which keyring picks when the system has no
    secret service, count as no keyring.
    """
    try:
        import keyring

        backend = keyring.get_keyring()
        if float(getattr(backend, "priority", 1) or 0) <= 0:
            return None
        if type(backend).__module__.endswith((".fail", ".null")):
            return None
        return keyring
    except Exception:
        return None


def _read_notify_secret_file(path: Optional[Path] = None) -> Dict[str, str]:
    """The secrets in a mode-600 secret file, or an empty dict.

    :param path: the file; the notification secret file when ``None``.
    """
    try:
        path = Path(path) if path is not None else _notify_secrets_path()
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v) for k, v in data.items() if v}
    except Exception:
        LOG.debug("could not read the notification secret file")
        return {}


def _write_notify_secret_file(values: Dict[str, str],
                              path: Optional[Path] = None) -> None:
    """Write a secret file readable by its owner only (mode 600).

    The file is created with mode 600 before anything is written to it and
    replaces the old one in one step; with nothing left to keep it is
    removed.

    :param values: secret name to value.
    :param path: the file; the notification secret file when ``None``.
    """
    path = Path(path) if path is not None else _notify_secrets_path()
    kept = {k: v for k, v in values.items() if v}
    if not kept:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}")
    descriptor = os.open(str(temporary),
                         os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(kept, handle)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _store_notify_secret(name: str, value: str) -> str:
    """Keep one notification secret, in the OS keyring when there is one.

    Without a usable keyring the secret goes to
    ``~/.spacr/notification_secrets.json``, mode 600. An empty value forgets
    the secret in both places. The value is never logged.

    :param name: one of ``_NOTIFY_SECRET_NAMES``.
    :param value: the secret.
    :returns: ``"keyring"``, ``"file"`` or ``"forgotten"``.
    :raises ValueError: for a name that is not a notification secret.
    """
    if name not in _NOTIFY_SECRET_NAMES:
        raise ValueError(f"unknown notification secret {name!r}")
    value = str(value or "")
    stored = _read_notify_secret_file()
    ring = _notify_keyring()
    if not value:
        if stored.pop(name, None) is not None:
            _write_notify_secret_file(stored)
        if ring is not None:
            try:
                ring.delete_password(_NOTIFY_KEYRING_SERVICE, name)
            except Exception:
                LOG.debug("no keyring entry to forget")
        return "forgotten"
    if ring is not None:
        try:
            ring.set_password(_NOTIFY_KEYRING_SERVICE, name, value)
            if stored.pop(name, None) is not None:
                _write_notify_secret_file(stored)
            return "keyring"
        except Exception as exc:
            LOG.info("the OS keyring refused a notification secret (%s); "
                     "keeping it in %s", type(exc).__name__,
                     _notify_secrets_path())
    stored[name] = value
    _write_notify_secret_file(stored)
    return "file"


def _load_notify_secret(name: str) -> str:
    """Read one notification secret: the OS keyring first, then the file.

    :param name: one of ``_NOTIFY_SECRET_NAMES``.
    :returns: the secret, or an empty string when none is stored.
    """
    ring = _notify_keyring()
    if ring is not None:
        try:
            value = ring.get_password(_NOTIFY_KEYRING_SERVICE, name)
            if value:
                return str(value)
        except Exception:
            LOG.debug("the OS keyring could not be read")
    return _read_notify_secret_file().get(name, "")


def _notification_config() -> Optional[Dict[str, Any]]:
    """The run-finished notification settings from Preferences, or None.

    None when notifications are off, when no channel is ready, when the
    Show alpha features gate hides them, and when this install cannot read
    Preferences at all.
    """
    try:
        from .qt.preferences import _run_notification_config
    except Exception:
        LOG.debug("no Preferences to read notification settings from")
        return None
    try:
        return _run_notification_config()
    except Exception:
        LOG.debug("could not read the notification settings", exc_info=True)
        return None


def _notify_duration(seconds: float) -> str:
    """Say a run's wall time the way a person would, e.g. ``1 h 02 min``."""
    seconds = max(0.0, float(seconds or 0.0))
    if seconds < 60:
        return f"{seconds:.0f} s"
    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes} min {secs:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


def _notify_output_pointer(run: "Run") -> str:
    """Where the run's results are: its report, destination or source."""
    settings = run.settings or {}
    for key in ("report_path", "dst", "src"):
        value = settings.get(key)
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if value not in (None, ""):
            return str(value)
    return ""


def _run_qc_summary(run: "Run") -> List[str]:
    """Short QC lines for a run: items processed, failures and key numbers.

    Read from what the run recorded -- finalized item ledgers, stage states
    and numeric stage metrics, warnings and hashed outputs; nothing is
    recomputed.

    :param run: the finished run.
    :returns: lines of text, possibly none.
    """
    lines: List[str] = []
    for ledger in run._ledgers[:6]:
        lines.append(
            f"{ledger['name']}: {ledger['succeeded']} of "
            f"{ledger['attempted']} items processed, "
            f"{ledger['failed']} failed")
    if run.stages:
        states: Dict[str, int] = {}
        for stage in run.stages:
            state = str(stage.get("state") or "pending")
            states[state] = states.get(state, 0) + 1
        lines.append("Stages: " + ", ".join(
            f"{count} {state}" for state, count in sorted(states.items())))
        numbers = []
        for stage in run.stages:
            for name, value in (stage.get("metrics") or {}).items():
                if isinstance(value, bool) or not isinstance(
                        value, (int, float)):
                    continue
                label = stage.get("label") or stage.get("id")
                shown = (f"{value:g}" if isinstance(value, float)
                         else str(value))
                numbers.append(f"{label} {name}: {shown}")
        lines.extend(numbers[:6])
    if run.run_warnings:
        lines.append(f"Warnings: {len(run.run_warnings)}")
    if run.output_hashes:
        lines.append(f"Output files recorded: {len(run.output_hashes)}")
    return lines


def _run_notification_message(run: "Run") -> Dict[str, Any]:
    """The title and body a finished or failed run is announced with.

    :param run: the closed run.
    :returns: ``{"title", "body", "failed"}``.
    """
    failed = run.status == "failed"
    outcome = "failed" if failed else "finished"
    name = run.app_key or "run"
    elapsed = (run.end_ts or time.time()) - run.start_ts
    lines = [
        f"Run: {name} ({run.dir.name})",
        f"Outcome: {outcome}",
        f"Duration: {_notify_duration(elapsed)}",
    ]
    if failed:
        error = [line.strip() for line in
                 (run.error_traceback or "").splitlines() if line.strip()]
        if error:
            lines.append(f"Error: {error[-1][:300]}")
    lines.extend(_run_qc_summary(run))
    output = _notify_output_pointer(run)
    if output:
        lines.append(f"Output: {output}")
    lines.append(f"Run record: {run.dir}")
    return {"title": f"spaCR run {outcome}: {name}",
            "body": "\n".join(lines), "failed": failed}


def _notify_scrub(text: Any, secrets: Iterable[str]) -> str:
    """``text`` with every secret replaced by ``***``, for a log line."""
    out = str(text)
    for secret in secrets:
        if secret and len(secret) >= 3:
            out = out.replace(secret, "***")
    return out


def _notify_http_post(url: str, data: bytes,
                      headers: Dict[str, str]) -> int:
    """POST ``data`` to an http(s) ``url`` and return the status code.

    :raises ValueError: for any other scheme.
    :raises RuntimeError: for a response outside 2xx.
    """
    import urllib.request
    from urllib.parse import urlparse

    if urlparse(url).scheme not in ("http", "https"):
        raise ValueError("the address must start with http:// or https://")
    request = urllib.request.Request(url, data=data, headers=headers,
                                     method="POST")
    with urllib.request.urlopen(request, timeout=_NOTIFY_TIMEOUT_S) as reply:
        status = int(getattr(reply, "status", 200) or 200)
    if not 200 <= status < 300:
        raise RuntimeError(f"the server answered {status}")
    return status


def _notify_by_email(message: Dict[str, Any], notify: Dict[str, Any],
                     password: str) -> None:
    """Send the message over SMTP (STARTTLS, SSL or plain, as configured)."""
    import smtplib
    import ssl
    from email.message import EmailMessage

    host = str(notify.get("smtp_host") or "").strip()
    recipients = [part.strip() for part in
                  re.split(r"[,;\s]+", str(notify.get("email_to") or ""))
                  if part.strip()]
    if not host or not recipients:
        raise ValueError("an SMTP server and a recipient are needed")
    port = int(notify.get("smtp_port") or 587)
    security = str(notify.get("smtp_security") or "starttls")
    user = str(notify.get("smtp_user") or "").strip()
    sender = (str(notify.get("email_from") or "").strip() or user
              or recipients[0])
    mail = EmailMessage()
    mail["Subject"] = message["title"]
    mail["From"] = sender
    mail["To"] = ", ".join(recipients)
    mail.set_content(message["body"])
    context = ssl.create_default_context()
    if security == "ssl":
        server = smtplib.SMTP_SSL(host, port, timeout=_NOTIFY_TIMEOUT_S,
                                  context=context)
    else:
        server = smtplib.SMTP(host, port, timeout=_NOTIFY_TIMEOUT_S)
    with server:
        if security == "starttls":
            server.starttls(context=context)
        if user and password:
            server.login(user, password)
        server.send_message(mail)


def _notify_by_slack(message: Dict[str, Any], webhook: str) -> None:
    """Post the message to a Slack incoming webhook."""
    if not webhook:
        raise ValueError("no Slack webhook address is saved")
    payload = {"text": f"*{message['title']}*\n{message['body']}"}
    _notify_http_post(webhook, json.dumps(payload).encode("utf-8"),
                      {"Content-Type": "application/json"})


def _notify_by_teams(message: Dict[str, Any], webhook: str) -> None:
    """Post an Adaptive Card to a Microsoft Teams workflow webhook.

    :param message: the notification's ``title``, ``body`` and ``failed``.
    :param webhook: the URL of a workflow allowing anyone to call it.
    :returns: ``None`` after the server accepts the card.
    :raises ValueError: when no webhook address is supplied.
    """
    if not webhook:
        raise ValueError("no Teams webhook address is saved")
    body = [{"type": "TextBlock", "text": message["title"],
             "weight": "Bolder", "wrap": True}]
    body.extend({"type": "TextBlock", "text": line, "wrap": True,
                 "spacing": "Small"}
                for line in message["body"].splitlines() if line)
    payload = {
        "type": "message",
        "attachments": [{
            "contentType": "application/vnd.microsoft.card.adaptive",
            "contentUrl": None,
            "content": {
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "type": "AdaptiveCard", "version": "1.2", "body": body,
            },
        }],
    }
    _notify_http_post(webhook, json.dumps(payload).encode("utf-8"),
                      {"Content-Type": "application/json"})


def _notify_by_webhook(message: Dict[str, Any], webhook: str,
                       token: str) -> None:
    """POST the notification as JSON to a generic webhook.

    :param message: the notification's ``title``, ``body`` and ``failed``.
    :param webhook: the receiving HTTP or HTTPS URL.
    :param token: optional bearer token for the receiving service.
    :returns: ``None`` after the server accepts the notification.
    :raises ValueError: when no webhook address is supplied.
    """
    if not webhook:
        raise ValueError("no webhook address is saved")
    payload = {"title": message["title"], "body": message["body"],
               "failed": bool(message["failed"])}
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    _notify_http_post(webhook, json.dumps(payload).encode("utf-8"), headers)


def _notify_by_ntfy(message: Dict[str, Any], notify: Dict[str, Any],
                    topic: str, token: str) -> None:
    """Publish the message to an ntfy topic."""
    from email.header import Header
    from urllib.parse import quote

    if not topic:
        raise ValueError("no ntfy topic is saved")
    server = str(notify.get("ntfy_server") or "https://ntfy.sh").rstrip("/")
    title = message["title"]
    if not title.isascii():
        title = Header(title, "utf-8").encode()
    headers = {
        "Title": title,
        "Tags": "x" if message["failed"] else "white_check_mark",
        "Priority": "high" if message["failed"] else "default",
        "Content-Type": "text/plain; charset=utf-8",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    _notify_http_post(f"{server}/{quote(topic, safe='')}",
                      message["body"].encode("utf-8"), headers)


def _desktop_os_notify(title: str, body: str) -> None:
    """Show a desktop notification without Qt: notify-send or osascript.

    :raises RuntimeError: when this system offers neither.
    """
    if sys.platform.startswith("linux"):
        program = shutil.which("notify-send")
        if program:
            subprocess.run([program, "--app-name=spaCR", title, body],
                           stdin=subprocess.DEVNULL, capture_output=True,
                           timeout=_NOTIFY_TIMEOUT_S, check=False)
            return
    if sys.platform == "darwin":
        def quoted(text: str) -> str:
            """``text`` as an AppleScript string literal on one line."""
            text = text.replace("\\", "\\\\").replace('"', '\\"')
            return '"' + " ".join(text.splitlines()) + '"'

        subprocess.run(
            ["osascript", "-e",
             f"display notification {quoted(body)} with title {quoted(title)}"],
            stdin=subprocess.DEVNULL, capture_output=True,
            timeout=_NOTIFY_TIMEOUT_S, check=False)
        return
    raise RuntimeError("no desktop notification service on this system")


def _notify_by_desktop(message: Dict[str, Any]) -> None:
    """Show the message on this computer's desktop.

    In the app, the notifier it installed shows it from the tray; elsewhere
    the operating system's own notification command is used.
    """
    notifier = _DESKTOP_NOTIFIER[0]
    if notifier is not None:
        notifier(message["title"], message["body"], bool(message["failed"]))
        return
    _desktop_os_notify(message["title"], message["body"])


def _send_notification(message: Dict[str, Any],
                       notify: Dict[str, Any]) -> Dict[str, str]:
    """Send one message by every channel ``notify`` switches on.

    Every channel is tried whatever happened to the ones before it. A
    failure is logged with every secret masked, and is returned rather than
    raised.

    :param message: ``{"title", "body", "failed"}``.
    :param notify: the notification settings, as
        :func:`_notification_config` returns them; an optional ``secrets``
        dict supplies secrets to try instead of the stored ones.
    :returns: channel name to ``"sent"`` or a short failure reason.
    """
    secrets: Dict[str, str] = {}
    results: Dict[str, str] = {}

    def secret(name: str) -> str:
        """The secret supplied to try, else the stored one."""
        given = (notify.get("secrets") or {}).get(name)
        secrets[name] = str(given) if given else _load_notify_secret(name)
        return secrets[name]

    channels = []
    if notify.get("desktop"):
        channels.append(("desktop", lambda: _notify_by_desktop(message)))
    if notify.get("email"):
        channels.append(("email", lambda: _notify_by_email(
            message, notify, secret("smtp_password"))))
    if notify.get("slack"):
        channels.append(("slack", lambda: _notify_by_slack(
            message, secret("slack_webhook"))))
    if notify.get("ntfy"):
        channels.append(("ntfy", lambda: _notify_by_ntfy(
            message, notify, secret("ntfy_topic"), secret("ntfy_token"))))
    if notify.get("teams"):
        channels.append(("teams", lambda: _notify_by_teams(
            message, secret("teams_webhook"))))
    if notify.get("webhook"):
        channels.append(("webhook", lambda: _notify_by_webhook(
            message, secret("webhook_url"), secret("webhook_token"))))
    for name, send in channels:
        try:
            send()
            results[name] = "sent"
        except Exception as exc:
            reason = _notify_scrub(f"{type(exc).__name__}: {exc}",
                                   secrets.values())
            results[name] = reason
            LOG.warning("run notification by %s failed: %s", name, reason)
    return results


def _dispatch_notification(message: Dict[str, Any],
                           notify: Dict[str, Any]) -> threading.Thread:
    """Send ``message`` on a thread of its own and return that thread.

    Not a daemon thread: a command-line run that has just finished waits
    for its notification, each channel for at most ``_NOTIFY_TIMEOUT_S``,
    rather than exiting before it is sent. The results land on the
    thread's ``results`` dict.

    :param message: ``{"title", "body", "failed"}``.
    :param notify: the notification settings.
    :returns: the started thread.
    """
    results: Dict[str, str] = {}

    def work() -> None:
        """Send, and keep what happened."""
        try:
            results.update(_send_notification(message, notify))
        except Exception:
            LOG.debug("the notification thread failed", exc_info=True)

    thread = threading.Thread(target=work, name="spacr-run-notification")
    thread.results = results
    thread.start()
    return thread


def _notify_run_finished(run: "Run") -> Optional[threading.Thread]:
    """Announce a finished or failed run, if Preferences asks for it.

    A cancelled run is not announced: the person who stopped it knows. A
    run shorter than the configured minimum is not either, and neither is
    a finished run when only failures are asked for. Nothing here raises
    or waits for the network: the sending happens on its own thread.

    :param run: the run :func:`open_run` has just closed.
    :returns: the sending thread, or None when nothing is sent.
    """
    try:
        if run.status not in ("success", "failed"):
            return None
        notify = _notification_config()
        if not notify:
            return None
        if run.status == "success" and notify.get("when") == "failed":
            return None
        elapsed = (run.end_ts or time.time()) - run.start_ts
        if elapsed < float(notify.get("min_minutes") or 0) * 60.0:
            return None
        return _dispatch_notification(_run_notification_message(run),
                                      notify)
    except Exception:
        LOG.debug("could not send the run-finished notification",
                  exc_info=True)
        return None



def _note_ledger_on_the_open_run(ledger: Any) -> None:
    """Hand a finalized item ledger to the run open on this thread, if any."""
    run = current_run()
    if run is not None:
        run._note_ledger(ledger)


def _listen_for_ledgers() -> None:
    """Have every finalized :class:`spacr.errors.RunLedger` reach the run."""
    try:
        from .errors import _FINALIZE_LISTENERS

        if _note_ledger_on_the_open_run not in _FINALIZE_LISTENERS:
            _FINALIZE_LISTENERS.append(_note_ledger_on_the_open_run)
    except Exception:
        LOG.debug("could not listen for item ledgers", exc_info=True)


_listen_for_ledgers()

def _run_dir_names(root: Path) -> List[str]:
    """Every run-folder name under ``root``, from ONE directory read.

    ``root.iterdir()`` followed by ``d.is_dir()`` is two syscalls per
    entry, and both `recent_runs` and `journal_totals` used to make that
    pass twice each. On a journal of 10,192 runs -- an ordinary number
    after a few weeks of use -- that was over 30,000 stat calls to answer
    a question the directory read had already answered, on the GUI
    thread, every time Home refreshed.

    ``os.scandir`` carries the directory-ness in the entry itself on
    Linux (``d_type``), so this is one read and no stats at all. Entries
    whose type the filesystem declines to report fall back to a stat,
    which is what ``is_dir()`` would have cost anyway.

    A name is returned rather than a Path because both callers sort by
    name before they touch anything on disk.
    """
    import os as _os

    try:
        with _os.scandir(root) as entries:
            return [e.name for e in entries if e.is_dir()]
    except OSError:
        return []


def recent_runs(limit: int = 10) -> List[Dict[str, Any]]:
    """Return the ``limit`` most-recent runs newest-first.

    Ordered by the manifest's ``start_utc`` timestamp (parsed as
    :class:`datetime.datetime`), so runs opened in the same wall-
    clock second still sort correctly — folder names alone truncate
    to seconds and would produce ties. Only the newest
    ``max(limit * 4, limit + 64)`` folders by name are opened at all
    (for non-negative ``limit``), so startup cost does not grow with the
    size of the journal.

    PASS ``None`` FOR "EVERY RUN", never a negative number. The truncation
    at the end is ``all_entries[:limit]``, so ``limit=-1`` reads the whole
    journal and then hands back all but the OLDEST entry -- observed
    2026-09-03 on an 11,027-run journal, which returned 11,026. A folder with no ``manifest.json`` is skipped
    quietly; one whose manifest cannot be parsed is skipped with a
    logged warning.

    Each entry is a dict with keys ``dir`` (Path), ``app_key`` (str),
    ``status`` (str), ``start_utc`` (ISO str), ``elapsed_s`` (float),
    and the raw ``manifest`` (dict, best-effort).
    """
    all_entries: List[Dict[str, Any]] = []
    root = runs_root()

    candidates = [
        root / name
        for name in sorted(_run_dir_names(root), reverse=True)
    ] if root.exists() else []
    if limit is not None and limit >= 0:
        candidates = candidates[:max(limit * 4, limit + 64)]

    for d in candidates:
        manifest_path = d / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            m = json.loads(manifest_path.read_text())
        except Exception as exc:
            LOG.warning("skipping run folder %s: its manifest.json could not "
                        "be read (%s)", d.name, exc)
            continue
        all_entries.append({
            "dir":       d,
            "app_key":   m.get("app_key", "?"),
            "status":    m.get("status", "?"),
            "start_utc": m.get("start_utc", ""),
            "elapsed_s": m.get("elapsed_s"),
            "manifest":  m,
        })
    def _sort_key(e):
        """Sort a recent-run entry by parsed start time, then folder mtime."""
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
            else:
                warnings_list.append(str(values))
        warnings_list.extend(str(error) for error in rec["errors"])

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
        """Return a UTC-aware start and resilient directory-mtime tiebreaker."""
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



#: Bumped when what is counted changes. An old cache is then WRONG
#: rather than merely stale.
_TOTALS_CACHE_VERSION = 1


def _totals_cache_path() -> Path:
    """Where the incremental totals live. Inside the runs root, as a FILE --
    `journal_totals` only iterates directories, so it cannot see itself."""
    return runs_root() / ".journal_totals.json"


def _read_totals_cache():
    """Cached totals, or None when absent, unreadable or the wrong shape.

    Any doubt returns None and the caller recounts from scratch. A wrong
    run count on the Home dashboard is worse than a slow one.
    """
    try:
        raw = json.loads(_totals_cache_path().read_text())
        if int(raw.get("version", 0)) != _TOTALS_CACHE_VERSION:
            return None
        return {
            "totals": {k: int(v) for k, v in dict(raw["totals"]).items()},
            "counted": set(raw["counted"]),
            "models": set(raw["models"]),
        }
    except Exception:
        return None


def _write_totals_cache(totals, counted, models) -> None:
    """Store the totals. Failure is silent -- this is an optimisation."""
    try:
        path = _totals_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_text(json.dumps({
            "version": _TOTALS_CACHE_VERSION,
            "totals": dict(totals),
            "counted": sorted(counted),
            "models": sorted(models),
        }))
        os.replace(tmp, path)
    except Exception:
        pass


def journal_totals() -> Dict[str, int]:
    """Return aggregate counts across every stored run.

    Powers the Home-screen insights dashboard: ``total_runs`` (all
    manifests seen), ``mask_runs`` / ``measure_runs`` / ``classify_runs``
    (per-app tallies), and ``models_recorded`` (distinct model hashes
    ever recorded, across runs of every app — not only mask runs).
    Returns zeros when no journal exists yet.

    Counting is incremental: the totals and the names of the folders
    already counted are persisted in ``.journal_totals.json`` in the runs
    root, so each call parses only the manifests it has not seen. A folder
    that has since been deleted cannot be subtracted, so that case falls
    back to a full recount. A manifest that cannot be parsed is left out
    of the totals with a logged warning.
    """
    totals = {"total_runs": 0, "mask_runs": 0, "measure_runs": 0,
                "classify_runs": 0, "models_recorded": 0}
    seen_models: set = set()
    root = runs_root()
    if not root.exists():
        return totals

    names = _run_dir_names(root)
    present = set(names)
    cached = _read_totals_cache()
    counted: set = set()
    if cached is not None and cached["counted"] <= present:
        totals.update(cached["totals"])
        seen_models |= cached["models"]
        counted = cached["counted"]

    for name in names:
        if name in counted:
            continue
        d = root / name
        manifest_path = d / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            m = json.loads(manifest_path.read_text())
        except Exception as exc:
            LOG.warning("run folder %s is not counted: its manifest.json "
                        "could not be read (%s)", d.name, exc)
            continue
        totals["total_runs"] += 1
        app_key = m.get("app_key", "")
        if app_key in ("mask", "measure", "classify"):
            totals[f"{app_key}_runs"] += 1
        hashes = m.get("model_hashes") or {}
        if isinstance(hashes, dict):
            for digest in hashes.values():
                if digest:
                    seen_models.add(digest)
        for model in m.get("models", []) or []:
            sha = model.get("sha256") if isinstance(model, dict) else None
            if sha:
                seen_models.add(sha)
        counted.add(d.name)
    totals["models_recorded"] = len(seen_models)
    _write_totals_cache(totals, counted, seen_models)
    return totals


def load_run_settings(run_dir: Path) -> Dict[str, Any]:
    """Read a run's ``settings.json`` (falling back to settings.csv).

    :param run_dir: journal run directory containing the settings files.
    """
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

    :param a: first settings value to compare after normalisation.
    :param b: second settings value to compare after normalisation.

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

    :param ref: run object, directory path, run id, or unambiguous id prefix.

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

    try:
        rec["settings"] = load_run_settings(d) or {}
    except FileNotFoundError:
        rec["errors"].append("no settings.json / settings.csv in run folder")
    except Exception as e:
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
    """Render a duration to one decimal second, or an em dash when invalid."""
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


_BLIND_CODE_PREFIX = "B"
_LOCK_IGNORED_KEYS = frozenset({"hash_inputs"})
_LOCK_FILE_KEY_PARTS = ("model", "gate", "checkpoint", "weights", "threshold")
#: A JSON file larger than this is not read as a Gate Editor gating strategy.
_GATE_FILE_MAX_BYTES = 16 * 1024 * 1024
_LOCK_STATUS_WORDS = {
    "verified": "verified, the run matches it",
    "deviation": "DEVIATION, changed since the lock",
    "post_hoc": "POST-HOC, changed after the key was unblinded",
    "tampered": "TAMPERED, the lock file no longer matches its own hash",
    "not_preregistered": "NOT PREREGISTERED, locked after the key had "
                         "been unblinded",
}


def _blinding_root() -> Path:
    """Where blinding keys and their logs live, beside the run journal."""
    root = runs_root().parent / "blinding"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _locks_root() -> Path:
    """Where preregistered analysis locks live, beside the run journal."""
    root = runs_root().parent / "analysis_locks"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _who() -> str:
    """``user@host`` for the person at this computer, as far as it is known."""
    try:
        import getpass
        user = getpass.getuser()
    except Exception:
        user = os.environ.get("USER") or os.environ.get("USERNAME") or ""
    host = platform.node() or ""
    return f"{user or 'unknown'}@{host}" if host else (user or "unknown")


def _utc_now() -> str:
    """The current time as an ISO 8601 UTC string, to the microsecond."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _record_id() -> str:
    """A sortable, unique name for a key or a lock file."""
    return (datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_%f")
            + "_" + uuid.uuid4().hex[:8])


def _scope_path(src: Any) -> str:
    """``src`` as an absolute, normalised path, or empty when there is none."""
    text = str(src or "").strip()
    if not text:
        return ""
    return os.path.normcase(os.path.abspath(os.path.expanduser(text)))


def _paths_overlap(a: str, b: str) -> bool:
    """Whether two scope paths are the same folder or one holds the other."""
    if not a or not b:
        return False
    if a == b:
        return True
    return a.startswith(b.rstrip(os.sep) + os.sep) or b.startswith(
        a.rstrip(os.sep) + os.sep)


def _append_blinding_event(key_id: str, event: Dict[str, Any]) -> None:
    """Append one event to a key's log, one JSON object per line."""
    path = _blinding_root() / f"{key_id}.log.jsonl"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")


def _blinding_events(key_id: str) -> List[Dict[str, Any]]:
    """Every event recorded for a blinding key, oldest first.

    :param key_id: the key's id, as :func:`start_blinding` returned it.
    :returns: dicts with ``event`` (``blinded``, ``unblinded`` or
        ``closed``), ``utc`` and ``who``, plus whatever the event carried;
        an empty list for a key without a log.
    """
    path = _blinding_root() / f"{Path(str(key_id)).name}.log.jsonl"
    events: List[Dict[str, Any]] = []
    if not path.is_file():
        return events
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _read_blinding_key(key_id: str) -> Dict[str, Any]:
    """The stored key, or ``FileNotFoundError`` naming the id."""
    path = _blinding_root() / f"{Path(str(key_id)).name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"no blinding key {key_id!r} in {path.parent}")
    return json.loads(path.read_text(encoding="utf-8"))


def start_blinding(items: Iterable[Any], *, scope: str, src: Any = "",
                   seed: Optional[int] = None) -> Dict[str, Any]:
    """Shuffle ``items`` under coded names and keep the key away from them.

    Blind scoring: the person scoring sees each item only by its code
    (``B0001``, ``B0002`` and so on, numbered in the shuffled order) and in
    that order, so neither a name nor its neighbours say which plate, well
    or condition it came from. The key that maps codes back to items is
    written to ``~/.spacr/blinding/<key id>.json``, outside the data
    folder, and every later event on it (unblinding, closing) is appended
    to ``<key id>.log.jsonl`` with who did it and when.

    :param items: the things being scored, such as crop paths or image
        files; repeats are kept once, in first-seen order.
    :param scope: what is being scored, such as ``"annotate"``; stored with
        the key.
    :param src: the experiment folder the items belong to. Analysis locks
        on the same folder read this key's unblinding record.
    :param seed: the shuffle's seed; a random one is drawn and stored when
        omitted, so the order can be rebuilt from the key.
    :returns: ``key_id``, ``order`` (the items, shuffled) and ``codes``
        (``{item: code}``).
    """
    order = list(dict.fromkeys(str(item) for item in items))
    if seed is None:
        seed = random.SystemRandom().randrange(2 ** 32)
    random.Random(int(seed)).shuffle(order)
    width = max(4, len(str(len(order))))
    codes = {item: f"{_BLIND_CODE_PREFIX}{index + 1:0{width}d}"
             for index, item in enumerate(order)}
    key_id = _record_id()
    record = {
        "key_id": key_id,
        "scope": str(scope),
        "src": _scope_path(src),
        "created_utc": _utc_now(),
        "created_by": _who(),
        "seed": int(seed),
        "n_items": len(order),
        "order": order,
        "codes": codes,
    }
    _atomic_write_text(_blinding_root() / f"{key_id}.json",
                       json.dumps(record, indent=1, sort_keys=True))
    _append_blinding_event(key_id, {
        "event": "blinded", "utc": record["created_utc"],
        "who": record["created_by"], "scope": record["scope"],
        "src": record["src"], "n_items": len(order)})
    return {"key_id": key_id, "order": order, "codes": codes}


def unblind(key_id: str, *, reason: str = "") -> Dict[str, str]:
    """Open a blinding key, and record who opened it and when.

    The record is appended to the key's log before the key is returned, so
    the identities cannot be read through this call without leaving the
    record. An analysis lock on the same folder treats a difference found
    after this moment as post-hoc.

    :param key_id: the key's id, as :func:`start_blinding` returned it.
    :param reason: why it was opened; stored with the record.
    :returns: ``{code: item}``.
    :raises FileNotFoundError: when there is no such key.
    """
    record = _read_blinding_key(key_id)
    _append_blinding_event(record["key_id"], {
        "event": "unblinded", "utc": _utc_now(), "who": _who(),
        "reason": str(reason or ""), "scope": record.get("scope", ""),
        "src": record.get("src", "")})
    return {code: item for item, code in (record.get("codes") or {}).items()}


def _close_blinding(key_id: str, *, reason: str = "") -> None:
    """Record that a blinded session ended without opening the key."""
    try:
        record = _read_blinding_key(key_id)
    except (FileNotFoundError, ValueError):
        return
    _append_blinding_event(record["key_id"], {
        "event": "closed", "utc": _utc_now(), "who": _who(),
        "reason": str(reason or ""), "src": record.get("src", "")})


def _unblinding_times(src: Any) -> List[str]:
    """When any key on ``src``, or on a folder in or around it, was unblinded.

    :param src: the experiment folder.
    :returns: ISO UTC times, oldest first.
    """
    scope = _scope_path(src)
    times: List[str] = []
    if not scope:
        return times
    for path in sorted(_blinding_root().glob("*.log.jsonl")):
        key_id = path.name[:-len(".log.jsonl")]
        for event in _blinding_events(key_id):
            if (event.get("event") == "unblinded"
                    and _paths_overlap(scope, str(event.get("src") or ""))):
                times.append(str(event.get("utc") or ""))
    return sorted(t for t in times if t)


def _lock_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """The settings a lock records: JSON-safe, without run-only switches."""
    kept = {str(k): v for k, v in (settings or {}).items()
            if not str(k).startswith("_") and str(k) not in _LOCK_IGNORED_KEYS}
    return json.loads(json.dumps(kept, sort_keys=True, default=str))


def _existing_file(value: Any) -> Optional[Path]:
    """``value`` as a path to an existing file, or ``None``.

    :param value: a string or path, possibly starting with ``~``.
    :returns: the path, not resolved, when it names a regular file.
    """
    if not isinstance(value, (str, Path)) or not str(value).strip():
        return None
    path = Path(os.path.expanduser(str(value)))
    try:
        return path if path.is_file() else None
    except OSError:
        return None


def _gate_payload(value: Any) -> Optional[Dict[str, Any]]:
    """A gating strategy as canonical plain data, or ``None`` if not one.

    Accepts what the Gate Editor keeps and saves: a gate set object (anything
    with ``to_dict()``, such as ``spacr.qt.widgets.gate_spec.GateSet``), the
    dict it turns into, or a ``.json`` gate file. Comparing the parsed gates
    rather than the file bytes means re-saving the same gates, or reformatting
    the file, is not a change.

    :param value: the candidate gate set.
    :returns: ``{"gates": [...]}`` with sorted keys, or ``None``.
    """
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        try:
            data = value.to_dict()
        except Exception:
            return None
    elif isinstance(value, dict):
        data = value
    else:
        path = _existing_file(value)
        if path is None or path.suffix.lower() != ".json":
            return None
        try:
            if path.stat().st_size > _GATE_FILE_MAX_BYTES:
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    if not isinstance(data, dict):
        return None
    rows = data.get("gates")
    if not isinstance(rows, list) or not all(
            isinstance(row, dict) and "kind" in row for row in rows):
        return None
    return json.loads(json.dumps({"gates": rows}, sort_keys=True, default=str))


def _gate_record(payload: Dict[str, Any], source: str) -> Dict[str, Any]:
    """What a lock keeps about one gating strategy.

    :param payload: the canonical gates, from :func:`_gate_payload`.
    :param source: ``"file"`` for a gate file, ``"memory"`` for gates handed
        over as an object.
    :returns: the source, the digest of the whole strategy, and one digest
        per gate by name, so a later change can say which gate moved.
    """
    per_gate = {}
    for index, row in enumerate(payload.get("gates") or []):
        name = str(row.get("name") or f"#{index + 1}")
        per_gate[name] = _json_digest(row)
    return {"source": source, "sha256": _json_digest(payload),
            "gates": per_gate}


def _gate_items(gates: Any) -> List[Tuple[str, Any]]:
    """Name each gating strategy handed to a lock or a check.

    :param gates: one gate set, gate file or gate dict; a list of them; or a
        mapping from a label of your choice to one.
    :returns: ``(label, item)`` pairs; a gate file's label is its resolved
        path, so the lock and a later check agree on it.
    """
    if gates is None:
        return []
    if isinstance(gates, dict) and "gates" not in gates:
        pairs = [(str(label), item) for label, item in gates.items()]
    elif isinstance(gates, (list, tuple)):
        pairs = [("", item) for item in gates]
    else:
        pairs = [("", gates)]
    named: List[Tuple[str, Any]] = []
    for index, (label, item) in enumerate(pairs):
        path = _existing_file(item) if isinstance(item, (str, Path)) else None
        if path is not None:
            label = str(path.resolve(strict=False))
        elif not label:
            label = "gates" if len(pairs) == 1 else f"gates[{index}]"
        named.append((label, item))
    return named


def _lock_gates(settings: Dict[str, Any], gates: Any) -> Dict[str, Dict[str, Any]]:
    """Every gating strategy a plan depends on, keyed by path or label.

    :param settings: the settings; any value that is a gate file is taken,
        whatever its key is called.
    :param gates: gating strategies handed over outright (see
        :func:`_gate_items`).
    :returns: ``{path or label: record}`` (:func:`_gate_record`).
    :raises ValueError: when something handed over is not a gating strategy.
    """
    found: Dict[str, Dict[str, Any]] = {}
    for value in (settings or {}).values():
        for candidate in (value if isinstance(value, (list, tuple))
                          else [value]):
            path = _existing_file(candidate)
            payload = _gate_payload(path) if path is not None else None
            if payload is not None:
                found[str(path.resolve(strict=False))] = _gate_record(
                    payload, "file")
    for label, item in _gate_items(gates):
        payload = _gate_payload(item)
        if payload is None:
            raise ValueError(f"{label!r} is not a gating strategy")
        source = "file" if _existing_file(label) is not None else "memory"
        found[label] = _gate_record(payload, source)
    return found


def _lock_files(settings: Dict[str, Any], extra: Iterable[Any]) -> Dict[str, str]:
    """Full SHA-256 of every model, gate or threshold file the plan names.

    Gate files are left to :func:`_lock_gates`, which compares the gates
    rather than the bytes.

    :param settings: the settings; a value is hashed when its key names a
        model, gate, checkpoint, weights or threshold and it is a file.
    :param extra: further file paths the plan names outright.
    :returns: ``{absolute path: sha256}``.
    """
    paths = []
    for key, value in (settings or {}).items():
        lowered = str(key).lower()
        if not any(part in lowered for part in _LOCK_FILE_KEY_PARTS):
            continue
        for candidate in (value if isinstance(value, (list, tuple))
                          else [value]):
            if isinstance(candidate, (str, Path)) and str(candidate).strip():
                paths.append(str(candidate))
    paths.extend(str(p) for p in (extra or ()) if str(p or "").strip())
    files: Dict[str, str] = {}
    for text in paths:
        path = _existing_file(text)
        if path is None or _gate_payload(path) is not None:
            continue
        digest = hash_file(path, full=True)
        if digest:
            files[str(path.resolve(strict=False))] = digest
    return files


def _lock_models(models: Any) -> Dict[str, Dict[str, str]]:
    """The models a plan names, as :meth:`Run.record_model` will see them.

    :param models: ``{name: checkpoint path}``, the names being those the
        pipeline records its models under.
    :returns: ``{name: {"path", "sha256"}}``.
    :raises FileNotFoundError: when a named checkpoint is not a file.
    """
    locked: Dict[str, Dict[str, str]] = {}
    for name, value in dict(models or {}).items():
        path = _existing_file(value)
        digest = hash_file(path, full=True) if path is not None else None
        if not digest:
            raise FileNotFoundError(f"model {name!r}: no file at {value!r}")
        locked[str(name)] = {"path": str(path.resolve(strict=False)),
                             "sha256": digest}
    return locked


def _recorded_models(app_key: str, src: Any, limit: int = 200) -> Dict[str, str]:
    """The models the newest journalled run of a pipeline on a folder used.

    Lets a lock made from the settings screen carry the models the analysis
    actually loads, which the settings alone do not always name (a built-in
    model resolves to a cached file).

    :param app_key: the pipeline.
    :param src: the folder its runs were on.
    :param limit: how many of the newest run folders to look through.
    :returns: ``{name: checkpoint path}`` for the files that still exist;
        empty when no such run recorded a model.
    """
    scope = _scope_path(src)
    root = runs_root()
    try:
        names = sorted((p.name for p in root.iterdir() if p.is_dir()),
                       reverse=True)[:limit]
    except OSError:
        return {}
    for name in names:
        folder = root / name
        try:
            manifest = json.loads((folder / "manifest.json").read_text(
                encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(manifest, dict) or str(
                manifest.get("app_key")) != str(app_key):
            continue
        try:
            settings = load_run_settings(folder)
        except Exception:
            continue
        if _scope_path(settings.get("src")) != scope:
            continue
        models = {str(k): str(v.get("path")) for k, v in (
            manifest.get("model_files") or {}).items()
            if isinstance(v, dict) and _existing_file(v.get("path"))}
        if models:
            return models
    return {}


def _lock_digest(record: Dict[str, Any]) -> str:
    """The hash a lock carries: over everything in it but the hash itself."""
    return _json_digest({k: v for k, v in record.items() if k != "sha256"})


def _lock_entries(record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The pipelines a lock covers, each with its ``src`` and settings.

    A lock written before locks could span pipelines (schema 1) covers the
    one pipeline named at its top level.

    :param record: the lock.
    :returns: ``{app_key: {"src", "settings"}}``.
    """
    entries = record.get("pipelines")
    if isinstance(entries, dict) and entries:
        return {str(k): v for k, v in entries.items() if isinstance(v, dict)}
    return {str(record.get("app_key")): {
        "src": record.get("src") or "",
        "settings": record.get("settings") or {}}}


def _lock_unblindings(record: Dict[str, Any]) -> List[str]:
    """Every unblinding on any folder the lock covers, oldest first."""
    times: Set[str] = set()
    for entry in _lock_entries(record).values():
        times.update(_unblinding_times(entry.get("src")))
    return sorted(times)


def lock_analysis(settings: Dict[str, Any], *, app_key: str,
                  hypotheses: str = "", thresholds: Any = None,
                  files: Iterable[Any] = (), note: str = "",
                  gates: Any = None, models: Any = None,
                  pipelines: Optional[Dict[str, Dict[str, Any]]] = None
                  ) -> Dict[str, Any]:
    """Freeze an analysis plan before its results are seen.

    The settings, the stated hypotheses and thresholds, the SHA-256 of every
    model, gate or threshold file the settings name (plus ``files``), the
    gating strategies and the models are written with the time and the user
    to ``~/.spacr/analysis_locks/<lock id>.json`` and hashed. From then on
    every journalled run of a locked pipeline on its ``src`` is checked
    against the newest lock by :func:`check_analysis_lock`, and every model a
    run records is checked as it is recorded; the verdict goes into the
    run's ``manifest.json``, and a difference is listed in the manifest's
    warnings, the report and the methods text.

    A lock made after a blinding key on a folder it covers was unblinded is
    marked as such, because it was not made blind.

    :param settings: the settings the analysis will run with.
    :param app_key: the pipeline it will run in, such as ``"classify"``.
    :param hypotheses: the hypotheses, in words.
    :param thresholds: the decision thresholds and gates, in any
        JSON-compatible form.
    :param files: further files the plan depends on.
    :param note: anything else worth keeping with the plan.
    :param gates: Strategies for selecting measurement rows. Accept a gate
        set, its dictionary representation, a saved strategy file, a list of
        these values, or a mapping from labels to these values. Also include
        strategy files referenced by settings. Check strategy files on every
        run. To check a strategy supplied as an object, pass it to
        :func:`check_analysis_lock` using the same label.
    :param models: ``{name: checkpoint path}`` for the models the analysis
        loads, under the names the pipeline records them by.
    :param pipelines: further pipelines of the same plan, as
        ``{app_key: settings}``; each is checked against its own settings on
        its own ``src``, and one unblinding on any of their folders counts
        for the whole plan.
    :returns: the stored lock, including ``lock_id``, ``locked_utc`` and
        ``sha256``.
    :raises ValueError: when ``pipelines`` repeats ``app_key``, or ``gates``
        holds something that is not a gating strategy.
    :raises FileNotFoundError: when a model in ``models`` is not a file.
    """
    plan = {str(app_key): settings or {}}
    for other, other_settings in dict(pipelines or {}).items():
        if str(other) in plan:
            raise ValueError(f"pipeline {other!r} is locked twice")
        plan[str(other)] = other_settings or {}
    entries = {key: {"src": _scope_path(value.get("src")),
                     "settings": _lock_settings(value)}
               for key, value in plan.items()}
    locked_files: Dict[str, str] = {}
    locked_gates: Dict[str, Dict[str, Any]] = {}
    for value in plan.values():
        locked_files.update(_lock_files(value, ()))
        locked_gates.update(_lock_gates(value, None))
    locked_files.update(_lock_files({}, files))
    locked_gates.update(_lock_gates({}, gates))
    record = {
        "schema": 2,
        "lock_id": _record_id(),
        "app_key": str(app_key),
        "src": entries[str(app_key)]["src"],
        "locked_utc": _utc_now(),
        "locked_by": _who(),
        "settings": entries[str(app_key)]["settings"],
        "pipelines": entries,
        "plan": json.loads(json.dumps({
            "hypotheses": str(hypotheses or ""),
            "thresholds": thresholds,
            "note": str(note or ""),
        }, sort_keys=True, default=str)),
        "files": locked_files,
        "gates": locked_gates,
        "models": _lock_models(models),
    }
    record["unblinded_before_lock"] = _lock_unblindings(record)
    record["sha256"] = _lock_digest(record)
    _atomic_write_text(_locks_root() / f"{record['lock_id']}.json",
                       json.dumps(record, indent=1, sort_keys=True))
    return record


def _find_lock(app_key: str, src: Any) -> Optional[Dict[str, Any]]:
    """The newest lock covering ``app_key`` on ``src``, or ``None``."""
    scope = _scope_path(src)
    newest = None
    for path in sorted(_locks_root().glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(record, dict):
            continue
        entry = _lock_entries(record).get(str(app_key))
        if entry is not None and str(entry.get("src") or "") == scope:
            newest = record
    return newest


def _gate_changes(locked: Dict[str, Any], now: Optional[Dict[str, Any]]
                  ) -> List[str]:
    """Which gates were added, removed or changed since the lock.

    :param locked: the lock's record of the strategy.
    :param now: the strategy's record now, or ``None`` when it is gone.
    :returns: phrases such as ``"changed CD4+"``, in gate-name order.
    """
    before = locked.get("gates") or {}
    after = (now or {}).get("gates") or {}
    changes = []
    for name in sorted(set(before) | set(after)):
        if name not in after:
            changes.append(f"removed {name}")
        elif name not in before:
            changes.append(f"added {name}")
        elif before[name] != after[name]:
            changes.append(f"changed {name}")
    return changes


def _gate_deviations(record: Dict[str, Any], gates: Any
                     ) -> List[Dict[str, Any]]:
    """What differs between the gates a lock holds and the gates now.

    A locked gate file is re-read every time. Gates locked as objects are
    compared only when ``gates`` hands over a strategy under the same label;
    a handed-over strategy the lock does not hold is a change too.

    :param record: the lock.
    :param gates: gating strategies in use now (see :func:`_gate_items`).
    :returns: deviations keyed ``gates:<path or label>``, each with the
        gates that moved under ``detail``.
    """
    locked = record.get("gates") or {}
    given = {}
    for label, item in _gate_items(gates):
        payload = _gate_payload(item)
        given[label] = (_gate_record(payload, "memory")
                        if payload is not None else None)
    changed = []
    for label in sorted(set(locked) | set(given)):
        before = locked.get(label)
        if before is None:
            now = given[label]
            changed.append({"key": f"gates:{label}", "locked": None,
                            "now": (now or {}).get("sha256", "")[:16]
                            or "not a gating strategy",
                            "detail": "not in the lock"})
            continue
        if label in given:
            now = given[label]
        elif before.get("source") == "file":
            payload = _gate_payload(label)
            now = _gate_record(payload, "file") if payload is not None else None
        else:
            continue
        if now is not None and now.get("sha256") == before.get("sha256"):
            continue
        changes = _gate_changes(before, now)
        changed.append({
            "key": f"gates:{label}",
            "locked": str(before.get("sha256") or "")[:16],
            "now": (now or {}).get("sha256", "")[:16] or "missing",
            "detail": ", ".join(changes) if now is not None
            else "the gate file is gone or no longer a gating strategy"})
    return changed


def _lock_deviations(record: Dict[str, Any], settings: Dict[str, Any],
                     app_key: Optional[str] = None,
                     gates: Any = None) -> List[Dict[str, Any]]:
    """What differs between a lock and a run of one of its pipelines now.

    :param record: the lock.
    :param settings: the settings of the run.
    :param app_key: the pipeline running them; the lock's own top-level
        pipeline when omitted or not in the lock.
    :param gates: gating strategies in use, for gates locked as objects.
    :returns: dicts with ``key``, ``locked`` and ``now``.
    """
    entry = _lock_entries(record).get(str(app_key)) if app_key else None
    if entry is None:
        entry = {"settings": record.get("settings") or {}}
    locked = entry.get("settings") or {}
    current = _lock_settings(settings)
    changed = []
    for key in sorted(set(locked) | set(current)):
        before, after = locked.get(key), current.get(key)
        if values_equal(before, after):
            continue
        changed.append({"key": key, "locked": before, "now": after})
    for path, digest in sorted((record.get("files") or {}).items()):
        now = hash_file(Path(path), full=True) if Path(path).is_file() else None
        if now != digest:
            changed.append({"key": f"file:{path}",
                            "locked": str(digest)[:16],
                            "now": (now or "missing")[:16]})
    changed.extend(_gate_deviations(record, gates))
    return changed


def _seen_log(record: Dict[str, Any]) -> Path:
    """The file recording when each difference from a lock was first seen."""
    name = Path(str(record.get("lock_id") or "unnamed")).name
    return _locks_root() / f"{name}.seen.jsonl"


def _note_first_seen(record: Dict[str, Any],
                     deviations: List[Dict[str, Any]]) -> None:
    """Stamp each difference with when it was first seen, and keep that.

    The first check that meets a difference (a run, or the lock dialog
    reading the form) appends it to the lock's ``.seen.jsonl`` log; later
    checks read the time back. That is how an edit made before the key was
    opened is told from one made after it, whatever order the runs came in.

    :param record: the lock.
    :param deviations: its differences now; each gains ``first_seen_utc``.
    """
    if not deviations:
        return
    path = _seen_log(record)
    seen: Dict[Tuple[str, str], str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines:
        try:
            event = json.loads(line)
            seen.setdefault((str(event["key"]), str(event["value"])),
                            str(event["utc"]))
        except (ValueError, KeyError, TypeError):
            continue
    new_events = []
    for deviation in deviations:
        ident = (str(deviation.get("key")), _json_digest(deviation.get("now")))
        if ident not in seen:
            seen[ident] = _utc_now()
            new_events.append({"key": ident[0], "value": ident[1],
                               "utc": seen[ident], "who": _who()})
        deviation["first_seen_utc"] = seen[ident]
    if new_events:
        with open(path, "a", encoding="utf-8") as handle:
            for event in new_events:
                handle.write(json.dumps(event, sort_keys=True) + "\n")


def _lock_summary(result: Dict[str, Any]) -> str:
    """One sentence saying how a run stands against its lock."""
    status = result.get("status", "unlocked")
    if status == "unlocked":
        return "No analysis lock applies to this run."
    text = (f"Analysis lock {str(result.get('sha256') or '')[:16]} "
            f"(locked {result.get('locked_utc')}): "
            f"{_LOCK_STATUS_WORDS.get(status, status)}")
    if result.get("unblinded_utc"):
        text += f" at {result['unblinded_utc']}"
    deviations = list(result.get("deviations") or ())
    mixed = status == "post_hoc" and not all(
        d.get("post_hoc") for d in deviations)
    names = [str(d.get("key")) + (
        (" (after unblinding)" if d.get("post_hoc") else " (before unblinding)")
        if mixed else "") for d in deviations]
    if names:
        text += ": " + ", ".join(names[:8]) + (" …" if len(names) > 8 else "")
    uncovered = list(result.get("uncovered_models") or ())
    if uncovered:
        text += "; models not covered by the lock: " + ", ".join(uncovered)
    return text + "."


def _lock_verdict(record: Dict[str, Any],
                  deviations: List[Dict[str, Any]],
                  uncovered_models: Iterable[str] = ()) -> Dict[str, Any]:
    """Judge a set of differences from a lock.

    :param record: the lock.
    :param deviations: its differences now; stamped with ``first_seen_utc``
        and ``post_hoc`` here.
    :param uncovered_models: models the run recorded that the lock does not
        name.
    :returns: the verdict :func:`check_analysis_lock` documents.
    """
    locked_utc = str(record.get("locked_utc") or "")
    after = [t for t in _lock_unblindings(record) if t > locked_utc]
    _note_first_seen(record, deviations)
    for deviation in deviations:
        deviation["post_hoc"] = bool(
            after and str(deviation.get("first_seen_utc") or "") > after[0])
    if record.get("sha256") != _lock_digest(record):
        status = "tampered"
    elif any(d["post_hoc"] for d in deviations):
        status = "post_hoc"
    elif deviations:
        status = "deviation"
    elif record.get("unblinded_before_lock"):
        status = "not_preregistered"
    else:
        status = "verified"
    result = {
        "status": status,
        "lock_id": record.get("lock_id"),
        "sha256": record.get("sha256"),
        "locked_utc": locked_utc,
        "locked_by": record.get("locked_by"),
        "pipelines": sorted(_lock_entries(record)),
        "deviations": deviations,
        "unblinded_utc": after[0] if after else None,
    }
    uncovered = sorted(set(uncovered_models))
    if uncovered:
        result["uncovered_models"] = uncovered
    result["summary"] = _lock_summary(result)
    return result


def _observe_settings_changes(settings, app_key, keys=None):
    """Record committed setting differences without checking files or gates.

    Only an existing, intact lock on this pipeline and source is eligible.
    ``keys`` scopes a field commit; omitted keys mean a complete bulk load.
    The return value is observation evidence, never a verification verdict.
    """
    record = _find_lock(app_key, (settings or {}).get("src"))
    if not record:
        return None
    if record.get("sha256") != _lock_digest(record):
        raise ValueError("The analysis lock has changed; edit timing was not recorded.")
    entry = _lock_entries(record).get(str(app_key)) or {}
    locked = entry.get("settings") or {}
    current = _lock_settings(settings)
    selected = (set(locked) | set(current)) if keys is None else set(keys)
    selected = {key for key in selected if not str(key).startswith("_")
                and key not in _LOCK_IGNORED_KEYS}
    differences = [{"key": key, "locked": locked.get(key),
                    "now": current.get(key)}
                   for key in sorted(selected)
                   if not values_equal(locked.get(key), current.get(key))]
    _note_first_seen(record, differences)
    return {"lock_id": record.get("lock_id"), "deviations": differences,
            "scope": "committed settings only"}


def check_analysis_lock(settings: Dict[str, Any], *, app_key: str,
                        lock: Optional[Dict[str, Any]] = None,
                        gates: Any = None) -> Dict[str, Any]:
    """Check a run's settings against its preregistered analysis lock.

    Use the supplied analysis plan, or find the newest plan created by
    :func:`lock_analysis` for ``app_key`` and the settings' ``src``.
    Check the plan against its stored hash to detect changes to the file.
    Then compare each setting, each hashed file, and each recorded strategy
    for selecting measurement rows.

    Each difference is stamped with when it was first seen (kept beside the
    lock), and it is post-hoc only when it was first seen after a blinding
    key on a folder of the plan was opened: an edit made while still blind
    stays a deviation even when the run that uses it comes later.

    :param settings: the settings of the run being checked.
    :param app_key: the pipeline running them.
    :param lock: a lock record to check against instead of the newest one.
    :param gates: the Gate Editor gating strategies in use, for gates the
        lock took as objects (see :func:`lock_analysis`).
    :returns: ``status`` -- ``"unlocked"`` (no lock applies),
        ``"verified"`` (nothing changed), ``"deviation"`` (changed, every
        change first seen while still blind), ``"post_hoc"`` (a change first
        seen after a key on the plan's folders was unblinded),
        ``"not_preregistered"`` (unchanged, but the lock was made after an
        unblinding) or ``"tampered"`` (the lock no longer matches its hash)
        -- with ``lock_id``, ``sha256``, ``locked_utc``, ``locked_by``,
        ``pipelines``, ``deviations`` (``key``, ``locked``, ``now``,
        ``first_seen_utc``, ``post_hoc``, and ``detail`` for gates),
        ``unblinded_utc`` and a one-sentence ``summary``.
    """
    record = lock if lock is not None else _find_lock(
        app_key, (settings or {}).get("src"))
    if not record:
        result: Dict[str, Any] = {"status": "unlocked", "deviations": []}
        result["summary"] = _lock_summary(result)
        return result
    return _lock_verdict(
        record, _lock_deviations(record, settings, app_key, gates))


def _gate_file_lock_notes(path: Any, gates: Any = None) -> List[str]:
    """How a saved gate file stands against every lock that holds it.

    What the Gate Editor says after saving or loading a gating strategy.
    Verdicts use the same first-seen timing, lock-integrity and unblinding
    policy as runs and exports. Only this strategy is compared; other locked
    gate files are not opened just to produce its note.

    :param path: the gate file.
    :param gates: the gates to compare, when they are not what the file
        holds; the file's own gates otherwise.
    :returns: one sentence per lock holding the file, oldest lock first;
        empty when no lock holds it.
    """
    target = _existing_file(path)
    if target is None:
        return []
    label = str(target.resolve(strict=False))
    payload = _gate_payload(gates if gates is not None else target)
    notes = []
    for lock_path in sorted(_locks_root().glob("*.json")):
        try:
            record = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        locked = (record.get("gates") or {}).get(label) if isinstance(
            record, dict) else None
        if not locked:
            continue
        scoped = {**record, "gates": {label: locked}}
        deviations = _gate_deviations(scoped, {label: payload})
        verdict = _lock_verdict(record, deviations)
        sha = str(record.get("sha256") or "")[:16]
        if verdict["status"] == "verified":
            notes.append(f"these gates match analysis lock {sha}")
        elif verdict["status"] in ("deviation", "post_hoc"):
            changes = ", ".join(change["detail"] for change in deviations) or "changed"
            word = "post-hoc, after unblinding" if verdict["status"] == "post_hoc" else "a deviation"
            notes.append(f"these gates differ from analysis lock {sha} "
                         f"({changes}): {word}")
        else:
            notes.append(verdict["summary"])
    return notes


def _check_lock_for_run(run: "Run") -> None:
    """Stamp a run with its lock verdict; a failed check never fails a run."""
    try:
        record = _find_lock(run.app_key, (run.settings or {}).get("src"))
        result = check_analysis_lock(run.settings, app_key=run.app_key,
                                     lock=record) if record else None
    except Exception as exc:
        run.provenance_warnings.append(f"analysis lock check failed: {exc}")
        return
    if not result or result.get("status") == "unlocked":
        return
    run._analysis_lock_record = record
    _set_run_lock_verdict(run, result)


def _set_run_lock_verdict(run: "Run", result: Dict[str, Any]) -> None:
    """Put a lock verdict on a run, replacing the warning of an older one."""
    old = getattr(run, "_analysis_lock", None)
    if old and old.get("summary") in run.provenance_warnings:
        run.provenance_warnings.remove(old["summary"])
    run._analysis_lock = result
    if result.get("status") != "verified":
        run.provenance_warnings.append(result["summary"])


def _check_model_for_run(run: "Run", name: str, path: Any) -> None:
    """Check a model a run just recorded against the run's lock.

    Models are recorded while the pipeline runs, after the settings were
    checked, so this re-judges the run: a locked model whose file differs,
    or a model the lock does not name when the lock names models, is a
    difference like any other. A model the lock cannot speak to (the lock
    names no models, and the file is not among its hashed files) is listed
    as not covered, without changing the verdict. A failure here is only
    logged: checking a model must never fail a run.

    :param run: the run, already checked by :func:`_check_lock_for_run`.
    :param name: the name the model was recorded under.
    :param path: its checkpoint file.
    """
    record = getattr(run, "_analysis_lock_record", None)
    result = getattr(run, "_analysis_lock", None)
    if not record or not result:
        return
    try:
        now = (run.model_files.get(name) or {}).get("sha256")
        resolved = str(Path(str(path)).resolve(strict=False))
        models = record.get("models") or {}
        deviations = [d for d in result.get("deviations") or ()
                      if d.get("key") != f"model:{name}"]
        uncovered = list(result.get("uncovered_models") or ())
        if name in models:
            locked = models[name].get("sha256")
            if locked and locked == now:
                return
        elif (resolved in (record.get("files") or {})
              or (now and now in {m.get("sha256") for m in models.values()})):
            return
        elif not models:
            uncovered.append(str(name))
            _set_run_lock_verdict(run, _lock_verdict(
                record, list(result.get("deviations") or ()), uncovered))
            return
        else:
            locked = None
        deviations.append({"key": f"model:{name}",
                           "locked": str(locked)[:16] if locked else None,
                           "now": (now or "unreadable")[:16]})
        _set_run_lock_verdict(run, _lock_verdict(record, deviations,
                                                 uncovered))
    except Exception as exc:
        LOG.warning("model %r could not be checked against the analysis "
                    "lock: %s", name, exc)


_PRUNE_KINDS = ("logs", "run_logs", "run_folders")
_PRUNE_DEFAULTS = {"logs": (30, 200), "run_logs": (30, 500),
                   "run_folders": (90, 2000)}
_DAY_SECONDS = 86400.0


def _tree_stats(path: Path) -> Tuple[int, float]:
    """Return the bytes below ``path`` and the newest modification time.

    Symbolic links are counted as links and never followed, so a link to a
    large folder elsewhere neither inflates the size nor is walked.

    :param path: a file or folder.
    """
    try:
        info = os.lstat(path)
    except OSError:
        return 0, 0.0
    size, newest = int(info.st_size), float(info.st_mtime)
    if not os.path.isdir(path) or os.path.islink(path):
        return size, newest
    stack = [str(path)]
    while stack:
        folder = stack.pop()
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    try:
                        stat = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    size += int(stat.st_size)
                    newest = max(newest, float(stat.st_mtime))
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
        except OSError:
            continue
    return size, newest


def _prune_root(kind: str) -> Path:
    """Return the folder a kind of home-folder storage lives in.

    :param kind: ``"logs"`` for the daily log files, ``"run_logs"`` for the
        per-run log files, ``"run_folders"`` for the run journal's folders.
    """
    if kind == "logs":
        from .logging_util import log_dir
        return Path(log_dir())
    if kind == "run_logs":
        from .runctx import runs_log_dir
        return Path(runs_log_dir())
    if kind == "run_folders":
        return runs_root()
    raise ValueError(f"unknown storage kind: {kind!r}")


def _is_prunable_name(kind: str, entry: Path) -> bool:
    """Whether ``entry`` is something pruning may consider for ``kind``.

    Daily logs are the ``spacr*.log`` files and their rotated copies; other
    files in the log folder are left alone. Per-run logs are ``.jsonl`` and
    ``.resources.json`` files. Run folders are real folders, never links.

    :param kind: one of the storage kinds.
    :param entry: a child of the kind's folder.
    """
    name = entry.name
    if kind == "logs":
        return (name.startswith("spacr") and ".log" in name
                and entry.is_file() and not entry.is_symlink())
    if kind == "run_logs":
        return ((name.endswith(".jsonl") or name.endswith(".resources.json"))
                and entry.is_file() and not entry.is_symlink())
    return entry.is_dir() and not entry.is_symlink()


def _protected_storage() -> Set[str]:
    """Return resolved paths and run ids pruning must never delete.

    The run open on this thread, the run id this process logs under, and
    every log file a logging handler holds open.
    """
    protected: Set[str] = set()
    try:
        run = current_run()
        if run is not None and getattr(run, "dir", None):
            protected.add(str(Path(run.dir).resolve()))
            protected.add(Path(run.dir).name)
        from .runctx import current_run_id
        run_id = current_run_id()
        if run_id:
            protected.add(run_id)
    except Exception:
        LOG.debug("Could not identify the running run", exc_info=True)
    loggers = [logging.getLogger()] + [
        logger for logger in list(logging.Logger.manager.loggerDict.values())
        if isinstance(logger, logging.Logger)]
    for logger in loggers:
        for handler in list(getattr(logger, "handlers", ())):
            name = getattr(handler, "baseFilename", None)
            if name:
                try:
                    protected.add(str(Path(name).resolve()))
                except OSError:
                    protected.add(str(name))
    return protected


def _is_protected(entry: Path, protected: Set[str]) -> bool:
    """Whether ``entry`` is the current run, its log, or an open log file.

    :param entry: a candidate.
    :param protected: from :func:`_protected_storage`.
    """
    try:
        resolved = str(entry.resolve())
    except OSError:
        resolved = str(entry)
    if resolved in protected or entry.name in protected:
        return True
    stem = entry.name.split(".", 1)[0]
    return bool(stem) and stem in protected


def _prune_plan(kind: str, keep_days: float, cap_mb: float, *,
                now: Optional[float] = None) -> Dict[str, Any]:
    """Work out what pruning one kind of storage would delete. Reads only.

    Nothing modified within the last ``keep_days`` days is ever chosen, and
    neither is the current run or a log file that is open. Among the older
    entries, the oldest go first. With a size cap above zero, entries are
    chosen only while the folder is over the cap; with a cap of zero, every
    older entry is chosen.

    :param kind: ``"logs"``, ``"run_logs"`` or ``"run_folders"``.
    :param keep_days: entries newer than this many days are kept; at least 1.
    :param cap_mb: size cap in megabytes; ``0`` for age alone.
    :param now: the time to measure age from; now when ``None``.
    :returns: ``kind``, ``root``, ``count``, ``total`` bytes, ``cutoff``
        time, and ``delete``, a list of ``(path, bytes)`` oldest first.
    """
    root = _prune_root(kind)
    now = time.time() if now is None else float(now)
    cutoff = now - max(1.0, float(keep_days)) * _DAY_SECONDS
    protected = _protected_storage()
    entries = []
    try:
        children = list(root.iterdir())
    except OSError:
        children = []
    for entry in children:
        try:
            if not _is_prunable_name(kind, entry):
                continue
        except OSError:
            continue
        size, newest = _tree_stats(entry)
        entries.append((newest, entry, size))
    entries.sort(key=lambda item: (item[0], item[1].name))
    total = sum(size for _newest, _entry, size in entries)
    cap = max(0.0, float(cap_mb)) * 1024 * 1024
    remaining, chosen = total, []
    for newest, entry, size in entries:
        if newest >= cutoff:
            break
        if cap and remaining <= cap:
            break
        if _is_protected(entry, protected):
            continue
        chosen.append((str(entry), size))
        remaining -= size
    return {"kind": kind, "root": str(root), "count": len(entries),
            "total": total, "cutoff": cutoff, "delete": chosen}


def _prune(plan: Dict[str, Any]) -> Tuple[int, int, List[str]]:
    """Delete what :func:`_prune_plan` chose, checking each entry again.

    An entry is deleted only if it is still directly inside the plan's
    folder, is still older than the plan's cut-off, and is not the current
    run or an open log file. Run folders go through :func:`delete_runs`.

    :param plan: a plan from :func:`_prune_plan`.
    :returns: ``(deleted, bytes freed, refusals)``.
    """
    kind = plan["kind"]
    root = _prune_root(kind).resolve()
    protected = _protected_storage()
    deleted, freed, refused = 0, 0, []
    for raw, size in plan.get("delete", ()):
        entry = Path(raw)
        try:
            if entry.resolve().parent != root or not _is_prunable_name(kind, entry):
                refused.append(f"{entry.name}: not in {root}")
                continue
        except OSError as error:
            refused.append(f"{entry.name}: {error}")
            continue
        if _is_protected(entry, protected):
            refused.append(f"{entry.name}: in use")
            continue
        if _tree_stats(entry)[1] >= float(plan["cutoff"]):
            refused.append(f"{entry.name}: changed since it was listed")
            continue
        if kind == "run_folders":
            done, why = delete_runs([entry])
            refused.extend(why)
            if done:
                deleted += done
                freed += int(size)
            continue
        try:
            entry.unlink()
        except OSError as error:
            refused.append(f"{entry.name}: {error}")
            continue
        deleted += 1
        freed += int(size)
    return deleted, freed, refused


_CACHE_LOCATIONS_FILE = "cache_locations.json"
_CACHE_ROWS = (
    ("models", "spaCR models", ""),
    ("cellpose", "Cellpose models", "CELLPOSE_LOCAL_MODELS_PATH"),
    ("huggingface", "Hugging Face", "HF_HOME"),
    ("torch", "Torch", "TORCH_HOME"),
    ("backends", "Backend environments", "SPACR_BACKENDS_DIR"),
    ("news", "News", "SPACR_NEWS_CACHE"),
)


def _cache_locations_path() -> Path:
    """Return the file recording the caches spaCR relocated."""
    return _spacr_home() / _CACHE_LOCATIONS_FILE


def _read_cache_locations() -> Dict[str, str]:
    """Return ``{environment variable: folder}`` for every relocated cache."""
    try:
        data = json.loads(_cache_locations_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def _apply_cache_locations() -> int:
    """Point this process at every relocated cache. Returns how many.

    A variable already set in the environment wins over the record.
    """
    applied = 0
    for name, folder in _read_cache_locations().items():
        if name and folder and not os.environ.get(name):
            os.environ[name] = folder
            applied += 1
    return applied


def _cache_folder(key: str) -> Path:
    """Return the folder one cache lives in now.

    :param key: a key of :data:`_CACHE_ROWS`.
    """
    home = Path.home()
    spacr_home = _spacr_home()
    xdg_cache = os.environ.get("XDG_CACHE_HOME", "").strip()
    cache = Path(xdg_cache) if xdg_cache else home / ".cache"
    env = dict((k, v) for k, _label, v in _CACHE_ROWS).get(key)
    if env is None:
        raise ValueError(f"unknown cache: {key!r}")
    configured = os.environ.get(env, "").strip() if env else ""
    if configured:
        return Path(configured).expanduser()
    defaults = {
        "models": spacr_home / "models",
        "cellpose": home / ".cellpose" / "models",
        "huggingface": cache / "huggingface",
        "torch": cache / "torch",
        "backends": spacr_home / "backends",
        "news": spacr_home / "news",
    }
    return defaults[key]


def _disk_caches() -> List[Dict[str, Any]]:
    """List the model, Hugging Face, Torch, backend and news caches.

    :returns: one dict per cache with ``key``, ``label``, ``path``, ``env``,
        ``exists``, ``size`` in bytes and ``relocatable``.
    """
    rows = []
    for key, label, env in _CACHE_ROWS:
        folder = _cache_folder(key)
        exists = folder.is_dir()
        rows.append({"key": key, "label": label, "path": str(folder),
                     "env": env, "exists": exists,
                     "size": _tree_stats(folder)[0] if exists else 0,
                     "relocatable": bool(env)})
    return rows


def _cache_refusal(folder: Path) -> Optional[str]:
    """Return why ``folder`` must not be emptied or moved, or ``None``.

    :param folder: a cache folder.
    """
    try:
        target = folder.resolve()
    except OSError as error:
        return str(error)
    home = Path.home().resolve()
    spacr_home = _spacr_home().resolve()
    if target == Path(target.anchor) or target == home or target in home.parents:
        return f"{target}: refusing a home or top-level folder"
    if target == spacr_home or target in spacr_home.parents:
        return f"{target}: refusing spaCR's own home folder"
    return None


def _clear_cache(key: str) -> Tuple[int, List[str]]:
    """Empty one cache folder, keeping the folder itself.

    Links inside are removed as links; what they point to is not touched.

    :param key: a key of :data:`_CACHE_ROWS`.
    :returns: ``(entries removed, refusals)``.
    """
    folder = _cache_folder(key)
    if not folder.is_dir():
        return 0, []
    why = _cache_refusal(folder)
    if why:
        return 0, [why]
    removed, refused = 0, []
    for child in list(folder.iterdir()):
        try:
            if child.is_symlink() or not child.is_dir():
                child.unlink()
            else:
                shutil.rmtree(child)
            removed += 1
        except OSError as error:
            refused.append(f"{child.name}: {error}")
    return removed, refused


def _relocate_cache(key: str, parent: Any) -> Path:
    """Move one cache into ``parent/spacr-<key>`` and remember it there.

    The move works across drives. The new place is recorded in
    ``cache_locations.json`` in spaCR's home folder and applied to this process; spaCR
    applies it again at every start. A cache whose variable was set outside
    spaCR is not moved.

    :param key: a key of :data:`_CACHE_ROWS` that has a variable.
    :param parent: the folder to move it into.
    :returns: the cache's new folder.
    :raises ValueError: when the cache cannot be moved there.
    """
    env = dict((k, v) for k, _label, v in _CACHE_ROWS).get(key)
    if not env:
        raise ValueError(f"{key}: this cache has no setting to move it with")
    recorded = _read_cache_locations()
    outside = os.environ.get(env, "").strip()
    if outside and outside != recorded.get(env, ""):
        raise ValueError(f"{env} is set outside spaCR; change it there")
    source = _cache_folder(key)
    target = Path(str(parent)).expanduser().resolve() / f"spacr-{key}"
    why = _cache_refusal(source) if source.exists() else None
    if why:
        raise ValueError(why)
    if target.exists() and any(target.iterdir()):
        raise ValueError(f"{target} already holds files")
    src = source.resolve() if source.exists() else source
    if src == target or src in target.parents or target in src.parents:
        raise ValueError(f"{target} overlaps {src}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        if target.exists():
            target.rmdir()
        shutil.move(str(source), str(target))
    else:
        target.mkdir(parents=True, exist_ok=True)
    recorded[env] = str(target)
    path = _cache_locations_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(path, json.dumps(recorded, indent=2, sort_keys=True))
    os.environ[env] = str(target)
    return target


try:
    _apply_cache_locations()
except Exception:
    LOG.debug("Relocated caches could not be applied", exc_info=True)
