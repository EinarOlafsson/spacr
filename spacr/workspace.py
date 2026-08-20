"""
Workspace bundles — what was OPEN around a run, saved beside the run itself.

:mod:`spacr.run_journal` records what a pipeline was GIVEN: the settings, the
environment, the hashes, the log. That is a reproducibility record and it is
complete for what it covers. It says nothing about the workspace the user
assembled around the run — the databases they attached, the montage they
generated, the level and colouring and thresholds they set on the volcano —
because none of that is in the settings dict. It lives in widgets and dies
with the process.

This module writes that half::

    ~/.spacr/runs/<run>/workspace.json
    ~/.spacr/runs/<run>/workspace/files/<digest>__<name>   # 'copy' mode only

Public API::

    from spacr import workspace

    workspace.register("volcano", lambda: the_panel)     # GUI, once
    doc = workspace.collect(workspace.providers())
    workspace.save(run_dir, doc, mode="reference")

    doc = workspace.load(run_dir)
    report = workspace.restore(workspace.providers(), doc)

THE PANELS ARE ASKED, NEVER INSPECTED. A collector that reached into widgets
and read private attributes would be a second copy of every panel's state
model, and would rot the first time one of them changed. A contributor is
anything with ``workspace_state() -> dict`` and
``apply_workspace_state(dict) -> bool``; the regression panel's existing
``plot_state`` / ``apply_plot_state`` pair is already that shape.

REFERENCE ALWAYS, COPY ON REQUEST. Every file the workspace names is recorded
with its size, mtime and SHA-256 — kilobytes, and enough for a restore to say
"that database is not where it was" instead of failing obscurely. The BYTES
are copied only when asked, because a spaCR source folder is tens to hundreds
of gigabytes of TIFFs and a measurements database is routinely several GB;
copying those on every run would fill the disk with duplicates of data the
user already has. A section may still mark individual files to carry
regardless — figures a session generated exist nowhere else.

Nothing is ever skipped silently: a file over the size limit is written into
the document as skipped, with its size and the limit that excluded it.

Standard library only, and deliberately: this is imported from the run
journal, which pipelines import, and neither may pay for pandas or Qt.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

LOG = logging.getLogger("spacr.workspace")

SCHEMA_VERSION = 1
"""On-disk schema of ``workspace.json``."""

DOC_NAME = "workspace.json"
FILES_DIR = "workspace/files"

MODES = ("off", "reference", "copy")
"""How much of the workspace a run carries.

``off``        nothing is written; the run folder is what it is today.
``reference``  the document, with every file recorded but only the files a
               section explicitly marks carried. The default.
``copy``       the same, plus the bytes of every recorded file under the
               per-file size limit.
"""

DEFAULT_MODE = "reference"
DEFAULT_COPY_LIMIT_MB = 512

#: Reserved key. A section may list files it wants CARRIED rather than merely
#: recorded, as ``[{"role": str, "path": str}, ...]``. Used for artifacts the
#: session generated — figures, exported tables — which exist nowhere else.
CARRY_KEY = "_workspace_carry"

#: Section keys that are metadata about the section rather than state to put
#: back, and are therefore not handed to ``apply_workspace_state``.
_RESERVED = (CARRY_KEY,)

_LOCK = threading.RLock()
_PROVIDERS: "Dict[str, Callable[[], Any]]" = {}
_DEFAULT_MODE = DEFAULT_MODE
_DEFAULT_LIMIT_MB = float(DEFAULT_COPY_LIMIT_MB)


def set_default_mode(mode: Any, copy_limit_mb: Any = None) -> str:
    """Set the process-wide default, and return what it resolved to.

    The preference lives in ``QSettings``, which the run journal cannot read:
    pipelines import the journal and must not pay for Qt. So the application
    pushes its answer down here at startup, and the journal asks a module both
    ends already import. A settings dict that names ``save_workspace``
    outranks this — a scripted run says what it wants.
    """
    global _DEFAULT_MODE, _DEFAULT_LIMIT_MB
    with _LOCK:
        _DEFAULT_MODE = resolve_mode(mode)
        if copy_limit_mb is not None:
            try:
                limit = float(copy_limit_mb)
                if limit >= 0:
                    _DEFAULT_LIMIT_MB = limit
            except (TypeError, ValueError):
                pass
        return _DEFAULT_MODE


def default_mode() -> str:
    """The process-wide default mode."""
    with _LOCK:
        return _DEFAULT_MODE


def default_copy_limit_mb() -> float:
    """The process-wide default per-file copy limit, in megabytes."""
    with _LOCK:
        return float(_DEFAULT_LIMIT_MB)


# ---------------------------------------------------------------------------
# The registry — how GUI state reaches a journal that cannot import Qt
# ---------------------------------------------------------------------------

def register(name: str, provider: Callable[[], Any]) -> None:
    """Register a contributor under ``name``.

    :param provider: a zero-argument callable returning the contributor —
        a widget with ``workspace_state()``, or a plain dict. A CALLABLE and
        not the object, because the panel a name refers to is rebuilt over a
        session's life and a captured reference would go stale (and, being a
        Qt object, would keep a deleted C++ peer alive to raise
        ``RuntimeError`` at collection time).
    """
    if not name:
        raise ValueError("a workspace contributor needs a name")
    if not callable(provider):
        raise TypeError("a workspace provider must be callable")
    with _LOCK:
        _PROVIDERS[str(name)] = provider


def unregister(name: str) -> bool:
    """Drop a contributor. Returns whether there was one."""
    with _LOCK:
        return _PROVIDERS.pop(str(name), None) is not None


def providers() -> "Dict[str, Callable[[], Any]]":
    """A snapshot of the registered contributors."""
    with _LOCK:
        return dict(_PROVIDERS)


def clear_providers() -> None:
    """Forget every contributor and the pushed default.

    For tests and for application shutdown. The default goes with them: a
    process that has torn its workspace down has no application left whose
    preference it would be.
    """
    global _DEFAULT_MODE, _DEFAULT_LIMIT_MB
    with _LOCK:
        _PROVIDERS.clear()
        _DEFAULT_MODE = DEFAULT_MODE
        _DEFAULT_LIMIT_MB = float(DEFAULT_COPY_LIMIT_MB)


# ---------------------------------------------------------------------------
# Collecting
# ---------------------------------------------------------------------------

def _state_of(source: Any) -> Optional[Dict[str, Any]]:
    """The state dict a contributor offers, or ``None`` if it offers none."""
    if source is None:
        return None
    if isinstance(source, Mapping):
        return dict(source)
    getter = getattr(source, "workspace_state", None)
    if callable(getter):
        state = getter()
        return dict(state) if isinstance(state, Mapping) else None
    # The regression panel already had this pair before the workspace
    # existed. Taking it as-is keeps ONE state model for the volcano rather
    # than a second one that has to be remembered alongside it.
    legacy = getattr(source, "plot_state", None)
    if callable(legacy):
        state = legacy()
        return dict(state) if isinstance(state, Mapping) else None
    return None


def section_states(
    contributors: Mapping[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """Ask every contributor for its slice.

    :param contributors: ``{name: provider-or-object}``. A callable value is
        called; anything else is used directly.
    :returns: ``(sections, problems)``. A contributor that raises is left out
        and NAMED in ``problems`` — one panel failing to describe itself must
        not cost the user the other five.
    """
    sections: Dict[str, Any] = {}
    problems: List[Dict[str, str]] = []
    for name in sorted(contributors):
        source = contributors[name]
        try:
            if callable(source) and not isinstance(source, Mapping):
                source = source()
            # NOT OPEN IS NOT A PROBLEM. Every screen registers the same
            # panels and most screens build none of them -- a measure screen
            # has no volcano. Reporting those would bury the one section that
            # genuinely failed under a dozen that were simply not there.
            if source is None:
                continue
            state = _state_of(source)
        except Exception as exc:                      # noqa: BLE001
            problems.append({"section": name, "why": f"{type(exc).__name__}: {exc}"})
            LOG.debug("workspace section %r could not be collected", name, exc_info=True)
            continue
        if state is None:
            problems.append({"section": name, "why": "offers no workspace state"})
            continue
        sections[name] = _jsonable(state)
    return sections, problems


def _jsonable(value: Any, _depth: int = 0) -> Any:
    """A JSON-writable copy of ``value``.

    Tuples become lists, Paths and everything exotic become strings. Depth is
    bounded because a widget state that accidentally contains a cycle must
    cost a truncated section, not the process.
    """
    if _depth > 12:
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=str) if isinstance(value, (set, frozenset)) else value
        return [_jsonable(v, _depth + 1) for v in items]
    return str(value)


_PATHISH = re.compile(r"[/\\]")


def _looks_like_a_path(value: Any) -> bool:
    """Whether a section value is worth asking the filesystem about.

    Cheap and deliberately conservative: a separator and a plausible length.
    The filesystem decides in the end — this only keeps the walk from calling
    ``stat`` on every gene name in a picked-cell list.
    """
    if not isinstance(value, str) or not (2 < len(value) < 4096):
        return False
    return bool(_PATHISH.search(value))


def _walk_strings(value: Any, trail: str = "", _depth: int = 0) -> Iterator[Tuple[str, str]]:
    """Yield ``(dotted-key, string)`` for every string inside ``value``."""
    if _depth > 12:
        return
    if isinstance(value, str):
        yield trail, value
    elif isinstance(value, Mapping):
        for k, v in value.items():
            if k in _RESERVED:
                continue
            yield from _walk_strings(v, f"{trail}.{k}" if trail else str(k), _depth + 1)
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from _walk_strings(v, f"{trail}[{i}]", _depth + 1)


def hash_file(path: Path, chunk_size: int = 1 << 20) -> Optional[str]:
    """A file's full SHA-256, or ``None`` if it cannot be read."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as exc:                          # noqa: BLE001
        LOG.debug("could not hash %s: %s", path, exc)
        return None


def _file_record(role: str, path: Path, *, want_hash: bool = True) -> Dict[str, Any]:
    """What is recorded about one file, whether or not its bytes are carried."""
    record: Dict[str, Any] = {"role": role, "path": str(path)}
    try:
        stat = path.stat()
    except Exception:                                 # noqa: BLE001
        record["exists"] = False
        return record
    record["exists"] = True
    if path.is_dir():
        record["kind"] = "directory"
        return record
    record["kind"] = "file"
    record["size"] = int(stat.st_size)
    record["mtime"] = round(stat.st_mtime, 3)
    if want_hash:
        digest = hash_file(path)
        if digest:
            record["sha256"] = digest
    return record


def _carried(section: Any) -> List[Dict[str, str]]:
    """The files a section explicitly asks to have carried."""
    if not isinstance(section, Mapping):
        return []
    out = []
    for entry in section.get(CARRY_KEY) or []:
        if isinstance(entry, Mapping) and entry.get("path"):
            out.append({"role": str(entry.get("role") or "carried"),
                        "path": str(entry["path"])})
        elif isinstance(entry, str):
            out.append({"role": "carried", "path": entry})
    return out


def inventory(sections: Mapping[str, Any], *, hash_files: bool = True) -> List[Dict[str, Any]]:
    """Record every file the sections name.

    Two ways in, and both are needed. The WALK finds every string in the
    document that turns out to name something on disk, so a panel that gains
    a new path key is covered without anyone remembering to declare it — the
    failure mode that actually happens. The CARRY LIST is explicit, because
    wanting the bytes is a decision a panel makes about its own artifacts and
    cannot be guessed from the value.
    """
    seen: Dict[str, Dict[str, Any]] = {}

    def add(role: str, raw: str, carry: bool) -> None:
        try:
            path = Path(raw).expanduser()
        except Exception:                             # noqa: BLE001
            return
        key = str(path)
        record = seen.get(key)
        if record is None:
            record = _file_record(role, path, want_hash=hash_files)
            if not record.get("exists"):
                return
            seen[key] = record
        if carry:
            record["carry"] = True

    for name in sorted(sections):
        section = sections[name]
        for entry in _carried(section):
            add(f"{name}:{entry['role']}", entry["path"], True)
        for trail, value in _walk_strings(section, name):
            if _looks_like_a_path(value):
                add(trail, value, False)
    return [seen[k] for k in sorted(seen)]


def collect(
    contributors: Mapping[str, Any],
    *,
    app_key: str = "",
    saved: str = "",
    hash_files: bool = True,
) -> Dict[str, Any]:
    """Build the workspace document. Writes nothing.

    :param saved: the timestamp to stamp, ISO-8601. Injected rather than read
        from the clock so a caller can produce a byte-identical document
        twice, which is what makes the writer testable.
    """
    sections, problems = section_states(contributors)
    doc: Dict[str, Any] = {
        "version": SCHEMA_VERSION,
        "saved": saved or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "app_key": str(app_key or ""),
        "sections": sections,
        "files": inventory(sections, hash_files=hash_files),
    }
    if problems:
        doc["problems"] = problems
    return doc


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def resolve_mode(value: Any) -> str:
    """Normalise a mode. Anything unrecognised is the default, not an error.

    A settings dict is user-editable and a typo there must not cost the run;
    the mode only decides how much is saved.
    """
    if value is None:
        return DEFAULT_MODE
    if isinstance(value, bool):
        return "reference" if value else "off"
    text = str(value).strip().lower()
    if text in MODES:
        return text
    if text in ("none", "no", "false", "0", ""):
        return "off"
    if text in ("all", "full", "yes", "true", "1"):
        return "copy"
    LOG.debug("unrecognised save_workspace=%r, using %s", value, DEFAULT_MODE)
    return DEFAULT_MODE


def mode_from_settings(settings: Mapping[str, Any]) -> str:
    """The mode a settings dict asks for."""
    if not isinstance(settings, Mapping) or settings.get("save_workspace") is None:
        return default_mode()
    return resolve_mode(settings.get("save_workspace"))


def copy_limit_from_settings(settings: Mapping[str, Any]) -> float:
    """The per-file copy limit in megabytes."""
    if isinstance(settings, Mapping) and settings.get("workspace_copy_limit_mb") is not None:
        try:
            limit = float(settings["workspace_copy_limit_mb"])
            if limit >= 0:
                return limit
        except (TypeError, ValueError):
            pass
    return default_copy_limit_mb()


def _safe_name(path: Path) -> str:
    """A filename that cannot escape the bundle or collide by basename."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.name)[:80] or "file"
    return stem


def save(
    run_dir: Any,
    doc: Mapping[str, Any],
    *,
    mode: str = DEFAULT_MODE,
    copy_limit_mb: float = DEFAULT_COPY_LIMIT_MB,
) -> Optional[Path]:
    """Write ``doc`` into ``run_dir``. Returns the document path, or ``None``.

    ``off`` writes nothing at all — not an empty document — so a run folder
    saved with the feature disabled is byte-for-byte what it was before this
    module existed.

    Copying happens here rather than in :func:`collect` because it is the
    only part that touches the user's disk, and a caller that only wants to
    look at the document should never pay for it.
    """
    mode = resolve_mode(mode)
    if mode == "off":
        return None
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    payload = dict(doc)
    payload["mode"] = mode
    limit_bytes = float(copy_limit_mb) * 1024 * 1024

    files = [dict(f) for f in payload.get("files") or []]
    if files:
        files_root = root / FILES_DIR
        for record in files:
            if not record.get("exists") or record.get("kind") != "file":
                continue
            wanted = mode == "copy" or record.get("carry")
            if not wanted:
                continue
            size = record.get("size") or 0
            # The limit bounds `copy`; a file a section asked to CARRY is
            # carried whatever its size, because the section is asserting it
            # exists nowhere else and a silently dropped figure is worse than
            # a large run folder.
            if mode == "copy" and not record.get("carry") and size > limit_bytes:
                record["copied"] = None
                record["skipped"] = (
                    f"{size / 1024 / 1024:.1f} MB is over the "
                    f"{copy_limit_mb:g} MB per-file limit"
                )
                continue
            source = Path(record["path"])
            digest = record.get("sha256") or hash_file(source) or ""
            target_name = f"{digest[:16]}__{_safe_name(source)}" if digest else _safe_name(source)
            target = files_root / target_name
            try:
                files_root.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copy2(source, target)
                record["copied"] = f"{FILES_DIR}/{target_name}"
            except Exception as exc:                  # noqa: BLE001
                record["copied"] = None
                record["skipped"] = f"could not copy: {type(exc).__name__}: {exc}"
                LOG.debug("workspace could not copy %s", source, exc_info=True)
        payload["files"] = files

    path = root / DOC_NAME
    text = json.dumps(payload, indent=2, sort_keys=False, default=str)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    LOG.info("workspace saved [%s] → %s", mode, path)
    return path


def save_for_run(
    run_dir: Any,
    settings: Optional[Mapping[str, Any]] = None,
    *,
    app_key: str = "",
    contributors: Optional[Mapping[str, Any]] = None,
) -> Optional[Path]:
    """Collect the registered workspace and write it beside a finished run.

    Returns the document path, or ``None`` when nothing was written.

    NOTHING IS WRITTEN WHEN THERE IS NOTHING OPEN, and that is what makes the
    default safe to leave on. A command-line pipeline registers no
    contributors, so an empty document in every run folder would be pure
    noise -- a file whose only content is that the feature exists. The
    application registers its panels and gets the bundle; ``spacr mask`` in a
    terminal gets exactly the run folder it got before.
    """
    mode = mode_from_settings(settings or {})
    if mode == "off":
        return None
    sources = providers() if contributors is None else contributors
    if not sources:
        return None
    doc = collect(sources, app_key=app_key)
    if not doc.get("sections") and not doc.get("files"):
        return None
    return save(run_dir, doc, mode=mode,
                copy_limit_mb=copy_limit_from_settings(settings or {}))


def load(run_dir: Any) -> Optional[Dict[str, Any]]:
    """Read a run folder's workspace document, or ``None`` if it has none."""
    path = Path(run_dir)
    if path.is_file():
        path = path if path.name == DOC_NAME else path.parent / DOC_NAME
    else:
        path = path / DOC_NAME
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:                          # noqa: BLE001
        LOG.warning("could not read %s: %s", path, exc)
        return None
    return doc if isinstance(doc, dict) else None


def has_workspace(run_dir: Any) -> bool:
    """Whether ``run_dir`` carries a workspace worth offering to restore."""
    try:
        return (Path(run_dir) / DOC_NAME).is_file()
    except Exception:                                 # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Putting it back
# ---------------------------------------------------------------------------

def check_files(doc: Mapping[str, Any], *, run_dir: Any = None) -> List[Dict[str, Any]]:
    """Say what became of every file the document names.

    ``state`` is one of ``present`` (there, and the same bytes),
    ``changed`` (there, different digest), ``carried`` (gone from its
    original place but inside the bundle), or ``missing``.
    """
    out: List[Dict[str, Any]] = []
    root = Path(run_dir) if run_dir is not None else None
    for record in doc.get("files") or []:
        if not isinstance(record, Mapping):
            continue
        path = Path(str(record.get("path", "")))
        entry = {"role": record.get("role", ""), "path": str(path)}
        copied = record.get("copied")
        bundled = (root / copied) if (root is not None and copied) else None
        if path.exists():
            digest = record.get("sha256")
            if record.get("kind") == "file" and digest:
                entry["state"] = "present" if hash_file(path) == digest else "changed"
            else:
                entry["state"] = "present"
        elif bundled is not None and bundled.exists():
            entry["state"] = "carried"
            entry["path"] = str(bundled)
        else:
            entry["state"] = "missing"
        if record.get("skipped"):
            entry["skipped"] = record["skipped"]
        out.append(entry)
    return out


def restore(
    contributors: Mapping[str, Any],
    doc: Mapping[str, Any],
    *,
    run_dir: Any = None,
) -> Dict[str, Any]:
    """Hand each section back to the contributor that owns it.

    Best-effort and LOUD ABOUT IT. A section whose panel is not present, or
    which the panel declines, is named in the report rather than dropped: a
    workspace that restored four of six panels and said nothing would leave
    the user looking at a screen they believe is the old one, which is worse
    than not restoring at all.
    """
    report: Dict[str, Any] = {"restored": [], "skipped": [], "files": []}
    sections = doc.get("sections") if isinstance(doc, Mapping) else None
    if not isinstance(sections, Mapping):
        report["skipped"].append({"section": "", "why": "no sections in the document"})
        return report
    for name in sorted(sections):
        state = sections[name]
        if not isinstance(state, Mapping):
            report["skipped"].append({"section": name, "why": "not a state document"})
            continue
        source = contributors.get(name)
        if source is None:
            report["skipped"].append({"section": name, "why": "nothing on screen owns it"})
            continue
        payload = {k: v for k, v in state.items() if k not in _RESERVED}
        try:
            if callable(source) and not isinstance(source, Mapping):
                source = source()
            setter = getattr(source, "apply_workspace_state", None)
            if not callable(setter):
                setter = getattr(source, "apply_plot_state", None)
            if not callable(setter):
                report["skipped"].append(
                    {"section": name, "why": "cannot take a workspace state back"})
                continue
            applied = setter(payload)
        except Exception as exc:                      # noqa: BLE001
            report["skipped"].append(
                {"section": name, "why": f"{type(exc).__name__}: {exc}"})
            LOG.debug("workspace section %r could not be restored", name, exc_info=True)
            continue
        # `False` is an ANSWER, not a failure: a panel with no table yet
        # cannot take a plot state and says so. It is reported as skipped
        # because from the user's side nothing was put back either way.
        if applied is False:
            report["skipped"].append({"section": name, "why": "the panel declined it"})
        else:
            report["restored"].append(name)
    report["files"] = check_files(doc, run_dir=run_dir)
    return report


# ---------------------------------------------------------------------------
# Saying what is in one
# ---------------------------------------------------------------------------

def _human_size(n: Any) -> str:
    try:
        size = float(n)
    except (TypeError, ValueError):
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def inventory_text(doc: Mapping[str, Any], *, run_dir: Any = None) -> str:
    """A human-readable inventory of a workspace document."""
    if not isinstance(doc, Mapping):
        return "no workspace document"
    lines = [
        f"workspace v{doc.get('version', '?')} "
        f"[{doc.get('mode', DEFAULT_MODE)}] saved {doc.get('saved', '?')}",
    ]
    if doc.get("app_key"):
        lines.append(f"  app: {doc['app_key']}")
    sections = doc.get("sections") or {}
    lines.append(f"  sections ({len(sections)}):")
    for name in sorted(sections):
        state = sections[name]
        keys = len(state) if isinstance(state, Mapping) else 0
        lines.append(f"    {name:<24} {keys} key{'' if keys == 1 else 's'}")
    files = doc.get("files") or []
    states = {e["path"]: e.get("state", "") for e in check_files(doc, run_dir=run_dir)}
    lines.append(f"  files ({len(files)}):")
    for record in files:
        if not isinstance(record, Mapping):
            continue
        path = str(record.get("path", ""))
        bits = [states.get(path, "")]
        if record.get("copied"):
            bits.append("carried")
        if record.get("skipped"):
            bits.append(f"skipped: {record['skipped']}")
        if record.get("size") is not None:
            bits.append(_human_size(record.get("size")))
        note = ", ".join(b for b in bits if b)
        lines.append(f"    {path}  ({note})")
    for problem in doc.get("problems") or []:
        lines.append(f"  ! {problem.get('section', '?')}: {problem.get('why', '')}")
    return "\n".join(lines)


def report_text(report: Mapping[str, Any]) -> str:
    """A human-readable restore report — what came back and what did not."""
    restored = report.get("restored") or []
    skipped = report.get("skipped") or []
    lines = []
    if restored:
        lines.append(f"restored: {', '.join(restored)}")
    else:
        lines.append("restored: nothing")
    for entry in skipped:
        name = entry.get("section") or "(document)"
        lines.append(f"  not restored — {name}: {entry.get('why', '')}")
    trouble = [f for f in (report.get("files") or [])
               if f.get("state") in ("missing", "changed")]
    for entry in trouble:
        lines.append(f"  {entry['state']} — {entry.get('role', '')}: {entry.get('path', '')}")
    return "\n".join(lines)
