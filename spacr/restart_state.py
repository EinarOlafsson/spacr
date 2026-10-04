"""Persist GUI state across a forced spaCR restart.

The restart record stores the current module, its settings, a summary of
active runs, and the locations of any run folders. :func:`save` verifies the
record before the current process exits, and :func:`take` consumes it when the
new process starts so that stale state is not restored repeatedly.

Only the interface configuration is restored. Active computations do not
resume after a forced restart; output written before the restart remains in
the recorded run folders.

Two further records live in the preference store rather than in a file. The
last session (module, its settings and its folder) is written while spaCR runs
and when it closes, so an ordinary start can reopen where the user left off
after any exit or crash. Settings drafts hold the unsaved settings of every
open module; they are cleared on a clean exit, so a draft found at start-up
means the previous run ended without closing.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

LOG = logging.getLogger("spacr.restart_state")

SCHEMA_VERSION = 1
FILE_NAME = "restart_state.json"

#: How long a saved state is worth honouring, in seconds. A state older than
#: this is from a restart that never completed -- the machine was rebooted, or
#: the relaunch failed -- and reopening a module from last week because of it
#: would be a surprise with no cause the user can see.
MAX_AGE_SECONDS = 24 * 60 * 60


def state_path() -> Path:
    """Return the path used for the pending restart record."""
    root = os.environ.get("SPACR_HOME") or os.path.join(
        os.path.expanduser("~"), ".spacr")
    return Path(root) / FILE_NAME


def describe_running(running: Sequence[Mapping[str, Any]]) -> str:
    """Format active module names and elapsed times for a restart prompt.

    Parameters
    ----------
    running
        Active-run records. Each record may contain ``module`` or ``name``
        and an elapsed ``seconds`` value.

    Returns
    -------
    str
        A comma-separated summary, or an empty string when no named runs are
        present.
    """
    parts = []
    for entry in running or ():
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("module") or entry.get("name") or "").strip()
        if not name:
            continue
        seconds = entry.get("seconds")
        parts.append(f"{name} ({_elapsed(seconds)})" if seconds is not None
                     else name)
    return ", ".join(parts)


def _elapsed(seconds: Any) -> str:
    """Format elapsed seconds for the restart warning, or say it is running."""
    try:
        total = int(float(seconds))
    except (TypeError, ValueError):
        return "running"
    if total < 60:
        return f"running {total} s"
    if total < 3600:
        return f"running {total // 60} min"
    hours, minutes = divmod(total // 60, 60)
    return f"running {hours} h {minutes:02d} min"


def warning_text(running: Sequence[Mapping[str, Any]],
                 run_folders: Sequence[str] = ()) -> str:
    """Build the confirmation text shown before a forced restart.

    The text identifies runs that will stop, explains that settings will be
    restored, and lists the locations of partial output when available.

    Parameters
    ----------
    running
        Active-run records accepted by :func:`describe_running`.
    run_folders
        Paths that may contain output written before the restart.

    Returns
    -------
    str
        Paragraphs suitable for a restart confirmation dialog.
    """
    lines = []
    named = describe_running(running)
    if named:
        lines.append(f"These runs will be stopped and will NOT resume: {named}.")
        lines.append("Their settings are saved and come back with them — it is "
                     "the runs that are lost, not the configuration.")
    else:
        lines.append("No other module is running.")
    folders = [str(f) for f in (run_folders or ()) if f]
    if folders:
        lines.append("Whatever reached disk before the restart is still in: "
                     + ", ".join(folders[:4])
                     + (f" (+{len(folders) - 4} more)" if len(folders) > 4
                        else ""))
    lines.append("spaCR will close and start again, and reopen this module "
                 "with the settings it has now.")
    return "\n\n".join(lines)


def save(*, module: str, settings: Optional[Mapping[str, Any]] = None,
         running: Sequence[Mapping[str, Any]] = (),
         run_folders: Sequence[str] = (),
         saved: str = "") -> Optional[Path]:
    """Write and verify a pending restart record.

    Parameters
    ----------
    module
        Key of the module to reopen.
    settings
        Module settings to restore. Values that JSON cannot encode directly
        are converted to strings.
    running
        Active-run records to display in the restart summary.
    run_folders
        Paths that may contain partial output from interrupted runs.
    saved
        Optional ISO 8601 timestamp. The current UTC time is used by default.

    Returns
    -------
    pathlib.Path or None
        Path to the verified record, or ``None`` when it could not be written
        and the restart must be cancelled.

    Notes
    -----
    The function logs write errors instead of raising because it is called
    during shutdown. Callers must not restart when ``None`` is returned.
    """
    document = {
        "version": SCHEMA_VERSION,
        "saved": saved or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "module": str(module or ""),
        "settings": _jsonable(settings or {}),
        "running": [dict(entry) for entry in (running or ())
                    if isinstance(entry, Mapping)],
        "run_folders": [str(f) for f in (run_folders or ()) if f],
    }
    path = state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(document, indent=2, default=str),
                       encoding="utf-8")
        os.replace(tmp, path)
        read_back = json.loads(path.read_text(encoding="utf-8"))
        if read_back.get("module") != document["module"]:
            raise ValueError("the state read back as a different module")
    except Exception as exc:                              # noqa: BLE001
        LOG.warning("could not save the restart state: %s", exc)
        return None
    LOG.info("restart state saved → %s", path)
    return path


def _jsonable(value: Any, _depth: int = 0) -> Any:
    """Return a JSON-compatible copy of a settings value."""
    if _depth > 12:
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=str) if isinstance(
            value, (set, frozenset)) else value
        return [_jsonable(v, _depth + 1) for v in items]
    return str(value)


def peek() -> Optional[Dict[str, Any]]:
    """Read the pending restart record without consuming it.

    Returns
    -------
    dict or None
        The saved record, or ``None`` when no valid record can be read.
    """
    try:
        document = json.loads(state_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:                              # noqa: BLE001
        LOG.warning("could not read the restart state: %s", exc)
        return None
    return document if isinstance(document, dict) else None


def take() -> Optional[Dict[str, Any]]:
    """Consume and return a recent restart record.

    The saved file is removed before this function returns, including when
    the record is invalid or older than :data:`MAX_AGE_SECONDS`.

    Returns
    -------
    dict or None
        A recent restart record, or ``None`` when none is available.
    """
    document = peek()
    discard()
    if document is None:
        return None
    if _too_old(document):
        LOG.info("ignoring a restart state older than %s s", MAX_AGE_SECONDS)
        return None
    return document


def _too_old(document: Mapping[str, Any]) -> bool:
    """Return whether a valid saved timestamp exceeds the restart-state age."""
    stamp = str(document.get("saved") or "")
    if not stamp:
        return False
    try:
        when = datetime.fromisoformat(stamp)
    except ValueError:
        return False
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - when).total_seconds()
    return age > MAX_AGE_SECONDS


def discard() -> bool:
    """Remove the pending restart record.

    Returns
    -------
    bool
        ``True`` when a record was removed; otherwise ``False``.
    """
    try:
        state_path().unlink()
        return True
    except FileNotFoundError:
        return False
    except Exception as exc:                              # noqa: BLE001
        LOG.warning("could not remove the restart state: %s", exc)
        return False


_SESSION_KEY = "session/last"
_DRAFTS_KEY = "session/drafts"
_FOLDER_KEYS = ("src", "source_folder", "folder", "input_folder")


def _store():
    """Return the preference store that holds the session and the drafts."""
    from .qt.preferences import _settings

    return _settings()


def _folder_of(settings: Mapping[str, Any]) -> str:
    """Return the input folder named in ``settings``, or an empty string."""
    for key in _FOLDER_KEYS:
        value = settings.get(key) if isinstance(settings, Mapping) else None
        if isinstance(value, (list, tuple)):
            value = value[0] if value else ""
        if value:
            return str(value)
    return ""


def _write_json(key: str, document: Mapping[str, Any]) -> bool:
    """Store ``document`` as JSON under ``key``; return whether it was kept."""
    try:
        store = _store()
        store.setValue(key, json.dumps(_jsonable(document), default=str))
        store.sync()
    except Exception as exc:                              # noqa: BLE001
        LOG.warning("could not store %s: %s", key, exc)
        return False
    return True


def _read_json(key: str) -> Optional[Dict[str, Any]]:
    """Return the JSON mapping stored under ``key``, or ``None``."""
    try:
        raw = _store().value(key, "")
    except Exception as exc:                              # noqa: BLE001
        LOG.warning("could not read %s: %s", key, exc)
        return None
    if not raw:
        return None
    try:
        document = json.loads(str(raw))
    except ValueError:
        return None
    return document if isinstance(document, dict) else None


def _save_session(module: str,
                  settings: Optional[Mapping[str, Any]] = None) -> bool:
    """Record the visible module, its settings and its folder.

    :param module: key of the module on screen, or ``""`` for Home.
    :param settings: the module's current settings.
    :returns: whether the record was stored.
    """
    settings = dict(settings or {})
    return _write_json(_SESSION_KEY, {
        "version": SCHEMA_VERSION,
        "saved": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "module": str(module or ""),
        "settings": settings,
        "folder": _folder_of(settings),
    })


def _last_session() -> Optional[Dict[str, Any]]:
    """Return the last recorded session, or ``None`` when there is none.

    The record is read, not consumed: every ordinary start may reopen it.
    """
    document = _read_json(_SESSION_KEY)
    if document is None or not str(document.get("module") or ""):
        return None
    if not isinstance(document.get("settings"), dict):
        document["settings"] = {}
    return document


def _save_drafts(drafts: Mapping[str, Mapping[str, Any]]) -> bool:
    """Store the unsaved settings of every open module.

    :param drafts: module key to that module's current settings.
    :returns: whether the drafts were stored.
    """
    modules = {str(k): dict(v) for k, v in (drafts or {}).items()
               if k and isinstance(v, Mapping)}
    if not modules:
        return _clear_drafts()
    return _write_json(_DRAFTS_KEY, {
        "version": SCHEMA_VERSION,
        "saved": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modules": modules,
    })


def _take_drafts() -> Dict[str, Dict[str, Any]]:
    """Consume the stored drafts and return them as module to settings.

    The drafts are removed before this returns, so a draft is offered once.
    """
    document = _read_json(_DRAFTS_KEY)
    _clear_drafts()
    modules = (document or {}).get("modules")
    if not isinstance(modules, dict):
        return {}
    return {str(k): v for k, v in modules.items() if isinstance(v, dict)}


def _clear_drafts() -> bool:
    """Remove the stored drafts; return whether the store accepted it."""
    try:
        store = _store()
        store.remove(_DRAFTS_KEY)
        store.sync()
    except Exception as exc:                              # noqa: BLE001
        LOG.warning("could not clear the settings drafts: %s", exc)
        return False
    return True


def command() -> List[str]:
    """Return the command used to restart the PySide6 application.

    The command uses the active Python interpreter and the explicit
    :mod:`spacr.qt` entry point, so a forced restart returns to the same GUI
    without depending on an executable found through ``PATH``.

    Returns
    -------
    list of str
        Command arguments suitable for :class:`subprocess.Popen`.
    """
    return [sys.executable, "-m", "spacr.qt"]
