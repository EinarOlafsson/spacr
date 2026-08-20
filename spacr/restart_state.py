"""
Come back to where you were — the state a Force restart carries across.

Instruction 142. A fit can be somewhere no cooperative check reaches: inside
statsmodels' optimiser, in C, finishing an iteration whatever the GUI wants.
Instruction 140 C put a checkpoint on the objective so most of that is now
answered, but "most" is not "all", and the last thing left when Stop does not
stop is killing the application from outside and losing the whole
configuration.

This is what makes that recoverable::

    from spacr import restart_state

    written = restart_state.save(module="regression", settings={...},
                                 running=[{"module": "mask", "seconds": 840}])
    if written is None:
        ...                          # do NOT restart; say why
    restart_state.command()          # what to launch
    restart_state.take()             # on the next start: read it and DELETE it

THE SAVE HAPPENS FIRST AND IS VERIFIED BEFORE ANYTHING IS KILLED, and
:func:`save` returns ``None`` rather than raising when it cannot write. A
restart that loses the settings is worse than a stuck run, because a stuck run
can at least be waited out.

IT IS TAKEN, NOT READ. :func:`take` deletes the file as it returns it, so a
crash on the way back up does not leave a state that reopens the same wedged
module on every launch afterwards -- which would turn one bad afternoon into a
permanently broken installation.

WHAT IS NOT PROMISED, and the dialog says so: the RUNS do not come back. Only
the configuration does. A killed run's partial output is whatever reached
disk, and the run folders are named so a user knows where to look rather than
assuming everything is gone or everything is fine.

Standard library only: this is imported on the way out of a process that is
already in trouble.
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
    """Where the state is kept: beside the run journal, under ``~/.spacr``."""
    root = os.environ.get("SPACR_HOME") or os.path.join(
        os.path.expanduser("~"), ".spacr")
    return Path(root) / FILE_NAME


def describe_running(running: Sequence[Mapping[str, Any]]) -> str:
    """Name every module that is running, and for how long. "" if none.

    NAMED INDIVIDUALLY, because "other modules are running" is not enough
    information to decide with: a user weighing up two hours of segmentation
    needs to know that is what they are weighing up.
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
    """What the dialog says before it restarts anything.

    Three separate facts, because they are three different losses and a user
    weighs them separately: which runs stop, that the SETTINGS come back, and
    where the partial output is.
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
    """Write the state. Returns the path, or ``None`` if it could not be.

    ``None`` IS THE ANSWER, NOT AN EXCEPTION. The caller's next step is to
    tell the user and NOT restart, and a raise here would arrive inside a
    dialog handler on the way out of a wedged application -- the worst place
    in the program to grow a second failure.

    VERIFIED BY READING IT BACK, because "the write returned" and "the file is
    there and parses" are different claims, and this one is only worth making
    if it is the second.
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
    """A JSON-writable copy. Settings hold Paths, tuples and numpy scalars."""
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
    """Read the state WITHOUT deleting it. For tests and for reporting."""
    try:
        document = json.loads(state_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:                              # noqa: BLE001
        LOG.warning("could not read the restart state: %s", exc)
        return None
    return document if isinstance(document, dict) else None


def take() -> Optional[Dict[str, Any]]:
    """Read the state and DELETE it. Returns ``None`` if there is none.

    Deleted whatever happens next, including when it turns out to be too old
    or unreadable: a state that survives its own use reopens the same wedged
    module on every launch afterwards, which turns one bad afternoon into a
    permanently broken installation.
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
    """Remove any saved state. Returns whether there was one."""
    try:
        state_path().unlink()
        return True
    except FileNotFoundError:
        return False
    except Exception as exc:                              # noqa: BLE001
        LOG.warning("could not remove the restart state: %s", exc)
        return False


def command() -> List[str]:
    """The command line that starts spaCR again.

    THE RUNNING INTERPRETER, not "spacr" off the PATH. A user in a conda
    environment, or running from a checkout, or from a bundled build, has a
    PATH entry that may point at a different installation entirely -- and
    coming back as a different spaCR than the one that was open is a worse
    outcome than not coming back.
    """
    return [sys.executable, "-m", "spacr"]
