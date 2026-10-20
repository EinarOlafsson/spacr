"""Per-user persistence of recent paths in spaCR's canonical Qt store.

Recent paths used to live under two legacy ``Olafsson Lab`` namespaces.
The first access copies their keys into ``QSettings("spacr", "qt")`` without
deleting or overwriting either legacy store, so upgrades and downgrades both
remain safe.
"""
from __future__ import annotations

import atexit
import logging
import threading
import time
from threading import RLock
from typing import Any, Dict, List, Tuple

from PySide6.QtCore import QSettings

ORG = "spacr"
APP = "qt"

_LEGACY_NAMESPACES = (
    ("Olafsson Lab", "spaCR"),
    ("Olafsson Lab", "SpaCR"),
)
_MIGRATED_FILES: set[str] = set()
_MIGRATION_LOCK = RLock()

LOG = logging.getLogger("spacr.qt.prefs")

_STORES = threading.Condition()
_PENDING_STORES: Dict[Tuple[str, int, str], Any] = {}
_STORE_STATE: Dict[str, Any] = {"busy": False, "thread": None}
_NOTHING = object()


def _store_key(settings: QSettings, key: str) -> Tuple[str, int, str]:
    """Name one value by its store's file, the file's format and its key.

    The file is resolved now, on the caller's thread, so a value is written
    to the store it was set in even if the store path moves before the
    writer reaches it.
    """
    return (str(settings.fileName()), int(settings.format().value), str(key))


def _store_later(settings: QSettings, key: str, value: Any) -> None:
    """Write one ``QSettings`` value on the preference writer thread.

    Ending a ``QSettings`` that holds a change writes the whole preference
    file on the thread that ends it: a lock file and a temporary file, each
    with ``fdatasync``, then a rename. That took 35-65 ms per write on an
    idle disk and seconds on a busy one, on the GUI thread, inside clips the
    tutorial recorder flagged. The value is held here until the writer has
    stored it, and :func:`_stored_later` answers reads in the meantime, so a
    value written is a value read. Only the newest value of a key is
    written.

    :param settings: a ``QSettings`` naming the store; it is only read.
    :param key: the settings key.
    :param value: the value to store.
    """
    item = _store_key(settings, key)
    with _STORES:
        _PENDING_STORES[item] = value
        thread = _STORE_STATE["thread"]
        if thread is None or not thread.is_alive():
            thread = threading.Thread(target=_run_stores,
                                      name="spacr-preference-writer",
                                      daemon=True)
            _STORE_STATE["thread"] = thread
            thread.start()
        _STORES.notify_all()


def _stored_later(settings: QSettings, key: str, default: Any = None) -> Any:
    """The value :func:`_store_later` still holds for ``key``, or ``default``.

    :param settings: a ``QSettings`` naming the store.
    :param key: the settings key.
    :param default: returned when nothing is waiting for that key.
    """
    with _STORES:
        return _PENDING_STORES.get(_store_key(settings, key), default)


def _write_stores(batch: Dict[Tuple[str, int, str], Any]) -> None:
    """Store one batch of values, one ``QSettings`` per file."""
    by_file: Dict[Tuple[str, int], list] = {}
    for (name, kind, key), value in batch.items():
        by_file.setdefault((name, kind), []).append((key, value))
    for (name, kind), values in by_file.items():
        try:
            settings = QSettings(name, QSettings.Format(kind))
            for key, value in values:
                settings.setValue(key, value)
            settings.sync()
        except Exception as exc:
            LOG.warning("a preference write failed: %s", exc)


def _run_stores() -> None:
    """The writer thread: store what is waiting, then wait for more."""
    while True:
        with _STORES:
            while not _PENDING_STORES:
                _STORES.wait()
            batch = dict(_PENDING_STORES)
            _STORE_STATE["busy"] = True
        try:
            _write_stores(batch)
        finally:
            with _STORES:
                for item, value in batch.items():
                    if _PENDING_STORES.get(item, _NOTHING) is value:
                        del _PENDING_STORES[item]
                _STORE_STATE["busy"] = False
                _STORES.notify_all()


def _flush_stores(timeout: float = 30.0) -> bool:
    """Wait until every value handed to :func:`_store_later` is stored.

    Values still waiting once the writer has stopped are stored on the
    caller's thread, so a clean exit loses none.

    :param timeout: longest wait, in seconds, for the writer.
    :returns: whether nothing is left waiting or being written.
    """
    deadline = time.monotonic() + max(0.0, float(timeout))
    with _STORES:
        while _PENDING_STORES or _STORE_STATE["busy"]:
            thread = _STORE_STATE["thread"]
            left = deadline - time.monotonic()
            if left <= 0 or thread is None or not thread.is_alive():
                break
            _STORES.wait(left)
        batch = {} if _STORE_STATE["busy"] else dict(_PENDING_STORES)
        if batch:
            _PENDING_STORES.clear()
        finished = not _STORE_STATE["busy"]
    if batch:
        _write_stores(batch)
    return finished


atexit.register(_flush_stores)


def _migrate_legacy_settings(current: QSettings) -> None:
    """Copy missing keys from both historical recent-path stores once.

    Existing values in the canonical store win: they were written by newer
    code and must not be replaced if an older spaCR build subsequently writes
    to a legacy store.  The old stores are read only and deliberately remain
    intact so downgrading cannot lose a user's recent paths.
    """
    current_file = str(current.fileName())
    with _MIGRATION_LOCK:
        if current_file in _MIGRATED_FILES:
            return
        for organization, application in _LEGACY_NAMESPACES:
            legacy = QSettings(*_store_args(organization, application))
            if str(legacy.fileName()) == current_file:
                continue
            legacy.sync()
            for key in legacy.allKeys():
                if not current.contains(key):
                    current.setValue(key, legacy.value(key))
        current.sync()
        _MIGRATED_FILES.add(current_file)


def _store_args(organization: str, application: str) -> tuple:
    """Return the ``QSettings`` constructor arguments for one store.

    Normally ``(organization, application)``, the platform's native store.
    In portable mode an INI file in the portable settings folder instead, on
    every platform, so the registry on Windows and the preferences plists on
    macOS are left untouched.
    """
    from ..logging_util import _portable_settings_dir

    folder = _portable_settings_dir()
    if folder is None:
        return (organization, application)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(folder))
    return (QSettings.IniFormat, QSettings.UserScope, organization,
            application)


def _s() -> QSettings:
    """Return spaCR's canonical store after a non-destructive migration."""
    settings = QSettings(*_store_args(ORG, APP))
    _migrate_legacy_settings(settings)
    return settings


def get_last_source(app_key: str) -> str:
    """Return the last folder used for a given app, or '' if unknown.

    :param app_key: the app key; the value is stored under
        ``recent/<app_key>/`` in the Qt settings.
    """
    v = _s().value(f"recent/{app_key}/last")
    return str(v) if v else ""


def set_last_source(app_key: str, path: str) -> None:
    """Remember ``path`` as the most-recent source folder for ``app_key``.

    :param app_key: the app key; the value is stored under
        ``recent/<app_key>/`` in the Qt settings.
    :param path: the folder to remember; an empty path is ignored.
    """
    if not path:
        return
    settings = _s()
    settings.setValue(f"recent/{app_key}/last", path)


def get_recent_sources(app_key: str, limit: int = 8) -> List[str]:
    """Return the recent-source list (most-recent first).

    :param app_key: the app key; the value is stored under
        ``recent/<app_key>/`` in the Qt settings.
    """
    v = _s().value(f"recent/{app_key}/list")
    if isinstance(v, str):
        items = [p for p in v.split("\n") if p]
    elif isinstance(v, list):
        items = [str(p) for p in v if p]
    else:
        items = []
    return items[:limit]


def push_recent_source(app_key: str, path: str, limit: int = 8) -> None:
    """Insert `path` at the head of the recent list and de-duplicate.

    :param app_key: the app key; the value is stored under
        ``recent/<app_key>/`` in the Qt settings.
    :param path: the folder to put first; an empty path is ignored. It also
        becomes the last source.
    """
    if not path:
        return
    items = [p for p in get_recent_sources(app_key, limit=limit + 1) if p != path]
    items.insert(0, path)
    _s().setValue(f"recent/{app_key}/list", "\n".join(items[:limit]))
    set_last_source(app_key, path)
