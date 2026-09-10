"""Per-user persistence of recent paths in spaCR's canonical Qt store.

Recent paths used to live under two legacy ``Olafsson Lab`` namespaces.
The first access copies their keys into ``QSettings("spacr", "qt")`` without
deleting or overwriting either legacy store, so upgrades and downgrades both
remain safe.
"""
from __future__ import annotations

from threading import RLock
from typing import List

from PySide6.QtCore import QSettings

ORG = "spacr"
APP = "qt"

_LEGACY_NAMESPACES = (
    ("Olafsson Lab", "spaCR"),
    ("Olafsson Lab", "SpaCR"),
)
_MIGRATED_FILES: set[str] = set()
_MIGRATION_LOCK = RLock()


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
            legacy = QSettings(organization, application)
            if str(legacy.fileName()) == current_file:
                continue
            legacy.sync()
            for key in legacy.allKeys():
                if not current.contains(key):
                    current.setValue(key, legacy.value(key))
        current.sync()
        _MIGRATED_FILES.add(current_file)


def _s() -> QSettings:
    """Return spaCR's canonical store after a non-destructive migration."""
    settings = QSettings(ORG, APP)
    _migrate_legacy_settings(settings)
    return settings


def get_last_source(app_key: str) -> str:
    """Return the last folder used for a given app, or '' if unknown."""
    v = _s().value(f"recent/{app_key}/last")
    return str(v) if v else ""


def set_last_source(app_key: str, path: str) -> None:
    """Remember ``path`` as the most-recent source folder for ``app_key``."""
    if not path:
        return
    settings = _s()
    settings.setValue(f"recent/{app_key}/last", path)


def get_recent_sources(app_key: str, limit: int = 8) -> List[str]:
    """Return the recent-source list (most-recent first)."""
    v = _s().value(f"recent/{app_key}/list")
    if isinstance(v, str):
        items = [p for p in v.split("\n") if p]
    elif isinstance(v, list):
        items = [str(p) for p in v if p]
    else:
        items = []
    return items[:limit]


def push_recent_source(app_key: str, path: str, limit: int = 8) -> None:
    """Insert `path` at the head of the recent list and de-duplicate."""
    if not path:
        return
    items = [p for p in get_recent_sources(app_key, limit=limit + 1) if p != path]
    items.insert(0, path)
    _s().setValue(f"recent/{app_key}/list", "\n".join(items[:limit]))
    set_last_source(app_key, path)
