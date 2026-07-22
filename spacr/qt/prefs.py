"""
Small QSettings wrapper for per-user persistence of "recent" paths.

The organization/application name is set by `launch()` in
`spacr.qt.app`, so this module works with the platform-appropriate
settings backend (macOS plist, Windows registry, Linux INI file).
"""
from __future__ import annotations

from typing import List

from PySide6.QtCore import QSettings


ORG = "Olafsson Lab"
APP = "SpaCR"


def _s() -> QSettings:
    """Return a QSettings pinned to the spaCR organization/app namespace."""
    return QSettings(ORG, APP)


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
