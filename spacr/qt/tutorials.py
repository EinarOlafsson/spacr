"""Which published lesson teaches a module, and the URL that opens it.

The lesson library at ``<docs>/tutorials/`` deep-links with ``#lesson=<id>``,
so pointing a reader at the lesson for the module they are hovering is a
lookup and a string. The lookup table is
``spacr/resources/tutorial_index.json``, written by
``tools/build_tutorial_index.py`` from the docs build's own lesson catalog --
see there for why it is not the catalog itself.

NO NETWORK, NO HEAVY IMPORT. This is read on the hover path, from Home and
from the dock, so it caches the table on first use and answers ``""`` for
anything it cannot resolve. A module with no lesson must cost a missing link,
never an exception in a hover handler.
"""
from __future__ import annotations

from typing import Dict, Optional

#: The published library, and the fragment it reads a lesson id out of.
#: Duplicated from :data:`spacr.qt.app.TUTORIALS_URL` rather than imported:
#: `app` builds the whole main window, and this module is imported from a
#: hover handler.
TUTORIALS_URL = "https://einarolafsson.github.io/spacr/tutorials/"

#: ``{app_key: {"lesson": id, "title": title}}``, or None until first read.
_INDEX: Optional[Dict[str, dict]] = None


def _index() -> Dict[str, dict]:
    """The bundled table, read once. ``{}`` when it cannot be read."""
    global _INDEX
    if _INDEX is None:
        try:
            import json
            from importlib.resources import files

            raw = (files("spacr.resources") / "tutorial_index.json")
            data = json.loads(raw.read_text(encoding="utf-8"))
            lessons = data.get("lessons") or {}
            _INDEX = {str(k): v for k, v in lessons.items()
                      if isinstance(v, dict)}
        except Exception:                                        # noqa: BLE001
            # No resource in this wheel, or an unreadable one. A missing
            # Tutorial link is the right failure; a traceback out of a hover
            # handler is not.
            _INDEX = {}
    return _INDEX


def lesson_for(app_key: str) -> str:
    """The lesson id that teaches ``app_key``, or ``""``."""
    entry = _index().get(str(app_key or ""))
    return str((entry or {}).get("lesson") or "")


def lesson_title(app_key: str) -> str:
    """The lesson's own title, for a tooltip. ``""`` when there is none."""
    entry = _index().get(str(app_key or ""))
    return str((entry or {}).get("title") or "")


def tutorial_url(app_key: str) -> str:
    """The URL that opens ``app_key``'s lesson, or ``""`` when it has none.

    Empty rather than the library's front page on purpose. The link is drawn
    only when this answers, so a word that would have dropped the reader at
    an index of seventy-three lessons is simply not offered -- the rule the
    tooltip footer already follows for **Animation**.
    """
    lesson = lesson_for(app_key)
    return f"{TUTORIALS_URL}#lesson={lesson}" if lesson else ""


def has_tutorial(app_key: str) -> bool:
    """Whether ``app_key`` has a lesson to link to."""
    return bool(lesson_for(app_key))
