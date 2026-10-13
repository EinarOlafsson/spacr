"""Which published lesson teaches a module, and the URL that opens it.

The lesson library at ``<docs>/tutorials/`` deep-links with ``#lesson=<id>``,
so pointing a reader at the lesson for the module they are hovering is a
lookup and a string. The lookup table is
``spacr/resources/tutorial_index.json``, written by
``tools/build_tutorial_index.py`` from the docs build's own lesson catalog --
see there for why it is not the catalog itself.

No network access and no heavy imports: this is read on the hover path, from Home and
from the dock, so it caches the table on first use and answers ``""`` for
anything it cannot resolve. A module with no lesson gets no link, and no
exception is raised in a hover handler.

The same table carries the whole lesson list, which ``spacr-tutorial`` prints
and resolves a lesson number, slug or screen name against; see
:func:`lessons` and :func:`find_lesson`.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

#: The published library, and the fragment it reads a lesson id out of.
#: Duplicated from :data:`spacr.qt.app.TUTORIALS_URL` rather than imported:
#: `app` builds the whole main window, and this module is imported from a
#: hover handler.
TUTORIALS_URL = "https://einarolafsson.github.io/spacr/tutorials/"

#: ``{app_key: {"lesson": id, "title": title}}``, or None until first read.
_INDEX: Optional[Dict[str, dict]] = None

#: Every lesson in lesson-number order, or None until first read.
_LESSONS: Optional[List[dict]] = None


def _resource(files, name: str):
    """``name`` in ``spacr/resources``, as ``importlib.resources`` gives it.

    ``spacr.resources`` is a folder without an ``__init__``, and Python 3.9's
    ``files`` raises ``TypeError`` on such a namespace package; the folder
    next to this package is read instead.

    :param files: ``importlib.resources.files``.
    :param name: the file name inside ``spacr/resources``.
    :returns: a path-like object with ``read_text``.
    """
    try:
        return files("spacr.resources") / name
    except TypeError:
        from pathlib import Path

        return Path(__file__).resolve().parents[1] / "resources" / name


def _index() -> Dict[str, dict]:
    """The bundled table, read once. ``{}`` when it cannot be read."""
    global _INDEX
    if _INDEX is None:
        try:
            import json
            from importlib.resources import files

            raw = _resource(files, "tutorial_index.json")
            data = json.loads(raw.read_text(encoding="utf-8"))
            lessons = data.get("lessons") or {}
            _INDEX = {str(k): v for k, v in lessons.items()
                      if isinstance(v, dict)}
        except Exception:                                        # noqa: BLE001
            _INDEX = {}
    return _INDEX


def lesson_for(app_key: str) -> str:
    """The lesson id that teaches ``app_key``, or ``""``.

    :param app_key: the app key looked up in the bundled
        ``tutorial_index.json``; None or empty gives no lesson.
    """
    entry = _index().get(str(app_key or ""))
    return str((entry or {}).get("lesson") or "")


def lesson_title(app_key: str) -> str:
    """The lesson's own title, for a tooltip. ``""`` when there is none.

    :param app_key: the app key looked up in the bundled
        ``tutorial_index.json``; None or empty gives no lesson.
    """
    entry = _index().get(str(app_key or ""))
    return str((entry or {}).get("title") or "")


def tutorial_url(app_key: str) -> str:
    """The URL that opens ``app_key``'s lesson, or ``""`` when it has none.

    Empty rather than the library's front page on purpose. The link is drawn
    only when this answers, so a word that would have dropped the reader at
    an index of seventy-three lessons is simply not offered -- the rule the
    tooltip footer already follows for **Animation**.

    :param app_key: the app key looked up in the bundled
        ``tutorial_index.json``; None or empty gives no lesson.
    """
    lesson = lesson_for(app_key)
    return lesson_url(lesson) if lesson else ""


def has_tutorial(app_key: str) -> bool:
    """Whether ``app_key`` has a lesson to link to.

    :param app_key: the app key looked up in the bundled
        ``tutorial_index.json``; None or empty gives no lesson.
    """
    return bool(lesson_for(app_key))


def lesson_url(lesson: str) -> str:
    """The URL that opens one lesson of the library.

    :param lesson: the lesson id, such as ``07_mask``.
    :returns: the library URL with the ``#lesson=`` fragment it reads.
    """
    return f"{TUTORIALS_URL}#lesson={lesson}"


def lessons() -> List[dict]:
    """Every lesson in the library, in lesson-number order.

    Read once from the bundled table and cached; an unreadable table gives
    an empty list, as the module lookups give ``""``.

    :returns: one dict per lesson with ``number`` (int), ``lesson`` (its id,
        such as ``07_mask``), ``slug``, ``title`` and ``app_key`` (the screen
        it covers, ``""`` for a lesson about no single screen).
    """
    global _LESSONS
    if _LESSONS is None:
        try:
            import json
            from importlib.resources import files

            raw = _resource(files, "tutorial_index.json")
            rows = json.loads(raw.read_text(encoding="utf-8")).get("catalog")
            _LESSONS = sorted(
                ({"number": int(row.get("number") or 0),
                  "lesson": str(row["lesson"]),
                  "slug": str(row.get("slug") or ""),
                  "title": str(row.get("title") or ""),
                  "app_key": str(row.get("app_key") or "")}
                 for row in rows or [] if isinstance(row, dict)
                 and row.get("lesson")),
                key=lambda row: row["number"])
        except (ImportError, OSError, ValueError, TypeError, AttributeError):
            _LESSONS = []
    return list(_LESSONS)


def _spelling(name: object) -> str:
    """Spell ``name`` the way the lesson lookup compares names.

    Lower case, ``_`` for each hyphen and space, and a number without its
    leading zeros, so ``Make-Masks`` is ``make_masks`` and ``07`` is ``7``.

    :param name: a typed name, or a catalog field.
    """
    text = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    return str(int(text)) if text.isdigit() else text


def find_lesson(name: str) -> Tuple[Optional[dict], List[str]]:
    """The lesson ``name`` names, or the names closest to it.

    ``name`` may be a lesson number (``7`` or ``07``), a lesson id
    (``07_mask``), the key of the screen a lesson covers (``measure``) or a
    slug (``plaque-assay``). Case, hyphens and spaces do not matter. When two
    lessons answer to one name, the earlier kind in that list wins, and among
    lessons of one kind the lowest-numbered.

    :param name: what the reader typed.
    :returns: ``(lesson, [])`` as :func:`lessons` gives it, or ``(None,
        suggestions)`` when nothing matches: up to five slugs and screen
        names, the closest spellings first, then those containing ``name``.
    """
    import difflib

    catalog = lessons()
    names: Dict[str, dict] = {}
    for field in ("number", "lesson", "app_key", "slug"):
        for lesson in catalog:
            key = _spelling(lesson[field])
            if key:
                names.setdefault(key, lesson)
    wanted = _spelling(name)
    if wanted in names:
        return names[wanted], []
    words = sorted({_spelling(lesson[field]) for lesson in catalog
                    for field in ("slug", "app_key")} - {""})
    close = difflib.get_close_matches(wanted, words, n=5, cutoff=0.6)
    within = [word for word in words if wanted and wanted in word]
    return None, list(dict.fromkeys(close + within))[:5]
