#!/usr/bin/env python
"""Bundle ``app_key -> tutorial lesson`` into ``spacr/resources/``.

The published lesson library at ``docs/source/_extra/tutorials/`` carries a
``lesson_catalog.js`` of 73 lessons, 67 of which name the module they teach.
The application needs the reverse of that -- given a module, which lesson --
so a **Tutorial** link can sit beside the **API** link in the hover strip,
and ``spacr-tutorial`` needs the lesson list itself: number, title, slug and
the screen each lesson covers.

WHY A SEPARATE, SMALLER FILE. ``lesson_catalog.js`` is a JavaScript file for
the docs site, several hundred KB, holding objectives, descriptions and media
references for every lesson. The application needs a few strings per lesson.
Shipping the catalog to get them would put the docs build's output in the
wheel and make the GUI parse JS at startup.

Run after the lesson catalog changes:

    python tools/build_tutorial_index.py

``--check`` exits non-zero when the bundled file is stale instead of writing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "docs" / "source" / "_extra" / "tutorials" / "lesson_catalog.js"
NAVIGATION = CATALOG.with_name("module_navigation.js")
TARGET = ROOT / "spacr" / "resources" / "tutorial_index.json"


def read_catalog(path: Path = CATALOG) -> dict:
    """Parse the ``window.SPACR_LESSON_CATALOG = Object.freeze({...})`` file.

    Sliced between the first ``(`` and the last ``)`` rather than regex-ed,
    because the payload is one JSON object and everything around it is the
    two lines of JavaScript that assign it.
    """
    text = path.read_text(encoding="utf-8")
    return json.loads(text[text.index("(") + 1:text.rindex(")")])


def display_order(catalog: dict, navigation_path: Path) -> list:
    """Lesson ids in the order the player numbers them.

    The player lists the navigation's lessons first, intro then each
    section's groups, and any catalog lesson the navigation does not place
    after them; a lesson's displayed number is its position in that list.
    Mirrored here so ``spacr-tutorial 7`` opens the lesson the library
    shows as 7, which stays contiguous when lessons are archived.
    """
    ids = [lesson.get("id") or "" for lesson in catalog.get("lessons") or []]
    order = []
    if navigation_path.is_file():
        navigation = read_catalog(navigation_path)
        order = list((navigation.get("intro") or {}).get("lessons") or [])
        for section in navigation.get("sections") or []:
            for group in section.get("groups") or []:
                order.extend(group.get("lessons") or [])
    known = set(ids)
    return [i for i in dict.fromkeys(order + ids) if i in known]


def archived_modules() -> dict:
    """``{app_key: {"lesson", "title"}}`` for modules whose lesson is archived."""
    sys.path.insert(0, str(ROOT / "tools" / "tutorials"))
    from archived import load_index
    return {entry["app_key"]: {"lesson": identity, "title": entry.get("title") or ""}
            for identity, entry in load_index()["lessons"].items()
            if entry.get("app_key")}


def build(path: Path = CATALOG, navigation_path: Path | None = None) -> dict:
    """The bundled table: ``lessons`` by module, and the whole ``catalog``.

    ``lessons`` is ``{app_key: {"lesson": id, "title": title}}``, first lesson
    wins. FIRST WINS because the catalog is ordered as the library presents
    it and several modules are taught by more than one lesson; the first is
    the one that introduces the module, which is what a reader following a
    link from the module itself is asking for.

    ``catalog`` is every lesson in library order, as ``{"number", "lesson",
    "slug", "title", "app_key"}``. ``spacr-tutorial`` lists it and resolves a
    lesson number, slug or screen name against it, so the command offers
    exactly the lessons the library and the in-app links offer.
    """
    catalog = read_catalog(path)
    navigation_path = navigation_path or path.with_name("module_navigation.js")
    numbers = {identity: position for position, identity
               in enumerate(display_order(catalog, navigation_path), 1)}
    index: dict = {}
    rows: list = []
    for lesson in catalog.get("lessons") or []:
        key = lesson.get("app_key") or ""
        title = (lesson.get("title") or "").strip()
        rows.append({"number": numbers.get(lesson.get("id") or "", 0),
                     "lesson": lesson.get("id") or "",
                     "slug": lesson.get("slug") or "",
                     "title": title,
                     "app_key": key})
        if not key or key in index:
            continue
        index[key] = {"lesson": lesson.get("id") or "", "title": title}
    rows.sort(key=lambda row: row["number"])
    coming_soon = {key: value for key, value in archived_modules().items()
                   if key not in index}
    return {"schema": 2, "lessons": index, "catalog": rows,
            "coming_soon": coming_soon}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="fail if the bundled file is out of date")
    args = parser.parse_args(argv)

    if not CATALOG.exists():
        print(f"no lesson catalog at {CATALOG}", file=sys.stderr)
        return 1
    fresh = build()
    text = json.dumps(fresh, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.check:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if current != text:
            print(f"{TARGET} is stale -- run "
                  f"`python tools/build_tutorial_index.py`", file=sys.stderr)
            return 1
        print(f"{TARGET}: up to date ({len(fresh['lessons'])} modules)")
        return 0
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(text, encoding="utf-8")
    print(f"wrote {TARGET} ({len(fresh['lessons'])} modules)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
