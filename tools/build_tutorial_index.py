#!/usr/bin/env python
"""Bundle ``app_key -> tutorial lesson`` into ``spacr/resources/``.

The published lesson library at ``docs/source/_extra/tutorials/`` carries a
``lesson_catalog.js`` of 73 lessons, 67 of which name the module they teach.
The application needs the reverse of that -- given a module, which lesson --
so a **Tutorial** link can sit beside the **API** link in the hover strip.

WHY A SEPARATE, SMALLER FILE. ``lesson_catalog.js`` is a JavaScript file for
the docs site, several hundred KB, holding objectives, descriptions and media
references for every lesson. The application needs two strings per module.
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
TARGET = ROOT / "spacr" / "resources" / "tutorial_index.json"


def read_catalog(path: Path = CATALOG) -> dict:
    """Parse the ``window.SPACR_LESSON_CATALOG = Object.freeze({...})`` file.

    Sliced between the first ``(`` and the last ``)`` rather than regex-ed,
    because the payload is one JSON object and everything around it is the
    two lines of JavaScript that assign it.
    """
    text = path.read_text(encoding="utf-8")
    return json.loads(text[text.index("(") + 1:text.rindex(")")])


def build(path: Path = CATALOG) -> dict:
    """``{app_key: {"lesson": id, "title": title}}``, first lesson wins.

    FIRST WINS because the catalog is ordered as the library presents it and
    several modules are taught by more than one lesson; the first is the one
    that introduces the module, which is what a reader following a link from
    the module itself is asking for.
    """
    catalog = read_catalog(path)
    index: dict = {}
    for lesson in catalog.get("lessons") or []:
        key = lesson.get("app_key")
        if not key or key in index:
            continue
        index[key] = {"lesson": lesson.get("id") or "",
                      "title": (lesson.get("title") or "").strip()}
    return {"schema": 1, "lessons": index}


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
