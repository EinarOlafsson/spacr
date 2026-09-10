#!/usr/bin/env python3
"""Replace one lesson in lessons_en.json from a maintained JSON override."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "catalog" / "lessons_en.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("override", type=Path)
    args = parser.parse_args()
    lesson = json.loads(args.override.read_text())
    catalog = json.loads(CATALOG.read_text())
    for index, current in enumerate(catalog["lessons"]):
        if current["id"] == lesson["id"]:
            catalog["lessons"][index] = lesson
            break
    else:
        raise KeyError(f"lesson {lesson['id']!r} is not in the catalog")
    CATALOG.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n"
    )
    print(lesson["id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
