#!/usr/bin/env python3
"""Publish compact catalog JavaScript and localized JSON for the web player."""
from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "catalog"
WEB = ROOT / "web"
PRODUCTION = ROOT / "production"


def main() -> int:
    english = json.loads((CATALOG / "lessons_en.json").read_text())
    for lesson in english["lessons"]:
        lesson["poster"] = f"{lesson['id']}/poster.jpg"
        lesson["silent"] = f"{lesson['id']}/video/{lesson['id']}_silent.mp4"
    (WEB / "lesson_catalog.js").write_text(
        "\"use strict\";\nwindow.SPACR_LESSON_CATALOG = Object.freeze(" +
        json.dumps(english, ensure_ascii=False, separators=(",", ":")) +
        ");\n",
        encoding="utf-8",
    )
    destination = WEB / "catalog"
    destination.mkdir(parents=True, exist_ok=True)
    for pattern in ("lessons_*.json", "captions_*.json"):
        for source in sorted(CATALOG.glob(pattern)):
            shutil.copy2(source, destination / source.name)
    print(WEB / "lesson_catalog.js")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
