#!/usr/bin/env python3
"""Render every available silent visual master, safely resuming by lesson."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "production"
RENDERER = Path(__file__).with_name("render_visual_master.py")


def selected(value: str) -> set[int]:
    result = set()
    for part in value.split(","):
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lessons", default="1-73")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    wanted = selected(args.lessons)
    rendered = skipped = unavailable = 0
    for lesson_root in sorted(PRODUCTION.iterdir()):
        try:
            number = int(lesson_root.name.split("_", 1)[0])
        except (ValueError, IndexError):
            continue
        if number not in wanted:
            continue
        scenes = lesson_root / "scenes.json"
        timings = lesson_root / "audio" / "en" / "af_heart.json"
        output = lesson_root / "video" / f"{lesson_root.name}_silent.mp4"
        if not scenes.exists() or not timings.exists():
            unavailable += 1
            continue
        if output.exists() and output.stat().st_size > 100_000 and not args.force:
            skipped += 1
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            sys.executable, str(RENDERER), str(scenes), "--timings",
            str(timings), "--output", str(output),
        ], check=True)
        rendered += 1
        print(f"visual {lesson_root.name} rendered={rendered}", flush=True)
    print(f"complete rendered={rendered} skipped={skipped} unavailable={unavailable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
