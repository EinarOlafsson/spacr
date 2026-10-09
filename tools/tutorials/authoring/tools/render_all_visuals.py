#!/usr/bin/env python3
"""Render every available silent visual master, safely resuming by lesson."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(os.environ.get('SPACR_TUTORIAL_WORKSPACE', Path(__file__).resolve().parents[1])).resolve()
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
        native_timings = lesson_root / 'video' / 'native-live-timings.json'
        if native_timings.exists() or (scenes.exists() and any(
                scene.get('clip') for scene in json.loads(scenes.read_text())['scenes'])):
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from native_live_timing import checked_native_timing
            timings = checked_native_timing(ROOT, lesson_root.name)
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
