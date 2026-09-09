#!/usr/bin/env python3
"""Create compact 4K posters and verify publishable tutorial media."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "production"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lessons",
        help="comma-separated lesson numbers or ids; default is all",
    )
    args = parser.parse_args()
    selected = None
    if args.lessons:
        selected = {item.strip() for item in args.lessons.split(",")
                    if item.strip()}
    missing = []
    for lesson in sorted(path for path in PRODUCTION.iterdir()
                         if path.is_dir() and path.name[:2].isdigit()):
        if selected is not None and not ({lesson.name, lesson.name[:2]}
                                         & selected):
            continue
        scenes_path = lesson / "scenes.json"
        video = lesson / "video" / f"{lesson.name}_silent.mp4"
        if not scenes_path.exists() or not video.exists():
            missing.append(lesson.name)
            continue
        scenes = json.loads(scenes_path.read_text())
        source = lesson / scenes["scenes"][0]["image"]
        image = Image.open(source).convert("RGB")
        image.save(lesson / "poster.jpg", quality=88, optimize=True,
                   progressive=True, subsampling=1)
        print(lesson / "poster.jpg")
    if missing:
        raise RuntimeError(f"missing visual masters: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
