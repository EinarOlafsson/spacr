#!/usr/bin/env python3
"""Create compact 4K posters and verify publishable tutorial media."""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import os
from pathlib import Path

from PIL import Image


ROOT = Path(os.environ.get('SPACR_TUTORIAL_WORKSPACE', Path(__file__).resolve().parents[1])).resolve()
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
        scenes = json.loads(scenes_path.read_text(encoding="utf-8"))
        if scenes.get("kind") == "published_native_intervals":
            # Reuse has no raw image paths. Validate its real source/voice bindings
            # before extracting the actual first decoded master frame.
            from published_native_reuse import checked_published_native_timing
            checked_published_native_timing(PRODUCTION.parent, lesson.name)
            first_frame = subprocess.run([
                "ffmpeg", "-nostdin", "-v", "error", "-threads", "2",
                "-i", str(video), "-frames:v", "1", "-f", "image2pipe",
                "-vcodec", "png", "-threads", "2", "pipe:1",
            ], check=True, stdout=subprocess.PIPE, timeout=60).stdout
            image = Image.open(io.BytesIO(first_frame)).convert("RGB")
        else:
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
