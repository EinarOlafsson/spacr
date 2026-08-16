#!/usr/bin/env python3
"""Recreate scene-level tutorial audit stills from committed silent masters."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "docs" / "source" / "_extra" / "tutorials"
AUDIO_ROOT = "https://huggingface.co/datasets/einarolafsson/spacr-tutorials/resolve/main"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_catalog(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    prefix = "window.SPACR_LESSON_CATALOG = Object.freeze("
    start = source.index(prefix) + len(prefix)
    return json.loads(source[start:source.rindex(");")])


def scene_midpoint(scene: dict) -> float:
    start = float(scene["speech_start"])
    end = float(scene["speech_end"])
    return start + max(0.05, (end - start) / 2.0)


def fetch_timing(lesson_id: str, voice: str, timeout: int) -> tuple[dict, bytes]:
    url = f"{AUDIO_ROOT}/{lesson_id}/audio/en/{voice}.json?download=true"
    request = urllib.request.Request(url, headers={"User-Agent": "spaCR tutorial frame audit"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    return json.loads(payload), payload


def extract_frame(video: Path, timestamp: float, target: Path, width: int) -> None:
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{timestamp:.3f}", "-i", str(video), "-frames:v", "1",
        "-vf", f"scale={width}:-2:flags=lanczos", "-q:v", "3", str(target),
    ]
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lesson", action="append", default=[],
                        help="lesson id; repeat for more than one")
    parser.add_argument("--all", action="store_true", help="sample all 73 lessons")
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not shutil.which("ffmpeg"):
        parser.error("ffmpeg is required")
    if args.all == bool(args.lesson):
        parser.error("choose either --all or at least one --lesson")

    catalog_path = TUTORIALS / "lesson_catalog.js"
    catalog = load_catalog(catalog_path)
    by_id = {lesson["id"]: lesson for lesson in catalog["lessons"]}
    selected = list(by_id) if args.all else args.lesson
    missing = sorted(set(selected) - set(by_id))
    if missing:
        parser.error(f"unknown lessons: {missing}")

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": 1,
        "catalog_sha256": _sha256(catalog_path),
        "voice": args.voice,
        "width": args.width,
        "lessons": [],
    }
    for lesson_id in selected:
        lesson = by_id[lesson_id]
        video = TUTORIALS / "production" / lesson_id / "video" / lesson["silent"]
        if not video.exists():
            # Catalog paths are relative to production/, not the lesson folder.
            video = TUTORIALS / "production" / lesson["silent"]
        timing, timing_bytes = fetch_timing(lesson_id, args.voice, args.timeout)
        if len(timing["scenes"]) != len(lesson["scenes"]):
            raise AssertionError(f"{lesson_id}: timing/catalog scene count differs")
        lesson_dir = args.output / lesson_id
        lesson_dir.mkdir(exist_ok=True)
        item = {
            "id": lesson_id,
            "title": lesson["title"],
            "video": str(video.relative_to(ROOT)),
            "video_sha256": _sha256(video),
            "timing_sha256": hashlib.sha256(timing_bytes).hexdigest(),
            "scenes": [],
        }
        for catalog_scene, timing_scene in zip(lesson["scenes"], timing["scenes"]):
            number = int(timing_scene["scene"])
            timestamp = scene_midpoint(timing_scene)
            target = lesson_dir / f"scene_{number:02d}.jpg"
            extract_frame(video, timestamp, target, args.width)
            item["scenes"].append({
                "scene": number,
                "timestamp": round(timestamp, 3),
                "narration": catalog_scene["narration"],
                "visual": catalog_scene["visual"],
                "image": str(target.relative_to(args.output)),
                "image_sha256": _sha256(target),
            })
        report["lessons"].append(item)

    manifest = args.output / "audit_manifest.json"
    manifest.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"sampled {sum(len(x['scenes']) for x in report['lessons'])} scenes "
          f"from {len(report['lessons'])} lessons into {args.output}")
    print(f"manifest={manifest} sha256={_sha256(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
