#!/usr/bin/env python3
"""Fail closed when any published lesson, language, voice, or 4K master is missing."""
from __future__ import annotations

import json
import re
import subprocess
from fractions import Fraction
from pathlib import Path

from PIL import Image

from pronunciation import assert_pronunciation_safe
from render_all_voices import LANGUAGES
from render_visual_master import validate_geometry


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "catalog"
PRODUCTION = ROOT / "production"
WEB = ROOT / "web"


def video_properties(path: Path) -> tuple[int, int, float, str, float, bool]:
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries",
        "stream=codec_type,codec_name,width,height,avg_frame_rate:format=duration",
        "-of", "json", str(path),
    ], check=True, text=True, capture_output=True)
    data = json.loads(result.stdout)
    videos = [
        stream for stream in data["streams"]
        if stream.get("codec_type") == "video"
    ]
    if len(videos) != 1:
        raise ValueError(f"expected one video stream, found {len(videos)}")
    stream = videos[0]
    frame_rate = float(Fraction(stream["avg_frame_rate"]))
    has_audio = any(
        item.get("codec_type") == "audio" for item in data["streams"]
    )
    return (
        int(stream["width"]),
        int(stream["height"]),
        float(data["format"]["duration"]),
        str(stream["codec_name"]),
        frame_rate,
        has_audio,
    )


def main() -> int:
    errors = []
    ids = None
    partial_videos = sorted(PRODUCTION.glob("*/video/.*.part.mp4"))
    if partial_videos:
        rendered_paths = ", ".join(str(path) for path in partial_videos)
        errors.append(
            f"partial visual masters remain: {rendered_paths}"
        )
    english = json.loads((CATALOG / "lessons_en.json").read_text())
    expected = len(english["lessons"])
    for lesson in english["lessons"]:
        for index, scene in enumerate(lesson["scenes"], start=1):
            try:
                assert_pronunciation_safe(
                    scene["narration"], scene.get("speech_text", ""))
            except ValueError as error:
                errors.append(
                    f"unsafe pronunciation {lesson['id']} scene {index}: "
                    f"{error}")
    for language in LANGUAGES:
        path = CATALOG / f"lessons_{language}.json"
        if not path.exists():
            errors.append(f"missing catalog {language}")
            continue
        catalog = json.loads(path.read_text())
        current = [lesson["id"] for lesson in catalog["lessons"]]
        if len(current) != expected or len(set(current)) != expected:
            errors.append(
                f"{language} catalog does not contain {expected} unique lessons")
        if ids is None:
            ids = current
        elif current != ids:
            errors.append(f"{language} lesson ids/order differ from English")

    if ids:
        for lesson_id in ids:
            root = PRODUCTION / lesson_id
            scenes_path = root / "scenes.json"
            video = root / "video" / f"{lesson_id}_silent.mp4"
            poster = root / "poster.jpg"
            if not scenes_path.exists():
                errors.append(f"missing scene specification {lesson_id}")
            else:
                try:
                    spec = json.loads(scenes_path.read_text())
                    validate_geometry(spec, root)
                    source_lesson = next(
                        item for item in english["lessons"]
                        if item["id"] == lesson_id)
                    if len(spec["scenes"]) != len(source_lesson["scenes"]):
                        errors.append(
                            f"scene count differs from English catalog: "
                            f"{lesson_id}")
                except Exception as error:
                    errors.append(
                        f"invalid scene geometry {lesson_id}: {error}")
            if not video.exists():
                errors.append(f"missing video {lesson_id}")
            else:
                try:
                    width, height, duration, codec, fps, has_audio = (
                        video_properties(video)
                    )
                    heart_timing = json.loads(
                        (root / "audio" / "en" / "af_heart.json").read_text()
                    )
                    expected_duration = float(heart_timing["total_duration"])
                    if (
                        (width, height) != (3840, 2160)
                        or duration < 20
                        or codec != "h264"
                        or abs(fps - 30.0) > 0.001
                        or has_audio
                        or abs(duration - expected_duration) > (1 / 30 + 0.002)
                    ):
                        errors.append(
                            f"invalid video {lesson_id}: {width}x{height} "
                            f"codec={codec} fps={fps:.3f} audio={has_audio} "
                            f"duration={duration:.3f}s "
                            f"expected={expected_duration:.3f}s"
                        )
                except Exception as error:
                    errors.append(f"invalid video {lesson_id}: {error}")
            if not poster.exists():
                errors.append(f"missing poster {lesson_id}")
            elif Image.open(poster).size != (3840, 2160):
                errors.append(f"poster is not 4K: {lesson_id}")
            expected_media = set()
            expected_captions = set()
            for language, (_, voices) in LANGUAGES.items():
                for voice in voices:
                    audio = root / "audio" / language / f"{voice}.m4a"
                    timing = root / "audio" / language / f"{voice}.json"
                    caption = root / "captions" / language / f"{voice}.vtt"
                    expected_media.update((audio, timing))
                    expected_captions.add(caption)
                    if not audio.exists() or audio.stat().st_size < 10_000:
                        errors.append(f"missing audio {lesson_id}/{language}/{voice}")
                    if not timing.exists() or timing.stat().st_size < 500:
                        errors.append(f"missing timings {lesson_id}/{language}/{voice}")
                    if not caption.exists() or caption.stat().st_size < 100:
                        errors.append(
                            f"missing captions {lesson_id}/{language}/{voice}"
                        )
            actual_media = {
                path
                for path in (root / "audio").glob("*/*")
                if path.is_file() and path.suffix in {".m4a", ".json"}
            }
            actual_captions = {
                path
                for path in (root / "captions").glob("*/*.vtt")
                if path.is_file()
            }
            for unexpected in sorted(actual_media - expected_media):
                errors.append(f"unsupported audio artifact: {unexpected}")
            for unexpected in sorted(actual_captions - expected_captions):
                errors.append(f"unsupported caption artifact: {unexpected}")

    web_text = "\n".join(
        path.read_text(errors="ignore")
        for path in WEB.rglob("*")
        if path.is_file()
        and path.suffix in {".html", ".js", ".css", ".json"}
    )
    if "spacr.org" in web_text.lower():
        errors.append("spacr.org appears in web assets")
    if "youtube" in web_text.lower():
        errors.append("YouTube appears in web assets")
    for filename in ("index.html", "styles.css", "app_v2.js", "voice_catalog.js",
                     "lesson_catalog.js", "logo_spacr.png", "favicon.svg"):
        if not (WEB / filename).exists():
            errors.append(f"missing web asset {filename}")
    player = (WEB / "app_v2.js").read_text(errors="ignore")
    for contract in (
            "async function fetchNarrationAudio",
            "response.blob()",
            "URL.createObjectURL(result.blob)",
            "function seekNarrationToVideo()",
            "function syncVideoToNarration(force = false)",
            'elements.audio.addEventListener("timeupdate"'):
        if contract not in player:
            errors.append(
                f"phone-safe narration download contract is missing: {contract}")
    if "elements.audio.src = audioSource()" in player:
        errors.append(
            "narration is assigned directly to the media element; external "
            "multi-range requests can fail on phones")
    for forbidden in (
            "function syncAudio(",
            "elements.audio.playbackRate = userPlaybackRate",
            'elements.video.addEventListener("timeupdate", () => { syncAudio'):
        if forbidden in player:
            errors.append(
                "normal playback may seek or rate-shift narration on phones: "
                f"{forbidden}")
    index = (WEB / "index.html").read_text(errors="ignore")
    player_cache_key = re.search(
        r'<script src="app_v2\.js\?v=([^"&]+)"></script>', index)
    if not player_cache_key:
        errors.append("tutorial player script is missing its phone cache-buster")
    elif player_cache_key.group(1) == "20260810-live-qa":
        errors.append("tutorial player still uses the superseded live-QA cache key")
    audio_tag = (index.split('<audio id="narration-audio"', 1)[1]
                 .split(">", 1)[0]
                 if '<audio id="narration-audio"' in index else "")
    if 'crossorigin="anonymous"' not in audio_tag:
        errors.append("narration audio element is missing its CORS contract")

    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors[:100]))
        print(f"errors={len(errors)}")
        return 1
    voice_count = sum(len(voices) for _, voices in LANGUAGES.values())
    print(
        f"verified lessons={expected} languages={len(LANGUAGES)} "
        f"voices={voice_count} tracks={expected * voice_count} "
        f"videos_4k={expected}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
