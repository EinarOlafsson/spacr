#!/usr/bin/env python3
"""Build scene specifications for every tutorial visual master."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "production"
CATALOG = ROOT / "catalog" / "lessons_en.json"


def selected_lesson_ids(value: str | None, lessons: list[dict]) -> set[str]:
    """Resolve a bounded number/range/id selection against ``lessons``."""
    if not value:
        return {lesson["id"] for lesson in lessons}
    numbers: set[int] = set()
    identifiers: set[str] = set()
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            continue
        if token.isdigit():
            numbers.add(int(token))
        elif "-" in token and all(part.isdigit()
                                   for part in token.split("-", 1)):
            start, end = (int(part) for part in token.split("-", 1))
            if end < start:
                raise ValueError(f"descending lesson range: {token}")
            numbers.update(range(start, end + 1))
        else:
            identifiers.add(token)
    selected = {
        lesson["id"] for lesson in lessons
        if lesson["number"] in numbers or lesson["id"] in identifiers
    }
    requested = identifiers | {str(number) for number in numbers}
    matched = identifiers & selected
    matched.update(
        str(lesson["number"]) for lesson in lessons
        if lesson["number"] in numbers
    )
    missing = requested - matched
    if missing:
        raise ValueError("unknown lesson selection: " + ", ".join(sorted(missing)))
    if not selected:
        raise ValueError("lesson selection is empty")
    return selected


def _fallback_focus(geometry: dict) -> list[int]:
    return geometry.get("overview") or [440, 66, 3400, 2050]


def _focus_for(visual: str, geometry: dict) -> list[int] | None:
    if visual == "overview":
        return None
    return geometry.get(visual) or geometry.get("settings") or _fallback_focus(geometry)


def build_app_lesson(lesson: dict) -> None:
    lesson_root = PRODUCTION / lesson["id"]
    keyframes = lesson_root / "keyframes"
    geometry = json.loads((keyframes / "geometry.json").read_text())
    frame_size = geometry.get("frame_size", [3840, 2160])
    scenes = []
    for source in lesson["scenes"]:
        visual = source["visual"]
        scene = {
            "image": "keyframes/01_module.png",
            "narration": source["narration"],
            "speech_text": source.get("speech_text", source["narration"]),
            "hold_after": source.get("hold_after", 0.45),
            "pointer": False,
        }
        focus = _focus_for(visual, geometry)
        if focus:
            scene["focus"] = focus
        scenes.append(scene)

    (lesson_root / "scenes.json").write_text(json.dumps({
        "schema": 1,
        "fps": 30,
        "size": frame_size,
        "lesson": lesson["id"],
        "scenes": scenes,
    }, indent=2, ensure_ascii=False) + "\n")


def build_hosted_app_lesson(lesson: dict) -> None:
    """Build a current folded lesson from its host-routed module capture.

    Older deep captures show the retired capability as a Home/sidebar module.
    The bounded consolidation refresh captures the capability inside its live
    host and intentionally keeps the existing scene count and narration, so
    every published voice timing remains compatible with the new visual.
    """
    lesson_root = PRODUCTION / lesson["id"]
    keyframes = lesson_root / "keyframes"
    geometry = json.loads((keyframes / "geometry.json").read_text())
    if geometry.get("frames"):
        raise RuntimeError(
            f"{lesson['id']} still has a standalone multi-frame capture; "
            f"recapture {lesson['app_key']!r} through its current host "
            f"{lesson['host_app_key']!r}"
        )
    captured_route = (
        geometry.get("app_key"), geometry.get("host_app_key")
    )
    expected_route = (lesson["app_key"], lesson["host_app_key"])
    # The first consolidated single-frame captures predate explicit route
    # metadata. They are distinguishable from retired standalone captures by
    # the absence of ``frames`` above. New captures always record both keys;
    # once present, they must match exactly.
    if captured_route != (None, None) and captured_route != expected_route:
        raise RuntimeError(
            f"{lesson['id']} capture route is {captured_route!r}, expected "
            f"{expected_route!r}; rerun capture_all_modules.py --force"
        )
    image = keyframes / "01_module.png"
    if not image.is_file():
        raise FileNotFoundError(
            f"{image} is required for hosted lesson {lesson['id']}")

    output_terms = {
        "console", "figures", "metrics", "previews", "benchmark",
        "confusion", "review", "crop", "results", "output", "pairwise",
    }
    run_terms = {
        "actions", "compute", "test", "compare", "comparison", "dialog",
        "run",
    }
    input_terms = {
        "input", "inputs", "source", "source_controls", "scan_controls",
        "field_controls",
    }
    overview_terms = {"overview", "screen", "catalogue"}
    scenes = []
    for index, source in enumerate(lesson["scenes"]):
        semantic = str(source.get("focus_key") or source["visual"])
        if semantic in overview_terms or index == 0:
            role = "overview"
        elif semantic in output_terms or index == len(lesson["scenes"]) - 1:
            role = "output"
        elif semantic in run_terms or index == len(lesson["scenes"]) - 2:
            role = "run"
        elif semantic in input_terms or index == 1:
            role = "input"
        else:
            role = "settings"
        scene = {
            "image": "keyframes/01_module.png",
            "narration": source["narration"],
            "speech_text": source.get("speech_text", source["narration"]),
            "hold_after": source.get("hold_after", 0.45),
            "pointer": False,
        }
        focus = None if role == "overview" else (
            geometry.get(role) or geometry.get("overview"))
        if focus:
            scene["focus"] = focus
        scenes.append(scene)
    (lesson_root / "scenes.json").write_text(json.dumps({
        "schema": 1,
        "fps": 30,
        "size": geometry.get("frame_size", [3840, 2160]),
        "lesson": lesson["id"],
        "host_app_key": lesson["host_app_key"],
        "scenes": scenes,
    }, indent=2, ensure_ascii=False) + "\n")


def build_home_lesson(lesson: dict) -> None:
    """Use the captured Home overview and a real opened-module screen."""
    lesson_root = PRODUCTION / lesson["id"]
    keyframes = lesson_root / "keyframes"
    performance_4k = keyframes / "02_performance_4k.png"
    opened_4k = keyframes / "03_mask_open_4k.png"
    geometry = json.loads((keyframes / "geometry.json").read_text())
    frame_size = geometry.get("frame_size", [3840, 2160])
    if frame_size != [3840, 2160]:
        raise RuntimeError(
            f"{lesson['id']} Home capture is not native 4K: {frame_size}"
        )
    if not performance_4k.is_file():
        raise FileNotFoundError(
            f"{performance_4k} is required; rerun capture_all_modules.py for Home"
        )
    if not opened_4k.is_file():
        raise FileNotFoundError(
            f"{opened_4k} is required; rerun capture_all_modules.py for Home"
        )
    if "performance" not in geometry or "open_module" not in geometry:
        raise RuntimeError(
            f"{lesson['id']} geometry is missing a Preferences or opened-module region"
        )
    visuals = {
        "overview": ("keyframes/01_module.png", None),
        "navigation": ("keyframes/01_module.png", geometry["nav"]),
        "modules": ("keyframes/01_module.png", geometry["modules"]),
        "performance": ("keyframes/02_performance_4k.png", geometry["performance"]),
        "open_module": ("keyframes/03_mask_open_4k.png", geometry["open_module"]),
    }
    scenes = []
    for source in lesson["scenes"]:
        image, focus = visuals[source["visual"]]
        scene = {
            "image": image,
            "narration": source["narration"],
            "speech_text": source.get("speech_text", source["narration"]),
            "hold_after": source.get("hold_after", 0.45),
            "pointer": False,
        }
        if focus:
            scene["focus"] = focus
        scenes.append(scene)
    (lesson_root / "scenes.json").write_text(json.dumps({
        "schema": 1, "fps": 30, "size": frame_size,
        "lesson": lesson["id"], "scenes": scenes,
    }, indent=2, ensure_ascii=False) + "\n")


def build_captured_lesson(lesson: dict) -> None:
    lesson_root = PRODUCTION / lesson["id"]
    geometry = json.loads((lesson_root / "keyframes" / "geometry.json").read_text())
    frames = geometry.get("frames")
    if not isinstance(frames, dict):
        # The deep Batch Runner capture predates schema 2: every narrated
        # state already has its own named 4K keyframe, while its geometry file
        # stores the stable screen regions once at the top level.  Preserve
        # that richer twelve-state capture instead of collapsing it back to
        # the old one-image app template.
        frames = {}
        for source in lesson["scenes"]:
            visual = source["visual"]
            image = lesson_root / "keyframes" / f"{visual}.png"
            if not image.is_file():
                raise RuntimeError(
                    f"{lesson['id']} legacy capture is missing {visual!r}")
            frames[visual] = {
                "file": image.name,
                "screen": geometry.get("overview"),
                "editor": geometry.get("input"),
                "table": geometry.get("output"),
                "toolbar": geometry.get("settings"),
                "run": geometry.get("run"),
                "summary": geometry.get("output"),
            }
    scenes = []
    for source in lesson["scenes"]:
        visual = source["visual"]
        frame = frames.get(visual)
        if not isinstance(frame, dict):
            image = lesson_root / "keyframes" / f"{visual}.png"
            if not image.is_file():
                raise RuntimeError(
                    f"{lesson['id']} capture geometry is missing {visual!r}")
            # Some completed-run frames predate per-frame geometry.  The
            # native 4K image is still authoritative; render it full-frame
            # instead of throwing away the real result or substituting a
            # generic settings screenshot.
            frame = {"file": image.name}
        image = frame.get("file")
        if not isinstance(image, str):
            # Captures made before schema 2 reserved ``file`` may have used
            # ``image`` for either the filename or the source-canvas rect.
            image = f"{visual}.png"
        focus_key = source.get("focus_key")
        click_key = source.get("click_key")
        focus = frame.get(focus_key) if focus_key else None
        click_rect = frame.get(click_key) if click_key else None
        scene = {
            "image": f"keyframes/{image}",
            "narration": source["narration"],
            "speech_text": source.get("speech_text", source["narration"]),
            "hold_after": source.get("hold_after", 0.45),
            "pointer": bool(click_rect),
        }
        motion_dir = lesson_root / "keyframes" / "motion" / visual
        if motion_dir.is_dir() and any(motion_dir.glob("*.png")):
            scene["motion_dir"] = str(
                motion_dir.relative_to(lesson_root)
            )
        if focus:
            scene["focus"] = focus
        if click_rect:
            scene["target"] = [
                click_rect[0] + click_rect[2] / 2,
                click_rect[1] + click_rect[3] / 2,
            ]
        scenes.append(scene)
    (lesson_root / "scenes.json").write_text(json.dumps({
        "schema": 1, "fps": 30,
        "size": geometry.get("frame_size", [3840, 2160]),
        "lesson": lesson["id"], "scenes": scenes,
    }, indent=2, ensure_ascii=False) + "\n")


def build_api_lesson(lesson: dict) -> None:
    lesson_root = PRODUCTION / lesson["id"]
    images = sorted((lesson_root / "keyframes").glob("*.png"))
    if len(images) != len(lesson["scenes"]):
        raise RuntimeError("API lesson needs five terminal keyframes")
    focus = [360, 430, 3120, 1380]
    scenes = []
    for image, source in zip(images, lesson["scenes"]):
        scenes.append({
            "image": str(image.relative_to(lesson_root)),
            "narration": source["narration"],
            "speech_text": source.get("speech_text", source["narration"]),
            "hold_after": source.get("hold_after", 0.45),
            "focus": focus,
            "pointer": False,
        })
    (lesson_root / "scenes.json").write_text(json.dumps({
        "schema": 1,
        "fps": 30,
        "size": [3840, 2160],
        "lesson": lesson["id"],
        "scenes": scenes,
    }, indent=2, ensure_ascii=False) + "\n")


def build_intro_lesson(lesson: dict) -> None:
    lesson_root = PRODUCTION / lesson["id"]
    geometry_path = lesson_root / "keyframes" / "geometry.json"
    geometry = json.loads(geometry_path.read_text())
    visual_scenes = geometry["scenes"]
    if len(visual_scenes) != len(lesson["scenes"]):
        raise RuntimeError(
            f"{lesson['id']} has {len(lesson['scenes'])} narration scenes but "
            f"{len(visual_scenes)} visual scenes"
        )
    scenes = []
    for visual, source in zip(visual_scenes, lesson["scenes"]):
        scene = {
            "image": visual["image"],
            "narration": source["narration"],
            "speech_text": source.get("speech_text", source["narration"]),
            "hold_after": source.get("hold_after", 0.45),
            "pointer": bool(visual.get("pointer", False)),
        }
        if visual.get("focus"):
            scene["focus"] = visual["focus"]
        if visual.get("target"):
            scene["target"] = visual["target"]
        scenes.append(scene)
    (lesson_root / "scenes.json").write_text(json.dumps({
        "schema": 1,
        "fps": 30,
        "size": geometry["frame_size"],
        "lesson": lesson["id"],
        "scenes": scenes,
    }, indent=2, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lessons",
        help="comma-separated lesson numbers, numeric ranges, or exact ids",
    )
    args = parser.parse_args()
    catalog = json.loads(CATALOG.read_text())
    wanted = selected_lesson_ids(args.lessons, catalog["lessons"])
    for lesson in catalog["lessons"]:
        if lesson["id"] not in wanted:
            continue
        if lesson["number"] <= 4:
            build_intro_lesson(lesson)
        elif lesson["number"] == 5:
            build_home_lesson(lesson)
        elif lesson["number"] == 6:
            build_api_lesson(lesson)
        elif lesson.get("host_app_key"):
            build_hosted_app_lesson(lesson)
        elif lesson["number"] >= 7:
            build_captured_lesson(lesson)
        else:
            build_app_lesson(lesson)
        print(lesson["id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
