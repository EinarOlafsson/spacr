#!/usr/bin/env python3
"""Render a sharp 4K silent master without streaming duplicate still frames.

Static spotlight frames are composited once. Only the brief magenta click-dot
movement is rendered as an image sequence; ffmpeg loops every other frame.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw


ACCENT = (74, 158, 255, 220)
CLICK_POINT = (255, 0, 153, 255)
DIM_ALPHA = 150
TRACK_TIMESCALE = "90000"


def ease(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return 0.5 * (1.0 - math.cos(math.pi * value))


def spotlight(frame: Image.Image, focus: list[int]) -> None:
    x, y, width, height = focus
    mask = Image.new("L", frame.size, DIM_ALPHA)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle(
        (x - 20, y - 20, x + width + 20, y + height + 20),
        radius=24,
        fill=0,
    )
    dim = Image.new("RGBA", frame.size, (0, 0, 0, 255))
    dim.putalpha(mask)
    frame.alpha_composite(dim)
    ImageDraw.Draw(frame).rounded_rectangle(
        (x - 8, y - 8, x + width + 8, y + height + 8),
        radius=20,
        outline=ACCENT,
        width=2,
    )


def pointer(frame: Image.Image, position: tuple[float, float]) -> None:
    x, y = (int(round(position[0])), int(round(position[1])))
    radius = 8  # four logical pixels at the 2x 4K capture scale
    ImageDraw.Draw(frame).ellipse(
        (x - radius, y - radius, x + radius, y + radius), fill=CLICK_POINT
    )


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def validate_geometry(spec: dict, root: Path) -> tuple[int, int]:
    """Fail closed when a focus box or pointer target leaves the frame."""
    size = tuple(int(value) for value in spec.get("size", (3840, 2160)))
    if len(size) != 2 or min(size) <= 0:
        raise ValueError(f"invalid tutorial frame size: {size}")
    width, height = size
    for index, scene in enumerate(spec["scenes"], start=1):
        image_path = root / scene["image"]
        with Image.open(image_path) as image:
            if image.size != size:
                raise ValueError(
                    f"scene {index} image is {image.size}, expected {size}: "
                    f"{image_path}"
                )
        if scene.get("focus") is not None:
            x, y, box_width, box_height = (
                int(value) for value in scene["focus"]
            )
            if (
                x < 0
                or y < 0
                or box_width <= 0
                or box_height <= 0
                or x + box_width > width
                or y + box_height > height
            ):
                raise ValueError(
                    f"scene {index} focus {scene['focus']} leaves "
                    f"{width}x{height} frame"
                )
        if scene.get("target") is not None:
            x, y = (float(value) for value in scene["target"])
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(
                    f"scene {index} target {scene['target']} leaves "
                    f"{width}x{height} frame"
                )
    return width, height


def encode_still(image: Path, duration: float, output: Path, fps: int) -> None:
    run([
        "ffmpeg", "-y", "-loglevel", "error", "-loop", "1",
        "-framerate", str(fps), "-i", str(image), "-t", f"{duration:.6f}",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage",
        "-threads", "2", "-crf", "18",
        "-pix_fmt", "yuv420p", "-r", str(fps),
        "-video_track_timescale", TRACK_TIMESCALE, str(output),
    ])


def encode_pointer_scene(
    base: Image.Image,
    start: tuple[float, float],
    target: tuple[float, float],
    duration: float,
    output: Path,
    work: Path,
    fps: int,
) -> None:
    travel = min(2.4, max(1.6, duration * 0.22))
    moving_frames = max(1, int(round(travel * fps)))
    sequence = work / f"{output.stem}_frames"
    sequence.mkdir()
    for index in range(moving_frames):
        progress = ease(index / max(1, moving_frames - 1))
        position = (
            start[0] + (target[0] - start[0]) * progress,
            start[1] + (target[1] - start[1]) * progress,
        )
        frame = base.copy()
        pointer(frame, position)
        frame.convert("RGB").save(sequence / f"{index:04d}.jpg", quality=96,
                                  subsampling=0)

    moving = work / f"{output.stem}_moving.mp4"
    run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
        "-i", str(sequence / "%04d.jpg"), "-an", "-c:v", "libx264",
        "-preset", "veryfast", "-threads", "2", "-crf", "18", "-pix_fmt", "yuv420p",
        "-r", str(fps), "-video_track_timescale", TRACK_TIMESCALE, str(moving),
    ])
    hold = max(0.0, duration - moving_frames / fps)
    if hold < 1 / fps:
        moving.replace(output)
        return
    final_image = work / f"{output.stem}_final.png"
    final = base.copy()
    pointer(final, target)
    final.convert("RGB").save(final_image, compress_level=2)
    held = work / f"{output.stem}_held.mp4"
    encode_still(final_image, hold, held, fps)
    concat([moving, held], output, work / f"{output.stem}_concat.txt")


def concat(parts: list[Path], output: Path, listing: Path) -> None:
    listing.write_text("".join(f"file '{part.as_posix()}'\n" for part in parts))
    run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
        "-i", str(listing), "-an", "-c", "copy", "-movflags", "+faststart",
        str(output),
    ])


def frame_aligned_durations(scenes: list[dict], fps: int) -> list[float]:
    """Quantize cumulative scene boundaries, avoiding per-scene drift."""
    if fps <= 0:
        raise ValueError(f"invalid tutorial frame rate: {fps}")
    cumulative_seconds = 0.0
    emitted_frames = 0
    durations = []
    for index, scene in enumerate(scenes, start=1):
        duration = float(scene["duration"])
        if duration <= 0:
            raise ValueError(f"scene {index} has invalid duration: {duration}")
        cumulative_seconds += duration
        boundary_frames = max(
            emitted_frames + 1,
            int(round(cumulative_seconds * fps)),
        )
        durations.append((boundary_frames - emitted_frames) / fps)
        emitted_frames = boundary_frames
    return durations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenes", type=Path)
    parser.add_argument("--timings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spec = json.loads(args.scenes.read_text())
    timing = json.loads(args.timings.read_text())
    if len(spec["scenes"]) != len(timing["scenes"]):
        raise ValueError("scene and timing counts differ")
    fps = int(spec.get("fps", 30))
    root = args.scenes.parent
    validate_geometry(spec, root)
    durations = frame_aligned_durations(timing["scenes"], fps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staged_output = args.output.with_name(
        f".{args.output.stem}.part{args.output.suffix}"
    )
    staged_output.unlink(missing_ok=True)

    try:
        with tempfile.TemporaryDirectory(prefix="spacr_visual_") as temp:
            work = Path(temp)
            parts = []
            previous = (3792.0, 2112.0)
            for index, (scene, duration) in enumerate(
                zip(spec["scenes"], durations), start=1
            ):
                base = Image.open(root / scene["image"]).convert("RGBA")
                if scene.get("focus"):
                    spotlight(base, [int(v) for v in scene["focus"]])
                part = work / f"scene_{index:02d}.mp4"
                if scene.get("pointer") and scene.get("target"):
                    target = tuple(float(v) for v in scene["target"])
                    encode_pointer_scene(
                        base, previous, target, duration, part, work, fps
                    )
                    previous = target
                else:
                    image = work / f"scene_{index:02d}.png"
                    base.convert("RGB").save(image, compress_level=2)
                    encode_still(image, duration, part, fps)
                parts.append(part)
            concat(parts, staged_output, work / "master_concat.txt")
        staged_output.replace(args.output)
    finally:
        staged_output.unlink(missing_ok=True)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
