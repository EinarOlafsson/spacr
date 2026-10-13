#!/usr/bin/env python3
"""Render pointer-and-spotlight videos from screenshot keyframes."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw


ACCENT = (74, 158, 255, 220)
CLICK_POINT = (255, 0, 153, 255)
DIM_ALPHA = 150


def _ease(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return 0.5 * (1.0 - math.cos(math.pi * value))


def _spotlight(
    frame: Image.Image,
    focus: list[int],
    opacity: int,
    render_scale: float = 1.0,
) -> None:
    x, y, width, height = focus
    padding = max(1, int(round(10 * render_scale)))
    mask_radius = max(1, int(round(12 * render_scale)))
    ring_offset = max(1, int(round(4 * render_scale)))
    ring_radius = max(1, int(round(10 * render_scale)))
    ring_width = max(1, int(round(render_scale)))
    mask = Image.new("L", frame.size, opacity)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle(
        (x - padding, y - padding,
         x + width + padding, y + height + padding),
        radius=mask_radius,
        fill=0,
    )
    dim = Image.new("RGBA", frame.size, (0, 0, 0, 255))
    dim.putalpha(mask)
    frame.alpha_composite(dim)

    ring = ImageDraw.Draw(frame)
    ring.rounded_rectangle(
        (x - ring_offset, y - ring_offset,
         x + width + ring_offset, y + height + ring_offset),
        radius=ring_radius,
        outline=ACCENT,
        width=ring_width,
    )


def _pointer(
    frame: Image.Image,
    position: tuple[float, float],
    render_scale: float = 1.0,
) -> None:
    """Paint the unobtrusive marker used only for scripted clicks."""
    x, y = (int(position[0]), int(position[1]))
    draw = ImageDraw.Draw(frame)
    radius = max(2, int(round(4 * render_scale)))
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=CLICK_POINT,
    )


def _ffmpeg_writer(output: Path, size: tuple[int, int], fps: int):
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{size[0]}x{size[1]}", "-r", str(fps),
        "-i", "-", "-an", "-c:v", "libx264",
        # CRF controls visual quality; the fast preset retains the same sharp
        # target while making repeated native-4K tutorial renders practical.
        "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(output),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenes", type=Path)
    parser.add_argument("--timings", type=Path, required=True)
    parser.add_argument("--silent-output", type=Path, required=True)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--output-size",
        help="Override output dimensions, for example 3840x2160. Scene "
             "coordinates remain relative to the JSON size.",
    )
    args = parser.parse_args()

    spec = json.loads(args.scenes.read_text())
    timing_spec = json.loads(args.timings.read_text())
    scenes = spec["scenes"]
    timings = timing_spec["scenes"]
    if len(scenes) != len(timings):
        raise ValueError("scene and timing counts differ")

    fps = int(spec.get("fps", 30))
    logical_size = tuple(spec.get("size", [1920, 1080]))
    if args.output_size:
        try:
            width, height = (
                int(value) for value in args.output_size.lower().split("x", 1)
            )
        except (TypeError, ValueError) as error:
            raise ValueError("--output-size must use WIDTHxHEIGHT") from error
        if width <= 0 or height <= 0:
            raise ValueError("--output-size dimensions must be positive")
        size = (width, height)
    else:
        size = logical_size
    scale_x = size[0] / logical_size[0]
    scale_y = size[1] / logical_size[1]
    render_scale = min(scale_x, scale_y)
    root = args.scenes.parent
    writer = _ffmpeg_writer(args.silent_output, size, fps)
    previous_target = (
        size[0] - 55.0 * scale_x,
        size[1] - 55.0 * scale_y,
    )

    try:
        for scene, timing in zip(scenes, timings):
            image_path = root / scene["image"]
            base = Image.open(image_path).convert("RGBA").resize(
                size, Image.Resampling.LANCZOS)
            motion_frames = []
            if scene.get("motion_dir"):
                motion_root = root / scene["motion_dir"]
                motion_frames = [
                    Image.open(path).convert("RGBA").resize(
                        size, Image.Resampling.LANCZOS
                    )
                    for path in sorted(motion_root.glob("*.png"))
                ]
            show_pointer = bool(scene.get("pointer", False))
            logical_target = scene.get("target")
            target = (
                (float(logical_target[0]) * scale_x,
                 float(logical_target[1]) * scale_y)
                if logical_target is not None else previous_target
            )
            duration = float(timing["duration"])
            frame_count = max(1, int(round(duration * fps)))
            # Give the viewer time to follow every transition. Long scenes
            # use a 2.4-second move; short scenes still receive at least a
            # 1.6-second eased move instead of a jump.
            travel_seconds = min(2.4, max(1.6, duration * 0.22))

            scaled_focus = None
            steady = base
            if scene.get("focus"):
                scaled_focus = [
                    int(round(scene["focus"][0] * scale_x)),
                    int(round(scene["focus"][1] * scale_y)),
                    int(round(scene["focus"][2] * scale_x)),
                    int(round(scene["focus"][3] * scale_y)),
                ]
                steady = base.copy()
                _spotlight(steady, scaled_focus, DIM_ALPHA, render_scale)
            steady_bytes = (
                None if show_pointer or motion_frames
                else steady.convert("RGB").tobytes()
            )

            for frame_index in range(frame_count):
                if motion_frames:
                    # Captured at roughly 12.5 fps; repeat each source frame
                    # twice at 30 fps for smooth, faithful movement without
                    # inventing a second DNA animation in post-production.
                    base = motion_frames[(frame_index // 2) % len(motion_frames)]
                    steady = base.copy()
                    if scaled_focus is not None:
                        _spotlight(steady, scaled_focus, DIM_ALPHA, render_scale)
                elapsed = frame_index / fps
                progress = _ease(elapsed / max(travel_seconds, 0.001))
                position = (
                    previous_target[0]
                    + (target[0] - previous_target[0]) * progress,
                    previous_target[1]
                    + (target[1] - previous_target[1]) * progress,
                )
                if scaled_focus is not None and elapsed < 0.25:
                    frame = base.copy()
                    fade = _ease(elapsed / 0.25)
                    _spotlight(
                        frame,
                        scaled_focus,
                        int(round(DIM_ALPHA * fade)),
                        render_scale,
                    )
                elif show_pointer or motion_frames:
                    frame = steady.copy()
                else:
                    writer.stdin.write(steady_bytes)
                    continue
                if show_pointer:
                    _pointer(frame, position, render_scale)
                writer.stdin.write(frame.convert("RGB").tobytes())
            if show_pointer:
                previous_target = target
    finally:
        if writer.stdin is not None:
            writer.stdin.close()
    return_code = writer.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg video writer failed with {return_code}")

    if args.audio and args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(args.silent_output),
            "-i", str(args.audio),
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart",
            str(args.output),
        ], check=True)
        print(args.output)
    print(args.silent_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
