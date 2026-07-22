"""CLI entry point for spacr-tutorial.

Usage:
    spacr-tutorial mask
    spacr-tutorial all
    spacr-tutorial home --out ~/tutorials --voice ~/.spacr/piper/foo.onnx
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .engine import render_tutorial
from .scripts import AVAILABLE_TUTORIALS


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="spacr-tutorial",
                                 description=__doc__.splitlines()[0])
    p.add_argument("app",
                    help="Which tutorial to render. One of: "
                         + ", ".join(AVAILABLE_TUTORIALS) + ", or 'all'")
    p.add_argument("--out", type=Path,
                    default=Path.home() / "spacr-tutorials",
                    help="Where to write the MP4 + SRT (default: "
                         "~/spacr-tutorials)")
    p.add_argument("--voice", type=Path,
                    default=None,
                    help="Path to a Piper .onnx voice model. Defaults "
                         "to ~/.spacr/piper/en_US-lessac-medium.onnx.")
    p.add_argument("--length-scale", type=float, default=1.0,
                    help="Piper length scale. Lower = faster speech.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    targets = AVAILABLE_TUTORIALS if args.app == "all" else [args.app]
    for name in targets:
        if name not in AVAILABLE_TUTORIALS:
            print(f"unknown tutorial: {name}", file=sys.stderr)
            return 2
        print(f"rendering {name}…")
        result = render_tutorial(
            name, out_dir=args.out,
            voice_model=args.voice,
            length_scale=args.length_scale,
        )
        print(f"  → {result.mp4} ({result.duration_s:.1f}s, "
                f"{result.frames} frames)")
        print(f"  → {result.srt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
