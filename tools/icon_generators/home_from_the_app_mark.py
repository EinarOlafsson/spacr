#!/usr/bin/env python3
"""``home.png`` — the dock's Home row, cut from the application's own mark.

Instruction 369, ask 1: "the home icon should be the spacr icon". Measured
2026-09-02: of 40 dock rows checked, 39 draw exactly what the Home screen
draws for the same key, and one does not -- ``__home__``, because
``iconset.bundled_icon_path("home")`` answered ``None`` and the row fell
back.

WHY THIS IS A DERIVED FILE RATHER THAN A COPY, which 369 asks to be decided
before wiring anything: ``app_icon.png`` is a 1024x1024 TILE -- a teal
rounded square, 97% opaque, carrying white line-art of a cell. It is the
right thing for a desktop launcher and the wrong thing for this dock. Every
other dock row is a monochrome ALPHA MASK that ``iconset`` re-inks for the
active theme, which is also the mechanism 369's ask 3 needs for "white to
blue on hover". Dropping the tile in as drawn would have given the Home row
a solid teal square that ignores all four themes and cannot change ink.

So the MARK is kept and the TILE is discarded: alpha is taken from how far
each pixel rises above the teal ground, which leaves the white drawing and
nothing else. ``app_icon.png`` is untouched and stays the launcher icon --
it is one of the four assets ``test_the_bundled_set_is_a_monochrome_alpha_
mask`` exempts, and it must go on being exempt.

Run: python tools/icon_generators/home_from_the_app_mark.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "spacr" / "resources" / "icons" / "app_icon.png"
TARGET = ROOT / "spacr" / "resources" / "icons" / "home.png"

#: Below this, a pixel is the teal ground or one of the faint interior grid
#: lines rather than the drawing. The grid is deliberate in the launcher
#: tile and is noise at 26 px, where it would read as a smudge inside the
#: cell instead of as squares.
INK_FLOOR = 0.55

#: How many of the drawing's parts to keep, largest first.
#:
#: THE MARK DOES NOT SURVIVE 26 px AS DRAWN, and that was measured rather
#: than assumed. The full drawing is a cell outline, a nucleus, an interior
#: grid, three parasites and four dots -- ten separate parts. Downscaled to
#: the dock's 26 px, NO PIXEL EVEN REACHES ALPHA 200: the strokes are about
#: 8 px wide in a 1024 px tile, which is a fifth of a pixel at dock size.
#: Dilating the whole drawing until it was visible turned it into a blob at
#: every radius tried (8, 16, 24, 32).
#:
#: So the mark is SIMPLIFIED rather than shrunk, which is what a favicon or
#: an app glyph normally is: the two largest parts are the cell outline
#: (20,068 px) and its nucleus (7,574 px), and those two ARE the mark's
#: identity. The grid, the parasites and the dots are detail that only
#: exists at launcher size. The result reads as a cell with a nucleus at
#: 26 px and matches the line-art weight of every other dock icon, which
#: none of the alternatives did.
KEEP_PARTS = 2

#: Dilation passes at source resolution, to bring the surviving strokes up
#: to the weight the other dock glyphs are drawn at.
DILATE_PASSES = 2
DILATE_KERNEL = 15


def main() -> int:
    art = np.asarray(Image.open(SOURCE).convert("RGBA")).astype(np.float32)
    rgb, alpha = art[..., :3], art[..., 3]

    # The ground is the most common opaque colour: the tile fill.
    opaque = alpha > 8
    flat = rgb[opaque].astype(np.uint8)
    ground = np.median(flat.reshape(-1, 3), axis=0)

    lum = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    ground_lum = float(ground @ np.array([0.2126, 0.7152, 0.0722]))
    rise = (lum - ground_lum) / max(1.0, 255.0 - ground_lum)
    rise = np.clip((rise - INK_FLOOR) / (1.0 - INK_FLOOR), 0.0, 1.0)
    rise *= (alpha / 255.0)

    # Keep the mark's two largest parts and drop the launcher-only detail.
    from PIL import ImageFilter
    from scipy import ndimage

    mask = rise > 0.38
    labelled, count = ndimage.label(mask)
    if count:
        areas = ndimage.sum(mask, labelled, range(1, count + 1))
        keep = np.argsort(-areas)[:KEEP_PARTS]
        kept = np.zeros_like(mask)
        for index in keep:
            kept |= labelled == index + 1
    else:                                    # pragma: no cover - defensive
        kept = mask

    thick = Image.fromarray((kept * 255).astype(np.uint8), "L")
    for _ in range(DILATE_PASSES):
        thick = thick.filter(ImageFilter.MaxFilter(DILATE_KERNEL))

    out = np.zeros(art.shape, dtype=np.uint8)
    out[..., :3] = 255                      # white ink; the theme re-inks it
    out[..., 3] = np.asarray(thick, dtype=np.uint8)
    Image.fromarray(out, "RGBA").save(TARGET)

    small = np.asarray(
        Image.fromarray(out, "RGBA").resize((26, 26), Image.LANCZOS))[..., 3]
    print(f"wrote {TARGET.relative_to(ROOT)}")
    print(f"ground rgb {tuple(int(v) for v in ground)}   parts kept "
          f"{KEEP_PARTS} of {count}")
    print(f"at 26 px: solid {float((small > 128).mean()):.1%}   "
          f"any ink {float((small > 8).mean()):.1%}   max alpha "
          f"{int(small.max())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
