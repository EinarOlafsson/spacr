"""Draw `volcano_explorer.png`, the Volcano Explorer module icon.

Asked for on 2026-08-31: "also give the volcano explorer a volcano plot
schematic icon." Until then the module borrowed a generated placeholder,
which said nothing about what it does.

KEPT AS A SCRIPT, not just the PNG it produces. The shipped icons are
1024-square white-on-transparent glyphs, and a raster is not editable:
changing the threshold rule or the point layout means redrawing, and
without the source that means redrawing by hand in another program. This
also makes the file reproducible -- the point positions are written down
rather than random, so re-running produces the same bytes.

    python tools/draw_volcano_explorer_icon.py

The shape IS the plot: a dense cloud of non-significant points along the
bottom, a dashed significance rule, and two wings of larger points
climbing away to high absolute fold change. That silhouette is what makes
a volcano plot recognisable at 48 px, which is the size a fold button
draws it at.
"""
from __future__ import annotations

import os

from PIL import Image, ImageDraw

#: Shipped icon size, and the supersample factor used to antialias.
SIZE = 1024
SUPERSAMPLE = 4

#: The icon language of every other file in the folder: pure white, with
#: shape carried entirely by the alpha channel so the theme can tint it.
INK = (255, 255, 255, 255)

#: Non-significant points: wide along the fold-change axis, low, thinning
#: as they climb. ``(x in [-1, 1], y in [0, 1])``.
CLOUD = (
    (-0.52, 0.04), (-0.36, 0.09), (-0.21, 0.03), (-0.07, 0.10),
    (0.08, 0.05), (0.23, 0.11), (0.38, 0.04), (0.53, 0.09),
    (-0.45, 0.18), (-0.29, 0.23), (-0.13, 0.17), (0.02, 0.22),
    (0.17, 0.16), (0.32, 0.21), (0.47, 0.17),
    (-0.34, 0.32), (-0.17, 0.29), (0.00, 0.35), (0.19, 0.30),
    (0.35, 0.33),
)

#: The hits: above the rule, out at the extremes, drawn larger because
#: they are what the plot is read for.
WINGS = (
    (-0.63, 0.55), (-0.75, 0.66), (-0.87, 0.81), (-0.70, 0.90),
    (0.62, 0.53), (0.74, 0.65), (0.86, 0.80), (0.69, 0.89),
)


def draw(path: str) -> None:
    """Render the icon to ``path``."""
    width = SIZE * SUPERSAMPLE
    image = Image.new("RGBA", (width, width), (255, 255, 255, 0))
    pen = ImageDraw.Draw(image)

    margin = int(width * 0.16)
    bottom, left = width - margin, margin
    stroke = int(width * 0.028)

    # The axis pair, as an L. Two lines read as "a plot" faster than a
    # full frame does, and leave more room for the points.
    pen.line([(left, int(margin * 0.75)), (left, bottom)],
             fill=INK, width=stroke)
    pen.line([(left, bottom), (width - int(margin * 0.75), bottom)],
             fill=INK, width=stroke)

    # The significance threshold, dashed: the one line that turns a
    # scatter into a VOLCANO plot rather than any other scatter.
    rule_y = int(margin + (bottom - margin) * 0.46)
    dash, gap = int(width * 0.045), int(width * 0.030)
    end = width - int(margin * 0.85)
    x = left + dash
    while x < end:
        pen.line([(x, rule_y), (min(x + dash, end), rule_y)],
                 fill=(255, 255, 255, 200), width=int(stroke * 0.7))
        x += dash + gap

    centre = (left + (width - margin)) / 2.0
    half = (width - margin - left) / 2.0

    def dot(fraction_x: float, fraction_y: float, radius: int) -> None:
        """One point, placed in plot coordinates rather than pixels."""
        x_px = centre + fraction_x * half * 0.88
        y_px = bottom - fraction_y * (bottom - margin) * 0.94
        pen.ellipse([x_px - radius, y_px - radius,
                     x_px + radius, y_px + radius], fill=INK)

    for fraction_x, fraction_y in CLOUD:
        dot(fraction_x, fraction_y, int(width * 0.019))
    for fraction_x, fraction_y in WINGS:
        dot(fraction_x, fraction_y, int(width * 0.032))

    image.resize((SIZE, SIZE), Image.LANCZOS).save(path)


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    draw(os.path.join(here, "spacr", "resources", "icons",
                      "volcano_explorer.png"))
    print("wrote volcano_explorer.png")
