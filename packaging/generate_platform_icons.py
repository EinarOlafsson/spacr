#!/usr/bin/env python3
"""Draw the three README installer-download icons.

The README advertises one download per platform. Those three links are drawn
marks rather than text, so the row reads as a row of equal choices.

House rules this file exists to keep:

* **One colour.** The artwork is pure white, nothing else.
* **One treatment.** All three are white *line* art at the same stroke width,
  not a mix of outlines and filled slabs. A solid four-pane Windows mark beside
  an outlined penguin puts three times the ink on Windows and turns a row of
  equals into a recommendation.
* **One weight.** ``main`` prints the white coverage of each glyph and fails if
  one falls outside the shared band, so no platform can quietly start shouting.
* **Visible in both README themes.** GitHub renders ``README.rst`` on white in
  light mode, where unbacked white art is invisible, and reStructuredText on
  GitHub cannot use the ``<picture>``/``prefers-color-scheme`` trick because
  its renderer runs docutils with raw HTML disabled. Each glyph therefore sits
  on the same dark rounded chip as ``app_icon.png``, which carries its own
  contrast into either theme.

Trademarks: Tux is free to use. The Apple and Windows marks are trademarks
whose use is restricted; pointing at an operating system's own download is the
nominative use every README on GitHub makes, and the fallback if it ever
matters is a generic laptop/window glyph.

Geometry is authored in a normalised 0..1 glyph box and mapped to pixels
*before* any union, subtraction or stroking happens. Qt flattens curves with a
tolerance expressed in the path's own coordinates, so a boolean operation run
in 0..1 space returns coarse polygons instead of curves.

Run::

    python packaging/generate_platform_icons.py
"""

from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import (  # noqa: E402
    QColor,
    QGuiApplication,
    QImage,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QTransform,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "spacr" / "resources" / "icons" / "platforms"

CANVAS = 512
#: identical to ``generate_app_icons.BACKGROUND``/``CORNER_RADIUS`` so the
#: download row belongs to the same family as the installed application icon
CHIP = QColor(0, 55, 55, 255)
CHIP_RADIUS = 188 * CANVAS / 1024
WHITE = QColor(255, 255, 255, 255)

#: stroke width of every line, in glyph-box units
STROKE = 0.062
#: white coverage band the three glyphs must share, as a fraction of the canvas
COVERAGE_LO = 0.100
COVERAGE_HI = 0.130


class Box:
    """A normalised 0..1 drawing box mapped onto the centre of the canvas."""

    def __init__(self, scale: float):
        self.size = scale * CANVAS
        origin = (CANVAS - self.size) / 2.0
        place = QTransform()
        place.translate(origin, origin)
        place.scale(self.size, self.size)
        self._place = place

    def ellipse(self, cx: float, cy: float, rx: float, ry: float,
                degrees: float = 0.0) -> QPainterPath:
        path = QPainterPath()
        path.addEllipse(QRectF(-rx, -ry, 2 * rx, 2 * ry))
        spin = QTransform()
        spin.translate(cx, cy)
        spin.rotate(degrees)
        return self._place.map(spin.map(path))

    def circle(self, cx: float, cy: float, r: float) -> QPainterPath:
        return self.ellipse(cx, cy, r, r)

    def round_rect(self, x: float, y: float, w: float, h: float,
                   radius: float) -> QPainterPath:
        path = QPainterPath()
        path.addRoundedRect(QRectF(x, y, w, h), radius, radius)
        return self._place.map(path)

    def lens(self, tip_a, tip_b, radius: float) -> QPainterPath:
        """A leaf: the lens where two equal circles through both tips meet."""
        (ax, ay), (bx, by) = tip_a, tip_b
        mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
        half = math.hypot(bx - ax, by - ay) / 2.0
        if radius <= half:
            raise ValueError("lens radius must exceed half the tip separation")
        offset = math.sqrt(radius * radius - half * half)
        ux, uy = (bx - ax) / (2 * half), (by - ay) / (2 * half)
        first = self.circle(mx - uy * offset, my + ux * offset, radius)
        second = self.circle(mx + uy * offset, my - ux * offset, radius)
        return first.intersected(second)

    def mirrored_outline(self, start, segments) -> QPainterPath:
        """Close a left-half cubic outline with its mirror image.

        ``segments`` are ``(control_1, control_2, end)`` triples walking down
        the left side; the right side is the same walk reflected about
        ``x = 0.5`` and traversed backwards, so the mark is exactly symmetric.
        """
        path = QPainterPath(QPointF(*start))
        for control_1, control_2, end in segments:
            path.cubicTo(QPointF(*control_1), QPointF(*control_2),
                         QPointF(*end))
        points = [start] + [end for _, _, end in segments]

        def flip(point):
            return QPointF(1.0 - point[0], point[1])

        for index in range(len(segments) - 1, -1, -1):
            control_1, control_2, _ = segments[index]
            path.cubicTo(flip(control_2), flip(control_1),
                         flip(points[index]))
        path.closeSubpath()
        return self._place.map(path)


# --------------------------------------------------------------------- glyphs

#: Tux's left silhouette, walked from the crown down to the bottom centre.
TUX_OUTLINE = (
    ((0.392, 0.020), (0.316, 0.098), (0.316, 0.202)),   # left of the head
    ((0.316, 0.268), (0.352, 0.306), (0.328, 0.352)),   # neck into the shoulder
    ((0.296, 0.402), (0.186, 0.430), (0.156, 0.556)),   # flipper, outer edge
    ((0.132, 0.672), (0.186, 0.788), (0.250, 0.846)),   # flipper tip to the ankle
    ((0.166, 0.882), (0.096, 0.952), (0.226, 0.974)),   # foot, out to the toe
    ((0.336, 0.992), (0.436, 0.958), (0.462, 0.910)),   # sole back to the ankle
    ((0.478, 0.888), (0.489, 0.904), (0.500, 0.904)),   # in to the bottom centre
)


def tux(box: Box) -> list[tuple[QPainterPath, bool]]:
    """Tux, in outline. ``(path, filled)`` pairs."""
    return [
        (box.mirrored_outline((0.500, 0.020), TUX_OUTLINE), False),
        (box.ellipse(0.500, 0.646, 0.156, 0.212), False),        # belly
        (box.ellipse(0.500, 0.258, 0.086, 0.048), False),        # beak
        (box.circle(0.428, 0.172, 0.030), True),                 # eyes
        (box.circle(0.572, 0.172, 0.030), True),
    ]


def apple(box: Box) -> list[tuple[QPainterPath, bool]]:
    """A bitten apple with a leaf, in outline."""
    body = box.ellipse(0.360, 0.640, 0.300, 0.330).united(
        box.ellipse(0.640, 0.640, 0.300, 0.330)).simplified()
    body = body.subtracted(box.ellipse(0.500, 0.262, 0.155, 0.120))
    body = body.subtracted(box.circle(0.930, 0.540, 0.150)).simplified()
    leaf = box.lens((0.508, 0.318), (0.742, 0.086), 0.210)
    return [(body, False), (leaf, False)]


def windows(box: Box) -> list[tuple[QPainterPath, bool]]:
    """The four-pane window mark, in outline."""
    gap = 0.115
    side = (1.0 - gap) / 2.0
    radius = 0.048
    return [
        (box.round_rect(x, y, side, side, radius), False)
        for x in (0.0, side + gap)
        for y in (0.0, side + gap)
    ]


#: ``name -> (glyph builder, glyph box as a fraction of the canvas)``
ICONS = {
    "linux": (tux, 0.630),
    "macos": (apple, 0.750),
    "windows": (windows, 0.540),
}


# ---------------------------------------------------------------------- paint

def ink(name: str) -> QPainterPath:
    """The whole glyph flattened to one filled path, in canvas pixels."""
    builder, scale = ICONS[name]
    box = Box(scale)
    stroker = QPainterPathStroker()
    stroker.setWidth(STROKE * box.size)
    stroker.setCapStyle(Qt.RoundCap)
    stroker.setJoinStyle(Qt.RoundJoin)

    painted = QPainterPath()
    for path, filled in builder(box):
        painted = painted.united(
            path if filled else stroker.createStroke(path))
    return painted.simplified()


def render(name: str) -> QImage:
    image = QImage(CANVAS, CANVAS, QImage.Format_ARGB32_Premultiplied)
    image.fill(QColor(0, 0, 0, 0))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    chip = QPainterPath()
    chip.addRoundedRect(
        QRectF(0, 0, CANVAS, CANVAS), CHIP_RADIUS, CHIP_RADIUS)
    painter.fillPath(chip, CHIP)
    painter.fillPath(ink(name), WHITE)
    painter.end()
    return image


def coverage(image: QImage) -> float:
    """Fraction of the canvas painted white rather than chip."""
    rgb = image.convertToFormat(QImage.Format_ARGB32)
    buf = bytes(rgb.constBits())
    # ARGB32 little-endian byte order is B G R A
    white = sum(1 for start in range(0, len(buf), 4)
                if buf[start] > 200 and buf[start + 1] > 200
                and buf[start + 2] > 200 and buf[start + 3] > 200)
    return white / float(rgb.width() * rgb.height())


def main() -> int:
    if QGuiApplication.instance() is None:
        QGuiApplication([])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    off_weight = 0
    for name in sorted(ICONS):
        image = render(name)
        target = OUT_DIR / f"{name}.png"
        if not image.save(str(target), "PNG"):
            raise RuntimeError(f"could not write {target}")
        seen = coverage(image)
        flag = "" if COVERAGE_LO <= seen <= COVERAGE_HI else "  <-- off-weight"
        off_weight |= bool(flag)
        print(f"{target.relative_to(ROOT)}  coverage {seen:6.3f}{flag}")
    print(f"band {COVERAGE_LO:.3f}..{COVERAGE_HI:.3f}")
    return int(off_weight)


if __name__ == "__main__":  # pragma: no cover - manual artwork regeneration
    raise SystemExit(main())
