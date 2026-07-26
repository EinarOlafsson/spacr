#!/usr/bin/env python3
"""Candidate icon generator for spaCR -- group: run / sequencing / settings /
train_cellpose / umap.

House style (measured from resources/icons/plaque.png and measure.png):
  * pure white artwork on a fully transparent background (alpha carries the shape)
  * flat: no gradients, no shading, no colour
  * mix of thin outlined strokes and solid white fills
  * square canvas, subject fills most of the frame, modest margin
  * stroke width ~2% of the canvas

Everything is drawn in a normalised 0..1 coordinate space and scaled to the
canvas, so any output size renders correctly.

Run standalone:
    QT_QPA_PLATFORM=offscreen python3 group_run_seq_settings_train_umap.py [outdir]

Deterministic: every random value is drawn from a locally seeded random.Random.
No new dependencies (PySide6 + Pillow are already required by spacr).
"""

from __future__ import annotations

import math
import os
import random
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt, QRectF, QPointF  # noqa: E402
from PySide6.QtGui import (  # noqa: E402
    QBrush,
    QColor,
    QFont,
    QGuiApplication,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QTransform,
)

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

SIZE = 1024
WHITE = QColor(255, 255, 255, 255)

SW = 0.022          # main stroke weight (fraction of canvas)
SW_THIN = 0.015     # secondary / detail strokes
SW_THICK = 0.030    # emphasis strokes
FAM = "DejaVu Sans"

TAU = math.pi * 2.0


# ---------------------------------------------------------------------------
# painter helpers (all coordinates normalised 0..1, y down)
# ---------------------------------------------------------------------------

def stroke(q: QPainter, w: float = SW, dash=None) -> None:
    """Switch the painter to outline mode with a white round-capped pen."""
    p = QPen(WHITE)
    p.setWidthF(w)
    p.setCapStyle(Qt.RoundCap)
    p.setJoinStyle(Qt.RoundJoin)
    if dash:
        p.setStyle(Qt.CustomDashLine)
        p.setDashPattern([d / w for d in dash])
        p.setCapStyle(Qt.FlatCap)
    q.setPen(p)
    q.setBrush(Qt.NoBrush)


def solid(q: QPainter) -> None:
    """Switch the painter to fill mode (no outline)."""
    q.setPen(Qt.NoPen)
    q.setBrush(QBrush(WHITE))


def solid_soft(q: QPainter, w: float = SW * 0.6) -> None:
    """Fill *and* stroke, which rounds the corners of polygonal solids."""
    p = QPen(WHITE)
    p.setWidthF(w)
    p.setCapStyle(Qt.RoundCap)
    p.setJoinStyle(Qt.RoundJoin)
    q.setPen(p)
    q.setBrush(QBrush(WHITE))


def dot(q: QPainter, x: float, y: float, r: float) -> None:
    solid(q)
    q.drawEllipse(QPointF(x, y), r, r)


def ring(q: QPainter, x: float, y: float, r: float, w: float = SW) -> None:
    stroke(q, w)
    q.drawEllipse(QPointF(x, y), r, r)


def line(q: QPainter, x1: float, y1: float, x2: float, y2: float,
         w: float = SW, dash=None) -> None:
    stroke(q, w, dash)
    q.drawLine(QPointF(x1, y1), QPointF(x2, y2))


def poly(q: QPainter, pts, w: float = SW, close: bool = False, fill: bool = False,
         dash=None) -> None:
    path = QPainterPath()
    path.moveTo(*pts[0])
    for p in pts[1:]:
        path.lineTo(*p)
    if close:
        path.closeSubpath()
    if fill:
        solid_soft(q, w)
    else:
        stroke(q, w, dash)
    q.drawPath(path)


def rrect(q: QPainter, x: float, y: float, w: float, h: float, r: float,
         sw: float = SW, fill: bool = False) -> None:
    if fill:
        solid_soft(q, sw * 0.4)
    else:
        stroke(q, sw)
    q.drawRoundedRect(QRectF(x, y, w, h), r, r)


def arrow(q: QPainter, x1: float, y1: float, x2: float, y2: float,
          w: float = SW, head: float = 0.055) -> None:
    """Straight arrow from (x1,y1) to (x2,y2) with an open V head."""
    stroke(q, w)
    a = math.atan2(y2 - y1, x2 - x1)
    q.drawLine(QPointF(x1, y1), QPointF(x2, y2))
    for s in (+1, -1):
        b = a + s * 2.55
        q.drawLine(QPointF(x2, y2),
                   QPointF(x2 + head * math.cos(b), y2 + head * math.sin(b)))


def chevron(q: QPainter, cx: float, cy: float, w: float, h: float,
            sw: float = SW_THICK) -> None:
    poly(q, [(cx - w / 2, cy - h / 2), (cx + w / 2, cy), (cx - w / 2, cy + h / 2)], sw)


def closed_smooth(pts) -> QPainterPath:
    """Closed Catmull-Rom spline through pts, as cubic beziers."""
    path = QPainterPath()
    n = len(pts)
    path.moveTo(*pts[0])
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6.0, p1[1] + (p2[1] - p0[1]) / 6.0)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6.0, p2[1] - (p3[1] - p1[1]) / 6.0)
        path.cubicTo(c1[0], c1[1], c2[0], c2[1], p2[0], p2[1])
    path.closeSubpath()
    return path


def blob_pts(cx: float, cy: float, r: float, seed: int = 0, n: int = 9,
             wobble: float = 0.14, sq: float = 1.0):
    rnd = random.Random(seed)
    pts = []
    for i in range(n):
        a = TAU * i / n
        rr = r * (1.0 + rnd.uniform(-wobble, wobble))
        pts.append((cx + rr * math.cos(a), cy + rr * sq * math.sin(a)))
    return pts


def blob(cx: float, cy: float, r: float, seed: int = 0, n: int = 9,
         wobble: float = 0.14, sq: float = 1.0) -> QPainterPath:
    return closed_smooth(blob_pts(cx, cy, r, seed, n, wobble, sq))


def draw_blob(q: QPainter, cx: float, cy: float, r: float, seed: int = 0,
              w: float = SW, fill: bool = False, n: int = 9,
              wobble: float = 0.14, sq: float = 1.0, dash=None) -> None:
    path = blob(cx, cy, r, seed, n, wobble, sq)
    if fill:
        solid(q)
    else:
        stroke(q, w, dash)
    q.drawPath(path)


def draw_cell(q: QPainter, cx: float, cy: float, r: float, seed: int = 0,
              w: float = SW, nuc: float = 0.36, organelles: int = 0) -> None:
    """House-style cell: wobbly outline + solid nucleus (+ optional specks)."""
    draw_blob(q, cx, cy, r, seed=seed, w=w)
    if nuc > 0:
        dot(q, cx, cy, r * nuc)
    if organelles:
        rnd = random.Random(seed + 991)
        for _ in range(organelles):
            a = rnd.uniform(0, TAU)
            rr = r * rnd.uniform(0.55, 0.78)
            dot(q, cx + rr * math.cos(a), cy + rr * math.sin(a), r * 0.09)


def play_path(cx: float, cy: float, r: float) -> QPainterPath:
    """Equilateral play triangle pointing right, 'radius' r."""
    p = QPainterPath()
    pts = [(cx + r, cy),
           (cx - r * 0.62, cy - r * 0.90),
           (cx - r * 0.62, cy + r * 0.90)]
    p.moveTo(*pts[0])
    p.lineTo(*pts[1])
    p.lineTo(*pts[2])
    p.closeSubpath()
    return p


def draw_play(q: QPainter, cx: float, cy: float, r: float, fill: bool = True,
              w: float = SW) -> None:
    if fill:
        solid_soft(q, w * 0.7)
    else:
        stroke(q, w)
    q.drawPath(play_path(cx, cy, r))


def gear_path(cx: float, cy: float, r: float, teeth: int = 8,
              depth: float = 0.26, span: float = 0.30,
              phase: float = 0.0) -> QPainterPath:
    step = TAU / teeth
    ri = r * (1.0 - depth)
    w = step * span
    v = step * 0.5 - w * 0.55
    pts = []
    for i in range(teeth):
        a = i * step + phase
        for rr, aa in ((ri, a - v), (r, a - w), (r, a + w), (ri, a + v)):
            pts.append((cx + rr * math.cos(aa), cy + rr * math.sin(aa)))
    p = QPainterPath()
    p.moveTo(*pts[0])
    for pt in pts[1:]:
        p.lineTo(*pt)
    p.closeSubpath()
    return p


def wrench_path(length: float = 0.74, head_r: float = 0.150) -> QPainterPath:
    """Open-end wrench, drawn around the origin, head up, handle down."""
    R = head_r
    r = R * 0.58
    hy = -length * 0.5 + R * 0.60
    head = QPainterPath()
    head.addEllipse(QPointF(0.0, hy), R, R)
    hole = QPainterPath()
    hole.addEllipse(QPointF(0.0, hy), r, r)
    jaw = QPainterPath()
    jw = r
    jaw.addRect(QRectF(-jw, hy - R * 1.8, 2 * jw, R * 1.8))
    handle = QPainterPath()
    hw = R * 0.44
    handle.addRoundedRect(QRectF(-hw, hy, 2 * hw, length * 0.5 - hy + R * 0.0), hw, hw)
    return head.subtracted(hole).united(handle).subtracted(jaw).simplified()


def screwdriver_path(length: float = 0.78) -> QPainterPath:
    """Screwdriver, handle up, tip down, drawn around the origin."""
    top = -length * 0.5
    p = QPainterPath()
    p.addRoundedRect(QRectF(-0.085, top, 0.17, length * 0.34), 0.055, 0.055)
    shaft = QPainterPath()
    shaft.addRect(QRectF(-0.034, top + length * 0.30, 0.068, length * 0.45))
    tip = QPainterPath()
    tip.moveTo(-0.034, top + length * 0.72)
    tip.lineTo(-0.021, top + length)
    tip.lineTo(0.021, top + length)
    tip.lineTo(0.034, top + length * 0.72)
    tip.closeSubpath()
    return p.united(shaft).united(tip).simplified()


def helix(q: QPainter, x0: float, x1: float, cy: float, amp: float,
          turns: float = 2.0, rungs: int = 7, w: float = SW) -> None:
    """Horizontal DNA double helix between x0 and x1."""
    stroke(q, w)
    for phase in (0.0, math.pi):
        path = QPainterPath()
        for i in range(81):
            t = i / 80.0
            x = x0 + (x1 - x0) * t
            y = cy + amp * math.sin(TAU * turns * t + phase)
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        q.drawPath(path)
    stroke(q, w * 0.8)
    for j in range(rungs):
        t = (j + 0.5) / rungs
        x = x0 + (x1 - x0) * t
        y1 = cy + amp * math.sin(TAU * turns * t)
        y2 = cy + amp * math.sin(TAU * turns * t + math.pi)
        q.drawLine(QPointF(x, y1), QPointF(x, y2))


_FONT_CACHE = {}


def text_path(s: str, px: int = 400, bold: bool = True) -> QPainterPath:
    key = (px, bold)
    f = _FONT_CACHE.get(key)
    if f is None:
        f = QFont(FAM)
        f.setPixelSize(px)
        f.setBold(bold)
        _FONT_CACHE[key] = f
    p = QPainterPath()
    p.addText(0.0, 0.0, f, s)
    return p


def draw_text(q: QPainter, s: str, cx: float, cy: float, h: float,
              bold: bool = True) -> None:
    """Fill text centred on (cx, cy) with glyph height h (normalised)."""
    p = text_path(s, 400, bold)
    br = p.boundingRect()
    if br.height() <= 0:
        return
    sc = h / br.height()
    t = QTransform()
    t.translate(cx, cy)
    t.scale(sc, sc)
    t.translate(-br.center().x(), -br.center().y())
    q.fillPath(t.map(p), QBrush(WHITE))


def arc(q: QPainter, cx: float, cy: float, r: float, a0: float, a1: float,
        w: float = SW) -> None:
    """Arc, angles in degrees CCW from east (Qt convention)."""
    stroke(q, w)
    q.drawArc(QRectF(cx - r, cy - r, 2 * r, 2 * r),
              int(round(a0 * 16)), int(round((a1 - a0) * 16)))


def arc_arrow_head(q: QPainter, cx: float, cy: float, r: float, adeg: float,
                   ccw: bool = True, size: float = 0.075, w: float = SW) -> None:
    """Arrow head tangent to a circle at angle adeg (degrees, Qt sense)."""
    a = math.radians(adeg)
    px, py = cx + r * math.cos(a), cy - r * math.sin(a)
    tang = a + (math.pi / 2 if ccw else -math.pi / 2)
    tx, ty = math.cos(tang), -math.sin(tang)
    base = math.atan2(ty, tx)
    stroke(q, w)
    for s in (+1, -1):
        b = base + math.pi + s * 0.55
        q.drawLine(QPointF(px, py),
                   QPointF(px + size * math.cos(b), py + size * math.sin(b)))


# ---------------------------------------------------------------------------
# RUN -- start the pipeline
# ---------------------------------------------------------------------------

def run_01(q):
    """Play triangle built out of cells."""
    poly(q, [(0.20, 0.09), (0.20, 0.91), (0.90, 0.50)], SW_THIN, close=True)
    cells = [(0.33, 0.50, 0.085), (0.47, 0.31, 0.062), (0.47, 0.69, 0.062),
             (0.50, 0.50, 0.070), (0.63, 0.42, 0.052), (0.63, 0.58, 0.052),
             (0.35, 0.24, 0.040), (0.35, 0.76, 0.040), (0.72, 0.50, 0.045)]
    for cx, cy, r in cells:
        dot(q, cx, cy, r)


def run_02(q):
    """Petri dish with an inscribed play triangle."""
    ring(q, 0.5, 0.5, 0.44, SW)
    ring(q, 0.5, 0.5, 0.385, SW_THIN)
    draw_play(q, 0.52, 0.5, 0.235)
    for a, rr, s in ((0.55, 0.32, 0.040), (2.45, 0.31, 0.034),
                     (3.95, 0.33, 0.030), (5.10, 0.30, 0.036)):
        dot(q, 0.5 + rr * math.cos(a), 0.5 + rr * math.sin(a), s)


def run_03(q):
    """Power / start button: broken ring with a play triangle."""
    arc(q, 0.5, 0.5, 0.42, 118, 422, SW_THICK)
    draw_play(q, 0.52, 0.5, 0.215)


def run_04(q):
    """Stopwatch: start a timed run."""
    cy = 0.585
    ring(q, 0.5, cy, 0.355, SW)
    rrect(q, 0.435, 0.075, 0.13, 0.075, 0.03, SW_THIN)
    line(q, 0.5, 0.15, 0.5, 0.228, SW)
    line(q, 0.75, 0.20, 0.815, 0.265, SW)
    stroke(q, SW_THIN)
    for k in range(12):
        a = TAU * k / 12
        r0, r1 = (0.30, 0.335) if k % 3 == 0 else (0.315, 0.335)
        q.drawLine(QPointF(0.5 + r0 * math.cos(a), cy + r0 * math.sin(a)),
                   QPointF(0.5 + r1 * math.cos(a), cy + r1 * math.sin(a)))
    draw_play(q, 0.515, cy, 0.165)


def run_05(q):
    """Three-stage pipeline: dish -> image tile -> plot, then go."""
    cy = 0.5
    ring(q, 0.145, cy, 0.135, SW)
    dot(q, 0.115, 0.455, 0.038)
    dot(q, 0.185, 0.545, 0.030)
    arrow(q, 0.305, cy, 0.375, cy, SW_THIN, 0.042)
    x0, s = 0.405, 0.27
    rrect(q, x0, cy - s / 2, s, s, 0.045, SW)
    stroke(q, SW_THIN)
    for k in (1, 2):
        q.drawLine(QPointF(x0 + s * k / 3, cy - s / 2), QPointF(x0 + s * k / 3, cy + s / 2))
        q.drawLine(QPointF(x0, cy - s / 2 + s * k / 3), QPointF(x0 + s, cy - s / 2 + s * k / 3))
    arrow(q, 0.705, cy, 0.775, cy, SW_THIN, 0.042)
    draw_play(q, 0.865, cy, 0.125)


def run_06(q):
    """Fast-forward chevrons consuming a row of samples."""
    for i, cx in enumerate((0.28, 0.50, 0.72)):
        chevron(q, cx, 0.42, 0.20, 0.44, SW_THICK)
    for i in range(5):
        x = 0.16 + i * 0.17
        dot(q, x, 0.845, 0.045 if i < 2 else 0.030)
    line(q, 0.11, 0.925, 0.89, 0.925, SW_THIN)


def run_07(q):
    """Microscope objective over a slide: start acquisition."""
    poly(q, [(0.245, 0.06), (0.665, 0.06), (0.665, 0.20), (0.575, 0.30),
             (0.575, 0.42), (0.335, 0.42), (0.335, 0.30), (0.245, 0.20)],
         SW, close=True)
    line(q, 0.245, 0.155, 0.665, 0.155, SW_THIN)
    line(q, 0.335, 0.325, 0.575, 0.325, SW_THIN)
    line(q, 0.435, 0.47, 0.335, 0.65, SW_THIN, dash=(0.045, 0.038))
    line(q, 0.475, 0.47, 0.575, 0.65, SW_THIN, dash=(0.045, 0.038))
    rrect(q, 0.055, 0.665, 0.70, 0.185, 0.035, SW)
    line(q, 0.165, 0.665, 0.165, 0.85, SW_THIN)
    line(q, 0.645, 0.665, 0.645, 0.85, SW_THIN)
    for x, y, r in ((0.345, 0.735, 0.032), (0.435, 0.785, 0.026), (0.525, 0.730, 0.022)):
        dot(q, x, y, r)
    draw_play(q, 0.885, 0.7575, 0.105)


def run_08(q):
    """Funnel: a stack of images in, one measured object out."""
    for i, (x, y) in enumerate(((0.185, 0.10), (0.395, 0.10), (0.605, 0.10))):
        rrect(q, x, y, 0.20, 0.155, 0.028, SW_THIN)
        dot(q, x + 0.10, y + 0.078, 0.030)
    poly(q, [(0.13, 0.33), (0.87, 0.33), (0.575, 0.60), (0.575, 0.80),
             (0.425, 0.80), (0.425, 0.60)], SW, close=True)
    draw_cell(q, 0.5, 0.905, 0.072, seed=7, w=SW_THIN, nuc=0.40)


def run_09(q):
    """Stepper: a run advancing through numbered stages."""
    y = 0.50
    xs = (0.325, 0.510, 0.695, 0.880)
    stroke(q, SW_THIN)
    for i in range(3):
        q.drawLine(QPointF(xs[i] + 0.078, y), QPointF(xs[i + 1] - 0.078, y))
    for i, x in enumerate(xs):
        if i < 2:
            dot(q, x, y, 0.062)
        else:
            ring(q, x, y, 0.058, SW)
    draw_play(q, 0.085, y, 0.100)
    stroke(q, SW_THIN)
    for x in xs:
        q.drawLine(QPointF(x, 0.665), QPointF(x, 0.745))


def run_10(q):
    """Terminal window: execute the pipeline command."""
    rrect(q, 0.08, 0.17, 0.84, 0.66, 0.06, SW)
    line(q, 0.08, 0.325, 0.92, 0.325, SW_THIN)
    dot(q, 0.155, 0.247, 0.028)
    dot(q, 0.243, 0.247, 0.028)
    dot(q, 0.331, 0.247, 0.028)
    poly(q, [(0.19, 0.45), (0.315, 0.56), (0.19, 0.67)], SW_THICK)
    rrect(q, 0.385, 0.625, 0.30, 0.048, 0.024, SW_THIN, fill=True)


# ---------------------------------------------------------------------------
# SEQUENCING -- reading barcodes off FASTQ
# ---------------------------------------------------------------------------

def seq_01(q):
    """Barcode over a DNA helix."""
    widths = [0.040, 0.020, 0.055, 0.020, 0.032, 0.062, 0.020, 0.044]
    gap = (0.80 - sum(widths)) / (len(widths) - 1)
    x = 0.10
    solid(q)
    for w in widths:
        q.drawRect(QRectF(x, 0.09, w, 0.33))
        x += w + gap
    helix(q, 0.09, 0.91, 0.735, 0.135, turns=1.5, rungs=5, w=SW)


def seq_02(q):
    """Flow-cell lanes with sequencing clusters."""
    rrect(q, 0.09, 0.13, 0.82, 0.74, 0.055, SW)
    stroke(q, SW_THIN)
    for i in range(1, 4):
        x = 0.09 + 0.82 * i / 4.0
        q.drawLine(QPointF(x, 0.13), QPointF(x, 0.87))
    rnd = random.Random(11)
    for lane in range(4):
        x0 = 0.09 + 0.82 * lane / 4.0
        for k in range(4):
            cx = x0 + 0.205 * rnd.uniform(0.28, 0.72)
            cy = 0.185 + rnd.uniform(0.0, 0.62)
            dot(q, cx, cy, rnd.uniform(0.024, 0.040))


def seq_03(q):
    """Nanopore: a strand threading a membrane."""
    def sx(t):
        y = 0.05 + 0.90 * t
        return 0.50 + 0.155 * math.sin(TAU * 1.35 * t) * (1.0 - math.exp(-((y - 0.5) ** 2) / 0.010)), y

    # the pore itself: an hourglass barrel spanning the membrane
    pore = QPainterPath()
    pore.moveTo(0.345, 0.275)
    pore.cubicTo(0.345, 0.445, 0.435, 0.445, 0.435, 0.50)
    pore.cubicTo(0.435, 0.555, 0.345, 0.555, 0.345, 0.725)
    pore.lineTo(0.655, 0.725)
    pore.cubicTo(0.655, 0.555, 0.565, 0.555, 0.565, 0.50)
    pore.cubicTo(0.565, 0.445, 0.655, 0.445, 0.655, 0.275)
    pore.closeSubpath()
    # membrane band with the pore punched out of it
    mem = QPainterPath()
    mem.addRoundedRect(QRectF(0.025, 0.375, 0.95, 0.25), 0.062, 0.062)
    grow = QTransform()
    grow.translate(0.5, 0.5)
    grow.scale(1.10, 1.10)
    grow.translate(-0.5, -0.5)
    solid(q)
    q.drawPath(mem.subtracted(grow.map(pore)).simplified())
    stroke(q, SW)
    q.drawPath(pore)
    path = QPainterPath()
    for i in range(81):
        x, y = sx(i / 80.0)
        if i == 0:
            path.moveTo(x, y)
        else:
            path.lineTo(x, y)
    stroke(q, SW)
    q.drawPath(path)
    for t in (0.02, 0.16, 0.84, 0.98):
        x, y = sx(t)
        dot(q, x, y, 0.032)


def seq_04(q):
    """Sanger chromatogram: four base traces on a baseline."""
    line(q, 0.06, 0.80, 0.94, 0.80, SW_THIN)
    peaks = [(0.155, 0.30), (0.30, 0.16), (0.44, 0.42), (0.585, 0.22),
             (0.725, 0.36), (0.865, 0.26)]
    stroke(q, SW)
    for cx, top in peaks:
        path = QPainterPath()
        w = 0.075
        path.moveTo(cx - w, 0.80)
        path.cubicTo(cx - w * 0.45, 0.80, cx - w * 0.42, top, cx, top)
        path.cubicTo(cx + w * 0.42, top, cx + w * 0.45, 0.80, cx + w, 0.80)
        q.drawPath(path)
    stroke(q, SW_THIN)
    for cx, _ in peaks:
        q.drawLine(QPointF(cx, 0.845), QPointF(cx, 0.90))


def seq_05(q):
    """Read pile-up aligned to a reference."""
    solid_soft(q, SW * 0.4)
    q.drawRoundedRect(QRectF(0.06, 0.465, 0.88, 0.070), 0.035, 0.035)
    reads = [(0.10, 0.34, 0.285), (0.42, 0.34, 0.30), (0.70, 0.24, 0.315),
             (0.14, 0.215, 0.24), (0.46, 0.215, 0.26),
             (0.08, 0.615, 0.30), (0.44, 0.615, 0.27), (0.75, 0.615, 0.19),
             (0.20, 0.735, 0.33), (0.60, 0.735, 0.24)]
    stroke(q, SW_THIN)
    for x, y, w in reads:
        q.drawRoundedRect(QRectF(x, y, w, 0.075), 0.030, 0.030)


def seq_06(q):
    """The four bases as code tiles."""
    letters = (("A", 0.09, 0.09), ("T", 0.535, 0.09),
               ("G", 0.09, 0.535), ("C", 0.535, 0.09 + 0.445))
    for ch, x, y in letters:
        rrect(q, x, y, 0.375, 0.375, 0.055, SW)
        draw_text(q, ch, x + 0.1875, y + 0.1875, 0.195)


def seq_07(q):
    """Barcode tag attached to a cell: barcode-to-cell mapping."""
    draw_cell(q, 0.315, 0.665, 0.275, seed=5, w=SW, nuc=0.32, organelles=3)
    line(q, 0.505, 0.475, 0.585, 0.395, SW_THIN)
    q.save()
    q.translate(0.715, 0.265)
    q.rotate(-30.0)
    stroke(q, SW)
    q.drawRoundedRect(QRectF(-0.235, -0.135, 0.47, 0.27), 0.045, 0.045)
    ring(q, -0.165, 0.0, 0.038, SW_THIN)
    solid(q)
    for i, w in enumerate((0.030, 0.018, 0.042, 0.020, 0.032)):
        x = -0.085 + i * 0.062
        q.drawRect(QRectF(x, -0.085, w, 0.17))
    q.restore()


def seq_08(q):
    """Magnifier reading a barcode."""
    lcx, lcy, lr = 0.605, 0.395, 0.275
    lens = QPainterPath()
    lens.addEllipse(QPointF(lcx, lcy), lr - SW_THICK * 0.5, lr - SW_THICK * 0.5)
    outside = QPainterPath()
    outside.addRect(QRectF(0, 0, 1, 1))
    outside = outside.subtracted(lens)

    q.save()
    q.setClipPath(outside)
    solid(q)
    x = 0.055
    for w in (0.026, 0.046, 0.018, 0.034, 0.018, 0.042, 0.024, 0.038):
        q.drawRect(QRectF(x, 0.185, w, 0.42))
        x += w + 0.040
    q.restore()

    q.save()
    q.setClipPath(lens)
    solid(q)
    x = 0.395
    for w in (0.048, 0.082, 0.034, 0.062):
        q.drawRect(QRectF(x, 0.115, w, 0.56))
        x += w + 0.062
    q.restore()

    ring(q, lcx, lcy, lr, SW_THICK)
    line(q, lcx + lr * 0.72, lcy + lr * 0.72, 0.935, 0.735, SW_THICK)


def seq_09(q):
    """A FASTQ record: header, sequence, plus line, quality."""
    stroke(q, SW)
    path = QPainterPath()
    path.moveTo(0.16, 0.06)
    path.lineTo(0.66, 0.06)
    path.lineTo(0.84, 0.24)
    path.lineTo(0.84, 0.94)
    path.lineTo(0.16, 0.94)
    path.closeSubpath()
    q.drawPath(path)
    poly(q, [(0.66, 0.06), (0.66, 0.24), (0.84, 0.24)], SW_THIN)
    draw_text(q, "@", 0.265, 0.375, 0.115)
    line(q, 0.345, 0.375, 0.735, 0.375, SW_THIN)
    solid(q)
    x = 0.245
    for w in (0.055, 0.030, 0.070, 0.038, 0.055, 0.030):
        q.drawRoundedRect(QRectF(x, 0.505, w, 0.052), 0.026, 0.026)
        x += w + 0.026
    draw_text(q, "+", 0.265, 0.665, 0.095)
    line(q, 0.345, 0.665, 0.615, 0.665, SW_THIN)
    solid(q)
    for i, h in enumerate((0.055, 0.085, 0.115, 0.100, 0.070, 0.120, 0.090, 0.045)):
        x = 0.245 + i * 0.062
        q.drawRoundedRect(QRectF(x, 0.855 - h, 0.036, h), 0.018, 0.018)


def seq_10(q):
    """Electrophoresis gel: lanes of bands."""
    rrect(q, 0.075, 0.10, 0.85, 0.80, 0.05, SW)
    for i in range(4):
        x = 0.135 + i * 0.205
        rrect(q, x, 0.155, 0.145, 0.055, 0.022, SW_THIN)
    bands = {0: (0.335, 0.545, 0.745), 1: (0.295, 0.615), 2: (0.365, 0.495, 0.675, 0.795),
             3: (0.445, 0.705)}
    solid(q)
    for i, ys in bands.items():
        x = 0.135 + i * 0.205
        for j, y in enumerate(ys):
            h = 0.052 if j % 2 == 0 else 0.040
            q.drawRoundedRect(QRectF(x, y, 0.145, h), 0.019, 0.019)


# ---------------------------------------------------------------------------
# SETTINGS -- configuration
# ---------------------------------------------------------------------------

def set_01(q):
    """Sliders."""
    for i, (y, t) in enumerate(((0.235, 0.68), (0.50, 0.34), (0.765, 0.56))):
        line(q, 0.10, y, 0.90, y, SW)
        x = 0.10 + 0.80 * t
        dot(q, x, y, 0.085)
        stroke(q, SW_THIN)
        q.drawLine(QPointF(0.10, y - 0.055), QPointF(0.10, y + 0.055))
        q.drawLine(QPointF(0.90, y - 0.055), QPointF(0.90, y + 0.055))


def set_02(q):
    """Toggle switches."""
    for i, (y, on) in enumerate(((0.20, True), (0.50, False), (0.80, True))):
        rrect(q, 0.16, y - 0.115, 0.68, 0.23, 0.115, SW)
        cx = 0.72 if on else 0.28
        if on:
            dot(q, cx, y, 0.077)
        else:
            ring(q, cx, y, 0.072, SW)


def set_03(q):
    """Rotary dials with tick arcs."""
    for cx, cy, r, ang in ((0.305, 0.305, 0.205, 145.0), (0.705, 0.705, 0.205, 50.0)):
        ring(q, cx, cy, r, SW)
        a = math.radians(ang)
        line(q, cx, cy, cx + r * 0.70 * math.cos(a), cy - r * 0.70 * math.sin(a), SW)
        stroke(q, SW_THIN)
        for k in range(7):
            t = math.radians(215 - k * 41.7)
            q.drawLine(QPointF(cx + (r + 0.030) * math.cos(t), cy - (r + 0.030) * math.sin(t)),
                       QPointF(cx + (r + 0.062) * math.cos(t), cy - (r + 0.062) * math.sin(t)))


def set_04(q):
    """A wrench adjusting a cell."""
    draw_cell(q, 0.735, 0.50, 0.225, seed=4, w=SW, nuc=0.32, organelles=3)
    q.save()
    q.translate(0.238, 0.50)
    q.rotate(90.0)
    stroke(q, SW)
    q.drawPath(wrench_path(0.44, 0.155))
    q.restore()


def set_05(q):
    """A cell whose nucleus is a gear."""
    draw_blob(q, 0.5, 0.5, 0.415, seed=12, w=SW, n=11, wobble=0.10)
    solid_soft(q, SW * 0.5)
    q.drawPath(gear_path(0.5, 0.5, 0.215, teeth=8, depth=0.30, span=0.30))
    q.setCompositionMode(QPainter.CompositionMode_Clear)
    q.setBrush(QBrush(QColor(0, 0, 0, 255)))
    q.setPen(Qt.NoPen)
    q.drawEllipse(QPointF(0.5, 0.5), 0.072, 0.072)
    q.setCompositionMode(QPainter.CompositionMode_SourceOver)
    for a, rr, s in ((0.75, 0.32, 0.042), (2.30, 0.31, 0.034), (3.85, 0.32, 0.030),
                     (5.35, 0.30, 0.038)):
        dot(q, 0.5 + rr * math.cos(a), 0.5 + rr * math.sin(a), s)


def set_06(q):
    """Equalizer / fader bank."""
    for i, t in enumerate((0.30, 0.62, 0.44, 0.78, 0.52)):
        x = 0.155 + i * 0.1725
        line(q, x, 0.10, x, 0.90, SW_THIN)
        y = 0.90 - 0.80 * t
        solid_soft(q, SW * 0.4)
        q.drawRoundedRect(QRectF(x - 0.062, y - 0.038, 0.124, 0.076), 0.030, 0.030)


def set_07(q):
    """A configuration checklist."""
    rrect(q, 0.075, 0.10, 0.85, 0.80, 0.06, SW)
    for i, checked in enumerate((True, True, False)):
        y = 0.275 + i * 0.225
        rrect(q, 0.155, y - 0.075, 0.15, 0.15, 0.035, SW_THIN)
        if checked:
            poly(q, [(0.187, y + 0.005), (0.222, y + 0.045), (0.283, y - 0.045)], SW)
        line(q, 0.375, y, 0.375 + (0.44 if i != 1 else 0.32), y, SW_THIN)


def set_08(q):
    """Histogram with an adjustable threshold."""
    line(q, 0.085, 0.865, 0.94, 0.865, SW_THIN)
    solid(q)
    hs = (0.14, 0.29, 0.46, 0.63, 0.50, 0.34, 0.22, 0.12)
    for i, h in enumerate(hs):
        x = 0.11 + i * 0.104
        q.drawRect(QRectF(x, 0.865 - h, 0.078, h))
    line(q, 0.635, 0.06, 0.635, 0.94, SW, dash=(0.055, 0.040))
    solid_soft(q, SW * 0.4)
    q.drawRoundedRect(QRectF(0.575, 0.055, 0.12, 0.085), 0.035, 0.035)


def set_09(q):
    """Crossed wrench and screwdriver."""
    ta = QTransform()
    ta.translate(0.42, 0.50)
    ta.rotate(-32.0)
    tb = QTransform()
    tb.translate(0.58, 0.50)
    tb.rotate(34.0)
    shape = ta.map(wrench_path(0.86, 0.150)).united(tb.map(screwdriver_path(0.88)))
    stroke(q, SW)
    q.drawPath(shape.simplified())


def set_10(q):
    """Meshing gear train: coupled parameters."""
    ra, rb, depth, span = 0.285, 0.205, 0.20, 0.34
    ax, ay = 0.315, 0.345
    theta = math.radians(43.0)
    dist = (ra + rb) * (1.0 - depth * 0.5)
    bx, by = ax + dist * math.cos(theta), ay + dist * math.sin(theta)
    na, nb = 8, 6
    ga = gear_path(ax, ay, ra, teeth=na, depth=depth, span=span, phase=theta)
    gb = gear_path(bx, by, rb, teeth=nb, depth=depth, span=span,
                   phase=theta + math.pi + TAU / (2 * nb))
    outside = QPainterPath()
    outside.addRect(QRectF(-0.1, -0.1, 1.2, 1.2))
    q.save()
    q.setClipPath(outside.subtracted(ga))
    stroke(q, SW)
    q.drawPath(gb)
    q.restore()
    stroke(q, SW)
    q.drawPath(ga)
    ring(q, ax, ay, 0.100, SW_THIN)
    dot(q, bx, by, 0.072)


# ---------------------------------------------------------------------------
# TRAIN_CELLPOSE -- training a segmentation model
# ---------------------------------------------------------------------------

def tr_01(q):
    """Hand-annotating an outline: the pencil makes the training label."""
    draw_blob(q, 0.375, 0.615, 0.275, seed=21, w=SW_THIN)
    dot(q, 0.375, 0.615, 0.088)
    pen = QPen(WHITE, SW, Qt.CustomDashLine, Qt.FlatCap, Qt.RoundJoin)
    pen.setDashPattern([0.055 / SW, 0.045 / SW])
    q.setPen(pen)
    q.setBrush(Qt.NoBrush)
    q.drawPath(blob(0.375, 0.615, 0.325, seed=21, n=9, wobble=0.10))
    q.save()
    q.translate(0.685, 0.315)
    q.rotate(40.0)
    stroke(q, SW)
    q.drawRoundedRect(QRectF(-0.070, -0.315, 0.14, 0.40), 0.028, 0.028)
    q.drawLine(QPointF(-0.070, 0.020), QPointF(0.070, 0.020))
    poly(q, [(-0.070, 0.085), (0.0, 0.265), (0.070, 0.085)], SW, close=True)
    poly(q, [(-0.028, 0.195), (0.0, 0.265), (0.028, 0.195)], SW_THIN, close=True, fill=True)
    q.restore()


def tr_02(q):
    """Image / label pairs: raw tile in, mask tile out."""
    rrect(q, 0.035, 0.185, 0.37, 0.37, 0.055, SW)
    draw_cell(q, 0.22, 0.37, 0.125, seed=31, w=SW_THIN, nuc=0.35)
    arrow(q, 0.445, 0.37, 0.555, 0.37, SW_THIN, 0.055)
    rrect(q, 0.595, 0.185, 0.37, 0.37, 0.055, SW)
    draw_blob(q, 0.78, 0.37, 0.125, seed=31, w=SW, fill=True)
    rrect(q, 0.115, 0.655, 0.245, 0.245, 0.045, SW_THIN)
    draw_cell(q, 0.2375, 0.7775, 0.080, seed=32, w=SW_THIN, nuc=0.35)
    arrow(q, 0.425, 0.7775, 0.535, 0.7775, SW_THIN, 0.048)
    rrect(q, 0.575, 0.655, 0.245, 0.245, 0.045, SW_THIN)
    draw_blob(q, 0.6975, 0.7775, 0.080, seed=32, w=SW, fill=True)
    for i in range(3):
        dot(q, 0.885, 0.700 + i * 0.078, 0.022)


def tr_03(q):
    """Loss curve falling over epochs."""
    poly(q, [(0.12, 0.10), (0.12, 0.86), (0.94, 0.86)], SW_THIN)
    stroke(q, SW)
    path = QPainterPath()
    path.moveTo(0.185, 0.185)
    path.cubicTo(0.36, 0.30, 0.40, 0.66, 0.60, 0.71)
    path.cubicTo(0.74, 0.745, 0.80, 0.755, 0.90, 0.762)
    q.drawPath(path)
    for x, y in ((0.185, 0.185), (0.345, 0.395), (0.525, 0.685), (0.715, 0.748), (0.90, 0.762)):
        dot(q, x, y, 0.040)
    stroke(q, SW_THIN)
    for i in range(4):
        x = 0.245 + i * 0.195
        q.drawLine(QPointF(x, 0.86), QPointF(x, 0.915))


def tr_04(q):
    """Layered network learning cell -> mask."""
    cols = ((0.215, 3), (0.50, 4), (0.785, 3))
    pos = []
    for cx, n in cols:
        ys = [0.5 + (i - (n - 1) / 2.0) * (0.215 if n == 3 else 0.185) for i in range(n)]
        pos.append([(cx, y) for y in ys])
    stroke(q, SW_THIN * 0.75)
    for a, b in ((0, 1), (1, 2)):
        for p in pos[a]:
            for r in pos[b]:
                q.drawLine(QPointF(p[0], p[1]), QPointF(r[0], r[1]))
    for j, col in enumerate(pos):
        for p in col:
            if j == 1:
                dot(q, p[0], p[1], 0.058)
            else:
                ring(q, p[0], p[1], 0.055, SW_THIN)
                solid(q)
                q.drawEllipse(QPointF(p[0], p[1]), 0.055 - SW_THIN, 0.055 - SW_THIN)
    draw_cell(q, 0.068, 0.5, 0.068, seed=41, w=SW_THIN, nuc=0.40)
    draw_blob(q, 0.932, 0.5, 0.068, seed=41, fill=True)


def tr_05(q):
    """Epoch loop: iterate until the mask fits."""
    r = 0.435
    arc(q, 0.5, 0.5, r, 120, 320, SW)
    arc_arrow_head(q, 0.5, 0.5, r, 320, ccw=False, size=0.090, w=SW)
    arc(q, 0.5, 0.5, r, -60, 140, SW)
    arc_arrow_head(q, 0.5, 0.5, r, 140, ccw=False, size=0.090, w=SW)
    draw_blob(q, 0.5, 0.5, 0.185, seed=51, w=SW)
    dot(q, 0.5, 0.5, 0.068)
    pen = QPen(WHITE, SW * 0.85, Qt.CustomDashLine, Qt.FlatCap, Qt.RoundJoin)
    pen.setDashPattern([0.050 / (SW * 0.85), 0.042 / (SW * 0.85)])
    q.setPen(pen)
    q.setBrush(Qt.NoBrush)
    q.drawPath(blob(0.5, 0.5, 0.265, seed=51))


def tr_06(q):
    """Cellpose flow field: vectors converging on the cell centre."""
    draw_blob(q, 0.5, 0.5, 0.415, seed=61, w=SW, n=11, wobble=0.11)
    dot(q, 0.5, 0.5, 0.075)
    for k in range(8):
        a = TAU * k / 8 + 0.20
        r0, r1 = 0.325, 0.175
        x1, y1 = 0.5 + r0 * math.cos(a), 0.5 + r0 * math.sin(a)
        x2, y2 = 0.5 + r1 * math.cos(a), 0.5 + r1 * math.sin(a)
        arrow(q, x1, y1, x2, y2, SW_THIN, 0.042)


def tr_07(q):
    """Ground truth vs prediction: the mismatch between the two outlines."""
    a = blob(0.415, 0.535, 0.335, seed=71, n=9, wobble=0.11)
    b = blob(0.575, 0.455, 0.335, seed=71, n=9, wobble=0.11)
    diff = a.subtracted(b).united(b.subtracted(a)).simplified()
    q.save()
    q.setClipPath(diff)
    stroke(q, SW_THIN * 0.85)
    for k in range(-12, 18):
        x = k * 0.078
        q.drawLine(QPointF(x, 0.0), QPointF(x + 1.0, 1.0))
    q.restore()
    q.setPen(QPen(WHITE, SW, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    q.setBrush(Qt.NoBrush)
    q.drawPath(a)
    pen = QPen(WHITE, SW, Qt.CustomDashLine, Qt.FlatCap, Qt.RoundJoin)
    pen.setDashPattern([0.052 / SW, 0.044 / SW])
    q.setPen(pen)
    q.drawPath(b)


def tr_08(q):
    """A labelled training set feeding the model."""
    rrect(q, 0.185, 0.115, 0.44, 0.44, 0.05, SW_THIN)
    rrect(q, 0.115, 0.215, 0.44, 0.44, 0.05, SW_THIN)
    rrect(q, 0.045, 0.315, 0.44, 0.44, 0.05, SW)
    clip = QPainterPath()
    clip.addRoundedRect(QRectF(0.045 + SW / 2, 0.315 + SW / 2, 0.44 - SW, 0.44 - SW),
                        0.05, 0.05)
    q.save()
    q.setClipPath(clip)
    draw_blob(q, 0.185, 0.455, 0.098, seed=81, fill=True)
    draw_blob(q, 0.355, 0.545, 0.078, seed=82, fill=True)
    draw_blob(q, 0.215, 0.665, 0.072, seed=83, fill=True)
    q.restore()
    arrow(q, 0.545, 0.535, 0.685, 0.535, SW_THIN, 0.050)
    rrect(q, 0.735, 0.365, 0.225, 0.34, 0.05, SW)
    stroke(q, SW_THIN)
    for i in range(3):
        y = 0.435 + i * 0.10
        q.drawLine(QPointF(0.785, y), QPointF(0.905, y))


def tr_09(q):
    """Teaching the model: a graduate cell."""
    draw_cell(q, 0.465, 0.705, 0.245, seed=91, w=SW, nuc=0.32, organelles=3)
    poly(q, [(0.335, 0.395), (0.335, 0.325), (0.635, 0.325), (0.635, 0.395)], SW)
    poly(q, [(0.50, 0.115), (0.945, 0.275), (0.50, 0.435), (0.055, 0.275)], SW, close=True)
    stroke(q, SW_THIN)
    path = QPainterPath()
    path.moveTo(0.835, 0.315)
    path.cubicTo(0.875, 0.42, 0.865, 0.50, 0.825, 0.545)
    q.drawPath(path)
    dot(q, 0.825, 0.585, 0.050)


def tr_10(q):
    """Model chip with a training feedback loop."""
    rrect(q, 0.275, 0.215, 0.45, 0.40, 0.055, SW)
    stroke(q, SW_THIN)
    for i in range(3):
        y = 0.285 + i * 0.135
        q.drawLine(QPointF(0.19, y), QPointF(0.275, y))
        q.drawLine(QPointF(0.725, y), QPointF(0.81, y))
    draw_text(q, "AI", 0.50, 0.415, 0.135)
    draw_cell(q, 0.085, 0.285, 0.070, seed=101, w=SW_THIN, nuc=0.40)
    draw_blob(q, 0.915, 0.285, 0.070, seed=101, fill=True)
    stroke(q, SW_THIN)
    path = QPainterPath()
    path.moveTo(0.915, 0.375)
    path.cubicTo(0.915, 0.80, 0.72, 0.845, 0.50, 0.845)
    path.cubicTo(0.28, 0.845, 0.085, 0.80, 0.085, 0.60)
    q.drawPath(path)
    for s in (+1, -1):
        b = -math.pi / 2 + s * 0.55
        q.drawLine(QPointF(0.085, 0.60),
                   QPointF(0.085 + 0.058 * math.cos(b), 0.60 + 0.058 * math.sin(b)))


# ---------------------------------------------------------------------------
# UMAP -- 2-D embedding of many objects
# ---------------------------------------------------------------------------

def _cluster(q, cx, cy, r, n, seed, rmin=0.026, rmax=0.040, gap=0.020):
    """Non-overlapping blue-noise cluster of dots (deterministic for a seed)."""
    rnd = random.Random(seed)
    pts = []
    for _ in range(6000):
        if len(pts) >= n:
            break
        a = rnd.uniform(0, TAU)
        rr = r * math.sqrt(rnd.random())
        x, y = cx + rr * math.cos(a), cy + rr * math.sin(a) * 0.92
        s = rnd.uniform(rmin, rmax)
        if all((x - px) ** 2 + (y - py) ** 2 >= (s + ps + gap) ** 2 for px, py, ps in pts):
            pts.append((x, y, s))
    for x, y, s in pts:
        dot(q, x, y, s)


def um_01(q):
    """The embedding plot: axes and separated clusters."""
    poly(q, [(0.115, 0.075), (0.115, 0.885), (0.945, 0.885)], SW_THIN)
    _cluster(q, 0.325, 0.285, 0.155, 7, 111, 0.032, 0.046)
    _cluster(q, 0.735, 0.315, 0.135, 5, 112, 0.032, 0.046)
    _cluster(q, 0.545, 0.665, 0.165, 8, 113, 0.032, 0.046)


def um_02(q):
    """Cells become points."""
    draw_cell(q, 0.145, 0.235, 0.115, seed=121, w=SW_THIN, nuc=0.36)
    draw_cell(q, 0.155, 0.55, 0.105, seed=122, w=SW_THIN, nuc=0.36)
    draw_cell(q, 0.145, 0.845, 0.110, seed=123, w=SW_THIN, nuc=0.36)
    arrow(q, 0.315, 0.54, 0.455, 0.54, SW_THIN, 0.050)
    _cluster(q, 0.665, 0.315, 0.150, 6, 124, 0.030, 0.042)
    _cluster(q, 0.775, 0.715, 0.155, 6, 125, 0.030, 0.042)


def um_03(q):
    """High-dimensional cube projected onto a plane."""
    a, b, o = 0.14, 0.60, 0.115
    poly(q, [(a, 0.10), (b, 0.10), (b, 0.44), (a, 0.44)], SW_THIN, close=True)
    poly(q, [(a + o, 0.10 - o), (b + o, 0.10 - o), (b + o, 0.44 - o), (a + o, 0.44 - o)],
         SW_THIN, close=True)
    stroke(q, SW_THIN)
    for x, y in ((a, 0.10), (b, 0.10), (b, 0.44), (a, 0.44)):
        q.drawLine(QPointF(x, y), QPointF(x + o, y - o))
    for x, y in ((0.235, 0.155), (0.345, 0.335), (0.455, 0.215), (0.415, 0.075),
                 (0.565, 0.365), (0.615, 0.155), (0.245, 0.375)):
        dot(q, x, y, 0.030)
    arrow(q, 0.5, 0.52, 0.5, 0.635, SW_THIN, 0.050)
    poly(q, [(0.10, 0.80), (0.62, 0.80), (0.90, 0.965), (0.38, 0.965)], SW, close=True)
    for x, y in ((0.30, 0.855), (0.43, 0.915), (0.55, 0.845), (0.65, 0.925), (0.71, 0.865)):
        dot(q, x, y, 0.030)


def um_04(q):
    """Cluster hulls around the embedded points."""
    for cx, cy, r, seed, n in ((0.295, 0.295, 0.225, 141, 5), (0.735, 0.335, 0.185, 142, 4),
                               (0.565, 0.755, 0.215, 143, 5)):
        draw_blob(q, cx, cy, r, seed=seed, w=SW_THIN, n=8, wobble=0.13)
        _cluster(q, cx, cy, r * 0.50, n, seed + 7, 0.032, 0.044, gap=0.022)


def um_05(q):
    """Density contours over the embedding."""
    for cx, cy, seed, rr in ((0.305, 0.325, 151, (0.100, 0.175, 0.255)),
                             (0.715, 0.710, 152, (0.090, 0.160, 0.235))):
        for k, r in enumerate(rr):
            draw_blob(q, cx, cy, r, seed=seed, w=SW if k == len(rr) - 1 else SW_THIN,
                      n=9, wobble=0.08)
        dot(q, cx, cy, 0.044)


def um_06(q):
    """Feature matrix reduced to a scatter."""
    x0, y0, c = 0.055, 0.235, 0.093
    stroke(q, SW_THIN)
    for i in range(5):
        q.drawLine(QPointF(x0, y0 + i * c), QPointF(x0 + 4 * c, y0 + i * c))
        q.drawLine(QPointF(x0 + i * c, y0), QPointF(x0 + i * c, y0 + 4 * c))
    rnd = random.Random(161)
    solid(q)
    for i in range(4):
        for j in range(4):
            if rnd.random() < 0.45:
                q.drawRect(QRectF(x0 + i * c + 0.012, y0 + j * c + 0.012,
                                  c - 0.024, c - 0.024))
    arrow(q, 0.475, 0.50, 0.60, 0.50, SW_THIN, 0.048)
    poly(q, [(0.665, 0.185), (0.665, 0.815), (0.955, 0.815)], SW_THIN)
    _cluster(q, 0.765, 0.365, 0.085, 4, 162, 0.028, 0.036, gap=0.018)
    _cluster(q, 0.875, 0.625, 0.075, 4, 163, 0.028, 0.036, gap=0.018)


def um_07(q):
    """The neighbour graph the embedding is built from."""
    pts = [(0.175, 0.30), (0.34, 0.185), (0.30, 0.46), (0.50, 0.35), (0.47, 0.60),
           (0.665, 0.235), (0.68, 0.52), (0.84, 0.375), (0.60, 0.78), (0.82, 0.715),
           (0.24, 0.72), (0.40, 0.845)]
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (5, 6), (5, 7),
             (6, 7), (4, 6), (4, 8), (8, 9), (6, 9), (2, 10), (10, 11), (11, 8), (4, 10)]
    stroke(q, SW_THIN * 0.85)
    for i, j in edges:
        q.drawLine(QPointF(*pts[i]), QPointF(*pts[j]))
    for i, p in enumerate(pts):
        dot(q, p[0], p[1], 0.042 if i % 3 == 0 else 0.034)


def um_08(q):
    """The manifold unrolled: swiss roll to a flat strip."""
    def spiral(t):
        a = TAU * 1.18 * t + 2.55
        r = 0.115 + 0.235 * t
        return 0.375 + r * math.cos(a), 0.345 + r * math.sin(a)

    stroke(q, SW)
    path = QPainterPath()
    for i in range(161):
        x, y = spiral(i / 160.0)
        if i == 0:
            path.moveTo(x, y)
        else:
            path.lineTo(x, y)
    q.drawPath(path)
    for i in range(6):
        x, y = spiral(0.04 + i * 0.19)
        dot(q, x, y, 0.032)
    arrow(q, 0.735, 0.415, 0.875, 0.585, SW_THIN, 0.055)
    line(q, 0.085, 0.865, 0.915, 0.865, SW)
    for i in range(6):
        dot(q, 0.145 + i * 0.142, 0.865, 0.038)


def um_09(q):
    """Many dimensions funnelled down to two."""
    solid(q)
    for i in range(8):
        x = 0.085 + i * 0.108
        h = (0.10, 0.20, 0.145, 0.235, 0.115, 0.19, 0.25, 0.135)[i]
        q.drawRoundedRect(QRectF(x, 0.285 - h, 0.070, h), 0.030, 0.030)
    poly(q, [(0.055, 0.345), (0.945, 0.345), (0.595, 0.60), (0.405, 0.60)],
         SW, close=True)
    poly(q, [(0.30, 0.68), (0.30, 0.94), (0.80, 0.94)], SW_THIN)
    for x, y in ((0.42, 0.78), (0.53, 0.85), (0.62, 0.755), (0.68, 0.875), (0.49, 0.72)):
        dot(q, x, y, 0.034)


def um_10(q):
    """Projection: a point cloud focused through a lens onto a plane."""
    # rays travelling from the cloud, through the lens, onto the plane
    stroke(q, SW_THIN * 0.9)
    rays = (((0.215, 0.355), 0.265), ((0.105, 0.505), 0.475),
            ((0.225, 0.665), 0.685), ((0.135, 0.815), 0.845))
    for (sx0, sy0), ty in rays:
        q.drawLine(QPointF(sx0 + 0.062, sy0), QPointF(0.878, ty))
    for x, y in ((0.115, 0.215), (0.215, 0.355), (0.105, 0.505), (0.225, 0.665),
                 (0.135, 0.815)):
        dot(q, x, y, 0.042)
    stroke(q, SW)
    path = QPainterPath()
    path.moveTo(0.54, 0.115)
    path.cubicTo(0.655, 0.30, 0.655, 0.70, 0.54, 0.885)
    path.cubicTo(0.425, 0.70, 0.425, 0.30, 0.54, 0.115)
    q.drawPath(path)
    line(q, 0.895, 0.135, 0.895, 0.925, SW)
    for _, y in rays:
        dot(q, 0.895, y, 0.040)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

GROUPS = {
    "run": [
        ("play triangle built from cells (the pipeline is made of objects)", run_01),
        ("petri dish with an inscribed play triangle", run_02),
        ("power / start button ring around a play triangle", run_03),
        ("stopwatch with a play triangle on the face: start a timed run", run_04),
        ("three-stage pipeline (dish -> image tile -> plot) ending in go", run_05),
        ("fast-forward chevrons advancing over a row of samples", run_06),
        ("microscope objective over a slide, with a start triangle", run_07),
        ("funnel: a stack of raw images in, one measured object out", run_08),
        ("stepper track: stages completed and stages still to run", run_09),
        ("terminal window with a prompt and cursor: execute the pipeline", run_10),
    ],
    "sequencing": [
        ("barcode bars above a DNA double helix", seq_01),
        ("flow cell: lanes with sequencing clusters", seq_02),
        ("nanopore: a strand threading a membrane pore", seq_03),
        ("Sanger chromatogram: base peaks on a baseline", seq_04),
        ("read pile-up aligned against a reference", seq_05),
        ("the four bases as A/T/G/C code tiles", seq_06),
        ("a barcode tag attached to a cell (barcode-to-cell mapping)", seq_07),
        ("magnifier reading a barcode", seq_08),
        ("a FASTQ record: @header, sequence, +, quality", seq_09),
        ("electrophoresis gel: lanes of bands", seq_10),
    ],
    "settings": [
        ("sliders", set_01),
        ("toggle switches", set_02),
        ("rotary dials with tick arcs", set_03),
        ("a wrench adjusting a cell", set_04),
        ("a cell whose nucleus is a gear", set_05),
        ("equalizer / fader bank", set_06),
        ("configuration checklist with ticked boxes", set_07),
        ("histogram with a draggable threshold line", set_08),
        ("crossed wrench and screwdriver", set_09),
        ("meshing gear train: coupled parameters", set_10),
    ],
    "train_cellpose": [
        ("pencil hand-annotating an outline (making the training label)", tr_01),
        ("image/label pairs: raw tile in, mask tile out", tr_02),
        ("loss curve falling over epochs", tr_03),
        ("layered neural net learning cell -> mask", tr_04),
        ("epoch loop: iterate until the mask fits", tr_05),
        ("Cellpose flow field: vectors converging on the cell centre", tr_06),
        ("ground truth vs prediction outlines, mismatch area hatched", tr_07),
        ("a labelled training set (deck of tiles) feeding the model", tr_08),
        ("graduation cap on a cell: teaching the model", tr_09),
        ("model chip with a training feedback loop", tr_10),
    ],
    "umap": [
        ("the embedding plot: axes with separated dot clusters", um_01),
        ("cells become points: outlines in, scatter out", um_02),
        ("high-dimensional cube projected onto a 2-D plane", um_03),
        ("cluster hulls drawn around the embedded points", um_04),
        ("density contours over the embedding", um_05),
        ("feature matrix reduced to a scatter", um_06),
        ("the k-nearest-neighbour graph the embedding is built from", um_07),
        ("swiss roll manifold unrolled into a flat strip", um_08),
        ("many feature dimensions funnelled down to two axes", um_09),
        ("point cloud focused through a lens onto a plane", um_10),
    ],
}


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def render(fn, size: int = SIZE) -> QImage:
    img = QImage(size, size, QImage.Format_RGBA8888)
    img.fill(QColor(255, 255, 255, 0))
    q = QPainter(img)
    q.setRenderHint(QPainter.Antialiasing, True)
    q.setRenderHint(QPainter.SmoothPixmapTransform, True)
    q.scale(size, size)
    fn(q)
    q.end()
    return img


def contact_sheet(paths, out, bg, fg, title, ink=None, cols=5, cell=300, pad=18):
    """Grid of every variant on one background.

    ink=None keeps the artwork white (dark sheet).  ink=(r,g,b) re-inks the
    alpha mask in that colour, which is the only way the light sheet can show
    anything at all: the PNGs are pure white on transparent.
    """
    from PIL import Image, ImageDraw, ImageFont

    def load(p, size):
        im = Image.open(p).convert("RGBA").resize((size, size), Image.LANCZOS)
        if ink is not None:
            im = Image.merge("RGBA", (
                Image.new("L", im.size, ink[0]), Image.new("L", im.size, ink[1]),
                Image.new("L", im.size, ink[2]), im.getchannel("A")))
        tile = Image.new("RGBA", (size, size), bg + (255,))
        tile.alpha_composite(im)
        return tile.convert("RGB")

    rows = (len(paths) + cols - 1) // cols
    lab, head = 46, 64
    W = cols * (cell + pad) + pad
    H = head + rows * (cell + pad + lab) + pad
    sheet = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 21)
    except Exception:
        font = small_font = ImageFont.load_default()
    d.text((pad, 14), title, fill=fg, font=font)
    if ink is not None:
        d.text((pad, 48), "artwork is white-on-transparent; shown re-inked so it is "
                          "visible on a light background", fill=fg, font=small_font)
    else:
        d.text((pad, 48), "48 px preview at the right of each cell",
               fill=fg, font=small_font)
    for i, p in enumerate(paths):
        r, c = divmod(i, cols)
        x = pad + c * (cell + pad)
        y = head + pad + r * (cell + pad + lab)
        sheet.paste(load(p, cell), (x, y))
        d.text((x + 4, y + cell + 6), os.path.basename(p).rsplit("_", 1)[-1].split(".")[0],
               fill=fg, font=font)
        sheet.paste(load(p, 48), (x + cell - 52, y + cell + 2))
    sheet.save(out)


def main(outroot=None):
    app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])
    here = os.path.dirname(os.path.abspath(__file__))
    outroot = outroot or os.path.abspath(os.path.join(here, os.pardir))
    for name, variants in GROUPS.items():
        d = os.path.join(outroot, name)
        os.makedirs(d, exist_ok=True)
        paths = []
        for i, (desc, fn) in enumerate(variants, start=1):
            p = os.path.join(d, "%s_%02d.png" % (name, i))
            render(fn).save(p)
            paths.append(p)
        with open(os.path.join(d, "CONCEPTS.md"), "w") as fh:
            fh.write("# %s -- candidate concepts\n\n" % name)
            for i, (desc, _) in enumerate(variants, start=1):
                fh.write("%d. **%s_%02d** - %s\n" % (i, name, i, desc))
            fh.write("\n_All 1024x1024 RGBA, white on transparent, house style "
                     "(flat, thin strokes + solid fills), matching "
                     "`plaque.png` / `measure.png`._\n")
            fh.write("\n_`_sheet_dark.png` shows the PNGs as they are. "
                     "`_sheet_light.png` re-inks the same alpha masks in dark grey, "
                     "because pure-white artwork is invisible on a light "
                     "background -- that is the point of the second sheet._\n")
        contact_sheet(paths, os.path.join(d, "_sheet_dark.png"), (20, 22, 26),
                      (255, 190, 90), "%s -- candidates on dark" % name)
        contact_sheet(paths, os.path.join(d, "_sheet_light.png"), (245, 246, 248),
                      (30, 30, 34), "%s -- candidates on light" % name,
                      ink=(26, 28, 32))
        print("wrote %2d icons + 2 sheets -> %s" % (len(paths), d))
    _ = app


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
