#!/usr/bin/env python3
"""Candidate icon generator for spaCR - group: classify / convert / default / download.

House style (derived from resources/icons/plaque.png and measure.png):
  * pure white artwork on a fully transparent background (RGB is white everywhere,
    alpha carries the shape - the files are effectively monochrome masks)
  * flat, no gradients, no colour
  * mix of thin outlined strokes and solid white fills
  * square canvas, subject fills most of the frame with a modest margin
  * literal-but-stylised biology / lab objects, not abstract glyphs

Everything is drawn in a normalised 0..1 coordinate space and scaled to the canvas,
so any variant renders correctly at any size.  Nothing here is random; the output is
byte-for-byte reproducible.

Usage:
    QT_QPA_PLATFORM=offscreen python3 classify_convert_default_download.py [OUTDIR]

OUTDIR defaults to the backup_icons directory two levels up from this file.
"""

from __future__ import annotations

import math
import os
import random
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, QRectF, Qt  # noqa: E402
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

_APP = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])

random.seed(20260726)

# --------------------------------------------------------------------------------------
# canvas / style constants
# --------------------------------------------------------------------------------------

SIZE = 1024
WHITE = QColor(255, 255, 255, 255)

W_MAIN = 0.022   # primary outline weight (normalised)
W_THIN = 0.015   # secondary / detail outlines
W_HAIR = 0.011   # fine detail (graph edges, ticks)

SHEET_BG_DARK = QColor(0x14, 0x16, 0x1a)
SHEET_BG_LIGHT = QColor(0xf5, 0xf6, 0xf8)


# --------------------------------------------------------------------------------------
# painter helpers - all coordinates normalised 0..1
# --------------------------------------------------------------------------------------


def _pen(w, cap=Qt.RoundCap, join=Qt.RoundJoin):
    q = QPen(WHITE)
    q.setWidthF(w)
    q.setCapStyle(cap)
    q.setJoinStyle(join)
    return q


def stroke(p, w=W_MAIN, cap=Qt.RoundCap, join=Qt.RoundJoin):
    """Switch the painter to outline mode at weight *w*."""
    p.setPen(_pen(w, cap, join))
    p.setBrush(Qt.NoBrush)


def solid(p):
    """Switch the painter to solid-white fill mode."""
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(WHITE))


def clear_mode(p, on=True):
    p.setCompositionMode(
        QPainter.CompositionMode_Clear if on else QPainter.CompositionMode_SourceOver
    )


def ln(p, x1, y1, x2, y2):
    p.drawLine(QPointF(x1, y1), QPointF(x2, y2))


def circ(p, cx, cy, r):
    p.drawEllipse(QPointF(cx, cy), r, r)


def ell(p, cx, cy, rx, ry, ang=0.0):
    p.save()
    p.translate(cx, cy)
    p.rotate(ang)
    p.drawEllipse(QPointF(0, 0), rx, ry)
    p.restore()


def rrect(p, x0, y0, x1, y1, r=0.04):
    p.drawRoundedRect(QRectF(x0, y0, x1 - x0, y1 - y0), r, r)


def poly(p, pts, close=True):
    path = QPainterPath(QPointF(*pts[0]))
    for q in pts[1:]:
        path.lineTo(QPointF(*q))
    if close:
        path.closeSubpath()
    p.drawPath(path)


def dot(p, cx, cy, r):
    solid(p)
    circ(p, cx, cy, r)


def ring(p, cx, cy, r, w=W_THIN):
    stroke(p, w)
    circ(p, cx, cy, r)


def arrow(p, x1, y1, x2, y2, w=W_MAIN, hl=0.11, hw=0.085):
    """Stroked shaft + solid triangular head, from (x1,y1) to (x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L <= 1e-9:
        return
    ux, uy = dx / L, dy / L
    bx, by = x2 - ux * hl, y2 - uy * hl
    stroke(p, w)
    ln(p, x1, y1, bx + ux * hl * 0.45, by + uy * hl * 0.45)
    px, py = -uy, ux
    solid(p)
    poly(
        p,
        [
            (x2, y2),
            (bx + px * hw / 2, by + py * hw / 2),
            (bx - px * hw / 2, by - py * hw / 2),
        ],
    )


def arc_pts(cx, cy, r, a0, a1, n=96, ry=None):
    """Sample an (elliptical) arc. Angles in degrees, maths convention, y-down."""
    rx, ry = r, r if ry is None else ry
    out = []
    for i in range(n + 1):
        a = math.radians(a0 + (a1 - a0) * i / n)
        out.append((cx + rx * math.cos(a), cy - ry * math.sin(a)))
    return out


def blob_union(circles=(), rects=(), scale=2000.0):
    """Smooth union outline of circles/rounded rects.

    Qt flattens curves when doing path booleans, so the union is computed in a
    blown-up coordinate space and mapped back down - otherwise the result is
    visibly faceted at 1024 px.
    """
    acc = QPainterPath()
    for cx, cy, r in circles:
        sub = QPainterPath()
        sub.addEllipse(QPointF(cx * scale, cy * scale), r * scale, r * scale)
        acc = sub if acc.isEmpty() else acc.united(sub)
    for x0, y0, x1, y1, rr in rects:
        sub = QPainterPath()
        sub.addRoundedRect(
            QRectF(x0 * scale, y0 * scale, (x1 - x0) * scale, (y1 - y0) * scale),
            rr * scale, rr * scale)
        acc = sub if acc.isEmpty() else acc.united(sub)
    acc = acc.simplified()
    t = QTransform()
    t.scale(1.0 / scale, 1.0 / scale)
    return t.map(acc)


def zstack(p, cx, y0, dy, n, hw, hh, w=W_THIN):
    """n isometric slices stacked vertically, drawn with a real gap between them."""
    stroke(p, w)
    for i in range(n):
        y = y0 + i * dy
        poly(p, [(cx - hw, y), (cx, y - hh), (cx + hw, y), (cx, y + hh)])


def arc(p, cx, cy, r, a0, a1, w=W_MAIN, n=96):
    stroke(p, w)
    poly(p, arc_pts(cx, cy, r, a0, a1, n), close=False)


def arc_arrow(p, cx, cy, r, a0, a1, w=W_MAIN, hl=0.10, hw=0.085):
    """Curved arrow along an arc, head at the a1 end."""
    pts = arc_pts(cx, cy, r, a0, a1, 160)
    # trim `hl` of arc-length off the tail end for the head
    acc = 0.0
    cut = len(pts) - 1
    for i in range(len(pts) - 1, 0, -1):
        acc += math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        if acc >= hl:
            cut = i
            break
    stroke(p, w)
    poly(p, pts[: cut + 1], close=False)
    tip = pts[-1]
    base = pts[cut]
    dx, dy = tip[0] - base[0], tip[1] - base[1]
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    px, py = -uy, ux
    solid(p)
    poly(
        p,
        [
            tip,
            (base[0] + px * hw / 2, base[1] + py * hw / 2),
            (base[0] - px * hw / 2, base[1] - py * hw / 2),
        ],
    )


def gear(p, cx, cy, r_out, r_in, teeth=10, w=W_MAIN):
    pts = []
    step = 2 * math.pi / teeth
    fr = [(0.00, r_out), (0.34, r_out), (0.50, r_in), (0.84, r_in)]
    for i in range(teeth):
        a0 = i * step
        for f, rr in fr:
            a = a0 + f * step
            pts.append((cx + rr * math.cos(a), cy - rr * math.sin(a)))
    stroke(p, w, join=Qt.RoundJoin)
    poly(p, pts, close=True)


def cell(p, cx, cy, r, w=W_THIN, nuc=0.36, organelles=0, seed_ang=0.6):
    """Standard spaCR cell glyph: outlined membrane + solid nucleus (+ organelles)."""
    stroke(p, w)
    circ(p, cx, cy, r)
    if organelles:
        stroke(p, max(W_HAIR, w * 0.8))
        for i in range(organelles):
            a = seed_ang + i * (2 * math.pi / organelles)
            ox, oy = cx + 0.60 * r * math.cos(a), cy - 0.60 * r * math.sin(a)
            ell(p, ox, oy, r * 0.24, r * 0.14, -math.degrees(a) * 0.5)
    if nuc:
        dot(p, cx - 0.10 * r, cy - 0.05 * r, r * nuc)


def lens(p, cx, cy, rx, ry, ang=0.0):
    """Solid lens/plaque shape - the plaque.png vocabulary."""
    solid(p)
    ell(p, cx, cy, rx, ry, ang)


def tile(p, x0, y0, x1, y1, w=W_THIN, r=0.03):
    stroke(p, w)
    rrect(p, x0, y0, x1, y1, r)


def pixel_grid(p, x0, y0, x1, y1, n=3, filled=(), w=W_HAIR):
    """n x n grid; cells listed in *filled* as (col,row) are solid."""
    cw, ch = (x1 - x0) / n, (y1 - y0) / n
    stroke(p, w)
    rrect(p, x0, y0, x1, y1, 0.012)
    for i in range(1, n):
        ln(p, x0 + i * cw, y0, x0 + i * cw, y1)
        ln(p, x0, y0 + i * ch, x1, y0 + i * ch)
    solid(p)
    for c, r in filled:
        g = min(cw, ch) * 0.16
        p.drawRect(QRectF(x0 + c * cw + g, y0 + r * ch + g, cw - 2 * g, ch - 2 * g))


def page(p, x0, y0, x1, y1, fold=0.20, w=W_MAIN):
    """Document glyph with a folded top-right corner."""
    f = (x1 - x0) * fold
    stroke(p, w)
    poly(p, [(x0, y0), (x1 - f, y0), (x1, y0 + f), (x1, y1), (x0, y1)])
    stroke(p, max(W_HAIR, w * 0.65))
    poly(p, [(x1 - f, y0), (x1 - f, y0 + f), (x1, y0 + f)], close=False)


def bowl(p, cx, rim_y, half_w, depth, w=W_MAIN):
    """Open dish seen from the side: rim line + lower half ellipse."""
    stroke(p, w)
    poly(p, arc_pts(cx, rim_y, half_w, 180, 360, 80, ry=depth), close=False)
    ln(p, cx - half_w, rim_y, cx + half_w, rim_y)


def occlude(p, draw_fill_shape):
    """Clear whatever *draw_fill_shape(painter)* paints, for clean overlap."""
    clear_mode(p, True)
    solid(p)
    draw_fill_shape(p)
    clear_mode(p, False)


# ======================================================================================
# CLASSIFY - a trained model sorting objects into classes
# ======================================================================================


def classify_01(p):
    """Decision boundary: two populations split by a learned line."""
    stroke(p, W_THIN)
    ln(p, 0.07, 0.05, 0.07, 0.93)
    ln(p, 0.07, 0.93, 0.95, 0.93)
    stroke(p, W_MAIN)
    ln(p, 0.12, 0.72, 0.92, 0.24)
    for x, y in [(0.24, 0.26), (0.38, 0.42), (0.34, 0.15), (0.53, 0.27), (0.20, 0.48)]:
        dot(p, x, y, 0.052)
    for x, y in [(0.60, 0.70), (0.74, 0.57), (0.85, 0.74), (0.67, 0.85), (0.89, 0.48)]:
        ring(p, x, y, 0.052, W_THIN)


def classify_02(p):
    """Two bins: a mixed population is split into two labelled containers."""
    stroke(p, W_MAIN)
    poly(p, [(0.06, 0.52), (0.06, 0.91), (0.44, 0.91), (0.44, 0.52)], close=False)
    poly(p, [(0.56, 0.52), (0.56, 0.91), (0.94, 0.91), (0.94, 0.52)], close=False)
    for x, y in [(0.16, 0.80), (0.34, 0.80), (0.25, 0.65)]:
        dot(p, x, y, 0.058)
    for x, y in [(0.66, 0.80), (0.84, 0.80), (0.75, 0.65)]:
        ring(p, x, y, 0.058, W_THIN)
    dot(p, 0.42, 0.09, 0.055)
    ring(p, 0.58, 0.09, 0.055, W_THIN)
    arrow(p, 0.41, 0.18, 0.26, 0.40, W_THIN, hl=0.09, hw=0.072)
    arrow(p, 0.59, 0.18, 0.74, 0.40, W_THIN, hl=0.09, hw=0.072)


def classify_03(p):
    """Decision tree: one object splits down branches into leaf classes."""
    stroke(p, W_THIN)
    ln(p, 0.50, 0.26, 0.26, 0.44)
    ln(p, 0.50, 0.26, 0.74, 0.44)
    ln(p, 0.22, 0.60, 0.13, 0.74)
    ln(p, 0.30, 0.60, 0.39, 0.74)
    ln(p, 0.70, 0.60, 0.61, 0.74)
    ln(p, 0.78, 0.60, 0.87, 0.74)
    cell(p, 0.50, 0.14, 0.105, W_MAIN, nuc=0.40)
    ring(p, 0.26, 0.52, 0.082, W_THIN)
    ring(p, 0.74, 0.52, 0.082, W_THIN)
    dot(p, 0.13, 0.82, 0.070)
    ring(p, 0.39, 0.82, 0.070, W_THIN)
    ring(p, 0.61, 0.82, 0.070, W_THIN)
    dot(p, 0.87, 0.82, 0.070)


def classify_04(p):
    """Confusion matrix: predicted vs true, the diagonal lit."""
    x0, y0, s = 0.30, 0.12, 0.205
    stroke(p, W_THIN)
    rrect(p, x0, y0, x0 + 3 * s, y0 + 3 * s, 0.02)
    for i in (1, 2):
        ln(p, x0 + i * s, y0, x0 + i * s, y0 + 3 * s)
        ln(p, x0, y0 + i * s, x0 + 3 * s, y0 + i * s)
    solid(p)
    for i in range(3):
        g = s * 0.13
        p.drawRoundedRect(
            QRectF(x0 + i * s + g, y0 + i * s + g, s - 2 * g, s - 2 * g), 0.012, 0.012
        )
    for i in range(3):
        cell(p, 0.155, y0 + (i + 0.5) * s, 0.062, W_THIN, nuc=0.40)
        cell(p, x0 + (i + 0.5) * s, 0.855, 0.062, W_THIN, nuc=0.40)


def classify_05(p):
    """Trained network: a cell goes in, one output class lights up."""
    hx, hy = 0.49, [0.17, 0.50, 0.83]
    ox, oy = 0.87, [0.26, 0.50, 0.74]
    stroke(p, W_HAIR)
    for y in hy:
        ln(p, 0.295, 0.50, hx - 0.05, y)
        for z in oy:
            ln(p, hx + 0.05, y, ox - 0.058, z)
    cell(p, 0.155, 0.50, 0.145, W_MAIN, nuc=0.36, organelles=3)
    for y in hy:
        ring(p, hx, y, 0.050, W_THIN)
    dot(p, ox, oy[0], 0.060)
    for z in oy[1:]:
        ring(p, ox, z, 0.060, W_THIN)


def classify_06(p):
    """Sorter: a Y channel routes each object down its own arm."""
    stroke(p, W_MAIN)
    poly(p, [(0.02, 0.415), (0.40, 0.415), (0.97, 0.135)], close=False)
    poly(p, [(0.02, 0.585), (0.40, 0.585), (0.97, 0.865)], close=False)
    splitter = QPainterPath(QPointF(0.97, 0.325))
    splitter.lineTo(QPointF(0.645, 0.462))
    splitter.quadTo(QPointF(0.575, 0.50), QPointF(0.645, 0.538))
    splitter.lineTo(QPointF(0.97, 0.675))
    p.drawPath(splitter)
    dot(p, 0.13, 0.50, 0.055)
    ring(p, 0.285, 0.50, 0.055, W_THIN)
    dot(p, 0.795, 0.302, 0.052)
    ring(p, 0.795, 0.698, 0.052, W_THIN)


def classify_07(p):
    """Sieve: a mesh separates the population by size."""
    stroke(p, W_MAIN)
    for x in (0.06, 0.24, 0.42, 0.60, 0.78):
        ln(p, x, 0.52, x + 0.14, 0.52)
    stroke(p, W_THIN)
    ell(p, 0.27, 0.34, 0.145, 0.115, -12)
    ell(p, 0.70, 0.33, 0.135, 0.120, 14)
    ell(p, 0.49, 0.13, 0.115, 0.095, -4)
    dot(p, 0.27, 0.34, 0.038)
    dot(p, 0.70, 0.33, 0.038)
    dot(p, 0.49, 0.13, 0.032)
    for x, y, r in [
        (0.20, 0.70, 0.048),
        (0.49, 0.76, 0.048),
        (0.77, 0.68, 0.048),
        (0.33, 0.90, 0.040),
        (0.63, 0.92, 0.040),
    ]:
        dot(p, x, y, r)


def classify_08(p):
    """Assigned label: a cell wearing a class tag."""
    cell(p, 0.40, 0.58, 0.315, W_MAIN, nuc=0.30, organelles=4, seed_ang=1.1)
    stroke(p, W_MAIN)
    poly(p, [(0.54, 0.19), (0.65, 0.07), (0.95, 0.07), (0.95, 0.31), (0.65, 0.31)])
    dot(p, 0.715, 0.19, 0.030)
    stroke(p, W_THIN)
    ln(p, 0.79, 0.152, 0.90, 0.152)
    ln(p, 0.79, 0.228, 0.90, 0.228)


def classify_09(p):
    """Class counts: a histogram whose bars are stacks of cells."""
    stroke(p, W_THIN)
    ln(p, 0.06, 0.93, 0.94, 0.93)
    ys = [0.815, 0.625, 0.435, 0.245]
    for y in ys:
        dot(p, 0.22, y, 0.082)
    for y in ys[:2]:
        ring(p, 0.50, y, 0.082, W_THIN)
    for y in ys[:3]:
        ring(p, 0.78, y, 0.082, W_THIN)
        dot(p, 0.78, y, 0.034)


def classify_10(p):
    """Class scores: the model's per-class confidence beside the object."""
    cell(p, 0.235, 0.50, 0.205, W_MAIN, nuc=0.33, organelles=3, seed_ang=0.4)
    stroke(p, W_THIN)
    ln(p, 0.485, 0.155, 0.485, 0.845)
    solid(p)
    p.drawRoundedRect(QRectF(0.50, 0.200, 0.44, 0.125), 0.022, 0.022)
    stroke(p, W_THIN)
    rrect(p, 0.50, 0.4375, 0.77, 0.5625, 0.022)
    rrect(p, 0.50, 0.675, 0.665, 0.800, 0.022)


# ======================================================================================
# CONVERT - nd2 / czi / lif -> Yokogawa TIFF
# ======================================================================================


def convert_01(p):
    """Two documents: a proprietary stack becomes a pixel-grid image."""
    page(p, 0.03, 0.19, 0.37, 0.81, 0.22, W_MAIN)
    stroke(p, W_HAIR)
    for y in (0.46, 0.56, 0.66):
        ell(p, 0.20, y, 0.10, 0.030)
    page(p, 0.63, 0.19, 0.97, 0.81, 0.22, W_MAIN)
    pixel_grid(p, 0.70, 0.44, 0.90, 0.68, 3, [(0, 0), (1, 1), (2, 0), (1, 2)])
    arrow(p, 0.405, 0.50, 0.595, 0.50, W_MAIN, hl=0.085, hw=0.075)


def convert_02(p):
    """Z stack in, Yokogawa well-plate layout out."""
    zstack(p, 0.215, 0.165, 0.172, 4, 0.185, 0.048, W_THIN)
    arrow(p, 0.435, 0.50, 0.565, 0.50, W_THIN, hl=0.075, hw=0.068)
    stroke(p, W_MAIN)
    rrect(p, 0.59, 0.245, 0.98, 0.755, 0.035)
    for i, x in enumerate((0.685, 0.785, 0.885)):
        for j, y in enumerate((0.375, 0.500, 0.625)):
            if (i + j) % 2 == 0:
                dot(p, x, y, 0.043)
            else:
                ring(p, x, y, 0.043, W_THIN)


def convert_03(p):
    """Channel split: one composite frame becomes separate channel images."""
    tile(p, 0.03, 0.31, 0.39, 0.69, W_MAIN, 0.035)
    stroke(p, W_THIN)
    circ(p, 0.16, 0.44, 0.075)
    circ(p, 0.27, 0.53, 0.075)
    ell(p, 0.19, 0.60, 0.085, 0.048, -25)
    arrow(p, 0.425, 0.50, 0.555, 0.50, W_THIN, hl=0.075, hw=0.068)
    for k, (y0, y1) in enumerate([(0.05, 0.31), (0.37, 0.63), (0.69, 0.95)]):
        tile(p, 0.60, y0, 0.96, y1, W_THIN, 0.03)
        cy = (y0 + y1) / 2
        if k == 0:
            dot(p, 0.78, cy, 0.070)
        elif k == 1:
            ring(p, 0.78, cy, 0.070, W_THIN)
        else:
            lens(p, 0.78, cy, 0.095, 0.048, -20)


def convert_04(p):
    """Same field, new representation: smooth on one side, pixels on the other."""
    stroke(p, W_MAIN)
    rrect(p, 0.06, 0.06, 0.94, 0.94, 0.10)
    stroke(p, W_THIN)
    ln(p, 0.50, 0.06, 0.50, 0.94)
    p.save()
    clip = QPainterPath()
    clip.addRect(QRectF(0.06, 0.06, 0.435, 0.88))
    p.setClipPath(clip)
    stroke(p, W_THIN)
    circ(p, 0.50, 0.50, 0.285)
    dot(p, 0.40, 0.45, 0.090)
    stroke(p, W_HAIR)
    ell(p, 0.36, 0.63, 0.070, 0.038, 25)
    p.restore()
    solid(p)
    step = 0.0712
    for i in range(5):
        for j in range(9):
            cx = 0.513 + (i + 0.5) * step
            cy = 0.50 + (j - 4) * step
            if math.hypot(cx - 0.50, cy - 0.50) < 0.285:
                p.drawRect(QRectF(cx - step / 2 + 0.006, cy - step / 2 + 0.006,
                                  step - 0.012, step - 0.012))


def convert_05(p):
    """Funnel: assorted vendor formats in, uniform tiles out."""
    stroke(p, W_MAIN)
    poly(p, [(0.04, 0.07), (0.96, 0.07), (0.585, 0.46), (0.585, 0.62),
             (0.415, 0.62), (0.415, 0.46)])
    stroke(p, W_THIN)
    circ(p, 0.300, 0.225, 0.072)
    lens(p, 0.50, 0.190, 0.098, 0.048, -15)
    stroke(p, W_THIN)
    circ(p, 0.700, 0.230, 0.058)
    dot(p, 0.700, 0.230, 0.024)
    for cx in (0.195, 0.50, 0.805):
        stroke(p, W_THIN)
        rrect(p, cx - 0.105, 0.745, cx + 0.105, 0.955, 0.03)
        dot(p, cx, 0.85, 0.036)


def convert_06(p):
    """A machine on the line: images enter raw and leave as tiles."""
    gear(p, 0.50, 0.185, 0.205, 0.155, 9, W_MAIN)
    ring(p, 0.50, 0.185, 0.070, W_THIN)
    stroke(p, W_THIN)
    ln(p, 0.50, 0.395, 0.50, 0.475)
    tile(p, 0.02, 0.49, 0.34, 0.81, W_MAIN, 0.035)
    stroke(p, W_HAIR)
    for y in (0.565, 0.650, 0.735):
        ell(p, 0.18, y, 0.088, 0.026)
    tile(p, 0.66, 0.49, 0.98, 0.81, W_MAIN, 0.035)
    pixel_grid(p, 0.705, 0.535, 0.935, 0.765, 3, [(0, 0), (1, 1), (2, 2), (0, 2)])
    arrow(p, 0.375, 0.65, 0.625, 0.65, W_THIN, hl=0.075, hw=0.068)
    stroke(p, W_THIN)
    ln(p, 0.02, 0.93, 0.98, 0.93)


def convert_07(p):
    """Format swap between two microscope slides."""
    for y0, kind in ((0.06, "raw"), (0.72, "tif")):
        stroke(p, W_MAIN)
        rrect(p, 0.04, y0, 0.96, y0 + 0.22, 0.045)
        stroke(p, W_THIN)
        ln(p, 0.26, y0 + 0.015, 0.26, y0 + 0.205)
        for hx in (0.09, 0.135, 0.18, 0.225):
            ln(p, hx, y0 + 0.045, hx, y0 + 0.175)
        if kind == "raw":
            lens(p, 0.45, y0 + 0.11, 0.085, 0.042, -18)
            lens(p, 0.64, y0 + 0.10, 0.060, 0.034, 22)
            lens(p, 0.81, y0 + 0.12, 0.070, 0.038, -8)
        else:
            for cx, fills in ((0.45, [(0, 0), (1, 1)]),
                              (0.64, [(1, 0), (0, 1)]),
                              (0.81, [(0, 0), (1, 1)])):
                pixel_grid(p, cx - 0.072, y0 + 0.038, cx + 0.072, y0 + 0.182, 2,
                           fills)
    arrow(p, 0.33, 0.36, 0.33, 0.66, W_MAIN, hl=0.095, hw=0.085)
    arrow(p, 0.67, 0.66, 0.67, 0.36, W_MAIN, hl=0.095, hw=0.085)


def convert_08(p):
    """Filmstrip of frames laid out as a contact grid."""
    stroke(p, W_MAIN)
    rrect(p, 0.05, 0.05, 0.37, 0.95, 0.03)
    stroke(p, W_HAIR)
    for y in (0.275, 0.50, 0.725):
        ln(p, 0.05, y, 0.37, y)
    solid(p)
    for y in (0.105, 0.245, 0.385, 0.525, 0.665, 0.805):
        for x in (0.085, 0.305):
            p.drawRoundedRect(QRectF(x, y, 0.038, 0.055), 0.010, 0.010)
    arrow(p, 0.415, 0.50, 0.545, 0.50, W_THIN, hl=0.075, hw=0.068)
    for i, x in enumerate((0.595, 0.715, 0.835)):
        for j, y in enumerate((0.325, 0.445, 0.565)):
            tile(p, x, y, x + 0.105, y + 0.105, W_THIN, 0.018)
            if (i + j) % 2 == 0:
                dot(p, x + 0.0525, y + 0.0525, 0.024)


def convert_09(p):
    """Re-encode in place: the image cycles through a new format."""
    tile(p, 0.29, 0.29, 0.71, 0.71, W_MAIN, 0.05)
    cell(p, 0.50, 0.50, 0.135, W_THIN, nuc=0.38, organelles=3)
    arc_arrow(p, 0.50, 0.50, 0.385, 168, 22, W_MAIN, hl=0.11, hw=0.095)
    arc_arrow(p, 0.50, 0.50, 0.385, -12, -158, W_MAIN, hl=0.11, hw=0.095)


def convert_10(p):
    """One container unpacked into many single images."""
    stroke(p, W_MAIN)
    rrect(p, 0.04, 0.55, 0.42, 0.95, 0.035)
    ln(p, 0.04, 0.665, 0.42, 0.665)
    stroke(p, W_THIN)
    rrect(p, 0.175, 0.585, 0.285, 0.645, 0.018)
    centers = [(0.335, 0.455), (0.505, 0.355), (0.675, 0.255), (0.845, 0.155)]
    for k, (cx, cy) in enumerate(centers):
        h = 0.095
        occlude(p, lambda q, cx=cx, cy=cy, h=h: q.drawRoundedRect(
            QRectF(cx - h, cy - h, 2 * h, 2 * h), 0.022, 0.022))
        tile(p, cx - h, cy - h, cx + h, cy + h, W_THIN, 0.022)
        if k == len(centers) - 1:
            pixel_grid(p, cx - 0.055, cy - 0.055, cx + 0.055, cy + 0.055, 2,
                       [(0, 0), (1, 1)])
        else:
            dot(p, cx, cy, 0.030)


# ======================================================================================
# DEFAULT - generic-but-deliberate fallback
# ======================================================================================


def default_01(p):
    """A cell in a frame: the plainest spaCR object there is."""
    stroke(p, W_MAIN)
    rrect(p, 0.06, 0.06, 0.94, 0.94, 0.16)
    cell(p, 0.50, 0.50, 0.265, W_THIN, nuc=0.34, organelles=4, seed_ang=0.8)


def default_02(p):
    """An empty dish waiting for something."""
    stroke(p, W_MAIN)
    circ(p, 0.50, 0.50, 0.445)
    stroke(p, W_THIN)
    circ(p, 0.50, 0.50, 0.380)
    cell(p, 0.50, 0.50, 0.225, W_THIN, nuc=0.34, organelles=4, seed_ang=0.9)


def default_03(p):
    """A well plate: the unit of a spaCR experiment."""
    stroke(p, W_MAIN)
    rrect(p, 0.05, 0.13, 0.95, 0.87, 0.055)
    for i, x in enumerate((0.26, 0.50, 0.74)):
        for j, y in enumerate((0.29, 0.50, 0.71)):
            if i == 1 and j == 1:
                dot(p, x, y, 0.088)
            else:
                ring(p, x, y, 0.088, W_THIN)


def default_04(p):
    """A module: a hexagon plugged into the pipeline."""
    R = 0.44
    pts = [(0.5 + R * math.cos(math.radians(a)), 0.5 - R * math.sin(math.radians(a)))
           for a in range(0, 360, 60)]
    stroke(p, W_MAIN)
    poly(p, pts)
    r2 = 0.235
    pts2 = [(0.5 + r2 * math.cos(math.radians(a)), 0.5 - r2 * math.sin(math.radians(a)))
            for a in range(0, 360, 60)]
    stroke(p, W_THIN)
    poly(p, pts2)
    dot(p, 0.50, 0.50, 0.095)


def default_05(p):
    """An objective iris: looking at something, subject unspecified."""
    R, ri = 0.445, 0.155
    stroke(p, W_MAIN)
    circ(p, 0.50, 0.50, R)
    stroke(p, W_THIN)
    for i in range(6):
        a = math.radians(i * 60)
        b = math.radians(i * 60 + 60)
        ln(p, 0.5 + R * math.cos(a), 0.5 - R * math.sin(a),
           0.5 + ri * math.cos(b), 0.5 - ri * math.sin(b))
    pts = [(0.5 + ri * math.cos(math.radians(a)), 0.5 - ri * math.sin(math.radians(a)))
           for a in range(0, 360, 60)]
    stroke(p, W_THIN)
    poly(p, pts)


def default_06(p):
    """A piece that fits: any tool, slot not yet named."""
    x0, y0, x1, y1 = 0.08, 0.10, 0.86, 0.88
    t = 0.130
    ym, xm = (y0 + y1) / 2, (x0 + x1) / 2
    path = QPainterPath(QPointF(x0, y0))
    path.lineTo(x1, y0)
    path.lineTo(x1, ym - t)
    path.arcTo(QRectF(x1 - t, ym - t, 2 * t, 2 * t), 90, -180)
    path.lineTo(x1, y1)
    path.lineTo(xm + t, y1)
    path.arcTo(QRectF(xm - t, y1 - t, 2 * t, 2 * t), 0, 180)
    path.lineTo(x0, y1)
    path.closeSubpath()
    stroke(p, W_MAIN)
    p.drawPath(path)
    cell(p, 0.455, 0.475, 0.180, W_THIN, nuc=0.36, organelles=3)


def default_07(p):
    """A step in a graph: something runs, contents unspecified."""
    stroke(p, W_THIN)
    ln(p, 0.22, 0.44, 0.42, 0.20)
    ln(p, 0.22, 0.56, 0.42, 0.80)
    ln(p, 0.58, 0.20, 0.78, 0.44)
    ln(p, 0.58, 0.80, 0.78, 0.56)
    dot(p, 0.13, 0.50, 0.115)
    ring(p, 0.50, 0.13, 0.115, W_MAIN)
    ring(p, 0.50, 0.87, 0.115, W_MAIN)
    ring(p, 0.87, 0.50, 0.115, W_MAIN)


def default_08(p):
    """A generic engine: gear turning around a nucleus."""
    gear(p, 0.50, 0.50, 0.455, 0.360, 10, W_MAIN)
    stroke(p, W_THIN)
    circ(p, 0.50, 0.50, 0.185)
    dot(p, 0.50, 0.50, 0.085)


def default_09(p):
    """A target: focus on the field, no particular assay."""
    stroke(p, W_MAIN)
    circ(p, 0.50, 0.50, 0.375)
    stroke(p, W_THIN)
    circ(p, 0.50, 0.50, 0.205)
    stroke(p, W_MAIN)
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        ln(p, 0.5 + dx * 0.475, 0.5 + dy * 0.475, 0.5 + dx * 0.275, 0.5 + dy * 0.275)
    dot(p, 0.50, 0.50, 0.078)


def default_10(p):
    """A stack of layers: some content, kind unknown."""
    for k, (ox, oy) in enumerate(((0.26, 0.16), (0.16, 0.32), (0.06, 0.48))):
        x0, y0, x1, y1 = ox, oy, ox + 0.68, oy + 0.36
        occlude(p, lambda q, a=x0, b=y0, c=x1, d=y1: q.drawRoundedRect(
            QRectF(a, b, c - a, d - b), 0.05, 0.05))
        stroke(p, W_MAIN)
        rrect(p, x0, y0, x1, y1, 0.05)
        if k == 2:
            dot(p, x0 + 0.11, y0 + 0.18, 0.055)
            stroke(p, W_THIN)
            ln(p, x0 + 0.24, y0 + 0.13, x0 + 0.58, y0 + 0.13)
            ln(p, x0 + 0.24, y0 + 0.24, x0 + 0.46, y0 + 0.24)


# ======================================================================================
# DOWNLOAD - fetching a model or a dataset
# ======================================================================================


def download_01(p):
    """From the cloud into your dish."""
    cloud = blob_union(
        circles=((0.315, 0.215, 0.120), (0.505, 0.155, 0.155),
                 (0.685, 0.225, 0.108)),
        rects=((0.235, 0.215, 0.775, 0.335, 0.055),),
    )
    stroke(p, W_MAIN)
    p.drawPath(cloud)
    arrow(p, 0.50, 0.395, 0.50, 0.625, W_MAIN, hl=0.105, hw=0.125)
    bowl(p, 0.50, 0.695, 0.395, 0.27, W_MAIN)


def download_02(p):
    """Arrow into the tray: the plain, unmistakable download."""
    stroke(p, W_MAIN)
    poly(p, [(0.09, 0.58), (0.09, 0.90), (0.91, 0.90), (0.91, 0.58)], close=False)
    arrow(p, 0.50, 0.08, 0.50, 0.71, W_MAIN, hl=0.20, hw=0.34)


def download_03(p):
    """Fetching model weights: the network file arrives."""
    stroke(p, W_MAIN)
    rrect(p, 0.12, 0.03, 0.88, 0.505, 0.055)
    stroke(p, W_HAIR)
    for y1 in (0.135, 0.268, 0.400):
        for y2 in (0.135, 0.268, 0.400):
            ln(p, 0.325, y1, 0.675, y2)
    for x in (0.30, 0.70):
        for y in (0.135, 0.268, 0.400):
            dot(p, x, y, 0.038)
    arrow(p, 0.50, 0.565, 0.50, 0.845, W_MAIN, hl=0.135, hw=0.195)
    stroke(p, W_MAIN)
    ln(p, 0.22, 0.935, 0.78, 0.935)


def download_04(p):
    """Pulled out of the repository, one object at a time."""
    rx, ry = 0.315, 0.100
    stroke(p, W_MAIN)
    ell(p, 0.50, 0.145, rx, ry)
    ln(p, 0.50 - rx, 0.145, 0.50 - rx, 0.395)
    ln(p, 0.50 + rx, 0.145, 0.50 + rx, 0.395)
    poly(p, arc_pts(0.50, 0.395, rx, 180, 360, 60, ry=ry), close=False)
    stroke(p, W_THIN)
    poly(p, arc_pts(0.50, 0.270, rx, 180, 360, 60, ry=ry), close=False)
    arrow(p, 0.50, 0.545, 0.50, 0.685, W_MAIN, hl=0.090, hw=0.110)
    cell(p, 0.50, 0.845, 0.150, W_MAIN, nuc=0.42)


def download_05(p):
    """In flight: the transfer ring closing round the arrow."""
    arc(p, 0.50, 0.50, 0.435, 118, -212, W_MAIN)
    arrow(p, 0.50, 0.235, 0.50, 0.735, W_MAIN, hl=0.155, hw=0.245)


def download_06(p):
    """A dataset coming down into local storage."""
    for x in (0.05, 0.37, 0.69):
        tile(p, x, 0.03, x + 0.26, 0.24, W_THIN, 0.028)
        dot(p, x + 0.13, 0.135, 0.048)
    arrow(p, 0.50, 0.315, 0.50, 0.585, W_MAIN, hl=0.125, hw=0.175)
    stroke(p, W_MAIN)
    poly(p, [(0.05, 0.955), (0.05, 0.635), (0.31, 0.635), (0.375, 0.715),
             (0.95, 0.715), (0.95, 0.955)])


def download_07(p):
    """The z stack itself descending into the workspace."""
    zstack(p, 0.50, 0.115, 0.180, 3, 0.300, 0.050, W_THIN)
    arrow(p, 0.50, 0.545, 0.50, 0.955, W_MAIN, hl=0.195, hw=0.315)


def download_08(p):
    """Landing on disk."""
    arrow(p, 0.50, 0.025, 0.50, 0.295, W_MAIN, hl=0.130, hw=0.175)
    stroke(p, W_MAIN)
    rrect(p, 0.04, 0.375, 0.96, 0.875, 0.06)
    stroke(p, W_THIN)
    circ(p, 0.355, 0.625, 0.170)
    dot(p, 0.355, 0.625, 0.050)
    stroke(p, W_THIN)
    poly(p, [(0.875, 0.455), (0.715, 0.520), (0.545, 0.565)], close=False)
    dot(p, 0.875, 0.455, 0.034)
    dot(p, 0.845, 0.800, 0.036)
    stroke(p, W_THIN)
    ln(p, 0.645, 0.800, 0.755, 0.800)


def download_09(p):
    """Coming down, straight into the dish."""
    stroke(p, W_MAIN)
    for y in (0.07, 0.28):
        poly(p, [(0.27, y), (0.50, y + 0.185), (0.73, y)], close=False)
    bowl(p, 0.50, 0.635, 0.42, 0.30, W_MAIN)


def download_10(p):
    """Delivered: the data dispensed into your workspace."""
    stroke(p, W_MAIN)
    poly(p, [(0.365, 0.02), (0.635, 0.02), (0.590, 0.335), (0.548, 0.455),
             (0.50, 0.535), (0.452, 0.455), (0.410, 0.335)])
    stroke(p, W_THIN)
    ln(p, 0.378, 0.130, 0.622, 0.130)
    ln(p, 0.392, 0.235, 0.608, 0.235)
    solid(p)
    path = QPainterPath(QPointF(0.50, 0.585))
    path.cubicTo(QPointF(0.572, 0.665), QPointF(0.562, 0.735),
                 QPointF(0.50, 0.735))
    path.cubicTo(QPointF(0.438, 0.735), QPointF(0.428, 0.665),
                 QPointF(0.50, 0.585))
    p.drawPath(path)
    bowl(p, 0.50, 0.790, 0.395, 0.185, W_MAIN)


# ======================================================================================
# concept text + registry
# ======================================================================================

GROUPS = {
    "classify": {
        "blurb": "a trained model sorting objects into classes",
        "items": [
            (classify_01, "Decision boundary - two populations split by the learned line, on plot axes"),
            (classify_02, "Sorting bins - a mixed population diverges into two labelled containers"),
            (classify_03, "Decision tree - one cell at the root branching down to leaf classes"),
            (classify_04, "Confusion matrix - predicted vs true grid with the diagonal lit, cells as row/col labels"),
            (classify_05, "Trained network - a cell feeds a node graph and one output class lights up"),
            (classify_06, "Microfluidic sorter - a Y channel routing each object down its own arm"),
            (classify_07, "Sieve - a mesh separating the population by size, large above and small below"),
            (classify_08, "Class label - a cell wearing an assigned tag"),
            (classify_09, "Class counts - a histogram whose bars are stacks of cells, one glyph per class"),
            (classify_10, "Class scores - the model's per-class confidence bars beside the object"),
        ],
    },
    "convert": {
        "blurb": "image format conversion, nd2/czi/lif to Yokogawa TIFF",
        "items": [
            (convert_01, "Document to document - a stack-of-slices file becomes a pixel-grid file"),
            (convert_02, "Z stack to plate - isometric slices reorganised into a Yokogawa well layout"),
            (convert_03, "Channel split - one composite frame demultiplexed into per-channel images"),
            (convert_04, "Same field, new representation - one frame, smooth on the left, pixels on the right"),
            (convert_05, "Funnel - assorted vendor shapes poured in, uniform tiles out the spout"),
            (convert_06, "Machine on the line - raw frames enter, a gear drives them, tiles leave"),
            (convert_07, "Slide swap - two microscope slides, specimen and pixel grid, exchanged by arrows"),
            (convert_08, "Filmstrip to contact grid - a sequential strip laid out as an indexed tile grid"),
            (convert_09, "Re-encode in place - cycle arrows around a single framed image"),
            (convert_10, "Unpack - one container file opening into a fan of individual images"),
        ],
    },
    "default": {
        "blurb": "fallback icon: generic-but-deliberate, never a broken image",
        "items": [
            (default_01, "Cell in a frame - the plainest spaCR object, deliberately framed"),
            (default_02, "Empty dish - a prepared but unassigned experiment"),
            (default_03, "Well plate - the unit of a spaCR run, nothing selected"),
            (default_04, "Hexagon module - a plug-in slot with an unnamed payload"),
            (default_05, "Objective iris - looking at something, subject unspecified"),
            (default_06, "Puzzle piece - a tool that fits, purpose not yet declared"),
            (default_07, "Pipeline node - a step in a graph, contents unspecified"),
            (default_08, "Gear with a nucleus - a generic engine, biology at the hub"),
            (default_09, "Target - focus on the field, no particular assay"),
            (default_10, "Layer stack - some content of an unknown kind"),
        ],
    },
    "download": {
        "blurb": "fetching a model or a dataset",
        "items": [
            (download_01, "Cloud to dish - remote asset landing in the local experiment"),
            (download_02, "Arrow into tray - the plain, unmistakable download"),
            (download_03, "Model weights - a card carrying a node graph arriving on the baseline"),
            (download_04, "Repository cylinder - an object pulled out of a remote store"),
            (download_05, "Transfer ring - in-flight progress arc closing round the arrow"),
            (download_06, "Dataset tiles - images descending out of a remote grid into local storage"),
            (download_07, "Descending z stack - the arrow body is the acquisition itself"),
            (download_08, "Landing on disk - arrow into a drive with a platter and head"),
            (download_09, "Double chevron into a dish - minimal, straight into the workspace"),
            (download_10, "Pipette - the data dispensed as a drop into your workspace"),
        ],
    },
}


# ======================================================================================
# rendering
# ======================================================================================


def render(fn, size=SIZE):
    img = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
    img.fill(QColor(0, 0, 0, 0))
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setRenderHint(QPainter.SmoothPixmapTransform, True)
    p.scale(size, size)
    fn(p)
    p.end()
    return img.convertToFormat(QImage.Format_ARGB32)


def coverage(img):
    """(fraction alpha>0, fraction alpha>=128)."""
    w, h = img.width(), img.height()
    bits = img.constBits()
    buf = bytes(bits)
    stride = img.bytesPerLine()
    any_a = 0
    op_a = 0
    for y in range(h):
        row = buf[y * stride: y * stride + w * 4]
        alphas = row[3::4]
        any_a += sum(1 for a in alphas if a > 0)
        op_a += sum(1 for a in alphas if a >= 128)
    n = w * h
    return any_a / n, op_a / n


def contact_sheet(images, labels, bg, cell_px=300, cols=5, pad=22, label_px=34):
    rows = (len(images) + cols - 1) // cols
    W = cols * cell_px + (cols + 1) * pad
    H = rows * (cell_px + label_px) + (rows + 1) * pad
    sheet = QImage(W, H, QImage.Format_ARGB32_Premultiplied)
    sheet.fill(bg)
    p = QPainter(sheet)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setRenderHint(QPainter.SmoothPixmapTransform, True)
    ink = QColor(0xf0, 0xf2, 0xf5) if bg == SHEET_BG_DARK else QColor(0x1a, 0x1c, 0x20)
    f = QFont()
    f.setPointSizeF(15)
    f.setBold(True)
    p.setFont(f)
    for i, (im, lab) in enumerate(zip(images, labels)):
        c, r = i % cols, i // cols
        x = pad + c * (cell_px + pad)
        y = pad + r * (cell_px + label_px + pad)
        # the artwork is white-on-transparent; on a light sheet the viewer needs to
        # see it, so tint a copy to the sheet's ink colour instead of dropping it.
        draw = im.scaled(cell_px, cell_px, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        if bg == SHEET_BG_LIGHT:
            tinted = QImage(draw.size(), QImage.Format_ARGB32_Premultiplied)
            tinted.fill(QColor(0, 0, 0, 0))
            tp = QPainter(tinted)
            tp.drawImage(0, 0, draw)
            tp.setCompositionMode(QPainter.CompositionMode_SourceIn)
            tp.fillRect(tinted.rect(), ink)
            tp.end()
            draw = tinted
        p.drawImage(x, y, draw)
        p.setPen(QPen(ink))
        p.drawText(QRectF(x, y + cell_px + 2, cell_px, label_px),
                   Qt.AlignHCenter | Qt.AlignVCenter, lab)
    p.end()
    return sheet.convertToFormat(QImage.Format_ARGB32)


def main(outroot=None):
    here = os.path.dirname(os.path.abspath(__file__))
    outroot = outroot or os.path.abspath(os.path.join(here, ".."))
    report = []
    for name, spec in GROUPS.items():
        d = os.path.join(outroot, name)
        os.makedirs(d, exist_ok=True)
        imgs, labels = [], []
        lines = [f"# {name} - candidate concepts",
                 "",
                 f"_{spec['blurb']}_",
                 "",
                 "Ten conceptually different metaphors for the same idea.",
                 "",
                 "The PNGs are pure white on transparent, matching `plaque.png` and",
                 "`measure.png`. `_sheet_light.png` shows the same artwork tinted dark so",
                 "the shapes can be judged on a light background - the shipped white files",
                 "would still be invisible there, which is the existing open bug, not",
                 "something these candidates fix on their own.",
                 ""]
        for i, (fn, concept) in enumerate(spec["items"], start=1):
            img = render(fn)
            path = os.path.join(d, f"{name}_{i:02d}.png")
            img.save(path)
            a_any, a_op = coverage(img)
            report.append((f"{name}_{i:02d}.png", img.width(), img.height(),
                           a_any, a_op))
            imgs.append(img)
            labels.append(f"{i:02d}")
            lines.append(f"{i}. **{concept}**")
        lines.append("")
        with open(os.path.join(d, "CONCEPTS.md"), "w") as fh:
            fh.write("\n".join(lines))
        contact_sheet(imgs, labels, SHEET_BG_DARK).save(
            os.path.join(d, "_sheet_dark.png"))
        contact_sheet(imgs, labels, SHEET_BG_LIGHT).save(
            os.path.join(d, "_sheet_light.png"))
    print(f"{'file':<20} {'size':>10} {'alpha>0':>9} {'alpha>=128':>11}")
    bad = 0
    for f, w, h, a0, a1 in report:
        flag = "" if 0.05 <= a1 <= 0.70 else "   <-- OUT OF RANGE"
        if flag:
            bad += 1
        print(f"{f:<20} {w}x{h:>5} {a0*100:8.2f}% {a1*100:10.2f}%{flag}")
    print(f"\n{len(report)} files, {bad} outside the 5-70% opaque band")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
