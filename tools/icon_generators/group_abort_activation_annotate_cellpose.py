#!/usr/bin/env python3
"""Candidate icon generator for spaCR -- group: abort, activation, annotate,
cellpose_all, cellpose_masks.

House style (measured from resources/icons/plaque.png and measure.png):
  * pure white artwork on a fully transparent background (alpha carries the shape)
  * flat: no gradients, no shading, no colour
  * a mix of thin outlined strokes and solid white fills
  * square canvas, subject fills most of the frame, modest margin
  * literal but stylised biology / lab objects, not abstract glyphs

Everything is drawn with QPainter in a normalised 0..1 coordinate space and
scaled to the output canvas, so any SIZE renders correctly.  Every random
element is seeded, so repeated runs are byte-identical.

Run standalone:

    QT_QPA_PLATFORM=offscreen python3 group_abort_activation_annotate_cellpose.py

Writes into  <repo>/spacr/resources/icons/backup_icons/<name>/  ...
    <name>_01.png .. <name>_10.png   (SIZE x SIZE RGBA, white on transparent)
    CONCEPTS.md
    _sheet_dark.png   /  _sheet_light.png

Nothing outside backup_icons/ is touched.
"""

import math
import os
import random
import sys

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QFont, QGuiApplication, QImage,
                           QPainter, QPainterPath, QPainterPathStroker, QPen,
                           QTransform)

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

SIZE = 1024
W_MAIN = 0.030          # main outline stroke, normalised units
W_THIN = 0.021          # secondary detail stroke
W_HAIR = 0.015          # fine detail stroke
WHITE = QColor(255, 255, 255)
WBRUSH = QBrush(WHITE)

HERE = os.path.dirname(os.path.abspath(__file__))
OUTROOT = os.path.abspath(os.path.join(HERE, os.pardir))    # .../backup_icons


# --------------------------------------------------------------------------
# low level helpers
# --------------------------------------------------------------------------

def pen(w=W_MAIN, dash=None):
    p = QPen(WHITE, w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    if dash:
        p.setStyle(Qt.CustomDashLine)
        p.setDashPattern(dash)
        p.setCapStyle(Qt.FlatCap)
    return p


def stroke_only(p, w=W_MAIN, dash=None):
    p.setPen(pen(w, dash))
    p.setBrush(Qt.NoBrush)


def fill_only(p):
    p.setPen(Qt.NoPen)
    p.setBrush(WBRUSH)


def erase_stroke(p, path, w):
    """Punch a transparent band along `path` -- used to separate overlapping art."""
    p.save()
    p.setCompositionMode(QPainter.CompositionMode_Clear)
    p.setPen(QPen(QColor(0, 0, 0, 255), w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    p.setBrush(Qt.NoBrush)
    p.drawPath(path)
    p.restore()


def erase_fill(p, path):
    """Punch a transparent hole shaped like `path`."""
    p.save()
    p.setCompositionMode(QPainter.CompositionMode_Clear)
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(QColor(0, 0, 0, 255)))
    p.drawPath(path)
    p.restore()


def occlude(p, path, gap=0.030):
    """Hide whatever is behind `path`, leaving a clean gap around its edge."""
    erase_fill(p, path)
    erase_stroke(p, path, gap)


def outline_of(path, w, cap=Qt.RoundCap):
    st = QPainterPathStroker()
    st.setWidth(w)
    st.setCapStyle(cap)
    st.setJoinStyle(Qt.RoundJoin)
    return st.createStroke(path)


def line_path(pts):
    path = QPainterPath()
    path.moveTo(pts[0][0], pts[0][1])
    for x, y in pts[1:]:
        path.lineTo(x, y)
    return path


def poly_path(pts, close=True):
    path = line_path(pts)
    if close:
        path.closeSubpath()
    return path


def smooth_closed(pts):
    """Closed Catmull-Rom spline through pts, as cubic bezier segments."""
    path = QPainterPath()
    n = len(pts)
    path.moveTo(pts[0][0], pts[0][1])
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


def blob(cx, cy, r, seed=1, n=9, jag=0.16, sx=1.0, sy=1.0, rot=0.0):
    """A stylised amoeboid cell outline: circle with seeded radial jitter."""
    rnd = random.Random(seed)
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n + rot
        rr = r * (1.0 + rnd.uniform(-jag, jag))
        pts.append((cx + rr * math.cos(a) * sx, cy + rr * math.sin(a) * sy))
    return smooth_closed(pts)


def cell(p, cx, cy, r, seed=1, w=W_MAIN, nucleus=True, nr=0.34, jag=0.15, n=9,
         rot=0.0):
    b = blob(cx, cy, r, seed, n=n, jag=jag, rot=rot)
    stroke_only(p, w)
    p.drawPath(b)
    if nucleus:
        fill_only(p)
        p.drawEllipse(QPointF(cx, cy), r * nr, r * nr)
    return b


def dot(p, x, y, r):
    fill_only(p)
    p.drawEllipse(QPointF(x, y), r, r)


def ring(p, x, y, r, w=W_THIN):
    stroke_only(p, w)
    p.drawEllipse(QPointF(x, y), r, r)


def arrow(p, x1, y1, x2, y2, w=W_MAIN, head=0.07, spread=28.0):
    stroke_only(p, w)
    p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
    a = math.atan2(y2 - y1, x2 - x1)
    for s in (1, -1):
        b = a + s * math.radians(180 - spread)
        p.drawLine(QPointF(x2, y2),
                   QPointF(x2 + head * math.cos(b), y2 + head * math.sin(b)))


def solid_arrow(p, x1, y1, x2, y2, w=0.036, head=0.10, hw=0.055):
    """Thick line with a solid triangular head -- reads at 48 px."""
    a = math.atan2(y2 - y1, x2 - x1)
    bx, by = x2 - head * math.cos(a), y2 - head * math.sin(a)
    stroke_only(p, w)
    p.drawLine(QPointF(x1, y1), QPointF(bx, by))
    nx, ny = -math.sin(a), math.cos(a)
    fill_only(p)
    p.drawPath(poly_path([(x2, y2), (bx + nx * hw, by + ny * hw),
                          (bx - nx * hw, by - ny * hw)]))


def tick(p, cx, cy, s, w=0.042):
    stroke_only(p, w)
    p.drawPath(line_path([(cx - s, cy + 0.03 * s / 0.15),
                          (cx - 0.25 * s, cy + 0.72 * s),
                          (cx + s, cy - 0.75 * s)]))


def cross(p, cx, cy, s, w=0.042):
    stroke_only(p, w)
    p.drawLine(QPointF(cx - s, cy - s), QPointF(cx + s, cy + s))
    p.drawLine(QPointF(cx + s, cy - s), QPointF(cx - s, cy + s))


def rrect(p, x0, y0, x1, y1, rad=0.05, w=W_MAIN, filled=False):
    r = QRectF(x0, y0, x1 - x0, y1 - y0)
    if filled:
        fill_only(p)
    else:
        stroke_only(p, w)
    p.drawRoundedRect(r, rad, rad)
    return r


def text_path(s, h, cx, cy, bold=True):
    f = QFont("DejaVu Sans")
    f.setPixelSize(200)
    f.setBold(bold)
    path = QPainterPath()
    path.addText(0.0, 0.0, f, s)
    br = path.boundingRect()
    if br.height() <= 0:
        return path
    k = h / br.height()
    t = QTransform()
    t.translate(cx, cy)
    t.scale(k, k)
    t.translate(-br.center().x(), -br.center().y())
    return t.map(path)


def speckle(p, path, seed, count=26, rmin=0.010, rmax=0.020):
    """Seeded dot texture clipped inside `path` -- stands in for raw signal."""
    rnd = random.Random(seed)
    br = path.boundingRect()
    p.save()
    p.setClipPath(path)
    fill_only(p)
    n = 0
    guard = 0
    while n < count and guard < count * 40:
        guard += 1
        x = br.x() + rnd.random() * br.width()
        y = br.y() + rnd.random() * br.height()
        if not path.contains(QPointF(x, y)):
            continue
        p.drawEllipse(QPointF(x, y), rnd.uniform(rmin, rmax),
                      rnd.uniform(rmin, rmax))
        n += 1
    p.restore()


# ==========================================================================
# ABORT -- stop a running pipeline
# ==========================================================================

def abort_01(p):
    """Octagonal stop sign with a solid square core."""
    r = 0.44
    pts = [(0.5 + r * math.cos(math.pi / 8 + 2 * math.pi * i / 8),
            0.5 + r * math.sin(math.pi / 8 + 2 * math.pi * i / 8)) for i in range(8)]
    stroke_only(p, 0.034)
    p.drawPath(poly_path(pts))
    fill_only(p)
    p.drawRoundedRect(QRectF(0.355, 0.355, 0.29, 0.29), 0.035, 0.035)


def abort_02(p):
    """Raised open palm (halt) in front of a petri dish."""
    stroke_only(p, W_MAIN)
    p.drawEllipse(QPointF(0.5, 0.5), 0.43, 0.43)
    p.drawEllipse(QPointF(0.5, 0.5), 0.375, 0.375)

    hand = QPainterPath()
    hand.addRoundedRect(QRectF(0.335, 0.44, 0.30, 0.36), 0.07, 0.07)
    tops = [0.255, 0.205, 0.215, 0.275]
    xs = [0.375, 0.445, 0.515, 0.585]
    for x, t in zip(xs, tops):
        hand = hand.united(outline_of(line_path([(x, 0.52), (x, t)]), 0.062))
    hand = hand.united(outline_of(line_path([(0.365, 0.62), (0.255, 0.47)]), 0.062))
    erase_stroke(p, hand, 0.052)
    fill_only(p)
    p.drawPath(hand.simplified())


def abort_03(p):
    """Emergency-stop mushroom button, seen from above."""
    stroke_only(p, W_MAIN)
    p.drawEllipse(QPointF(0.5, 0.5), 0.45, 0.45)
    stroke_only(p, W_THIN)
    p.drawEllipse(QPointF(0.5, 0.5), 0.355, 0.355)
    fill_only(p)
    p.drawEllipse(QPointF(0.5, 0.5), 0.235, 0.235)
    stroke_only(p, 0.026)
    for i in range(8):
        a = math.pi / 8 + 2 * math.pi * i / 8
        p.drawLine(QPointF(0.5 + 0.375 * math.cos(a), 0.5 + 0.375 * math.sin(a)),
                   QPointF(0.5 + 0.435 * math.cos(a), 0.5 + 0.435 * math.sin(a)))


def abort_04(p):
    """Plug yanked out of the socket."""
    stroke_only(p, W_MAIN)
    p.drawRoundedRect(QRectF(0.045, 0.235, 0.265, 0.53), 0.065, 0.065)
    fill_only(p)
    p.drawRoundedRect(QRectF(0.195, 0.335, 0.065, 0.115), 0.03, 0.03)
    p.drawRoundedRect(QRectF(0.195, 0.55, 0.065, 0.115), 0.03, 0.03)

    stroke_only(p, W_MAIN)
    p.drawRoundedRect(QRectF(0.585, 0.255, 0.235, 0.49), 0.065, 0.065)
    stroke_only(p, 0.050)
    p.drawLine(QPointF(0.455, 0.385), QPointF(0.585, 0.385))
    p.drawLine(QPointF(0.455, 0.615), QPointF(0.585, 0.615))
    # cord
    cord = QPainterPath()
    cord.moveTo(0.82, 0.50)
    cord.cubicTo(0.955, 0.50, 0.985, 0.30, 0.895, 0.135)
    stroke_only(p, W_THIN)
    p.drawPath(cord)
    # break sparks in the gap
    stroke_only(p, 0.022)
    for a in (-52, 0, 52):
        r = math.radians(a)
        p.drawLine(QPointF(0.385 + 0.045 * math.cos(r), 0.50 + 0.045 * math.sin(r)),
                   QPointF(0.385 + 0.115 * math.cos(r), 0.50 + 0.115 * math.sin(r)))


def abort_05(p):
    """Pipeline arrow slamming into a wall."""
    solid_arrow(p, 0.08, 0.50, 0.545, 0.50, w=0.048, head=0.13, hw=0.095)
    fill_only(p)
    p.drawRoundedRect(QRectF(0.665, 0.14, 0.085, 0.72), 0.04, 0.04)
    stroke_only(p, 0.024)
    for dy in (-0.20, 0.0, 0.20):
        p.drawLine(QPointF(0.60, 0.50 + dy * 0.85), QPointF(0.635, 0.50 + dy))


def abort_06(p):
    """Prohibition sign struck over a cell."""
    cell(p, 0.5, 0.5, 0.255, seed=11, w=0.026, nucleus=False, jag=0.19, n=8)
    dot(p, 0.395, 0.405, 0.072)
    stroke_only(p, W_HAIR)
    p.drawEllipse(QPointF(0.60, 0.605), 0.042, 0.042)
    stroke_only(p, 0.040)
    p.drawEllipse(QPointF(0.5, 0.5), 0.42, 0.42)
    bar = outline_of(line_path([(0.5 - 0.297, 0.5 + 0.297),
                                (0.5 + 0.297, 0.5 - 0.297)]), 0.075)
    erase_stroke(p, bar, 0.030)
    fill_only(p)
    p.drawPath(bar)


def abort_07(p):
    """Progress bar terminated mid-run."""
    stroke_only(p, W_MAIN)
    p.drawRoundedRect(QRectF(0.07, 0.365, 0.86, 0.27), 0.10, 0.10)
    fill_only(p)
    p.drawRoundedRect(QRectF(0.105, 0.40, 0.30, 0.20), 0.075, 0.075)
    x = QPainterPath()
    x = outline_of(line_path([(0.62, 0.39), (0.80, 0.61)]), 0.055)
    x = x.united(outline_of(line_path([(0.80, 0.39), (0.62, 0.61)]), 0.055))
    erase_stroke(p, x, 0.036)
    fill_only(p)
    p.drawPath(x.simplified())


def abort_08(p):
    """Scissors cutting the run line."""
    stroke_only(p, W_THIN, dash=[2.0, 1.9])
    p.drawLine(QPointF(0.04, 0.50), QPointF(0.96, 0.50))
    blades = outline_of(line_path([(0.155, 0.185), (0.925, 0.72)]), 0.038)
    blades = blades.united(outline_of(line_path([(0.155, 0.815), (0.925, 0.28)]), 0.038))
    erase_stroke(p, blades, 0.030)
    fill_only(p)
    p.drawPath(blades.simplified())
    stroke_only(p, 0.032)
    p.drawEllipse(QPointF(0.155, 0.185), 0.095, 0.095)
    p.drawEllipse(QPointF(0.155, 0.815), 0.095, 0.095)
    fill_only(p)
    p.drawEllipse(QPointF(0.54, 0.50), 0.038, 0.038)


def abort_09(p):
    """Power / shutdown symbol."""
    r = QRectF(0.5 - 0.38, 0.5 - 0.38, 0.76, 0.76)
    arc = QPainterPath()
    arc.arcMoveTo(r, 110)
    arc.arcTo(r, 110, 320)
    stroke_only(p, 0.058)
    p.drawPath(arc)
    bar = outline_of(line_path([(0.5, 0.10), (0.5, 0.46)]), 0.058)
    erase_stroke(p, bar, 0.034)
    fill_only(p)
    p.drawPath(bar)


def abort_10(p):
    """Traffic signal with the stop lamp lit."""
    stroke_only(p, W_MAIN)
    p.drawRoundedRect(QRectF(0.29, 0.055, 0.42, 0.76), 0.10, 0.10)
    fill_only(p)
    p.drawEllipse(QPointF(0.50, 0.205), 0.115, 0.115)
    stroke_only(p, W_THIN)
    p.drawEllipse(QPointF(0.50, 0.435), 0.105, 0.105)
    p.drawEllipse(QPointF(0.50, 0.660), 0.105, 0.105)
    stroke_only(p, 0.034)
    p.drawLine(QPointF(0.50, 0.815), QPointF(0.50, 0.945))
    p.drawLine(QPointF(0.375, 0.945), QPointF(0.625, 0.945))


# ==========================================================================
# ACTIVATION -- model attention / Grad-CAM saliency over a cell
# ==========================================================================

HOT = (0.585, 0.415)


def activation_01(p):
    """Cell with concentric saliency contours around a hot spot."""
    cell(p, 0.5, 0.5, 0.42, seed=3, w=W_MAIN, nucleus=False, jag=0.13)
    hx, hy = HOT
    for r in (0.235, 0.165, 0.100):
        stroke_only(p, W_THIN)
        p.drawPath(blob(hx, hy, r, seed=7, n=8, jag=0.15))
    fill_only(p)
    p.drawPath(blob(hx, hy, 0.055, seed=7, n=8, jag=0.15))


def activation_02(p):
    """Saliency rendered as a block heat map inside the cell."""
    b = cell(p, 0.5, 0.5, 0.42, seed=5, w=W_MAIN, nucleus=False, jag=0.12)
    hx, hy = HOT
    p.save()
    p.setClipPath(b)
    fill_only(p)
    step = 0.1175
    y = 0.06
    while y < 0.94:
        x = 0.06
        while x < 0.94:
            cxx, cyy = x + step / 2, y + step / 2
            d = math.hypot(cxx - hx, cyy - hy)
            k = max(0.0, 1.0 - d / 0.40)
            if k > 0.06:
                s = step * 0.86 * (0.28 + 0.72 * k)
                p.drawRoundedRect(QRectF(cxx - s / 2, cyy - s / 2, s, s),
                                  s * 0.22, s * 0.22)
            x += step
        y += step
    p.restore()


def activation_03(p):
    """Spotlight beam picking out one cell."""
    fill_only(p)
    p.drawPath(poly_path([(0.035, 0.115), (0.195, 0.035), (0.275, 0.195),
                          (0.115, 0.275)]))
    stroke_only(p, W_THIN)
    p.drawPath(poly_path([(0.055, 0.335), (0.245, 0.19),
                          (0.975, 0.475), (0.575, 0.955)]))
    stroke_only(p, W_HAIR)
    p.drawLine(QPointF(0.175, 0.355), QPointF(0.335, 0.435))
    p.drawLine(QPointF(0.145, 0.435), QPointF(0.285, 0.61))
    lit = blob(0.555, 0.575, 0.23, seed=9, n=9, jag=0.15)
    erase_fill(p, lit)
    cell(p, 0.555, 0.575, 0.185, seed=9, w=W_MAIN, nr=0.36)


def activation_04(p):
    """An eye attending to a cell."""
    eye = QPainterPath()
    eye.moveTo(0.045, 0.42)
    eye.quadTo(0.245, 0.185, 0.445, 0.42)
    eye.quadTo(0.245, 0.655, 0.045, 0.42)
    stroke_only(p, W_MAIN)
    p.drawPath(eye)
    fill_only(p)
    p.drawEllipse(QPointF(0.245, 0.42), 0.082, 0.082)
    stroke_only(p, W_HAIR)
    for t in (0.18, 0.5, 0.82):
        p.drawLine(QPointF(0.46, 0.44 + 0.02 * t), QPointF(0.66, 0.48 + 0.30 * t))
    cell(p, 0.735, 0.735, 0.215, seed=13, w=W_THIN, nr=0.34)


def activation_05(p):
    """Network activations feeding a cell."""
    cols = [(0.065, 3), (0.30, 3)]
    nodes = []
    for x, n in cols:
        nodes.append([(x, 0.5 + (i - (n - 1) / 2.0) * (0.66 / max(n - 1, 1)))
                      for i in range(n)])
    stroke_only(p, 0.017)
    for a in nodes[0]:
        for b in nodes[1]:
            p.drawLine(QPointF(a[0], a[1]), QPointF(b[0], b[1]))
    for col in nodes:
        for x, y in col:
            fill_only(p)
            p.drawEllipse(QPointF(x, y), 0.052, 0.052)
    arrow(p, 0.395, 0.5, 0.495, 0.5, w=0.026, head=0.055)
    cell(p, 0.755, 0.5, 0.225, seed=17, w=W_MAIN, nucleus=False, jag=0.13)
    fill_only(p)
    p.drawPath(blob(0.805, 0.435, 0.082, seed=21, n=8, jag=0.18))


def activation_06(p):
    """Hot spot radiating inside the cell."""
    cell(p, 0.5, 0.5, 0.43, seed=23, w=W_MAIN, nucleus=False, jag=0.12)
    hx, hy = 0.5, 0.5
    fill_only(p)
    p.drawEllipse(QPointF(hx, hy), 0.075, 0.075)
    stroke_only(p, 0.024)
    for i in range(12):
        a = 2 * math.pi * i / 12 + math.pi / 12
        r0 = 0.115
        r1 = 0.245 if i % 2 == 0 else 0.185
        p.drawLine(QPointF(hx + r0 * math.cos(a), hy + r0 * math.sin(a)),
                   QPointF(hx + r1 * math.cos(a), hy + r1 * math.sin(a)))


def activation_07(p):
    """Saliency map as a topographic contour plot."""
    stroke_only(p, W_MAIN)
    p.drawRect(QRectF(0.075, 0.075, 0.85, 0.85))
    hx, hy = 0.575, 0.44
    for r in (0.325, 0.255, 0.185, 0.120):
        stroke_only(p, W_THIN)
        p.drawPath(blob(hx, hy, r, seed=29, n=9, jag=0.20))
    fill_only(p)
    p.drawPath(blob(hx, hy, 0.062, seed=29, n=9, jag=0.20))
    stroke_only(p, W_HAIR)
    p.drawPath(blob(0.245, 0.775, 0.095, seed=31, n=7, jag=0.22))


def activation_08(p):
    """One cell, half plain and half saturated with attention."""
    b = blob(0.5, 0.5, 0.43, seed=37, n=9, jag=0.13)
    stroke_only(p, W_MAIN)
    p.drawPath(b)
    p.save()
    p.setClipPath(b)
    fill_only(p)
    rnd = random.Random(101)
    for gx in range(9):
        for gy in range(9):
            x = 0.09 + gx * 0.0975 + rnd.uniform(-0.012, 0.012)
            y = 0.09 + gy * 0.0975 + rnd.uniform(-0.012, 0.012)
            if x > 0.505:
                p.drawEllipse(QPointF(x, y), 0.028, 0.028)
            elif rnd.random() < 0.30:
                p.drawEllipse(QPointF(x, y), 0.011, 0.011)
    p.restore()
    stroke_only(p, W_THIN)
    p.drawLine(QPointF(0.5, 0.085), QPointF(0.5, 0.915))


def activation_09(p):
    """Neuron firing an action-potential spike train."""
    soma = blob(0.245, 0.50, 0.155, seed=41, n=7, jag=0.20)
    stroke_only(p, W_MAIN)
    p.drawPath(soma)
    fill_only(p)
    p.drawEllipse(QPointF(0.245, 0.50), 0.055, 0.055)
    stroke_only(p, W_THIN)
    for a, ln in ((200, 0.19), (145, 0.20), (250, 0.18), (105, 0.17)):
        r = math.radians(a)
        x0, y0 = 0.245 + 0.13 * math.cos(r), 0.50 + 0.13 * math.sin(r)
        x1, y1 = 0.245 + (0.13 + ln) * math.cos(r), 0.50 + (0.13 + ln) * math.sin(r)
        p.drawLine(QPointF(x0, y0), QPointF(x1, y1))
        for s in (-0.55, 0.55):
            p.drawLine(QPointF(x1, y1),
                       QPointF(x1 + 0.075 * math.cos(r + s),
                               y1 + 0.075 * math.sin(r + s)))
    stroke_only(p, 0.026)
    p.drawLine(QPointF(0.39, 0.50), QPointF(0.53, 0.50))
    pts = [(0.53, 0.50)]
    x = 0.53
    for i in range(3):
        pts += [(x, 0.50), (x + 0.03, 0.22), (x + 0.075, 0.22), (x + 0.105, 0.50),
                (x + 0.145, 0.50)]
        x += 0.145
    pts.append((0.965, 0.50))
    p.drawPath(line_path(pts))


def activation_10(p):
    """Attention brackets locking onto a sub-region of the cell."""
    cell(p, 0.415, 0.585, 0.355, seed=43, w=W_MAIN, nucleus=False, jag=0.13)
    stroke_only(p, W_HAIR)
    for cx, cy, r in ((0.27, 0.68, 0.055), (0.42, 0.80, 0.042), (0.235, 0.47, 0.040)):
        p.drawEllipse(QPointF(cx, cy), r, r)
    x0, y0, x1, y1 = 0.535, 0.075, 0.955, 0.495
    L = 0.145
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    corners = QPainterPath()
    for (ax, ay, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1),
                             (x0, y1, 1, -1), (x1, y1, -1, -1)):
        corners = corners.united(outline_of(
            line_path([(ax, ay + dy * L), (ax, ay), (ax + dx * L, ay)]), 0.040))
    hair = outline_of(line_path([(cx - 0.09, cy), (cx + 0.09, cy)]), 0.024)
    hair = hair.united(outline_of(line_path([(cx, cy - 0.09), (cx, cy + 0.09)]), 0.024))
    art = corners.united(hair).simplified()
    erase_stroke(p, art, 0.034)
    fill_only(p)
    p.drawPath(art)
    p.drawEllipse(QPointF(cx, cy), 0.045, 0.045)


# ==========================================================================
# ANNOTATE -- human labelling of crops
# ==========================================================================

def _crop_tile(p, x0, y0, x1, y1, seed, w=W_MAIN, rad=0.045, nr=0.36):
    rrect(p, x0, y0, x1, y1, rad=rad, w=w)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    r = min(x1 - x0, y1 - y0) * 0.31
    cell(p, cx, cy, r, seed=seed, w=w * 0.85, nr=nr, jag=0.16)


def annotate_01(p):
    """A grid of crops being scored one by one."""
    coords = [(0.055, 0.055), (0.535, 0.055), (0.055, 0.535), (0.535, 0.535)]
    seeds = [3, 5, 7, 11]
    for (x, y), s in zip(coords, seeds):
        _crop_tile(p, x, y, x + 0.41, y + 0.41, seed=s, w=0.026)
    badge = QPainterPath()
    badge.addEllipse(QPointF(0.465, 0.465), 0.135, 0.135)
    erase_fill(p, badge)
    fill_only(p)
    p.drawEllipse(QPointF(0.465, 0.465), 0.135, 0.135)
    p.save()
    p.setCompositionMode(QPainter.CompositionMode_Clear)
    p.setPen(QPen(QColor(0, 0, 0), 0.040, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    p.setBrush(Qt.NoBrush)
    p.drawPath(line_path([(0.395, 0.470), (0.445, 0.523), (0.545, 0.408)]))
    p.restore()


def annotate_02(p):
    """Pencil writing a label on a crop."""
    _crop_tile(p, 0.045, 0.115, 0.585, 0.655, seed=13, w=0.028)
    stroke_only(p, 0.022)
    p.drawLine(QPointF(0.075, 0.755), QPointF(0.50, 0.755))
    p.drawLine(QPointF(0.075, 0.855), QPointF(0.35, 0.855))
    body = poly_path([(0.985, 0.055), (0.86, 0.075), (0.535, 0.735),
                      (0.665, 0.83)])
    erase_stroke(p, body, 0.045)
    stroke_only(p, 0.028)
    p.drawPath(body)
    p.drawLine(QPointF(0.815, 0.175), QPointF(0.945, 0.265))
    fill_only(p)
    p.drawPath(poly_path([(0.535, 0.735), (0.665, 0.83), (0.475, 0.925)]))


def annotate_03(p):
    """Pointer clicking through crops."""
    _crop_tile(p, 0.055, 0.055, 0.665, 0.665, seed=17, w=0.030)
    curs = poly_path([(0.545, 0.435), (0.545, 0.955), (0.665, 0.835),
                      (0.755, 0.985), (0.845, 0.925), (0.755, 0.785),
                      (0.925, 0.765)])
    erase_stroke(p, curs, 0.055)
    fill_only(p)
    p.drawPath(curs)
    stroke_only(p, 0.022)
    for r in (0.085, 0.145):
        rect = QRectF(0.545 - r, 0.435 - r, 2 * r, 2 * r)
        a = QPainterPath()
        a.arcMoveTo(rect, 20)
        a.arcTo(rect, 20, 65)
        p.drawPath(a)


def annotate_04(p):
    """A label tag tied to a cell."""
    cell(p, 0.315, 0.605, 0.275, seed=19, w=W_MAIN, nr=0.34)
    tagpath = poly_path([(0.475, 0.235), (0.885, 0.055), (0.965, 0.245),
                         (0.555, 0.425)])
    stroke_only(p, W_MAIN)
    p.drawPath(tagpath)
    fill_only(p)
    p.drawEllipse(QPointF(0.555, 0.255), 0.045, 0.045)
    stroke_only(p, W_HAIR)
    p.drawLine(QPointF(0.665, 0.235), QPointF(0.855, 0.155))
    p.drawLine(QPointF(0.705, 0.325), QPointF(0.835, 0.268))
    stroke_only(p, W_THIN)
    string = QPainterPath()
    string.moveTo(0.520, 0.290)
    string.cubicTo(0.435, 0.335, 0.415, 0.360, 0.395, 0.400)
    p.drawPath(string)


def annotate_05(p):
    """A scoring checklist beside the object."""
    cell(p, 0.245, 0.50, 0.215, seed=23, w=W_MAIN, nr=0.34)
    ys = [0.235, 0.455, 0.675]
    for i, y in enumerate(ys):
        stroke_only(p, 0.024)
        p.drawRoundedRect(QRectF(0.525, y, 0.145, 0.145), 0.03, 0.03)
        p.drawLine(QPointF(0.725, y + 0.072), QPointF(0.955, y + 0.072))
        if i == 0:
            t = outline_of(line_path([(0.555, 0.305), (0.598, 0.352),
                                      (0.680, 0.235)]), 0.046)
            erase_stroke(p, t, 0.028)
            fill_only(p)
            p.drawPath(t.simplified())


def annotate_06(p):
    """Bounding box with grab handles around a cell."""
    cell(p, 0.50, 0.50, 0.30, seed=29, w=W_MAIN, nr=0.34)
    stroke_only(p, 0.022, dash=[2.4, 2.2])
    p.drawRect(QRectF(0.115, 0.115, 0.77, 0.77))
    fill_only(p)
    for x in (0.115, 0.50, 0.885):
        for y in (0.115, 0.50, 0.885):
            if x == 0.50 and y == 0.50:
                continue
            p.drawRect(QRectF(x - 0.045, y - 0.045, 0.09, 0.09))


def annotate_07(p):
    """A stamp pressing a verdict onto a crop."""
    fill_only(p)
    p.drawRoundedRect(QRectF(0.415, 0.045, 0.17, 0.185), 0.06, 0.06)
    stroke_only(p, W_MAIN)
    p.drawPath(poly_path([(0.355, 0.435), (0.415, 0.245), (0.585, 0.245),
                          (0.645, 0.435)]))
    fill_only(p)
    p.drawRoundedRect(QRectF(0.295, 0.425, 0.41, 0.085), 0.035, 0.035)
    stroke_only(p, 0.024)
    p.drawLine(QPointF(0.185, 0.32), QPointF(0.255, 0.32))
    p.drawLine(QPointF(0.745, 0.32), QPointF(0.815, 0.32))
    stroke_only(p, W_MAIN)
    p.drawRoundedRect(QRectF(0.155, 0.605, 0.69, 0.345), 0.05, 0.05)
    t = outline_of(line_path([(0.375, 0.775), (0.455, 0.855), (0.635, 0.685)]),
                   0.062)
    fill_only(p)
    p.drawPath(t.simplified())


def annotate_08(p):
    """A comment attached to an object."""
    bub = QPainterPath()
    bub.addRoundedRect(QRectF(0.325, 0.055, 0.63, 0.42), 0.075, 0.075)
    tail = poly_path([(0.435, 0.44), (0.585, 0.44), (0.395, 0.635)])
    bub = bub.united(tail).simplified()
    stroke_only(p, W_MAIN)
    p.drawPath(bub)
    stroke_only(p, 0.024)
    p.drawLine(QPointF(0.405, 0.185), QPointF(0.875, 0.185))
    p.drawLine(QPointF(0.405, 0.275), QPointF(0.875, 0.275))
    p.drawLine(QPointF(0.405, 0.365), QPointF(0.695, 0.365))
    cell(p, 0.275, 0.735, 0.235, seed=31, w=W_MAIN, nr=0.34)


def annotate_09(p):
    """Sorting crops into two bins."""
    stroke_only(p, 0.026)
    p.drawRoundedRect(QRectF(0.365, 0.035, 0.42, 0.30), 0.045, 0.045)
    mid = QPainterPath()
    mid.addRoundedRect(QRectF(0.315, 0.075, 0.42, 0.30), 0.045, 0.045)
    occlude(p, mid, 0.028)
    stroke_only(p, 0.026)
    p.drawPath(mid)
    frontc = QPainterPath()
    frontc.addRoundedRect(QRectF(0.255, 0.115, 0.42, 0.30), 0.045, 0.045)
    occlude(p, frontc, 0.028)
    stroke_only(p, W_MAIN)
    p.drawPath(frontc)
    cell(p, 0.465, 0.265, 0.105, seed=37, w=0.024, nr=0.36)
    arrow(p, 0.275, 0.475, 0.135, 0.635, w=0.024, head=0.055)
    arrow(p, 0.655, 0.475, 0.795, 0.635, w=0.024, head=0.055)
    stroke_only(p, 0.028)
    p.drawPath(line_path([(0.025, 0.695), (0.025, 0.925), (0.315, 0.925),
                          (0.315, 0.695)]))
    p.drawPath(line_path([(0.685, 0.695), (0.685, 0.925), (0.975, 0.925),
                          (0.975, 0.695)]))
    fill_only(p)
    p.drawRoundedRect(QRectF(0.055, 0.795, 0.23, 0.10), 0.04, 0.04)


def annotate_10(p):
    """Tracing an outline by hand with a stylus."""
    cell(p, 0.455, 0.545, 0.245, seed=41, w=W_THIN, nr=0.34)
    trace = blob(0.455, 0.545, 0.355, seed=41, n=9, jag=0.15)
    stroke_only(p, 0.026, dash=[2.2, 2.0])
    p.drawPath(trace)
    nib = poly_path([(0.965, 0.045), (0.735, 0.275), (0.665, 0.395),
                     (0.795, 0.325), (0.985, 0.135)])
    erase_stroke(p, nib, 0.045)
    stroke_only(p, 0.026)
    p.drawPath(nib)
    fill_only(p)
    p.drawPath(poly_path([(0.665, 0.395), (0.735, 0.275), (0.795, 0.325)]))


# ==========================================================================
# CELLPOSE_ALL -- run Cellpose over the whole dataset (masks + measurements)
# ==========================================================================

def cellpose_all_01(p):
    """A whole batch of frames, the front one segmented."""
    stroke_only(p, 0.024)
    p.drawRoundedRect(QRectF(0.28, 0.055, 0.665, 0.665), 0.05, 0.05)
    p.drawRoundedRect(QRectF(0.175, 0.135, 0.665, 0.665), 0.05, 0.05)
    front = QRectF(0.055, 0.225, 0.665, 0.665)
    fp = QPainterPath()
    fp.addRoundedRect(front, 0.05, 0.05)
    occlude(p, fp, 0.032)
    stroke_only(p, W_MAIN)
    p.drawRoundedRect(front, 0.05, 0.05)
    clip = QPainterPath()
    clip.addRoundedRect(QRectF(0.070, 0.240, 0.635, 0.635), 0.04, 0.04)
    p.save()
    p.setClipPath(clip)
    cell(p, 0.205, 0.395, 0.105, seed=3, w=0.024, nr=0.36)
    cell(p, 0.455, 0.375, 0.115, seed=5, w=0.024, nr=0.36)
    cell(p, 0.315, 0.665, 0.115, seed=7, w=0.024, nr=0.36)
    cell(p, 0.575, 0.685, 0.098, seed=11, w=0.024, nr=0.36)
    p.restore()


def cellpose_all_02(p):
    """Every well of the plate segmented."""
    stroke_only(p, W_MAIN)
    p.drawRoundedRect(QRectF(0.045, 0.155, 0.91, 0.69), 0.055, 0.055)
    xs = [0.185, 0.375, 0.565, 0.755]
    ys = [0.345, 0.66]
    n = 0
    for y in ys:
        for x in xs:
            stroke_only(p, 0.020)
            p.drawEllipse(QPointF(x, y), 0.095, 0.095)
            fill_only(p)
            p.drawPath(blob(x, y, 0.048, seed=13 + n, n=8, jag=0.22))
            n += 1
    stroke_only(p, W_HAIR)
    p.drawLine(QPointF(0.075, 0.245), QPointF(0.925, 0.245))


def cellpose_all_03(p):
    """End to end: image, then masks, then measurements."""
    stroke_only(p, 0.026)
    p.drawRoundedRect(QRectF(0.025, 0.325, 0.245, 0.35), 0.04, 0.04)
    sp = blob(0.148, 0.50, 0.085, seed=17, n=8, jag=0.18)
    speckle(p, sp, seed=17, count=9, rmin=0.012, rmax=0.020)
    stroke_only(p, 0.026)
    p.drawRoundedRect(QRectF(0.375, 0.325, 0.245, 0.35), 0.04, 0.04)
    fill_only(p)
    p.drawPath(blob(0.498, 0.50, 0.088, seed=17, n=8, jag=0.18))
    stroke_only(p, 0.026)
    p.drawRoundedRect(QRectF(0.725, 0.325, 0.25, 0.35), 0.04, 0.04)
    stroke_only(p, W_HAIR)
    for y in (0.415, 0.50, 0.585):
        p.drawLine(QPointF(0.755, y), QPointF(0.945, y))
    p.drawLine(QPointF(0.85, 0.355), QPointF(0.85, 0.645))
    arrow(p, 0.295, 0.50, 0.35, 0.50, w=0.022, head=0.042)
    arrow(p, 0.645, 0.50, 0.70, 0.50, w=0.022, head=0.042)


def cellpose_all_04(p):
    """Frames streaming past the segmenter on a belt."""
    # scan head + beam
    fill_only(p)
    p.drawRoundedRect(QRectF(0.275, 0.035, 0.45, 0.095), 0.045, 0.045)
    stroke_only(p, 0.022)
    p.drawLine(QPointF(0.305, 0.145), QPointF(0.395, 0.365))
    p.drawLine(QPointF(0.695, 0.145), QPointF(0.605, 0.365))
    stroke_only(p, W_HAIR)
    p.drawLine(QPointF(0.50, 0.165), QPointF(0.50, 0.245))
    # frames on the belt
    stroke_only(p, 0.026)
    for x in (0.055, 0.3825, 0.71):
        p.drawRoundedRect(QRectF(x, 0.395, 0.235, 0.285), 0.04, 0.04)
    cell(p, 0.1725, 0.5375, 0.085, seed=19, w=0.022, nr=0.36)
    cell(p, 0.50, 0.5375, 0.09, seed=23, w=0.022, nr=0.36)
    cell(p, 0.8275, 0.5375, 0.08, seed=29, w=0.022, nr=0.36)
    # belt
    fill_only(p)
    p.drawRoundedRect(QRectF(0.045, 0.715, 0.91, 0.055), 0.027, 0.027)
    stroke_only(p, 0.024)
    for x in (0.155, 0.385, 0.615, 0.845):
        p.drawEllipse(QPointF(x, 0.865), 0.078, 0.078)


def cellpose_all_05(p):
    """Point it at a folder and it does the lot."""
    f = poly_path([(0.055, 0.945), (0.055, 0.445), (0.365, 0.445),
                   (0.455, 0.565), (0.945, 0.565), (0.945, 0.945)])
    stroke_only(p, W_MAIN)
    p.drawPath(f)
    erase_stroke(p, blob(0.245, 0.275, 0.185, seed=31, n=9, jag=0.15), 0.030)
    erase_stroke(p, blob(0.555, 0.175, 0.155, seed=37, n=9, jag=0.15), 0.030)
    erase_stroke(p, blob(0.815, 0.315, 0.165, seed=41, n=9, jag=0.15), 0.030)
    cell(p, 0.245, 0.275, 0.155, seed=31, w=0.026, nr=0.36)
    cell(p, 0.555, 0.175, 0.125, seed=37, w=0.026, nr=0.36)
    cell(p, 0.815, 0.315, 0.135, seed=41, w=0.026, nr=0.36)


def cellpose_all_06(p):
    """A contact sheet: every field, every shape."""
    stroke_only(p, W_MAIN)
    p.drawRoundedRect(QRectF(0.045, 0.045, 0.91, 0.91), 0.05, 0.05)
    stroke_only(p, 0.020)
    for t in (0.348, 0.652):
        p.drawLine(QPointF(t, 0.045), QPointF(t, 0.955))
        p.drawLine(QPointF(0.045, t), QPointF(0.955, t))
    cs = [0.1965, 0.50, 0.8035]
    n = 0
    for y in cs:
        for x in cs:
            cell(p, x, y, 0.098, seed=43 + n * 7, w=0.020, nr=0.38, jag=0.20)
            n += 1


def cellpose_all_07(p):
    """All channels folded into one segmentation."""
    cx, hw, hh = 0.285, 0.245, 0.085
    for cy in (0.235, 0.415, 0.595):
        stroke_only(p, 0.026)
        p.drawPath(poly_path([(cx - hw, cy), (cx, cy - hh),
                              (cx + hw, cy), (cx, cy + hh)]))
    fill_only(p)
    p.drawEllipse(QPointF(cx, 0.235), 0.048, 0.048)
    p.drawPath(blob(cx, 0.415, 0.058, seed=53, n=8, jag=0.20))
    stroke_only(p, 0.022)
    p.drawEllipse(QPointF(cx, 0.595), 0.052, 0.052)
    arrow(p, 0.555, 0.415, 0.635, 0.415, w=0.026, head=0.055)
    cell(p, 0.815, 0.435, 0.165, seed=53, w=W_MAIN, nucleus=False, jag=0.14)
    fill_only(p)
    p.drawEllipse(QPointF(0.795, 0.405), 0.055, 0.055)
    stroke_only(p, W_HAIR)
    p.drawEllipse(QPointF(0.885, 0.515), 0.038, 0.038)


def cellpose_all_08(p):
    """Masks and the measurements that come with them."""
    cell(p, 0.50, 0.30, 0.245, seed=59, w=W_MAIN, nr=0.34)
    stroke_only(p, W_THIN)
    p.drawLine(QPointF(0.075, 0.925), QPointF(0.925, 0.925))
    fill_only(p)
    for x, h in ((0.145, 0.20), (0.325, 0.33), (0.505, 0.145), (0.685, 0.265)):
        p.drawRect(QRectF(x, 0.905 - h, 0.145, h))


def cellpose_all_09(p):
    """Everything in, uniform masks out."""
    cell(p, 0.185, 0.155, 0.115, seed=61, w=0.024, nr=0.0, nucleus=False, jag=0.25)
    cell(p, 0.50, 0.115, 0.095, seed=67, w=0.024, nucleus=False, jag=0.25)
    cell(p, 0.815, 0.155, 0.12, seed=71, w=0.024, nucleus=False, jag=0.25)
    stroke_only(p, W_MAIN)
    p.drawPath(line_path([(0.075, 0.325), (0.425, 0.615), (0.425, 0.775)]))
    p.drawPath(line_path([(0.925, 0.325), (0.575, 0.615), (0.575, 0.775)]))
    p.drawLine(QPointF(0.075, 0.325), QPointF(0.925, 0.325))
    fill_only(p)
    p.drawPath(blob(0.50, 0.905, 0.085, seed=73, n=8, jag=0.16))


def cellpose_all_10(p):
    """A queue of jobs, all of them running."""
    stroke_only(p, 0.024)
    for y in (0.115, 0.415, 0.715):
        p.drawRoundedRect(QRectF(0.045, y, 0.20, 0.20), 0.04, 0.04)
    cell(p, 0.145, 0.215, 0.062, seed=79, w=0.020, nr=0.38)
    cell(p, 0.145, 0.515, 0.062, seed=83, w=0.020, nr=0.38)
    cell(p, 0.145, 0.815, 0.062, seed=89, w=0.020, nr=0.38)
    for y, frac in ((0.175, 1.0), (0.475, 1.0), (0.775, 0.45)):
        stroke_only(p, 0.020)
        p.drawRoundedRect(QRectF(0.315, y, 0.64, 0.085), 0.042, 0.042)
        fill_only(p)
        p.drawRoundedRect(QRectF(0.335, y + 0.018, (0.60) * frac, 0.049),
                          0.024, 0.024)


# ==========================================================================
# CELLPOSE_MASKS -- produce the mask for an object
# ==========================================================================

def cellpose_masks_01(p):
    """Raw signal on one side, finished mask on the other."""
    b = blob(0.50, 0.50, 0.42, seed=3, n=9, jag=0.14)
    p.save()
    p.setClipPath(b)
    fill_only(p)
    p.drawRect(QRectF(0.50, 0.0, 0.55, 1.0))
    p.restore()
    left = QPainterPath()
    left.addRect(QRectF(0.0, 0.0, 0.50, 1.0))
    speckle(p, b.intersected(left), seed=11, count=13, rmin=0.030, rmax=0.050)
    stroke_only(p, W_MAIN)
    p.drawPath(b)
    p.save()
    p.setClipPath(b)
    stroke_only(p, W_THIN)
    p.drawLine(QPointF(0.50, 0.0), QPointF(0.50, 1.0))
    p.restore()


def cellpose_masks_02(p):
    """A polygon ROI with its vertices."""
    rnd = random.Random(5)
    n = 9
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n - math.pi / 2 + rnd.uniform(-0.20, 0.20)
        r = 0.435 * rnd.uniform(0.72, 1.0)
        pts.append((0.5 + r * math.cos(a), 0.5 + r * math.sin(a)))
    stroke_only(p, W_MAIN)
    p.drawPath(poly_path(pts))
    fill_only(p)
    for x, y in pts:
        p.drawRect(QRectF(x - 0.042, y - 0.042, 0.084, 0.084))


def cellpose_masks_03(p):
    """The mask as a stencil cut out of the field."""
    fill_only(p)
    p.drawRoundedRect(QRectF(0.065, 0.065, 0.87, 0.87), 0.07, 0.07)
    hole = blob(0.50, 0.50, 0.30, seed=7, n=9, jag=0.16)
    erase_fill(p, hole)
    p.save()
    p.setCompositionMode(QPainter.CompositionMode_Clear)
    p.setPen(QPen(QColor(0, 0, 0), 0.028, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    p.setBrush(Qt.NoBrush)
    p.drawPath(blob(0.50, 0.50, 0.375, seed=7, n=9, jag=0.16))
    p.restore()


def cellpose_masks_04(p):
    """Instance labels: every object gets its own number."""
    specs = [(0.285, 0.275, 0.215, 11, "1"),
             (0.735, 0.335, 0.195, 13, "2"),
             (0.455, 0.735, 0.225, 17, "3")]
    for cx, cy, r, s, t in specs:
        stroke_only(p, W_MAIN)
        p.drawPath(blob(cx, cy, r, seed=s, n=8, jag=0.16))
        fill_only(p)
        p.drawPath(text_path(t, r * 0.85, cx, cy))


def cellpose_masks_05(p):
    """The mask is per pixel."""
    b = blob(0.50, 0.50, 0.42, seed=19, n=9, jag=0.13)
    step = 0.1365
    fill_only(p)
    y = 0.045
    while y < 0.955:
        x = 0.045
        while x < 0.955:
            if b.contains(QPointF(x + step / 2, y + step / 2)):
                p.drawRect(QRectF(x + 0.010, y + 0.010, step - 0.020, step - 0.020))
            x += step
        y += step
    stroke_only(p, 0.022)
    p.drawPath(b)


def cellpose_masks_06(p):
    """Cellpose flow vectors converging on the centre."""
    b = blob(0.50, 0.50, 0.44, seed=23, n=9, jag=0.13)
    stroke_only(p, W_MAIN)
    p.drawPath(b)
    for i in range(10):
        a = 2 * math.pi * i / 10 + 0.18
        r0 = 0.365 if i % 2 == 0 else 0.30
        x0, y0 = 0.5 + r0 * math.cos(a), 0.5 + r0 * math.sin(a)
        x1, y1 = 0.5 + 0.135 * math.cos(a), 0.5 + 0.135 * math.sin(a)
        arrow(p, x0, y0, x1, y1, w=0.019, head=0.045, spread=30)
    fill_only(p)
    p.drawEllipse(QPointF(0.50, 0.50), 0.058, 0.058)


def cellpose_masks_07(p):
    """The mask layer sitting above the image layer."""
    stroke_only(p, 0.026)
    p.drawPath(poly_path([(0.055, 0.735), (0.475, 0.545), (0.945, 0.735),
                          (0.525, 0.925)]))
    sp = blob(0.50, 0.735, 0.115, seed=29, n=8, jag=0.18, sy=0.45)
    speckle(p, sp, seed=29, count=8, rmin=0.014, rmax=0.024)
    stroke_only(p, 0.026)
    p.drawPath(poly_path([(0.055, 0.345), (0.475, 0.155), (0.945, 0.345),
                          (0.525, 0.535)]))
    fill_only(p)
    p.drawPath(blob(0.50, 0.345, 0.155, seed=29, n=8, jag=0.18, sy=0.45))
    stroke_only(p, W_HAIR)
    p.drawLine(QPointF(0.075, 0.395), QPointF(0.075, 0.685))
    p.drawLine(QPointF(0.925, 0.395), QPointF(0.925, 0.685))


def cellpose_masks_08(p):
    """The model diameter fitted to the object."""
    cell(p, 0.50, 0.53, 0.335, seed=31, w=W_MAIN, nr=0.30, jag=0.15)
    stroke_only(p, 0.022, dash=[2.4, 2.2])
    p.drawEllipse(QPointF(0.50, 0.53), 0.40, 0.40)
    span = outline_of(line_path([(0.155, 0.53), (0.845, 0.53)]), 0.026)
    for s in (1, -1):
        x = 0.50 + s * 0.345
        span = span.united(poly_path([(x, 0.53), (x - s * 0.085, 0.485),
                                      (x - s * 0.085, 0.575)]))
    span = span.simplified()
    erase_stroke(p, span, 0.034)
    fill_only(p)
    p.drawPath(span)
    stroke_only(p, W_HAIR)
    p.drawLine(QPointF(0.155, 0.115), QPointF(0.155, 0.44))
    p.drawLine(QPointF(0.845, 0.115), QPointF(0.845, 0.44))


def cellpose_masks_09(p):
    """Touching objects split along their ridges."""
    specs = [(0.315, 0.325, 0.255, 37), (0.715, 0.335, 0.235, 41),
             (0.485, 0.735, 0.245, 43)]
    for cx, cy, r, s in specs:
        stroke_only(p, 0.036)
        p.drawPath(blob(cx, cy, r, seed=s, n=8, jag=0.13))
        fill_only(p)
        p.drawEllipse(QPointF(cx, cy), r * 0.30, r * 0.30)


def cellpose_masks_10(p):
    """Filled object in, boundary out."""
    fill_only(p)
    p.drawPath(blob(0.215, 0.50, 0.205, seed=47, n=9, jag=0.16))
    solid_arrow(p, 0.445, 0.50, 0.575, 0.50, w=0.032, head=0.075, hw=0.052)
    stroke_only(p, 0.042)
    p.drawPath(blob(0.795, 0.50, 0.205, seed=47, n=9, jag=0.16))


# ==========================================================================
# registry
# ==========================================================================

ICONS = {
    "abort": [abort_01, abort_02, abort_03, abort_04, abort_05,
              abort_06, abort_07, abort_08, abort_09, abort_10],
    "activation": [activation_01, activation_02, activation_03, activation_04,
                   activation_05, activation_06, activation_07, activation_08,
                   activation_09, activation_10],
    "annotate": [annotate_01, annotate_02, annotate_03, annotate_04,
                 annotate_05, annotate_06, annotate_07, annotate_08,
                 annotate_09, annotate_10],
    "cellpose_all": [cellpose_all_01, cellpose_all_02, cellpose_all_03,
                     cellpose_all_04, cellpose_all_05, cellpose_all_06,
                     cellpose_all_07, cellpose_all_08, cellpose_all_09,
                     cellpose_all_10],
    "cellpose_masks": [cellpose_masks_01, cellpose_masks_02, cellpose_masks_03,
                       cellpose_masks_04, cellpose_masks_05, cellpose_masks_06,
                       cellpose_masks_07, cellpose_masks_08, cellpose_masks_09,
                       cellpose_masks_10],
}

CONCEPTS = {
    "abort": [
        "Octagonal stop sign with a solid square core.",
        "Raised open palm (halt) in front of a petri dish.",
        "Emergency-stop mushroom button seen from above.",
        "Power plug yanked out of its socket.",
        "Pipeline arrow slamming into a wall.",
        "Prohibition sign struck over a cell.",
        "Progress bar terminated mid-run.",
        "Scissors cutting the run line.",
        "Power / shutdown symbol.",
        "Traffic signal with the stop lamp lit.",
    ],
    "activation": [
        "Cell with concentric Grad-CAM contours around a hot spot.",
        "Saliency rendered as a block heat map inside the cell.",
        "Spotlight beam picking out one cell.",
        "An eye attending to a cell.",
        "Network activations fanning into a cell.",
        "Hot spot radiating inside the cell.",
        "Saliency map as a topographic contour plot.",
        "One cell, half plain and half saturated with attention.",
        "Neuron firing an action-potential spike train.",
        "Attention brackets locking onto a sub-region of the cell.",
    ],
    "annotate": [
        "A grid of crops being scored one by one.",
        "Pencil writing a label on a crop.",
        "Pointer clicking through crops.",
        "A label tag tied to a cell.",
        "A scoring checklist beside the object.",
        "Bounding box with grab handles around a cell.",
        "A stamp pressing a verdict onto a crop.",
        "A comment attached to an object.",
        "Sorting crops into two bins.",
        "Tracing an outline by hand with a stylus.",
    ],
    "cellpose_all": [
        "A whole batch of frames, the front one segmented.",
        "Every well of the plate segmented.",
        "End to end: image, then masks, then measurements.",
        "Frames streaming past the segmenter on a belt.",
        "Point it at a folder and it does the lot.",
        "A contact sheet: every field, every shape.",
        "All channels folded into one segmentation.",
        "Masks and the measurements that come with them.",
        "Everything in, uniform masks out (funnel).",
        "A queue of jobs, all of them running.",
    ],
    "cellpose_masks": [
        "Raw signal on one side, finished mask on the other.",
        "A polygon ROI with its vertices.",
        "The mask as a stencil cut out of the field.",
        "Instance labels: every object gets its own number.",
        "The mask is per pixel (pixelated silhouette).",
        "Cellpose flow vectors converging on the centre.",
        "The mask layer sitting above the image layer.",
        "The model diameter fitted to the object (Cellpose's `diameter`).",
        "Touching objects split along their ridges.",
        "Filled object in, boundary out.",
    ],
}


# ==========================================================================
# rendering
# ==========================================================================

def render(fn, size=SIZE):
    img = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setRenderHint(QPainter.SmoothPixmapTransform, True)
    p.scale(size, size)
    fn(p)
    p.end()
    return img


def _tinted(src, ink):
    """Return `src`'s alpha painted in colour `ink` (icons are alpha masks)."""
    out = QImage(src.size(), QImage.Format_ARGB32_Premultiplied)
    out.fill(Qt.transparent)
    p = QPainter(out)
    p.fillRect(out.rect(), ink)
    p.setCompositionMode(QPainter.CompositionMode_DestinationIn)
    p.drawImage(0, 0, src)
    p.end()
    return out


def contact_sheet(name, imgs, bg, ink, subtitle, path):
    cols, rows = 5, 2
    tile, pad, lab, header = 300, 20, 62, 78
    W = cols * tile + (cols + 1) * pad
    H = header + rows * (tile + lab) + (rows + 1) * pad
    sheet = QImage(W, H, QImage.Format_ARGB32_Premultiplied)
    sheet.fill(bg)
    p = QPainter(sheet)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setRenderHint(QPainter.SmoothPixmapTransform, True)

    f = QFont("DejaVu Sans")
    f.setPixelSize(34)
    f.setBold(True)
    p.setFont(f)
    p.setPen(QPen(ink))
    p.drawText(QRectF(pad, 14, W - 2 * pad, 44), Qt.AlignLeft | Qt.AlignVCenter,
               "%s  -  10 candidates" % name)
    f2 = QFont("DejaVu Sans")
    f2.setPixelSize(22)
    p.setFont(f2)
    p.drawText(QRectF(pad, 46, W - 2 * pad, 30), Qt.AlignLeft | Qt.AlignVCenter,
               subtitle)

    grid = QColor(ink)
    grid.setAlpha(48)
    for i, img in enumerate(imgs):
        c, r = i % cols, i // cols
        x = pad + c * (tile + pad)
        y = header + pad + r * (tile + lab + pad)
        p.setPen(QPen(grid, 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(QRectF(x, y, tile, tile + lab))
        big = _tinted(img.scaled(tile - 24, tile - 24, Qt.KeepAspectRatio,
                                 Qt.SmoothTransformation), ink)
        p.drawImage(int(x + 12), int(y + 12), big)
        p.setPen(QPen(ink))
        f3 = QFont("DejaVu Sans")
        f3.setPixelSize(26)
        f3.setBold(True)
        p.setFont(f3)
        p.drawText(QRectF(x + 14, y + tile, 120, lab), Qt.AlignLeft | Qt.AlignVCenter,
                   "%02d" % (i + 1))
        small = _tinted(img.scaled(48, 48, Qt.KeepAspectRatio,
                                   Qt.SmoothTransformation), ink)
        p.drawImage(int(x + tile - 60), int(y + tile + (lab - 48) // 2), small)
    p.end()
    sheet.save(path, "PNG")


def alpha_stats(path):
    from PIL import Image
    import numpy as np
    im = Image.open(path)
    assert im.mode == "RGBA", "%s is %s" % (path, im.mode)
    a = np.array(im)
    return im.size, float((a[..., 3] > 10).mean()), int(a[..., :3][a[..., 3] > 10].min())


def main():
    app = QGuiApplication(sys.argv)
    _ = app
    report = []
    for name, fns in ICONS.items():
        outdir = os.path.join(OUTROOT, name)
        os.makedirs(outdir, exist_ok=True)
        imgs = []
        for i, fn in enumerate(fns, start=1):
            img = render(fn)
            fp = os.path.join(outdir, "%s_%02d.png" % (name, i))
            img.save(fp, "PNG")
            imgs.append(img)
            sz, frac, mn = alpha_stats(fp)
            ok = "ok " if 0.05 <= frac <= 0.70 else "BAD"
            report.append("%s %-16s %02d  %sx%s  alpha=%.3f  minRGB=%d"
                          % (ok, name, i, sz[0], sz[1], frac, mn))
        with open(os.path.join(outdir, "CONCEPTS.md"), "w") as fh:
            fh.write("# %s -- 10 candidate concepts\n\n" % name)
            fh.write("White artwork on transparent background, 1024x1024 RGBA, "
                     "house style of `plaque.png` / `measure.png`.\n"
                     "Candidates for review only; nothing here is installed.\n\n")
            for i, line in enumerate(CONCEPTS[name], start=1):
                fh.write("%d. **%s_%02d** - %s\n" % (i, name, i, line))
            fh.write("\nContact sheets: `_sheet_dark.png` (white ink on #14161a) "
                     "and `_sheet_light.png` (the same alpha masks tinted dark on "
                     "#f5f6f8, which is how they would have to be drawn in a light "
                     "theme -- as shipped they are pure white and invisible there).\n")
        contact_sheet(name, imgs, QColor("#14161a"), QColor("#ffffff"),
                      "white ink on dark - 48 px preview at the right of each label",
                      os.path.join(outdir, "_sheet_dark.png"))
        contact_sheet(name, imgs, QColor("#f5f6f8"), QColor("#14161a"),
                      "same alpha masks tinted dark for light theme - 48 px preview "
                      "at the right of each label",
                      os.path.join(outdir, "_sheet_light.png"))
    print("\n".join(report))
    bad = [r for r in report if r.startswith("BAD")]
    print("\n%d files, %d outside the 5-70%% alpha window" % (len(report), len(bad)))


if __name__ == "__main__":
    main()
