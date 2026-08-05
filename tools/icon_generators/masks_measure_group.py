#!/usr/bin/env python3
"""
Candidate icon generator for spaCR -- group: make_masks, map_barcodes, mask, measure.

House style (derived from spacr/resources/icons/plaque.png and measure.png):
  * pure white artwork on a fully transparent background (alpha carries the shape)
  * flat, no gradients / no colour
  * mix of thin outlined strokes and solid white fills
  * square canvas, subject fills most of the frame, modest margin
  * literal but stylised biology / lab objects, not abstract glyphs

Everything is drawn in a normalised 0..1 coordinate space and scaled to the output
canvas, so any SIZE renders identically.  Deterministic: every random shape is
driven by an explicit seed.

Run standalone:
    QT_QPA_PLATFORM=offscreen python3 masks_measure_group.py [outdir]

Default outdir is the backup_icons directory two levels up from this file.
"""

import os
import sys
import math
import random

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt, QRectF, QPointF  # noqa: E402
from PySide6.QtGui import (  # noqa: E402
    QImage, QPainter, QPainterPath, QPen, QColor, QTransform, QGuiApplication,
)

SIZE = 1024

# stroke weights, in normalised units (fraction of the canvas edge)
W_MAIN = 0.018
W_FINE = 0.013
W_HAIR = 0.010

WHITE = QColor(255, 255, 255)


# --------------------------------------------------------------------------- #
# canvas helpers
# --------------------------------------------------------------------------- #
def new_canvas(size=SIZE):
    img = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setRenderHint(QPainter.SmoothPixmapTransform, True)
    p.scale(size, size)
    return img, p


def pen(p, w=W_MAIN, dash=None):
    q = QPen(WHITE, w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    if dash:
        q.setStyle(Qt.CustomDashLine)
        q.setDashPattern(dash)
    p.setPen(q)
    p.setBrush(Qt.NoBrush)


def flat_pen(p, w=W_MAIN, dash=None):
    q = QPen(WHITE, w, Qt.SolidLine, Qt.FlatCap, Qt.MiterJoin)
    if dash:
        q.setStyle(Qt.CustomDashLine)
        q.setDashPattern(dash)
    p.setPen(q)
    p.setBrush(Qt.NoBrush)


def fill(p, path):
    p.setPen(Qt.NoPen)
    p.setBrush(WHITE)
    p.drawPath(path)


def clear(p, path):
    """Knock a hole in what has already been drawn."""
    p.save()
    p.setCompositionMode(QPainter.CompositionMode_Clear)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(0, 0, 0, 255))
    p.drawPath(path)
    p.restore()


# --------------------------------------------------------------------------- #
# path helpers
# --------------------------------------------------------------------------- #
def catmull_closed(pts):
    """Smooth closed path through pts (Catmull-Rom -> cubic beziers)."""
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


def blob_pts(cx, cy, rx, ry, seed=0, n=11, jitter=0.14, rot=0.0):
    rng = random.Random(seed)
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n + rot
        k = 1.0 + jitter * (rng.random() * 2 - 1)
        pts.append((cx + math.cos(a) * rx * k, cy + math.sin(a) * ry * k))
    return pts


def blob(cx, cy, rx, ry=None, seed=0, n=11, jitter=0.14, rot=0.0):
    """Organic, cell-like closed path."""
    if ry is None:
        ry = rx
    return catmull_closed(blob_pts(cx, cy, rx, ry, seed, n, jitter, rot))


def blob_scaled(cx, cy, rx, ry, seed, k, n=11, jitter=0.14, rot=0.0):
    """Same blob shape, radii multiplied by k (concentric family)."""
    return catmull_closed(blob_pts(cx, cy, rx * k, ry * k, seed, n, jitter, rot))


def ell(cx, cy, rx, ry=None):
    if ry is None:
        ry = rx
    path = QPainterPath()
    path.addEllipse(QRectF(cx - rx, cy - ry, 2 * rx, 2 * ry))
    return path


def rect(x, y, w, h, r=0.0):
    path = QPainterPath()
    if r > 0:
        path.addRoundedRect(QRectF(x, y, w, h), r, r)
    else:
        path.addRect(QRectF(x, y, w, h))
    return path


def poly(points, close=True):
    path = QPainterPath()
    path.moveTo(points[0][0], points[0][1])
    for x, y in points[1:]:
        path.lineTo(x, y)
    if close:
        path.closeSubpath()
    return path


def line(x1, y1, x2, y2):
    path = QPainterPath()
    path.moveTo(x1, y1)
    path.lineTo(x2, y2)
    return path


def rot_path(path, cx, cy, deg):
    t = QTransform()
    t.translate(cx, cy)
    t.rotate(deg)
    t.translate(-cx, -cy)
    return t.map(path)


def arc_path(cx, cy, r, a0, a1, steps=64):
    """Open arc, angles in degrees, screen coords (y down, clockwise positive)."""
    path = QPainterPath()
    for i in range(steps + 1):
        a = math.radians(a0 + (a1 - a0) * i / steps)
        x = cx + math.cos(a) * r
        y = cy + math.sin(a) * r
        if i == 0:
            path.moveTo(x, y)
        else:
            path.lineTo(x, y)
    return path


def arrow(p, x1, y1, x2, y2, w=W_FINE, head=0.055):
    """Straight arrow: stroked shaft + solid triangular head."""
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return
    ux, uy = dx / L, dy / L
    bx, by = x2 - ux * head, y2 - uy * head
    pen(p, w)
    p.drawPath(line(x1, y1, bx, by))
    hw = head * 0.44
    tri = poly([(x2, y2),
                (bx - uy * hw, by + ux * hw),
                (bx + uy * hw, by - ux * hw)])
    fill(p, tri)


def curved_arrow(p, x1, y1, cx, cy, x2, y2, w=W_FINE, head=0.055):
    path = QPainterPath()
    path.moveTo(x1, y1)
    path.quadTo(cx, cy, x2, y2)
    pen(p, w)
    p.drawPath(path)
    dx, dy = x2 - cx, y2 - cy
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    bx, by = x2 - ux * head, y2 - uy * head
    hw = head * 0.44
    fill(p, poly([(x2, y2),
                  (bx - uy * hw, by + ux * hw),
                  (bx + uy * hw, by - ux * hw)]))


def dot(p, cx, cy, r):
    fill(p, ell(cx, cy, r))


def knockout_halo(p, paths, w=0.032):
    """Erase a fattened silhouette so an overlapping tool reads against fill."""
    p.save()
    p.setCompositionMode(QPainter.CompositionMode_Clear)
    p.setPen(QPen(QColor(0, 0, 0, 255), w, Qt.SolidLine, Qt.RoundCap,
                  Qt.RoundJoin))
    p.setBrush(QColor(0, 0, 0, 255))
    for pth in paths:
        p.drawPath(pth)
    p.restore()


# --------------------------------------------------------------------------- #
# shared subjects
# --------------------------------------------------------------------------- #
def cell(p, cx, cy, r, seed=1, w=W_MAIN, nucleus=0.34, jitter=0.11, n=11):
    """House-style cell: organic outline + solid nucleus."""
    pen(p, w)
    p.drawPath(blob(cx, cy, r, r * 0.94, seed=seed, n=n, jitter=jitter))
    if nucleus:
        fill(p, ell(cx - r * 0.06, cy + r * 0.03, r * nucleus, r * nucleus * 0.92))


def mito(p, cx, cy, L, H, deg=0.0, w=W_HAIR):
    """Small outlined organelle with two dots inside (as in measure.png)."""
    body = rect(cx - L / 2, cy - H / 2, L, H, H / 2)
    body = rot_path(body, cx, cy, deg)
    pen(p, w)
    p.drawPath(body)
    for s in (-1, 1):
        d = QPointF(cx + s * L * 0.2, cy)
        d = rot_path(ell(d.x(), d.y(), H * 0.15), cx, cy, deg)
        fill(p, d)


def plate(p, x, y, w_, h_, cols, rows, w=W_FINE, well_k=0.34, rounded=True):
    """Microwell plate frame + grid of wells.  Returns list of (cx, cy, r)."""
    pen(p, w)
    p.drawPath(rect(x, y, w_, h_, min(w_, h_) * 0.09 if rounded else 0.0))
    cw = w_ / cols
    ch = h_ / rows
    r = min(cw, ch) * well_k
    out = []
    for j in range(rows):
        for i in range(cols):
            cx = x + cw * (i + 0.5)
            cy = y + ch * (j + 0.5)
            out.append((cx, cy, r))
            pen(p, W_HAIR)
            p.drawPath(ell(cx, cy, r))
    return out


def barcode_bars(seed, n, x, w_, gapk=0.55):
    """Deterministic list of (bar_x, bar_w) spanning [x, x+w_]."""
    rng = random.Random(seed)
    widths = [rng.choice([1.0, 1.0, 1.8, 2.6]) for _ in range(n)]
    gaps = [rng.choice([1.0, 1.0, 1.7]) for _ in range(n - 1)]
    total = sum(widths) + gapk * sum(gaps)
    u = w_ / total
    out = []
    cx = x
    for i, bw in enumerate(widths):
        out.append((cx, bw * u))
        cx += bw * u
        if i < n - 1:
            cx += gaps[i] * gapk * u
    return out


def barcode(p, x, y, w_, h_, seed=0, n=9, vertical=True):
    """Barcode strip.  vertical=True -> vertical bars in a horizontal strip."""
    if vertical:
        for bx, bw in barcode_bars(seed, n, x, w_):
            fill(p, rect(bx, y, bw, h_))
    else:
        for by, bh in barcode_bars(seed, n, y, h_):
            fill(p, rect(x, by, w_, bh))


def sparkle(p, cx, cy, r, thin=0.30):
    """Four-point star."""
    pts = []
    for k in range(4):
        a = math.pi / 2 * k
        pts.append((cx + math.cos(a) * r, cy + math.sin(a) * r))
        a2 = a + math.pi / 4
        pts.append((cx + math.cos(a2) * r * thin, cy + math.sin(a2) * r * thin))
    fill(p, poly(pts))


def ticks(p, x, y, dx, dy, count, length, long_every=5, long_k=1.9, w=W_FINE,
          nx=0.0, ny=1.0):
    """Row of ruler ticks starting at (x,y), stepping by (dx,dy),
    each drawn along (nx,ny)."""
    for i in range(count):
        L = length * (long_k if (i % long_every == 0) else 1.0)
        pen(p, w)
        p.drawPath(line(x + dx * i, y + dy * i,
                        x + dx * i + nx * L, y + dy * i + ny * L))


# =========================================================================== #
#  make_masks  --  the MANUAL mask editor (paint / erase / split / merge)
# =========================================================================== #
def brush_tool(p, tx, ty, deg, length=0.48, half=0.070, w=W_FINE):
    """Paint brush with its tip at (tx,ty); body extends along +x then rotated."""
    tip = poly([(0.0, 0.0), (0.165, -half * 0.95), (0.165, half * 0.95)])
    ferrule = rect(0.155, -half, 0.10, 2 * half, 0.014)
    handle = poly([(0.250, -half * 0.84), (length, -half * 0.40),
                   (length + 0.035, 0.0), (length, half * 0.40),
                   (0.250, half * 0.84)])
    t = QTransform()
    t.translate(tx, ty)
    t.rotate(deg)
    tip, ferrule, handle = t.map(tip), t.map(ferrule), t.map(handle)
    knockout_halo(p, [tip, ferrule, handle], 0.048)
    fill(p, tip)
    fill(p, ferrule)
    pen(p, w)
    p.drawPath(handle)
    knockout_halo(p, [t.map(line(0.185, -half * 0.70, 0.185, half * 0.70)),
                      t.map(line(0.225, -half * 0.70, 0.225, half * 0.70))],
                  W_HAIR)


def mm_01_brush(p):
    """Paintbrush filling a cell outline with mask paint."""
    b = blob(0.38, 0.615, 0.315, 0.305, seed=11, jitter=0.13)
    wave = QPainterPath()
    wave.moveTo(-0.05, 0.755)
    wave.cubicTo(0.10, 0.655, 0.26, 0.805, 0.40, 0.700)
    wave.cubicTo(0.52, 0.610, 0.66, 0.720, 1.05, 0.640)
    wave.lineTo(1.05, 1.1)
    wave.lineTo(-0.05, 1.1)
    wave.closeSubpath()
    fill(p, b.intersected(wave))
    pen(p, W_MAIN)
    p.drawPath(b)
    brush_tool(p, 0.545, 0.660, -46.0)


def mm_02_eraser(p):
    """Eraser lifting part of a wrong mask away."""
    b = blob(0.44, 0.64, 0.30, 0.29, seed=23, jitter=0.13)
    sweep = rot_path(rect(0.40, 0.16, 0.62, 0.44, 0.03), 0.71, 0.38, -28.0)
    fill(p, b.subtracted(sweep))
    p.save()
    p.setClipPath(sweep)
    pen(p, W_HAIR, dash=[2.4, 2.2])
    p.drawPath(b)
    p.restore()
    body = rot_path(rect(0.46, 0.20, 0.44, 0.20, 0.035), 0.68, 0.30, -28.0)
    pen(p, W_MAIN)
    p.drawPath(body)
    band = rot_path(rect(0.46, 0.335, 0.44, 0.065, 0.02), 0.68, 0.30, -28.0)
    fill(p, band.intersected(body))
    pen(p, W_FINE)
    p.drawPath(rot_path(line(0.635, 0.205, 0.635, 0.395), 0.68, 0.30, -28.0))
    for cx, cy, r in ((0.30, 0.31, 0.020), (0.24, 0.42, 0.014), (0.36, 0.22, 0.012)):
        dot(p, cx, cy, r)


def mm_03_split(p):
    """Scalpel splitting an over-merged doublet along a cut line."""
    b1 = blob(0.34, 0.66, 0.21, 0.22, seed=31, jitter=0.10)
    b2 = blob(0.63, 0.66, 0.21, 0.22, seed=32, jitter=0.10)
    u = b1.united(b2).simplified()
    pen(p, W_MAIN)
    p.drawPath(u)
    fill(p, ell(0.32, 0.68, 0.062))
    fill(p, ell(0.65, 0.68, 0.062))
    pen(p, W_FINE, dash=[2.0, 1.9])
    p.drawPath(line(0.487, 0.395, 0.487, 0.95))
    t = QTransform()
    t.translate(0.487, 0.395)
    t.rotate(-41.0)
    blade = QPainterPath()
    blade.moveTo(0.0, 0.0)
    blade.cubicTo(0.075, -0.010, 0.150, -0.028, 0.215, -0.062)
    blade.lineTo(0.215, 0.032)
    blade.cubicTo(0.145, 0.034, 0.070, 0.022, 0.0, 0.0)
    blade.closeSubpath()
    handle = poly([(0.215, -0.040), (0.560, -0.030),
                   (0.590, 0.0), (0.560, 0.030), (0.215, 0.040)])
    blade, handle = t.map(blade), t.map(handle)
    knockout_halo(p, [blade, handle], 0.045)
    fill(p, blade)
    pen(p, W_FINE)
    p.drawPath(handle)
    pen(p, W_HAIR)
    for hx in (0.270, 0.305, 0.340):
        p.drawPath(t.map(line(hx, -0.032, hx, 0.032)))
    arrow(p, 0.395, 0.905, 0.285, 0.905, W_HAIR, 0.045)
    arrow(p, 0.580, 0.905, 0.690, 0.905, W_HAIR, 0.045)


def mm_04_merge(p):
    """Merging two masks into one: internal boundary dissolves."""
    b1 = blob(0.36, 0.61, 0.23, 0.235, seed=41, jitter=0.10)
    b2 = blob(0.64, 0.61, 0.23, 0.235, seed=42, jitter=0.10)
    pen(p, W_HAIR, dash=[2.2, 2.0])
    p.drawPath(b1)
    p.drawPath(b2)
    pen(p, W_MAIN)
    p.drawPath(b1.united(b2).simplified())
    fill(p, ell(0.335, 0.63, 0.055))
    fill(p, ell(0.665, 0.63, 0.055))
    arrow(p, 0.10, 0.20, 0.415, 0.20, W_FINE, 0.058)
    arrow(p, 0.90, 0.20, 0.585, 0.20, W_FINE, 0.058)
    pen(p, W_HAIR)
    p.drawPath(line(0.50, 0.135, 0.50, 0.265))


def cursor(p, x, y, s=0.13, deg=0.0, w=W_HAIR):
    """Classic mouse pointer, outlined."""
    pth = poly([(0.0, 0.0), (0.0, 1.0), (0.26, 0.76), (0.42, 1.07),
                (0.58, 0.99), (0.42, 0.69), (0.74, 0.66)])
    t = QTransform()
    t.translate(x, y)
    t.rotate(deg)
    t.scale(s, s)
    pth = t.map(pth)
    fill(p, pth)
    p.save()
    p.setCompositionMode(QPainter.CompositionMode_Clear)
    p.setPen(QPen(QColor(0, 0, 0, 255), w * 0.9, Qt.SolidLine, Qt.RoundCap,
                  Qt.RoundJoin))
    p.setBrush(Qt.NoBrush)
    inner = QTransform()
    inner.translate(x, y)
    inner.rotate(deg)
    inner.scale(s * 0.60, s * 0.60)
    inner.translate(0.16, 0.16)
    p.drawPath(inner.map(poly([(0.0, 0.0), (0.0, 1.0), (0.26, 0.76),
                               (0.42, 1.07), (0.58, 0.99), (0.42, 0.69),
                               (0.74, 0.66)])))
    p.restore()


def mm_05_lasso(p):
    """Polygon lasso with draggable vertex handles."""
    pen(p, W_HAIR)
    p.drawPath(blob(0.47, 0.56, 0.24, 0.23, seed=51, jitter=0.12))
    fill(p, ell(0.45, 0.58, 0.062))
    verts = [(0.47 + math.cos(2 * math.pi * i / 8 - 0.4) * 0.33,
              0.56 + math.sin(2 * math.pi * i / 8 - 0.4) * 0.32) for i in range(8)]
    old = verts[7]
    verts[7] = (0.80, 0.19)
    pen(p, W_FINE)
    p.drawPath(poly(verts))
    for i, (vx, vy) in enumerate(verts):
        s = 0.030
        fill(p, rect(vx - s, vy - s, 2 * s, 2 * s))
    pen(p, W_HAIR, dash=[2.2, 2.0])
    p.drawPath(line(old[0], old[1], verts[7][0], verts[7][1]))
    cursor(p, 0.815, 0.215, 0.15, 0.0)


def mm_06_undo(p):
    """Edit history: ghost previous mask outlines + an undo arc."""
    cx, cy = 0.50, 0.575
    for k in (1.46, 1.24):
        pen(p, W_HAIR, dash=[2.4, 2.2])
        p.drawPath(blob_scaled(cx, cy, 0.205, 0.200, 63, k, jitter=0.13))
    pen(p, W_MAIN)
    p.drawPath(blob_scaled(cx, cy, 0.205, 0.200, 63, 1.0, jitter=0.13))
    fill(p, ell(cx - 0.02, cy + 0.02, 0.058))
    R = 0.415
    pen(p, W_FINE)
    p.drawPath(arc_path(cx, cy, R, 196.0, 344.0))
    a0 = math.radians(196.0)
    a1 = math.radians(212.0)
    arrow(p, cx + math.cos(a1) * R, cy + math.sin(a1) * R,
          cx + math.cos(a0) * R, cy + math.sin(a0) * R, W_FINE, 0.075)


def mm_07_bucket(p):
    """Flood fill: paint bucket pouring mask into a cell."""
    b = blob(0.36, 0.735, 0.265, 0.235, seed=71, jitter=0.12)
    fill(p, b.intersected(rect(-0.05, 0.745, 1.1, 0.5)))
    pen(p, W_MAIN)
    p.drawPath(b)
    t = QTransform()
    t.translate(0.665, 0.285)
    t.rotate(-128.0)
    body = QPainterPath()
    body.moveTo(-0.200, -0.115)
    body.lineTo(-0.135, 0.195)
    body.cubicTo(-0.055, 0.225, 0.055, 0.225, 0.135, 0.195)
    body.lineTo(0.200, -0.115)
    pen(p, W_MAIN)
    p.drawPath(t.map(body))
    p.drawPath(t.map(ell(0.0, -0.115, 0.200, 0.062)))
    hp = QPainterPath()
    hp.moveTo(-0.175, -0.150)
    hp.cubicTo(-0.070, -0.385, 0.070, -0.385, 0.175, -0.150)
    pen(p, W_FINE)
    p.drawPath(t.map(hp))
    stream = QPainterPath()
    stream.moveTo(0.540, 0.380)
    stream.cubicTo(0.496, 0.490, 0.458, 0.575, 0.440, 0.685)
    stream.lineTo(0.332, 0.663)
    stream.cubicTo(0.358, 0.558, 0.394, 0.462, 0.412, 0.332)
    stream.closeSubpath()
    knockout_halo(p, [stream], 0.040)
    fill(p, stream)
    dot(p, 0.255, 0.600, 0.019)
    dot(p, 0.205, 0.685, 0.013)


def mm_08_layers(p):
    """Mask layer lifted off the image layer."""
    DX, DY = 0.285, -0.325
    base = rect(0.075, 0.415, 0.545, 0.505, 0.045)
    pen(p, W_FINE)
    p.drawPath(base)
    p.save()
    p.setClipPath(base)
    pen(p, W_HAIR)
    p.drawPath(blob(0.255, 0.635, 0.135, 0.128, seed=81, jitter=0.13))
    p.drawPath(ell(0.245, 0.650, 0.045))
    p.drawPath(blob(0.455, 0.805, 0.095, 0.090, seed=82, jitter=0.13))
    p.drawPath(ell(0.450, 0.812, 0.032))
    p.restore()
    top = rot_path(rect(0.075 + DX, 0.415 + DY, 0.545, 0.505, 0.045),
                   0.62, 0.34, -8.0)
    pen(p, W_MAIN)
    p.drawPath(top)
    p.save()
    p.setClipPath(top)
    fill(p, rot_path(blob(0.255 + DX, 0.635 + DY, 0.135, 0.128,
                          seed=81, jitter=0.13), 0.62, 0.34, -8.0))
    fill(p, rot_path(blob(0.455 + DX, 0.805 + DY, 0.095, 0.090,
                          seed=82, jitter=0.13), 0.62, 0.34, -8.0))
    p.restore()
    pen(p, W_HAIR)
    for i in range(3):
        y = 0.215 + i * 0.075
        p.drawPath(line(0.075, y, 0.245, y))


def mm_09_pencil(p):
    """Pencil hand-drawing a contour: dashed ahead, solid behind."""
    pts = blob_pts(0.44, 0.50, 0.30, 0.29, seed=91, n=13, jitter=0.11)
    full = catmull_closed(pts)
    done = QPainterPath()
    done.moveTo(pts[0][0], pts[0][1])
    n = len(pts)
    for i in range(0, 9):
        p0 = pts[(i - 1) % n]
        p1 = pts[i % n]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6.0, p1[1] + (p2[1] - p0[1]) / 6.0)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6.0, p2[1] - (p3[1] - p1[1]) / 6.0)
        done.cubicTo(c1[0], c1[1], c2[0], c2[1], p2[0], p2[1])
    pen(p, W_HAIR, dash=[2.4, 2.2])
    p.drawPath(full)
    pen(p, W_MAIN)
    p.drawPath(done)
    fill(p, ell(0.42, 0.52, 0.062))
    tipx, tipy = pts[9 % n]
    t = QTransform()
    t.translate(tipx, tipy)
    t.rotate(-56.0)
    fill(p, t.map(poly([(0.0, 0.0), (0.115, -0.055), (0.115, 0.055)])))
    pen(p, W_FINE)
    p.drawPath(t.map(line(0.115, -0.055, 0.115, 0.055)))
    p.drawPath(t.map(rect(0.115, -0.055, 0.34, 0.11, 0.012)))
    p.drawPath(t.map(line(0.30, -0.055, 0.30, 0.055)))
    p.drawPath(t.map(line(0.175, -0.055, 0.175, 0.055)))


def mm_10_wand(p):
    """Magic wand selecting a cell: marching-ants outline + sparkles."""
    pen(p, W_MAIN, dash=[2.1, 1.7])
    p.drawPath(blob(0.40, 0.63, 0.28, 0.27, seed=101, jitter=0.12))
    fill(p, ell(0.38, 0.65, 0.070))
    t = QTransform()
    t.translate(0.58, 0.42)
    t.rotate(-48.0)
    pen(p, W_FINE)
    p.drawPath(t.map(poly([(0.03, -0.028), (0.44, -0.017),
                           (0.44, 0.017), (0.03, 0.028)])))
    fill(p, t.map(rect(0.30, -0.024, 0.055, 0.048, 0.008)))
    sparkle(p, 0.575, 0.415, 0.085)
    sparkle(p, 0.735, 0.185, 0.048)
    sparkle(p, 0.845, 0.335, 0.033)
    sparkle(p, 0.185, 0.245, 0.038)


# =========================================================================== #
#  mask  --  AUTOMATIC segmentation
# =========================================================================== #
def mk_01_before_after(p):
    """Raw image half vs segmented half of the same field."""
    frame = rect(0.07, 0.16, 0.86, 0.68, 0.03)
    pen(p, W_FINE)
    p.drawPath(frame)
    pen(p, W_HAIR, dash=[2.2, 2.0])
    p.drawPath(line(0.50, 0.16, 0.50, 0.84))
    specs = [(0.21, 0.34, 0.090, 1), (0.30, 0.615, 0.105, 2), (0.395, 0.335, 0.058, 3)]
    p.save()
    p.setClipPath(frame)
    for cx, cy, r, s in specs:
        fill(p, blob(cx, cy, r, r * 0.93, seed=110 + s, jitter=0.16))
    for cx, cy, r, s in specs:
        pen(p, W_FINE)
        p.drawPath(blob(cx + 0.43, cy, r, r * 0.93, seed=110 + s, jitter=0.16))
        fill(p, ell(cx + 0.43, cy, r * 0.30))
    p.restore()


def mk_02_model(p):
    """A trained model turning an image into contours."""
    cols = [(0.115, 3), (0.245, 4), (0.375, 3)]
    nodes = []
    for x, k in cols:
        col = []
        for i in range(k):
            y = 0.50 + (i - (k - 1) / 2.0) * 0.155
            col.append((x, y))
        nodes.append(col)
    pen(p, W_HAIR)
    for a, b in zip(nodes, nodes[1:]):
        for xa, ya in a:
            for xb, yb in b:
                p.drawPath(line(xa, ya, xb, yb))
    for col in nodes:
        for x, y in col:
            p.save()
            p.setCompositionMode(QPainter.CompositionMode_Clear)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 255))
            p.drawPath(ell(x, y, 0.036))
            p.restore()
            pen(p, W_FINE)
            p.drawPath(ell(x, y, 0.032))
    arrow(p, 0.455, 0.50, 0.585, 0.50, W_FINE, 0.055)
    pen(p, W_MAIN)
    p.drawPath(blob(0.77, 0.50, 0.185, 0.180, seed=121, jitter=0.12))
    fill(p, ell(0.755, 0.52, 0.055))
    pen(p, W_HAIR)
    p.drawPath(blob(0.87, 0.235, 0.085, 0.082, seed=122, jitter=0.14))
    p.drawPath(blob(0.83, 0.775, 0.095, 0.090, seed=123, jitter=0.14))


def mk_03_watershed(p):
    """Touching cells split by watershed boundaries inside a field of view."""
    field = ell(0.50, 0.52, 0.415, 0.415)
    pen(p, W_FINE)
    p.drawPath(field)
    p.save()
    p.setClipPath(field)
    centers = [(0.50, 0.52, 0.155)]
    for i in range(6):
        a = math.radians(-90 + 60 * i)
        centers.append((0.50 + math.cos(a) * 0.275,
                        0.52 + math.sin(a) * 0.275, 0.155))
    for i, (cx, cy, r) in enumerate(centers):
        pen(p, W_MAIN)
        p.drawPath(blob(cx, cy, r, r * 0.95, seed=130 + i, jitter=0.10))
        fill(p, ell(cx, cy + r * 0.05, r * 0.30))
    p.restore()


def mk_04_labels(p):
    """Instance labels: each detected object filled and numbered."""
    specs = [(0.29, 0.30, 0.175, 1), (0.70, 0.28, 0.150, 2),
             (0.27, 0.71, 0.150, 3), (0.68, 0.70, 0.180, 4)]
    for cx, cy, r, k in specs:
        fill(p, blob(cx, cy, r, r * 0.94, seed=140 + k, jitter=0.14))
    for cx, cy, r, k in specs:
        rr = r * 0.135
        if k == 1:
            pts = [(cx, cy)]
        else:
            span = math.radians(38 * (k - 1))
            pts = []
            for i in range(k):
                a = math.pi / 2 + span * (i / (k - 1.0) - 0.5)
                pts.append((cx + math.cos(a) * r * 0.44,
                            cy - r * 0.10 + math.sin(a) * r * 0.44))
        for px, py in pts:
            clear(p, ell(px, py, rr))


def mk_05_active_contour(p):
    """Active contour shrinking onto a cell."""
    for k, dash in ((1.62, [2.4, 2.2]), (1.30, [2.4, 2.2])):
        pen(p, W_HAIR, dash=dash)
        p.drawPath(blob_scaled(0.50, 0.53, 0.235, 0.230, 151, k, jitter=0.13))
    pen(p, W_MAIN)
    p.drawPath(blob_scaled(0.50, 0.53, 0.235, 0.230, 151, 1.0, jitter=0.13))
    fill(p, ell(0.485, 0.555, 0.065))
    for a in (-90, -20, 45, 135, 200):
        ar = math.radians(a)
        x1 = 0.50 + math.cos(ar) * 0.435
        y1 = 0.53 + math.sin(ar) * 0.425
        x2 = 0.50 + math.cos(ar) * 0.315
        y2 = 0.53 + math.sin(ar) * 0.308
        arrow(p, x1, y1, x2, y2, W_HAIR, 0.045)


def mk_06_nested(p):
    """Nested multi-class masks: cell, nucleus, pathogen vacuole."""
    pen(p, W_MAIN)
    p.drawPath(blob(0.50, 0.52, 0.405, 0.395, seed=161, n=13, jitter=0.09))
    pen(p, W_FINE)
    p.drawPath(blob(0.355, 0.375, 0.150, 0.145, seed=162, jitter=0.10))
    fill(p, ell(0.355, 0.375, 0.045))
    pen(p, W_FINE)
    p.drawPath(blob(0.615, 0.660, 0.175, 0.170, seed=163, jitter=0.09))
    rng = random.Random(7)
    for i in range(4):
        a = math.radians(-90 + 90 * i + 22)
        px = 0.615 + math.cos(a) * 0.085
        py = 0.660 + math.sin(a) * 0.082
        e = rot_path(ell(px, py, 0.062, 0.026), px, py,
                     math.degrees(a) + 90 + rng.uniform(-12, 12))
        fill(p, e)
    pen(p, W_HAIR)
    p.drawPath(ell(0.245, 0.700, 0.055))
    p.drawPath(ell(0.700, 0.300, 0.042))


def mk_07_pixels(p):
    """Pixel-wise labelling: contour rasterised onto the image grid."""
    x0, y0, w_, N = 0.10, 0.10, 0.80, 11
    step = w_ / N
    b = blob(0.50, 0.50, 0.305, 0.295, seed=171, jitter=0.12)
    for j in range(N):
        for i in range(N):
            cx = x0 + step * (i + 0.5)
            cy = y0 + step * (j + 0.5)
            if b.contains(QPointF(cx, cy)):
                fill(p, rect(cx - step * 0.40, cy - step * 0.40,
                             step * 0.80, step * 0.80, step * 0.10))
    pen(p, W_HAIR)
    for i in range(N + 1):
        p.drawPath(line(x0 + step * i, y0, x0 + step * i, y0 + w_))
        p.drawPath(line(x0, y0 + step * i, x0 + w_, y0 + step * i))
    pen(p, W_MAIN)
    p.drawPath(b)


def mk_08_threshold(p):
    """Intensity threshold turning a histogram into a binary mask."""
    base = 0.46
    heights = [0.05, 0.10, 0.20, 0.31, 0.26, 0.14, 0.08, 0.11, 0.20, 0.15, 0.07]
    x0, w_ = 0.09, 0.82
    bw = w_ / len(heights)
    for i, h in enumerate(heights):
        fill(p, rect(x0 + bw * i + bw * 0.12, base - h, bw * 0.76, h))
    pen(p, W_FINE)
    p.drawPath(line(0.07, base, 0.93, base))
    pen(p, W_FINE, dash=[2.2, 2.0])
    p.drawPath(line(0.565, 0.055, 0.565, 0.52))
    arrow(p, 0.50, 0.545, 0.50, 0.635, W_FINE, 0.05)
    fill(p, blob(0.475, 0.795, 0.155, 0.150, seed=181, jitter=0.14))
    clear(p, ell(0.455, 0.815, 0.048))
    fill(p, blob(0.735, 0.845, 0.095, 0.092, seed=182, jitter=0.14))
    clear(p, ell(0.745, 0.855, 0.030))


def mk_09_detect(p):
    """Automatic detection: corner brackets snapped onto each object."""
    specs = [(0.30, 0.31, 0.150), (0.70, 0.35, 0.115), (0.50, 0.72, 0.170)]
    for i, (cx, cy, r) in enumerate(specs):
        pen(p, W_FINE)
        p.drawPath(blob(cx, cy, r, r * 0.93, seed=190 + i, jitter=0.13))
        fill(p, ell(cx, cy + r * 0.04, r * 0.28))
        m = r * 1.30
        L = m * 0.55
        for sx in (-1, 1):
            for sy in (-1, 1):
                ax, ay = cx + sx * m, cy + sy * m
                pen(p, W_MAIN)
                p.drawPath(line(ax, ay, ax - sx * L, ay))
                p.drawPath(line(ax, ay, ax, ay - sy * L))


def mk_10_stencil(p):
    """A stencil: solid mask sheet with the objects punched out."""
    plate_ = rect(0.10, 0.14, 0.80, 0.72, 0.05)
    fill(p, plate_)
    holes = [(0.32, 0.36, 0.145, 1), (0.66, 0.32, 0.115, 2),
             (0.36, 0.68, 0.110, 3), (0.66, 0.66, 0.150, 4)]
    for cx, cy, r, s in holes:
        clear(p, blob(cx, cy, r, r * 0.94, seed=200 + s, jitter=0.15))
    for cx, cy, r, s in holes:
        fill(p, ell(cx, cy, r * 0.20))
    piece = blob(0.855, 0.895, 0.105, 0.10, seed=204, jitter=0.15)
    clear(p, blob_scaled(0.855, 0.895, 0.105, 0.10, 204, 1.32, jitter=0.15))
    fill(p, piece)


# =========================================================================== #
#  map_barcodes  --  mapping sequencing barcodes to plate wells
# =========================================================================== #
def mb_01_barcode_to_plate(p):
    """A barcode read resolving to one well of a plate."""
    barcode(p, 0.09, 0.09, 0.60, 0.24, seed=301, n=10)
    pen(p, W_HAIR)
    p.drawPath(rect(0.065, 0.065, 0.65, 0.29, 0.015))
    wells = plate(p, 0.10, 0.50, 0.80, 0.40, 6, 3)
    tgt = wells[6 * 1 + 3]
    fill(p, ell(tgt[0], tgt[1], tgt[2] * 0.98))
    curved_arrow(p, 0.735, 0.21, 0.90, 0.30, tgt[0] + 0.055, tgt[1] - 0.075,
                 W_FINE, 0.052)


def mb_02_helix(p):
    """Sequence read spooling out of a DNA helix into the plate."""
    top, bot = 0.055, 0.505
    mid, amp = 0.50, 0.195
    turns = 3.30 * math.pi
    s1 = QPainterPath()
    s2 = QPainterPath()
    N = 90
    for i in range(N + 1):
        t = i / N
        y = top + (bot - top) * t
        ph = t * turns
        x1 = mid + math.sin(ph) * amp
        x2 = mid - math.sin(ph) * amp
        if i == 0:
            s1.moveTo(x1, y)
            s2.moveTo(x2, y)
        else:
            s1.lineTo(x1, y)
            s2.lineTo(x2, y)
    pen(p, W_MAIN)
    p.drawPath(s1)
    p.drawPath(s2)
    for i in range(1, 10):
        t = i / 10.0
        y = top + (bot - top) * t
        ph = t * turns
        x1 = mid + math.sin(ph) * amp
        x2 = mid - math.sin(ph) * amp
        if abs(x1 - x2) < amp * 0.95:
            continue
        pen(p, W_HAIR)
        p.drawPath(line(x1, y, x2, y))
    wells = plate(p, 0.10, 0.615, 0.80, 0.31, 6, 2)
    for k in (2, 9):
        fill(p, ell(wells[k][0], wells[k][1], wells[k][2] * 0.98))
    arrow(p, 0.50, 0.520, 0.50, 0.598, W_HAIR, 0.042)


def mb_03_row_col(p):
    """Row barcode x column barcode resolving to one well."""
    px, py, pw, ph = 0.34, 0.34, 0.58, 0.58
    wells = plate(p, px, py, pw, ph, 4, 4)
    barcode(p, px, 0.075, pw, 0.185, seed=331, n=9)
    for by, bh in barcode_bars(332, 9, py, ph):
        fill(p, rect(0.075, by, 0.185, bh))
    col, row = 2, 1
    tgt = wells[row * 4 + col]
    fill(p, ell(tgt[0], tgt[1], tgt[2] * 1.0))
    pen(p, W_HAIR, dash=[2.2, 2.0])
    p.drawPath(line(tgt[0], 0.275, tgt[0], tgt[1] - tgt[2] - 0.02))
    p.drawPath(line(0.275, tgt[1], tgt[0] - tgt[2] - 0.02, tgt[1]))
    pen(p, W_FINE)
    p.drawPath(ell(tgt[0], tgt[1], tgt[2] * 1.75))


def mb_04_wells_with_codes(p):
    """Every well carries its own barcode."""
    pen(p, W_FINE)
    p.drawPath(rect(0.07, 0.15, 0.86, 0.70, 0.045))
    cols, rows = 3, 3
    x0, y0 = 0.115, 0.195
    cw, ch = 0.77 / cols, 0.61 / rows
    for j in range(rows):
        for i in range(cols):
            cx = x0 + cw * i
            cy = y0 + ch * j
            pen(p, W_HAIR)
            p.drawPath(rect(cx, cy, cw * 0.82, ch * 0.78, cw * 0.10))
            barcode(p, cx + cw * 0.10, cy + ch * 0.16,
                    cw * 0.62, ch * 0.46, seed=340 + j * 3 + i, n=5)


def mb_05_lookup(p):
    """Lookup table: barcodes on the left, wells on the right."""
    rows = 4
    y0, dy = 0.13, 0.215
    for i in range(rows):
        y = y0 + dy * i
        pen(p, W_HAIR)
        p.drawPath(rect(0.055, y, 0.30, 0.145, 0.014))
        barcode(p, 0.075, y + 0.028, 0.26, 0.09, seed=350 + i, n=6)
    pen(p, W_FINE)
    p.drawPath(rect(0.735, 0.075, 0.21, 0.855, 0.03))
    wy = [0.175, 0.375, 0.575, 0.775]
    for y in wy:
        pen(p, W_HAIR)
        p.drawPath(ell(0.84, y, 0.062))
    order = [1, 3, 0, 2]
    for i in range(rows):
        y = y0 + dy * i + 0.0725
        pth = QPainterPath()
        pth.moveTo(0.365, y)
        pth.cubicTo(0.52, y, 0.60, wy[order[i]], 0.765, wy[order[i]])
        pen(p, W_HAIR)
        p.drawPath(pth)
    fill(p, ell(0.84, wy[order[1]], 0.062))


def mb_06_pipette(p):
    """A barcoded sample being dispensed into a well."""
    pen(p, W_FINE)
    p.drawPath(rect(0.468, 0.028, 0.064, 0.075, 0.015))
    p.drawPath(line(0.50, 0.103, 0.50, 0.135))
    body = poly([(0.408, 0.135), (0.592, 0.135), (0.562, 0.285),
                 (0.535, 0.345), (0.465, 0.345), (0.438, 0.285)])
    pen(p, W_MAIN)
    p.drawPath(body)
    pen(p, W_FINE)
    p.drawPath(line(0.422, 0.215, 0.578, 0.215))
    fill(p, poly([(0.465, 0.350), (0.535, 0.350), (0.518, 0.435),
                  (0.482, 0.435)]))
    drop = QPainterPath()
    drop.moveTo(0.50, 0.455)
    drop.cubicTo(0.635, 0.585, 0.645, 0.700, 0.50, 0.715)
    drop.cubicTo(0.355, 0.700, 0.365, 0.585, 0.50, 0.455)
    drop.closeSubpath()
    fill(p, drop)
    for bx, bw in barcode_bars(361, 5, 0.415, 0.170):
        clear(p, rect(bx, 0.560, bw * 0.90, 0.110))
    wells = plate(p, 0.10, 0.755, 0.80, 0.195, 6, 2)
    pen(p, W_FINE)
    p.drawPath(ell(wells[3][0], wells[3][1], wells[3][2] * 1.7))


def mb_07_read_segments(p):
    """One read, three barcode segments, three plate coordinates."""
    pen(p, W_MAIN)
    p.drawPath(rect(0.06, 0.09, 0.88, 0.155, 0.075))
    xs = [0.06, 0.353, 0.647, 0.94]
    pen(p, W_FINE)
    for x in xs[1:3]:
        p.drawPath(line(x, 0.09, x, 0.245))
    for k in range(3):
        a, b = xs[k], xs[k + 1]
        if k == 0:
            barcode(p, a + 0.035, 0.125, (b - a) - 0.07, 0.085, seed=371, n=5)
        elif k == 1:
            rng = random.Random(372)
            for i in range(6):
                dot(p, a + 0.045 + i * ((b - a) - 0.09) / 5.0,
                    0.1675 + rng.uniform(-0.018, 0.018), 0.019)
        else:
            pen(p, W_HAIR)
            for i in range(6):
                x = a + 0.045 + i * ((b - a) - 0.09) / 5.0
                p.drawPath(line(x, 0.215, x + 0.045, 0.125))
    wells = plate(p, 0.10, 0.58, 0.80, 0.34, 5, 2)
    targets = [wells[0], wells[7], wells[4]]
    for k, tg in enumerate(targets):
        sx = (xs[k] + xs[k + 1]) / 2
        curved_arrow(p, sx, 0.275, sx, 0.44, tg[0], tg[1] - tg[2] - 0.055,
                     W_HAIR, 0.048)
        fill(p, ell(tg[0], tg[1], tg[2] * 0.98))


def mb_08_pin(p):
    """The plate read as a map; a pin drops on the matched well."""
    wells = plate(p, 0.07, 0.30, 0.86, 0.58, 6, 4)
    tgt = wells[6 * 1 + 2]
    pin = QPainterPath()
    pin.moveTo(tgt[0], tgt[1] + 0.035)
    pin.cubicTo(tgt[0] - 0.075, tgt[1] - 0.085,
                tgt[0] - 0.105, tgt[1] - 0.205, tgt[0], tgt[1] - 0.235)
    pin.cubicTo(tgt[0] + 0.105, tgt[1] - 0.205,
                tgt[0] + 0.075, tgt[1] - 0.085, tgt[0], tgt[1] + 0.035)
    pin.closeSubpath()
    fill(p, pin)
    clear(p, ell(tgt[0], tgt[1] - 0.135, 0.045))
    fill(p, ell(tgt[0], tgt[1], tgt[2] * 0.55))
    barcode(p, 0.20, 0.065, 0.60, 0.145, seed=381, n=9)
    pen(p, W_HAIR, dash=[2.0, 1.9])
    p.drawPath(line(0.50, 0.235, tgt[0], tgt[1] - 0.265))


def mb_09_key(p):
    """The barcode is the key that opens one well."""
    pen(p, W_MAIN)
    p.drawPath(ell(0.155, 0.50, 0.115, 0.115))
    fill(p, rect(0.155, 0.472, 0.42, 0.056, 0.012))
    teeth = [(0.40, 0.075), (0.45, 0.115), (0.505, 0.075)]
    for tx, th in teeth:
        fill(p, rect(tx, 0.528, 0.036, th, 0.008))
    pen(p, W_HAIR)
    p.drawPath(ell(0.155, 0.50, 0.048))
    pen(p, W_FINE)
    p.drawPath(rect(0.60, 0.175, 0.335, 0.65, 0.045))
    for j in range(3):
        for i in range(2):
            cx = 0.665 + i * 0.205
            cy = 0.275 + j * 0.225
            if (i, j) == (0, 1):
                continue
            pen(p, W_HAIR)
            p.drawPath(ell(cx, cy, 0.062))
    fill(p, ell(0.665, 0.50, 0.088))
    kh = QPainterPath()
    kh.addEllipse(QRectF(0.665 - 0.030, 0.50 - 0.045, 0.060, 0.060))
    kh.addRect(QRectF(0.665 - 0.017, 0.50 - 0.005, 0.034, 0.062))
    clear(p, kh.simplified())


def mb_10_flowcell(p):
    """Cluster tile off the sequencer mapped onto the plate."""
    tile = rect(0.06, 0.07, 0.52, 0.40, 0.03)
    pen(p, W_FINE)
    p.drawPath(tile)
    pen(p, W_HAIR)
    for i in range(1, 4):
        p.drawPath(line(0.06 + 0.52 * i / 4.0, 0.07, 0.06 + 0.52 * i / 4.0, 0.47))
    rng = random.Random(391)
    for _ in range(54):
        x = rng.uniform(0.075, 0.565)
        y = rng.uniform(0.085, 0.455)
        dot(p, x, y, rng.choice([0.010, 0.013, 0.016]))
    wells = plate(p, 0.40, 0.53, 0.55, 0.40, 4, 3)
    for k in (1, 6, 10):
        fill(p, ell(wells[k][0], wells[k][1], wells[k][2] * 0.98))
    curved_arrow(p, 0.605, 0.30, 0.72, 0.36, 0.72, 0.485, W_FINE, 0.055)


# =========================================================================== #
#  measure  --  quantifying objects
# =========================================================================== #
def _measure_cell(p, cx, cy, r, seed=411):
    """The organelle-rich cell used in the existing measure.png."""
    pen(p, W_MAIN)
    p.drawPath(ell(cx, cy, r))
    pen(p, W_FINE)
    p.drawPath(ell(cx - r * 0.05, cy + r * 0.05, r * 0.30))
    pen(p, W_HAIR)
    p.drawPath(ell(cx - r * 0.10, cy - r * 0.55, r * 0.09))
    fill(p, ell(cx - r * 0.42, cy - r * 0.50, r * 0.045))
    for a, d in ((-35, 0.62), (60, 0.60), (170, 0.62)):
        ar = math.radians(a)
        mito(p, cx + math.cos(ar) * r * d, cy + math.sin(ar) * r * d,
             r * 0.44, r * 0.24, a + 90)
    for k, rr in enumerate((0.40, 0.52, 0.64)):
        pen(p, W_HAIR)
        p.drawPath(arc_path(cx, cy, r * rr, 150 + k * 6, 215 - k * 6))
    pen(p, W_HAIR)
    p.drawPath(arc_path(cx, cy, r * 0.72, 20, 55))
    p.drawPath(line(cx + r * 0.20, cy + r * 0.34, cx + r * 0.32, cy + r * 0.48))


def me_01_ruler(p):
    """Ruler laid over a cell (the existing icon's idea)."""
    _measure_cell(p, 0.605, 0.605, 0.365)
    pen(p, W_FINE)
    edge = QPainterPath()
    edge.moveTo(0.955, 0.095)
    edge.lineTo(0.145, 0.095)
    edge.lineTo(0.095, 0.145)
    edge.lineTo(0.095, 0.955)
    p.drawPath(edge)
    for i in range(9):
        x = 0.195 + i * 0.095
        L = 0.085 if i % 4 == 0 else 0.050
        pen(p, W_FINE)
        p.drawPath(line(x, 0.095, x, 0.095 + L))
    for i in range(9):
        y = 0.195 + i * 0.095
        L = 0.085 if i % 4 == 0 else 0.050
        pen(p, W_FINE)
        p.drawPath(line(0.095, y, 0.095 + L, y))


def me_02_calipers(p):
    """Vernier calipers closed onto a cell."""
    cx, cy, r = 0.505, 0.395, 0.195
    cell(p, cx, cy, r, seed=421)
    pen(p, W_MAIN)
    p.drawPath(rect(0.065, 0.700, 0.87, 0.095, 0.022))
    for i in range(15):
        x = 0.095 + i * 0.058
        L = 0.048 if i % 4 == 0 else 0.028
        pen(p, W_HAIR)
        p.drawPath(line(x, 0.700, x, 0.700 + L))
    jaw_l = poly([(0.212, 0.700), (0.212, 0.360), (0.272, 0.175),
                  (0.302, 0.175), (0.302, 0.700)])
    jaw_r = poly([(0.798, 0.700), (0.798, 0.360), (0.738, 0.175),
                  (0.708, 0.175), (0.708, 0.700)])
    knockout_halo(p, [jaw_l, jaw_r], 0.036)
    pen(p, W_MAIN)
    p.drawPath(jaw_l)
    p.drawPath(jaw_r)
    pen(p, W_FINE)
    p.drawPath(rect(0.660, 0.665, 0.265, 0.235, 0.028))
    pen(p, W_HAIR)
    for i in range(5):
        p.drawPath(line(0.700 + i * 0.050, 0.815, 0.700 + i * 0.050, 0.878))
    pen(p, W_HAIR, dash=[2.2, 2.0])
    p.drawPath(line(0.302, 0.115, 0.708, 0.115))
    for x in (0.302, 0.708):
        pen(p, W_HAIR)
        p.drawPath(line(x, 0.085, x, 0.148))


def me_03_scalebar(p):
    """Object plus the scale bar that calibrates it."""
    cell(p, 0.50, 0.415, 0.315, seed=431, nucleus=0.30)
    pen(p, W_HAIR)
    p.drawPath(ell(0.62, 0.30, 0.055))
    mito(p, 0.36, 0.28, 0.155, 0.082, 22)
    mito(p, 0.66, 0.55, 0.155, 0.082, -35)
    fill(p, rect(0.145, 0.815, 0.71, 0.055, 0.012))
    for i in range(5):
        x = 0.145 + i * 0.1775
        fill(p, rect(x - 0.012, 0.755, 0.024, 0.062))
    pen(p, W_HAIR)
    p.drawPath(line(0.145, 0.915, 0.325, 0.915))
    p.drawPath(line(0.365, 0.915, 0.455, 0.915))


def me_04_bbox(p):
    """Bounding box with dimension callouts."""
    cx, cy = 0.475, 0.435
    cell(p, cx, cy, 0.245, seed=441)
    x0, x1 = 0.19, 0.76
    y0, y1 = 0.145, 0.725
    pen(p, W_FINE, dash=[2.2, 2.0])
    p.drawPath(rect(x0, y0, x1 - x0, y1 - y0))
    dy = 0.855
    pen(p, W_HAIR)
    p.drawPath(line(x0, y1 + 0.02, x0, dy + 0.045))
    p.drawPath(line(x1, y1 + 0.02, x1, dy + 0.045))
    arrow(p, (x0 + x1) / 2, dy, x0 + 0.012, dy, W_HAIR, 0.045)
    arrow(p, (x0 + x1) / 2, dy, x1 - 0.012, dy, W_HAIR, 0.045)
    dx = 0.895
    pen(p, W_HAIR)
    p.drawPath(line(x1 + 0.02, y0, dx + 0.045, y0))
    p.drawPath(line(x1 + 0.02, y1, dx + 0.045, y1))
    arrow(p, dx, (y0 + y1) / 2, dx, y0 + 0.012, W_HAIR, 0.045)
    arrow(p, dx, (y0 + y1) / 2, dx, y1 - 0.012, W_HAIR, 0.045)
    for sx, sy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
        fill(p, ell(sx, sy, 0.020))


def me_05_contours(p):
    """Intensity contours with a radial profile line."""
    cx, cy = 0.50, 0.50
    for k in (1.0, 0.78, 0.56, 0.34):
        w = W_MAIN if k == 1.0 else W_HAIR
        pen(p, w)
        p.drawPath(blob_scaled(cx, cy, 0.38, 0.365, 451, k, n=13, jitter=0.10))
    fill(p, ell(cx, cy, 0.055, 0.052))
    ar = math.radians(-32)
    ex = cx + math.cos(ar) * 0.415
    ey = cy + math.sin(ar) * 0.40
    pen(p, W_FINE)
    p.drawPath(line(cx, cy, ex, ey))
    for k in (0.34, 0.56, 0.78, 1.0):
        px = cx + math.cos(ar) * 0.38 * k
        py = cy + math.sin(ar) * 0.365 * k
        nx, ny = -math.sin(ar), math.cos(ar)
        pen(p, W_HAIR)
        p.drawPath(line(px - nx * 0.035, py - ny * 0.035,
                        px + nx * 0.035, py + ny * 0.035))


def me_06_histogram(p):
    """Population statistics: cells counted into a histogram."""
    x0, y0, w_, h_ = 0.10, 0.13, 0.82, 0.70
    pen(p, W_FINE)
    p.drawPath(line(x0, y0 + h_, x0 + w_, y0 + h_))
    p.drawPath(line(x0, y0, x0, y0 + h_))
    counts = [1, 3, 4, 2]
    bw = (w_ - 0.06) / len(counts)
    r = bw * 0.30
    for i, c in enumerate(counts):
        bx = x0 + 0.05 + bw * i
        for j in range(c):
            cy = y0 + h_ - r * 1.20 - j * (r * 2.25)
            pen(p, W_FINE)
            p.drawPath(blob(bx + bw * 0.42, cy, r, r * 0.94,
                            seed=460 + i * 5 + j, jitter=0.12))
            fill(p, ell(bx + bw * 0.42, cy, r * 0.30))
    for i in range(len(counts)):
        pen(p, W_HAIR)
        x = x0 + 0.05 + bw * (i + 0.42)
        p.drawPath(line(x, y0 + h_, x, y0 + h_ + 0.045))


def me_07_micrometer(p):
    """Micrometer screw gauge measuring one cell."""
    frame = QPainterPath()
    frame.moveTo(0.605, 0.155)
    frame.lineTo(0.330, 0.155)
    frame.cubicTo(0.115, 0.185, 0.115, 0.815, 0.330, 0.845)
    frame.lineTo(0.605, 0.845)
    pen(p, 0.030)
    p.drawPath(frame)
    fill(p, rect(0.205, 0.435, 0.062, 0.130, 0.012))
    cell(p, 0.385, 0.500, 0.108, seed=471, w=W_FINE, nucleus=0.36)
    fill(p, rect(0.495, 0.472, 0.135, 0.056, 0.012))
    pen(p, W_MAIN)
    p.drawPath(rect(0.615, 0.400, 0.185, 0.200, 0.028))
    pen(p, W_HAIR)
    for i in range(5):
        x = 0.645 + i * 0.036
        p.drawPath(line(x, 0.420, x, 0.500))
    pen(p, W_MAIN)
    p.drawPath(rect(0.800, 0.375, 0.135, 0.250, 0.035))
    pen(p, W_HAIR)
    for i in range(4):
        y = 0.420 + i * 0.055
        p.drawPath(line(0.815, y, 0.920, y))
    fill(p, rect(0.795, 0.375, 0.030, 0.250, 0.012))


def me_08_crosshair(p):
    """Crosshair readout: object position on calibrated axes."""
    cx, cy = 0.585, 0.415
    cell(p, cx, cy, 0.215, seed=481)
    pen(p, W_FINE)
    p.drawPath(line(0.115, 0.905, 0.955, 0.905))
    p.drawPath(line(0.115, 0.905, 0.115, 0.075))
    for i in range(8):
        pen(p, W_HAIR)
        p.drawPath(line(0.185 + i * 0.098, 0.905, 0.185 + i * 0.098, 0.868))
        p.drawPath(line(0.115, 0.835 - i * 0.096, 0.152, 0.835 - i * 0.096))
    pen(p, W_HAIR, dash=[2.2, 2.0])
    p.drawPath(line(cx, cy, cx, 0.905))
    p.drawPath(line(cx, cy, 0.115, cy))
    pen(p, W_MAIN)
    p.drawPath(ell(cx, cy, 0.075))
    p.drawPath(line(cx - 0.145, cy, cx - 0.045, cy))
    p.drawPath(line(cx + 0.045, cy, cx + 0.145, cy))
    p.drawPath(line(cx, cy - 0.145, cx, cy - 0.045))
    p.drawPath(line(cx, cy + 0.045, cx, cy + 0.145))
    fill(p, poly([(cx - 0.030, 0.905), (cx + 0.030, 0.905), (cx, 0.855)]))
    fill(p, poly([(0.115, cy - 0.030), (0.115, cy + 0.030), (0.165, cy)]))


def me_09_table(p):
    """Extracted feature table beside the object."""
    cell(p, 0.245, 0.485, 0.195, seed=491)
    x0, y0, w_, h_ = 0.505, 0.135, 0.435, 0.735
    pen(p, W_FINE)
    p.drawPath(rect(x0, y0, w_, h_, 0.022))
    rows = 5
    rh = h_ / rows
    pen(p, W_HAIR)
    for i in range(1, rows):
        p.drawPath(line(x0, y0 + rh * i, x0 + w_, y0 + rh * i))
    p.drawPath(line(x0 + w_ * 0.52, y0, x0 + w_ * 0.52, y0 + h_))
    fill(p, rect(x0, y0, w_, rh, 0.022).intersected(
        rect(x0, y0, w_, rh * 0.72)))
    rng = random.Random(492)
    for i in range(1, rows):
        cy = y0 + rh * (i + 0.5)
        fill(p, rect(x0 + w_ * 0.08, cy - 0.016,
                     w_ * 0.34 * rng.uniform(0.7, 1.0), 0.032, 0.014))
        fill(p, rect(x0 + w_ * 0.60, cy - 0.016,
                     w_ * 0.30 * rng.uniform(0.6, 1.0), 0.032, 0.014))
    pen(p, W_HAIR)
    pth = QPainterPath()
    pth.moveTo(0.445, 0.455)
    pth.cubicTo(0.475, 0.44, 0.475, 0.34, 0.50, 0.285)
    p.drawPath(pth)
    fill(p, ell(0.445, 0.455, 0.020))


def me_10_protractor(p):
    """Protractor measuring an angle across the object."""
    cx, cy, R = 0.50, 0.735, 0.415
    pen(p, W_MAIN)
    p.drawPath(arc_path(cx, cy, R, 180, 360))
    p.drawPath(line(cx - R, cy, cx + R, cy))
    for i in range(19):
        a = math.radians(180 + i * 10)
        L = R * (0.135 if i % 3 == 0 else 0.075)
        pen(p, W_HAIR)
        p.drawPath(line(cx + math.cos(a) * R, cy + math.sin(a) * R,
                        cx + math.cos(a) * (R - L), cy + math.sin(a) * (R - L)))
    pen(p, W_HAIR)
    p.drawPath(blob(0.50, 0.545, 0.185, 0.175, seed=501, jitter=0.12))
    fill(p, ell(0.485, 0.565, 0.052))
    a1, a2 = math.radians(212), math.radians(324)
    pen(p, W_FINE)
    p.drawPath(line(cx, cy, cx + math.cos(a1) * R * 0.92,
                    cy + math.sin(a1) * R * 0.92))
    p.drawPath(line(cx, cy, cx + math.cos(a2) * R * 0.92,
                    cy + math.sin(a2) * R * 0.92))
    pen(p, W_HAIR, dash=[2.2, 2.0])
    p.drawPath(arc_path(cx, cy, R * 0.42, 212, 324))
    fill(p, ell(cx, cy, 0.028))


# =========================================================================== #
#  registry
# =========================================================================== #
GROUPS = {
    "make_masks": [
        ("Paintbrush laying mask paint inside a cell outline (half painted)",
         mm_01_brush),
        ("Eraser lifting a wrong region off an existing mask", mm_02_eraser),
        ("Scalpel splitting an over-merged doublet along a cut line",
         mm_03_split),
        ("Merging two masks into one: shared boundary dissolves", mm_04_merge),
        ("Polygon lasso with draggable vertex handles and a cursor",
         mm_05_lasso),
        ("Edit history: ghost previous outlines under an undo arc", mm_06_undo),
        ("Paint-bucket flood fill pouring mask into a cell", mm_07_bucket),
        ("Mask layer lifted off the image layer (layer stack)", mm_08_layers),
        ("Pencil hand-drawing a contour: dashed ahead, solid behind",
         mm_09_pencil),
        ("Magic wand selecting a cell: marching ants plus sparkles",
         mm_10_wand),
    ],
    "mask": [
        ("Split field: raw objects on one side, contours on the other",
         mk_01_before_after),
        ("Trained model turning an image into contours (network to cell)",
         mk_02_model),
        ("Watershed: touching cells separated inside a field of view",
         mk_03_watershed),
        ("Instance labels: every object filled and given an ID", mk_04_labels),
        ("Active contour shrinking onto a cell", mk_05_active_contour),
        ("Nested classes: cell, nucleus and pathogen vacuole", mk_06_nested),
        ("Pixel-wise labelling: contour rasterised onto the image grid",
         mk_07_pixels),
        ("Intensity threshold turning a histogram into a binary silhouette",
         mk_08_threshold),
        ("Automatic detection: corner brackets snapped onto each object",
         mk_09_detect),
        ("Stencil sheet with the objects punched out", mk_10_stencil),
    ],
    "map_barcodes": [
        ("Barcode read resolving to a single well of a plate",
         mb_01_barcode_to_plate),
        ("Sequence spooling out of a DNA helix into plate wells", mb_02_helix),
        ("Row barcode x column barcode intersecting on one well", mb_03_row_col),
        ("Every well carrying its own barcode", mb_04_wells_with_codes),
        ("Lookup table: barcodes on the left wired to wells on the right",
         mb_05_lookup),
        ("Pipette dispensing a barcoded sample into a well", mb_06_pipette),
        ("One read, three barcode segments, three plate coordinates",
         mb_07_read_segments),
        ("Plate as a map with a location pin dropped on the matched well",
         mb_08_pin),
        ("Barcode as the key that opens one well", mb_09_key),
        ("Sequencer cluster tile mapped onto the plate", mb_10_flowcell),
    ],
    "measure": [
        ("Ruler laid over a cell (the existing icon's idea)", me_01_ruler),
        ("Vernier calipers closed onto a cell", me_02_calipers),
        ("Object above a calibrated scale bar", me_03_scalebar),
        ("Bounding box with width and height dimension callouts", me_04_bbox),
        ("Intensity contour rings with a radial profile line", me_05_contours),
        ("Population histogram whose bars are stacks of cells",
         me_06_histogram),
        ("Micrometer screw gauge closed on a cell", me_07_micrometer),
        ("Crosshair readout: object position on calibrated axes",
         me_08_crosshair),
        ("Extracted feature table beside the object", me_09_table),
        ("Protractor measuring an angle across the object", me_10_protractor),
    ],
}


# --------------------------------------------------------------------------- #
# rendering / contact sheets
# --------------------------------------------------------------------------- #
def render(fn, size=SIZE):
    img, p = new_canvas(size)
    try:
        fn(p)
    finally:
        p.end()
    return img


def alpha_coverage(img):
    buf = img.constBits().tobytes()
    w, h = img.width(), img.height()
    stride = img.bytesPerLine()
    opaque = 0
    for y in range(h):
        row = buf[y * stride:y * stride + w * 4]
        opaque += sum(1 for x in range(w) if row[x * 4 + 3] > 25)
    return opaque / float(w * h)


def contact_sheet(pngs, out_path, bg, fg, cols=5, ink=None):
    """Numbered contact sheet.  Each cell shows the icon at 272 px and at 48 px.

    ``ink`` recolours the artwork through its alpha channel.  The assets are pure
    white, which is invisible on a light background, so the light sheet is drawn
    in dark ink -- same shapes, judgeable contrast.
    """
    from PIL import Image, ImageDraw, ImageFont
    cw, chh = 340, 410
    rows = (len(pngs) + cols - 1) // cols
    sheet = Image.new("RGB", (cw * cols, chh * rows), bg)
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
    except Exception:
        font = ImageFont.load_default()
    for i, path in enumerate(pngs):
        r, c = divmod(i, cols)
        ox, oy = c * cw, r * chh
        im = Image.open(path).convert("RGBA")
        if ink is not None:
            tinted = Image.new("RGBA", im.size, ink + (255,))
            tinted.putalpha(im.getchannel("A"))
            im = tinted
        big = im.resize((272, 272), Image.LANCZOS)
        sheet.paste(big, (ox + 34, oy + 40), big)
        small = im.resize((48, 48), Image.LANCZOS)
        sheet.paste(small, (ox + 146, oy + 336), small)
        d.text((ox + 16, oy + 10), f"{i + 1:02d}", fill=fg, font=font)
        d.rectangle([ox + 1, oy + 1, ox + cw - 2, oy + chh - 2],
                    outline=fg, width=1)
    sheet.save(out_path)


def main(outdir=None):
    app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])
    app.processEvents()
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = outdir or os.path.abspath(os.path.join(here, ".."))
    report = []
    for name, entries in GROUPS.items():
        d = os.path.join(outdir, name)
        os.makedirs(d, exist_ok=True)
        pngs = []
        for i, (concept, fn) in enumerate(entries, 1):
            path = os.path.join(d, f"{name}_{i:02d}.png")
            img = render(fn)
            img.save(path)
            pngs.append(path)
            report.append((path, alpha_coverage(img)))
        with open(os.path.join(d, "CONCEPTS.md"), "w") as fh:
            fh.write(f"# {name} - candidate concepts\n\n")
            fh.write("White-on-transparent, 1024x1024 RGBA, spaCR house style\n")
            fh.write("(flat, thin outlines + solid fills, no colour).\n\n")
            for i, (concept, _fn) in enumerate(entries, 1):
                fh.write(f"{i}. **{name}_{i:02d}** - {concept}\n")
            fh.write("\nSee `_sheet_dark.png` / `_sheet_light.png` for a numbered "
                     "contact sheet;\neach cell also shows the icon at 48 px.\n\n"
                     "`_sheet_light.png` recolours the artwork through its alpha "
                     "channel to dark ink.\nThe PNGs themselves are pure white, "
                     "so on a light background they are invisible\n(the known "
                     "light-theme bug) - the tinted sheet lets the *shape* be "
                     "judged there.\n\n"
                     "Regenerate with:\n"
                     "`QT_QPA_PLATFORM=offscreen python3 "
                     "_generators/masks_measure_group.py`\n")
        contact_sheet(pngs, os.path.join(d, "_sheet_dark.png"),
                      (0x14, 0x16, 0x1a), (0xe8, 0xea, 0xee))
        contact_sheet(pngs, os.path.join(d, "_sheet_light.png"),
                      (0xf5, 0xf6, 0xf8), (0x30, 0x34, 0x3a),
                      ink=(0x1c, 0x1f, 0x24))
    for path, cov in report:
        print(f"{cov * 100:6.2f}%  {path}")
    bad = [(p_, c) for p_, c in report if not (0.05 <= c <= 0.70)]
    print(f"\n{len(report)} icons, {len(bad)} outside the 5-70% alpha band")
    for p_, c in bad:
        print(f"  OUT OF BAND {c * 100:.2f}%  {p_}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
