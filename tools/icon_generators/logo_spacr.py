#!/usr/bin/env python3
"""Candidate marks for the spaCR application logo (``logo_spacr.png``).

Twenty conceptually distinct directions, drawn with QPainter in a normalised
0..1 coordinate space so every path renders cleanly at any raster size.

House style (taken from ``plaque.png`` / ``measure.png``, the two the user
called good):

    * pure white artwork on a transparent background -- alpha carries the shape
    * flat, no gradients, no colour
    * a mix of thin outlined strokes and solid white fills
    * square canvas, subject fills most of the frame
    * literal but stylised biology, not abstract glyphs

Secondary structure (grids, guides) may use white at reduced alpha; the current
``logo_spacr.png`` already does exactly this for its spatial grid.

Run standalone::

    QT_QPA_PLATFORM=offscreen python3 logo_spacr.py [outdir]

Fully deterministic: every "random" placement is drawn from a seeded
``random.Random`` or a hard-coded table.
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
    QImage,
    QPainter,
    QPainterPath,
    QPen,
)

# --------------------------------------------------------------------------
# palette / weights
# --------------------------------------------------------------------------

W = QColor(255, 255, 255, 255)      # primary artwork
W2 = QColor(255, 255, 255, 150)     # secondary structure (grids, guides)
W3 = QColor(255, 255, 255, 90)      # faint structure

HAIR = 0.014
THIN = 0.022
MED = 0.030
BOLD = 0.040
HEAVY = 0.055


# --------------------------------------------------------------------------
# drawing helpers (all coordinates normalised 0..1, y down)
# --------------------------------------------------------------------------

def mkpen(w, col=W, cap=Qt.RoundCap, join=Qt.RoundJoin):
    p = QPen(col)
    p.setWidthF(w)
    p.setCapStyle(cap)
    p.setJoinStyle(join)
    return p


def stroke(p, path, w=MED, col=W, cap=Qt.RoundCap):
    p.setBrush(Qt.NoBrush)
    p.setPen(mkpen(w, col, cap))
    p.drawPath(path)


def fill(p, path, col=W):
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(col))
    p.drawPath(path)


def line(p, x0, y0, x1, y1, w=MED, col=W, cap=Qt.RoundCap):
    p.setBrush(Qt.NoBrush)
    p.setPen(mkpen(w, col, cap))
    p.drawLine(QPointF(x0, y0), QPointF(x1, y1))


def poly_path(pts, close=True):
    path = QPainterPath()
    path.moveTo(pts[0][0], pts[0][1])
    for x, y in pts[1:]:
        path.lineTo(x, y)
    if close:
        path.closeSubpath()
    return path


def circle_path(cx, cy, r, ry=None):
    ry = r if ry is None else ry
    path = QPainterPath()
    path.addEllipse(QRectF(cx - r, cy - ry, 2 * r, 2 * ry))
    return path


def ring(p, cx, cy, r, w=MED, col=W, ry=None):
    stroke(p, circle_path(cx, cy, r, ry), w, col)


def dot(p, cx, cy, r, col=W, ry=None):
    fill(p, circle_path(cx, cy, r, ry), col)


def arc_path(cx, cy, r, start_deg, sweep_deg, ry=None):
    ry = r if ry is None else ry
    path = QPainterPath()
    rect = QRectF(cx - r, cy - ry, 2 * r, 2 * ry)
    path.arcMoveTo(rect, start_deg)
    path.arcTo(rect, start_deg, sweep_deg)
    return path


def arc(p, cx, cy, r, start_deg, sweep_deg, w=MED, col=W, ry=None):
    stroke(p, arc_path(cx, cy, r, start_deg, sweep_deg, ry), w, col)


def rrect_path(x0, y0, x1, y1, rad):
    path = QPainterPath()
    path.addRoundedRect(QRectF(x0, y0, x1 - x0, y1 - y0), rad, rad)
    return path


def catmull_closed(pts):
    """Smooth closed curve through ``pts`` (Catmull-Rom -> cubic beziers)."""
    n = len(pts)
    path = QPainterPath()
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


def catmull_open(pts):
    """Smooth open curve through ``pts``."""
    n = len(pts)
    path = QPainterPath()
    path.moveTo(pts[0][0], pts[0][1])
    for i in range(n - 1):
        p0 = pts[max(i - 1, 0)]
        p1 = pts[i]
        p2 = pts[i + 1]
        p3 = pts[min(i + 2, n - 1)]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6.0, p1[1] + (p2[1] - p0[1]) / 6.0)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6.0, p2[1] - (p3[1] - p1[1]) / 6.0)
        path.cubicTo(c1[0], c1[1], c2[0], c2[1], p2[0], p2[1])
    return path


def blob_path(cx, cy, r, n=9, jitter=0.16, seed=0, sx=1.0, sy=1.0, rot=0.0):
    """Irregular but smooth cell-like outline. Deterministic for a given seed."""
    rnd = random.Random(seed)
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n + rot
        rr = r * (1.0 + rnd.uniform(-jitter, jitter))
        pts.append((cx + sx * rr * math.cos(a), cy + sy * rr * math.sin(a)))
    return catmull_closed(pts)


def helix_paths(x0, x1, yc, amp, cycles, rungs=7, phase=0.0, samples=96):
    """Horizontal double helix: (strand_a, strand_b, [rung endpoints])."""
    a_pts, b_pts = [], []
    for i in range(samples + 1):
        t = i / samples
        x = x0 + (x1 - x0) * t
        s = math.sin(2 * math.pi * cycles * t + phase)
        a_pts.append((x, yc + amp * s))
        b_pts.append((x, yc - amp * s))
    rung_list = []
    for j in range(rungs):
        t = (j + 0.5) / rungs
        x = x0 + (x1 - x0) * t
        s = math.sin(2 * math.pi * cycles * t + phase)
        rung_list.append((x, yc + amp * s, x, yc - amp * s))
    return catmull_open(a_pts), catmull_open(b_pts), rung_list


def arrow_head(p, x, y, ang_deg, size, col=W):
    """Solid triangular arrow head pointing along ``ang_deg`` (screen degrees,
    0 = +x, positive = counter-clockwise on screen)."""
    a = math.radians(-ang_deg)
    tip = (x, y)
    back = (x - size * math.cos(a), y - size * math.sin(a))
    nx, ny = -math.sin(a), math.cos(a)
    left = (back[0] + nx * size * 0.52, back[1] + ny * size * 0.52)
    right = (back[0] - nx * size * 0.52, back[1] - ny * size * 0.52)
    fill(p, poly_path([tip, left, right]), col)


def organelle(p, cx, cy, r, ang_deg=0.0, w=THIN, col=W):
    """The little outlined capsule-with-dots used all over the existing set."""
    p.save()
    p.translate(cx, cy)
    p.rotate(ang_deg)
    path = rrect_path(-r, -r * 0.52, r, r * 0.52, r * 0.5)
    stroke(p, path, w, col)
    dot(p, -r * 0.38, 0.0, r * 0.22, col)
    dot(p, r * 0.30, 0.0, r * 0.20, col)
    p.restore()


# --------------------------------------------------------------------------
# monoline geometric letterforms (for the wordmark / monogram directions)
# --------------------------------------------------------------------------

def glyph_s(x, base, h, w):
    """Lowercase monoline 's'. ``h`` = x-height, ``w`` = advance width."""
    r = min(h / 4.0, w / 2.0)
    cx = x + w / 2.0
    ytc = base - h + r
    ybc = base - r
    path = QPainterPath()
    path.arcMoveTo(QRectF(cx - r, ytc - r, 2 * r, 2 * r), 20)
    path.arcTo(QRectF(cx - r, ytc - r, 2 * r, 2 * r), 20, 250)
    path.arcTo(QRectF(cx - r, ybc - r, 2 * r, 2 * r), 90, -250)
    return path


def glyph_p(x, base, h, w, desc):
    r = min(h / 2.0, w / 2.0)
    cx = x + r
    cy = base - r
    path = QPainterPath()
    path.moveTo(x, base - h)
    path.lineTo(x, base + desc)
    path.addEllipse(QRectF(cx - r, cy - r, 2 * r, 2 * r))
    return path


def glyph_a_cell(p, x, base, h, w, stem_w, nucleus=True):
    """Single-storey 'a' whose bowl is a cell: ring + solid nucleus."""
    r = min(h / 2.0, w / 2.0)
    cx = x + r
    cy = base - r
    ring(p, cx, cy, r, stem_w)
    line(p, cx + r, cy - r, cx + r, base, stem_w)
    if nucleus:
        dot(p, cx, cy, r * 0.40)


def glyph_C(x, base, caph, w):
    r = min(caph / 2.0, w / 2.0)
    cx = x + r
    cy = base - r
    return arc_path(cx, cy, r, 48, 264)


def glyph_R(x, base, caph, w):
    path = QPainterPath()
    path.moveTo(x, base)
    path.lineTo(x, base - caph)
    r = caph * 0.27
    bowl_top = base - caph
    path.moveTo(x, bowl_top)
    path.lineTo(x + w * 0.42, bowl_top)
    path.arcTo(QRectF(x + w * 0.42 - r, bowl_top, 2 * r, 2 * r), 90, -180)
    path.lineTo(x, bowl_top + 2 * r)
    path.moveTo(x + w * 0.30, bowl_top + 2 * r)
    path.lineTo(x + w, base)
    return path


# ==========================================================================
# the twenty candidates
# ==========================================================================

def v01_wordmark_lockup(p):
    """spaCR wordmark, monoline geometric, the 'a' drawn as a cell."""
    # mark above the word: a cell with a spatial grid
    cx, cy, r = 0.5, 0.300, 0.245
    clip = blob_path(cx, cy, r, n=10, jitter=0.075, seed=13)
    p.save()
    p.setClipPath(clip)
    for k in range(-2, 3):
        line(p, cx + k * r * 0.55, cy - r * 1.3, cx + k * r * 0.55, cy + r * 1.3, 0.016, W2)
        line(p, cx - r * 1.3, cy + k * r * 0.55, cx + r * 1.3, cy + k * r * 0.55, 0.016, W2)
    p.restore()
    stroke(p, clip, 0.034)
    dot(p, cx + 0.02, cy + 0.01, r * 0.34)
    organelle(p, cx - r * 0.52, cy - r * 0.48, r * 0.30, -22, 0.016)

    # wordmark
    sw = 0.028
    base = 0.870
    xh = 0.150
    caph = 0.212
    desc = 0.078
    adv = [0.124, 0.140, 0.146, 0.176, 0.166]
    gap = 0.014
    total = sum(adv) + gap * 4
    x = (1.0 - total) / 2.0
    stroke(p, glyph_s(x, base, xh, adv[0]), sw)
    x += adv[0] + gap
    stroke(p, glyph_p(x + sw / 2, base, xh, adv[1] - sw, desc), sw)
    x += adv[1] + gap
    glyph_a_cell(p, x, base, xh, adv[2], sw)
    x += adv[2] + gap
    stroke(p, glyph_C(x, base, caph, adv[3]), sw + 0.006)
    x += adv[3] + gap
    stroke(p, glyph_R(x + sw / 2, base, caph, adv[4] - sw), sw + 0.006)


def v02_monogram_cr(p):
    """'CR' monogram -- the two capitals of spaCR; the C's counter is a cell."""
    sw = 0.088
    cy, r = 0.5, 0.255
    cx = 0.285
    stroke(p, arc_path(cx, cy, r, 46, 268), sw)
    dot(p, cx, cy, 0.098)
    # R, built heavy to match
    x = 0.605
    base, caph, wid = cy + r + sw / 2, 2 * r + sw, 0.305
    top = base - caph
    br = caph * 0.265
    stem = QPainterPath()
    stem.moveTo(x, base)
    stem.lineTo(x, top)
    bowl = QPainterPath()
    bowl.moveTo(x, top)
    bowl.lineTo(x + wid * 0.40, top)
    bowl.arcTo(QRectF(x + wid * 0.40 - br, top, 2 * br, 2 * br), 90, -180)
    bowl.lineTo(x, top + 2 * br)
    leg = QPainterPath()
    leg.moveTo(x + wid * 0.30, top + 2 * br)
    leg.lineTo(x + wid, base)
    for pth in (stem, bowl, leg):
        stroke(p, pth, sw, W, Qt.FlatCap)


def v03_well_plate(p):
    """A microtitre plate with a notched A1 corner; hit wells filled."""
    x0, y0, x1, y1 = 0.075, 0.185, 0.925, 0.815
    ch = 0.085
    outer = poly_path([
        (x0 + ch, y0), (x1 - 0.035, y0), (x1, y0 + 0.035),
        (x1, y1 - 0.035), (x1 - 0.035, y1), (x0 + 0.035, y1),
        (x0, y1 - 0.035), (x0, y0 + ch),
    ])
    stroke(p, outer, 0.034)
    inner = poly_path([
        (x0 + ch + 0.028, y0 + 0.052), (x1 - 0.052, y0 + 0.052),
        (x1 - 0.052, y1 - 0.052), (x0 + 0.052, y1 - 0.052),
        (x0 + 0.052, y0 + ch + 0.010),
    ])
    stroke(p, inner, 0.016, W2)
    cols, rows = 4, 3
    r = 0.072
    gx = (x1 - x0 - 2 * 0.115) / (cols - 1)
    gy = (y1 - y0 - 2 * 0.135) / (rows - 1)
    hits = {(0, 1), (1, 2), (2, 1), (2, 2)}
    for j in range(rows):
        for i in range(cols):
            wx = x0 + 0.115 + i * gx
            wy = y0 + 0.135 + j * gy
            if (j, i) in hits:
                dot(p, wx, wy, r)
            else:
                ring(p, wx, wy, r, 0.026)


def v04_dish_coords(p):
    """Petri dish as a coordinate frame: crosshair axes locate one colony."""
    cx, cy = 0.5, 0.5
    ring(p, cx, cy, 0.415, 0.034)
    ring(p, cx, cy, 0.352, 0.014, W2)
    p.save()
    p.setClipPath(circle_path(cx, cy, 0.352))
    line(p, 0.02, cy, 0.98, cy, 0.016, W2)
    line(p, cx, 0.02, cx, 0.98, 0.016, W2)
    for k in range(1, 4):
        d = k * 0.092
        for sgn in (-1, 1):
            line(p, cx + sgn * d, cy - 0.030, cx + sgn * d, cy + 0.030, 0.016, W2)
            line(p, cx - 0.030, cy + sgn * d, cx + 0.030, cy + sgn * d, 0.016, W2)
    p.restore()
    tx, ty = 0.645, 0.352
    line(p, cx, ty, tx, ty, 0.016, W2)
    line(p, tx, cy, tx, ty, 0.016, W2)
    ring(p, tx, ty, 0.098, 0.028)
    dot(p, tx, ty, 0.048)
    dot(p, 0.335, 0.652, 0.036)
    dot(p, 0.665, 0.672, 0.028)
    dot(p, 0.335, 0.375, 0.026)


def v05_cas9_guide(p):
    """Cas9 as a notched clamp biting the duplex; guide RNA looped inside."""
    # duplex, entering the jaw from the lower right
    a, b, rungs = helix_paths(0.588, 0.998, 0.602, 0.058, 1.1, rungs=5)
    stroke(p, a, 0.030)
    stroke(p, b, 0.030)
    for x0, y0, x1, y1 in rungs:
        line(p, x0, y0, x1, y1, 0.020, W2)
    # body: a rounded protein with a wide V bitten out of the right
    body = catmull_closed([
        (0.455, 0.048), (0.700, 0.115), (0.845, 0.320),
        (0.848, 0.495),                      # upper jaw, going in
        (0.628, 0.602),                      # jaw apex
        (0.858, 0.706),                      # lower jaw, coming out
        (0.660, 0.905), (0.375, 0.930),
        (0.120, 0.780), (0.055, 0.480), (0.180, 0.170),
    ])
    p.setCompositionMode(QPainter.CompositionMode_Clear)
    p.setPen(mkpen(0.072, QColor(0, 0, 0, 255)))
    p.setBrush(QBrush(QColor(0, 0, 0, 255)))
    p.drawPath(body)
    p.setCompositionMode(QPainter.CompositionMode_SourceOver)
    stroke(p, body, 0.040)
    # sgRNA threaded through the protein towards the jaw
    p.save()
    p.setClipPath(body)
    guide = catmull_open([
        (0.408, 0.398), (0.312, 0.368), (0.292, 0.276),
        (0.380, 0.228), (0.456, 0.302), (0.446, 0.428),
        (0.496, 0.528), (0.572, 0.592),
    ])
    stroke(p, guide, 0.030)
    p.restore()


def v06_helix_roundel(p):
    """The double helix, contained in a disc. The CRISPR/sequence half."""
    ring(p, 0.5, 0.5, 0.415, 0.036)
    p.save()
    p.setClipPath(circle_path(0.5, 0.5, 0.372))
    p.translate(0.5, 0.5)
    p.rotate(-90)
    p.translate(-0.5, -0.5)
    a, b, rungs = helix_paths(0.09, 0.91, 0.5, 0.20, 1.5, rungs=9)
    stroke(p, a, 0.040)
    stroke(p, b, 0.040)
    for x0, y0, x1, y1 in rungs:
        line(p, x0, y0, x1, y1, 0.026, W)
    p.restore()


def v07_scissors_dna(p):
    """The edit: scissors closing on a duplex, with the break already open."""
    a, b, rungs = helix_paths(0.025, 0.975, 0.128, 0.048, 1.9, rungs=10)
    stroke(p, a, 0.030)
    stroke(p, b, 0.030)
    for x0, y0, x1, y1 in rungs:
        line(p, x0, y0, x1, y1, 0.020, W2)
    # open the break
    p.setCompositionMode(QPainter.CompositionMode_Clear)
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(QColor(0, 0, 0, 255)))
    p.drawRect(QRectF(0.452, 0.010, 0.096, 0.230))
    p.setCompositionMode(QPainter.CompositionMode_SourceOver)

    px, py = 0.500, 0.588
    for sgn in (-1, 1):
        tip = (px + sgn * 0.046, 0.172)
        basep = (px + sgn * 0.016, py)
        dx, dy = basep[0] - tip[0], basep[1] - tip[1]
        L = math.hypot(dx, dy)
        nx, ny = -dy / L, dx / L
        wt, wb = 0.009, 0.036
        fill(p, poly_path([
            (tip[0] + nx * wt, tip[1] + ny * wt),
            (basep[0] + nx * wb, basep[1] + ny * wb),
            (basep[0] - nx * wb, basep[1] - ny * wb),
            (tip[0] - nx * wt, tip[1] - ny * wt),
        ]))
        hx, hy = px - sgn * 0.198, 0.792
        line(p, basep[0], basep[1] - 0.010, hx + sgn * 0.014, hy - 0.100, 0.044)
        ring(p, hx, hy, 0.110, 0.044)
    dot(p, px, 0.602, 0.042)


def v08_screen_array(p):
    """The screen: an array of cells, one of them a hit."""
    r = 0.108
    step = 0.278
    ox, oy = 0.5 - step, 0.5 - step
    hit = (0, 1)
    for j in range(3):
        for i in range(3):
            cx, cy = ox + i * step, oy + j * step
            if (j, i) == hit:
                continue
            seed = 100 + j * 3 + i
            stroke(p, blob_path(cx, cy, r, n=8, jitter=0.10, seed=seed), 0.028)
            dot(p, cx + r * 0.12, cy - r * 0.06, r * 0.36)
    hx, hy = ox + hit[1] * step, oy + hit[0] * step
    fill(p, blob_path(hx, hy, r, n=8, jitter=0.10, seed=101))
    # selection bracket around the hit
    s = r + 0.072
    t = 0.062
    for sx in (-1, 1):
        for sy in (-1, 1):
            line(p, hx + sx * s, hy + sy * s, hx + sx * (s - t), hy + sy * s, 0.032)
            line(p, hx + sx * s, hy + sy * s, hx + sx * s, hy + sy * (s - t), 0.032)


def v09_pin_on_cell(p):
    """A map pin dropped on a cell: the phenotype, located."""
    cell = blob_path(0.455, 0.575, 0.355, n=9, jitter=0.14, seed=11)
    stroke(p, cell, 0.034)
    dot(p, 0.375, 0.635, 0.088)
    organelle(p, 0.30, 0.435, 0.075, -18, 0.020)
    organelle(p, 0.50, 0.795, 0.070, 12, 0.020)
    dot(p, 0.245, 0.735, 0.032)
    # pin: head circle joined to the tip by its two tangents (stays smooth)
    px, hy, hr = 0.700, 0.250, 0.190
    tipy = 0.680
    d = tipy - hy
    th = math.degrees(math.acos(hr / d))
    phi = -90.0                       # Qt angle of head-centre -> tip
    rect = QRectF(px - hr, hy - hr, 2 * hr, 2 * hr)
    pin = QPainterPath()
    pin.arcMoveTo(rect, phi + th)
    pin.arcTo(rect, phi + th, 360.0 - 2 * th)
    pin.lineTo(px, tipy)
    pin.closeSubpath()
    p.setCompositionMode(QPainter.CompositionMode_Clear)
    p.setPen(mkpen(0.078, QColor(0, 0, 0, 255)))
    p.setBrush(QBrush(QColor(0, 0, 0, 255)))
    p.drawPath(pin)
    p.setCompositionMode(QPainter.CompositionMode_SourceOver)
    fill(p, pin)
    p.setCompositionMode(QPainter.CompositionMode_Clear)
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(QColor(0, 0, 0, 255)))
    p.drawEllipse(QPointF(px, hy), hr * 0.40, hr * 0.40)
    p.setCompositionMode(QPainter.CompositionMode_SourceOver)


def v10_radar_scan(p):
    """A spatial scan: concentric range rings, a sweep, objects detected."""
    cx, cy = 0.5, 0.5
    for r, w in ((0.415, 0.034), (0.290, 0.020), (0.165, 0.020)):
        ring(p, cx, cy, r, w, W if r > 0.4 else W2)
    line(p, 0.085, cy, 0.915, cy, 0.014, W3)
    line(p, cx, 0.085, cx, 0.915, 0.014, W3)
    wedge = QPainterPath()
    wedge.moveTo(cx, cy)
    wedge.arcTo(QRectF(cx - 0.415, cy - 0.415, 0.83, 0.83), 30, 55)
    wedge.closeSubpath()
    fill(p, wedge, QColor(255, 255, 255, 70))
    line(p, cx, cy, cx + 0.415 * math.cos(math.radians(30)),
         cy - 0.415 * math.sin(math.radians(30)), 0.028)
    dot(p, cx, cy, 0.062)
    for tx, ty, rr in ((0.700, 0.365, 0.046), (0.330, 0.360, 0.036),
                       (0.375, 0.685, 0.052), (0.660, 0.675, 0.030)):
        dot(p, tx, ty, rr)


def v11_hex_tissue(p):
    """A segmented monolayer: packed cells, the centre one called."""
    R = 0.132
    dx = R * math.sqrt(3)
    dy = R * 1.5

    def hexpath(cx, cy, r):
        pts = [(cx + r * math.cos(math.radians(90 + 60 * k)),
                cy - r * math.sin(math.radians(90 + 60 * k))) for k in range(6)]
        return poly_path(pts)

    centres = []
    for row in range(-4, 5):
        for col in range(-4, 5):
            cx = 0.5 + col * dx + (dx / 2 if row % 2 else 0.0)
            cy = 0.5 + row * dy
            if abs(cx - 0.5) < 0.60 and abs(cy - 0.5) < 0.60:
                centres.append((cx, cy))

    frame = rrect_path(0.070, 0.070, 0.930, 0.930, 0.075)
    p.save()
    p.setClipPath(frame)
    for cx, cy in centres:
        stroke(p, hexpath(cx, cy, R), 0.022)
        if abs(cx - 0.5) < 1e-6 and abs(cy - 0.5) < 1e-6:
            continue
        dot(p, cx, cy, R * 0.235)
    solid = hexpath(0.5, 0.5, R).subtracted(circle_path(0.5, 0.5, R * 0.36))
    fill(p, solid)
    p.restore()
    stroke(p, frame, 0.036)


def v12_nucleus_orbit(p):
    """Spatial relationships: a nucleus with satellites on their orbit."""
    p.save()
    p.translate(0.5, 0.5)
    p.rotate(-24)
    p.translate(-0.5, -0.5)
    ring(p, 0.5, 0.5, 0.415, 0.030, W, ry=0.185)
    p.restore()
    p.save()
    p.translate(0.5, 0.5)
    p.rotate(56)
    p.translate(-0.5, -0.5)
    ring(p, 0.5, 0.5, 0.415, 0.020, W2, ry=0.185)
    p.restore()
    dot(p, 0.5, 0.5, 0.175)
    for ang, rr in ((-24 + 0, 0.062), (-24 + 128, 0.050), (-24 + 232, 0.044)):
        # point on the rotated ellipse
        ex, ey = 0.415 * math.cos(math.radians(0)), 0.185 * math.sin(math.radians(0))
        t = math.radians(ang + 24)
        ex, ey = 0.415 * math.cos(t), 0.185 * math.sin(t)
        ca, sa = math.cos(math.radians(-24)), math.sin(math.radians(-24))
        x = 0.5 + ex * ca + ey * sa
        y = 0.5 - ex * sa + ey * ca
        dot(p, x, y, rr)
    t = math.radians(72)
    ex, ey = 0.415 * math.cos(t), 0.185 * math.sin(t)
    ca, sa = math.cos(math.radians(56)), math.sin(math.radians(56))
    dot(p, 0.5 + ex * ca + ey * sa, 0.5 - ex * sa + ey * ca, 0.044, W2)


def v13_objective_slide(p):
    """The instrument: an objective over a gridded slide with one cell."""
    # slide
    x0, y0, x1, y1 = 0.055, 0.645, 0.945, 0.885
    stroke(p, rrect_path(x0, y0, x1, y1, 0.028), 0.032)
    p.save()
    p.setClipPath(rrect_path(x0, y0, x1, y1, 0.028))
    for k in range(1, 6):
        x = x0 + (x1 - x0) * k / 6.0
        line(p, x, y0, x, y1, 0.014, W3)
    line(p, x0, (y0 + y1) / 2, x1, (y0 + y1) / 2, 0.014, W3)
    p.restore()
    ring(p, 0.50, 0.765, 0.070, 0.024)
    dot(p, 0.50, 0.765, 0.028)
    dot(p, 0.235, 0.762, 0.030)
    dot(p, 0.765, 0.762, 0.030)
    # objective: wide barrel, shoulder, short cone
    bx, sx2 = 0.215, 0.145
    body = poly_path([
        (0.5 - bx, 0.065), (0.5 + bx, 0.065), (0.5 + bx, 0.255),
        (0.5 + sx2, 0.300), (0.5 + sx2, 0.395), (0.5 + 0.052, 0.480),
        (0.5 - 0.052, 0.480), (0.5 - sx2, 0.395), (0.5 - sx2, 0.300),
        (0.5 - bx, 0.255),
    ])
    stroke(p, body, 0.036)
    for y in (0.130, 0.192):
        line(p, 0.5 - bx + 0.018, y, 0.5 + bx - 0.018, y, 0.024, W2)
    # light cone down to the slide
    line(p, 0.5 - 0.040, 0.500, 0.5 - 0.088, 0.632, 0.018, W2)
    line(p, 0.5 + 0.040, 0.500, 0.5 + 0.088, 0.632, 0.018, W2)


def v14_barcode_cell(p):
    """Barcode to phenotype: the readout on the left, the cell on the right."""
    bars = [(0.070, 0.030), (0.118, 0.016), (0.152, 0.038), (0.208, 0.014),
            (0.240, 0.026), (0.288, 0.044), (0.352, 0.018)]
    for x, w in bars:
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(W))
        p.drawRect(QRectF(x, 0.215, w, 0.570))
    for x, w in ((0.070, 0.030), (0.208, 0.014)):
        pass
    # linking strokes into the cell
    line(p, 0.395, 0.325, 0.470, 0.325, 0.020, W2)
    line(p, 0.395, 0.500, 0.470, 0.500, 0.020, W2)
    line(p, 0.395, 0.675, 0.470, 0.675, 0.020, W2)
    cell = blob_path(0.705, 0.5, 0.255, n=9, jitter=0.11, seed=5)
    stroke(p, cell, 0.034)
    dot(p, 0.725, 0.487, 0.090)
    organelle(p, 0.640, 0.665, 0.062, -20, 0.018)
    dot(p, 0.640, 0.360, 0.030)


def v15_aperture_c(p):
    """The minimal mark: a heavy C-membrane around a nucleus. 16px-proof."""
    stroke(p, arc_path(0.5, 0.5, 0.335, 40, 280), 0.135)
    dot(p, 0.5, 0.5, 0.130)


def v16_plasmid_guide(p):
    """The vector: a plasmid ring carrying one heavy guide cassette."""
    R = 0.360
    ring(p, 0.5, 0.5, R, 0.036)
    # guide cassette: a heavy arc finished by a tangential arrow head
    a0, sweep = 82, 74
    arc(p, 0.5, 0.5, R, a0, sweep, 0.104)
    th = math.radians(a0 + sweep)
    hx = 0.5 + R * math.cos(th)
    hy = 0.5 - R * math.sin(th)
    tang = math.degrees(math.atan2(math.cos(th), -math.sin(th)))
    arrow_head(p, hx, hy, tang, 0.106)
    # other annotated features on the backbone
    for start, sweep in ((208, 40), (272, 30)):
        arc(p, 0.5, 0.5, R, start, sweep, 0.060)
    # restriction sites
    for ang in (0, 185)  :
        t = math.radians(ang)
        line(p, 0.5 + (R - 0.072) * math.cos(t), 0.5 - (R - 0.072) * math.sin(t),
             0.5 + (R + 0.072) * math.cos(t), 0.5 - (R + 0.072) * math.sin(t),
             0.026, W)
    # the guide insert itself, a short duplex across the lumen
    aa, bb, rr = helix_paths(0.335, 0.665, 0.5, 0.046, 1.2, rungs=5)
    stroke(p, aa, 0.028, W2)
    stroke(p, bb, 0.028, W2)
    for x0, y0, x1, y1 in rr:
        line(p, x0, y0, x1, y1, 0.018, W3)


def v17_phenotype_space(p):
    """Phenotype space: the measured cloud, with one population gated."""
    line(p, 0.115, 0.885, 0.115, 0.095, 0.034, W, Qt.FlatCap)
    line(p, 0.115, 0.885, 0.905, 0.885, 0.034, W, Qt.FlatCap)
    arrow_head(p, 0.115, 0.075, 90, 0.075)
    arrow_head(p, 0.925, 0.885, 0, 0.075)
    def scatter(cx, cy, spread, n, minsep, seed):
        rnd = random.Random(seed)
        pts = []
        tries = 0
        while len(pts) < n and tries < 4000:
            tries += 1
            x = rnd.gauss(cx, spread)
            y = rnd.gauss(cy, spread)
            if not (0.20 < x < 0.86 and 0.16 < y < 0.80):
                continue
            if all((x - a) ** 2 + (y - b) ** 2 > minsep ** 2 for a, b in pts):
                pts.append((x, y))
        return pts

    for x, y in scatter(0.360, 0.660, 0.090, 9, 0.115, 20240607):
        ring(p, x, y, 0.043, 0.024)
    for x, y in scatter(0.690, 0.330, 0.072, 7, 0.115, 991):
        dot(p, x, y, 0.045)
    stroke(p, circle_path(0.690, 0.330, 0.180), 0.022, W2)


def v18_roi_box(p):
    """The spatial primitive: a cell inside its measured bounding box."""
    cell = blob_path(0.5, 0.515, 0.290, n=9, jitter=0.15, seed=23)
    stroke(p, cell, 0.032)
    dot(p, 0.520, 0.505, 0.092)
    organelle(p, 0.395, 0.665, 0.062, -18, 0.018)
    dot(p, 0.400, 0.370, 0.030)
    x0, y0, x1, y1 = 0.150, 0.185, 0.850, 0.845
    t = 0.115
    for sx, x in ((1, x0), (-1, x1)):
        for sy, y in ((1, y0), (-1, y1)):
            line(p, x, y, x + sx * t, y, 0.040)
            line(p, x, y, x, y + sy * t, 0.040)
    for k in (0.25, 0.5, 0.75):
        line(p, x0 + (x1 - x0) * k, y1, x0 + (x1 - x0) * k, y1 + 0.060, 0.020, W2)
        line(p, x0 - 0.060, y0 + (y1 - y0) * k, x0, y0 + (y1 - y0) * k, 0.020, W2)
    # origin marker at the box's top-left
    line(p, x0, y0 - 0.098, x0, y0 - 0.020, 0.020, W2)
    line(p, x0 - 0.098, y0, x0 - 0.020, y0, 0.020, W2)


def v19_z_stack(p):
    """The data: a stack of imaged planes, the front one resolved."""
    side = 0.560
    off = 0.092
    for dx, dy in ((-off, -off), (0.0, 0.0)):
        cx = 0.5 + dx + off / 2
        cy = 0.5 + dy + off / 2
        stroke(p, rrect_path(cx - side / 2, cy - side / 2, cx + side / 2,
                             cy + side / 2, 0.060), 0.030, W2)
    cx, cy = 0.5 + off + off / 2, 0.5 + off + off / 2
    front = rrect_path(cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2, 0.060)
    p.setCompositionMode(QPainter.CompositionMode_Clear)
    p.setPen(mkpen(0.075, QColor(0, 0, 0, 255)))
    p.setBrush(QBrush(QColor(0, 0, 0, 255)))
    p.drawPath(front)
    p.setCompositionMode(QPainter.CompositionMode_SourceOver)
    stroke(p, front, 0.036)
    cell = blob_path(cx - 0.012, cy - 0.008, 0.170, n=9, jitter=0.13, seed=31)
    stroke(p, cell, 0.030)
    dot(p, cx + 0.006, cy - 0.004, 0.058)
    organelle(p, cx - 0.092, cy + 0.086, 0.044, -22, 0.017)
    dot(p, cx + 0.092, cy + 0.078, 0.024)


def v20_guide_entry(p):
    """The perturbation: a guide threaded through the membrane to the nucleus."""
    ring(p, 0.5, 0.5, 0.400, 0.038)
    # punch a gap in the membrane where the guide enters
    p.setCompositionMode(QPainter.CompositionMode_Clear)
    a = math.radians(140)
    fill(p, circle_path(0.5 + 0.400 * math.cos(a), 0.5 - 0.400 * math.sin(a), 0.085),
         QColor(0, 0, 0, 255))
    p.setCompositionMode(QPainter.CompositionMode_SourceOver)
    ring(p, 0.590, 0.585, 0.170, 0.032)
    guide = catmull_open([
        (0.045, 0.070), (0.170, 0.195), (0.225, 0.305),
        (0.345, 0.395), (0.408, 0.492), (0.400, 0.560),
    ])
    stroke(p, guide, 0.034)
    arrow_head(p, 0.408, 0.600, -78, 0.115)
    dot(p, 0.735, 0.320, 0.044)
    dot(p, 0.770, 0.660, 0.032)
    organelle(p, 0.320, 0.760, 0.070, 16, 0.020)


VARIANTS = [
    ("wordmark_lockup", v01_wordmark_lockup,
     "Monoline geometric 'spaCR' wordmark under a grid-cell mark; the 'a' bowl is a cell."),
    ("monogram_cr", v02_monogram_cr,
     "'CR' monogram -- the two capitals of spaCR, the C read as a membrane around a nucleus."),
    ("well_plate", v03_well_plate,
     "Microtitre plate with the notched A1 corner; four hit wells filled solid."),
    ("dish_coords", v04_dish_coords,
     "Petri dish used as a coordinate frame -- crosshair axes and ticks locate one colony."),
    ("cas9_guide", v05_cas9_guide,
     "Cas9 as a notched clamp biting a DNA duplex, the sgRNA threaded through the body."),
    ("helix_roundel", v06_helix_roundel,
     "A double helix contained in a disc -- the sequencing/CRISPR half of the name, as a roundel."),
    ("scissors_dna", v07_scissors_dna,
     "The edit: scissors closing on a duplex with the double-strand break already open."),
    ("screen_array", v08_screen_array,
     "The screen: a 3x3 array of cells with one filled hit picked out by a selection bracket."),
    ("pin_on_cell", v09_pin_on_cell,
     "A map pin dropped on a cell -- the phenotype, located in space."),
    ("radar_scan", v10_radar_scan,
     "A spatial scan: concentric range rings, a sweep wedge, and detected objects as dots."),
    ("hex_tissue", v11_hex_tissue,
     "A segmented monolayer as packed hexagonal cells, the centre one solid (called)."),
    ("nucleus_orbit", v12_nucleus_orbit,
     "Spatial relationships: a solid nucleus with satellites travelling two orbits."),
    ("objective_slide", v13_objective_slide,
     "The instrument: a microscope objective over a gridded slide carrying one cell."),
    ("barcode_cell", v14_barcode_cell,
     "Barcode to phenotype: sequencing bars on the left wired across into a cell on the right."),
    ("aperture_c", v15_aperture_c,
     "The minimal mark: one heavy C-membrane around a solid nucleus. Built for 16px."),
    ("plasmid_guide", v16_plasmid_guide,
     "The vector: a plasmid ring carrying one heavy highlighted guide cassette with an arrow."),
    ("phenotype_space", v17_phenotype_space,
     "Phenotype space: a measured scatter with one population gated and filled."),
    ("roi_box", v18_roi_box,
     "The spatial primitive: a cell inside its measured bounding box with dimension ticks."),
    ("z_stack", v19_z_stack,
     "The data: a stack of imaged planes, only the front one resolved into a cell."),
    ("guide_entry", v20_guide_entry,
     "The perturbation: a guide strand threaded through a gap in the membrane to the nucleus."),
]


# Variants that still resolve at 16x16 -- verified by eye on _sheet_small.png.
SMALL_SAFE = {2, 6, 9, 12, 15}


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def render(fn, size=1024):
    img = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setRenderHint(QPainter.SmoothPixmapTransform, True)
    p.scale(size, size)
    p.setBrush(Qt.NoBrush)
    fn(p)
    p.end()
    return img.convertToFormat(QImage.Format_ARGB32)


def tinted(src, ink):
    """Re-ink an alpha mask. The artwork is white-on-transparent, so on a light
    background it has to be recoloured before it can be judged at all."""
    out = QImage(src.size(), QImage.Format_ARGB32_Premultiplied)
    out.fill(Qt.transparent)
    q = QPainter(out)
    q.drawImage(0, 0, src)
    q.setCompositionMode(QPainter.CompositionMode_SourceIn)
    q.fillRect(out.rect(), QColor(ink))
    q.end()
    return out


def contact_sheet(paths, bg, out, cols=5, cell=300, pad=14, label_h=34,
                  ink=None, note=None):
    n = len(paths)
    rows = (n + cols - 1) // cols
    head = 40 if note else 0
    wpx = cols * cell
    hpx = head + rows * (cell + label_h)
    img = QImage(wpx, hpx, QImage.Format_ARGB32_Premultiplied)
    img.fill(QColor(bg))
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setRenderHint(QPainter.SmoothPixmapTransform, True)
    dark = QColor(bg).lightnessF() < 0.5
    fg = QColor(235, 237, 240) if dark else QColor(30, 32, 36)
    grid = QColor(255, 255, 255, 28) if dark else QColor(0, 0, 0, 28)
    f = QFont()
    f.setPixelSize(19)
    f.setBold(True)
    p.setFont(f)
    if note:
        p.setPen(QPen(fg))
        p.drawText(QRectF(0, 0, wpx, head), int(Qt.AlignCenter), note)
    for i, path in enumerate(paths):
        r, c = divmod(i, cols)
        x = c * cell
        y = head + r * (cell + label_h)
        p.setPen(QPen(grid, 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(x, y, cell - 1, cell + label_h - 1)
        src = QImage(path)
        if ink is not None:
            src = tinted(src, ink)
        inner = cell - 2 * pad
        p.drawImage(QRectF(x + pad, y + pad, inner, inner), src)
        p.setPen(QPen(fg))
        p.drawText(QRectF(x, y + cell - 4, cell, label_h),
                   int(Qt.AlignHCenter | Qt.AlignVCenter),
                   "%02d  %s" % (i + 1, VARIANTS[i][0]))
    p.end()
    img.convertToFormat(QImage.Format_ARGB32).save(out)


def small_sheet(paths, out, sizes=(16, 32, 48), bg="#14161a"):
    """Every variant rendered at title-bar / favicon sizes, 3x nearest-zoom."""
    zoom = 3
    colw = sum(s * zoom for s in sizes) + 24 * len(sizes) + 150
    rowh = max(sizes) * zoom + 26
    cols = 2
    rows = (len(paths) + cols - 1) // cols
    img = QImage(colw * cols, rowh * rows, QImage.Format_ARGB32_Premultiplied)
    img.fill(QColor(bg))
    p = QPainter(img)
    f = QFont()
    f.setPixelSize(17)
    p.setFont(f)
    for i, path in enumerate(paths):
        r, c = divmod(i, cols)
        x0 = c * colw
        y0 = r * rowh
        p.setPen(QPen(QColor(235, 237, 240)))
        p.drawText(QRectF(x0 + 8, y0, 142, rowh), int(Qt.AlignVCenter | Qt.AlignLeft),
                   "%02d %s" % (i + 1, VARIANTS[i][0][:14]))
        x = x0 + 150
        src = QImage(path)
        for s in sizes:
            small = src.scaled(s, s, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            big = small.scaled(s * zoom, s * zoom, Qt.KeepAspectRatio, Qt.FastTransformation)
            p.drawImage(QPointF(x, y0 + (rowh - 26 - s * zoom) / 2 + 13), big)
            x += s * zoom + 24
    p.end()
    img.convertToFormat(QImage.Format_ARGB32).save(out)


def main(outdir=None):
    from PySide6.QtGui import QGuiApplication
    if QGuiApplication.instance() is None:
        QGuiApplication(sys.argv[:1])
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = outdir or os.path.join(os.path.dirname(here), "logo_spacr")
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for i, (name, fn, _desc) in enumerate(VARIANTS, start=1):
        out = os.path.join(outdir, "logo_spacr_%02d.png" % i)
        render(fn).save(out)
        paths.append(out)
        print("wrote", out)

    contact_sheet(paths, "#14161a", os.path.join(outdir, "_sheet_dark.png"))
    contact_sheet(
        paths, "#f5f6f8", os.path.join(outdir, "_sheet_light.png"),
        ink="#1b1e23",
        note="artwork is white-on-transparent; re-inked dark here so the form "
             "can be judged on a light background")
    small_sheet(paths, os.path.join(outdir, "_sheet_small.png"))

    with open(os.path.join(outdir, "CONCEPTS.md"), "w") as fh:
        fh.write("# logo_spacr -- 20 candidate marks\n\n")
        fh.write("Twenty different metaphors for the same idea, not one drawing "
                 "restyled twenty times.\n"
                 "White on transparent, flat, 1024x1024 RGBA, same treatment as "
                 "`plaque.png` / `measure.png`.\n"
                 "Numbering matches `_sheet_dark.png`, `_sheet_light.png` and "
                 "`_sheet_small.png`.\n\n")
        for i, (name, _fn, desc) in enumerate(VARIANTS, start=1):
            tag = " **[16px-safe]**" if i in SMALL_SAFE else ""
            fh.write("%2d. **%s** -- %s%s\n" % (i, name, desc, tag))
        fh.write("\n## Sheets\n\n"
                 "* `_sheet_dark.png` -- all 20 on #14161a, as shipped.\n"
                 "* `_sheet_light.png` -- all 20 on #f5f6f8, **re-inked dark**. "
                 "The artwork itself is white-on-transparent and is invisible on "
                 "a light background (the known open bug); the sheet re-inks the "
                 "alpha mask so the form can still be judged.\n"
                 "* `_sheet_small.png` -- every variant at 16 / 32 / 48 px, "
                 "nearest-neighbour zoomed 3x. This is the favicon and title-bar "
                 "case and the hardest constraint.\n\n"
                 "Marks tagged **[16px-safe]** still read at 16x16: %s.\n"
                 % ", ".join("%02d %s" % (i, VARIANTS[i - 1][0])
                             for i in sorted(SMALL_SAFE)))
        fh.write("\nRegenerate: `QT_QPA_PLATFORM=offscreen python3 "
                 "_generators/logo_spacr.py`\n")
    print("wrote sheets + CONCEPTS.md in", outdir)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
