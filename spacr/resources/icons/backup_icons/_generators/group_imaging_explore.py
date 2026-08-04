#!/usr/bin/env python3
"""Candidate spaCR icons: the imaging / exploration group.

Ten conceptually different designs for each of eight apps, white-on-transparent
flat vector art in the house style set by ``plaque.png`` / ``measure.png``.
All artwork is drawn with the shared primitives in :mod:`_draw` and packaged by
:func:`_emit.emit_groups`, so these folders cannot drift from the others.

The eight apps here are all "look at the data" tools, which is exactly why they
are easy to draw badly: a track, a plot and a stack of frames will collapse into
one another unless every set holds its own line.  The lines held below:

* **timelapse** -- the SAME objects across FRAMES.  Time is the axis: a strip of
  time points, a track threading them, a division seen *over frames*.  Nothing
  here measures a speed.
* **motility** -- the PATH and its SPEED.  A track is here to be *measured*:
  velocity readouts, step spacing, displacement versus path length, fast against
  slow.  Nothing here is a stack of frames.
* **curate** -- a HUMAN HAND fixing something, ON THE RECORD.  Every design has
  both halves: an edit being made (brush, split, merge, erase, re-link) and the
  book-keeping around it (log, revision history, sign-off, accept/reject).
* **lineage** -- CONTAINMENT, not ancestry.  Cell contains nucleus contains
  vacuole.  Deliberately NO family tree of dividing cells: division over time is
  ``timelapse``'s territory, and a doubling parasite is ``replication``'s.
* **layer_viewer** -- a STACK of overlays that can be toggled.  Sheets seen in
  perspective, eyes and sliders that switch them, one world they all register
  into.
* **image_scatter** -- a SCATTER PLOT WHOSE POINTS ARE CELLS.  Every design ties
  a point on the axes to a picture of the object behind it.
* **graph_builder** -- COLUMNS DRAGGED ONTO SHELVES.  Chips, drop targets and
  the encoding each shelf controls (x, y, colour, size, facet).
* **analyze_plaques** -- a MONOLAYER with CLEARED ZONES, counted and sized.  The
  existing ``plaque`` candidate set already owns the petri dish, the six-well
  plate, the side view, the counting grid, the time course, the flask, the bar
  chart and the callipers; this set deliberately goes elsewhere -- negative
  space, scan profiles, control-versus-treated, coverage fraction.

Deterministic: no random draws at all.  Run standalone:

    QT_QPA_PLATFORM=offscreen python3 group_imaging_explore.py [OUTDIR]

Default OUTDIR is the backup_icons directory one level up.  Writes
``<OUTDIR>/<key>/<key>_NN.png`` plus CONCEPTS.md and the two contact sheets.
It never touches anything in ``spacr/resources/icons/*.png``.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtCore import QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QPainterPath  # noqa: E402

from _draw import W_FINE, W_MAIN, W_SEC  # noqa: E402
from _emit import default_outdir, emit_groups  # noqa: E402

TAU = math.pi * 2.0


# ---------------------------------------------------------------------------
# shared sub-drawings
# ---------------------------------------------------------------------------

#: fixed radial jitter, so "organic" outlines are still byte-deterministic
_WOBBLE = (1.07, 0.93, 1.04, 0.91, 1.06, 0.95, 1.02, 0.97)


def blob(c, cx, cy, r, w=W_SEC, filled=False, dash=None, phase=0, squash=0.92):
    """An organic closed outline -- a cell, a mask, a cleared zone."""
    pts = []
    n_ = len(_WOBBLE)
    for i in range(n_):
        a = TAU * i / n_
        k = _WOBBLE[(i + phase) % n_]
        pts.append((cx + r * k * math.cos(a), cy + r * k * squash * math.sin(a)))
    c.smooth(pts, w=w, closed=True, filled=filled, dash=dash)


def crop(c, x, y, s, w=W_SEC, r=0.02, obj=0.30, nuc=0.42):
    """A square image crop with one cell in it."""
    c.rect(x, y, s, s, w=w, r=r)
    if obj > 0:
        c.cell(x + s / 2.0, y + s / 2.0, s * obj, w=W_FINE, nuc=nuc)


def eye(c, cx, cy, r, w=W_SEC, open_=True):
    """Visibility toggle: an open eye with a solid pupil, or a shut lid."""
    h = r * 0.62
    if open_:
        c.smooth([(cx - r, cy), (cx - r * 0.48, cy - h), (cx + r * 0.48, cy - h),
                  (cx + r, cy), (cx + r * 0.48, cy + h), (cx - r * 0.48, cy + h)],
                 w=w, closed=True)
        c.disc(cx, cy, r * 0.34)
    else:
        c.smooth([(cx - r, cy), (cx - r * 0.48, cy - h), (cx + r * 0.48, cy - h),
                  (cx + r, cy), (cx + r * 0.48, cy + h), (cx - r * 0.48, cy + h)],
                 w=w, closed=True)
        c.line(cx - r * 0.86, cy + h * 1.15, cx + r * 0.86, cy - h * 1.15, w)


def sheet(c, cx, cy, w_, h, w=W_SEC):
    """One layer seen in perspective: a flat diamond plane."""
    c.polyline([(cx, cy - h / 2.0), (cx + w_ / 2.0, cy),
                (cx, cy + h / 2.0), (cx - w_ / 2.0, cy)], w=w, close=True)


def rot_rect(c, cx, cy, w_, h, ang, w=W_SEC, filled=False):
    """A rectangle rotated about its centre."""
    a = math.radians(ang)
    ca, sa = math.cos(a), math.sin(a)
    pts = [(cx + dx * ca - dy * sa, cy + dx * sa + dy * ca)
           for dx, dy in ((-w_ / 2, -h / 2), (w_ / 2, -h / 2),
                          (w_ / 2, h / 2), (-w_ / 2, h / 2))]
    c.polyline(pts, w=w, close=True, filled=filled)
    return pts


def cursor(c, x, y, s=0.20):
    """A mouse pointer with its tip at (x, y)."""
    shape = [(0.00, 0.00), (0.00, 1.00), (0.26, 0.77), (0.43, 1.13),
             (0.60, 1.05), (0.43, 0.70), (0.72, 0.66)]
    c.polyline([(x + px * s, y + py * s) for px, py in shape],
               close=True, filled=True)


def stylus(c, tx, ty, L=0.42, ang=45.0, wide=0.10):
    """A pen / brush whose tip touches (tx, ty), body running along ``ang``."""
    a = math.radians(ang)
    dx, dy = math.cos(a), math.sin(a)
    px, py = -dy, dx
    bx, by = tx + dx * L * 0.24, ty + dy * L * 0.24
    c.polyline([(tx, ty), (bx + px * wide / 2, by + py * wide / 2),
                (bx - px * wide / 2, by - py * wide / 2)], close=True, filled=True)
    ex, ey = tx + dx * L, ty + dy * L
    c.polyline([(bx + px * wide / 2, by + py * wide / 2),
                (ex + px * wide / 2, ey + py * wide / 2),
                (ex - px * wide / 2, ey - py * wide / 2),
                (bx - px * wide / 2, by - py * wide / 2)], w=W_SEC, close=True)


def tick(c, cx, cy, s=0.10, w=W_MAIN):
    """A check mark centred on (cx, cy)."""
    c.polyline([(cx - s, cy), (cx - s * 0.25, cy + s * 0.72),
                (cx + s, cy - s * 0.78)], w=w)


def cross(c, cx, cy, s=0.09, w=W_MAIN):
    """An X centred on (cx, cy)."""
    c.line(cx - s, cy - s, cx + s, cy + s, w)
    c.line(cx + s, cy - s, cx - s, cy + s, w)


def chip(c, x, y, w_, h, filled=False, w=W_SEC):
    """A draggable column chip: a rounded capsule."""
    c.rect(x, y, w_, h, w=w, filled=filled, r=h / 2.0)


def slot(c, x, y, w_, h, w=W_SEC, r=None):
    """An empty drop target: a dashed rounded rectangle."""
    rr = min(w_, h) / 2.0 if r is None else r
    pa = QPainterPath()
    pa.addRoundedRect(QRectF(x * c.n, y * c.n, w_ * c.n, h * c.n),
                      rr * c.n, rr * c.n)
    c.stroke(pa, w, dash=[3, 3])


def punched(c, x, y, w_, h, holes, r=0.03):
    """A solid slab with holes knocked out of it (even-odd fill)."""
    pa = QPainterPath()
    pa.setFillRule(Qt.OddEvenFill)
    pa.addRoundedRect(QRectF(x * c.n, y * c.n, w_ * c.n, h * c.n),
                      r * c.n, r * c.n)
    for hx, hy, hr in holes:
        pa.addEllipse(QPointF(hx * c.n, hy * c.n), hr * c.n, hr * c.n * 0.86)
    c.fill(pa)


def lawn(c, x, y, w_, h, holes, cols=5, rows=4, r=0.072, w=W_FINE, stagger=0.0):
    """A confluent monolayer of cells, cleared where a plaque sits."""
    for j in range(rows):
        for i in range(cols):
            cx = x + w_ * (i + 0.5) / cols + (stagger if j % 2 else 0.0)
            cy = y + h * (j + 0.5) / rows
            if cx > x + w_:
                continue
            if any((cx - hx) ** 2 + (cy - hy) ** 2 < hr * hr
                   for hx, hy, hr in holes):
                continue
            c.circ(cx, cy, r, w)


def track(c, pts, w=W_MAIN, dots=0.0, dash=None):
    """A trajectory through ``pts``, optionally with detections marked on it."""
    c.smooth(pts, w=w, dash=dash)
    if dots > 0:
        for px, py in pts:
            c.disc(px, py, dots)


# =====================================================================
# timelapse -- the SAME objects across FRAMES; time is the axis
# =====================================================================

def timelapse_01(c):
    """Filmstrip of three frames, the same cell further along in each."""
    c.rect(0.03, 0.22, 0.94, 0.56, w=W_MAIN, r=0.025)
    for i in range(4):
        x = 0.10 + i * 0.23
        c.rect(x, 0.245, 0.095, 0.060, w=W_FINE, filled=True, r=0.016)
        c.rect(x, 0.695, 0.095, 0.060, w=W_FINE, filled=True, r=0.016)
    for i, fx in enumerate((0.07, 0.375, 0.68)):
        c.rect(fx, 0.345, 0.275, 0.31, w=W_SEC, r=0.02)
        c.cell(fx + 0.085 + i * 0.055, 0.50, 0.070, w=W_FINE, nuc=0.46)


def timelapse_02(c):
    """One track threading the same object through three frames in a row."""
    pos = [(0.19, 0.62), (0.50, 0.36), (0.81, 0.54)]
    for x in (0.05, 0.36, 0.67):
        c.rect(x, 0.22, 0.28, 0.52, w=W_SEC, r=0.02)
    c.smooth(pos, w=W_MAIN)
    for px, py in pos:
        c.disc(px, py, 0.048)
    c.arrow(0.08, 0.90, 0.92, 0.90, W_SEC, head=0.065)


def timelapse_03(c):
    """Kymograph: the object's position streaked down a vertical time axis."""
    c.rect(0.26, 0.08, 0.66, 0.84, w=W_SEC, r=0.02)
    c.smooth([(0.36, 0.16), (0.46, 0.40), (0.62, 0.62), (0.80, 0.84)],
             w=W_MAIN * 2.4)
    c.arrow(0.14, 0.10, 0.14, 0.90, W_SEC, head=0.075)


def timelapse_04(c):
    """One cell in the first frame and two in the last: a division caught over frames."""
    for x in (0.05, 0.36, 0.67):
        c.rect(x, 0.20, 0.28, 0.46, w=W_SEC, r=0.02)
    c.cell(0.19, 0.43, 0.105, w=W_SEC, nuc=0.42)
    c.ell(0.44, 0.43, 0.075, 0.098, 0, W_SEC)
    c.ell(0.56, 0.43, 0.075, 0.098, 0, W_SEC)
    c.disc(0.44, 0.43, 0.032)
    c.disc(0.56, 0.43, 0.032)
    c.cell(0.74, 0.43, 0.072, w=W_SEC, nuc=0.44)
    c.cell(0.88, 0.43, 0.072, w=W_SEC, nuc=0.44)
    c.arrow(0.08, 0.84, 0.92, 0.84, W_SEC, head=0.065)


def timelapse_05(c):
    """A clock over the field: the cell now, and the dashed outline of where it was."""
    c.circ(0.74, 0.24, 0.185, W_SEC)
    c.line(0.74, 0.24, 0.74, 0.11, W_SEC)
    c.line(0.74, 0.24, 0.85, 0.29, W_SEC)
    c.ell(0.22, 0.74, 0.130, 0.115, 0, W_SEC, dash=[6, 6])
    c.cell(0.60, 0.68, 0.155, w=W_MAIN, nuc=0.40)
    c.arrow(0.35, 0.74, 0.44, 0.71, W_SEC, head=0.055)


def timelapse_06(c):
    """A playhead running along a bar of frame ticks, the frame it lands on above."""
    c.rect(0.20, 0.08, 0.60, 0.46, w=W_SEC, r=0.02)
    c.cell(0.50, 0.31, 0.135, w=W_SEC, nuc=0.42)
    c.polyline([(0.06, 0.70), (0.06, 0.92), (0.24, 0.81)], close=True, filled=True)
    c.line(0.34, 0.81, 0.94, 0.81, W_SEC)
    for x in (0.40, 0.52, 0.64, 0.76, 0.88):
        c.line(x, 0.74, x, 0.88, W_FINE)
    c.disc(0.52, 0.81, 0.055)


def timelapse_07(c):
    """The same cell three times over: dashed where it was, solid where it is now."""
    c.ell(0.19, 0.74, 0.115, 0.100, 0, W_SEC, dash=[5, 5])
    c.ell(0.43, 0.56, 0.128, 0.112, 0, W_SEC, dash=[6, 6])
    c.cell(0.73, 0.34, 0.165, w=W_MAIN, nuc=0.40)
    c.line(0.24, 0.68, 0.36, 0.62, W_FINE)
    c.line(0.51, 0.49, 0.61, 0.43, W_FINE)


def timelapse_08(c):
    """Two objects keeping their identity: matching links drawn from frame to frame."""
    c.rect(0.04, 0.20, 0.38, 0.60, w=W_SEC, r=0.02)
    c.rect(0.58, 0.20, 0.38, 0.60, w=W_SEC, r=0.02)
    c.disc(0.16, 0.36, 0.065)
    c.disc(0.28, 0.64, 0.065)
    c.disc(0.72, 0.30, 0.065)
    c.disc(0.86, 0.68, 0.065)
    c.line(0.16, 0.36, 0.72, 0.30, W_FINE, dash=[7, 7])
    c.line(0.28, 0.64, 0.86, 0.68, W_FINE, dash=[7, 7])
    c.arrow(0.20, 0.92, 0.80, 0.92, W_FINE, head=0.05)


def timelapse_09(c):
    """Frames stacked back into depth, the newest in front, along the time arrow."""
    for i, (x, y) in enumerate(((0.42, 0.04), (0.27, 0.21), (0.12, 0.38))):
        c.rect(x, y, 0.42, 0.38, w=W_SEC if i == 2 else W_FINE, r=0.02)
    c.cell(0.30, 0.68, 0.078, w=W_SEC, nuc=0.44)
    c.arrow(0.44, 0.92, 0.86, 0.62, W_SEC, head=0.075)


def timelapse_10(c):
    """A film reel: the strip spooled up, one cell showing through a window."""
    c.circ(0.46, 0.50, 0.400, W_MAIN)
    c.disc(0.46, 0.50, 0.070)
    for i in range(3):
        a = -math.pi / 2 + TAU * i / 3
        hx, hy = 0.46 + 0.225 * math.cos(a), 0.50 + 0.225 * math.sin(a)
        if i == 0:
            c.circ(hx, hy, 0.115, W_SEC)
            c.cell(hx, hy, 0.062, w=W_FINE, nuc=0.44)
        else:
            c.circ(hx, hy, 0.115, W_SEC)
    c.line(0.84, 0.36, 0.98, 0.30, W_SEC)
    c.line(0.86, 0.56, 0.99, 0.50, W_SEC)


# =====================================================================
# motility -- the PATH and its SPEED; a track here is measured
# =====================================================================

def motility_01(c):
    """A parasite at the head of the circular trail it has glided along."""
    c.circ(0.44, 0.54, 0.290, W_SEC)
    c.arrow(0.30, 0.245, 0.53, 0.270, W_MAIN, head=0.085)
    c.parasite(0.76, 0.54, 0.340, rot=100)
    c.disc(0.44, 0.83, 0.052)


def motility_02(c):
    """A trajectory with a speed dial reading its velocity off underneath."""
    c.smooth([(0.06, 0.32), (0.26, 0.18), (0.48, 0.34), (0.70, 0.18),
              (0.94, 0.30)], w=W_MAIN)
    c.disc(0.06, 0.32, 0.045)
    c.arc(0.50, 0.82, 0.290, 10, 160, W_SEC)
    for a in (20, 85, 150):
        r0, r1 = 0.235, 0.290
        ar = math.radians(a)
        c.line(0.50 + r0 * math.cos(ar), 0.82 - r0 * math.sin(ar),
               0.50 + r1 * math.cos(ar), 0.82 - r1 * math.sin(ar), W_FINE)
    ar = math.radians(58)
    c.line(0.50, 0.82, 0.50 + 0.235 * math.cos(ar), 0.82 - 0.235 * math.sin(ar),
           W_MAIN)
    c.disc(0.50, 0.82, 0.045)


def motility_03(c):
    """Equal time steps marked along one track: wide gaps where it moves fast."""
    pts = [(0.09, 0.76), (0.17, 0.735), (0.27, 0.695), (0.41, 0.625),
           (0.58, 0.525), (0.77, 0.400), (0.94, 0.255)]
    c.smooth(pts, w=W_SEC)
    for px, py in pts:
        c.disc(px, py, 0.044)
    c.arrow(0.09, 0.92, 0.94, 0.92, W_FINE, head=0.05)


def motility_04(c):
    """Two tracks compared: one long and straight, one short and tangled."""
    c.disc(0.08, 0.26, 0.050)
    c.arrow(0.08, 0.26, 0.94, 0.26, W_MAIN, head=0.075)
    c.line(0.04, 0.50, 0.96, 0.50, W_FINE, dash=[6, 7])
    c.disc(0.12, 0.72, 0.052)
    c.smooth([(0.12, 0.72), (0.30, 0.60), (0.44, 0.84), (0.24, 0.92),
              (0.34, 0.68), (0.50, 0.74)], w=W_MAIN)


def motility_05(c):
    """A wandering path with the straight start-to-end displacement struck across it."""
    c.smooth([(0.10, 0.80), (0.24, 0.42), (0.42, 0.66), (0.62, 0.28),
              (0.88, 0.42)], w=W_MAIN)
    c.disc(0.10, 0.80, 0.052)
    c.arrow(0.10, 0.80, 0.88, 0.42, W_SEC, head=0.070)


def motility_06(c):
    """The moving cell above, its speed plotted against time below."""
    c.cell(0.18, 0.20, 0.125, w=W_SEC, nuc=0.42)
    c.arrow(0.36, 0.20, 0.74, 0.20, W_SEC, head=0.065)
    c.axes(0.12, 0.46, 0.94, 0.92, w=W_SEC)
    c.smooth([(0.18, 0.86), (0.34, 0.58), (0.50, 0.78), (0.68, 0.52),
              (0.88, 0.66)], w=W_MAIN)


def motility_07(c):
    """Every track redrawn from one origin: displacement spokes of unequal length."""
    c.circ(0.50, 0.50, 0.400, W_FINE, dash=[7, 8])
    for a, r in ((-0.35, 0.38), (0.85, 0.22), (2.05, 0.34),
                 (3.35, 0.16), (4.55, 0.29)):
        c.arrow(0.50, 0.50, 0.50 + r * math.cos(a), 0.50 + r * math.sin(a),
                W_SEC, head=0.062)
    c.disc(0.50, 0.50, 0.048)


def motility_08(c):
    """The path length read off against a graduated scale bar beneath it."""
    c.smooth([(0.08, 0.52), (0.26, 0.24), (0.46, 0.48), (0.66, 0.22),
              (0.92, 0.40)], w=W_MAIN)
    c.disc(0.08, 0.52, 0.048)
    c.line(0.10, 0.80, 0.90, 0.80, W_SEC)
    for x in (0.10, 0.30, 0.50, 0.70, 0.90):
        c.line(x, 0.72, x, 0.88, W_SEC)


def motility_09(c):
    """The cell carrying its velocity vector: one big arrow off the object it moves."""
    c.smooth([(0.06, 0.86), (0.24, 0.76), (0.40, 0.64)], w=W_SEC, dash=[7, 7])
    c.cell(0.44, 0.60, 0.175, w=W_MAIN, nuc=0.40)
    c.arrow(0.58, 0.48, 0.94, 0.16, W_MAIN, head=0.105)


def motility_10(c):
    """Same elapsed time, two lanes: the fast object far past the slow one."""
    c.circ(0.50, 0.13, 0.110, W_SEC)
    c.line(0.50, 0.13, 0.50, 0.05, W_FINE)
    c.line(0.50, 0.13, 0.57, 0.17, W_FINE)
    c.line(0.06, 0.34, 0.06, 0.94, W_SEC)
    c.line(0.04, 0.64, 0.96, 0.64, W_FINE, dash=[6, 7])
    c.line(0.06, 0.48, 0.70, 0.48, W_SEC)
    c.disc(0.79, 0.48, 0.080)
    c.line(0.06, 0.82, 0.24, 0.82, W_SEC)
    c.disc(0.33, 0.82, 0.080)


# =====================================================================
# curate -- a human hand fixing something, on the record
# =====================================================================

def curate_01(c):
    """A brush pushing a wrong mask boundary back onto the cell."""
    blob(c, 0.42, 0.58, 0.300, w=W_MAIN)
    c.arc(0.42, 0.58, 0.380, 20, 130, W_FINE)
    c.disc(0.42, 0.58, 0.070)
    stylus(c, 0.62, 0.32, L=0.44, ang=42.0, wide=0.105)


def curate_02(c):
    """A pointer dragging one square control handle of a contour into place."""
    blob(c, 0.46, 0.56, 0.280, w=W_MAIN, phase=3)
    for a in (0.9, 2.5, 4.1, 5.6):
        hx = 0.46 + 0.295 * math.cos(a)
        hy = 0.56 + 0.258 * math.sin(a)
        c.rect(hx - 0.050, hy - 0.050, 0.100, 0.100, w=W_FINE, filled=True)
    c.rect(0.60, 0.13, 0.095, 0.095, w=W_SEC)
    c.line(0.605, 0.235, 0.545, 0.330, W_FINE, dash=[5, 5])
    cursor(c, 0.735, 0.215, 0.21)


def curate_03(c):
    """A hand-drawn stroke cutting one wrongly merged blob into two objects."""
    c.smooth([(0.14, 0.60), (0.26, 0.34), (0.46, 0.48), (0.66, 0.34),
              (0.78, 0.60), (0.66, 0.86), (0.46, 0.72), (0.26, 0.86)],
             w=W_MAIN, closed=True)
    c.disc(0.28, 0.60, 0.062)
    c.disc(0.64, 0.60, 0.062)
    c.line(0.46, 0.38, 0.46, 0.82, W_MAIN)
    stylus(c, 0.48, 0.34, L=0.40, ang=-40.0, wide=0.100)


def curate_04(c):
    """Two fragments pushed together and stitched into a single object."""
    blob(c, 0.24, 0.56, 0.175, w=W_MAIN, phase=1)
    blob(c, 0.76, 0.56, 0.175, w=W_MAIN, phase=5)
    c.disc(0.24, 0.56, 0.055)
    c.disc(0.76, 0.56, 0.055)
    c.ell(0.50, 0.56, 0.450, 0.290, 0, W_FINE, dash=[8, 8])
    c.arrow(0.40, 0.56, 0.475, 0.56, W_SEC, head=0.065)
    c.arrow(0.60, 0.56, 0.525, 0.56, W_SEC, head=0.065)
    tick(c, 0.50, 0.13, 0.095, W_MAIN)


def curate_05(c):
    """A broken track re-linked by hand across the gap it lost."""
    c.smooth([(0.06, 0.72), (0.18, 0.60), (0.30, 0.56)], w=W_MAIN)
    c.smooth([(0.66, 0.44), (0.80, 0.34), (0.94, 0.30)], w=W_MAIN)
    for px, py in ((0.06, 0.72), (0.30, 0.56), (0.66, 0.44), (0.94, 0.30)):
        c.disc(px, py, 0.045)
    c.line(0.32, 0.555, 0.64, 0.445, W_SEC, dash=[6, 6])
    stylus(c, 0.48, 0.56, L=0.38, ang=68.0, wide=0.095)


def curate_06(c):
    """An eraser lifting a false object off the field, the real ones left alone."""
    c.rect(0.06, 0.22, 0.88, 0.66, w=W_SEC, r=0.03)
    c.cell(0.24, 0.44, 0.115, w=W_SEC, nuc=0.42)
    c.cell(0.26, 0.74, 0.105, w=W_SEC, nuc=0.42)
    c.ell(0.64, 0.66, 0.125, 0.110, 0, W_FINE, dash=[5, 5])
    rot_rect(c, 0.70, 0.42, 0.34, 0.20, -32.0, w=W_MAIN)
    c.line(0.615, 0.325, 0.72, 0.49, W_FINE)


def curate_07(c):
    """One mask kept and one thrown out: a tick on the good outline, a cross on the bad."""
    c.rect(0.05, 0.12, 0.40, 0.46, w=W_SEC, r=0.03)
    c.cell(0.25, 0.35, 0.135, w=W_SEC, nuc=0.42)
    tick(c, 0.25, 0.78, 0.130, W_MAIN)
    c.rect(0.55, 0.12, 0.40, 0.46, w=W_FINE, r=0.03)
    blob(c, 0.75, 0.35, 0.135, w=W_FINE, phase=2)
    cross(c, 0.75, 0.78, 0.115, W_MAIN)


def curate_08(c):
    """Stepping back through revisions: an undo arc over the mask, versions as dots."""
    blob(c, 0.50, 0.62, 0.245, w=W_SEC)
    c.disc(0.50, 0.62, 0.060)
    c.arc(0.50, 0.44, 0.330, 20, 145, W_MAIN)
    c.arrow(0.24, 0.30, 0.17, 0.44, W_MAIN, head=0.085, tail=False)
    c.line(0.16, 0.94, 0.84, 0.94, W_FINE)
    for i, x in enumerate((0.16, 0.39, 0.61, 0.84)):
        if i == 2:
            c.circ(x, 0.94, 0.058, W_SEC)
        c.disc(x, 0.94, 0.030)


def curate_09(c):
    """The corrected object beside the log line the correction was written into."""
    blob(c, 0.26, 0.44, 0.200, w=W_MAIN)
    c.arc(0.26, 0.44, 0.255, 200, 130, W_FINE, dash=[5, 5])
    c.disc(0.26, 0.44, 0.055)
    c.rect(0.54, 0.14, 0.42, 0.72, w=W_SEC, r=0.03)
    for y in (0.30, 0.44, 0.58):
        c.line(0.61, y, 0.89, y, W_FINE)
    tick(c, 0.75, 0.73, 0.085, W_SEC)


def curate_10(c):
    """The finished mask signed off: an approval stamp pressed onto the object."""
    blob(c, 0.40, 0.40, 0.300, w=W_MAIN)
    c.disc(0.40, 0.40, 0.070)
    c.circ(0.73, 0.76, 0.195, W_SEC)
    c.circ(0.73, 0.76, 0.145, W_FINE, dash=[6, 6])
    tick(c, 0.73, 0.76, 0.088, W_MAIN)


# =====================================================================
# lineage -- CONTAINMENT: what is inside what (never a family tree)
# =====================================================================

def lineage_01(c):
    """Concentric rings: the pathogen solid at the core, inside the vacuole, inside the cell."""
    c.circ(0.50, 0.50, 0.420, W_MAIN)
    c.circ(0.50, 0.50, 0.280, W_SEC)
    c.circ(0.50, 0.50, 0.155, W_SEC)
    c.disc(0.50, 0.50, 0.075)


def lineage_02(c):
    """A containment tree: the cell node above, its nucleus and vacuole hanging under it."""
    c.cell(0.50, 0.18, 0.135, w=W_SEC, nuc=0.40)
    c.polyline([(0.24, 0.52), (0.24, 0.40), (0.76, 0.40), (0.76, 0.52)], w=W_FINE)
    c.line(0.50, 0.32, 0.50, 0.40, W_FINE)
    c.disc(0.24, 0.66, 0.135)
    c.ring(0.76, 0.66, 0.115, 0.045)
    c.disc(0.76, 0.66, 0.045)


def lineage_03(c):
    """Three frames nested one inside the next, the innermost filled solid."""
    c.rect(0.06, 0.06, 0.88, 0.88, w=W_MAIN, r=0.04)
    c.rect(0.20, 0.20, 0.60, 0.60, w=W_SEC, r=0.035)
    c.rect(0.33, 0.33, 0.34, 0.34, w=W_SEC, r=0.03)
    c.rect(0.42, 0.42, 0.16, 0.16, w=W_SEC, filled=True, r=0.03)


def lineage_04(c):
    """A wedge cut out of the cell, showing the nucleus and the vacuole inside it."""
    c.arc(0.50, 0.50, 0.420, -55, 305, W_MAIN)
    c.arc(0.50, 0.50, 0.270, -55, 305, W_SEC)
    a0, a1 = math.radians(-55), math.radians(-55 + 305)
    for a in (a0, a1):
        c.line(0.50, 0.50, 0.50 + 0.420 * math.cos(a), 0.50 - 0.420 * math.sin(a),
               W_FINE)
    c.disc(0.50, 0.50, 0.110)


def lineage_05(c):
    """'Is inside' read as a chain: the pathogen belongs to the vacuole belongs to the cell."""
    c.parasite(0.13, 0.50, 0.190, rot=90)
    c.arrow(0.30, 0.50, 0.22, 0.50, W_SEC, head=0.055)
    c.ring(0.46, 0.50, 0.115, 0.040)
    c.arrow(0.68, 0.50, 0.60, 0.50, W_SEC, head=0.055)
    c.cell(0.83, 0.50, 0.155, w=W_MAIN, nuc=0.36)


def lineage_06(c):
    """The cell taken apart: its nucleus and vacuole pulled out on dashed leaders."""
    c.ell(0.29, 0.68, 0.275, 0.250, 0, W_MAIN)
    c.circ(0.24, 0.62, 0.105, W_FINE, dash=[6, 6])
    c.circ(0.40, 0.80, 0.075, W_FINE, dash=[6, 6])
    c.line(0.28, 0.53, 0.60, 0.26, W_FINE, dash=[7, 7])
    c.line(0.48, 0.79, 0.78, 0.72, W_FINE, dash=[7, 7])
    c.disc(0.72, 0.20, 0.135)
    c.ring(0.88, 0.71, 0.095, 0.042)


def lineage_07(c):
    """An indented object list: the cell, the nucleus under it, the pathogen under that."""
    rows = ((0.10, 0.20), (0.28, 0.50), (0.46, 0.80))
    c.polyline([(0.155, 0.28), (0.155, 0.50), (0.245, 0.50)], w=W_FINE)
    c.polyline([(0.335, 0.58), (0.335, 0.80), (0.425, 0.80)], w=W_FINE)
    c.cell(0.155, 0.20, 0.090, w=W_SEC, nuc=0.42)
    c.disc(0.335, 0.50, 0.078)
    c.ring(0.515, 0.80, 0.070, 0.030)
    for x, y in rows:
        c.bar(x + 0.16, y - 0.045, 0.90 - (x + 0.16), 0.090, filled=True)


def lineage_08(c):
    """Brackets inside brackets: the outer holds the inner holds the object."""
    for x0, x1, y0, y1, w in ((0.08, 0.92, 0.10, 0.90, W_MAIN),
                              (0.26, 0.74, 0.26, 0.74, W_SEC)):
        arm = (x1 - x0) * 0.16
        c.polyline([(x0 + arm, y0), (x0, y0), (x0, y1), (x0 + arm, y1)], w=w)
        c.polyline([(x1 - arm, y0), (x1, y0), (x1, y1), (x1 - arm, y1)], w=w)
    c.disc(0.50, 0.50, 0.135)


def lineage_09(c):
    """One cell holding two vacuoles, and each vacuole holding its own pathogen."""
    c.ell(0.50, 0.52, 0.430, 0.400, 0, W_MAIN)
    c.disc(0.50, 0.19, 0.075)
    c.circ(0.34, 0.56, 0.150, W_SEC)
    c.disc(0.34, 0.56, 0.070)
    c.circ(0.70, 0.66, 0.125, W_SEC)
    c.disc(0.70, 0.66, 0.058)


def lineage_10(c):
    """Zooming inward: the cell, blown up into the vacuole it contains."""
    c.circ(0.30, 0.30, 0.240, W_MAIN)
    c.disc(0.22, 0.22, 0.065)
    c.circ(0.36, 0.38, 0.075, W_SEC)
    c.line(0.310, 0.436, 0.574, 0.865, W_FINE)
    c.line(0.410, 0.324, 0.866, 0.535, W_FINE)
    c.circ(0.72, 0.70, 0.220, W_MAIN)
    c.disc(0.72, 0.70, 0.100)


# =====================================================================
# layer_viewer -- a stack of overlays that can be toggled
# =====================================================================

def layer_viewer_01(c):
    """Three sheets floating apart: the points, the mask contour, the image."""
    sheet(c, 0.50, 0.18, 0.86, 0.28, W_SEC)
    for px, py in ((0.38, 0.17), (0.54, 0.23), (0.62, 0.13)):
        c.disc(px, py, 0.040)
    sheet(c, 0.50, 0.50, 0.86, 0.28, W_SEC)
    blob(c, 0.50, 0.50, 0.120, w=W_SEC, squash=0.55)
    sheet(c, 0.50, 0.82, 0.86, 0.28, W_SEC)
    c.ell(0.50, 0.82, 0.140, 0.062, 0, W_SEC)
    c.disc(0.50, 0.82, 0.042)


def layer_viewer_02(c):
    """A layer list: three rows, each with its own eye, the last one switched off."""
    for i, y in enumerate((0.24, 0.50, 0.76)):
        eye(c, 0.18, y, 0.115, W_SEC, open_=(i < 2))
        c.bar(0.38, y - 0.070, 0.56, 0.140, filled=(i == 0), w=W_SEC)


def layer_viewer_03(c):
    """The stack pulled apart, dashed guides keeping the sheets in register."""
    for cy in (0.20, 0.52, 0.84):
        sheet(c, 0.42, cy, 0.70, 0.24, W_SEC)
    c.line(0.07, 0.20, 0.07, 0.84, W_FINE, dash=[6, 7])
    c.line(0.77, 0.20, 0.77, 0.84, W_FINE, dash=[6, 7])
    c.arrow(0.92, 0.52, 0.92, 0.08, W_SEC, head=0.075)
    c.arrow(0.92, 0.52, 0.92, 0.96, W_SEC, head=0.075)


def layer_viewer_04(c):
    """The mask overlay peeled back off the image at one corner."""
    c.polyline([(0.08, 0.12), (0.62, 0.12), (0.92, 0.42), (0.92, 0.88),
                (0.08, 0.88)], w=W_MAIN, close=True)
    c.polyline([(0.62, 0.12), (0.92, 0.42), (0.62, 0.42)], w=W_SEC, close=True)
    c.cell(0.34, 0.58, 0.180, w=W_SEC, nuc=0.40)
    c.disc(0.76, 0.34, 0.048)


def layer_viewer_05(c):
    """Seen edge-on: an eye looking down through three separate sheets."""
    eye(c, 0.50, 0.16, 0.230, W_MAIN)
    c.line(0.50, 0.30, 0.50, 0.92, W_FINE, dash=[6, 7])
    for i, y in enumerate((0.48, 0.66, 0.84)):
        off = 0.06 * i
        c.line(0.12 + off, y, 0.88 - off, y, W_MAIN)


def layer_viewer_06(c):
    """One frame carrying every kind at once: the contour, the points and an ROI box."""
    c.rect(0.06, 0.10, 0.88, 0.80, w=W_SEC, r=0.03)
    blob(c, 0.36, 0.42, 0.195, w=W_MAIN)
    c.rect(0.50, 0.52, 0.38, 0.30, w=W_FINE)
    for px, py in ((0.60, 0.66), (0.72, 0.60), (0.80, 0.72)):
        c.disc(px, py, 0.040)


def layer_viewer_07(c):
    """A slider dimming the sheet on top of the one below it."""
    sheet(c, 0.50, 0.36, 0.80, 0.32, W_SEC)
    sheet(c, 0.50, 0.62, 0.80, 0.32, W_SEC)
    c.disc(0.50, 0.36, 0.060)
    c.line(0.14, 0.88, 0.86, 0.88, W_SEC)
    c.disc(0.62, 0.88, 0.080)
    c.line(0.14, 0.88, 0.54, 0.88, W_MAIN)


def layer_viewer_08(c):
    """A deck of sheets with one pulled out sideways to be worked on."""
    for i, (x, y) in enumerate(((0.06, 0.06), (0.10, 0.14), (0.14, 0.22))):
        c.rect(x, y, 0.40, 0.30, w=W_FINE if i < 2 else W_SEC, r=0.025)
    c.rect(0.48, 0.56, 0.46, 0.38, w=W_MAIN, r=0.025)
    c.cell(0.71, 0.75, 0.125, w=W_SEC, nuc=0.42)
    c.arrow(0.40, 0.56, 0.56, 0.68, W_SEC, head=0.065)


def layer_viewer_09(c):
    """Sheets fanned out like cards, each carrying a different kind of mark."""
    for ang, dx, mark in ((-20.0, -0.27, "dot"), (0.0, 0.0, "ring"),
                          (20.0, 0.27, "cross")):
        cx = 0.50 + dx
        rot_rect(c, cx, 0.54, 0.26, 0.60, ang, w=W_SEC)
        if mark == "dot":
            c.disc(cx, 0.54, 0.062)
        elif mark == "ring":
            c.circ(cx, 0.54, 0.065, W_SEC)
        else:
            cross(c, cx, 0.54, 0.065, W_SEC)


def layer_viewer_10(c):
    """One world: a pin driven through all three sheets, holding them in register."""
    for cy in (0.30, 0.54, 0.78):
        sheet(c, 0.50, cy, 0.80, 0.30, W_SEC)
    c.line(0.50, 0.16, 0.50, 0.86, W_MAIN)
    c.disc(0.50, 0.13, 0.075)


# =====================================================================
# image_scatter -- a scatter plot whose points are cell thumbnails
# =====================================================================

def _points(c, pts, r=0.035):
    for px, py in pts:
        c.disc(px, py, r)


def image_scatter_01(c):
    """A point on the axes blown up into a framed crop of the cell behind it."""
    c.axes(0.08, 0.08, 0.92, 0.90, w=W_SEC)
    _points(c, ((0.20, 0.74), (0.30, 0.58), (0.24, 0.44), (0.42, 0.66),
                (0.36, 0.30)), 0.038)
    c.circ(0.30, 0.58, 0.085, W_FINE)
    c.line(0.38, 0.53, 0.56, 0.36, W_FINE, dash=[6, 6])
    c.rect(0.54, 0.10, 0.36, 0.36, w=W_MAIN, r=0.02)
    c.cell(0.72, 0.28, 0.115, w=W_SEC, nuc=0.42)


def image_scatter_02(c):
    """A cursor resting on a point, the cell popping up beside it."""
    c.axes(0.08, 0.10, 0.90, 0.88, w=W_SEC)
    _points(c, ((0.20, 0.74), (0.32, 0.62), (0.26, 0.46), (0.44, 0.70),
                (0.52, 0.50), (0.40, 0.34)), 0.038)
    c.rect(0.50, 0.12, 0.36, 0.30, w=W_SEC, r=0.03)
    c.cell(0.68, 0.27, 0.095, w=W_FINE, nuc=0.42)
    c.line(0.60, 0.42, 0.55, 0.50, W_FINE)
    cursor(c, 0.54, 0.52, 0.22)


def image_scatter_03(c):
    """The points themselves are crops: thumbnails sitting where their cells fall."""
    c.axes(0.08, 0.08, 0.92, 0.92, w=W_SEC)
    for x, y, s in ((0.16, 0.60, 0.22), (0.42, 0.30, 0.22), (0.58, 0.62, 0.22)):
        crop(c, x, y, s, w=W_SEC, obj=0.28)
    _points(c, ((0.44, 0.84), (0.80, 0.30), (0.86, 0.74)), 0.042)


def image_scatter_04(c):
    """A lens over the cloud, magnifying one point into the cell it stands for."""
    c.axes(0.06, 0.08, 0.94, 0.92, w=W_SEC)
    _points(c, ((0.18, 0.78), (0.28, 0.62), (0.20, 0.46), (0.40, 0.80),
                (0.34, 0.44), (0.46, 0.62), (0.30, 0.28)), 0.038)
    c.magnifier(0.68, 0.40, 0.245, ang_deg=55.0, w=W_MAIN, handle=0.55)
    c.cell(0.68, 0.40, 0.150, w=W_SEC, nuc=0.42)


def image_scatter_05(c):
    """Crops on the left becoming points on the axes on the right."""
    for i, y in enumerate((0.14, 0.42, 0.70)):
        crop(c, 0.04, y, 0.24, w=W_SEC, obj=0.30)
    c.arrow(0.33, 0.50, 0.45, 0.50, W_SEC, head=0.060)
    c.axes(0.50, 0.14, 0.96, 0.90, w=W_SEC)
    _points(c, ((0.60, 0.72), (0.70, 0.50), (0.86, 0.32)), 0.048)


def image_scatter_06(c):
    """A lasso thrown round a few points, and the crop of one of them."""
    c.axes(0.06, 0.10, 0.94, 0.90, w=W_SEC)
    _points(c, ((0.22, 0.74), (0.34, 0.62), (0.26, 0.52), (0.56, 0.78),
                (0.68, 0.60), (0.80, 0.40)), 0.040)
    c.smooth([(0.14, 0.62), (0.28, 0.42), (0.46, 0.58), (0.34, 0.82),
              (0.16, 0.78)], w=W_SEC, closed=True, dash=[7, 7])
    c.rect(0.56, 0.14, 0.34, 0.30, w=W_MAIN, r=0.03)
    c.cell(0.73, 0.29, 0.100, w=W_FINE, nuc=0.42)


def image_scatter_07(c):
    """A window split in two: the plot on one side, the picked cell previewed on the other."""
    c.rect(0.04, 0.12, 0.92, 0.76, w=W_SEC, r=0.03)
    c.line(0.58, 0.12, 0.58, 0.88, W_SEC)
    c.axes(0.12, 0.22, 0.52, 0.80, w=W_FINE)
    _points(c, ((0.20, 0.68), (0.30, 0.54), (0.24, 0.42), (0.40, 0.60),
                (0.44, 0.36)), 0.034)
    c.circ(0.30, 0.54, 0.070, W_FINE)
    c.cell(0.77, 0.50, 0.150, w=W_MAIN, nuc=0.42)


def image_scatter_08(c):
    """Two clusters, one crop opened out of each, so the groups can be compared."""
    _points(c, ((0.16, 0.70), (0.26, 0.80), (0.28, 0.62), (0.18, 0.86)), 0.042)
    _points(c, ((0.60, 0.34), (0.70, 0.44), (0.72, 0.26), (0.60, 0.50)), 0.042)
    c.line(0.30, 0.70, 0.44, 0.60, W_FINE, dash=[6, 6])
    c.rect(0.44, 0.56, 0.26, 0.26, w=W_SEC, r=0.02)
    c.cell(0.57, 0.69, 0.082, w=W_FINE, nuc=0.42)
    c.line(0.74, 0.32, 0.80, 0.24, W_FINE, dash=[6, 6])
    c.rect(0.70, 0.06, 0.26, 0.26, w=W_SEC, r=0.02)
    c.cell(0.83, 0.19, 0.082, w=W_FINE, nuc=0.42)


def image_scatter_09(c):
    """A gallery strip under the plot, each crop tied back to its point."""
    c.axes(0.08, 0.08, 0.92, 0.58, w=W_SEC)
    _points(c, ((0.22, 0.44), (0.42, 0.28), (0.62, 0.40), (0.78, 0.20)), 0.044)
    for i, x in enumerate((0.10, 0.38, 0.66)):
        crop(c, x, 0.70, 0.24, w=W_SEC, obj=0.30)
    c.line(0.22, 0.48, 0.22, 0.68, W_FINE, dash=[5, 5])
    c.line(0.50, 0.32, 0.50, 0.68, W_FINE, dash=[5, 5])


def image_scatter_10(c):
    """The plot binned into tiles, each tile showing the cell that stands for it."""
    c.rect(0.10, 0.10, 0.80, 0.80, w=W_SEC, r=0.02)
    c.line(0.10, 0.3667, 0.90, 0.3667, W_FINE)
    c.line(0.10, 0.6333, 0.90, 0.6333, W_FINE)
    c.line(0.3667, 0.10, 0.3667, 0.90, W_FINE)
    c.line(0.6333, 0.10, 0.6333, 0.90, W_FINE)
    c.cell(0.2333, 0.50, 0.085, w=W_SEC, nuc=0.44)
    c.cell(0.50, 0.2333, 0.085, w=W_SEC, nuc=0.44)
    c.cell(0.7667, 0.7667, 0.085, w=W_SEC, nuc=0.44)
    _points(c, ((0.50, 0.50), (0.7667, 0.2333), (0.2333, 0.7667)), 0.040)


# =====================================================================
# graph_builder -- columns dragged onto shelves
# =====================================================================

def graph_builder_01(c):
    """Three column chips on the left, and the chart they build on the right."""
    for y in (0.16, 0.44, 0.72):
        chip(c, 0.04, y, 0.34, 0.13, w=W_SEC)
    c.arrow(0.42, 0.50, 0.54, 0.50, W_SEC, head=0.060)
    c.axes(0.60, 0.14, 0.96, 0.88, w=W_SEC)
    for i, (x, h) in enumerate(((0.66, 0.24), (0.76, 0.46), (0.86, 0.34))):
        c.rect(x, 0.88 - h, 0.085, h, w=W_SEC, filled=True)


def graph_builder_02(c):
    """A cursor dragging a column chip into a dashed drop target."""
    slot(c, 0.10, 0.62, 0.80, 0.20, W_SEC)
    chip(c, 0.16, 0.16, 0.56, 0.20, filled=False, w=W_MAIN)
    c.line(0.44, 0.38, 0.44, 0.58, W_FINE, dash=[6, 6])
    c.arrow(0.44, 0.40, 0.44, 0.60, W_SEC, head=0.070, tail=False)
    cursor(c, 0.60, 0.24, 0.22)


def graph_builder_03(c):
    """Table header cells lifted out of the table and flown onto the axes."""
    c.rect(0.06, 0.52, 0.44, 0.42, w=W_SEC, r=0.02)
    c.line(0.06, 0.66, 0.50, 0.66, W_SEC)
    c.line(0.28, 0.52, 0.28, 0.94, W_FINE)
    c.line(0.06, 0.80, 0.50, 0.80, W_FINE)
    chip(c, 0.30, 0.10, 0.26, 0.13, w=W_SEC)
    c.arrow(0.18, 0.50, 0.34, 0.26, W_SEC, head=0.060)
    c.axes(0.62, 0.14, 0.96, 0.60, w=W_SEC)
    c.smooth([(0.68, 0.52), (0.78, 0.32), (0.92, 0.22)], w=W_MAIN)


def graph_builder_04(c):
    """Four empty shelves, one for each encoding: x, y, colour and size."""
    marks = ("x", "y", "colour", "size")
    for i, y in enumerate((0.10, 0.32, 0.54, 0.76)):
        m = marks[i]
        if m == "x":
            c.arrow(0.06, y + 0.07, 0.24, y + 0.07, W_SEC, head=0.050)
        elif m == "y":
            c.arrow(0.15, y + 0.14, 0.15, y, W_SEC, head=0.050)
        elif m == "colour":
            c.disc(0.15, y + 0.07, 0.070)
        else:
            c.circ(0.10, y + 0.07, 0.035, W_SEC)
            c.circ(0.21, y + 0.07, 0.062, W_SEC)
        slot(c, 0.32, y, 0.62, 0.14, W_SEC)


def graph_builder_05(c):
    """Bars snapping into an empty chart frame out of a stack of blocks."""
    c.axes(0.48, 0.10, 0.96, 0.90, w=W_SEC)
    for x, h in ((0.56, 0.34), (0.68, 0.56)):
        c.rect(x, 0.90 - h, 0.10, h, w=W_SEC, filled=True)
    slot(c, 0.80, 0.46, 0.10, 0.44, W_FINE, r=0.02)
    for y in (0.16, 0.38, 0.60):
        c.rect(0.06, y, 0.26, 0.16, w=W_SEC, r=0.02)
    c.arrow(0.36, 0.46, 0.44, 0.52, W_SEC, head=0.055)


def graph_builder_06(c):
    """A column dropped on the facet shelf splits one chart into four small ones."""
    chip(c, 0.06, 0.06, 0.34, 0.15, filled=True, w=W_SEC)
    c.arrow(0.46, 0.135, 0.62, 0.135, W_SEC, head=0.060)
    for x in (0.14, 0.56):
        for y in (0.34, 0.68):
            c.rect(x, y, 0.30, 0.26, w=W_SEC, r=0.02)
            c.smooth([(x + 0.05, y + 0.20), (x + 0.15, y + 0.09),
                      (x + 0.25, y + 0.14)], w=W_FINE)


def graph_builder_07(c):
    """A column on the size shelf: the points come out big and small."""
    chip(c, 0.06, 0.10, 0.34, 0.15, filled=True, w=W_SEC)
    slot(c, 0.06, 0.36, 0.34, 0.15, W_SEC)
    c.arrow(0.30, 0.28, 0.30, 0.35, W_SEC, head=0.055, tail=False)
    c.axes(0.50, 0.14, 0.96, 0.90, w=W_SEC)
    c.disc(0.62, 0.74, 0.035)
    c.disc(0.74, 0.54, 0.062)
    c.disc(0.88, 0.32, 0.090)


def graph_builder_08(c):
    """A column on the colour shelf: the marks split into filled and hollow."""
    chip(c, 0.06, 0.10, 0.34, 0.15, filled=True, w=W_SEC)
    slot(c, 0.06, 0.36, 0.34, 0.15, W_SEC)
    c.arrow(0.30, 0.28, 0.30, 0.35, W_SEC, head=0.055, tail=False)
    c.axes(0.50, 0.14, 0.96, 0.90, w=W_SEC)
    c.disc(0.62, 0.72, 0.055)
    c.disc(0.74, 0.60, 0.055)
    c.circ(0.78, 0.36, 0.055, W_SEC)
    c.circ(0.90, 0.50, 0.055, W_SEC)


def graph_builder_09(c):
    """An empty chart blueprint: dashed slots waiting on the x and the y axis."""
    c.axes(0.22, 0.10, 0.94, 0.72, w=W_MAIN)
    slot(c, 0.34, 0.80, 0.48, 0.15, W_SEC)
    c.polyline([(0.04, 0.62), (0.04, 0.22), (0.19, 0.22), (0.19, 0.62)],
               w=W_SEC, close=True, dash=[3, 3])
    c.smooth([(0.30, 0.62), (0.52, 0.42), (0.86, 0.22)], w=W_SEC, dash=[6, 6])


def graph_builder_10(c):
    """The finished chart with the chips that made it still docked on its axes."""
    c.axes(0.26, 0.06, 0.94, 0.70, w=W_SEC)
    for i, (x, h) in enumerate(((0.34, 0.26), (0.48, 0.48), (0.62, 0.36),
                                (0.76, 0.56))):
        c.rect(x, 0.70 - h, 0.10, h, w=W_SEC, filled=True)
    chip(c, 0.34, 0.80, 0.56, 0.15, w=W_MAIN)
    rot_rect(c, 0.11, 0.38, 0.56, 0.15, -90.0, w=W_MAIN)


# =====================================================================
# analyze_plaques -- cleared zones in a monolayer, counted and sized
# =====================================================================

def analyze_plaques_01(c):
    """A solid lawn with three plaque-shaped holes knocked clean through it."""
    punched(c, 0.06, 0.16, 0.88, 0.68,
            ((0.28, 0.40, 0.135), (0.62, 0.34, 0.095), (0.58, 0.66, 0.165)),
            r=0.04)


def analyze_plaques_02(c):
    """A square monolayer of packed cells with three bare gaps in it."""
    holes = ((0.32, 0.34, 0.190), (0.68, 0.67, 0.215))
    c.rect(0.04, 0.08, 0.92, 0.84, w=W_MAIN, r=0.03)
    lawn(c, 0.09, 0.13, 0.82, 0.74, holes, cols=6, rows=5, r=0.061, w=W_SEC)


def analyze_plaques_03(c):
    """Each clearing ringed by a detection marker, with the running count below."""
    c.rect(0.06, 0.08, 0.88, 0.62, w=W_SEC, r=0.03)
    spots = ((0.24, 0.26, 0.105), (0.66, 0.22, 0.070), (0.60, 0.52, 0.120))
    for sx, sy, sr in spots:
        c.lens(sx, sy, sr, sr * 0.55)
        c.circ(sx, sy, sr * 1.38, W_SEC, dash=[6, 6])
    for i, x in enumerate((0.30, 0.50, 0.70)):
        c.rect(x - 0.055, 0.80, 0.11, 0.11, w=W_SEC, filled=True, r=0.02)


def analyze_plaques_04(c):
    """The clearings redrawn largest to smallest along a size axis."""
    c.line(0.04, 0.80, 0.96, 0.80, W_SEC)
    for x, r in ((0.22, 0.180), (0.56, 0.125), (0.83, 0.075)):
        c.lens(x, 0.80 - r * 0.70, r, r * 0.62)
        c.line(x - r, 0.86, x + r, 0.86, W_FINE)
        c.line(x - r, 0.83, x - r, 0.89, W_FINE)
        c.line(x + r, 0.83, x + r, 0.89, W_FINE)
    c.arrow(0.10, 0.16, 0.90, 0.16, W_SEC, head=0.065)


def analyze_plaques_05(c):
    """A scan line across the field, its trace dropping to the floor over each clearing."""
    c.rect(0.06, 0.10, 0.88, 0.30, w=W_SEC, r=0.03)
    for x, r in ((0.28, 0.085), (0.52, 0.055), (0.76, 0.105)):
        c.lens(x, 0.25, r, r * 0.62)
    c.line(0.06, 0.25, 0.94, 0.25, W_FINE, dash=[6, 6])
    c.polyline([(0.06, 0.58), (0.20, 0.58), (0.20, 0.86), (0.36, 0.86),
                (0.36, 0.58), (0.46, 0.58), (0.46, 0.86), (0.58, 0.86),
                (0.58, 0.58), (0.68, 0.58), (0.68, 0.86), (0.84, 0.86),
                (0.84, 0.58), (0.94, 0.58)], w=W_MAIN)


def analyze_plaques_06(c):
    """Control against treated: one field full of clearings, one with a single clearing."""
    c.rect(0.04, 0.24, 0.42, 0.52, w=W_SEC, r=0.03)
    for x, y, r in ((0.14, 0.38, 0.062), (0.31, 0.34, 0.048),
                    (0.20, 0.60, 0.072), (0.36, 0.62, 0.052)):
        c.lens(x, y, r, r * 0.62)
    c.rect(0.54, 0.24, 0.42, 0.52, w=W_SEC, r=0.03)
    c.lens(0.75, 0.50, 0.070, 0.044)
    c.line(0.50, 0.14, 0.50, 0.86, W_FINE, dash=[6, 7])


def analyze_plaques_07(c):
    """The raw field on one side, its clearings picked out as solid shapes on the other."""
    c.rect(0.04, 0.24, 0.42, 0.52, w=W_SEC, r=0.03)
    for x, y, r in ((0.16, 0.40, 0.078), (0.32, 0.62, 0.092)):
        c.circ(x, y, r, W_SEC, dash=[5, 5])
    c.arrow(0.47, 0.50, 0.53, 0.50, W_SEC, head=0.050)
    c.rect(0.58, 0.24, 0.38, 0.52, w=W_SEC, r=0.03)
    c.lens(0.70, 0.40, 0.075, 0.048)
    c.lens(0.86, 0.62, 0.090, 0.058)


def analyze_plaques_08(c):
    """One clearing sized from its centre out: a radius arrow and the sizing rings."""
    c.lens(0.50, 0.50, 0.230, 0.140)
    c.circ(0.50, 0.50, 0.300, W_FINE, dash=[6, 6])
    c.circ(0.50, 0.50, 0.400, W_FINE, dash=[6, 6])
    c.arrow(0.50, 0.50, 0.50 + 0.400 * math.cos(-0.75),
            0.50 + 0.400 * math.sin(-0.75), W_MAIN, head=0.075)
    c.disc(0.50, 0.50, 0.045)


def analyze_plaques_09(c):
    """The clearings lifted out of the lawn into a row of separate objects."""
    c.rect(0.06, 0.10, 0.88, 0.42, w=W_SEC, r=0.03)
    src = ((0.24, 0.30, 0.095), (0.52, 0.26, 0.065), (0.76, 0.34, 0.115))
    for x, y, r in src:
        c.circ(x, y, r, W_FINE, dash=[5, 5])
    for i, (x, r) in enumerate(((0.24, 0.095), (0.52, 0.065), (0.76, 0.115))):
        c.arrow(x, 0.58, x, 0.66, W_FINE, head=0.045)
        c.lens(x, 0.82, r, r * 0.62)


def analyze_plaques_10(c):
    """A coverage bar reading off how much of the monolayer has been cleared."""
    c.rect(0.06, 0.10, 0.88, 0.50, w=W_SEC, r=0.03)
    for x, y, r in ((0.26, 0.28, 0.105), (0.62, 0.24, 0.070),
                    (0.60, 0.46, 0.090)):
        c.lens(x, y, r, r * 0.62)
    c.rect(0.06, 0.72, 0.88, 0.18, w=W_MAIN, r=0.09)
    c.bar(0.09, 0.75, 0.30, 0.12, filled=True)


# =====================================================================
# manifest
# =====================================================================

GROUPS = {
    "timelapse": ("timelapse -- the SAME objects across FRAMES; time is the axis", [
        ("Filmstrip of three frames, the same cell further along in each.",
         timelapse_01),
        ("One track threading the same object through three frames in a row.",
         timelapse_02),
        ("Kymograph: the object's position streaked down a vertical time axis.",
         timelapse_03),
        ("One cell in the first frame and two in the last: a division caught over frames.",
         timelapse_04),
        ("A clock over the field: the cell now, and the dashed outline of where it was.",
         timelapse_05),
        ("A playhead running along a bar of frame ticks, the frame it lands on above.",
         timelapse_06),
        ("The same cell three times over: dashed where it was, solid where it is now.",
         timelapse_07),
        ("Two objects keeping their identity: matching links drawn from frame to frame.",
         timelapse_08),
        ("Frames stacked back into depth, the newest in front, along the time arrow.",
         timelapse_09),
        ("A film reel: the strip spooled up, one cell showing through a window.",
         timelapse_10),
    ]),
    "motility": ("motility -- the PATH and its SPEED: a track here is measured", [
        ("A parasite at the head of the circular trail it has glided along.",
         motility_01),
        ("A trajectory with a speed dial reading its velocity off underneath.",
         motility_02),
        ("Equal time steps marked along one track: wide gaps where it moves fast.",
         motility_03),
        ("Two tracks compared: one long and straight, one short and tangled.",
         motility_04),
        ("A wandering path with the straight start-to-end displacement struck across it.",
         motility_05),
        ("The moving cell above, its speed plotted against time below.",
         motility_06),
        ("Every track redrawn from one origin: displacement spokes of unequal length.",
         motility_07),
        ("The path length read off against a graduated scale bar beneath it.",
         motility_08),
        ("The cell carrying its velocity vector: one big arrow off the object it moves.",
         motility_09),
        ("Same elapsed time, two lanes: the fast object far past the slow one.",
         motility_10),
    ]),
    "curate": ("curate -- a human hand fixing something, on the record", [
        ("A brush pushing a wrong mask boundary back onto the cell.", curate_01),
        ("A pointer dragging one square control handle of a contour into place.",
         curate_02),
        ("A hand-drawn stroke cutting one wrongly merged blob into two objects.",
         curate_03),
        ("Two fragments pushed together and stitched into a single object.",
         curate_04),
        ("A broken track re-linked by hand across the gap it lost.", curate_05),
        ("An eraser lifting a false object off the field, the real ones left alone.",
         curate_06),
        ("One mask kept and one thrown out: a tick on the good outline, a cross on the bad.",
         curate_07),
        ("Stepping back through revisions: an undo arc over the mask, versions as dots.",
         curate_08),
        ("The corrected object beside the log line the correction was written into.",
         curate_09),
        ("The finished mask signed off: an approval stamp pressed onto the object.",
         curate_10),
    ]),
    "lineage": ("lineage -- CONTAINMENT: what is inside what, cell to nucleus to pathogen", [
        ("Concentric rings: the pathogen solid at the core, inside the vacuole, inside the cell.",
         lineage_01),
        ("A containment tree: the cell node above, its nucleus and vacuole hanging under it.",
         lineage_02),
        ("Three frames nested one inside the next, the innermost filled solid.",
         lineage_03),
        ("A wedge cut out of the cell, showing the nucleus and the vacuole inside it.",
         lineage_04),
        ("'Is inside' read as a chain: the pathogen belongs to the vacuole belongs to the cell.",
         lineage_05),
        ("The cell taken apart: its nucleus and vacuole pulled out on dashed leaders.",
         lineage_06),
        ("An indented object list: the cell, the nucleus under it, the pathogen under that.",
         lineage_07),
        ("Brackets inside brackets: the outer holds the inner holds the object.",
         lineage_08),
        ("One cell holding two vacuoles, and each vacuole holding its own pathogen.",
         lineage_09),
        ("Zooming inward: the cell, blown up into the vacuole it contains.",
         lineage_10),
    ]),
    "layer_viewer": ("layer_viewer -- a stack of overlays that can be toggled", [
        ("Three sheets floating apart: the points, the mask contour, the image.",
         layer_viewer_01),
        ("A layer list: three rows, each with its own eye, the last one switched off.",
         layer_viewer_02),
        ("The stack pulled apart, dashed guides keeping the sheets in register.",
         layer_viewer_03),
        ("The mask overlay peeled back off the image at one corner.", layer_viewer_04),
        ("Seen edge-on: an eye looking down through three separate sheets.",
         layer_viewer_05),
        ("One frame carrying every kind at once: the contour, the points and an ROI box.",
         layer_viewer_06),
        ("A slider dimming the sheet on top of the one below it.", layer_viewer_07),
        ("A deck of sheets with one pulled out sideways to be worked on.",
         layer_viewer_08),
        ("Sheets fanned out like cards, each carrying a different kind of mark.",
         layer_viewer_09),
        ("One world: a pin driven through all three sheets, holding them in register.",
         layer_viewer_10),
    ]),
    "image_scatter": ("image_scatter -- a scatter plot whose points are cells", [
        ("A point on the axes blown up into a framed crop of the cell behind it.",
         image_scatter_01),
        ("A cursor resting on a point, the cell popping up beside it.",
         image_scatter_02),
        ("The points themselves are crops: thumbnails sitting where their cells fall.",
         image_scatter_03),
        ("A lens over the cloud, magnifying one point into the cell it stands for.",
         image_scatter_04),
        ("Crops on the left becoming points on the axes on the right.",
         image_scatter_05),
        ("A lasso thrown round a few points, and the crop of one of them.",
         image_scatter_06),
        ("A window split in two: the plot on one side, the picked cell previewed on the other.",
         image_scatter_07),
        ("Two clusters, one crop opened out of each, so the groups can be compared.",
         image_scatter_08),
        ("A gallery strip under the plot, each crop tied back to its point.",
         image_scatter_09),
        ("The plot binned into tiles, each tile showing the cell that stands for it.",
         image_scatter_10),
    ]),
    "graph_builder": ("graph_builder -- columns dragged onto x / y / colour / size shelves", [
        ("Three column chips on the left, and the chart they build on the right.",
         graph_builder_01),
        ("A cursor dragging a column chip into a dashed drop target.",
         graph_builder_02),
        ("Table header cells lifted out of the table and flown onto the axes.",
         graph_builder_03),
        ("Four empty shelves, one for each encoding: x, y, colour and size.",
         graph_builder_04),
        ("Bars snapping into an empty chart frame out of a stack of blocks.",
         graph_builder_05),
        ("A column dropped on the facet shelf splits one chart into four small ones.",
         graph_builder_06),
        ("A column on the size shelf: the points come out big and small.",
         graph_builder_07),
        ("A column on the colour shelf: the marks split into filled and hollow.",
         graph_builder_08),
        ("An empty chart blueprint: dashed slots waiting on the x and the y axis.",
         graph_builder_09),
        ("The finished chart with the chips that made it still docked on its axes.",
         graph_builder_10),
    ]),
    "analyze_plaques": ("analyze_plaques -- cleared zones in a monolayer, counted and sized", [
        ("A solid lawn with three plaque-shaped holes knocked clean through it.",
         analyze_plaques_01),
        ("A square monolayer of packed cells with three bare gaps in it.",
         analyze_plaques_02),
        ("Each clearing ringed by a detection marker, with the running count below.",
         analyze_plaques_03),
        ("The clearings redrawn largest to smallest along a size axis.",
         analyze_plaques_04),
        ("A scan line across the field, its trace dropping to the floor over each clearing.",
         analyze_plaques_05),
        ("Control against treated: one field full of clearings, one with a single clearing.",
         analyze_plaques_06),
        ("The raw field on one side, its clearings picked out as solid shapes on the other.",
         analyze_plaques_07),
        ("One clearing sized from its centre out: a radius arrow and the sizing rings.",
         analyze_plaques_08),
        ("The clearings lifted out of the lawn into a row of separate objects.",
         analyze_plaques_09),
        ("A coverage bar reading off how much of the monolayer has been cleared.",
         analyze_plaques_10),
    ]),
}


def main(outdir):
    return emit_groups(outdir, GROUPS, "group_imaging_explore.py")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else default_outdir(__file__)))
