#!/usr/bin/env python3
"""Candidate spaCR icons: the five registered apps that had no candidate folder.

``pca``, ``tabulate``, ``feature_dict``, ``outliers`` and ``dose_response``
landed after the drawing list was derived, so nobody ever drew for them.  Ten
conceptually different designs each, white-on-transparent flat vector art in
the house style set by ``plaque.png`` / ``measure.png``.

Four of these five are, left to themselves, "a chart with something on it" --
and the registry already contains ``graph_builder``, ``power``, ``barcode_qc``,
``classifier_evaluation``, ``regression``, ``hit_list``, ``plate_view`` and
``image_scatter``, which are also a chart with something on it.  Every
candidate below is held to one of these lines and none is allowed to wander::

    pca             AXES THAT WERE DERIVED RATHER THAN MEASURED.  Many
                    columns collapsing to two, a tilted pair of arrows that
                    is not the frame the data was recorded in, a scree plot,
                    a loadings biplot, a projection onto a plane.  Never a
                    bare scatter with a fitted line through it (``regression``)
                    and never a crop hanging off a point (``image_scatter``).
                    The one thing pca must NOT draw is a curve rising to a
                    plateau with a read-off dropped onto the x axis: that is
                    ``power_01`` exactly, which is why the "how many
                    components" idea here is a pie of shares and a scree
                    elbow rather than a cumulative-variance curve.

    tabulate        MANY ROWS BECOMING ONE CELL.  Row headers down the side,
                    column headers across the top, margins, a brace over a
                    group giving one output row.  The grid is always a TABLE
                    -- a header band, square cells, >= 3 columns -- never the
                    rounded wells inside a plate outline (``plate_view``,
                    ``experiment_design``) and never the 2x2 square with a
                    diagonal, which belongs to ``agreement``.  Nothing here
                    is dragged onto a shelf: the chip-onto-a-drop-target
                    picture is ``graph_builder``'s and is not reused, even
                    though Tabulate really does have shelves.

    feature_dict    A NAME RESOLVING TO A MEANING.  The cells hold WORDS.
                    Every candidate contains a name and the thing the name
                    resolves to -- a definition, a card, a book, a question
                    asked of a column header.  Nothing is measured and no
                    cell holds a value, which is the whole line against
                    ``tabulate`` and ``measure``.

    outliers        ONE MARK THAT DOES NOT BELONG WITH THE OTHERS.  A crowd
                    and a stray, always singled out by DISTANCE from the
                    crowd and never by rank -- ``hit_list`` owns the sorted
                    list with its leaders starred, and an outlier is not the
                    top of anything.  The fences are estimated from the crowd
                    itself (a box, an ellipse), never learned from labelled
                    classes, which is the line against
                    ``classifier_evaluation_09``.  No well ever carries a
                    graded value: every other well is identical, so the odd
                    one is odd rather than hot (``plate_view``).

    dose_response   A SIGMOID READ AT ITS INFLECTION.  ``power`` already owns
                    "a curve rising to a plateau with a marker on the x
                    axis", so the line is drawn at the shape: every curve
                    here has TWO flat ends, and the marker sits at the
                    half-way point between them with the 50% guide drawn in
                    -- a midpoint, not a threshold crossing.  The x axis is a
                    log concentration axis and shows it.  ``regression_08``
                    is "a sigmoid fitted to the points", so no candidate here
                    leads with scattered observations and a fit through them;
                    these lead with the read-off, the interval on it, and the
                    two cases where the module refuses to give a number.

48 px is the real constraint -- that is the size the tile is drawn at, and it
is what discarded the first pass of several of these.  So every design is a
handful of large, high-contrast elements: grids are 3x3 or 4x3 and never a
real pivot table, a "word" is a filled capsule bar rather than a hairline, dot
clouds are six or seven discs rather than a spray, and strokes stay at
``W_MAIN``/``W_SEC`` because ``W_FINE`` is half a pixel at 48 px.

Run standalone (deterministic -- no random draws at all)::

    QT_QPA_PLATFORM=offscreen python group_pca_tabulate_dict_outliers_dose.py [OUTDIR]

Default OUTDIR is the backup_icons directory one level up.  Writes
``<OUTDIR>/<key>/<key>_NN.png`` plus CONCEPTS.md and the two contact sheets via
:func:`_emit.emit_groups`.  It never touches anything in
``spacr/resources/icons/*.png``.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _draw import W_FINE, W_MAIN, W_SEC  # noqa: E402
from _emit import default_outdir, emit_groups  # noqa: E402

TAU = math.pi * 2
DASH = [1.4, 1.6]


# ---------------------------------------------------------------------------
# shared sub-drawings
# ---------------------------------------------------------------------------

def dots(c, pts, r=0.030):
    """A handful of solid points -- a 'population', never a spray."""
    for x, y in pts:
        c.disc(x, y, r)


def ringed(c, cx, cy, r=0.030, gap=0.042, w=W_SEC):
    """A point with a ring held off it: THIS one."""
    c.disc(cx, cy, r)
    c.circ(cx, cy, r + gap, w)


def flag(c, x, ybase, h=0.17, fw=0.105, w=W_SEC):
    """A pole with a solid pennant -- the 'flagged' mark."""
    c.line(x, ybase, x, ybase - h, w)
    c.polyline([(x, ybase - h), (x + fw, ybase - h + 0.048),
                (x, ybase - h + 0.096)], close=True, filled=True)


def caliper(c, x1, y1, x2, y2, t=0.038, w=W_SEC):
    """A measured span: a line with a perpendicular bar at each end."""
    ang = math.atan2(y2 - y1, x2 - x1)
    px, py = -math.sin(ang) * t, math.cos(ang) * t
    c.line(x1, y1, x2, y2, w)
    c.line(x1 - px, y1 - py, x1 + px, y1 + py, w)
    c.line(x2 - px, y2 - py, x2 + px, y2 + py, w)


def grid(c, x, y, w, h, rows, cols, w_=W_SEC, outer=W_MAIN):
    """A TABLE: square cells, ruled, framed.  Never a plate of wells."""
    c.rect(x, y, w, h, w=outer)
    for i in range(1, cols):
        c.line(x + w * i / cols, y, x + w * i / cols, y + h, w_)
    for j in range(1, rows):
        c.line(x, y + h * j / rows, x + w, y + h * j / rows, w_)


def cell_rect(x, y, w, h, rows, cols, j, i, inset=0.20):
    """Geometry of cell (row j, col i) shrunk by ``inset`` of the cell."""
    cw, ch = w / cols, h / rows
    ix, iy = cw * inset, ch * inset
    return (x + cw * i + ix, y + ch * j + iy, cw - 2 * ix, ch - 2 * iy)


def cell_fill(c, x, y, w, h, rows, cols, j, i, inset=0.20, r=0.010):
    c.rect(*cell_rect(x, y, w, h, rows, cols, j, i, inset), filled=True, r=r)


def cell_bar(c, x, y, w, h, rows, cols, j, i, frac=0.62, th=0.030):
    """A capsule 'value' inside a cell -- a number, drawn thick enough to survive."""
    cx0, cy0, cw, ch = cell_rect(x, y, w, h, rows, cols, j, i, inset=0.12)
    bw = cw * frac
    c.bar(cx0 + (cw - bw) / 2.0, cy0 + ch / 2.0 - th / 2.0, bw, th, filled=True)


def textline(c, x, y, w, h=0.032):
    """A capsule bar standing in for a word (survives 48 px)."""
    c.bar(x, y - h / 2.0, w, h, filled=True)


def brace_down(c, x0, x1, y, depth=0.055, w=W_SEC):
    """A brace opening downwards: 'these, together'."""
    c.polyline([(x0, y - depth), (x0, y), (x1, y), (x1, y - depth)], w=w)


def brace_right(c, y0, y1, x, depth=0.055, w=W_SEC):
    """A brace opening to the right: 'these rows, together'."""
    c.polyline([(x - depth, y0), (x, y0), (x, y1), (x - depth, y1)], w=w)


def derived_cross(c, cx, cy, L=0.30, rot=-30.0, minor=0.60, w=W_MAIN,
                  head=0.070):
    """A TILTED cross through a centre: axes that were derived, not measured.

    Drawn through the middle rather than off a corner, because a pair of arrows
    from a corner reads as a tick or a V at 48 px and a cross does not.
    """
    a = math.radians(rot)
    b = a - math.pi / 2
    c.line(cx - L * math.cos(a), cy - L * math.sin(a),
           cx + L * math.cos(a), cy + L * math.sin(a), w)
    c.arrow(cx, cy, cx + L * math.cos(a), cy + L * math.sin(a), w, head=head,
            tail=False)
    m = L * minor
    c.line(cx - m * math.cos(b), cy - m * math.sin(b),
           cx + m * math.cos(b), cy + m * math.sin(b), w)
    c.arrow(cx, cy, cx + m * math.cos(b), cy + m * math.sin(b), w, head=head,
            tail=False)


def sig_pts(x0, x1, ybot, ytop, k=11.0, mid=0.5, n=34):
    """A FULL sigmoid: flat at both ends, normalised to run ybot -> ytop."""
    vals = [1.0 / (1.0 + math.exp(-k * (i / (n - 1.0) - mid))) for i in range(n)]
    lo, hi = vals[0], vals[-1]
    return [(x0 + (x1 - x0) * i / (n - 1.0),
             ybot - (ybot - ytop) * (v - lo) / (hi - lo))
            for i, v in enumerate(vals)]


def log_ticks(c, x0, x1, y, n=5, t=0.030, w=W_SEC):
    """Decade ticks compressing to the right: this axis is logarithmic."""
    for i in range(n):
        f = math.log10(1.0 + 9.0 * i / (n - 1.0))
        x = x0 + (x1 - x0) * f
        c.line(x, y, x, y + t, w)


def qmark(c, cx, cy, s, w=W_MAIN):
    """A question mark, drawn as a path (no font is ever rasterised here)."""
    c.smooth([(cx - 0.58 * s, cy - 0.44 * s), (cx - 0.34 * s, cy - 0.86 * s),
              (cx + 0.20 * s, cy - 0.88 * s), (cx + 0.46 * s, cy - 0.48 * s),
              (cx + 0.10 * s, cy - 0.12 * s), (cx, cy + 0.20 * s)], w=w)
    c.disc(cx, cy + 0.62 * s, s * 0.155)


def magn_over(c, cx, cy, r, w=W_MAIN, ang=52.0, handle=0.72):
    """A magnifier -- used here only ever over a NAME, never over data."""
    c.circ(cx, cy, r, w)
    a = math.radians(ang)
    c.line(cx + r * math.cos(a), cy + r * math.sin(a),
           cx + r * (1 + handle) * math.cos(a), cy + r * (1 + handle) * math.sin(a),
           w * 1.3)


def tick(c, cx, cy, s, w=W_MAIN):
    c.polyline([(cx - s, cy + s * 0.05), (cx - s * 0.25, cy + s * 0.70),
                (cx + s, cy - s * 0.78)], w=w)


def cross(c, cx, cy, s, w=W_MAIN):
    c.line(cx - s * 0.74, cy - s * 0.74, cx + s * 0.74, cy + s * 0.74, w)
    c.line(cx + s * 0.74, cy - s * 0.74, cx - s * 0.74, cy + s * 0.74, w)


def wedge(c, cx, cy, r, a0, a1):
    """A solid pie slice, angles in degrees measured clockwise from 12 o'clock."""
    from PySide6.QtGui import QPainterPath
    pa = QPainterPath()
    pa.moveTo(c.pt(cx, cy))
    steps = max(2, int(abs(a1 - a0) / 6) + 2)
    for i in range(steps + 1):
        a = math.radians(-90.0 + a0 + (a1 - a0) * i / steps)
        pa.lineTo(c.pt(cx + r * math.cos(a), cy + r * math.sin(a)))
    pa.closeSubpath()
    c.fill(pa)


# =====================================================================
# pca -- AXES THAT WERE DERIVED RATHER THAN MEASURED
# =====================================================================

def pca_01(c):
    """Many feature columns collapsed onto two derived axes."""
    for j in range(5):
        textline(c, 0.04, 0.22 + j * 0.145, 0.28, h=0.070)
    c.arrow(0.36, 0.50, 0.50, 0.50, W_MAIN, head=0.080)
    derived_cross(c, 0.74, 0.50, L=0.24, rot=-32.0, minor=0.66, w=W_MAIN,
                  head=0.075)
    dots(c, [(0.66, 0.30), (0.88, 0.72)], r=0.042)


def pca_02(c):
    """A scree plot: component variance falling away, the elbow ringed."""
    c.axes(0.14, 0.14, 0.94, 0.82, W_SEC)
    base, top = 0.82, 0.18
    hs = (1.00, 0.58, 0.26, 0.15, 0.10)
    bw = 0.115
    for i, f in enumerate(hs):
        x = 0.20 + i * 0.152
        # kept components solid, the tail past the elbow left hollow
        c.rect(x - bw / 2, base - (base - top) * f, bw, (base - top) * f,
               filled=(i < 2), w=W_SEC)
    ex = 0.20 + 2 * 0.152
    c.circ(ex, base - (base - top) * hs[2], 0.092, W_MAIN)


def pca_03(c):
    """A loadings biplot: the features drawn as vectors off the origin."""
    ox, oy = 0.46, 0.54
    dots(c, [(0.30, 0.74), (0.68, 0.30)], r=0.036)
    for ang, L in ((-62, 0.42), (-6, 0.40), (62, 0.34), (156, 0.36)):
        a = math.radians(ang)
        c.arrow(ox, oy, ox + L * math.cos(a), oy + L * math.sin(a), W_MAIN,
                head=0.078)
    c.disc(ox, oy, 0.040)


def pca_04(c):
    """The cloud's own long and short directions, drawn through it."""
    c.ell(0.50, 0.50, 0.36, 0.20, rot=-27.0, w=W_MAIN)
    a = math.radians(-27.0)
    b = a + math.pi / 2
    c.line(0.50 - 0.33 * math.cos(a), 0.50 - 0.33 * math.sin(a),
           0.50 + 0.33 * math.cos(a), 0.50 + 0.33 * math.sin(a), W_SEC)
    c.line(0.50 - 0.17 * math.cos(b), 0.50 - 0.17 * math.sin(b),
           0.50 + 0.17 * math.cos(b), 0.50 + 0.17 * math.sin(b), W_SEC)
    dots(c, [(0.34, 0.60), (0.44, 0.52), (0.58, 0.46), (0.66, 0.40)], r=0.030)


def pca_05(c):
    """The measured frame turned into the derived one."""
    c.line(0.08, 0.50, 0.92, 0.50, W_SEC, dash=DASH)
    c.line(0.50, 0.08, 0.50, 0.92, W_SEC, dash=DASH)
    derived_cross(c, 0.50, 0.50, L=0.40, rot=-36.0, minor=0.70, w=W_MAIN,
                  head=0.078)
    c.arc(0.50, 0.50, 0.30, 40, 50, W_SEC)
    a1, a2 = math.radians(46.0), math.radians(38.0)
    c.arrow(0.50 + 0.30 * math.cos(a1), 0.50 - 0.30 * math.sin(a1),
            0.50 + 0.30 * math.cos(a2), 0.50 - 0.30 * math.sin(a2),
            W_SEC, head=0.070, tail=False)


def pca_06(c):
    """Points dropped out of many dimensions onto one flat plane."""
    c.polyline([(0.16, 0.72), (0.50, 0.60), (0.92, 0.72), (0.56, 0.88)],
               w=W_MAIN, close=True)
    for (px, py), (qx, qy) in (((0.34, 0.26), (0.36, 0.74)),
                               ((0.56, 0.16), (0.58, 0.68)),
                               ((0.72, 0.34), (0.72, 0.76))):
        c.disc(px, py, 0.040)
        c.line(px, py + 0.052, qx, qy - 0.032, W_SEC, dash=DASH)
        c.disc(qx, qy, 0.026)


def pca_07(c):
    """How much of the picture each component is: shares of one whole."""
    c.circ(0.50, 0.50, 0.36, W_MAIN)
    wedge(c, 0.50, 0.50, 0.355, 0, 175)
    c.line(0.50, 0.50, 0.50, 0.14, W_MAIN)
    a = math.radians(-90.0 + 265.0)
    c.line(0.50, 0.50, 0.50 + 0.355 * math.cos(a), 0.50 + 0.355 * math.sin(a),
           W_MAIN)


def pca_08(c):
    """Two groups that only come apart once the axes are derived."""
    c.arrow(0.14, 0.86, 0.14, 0.16, W_SEC, head=0.062)
    c.arrow(0.14, 0.86, 0.92, 0.86, W_MAIN, head=0.072)
    dots(c, [(0.28, 0.36), (0.36, 0.48), (0.24, 0.54), (0.36, 0.28)], r=0.042)
    dots(c, [(0.72, 0.40), (0.84, 0.52), (0.86, 0.30), (0.72, 0.60)], r=0.042)
    c.line(0.545, 0.20, 0.545, 0.70, W_SEC, dash=DASH)


def pca_09(c):
    """One component alone: the cloud folded down onto a single direction."""
    c.line(0.12, 0.74, 0.88, 0.30, W_MAIN)
    for t, off in ((0.16, -0.15), (0.38, 0.14), (0.60, -0.13), (0.82, 0.15)):
        bx, by = 0.12 + 0.76 * t, 0.74 - 0.44 * t
        # offset along the normal of the component line
        px, py = bx + off * 0.50, by + off * 0.87
        c.disc(px, py, 0.034)
        c.line(px, py, bx, by, W_SEC, dash=DASH)
        c.disc(bx, by, 0.024)


def pca_10(c):
    """Features on wildly different scales brought to one before decomposing."""
    base = 0.74
    c.line(0.04, base, 0.96, base, W_SEC)
    for i, f in enumerate((0.52, 0.14, 0.34, 0.08)):
        x = 0.07 + i * 0.085
        c.rect(x, base - f, 0.058, f, filled=True)
    c.arrow(0.43, 0.48, 0.57, 0.48, W_MAIN, head=0.072)
    for i in range(4):
        x = 0.62 + i * 0.085
        c.rect(x, base - 0.30, 0.058, 0.30, filled=True)
    c.line(0.60, base - 0.30, 0.96, base - 0.30, W_SEC, dash=DASH)


# =====================================================================
# tabulate -- MANY ROWS BECOMING ONE CELL
# =====================================================================

def tabulate_01(c):
    """The pivot: keys down the side, keys across the top, summaries between."""
    x, y, w, h = 0.10, 0.16, 0.80, 0.68
    grid(c, x, y, w, h, 4, 4)
    for i in range(1, 4):
        cell_fill(c, x, y, w, h, 4, 4, 0, i, inset=0.20)
    for j in range(1, 4):
        cell_fill(c, x, y, w, h, 4, 4, j, 0, inset=0.20)
    for j in range(1, 4):
        for i in range(1, 4):
            cell_bar(c, x, y, w, h, 4, 4, j, i, frac=0.66, th=0.034)


def tabulate_02(c):
    """A stack of raw rows collapsed into one summary cell."""
    for j in range(4):
        textline(c, 0.08, 0.22 + j * 0.115, 0.30, h=0.050)
    brace_right(c, 0.19, 0.60, 0.44, depth=0.055, w=W_SEC)
    c.arrow(0.50, 0.395, 0.62, 0.395, W_MAIN, head=0.072)
    c.rect(0.66, 0.26, 0.28, 0.28, w=W_MAIN)
    textline(c, 0.73, 0.40, 0.14, h=0.056)


def tabulate_03(c):
    """Margins: the totals along the far edge, ruled off from the body."""
    x, y, w, h = 0.08, 0.18, 0.84, 0.64
    grid(c, x, y, w, h, 4, 4)
    for i in range(4):
        cell_fill(c, x, y, w, h, 4, 4, 3, i, inset=0.24)
    for j in range(3):
        cell_fill(c, x, y, w, h, 4, 4, j, 3, inset=0.24)
    # a DOUBLE rule, so the margins are visibly set off from the body
    for d in (-0.014, 0.014):
        c.line(x + w * 0.75 + d, y, x + w * 0.75 + d, y + h, W_SEC)
        c.line(x, y + h * 0.75 + d, x + w, y + h * 0.75 + d, W_SEC)


def tabulate_04(c):
    """One group, three statistics of it: the count, the middle and the spread."""
    dots(c, [(0.28, 0.16), (0.44, 0.22), (0.60, 0.15), (0.74, 0.22)], r=0.040)
    brace_down(c, 0.24, 0.78, 0.36, depth=0.070, w=W_SEC)
    c.line(0.51, 0.36, 0.51, 0.46, W_SEC)
    x, y, w, h = 0.08, 0.50, 0.84, 0.30
    grid(c, x, y, w, h, 1, 3)
    c.disc(0.22, 0.65, 0.062)
    c.bar(0.42, 0.632, 0.16, 0.036, filled=True)
    c.line(0.78, 0.56, 0.78, 0.74, W_MAIN)
    c.line(0.71, 0.56, 0.85, 0.56, W_SEC)
    c.line(0.71, 0.74, 0.85, 0.74, W_SEC)


def tabulate_05(c):
    """The long raw table beside the short summary it becomes."""
    c.rect(0.05, 0.10, 0.30, 0.80, w=W_MAIN)
    for j in range(6):
        textline(c, 0.10, 0.185 + j * 0.128, 0.20, h=0.048)
    c.arrow(0.41, 0.50, 0.55, 0.50, W_SEC, head=0.062)
    c.rect(0.61, 0.32, 0.34, 0.36, w=W_MAIN)
    for j in range(2):
        textline(c, 0.67, 0.42 + j * 0.16, 0.22, h=0.058)


def tabulate_06(c):
    """A row key and a column key crossing on the one cell they share."""
    x, y, w, h = 0.10, 0.14, 0.80, 0.72
    grid(c, x, y, w, h, 3, 3)
    cw, ch = w / 3.0, h / 3.0
    for j in range(3):
        if j != 1:
            cell_bar(c, x, y, w, h, 3, 3, j, 1, frac=0.66, th=0.040)
    for i in range(3):
        if i != 1:
            cell_bar(c, x, y, w, h, 3, 3, 1, i, frac=0.66, th=0.040)
    cell_fill(c, x, y, w, h, 3, 3, 1, 1, inset=0.16, r=0.012)
    c.arrow(x + cw * 1.5, 0.045, x + cw * 1.5, y - 0.01, W_MAIN, head=0.062)
    c.arrow(0.035, y + ch * 1.5, x - 0.01, y + ch * 1.5, W_MAIN, head=0.062)


def tabulate_07(c):
    """Rows braced into groups, each brace giving out one row."""
    for k, y0 in ((0, 0.16), (1, 0.58)):
        for j in range(2):
            textline(c, 0.06, y0 + j * 0.13, 0.28, h=0.050)
        brace_right(c, y0 - 0.035, y0 + 0.165, 0.42, depth=0.055, w=W_SEC)
        c.arrow(0.47, y0 + 0.065, 0.60, y0 + 0.065, W_SEC, head=0.055)
        c.rect(0.64, y0 - 0.025, 0.30, 0.18, w=W_MAIN)
        textline(c, 0.71, y0 + 0.065, 0.16, h=0.044)


def tabulate_08(c):
    """A group opened into the subgroups it is made of."""
    c.rect(0.05, 0.10, 0.90, 0.80, w=W_MAIN)
    textline(c, 0.11, 0.24, 0.36, h=0.064)
    c.bar(0.74, 0.208, 0.15, 0.064, filled=True)
    for j in range(2):
        yy = 0.48 + j * 0.22
        c.polyline([(0.17, 0.30), (0.17, yy), (0.28, yy)], w=W_MAIN)
        textline(c, 0.32, yy, 0.24, h=0.056)
        c.bar(0.74, yy - 0.028, 0.15, 0.056, filled=True)


def tabulate_09(c):
    """A hole in the grid: the combination no row was found for."""
    x, y, w, h = 0.10, 0.16, 0.80, 0.68
    grid(c, x, y, w, h, 3, 3)
    for j in range(3):
        for i in range(3):
            if (j, i) == (1, 1):
                continue
            cell_bar(c, x, y, w, h, 3, 3, j, i, frac=0.60, th=0.036)
    bx, by, bw, bh = cell_rect(x, y, w, h, 3, 3, 1, 1, inset=0.14)
    c.rect(bx, by, bw, bh, w=W_SEC)
    c.line(bx, by + bh, bx + bw, by, W_SEC)


def tabulate_10(c):
    """The summary above, and the chart of it drawn column for column below."""
    x, y, w, h = 0.10, 0.10, 0.80, 0.36
    grid(c, x, y, w, h, 2, 3)
    for i in range(3):
        cell_fill(c, x, y, w, h, 2, 3, 0, i, inset=0.26)
    for i in range(3):
        cell_bar(c, x, y, w, h, 2, 3, 1, i, frac=0.60, th=0.034)
    base = 0.90
    for i, f in enumerate((0.20, 0.32, 0.12)):
        cx = x + w * (i + 0.5) / 3.0
        c.rect(cx - 0.075, base - f, 0.15, f, filled=True)
    c.line(0.06, base, 0.94, base, W_SEC)


# =====================================================================
# feature_dict -- A NAME RESOLVING TO A MEANING
# =====================================================================

def feature_dict_01(c):
    """A column name, and the meaning it resolves to."""
    c.rect(0.08, 0.08, 0.52, 0.18, w=W_MAIN, r=0.030)
    textline(c, 0.15, 0.17, 0.38, h=0.060)
    c.arrow(0.34, 0.30, 0.34, 0.50, W_MAIN, head=0.085)
    c.rect(0.06, 0.56, 0.88, 0.36, w=W_MAIN, r=0.030)
    textline(c, 0.14, 0.68, 0.66, h=0.056)
    textline(c, 0.14, 0.82, 0.44, h=0.056)


def feature_dict_02(c):
    """A search run over the column names, and the one it lands on."""
    c.rect(0.06, 0.08, 0.88, 0.20, w=W_MAIN, r=0.045)
    magn_over(c, 0.19, 0.18, 0.062, w=W_SEC, ang=52.0, handle=0.65)
    textline(c, 0.32, 0.18, 0.34, h=0.048)
    for j in range(3):
        yy = 0.44 + j * 0.20
        if j == 1:
            c.rect(0.06, yy - 0.075, 0.88, 0.15, w=W_SEC, r=0.030)
            textline(c, 0.14, yy, 0.62, h=0.054)
        else:
            textline(c, 0.14, yy, 0.44, h=0.044)


def feature_dict_03(c):
    """The dictionary itself: the reference, opened."""
    c.polyline([(0.06, 0.28), (0.48, 0.36), (0.48, 0.86), (0.06, 0.76)],
               w=W_MAIN, close=True)
    c.polyline([(0.94, 0.28), (0.52, 0.36), (0.52, 0.86), (0.94, 0.76)],
               w=W_MAIN, close=True)
    c.line(0.50, 0.355, 0.50, 0.865, W_MAIN)
    for j in range(2):
        textline(c, 0.13, 0.50 + j * 0.15, 0.26, h=0.042)
        textline(c, 0.60, 0.50 + j * 0.15, 0.26, h=0.042)


def feature_dict_04(c):
    """What is this column? -- asked of the header of a results table."""
    x, y, w, h = 0.06, 0.34, 0.62, 0.52
    grid(c, x, y, w, h, 3, 3)
    for i in range(3):
        cell_fill(c, x, y, w, h, 3, 3, 0, i, inset=0.26)
    cw = w / 3.0
    c.rect(x + cw, y, cw, h, w=W_MAIN)
    qmark(c, 0.84, 0.19, 0.19, W_MAIN)
    c.arrow(0.82, 0.52, 0.56, 0.46, W_MAIN, head=0.070)


def feature_dict_05(c):
    """Looked up by the idea, not by the name."""
    for j, ww in enumerate((0.26, 0.30, 0.22)):
        yy = 0.20 + j * 0.30
        c.rect(0.05, yy - 0.085, ww, 0.17, w=W_MAIN, r=0.085)
        textline(c, 0.10, yy, ww - 0.10, h=0.042)
        c.arrow(0.05 + ww + 0.02, yy, 0.58, 0.38 + j * 0.12, W_SEC, head=0.052)
    c.rect(0.62, 0.28, 0.33, 0.44, w=W_MAIN, r=0.028)
    for j in range(3):
        textline(c, 0.68, 0.38 + j * 0.12, 0.20, h=0.040)


def feature_dict_06(c):
    """One entry's card: the name, then the fields the name is explained by."""
    c.rect(0.06, 0.10, 0.88, 0.80, w=W_MAIN, r=0.035)
    c.rect(0.06, 0.10, 0.88, 0.20, filled=True, r=0.035)
    c.rect(0.06, 0.24, 0.88, 0.06, filled=True)
    for j in range(3):
        yy = 0.44 + j * 0.16
        c.bar(0.13, yy - 0.026, 0.16, 0.052, filled=True)
        c.rect(0.36, yy - 0.028, 0.50, 0.056, w=W_SEC, r=0.028)


def feature_dict_07(c):
    """A term list with its sections tabbed down the edge."""
    c.rect(0.06, 0.10, 0.72, 0.80, w=W_MAIN)
    for j in range(4):
        textline(c, 0.13, 0.22 + j * 0.19, 0.44, h=0.048)
    for j in range(4):
        yy = 0.16 + j * 0.19
        c.rect(0.78, yy, 0.16, 0.13, filled=(j == 1), r=0.030,
               w=W_SEC)


def feature_dict_08(c):
    """The same name checked against the objects it actually exists for."""
    c.rect(0.10, 0.10, 0.80, 0.18, w=W_MAIN, r=0.035)
    textline(c, 0.18, 0.19, 0.56, h=0.052)
    for i, ok in enumerate((True, True, False)):
        cx = 0.22 + i * 0.28
        c.ell(cx, 0.56, 0.105, 0.092, w=W_SEC)
        c.disc(cx, 0.56, 0.038)
        if ok:
            tick(c, cx, 0.79, 0.070, W_MAIN)
        else:
            cross(c, cx, 0.79, 0.070, W_MAIN)


def feature_dict_09(c):
    """Which channel the feature is about, picked out of the stack."""
    c.rect(0.08, 0.10, 0.84, 0.16, w=W_MAIN, r=0.032)
    textline(c, 0.16, 0.18, 0.60, h=0.048)
    for j in range(3):
        yy = 0.40 + j * 0.19
        if j == 1:
            c.rect(0.14, yy - 0.075, 0.56, 0.15, filled=True, r=0.020)
            c.arrow(0.92, yy, 0.74, yy, W_MAIN, head=0.075)
        else:
            c.rect(0.14, yy - 0.075, 0.56, 0.15, w=W_SEC, r=0.020)


def feature_dict_10(c):
    """A long name pulled apart into the object, the channel and the statistic."""
    xs = ((0.04, 0.26), (0.36, 0.22), (0.64, 0.32))
    for x0, ww in xs:
        c.bar(x0, 0.12, ww, 0.13, filled=True, r=0.055)
    for x0, ww in xs:
        brace_down(c, x0 + 0.01, x0 + ww - 0.01, 0.42, depth=0.090, w=W_MAIN)
    c.ell(0.17, 0.68, 0.110, 0.094, w=W_MAIN)
    c.disc(0.17, 0.68, 0.044)
    for j in range(2):
        c.rect(0.37, 0.56 + j * 0.13, 0.22, 0.095, w=W_MAIN)
    for i, f in enumerate((0.12, 0.24, 0.17)):
        c.rect(0.68 + i * 0.10, 0.78 - f, 0.075, f, filled=True)


# =====================================================================
# outliers -- ONE MARK THAT DOES NOT BELONG WITH THE OTHERS
# =====================================================================

CROWD = [(0.30, 0.62), (0.40, 0.70), (0.26, 0.74), (0.44, 0.58),
         (0.36, 0.80), (0.20, 0.66)]


def outliers_01(c):
    """A tight crowd, and the one that is nowhere near it."""
    dots(c, CROWD, r=0.052)
    ringed(c, 0.80, 0.22, r=0.052, gap=0.062, w=W_MAIN)


def outliers_02(c):
    """A plate of wells all alike, and the one that is not."""
    x, y, w, h = 0.06, 0.30, 0.88, 0.46
    c.rect(x, y, w, h, w=W_MAIN, r=0.035)
    for j in range(3):
        for i in range(4):
            cx = x + w * (i + 0.5) / 4.0
            cy = y + h * (j + 0.5) / 3.0
            if (j, i) == (1, 2):
                ringed(c, cx, cy, r=0.048, gap=0.048, w=W_MAIN)
            else:
                c.circ(cx, cy, 0.048, W_SEC)
    flag(c, x + w * 2.5 / 4.0, y - 0.02, h=0.20, fw=0.115, w=W_SEC)


def outliers_03(c):
    """Past the whisker: a robust fence built out of the crowd itself."""
    c.line(0.10, 0.50, 0.24, 0.50, W_SEC)
    c.line(0.10, 0.40, 0.10, 0.60, W_SEC)
    c.rect(0.24, 0.32, 0.34, 0.36, w=W_MAIN)
    c.line(0.40, 0.32, 0.40, 0.68, W_MAIN)
    c.line(0.58, 0.50, 0.70, 0.50, W_SEC)
    c.line(0.70, 0.40, 0.70, 0.60, W_SEC)
    ringed(c, 0.88, 0.50, r=0.048, gap=0.055, w=W_MAIN)


def outliers_04(c):
    """A tolerance ellipse drawn round the crowd, and the point left outside."""
    c.ell(0.42, 0.60, 0.32, 0.22, rot=-22.0, w=W_MAIN, dash=DASH)
    dots(c, [(0.30, 0.66), (0.42, 0.60), (0.52, 0.54), (0.36, 0.54),
             (0.50, 0.66)], r=0.042)
    ringed(c, 0.82, 0.20, r=0.046, gap=0.056, w=W_MAIN)


def outliers_05(c):
    """A column written, not a row dropped."""
    y, h, rows = 0.16, 0.68, 4
    grid(c, 0.05, y, 0.58, h, rows, 2)
    grid(c, 0.71, y, 0.24, h, rows, 1)
    rh = h / rows
    for j in range(rows):
        cell_bar(c, 0.05, y, 0.58, h, rows, 2, j, 0, frac=0.70, th=0.040)
        cell_bar(c, 0.05, y, 0.58, h, rows, 2, j, 1, frac=0.56, th=0.040)
    flag(c, 0.78, y + rh * 2.86, h=0.135, fw=0.105, w=W_MAIN)


def outliers_06(c):
    """The bulk of the distribution, and the one stranded far out in the tail."""
    base = 0.76
    pts = []
    for i in range(19):
        t = -1.0 + 2.0 * i / 18.0
        pts.append((0.26 + t * 0.20, base - 0.44 * math.exp(-4.0 * t * t)))
    c.smooth(pts, w=W_MAIN)
    c.line(0.04, base, 0.96, base, W_SEC)
    c.line(0.70, 0.20, 0.70, base, W_SEC, dash=DASH)
    c.line(0.86, base, 0.86, 0.50, W_MAIN)
    c.disc(0.86, 0.46, 0.050)


def outliers_07(c):
    """Two questions, two answers: the odd object and the odd well."""
    c.line(0.50, 0.10, 0.50, 0.90, W_SEC)
    dots(c, [(0.16, 0.62), (0.26, 0.70), (0.30, 0.56), (0.20, 0.50)], r=0.040)
    ringed(c, 0.38, 0.26, r=0.040, gap=0.048, w=W_SEC)
    for j in range(2):
        for i in range(3):
            cx, cy = 0.60 + i * 0.14, 0.42 + j * 0.20
            if (j, i) == (0, 2):
                ringed(c, cx, cy, r=0.038, gap=0.044, w=W_SEC)
            else:
                c.circ(cx, cy, 0.038, W_SEC)


def outliers_08(c):
    """How far out it is: the distance from the crowd measured."""
    dots(c, [(0.14, 0.66), (0.36, 0.66), (0.13, 0.88), (0.36, 0.88)], r=0.044)
    c.circ(0.25, 0.77, 0.034, W_MAIN)
    c.line(0.15, 0.77, 0.35, 0.77, W_SEC)
    c.line(0.25, 0.67, 0.25, 0.87, W_SEC)
    caliper(c, 0.31, 0.71, 0.76, 0.28, t=0.052, w=W_MAIN)
    c.disc(0.84, 0.20, 0.056)


def outliers_09(c):
    """Belonging to neither group -- odd without being extreme."""
    dots(c, [(0.20, 0.30), (0.30, 0.22), (0.30, 0.38), (0.19, 0.44)], r=0.044)
    dots(c, [(0.78, 0.68), (0.68, 0.76), (0.80, 0.82), (0.88, 0.70)], r=0.044)
    ringed(c, 0.50, 0.50, r=0.046, gap=0.056, w=W_MAIN)


def outliers_10(c):
    """A whole well shifted together, though not one object in it looks wrong."""
    c.line(0.16, 0.68, 0.40, 0.68, W_SEC)
    c.line(0.28, 0.56, 0.28, 0.80, W_SEC)
    c.circ(0.28, 0.68, 0.030, W_MAIN)
    dots(c, [(0.12, 0.54), (0.42, 0.54), (0.10, 0.84), (0.44, 0.84)], r=0.040)
    c.rect(0.60, 0.10, 0.36, 0.32, w=W_MAIN, r=0.045)
    dots(c, [(0.70, 0.20), (0.86, 0.20), (0.78, 0.33)], r=0.040)
    c.arrow(0.44, 0.62, 0.60, 0.44, W_MAIN, head=0.070)


# =====================================================================
# dose_response -- A SIGMOID READ AT ITS INFLECTION
# =====================================================================

def dose_response_01(c):
    """The full S, and the EC50 read off it at half the response."""
    c.axes(0.14, 0.12, 0.94, 0.82, W_SEC)
    pts = sig_pts(0.16, 0.92, 0.78, 0.18, k=11.0, mid=0.5)
    c.smooth(pts, w=W_MAIN)
    mx, my = 0.16 + 0.76 * 0.5, (0.78 + 0.18) / 2.0
    c.line(0.14, my, mx, my, W_SEC, dash=DASH)
    c.line(mx, my, mx, 0.82, W_SEC, dash=DASH)
    c.disc(mx, my, 0.045)
    c.polyline([(mx, 0.82), (mx - 0.055, 0.92), (mx + 0.055, 0.92)],
               close=True, filled=True)


def dose_response_02(c):
    """The concentration axis is logarithmic, and the curve is read on it."""
    c.smooth(sig_pts(0.10, 0.94, 0.62, 0.14, k=11.0, mid=0.5), w=W_MAIN)
    c.line(0.06, 0.72, 0.96, 0.72, W_MAIN)
    for k in range(3):
        x0 = 0.08 + k * 0.30
        log_ticks(c, x0, x0 + 0.30, 0.72, n=5, t=0.055, w=W_SEC)
        c.line(x0, 0.72, x0, 0.72 + 0.105, W_MAIN)
    c.line(0.98 - 0.02, 0.72, 0.98 - 0.02, 0.825, W_MAIN)


def dose_response_03(c):
    """The inflection: the one place the response is actually changing."""
    c.axes(0.12, 0.12, 0.94, 0.84, W_SEC)
    pts = sig_pts(0.16, 0.92, 0.78, 0.18, k=11.0, mid=0.5)
    c.smooth(pts, w=W_MAIN)
    mx, my = 0.54, (0.78 + 0.18) / 2.0
    c.line(mx - 0.24, my + 0.26, mx + 0.24, my - 0.26, W_SEC)
    c.disc(mx, my, 0.055)


def dose_response_04(c):
    """The EC50 with the interval it is known to, bracketed on the axis."""
    c.smooth(sig_pts(0.12, 0.92, 0.62, 0.14, k=11.0, mid=0.5), w=W_MAIN)
    mx, my = 0.52, 0.38
    c.line(mx, my, mx, 0.72, W_SEC, dash=DASH)
    c.disc(mx, my, 0.042)
    c.line(0.06, 0.72, 0.96, 0.72, W_SEC)
    caliper(c, 0.32, 0.86, 0.72, 0.86, t=0.062, w=W_MAIN)
    c.disc(mx, 0.86, 0.052)


def dose_response_05(c):
    """Two compounds: the more potent one's midpoint sits further left."""
    c.line(0.06, 0.78, 0.96, 0.78, W_SEC)
    for mid, x0 in ((0.34, 0.10), (0.66, 0.10)):
        c.smooth(sig_pts(x0, 0.94, 0.70, 0.16, k=11.0, mid=mid), w=W_MAIN)
        mx = x0 + (0.94 - x0) * mid
        c.disc(mx, (0.70 + 0.16) / 2.0, 0.042)
        c.line(mx, (0.70 + 0.16) / 2.0, mx, 0.78, W_SEC, dash=DASH)
        c.polyline([(mx, 0.78), (mx - 0.048, 0.88), (mx + 0.048, 0.88)],
                   close=True, filled=True)


def dose_response_06(c):
    """Still rising at the last dose: the answer is a bound, not a number."""
    c.axes(0.12, 0.08, 0.94, 0.78, W_SEC)
    pts = sig_pts(0.14, 1.76, 0.70, 0.10, k=11.0, mid=0.50)
    c.smooth([(x, y) for x, y in pts if x <= 0.80], w=W_MAIN)
    c.line(0.80, 0.08, 0.80, 0.78, W_MAIN, dash=DASH)
    c.line(0.14, 0.40, 0.80, 0.40, W_SEC, dash=DASH)
    c.arrow(0.62, 0.92, 0.94, 0.92, W_MAIN, head=0.080)
    c.line(0.50, 0.855, 0.50, 0.985, W_MAIN)


def dose_response_07(c):
    """Up and back down again: not a dose-response, so not fitted."""
    c.axes(0.12, 0.12, 0.92, 0.80, W_SEC)
    pts = [(0.20, 0.66), (0.34, 0.44), (0.48, 0.26), (0.62, 0.46),
           (0.78, 0.68)]
    for p in pts:
        c.disc(p[0], p[1], 0.045)
    c.smooth(pts, w=W_SEC, dash=DASH)
    c.line(0.24, 0.22, 0.76, 0.74, W_MAIN)


def dose_response_08(c):
    """Both plateaus stated, and the span the response actually moves over."""
    c.smooth(sig_pts(0.10, 0.78, 0.72, 0.18, k=11.0, mid=0.5), w=W_MAIN)
    c.line(0.06, 0.72, 0.82, 0.72, W_SEC, dash=DASH)
    c.line(0.06, 0.18, 0.82, 0.18, W_SEC, dash=DASH)
    caliper(c, 0.90, 0.18, 0.90, 0.72, t=0.060, w=W_MAIN)
    c.disc(0.44, 0.45, 0.045)


def dose_response_09(c):
    """The fitted EC50 chosen as the dose the next experiment will use."""
    c.smooth(sig_pts(0.06, 0.60, 0.58, 0.12, k=11.0, mid=0.5), w=W_MAIN)
    mx = 0.33
    c.line(0.04, 0.66, 0.64, 0.66, W_SEC)
    c.line(mx, 0.35, mx, 0.66, W_SEC, dash=DASH)
    c.disc(mx, 0.35, 0.042)
    c.arrow(mx, 0.72, 0.66, 0.86, W_SEC, head=0.062)
    c.circ(0.80, 0.80, 0.145, W_MAIN)
    c.disc(0.80, 0.80, 0.062)


def dose_response_10(c):
    """One midpoint, two steepnesses: how sharply the response switches."""
    c.line(0.06, 0.78, 0.96, 0.78, W_SEC)
    c.smooth(sig_pts(0.10, 0.92, 0.70, 0.14, k=22.0, mid=0.5), w=W_MAIN)
    c.smooth(sig_pts(0.10, 0.92, 0.70, 0.14, k=5.0, mid=0.5), w=W_SEC,
             dash=DASH)
    mx = 0.51
    c.disc(mx, 0.42, 0.050)
    c.line(mx, 0.42, mx, 0.78, W_SEC, dash=DASH)


# ---------------------------------------------------------------------------
# the folders
# ---------------------------------------------------------------------------

GROUPS = {
    "pca": ("pca -- axes that were derived rather than measured", [
        ("Many feature columns collapsed onto two derived axes.", pca_01),
        ("A scree plot: component variance falling away, the elbow ringed.",
         pca_02),
        ("A loadings biplot: the features drawn as vectors off the origin.",
         pca_03),
        ("The cloud's own long and short directions, drawn through it.",
         pca_04),
        ("The measured frame turned into the derived one.", pca_05),
        ("Points dropped out of many dimensions onto one flat plane.",
         pca_06),
        ("How much of the picture each component is: shares of one whole.",
         pca_07),
        ("Two groups that only come apart once the axes are derived.",
         pca_08),
        ("One component alone: the cloud folded down onto a single direction.",
         pca_09),
        ("Features on wildly different scales brought to one before "
         "decomposing.", pca_10),
    ]),
    "tabulate": ("tabulate -- many rows becoming one cell", [
        ("The pivot: keys down the side, keys across the top, summaries "
         "between.", tabulate_01),
        ("A stack of raw rows collapsed into one summary cell.", tabulate_02),
        ("Margins: the totals along the far edge, ruled off from the body.",
         tabulate_03),
        ("One group, three statistics of it: the count, the middle and the "
         "spread.", tabulate_04),
        ("The long raw table beside the short summary it becomes.",
         tabulate_05),
        ("A row key and a column key crossing on the one cell they share.",
         tabulate_06),
        ("Rows braced into groups, each brace giving out one row.",
         tabulate_07),
        ("A group opened into the subgroups it is made of.", tabulate_08),
        ("A hole in the grid: the combination no row was found for.",
         tabulate_09),
        ("The summary above, and the chart of it drawn column for column "
         "below.", tabulate_10),
    ]),
    "feature_dict": ("feature_dict -- a name resolving to a meaning", [
        ("A column name, and the meaning it resolves to.", feature_dict_01),
        ("A search run over the column names, and the one it lands on.",
         feature_dict_02),
        ("The dictionary itself: the reference, opened.", feature_dict_03),
        ("What is this column? -- asked of the header of a results table.",
         feature_dict_04),
        ("Looked up by the idea, not by the name.", feature_dict_05),
        ("One entry's card: the name, then the fields the name is explained "
         "by.", feature_dict_06),
        ("A term list with its sections tabbed down the edge.",
         feature_dict_07),
        ("The same name checked against the objects it actually exists for.",
         feature_dict_08),
        ("Which channel the feature is about, picked out of the stack.",
         feature_dict_09),
        ("A long name pulled apart into the object, the channel and the "
         "statistic.", feature_dict_10),
    ]),
    "outliers": ("outliers -- one mark that does not belong with the others", [
        ("A tight crowd, and the one that is nowhere near it.", outliers_01),
        ("A plate of wells all alike, and the one that is not.", outliers_02),
        ("Past the whisker: a robust fence built out of the crowd itself.",
         outliers_03),
        ("A tolerance ellipse drawn round the crowd, and the point left "
         "outside.", outliers_04),
        ("A column written, not a row dropped.", outliers_05),
        ("The bulk of the distribution, and the one stranded far out in the "
         "tail.", outliers_06),
        ("Two questions, two answers: the odd object and the odd well.",
         outliers_07),
        ("How far out it is: the distance from the crowd measured.",
         outliers_08),
        ("Belonging to neither group -- odd without being extreme.",
         outliers_09),
        ("A whole well shifted together, though not one object in it looks "
         "wrong.", outliers_10),
    ]),
    "dose_response": ("dose_response -- a sigmoid read at its inflection", [
        ("The full S, and the EC50 read off it at half the response.",
         dose_response_01),
        ("The concentration axis is logarithmic, and the curve is read on it.",
         dose_response_02),
        ("The inflection: the one place the response is actually changing.",
         dose_response_03),
        ("The EC50 with the interval it is known to, bracketed on the axis.",
         dose_response_04),
        ("Two compounds: the more potent one's midpoint sits further left.",
         dose_response_05),
        ("Still rising at the last dose: the answer is a bound, not a number.",
         dose_response_06),
        ("Up and back down again: not a dose-response, so not fitted.",
         dose_response_07),
        ("Both plateaus stated, and the span the response actually moves "
         "over.", dose_response_08),
        ("The fitted EC50 chosen as the dose the next experiment will use.",
         dose_response_09),
        ("One midpoint, two steepnesses: how sharply the response switches.",
         dose_response_10),
    ]),
}


def main(outdir):
    return emit_groups(outdir, GROUPS,
                       "group_pca_tabulate_dict_outliers_dose.py")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else default_outdir(__file__)))
