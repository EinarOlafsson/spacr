#!/usr/bin/env python3
"""Candidate spaCR icons: the results / QC end of the app registry.

Nine apps, ten conceptually different designs each, white-on-transparent flat
vector art in the house style set by ``plaque.png`` / ``measure.png``.

These nine collide more easily than any other group in the app registry: four
of them naturally want to draw a microtitre plate and three of them naturally
want to draw "a chart with a threshold line on it".  Every candidate below is
held to one of these lines, and no candidate is allowed to wander across::

    plate_view          the plate AS A HEATMAP.  Wells filled / outlined /
                        empty by value, a gradient running across the plate,
                        a hot edge row.  A measurement is being READ OFF the
                        plate.  Nothing is being authored.
    experiment_design   the plate as a LAYOUT BEING AUTHORED.  Conditions,
                        controls and replicates assigned to wells, a legend of
                        marks, blocks of treatment, the layout exported.
                        Nothing has been measured yet -- no well carries a
                        value.
    power               a CURVE YOU READ A NUMBER OFF.  Power against n, an
                        effect size, and a required-n marker dropped onto the
                        x axis.  The sample size is the answer, so every
                        design has to end on an axis.
    qc_dashboard        SEVERAL VERDICTS AT ONCE RESOLVING INTO ONE.  Multiple
                        small check panels, ticks and crosses, gauges, and the
                        single overall pass/fail they add up to.
    report              a SHAREABLE DOCUMENT.  A page with figures and text on
                        it, exported, linked, sent.  The artefact, never the
                        analysis that produced it.
    agreement           TWO ANNOTATORS' LABELS COMPARED.  Two label columns,
                        matches and mismatches, the 2x2 square, a kappa dial.
                        The 2x2 grid belongs to this module and to no other --
                        which is why classifier_evaluation below never draws a
                        confusion matrix.
    classifier_evaluation
                        HELD-OUT PREDICTION QUALITY.  ROC, calibration against
                        the y=x line, CV folds, a sealed test set, leakage.
                        Curves and folds only; the confusion matrix is ceded
                        to ``agreement``.
    barcode_qc          a READ-ABUNDANCE DISTRIBUTION WITH A CUTOFF.  Reads
                        counted onto barcodes, a steeply decaying rank curve,
                        a threshold placed on it.  Its curve falls away; the
                        power curve rises to a plateau; the ROC bulges above a
                        diagonal.  The three are never the same picture.
    hit_list            a RANKED LIST.  Sorted rows, a volcano with its top
                        corners starred, a filtered and flagged shortlist.

48 px is the real constraint, so every design is a handful of large,
high-contrast elements.  Well grids are 4x3 or 5x3 and never a literal 8x12;
"text" inside a document is a filled capsule bar, not a hairline; strokes stay
at ``W_MAIN``/``W_SEC`` because ``W_FINE`` is half a pixel at 48 px.

Run standalone (deterministic -- no random draws at all)::

    QT_QPA_PLATFORM=offscreen python group_results_qc.py [OUTDIR]

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


# ---------------------------------------------------------------------------
# shared sub-drawings
# ---------------------------------------------------------------------------

def tick(c, cx, cy, s, w=W_MAIN):
    """A check mark centred on (cx, cy), half-width s."""
    c.polyline([(cx - s, cy + s * 0.05), (cx - s * 0.25, cy + s * 0.70),
                (cx + s, cy - s * 0.78)], w=w)


def cross(c, cx, cy, s, w=W_MAIN):
    """An x mark centred on (cx, cy), half-width s."""
    c.line(cx - s * 0.78, cy - s * 0.78, cx + s * 0.78, cy + s * 0.78, w)
    c.line(cx + s * 0.78, cy - s * 0.78, cx - s * 0.78, cy + s * 0.78, w)


def star(c, cx, cy, r, points=5):
    """Solid five-pointed star -- the 'flagged / top hit' mark."""
    pts = []
    for i in range(2 * points):
        a = -math.pi / 2 + math.pi * i / points
        rr = r if i % 2 == 0 else r * 0.44
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    c.polyline(pts, close=True, filled=True)


def well(c, cx, cy, r, level):
    """One heatmap well: 2 = solid (hot), 1 = outlined (mid), 0 = a dot."""
    if level >= 2:
        c.disc(cx, cy, r)
    elif level == 1:
        c.circ(cx, cy, r * 0.90, W_MAIN)
    else:
        c.disc(cx, cy, r * 0.30)


def plate(c, x, y, w, h, levels, r=None, w_=W_MAIN, corner=0.032, frame=True):
    """Plate outline with a grid of wells filled from a level matrix."""
    rows, cols = len(levels), len(levels[0])
    if frame:
        c.rect(x, y, w, h, w=w_, r=corner)
    cw, ch = w / cols, h / rows
    rr = r if r is not None else min(cw, ch) * 0.34
    for j, row in enumerate(levels):
        for i, lv in enumerate(row):
            well(c, x + cw * (i + 0.5), y + ch * (j + 0.5), rr, lv)
    return rr


def mark(c, cx, cy, r, kind):
    """One condition mark for a *layout*: a shape, never a measured value."""
    if kind == "fill":
        c.disc(cx, cy, r)
    elif kind == "ring":
        c.circ(cx, cy, r * 0.92, W_MAIN)
    elif kind == "sq":
        c.rect(cx - r * 0.82, cy - r * 0.82, r * 1.64, r * 1.64, w=W_MAIN)
    elif kind == "sqf":
        c.rect(cx - r * 0.80, cy - r * 0.80, r * 1.60, r * 1.60, filled=True)
    elif kind == "tri":
        c.polyline([(cx, cy - r), (cx + r * 0.94, cy + r * 0.74),
                    (cx - r * 0.94, cy + r * 0.74)], close=True, filled=True)
    elif kind == "dash":
        c.circ(cx, cy, r * 0.92, W_SEC, dash=[1.1, 1.5])
    elif kind == "dot":
        c.disc(cx, cy, r * 0.32)


def page(c, x, y, w, h, fold=0.17, w_=W_MAIN):
    """A document page with a folded top-right corner."""
    f = fold * min(w, h) / max(w, h) * max(w, h)
    c.polyline([(x, y), (x + w - f, y), (x + w, y + f), (x + w, y + h),
                (x, y + h)], w=w_, close=True)
    c.polyline([(x + w - f, y), (x + w - f, y + f), (x + w, y + f)], w=W_SEC)


def textline(c, x, y, w, h=0.030):
    """A capsule bar standing in for a line of text (survives 48 px)."""
    c.bar(x, y - h / 2.0, w, h, filled=True)


def gauge(c, cx, cy, r, frac, w=W_MAIN, needle=0.78):
    """Semicircular dial with a needle at fraction ``frac`` of the sweep."""
    c.arc(cx, cy, r, 0, 180, w)
    c.line(cx - r, cy, cx + r, cy, W_SEC)
    a = math.radians(180.0 - 180.0 * frac)
    c.line(cx, cy, cx + r * needle * math.cos(a), cy - r * needle * math.sin(a), w)
    c.disc(cx, cy, r * 0.13)


def bell(c, cx, base, halfw, height, w=W_MAIN, n=15):
    """A normal-ish hump sitting on the baseline ``base``."""
    pts = []
    for i in range(n):
        t = -1.0 + 2.0 * i / (n - 1)
        pts.append((cx + t * halfw, base - height * math.exp(-4.2 * t * t)))
    c.smooth(pts, w=w)


def scurve_pts(x0, x1, ybase, ytop, k=9.0, mid=0.40, n=25):
    """Saturating power curve, normalised to run from ybase up to ytop."""
    vals = [1.0 / (1.0 + math.exp(-k * (i / (n - 1.0) - mid))) for i in range(n)]
    lo, hi = vals[0], vals[-1]
    return [(x0 + (x1 - x0) * i / (n - 1.0),
             ybase - (ybase - ytop) * (v - lo) / (hi - lo))
            for i, v in enumerate(vals)]


def at_level(pts, ybase, ytop, frac):
    """x of the first point on ``pts`` reaching ``frac`` of the rise."""
    target = ybase - (ybase - ytop) * frac
    for x, y in pts:
        if y <= target:
            return x
    return pts[-1][0]


def decay_pts(x0, x1, ytop, ybase, k=5.0, n=25):
    """Rank-abundance style curve: a steep fall into a long flat tail."""
    return [(x0 + (x1 - x0) * i / (n - 1.0),
             ytop + (ybase - ytop) * (1.0 - math.exp(-k * i / (n - 1.0))))
            for i in range(n)]


def roc_pts(x0, x1, ybase, ytop, p=0.34, n=22):
    """Concave ROC-like curve bulging above the chance diagonal."""
    return [(x0 + (x1 - x0) * i / (n - 1.0),
             ybase - (ybase - ytop) * (i / (n - 1.0)) ** p)
            for i in range(n)]


def barcode(c, x, y, w, h, widths=(0.9, 0.35, 0.6, 0.25, 0.8)):
    """A chunky barcode: five stripes, deliberately not hairline hatching."""
    total = sum(widths)
    gap = w * 0.10 / max(1, len(widths) - 1)
    span = w - gap * (len(widths) - 1)
    cx = x
    for wd in widths:
        bw = span * wd / total
        c.rect(cx, y, bw, h, filled=True)
        cx += bw + gap


def brace(c, x0, x1, y, depth=0.05, w=W_SEC):
    """A downward-opening brace spanning x0..x1 (a 'this many' bracket)."""
    c.polyline([(x0, y - depth), (x0, y), (x1, y), (x1, y - depth)], w=w)


# =====================================================================
# plate_view -- the plate AS A HEATMAP: values read off the wells
# =====================================================================

def plate_view_01(c):
    """Wells graded solid to empty across the plate: a heatmap gradient."""
    plate(c, 0.08, 0.24, 0.84, 0.52,
          [[2, 2, 1, 0], [2, 1, 1, 0], [2, 1, 0, 0]], r=0.062)
    c.arrow(0.16, 0.88, 0.84, 0.88, W_SEC, head=0.055)


def plate_view_02(c):
    """A hot outer ring: the edge wells solid, the interior barely marked."""
    plate(c, 0.06, 0.20, 0.88, 0.60,
          [[2, 2, 2, 2, 2], [2, 0, 0, 0, 2], [2, 2, 2, 2, 2]], r=0.058)


def plate_view_03(c):
    """A magnifier over the plate, the well under it read as a solid disc."""
    plate(c, 0.04, 0.42, 0.58, 0.46, [[1, 2, 0], [0, 1, 2]], r=0.062,
          corner=0.030)
    c.magnifier(0.68, 0.26, 0.230, ang_deg=45, w=W_MAIN, handle=0.55)
    c.disc(0.68, 0.26, 0.115)


def plate_view_04(c):
    """The plate beside its value key: solid, outlined and empty wells."""
    plate(c, 0.06, 0.26, 0.58, 0.48,
          [[2, 1, 0], [2, 2, 1], [1, 0, 0]], r=0.058, corner=0.026)
    c.line(0.78, 0.22, 0.78, 0.78, W_SEC)
    for cy, lv in ((0.32, 2), (0.50, 1), (0.68, 0)):
        well(c, 0.78, cy, 0.060, lv)
    c.arrow(0.90, 0.28, 0.90, 0.72, W_SEC, head=0.055)


def plate_view_05(c):
    """A column profile drawn under the plate, rising at both edges."""
    plate(c, 0.08, 0.14, 0.84, 0.40,
          [[2, 1, 0, 1, 2], [2, 1, 0, 1, 2]], r=0.052, corner=0.028)
    c.smooth([(0.16, 0.66), (0.33, 0.82), (0.50, 0.88), (0.67, 0.82),
              (0.84, 0.66)], w=W_MAIN)
    c.line(0.08, 0.92, 0.92, 0.92, W_SEC)


def plate_view_06(c):
    """A measured column of bars poured into the plate as well values."""
    for i, h in enumerate((0.30, 0.18, 0.40)):
        c.bar(0.08, 0.24 + i * 0.18, h, 0.090, filled=True)
    c.arrow(0.50, 0.50, 0.62, 0.50, W_SEC, head=0.060)
    plate(c, 0.66, 0.24, 0.28, 0.52, [[2, 1], [1, 0], [2, 2]], r=0.052,
          corner=0.024)


def plate_view_07(c):
    """A diagonal gradient: one corner of the plate hot, the far one cold."""
    plate(c, 0.08, 0.22, 0.84, 0.56,
          [[2, 2, 1, 0], [2, 1, 0, 0], [1, 0, 0, 0]], r=0.062)
    c.arrow(0.20, 0.34, 0.80, 0.68, W_SEC, head=0.060)


def plate_view_08(c):
    """The plate as heat tiles: filled squares against empty ones."""
    x, y, w, h = 0.10, 0.24, 0.80, 0.52
    c.rect(x, y, w, h, w=W_MAIN, r=0.030)
    lv = [[2, 2, 1, 0], [2, 1, 0, 0], [1, 1, 0, 2]]
    cw, ch = w / 4.0, h / 3.0
    for j in range(3):
        for i in range(4):
            tx, ty = x + cw * i + cw * 0.16, y + ch * j + ch * 0.16
            tw, th = cw * 0.68, ch * 0.68
            if lv[j][i] == 2:
                c.rect(tx, ty, tw, th, filled=True)
            elif lv[j][i] == 1:
                c.rect(tx, ty, tw, th, w=W_MAIN)
            else:
                c.disc(tx + tw / 2, ty + th / 2, min(tw, th) * 0.16)


def plate_view_09(c):
    """A callout hanging off one well, showing the value it holds."""
    plate(c, 0.06, 0.46, 0.60, 0.44,
          [[2, 1, 0, 1], [0, 1, 1, 0]], r=0.055, corner=0.026)
    c.polyline([(0.34, 0.36), (0.34, 0.10), (0.94, 0.10), (0.94, 0.36),
                (0.30, 0.36), (0.20, 0.50)], w=W_MAIN, close=True)
    textline(c, 0.44, 0.23, 0.42, 0.056)


def plate_view_10(c):
    """One column of the plate reading hot: a plating artefact spotted."""
    plate(c, 0.08, 0.34, 0.84, 0.52,
          [[0, 2, 0, 0, 0], [0, 2, 0, 0, 0], [0, 2, 0, 0, 0]], r=0.060)
    c.arrow(0.33, 0.06, 0.33, 0.28, W_MAIN, head=0.090)


# =====================================================================
# experiment_design -- the plate as a LAYOUT BEING AUTHORED
# =====================================================================

def experiment_design_01(c):
    """Condition marks assigned across the plate, with a key beside it."""
    x, y, w, h = 0.04, 0.26, 0.60, 0.48
    c.rect(x, y, w, h, w=W_MAIN, r=0.028)
    kinds = [["fill", "sqf", "ring"], ["fill", "sqf", "ring"]]
    for j in range(2):
        for i in range(3):
            mark(c, x + w * (i + 0.5) / 3, y + h * (j + 0.5) / 2, 0.072,
                 kinds[j][i])
    for k, kind in enumerate(("fill", "sqf", "ring")):
        mark(c, 0.76, 0.28 + k * 0.22, 0.062, kind)
        c.bar(0.86, 0.28 + k * 0.22 - 0.022, 0.12, 0.044, filled=True)


def experiment_design_02(c):
    """The plate cut into three treatment blocks, each braced as one."""
    c.rect(0.06, 0.32, 0.88, 0.48, w=W_MAIN, r=0.028)
    kinds = ("fill", "sqf", "tri")
    for b in range(3):
        x0 = 0.06 + 0.88 * b / 3.0
        if b:
            c.line(x0, 0.32, x0, 0.80, W_MAIN)
        brace(c, x0 + 0.030, x0 + 0.263, 0.26, depth=0.075, w=W_MAIN)
        for j in range(2):
            mark(c, x0 + 0.147, 0.32 + 0.48 * (j + 0.5) / 2, 0.078, kinds[b])


def experiment_design_03(c):
    """Controls parked in the outside columns, samples in the middle."""
    x, y, w, h = 0.06, 0.28, 0.88, 0.44
    c.rect(x, y, w, h, w=W_MAIN, r=0.030)
    for j in range(2):
        for i in range(5):
            cx, cy = x + w * (i + 0.5) / 5, y + h * (j + 0.5) / 2
            mark(c, cx, cy, 0.070, "ring" if i in (0, 4) else "fill")
    c.line(x + w * 0.20, y - 0.070, x + w * 0.20, y + h + 0.070, W_MAIN)
    c.line(x + w * 0.80, y - 0.070, x + w * 0.80, y + h + 0.070, W_MAIN)


def experiment_design_04(c):
    """One condition chip copied into three replicate wells."""
    c.rect(0.06, 0.40, 0.24, 0.22, w=W_MAIN, r=0.036)
    mark(c, 0.18, 0.51, 0.060, "fill")
    for j, cy in enumerate((0.22, 0.51, 0.80)):
        c.arrow(0.34, 0.51, 0.58, cy, W_SEC, head=0.055)
        c.circ(0.74, cy, 0.090, W_MAIN)
        mark(c, 0.74, cy, 0.052, "fill")


def experiment_design_05(c):
    """A pen dropping the next condition into an empty well."""
    plate_x, plate_y, w, h = 0.06, 0.44, 0.70, 0.44
    c.rect(plate_x, plate_y, w, h, w=W_MAIN, r=0.028)
    filled = [["fill", "fill", "sqf", "dash"], ["fill", "sqf", "dash", "dash"]]
    for j in range(2):
        for i in range(4):
            mark(c, plate_x + w * (i + 0.5) / 4, plate_y + h * (j + 0.5) / 2,
                 0.055, filled[j][i])
    c.polyline([(0.96, 0.10), (0.74, 0.28), (0.70, 0.42), (0.82, 0.34),
                (0.96, 0.10)], w=W_MAIN, close=True)
    c.disc(0.718, 0.390, 0.024)


def experiment_design_06(c):
    """Condition chips waiting in a palette, one dragged onto the plate."""
    for k, kind in enumerate(("fill", "sqf", "tri")):
        c.rect(0.06, 0.16 + k * 0.24, 0.20, 0.18, w=W_MAIN, r=0.034)
        mark(c, 0.16, 0.25 + k * 0.24, 0.052, kind)
    c.rect(0.44, 0.22, 0.50, 0.56, w=W_MAIN, r=0.028)
    for j in range(3):
        for i in range(2):
            mark(c, 0.44 + 0.50 * (i + 0.5) / 2, 0.22 + 0.56 * (j + 0.5) / 3,
                 0.052, "dash" if (i + j) > 1 else "fill")
    c.arrow(0.28, 0.25, 0.55, 0.34, W_SEC, head=0.060)


def experiment_design_07(c):
    """Randomisation: two crossed arrows shuffling the marks about."""
    x, y, w, h = 0.08, 0.30, 0.84, 0.50
    c.rect(x, y, w, h, w=W_MAIN, r=0.028)
    kinds = [["sqf", "fill", "tri", "fill"], ["fill", "tri", "fill", "sqf"]]
    for j in range(2):
        for i in range(4):
            mark(c, x + w * (i + 0.5) / 4, y + h * (j + 0.5) / 2, 0.054,
                 kinds[j][i])
    c.arrow(0.24, 0.16, 0.74, 0.16, W_SEC, head=0.055)
    c.arrow(0.76, 0.90, 0.26, 0.90, W_SEC, head=0.055)


def experiment_design_08(c):
    """A half-authored plate: assigned wells solid, the rest still blank."""
    x, y, w, h = 0.06, 0.28, 0.88, 0.44
    c.rect(x, y, w, h, w=W_MAIN, r=0.030)
    for j in range(2):
        for i in range(4):
            cx, cy = x + w * (i + 0.5) / 4, y + h * (j + 0.5) / 2
            mark(c, cx, cy, 0.078, "fill" if i < 2 else "dash")
    c.line(x + w * 0.5, y - 0.080, x + w * 0.5, y + h + 0.080, W_MAIN)


def experiment_design_09(c):
    """A dose series laid along a row, the mark growing well by well."""
    c.rect(0.06, 0.34, 0.88, 0.32, w=W_MAIN, r=0.034)
    for i in range(5):
        cx = 0.06 + 0.88 * (i + 0.5) / 5
        mark(c, cx, 0.50, 0.022 + i * 0.014, "fill")
    c.arrow(0.14, 0.80, 0.86, 0.80, W_SEC, head=0.055)


def experiment_design_10(c):
    """The finished layout exported as a table for the pipeline."""
    c.rect(0.04, 0.30, 0.40, 0.40, w=W_MAIN, r=0.026)
    kinds = [["fill", "ring"], ["sqf", "ring"]]
    for j in range(2):
        for i in range(2):
            mark(c, 0.04 + 0.40 * (i + 0.5) / 2, 0.30 + 0.40 * (j + 0.5) / 2,
                 0.056, kinds[j][i])
    c.arrow(0.47, 0.50, 0.58, 0.50, W_SEC, head=0.055)
    c.rect(0.62, 0.26, 0.34, 0.48, w=W_MAIN, r=0.024)
    c.line(0.62, 0.38, 0.96, 0.38, W_MAIN)
    c.line(0.79, 0.26, 0.79, 0.74, W_SEC)
    for y in (0.48, 0.60, 0.70):
        textline(c, 0.66, y, 0.09, 0.030)
        textline(c, 0.83, y, 0.09, 0.030)


# =====================================================================
# power -- a CURVE you read the required sample size off
# =====================================================================

def power_01(c):
    """A power curve with the required n dropped onto the x axis."""
    c.axes(0.14, 0.10, 0.94, 0.84, w=W_MAIN)
    pts = scurve_pts(0.16, 0.92, 0.84, 0.16, k=9.0, mid=0.38)
    c.smooth(pts, w=W_MAIN)
    xn = at_level(pts, 0.84, 0.16, 0.80)
    c.line(0.14, 0.30, xn, 0.30, W_SEC, dash=[6, 7])
    c.line(xn, 0.30, xn, 0.84, W_SEC, dash=[6, 7])
    c.disc(xn, 0.84, 0.048)


def power_02(c):
    """Two effect sizes, two curves, two required n's on the same axis."""
    c.axes(0.12, 0.10, 0.94, 0.84, w=W_MAIN)
    a = scurve_pts(0.14, 0.92, 0.84, 0.16, k=13.0, mid=0.22)
    b = scurve_pts(0.14, 0.92, 0.84, 0.26, k=12.0, mid=0.70)
    c.smooth(a, w=W_MAIN)
    c.smooth(b, w=W_SEC)
    for pts, top in ((a, 0.16), (b, 0.26)):
        xn = at_level(pts, 0.84, top, 0.78)
        c.line(xn, 0.84 - (0.84 - top) * 0.78, xn, 0.84, W_SEC, dash=[5, 6])
        c.disc(xn, 0.84, 0.044)


def power_03(c):
    """The effect size itself: two humps with the gap measured between them."""
    c.line(0.06, 0.76, 0.94, 0.76, W_MAIN)
    bell(c, 0.32, 0.76, 0.24, 0.42)
    bell(c, 0.68, 0.76, 0.24, 0.42)
    c.line(0.32, 0.34, 0.32, 0.16, W_SEC, dash=[5, 6])
    c.line(0.68, 0.34, 0.68, 0.16, W_SEC, dash=[5, 6])
    c.arrow(0.32, 0.20, 0.68, 0.20, W_MAIN, head=0.055)
    c.arrow(0.68, 0.20, 0.32, 0.20, W_MAIN, head=0.055, tail=False)


def power_04(c):
    """Cells per well: a well packed with cells feeding a rising curve."""
    c.circ(0.26, 0.16, 0.145, W_MAIN)
    for a in range(6):
        ang = TAU * a / 6.0
        c.disc(0.26 + 0.084 * math.cos(ang), 0.16 + 0.084 * math.sin(ang), 0.030)
    c.disc(0.26, 0.16, 0.030)
    c.arrow(0.44, 0.16, 0.66, 0.16, W_SEC, head=0.060)
    c.axes(0.12, 0.36, 0.94, 0.88, w=W_MAIN)
    pts = scurve_pts(0.16, 0.92, 0.88, 0.42, k=9.0, mid=0.42)
    c.smooth(pts, w=W_MAIN)
    xn = at_level(pts, 0.88, 0.42, 0.80)
    c.line(xn, 0.88 - 0.46 * 0.80, xn, 0.88, W_SEC, dash=[5, 6])
    c.disc(xn, 0.88, 0.048)


def power_05(c):
    """A slider set to an effect size, the curve above answering with n."""
    c.axes(0.12, 0.08, 0.94, 0.62, w=W_MAIN)
    pts = scurve_pts(0.14, 0.92, 0.62, 0.14, k=9.0, mid=0.40)
    c.smooth(pts, w=W_MAIN)
    xn = at_level(pts, 0.62, 0.14, 0.78)
    c.line(xn, 0.62 - 0.48 * 0.78, xn, 0.62, W_SEC, dash=[5, 6])
    c.disc(xn, 0.62, 0.046)
    c.bar(0.14, 0.80, 0.78, 0.056, filled=False, w=W_MAIN)
    c.disc(0.52, 0.828, 0.070)


def power_06(c):
    """Error bars shrinking as n grows until one clears the effect line."""
    c.line(0.08, 0.30, 0.94, 0.30, W_SEC, dash=[7, 8])
    for i, (x, half) in enumerate(((0.22, 0.30), (0.50, 0.19), (0.78, 0.10))):
        c.line(x, 0.52 - half, x, 0.52 + half, W_MAIN)
        c.line(x - 0.055, 0.52 - half, x + 0.055, 0.52 - half, W_SEC)
        c.line(x - 0.055, 0.52 + half, x + 0.055, 0.52 + half, W_SEC)
        c.disc(x, 0.52, 0.050 if i == 2 else 0.036)
    c.arrow(0.14, 0.94, 0.88, 0.94, W_MAIN, head=0.060)


def power_07(c):
    """How many wells: a run of wells counted off under one brace."""
    for i in range(6):
        cx = 0.12 + i * 0.152
        c.circ(cx, 0.24, 0.062, W_MAIN)
        c.disc(cx, 0.24, 0.026)
    brace(c, 0.12, 0.88, 0.48, depth=0.090, w=W_MAIN)
    c.arrow(0.50, 0.50, 0.50, 0.74, W_MAIN, head=0.080)
    c.line(0.08, 0.86, 0.92, 0.86, W_MAIN)
    c.disc(0.50, 0.86, 0.062)


def power_08(c):
    """A trade-off curve: pick an effect size, read the n it costs."""
    c.axes(0.14, 0.08, 0.94, 0.86, w=W_MAIN)
    pts = [(0.18 + 0.74 * t / 24.0, 0.16 + 0.68 * (1.0 - (1.0 - t / 24.0) ** 2.4))
           for t in range(25)]
    c.smooth([(x, 1.02 - y) for x, y in pts], w=W_MAIN)
    px, py = pts[9][0], 1.02 - pts[9][1]
    c.disc(px, py, 0.050)
    c.line(px, py, px, 0.86, W_SEC, dash=[5, 6])
    c.line(px, py, 0.14, py, W_SEC, dash=[5, 6])


def power_09(c):
    """Bars of growing n against a detectable line: the first to clear wins."""
    c.line(0.06, 0.88, 0.94, 0.88, W_MAIN)
    c.line(0.06, 0.40, 0.94, 0.40, W_SEC, dash=[7, 8])
    for i, h in enumerate((0.16, 0.28, 0.42, 0.58)):
        x = 0.12 + i * 0.21
        if h > 0.48:
            c.rect(x, 0.88 - h, 0.14, h, filled=True)
        else:
            c.rect(x, 0.88 - h, 0.14, h, w=W_MAIN)
    c.disc(0.82, 0.88, 0.048)


def power_10(c):
    """Nested power contours with the chosen n and effect size pinned on."""
    c.axes(0.12, 0.08, 0.94, 0.88, w=W_MAIN)
    for k, s in enumerate((0.20, 0.34, 0.50)):
        c.smooth([(0.16 + 0.76 * t / 16.0,
                   0.86 - s * 0.98 * (1.0 - (1.0 - t / 16.0) ** 2.2))
                  for t in range(17)], w=W_MAIN if k == 1 else W_SEC)
    c.disc(0.58, 0.55, 0.052)
    c.line(0.58, 0.55, 0.58, 0.88, W_SEC, dash=[5, 6])
    c.line(0.58, 0.55, 0.12, 0.55, W_SEC, dash=[5, 6])


# =====================================================================
# qc_dashboard -- several verdicts at once resolving into one
# =====================================================================

def qc_dashboard_01(c):
    """Four check panels, three ticked and one crossed, under one verdict."""
    for j in range(2):
        for i in range(2):
            x, y = 0.08 + i * 0.44, 0.08 + j * 0.30
            c.rect(x, y, 0.36, 0.24, w=W_MAIN, r=0.032)
            if (i, j) == (1, 1):
                cross(c, x + 0.18, y + 0.12, 0.075)
            else:
                tick(c, x + 0.18, y + 0.12, 0.085)
    c.line(0.08, 0.74, 0.92, 0.74, W_SEC)
    tick(c, 0.50, 0.87, 0.130, W_MAIN)


def qc_dashboard_02(c):
    """Three small dials read at once, answered by one big dial."""
    for i, f in enumerate((0.82, 0.55, 0.30)):
        gauge(c, 0.18 + i * 0.32, 0.36, 0.135, f, w=W_MAIN)
    gauge(c, 0.50, 0.90, 0.230, 0.72, w=W_MAIN)


def qc_dashboard_03(c):
    """A checklist of QC rows, ticked or crossed, totalled at the bottom."""
    for j, ok in enumerate((True, False, True)):
        y = 0.16 + j * 0.20
        c.bar(0.06, y - 0.040, 0.54, 0.080, filled=True)
        if ok:
            tick(c, 0.82, y, 0.090)
        else:
            cross(c, 0.82, y, 0.085)
    c.line(0.06, 0.70, 0.94, 0.70, W_SEC)
    c.bar(0.06, 0.82, 0.40, 0.110, filled=True)
    tick(c, 0.78, 0.875, 0.120)


def qc_dashboard_04(c):
    """One frame holding four unlike mini-panels and a verdict badge."""
    c.rect(0.06, 0.08, 0.88, 0.60, w=W_MAIN, r=0.036)
    c.line(0.50, 0.08, 0.50, 0.68, W_SEC)
    c.line(0.06, 0.38, 0.94, 0.38, W_SEC)
    for i, h in enumerate((0.10, 0.17, 0.13)):
        c.rect(0.13 + i * 0.10, 0.32 - h, 0.062, h, filled=True)
    c.smooth([(0.56, 0.30), (0.66, 0.18), (0.76, 0.26), (0.88, 0.13)], w=W_MAIN)
    for j in range(2):
        for i in range(2):
            well(c, 0.18 + i * 0.16, 0.48 + j * 0.13, 0.048,
                 2 if (i + j) % 2 == 0 else 0)
    gauge(c, 0.72, 0.62, 0.125, 0.72, w=W_MAIN)
    c.circ(0.50, 0.85, 0.130, W_MAIN)
    tick(c, 0.50, 0.85, 0.078)


def qc_dashboard_05(c):
    """Three checks feeding a traffic light that shows the overall call."""
    for j, ok in enumerate((True, False, True)):
        y = 0.16 + j * 0.30
        if ok:
            tick(c, 0.16, y, 0.080)
        else:
            cross(c, 0.16, y, 0.075)
        c.arrow(0.28, y, 0.46, 0.46 + (j - 1) * 0.06, W_SEC, head=0.048)
    c.rect(0.56, 0.08, 0.30, 0.84, w=W_MAIN, r=0.070)
    c.disc(0.71, 0.24, 0.098)
    c.circ(0.71, 0.50, 0.098, W_MAIN)
    c.circ(0.71, 0.76, 0.098, W_MAIN)


def qc_dashboard_06(c):
    """Separate ticks converging into a single overall tick."""
    for j, y in enumerate((0.14, 0.44, 0.74)):
        tick(c, 0.16, y, 0.085)
        c.arrow(0.30, y, 0.50, 0.44 + (j - 1) * 0.02, W_SEC, head=0.050)
    c.circ(0.74, 0.44, 0.215, W_MAIN)
    tick(c, 0.74, 0.44, 0.130)


def qc_dashboard_07(c):
    """A pass badge assembled from four separate checks around it."""
    c.polyline([(0.50, 0.12), (0.80, 0.26), (0.80, 0.56), (0.50, 0.82),
                (0.20, 0.56), (0.20, 0.26)], w=W_MAIN, close=True)
    tick(c, 0.50, 0.44, 0.150)
    for a in (-0.62, 0.62, 2.52, 3.76):
        c.disc(0.50 + 0.44 * math.cos(a), 0.47 + 0.44 * math.sin(a), 0.048)


def qc_dashboard_08(c):
    """A scorecard: four segments filled bar one, and the resulting call."""
    for i in range(4):
        x = 0.08 + i * 0.22
        if i == 2:
            c.rect(x, 0.20, 0.18, 0.30, w=W_MAIN, r=0.030)
        else:
            c.rect(x, 0.20, 0.18, 0.30, filled=True, r=0.030)
    c.rect(0.08, 0.62, 0.84, 0.22, w=W_MAIN, r=0.060)
    c.rect(0.08, 0.62, 0.62, 0.22, filled=True, r=0.060)
    tick(c, 0.50, 0.73, 0.075)


def qc_dashboard_09(c):
    """One dial split into a pass side and a fail side, needle on pass."""
    c.arc(0.50, 0.72, 0.360, 0, 180, W_MAIN)
    c.line(0.14, 0.72, 0.86, 0.72, W_SEC)
    c.line(0.50, 0.36, 0.50, 0.24, W_MAIN)
    tick(c, 0.26, 0.63, 0.075)
    cross(c, 0.74, 0.63, 0.070)
    a = math.radians(118.0)
    c.line(0.50, 0.72, 0.50 + 0.25 * math.cos(a), 0.72 - 0.25 * math.sin(a),
           W_MAIN)
    c.disc(0.50, 0.72, 0.052)


def qc_dashboard_10(c):
    """A deck of check cards with the summed verdict on the front one."""
    for dx, dy in ((0.18, -0.16), (0.09, -0.08)):
        c.polyline([(0.08 + dx, 0.30 + dy), (0.70 + dx, 0.30 + dy),
                    (0.70 + dx, 0.90 + dy)], w=W_SEC)
    c.rect(0.08, 0.30, 0.62, 0.60, w=W_MAIN, r=0.044)
    tick(c, 0.39, 0.50, 0.150)
    for i, ok in enumerate((True, False, True)):
        x = 0.18 + i * 0.21
        if ok:
            tick(c, x, 0.76, 0.058, W_SEC)
        else:
            cross(c, x, 0.76, 0.054, W_SEC)


# =====================================================================
# report -- a shareable DOCUMENT, not the analysis
# =====================================================================

def report_01(c):
    """A page carrying a bar figure and three lines of text."""
    page(c, 0.18, 0.06, 0.64, 0.88)
    for i, h in enumerate((0.14, 0.24, 0.19)):
        c.rect(0.27 + i * 0.14, 0.44 - h, 0.095, h, filled=True)
    c.line(0.25, 0.44, 0.75, 0.44, W_MAIN)
    for y in (0.58, 0.70, 0.82):
        textline(c, 0.25, y, 0.50 if y != 0.82 else 0.32, 0.040)


def report_02(c):
    """A finished page with a share arrow springing off it."""
    page(c, 0.10, 0.20, 0.56, 0.72)
    c.rect(0.18, 0.34, 0.40, 0.22, w=W_MAIN, r=0.020)
    for y in (0.68, 0.80):
        textline(c, 0.18, y, 0.40, 0.040)
    c.arrow(0.62, 0.32, 0.94, 0.08, W_MAIN, head=0.110)


def report_03(c):
    """A page dropping into an export tray."""
    page(c, 0.26, 0.04, 0.48, 0.46, fold=0.20)
    for y in (0.16, 0.28, 0.38):
        textline(c, 0.33, y, 0.28 if y != 0.38 else 0.18, 0.038)
    c.arrow(0.50, 0.54, 0.50, 0.70, W_MAIN, head=0.090)
    c.polyline([(0.12, 0.66), (0.12, 0.92), (0.88, 0.92), (0.88, 0.66)],
               w=W_MAIN)


def report_04(c):
    """Two pages fanned out: the figures sheet over the write-up."""
    c.polyline([(0.34, 0.10), (0.88, 0.10), (0.88, 0.84), (0.68, 0.84)],
               w=W_SEC)
    page(c, 0.12, 0.18, 0.54, 0.74)
    c.rect(0.20, 0.28, 0.38, 0.24, w=W_MAIN, r=0.018)
    c.smooth([(0.23, 0.46), (0.32, 0.34), (0.42, 0.42), (0.55, 0.31)], w=W_MAIN)
    for y in (0.64, 0.75, 0.85):
        textline(c, 0.20, y, 0.38 if y != 0.85 else 0.24, 0.038)


def report_05(c):
    """The QC verdict stamped at the head of the page, findings beneath."""
    page(c, 0.16, 0.06, 0.68, 0.88)
    c.circ(0.50, 0.30, 0.150, W_MAIN)
    tick(c, 0.50, 0.30, 0.090)
    c.line(0.22, 0.52, 0.78, 0.52, W_MAIN)
    for y in (0.63, 0.74, 0.85):
        textline(c, 0.24, y, 0.52 if y != 0.85 else 0.32, 0.042)


def report_06(c):
    """The HTML version: a browser window with a figure inside it."""
    c.rect(0.06, 0.16, 0.88, 0.68, w=W_MAIN, r=0.038)
    c.line(0.06, 0.32, 0.94, 0.32, W_MAIN)
    for i in range(3):
        c.disc(0.14 + i * 0.075, 0.24, 0.026)
    for i, h in enumerate((0.14, 0.26, 0.20)):
        c.rect(0.16 + i * 0.13, 0.72 - h, 0.090, h, filled=True)
    c.smooth([(0.60, 0.68), (0.70, 0.50), (0.80, 0.60), (0.90, 0.44)], w=W_MAIN)


def report_07(c):
    """A page published behind a shareable link."""
    page(c, 0.06, 0.14, 0.50, 0.72)
    for y in (0.34, 0.48, 0.62, 0.74):
        textline(c, 0.14, y, 0.34 if y != 0.74 else 0.20, 0.040)
    c.ring(0.70, 0.38, 0.115, 0.050)
    c.ring(0.83, 0.58, 0.115, 0.050)


def report_08(c):
    """The report sent out: a page tucked into an envelope."""
    page(c, 0.28, 0.04, 0.44, 0.42, fold=0.20)
    for y in (0.16, 0.28, 0.37):
        textline(c, 0.34, y, 0.26 if y != 0.37 else 0.16, 0.036)
    c.rect(0.08, 0.44, 0.84, 0.48, w=W_MAIN, r=0.036)
    c.polyline([(0.08, 0.46), (0.50, 0.74), (0.92, 0.46)], w=W_MAIN)


def report_09(c):
    """One page holding all of it: a figure, a table and the settings."""
    page(c, 0.14, 0.04, 0.72, 0.92)
    c.rect(0.22, 0.16, 0.56, 0.24, w=W_MAIN, r=0.020)
    c.smooth([(0.26, 0.34), (0.38, 0.22), (0.52, 0.30), (0.74, 0.20)], w=W_MAIN)
    c.rect(0.22, 0.48, 0.56, 0.22, w=W_MAIN, r=0.016)
    c.line(0.50, 0.48, 0.50, 0.70, W_SEC)
    c.line(0.22, 0.59, 0.78, 0.59, W_SEC)
    for y in (0.80, 0.89):
        textline(c, 0.22, y, 0.56 if y != 0.89 else 0.34, 0.038)


def report_10(c):
    """A version tag hung on the page: the settings and versions on record."""
    page(c, 0.08, 0.06, 0.60, 0.78)
    for y in (0.24, 0.38, 0.52, 0.64):
        textline(c, 0.16, y, 0.42 if y != 0.64 else 0.26, 0.040)
    c.polyline([(0.42, 0.74), (0.78, 0.74), (0.96, 0.86), (0.78, 0.98),
                (0.42, 0.98)], w=W_MAIN, close=True)
    c.disc(0.79, 0.86, 0.048)
    textline(c, 0.50, 0.86, 0.18, 0.048)


# =====================================================================
# agreement -- two annotators' labels compared
# =====================================================================

def agreement_01(c):
    """Two label columns joined pair by pair, one pair crossed out."""
    for j, y in enumerate((0.16, 0.40, 0.64, 0.88)):
        left = ("fill", "fill", "ring", "fill")[j]
        right = ("fill", "fill", "fill", "fill")[j]
        mark(c, 0.14, y, 0.070, left)
        mark(c, 0.86, y, 0.070, right)
        if left == right:
            c.line(0.24, y, 0.76, y, W_MAIN)
        else:
            c.line(0.24, y, 0.42, y, W_SEC)
            c.line(0.58, y, 0.76, y, W_SEC)
            cross(c, 0.50, y, 0.075)


def agreement_02(c):
    """A 2x2 square of the two annotators' calls, agreements on the diagonal."""
    x, y, s = 0.14, 0.14, 0.36
    for j in range(2):
        for i in range(2):
            if i == j:
                c.rect(x + i * s, y + j * s, s, s, filled=True)
            else:
                c.rect(x + i * s, y + j * s, s, s, w=W_MAIN)
    c.rect(x, y, 2 * s, 2 * s, w=W_MAIN)
    c.arrow(0.14, 0.94, 0.86, 0.94, W_SEC, head=0.050)
    c.arrow(0.06, 0.14, 0.06, 0.86, W_SEC, head=0.050)


def agreement_03(c):
    """A kappa dial reading how far past chance the two raters got."""
    c.arc(0.50, 0.72, 0.340, 0, 180, W_MAIN)
    c.line(0.16, 0.72, 0.84, 0.72, W_SEC)
    for i in range(1, 4):
        a = math.radians(180.0 * i / 4.0)
        c.line(0.50 + 0.290 * math.cos(a), 0.72 - 0.290 * math.sin(a),
               0.50 + 0.345 * math.cos(a), 0.72 - 0.345 * math.sin(a), W_SEC)
    a = math.radians(126.0)
    c.line(0.50, 0.72, 0.50 + 0.265 * math.cos(a), 0.72 - 0.265 * math.sin(a),
           W_MAIN)
    c.disc(0.50, 0.72, 0.056)
    mark(c, 0.16, 0.90, 0.070, "fill")
    mark(c, 0.84, 0.90, 0.070, "ring")


def agreement_04(c):
    """Two label sets overlapping, the agreed middle solid."""
    c.clip_circle(0.36, 0.50, 0.300)
    c.disc(0.64, 0.50, 0.300)
    c.unclip()
    c.circ(0.36, 0.50, 0.300, W_MAIN)
    c.circ(0.64, 0.50, 0.300, W_MAIN)


def agreement_05(c):
    """Two annotators nodding at the same call."""
    for cx in (0.20, 0.80):
        c.circ(cx, 0.30, 0.140, W_MAIN)
        c.arc(cx, 0.86, 0.250, 20, 140, W_MAIN)
    c.circ(0.50, 0.42, 0.175, W_MAIN)
    tick(c, 0.50, 0.42, 0.100)


def agreement_06(c):
    """A disputed image pulled up for review, a tick against a cross."""
    c.rect(0.34, 0.40, 0.32, 0.36, w=W_MAIN, r=0.032)
    c.disc(0.50, 0.58, 0.072)
    tick(c, 0.14, 0.20, 0.110)
    cross(c, 0.86, 0.20, 0.105)
    c.arrow(0.20, 0.34, 0.38, 0.46, W_SEC, head=0.060)
    c.arrow(0.80, 0.34, 0.62, 0.46, W_SEC, head=0.060)


def agreement_07(c):
    """Two label ribbons compared tile by tile, the odd tile flagged."""
    top = (2, 0, 2, 0)
    bot = (2, 0, 0, 0)
    for i in range(4):
        x = 0.06 + i * 0.230
        for j, row in enumerate((top, bot)):
            y = 0.16 + j * 0.28
            if row[i]:
                c.rect(x, y, 0.190, 0.220, filled=True, r=0.030)
            else:
                c.rect(x, y, 0.190, 0.220, w=W_MAIN, r=0.030)
        if top[i] != bot[i]:
            cross(c, x + 0.095, 0.82, 0.090)


def agreement_08(c):
    """The two annotators' calls weighed against each other."""
    c.line(0.50, 0.30, 0.50, 0.88, W_MAIN)
    c.line(0.30, 0.88, 0.70, 0.88, W_MAIN)
    c.line(0.14, 0.30, 0.86, 0.30, W_MAIN)
    c.line(0.14, 0.30, 0.14, 0.44, W_SEC)
    c.line(0.86, 0.30, 0.86, 0.44, W_SEC)
    c.arc(0.14, 0.44, 0.140, 200, 140, W_MAIN)
    c.arc(0.86, 0.44, 0.140, 200, 140, W_MAIN)
    c.disc(0.50, 0.30, 0.055)
    mark(c, 0.14, 0.14, 0.070, "fill")
    mark(c, 0.86, 0.14, 0.070, "ring")


def agreement_09(c):
    """Three raters' columns resolved into one majority column."""
    calls = ((2, 2, 2), (2, 0, 2), (0, 0, 0))
    for j in range(3):
        for i in range(3):
            mark(c, 0.10 + i * 0.19, 0.18 + j * 0.32, 0.072,
                 "fill" if calls[j][i] else "ring")
    c.line(0.60, 0.06, 0.60, 0.94, W_MAIN)
    for j, row in enumerate(calls):
        maj = 1 if sum(1 for v in row if v) >= 2 else 0
        mark(c, 0.84, 0.18 + j * 0.32, 0.100, "fill" if maj else "ring")


def agreement_10(c):
    """Observed agreement with the chance share cut off the front."""
    c.rect(0.06, 0.30, 0.88, 0.26, w=W_MAIN, r=0.070)
    c.rect(0.38, 0.30, 0.36, 0.26, filled=True, r=0.070)
    c.line(0.34, 0.18, 0.34, 0.68, W_MAIN, dash=[6, 7])
    brace(c, 0.38, 0.74, 0.78, depth=0.080, w=W_MAIN)
    c.disc(0.56, 0.90, 0.058)


# =====================================================================
# classifier_evaluation -- held-out prediction quality (never a 2x2 grid)
# =====================================================================

def classifier_evaluation_01(c):
    """An ROC curve bulging away from the chance diagonal."""
    c.axes(0.14, 0.08, 0.94, 0.88, w=W_MAIN)
    c.line(0.16, 0.88, 0.92, 0.12, W_SEC, dash=[8, 9])
    c.smooth(roc_pts(0.16, 0.92, 0.88, 0.12, p=0.32), w=W_MAIN)
    c.disc(0.40, 0.32, 0.050)


def classifier_evaluation_02(c):
    """Calibration: predicted against observed, points sagging off the line."""
    c.axes(0.14, 0.08, 0.94, 0.88, w=W_MAIN)
    c.line(0.16, 0.88, 0.92, 0.12, W_SEC, dash=[8, 9])
    pts = ((0.26, 0.82), (0.42, 0.72), (0.58, 0.52), (0.74, 0.40), (0.88, 0.20))
    c.smooth(pts, w=W_MAIN)
    for x, y in pts:
        c.disc(x, y, 0.040)


def classifier_evaluation_03(c):
    """Cross-validation folds: a different block held out in every row."""
    held = (2, 0, 3)
    for j in range(3):
        y = 0.18 + j * 0.26
        for i in range(4):
            x = 0.06 + i * 0.23
            c.rect(x, y, 0.19, 0.17, filled=(i == held[j]), w=W_MAIN, r=0.085)


def classifier_evaluation_04(c):
    """A test set sealed away from training and only opened to score."""
    c.rect(0.06, 0.26, 0.46, 0.48, w=W_MAIN, r=0.036)
    for j in range(2):
        for i in range(3):
            c.disc(0.14 + i * 0.15, 0.40 + j * 0.20, 0.045)
    c.rect(0.62, 0.32, 0.32, 0.32, w=W_MAIN, r=0.030)
    c.arc(0.78, 0.32, 0.110, 0, 180, W_MAIN)
    c.disc(0.78, 0.47, 0.055)
    c.arrow(0.78, 0.70, 0.78, 0.84, W_MAIN, head=0.070)
    c.bar(0.62, 0.88, 0.32, 0.090, filled=True)


def classifier_evaluation_05(c):
    """Two score humps pulling apart either side of the decision point."""
    c.line(0.06, 0.78, 0.94, 0.78, W_MAIN)
    bell(c, 0.30, 0.78, 0.24, 0.44)
    bell(c, 0.72, 0.78, 0.22, 0.36, w=W_SEC)
    c.line(0.51, 0.14, 0.51, 0.90, W_MAIN, dash=[7, 8])
    c.disc(0.51, 0.94, 0.048)


def classifier_evaluation_06(c):
    """A precision-recall curve falling away as recall is pushed."""
    c.axes(0.14, 0.08, 0.94, 0.88, w=W_MAIN)
    c.smooth([(0.16 + 0.76 * t / 10.0,
               0.16 + 0.66 * (t / 10.0) ** 2.6) for t in range(11)], w=W_MAIN)
    c.disc(0.64, 0.31, 0.050)
    c.line(0.14, 0.31, 0.64, 0.31, W_SEC, dash=[6, 7])


def classifier_evaluation_07(c):
    """Leakage: the same item found in both the train and the test block."""
    c.rect(0.04, 0.14, 0.38, 0.38, w=W_MAIN, r=0.032)
    c.rect(0.58, 0.14, 0.38, 0.38, w=W_MAIN, r=0.032)
    c.disc(0.23, 0.33, 0.090)
    c.disc(0.77, 0.33, 0.090)
    c.line(0.23, 0.52, 0.23, 0.78, W_MAIN)
    c.line(0.77, 0.52, 0.77, 0.78, W_MAIN)
    c.line(0.23, 0.78, 0.77, 0.78, W_MAIN)
    cross(c, 0.50, 0.78, 0.150)


def classifier_evaluation_08(c):
    """Held-out accuracy plate by plate, one plate falling off the line."""
    c.line(0.06, 0.90, 0.94, 0.90, W_MAIN)
    c.line(0.06, 0.46, 0.94, 0.46, W_SEC, dash=[7, 8])
    for i, h in enumerate((0.52, 0.58, 0.22, 0.56)):
        x = 0.11 + i * 0.21
        if h < 0.40:
            c.rect(x, 0.90 - h, 0.15, h, w=W_MAIN)
            cross(c, x + 0.075, 0.56, 0.062)
        else:
            c.rect(x, 0.90 - h, 0.15, h, filled=True)


def classifier_evaluation_09(c):
    """A decision boundary with the few points landing on the wrong side."""
    c.line(0.10, 0.86, 0.88, 0.16, W_MAIN)
    for x, y in ((0.20, 0.28), (0.34, 0.20), (0.30, 0.44), (0.48, 0.22)):
        c.disc(x, y, 0.052)
    for x, y in ((0.56, 0.66), (0.70, 0.56), (0.72, 0.80), (0.50, 0.82)):
        c.circ(x, y, 0.052, W_MAIN)
    c.circ(0.24, 0.60, 0.052, W_MAIN)
    c.disc(0.72, 0.34, 0.052)


def classifier_evaluation_10(c):
    """The same curve redrawn for every fold: how much the score wobbles."""
    c.axes(0.14, 0.08, 0.94, 0.88, w=W_MAIN)
    c.line(0.16, 0.88, 0.92, 0.12, W_SEC, dash=[8, 9])
    for k, p in enumerate((0.26, 0.34, 0.44)):
        c.smooth(roc_pts(0.16, 0.92, 0.88, 0.12, p=p),
                 w=W_MAIN if k == 1 else W_SEC)


# =====================================================================
# barcode_qc -- read abundance, and where the cutoff lands on it
# =====================================================================

def barcode_qc_01(c):
    """A rank-abundance curve falling through the abundance cutoff."""
    c.axes(0.12, 0.08, 0.94, 0.88, w=W_MAIN)
    c.smooth(decay_pts(0.16, 0.92, 0.14, 0.78, k=4.4), w=W_MAIN)
    c.line(0.12, 0.60, 0.94, 0.60, W_MAIN, dash=[8, 9])
    c.disc(0.06, 0.60, 0.050)


def barcode_qc_02(c):
    """Counts sorted tallest first, the tail below the cutoff crossed off."""
    c.line(0.06, 0.90, 0.94, 0.90, W_MAIN)
    hs = (0.68, 0.52, 0.36, 0.14)
    for i, h in enumerate(hs):
        x = 0.08 + i * 0.19
        if h >= 0.30:
            c.rect(x, 0.90 - h, 0.14, h, filled=True)
        else:
            c.rect(x, 0.90 - h, 0.14, h, w=W_MAIN)
    c.line(0.04, 0.60, 0.96, 0.60, W_MAIN, dash=[7, 8])
    cross(c, 0.88, 0.78, 0.080)


def barcode_qc_03(c):
    """The mapping run came back clean: a barcode with a tick on it."""
    barcode(c, 0.08, 0.10, 0.84, 0.34)
    c.circ(0.70, 0.74, 0.210, W_MAIN)
    tick(c, 0.70, 0.74, 0.125)


def barcode_qc_04(c):
    """A funnel: all the reads narrowing to the ones that pass."""
    c.polyline([(0.06, 0.10), (0.94, 0.10), (0.60, 0.54), (0.60, 0.86),
                (0.40, 0.94), (0.40, 0.54)], w=W_MAIN, close=True)
    c.line(0.20, 0.30, 0.80, 0.30, W_SEC, dash=[7, 8])
    c.line(0.34, 0.47, 0.66, 0.47, W_SEC, dash=[7, 8])
    for i in range(4):
        c.disc(0.20 + i * 0.20, 0.19, 0.036)


def barcode_qc_05(c):
    """Reads stacked onto one barcode: how deep the coverage went."""
    rows = ((0.10, 0.44), (0.30, 0.40), (0.16, 0.52), (0.42, 0.32), (0.24, 0.38))
    for j, (x, w) in enumerate(rows):
        c.bar(x, 0.60 - j * 0.130, w, 0.082, filled=True)
    barcode(c, 0.08, 0.80, 0.84, 0.140, widths=(0.9, 0.4, 0.7, 0.3))


def barcode_qc_06(c):
    """The count histogram with the knee where background stops."""
    c.line(0.08, 0.88, 0.94, 0.88, W_MAIN)
    hs = (0.16, 0.28, 0.52, 0.70, 0.44)
    for i, h in enumerate(hs):
        c.rect(0.08 + i * 0.18, 0.88 - h, 0.130, h, filled=True)
    c.line(0.395, 0.10, 0.395, 0.88, W_MAIN, dash=[7, 8])
    c.disc(0.395, 0.95, 0.055)


def barcode_qc_07(c):
    """A threshold handle dragged along the abundance axis."""
    c.smooth(decay_pts(0.10, 0.94, 0.12, 0.54, k=4.2), w=W_MAIN)
    c.line(0.59, 0.44, 0.59, 0.68, W_SEC, dash=[5, 6])
    c.line(0.06, 0.78, 0.94, 0.78, W_MAIN)
    c.rect(0.552, 0.68, 0.075, 0.20, filled=True, r=0.030)
    c.arrow(0.68, 0.78, 0.88, 0.78, W_SEC, head=0.055)
    c.arrow(0.50, 0.78, 0.30, 0.78, W_SEC, head=0.055)


def barcode_qc_08(c):
    """Reads splitting into the mapped pile and the unmapped one."""
    c.bar(0.34, 0.06, 0.32, 0.14, filled=True)
    c.arrow(0.44, 0.24, 0.22, 0.46, W_MAIN, head=0.075)
    c.arrow(0.56, 0.24, 0.78, 0.46, W_MAIN, head=0.075)
    c.rect(0.06, 0.54, 0.32, 0.26, filled=True, r=0.036)
    c.rect(0.62, 0.60, 0.32, 0.20, w=W_MAIN, r=0.036)
    tick(c, 0.22, 0.94, 0.080)
    cross(c, 0.78, 0.94, 0.075)


def barcode_qc_09(c):
    """Two barcodes side by side: one abundant, one down at background."""
    c.line(0.06, 0.88, 0.94, 0.88, W_MAIN)
    c.rect(0.14, 0.20, 0.24, 0.68, filled=True)
    c.rect(0.62, 0.72, 0.24, 0.16, w=W_MAIN)
    c.line(0.06, 0.62, 0.94, 0.62, W_MAIN, dash=[8, 9])
    cross(c, 0.74, 0.42, 0.090)


def barcode_qc_10(c):
    """A waterline across the barcodes: above it kept, below it dropped."""
    c.line(0.06, 0.50, 0.94, 0.50, W_MAIN, dash=[9, 10])
    for x, y in ((0.16, 0.16), (0.34, 0.28), (0.52, 0.20), (0.70, 0.34),
                 (0.86, 0.24)):
        c.disc(x, y, 0.056)
    for x, y in ((0.20, 0.68), (0.38, 0.80), (0.58, 0.70), (0.78, 0.84)):
        c.circ(x, y, 0.056, W_MAIN)
    c.disc(0.06, 0.50, 0.048)


# =====================================================================
# hit_list -- a RANKED list of hits
# =====================================================================

def hit_list_01(c):
    """Rows sorted longest first, the leaders starred."""
    for j, w in enumerate((0.62, 0.52, 0.40, 0.30, 0.22)):
        y = 0.14 + j * 0.19
        c.bar(0.24, y - 0.045, w, 0.090, filled=True)
        if j < 2:
            star(c, 0.13, y, 0.078)
        else:
            c.disc(0.13, y, 0.030)


def hit_list_02(c):
    """A volcano with the two far corners picked out as hits."""
    c.axes(0.12, 0.08, 0.94, 0.88, w=W_MAIN)
    c.line(0.53, 0.08, 0.53, 0.88, W_SEC, dash=[7, 8])
    for x, y in ((0.32, 0.72), (0.44, 0.82), (0.53, 0.76), (0.64, 0.82),
                 (0.75, 0.70), (0.40, 0.60), (0.66, 0.58)):
        c.disc(x, y, 0.046)
    star(c, 0.22, 0.24, 0.100)
    star(c, 0.85, 0.20, 0.100)


def hit_list_03(c):
    """A podium: the top three hits ranked one, two, three."""
    c.rect(0.36, 0.24, 0.28, 0.70, filled=True)
    c.rect(0.06, 0.44, 0.28, 0.50, w=W_MAIN)
    c.rect(0.66, 0.56, 0.28, 0.38, w=W_MAIN)
    star(c, 0.50, 0.12, 0.105)
    c.disc(0.20, 0.32, 0.060)
    c.disc(0.80, 0.44, 0.060)


def hit_list_04(c):
    """A ranked list cut by the FDR line: kept above, dropped below."""
    for j, w in enumerate((0.66, 0.56, 0.44, 0.34, 0.24)):
        y = 0.12 + j * 0.19
        if j < 3:
            c.bar(0.10, y - 0.048, w, 0.096, filled=True)
        else:
            c.bar(0.10, y - 0.048, w, 0.096, filled=False, w=W_MAIN)
    c.line(0.04, 0.60, 0.96, 0.60, W_MAIN)
    c.disc(0.92, 0.60, 0.052)


def hit_list_05(c):
    """A long list filtered down through a funnel into a shortlist."""
    for j, w in enumerate((0.70, 0.62, 0.70, 0.56)):
        c.bar(0.15, 0.06 + j * 0.095, w, 0.062, filled=True)
    c.polyline([(0.14, 0.46), (0.86, 0.46), (0.58, 0.66), (0.42, 0.66)],
               w=W_MAIN, close=True)
    for j, w in enumerate((0.40, 0.30)):
        c.bar(0.30, 0.74 + j * 0.115, w, 0.070, filled=True)
    star(c, 0.82, 0.79, 0.070)


def hit_list_06(c):
    """Guides agreeing on the top hit and scattering on the one below."""
    for j, y in enumerate((0.28, 0.72)):
        c.bar(0.20, y - 0.058, 0.32, 0.116, filled=True)
        if j == 0:
            star(c, 0.09, y, 0.090)
            for i in range(3):
                c.disc(0.64 + i * 0.15, y, 0.060)
        else:
            c.disc(0.09, y, 0.040)
            for i, dy in enumerate((-0.13, 0.04, 0.15)):
                c.circ(0.64 + i * 0.15, y + dy, 0.060, W_MAIN)


def hit_list_07(c):
    """One hit lifted out of the ranking for a closer look."""
    for j, w in enumerate((0.54, 0.44, 0.34)):
        c.bar(0.10, 0.52 + j * 0.16, w, 0.090, filled=True)
    c.bar(0.10, 0.36, 0.62, 0.090, filled=False, w=W_MAIN)
    c.arrow(0.42, 0.34, 0.42, 0.16, W_MAIN, head=0.080)
    star(c, 0.78, 0.16, 0.105)


def hit_list_08(c):
    """Effect sizes fanned either side of no-effect, biggest at the top."""
    c.line(0.50, 0.06, 0.50, 0.94, W_MAIN, dash=[8, 9])
    spans = ((-0.40, 0.14), (0.32, 0.32), (-0.24, 0.50), (0.16, 0.68),
             (-0.10, 0.86))
    for d, y in spans:
        if d < 0:
            c.bar(0.50 + d, y - 0.048, -d, 0.096, filled=True)
        else:
            c.bar(0.50, y - 0.048, d, 0.096, filled=True)
    star(c, 0.86, 0.14, 0.072)


def hit_list_09(c):
    """The ranking column being sorted, top to bottom."""
    c.rect(0.06, 0.06, 0.88, 0.88, w=W_MAIN, r=0.036)
    c.line(0.06, 0.28, 0.94, 0.28, W_MAIN)
    c.arrow(0.80, 0.10, 0.80, 0.24, W_MAIN, head=0.070)
    textline(c, 0.14, 0.17, 0.46, 0.048)
    for j, w in enumerate((0.56, 0.44, 0.32)):
        c.bar(0.14, 0.40 + j * 0.18, w, 0.090, filled=True)


def hit_list_10(c):
    """A shortlist flagged out of the ranking, hit by hit."""
    for j, w in enumerate((0.50, 0.42, 0.34, 0.26)):
        y = 0.16 + j * 0.22
        c.bar(0.34, y - 0.050, w, 0.100, filled=True)
        c.line(0.10, y - 0.115, 0.10, y + 0.115, W_MAIN)
        pen = [(0.10, y - 0.105), (0.26, y - 0.055), (0.10, y - 0.005)]
        if j in (0, 2):
            c.polyline(pen, close=True, filled=True)
        else:
            c.polyline(pen, w=W_MAIN, close=True)


# =====================================================================
# manifest
# =====================================================================

GROUPS = {
    "plate_view": ("plate_view -- the plate AS A HEATMAP, values read off it", [
        ("Wells graded solid to empty across the plate: a heatmap gradient.",
         plate_view_01),
        ("A hot outer ring: the edge wells solid, the interior barely marked.",
         plate_view_02),
        ("A magnifier over the plate, the well under it read as a solid disc.",
         plate_view_03),
        ("The plate beside its value key: solid, outlined and empty wells.",
         plate_view_04),
        ("A column profile drawn under the plate, rising at both edges.",
         plate_view_05),
        ("A measured column of bars poured into the plate as well values.",
         plate_view_06),
        ("A diagonal gradient: one corner of the plate hot, the far one cold.",
         plate_view_07),
        ("The plate rendered as heat tiles: filled squares against empty ones.",
         plate_view_08),
        ("A callout hanging off one well, showing the value it holds.",
         plate_view_09),
        ("One column of the plate reading hot: a plating artefact spotted.",
         plate_view_10),
    ]),
    "experiment_design": (
        "experiment_design -- the plate as a LAYOUT being authored", [
            ("Condition marks assigned across the plate, with a key beside it.",
             experiment_design_01),
            ("The plate cut into three treatment blocks, each braced as one.",
             experiment_design_02),
            ("Controls parked in the outside columns, samples in the middle.",
             experiment_design_03),
            ("One condition chip copied into three replicate wells.",
             experiment_design_04),
            ("A pen dropping the next condition into an empty well.",
             experiment_design_05),
            ("Condition chips waiting in a palette, one dragged onto the plate.",
             experiment_design_06),
            ("Randomisation: two crossed arrows shuffling the marks about.",
             experiment_design_07),
            ("A half-authored plate: assigned wells solid, the rest still blank.",
             experiment_design_08),
            ("A dose series laid along a row, the mark growing well by well.",
             experiment_design_09),
            ("The finished layout exported as a table for the pipeline.",
             experiment_design_10),
        ]),
    "power": ("power -- a CURVE you read the required sample size off", [
        ("A power curve with the required n dropped onto the x axis.", power_01),
        ("Two effect sizes, two curves, two required n's on the same axis.",
         power_02),
        ("The effect size itself: two humps with the gap measured between them.",
         power_03),
        ("Cells per well: a well packed with cells feeding a rising curve.",
         power_04),
        ("A slider set to an effect size, the curve above answering with n.",
         power_05),
        ("Error bars shrinking as n grows until one clears the effect line.",
         power_06),
        ("How many wells: a run of wells counted off under one brace.",
         power_07),
        ("A trade-off curve: pick an effect size, read the n it costs.",
         power_08),
        ("Bars of growing n against a detectable line: the first to clear wins.",
         power_09),
        ("Nested power contours with the chosen n and effect size pinned on.",
         power_10),
    ]),
    "qc_dashboard": (
        "qc_dashboard -- several verdicts at once resolving into one", [
            ("Four check panels, three ticked and one crossed, under one verdict.",
             qc_dashboard_01),
            ("Three small dials read at once, answered by one big dial.",
             qc_dashboard_02),
            ("A checklist of QC rows, ticked or crossed, totalled at the bottom.",
             qc_dashboard_03),
            ("One frame holding four unlike mini-panels and a verdict badge.",
             qc_dashboard_04),
            ("Three checks feeding a traffic light that shows the overall call.",
             qc_dashboard_05),
            ("Separate ticks converging into a single overall tick.",
             qc_dashboard_06),
            ("A pass badge assembled from four separate checks around it.",
             qc_dashboard_07),
            ("A scorecard: four segments filled bar one, and the resulting call.",
             qc_dashboard_08),
            ("One dial split into a pass side and a fail side, needle on pass.",
             qc_dashboard_09),
            ("A deck of check cards with the summed verdict on the front one.",
             qc_dashboard_10),
        ]),
    "report": ("report -- the shareable DOCUMENT, not the analysis", [
        ("A page carrying a bar figure and three lines of text.", report_01),
        ("A finished page with a share arrow springing off it.", report_02),
        ("A page dropping into an export tray.", report_03),
        ("Two pages fanned out: the figures sheet over the write-up.",
         report_04),
        ("The QC verdict stamped at the head of the page, findings beneath.",
         report_05),
        ("The HTML version: a browser window with a figure inside it.",
         report_06),
        ("A page published behind a shareable link.", report_07),
        ("The report sent out: a page tucked into an envelope.", report_08),
        ("One page holding all of it: a figure, a table and the settings.",
         report_09),
        ("A page sealed with a stamp: the versions and settings on record.",
         report_10),
    ]),
    "agreement": ("agreement -- two annotators' labels compared", [
        ("Two label columns joined pair by pair, one pair crossed out.",
         agreement_01),
        ("A 2x2 square of the two annotators' calls, agreements on the diagonal.",
         agreement_02),
        ("A kappa dial reading how far past chance the two raters got.",
         agreement_03),
        ("Two label sets overlapping, the agreed middle solid.", agreement_04),
        ("Two annotators nodding at the same call.", agreement_05),
        ("A disputed image pulled up for review, a tick against a cross.",
         agreement_06),
        ("Two label ribbons compared tile by tile, the odd tile flagged.",
         agreement_07),
        ("The two annotators' calls weighed against each other.", agreement_08),
        ("Three raters' columns resolved into one majority column.",
         agreement_09),
        ("Observed agreement with the chance share cut off the front.",
         agreement_10),
    ]),
    "classifier_evaluation": (
        "classifier_evaluation -- held-out prediction quality", [
            ("An ROC curve bulging away from the chance diagonal.",
             classifier_evaluation_01),
            ("Calibration: predicted against observed, points sagging off the line.",
             classifier_evaluation_02),
            ("Cross-validation folds: a different block held out in every row.",
             classifier_evaluation_03),
            ("A test set sealed away from training and only opened to score.",
             classifier_evaluation_04),
            ("Two score humps pulling apart either side of the decision point.",
             classifier_evaluation_05),
            ("A precision-recall curve falling away as recall is pushed.",
             classifier_evaluation_06),
            ("Leakage: the same item found in both the train and the test block.",
             classifier_evaluation_07),
            ("Held-out accuracy plate by plate, one plate falling off the line.",
             classifier_evaluation_08),
            ("A decision boundary with the few points landing on the wrong side.",
             classifier_evaluation_09),
            ("The same curve redrawn for every fold: how much the score wobbles.",
             classifier_evaluation_10),
        ]),
    "barcode_qc": ("barcode_qc -- read abundance and where the cutoff lands", [
        ("A rank-abundance curve falling through the abundance cutoff.",
         barcode_qc_01),
        ("Counts sorted tallest first, the tail below the cutoff crossed off.",
         barcode_qc_02),
        ("The mapping run came back clean: a barcode with a tick on it.",
         barcode_qc_03),
        ("A funnel: all the reads narrowing to the ones that pass.",
         barcode_qc_04),
        ("Reads stacked onto one barcode: how deep the coverage went.",
         barcode_qc_05),
        ("The count histogram with the knee where background stops.",
         barcode_qc_06),
        ("A threshold handle dragged along the abundance axis.", barcode_qc_07),
        ("Reads splitting into the mapped pile and the unmapped one.",
         barcode_qc_08),
        ("Two barcodes side by side: one abundant, one down at background.",
         barcode_qc_09),
        ("A waterline across the barcodes: above it kept, below it dropped.",
         barcode_qc_10),
    ]),
    "hit_list": ("hit_list -- the ranked, flagged shortlist of hits", [
        ("Rows sorted longest first, the leaders starred.", hit_list_01),
        ("A volcano with the two far corners picked out as hits.", hit_list_02),
        ("A podium: the top three hits ranked one, two, three.", hit_list_03),
        ("A ranked list cut by the FDR line: kept above, dropped below.",
         hit_list_04),
        ("A long list filtered down through a funnel into a shortlist.",
         hit_list_05),
        ("Guides agreeing on the top hit and scattering on the one below.",
         hit_list_06),
        ("One hit lifted out of the ranking for a closer look.", hit_list_07),
        ("Effect sizes fanned either side of no-effect, biggest at the top.",
         hit_list_08),
        ("The ranking column being sorted, top to bottom.", hit_list_09),
        ("A shortlist flagged out of the ranking, hit by hit.", hit_list_10),
    ]),
}


def main(outdir):
    return emit_groups(outdir, GROUPS, "group_results_qc.py")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else default_outdir(__file__)))
