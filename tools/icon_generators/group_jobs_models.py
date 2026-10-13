#!/usr/bin/env python3
"""Candidate spaCR icons: the job/run/model management apps.

Ten conceptually different designs per app key, white-on-transparent flat
vector art in the house style set by ``plaque.png`` / ``measure.png``.

Eight apps, and the whole difficulty of the set is that half of them are
some flavour of "two things next to each other".  A candidate that would
serve equally well for a neighbouring key is a failed candidate, so each
group is pinned to one subject and every design in it has to keep that
line:

* ``distributed_jobs`` -- **WHERE the work runs**: remote machines, a
  scheduler fanning a queue out to nodes, a cable to somewhere else.
  Never a chart, never a record.
* ``run_history`` -- **the ARCHIVE and searching it**: many past runs,
  a query, a timeline, pass/fail marks.  Plural and retrospective.
* ``run_compare`` -- **two whole RUNS diffed**: two record cards, the
  changed lines marked, the counts differing.  Exactly two, and the
  subject is the *records*, not the images and not the curves.
* ``pipeline_graph`` -- **a DAG of artefacts**: nodes and edges, with
  stale and missing nodes drawn differently from fresh ones.
* ``model_compare`` -- **two SEGMENTATIONS of the SAME field**: the same
  cells outlined two different ways, the disagreement being the subject.
  If you can't see cells, it isn't this icon.
* ``model_zoo`` -- **a CATALOGUE to pick from**: shelved/carded models,
  a download, a bench score.  Many models, none of them running.
* ``train_compare`` -- **training CURVES overlaid on one axis**: several
  loss/accuracy traces in one frame, plus a settings diff.  One axis,
  many curves -- not two panels.
* ``profiler`` -- **ONE input moved, ONE prediction responding**: a lever
  and a readout, a partial-dependence curve with a cursor on it.  One
  model, one knob.

Everything is authored for 48 px first: few, large, high-contrast
elements, no text-like micro-detail, no thin hatching, and never more
than about four significant shapes.

Run standalone (deterministic -- no random draws at all):

    python group_jobs_models.py [OUTDIR]

Default OUTDIR is the backup_icons directory one level up.  Writes
``<OUTDIR>/<key>/<key>_NN.png`` plus CONCEPTS.md and the two contact
sheets, via the shared output stage in ``_emit``.  It never touches
anything in ``spacr/resources/icons/*.png``.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtGui import QPainter  # noqa: E402

from _draw import W_FINE, W_MAIN, W_SEC  # noqa: E402
from _emit import default_outdir, emit_groups  # noqa: E402

TAU = math.pi * 2


# ---------------------------------------------------------------------------
# shared sub-drawings
# ---------------------------------------------------------------------------

def card(c, x, y, w_, h, rows=3, w=W_SEC, r=0.03, lines=None, lw=W_FINE,
         inset=0.16):
    """A record card: rounded outline with a few horizontal text lines."""
    c.rect(x, y, w_, h, w=w, r=r)
    m = w_ * inset
    for i in range(rows):
        yy = y + h * (i + 1.0) / (rows + 1.0)
        frac = lines[i] if lines else 0.74
        if frac > 0:
            c.line(x + m, yy, x + m + (w_ - 2 * m) * frac, yy, lw)


def dashed_rect(c, x, y, w_, h, w=W_SEC, dash=(4, 5)):
    """A rectangle drawn with a broken outline (missing / not yet made)."""
    c.polyline([(x, y), (x + w_, y), (x + w_, y + h), (x, y + h)],
               w=w, close=True, dash=list(dash))


def tower(c, x, y, w_, h, lit=True, bays=2):
    """A machine: an upright box with bays and a status lamp."""
    c.rect(x, y, w_, h, w=W_SEC, r=0.02)
    rr = min(w_ / bays, h / bays) * 0.20
    for i in range(bays):
        yy = y + h * (i + 0.5) / bays
        if i:
            c.line(x, y + h * i / bays, x + w_, y + h * i / bays, W_FINE)
        if lit and i == 0:
            c.disc(x + w_ * 0.26, yy, rr)
        else:
            c.circ(x + w_ * 0.26, yy, rr, W_FINE)
        c.line(x + w_ * 0.50, yy, x + w_ * 0.80, yy, W_FINE)


def laptop(c, x, y, w_, h):
    """A local workstation: lid plus a splayed base."""
    c.rect(x, y, w_, h, w=W_SEC, r=0.02)
    c.polyline([(x - w_ * 0.14, y + h + h * 0.26),
                (x + w_ * 1.14, y + h + h * 0.26),
                (x + w_ * 0.96, y + h), (x + w_ * 0.04, y + h)],
               w=W_FINE, close=True)


def cloud(c, cx, cy, w_, h):
    """A solid cloud silhouette."""
    c.disc(cx - w_ * 0.26, cy + h * 0.04, h * 0.40)
    c.disc(cx + w_ * 0.24, cy + h * 0.06, h * 0.34)
    c.disc(cx - w_ * 0.02, cy - h * 0.14, h * 0.52)
    c.rect(cx - w_ * 0.44, cy + h * 0.02, w_ * 0.88, h * 0.42,
           filled=True, r=h * 0.21)


def tick(c, x, y, s=0.050, w=W_SEC):
    c.polyline([(x - s, y), (x - s * 0.22, y + s * 0.72),
                (x + s, y - s * 0.82)], w=w)


def cross(c, x, y, s=0.044, w=W_SEC):
    c.line(x - s, y - s, x + s, y + s, w)
    c.line(x - s, y + s, x + s, y - s, w)


def warn(c, x, y, s=0.075, w=W_SEC):
    """Warning triangle with a bang inside."""
    c.polyline([(x, y - s), (x + s * 0.98, y + s * 0.72),
                (x - s * 0.98, y + s * 0.72)], w=w, close=True)
    c.line(x, y - s * 0.26, x, y + s * 0.22, W_FINE)
    c.disc(x, y + s * 0.48, s * 0.10)


def magnifier(c, cx, cy, r, w=W_MAIN):
    c.circ(cx, cy, r, w)
    a = math.radians(48.0)
    c.line(cx + r * math.cos(a), cy + r * math.sin(a),
           cx + r * 2.0 * math.cos(a), cy + r * 2.0 * math.sin(a), w * 1.3)


def netglyph(c, cx, cy, s, w=W_FINE):
    """Three-node network mark: the 'this is a model' stamp."""
    a = (cx - s, cy - s * 0.78)
    b = (cx - s, cy + s * 0.78)
    d = (cx + s, cy)
    c.line(a[0], a[1], d[0], d[1], w)
    c.line(b[0], b[1], d[0], d[1], w)
    for px, py in (a, b, d):
        c.disc(px, py, s * 0.30)


def model_card(c, x, y, w_, h, w=W_SEC, glyph=True, bar_=True):
    """A catalogue card standing for one model."""
    c.rect(x, y, w_, h, w=w, r=0.03)
    if glyph:
        netglyph(c, x + w_ * 0.50, y + h * 0.38, min(w_, h) * 0.26)
    if bar_:
        c.line(x + w_ * 0.20, y + h * 0.78, x + w_ * 0.80, y + h * 0.78, W_FINE)


def shelf(c, x, y, w_, t=0.028):
    c.rect(x, y, w_, t, filled=True, r=t / 2)


#: fixed radial wobble -- keeps every "cell" deterministic and organic
WOBBLE = (1.05, 1.00, 0.96, 1.02, 1.06, 0.99, 0.94, 1.01,
          1.04, 0.97, 1.00, 0.95)


def blob(c, cx, cy, r, w=W_SEC, phase=0, dash=None, scale=1.0, filled=False,
         squash=0.94):
    """A cell-like closed outline; `phase`/`scale` give rival contours."""
    n = len(WOBBLE)
    pts = []
    for i in range(n):
        a = TAU * i / n
        rr = r * WOBBLE[(i + phase) % n] * scale
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a) * squash))
    c.smooth(pts, w=w, closed=True, dash=dash, filled=filled)


def frame(c, x, y, w_, h=None, w=W_SEC):
    """A field-of-view frame."""
    c.rect(x, y, w_, h if h is not None else w_, w=w, r=0.025)


def slider(c, x0, x1, y, hx, w=W_SEC, hr=0.058):
    """A horizontal input track with a solid handle."""
    c.line(x0, y, x1, y, w)
    c.line(x0, y - 0.032, x0, y + 0.032, W_FINE)
    c.line(x1, y - 0.032, x1, y + 0.032, W_FINE)
    c.disc(hx, y, hr)


def gauge(c, cx, cy, r, ang_deg, w=W_MAIN):
    """A half-dial readout with a needle: 'the prediction, right now'."""
    c.arc(cx, cy, r, 0, 180, w)
    c.line(cx - r, cy, cx - r * 0.74, cy, W_SEC)
    c.line(cx + r, cy, cx + r * 0.74, cy, W_SEC)
    a = math.radians(ang_deg)
    c.line(cx, cy, cx + r * 0.80 * math.cos(a), cy - r * 0.80 * math.sin(a),
           W_SEC)
    c.disc(cx, cy, r * 0.17)


def pd_curve(c, x0, x1, y0, y1, w=W_MAIN, dash=None, shape=0):
    """A partial-dependence style response curve across [x0,x1]."""
    pts = []
    for i in range(9):
        t = i / 8.0
        if shape == 0:
            v = 1.0 / (1.0 + math.exp(-9.0 * (t - 0.5)))
        elif shape == 1:
            v = t * t * (3 - 2 * t)
            v = 0.15 + 0.8 * v
        else:
            v = 0.5 + 0.5 * math.sin(TAU * 0.75 * t - 1.2)
        pts.append((x0 + (x1 - x0) * t, y0 - (y0 - y1) * v))
    c.smooth(pts, w=w, dash=dash)
    return pts


def pd_at(x0, x1, y0, y1, t, shape=0):
    v = 1.0 / (1.0 + math.exp(-9.0 * (t - 0.5)))
    if shape == 1:
        s = t * t * (3 - 2 * t)
        v = 0.15 + 0.8 * s
    elif shape == 2:
        v = 0.5 + 0.5 * math.sin(TAU * 0.75 * t - 1.2)
    return (x0 + (x1 - x0) * t, y0 - (y0 - y1) * v)


def loss_curve(c, x0, x1, y_hi, y_lo, k=3.0, floor=0.06, w=W_MAIN, dash=None,
               rising=False, n=9, start=1.0):
    """A decaying loss trace (or a rising accuracy trace).

    ``start`` scales the initial loss so overlaid runs do not all leave the
    y axis from the exact same pixel (which reads as one fat wedge).
    """
    pts = []
    for i in range(n):
        t = i / (n - 1.0)
        v = math.exp(-k * t) * (start - floor) + floor
        if rising:
            v = 1.0 - v
        pts.append((x0 + (x1 - x0) * t, y_lo - (y_lo - y_hi) * v))
    c.smooth(pts, w=w, dash=dash)
    return pts


def pnode(c, cx, cy, r, kind="fresh", w=W_MAIN):
    """A DAG node: fresh (solid), stale (hollow), missing (broken)."""
    if kind == "fresh":
        c.disc(cx, cy, r)
    elif kind == "stale":
        c.circ(cx, cy, r, w)
    else:
        c.circ(cx, cy, r, w, dash=[4, 5])


def edge(c, x1, y1, x2, y2, r1, r2, w=W_SEC, dash=None, head=0.058):
    """An arrow from the rim of one node to the rim of the next."""
    a = math.atan2(y2 - y1, x2 - x1)
    sx, sy = x1 + r1 * math.cos(a), y1 + r1 * math.sin(a)
    ex, ey = x2 - r2 * math.cos(a), y2 - r2 * math.sin(a)
    if dash:
        c.line(sx, sy, ex, ey, w, dash=list(dash))
    else:
        c.arrow(sx, sy, ex, ey, w, head=head)


# =====================================================================
# distributed_jobs -- WHERE the work runs
# =====================================================================

def distributed_jobs_01(c):
    """A head node fanning the work out to three worker machines."""
    c.rect(0.34, 0.06, 0.32, 0.17, w=W_MAIN, r=0.035)
    c.disc(0.50, 0.145, 0.040)
    for x in (0.16, 0.50, 0.84):
        c.arrow(0.50, 0.25, x, 0.52, W_SEC, head=0.058)
        tower(c, x - 0.145, 0.60, 0.29, 0.30, bays=1)


def distributed_jobs_02(c):
    """A rack cabinet of three blades, the top two running."""
    c.rect(0.18, 0.06, 0.64, 0.88, w=W_MAIN, r=0.045)
    for i, y in enumerate((0.15, 0.41, 0.67)):
        c.rect(0.26, y, 0.48, 0.19, w=W_SEC, r=0.025)
        if i < 2:
            c.disc(0.345, y + 0.095, 0.038)
        else:
            c.circ(0.345, y + 0.095, 0.038, W_FINE)
        c.line(0.44, y + 0.095, 0.66, y + 0.095, W_FINE)


def distributed_jobs_03(c):
    """A job card lifting off into a cloud."""
    cloud(c, 0.54, 0.28, 0.80, 0.40)
    card(c, 0.26, 0.66, 0.48, 0.28, rows=2, w=W_MAIN, lines=(0.80, 0.50))
    c.arrow(0.50, 0.64, 0.50, 0.46, W_MAIN, head=0.075)


def distributed_jobs_04(c):
    """A terminal prompt wired down a cable to a machine somewhere else."""
    c.rect(0.04, 0.08, 0.52, 0.38, w=W_MAIN, r=0.035)
    c.line(0.04, 0.19, 0.56, 0.19, W_FINE)
    c.polyline([(0.14, 0.26), (0.23, 0.33), (0.14, 0.40)], w=W_SEC)
    c.line(0.28, 0.40, 0.44, 0.40, W_SEC)
    c.smooth([(0.22, 0.50), (0.36, 0.72), (0.56, 0.74)], w=W_SEC, dash=[7, 8])
    tower(c, 0.58, 0.52, 0.36, 0.42, bays=2)


def distributed_jobs_05(c):
    """A scheduler in the middle with worker nodes spoked around it."""
    c.disc(0.50, 0.50, 0.115)
    for i in range(5):
        a = -math.pi / 2 + TAU * i / 5
        x, y = 0.50 + 0.335 * math.cos(a), 0.50 + 0.335 * math.sin(a)
        c.line(0.50 + 0.135 * math.cos(a), 0.50 + 0.135 * math.sin(a),
               x - 0.098 * math.cos(a), y - 0.098 * math.sin(a), W_SEC)
        c.circ(x, y, 0.093, W_MAIN)


def distributed_jobs_06(c):
    """A scheduler allocation chart: jobs booked across node rows and time."""
    c.axes(0.16, 0.10, 0.94, 0.86, w=W_SEC)
    for j, (y, x0, x1) in enumerate(((0.22, 0.22, 0.62),
                                     (0.44, 0.34, 0.90),
                                     (0.66, 0.22, 0.50))):
        c.disc(0.09, y, 0.040)
        c.bar(x0, y - 0.055, x1 - x0, 0.11, filled=True)


def distributed_jobs_07(c):
    """A laptop pushing a run down a long cable to a far bigger machine."""
    laptop(c, 0.06, 0.46, 0.30, 0.22)
    c.arrow(0.40, 0.56, 0.60, 0.56, W_SEC, head=0.060)
    tower(c, 0.62, 0.14, 0.32, 0.72, bays=3)


def distributed_jobs_08(c):
    """One submitted job splitting into three parallel lanes of work."""
    card(c, 0.04, 0.36, 0.24, 0.28, rows=2, w=W_MAIN, lines=(0.78, 0.46))
    for j, y in enumerate((0.18, 0.50, 0.82)):
        c.arrow(0.30, 0.50, 0.50, y, W_FINE, head=0.048)
        c.bar(0.54, y - 0.070, 0.42, 0.140, filled=False, w=W_SEC)
        c.bar(0.567, y - 0.043, 0.33 - 0.11 * j, 0.086, filled=True)


def distributed_jobs_09(c):
    """A remote machine reporting a live progress bar back home."""
    tower(c, 0.30, 0.06, 0.40, 0.44, bays=2)
    c.bar(0.10, 0.60, 0.80, 0.13, filled=False, w=W_SEC)
    c.bar(0.125, 0.628, 0.44, 0.074, filled=True)
    c.polyline([(0.10, 0.87), (0.30, 0.87), (0.38, 0.79), (0.48, 0.94),
                (0.56, 0.87), (0.90, 0.87)], w=W_FINE)


def distributed_jobs_10(c):
    """A submit button firing a paper plane at a queue of remote machines."""
    c.polyline([(0.06, 0.22), (0.44, 0.10), (0.30, 0.36), (0.24, 0.27)],
               close=True, filled=True)
    c.smooth([(0.28, 0.34), (0.44, 0.44), (0.52, 0.42)], w=W_FINE, dash=[5, 6])
    for i, x in enumerate((0.20, 0.50, 0.80)):
        tower(c, x - 0.135, 0.56, 0.27, 0.36, lit=(i == 0))


# =====================================================================
# run_history -- the ARCHIVE, and searching it
# =====================================================================

def run_history_01(c):
    """A magnifier held over a deep stack of finished run records."""
    for dx, dy in ((0.14, 0.14), (0.09, 0.22), (0.04, 0.30)):
        c.rect(dx, dy, 0.56, 0.56, w=W_FINE, r=0.035)
    card(c, 0.04, 0.30, 0.56, 0.56, rows=3, w=W_MAIN,
         lines=(0.80, 0.58, 0.70))
    magnifier(c, 0.70, 0.36, 0.180, W_MAIN)


def run_history_02(c):
    """A timeline of past runs, each marked passed or failed."""
    c.line(0.05, 0.60, 0.95, 0.60, W_MAIN)
    for i, x in enumerate((0.16, 0.38, 0.60, 0.84)):
        c.line(x, 0.60, x, 0.70, W_SEC)
        c.disc(x, 0.60, 0.036)
        if i == 2:
            cross(c, x, 0.36, 0.055, W_SEC)
        else:
            tick(c, x, 0.36, 0.060, W_SEC)


def run_history_03(c):
    """A search field over the run rows it matched."""
    c.bar(0.06, 0.10, 0.88, 0.22, filled=False, w=W_MAIN)
    magnifier(c, 0.24, 0.21, 0.062, W_SEC)
    c.line(0.38, 0.21, 0.78, 0.21, W_SEC)
    for j, y in enumerate((0.48, 0.68, 0.88)):
        c.line(0.10, y, 0.62 - 0.10 * j, y, W_SEC)
        if j == 0:
            c.disc(0.86, y, 0.040)
        else:
            c.circ(0.86, y, 0.040, W_FINE)


def run_history_04(c):
    """A drawer of run folders with one pulled up out of the file."""
    c.rect(0.04, 0.48, 0.92, 0.46, w=W_MAIN, r=0.04)
    c.line(0.32, 0.72, 0.68, 0.72, W_SEC)
    for x in (0.14, 0.86):
        c.polyline([(x - 0.075, 0.48), (x - 0.075, 0.30), (x + 0.075, 0.30),
                    (x + 0.075, 0.48)], w=W_FINE)
    c.polyline([(0.30, 0.48), (0.30, 0.08), (0.52, 0.08), (0.57, 0.16),
                (0.74, 0.16), (0.74, 0.48)], w=W_SEC, close=True)


def run_history_05(c):
    """A clock wound backwards over the list of runs already done."""
    c.circ(0.32, 0.30, 0.200, W_MAIN)
    c.line(0.32, 0.30, 0.32, 0.17, W_SEC)
    c.line(0.32, 0.30, 0.43, 0.36, W_SEC)
    c.arc(0.32, 0.30, 0.280, 55, 195, W_SEC)
    c.arrow(0.16, 0.16, 0.08, 0.28, W_SEC, head=0.065, tail=False)
    for j, y in enumerate((0.60, 0.76, 0.92)):
        c.disc(0.12, y, 0.030)
        c.line(0.22, y, 0.90 - 0.14 * j, y, W_SEC)


def run_history_06(c):
    """A ledger of past runs with a pass/fail column down the right."""
    c.rect(0.06, 0.10, 0.88, 0.80, w=W_MAIN, r=0.04)
    c.line(0.06, 0.30, 0.94, 0.30, W_SEC)
    c.line(0.70, 0.10, 0.70, 0.90, W_SEC)
    c.line(0.14, 0.20, 0.50, 0.20, W_FINE)
    for j, y in enumerate((0.44, 0.60, 0.76)):
        c.line(0.14, y, 0.60 - 0.08 * j, y, W_FINE)
        if j == 1:
            cross(c, 0.82, y, 0.045, W_SEC)
        else:
            tick(c, 0.82, y, 0.050, W_SEC)


def run_history_07(c):
    """A calendar month with the days that were run marked off."""
    c.rect(0.08, 0.14, 0.84, 0.78, w=W_MAIN, r=0.045)
    c.line(0.08, 0.36, 0.92, 0.36, W_SEC)
    c.line(0.24, 0.06, 0.24, 0.20, W_SEC)
    c.line(0.76, 0.06, 0.76, 0.20, W_SEC)
    for i, x in enumerate((0.22, 0.50, 0.78)):
        for j, y in enumerate((0.50, 0.74)):
            if (i + j) % 2 == 0:
                c.disc(x, y, 0.058)
            else:
                c.circ(x, y, 0.058, W_FINE)


def run_history_08(c):
    """A run log unrolled, with the warning line flagged in it."""
    c.polyline([(0.16, 0.06), (0.86, 0.06), (0.86, 0.94), (0.16, 0.94)],
               w=W_MAIN, close=True)
    c.arc(0.16, 0.14, 0.080, 90, 180, W_SEC)
    c.arc(0.16, 0.86, 0.080, 90, 180, W_SEC)
    for y in (0.26, 0.42, 0.72, 0.86):
        c.line(0.26, y, 0.76, y, W_FINE)
    warn(c, 0.34, 0.56, 0.082, W_SEC)
    c.line(0.46, 0.57, 0.78, 0.57, W_SEC)


def run_history_09(c):
    """How long every past run took, plotted run after run."""
    c.axes(0.12, 0.08, 0.94, 0.84, w=W_SEC)
    hs = (0.26, 0.46, 0.34, 0.62, 0.20)
    for i, hgt in enumerate(hs):
        x = 0.18 + i * 0.155
        c.bar(x, 0.84 - hgt, 0.105, hgt, r=0.02, filled=True)


def run_history_10(c):
    """Boxed-up past runs on the shelf, the newest one still open."""
    c.rect(0.05, 0.44, 0.40, 0.44, w=W_MAIN, r=0.03)
    c.line(0.05, 0.60, 0.45, 0.60, W_SEC)
    c.line(0.25, 0.44, 0.25, 0.60, W_SEC)
    c.rect(0.55, 0.44, 0.40, 0.44, w=W_MAIN, r=0.03)
    c.polyline([(0.52, 0.44), (0.62, 0.30), (0.96, 0.30), (0.88, 0.44)],
               w=W_SEC, close=True)
    card(c, 0.58, 0.02, 0.34, 0.22, rows=2, w=W_SEC, lines=(0.80, 0.50))
    shelf(c, 0.02, 0.90, 0.96, 0.042)


# =====================================================================
# run_compare -- two whole RUNS diffed, record against record
# =====================================================================

def run_compare_01(c):
    """Two run records side by side, the one row that differs flagged in both."""
    for x, ln in ((0.05, 0.26), (0.55, 0.16)):
        c.rect(x, 0.12, 0.40, 0.76, w=W_MAIN, r=0.04)
        for j, y in enumerate((0.30, 0.50, 0.70)):
            c.line(x + 0.10, y, x + 0.34, y, W_FINE)
        c.line(x + 0.10, 0.50, x + 0.10 + ln, 0.50, W_MAIN)
        c.disc(x + 0.05, 0.50, 0.026)


def run_compare_02(c):
    """A diff gutter: rows removed on the left, added on the right."""
    c.rect(0.06, 0.10, 0.34, 0.80, w=W_SEC, r=0.035)
    c.rect(0.60, 0.10, 0.34, 0.80, w=W_SEC, r=0.035)
    for j, y in enumerate((0.28, 0.50, 0.72)):
        c.line(0.12, y, 0.34, y, W_FINE)
        c.line(0.66, y, 0.88, y, W_FINE)
    c.line(0.44, 0.28, 0.56, 0.28, W_MAIN)
    c.line(0.44, 0.72, 0.56, 0.72, W_MAIN)
    c.line(0.50, 0.66, 0.50, 0.78, W_MAIN)


def run_compare_03(c):
    """The same three counts from two runs, plotted at different heights."""
    c.line(0.06, 0.88, 0.94, 0.88, W_SEC)
    pairs = ((0.30, 0.54), (0.62, 0.44), (0.40, 0.72))
    for i, (a, b) in enumerate(pairs):
        x = 0.12 + i * 0.29
        c.bar(x, 0.88 - a, 0.105, a, r=0.02, filled=True)
        c.rect(x + 0.125, 0.88 - b, 0.105, b, w=W_SEC, r=0.02)


def run_compare_04(c):
    """Two hit lists as overlapping sets: shared hits and hits unique to one."""
    c.circ(0.37, 0.50, 0.290, W_MAIN)
    c.circ(0.63, 0.50, 0.290, W_MAIN)
    c.disc(0.50, 0.50, 0.055)
    c.disc(0.50, 0.34, 0.045)
    c.disc(0.50, 0.66, 0.045)
    c.circ(0.20, 0.50, 0.048, W_FINE)
    c.circ(0.80, 0.50, 0.048, W_FINE)


def run_compare_05(c):
    """A balance with a run record in each pan, tipped to the better one."""
    c.line(0.50, 0.20, 0.50, 0.86, W_SEC)
    c.line(0.26, 0.90, 0.74, 0.90, W_SEC)
    c.line(0.14, 0.24, 0.86, 0.38, W_MAIN)
    c.line(0.14, 0.24, 0.14, 0.36, W_FINE)
    c.line(0.86, 0.38, 0.86, 0.50, W_FINE)
    card(c, 0.01, 0.36, 0.26, 0.26, rows=2, w=W_SEC, lines=(0.76, 0.46))
    card(c, 0.73, 0.50, 0.26, 0.26, rows=2, w=W_SEC, lines=(0.76, 0.46))
    c.disc(0.50, 0.24, 0.048)


def run_compare_06(c):
    """Two run cards swapped back and forth against each other."""
    card(c, 0.05, 0.06, 0.44, 0.38, rows=2, w=W_MAIN, lines=(0.82, 0.52))
    card(c, 0.51, 0.56, 0.44, 0.38, rows=2, w=W_MAIN, lines=(0.82, 0.52))
    c.arrow(0.56, 0.24, 0.94, 0.24, W_SEC, head=0.060)
    c.arrow(0.44, 0.76, 0.06, 0.76, W_SEC, head=0.060)


def run_compare_07(c):
    """A delta between one run stacked over the other."""
    card(c, 0.06, 0.06, 0.88, 0.28, rows=2, w=W_SEC, lines=(0.72, 0.44))
    card(c, 0.06, 0.66, 0.88, 0.28, rows=2, w=W_SEC, lines=(0.72, 0.44))
    c.polyline([(0.50, 0.38), (0.66, 0.62), (0.34, 0.62)], w=W_MAIN,
               close=True)


def run_compare_08(c):
    """Two ranked hit lists, with one hit that jumped up the order."""
    for x in (0.06, 0.56):
        for j, y in enumerate((0.22, 0.44, 0.66, 0.88)):
            c.line(x, y, x + 0.32, y, W_SEC)
    c.disc(0.14, 0.66, 0.052)
    c.disc(0.64, 0.22, 0.052)
    c.smooth([(0.22, 0.66), (0.40, 0.62), (0.44, 0.30), (0.56, 0.24)],
             w=W_FINE)
    c.arrow(0.50, 0.28, 0.57, 0.235, W_FINE, head=0.050, tail=False)


def run_compare_09(c):
    """One settings sheet cut down the middle: before on the left, after right."""
    c.rect(0.08, 0.10, 0.84, 0.80, w=W_MAIN, r=0.04)
    c.line(0.50, 0.06, 0.50, 0.94, W_SEC, dash=[7, 8])
    for j, y in enumerate((0.28, 0.46, 0.64, 0.82)):
        c.line(0.15, y, 0.43, y, W_FINE)
        c.line(0.57, y, 0.85 - (0.20 if j == 1 else 0.0), y, W_FINE)
    c.disc(0.66, 0.46, 0.048)


def run_compare_10(c):
    """Two records checked line for line: three agree, one does not."""
    c.rect(0.06, 0.14, 0.34, 0.72, w=W_SEC, r=0.035)
    c.rect(0.60, 0.14, 0.34, 0.72, w=W_SEC, r=0.035)
    for j, y in enumerate((0.28, 0.50, 0.72)):
        c.line(0.12, y, 0.34, y, W_FINE)
        c.line(0.66, y, 0.88, y, W_FINE)
        if j == 1:
            cross(c, 0.50, y, 0.062, W_MAIN)
        else:
            tick(c, 0.50, y, 0.066, W_MAIN)


# =====================================================================
# pipeline_graph -- a DAG of artefacts, stale and missing marked
# =====================================================================

def pipeline_graph_01(c):
    """A three-step chain whose last artefact was never produced."""
    pnode(c, 0.50, 0.16, 0.115, "fresh")
    pnode(c, 0.50, 0.50, 0.115, "fresh")
    pnode(c, 0.50, 0.84, 0.115, "missing")
    edge(c, 0.50, 0.16, 0.50, 0.50, 0.115, 0.115, W_SEC)
    edge(c, 0.50, 0.50, 0.50, 0.84, 0.115, 0.115, W_SEC, dash=(5, 6))


def pipeline_graph_02(c):
    """One source forking to two products, one of them flagged bad."""
    pnode(c, 0.50, 0.16, 0.120, "fresh")
    pnode(c, 0.22, 0.72, 0.120, "fresh")
    pnode(c, 0.74, 0.72, 0.120, "stale")
    edge(c, 0.50, 0.16, 0.22, 0.72, 0.120, 0.120, W_SEC)
    edge(c, 0.50, 0.16, 0.74, 0.72, 0.120, 0.120, W_SEC)
    warn(c, 0.74, 0.72, 0.070, W_SEC)


def pipeline_graph_03(c):
    """Two branches converging on a join that is now out of date."""
    pnode(c, 0.20, 0.16, 0.105, "fresh")
    pnode(c, 0.80, 0.16, 0.105, "fresh")
    pnode(c, 0.50, 0.50, 0.115, "stale")
    pnode(c, 0.50, 0.86, 0.105, "missing")
    edge(c, 0.20, 0.16, 0.50, 0.50, 0.105, 0.115, W_SEC)
    edge(c, 0.80, 0.16, 0.50, 0.50, 0.105, 0.115, W_SEC)
    edge(c, 0.50, 0.50, 0.50, 0.86, 0.115, 0.105, W_SEC, dash=(5, 6))


def pipeline_graph_04(c):
    """File to file: two artefacts made, the third one still blank."""
    c.rect(0.05, 0.34, 0.24, 0.32, w=W_SEC, r=0.03)
    c.line(0.10, 0.46, 0.24, 0.46, W_FINE)
    c.line(0.10, 0.56, 0.24, 0.56, W_FINE)
    c.rect(0.38, 0.34, 0.24, 0.32, w=W_SEC, r=0.03)
    c.line(0.43, 0.46, 0.57, 0.46, W_FINE)
    c.line(0.43, 0.56, 0.57, 0.56, W_FINE)
    dashed_rect(c, 0.71, 0.34, 0.24, 0.32, W_SEC)
    c.arrow(0.31, 0.50, 0.36, 0.50, W_SEC, head=0.050)
    c.arrow(0.64, 0.50, 0.69, 0.50, W_SEC, head=0.050)


def pipeline_graph_05(c):
    """The link between two steps snapped in half."""
    pnode(c, 0.18, 0.26, 0.150, "fresh")
    pnode(c, 0.82, 0.74, 0.150, "missing")
    c.polyline([(0.30, 0.36), (0.46, 0.49), (0.38, 0.54)], w=W_MAIN)
    c.polyline([(0.70, 0.64), (0.54, 0.51), (0.62, 0.46)], w=W_MAIN)


def pipeline_graph_06(c):
    """Along one branch: made, gone stale, never made at all."""
    pnode(c, 0.16, 0.72, 0.125, "fresh")
    pnode(c, 0.50, 0.42, 0.125, "stale")
    pnode(c, 0.84, 0.72, 0.125, "missing")
    edge(c, 0.16, 0.72, 0.50, 0.42, 0.125, 0.125, W_SEC)
    edge(c, 0.50, 0.42, 0.84, 0.72, 0.125, 0.125, W_SEC, dash=(5, 6))


def pipeline_graph_07(c):
    """A node marked for rebuild, with everything below it waiting on it."""
    c.disc(0.50, 0.28, 0.130)
    c.arc(0.50, 0.28, 0.215, 30, 280, W_SEC)
    c.arrow(0.66, 0.13, 0.72, 0.24, W_SEC, head=0.062, tail=False)
    pnode(c, 0.22, 0.80, 0.115, "stale")
    pnode(c, 0.76, 0.80, 0.115, "stale")
    edge(c, 0.50, 0.28, 0.22, 0.80, 0.130, 0.115, W_SEC, dash=(6, 7))
    edge(c, 0.50, 0.28, 0.76, 0.80, 0.130, 0.115, W_SEC, dash=(6, 7))


def pipeline_graph_08(c):
    """A node carrying a clock: its inputs moved on without it."""
    pnode(c, 0.18, 0.20, 0.105, "fresh")
    pnode(c, 0.18, 0.80, 0.105, "fresh")
    c.circ(0.66, 0.50, 0.215, W_MAIN)
    c.line(0.66, 0.50, 0.66, 0.36, W_SEC)
    c.line(0.66, 0.50, 0.77, 0.56, W_SEC)
    edge(c, 0.18, 0.20, 0.66, 0.50, 0.105, 0.215, W_SEC)
    edge(c, 0.18, 0.80, 0.66, 0.50, 0.105, 0.215, W_SEC)


def pipeline_graph_09(c):
    """A step struck out, and everything it fed left dangling."""
    pnode(c, 0.24, 0.22, 0.115, "fresh")
    c.circ(0.24, 0.72, 0.115, W_MAIN)
    cross(c, 0.24, 0.72, 0.090, W_MAIN)
    edge(c, 0.24, 0.22, 0.24, 0.72, 0.115, 0.115, W_SEC)
    pnode(c, 0.76, 0.72, 0.115, "missing")
    edge(c, 0.24, 0.72, 0.76, 0.72, 0.115, 0.115, W_SEC, dash=(5, 6))


def pipeline_graph_10(c):
    """Provenance: one output traced back up to the two inputs that made it."""
    pnode(c, 0.20, 0.14, 0.100, "fresh")
    pnode(c, 0.72, 0.14, 0.100, "fresh")
    pnode(c, 0.46, 0.50, 0.115, "fresh")
    c.rect(0.24, 0.76, 0.46, 0.20, w=W_MAIN, r=0.035)
    c.line(0.31, 0.86, 0.63, 0.86, W_FINE)
    edge(c, 0.20, 0.14, 0.46, 0.50, 0.100, 0.115, W_SEC)
    edge(c, 0.72, 0.14, 0.46, 0.50, 0.100, 0.115, W_SEC)
    c.arrow(0.46, 0.615, 0.46, 0.74, W_SEC, head=0.058)


# =====================================================================
# model_compare -- two SEGMENTATIONS of the SAME field
# =====================================================================

def model_compare_01(c):
    """One cell wearing two rival contours, solid and broken, that disagree."""
    c.disc(0.50, 0.52, 0.070)
    blob(c, 0.48, 0.54, 0.360, w=W_MAIN, phase=0, scale=1.0)
    blob(c, 0.56, 0.47, 0.360, w=W_MAIN, phase=6, scale=0.78, dash=[13, 12])


def model_compare_02(c):
    """The same field wiped down the middle: one model's outlines each side."""
    frame(c, 0.05, 0.12, 0.90, 0.76, w=W_SEC)
    c.line(0.50, 0.06, 0.50, 0.94, W_SEC, dash=[7, 8])
    blob(c, 0.27, 0.34, 0.150, w=W_MAIN, phase=0)
    blob(c, 0.27, 0.66, 0.150, w=W_MAIN, phase=2)
    c.polyline([(0.62, 0.24), (0.82, 0.28), (0.86, 0.44), (0.66, 0.46)],
               w=W_MAIN, close=True)
    c.polyline([(0.62, 0.58), (0.84, 0.56), (0.86, 0.76), (0.64, 0.74)],
               w=W_MAIN, close=True)


def model_compare_03(c):
    """Two touching cells: one model calls them one object, the other two."""
    blob(c, 0.30, 0.32, 0.230, w=W_MAIN, phase=0)
    c.disc(0.21, 0.32, 0.048)
    c.disc(0.40, 0.32, 0.048)
    blob(c, 0.70, 0.72, 0.230, w=W_MAIN, phase=0)
    c.line(0.70, 0.53, 0.70, 0.91, W_SEC)
    c.disc(0.61, 0.72, 0.048)
    c.disc(0.80, 0.72, 0.048)


def model_compare_04(c):
    """Same three cells, but one model found a fourth: the extra one ringed."""
    frame(c, 0.04, 0.16, 0.42, 0.68, w=W_FINE)
    frame(c, 0.54, 0.16, 0.42, 0.68, w=W_FINE)
    for cx, cy in ((0.16, 0.34), (0.32, 0.62), (0.16, 0.72)):
        blob(c, cx, cy, 0.095, w=W_SEC, phase=1)
    for cx, cy in ((0.66, 0.34), (0.82, 0.62), (0.66, 0.72)):
        blob(c, cx, cy, 0.095, w=W_SEC, phase=1)
    blob(c, 0.85, 0.32, 0.078, w=W_SEC, phase=3)
    c.circ(0.85, 0.32, 0.130, W_MAIN)


def model_compare_05(c):
    """The sliver where the two outlines disagree, filled in solid."""
    blob(c, 0.44, 0.50, 0.340, w=W_MAIN, phase=0, filled=True)
    c.p.setCompositionMode(QPainter.CompositionMode_Clear)
    blob(c, 0.58, 0.50, 0.340, w=W_MAIN, phase=6, filled=True)
    c.p.setCompositionMode(QPainter.CompositionMode_SourceOver)
    blob(c, 0.44, 0.50, 0.340, w=W_MAIN, phase=0)
    blob(c, 0.58, 0.50, 0.340, w=W_MAIN, phase=6)


def model_compare_06(c):
    """One cell, a tight outline and a loose one, the gap between measured."""
    blob(c, 0.46, 0.52, 0.400, w=W_SEC, phase=0)
    blob(c, 0.46, 0.52, 0.400, w=W_MAIN, phase=0, scale=0.64)
    c.disc(0.46, 0.52, 0.055)
    c.line(0.715, 0.52, 0.855, 0.52, W_SEC)
    c.line(0.715, 0.46, 0.715, 0.58, W_SEC)
    c.line(0.855, 0.46, 0.855, 0.58, W_SEC)


def model_compare_07(c):
    """A cell that one model outlined and the other missed entirely."""
    frame(c, 0.04, 0.16, 0.42, 0.68, w=W_FINE)
    frame(c, 0.54, 0.16, 0.42, 0.68, w=W_FINE)
    blob(c, 0.17, 0.36, 0.105, w=W_SEC, phase=1)
    blob(c, 0.32, 0.66, 0.105, w=W_SEC, phase=2)
    blob(c, 0.67, 0.36, 0.105, w=W_SEC, phase=1)
    blob(c, 0.80, 0.64, 0.095, w=W_SEC, phase=2, filled=True)
    c.circ(0.80, 0.64, 0.145, W_MAIN, dash=[10, 11])


def model_compare_08(c):
    """The same cells outlined twice, and the two object counts beneath."""
    blob(c, 0.28, 0.32, 0.190, w=W_MAIN, phase=0)
    blob(c, 0.30, 0.30, 0.190, w=W_SEC, phase=3, scale=0.82, dash=[7, 8])
    blob(c, 0.72, 0.32, 0.170, w=W_MAIN, phase=2)
    blob(c, 0.70, 0.34, 0.170, w=W_SEC, phase=5, scale=0.82, dash=[7, 8])
    c.line(0.06, 0.94, 0.94, 0.94, W_FINE)
    c.bar(0.20, 0.66, 0.16, 0.26, r=0.03, filled=True)
    c.rect(0.60, 0.74, 0.16, 0.18, w=W_SEC, r=0.03)


def model_compare_09(c):
    """Objects matched up one to one between the two labellings."""
    for j, y in enumerate((0.18, 0.50, 0.82)):
        blob(c, 0.16, y, 0.115, w=W_SEC, phase=j)
    for j, y in enumerate((0.18, 0.50, 0.82)):
        blob(c, 0.84, y, 0.115, w=W_SEC, phase=j + 2)
    c.line(0.30, 0.18, 0.70, 0.18, W_FINE)
    c.line(0.30, 0.50, 0.70, 0.82, W_FINE)
    c.line(0.30, 0.82, 0.70, 0.50, W_FINE)


def model_compare_10(c):
    """A checkerboard of the same field, alternating whose outline is drawn."""
    frame(c, 0.06, 0.06, 0.88, 0.88, w=W_SEC)
    c.line(0.50, 0.06, 0.50, 0.94, W_FINE)
    c.line(0.06, 0.50, 0.94, 0.50, W_FINE)
    blob(c, 0.28, 0.28, 0.135, w=W_MAIN, phase=0)
    blob(c, 0.72, 0.72, 0.135, w=W_MAIN, phase=1)
    blob(c, 0.72, 0.28, 0.135, w=W_SEC, phase=2, dash=[6, 7])
    blob(c, 0.28, 0.72, 0.135, w=W_SEC, phase=3, dash=[6, 7])


# =====================================================================
# model_zoo -- a CATALOGUE of models to pick from
# =====================================================================

def model_zoo_01(c):
    """A shelf of model cards with the middle one pulled out."""
    model_card(c, 0.06, 0.34, 0.24, 0.46, w=W_SEC)
    model_card(c, 0.38, 0.16, 0.24, 0.64, w=W_MAIN)
    model_card(c, 0.70, 0.34, 0.24, 0.46, w=W_SEC)
    shelf(c, 0.03, 0.82, 0.94, 0.040)


def model_zoo_02(c):
    """A model downloading out of the cloud onto the local shelf."""
    cloud(c, 0.50, 0.22, 0.76, 0.34)
    c.arrow(0.50, 0.44, 0.50, 0.62, W_MAIN, head=0.075)
    model_card(c, 0.34, 0.64, 0.32, 0.22, w=W_SEC, bar_=False)
    shelf(c, 0.14, 0.90, 0.72, 0.040)


def model_zoo_03(c):
    """A model card stamped verified after its checksum matched."""
    model_card(c, 0.10, 0.10, 0.60, 0.70, w=W_MAIN)
    c.disc(0.72, 0.74, 0.190)
    c.p.setCompositionMode(QPainter.CompositionMode_Clear)
    tick(c, 0.72, 0.74, 0.105, W_MAIN * 1.5)
    c.p.setCompositionMode(QPainter.CompositionMode_SourceOver)


def model_zoo_04(c):
    """A leaderboard: three models ranked by how well they benched."""
    for j, (y, ln) in enumerate(((0.18, 0.72), (0.46, 0.52), (0.74, 0.34))):
        netglyph(c, 0.14, y + 0.08, 0.075)
        c.bar(0.28, y, ln, 0.16, r=0.045, filled=True)
    c.line(0.26, 0.08, 0.26, 0.94, W_SEC)


def model_zoo_05(c):
    """A wall of model tiles, each a different kind of model."""
    for i, x in enumerate((0.08, 0.39, 0.70)):
        for j, y in enumerate((0.10, 0.55)):
            c.rect(x, y, 0.22, 0.35, w=W_SEC, r=0.03)
            k = (i + 2 * j) % 3
            if k == 0:
                netglyph(c, x + 0.11, y + 0.175, 0.062)
            elif k == 1:
                c.disc(x + 0.11, y + 0.175, 0.070)
            else:
                blob(c, x + 0.11, y + 0.175, 0.082, w=W_FINE, phase=j)


def model_zoo_06(c):
    """One model from the shelf tried out on three of your own fields."""
    model_card(c, 0.06, 0.30, 0.30, 0.44, w=W_MAIN)
    for j, y in enumerate((0.12, 0.42, 0.72)):
        c.arrow(0.38, 0.52, 0.54, y + 0.08, W_FINE, head=0.046)
        c.rect(0.58, y, 0.34, 0.17, w=W_SEC, r=0.025)
        blob(c, 0.75, y + 0.085, 0.062, w=W_FINE, phase=j)


def model_zoo_07(c):
    """A model being slotted into an empty bay."""
    c.rect(0.06, 0.52, 0.88, 0.42, w=W_MAIN, r=0.035)
    dashed_rect(c, 0.16, 0.62, 0.32, 0.22, W_SEC)
    c.rect(0.56, 0.62, 0.32, 0.22, w=W_SEC, r=0.02)
    netglyph(c, 0.72, 0.73, 0.062)
    model_card(c, 0.20, 0.04, 0.30, 0.30, w=W_SEC, bar_=False)
    c.arrow(0.34, 0.38, 0.34, 0.58, W_SEC, head=0.062)


def model_zoo_08(c):
    """Two shelves: the segmentation models above, the classifiers below."""
    for i, x in enumerate((0.06, 0.38, 0.70)):
        c.rect(x, 0.06, 0.24, 0.30, w=W_SEC, r=0.03)
        blob(c, x + 0.12, 0.21, 0.082, w=W_FINE, phase=i * 3)
    shelf(c, 0.03, 0.39, 0.94, 0.036)
    for i, x in enumerate((0.06, 0.38, 0.70)):
        c.rect(x, 0.54, 0.24, 0.30, w=W_SEC, r=0.03)
        netglyph(c, x + 0.12, 0.69, 0.068)
    shelf(c, 0.03, 0.87, 0.94, 0.036)


def model_zoo_09(c):
    """A model card timed on the bench by a stopwatch."""
    model_card(c, 0.06, 0.22, 0.42, 0.56, w=W_MAIN)
    c.circ(0.72, 0.58, 0.220, W_MAIN)
    c.line(0.72, 0.58, 0.72, 0.42, W_SEC)
    c.line(0.72, 0.58, 0.84, 0.66, W_FINE)
    c.bar(0.655, 0.28, 0.13, 0.062, r=0.030, filled=True)


def model_zoo_10(c):
    """Models fanned out like a hand of cards, one lifted to be chosen."""
    for rot, dx in ((-22.0, -0.20), (0.0, 0.0), (22.0, 0.20)):
        c.p.save()
        c.p.translate((0.50 + dx) * c.n, 0.92 * c.n)
        c.p.rotate(rot)
        c.p.translate(-0.50 * c.n, -0.92 * c.n)
        c.rect(0.36, 0.42, 0.28, 0.48, w=W_SEC, r=0.03)
        c.p.restore()
    model_card(c, 0.34, 0.04, 0.32, 0.32, w=W_MAIN, bar_=False)
    c.arrow(0.50, 0.42, 0.50, 0.38, W_FINE, head=0.048, tail=False)


# =====================================================================
# train_compare -- several training CURVES overlaid on one axis
# =====================================================================

def train_compare_01(c):
    """Three losses falling together on one axis, one settling lower."""
    c.axes(0.12, 0.08, 0.94, 0.86, w=W_SEC)
    loss_curve(c, 0.16, 0.92, 0.14, 0.80, k=1.4, floor=0.30, w=W_SEC,
               dash=[6, 9], start=1.00)
    loss_curve(c, 0.16, 0.92, 0.14, 0.80, k=2.2, floor=0.14, w=W_MAIN,
               start=0.80)
    loss_curve(c, 0.16, 0.92, 0.14, 0.80, k=3.4, floor=0.03, w=W_SEC,
               dash=[13, 12], start=0.60)


def train_compare_02(c):
    """Loss falling and accuracy rising on the same frame."""
    c.axes(0.12, 0.08, 0.94, 0.86, w=W_SEC)
    loss_curve(c, 0.16, 0.92, 0.14, 0.80, k=3.0, floor=0.06, w=W_MAIN)
    loss_curve(c, 0.16, 0.92, 0.14, 0.80, k=2.6, floor=0.08, w=W_MAIN,
               rising=True)
    c.disc(0.92, 0.80 - (0.80 - 0.14) * 0.10, 0.040)
    c.disc(0.92, 0.80 - (0.80 - 0.14) * 0.92, 0.040)


def train_compare_03(c):
    """Two runs' curves, and beside them the setting that differed."""
    c.axes(0.08, 0.08, 0.58, 0.84, w=W_SEC)
    loss_curve(c, 0.12, 0.56, 0.14, 0.78, k=3.2, floor=0.06, w=W_MAIN)
    loss_curve(c, 0.12, 0.56, 0.14, 0.78, k=1.6, floor=0.30, w=W_SEC,
               dash=[11, 11], start=0.80)
    c.rect(0.66, 0.14, 0.30, 0.62, w=W_SEC, r=0.03)
    for j, y in enumerate((0.28, 0.44, 0.60)):
        c.line(0.71, y, 0.91, y, W_FINE)
    c.line(0.69, 0.44, 0.93, 0.44, W_MAIN)


def train_compare_04(c):
    """A cursor dropped at one epoch, reading every run's loss there."""
    c.axes(0.12, 0.08, 0.94, 0.86, w=W_SEC)
    loss_curve(c, 0.16, 0.92, 0.14, 0.80, k=3.2, floor=0.05, w=W_MAIN)
    loss_curve(c, 0.16, 0.92, 0.14, 0.80, k=1.7, floor=0.26, w=W_SEC,
               dash=[11, 11], start=0.82)
    xc = 0.54
    c.line(xc, 0.06, xc, 0.86, W_SEC, dash=[7, 8])
    for k, floor, start in ((3.2, 0.05, 1.0), (1.7, 0.26, 0.82)):
        t = (xc - 0.16) / 0.76
        v = math.exp(-k * t) * (start - floor) + floor
        c.disc(xc, 0.80 - (0.80 - 0.14) * v, 0.055)


def train_compare_05(c):
    """The gap between two runs' curves, filled in to show how far apart."""
    c.axes(0.12, 0.08, 0.94, 0.86, w=W_SEC)
    n = 13
    top, bot = [], []
    for i in range(n):
        t = i / (n - 1.0)
        x = 0.16 + 0.76 * t
        va = math.exp(-3.4 * t) * 0.94 + 0.06
        vb = math.exp(-1.5 * t) * 0.70 + 0.30
        top.append((x, 0.80 - 0.66 * vb))
        bot.append((x, 0.80 - 0.66 * va))
    c.polyline(top + bot[::-1], close=True, filled=True)
    c.smooth(top, w=W_SEC)
    c.smooth(bot, w=W_SEC)


def train_compare_06(c):
    """One run stopped early at its best epoch while the other ran on."""
    c.axes(0.12, 0.08, 0.94, 0.86, w=W_SEC)
    loss_curve(c, 0.16, 0.92, 0.14, 0.80, k=1.8, floor=0.22, w=W_SEC,
               dash=[11, 11], start=0.82)
    pts = []
    for i in range(9):
        t = i / 8.0
        v = math.exp(-3.4 * t) * 0.9 + 0.10 + 0.30 * max(0.0, t - 0.55) ** 2
        pts.append((0.16 + 0.76 * t, 0.80 - 0.66 * v))
    c.smooth(pts, w=W_MAIN)
    fx, fy = pts[4]
    c.line(fx, fy, fx, fy - 0.26, W_SEC)
    c.polyline([(fx, fy - 0.26), (fx + 0.17, fy - 0.20), (fx, fy - 0.14)],
               close=True, filled=True)


def train_compare_07(c):
    """A legend of three runs, tied to the three traces beside it."""
    c.axes(0.34, 0.08, 0.94, 0.86, w=W_SEC)
    loss_curve(c, 0.38, 0.92, 0.14, 0.78, k=1.2, floor=0.40, w=W_SEC,
               dash=[6, 9], start=1.00)
    loss_curve(c, 0.38, 0.92, 0.14, 0.78, k=2.0, floor=0.22, w=W_MAIN,
               start=0.80)
    loss_curve(c, 0.38, 0.92, 0.14, 0.78, k=3.2, floor=0.06, w=W_SEC,
               dash=[13, 12], start=0.60)
    c.line(0.05, 0.22, 0.26, 0.22, W_MAIN)
    c.line(0.05, 0.46, 0.26, 0.46, W_SEC, dash=[13, 12])
    c.line(0.05, 0.70, 0.26, 0.70, W_SEC, dash=[6, 9])


def train_compare_08(c):
    """Overlaid curves above, and the score each run finished on."""
    c.rect(0.06, 0.06, 0.88, 0.52, w=W_SEC, r=0.03)
    loss_curve(c, 0.12, 0.88, 0.14, 0.50, k=3.2, floor=0.06, w=W_MAIN)
    loss_curve(c, 0.12, 0.88, 0.14, 0.50, k=1.6, floor=0.30, w=W_SEC,
               dash=[11, 11], start=0.80)
    c.bar(0.14, 0.66, 0.56, 0.10, r=0.05, filled=True)
    c.bar(0.14, 0.82, 0.34, 0.10, r=0.05, filled=True)


def train_compare_09(c):
    """A whole sweep of runs fanning apart from the same starting loss."""
    c.axes(0.12, 0.08, 0.94, 0.86, w=W_SEC)
    for k, floor, w, st in ((3.8, 0.03, W_SEC, 1.00), (2.6, 0.14, W_MAIN, 0.94),
                            (1.7, 0.30, W_SEC, 0.88), (1.0, 0.48, W_SEC, 0.82)):
        loss_curve(c, 0.16, 0.92, 0.16, 0.80, k=k, floor=floor, w=w, start=st)


def train_compare_10(c):
    """Two runs whose curves cross, the crossing point ringed."""
    c.axes(0.12, 0.08, 0.94, 0.86, w=W_SEC)
    a = []
    b = []
    for i in range(11):
        t = i / 10.0
        va = 0.40 + 0.50 * math.exp(-8.0 * t)
        vb = 0.10 + 0.90 * math.exp(-2.6 * t)
        a.append((0.16 + 0.76 * t, 0.80 - 0.66 * va))
        b.append((0.16 + 0.76 * t, 0.80 - 0.66 * vb))
    c.smooth(a, w=W_MAIN)
    c.smooth(b, w=W_SEC, dash=[13, 12])
    c.circ(0.464, 0.523, 0.115, W_MAIN)


# =====================================================================
# profiler -- ONE input moved, ONE prediction responding
# =====================================================================

def profiler_01(c):
    """A response curve with the cursor riding on it above its slider."""
    px, py = pd_at(0.12, 0.92, 0.62, 0.12, 0.62)
    pd_curve(c, 0.12, 0.92, 0.62, 0.12, w=W_MAIN)
    c.line(px, py, px, 0.80, W_FINE, dash=[6, 7])
    c.disc(px, py, 0.062)
    slider(c, 0.12, 0.92, 0.86, px, w=W_SEC, hr=0.058)


def profiler_02(c):
    """One slider pushed along, and the output bar that grew with it."""
    slider(c, 0.06, 0.56, 0.28, 0.42, w=W_MAIN, hr=0.085)
    c.arrow(0.18, 0.10, 0.44, 0.10, W_SEC, head=0.060)
    c.bar(0.70, 0.18, 0.24, 0.74, r=0.075, filled=False, w=W_SEC)
    c.bar(0.727, 0.42, 0.186, 0.473, r=0.055, filled=True)
    c.arrow(0.62, 0.64, 0.62, 0.42, W_SEC, head=0.060)


def profiler_03(c):
    """A knob turned one way, the needle on the readout following it."""
    c.circ(0.24, 0.62, 0.190, W_MAIN)
    c.line(0.24, 0.62, 0.35, 0.48, W_MAIN)
    c.arc(0.24, 0.62, 0.280, 55, 100, W_SEC)
    c.arrow(0.18, 0.32, 0.32, 0.30, W_SEC, head=0.062, tail=False)
    gauge(c, 0.72, 0.80, 0.240, 58.0)


def profiler_04(c):
    """Three inputs held still, one moved off centre, one prediction out."""
    for j, y in enumerate((0.16, 0.40, 0.64)):
        hx = 0.44 if j == 1 else 0.24
        slider(c, 0.06, 0.60, y, hx, w=W_SEC, hr=0.055 if j != 1 else 0.075)
    c.arrow(0.68, 0.40, 0.84, 0.40, W_SEC, head=0.058)
    c.bar(0.30, 0.84, 0.52, 0.13, r=0.065, filled=False, w=W_SEC)
    c.bar(0.325, 0.865, 0.30, 0.08, r=0.04, filled=True)


def profiler_05(c):
    """The local slope of the response, read off at the cursor."""
    px, py = pd_at(0.10, 0.92, 0.78, 0.14, 0.50)
    pd_curve(c, 0.10, 0.92, 0.78, 0.14, w=W_MAIN)
    c.line(px - 0.24, py + 0.185, px + 0.24, py - 0.185, W_SEC)
    c.disc(px, py, 0.070)
    c.line(0.10, 0.90, 0.92, 0.90, W_FINE)
    c.disc(px, 0.90, 0.045)


def profiler_06(c):
    """An input pushed into the fitted model, a prediction coming out."""
    slider(c, 0.03, 0.24, 0.34, 0.17, w=W_SEC, hr=0.058)
    c.rect(0.36, 0.12, 0.30, 0.44, w=W_MAIN, r=0.04)
    netglyph(c, 0.51, 0.34, 0.098, W_SEC)
    c.arrow(0.26, 0.34, 0.34, 0.34, W_SEC, head=0.050)
    c.arrow(0.68, 0.42, 0.78, 0.54, W_SEC, head=0.058)
    gauge(c, 0.72, 0.92, 0.240, 55.0)


def profiler_07(c):
    """The cursor moved from here to there, and the step in the prediction."""
    ax, ay = pd_at(0.10, 0.90, 0.76, 0.16, 0.32)
    bx, by = pd_at(0.10, 0.90, 0.76, 0.16, 0.70)
    pd_curve(c, 0.10, 0.90, 0.76, 0.16, w=W_MAIN)
    c.circ(ax, ay, 0.060, W_SEC)
    c.disc(bx, by, 0.070)
    c.line(0.10, ay, ax, ay, W_FINE, dash=[6, 7])
    c.line(0.10, by, bx, by, W_FINE, dash=[6, 7])
    c.arrow(0.10, ay, 0.10, by + 0.03, W_SEC, head=0.060)


def profiler_08(c):
    """The output column rising as the one handle is dragged right."""
    c.bar(0.62, 0.10, 0.26, 0.62, r=0.07, filled=False, w=W_SEC)
    c.bar(0.645, 0.34, 0.21, 0.355, r=0.055, filled=True)
    c.line(0.62, 0.34, 0.36, 0.34, W_FINE, dash=[5, 6])
    c.line(0.62, 0.60, 0.36, 0.60, W_FINE, dash=[5, 6])
    slider(c, 0.06, 0.56, 0.88, 0.40, w=W_MAIN, hr=0.085)
    c.circ(0.18, 0.88, 0.062, W_FINE, dash=[4, 5])


def profiler_09(c):
    """A lever pressed at one end, the readout swinging at the other."""
    c.polyline([(0.08, 0.74), (0.92, 0.52)], w=W_MAIN)
    c.polyline([(0.46, 0.68), (0.62, 0.94), (0.30, 0.94)], w=W_SEC,
               close=True)
    c.disc(0.10, 0.735, 0.078)
    c.arrow(0.10, 0.42, 0.10, 0.62, W_SEC, head=0.065)
    c.line(0.90, 0.50, 0.90, 0.30, W_SEC)
    c.bar(0.66, 0.12, 0.30, 0.16, r=0.08, filled=False, w=W_SEC)
    c.bar(0.685, 0.145, 0.18, 0.11, r=0.055, filled=True)


def profiler_10(c):
    """One feature in the input row overwritten, and the prediction shifting."""
    for i, x in enumerate((0.06, 0.28, 0.50)):
        c.rect(x, 0.34, 0.18, 0.26, w=W_SEC if i != 1 else W_MAIN, r=0.03)
    c.arrow(0.37, 0.28, 0.37, 0.12, W_SEC, head=0.060)
    c.arrow(0.72, 0.47, 0.84, 0.47, W_SEC, head=0.055)
    c.bar(0.66, 0.70, 0.30, 0.12, r=0.06, filled=True)
    c.bar(0.66, 0.86, 0.16, 0.12, r=0.06, filled=False, w=W_FINE)


# =====================================================================
# manifest
# =====================================================================

GROUPS = {
    "distributed_jobs": ("distributed_jobs -- WHERE the work runs: remote "
                         "machines, a scheduler, a queue fanning out", [
        ("A head node fanning the work out to three worker machines.",
         distributed_jobs_01),
        ("A rack cabinet of three blades, the top two running.",
         distributed_jobs_02),
        ("A job card lifting off into a cloud.", distributed_jobs_03),
        ("A terminal prompt wired down a cable to a machine somewhere else.",
         distributed_jobs_04),
        ("A scheduler in the middle with worker nodes spoked around it.",
         distributed_jobs_05),
        ("A scheduler allocation chart: jobs booked across node rows and time.",
         distributed_jobs_06),
        ("A laptop pushing a run down a long cable to a far bigger machine.",
         distributed_jobs_07),
        ("One submitted job splitting into three parallel lanes of work.",
         distributed_jobs_08),
        ("A remote machine reporting a live progress bar back home.",
         distributed_jobs_09),
        ("A submit button firing a paper plane at a queue of remote machines.",
         distributed_jobs_10),
    ]),
    "run_history": ("run_history -- the ARCHIVE of past runs, and searching it",
                    [
        ("A magnifier held over a deep stack of finished run records.",
         run_history_01),
        ("A timeline of past runs, each marked passed or failed.",
         run_history_02),
        ("A search field over the run rows it matched.", run_history_03),
        ("A drawer of run folders with one pulled up out of the file.",
         run_history_04),
        ("A clock wound backwards over the list of runs already done.",
         run_history_05),
        ("A ledger of past runs with a pass/fail column down the right.",
         run_history_06),
        ("A calendar month with the days that were run marked off.",
         run_history_07),
        ("A run log unrolled, with the warning line flagged in it.",
         run_history_08),
        ("How long every past run took, plotted run after run.",
         run_history_09),
        ("Boxed-up past runs on the shelf, the newest one still open.",
         run_history_10),
    ]),
    "run_compare": ("run_compare -- two whole RUNS diffed: records, settings "
                    "and counts", [
        ("Two run records side by side, the one row that differs flagged in "
         "both.", run_compare_01),
        ("A diff gutter: rows removed on the left, added on the right.",
         run_compare_02),
        ("The same three counts from two runs, plotted at different heights.",
         run_compare_03),
        ("Two hit lists as overlapping sets: shared hits, and hits unique to "
         "one.", run_compare_04),
        ("A balance with a run record in each pan, tipped to the better one.",
         run_compare_05),
        ("Two run cards swapped back and forth against each other.",
         run_compare_06),
        ("A delta between one run stacked over the other.", run_compare_07),
        ("Two ranked hit lists, with one hit that jumped up the order.",
         run_compare_08),
        ("One settings sheet cut down the middle: before on the left, after "
         "on the right.", run_compare_09),
        ("Two records checked line for line: three agree, one does not.",
         run_compare_10),
    ]),
    "pipeline_graph": ("pipeline_graph -- a DAG of artefacts, with stale and "
                       "missing nodes marked", [
        ("A three-step chain whose last artefact was never produced.",
         pipeline_graph_01),
        ("One source forking to two products, one of them flagged bad.",
         pipeline_graph_02),
        ("Two branches converging on a join that is now out of date.",
         pipeline_graph_03),
        ("File to file: two artefacts made, the third one still blank.",
         pipeline_graph_04),
        ("The link between two steps snapped in half.", pipeline_graph_05),
        ("Along one branch: made, gone stale, never made at all.",
         pipeline_graph_06),
        ("A node marked for rebuild, with everything below it waiting on it.",
         pipeline_graph_07),
        ("A node carrying a clock: its inputs moved on without it.",
         pipeline_graph_08),
        ("A step struck out, and everything it fed left dangling.",
         pipeline_graph_09),
        ("Provenance: one output traced back up to the two inputs that made "
         "it.", pipeline_graph_10),
    ]),
    "model_compare": ("model_compare -- two SEGMENTATIONS of the SAME field, "
                      "and where they disagree", [
        ("One cell wearing two rival contours, solid and broken, that "
         "disagree.", model_compare_01),
        ("The same field wiped down the middle: one model's outlines each "
         "side.", model_compare_02),
        ("Two touching cells: one model calls them one object, the other two.",
         model_compare_03),
        ("Same three cells, but one model found a fourth: the extra one "
         "ringed.", model_compare_04),
        ("The sliver where the two outlines disagree, filled in solid.",
         model_compare_05),
        ("One cell, a tight outline and a loose one, the gap between "
         "measured.", model_compare_06),
        ("A cell that one model outlined and the other missed entirely.",
         model_compare_07),
        ("The same cells outlined twice, and the two object counts beneath.",
         model_compare_08),
        ("Objects matched up one to one between the two labellings.",
         model_compare_09),
        ("A checkerboard of the same field, alternating whose outline is "
         "drawn.", model_compare_10),
    ]),
    "model_zoo": ("model_zoo -- a CATALOGUE of models to browse, fetch and "
                  "bench", [
        ("A shelf of model cards with the middle one pulled out.",
         model_zoo_01),
        ("A model downloading out of the cloud onto the local shelf.",
         model_zoo_02),
        ("A model card stamped verified after its checksum matched.",
         model_zoo_03),
        ("A leaderboard: three models ranked by how well they benched.",
         model_zoo_04),
        ("A wall of model tiles, each a different kind of model.",
         model_zoo_05),
        ("One model from the shelf tried out on three of your own fields.",
         model_zoo_06),
        ("A model being slotted into an empty bay.", model_zoo_07),
        ("Two shelves: the segmentation models above, the classifiers below.",
         model_zoo_08),
        ("A model card timed on the bench by a stopwatch.", model_zoo_09),
        ("Models fanned out like a hand of cards, one lifted to be chosen.",
         model_zoo_10),
    ]),
    "train_compare": ("train_compare -- several training CURVES overlaid on "
                      "one axis, plus the settings diff", [
        ("Three losses falling together on one axis, one settling lower.",
         train_compare_01),
        ("Loss falling and accuracy rising on the same frame.",
         train_compare_02),
        ("Two runs' curves, and beside them the setting that differed.",
         train_compare_03),
        ("A cursor dropped at one epoch, reading every run's loss there.",
         train_compare_04),
        ("The gap between two runs' curves, filled in to show how far apart.",
         train_compare_05),
        ("One run stopped early at its best epoch while the other ran on.",
         train_compare_06),
        ("A legend of three runs, tied to the three traces beside it.",
         train_compare_07),
        ("Overlaid curves above, and the score each run finished on.",
         train_compare_08),
        ("A whole sweep of runs fanning apart from the same starting loss.",
         train_compare_09),
        ("Two runs whose curves cross, the crossing point ringed.",
         train_compare_10),
    ]),
    "profiler": ("profiler -- ONE input moved, ONE prediction responding", [
        ("A response curve with the cursor riding on it above its slider.",
         profiler_01),
        ("One slider pushed along, and the output bar that grew with it.",
         profiler_02),
        ("A knob turned one way, the needle on the readout following it.",
         profiler_03),
        ("Three inputs held still, one moved off centre, one prediction out.",
         profiler_04),
        ("The local slope of the response, read off at the cursor.",
         profiler_05),
        ("An input pushed into the fitted model, a prediction coming out.",
         profiler_06),
        ("The cursor moved from here to there, and the step in the "
         "prediction.", profiler_07),
        ("The output column rising as the one handle is dragged right.",
         profiler_08),
        ("A lever pressed at one end, the readout swinging at the other.",
         profiler_09),
        ("One feature in the input row overwritten, and the prediction "
         "shifting.", profiler_10),
    ]),
}


def main(outdir):
    return emit_groups(outdir, GROUPS, "group_jobs_models.py")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else
                  default_outdir(__file__)))
