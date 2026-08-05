#!/usr/bin/env python3
"""Candidate spaCR icons: the four remaining registered apps with no folder.

``trellis`` (shown to users as **Small Multiples**), ``gate_editor``,
``feature_explorer`` and ``napari_bridge`` landed after the drawing list was
derived, so nobody ever drew for them.  Ten conceptually different designs
each, white-on-transparent flat vector art in the house style set by
``plaque.png`` / ``measure.png``.

Three of these four are, left to themselves, "a chart with something on it",
and the registry is already full of those: ``graph_builder``, ``power``,
``regression``, ``classifier_evaluation``, ``hit_list``, ``image_scatter``,
``plate_view``, ``qc_dashboard``, ``barcode_qc``, plus ``pca``, ``tabulate``,
``feature_dict``, ``outliers`` and ``dose_response`` being drawn in parallel.
Every candidate below is held to one line and none is allowed to wander::

    trellis         THE REPETITION IS THE SUBJECT.  Never one chart.  Every
                    candidate is a grid of frames that are DELIBERATELY THE
                    SAME -- same kind of plot, same axes, same shape at a
                    different height -- because the sameness of the frames is
                    what small multiples are.  ``qc_dashboard_04`` is already
                    "one frame holding four UNLIKE mini-panels and a verdict
                    badge", so nothing here mixes panel kinds and nothing
                    here carries a verdict, a tick or a badge.  Panels are
                    rectangular framed plots, never rounded wells inside a
                    plate outline (``plate_view``, ``experiment_design``).
                    And ``graph_builder`` owns the whole drag-a-column-onto-a
                    -shelf vocabulary -- ``graph_builder_06`` is literally "a
                    column dropped on the facet shelf splits one chart into
                    four" -- so there is no chip, no shelf, no drop target
                    and no split-in-two ACTION anywhere in this set.  The
                    grid here already exists; what is drawn is what makes it
                    readable (the shared axis, the facet headers, the common
                    rule, the printed n, the empty panel that is still drawn).

    gate_editor     A BOUNDARY DRAWN BY HAND, AND THE POPULATION IT KEEPS.
                    Always a human gesture on the picture: a cursor, a nib,
                    square vertex handles, a drag handle, a half-closed
                    polygon.  That is the line against every automatic split
                    in the registry -- ``mask_08`` (a threshold falling out
                    of a histogram by itself), ``outliers`` (fences ESTIMATED
                    from the crowd) and ``classifier_evaluation_09`` (a
                    boundary LEARNED from labels).  It is also the line
                    against ``image_scatter_06``, "a lasso thrown round a few
                    points and the crop of one of them": no gate here ever
                    opens an image crop, because a gate is a predicate and
                    not a set of picked objects -- so the gates get named,
                    nested, re-applied to the next plate and re-drawn over
                    their older selves, which a lasso cannot do.  Where a
                    histogram appears it holds ONE distribution with a
                    hand-placed sweep across it, never two humps either side
                    of a decision point (``classifier_evaluation_05``).  And
                    the hand-drawn stroke always lands on a CHART: tracing an
                    outline round a cell is ``annotate_10`` and pushing a
                    mask boundary back into place is ``curate_01``.

    feature_explorer   MANY FEATURES RANKED BY HOW WELL THEY SEPARATE TWO
                    CLASSES.  Never one plot: the ranked list of columns is
                    the subject and the distributions are what the score
                    means.  Every ranked row therefore carries a TWO-CLASS
                    glyph -- a pair of humps, a pair of dot rows, a direction
                    -- and never a bare sorted bar with the leaders starred,
                    which is ``hit_list_01`` and ``hit_list_10``: the rows
                    here are COLUMNS OF THE TABLE, nothing is called a hit,
                    and there is no FDR line and no volcano.  The statistic
                    is AUC, so the one picture this module may never draw is
                    an ROC curve -- ``classifier_evaluation_01`` owns "a
                    curve bulging away from the chance diagonal", and an AUC
                    appears here only ever as a score in a ranking or as a
                    position on a bounded 0.5..1 scale.  The columns are the
                    measured ones, unchanged, which is the line against
                    ``pca``; they hold values rather than meanings, which is
                    the line against ``feature_dict``.

    napari_bridge   THE HANDOFF -- and, for half the set, napari's own mark.
                    Split deliberately, see the note below.  The original
                    half is always TWO PLACES AND TRAFFIC BETWEEN THEM: two
                    panes side by side with something crossing.  That is the
                    line against ``layer_viewer``, which owns sheets stacked
                    in register with an eye on each, and against
                    ``foreign_03``, which already owns the plug-and-socket
                    picture for a third-party format.  A correction coming
                    back is drawn as a CROSSING that is checked, never as a
                    correction being written into a ledger line, which is
                    ``curate_09``.

    (``control_chart`` is a registered screen with no candidate folder either
    and is not drawn here, but it is held against: it owns "a series along
    run order between limit lines with the violating point flagged".  No
    trellis panel is a run chart and no gate is a control limit.)

napari's mark, and why five of the ten carry it
-----------------------------------------------

The user asked for "a black and white version of the napari logo".  napari's
mark is **napari's trademark, not spaCR's**.  Using it to label a bridge *to*
napari is ordinary nominative use and is what most integrations do, but it is
a decision the user should make with the alternative in front of them, so both
kinds are drawn:

* ``napari_bridge_01`` .. ``05`` evoke **napari's own visual identity** -- a
  monochrome four-petal rosette in the manner of their mark.  Choosing one of
  these means shipping a monochrome derivative of a third-party trademark.
* ``napari_bridge_06`` .. ``10`` are **original spaCR marks about the
  handoff** and carry no third-party mark at all.

Nothing was copied: the rosette is built from the same ``_draw`` primitives as
everything else in this directory, and no napari image file exists in this
repository.

48 px is the real constraint -- that is the size the tile is drawn at, and it
is what discarded the first pass of several of these.  So: grids are at most
3x3 and a panel's content is ONE solid element rather than a little chart;
distributions are solid filled humps rather than outlined curves; dot clouds
are five or six discs rather than a spray; a "word" is a filled capsule bar;
and strokes stay at ``W_MAIN``/``W_SEC`` because ``W_FINE`` is half a pixel at
48 px.

Run standalone (deterministic -- no random draws at all)::

    QT_QPA_PLATFORM=offscreen python group_trellis_gate_feature_napari.py [OUTDIR]

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

DASH = [1.4, 1.6]
DASH_L = [2.0, 2.0]


# ---------------------------------------------------------------------------
# shared sub-drawings
# ---------------------------------------------------------------------------

def dots(c, pts, r=0.030):
    """A handful of solid points -- a 'population', never a spray."""
    for x, y in pts:
        c.disc(x, y, r)


def rings(c, pts, r=0.028, w=W_SEC):
    """The same population, hollow: the objects a gate did not keep."""
    for x, y in pts:
        c.circ(x, y, r, w)


def textline(c, x, y, w, h=0.034):
    """A capsule bar standing in for a word (survives 48 px)."""
    c.bar(x, y - h / 2.0, w, h, filled=True)


def cursor(c, x, y, s=0.085, w=W_MAIN):
    """A pointer arrowhead: a human is doing this, not the software."""
    c.polyline([(x, y), (x + s * 0.30, y + s * 1.00), (x + s * 0.46, y + s * 0.52),
                (x + s * 0.92, y + s * 0.40)], close=True, filled=True)


def nib(c, x, y, s=0.13, ang=-135.0, w=W_MAIN):
    """A pen nib, tip exactly at (x, y): the boundary is being drawn by hand."""
    a = math.radians(ang)
    px, py = -math.sin(a), math.cos(a)
    bx, by = x - s * math.cos(a), y - s * math.sin(a)
    c.polyline([(x, y), (bx + px * s * 0.26, by + py * s * 0.26),
                (bx - px * s * 0.26, by - py * s * 0.26)], close=True, filled=True)
    c.line(bx, by, bx - s * 0.72 * math.cos(a), by - s * 0.72 * math.sin(a), W_MAIN)


def handles(c, pts, s=0.030, w=W_SEC):
    """Square grab handles on a drawn shape's vertices."""
    for x, y in pts:
        c.rect(x - s, y - s, 2 * s, 2 * s, filled=True)


def hump(c, cx, ybase, w, h, filled=True, wd=W_SEC, spread=2.9, n=25):
    """A solid distribution hump sitting on a baseline (reads at 48 px)."""
    pts = []
    for i in range(n):
        t = i / (n - 1.0)
        pts.append((cx - w / 2 + w * t,
                    ybase - h * math.exp(-((t - 0.5) * spread) ** 2)))
    if filled:
        c.polyline(pts + [(cx + w / 2, ybase), (cx - w / 2, ybase)],
                   close=True, filled=True)
    else:
        c.polyline(pts, w=wd)


def frames(c, x, y, w, h, rows, cols, gap=0.030, wd=W_SEC):
    """A grid of IDENTICAL chart frames.  Returns their geometry."""
    cw = (w - gap * (cols - 1)) / cols
    ch = (h - gap * (rows - 1)) / rows
    out = []
    for j in range(rows):
        for i in range(cols):
            px, py = x + i * (cw + gap), y + j * (ch + gap)
            c.rect(px, py, cw, ch, w=wd)
            out.append((px, py, cw, ch))
    return out


def panel_bar(c, px, py, cw, ch, frac, inset=0.26):
    """One solid element per panel: the same mark at a different height."""
    bw = cw * (1.0 - 2 * inset)
    h = (ch - ch * 0.16) * frac
    c.rect(px + cw * inset, py + ch - ch * 0.08 - h, bw, h, filled=True)


def panel_wedge(c, px, py, cw, ch, frac):
    """The same rising shape in every panel, at a different level."""
    y0 = py + ch * 0.90
    top = py + ch * 0.90 - (ch * 0.74) * frac
    c.polyline([(px + cw * 0.16, y0), (px + cw * 0.50, (y0 + top) / 2.0),
                (px + cw * 0.84, top)], w=W_MAIN)


def caliper(c, x1, y1, x2, y2, t=0.040, w=W_SEC):
    """A measured span: a line with a perpendicular bar at each end."""
    ang = math.atan2(y2 - y1, x2 - x1)
    px, py = -math.sin(ang) * t, math.cos(ang) * t
    c.line(x1, y1, x2, y2, w)
    c.line(x1 - px, y1 - py, x1 + px, y1 + py, w)
    c.line(x2 - px, y2 - py, x2 + px, y2 + py, w)


def blob(c, cx, cy, r, rot=0.0, k=(1.00, 0.86, 1.10, 0.80, 1.04, 0.90),
         filled=False, w=W_MAIN, dash=None):
    """A hand-drawn closed boundary: irregular, obviously not a fitted ellipse."""
    pts = []
    for i, f in enumerate(k):
        a = math.radians(rot) + 2 * math.pi * i / len(k)
        pts.append((cx + r * f * math.cos(a), cy + r * f * 0.88 * math.sin(a)))
    c.smooth(pts, w=w, closed=True, filled=filled, dash=dash)


def cross(c, cx, cy, s, w=W_MAIN):
    c.line(cx - s, cy - s, cx + s, cy + s, w)
    c.line(cx + s, cy - s, cx - s, cy + s, w)


def petal(c, cx, cy, ang, d0, L, half, filled=True, w=W_MAIN, n=22, swirl=0.0):
    """One rosette petal: a leaf, pointed at the hub, swept round by ``swirl``.

    ``swirl`` (radians of turn over the petal's length) is what makes the four
    of them read as a pinwheel rather than a clover at 48 px.
    """
    def prof(t):
        return (t ** 0.60) * ((1.0 - t) ** 0.62)

    norm = max(prof(i / 200.0) for i in range(1, 200))
    loc = []
    for sgn in (1.0, -1.0):
        rng = range(n + 1) if sgn > 0 else range(n, -1, -1)
        for i in rng:
            t = i / float(n)
            loc.append((t, sgn * half * prof(t) / norm))
    pts = []
    for t, v in loc:
        a = ang + swirl * t
        d = d0 + L * t
        pts.append((cx + d * math.cos(a) - v * math.sin(a),
                    cy + d * math.sin(a) + v * math.cos(a)))
    if filled:
        c.polyline(pts, close=True, filled=True)
    else:
        c.polyline(pts, close=True, w=w)


def rosette(c, cx, cy, R, filled=True, w=W_MAIN, n=4, a0=-72.0, gap=0.14,
            swirl=0.34):
    """A four-petal pinwheel rosette in the manner of napari's mark.

    napari's mark is napari's trademark; this is a monochrome evocation of it
    built from the same primitives as every other icon here, not a copy of any
    napari file.
    """
    for k in range(n):
        a = math.radians(a0) + 2 * math.pi * k / n
        petal(c, cx, cy, a, R * gap, R * (1.0 - gap), R * 0.40,
              filled=filled, w=w, swirl=swirl)


def window(c, x, y, w_, h, wd=W_SEC, bar=0.16):
    """Another application's window: a frame with a title band."""
    c.rect(x, y, w_, h, w=wd, r=0.024)
    c.line(x, y + h * bar, x + w_, y + h * bar, wd)
    c.disc(x + w_ * 0.09, y + h * bar * 0.52, min(w_, h) * 0.045)
    c.disc(x + w_ * 0.20, y + h * bar * 0.52, min(w_, h) * 0.045)


def field(c, x, y, w_, h, wd=W_SEC, n=3):
    """A spaCR field of view: a frame with a couple of cells in it."""
    c.rect(x, y, w_, h, w=wd)
    ps = ((0.32, 0.36, 0.19), (0.68, 0.62, 0.23), (0.34, 0.74, 0.14))[:n]
    for fx, fy, fr in ps:
        c.ell(x + w_ * fx, y + h * fy, w_ * fr, h * fr * 0.86, 0.0, W_SEC)


# =====================================================================
# trellis (Small Multiples) -- THE REPETITION IS THE SUBJECT
# =====================================================================

def trellis_01(c):
    """A 3x3 of identical frames, the same mark standing at a different height."""
    gs = frames(c, 0.08, 0.08, 0.84, 0.84, 3, 3, gap=0.032)
    levels = (0.78, 0.46, 0.62, 0.34, 0.86, 0.52, 0.66, 0.28, 0.72)
    for (px, py, cw, ch), f in zip(gs, levels):
        panel_bar(c, px, py, cw, ch, f)


def trellis_02(c):
    """Shared axes: one y axis and one x axis serving the whole grid."""
    c.polyline([(0.22, 0.04), (0.22, 0.84), (0.98, 0.84)], w=W_MAIN * 1.6)
    for i in range(1, 4):
        y = 0.84 - 0.76 * i / 4.0
        c.line(0.11, y, 0.22, y, W_MAIN)
    for i in range(1, 4):
        x = 0.22 + 0.72 * i / 4.0
        c.line(x, 0.84, x, 0.95, W_MAIN)
    gs = frames(c, 0.26, 0.06, 0.68, 0.74, 2, 3, gap=0.028, wd=W_SEC * 0.85)
    levels = (0.80, 0.50, 0.66, 0.36, 0.74, 0.44)
    for (px, py, cw, ch), f in zip(gs, levels):
        panel_bar(c, px, py, cw, ch, f, inset=0.22)


def trellis_03(c):
    """Two-way faceting: condition across the top, plate down the side."""
    for i in range(3):
        c.bar(0.30 + i * 0.222, 0.07, 0.166, 0.070, filled=True)
    for j in range(3):
        c.bar(0.06, 0.235 + j * 0.222, 0.166, 0.070, filled=True)
    gs = frames(c, 0.30, 0.20, 0.632, 0.632, 3, 3, gap=0.056)
    levels = (0.72, 0.44, 0.60, 0.40, 0.80, 0.50, 0.64, 0.30, 0.70)
    for (px, py, cw, ch), f in zip(gs, levels):
        panel_bar(c, px, py, cw, ch, f, inset=0.24)


def trellis_04(c):
    """One reference level ruled straight through every panel."""
    gs = frames(c, 0.08, 0.14, 0.84, 0.72, 2, 3, gap=0.032)
    levels = (0.86, 0.40, 0.62, 0.34, 0.78, 0.48)
    for (px, py, cw, ch), f in zip(gs, levels):
        panel_bar(c, px, py, cw, ch, f, inset=0.28)
    for j in range(2):
        y = 0.14 + j * (0.72 - 0.032) / 2.0 + ((0.72 - 0.032) / 2.0) * 0.42
        c.line(0.04, y, 0.96, y, W_MAIN, dash=DASH_L)


def trellis_05(c):
    """Every panel prints its n: no panel is allowed to hide how few it holds."""
    gs = frames(c, 0.06, 0.06, 0.88, 0.88, 2, 2, gap=0.048, wd=W_MAIN)
    levels = (0.66, 0.40, 0.52, 0.78)
    counts = (3, 3, 1, 3)
    for (px, py, cw, ch), f, k in zip(gs, levels, counts):
        panel_bar(c, px, py, cw, ch, f, inset=0.36)
        for i in range(k):
            c.disc(px + cw * 0.20 + i * cw * 0.23, py + ch * 0.19, cw * 0.088)


def trellis_06(c):
    """The same shape twice, comparable only because the axis is shared."""
    c.polyline([(0.16, 0.08), (0.16, 0.92), (0.98, 0.92)], w=W_MAIN * 1.6)
    for i in range(1, 5):
        y = 0.92 - 0.80 * i / 5.0
        c.line(0.05, y, 0.16, y, W_MAIN)
    c.rect(0.22, 0.10, 0.34, 0.76, w=W_SEC)
    c.rect(0.62, 0.10, 0.34, 0.76, w=W_SEC)
    panel_bar(c, 0.22, 0.10, 0.34, 0.76, 0.94, inset=0.30)
    panel_bar(c, 0.62, 0.10, 0.34, 0.76, 0.26, inset=0.30)
    c.line(0.16, 0.86 - 0.64 * 0.26, 0.98, 0.86 - 0.64 * 0.26, W_MAIN, dash=DASH_L)


def trellis_07(c):
    """A long strip of levels wrapping onto the next row."""
    gs = frames(c, 0.04, 0.08, 0.78, 0.30, 1, 3, gap=0.030)
    for (px, py, cw, ch), f in zip(gs, (0.80, 0.52, 0.66)):
        panel_bar(c, px, py, cw, ch, f, inset=0.30)
    gs2 = frames(c, 0.18, 0.62, 0.78, 0.30, 1, 3, gap=0.030)
    for (px, py, cw, ch), f in zip(gs2, (0.44, 0.72, 0.34)):
        panel_bar(c, px, py, cw, ch, f, inset=0.30)
    c.polyline([(0.84, 0.24), (0.94, 0.24), (0.94, 0.50), (0.10, 0.50),
                (0.10, 0.70)], w=W_MAIN * 1.5)
    c.arrow(0.10, 0.62, 0.10, 0.76, W_MAIN, head=0.075, tail=False)


def trellis_08(c):
    """An empty panel is still drawn: nothing measured stays a different picture."""
    gs = frames(c, 0.08, 0.08, 0.84, 0.84, 3, 3, gap=0.032)
    levels = (0.74, 0.44, 0.62, 0.36, None, 0.54, 0.68, 0.30, 0.78)
    for (px, py, cw, ch), f in zip(gs, levels):
        if f is not None:
            panel_bar(c, px, py, cw, ch, f)


def trellis_09(c):
    """A brush on one panel picks the same objects out of all the others."""
    gs = frames(c, 0.05, 0.05, 0.90, 0.90, 2, 2, gap=0.046, wd=W_SEC)
    inside = ((0.32, 0.34),)
    outside = ((0.70, 0.68),)
    for px, py, cw, ch in gs:
        for fx, fy in inside:
            c.disc(px + cw * fx, py + ch * fy, cw * 0.17)
        for fx, fy in outside:
            c.circ(px + cw * fx, py + ch * fy, cw * 0.15, W_MAIN)
    px, py, cw, ch = gs[0]
    c.rect(px + cw * 0.10, py + ch * 0.12, cw * 0.46, ch * 0.46, w=W_MAIN * 1.4)


def trellis_10(c):
    """The same chart, many times over: identical frames stepping back."""
    for k, (dx, dy) in enumerate(((0.18, -0.16), (0.09, -0.08), (0.0, 0.0))):
        c.rect(0.14 + dx, 0.30 + dy, 0.56, 0.52, w=W_MAIN if k == 2 else W_SEC)
    panel_wedge(c, 0.14, 0.30, 0.56, 0.52, 0.72)
    for fx in (0.30, 0.52):
        c.disc(0.14 + 0.56 * fx, 0.30 + 0.52 * 0.30, 0.028)


# =====================================================================
# gate_editor -- A BOUNDARY DRAWN BY HAND, AND THE POPULATION IT KEEPS
# =====================================================================

def gate_editor_01(c):
    """A threshold swept across a histogram by hand; beyond it is kept."""
    base = 0.80
    c.line(0.06, base, 0.96, base, W_SEC)
    hs = (0.12, 0.26, 0.46, 0.62, 0.50, 0.34, 0.20, 0.10)
    for i, f in enumerate(hs):
        x = 0.09 + i * 0.108
        h = f * 0.56
        if i >= 4:
            c.rect(x, base - h, 0.086, h, filled=True)
        else:
            c.rect(x, base - h, 0.086, h, w=W_SEC)
    c.line(0.525, 0.10, 0.525, base + 0.055, W_MAIN)
    c.rect(0.495, 0.10, 0.060, 0.060, filled=True)
    cursor(c, 0.545, 0.30)


def gate_editor_02(c):
    """A polygon closed vertex by vertex round the cloud, handles on each."""
    dots(c, [(0.36, 0.40), (0.52, 0.52), (0.40, 0.62)], r=0.046)
    v = [(0.20, 0.44), (0.42, 0.16), (0.72, 0.34), (0.66, 0.70)]
    c.polyline(v, w=W_MAIN * 1.5)
    c.line(v[-1][0], v[-1][1], 0.30, 0.82, W_MAIN, dash=DASH_L)
    handles(c, v, s=0.050)
    cursor(c, 0.30, 0.82, s=0.13)


def gate_editor_03(c):
    """Gates chained: each hand-drawn boundary sits inside the last."""
    blob(c, 0.50, 0.50, 0.46, rot=8, w=W_MAIN * 1.5)
    blob(c, 0.47, 0.50, 0.29, rot=54, k=(1.0, 0.84, 1.12, 0.86, 1.06, 0.92),
         w=W_MAIN * 1.5)
    blob(c, 0.45, 0.51, 0.140, rot=100, k=(1.0, 0.90, 1.08, 0.88, 1.04, 0.94),
         filled=True)
    nib(c, 0.86, 0.14, s=0.16, ang=-140.0)


def gate_editor_04(c):
    """The hierarchy: each gate a row, each row the fraction that survived."""
    for j, (ind, frac) in enumerate(((0.00, 0.86), (0.13, 0.58), (0.26, 0.26))):
        y = 0.26 + j * 0.24
        c.rect(0.08 + ind, y - 0.058, 0.116, 0.116, w=W_MAIN)
        blob(c, 0.138 + ind, y, 0.040, rot=20, filled=True)
        c.bar(0.25 + ind, y - 0.036, (0.66 - ind) * frac, 0.072, filled=True)
        if j:
            c.polyline([(0.076 + ind - 0.13, y - 0.24 + 0.058),
                        (0.076 + ind - 0.13, y), (0.08 + ind, y)], w=W_SEC)


def gate_editor_05(c):
    """A rectangle dragged across a two-parameter scatter, both extents read."""
    c.axes(0.08, 0.04, 0.96, 0.90, W_MAIN)
    dots(c, [(0.32, 0.34), (0.48, 0.26), (0.42, 0.50)], r=0.046)
    rings(c, [(0.78, 0.68), (0.26, 0.78)], r=0.042)
    c.rect(0.20, 0.14, 0.44, 0.46, w=W_MAIN * 1.6)
    c.rect(0.606, 0.566, 0.072, 0.072, filled=True)
    cursor(c, 0.66, 0.62, s=0.13)


def gate_editor_06(c):
    """A gate is named: the shape carries the label it becomes a filter under."""
    dots(c, [(0.30, 0.60), (0.42, 0.52), (0.36, 0.70), (0.50, 0.64)], r=0.032)
    rings(c, [(0.72, 0.34), (0.78, 0.62)], r=0.030)
    blob(c, 0.40, 0.62, 0.24, rot=14, w=W_MAIN)
    c.line(0.56, 0.46, 0.66, 0.28, W_SEC)
    c.rect(0.60, 0.10, 0.36, 0.20, w=W_MAIN, r=0.030)
    textline(c, 0.66, 0.20, 0.24, h=0.052)


def gate_editor_07(c):
    """A predicate, not a list: the same drawn shape laid onto the next plate."""
    c.rect(0.02, 0.22, 0.38, 0.56, w=W_SEC)
    dots(c, [(0.15, 0.44), (0.23, 0.54)], r=0.042)
    rings(c, [(0.33, 0.68)], r=0.038)
    blob(c, 0.18, 0.50, 0.170, rot=10, w=W_MAIN * 1.6)
    c.rect(0.60, 0.22, 0.38, 0.56, w=W_SEC)
    dots(c, [(0.73, 0.52)], r=0.042)
    rings(c, [(0.68, 0.70), (0.90, 0.34)], r=0.038)
    blob(c, 0.76, 0.50, 0.170, rot=10, w=W_MAIN * 1.6)
    c.arrow(0.42, 0.50, 0.58, 0.50, W_MAIN * 1.3, head=0.078)


def gate_editor_08(c):
    """One drawn shape, and every open view narrowed to what is inside it."""
    blob(c, 0.50, 0.22, 0.19, rot=16, w=W_MAIN)
    dots(c, [(0.44, 0.20), (0.56, 0.26)], r=0.030)
    for tx in (0.18, 0.50, 0.82):
        c.arrow(0.50, 0.44, tx, 0.58, W_SEC, head=0.050)
    c.rect(0.04, 0.64, 0.27, 0.28, w=W_SEC)
    dots(c, [(0.12, 0.78), (0.22, 0.84)], r=0.030)
    c.rect(0.365, 0.64, 0.27, 0.28, w=W_SEC)
    c.rect(0.42, 0.78, 0.055, 0.10, filled=True)
    c.rect(0.51, 0.72, 0.055, 0.16, filled=True)
    c.rect(0.69, 0.64, 0.27, 0.28, w=W_SEC)
    for i in range(2):
        for j in range(2):
            c.disc(0.755 + i * 0.14, 0.715 + j * 0.14, 0.030)


def gate_editor_09(c):
    """The strategy as a chain: each canvas shows only what the last gate kept."""
    for k, (x, keep) in enumerate(((0.02, 3), (0.365, 2), (0.71, 1))):
        c.rect(x, 0.26, 0.27, 0.48, w=W_SEC)
        pts = [(x + 0.085, 0.42), (x + 0.175, 0.52), (x + 0.085, 0.62)]
        dots(c, pts[:keep], r=0.038)
        rings(c, pts[keep:], r=0.034)
        blob(c, x + 0.125, 0.50, 0.098 - 0.012 * k, rot=25 * k, w=W_MAIN * 1.5)
        if k < 2:
            c.arrow(x + 0.285, 0.50, x + 0.348, 0.50, W_MAIN, head=0.062)


def gate_editor_10(c):
    """Re-drawn, a gate replaces its older self rather than stacking on it."""
    dots(c, [(0.52, 0.52), (0.66, 0.60)], r=0.046)
    rings(c, [(0.22, 0.34)], r=0.042)
    blob(c, 0.36, 0.38, 0.28, rot=0, w=W_MAIN * 1.3, dash=DASH_L)
    blob(c, 0.58, 0.58, 0.30, rot=40, k=(1.0, 0.88, 1.10, 0.82, 1.06, 0.92),
         w=W_MAIN * 1.8)
    cursor(c, 0.82, 0.10, s=0.15)


# =====================================================================
# feature_explorer -- MANY FEATURES RANKED BY SEPARATION
# =====================================================================

def feature_explorer_01(c):
    """Every row a feature, every feature two humps, sorted by how far apart."""
    for j, sep in enumerate((0.34, 0.21, 0.10, 0.02)):
        y = 0.30 + j * 0.215
        hump(c, 0.48 - sep, y, 0.42, 0.155)
        hump(c, 0.48 + sep, y, 0.42, 0.155, filled=False, wd=W_MAIN)
        c.line(0.05, y, 0.95, y, W_FINE * 1.2)


def feature_explorer_02(c):
    """One feature in detail: the distance between the two classes, measured."""
    base = 0.76
    c.line(0.04, base, 0.96, base, W_SEC)
    hump(c, 0.33, base, 0.52, 0.42)
    hump(c, 0.69, base, 0.52, 0.42, filled=False, wd=W_MAIN)
    caliper(c, 0.33, 0.20, 0.69, 0.20, t=0.048, w=W_MAIN)


def feature_explorer_03(c):
    """The table's column heads lifted off and stacked into ranked order."""
    c.rect(0.03, 0.16, 0.44, 0.68, w=W_MAIN)
    for i in range(1, 3):
        c.line(0.03 + 0.44 * i / 3.0, 0.16, 0.03 + 0.44 * i / 3.0, 0.84, W_SEC)
    c.line(0.03, 0.34, 0.47, 0.34, W_MAIN)
    for i in range(3):
        textline(c, 0.065 + i * 0.147, 0.25, 0.085, h=0.052)
    c.arrow(0.51, 0.50, 0.60, 0.50, W_MAIN, head=0.066)
    for j, wdt in enumerate((0.34, 0.24, 0.14)):
        c.bar(0.64, 0.28 + j * 0.165, wdt, 0.105, filled=True)


def feature_explorer_04(c):
    """Hundreds of columns scored, and the few that actually separate."""
    base = 0.86
    c.line(0.03, base, 0.97, base, W_SEC)
    hs = (0.10, 0.62, 0.16, 0.09, 0.44, 0.12, 0.20, 0.11, 0.30, 0.08)
    for i, f in enumerate(hs):
        x = 0.05 + i * 0.094
        c.rect(x, base - f * 0.70, 0.060, f * 0.70, filled=True)
    c.line(0.03, base - 0.40 * 0.70, 0.97, base - 0.40 * 0.70, W_SEC, dash=DASH)
    for i in (1, 4):
        x = 0.05 + i * 0.094
        c.circ(x + 0.030, base - hs[i] * 0.70 - 0.020, 0.070, W_MAIN)


def feature_explorer_05(c):
    """Separation on a bounded scale: a coin flip at one end, gateable at the other."""
    y = 0.56
    c.line(0.08, y, 0.92, y, W_MAIN)
    c.line(0.08, y - 0.055, 0.08, y + 0.055, W_MAIN)
    c.line(0.92, y - 0.055, 0.92, y + 0.055, W_MAIN)
    hump(c, 0.16, 0.34, 0.26, 0.21)
    hump(c, 0.16, 0.34, 0.26, 0.21, filled=False, wd=W_MAIN)
    hump(c, 0.74, 0.34, 0.22, 0.21)
    hump(c, 0.90, 0.34, 0.22, 0.21, filled=False, wd=W_MAIN)
    for fx, r in ((0.26, 0.036), (0.40, 0.036), (0.58, 0.036), (0.80, 0.060)):
        c.disc(fx, y + 0.16, r)
        c.line(fx, y, fx, y + 0.16 - r, W_SEC)


def feature_explorer_06(c):
    """The blind spot: same centre, different spread -- scored nothing, obviously real."""
    base = 0.72
    c.line(0.08, base, 0.92, base, W_SEC)
    hump(c, 0.50, base, 0.76, 0.30)
    hump(c, 0.50, base, 0.30, 0.44, filled=False, wd=W_MAIN)
    cross(c, 0.83, 0.22, 0.090, W_MAIN)


def feature_explorer_07(c):
    """Each ranked row says which class it is the higher one in."""
    for j, (wdt, up) in enumerate(((0.46, True), (0.36, False),
                                   (0.26, True), (0.16, False))):
        y = 0.20 + j * 0.20
        c.bar(0.08, y, 0.46 * (wdt / 0.46), 0.096, filled=True)
        cx = 0.66
        if up:
            c.arrow(cx, y + 0.14, cx, y - 0.03, W_MAIN, head=0.062)
        else:
            c.arrow(cx, y - 0.03, cx, y + 0.14, W_MAIN, head=0.062)
        c.disc(0.84, y + 0.048, 0.040) if up else c.circ(0.84, y + 0.048,
                                                         0.038, W_MAIN)


def feature_explorer_08(c):
    """Ranked by separation, not by size: the small clean one beats the big blurred one."""
    c.rect(0.04, 0.20, 0.42, 0.70, w=W_SEC)
    hump(c, 0.19, 0.78, 0.34, 0.42)
    hump(c, 0.29, 0.78, 0.34, 0.42, filled=False, wd=W_MAIN)
    c.rect(0.54, 0.20, 0.42, 0.70, w=W_MAIN * 1.7)
    hump(c, 0.66, 0.78, 0.20, 0.26)
    hump(c, 0.85, 0.78, 0.20, 0.26, filled=False, wd=W_MAIN)
    c.arrow(0.75, 0.16, 0.75, 0.03, W_MAIN * 1.4, head=0.080)


def feature_explorer_09(c):
    """What the score means: the two classes interleaved, and the two ordered."""
    for i in range(4):
        c.disc(0.14 + i * 0.24, 0.30, 0.048)
        c.circ(0.24 + i * 0.24, 0.30, 0.046, W_MAIN)
    c.line(0.04, 0.42, 0.96, 0.42, W_SEC)
    c.arrow(0.50, 0.50, 0.50, 0.60, W_SEC, head=0.055)
    for i in range(4):
        c.disc(0.11 + i * 0.108, 0.80, 0.048)
        c.circ(0.57 + i * 0.108, 0.80, 0.046, W_MAIN)
    c.line(0.04, 0.92, 0.96, 0.92, W_SEC)


def feature_explorer_10(c):
    """Every continuous column scored against the one class column."""
    x, y, w, h = 0.06, 0.10, 0.88, 0.46
    c.rect(x, y, w, h, w=W_SEC)
    for i in range(1, 4):
        c.line(x + w * i / 4.0, y, x + w * i / 4.0, y + h, W_SEC)
    c.line(x, y + 0.13, x + w, y + 0.13, W_MAIN)
    c.rect(x + w * 0.75, y, w * 0.25, h, filled=True)
    for i in range(3):
        cx = x + w * (i + 0.5) / 4.0
        c.arrow(cx, y + h + 0.05, cx, y + h + 0.20, W_SEC, head=0.055)
        c.bar(cx - 0.09, y + h + 0.24, 0.18 * (0.5 + 0.25 * (2 - i)), 0.080,
              filled=True)


# =====================================================================
# napari_bridge
#   01-05  derived from napari's mark  (napari's trademark, not spaCR's)
#   06-10  original spaCR marks about the handoff (no third-party mark)
# =====================================================================

def napari_bridge_01(c):
    """[napari mark] The four-petal rosette on its own, monochrome."""
    rosette(c, 0.50, 0.50, 0.44)
    c.disc(0.50, 0.50, 0.052)


def napari_bridge_02(c):
    """[napari mark] The rosette open: petals as outlines around a clear centre."""
    rosette(c, 0.50, 0.50, 0.44, filled=False, w=W_MAIN)
    c.circ(0.50, 0.50, 0.085, W_MAIN)


def napari_bridge_03(c):
    """[napari mark] The rosette in another application's window."""
    window(c, 0.06, 0.16, 0.88, 0.70, wd=W_MAIN, bar=0.18)
    rosette(c, 0.50, 0.585, 0.235)
    c.disc(0.50, 0.585, 0.030)


def napari_bridge_04(c):
    """[napari mark] A mask handed out to the rosette and taken back corrected."""
    rosette(c, 0.72, 0.50, 0.245)
    c.disc(0.72, 0.50, 0.032)
    c.ell(0.20, 0.50, 0.145, 0.125, 0.0, W_MAIN)
    c.disc(0.20, 0.50, 0.052)
    c.arrow(0.34, 0.36, 0.50, 0.30, W_SEC, head=0.058)
    c.arrow(0.50, 0.70, 0.34, 0.64, W_SEC, head=0.058)


def napari_bridge_05(c):
    """[napari mark] Their brush, our data: a brush laid across the rosette."""
    rosette(c, 0.46, 0.56, 0.38)
    c.disc(0.46, 0.56, 0.044)
    c.polyline([(0.60, 0.42), (0.86, 0.16), (0.96, 0.26), (0.70, 0.52)],
               close=True, filled=True)
    c.polyline([(0.60, 0.42), (0.52, 0.60), (0.70, 0.52)], close=True,
               filled=True)


def napari_bridge_06(c):
    """[original] Two panes and a span between them, an object walking across."""
    c.rect(0.02, 0.06, 0.30, 0.50, w=W_MAIN * 1.4)
    c.rect(0.68, 0.06, 0.30, 0.50, w=W_MAIN * 1.4)
    c.line(0.16, 0.72, 0.84, 0.72, W_MAIN * 2.2)
    c.arc(0.50, 0.72, 0.26, 180, 180, W_MAIN * 1.4)
    c.line(0.24, 0.72, 0.24, 0.94, W_MAIN * 1.2)
    c.line(0.76, 0.72, 0.76, 0.94, W_MAIN * 1.2)
    c.ell(0.50, 0.60, 0.115, 0.100, 0.0, W_MAIN * 1.4)
    c.disc(0.50, 0.60, 0.044)
    c.arrow(0.37, 0.31, 0.65, 0.31, W_MAIN * 1.2, head=0.075)


def napari_bridge_07(c):
    """[original] The mask goes out rough and comes back with its boundary fixed."""
    c.rect(0.04, 0.06, 0.40, 0.40, w=W_SEC)
    blob(c, 0.24, 0.26, 0.135, rot=18, k=(1.0, 0.72, 1.18, 0.70, 1.14, 0.80),
         w=W_MAIN)
    c.rect(0.56, 0.54, 0.40, 0.40, w=W_SEC)
    c.ell(0.76, 0.74, 0.145, 0.128, 0.0, W_MAIN)
    c.disc(0.76, 0.74, 0.050)
    c.arrow(0.50, 0.20, 0.86, 0.44, W_MAIN, head=0.070)
    c.arrow(0.50, 0.80, 0.14, 0.56, W_MAIN, head=0.070)


def napari_bridge_08(c):
    """[original] Two windows overlapping, one object shared in the seam."""
    c.rect(0.04, 0.10, 0.56, 0.56, w=W_MAIN)
    c.rect(0.40, 0.34, 0.56, 0.56, w=W_MAIN)
    c.ell(0.50, 0.50, 0.135, 0.118, 0.0, W_MAIN)
    c.disc(0.50, 0.50, 0.048)
    c.line(0.40, 0.34, 0.40, 0.66, W_SEC, dash=DASH)
    c.line(0.40, 0.66, 0.60, 0.66, W_SEC, dash=DASH)


def napari_bridge_09(c):
    """[original] It comes back as itself: the label number survives the crossing."""
    for x in (0.02, 0.62):
        c.rect(x, 0.20, 0.36, 0.60, w=W_MAIN * 1.3)
        blob(c, x + 0.18, 0.44, 0.130, rot=10, filled=True)
        c.rect(x + 0.075, 0.62, 0.210, 0.120, w=W_MAIN, r=0.032)
        c.disc(x + 0.130, 0.680, 0.031)
        c.disc(x + 0.230, 0.680, 0.031)
    c.arrow(0.40, 0.50, 0.60, 0.50, W_MAIN * 1.4, head=0.072)


def napari_bridge_10(c):
    """[original] Nothing crosses back unchecked: the wrong shape is turned away."""
    c.line(0.42, 0.02, 0.42, 0.98, W_MAIN * 1.6, dash=DASH_L)
    c.ell(0.82, 0.22, 0.125, 0.110, 0.0, W_MAIN * 1.4)
    c.disc(0.82, 0.22, 0.046)
    c.arrow(0.64, 0.22, 0.06, 0.22, W_MAIN * 1.4, head=0.085)
    blob(c, 0.84, 0.74, 0.145, rot=20, k=(1.0, 0.56, 1.28, 0.54, 1.24, 0.64),
         w=W_MAIN * 1.4)
    c.rect(0.475, 0.52, 0.055, 0.44, filled=True)
    c.arrow(0.66, 0.74, 0.555, 0.74, W_MAIN * 1.4, head=0.085)


# ---------------------------------------------------------------------------
GROUPS = {
    "trellis": ("Small Multiples (trellis) - the repetition is the subject", [
        ("A 3x3 of identical frames, the same mark standing at a different height.",
         trellis_01),
        ("Shared axes: one y axis and one x axis serving the whole grid.",
         trellis_02),
        ("Two-way faceting: condition across the top, plate down the side.",
         trellis_03),
        ("One reference level ruled straight through every panel.", trellis_04),
        ("Every panel prints its n, so a panel of three cannot pass for a panel of thousands.",
         trellis_05),
        ("The same shape at two very different levels, comparable only because the axis is shared.",
         trellis_06),
        ("A long strip of levels wrapping onto the next row.", trellis_07),
        ("An empty panel is still drawn: 'measured, nothing survived' stays its own picture.",
         trellis_08),
        ("A brush on one panel picks the same objects out of every other panel.",
         trellis_09),
        ("The same chart, many times over: identical frames stepping back into depth.",
         trellis_10),
    ]),
    "gate_editor": ("gate_editor - a boundary drawn by hand, and what it keeps", [
        ("A threshold swept across a histogram by hand; the bars beyond it are kept.",
         gate_editor_01),
        ("A polygon closed vertex by vertex round the cloud, a grab handle on each.",
         gate_editor_02),
        ("Gates chained: each hand-drawn boundary sits inside the one before it.",
         gate_editor_03),
        ("The gating hierarchy: each gate a row, each row the fraction that survived it.",
         gate_editor_04),
        ("A rectangle dragged across a two-parameter scatter, both extents read.",
         gate_editor_05),
        ("A gate is named: the shape carries the label it becomes a filter under.",
         gate_editor_06),
        ("A predicate, not a list of objects: the same drawn shape laid onto the next plate.",
         gate_editor_07),
        ("One drawn shape, and every open view narrowed to what is inside it.",
         gate_editor_08),
        ("The strategy as a chain: each canvas shows only what the last gate kept.",
         gate_editor_09),
        ("Re-drawn, a gate replaces its older self instead of stacking on it.",
         gate_editor_10),
    ]),
    "feature_explorer": ("feature_explorer - features ranked by separation", [
        ("Every row a feature, every feature two humps, sorted by how far apart they sit.",
         feature_explorer_01),
        ("One feature in detail: the distance between the two classes, measured.",
         feature_explorer_02),
        ("The table's column heads lifted off and stacked into ranked order.",
         feature_explorer_03),
        ("Hundreds of columns scored, and the two that actually separate ringed.",
         feature_explorer_04),
        ("Separation on a bounded scale: a coin flip at one end, nearly gateable at the other.",
         feature_explorer_05),
        ("The blind spot: same centre, different spread - scores nothing, obviously informative.",
         feature_explorer_06),
        ("Each ranked row also says which of the two classes is the higher one.",
         feature_explorer_07),
        ("Ranked by separation, not by size: the small clean feature beats the big blurred one.",
         feature_explorer_08),
        ("What the score means: the two classes interleaved, and the two fully ordered.",
         feature_explorer_09),
        ("Every continuous column scored against the one class column.",
         feature_explorer_10),
    ]),
    "napari_bridge": ("napari_bridge - 01-05 napari's mark, 06-10 original", [
        ("[napari mark - their trademark] The four-petal rosette on its own, monochrome.",
         napari_bridge_01),
        ("[napari mark - their trademark] The rosette open: petals as outlines round a clear centre.",
         napari_bridge_02),
        ("[napari mark - their trademark] The rosette inside another application's window.",
         napari_bridge_03),
        ("[napari mark - their trademark] A mask handed out to the rosette and taken back corrected.",
         napari_bridge_04),
        ("[napari mark - their trademark] Their brush, our data: a brush laid across the rosette.",
         napari_bridge_05),
        ("[original - no third-party mark] Two panes and a span between them, an object walking across.",
         napari_bridge_06),
        ("[original - no third-party mark] The mask goes out rough and comes back with its boundary fixed.",
         napari_bridge_07),
        ("[original - no third-party mark] Two windows overlapping, one object shared in the seam.",
         napari_bridge_08),
        ("[original - no third-party mark] It comes back as itself: the label number survives the crossing.",
         napari_bridge_09),
        ("[original - no third-party mark] Nothing crosses back unchecked: the wrong shape is turned away.",
         napari_bridge_10),
    ]),
}


TRADEMARK_NOTE = """
Whose mark is on which candidate
--------------------------------

**01-05 carry napari's own visual identity** -- a monochrome four-petal
rosette in the manner of napari's mark. That mark is **napari's trademark, not
spaCR's**. Labelling a bridge *to* napari with it is ordinary nominative use
and is what most integrations do, but picking one of these means shipping a
monochrome derivative of a third-party trademark, so it should be a decision
rather than an accident.

**06-10 are original spaCR marks about the handoff** -- two panes and traffic
between them, a mask leaving rough and returning corrected, a label value that
survives the crossing, a return that is checked. They carry no third-party
mark at all.

Nothing was copied. No napari image file exists in this repository; the
rosette is built from the same `_draw` primitives as every other icon here.
"""


def main(argv):
    outdir = argv[1] if len(argv) > 1 else default_outdir(__file__)
    rc = emit_groups(outdir, GROUPS,
                     "group_trellis_gate_feature_napari.py")
    with open(os.path.join(outdir, "napari_bridge", "CONCEPTS.md"), "a") as fh:
        fh.write(TRADEMARK_NOTE)
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
