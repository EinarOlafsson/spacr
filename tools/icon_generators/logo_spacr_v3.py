#!/usr/bin/env python3
"""Third round of candidate marks for the spaCR logo itself.

The second round (``logo_spacr_v2.py``) was rejected in full: *"lines should be
thinner, shapes should be simpler"*.  That round let one constraint drive every
decision -- "must survive 16 px" -- and drew heavy to satisfy it, then piled on
parts to keep meaning once the drawing had gone blunt.

This round inverts the priority:

  * **Draw thin first, measure afterwards.**  The primary stroke here is 24
    canvas units against the previous round's 76-96.  Where that softens below
    32 px, ``CONCEPTS.md`` says so per candidate and leaves the trade to the
    user -- a mark that is beautiful at 128 px and soft at 16 may still be the
    right answer.
  * **Two elements is the common case, one is reachable.**  Nothing here needs
    more than two parts to read, and eight of the thirty are a single form.
  * Directions the rejected round never tried: pure negative space, one
    unbroken line, a single filled form with one knockout, asymmetry, and marks
    that occupy 45-60% of the frame instead of filling it.

Thirty candidates in four groups, named so the group is obvious::

  concept_NN_*   10 new ideas, white on transparent
  variant_NN_*   10 variants of those ideas, white on transparent
  thin_01_*      1 hairline variant
  colour_NN_*    9 colour treatments

Colour inks are inherited unchanged from round two: relative luminance held
between 0.13 and 0.27, so the *same file* clears 3:1 against both the dark
(#14161a) and the light (#f5f6f8) background.

Run standalone::

    QT_QPA_PLATFORM=offscreen python3 logo_spacr_v3.py [outdir]
"""

from __future__ import annotations

import math
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtCore import QRectF, Qt  # noqa: E402
from PySide6.QtGui import QImage, QPainterPath, QTransform  # noqa: E402

# Round two's canvas, output stage and palette are reused verbatim; only the
# drawing is new.
from logo_spacr_v2 import (  # noqa: E402
    CORAL,
    DARK_BG,
    INDIGO,
    LIGHT_BG,
    SLATE,
    TEAL,
    _coverage,
    _lum,
    _ratio,
    p_catmull,
    p_circle,
    render,
    sheet,
    small_sheet,
    stroked,
)

# ---------------------------------------------------------------- weights
# In 1024-canvas units.  Divide by 8 for the width in px at 128, by 64 at 16.
LINE = 24.0     # primary stroke   -- 3.0 px at 128, 0.75 px at 32
FINE = 17.0     # secondary stroke -- 2.1 px at 128
HAIR = 11.0     # the group C hairline
CUT = 84.0      # width of a *knocked out* line: negative space needs more room


# --------------------------------------------------------------------------
# geometry shared by more than one candidate
# --------------------------------------------------------------------------

#: a calm irregular cell silhouette -- the parent logo's identity without its
#: five moving parts.  (angle deg, radius), screen coordinates, y down.
BLOB = [
    (0, 0.346), (40, 0.322), (80, 0.294), (120, 0.306),
    (160, 0.336), (200, 0.354), (240, 0.338), (280, 0.300), (320, 0.328),
]


def blob_pts(cx=0.500, cy=0.500, s=1.0, rot=0.0, table=BLOB):
    out = []
    for a, r in table:
        t = math.radians(a + rot)
        out.append((cx + s * r * math.cos(t), cy + s * r * math.sin(t)))
    return out


def blob_path(c, cx=0.500, cy=0.500, s=1.0, rot=0.0, table=BLOB) -> QPainterPath:
    return p_catmull(c, blob_pts(cx, cy, s, rot, table), closed=True)


def trace_pts(sweep, cx=0.492, cy=0.512, base=0.300, steps=120, spiral=True):
    """The hand-drawn contour: one pass around, optionally overshooting.

    With ``spiral`` the radius drifts outward for most of the sweep and then
    dives back inside, so the tail crosses its own start instead of merely
    lying beside it.
    """
    pts = []
    for i in range(steps + 1):
        f = i / steps
        t = sweep * f
        r = base + 0.021 * math.sin(2 * t + 0.6) + 0.013 * math.sin(3 * t + 1.9)
        if spiral:
            s = 0.11 * f if f <= 0.78 else 0.086 - 0.196 * (f - 0.78) / 0.22
            r *= 1.0 + s
        pts.append((cx + r * math.cos(t), cy + r * math.sin(t)))
    return pts


def half_plane(c, px, py, ang_deg, gap):
    """A rect covering everything ``gap`` beyond the line through (px, py)."""
    pa = QPainterPath()
    pa.addRect(QRectF(-1.5 * c.n, gap * c.n, 3.0 * c.n, 2.0 * c.n))
    t = QTransform()
    t.translate(px * c.n, py * c.n)
    t.rotate(ang_deg)
    return t.map(pa)


def p_arc(c, cx, cy, r, a0, span) -> QPainterPath:
    rr = QRectF((cx - r) * c.n, (cy - r) * c.n, 2 * r * c.n, 2 * r * c.n)
    pa = QPainterPath()
    pa.arcMoveTo(rr, a0)
    pa.arcTo(rr, a0, span)
    return pa


# ==========================================================================
# GROUP A -- ten new ideas.  Each is one or two elements.
# ==========================================================================

def sh_trace(c, col=None, w=LINE, sweep=None, spiral=True, dot=None, c_dot=None):
    """One unbroken line: a contour drawn past its own start."""
    if sweep is None:
        sweep = 2 * math.pi + 0.92
    c.set_ink(col)
    c.smooth(trace_pts(sweep, spiral=spiral), w=w, closed=False)
    if dot:
        c.set_ink(c_dot or col)
        c.disc(dot[0], dot[1], dot[2])


def sh_counter_c(c, col=None, R=0.352, r=0.190, cut=CUT):
    """Pure negative space: the C exists only as what is missing from the disc."""
    c.set_ink(col)
    void = stroked(c, p_arc(c, 0.500, 0.500, r, 46.0, 268.0), cut, cap=Qt.RoundCap)
    c.fill(p_circle(c, 0.500, 0.500, R).subtracted(void))


def sh_offset_nucleus(c, c_ring=None, c_dot=None, w=LINE, r=0.318,
                      dot=(0.626, 0.418, 0.053)):
    """A membrane and its nucleus, and the nucleus is nowhere near the middle."""
    c.set_ink(c_ring)
    c.circ(0.500, 0.500, r, w=w)
    c.set_ink(c_dot)
    c.disc(*dot)


def sh_open_c(c, col=None, w=26.0, r=0.330, a0=38.0, span=284.0, cx=0.500, cy=0.500):
    """One arc.  The C of spaCR and a membrane are the same stroke."""
    c.set_ink(col)
    c.stroke(p_arc(c, cx, cy, r, a0, span), w)


def sh_chord(c, c_ring=None, c_cut=None, w=LINE, r=0.316, off=0.104, ang=27.0,
             fill_cap=False):
    """One boundary, one cut across it -- off centre, so it is a decision and
    not a symmetry."""
    a = math.radians(ang)
    nx, ny = -math.sin(a), math.cos(a)
    px, py = 0.500 + nx * off, 0.500 + ny * off
    c.set_ink(c_ring)
    c.circ(0.500, 0.500, r, w=w)
    if fill_cap:
        # the straight edge of the fill *is* the cut, so no line is drawn
        inner = p_circle(c, 0.500, 0.500, r - (w / 1024.0) / 2)
        c.set_ink(c_cut)
        c.fill(inner.intersected(half_plane(c, px, py, ang, 0.0)))
        return
    half = math.sqrt(max(r * r - off * off, 0.0)) + 0.036
    c.set_ink(c_cut)
    c.line(px - half * math.cos(a), py - half * math.sin(a),
           px + half * math.cos(a), py + half * math.sin(a), w=w)


def sh_solid_cell(c, col=None, cx=0.472, cy=0.534, s=0.86, nuc=(0.588, 0.436, 0.072)):
    """A single filled form with one knockout.  No outline, no grid, no arcs."""
    c.set_ink(col)
    body = blob_path(c, cx, cy, s, rot=-14.0)
    c.fill(body.subtracted(p_circle(c, nuc[0], nuc[1], nuc[2])))


def sh_pair(c, c_big=None, c_small=None, w=LINE, solid=False,
            small=(0.712, 0.368, 0.138)):
    """Two cells, unequal, touching: the smallest picture of segmentation."""
    c.set_ink(c_big)
    c.circ(0.412, 0.556, 0.258, w=w)
    c.set_ink(c_small)
    if solid:
        c.disc(*small)
    else:
        c.circ(small[0], small[1], small[2], w=w)


def sh_taper(c, col=None, r=0.312, wmin=4.0, wmax=52.0, phase=2.35, steps=280):
    """One closed continuous line whose weight swells and thins once."""
    c.set_ink(col)
    outer, inner = [], []
    for i in range(steps + 1):
        t = 2 * math.pi * i / steps
        w = (wmin + (wmax - wmin) * (0.5 - 0.5 * math.cos(t - phase))) / 1024.0
        outer.append((0.500 + (r + w / 2) * math.cos(t),
                      0.500 + (r + w / 2) * math.sin(t)))
        inner.append((0.500 + (r - w / 2) * math.cos(t),
                      0.500 + (r - w / 2) * math.sin(t)))
    pa = QPainterPath()
    pa.moveTo(c.pt(*outer[0]))
    for q in outer[1:]:
        pa.lineTo(c.pt(*q))
    for q in reversed(inner):
        pa.lineTo(c.pt(*q))
    pa.closeSubpath()
    c.fill(pa)


def sh_two_arcs(c, col=None, w=LINE, r=0.316, gap=34.0, rot=55.0):
    """One boundary drawn in two strokes; the cell is the space they enclose."""
    c.set_ink(col)
    span = 180.0 - gap
    c.stroke(p_arc(c, 0.500, 0.500, r, rot + gap / 2, span), w)
    c.stroke(p_arc(c, 0.500, 0.500, r, rot + 180.0 + gap / 2, span), w)


def sh_cropped(c, c_ring=None, c_dot=None, w=LINE):
    """A cell that runs out of the field of view -- the frame is a crop, not a
    stage."""
    c.set_ink(c_ring)
    c.circ(0.284, 0.682, 0.456, w=w)
    c.set_ink(c_dot)
    c.disc(0.272, 0.600, 0.066)


def sh_arc_dot(c, c_arc=None, c_dot=None, w=LINE):
    """A fragment of membrane and the body it belongs to -- 45% of the frame,
    the rest air."""
    c.set_ink(c_arc)
    c.stroke(p_arc(c, 0.780, 0.500, 0.420, 138.0, 84.0), w)
    c.set_ink(c_dot)
    c.disc(0.584, 0.462, 0.082)


def sh_reverse(c, col=None, R=0.354, s=0.585):
    """The cell as pure absence: a disc with the silhouette cut out of it."""
    c.set_ink(col)
    c.fill(p_circle(c, 0.500, 0.500, R).subtracted(
        blob_path(c, 0.536, 0.470, s, rot=26.0)))


def sh_blob(c, c_edge=None, c_dot=None, w=LINE, dot=None):
    """The parent silhouette, thin, with at most one thing inside it."""
    c.set_ink(c_edge)
    c.stroke(blob_path(c, 0.500, 0.502, 0.96, rot=-14.0), w)
    if dot:
        c.set_ink(c_dot)
        c.disc(*dot)


def sh_c_dot(c, c_ring=None, c_dot=None, w=26.0, r=0.318, dot=(0.478, 0.500, 0.058)):
    """The open C with one nucleus: membrane and body, two elements."""
    sh_open_c(c, col=c_ring, w=w, r=r, a0=40.0, span=280.0)
    c.set_ink(c_dot)
    c.disc(*dot)


# ==========================================================================
# catalogue
# ==========================================================================

def _wrap(fn, **kw):
    return lambda c: fn(c, **kw)


CANDIDATES = [
    # ---- group A: 10 new concepts, black and white --------------------
    ("concept_01_trace", "concept", False,
     "One unbroken line: a contour drawn past its own start - the boundary "
     "while it is being drawn."),
    ("concept_02_counter_c", "concept", False,
     "Pure negative space: one disc, and the C is only what has been cut out "
     "of it."),
    ("concept_03_offset_nucleus", "concept", False,
     "A thin membrane and one small nucleus, deliberately off centre."),
    ("concept_04_open_c", "concept", False,
     "A single arc, nothing else - the C of spaCR and a membrane in one stroke."),
    ("concept_05_chord", "concept", False,
     "One boundary and one cut across it, off centre."),
    ("concept_06_solid_cell", "concept", False,
     "One filled cell with the nucleus knocked out of it - no outline anywhere."),
    ("concept_07_pair", "concept", False,
     "Two unequal cells, touching - the smallest picture of segmentation."),
    ("concept_08_taper", "concept", False,
     "One closed continuous line whose weight swells and thins once."),
    ("concept_09_two_arcs", "concept", False,
     "Two strokes; the cell is the space between them."),
    ("concept_10_cropped", "concept", False,
     "A cell running out of the field of view - the frame reads as a crop."),

    # ---- group B: 10 variants, black and white ------------------------
    ("variant_01_trace_open", "variant", False,
     "The trace left open: one pass, a gap where it started."),
    ("variant_02_trace_dot", "variant", False,
     "The trace with its nucleus - two elements, one of them a dot."),
    ("variant_03_c_dot", "variant", False,
     "The open C with one small nucleus inside it."),
    ("variant_04_c_wide", "variant", False,
     "The C at 47% of the frame with a wide mouth - the white-space study."),
    ("variant_05_blob", "variant", False,
     "The parent silhouette, thin, and nothing else inside it."),
    ("variant_06_blob_dot", "variant", False,
     "The parent silhouette with one off-centre nucleus - its identity in two "
     "parts."),
    ("variant_07_reverse", "variant", False,
     "Reversed: a disc with the cell silhouette cut out of it."),
    ("variant_08_cut_fill", "variant", False,
     "The cut, with the smaller side filled - one cell resolved out of one "
     "boundary."),
    ("variant_09_pair_solid", "variant", False,
     "The pair with the smaller cell solid."),
    ("variant_10_arc_dot", "variant", False,
     "A fragment of membrane above one body - the most reduced mark in the set."),

    # ---- group C: 1 hairline variant ----------------------------------
    ("thin_01_c_dot", "thin", False,
     "Variant 13 at hairline weight - 11 units, under half the primary stroke."),

    # ---- group D: 9 colour --------------------------------------------
    ("colour_01_teal_trace", "colour", True,
     "Single colour: the trace in one teal."),
    ("colour_02_teal_open_c", "colour", True,
     "Single colour: the open C in one teal."),
    ("colour_03_indigo_offset", "colour", True,
     "Single colour: membrane and off-centre nucleus in one indigo."),
    ("colour_04_coral_solid_cell", "colour", True,
     "Single colour: the filled cell in one coral."),
    ("colour_05_teal_coral_c_dot", "colour", True,
     "Two colours: teal C, coral nucleus."),
    ("colour_06_slate_coral_trace", "colour", True,
     "Two colours: slate trace, coral nucleus."),
    ("colour_07_teal_coral_blob", "colour", True,
     "Two colours: teal silhouette, coral nucleus."),
    ("colour_08_indigo_counter_c", "colour", True,
     "Single colour: the negative-space C in one indigo."),
    ("colour_09_teal_indigo_pair", "colour", True,
     "Two colours: teal cell, indigo cell."),
]

_BLOB_DOT = (0.586, 0.446, 0.070)
_TRACE_DOT = (0.560, 0.452, 0.062)

DRAW = {
    "concept_01_trace": _wrap(sh_trace),
    "concept_02_counter_c": _wrap(sh_counter_c),
    "concept_03_offset_nucleus": _wrap(sh_offset_nucleus),
    "concept_04_open_c": _wrap(sh_open_c),
    "concept_05_chord": _wrap(sh_chord),
    "concept_06_solid_cell": _wrap(sh_solid_cell),
    "concept_07_pair": _wrap(sh_pair),
    "concept_08_taper": _wrap(sh_taper),
    "concept_09_two_arcs": _wrap(sh_two_arcs),
    "concept_10_cropped": _wrap(sh_cropped),

    "variant_01_trace_open": _wrap(sh_trace, sweep=2 * math.pi - 0.72, spiral=False),
    "variant_02_trace_dot": _wrap(sh_trace, dot=_TRACE_DOT),
    "variant_03_c_dot": _wrap(sh_c_dot),
    "variant_04_c_wide": _wrap(sh_open_c, w=22.0, r=0.236, a0=60.0, span=240.0),
    "variant_05_blob": _wrap(sh_blob),
    "variant_06_blob_dot": _wrap(sh_blob, dot=_BLOB_DOT),
    "variant_07_reverse": _wrap(sh_reverse),
    "variant_08_cut_fill": _wrap(sh_chord, fill_cap=True),
    "variant_09_pair_solid": _wrap(sh_pair, solid=True, small=(0.744, 0.340, 0.136)),
    "variant_10_arc_dot": _wrap(sh_arc_dot),

    "thin_01_c_dot": _wrap(sh_c_dot, w=HAIR, dot=(0.478, 0.500, 0.050)),

    "colour_01_teal_trace": _wrap(sh_trace, col=TEAL),
    "colour_02_teal_open_c": _wrap(sh_open_c, col=TEAL),
    "colour_03_indigo_offset": _wrap(sh_offset_nucleus, c_ring=INDIGO, c_dot=INDIGO),
    "colour_04_coral_solid_cell": _wrap(sh_solid_cell, col=CORAL),
    "colour_05_teal_coral_c_dot": _wrap(sh_c_dot, c_ring=TEAL, c_dot=CORAL),
    "colour_06_slate_coral_trace": _wrap(sh_trace, col=SLATE, dot=_TRACE_DOT,
                                         c_dot=CORAL),
    "colour_07_teal_coral_blob": _wrap(sh_blob, c_edge=TEAL, c_dot=CORAL,
                                       dot=_BLOB_DOT),
    "colour_08_indigo_counter_c": _wrap(sh_counter_c, col=INDIGO),
    "colour_09_teal_indigo_pair": _wrap(sh_pair, c_big=TEAL, c_small=INDIGO),
}


# ==========================================================================
# output
# ==========================================================================

def main(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = argv[1] if len(argv) > 1 else os.path.normpath(
        os.path.join(here, "..", "logo_spacr"))
    os.makedirs(outdir, exist_ok=True)

    images = []
    for name, _group, _is_colour, _desc in CANDIDATES:
        path = os.path.join(outdir, name + ".png")
        render(DRAW[name], path)
        images.append(QImage(path))
        print("%6.2f%%  %s" % (_coverage(path) * 100, name))

    sheet(CANDIDATES, images, os.path.join(outdir, "_sheet_dark.png"), DARK_BG)
    sheet(CANDIDATES, images, os.path.join(outdir, "_sheet_light.png"), LIGHT_BG)
    small_sheet(CANDIDATES, images, os.path.join(outdir, "_sheet_small.png"),
                DARK_BG)
    small_sheet(CANDIDATES, images, os.path.join(outdir, "_sheet_small_light.png"),
                LIGHT_BG)

    print("\nstroke weights: primary %.0f, secondary %.0f, hairline %.0f units"
          % (LINE, FINE, HAIR))
    print("  = %.2f / %.2f / %.2f px at 128 px, %.2f / %.2f / %.2f px at 32 px"
          % (LINE / 8, FINE / 8, HAIR / 8, LINE / 32, FINE / 32, HAIR / 32))
    print("\ncolour contrast (ink vs each background):")
    for label, col in (("TEAL", TEAL), ("INDIGO", INDIGO), ("CORAL", CORAL),
                       ("SLATE", SLATE)):
        print("  %-7s %s  lum %.3f   dark %.2f:1   light %.2f:1"
              % (label, col, _lum(col), _ratio(col, DARK_BG), _ratio(col, LIGHT_BG)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
