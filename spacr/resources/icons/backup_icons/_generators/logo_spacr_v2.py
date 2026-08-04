#!/usr/bin/env python3
"""Second round of candidate marks for the spaCR logo itself.

Brief (the user's words): *streamlined, simple, intuitive, elegant*.  The mark
has to work as an app icon, a favicon and a splash, so the governing constraint
here is **16 px**, not 1024 px.

At 16 px one canvas unit of a 1024 design is 1/64 px, so a stroke only survives
if it is roughly 0.07-0.10 of the canvas (72-96 units).  Everything in the
``concept_`` group is drawn to that budget; the ``variant_`` group inherits more
structure from the existing ``logo_spacr.png`` and is honestly weaker at 16 px.

Thirty candidates in four groups, named so the group is obvious:

  * ``concept_NN_*``  -- 10 new ideas, white on transparent
  * ``variant_NN_*``  -- 10 refinements of the existing logo, white on transparent
  * ``thin_01_*``     -- 1 refinement drawn with thinner lines
  * ``colour_NN_*``   -- 9 colour treatments of the strongest forms

Colour candidates use only inks whose relative luminance sits between 0.13 and
0.27, so the *same file* clears 3:1 against both the dark (#14161a) and the
light (#f5f6f8) background; the contrast ratios are printed on every run.

Run standalone::

    QT_QPA_PLATFORM=offscreen python3 logo_spacr_v2.py [outdir]
"""

from __future__ import annotations

import math
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtCore import QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import (  # noqa: E402
    QBrush,
    QColor,
    QFont,
    QImage,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPen,
    QTransform,
)

from _draw import N, Cv  # noqa: E402

DARK_BG = "#14161a"
LIGHT_BG = "#f5f6f8"

# stroke weights in 1024-canvas units; /64 gives the width in px at 16x16
HEAVY = 96.0    # 1.50 px at 16 px
BOLD = 76.0     # 1.19 px at 16 px
MED = 56.0      # 0.88 px at 16 px
THIN = 38.0     # 0.59 px at 16 px
HAIR = 24.0     # 0.38 px at 16 px

# ---------------------------------------------------------------- palette
# mid-tone inks: luminance kept inside [0.126, 0.269] so one file reads on
# both backgrounds (see _contrast_report)
TEAL = "#0E9488"
INDIGO = "#5B63D6"
CORAL = "#E0533A"
SLATE = "#6F7B8A"
VIOLET = "#8B5CD6"
AMBER = "#A8761A"


# --------------------------------------------------------------------------
# canvas
# --------------------------------------------------------------------------

class Ink(Cv):
    """:class:`_draw.Cv` with a settable ink colour instead of hard white."""

    def __init__(self, n: int = N):
        super().__init__(n)
        self._ink = QColor(255, 255, 255)

    def set_ink(self, col=None) -> None:
        self._ink = QColor(col) if col else QColor(255, 255, 255)

    # -- overridden so every primitive in Cv picks the current ink up -------
    def pen(self, w: float, cap=Qt.RoundCap) -> QPen:
        pen = QPen(self._ink)
        pen.setWidthF(max(0.4, w * self.n / 1024.0))
        pen.setCapStyle(cap)
        pen.setJoinStyle(Qt.RoundJoin)
        return pen

    def fill(self, path: QPainterPath) -> None:
        self.p.setPen(Qt.NoPen)
        self.p.setBrush(QBrush(self._ink))
        self.p.drawPath(path)

    # -- extras -------------------------------------------------------------
    def clip_path(self, path: QPainterPath) -> None:
        self.p.save()
        self.p.setClipPath(path)


def render(fn, path, n=N):
    c = Ink(n)
    fn(c)
    c.save(path)
    return path


# --------------------------------------------------------------------------
# path helpers (device coordinates, i.e. already multiplied by n)
# --------------------------------------------------------------------------

def p_circle(c, cx, cy, r) -> QPainterPath:
    pa = QPainterPath()
    pa.addEllipse(c.pt(cx, cy), r * c.n, r * c.n)
    return pa


def p_rect(c, x, y, w, h, r=0.0) -> QPainterPath:
    pa = QPainterPath()
    rr = QRectF(x * c.n, y * c.n, w * c.n, h * c.n)
    if r > 0:
        pa.addRoundedRect(rr, r * c.n, r * c.n)
    else:
        pa.addRect(rr)
    return pa


def p_line(c, x0, y0, x1, y1) -> QPainterPath:
    pa = QPainterPath()
    pa.moveTo(c.pt(x0, y0))
    pa.lineTo(c.pt(x1, y1))
    return pa


def p_catmull(c, pts, closed=True) -> QPainterPath:
    p = [tuple(q) for q in pts]
    ext = ([p[-1]] + p + [p[0], p[1]]) if closed else ([p[0]] + p + [p[-1]])
    pa = QPainterPath()
    pa.moveTo(c.pt(*p[0]))
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6.0, p1[1] + (p2[1] - p0[1]) / 6.0)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6.0, p2[1] - (p3[1] - p1[1]) / 6.0)
        pa.cubicTo(c.pt(*c1), c.pt(*c2), c.pt(*p2))
    if closed:
        pa.closeSubpath()
    return pa


def stroked(c, path, w, cap=Qt.FlatCap) -> QPainterPath:
    """Outline ``path`` as an area, so it can take part in boolean ops."""
    st = QPainterPathStroker()
    st.setWidth(w * c.n / 1024.0)
    st.setCapStyle(cap)
    st.setJoinStyle(Qt.RoundJoin)
    return st.createStroke(path)


# --------------------------------------------------------------------------
# the house cell silhouette, inherited from the current logo_spacr.png
# --------------------------------------------------------------------------

#: (angle deg, radius) -- designed, not jittered, so it is reproducible and
#: can be tuned; screen coordinates, y down
BLOB = [
    (0, 0.394), (36, 0.375), (72, 0.328), (108, 0.302), (144, 0.344),
    (180, 0.380), (216, 0.407), (252, 0.372), (288, 0.388), (324, 0.414),
]

#: a deliberately calmer silhouette for the "regularised" variant
BLOB_SMOOTH = [
    (0, 0.386), (36, 0.378), (72, 0.360), (108, 0.348), (144, 0.360),
    (180, 0.376), (216, 0.388), (252, 0.378), (288, 0.386), (324, 0.392),
]

BCX, BCY = 0.500, 0.502
NUC = (0.548, 0.500, 0.108)     # nucleus cx, cy, r
GRID = 0.135                    # grid pitch about the centre


def blob_pts(scale=1.0, table=BLOB, cx=BCX, cy=BCY):
    out = []
    for a, r in table:
        t = math.radians(a)
        out.append((cx + scale * r * math.cos(t), cy + scale * r * math.sin(t)))
    return out


def blob_path(c, scale=1.0, table=BLOB, cx=BCX, cy=BCY) -> QPainterPath:
    return p_catmull(c, blob_pts(scale, table, cx, cy), closed=True)


def grid_path(c, pitch=GRID, cx=BCX, cy=BCY, span=0.62) -> QPainterPath:
    pa = QPainterPath()
    for k in (-1, 1):
        pa.addPath(p_line(c, cx + k * pitch, cy - span, cx + k * pitch, cy + span))
        pa.addPath(p_line(c, cx - span, cy + k * pitch, cx + span, cy + k * pitch))
    return pa


#: parasitophorous vacuole of the parent logo: cx, cy, length, rotation
VAC = (0.362, 0.336, 0.208, -20.0)


def vacuole_paths(c, cx, cy, length, rot=0.0):
    """(capsule outline, tachyzoite a, tachyzoite b) in device coordinates."""
    h = length * 0.46
    body = p_rect(c, -length / 2, -h / 2, length, h, r=h / 2)
    d1 = p_circle(c, -length * 0.19, 0.0, h * 0.21)
    d2 = p_circle(c, length * 0.17, 0.0, h * 0.19)
    t = QTransform()
    t.translate(cx * c.n, cy * c.n)
    t.rotate(rot)
    return t.map(body), t.map(d1), t.map(d2)


def vacuole_halo(c, w, pad=40.0, vac=VAC):
    """The capsule grown by ``pad``, so the grid can be cut clear of it -- the
    parent logo achieves the same separation by greying the grid down."""
    body, _a, _b = vacuole_paths(c, *vac)
    return body.united(stroked(c, body, w + 2 * pad, cap=Qt.RoundCap))


def draw_vacuole(c, w=THIN, vac=VAC):
    body, d1, d2 = vacuole_paths(c, *vac)
    c.stroke(body, w)
    c.fill(d1)
    c.fill(d2)


# ==========================================================================
# GROUP A -- 10 new concepts.  Everything here is drawn to the 16 px budget.
# ==========================================================================

def sh_c_dot(c, c_ring=None, c_dot=None):
    """One open C-membrane around one solid nucleus."""
    c.set_ink(c_ring)
    c.arc(0.500, 0.500, 0.330, 42.0, 288.0, w=HEAVY)
    c.set_ink(c_dot)
    c.disc(0.500, 0.500, 0.128)


def sh_quadrant(c, c_frame=None, c_hit=None):
    """A field split in four; exactly one cell is a hit."""
    x0, x1 = 0.150, 0.850
    m = 0.500
    g = BOLD / 1024.0          # keep the fill clear of the rules
    c.set_ink(c_hit)
    c.rect(m + g, x0 + g, x1 - m - 2 * g, m - x0 - 2 * g, filled=True, r=0.026)
    c.set_ink(c_frame)
    c.rect(x0, x0, x1 - x0, x1 - x0, w=BOLD, r=0.060)
    c.line(m, x0, m, x1, w=BOLD)
    c.line(x0, m, x1, m, w=BOLD)


def sh_orbit(c, c_orbit=None, c_dot=None):
    """One body, one ring: the smallest statement of a spatial relationship.
    The body deliberately overruns the ring on both sides so the mark cannot
    collapse into an eye at small sizes."""
    c.set_ink(c_orbit)
    c.ell(0.500, 0.500, 0.455, 0.138, rot=-31.0, w=BOLD)
    c.set_ink(c_dot)
    c.disc(0.500, 0.500, 0.225)


def sh_plate(c, c_frame=None, c_dot=None):
    """The plate: a notched A1 corner and one well called.  The only
    rectilinear silhouette in the set, so it cannot be confused with the rest."""
    x0, x1, n = 0.155, 0.845, 0.190
    c.set_ink(c_frame)
    c.polyline([(x0 + n, x0), (x1, x0), (x1, x1), (x0, x1), (x0, x0 + n)],
               w=BOLD, close=True)
    c.set_ink(c_dot)
    c.disc(0.500, 0.500, 0.172)


def sh_matrix(c, c_dot=None, c_hit=None):
    """A screen: a regular array with one well called."""
    step = 0.255
    for iy in (-1, 0, 1):
        for ix in (-1, 0, 1):
            if (ix, iy) == (1, -1):
                continue
            c.set_ink(c_dot)
            c.disc(0.500 + ix * step, 0.500 + iy * step, 0.085)
    c.set_ink(c_hit)
    c.disc(0.500 + step, 0.500 - step, 0.150)


def sh_split(c, c_solid=None, c_open=None):
    """Segmentation as one gesture: the same object, raw on one side and
    resolved on the other."""
    r = 0.320
    c.set_ink(c_solid)
    c.clip_rect(0.0, 0.0, 0.500, 1.0)
    c.disc(0.455, 0.500, r)
    c.unclip()
    c.set_ink(c_open)
    c.clip_rect(0.545, 0.0, 0.500, 1.0)
    c.circ(0.590, 0.500, r - BOLD / 2048.0, w=BOLD)
    c.unclip()


def sh_crescent(c, col=None):
    """The organism, reduced to a single shape."""
    pa = QPainterPath()
    pa.setFillRule(Qt.OddEvenFill)
    pa.addEllipse(c.pt(0.470, 0.500), 0.365 * c.n, 0.365 * c.n)
    pa.addEllipse(c.pt(0.660, 0.410), 0.330 * c.n, 0.330 * c.n)
    c.set_ink(col)
    c.fill(pa)


def sh_pin(c, c_body=None, c_hole=None):
    """A phenotype, located."""
    cx, cy, r = 0.500, 0.392, 0.252
    body = QPainterPath()
    body.addEllipse(c.pt(cx, cy), r * c.n, r * c.n)
    tail = QPainterPath()
    tail.moveTo(c.pt(cx, 0.888))
    tail.lineTo(c.pt(cx - r * 0.86, cy + r * 0.52))
    tail.lineTo(c.pt(cx + r * 0.86, cy + r * 0.52))
    tail.closeSubpath()
    body = body.united(tail)
    if c_hole is None:
        body = body.subtracted(p_circle(c, cx, cy, 0.108))
        c.set_ink(c_body)
        c.fill(body)
    else:
        c.set_ink(c_body)
        c.fill(body)
        c.set_ink(c_hole)
        c.fill(p_circle(c, cx, cy, 0.108))


def sh_cut(c, c_a=None, c_b=None):
    """The edit: one body, one clean cut, the two halves slid apart."""
    ang = -32.0
    gap = 0.070
    disc = p_circle(c, 0.500, 0.500, 0.352)
    knife = QPainterPath()
    knife.addRect(QRectF(-1.0 * c.n, 0.0, 2.0 * c.n, 1.0 * c.n))
    t = QTransform()
    t.translate(0.500 * c.n, 0.500 * c.n)
    t.rotate(ang)
    lower = t.map(knife)
    dx = gap / 2.0 * math.sin(math.radians(ang))
    dy = gap / 2.0 * math.cos(math.radians(ang))
    off = QTransform()
    off.translate(dx * c.n, dy * c.n)
    c.set_ink(c_a)
    c.fill(off.map(disc.intersected(lower)))
    off2 = QTransform()
    off2.translate(-dx * c.n, -dy * c.n)
    c.set_ink(c_b)
    c.fill(off2.map(disc.subtracted(lower)))


def sh_focus(c, c_frame=None, c_dot=None):
    """A region of interest, and the one object inside it."""
    a, b, L = 0.150, 0.850, 0.215
    c.set_ink(c_frame)
    c.polyline([(a, a + L), (a, a), (a + L, a)], w=HEAVY)
    c.polyline([(b, b - L), (b, b), (b - L, b)], w=HEAVY)
    c.set_ink(c_dot)
    c.disc(0.500, 0.500, 0.185)


# ==========================================================================
# GROUP B -- 10 refinements of the existing logo_spacr.png
# ==========================================================================

def sh_blob(c, grid=True, nucleus="solid", vacuole=False, satellite=False,
            hit=False, ticks=False, cross=False, knockout=False,
            table=BLOB, w_edge=68.0, w_grid=34.0, w_nuc=44.0, nuc=NUC,
            c_edge=None, c_grid=None, c_body=None):
    """The parent mark, parameterised.  Every group-B variant is this call
    with a different set of parts switched on."""
    bp = blob_path(c, table=table)

    if knockout:
        area = QPainterPath(bp)
        if grid:
            area = area.subtracted(stroked(c, grid_path(c), 46.0))
        if nucleus:
            area = area.subtracted(p_circle(c, *nuc))
        c.set_ink(c_edge)
        c.fill(area)
        return

    if grid:
        c.set_ink(c_grid)
        area = bp
        # cut the grid clear of the bodies, the way the parent logo separates
        # them by greying the grid down
        if nucleus == "solid":
            area = area.subtracted(p_circle(c, nuc[0], nuc[1], nuc[2] + 0.036))
        if vacuole:
            area = area.subtracted(vacuole_halo(c, w_nuc))
        c.clip_path(area)
        c.stroke(grid_path(c), w_grid)
        c.unclip()
        if hit:
            cellp = p_rect(c, BCX - GRID, BCY - GRID, 2 * GRID, 2 * GRID)
            c.fill(cellp.subtracted(p_circle(c, *nuc)))
    if cross:
        c.set_ink(c_grid)
        c.clip_path(bp)
        c.line(BCX, BCY - 0.62, BCX, BCY + 0.62, w=w_grid)
        c.line(BCX - 0.62, BCY, BCX + 0.62, BCY, w=w_grid)
        c.unclip()
    if ticks:
        band = stroked(c, bp, 150.0)
        c.set_ink(c_grid)
        c.fill(stroked(c, grid_path(c), w_grid).intersected(band))

    c.set_ink(c_edge)
    c.stroke(bp, w_edge)

    c.set_ink(c_body)
    if nucleus == "solid":
        c.disc(*nuc)
    elif nucleus == "ring":
        c.circ(nuc[0], nuc[1], nuc[2], w=w_nuc)
    if satellite:
        c.disc(0.372, 0.646, 0.058)
    if vacuole:
        draw_vacuole(c, w=w_nuc)


# ==========================================================================
# candidate table
# ==========================================================================

def _wrap(fn, **kw):
    return lambda c: fn(c, **kw)


CANDIDATES = [
    # ---- group A: 10 new concepts, black and white --------------------
    ("concept_01_c_dot", "concept", False,
     "One open C-membrane around one solid nucleus - the C of spaCR and a cell "
     "read as the same shape."),
    ("concept_02_quadrant", "concept", False,
     "A field split in four with exactly one cell filled - the screen, and its hit."),
    ("concept_03_orbit", "concept", False,
     "One body, one orbit - the smallest statement of a spatial relationship."),
    ("concept_04_plate", "concept", False,
     "A plate with its notched A1 corner and one well called - the only "
     "rectilinear form in the set."),
    ("concept_05_matrix", "concept", False,
     "A 3x3 array with one well called - the screen as a rhythm, no container."),
    ("concept_06_split", "concept", False,
     "One object, half raw and half resolved - segmentation in a single gesture."),
    ("concept_07_crescent", "concept", False,
     "The organism as one shape: a solid crescent, nothing else."),
    ("concept_08_pin", "concept", False,
     "A map pin whose counter is the cell - a phenotype, located."),
    ("concept_09_cut", "concept", False,
     "One body, one clean cut, the halves slid apart - the edit."),
    ("concept_10_focus", "concept", False,
     "Two corner brackets and the one object between them - a region of interest."),

    # ---- group B: 10 refinements of the current logo ------------------
    ("variant_01_grid_nucleus", "variant", False,
     "The parent stripped to three parts: cell outline, spatial grid, solid nucleus."),
    ("variant_02_grid_vacuole", "variant", False,
     "As 11, plus the one parasitophorous vacuole - keeps the biology, drops the rest."),
    ("variant_03_bare", "variant", False,
     "Cell outline and nucleus only, drawn heavy - the parent with the grid removed."),
    ("variant_04_cross", "variant", False,
     "The grid reduced to a single cross - two lines instead of four."),
    ("variant_05_solid", "variant", False,
     "The cell as solid mass with the nucleus knocked out of it."),
    ("variant_06_solid_grid", "variant", False,
     "Solid cell with the grid and nucleus knocked out as negative space."),
    ("variant_07_satellite", "variant", False,
     "Outline, nucleus and one satellite - the parent's asymmetry without its grid."),
    ("variant_08_regular", "variant", False,
     "The silhouette regularised: the same cell, calmer, with grid and nucleus."),
    ("variant_09_grid_hit", "variant", False,
     "One square of the spatial grid filled solid - the hit, inside the cell."),
    ("variant_10_ticks", "variant", False,
     "The grid implied by ticks straddling the membrane rather than drawn across it."),

    # ---- group C: 1 thinner-line variant ------------------------------
    ("thin_01_grid_nucleus", "thin", False,
     "Variant 11 redrawn at roughly 60% stroke weight throughout."),

    # ---- group D: 9 colour variants -----------------------------------
    ("colour_01_teal_c_dot", "colour", True,
     "Single colour: concept 1 in one teal."),
    ("colour_02_teal_blob", "colour", True,
     "Single colour: variant 11 in one teal."),
    ("colour_03_slate_coral_blob", "colour", True,
     "Two colours: slate membrane and grid, coral nucleus."),
    ("colour_04_teal_coral_c_dot", "colour", True,
     "Two colours: teal membrane, coral nucleus."),
    ("colour_05_indigo_orbit", "colour", True,
     "Single colour: concept 3 in one indigo."),
    ("colour_06_slate_coral_matrix", "colour", True,
     "Two colours: slate array, coral hit."),
    ("colour_07_teal_indigo_split", "colour", True,
     "Two colours: teal solid half, indigo resolved half."),
    ("colour_08_coral_crescent", "colour", True,
     "Single colour: concept 7 in one coral."),
    ("colour_09_trio_blob", "colour", True,
     "Three colours: teal membrane, slate grid, coral nucleus and vacuole."),
]

DRAW = {
    "concept_01_c_dot": _wrap(sh_c_dot),
    "concept_02_quadrant": _wrap(sh_quadrant),
    "concept_03_orbit": _wrap(sh_orbit),
    "concept_04_plate": _wrap(sh_plate),
    "concept_05_matrix": _wrap(sh_matrix),
    "concept_06_split": _wrap(sh_split),
    "concept_07_crescent": _wrap(sh_crescent),
    "concept_08_pin": _wrap(sh_pin),
    "concept_09_cut": _wrap(sh_cut),
    "concept_10_focus": _wrap(sh_focus),

    "variant_01_grid_nucleus": _wrap(sh_blob),
    "variant_02_grid_vacuole": _wrap(sh_blob, vacuole=True),
    "variant_03_bare": _wrap(sh_blob, grid=False, w_edge=HEAVY),
    "variant_04_cross": _wrap(sh_blob, grid=False, cross=True, w_grid=MED),
    "variant_05_solid": _wrap(sh_blob, grid=False, knockout=True,
                              nuc=(0.548, 0.500, 0.130)),
    "variant_06_solid_grid": _wrap(sh_blob, knockout=True,
                                   nuc=(0.500, 0.502, 0.098)),
    "variant_07_satellite": _wrap(sh_blob, grid=False, satellite=True, w_edge=HEAVY),
    "variant_08_regular": _wrap(sh_blob, table=BLOB_SMOOTH),
    "variant_09_grid_hit": _wrap(sh_blob, hit=True, nucleus=None),
    "variant_10_ticks": _wrap(sh_blob, grid=False, ticks=True, w_grid=MED,
                              w_edge=BOLD),

    "thin_01_grid_nucleus": _wrap(sh_blob, w_edge=68.0 * 0.60, w_grid=34.0 * 0.60,
                                  w_nuc=44.0 * 0.60),

    "colour_01_teal_c_dot": _wrap(sh_c_dot, c_ring=TEAL, c_dot=TEAL),
    "colour_02_teal_blob": _wrap(sh_blob, c_edge=TEAL, c_grid=TEAL, c_body=TEAL),
    "colour_03_slate_coral_blob": _wrap(sh_blob, c_edge=SLATE, c_grid=SLATE,
                                        c_body=CORAL),
    "colour_04_teal_coral_c_dot": _wrap(sh_c_dot, c_ring=TEAL, c_dot=CORAL),
    "colour_05_indigo_orbit": _wrap(sh_orbit, c_orbit=INDIGO, c_dot=INDIGO),
    "colour_06_slate_coral_matrix": _wrap(sh_matrix, c_dot=SLATE, c_hit=CORAL),
    "colour_07_teal_indigo_split": _wrap(sh_split, c_solid=TEAL, c_open=INDIGO),
    "colour_08_coral_crescent": _wrap(sh_crescent, col=CORAL),
    "colour_09_trio_blob": _wrap(sh_blob, vacuole=True, c_edge=TEAL, c_grid=SLATE,
                                 c_body=CORAL),
}

GROUP_TITLE = {
    "concept": "A. new concepts (black & white)",
    "variant": "B. refinements of the existing logo (black & white)",
    "thin": "C. thinner lines (black & white)",
    "colour": "D. colour",
}


# ==========================================================================
# output
# ==========================================================================

def _lin(v):
    v /= 255.0
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def _lum(hexcol):
    col = QColor(hexcol)
    return (0.2126 * _lin(col.red()) + 0.7152 * _lin(col.green())
            + 0.0722 * _lin(col.blue()))


def _ratio(a, b):
    la, lb = _lum(a), _lum(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _tint(img, ink):
    out = QImage(img.size(), QImage.Format_ARGB32_Premultiplied)
    out.fill(Qt.transparent)
    p = QPainter(out)
    p.drawImage(0, 0, img)
    p.setCompositionMode(QPainter.CompositionMode_SourceIn)
    p.fillRect(out.rect(), QColor(ink))
    p.end()
    return out


def sheet(entries, images, out_path, bg, cols=5, cell=300, pad=24, label_h=46):
    """Contact sheet: every candidate at full size, plus a 48 px inset.

    White artwork is re-inked on the light sheet (it would otherwise be
    invisible); colour artwork is drawn exactly as it ships, because the whole
    point of the colour group is that one file works on both backgrounds.
    """
    rows = (len(images) + cols - 1) // cols
    head = 64
    w = cols * cell + (cols + 1) * pad
    h = head + rows * (cell + label_h) + (rows + 1) * pad
    s = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
    s.fill(QColor(bg))
    p = QPainter(s)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setRenderHint(QPainter.SmoothPixmapTransform, True)
    dark = QColor(bg).lightnessF() < 0.5
    ink = QColor(255, 255, 255) if dark else QColor(20, 22, 26)
    faint = QColor(255, 255, 255, 46) if dark else QColor(20, 22, 26, 46)

    nf = QFont()
    nf.setPixelSize(25)
    p.setFont(nf)
    p.setPen(QPen(ink))
    note = ("spaCR logo candidates - 30 in four groups. "
            + ("dark background, artwork as it ships"
               if dark else "light background; the white candidates are "
                            "re-inked dark, the colour ones are untouched"))
    p.drawText(QRectF(pad, 10, w - 2 * pad, head), Qt.AlignLeft | Qt.AlignVCenter, note)

    lf = QFont()
    lf.setPixelSize(19)
    lf.setBold(True)
    for i, img in enumerate(images):
        name, group, is_colour, _desc = entries[i]
        r, cix = divmod(i, cols)
        x = pad + cix * (cell + pad)
        y = head + pad + r * (cell + label_h + pad)
        p.setPen(QPen(faint, 2))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(QRectF(x - 6, y - 6, cell + 12, cell + 12), 12, 12)
        big = img.scaled(cell, cell, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        small = img.scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        if not dark and not is_colour:
            big, small = _tint(big, ink), _tint(small, ink)
        p.drawImage(int(x), int(y), big)
        p.setPen(QPen(faint, 2))
        p.drawRect(QRectF(x + cell - 54, y + cell - 54, 52, 52))
        p.drawImage(int(x + cell - 53), int(y + cell - 53), small)
        p.setFont(lf)
        p.setPen(QPen(ink))
        p.drawText(QRectF(x, y + cell + 2, cell, label_h),
                   Qt.AlignHCenter | Qt.AlignVCenter, "%02d  %s" % (i + 1, name))
    p.end()
    s.save(out_path)
    return out_path


def small_sheet(entries, images, out_path, bg, zoom=4, cols=3):
    """16 / 32 / 48 px, nearest-neighbour zoomed.  This is the real test."""
    sizes = (16, 32, 48)
    strip_w = sum(sz * zoom for sz in sizes) + 2 * 18
    cell_h = 48 * zoom
    lab = 210
    colw = lab + strip_w + 34
    rows = (len(images) + cols - 1) // cols
    head = 56
    w = cols * colw + 24
    h = head + rows * (cell_h + 22) + 24
    s = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
    s.fill(QColor(bg))
    p = QPainter(s)
    p.setRenderHint(QPainter.Antialiasing, True)
    dark = QColor(bg).lightnessF() < 0.5
    ink = QColor(255, 255, 255) if dark else QColor(20, 22, 26)
    nf = QFont()
    nf.setPixelSize(24)
    p.setFont(nf)
    p.setPen(QPen(ink))
    p.drawText(QRectF(20, 8, w - 40, head), Qt.AlignLeft | Qt.AlignVCenter,
               "16 / 32 / 48 px, zoomed %dx nearest-neighbour - %s background"
               % (zoom, "dark" if dark else "light"))
    lf = QFont()
    lf.setPixelSize(17)
    for i, img in enumerate(images):
        name, group, is_colour, _d = entries[i]
        r, cix = divmod(i, cols)
        x0 = 20 + cix * colw
        y0 = head + 12 + r * (cell_h + 22)
        p.setFont(lf)
        p.setPen(QPen(ink))
        p.drawText(QRectF(x0, y0, lab - 10, cell_h),
                   Qt.AlignLeft | Qt.AlignVCenter, "%02d %s" % (i + 1, name))
        x = x0 + lab
        for sz in sizes:
            t = img.scaled(sz, sz, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            if not dark and not is_colour:
                t = _tint(t, ink)
            big = t.scaled(sz * zoom, sz * zoom, Qt.KeepAspectRatio,
                           Qt.FastTransformation)
            p.drawImage(int(x), int(y0 + (cell_h - sz * zoom) / 2), big)
            x += sz * zoom + 18
    p.end()
    s.save(out_path)
    return out_path


def _coverage(path):
    img = QImage(path).convertToFormat(QImage.Format_ARGB32)
    buf = bytes(img.constBits())
    return sum(buf[3::4]) / (255.0 * img.width() * img.height())


def main(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = argv[1] if len(argv) > 1 else os.path.normpath(
        os.path.join(here, "..", "logo_spacr"))
    os.makedirs(outdir, exist_ok=True)

    images = []
    for name, group, is_colour, _desc in CANDIDATES:
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

    print("\ncolour contrast (ink vs each background):")
    for label, col in (("TEAL", TEAL), ("INDIGO", INDIGO), ("CORAL", CORAL),
                       ("SLATE", SLATE), ("VIOLET", VIOLET), ("AMBER", AMBER)):
        print("  %-7s %s  lum %.3f   dark %.2f:1   light %.2f:1"
              % (label, col, _lum(col), _ratio(col, DARK_BG), _ratio(col, LIGHT_BG)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
