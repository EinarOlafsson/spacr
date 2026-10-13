#!/usr/bin/env python3
"""Candidate spaCR icons: the eight data in/out apps.

Ten conceptually different designs per icon, white-on-transparent flat
vector art in the house style set by ``plaque.png`` / ``measure.png``.

These eight apps sit dangerously close together -- three of them take
data *into* spaCR and two push data *out* -- so a generic "arrow into a
folder" would fit any of them and therefore identifies none of them.
Every design below is pinned to the one thing its app actually does:

* **align** is about **geometry**: overlapping tiles snapping into
  register, seams closing, one huge canvas assembled from small fields.
  Never a generic import -- if the picture would work with the tiles
  replaced by files, it has failed.
* **foreign** is about **column mapping**: someone else's schema being
  remapped onto spaCR's. The mapping *is* the picture -- wires, a
  crosswalk grid, a socket, a swapped header row. The images coming
  along for the ride are never the subject.
* **external_masks** is about **masks that already exist** arriving
  beside the images: the labels were drawn elsewhere, spaCR only
  measures them. The segmentation step is skipped, not performed --
  which is exactly what separates this from the ``mask`` app.
* **illumination** is about **an uneven field being flattened**: a
  vignette, a corner falloff, a profile curve going straight, a
  division. The unevenness has to be visible or the fix means nothing.
* **data_manager** is about **disk cost and reclaiming it**: sizes,
  a breakdown, space freed while the originals survive.
* **db_browser** is about **a table you can query**: rows, columns,
  a filter, a selection, an export.
* **anndata_export** is about **the h5ad matrix layout**: obs x var,
  a cells-by-features matrix leaving for scanpy.
* **methods_export** is about **prose whose numbers are traced back**:
  a manuscript paragraph tied by leader lines to the run that produced
  its figures.

Because no colour and no shading are allowed, brightness is drawn as
geometry: an uneven field is a set of contour rings or a lattice of
spots that shrink outward, never a grey ramp.

Run standalone (deterministic -- no random draws at all):

    python group_data_io.py [OUTDIR]

Default OUTDIR is the backup_icons directory one level up. Writes
``<OUTDIR>/<key>/<key>_NN.png`` plus CONCEPTS.md and the two contact
sheets, via the shared output stage in ``_emit``. It never touches
anything in ``spacr/resources/icons/*.png``.
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

def dashed_rect(c, x, y, w, h, w_=W_FINE, dash=(6, 7)):
    """A rectangle in dashes (``Cv.rect`` cannot dash)."""
    c.polyline([(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
               w=w_, close=True, dash=list(dash))


def rot_rect(c, cx, cy, w, h, deg, w_=W_SEC):
    """A rectangle rotated about its own centre."""
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    pts = [(cx + dx * ca - dy * sa, cy + dx * sa + dy * ca)
           for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2),
                          (w / 2, h / 2), (-w / 2, h / 2))]
    c.polyline(pts, w=w_, close=True)


def grid_lines(c, x, y, w, h, cols, rows, w_=W_FINE, dash=None):
    """Interior rules of a grid, without its outer frame."""
    for i in range(1, cols):
        c.line(x + w * i / cols, y, x + w * i / cols, y + h, w_, dash)
    for j in range(1, rows):
        c.line(x, y + h * j / rows, x + w, y + h * j / rows, w_, dash)


def table(c, x, y, w, h, cols, rows, w_=W_SEC, head=True):
    """A data table: outer frame, interior rules, solid header band."""
    c.rect(x, y, w, h, w=w_)
    grid_lines(c, x, y, w, h, cols, rows)
    if head:
        c.rect(x, y, w, h / rows, filled=True)


def cell_fill(c, x, y, w, h, cols, rows, cells, inset=0.012):
    """Fill named (col, row) cells of a grid solid."""
    for i, j in cells:
        c.rect(x + w * i / cols + inset, y + h * j / rows + inset,
               w / cols - 2 * inset, h / rows - 2 * inset, filled=True)


def sheet(c, x, y, w, h, fold=0.13, w_=W_SEC):
    """A page with a folded corner."""
    c.polyline([(x, y), (x + w - fold, y), (x + w, y + fold),
                (x + w, y + h), (x, y + h)], w=w_, close=True)
    c.polyline([(x + w - fold, y), (x + w - fold, y + fold),
                (x + w, y + fold)], w=W_FINE)


def text_lines(c, x, y, w, n, gap, w_=W_SEC, last=0.55):
    """Body copy abstracted to a stack of rules."""
    for i in range(n):
        c.line(x, y + i * gap, x + (w if i < n - 1 else w * last),
               y + i * gap, w_)


def folder(c, x, y, w, h, w_=W_SEC):
    """A folder with a raised tab on the left."""
    t = h * 0.20
    c.polyline([(x, y + h), (x, y), (x + w * 0.40, y),
                (x + w * 0.48, y + t), (x + w, y + t), (x + w, y + h)],
               w=w_, close=True)


def cylinder(c, x, y, w, h, w_=W_SEC):
    """A database drum: top ellipse, straight sides, bulged base."""
    ry, cx = w * 0.17, x + w / 2
    c.ell(cx, y + ry, w / 2, ry, 0, w_)
    c.line(x, y + ry, x, y + h - ry, w_)
    c.line(x + w, y + ry, x + w, y + h - ry, w_)
    c.smooth([(x, y + h - ry), (cx, y + h), (x + w, y + h - ry)], w=w_)


def ring_field(c, cx, cy, n=4, r0=0.06, dr=0.075, w=W_FINE):
    """Contour rings of an uneven illumination field."""
    for i in range(n):
        c.circ(cx, cy, r0 + i * dr, w)


def tick(c, x, y, s=0.05, w=W_SEC):
    """A check mark."""
    c.polyline([(x - s, y), (x - s * 0.25, y + s * 0.75),
                (x + s, y - s * 0.85)], w=w)


def cross(c, x, y, s=0.045, w=W_SEC):
    """An X mark."""
    c.line(x - s, y - s, x + s, y + s, w)
    c.line(x - s, y + s, x + s, y - s, w)


def blobs(c, spots, filled):
    """A field of objects, either outlined cells or solid labels."""
    for cx, cy, r in spots:
        if filled:
            c.ell(cx, cy, r, r * 0.82, 0, filled=True)
        else:
            c.cell(cx, cy, r, r * 0.82, 0, w=W_FINE, nuc=0.34)


def caliper(c, x1, x2, y, w=W_FINE, tick_h=0.035, head=0.045):
    """A horizontal dimension with end stops."""
    c.arrow(x1, y, x2, y, w, head=head)
    c.arrow(x2, y, x1, y, w, head=head)
    c.line(x1, y - tick_h, x1, y + tick_h, w)
    c.line(x2, y - tick_h, x2, y + tick_h, w)


# =====================================================================
# align -- overlapping tiles snapping into one stitched canvas
# =====================================================================

def align_01(c):
    """Two camera fields overlapping, the strip they share filled solid."""
    c.rect(0.04, 0.08, 0.54, 0.58, w=W_MAIN, r=0.03)
    c.rect(0.42, 0.34, 0.54, 0.58, w=W_MAIN, r=0.03)
    c.rect(0.43, 0.35, 0.14, 0.30, filled=True)


def align_02(c):
    """The shift measured from one speck to the same speck in the next tile."""
    c.rect(0.06, 0.10, 0.50, 0.50, w=W_SEC, r=0.03)
    dashed_rect(c, 0.36, 0.38, 0.50, 0.50, W_SEC)
    c.disc(0.34, 0.34, 0.045)
    c.disc(0.64, 0.62, 0.045)
    c.arrow(0.36, 0.36, 0.615, 0.595, W_MAIN, head=0.075)


def align_03(c):
    """One peak on the shift map: crosshairs on the offset that won."""
    c.rect(0.16, 0.16, 0.68, 0.68, w=W_SEC)
    grid_lines(c, 0.16, 0.16, 0.68, 0.68, 4, 4)
    cell_fill(c, 0.16, 0.16, 0.68, 0.68, 4, 4, [(2, 1)], inset=0.018)
    c.line(0.50, 0.04, 0.50, 0.96, W_FINE)
    c.line(0.04, 0.335, 0.96, 0.335, W_FINE)


def align_04(c):
    """A registration target where two tile corners have to land together."""
    c.rect(0.05, 0.05, 0.55, 0.55, w=W_SEC, r=0.03)
    dashed_rect(c, 0.40, 0.40, 0.55, 0.55, W_SEC)
    c.circ(0.50, 0.50, 0.145, W_MAIN)
    c.disc(0.50, 0.50, 0.050)
    c.line(0.50, 0.28, 0.50, 0.72, W_SEC)
    c.line(0.28, 0.50, 0.72, 0.50, W_SEC)


def align_05(c):
    """The stage's path strung through the tile centres in visiting order."""
    c.rect(0.10, 0.16, 0.80, 0.68, w=W_SEC)
    grid_lines(c, 0.10, 0.16, 0.80, 0.68, 3, 3)
    xs = [0.10 + 0.80 * (i + 0.5) / 3 for i in range(3)]
    ys = [0.16 + 0.68 * (j + 0.5) / 3 for j in range(3)]
    path = []
    for j, y in enumerate(ys):
        row = xs if j % 2 == 0 else xs[::-1]
        path.extend([(x, y) for x in row])
    c.polyline(path[:-1], w=W_MAIN)
    c.arrow(path[-2][0], path[-2][1], path[-1][0], path[-1][1],
            W_MAIN, head=0.070)
    c.disc(path[0][0], path[0][1], 0.038)


def align_06(c):
    """A cell cut in half by the seam, whole again once the halves register."""
    c.rect(0.03, 0.22, 0.45, 0.56, w=W_SEC)
    c.rect(0.52, 0.22, 0.45, 0.56, w=W_SEC)
    c.line(0.50, 0.18, 0.50, 0.82, W_FINE, dash=[7, 8])
    c.circ(0.50, 0.50, 0.195, W_MAIN)
    c.disc(0.50, 0.50, 0.070)


def align_07(c):
    """The mosaic filling in tile by tile, one tile held at a time."""
    c.rect(0.08, 0.34, 0.84, 0.60, w=W_SEC)
    grid_lines(c, 0.08, 0.34, 0.84, 0.60, 4, 3)
    cell_fill(c, 0.08, 0.34, 0.84, 0.60, 4, 3,
              [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1)], inset=0.014)
    c.rect(0.50, 0.05, 0.19, 0.15, w=W_SEC, r=0.02)
    c.arrow(0.595, 0.22, 0.595, 0.42, W_MAIN, head=0.065)


def align_08(c):
    """Corner brackets pulling a loose tile square onto the grid."""
    dashed_rect(c, 0.26, 0.30, 0.46, 0.46, W_FINE)
    rot_rect(c, 0.55, 0.46, 0.44, 0.44, -11, W_MAIN)
    b, s = 0.10, 0.05
    for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        x = 0.49 - sx * (0.23 + s)
        y = 0.53 - sy * (0.23 + s)
        c.line(x, y, x + sx * b, y, W_SEC)
        c.line(x, y, x, y + sy * b, W_SEC)


def align_09(c):
    """A heap of skewed tiles above, squared into a clean grid below."""
    rot_rect(c, 0.30, 0.19, 0.30, 0.26, -18, W_SEC)
    rot_rect(c, 0.66, 0.23, 0.30, 0.26, 14, W_SEC)
    c.arrow(0.50, 0.40, 0.50, 0.55, W_MAIN, head=0.065)
    c.rect(0.18, 0.60, 0.64, 0.34, w=W_MAIN)
    grid_lines(c, 0.18, 0.60, 0.64, 0.34, 2, 2, W_SEC)


def align_10(c):
    """Tiles laid down one after another, every overlap solid where they meet."""
    for i in range(4):
        c.rect(0.06 + i * 0.19, 0.06 + i * 0.17, 0.34, 0.34, w=W_SEC, r=0.03)
    for i in range(3):
        c.rect(0.06 + (i + 1) * 0.19, 0.06 + (i + 1) * 0.17, 0.15, 0.17,
               filled=True)


# =====================================================================
# foreign -- their column names remapped onto spaCR's
# =====================================================================

def foreign_01(c):
    """Their column names on the left wired across to ours on the right."""
    ys = (0.20, 0.50, 0.80)
    for i, y in enumerate(ys):
        c.bar(0.04, y - 0.055, 0.26, 0.11, filled=True)
        c.bar(0.70, y - 0.055, 0.26, 0.11, filled=False, w=W_SEC)
    for a, b in ((0, 1), (1, 2), (2, 0)):
        c.line(0.32, ys[a], 0.68, ys[b], W_SEC)


def foreign_02(c):
    """A foreign table with its header row lifted off and ours dropped in."""
    c.rect(0.12, 0.42, 0.76, 0.52, w=W_SEC)
    grid_lines(c, 0.12, 0.42, 0.76, 0.52, 3, 2)
    dashed_rect(c, 0.12, 0.42, 0.76, 0.16, W_FINE)
    c.rect(0.12, 0.04, 0.76, 0.16, filled=True)
    c.arrow(0.50, 0.23, 0.50, 0.39, W_MAIN, head=0.060)


def foreign_03(c):
    """A foreign-shaped plug machined to fit spaCR's socket."""
    c.polyline([(0.04, 0.30), (0.40, 0.30), (0.40, 0.38), (0.52, 0.38),
                (0.52, 0.48), (0.40, 0.48), (0.40, 0.56), (0.52, 0.56),
                (0.52, 0.66), (0.40, 0.66), (0.40, 0.74), (0.04, 0.74)],
               w=W_SEC, close=True)
    c.polyline([(0.96, 0.24), (0.62, 0.24), (0.62, 0.34), (0.74, 0.34),
                (0.74, 0.44), (0.62, 0.44), (0.62, 0.60), (0.74, 0.60),
                (0.74, 0.70), (0.62, 0.70), (0.62, 0.80), (0.96, 0.80)],
               w=W_MAIN, close=True)


def foreign_04(c):
    """A crosswalk grid, ticked where their column answers to ours."""
    for i in range(3):
        c.bar(0.30 + i * 0.22, 0.05, 0.18, 0.08, filled=True)
        c.bar(0.05, 0.28 + i * 0.22, 0.18, 0.08, filled=False, w=W_SEC)
    c.rect(0.28, 0.20, 0.68, 0.68, w=W_SEC)
    grid_lines(c, 0.28, 0.20, 0.68, 0.68, 3, 3)
    for i, j in ((1, 0), (2, 1), (0, 2)):
        tick(c, 0.28 + 0.68 * (i + 0.5) / 3, 0.20 + 0.68 * (j + 0.5) / 3,
             0.062, W_SEC)


def foreign_05(c):
    """Their column heads dropping into the holes whose shape they match."""
    c.disc(0.20, 0.14, 0.075)
    c.rect(0.425, 0.065, 0.15, 0.15, filled=True)
    c.polyline([(0.80, 0.06), (0.885, 0.21), (0.715, 0.21)],
               close=True, filled=True)
    for x in (0.20, 0.50, 0.80):
        c.arrow(x, 0.28, x, 0.45, W_FINE, head=0.050)
    c.rect(0.06, 0.52, 0.88, 0.36, w=W_MAIN, r=0.03)
    c.circ(0.20, 0.70, 0.080, W_SEC)
    c.rect(0.418, 0.618, 0.164, 0.164, w=W_SEC)
    c.polyline([(0.80, 0.61), (0.892, 0.78), (0.708, 0.78)],
               w=W_SEC, close=True)


def foreign_06(c):
    """Ragged foreign headings funnelled into evenly cut spaCR columns."""
    for i, (x, w) in enumerate(((0.06, 0.30), (0.40, 0.20), (0.64, 0.30))):
        c.bar(x, 0.08, w, 0.10, filled=True)
    c.polyline([(0.04, 0.28), (0.96, 0.28), (0.62, 0.56), (0.38, 0.56)],
               w=W_MAIN, close=True)
    for i in range(3):
        c.rect(0.24 + i * 0.19, 0.70, 0.13, 0.26, w=W_SEC, r=0.02)


def foreign_07(c):
    """One column lifted out of their table and dropped in our empty slot."""
    c.rect(0.04, 0.30, 0.40, 0.64, w=W_SEC)
    grid_lines(c, 0.04, 0.30, 0.40, 0.64, 3, 3)
    cell_fill(c, 0.04, 0.30, 0.40, 0.64, 3, 1, [(1, 0)], inset=0.012)
    c.rect(0.56, 0.30, 0.40, 0.64, w=W_SEC)
    grid_lines(c, 0.56, 0.30, 0.40, 0.64, 3, 3)
    dashed_rect(c, 0.69, 0.31, 0.134, 0.62, W_FINE)
    c.smooth([(0.24, 0.26), (0.44, 0.06), (0.66, 0.14)], w=W_MAIN)
    c.arrow(0.62, 0.11, 0.755, 0.26, W_MAIN, head=0.065, tail=False)


def foreign_08(c):
    """A rename tag tied onto the one column it re-labels."""
    c.rect(0.04, 0.30, 0.52, 0.64, w=W_SEC)
    grid_lines(c, 0.04, 0.30, 0.52, 0.64, 3, 3)
    cell_fill(c, 0.04, 0.30, 0.52, 0.64, 3, 1, [(2, 0)], inset=0.012)
    c.smooth([(0.47, 0.30), (0.53, 0.18), (0.62, 0.14)], w=W_SEC)
    c.polyline([(0.60, 0.14), (0.74, 0.03), (0.98, 0.03), (0.98, 0.25),
                (0.74, 0.25)], w=W_MAIN, close=True)
    c.disc(0.79, 0.14, 0.030)
    c.line(0.86, 0.14, 0.94, 0.14, W_SEC)


def foreign_09(c):
    """An unclaimed column, with the spaCR names it could be renamed to."""
    c.rect(0.06, 0.16, 0.36, 0.78, w=W_SEC)
    grid_lines(c, 0.06, 0.16, 0.36, 0.78, 2, 4)
    dashed_rect(c, 0.24, 0.17, 0.17, 0.76, W_FINE)
    c.arrow(0.46, 0.34, 0.58, 0.34, W_SEC, head=0.055)
    for i, y in enumerate((0.22, 0.48, 0.74)):
        c.bar(0.62, y, 0.32, 0.16, filled=(i == 0), w=W_SEC)


def foreign_10(c):
    """A stamp pressing spaCR's headings onto someone else's sheet."""
    c.rect(0.24, 0.06, 0.52, 0.20, w=W_MAIN, r=0.03)
    c.bar(0.44, 0.26, 0.12, 0.14, filled=True)
    c.rect(0.30, 0.40, 0.40, 0.12, filled=True)
    c.arrow(0.14, 0.30, 0.14, 0.52, W_SEC, head=0.055)
    c.arrow(0.86, 0.30, 0.86, 0.52, W_SEC, head=0.055)
    c.rect(0.10, 0.56, 0.80, 0.38, w=W_SEC)
    grid_lines(c, 0.10, 0.56, 0.80, 0.38, 3, 2)


# =====================================================================
# external_masks -- labels drawn elsewhere, measured here
# =====================================================================

def external_masks_01(c):
    """The pair that arrives: a field of cells, and its labels already solid."""
    spots = ((0.28, 0.20, 0.095), (0.60, 0.14, 0.075), (0.52, 0.34, 0.085))
    c.rect(0.10, 0.04, 0.80, 0.42, w=W_SEC, r=0.03)
    blobs(c, spots, False)
    c.rect(0.10, 0.54, 0.80, 0.42, w=W_SEC, r=0.03)
    blobs(c, [(x, y + 0.50, r) for x, y, r in spots], True)


def external_masks_02(c):
    """The segmentation step struck out: images and labels go straight to measure."""
    c.rect(0.02, 0.44, 0.28, 0.34, w=W_SEC, r=0.03)
    c.disc(0.16, 0.61, 0.082)
    c.circ(0.50, 0.61, 0.155, W_SEC)
    cross(c, 0.50, 0.61, 0.125, W_MAIN)
    c.rect(0.70, 0.44, 0.28, 0.34, w=W_SEC, r=0.03)
    c.line(0.75, 0.56, 0.87, 0.56, W_SEC)
    c.line(0.75, 0.67, 0.93, 0.67, W_SEC)
    c.smooth([(0.16, 0.38), (0.50, 0.08), (0.84, 0.38)], w=W_MAIN)
    c.arrow(0.74, 0.26, 0.84, 0.40, W_MAIN, head=0.065, tail=False)


def external_masks_03(c):
    """An outline sheet drawn elsewhere, dropped onto the image beneath it."""
    c.rect(0.04, 0.36, 0.56, 0.56, w=W_SEC)
    c.ell(0.24, 0.62, 0.135, 0.110, 0, filled=True)
    c.ell(0.46, 0.83, 0.095, 0.078, 0, filled=True)
    c.rect(0.36, 0.08, 0.56, 0.56, w=W_MAIN)
    c.ell(0.56, 0.34, 0.135, 0.110, 0, W_SEC)
    c.ell(0.78, 0.55, 0.095, 0.078, 0, W_SEC)
    c.arrow(0.30, 0.10, 0.16, 0.26, W_SEC, head=0.060)


def external_masks_04(c):
    """Labels that arrive already numbered, each object carrying its own id."""
    c.rect(0.08, 0.18, 0.84, 0.74, w=W_SEC, r=0.03)
    for cx, cy, r in ((0.32, 0.42, 0.135), (0.68, 0.38, 0.110),
                      (0.50, 0.72, 0.125)):
        c.ell(cx, cy, r, r * 0.84, 0, filled=True)
        c.circ(cx, cy, r * 0.34, W_SEC)
    c.arrow(0.14, 0.04, 0.32, 0.22, W_MAIN, head=0.070)


def external_masks_05(c):
    """A label from somewhere else flying in and landing on the cell it fits."""
    c.rect(0.10, 0.36, 0.60, 0.58, w=W_SEC, r=0.03)
    c.cell(0.40, 0.66, 0.185, 0.155, 0, w=W_MAIN, nuc=0.34)
    c.ell(0.78, 0.20, 0.130, 0.108, -22, filled=True)
    c.smooth([(0.74, 0.30), (0.60, 0.44), (0.50, 0.52)], w=W_FINE,
             dash=[6, 7])
    c.arrow(0.55, 0.48, 0.455, 0.585, W_SEC, head=0.058, tail=False)


def external_masks_06(c):
    """Every image tile matched to the mask file that came with it."""
    for x in (0.10, 0.56):
        c.rect(x, 0.06, 0.34, 0.34, w=W_SEC, r=0.03)
        c.cell(x + 0.17, 0.23, 0.105, 0.086, 0, w=W_SEC, nuc=0.36)
        c.rect(x, 0.60, 0.34, 0.34, w=W_SEC, r=0.03)
        c.ell(x + 0.17, 0.77, 0.105, 0.086, 0, filled=True)
        c.line(x + 0.17, 0.42, x + 0.17, 0.58, W_SEC, dash=[6, 7])


def external_masks_07(c):
    """A ready-made label being measured, not drawn: calipers across a blob."""
    c.ell(0.48, 0.44, 0.320, 0.260, -8, filled=True)
    caliper(c, 0.16, 0.80, 0.80, W_SEC, tick_h=0.05, head=0.055)
    c.line(0.16, 0.68, 0.16, 0.86, W_SEC)
    c.line(0.80, 0.68, 0.80, 0.86, W_SEC)


def external_masks_08(c):
    """Two layers handed over together: pixels below, their label map on top."""
    for dy, filled in ((0.00, True), (0.34, False)):
        c.polyline([(0.50, 0.06 + dy), (0.94, 0.28 + dy),
                    (0.50, 0.50 + dy), (0.06, 0.28 + dy)],
                   w=W_MAIN if filled else W_SEC, close=True)
    c.ell(0.50, 0.28, 0.150, 0.075, 0, filled=True)
    c.ell(0.50, 0.62, 0.150, 0.075, 0, W_SEC)
    c.line(0.50, 0.34, 0.50, 0.54, W_FINE, dash=[5, 6])


def external_masks_09(c):
    """Supplied labels going straight into the measurement table."""
    c.rect(0.02, 0.26, 0.40, 0.48, w=W_SEC, r=0.03)
    c.ell(0.16, 0.42, 0.090, 0.075, 0, filled=True)
    c.ell(0.31, 0.60, 0.075, 0.062, 0, filled=True)
    c.arrow(0.44, 0.50, 0.56, 0.50, W_MAIN, head=0.065)
    table(c, 0.58, 0.26, 0.40, 0.48, 2, 4)


def external_masks_10(c):
    """An image and its label map travelling together into a spaCR project."""
    c.rect(0.04, 0.16, 0.34, 0.30, w=W_SEC, r=0.025)
    c.cell(0.21, 0.31, 0.090, 0.074, 0, w=W_FINE, nuc=0.36)
    c.rect(0.04, 0.54, 0.34, 0.30, w=W_SEC, r=0.025)
    c.ell(0.21, 0.69, 0.090, 0.074, 0, filled=True)
    c.arrow(0.42, 0.50, 0.58, 0.50, W_MAIN, head=0.070)
    folder(c, 0.62, 0.26, 0.34, 0.48)


# =====================================================================
# illumination -- an uneven field flattened
# =====================================================================

def illumination_01(c):
    """A vignetted field's contour rings, corrected to a field with none."""
    c.rect(0.03, 0.28, 0.38, 0.44, w=W_SEC, r=0.02)
    c.clip_rect(0.03, 0.28, 0.38, 0.44)
    ring_field(c, 0.22, 0.50, 4, 0.055, 0.070, W_SEC)
    c.unclip()
    c.arrow(0.45, 0.50, 0.57, 0.50, W_MAIN, head=0.060)
    c.rect(0.61, 0.28, 0.38, 0.44, w=W_MAIN, r=0.02)


def illumination_02(c):
    """The intensity profile across the field, pulled down onto a level line."""
    c.axes(0.10, 0.08, 0.94, 0.86, W_SEC)
    c.smooth([(0.14, 0.62), (0.32, 0.30), (0.50, 0.20), (0.68, 0.30),
              (0.90, 0.62)], w=W_MAIN)
    c.line(0.14, 0.68, 0.90, 0.68, W_MAIN)
    for x in (0.32, 0.50, 0.68):
        c.arrow(x, 0.40 - abs(0.50 - x) * 0.9, x, 0.62, W_FINE, head=0.045)


def illumination_03(c):
    """Field divided by its flat-field: two tiles, a division sign, a clean tile."""
    c.rect(0.02, 0.34, 0.22, 0.32, w=W_SEC, r=0.02)
    c.clip_rect(0.02, 0.34, 0.22, 0.32)
    ring_field(c, 0.13, 0.50, 3, 0.040, 0.050, W_SEC)
    c.unclip()
    c.disc(0.30, 0.44, 0.026)
    c.line(0.255, 0.50, 0.345, 0.50, W_SEC)
    c.disc(0.30, 0.56, 0.026)
    c.rect(0.38, 0.34, 0.22, 0.32, w=W_SEC, r=0.02)
    c.clip_rect(0.38, 0.34, 0.22, 0.32)
    ring_field(c, 0.49, 0.50, 3, 0.040, 0.050, W_SEC)
    c.unclip()
    c.line(0.645, 0.455, 0.735, 0.455, W_SEC)
    c.line(0.645, 0.545, 0.735, 0.545, W_SEC)
    c.rect(0.76, 0.34, 0.22, 0.32, w=W_MAIN, r=0.02)


def illumination_04(c):
    """A corner falling away: arcs crowding one corner of the frame."""
    c.rect(0.08, 0.08, 0.84, 0.84, w=W_MAIN)
    c.clip_rect(0.08, 0.08, 0.84, 0.84)
    for r in (0.20, 0.34, 0.48, 0.62):
        c.circ(0.06, 0.06, r, W_SEC)
    c.unclip()
    c.arrow(0.88, 0.88, 0.34, 0.34, W_SEC, head=0.070)


def illumination_05(c):
    """A buckled surface over the field, pressed down into a flat plane."""
    for i in range(4):
        y = 0.10 + i * 0.075
        c.smooth([(0.08, y + 0.10), (0.30, y - 0.055), (0.50, y - 0.085),
                  (0.70, y - 0.055), (0.92, y + 0.10)], w=W_SEC)
    c.arrow(0.50, 0.50, 0.50, 0.64, W_MAIN, head=0.060)
    for i in range(4):
        c.line(0.08, 0.74 + i * 0.06, 0.92, 0.74 + i * 0.06, W_SEC)


def illumination_06(c):
    """One field shown twice across a hard edge: vignetted, then flat."""
    c.rect(0.08, 0.18, 0.84, 0.64, w=W_MAIN)
    c.clip_rect(0.08, 0.18, 0.42, 0.64)
    ring_field(c, 0.28, 0.50, 5, 0.055, 0.075, W_SEC)
    c.unclip()
    c.line(0.50, 0.18, 0.50, 0.82, W_MAIN)


def illumination_07(c):
    """The field estimated from the plate itself: many wells averaged into one."""
    for dx, dy in ((0.00, 0.00), (0.05, -0.05), (0.10, -0.10)):
        c.rect(0.03 + dx, 0.36 + dy, 0.34, 0.34, w=W_SEC, r=0.02)
    c.dot_ring(0.20, 0.53, 0.095, 4, 0.030, a0=TAU / 8)
    c.arrow(0.48, 0.46, 0.60, 0.46, W_MAIN, head=0.060)
    c.rect(0.64, 0.28, 0.34, 0.34, w=W_MAIN, r=0.02)
    c.clip_rect(0.64, 0.28, 0.34, 0.34)
    ring_field(c, 0.81, 0.45, 3, 0.050, 0.060, W_SEC)
    c.unclip()


def illumination_08(c):
    """Spots shrinking toward the corners, then all the same size again."""
    c.rect(0.02, 0.30, 0.42, 0.42, w=W_SEC, r=0.02)
    for i in range(3):
        for j in range(3):
            x = 0.02 + 0.42 * (i + 0.5) / 3
            y = 0.30 + 0.42 * (j + 0.5) / 3
            d = math.hypot(i - 1, j - 1)
            c.disc(x, y, 0.048 - 0.014 * d)
    c.arrow(0.48, 0.51, 0.58, 0.51, W_MAIN, head=0.055)
    c.rect(0.62, 0.30, 0.42, 0.42, w=W_SEC, r=0.02)
    for i in range(3):
        for j in range(3):
            c.disc(0.62 + 0.42 * (i + 0.5) / 3, 0.30 + 0.42 * (j + 0.5) / 3,
                   0.040)


def illumination_09(c):
    """Well means arching up in the middle of the plate, then levelled."""
    base = 0.74
    for i, h in enumerate((0.20, 0.34, 0.42, 0.34, 0.20)):
        c.rect(0.06 + i * 0.075, base - h, 0.052, h, filled=True)
    c.arrow(0.46, 0.54, 0.57, 0.54, W_MAIN, head=0.060)
    for i in range(5):
        c.rect(0.62 + i * 0.075, base - 0.30, 0.052, 0.30, filled=True)
    c.line(0.03, base + 0.03, 0.97, base + 0.03, W_SEC)


def illumination_10(c):
    """The lamp's hot spot pooled in the centre of the frame."""
    c.rect(0.30, 0.02, 0.40, 0.12, w=W_MAIN, r=0.02)
    c.polyline([(0.34, 0.16), (0.66, 0.16), (0.92, 0.54), (0.08, 0.54)],
               w=W_SEC, close=True)
    c.rect(0.06, 0.54, 0.88, 0.42, w=W_MAIN)
    c.ell(0.50, 0.75, 0.220, 0.130, 0, filled=True)
    c.ell(0.50, 0.75, 0.340, 0.190, 0, W_SEC)


# =====================================================================
# data_manager -- what the project costs on disk, and getting it back
# =====================================================================

def data_manager_01(c):
    """A treemap of the project: the big folder dwarfing the rest."""
    c.rect(0.06, 0.12, 0.88, 0.76, w=W_MAIN)
    c.rect(0.08, 0.14, 0.48, 0.72, filled=True)
    c.line(0.58, 0.12, 0.58, 0.88, W_SEC)
    c.line(0.58, 0.50, 0.94, 0.50, W_SEC)
    c.line(0.76, 0.50, 0.76, 0.88, W_SEC)
    c.rect(0.60, 0.16, 0.32, 0.30, filled=True)


def data_manager_02(c):
    """A disk-use ring with one wedge pulled clear of the rest."""
    c.arc(0.46, 0.56, 0.255, 86, 278, 150, dash=[9999, 1])
    c.arc(0.61, 0.41, 0.255, 14, 62, 150, dash=[9999, 1])


def data_manager_03(c):
    """A drive whose fill level drops back down."""
    c.rect(0.14, 0.10, 0.72, 0.80, w=W_MAIN, r=0.05)
    c.rect(0.20, 0.48, 0.60, 0.36, filled=True)
    c.line(0.20, 0.30, 0.80, 0.30, W_SEC, dash=[8, 9])
    c.arrow(0.50, 0.32, 0.50, 0.44, W_MAIN, head=0.060)


def data_manager_04(c):
    """Derived tiles going in the bin while the originals stay locked."""
    c.polyline([(0.06, 0.42), (0.14, 0.94), (0.42, 0.94), (0.50, 0.42)],
               w=W_MAIN, close=True)
    c.line(0.02, 0.42, 0.54, 0.42, W_MAIN)
    c.rect(0.20, 0.14, 0.16, 0.16, w=W_SEC, r=0.02)
    c.arrow(0.28, 0.32, 0.28, 0.40, W_FINE, head=0.045)
    for i in range(3):
        c.rect(0.62, 0.50 + i * 0.16, 0.34, 0.12, filled=True)
    c.arc(0.79, 0.30, 0.058, 0, 180, W_SEC)
    c.rect(0.70, 0.30, 0.18, 0.13, w=W_SEC, r=0.02)


def data_manager_05(c):
    """A tall stack of files shrunk to a short one."""
    for i in range(5):
        c.rect(0.04, 0.20 + i * 0.15, 0.36, 0.11, filled=True)
    c.arrow(0.48, 0.55, 0.60, 0.55, W_MAIN, head=0.065)
    for i in range(2):
        c.rect(0.64, 0.65 + i * 0.15, 0.32, 0.11, filled=True)
    dashed_rect(c, 0.64, 0.20, 0.32, 0.41, W_FINE)


def data_manager_06(c):
    """A capacity gauge with its needle swung back off full."""
    c.arc(0.50, 0.68, 0.360, 200, -220, 26)
    c.tick_ring(0.50, 0.68, 0.36, 5, 0.25, 0.30, W_SEC, a0=math.pi)
    c.disc(0.50, 0.68, 0.055)
    c.line(0.50, 0.68, 0.50 + 0.30 * math.cos(math.radians(35)),
           0.68 - 0.30 * math.sin(math.radians(35)), W_FINE, dash=[7, 8])
    c.arrow(0.50, 0.68, 0.50 + 0.30 * math.cos(math.radians(140)),
            0.68 - 0.30 * math.sin(math.radians(140)), W_MAIN, head=0.070)


def data_manager_07(c):
    """A folder list with a size bar against every entry."""
    for i, (y, w) in enumerate(((0.14, 0.44), (0.42, 0.28), (0.70, 0.14))):
        folder(c, 0.05, y, 0.20, 0.18, W_SEC)
        c.rect(0.32, y + 0.03, w, 0.12, filled=True)


def data_manager_08(c):
    """A block of data squeezed down between two jaws."""
    c.rect(0.06, 0.44, 0.88, 0.12, filled=True)
    c.rect(0.28, 0.03, 0.44, 0.14, w=W_MAIN)
    c.rect(0.28, 0.83, 0.44, 0.14, w=W_MAIN)
    c.arrow(0.50, 0.20, 0.50, 0.38, W_MAIN, head=0.070)
    c.arrow(0.50, 0.80, 0.50, 0.62, W_MAIN, head=0.070)


def data_manager_09(c):
    """The project folder measured end to end for what it costs."""
    folder(c, 0.04, 0.14, 0.62, 0.72, W_MAIN)
    for i in range(3):
        c.rect(0.14, 0.46 + i * 0.14, 0.42 - i * 0.12, 0.09, filled=True)
    c.arrow(0.84, 0.14, 0.84, 0.86, W_SEC, head=0.062)
    c.arrow(0.84, 0.86, 0.84, 0.14, W_SEC, head=0.062)
    c.line(0.74, 0.14, 0.94, 0.14, W_SEC)
    c.line(0.74, 0.86, 0.94, 0.86, W_SEC)


def data_manager_10(c):
    """Two piles: what is kept, and what can go without losing anything."""
    for i in range(3):
        c.rect(0.06, 0.30 + i * 0.17, 0.34, 0.13, filled=True)
    tick(c, 0.23, 0.19, 0.085, W_MAIN)
    for i in range(3):
        dashed_rect(c, 0.60, 0.30 + i * 0.17, 0.34, 0.13, W_SEC)
    cross(c, 0.77, 0.19, 0.075, W_MAIN)


# =====================================================================
# db_browser -- rows, columns, a query, an export
# =====================================================================

def db_browser_01(c):
    """A lens held over the rows of a table."""
    table(c, 0.04, 0.12, 0.76, 0.62, 3, 4)
    c.magnifier(0.66, 0.66, 0.230, 45.0, W_MAIN, handle=0.55)


def db_browser_02(c):
    """A query typed above the table, and the rows that answered it."""
    c.bar(0.06, 0.06, 0.88, 0.18, filled=False, w=W_MAIN)
    c.line(0.14, 0.15, 0.60, 0.15, W_SEC)
    c.disc(0.82, 0.15, 0.045)
    c.rect(0.06, 0.34, 0.88, 0.60, w=W_SEC)
    grid_lines(c, 0.06, 0.34, 0.88, 0.60, 3, 4)
    cell_fill(c, 0.06, 0.34, 0.88, 0.60, 1, 4, [(0, 1), (0, 3)], inset=0.014)


def db_browser_03(c):
    """Many rows into the filter, a few rows out."""
    for i in range(4):
        c.rect(0.06, 0.05 + i * 0.10, 0.88, 0.07, filled=True)
    c.polyline([(0.04, 0.48), (0.96, 0.48), (0.62, 0.68), (0.38, 0.68)],
               w=W_MAIN, close=True)
    for i in range(2):
        c.rect(0.30, 0.76 + i * 0.12, 0.40, 0.07, filled=True)


def db_browser_04(c):
    """One column sorted: the rows shuffled into order under a caret."""
    c.polyline([(0.14, 0.16), (0.24, 0.04), (0.34, 0.16)], w=W_MAIN)
    c.line(0.52, 0.06, 0.52, 0.94, W_SEC)
    for i in range(4):
        y = 0.28 + i * 0.20
        c.rect(0.10, y, 0.28, 0.13, filled=True)
        c.rect(0.58, y, 0.10 + i * 0.10, 0.13, filled=True)


def db_browser_05(c):
    """Rows poured out of the database file into a readable grid."""
    cylinder(c, 0.10, 0.04, 0.36, 0.40, W_MAIN)
    c.arrow(0.30, 0.48, 0.44, 0.62, W_MAIN, head=0.070)
    c.rect(0.34, 0.56, 0.62, 0.40, w=W_SEC)
    grid_lines(c, 0.34, 0.56, 0.62, 0.40, 3, 3)


def db_browser_06(c):
    """One cell picked out, its row and its column banded across the sheet."""
    c.rect(0.06, 0.10, 0.88, 0.80, w=W_SEC)
    grid_lines(c, 0.06, 0.10, 0.88, 0.80, 5, 5)
    c.rect(0.06, 0.42, 0.88, 0.16, filled=True)
    c.rect(0.412, 0.10, 0.176, 0.80, filled=True)
    c.rect(0.424, 0.432, 0.152, 0.136, w=W_MAIN)


def db_browser_07(c):
    """A slice of the table walked out as a plain sheet of rows."""
    c.rect(0.04, 0.20, 0.42, 0.60, w=W_SEC)
    grid_lines(c, 0.04, 0.20, 0.42, 0.60, 2, 3)
    c.rect(0.04, 0.20, 0.42, 0.20, filled=True)
    c.arrow(0.50, 0.50, 0.62, 0.50, W_MAIN, head=0.065)
    sheet(c, 0.66, 0.18, 0.32, 0.64, 0.11, W_MAIN)
    text_lines(c, 0.72, 0.42, 0.20, 4, 0.10, W_SEC)


def db_browser_08(c):
    """The browser itself: tables listed down one side, rows on the other."""
    c.rect(0.04, 0.10, 0.92, 0.80, w=W_MAIN, r=0.03)
    c.line(0.34, 0.10, 0.34, 0.90, W_SEC)
    for i in range(3):
        c.rect(0.09, 0.24 + i * 0.20, 0.20, 0.10, filled=(i == 1))
        if i != 1:
            c.rect(0.09, 0.24 + i * 0.20, 0.20, 0.10, w=W_FINE)
    grid_lines(c, 0.34, 0.10, 0.62, 0.80, 2, 4, W_SEC)


def db_browser_09(c):
    """Two tables joined on the key column they share."""
    c.rect(0.02, 0.20, 0.36, 0.56, w=W_SEC)
    grid_lines(c, 0.02, 0.20, 0.36, 0.56, 2, 3)
    cell_fill(c, 0.02, 0.20, 0.36, 0.56, 2, 1, [(1, 0)], inset=0.012)
    c.rect(0.62, 0.20, 0.36, 0.56, w=W_SEC)
    grid_lines(c, 0.62, 0.20, 0.36, 0.56, 2, 3)
    cell_fill(c, 0.62, 0.20, 0.36, 0.56, 2, 1, [(0, 0)], inset=0.012)
    c.line(0.38, 0.34, 0.62, 0.34, W_MAIN)
    c.line(0.38, 0.62, 0.62, 0.62, W_MAIN)


def db_browser_10(c):
    """Paging through the rows: a long table with a grip on its scrollbar."""
    c.rect(0.06, 0.06, 0.66, 0.88, w=W_SEC)
    grid_lines(c, 0.06, 0.06, 0.66, 0.88, 2, 5)
    c.rect(0.06, 0.06, 0.66, 0.176, filled=True)
    c.rect(0.80, 0.06, 0.14, 0.88, w=W_SEC, r=0.06)
    c.rect(0.82, 0.30, 0.10, 0.34, filled=True)
    c.polyline([(0.83, 0.19), (0.87, 0.13), (0.91, 0.19)], w=W_SEC)
    c.polyline([(0.83, 0.75), (0.87, 0.81), (0.91, 0.75)], w=W_SEC)


# =====================================================================
# anndata_export -- the obs x var matrix, written out for scanpy
# =====================================================================

def anndata_export_01(c):
    """The h5ad layout: the X block with var along the top and obs down the side."""
    c.rect(0.28, 0.28, 0.68, 0.68, w=W_MAIN)
    grid_lines(c, 0.28, 0.28, 0.68, 0.68, 3, 3, W_SEC)
    for i in range(3):
        c.rect(0.30 + i * 0.227, 0.06, 0.19, 0.14, filled=True)
    for j in range(3):
        c.rect(0.06, 0.30 + j * 0.227, 0.14, 0.19, filled=True)


def anndata_export_02(c):
    """A sparse matrix: most of the grid empty, a scatter of cells filled."""
    c.rect(0.08, 0.08, 0.84, 0.84, w=W_MAIN)
    grid_lines(c, 0.08, 0.08, 0.84, 0.84, 4, 4, W_FINE)
    cell_fill(c, 0.08, 0.08, 0.84, 0.84, 4, 4,
              [(0, 0), (2, 1), (3, 2), (1, 3)], inset=0.016)


def anndata_export_03(c):
    """Rows are cells, columns are features: the two margins spelled out."""
    c.rect(0.30, 0.30, 0.66, 0.66, w=W_MAIN)
    grid_lines(c, 0.30, 0.30, 0.66, 0.66, 3, 3, W_SEC)
    for j in range(3):
        c.cell(0.14, 0.41 + j * 0.22, 0.085, 0.070, 0, w=W_SEC, nuc=0.38)
    for i in range(3):
        c.rect(0.34 + i * 0.22, 0.06, 0.055, 0.16, filled=True)


def anndata_export_04(c):
    """One file, three layers: the matrix with obs and var stacked behind it."""
    for i, dy in enumerate((0.32, 0.16, 0.00)):
        pts = [(0.50, 0.14 + dy), (0.94, 0.34 + dy),
               (0.50, 0.54 + dy), (0.06, 0.34 + dy)]
        c.polyline(pts, w=W_MAIN if i == 2 else W_SEC, close=True)
    c.polyline([(0.50, 0.60), (0.62, 0.66), (0.50, 0.72), (0.38, 0.66)],
               close=True, filled=True)


def anndata_export_05(c):
    """The matrix handed over and read back as a cloud of cells."""
    c.rect(0.03, 0.28, 0.34, 0.44, w=W_MAIN)
    grid_lines(c, 0.03, 0.28, 0.34, 0.44, 3, 3, W_FINE)
    c.arrow(0.42, 0.50, 0.55, 0.50, W_MAIN, head=0.065)
    for x, y, r in ((0.68, 0.32, 0.055), (0.82, 0.24, 0.042),
                    (0.90, 0.42, 0.050), (0.72, 0.50, 0.038),
                    (0.66, 0.68, 0.048), (0.82, 0.74, 0.058),
                    (0.92, 0.62, 0.036)):
        c.disc(x, y, r)


def anndata_export_06(c):
    """The file's tree: one root holding X, obs and var."""
    c.rect(0.36, 0.04, 0.28, 0.20, w=W_MAIN, r=0.03)
    c.line(0.50, 0.24, 0.50, 0.40, W_SEC)
    c.line(0.16, 0.40, 0.84, 0.40, W_SEC)
    for x in (0.16, 0.50, 0.84):
        c.line(x, 0.40, x, 0.54, W_SEC)
    c.rect(0.04, 0.54, 0.24, 0.24, w=W_SEC)
    grid_lines(c, 0.04, 0.54, 0.24, 0.24, 2, 2, W_FINE)
    c.rect(0.38, 0.54, 0.24, 0.24, filled=True)
    c.rect(0.72, 0.54, 0.24, 0.24, w=W_SEC)


def anndata_export_07(c):
    """The whole matrix folded down into one file on disk."""
    c.rect(0.06, 0.06, 0.44, 0.44, w=W_SEC)
    grid_lines(c, 0.06, 0.06, 0.44, 0.44, 3, 3, W_FINE)
    c.smooth([(0.50, 0.28), (0.66, 0.30), (0.72, 0.46)], w=W_MAIN)
    c.arrow(0.71, 0.40, 0.73, 0.54, W_MAIN, head=0.065, tail=False)
    sheet(c, 0.54, 0.56, 0.40, 0.40, 0.13, W_MAIN)
    c.rect(0.62, 0.70, 0.22, 0.16, filled=True)


def anndata_export_08(c):
    """The shape being written: so many cells by so many features."""
    c.rect(0.20, 0.10, 0.72, 0.66, w=W_MAIN)
    grid_lines(c, 0.20, 0.10, 0.72, 0.66, 3, 3, W_FINE)
    caliper(c, 0.20, 0.92, 0.90, W_SEC, tick_h=0.045, head=0.052)
    c.arrow(0.09, 0.10, 0.09, 0.76, W_SEC, head=0.052)
    c.arrow(0.09, 0.76, 0.09, 0.10, W_SEC, head=0.052)


def anndata_export_09(c):
    """One cell's row of features lifted out of the matrix as a vector."""
    c.rect(0.06, 0.34, 0.88, 0.60, w=W_SEC)
    grid_lines(c, 0.06, 0.34, 0.88, 0.60, 4, 3, W_FINE)
    c.rect(0.06, 0.54, 0.88, 0.20, filled=True)
    c.rect(0.06, 0.06, 0.88, 0.16, w=W_MAIN)
    grid_lines(c, 0.06, 0.06, 0.88, 0.16, 4, 1, W_SEC)
    c.arrow(0.50, 0.50, 0.50, 0.28, W_MAIN, head=0.060)


def anndata_export_10(c):
    """The measurement rows turned on their side into the X matrix."""
    c.rect(0.04, 0.54, 0.40, 0.42, w=W_SEC)
    grid_lines(c, 0.04, 0.54, 0.40, 0.42, 2, 4, W_FINE)
    c.rect(0.04, 0.54, 0.40, 0.105, filled=True)
    c.smooth([(0.30, 0.44), (0.50, 0.26), (0.70, 0.36)], w=W_MAIN)
    c.arrow(0.62, 0.30, 0.74, 0.44, W_MAIN, head=0.065, tail=False)
    c.rect(0.56, 0.48, 0.40, 0.44, w=W_MAIN)
    grid_lines(c, 0.56, 0.48, 0.40, 0.44, 3, 3, W_SEC)


# =====================================================================
# methods_export -- prose whose every number traces back to the run
# =====================================================================

def methods_export_01(c):
    """A paragraph with two markers, each on a leader line down to its bar."""
    text_lines(c, 0.06, 0.10, 0.66, 3, 0.13, W_MAIN)
    c.disc(0.80, 0.10, 0.046)
    c.disc(0.58, 0.36, 0.046)
    c.line(0.80, 0.16, 0.80, 0.60, W_FINE, dash=[6, 7])
    c.line(0.58, 0.42, 0.30, 0.60, W_FINE, dash=[6, 7])
    for i, h in enumerate((0.14, 0.26, 0.20)):
        c.rect(0.22 + i * 0.18, 0.90 - h, 0.11, h, filled=True)
    c.line(0.16, 0.92, 0.94, 0.92, W_SEC)


def methods_export_02(c):
    """Blanks in the draft filled from the cells of the results table."""
    text_lines(c, 0.08, 0.10, 0.60, 2, 0.12, W_SEC)
    for i, x in enumerate((0.10, 0.42, 0.70)):
        c.rect(x, 0.28, 0.20, 0.11, w=W_MAIN)
        c.arrow(x + 0.10, 0.62, x + 0.10, 0.42, W_FINE, head=0.050)
    c.rect(0.06, 0.66, 0.88, 0.28, w=W_SEC)
    grid_lines(c, 0.06, 0.66, 0.88, 0.28, 3, 2, W_FINE)
    c.rect(0.06, 0.66, 0.88, 0.14, filled=True)


def methods_export_03(c):
    """Methods on one page, results on the other, one figure feeding both."""
    sheet(c, 0.03, 0.06, 0.34, 0.52, 0.10, W_SEC)
    text_lines(c, 0.08, 0.22, 0.22, 3, 0.09, W_FINE)
    sheet(c, 0.63, 0.06, 0.34, 0.52, 0.10, W_SEC)
    text_lines(c, 0.68, 0.22, 0.22, 3, 0.09, W_FINE)
    c.rect(0.30, 0.68, 0.40, 0.28, w=W_MAIN)
    for i, h in enumerate((0.10, 0.18, 0.14)):
        c.rect(0.36 + i * 0.10, 0.92 - h, 0.06, h, filled=True)
    c.smooth([(0.20, 0.60), (0.26, 0.74), (0.30, 0.80)], w=W_SEC)
    c.smooth([(0.80, 0.60), (0.74, 0.74), (0.70, 0.80)], w=W_SEC)


def methods_export_04(c):
    """A sentence's number opened up to show the value behind it."""
    text_lines(c, 0.06, 0.12, 0.66, 3, 0.12, W_SEC)
    c.rect(0.60, 0.30, 0.22, 0.12, filled=True)
    c.polyline([(0.20, 0.62), (0.86, 0.62), (0.86, 0.94), (0.20, 0.94)],
               w=W_MAIN, close=True)
    c.polyline([(0.62, 0.62), (0.68, 0.48), (0.74, 0.62)],
               w=W_MAIN, close=True, filled=True)
    c.rect(0.28, 0.72, 0.34, 0.12, filled=True)
    c.line(0.28, 0.88, 0.78, 0.88, W_FINE)


def methods_export_05(c):
    """A page whose lower half is the figure the paragraph is talking about."""
    sheet(c, 0.10, 0.04, 0.80, 0.92, 0.16, W_MAIN)
    text_lines(c, 0.18, 0.26, 0.60, 3, 0.10, W_SEC)
    c.rect(0.18, 0.58, 0.64, 0.24, w=W_SEC)
    c.smooth([(0.22, 0.78), (0.38, 0.64), (0.56, 0.72), (0.78, 0.62)],
             w=W_SEC)
    c.line(0.18, 0.90, 0.62, 0.90, W_FINE)


def methods_export_06(c):
    """A thread from the marker in the text back down to the plate it came from."""
    text_lines(c, 0.06, 0.10, 0.62, 3, 0.11, W_SEC)
    c.circ(0.80, 0.21, 0.055, W_MAIN)
    c.disc(0.80, 0.21, 0.024)
    c.smooth([(0.80, 0.28), (0.72, 0.48), (0.50, 0.58)], w=W_SEC,
             dash=[7, 8])
    c.rect(0.14, 0.62, 0.72, 0.32, w=W_MAIN, r=0.04)
    for i in range(4):
        for j in range(2):
            c.disc(0.22 + i * 0.19, 0.72 + j * 0.14, 0.030)


def methods_export_07(c):
    """A footnote rule, with the source line the paragraph is standing on."""
    text_lines(c, 0.08, 0.14, 0.66, 3, 0.14, W_MAIN)
    c.disc(0.84, 0.14, 0.050)
    c.line(0.08, 0.62, 0.40, 0.62, W_MAIN)
    c.disc(0.13, 0.78, 0.042)
    c.line(0.23, 0.78, 0.90, 0.78, W_SEC)
    c.line(0.23, 0.91, 0.62, 0.91, W_SEC)


def methods_export_08(c):
    """The draft being written straight off the stack of run outputs."""
    for i in range(3):
        c.rect(0.04, 0.62 + i * 0.13, 0.38, 0.09, filled=True)
    c.arrow(0.46, 0.64, 0.56, 0.54, W_SEC, head=0.055)
    sheet(c, 0.40, 0.06, 0.54, 0.44, 0.12, W_MAIN)
    text_lines(c, 0.46, 0.20, 0.36, 3, 0.09, W_SEC)
    c.polyline([(0.64, 0.62), (0.86, 0.84), (0.96, 0.74), (0.74, 0.52)],
               w=W_SEC, close=True)
    c.polyline([(0.64, 0.62), (0.74, 0.52), (0.605, 0.485)],
               close=True, filled=True)


def methods_export_09(c):
    """Numbered references down the margin, each tied to a bar in the chart."""
    for i, y in enumerate((0.14, 0.42, 0.70)):
        c.disc(0.10, y + 0.06, 0.040)
        c.line(0.18, y + 0.06, 0.46, y + 0.06, W_SEC)
        c.line(0.52, y + 0.06, 0.62, 0.86 - i * 0.20, W_FINE, dash=[5, 6])
    c.axes(0.66, 0.10, 0.98, 0.94, W_SEC)
    for i, h in enumerate((0.18, 0.30, 0.44)):
        c.rect(0.70 + i * 0.10, 0.92 - h, 0.07, h, filled=True)


def methods_export_10(c):
    """One claim in the text branching down onto the three numbers behind it."""
    c.rect(0.14, 0.08, 0.72, 0.16, filled=True)
    c.line(0.50, 0.24, 0.50, 0.40, W_SEC)
    c.line(0.16, 0.40, 0.84, 0.40, W_SEC)
    for x in (0.16, 0.50, 0.84):
        c.line(x, 0.40, x, 0.54, W_SEC)
    c.rect(0.04, 0.54, 0.24, 0.24, w=W_MAIN)
    c.circ(0.50, 0.66, 0.120, W_MAIN)
    c.polyline([(0.84, 0.54), (0.96, 0.78), (0.72, 0.78)],
               w=W_MAIN, close=True)


# =====================================================================
# manifest
# =====================================================================

GROUPS = {
    "align": ("align -- overlapping tiles registered into one stitched canvas", [
        ("Two camera fields overlapping, the strip they share filled solid.",
         align_01),
        ("The shift measured from one speck to the same speck in the next tile.",
         align_02),
        ("One peak on the shift map: crosshairs on the offset that won.",
         align_03),
        ("A registration target where two tile corners have to land together.",
         align_04),
        ("The stage's path strung through the tile centres in visiting order.",
         align_05),
        ("A cell cut in half by the seam, whole again once the halves register.",
         align_06),
        ("The mosaic filling in tile by tile, one tile held at a time.",
         align_07),
        ("Corner brackets pulling a loose tile square onto the grid.",
         align_08),
        ("A heap of skewed tiles above, squared into a clean grid below.",
         align_09),
        ("Tiles laid down one after another, every overlap solid where they meet.",
         align_10),
    ]),
    "foreign": ("foreign -- their columns remapped onto spaCR's", [
        ("Their column names on the left wired across to ours on the right.",
         foreign_01),
        ("A foreign table with its header row lifted off and ours dropped in.",
         foreign_02),
        ("A foreign-shaped plug machined to fit spaCR's socket.", foreign_03),
        ("A crosswalk grid, ticked where their column answers to ours.",
         foreign_04),
        ("Their column heads dropping into the holes whose shape they match.",
         foreign_05),
        ("Ragged foreign headings funnelled into evenly cut spaCR columns.",
         foreign_06),
        ("One column lifted out of their table and dropped in our empty slot.",
         foreign_07),
        ("A rename tag tied onto the one column it re-labels.", foreign_08),
        ("An unclaimed column, with the spaCR names it could be renamed to.",
         foreign_09),
        ("A stamp pressing spaCR's headings onto someone else's sheet.",
         foreign_10),
    ]),
    "external_masks": ("external_masks -- labels drawn elsewhere, measured here", [
        ("The pair that arrives: a field of cells, and its labels already solid.",
         external_masks_01),
        ("The segmentation step struck out: images and labels go straight to measure.",
         external_masks_02),
        ("An outline sheet drawn elsewhere, dropped onto the image beneath it.",
         external_masks_03),
        ("Labels that arrive already numbered, each object carrying its own id.",
         external_masks_04),
        ("A label from somewhere else flying in and landing on the cell it fits.",
         external_masks_05),
        ("Every image tile matched to the mask file that came with it.",
         external_masks_06),
        ("A ready-made label being measured, not drawn: calipers across a blob.",
         external_masks_07),
        ("Two layers handed over together: pixels below, their label map on top.",
         external_masks_08),
        ("Supplied labels going straight into the measurement table.",
         external_masks_09),
        ("An image and its label map travelling together into a spaCR project.",
         external_masks_10),
    ]),
    "illumination": ("illumination -- an uneven field flattened before measuring", [
        ("A vignetted field's contour rings, corrected to a field with none.",
         illumination_01),
        ("The intensity profile across the field, pulled down onto a level line.",
         illumination_02),
        ("Field divided by its flat-field: two tiles, a division sign, a clean tile.",
         illumination_03),
        ("A corner falling away: arcs crowding one corner of the frame.",
         illumination_04),
        ("A buckled surface over the field, pressed down into a flat plane.",
         illumination_05),
        ("One field shown twice across a hard edge: vignetted, then flat.",
         illumination_06),
        ("The field estimated from the plate itself: many wells averaged into one.",
         illumination_07),
        ("Spots shrinking toward the corners, then all the same size again.",
         illumination_08),
        ("Well means arching up in the middle of the plate, then levelled.",
         illumination_09),
        ("The lamp's hot spot pooled in the centre of the frame.",
         illumination_10),
    ]),
    "data_manager": ("data_manager -- what the project costs on disk, and getting it back", [
        ("A treemap of the project: the big folder dwarfing the rest.",
         data_manager_01),
        ("A disk-use ring with one wedge pulled clear of the rest.",
         data_manager_02),
        ("A drive whose fill level drops back down.", data_manager_03),
        ("Derived tiles going in the bin while the originals stay locked.",
         data_manager_04),
        ("A tall stack of files shrunk to a short one.", data_manager_05),
        ("A capacity gauge with its needle swung back off full.",
         data_manager_06),
        ("A folder list with a size bar against every entry.",
         data_manager_07),
        ("A block of data squeezed down between two jaws.", data_manager_08),
        ("The project folder measured end to end for what it costs.",
         data_manager_09),
        ("Two piles: what is kept, and what can go without losing anything.",
         data_manager_10),
    ]),
    "db_browser": ("db_browser -- rows, columns, a query, an export", [
        ("A lens held over the rows of a table.", db_browser_01),
        ("A query typed above the table, and the rows that answered it.",
         db_browser_02),
        ("Many rows into the filter, a few rows out.", db_browser_03),
        ("One column sorted: the rows shuffled into order under a caret.",
         db_browser_04),
        ("Rows poured out of the database file into a readable grid.",
         db_browser_05),
        ("One cell picked out, its row and its column banded across the sheet.",
         db_browser_06),
        ("A slice of the table walked out as a plain sheet of rows.",
         db_browser_07),
        ("The browser itself: tables listed down one side, rows on the other.",
         db_browser_08),
        ("Two tables joined on the key column they share.", db_browser_09),
        ("Paging through the rows: a long table with a grip on its scrollbar.",
         db_browser_10),
    ]),
    "anndata_export": ("anndata_export -- the obs x var matrix written out for scanpy", [
        ("The h5ad layout: the X block with var along the top and obs down the side.",
         anndata_export_01),
        ("A sparse matrix: most of the grid empty, a scatter of cells filled.",
         anndata_export_02),
        ("Rows are cells, columns are features: the two margins spelled out.",
         anndata_export_03),
        ("One file, three layers: the matrix with obs and var stacked behind it.",
         anndata_export_04),
        ("The matrix handed over and read back as a cloud of cells.",
         anndata_export_05),
        ("The file's tree: one root holding X, obs and var.", anndata_export_06),
        ("The whole matrix folded down into one file on disk.",
         anndata_export_07),
        ("The shape being written: so many cells by so many features.",
         anndata_export_08),
        ("One cell's row of features lifted out of the matrix as a vector.",
         anndata_export_09),
        ("The measurement rows turned on their side into the X matrix.",
         anndata_export_10),
    ]),
    "methods_export": ("methods_export -- prose whose every number traces back to the run", [
        ("A paragraph with two markers, each on a leader line down to its bar.",
         methods_export_01),
        ("Blanks in the draft filled from the cells of the results table.",
         methods_export_02),
        ("Methods on one page, results on the other, one figure feeding both.",
         methods_export_03),
        ("A sentence's number opened up to show the value behind it.",
         methods_export_04),
        ("A page whose lower half is the figure the paragraph is talking about.",
         methods_export_05),
        ("A thread from the marker in the text back down to the plate it came from.",
         methods_export_06),
        ("A footnote rule, with the source line the paragraph is standing on.",
         methods_export_07),
        ("The draft being written straight off the stack of run outputs.",
         methods_export_08),
        ("Numbered references down the margin, each tied to a bar in the chart.",
         methods_export_09),
        ("One claim in the text branching down onto the three numbers behind it.",
         methods_export_10),
    ]),
}


def main(outdir):
    return emit_groups(outdir, GROUPS, "group_data_io.py")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else default_outdir(__file__)))
