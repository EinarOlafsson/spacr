#!/usr/bin/env python3
"""Candidate spaCR icons: queue, batch, invasion, replication.

Eight conceptually different designs per icon, white-on-transparent flat
vector art in the house style set by ``plaque.png`` / ``measure.png``.

Four icons that had nothing worth keeping:

* **queue** and **batch** both aliased ``sequencing.png`` — a DNA helix,
  chosen as "the closest visual match for now". Neither has ever had
  artwork of its own, and the two apps are not the same thing, so the
  designs below have to carry the distinction: **queue is the SAME
  settings over many plates** (one settings card, repeated identically),
  **batch is ARBITRARY module+plate combinations in sequence** (a
  schedule of *different* jobs).
* **invasion** rendered a Font Awesome arrow-into-a-bracket because no
  bundled PNG read as "inside vs outside".
* **replication** did not exist at all — the module was never wired up,
  though ``spacr.submodules.analyze_endodyogeny`` has been there the
  whole time.

``invasion`` and ``replication`` are both "parasite + host" pictures and
must not collapse into each other: invasion is about **the membrane** —
attached on the outside versus invaded on the inside — and replication
is about **the vacuole** — one parasite becoming two, then four, inside
it. Every design below keeps that line.

Run standalone (deterministic — no random draws at all):

    python group_queue_batch_invasion_replication.py [OUTDIR]

Default OUTDIR is the backup_icons directory two levels up. Writes
``<OUTDIR>/<name>/<name>_NN.png`` plus CONCEPTS.md and the two contact
sheets. It never touches anything in ``spacr/resources/icons/*.png``.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _draw import N, W_FINE, W_MAIN, W_SEC, Cv, contact_sheet, render  # noqa: E402

DARK_BG = "#14161a"
LIGHT_BG = "#f5f6f8"

TAU = math.pi * 2


# ---------------------------------------------------------------------------
# shared sub-drawings
# ---------------------------------------------------------------------------

def plate(c, x, y, w, h, cols=4, rows=3, r=0.022, dot=0.018, w_=W_SEC):
    """A microtitre plate: rounded outline with a grid of wells."""
    c.rect(x, y, w, h, w=w_, r=r)
    for i in range(cols):
        for j in range(rows):
            cx = x + w * (i + 0.5) / cols
            cy = y + h * (j + 0.5) / rows
            c.disc(cx, cy, dot)


def gear(c, cx, cy, r, teeth=8, tooth=0.30, w=W_SEC):
    """Small settings gear — the 'same settings' mark."""
    for i in range(teeth):
        a = TAU * i / teeth
        c.line(cx + r * math.cos(a), cy + r * math.sin(a),
               cx + r * (1 + tooth) * math.cos(a),
               cy + r * (1 + tooth) * math.sin(a), w)
    c.circ(cx, cy, r, w)
    c.disc(cx, cy, r * 0.38)


def vacuole(c, cx, cy, r, w=W_SEC, dash=None):
    """The parasitophorous vacuole: a thin ring the parasites sit in."""
    c.circ(cx, cy, r, w, dash)


def rosette(c, cx, cy, r, n_, L=0.20, a0=-math.pi / 2, fat=0.30):
    """Parasites arranged apex-in, the classic Toxoplasma rosette."""
    for i in range(n_):
        a = a0 + TAU * i / n_
        px, py = cx + r * math.cos(a), cy + r * math.sin(a)
        c.parasite(px, py, L, rot=math.degrees(a) + 90, fat=fat)


# =====================================================================
# queue -- the SAME settings across many plates
# =====================================================================

def queue_01(c):
    """One settings card feeding a column of identical plates."""
    gear(c, 0.20, 0.20, 0.085)
    for i, y in enumerate((0.40, 0.60, 0.80)):
        plate(c, 0.36, y - 0.085, 0.50, 0.17, cols=4, rows=2, dot=0.017)
        c.line(0.20, 0.30, 0.20, y, W_FINE)
        c.arrow(0.20, y, 0.33, y, W_FINE, head=0.045)


def queue_02(c):
    """A stack of identical plates, each stamped with the same gear."""
    for i, (dx, dy) in enumerate(((0.00, 0.00), (0.05, -0.13), (0.10, -0.26))):
        plate(c, 0.14 + dx, 0.56 + dy, 0.56, 0.20, cols=5, rows=2, dot=0.016)
        gear(c, 0.80 + dx, 0.66 + dy, 0.052, teeth=8, tooth=0.34, w=W_FINE)


def queue_03(c):
    """A waiting line: plates queued behind a 'now running' marker."""
    c.arrow(0.06, 0.50, 0.24, 0.50, W_SEC, head=0.06)
    for i, x in enumerate((0.28, 0.52, 0.76)):
        plate(c, x, 0.36, 0.20, 0.28, cols=3, rows=3, dot=0.016,
              w_=W_SEC if i == 0 else W_FINE)
    c.circ(0.38, 0.24, 0.045, W_FINE)
    c.disc(0.38, 0.24, 0.022)


def queue_04(c):
    """One recipe, many runs: a settings sheet cloned down the stack."""
    for dx, dy in ((0.16, 0.14), (0.10, 0.22), (0.04, 0.30)):
        c.rect(dx, dy, 0.44, 0.52, w=W_FINE, r=0.03)
    c.rect(0.04, 0.30, 0.44, 0.52, w=W_SEC, r=0.03)
    for y in (0.42, 0.52, 0.62, 0.72):
        c.line(0.11, y, 0.41, y, W_FINE)
    c.arrow(0.54, 0.56, 0.74, 0.56, W_SEC, head=0.06)
    plate(c, 0.78, 0.42, 0.18, 0.28, cols=3, rows=3, dot=0.015)


def queue_05(c):
    """Plates on a belt, all carrying the same mark."""
    c.line(0.06, 0.74, 0.94, 0.74, W_SEC)
    for x in (0.16, 0.36, 0.56, 0.76):
        c.disc(x, 0.80, 0.030)
    for x in (0.18, 0.46, 0.74):
        plate(c, x - 0.09, 0.48, 0.18, 0.22, cols=3, rows=2, dot=0.015)
        gear(c, x, 0.32, 0.045, teeth=6, tooth=0.36, w=W_FINE)


def queue_06(c):
    """A numbered running order down one column of plates."""
    for i, y in enumerate((0.20, 0.44, 0.68)):
        c.disc(0.14, y + 0.10, 0.038)
        plate(c, 0.28, y, 0.58, 0.20, cols=5, rows=2, dot=0.016)
    c.line(0.14, 0.34, 0.14, 0.74, W_FINE, dash=[3, 4])
    gear(c, 0.14, 0.92, 0.052, w=W_FINE)


def queue_07(c):
    """Hopper: identical plates dropping through one set of settings."""
    c.polyline([(0.14, 0.14), (0.86, 0.14), (0.60, 0.50), (0.40, 0.50)],
               w=W_SEC, close=True)
    gear(c, 0.50, 0.32, 0.070, w=W_FINE)
    for i, y in enumerate((0.60, 0.76, 0.92)):
        plate(c, 0.50 - 0.19, y - 0.055, 0.38, 0.11, cols=4, rows=1,
              dot=0.014, w_=W_FINE)


def queue_08(c):
    """Same settings, ticked off plate after plate."""
    gear(c, 0.16, 0.16, 0.090)
    for i, y in enumerate((0.44, 0.64, 0.84)):
        plate(c, 0.10, y - 0.075, 0.56, 0.15, cols=5, rows=2, dot=0.014)
        if i < 2:
            c.polyline([(0.76, y), (0.83, y + 0.055), (0.95, y - 0.065)],
                       w=W_SEC)
        else:
            c.circ(0.85, y, 0.055, W_FINE, dash=[3, 4])


# =====================================================================
# batch -- ARBITRARY module + plate combinations, run in sequence
# =====================================================================

def batch_01(c):
    """A schedule: different module marks against different plates."""
    c.rect(0.08, 0.14, 0.84, 0.72, w=W_SEC, r=0.035)
    c.line(0.08, 0.32, 0.92, 0.32, W_FINE)
    c.line(0.36, 0.14, 0.36, 0.86, W_FINE)
    for j, y in enumerate((0.42, 0.59, 0.76)):
        plate(c, 0.12, y - 0.055, 0.20, 0.11, cols=3, rows=1, dot=0.013,
              w_=W_FINE)
    marks = [(0.50, 0.23), (0.66, 0.23), (0.82, 0.23)]
    c.disc(*marks[0], 0.030)
    c.circ(*marks[1], 0.030, W_FINE)
    c.rect(marks[2][0] - 0.028, marks[2][1] - 0.028, 0.056, 0.056, w=W_FINE)
    grid = ((1, 0, 1), (0, 1, 1), (1, 1, 0))
    for j, y in enumerate((0.42, 0.59, 0.76)):
        for i, x in enumerate((0.50, 0.66, 0.82)):
            if grid[j][i]:
                c.polyline([(x - 0.030, y), (x - 0.006, y + 0.026),
                            (x + 0.034, y - 0.030)], w=W_FINE)


def batch_02(c):
    """A chain of unlike jobs, run one after another."""
    shapes = ((0.16, "disc"), (0.38, "square"), (0.60, "tri"), (0.82, "ring"))
    for i, (x, kind) in enumerate(shapes):
        if kind == "disc":
            c.disc(x, 0.50, 0.070)
        elif kind == "square":
            c.rect(x - 0.066, 0.434, 0.132, 0.132, w=W_SEC, r=0.02)
        elif kind == "tri":
            c.polyline([(x, 0.428), (x + 0.072, 0.566), (x - 0.072, 0.566)],
                       w=W_SEC, close=True)
        else:
            c.ring(x, 0.50, 0.058, 0.030)
        if i < len(shapes) - 1:
            c.arrow(x + 0.085, 0.50, shapes[i + 1][0] - 0.085, 0.50,
                    W_FINE, head=0.045)
    c.line(0.10, 0.76, 0.90, 0.76, W_FINE, dash=[4, 5])
    for i, (x, _k) in enumerate(shapes):
        c.disc(x, 0.76, 0.020)


def batch_03(c):
    """A worklist of mixed jobs with a run button."""
    c.polyline([(0.16, 0.14), (0.16, 0.42), (0.42, 0.28)], w=W_SEC,
               close=True, filled=True)
    for j, y in enumerate((0.58, 0.72, 0.86)):
        c.bar(0.10, y - 0.028, 0.20 + 0.10 * j, 0.056, filled=False, w=W_FINE)
        c.disc(0.86 - 0.10 * j, y, 0.030 - 0.006 * j)
    c.line(0.60, 0.14, 0.90, 0.14, W_FINE)
    c.line(0.60, 0.28, 0.90, 0.28, W_FINE)
    c.line(0.60, 0.42, 0.78, 0.42, W_FINE)


def batch_04(c):
    """Cards in a hopper, each a different module."""
    c.polyline([(0.10, 0.86), (0.10, 0.52), (0.90, 0.52), (0.90, 0.86)],
               w=W_SEC)
    specs = ((0.18, 0.28, "tri"), (0.40, 0.20, "square"),
             (0.62, 0.26, "disc"), (0.80, 0.18, "ring"))
    for x, y, kind in specs:
        c.rect(x - 0.075, y - 0.10, 0.15, 0.34, w=W_FINE, r=0.02)
        if kind == "disc":
            c.disc(x, y + 0.06, 0.036)
        elif kind == "square":
            c.rect(x - 0.034, y + 0.026, 0.068, 0.068, w=W_FINE)
        elif kind == "tri":
            c.polyline([(x, y + 0.020), (x + 0.040, y + 0.096),
                        (x - 0.040, y + 0.096)], w=W_FINE, close=True)
        else:
            c.ring(x, y + 0.06, 0.030, 0.018)


def batch_05(c):
    """Overnight: a clock over a stack of unlike jobs."""
    c.circ(0.74, 0.26, 0.150, W_SEC)
    c.line(0.74, 0.26, 0.74, 0.16, W_FINE)
    c.line(0.74, 0.26, 0.82, 0.30, W_FINE)
    for j, y in enumerate((0.54, 0.70, 0.86)):
        c.rect(0.08, y - 0.055, 0.62, 0.11, w=W_FINE, r=0.025)
        if j == 0:
            c.disc(0.17, y, 0.032)
        elif j == 1:
            c.rect(0.138, y - 0.032, 0.064, 0.064, w=W_FINE)
        else:
            c.polyline([(0.17, y - 0.036), (0.206, y + 0.030),
                        (0.134, y + 0.030)], w=W_FINE, close=True)
        c.line(0.24, y, 0.46 + 0.08 * j, y, W_FINE)


def batch_06(c):
    """Many modules × many plates, folded into one run."""
    for j, y in enumerate((0.16, 0.32, 0.48)):
        if j == 0:
            c.disc(0.14, y, 0.045)
        elif j == 1:
            c.rect(0.096, y - 0.044, 0.088, 0.088, w=W_SEC, r=0.016)
        else:
            c.polyline([(0.14, y - 0.050), (0.192, y + 0.042),
                        (0.088, y + 0.042)], w=W_SEC, close=True)
        for i, py in enumerate((0.16, 0.32, 0.48)):
            c.line(0.20, y, 0.44, py, W_FINE)
    for py in (0.16, 0.32, 0.48):
        plate(c, 0.46, py - 0.055, 0.20, 0.11, cols=3, rows=1, dot=0.013,
              w_=W_FINE)
    c.arrow(0.50, 0.62, 0.50, 0.80, W_SEC, head=0.06)
    c.rect(0.20, 0.82, 0.60, 0.13, w=W_SEC, r=0.03)


def batch_07(c):
    """A numbered running order of unlike steps."""
    for j, (y, kind) in enumerate(((0.20, "disc"), (0.42, "square"),
                                   (0.64, "tri"), (0.86, "ring"))):
        c.circ(0.13, y, 0.058, W_FINE)
        c.disc(0.13, y, 0.020 + 0.006 * j)
        if kind == "disc":
            c.disc(0.34, y, 0.048)
        elif kind == "square":
            c.rect(0.294, y - 0.046, 0.092, 0.092, w=W_SEC, r=0.016)
        elif kind == "tri":
            c.polyline([(0.34, y - 0.052), (0.394, y + 0.044),
                        (0.286, y + 0.044)], w=W_SEC, close=True)
        else:
            c.ring(0.34, y, 0.046, 0.024)
        c.line(0.44, y, 0.90 - 0.10 * j, y, W_FINE)


def batch_08(c):
    """A pipeline of different stages emptying into one output."""
    ys = (0.16, 0.34, 0.52, 0.70)
    for j, y in enumerate(ys):
        if j == 0:
            c.disc(0.14, y, 0.048)
        elif j == 1:
            c.rect(0.092, y - 0.046, 0.096, 0.092, w=W_SEC, r=0.016)
        elif j == 2:
            c.polyline([(0.14, y - 0.052), (0.196, y + 0.044),
                        (0.084, y + 0.044)], w=W_SEC, close=True)
        else:
            c.ring(0.14, y, 0.046, 0.024)
        c.smooth([(0.22, y), (0.42, y), (0.56, 0.44), (0.62, 0.44)], w=W_FINE)
    c.arrow(0.64, 0.44, 0.78, 0.44, W_SEC, head=0.055)
    plate(c, 0.62, 0.62, 0.32, 0.26, cols=4, rows=3, dot=0.016)


# =====================================================================
# invasion -- attached (outside) vs invaded (inside)
# =====================================================================

def invasion_01(c):
    """Host cell: outlined parasites on the membrane, solid ones within."""
    c.circ(0.50, 0.52, 0.320, W_MAIN)
    c.disc(0.62, 0.40, 0.055)
    for a, r in ((-0.5, 0.17), (0.9, 0.20), (2.4, 0.15)):
        c.parasite(0.50 + r * math.cos(a), 0.52 + r * math.sin(a),
                   0.20, rot=math.degrees(a) + 90)
    for a in (-1.9, 0.35, 2.9):
        px, py = 0.50 + 0.375 * math.cos(a), 0.52 + 0.375 * math.sin(a)
        c.ell(px, py, 0.098, 0.036, math.degrees(a) + 90, W_SEC)


def invasion_02(c):
    """One parasite caught halfway through the membrane."""
    c.arc(0.50, 0.54, 0.330, 200, 300, W_MAIN)
    c.disc(0.36, 0.62, 0.052)
    c.parasite(0.66, 0.30, 0.30, rot=52)
    c.ell(0.66, 0.30, 0.150, 0.055, 52, W_SEC)
    c.arrow(0.86, 0.14, 0.60, 0.42, W_SEC, head=0.070)


def invasion_03(c):
    """Two panels: attached on the left, invaded on the right."""
    c.line(0.50, 0.10, 0.50, 0.90, W_FINE, dash=[5, 6])
    c.circ(0.25, 0.50, 0.180, W_SEC)
    c.disc(0.25, 0.50, 0.048)
    for a in (-1.2, 0.4, 2.0):
        c.ell(0.25 + 0.225 * math.cos(a), 0.50 + 0.225 * math.sin(a),
              0.070, 0.028, math.degrees(a) + 90, W_FINE)
    c.circ(0.75, 0.50, 0.180, W_SEC)
    c.disc(0.75, 0.50, 0.048)
    for a in (-1.2, 0.4, 2.0):
        c.parasite(0.75 + 0.095 * math.cos(a), 0.50 + 0.095 * math.sin(a),
                   0.145, rot=math.degrees(a) + 90)


def invasion_04(c):
    """The moving junction: a parasite pinching through the membrane."""
    c.smooth([(0.10, 0.72), (0.30, 0.60), (0.44, 0.56), (0.50, 0.44),
              (0.56, 0.56), (0.70, 0.60), (0.90, 0.72)], w=W_MAIN)
    c.parasite(0.50, 0.30, 0.34, rot=90)
    c.line(0.44, 0.56, 0.44, 0.44, W_FINE)
    c.line(0.56, 0.56, 0.56, 0.44, W_FINE)
    c.disc(0.24, 0.86, 0.050)
    c.smooth([(0.06, 0.94), (0.34, 0.90), (0.62, 0.94), (0.94, 0.90)],
             w=W_FINE)


def invasion_05(c):
    """Outside stain vs inside stain: a ring around the attached ones."""
    c.circ(0.50, 0.50, 0.300, W_MAIN)
    c.disc(0.50, 0.50, 0.058)
    for a in (0.6, 2.7, 4.6):
        c.parasite(0.50 + 0.165 * math.cos(a), 0.50 + 0.165 * math.sin(a),
                   0.185, rot=math.degrees(a) + 90)
    for a in (-0.4, 1.7, 3.7):
        px, py = 0.50 + 0.350 * math.cos(a), 0.50 + 0.350 * math.sin(a)
        c.parasite(px, py, 0.150, rot=math.degrees(a) + 90)
        c.ell(px, py, 0.108, 0.052, math.degrees(a) + 90, W_FINE, dash=[3, 4])


def invasion_06(c):
    """A doorway in the membrane, with a parasite going through it."""
    c.arc(0.50, 0.50, 0.320, 300, 300, W_MAIN)
    c.disc(0.34, 0.62, 0.050)
    c.parasite(0.76, 0.26, 0.28, rot=45)
    c.parasite(0.52, 0.50, 0.24, rot=45)
    c.line(0.62, 0.20, 0.86, 0.44, W_FINE, dash=[4, 5])


def invasion_07(c):
    """Counted: three attached outside, two inside, one host nucleus."""
    c.ell(0.50, 0.54, 0.330, 0.290, 0, W_MAIN)
    c.disc(0.66, 0.44, 0.050)
    c.parasite(0.38, 0.50, 0.20, rot=20)
    c.parasite(0.44, 0.68, 0.20, rot=-25)
    for a, r in ((-2.2, 0.36), (-0.9, 0.36), (1.1, 0.34)):
        px, py = 0.50 + r * math.cos(a), 0.54 + r * math.sin(a) * 0.88
        c.ell(px, py, 0.092, 0.034, math.degrees(a) + 90, W_SEC)
        c.line(px, py, 0.50 + r * 0.80 * math.cos(a),
               0.54 + r * 0.80 * math.sin(a) * 0.88, W_FINE)


def invasion_08(c):
    """Two-colour stain read as two outlines: solid in, hollow out."""
    c.circ(0.50, 0.50, 0.310, W_SEC)
    c.circ(0.50, 0.50, 0.400, W_FINE, dash=[6, 7])
    c.disc(0.50, 0.50, 0.055)
    for a in (-1.05, 1.05, math.pi):
        c.parasite(0.50 + 0.175 * math.cos(a), 0.50 + 0.175 * math.sin(a),
                   0.185, rot=math.degrees(a) + 90)
    for a in (0.0, 2.1, 4.2):
        c.ell(0.50 + 0.355 * math.cos(a), 0.50 + 0.355 * math.sin(a),
              0.090, 0.034, math.degrees(a) + 90, W_SEC)


# =====================================================================
# replication -- endodyogeny: one parasite becomes two, in the vacuole
# =====================================================================

def replication_01(c):
    """Two daughters forming inside the mother, inside the vacuole."""
    vacuole(c, 0.50, 0.50, 0.380, W_SEC, dash=[7, 8])
    c.ell(0.50, 0.50, 0.290, 0.185, 0, W_MAIN)
    for dx in (-0.105, 0.105):
        c.ell(0.50 + dx, 0.50, 0.095, 0.135, 0, W_SEC)
        c.disc(0.50 + dx, 0.50, 0.040)


def replication_02(c):
    """The rosette: four daughters, apexes in, inside one vacuole."""
    vacuole(c, 0.50, 0.50, 0.400, W_SEC)
    rosette(c, 0.50, 0.50, 0.185, 4, L=0.30)
    c.disc(0.50, 0.50, 0.030)


def replication_03(c):
    """One becomes two: the count doubling across the arrow."""
    vacuole(c, 0.25, 0.50, 0.200, W_FINE)
    c.parasite(0.25, 0.50, 0.235, rot=90)
    c.arrow(0.44, 0.50, 0.58, 0.50, W_SEC, head=0.060)
    vacuole(c, 0.76, 0.50, 0.215, W_FINE)
    c.parasite(0.70, 0.50, 0.235, rot=90)
    c.parasite(0.82, 0.50, 0.235, rot=90)


def replication_04(c):
    """Budding: two daughter outlines nested in one mother."""
    c.ell(0.50, 0.50, 0.230, 0.360, 0, W_MAIN)
    c.disc(0.50, 0.84, 0.045)
    for dx in (-0.082, 0.082):
        c.ell(0.50 + dx, 0.44, 0.072, 0.215, 0, W_SEC)
        c.disc(0.50 + dx, 0.34, 0.034)


def replication_05(c):
    """1, then 2, then 4 — a doubling series in three vacuoles."""
    vacuole(c, 0.18, 0.50, 0.150, W_FINE)
    c.parasite(0.18, 0.50, 0.180, rot=90)
    vacuole(c, 0.50, 0.50, 0.180, W_FINE)
    for dx in (-0.052, 0.052):
        c.parasite(0.50 + dx, 0.50, 0.180, rot=90)
    vacuole(c, 0.84, 0.50, 0.200, W_FINE)
    rosette(c, 0.84, 0.50, 0.088, 4, L=0.155)
    for x in (0.335, 0.665):
        c.arrow(x - 0.030, 0.50, x + 0.030, 0.50, W_FINE, head=0.038)


def replication_06(c):
    """Eight in a rosette: a mature vacuole, one division line."""
    vacuole(c, 0.50, 0.50, 0.415, W_SEC)
    rosette(c, 0.50, 0.50, 0.215, 8, L=0.245, fat=0.26)
    c.circ(0.50, 0.50, 0.048, W_FINE)


def replication_07(c):
    """A mother splitting down the middle into two daughters."""
    vacuole(c, 0.50, 0.50, 0.410, W_FINE, dash=[7, 8])
    c.parasite(0.34, 0.50, 0.400, rot=90)
    c.parasite(0.66, 0.50, 0.400, rot=90)
    c.line(0.50, 0.16, 0.50, 0.84, W_FINE, dash=[5, 6])
    c.disc(0.34, 0.60, 0.042)
    c.disc(0.66, 0.60, 0.042)


def replication_08(c):
    """Inside a host cell: one vacuole holding four parasites."""
    c.ell(0.50, 0.52, 0.420, 0.360, 0, W_SEC)
    c.disc(0.20, 0.34, 0.052)
    vacuole(c, 0.55, 0.54, 0.245, W_MAIN)
    rosette(c, 0.55, 0.54, 0.115, 4, L=0.195)


# =====================================================================
# manifest
# =====================================================================

GROUPS = {
    "queue": (
        [queue_01, queue_02, queue_03, queue_04,
         queue_05, queue_06, queue_07, queue_08],
        "queue -- the SAME settings across many plates",
        ["One settings card feeding a column of identical plates.",
         "A stack of identical plates, each stamped with the same gear.",
         "A waiting line: plates queued behind a 'now running' marker.",
         "One recipe, many runs: a settings sheet cloned down the stack.",
         "Plates on a belt, all carrying the same mark.",
         "A numbered running order down one column of plates.",
         "Hopper: identical plates dropping through one set of settings.",
         "Same settings, ticked off plate after plate."],
    ),
    "batch": (
        [batch_01, batch_02, batch_03, batch_04,
         batch_05, batch_06, batch_07, batch_08],
        "batch -- ARBITRARY module + plate combinations, run in sequence",
        ["A schedule: different module marks against different plates.",
         "A chain of unlike jobs, run one after another.",
         "A worklist of mixed jobs with a run button.",
         "Cards in a hopper, each a different module.",
         "Overnight: a clock over a stack of unlike jobs.",
         "Many modules x many plates, folded into one run.",
         "A numbered running order of unlike steps.",
         "A pipeline of different stages emptying into one output."],
    ),
    "invasion": (
        [invasion_01, invasion_02, invasion_03, invasion_04,
         invasion_05, invasion_06, invasion_07, invasion_08],
        "invasion -- attached (outside) vs invaded (inside)",
        ["Host cell: outlined parasites on the membrane, solid ones within.",
         "One parasite caught halfway through the membrane.",
         "Two panels: attached on the left, invaded on the right.",
         "The moving junction: a parasite pinching through the membrane.",
         "Outside stain vs inside stain: a ring around the attached ones.",
         "A doorway in the membrane, with a parasite going through it.",
         "Counted: three attached outside, two inside, one host nucleus.",
         "Two-colour stain read as two outlines: solid in, hollow out."],
    ),
    "replication": (
        [replication_01, replication_02, replication_03, replication_04,
         replication_05, replication_06, replication_07, replication_08],
        "replication -- endodyogeny: one parasite becomes two, in the vacuole",
        ["Two daughters forming inside the mother, inside the vacuole.",
         "The rosette: four daughters, apexes in, inside one vacuole.",
         "One becomes two: the count doubling across the arrow.",
         "Budding: two daughter outlines nested in one mother.",
         "1, then 2, then 4 -- a doubling series in three vacuoles.",
         "Eight in a rosette: a mature vacuole, one division line.",
         "A mother splitting down the middle into two daughters.",
         "Inside a host cell: one vacuole holding four parasites."],
    ),
}


def main(outdir: str) -> None:
    for name, (fns, title, notes) in GROUPS.items():
        folder = os.path.join(outdir, name)
        os.makedirs(folder, exist_ok=True)
        images = []
        for i, fn in enumerate(fns, start=1):
            path = os.path.join(folder, f"{name}_{i:02d}.png")
            render(fn, path)
            from PySide6.QtGui import QImage
            images.append(QImage(path))
            print("wrote", path)
        contact_sheet(images, os.path.join(folder, "_sheet_dark.png"),
                      DARK_BG, cols=4, note=title)
        contact_sheet(images, os.path.join(folder, "_sheet_light.png"),
                      LIGHT_BG, cols=4, note=title)
        with open(os.path.join(folder, "CONCEPTS.md"), "w") as handle:
            handle.write(f"# {name} -- {len(fns)} candidate concepts\n\n")
            handle.write(
                "White artwork on transparent background, 1024x1024 RGBA,\n"
                "house style of `plaque.png` / `measure.png`.\n"
                "Candidates for review only; the installed icon is whichever\n"
                "one was copied to `spacr/resources/icons/`.\n\n")
            for i, note in enumerate(notes, start=1):
                handle.write(f"{i}. **{name}_{i:02d}** - {note}\n")
            handle.write(
                "\nContact sheets: `_sheet_dark.png` (white ink on #14161a)\n"
                "and `_sheet_light.png` (the same alpha masks tinted dark on\n"
                "#f5f6f8, which is how a light theme re-inks them).\n"
                "\nRegenerate with:\n"
                "`python _generators/group_queue_batch_invasion_replication.py`\n")


if __name__ == "__main__":
    default = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    main(sys.argv[1] if len(sys.argv) > 1 else default)
