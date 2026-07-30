#!/usr/bin/env python3
"""Candidate spaCR icons: ml_analyze, plaque, recruitment, regression.

Ten conceptually different designs per icon.  White-on-transparent flat vector
art in the house style set by plaque.png / measure.png.

Run standalone (deterministic - every random draw is seeded):

    python group_ml_plaque_recruitment_regression.py [OUTDIR]

Default OUTDIR is the backup_icons directory two levels up from this file.
Writes <OUTDIR>/<name>/<name>_NN.png plus CONCEPTS.md and the two contact
sheets.  It never touches anything in spacr/resources/icons/*.png.
"""

from __future__ import annotations

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _draw import N, W_FINE, W_MAIN, W_SEC, Cv, contact_sheet  # noqa: E402

DARK_BG = "#14161a"
LIGHT_BG = "#f5f6f8"

TAU = math.pi * 2


def rng(seed):
    return random.Random(seed)


# =====================================================================
# ml_analyze  -- machine-learning feature analysis
# =====================================================================

def ml_01(c):
    """Decision tree whose nodes are cells."""
    c.line(0.50, 0.20, 0.27, 0.46, W_SEC)
    c.line(0.50, 0.20, 0.73, 0.46, W_SEC)
    c.line(0.27, 0.46, 0.16, 0.76, W_SEC)
    c.line(0.27, 0.46, 0.40, 0.76, W_SEC)
    c.line(0.73, 0.46, 0.60, 0.76, W_SEC)
    c.line(0.73, 0.46, 0.84, 0.76, W_SEC)
    c.cell(0.50, 0.20, 0.105, w=W_SEC, nuc=0.40)
    c.cell(0.27, 0.46, 0.085, w=W_SEC, nuc=0.42)
    c.cell(0.73, 0.46, 0.085, w=W_SEC, nuc=0.42)
    for x in (0.16, 0.60):
        c.disc(x, 0.79, 0.072)
    for x in (0.40, 0.84):
        c.circ(x, 0.79, 0.066, W_SEC)


def ml_02(c):
    """Neural network: a cell feeds three fully connected layers."""
    c.cell(0.135, 0.500, 0.115, w=W_SEC, nuc=0.40)
    c.arrow(0.265, 0.500, 0.345, 0.500, W_SEC, head=0.048)
    xs = (0.435, 0.665, 0.895)
    L1 = [0.250, 0.500, 0.750]
    L2 = [0.190, 0.400, 0.610, 0.820]
    L3 = [0.360, 0.640]
    for a in L1:
        for b in L2:
            c.line(xs[0], a, xs[1], b, W_FINE)
    for b in L2:
        for d in L3:
            c.line(xs[1], b, xs[2], d, W_FINE)
    for a in L1:
        c.circ(xs[0], a, 0.055, W_SEC)
    for b in L2:
        c.disc(xs[1], b, 0.050)
    for d in L3:
        c.circ(xs[2], d, 0.060, W_SEC)


def ml_03(c):
    """Cell distilled into a feature vector of bars."""
    c.cell(0.205, 0.50, 0.180, w=W_SEC, nuc=0.34, nuc_off=(-0.22, -0.18))
    c.disc(0.268, 0.588, 0.030)
    c.circ(0.108, 0.596, 0.034, W_FINE)
    c.arrow(0.415, 0.50, 0.520, 0.50, W_SEC, head=0.052)
    ws = [0.290, 0.155, 0.335, 0.105, 0.245]
    y = 0.190
    for i, wv in enumerate(ws):
        c.bar(0.585, y, wv, 0.082, filled=(i % 2 == 0), w=W_SEC)
        y += 0.128
    c.polyline([(0.578, 0.176), (0.556, 0.176), (0.556, 0.824), (0.578, 0.824)], w=W_FINE)
    c.polyline([(0.938, 0.176), (0.960, 0.176), (0.960, 0.824), (0.938, 0.824)], w=W_FINE)


def ml_04(c):
    """Ranked feature-importance bars beside the cell they came from."""
    c.cell(0.190, 0.190, 0.145, w=W_SEC, nuc=0.38)
    c.line(0.100, 0.400, 0.100, 0.905, W_SEC)
    ws = [0.72, 0.55, 0.42, 0.26, 0.14]
    y = 0.442
    for i, wv in enumerate(ws):
        c.bar(0.122, y, 0.075 + wv * 1.06, 0.076, filled=(i < 2), w=W_SEC)
        y += 0.108
    c.arrow(0.190, 0.348, 0.190, 0.418, W_SEC, head=0.048)


def ml_05(c):
    """Confusion matrix: cells sorted into a 2x2 grid, diagonal correct."""
    x0, y0, s = 0.290, 0.290, 0.330
    for i in range(3):
        c.line(x0 + i * s, y0, x0 + i * s, y0 + 2 * s, W_SEC)
        c.line(x0, y0 + i * s, x0 + 2 * s, y0 + i * s, W_SEC)
    # class headers: a solid-nucleus cell and a hollow cell
    c.cell(x0 + s * 0.5, 0.145, 0.088, w=W_SEC, nuc=0.62)
    c.cell(x0 + s * 1.5, 0.145, 0.088, w=W_SEC, nuc=0.0)
    c.cell(0.145, y0 + s * 0.5, 0.088, w=W_SEC, nuc=0.62)
    c.cell(0.145, y0 + s * 1.5, 0.088, w=W_SEC, nuc=0.0)
    # diagonal = correct (many), off-diagonal = confused (few)
    c.disc(x0 + s * 0.5, y0 + s * 0.5, 0.108)
    c.disc(x0 + s * 1.5, y0 + s * 1.5, 0.108)
    c.circ(x0 + s * 1.5, y0 + s * 0.5, 0.052, W_SEC)
    c.circ(x0 + s * 0.5, y0 + s * 1.5, 0.052, W_SEC)


def ml_06(c):
    """Two feature clusters split by a decision boundary."""
    r = rng(6)
    c.axes(0.135, 0.115, 0.905, 0.885, W_SEC)
    c.line(0.20, 0.845, 0.855, 0.185, W_MAIN, dash=[2.6, 2.2])
    for _ in range(9):
        x = 0.24 + r.random() * 0.36
        y = 0.20 + r.random() * 0.34
        if y > 0.86 - 1.0 * (x - 0.20) - 0.06:
            continue
        c.disc(x, y, 0.040)
    for _ in range(11):
        x = 0.42 + r.random() * 0.40
        y = 0.44 + r.random() * 0.36
        c.circ(x, y, 0.040, W_SEC)


def ml_07(c):
    """Convolution window sliding over an image patch."""
    x0, y0, s = 0.070, 0.205, 0.122
    r = rng(7)
    for i in range(5):
        for j in range(5):
            x, y = x0 + i * s, y0 + j * s
            if r.random() < 0.36:
                c.rect(x + 0.027, y + 0.027, s - 0.054, s - 0.054, filled=True, r=0.010)
            else:
                c.rect(x + 0.027, y + 0.027, s - 0.054, s - 0.054, w=W_FINE * 0.85, r=0.010)
    c.rect(x0 + 1 * s + 0.006, y0 + 1 * s + 0.006, 3 * s - 0.012, 3 * s - 0.012,
           w=W_MAIN * 1.4, r=0.024)
    c.arrow(0.700, 0.500, 0.812, 0.500, W_SEC, head=0.052)
    c.rect(0.845, 0.415, 0.115, 0.170, w=W_SEC, r=0.024)
    c.disc(0.9025, 0.500, 0.034)


def ml_08(c):
    """Classifier funnel sorting cells into two bins."""
    for x, y, rr in ((0.30, 0.115, 0.062), (0.50, 0.088, 0.070), (0.70, 0.125, 0.058)):
        c.cell(x, y, rr, w=W_SEC, nuc=0.42)
    c.polyline([(0.13, 0.255), (0.87, 0.255), (0.585, 0.545), (0.585, 0.635)], w=W_MAIN)
    c.polyline([(0.13, 0.255), (0.415, 0.545), (0.415, 0.635)], w=W_MAIN)
    c.polyline([(0.155, 0.735), (0.155, 0.905), (0.445, 0.905), (0.445, 0.735)], w=W_SEC)
    c.polyline([(0.555, 0.735), (0.555, 0.905), (0.845, 0.905), (0.845, 0.735)], w=W_SEC)
    c.disc(0.235, 0.825, 0.052)
    c.disc(0.365, 0.825, 0.052)
    c.circ(0.635, 0.825, 0.052, W_SEC)
    c.circ(0.765, 0.825, 0.052, W_SEC)


def ml_09(c):
    """Random forest: an ensemble of three small trees voting."""
    def tree(ox, oy, sc, solid):
        c.line(ox, oy, ox - 0.070 * sc, oy + 0.135 * sc, W_SEC)
        c.line(ox, oy, ox + 0.070 * sc, oy + 0.135 * sc, W_SEC)
        c.line(ox - 0.070 * sc, oy + 0.135 * sc, ox - 0.125 * sc, oy + 0.285 * sc, W_SEC)
        c.line(ox - 0.070 * sc, oy + 0.135 * sc, ox - 0.015 * sc, oy + 0.285 * sc, W_SEC)
        c.line(ox + 0.070 * sc, oy + 0.135 * sc, ox + 0.125 * sc, oy + 0.285 * sc, W_SEC)
        c.disc(ox, oy, 0.042 * sc)
        for dx in (-0.070, 0.070):
            c.circ(ox + dx * sc, oy + 0.135 * sc, 0.036 * sc, W_SEC)
        for dx in (-0.125, -0.015, 0.125):
            if solid:
                c.disc(ox + dx * sc, oy + 0.285 * sc, 0.034 * sc)
            else:
                c.circ(ox + dx * sc, oy + 0.285 * sc, 0.030 * sc, W_SEC)
    tree(0.175, 0.115, 1.0, True)
    tree(0.500, 0.115, 1.0, False)
    tree(0.825, 0.115, 1.0, True)
    for x in (0.175, 0.500, 0.825):
        c.line(x, 0.470, x, 0.600, W_FINE, dash=[2.4, 2.2])
    c.line(0.175, 0.600, 0.825, 0.600, W_SEC)
    c.arrow(0.500, 0.600, 0.500, 0.740, W_SEC, head=0.052)
    c.cell(0.500, 0.855, 0.115, w=W_SEC, nuc=0.62)


def ml_10(c):
    """ROC curve with the chance diagonal."""
    c.axes(0.145, 0.105, 0.915, 0.870, W_SEC)
    c.line(0.145, 0.870, 0.915, 0.105, W_FINE, dash=[2.6, 2.4])
    c.smooth([(0.145, 0.870), (0.235, 0.560), (0.395, 0.310), (0.605, 0.195),
              (0.915, 0.150)], w=W_MAIN)
    for x, y in ((0.235, 0.560), (0.395, 0.310), (0.605, 0.195)):
        c.disc(x, y, 0.036)
    c.cell(0.745, 0.680, 0.100, w=W_SEC, nuc=0.40)


# =====================================================================
# plaque
# =====================================================================

def plq_01(c):
    """Petri dish full of lens-shaped plaques (the existing idea, redrawn)."""
    r = rng(101)
    c.circ(0.50, 0.50, 0.435, W_FINE)
    pts = []
    tries = 0
    while len(pts) < 15 and tries < 4000:
        tries += 1
        a = r.random() * TAU
        rad = 0.40 * math.sqrt(r.random())
        x, y = 0.50 + rad * math.cos(a), 0.50 + rad * math.sin(a)
        sz = 0.045 + r.random() * 0.055
        if any((x - px) ** 2 + (y - py) ** 2 < (sz + ps) ** 2 for px, py, ps in pts):
            continue
        pts.append((x, y, sz))
    for x, y, sz in pts:
        c.lens(x, y, sz, sz * (0.30 + r.random() * 0.28), rot=r.random() * 180.0)


def plq_02(c):
    """Six-well plate, wells carrying different plaque loads."""
    r = rng(102)
    c.rect(0.045, 0.175, 0.910, 0.650, w=W_SEC, r=0.045)
    for j in range(2):
        for i in range(3):
            cx = 0.205 + i * 0.295
            cy = 0.355 + j * 0.290
            c.circ(cx, cy, 0.118, W_SEC)
            n = (2, 4, 7, 3, 6, 9)[j * 3 + i]
            placed = []
            t = 0
            while len(placed) < n and t < 900:
                t += 1
                a = r.random() * TAU
                rad = 0.082 * math.sqrt(r.random())
                x, y = cx + rad * math.cos(a), cy + rad * math.sin(a)
                if any((x - px) ** 2 + (y - py) ** 2 < 0.0028 for px, py in placed):
                    continue
                placed.append((x, y))
                c.disc(x, y, 0.0175)


def plq_03(c):
    """Cross-section: a monolayer with a lysed crater punched through it."""
    floor = 0.800
    c.polyline([(0.080, 0.265), (0.080, floor), (0.920, floor), (0.920, 0.265)], w=W_MAIN)
    c.line(0.040, 0.265, 0.122, 0.265, W_SEC)
    c.line(0.878, 0.265, 0.960, 0.265, W_SEC)
    c.line(0.092, 0.370, 0.908, 0.370, W_FINE, dash=[3.0, 2.6])
    for i in range(7):
        if i in (2, 3, 4):
            continue
        x = 0.145 + i * 0.118
        c.arc(x, floor, 0.059, 0.0, 180.0, W_SEC)
        c.disc(x, floor - 0.034, 0.022)
    for x in (0.322, 0.664):
        c.line(x, floor - 0.012, x, 0.455, W_FINE, dash=[2.8, 2.6])
    c.line(0.322, 0.495, 0.664, 0.495, W_FINE)
    c.arrow(0.493, 0.495, 0.326, 0.495, W_FINE, head=0.046, tail=False)
    c.arrow(0.493, 0.495, 0.660, 0.495, W_FINE, head=0.046, tail=False)
    c.parasite(0.384, 0.742, 0.120, rot=-44, fat=0.32)
    c.parasite(0.606, 0.736, 0.120, rot=40, fat=0.32)


def plq_04(c):
    """One plaque magnified: cleared centre ringed by an infection front."""
    r = rng(104)
    c.circ(0.50, 0.50, 0.462, W_FINE)
    c.circ(0.50, 0.50, 0.318, W_SEC, dash=[3.0, 2.6])
    c.disc(0.50, 0.50, 0.150)
    for i in range(7):
        a = TAU * i / 7 + 0.22
        rad = 0.245
        c.parasite(0.50 + rad * math.cos(a), 0.50 + rad * math.sin(a),
                   0.120, rot=math.degrees(a) + 90, fat=0.20)
    for i in range(11):
        a = TAU * i / 11 + 0.14
        rad = 0.392 + (r.random() - 0.5) * 0.026
        x, y = 0.50 + rad * math.cos(a), 0.50 + rad * math.sin(a)
        c.ell(x, y, 0.040, 0.032, math.degrees(a), W_FINE)
        c.disc(x, y, 0.014)


def plq_05(c):
    """Dish with a counting grid laid over it."""
    r = rng(105)
    c.circ(0.50, 0.50, 0.440, W_SEC)
    c.clip_circle(0.50, 0.50, 0.440)
    for i in range(1, 5):
        v = 0.06 + i * 0.176
        c.line(v, 0.02, v, 0.98, W_FINE)
        c.line(0.02, v, 0.98, v, W_FINE)
    c.unclip()
    placed = []
    t = 0
    while len(placed) < 11 and t < 3000:
        t += 1
        a = r.random() * TAU
        rad = 0.375 * math.sqrt(r.random())
        x, y = 0.50 + rad * math.cos(a), 0.50 + rad * math.sin(a)
        if any((x - px) ** 2 + (y - py) ** 2 < 0.021 for px, py in placed):
            continue
        placed.append((x, y))
        c.disc(x, y, 0.0345)


def plq_06(c):
    """Time course: the same plaque widening across three dishes."""
    for i, (cx, rr) in enumerate(((0.175, 0.032), (0.500, 0.068), (0.825, 0.108))):
        c.circ(cx, 0.390, 0.155, W_SEC)
        c.disc(cx, 0.390, rr)
        if i:
            c.circ(cx, 0.390, rr + 0.036, W_FINE, dash=[3.2, 2.8])
    c.line(0.075, 0.760, 0.925, 0.760, W_SEC)
    c.arrow(0.860, 0.760, 0.930, 0.760, W_SEC, head=0.050, tail=False)
    for x in (0.175, 0.500, 0.825):
        c.line(x, 0.570, x, 0.690, W_FINE, dash=[2.6, 2.4])
        c.line(x, 0.716, x, 0.804, W_SEC)


def plq_07(c):
    """Confluent monolayer with the plaques as clearings in the texture."""
    c.circ(0.50, 0.50, 0.455, W_SEC)
    holes = [(0.335, 0.360, 0.115), (0.655, 0.480, 0.095), (0.455, 0.700, 0.088)]
    a = 0.0775                      # hex circumradius
    dx = a * math.sqrt(3.0)
    dy = a * 1.5
    c.clip_circle(0.50, 0.50, 0.442)
    j = -1
    y = 0.03
    while y < 0.98:
        j += 1
        x = 0.03 + (dx / 2 if j % 2 else 0.0)
        while x < 0.99:
            if not any((x - hx) ** 2 + (y - hy) ** 2 < (hr + a * 1.05) ** 2
                       for hx, hy, hr in holes):
                pts = [(x + a * 0.92 * math.cos(TAU * k / 6 + math.pi / 2),
                        y + a * 0.92 * math.sin(TAU * k / 6 + math.pi / 2)) for k in range(6)]
                c.polyline(pts, w=W_FINE * 0.85, close=True)
            x += dx
        y += dy
    c.unclip()
    for hx, hy, hr in holes:
        c.circ(hx, hy, hr, W_FINE, dash=[3.2, 2.8])


def plq_08(c):
    """Stained culture flask with plaques."""
    r = rng(108)
    c.polyline([(0.235, 0.180), (0.235, 0.140), (0.360, 0.140), (0.360, 0.180)], w=W_SEC)
    c.polyline([(0.360, 0.180), (0.575, 0.245), (0.900, 0.245), (0.900, 0.870),
                (0.100, 0.870), (0.100, 0.310), (0.235, 0.180), (0.360, 0.180)], w=W_MAIN)
    c.line(0.100, 0.310, 0.360, 0.310, W_FINE)
    placed = []
    t = 0
    while len(placed) < 12 and t < 4000:
        t += 1
        x = 0.155 + r.random() * 0.690
        y = 0.340 + r.random() * 0.470
        if x < 0.36 and y < 0.36:
            continue
        if any((x - px) ** 2 + (y - py) ** 2 < 0.019 for px, py in placed):
            continue
        placed.append((x, y))
        c.lens(x, y, 0.055, 0.017 + r.random() * 0.014, rot=r.random() * 180.0)


def plq_09(c):
    """Plaque counts quantified as a bar chart beside the dish."""
    r = rng(109)
    c.circ(0.255, 0.470, 0.250, W_SEC)
    placed = []
    t = 0
    while len(placed) < 8 and t < 2000:
        t += 1
        a = r.random() * TAU
        rad = 0.190 * math.sqrt(r.random())
        x, y = 0.255 + rad * math.cos(a), 0.470 + rad * math.sin(a)
        if any((x - px) ** 2 + (y - py) ** 2 < 0.0098 for px, py in placed):
            continue
        placed.append((x, y))
        c.lens(x, y, 0.050, 0.016 + r.random() * 0.013, rot=r.random() * 180.0)
    c.arrow(0.535, 0.470, 0.620, 0.470, W_SEC, head=0.050)
    c.axes(0.672, 0.250, 0.975, 0.760, W_SEC)
    hs = [0.330, 0.205, 0.430, 0.130]
    for i, h in enumerate(hs):
        x = 0.706 + i * 0.070
        c.rect(x, 0.760 - h, 0.048, h, w=W_SEC, filled=(i % 2 == 0))


def plq_10(c):
    """A single plaque measured: diameter callipers plus a scale bar."""
    r = rng(110)
    c.circ(0.50, 0.415, 0.310, W_FINE)
    c.disc(0.50, 0.415, 0.185)
    for i in range(10):
        a = TAU * i / 10 + 0.31
        rad = 0.252 + (r.random() - 0.5) * 0.02
        c.disc(0.50 + rad * math.cos(a), 0.415 + rad * math.sin(a), 0.023)
    c.line(0.190, 0.430, 0.190, 0.790, W_SEC)
    c.line(0.810, 0.430, 0.810, 0.790, W_SEC)
    c.line(0.190, 0.735, 0.810, 0.735, W_FINE)
    c.arrow(0.500, 0.735, 0.198, 0.735, W_FINE, head=0.046, tail=False)
    c.arrow(0.500, 0.735, 0.802, 0.735, W_FINE, head=0.046, tail=False)
    c.rect(0.310, 0.880, 0.380, 0.036, filled=True, r=0.014)


# =====================================================================
# recruitment -- protein concentrating at the vacuole membrane
# =====================================================================

def rec_01(c):
    """Cytosolic protein beaded densely onto the vacuole membrane."""
    r = rng(201)
    c.circ(0.50, 0.50, 0.335, W_SEC)
    c.dot_ring(0.50, 0.50, 0.335, 16, 0.042, a0=0.06)
    for _ in range(22):
        a = r.random() * TAU
        rad = 0.430 + r.random() * 0.075
        x, y = 0.50 + rad * math.cos(a), 0.50 + rad * math.sin(a)
        if not (0.04 < x < 0.96 and 0.04 < y < 0.96):
            continue
        c.disc(x, y, 0.019)


def rec_02(c):
    """Recruitment as inward flux: arrows converging, cargo at the tips."""
    c.circ(0.50, 0.50, 0.250, W_SEC)
    c.ring(0.50, 0.50, 0.250, 0.026)
    for i in range(8):
        a = TAU * i / 8 + math.pi / 8
        ca, sa = math.cos(a), math.sin(a)
        c.arrow(0.50 + 0.465 * ca, 0.50 + 0.465 * sa,
                0.50 + 0.300 * ca, 0.50 + 0.300 * sa, W_SEC, head=0.055)
        c.disc(0.50 + 0.470 * ca, 0.50 + 0.470 * sa, 0.026)


def rec_03(c):
    """Tachyzoite inside a vacuole whose membrane is coated."""
    c.circ(0.50, 0.475, 0.335, W_SEC)
    c.circ(0.50, 0.475, 0.385, W_FINE, dash=[2.4, 2.2])
    c.dot_ring(0.50, 0.475, 0.335, 20, 0.030, a0=0.16)
    c.parasite(0.50, 0.560, 0.330, rot=-14, fat=0.30)
    c.disc(0.585, 0.505, 0.038)


def rec_04(c):
    """Line scan across the vacuole: the trace peaks at both membranes."""
    cx, cy, rr = 0.50, 0.270, 0.235
    c.circ(cx, cy, rr, W_SEC)
    c.dot_ring(cx, cy, rr, 14, 0.035, a0=0.22)
    c.line(0.095, cy, 0.925, cy, W_FINE, dash=[2.8, 2.4])
    c.arrow(0.860, cy, 0.930, cy, W_FINE, head=0.045, tail=False)
    for x in (cx - rr, cx + rr):
        c.line(x, cy + 0.055, x, 0.560, W_FINE, dash=[2.8, 2.4])
    c.axes(0.095, 0.545, 0.945, 0.905, W_SEC)
    c.smooth([(0.105, 0.885), (0.190, 0.868), (cx - rr - 0.008, 0.600),
              (0.345, 0.845), (0.50, 0.858), (0.655, 0.845),
              (cx + rr + 0.008, 0.600), (0.855, 0.868), (0.935, 0.885)], w=W_MAIN)


def rec_05(c):
    """Density gradient: dots crowd toward the membrane."""
    r = rng(205)
    c.circ(0.50, 0.50, 0.270, W_SEC)
    for count, lo, hi, rad in ((20, 0.295, 0.315, 0.0245),
                               (13, 0.360, 0.385, 0.0205),
                               (8, 0.430, 0.460, 0.0165)):
        for i in range(count):
            a = TAU * i / count + (r.random() - 0.5) * 0.18
            rr = lo + r.random() * (hi - lo)
            x, y = 0.50 + rr * math.cos(a), 0.50 + rr * math.sin(a)
            if not (0.03 < x < 0.97 and 0.03 < y < 0.97):
                continue
            c.disc(x, y, rad)
    c.parasite(0.50, 0.555, 0.245, rot=-12)


def rec_06(c):
    """Protein trafficking in along curved tracks and landing on the rim."""
    c.circ(0.50, 0.50, 0.255, W_SEC)
    c.ring(0.50, 0.50, 0.255, 0.026)
    for i in range(6):
        a = TAU * i / 6 + 0.26
        ca, sa = math.cos(a), math.sin(a)
        p1 = (0.50 + 0.455 * math.cos(a + 0.78), 0.50 + 0.455 * math.sin(a + 0.78))
        p2 = (0.50 + 0.415 * math.cos(a + 0.44), 0.50 + 0.415 * math.sin(a + 0.44))
        p3 = (0.50 + 0.375 * ca, 0.50 + 0.375 * sa)
        c.smooth([p1, p2, p3], w=W_SEC)
        c.arrow(p3[0], p3[1], 0.50 + 0.310 * ca, 0.50 + 0.310 * sa,
                W_SEC, head=0.050, tail=False)
        c.disc(p1[0], p1[1], 0.030)
        c.disc(0.50 + 0.255 * ca, 0.50 + 0.255 * sa, 0.038)


def rec_07(c):
    """Partial recruitment: only one face of the vacuole is decorated."""
    c.circ(0.50, 0.50, 0.320, W_SEC)
    for i in range(9):
        a = -math.pi * 0.46 + math.pi * 0.92 * i / 8
        c.disc(0.50 + 0.320 * math.cos(a), 0.50 + 0.320 * math.sin(a), 0.044)
    for i in range(3):
        a = -math.pi * 0.30 + math.pi * 0.60 * i / 2
        c.arrow(0.50 + 0.470 * math.cos(a), 0.50 + 0.470 * math.sin(a),
                0.50 + 0.382 * math.cos(a), 0.50 + 0.382 * math.sin(a),
                W_SEC, head=0.052)
    c.parasite(0.50, 0.560, 0.290, rot=-12)


def rec_08(c):
    """Whole infected host cell: nucleus, and a haloed vacuole beside it."""
    c.ell(0.50, 0.50, 0.455, 0.395, 0.0, W_SEC)
    c.ell(0.265, 0.400, 0.135, 0.115, -18.0, W_SEC)
    c.disc(0.265, 0.400, 0.055)
    c.circ(0.615, 0.575, 0.185, W_SEC)
    c.dot_ring(0.615, 0.575, 0.185, 11, 0.033, a0=0.2)
    c.parasite(0.615, 0.625, 0.190, rot=-16)
    c.circ(0.615, 0.575, 0.245, W_FINE, dash=[2.4, 2.2])


def rec_09(c):
    """Magnified rim: the coat resolved into discrete puncta."""
    cx, cy, R = 0.360, 0.380, 0.265
    c.circ(cx, cy, R, W_SEC)
    c.dot_ring(cx, cy, R, 16, 0.026, a0=0.10)
    mx, my, mr = 0.665, 0.645, 0.245
    ux, uy = mx - cx, my - cy
    L = math.hypot(ux, uy)
    ux, uy = ux / L, uy / L
    Rz = R * 3.4
    zx, zy = mx - Rz * ux, my - Rz * uy
    a0 = math.atan2(my - zy, mx - zx)
    c.clip_circle(mx, my, mr - 0.014)
    c.circ(zx, zy, Rz, W_MAIN)
    for k in range(-2, 3):
        a = a0 + k * 0.112
        c.disc(zx + Rz * math.cos(a), zy + Rz * math.sin(a), 0.043)
    c.unclip()
    c.magnifier(mx, my, mr, ang_deg=48.0, w=W_MAIN, handle=0.36)


def rec_10(c):
    """Effector proteins docking on the membrane as Y-shaped ligands."""
    c.circ(0.50, 0.50, 0.290, W_SEC)
    for i in range(8):
        a = TAU * i / 8 - math.pi / 2
        ca, sa = math.cos(a), math.sin(a)
        bx, by = 0.50 + 0.290 * ca, 0.50 + 0.290 * sa
        sx, sy = 0.50 + 0.372 * ca, 0.50 + 0.372 * sa
        c.line(bx, by, sx, sy, W_SEC)
        px, py = -sa, ca
        for s in (1, -1):
            c.line(sx, sy,
                   sx + 0.066 * ca + s * 0.062 * px,
                   sy + 0.066 * sa + s * 0.062 * py, W_SEC)
        c.disc(bx, by, 0.034)
    c.parasite(0.50, 0.560, 0.270, rot=-12)


# =====================================================================
# regression -- phenotype vs sgRNA abundance
# =====================================================================

def _scatter(c, r, n, fn, jitter, x0=0.20, x1=0.90, rad=0.030, filled=True):
    for _ in range(n):
        x = x0 + r.random() * (x1 - x0)
        y = fn(x) + (r.random() - 0.5) * jitter
        if filled:
            c.disc(x, y, rad)
        else:
            c.circ(x, y, rad, W_SEC)


def reg_01(c):
    """Scatter with a least-squares line through it."""
    r = rng(301)
    c.axes(0.135, 0.095, 0.930, 0.880, W_SEC, ticks=3)
    c.line(0.185, 0.815, 0.895, 0.185, W_MAIN)
    _scatter(c, r, 16, lambda x: 0.815 - (x - 0.185) * 0.887, 0.155, 0.20, 0.875)


def reg_02(c):
    """Fit plus its dashed confidence band."""
    r = rng(302)
    c.axes(0.135, 0.095, 0.930, 0.880, W_SEC)

    def fit(x):
        return 0.800 - (x - 0.180) * 0.850

    def band(x):
        return 0.036 + 0.052 * ((x - 0.535) / 0.355) ** 2

    c.line(0.180, fit(0.180), 0.895, fit(0.895), W_MAIN)
    up = [(0.180 + i * 0.0715, fit(0.180 + i * 0.0715) - band(0.180 + i * 0.0715))
          for i in range(11)]
    dn = [(0.180 + i * 0.0715, fit(0.180 + i * 0.0715) + band(0.180 + i * 0.0715))
          for i in range(11)]
    c.smooth(up, w=W_FINE, dash=[3.0, 2.6])
    c.smooth(dn, w=W_FINE, dash=[3.0, 2.6])
    _scatter(c, r, 12, fit, 0.185, 0.22, 0.86, rad=0.028)


def reg_03(c):
    """Residuals: every point dropped onto the fit."""
    r = rng(303)
    c.axes(0.135, 0.095, 0.930, 0.880, W_SEC)

    def fit(x):
        return 0.795 - (x - 0.180) * 0.840

    c.line(0.180, fit(0.180), 0.900, fit(0.900), W_MAIN)
    for i in range(8):
        x = 0.220 + i * 0.092
        d = (r.random() - 0.5) * 0.36
        y = fit(x) + math.copysign(max(abs(d), 0.075), d)
        c.line(x, y, x, fit(x), W_FINE)
        c.disc(x, y, 0.033)


def reg_04(c):
    """Literal least squares: a square raised on each residual."""
    r = rng(304)
    c.axes(0.115, 0.085, 0.935, 0.885, W_SEC)

    def fit(x):
        return 0.790 - (x - 0.160) * 0.800

    c.line(0.160, fit(0.160), 0.905, fit(0.905), W_MAIN)
    for i in range(5):
        x = 0.215 + i * 0.135
        d = (r.random() - 0.5) * 0.28
        d = math.copysign(max(abs(d), 0.075), d)
        y = fit(x) + d
        s = abs(d)
        c.rect(x, min(y, fit(x)), s, s, w=W_FINE)
        c.disc(x, y, 0.028)


def reg_05(c):
    """Volcano plot: effect size against significance, hits called out."""
    r = rng(305)
    c.axes(0.105, 0.085, 0.945, 0.885, W_SEC)
    c.line(0.115, 0.600, 0.935, 0.600, W_FINE, dash=[2.6, 2.4])
    c.line(0.525, 0.885, 0.525, 0.120, W_FINE, dash=[2.6, 2.4])
    for _ in range(22):
        u = r.random() - r.random()
        x = 0.525 + u * 0.205
        y = 0.855 - abs(u) * 0.215 - r.random() * 0.050
        c.disc(x, y, 0.023)
    for side in (-1, 1):
        for k in range(4):
            m = 0.180 + 0.060 * k + r.random() * 0.030
            x = 0.525 + side * m
            y = 0.545 - (m - 0.180) * 1.45 - r.random() * 0.055
            c.circ(x, y, 0.038, W_SEC)


def reg_06(c):
    """sgRNA read-count bars with the trend fitted through their tops."""
    r = rng(306)
    c.axes(0.115, 0.085, 0.940, 0.880, W_SEC)
    tops = []
    for i in range(7):
        x = 0.165 + i * 0.106
        base = 0.760 - i * 0.082
        y = base + (r.random() - 0.5) * 0.100
        y = min(max(y, 0.185), 0.790)
        c.rect(x, y, 0.076, 0.880 - y, w=W_SEC, filled=(i % 2 == 0))
        tops.append((x + 0.038, y))
    c.line(0.170, 0.775, 0.900, 0.185, W_MAIN)
    for x, y in tops:
        c.disc(x, y, 0.027)


def reg_07(c):
    """The coefficient itself: a slope triangle on the fitted line."""
    r = rng(307)
    c.axes(0.125, 0.085, 0.935, 0.880, W_SEC)
    x0, y0, x1, y1 = 0.185, 0.800, 0.900, 0.175
    slope = (y0 - y1) / (x1 - x0)

    def fit(x):
        return y0 - (x - x0) * slope

    _scatter(c, r, 10, fit, 0.115, 0.215, 0.875, rad=0.030, filled=True)
    c.line(x0, y0, x1, y1, W_MAIN)
    ax, bx = 0.330, 0.720
    ay, by = fit(ax), fit(bx)
    c.line(ax, ay, ax, by, W_SEC)
    c.line(ax, by, bx, by, W_SEC)
    c.rect(ax, by, 0.058, 0.058, w=W_FINE)


def reg_08(c):
    """Dose-response: a sigmoid fitted to the points."""
    r = rng(308)
    c.axes(0.125, 0.090, 0.935, 0.880, W_SEC)

    def sig(x):
        t = (x - 0.520) / 0.088
        return 0.815 - 0.660 / (1.0 + math.exp(-t))

    pts = [(0.155 + i * 0.0805, sig(0.155 + i * 0.0805)) for i in range(10)]
    c.smooth(pts, w=W_MAIN)
    for i in range(8):
        x = 0.190 + i * 0.092
        c.disc(x, sig(x) + (r.random() - 0.5) * 0.085, 0.030)
    c.line(0.520, 0.880, 0.520, sig(0.520), W_FINE, dash=[2.6, 2.4])


def reg_09(c):
    """Forest plot: coefficients with confidence intervals around zero."""
    c.line(0.510, 0.095, 0.510, 0.905, W_FINE, dash=[2.8, 2.4])
    c.line(0.075, 0.905, 0.940, 0.905, W_SEC)
    rows = [(0.735, 0.180, True), (0.640, 0.150, True), (0.480, 0.120, False),
            (0.330, 0.165, True), (0.255, 0.195, False)]
    y = 0.190
    for cxv, half, sig in rows:
        c.line(cxv - half, y, cxv + half, y, W_SEC)
        for e in (cxv - half, cxv + half):
            c.line(e, y - 0.042, e, y + 0.042, W_FINE)
        if sig:
            c.disc(cxv, y, 0.044)
        else:
            c.circ(cxv, y, 0.040, W_SEC)
        y += 0.152


def reg_10(c):
    """Ranked effect sizes: a waterfall of genes with the fitted curve over it."""
    c.line(0.055, 0.500, 0.960, 0.500, W_SEC)
    vals = [0.330, 0.240, 0.165, 0.100, 0.040, -0.030, -0.100, -0.185, -0.310]
    pts = []
    for i, v in enumerate(vals):
        x = 0.125 + i * 0.096
        c.line(x, 0.500, x, 0.500 - v, W_SEC)
        if abs(v) > 0.15:
            c.disc(x, 0.500 - v, 0.040)
        else:
            c.circ(x, 0.500 - v, 0.034, W_SEC)
        pts.append((x, 0.500 - v))
    c.smooth(pts, w=W_FINE, dash=[3.0, 2.6])


# =====================================================================
GROUPS = {
    "ml_analyze": (
        [ml_01, ml_02, ml_03, ml_04, ml_05, ml_06, ml_07, ml_08, ml_09, ml_10],
        [
            "Decision tree whose nodes are cells - the classifier splitting a population.",
            "Neural network: a cell feeding three fully connected layers.",
            "Cell distilled into a feature vector of bars - featurisation.",
            "Ranked feature-importance bar chart beside the cell the features came from.",
            "Confusion matrix: cells sorted into a 2x2 grid, diagonal correct.",
            "Two feature clusters separated by a dashed decision boundary.",
            "Convolution window sliding over an image patch - CNN on raw pixels.",
            "Classifier funnel sorting cells into two labelled bins.",
            "Random forest: an ensemble of three trees voting on one cell.",
            "ROC curve above the chance diagonal - model performance.",
        ],
    ),
    "plaque": (
        [plq_01, plq_02, plq_03, plq_04, plq_05, plq_06, plq_07, plq_08, plq_09, plq_10],
        [
            "Petri dish full of lens-shaped plaques (the existing idea, redrawn).",
            "Six-well plate, each well carrying a different plaque load.",
            "Side view of the dish: a hole in the monolayer, measured, with parasites at the front.",
            "One plaque magnified: cleared centre ringed by the advancing infection front.",
            "Dish with a counting grid laid over it - plaques as countable events.",
            "Time course: the same plaque widening across three dishes.",
            "Confluent monolayer texture with the plaques as clearings in it.",
            "Stained tissue-culture flask with plaques on its growth surface.",
            "Dish plus the bar chart of counts it produces - plaque assay as readout.",
            "A single plaque measured: diameter callipers and a scale bar.",
        ],
    ),
    "recruitment": (
        [rec_01, rec_02, rec_03, rec_04, rec_05, rec_06, rec_07, rec_08, rec_09, rec_10],
        [
            "Cytosolic protein beaded densely onto the vacuole membrane.",
            "Recruitment as inward flux: arrows converging, cargo carried at the tips.",
            "Tachyzoite inside a vacuole whose membrane is coated.",
            "Line scan across the vacuole; the intensity trace peaks at both membranes.",
            "Density gradient: dots crowding progressively toward the membrane.",
            "Protein trafficking in along curved tracks and landing on the rim.",
            "Partial recruitment: only one face of the vacuole is decorated.",
            "Whole infected host cell - nucleus plus a haloed vacuole beside it.",
            "Magnified rim: the coat resolved into discrete puncta under a lens.",
            "Effector proteins docking on the membrane as Y-shaped ligands.",
        ],
    ),
    "regression": (
        [reg_01, reg_02, reg_03, reg_04, reg_05, reg_06, reg_07, reg_08, reg_09, reg_10],
        [
            "Scatter with a least-squares line through it - the classic fit.",
            "Fit plus its dashed confidence band - uncertainty on the slope.",
            "Residuals: every observation dropped onto the fit.",
            "Literal least squares: a square raised on each residual.",
            "Volcano plot: effect size against significance, hits called out.",
            "sgRNA read-count bars with the trend fitted through their tops.",
            "The coefficient itself: a slope triangle (dy/dx) on the fitted line.",
            "Dose-response: a sigmoid fitted to the points.",
            "Forest plot: per-gene coefficients with confidence intervals around zero.",
            "Ranked effect sizes: genes as stems either side of zero, dashed fit over them.",
        ],
    ),
}

LIGHT_NOTE = ("Artwork is white-on-transparent; re-tinted to dark ink here so the "
              "shape is judgeable on a light background.")


def build(outdir, only=None):
    written = []
    for name, (fns, concepts) in GROUPS.items():
        if only and name not in only:
            continue
        d = os.path.join(outdir, name)
        os.makedirs(d, exist_ok=True)
        imgs = []
        for i, fn in enumerate(fns, 1):
            p = os.path.join(d, "%s_%02d.png" % (name, i))
            c = Cv(N)
            fn(c)
            img = c.finish()
            img.save(p)
            imgs.append(img)
            written.append(p)
        cm = os.path.join(d, "CONCEPTS.md")
        with open(cm, "w") as fh:
            fh.write("# %s - candidate concepts\n\n" % name)
            fh.write("White-on-transparent flat vector, 1024x1024 RGBA, house style of\n"
                     "`plaque.png` / `measure.png`.  Ten conceptually distinct metaphors.\n\n")
            for i, t in enumerate(concepts, 1):
                fh.write("%d. %s\n" % (i, t))
            fh.write("\n_Sheets: `_sheet_dark.png` (#14161a), `_sheet_light.png` (#f5f6f8, "
                     "artwork re-tinted so it is visible). Each cell also shows a 48 px "
                     "thumbnail bottom-right._\n")
        written.append(cm)
        written.append(contact_sheet(imgs, os.path.join(d, "_sheet_dark.png"), DARK_BG))
        written.append(contact_sheet(imgs, os.path.join(d, "_sheet_light.png"), LIGHT_BG,
                                     note=LIGHT_NOTE))
    return written


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(here)
    for p in build(outdir):
        print(p)


if __name__ == "__main__":
    main()
