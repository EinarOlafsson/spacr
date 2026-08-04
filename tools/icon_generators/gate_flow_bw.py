"""The gate editor as a flow-cytometry plot, from the image the user supplied.

They uploaded a real FSC-A / SSC-A density plot: two axes with K-scaled
ticks, a cloud dense in the middle and sparse at the edges, a polygon gate
drawn by hand round the main population, and the percentage inside it
printed beside the boundary. They asked for something more like that.

The existing ten in this folder each pick one *idea* about gating. These six
instead reproduce the picture a flow user already knows, because that
recognition is the point: someone who has drawn a gate in FlowJo should see
this and know what the app does before reading the label.

What makes it read as flow rather than as a generic scatter, in order of how
much each contributes:

1. **A density cloud, not scattered points.** Dense core, sparse halo. A
   uniform sprinkle reads as a scatter plot; the gradient is what says
   "hundreds of thousands of events".
2. **The polygon is irregular and closed by hand** -- seven or eight
   vertices at no particular angles. A rectangle or an ellipse reads as an
   automatic threshold, which is the one thing this app is not.
3. **Two axes with an origin corner.** Flow plots are always framed bottom
   and left, never boxed.
4. **The percentage.** In the original it is the only text and it sits just
   inside the boundary. Kept as a short tick-mark glyph at 48 px rather than
   real digits, which the house style has no way to draw legibly.

Numbered from 11 so the ten already in the folder survive. Cloud points come
from a fixed table, not an RNG, so the set regenerates identically.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _draw import Cv, W_MAIN, W_SEC, W_FINE  # noqa: E402

#: Plot frame in normalised coordinates: bottom-left origin, like the
#: supplied image. Left and bottom only -- a flow plot is never boxed.
_X0, _Y0, _X1, _Y1 = 0.20, 0.82, 0.90, 0.16


def _axes(c, w=W_SEC, ticks=4):
    """The two axes and their ticks, with an arrow on x like the original."""
    c.line(_X0, _Y0, _X1, _Y0, w=w)          # x
    c.line(_X0, _Y0, _X0, _Y1, w=w)          # y
    for i in range(1, ticks + 1):
        t = i / (ticks + 1.0)
        x = _X0 + (_X1 - _X0) * t
        y = _Y0 + (_Y1 - _Y0) * t
        c.line(x, _Y0, x, _Y0 + 0.018, w=W_FINE)
        c.line(_X0 - 0.018, y, _X0, y, w=W_FINE)


#: The gate: eight vertices at deliberately uneven angles, tracing the main
#: population the way a hand does -- wide over the bulk, pulled in under it,
#: and a long shallow run along the bottom where the debris is excluded.
_GATE = (
    (0.285, 0.300), (0.430, 0.215), (0.610, 0.238), (0.755, 0.300),
    (0.828, 0.430), (0.845, 0.560), (0.700, 0.700), (0.360, 0.700),
    (0.300, 0.520),
)

#: The cloud. Two populations, each a 2-D Gaussian -- the main one and the
#: small debris cluster low-left the supplied plot also has.
#:
#: The first version of this drew concentric rings of points, which produced
#: a rosette: perfectly radially symmetric, and the user correctly called it
#: "a weird star shape". Real flow data has no symmetry at all. Points now
#: come from a seeded normal distribution, which is reproducible (fixed seed,
#: rolled once at import) and looks like data because it IS distributed like
#: data. Sizes vary too -- a cloud of identical dots reads as a pattern.
_CLOUD_POPULATIONS = (
    # (cx, cy, sx, sy, rho, n)  -- rho tilts the ellipse, which the main
    # population in the supplied plot clearly has.
    (0.520, 0.520, 0.135, 0.105, 0.45, 150),
    (0.300, 0.700, 0.045, 0.035, 0.10, 30),
)


def _cloud_points():
    """Deterministic scatter: a fixed seed, rolled once."""
    import random
    rng = random.Random(20260804)
    pts = []
    for cx, cy, sx, sy, rho, n in _CLOUD_POPULATIONS:
        for _ in range(n):
            u, v = rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)
            # correlate the two axes so the cloud leans, as real FSC/SSC does
            x = cx + sx * u
            y = cy + sy * (rho * u + (1.0 - rho ** 2) ** 0.5 * v)
            if _X0 + 0.02 < x < _X1 - 0.01 and _Y1 + 0.01 < y < _Y0 - 0.02:
                pts.append((x, y, rng.uniform(0.006, 0.013)))
    return tuple(pts)


_CLOUD = _cloud_points()


def _cloud(c, rings=None, dense=True):
    """Draw the scatter. `rings`/`dense` kept so callers need no change."""
    step = 1 if dense else 2
    for x, y, r in _CLOUD[::step]:
        c.disc(x, y, r)


def _pct(c, x=0.78, y=0.470):
    """The percentage the original prints just inside the boundary.

    Two short bars rather than digits: at 48 px real numerals are illegible
    and the house style ships no text, but the *presence* of a label there is
    part of what makes the picture read as a gate report.
    """
    c.bar(x, y, 0.075, 0.016, filled=True)
    c.bar(x, y + 0.036, 0.052, 0.016, filled=True)


def gate_editor_11(c):
    """Density cloud, hand-drawn polygon gate, percentage -- the full plot."""
    _axes(c)
    _cloud(c)
    c.polyline(_GATE, w=W_MAIN, close=True)
    _pct(c)


def gate_editor_12(c):
    """The same without the percentage: cloud, axes, gate."""
    _axes(c)
    _cloud(c)
    c.polyline(_GATE, w=W_MAIN, close=True)


def gate_editor_13(c):
    """The gate solid over the cloud -- what the gate SELECTS, filled."""
    _axes(c)
    _cloud(c)
    c.polyline(_GATE, w=W_MAIN, close=True, filled=True)


def gate_editor_14(c):
    """Half-drawn: the hand is still closing the polygon.

    The open edge and the vertex handles are what say a human is drawing
    this, which is the line this app holds against every automatic threshold.
    """
    _axes(c)
    _cloud(c)
    c.polyline(_GATE[:6], w=W_MAIN)
    for x, y in _GATE[:6]:
        c.disc(x, y, 0.017)


def gate_editor_15(c):
    """Gate inside gate: the second drawn on what the first kept."""
    _axes(c)
    _cloud(c)
    c.polyline(_GATE, w=W_SEC, close=True)
    inner = tuple((0.5 + (x - 0.5) * 0.58, 0.5 + (y - 0.5) * 0.58)
                  for x, y in _GATE)
    c.polyline(inner, w=W_MAIN, close=True)


def gate_editor_16(c):
    """No axes -- cloud and gate only. The most whitespace, best at 16 px."""
    _cloud(c, rings=6)
    c.polyline(_GATE, w=W_MAIN, close=True)


ENTRIES = [
    ("Density cloud, a hand-drawn polygon gate and the percentage inside "
     "it -- the flow plot in full.", gate_editor_11),
    ("Cloud, axes and gate, without the percentage.", gate_editor_12),
    ("The gate filled: what it selects, solid over the cloud.",
     gate_editor_13),
    ("Half-drawn -- the polygon still open, vertex handles showing the hand.",
     gate_editor_14),
    ("A gate drawn inside a gate, on what the first one kept.",
     gate_editor_15),
    ("Cloud and gate with no axes; the most whitespace and the best of these "
     "at 16 px.", gate_editor_16),
]


if __name__ == "__main__":
    # Not emit_groups: it regenerates a folder from scratch and would delete
    # the ten already there. These are additions from 11.
    from _draw import render

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.abspath(os.path.join(here, "..", "gate_editor"))
    os.makedirs(out, exist_ok=True)
    lines = []
    for n, (desc, fn) in enumerate(ENTRIES, start=11):
        name = "gate_editor_%02d" % n
        render(fn, os.path.join(out, name + ".png"))
        lines.append("%d. **%s** - %s" % (n, name, desc))
        print("wrote", name + ".png")
    with open(os.path.join(out, "CONCEPTS.md"), "a", encoding="utf-8") as fh:
        fh.write("\n\nFlow-plot set, drawn from a real FSC-A / SSC-A density\n"
                 "plot the user supplied. Where 01-10 each pick one idea\n"
                 "about gating, these reproduce the picture a flow user\n"
                 "already recognises.\n\n")
        fh.write("\n".join(lines) + "\n")
    print("appended", len(lines), "entries to CONCEPTS.md")
