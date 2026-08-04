"""Illumination as a pulsar map: thin rays of unequal length from one point.

Asked for directly: "a star with thin lines of different lengths emanating
from the center. like a denser version of the voyager one star symbol."

The Voyager plaque's pulsar map is a set of straight lines radiating from a
single origin, each a different length, at irregular angles. It reads as a
star without being a five-pointed star, which is exactly why it suits
illumination: the subject is *light leaving a source unevenly*, and unequal
rays say that in one gesture. A regular starburst would say the opposite.

Six variants rather than one, because the two free parameters -- how many
rays, and how unequal -- are a matter of taste and the difference is only
visible side by side:

* ``01`` 24 rays, the reference density.
* ``02`` 36 rays, denser, closer to the plaque's own crowding.
* ``03`` 16 rays, sparser, the most legible at 16 px.
* ``04`` 24 rays over a field circle -- the correction target made explicit.
* ``05`` 24 rays with the long ones to one side: an uneven field, which is
  the defect illumination correction removes rather than the ideal.
* ``06`` 28 rays and a solid core, the strongest silhouette of the set.

Every ray is one straight line from the origin. Lengths come from a fixed
table, not an RNG, so the set regenerates identically -- the irregularity has
to survive a rebuild or it is not a design, it is a roll of the dice.

Drawn with W_FINE. The brief said thin, and at 16 px a 1024-unit stroke is
w/64 px, so W_FINE lands near a pixel and W_MAIN would fuse the rays into a
disc. That is the constraint the density choice is really made against.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _draw import Cv, W_FINE, W_SEC  # noqa: E402
from _emit import emit_groups  # noqa: E402

#: Ray lengths as a fraction of the maximum radius. Deliberately a literal:
#: the point of a pulsar map is that the lengths look chosen rather than
#: generated, and a fixed table is also what makes the output reproducible.
_LENGTHS = (
    1.00, 0.42, 0.78, 0.55, 0.93, 0.34, 0.69, 0.86,
    0.48, 0.97, 0.61, 0.39, 0.82, 0.52, 0.74, 0.45,
    0.90, 0.58, 0.36, 0.79, 0.66, 0.44, 0.88, 0.51,
    0.71, 0.40, 0.84, 0.63, 0.47, 0.76, 0.56, 0.95,
    0.38, 0.68, 0.49, 0.81,
)

#: Angular jitter, in fractions of the even spacing. Same reasoning: evenly
#: spaced rays of unequal length still read as a mechanical starburst; it is
#: the angular irregularity that makes it a map.
_SKEW = (
    0.00, 0.18, -0.12, 0.09, -0.20, 0.14, -0.06, 0.21,
    -0.15, 0.04, 0.19, -0.09, 0.12, -0.18, 0.07, 0.16,
    -0.11, 0.02, 0.20, -0.14, 0.10, -0.19, 0.05, 0.17,
    -0.08, 0.13, -0.16, 0.06, 0.22, -0.10, 0.15, -0.04,
    0.11, -0.21, 0.08, 0.18,
)


def _rays(c, n, *, rmax=0.40, rmin=0.0, w=W_FINE, bias=None, core=0.0):
    """Draw ``n`` rays from the centre, unequal in length and angle.

    :param bias: optional ``(angle, gain)`` making rays near ``angle`` longer,
        which is what turns an even field into an uneven one.
    :param core: radius of a solid dot at the origin, 0 for none.
    """
    cx = cy = 0.5
    for i in range(n):
        base = 2.0 * math.pi * i / n
        theta = base + _SKEW[i % len(_SKEW)] * (2.0 * math.pi / n)
        length = _LENGTHS[i % len(_LENGTHS)]
        if bias is not None:
            angle, gain = bias
            # cos of the angular distance: 1 on the bias axis, -1 opposite.
            length *= 1.0 + gain * math.cos(theta - angle)
            length = max(0.18, min(1.0, length))
        r0 = rmin
        r1 = rmax * length
        c.line(cx + r0 * math.cos(theta), cy + r0 * math.sin(theta),
               cx + r1 * math.cos(theta), cy + r1 * math.sin(theta), w=w)
    if core > 0:
        c.disc(cx, cy, core)


def illumination_pulsar_01(c):
    """24 thin rays of unequal length from one point -- the reference."""
    _rays(c, 24)
    return None


def illumination_pulsar_02(c):
    """36 rays: denser, closest to the Voyager plaque's own crowding."""
    _rays(c, 36, w=W_FINE * 0.85)
    return None


def illumination_pulsar_03(c):
    """16 rays: sparser, and the most legible of the set at 16 px."""
    _rays(c, 16, w=W_FINE * 1.25)
    return None


def illumination_pulsar_04(c):
    """24 rays inside the field circle they are correcting."""
    _rays(c, 24, rmax=0.34)
    c.circ(0.5, 0.5, 0.42, w=W_SEC)
    return None


def illumination_pulsar_05(c):
    """An uneven field: the long rays all fall to one side."""
    _rays(c, 24, bias=(-math.pi / 4.0, 0.45))
    return None


def illumination_pulsar_06(c):
    """28 rays leaving a solid core -- the strongest silhouette here."""
    _rays(c, 28, rmin=0.075, w=W_FINE, core=0.055)
    return None


GROUPS = {
    "illumination": (
        "illumination -- a pulsar map: thin rays of unequal length from "
        "one source, after the Voyager plaque",
        [
            ("24 thin rays of unequal length from one point.",
             illumination_pulsar_01),
            ("36 rays, denser, closest to the Voyager plaque's crowding.",
             illumination_pulsar_02),
            ("16 rays, sparser, the most legible at 16 px.",
             illumination_pulsar_03),
            ("24 rays inside the field circle they are correcting.",
             illumination_pulsar_04),
            ("An uneven field: the long rays all fall to one side.",
             illumination_pulsar_05),
            ("28 rays leaving a solid core.",
             illumination_pulsar_06),
        ],
    ),
}


if __name__ == "__main__":
    # Deliberately NOT emit_groups: that regenerates a folder from scratch and
    # would delete illumination's existing ten candidates. These are additions
    # numbered from 11, and the descriptions are appended to CONCEPTS.md.
    from _draw import render

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.abspath(os.path.join(here, "..", "illumination"))
    _title, entries = GROUPS["illumination"]
    lines = []
    for n, (desc, fn) in enumerate(entries, start=11):
        name = "illumination_%02d" % n
        render(fn, os.path.join(out, name + ".png"))
        lines.append("%d. **%s** - %s" % (n, name, desc))
        print("wrote", name + ".png")
    concepts = os.path.join(out, "CONCEPTS.md")
    with open(concepts, "a", encoding="utf-8") as fh:
        fh.write("\n\nPulsar-map set, added on request: thin rays of unequal\n"
                 "length from one source, after the Voyager plaque.\n\n")
        fh.write("\n".join(lines) + "\n")
    print("appended", len(lines), "entries to CONCEPTS.md")
