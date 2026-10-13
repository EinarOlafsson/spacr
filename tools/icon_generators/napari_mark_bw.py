"""napari's mark, monochrome, from the image the user supplied.

They uploaded napari's app icon -- a rounded square, purple gradient, cream
rim, and inside it the napari bean: one closed blob with a deep notch on its
upper right, an upper-left lobe and a lower-right lobe -- and asked for a
black and white version for the napari bridge.

**These are derived from napari's trademark, not spaCR's.** Shipping one
means shipping a monochrome derivative of a third party's mark. That is
ordinary nominative use for a button that opens napari, and it is what most
integrations do -- but it is a choice, so it is stated here and in
CONCEPTS.md rather than buried. `napari_bridge_06..10` in the same folder are
original marks about the handoff and carry no third-party mark at all.

Numbered from 11 so the ten candidates already in the folder survive.

Six variants, because the two things worth deciding are separable: whether
the bean is solid or outlined, and whether it keeps the app-icon frame. The
frame is what makes it read as "the napari application" rather than as a
generic blob -- which matters here, since a lone bean in spaCR's own flat
white style looks like one of spaCR's own cell icons.

The bean is one closed Catmull-Rom loop through fourteen points traced off
the supplied image. Fixed literals, no RNG, so it regenerates identically.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _draw import Cv, W_MAIN, W_SEC, W_FINE  # noqa: E402

#: The napari bean, traced off the supplied icon in normalised coordinates.
#: Anticlockwise from the top of the upper-left lobe. The notch is the run
#: from (0.50, 0.30) to (0.60, 0.44): that concavity is the whole character
#: of the mark, and a spline that rounds it off stops being napari's shape.
_BEAN = (
    (0.315, 0.115), (0.425, 0.150), (0.470, 0.245), (0.500, 0.300),
    (0.560, 0.395), (0.660, 0.400), (0.790, 0.430), (0.875, 0.545),
    (0.865, 0.685), (0.775, 0.805), (0.620, 0.878), (0.435, 0.855),
    (0.280, 0.735), (0.185, 0.545), (0.145, 0.330), (0.190, 0.180),
)


def _bean(c, w=W_MAIN, filled=False, scale=1.0):
    """The mark itself, optionally shrunk about the canvas centre."""
    pts = _BEAN
    if scale != 1.0:
        pts = tuple((0.5 + (x - 0.5) * scale, 0.5 + (y - 0.5) * scale)
                    for x, y in pts)
    c.smooth(pts, w=w, closed=True, filled=filled)


def _frame(c, w=W_SEC, r=0.20, inset=0.055):
    """The rounded square the app icon sits in."""
    c.rect(inset, inset, 1.0 - 2 * inset, 1.0 - 2 * inset, w=w, r=r)


def napari_bridge_11(c):
    """The napari bean, solid."""
    _bean(c, filled=True)


def napari_bridge_12(c):
    """The napari bean as an outline."""
    _bean(c, w=W_MAIN)


def napari_bridge_13(c):
    """The bean solid inside the app-icon frame."""
    _frame(c)
    _bean(c, filled=True, scale=0.86)


def napari_bridge_14(c):
    """The bean outlined inside the app-icon frame."""
    _frame(c)
    _bean(c, w=W_SEC, scale=0.86)


def napari_bridge_15(c):
    """Solid bean with the rim the original draws in cream, knocked out.

    The supplied icon has three layers -- dark outline, cream rim, blue body.
    Monochrome cannot carry three tones, so the rim becomes a gap: the body
    is solid and a concentric outline stands off it. That keeps the original's
    layered read without inventing a grey the house style does not use.
    """
    _bean(c, filled=True, scale=0.80)
    _bean(c, w=W_FINE, scale=0.94)


def napari_bridge_16(c):
    """The bean small in its frame -- most whitespace, best at 16 px."""
    _frame(c, w=W_FINE, inset=0.10, r=0.22)
    _bean(c, filled=True, scale=0.62)


ENTRIES = [
    ("[napari mark - their trademark] The napari bean, solid.",
     napari_bridge_11),
    ("[napari mark - their trademark] The napari bean as an outline.",
     napari_bridge_12),
    ("[napari mark - their trademark] The bean solid inside the app-icon "
     "frame.", napari_bridge_13),
    ("[napari mark - their trademark] The bean outlined inside the frame.",
     napari_bridge_14),
    ("[napari mark - their trademark] Solid bean with the cream rim of the "
     "original carried as a knocked-out gap.", napari_bridge_15),
    ("[napari mark - their trademark] The bean small in its frame; the most "
     "whitespace and the best of these at 16 px.", napari_bridge_16),
]


if __name__ == "__main__":
    # Not emit_groups: that regenerates a folder from scratch and would
    # delete the ten candidates already in it. These are additions from 11.
    from _draw import render

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.abspath(os.path.join(here, "..", "napari_bridge"))
    os.makedirs(out, exist_ok=True)
    lines = []
    for n, (desc, fn) in enumerate(ENTRIES, start=11):
        name = "napari_bridge_%02d" % n
        render(fn, os.path.join(out, name + ".png"))
        lines.append("%d. **%s** - %s" % (n, name, desc))
        print("wrote", name + ".png")
    with open(os.path.join(out, "CONCEPTS.md"), "a", encoding="utf-8") as fh:
        fh.write("\n\nDrawn from the napari app icon the user supplied. Every\n"
                 "one of these is a monochrome derivative of napari's mark,\n"
                 "which is their trademark and not spaCR's -- picking one\n"
                 "means shipping it as such. 06-10 above are original and\n"
                 "carry no third-party mark.\n\n")
        fh.write("\n".join(lines) + "\n")
    print("appended", len(lines), "entries to CONCEPTS.md")
