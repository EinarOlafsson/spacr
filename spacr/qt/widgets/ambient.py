"""Ambient animated backdrop — soft motion behind every module screen.

The sequencing screen has its own backdrop (:mod:`spacr.qt.widgets.dna_rain`,
the ATGC cascade). This is the one for *everything else*: a slow, diffuse
animation that sits behind the settings form and the console, takes no focus
and no mouse events, and can be switched off entirely in Preferences.

Six themes, chosen so each reads as a different kind of movement rather than
as a re-skin of the same one:

``blobs``   (default)
    Big and small colour blobs drifting over the page, each pulsing in size on
    its own period. They overlap and blend, so the result reads as soft colour
    *fields* rather than as a bag of circles. This is the one the feature was
    asked for.
``aurora``
    Three overlapping curtains of vertical rays, folding along their own
    length. The folds are travelling waves — several superposed frequencies
    running lengthwise along the arc — with brightness surges on a separate
    schedule, a sharp lower edge, a diffuse top, and the real thing's
    vertical colour order: green through the body, red high up, a violet
    fringe underneath.
``ripple``
    Concentric rings expanding out of three fixed sources and fading as they
    grow, like rain on water. Soft-edged, so it never reads as line work.
``drift``
    A slow starfield in three parallax layers: small, dim, slow ones behind;
    bigger, brighter, faster ones in front. The one crisp theme. It travels
    up, down, or every which way — see :data:`DRIFT_DIRECTIONS`.
``bokeh``
    Out-of-focus points of light, the way a fluorescence field looks off the
    focal plane: an aperture image is a *disc with a bright rim*, not a
    Gaussian smudge, and the ones further out of focus are larger and flatter.
``cells``
    Cells drifting through the field, turning as they go — a soft body, a
    slightly brighter membrane where the edge is seen nearly edge-on, and a
    distinctly brighter nucleus set off centre.

Two more were built and thrown away, on measurements rather than taste, and
the numbers are recorded here so nobody spends the afternoon again:

*a constellation of points with faint links between them*, and *a slowly
branching mycelium*. Both are the same shape of thing — a few dozen short,
thin, translucent, antialiased lines — and at 1920x1080 sixty points with
196 links between them cost **8.2 ms** a frame drawn one at a time and
**5.7 ms** batched into ``drawLines`` calls by alpha. That is four times what
this entire module is allowed, for one theme, and it is the module's own rule
about line work arriving again: a translucent antialiased line is a thousand
one-pixel spans, and Qt's raster engine charges for every one. Painting them
into the small buffer instead makes them affordable and also makes them not
lines any more, which is the whole thing that would have made either theme
worth having.

*Caustics* and *a nebula / flow field* were not built. Both are per-pixel
noise fields, so they cannot be drawn with Qt primitives at all and have to
be evaluated in NumPy over the buffer — and the separable box blur measured
for the blur control, which is the same shape of work over the same array,
came to 12.8 ms on a 640x360 buffer. Evaluating them small enough to afford
would put them back at the resolution this change exists to get away from.

*Brownian diffusion of specks* was asked for and is here, but as ``drift``'s
``random`` direction rather than as a theme. It would have been the starfield
engine with one constant changed, and a theme menu that lists the same
animation twice is a worse answer than a setting that says what differs.

Palettes
--------
Every theme declares the palettes it offers (:func:`palettes_for`), because a
palette that works as a 400 px blob does not necessarily work as a 2 px star.
:data:`PALETTE_SETS` holds the colours themselves; ``spacr`` is built from the
three brand hues the user picked for the module-maturity legend, and ``okabe``
is the Okabe–Ito set for red–green colour deficiency (see its note).

Both dark and light
-------------------
spaCR ships five themes, and a blob set tuned only for a near-black page turns
to mud on a white one. So the *composition mode follows the background*:

* dark page  -> ``CompositionMode_Plus``. Overlapping blobs add up and glow,
  which is what makes two circles read as one colour field. Additive over a
  light page just clips to white and the whole effect vanishes.
* light page -> ``CompositionMode_Multiply``, with the palette colour mixed
  toward white first. Multiply is the exact dual: overlaps get *darker* and
  still blend hue-wise, so the same geometry reads the same way. Plain
  ``SourceOver`` was tried first and is wrong here — the later blob simply
  covers the earlier one, so nothing blends and the field falls apart into
  discrete discs.

:func:`AmbientWidget.set_background_color` re-derives all of that, so a live
theme switch is one call.

Cost
----
This paints behind every module screen, on machines that are simultaneously
running Cellpose on a GPU and a 40-plate pipeline, so cost is a correctness
requirement rather than a nicety. Two things get it there:

1. *The timer stops whenever the widget is not on screen* — hidden, on another
   tab, or in a minimised window. Zero frames, zero CPU. These screens stay
   open for hours, so this is the whole ball game.
2. *The soft themes are painted into a small reusable QImage and scaled up*,
   never at full resolution. The buffer's long edge is whatever the theme
   declares (:attr:`_BufferedEngine.base_edge`) times the user's resolution
   setting, so the diffuse fields shade ~37 000 pixels instead of ~2 000 000
   and the aurora, which has real structure in it, shades ~520 000. On the
   synchronous path the one allocation happens on resize and never per
   frame; the shading thread copies the finished buffer once a frame so the
   GUI thread can blit it while the next one is being drawn, which measures
   0.003 ms for ``blobs`` and 0.035 for the aurora's 2 MiB — 2 % of the
   shading pass it makes safe.
3. *The shading happens on its own thread* (:class:`_FrameProducer`), so the
   GUI thread's whole share of a frame is one ``drawImage``. That is the next
   section, and it is the one that matters while a pipeline is running.

While a run is going
--------------------
"When something is running after hitting run, the theme starts lagging"
(instruction 126). It does, it reproduces, and the cause is not the obvious
one. Measured on a real X server at 1920x1080 with a real ``ConsolePanel``
under the real stylesheet and a real Qt event loop, ``blobs`` at the shipped
24 fps cap, best of five interleaved rounds:

==============================================  ==========  ==========
 condition                                       delivered   GUI paint
==============================================  ==========  ==========
 idle                                             25.0 fps     2.04 ms
 a numpy thread (1024² matmul + FFT), flat out    24.5 fps     2.10 ms
 200 console lines a second, nothing else         24.9 fps     1.27 ms
 **one pure-Python thread**                       24.5 fps  **17.21 ms**
 a worker doing Python work *and* printing        17.3 fps    17.16 ms
==============================================  ==========  ==========

So **CPU saturation is not a cause**: numpy releases the interpreter lock and
a core burning flat out costs this module nothing. **A signal flood is not a
cause on its own**: 200 lines a second are free, and it only bites in the
thousands, where the console's own per-line work saturates the GUI thread and
nothing in *this* module can help. What is left is **the interpreter lock**:
identical drawing work, eleven times slower, because the shading pass is
Python and numpy and something else is holding the lock.

There is a fourth thing, which nobody named and which doubles the bill: **the
frame-rate cap is not a cap.** The console sits on a translucent surface over
this widget, so every line it prints exposes it and Qt asks for a whole frame
— 0.99 ambient repaints per console line, measured with the animation timer
*stopped*. Every one of those used to be a full shading pass.

The fix is to split a frame at the seam where the cost actually is. Per
theme, milliseconds, idle against one Python thread, min of nine interleaved
rounds:

=========  =====================  =====================
 theme      shading (moved)        soften + blit (stays)
=========  =====================  =====================
 blobs      0.240 ->  0.572        0.651 -> 1.097
 aurora     1.396 ->  7.176        0.797 -> 1.043
 ripple     0.367 ->  0.589        0.644 -> 1.196
 bokeh      0.663 ->  3.533        0.679 -> 1.008
 cells      0.538 -> 26.179        0.663 -> 0.909
 drift      0.528 ->  1.084        (no buffer)
=========  =====================  =====================

The half that explodes under contention is exactly the half that does not
have to be on the GUI thread, and the half that must stay there is bounded at
1.2 ms even loaded, because it is Qt's C++ raster engine with the lock
released. So :meth:`_BufferedEngine.shade` moves to :class:`_FrameProducer`
and :meth:`_BufferedEngine.blit` stays. Before and after, in one process,
HEAD's module exec'd from git beside the working tree's, interleaved:

=============================  ====================  ====================
 condition (``cells``, 1080p)   before                after
=============================  ====================  ====================
 idle                           24.9 fps / 2.53 ms    25.0 fps / 1.52 ms
 one Python thread              19.9 fps / 33.23 ms   24.9 fps / 1.76 ms
 worker + 20 lines a second     15.0 fps / 18.80 ms   25.7 fps / 1.49 ms
=============================  ====================  ====================

``blobs``, the default, goes 17.3 fps to 24.7 on the same worker. What this
does **not** fix is a genuinely chatty run: at 200 lines a second both land at
about 4 fps, because by then the GUI thread is inside ``ConsolePanel`` and not
in here at all. Two levers finish the job and neither is in this file —
``sys.setswitchinterval(0.001)`` in the Qt bootstrap (measured on its own:
32 % of the frame rate to 99 %, for about 6 % of the worker's throughput) and
coalescing ``PipelineWorker.line_ready``. Both want their own instruction.

Three things this design deliberately is not:

*Not a precomputed loop.* The instruction asked for one, and the reason it
cannot exist is not memory: **there is no period.** Every element's period is
drawn from ``rng.uniform`` over a continuous range —
:data:`BLOB_DRIFT_PERIOD` (24-70 s), :data:`BLOB_PULSE_PERIOD` (7-19 s),
:data:`AURORA_DRIFT_PERIOD` (30-90 s), :data:`RIPPLE_PERIOD` (14-26 s) and
the rest — so the composite period is irrational with probability 1 and a
fixed-length loop is a visible *cut*, not a period: the picture jumps every
time it wraps. Memory says the same thing more loudly. At 1080p and 24 fps a
second of loop is 3.0 MiB for ``blobs``, 11.9 for ``bokeh`` and ``cells``,
47.5 for the aurora and 190 for ``drift``, which has no buffer at all — a
one-minute loop is 178 MiB of blobs or 2.85 GiB of aurora.

*Not a faster shading pass.* The thread does not make anything quicker: same
Python, same lock, same 26 ms for ``cells`` under load. It changes **who
waits**. The GUI thread stops shading, and a frame that is not ready becomes
a repeated frame rather than a blocked interface —
:attr:`AmbientWidget.repeated_frames` is that promise as a number, and it
rises to about a third of frames under a Python worker.

*Not a second clock.* The animation clock stays on the GUI thread (two
attribute stores; there is nothing to gain by moving it and a public contract
to lose) and the thread only reads it. A frame is still a pure function of
``(seed, clock, size)``, which is why the off-thread shade is byte-identical
to the on-thread one for all five buffered themes —
``tests/qt/test_the_backdrop_survives_a_run.py`` asserts it rather than
claiming it.

``drift`` keeps the synchronous path, and its row above is the reason: it is
the one engine with no buffer, it degrades the least of the six (2.1x against
48.7x for ``cells``), and threading it would mean publishing a full-resolution
frame — 7.91 MiB a slot against 126.6 KiB for ``blobs`` — to buy the smallest
improvement on the list.

At 1920x1080, offscreen raster. The engine as it stood before this change
is loaded into the same process out of git and every configuration is timed
round-robin, best of thirteen — this machine is shared with four other test
suites, so a run now and a run in ten minutes are not comparable and only the
minimum of an interleaved set means anything.

That also means the *absolute* frame figures in the previous edition of this
table could not be reproduced honestly today: the same unchanged ``blobs``
frame measured anywhere between 1.4 and 3.0 ms depending on what else the
box was doing. What is stable under load is the **shading pass** — the part
resolution and density actually move — because it is a few hundred thousand
pixels rather than two million, so it is what is tabulated:

=========  ==================  ==================  ============
 theme      shading was         shading now         full frame
            (dark / light)      (dark / light)      (dark)
=========  ==================  ==================  ============
 blobs      0.283 / 0.331 ms    0.283 / 0.324 ms    1.39 ms
 aurora     0.487 / 0.546 ms    1.782 / 2.270 ms    2.89 ms
 ripple     0.398 / 0.437 ms    0.394 / 0.442 ms    1.51 ms
 bokeh      —                   0.795 / 0.738 ms    1.91 ms
 cells      —                   0.748 / 0.761 ms    1.86 ms
 drift      0.747 / 0.727 ms    0.714 / 0.706 ms    0.71 ms
=========  ==================  ==================  ============

``drift`` has no buffer, so for it the first three columns are all the whole
frame and the shading columns are the same measurement twice.

The last column is the middle one plus the *fixed* part of a buffered frame
— filling the page and blitting the buffer up to 1920x1080 — which measured
**1.111 ms** and is bound by the destination pixels, so it is the same
whatever was drawn into the source. (The page fill alone is 0.101 ms of it;
the rest is the upscale.) Light adds the multiply penalty to that blit as it
always did. Off screen the whole thing is 0 %, which remains the number that
matters most.

**The aurora is the one theme that moved, and it is the one shipped default
this change spends anything on.** Its shading went from 0.487 ms to 1.782 —
3.7 times — because its buffer went from 240x135 to 960x540 at 1080p. In
frame terms that is 1.60 ms to 2.89, and it buys the thing the change is
for: the ray comb, 36 screen pixels per ray, was being resolved at four and
a half buffer pixels and had lost 23 % of its contrast (see
:data:`AURORA_BUFFER_EDGE` for the table). Nothing else moved — blobs,
ripples and the starfield render byte-for-byte what they did before, and the
tests assert exactly that against the engine pulled out of git history.

The user controls move the shading cost, deliberately, and reading this
table is how to know which of them is worth turning down first (dark, best
of nine interleaved, milliseconds of shading):

===========================  ======  ======  ======  ======  ======
 setting                      blobs   auror   rippl   bokeh   cells
===========================  ======  ======  ======  ======  ======
 detail 50 %                  0.191   0.829   0.194   0.412   0.409
 default                      0.282   1.656   0.354   0.685   0.605
 detail 200 %                 0.542   3.732   0.652   1.349   1.000
 blur 100 %                   0.267   1.753   0.350   0.775   0.810
 blur 300 %                   0.295   1.770   0.382   0.715   0.707
 blur 300 % + detail 200 %    0.579   3.979   0.720   1.501   1.195
 density 300 %                0.852   4.506   0.945   1.936   1.762
===========================  ======  ======  ======  ======  ======

Four things in that table are worth reading rather than skipping:

* **Blur is nearly free.** Going from 0 % to 300 % costs 0.01-0.11 ms, which
  on three of the five themes is inside the run-to-run spread. It is one
  area-averaging pass over a buffer of half a megapixel at most, and the
  blit that carries the result to the screen was going to happen anyway.
  The honest alternative — a separable box blur over the buffer in NumPy,
  two cumulative sums per axis — was written and measured at **12.8 ms** on
  a 640x360 buffer, nine times this module's whole budget, which is why it
  is not what ships. ``drift`` is the exception it always was: it has no
  buffer, so its blur is a second wider pass per dot and it has a cap
  (:data:`DRIFT_HALO_MAX_PX`).
* **Detail costs roughly the square of what it says**, and that is the whole
  reason it is a separate control from blur. Somebody who wants a softer
  backdrop can now have one for nothing instead of paying for it in blocks
  — and somebody who cannot afford the backdrop at all has a control that
  actually reduces it, which "blur" used to be by accident.
* **Density is linear**, as it should be: 300 % is 2.7-3.0 times the shading
  on every theme. What it is *not* is brighter — see
  :meth:`AmbientEngine.alpha_scale` for why three times the elements at a
  third of the alpha is the only reading of the control that leaves the
  backdrop legible at both ends of its range.
* **The budget works.** Density 300 % with detail 200 % is twelve times the
  default work; :data:`WORK_BUDGET` trims the density until it is four, and
  the aurora lands at about 4 ms of shading rather than the 13 it would
  otherwise ask for.

The design was picked on measurements, not taste. For blobs at 1920x1080:
full-resolution gradients 2.18 ms, buffered-and-upscaled 1.16 ms,
pre-rendered sprite blits 3.28 ms — the sprite version is the slow one
because a bilinear-sampled translucent blit costs more per pixel than
shading the gradient does.

Four candidate themes were cut on the same kind of numbers rather than
shipped slow — a mesh lattice (3.2–8.1 ms), contour polylines (10.8 ms), a
linked constellation and a branching mycelium (5.7–8.2 ms; see the theme
list above). All four are crisp line work, and in Qt's raster engine a
translucent antialiased line costs an order of magnitude more than the same
pixels as a gradient — a vertical line is a thousand one-pixel spans.
Painting them into the small buffer instead makes them cheap and also makes
them not lines any more. The rule that fell out: *soft is cheap, crisp is
expensive*, and ``drift`` is crisp only because a couple of hundred dots
light 0.65 % of the page.
"""
from __future__ import annotations

import math
import random
import threading
import time
from dataclasses import dataclass
from typing import (Callable, Dict, List, NamedTuple, Optional, Sequence,
                    Tuple, Union)

from PySide6.QtCore import (QElapsedTimer, QEvent, QPoint, QPointF, QRect,
                            QRectF, Qt, QTimer)
from PySide6.QtGui import (QBrush, QColor, QImage, QLinearGradient, QPainter,
                           QPainterPath, QPen, QPixmap, QRadialGradient,
                           QTransform)
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..theme import page_colour, palette_for, relative_luminance

__all__ = [
    "AMBIENT_THEMES", "ANIMATION_CHOICES", "NO_ANIMATION",
    "animation_label", "animation_note", "is_animation_choice",
    "total_frames_painted",
    "DEFAULT_THEME", "DEFAULT_PALETTE", "PALETTE_SETS",
    "AmbientWidget", "install_ambient", "theme_label", "theme_note",
    "palettes_for", "palette_label", "palette_note", "palette_colors",
    "default_palette_for", "is_valid_theme", "is_valid_palette",
    "BLUR_RANGE", "SPEED_RANGE", "SIZE_RANGE", "RESOLUTION_RANGE",
    "DENSITY_RANGE", "DEFAULT_BLUR", "DEFAULT_SPEED", "DEFAULT_SIZE",
    "DEFAULT_RESOLUTION", "DEFAULT_DENSITY", "DRIFT_DIRECTIONS",
    "DEFAULT_DRIFT_DIRECTION", "drift_direction_label",
    "drift_direction_note", "is_valid_drift_direction", "Motion",
    "preferred_motion",
]


# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------

#: Every theme, in the order a menu should list them.
#:
#: These are the *paintable* ones — every name here has an engine behind it.
#: The menu the user sees is :data:`ANIMATION_CHOICES`, which is this list
#: with "no animation at all" in front of it; keeping the two apart is what
#: lets ``make_engine``, ``_require_theme`` and every engine test go on
#: meaning "a thing that can be drawn".
AMBIENT_THEMES: Tuple[str, ...] = ("blobs", "aurora", "ripple", "drift",
                                   "bokeh", "cells")

#: The animation choice that draws nothing and runs no timer.
#:
#: Not a seventh engine that happens to paint an empty frame — that would
#: still be a timer, a repaint and a composite sixty times a second for a
#: picture that is identical every time. It is the absence of the widget:
#: :func:`spacr.qt.preferences.get_ambient_enabled` reports ``False`` while
#: it is selected, and the three install sites (``AppScreen``, Home and
#: ``MainWindow._theme_screen``) all read that *before* they construct
#: anything. The cost is zero because nothing exists, which is the only kind
#: of zero worth claiming.
NO_ANIMATION = "none"

#: What the Animation preference offers, in menu order: nothing, then the
#: six animations.
ANIMATION_CHOICES: Tuple[str, ...] = (NO_ANIMATION,) + AMBIENT_THEMES

#: What the feature was asked for, so it is what you get by default.
DEFAULT_THEME = "blobs"

#: spaCR's own colours, likewise.
DEFAULT_PALETTE = "spacr"

_THEME_LABELS = {
    "blobs": "Blobs",
    "aurora": "Aurora",
    "ripple": "Ripples",
    "drift": "Starfield",
    "bokeh": "Bokeh",
    "cells": "Cells",
}

_THEME_NOTES = {
    "blobs": ("Soft colour blobs, large and small, drifting and slowly "
              "changing size."),
    "aurora": ("Folded curtains of vertical rays, rippling along their own "
               "length the way the northern lights do."),
    "ripple": "Rings spreading out from a few points and fading as they grow.",
    "drift": "A slow starfield in three layers of depth.",
    "bokeh": ("Out-of-focus points of light, the way a fluorescence field "
              "looks off the focal plane: bright rims, flat centres."),
    "cells": ("Cells drifting through the field — soft bodies with a "
              "brighter nucleus, turning slowly as they go."),
}


# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------

class PaletteSpec(NamedTuple):
    """One named colour set: what to call it and what it is made of."""

    label: str
    colors: Tuple[str, ...]
    note: str


#: The colour sets, shared across themes. A palette is a *set of hues*; how
#: strongly they are applied is the theme's business (a 2 px star needs a very
#: different alpha from a 400 px blob), which is why the alphas live in the
#: engines and not here.
PALETTE_SETS: Dict[str, PaletteSpec] = {
    "spacr": PaletteSpec(
        "spaCR",
        ("#3B82F6", "#FF00FF", "#00CEC8"),
        "spaCR's own three colours — the blue, magenta and green-cyan that "
        "mark a module stable, beta or alpha."),
    "ember": PaletteSpec(
        "Ember",
        ("#FF6B35", "#FFB020", "#E2374A", "#FF8FA3"),
        "Warm — orange, amber and rose."),
    "ocean": PaletteSpec(
        "Ocean",
        ("#0EA5E9", "#22D3EE", "#2DD4BF", "#3B82F6"),
        "Cool — teal, aqua and deep blue."),
    "pastel": PaletteSpec(
        "Pastel",
        ("#A8D8EA", "#FFB5E8", "#B5EAD7", "#FFDAC1"),
        "Pale and low contrast, for when the backdrop should be barely "
        "there."),
    "mono": PaletteSpec(
        "Monochrome",
        # Neutral greys, not slate: a set named Monochrome that carries a
        # blue cast is a set that lies about what it is.
        ("#A3A3A3", "#CFCFCF", "#707070"),
        "Greys only — motion without colour, for when colour is a "
        "distraction."),
    "okabe": PaletteSpec(
        "Colour-blind safe",
        ("#0072B2", "#E69F00", "#009E73", "#56B4E9", "#D55E00", "#F0E442"),
        "The Okabe–Ito set. Its colours stay distinguishable under "
        "protanopia and deuteranopia — red–green deficiency, the common "
        "kind — because no pair in it differs by red versus green alone."),
    # Not invented: these are the emission lines the sky actually radiates,
    # converted to sRGB. Ordered by the role the aurora engine gives them —
    # main, high, fringe, blend — see AURORA_RAMP.
    "borealis": PaletteSpec(
        "Aurora borealis",
        ("#7CFC9E", "#FF3C5A", "#5B6BFF", "#D9FFA8"),
        "The real thing's emission lines: atomic oxygen at 557.7 nm (the "
        "dominant green), atomic oxygen at 630.0 nm (the red that only "
        "appears high up), ionised nitrogen at 427.8 nm (the blue-violet "
        "lower fringe), and the pale yellow-green where the green and the "
        "red overlap."),
    # Also not invented: the three filter cubes on essentially every
    # fluorescence scope, at the wavelength the eyepiece sees.
    "fluor": PaletteSpec(
        "Fluorescence",
        ("#3AA0FF", "#3DFF6E", "#FF5A3C", "#FFD24A"),
        "The standard filter set as the eyepiece sees it: DAPI at 461 nm "
        "(blue), FITC at 519 nm (green), TRITC at 576 nm (orange-red), and "
        "the yellow where green and red overlap."),
}

#: Which palettes each theme offers, and why the excluded ones are excluded.
#:
#: ``ripple`` and ``drift`` drop ``pastel``: a starfield is 1-4 px dots and a
#: ripple is a halo at a fifth of the alpha a blob gets, and a pale
#: low-contrast hue at either scale is indistinguishable from the page.
#: Offering it would be offering a setting that does nothing.
#:
#: ``borealis`` is offered wherever the animation reads as *sky* — the
#: curtains it was built for, the diffuse fields of ``blobs`` (a quiet
#: aurora is exactly that: 557.7 nm green low down with 630.0 nm red above
#: it), and the ``drift`` starfield. It is withheld from ``ripple`` alone,
#: whose motion is rain on water: a set named after the northern lights on
#: a pond would be decoration, not a colour choice.
#:
#: ``fluor`` is the mirror of that rule: it is offered where the animation
#: reads as something seen down a microscope — ``bokeh`` and ``cells``, which
#: were built for it, and ``blobs``, whose merged fields are what a
#: badly-focused multichannel overlay looks like. It is withheld from the
#: aurora and the ripples for the same reason ``borealis`` is withheld from
#: the ripples.
_THEME_PALETTES: Dict[str, Tuple[str, ...]] = {
    "blobs": ("spacr", "ember", "ocean", "pastel", "mono", "okabe",
              "borealis", "fluor"),
    "aurora": ("spacr", "ember", "ocean", "pastel", "mono", "okabe",
               "borealis"),
    "ripple": ("spacr", "ember", "ocean", "mono", "okabe"),
    "drift": ("spacr", "ember", "ocean", "mono", "okabe", "borealis",
              "fluor"),
    "bokeh": ("spacr", "ember", "ocean", "pastel", "mono", "okabe", "fluor"),
    "cells": ("spacr", "ember", "ocean", "pastel", "mono", "okabe", "fluor"),
}


def is_valid_theme(name) -> bool:
    """True when ``name`` is one of :data:`AMBIENT_THEMES`.

    The predicate exists so a caller validating stored preferences does not
    have to catch :class:`ValueError` from the strict accessors below.
    """
    return name in AMBIENT_THEMES


def is_valid_palette(theme, palette) -> bool:
    """True when ``palette`` is offered by ``theme``. Never raises."""
    return palette in _THEME_PALETTES.get(theme, ())


def _require_theme(name: str) -> str:
    if name not in AMBIENT_THEMES:
        raise ValueError(
            f"unknown ambient theme {name!r}; expected one of "
            f"{', '.join(AMBIENT_THEMES)}")
    return name


def _require_palette(theme: str, name: str) -> str:
    """Validate ``name`` *for this theme*, loudly.

    Two different failures, and they are worth telling apart in the message:
    a palette nobody has ever heard of, and a real palette this theme does
    not offer.
    """
    _require_theme(theme)
    offered = _THEME_PALETTES[theme]
    if name in offered:
        return name
    if name in PALETTE_SETS:
        raise ValueError(
            f"the {theme!r} ambient theme does not offer the {name!r} "
            f"palette; expected one of {', '.join(offered)}")
    raise ValueError(
        f"unknown ambient palette {name!r}; the {theme!r} theme expects one "
        f"of {', '.join(offered)}")


def theme_label(name: str) -> str:
    """Human label for ``name``, for a menu. Raises on an unknown theme."""
    return _THEME_LABELS[_require_theme(name)]


def theme_note(name: str) -> str:
    """One-line description of ``name``, for a tooltip."""
    return _THEME_NOTES[_require_theme(name)]


def is_animation_choice(name) -> bool:
    """True for anything the Animation preference may hold — including
    :data:`NO_ANIMATION`, which :func:`is_valid_theme` rejects because it
    cannot be painted."""
    return name in ANIMATION_CHOICES


def animation_label(name: str) -> str:
    """Human label for an entry of :data:`ANIMATION_CHOICES`.

    "None" rather than "Off": the row is called Animation and this is one of
    the animations it can be set to, the way a font size can be set to zero.
    """
    if name == NO_ANIMATION:
        return "None"
    return theme_label(name)


def animation_note(name: str) -> str:
    """One-line description of an animation choice, for a tooltip.

    The note for "None" states the cost, because that is the only reason a
    reader picks it — and the claim is asserted rather than advertised: see
    ``tests/qt/test_ambient_none.py``, which counts painted frames over a
    real second instead of trusting this sentence.
    """
    if name == NO_ANIMATION:
        return ("No backdrop at all: nothing is drawn behind the module "
                "pages and no animation timer runs anywhere in spaCR, so "
                "the cost while idle is exactly zero rather than nearly "
                "nothing. The page keeps its ordinary theme colour.")
    return theme_note(name)


def palettes_for(theme: str) -> Tuple[str, ...]:
    """The palette names ``theme`` offers, in menu order.

    Never empty. Raises :class:`ValueError` on an unknown theme rather than
    returning ``()``, because an empty tuple reads as "this theme has no
    palettes" and would quietly leave a settings menu blank.
    """
    return _THEME_PALETTES[_require_theme(theme)]


def default_palette_for(theme: str) -> str:
    """The palette ``theme`` falls back to — :data:`DEFAULT_PALETTE` when it
    is on offer, otherwise the first one listed."""
    offered = palettes_for(theme)
    return DEFAULT_PALETTE if DEFAULT_PALETTE in offered else offered[0]


def palette_label(theme: str, palette: str) -> str:
    """Human label for ``palette`` as offered by ``theme``."""
    return PALETTE_SETS[_require_palette(theme, palette)].label


def palette_note(theme: str, palette: str) -> str:
    """One-line description of ``palette``, for a tooltip."""
    return PALETTE_SETS[_require_palette(theme, palette)].note


def palette_colors(theme: str, palette: str) -> Tuple[str, ...]:
    """The ``#rrggbb`` colours behind ``palette``, for ``theme``."""
    return PALETTE_SETS[_require_palette(theme, palette)].colors


def coerce_palette(theme: str, palette: str) -> str:
    """``palette`` if ``theme`` offers it, else that theme's default.

    Only for *stored* values — a preferences file that still names the
    palette the user picked under a different theme should not stop a screen
    from being built. An unknown name is still an error: it is a bug, not a
    stale setting.
    """
    _require_theme(theme)
    if palette not in PALETTE_SETS:
        raise ValueError(
            f"unknown ambient palette {palette!r}; expected one of "
            f"{', '.join(sorted(PALETTE_SETS))}")
    return palette if is_valid_palette(theme, palette) \
        else default_palette_for(theme)


# ---------------------------------------------------------------------------
# Timing and sizing constants
# ---------------------------------------------------------------------------

#: Frame-rate cap. Nothing here moves fast enough to need more, and this
#: matches the DNA rain so the app has one animation cadence.
DEFAULT_FPS = 24
MIN_FPS = 1
MAX_FPS = 60

#: Largest simulation step accepted from the wall clock, in seconds. If the
#: app was busy for two seconds the animation resumes where it was instead of
#: teleporting.
MAX_DT = 0.25

#: Longest edge, at resolution 1.0, of the buffer a soft theme shades into.
#: This is the *diffuse* themes' figure. 256 px upscales to 1920 with nothing
#: measurably lost — a frame of ``blobs`` shaded here differs from the same
#: frame shaded at 1920x1080 by at most 2.9 luminance levels out of 255, and
#: ``ripple`` by 3.2 — because a diffuse gradient has no detail to lose.
#: Themes with an edge or a fine repeat in them declare their own, larger,
#: figure; see :attr:`_BufferedEngine.base_edge` and
#: :data:`AURORA_BUFFER_EDGE`.
BUFFER_MAX_EDGE = 256

#: Hard limits on the derived buffer edge. The low end is where bilinear
#: upscaling stops looking like softness and starts looking like blocks; the
#: high end is where a 4K canvas would be shaded at full resolution.
BUFFER_MIN_EDGE = 96
BUFFER_EDGE_CEILING = 2048

#: A second, absolute cost ceiling on the buffer, in pixels. The edge is a
#: *ratio* to the canvas, and a ratio alone lets a 5K display quietly ask for
#: a five-megapixel shading pass. Shading is per buffer pixel, so this is the
#: number that actually bounds the frame.
#:
#: 1920x1080, because that is where the highest setting on the highest-detail
#: theme lands on the display this was measured on — full-resolution aurora,
#: 4.76 ms — and it is the user's choice to ask for it. It is a ceiling on
#: what a *bigger screen* can silently cost, not a veto on the top of the
#: slider: on a 4K panel the same setting shades 1920x1080 and upscales two-
#: fold rather than shading 8.3 megapixels.
BUFFER_MAX_PIXELS = 1920 * 1080


# ---------------------------------------------------------------------------
# The user controls: resolution, blur, speed, size and density
# ---------------------------------------------------------------------------
# All of them are *multipliers on what the theme already does*, never absolute
# pixels or seconds. 1.0 is the shipped animation, exactly — every engine is
# written so that multiplying by 1.0 is the identity, and the tests assert
# the default frame is byte-for-byte the frame from before these existed.
# A multiplier is also the only formulation that means the same thing in every
# theme: "twice as big" is meaningful for a blob radius, a curtain, a ripple
# wavelength and a 2 px star, where "40 px" is meaningful for none of them.
#
# Resolution and blur used to be ONE control, and that was the bug this pair
# replaces. Blur was implemented *as* the buffer resolution — softer meant
# shading fewer pixels and stretching them further — so "sharper" and "less
# blocky" were the same slider, and a sharp *soft* backdrop could not be
# asked for at all. They are two different questions and they now have two
# different answers:
#
#   resolution  how many pixels the scene is shaded into. Decides how much
#               of the geometry survives: where the aurora's lower edge
#               falls between two pixels, how wide each ray of its comb is.
#   blur        how much of what was shaded is then thrown away, by an
#               area average over the finished buffer. Decides how soft it
#               looks, and nothing else.
#
# The order matters and is the whole point. Shading at a low resolution
# *point-samples* the geometry: the fold's position is quantised, the ray
# comb aliases against the buffer grid, and no amount of subsequent blurring
# puts back what was never computed. Shading high and averaging down
# *prefilters* it: the same softness, with every edge still where the model
# put it. That is the difference between "soft" and "blocky", and it is why
# a picture can now be both sharp and soft.

#: How much detail is computed, as a multiplier on the theme's own buffer
#: edge. Above 1.0 costs quadratically more (shading is per buffer pixel);
#: below 1.0 is the escape hatch for a machine that cannot afford the
#: backdrop at all.
RESOLUTION_RANGE = (0.25, 2.0)
DEFAULT_RESOLUTION = 1.0

#: How soft the result is, in units of :data:`BLUR_UNIT_PX` screen pixels of
#: area averaging. 0.0 — the default — is no softening pass at all, which is
#: also why the default frame is still byte-for-byte the shipped one.
#:
#: This is *not* the old blur. The old one ran from 0.25 (sharp) through 1.0
#: (as shipped) to 3.0 (soft) and sharpened by enlarging the buffer; that job
#: now belongs to :data:`RESOLUTION_RANGE`, and this control only ever
#: softens. The rename of the meaning is deliberate and is called out in the
#: preferences module, which migrates a stored value from the old scale.
BLUR_RANGE = (0.0, 3.0)
DEFAULT_BLUR = 0.0

#: What one unit of blur is worth, in screen pixels. 8 is not arbitrary: it
#: is exactly the smoothing the diffuse themes shipped with, when a 240x135
#: buffer was stretched over 1920x1080. So blur 1.0 asks for "the softness
#: the backdrop always had" and gets it at whatever resolution is set —
#: which is the sharp-and-soft frame that could not be asked for before.
#:
#: Expressed in *screen* pixels rather than buffer pixels on purpose: in
#: buffer pixels the two controls would still be coupled, and raising the
#: resolution would silently sharpen the picture again.
BLUR_UNIT_PX = 8.0

#: A multiplier on the animation clock, so every per-theme period, drift rate
#: and travel speed scales together and nothing has to be re-tuned. Applied
#: in :meth:`AmbientEngine.advance`, never in :meth:`geometry` — changing the
#: speed must not teleport an animation that is already on screen.
SPEED_RANGE = (0.1, 4.0)
DEFAULT_SPEED = 1.0

#: A multiplier on each theme's own size range: blob radius, aurora curtain
#: height and ray spacing, ripple wavelength, starfield dot diameter.
SIZE_RANGE = (0.25, 2.5)
DEFAULT_SIZE = 1.0

#: A multiplier on how many *elements* a theme draws: blobs, curtains, ripple
#: sources, stars, bokeh discs, cells. Every engine rolls a pool big enough
#: for the top of this range once, at construction, and then paints a prefix
#: of it — so turning the slider never re-rolls the field and never makes the
#: animation jump.
DENSITY_RANGE = (0.25, 3.0)
DEFAULT_DENSITY = 1.0

#: The shared account resolution and density both draw on, as a multiple of
#: what the theme costs at its own defaults.
#:
#: Shading cost is (buffer pixels) x (elements), and both halves are now on a
#: slider: 2.0 resolution is four times the pixels and 3.0 density is three
#: times the elements, so the two together could ask for twelve times the
#: work — 20-odd milliseconds a frame behind every screen in the app. Past
#: this budget the *density* is scaled back rather than the resolution,
#: because a backdrop that has lost a blob still looks right and a backdrop
#: made of visible blocks does not. Four is chosen so that either control on
#: its own reaches the top of its range untouched, and only the combination
#: is trimmed. See :meth:`AmbientEngine.effective_density`.
WORK_BUDGET = 4.0

#: Which way the starfield goes.
#:
#: A preference on the theme rather than three entries in the theme menu.
#: They are one animation with one constant changed — the same particles,
#: pool, parallax layers, sway and twinkle — so three menu entries would put
#: two thirds of a list in front of the user to express one axis, and would
#: then have to answer what a palette or a density means "for Starfield
#: (down)" separately three times. It also composes: direction is orthogonal
#: to speed, size and density, and a menu entry is not.
DRIFT_DIRECTIONS: Tuple[str, ...] = ("up", "down", "random")
DEFAULT_DRIFT_DIRECTION = "up"

_DRIFT_DIRECTION_LABELS = {
    "up": "Up",
    "down": "Down",
    "random": "Every which way",
}

_DRIFT_DIRECTION_NOTES = {
    "up": "Everything rises, the way the shipped starfield always did.",
    "down": "Everything falls, like snow.",
    "random": ("Each speck goes its own way and wanders as it goes — "
               "Brownian motion rather than one shared current."),
}


def is_valid_drift_direction(name) -> bool:
    """True when ``name`` is one of :data:`DRIFT_DIRECTIONS`. Never raises."""
    return name in DRIFT_DIRECTIONS


def _require_drift_direction(name: str) -> str:
    if name not in DRIFT_DIRECTIONS:
        raise ValueError(
            f"unknown starfield direction {name!r}; expected one of "
            f"{', '.join(DRIFT_DIRECTIONS)}")
    return name


def drift_direction_label(name: str) -> str:
    """Human label for a starfield direction, for a menu."""
    return _DRIFT_DIRECTION_LABELS[_require_drift_direction(name)]


def drift_direction_note(name: str) -> str:
    """One-line description of a starfield direction, for a tooltip."""
    return _DRIFT_DIRECTION_NOTES[_require_drift_direction(name)]

#: A background at or below this WCAG relative luminance is treated as dark,
#: which selects additive compositing. The five shipped themes measure 0.000
#: (dark), 0.002 (space), 0.002 (cell), 0.004 (glass) and 0.956 (light), so
#: the exact threshold only ever matters for a custom mid-grey; it sits high
#: because additive is the more forgiving of the two on a mid tone.
DARK_LUMINANCE_MAX = 0.30

#: How far a palette colour is mixed toward white before it is multiplied
#: onto a light page. Undiluted saturated hues multiply to something muddy
#: and far too strong.
LIGHT_TINT = 0.55


# ---------------------------------------------------------------------------
# Small colour helpers
# ---------------------------------------------------------------------------
# Deliberately local rather than imported from ``dna_rain``: this module is
# installed on *every* module screen, and it should not drag the sequencing
# backdrop in behind it just to reuse fifteen lines of arithmetic.

def _as_color(value: Union[QColor, str, None], fallback: QColor) -> QColor:
    """Coerce ``value`` to a valid opaque QColor, or fall back to a colour
    that is one."""
    if value is None:
        color = QColor(fallback)
    else:
        color = QColor(value)
        if not color.isValid():
            color = QColor(fallback)
    color.setAlpha(255)
    return color


def _as_pixmap(value) -> Optional[QPixmap]:
    """Coerce a path / QPixmap / QImage to a usable QPixmap, or ``None``.

    Never raises and never returns a null pixmap: a wallpaper deleted between
    the stylesheet being built and this widget being constructed is a cosmetic
    miss, not a crash.
    """
    if value is None:
        return None
    try:
        if isinstance(value, QPixmap):
            pixmap = value
        elif isinstance(value, QImage):
            pixmap = QPixmap.fromImage(value)
        else:
            pixmap = QPixmap(str(value))
    except Exception:
        return None
    return None if pixmap.isNull() else pixmap


def _mix(a: QColor, b: QColor, t: float) -> QColor:
    """Linear RGB mix, ``t=0`` -> ``a``, ``t=1`` -> ``b``."""
    t = max(0.0, min(1.0, float(t)))
    return QColor(
        int(round(a.red() + (b.red() - a.red()) * t)),
        int(round(a.green() + (b.green() - a.green()) * t)),
        int(round(a.blue() + (b.blue() - a.blue()) * t)),
    )


def _with_alpha(color: QColor, alpha: float) -> QColor:
    out = QColor(color)
    out.setAlphaF(max(0.0, min(1.0, float(alpha))))
    return out


def is_dark_background(color: Union[QColor, str]) -> bool:
    """True when ``color`` is dark enough for additive compositing."""
    return relative_luminance(_as_color(color, QColor("#000000")).name()) \
        <= DARK_LUMINANCE_MAX


def _clamp_int(value, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _pool_size(base: int) -> int:
    """How many elements to roll for a theme whose own count is ``base``.

    Enough for the top of :data:`DENSITY_RANGE`, rolled once at construction.
    Density then paints a prefix of the pool, so the slider never re-rolls
    the field, never makes what is on screen jump, and — because the extra
    draws happen *after* the originals — never disturbs the numbers the
    shipped elements were built from.
    """
    return max(1, int(math.ceil(base * DENSITY_RANGE[1])))


def _theme_background() -> QColor:
    """The current theme's flat page colour, or the dark one if unavailable.

    ``page``, not ``bg``. The docstring said "page colour" all along and
    the code read the *window* colour, which on the dark theme is
    ``#000000`` — so the flat fill under the animation was pure black,
    and on the frames and in the gaps where the animation is thin that is
    what reached the eye. See the ``page`` block in :mod:`spacr.qt.theme`.
    """
    try:
        from ..theme import active_page_colour
        return QColor(active_page_colour())
    except Exception:
        # `page_colour` is imported at module scope, so this arm cannot
        # itself fail on an import the way a second `from ..theme import`
        # could — and a backdrop must never raise on its way to a screen.
        return QColor(page_colour("dark"))


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------

class AmbientEngine:
    """Base class: a deterministic, time-parameterised painter.

    An engine owns no widget and no timer. It holds the constants rolled once
    from its seed, a clock (:attr:`time`), and whatever reusable buffer it
    paints through. Everything it draws is a pure function of ``(seed, time,
    width, height, colours, background)`` — which is what lets a test render
    the same frame twice and compare it byte for byte, and what lets
    :meth:`AmbientWidget.set_theme` swap engines without the animation
    jumping.

    Positions are rolled in *normalised* 0..1 units and multiplied up at paint
    time, so a resize re-frames the animation instead of re-rolling it.
    """

    name = ""

    def __init__(self, colors: Sequence[Union[QColor, str]],
                 background: Union[QColor, str],
                 seed: Optional[int] = None,
                 blur: float = DEFAULT_BLUR,
                 speed: float = DEFAULT_SPEED,
                 size: float = DEFAULT_SIZE,
                 resolution: float = DEFAULT_RESOLUTION,
                 density: float = DEFAULT_DENSITY,
                 direction: str = DEFAULT_DRIFT_DIRECTION):
        self.seed = seed
        self.time = 0.0
        self.frames = 0
        # Before ``_configure``: a subclass may size something from them.
        self.blur = _clamp(blur, *BLUR_RANGE)
        self.speed = _clamp(speed, *SPEED_RANGE)
        self.size = _clamp(size, *SIZE_RANGE)
        self.resolution = _clamp(resolution, *RESOLUTION_RANGE)
        self.density = _clamp(density, *DENSITY_RANGE)
        self.direction = direction if is_valid_drift_direction(direction) \
            else DEFAULT_DRIFT_DIRECTION
        self._colors = self._coerce_colors(colors)
        self._background = _as_color(background, QColor("#000000"))
        self._configure(random.Random(seed))
        self._restyle()

    # -- construction hooks -------------------------------------------
    def _configure(self, rng: random.Random) -> None:
        """Roll the per-element constants. Called exactly once."""

    def _restyle(self) -> None:
        """Re-derive everything that depends on the colours or background."""
        self.dark = is_dark_background(self._background)
        self.mode = (QPainter.CompositionMode_Plus if self.dark
                     else QPainter.CompositionMode_Multiply)
        #: The colour that leaves the layer underneath untouched under
        #: :attr:`mode` — 0 adds nothing, white multiplies to identity.
        self.identity = QColor(0, 0, 0) if self.dark else QColor(255, 255, 255)
        self.paint_colors = [self._tint(c) for c in self._colors]

    def _tint(self, color: QColor) -> QColor:
        """The colour as actually painted, given the background."""
        return QColor(color) if self.dark \
            else _mix(color, QColor(255, 255, 255), LIGHT_TINT)

    @staticmethod
    def _coerce_colors(colors: Sequence[Union[QColor, str]]) -> List[QColor]:
        out = [QColor(c) for c in colors or ()]
        out = [c for c in out if c.isValid()]
        return out or [QColor("#808080")]

    # -- state ---------------------------------------------------------
    @property
    def colors(self) -> List[QColor]:
        return [QColor(c) for c in self._colors]

    @property
    def background(self) -> QColor:
        return QColor(self._background)

    def set_colors(self, colors: Sequence[Union[QColor, str]]) -> None:
        """Swap the palette without disturbing the motion."""
        self._colors = self._coerce_colors(colors)
        self._restyle()

    def set_background(self, color: Union[QColor, str]) -> None:
        """Tell the engine what it is painting onto — this is what decides
        additive versus multiply, so it must be called on a theme switch."""
        self._background = _as_color(color, self._background)
        self._restyle()

    # -- the user controls ---------------------------------------------
    def set_blur(self, value: float) -> None:
        """How much the finished picture is softened. 0.0 is untouched.

        Clamped to :data:`BLUR_RANGE`. Engines that cache anything sized by
        it drop that cache here, never per frame.
        """
        value = _clamp(value, *BLUR_RANGE)
        if value == self.blur:
            return
        self.blur = value
        self._reblur()

    def set_resolution(self, value: float) -> None:
        """How many pixels the scene is shaded into, as a multiplier on this
        theme's own buffer edge. Clamped to :data:`RESOLUTION_RANGE`."""
        value = _clamp(value, *RESOLUTION_RANGE)
        if value == self.resolution:
            return
        self.resolution = value
        self._reresolve()

    def set_density(self, value: float) -> None:
        """How many elements the theme draws, as a multiplier on its own
        count. Clamped to :data:`DENSITY_RANGE`.

        Never re-rolls anything: the pool was built for the top of the range
        at construction and this only changes how much of it is painted, so
        the elements that were on screen stay exactly where they were.
        """
        value = _clamp(value, *DENSITY_RANGE)
        if value == self.density:
            return
        self.density = value
        self._redensify()

    def set_direction(self, name: str) -> None:
        """Which way the elements travel, for the themes that have a way.

        Silently ignores an unknown name rather than raising: this reaches
        every engine, and most of them have nothing to do with it.
        """
        if is_valid_drift_direction(name) and name != self.direction:
            self.direction = name
            self._redirect()

    @property
    def work(self) -> float:
        """What this engine is asking for, as a multiple of its own default.

        Shading cost is buffer pixels times elements. Resolution is a linear
        scale on the buffer's edge, so it enters squared; density is linear
        in the elements. Overridden by the one engine that has no buffer.
        """
        return self.resolution ** 2 * self.density

    def effective_density(self) -> float:
        """:attr:`density`, trimmed to keep :attr:`work` inside
        :data:`WORK_BUDGET`.

        A function of the two settings alone, never of the canvas, so what a
        test measures on ``geometry()`` is what gets painted.
        """
        over = self.work / WORK_BUDGET
        return self.density / over if over > 1.0 else self.density

    def element_count(self, base: int, pool: int) -> int:
        """How many of a pool of ``pool`` elements to draw, when the theme's
        own count is ``base``. At least one: a density slider that can empty
        the screen is an off switch wearing a disguise."""
        return _clamp_int(round(base * self.effective_density()), 1, pool)

    def alpha_scale(self) -> float:
        """What to multiply every element's peak alpha by, given the density.

        Additive compositing means N overlapping shapes are N times the
        light, so a density control with no compensation is a *brightness*
        control wearing a misleading name. Measured, on a page at 0.076:
        mean frame lightness went from 0.135 at density 1.0 to 0.288 at 3.0.
        The backdrop would have become the loudest thing behind a settings
        form — which the alphas in this module were set on a rendered frame
        specifically to prevent (see :data:`AURORA_ALPHA_DARK`).

        So above 1.0 the field's light is *divided among* more elements
        rather than added to it. Below 1.0 nothing is done: quadrupling the
        alpha of a quarter as many blobs clips to white rather than
        compensating, and a sparser field being a quieter one is the right
        answer anyway.

        Density therefore changes the *texture* of the field — how many
        shapes it is made of, and how strongly each one states itself — and
        not how loud the field is. That is the only reading of the control
        that leaves the backdrop legible at both ends of its range.
        """
        return 1.0 / max(1.0, self.effective_density())

    def set_speed(self, value: float) -> None:
        """Multiply every motion in the theme. Clamped to :data:`SPEED_RANGE`.

        Deliberately a clock multiplier rather than a factor inside
        :meth:`geometry`: a user dragging the slider in Preferences changes
        how fast the animation goes *from here*, and never makes what is
        already on screen jump to a different place.
        """
        self.speed = _clamp(value, *SPEED_RANGE)

    def set_size(self, value: float) -> None:
        """Scale every element's size. Clamped to :data:`SIZE_RANGE`."""
        value = _clamp(value, *SIZE_RANGE)
        if value == self.size:
            return
        self.size = value
        self._resize()

    def _reblur(self) -> None:
        """Drop whatever the blur setting sized. Default: nothing to do."""

    def _reresolve(self) -> None:
        """Drop whatever the resolution setting sized. Default: nothing."""

    def _redensify(self) -> None:
        """Drop whatever the density setting sized. Default: nothing."""

    def _redirect(self) -> None:
        """React to a direction change. Default: nothing."""

    def _resize(self) -> None:
        """Drop whatever the size setting sized. Default: nothing to do."""

    def advance(self, dt: float) -> None:
        """Step the clock by ``dt`` seconds. Negative steps are ignored.

        The step is scaled by :attr:`speed`, so the clock counts *animation*
        seconds rather than wall-clock ones: every period, rate and travel
        speed in every theme is expressed against this clock and therefore
        scales with one multiplier.

        :attr:`frames` counts every call, including the ignored ones, so a
        test can prove a hidden widget stopped *asking* for frames rather
        than only proving that its clock stood still.
        """
        if dt > 0:
            self.time += float(dt) * self.speed
        self.frames += 1

    def set_time(self, seconds: float) -> None:
        """Jump the clock — used by the tests, and to carry the clock across
        a theme change."""
        self.time = float(seconds)

    # -- painting ------------------------------------------------------
    def paint(self, painter: QPainter, width: int, height: int) -> None:
        raise NotImplementedError

    def geometry(self, width: int, height: int) -> Tuple[tuple, ...]:
        """What this engine would draw right now, in pixels.

        Every engine answers this and every engine paints *from* it, so a
        test can assert on the geometry and know it is asserting on the
        frame. The tuple shape is per engine and documented there.
        """
        raise NotImplementedError


class _BufferedEngine(AmbientEngine):
    """An engine that paints a soft field into a small reusable QImage.

    The buffer is allocated on the first paint and then only when the widget
    size changes — never per frame. The field is composited into it with the
    same mode used to composite the buffer onto the page, which is what makes
    the layer background-agnostic: addition and multiplication are both
    associative, so ``(black + blobs) + page`` and ``(white * blobs) * page``
    give exactly the result of drawing the blobs straight onto the page. That
    is what lets the same code paint over the flat themes *and* over the Space
    and Cell wallpapers without hiding them.
    """

    #: Longest buffer edge this theme wants at resolution 1.0. Diffuse
    #: themes keep :data:`BUFFER_MAX_EDGE`; a theme with an edge or a fine
    #: repeat in it raises its own.
    base_edge = BUFFER_MAX_EDGE

    def __init__(self, *args, **kwargs):
        self._buffer: Optional[QImage] = None
        super().__init__(*args, **kwargs)

    def _reresolve(self) -> None:
        """A new resolution means a new buffer size — drop the old one now
        rather than leaving the next paint to notice."""
        self._buffer = None

    def resolution_edge(self) -> int:
        """Longest buffer edge under the current resolution setting."""
        return _clamp_int(round(self.base_edge * self.resolution),
                          BUFFER_MIN_EDGE, BUFFER_EDGE_CEILING)

    def buffer_scale(self, width: int, height: int) -> int:
        """Screen pixels per buffer pixel, for a ``width`` x ``height``
        canvas. Always a whole number: a fractional one puts the upscale
        lattice on a beat with itself instead of on the pixel grid.

        The second loop is :data:`BUFFER_MAX_PIXELS`, which is the ceiling
        that actually bounds the cost — the edge alone is a ratio, and a
        ratio does not know how big the display is.
        """
        width, height = max(1, int(width)), max(1, int(height))
        scale = max(1, int(math.ceil(max(width, height)
                                     / self.resolution_edge())))
        while scale < 64 and \
                (width // scale) * (height // scale) > BUFFER_MAX_PIXELS:
            scale += 1
        return scale

    def buffer_size(self, width: int, height: int) -> Tuple[int, int]:
        """Buffer dimensions for a ``width`` x ``height`` canvas."""
        scale = self.buffer_scale(width, height)
        return (max(1, int(width) // scale), max(1, int(height) // scale))

    def blur_scale(self, width: int, height: int) -> float:
        """How far the shaded buffer is averaged down before it goes to the
        canvas, in buffer pixels. 1.0 means "untouched", and the default
        setting means exactly that.

        :data:`BLUR_UNIT_PX` is in *screen* pixels, so this divides by the
        upscale factor: the same blur setting asks for the same softness on
        screen whatever resolution it is shaded at, which is the property
        that makes the two controls independent. It never returns less than
        1.0 — the upscale on its own already softens by ``scale`` pixels, and
        a downscale below 1.0 would be a sharpen, which no amount of
        arithmetic can deliver.
        """
        if self.blur <= 0.0:
            return 1.0
        return max(1.0, BLUR_UNIT_PX * self.blur
                   / self.buffer_scale(width, height))

    def _ensure_buffer(self, width: int, height: int) -> QImage:
        bw, bh = self.buffer_size(width, height)
        buf = self._buffer
        if buf is None or buf.width() != bw or buf.height() != bh:
            buf = QImage(bw, bh, QImage.Format_RGB32)
            self._buffer = buf
        return buf

    def paint(self, painter: QPainter, width: int, height: int) -> None:
        """Shade a frame and put it on the canvas, both here and now.

        Exactly :meth:`shade` followed by :meth:`blit`, minus the copy — the
        buffer goes straight to the canvas, so this path still allocates once
        on resize and never per frame. That is what a widget with no shading
        thread does, and it is what every engine test measures.
        """
        if width <= 0 or height <= 0:
            return
        self.blit(painter, self._shade(width, height), width, height)

    def _shade(self, width: int, height: int) -> QImage:
        """The finished field, in the engine's *own* buffer.

        Returns the buffer itself when the blur is off (which is the
        default), so the result is only valid until the next call. Callers
        that keep it want :meth:`shade`.
        """
        buf = self._ensure_buffer(width, height)
        inner = QPainter(buf)
        inner.fillRect(buf.rect(), self.identity)
        inner.setCompositionMode(self.mode)
        inner.setPen(Qt.NoPen)
        self._paint_field(inner, buf.width(), buf.height())
        inner.end()
        return self._soften(buf, width, height)

    def shade(self, width: int, height: int) -> Optional[QImage]:
        """One finished frame as an image the caller owns. **Any thread.**

        This is the half of a frame that does not have to happen on the GUI
        thread, and the half that a Python worker makes expensive: a
        ``blobs`` field costs 0.240 ms idle and 0.572 ms with one Python
        thread running, ``cells`` 0.538 ms and **26.179 ms** — 48.7 times —
        because it is Python and numpy under the interpreter lock. Splitting
        it out is what lets :class:`_FrameProducer` pay that on a thread
        nobody is looking at. See the module docstring for the whole table.

        A ``QImage`` and a ``QPainter`` over it are legal off the GUI thread
        (a ``QWidget`` is not, and nothing here touches one), and the result
        is byte-identical to the same clock shaded on the GUI thread — which
        ``test_the_backdrop_survives_a_run.py`` asserts for all five buffered
        themes rather than trusting this paragraph.

        Returns ``None`` for an empty canvas. The copy is what makes the
        image the caller's: the producer publishes it and immediately starts
        shading the next frame into the buffer underneath. It costs 0.003 ms
        for ``blobs``, 0.035 ms for the aurora's 2 MiB buffer — 2 % of the
        shading pass it protects.
        """
        if width <= 0 or height <= 0:
            return None
        image = self._shade(width, height)
        return image.copy() if image is self._buffer else image

    def blit(self, painter: QPainter, image: Optional[QImage],
             width: int, height: int) -> None:
        """Put a frame from :meth:`shade` on the canvas. **GUI thread only.**

        The fixed remainder of a frame, and the half that has to stay here:
        it is Qt's C++ raster engine with the interpreter lock released, so
        it is bounded at ~1.2 ms even while a Python worker is running (1.7x
        its idle cost, against 48.7x for the shading it replaces).

        A ``None`` or empty image draws nothing rather than raising, because
        the one caller that can hand it one is a widget whose shading thread
        has not published yet.
        """
        if image is None or image.isNull() or width <= 0 or height <= 0:
            return
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.setCompositionMode(self.mode)
        painter.drawImage(QRect(0, 0, int(width), int(height)), image)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

    def _soften(self, buf: QImage, width: int, height: int) -> QImage:
        """The blur: one area-averaging pass over the finished buffer.

        ``QImage.scaled(..., SmoothTransformation)`` box-filters on the way
        down — it is a real low-pass, not a resample — and the blit that was
        already there carries the result back up. So the whole blur is *one*
        extra read of the buffer and a small write, and the picture makes
        exactly two trips through a filter rather than the three a
        down-up-blit would take.

        A separable box blur at the buffer size was written and measured
        first, because it is the honest answer: NumPy, two cumulative sums
        per axis, 12.8 ms a frame on a 640x360 buffer at 1920x1080. That is
        nine times this whole module's budget, so it is not what ships. The
        cost of what does ship is 0.01-0.11 ms, which on most themes is
        inside the run-to-run spread; see the table in the module docstring.

        Returns the buffer itself when there is nothing to do, which is the
        default and costs nothing.
        """
        factor = self.blur_scale(width, height)
        if factor <= 1.0:
            return buf
        return buf.scaled(max(2, int(round(buf.width() / factor))),
                          max(2, int(round(buf.height() / factor))),
                          Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

    def _paint_field(self, painter: QPainter, width: int, height: int) -> None:
        raise NotImplementedError


# -- blobs ------------------------------------------------------------------

#: How many blobs. Cheap enough to raise (the shading happens over 37 000
#: buffer pixels), but past about twenty the fields merge into a single wash
#: and the individual motion stops being readable.
BLOB_COUNT = 14

#: Every third blob is a small one, so the field has a sense of scale.
BLOB_SMALL_EVERY = 3
BLOB_LARGE_RADIUS = (0.22, 0.46)   # fraction of the short edge
BLOB_SMALL_RADIUS = (0.07, 0.16)

#: Drift, as a fraction of the canvas, and the period of that drift. Long
#: periods are the point: this must never look like it is *moving*, only like
#: it has moved when you look back at it.
BLOB_DRIFT = (0.04, 0.12)
BLOB_DRIFT_PERIOD = (24.0, 70.0)   # seconds

#: The pulse — "changing size" — as a fraction of the base radius.
BLOB_PULSE = (0.12, 0.35)
BLOB_PULSE_PERIOD = (7.0, 19.0)    # seconds

#: Peak alpha at a blob's centre. Light needs more than dark because the
#: colour has been mixed 55 % toward white before it is multiplied — but not
#: as much more as the arithmetic suggests, because multiply keeps its
#: contrast where additive runs out of headroom.
BLOB_ALPHA_DARK = 0.30
BLOB_ALPHA_LIGHT = 0.46

#: The falloff, as ``(stop, alpha multiplier)``. Roughly Gaussian; the point
#: is that it reaches zero *before* the edge of the ellipse, so no blob ever
#: shows a rim.
BLOB_FALLOFF = ((0.0, 1.0), (0.35, 0.60), (0.70, 0.18), (1.0, 0.0))


@dataclass
class Blob:
    """One drifting, pulsing blob, in normalised units."""

    x: float
    y: float
    drift_x: float
    drift_y: float
    rate_x: float
    rate_y: float
    phase_x: float
    phase_y: float
    radius: float
    pulse: float
    pulse_rate: float
    pulse_phase: float
    color: int


class BlobsEngine(_BufferedEngine):
    """Diffuse colour blobs, drifting and pulsing.

    Motion is two independent sines per blob rather than a random walk, so
    position is a pure function of the clock: no accumulated error, and
    ``set_time`` can jump anywhere.

    :meth:`geometry` yields ``(cx, cy, radius)`` per blob, in pixels.
    """

    name = "blobs"

    def _configure(self, rng: random.Random) -> None:
        # Seed the blobs on a jittered 5x3 grid rather than uniformly at
        # random: with only fourteen of them, uniform sampling reliably
        # leaves one corner empty and clumps three in the middle.
        cols, rows = 5, 3
        cells = list(range(cols * rows))
        rng.shuffle(cells)
        self.blobs: List[Blob] = []
        # The pool is rolled for the top of the density range and then a
        # prefix of it is painted. Extending the loop is the one way to add
        # them that leaves the first BLOB_COUNT draws bit-identical: the RNG
        # is consumed in order, so blob 3 gets the numbers it always got.
        for i in range(_pool_size(BLOB_COUNT)):
            cell = cells[i % len(cells)]
            col, row = cell % cols, cell // cols
            small = (i % BLOB_SMALL_EVERY) == 0
            lo, hi = BLOB_SMALL_RADIUS if small else BLOB_LARGE_RADIUS
            self.blobs.append(Blob(
                x=(col + 0.15 + 0.7 * rng.random()) / cols,
                y=(row + 0.15 + 0.7 * rng.random()) / rows,
                drift_x=rng.uniform(*BLOB_DRIFT),
                drift_y=rng.uniform(*BLOB_DRIFT),
                rate_x=2 * math.pi / rng.uniform(*BLOB_DRIFT_PERIOD),
                rate_y=2 * math.pi / rng.uniform(*BLOB_DRIFT_PERIOD),
                phase_x=rng.uniform(0.0, 2 * math.pi),
                phase_y=rng.uniform(0.0, 2 * math.pi),
                radius=rng.uniform(lo, hi),
                pulse=rng.uniform(*BLOB_PULSE),
                pulse_rate=2 * math.pi / rng.uniform(*BLOB_PULSE_PERIOD),
                pulse_phase=rng.uniform(0.0, 2 * math.pi),
                # Straight round-robin, not a random pick: with three
                # colours and fourteen blobs a random assignment leaves a
                # palette colour missing about one run in fifty.
                color=i,
            ))

    def count(self) -> int:
        """How many blobs are painted right now."""
        return self.element_count(BLOB_COUNT, len(self.blobs))

    def geometry(self, width: int, height: int) -> Tuple[tuple, ...]:
        t = self.time
        short = min(width, height)
        out = []
        for blob in self.blobs[:self.count()]:
            cx = (blob.x + blob.drift_x
                  * math.sin(blob.rate_x * t + blob.phase_x)) * width
            cy = (blob.y + blob.drift_y
                  * math.sin(blob.rate_y * t + blob.phase_y)) * height
            radius = blob.radius * short * self.size * (
                1.0 + blob.pulse
                * math.sin(blob.pulse_rate * t + blob.pulse_phase))
            out.append((cx, cy, max(1.0, radius)))
        return tuple(out)

    def _paint_field(self, painter: QPainter, width: int, height: int) -> None:
        peak = (BLOB_ALPHA_DARK if self.dark else BLOB_ALPHA_LIGHT) \
            * self.alpha_scale()
        colors = self.paint_colors
        for blob, (cx, cy, radius) in zip(self.blobs,
                                          self.geometry(width, height)):
            color = colors[blob.color % len(colors)]
            gradient = QRadialGradient(cx, cy, radius)
            for stop, scale in BLOB_FALLOFF:
                gradient.setColorAt(stop, _with_alpha(color, peak * scale))
            painter.setBrush(gradient)
            painter.drawEllipse(QPointF(cx, cy), radius, radius)


# -- aurora -----------------------------------------------------------------
#
# What the real thing does, and what each part of it costs here.
#
# An aurora is not a band of colour sliding across the sky. It is a *folded
# sheet seen edge-on*: charged particles spiral down the geomagnetic field
# lines and light the atmosphere along them, so the sheet is made of vertical
# rays, and the arc is where that sheet crosses the sky. Four things follow,
# and all four are what makes it recognisable:
#
# 1. *Vertical ray striations.* The rays are the defining feature. A smooth
#    band with no rays reads as a gradient, which is what this theme was
#    before.
# 2. *The folds travel ALONG the arc.* The sheet ripples the way a ribbon
#    held at one end does — the fold pattern propagates lengthwise while the
#    arc itself stays where it is. It does not slide sideways as a body.
#    Every wave below is written ``sin(2*pi*(u - v*t)/lambda)``: a phase that
#    depends on position and travels at ``v``, which is a travelling wave and
#    nothing else.
# 3. *Several frequencies at once.* One slow, long, deep fold with faster
#    small ripples riding on it. A single sine is a flag, not an aurora.
# 4. *Brightness surges* running along the arc on their own schedule, faster
#    than the folds and on a different wavelength, so the two are visibly
#    independent phenomena rather than one driving the other.
#
# Then the vertical colour structure, which is pure atomic physics and the
# cheapest realism available. 557.7 nm atomic oxygen (green) through the body,
# 630.0 nm atomic oxygen (red) at the top where the air is thin enough for
# that slow transition to survive the wait, and 427.8 nm ionised nitrogen
# (blue-violet) along the bottom. The lower edge is *sharp* — it is where the
# particles finally run out of altitude — and the top is diffuse.
#
# The single most useful thing to know about that colour structure is that it
# is a function of ALTITUDE, not of position within the curtain. The ramp is
# therefore anchored to the frame and not to the folded edge, and everything
# falls out of it for free:
#
# * where the fold dips low, the sheet's edge reaches into the violet, and
#   where it rises, it does not — which is exactly what photographs show;
# * the curtain looks *taller* where the fold dips, because its top is at a
#   fixed altitude and only its bottom moved. Real rays behave that way for
#   the same reason, and it costs nothing to reproduce;
# * one affine brush transform per curtain is correct, so the curtain is one
#   filled path and not a strip of separately anchored pieces. Anchoring per
#   piece was tried first: it puts a step in the ramp at every seam, and at
#   these alphas the step is ~13/255 — visible as vertical banding.
#
# How it is drawn, and why. Measured at 1920x1080 in the 256 px buffer: one
# ``drawImage`` per ray costs ~5 us of Python-to-Qt overhead, so 150 rays is
# 0.8 ms — more than this theme is allowed in total. Instead each curtain is
# ONE filled path (the folded sheet) painted with a *tiled texture brush*
# whose tile is the ray comb crossed with the colour ramp, and then filled a
# second time with a small per-frame image holding the surge. The ray count
# costs nothing because the rays are the brush rather than the geometry.

#: Three curtains at different depths. Two read as one curtain and a copy;
#: four stop being separable at these alphas.
AURORA_CURTAINS = 3

#: The aurora shades into a buffer four times the linear resolution the
#: diffuse themes use, and this is the measurement that says why.
#:
#: It is the one soft theme with hard structure in it: a sharp lower edge,
#: and a ray comb whose period at 1920 px wide is 36 screen pixels. In the
#: 240x135 buffer the others are happy with, that comb is 4.6 *buffer* pixels
#: across and each ray inside it is one and a half — quantised to whole
#: pixels when the tile is built, then stretched eight-fold. Measured at
#: 1920x1080 against the same frame shaded at full resolution:
#:
#: ===========  =======  ==================  ========================
#:  buffer       scale    ray-comb contrast   lattice on lower edge
#: ===========  =======  ==================  ========================
#:  240x135      8x       77.4 %              1.724
#:  480x270      4x       92.8 %              1.326
#:  960x540      2x       97.8 %              1.033
#:  1920x1080    1x       100 %               1.000
#: ===========  =======  ==================  ========================
#:
#: "Ray-comb contrast" is the RMS of the high-frequency part of a horizontal
#: luminance profile through a curtain, as a share of the same measurement on
#: the fully-resolved frame. Under-resolving a comb does not move the rays,
#: it *smears* them, so this is the number that says whether they survived —
#: and nearly a quarter of them did not.
#:
#: "Lattice" is the block-boundary energy ratio: how much more second-
#: difference energy sits on one phase of the upscale grid than on the
#: others, phase-searched, over the band the front curtain's lower edge runs
#: through. 1.000 means the grid cannot be found in the picture at all.
#:
#: 960 is where both numbers stop moving. Odd scale factors are deliberately
#: skipped over: 3x measured *worse* than 4x (1.580 against 1.326), because
#: an odd upscale beats against the ray comb, and 960 gives an even 2x at
#: 1080p. ``test_the_aurora_is_no_longer_pixelated_at_1080p`` is this table
#: asserted rather than remembered.
AURORA_BUFFER_EDGE = 960

#: Ray length — how far up the sheet is lit — as a fraction of the canvas
#: height, scaled by the size setting. Comfortably deeper than the fold
#: reaches, or a fold crest would lift the sheet's lower edge past the green
#: and out of the top of its own colour ramp.
AURORA_THICKNESS = (0.42, 0.70)

#: Where each curtain's lower edge rests, as a fraction of the canvas height,
#: and the jitter around it. Spread down the frame so the three overlap in
#: depth rather than sitting on top of one another.
AURORA_BASE = (0.62, 0.76, 0.90)
AURORA_BASE_JITTER = 0.05

#: How far down the extra curtains a raised density asks for are pushed,
#: per tier of three. Slightly under half the spacing between the three
#: shipped bases, so a denser aurora interleaves with itself instead of
#: doubling up on the same three altitudes.
AURORA_TIER_OFFSET = 0.055

#: The arc's slope across the frame, as a fraction of the canvas height. An
#: arc that is exactly level reads as a horizon line. Small, because the
#: colour ramp is anchored to altitude: a steeply tilted arc would have one
#: end of it sitting in a different colour from the other.
AURORA_TILT = 0.06

#: How much wider than the canvas the arc is drawn. The folds have to enter
#: and leave the frame rather than terminating at its edges.
AURORA_OVERHANG = 1.12

#: The slow bob of the whole arc's altitude, and its period. This is the only
#: bulk motion the curtain has, and it is vertical: the folds do the rest.
AURORA_DRIFT = (0.05, 0.18)
AURORA_DRIFT_PERIOD = (30.0, 90.0)  # seconds
AURORA_HUE_PERIOD = (18.0, 46.0)    # seconds per colour cross-fade cycle

#: The fold, as three superposed travelling waves: ``(amplitude as a fraction
#: of the canvas height, wavelength as a fraction of the arc's length, travel
#: speed in arc-lengths per second)``. Long slow fold, medium ripple, fine
#: ripple — the ratio between them is what stops it reading as a single sine,
#: and the speeds differ so the pattern never repeats itself.
AURORA_FOLDS = (
    (0.055, 0.85, 0.020),
    (0.022, 0.33, 0.052),
    (0.009, 0.17, 0.088),
)

#: How far the fold can reach either way, which is what the colour ramp has
#: to be anchored below.
AURORA_FOLD_REACH = sum(amp for amp, _wl, _v in AURORA_FOLDS)

#: The brightness surge running along the arc — faster than any fold and on
#: its own wavelength. ``(depth, wavelength, speed)``.
AURORA_PULSE = (0.62, 0.34, 0.075)

#: How far up the curtain a surge reaches, as a share of the ray length, and
#: how strong it is against the curtain's own peak alpha. Surges brighten the
#: base of the sheet; the diffuse top does not pulse.
#:
#: The surge is painted over its own shorter path rather than over the whole
#: sheet. Above :data:`AURORA_PULSE_HEIGHT` its texture is transparent, and a
#: transparent source pixel still costs a read and a write of the destination
#: — 45 % of the curtain's area, for nothing. That one change took the pass
#: from 0.49 ms to 0.20.
AURORA_PULSE_HEIGHT = 0.55
AURORA_PULSE_GAIN = 0.85

#: Resolution of the per-frame surge image. It is stretched over the curtain
#: with bilinear filtering, so it only has to resolve the pulse: 16 samples
#: across an arc holding three wavelengths is five per wavelength, and the
#: gradient's linear interpolation between them is under 5 % off a sine. It
#: started at 40x16 and that cost 0.27 ms a frame in ``setColorAt`` calls
#: alone — three quarters of it thrown away by the bilinear filter.
AURORA_PULSE_TEXTURE = (16, 16)

#: How far past the curtain, as a share of the ray length, that image is
#: stretched. A texture brush *wraps*, and a bilinear sample taken on the
#: image's first row blends it with its last one — which drew a bright
#: hairline straight across the top of every curtain until this padding put
#: both rows outside the sheet, where nothing can sample them.
AURORA_PULSE_PAD = 0.1

#: How finely the surge image is cached. Its content is a pure function of
#: the pulse's phase, so it does not have to be rebuilt every frame — and
#: rebuilding it was 0.15 ms of a frame, nearly all of it spent constructing
#: gradient stops. 64 steps is one every 0.07 s at the shipped pulse speed,
#: which moves the pattern half a percent of the arc at a time.
AURORA_PULSE_STEPS = 64
AURORA_PULSE_CACHE = 256

#: Per curtain: ``(rate multiplier, ray-spacing multiplier, alpha
#: multiplier)``. Different rates are what stop the three from reading as one
#: thick curtain; the further ones have finer rays and less of them.
AURORA_DEPTHS = (
    (1.00, 1.00, 1.00),
    (0.62, 0.74, 0.72),
    (1.45, 1.36, 0.55),
)

#: Spacing between ray centres as a fraction of the canvas width, scaled by
#: the size setting. Expressed against the *canvas*, not the buffer, so the
#: blur setting changes how soft the rays are and not how many there are.
AURORA_RAY_SPACING = 0.019
AURORA_RAY_MIN_PX = 1.5

#: Samples along the arc. 40 resolves the 0.17 fold (Nyquist wants 12) with
#: room to spare, and every one of them is Python arithmetic on every frame.
AURORA_COLUMNS = 40

#: The tile: three rays of different widths, so the comb repeats every third
#: ray instead of every ray and never reads as a picket fence. The rays sit
#: on a floor rather than on nothing, because the sheet between them still
#: glows — rays are a modulation of a curtain, not a row of separate bars.
#: ``(centre, half width, intensity)``, all as fractions of the tile.
#: Sharper than the first version, on request ("the rays of light should be
#: even sharper"). Two changes make an edge, and only together: the half
#: widths come in by about a third, and the floor between the rays drops.
#: Narrowing alone leaves thin rays sitting on a bright sheet, which reads
#: as a lighter curtain rather than as a defined ray.
AURORA_TILE_RAYS = ((0.17, 0.075, 1.00), (0.49, 0.055, 0.86),
                    (0.80, 0.065, 0.94))
AURORA_TILE_FLOOR = 0.34

#: How long each ray in the tile is, as a fraction of the full ray length,
#: and how fast it breathes. One entry per entry in AURORA_TILE_RAYS.
#:
#: "each ray should at different speeds be changing length" -- so the
#: periods are deliberately not multiples of one another, or the three
#: would return to the same arrangement on a short cycle and the eye would
#: find it. 11, 17 and 7 seconds beat against each other for 21 minutes.
#:
#: Never reaching 1.0 for the longest, nor 0 for the shortest: a ray that
#: touches the full height reads as the curtain itself rather than as a ray
#: in it, and one that vanishes leaves a gap that looks like a rendering
#: fault rather than like weather.
AURORA_RAY_LIFE = ((11.0, 0.00), (17.0, 0.37), (7.0, 0.71))
AURORA_RAY_LENGTH = (0.55, 0.98)

#: Quantisation of the breathing, for the same reason the shimmer is
#: quantised: the tile is a cached texture, and a length that follows the
#: clock exactly would rebuild all three tiles every frame. Eight steps
#: across the range is about 5% of the ray length per step, which is below
#: what the eye resolves on a slow fade at this size.
AURORA_LENGTH_STEPS = 8

#: How much of a ray's tip is taper, as a fraction of the FULL ray length.
#: A square cut gives a shortened ray a flat top, which reads as a broken
#: ray and measurably sharpens the curtain's upper edge -- the asymmetry
#: between the hard lower edge and the diffuse top is as recognisable as
#: the colour, and there is a test on it.
AURORA_RAY_FEATHER = 0.22

#: Where in the tile the colour ramp sits, as fractions of its height. What
#: is left over at each end is a transparent guard band. A tiled brush
#: *repeats*, so the instant the sheet reached past the ramp it would wrap
#: round and paint the violet fringe along the top of the curtain. The guards
#: make that impossible rather than unlikely.
#:
#: There is no tile *size* here on purpose. The tile is built at exactly the
#: pixel size it will be painted at — one tile per ray period across, one ray
#: length plus its guards down — so the brush needs a translation and nothing
#: else. Measured, per curtain fill at 1920x1080: a brush carrying a scale
#: costs 0.106 ms, a pre-scaled one carrying only a translation costs 0.061,
#: which is what a flat colour costs. Qt's raster engine has a fast tiled
#: blit for ``TxTranslate`` brushes and a per-pixel inverse transform for
#: everything else, and this is the whole difference between them.
AURORA_TILE_RAMP = (0.10, 0.90)

#: Smallest tile, in pixels. Below about this the ray comb is finer than the
#: buffer can hold and turns into noise.
AURORA_TILE_MIN_PX = 3

#: How many distinct tiles to keep. Three curtains times twelve shimmer steps
#: is 36, and nine curtains — the top of the density range — times twelve is
#: 108, which is why this is not the 96 it started at: a cache one short of
#: the working set is a cache that is cleared every frame. The rest of the
#: headroom is for a window being resized, which changes the pixel size the
#: tiles are built at.
#: Raised for the breathing: the working set is now curtains x shimmer
#: steps x length steps, and a cache one short of the working set is a
#: cache that is cleared every frame -- which is the mistake this number
#: already carries a comment about.
#: 9 curtains x 12 shimmer steps x 8 length steps = 864, plus headroom for
#: a window being resized. 768 was the first guess and it was 96 SHORT of
#: the densest working set -- exactly the mistake the paragraph above
#: describes, made again while adding a dimension to it.
AURORA_TILE_CACHE = 1024

#: The vertical structure, lower edge upward: ``(height fraction, palette
#: role, alpha)``. Full strength immediately at the bottom — the sheet's lower
#: edge is a hard cut, and it is made by the polygon, not by the ramp. Above
#: the middle it fades out over half the ray length, which is the diffuse top.
#: The asymmetry between those two edges is as recognisable as the colour.
AURORA_RAMP = (
    (0.00, "fringe", 0.66),
    (0.05, "fringe", 0.92),
    (0.11, "main", 1.00),
    (0.42, "main", 0.74),
    (0.62, "blend", 0.36),
    (0.82, "high", 0.15),
    (1.00, "high", 0.00),
)

#: How far the curtain's body colour is allowed to wander towards another
#: entry in the palette. Small on purpose: the body of an aurora is one
#: emission line and stays that colour — it shimmers, it does not turn red.
AURORA_HUE_BLEND = 0.28

#: Quantisation of that shimmer, so the ray tile is built a few dozen times
#: in the life of the widget instead of three times a frame.
AURORA_HUE_STEPS = 12

#: Higher than the old flat bands needed, because the ray comb, the ramp and
#: the depth multiplier each take a bite out of it before anything reaches
#: the page. Set on the mean lightness of a rendered frame rather than by
#: eye, because "does it look too strong" is exactly the judgement that goes
#: wrong on somebody else's monitor: 0.168 here, against 0.161 for blobs and
#: 0.151 for ripple on a page at 0.078. This paints behind a settings form
#: and is not allowed to be the loudest thing on it.
AURORA_ALPHA_DARK = 0.35
AURORA_ALPHA_LIGHT = 0.52


@dataclass
class Curtain:
    """One aurora curtain, in normalised units."""

    y: float
    height: float
    tilt: float
    drift: float
    rate: float
    phase: float
    hue_rate: float
    hue_phase: float
    fold_phase: Tuple[float, ...]
    pulse_phase: float
    depth: int
    color: int


class AuroraEngine(_BufferedEngine):
    """Folded curtains of vertical rays, rippling along their own length.

    See the block comment above for the phenomenon, for why the colour ramp
    is anchored to the frame rather than to the curtain, and for why it is
    painted as two brush fills per curtain rather than as a few hundred
    sprites.

    :meth:`geometry` yields ``(x, y_bottom, visible_height, brightness)`` per
    sampled column of every curtain, in pixels, ``AURORA_COLUMNS + 1`` of them
    per curtain in curtain order. The painter builds its paths from exactly
    those numbers, so a test that tracks a fold crest through ``geometry`` is
    tracking the crest that is on screen. ``brightness`` is the surge, and it
    is the *model's* value: what gets painted is the same function with its
    phase quantised into :data:`AURORA_PULSE_STEPS` so the surge texture can
    be cached (see :meth:`_surge`).
    """

    name = "aurora"
    base_edge = AURORA_BUFFER_EDGE

    def __init__(self, *args, **kwargs):
        self._tiles: Dict[Tuple[int, int], QImage] = {}
        self._surges: Dict[int, QImage] = {}
        self._pulse_mask: Optional[QImage] = None
        super().__init__(*args, **kwargs)

    def _configure(self, rng: random.Random) -> None:
        self.curtains: List[Curtain] = []
        for i in range(_pool_size(AURORA_CURTAINS)):
            # Each extra tier of three sits a little lower than the last, or
            # a dense aurora would be three curtains painted on top of each
            # other inside one jitter's width rather than a deeper one.
            base = (AURORA_BASE[i % len(AURORA_BASE)]
                    + (i // len(AURORA_BASE)) * AURORA_TIER_OFFSET)
            self.curtains.append(Curtain(
                y=base + rng.uniform(-AURORA_BASE_JITTER, AURORA_BASE_JITTER),
                height=rng.uniform(*AURORA_THICKNESS),
                tilt=rng.uniform(-AURORA_TILT, AURORA_TILT),
                drift=rng.uniform(*AURORA_DRIFT),
                rate=2 * math.pi / rng.uniform(*AURORA_DRIFT_PERIOD),
                phase=rng.uniform(0.0, 2 * math.pi),
                hue_rate=2 * math.pi / rng.uniform(*AURORA_HUE_PERIOD),
                hue_phase=rng.uniform(0.0, 2 * math.pi),
                # Every wave gets its own phase, or the three curtains fold
                # in lockstep and the depth illusion collapses.
                fold_phase=tuple(rng.uniform(0.0, 2 * math.pi)
                                 for _ in AURORA_FOLDS),
                pulse_phase=rng.uniform(0.0, 2 * math.pi),
                depth=i,
                color=i,
            ))

    def _restyle(self) -> None:
        super()._restyle()
        self._tiles = {}
        self._surges = {}

    def _resize(self) -> None:
        # Both caches are keyed on pixel sizes derived from it.
        self._tiles = {}
        self._surges = {}

    # -- the model -----------------------------------------------------
    def count(self) -> int:
        """How many curtains are painted right now."""
        return self.element_count(AURORA_CURTAINS, len(self.curtains))

    def _rate(self, curtain: Curtain) -> float:
        return AURORA_DEPTHS[curtain.depth % len(AURORA_DEPTHS)][0]

    def fold(self, curtain: Curtain, u: float, t: float) -> float:
        """Fold displacement at position ``u`` along the arc, as a fraction
        of the canvas height.

        ``u`` runs 0..1 from one end of the arc to the other. Each component
        is ``sin(2*pi*(u - v*t)/lambda)``: at a fixed time it is a shape in
        ``u``, and as ``t`` advances that shape *slides along u* at ``v``
        while the arc itself goes nowhere. That is the whole difference
        between an aurora and a curtain being dragged sideways, and it is the
        one property of this engine worth testing directly.

        Scaled by the size setting along with everything else: a curtain half
        the height with folds the same depth is a different phenomenon, not a
        smaller one.
        """
        rate = self._rate(curtain)
        total = 0.0
        for (amp, wavelength, speed), phase in zip(AURORA_FOLDS,
                                                   curtain.fold_phase):
            total += amp * math.sin(
                2 * math.pi * (u - speed * rate * t) / wavelength + phase)
        return total * self.size

    def pulse(self, curtain: Curtain, u: float, t: float) -> float:
        """The surge's brightness at ``u``, in 0..1. Another travelling wave,
        deliberately faster and shorter than every fold."""
        depth, wavelength, speed = AURORA_PULSE
        travelling = 0.5 + 0.5 * math.sin(
            2 * math.pi * (u - speed * self._rate(curtain) * t) / wavelength
            + curtain.pulse_phase)
        return 1.0 - depth + depth * travelling

    def anchor(self, curtain: Curtain, height: int) -> Tuple[float, float]:
        """``(ramp zero, ray length)`` for one curtain, in pixels.

        The ramp's zero is the altitude the emission stops at, so it sits
        below everything the fold and the tilt can do — that is what keeps
        every column of the sheet inside its own colour ramp.
        """
        base = curtain.y + curtain.drift * math.sin(
            curtain.rate * self.time + curtain.phase)
        reach = (AURORA_FOLD_REACH + abs(curtain.tilt) * 0.5) * self.size
        return ((base + reach) * height,
                max(1.0, curtain.height * self.size * height))

    def geometry(self, width: int, height: int) -> Tuple[tuple, ...]:
        """Every travelling wave, evaluated along every arc.

        The loop body is :meth:`fold` and :meth:`pulse` written out with
        their constant parts hoisted — ``sin(2*pi*(u - v*t)/lambda + phi)``
        is ``sin(k*u + (phi - k*v*t))``, and ``k`` and the bracket do not
        depend on the column. It is the same arithmetic; it is here rather
        than behind those two calls because this runs a hundred and twenty
        times a frame and a Python call is not free. ``test_aurora_geometry_
        is_the_model_it_documents`` holds the two forms together.
        """
        t = self.time
        out = []
        sin = math.sin
        two_pi = 2 * math.pi
        span = width * AURORA_OVERHANG
        left = (width - span) * 0.5
        columns = AURORA_COLUMNS
        p_depth, p_wavelength, p_speed = AURORA_PULSE
        for curtain in self.curtains[:self.count()]:
            rate = self._rate(curtain)
            depth_alpha = AURORA_DEPTHS[
                curtain.depth % len(AURORA_DEPTHS)][2]
            zero, ray = self.anchor(curtain, height)
            base = curtain.y + curtain.drift * math.sin(
                curtain.rate * t + curtain.phase)
            top = zero - ray
            tilt = curtain.tilt * self.size
            folds = [(amp * self.size, two_pi / wavelength,
                      phase - two_pi * speed * rate * t / wavelength)
                     for (amp, wavelength, speed), phase
                     in zip(AURORA_FOLDS, curtain.fold_phase)]
            p_k = two_pi / p_wavelength
            p_phase = (curtain.pulse_phase
                       - two_pi * p_speed * rate * t / p_wavelength)
            for i in range(columns + 1):
                u = i / columns
                displacement = base + tilt * (u - 0.5)
                for amp, k, phase in folds:
                    displacement += amp * sin(k * u + phase)
                y = displacement * height
                bright = depth_alpha * (
                    1.0 - p_depth + p_depth
                    * (0.5 + 0.5 * sin(p_k * u + p_phase)))
                out.append((left + u * span, y,
                            y - top if y > top else 0.0, bright))
        return tuple(out)

    def hue_phase(self, curtain: Curtain) -> float:
        """Where this curtain's slow colour shimmer stands, in 0..1."""
        return 0.5 + 0.5 * math.sin(
            curtain.hue_rate * self.time + curtain.hue_phase)

    def curtain_color(self, curtain: Curtain, quantised: bool = False
                      ) -> QColor:
        """The curtain's body colour right now.

        Always built from the palette's *first* colour, wandering up to
        :data:`AURORA_HUE_BLEND` of the way towards one of the others and
        back. Every curtain shares that body colour on purpose: the body of
        an aurora is a single emission line — 557.7 nm oxygen — and the
        palette's remaining entries are the top and the fringe, which the
        ramp puts above and below it. Giving curtain two a red body and
        curtain three a violet one, which is what indexing the palette by
        curtain would do, is the one thing that stops the whole theme reading
        as an aurora.

        :param quantised: snap the shimmer to :data:`AURORA_HUE_STEPS` so the
            ray tile can be cached.
        """
        colors = self.paint_colors
        body = colors[0]
        # Never index 0: a curtain whose wander target is its own body colour
        # does not shimmer at all, which is what happened to the third one on
        # every three-colour palette.
        wander = colors[1 + curtain.color % (len(colors) - 1)] \
            if len(colors) > 1 else body
        u = self.hue_phase(curtain)
        if quantised:
            u = round(u * (AURORA_HUE_STEPS - 1)) / (AURORA_HUE_STEPS - 1)
        return _mix(body, wander, AURORA_HUE_BLEND * u)

    def ramp_colors(self, curtain: Curtain, quantised: bool = False
                    ) -> Dict[str, QColor]:
        """The four palette roles for one curtain: the body, the high red,
        the low fringe, and the overlap between body and high.

        Fixed roles rather than a rotation, because the vertical order is
        physics. With ``borealis``, ``main`` is the 557.7 nm green, ``high``
        the 630.0 nm red, ``fringe`` the 427.8 nm violet and ``blend`` the
        pale yellow-green where the first two overlap. A palette with fewer
        than four colours reuses what it has.
        """
        colors = self.paint_colors
        n = len(colors)
        main = self.curtain_color(curtain, quantised=quantised)
        high = colors[1 % n]
        fringe = colors[2 % n]
        blend = colors[3] if n > 3 else _mix(main, high, 0.5)
        return {"main": main, "high": high, "fringe": fringe, "blend": blend}

    # -- painting ------------------------------------------------------
    def ray_lengths(self, curtain: Curtain) -> Tuple[float, ...]:
        """Each ray's current length, as a fraction of the full one.

        One value per entry in :data:`AURORA_TILE_RAYS`, quantised into
        :data:`AURORA_LENGTH_STEPS` so the tile stays cacheable -- a length
        that followed the clock exactly would rebuild every tile every
        frame, which is the cost the tile cache exists to avoid.

        The periods in :data:`AURORA_RAY_LIFE` are deliberately not
        multiples of one another, and the curtain's own phase is added, so
        two curtains never breathe together either.
        """
        low, high = AURORA_RAY_LENGTH
        out = []
        for period, offset in AURORA_RAY_LIFE:
            angle = (2 * math.pi * (self.time / period + offset)
                     + curtain.pulse_phase)
            # Quantise the UNIT and then map, not the mapped value. The
            # other way round quantises [low, high] against a 0..1 grid, so
            # only the top of the range survives -- measured, it gave four
            # levels spanning 0.80..0.98 out of an intended 0.55..0.98, and
            # the breathing was a fifth of the depth it should have been.
            unit = 0.5 * (1.0 + math.sin(angle))
            stepped = round(unit * (AURORA_LENGTH_STEPS - 1)) \
                / (AURORA_LENGTH_STEPS - 1)
            out.append(low + (high - low) * stepped)
        return tuple(out)

    def _tile(self, curtain: Curtain, peak: float, width: int,
              height: int) -> QImage:
        """The ray comb crossed with the vertical colour ramp, as a tiling
        texture, built at the exact pixel size it will be painted at.

        Cached per (curtain, quantised shimmer, size). Nothing about it
        changes from frame to frame: the ray period and the ray length are
        fixed for a given canvas, and the curtain's slow colour shimmer is
        quantised into :data:`AURORA_HUE_STEPS`. Three dozen of these get
        built in the life of the widget, against three a frame if the tile
        followed the clock.
        """
        step = int(round(self.hue_phase(curtain) * (AURORA_HUE_STEPS - 1)))
        lengths = self.ray_lengths(curtain)
        key = (curtain.depth, step, width, height, lengths)
        tile = self._tiles.get(key)
        if tile is not None:
            return tile
        if len(self._tiles) >= AURORA_TILE_CACHE:
            # Only a long run of resizes can get here. Start again rather
            # than grow without bound.
            self._tiles = {}

        top_f, bottom_f = AURORA_TILE_RAMP
        ramp_top = int(round(top_f * height))
        ramp_bottom = max(ramp_top + 1, int(round(bottom_f * height)))
        tile = QImage(width, height, QImage.Format_ARGB32_Premultiplied)
        tile.fill(Qt.transparent)
        roles = self.ramp_colors(curtain, quantised=True)
        alpha = peak * AURORA_DEPTHS[curtain.depth % len(AURORA_DEPTHS)][2]
        inner = QPainter(tile)
        inner.setPen(Qt.NoPen)
        # Ramp position 0 is the curtain's lower edge, which is the *bottom*
        # of the ramp rows, so the gradient runs upward through the image.
        gradient = QLinearGradient(0.0, float(ramp_bottom), 0.0,
                                   float(ramp_top))
        for stop, role, scale in AURORA_RAMP:
            gradient.setColorAt(stop, _with_alpha(roles[role], alpha * scale))
        inner.setBrush(gradient)
        inner.drawRect(0, ramp_top, width, ramp_bottom - ramp_top)
        # ... then cut the ray comb out of it. DestinationIn keeps the colour
        # and replaces the alpha, which is one pass over 1 920 pixels, done
        # about three dozen times in the life of the widget.
        inner.setCompositionMode(QPainter.CompositionMode_DestinationIn)
        comb = QLinearGradient(0.0, 0.0, float(width), 0.0)
        floor = QColor(0, 0, 0, int(round(255 * AURORA_TILE_FLOOR)))
        comb.setColorAt(0.0, floor)
        for centre, half, strength in AURORA_TILE_RAYS:
            comb.setColorAt(max(0.0, centre - half), floor)
            comb.setColorAt(centre,
                            QColor(0, 0, 0, int(round(255 * strength))))
            comb.setColorAt(min(1.0, centre + half), floor)
        comb.setColorAt(1.0, floor)
        inner.setBrush(comb)
        inner.drawRect(0, 0, width, height)

        # Each ray cut to its own length. Done AFTER the comb, still in
        # DestinationIn, so it takes alpha away from one ray's band without
        # touching its neighbours or the sheet between them.
        #
        # Cut from the TOP: an aurora ray is anchored at the lower edge and
        # reaches upward, so a ray that is breathing shortens away from its
        # tip. Shortening from the bottom would lift it off the curtain's
        # edge and look like it is floating.
        for (centre, half, _strength), length in zip(AURORA_TILE_RAYS,
                                                     lengths):
            if length >= 1.0:
                continue
            left = int(round(max(0.0, centre - half) * width))
            right = int(round(min(1.0, centre + half) * width))
            if right <= left:
                continue
            # `ramp_bottom` is the curtain's lower edge and `ramp_top` its
            # tip, so the kept part runs upward from the bottom.
            kept = int(round((ramp_bottom - ramp_top) * length))
            cut_bottom = ramp_bottom - kept
            # The tip fades over a fixed share of the FULL ray length, so a
            # short ray and a long one taper alike rather than the short one
            # being all taper.
            feather = max(1, int(round(
                (ramp_bottom - ramp_top) * AURORA_RAY_FEATHER)))
            if cut_bottom <= 0:
                continue
            # FEATHERED, not a hard rectangle. A square cut gives a
            # shortened ray a flat tip, which is both wrong -- a real ray
            # fades out at the top -- and measurable: it sharpened the
            # curtain's upper edge until the lower-edge-to-upper-edge
            # contrast fell from 2.5x to 2.1x and
            # `test_the_lower_edge_is_sharp_and_the_top_is_diffuse` caught
            # it. The asymmetry between the two edges is as recognisable
            # as the colour, so it is not something to trade away for a
            # cheaper fill.
            fade = QLinearGradient(0.0, float(max(0, cut_bottom - feather)),
                                   0.0, float(cut_bottom))
            fade.setColorAt(0.0, QColor(0, 0, 0, 0))
            fade.setColorAt(1.0, QColor(0, 0, 0, 255))
            inner.setBrush(fade)
            inner.drawRect(left, 0, right - left, cut_bottom)
        inner.end()
        self._tiles[key] = tile
        return tile

    def _mask(self) -> QImage:
        """The surge's vertical profile: solid along the lower edge, gone by
        :data:`AURORA_PULSE_HEIGHT` of the way up. Built once, then reused as
        the alpha of every per-frame surge image.

        Positioned in the *padded* band (see :data:`AURORA_PULSE_PAD`), which
        is why the stops are not at 0 and ``AURORA_PULSE_HEIGHT``: the
        curtain's lower edge sits a padding's worth up from the bottom of the
        image, and the ray length is a padded fraction of its height.
        """
        if self._pulse_mask is None:
            width, height = AURORA_PULSE_TEXTURE
            pad = AURORA_PULSE_PAD
            band = 1.0 + 2 * pad
            edge = pad / band              # the curtain's lower edge
            reach = AURORA_PULSE_HEIGHT / band
            mask = QImage(width, height, QImage.Format_ARGB32_Premultiplied)
            mask.fill(Qt.transparent)
            inner = QPainter(mask)
            inner.setPen(Qt.NoPen)
            # Stops run bottom-to-top, so 0.0 is the bottom of the image.
            fade = QLinearGradient(0.0, float(height), 0.0, 0.0)
            fade.setColorAt(0.0, QColor(0, 0, 0, 255))
            fade.setColorAt(edge, QColor(0, 0, 0, 255))
            fade.setColorAt(edge + reach * 0.45, QColor(0, 0, 0, 185))
            fade.setColorAt(min(1.0, edge + reach), QColor(0, 0, 0, 0))
            fade.setColorAt(1.0, QColor(0, 0, 0, 0))
            inner.setBrush(fade)
            inner.drawRect(0, 0, width, height)
            inner.end()
            self._pulse_mask = mask
        return self._pulse_mask

    def _surge(self, curtain: Curtain, peak: float) -> QImage:
        """The travelling surge for one curtain, as a small image.

        Horizontally it is the pulse; vertically it is the cached fade. It
        has to be a two-dimensional texture rather than a gradient brush: a
        horizontal gradient alone has no vertical falloff, so it would cut
        off in a hard line across the curtain, and putting the falloff in the
        path instead only moves the hard line somewhere else.

        Cached on the pulse's phase, quantised, plus the curtain's shimmer
        step — which is everything its content depends on, so the cache is a
        memo and not an approximation of the model. It still steps in time,
        and :data:`AURORA_PULSE_STEPS` is what decides how finely.
        """
        width, height = AURORA_PULSE_TEXTURE
        _depth, wavelength, speed = AURORA_PULSE
        phase = (curtain.pulse_phase
                 - 2 * math.pi * speed * self._rate(curtain) * self.time
                 / wavelength)
        step = int(round(phase % (2 * math.pi)
                         / (2 * math.pi) * AURORA_PULSE_STEPS))
        hue = int(round(self.hue_phase(curtain) * (AURORA_HUE_STEPS - 1)))
        key = (curtain.depth, step % AURORA_PULSE_STEPS, hue)
        image = self._surges.get(key)
        if image is not None:
            return image
        if len(self._surges) >= AURORA_PULSE_CACHE:
            self._surges = {}

        image = QImage(width, height, QImage.Format_ARGB32_Premultiplied)
        image.fill(Qt.transparent)
        inner = QPainter(image)
        inner.setPen(Qt.NoPen)
        color = self.curtain_color(curtain, quantised=True)
        gain = peak * AURORA_PULSE_GAIN
        depth_alpha = AURORA_DEPTHS[curtain.depth % len(AURORA_DEPTHS)][2]
        quantised = step % AURORA_PULSE_STEPS * 2 * math.pi \
            / AURORA_PULSE_STEPS
        gradient = QLinearGradient(0.0, 0.0, float(width), 0.0)
        stops = width
        for k in range(stops):
            u = k / (stops - 1)
            bright = depth_alpha * self._pulse_at(u, quantised)
            gradient.setColorAt(u, _with_alpha(color, gain * bright))
        inner.setBrush(gradient)
        inner.drawRect(0, 0, width, height)
        inner.setCompositionMode(QPainter.CompositionMode_DestinationIn)
        inner.drawImage(0, 0, self._mask())
        inner.end()
        self._surges[key] = image
        return image

    @staticmethod
    def _pulse_at(u: float, phase: float) -> float:
        """The surge profile at ``u`` for a given travelling phase."""
        depth, wavelength, _speed = AURORA_PULSE
        return 1.0 - depth + depth * (
            0.5 + 0.5 * math.sin(2 * math.pi * u / wavelength + phase))

    def _paint_field(self, painter: QPainter, width: int, height: int) -> None:
        peak = (AURORA_ALPHA_DARK if self.dark else AURORA_ALPHA_LIGHT) \
            * self.alpha_scale()
        # The fold is a near-horizontal edge in a buffer that is about to be
        # stretched sevenfold. Without antialiasing it upscales as a visible
        # staircase; with it, it costs about 0.03 ms.
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        samples = self.geometry(width, height)
        stride = AURORA_COLUMNS + 1
        top_f, bottom_f = AURORA_TILE_RAMP
        rays_per_tile = len(AURORA_TILE_RAYS)
        pulse_w, pulse_h = AURORA_PULSE_TEXTURE
        ray_px = max(AURORA_RAY_MIN_PX,
                     AURORA_RAY_SPACING * self.size * width)
        for index, curtain in enumerate(self.curtains[:self.count()]):
            columns = samples[index * stride:(index + 1) * stride]
            if len(columns) < 2:
                continue
            zero, ray = self.anchor(curtain, height)
            top = zero - ray
            sheet = self._sheet(columns, top)

            spacing = ray_px * AURORA_DEPTHS[
                curtain.depth % len(AURORA_DEPTHS)][1]
            tile = self._tile(
                curtain, peak,
                max(AURORA_TILE_MIN_PX,
                    int(round(spacing * rays_per_tile))),
                max(AURORA_TILE_MIN_PX,
                    int(round(ray / (bottom_f - top_f)))))
            brush = QBrush(tile)
            # Translation only — see AURORA_TILE_RAMP for why that matters.
            brush.setTransform(QTransform.fromTranslate(
                0.0, zero - bottom_f * tile.height()))
            painter.setBrush(brush)
            painter.drawPath(sheet)

            # The surge, over its own shorter path: it is transparent above
            # AURORA_PULSE_HEIGHT and the rest of the sheet is not worth
            # compositing nothing onto. It is a texture brush rather than a
            # blit so that the path clips it — a surge that spilled past the
            # sheet's lower edge would soften the one edge that has to stay
            # hard.
            left, right = columns[0][0], columns[-1][0]
            band = ray * (1.0 + 2 * AURORA_PULSE_PAD)
            surge = QBrush(self._surge(curtain, peak))
            surge.setTransform(QTransform(
                (right - left) / pulse_w, 0.0, 0.0, band / pulse_h,
                left, top - ray * AURORA_PULSE_PAD))
            painter.setBrush(surge)
            painter.drawPath(self._sheet(
                columns, zero - ray * (AURORA_PULSE_HEIGHT
                                       + AURORA_PULSE_PAD)))
        painter.setRenderHint(QPainter.Antialiasing, False)

    @staticmethod
    def _sheet(columns, top: float) -> QPainterPath:
        """The sheet as a closed path: along its folded lower edge, then
        straight back across a flat top.

        The top is flat, and that is not a shortcut. It sits exactly where
        the colour ramp has faded to nothing, so the polygon's upper boundary
        is invisible — which is the only way to get a *diffuse* top out of a
        hard-edged polygon. All the visible shape is in the lower edge, which
        is where a real curtain keeps it too.
        """
        path = QPainterPath()
        path.moveTo(columns[0][0], columns[0][1])
        for x, y, _h, _b in columns[1:]:
            path.lineTo(x, y)
        path.lineTo(columns[-1][0], top)
        path.lineTo(columns[0][0], top)
        path.closeSubpath()
        return path

# -- ripple -----------------------------------------------------------------

RIPPLE_SOURCES = 3
RIPPLE_RINGS = 4
RIPPLE_PERIOD = (14.0, 26.0)     # seconds for a ring to cross its reach
RIPPLE_REACH = (0.55, 0.95)      # fraction of half the canvas diagonal

#: Ring thickness, as a fraction of its own radius. Started at 0.38, which
#: drew four crisp concentric circles per source and read as a dartboard.
#: Wide and soft is the point — it should look like something moved through
#: the page, not like a diagram.
RIPPLE_BAND = 0.72
RIPPLE_ALPHA_DARK = 0.20
RIPPLE_ALPHA_LIGHT = 0.38


@dataclass
class Source:
    """One ripple origin, in normalised units."""

    x: float
    y: float
    period: float
    phase: float
    reach: float
    color: int


class RippleEngine(_BufferedEngine):
    """Concentric rings expanding from a few sources and fading as they grow.

    Each ring is a radial gradient annulus rather than a stroked circle: a
    stroked one is line work, which is both expensive and far too crisp for
    something meant to sit behind a settings form.

    :meth:`geometry` yields ``(cx, cy, radius, fade)`` per ring, in pixels,
    with ``fade`` in 0..1 — 0 as the ring is born and as it dies at the
    edge of its reach, 1 halfway.
    """

    name = "ripple"

    def _configure(self, rng: random.Random) -> None:
        # The first three are the shipped anchors. The rest exist for the
        # density control and are placed on a jittered ring around the
        # middle: reusing the same three with a wider jitter puts two
        # sources close enough that their rings arrive together, which
        # reads as one source with a doubled amplitude rather than as two.
        anchors = ((0.24, 0.28), (0.76, 0.22), (0.5, 0.82))
        self.sources: List[Source] = []
        for i in range(_pool_size(RIPPLE_SOURCES)):
            if i < len(anchors):
                ax, ay = anchors[i]
            else:
                angle = 2 * math.pi * (i - len(anchors)) \
                    / max(1, _pool_size(RIPPLE_SOURCES) - len(anchors))
                ax = 0.5 + 0.36 * math.cos(angle)
                ay = 0.5 + 0.30 * math.sin(angle)
            self.sources.append(Source(
                x=ax + rng.uniform(-0.08, 0.08),
                y=ay + rng.uniform(-0.08, 0.08),
                period=rng.uniform(*RIPPLE_PERIOD),
                phase=rng.random(),
                reach=rng.uniform(*RIPPLE_REACH),
                color=i,
            ))

    def count(self) -> int:
        """How many ripple sources are painted right now."""
        return self.element_count(RIPPLE_SOURCES, len(self.sources))

    def geometry(self, width: int, height: int) -> Tuple[tuple, ...]:
        t = self.time
        half_diagonal = 0.5 * math.hypot(width, height)
        out = []
        for source in self.sources[:self.count()]:
            cx, cy = source.x * width, source.y * height
            # The size setting is the ripple's *wavelength*: the rings of one
            # source are evenly spaced across its reach, so stretching the
            # reach stretches the spacing between them by the same factor.
            reach = source.reach * half_diagonal * self.size
            for k in range(RIPPLE_RINGS):
                u = (t / source.period + source.phase
                     + k / RIPPLE_RINGS) % 1.0
                out.append((cx, cy, max(1.0, u * reach),
                            math.sin(math.pi * u)))
        return tuple(out)

    def _paint_field(self, painter: QPainter, width: int, height: int) -> None:
        peak = (RIPPLE_ALPHA_DARK if self.dark else RIPPLE_ALPHA_LIGHT) \
            * self.alpha_scale()
        colors = self.paint_colors
        inner = max(0.0, 1.0 - RIPPLE_BAND)
        for index, (cx, cy, radius, fade) in enumerate(
                self.geometry(width, height)):
            color = colors[(index // RIPPLE_RINGS) % len(colors)]
            gradient = QRadialGradient(cx, cy, radius)
            gradient.setColorAt(0.0, _with_alpha(color, 0.0))
            gradient.setColorAt(inner, _with_alpha(color, 0.0))
            gradient.setColorAt(1.0 - RIPPLE_BAND * 0.5,
                                _with_alpha(color, peak * fade))
            gradient.setColorAt(1.0, _with_alpha(color, 0.0))
            painter.setBrush(gradient)
            painter.drawEllipse(QPointF(cx, cy), radius, radius)


# -- drift ------------------------------------------------------------------

#: The particle pool. The whole pool is rolled once; how many of them are
#: actually painted depends on the canvas (see :data:`DRIFT_AREA_PER_PARTICLE`)
#: so that a small screen is not a snowstorm — but the pool itself never
#: changes, which keeps a resize from re-rolling the field.
DRIFT_POOL = 240
DRIFT_AREA_PER_PARTICLE = 9500     # pixels of canvas per particle
DRIFT_MIN_PARTICLES = 40

#: Three depth layers: ``(dot diameter in px, alpha, speed in canvas heights
#: per second)``. The parallax is the whole effect — one layer looks like
#: dust on the lens.
DRIFT_LAYERS = (
    (1.4, 0.35, 0.006),
    (2.4, 0.55, 0.011),
    (3.6, 0.80, 0.018),
)

#: Sideways sway, as a fraction of the canvas width, and its period.
DRIFT_SWAY = (0.01, 0.05)
DRIFT_SWAY_PERIOD = (18.0, 52.0)

#: Slow per-layer breathing, quantised into this many alpha steps so the pen
#: cache stays small — a pen rebuilt per particle per frame is the one way to
#: make this theme expensive.
DRIFT_TWINKLE = 0.25
DRIFT_TWINKLE_PERIOD = (9.0, 17.0)
DRIFT_ALPHA_STEPS = 8

#: On a light page the dots are darkened instead of brightened, or they are
#: invisible; they are also drawn a little harder, since a dark dot on white
#: has less room than a bright dot on black.
DRIFT_DARKEN_ON_LIGHT = 0.35
DRIFT_LIGHT_BOOST = 1.25

#: Blur, for the one theme that is not painted through the blur buffer. A dot
#: cannot be softened by shading it over fewer pixels — it *is* one pixel — so
#: above 1.0 each one gets a second, wider, dimmer pass around it: a halo.
#: The widening and the dimming are tied together so the dot's total light
#: stays roughly constant, which is what "the same star, out of focus" means.
DRIFT_HALO_SPREAD = 1.6      # extra diameter per unit of blur above 1
DRIFT_HALO_ALPHA = 0.34      # halo alpha as a share of the dot's own
#: A cap, because this is the one theme whose cost is *area* rather than a
#: fixed buffer: two hundred dots at maximum blur and maximum size would
#: otherwise light a quarter of the page and cost 3.2 ms a frame, which is
#: more than the whole module is allowed. Blurrier than this looks the same
#: anyway — the dot is already a soft disc by then.
DRIFT_HALO_MAX_PX = 14.0
#: Below this *resolution*, the dots stop being antialiased: they lose their
#: soft rim entirely, which is the only way left to make a 2 px dot
#: harder-edged, and is the only thing a resolution setting can mean for a
#: theme with no buffer to resolve. (It used to hang off the blur control,
#: which is exactly the conflation this pair of settings exists to undo.)
DRIFT_HARD_EDGE_RESOLUTION = 0.8

#: The ``random`` direction's wander: how far a speck slides sideways off its
#: own heading, as a fraction of the canvas, and over what period. Two
#: incommensurate sines per axis, so the path never closes and never
#: repeats — which is what "Brownian-ish" has to mean here, because every
#: position in this module is a pure function of the clock and an accumulated
#: random walk is not.
DRIFT_WANDER = (0.02, 0.07)
DRIFT_WANDER_PERIOD = (11.0, 37.0)


@dataclass
class Particle:
    """One drifting dot, in normalised units."""

    x: float
    y: float
    layer: int
    speed: float
    sway: float
    sway_rate: float
    sway_phase: float
    color: int
    #: Heading for the ``random`` direction, in radians. Unused by ``up``
    #: and ``down``, which share one heading between all of them.
    heading: float = 0.0
    wander: float = 0.0
    wander_rate: float = 0.0
    wander_phase: float = 0.0


class DriftEngine(AmbientEngine):
    """A slow starfield in three parallax layers.

    The one theme painted at full resolution, because dots have to be crisp
    to read as dots — and it is affordable exactly because a couple of
    hundred dots of 1-4 px touch almost no pixels (0.65 % of the page).
    Everything is batched into one ``drawPoints`` call per (colour, layer,
    alpha step) bucket with a cached pen; the per-call overhead of a pen
    change dominates this theme, not the pixels.

    :meth:`geometry` yields ``(x, y, diameter)`` per painted particle, in
    pixels.
    """

    name = "drift"

    def __init__(self, *args, **kwargs):
        self._pens: Dict[Tuple[int, int, int], QPen] = {}
        super().__init__(*args, **kwargs)

    def _configure(self, rng: random.Random) -> None:
        self.particles: List[Particle] = []

        def roll(i: int) -> Particle:
            layer = i % len(DRIFT_LAYERS)
            _, _, speed = DRIFT_LAYERS[layer]
            return Particle(
                x=rng.random(),
                y=rng.random(),
                layer=layer,
                speed=speed * rng.uniform(0.8, 1.25),
                sway=rng.uniform(*DRIFT_SWAY),
                sway_rate=2 * math.pi / rng.uniform(*DRIFT_SWAY_PERIOD),
                sway_phase=rng.uniform(0.0, 2 * math.pi),
                color=i,
            )

        # The draw order here is load-bearing and is the reason this reads
        # oddly. The shipped pool and the shipped twinkle came off this RNG
        # in this order, and the frame the tests hold this engine to is the
        # one those exact numbers produce. Everything density and direction
        # added is drawn *after* both, so the shipped starfield is untouched.
        for i in range(DRIFT_POOL):
            self.particles.append(roll(i))
        self.twinkle_rates = [
            2 * math.pi / rng.uniform(*DRIFT_TWINKLE_PERIOD)
            for _ in DRIFT_LAYERS]
        self.twinkle_phases = [rng.uniform(0.0, 2 * math.pi)
                               for _ in DRIFT_LAYERS]
        for i in range(DRIFT_POOL, _pool_size(DRIFT_POOL)):
            self.particles.append(roll(i))
        for particle in self.particles:
            particle.heading = rng.uniform(0.0, 2 * math.pi)
            particle.wander = rng.uniform(*DRIFT_WANDER)
            particle.wander_rate = 2 * math.pi / rng.uniform(
                *DRIFT_WANDER_PERIOD)
            particle.wander_phase = rng.uniform(0.0, 2 * math.pi)

    def _restyle(self) -> None:
        super()._restyle()
        self._pens = {}

    def _reblur(self) -> None:
        self._pens = {}

    def _reresolve(self) -> None:
        self._pens = {}

    def _resize(self) -> None:
        self._pens = {}

    def _tint(self, color: QColor) -> QColor:
        return QColor(color) if self.dark \
            else _mix(color, QColor(0, 0, 0), DRIFT_DARKEN_ON_LIGHT)

    def dot_size(self, layer: int) -> float:
        """Diameter of a ``layer`` dot, in pixels, under the size setting."""
        return DRIFT_LAYERS[layer][0] * self.size

    def halo_size(self, layer: int) -> float:
        """Diameter of the soft pass around a dot. Equal to the dot itself
        at blur 0, which is how the default frame stays exactly what it
        was."""
        dot = self.dot_size(layer)
        return min(DRIFT_HALO_MAX_PX,
                   dot * (1.0 + DRIFT_HALO_SPREAD * max(0.0, self.blur)))

    @property
    def work(self) -> float:
        """Density only. This is the one theme with no buffer, so the
        resolution setting costs it nothing and must not be allowed to
        spend its density budget."""
        return self.density

    def alpha_scale(self) -> float:
        """Untouched, unlike every buffered theme.

        The compensation on the base class exists because overlapping
        translucent *fields* pile up additively. A starfield does not have
        that problem — measured, its mean frame lightness moves from 0.076
        to 0.077 across the whole density range, because a couple of hundred
        dots light 0.65 % of the page and almost never land on each other.
        Dividing their alpha by three would not un-brighten anything; it
        would simply delete two thirds of the stars into the background.
        """
        return 1.0

    def count_for(self, width: int, height: int) -> int:
        """How many of the pool this canvas gets.

        Area-based, so a small window is not a snowstorm, and then scaled by
        the density setting — which is the only one of the two the user
        controls.
        """
        wanted = int(width) * int(height) // DRIFT_AREA_PER_PARTICLE
        wanted = _clamp_int(wanted, min(DRIFT_MIN_PARTICLES, DRIFT_POOL),
                            DRIFT_POOL)
        return self.element_count(wanted, len(self.particles))

    def geometry(self, width: int, height: int) -> Tuple[tuple, ...]:
        """``(x, y, diameter)`` per painted particle, in pixels.

        Three directions, one expression each, and all three are pure
        functions of the clock — no accumulated state, so ``set_time`` still
        jumps anywhere and two engines on the same seed still agree.

        ``up`` and ``down`` are the same shared vector with opposite signs.
        ``random`` gives every speck its own heading and adds a slow wander
        across it, which is a smooth wandering path rather than a straight
        line with a wobble; the headings are isotropic, so the field spreads
        and mixes instead of travelling.
        """
        t = self.time
        out = []
        random_walk = self.direction == "random"
        sign = 1.0 if self.direction == "down" else -1.0
        for particle in self.particles[:self.count_for(width, height)]:
            sway = particle.sway * math.sin(
                particle.sway_rate * t + particle.sway_phase)
            if random_walk:
                travel = particle.speed * t
                wander = particle.wander * math.sin(
                    particle.wander_rate * t + particle.wander_phase)
                x = (particle.x + math.cos(particle.heading) * travel
                     + sway + wander * math.sin(particle.heading))
                y = (particle.y + math.sin(particle.heading) * travel
                     - wander * math.cos(particle.heading))
            else:
                x = particle.x + sway
                # Wrapped: the field never runs out.
                y = particle.y + sign * particle.speed * t
            out.append((x % 1.0 * width, y % 1.0 * height,
                        self.dot_size(particle.layer)))
        return tuple(out)

    def _alpha_step(self, layer: int) -> int:
        """The quantised alpha for ``layer`` right now, as a 0..N-1 step."""
        _, alpha, _ = DRIFT_LAYERS[layer]
        breath = 1.0 - DRIFT_TWINKLE + DRIFT_TWINKLE * (
            0.5 + 0.5 * math.sin(self.twinkle_rates[layer] * self.time
                                 + self.twinkle_phases[layer]))
        if not self.dark:
            breath *= DRIFT_LIGHT_BOOST
        value = _clamp(alpha * breath, 0.0, 1.0)
        return _clamp_int(round(value * (DRIFT_ALPHA_STEPS - 1)),
                          0, DRIFT_ALPHA_STEPS - 1)

    def _pen(self, color_index: int, layer: int, step: int,
             halo: bool = False) -> QPen:
        key = (color_index, layer, step, halo)
        pen = self._pens.get(key)
        if pen is None:
            colors = self.paint_colors
            alpha = step / (DRIFT_ALPHA_STEPS - 1)
            if halo:
                alpha *= DRIFT_HALO_ALPHA
            pen = QPen(_with_alpha(colors[color_index % len(colors)], alpha))
            # A round-capped pen makes drawPoints draw filled circles, which
            # is how a batch of dots gets drawn in one call.
            pen.setWidthF(self.halo_size(layer) if halo
                          else self.dot_size(layer))
            pen.setCapStyle(Qt.RoundCap)
            self._pens[key] = pen
        return pen

    def paint(self, painter: QPainter, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            return
        painter.setRenderHint(QPainter.Antialiasing,
                              self.resolution > DRIFT_HARD_EDGE_RESOLUTION)
        n_colors = len(self.paint_colors)
        steps = [self._alpha_step(i) for i in range(len(DRIFT_LAYERS))]
        buckets: Dict[Tuple[int, int, int], List[QPointF]] = {}
        for particle, (x, y, _size) in zip(
                self.particles, self.geometry(width, height)):
            key = (particle.color % n_colors, particle.layer,
                   steps[particle.layer])
            buckets.setdefault(key, []).append(QPointF(x, y))
        # The halo goes down first, so the crisp core sits on top of it
        # rather than being washed out by it.
        if self.blur > 0.0:
            for key, points in buckets.items():
                painter.setPen(self._pen(*key, halo=True))
                painter.drawPoints(points)
        for key, points in buckets.items():
            painter.setPen(self._pen(*key))
            painter.drawPoints(points)


# -- bokeh ------------------------------------------------------------------
#
# What an epifluorescence field looks like off the focal plane, which is a
# state every user of this app has spent hours staring at. A point source out
# of focus does not become a Gaussian smudge: it becomes an image of the
# aperture — a disc, brighter at its rim than in its middle, with a hard-ish
# edge. That inversion is the whole reason bokeh is recognisable and is the
# whole reason this is not "blobs with different numbers": a blob is a
# Gaussian, brightest in the centre and gone by the edge.
#
# Two things follow and both are cheap:
#
# 1. *Focus varies per disc.* A field has depth, so some sources are nearly
#    in focus (small, tight, bright rim) and some are far out (large, flat,
#    faint). One radial gradient expresses both — see :meth:`_stops`.
# 2. *They overlap and add.* Additive compositing over a dark page is
#    literally correct here rather than merely convenient: two out-of-focus
#    emitters really do sum.

#: How many discs. Fewer than blobs on purpose: each one has to stay
#: readable *as a disc*, and past about a dozen the rims start crossing
#: often enough that the field reads as a mesh.
BOKEH_COUNT = 11

#: Disc radius, as a fraction of the short edge. The wide range is the depth
#: of field: the far ones are big and flat, the near ones small and tight.
BOKEH_RADIUS = (0.05, 0.30)

#: Focus, 0 = far out of focus (flat, no rim), 1 = nearly sharp (bright
#: narrow rim, dark middle). Rolled per disc and *correlated with radius* in
#: :meth:`_configure`, because a big sharp-rimmed disc is not a thing an
#: objective can produce.
BOKEH_FOCUS = (0.15, 0.95)

#: Where the rim sits, as a fraction of the radius, and how far the disc
#: fades out past it. A real aperture image has a hard edge; this keeps a
#: couple of per-cent of softness so it does not alias.
BOKEH_RIM = 0.88
BOKEH_EDGE = 0.99

#: How much brighter the rim is than the middle, at full focus. At focus 0
#: the two are equal and the disc is flat.
BOKEH_RIM_GAIN = 2.4

#: Drift and its period, plus the slow independent brightness breathing that
#: keeps the field from looking like a still photograph.
BOKEH_DRIFT = (0.02, 0.09)
BOKEH_DRIFT_PERIOD = (26.0, 80.0)
BOKEH_BREATH = 0.30
BOKEH_BREATH_PERIOD = (9.0, 23.0)

#: Peak alpha. Lower than blobs: there are rims here, and a rim carries far
#: more attention per unit of alpha than a gradient does.
BOKEH_ALPHA_DARK = 0.22
BOKEH_ALPHA_LIGHT = 0.34

#: A rim is an edge, so this theme wants more resolution than the diffuse
#: ones — but much less than the aurora, because a rim is one edge per disc
#: at a radius of tens of pixels, not a comb repeating every 36.
BOKEH_BUFFER_EDGE = 512


@dataclass
class Disc:
    """One out-of-focus point source, in normalised units."""

    x: float
    y: float
    drift_x: float
    drift_y: float
    rate_x: float
    rate_y: float
    phase_x: float
    phase_y: float
    radius: float
    focus: float
    breath_rate: float
    breath_phase: float
    color: int


class BokehEngine(_BufferedEngine):
    """Defocused points of light: flat discs with bright rims.

    :meth:`geometry` yields ``(cx, cy, radius, focus)`` per disc, in pixels,
    with ``focus`` in 0..1 — the same tuple the painter builds its gradients
    from, so a test that asserts on it is asserting on the frame.
    """

    name = "bokeh"
    base_edge = BOKEH_BUFFER_EDGE

    def _configure(self, rng: random.Random) -> None:
        cols, rows = 4, 3
        cells = list(range(cols * rows))
        rng.shuffle(cells)
        self.discs: List[Disc] = []
        for i in range(_pool_size(BOKEH_COUNT)):
            cell = cells[i % len(cells)]
            col, row = cell % cols, cell // cols
            lo, hi = BOKEH_RADIUS
            radius = rng.uniform(lo, hi)
            # Focus falls with size. An aperture image is large exactly
            # because it is far out of focus, so a big disc with a knife
            # edge on it is not a defocused anything — it is a ring.
            near = 1.0 - (radius - lo) / max(1e-6, hi - lo)
            flo, fhi = BOKEH_FOCUS
            focus = flo + (fhi - flo) * (0.35 + 0.65 * near) * rng.uniform(
                0.75, 1.0)
            self.discs.append(Disc(
                x=(col + 0.12 + 0.76 * rng.random()) / cols,
                y=(row + 0.12 + 0.76 * rng.random()) / rows,
                drift_x=rng.uniform(*BOKEH_DRIFT),
                drift_y=rng.uniform(*BOKEH_DRIFT),
                rate_x=2 * math.pi / rng.uniform(*BOKEH_DRIFT_PERIOD),
                rate_y=2 * math.pi / rng.uniform(*BOKEH_DRIFT_PERIOD),
                phase_x=rng.uniform(0.0, 2 * math.pi),
                phase_y=rng.uniform(0.0, 2 * math.pi),
                radius=radius,
                focus=_clamp(focus, 0.0, 1.0),
                breath_rate=2 * math.pi / rng.uniform(*BOKEH_BREATH_PERIOD),
                breath_phase=rng.uniform(0.0, 2 * math.pi),
                color=i,
            ))

    def count(self) -> int:
        """How many discs are painted right now."""
        return self.element_count(BOKEH_COUNT, len(self.discs))

    def geometry(self, width: int, height: int) -> Tuple[tuple, ...]:
        t = self.time
        short = min(width, height)
        out = []
        for disc in self.discs[:self.count()]:
            cx = (disc.x + disc.drift_x
                  * math.sin(disc.rate_x * t + disc.phase_x)) * width
            cy = (disc.y + disc.drift_y
                  * math.sin(disc.rate_y * t + disc.phase_y)) * height
            out.append((cx, cy, max(1.0, disc.radius * short * self.size),
                        disc.focus))
        return tuple(out)

    @staticmethod
    def _stops(focus: float, peak: float):
        """The radial profile of one defocused point, as gradient stops.

        At ``focus`` 0 it is a flat disc: middle and rim the same, a soft
        shoulder at the edge. At 1 the middle has dropped to a third and the
        rim is :data:`BOKEH_RIM_GAIN` times brighter than it — the classic
        doughnut an out-of-focus point makes through a clear aperture.
        """
        middle = peak * (1.0 - 0.62 * focus)
        rim = peak * (1.0 + (BOKEH_RIM_GAIN - 1.0) * focus)
        return ((0.0, middle * 0.92), (0.55, middle),
                (BOKEH_RIM, rim), (BOKEH_EDGE, rim * 0.35), (1.0, 0.0))

    def _paint_field(self, painter: QPainter, width: int, height: int) -> None:
        # A bokeh disc has an edge — that is what makes it a disc and not a
        # blob — and an edge in a buffer that is about to be stretched
        # fourfold upscales as a staircase without this. Same trade the
        # aurora makes for its fold, and the same ~0.03 ms.
        painter.setRenderHint(QPainter.Antialiasing, True)
        base = (BOKEH_ALPHA_DARK if self.dark else BOKEH_ALPHA_LIGHT) \
            * self.alpha_scale()
        colors = self.paint_colors
        t = self.time
        for disc, (cx, cy, radius, focus) in zip(
                self.discs, self.geometry(width, height)):
            breath = 1.0 - BOKEH_BREATH + BOKEH_BREATH * (
                0.5 + 0.5 * math.sin(disc.breath_rate * t + disc.breath_phase))
            color = colors[disc.color % len(colors)]
            gradient = QRadialGradient(cx, cy, radius)
            for stop, alpha in self._stops(focus, base * breath):
                gradient.setColorAt(stop, _with_alpha(color, alpha))
            painter.setBrush(gradient)
            painter.drawEllipse(QPointF(cx, cy), radius, radius)
        painter.setRenderHint(QPainter.Antialiasing, False)


# -- cells ------------------------------------------------------------------
#
# The other thing this app's users look at all day. A cell in a widefield
# image is three concentric statements, not one: a soft cytoplasmic body, a
# slightly brighter membrane where the edge is seen nearly edge-on, and a
# distinctly brighter nucleus sitting off-centre. Draw those three and the
# shape reads as a cell at any size; draw only the first and it is a blob.
#
# They are ellipses rather than circles, they are not all pointing the same
# way, and they turn as they drift — slowly, because this sits behind a
# settings form. The rotation is a painter transform per cell (nine of them a
# frame, in a 480x270 buffer), which measures free next to the two gradient
# fills each one already costs.

#: How many cells. Nine reads as a sparse field at 1080p; the density
#: control is there for anyone who wants a confluent one.
CELL_COUNT = 9

#: Body radius along the major axis, as a fraction of the short edge, and how
#: much shorter the minor axis is.
CELL_RADIUS = (0.07, 0.17)
CELL_FLATTEN = (0.55, 0.92)

#: Drift and turn. The turn is a rate in radians per second, signed, so half
#: of them go one way.
CELL_DRIFT = (0.03, 0.11)
CELL_DRIFT_PERIOD = (30.0, 85.0)
CELL_TURN = (0.010, 0.055)

#: The nucleus: radius as a share of the body's minor axis, how far off
#: centre it sits as a share of the major axis, and how much brighter it is.
CELL_NUCLEUS = (0.34, 0.52)
CELL_NUCLEUS_OFFSET = 0.28
CELL_NUCLEUS_GAIN = 1.9

#: The membrane: where the body's own gradient brightens again before it
#: fades, and by how much. Small — a membrane that reads as an outline turns
#: the field into clip art.
CELL_MEMBRANE = 0.86
CELL_MEMBRANE_GAIN = 1.45

CELL_ALPHA_DARK = 0.20
CELL_ALPHA_LIGHT = 0.32

#: Same reasoning as bokeh: there is a rim in here, so it wants more than a
#: pure gradient field does and much less than the aurora.
CELL_BUFFER_EDGE = 512


@dataclass
class Cell:
    """One drifting cell, in normalised units."""

    x: float
    y: float
    drift_x: float
    drift_y: float
    rate_x: float
    rate_y: float
    phase_x: float
    phase_y: float
    radius: float
    flatten: float
    angle: float
    turn: float
    nucleus: float
    nucleus_angle: float
    color: int


class CellsEngine(_BufferedEngine):
    """Cells drifting through the field, turning as they go.

    :meth:`geometry` yields ``(cx, cy, major, minor, angle)`` per cell, in
    pixels and radians.
    """

    name = "cells"
    base_edge = CELL_BUFFER_EDGE

    def _configure(self, rng: random.Random) -> None:
        cols, rows = 4, 3
        cells = list(range(cols * rows))
        rng.shuffle(cells)
        self.cells: List[Cell] = []
        for i in range(_pool_size(CELL_COUNT)):
            cell = cells[i % len(cells)]
            col, row = cell % cols, cell // cols
            self.cells.append(Cell(
                x=(col + 0.15 + 0.7 * rng.random()) / cols,
                y=(row + 0.15 + 0.7 * rng.random()) / rows,
                drift_x=rng.uniform(*CELL_DRIFT),
                drift_y=rng.uniform(*CELL_DRIFT),
                rate_x=2 * math.pi / rng.uniform(*CELL_DRIFT_PERIOD),
                rate_y=2 * math.pi / rng.uniform(*CELL_DRIFT_PERIOD),
                phase_x=rng.uniform(0.0, 2 * math.pi),
                phase_y=rng.uniform(0.0, 2 * math.pi),
                radius=rng.uniform(*CELL_RADIUS),
                flatten=rng.uniform(*CELL_FLATTEN),
                angle=rng.uniform(0.0, 2 * math.pi),
                turn=rng.choice((-1.0, 1.0)) * rng.uniform(*CELL_TURN),
                nucleus=rng.uniform(*CELL_NUCLEUS),
                nucleus_angle=rng.uniform(0.0, 2 * math.pi),
                color=i,
            ))

    def count(self) -> int:
        """How many cells are painted right now."""
        return self.element_count(CELL_COUNT, len(self.cells))

    def geometry(self, width: int, height: int) -> Tuple[tuple, ...]:
        t = self.time
        short = min(width, height)
        out = []
        for cell in self.cells[:self.count()]:
            cx = (cell.x + cell.drift_x
                  * math.sin(cell.rate_x * t + cell.phase_x)) * width
            cy = (cell.y + cell.drift_y
                  * math.sin(cell.rate_y * t + cell.phase_y)) * height
            major = max(1.0, cell.radius * short * self.size)
            out.append((cx, cy, major, major * cell.flatten,
                        cell.angle + cell.turn * t))
        return tuple(out)

    def _paint_field(self, painter: QPainter, width: int, height: int) -> None:
        # The membrane is an edge; see BokehEngine._paint_field.
        painter.setRenderHint(QPainter.Antialiasing, True)
        peak = (CELL_ALPHA_DARK if self.dark else CELL_ALPHA_LIGHT) \
            * self.alpha_scale()
        colors = self.paint_colors
        for cell, (cx, cy, major, minor, angle) in zip(
                self.cells, self.geometry(width, height)):
            color = colors[cell.color % len(colors)]
            painter.save()
            painter.translate(cx, cy)
            painter.rotate(math.degrees(angle))
            # Body plus membrane: one gradient, because the membrane is a
            # brightening of the body's own falloff and not a stroked
            # outline. A stroked one is line work — see the module docstring
            # for what that costs — and it also looks drawn rather than
            # imaged.
            body = QRadialGradient(0.0, 0.0, major)
            body.setColorAt(0.0, _with_alpha(color, peak * 0.55))
            body.setColorAt(0.62, _with_alpha(color, peak * 0.72))
            body.setColorAt(CELL_MEMBRANE,
                            _with_alpha(color, peak * CELL_MEMBRANE_GAIN))
            body.setColorAt(0.97, _with_alpha(color, peak * 0.30))
            body.setColorAt(1.0, _with_alpha(color, 0.0))
            painter.setBrush(body)
            # The ellipse is the circle the gradient was built for, squashed
            # on one axis — so the gradient squashes with it and the membrane
            # stays on the edge all the way round.
            painter.save()
            painter.scale(1.0, minor / major)
            painter.drawEllipse(QPointF(0.0, 0.0), major, major)
            painter.restore()

            offset = CELL_NUCLEUS_OFFSET * major
            nx = offset * math.cos(cell.nucleus_angle)
            ny = offset * math.sin(cell.nucleus_angle) * (minor / major)
            radius = max(1.0, cell.nucleus * minor)
            nucleus = QRadialGradient(nx, ny, radius)
            nucleus.setColorAt(0.0,
                               _with_alpha(color, peak * CELL_NUCLEUS_GAIN))
            nucleus.setColorAt(0.6,
                               _with_alpha(color, peak * CELL_NUCLEUS_GAIN
                                           * 0.6))
            nucleus.setColorAt(1.0, _with_alpha(color, 0.0))
            painter.setBrush(nucleus)
            painter.drawEllipse(QPointF(nx, ny), radius, radius)
            painter.restore()
        painter.setRenderHint(QPainter.Antialiasing, False)


_ENGINES = {
    "blobs": BlobsEngine,
    "aurora": AuroraEngine,
    "ripple": RippleEngine,
    "drift": DriftEngine,
    "bokeh": BokehEngine,
    "cells": CellsEngine,
}


def make_engine(theme: str, palette: str, background: Union[QColor, str],
                seed: Optional[int] = None,
                blur: float = DEFAULT_BLUR,
                speed: float = DEFAULT_SPEED,
                size: float = DEFAULT_SIZE,
                resolution: float = DEFAULT_RESOLUTION,
                density: float = DEFAULT_DENSITY,
                direction: str = DEFAULT_DRIFT_DIRECTION) -> AmbientEngine:
    """Build the engine for ``theme``/``palette``. Raises on unknown names.

    Everything after ``seed`` is a user control; the defaults are the shipped
    animation exactly.
    """
    _require_theme(theme)
    _require_palette(theme, palette)
    return _ENGINES[theme](palette_colors(theme, palette), background,
                           seed=seed, blur=blur, speed=speed, size=size,
                           resolution=resolution, density=density,
                           direction=direction)


class Motion(NamedTuple):
    """Every user control that shapes the animation, in one value.

    A named tuple rather than five arguments because the set grows: it went
    from three to five in one change, and every install site that had
    unpacked a plain tuple would have broken.
    """

    blur: float
    speed: float
    size: float
    resolution: float
    density: float
    direction: str


def preferred_motion() -> Motion:
    """The animation controls, from the user's preferences.

    Read here rather than passed in by every install site, for the same
    reason :func:`_theme_background` is: the two callers that build ambient
    widgets are a module screen and Home, and neither of them has any
    business knowing what the animation's knobs are called. Falls back to
    the shipped defaults if preferences cannot be read at all.
    """
    fallback = Motion(DEFAULT_BLUR, DEFAULT_SPEED, DEFAULT_SIZE,
                      DEFAULT_RESOLUTION, DEFAULT_DENSITY,
                      DEFAULT_DRIFT_DIRECTION)
    try:
        from ..preferences import (get_ambient_blur, get_ambient_density,
                                   get_ambient_drift_direction,
                                   get_ambient_resolution, get_ambient_size,
                                   get_ambient_speed)
        return Motion(get_ambient_blur(), get_ambient_speed(),
                      get_ambient_size(), get_ambient_resolution(),
                      get_ambient_density(), get_ambient_drift_direction())
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

#: Frames painted by every ambient backdrop in this process, ever.
#:
#: Process-wide and not per widget, because the claim "None costs nothing"
#: is about the *application*, and the widget it would be counted on is the
#: one that does not exist. A test selects None, drives a real screen, and
#: asserts this number does not move.
_TOTAL_FRAMES = 0


def total_frames_painted() -> int:
    """How many ambient frames this process has painted. For tests."""
    return _TOTAL_FRAMES


#: How long :func:`_retire_producer` waits for a shading thread to notice it
#: has been asked to stop, in seconds.
#:
#: One loop iteration is at most one shading pass, and the sleep between
#: passes wakes on the stop event rather than expiring — so a healthy thread
#: is gone in microseconds and this only bounds a pathological one. A thread
#: that has not stopped by then is a daemon that will exit on its own, and
#: waiting longer for it on the GUI thread would be a worse bug than the one
#: it is guarding.
#:
#: The wait is on the GUI thread, in ``hideEvent`` — i.e. on every tab switch
#: — and it does cost something when the machine is busy: **47 ms worst of
#: eight**, hiding a ``cells`` backdrop at 1080p while a Python worker runs,
#: which is one shading pass already in flight finishing. That is the trade
#: this whole change makes and it is strongly the right way round: a one-off
#: 47 ms on a tab switch against 18-33 ms on *every frame* beforehand.
PRODUCER_JOIN_S = 2.0


class _FrameProducer:
    """The shading thread: turns ``(engine, size)`` into finished frames.

    One thread per running backdrop, one frame deep. It calls
    :meth:`_BufferedEngine.shade` under ``engine_lock`` and publishes the
    result into a single slot; the GUI thread takes whatever is in that slot
    and blits it. Nothing here touches a ``QWidget`` — a widget painted off
    the GUI thread is undefined behaviour, and a ``QImage`` painted off it is
    supported and is the whole reason this works.

    **The clock is not moved here.** Stepping it costs two attribute stores,
    so there is nothing to gain by moving it and a contract to lose:
    :meth:`AmbientWidget.time` means seconds of animation, several tests
    assert the clock is exactly what was asked for, and a clock advanced from
    two threads is neither. The GUI thread advances it between frames (see
    :meth:`AmbientWidget._on_tick`, which takes the same lock without ever
    waiting on it) and this thread only ever *reads* it — so a frame remains
    a pure function of ``(seed, clock, size)`` even while it is shaded on
    another thread, and the animation keeps its real-time pace when frames
    are being dropped instead of slowing down with the thread.

    A plain :class:`threading.Thread` and not a ``QThread``: it owns no
    object with Qt thread affinity, it emits no signals, and
    :mod:`spacr.qt.bridge` is a standing record of what QThread lifetime
    costs when the thing on it does not need one.

    **What it does not do is speed the shading up.** It is the same Python
    under the same interpreter lock, so it takes the same 26 ms for ``cells``
    under load that the GUI thread took. What changes is *who waits*: the GUI
    thread stops shading, so its frame costs one ``drawImage`` (0.24-0.32 ms
    at every load measured) and the animation degrades by repeating a frame
    instead of by blocking the interface. That distinction is the fix.

    :param engine: a :class:`_BufferedEngine`. Unbuffered engines
        (``drift``) have no frame to hand over and keep the synchronous path.
    :param engine_lock: the widget's lock over the engine. Held across the
        shading pass here, and taken by every widget setter that mutates the
        engine, so a live Preferences change cannot land in the middle of a
        frame.
    :param fps: frame-rate cap, matching the widget's timer.
    :param size: ``(width, height)`` of the canvas.
    """

    def __init__(self, engine: "_BufferedEngine", engine_lock,
                 fps: int, size: Tuple[int, int]):
        self._engine = engine
        self._engine_lock = engine_lock
        self._interval = 1.0 / max(1, int(fps))
        #: Canvas size, written by the GUI thread and read by this one.
        #: A plain attribute holding an immutable tuple, deliberately: the
        #: assignment is a single bytecode and the reader either sees the old
        #: pair or the new one, never half of each, so a lock here would buy
        #: nothing but a chance for the GUI thread to wait on it.
        self.size: Tuple[int, int] = (int(size[0]), int(size[1]))
        #: Frames actually shaded. Under load this falls below the frame
        #: rate, which is the degradation this design chooses.
        self.frames_shaded = 0
        self._frame_lock = threading.Lock()
        self._frame: Optional[QImage] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- lifetime ------------------------------------------------------
    def start(self) -> None:
        """Begin shading. A second call is a no-op."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="spacr-ambient-shade", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Ask the thread to finish the frame it is on and exit, then wait
        up to :data:`PRODUCER_JOIN_S` for it.

        The thread reference is *kept*, not cleared, so :meth:`is_alive` goes
        on answering about the operating system's thread rather than about a
        variable this method set to ``None`` — otherwise "the backdrop stopped
        its thread" is a claim nothing can check. A producer is never
        restarted (:meth:`AmbientWidget._start_producer` builds a new one), so
        there is nothing to gain by forgetting it.
        """
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=PRODUCER_JOIN_S)

    def is_alive(self) -> bool:
        """Whether the shading thread is actually running."""
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def set_fps(self, fps: int) -> None:
        """Shade no faster than ``fps`` frames a second. Takes effect after
        the frame in flight."""
        self._interval = 1.0 / max(1, int(fps))

    # -- the frame slot ------------------------------------------------
    def publish(self, image: QImage) -> None:
        """Make ``image`` the frame the next paint will use."""
        with self._frame_lock:
            self._frame = image

    def latest(self) -> Optional[QImage]:
        """The newest published frame, or ``None``. **Never blocks.**

        ``None`` means "ask again next frame" and covers both cases the
        caller has to survive: nothing has been shaded yet, and the slot is
        being written this instant. Neither is worth waiting for — the
        caller already holds the previous frame and repeating it is a frame
        of animation, where waiting is a frozen interface.
        """
        if not self._frame_lock.acquire(blocking=False):
            return None
        try:
            return self._frame
        finally:
            self._frame_lock.release()

    # -- the loop ------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            width, height = self.size
            image = None
            if width > 0 and height > 0:
                with self._engine_lock:
                    image = self._engine.shade(width, height)
            if image is not None:
                self.publish(image)
                self.frames_shaded += 1
            remaining = self._interval - (time.monotonic() - started)
            # Sleeping on the stop event rather than time.sleep is what makes
            # stop() return immediately instead of at the end of the beat.
            #
            # A pass that overran the beat gets no sleep at all, which is how
            # this thread degrades: it keeps shading as fast as it can and the
            # GUI thread repeats whatever the last finished frame was. That
            # cannot make the machine worse, and the reason is arithmetic
            # rather than good intentions: the no-sleep branch is only reached
            # when one pass already took longer than the interval, so the rate
            # is 1/pass and therefore *below* the cap by construction. Driven
            # and counted at 1080p on ``cells``, cap 24: 23.5 passes a second
            # idle, 15.4 with one Python worker, 0.7 with three. The total
            # shading work is what the GUI thread used to do, on a thread
            # nobody is waiting for.
            self._stop.wait(remaining if remaining > 0 else 0.0)


def _retire_producer(box: List[Optional[_FrameProducer]]) -> None:
    """Stop and join whatever shading thread is in ``box``, and empty it.

    A module function over a one-element list rather than a method, because
    :attr:`QWidget.destroyed` is connected to it: a slot that captured the
    widget would run against a Python wrapper whose C++ half is already gone,
    which is the crash :mod:`spacr.qt.bridge` documents in another guise. The
    box holds nothing but the thread, so the closure is safe to outlive
    everything else.
    """
    producer = box[0] if box else None
    if box:
        box[0] = None
    if producer is not None:
        producer.stop()


class AmbientWidget(QWidget):
    """The live backdrop: paints an :class:`AmbientEngine` at a capped rate.

    Screen content sits in front of it, so it never takes focus, is
    transparent to mouse events, and lowers itself to the bottom of the
    sibling stacking order. It is fully opaque — it paints the page colour (or
    the theme wallpaper) itself and the animation on top — so the widget it
    covers has nothing to repaint underneath.

    :param parent: parent widget; :meth:`follow_parent` sizes it to that.
    :param theme: one of :data:`AMBIENT_THEMES`.
    :param palette: one of :func:`palettes_for` for that theme. A palette
        that exists but is not offered by the theme is downgraded to the
        theme's default (stale preferences must not break a screen); an
        unknown name raises.
    :param background: the flat colour under the animation; defaults to the
        current theme's page colour.
    :param backdrop: an image to paint under the animation instead of the
        flat colour — a path, a ``QPixmap``/``QImage``, or ``None``. Give it
        the Space/Cell wallpaper and the animation composites over the
        picture rather than replacing it.
    :param fps: frame-rate cap.
    :param seed: RNG seed, for a reproducible animation.
    :param blur: how much the finished picture is softened; ``None`` reads
        Preferences.
    :param speed: motion multiplier; ``None`` reads Preferences.
    :param size: element-size multiplier; ``None`` reads Preferences.
    :param resolution: how much detail is shaded, as a multiplier on the
        theme's own buffer; ``None`` reads Preferences.
    :param density: how many elements are drawn, as a multiplier on the
        theme's own count; ``None`` reads Preferences.
    :param direction: which way the starfield travels, one of
        :data:`DRIFT_DIRECTIONS`; ``None`` reads Preferences. Meaningless to
        the other themes, and kept anyway so switching away and back does
        not lose it.
    """

    def __init__(self, parent: Optional[QWidget] = None, *,
                 theme: str = DEFAULT_THEME,
                 palette: str = DEFAULT_PALETTE,
                 background: Union[QColor, str, None] = None,
                 backdrop=None,
                 fps: int = DEFAULT_FPS,
                 seed: Optional[int] = None,
                 blur: Optional[float] = None,
                 speed: Optional[float] = None,
                 size: Optional[float] = None,
                 resolution: Optional[float] = None,
                 density: Optional[float] = None,
                 direction: Optional[str] = None):
        super().__init__(parent)
        # -- the shading thread ---------------------------------------
        # First, before anything that could reach a setter: every one of
        # them takes the lock, so a construction order that reached one
        # early would raise AttributeError rather than race.
        #
        #: Guards every touch of the engine, because :class:`_FrameProducer`
        #: shades it on another thread while the GUI thread's setters change
        #: it. :meth:`paintEvent` never *waits* on this lock — see there.
        self._engine_lock = threading.RLock()
        #: One-element box holding the shading thread, so ``destroyed`` can
        #: retire it through a closure that captures no widget.
        self._producer_box: List[Optional[_FrameProducer]] = [None]
        #: The frame currently on screen. Held so a paint with nothing new
        #: to show can put it up again instead of waiting for one.
        self._last_frame: Optional[QImage] = None
        #: Paints that showed the previous frame again because the shading
        #: thread had not finished a new one. This counter *is* the
        #: "degrade rather than stutter" promise: it is what rises when the
        #: machine is busy, in place of the frame interval.
        self.repeated_frames = 0
        #: Clock time a tick could not apply because the shading thread had
        #: the engine, carried to the next tick. See :meth:`_on_tick`.
        self._pending_dt = 0.0
        box = self._producer_box
        self.destroyed.connect(lambda *_: _retire_producer(box))

        self._theme = _require_theme(theme)
        self._palette = coerce_palette(self._theme, palette)
        self._seed = seed
        # Unset means "whatever the user asked for in Preferences", so a
        # screen built after a settings change comes up already correct
        # instead of waiting for the next apply_ambient_preferences().
        asked = (blur, speed, size, resolution, density, direction)
        stored = preferred_motion() if None in asked else None
        self._blur = _clamp(stored.blur if blur is None else blur,
                            *BLUR_RANGE)
        self._speed = _clamp(stored.speed if speed is None else speed,
                             *SPEED_RANGE)
        self._size = _clamp(stored.size if size is None else size, *SIZE_RANGE)
        self._resolution = _clamp(
            stored.resolution if resolution is None else resolution,
            *RESOLUTION_RANGE)
        self._density = _clamp(
            stored.density if density is None else density, *DENSITY_RANGE)
        wanted = stored.direction if direction is None else direction
        self._direction = wanted if is_valid_drift_direction(wanted) \
            else DEFAULT_DRIFT_DIRECTION
        # Remember whether the caller *chose* the colour. If they did, a
        # later application palette change is theirs to react to; if they did
        # not, this widget follows the theme itself rather than leaving a
        # black rectangle on a white page.
        self._background_explicit = background is not None
        self._background = _as_color(background, _theme_background())
        self._backdrop: Optional[QPixmap] = _as_pixmap(backdrop)

        # Never in front of, never in the way of, the real content.
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setFocusPolicy(Qt.NoFocus)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self._engine = make_engine(self._theme, self._palette,
                                   self._background, seed=seed,
                                   blur=self._blur, speed=self._speed,
                                   size=self._size,
                                   resolution=self._resolution,
                                   density=self._density,
                                   direction=self._direction)

        self._animating = True
        #: Frames this backdrop has actually painted. The activity spinner
        #: carries the same counter for the same reason: "it costs nothing
        #: while it is off" is a claim about frames, and a test that reads a
        #: flag instead would pass just as happily on a timer that is
        #: running and drawing something invisible.
        self.frames_painted = 0
        self._fps = _clamp_int(fps, MIN_FPS, MAX_FPS)
        self._clock = QElapsedTimer()
        # One timer for the life of the widget. Switching theme swaps the
        # engine underneath it and never creates a second one.
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.CoarseTimer)
        self._timer.setInterval(max(1, 1000 // self._fps))
        self._timer.timeout.connect(self._on_tick)
        self._watched: Optional[QWidget] = None

    def focusInEvent(self, event) -> None:  # noqa: N802 (Qt override)
        """Reject even programmatic focus; this widget is decorative only."""
        event.ignore()
        self.clearFocus()

    # -- what is being painted ----------------------------------------
    @property
    def engine(self) -> AmbientEngine:
        """The live engine. Replaced wholesale by :meth:`set_theme`."""
        return self._engine

    def theme(self) -> str:
        return self._theme

    def palette_name(self) -> str:
        """The ambient palette's name.

        Not ``palette()`` — :class:`QWidget` already owns that name and it
        returns a ``QPalette``.
        """
        return self._palette

    def set_theme(self, name: str) -> None:
        """Switch animation, live. Raises :class:`ValueError` on an unknown
        name.

        The clock carries over and the old engine is dropped — there is no
        second timer and no second engine, so a user flipping through the
        menu cannot leave anything ticking behind them. If the current
        palette is not one this theme offers, it downgrades to the theme's
        default (see :func:`palettes_for` for why the lists differ).
        """
        name = _require_theme(name)
        if name == self._theme:
            return
        self._theme = name
        self._palette = coerce_palette(name, self._palette)
        self._rebuild_engine()

    def set_palette(self, name: str) -> None:
        """Switch colour set, live, keeping the motion exactly where it is.

        Raises :class:`ValueError` if the current theme does not offer
        ``name`` — an explicit request for a palette is not something to
        silently substitute.
        """
        name = _require_palette(self._theme, name)
        if name == self._palette:
            return
        self._palette = name
        self._mutate_engine(
            lambda: self._engine.set_colors(palette_colors(self._theme, name)))

    def _mutate_engine(self, change: Callable[[], None]) -> None:
        """Apply ``change`` to the engine with the shading thread locked out,
        re-shade once, and repaint.

        Every live control goes through here, because
        :func:`spacr.qt.preferences.apply_ambient_preferences` walks
        ``app.allWidgets()`` and calls eight of them on every backdrop that
        happens to be alive — a settings change is the one moment a widget is
        mutated *while* it is running, so it is the one the lock exists for.

        The re-shade is what keeps the promise every setter has always made:
        change it, and the next paint shows the change. Without it the next
        paint would blit the frame the shading thread finished *before* the
        change, and a caller that sets something and grabs the widget would
        get the old picture — which is a real regression and not a subtle
        one, because :meth:`set_background_color` flips the composition mode
        and a dark-shaded frame blitted onto a light page is inverted, not
        stale.

        Bounded and one-off: a single shading pass on the GUI thread (0.24 ms
        for ``blobs``, 1.5 ms for the aurora, idle) when somebody moves a
        slider, against zero per frame. It is deliberately *not* on the timer
        path — see :meth:`_on_tick`.
        """
        with self._engine_lock:
            change()
            self._republish()
        self.update()

    def _republish(self) -> None:
        """Re-shade the current clock and make it the frame on screen.

        Called with :attr:`_engine_lock` held. A no-op with no shading
        thread, where the next ``paintEvent`` shades synchronously anyway.
        """
        producer = self._producer_box[0]
        if producer is None:
            return
        width, height = producer.size
        fresh = self._engine.shade(width, height)
        if fresh is not None:
            producer.publish(fresh)
            self._last_frame = fresh

    def _rebuild_engine(self) -> None:
        """Replace the engine, preserving the clock. The old one is dropped
        on the next line and collected; it owns no Qt parent and no timer.

        The shading thread holds the *old* engine, so it is retired and a new
        one started around the swap. That join is bounded by one shading pass
        — worst measured 26 ms, for ``cells`` under a Python worker — and it
        happens when somebody picks a different animation from a menu, not
        per frame.
        """
        running = self._producer_box[0] is not None
        _retire_producer(self._producer_box)
        engine = make_engine(self._theme, self._palette, self._background,
                             seed=self._seed, blur=self._blur,
                             speed=self._speed, size=self._size,
                             resolution=self._resolution,
                             density=self._density,
                             direction=self._direction)
        with self._engine_lock:
            engine.set_time(self._engine.time)
            self._engine = engine
        # The old engine's last frame was drawn by the old engine; keeping it
        # would blit the theme the user just switched away from.
        self._last_frame = None
        if running:
            self._start_producer()
        self.update()

    # -- the user controls ---------------------------------------------
    def blur(self) -> float:
        """How much the picture is softened; 0.0 is the shipped animation."""
        return self._blur

    def set_blur(self, value: float) -> None:
        """Set the softening. Clamped to :data:`BLUR_RANGE`."""
        self._blur = _clamp(value, *BLUR_RANGE)
        self._mutate_engine(lambda: self._engine.set_blur(self._blur))

    def resolution(self) -> float:
        """How much detail is shaded; 1.0 is each theme's own buffer."""
        return self._resolution

    def set_resolution(self, value: float) -> None:
        """Set the detail multiplier. Clamped to :data:`RESOLUTION_RANGE`."""
        self._resolution = _clamp(value, *RESOLUTION_RANGE)
        self._mutate_engine(
            lambda: self._engine.set_resolution(self._resolution))

    def density(self) -> float:
        """How many elements are drawn; 1.0 is each theme's own count."""
        return self._density

    def set_density(self, value: float) -> None:
        """Set the element-count multiplier. Clamped to
        :data:`DENSITY_RANGE`."""
        self._density = _clamp(value, *DENSITY_RANGE)
        self._mutate_engine(lambda: self._engine.set_density(self._density))

    def direction(self) -> str:
        """Which way the starfield travels. Meaningless to the others, and
        kept anyway, so switching themes and back does not lose it."""
        return self._direction

    def set_direction(self, name: str) -> None:
        """Set the starfield direction. An unknown name is ignored."""
        if not is_valid_drift_direction(name):
            return
        self._direction = name
        self._mutate_engine(lambda: self._engine.set_direction(name))

    def speed(self) -> float:
        """The motion multiplier; 1.0 is the shipped animation."""
        return self._speed

    def set_speed(self, value: float) -> None:
        """Set the motion multiplier. Clamped to :data:`SPEED_RANGE`.

        Takes effect on the next step, so nothing already on screen moves.
        """
        self._speed = _clamp(value, *SPEED_RANGE)
        with self._engine_lock:
            self._engine.set_speed(self._speed)

    def size_scale(self) -> float:
        """The element-size multiplier; 1.0 is the shipped animation.

        Not ``size()`` — :class:`QWidget` already owns that name and it
        returns a ``QSize``.
        """
        return self._size

    def set_size_scale(self, value: float) -> None:
        """Set the element-size multiplier. Clamped to :data:`SIZE_RANGE`."""
        self._size = _clamp(value, *SIZE_RANGE)
        self._mutate_engine(lambda: self._engine.set_size(self._size))

    # -- appearance ----------------------------------------------------
    def background_color(self) -> QColor:
        return QColor(self._background)

    def set_background_color(self, color: Union[QColor, str]) -> None:
        """Set the flat fill under the animation.

        This is also what tells the engine whether it is painting on a dark
        or a light page, which picks additive versus multiply compositing —
        so it must be called on a live theme switch, or a dark-tuned frame
        ends up on a white page.
        """
        self._apply_background(color, explicit=True)

    def _apply_background(self, color, explicit: bool) -> None:
        """Re-derive the page colour, and re-shade *now* rather than next
        frame.

        The one setter that cannot let a published frame stand for a frame.
        A dark page composites additively and a light one multiplies (see the
        module docstring), so a frame shaded for dark and blitted onto light
        is not one frame stale, it is inverted — a white flash across the
        whole window on every theme switch. One shading pass on the GUI
        thread, at the moment somebody changes the application theme, buys
        that away.
        """
        self._background = _as_color(color, self._background)
        if explicit:
            self._background_explicit = True
        self._mutate_engine(
            lambda: self._engine.set_background(self._background))

    def backdrop(self) -> Optional[QPixmap]:
        """The image painted under the animation, or ``None``."""
        return self._backdrop

    def set_backdrop(self, source) -> None:
        """Paint ``source`` under the animation instead of the flat colour.

        The animation composites (adds on dark, multiplies on light), so a
        wallpaper handed in here shows *through* it rather than being
        replaced. ``None`` goes back to the flat fill.
        """
        self._backdrop = _as_pixmap(source)
        self.update()

    def changeEvent(self, event) -> None:
        """Follow a live theme switch when nobody else is going to.

        A host that passed its own ``background`` owns that colour and is
        expected to re-set it (that is what ``app_screen`` does, because it
        also has to re-resolve the wallpaper). A host that did not gets this
        for free instead of a stale dark rectangle on a white page.
        """
        super().changeEvent(event)
        if event.type() == QEvent.ApplicationPaletteChange \
                and not self._background_explicit:
            self._apply_background(_theme_background(), explicit=False)

    # -- run state -----------------------------------------------------
    def fps(self) -> int:
        return self._fps

    def set_fps(self, fps: int) -> None:
        """Cap the frame rate. Caps the shading thread with it, so a lowered
        cap actually reduces the work rather than just how much of it is
        shown."""
        self._fps = _clamp_int(fps, MIN_FPS, MAX_FPS)
        self._timer.setInterval(max(1, 1000 // self._fps))
        producer = self._producer_box[0]
        if producer is not None:
            producer.set_fps(self._fps)

    def is_running(self) -> bool:
        """True while the animation timer is ticking."""
        return self._timer.isActive()

    def is_animating(self) -> bool:
        """The requested state, whether or not the widget is on screen."""
        return self._animating

    def set_animating(self, on: bool) -> None:
        """Pause or resume without destroying anything.

        A paused widget keeps its last frame on screen and its engine in
        memory; it simply stops ticking. This is the "off" switch for the
        Preferences toggle when the user wants the colours but not the
        motion — turning the feature off entirely is the install site's job,
        not this widget's.
        """
        on = bool(on)
        if on == self._animating:
            return
        self._animating = on
        self._sync_run_state()

    def start(self) -> None:
        """Start ticking, and with it the shading thread (no-op if already
        running).

        The two are started and stopped *together*, and that is the whole
        lifetime rule: every gate that already stops the timer — hidden,
        another tab, a minimised window, the Preferences toggle — therefore
        stops the thread for free, and "off screen is 0 %" stays literally
        true rather than becoming "0 % of the GUI thread". It also means a
        widget the tests never show has no thread and takes the synchronous
        path, so nothing that drives ``advance_frame`` by hand goes
        non-deterministic.
        """
        if not self._timer.isActive():
            self._clock.restart()
            self._timer.start()
        self._start_producer()

    def stop(self) -> None:
        """Stop ticking and retire the shading thread. Costs exactly nothing
        while stopped — no timer, no thread, and no frame held in memory.

        Dropping the published frame matters because these screens stay
        built: a dozen module screens the user has visited would otherwise
        each keep a slot warm behind a tab nobody is on, which is 2 MiB apiece
        for the aurora. The next :meth:`start` shades a replacement before it
        starts the thread, so there is nothing to show for it.
        """
        self._timer.stop()
        _retire_producer(self._producer_box)
        self._last_frame = None

    def _start_producer(self) -> None:
        """Put the shading on its own thread, if this engine has a frame to
        hand over.

        ``drift`` is deliberately left out, and the number is the reason: it
        is the one engine with no buffer, it degrades the least of the six
        under a Python worker (0.528 -> 1.084 ms, 2.1x, against 48.7x for
        ``cells``), and threading it would mean publishing a full-resolution
        ARGB32 frame — 7.91 MiB a slot at 1080p against 126.6 KiB for
        ``blobs`` — to buy the smallest improvement on the list.

        The first frame is shaded here, synchronously, so the first paint
        after a show has a picture to blit rather than a flat rectangle.
        That is the pass the first ``paintEvent`` used to do anyway, moved a
        few microseconds earlier.
        """
        if self._producer_box[0] is not None:
            return
        engine = self._engine
        if not isinstance(engine, _BufferedEngine):
            return
        size = (max(0, self.width()), max(0, self.height()))
        producer = _FrameProducer(engine, self._engine_lock, self._fps, size)
        if size[0] > 0 and size[1] > 0:
            with self._engine_lock:
                first = engine.shade(*size)
            if first is not None:
                producer.publish(first)
        self._last_frame = None
        self._producer_box[0] = producer
        producer.start()

    def shading_thread_alive(self) -> bool:
        """Whether a shading thread is running for this backdrop.

        The CPU guarantee used to be a claim about a timer and is now also a
        claim about a thread, so it needs something to assert on: a backdrop
        behind a screen nobody is looking at must not be keeping a core warm.
        """
        producer = self._producer_box[0]
        return bool(producer is not None and producer.is_alive())

    def frames_shaded(self) -> int:
        """Frames the shading thread has finished for this backdrop.

        Below :attr:`frames_painted` under load, by design: the difference is
        :attr:`repeated_frames`.
        """
        producer = self._producer_box[0]
        return 0 if producer is None else producer.frames_shaded

    def _should_run(self) -> bool:
        if not self._animating or not self.isVisible():
            return False
        window = self.window()
        if window is not None and window.isMinimized():
            return False
        return True

    def _sync_run_state(self) -> None:
        """Start or stop the timer to match visibility. The CPU guarantee."""
        if self._should_run():
            self.start()
        else:
            self.stop()

    # -- Qt events -----------------------------------------------------
    def showEvent(self, event):
        super().showEvent(event)
        window = self.window()
        if window is not None and window is not self._watched:
            if self._watched is not None:
                self._watched.removeEventFilter(self)
            window.installEventFilter(self)
            self._watched = window
        self._sync_run_state()

    def hideEvent(self, event):
        """The whole performance story: a screen the user is not looking at
        costs nothing. Qt sends this to the children of a hidden parent too,
        so switching tabs stops the animation on the tab you left."""
        super().hideEvent(event)
        self.stop()

    def eventFilter(self, obj, event):
        """Follow the parent's size; pause when the window is minimised."""
        etype = event.type()
        if etype == QEvent.Resize and obj is self.parent():
            self.setGeometry(obj.rect())
        elif obj is self._watched and etype in (
                QEvent.WindowStateChange, QEvent.Hide, QEvent.Show):
            self._sync_run_state()
        return super().eventFilter(obj, event)

    def follow_parent(self) -> None:
        """Track the parent's geometry and sit below its other children."""
        parent = self.parent()
        if isinstance(parent, QWidget):
            parent.installEventFilter(self)
            self.setGeometry(parent.rect())
        self.lower()

    # -- animation -----------------------------------------------------
    def _on_tick(self) -> None:
        """One beat: step the clock, ask for a repaint. Never waits.

        The timer path deliberately does *not* go through
        :meth:`advance_frame`, and the difference is the whole point of the
        shading thread: this steps the clock (two attribute stores) and
        leaves the shading to the thread, where ``advance_frame`` shades
        synchronously for the callers that need the frame back immediately.

        The lock is taken **without blocking**. A shading pass takes 0.24 ms
        idle and up to 26 ms for ``cells`` under a Python worker, and waiting
        even that once a frame is the bug this change exists to remove — so a
        tick that finds the lock busy carries its ``dt`` into the next one
        rather than losing it. The clock therefore only ever moves *between*
        shading passes, which is what keeps a frame a pure function of the
        clock even though two threads are involved.
        """
        dt = self._clock.restart() / 1000.0
        self._pending_dt += min(MAX_DT, dt) if dt > 0 else 1.0 / self._fps
        if self._engine_lock.acquire(blocking=False):
            try:
                self._engine.advance(self._pending_dt)
                self._pending_dt = 0.0
            finally:
                self._engine_lock.release()
        self.update()

    def advance_frame(self, dt: float) -> None:
        """Step the animation by ``dt`` seconds and schedule a repaint.

        Called directly by the tests and by the tutorial recorder, so no
        caller ever waits on a real clock. Re-shades before it returns, so
        the next paint shows the frame that was asked for rather than
        whatever the shading thread last finished — the timer does not come
        through here for exactly that reason (:meth:`_on_tick`).
        """
        self._mutate_engine(lambda: self._engine.advance(dt))

    def time(self) -> float:
        """The animation clock, in seconds."""
        return self._engine.time

    def set_time(self, seconds: float) -> None:
        """Jump the animation clock and repaint."""
        self._mutate_engine(lambda: self._engine.set_time(seconds))

    # -- painting ------------------------------------------------------
    def _paint_base(self, painter: QPainter, rect: QRect) -> None:
        """Whatever sits *under* the animation: the flat colour, or the
        matching piece of the backdrop over it.

        Any part of ``rect`` the backdrop does not reach — a window wider
        than the wallpaper — still gets the flat colour, so the widget stays
        fully opaque and ``WA_OpaquePaintEvent`` remains honest.
        """
        pixmap = self._backdrop
        if pixmap is None:
            painter.fillRect(rect, self._background)
            return
        origin = self._backdrop_origin()
        covered = rect.intersected(
            QRect(origin.x(), origin.y(), pixmap.width(), pixmap.height()))
        if covered != rect:
            painter.fillRect(rect, self._background)
        if not covered.isEmpty():
            painter.drawPixmap(covered, pixmap,
                               covered.translated(-origin.x(), -origin.y()))

    def _backdrop_origin(self) -> QPoint:
        """Top-left of the backdrop in this widget's own coordinates.

        The window paints its wallpaper centred on itself
        (``background-position: center center`` in the QSS, which does not
        repeat), and this has to land on exactly the same pixels or the
        picture visibly jumps at the widget's edge.
        """
        pixmap = self._backdrop
        window = self.window()   # never None; the widget itself if top level
        x = (window.width() - pixmap.width()) // 2
        y = (window.height() - pixmap.height()) // 2
        offset = self.mapTo(window, QPoint(0, 0))
        return QPoint(x - offset.x(), y - offset.y())

    def paintEvent(self, event):
        """Put the page down, then the newest frame the shading thread has.

        **This method never waits for anything.** That is the requirement the
        whole change exists to meet — "a frame that is not ready is a
        repeated frame, never a blocked GUI thread" — and it is met the only
        way it can be: :meth:`_FrameProducer.latest` refuses the slot rather
        than blocking on it, the widget keeps a reference to the frame it is
        already showing, and a paint with nothing new blits that one again
        and counts it in :attr:`repeated_frames`.

        With a shading thread the engine lock is never *waited* on here, and
        with no shading thread it cannot be contended — the one path that
        takes it is the case where nothing has been published yet (a widget
        shown at zero size and resized in the same tick), and it takes it
        without blocking, settling for the flat page if the thread is
        mid-frame.

        Repainting is not only the timer's doing: every console line the
        window paints over a translucent surface exposes this widget and Qt
        asks it for a whole frame. Measured on a real X server with the real
        stylesheet, **one ambient repaint per console line** — 0.99 of them,
        with the animation timer stopped. That is the fps cap being bypassed
        entirely, and it is why the cost of a repaint matters more than the
        cap suggests: shading it cost 1.7 ms idle and 22 ms under a Python
        worker, and blitting an already-shaded frame costs 0.24-0.32 ms at
        every load measured.
        """
        global _TOTAL_FRAMES
        self.frames_painted += 1
        _TOTAL_FRAMES += 1
        painter = QPainter(self)
        rect = self.rect()
        self._paint_base(painter, rect)
        width, height = rect.width(), rect.height()

        producer = self._producer_box[0]
        if producer is None:
            self._engine.paint(painter, width, height)
            return

        producer.size = (width, height)
        fresh = producer.latest()
        if fresh is not None and fresh is not self._last_frame:
            self._last_frame = fresh
        else:
            self.repeated_frames += 1
        if self._last_frame is not None:
            self._engine.blit(painter, self._last_frame, width, height)
        elif self._engine_lock.acquire(blocking=False):
            try:
                self._engine.paint(painter, width, height)
            finally:
                self._engine_lock.release()


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

def install_ambient(host: QWidget, layout=None, *,
                    theme: str = DEFAULT_THEME,
                    palette: str = DEFAULT_PALETTE,
                    backdrop=None, **kwargs) -> AmbientWidget:
    """Put a live ambient backdrop behind ``host``.

    The widget becomes a child of ``host``, tracks its geometry, and is
    lowered to the bottom of the sibling stacking order so every screen
    widget paints in front of it. It takes no focus and no mouse events, and
    it does not tick until ``host`` is actually on screen.

    Note that a backdrop is only as visible as its siblings are transparent:
    under dark and light every container is an opaque page colour, and an
    animation behind them reaches the eye through nothing but the few pixels
    of layout spacing. The caller is responsible for clearing those surfaces
    first — see ``AppScreen._clear_page_surfaces``.

    :param host: the screen the animation sits behind.
    :param layout: accepted for signature compatibility with
        :func:`spacr.qt.widgets.dna_rain.install_dna_rain`, which appends its
        settings bar to it. The ambient backdrop has no on-screen controls —
        it is configured in Preferences — so nothing is added here, and the
        two installers stay interchangeable at a call site.
    :param theme: one of :data:`AMBIENT_THEMES`.
    :param palette: one of :func:`palettes_for` for that theme.
    :param backdrop: wallpaper to composite over; see
        :meth:`AmbientWidget.set_backdrop`.
    :param kwargs: forwarded to :class:`AmbientWidget` (``background``,
        ``fps``, ``seed``, ``blur``, ``speed``, ``size``, ``resolution``,
        ``density``, ``direction``). Everything from ``blur`` on defaults to
        the user's preferences, so a caller that does not care about them
        should not pass them.
    :returns: the widget, already shown and lowered.
    """
    widget = AmbientWidget(host, theme=theme, palette=palette,
                           backdrop=backdrop, **kwargs)
    widget.follow_parent()
    widget.show()
    return widget
