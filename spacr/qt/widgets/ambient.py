"""Ambient animated backdrop — soft motion behind every module screen.

The sequencing screen has its own backdrop (:mod:`spacr.qt.widgets.dna_rain`,
the ATGC cascade). This is the one for *everything else*: a slow, diffuse
animation that sits behind the settings form and the console, takes no focus
and no mouse events, and can be switched off entirely in Preferences.

Four themes, chosen so each reads as a different kind of movement rather than
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
    bigger, brighter, faster ones in front. The one crisp theme.

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
2. *The three soft themes are painted into a small reusable QImage and scaled
   up*, never at full resolution. A diffuse gradient has no detail to lose,
   and the buffer's long edge is capped at :data:`BUFFER_MAX_EDGE` px, so the
   gradient shading is done over ~37 000 pixels instead of ~2 000 000. The one
   allocation happens on resize, never per frame.

At 1920x1080, offscreen raster, 120 frames each, including the full-screen
background fill every frame, best of five interleaved runs (the machine is
shared, so a run now and a run in ten minutes are not comparable — the old
and the new engine are loaded into one process and measured alternately):

=========  ========  =========  ===================  ==================
 theme      dark      light      share of one core    at 60 fps
=========  ========  =========  ===================  ==================
 blobs      1.21 ms   1.70 ms    2.9 % / 4.1 %        7.3 % / 10.2 %
 aurora     1.40 ms   1.96 ms    3.4 % / 4.7 %        8.4 % / 11.8 %
 ripple     1.31 ms   1.80 ms    3.1 % / 4.3 %        7.9 % / 10.8 %
 drift      0.66 ms   0.66 ms    1.6 %                4.0 %
=========  ========  =========  ===================  ==================

and 0 % off screen. The middle column is this module's own 24 fps cadence,
which is what it actually costs; the last one is at 60 fps, for comparison
with the DNA rain's documented 0.53 ms and 3.2 %.

Light costs more because multiply is a slower blend than addition.

The aurora rewrite is the one theme that moved: 0.96 -> 1.40 ms dark and
1.52 -> 1.96 light, measured against the old curtains in the same process on
the same frames. It buys ray striations, three travelling wave trains, a
surge running along the arc on its own schedule, and the altitude-ordered
colour ramp. It costs 0.44 ms, which makes it the dearest of the four by
about 0.1 ms rather than the cheapest — a swap of places inside the range
this module already documents, not a new order of magnitude.

Most of what a buffered theme costs is not its artwork at all: clearing the
buffer, blitting it up to 1920x1080 and filling the page underneath is
0.93 ms of every frame in this table, and that number is bound by the
destination pixels, so it does not care what was drawn into the source. The
aurora's own drawing is 0.47 ms of its 1.40.

The three user controls move the cost, deliberately and in the useful
direction — a machine that cannot afford the backdrop can be told to soften
it, and softening it is *cheaper* rather than dearer:

===================  =======  ========  ========  =======
 setting              blobs    aurora    ripple    drift
===================  =======  ========  ========  =======
 default              1.32     1.58      1.42      0.65
 blur 25 %  (sharp)   1.48     1.79      1.51      0.56
 blur 300 % (soft)    1.10     1.28      1.26      1.08
 size 25 %            1.13     1.57      1.23      0.34
 size 250 %           1.68     1.88      1.77      0.73
===================  =======  ========  ========  =======

Blur runs backwards to intuition and forwards for cost, because it *is* the
buffer resolution: softer means shading fewer pixels and stretching them
further. Only ``drift`` inverts that, being the one theme with no buffer —
its blur is a second, wider pass per dot, which is why it has a cap
(:data:`DRIFT_HALO_MAX_PX`).

The design was picked on measurements, not taste. For blobs at 1920x1080:
full-resolution gradients 2.18 ms, buffered-and-upscaled 1.16 ms,
pre-rendered sprite blits 3.28 ms — the sprite version is the slow one
because a bilinear-sampled translucent blit costs more per pixel than
shading the gradient does.

Two candidate themes were cut on the same numbers rather than shipped slow:
a mesh lattice (3.2–8.1 ms) and contour polylines (10.8 ms). Both are crisp
full-screen line work, and in Qt's raster engine a translucent antialiased
line costs an order of magnitude more than the same pixels as a gradient — a
vertical line is a thousand one-pixel spans. Painting them into the small
buffer instead makes them cheap and also makes them not lines any more.
The rule that fell out: *soft is cheap, crisp is expensive*, and ``drift``
is crisp only because a couple of hundred dots light 0.65 % of the page.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

from PySide6.QtCore import (QElapsedTimer, QEvent, QPoint, QPointF, QRect,
                            QRectF, Qt, QTimer)
from PySide6.QtGui import (QBrush, QColor, QImage, QLinearGradient, QPainter,
                           QPainterPath, QPen, QPixmap, QRadialGradient,
                           QTransform)
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..theme import palette_for, relative_luminance

__all__ = [
    "AMBIENT_THEMES", "DEFAULT_THEME", "DEFAULT_PALETTE", "PALETTE_SETS",
    "AmbientWidget", "install_ambient", "theme_label", "theme_note",
    "palettes_for", "palette_label", "palette_note", "palette_colors",
    "default_palette_for", "is_valid_theme", "is_valid_palette",
    "BLUR_RANGE", "SPEED_RANGE", "SIZE_RANGE",
    "DEFAULT_BLUR", "DEFAULT_SPEED", "DEFAULT_SIZE", "preferred_motion",
]


# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------

#: Every theme, in the order a menu should list them.
AMBIENT_THEMES: Tuple[str, ...] = ("blobs", "aurora", "ripple", "drift")

#: What the feature was asked for, so it is what you get by default.
DEFAULT_THEME = "blobs"

#: spaCR's own colours, likewise.
DEFAULT_PALETTE = "spacr"

_THEME_LABELS = {
    "blobs": "Blobs",
    "aurora": "Aurora",
    "ripple": "Ripples",
    "drift": "Starfield",
}

_THEME_NOTES = {
    "blobs": ("Soft colour blobs, large and small, drifting and slowly "
              "changing size."),
    "aurora": ("Folded curtains of vertical rays, rippling along their own "
               "length the way the northern lights do."),
    "ripple": "Rings spreading out from a few points and fading as they grow.",
    "drift": "A slow starfield in three layers of depth.",
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
_THEME_PALETTES: Dict[str, Tuple[str, ...]] = {
    "blobs": ("spacr", "ember", "ocean", "pastel", "mono", "okabe",
              "borealis"),
    "aurora": ("spacr", "ember", "ocean", "pastel", "mono", "okabe",
               "borealis"),
    "ripple": ("spacr", "ember", "ocean", "mono", "okabe"),
    "drift": ("spacr", "ember", "ocean", "mono", "okabe", "borealis"),
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

#: Longest edge of the low-resolution buffer the soft themes paint into.
#: 256 px upscales to 1920 with no visible artefact (the content is gradients)
#: and keeps the gradient shading at ~2 % of the pixels. Raising it to 320
#: measured 1.27 ms against 1.16 ms and looked identical.
#:
#: This is also the *blur* control: the buffer is upscaled with bilinear
#: filtering, so shading the same picture over fewer pixels and stretching it
#: further is a blur — and a free one, which a per-frame Gaussian is not.
#: :data:`BLUR_RANGE` divides this edge. See :meth:`_BufferedEngine.blur_edge`.
BUFFER_MAX_EDGE = 256

#: Hard limits on the derived buffer edge. The low end is where bilinear
#: upscaling starts to show its interpolation lattice rather than a blur; the
#: high end is a cost ceiling — shading is per buffer pixel, so 384 already
#: costs about twice what 256 does.
BUFFER_MIN_EDGE = 96
BUFFER_EDGE_CEILING = 384


# ---------------------------------------------------------------------------
# The three user controls: blur, speed and size
# ---------------------------------------------------------------------------
# All three are *multipliers on what the theme already does*, never absolute
# pixels or seconds. 1.0 is the shipped animation, exactly — every engine is
# written so that multiplying by 1.0 is the identity, and the tests assert
# the default frame is byte-for-byte the frame from before these existed.
# A multiplier is also the only formulation that means the same thing in all
# four themes: "twice as big" is meaningful for a blob radius, a curtain, a
# ripple wavelength and a 2 px star, where "40 px" is meaningful for none of
# them.

#: How soft the shapes are. Above 1.0 the buffer shrinks and the upscale
#: stretches further; below 1.0 it grows and the picture sharpens (and costs
#: more, which is the trade the user is making).
BLUR_RANGE = (0.25, 3.0)
DEFAULT_BLUR = 1.0

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


def _theme_background() -> QColor:
    """The current theme's flat page colour, or the dark one if unavailable."""
    try:
        from ..preferences import resolve_effective_theme
        return QColor(palette_for(resolve_effective_theme())["bg"])
    except Exception:
        return QColor(palette_for("dark")["bg"])


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
                 size: float = DEFAULT_SIZE):
        self.seed = seed
        self.time = 0.0
        self.frames = 0
        # Before ``_configure``: a subclass may size something from them.
        self.blur = _clamp(blur, *BLUR_RANGE)
        self.speed = _clamp(speed, *SPEED_RANGE)
        self.size = _clamp(size, *SIZE_RANGE)
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

    # -- the three user controls ---------------------------------------
    def set_blur(self, value: float) -> None:
        """How soft the shapes are, as a multiplier. 1.0 is as shipped.

        Clamped to :data:`BLUR_RANGE`. Engines that cache anything sized by
        it drop that cache here, never per frame.
        """
        value = _clamp(value, *BLUR_RANGE)
        if value == self.blur:
            return
        self.blur = value
        self._reblur()

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

    def __init__(self, *args, **kwargs):
        self._buffer: Optional[QImage] = None
        super().__init__(*args, **kwargs)

    def _reblur(self) -> None:
        """A new blur means a new buffer size — drop the old one now rather
        than leaving the next paint to notice."""
        self._buffer = None

    def blur_edge(self) -> int:
        """Longest buffer edge under the current blur setting.

        The whole blur implementation: the buffer is upscaled to the canvas
        with bilinear filtering, so halving its resolution doubles how far
        every shaded pixel is stretched — a real blur, at *negative* cost.
        A per-frame Gaussian over 2 000 000 pixels would be an order of
        magnitude more expensive than everything else this module does put
        together.
        """
        return _clamp_int(round(BUFFER_MAX_EDGE / self.blur),
                          BUFFER_MIN_EDGE, BUFFER_EDGE_CEILING)

    def buffer_size(self, width: int, height: int) -> Tuple[int, int]:
        """Buffer dimensions for a ``width`` x ``height`` canvas."""
        longest = max(int(width), int(height))
        scale = max(1, int(math.ceil(longest / self.blur_edge())))
        return (max(1, int(width) // scale), max(1, int(height) // scale))

    def _ensure_buffer(self, width: int, height: int) -> QImage:
        bw, bh = self.buffer_size(width, height)
        buf = self._buffer
        if buf is None or buf.width() != bw or buf.height() != bh:
            buf = QImage(bw, bh, QImage.Format_RGB32)
            self._buffer = buf
        return buf

    def paint(self, painter: QPainter, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            return
        buf = self._ensure_buffer(width, height)
        inner = QPainter(buf)
        inner.fillRect(buf.rect(), self.identity)
        inner.setCompositionMode(self.mode)
        inner.setPen(Qt.NoPen)
        self._paint_field(inner, buf.width(), buf.height())
        inner.end()

        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.setCompositionMode(self.mode)
        painter.drawImage(QRect(0, 0, int(width), int(height)), buf)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

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
        for i in range(BLOB_COUNT):
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

    def geometry(self, width: int, height: int) -> Tuple[tuple, ...]:
        t = self.time
        short = min(width, height)
        out = []
        for blob in self.blobs:
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
        peak = BLOB_ALPHA_DARK if self.dark else BLOB_ALPHA_LIGHT
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
AURORA_TILE_RAYS = ((0.17, 0.115, 1.00), (0.49, 0.085, 0.86),
                    (0.80, 0.100, 0.94))
AURORA_TILE_FLOOR = 0.58

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
#: is 36; the rest of the headroom is for a window being resized, which
#: changes the pixel size the tiles are built at.
AURORA_TILE_CACHE = 96

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

    def __init__(self, *args, **kwargs):
        self._tiles: Dict[Tuple[int, int], QImage] = {}
        self._surges: Dict[int, QImage] = {}
        self._pulse_mask: Optional[QImage] = None
        super().__init__(*args, **kwargs)

    def _configure(self, rng: random.Random) -> None:
        self.curtains: List[Curtain] = []
        for i in range(AURORA_CURTAINS):
            base = AURORA_BASE[i % len(AURORA_BASE)]
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
        for curtain in self.curtains:
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
        key = (curtain.depth, step, width, height)
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
        peak = AURORA_ALPHA_DARK if self.dark else AURORA_ALPHA_LIGHT
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
        for index, curtain in enumerate(self.curtains):
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
        anchors = ((0.24, 0.28), (0.76, 0.22), (0.5, 0.82))
        self.sources: List[Source] = []
        for i in range(RIPPLE_SOURCES):
            ax, ay = anchors[i % len(anchors)]
            self.sources.append(Source(
                x=ax + rng.uniform(-0.08, 0.08),
                y=ay + rng.uniform(-0.08, 0.08),
                period=rng.uniform(*RIPPLE_PERIOD),
                phase=rng.random(),
                reach=rng.uniform(*RIPPLE_REACH),
                color=i,
            ))

    def geometry(self, width: int, height: int) -> Tuple[tuple, ...]:
        t = self.time
        half_diagonal = 0.5 * math.hypot(width, height)
        out = []
        for source in self.sources:
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
        peak = RIPPLE_ALPHA_DARK if self.dark else RIPPLE_ALPHA_LIGHT
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
#: Below this, antialiasing goes off: the dots stop having a soft rim at all,
#: which is the only way left to make a 2 px dot harder-edged.
DRIFT_HARD_EDGE_BLUR = 0.8


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
        for i in range(DRIFT_POOL):
            layer = i % len(DRIFT_LAYERS)
            _, _, speed = DRIFT_LAYERS[layer]
            self.particles.append(Particle(
                x=rng.random(),
                y=rng.random(),
                layer=layer,
                speed=speed * rng.uniform(0.8, 1.25),
                sway=rng.uniform(*DRIFT_SWAY),
                sway_rate=2 * math.pi / rng.uniform(*DRIFT_SWAY_PERIOD),
                sway_phase=rng.uniform(0.0, 2 * math.pi),
                color=i,
            ))
        self.twinkle_rates = [
            2 * math.pi / rng.uniform(*DRIFT_TWINKLE_PERIOD)
            for _ in DRIFT_LAYERS]
        self.twinkle_phases = [rng.uniform(0.0, 2 * math.pi)
                               for _ in DRIFT_LAYERS]

    def _restyle(self) -> None:
        super()._restyle()
        self._pens = {}

    def _reblur(self) -> None:
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
        when there is no blur to apply, which is how the default frame stays
        exactly what it was."""
        dot = self.dot_size(layer)
        return min(DRIFT_HALO_MAX_PX,
                   dot * (1.0 + DRIFT_HALO_SPREAD
                          * max(0.0, self.blur - 1.0)))

    def count_for(self, width: int, height: int) -> int:
        """How many of the pool this canvas gets."""
        wanted = int(width) * int(height) // DRIFT_AREA_PER_PARTICLE
        return _clamp_int(wanted, min(DRIFT_MIN_PARTICLES, DRIFT_POOL),
                          DRIFT_POOL)

    def geometry(self, width: int, height: int) -> Tuple[tuple, ...]:
        t = self.time
        out = []
        for particle in self.particles[:self.count_for(width, height)]:
            x = (particle.x + particle.sway
                 * math.sin(particle.sway_rate * t + particle.sway_phase))
            # Upward, and wrapped: the field never runs out.
            y = (particle.y - particle.speed * t) % 1.0
            out.append((x % 1.0 * width, y * height,
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
                              self.blur > DRIFT_HARD_EDGE_BLUR)
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
        if self.blur > 1.0:
            for key, points in buckets.items():
                painter.setPen(self._pen(*key, halo=True))
                painter.drawPoints(points)
        for key, points in buckets.items():
            painter.setPen(self._pen(*key))
            painter.drawPoints(points)


_ENGINES = {
    "blobs": BlobsEngine,
    "aurora": AuroraEngine,
    "ripple": RippleEngine,
    "drift": DriftEngine,
}


def make_engine(theme: str, palette: str, background: Union[QColor, str],
                seed: Optional[int] = None,
                blur: float = DEFAULT_BLUR,
                speed: float = DEFAULT_SPEED,
                size: float = DEFAULT_SIZE) -> AmbientEngine:
    """Build the engine for ``theme``/``palette``. Raises on unknown names.

    ``blur``/``speed``/``size`` are the user's three multipliers; the
    defaults are the shipped animation exactly.
    """
    _require_theme(theme)
    _require_palette(theme, palette)
    return _ENGINES[theme](palette_colors(theme, palette), background,
                           seed=seed, blur=blur, speed=speed, size=size)


def preferred_motion() -> Tuple[float, float, float]:
    """``(blur, speed, size)`` from the user's preferences.

    Read here rather than passed in by every install site, for the same
    reason :func:`_theme_background` is: the two callers that build ambient
    widgets are a module screen and Home, and neither of them has any
    business knowing what the animation's knobs are called. Falls back to
    the shipped defaults if preferences cannot be read at all.
    """
    try:
        from ..preferences import (get_ambient_blur, get_ambient_size,
                                   get_ambient_speed)
        return (get_ambient_blur(), get_ambient_speed(), get_ambient_size())
    except Exception:
        return (DEFAULT_BLUR, DEFAULT_SPEED, DEFAULT_SIZE)


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

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
                 size: Optional[float] = None):
        super().__init__(parent)
        self._theme = _require_theme(theme)
        self._palette = coerce_palette(self._theme, palette)
        self._seed = seed
        # Unset means "whatever the user asked for in Preferences", so a
        # screen built after a settings change comes up already correct
        # instead of waiting for the next apply_ambient_preferences().
        stored = preferred_motion() if None in (blur, speed, size) else None
        self._blur = _clamp(stored[0] if blur is None else blur, *BLUR_RANGE)
        self._speed = _clamp(stored[1] if speed is None else speed,
                             *SPEED_RANGE)
        self._size = _clamp(stored[2] if size is None else size, *SIZE_RANGE)
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
                                   size=self._size)

        self._animating = True
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
        self._engine.set_colors(palette_colors(self._theme, name))
        self.update()

    def _rebuild_engine(self) -> None:
        """Replace the engine, preserving the clock. The old one is dropped
        on the next line and collected; it owns no Qt parent and no timer."""
        engine = make_engine(self._theme, self._palette, self._background,
                             seed=self._seed, blur=self._blur,
                             speed=self._speed, size=self._size)
        engine.set_time(self._engine.time)
        self._engine = engine
        self.update()

    # -- blur, speed and size ------------------------------------------
    def blur(self) -> float:
        """How soft the shapes are; 1.0 is the shipped animation."""
        return self._blur

    def set_blur(self, value: float) -> None:
        """Set the softness multiplier. Clamped to :data:`BLUR_RANGE`."""
        self._blur = _clamp(value, *BLUR_RANGE)
        self._engine.set_blur(self._blur)
        self.update()

    def speed(self) -> float:
        """The motion multiplier; 1.0 is the shipped animation."""
        return self._speed

    def set_speed(self, value: float) -> None:
        """Set the motion multiplier. Clamped to :data:`SPEED_RANGE`.

        Takes effect on the next step, so nothing already on screen moves.
        """
        self._speed = _clamp(value, *SPEED_RANGE)
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
        self._engine.set_size(self._size)
        self.update()

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
        self._background = _as_color(color, self._background)
        if explicit:
            self._background_explicit = True
        self._engine.set_background(self._background)
        self.update()

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
        """Cap the frame rate."""
        self._fps = _clamp_int(fps, MIN_FPS, MAX_FPS)
        self._timer.setInterval(max(1, 1000 // self._fps))

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
        """Start ticking (no-op if already running)."""
        if not self._timer.isActive():
            self._clock.restart()
            self._timer.start()

    def stop(self) -> None:
        """Stop ticking. Costs exactly nothing while stopped."""
        self._timer.stop()

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
        dt = self._clock.restart() / 1000.0
        self.advance_frame(min(MAX_DT, dt) if dt > 0 else 1.0 / self._fps)

    def advance_frame(self, dt: float) -> None:
        """Step the animation by ``dt`` seconds and schedule a repaint.

        Called by the timer, and directly by the tests so no test ever waits
        on a real clock.
        """
        self._engine.advance(dt)
        self.update()

    def time(self) -> float:
        """The animation clock, in seconds."""
        return self._engine.time

    def set_time(self, seconds: float) -> None:
        """Jump the animation clock and repaint."""
        self._engine.set_time(seconds)
        self.update()

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
        painter = QPainter(self)
        rect = self.rect()
        self._paint_base(painter, rect)
        self._engine.paint(painter, rect.width(), rect.height())


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
        ``fps``, ``seed``, ``blur``, ``speed``, ``size``). The last three
        default to the user's preferences, so a caller that does not care
        about them should not pass them.
    :returns: the widget, already shown and lowered.
    """
    widget = AmbientWidget(host, theme=theme, palette=palette,
                           backdrop=backdrop, **kwargs)
    widget.follow_parent()
    widget.show()
    return widget
