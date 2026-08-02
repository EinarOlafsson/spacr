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
    Wide tilted curtains that slide across and cross-fade between two palette
    colours — a gradient wash whose hue keeps changing.
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

As shipped, at 1920x1080, offscreen raster, 120 frames each, including the
full-screen background fill every frame:

=========  ========  =========  =====================
 theme      dark      light      share of one core
=========  ========  =========  =====================
 blobs      1.28 ms   1.84 ms    3.1 % / 4.4 %
 aurora     1.16 ms   1.67 ms    2.8 % / 4.0 %
 ripple     1.47 ms   2.01 ms    3.5 % / 4.8 %
 drift      0.65 ms   0.65 ms    1.6 %
=========  ========  =========  =====================

and 0 % off screen. Light costs more because multiply is a slower blend than
addition; the DNA rain next door costs 4.4 %, so this is in family.

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
from PySide6.QtGui import (QColor, QImage, QLinearGradient, QPainter, QPen,
                           QPixmap, QRadialGradient)
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..theme import palette_for, relative_luminance

__all__ = [
    "AMBIENT_THEMES", "DEFAULT_THEME", "DEFAULT_PALETTE", "PALETTE_SETS",
    "AmbientWidget", "install_ambient", "theme_label", "theme_note",
    "palettes_for", "palette_label", "palette_note", "palette_colors",
    "default_palette_for", "is_valid_theme", "is_valid_palette",
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
    "aurora": "Wide curtains of colour sliding past and shifting hue.",
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
}

#: Which palettes each theme offers, and why the excluded ones are excluded.
#:
#: ``ripple`` and ``drift`` drop ``pastel``: a starfield is 1-4 px dots and a
#: ripple is a halo at a fifth of the alpha a blob gets, and a pale
#: low-contrast hue at either scale is indistinguishable from the page.
#: Offering it would be offering a setting that does nothing.
_THEME_PALETTES: Dict[str, Tuple[str, ...]] = {
    "blobs": ("spacr", "ember", "ocean", "pastel", "mono", "okabe"),
    "aurora": ("spacr", "ember", "ocean", "pastel", "mono", "okabe"),
    "ripple": ("spacr", "ember", "ocean", "mono", "okabe"),
    "drift": ("spacr", "ember", "ocean", "mono", "okabe"),
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
BUFFER_MAX_EDGE = 256

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
                 seed: Optional[int] = None):
        self.seed = seed
        self.time = 0.0
        self.frames = 0
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

    def advance(self, dt: float) -> None:
        """Step the clock by ``dt`` seconds. Negative steps are ignored.

        :attr:`frames` counts every call, including the ignored ones, so a
        test can prove a hidden widget stopped *asking* for frames rather
        than only proving that its clock stood still.
        """
        if dt > 0:
            self.time += float(dt)
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

    def buffer_size(self, width: int, height: int) -> Tuple[int, int]:
        """Buffer dimensions for a ``width`` x ``height`` canvas."""
        longest = max(int(width), int(height))
        scale = max(1, int(math.ceil(longest / BUFFER_MAX_EDGE)))
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
            radius = blob.radius * short * (
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

#: Four fat curtains, not five thin ones. The first cut had five evenly
#: spaced bands of similar thickness and it read as a test pattern — regular
#: horizontal stripes. Fewer, thicker, wildly different sizes, overlapping,
#: and each tilted a few degrees off horizontal: that reads as a wash.
AURORA_BANDS = 4
AURORA_THICKNESS = (0.22, 0.55)     # fraction of the canvas height
AURORA_TILT = 14.0                  # max degrees off horizontal
#: How far past the canvas edges a curtain is drawn. A tilted rectangle has
#: to overhang or it leaves bare triangles in the corners.
AURORA_OVERHANG = 1.9
AURORA_DRIFT = (0.05, 0.18)
AURORA_DRIFT_PERIOD = (30.0, 90.0)  # seconds
AURORA_HUE_PERIOD = (18.0, 46.0)    # seconds per colour cross-fade cycle
AURORA_ALPHA_DARK = 0.22
AURORA_ALPHA_LIGHT = 0.42


@dataclass
class Band:
    """One aurora curtain, in normalised units."""

    y: float
    thickness: float
    tilt: float
    drift: float
    rate: float
    phase: float
    hue_rate: float
    hue_phase: float
    color: int


class AuroraEngine(_BufferedEngine):
    """Wide curtains that slide past and cross-fade between hues.

    The bands are thick, tilted and overlap heavily on purpose — thin level
    ones read as stripes, fat tilted ones read as a wash.

    :meth:`geometry` yields ``(cy, half_thickness)`` per band, in pixels.
    """

    name = "aurora"

    def _configure(self, rng: random.Random) -> None:
        self.bands: List[Band] = []
        for i in range(AURORA_BANDS):
            self.bands.append(Band(
                # Spread the resting positions evenly, then jitter: bands
                # rolled uniformly pile up and leave the top third empty.
                y=(i + 0.5) / AURORA_BANDS + rng.uniform(-0.06, 0.06),
                thickness=rng.uniform(*AURORA_THICKNESS),
                tilt=rng.uniform(-AURORA_TILT, AURORA_TILT),
                drift=rng.uniform(*AURORA_DRIFT),
                rate=2 * math.pi / rng.uniform(*AURORA_DRIFT_PERIOD),
                phase=rng.uniform(0.0, 2 * math.pi),
                hue_rate=2 * math.pi / rng.uniform(*AURORA_HUE_PERIOD),
                hue_phase=rng.uniform(0.0, 2 * math.pi),
                color=i,
            ))

    def geometry(self, width: int, height: int) -> Tuple[tuple, ...]:
        t = self.time
        out = []
        for band in self.bands:
            cy = (band.y + band.drift
                  * math.sin(band.rate * t + band.phase)) * height
            out.append((cy, max(1.0, band.thickness * height * 0.5)))
        return tuple(out)

    def band_color(self, band: Band) -> QColor:
        """The band's colour right now — a cross-fade between two palette
        entries, which is what "slowly shifts hue" means here."""
        colors = self.paint_colors
        a = colors[band.color % len(colors)]
        b = colors[(band.color + 1) % len(colors)]
        u = 0.5 + 0.5 * math.sin(band.hue_rate * self.time + band.hue_phase)
        return _mix(a, b, u)

    def _paint_field(self, painter: QPainter, width: int, height: int) -> None:
        peak = AURORA_ALPHA_DARK if self.dark else AURORA_ALPHA_LIGHT
        span = width * AURORA_OVERHANG
        for band, (cy, half) in zip(self.bands, self.geometry(width, height)):
            color = self.band_color(band)
            gradient = QLinearGradient(0.0, -half, 0.0, half)
            gradient.setColorAt(0.0, _with_alpha(color, 0.0))
            gradient.setColorAt(0.5, _with_alpha(color, peak))
            gradient.setColorAt(1.0, _with_alpha(color, 0.0))
            painter.save()
            painter.translate(width * 0.5, cy)
            painter.rotate(band.tilt)
            painter.setBrush(gradient)
            painter.drawRect(QRectF(-span * 0.5, -half, span, 2 * half))
            painter.restore()


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
            reach = source.reach * half_diagonal
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

    def _tint(self, color: QColor) -> QColor:
        return QColor(color) if self.dark \
            else _mix(color, QColor(0, 0, 0), DRIFT_DARKEN_ON_LIGHT)

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
                        DRIFT_LAYERS[particle.layer][0]))
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

    def _pen(self, color_index: int, layer: int, step: int) -> QPen:
        key = (color_index, layer, step)
        pen = self._pens.get(key)
        if pen is None:
            colors = self.paint_colors
            alpha = step / (DRIFT_ALPHA_STEPS - 1)
            pen = QPen(_with_alpha(colors[color_index % len(colors)], alpha))
            # A round-capped pen makes drawPoints draw filled circles, which
            # is how a batch of dots gets drawn in one call.
            pen.setWidthF(DRIFT_LAYERS[layer][0])
            pen.setCapStyle(Qt.RoundCap)
            self._pens[key] = pen
        return pen

    def paint(self, painter: QPainter, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            return
        painter.setRenderHint(QPainter.Antialiasing, True)
        n_colors = len(self.paint_colors)
        steps = [self._alpha_step(i) for i in range(len(DRIFT_LAYERS))]
        buckets: Dict[Tuple[int, int, int], List[QPointF]] = {}
        for particle, (x, y, _size) in zip(
                self.particles, self.geometry(width, height)):
            key = (particle.color % n_colors, particle.layer,
                   steps[particle.layer])
            buckets.setdefault(key, []).append(QPointF(x, y))
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
                seed: Optional[int] = None) -> AmbientEngine:
    """Build the engine for ``theme``/``palette``. Raises on unknown names."""
    _require_theme(theme)
    _require_palette(theme, palette)
    return _ENGINES[theme](palette_colors(theme, palette), background,
                           seed=seed)


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
                 seed: Optional[int] = None):
        super().__init__(parent)
        self._theme = _require_theme(theme)
        self._palette = coerce_palette(self._theme, palette)
        self._seed = seed
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
                                   self._background, seed=seed)

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
                             seed=self._seed)
        engine.set_time(self._engine.time)
        self._engine = engine
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
        ``fps``, ``seed``).
    :returns: the widget, already shown and lowered.
    """
    widget = AmbientWidget(host, theme=theme, palette=palette,
                           backdrop=backdrop, **kwargs)
    widget.follow_parent()
    widget.show()
    return widget
