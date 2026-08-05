"""Live DNA rain — a Matrix-style ATGC cascade painted *behind* a screen.

This is the Qt descendant of :func:`spacr.gui_elements.generate_dna_matrix`,
which renders the same idea offline to a GIF/MP4. The ideas carried over
from it are the ones that make the effect read as "rain" rather than as a
scrolling table:

* every column picks its own string length (10-100 glyphs, capped so
  it cannot dwarf the canvas — see :data:`MAX_STRING_SCREENS`),
* every column starts at its own row *above* the canvas, so columns do
  not enter in lockstep,
* the leading glyph is drawn in a different colour from the trail,
* a short run of glyphs inside each string is highlighted.

Three deliberate departures:

``ATGC only``
    The offline renderer had a ``lowercase_prob`` that mixed ``a/t/g/c``
    in. Here the alphabet is exactly ``A``, ``T``, ``G``, ``C`` — plus
    the ``spaCR`` splice below.

``Live, not rendered``
    Nothing is written to disk. :class:`DnaRainWidget` paints frames in
    real time, driven by a capped timer that *stops whenever the widget
    is not on screen* — this sits behind the sequencing pipeline, which
    is doing real work, so it must not cost a core.

``The tail fades, not the head``
    The offline version faded the glyphs nearest the head. That reads
    backwards; here the trailing (upper) end of each string fades out.

The ``spaCR`` splice
--------------------
Very infrequently a column carries the literal word ``spaCR``, spelled down
the column one letter per cell, exactly like the bases around it::

    s
    p
    a
    C
    R

The casing is load-bearing; the orientation is what makes it part of the rain.
It used to be stored as a single token in a single cell and drawn
horizontally, overflowing rightwards across its neighbours — which read as a
label pasted over the effect rather than as part of it, and cost the renderer
a whole second paint pass, measured token widths, widened dirty rectangles and
a cleared backing rectangle so neighbouring glyphs did not tangle with the
letters. None of that machinery exists now: every cell holds exactly one
character, so the word is cached and blitted like any other glyph and cannot
overdraw anything. See :data:`SPACR_SPLICE_PROBABILITY` for the rate and what
it works out to in practice.

Legibility
----------
Screen content sits in front of this widget. It therefore never takes
focus, is transparent to mouse events, lowers itself to the bottom of
the sibling stacking order, and paints its glyphs at
:data:`DEFAULT_OPACITY` over the theme background so anything in front
of it stays readable.

Settings
--------
Colour (fixed, or a random hue per column), speed, visibility and font
size, all live. They are **not** on the screen: they live in a popover
behind a ``DNA`` button beside the ``AI`` toggle — see
:mod:`spacr.qt.widgets.dna_rain_settings`. :class:`DnaRainSettingsBar`
is the panel itself, which lays out either as that popover's grid or as
the original single row.

Cost
----
This runs behind the sequencing pipeline, so it has to be close to
free. Three things get it there, and all three were measured rather
than assumed (1920x1080, 120 columns, 67 rows):

1. *The timer stops whenever the widget is not on screen* — hidden, or
   in a minimised window. Zero frames, zero CPU.
2. *Each string is pre-rendered once into an opaque pixmap* with the
   background already composited in, and blitted at its current offset
   every frame. Drawing 5400 glyphs with ``drawText`` costs 35 ms a
   frame; the same glyphs as 120 opaque strips cost 0.46 ms — and an
   opaque strip is 25x cheaper to blit than a translucent one, which
   is why the alpha is baked in rather than applied by the painter.
   A strip is re-rendered only when its column respawns or the
   styling changes; the cache is ~7 MB at 1920x1080.
   :meth:`DnaRainWidget.set_backdrop` gives that up deliberately — a
   picture under the rain is not a constant to bake against — so the
   translucent path is taken only by the themes that have a wallpaper
   to show, and dark and light keep the numbers above.
3. *Only the columns that moved are repainted.* Positions are
   quantised to whole cells, so a column is dirty only when its
   integer row changes — slow columns cost nothing on most ticks. See
   :data:`MAX_DIRTY_RECTS` for where partial repaints stop paying.

Together: 0.53 ms a frame, 3.2 % of one core at 1920x1080 and 60 fps,
and 0.00 % while off screen. Random colour does not move that number —
the hues are baked into the same cached strips, and a pen set is built
once per whole degree of hue (:func:`_hue_bucket`) rather than per
column, per frame.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from PySide6.QtCore import (QElapsedTimer, QEvent, QPoint, QRect, Qt, QTimer,
                            Signal)
from PySide6.QtGui import (QColor, QFont, QFontDatabase, QFontMetricsF,
                           QImage, QPainter, QPen, QPixmap)
from PySide6.QtWidgets import (QColorDialog, QGridLayout,
                               QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                               QSlider, QWidget)

from ..theme import RADIUS, SPACING, palette_for
from .toggle import Toggle

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The whole alphabet. ATGC, uppercase, nothing else.
BASES = ("A", "T", "G", "C")

#: The shipped glyph colour. Teal reads as DNA-ish without competing with
#: spaCR's own accent, and stays legible on both flat themes.
#:
#: This is the *default*, not a theme lookup. The rain used to take the
#: theme's ``accent``, which is the same blue as the Run button and the AI
#: toggle — so the backdrop read as chrome. :meth:`DnaRainWidget.apply_theme`
#: still exists for anyone who explicitly wants the palette's colour.
DEFAULT_COLOR = "#009B9B"

#: With random colour on, every column gets its own hue. Saturation and
#: lightness are borrowed from the picked colour so the field stays one
#: family, but they are floored/clamped: a near-grey pick would otherwise
#: make "random" produce twelve indistinguishable greys, and a near-white
#: one would wash every hue out.
RANDOM_MIN_SATURATION = 0.55
RANDOM_MIN_LIGHTNESS = 0.30
RANDOM_MAX_LIGHTNESS = 0.72

#: The one non-base token. Exact casing is load-bearing.
SPACR_TOKEN = "spaCR"

#: Probability that a freshly (re)spawned column carries a ``spaCR``.
#:
#: "Very infrequently" needs a number to be testable, so here is the
#: arithmetic. At 1920x1080 with the default 16 px font there are 120
#: columns and 67 rows. A column lives for ``(length + rows) / speed``
#: seconds — mean length 55, mean speed 13 cells/s — which measures out
#: at 10.2 respawns per second across the whole field. At this
#: probability that is **one ``spaCR`` every ~61 seconds** of viewing:
#: about once a minute, often enough to be noticed and rare enough to
#: stay a surprise. Halve the font size and you double the column count
#: and roughly halve the interval.
SPACR_SPLICE_PROBABILITY = 0.0016

#: String length range in cells, inherited from ``generate_dna_matrix``.
MIN_STRING_CELLS = 10
MAX_STRING_CELLS = 100

#: A string is additionally capped at this multiple of the canvas
#: height. Only bites at large font sizes, where a 100-cell string
#: would be nine screens tall — it would read as a solid bar, and its
#: pre-rendered strip would be megabytes.
MAX_STRING_SCREENS = 1.5

#: Per-column base fall speed, in cells per second. The spread is the
#: whole point: it is what stops the columns marching in lockstep.
MIN_SPEED_CELLS_PER_S = 4.0
MAX_SPEED_CELLS_PER_S = 22.0

#: Length of the highlighted run inside a string.
HIGHLIGHT_RUN_CELLS = 8

#: Fraction of a string (from the trailing end) over which it fades out,
#: and the alpha multiplier the very last glyph fades to.
TAIL_FADE_FRACTION = 0.35
MIN_TAIL_ALPHA = 0.12

#: Number of precomputed trail pens. Building a QColor per glyph per
#: frame is the single easiest way to burn a core here; this is the LUT
#: that avoids it.
TRAIL_STEPS = 12

DEFAULT_FONT_PX = 16
MIN_FONT_PX = 4
MAX_FONT_PX = 96

#: Frame-rate cap. 24 fps was enough for the fast columns and visibly stepped on the slow
#: ones: a column at 4 cells/s advances one whole glyph every 6 frames, so it
#: sat still and then jumped. Positions are quantised to whole cells, so the
#: only way to smooth that is to give the slow columns more frames to move in.
#: 60 costs more, but the dirty-rect repaint means an idle column still costs
#: nothing — only the columns that actually crossed a cell boundary repaint.
DEFAULT_FPS = 60
MIN_FPS = 1
MAX_FPS = 60

#: Glyph opacity. The rain sits BEHIND the screen content, so this trades
#: visibility against the legibility of whatever is in front of it. It began
#: at 0.22, which the user found too faint to read as an effect at all --
#: especially over a light theme, where a low-alpha accent colour on a pale
#: surface has almost no contrast to spend. Raised, and exposed as a slider so
#: it can be dialled per taste and per theme rather than guessed at once here.
#: Glyph opacity. 20% keeps the rain a texture behind the screen content
#: rather than something competing with it.
DEFAULT_OPACITY = 0.20

#: Slider bounds for the opacity control, as whole percent.
MIN_OPACITY_PCT = 5
MAX_OPACITY_PCT = 90

#: The head glyph is drawn at this multiple of the trail opacity
#: (clamped to 1.0) so it stays the brightest thing in the column.
HEAD_ALPHA_BOOST = 3.0

#: Largest simulation step accepted from the wall clock. If the app was
#: busy for two seconds the rain resumes where it was rather than
#: teleporting every column down the screen.
MAX_DT = 0.25

#: Dirty spans of adjacent columns are merged, and above this many
#: resulting rectangles the widget asks for one full repaint instead.
#:
#: Measured, not guessed. Because the pre-rendered strips make the
#: painting itself nearly free, what dominates a partial repaint is the
#: per-rectangle bookkeeping — invalidate, clip, flush — at roughly
#: 0.2 ms each. End-to-end median frame cost at 1920x1080:
#:
#:     rects/frame   partial   one full repaint
#:         1.2       0.25 ms       0.51 ms
#:         3.2       0.63 ms       0.51 ms
#:         7.8       1.68 ms       0.71 ms
#:        29.0       5.98 ms       0.80 ms
#:
#: so partial repaints win up to about two rectangles and lose badly
#: above that.
MAX_DIRTY_RECTS = 2

#: Speed-multiplier range offered by the settings bar.
MIN_SPEED_MULTIPLIER = 0.2
MAX_SPEED_MULTIPLIER = 4.0

#: How far, in HSL lightness, the head glyph is pushed away from the
#: trail colour, and the minimum separation it must end up with from
#: both the trail and the background before we give up and go to an
#: extreme.
HEAD_LIGHT_STEP = 0.45
HEAD_MIN_SEPARATION = 0.20


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def derive_head_color(base: QColor, background: QColor) -> QColor:
    """Return the leading-glyph colour for a trail colour of ``base``.

    The user picks one colour; the head is *derived* from it rather than
    being a second colour they cannot control. It is pushed away in HSL
    lightness, in whichever direction gains the most separation from
    both the trail and the background — so it survives the extremes,
    including a trail colour identical to the background.

    :param base: the user's trail colour.
    :param background: what the rain is painted onto.
    :returns: a colour distinguishable from both, hue/saturation kept.
    """
    bl = base.lightnessF()
    gl = background.lightnessF()

    def gain(light: float) -> float:
        return min(abs(light - bl), abs(light - gl))

    up = min(1.0, bl + HEAD_LIGHT_STEP)
    down = max(0.0, bl - HEAD_LIGHT_STEP)
    target = up if gain(up) >= gain(down) else down
    if gain(target) < HEAD_MIN_SEPARATION:
        # A near-white trail over a mid-grey background has nowhere to
        # go within one step: up hits the ceiling, down lands on the
        # background. Give up on the step and take whichever end of the
        # lightness axis is furthest from *both*.
        target = max((0.0, 1.0), key=gain)
    hue = base.hueF()
    sat = base.saturationF()
    if hue < 0.0:          # achromatic: QColor reports hue -1
        hue, sat = 0.0, 0.0
    return QColor.fromHslF(hue, sat, target)


def random_hue_color(base: QColor, hue: float) -> QColor:
    """Return ``base`` moved to ``hue``, keeping the family it belongs to.

    Random colour is *per column*, and a column is one falling string
    among a hundred. Rolling all three HSL components would have given
    the field a scatter of near-blacks, near-whites and greys — most of
    which do not read as glyphs at 20 % opacity behind a settings form.
    Only the hue is random; saturation and lightness are taken from the
    colour in the picker, floored and clamped so the result is always a
    colour and always visible.

    :param base: the picked colour, which lends its saturation/lightness.
    :param hue: hue in ``0..1``.
    :returns: a fully saturated-enough, mid-lightness colour at ``hue``.
    """
    sat = max(RANDOM_MIN_SATURATION, base.saturationF())
    light = min(RANDOM_MAX_LIGHTNESS,
                max(RANDOM_MIN_LIGHTNESS, base.lightnessF()))
    return QColor.fromHslF(max(0.0, min(0.9999, float(hue))), sat, light)


def blend(a: QColor, b: QColor, t: float) -> QColor:
    """Linear RGB blend, ``t=0`` -> ``a``, ``t=1`` -> ``b``."""
    t = max(0.0, min(1.0, float(t)))
    return QColor(
        int(round(a.red() + (b.red() - a.red()) * t)),
        int(round(a.green() + (b.green() - a.green()) * t)),
        int(round(a.blue() + (b.blue() - a.blue()) * t)),
    )


def _as_color(value: Union[QColor, str, None], fallback: QColor) -> QColor:
    """Coerce ``value`` to a valid QColor, falling back when it is not."""
    if value is None:
        return QColor(fallback)
    color = QColor(value)
    return color if color.isValid() else QColor(fallback)


def _as_pixmap(value) -> Optional[QPixmap]:
    """Coerce a path / QPixmap / QImage to a usable QPixmap, or ``None``.

    Never raises and never returns a null pixmap: a wallpaper that has
    been deleted between the stylesheet being built and this widget
    being constructed is a cosmetic miss, not a crash, and the rain
    falls back to its flat background colour.
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


def _region_rects(region, fallback: QRect) -> List[QRect]:
    """The individual rectangles of a paint region.

    Qt hands ``paintEvent`` a *region* but reports ``event.rect()`` as
    its bounding box; using the box would repaint the whole canvas as
    soon as one column on each edge moved. Older/other Qt bindings do
    not expose the rectangles, so fall back to the bounding box.
    """
    try:
        rects = [QRect(r) for r in region]
    except Exception:
        rects = []
    return rects or [fallback]


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class Column:
    """One falling string.

    :ivar tokens: one entry per *cell*; usually a single base, but a
        spliced entry is the whole ``spaCR`` word.
    :ivar length: number of cells (== ``len(tokens)``).
    :ivar speed: base fall rate in cells/second, before the multiplier.
    :ivar head: row of the leading glyph, fractional and often negative
        (the string starts above the canvas).
    :ivar row: ``floor(head)`` — the cell the head occupies. Still used for
        the dirty-span arithmetic, which works in rows.
    :ivar y_px: the strip's top edge in PIXELS, rounded from the fractional
        head. THIS is what the column is painted at and what decides whether
        it is dirty. Painting at ``row * cell`` quantised every column to
        whole glyph heights, so a slow column (4 cells/s) sat still for six
        frames and then jumped a whole character — the stepping that reads as
        choppy. Raising the frame rate cannot fix that on its own: the
        position simply has fewer places it is allowed to be.
    :ivar hi_start: first cell index of the highlighted run.
    :ivar hi_end: one past the last cell index of the highlighted run.
    :ivar word_index: cell index of the multi-character token, or -1.
    :ivar generation: bumped on every respawn; the widget's pre-rendered
        strip cache keys off it.
    :ivar hue: this string's own hue in ``0..1``, re-rolled on every
        respawn. Only read when the widget is in random-colour mode; it
        is rolled unconditionally, and from a stream of its own, so that
        turning random colour on or off cannot change *where* anything
        falls (see :class:`DnaRainEngine`).
    """

    tokens: List[str]
    length: int
    speed: float
    head: float
    row: int
    y_px: int
    hi_start: int
    hi_end: int
    word_index: int
    generation: int = 0
    hue: float = 0.0

    @property
    def has_word(self) -> bool:
        """True when this string carries a multi-character token."""
        return self.word_index >= 0


class DnaRainEngine:
    """Qt-free simulation of the falling columns.

    Deterministic: the same ``seed`` and the same sequence of calls
    always produce the same animation.

    Per-column hues come from a **second** RNG rather than the main one.
    Drawing them from the main stream would have shifted every length,
    speed and start row by one draw, so a seeded rain would have fallen
    differently depending on a purely cosmetic setting. Two streams keep
    ``snapshot()`` byte-identical whether random colour is on or off.

    :param width: canvas width in pixels.
    :param height: canvas height in pixels.
    :param font_size: glyph size in pixels; also the cell/column stride.
    :param seed: RNG seed. ``None`` seeds from the system entropy.
    :param spacr_probability: chance per respawn of a ``spaCR`` splice.
    """

    def __init__(self, width: int = 0, height: int = 0,
                 font_size: int = DEFAULT_FONT_PX,
                 seed: Optional[int] = None,
                 spacr_probability: float = SPACR_SPLICE_PROBABILITY):
        self._rng = random.Random(seed)
        # Its own stream, offset from the seed so it is neither the same
        # sequence nor correlated with it, and still reproducible.
        self._hue_rng = random.Random(
            None if seed is None else (int(seed) ^ 0x9E3779B9))
        self.seed = seed
        self.spacr_probability = float(spacr_probability)
        self.speed_multiplier = 1.0
        self.spacr_splices = 0
        self.respawns = 0
        self.frames = 0
        self._font_size = _clamp_int(font_size, MIN_FONT_PX, MAX_FONT_PX)
        self._width = max(0, int(width))
        self._height = max(0, int(height))
        self.columns: List[Column] = []
        self.relayout()

    # -- geometry ------------------------------------------------------
    @property
    def cell_size(self) -> int:
        """Row height and column stride, in pixels."""
        return self._font_size

    @property
    def font_size(self) -> int:
        return self._font_size

    @property
    def n_rows(self) -> int:
        """Number of whole cells that fit vertically. May be 0."""
        return self._height // self._font_size

    @property
    def n_columns(self) -> int:
        return len(self.columns)

    @property
    def max_length(self) -> int:
        """Longest string this canvas allows — see MAX_STRING_SCREENS."""
        cap = int(self.n_rows * MAX_STRING_SCREENS)
        return max(MIN_STRING_CELLS, min(MAX_STRING_CELLS, cap))

    @property
    def size(self) -> Tuple[int, int]:
        """Canvas size in pixels."""
        return (self._width, self._height)

    def resize(self, width: int, height: int) -> bool:
        """Resize the canvas and re-lay-out the columns.

        :returns: True when the size actually changed.
        """
        width, height = max(0, int(width)), max(0, int(height))
        if (width, height) == (self._width, self._height):
            return False
        self._width, self._height = width, height
        self.relayout()
        return True

    def set_font_size(self, px: int) -> None:
        """Set the glyph size, which is also the column stride."""
        px = _clamp_int(px, MIN_FONT_PX, MAX_FONT_PX)
        if px == self._font_size:
            return
        self._font_size = px
        self.relayout()

    def set_speed_multiplier(self, factor: float) -> None:
        """Scale every column's speed, preserving their relative rates."""
        self.speed_multiplier = max(0.0, float(factor))

    def relayout(self) -> None:
        """Rebuild the column list for the current size and font."""
        wanted = self._width // self._font_size if self._font_size else 0
        wanted = max(0, wanted)
        self.columns = [self._spawn(initial=True) for _ in range(wanted)]

    # -- columns -------------------------------------------------------
    def _new_tokens(self, length: int) -> Tuple[List[str], int]:
        """Roll a string of bases, occasionally splicing in ``spaCR``.

        :returns: ``(tokens, word index)``; the index is -1 when no splice
            happened, otherwise the cell index of the word's FIRST letter.

        The word occupies ``len(SPACR_TOKEN)`` consecutive cells, one letter
        each, so it falls down the column exactly like the bases around it::

            s
            p
            a
            C
            R

        It used to be written into a single cell as the whole string and drawn
        horizontally, which made it read as a label pasted across the rain
        rather than as part of it — and forced a second paint pass that
        cleared a rectangle running over its neighbouring columns.

        A string shorter than the word is left unspliced rather than truncated:
        half a word is not the surprise this is for.
        """
        rng = self._rng
        tokens = [rng.choice(BASES) for _ in range(length)]
        word_index = -1
        word_len = len(SPACR_TOKEN)
        if length >= word_len and rng.random() < self.spacr_probability:
            word_index = rng.randrange(length - word_len + 1)
            tokens[word_index:word_index + word_len] = list(SPACR_TOKEN)
            self.spacr_splices += 1
        return tokens, word_index

    def _roll(self, column: Optional[Column], initial: bool) -> Column:
        rng = self._rng
        length = rng.randint(MIN_STRING_CELLS, self.max_length)
        speed = rng.uniform(MIN_SPEED_CELLS_PER_S, MAX_SPEED_CELLS_PER_S)
        tokens, word_index = self._new_tokens(length)
        if initial:
            # Spread the initial heads over a whole life cycle so the
            # field looks like it has been running, and so no two
            # columns share a start time.
            head = rng.uniform(-(length + self.n_rows), float(self.n_rows))
        else:
            head = -1.0
        # The highlighted run, clamped so it fits however short the
        # string is.
        run = min(HIGHLIGHT_RUN_CELLS, length)
        hi_start = rng.randrange(0, length - run + 1)
        hi_end = hi_start + run
        hue = self._hue_rng.random()
        if column is None:
            return Column(tokens=tokens, length=length, speed=speed,
                          head=head, row=int(math.floor(head)),
                          y_px=0,
                          hi_start=hi_start, hi_end=hi_end,
                          word_index=word_index, hue=hue)
        column.hue = hue
        column.tokens = tokens
        column.length = length
        column.speed = speed
        column.head = head
        column.row = int(math.floor(head))
        column.y_px = self._y_px(column)
        column.hi_start = hi_start
        column.hi_end = hi_end
        column.word_index = word_index
        column.generation += 1
        return column

    def _spawn(self, initial: bool) -> Column:
        return self._roll(None, initial)

    def _respawn(self, column: Column) -> None:
        self._roll(column, initial=False)
        self.respawns += 1

    # -- stepping ------------------------------------------------------
    def _y_px(self, column: "Column") -> int:
        """Top edge of ``column``'s strip in pixels, from its FRACTIONAL head.

        Rounding here rather than flooring to a cell is the whole of the
        smoothing: the strip may now sit at any pixel, so a slow column
        creeps a pixel at a time instead of holding still and jumping a full
        glyph. It is still an integer, so a column that has not moved a whole
        pixel yet is skipped and costs nothing.
        """
        return int(round((column.head - column.length + 1) * self.cell_size))

    def advance(self, dt: float) -> List[Tuple[int, int, int]]:
        """Step the simulation by ``dt`` seconds.

        :param dt: elapsed time in seconds.
        :returns: ``(column index, first row, last row)`` spans that
            changed, already clipped to the canvas. Columns whose
            integer row did not move contribute nothing — that is the
            whole point of quantising to cells.
        """
        self.frames += 1
        rows = self.n_rows
        if not self.columns or rows <= 0 or dt <= 0.0:
            return []
        dirty: List[Tuple[int, int, int]] = []
        for index, column in enumerate(self.columns):
            old_row = column.row
            old_top = old_row - column.length + 1
            column.head += column.speed * self.speed_multiplier * dt
            respawned = False
            if column.head - column.length + 1 > rows:
                self._respawn(column)
                respawned = True
            new_row = int(math.floor(column.head))
            old_y = column.y_px
            new_y = self._y_px(column)
            # Dirty on PIXEL movement, not cell movement. The old test — "did
            # the integer row change" — is what made slow columns step: they
            # were skipped for five frames out of six and then redrawn a whole
            # glyph lower.
            if new_y == old_y and not respawned:
                continue
            column.row = new_row
            column.y_px = new_y
            # The span still has to be expressed in rows, so widen it by a
            # cell on each side to cover a strip that now straddles a
            # boundary.
            cell = max(1, self.cell_size)
            top = max(0, min(old_y, new_y) // cell - 1)
            bottom = min(rows - 1,
                         (max(old_y, new_y) + column.length * cell) // cell + 1)
            if bottom >= top:
                dirty.append((index, top, bottom))
        return dirty

    # -- introspection -------------------------------------------------
    def column_text(self, index: int) -> str:
        """The full string of column ``index``, tokens concatenated."""
        return "".join(self.columns[index].tokens)

    def tokens(self) -> List[str]:
        """Every token currently in the field, flattened."""
        return [tok for column in self.columns for tok in column.tokens]

    def snapshot(self) -> Tuple:
        """A hashable state summary — used to compare two runs."""
        return tuple(
            (round(c.head, 6), round(c.speed, 6), c.length,
             c.hi_start, tuple(c.tokens))
            for c in self.columns
        )


def _clamp_int(value, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def _hue_bucket(hue: float) -> int:
    """``hue`` in ``0..1`` as a whole degree — the pen cache's key.

    360 buckets is finer than anyone can see at 20 % alpha and bounds
    the cache at 360 entries however long the rain runs.
    """
    return int(float(hue) * 360) % 360


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class DnaRainWidget(QWidget):
    """The live backdrop: paints :class:`DnaRainEngine` at a capped rate.

    :param parent: parent widget; the rain sizes itself to it when
        :meth:`follow_parent` is called.
    :param seed: RNG seed for a reproducible animation.
    :param font_size: glyph size in px (also the column stride).
    :param fps: frame-rate cap.
    :param color: trail colour; defaults to :data:`DEFAULT_COLOR`.
    :param background: colour painted under the glyphs; defaults to the
        theme background.
    :param random_colors: when true every column takes its own hue
        instead of all of them sharing ``color``.
    :param backdrop: image painted under the glyphs instead of the flat
        colour — a path, a ``QPixmap``/``QImage``, or ``None``. Give it
        the image theme's wallpaper and the rain stops hiding it.
    :param opacity: glyph alpha in ``0..1``.
    :param spacr_probability: per-respawn chance of a ``spaCR`` splice.
    :param theme: palette to take defaults from; defaults to the user's
        effective theme.
    """

    def __init__(self, parent: Optional[QWidget] = None, *,
                 seed: Optional[int] = None,
                 font_size: int = DEFAULT_FONT_PX,
                 fps: int = DEFAULT_FPS,
                 color: Union[QColor, str, None] = None,
                 background: Union[QColor, str, None] = None,
                 backdrop=None,
                 opacity: float = DEFAULT_OPACITY,
                 random_colors: bool = False,
                 spacr_probability: float = SPACR_SPLICE_PROBABILITY,
                 theme: Optional[str] = None):
        super().__init__(parent)
        palette = palette_for(theme or _effective_theme())
        self._bg = _as_color(background, QColor(palette["bg"]))
        self._bg.setAlpha(255)
        # The shipped teal, NOT the theme accent: the accent is the Run
        # button and the AI toggle, and a backdrop in it read as chrome.
        self._color = _as_color(color, QColor(DEFAULT_COLOR))
        self._opacity = max(0.0, min(1.0, float(opacity)))
        self._random_colors = bool(random_colors)
        self._backdrop: Optional[QPixmap] = _as_pixmap(backdrop)

        # Never in front of, never in the way of, the real content.
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setFocusPolicy(Qt.NoFocus)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self._engine = DnaRainEngine(
            width=max(0, self.width()), height=max(0, self.height()),
            font_size=font_size, seed=seed,
            spacr_probability=spacr_probability)

        self._font = QFont()
        self._ascent = 0
        self._trail_pens: List[QPen] = []
        self._head_pen = QPen()
        self._hi_pen = QPen()
        # Per-hue pen sets for random-colour mode, quantised to whole
        # degrees. Strips are already cached per column, so this is only
        # ever hit on a respawn or a restyle — but a restyle re-renders
        # every column at once, and 120 pen sets built one after another
        # is the sort of thing that shows up in a frame budget of half a
        # millisecond.
        self._hue_pens: dict = {}
        # Pre-rendered opaque strip per column, keyed by
        # (column generation, styling generation).
        self._strips: List[Optional[QPixmap]] = []
        self._strip_keys: List[Tuple[int, int]] = []
        self._style_gen = 0
        self._rebuild_font()
        self._rebuild_pens()

        #: True when the last :meth:`advance_frame` gave up on partial
        #: repaints and asked for a full one.
        self.last_full_repaint = False
        self._fps = _clamp_int(fps, MIN_FPS, MAX_FPS)
        self._clock = QElapsedTimer()
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.CoarseTimer)
        self._timer.setInterval(max(1, 1000 // self._fps))
        self._timer.timeout.connect(self._on_tick)
        self._watched: Optional[QWidget] = None

    def focusInEvent(self, event) -> None:  # noqa: N802 (Qt override)
        """Reject even programmatic focus; this widget is decorative only."""
        event.ignore()
        self.clearFocus()

    # -- appearance ----------------------------------------------------
    def color(self) -> QColor:
        """Current trail colour."""
        return QColor(self._color)

    def head_color(self) -> QColor:
        """Current leading-glyph colour, derived from the trail colour."""
        return derive_head_color(self._color, self._bg)

    def background_color(self) -> QColor:
        return QColor(self._bg)

    def opacity(self) -> float:
        return self._opacity

    def random_colors(self) -> bool:
        """True while every column paints in its own hue."""
        return self._random_colors

    def set_random_colors(self, on: bool) -> None:
        """Give every column its own hue (or put them all back on one).

        Per column, not per session, and re-rolled on every respawn —
        see :meth:`column_color`. The colours are baked into the
        pre-rendered strips, so flipping this drops the strip cache; it
        costs one full re-render, the same as moving the colour picker.
        """
        on = bool(on)
        if on == self._random_colors:
            return
        self._random_colors = on
        self._invalidate_strips()
        self.update()

    def column_color(self, index: int) -> QColor:
        """The trail colour column ``index`` is actually painted in.

        The picked colour for every column in fixed mode; that column's
        own hue in random mode. This is what the settings popover's
        swatch cannot show and what a test has to look at.
        """
        return self._column_color(self._engine.columns[index])

    def _column_color(self, column: Column) -> QColor:
        if not self._random_colors:
            return QColor(self._color)
        # Quantised exactly as the pen cache quantises it, so this
        # reports the colour that is on screen rather than one a
        # fraction of a degree away from it.
        return random_hue_color(self._color, _hue_bucket(column.hue) / 360.0)

    def set_color(self, color: Union[QColor, str]) -> None:
        """Set the trail colour; the head colour re-derives from it.

        In random-colour mode this is still live: the picked colour
        lends its saturation and lightness to every column's hue, so it
        chooses how vivid and how bright the random field is.
        """
        self._color = _as_color(color, self._color)
        self._rebuild_pens()
        self.update()

    def set_background_color(self, color: Union[QColor, str]) -> None:
        """Set the colour painted under the glyphs.

        The rain paints its own background because it repaints only the
        cells that changed; a translucent backdrop would smear, so the
        colour is forced opaque. Under the Space theme this is the
        palette's flat-sky fallback rather than the star field.
        """
        self._bg = _as_color(color, self._bg)
        self._bg.setAlpha(255)
        self._rebuild_pens()
        self.update()

    def backdrop(self) -> Optional[QPixmap]:
        """The image painted under the glyphs, or ``None`` for a flat fill."""
        return self._backdrop

    def set_backdrop(self, source) -> None:
        """Paint ``source`` under the glyphs instead of a flat colour.

        The rain is an opaque backdrop by construction — it repaints
        only the cells that changed, so it has to be able to *clear*
        them, and clearing to a translucent colour smears the previous
        frame. That is the right trade on the dark and light themes,
        where the thing behind it is a flat ``bg`` the rain can
        reproduce exactly. On an image theme it is not: the flat colour
        is nothing like the wallpaper, and an opaque rain hid the
        photograph completely on the one screen that has a rain.

        Handing the wallpaper in fixes that without giving up the
        dirty-rectangle repaint: the clear becomes a blit of the
        corresponding piece of the image, aligned to where the window's
        own stylesheet paints it, and the strips switch to per-pixel
        alpha so the glyphs composite over the picture instead of over
        a colour baked into them.

        The cost is the one the module docstring quantifies: a
        translucent strip is roughly 25x more expensive to blit than an
        opaque one, so this path is used *only* when there is a picture
        to show. ``set_backdrop(None)`` puts the fast path back.

        :param source: a path, a ``QPixmap``, a ``QImage`` or ``None``.
        """
        self._backdrop = _as_pixmap(source)
        self._invalidate_strips()
        self.update()

    def _backdrop_origin(self) -> QPoint:
        """Top-left of the backdrop in this widget's own coordinates.

        The window paints its wallpaper centred on itself
        (``background-position: center center`` in the QSS, which does
        not repeat), and the rain has to land on exactly the same
        pixels or the picture visibly jumps at the widget's edge. So:
        centre the image on the *window*, then subtract where this
        widget sits inside it.
        """
        pixmap = self._backdrop
        # `window()` is the widget itself when it has no parent, never
        # None, so a parentless rain centres the image on itself.
        window = self.window()
        x = (window.width() - pixmap.width()) // 2
        y = (window.height() - pixmap.height()) // 2
        offset = self.mapTo(window, QPoint(0, 0))
        return QPoint(x - offset.x(), y - offset.y())

    def _clear(self, painter: QPainter, rect: QRect) -> None:
        """Reset ``rect`` to whatever sits *under* the glyphs.

        The flat colour, or the matching piece of the backdrop. Any part
        of ``rect`` the backdrop does not reach — a window wider than
        the wallpaper — still gets the flat colour, so the widget stays
        fully opaque and ``WA_OpaquePaintEvent`` remains honest.
        """
        pixmap = self._backdrop
        if pixmap is None:
            painter.fillRect(rect, self._bg)
            return
        origin = self._backdrop_origin()
        covered = rect.intersected(
            QRect(origin.x(), origin.y(), pixmap.width(), pixmap.height()))
        if covered != rect:
            painter.fillRect(rect, self._bg)
        if not covered.isEmpty():
            painter.drawPixmap(covered, pixmap,
                               covered.translated(-origin.x(), -origin.y()))

    def set_opacity(self, value: float) -> None:
        """Set glyph alpha in ``0..1``. Low keeps content in front legible."""
        self._opacity = max(0.0, min(1.0, float(value)))
        self._rebuild_pens()
        self.update()

    def apply_theme(self, theme: str) -> None:
        """Re-take the colours from ``theme``'s palette.

        Opt-in, and it overrides the shipped teal with the theme accent
        — which is the Run button's blue. Nothing in the app calls it;
        the screen pushes only ``set_background_color`` on a theme
        switch, precisely so a colour choice survives one.
        """
        palette = palette_for(theme)
        self.set_background_color(palette["bg"])
        self.set_color(palette["accent"])

    # -- simulation knobs ----------------------------------------------
    def font_size(self) -> int:
        return self._engine.font_size

    def set_font_size(self, px: int) -> None:
        """Set the glyph size, which re-lays-out the columns."""
        self._sync_size()
        px = _clamp_int(px, MIN_FONT_PX, MAX_FONT_PX)
        if px == self._engine.font_size:
            return
        self._engine.set_font_size(px)
        self._rebuild_font()
        self.update()

    def speed(self) -> float:
        return self._engine.speed_multiplier

    def set_speed(self, factor: float) -> None:
        """Scale every column's speed. Relative rates are preserved, so
        the columns stay as asynchronous as they were."""
        self._engine.set_speed_multiplier(factor)

    def set_fps(self, fps: int) -> None:
        """Cap the frame rate."""
        self._fps = _clamp_int(fps, MIN_FPS, MAX_FPS)
        self._timer.setInterval(max(1, 1000 // self._fps))

    def fps(self) -> int:
        return self._fps

    @property
    def engine(self) -> DnaRainEngine:
        return self._engine

    # -- run state -----------------------------------------------------
    def is_running(self) -> bool:
        """True while the animation timer is ticking."""
        return self._timer.isActive()

    def start(self) -> None:
        """Start ticking (no-op if already running)."""
        if not self._timer.isActive():
            self._clock.restart()
            self._timer.start()

    def stop(self) -> None:
        """Stop ticking. Costs exactly nothing while stopped."""
        self._timer.stop()

    def _should_run(self) -> bool:
        if not self.isVisible():
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
        self._sync_size()
        self._sync_run_state()

    def hideEvent(self, event):
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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._sync_size()
        self.update()

    def _sync_size(self) -> None:
        """Match the engine to the widget's geometry.

        Qt defers the resize event of a hidden widget until it is shown,
        so relying on :meth:`resizeEvent` alone would leave a rain that
        was sized before it was on screen laying out for the wrong
        canvas. Checking here costs one tuple comparison a frame.
        """
        if self._engine.resize(self.width(), self.height()):
            self._invalidate_strips()

    # -- animation -----------------------------------------------------
    def _on_tick(self) -> None:
        dt = self._clock.restart() / 1000.0
        self.advance_frame(min(MAX_DT, dt) if dt > 0 else 1.0 / self._fps)

    def advance_frame(self, dt: float) -> List[QRect]:
        """Step the simulation and schedule repaints of what changed.

        Called by the timer, and directly by the tests so the animation
        never depends on a real clock.

        :param dt: elapsed seconds.
        :returns: the rectangles that were invalidated. Empty when
            nothing moved (no repaint is scheduled at all) or when the
            frame fell back to a single full repaint — see
            :attr:`last_full_repaint`.
        """
        self.last_full_repaint = False
        self._sync_size()
        spans = self._engine.advance(dt)
        if not spans:
            return []
        rects = self._coalesce(spans)
        if len(rects) > MAX_DIRTY_RECTS:
            self.last_full_repaint = True
            self.update()
            return []
        for rect in rects:
            self.update(rect)
        return rects

    def _coalesce(self, spans: List[Tuple[int, int, int]]) -> List[QRect]:
        """Merge dirty spans of adjacent columns into single rectangles.

        ``spans`` arrives in column order, so this is one pass.
        """
        cell = self._engine.cell_size
        columns = self._engine.columns
        rects: List[QRect] = []
        first_col = last_col = top = bottom = 0
        extra = 0
        open_run = False

        def close():
            width = (last_col - first_col + 1) * cell + extra
            rects.append(QRect(first_col * cell, top * cell, width,
                               (bottom - top + 1) * cell))

        for index, span_top, span_bottom in spans:
            # Every column is exactly one cell wide now. The spaCR splice used
            # to make its column wider than its stride, because the word was
            # drawn horizontally out of a single cell and bled over its
            # neighbours; five one-letter cells cannot.
            over = 0
            if open_run and index == last_col + 1:
                last_col = index
                top = min(top, span_top)
                bottom = max(bottom, span_bottom)
                extra = max(extra, over)
                continue
            if open_run:
                close()
            first_col = last_col = index
            top, bottom, extra = span_top, span_bottom, over
            open_run = True
        if open_run:
            close()
        return rects

    # -- painting ------------------------------------------------------
    def _rebuild_font(self) -> None:
        font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        font.setPixelSize(max(MIN_FONT_PX, self._engine.font_size))
        font.setBold(True)
        self._font = font
        metrics = QFontMetricsF(font)
        self._ascent = int(math.ceil(metrics.ascent()))
        self._invalidate_strips()

    def _pens_for(self, base: QColor) -> Tuple[List[QPen], QPen, QPen]:
        """Every pen a strip render can need, for one trail colour.

        Building QColors per glyph is the fastest way to make this
        expensive, so the trail alpha ramp is a small LUT.

        :returns: ``(trail LUT, head pen, highlight pen)``.
        """
        trail = []
        for step in range(TRAIL_STEPS):
            frac = step / max(1, TRAIL_STEPS - 1)
            alpha = self._opacity * (MIN_TAIL_ALPHA
                                     + (1.0 - MIN_TAIL_ALPHA) * frac)
            color = QColor(base)
            color.setAlphaF(max(0.0, min(1.0, alpha)))
            trail.append(QPen(color))
        head_color = derive_head_color(base, self._bg)
        head = QColor(head_color)
        head.setAlphaF(min(1.0, self._opacity * HEAD_ALPHA_BOOST))
        highlight = blend(base, head_color, 0.5)
        highlight.setAlphaF(min(1.0, self._opacity * 2.0))
        return trail, QPen(head), QPen(highlight)

    def _rebuild_pens(self) -> None:
        """Recompute the shared pens after a colour/opacity change."""
        (self._trail_pens, self._head_pen,
         self._hi_pen) = self._pens_for(self._color)
        self._hue_pens = {}
        self._invalidate_strips()

    def _pens_for_column(self, column: Column) -> Tuple[List[QPen],
                                                        QPen, QPen]:
        """The pens for one column — shared, or its own random hue."""
        if not self._random_colors:
            return self._trail_pens, self._head_pen, self._hi_pen
        key = _hue_bucket(column.hue)
        pens = self._hue_pens.get(key)
        if pens is None:
            pens = self._pens_for(random_hue_color(self._color, key / 360.0))
            self._hue_pens[key] = pens
        return pens

    def _invalidate_strips(self) -> None:
        """Drop the pre-rendered strips; they re-render on next paint."""
        self._style_gen += 1
        self._strips = []
        self._strip_keys = []

    def _render_strip(self, column: Column) -> QPixmap:
        """Draw one whole string into an opaque pixmap.

        Opaque matters: the trail alphas are composited against the
        background *here*, once per respawn, so the per-frame blit is a
        straight copy instead of an alpha blend. That is the difference
        between 0.46 ms and 12 ms for a full canvas.

        With a backdrop set there is nothing constant to bake the alphas
        against — the picture under a string changes as the string
        falls — so the strip keeps its own transparency and the blend
        happens at blit time. That is the expensive path, and it is
        taken only by the themes that have a wallpaper to show.
        """
        cell = self._engine.cell_size
        strip = QPixmap(cell, max(1, column.length * cell))
        strip.fill(Qt.transparent if self._backdrop is not None else self._bg)
        painter = QPainter(strip)
        painter.setFont(self._font)
        fade_cells = max(1, int(column.length * TAIL_FADE_FRACTION))
        top_step = TRAIL_STEPS - 1
        head_index = column.length - 1
        trail_pens, head_pen, hi_pen = self._pens_for_column(column)
        for i, token in enumerate(column.tokens):
            if len(token) > 1:
                # The word is wider than a cell, so it is drawn live at
                # full canvas width instead of baked into this strip.
                continue
            if i == head_index:
                painter.setPen(head_pen)
            elif column.hi_start <= i < column.hi_end:
                painter.setPen(hi_pen)
            else:
                painter.setPen(
                    trail_pens[min(top_step,
                                   int(i / fade_cells * top_step))])
            painter.drawText(0, i * cell + self._ascent, token)
        painter.end()
        return strip

    def _strip_for(self, index: int) -> QPixmap:
        """The cached strip for column ``index``, rendering it if stale."""
        count = self._engine.n_columns
        if len(self._strips) != count:
            self._strips = [None] * count
            self._strip_keys = [(-1, -1)] * count
        column = self._engine.columns[index]
        key = (column.generation, self._style_gen)
        strip = self._strips[index]
        if strip is None or self._strip_keys[index] != key:
            strip = self._render_strip(column)
            self._strips[index] = strip
            self._strip_keys[index] = key
        return strip

    def paintEvent(self, event):
        self._sync_size()
        painter = QPainter(self)
        engine = self._engine
        cell = engine.cell_size
        rects = _region_rects(event.region(), event.rect())

        # Pass 1: clear only the rectangles Qt asked for. Everything
        # else in the backing store is still valid.
        #
        # One fillRect per region rectangle beats clearing per column
        # around each string: at 120 columns the per-call overhead of
        # 240 small fills costs about 2 ms more than the ~0.1 ms of
        # pixels they save. Measured, twice, in both directions.
        touched = set()
        last = engine.n_columns - 1
        for rect in rects:
            self._clear(painter, rect)
            # No left-hand margin needed. This used to reach extra columns
            # leftward because a spaCR splice over there drew across into this
            # one; the word is now confined to its own column.
            first = max(0, rect.left() // cell)
            touched.update(range(first, min(last, rect.right() // cell) + 1))
        if not touched or engine.n_rows <= 0:
            return

        # Pass 2: blit each touched string. Qt has already clipped the
        # painter to the region, so nothing outside it is written.
        order = sorted(touched)
        for index in order:
            column = engine.columns[index]
            painter.drawPixmap(index * cell, column.y_px,
                               self._strip_for(index))

        # There is no second pass. The spaCR splice used to need one: the word
        # lived in a single cell, was drawn horizontally, and had to clear a
        # backing rectangle that ran across its neighbouring columns — so it
        # had to come last, after every strip was down. Now it is five
        # ordinary one-letter cells inside its own column's strip, so it is
        # rendered by the loop above like any other glyph, cached like any
        # other glyph, and cannot overdraw anything.


def _effective_theme() -> str:
    """The user's resolved theme, or dark when preferences are unavailable."""
    try:
        from ..preferences import resolve_effective_theme
        return resolve_effective_theme()
    except Exception:
        return "dark"


# ---------------------------------------------------------------------------
# Settings bar
# ---------------------------------------------------------------------------

class DnaRainSettingsBar(QWidget):
    """Colour / speed / visibility / font-size controls for a rain widget.

    Everything applies live — :meth:`bind` wires the signals straight at
    the rain widget's setters, no restart involved.

    Two layouts, one set of controls. ``vertical=True`` puts them in a
    label/control/readout grid, which is the shape a popover wants; the
    default row is the original bar. The controls, the state and the
    signals are identical either way — only the geometry differs.

    :param vertical: lay the controls out as a grid instead of a row.
    """

    color_changed = Signal(QColor)
    speed_changed = Signal(float)
    font_size_changed = Signal(int)
    opacity_changed = Signal(float)
    random_color_changed = Signal(bool)

    def __init__(self, parent: Optional[QWidget] = None, *,
                 color: Union[QColor, str, None] = None,
                 speed: float = 1.0,
                 font_size: int = DEFAULT_FONT_PX,
                 opacity: float = DEFAULT_OPACITY,
                 random_color: bool = False,
                 vertical: bool = False,
                 theme: Optional[str] = None):
        super().__init__(parent)
        self._color = _as_color(color, QColor(DEFAULT_COLOR))
        self._rain: Optional[DnaRainWidget] = None

        # The rain is painted behind this bar, and the global
        # ``QWidget { background }`` rule does not reach a widget that
        # has its own stylesheet — so give the bar an opaque surface of
        # its own or its labels land on falling glyphs.
        self.setObjectName("DnaRainBar")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.restyle_for_theme(theme)

        self._swatch = QPushButton()
        self._swatch.setObjectName("DnaRainSwatch")
        self._swatch.setFixedSize(44, 20)
        self._swatch.setToolTip("Pick the DNA rain colour")
        self._swatch.clicked.connect(self.pick_color)
        self._paint_swatch()

        self._random = Toggle("Random")
        self._random.setToolTip(
            "Give every falling string its own colour. The picked colour "
            "still sets how vivid and how bright they are; only the hue is "
            "random, and each string takes a new one every time it restarts "
            "at the top.")
        self._random.setChecked(bool(random_color))
        self._random.toggled.connect(self._on_random)

        self._speed = QSlider(Qt.Horizontal)
        self._speed.setToolTip("Scales every column; they keep their "
                               "individual rates")
        self._speed.setRange(int(MIN_SPEED_MULTIPLIER * 100),
                             int(MAX_SPEED_MULTIPLIER * 100))
        self._speed.setValue(int(max(MIN_SPEED_MULTIPLIER,
                                     min(MAX_SPEED_MULTIPLIER,
                                         float(speed))) * 100))
        self._speed.setFixedWidth(120)
        self._speed.valueChanged.connect(self._on_speed)
        self._speed_value = _muted_label("")

        self._opacity = QSlider(Qt.Horizontal)
        self._opacity.setToolTip(
            "How strongly the rain shows through behind the screen content. "
            "Higher is more visible; too high and the settings in front of it "
            "get harder to read.")
        self._opacity.setRange(MIN_OPACITY_PCT, MAX_OPACITY_PCT)
        self._opacity.setValue(_clamp_int(round(opacity * 100),
                                          MIN_OPACITY_PCT, MAX_OPACITY_PCT))
        self._opacity.setFixedWidth(120)
        self._opacity.valueChanged.connect(self._on_opacity)
        self._opacity_value = _muted_label("")

        self._font = QSlider(Qt.Horizontal)
        self._font.setToolTip("Glyph size, which is also the column stride")
        self._font.setRange(MIN_FONT_PX, MAX_FONT_PX)
        self._font.setValue(_clamp_int(font_size, MIN_FONT_PX, MAX_FONT_PX))
        self._font.setFixedWidth(120)
        self._font.valueChanged.connect(self._on_font)
        self._font_value = _muted_label("")

        if vertical:
            self._build_grid()
        else:
            self._build_row()
        self._refresh_readouts()

    def restyle_for_theme(self, theme: Optional[str] = None) -> None:
        """Re-take this panel's own surface colour from a theme's palette.

        The bar states its own background, so unlike the rest of the
        screen it is NOT re-styled by re-applying the application
        stylesheet — it would keep the dark theme's surface behind
        freshly light text. The popover calls this every time it opens,
        which is the only moment the panel is on screen.

        Only the chrome. The user's chosen colour is never touched.

        :param theme: palette to take; defaults to the effective theme.
        """
        palette = palette_for(theme or _effective_theme())
        self.setStyleSheet(
            f"QWidget#DnaRainBar {{ background: {palette['surface']};"
            f" border: 1px solid {palette['border_soft']};"
            f" border-radius: {RADIUS['sm']}px; }}")

    def _build_row(self) -> None:
        """The original one-line bar."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACING["md"], SPACING["sm"],
                                  SPACING["md"], SPACING["sm"])
        layout.setSpacing(SPACING["md"])
        layout.addWidget(_muted_label("DNA rain"))
        layout.addWidget(_muted_label("Colour"))
        layout.addWidget(self._swatch)
        layout.addWidget(self._random)
        layout.addWidget(_muted_label("Speed"))
        layout.addWidget(self._speed)
        layout.addWidget(self._speed_value)
        layout.addWidget(_muted_label("Visibility"))
        layout.addWidget(self._opacity)
        layout.addWidget(self._opacity_value)
        layout.addWidget(_muted_label("Font"))
        layout.addWidget(self._font)
        layout.addWidget(self._font_value)
        layout.addStretch(1)

    def _build_grid(self) -> None:
        """Label / control / readout, one setting per line."""
        grid = QGridLayout(self)
        grid.setContentsMargins(SPACING["md"], SPACING["md"],
                                SPACING["md"], SPACING["md"])
        grid.setHorizontalSpacing(SPACING["md"])
        grid.setVerticalSpacing(SPACING["sm"])

        title = _muted_label("DNA rain")
        grid.addWidget(title, 0, 0, 1, 3)

        grid.addWidget(_muted_label("Colour"), 1, 0)
        colour_row = QHBoxLayout()
        colour_row.setContentsMargins(0, 0, 0, 0)
        colour_row.setSpacing(SPACING["sm"])
        colour_row.addWidget(self._swatch)
        colour_row.addStretch(1)
        grid.addLayout(colour_row, 1, 1)
        grid.addWidget(self._random, 1, 2)

        for row, (name, control, readout) in enumerate((
                ("Speed", self._speed, self._speed_value),
                ("Visibility", self._opacity, self._opacity_value),
                ("Font size", self._font, self._font_value)), start=2):
            grid.addWidget(_muted_label(name), row, 0)
            grid.addWidget(control, row, 1)
            grid.addWidget(readout, row, 2)

    # -- state ---------------------------------------------------------
    def color(self) -> QColor:
        return QColor(self._color)

    def speed(self) -> float:
        return self._speed.value() / 100.0

    def font_size(self) -> int:
        return self._font.value()

    def opacity(self) -> float:
        return self._opacity.value() / 100.0

    def random_color(self) -> bool:
        """True when the rain is set to colour each column separately."""
        return self._random.isChecked()

    def _paint_swatch(self) -> None:
        self._swatch.setStyleSheet(
            "QPushButton#DnaRainSwatch {"
            f" background: {self._color.name()};"
            " border: 1px solid rgba(255,255,255,0.35);"
            " border-radius: 3px; }")

    def _refresh_readouts(self) -> None:
        self._speed_value.setText(f"{self.speed():.1f}x")
        self._font_value.setText(f"{self.font_size()} px")
        self._opacity_value.setText(f"{round(self.opacity() * 100)}%")

    # -- controls ------------------------------------------------------
    def set_color(self, color: Union[QColor, str]) -> None:
        """Set the colour and emit :attr:`color_changed`."""
        self._color = _as_color(color, self._color)
        self._paint_swatch()
        self.color_changed.emit(QColor(self._color))

    def pick_color(self) -> None:
        """Open the colour picker; keep the current colour on cancel."""
        chosen = QColorDialog.getColor(self._color, self, "DNA rain colour")
        if chosen.isValid():
            self.set_color(chosen)

    def set_speed(self, factor: float) -> None:
        self._speed.setValue(int(round(float(factor) * 100)))

    def set_font_size(self, px: int) -> None:
        self._font.setValue(_clamp_int(px, MIN_FONT_PX, MAX_FONT_PX))

    def set_random_color(self, on: bool) -> None:
        """Turn per-column colour on or off; emits only on a real change."""
        self._random.setChecked(bool(on))

    def _on_random(self, on: bool) -> None:
        self.random_color_changed.emit(bool(on))

    def _on_speed(self, _value: int) -> None:
        self._refresh_readouts()
        self.speed_changed.emit(self.speed())

    def set_opacity(self, value: float) -> None:
        self._opacity.setValue(
            _clamp_int(round(float(value) * 100), MIN_OPACITY_PCT,
                       MAX_OPACITY_PCT))

    def _on_font(self, value: int) -> None:
        self._refresh_readouts()
        self.font_size_changed.emit(int(value))

    def _on_opacity(self, _value: int) -> None:
        self._refresh_readouts()
        self.opacity_changed.emit(self.opacity())

    # -- wiring --------------------------------------------------------
    def bind(self, rain: DnaRainWidget) -> None:
        """Drive ``rain`` from this bar, and seed the bar from the rain."""
        self._rain = rain
        self._color = rain.color()
        self._paint_swatch()
        self._speed.blockSignals(True)
        self._speed.setValue(int(round(rain.speed() * 100)))
        self._speed.blockSignals(False)
        self._font.blockSignals(True)
        self._font.setValue(rain.font_size())
        self._font.blockSignals(False)
        self._opacity.blockSignals(True)
        self._opacity.setValue(
            _clamp_int(round(rain.opacity() * 100), MIN_OPACITY_PCT,
                       MAX_OPACITY_PCT))
        self._opacity.blockSignals(False)
        self._random.blockSignals(True)
        self._random.setChecked(rain.random_colors())
        self._random.blockSignals(False)
        self._refresh_readouts()
        self.color_changed.connect(rain.set_color)
        self.speed_changed.connect(rain.set_speed)
        self.font_size_changed.connect(rain.set_font_size)
        self.opacity_changed.connect(rain.set_opacity)
        self.random_color_changed.connect(rain.set_random_colors)


def _muted_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("Muted")
    return label


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

def install_dna_rain(host: QWidget, layout=None, *,
                     anchor: Optional[QWidget] = None,
                     **kwargs) -> DnaRainWidget:
    """Put a live DNA rain behind ``host``, and a DNA button in the chrome.

    The rain becomes a child of ``host``, tracks its geometry, and is
    lowered to the bottom of the sibling stacking order so every screen
    widget paints in front of it. It takes no focus and no mouse events.

    The controls are **not** placed on the screen. They live in a
    popover behind a ``DNA`` toggle built from the same class as the
    ``AI`` toggle beside it; a decorative backdrop does not get to keep
    a permanent strip of a screen whose job is a settings form.

    Where the button lands, in order: beside ``anchor`` if one is
    given; else beside the host's own ``AI`` toggle, which is the row
    this control belongs in and the reason no caller has to say so;
    else appended to ``layout``; else nowhere, and the caller places
    ``rain.settings_button`` itself.

    :param host: the screen the rain sits behind.
    :param layout: optional layout to append the DNA button to. Used
        only when there is no anchor to sit beside.
    :param anchor: widget to sit beside — the button is inserted into
        ``anchor``'s layout immediately before it. Defaults to the AI
        toggle found under ``host``.
    :param kwargs: forwarded to :class:`DnaRainWidget`. Pass
        ``backdrop=<wallpaper path>`` on an image theme so the rain
        shows the picture through itself rather than replacing it.
    :returns: the rain widget, with ``.settings_bar``,
        ``.settings_button`` and ``.settings_popover`` attached.
    """
    rain = DnaRainWidget(host, **kwargs)
    rain.follow_parent()
    rain.show()
    # No parent: the popover adopts it. Parenting it to `host` first
    # would flash a settings bar across the screen for one event loop.
    bar = DnaRainSettingsBar(None, theme=kwargs.get("theme"), vertical=True)
    bar.bind(rain)
    # Imported here rather than at module scope: the popover module
    # imports this one for the bar, so a top-level import either way
    # round is a cycle.
    from .dna_rain_settings import DnaSettingsButton
    button = DnaSettingsButton(bar, parent=host)
    if not _place_beside(button, anchor or _find_ai_toggle(host)):
        if layout is not None:
            layout.addWidget(button)
    rain.settings_bar = bar
    rain.settings_button = button
    rain.settings_popover = button.popover
    return rain


def _find_ai_toggle(host: QWidget) -> Optional[QWidget]:
    """The host's ``AI`` toggle, or ``None`` if it has not got one.

    Matched on the untranslated source text every ``AiToggleLabel``
    keeps in ``_spacr_i18n_text``, not on what is on screen: the label
    is translated, and matching the visible text would have put the
    button in the right place in English and nowhere in Swedish. The
    class is shared with the ``Live`` and hyperparameter switches, so
    the text is what distinguishes them.

    Never raises: a host with no such toggle simply gets the fallback
    placement.
    """
    try:
        from .ai_toggle_label import AiToggleLabel
        for label in host.findChildren(AiToggleLabel):
            if label.property("_spacr_i18n_text") == "AI":
                return label
    except Exception:
        pass
    return None


def _place_beside(button: QWidget, anchor: Optional[QWidget]) -> bool:
    """Insert ``button`` immediately before ``anchor`` in its own layout.

    Before, not after: the AI toggle is followed by its provider
    chevron, and splitting that pair would read as the chevron belonging
    to DNA.

    :returns: True when the button was placed.
    """
    if anchor is None:
        return False
    parent = anchor.parentWidget()
    layout = parent.layout() if parent is not None else None
    if layout is None:
        return False
    index = layout.indexOf(anchor)
    if index < 0:
        return False
    try:
        layout.insertWidget(index, button)
    except AttributeError:
        # Not a box layout — better beside nothing than not at all.
        layout.addWidget(button)
    return True
