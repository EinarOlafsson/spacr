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
Very infrequently a column carries the literal word ``spaCR``. It is
stored as a *single token* occupying a single cell and drawn
**horizontally**, overflowing to the right across its neighbours, so it
reads left-to-right with exactly that casing. That is why nothing in the
renderer may assume one character per cell: token widths are measured,
dirty rectangles are widened for a column that carries a word, and a
backing rectangle is cleared behind it so neighbouring glyphs do not
tangle with the letters. See :data:`SPACR_SPLICE_PROBABILITY` for the
rate and what it works out to in practice.

Legibility
----------
Screen content sits in front of this widget. It therefore never takes
focus, is transparent to mouse events, lowers itself to the bottom of
the sibling stacking order, and paints its glyphs at
:data:`DEFAULT_OPACITY` over the theme background so anything in front
of it stays readable.

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
3. *Only the columns that moved are repainted.* Positions are
   quantised to whole cells, so a column is dirty only when its
   integer row changes — slow columns cost nothing on most ticks. See
   :data:`MAX_DIRTY_RECTS` for where partial repaints stop paying.

Together: a median frame of 0.6 ms and 4.4 % of one core at 1920x1080
and 24 fps, and 0.00 % while off screen.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from PySide6.QtCore import QElapsedTimer, QEvent, QRect, Qt, QTimer, Signal
from PySide6.QtGui import (QColor, QFont, QFontDatabase, QFontMetricsF,
                           QPainter, QPen, QPixmap)
from PySide6.QtWidgets import (QColorDialog, QHBoxLayout, QLabel, QPushButton,
                               QSizePolicy, QSlider, QWidget)

from ..theme import RADIUS, SPACING, palette_for

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The whole alphabet. ATGC, uppercase, nothing else.
BASES = ("A", "T", "G", "C")

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

#: Frame-rate cap. 24 fps is plenty for falling text and leaves the
#: pipeline the other 97 % of the tick.
DEFAULT_FPS = 24
MIN_FPS = 1
MAX_FPS = 60

#: Glyph opacity. Low by default — screen content sits in front.
DEFAULT_OPACITY = 0.22

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
    :ivar row: ``floor(head)`` as painted — the column is only dirty
        when *this* changes, which is what keeps slow columns cheap.
    :ivar hi_start: first cell index of the highlighted run.
    :ivar hi_end: one past the last cell index of the highlighted run.
    :ivar word_index: cell index of the multi-character token, or -1.
    :ivar generation: bumped on every respawn; the widget's pre-rendered
        strip cache keys off it.
    """

    tokens: List[str]
    length: int
    speed: float
    head: float
    row: int
    hi_start: int
    hi_end: int
    word_index: int
    generation: int = 0

    @property
    def has_word(self) -> bool:
        """True when this string carries a multi-character token."""
        return self.word_index >= 0


class DnaRainEngine:
    """Qt-free simulation of the falling columns.

    Deterministic: the same ``seed`` and the same sequence of calls
    always produce the same animation.

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

        :returns: ``(tokens, word index)``; the index is -1 when no
            splice happened. A spliced token replaces one base and
            therefore still occupies exactly one *cell* — it is simply
            five characters wide when drawn.
        """
        rng = self._rng
        tokens = [rng.choice(BASES) for _ in range(length)]
        word_index = -1
        if rng.random() < self.spacr_probability:
            word_index = rng.randrange(length)
            tokens[word_index] = SPACR_TOKEN
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
        if column is None:
            return Column(tokens=tokens, length=length, speed=speed,
                          head=head, row=int(math.floor(head)),
                          hi_start=hi_start, hi_end=hi_end,
                          word_index=word_index)
        column.tokens = tokens
        column.length = length
        column.speed = speed
        column.head = head
        column.row = int(math.floor(head))
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
            if new_row == old_row and not respawned:
                continue
            column.row = new_row
            top = max(0, min(old_top, new_row - column.length + 1))
            bottom = min(rows - 1, max(old_row, new_row))
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
    :param color: trail colour; defaults to the theme accent.
    :param background: colour painted under the glyphs; defaults to the
        theme background.
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
                 opacity: float = DEFAULT_OPACITY,
                 spacr_probability: float = SPACR_SPLICE_PROBABILITY,
                 theme: Optional[str] = None):
        super().__init__(parent)
        palette = palette_for(theme or _effective_theme())
        self._bg = _as_color(background, QColor(palette["bg"]))
        self._bg.setAlpha(255)
        self._color = _as_color(color, QColor(palette["accent"]))
        self._opacity = max(0.0, min(1.0, float(opacity)))

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
        self._word_px = 0
        self._word_cols = 1
        self._trail_pens: List[QPen] = []
        self._head_pen = QPen()
        self._hi_pen = QPen()
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

    def set_color(self, color: Union[QColor, str]) -> None:
        """Set the trail colour; the head colour re-derives from it."""
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

    def set_opacity(self, value: float) -> None:
        """Set glyph alpha in ``0..1``. Low keeps content in front legible."""
        self._opacity = max(0.0, min(1.0, float(value)))
        self._rebuild_pens()
        self.update()

    def apply_theme(self, theme: str) -> None:
        """Re-take the colour defaults from ``theme``'s palette."""
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
            # A spliced column is wider than its stride: the word runs
            # horizontally past its neighbours.
            over = max(0, self._word_px - cell) if columns[index].has_word \
                else 0
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
        self._word_px = int(math.ceil(metrics.horizontalAdvance(SPACR_TOKEN)))
        cell = max(1, self._engine.cell_size)
        self._word_cols = max(1, int(math.ceil(self._word_px / cell)))
        self._invalidate_strips()

    def _rebuild_pens(self) -> None:
        """Precompute every pen a strip render can need.

        Building QColors per glyph is the fastest way to make this
        expensive, so the trail alpha ramp is a small LUT.
        """
        self._trail_pens = []
        for step in range(TRAIL_STEPS):
            frac = step / max(1, TRAIL_STEPS - 1)
            alpha = self._opacity * (MIN_TAIL_ALPHA
                                     + (1.0 - MIN_TAIL_ALPHA) * frac)
            color = QColor(self._color)
            color.setAlphaF(max(0.0, min(1.0, alpha)))
            self._trail_pens.append(QPen(color))
        head = self.head_color()
        head.setAlphaF(min(1.0, self._opacity * HEAD_ALPHA_BOOST))
        self._head_pen = QPen(head)
        highlight = blend(self._color, self.head_color(), 0.5)
        highlight.setAlphaF(min(1.0, self._opacity * 2.0))
        self._hi_pen = QPen(highlight)
        self._invalidate_strips()

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
        """
        cell = self._engine.cell_size
        strip = QPixmap(cell, max(1, column.length * cell))
        strip.fill(self._bg)
        painter = QPainter(strip)
        painter.setFont(self._font)
        fade_cells = max(1, int(column.length * TAIL_FADE_FRACTION))
        top_step = TRAIL_STEPS - 1
        head_index = column.length - 1
        for i, token in enumerate(column.tokens):
            if len(token) > 1:
                # The word is wider than a cell, so it is drawn live at
                # full canvas width instead of baked into this strip.
                continue
            if i == head_index:
                painter.setPen(self._head_pen)
            elif column.hi_start <= i < column.hi_end:
                painter.setPen(self._hi_pen)
            else:
                painter.setPen(
                    self._trail_pens[min(top_step,
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
            painter.fillRect(rect, self._bg)
            # Reach a few columns further left than the region starts:
            # a spaCR splice over there draws across into this one.
            first = max(0, rect.left() // cell - self._word_cols)
            touched.update(range(first, min(last, rect.right() // cell) + 1))
        if not touched or engine.n_rows <= 0:
            return

        # Pass 2: blit each touched string. Qt has already clipped the
        # painter to the region, so nothing outside it is written.
        order = sorted(touched)
        for index in order:
            column = engine.columns[index]
            painter.drawPixmap(index * cell,
                               (column.row - column.length + 1) * cell,
                               self._strip_for(index))

        # Pass 2: the spaCR splices, last, because each one clears a
        # backing rectangle that runs across its neighbours.
        painter.setFont(self._font)
        painter.setPen(self._head_pen)
        for index in order:
            column = engine.columns[index]
            if not column.has_word:
                continue
            x = index * cell
            y = (column.row - column.length + 1 + column.word_index) * cell
            painter.fillRect(QRect(x, y, self._word_px, cell), self._bg)
            painter.drawText(x, y + self._ascent, SPACR_TOKEN)


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
    """Colour / speed / font-size controls for a :class:`DnaRainWidget`.

    Everything applies live — :meth:`bind` wires the three signals
    straight at the rain widget's setters, no restart involved.
    """

    color_changed = Signal(QColor)
    speed_changed = Signal(float)
    font_size_changed = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None, *,
                 color: Union[QColor, str, None] = None,
                 speed: float = 1.0,
                 font_size: int = DEFAULT_FONT_PX,
                 theme: Optional[str] = None):
        super().__init__(parent)
        palette = palette_for(theme or _effective_theme())
        self._color = _as_color(color, QColor(palette["accent"]))
        self._rain: Optional[DnaRainWidget] = None

        # The rain is painted behind this bar, and the global
        # ``QWidget { background }`` rule does not reach a widget that
        # has its own stylesheet — so give the bar an opaque surface of
        # its own or its labels land on falling glyphs.
        self.setObjectName("DnaRainBar")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            f"QWidget#DnaRainBar {{ background: {palette['surface']};"
            f" border: 1px solid {palette['border_soft']};"
            f" border-radius: {RADIUS['sm']}px; }}")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACING["md"], SPACING["sm"],
                                  SPACING["md"], SPACING["sm"])
        layout.setSpacing(SPACING["md"])

        layout.addWidget(_muted_label("DNA rain"))
        layout.addWidget(_muted_label("Colour"))
        self._swatch = QPushButton()
        self._swatch.setObjectName("DnaRainSwatch")
        self._swatch.setFixedSize(44, 20)
        self._swatch.setToolTip("Pick the DNA rain colour")
        self._swatch.clicked.connect(self.pick_color)
        self._paint_swatch()
        layout.addWidget(self._swatch)

        layout.addWidget(_muted_label("Speed"))
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
        layout.addWidget(self._speed)
        self._speed_value = _muted_label("")
        layout.addWidget(self._speed_value)

        layout.addWidget(_muted_label("Font"))
        self._font = QSlider(Qt.Horizontal)
        self._font.setToolTip("Glyph size, which is also the column stride")
        self._font.setRange(MIN_FONT_PX, MAX_FONT_PX)
        self._font.setValue(_clamp_int(font_size, MIN_FONT_PX, MAX_FONT_PX))
        self._font.setFixedWidth(120)
        self._font.valueChanged.connect(self._on_font)
        layout.addWidget(self._font)
        self._font_value = _muted_label("")
        layout.addWidget(self._font_value)
        layout.addStretch(1)

        self._refresh_readouts()

    # -- state ---------------------------------------------------------
    def color(self) -> QColor:
        return QColor(self._color)

    def speed(self) -> float:
        return self._speed.value() / 100.0

    def font_size(self) -> int:
        return self._font.value()

    def _paint_swatch(self) -> None:
        self._swatch.setStyleSheet(
            "QPushButton#DnaRainSwatch {"
            f" background: {self._color.name()};"
            " border: 1px solid rgba(255,255,255,0.35);"
            " border-radius: 3px; }")

    def _refresh_readouts(self) -> None:
        self._speed_value.setText(f"{self.speed():.1f}x")
        self._font_value.setText(f"{self.font_size()} px")

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

    def _on_speed(self, _value: int) -> None:
        self._refresh_readouts()
        self.speed_changed.emit(self.speed())

    def _on_font(self, value: int) -> None:
        self._refresh_readouts()
        self.font_size_changed.emit(int(value))

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
        self._refresh_readouts()
        self.color_changed.connect(rain.set_color)
        self.speed_changed.connect(rain.set_speed)
        self.font_size_changed.connect(rain.set_font_size)


def _muted_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("Muted")
    return label


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

def install_dna_rain(host: QWidget, layout=None, **kwargs) -> DnaRainWidget:
    """Put a live DNA rain behind ``host`` and its controls at the bottom.

    The rain becomes a child of ``host``, tracks its geometry, and is
    lowered to the bottom of the sibling stacking order so every screen
    widget paints in front of it. It takes no focus and no mouse events.

    :param host: the screen the rain sits behind.
    :param layout: optional layout to append the settings bar to; when
        ``None`` the bar is created but not placed, and is reachable as
        ``rain.settings_bar``.
    :param kwargs: forwarded to :class:`DnaRainWidget`.
    :returns: the rain widget, with ``.settings_bar`` attached.
    """
    rain = DnaRainWidget(host, **kwargs)
    rain.follow_parent()
    rain.show()
    bar = DnaRainSettingsBar(host, theme=kwargs.get("theme"))
    bar.bind(rain)
    if layout is not None:
        layout.addWidget(bar)
    rain.settings_bar = bar
    return rain
