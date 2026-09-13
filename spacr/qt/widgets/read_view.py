"""Show sequencing reads as text, one read per row, with matches coloured.

The Map Barcodes screen asks the user to believe a claim about their FASTQ
files: that a particular barcode was found in a particular place, in a
particular orientation. A percentage in a summary box cannot settle an
argument about that, because a percentage looks the same whether the matches
are real or accidental. Reads do settle it. Seeing the gRNA land in the same
columns on row after row is the difference between a number the user trusts
and a number the user has to take on faith, and seeing nothing line up is how
a misconfigured run announces itself in one glance instead of after an hour
of mapping that produces no counts.

So this widget is deliberately plain. It draws the read text in a fixed pitch
face, one read to a line, and it paints the characters a barcode matched in a
colour belonging to that barcode type. Nothing wraps, nothing is elided, and
the character in column forty of one read sits directly above the character in
column forty of the next, which is the entire reason for looking at reads
rather than at a table.

This module knows nothing about how matches are found. It takes reads and
spans that somebody else computed, so the search engine and the screen that
hosts it can change freely without touching anything here, and so this can be
tested without either of them. The contract is the pair of small record types
below, and it is meant to be written by hand as easily as it is generated.

A note on colour, because it is the part that is easy to get wrong. Every
colour painted here is derived from the theme that is on screen, never written
as a literal, and the derivation preserves the relative luminance of the
theme's own accent role. Contrast is a function of relative luminance and of
nothing else, so a colour made this way clears exactly the contrast that the
accent already clears on that theme, including the themes whose panels are
translucent over a photograph. That is what lets an arbitrary number of
barcode types each get a colour of their own without any of them becoming
unreadable on some theme nobody checked.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

from PySide6.QtCore import QAbstractListModel, QEvent, QModelIndex, \
    QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QFontMetricsF, \
    QPainter
from PySide6.QtWidgets import QAbstractItemView, QFrame, QListView, \
    QSizePolicy, QStyle, QStyledItemDelegate, QVBoxLayout, QWidget

from ..theme import SPACING, active_palette, font_px, spaceout_palette


# ---------------------------------------------------------------------------
# What it costs, measured
# ---------------------------------------------------------------------------
# 150-base reads with three matched spans each, offscreen, a 900x700 view on
# the maintainer's box. The column that matters is the last one: the number of
# reads whose character ownership had to be worked out does not depend on how
# many reads there are, because only the rows on screen are ever asked for.
#
#     reads    set_reads   first paint   resolved   scroll to end   resolved
#      1 000      0.3 ms       16.5 ms         38          8.9 ms         76
#     10 000      1.1 ms       23.2 ms         38          9.1 ms         76
#     50 000      5.9 ms       76.7 ms         38          9.6 ms         76
#
# The first paint is the one number that grows with the file, and it is Qt
# laying the rows out on the posted event after the model reset rather than
# anything here formatting text. At fifty thousand reads it is 77 ms once,
# against the 400 ms the preview responsiveness guard next door treats as a
# freeze, and it does not come back when the user scrolls.


# ---------------------------------------------------------------------------
# The colour rule
# ---------------------------------------------------------------------------
# One mechanism, no table of literals, and no ceiling on how many barcode
# types it can serve. Colour number `i` is the theme's own `accent` role put
# through `spaceout_palette`, the published re-hue whose documented job is to
# move a role in hue while preserving the luminance that makes it readable.
#
# WHY THAT IS SAFE RATHER THAN MERELY PRETTY. `CONTRAST_RULES` holds `accent`
# at 4.5:1 against every page surface of every theme, and WCAG contrast is a
# function of relative luminance alone. `spaceout_palette` leaves the
# luminance where it found it -- measured drift below 0.003 across all four
# palettes -- so every hue it hands back clears what `accent` clears. Measured
# worst case over the first eight colours, against every page surface:
#
#     dark 5.55:1   light 4.55:1   cell 7.78:1   glass 5.68:1
#
# and against `accent_soft`, which is what a selected row is painted with:
#
#     dark 4.52:1   light 4.69:1   cell 8.75:1   glass 6.86:1
#
# THE STEP IS THE GOLDEN ANGLE, not `360 / count`. Even spacing would make a
# type's colour depend on how many types there are, so finding a fourth
# barcode would recolour the three already on screen and the user would have
# to re-read the legend. The golden angle spreads any prefix of the sequence
# about as well as even spacing does while leaving colour number three the
# same colour it was before colour number four existed.

#: Degrees between one barcode type's hue and the next.
_GOLDEN_ANGLE = 137.50776405003785

#: Rows whose character ownership is kept resolved at once. Resolution is
#: lazy and a screenful is a few dozen rows, so this is loose enough never to
#: be hit by scrolling and tight enough that a long session cannot grow a
#: cache entry for every read in the file.
_RUN_CACHE_ROWS = 4096

#: Space left either side of the read text, in pixels.
_TEXT_PAD = 6

#: Extra height given to a row beyond the font's own line height.
_ROW_PAD = 2


class BarcodeSpan(NamedTuple):
    """One stretch of a read that a barcode of some type matched.

    The interval is half open and counted in characters from the start of the
    read, so it follows Python slicing exactly and the matched text is
    ``read[start:stop]``. Positions outside the read are clamped when the span
    is drawn, and a span whose stop is not after its start contributes
    nothing, so a caller doing arithmetic on read lengths cannot make the view
    raise.

    The kind is an opaque label. It is whatever the caller calls that family
    of barcode, it is what the legend shows, and it is what picks the colour,
    so two spans sharing a kind are guaranteed to share a colour on every row.

    :param start: First matched character, counting from zero.
    :param stop: One past the last matched character.
    :param kind: Name of the barcode type that matched.
    """

    start: int
    stop: int
    kind: str


class ReadRow(NamedTuple):
    """One read and everything that matched inside it.

    :param text: The read as it should appear on screen, already trimmed to
        whatever window the caller wants shown.
    :param spans: The matches inside that text. Order matters when two of
        them overlap, as described on :class:`ReadView`.
    """

    text: str
    spans: Tuple[BarcodeSpan, ...]


def barcode_colours(kinds: Sequence[str]) -> Dict[str, str]:
    """Give each barcode type a colour of its own, taken from the theme.

    Colours are assigned by position, so a type keeps its colour as long as
    the caller keeps passing the types in the same order, and adding a type to
    the end of the sequence never changes the colours already handed out.
    Repeats are ignored after the first appearance.

    The result follows the theme on screen at the moment of the call. It
    differs between the light and dark palettes, and between both and the
    image palettes, which is deliberate rather than incidental: each palette
    has its own readable luminance and the colours are derived from it.

    :param kinds: Barcode type names, in the order they should be coloured.
    :returns: A mapping from each name to a hex colour string.
    """
    accent = active_palette()["accent"]
    out: Dict[str, str] = {}
    for index, kind in enumerate(dict.fromkeys(str(k) for k in kinds)):
        drift = (index * _GOLDEN_ANGLE) % 360.0
        out[kind] = spaceout_palette({"accent": accent}, drift)["accent"]
    return out


def _as_row(item: object) -> ReadRow:
    """Accept the several shapes a caller might reasonably pass for a read.

    A bare string is a read with no matches, a two element sequence is a read
    and its spans, and a :class:`ReadRow` is itself. Being generous here means
    the integrating screen can hand over whatever its search engine already
    produces instead of building records to suit this widget.

    :param item: A read in any of the accepted shapes.
    :returns: The equivalent :class:`ReadRow`.
    """
    if isinstance(item, ReadRow):
        return item
    if isinstance(item, str):
        return ReadRow(item, ())
    text, spans = item
    return ReadRow(str(text), tuple(spans))


def _as_span(item: object, length: int) -> Optional[BarcodeSpan]:
    """Clamp one span to a read, or reject it.

    :param item: A :class:`BarcodeSpan` or any three element sequence holding
        a start, a stop and a kind.
    :param length: Characters in the read the span belongs to.
    :returns: The clamped span, or ``None`` when it covers nothing.
    """
    start, stop, kind = item
    start = max(0, min(int(start), length))
    stop = max(0, min(int(stop), length))
    if stop <= start:
        return None
    return BarcodeSpan(start, stop, str(kind))


def _resolve_runs(text: str,
                  spans: Iterable[object]) -> Tuple[Tuple[int, int,
                                                          Optional[str]], ...]:
    """Work out which barcode type owns each character, then group them.

    Ownership is decided before anything is drawn, and every character ends up
    owned by exactly one type or by none, which is what makes overlapping
    spans harmless. The first span in the sequence that covers a character
    keeps it; a later span still colours whichever of its characters are
    left over, so an overlapped match loses only the part that was taken and
    stays visible for the rest.

    :param text: The read being drawn.
    :param spans: The matches inside it, most important first.
    :returns: Consecutive runs as start, stop and owning type, covering the
        whole read, where an owner of ``None`` means no barcode matched there.
    """
    length = len(text)
    if not length:
        return ()
    owner: List[Optional[str]] = [None] * length
    for raw in spans:
        span = _as_span(raw, length)
        if span is None:
            continue
        for position in range(span.start, span.stop):
            if owner[position] is None:
                owner[position] = span.kind
    runs: List[Tuple[int, int, Optional[str]]] = []
    start = 0
    while start < length:
        here = owner[start]
        stop = start + 1
        while stop < length and owner[stop] == here:
            stop += 1
        runs.append((start, stop, here))
        start = stop
    return tuple(runs)


def _mono_font() -> QFont:
    """Build the fixed pitch font the reads are drawn in.

    Asked for explicitly rather than inherited from the widget, because the
    application stylesheet opens with a family rule that applies to every
    widget and a stylesheet font beats one set in code. A widget that merely
    calls ``setFont`` with the system fixed font therefore renders
    proportionally and the columns stop lining up. Nothing here reads the
    widget font: the font below is handed straight to the painter, where no
    stylesheet can reach it.

    :returns: A fixed pitch font at the user's current text size.
    """
    font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
    font.setFamilies([font.family(), "DejaVu Sans Mono", "Menlo",
                      "Consolas", "Courier New", "monospace"])
    font.setStyleHint(QFont.Monospace)
    font.setFixedPitch(True)
    font.setPixelSize(font_px("small"))
    return font


class _ReadModel(QAbstractListModel):
    """Hold the reads and hand out one row at a time.

    A list model rather than formatted text is the whole performance story.
    Adding reads costs a list assignment, the view asks for the rows it can
    actually show, and character ownership is resolved when a row is first
    painted instead of when the reads arrive. Ten thousand reads therefore
    cost the same to set as ten.

    :param parent: Optional Qt owner responsible for the model's lifetime.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Start empty, with no reads and no colours."""
        super().__init__(parent)
        self._rows: List[ReadRow] = []
        self._colours: Dict[str, str] = {}
        self._runs: Dict[int, Tuple[Tuple[int, int, Optional[str]], ...]] = {}
        self._widest = 0

    def set_rows(self, rows: Sequence[object],
                 colours: Dict[str, str]) -> None:
        """Replace every read, and the colour each barcode type is drawn in.

        :param rows: The reads, in any shape :func:`_as_row` accepts.
        :param colours: Mapping from barcode type name to hex colour.
        """
        self.beginResetModel()
        self._rows = [_as_row(row) for row in rows]
        self._colours = dict(colours)
        self._runs.clear()
        self._widest = max((len(row.text) for row in self._rows), default=0)
        self.endResetModel()

    def set_colours(self, colours: Dict[str, str]) -> None:
        """Repoint the colours without disturbing the reads or the scroll.

        :param colours: Mapping from barcode type name to hex colour.
        """
        self._colours = dict(colours)

    def colour_for(self, kind: Optional[str]) -> Optional[str]:
        """Return the colour a barcode type is drawn in, if it has one.

        :param kind: Barcode type name, or ``None`` for unmatched text.
        :returns: A hex colour string, or ``None`` when the caller should use
            the ordinary text ink.
        """
        if kind is None:
            return None
        return self._colours.get(kind)

    def widest(self) -> int:
        """Characters in the longest read, which sets the scrolling width.

        :returns: The character count, or zero when there are no reads.
        """
        return self._widest

    def runs_for(self, row: int) -> Tuple[Tuple[int, int, Optional[str]], ...]:
        """Return the coloured runs of one read, resolving them on first ask.

        :param row: Index of the read.
        :returns: The runs, as described by :func:`_resolve_runs`.
        """
        cached = self._runs.get(row)
        if cached is not None:
            return cached
        if len(self._runs) >= _RUN_CACHE_ROWS:
            self._runs.clear()
        item = self._rows[row]
        runs = _resolve_runs(item.text, item.spans)
        self._runs[row] = runs
        return runs

    def resolved_rows(self) -> int:
        """How many reads have had their character ownership worked out.

        This is the measurement behind the claim that showing the view does
        not format every read. It is expected to stay near the number of rows
        that fit on screen however many reads were handed over.

        :returns: The number of rows currently resolved.
        """
        return len(self._runs)

    def rowCount(self,  # noqa: N802
                 parent: QModelIndex = QModelIndex()) -> int:  # noqa: B008
        """Return the number of reads for the root, and zero for children.

        :param parent: Qt parent index whose child count is requested.
        """
        return 0 if parent.isValid() else len(self._rows)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> object:
        """Return the read text, which is also its tooltip.

        :param index: Model index identifying the read.
        :param role: Qt data role; anything else returns ``None``.
        """
        if not index.isValid():
            return None
        if role in (Qt.DisplayRole, Qt.ToolTipRole):
            return self._rows[index.row()].text
        return None


class _ReadDelegate(QStyledItemDelegate):
    """Paint one read, colouring the characters each barcode matched.

    Painting rather than styling is what keeps the promise of one read per
    row. Rich text would wrap, elide, and cost a formatting pass per read;
    a painter draws exactly the characters asked for at exactly the character
    positions asked for and does it only for the rows on screen.

    Colour is not the only cue. A matched run also gets a rule drawn under it
    in the same colour, so the match is still visible to a reader who cannot
    separate two hues, and so a single matched character is noticeable at all.

    :param parent: Optional Qt owner responsible for the delegate's lifetime.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Build the fixed pitch font and take the theme's text colours."""
        super().__init__(parent)
        self._font = _mono_font()
        self._metrics = QFontMetricsF(self._font)
        self._ink = "#ffffff"
        self._selection = "#000000"
        self.refresh_theme()

    def refresh_theme(self) -> None:
        """Re-read the font size and the two theme colours this draws with.

        Called when the theme or the text size changes. Resolving them here,
        once, rather than inside the paint means a repaint never touches the
        preference store.
        """
        self._font = _mono_font()
        self._metrics = QFontMetricsF(self._font)
        palette = active_palette()
        self._ink = palette["fg"]
        self._selection = palette["accent_soft"]

    def font(self) -> QFont:
        """Return the fixed pitch font reads are drawn in.

        :returns: The font, which callers may measure but should not mutate.
        """
        return QFont(self._font)

    def row_height(self) -> int:
        """Return the height of one read row in pixels.

        :returns: The line height of the fixed pitch font plus padding.
        """
        return int(round(self._metrics.height())) + _ROW_PAD * 2

    def advance(self) -> float:
        """Return the width of one character in the fixed pitch font.

        :returns: The horizontal advance in pixels, never zero.
        """
        return self._metrics.horizontalAdvance("0") or 8.0

    def sizeHint(self, option, index) -> QSize:  # noqa: N802
        """Return one row's size, wide enough for the longest read.

        Every row is given the same width so that nothing wraps and the view
        scrolls sideways instead. The view is told the sizes are uniform, so
        this is asked once rather than once per read.

        :param option: Qt style option for the item.
        :param index: Model index of the item being measured.
        """
        model = index.model()
        widest = model.widest() if hasattr(model, "widest") else 0
        width = int(round(self.advance() * max(widest, 1))) + _TEXT_PAD * 2
        return QSize(width, self.row_height())

    def paint(self, painter: QPainter, option, index) -> None:
        """Draw one read, run by run, in the colour each run's type owns.

        :param painter: Painter supplied by the view.
        :param option: Qt style option carrying the row rectangle and state.
        :param index: Model index of the read being drawn.
        """
        model = index.model()
        text = index.data(Qt.DisplayRole) or ""
        painter.save()
        if option.state & QStyle.State_Selected:
            painter.fillRect(option.rect, QColor(self._selection))
        painter.setFont(self._font)
        advance = self.advance()
        rect = option.rect
        baseline = (rect.top()
                    + (rect.height() + self._metrics.ascent()
                       - self._metrics.descent()) / 2.0)
        left = rect.left() + _TEXT_PAD
        rule_top = baseline + self._metrics.descent() * 0.35
        rule_height = max(1.0, round(self._metrics.descent() * 0.3))
        runs = (model.runs_for(index.row())
                if hasattr(model, "runs_for")
                else ((0, len(text), None),))
        for start, stop, kind in runs:
            colour = (model.colour_for(kind)
                      if hasattr(model, "colour_for") else None)
            painter.setPen(QColor(colour or self._ink))
            painter.drawText(QPointF(left + start * advance, baseline),
                             text[start:stop])
            if colour:
                painter.fillRect(
                    QRectF(left + start * advance, rule_top,
                           (stop - start) * advance, rule_height),
                    QColor(colour))
        painter.restore()


class _Legend(QWidget):
    """Name each colour, wrapping onto more lines when there are many types.

    Painted rather than built from labels, for the same reason the reads are:
    a per widget stylesheet would freeze the colours it was built with, and
    these colours change with the theme. Everything here is resolved at paint
    time, so a theme switch is a repaint.

    :param parent: Optional Qt owner responsible for the legend's lifetime.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Start with nothing to explain, and therefore no height."""
        super().__init__(parent)
        self._entries: Tuple[Tuple[str, str], ...] = ()
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

    def set_entries(self, entries: Sequence[Tuple[str, str]]) -> None:
        """Replace what the legend explains.

        :param entries: Pairs of barcode type name and hex colour, in the
            order they should be listed.
        """
        self._entries = tuple((str(name), str(colour))
                              for name, colour in entries)
        self.updateGeometry()
        self.update()

    def entries(self) -> Tuple[Tuple[str, str], ...]:
        """Return the pairs currently listed.

        :returns: Pairs of barcode type name and hex colour.
        """
        return self._entries

    def _placements(self, width: int) -> Tuple[List[Tuple[float, float, str,
                                                          str]], int]:
        """Work out where each entry goes, and how tall that makes the legend.

        :param width: The width the legend has to lay out inside.
        :returns: The placements, as left edge, top edge, name and colour,
            together with the total height in pixels.
        """
        font = QFont()
        font.setPixelSize(font_px("small"))
        metrics = QFontMetricsF(font)
        line = metrics.height()
        swatch = line * 0.6
        gap = SPACING["sm"]
        step = line + SPACING["xs"]
        placements: List[Tuple[float, float, str, str]] = []
        x, y, tallest = 0.0, 0.0, line if self._entries else 0.0
        for name, colour in self._entries:
            span = swatch + SPACING["xs"] + metrics.horizontalAdvance(name)
            if x and x + span > max(width, 1):
                x, y = 0.0, y + step
                tallest = y + line
            placements.append((x, y, name, colour))
            x += span + gap
        return placements, int(round(tallest))

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        """Report that the legend's height depends on how wide it is.

        :returns: Always ``True``, because entries wrap onto further lines.
        """
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        """Return the height the entries need at a given width.

        :param width: Width available to the legend, in pixels.
        """
        return self._placements(width)[1]

    def sizeHint(self) -> QSize:  # noqa: N802
        """Return the legend's preferred size at its current width.

        :returns: The current width paired with the height the entries need.
        """
        return QSize(self.width() or 1, self.heightForWidth(self.width() or 1))

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        """Return the smallest useful legend size.

        :returns: A single line's height, so one entry always has room.
        """
        return QSize(1, self._placements(1)[1])

    def paintEvent(self, event) -> None:  # noqa: N802
        """Draw a swatch and a name for every barcode type.

        :param event: Qt paint event, whose region this ignores because the
            legend is a handful of short items.
        """
        if not self._entries:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        font = QFont()
        font.setPixelSize(font_px("small"))
        painter.setFont(font)
        metrics = QFontMetricsF(font)
        line = metrics.height()
        swatch = line * 0.6
        for x, y, name, colour in self._placements(self.width())[0]:
            ink = QColor(colour)
            painter.fillRect(
                QRectF(x, y + (line - swatch) / 2.0, swatch, swatch), ink)
            painter.setPen(ink)
            painter.drawText(
                QPointF(x + swatch + SPACING["xs"], y + metrics.ascent()),
                name)
        painter.end()


class ReadView(QWidget):
    """Reads as text, one to a row, with each barcode type in its own colour.

    Hand it reads and the spans somebody else matched inside them, and it
    shows them. It never searches for anything itself and never imports the
    code that does, so it can be dropped into any screen that has reads and
    positions to show.

    What it guarantees. Reads are drawn in a fixed pitch face at fixed
    character positions, so column forty of one read sits above column forty
    of the next. A read never wraps onto a second visual row: a long one makes
    the view scroll sideways instead, because a wrapped read would break the
    one read per row promise that the whole widget exists to keep. A barcode
    type keeps one colour for as long as the view is showing it, so the eye
    can follow a match down the rows, and a legend above the reads says which
    colour is which.

    How overlaps are settled. The spans of a read are considered in the order
    they were given, and the first one to cover a character keeps it. A span
    that arrives later still colours every character not already taken, so it
    is never hidden completely, only trimmed. Deciding this before any drawing
    starts is what makes overlapping spans merely a question of priority
    rather than a way to corrupt the picture. A caller who wants a particular
    barcode type to win an overlap should list its spans first.

    What it costs. The reads are held in a list model and drawn by a delegate,
    so handing over reads costs a list assignment and drawing costs only the
    rows that are actually on screen. Setting ten thousand reads and showing
    the view resolves the character ownership of a screenful of rows, not of
    ten thousand, which is what keeps the interface responsive on a real
    FASTQ sample rather than on a toy one.

    :param parent: Optional Qt owner responsible for the widget's lifetime.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Build an empty view with a legend above a list of reads."""
        super().__init__(parent)
        self._kinds: List[str] = []
        self._colours: Dict[str, str] = {}
        self._model = _ReadModel(self)
        self._delegate = _ReadDelegate(self)
        self._legend = _Legend(self)
        self._list = QListView(self)
        self._list.setModel(self._model)
        self._list.setItemDelegate(self._delegate)
        # Uniform sizes is the setting that makes the view ask the delegate
        # for one size instead of one per read, and wrapping off with elide
        # off is what stops a long read becoming two visual rows.
        self._list.setUniformItemSizes(True)
        self._list.setWordWrap(False)
        self._list.setTextElideMode(Qt.ElideNone)
        self._list.setWrapping(False)
        self._list.setFlow(QListView.TopToBottom)
        self._list.setResizeMode(QListView.Fixed)
        self._list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._list.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self._list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._list.setFrameShape(QFrame.NoFrame)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING["xs"])
        layout.addWidget(self._legend)
        layout.addWidget(self._list, 1)

    def set_reads(self, rows: Sequence[object],
                  kinds: Optional[Sequence[str]] = None) -> None:
        """Show these reads, with these barcode types in the legend.

        Passing the barcode types explicitly is worth doing whenever they are
        known, because it fixes both the legend order and which type gets
        which colour. A type that is searched for but found in none of the
        reads on screen then still appears in the legend, which is itself
        information: it says the search ran and came back empty rather than
        leaving the user to wonder whether it ran at all.

        Left unsaid, the types are taken from the spans in the order they are
        first seen, and any type already being shown keeps the colour it has.

        :param rows: The reads, each a :class:`ReadRow`, a plain string for a
            read with no matches, or a text and spans pair.
        :param kinds: Barcode type names in legend order, or ``None`` to take
            them from the spans.
        """
        prepared = [_as_row(row) for row in rows]
        seen = list(self._kinds)
        for name in (kinds if kinds is not None
                     else (span[2] for row in prepared for span in row.spans)):
            name = str(name)
            if name not in seen:
                seen.append(name)
        self._kinds = seen
        self._colours = barcode_colours(self._kinds)
        self._model.set_rows(prepared, self._colours)
        self._legend.set_entries([(name, self._colours[name])
                                  for name in self._kinds])

    def clear(self) -> None:
        """Drop every read and forget which barcode types were being shown.

        Forgetting the types is the point of having this at all rather than
        setting an empty list of reads. A fresh run may search for a different
        set of barcodes, and carrying the previous run's assignment over would
        give the new first barcode the old second barcode's colour.
        """
        self._kinds = []
        self._colours = {}
        self._model.set_rows((), {})
        self._legend.set_entries(())

    def row_count(self) -> int:
        """Return how many reads are being shown.

        :returns: The number of rows in the view.
        """
        return self._model.rowCount()

    def kinds(self) -> Tuple[str, ...]:
        """Return the barcode types being shown, in legend order.

        :returns: The type names, which is also the order that assigned their
            colours.
        """
        return tuple(self._kinds)

    def colour_for(self, kind: str) -> Optional[str]:
        """Return the colour one barcode type is drawn in.

        :param kind: The barcode type name.
        :returns: A hex colour string, or ``None`` if the view has never been
            shown that type.
        """
        return self._colours.get(str(kind))

    def changeEvent(self, event) -> None:  # noqa: N802
        """Follow a theme or text size change without being told about it.

        The application re-applies its stylesheet and palette when the theme
        changes, and Qt delivers that to every widget as a change event.
        Taking the colours again here means the reads re-theme with everything
        else instead of keeping the palette they were first drawn in.

        :param event: The Qt change event being delivered.
        """
        super().changeEvent(event)
        # Qt can deliver a change event while the base class is still
        # constructing, before anything below exists to refresh.
        if not hasattr(self, "_delegate"):
            return
        if event.type() in (QEvent.Type.StyleChange,
                            QEvent.Type.PaletteChange,
                            QEvent.Type.ApplicationPaletteChange,
                            QEvent.Type.FontChange):
            self.refresh_colours()

    def refresh_colours(self) -> None:
        """Take the colours from the theme again and repaint.

        Safe to call at any time and cheap enough to call on a whim: it
        re-derives one colour per barcode type, not per read.
        """
        self._delegate.refresh_theme()
        if self._kinds:
            self._colours = barcode_colours(self._kinds)
            self._model.set_colours(self._colours)
            self._legend.set_entries([(name, self._colours[name])
                                      for name in self._kinds])
        # A new text size means a new row height and a new row width, and
        # the view has cached the old ones because it was told the sizes are
        # uniform. Laying the items out again asks for them afresh, which
        # resetting the view would also do at the cost of throwing away the
        # user's scroll position and selection.
        self._list.doItemsLayout()
        self._list.viewport().update()

    def _resolved_row_count(self) -> int:
        """Return how many reads have had their character ownership resolved.

        Used by the tests that pin the laziness this widget depends on.

        :returns: The number of resolved rows.
        """
        return self._model.resolved_rows()
