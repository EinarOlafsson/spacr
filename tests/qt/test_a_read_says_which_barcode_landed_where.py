"""The read view has to be believable, which means checking actual pixels.

The Map Barcodes screen shows reads so that a user can settle an argument the
summary numbers cannot settle: did this barcode really land here, in this
orientation, on read after read? A view that put the colour one character to
the left would answer that question wrongly and look exactly as convincing
while doing it. So most of the file below grabs the rendered widget and asks
which character columns carry which colour, rather than asking the widget what
it intended to draw.

``_columns_carrying`` is what makes that practical. Every matched run is drawn
with a rule underneath it filled in the barcode's exact colour, so a matched
character column contains at least one pixel of that exact value and an
unmatched one contains none. Probing the middle half of each column keeps the
answer away from the fractional pixel where one character's rule meets the
next one's.

The contrast test at the bottom is the one that does not need a screen. It is
also the one that would have caught the failure this widget's colour rule is
designed against: a palette of hand-picked colours looks fine on whichever
theme its author had open and turns unreadable on one of the other three.
"""
from __future__ import annotations

import pathlib
import re
import time

import pytest

from PySide6.QtGui import QFontInfo

from spacr.qt import preferences, theme
from spacr.qt.widgets import read_view
from spacr.qt.widgets.read_view import (
    BarcodeSpan,
    ReadRow,
    ReadView,
    _resolve_runs,
    barcode_colours,
)


#: A read with no repeating structure, so a colour landing on the wrong
#: character cannot be mistaken for one landing on the right character of an
#: identical neighbour.
READ = "ACGTTGCAACGTTGCAACGTTGCA"

#: Reads used by the laziness test. Far more than any screen can show, which
#: is the point of the measurement.
MANY = 10_000

#: The longest the view may take to accept :data:`MANY` reads. Measured at
#: 1.3 ms on the maintainer's box, so this is three orders of magnitude of
#: headroom and cannot flake under a parallel run; the assertion that carries
#: the real weight is the resolved-row count next to it.
ACCEPT_BUDGET_S = 2.0


def _view(qtbot, qt_theme_applied, rows, kinds=None, size=(760, 260)):
    """Build a shown read view holding ``rows``.

    :param qtbot: pytest-qt bot that adopts the widget for cleanup.
    :param qt_theme_applied: session fixture guaranteeing the spaCR palette
        and stylesheet are on the application, so the ink is the theme's.
    :param rows: reads to show, in any shape ``set_reads`` accepts.
    :param kinds: barcode type names in legend order, or ``None``.
    :param size: width and height to give the widget.
    :returns: the shown :class:`ReadView`.
    """
    view = ReadView()
    qtbot.addWidget(view)
    view.resize(*size)
    view.show()
    qtbot.waitExposed(view)
    view.set_reads(rows, kinds=kinds)
    # The list lays its items out on a posted event rather than inside
    # ``set_reads``, which is part of why accepting reads is cheap. Nothing
    # about the geometry is settled until that event has been delivered.
    qtbot.wait(1)
    view._list.viewport().repaint()
    return view


def _columns_carrying(view, row, colour):
    """Return the character columns of one read that carry ``colour``.

    Grabs the list viewport and probes the middle half of every character
    cell, so the answer is which characters were painted rather than which
    characters the widget meant to paint.

    :param view: the shown :class:`ReadView`.
    :param row: index of the read to probe.
    :param colour: hex colour to look for, as the widget spells it.
    :returns: the set of zero-based character positions carrying it.
    """
    model = view._list.model()
    index = model.index(row, 0)
    rect = view._list.visualRect(index)
    image = view._list.viewport().grab().toImage()
    advance = view._delegate.advance()
    left = rect.left() + 6
    text = model.data(index) or ""
    found = set()
    top = max(0, rect.top())
    bottom = min(image.height(), rect.bottom() + 1)
    for position in range(len(text)):
        start = left + (position + 0.25) * advance
        stop = left + (position + 0.75) * advance
        for x in range(int(start), max(int(start) + 1, int(stop))):
            if x < 0 or x >= image.width():
                continue
            for y in range(top, bottom):
                if image.pixelColor(x, y).name() == colour:
                    found.add(position)
                    break
            if position in found:
                break
    return found


# ---------------------------------------------------------------------------
# What the colours mean
# ---------------------------------------------------------------------------

def test_a_span_colours_exactly_the_characters_it_covers(qtbot,
                                                         qt_theme_applied):
    """The whole promise of the widget, checked against rendered pixels."""
    view = _view(qtbot, qt_theme_applied,
                 [ReadRow(READ, (BarcodeSpan(4, 8, "grna"),))])

    assert _columns_carrying(view, 0, view.colour_for("grna")) == {4, 5, 6, 7}


def test_two_barcode_types_are_drawn_in_two_different_colours(
        qtbot, qt_theme_applied):
    """Following one barcode down a column needs the colours to differ."""
    view = _view(qtbot, qt_theme_applied,
                 [ReadRow(READ, (BarcodeSpan(2, 6, "grna"),
                                 BarcodeSpan(12, 16, "rowID")))])

    grna = view.colour_for("grna")
    row_id = view.colour_for("rowID")
    assert grna != row_id
    assert _columns_carrying(view, 0, grna) == {2, 3, 4, 5}
    assert _columns_carrying(view, 0, row_id) == {12, 13, 14, 15}


def test_one_barcode_type_keeps_its_colour_down_every_row(qtbot,
                                                          qt_theme_applied):
    """A colour that drifted between rows would defeat reading columns."""
    reads = [ReadRow(READ, (BarcodeSpan(4, 8, "grna"),)) for _ in range(6)]
    view = _view(qtbot, qt_theme_applied, reads)

    colour = view.colour_for("grna")
    for row in range(6):
        assert _columns_carrying(view, row, colour) == {4, 5, 6, 7}


def test_a_read_with_no_matches_carries_none_of_the_barcode_colours(
        qtbot, qt_theme_applied):
    """An unmatched read is plain text, not a faintly tinted one."""
    view = _view(qtbot, qt_theme_applied,
                 [ReadRow(READ, (BarcodeSpan(4, 8, "grna"),)),
                  ReadRow(READ, ())])

    colour = view.colour_for("grna")
    assert _columns_carrying(view, 0, colour) == {4, 5, 6, 7}
    assert _columns_carrying(view, 1, colour) == set()


def test_a_plain_string_is_accepted_as_a_read_with_no_matches(
        qtbot, qt_theme_applied):
    """The contract is meant to be writable by hand as well as generated."""
    view = _view(qtbot, qt_theme_applied, [READ], kinds=("grna",))

    assert view.row_count() == 1
    assert _columns_carrying(view, 0, view.colour_for("grna")) == set()


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------

def test_the_first_span_wins_an_overlap_and_the_second_keeps_the_rest(
        qtbot, qt_theme_applied):
    """The documented rule, checked where it is visible: on screen."""
    view = _view(qtbot, qt_theme_applied,
                 [ReadRow(READ, (BarcodeSpan(0, 6, "grna"),
                                 BarcodeSpan(4, 10, "rowID")))])

    assert _columns_carrying(view, 0, view.colour_for("grna")) == {0, 1, 2, 3,
                                                                   4, 5}
    assert _columns_carrying(view, 0, view.colour_for("rowID")) == {6, 7, 8, 9}


def test_every_character_of_an_overlap_belongs_to_exactly_one_type():
    """Ownership is settled before anything is drawn, so it cannot blend."""
    runs = _resolve_runs("AAAAAAAAAA", (BarcodeSpan(0, 6, "grna"),
                                        BarcodeSpan(4, 10, "rowID")))

    assert runs == ((0, 6, "grna"), (6, 10, "rowID"))
    assert sum(stop - start for start, stop, _kind in runs) == 10


def test_a_span_reaching_past_the_read_is_clamped_rather_than_fatal():
    """Read lengths and match offsets come from different places."""
    runs = _resolve_runs("ACGT", (BarcodeSpan(-5, 99, "grna"),))

    assert runs == ((0, 4, "grna"),)


def test_a_span_covering_nothing_is_ignored():
    """An empty or inverted interval is an answer, not a reason to raise."""
    assert _resolve_runs("ACGT", (BarcodeSpan(2, 2, "grna"),)) == (
        (0, 4, None),)
    assert _resolve_runs("ACGT", (BarcodeSpan(3, 1, "grna"),)) == (
        (0, 4, None),)


# ---------------------------------------------------------------------------
# The theme
# ---------------------------------------------------------------------------

def test_the_colours_change_when_the_theme_does(monkeypatch):
    """A colour frozen at build time is the bug the theme module warns about.

    Light and dark do not merely relabel the same hexes: each palette has its
    own readable luminance, and the colours are derived from it, so the same
    barcode type is a different colour on each.
    """
    monkeypatch.setattr(preferences, "resolve_effective_theme", lambda: "dark")
    dark = barcode_colours(("grna", "rowID", "columnID"))
    monkeypatch.setattr(preferences, "resolve_effective_theme",
                        lambda: "light")
    light = barcode_colours(("grna", "rowID", "columnID"))

    assert set(dark) == set(light) == {"grna", "rowID", "columnID"}
    for kind in dark:
        assert dark[kind] != light[kind], kind


def test_a_shown_view_repaints_itself_in_the_new_theme(qtbot,
                                                       qt_theme_applied,
                                                       monkeypatch):
    """The pixels have to move too, not only the numbers behind them."""
    monkeypatch.setattr(preferences, "resolve_effective_theme", lambda: "dark")
    view = _view(qtbot, qt_theme_applied,
                 [ReadRow(READ, (BarcodeSpan(4, 8, "grna"),))])
    dark = view.colour_for("grna")
    assert _columns_carrying(view, 0, dark) == {4, 5, 6, 7}

    monkeypatch.setattr(preferences, "resolve_effective_theme",
                        lambda: "light")
    view.refresh_colours()
    qtbot.wait(1)
    view._list.viewport().repaint()

    light = view.colour_for("grna")
    assert light != dark
    assert _columns_carrying(view, 0, light) == {4, 5, 6, 7}
    assert _columns_carrying(view, 0, dark) == set()


@pytest.mark.parametrize("name", theme.THEMES)
def test_every_barcode_colour_stays_readable_on_every_theme(name,
                                                            monkeypatch):
    """The measurement the colour rule exists to guarantee.

    Twenty barcode types is far more than a real plate layout needs, and it
    is deliberately more than a hand-written palette would have supplied. The
    surfaces are resolved the way the theme module resolves them, so the two
    image themes are judged on the scrim composited over the wallpaper rather
    than on a colour the user never sees.
    """
    monkeypatch.setattr(preferences, "resolve_effective_theme", lambda: name)
    colours = barcode_colours([f"barcode_{n}" for n in range(20)])

    for kind, colour in colours.items():
        for surface in theme.PAGE_SURFACES:
            ratio = theme.contrast_ratio(
                colour, theme.effective_surface(name, surface))
            assert ratio >= 4.5, (
                f"{kind} at {colour} is {ratio:.2f}:1 on {name}/{surface}")


@pytest.mark.parametrize("name", theme.THEMES)
def test_a_selected_read_keeps_its_barcode_colours_readable(name,
                                                            monkeypatch):
    """Selection paints behind the text, so it has to be judged too.

    The selected row is filled with ``accent_soft``, which is the one surface
    the theme already holds ``accent`` against at the same ratio. Since every
    barcode colour carries the accent's luminance, the guarantee carries over
    rather than needing a second solve.
    """
    monkeypatch.setattr(preferences, "resolve_effective_theme", lambda: name)
    colours = barcode_colours([f"barcode_{n}" for n in range(20)])
    behind = theme.effective_surface(name, "accent_soft")

    for kind, colour in colours.items():
        ratio = theme.contrast_ratio(colour, behind)
        assert ratio >= 4.5, f"{kind} at {colour} is {ratio:.2f}:1 on {name}"


def test_a_selected_read_is_painted_on_the_surface_the_ratios_assume(
        qtbot, qt_theme_applied, monkeypatch):
    """Close the loop on the test above by checking what is really behind.

    The contrast figures next door are worth nothing if the selected row is
    filled with some other colour. Probing the fill rather than trusting it
    is what ties the measurement to the widget.
    """
    monkeypatch.setattr(preferences, "resolve_effective_theme", lambda: "dark")
    view = _view(qtbot, qt_theme_applied,
                 [ReadRow(READ, (BarcodeSpan(4, 8, "grna"),))])
    view._list.setCurrentIndex(view._list.model().index(0, 0))
    qtbot.wait(1)
    view._list.viewport().repaint()

    behind = theme.palette_for("dark")["accent_soft"]
    rect = view._list.visualRect(view._list.model().index(0, 0))
    image = view._list.viewport().grab().toImage()
    filled = sum(1
                 for y in range(max(0, rect.top()),
                                min(image.height(), rect.bottom() + 1))
                 for x in range(0, min(image.width(), 200))
                 if image.pixelColor(x, y).name() == behind)

    assert filled > 1000, f"only {filled} pixels of {behind} behind the read"
    assert view.colour_for("grna") in {
        image.pixelColor(x, y).name()
        for y in range(max(0, rect.top()),
                       min(image.height(), rect.bottom() + 1))
        for x in range(0, min(image.width(), 200))}


def test_no_colour_is_written_as_a_literal_in_the_widget():
    """The rule that keeps the guarantees above from being bypassed later."""
    source = pathlib.Path(read_view.__file__).read_text(encoding="utf-8")
    body = "\n".join(line for line in source.splitlines()
                     if not line.lstrip().startswith("#"))
    hexes = set(re.findall(r"\"#[0-9a-fA-F]{6}\"", body))

    # The two fallbacks the delegate holds before it has ever read a palette
    # are the only ones allowed, and they are replaced in its constructor.
    assert hexes <= {'"#ffffff"', '"#000000"'}, hexes


# ---------------------------------------------------------------------------
# One read per row, and what that costs
# ---------------------------------------------------------------------------

def test_the_reads_are_drawn_in_a_fixed_pitch_face(qtbot, qt_theme_applied):
    """Column alignment is the reason to look at reads at all.

    Checked through ``QFontInfo``, which reports the face actually resolved,
    because asking a ``QFont`` what family it was assigned says nothing about
    what the font system substituted.
    """
    view = _view(qtbot, qt_theme_applied, [READ])

    assert QFontInfo(view._delegate.font()).fixedPitch()
    assert view._delegate.advance() > 0


def test_a_long_read_scrolls_sideways_instead_of_wrapping(qtbot,
                                                          qt_theme_applied):
    """One read per row is a promise that wrapping would quietly break."""
    long_read = "ACGT" * 40
    view = _view(qtbot, qt_theme_applied, [ReadRow(long_read, ())],
                 size=(320, 200))

    assert view._list.horizontalScrollBar().maximum() > 0
    height = view._list.visualRect(view._list.model().index(0, 0)).height()
    assert height < 2 * view._delegate.row_height(), (
        "a row grew tall enough to hold a wrapped second line")


def test_two_reads_of_different_lengths_still_start_in_the_same_column(
        qtbot, qt_theme_applied):
    """Alignment is per column, so a short read must not shift a long one."""
    view = _view(qtbot, qt_theme_applied,
                 [ReadRow(READ, (BarcodeSpan(4, 8, "grna"),)),
                  ReadRow(READ[:10], (BarcodeSpan(4, 8, "grna"),))])

    colour = view.colour_for("grna")
    assert _columns_carrying(view, 0, colour) == {4, 5, 6, 7}
    assert _columns_carrying(view, 1, colour) == {4, 5, 6, 7}


def test_ten_thousand_reads_cost_a_screenful_of_work_not_ten_thousand(
        qtbot, qt_theme_applied):
    """The guard against formatting the whole file on the GUI thread.

    Accepting the reads is a list assignment, and the character ownership of
    a read is worked out when that read is first painted. So the number of
    reads that have been resolved after showing the view is a count of what
    fits on screen, and it stays that way after scrolling to the end.
    """
    reads = [ReadRow(READ, (BarcodeSpan(4, 8, "grna"),))
             for _ in range(MANY)]
    view = ReadView()
    qtbot.addWidget(view)
    view.resize(760, 260)
    view.show()
    qtbot.waitExposed(view)

    started = time.perf_counter()
    view.set_reads(reads, kinds=("grna",))
    elapsed = time.perf_counter() - started
    qtbot.wait(1)
    view._list.viewport().repaint()

    assert view.row_count() == MANY
    assert elapsed < ACCEPT_BUDGET_S, (
        f"accepting {MANY} reads took {elapsed * 1000:.0f} ms")
    on_screen = view._resolved_row_count()
    assert 0 < on_screen < 200, on_screen

    bar = view._list.verticalScrollBar()
    bar.setValue(bar.maximum())
    qtbot.wait(1)
    view._list.viewport().repaint()
    assert view._resolved_row_count() < 400, view._resolved_row_count()


# ---------------------------------------------------------------------------
# The legend, and starting again
# ---------------------------------------------------------------------------

def test_the_legend_names_every_barcode_type_in_its_own_colour(
        qtbot, qt_theme_applied):
    """A colour nobody can name is decoration rather than information."""
    view = _view(qtbot, qt_theme_applied,
                 [ReadRow(READ, (BarcodeSpan(4, 8, "grna"),))],
                 kinds=("grna", "rowID", "columnID"))

    entries = view._legend.entries()
    assert [name for name, _colour in entries] == ["grna", "rowID",
                                                   "columnID"]
    assert all(colour == view.colour_for(name) for name, colour in entries)
    assert len({colour for _name, colour in entries}) == 3


def test_a_barcode_that_matched_nothing_still_appears_in_the_legend(
        qtbot, qt_theme_applied):
    """That a search ran and found nothing is itself worth showing."""
    view = _view(qtbot, qt_theme_applied, [ReadRow(READ, ())],
                 kinds=("grna", "rowID"))

    assert view.kinds() == ("grna", "rowID")
    assert view.colour_for("rowID") is not None


def test_barcode_types_left_unsaid_are_taken_from_the_spans(
        qtbot, qt_theme_applied):
    """The caller should not have to enumerate what the spans already say."""
    view = _view(qtbot, qt_theme_applied,
                 [ReadRow(READ, (BarcodeSpan(0, 4, "rowID"),
                                 BarcodeSpan(8, 12, "grna")))])

    assert view.kinds() == ("rowID", "grna")


def test_clearing_forgets_the_barcode_types_as_well_as_the_reads(
        qtbot, qt_theme_applied):
    """A new run may search for different barcodes than the last one did."""
    view = _view(qtbot, qt_theme_applied,
                 [ReadRow(READ, (BarcodeSpan(0, 4, "grna"),))])
    first = view.colour_for("grna")

    view.clear()
    assert view.row_count() == 0
    assert view.kinds() == ()
    assert view.colour_for("grna") is None

    view.set_reads([ReadRow(READ, (BarcodeSpan(0, 4, "rowID"),))])
    assert view.colour_for("rowID") == first


def test_showing_more_reads_does_not_recolour_the_types_already_shown(
        qtbot, qt_theme_applied):
    """A colour that moved as results streamed in would be unreadable."""
    view = _view(qtbot, qt_theme_applied,
                 [ReadRow(READ, (BarcodeSpan(0, 4, "grna"),))])
    first = view.colour_for("grna")

    view.set_reads([ReadRow(READ, (BarcodeSpan(0, 4, "grna"),
                                   BarcodeSpan(8, 12, "rowID")))])

    assert view.colour_for("grna") == first
    assert view.kinds() == ("grna", "rowID")


def test_an_empty_view_shows_nothing_and_claims_nothing(qtbot,
                                                        qt_theme_applied):
    """Constructing the widget must not invent reads or barcode types."""
    view = ReadView()
    qtbot.addWidget(view)
    view.resize(400, 120)
    view.show()
    qtbot.waitExposed(view)
    qtbot.wait(1)
    view._list.viewport().repaint()

    assert view.row_count() == 0
    assert view.kinds() == ()
    assert view._legend.entries() == ()
