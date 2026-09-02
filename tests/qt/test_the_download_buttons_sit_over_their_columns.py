"""Instruction 353's second half: the Download row lines up with the table.

The first half moved the row to the top of Regression's Input Tables. This
file is the preference the same request also asked for -- "if you can allign
each button to their respective columns below in the table that would be
perfect ... center the butons to the center of each cell and lock the width of
the downloade table columns to the width of the paired data column
counterparts below."

MEASURED, NOT LOOKED AT. Every assertion here is a number read out of a shown
screen: the button's centre in the screen's own coordinates against the centre
of the header section it fills. Before the alignment existed the three mapped
buttons sat, on a 1400x900 screen with the table's default 100 px columns,
111.0, 125.0 and 115.5 px to the LEFT of the columns they fill -- the row was
laid out by a plain QHBoxLayout starting at the strip's left edge, which is
also the table's left edge, so every button was over the "Plate / proposal"
column or the one after it.

THE THREE THAT MAP AND THE ONE THAT DOES NOT. Score fills column 1, Count
fills column 2 and Measurements (.db) fills column 3 -- the same
``SIDE_COLUMNS`` the drop router files a dropped file into, so there is one
statement of which column is which. Image crops fills ``src``, a setting row
of the same form and not a column of this table, so it is asserted to TRAIL
the aligned run rather than to be centred on a column it has nothing to do
with.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint                             # noqa: E402
from PySide6.QtWidgets import QPushButton                     # noqa: E402

from spacr.qt.screens.app_screen import AppScreen             # noqa: E402
from spacr.qt.widgets.column_aligned_row import (             # noqa: E402
    TRAILING_SPACING, ColumnAlignedRow)
from spacr.qt.widgets.file_list import PairedFileTableWidget  # noqa: E402

#: How far a button's centre may sit from its column's, in px.
#:
#: "each button's horizontal centre is within a pixel or two of its column's
#: centre" -- instruction 353's own bar. Not zero: a column whose width and
#: its button's width differ by an odd number of pixels loses half a pixel to
#: integer division, which is the -0.5 measured for Measurements (.db) over a
#: 160 px column.
TOLERANCE = 2.0

#: Which button fills which column, kept as the pairing a reader of this test
#: can check against the table's own ``SIDE_COLUMNS`` rather than as four
#: numbers typed out again.
MAPPED = {
    "Score": PairedFileTableWidget.SIDE_COLUMNS["score"],
    "Count": PairedFileTableWidget.SIDE_COLUMNS["count"],
    "Measurements (.db)": PairedFileTableWidget.SIDE_COLUMNS["database"],
}

#: The button that fills a setting instead of a column.
UNMAPPED = "Image crops"


def _shown_regression(qtbot):
    """A Regression screen on screen, with Input Tables open.

    SHOWN, because none of this is decided until Qt lays the form out: an
    unshown screen reports the geometry every widget was constructed with,
    which is 100x30 at the origin for all of them and would let a broken
    alignment pass.
    """
    screen = AppScreen(app_key="regression")
    qtbot.addWidget(screen)
    screen.resize(1400, 900)
    screen.show()
    qtbot.waitExposed(screen)
    section = next(
        (s for s in getattr(screen, "_settings_sections", [])
         if s.title().strip().lower() == "input tables"), None)
    assert section is not None, "Regression has no Input Tables section"
    section.set_expanded(True)
    qtbot.wait(10)
    return screen, section


def _table(screen):
    """Regression's paired-data widget, straight out of the settings model."""
    widget = screen._settings_model._widgets.get("paired_data")
    assert isinstance(widget, PairedFileTableWidget), (
        f"paired_data is {type(widget).__name__}, not the paired table")
    return widget


def _buttons(screen):
    """The Download row's buttons, by the text on them."""
    found = {}
    for name in ("_example_scores_button", "_example_counts_button",
                 "_screen_feature_button", "_screen_crops_button"):
        button = getattr(screen, name, None)
        assert isinstance(button, QPushButton), f"no {name} on the screen"
        found[button.text()] = button
    return found


def _centre(screen, widget) -> float:
    """``widget``'s horizontal centre in ``screen``'s coordinates."""
    return (widget.mapTo(screen, QPoint(0, 0)).x() + widget.width() / 2.0)


def _column_centre(screen, table, column: int) -> float:
    """Column ``column``'s centre in ``screen``'s coordinates."""
    header = table.table.horizontalHeader()
    viewport = table.table.viewport()
    left = viewport.mapTo(screen,
                          QPoint(header.sectionViewportPosition(column), 0)).x()
    return left + header.sectionSize(column) / 2.0


def _report(screen, table, buttons) -> dict:
    """``{label: (button centre, column centre)}`` for the mapped three."""
    return {label: (_centre(screen, buttons[label]),
                    _column_centre(screen, table, column))
            for label, column in MAPPED.items()}


# ---------------------------------------------------------------------------
# The alignment itself
# ---------------------------------------------------------------------------


def test_each_button_sits_over_the_column_it_fills(qtbot):
    """Measured 111.0 / 125.0 / 115.5 px left of their columns before this."""
    screen, _section = _shown_regression(qtbot)
    table = _table(screen)
    buttons = _buttons(screen)
    for label, (button_centre, column_centre) in _report(
            screen, table, buttons).items():
        assert abs(button_centre - column_centre) <= TOLERANCE, (
            f"{label} is centred at {button_centre:.1f} but column "
            f"{MAPPED[label]} is centred at {column_centre:.1f} -- "
            f"{button_centre - column_centre:+.1f} px out")


def test_the_alignment_survives_a_column_resize(qtbot):
    """A width the user dragged is the width the button must follow.

    The whole reason the strip reads the header rather than being handed a
    copy of its widths: a layout built once at 100 px columns is wrong the
    moment anybody drags a section edge, and silently so.
    """
    screen, _section = _shown_regression(qtbot)
    table = _table(screen)
    buttons = _buttons(screen)
    before = _report(screen, table, buttons)
    table.table.setColumnWidth(MAPPED["Score"], 180)
    table.table.setColumnWidth(MAPPED["Measurements (.db)"], 160)
    qtbot.wait(10)
    after = _report(screen, table, buttons)
    assert after["Score"][1] != before["Score"][1], (
        "the resize did not move the column, so this proves nothing")
    for label, (button_centre, column_centre) in after.items():
        assert abs(button_centre - column_centre) <= TOLERANCE, (
            f"after the resize {label} is {button_centre - column_centre:+.1f}"
            f" px from the centre of column {MAPPED[label]}")


def test_a_button_is_never_wider_than_the_column_it_sits_over(qtbot):
    """"lock the width of the downloade table columns to the width of the
    paired data column counterparts below".

    Measured: "Measurements (.db)" asks for 127 px and the column is 100 px
    wide by default, so an unclamped button would reach 3.5 px into the Count
    button beside it -- two download buttons drawn overlapping, which is the
    confusion the alignment exists to remove.
    """
    screen, _section = _shown_regression(qtbot)
    table = _table(screen)
    buttons = _buttons(screen)
    header = table.table.horizontalHeader()
    for label, column in MAPPED.items():
        assert buttons[label].width() <= header.sectionSize(column), (
            f"{label} is {buttons[label].width()} px wide over a "
            f"{header.sectionSize(column)} px column")


def test_no_two_download_buttons_overlap(qtbot):
    """Including the one with no column, which trails the aligned run."""
    screen, _section = _shown_regression(qtbot)
    buttons = _buttons(screen)
    spans = sorted((b.mapTo(screen, QPoint(0, 0)).x(),
                    b.mapTo(screen, QPoint(0, 0)).x() + b.width(), text)
                   for text, b in buttons.items())
    for (_left, right, first), (left, _r, second) in zip(spans, spans[1:]):
        assert left >= right, (
            f"{first} ends at {right} and {second} starts at {left}")


def test_the_button_with_no_column_trails_the_aligned_ones(qtbot):
    """Image crops fills ``src``, not a column, so it is not centred on one.

    It goes AFTER the last aligned column rather than being dropped or being
    parked over "Plate rule", which it has nothing to do with either.
    """
    screen, _section = _shown_regression(qtbot)
    table = _table(screen)
    buttons = _buttons(screen)
    header = table.table.horizontalHeader()
    last = max(MAPPED.values())
    right_edge = table.table.viewport().mapTo(
        screen, QPoint(header.sectionViewportPosition(last), 0)).x() \
        + header.sectionSize(last)
    left = buttons[UNMAPPED].mapTo(screen, QPoint(0, 0)).x()
    assert left >= right_edge, (
        f"{UNMAPPED} starts at {left}, inside the aligned run that ends at "
        f"{right_edge}")
    assert left <= right_edge + 4 * TRAILING_SPACING, (
        f"{UNMAPPED} starts at {left}, far past the aligned run's "
        f"{right_edge} -- it should trail it, not float")


# ---------------------------------------------------------------------------
# How it is built, which is the half that cannot drift
# ---------------------------------------------------------------------------


def test_the_strip_follows_one_column_model(qtbot):
    """No second header, and nothing written back to the first one.

    The build the request offered -- "make another table above Paired data
    table" -- is two column models kept in step by hand, which is the drift
    the WATCH section of instruction 353 rules out. The strip holds a layout
    that READS the paired table's own header instead.
    """
    screen, _section = _shown_regression(qtbot)
    table = _table(screen)
    strip = getattr(screen, "_example_scores_button").parentWidget()
    layout = strip.layout()
    assert isinstance(layout, ColumnAlignedRow), (
        f"the Download row is laid out by {type(layout).__name__}")
    assert layout._header is table.table.horizontalHeader(), (
        "the strip follows a header that is not the paired table's")


def test_a_bare_table_aligns_nothing_and_does_not_raise(qtbot):
    """Built outside a screen there is no Download row to find.

    The widget is Regression's alone, but it is constructed bare by several
    tests and by the settings model before it is put into a form, and a
    lookup that raised there would take the input table down with it.
    """
    table = PairedFileTableWidget()
    qtbot.addWidget(table)
    table.show()
    qtbot.wait(10)
    assert table.align_download_buttons() is False


# ---------------------------------------------------------------------------
# ...and every caption still fits, which alignment alone does not give you
# ---------------------------------------------------------------------------
#
# Aligning a button to a column NARROWER than the button leaves two bad
# choices: overlap the neighbour, or clip the caption. The strip's layout
# takes the second, so at Qt's default 100 px section "Measurements (.db)"
# lost its ending -- alignment had made a readable button unreadable, which
# instruction 350 forbids in as many words.
#
# The column is widened instead, from the TABLE, which owns its header. Two
# of the five headings did not fit 100 px either, before anybody pressed
# anything: "Plate / proposal" drew as "late / propos" and "Measurements DB"
# lost its last word. The button and the heading are two names for one
# column, so the wider of them is what the column has to hold.

def test_no_heading_is_clipped_at_the_width_it_opens_at(qtbot):
    """The pre-existing half: two of five headings did not fit."""
    from PySide6.QtGui import QFontMetrics

    screen, _section = _shown_regression(qtbot)
    table = _table(screen)
    header = table.table.horizontalHeader()
    metrics = QFontMetrics(header.font())

    clipped = []
    for column in range(table.table.columnCount()):
        item = table.table.horizontalHeaderItem(column)
        text = item.text() if item is not None else ""
        needed = metrics.horizontalAdvance(text) + 18
        if header.sectionSize(column) < needed:
            clipped.append(f"{text!r} has {header.sectionSize(column)} px "
                           f"and needs {needed}")
    assert not clipped, "; ".join(clipped)


def test_alignment_never_clips_a_button_it_aligned(qtbot):
    """A button at less than its own size hint is a caption cut off."""
    screen, _section = _shown_regression(qtbot)
    _table(screen).align_download_buttons()
    qtbot.wait(10)

    clipped = [f"{b.text()!r} is {b.width()} px and wants "
               f"{b.sizeHint().width()}"
               for b in _buttons(screen).values()
               if b.width() < b.sizeHint().width()]
    assert not clipped, "; ".join(clipped)


def test_a_column_the_user_narrowed_is_not_widened_back(qtbot):
    """The widening happens once. A column dragged narrow is the user's, and
    a table that argued with every show would be unusable."""
    screen, _section = _shown_regression(qtbot)
    table = _table(screen)
    table.align_download_buttons()
    header = table.table.horizontalHeader()

    header.resizeSection(3, 60)
    table.align_download_buttons()
    qtbot.wait(10)

    assert header.sectionSize(3) == 60
