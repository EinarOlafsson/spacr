"""Frames that repaint nothing, words that are not baked, and a button with
nowhere obvious to sit.

The rain is a decorative backdrop, so every one of these paths is about it
staying cheap and staying out of the way: a frame in which nothing moved must
cost no repaint, a multi-cell word must not be flattened into the per-column
strip, and a DNA button that cannot find its usual neighbour must still be
placed rather than lost.
"""
from __future__ import annotations

import builtins

import pytest

from PySide6.QtWidgets import QGridLayout, QPushButton, QVBoxLayout, QWidget

from spacr.qt.widgets.dna_rain import (
    DnaRainEngine, DnaRainWidget, _find_ai_toggle, _place_beside)


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

def test_a_frame_in_which_nothing_moved_a_pixel_repaints_nothing():
    """Dirty spans are reported on PIXEL movement, not on elapsed time.

    A column at four cells a second moves less than a pixel in a millisecond.
    Reporting it dirty anyway would repaint the whole canvas every frame for
    no visible change, which is the cost this quantisation exists to avoid.
    """
    engine = DnaRainEngine(width=200, height=200, font_size=14, seed=3)
    engine.advance(0.5)
    before = [column.y_px for column in engine.columns]

    assert engine.advance(1e-9) == []
    assert [column.y_px for column in engine.columns] == before


def test_a_frame_in_which_something_moved_reports_the_span():
    """The skip above must not have turned the animation off entirely."""
    engine = DnaRainEngine(width=200, height=200, font_size=14, seed=3)

    assert engine.advance(0.5), "half a second moves at least one column"


# ---------------------------------------------------------------------------
# The pre-rendered strip
# ---------------------------------------------------------------------------

@pytest.mark.qt
def test_a_spliced_word_is_not_baked_into_the_column_strip(qapp):
    """A word wider than a cell is drawn live at full canvas width.

    Baking it into the one-cell-wide strip would clip every character after
    the first, so the splice has to be skipped here and painted separately.
    The strip must therefore come out identical to one whose cell is empty.
    """
    widget = DnaRainWidget(None, seed=5, font_size=14)
    widget.resize(200, 200)
    column = widget._engine.columns[0]
    assert column.length > 4

    column.tokens = list(column.tokens)
    column.tokens[2] = "spaCR"
    column.word_index = 2
    with_word = widget._render_strip(column).toImage()

    column.tokens[2] = ""
    column.word_index = -1
    with_gap = widget._render_strip(column).toImage()

    assert with_word == with_gap


# ---------------------------------------------------------------------------
# Placing the DNA button
# ---------------------------------------------------------------------------

@pytest.mark.qt
def test_a_host_whose_ai_toggle_cannot_be_looked_up_simply_has_none(
        qapp, monkeypatch):
    """The button's preferred neighbour is optional, and so is finding it.

    A host built without the AI toggle module available must fall back to the
    caller's own placement rather than taking the screen down while arranging
    a decorative control.
    """
    real_import = builtins.__import__

    def _no_toggle(name, *args, **kwargs):
        if name.endswith("ai_toggle_label"):
            raise ImportError("no AI toggle here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_toggle)

    assert _find_ai_toggle(QWidget()) is None


@pytest.mark.qt
def test_an_anchor_with_no_layout_cannot_be_sat_beside(qapp):
    """A widget whose parent lays nothing out has no slot to insert into.

    Saying so lets the caller fall back to its own layout; claiming success
    would leave the button parented but never positioned, which is a control
    the user cannot see or reach.
    """
    parent = QWidget()
    anchor = QPushButton("AI", parent)

    assert _place_beside(QPushButton("DNA"), anchor) is False


@pytest.mark.qt
def test_an_anchor_in_a_grid_still_gets_the_button_beside_it(qapp):
    """Not every chrome row is a box layout, and the button still has to land.

    A grid has no notion of inserting before a widget, so the button is
    appended instead: beside nothing is better than not on screen at all.
    """
    parent = QWidget()
    grid = QGridLayout(parent)
    anchor = QPushButton("AI")
    grid.addWidget(anchor, 0, 0)
    button = QPushButton("DNA")

    assert _place_beside(button, anchor) is True
    assert grid.indexOf(button) >= 0


@pytest.mark.qt
def test_an_anchor_in_a_row_keeps_the_button_before_it(qapp):
    """Before, not after: the AI toggle is followed by its provider chevron.

    Splitting that pair would read as the chevron belonging to DNA.
    """
    parent = QWidget()
    row = QVBoxLayout(parent)
    anchor = QPushButton("AI")
    row.addWidget(anchor)
    button = QPushButton("DNA")

    assert _place_beside(button, anchor) is True
    assert row.indexOf(button) < row.indexOf(anchor)
