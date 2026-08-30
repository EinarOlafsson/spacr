"""Choosing a colour, choosing a shape, and a page that will not be read.

Pins the Save-figure dialog's less-travelled routes: the composer that
punctuates a value and its explanation, a build with no shape table to read,
the colour chooser in each of the four states it can come back in, a fast
plot whose page size cannot be read, and a Matplotlib figure reshaped by
writing its height.
"""
from __future__ import annotations

import sys
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from PySide6.QtGui import QColor

from spacr.qt.widgets import save_figure_dialog as SFD
from spacr.qt.widgets.save_figure_dialog import SaveFigureDialog


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _figure(width: float = 8.0, height: float = 6.0):
    fig = plt.figure(figsize=(width, height))
    axis = fig.add_subplot()
    axis.plot([0, 1], [0, 1])
    return fig


class _FastPlot:
    """A pyqtgraph-shaped plot whose page size can be told to fail."""

    def __init__(self, size=(100.0, 80.0)):
        self._size = size

    def export_size(self):
        if isinstance(self._size, Exception):
            raise self._size
        return self._size

    def styled_snapshot(self, width, **kwargs):
        return None

    def export_styled(self, path, **kwargs):
        return path


# --------------------------------------------------------------------------
# one composer for a value and its explanation
# --------------------------------------------------------------------------

def test_a_value_with_nothing_to_explain_is_left_exactly_as_it_is():
    """One composer, so a label and the note beside it are punctuated the
    same way. A value with no sentence behind it must not gain a dash."""
    assert SFD._with_reason("300 × 200 pixels") == "300 × 200 pixels"
    assert SFD._with_reason("300 × 200 pixels", "set on the plot's menu") == \
        "300 × 200 pixels — set on the plot's menu"
    assert SFD._with_reason("", "set on the plot's menu") == \
        "set on the plot's menu"


# --------------------------------------------------------------------------
# a build with no shape table
# --------------------------------------------------------------------------

def test_with_no_shape_table_the_dialog_offers_only_the_shape_as_drawn(
        qtbot, monkeypatch):
    """The shape names are read from the graph's own menu so the two agree.
    A build that cannot reach that table still has to open -- with the one
    shape it can honour, which is the figure as it stands."""
    ordinary = SaveFigureDialog(_figure())
    qtbot.addWidget(ordinary)
    assert ordinary.graph_shape.count() > 1
    assert ordinary._shape_ratios["square"] == 1.0

    empty = types.ModuleType("spacr.qt.widgets.fast_plots")
    monkeypatch.setitem(sys.modules, "spacr.qt.widgets.fast_plots", empty)
    dialog = SaveFigureDialog(_figure())
    qtbot.addWidget(dialog)

    assert dialog.graph_shape.count() == 1
    assert dialog.graph_shape.itemData(0) == ""
    assert dialog._shape_ratios == {}


# --------------------------------------------------------------------------
# the colour chooser
# --------------------------------------------------------------------------

def _chooser_row(box) -> int:
    return box.findData(SFD._CHOOSE)


def _answer(monkeypatch, colour):
    from spacr.qt.widgets import colour_picker

    monkeypatch.setattr(colour_picker, "pick_colour",
                        lambda *args, **kwargs: colour)


def test_a_cancelled_colour_chooser_puts_the_combo_back(qtbot, monkeypatch):
    """The sentinel is not a colour. Left selected it would be read back as
    one, so a cancelled chooser has to return the combo to its first entry."""
    dialog = SaveFigureDialog(_figure())
    qtbot.addWidget(dialog)
    box = dialog.background
    box.setCurrentIndex(2)                       # black

    _answer(monkeypatch, QColor())               # cancelled: not a valid colour
    box.setCurrentIndex(_chooser_row(box))
    assert box.currentIndex() == 0
    assert box.currentData() == ""               # transparent, not the sentinel


def test_a_chosen_colour_the_list_already_has_selects_that_row(qtbot,
                                                               monkeypatch):
    """``QColor.name()`` answers in lower case and the shipped entries are
    written upper, so an exact match misses white for #ffffff and inserts a
    second, visually identical row -- once for every time it is picked."""
    dialog = SaveFigureDialog(_figure())
    qtbot.addWidget(dialog)
    box = dialog.background
    before = box.count()
    white = box.findData("#FFFFFF")

    _answer(monkeypatch, QColor("#ffffff"))
    box.setCurrentIndex(_chooser_row(box))
    assert box.count() == before
    assert box.currentIndex() == white

    # Picked a second time it still lands on the same row.
    box.setCurrentIndex(_chooser_row(box))
    assert box.count() == before
    assert box.currentIndex() == white


def test_a_colour_the_list_does_not_have_is_added_before_the_chooser(
        qtbot, monkeypatch):
    """The chooser stays last, because a row after it would be a colour the
    user has to scroll past the chooser to find."""
    dialog = SaveFigureDialog(_figure())
    qtbot.addWidget(dialog)
    box = dialog.background
    before = box.count()

    _answer(monkeypatch, QColor("#123456"))
    box.setCurrentIndex(_chooser_row(box))

    assert box.count() == before + 1
    assert box.currentData() == "#123456"
    assert box.itemText(box.count() - 1) == SFD._CHOOSE_LABEL
    added = box.count() - 2
    assert box.currentIndex() == added

    # Picked again it is now a colour the list HAS, found by an exact match,
    # so the row is selected rather than added a second time.
    box.setCurrentIndex(_chooser_row(box))
    assert box.count() == before + 1
    assert box.currentIndex() == added


def test_choosing_the_row_that_is_already_the_colour_opens_no_chooser(
        qtbot, monkeypatch):
    """Only the chooser row opens a chooser. Every other selection is a
    colour already, and re-asking would make picking one impossible."""
    dialog = SaveFigureDialog(_figure())
    qtbot.addWidget(dialog)
    box = dialog.background

    asked = []

    def _record(*args, **kwargs):
        asked.append(args)
        return QColor("#123456")

    from spacr.qt.widgets import colour_picker

    monkeypatch.setattr(colour_picker, "pick_colour", _record)
    box.setCurrentIndex(2)                       # a colour, not the chooser
    assert asked == []
    assert box.currentData() == "#000000"

    box.setCurrentIndex(_chooser_row(box))       # the chooser row does ask
    assert len(asked) == 1


# --------------------------------------------------------------------------
# a page that cannot be read
# --------------------------------------------------------------------------

def test_a_plot_that_will_not_give_its_page_size_shows_no_page(qtbot):
    """The millimetres are the plot's own answer. A plot that cannot give
    them leaves the row blank rather than printing a number this dialog made
    up, and the dialog still opens."""
    good = SaveFigureDialog(_FastPlot((100.0, 80.0)))
    qtbot.addWidget(good)
    assert good._page_mm() == (100.0, 80.0)
    assert "100 × 80 mm" in good._size_note.text()

    broken = SaveFigureDialog(_FastPlot(RuntimeError("no page here")))
    qtbot.addWidget(broken)
    assert broken._page_mm() == (None, None)
    assert broken._size_note.text() == ""


# --------------------------------------------------------------------------
# reshaping a Matplotlib page
# --------------------------------------------------------------------------

def test_a_shape_chosen_for_a_matplotlib_figure_is_written_into_its_height(
        qtbot):
    """A pyqtgraph plot's page is millimetres and is recomputed; a
    Matplotlib figure's page IS the inches in the size row. Writing the
    height there keeps rule 3, because the number shown is then the number
    the save uses."""
    dialog = SaveFigureDialog(_figure(8.0, 6.0))
    qtbot.addWidget(dialog)
    assert dialog._drawn_ratio == pytest.approx(0.75)
    assert dialog.height.value() == pytest.approx(6.0)

    square = dialog.graph_shape.findData("square")
    dialog.graph_shape.setCurrentIndex(square)
    assert dialog.height.value() == pytest.approx(dialog.width.value())

    tall = dialog.graph_shape.findData("tall")
    dialog.graph_shape.setCurrentIndex(tall)
    assert dialog.height.value() == pytest.approx(
        round(dialog.width.value() * 1.5, 2))

    dialog.graph_shape.setCurrentIndex(0)        # back to "as drawn"
    assert dialog.height.value() == pytest.approx(
        round(dialog.width.value() * 0.75, 2))


# --------------------------------------------------------------------------
# `_clear_holder`'s "if widget is not None" has no false side to reach, and
# is left standing rather than silenced. `self._holder` is a private layout
# and nothing outside this module names it; the only four things ever put
# into it are `addWidget` calls (the un-previewable note, the preview canvas,
# the failure label and the empty-plot label), so every item `takeAt` hands
# back owns a widget. A spacer or a nested layout would make the guard fire,
# and no code path adds one.
