"""A run's figures fold away under their heading, the console's way.

Asked for on 2026-08-17, instruction 125 C:

    "in the figure view when all the figures are visible , each set should be
     in its own section that can be minimized like in the console."

124 B gave each run its own SECTION with its own lettering. What was missing
is the gesture: a heading that folds the set under it away. The maintainer
named the console as the model, so this pins that it IS the console's
affordance -- a disclosure chevron in the heading, a pointing hand over the
bar, the keyboard reaching it, and "reach the section first, fold it second"
on the click -- rather than a second opinion about what a collapsible section
looks like.

Two behaviours matter more than the gesture, and the instruction calls both
out by name:

  * A FOLDED RUN STAYS FOLDED when the next run finishes. A sweep of sixty
    trials that re-expanded everything on each completion would be unusable,
    and that is exactly what a naive rebuild-from-scratch does -- the grid is
    rebuilt on a debounce timer every time a figure arrives.
  * THE NEWEST RUN IS THE ONE THAT MATTERS, so it arrives open.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def _pixmaps(count, width=120, height=90):
    from PySide6.QtGui import QPixmap

    out = []
    for _ in range(count):
        pixmap = QPixmap(width, height)
        pixmap.fill()
        out.append(pixmap)
    return out


@pytest.fixture
def grid(qtbot):
    from spacr.qt.widgets.figure_grid_view import FigureGridView

    widget = FigureGridView()
    qtbot.addWidget(widget)
    widget.resize(900, 600)
    return widget


def _two_runs(grid, first=3, second=3):
    sections = [("first run", 0, first), ("second run", first, second)]
    grid.set_figures(_pixmaps(first + second),
                     [f"figure {i}" for i in range(first + second)],
                     sections=sections)
    return sections


def _header(grid, text):
    for header in grid._headers:
        if header.text() == text:
            return header
    raise AssertionError(f"no heading reads {text!r}: "
                         f"{[h.text() for h in grid._headers]}")


def _visible_cells(grid):
    """Which figures the grid would show. ``isVisibleTo``, not ``isVisible``:
    the second is False for every cell of a grid nobody has shown yet, which
    would make a folded run and an unshown window look the same."""
    return [cell.index for cell in grid._cells
            if cell.isVisibleTo(grid._body)]


# --------------------------------------------------------------------------- #
#  It is the console's affordance, not a new one
# --------------------------------------------------------------------------- #

def test_a_run_heading_carries_a_disclosure_chevron(grid):
    """A toggle with no indicator is a control users find by accident. The
    console's topic bar says so in those words; this is the same bar."""
    _two_runs(grid)

    header = _header(grid, "first run")
    assert header._chevron.text() == "▾"
    assert header.is_expanded() is True


def test_the_heading_says_it_is_clickable_before_it_is_clicked(grid):
    """A pointing hand and a focus stop, exactly as the console's topic bar:
    a control only a mouse can reach is one some users cannot reach at all."""
    from PySide6.QtCore import Qt

    _two_runs(grid)
    header = _header(grid, "first run")

    assert header.cursor().shape() == Qt.PointingHandCursor
    assert header.focusPolicy() == Qt.StrongFocus


def test_a_real_mouse_click_on_the_heading_folds_the_run_away(qtbot, grid):
    """Driven through the widget, not through the method it calls. A fold
    that works in theory and not in the widget is the failure mode here."""
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    _two_runs(grid)
    grid.show()
    qtbot.waitExposed(grid)
    assert _visible_cells(grid) == [0, 1, 2, 3, 4, 5]

    header = _header(grid, "first run")
    QTest.mouseClick(header, Qt.LeftButton, pos=QPoint(5, 5))
    qtbot.wait(1)

    assert _visible_cells(grid) == [3, 4, 5], (
        "clicking the heading did not fold the run's figures away")
    assert _header(grid, "first run")._chevron.text() == "▸"


def test_the_keyboard_folds_a_run_too(qtbot, grid):
    """Return, Enter or Space -- the keys the console's heading answers."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    _two_runs(grid)
    grid.show()
    qtbot.waitExposed(grid)

    QTest.keyClick(_header(grid, "first run"), Qt.Key_Return)
    qtbot.wait(1)

    assert _visible_cells(grid) == [3, 4, 5]


def test_a_second_press_brings_the_run_back(qtbot, grid):
    _two_runs(grid)
    grid.show()
    qtbot.waitExposed(grid)

    grid.toggle_section(_header(grid, "first run"))
    assert _visible_cells(grid) == [3, 4, 5]
    grid.toggle_section(_header(grid, "first run"))

    assert _visible_cells(grid) == [0, 1, 2, 3, 4, 5]
    assert _header(grid, "first run")._chevron.text() == "▾"


def test_a_heading_below_the_fold_is_reached_before_it_is_folded(qtbot, grid):
    """THE CONSOLE'S RULE, and the reason it exists: a first click that hid
    the very section the user was reaching for spends the gesture on the
    opposite of what it looked like. Only a heading already at the top has
    nowhere left to go, and there folding is what the click can still mean."""
    grid.resize(700, 300)
    grid.set_figures(_pixmaps(24), [f"figure {i}" for i in range(24)],
                     sections=[("first run", 0, 12), ("second run", 12, 12)])
    grid.show()
    qtbot.waitExposed(grid)

    bar = grid.verticalScrollBar()
    bar.setValue(0)
    assert bar.maximum() > 0, (
        "this grid does not scroll, so 'reached' cannot be told from 'not "
        "reached' and the test proves nothing")

    header = _header(grid, "second run")
    assert grid._is_raised(header) is False
    grid.toggle_section(header)
    qtbot.wait(10)

    assert grid.is_section_collapsed("second run", 12) is False, (
        "the first click folded a heading the user was still reaching for")
    assert bar.value() > 0, "the first click did not go to the section"

    # Now it IS at the top, and the same gesture means the one thing left.
    grid.toggle_section(_header(grid, "second run"))
    assert grid.is_section_collapsed("second run", 12) is True
    assert _visible_cells(grid) == list(range(12))


# --------------------------------------------------------------------------- #
#  What a sweep of sixty trials needs
# --------------------------------------------------------------------------- #

def test_a_folded_run_stays_folded_when_the_next_run_arrives(qtbot, grid):
    """The grid is rebuilt from scratch every time a figure lands, so the
    fold cannot live on the heading widget. A run that re-expanded everything
    each time it finished would be unusable during a sixty-trial sweep."""
    _two_runs(grid)
    grid.show()
    qtbot.waitExposed(grid)
    grid.set_section_collapsed("first run", 0, True)
    assert _visible_cells(grid) == [3, 4, 5]

    grid.set_figures(_pixmaps(9), [f"figure {i}" for i in range(9)],
                     sections=[("first run", 0, 3), ("second run", 3, 3),
                               ("third run", 6, 3)])

    assert grid.is_section_collapsed("first run", 0) is True
    assert _visible_cells(grid) == [3, 4, 5, 6, 7, 8]
    assert _header(grid, "first run")._chevron.text() == "▸"


def test_the_newest_run_arrives_open(qtbot, grid):
    """"The newest section is the one that matters and should be the one
    open." A run nobody has folded has never been folded."""
    _two_runs(grid)
    grid.set_section_collapsed("first run", 0, True)
    grid.set_section_collapsed("second run", 3, True)

    grid.set_figures(_pixmaps(9), [f"figure {i}" for i in range(9)],
                     sections=[("first run", 0, 3), ("second run", 3, 3),
                               ("third run", 6, 3)])

    assert _header(grid, "third run").is_expanded() is True
    assert _visible_cells(grid) == [6, 7, 8]


def test_a_run_that_keeps_growing_stays_folded(qtbot, grid):
    """A section's count changes while its run streams figures in; its start
    does not. Keying the fold on the count would unfold it on every arrival."""
    _two_runs(grid, first=3, second=1)
    grid.set_section_collapsed("first run", 0, True)

    for extra in range(2, 5):
        grid.set_figures(_pixmaps(3 + extra),
                         [f"figure {i}" for i in range(3 + extra)],
                         sections=[("first run", 0, 3),
                                   ("second run", 3, extra)])
        assert grid.is_section_collapsed("first run", 0) is True
        assert _visible_cells(grid) == list(range(3, 3 + extra))


def test_a_folded_run_leaves_no_hole_for_the_next_one(qtbot, grid):
    """Hidden is not enough: a widget left in the layout keeps its cell, so
    the next run would start four rows below a one-line heading."""
    _two_runs(grid)
    grid.show()
    qtbot.waitExposed(grid)

    def row_of(cell):
        index = grid._grid.indexOf(cell)
        assert index >= 0, "the cell is not in the layout at all"
        return grid._grid.getItemPosition(index)[0]

    before = row_of(grid._cells[3])
    grid.set_section_collapsed("first run", 0, True)
    after = row_of(grid._cells[3])

    assert after < before, (
        f"the second run did not move up (row {before} -> {after}); the "
        "folded run is still taking its cells")


def test_a_folded_figure_stops_painting_itself(qtbot, grid):
    """A widget taken out of a layout is not hidden by that -- it keeps
    drawing where it last was, over whatever now occupies the space."""
    _two_runs(grid)
    grid.show()
    qtbot.waitExposed(grid)

    grid.set_section_collapsed("first run", 0, True)

    assert [cell.isVisible() for cell in grid._cells[:3]] == [False] * 3


# --------------------------------------------------------------------------- #
#  The headings themselves
# --------------------------------------------------------------------------- #

def test_a_resize_does_not_leave_a_second_copy_of_every_heading(qtbot, grid):
    """Found while making the heading a control. `takeAt` drops the layout
    item and leaves the widget a visible child at its old geometry, so every
    relayout -- and a window resize is one -- painted another heading. Three
    relayouts of a two-run grid left six. A ghost heading is now a ghost
    CONTROL, which is why this is pinned rather than tolerated."""
    from PySide6.QtWidgets import QLabel

    _two_runs(grid)
    grid.show()
    qtbot.waitExposed(grid)
    grid.resize(700, 520)
    qtbot.wait(1)
    grid._relayout()
    grid._relayout()

    texts = [w.text() for w in grid._body.findChildren(QLabel)]
    assert texts.count("first run") == 1, texts
    assert texts.count("second run") == 1, texts


def test_one_run_IS_offered_a_fold(grid):
    """THE FIRST RUN FOLDS TOO.

    This asserted the opposite, on 124 B's argument that a heading over the
    only section is furniture -- which was about the LABEL, and true of it.
    But the heading is also the FOLD CONTROL, so with one run there was
    nothing to click and the maintainer reported the figures as "still not
    colapsable into runs" while the folding worked perfectly from the second
    run onwards. Changed 2026-08-17 at their request.
    """
    grid.set_figures(_pixmaps(3), ["a", "b", "c"],
                     sections=[("the only run", 0, 3)])

    assert [h.text() for h in grid._headers] == ["the only run"]
    assert _visible_cells(grid) == [0, 1, 2]

    grid.set_section_collapsed("the only run", 0, True)
    assert _visible_cells(grid) == []


def test_a_stale_fold_cannot_hide_the_only_run_on_screen(grid):
    """Picking a saved trial replaces the grid with pictures from disk and no
    sections at all. A fold remembered from the run before must not blank it."""
    _two_runs(grid)
    grid.set_section_collapsed("first run", 0, True)

    grid.set_figures(_pixmaps(2), ["one", "two"])

    assert _visible_cells(grid) == [0, 1]


def test_the_heading_text_is_still_the_heading_text(grid):
    """Making it a control must not make it a different heading: 124 B's
    wording and its lettering are what the reader is orienting by."""
    _two_runs(grid)

    assert [h.text() for h in grid._headers] == ["first run", "second run"]
    assert [cell.letter for cell in grid._cells] == \
        ["A", "B", "C", "A", "B", "C"]
