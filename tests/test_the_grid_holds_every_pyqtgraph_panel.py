"""The all-figures grid carries every pyqtgraph panel, not just the volcano.

Instruction 129 C, asked for on 2026-08-17:

    "i would still like to retain the grid to the right"
    "so same information graphed, but graphed with pyqtgraph. same grid
     overview but pyqtgraph versions and new tabs for each graph."

The grid already had ONE live tile -- the interactive volcano, pinned by
`set_pinned`. This is the general form: a tile per pyqtgraph panel, in a
section of their own that folds like a run's.

THE TILES ARE PICTURES, AND THAT WAS MEASURED RATHER THAN ASSUMED. The
instruction says "MEASURE IT before choosing, the way the 49 ms redraw was
measured", so it was, on the real widgets under the offscreen platform at the
volcano's 1,213 points:

    per window-drag frame     3 tiles   6 tiles  12 tiles  18 tiles
      live pyqtgraph widgets  11.98 ms  25.68 ms  48.55 ms  74.99 ms
      snapshot pictures        1.08 ms   1.64 ms   3.59 ms   5.19 ms
    resident memory
      live pyqtgraph widgets   7.6 MB   13.7 MB   29.8 MB   44.0 MB
      snapshot pictures        3.3 MB    5.8 MB    9.6 MB   14.7 MB

A window drag emits a resize per frame and a resize is a full relayout, so
the budget is 16.7 ms: six live tiles already miss it. Hover -- the cost the
instruction named -- is real and is the smaller half: 60 mouse-moves over one
live volcano cost 12.31 ms against 0.52 ms over a picture of it.

THREE THINGS THE INSTRUCTION CALLS OUT BY NAME are what most of this file
tests:

  * the index mapping stays honest. `_FigureCell.index` is the position in
    the pixmap list and `figure_activated` hands it straight to
    `FigureQueue.show_index`, so anything INSERTED into that list opens the
    wrong figure from every tile after it.
  * the collapsible run sections keep working, including for a single run.
  * the tiles do not accumulate. "pin one volcano, not every volcano so far"
    was the 2026-08-17 bug (`33e1c8ee`); with a tile per panel the same
    mistake would stack every panel, once per 250 ms refresh.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


# --------------------------------------------------------------------------- #
#  Pictures, shaped like what FastPlot.snapshot actually returns
# --------------------------------------------------------------------------- #

def _dotted(x, colour, width=200, height=140):
    """A TRANSPARENT page with one opaque dot, at ``x``.

    The shape `FastPlot.snapshot` returns -- the exporter is handed a
    transparent background so a tile does not sit on a slab. Each photograph
    puts its dot somewhere else, so counting the dots visible at once counts
    the photographs painted at once.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QPainter, QPixmap

    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(colour))
    painter.drawEllipse(x, 50, 30, 30)
    painter.end()
    return pixmap


def _block(colour="#777777", width=160, height=120):
    from PySide6.QtGui import QColor, QPixmap

    pixmap = QPixmap(width, height)
    pixmap.fill(QColor(colour))
    return pixmap


def _tiles(grid):
    """Every tile still parented to the grid's BODY.

    The body, not the layout: a tile that has left the layout and is still a
    visible child painting itself is the whole of the stacking bug, and a
    count taken off the layout reports it as gone.
    """
    from spacr.qt.widgets.figure_grid_view import _FigureCell

    return [child for child in grid._body.children()
            if isinstance(child, _FigureCell)]


def _blobs(columns, gap=6) -> int:
    """How many separate dots the lit columns describe.

    Counted as runs rather than by bucketing x into bands: a 30 px dot lands
    wherever the layout puts it and straddles any fixed band boundary sooner
    or later, which counts one dot as two and fails a passing test.
    """
    if not columns:
        return 0
    ordered = sorted(columns)
    return 1 + sum(1 for before, after in zip(ordered, ordered[1:])
                   if after - before > gap)


def _panel_tiles(keys=("regression", "qq", "controls")):
    return [(key, _block(), key.title()) for key in keys]


@pytest.fixture
def grid(qtbot):
    from spacr.qt.widgets.figure_grid_view import FigureGridView

    view = FigureGridView()
    qtbot.addWidget(view)
    view.resize(900, 640)
    return view


def _run_of(grid, count=4, label="run one"):
    grid.set_figures([_block() for _ in range(count)],
                     [f"figure {i}" for i in range(count)],
                     sections=[(label, 0, count)])


# --------------------------------------------------------------------------- #
#  A tile per panel, named by what it is
# --------------------------------------------------------------------------- #

def test_every_pyqtgraph_panel_gets_its_own_tile(grid):
    """"same grid overview but pyqtgraph versions" -- one tile per panel, not
    one tile for the volcano and nothing for the other seven."""
    assert grid.set_live_tiles(_panel_tiles()) == 3

    assert grid.live_tile_keys() == ["regression", "qq", "controls"]
    assert len(_tiles(grid)) == 3


def test_a_tile_is_identified_by_its_panel_not_by_its_position(grid):
    """A caller has to know WHICH graph to raise, and position cannot say: the
    panel set varies with the run -- a fit with no model has no residual plot
    -- so slot 2 is a different graph on two different runs."""
    grid.set_live_tiles(_panel_tiles(("regression", "qq", "residuals")))

    opened = []
    grid.live_tile_activated.connect(opened.append)
    for cell in grid._live:
        cell.clicked.emit(cell.index)

    assert opened == ["regression", "qq", "residuals"]


def test_a_panel_that_cannot_be_photographed_gets_no_tile_at_all(grid):
    """A residual plot with nothing fitted returns None from snapshot(), and
    an empty tile invites a click that opens an empty plot. It must also be
    absent from `live_tile_keys`, or a caller cannot tell "no data" from
    "drawn blank"."""
    from PySide6.QtGui import QPixmap

    shown = grid.set_live_tiles([
        ("regression", _block(), "Volcano"),
        ("residuals", None, "Residuals"),
        ("qq", QPixmap(), "Q-Q"),
    ])

    assert shown == 1
    assert grid.live_tile_keys() == ["regression"]
    assert len(_tiles(grid)) == 1


def test_the_helper_drops_a_panel_that_photographs_as_nothing():
    """The rule lives in one place because there are eight panels and 129 B
    adds more; re-derived per call site it gets it wrong somewhere."""
    from spacr.qt.widgets.figure_grid_view import live_tiles_from_panels

    class _Photographs:
        def __init__(self, pixmap):
            self._pixmap = pixmap

        def snapshot(self):
            return self._pixmap

    class _Raises:
        def snapshot(self):
            raise RuntimeError("the scene was rebuilt under the exporter")

    tiles = live_tiles_from_panels([
        ("volcano", "Volcano", _Photographs(_block())),
        ("residuals", "Residuals", _Photographs(None)),
        ("qq", "Q-Q", _Raises()),
        ("controls", "Controls", None),
    ])

    assert [key for key, _pixmap, _title in tiles] == ["volcano"]
    assert tiles[0][2] == "Volcano"


def test_a_panel_whose_photograph_raises_costs_a_tile_and_not_the_screen():
    """`snapshot` renders through pyqtgraph's exporter, into a scene the panel
    may be part-way through rebuilding. A photograph is never worth a
    traceback -- the tile is simply missing this refresh."""
    from spacr.qt.widgets.figure_grid_view import live_tiles_from_panels

    class _Raises:
        def snapshot(self):
            raise ValueError("no")

    assert live_tiles_from_panels([("qq", "Q-Q", _Raises())]) == []


# --------------------------------------------------------------------------- #
#  The index mapping, which is the part the instruction says is hard
# --------------------------------------------------------------------------- #

def test_eight_live_tiles_do_not_shift_which_figure_a_tile_opens(grid):
    """THE failure 129 C names: "clicking tile 4 opens figure 7".

    `figure_activated` carries the position in the pixmap list straight into
    `FigureQueue.show_index`. Eight live tiles inserted into that list would
    move every figure eight places along.
    """
    grid.set_live_tiles(_panel_tiles(
        ("regression", "rank", "spread", "controls",
         "agreement", "p_values", "qq", "residuals")))
    _run_of(grid, count=6)

    opened = []
    grid.figure_activated.connect(opened.append)
    for cell in grid._cells:
        cell.clicked.emit(cell.index)

    assert opened == [0, 1, 2, 3, 4, 5]
    assert [cell.index for cell in grid._cells] == [0, 1, 2, 3, 4, 5]


def test_a_live_tile_is_never_one_of_the_runs_figures(grid):
    """It carries -1 and stays out of `_cells`, so no sentinel index can ever
    travel down the shared signal and open a figure that is not there."""
    grid.set_live_tiles(_panel_tiles())
    _run_of(grid, count=3)

    assert [cell.index for cell in grid._live] == [-1, -1, -1]
    assert not set(map(id, grid._live)) & set(map(id, grid._cells))

    opened = []
    grid.figure_activated.connect(opened.append)
    for cell in grid._live:
        cell.clicked.emit(cell.index)

    assert opened == [], "a live tile reached figure_activated"


def test_the_live_tiles_take_no_panel_letters(grid):
    """A panel letter belongs to the published figure, and the interactive
    graphs are not on it. Lettering that counted them would call the run's
    first panel D."""
    grid.set_live_tiles(_panel_tiles())
    _run_of(grid, count=3)

    assert [cell.letter for cell in grid._live] == ["", "", ""]
    assert [cell.letter for cell in grid._cells] == ["A", "B", "C"]


# --------------------------------------------------------------------------- #
#  They do not accumulate -- "pin one volcano, not every volcano so far"
# --------------------------------------------------------------------------- #

def test_twelve_refreshes_leave_one_tile_per_panel_not_twelve(grid):
    """The 2026-08-17 bug, generalised. The screen re-photographs its panels
    on a 250 ms debounce, on every return to the grid and after every restyle,
    so a set that is not destroyed accumulates once per refresh -- and
    `snapshot` returns a TRANSPARENT pixmap, so every buried copy shows
    through the ones in front."""
    for _ in range(12):
        assert grid.set_live_tiles(_panel_tiles()) == 3

    assert len(_tiles(grid)) == 3


def test_only_the_newest_photograph_of_each_panel_is_painted(grid, qtbot):
    """The visible symptom, driven end to end.

    Two panels, three refreshes, every photograph's dot somewhere else. If
    the tiles stack, six dots are on screen at once. The event loop is turned
    BETWEEN refreshes because a stray only paints once it has been through a
    show and a layout pass -- which is exactly what a real screen gives it,
    seconds apart on the debounce.
    """
    grid.show()
    qtbot.waitExposed(grid)
    for x, first, second in ((10, "#ff0000", "#00ff00"),
                             (70, "#ff0000", "#00ff00"),
                             (130, "#ff0000", "#00ff00")):
        grid.set_live_tiles([("regression", _dotted(x, first), "a"),
                             ("qq", _dotted(x, second), "b")])
        qtbot.wait(10)

    image = grid._body.grab().toImage()
    red, green = set(), set()
    for y in range(image.height()):
        for x in range(image.width()):
            colour = image.pixelColor(x, y)
            if colour.alpha() < 200:
                continue
            if colour.red() > 200 and colour.green() < 60:
                red.add(x)
            elif colour.green() > 200 and colour.red() < 60:
                green.add(x)

    assert _blobs(red) == 1, (
        f"{_blobs(red)} volcano photographs painted at once")
    assert _blobs(green) == 1, (
        f"{_blobs(green)} Q-Q photographs painted at once")


def test_a_panel_that_vanishes_takes_its_tile_with_it(grid):
    """A re-fit that lost its model has no residual plot any more. Leaving the
    tile behind leaves a photograph of a fit that no longer exists, which is
    worse than an empty tile: it looks current."""
    grid.set_live_tiles(_panel_tiles(("regression", "qq", "residuals")))
    assert len(_tiles(grid)) == 3

    grid.set_live_tiles(_panel_tiles(("regression",)))

    assert grid.live_tile_keys() == ["regression"]
    assert len(_tiles(grid)) == 1


def test_clearing_the_live_tiles_leaves_the_grid_with_none(grid):
    """Before a run there is nothing to photograph, and "nothing" has to mean
    no widget -- not an unparented one still painting the last screen."""
    grid.set_live_tiles(_panel_tiles())

    assert grid.set_live_tiles([]) == 0
    assert grid.live_tile_keys() == []
    assert _tiles(grid) == []


def test_the_live_tiles_survive_the_runs_figures_being_replaced(grid):
    """`clear()` replaces the run's figures. The interactive graphs are not
    the run's figures, and a run streaming new ones must not make them
    disappear."""
    grid.set_live_tiles(_panel_tiles())

    _run_of(grid, count=2)

    assert grid.live_tile_keys() == ["regression", "qq", "controls"]
    assert len(_tiles(grid)) == 5           # three live plus two figures


# --------------------------------------------------------------------------- #
#  The section folds, and folding it is not folding the run
# --------------------------------------------------------------------------- #

def _header(grid, text):
    for header in grid._headers:
        if header.text() == text:
            return header
    raise AssertionError(f"no heading reads {text!r}: "
                         f"{[h.text() for h in grid._headers]}")


def test_the_live_tiles_sit_under_a_heading_of_their_own(grid):
    """Eight tiles is three rows of the grid before the run's figures begin.
    125 C's argument for folding a run applies exactly, and there was
    previously no control that could put them away."""
    from spacr.qt.widgets.figure_grid_view import LIVE_SECTION_LABEL

    grid.set_live_tiles(_panel_tiles())
    _run_of(grid, count=3)

    assert [h.text() for h in grid._headers] == [LIVE_SECTION_LABEL,
                                                 "run one"]
    assert _header(grid, LIVE_SECTION_LABEL).is_expanded() is True


def test_a_grid_with_no_live_tiles_grows_no_empty_heading(grid):
    """A fold control for nothing, on every module screen in the application,
    none of which has a live plot at all."""
    _run_of(grid, count=3)

    assert [h.text() for h in grid._headers] == ["run one"]


def test_clicking_the_live_heading_folds_the_interactive_graphs_away(
        qtbot, grid):
    """Driven through the widget. It is the console's gesture, through the
    same `toggle_section`, so a heading already at the top folds on the press
    where the user's hand already is."""
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from spacr.qt.widgets.figure_grid_view import LIVE_SECTION_LABEL

    grid.set_live_tiles(_panel_tiles())
    _run_of(grid, count=3)
    grid.show()
    qtbot.waitExposed(grid)
    assert all(cell.isVisibleTo(grid._body) for cell in grid._live)

    QTest.mouseClick(_header(grid, LIVE_SECTION_LABEL), Qt.LeftButton,
                     pos=QPoint(5, 5))

    assert grid.is_live_section_collapsed() is True
    assert not any(cell.isVisibleTo(grid._body) for cell in grid._live)


def test_folding_the_interactive_graphs_leaves_the_runs_figures_showing(grid):
    """Two sections, two folds. One collapsed set, and a key that cannot
    collide: a run's start is a position in the figure list and is never
    negative."""
    from spacr.qt.widgets.figure_grid_view import (LIVE_SECTION_LABEL,
                                                   LIVE_SECTION_START)

    grid.set_live_tiles(_panel_tiles())
    _run_of(grid, count=3)

    grid.set_section_collapsed(LIVE_SECTION_LABEL, LIVE_SECTION_START, True)

    assert not any(cell.isVisibleTo(grid._body) for cell in grid._live)
    assert all(cell.isVisibleTo(grid._body) for cell in grid._cells)
    assert grid.is_section_collapsed("run one", 0) is False


def test_the_only_run_still_folds_with_the_interactive_graphs_above_it(grid):
    """129 C: "the collapsible run sections must keep working". The single-run
    case is the one that was broken last time -- the heading only appeared
    from the second run onwards, so there was nothing to click."""
    grid.set_live_tiles(_panel_tiles())
    _run_of(grid, count=3, label="the only run")

    grid.set_section_collapsed("the only run", 0, True)

    assert not any(cell.isVisibleTo(grid._body) for cell in grid._cells)
    assert all(cell.isVisibleTo(grid._body) for cell in grid._live), (
        "folding the run took the interactive graphs with it")


def test_a_folded_live_section_stays_folded_when_the_next_run_arrives(grid):
    """The grid is rebuilt on a debounce whenever a figure lands. A fold that
    came undone on each rebuild is the unusable sweep 125 C exists to
    prevent, and the live tiles are refreshed more often than any run."""
    from spacr.qt.widgets.figure_grid_view import (LIVE_SECTION_LABEL,
                                                   LIVE_SECTION_START)

    grid.set_live_tiles(_panel_tiles())
    grid.set_section_collapsed(LIVE_SECTION_LABEL, LIVE_SECTION_START, True)

    grid.set_live_tiles(_panel_tiles())
    _run_of(grid, count=3)

    assert grid.is_live_section_collapsed() is True
    assert not any(cell.isVisibleTo(grid._body) for cell in grid._live)


def test_a_resize_does_not_leave_a_second_copy_of_the_live_heading(grid):
    """`takeAt` removes a layout item and leaves the widget a visible child at
    its old geometry. Every resize is a relayout, so a heading that is not
    destroyed is redrawn on top of itself once per drag frame."""
    from spacr.qt.widgets.figure_grid_view import LIVE_SECTION_LABEL

    grid.set_live_tiles(_panel_tiles())
    _run_of(grid, count=3)

    for width in (700, 820, 940):
        grid.resize(width, 640)

    headings = [h.text() for h in grid._headers]
    assert headings.count(LIVE_SECTION_LABEL) == 1
    assert headings.count("run one") == 1


# --------------------------------------------------------------------------- #
#  The regression tile keeps the wiring it already had
# --------------------------------------------------------------------------- #

def test_the_regression_tile_still_answers_to_the_signal_it_always_had(grid):
    """`pinned_activated` is what the regression screen is wired to. The
    general form must not quietly retire it, or "open the interactive volcano"
    stops working on the day the other panels arrive."""
    grid.set_live_tiles(_panel_tiles(("regression", "qq")))

    pinned, keyed = [], []
    grid.pinned_activated.connect(lambda: pinned.append(1))
    grid.live_tile_activated.connect(keyed.append)
    grid._pinned.clicked.emit(-1)

    assert pinned == [1]
    assert keyed == ["regression"]


def test_only_the_regression_tile_answers_to_the_pinned_signal(grid):
    """A Q-Q that fired `pinned_activated` would raise the volcano -- the
    exact "sentinel index opens the wrong graph" failure, one layer up."""
    grid.set_live_tiles(_panel_tiles(("regression", "qq")))

    pinned = []
    grid.pinned_activated.connect(lambda: pinned.append(1))
    [cell for cell in grid._live if cell.live_key == "qq"][0].clicked.emit(-1)

    assert pinned == []


def test_the_pinned_tile_is_found_by_name_not_by_being_first(grid):
    """It used to be "the first live tile" because it was the only one. With
    a tile per panel that becomes "whichever the caller listed first"."""
    grid.set_live_tiles([("qq", _block(), "Q-Q"),
                         ("regression", _block(), "Volcano")])

    assert grid._pinned is not None
    assert grid._pinned.live_key == "regression"
    assert grid._live[0].live_key == "qq"


def test_re_pinning_the_volcano_leaves_the_other_panels_alone(grid):
    """The screen re-photographs the volcano on a debounce and after every
    restyle. A call that cleared the other seven each time would make the
    whole section flicker."""
    grid.set_live_tiles(_panel_tiles(("regression", "qq", "controls")))

    assert grid.set_pinned(_block("#3355ff"), "regression — interactive")

    assert grid.live_tile_keys() == ["regression", "qq", "controls"]
    assert len(_tiles(grid)) == 3


def test_unpinning_the_volcano_leaves_the_other_panels_alone(grid):
    """A run that has fitted nothing loses its volcano tile and keeps the
    panels that still have something to show."""
    grid.set_live_tiles(_panel_tiles(("regression", "qq")))

    assert grid.set_pinned(None) is False

    assert grid._pinned is None
    assert grid.live_tile_keys() == ["qq"]
    assert len(_tiles(grid)) == 1


def test_a_right_click_on_a_live_tile_says_which_panel_it_was(grid):
    """Each pyqtgraph panel builds its own restyle menu, so the caller needs
    the key to ask the right one. "all gigures should be editable by right
    clicking" -- these most of all, they are the real graphs."""
    grid.set_live_tiles(_panel_tiles(("regression", "qq")))

    seen = []
    grid.live_tile_menu_requested.connect(
        lambda key, position: seen.append((key, position)))
    [c for c in grid._live if c.live_key == "qq"][0].menu_requested.emit(
        -1, "somewhere")

    assert seen == [("qq", "somewhere")]


# --------------------------------------------------------------------------- #
#  Through the real pyqtgraph panels
# --------------------------------------------------------------------------- #

def test_the_real_panels_become_tiles_and_no_scene_is_left_on_the_grid(qtbot):
    """The measured decision, asserted as behaviour rather than as a timing.

    A timing assertion in CI is a flake; the invariant the measurement bought
    is structural and is not. The grid holds PICTURES: no pyqtgraph widget is
    ever parented into its body, so there is no second scene, no second view
    box and no second set of hover hit-boxes per tile -- which is what made
    eighteen live tiles cost 74.99 ms of every window-drag frame.
    """
    pytest.importorskip("pyqtgraph")
    import numpy as np
    import pandas as pd

    from spacr.qt.widgets.fast_plots import QQPlot, VolcanoPlot
    from spacr.qt.widgets.figure_grid_view import (FigureGridView,
                                                   live_tiles_from_panels)

    rng = np.random.default_rng(0)
    n = 1213
    frame = pd.DataFrame({
        "feature": [f"gene_fraction:gene[g{i}]" for i in range(n)],
        "coefficient": rng.normal(0, 0.4, n),
        "p_value": rng.random(n) ** 3,
    })
    volcano = VolcanoPlot()
    qtbot.addWidget(volcano)
    volcano.set_results(frame, effect="coefficient", p_column="p_value")
    qq = QQPlot()
    qtbot.addWidget(qq)
    qq.set_p_values(frame["p_value"].to_numpy())

    grid = FigureGridView()
    qtbot.addWidget(grid)
    grid.resize(900, 640)
    shown = grid.set_live_tiles(live_tiles_from_panels([
        ("regression", "regression — interactive", volcano),
        ("qq", "Q-Q", qq)]))

    assert shown == 2
    assert grid.live_tile_keys() == ["regression", "qq"]

    from PySide6.QtWidgets import QGraphicsView

    descendants = grid._body.findChildren(QGraphicsView)
    assert descendants == [], (
        f"{len(descendants)} live scenes were parented into the grid; the "
        f"tiles are supposed to be pictures")


def test_the_real_screens_whole_panel_set_lands_on_its_own_grid(qtbot):
    """The maintainer's path, with nothing stubbed.

    A real coefficient table into the real `RegressionResultsPanel`, its own
    live pyqtgraph panels photographed through `live_tiles_from_panels`, and
    the result handed to the real screen's grid alongside the run's figures.

    THIS IS THE TEST THE WIRING LEANS ON. `_pin_regression_graph` still pins
    only the volcano, because `spacr/qt/screens/app_screen.py` belongs to
    another session; the two lines that replace it are exactly the two here.
    Until they land this proves the grid half is ready and correct, and after
    they land it is the end-to-end check that the panel set reaches the grid.
    """
    pytest.importorskip("pyqtgraph")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import numpy as np
    import pandas as pd

    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widgets.figure_grid_view import live_tiles_from_panels

    rng = np.random.default_rng(11)
    n = 300
    frame = pd.DataFrame({
        "feature": ["Intercept"] + [f"fraction:grna[{i}_1]" for i in range(n)],
        "coefficient": np.concatenate([[0.19], rng.normal(size=n)]),
        "p_value": np.concatenate([[3e-46], rng.uniform(size=n)]),
    })

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    panel = screen._results_panel
    panel.set_frame(frame)
    screen._refresh_figure_grid()

    grid = screen._figure_grid
    named = [("regression", "regression — interactive", panel.volcano),
             ("p_values", "p-values", panel.p_values),
             ("qq", "Q-Q", panel.qq),
             ("controls", "Controls", panel.controls),
             ("agreement", "Guide support", panel.agreement)]
    tiles = live_tiles_from_panels(named)
    shown = grid.set_live_tiles(tiles)

    # Whatever this table can support, the volcano is on it -- it is the one
    # panel a coefficient table always fills -- and every tile that IS there
    # is one of the panels asked for, never a leftover or a duplicate.
    assert shown >= 1
    assert "regression" in grid.live_tile_keys()
    assert set(grid.live_tile_keys()) <= {key for key, _t, _w in named}
    assert len(grid.live_tile_keys()) == len(set(grid.live_tile_keys()))
    assert len(_tiles(grid)) == shown

    # Refreshing the way the screen does -- on a debounce, after every
    # restyle -- must not stack a second copy of any of them.
    for _ in range(5):
        grid.set_live_tiles(live_tiles_from_panels(named))
    assert len(_tiles(grid)) == shown

    # And the pinned wiring the screen already has still reaches the volcano.
    opened = []
    grid.pinned_activated.connect(lambda: opened.append(1))
    grid._pinned.clicked.emit(-1)
    assert opened == [1]


def test_a_panel_with_nothing_fitted_photographs_as_nothing(qtbot):
    """`snapshot` returns None for an empty plot on purpose, and the grid has
    to honour that rather than drawing a tile of blank axes. This is the
    residual plot before any run, which is the ordinary state of the screen
    when it opens."""
    pytest.importorskip("pyqtgraph")

    from spacr.qt.widgets.fast_plots import ResidualPlot
    from spacr.qt.widgets.figure_grid_view import (FigureGridView,
                                                   live_tiles_from_panels)

    empty = ResidualPlot()
    qtbot.addWidget(empty)
    assert empty.snapshot() is None, "an empty plot photographed as something"

    grid = FigureGridView()
    qtbot.addWidget(grid)
    assert grid.set_live_tiles(
        live_tiles_from_panels([("residuals", "Residuals", empty)])) == 0
    assert _tiles(grid) == []
