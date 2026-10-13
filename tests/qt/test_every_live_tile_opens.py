"""199: clicking a tile in the figure panel opens that figure.

    "in the figure pannel. i am able to click on the colcano plot to open it
     but not the other pyqtgraphs."

THE GRID ALREADY KNEW WHICH TILE WAS PRESSED. `FigureGridView` photographs
nine live pyqtgraph panels, gives every tile a key, and emits
`live_tile_activated(key)` when one is pressed -- but the regression screen
was connected only to `pinned_activated`, which the grid emits for the
volcano alone. So eight tiles were drawn, took a click, and reached a signal
with no receiver.

A TILE THAT DOES NOT OPEN IS WORSE THAN NO TILE: it is drawn, it is in the
grid, it takes the click, and nothing happens -- so the user tries twice and
concludes the application is broken rather than that this picture has no
door.

These drive the MOUSE, not the handler. A test that calls
`_open_live_tile("qq")` would have passed against the broken build, because
what was broken was the connection, not the handler.
"""
from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

pytestmark = pytest.mark.qt


@pytest.fixture
def screen(qtbot):
    """The regression screen WITH A FITTED RUN ON IT.

    The frame is not decoration. `_pin_regression_graph` refuses to pin a
    tile when nothing has been fitted -- "an empty plot tile invites a click
    that opens an empty plot" -- so a screen without one has no tiles, and
    every test here would skip while claiming to cover the click.
    """
    import numpy as np
    import pandas as pd

    from spacr.qt.screens.app_screen import AppScreen

    rng = np.random.default_rng(11)
    n = 300
    frame = pd.DataFrame({
        "feature": ["Intercept"] + [f"fraction:grna[{i}_1]" for i in range(n)],
        "coefficient": np.concatenate([[0.19], rng.normal(size=n)]),
        "p_value": np.concatenate([[3e-46], rng.uniform(size=n)]),
    })

    widget = AppScreen("regression")
    qtbot.addWidget(widget)
    widget._results_panel.set_frame(frame)
    return widget


def _tiles(screen):
    """The live tiles on the grid, by key."""
    grid = screen._figure_grid
    return {cell.live_key: cell for cell in grid._live if cell.live_key}


def _pin_the_panels(screen):
    """Put the run's live tiles on the grid, as a finished run does."""
    screen._pin_regression_graph()
    return _tiles(screen)


def _click(qtbot, cell):
    """Press the tile the way a user does."""
    from PySide6.QtCore import Qt

    cell.resize(120, 90)
    qtbot.mouseClick(cell, Qt.LeftButton)


class TestTheGridEmitsForEveryTile:
    """The half that already worked, asserted so it keeps working."""

    def test_a_live_tile_says_which_one_it_is(self, qtbot):
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QPixmap

        from spacr.qt.widgets.figure_grid_view import FigureGridView

        grid = FigureGridView()
        qtbot.addWidget(grid)
        pixmap = QPixmap(40, 30)
        pixmap.fill(Qt.red)
        grid.set_live_tiles([("qq", pixmap, "Q-Q"),
                             ("residuals", pixmap, "Residuals")])

        seen = []
        grid.live_tile_activated.connect(seen.append)
        for cell in grid._live:
            _click(qtbot, cell)

        assert seen == ["qq", "residuals"]


class TestThePanelHasADoorForEveryKey:

    def test_every_grid_key_raises_its_tab(self, qtbot):
        from spacr.qt.widgets.regression_results import RegressionResultsPanel

        panel = RegressionResultsPanel()
        qtbot.addWidget(panel)

        for key in RegressionResultsPanel._PANEL_TABS:
            assert panel.show_panel(key), key
            index = panel.tabs.currentIndex()
            assert index >= 0

    def test_an_unknown_key_is_refused_rather_than_guessed(self, qtbot):
        from spacr.qt.widgets.regression_results import RegressionResultsPanel

        panel = RegressionResultsPanel()
        qtbot.addWidget(panel)
        before = panel.tabs.currentIndex()

        assert panel.show_panel("nonsense") is False
        assert panel.show_panel("") is False
        # And it did not wander to some other tab on the way to saying no.
        assert panel.tabs.currentIndex() == before

    def test_a_key_whose_tab_is_absent_says_false(self, qtbot):
        """FALSE IS A REAL ANSWER: the tab set is not fixed. With the volcano
        shown outside the panel, the volcano and gene tabs are never added,
        and a caller that assumed success would leave the user looking at
        whatever tab happened to be up."""
        from spacr.qt.widgets.regression_results import RegressionResultsPanel

        panel = RegressionResultsPanel(external_volcano=True)
        qtbot.addWidget(panel)

        assert panel.show_panel("gene") is False
        # A panel that IS there still opens, so this is not a blanket no.
        assert panel.show_panel("qq") is True


class TestTheClickOpensTheGraph:
    """The reported symptom, driven through the mouse on the real screen."""

    def test_clicking_a_pyqtgraph_tile_raises_its_tab(self, qtbot, screen):
        tiles = _pin_the_panels(screen)
        if "qq" not in tiles:
            pytest.skip("this build pinned no Q-Q tile")

        panel = screen._results_panel
        panel.show_panel("controls")                   # somewhere else first
        _click(qtbot, tiles["qq"])

        # THE WIDGET, not the label. A tab's text is a display string that
        # gains a suffix once a run is loaded -- "Q-Q" becomes "Q-Q
        # (guides)" -- and a test that matches on it breaks on a rename
        # while missing the thing it was written to catch.
        assert panel.tabs.currentWidget() is panel.qq

    def test_it_is_not_only_the_volcano(self, qtbot, screen):
        """The exact complaint: the volcano opens and its neighbours do not."""
        tiles = _pin_the_panels(screen)
        panel = screen._results_panel
        tabs = panel.tabs

        opened = {}
        for key, cell in tiles.items():
            if key == "regression":
                continue
            panel.show_panel("controls")
            _click(qtbot, cell)
            opened[key] = tabs.currentWidget()

        if not opened:
            pytest.skip("this build pinned no live tiles besides the volcano")
        # Every one of them moved the tabs somewhere, and no two keys share a
        # destination -- which is what would happen if they all fell through
        # to one default.
        assert len(set(map(id, opened.values()))) == len(opened), list(opened)

    def test_the_volcano_still_opens(self, qtbot, screen):
        """It worked before; the new route must not have taken it away.

        It has a page of its own rather than a tab, because the gene tile
        goes beside it."""
        tiles = _pin_the_panels(screen)
        if "regression" not in tiles:
            pytest.skip("this build pinned no volcano tile")

        screen._figures_stack.setCurrentIndex(0)
        _click(qtbot, tiles["regression"])

        assert (screen._figures_stack.currentWidget()
                is screen._volcano_page)

    def test_the_volcano_is_raised_once_not_twice(self, qtbot, screen):
        """The grid emits BOTH signals for the volcano. Both handlers acting
        would raise it twice -- harmless here, and the kind of thing that
        stops being harmless when one of them starts doing more."""
        raised = []
        original = screen._show_regression_graph

        def counted():
            raised.append(1)
            original()

        screen._show_regression_graph = counted
        tiles = _pin_the_panels(screen)
        if "regression" not in tiles:
            pytest.skip("this build pinned no volcano tile")

        _click(qtbot, tiles["regression"])
        assert len(raised) == 1


class TestNoTileIsSilent:

    def test_a_key_with_no_tab_says_so_in_the_console(self, qtbot, screen):
        """The one case that still ends in nothing visible happening is the
        one case that has to be said out loud."""
        said = []
        screen._console.append_notice = lambda text, **kw: said.append(text)

        screen._open_live_tile("effect_distribution_that_is_not_there")

        assert said and "no tab" in said[0].lower()
