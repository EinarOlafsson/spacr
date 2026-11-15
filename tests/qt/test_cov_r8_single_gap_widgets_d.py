"""Six more single-decision widgets, each an "and what if there is none".

An empty panel behind a shared axis, a plot exported before its spec
arrives, a result with nothing to warn about, a filter outliving its
application, a window with no screen under it, and a menu installed
twice.
"""
from __future__ import annotations


import pytest

pytest.importorskip("PySide6")


pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# graph_spec.py -- a panel with rows in the grid and none after its filter
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# grouped_plot.py -- exported before a spec arrives
# ---------------------------------------------------------------------------

class TestExportingAPlotWithNoSpec:


    def test_a_plot_with_no_spec_exports_only_what_it_inherits(self, qtbot):
        """THE UNCOVERED ARC.

        The panel exists before any data reaches it -- it is built with
        the screen and filled when a table loads. Reading spec.kind then
        is an AttributeError on None, and it would happen while writing
        the settings file that records what the user was looking at.
        """
        from spacr.qt.widgets.grouped_plot import GroupedPlot

        plot = GroupedPlot()
        qtbot.addWidget(plot)
        assert plot.spec is None

        exported = plot.export_settings()
        assert "group" not in exported and "kind" not in exported
        assert plot.comparison_unit() == "observation"


# ---------------------------------------------------------------------------
# dna_rain_settings.py -- placed with no screen under the anchor
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# walkthrough.py -- the handler is made once per window
# ---------------------------------------------------------------------------

class TestTheWalkthroughHandlerIsMadeOnce:

    def test_installing_the_menu_makes_a_handler(self, qtbot):
        from PySide6.QtWidgets import QMainWindow

        from spacr.qt import walkthrough

        window = QMainWindow()
        qtbot.addWidget(window)
        window.menuBar().addMenu("Help")

        submenu = walkthrough.install_help_menu(window)
        assert submenu is not None
        assert getattr(window, "_walkthrough_handler", None) is not None

    def test_a_window_that_already_has_one_keeps_it(self, qtbot):
        """THE UNCOVERED ARC.

        The handler owns the window's walkthrough state. A second one
        would answer the same menu with a different memory of which
        walkthrough had been shown, so the existing one is reused.
        """
        from PySide6.QtWidgets import QMainWindow

        from spacr.qt import walkthrough

        window = QMainWindow()
        qtbot.addWidget(window)
        window.menuBar().addMenu("Help")

        first = walkthrough._WalkthroughHandler(window)
        window._walkthrough_handler = first

        walkthrough.install_help_menu(window)
        assert window._walkthrough_handler is first, (
            "a second handler replaced the one holding the window's state")
