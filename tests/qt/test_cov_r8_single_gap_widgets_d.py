"""Six more single-decision widgets, each an "and what if there is none".

An empty panel behind a shared axis, a plot exported before its spec
arrives, a result with nothing to warn about, a filter outliving its
application, a window with no screen under it, and a menu installed
twice.
"""
from __future__ import annotations


import pytest

pytest.importorskip("PySide6")

import numpy as np
import pandas as pd

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
# outlier_model.py -- a report with nothing to warn about
# ---------------------------------------------------------------------------

def _scan(frame, **kwargs):
    from spacr.qt.widgets.outlier_model import OutlierSpec, detect_outliers

    return detect_outliers(frame, OutlierSpec(**kwargs))


class TestAReportWithNoCaveats:

    def test_a_report_carries_its_caveats_when_it_has_them(self):
        from spacr.qt.widgets.outlier_model import METHOD_MAHALANOBIS

        rng = np.random.default_rng(0)
        frame = pd.DataFrame({
            "plateID": "p1", "rowID": "r1",
            "columnID": [f"c{i % 4 + 1}" for i in range(120)],
            "fieldID": "f1", "object_label": range(120),
            "cell_area": rng.lognormal(0, 0.2, 120),
            "cell_perimeter": rng.lognormal(0, 0.2, 120)})

        result = _scan(frame, features=("cell_area", "cell_perimeter"),
                       method=METHOD_MAHALANOBIS)
        assert result.caveats()
        assert "  ! " in result.report()

    def test_no_scan_can_produce_a_report_with_no_caveats(self):
        """THE PIN.

        `if caveats:` guards the blank line that introduces the warning
        block -- emitting it with nothing after it ends the report in
        whitespace that reads, in a copied report, as a section cut off.
        It can never fire, because `caveats()` ends with an
        unconditional line: flagged is not deleted.

        That sentence is the one caveat that is true of every scan, and
        it is the last thing a reader sees before acting on the flags.
        If it ever becomes conditional, this fails and the guard needs a
        test of its own.
        """
        rng = np.random.default_rng(1)
        frame = pd.DataFrame({
            "plateID": "p1", "rowID": "r1",
            "columnID": [f"c{i % 4 + 1}" for i in range(80)],
            "fieldID": "f1", "object_label": range(80),
            "cell_area": rng.lognormal(0, 0.2, 80)})

        for kwargs in ({}, {"min_well_objects": 1}, {"min_well_objects": 500}):
            result = _scan(frame, features=("cell_area",), **kwargs)
            caveats = result.caveats()
            assert caveats, f"a scan produced no caveats at all: {kwargs}"
            assert "Flagged is not deleted" in caveats[-1], (
                "the unconditional last caveat is gone, so caveats() can "
                "now be empty and the report's guard is live")
            assert "  ! " in result.report()


# ---------------------------------------------------------------------------
# feature_dictionary.py -- the filter outliving its application
# ---------------------------------------------------------------------------

class TestRemovingTheContextMenuFilter:

    def test_removing_a_filter_that_was_never_installed_says_so(self):
        from spacr.qt.widgets import feature_dictionary

        feature_dictionary._FILTER = None
        assert feature_dictionary.remove_context_menu_filter() is False

    def test_a_filter_with_no_application_left_is_still_dropped(self,
                                                                monkeypatch):
        """THE UNCOVERED ARC.

        At shutdown the QApplication goes first. `removeEventFilter` on
        None is an AttributeError raised from a teardown path, which is
        the worst place for one -- so the filter is dropped without
        removing it, which is all that is left to do.
        """
        from PySide6.QtWidgets import QApplication

        from spacr.qt.widgets import feature_dictionary

        sentinel = object()
        feature_dictionary._FILTER = sentinel
        monkeypatch.setattr(QApplication, "instance", staticmethod(
            lambda: None))

        assert feature_dictionary.remove_context_menu_filter() is True
        assert feature_dictionary._FILTER is None


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
