"""Opening Methods & Results does not build Regression's hidden results panel.

The first Methods & Results page froze the window for most of a second: the
fold asked for the results panel to read a run folder from, and asking built
the whole panel for a page whose sources the user types in. An unbuilt panel
holds no run, so the page is seeded only from a panel that has been built.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens import app_screen, regression
from spacr.qt.screens.app_screen import AppScreen

pytestmark = pytest.mark.qt


def test_the_methods_page_opens_without_building_the_results_panel(
        qtbot, qt_theme_applied):
    screen = AppScreen(app_key="regression")
    qtbot.addWidget(screen)
    assert regression.install_folds(screen) is not None
    assert screen._part_is_owed(app_screen._REGRESSION_RESULTS)

    page = regression.build_methods_export(None, screen)
    qtbot.addWidget(page)

    assert screen._part_is_owed(app_screen._REGRESSION_RESULTS)
    assert screen._results_panel_if_built() is None
