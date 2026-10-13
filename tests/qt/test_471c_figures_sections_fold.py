"""Item 471 slice C: the sections inside the figures panel fold and drag.

The figure queue's thumbnail strip collapses to the left by its handle and
drags to any width; Regression's results beside its figure pages collapse to
the left the same way; the volcano and its coefficient table, the guide
agreement plot and its table, the cell montage and its caption, and the gene
record and what is known about the gene each fold by their heading, a folded
one sits at the bottom, and a dragged size is remembered. The deferred
Regression results (items 284/380) stay unbuilt through all of it.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication                   # noqa: E402

from spacr.qt.widgets.collapsible_splitter import (          # noqa: E402
    EDGE, FoldSection, get_pane_extents)


def _pump(n: int = 10) -> None:
    for _ in range(n):
        QApplication.processEvents()


def _shown(qtbot, widget, width=1100, height=760):
    qtbot.addWidget(widget)
    widget.resize(width, height)
    widget.show()
    _pump()
    return widget


def _bottom_of(section, container) -> int:
    heading = section.heading
    return heading.mapTo(container, heading.rect().bottomLeft()).y()


def test_the_thumbnail_strip_collapses_left_and_its_width_is_kept(qtbot):
    from spacr.qt.widgets.figure_queue import FigureQueue

    queue = _shown(qtbot, FigureQueue())
    split = queue._body_split
    assert split.pane("Figure list").mode == EDGE
    assert split.widget(0) is queue._list
    split.moveSplitter(240, 1)
    _pump()
    assert get_pane_extents("figures::body").get("Figure list", 0) > 200
    split.toggle_pane("Figure list", by_user=False)
    _pump()
    assert split.sizes()[0] == 0
    assert split.handle(1).edge_pane() is split.pane("Figure list")
    split.toggle_pane("Figure list", by_user=False)
    _pump()
    assert split.sizes()[0] > 200


def _results(qtbot):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    return _shown(qtbot, RegressionResultsPanel())


@pytest.mark.parametrize("tab_attr,names", [
    ("_volcano_tab", ("Volcano plot", "Coefficient table")),
    ("_support_tab", ("Guide agreement", "Guide support table")),
])
def test_the_result_tabs_plots_and_tables_fold_to_the_bottom(
        qtbot, tab_attr, names):
    panel = _results(qtbot)
    split = getattr(panel, tab_attr)
    panel.tabs.setCurrentWidget(split)
    _pump()
    sections = [split.pane(n).widget for n in names]
    assert all(isinstance(s, FoldSection) for s in sections)
    for section in sections:
        section.set_folded(True, by_user=False)
    _pump()
    last = sections[-1]
    assert _bottom_of(last, split) >= split.height() - 12
    assert _bottom_of(sections[0], split) > split.height() // 2
    sections[0].set_folded(False, by_user=False)
    _pump()
    assert sections[0].body.isVisible()


def test_the_montage_and_the_gene_panel_fold(qtbot):
    from spacr.qt.widgets.cell_montage_view import _WellTab
    from spacr.qt.widgets.gene_panel import GenePanel

    montage = _shown(qtbot, _WellTab(("A01", "g1"), "A01 g1"))
    caption = montage._split.pane("Caption").widget
    caption.set_folded(True, by_user=False)
    _pump()
    assert _bottom_of(caption, montage._split) >= montage._split.height() - 12

    gene = _shown(qtbot, GenePanel(frame_provider=lambda: None))
    known = gene.split.pane("Known about this gene").widget
    record = gene.split.pane("Gene record").widget
    record.set_folded(True, by_user=False)
    _pump()
    assert known.body.isVisible()
    assert gene.split.sizes()[1] > gene.split.sizes()[0]


def test_regression_results_stay_unbuilt_and_then_fold(qtbot):
    from spacr.qt.screens.app_screen import _REGRESSION_RESULTS, AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    screen.resize(1366, 768)
    screen.show()
    _pump()
    assert screen._part_is_owed(_REGRESSION_RESULTS), (
        "opening Regression built its deferred results")

    screen._build_owed_part(_REGRESSION_RESULTS)
    split = screen._if_built("_figures_split")
    assert split.pane("Results").mode == EDGE
    assert split.widget(0) is screen._if_built("_results_tabs")
    gene = screen._if_built("_gene_split")
    assert gene.is_collapsed("Gene") and gene.sizes()[1] == 0
