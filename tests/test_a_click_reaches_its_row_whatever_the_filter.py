"""A point you can click has a row you can reach.

Instruction 128, found 2026-08-18. The gene/guide filter reaches every tab
(128 L), and the guide-agreement plot is ONE ROW PER GENE by construction --
it is the plot that answers "do this gene's guides agree". So while the filter
sat on `grna`, the coefficient table held no row for anything drawn on it and
a click landed nowhere: no ring, no selected row, no gene tile, and nothing
anywhere saying why, on a plot that still looked clickable.

THE CLICK IS MORE SPECIFIC THAN THE STANDING FILTER, so it wins.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

pytestmark = pytest.mark.qt


@pytest.fixture
def results():
    """Guides and gene-level terms, as a real fit writes both."""
    rng = np.random.default_rng(3)
    rows = []
    for index in range(30):
        gene = f"{200000 + index * 10}"
        for guide in range(3):
            rows.append({"feature": f"fraction:grna[{gene}_{guide}]",
                         "coefficient": float(rng.normal()),
                         "p_value": float(rng.uniform(1e-6, 1)),
                         "grna": f"{gene}_{guide}", "gene": gene})
        rows.append({"feature": f"gene_fraction:gene[{gene}]",
                     "coefficient": float(rng.normal()),
                     "p_value": float(rng.uniform(1e-6, 1)),
                     "grna": None, "gene": gene})
    return pd.DataFrame(rows)


@pytest.fixture
def panel(qtbot, results):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    widget = RegressionResultsPanel()
    qtbot.addWidget(widget)
    widget.set_frame(results, source="results.csv")
    return widget


def _selected(panel):
    items = panel.table.table.selectedItems()
    if not items:
        return None
    column = list(panel._frame.columns).index("feature")
    return panel.table.table.item(items[0].row(), column).text()


def test_the_table_opens_on_guides(panel):
    """The premise. A gene drawn once per guide is the duplication this
    default exists to prevent."""
    assert panel.level() == "grna"


def test_a_gene_level_key_is_not_reachable_at_guide_level(panel):
    """The condition the fix keys off, asserted rather than assumed."""
    assert panel._reachable("fraction:grna[200000_0]") is True
    assert panel._reachable("gene_fraction:gene[200000]") is False


def test_clicking_a_gene_reaches_its_row(panel, qtbot):
    """The bug. It used to select nothing at all."""
    panel._select_from_a_plot("gene_fraction:gene[200100]")
    qtbot.waitUntil(lambda: _selected(panel) is not None, timeout=2000)

    assert _selected(panel) == "gene_fraction:gene[200100]"
    assert panel.level() == "gene", (
        "the filter had to move for the row to exist")


def test_and_it_says_the_filter_moved(panel, qtbot):
    """A view that changes under the user without a word is the other half of
    the same failure."""
    panel._select_from_a_plot("gene_fraction:gene[200100]")
    qtbot.waitUntil(lambda: _selected(panel) is not None, timeout=2000)

    assert "so the point you clicked has a row" in panel._status


def test_an_ordinary_click_does_not_move_the_filter(panel, qtbot):
    """The filter must not twitch under a user browsing within it."""
    panel._select_from_a_plot("fraction:grna[200100_1]")
    qtbot.waitUntil(lambda: _selected(panel) is not None, timeout=2000)

    assert panel.level() == "grna"
    assert _selected(panel) == "fraction:grna[200100_1]"


def test_a_guide_clicked_from_a_gene_view_comes_back(panel, qtbot):
    """The move works in both directions, not just one."""
    panel.set_level("gene")
    panel._select_from_a_plot("fraction:grna[200100_1]")
    qtbot.waitUntil(lambda: _selected(panel) is not None, timeout=2000)

    assert panel.level() == "grna"
    assert _selected(panel) == "fraction:grna[200100_1]"


def test_the_whole_fit_hides_nothing_so_nothing_moves(panel, qtbot):
    panel.set_level(None)
    assert panel._reachable("gene_fraction:gene[200100]") is True
    panel._select_from_a_plot("gene_fraction:gene[200100]")
    qtbot.waitUntil(lambda: _selected(panel) is not None, timeout=2000)
    assert panel.level() is None


def test_a_key_that_is_in_no_table_does_not_move_the_filter(panel, qtbot):
    """An unknown key is somebody else's problem to report, not a reason to
    rearrange the panel around it."""
    before = panel.level()
    panel._select_from_a_plot("gene_fraction:gene[999999]")
    assert panel.level() == before


def test_reachable_says_yes_when_there_is_nothing_to_hide(qtbot):
    """No frame, no filter and no feature column are three different ways of
    hiding nothing, and each must be a yes rather than a crash."""
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    widget = RegressionResultsPanel()
    qtbot.addWidget(widget)
    assert widget._reachable("anything") is True
    assert widget._reachable("") is True

    widget.set_frame(pd.DataFrame({"feature": ["fraction:grna[1_1]"],
                                   "coefficient": [1.0], "p_value": [0.1]}))
    widget.set_level(None)
    assert widget._reachable("fraction:grna[1_1]") is True
