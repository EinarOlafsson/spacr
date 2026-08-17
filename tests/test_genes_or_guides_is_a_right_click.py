"""Show only genes, or only guides, by right-clicking the volcano.

Asked for on 2026-08-17: "instead of having a gene and a grna mode, id like
the option to only show genes and only show grnas in the volcano plot by
right clicking on the plot itself".

IT NEEDS NO RE-FIT. The coefficient table already carries both -- `feature`
is `gene_fraction:gene[...]` or `fraction:grna[...]` -- so this is a mask,
not a second run, which is exactly why it belongs on the plot rather than in
the settings where it used to live.
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

GENES, GUIDES_PER_GENE = 300, 3


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(GENES):
        rows.append({"feature": f"gene_fraction:gene[{400000 + g}]",
                     "coefficient": rng.normal(0, .5),
                     "p_value": rng.uniform()})
        for k in range(GUIDES_PER_GENE):
            rows.append({"feature": f"fraction:grna[{400000 + g}_{k}]",
                         "coefficient": rng.normal(0, .5),
                         "p_value": rng.uniform()})
    return pd.DataFrame(rows)


def _panel(qtbot, frame=None):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    panel.set_frame(_frame() if frame is None else frame)
    return panel


def _menu(panel):
    return [a.text() for a in panel.volcano.build_style_menu().actions()]


# --------------------------------------------------------------------------- #
#  The gesture
# --------------------------------------------------------------------------- #

def test_the_volcano_offers_all_three(qtbot):
    text = " ".join(_menu(_panel(qtbot)))

    assert "genes only" in text
    assert "guides only" in text
    assert "genes and guides" in text


def test_the_counts_are_in_the_menu(qtbot):
    """"genes only" that silently draws 300 of 1,200 points is a filter a
    user applies without knowing what they gave up."""
    text = " ".join(_menu(_panel(qtbot)))

    assert f"genes only ({GENES})" in text, text
    assert f"guides only ({GENES * GUIDES_PER_GENE})" in text, text


def test_the_filter_is_separated_from_the_restyling(qtbot):
    """It changes WHICH ROWS are on the plot. A filtered plot that looks like
    a restyled one is read as the whole screen."""
    items = ["|" if a.isSeparator() else a.text()
             for a in _panel(qtbot).volcano.build_style_menu().actions()]

    level = next(i for i, t in enumerate(items) if "genes only" in t)
    size = next(i for i, t in enumerate(items) if "Point size" in t)
    assert "|" in items[size:level], items


# --------------------------------------------------------------------------- #
#  What it draws
# --------------------------------------------------------------------------- #

def test_guides_are_the_default(qtbot):
    """CHANGED 2026-08-17, and it is the fix for a reported bug.

    "both" drew a gene once per guide PLUS once for itself -- on the real
    screen `225160` was four points (three guides and the gene row), reported
    as "occur in the top right side of the graph 4 times each which is
    obviously wrong". Mixing two levels on one plot is what 128 R fixes
    properly, by fitting them separately; this is the display half.

    Guides rather than genes: it is the unit the screen measures, and a
    permutation run reports guides ONLY, so it is the level on which the two
    inference modes agree.
    """
    panel = _panel(qtbot)

    assert panel._level == "grna"
    assert len(panel.volcano._row_xy) == GENES * GUIDES_PER_GENE


def test_genes_only_draws_the_genes(qtbot):
    panel = _panel(qtbot)
    panel.set_level("gene")

    assert len(panel.volcano._row_xy) == GENES


def test_guides_only_draws_the_guides(qtbot):
    panel = _panel(qtbot)
    panel.set_level("grna")

    assert len(panel.volcano._row_xy) == GENES * GUIDES_PER_GENE


def test_the_two_partition_the_table(qtbot):
    """Every row is one or the other, so a coefficient cannot vanish from
    both views -- which is how a hit disappears without anyone noticing."""
    panel = _panel(qtbot)
    counts = panel.level_counts()

    assert counts["gene"] + counts["grna"] == counts[None]


def test_going_back_restores_everything(qtbot):
    panel = _panel(qtbot)
    panel.set_level("gene")
    panel.set_level(None)

    assert len(panel.volcano._row_xy) == GENES * (GUIDES_PER_GENE + 1)


# --------------------------------------------------------------------------- #
#  What it must not break
# --------------------------------------------------------------------------- #

def test_the_selection_survives_the_filter(qtbot):
    """The same rule the colouring and the baseline follow: the ring the user
    was reading must not vanish and leave them to find their guide again."""
    panel = _panel(qtbot)
    key = panel._frame["feature"].iloc[0]        # a gene row
    panel._select_key(key)
    panel.set_level("gene")

    assert panel.volcano._selected_key == key


def test_it_says_what_it_is_showing(qtbot):
    panel = _panel(qtbot)
    panel.set_level("gene")

    said = panel.status_text()
    assert str(GENES) in said and "genes only" in said


def test_a_new_table_is_not_still_filtered(qtbot):
    """A new run is a new experiment. Inheriting the filter would show a
    subset of it with nothing saying so."""
    panel = _panel(qtbot)
    panel.set_level("gene")
    panel.set_frame(_frame(seed=2))

    assert panel._level == "grna"


def test_the_run_s_own_table_is_not_filtered(qtbot):
    """The RUN's table keeps every row: the filter is a view, not an edit.

    CONTRACT CORRECTED 2026-08-17, instruction 128 L. This test used to read
    "the coefficient table beside the plot still shows every row", which was
    true when the filter reached the volcano and nothing else. The maintainer
    asked for the opposite -- "i should be able to right click on the
    coeffisients table and only see grna or genes and this should also filer
    the subsequent data/graphs in the subsequent tabs" -- so the TABLE now
    narrows with the plot; see
    tests/test_the_gene_guide_filter_reaches_every_tab.py.

    What survives, and is what this test was really protecting, is that the
    run's own frame is never edited: a caller exporting the results gets the
    fit rather than whatever the user last right-clicked.
    """
    panel = _panel(qtbot)
    before = len(panel._frame)
    panel.set_level("gene")

    assert len(panel._frame) == before
    assert len(panel.results_frame()) == before
    assert len(panel.filtered_frame()) == GENES
