"""A results table with no guide terms opens showing its genes.

The regression the default level introduced. `_level` was set to "grna"
unconditionally on every new table, which is right for a mixed or
hierarchical fit -- that table carries both levels and drawing a gene once per
guide is the four-fold duplication the default exists to prevent.

It is WRONG for the table instruction 128 R produces. Splitting the fit writes
`results_gene.csv`, whose every row is a gene term, and a guide filter over it
selects nothing: an empty volcano beside a full coefficient table, which reads
as a broken plot rather than as an empty filter.
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


def _frame(kind, n=40, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "gene":
        features = [f"gene_fraction:gene[{200000 + i}]" for i in range(n)]
    elif kind == "grna":
        features = [f"fraction:grna[{200000 + i}_1]" for i in range(n)]
    elif kind == "both":
        features = ([f"gene_fraction:gene[{200000 + i}]" for i in range(n // 2)]
                    + [f"fraction:grna[{200000 + i}_1]"
                       for i in range(n // 2)])
    else:
        features = ["Intercept"] + [f"rowID[T.r{i:02d}]" for i in range(n - 1)]
    return pd.DataFrame({
        "feature": features,
        "coefficient": rng.normal(0, 0.5, n),
        "p_value": rng.uniform(size=n),
    })


def _panel(qtbot, frame):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    panel.set_frame(frame)
    return panel


def test_a_gene_only_table_draws_its_genes(qtbot):
    panel = _panel(qtbot, _frame("gene"))
    assert panel.level() == "gene"
    assert len(panel.volcano._row_xy) == 40


def test_a_guide_table_still_opens_on_guides(qtbot):
    """The four-fold-duplication fix is untouched."""
    panel = _panel(qtbot, _frame("grna"))
    assert panel.level() == "grna"
    assert len(panel.volcano._row_xy) == 40


def test_a_table_with_both_opens_on_guides(qtbot):
    """Both levels present is the case the "grna" default was chosen for.

    A gene and its four guides all name the same gene, so drawing both puts
    five dots where the screen measured one thing -- and on the ADJUSTED axis
    Benjamini-Hochberg ties pull them to one height and they stack.
    """
    panel = _panel(qtbot, _frame("both"))
    assert panel.level() == "grna"
    assert len(panel.volcano._row_xy) == 20


def test_a_table_of_nothing_but_nuisance_terms_names_no_level(qtbot):
    """No filter is applied, because there is no level to filter to.

    `Intercept` and the row/column terms are covariates, not hypotheses --
    `spacr.hits.tested_family` is the one place that line is drawn and the
    volcano already refuses to plot them. So this table draws nothing either
    way; what matters is that the panel does not CLAIM a level, which is what
    `level_counts()` counting them as genes would have made it do.
    """
    panel = _panel(qtbot, _frame("nuisance"))
    assert panel.level() is None
    assert len(panel.volcano._row_xy) == 0


def test_reloading_a_gene_table_over_a_guide_one_moves_the_level(qtbot):
    """The default is recomputed per table, not remembered from the last."""
    panel = _panel(qtbot, _frame("grna"))
    assert panel.level() == "grna"
    panel.set_frame(_frame("gene"))
    assert panel.level() == "gene"
    assert len(panel.volcano._row_xy) == 40


def test_the_user_can_still_choose_an_empty_level(qtbot):
    """Chosen is different from defaulted.

    A user who asks for guides on a gene-only table gets an empty plot, and
    should: the menu says "guides only (0)" beside it, so the emptiness is
    the answer to what they asked rather than a plot that failed.
    """
    panel = _panel(qtbot, _frame("gene"))
    panel.set_level("grna")
    assert len(panel.volcano._row_xy) == 0
    labels = [action.text() for action in panel.build_level_menu().actions()
              if action.isCheckable()]
    assert any("(0)" in label for label in labels)
