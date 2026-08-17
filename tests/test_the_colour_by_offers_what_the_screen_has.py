"""What a screen can colour its volcano by, and why an option is missing.

Asked for on 2026-08-17: "the color by dosnt include condition" and "lopit
should be in color by".

MEASURED BEFORE DIAGNOSING. On a synthetic table with nc/pc/other, condition
WAS already offered -- so the filter was not blanket-excluding it. The two
ways a real screen falls out of the rule (object dtype AND
1 < distinct <= max(40, len//20)) are a column that is absent, and a column
with ONE distinct value. The second is the interesting one: every row reads
'other' when the negative/positive control NAMES matched no feature, and that
is a FINDING about the run, not a boring column. Hiding it hides the finding.
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


def _lopit_genes(n=400):
    from spacr import localisation

    genes = list(localisation.table())[:n]
    if not genes:
        pytest.skip("the bundled LOPIT table is not present")
    return genes


def _frame(condition=None, lopit=True, n=400, seed=0):
    rng = np.random.default_rng(seed)
    genes = _lopit_genes(n) if lopit else [str(900000 + i) for i in range(n)]
    n = len(genes)
    if condition is None:
        condition = list(rng.choice(["nc", "pc", "other"], n, p=[.1, .03, .87]))
    return pd.DataFrame({
        "feature": [f"gene_fraction:gene[{g}]" for g in genes],
        "coefficient": rng.normal(0, .5, n),
        "p_value": rng.uniform(size=n),
        "grna": [f"{g}_1" for g in genes],
        "condition": condition,
    })


def _panel(qtbot, frame):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    panel.set_frame(frame)
    return panel


def _options(panel):
    return [panel._colour_by.itemText(i)
            for i in range(panel._colour_by.count())]


# --------------------------------------------------------------------------- #
#  condition
# --------------------------------------------------------------------------- #

def test_condition_is_offered(qtbot):
    assert any("condition" in o for o in _options(_panel(qtbot, _frame())))


def test_condition_is_the_default_colouring(qtbot):
    """It is what a screen labels its controls with, so it is the colouring
    a reader wants first."""
    panel = _panel(qtbot, _frame())

    assert panel._colour_by.currentData() == "condition"


def test_a_single_valued_condition_is_still_offered(qtbot):
    """One value means the control names matched NO feature. That is a
    finding about the run; dropping the column silently hides it, and the
    maintainer reported exactly that as "the color by doesn't include
    condition"."""
    panel = _panel(qtbot, _frame(condition=["other"] * 400))

    assert any("condition (1)" in o for o in _options(panel))


def test_and_it_says_why_that_one_is_useless(qtbot):
    panel = _panel(qtbot, _frame(condition=["other"] * 400))

    assert "1 distinct value" in panel._colour_by_note
    assert "control names matched no feature" in panel._colour_by_note
    # And it reaches the user, not just the attribute.
    assert "Colouring:" in panel.status_text()


def test_a_useful_condition_says_nothing(qtbot):
    """A note that fires every time is a note nobody reads."""
    panel = _panel(qtbot, _frame())

    assert panel._colour_by_note == ""


# --------------------------------------------------------------------------- #
#  LOPIT
# --------------------------------------------------------------------------- #

def test_lopit_is_offered_on_a_toxoplasma_screen(qtbot):
    """It is NOT a column -- it is joined from the bundled TAGM table -- so
    the walk over frame.columns cannot see it."""
    options = _options(_panel(qtbot, _frame()))

    assert any("LOPIT" in o for o in options), options


def test_lopit_is_not_offered_when_nothing_is_annotated(qtbot):
    """An option that would colour nothing is indistinguishable from a broken
    one."""
    options = _options(_panel(qtbot, _frame(lopit=False)))

    assert not any("LOPIT" in o for o in options), options


def test_choosing_lopit_actually_colours(qtbot):
    panel = _panel(qtbot, _frame())
    index = panel._colour_by.findData(panel.LOPIT_KEY)
    assert index >= 0
    panel._colour_by.setCurrentIndex(index)

    assert len(panel.volcano._row_xy) == 400
    assert len(panel.volcano._legend_colours) > 1


def test_lopit_does_not_write_a_column_into_the_run_s_table(qtbot):
    """The same rule the baseline follows: the coefficient table beside the
    plot is the run's own, and a column appearing in it that the run never
    produced would be in the export too."""
    frame = _frame()
    before = list(frame.columns)
    panel = _panel(qtbot, frame)
    panel._colour_by.setCurrentIndex(panel._colour_by.findData(panel.LOPIT_KEY))

    assert list(frame.columns) == before
    assert list(panel._frame.columns) == before


def test_an_unannotated_gene_is_named_rather_than_blank(qtbot):
    """A blank legend entry reads as a rendering fault."""
    from spacr import localisation

    genes = _lopit_genes(50) + ["999999999"] * 10
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({
        "feature": [f"gene_fraction:gene[{g}]" for g in genes],
        "coefficient": rng.normal(0, .5, len(genes)),
        "p_value": rng.uniform(size=len(genes)),
        "condition": ["other"] * len(genes)})
    panel = _panel(qtbot, frame)
    panel._colour_by.setCurrentIndex(panel._colour_by.findData(panel.LOPIT_KEY))

    assert any("unannotated" in str(k)
               for k in panel.volcano._legend_colours), \
        panel.volcano._legend_colours
