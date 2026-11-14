"""The last turns in the statistics modules.

Closed here:

* :mod:`spacr.regression_summary` -- the unidentifiability warning without a
  rank deficiency, a cell count that is not finite, and a LASSO selection
  table with no coefficient column to intersect with.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from spacr import regression_summary


# ---------------------------------------------------------------------------
# regression_summary
# ---------------------------------------------------------------------------

def _run(**over):
    base = dict(
        res_folder=None, model=None, settings={}, coef_df=None,
        regression_type="ols", nonparametric=False, penalised=False,
        data=None, data_note="no run folder was given", metrics={},
    )
    base.update(over)
    return regression_summary._Run(**base)


def test_a_wide_fit_warns_once_and_a_rank_deficient_one_warns_twice():
    """More parameters than rows is one sentence; a short rank adds another.

    The second sentence quotes a rank, so it must not be printed when the run
    recorded no rank at all -- a blank there reads as rank zero.
    """
    wide = regression_summary._warnings(
        _run(metrics={"n_parameters": 12, "n_observations": 8}))
    assert len(wide) == 1
    assert "8 analysed observations" in wide[0]

    deficient = regression_summary._warnings(
        _run(metrics={"n_parameters": 12, "n_observations": 8,
                      "design_rank": 5}))
    assert len(deficient) == 2
    assert "rank 5 against 12 columns" in deficient[1]
    assert "7 direction(s)" in deficient[1]


def test_a_cell_count_that_is_not_finite_is_reported_as_unknown():
    """``n_cells`` stays None rather than becoming a nonsense integer.

    ``int(inf)`` raises and ``int(nan)`` is undefined, so a column carrying
    either has to leave the field unset -- the summary's own rule is that a
    number it cannot compute is named, never guessed at.
    """
    infinite = pd.DataFrame({"prc": ["p1_r1_c1", "p1_r1_c2"],
                             "cell_count": [np.inf, 3.0]})
    counts = regression_summary._design_counts(infinite)
    assert counts["n_cells"] is None
    assert counts["n_wells"] == 2
    assert counts["n_rows_prepared"] == 2
    assert "n_rows_fitted" not in counts

    finite = infinite.assign(cell_count=[5.0, 3.0])
    assert regression_summary._design_counts(finite)["n_cells"] == 8


def test_a_selection_table_with_no_coefficient_column_still_counts_hits():
    """Selection frequency alone is enough to say what was called.

    A bootstrap-LASSO table normally carries the coefficient too, and the
    called set is the intersection of "selected often" and "not shrunk to
    zero". With no coefficient column there is nothing to intersect, and the
    frequency alone has to stand -- the note says so rather than the count
    silently meaning something else.
    """
    frame = pd.DataFrame({"grna": ["a", "b", "c"],
                          "selection_frequency": [0.9, 0.1, 0.7]})
    mask, note = regression_summary._hit_mask(
        _run(coef_df=frame, penalised=True))
    assert list(mask) == [True, False, True]
    assert "NOT a P value" in note

    # With the coefficient column present the zero-coefficient row drops out,
    # which is the intersection the missing column cannot be part of.
    with_coefficients = frame.assign(coefficient=[0.4, 0.0, 0.0])
    intersected, _note = regression_summary._hit_mask(
        _run(coef_df=with_coefficients, penalised=True))
    assert list(intersected) == [True, False, False]
