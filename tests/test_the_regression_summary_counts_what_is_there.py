"""Columns the regression summary counts only when the table actually has them.

A coefficient table's columns depend on which fit produced it. A permutation
run has no ``significant`` column, a mixed fit has no ``passes_effect_size``,
an aggregated table has no ``cell_count``, and a re-read CSV can be missing any
of them. Every arc here is the summary asking first -- and the reason it must
is that the alternative is a KeyError in the middle of writing a report about a
fit that succeeded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _run(coef_df, **changes):
    """A ``_Run`` with only the fields these functions read populated."""
    from spacr.regression_summary import _Run

    fields = dict(res_folder=None, model=None, settings={}, coef_df=coef_df,
                  regression_type="ols", nonparametric=False, penalised=False,
                  data=None, data_note="", metrics={}, diagnostics={})
    fields.update(changes)
    return _Run(**fields)


# ---------------------------------------------------------------------------
# _design_counts — columns that are not there
# ---------------------------------------------------------------------------

def test_a_design_missing_the_id_columns_counts_only_its_rows():
    """Arcs 952 -> 950 and 954 -> 960: every optional count skipped.

    An aggregated design -- one row per gene, say -- has none of prc, grna or
    gene as columns. The row count is still true and still worth reporting,
    and the missing counts must be ABSENT rather than zero: a summary saying
    "0 wells" would describe a screen that had none.
    """
    from spacr.regression_summary import _design_counts

    out = _design_counts(pd.DataFrame({"effect": [1.0, 2.0, 3.0]}))

    assert out["n_rows_fitted"] == 3
    for absent in ("n_wells", "n_guides", "n_genes"):
        assert absent not in out or out[absent] is None
    assert "n_cells" not in out or out.get("n_cells") is None


def test_a_full_design_counts_every_column_it_has():
    """The taken sides, so the omissions above are visibly decisions."""
    from spacr.regression_summary import _design_counts

    frame = pd.DataFrame({
        "prc": ["p1_r1_c1", "p1_r1_c1", "p1_r1_c2"],
        "grna": ["g1", "g2", "g1"],
        "gene": ["A", "A", "B"],
        "cell_count": [100, 100, 50],
    })

    out = _design_counts(frame)

    assert out["n_rows_fitted"] == 3
    assert out["n_wells"] == 2
    assert out["n_guides"] == 2
    assert out["n_genes"] == 2


def test_something_that_is_not_a_frame_counts_none_of_anything():
    """The guard above the lot.

    Every key is present and every value is None. That shape matters: the
    caller formats the dict, and a MISSING key would raise while a None
    prints as "not recorded" -- so the guard returns the full set of keys
    rather than an empty mapping.
    """
    from spacr.regression_summary import _design_counts

    for value in (None, "not a frame"):
        out = _design_counts(value)
        assert set(out) >= {"n_rows_fitted", "n_wells", "n_guides", "n_genes"}
        assert all(v is None for v in out.values())


# ---------------------------------------------------------------------------
# _tested_mask — a table with no P-value column
# ---------------------------------------------------------------------------

def test_a_table_with_no_p_value_column_still_yields_a_tested_mask():
    """Arc 1478 -> 1480.

    The family is decided by the term NAME, and the P-value filter only
    narrows it. A permutation table re-read from disk can carry neither
    p_value nor permutation_p_value, and requiring one would make the summary
    report no tests at all for a run that plainly performed some.
    """
    from spacr.regression_summary import _tested_mask

    frame = pd.DataFrame({
        "feature": ["grna[g1]", "grna[g2]", "Intercept"],
        "coefficient": [0.5, -0.2, 1.0],
    })

    mask = _tested_mask(_run(frame))

    assert mask is not None
    assert mask.dtype == bool and mask.size == 3
    assert not mask[2]                       # the intercept is not a test


def test_a_table_with_a_p_value_column_narrows_by_it():
    """The taken side: a row with no P value is not a test."""
    from spacr.regression_summary import _tested_mask

    frame = pd.DataFrame({
        "feature": ["grna[g1]", "grna[g2]"],
        "coefficient": [0.5, -0.2],
        "p_value": [0.01, np.nan],
    })

    mask = _tested_mask(_run(frame))

    assert mask is not None
    assert bool(mask[0]) and not bool(mask[1])


# ---------------------------------------------------------------------------
# _hit_mask — an effect-size cut that was never applied
# ---------------------------------------------------------------------------

def test_hits_are_counted_on_significance_alone_when_no_effect_cut_exists():
    """Arc 1504 -> 1507, and the note must not claim a cut that was not made.

    ``passes_effect_size`` is written only when an effect-size threshold was
    configured. The note is printed beside the count, so adding "AND at least
    as wide as the effect-size cut" to a run that had none would misdescribe
    how the hits were chosen -- in the sentence a reader quotes.
    """
    from spacr.regression_summary import _hit_mask

    frame = pd.DataFrame({
        "feature": ["grna[g1]", "grna[g2]"],
        "significant": [True, False],
    })

    mask, note = _hit_mask(_run(frame))

    assert mask is not None and list(mask) == [True, False]
    assert "corrected P below" in note
    assert "effect-size cut" not in note


def test_hits_are_narrowed_by_the_effect_cut_when_one_was_applied():
    """The taken side, and the note says so."""
    from spacr.regression_summary import _hit_mask

    frame = pd.DataFrame({
        "feature": ["grna[g1]", "grna[g2]"],
        "significant": [True, True],
        "passes_effect_size": [True, False],
    })

    mask, note = _hit_mask(_run(frame))

    assert list(mask) == [True, False]
    assert "effect-size cut" in note


def test_no_coefficient_table_reports_why_rather_than_a_count():
    """The early return, which is the reason the count is Optional."""
    from spacr.regression_summary import _hit_mask

    mask, note = _hit_mask(_run(None))

    assert mask is None
    assert "no coefficient table" in note
