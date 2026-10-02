"""What a regression records about the design it actually fitted.

``coef_df.attrs['fit_design']`` is read by the run summary to say how many
rows, wells, guides and genes a fit rested on. These tests pin the cases the
ordinary statsmodels fits never reach: a mixed-model result that carries no
statsmodels data (the torch backend's), a model that dropped rows from a table
whose index repeats, and a wide design whose frame keeps only well columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")

from spacr import ml
from tests.test_cov_12_ml_regression_pipeline import NC, PC, fit, wells_frame


class _BareMixedResult:
    """A mixed-model result with a row count and a fixed-effect count and
    none of statsmodels' ``model.data`` / ``model.exog``."""

    def __init__(self, nobs, k_fe):
        self.nobs = nobs
        self.k_fe = k_fe


def _coefficients():
    return pd.DataFrame({"feature": ["gene3", "gene4"],
                         "coefficient": [0.2, -0.1],
                         "p_value": [0.04, 0.6]})


def test_a_mixed_result_without_statsmodels_data_still_reports_its_design(
        tmp_path, monkeypatch):
    frame = wells_frame(seed=7)
    captured = {}

    def bare_fit(df, formula, dst, **kwargs):
        captured["rows"] = len(df)
        return _BareMixedResult(len(df), 5), _coefficients()

    monkeypatch.setattr(ml, "fit_mixed_model", bare_fit)
    _model, coef_df, kind = fit(frame, tmp_path, regression_type="mixed")
    design = coef_df.attrs["fit_design"]
    assert kind == "mixed"
    assert design["n_rows_fitted"] == captured["rows"]
    assert design["n_design_columns"] == 5
    assert design["layout"] == "long"
    assert design["n_wells"] == frame["prc"].nunique()
    assert design["n_genes"] == frame["gene"].nunique()


def test_a_model_that_dropped_rows_from_a_repeating_index_counts_only_rows(
        tmp_path):
    """Rows the formula cannot use leave the design; with a repeating index
    they cannot be matched back, so the counts rest on the model alone."""
    frame = wells_frame(seed=8)
    frame.index = np.arange(len(frame)) // 2
    frame["rowID"] = frame["rowID"].astype(object)
    frame.loc[frame.index[:4], "rowID"] = None
    _model, coef_df, _kind = fit(frame, tmp_path)
    design = coef_df.attrs["fit_design"]
    assert design["n_rows_fitted"] < len(frame)
    assert "n_wells" not in design and "n_guides" not in design
