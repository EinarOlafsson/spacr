"""Regression QC edges that are refused before a statistic can mislead.

They sit where a statistic would otherwise be quoted off data that
cannot support it -- which in a QC report is worse than no statistic,
because the reader takes the number as the answer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from matplotlib.figure import Figure

from spacr import regression_qc as rq


def _axes():
    """A bare axes on an object-oriented figure, as the report driver makes."""
    return Figure(figsize=(4.6, 3.6)).add_subplot(1, 1, 1)


class _Fit:
    def __init__(self, **attributes):
        self.__dict__.update(attributes)


def _context(fitted, y, X=None, **kwargs):
    response = np.asarray(y, dtype=float)
    n = response.size
    if X is None:
        X = pd.DataFrame({"intercept": np.ones(n),
                          "x": np.linspace(0.0, 1.0, n)})
    return rq.build_context(_Fit(fittedvalues=np.asarray(fitted, dtype=float)),
                            X, response, **kwargs)


class TestRecoveringTheDesignFromAModel:

    def test_a_model_without_its_design_is_refused_by_name(self):
        """A fitted estimator that stores no design cannot recreate residuals."""
        with pytest.raises(rq.PanelUnavailable, match="does not keep the design matrix"):
            rq.context_from_model(_Fit(), regression_type="lasso")

    def test_a_statsmodels_result_rebuilds_the_context_it_fitted(self):
        """The refusal above must not swallow models that do retain their data."""
        sm = pytest.importorskip("statsmodels.api")
        X = np.column_stack([np.ones(12), np.linspace(-1.0, 1.0, 12)])
        y = 1.0 + 2.0 * X[:, 1]
        model = sm.OLS(y, X).fit()

        ctx = rq.context_from_model(model, regression_type="ols")

        assert ctx.n == 12 and ctx.p == 2
        np.testing.assert_allclose(ctx.X.to_numpy(), X)
        np.testing.assert_allclose(ctx.y, y)


class TestThePValueExpectation:

    def test_every_admitted_histogram_has_a_finite_uniform_expectation(self):
        """A non-empty finite p-value sample always has ``n / 20`` to draw."""
        for n in range(1, 257):
            diag = rq.diagnose_p_value_histogram(np.linspace(0.0, 1.0, n))
            assert diag["expected"] == pytest.approx(n / 20.0)
            assert np.isfinite(diag["expected"])

    def test_a_histogram_with_no_finite_p_values_is_refused_first(self):
        """The only non-finite expectation belongs to the sample not admitted."""
        ctx = _context(
            np.arange(6.0), np.arange(6.0),
            coef_df=pd.DataFrame({"p_value": [np.nan, np.inf]}),
        )

        with pytest.raises(rq.PanelUnavailable, match="every p-value is non-finite"):
            rq._panel_p_value_histogram(ctx, _axes())
