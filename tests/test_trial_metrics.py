"""What a sweep row has to carry to be worth reading.

A row saying "10 hits" cannot be used to CHOOSE a configuration, which is the
only reason to sweep. The nightly run made that concrete: min_cell_count=50
reported MORE hits than 100 while quietly losing GRA14, and no column said so.
"""
import numpy as np
import pandas as pd
import pytest


def _results():
    return pd.DataFrame({
        "feature": ["Intercept",
                    "gene_fraction:gene[239740]",   # the positive control
                    "gene_fraction:gene[233460]",   # the negative control
                    "gene_fraction:gene[225160]",
                    "fraction:grna[225160_1]"],
        "coefficient": [0.1, 1.4, 0.05, 1.2, 0.9],
        "p_value": [1e-30, 6.4e-08, 0.28, 4.6e-08, 0.4],
        "q_value": [1e-28, 1.9e-05, 0.51, 1.8e-05, 0.6],
    })


class TestControlRecovery:
    """The yardstick, when the user has named one."""

    def test_the_positive_control_is_ranked_among_real_coefficients(self):
        """The intercept is not a candidate hit; counting it shifts every
        rank by one."""
        from spacr.trial_metrics import control_recovery

        out = control_recovery(_results(),
                               {"positive_control": "239740",
                                "negative_control": "233460"})
        assert out["positive_control_found"]
        # EAF1 is smaller, so the positive control is second among the reals.
        assert out["positive_control_rank"] == 2
        assert out["positive_control_q"] == pytest.approx(1.9e-05)

    def test_the_negative_control_is_reported_too(self):
        """A negative control promoted into the hits is the most important
        thing a configuration can tell you."""
        from spacr.trial_metrics import control_recovery

        out = control_recovery(_results(), {"negative_control": "233460"})
        assert out["negative_control_found"]
        assert out["negative_control_q"] == pytest.approx(0.51)

    def test_the_separation_between_them_is_one_number(self):
        from spacr.trial_metrics import control_recovery

        out = control_recovery(_results(),
                               {"positive_control": "239740",
                                "negative_control": "233460"})
        assert out["control_rank_separation"] == (
            out["negative_control_rank"] - out["positive_control_rank"])

    def test_a_control_that_is_absent_says_so_rather_than_vanishing(self):
        from spacr.trial_metrics import control_recovery

        out = control_recovery(_results(), {"positive_control": "999999"})
        assert out["positive_control_found"] is False

    def test_no_control_named_means_no_columns_invented(self):
        from spacr.trial_metrics import control_recovery

        assert control_recovery(_results(), {}) == {}


class TestFitQuality:

    def test_it_reads_what_the_model_reports(self):
        import statsmodels.api as sm

        from spacr.trial_metrics import fit_quality

        rng = np.random.default_rng(0)
        X = sm.add_constant(rng.normal(size=(200, 3)))
        y = X @ np.array([1, 2, -1, 0.5]) + rng.normal(size=200)
        out = fit_quality(sm.OLS(y, X).fit())

        assert 0.0 < out["r_squared"] < 1.0
        assert "aic" in out and "bic" in out
        assert out["n_observations"] == 200
        assert out["residual_se"] > 0

    def test_a_family_without_r_squared_contributes_nothing_rather_than_nan(self):
        """Reporting an absent statistic is honest; inventing one is not.

        The thirteen families do not agree on which of these exist -- a robust
        fit has no R-squared and a permutation test has no model at all.
        """
        from spacr.trial_metrics import fit_quality

        class _Bare:
            pass

        assert fit_quality(_Bare()) == {}
        assert fit_quality(None) == {}


class TestResidualDiagnostics:
    """These decide whether the p-values mean anything."""

    def test_a_well_behaved_fit_passes_its_own_checks(self):
        import statsmodels.api as sm

        from spacr.trial_metrics import residual_diagnostics

        rng = np.random.default_rng(0)
        X = sm.add_constant(rng.normal(size=(300, 2)))
        y = X @ np.array([1, 2, -1]) + rng.normal(size=300)
        out = residual_diagnostics(sm.OLS(y, X).fit())

        assert out["breusch_pagan_p"] > 0.01      # homoscedastic
        assert out["jarque_bera_p"] > 0.01        # normal
        assert 1.5 < out["durbin_watson"] < 2.5   # uncorrelated

    def test_heteroscedasticity_is_detected(self):
        """A funnel inflates or deflates every standard error in the fit, so
        a heteroscedastic model can rank genes plausibly and be wrong about
        all of them."""
        import statsmodels.api as sm

        from spacr.trial_metrics import residual_diagnostics

        rng = np.random.default_rng(0)
        x = rng.uniform(1, 10, size=400)
        X = sm.add_constant(x)
        y = 2 * x + rng.normal(size=400) * x      # variance grows with x
        out = residual_diagnostics(sm.OLS(y, X).fit())
        assert out["breusch_pagan_p"] < 0.05

    def test_too_few_residuals_is_not_an_error(self):
        from spacr.trial_metrics import residual_diagnostics

        class _Tiny:
            resid = np.array([1.0, -1.0])
            fittedvalues = np.array([0.5, 0.5])

        assert residual_diagnostics(_Tiny()) == {}


class TestCalibration:

    def test_a_flat_null_reports_inflation_near_one(self):
        from spacr.trial_metrics import calibration

        rng = np.random.default_rng(0)
        out = calibration(pd.DataFrame({"p_value": rng.uniform(size=2000)}))
        assert 0.8 < out["genomic_inflation"] < 1.25

    def test_signal_shows_as_a_first_bin_excess(self):
        from spacr.trial_metrics import calibration

        rng = np.random.default_rng(0)
        p = np.concatenate([rng.uniform(size=1000), np.full(100, 1e-8)])
        out = calibration(pd.DataFrame({"p_value": p}))
        assert out["p_first_bin_excess"] > 50


class TestTheWholeRow:

    def test_one_missing_metric_is_a_nan_not_a_lost_row(self):
        """A family with no R-squared must still contribute its controls."""
        from spacr.trial_metrics import summarise_trial

        row = summarise_trial({"results": _results(), "model": None},
                              {"positive_control": "239740",
                               "negative_control": "233460"})
        assert "r_squared" not in row            # no model to ask
        assert row["positive_control_rank"] == 2  # but this survived
        assert "genomic_inflation" not in row or row["n_tests"] >= 0

    def test_an_empty_output_returns_an_empty_row(self):
        from spacr.trial_metrics import summarise_trial

        assert summarise_trial({}, {}) == {}
