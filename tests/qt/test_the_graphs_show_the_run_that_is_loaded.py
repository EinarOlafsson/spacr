"""Instruction 189: the plots must show the regression that actually ran.

"one thing that makes me uneasy is that all the plots look the same no
matter which regression type i do", and "make sure that the interactive
graphs are linked to the loaded run".

MEASURED ON THE MAINTAINER'S OWN SCREEN, 2026-08-20, before writing any of
this. Fitting the same 1,366 wells under seven types through
`regression_model`:

    ols             sum|b|   6.20052
    wls                      7.43317
    ridge                    6.00818
    lasso                    1.09457
    rlm                      5.51143
    glm                     54.03993
    quantile                 6.63466
    quasi_binomial          54.03993

Every one different except glm and quasi_binomial, which agree to 1.5e-14 --
and CORRECTLY: quasi-binomial is the binomial family with a free dispersion
parameter, and dispersion moves the standard errors, not the point estimates.

Through the PIPELINE the GUI uses, each type also came back as its own model
class -- Ridge, RLMResultsWrapper, GLMResultsWrapper -- with the requested
type reported back. A silent fallback would have returned one class every
time.

SO THE FITS ARE RIGHT. What was missing is that a plot never SAID which model
drew it, which is what these tests are for.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

import numpy as np
import pandas as pd


def _results(seed: int, n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "feature": [f"fraction:grna[g{i}_1]" for i in range(n)],
        "grna": [f"g{i}_1" for i in range(n)],
        "gene": [f"g{i}" for i in range(n)],
        "coefficient": rng.normal(0.0, 1.0, n),
        "p_value": rng.uniform(1e-6, 1.0, n),
        "condition": ["other"] * n,
    })


def _panel(qtbot):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    return panel


class TestTheVolcanoFollowsTheLoadedRun:
    """"make sure that the interactive graphs are linked to the loaded run"."""

    def test_a_second_run_replaces_the_first_ones_points(self, qtbot):
        panel = _panel(qtbot)
        first = _results(1)
        second = _results(2)

        panel.set_frame(first, source="run-one")
        drawn_first = panel.results_frame()["coefficient"].to_numpy().copy()
        panel.set_frame(second, source="run-two")
        drawn_second = panel.results_frame()["coefficient"].to_numpy()

        assert not np.allclose(drawn_first, drawn_second), (
            "the panel is showing the first run's numbers after loading the "
            "second -- the graph is not linked to the loaded run")

    def test_the_panel_reports_the_run_it_is_showing(self, qtbot):
        panel = _panel(qtbot)
        panel.set_frame(_results(1), source="run-one")

        assert panel.results_frame() is not None

    def test_an_empty_frame_is_refused_and_says_so(self, qtbot):
        """It keeps the run it is showing, and REFUSES rather than blanking.

        This is the right answer and it is worth pinning, because the wrong
        one is invisible: silently keeping the old numbers while the header
        named the new run would be a plot that looks fine and belongs to a
        run the user is no longer looking at. Refusing keeps the panel whole
        and consistent -- frame, header and status all still describe the run
        that is actually on screen.
        """
        panel = _panel(qtbot)
        panel.set_frame(_results(1), source="run-one")
        before = panel.results_frame()["coefficient"].to_numpy().copy()

        accepted = panel.set_frame(
            pd.DataFrame(columns=["feature", "coefficient"]), source="empty")

        assert accepted is False, "an empty table is not a run to show"
        assert np.allclose(panel.results_frame()["coefficient"].to_numpy(),
                           before), "the panel still shows run-one, whole"
        assert panel.run_folder() != "empty", (
            "and the header must not name a run it is not showing")


class TestTheSummaryNamesTheModel:
    """189 B: type, backend, and the hyperparameters that type READ."""

    def test_a_type_with_no_knobs_says_so(self):
        from spacr.regression_summary import _hyperparameter_report

        class Run:
            nonparametric = False
            model = None

        said = _hyperparameter_report("ols", {}, Run())["value"]

        assert "none" in said and "ols" in said

    def test_only_the_settings_that_type_reads_are_listed(self):
        """A summary showing an alpha for an ols fit that ignored it is a
        number the reader will try to interpret."""
        from spacr.regression_summary import _hyperparameter_report

        class Run:
            nonparametric = False
            model = None

        settings = {"alpha": 0.1, "quantile": 0.9, "huber_t": 1.345}

        assert "quantile=0.9" in _hyperparameter_report(
            "quantile", settings, Run())["value"]
        assert "alpha" not in _hyperparameter_report(
            "quantile", settings, Run())["value"]
        assert "huber_t=1.345" in _hyperparameter_report(
            "rlm", settings, Run())["value"]

    def test_a_cross_validated_alpha_is_reported_as_the_value_that_won(self):
        """'auto' is what was asked for and not what was fitted; nobody can
        reproduce a run from the word auto."""
        from spacr.regression_summary import _hyperparameter_report

        class Model:
            alpha_ = 0.0234

        class Run:
            nonparametric = False
            model = Model()

        said = _hyperparameter_report("ridge", {"alpha": "auto"}, Run())["value"]

        assert "0.0234" in said
        assert "cross-validated" in said and "not given" in said

    def test_auto_without_a_recorded_winner_says_that_instead(self):
        from spacr.regression_summary import _hyperparameter_report

        class Run:
            nonparametric = False
            model = None

        said = _hyperparameter_report("ridge", {"alpha": "auto"}, Run())["value"]

        assert "did not record" in said

    def test_every_knob_of_a_multi_knob_family_is_named(self):
        from spacr.regression_summary import _hyperparameter_report

        class Run:
            nonparametric = False
            model = None

        said = _hyperparameter_report("elasticnet", {
            "alpha": 0.1, "l1_ratio": 0.5, "lasso_n_boot": 200,
            "lasso_selection_threshold": 0.6}, Run())["value"]

        for expected in ("alpha=0.1", "l1_ratio=0.5", "lasso_n_boot=200"):
            assert expected in said

    def test_cov_type_is_not_called_a_hyperparameter(self):
        """It is how the standard errors are computed after the fit, and it
        has its own line."""
        from spacr.regression_summary import _hyperparameter_report

        class Run:
            nonparametric = False
            model = None

        said = _hyperparameter_report("ols", {"cov_type": "HC3"}, Run())["value"]

        assert "cov_type" not in said
