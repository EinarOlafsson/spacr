"""The residual diagnostics are live tabs, not only a PDF on disk.

Instruction 128 E, in the maintainer's words:

    "in the tabs Q-Q and Controls, there should be Tabs like residuals showing
     the residuals and regression controll graphs like that."

Every tab the panel had was drawn from the COEFFICIENT table -- one row per
guide -- which says nothing about how well the model fitted the wells it was
given. `spacr.regression_qc` computes exactly that and had one consumer: a
folder of PDFs written beside the results. Nobody looks in a folder.

THE NUMBERS COME FROM `regression_qc`, not from arithmetic repeated here. A
live influence panel that named different wells than the report beside it
would be worse than no live panel, and two implementations of Cook's distance
disagree within a week.

THE TABS EXIST BEFORE THEY CAN BE FILLED, and say what they are waiting for.
A diagnostic that appears only once it happens to be computable is one nobody
knows to look for, and a fit that CANNOT be diagnosed -- the penalised
backends keep no design matrix -- must not look like a broken tab.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
sm = pytest.importorskip("statsmodels.api")

pytestmark = pytest.mark.qt


DIAGNOSTIC_TABS = ("Residuals", "Scale-location", "Influence")


@pytest.fixture()
def coefficients():
    rng = np.random.default_rng(4)
    n = 200
    return pd.DataFrame({
        "feature": ["Intercept"] + [f"fraction:grna[{i}_1]" for i in range(n)],
        "coefficient": np.concatenate([[0.2], rng.normal(size=n)]),
        "p_value": np.concatenate([[1e-40], rng.uniform(size=n)]),
    })


def _design(n=240, seed=7):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "Intercept": np.ones(n),
        "fraction:grna[a]": rng.random(n),
        "fraction:grna[b]": rng.random(n),
    })
    y = 0.3 + 1.2 * X["fraction:grna[a]"] + rng.normal(scale=0.2, size=n)
    return X, y


def _ols(n=240, seed=7):
    X, y = _design(n, seed)
    return sm.OLS(y, X).fit()


@pytest.fixture()
def panel(qtbot):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    widget = RegressionResultsPanel()
    qtbot.addWidget(widget)
    return widget


def _tab_names(panel):
    return [panel.tabs.tabText(i) for i in range(panel.tabs.count())]


# --------------------------------------------------------------------------- #
#  They are tabs, and they are beside Q-Q and Controls
# --------------------------------------------------------------------------- #

def test_the_residual_diagnostics_sit_beside_q_q_and_controls(panel):
    """Asked for by position, not merely by existence: "in the tabs Q-Q and
    Controls, there should be Tabs like residuals"."""
    names = _tab_names(panel)

    for wanted in DIAGNOSTIC_TABS:
        assert wanted in names, f"{wanted} is not a tab: {names}"
    assert names.index("Residuals") == names.index("Controls") + 1, (
        f"the residuals are not next to the controls: {names}")


def test_an_unfilled_diagnostic_tab_says_what_it_is_waiting_for(panel):
    """An empty plot with no sentence is indistinguishable from a broken one,
    and this one is empty for a reason the user can act on."""
    for plot in panel.diagnostic_plots():
        said = plot._status.text()
        assert "fitted model" in said, said
        assert "regression_qc/" in said, (
            "the panel does not say the same diagnostics are already on disk")


def test_a_table_read_off_disk_leaves_them_empty_and_says_so(panel,
                                                             coefficients):
    """A results CSV is one row per guide. There is no well in it, so there is
    no residual, and the panel has to say that rather than draw nothing."""
    assert panel.set_frame(coefficients, source="results.csv")

    for plot in panel.diagnostic_plots():
        assert not len(plot.plot.listDataItems())
        assert "only a run in this session" in plot._status.text()


# --------------------------------------------------------------------------- #
#  Filled from the fit, with regression_qc's own numbers
# --------------------------------------------------------------------------- #

def test_the_run_s_own_model_fills_all_three(panel, coefficients):
    panel.set_frame(coefficients)

    assert panel.set_diagnostics(_ols(), regression_type="ols") is True

    for plot in panel.diagnostic_plots():
        assert len(plot.plot.listDataItems()), (
            f"{type(plot).__name__} drew nothing from a fitted OLS")


def test_the_influence_panel_names_the_same_wells_as_the_saved_report(
        panel, coefficients):
    """The whole reason the arrays come from `regression_qc`.

    Cook's distance computed twice, in two modules, is two answers within a
    week -- and the one on screen would be the one the user acts on while the
    PDF beside it said something else.
    """
    from spacr.regression_qc import context_from_model, cooks_distance

    model = _ols()
    panel.set_frame(coefficients)
    panel.set_diagnostics(model, regression_type="ols")

    ctx = context_from_model(model, regression_type="ols")
    expected = cooks_distance(ctx.std_resid, ctx.leverage, ctx.p)
    flagged = int(np.sum(expected > 4.0 / len(expected)))

    assert flagged > 1, "the fixture flags no influential wells to compare"
    assert f"{flagged} past Cook's D" in panel.influence._status.text()
    assert np.allclose(panel.influence._cooks, expected, equal_nan=True)


def test_the_residual_plot_holds_one_point_per_well_not_per_guide(
        panel, coefficients):
    """The distinction the whole tab exists for. The coefficient table has 201
    rows and the fit had 240 wells; a residual tab drawn from the table would
    be plotting the wrong population and would look entirely plausible."""
    panel.set_frame(coefficients)
    assert len(coefficients) != 240

    panel.set_diagnostics(_ols(n=240), regression_type="ols")

    assert "240 residuals" in panel.residuals._status.text()


def test_the_scale_location_panel_reports_the_variance_trend(panel):
    """The variance-homogeneity panel the maintainer asked for by name. Its
    whole content is the slope: flat is constant variance, rising means every
    p-value on the volcano depends on the fitted value."""
    panel.set_diagnostics(_ols(), regression_type="ols")

    said = panel.scale_location._status.text()
    assert "Trend slope" in said and "constant variance" in said


# --------------------------------------------------------------------------- #
#  A fit that cannot answer says so, rather than looking broken
# --------------------------------------------------------------------------- #

def test_a_penalised_fit_says_why_it_has_no_residuals(panel):
    """spaCR's lasso/ridge/elasticnet backends are sklearn estimators that
    keep neither the design nor the response, so nothing can be recomputed
    from them. "this fit cannot answer that" and "this tab is broken" must not
    look the same."""
    Lasso = pytest.importorskip("sklearn.linear_model").Lasso
    X, y = _design()

    assert panel.set_diagnostics(Lasso(alpha=0.1).fit(X.to_numpy(),
                                                      y.to_numpy())) is False

    for plot in panel.diagnostic_plots():
        said = plot._status.text()
        assert "does not keep the design matrix" in said, said
        assert not len(plot.plot.listDataItems())


def test_no_model_at_all_is_an_answer_not_a_crash(panel):
    assert panel.set_diagnostics(None) is False
    assert "fitted model" in panel.residuals._status.text()


def test_a_glm_binomial_fit_is_diagnosed_on_its_pearson_residual(panel):
    """spaCR routes logit/probit through GLM-Binomial with a fraction response
    weighted by cell count. Standardising THAT on the OLS error variance would
    be a z-score of the wrong quantity; `regression_qc` resolves it per model
    class, which is why this panel asks it instead of dividing by a scale."""
    from spacr.regression_qc import context_from_model

    rng = np.random.default_rng(2)
    X, _ = _design(180)
    fraction = np.clip(0.2 + 0.5 * X["fraction:grna[a]"]
                       + rng.normal(scale=0.05, size=180), 0.01, 0.99)
    weights = rng.integers(20, 300, 180).astype(float)
    model = sm.GLM(fraction, X, family=sm.families.Binomial(),
                   var_weights=weights).fit()

    assert panel.set_diagnostics(model, regression_type="logit") is True
    ctx = context_from_model(model, regression_type="logit")
    assert "Pearson residual" in ctx.standardisation.metric
    assert len(panel.influence.plot.listDataItems())


def test_a_new_table_clears_the_previous_fit_s_residuals(panel, coefficients):
    """A new table is a new experiment. Leaving the last run's residuals up
    would describe a fit the user is no longer looking at, with nothing on
    screen saying so -- the same failure the selection clearing exists for."""
    panel.set_frame(coefficients)
    panel.set_diagnostics(_ols(), regression_type="ols")
    assert len(panel.residuals.plot.listDataItems())

    panel.set_frame(coefficients.iloc[:100])

    assert not len(panel.residuals.plot.listDataItems())
    assert "only a run in this session" in panel.residuals._status.text()


# --------------------------------------------------------------------------- #
#  The screen hands the model over
# --------------------------------------------------------------------------- #

def test_the_run_s_payload_reaches_the_diagnostic_tabs(qtbot, coefficients):
    """`perform_regression` returns the fitted model and the screen was
    dropping it. It is the ONLY place a model is available -- there is no path
    from a results folder back to a fit -- so a screen that ignored it left
    the three tabs permanently empty."""
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)

    screen._on_pipeline_result({
        "results": coefficients,
        "model": _ols(),
        "regression_type": "ols",
        "res_folder": "",
        "settings": {},
    })

    panel = screen._results_panel
    assert len(panel.residuals.plot.listDataItems()), (
        "the run handed over a model and the residual tab stayed empty")
    assert "240 residuals" in panel.residuals._status.text()


def test_a_payload_with_no_model_does_not_take_the_screen_down(
        qtbot, coefficients):
    """Not every module's payload carries one, and a KeyError here would lose
    the coefficient table as well as the diagnostics."""
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)

    screen._on_pipeline_result({"results": coefficients, "res_folder": ""})

    assert screen._results_panel.table.table.rowCount() == len(coefficients)
    assert "fitted model" in screen._results_panel.residuals._status.text()
