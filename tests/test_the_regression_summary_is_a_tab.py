"""The statsmodels summary, as a tab.

Asked for on 2026-08-17: "statmodels provides a summary, please generate a
tab for the regression summary".

Rendered VERBATIM from `model.summary()` rather than rebuilt: the point of
asking for the statsmodels summary is to get the statsmodels summary, and a
re-implementation would differ from every textbook and every other tool a
reader compares it against.

THE DANGEROUS CASE IS THE ONE THIS SCREEN ACTUALLY HAS. The tsg101 fit is not
identifiable -- 610 wells estimating 827 parameters -- and statsmodels prints
a full table of standard errors and P values regardless, which looks exactly
like a summary of a well-posed fit. A summary pasted into a methods section
from an unidentifiable fit is the worst thing this panel can produce, and it
is one click away, so the warning travels WITH the table.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sm = pytest.importorskip("statsmodels.api")

from spacr.qt.widgets.regression_results import summary_text  # noqa: E402


def _well_posed():
    rng = np.random.default_rng(0)
    X = sm.add_constant(rng.normal(size=(80, 3)))
    y = X @ [1, .5, -.3, .2] + rng.normal(0, .3, 80)
    return sm.OLS(y, X).fit()


def _under_determined():
    """More parameters than observations -- the real screen's shape."""
    rng = np.random.default_rng(1)
    X = sm.add_constant(rng.normal(size=(20, 25)))
    return sm.OLS(rng.normal(size=20), X).fit()


# --------------------------------------------------------------------------- #
#  It is the real summary
# --------------------------------------------------------------------------- #

def test_it_is_statsmodels_own_summary():
    text = summary_text(_well_posed())

    assert "OLS Regression Results" in text
    assert "R-squared" in text


def test_the_header_numbers_are_there():
    """The header is the useful part and the part that scrolls away."""
    text = summary_text(_well_posed())

    for field in ("R-squared", "F-statistic", "AIC", "Log-Likelihood",
                  "Df Residuals"):
        assert field in text, field


def test_a_well_posed_fit_gets_no_warning():
    """A warning that fires every time is a warning nobody reads."""
    assert "NOT IDENTIFIABLE" not in summary_text(_well_posed())


# --------------------------------------------------------------------------- #
#  The dangerous case
# --------------------------------------------------------------------------- #

def test_an_unidentifiable_fit_says_so_above_its_own_table():
    """statsmodels prints the table regardless and it looks well posed."""
    text = summary_text(_under_determined())

    assert text.startswith("THIS FIT IS NOT IDENTIFIABLE")
    # And the table is still there -- the warning annotates, it does not
    # replace. A reader who wants the numbers should still get them.
    assert "OLS Regression Results" in text


def test_the_warning_names_the_two_numbers():
    """"not identifiable" without the counts is an assertion; with them it is
    a diagnosis the reader can check."""
    text = summary_text(_under_determined())

    assert "20 analysed observations" in text
    assert "26 parameters" in text


def test_the_warning_names_the_way_out():
    text = summary_text(_under_determined())

    assert "inference='nonparametric'" in text


def test_the_warning_is_read_off_the_model():
    """Not off the settings, so the tab cannot disagree with the table it is
    printed above."""
    import inspect

    from spacr.qt.widgets import regression_results

    source = inspect.getsource(regression_results._identifiability_warning)
    assert "nobs" in source
    assert "params" in source


# --------------------------------------------------------------------------- #
#  Backends that have no summary
# --------------------------------------------------------------------------- #

def test_a_non_statsmodels_backend_says_which_one():
    """lasso, ridge, elasticnet and hinge are sklearn fits with no summary.
    An absent or empty tab reads as a bug."""

    class Lasso:
        pass

    text = summary_text(Lasso(), "lasso")

    assert text.startswith("No summary")
    assert "lasso" in text
    # And it says what to look at instead.
    assert "selection frequency" in text


def test_no_model_says_why_rather_than_being_blank():
    text = summary_text(None)

    assert text.startswith("No summary")
    assert "results table on disk" in text


def test_a_summary_that_raises_is_reported_not_propagated():
    """A tab that takes the panel down with it is worse than one that says
    it could not render."""

    class Broken:
        def summary(self):
            raise RuntimeError("no design matrix")

    text = summary_text(Broken(), "glm")

    assert text.startswith("No summary")
    assert "no design matrix" in text


# --------------------------------------------------------------------------- #
#  The tab
# --------------------------------------------------------------------------- #

@pytest.mark.qt
def test_the_panel_has_a_summary_tab(qtbot):
    pytest.importorskip("PySide6")
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)

    tabs = [panel.tabs.tabText(i) for i in range(panel.tabs.count())]
    assert "Summary" in tabs, tabs


@pytest.mark.qt
def test_it_is_selectable_and_monospaced(qtbot):
    """The reason to want it is to paste a number into a methods section. A
    summary you cannot select is a summary you retype, and a proportional
    font destroys the column alignment statsmodels relies on."""
    pytest.importorskip("PySide6")
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    panel.set_summary(_well_posed())

    from PySide6.QtGui import QFontDatabase

    assert panel._summary.isReadOnly()
    # Compared against the font the widget was GIVEN rather than against a
    # style hint: Qt resolves a fixed font per platform and `fixedPitch()`
    # reports the request, not the resolution, so it is False on a font that
    # is in fact monospaced.
    expected = QFontDatabase.systemFont(QFontDatabase.FixedFont)
    assert panel._summary.font().family() == expected.family()
    assert "R-squared" in panel._summary.toPlainText()


@pytest.mark.qt
def test_the_run_hands_the_model_to_the_summary(qtbot):
    """The seam: `set_diagnostics` and `set_summary` take the same model at
    the same moment, so the two tabs cannot describe different fits."""
    import inspect

    from spacr.qt.screens import app_screen

    source = inspect.getsource(app_screen.AppScreen._on_pipeline_result)
    assert "panel.set_summary(" in source
    assert source.index("set_diagnostics(") < source.index("set_summary(")
