"""The volcano's horizontal axis says which quantity it is drawing.

The permutation path copies its `standardized_marginal_effect` into
`coefficient` so the rest of the screen can read one name. That left the
axis calling a PARTIAL CORRELATION a coefficient, and pyqtgraph factored a
common power of ten out of the ticks -- so a quantity running -0.06 to 0.50
was drawn as an axis reading -100 to 400 titled "coefficient (x0.001)".

The maintainer, reading it: "the coefficients column in the results is on
the scale of -1 to 1 ... i dont get why it says -100 to 400 that is
confusing. are these actually coefficients?" They are not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.fast_plots import VolcanoPlot                # noqa: E402
from spacr.qt.widgets.regression_results import (                  # noqa: E402
    RegressionResultsPanel)


def _fitted(n=12):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "feature": [f"fraction:grna[g{i}]" for i in range(n)],
        "grna": [f"g{i}" for i in range(n)],
        "level": ["grna"] * n,
        "coefficient": rng.normal(0, 4.0, n),      # unbounded, a real slope
        "p_value": rng.uniform(1e-6, 0.9, n),
        "q_value": rng.uniform(1e-4, 0.9, n),
    })


def _permuted(n=12):
    frame = _fitted(n)
    frame["coefficient"] = np.linspace(-0.06, 0.5, n)   # a partial correlation
    frame["standardized_marginal_effect"] = frame["coefficient"]
    frame["permutation_p_value"] = frame["p_value"]
    frame["permutation_exceedances"] = 0
    frame["permutations"] = 200000
    return frame


def _panel(qtbot, frame, tmp_path, name="results.csv"):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    panel.load(str(path))
    return panel


def test_a_permutation_axis_names_the_partial_correlation(qtbot, tmp_path):
    panel = _panel(qtbot, _permuted(), tmp_path)
    assert panel._analysis_path() == "permutation"
    label = panel.volcano.plot.getAxis("bottom").labelText
    assert "partial correlation" in label
    assert "coefficient" not in label


def test_a_fitted_axis_still_says_coefficient(qtbot, tmp_path):
    panel = _panel(qtbot, _fitted(), tmp_path)
    assert panel._analysis_path() == "fitted"
    assert panel.volcano.plot.getAxis("bottom").labelText == "coefficient"


def test_the_table_decides_not_the_settings(qtbot, tmp_path):
    """A saved settings file carries the MODULE's default inference, not what
    the run did. An OLS folder whose settings say 'nonparametric' was
    labelled a partial correlation until the columns were made to decide."""
    panel = _panel(qtbot, _fitted(), tmp_path)
    panel._run_settings = {"inference": "nonparametric",
                           "analysis_mode": "guide_permutation"}
    assert panel._analysis_path() == "fitted"


def test_no_si_prefix_multiplies_the_effect_axis():
    """-0.06 to 0.50 must read as itself, not as -60 to 500 with a note."""
    plot = VolcanoPlot()
    assert plot.plot.getAxis("bottom").autoSIPrefix is False


def test_the_axis_shows_the_data_in_its_own_units(qtbot, tmp_path):
    panel = _panel(qtbot, _permuted(), tmp_path)
    low, high = panel.volcano.plot.getViewBox().viewRange()[0]
    assert -2.0 <= low <= 0.0 and 0.0 <= high <= 2.0, (
        f"a partial correlation drawn on a {low:.1f}..{high:.1f} axis")


def test_naming_an_unknown_path_falls_back_to_fitted():
    plot = VolcanoPlot()
    assert plot.name_the_effect("something else") == "coefficient"


def test_both_labels_are_declared_once():
    assert set(VolcanoPlot.EFFECT_LABELS) == {"fitted", "permutation"}


def test_the_table_does_not_show_the_same_number_twice(qtbot, tmp_path):
    """`coefficient` IS `standardized_marginal_effect` on this path -- one
    column copied so the rest of the screen can read one name. Showing both
    is what makes a reader ask which is the real one."""
    from spacr.qt.widgets.regression_results import for_table

    narrowed = for_table(_permuted())
    assert "standardized_marginal_effect" in narrowed.columns
    assert "coefficient" not in narrowed.columns


def test_a_fitted_table_keeps_its_coefficient(qtbot, tmp_path):
    """Nothing to deduplicate: a fit has no marginal-effect column at all."""
    from spacr.qt.widgets.regression_results import for_table

    assert "coefficient" in for_table(_fitted()).columns


def test_two_columns_that_merely_both_exist_are_both_kept():
    """Only an EXACT duplicate goes. A permutation whose two columns differ
    is a real disagreement and must stay visible."""
    from spacr.qt.widgets.regression_results import for_table

    frame = _permuted()
    frame.loc[0, "coefficient"] = frame.loc[0, "coefficient"] + 0.25
    narrowed = for_table(frame)
    assert "coefficient" in narrowed.columns
    assert "standardized_marginal_effect" in narrowed.columns
