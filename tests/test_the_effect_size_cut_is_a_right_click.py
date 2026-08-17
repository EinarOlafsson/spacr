"""The effect-size cut: how it is measured and how wide, from the plot.

Asked for 2026-08-17: "I sometimes have trouble seeing the coefficient
significance gate, sometimes the values are grayed out. I want the value to
be available when i right click on the graph ... the coefficent threshold
multiplyer, coefficient threshold mode (none, var, std, also add several
other methods that make sense at least 4 more)".

WHY THEY GREY OUT, since that was the complaint and it is not a bug:
`threshold_method` and `threshold_multiplier` carry a dependency rule whose
sources are ('inference', 'analysis_mode'), so they grey under
`inference='nonparametric'` / `analysis_mode='guide_permutation'`. That is
CORRECT -- the permutation path tests each guide as a marginal association
and uses no control-spread cut at all -- but it made the controls
unfindable. They are on the plot now, where the cut is drawn.

Seven methods, up from two.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from spacr.thresholds import (DIMENSIONALLY_ODD, METHODS,  # noqa: E402
                              canonical, coefficient_threshold, describe)


def _controls(n=24, sd=0.28, seed=0):
    return np.random.default_rng(seed).normal(0, sd, n)


# --------------------------------------------------------------------------- #
#  The methods
# --------------------------------------------------------------------------- #

def test_there_are_at_least_six_methods():
    """"none, var, std, also add several other methods ... at least 4 more"."""
    assert len(METHODS) >= 6, list(METHODS)
    for required in ("none", "std", "var"):
        assert required in METHODS


@pytest.mark.parametrize("method", sorted(METHODS))
def test_every_method_produces_a_cut_and_a_sentence(method):
    value, rule = coefficient_threshold(_controls(), method, 3.0)

    assert rule
    if method == "none":
        assert value is None
    else:
        assert value is not None and value > 0


def test_a_wider_multiplier_is_a_wider_cut():
    narrow, _ = coefficient_threshold(_controls(), "std", 2.0)
    wide, _ = coefficient_threshold(_controls(), "std", 4.0)

    assert wide > narrow


def test_the_robust_methods_are_not_moved_by_one_outlier():
    """`000000_22` is a non-targeting control and the strongest effect in the
    real screen at +4.37. A cut that moves when one control has a phenotype
    is a cut that screen can shift by itself."""
    clean = _controls()
    spiked = np.append(clean.copy(), 4.37)

    robust_before, _ = coefficient_threshold(clean, "mad", 3.0)
    robust_after, _ = coefficient_threshold(spiked, "mad", 3.0)
    std_before, _ = coefficient_threshold(clean, "std", 3.0)
    std_after, _ = coefficient_threshold(spiked, "std", 3.0)

    assert abs(robust_after - robust_before) < abs(std_after - std_before)


def test_the_centre_is_the_median_not_the_mean():
    """Same reason: one control with a real phenotype must not move the
    centre for every guide."""
    import inspect

    from spacr import thresholds

    source = inspect.getsource(thresholds.coefficient_threshold)
    assert "np.median(array)" in source


# --------------------------------------------------------------------------- #
#  var, which the maintainer asked for by name and which is odd
# --------------------------------------------------------------------------- #

def test_var_is_flagged_as_dimensionally_odd():
    """k x variance is not k spreads from the centre -- it adds a squared
    quantity to a coefficient. Kept because it is what spaCR shipped and what
    was asked for, but it must not read as interchangeable with std."""
    assert "var" in DIMENSIONALLY_ODD
    assert "not k spreads" in describe("var")


def test_var_and_std_disagree_in_the_direction_the_note_says():
    """Below a spread of 1 the variance is NARROWER; that is the trap."""
    controls = _controls(sd=0.28)          # spread well below 1
    var_cut, _ = coefficient_threshold(controls, "var", 3.0)
    std_cut, _ = coefficient_threshold(controls, "std", 3.0)

    assert var_cut < std_cut


# --------------------------------------------------------------------------- #
#  Refusing rather than guessing
# --------------------------------------------------------------------------- #

def test_too_few_controls_gives_no_cut_and_a_reason():
    """Never a silent 0, which would call every coefficient a hit."""
    value, rule = coefficient_threshold([0.1], "std", 3.0)

    assert value is None
    assert "not enough" in rule


def test_constant_controls_give_no_cut():
    value, rule = coefficient_threshold([0.2] * 10, "std", 3.0)

    assert value is None
    assert "no std spread" in rule


def test_an_unknown_method_names_them_all():
    with pytest.raises(ValueError) as caught:
        canonical("banana")

    for method in METHODS:
        assert method in str(caught.value)


def test_the_old_spellings_still_load():
    """A settings CSV carries spaCR's own historical misspelling."""
    assert canonical("standard_deveation") == "std"
    assert canonical("variance") == "var"


# --------------------------------------------------------------------------- #
#  On the plot
# --------------------------------------------------------------------------- #

@pytest.mark.qt
def test_the_menu_offers_every_method_and_the_multiplier(qtbot):
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    import pandas as pd

    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    rng = np.random.default_rng(0)
    n = 600
    frame = pd.DataFrame({
        "feature": [f"fraction:grna[{i}_1]" for i in range(n)],
        "coefficient": rng.normal(0, .5, n),
        "p_value": rng.uniform(size=n),
        "condition": list(rng.choice(["nc", "pc", "other"], n,
                                     p=[.04, .01, .95]))})
    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    panel.set_frame(frame)

    text = " ".join(a.text() for a in panel.volcano.build_style_menu().actions())
    assert "Multiplier" in text
    for method in METHODS:
        assert method in text, method


@pytest.mark.qt
def test_choosing_one_says_the_number_and_the_rule(qtbot):
    """A threshold a reader cannot attribute is one they cannot report."""
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    import pandas as pd

    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    rng = np.random.default_rng(0)
    n = 400
    frame = pd.DataFrame({
        "feature": [f"fraction:grna[{i}_1]" for i in range(n)],
        "coefficient": rng.normal(0, .5, n),
        "p_value": rng.uniform(size=n),
        "condition": list(rng.choice(["nc", "other"], n, p=[.06, .94]))})
    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    panel.set_frame(frame)
    panel.set_threshold_method("std")

    said = panel.status_text()
    assert "Effect-size cut" in said
    assert "std of" in said
    assert "controls" in said
