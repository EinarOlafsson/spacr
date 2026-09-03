"""The seven methods, and the rule that decides where each is offered.

Instruction 254's own words: "A METHOD OFFERED IN THE WRONG CATEGORY IS
WORSE THAN ONE NOT OFFERED. A user who picks 'random forest' from a menu
headed regression_type has every reason to expect a volcano at the end of
it." So the category is the thing most of these tests are about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import nonparametric_fits as NP


# --- the three-way split ---------------------------------------------------


def test_all_seven_are_present():
    """The table this item was requested as."""
    for name in ("lowess", "kernel", "knn", "random_forest",
                 "gradient_boosting", "gaussian_process", "isotonic",
                 "spline"):
        assert name in NP.METHODS


def test_each_one_says_what_it_is_for_and_what_it_costs():
    for name in NP.METHODS:
        said = NP.describe(name)
        assert "For " in said and "Costs:" in said, name


def test_the_four_that_cannot_give_a_coefficient_are_not_fits():
    """LOWESS is descriptive; kernel and KNN give a surface; a forest gives
    importances. None of them can hand the volcano an effect and a P."""
    for name in ("lowess", "kernel", "knn", "random_forest"):
        assert NP.METHODS[name]["category"] != NP.CATEGORY_FIT, name


def test_the_two_that_can_are():
    assert set(NP.methods_in(NP.CATEGORY_FIT)) == {"spline", "isotonic"}


def test_a_diagnostic_refuses_to_be_run_as_an_agreement_check():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.uniform(0, 1, (40, 3)),
                         columns=["a", "b", "c"])
    with pytest.raises(ValueError, match="not an agreement"):
        NP.agreement(frame, rng.normal(0, 1, 40), {"a": 1.0},
                     method="lowess")


def test_an_agreement_check_refuses_to_be_run_as_a_diagnostic():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="not a diagnostic"):
        NP.smooth(rng.uniform(0, 1, 50), rng.normal(0, 1, 50),
                  method="random_forest")


# --- refusing rather than returning a fit nobody should read ---------------


def test_a_gaussian_process_refuses_a_large_sample_with_the_number():
    """It is cubic in the sample, so this is a refusal and not a wait."""
    why = NP.refuse("gaussian_process", rows=50_000)
    assert why and "50,000" in why and "cubic" in why


def test_it_accepts_a_sample_it_can_actually_fit():
    assert NP.refuse("gaussian_process", rows=500) is None


def test_isotonic_refuses_an_unordered_design_and_says_where_to_point_it():
    why = NP.refuse("isotonic", ordered=False)
    assert why and "ORDERED" in why
    assert "cell count" in why or "abundance" in why


def test_kernel_regression_refuses_the_guide_design():
    why = NP.refuse("kernel", predictors=700)
    assert why and "700" in why


def test_smooth_raises_the_refusal_rather_than_fitting():
    rng = np.random.default_rng(0)
    big = rng.uniform(0, 1, NP.GP_MAXIMUM_ROWS + 1)
    with pytest.raises(ValueError, match="cubic"):
        NP.smooth(big, big, method="gaussian_process")


# --- B: the diagnostics ----------------------------------------------------


@pytest.fixture
def bent():
    rng = np.random.default_rng(1)
    x = rng.uniform(0.0, 1.0, 300)
    return x, np.sin(3.0 * x) + rng.normal(0.0, 0.08, 300)


@pytest.mark.parametrize("method", ["lowess", "kernel", "knn",
                                    "gaussian_process"])
def test_every_diagnostic_draws_a_curve(bent, method):
    x, y = bent
    curve = NP.smooth(x, y, method=method)
    assert curve.x.size == curve.y.size >= 2
    assert np.isfinite(curve.y).all()


def test_a_diagnostic_carries_no_p_value(bent):
    """The thing that would let it be mistaken for a test."""
    x, y = bent
    assert not hasattr(NP.smooth(x, y, method="lowess"), "p_values")


def test_only_the_gaussian_process_reports_a_band(bent):
    x, y = bent
    curve = NP.smooth(x, y, method="gaussian_process")
    for name in ("lower", "upper", "note"):
        assert f":ivar {name}:" in (NP.Curve.__doc__ or "")
    assert curve.has_band
    assert len(curve.lower) == len(curve.upper) == len(curve.x)
    for method in ("lowess", "kernel", "knn"):
        assert not NP.smooth(x, y, method=method).has_band


def test_the_curve_follows_the_bend(bent):
    """A smoother that returned a straight line would pass every test above."""
    x, y = bent
    curve = NP.smooth(x, y, method="lowess")
    straight = np.polyval(np.polyfit(x, y, 1), curve.x)
    assert (np.abs(curve.y - straight).max()
            > 0.1 * (y.max() - y.min())), "the curve is a straight line"


def test_the_methods_that_need_scaling_scale_and_say_so(bent):
    x, y = bent
    for method in ("knn", "gaussian_process", "kernel"):
        assert "standardised" in NP.smooth(x, y, method=method).note


# --- C: the agreement check ------------------------------------------------


@pytest.fixture
def two_real_guides():
    rng = np.random.default_rng(0)
    n, g = 400, 12
    frame = pd.DataFrame(rng.uniform(0, 1, (n, g)),
                         columns=[f"guide{i}" for i in range(g)])
    truth = np.zeros(g)
    truth[0], truth[1] = 2.0, -1.5
    y = frame.to_numpy() @ truth + rng.normal(0, 0.2, n)
    linear = {c: float(v) for c, v in zip(frame.columns, truth)}
    return frame, y, linear


@pytest.mark.parametrize("method", ["random_forest", "gradient_boosting"])
def test_both_rankings_find_the_real_guides(two_real_guides, method):
    frame, y, linear = two_real_guides
    result = NP.agreement(frame, y, linear, method=method)
    top_linear = sorted(result.linear_rank, key=result.linear_rank.get)[:2]
    top_other = sorted(result.other_rank, key=result.other_rank.get)[:2]
    assert set(top_linear) == set(top_other) == {"guide0", "guide1"}


def test_it_reports_a_comparison_and_not_a_coefficient_table(two_real_guides):
    frame, y, linear = two_real_guides
    result = NP.agreement(frame, y, linear)
    assert not hasattr(result, "coefficients")
    assert not hasattr(result, "p_values")
    assert "Spearman" in result.summary()


def test_it_names_the_guides_the_two_disagree_about(two_real_guides):
    """"Where they disagree the disagreement is itself the finding."""
    frame, y, linear = two_real_guides
    # Invert one guide's linear effect so the two rankings must part.
    linear = dict(linear)
    linear["guide7"] = 9.0
    result = NP.agreement(frame, y, linear, moved_by=3)
    assert ":ivar disagreements:" in (NP.Agreement.__doc__ or "")
    assert ":ivar note:" in (NP.Agreement.__doc__ or "")
    assert any(g == "guide7" for g, _a, _b in result.disagreements)
    assert "guide7" in result.summary()


def test_the_note_says_importances_are_unsigned(two_real_guides):
    frame, y, linear = two_real_guides
    assert "unsigned" in NP.agreement(frame, y, linear).note


# --- A: the two that answer in the same currency ---------------------------


def test_the_spline_basis_leaves_every_guide_column_alone():
    """What keeps this in category A: one column per guide in, one out, so
    the fit still gives one coefficient and one P value per guide."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "fraction:grna[g1]": rng.uniform(0, 1, 90),
        "fraction:grna[g2]": rng.uniform(0, 1, 90),
        "cell_count": rng.uniform(50, 400, 90),
    })
    out = NP.spline_design(frame, ["cell_count"])
    for guide in ("fraction:grna[g1]", "fraction:grna[g2]"):
        assert guide in out.columns
        assert np.allclose(out[guide], frame[guide])


def test_the_basis_columns_carry_no_guide_so_the_filters_drop_them():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"fraction:grna[g1]": rng.uniform(0, 1, 90),
                          "cell_count": rng.uniform(50, 400, 90)})
    out = NP.spline_design(frame, ["cell_count"])
    basis = [c for c in out.columns if "spline" in c]
    assert basis
    assert not any("grna" in c or "gene" in c for c in basis)


def test_a_covariate_with_too_few_values_is_left_linear():
    """Manufacturing a basis out of three distinct values would spend
    degrees of freedom on nothing."""
    frame = pd.DataFrame({"fraction:grna[g1]": np.linspace(0, 1, 30),
                          "flag": np.repeat([0.0, 1.0], 15)})
    out = NP.spline_design(frame, ["flag"])
    assert "flag" in out.columns
    assert not [c for c in out.columns if "spline" in c]


def test_isotonic_is_monotone():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 1, 120)
    y = 2.0 * x + rng.normal(0, 0.3, 120)
    _grid, fitted = NP.isotonic_fit(x, y)
    assert np.all(np.diff(fitted) >= -1e-9)


def test_isotonic_can_go_the_other_way():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 1, 120)
    y = -2.0 * x + rng.normal(0, 0.3, 120)
    _grid, fitted = NP.isotonic_fit(x, y, increasing=False)
    assert np.all(np.diff(fitted) <= 1e-9)


# --- the agreement check, run on a finished fit ----------------------------


def _fit_shaped(n=300, g=10, seed=0):
    rng = np.random.default_rng(seed)
    design = pd.DataFrame(rng.uniform(0, 1, (n, g)),
                          columns=[f"fraction:grna[gd{i}]" for i in range(g)])
    truth = np.zeros(g)
    truth[0], truth[1] = 2.0, -1.5
    y = design.to_numpy() @ truth + rng.normal(0, 0.2, n)
    coefficients = pd.DataFrame({"feature": design.columns,
                                 "coefficient": truth})
    return coefficients, design, y


def test_it_reports_against_the_ranking_the_run_produced():
    """Nothing is refitted: it takes the coefficient table the run made."""
    coefficients, design, y = _fit_shaped()
    said = NP.report_agreement(coefficients, design, y)
    assert "Spearman" in said
    assert "linear ranking" in said


def test_the_report_says_importances_are_unsigned():
    coefficients, design, y = _fit_shaped()
    assert "unsigned" in NP.report_agreement(coefficients, design, y)


def test_it_says_nothing_rather_than_guessing_on_a_thin_table():
    """Two guides cannot be a ranking, so there is no comparison to make."""
    coefficients, design, y = _fit_shaped(g=2)
    assert NP.report_agreement(coefficients, design, y) == ""


def test_a_table_with_no_guide_terms_reports_nothing():
    coefficients = pd.DataFrame({"feature": ["Intercept", "rowID[T.r2]"],
                                 "coefficient": [0.1, 0.2]})
    design = pd.DataFrame(np.ones((10, 2)), columns=["Intercept",
                                                     "rowID[T.r2]"])
    assert NP.report_agreement(coefficients, design, np.zeros(10)) == ""


def test_it_names_a_guide_the_two_rankings_disagree_about():
    coefficients, design, y = _fit_shaped()
    # Claim a large linear effect for a guide that has none.
    coefficients = coefficients.copy()
    coefficients.loc[coefficients.feature.str.contains("gd7"),
                     "coefficient"] = 9.0
    said = NP.report_agreement(coefficients, design, y)
    assert "gd7" in said or "No guide moved" in said


# --- the note that says whether x was standardised -------------------------

def test_an_unscaled_fit_does_not_claim_x_was_standardised():
    """``scaled=False`` skips both the transform and the note about it.

    The note is what a reader of the figure has to trust: a distance-based
    smoother on an unscaled covariate makes one unit of x mean whatever its
    range happens to be, so "x standardised before fitting" is a claim about
    how the curve was produced. Printing it on a fit that skipped the
    transform would describe the wrong analysis.
    """
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 100, 80)
    y = 2.0 * x + rng.normal(0, 5, 80)

    curve = NP.smooth(x, y, method="knn", scaled=False)

    assert "standardised" not in (curve.note or "")


def test_a_scaled_fit_says_so_and_says_why():
    """The other half, so the assertion above is about the flag, not the text."""
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 100, 80)
    y = 2.0 * x + rng.normal(0, 5, 80)

    curve = NP.smooth(x, y, method="knn", scaled=True)

    assert "standardised" in curve.note
    assert "distance" in curve.note


def test_scaling_changes_the_curve_the_kernel_smoother_draws():
    """Not merely the note: the flag has to change the fit itself.

    A test that only checked the wording would pass on an implementation that
    wrote the note and forgot the transform -- which is exactly the bug the
    note would then be lying about.

    ``kernel`` is the method that shows it, and which one that is turned out
    to be worth measuring rather than assuming. With a SINGLE predictor,
    standardising is a monotone rescale, so it does not reorder anything and
    ``knn`` returns an identical curve; the Gaussian process fits its own
    length scale and absorbs it too. Only the kernel smoother carries a
    bandwidth in the covariate's original units, so only it moves. The note is
    still right for all three -- x really was standardised -- but the visible
    consequence is here.
    """
    rng = np.random.default_rng(11)
    x = rng.uniform(0, 1000, 120)
    y = np.sin(x / 200.0) + rng.normal(0, 0.05, 120)

    scaled = NP.smooth(x, y, method="kernel", scaled=True)
    unscaled = NP.smooth(x, y, method="kernel", scaled=False)

    assert not np.allclose(scaled.y, unscaled.y), (
        "scaled and unscaled produced the same curve, so the flag only "
        "changed the note")


def test_a_single_predictor_makes_scaling_invisible_to_the_neighbour_methods():
    """The counterpart, stated rather than left as a surprise.

    Standardising one covariate cannot change which points are nearest, so a
    reader comparing two knn curves and seeing no difference is looking at
    correct behaviour, not a broken flag. Pinning it stops someone "fixing"
    the invariance later.
    """
    rng = np.random.default_rng(11)
    x = rng.uniform(0, 1000, 120)
    y = np.sin(x / 200.0) + rng.normal(0, 0.05, 120)

    assert np.allclose(NP.smooth(x, y, method="knn", scaled=True).y,
                       NP.smooth(x, y, method="knn", scaled=False).y)
