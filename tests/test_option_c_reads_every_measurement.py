"""Instruction 173, option C: attribute a cell from ALL its measurements.

    "best case i can use all the fraction information and all the measurement
    and classefication data to estimate which grna is linked to which cell
    ... eaven if it only holds a timy little bit of information it still
    might work, right?"

Right in principle, and two things decide whether the "might" is honest.

LOG SPACE. A product of 785 densities underflows to exactly zero in double
precision long before it reaches the end, and every cell then looks equally
impossible -- which the solver answers by handing back the prior. The bug
would present as "option C always says ambiguous", which reads as a property
of the data rather than of the arithmetic.

THE MEASUREMENTS ARE NOT INDEPENDENT. `cell_area` and `cell_perimeter` are
one measurement wearing two names, and multiplying their likelihoods counts
the same evidence twice. Without the design-effect correction the posterior
saturates at 0 or 1 for every cell and the 0.55 threshold is decorative.
"""
import numpy as np
import pytest

from spacr.guide_attribution import (effective_dimension, posterior,
                                     posterior_multivariate)


# --------------------------------------------------- the effective dimension


def test_independent_columns_count_themselves():
    rng = np.random.default_rng(0)
    n_eff = effective_dimension(rng.normal(size=(4000, 5)))
    assert 4.5 < n_eff <= 5.0


def test_copies_of_one_column_count_once():
    """THE WHOLE POINT: 785 columns of the same thing are one measurement."""
    rng = np.random.default_rng(1)
    one = rng.normal(size=(500, 1))
    assert effective_dimension(np.repeat(one, 12, axis=1)) == pytest.approx(1.0)


def test_a_constant_column_is_dropped_not_counted():
    """It carries no information; counted, it would look independent."""
    rng = np.random.default_rng(2)
    varying = rng.normal(size=(400, 3))
    with_dead = np.column_stack([varying, np.ones(400)])
    assert effective_dimension(with_dead) == pytest.approx(
        effective_dimension(varying), abs=0.05)


def test_an_all_constant_matrix_is_zero_not_an_error():
    assert effective_dimension(np.ones((10, 4))) == 0.0


def test_one_column_is_one():
    rng = np.random.default_rng(3)
    assert effective_dimension(rng.normal(size=(100, 1))) == 1.0


def test_an_empty_matrix_is_not_an_error():
    assert effective_dimension(np.zeros((0, 0))) == 0.0


# ------------------------------------------------------------- the posterior


def _screen(n_cells=300, n_measures=8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_cells, n_measures))


def test_every_cell_carries_exactly_one_guide():
    r, _guides, _report = posterior_multivariate(
        _screen(), {"a": 0.3, "b": 0.7},
        {"a": [1.0] * 8, "b": [-1.0] * 8})
    assert np.allclose(r.sum(axis=1), 1.0)


def test_the_guide_masses_match_the_sequencing():
    """CALIBRATION, which 173 names: sum_i r_ig / N == pi_g by construction,
    so this proves the solver converged rather than that the model is right."""
    priors = {"a": 0.2, "b": 0.5, "c": 0.3}
    n = 400
    r, guides, _report = posterior_multivariate(
        _screen(n_cells=n), priors,
        {"a": [1.0] * 8, "b": [0.0] * 8, "c": [-1.0] * 8})
    for index, guide in enumerate(guides):
        assert r[:, index].sum() / n == pytest.approx(priors[guide], abs=1e-6)


def test_many_measurements_do_not_underflow_to_the_prior():
    """785 densities multiplied is exp(-4000). In log space it is a number.

    Without the shift every cell comes back at its prior, which reads as
    "the data says nothing" when it means "the arithmetic overflowed".
    """
    rng = np.random.default_rng(4)
    n_measures = 800
    values = rng.normal(size=(120, n_measures))
    # A real difference, spread thinly over all 800 columns.
    values[:60] += 0.25
    priors = {"a": 0.5, "b": 0.5}
    effects = {"a": [0.25] * n_measures, "b": [-0.25] * n_measures}

    r, guides, report = posterior_multivariate(values, priors, effects)

    assert np.all(np.isfinite(r))
    assert report["n_measurements"] == n_measures
    # Not every cell sitting at its prior -- that is the underflow signature.
    assert r[:, 0].std() > 1e-6, "every posterior collapsed to the prior"


def test_correlated_measurements_do_not_count_twice():
    """Twelve copies of one column must not be twelve times the evidence."""
    rng = np.random.default_rng(5)
    one = rng.normal(size=(300, 1))
    duplicated = np.repeat(one, 12, axis=1)
    priors = {"a": 0.5, "b": 0.5}
    effects = {"a": [0.8] * 12, "b": [-0.8] * 12}

    corrected, _g, report = posterior_multivariate(duplicated, priors, effects)
    naive, _g2, _r2 = posterior_multivariate(
        duplicated, priors, effects, correct_for_correlation=False)

    assert report["effective_dimension"] == pytest.approx(1.0)
    assert report["scale_factor"] == pytest.approx(1.0 / 12.0)
    # The naive version is more confident about every cell it is confident
    # about -- that extra confidence is the double counting.
    assert naive.max(axis=1).mean() > corrected.max(axis=1).mean()


def test_the_correction_is_reported_not_silent():
    """A reader has to see how much of their evidence was discounted."""
    _r, _g, report = posterior_multivariate(
        _screen(), {"a": 0.5, "b": 0.5}, {"a": [1.0] * 8, "b": [-1.0] * 8})
    assert set(report) == {"n_measurements", "effective_dimension",
                           "scale_factor"}
    assert 0.0 < report["scale_factor"] <= 1.0


def test_two_guides_with_identical_effects_come_back_at_their_priors():
    """173's own bar. "The data cannot separate them and the method must not
    pretend otherwise." """
    priors = {"a": 0.25, "b": 0.75}
    r, guides, _report = posterior_multivariate(
        _screen(seed=6), priors, {"a": [0.5] * 8, "b": [0.5] * 8})
    for index, guide in enumerate(guides):
        assert np.allclose(r[:, index], priors[guide], atol=1e-6)


def test_a_permutation_collapses_the_posterior_to_the_prior():
    """173: "Structure surviving a permutation is structure the method
    invented." With the effects shuffled away from the data, nothing in a
    cell's measurements favours one guide."""
    priors = {"a": 0.4, "b": 0.6}
    r, guides, _report = posterior_multivariate(
        _screen(seed=7), priors, {"a": [0.0] * 8, "b": [0.0] * 8})
    for index, guide in enumerate(guides):
        assert np.allclose(r[:, index], priors[guide], atol=1e-6)


def test_a_guide_with_no_effects_is_flat_not_an_error():
    """The honest prior for a guide nothing was fitted for."""
    r, guides, _report = posterior_multivariate(
        _screen(seed=8), {"a": 0.5, "b": 0.5}, {"a": [1.0] * 8})
    assert np.all(np.isfinite(r))
    assert np.allclose(r.sum(axis=1), 1.0)


def test_a_missing_measurement_contributes_nothing_not_a_zero():
    """A NaN is not the value 0 -- on a standardised column that is the mean,
    and on a raw one it is usually extreme."""
    values = _screen(seed=9)
    holed = values.copy()
    holed[:, 3] = np.nan
    priors = {"a": 0.5, "b": 0.5}
    effects = {"a": [1.0] * 8, "b": [-1.0] * 8}

    r, _g, _report = posterior_multivariate(holed, priors, effects)
    dropped, _g2, _r2 = posterior_multivariate(
        np.delete(values, 3, axis=1), priors,
        {"a": [1.0] * 7, "b": [-1.0] * 7})

    assert np.all(np.isfinite(r))
    assert np.allclose(r.sum(axis=1), 1.0)
    assert np.corrcoef(r[:, 0], dropped[:, 0])[0, 1] > 0.9


def test_one_measurement_agrees_with_option_a():
    """Option C on a single column IS option A -- same densities, same IPF.

    If they disagreed, one of them would be wrong and there would be no way
    to tell which.
    """
    rng = np.random.default_rng(10)
    scores = rng.normal(size=240)
    priors = {"a": 0.35, "b": 0.65}

    single, guides_a = posterior(scores, priors, {"a": 0.7, "b": -0.7},
                                 centre=0.0, scale=1.0)
    multi, guides_c, _report = posterior_multivariate(
        scores[:, None], priors, {"a": [0.7], "b": [-0.7]},
        centres=[0.0], scales=[1.0])

    assert guides_a == guides_c
    assert np.allclose(single, multi, atol=1e-8)


def test_no_cells_is_not_an_error():
    r, guides, report = posterior_multivariate(
        np.zeros((0, 4)), {"a": 1.0}, {"a": [0.0] * 4})
    assert r.shape == (0, 1)
    assert guides == ("a",)
    assert report["scale_factor"] == 1.0


# ------------------------------------------------------ reachable as a picker


def test_it_is_one_of_the_picking_modes():
    """"this should be one of the options the user can choose to pick cells"."""
    from spacr.cell_montage import PICKING_MODES

    assert "multivariate" in PICKING_MODES


def test_the_settings_window_offers_it():
    from spacr.picture_settings import offered_values

    offered = {value for value, _label in offered_values("cell_picking")}
    assert offered == {
        "rank", "attributed", "assigned", "multivariate", "sudoku",
    }


def test_the_label_says_it_needs_a_sweep():
    """Option C reads one effect per MEASUREMENT per guide, which only the
    gene x measurement sweep produces. A picker that silently did nothing
    would look like a broken montage."""
    from spacr.picture_settings import offered_values

    labels = dict(offered_values("cell_picking"))
    assert "sweep" in labels["multivariate"].lower()


def test_the_sudoku_label_explains_its_cross_well_evidence():
    from spacr.picture_settings import offered_values

    label = dict(offered_values("cell_picking"))["sudoku"].lower()
    assert "every well" in label
