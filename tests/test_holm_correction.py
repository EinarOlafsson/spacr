"""spaCR's own Holm correction is statsmodels' Holm, without the collection.

statsmodels runs ``gc.collect()`` inside ``multipletests(method='holm')``.
On a pipeline worker that full collection walks the GUI heap with the
interpreter lock held (about 170 ms per call in the Invasion run), so spaCR
computes Holm itself and must agree with statsmodels to the last bit.
"""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from spacr.multiple_testing import (_holm, _multipletests, _posthoc_dunn,
                                    adjust_p_values)

ALPHAS = (0.001, 0.01, 0.05, 0.1, 0.5)


@pytest.fixture
def quiet_statsmodels(monkeypatch):
    """Skip the full collection statsmodels runs per Holm call; it changes no result."""
    monkeypatch.setattr(gc, "collect", lambda *args: 0)


def _draw(rng, size):
    """P values with ties, exact 0 and 1, values near alpha, and sometimes NaN."""
    kind = rng.integers(5)
    if kind == 0:
        values = rng.random(size)
    elif kind == 1:
        values = rng.choice(rng.random(max(1, size // 4)), size)
    elif kind == 2:
        values = rng.choice([0.0, 1.0, 0.05, 0.01, 0.5], size)
    elif kind == 3:
        values = rng.random(size) ** 6
    else:
        values = np.round(rng.random(size), 2)
    if size > 2 and rng.random() < 0.3:
        values[rng.random(size) < 0.1] = np.nan
    return values


def _same(ours, theirs):
    reject, adjusted = ours
    expected_reject, expected_adjusted = theirs[0], theirs[1]
    assert reject.dtype == expected_reject.dtype
    assert adjusted.dtype == expected_adjusted.dtype
    assert np.array_equal(reject, expected_reject)
    assert np.array_equal(adjusted, expected_adjusted, equal_nan=True)


def test_holm_matches_statsmodels_bit_for_bit_on_seeded_inputs(
        quiet_statsmodels):
    rng = np.random.default_rng(20261018)
    sizes = list(range(1, 41)) + [64, 99, 100, 128, 250, 333, 499, 500]
    for size in sizes:
        for _ in range(12):
            values = _draw(rng, size)
            for alpha in ALPHAS:
                _same(_holm(values, alpha=alpha),
                      multipletests(values, alpha=alpha, method="holm"))


def test_holm_matches_statsmodels_under_hypothesis(quiet_statsmodels):
    hypothesis = pytest.importorskip("hypothesis")
    from hypothesis import strategies as st
    from hypothesis.extra.numpy import arrays

    probability = st.one_of(
        st.floats(0.0, 1.0, allow_nan=False),
        st.sampled_from([0.0, 1.0, 0.05, np.nan]))

    @hypothesis.settings(max_examples=300, deadline=None, database=None)
    @hypothesis.given(arrays(np.float64, st.integers(1, 500),
                             elements=probability),
                      st.sampled_from(ALPHAS))
    def check(values, alpha):
        _same(_holm(values, alpha=alpha),
              multipletests(values, alpha=alpha, method="holm"))

    check()


@pytest.mark.parametrize("spelling", ["holm", "h", "Holm"])
def test_every_holm_spelling_goes_through_spacr(spelling,
                                                quiet_statsmodels):
    values = np.array([0.01, 0.04, 0.03, 0.2, 0.01])
    _same(_multipletests(values, method=spelling),
          multipletests(values, method="holm"))


def test_other_methods_still_come_from_statsmodels():
    values = np.random.default_rng(3).random(50)
    for method in ("fdr_bh", "bonferroni", "sidak", "holm-sidak"):
        _same(_multipletests(values, method=method),
              multipletests(values, method=method))


def test_the_public_holm_option_is_unchanged(quiet_statsmodels):
    values = np.array([0.001, np.nan, 0.02, 0.02, 0.9, 0.04])
    adjusted, rejected = adjust_p_values(values, method="holm")
    finite = np.isfinite(values)
    expected = multipletests(values[finite], method="holm")
    assert np.array_equal(adjusted[finite], expected[1])
    assert np.isnan(adjusted[~finite]).all()
    assert np.array_equal(rejected[finite], expected[0])
    assert not rejected[~finite].any()


def _dunn_data():
    rng = np.random.default_rng(11)
    groups = np.repeat(["a", "b", "c", "d"], 25)
    values = rng.normal(np.repeat([0.0, 0.3, 0.6, 1.5], 25), 1.0)
    return pd.DataFrame({"value": values, "group": groups})


def test_dunn_with_holm_is_identical_to_scikit_posthocs(quiet_statsmodels):
    import scikit_posthocs as sp

    data = _dunn_data()
    expected = sp.posthoc_dunn(data, val_col="value", group_col="group",
                               p_adjust="holm")
    result = _posthoc_dunn(data, val_col="value", group_col="group",
                           p_adjust="holm")
    pd.testing.assert_frame_equal(result, expected, check_exact=True)


def test_no_holm_path_runs_a_full_collection(monkeypatch):
    def refuse(*args):
        raise AssertionError("gc.collect ran during a Holm correction")

    monkeypatch.setattr(gc, "collect", refuse)
    values = np.random.default_rng(5).random(30)
    with pytest.raises(AssertionError, match="gc.collect"):
        multipletests(values, method="holm")

    _holm(values)
    _multipletests(values, method="holm")
    adjust_p_values(values, method="holm")
    _posthoc_dunn(_dunn_data(), val_col="value", group_col="group",
                  p_adjust="holm")
