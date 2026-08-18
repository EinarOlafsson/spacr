"""Group lasso drops a gene's guides as a set, never one at a time.

Instruction 133, asked for on 2026-08-17. The reason it is here rather than
plain lasso is the first test below: from a gene's four correlated guides,
lasso keeps whichever one fits best and drops the rest, which reads as "one
guide works and three do not" when the truth is "the gene matters and these
are four measurements of it".
"""

import numpy as np
import pytest

from spacr import group_lasso as gl


def _screen(n=200, n_genes=30, k=4, seed=0, correlated=False):
    """A screen with genes ``g00`` and ``g01`` planted and the rest null."""
    rng = np.random.default_rng(seed)
    labels = np.repeat([f"g{i:02d}" for i in range(n_genes)], k)
    if correlated:
        # Four guides of a gene are four measurements of the same knockout,
        # so their columns are correlated -- which is precisely the case
        # plain lasso handles badly.
        latent = rng.normal(size=(n, n_genes))
        X = np.repeat(latent, k, axis=1) + rng.normal(
            scale=0.05, size=(n, n_genes * k))
    else:
        X = rng.normal(size=(n, n_genes * k))
    beta = np.zeros(n_genes * k)
    beta[0:k] = 1.5
    beta[k:2 * k] = -1.2
    y = X @ beta + rng.normal(scale=0.5, size=n) + 3.0
    return X, y, labels


def test_a_planted_genes_guides_are_all_kept_or_all_dropped():
    """The property the whole method exists for."""
    X, y, labels = _screen(correlated=True)
    beta, _intercept, converged = gl.fit(X, y, labels, lam=0.10)
    assert converged
    for start in range(0, beta.size, 4):
        block = beta[start:start + 4]
        assert np.all(block == 0) or np.all(block != 0)


def test_only_the_planted_genes_survive():
    X, y, labels = _screen()
    out = gl.gene_effects(X, y, labels, lam=0.10)
    assert set(out.loc[out["selected"], "gene"]) == {"g00", "g01"}
    assert out.iloc[0]["gene"] == "g00"


def _split_genes(beta, k=4):
    """Genes with SOME but not all of their guides kept."""
    return sum(0 < np.count_nonzero(beta[i:i + k]) < k
               for i in range(0, beta.size, k))


def test_lasso_splits_a_gene_where_group_lasso_cannot():
    """The comparison, run rather than asserted from theory.

    Lasso here is this same solver with each guide its own group, which IS
    the plain L1 problem. On four near-identical guides it keeps an
    arbitrary subset -- and a reader of that result concludes one guide works
    and the others do not, which is a claim about guide quality the data does
    not support.
    """
    X, y, labels = _screen(correlated=True)
    alone = np.array([f"guide{i}" for i in range(X.shape[1])], dtype=object)
    grouped, _i, _c = gl.fit(X, y, labels, lam=0.10, max_iterations=3000)
    singly, _i, _c = gl.fit(X, y, alone, lam=0.10, max_iterations=3000)
    assert _split_genes(grouped) == 0
    assert _split_genes(singly) >= 1


def test_no_penalty_is_ordinary_least_squares():
    """The solver checked against an answer that has a closed form."""
    X, y, labels = _screen(n=200, n_genes=6)
    beta, intercept, converged = gl.fit(X, y, labels, lam=0.0,
                                        max_iterations=5000, tolerance=1e-12)
    centred = X - X.mean(axis=0)
    ols = np.linalg.lstsq(centred, y - y.mean(), rcond=None)[0]
    assert converged
    assert np.allclose(beta, ols, atol=1e-6)
    assert intercept == pytest.approx(float(y.mean() - X.mean(axis=0) @ beta))


def test_the_intercept_is_never_penalised():
    """Shrinking the response's mean toward zero is not a claim anybody wants."""
    X, y, labels = _screen()
    _beta, intercept, _converged = gl.fit(X, y, labels, lam=0.5)
    assert intercept == pytest.approx(3.0, abs=0.2)


def test_the_intercept_can_be_declined():
    X, y, labels = _screen()
    _beta, intercept, _converged = gl.fit(X, y, labels, lam=0.1,
                                          fit_intercept=False)
    assert intercept == 0.0


def test_max_lambda_is_the_smallest_penalty_that_zeroes_everything():
    X, y, labels = _screen()
    top = gl.max_lambda(X, y, labels)
    at_top, _i, _c = gl.fit(X, y, labels, lam=top * 1.001)
    below, _i, _c = gl.fit(X, y, labels, lam=top * 0.5)
    assert not np.any(at_top)
    assert np.any(below)


def test_max_lambda_of_nothing_is_zero():
    assert gl.max_lambda(np.zeros((4, 0)), np.zeros(4), []) == 0.0


def test_a_dead_column_block_is_skipped_not_divided_by():
    """A gene whose guides are all-zero columns has no gradient to step along.

    Real screens have them: a guide with zero counts in every well. The step
    size is 1/L and L is zero, so the block is skipped rather than producing
    an infinite step.
    """
    X, y, labels = _screen(n_genes=4)
    X[:, 4:8] = 0.0                      # every guide of g01, dead
    beta, _intercept, converged = gl.fit(X, y, labels, lam=0.05)
    assert converged
    assert np.all(beta[4:8] == 0.0)
    assert np.isfinite(beta).all()


def test_a_run_that_does_not_converge_says_so():
    """False, not a silent partial answer that reads like a fit."""
    X, y, labels = _screen()
    _beta, _intercept, converged = gl.fit(X, y, labels, lam=0.01,
                                          max_iterations=1)
    assert converged is False


def test_the_block_shrinks_to_exactly_zero_not_nearly():
    """Exactly. A gene at 1e-18 is still "selected" to any `!= 0` check."""
    assert np.all(gl._soft_threshold_block(np.array([0.3, 0.4]), 0.5) == 0.0)
    assert np.all(gl._soft_threshold_block(np.zeros(3), 0.1) == 0.0)
    shrunk = gl._soft_threshold_block(np.array([3.0, 4.0]), 1.0)
    assert np.linalg.norm(shrunk) == pytest.approx(4.0)


@pytest.mark.parametrize("args, message", [
    ((np.zeros(4), np.zeros(4), ["a"] * 4), "must be two-dimensional"),
    ((np.zeros((4, 2)), np.zeros(3), ["a", "a"]), "4 row.* and y has 3"),
    ((np.zeros((4, 2)), np.zeros(4), ["a"]), "labels has 1 entr"),
])
def test_shapes_that_disagree_say_both_numbers(args, message):
    with pytest.raises(ValueError, match=message):
        gl.fit(*args)


def test_a_negative_penalty_is_refused():
    with pytest.raises(ValueError, match="must not be negative"):
        gl.fit(np.zeros((4, 2)), np.zeros(4), ["a", "a"], lam=-1.0)


def test_a_genes_effect_is_the_norm_of_its_block_and_never_negative():
    """Group lasso says the guides move together, not which way each one does."""
    X, y, labels = _screen()
    out = gl.gene_effects(X, y, labels, lam=0.10)
    assert (out["effect"] >= 0).all()
    # g01's true coefficients are all NEGATIVE and its effect is still positive.
    assert out.loc[out["gene"] == "g01", "effect"].iloc[0] > 1.0
    assert list(out["effect"]) == sorted(out["effect"], reverse=True)


def test_stability_selection_separates_the_planted_from_the_noise():
    X, y, labels = _screen()
    out = gl.stability_selection(X, y, labels, lam=0.10, n_boot=25, seed=1)
    planted = out.loc[out["gene"].isin(["g00", "g01"]), "selection_frequency"]
    rest = out.loc[~out["gene"].isin(["g00", "g01"]), "selection_frequency"]
    assert (planted >= 0.9).all()
    assert rest.max() < 0.5


def test_stability_selection_is_reproducible():
    X, y, labels = _screen(n_genes=8)
    a = gl.stability_selection(X, y, labels, n_boot=10, seed=4)
    b = gl.stability_selection(X, y, labels, n_boot=10, seed=4)
    assert list(a["selection_frequency"]) == list(b["selection_frequency"])


@pytest.mark.parametrize("kwargs, message", [
    (dict(fraction=0.0), "fraction must be in"),
    (dict(fraction=1.5), "fraction must be in"),
    (dict(n_boot=0), "n_boot must be at least 1"),
])
def test_a_stability_setting_outside_its_range_says_so(kwargs, message):
    X, y, labels = _screen(n=20, n_genes=2)
    with pytest.raises(ValueError, match=message):
        gl.stability_selection(X, y, labels, **kwargs)


def test_the_description_carries_the_formula_and_the_honest_caveat():
    text = gl.describe(0.05)
    assert "sum_g sqrt(p_g)" in text
    assert "Recommended for CRISPR screens" in text
    assert "no P value" in text
    assert "0.05" in text


def test_the_gene_never_becomes_a_column_of_its_own():
    """The reason this backend cannot be singular the way the mixed design is.

    `gene_fraction` is the SUM of a gene's guide fractions, so a design
    carrying both blocks is rank deficient by construction. Here the gene is
    a GROUPING of the guide columns: the design handed to the solver has
    exactly one column per guide and none per gene.
    """
    X, y, labels = _screen(n_genes=5, k=4)
    assert X.shape[1] == 20
    assert len(labels) == X.shape[1]
    beta, _intercept, _converged = gl.fit(X, y, labels, lam=0.1)
    assert beta.size == X.shape[1] == 20
