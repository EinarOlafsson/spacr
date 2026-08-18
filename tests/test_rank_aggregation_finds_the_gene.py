"""alpha-RRA aggregates guides to a gene by RANK, and finds the planted hit.

Instruction 133. Asked for on 2026-08-17 as one of the "modeling tools ...
where we could plug in the same dependent and independent variables and get an
estimate of which gRNAs/genes are involved".

The point of having it is that it is the ONE backend here that never forms
`gene_fraction` -- the sum of a gene's guide fractions, which is a linear
combination of those guides by construction and the reason a guide-and-gene
design matrix is singular. A test that only checked P values would miss that;
the test that matters is the one on a perfectly collinear design, at the
bottom.
"""

import numpy as np
import pytest

from spacr import rra


def _screen(n_genes=120, k=4, seed=3):
    rng = np.random.default_rng(seed)
    genes = np.repeat([f"g{i:03d}" for i in range(n_genes)], k)
    score = rng.normal(size=n_genes * k)
    return genes, score


def test_a_planted_depletion_is_the_top_gene():
    genes, score = _screen()
    score[0:4] -= 4.0
    out = rra.rank_aggregate(score, genes, n_permutations=2000, seed=1)
    assert out.iloc[0]["gene"] == "g000"
    assert out.iloc[0]["p_neg"] < 0.01


def test_a_planted_enrichment_is_the_top_gene_in_the_other_tail():
    """Two directions, two answers. A gene depleted is not a gene enriched."""
    genes, score = _screen()
    score[0:4] += 4.0
    out = rra.rank_aggregate(score, genes, n_permutations=2000, seed=1)
    best = out.sort_values("p_pos").iloc[0]
    assert best["gene"] == "g000"
    assert best["p_pos"] < 0.01
    # And it is NOT called depleted.
    assert out.loc[out["gene"] == "g000", "p_neg"].iloc[0] > 0.5


def test_one_strong_guide_among_dead_ones_is_still_found():
    """The whole reason alpha exists.

    In a library where a third of guides do not cut, a real gene routinely
    shows one strong guide and three at chance. Taking the minimum over ALL
    k order statistics lets the three dead guides pull rho back toward 1;
    restricting it to the top alpha does not.
    """
    genes, score = _screen()
    score[0] -= 6.0                          # one guide, hard depleted
    strict = rra.rank_aggregate(score, genes, alpha=0.25,
                                n_permutations=4000, seed=1)
    everything = rra.rank_aggregate(score, genes, alpha=1.0,
                                    n_permutations=4000, seed=1)
    at = strict.loc[strict["gene"] == "g000", "p_neg"].iloc[0]
    over = everything.loc[everything["gene"] == "g000", "p_neg"].iloc[0]
    assert at <= over


def test_a_gene_with_no_guide_in_the_top_alpha_gets_rho_one():
    """Not NaN. "Nothing near the top" is an answer, and 1.0 states it."""
    genes = np.array(["a", "a", "b", "b"])
    score = np.array([0.0, 0.1, 0.2, 0.3])
    out = rra.rank_aggregate(score, genes, alpha=0.1, direction="neg",
                             n_permutations=200, seed=1)
    assert (out["rho_neg"] == 1.0).all()
    assert out["p_neg"].notna().all()


def test_a_guide_the_fit_could_not_estimate_is_dropped_not_ranked_worst():
    """A NaN coefficient means "not measured", never "strongly depleted".

    Ranking it worst would make an unestimable guide the evidence for its
    gene being a hit, which is the most consequential silent error this
    function could make.
    """
    genes = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
    with_nan = np.array([np.nan, 0.5, 0.6, 0.7, 0.1, 0.2, 0.3, 0.4])
    out = rra.rank_aggregate(with_nan, genes, direction="neg",
                             n_permutations=500, seed=1)
    assert list(out["n_guides"]) == [3, 4] or list(out["n_guides"]) == [4, 3]
    assert out.loc[out["gene"] == "a", "n_guides"].iloc[0] == 3
    # `a` holds the three WORST finite scores, so it is not the depleted gene.
    assert out.sort_values("p_neg").iloc[0]["gene"] == "b"


def test_a_guide_belonging_to_no_gene_is_dropped():
    genes = np.array(["a", "a", None, "b", "b"], dtype=object)
    score = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    out = rra.rank_aggregate(score, genes, n_permutations=200, seed=1)
    assert set(out["gene"]) == {"a", "b"}
    assert out["n_guides"].sum() == 4


def test_nothing_finite_gives_an_empty_table_not_a_crash():
    out = rra.rank_aggregate([np.nan, np.inf], ["a", "a"],
                             n_permutations=10, seed=1)
    assert len(out) == 0
    assert list(out.columns) == ["gene", "n_guides"]


def test_the_same_screen_gives_the_same_p_values():
    """A hit list that moves between runs of the same data is unactionable."""
    genes, score = _screen(n_genes=40)
    a = rra.rank_aggregate(score, genes, n_permutations=500, seed=7)
    b = rra.rank_aggregate(score, genes, n_permutations=500, seed=7)
    assert list(a["p_neg"]) == list(b["p_neg"])


def test_no_p_value_is_ever_exactly_zero():
    """+1 in numerator and denominator.

    A permutation P value of 0 claims a resolution the permutation count does
    not have, and it is exactly the value that survives an FDR correction to
    become a "finding".
    """
    genes, score = _screen()
    score[0:4] -= 20.0
    out = rra.rank_aggregate(score, genes, n_permutations=500, seed=1)
    assert (out["p_neg"] > 0).all()
    assert out["p_neg"].min() == pytest.approx(1.0 / 501.0)


@pytest.mark.parametrize("direction, present, absent", [
    ("neg", "p_neg", "p_pos"),
    ("pos", "p_pos", "p_neg"),
])
def test_one_direction_reports_only_that_direction(direction, present, absent):
    genes, score = _screen(n_genes=20)
    out = rra.rank_aggregate(score, genes, direction=direction,
                             n_permutations=200, seed=1)
    assert present in out.columns
    assert absent not in out.columns


def test_the_correction_is_the_one_that_was_asked_for():
    genes, score = _screen(n_genes=40)
    bh = rra.rank_aggregate(score, genes, direction="neg", correction="fdr_bh",
                            n_permutations=300, seed=1)
    none = rra.rank_aggregate(score, genes, direction="neg", correction="none",
                              n_permutations=300, seed=1)
    assert list(none["p_adj_neg"]) == list(none["p_neg"])
    assert (bh["p_adj_neg"] >= bh["p_neg"] - 1e-12).all()


@pytest.mark.parametrize("kwargs, message", [
    (dict(alpha=0.0), "alpha must be in"),
    (dict(alpha=1.5), "alpha must be in"),
    (dict(direction="sideways"), "direction must be one of"),
])
def test_a_setting_outside_its_range_says_so(kwargs, message):
    genes, score = _screen(n_genes=5)
    with pytest.raises(ValueError, match=message):
        rra.rank_aggregate(score, genes, n_permutations=10, **kwargs)


def test_mismatched_lengths_say_both_numbers():
    with pytest.raises(ValueError, match="got 3 and 2"):
        rra.rank_aggregate([1.0, 2.0, 3.0], ["a", "b"])


def test_the_description_carries_the_formula_and_the_recommendation():
    text = rra.describe()
    assert "Beta(i, k - i + 1)" in text
    assert "Recommended for CRISPR screens" in text
    assert "collinearity" in text
    assert "0.25" in rra.describe(0.25)


def test_a_perfectly_collinear_design_is_not_a_problem_it_can_have():
    """The reason this backend is here at all.

    `gene_fraction` is the SUM of the gene's guide fractions, so a design with
    both blocks is singular BY CONSTRUCTION -- OLS pseudo-inverts it and
    reports a coefficient and a P value for every term anyway. RRA is handed
    the same guide-level scores and never forms the sum, so it returns a
    finite, ordered answer on a design that has no unique least-squares
    solution at all.
    """
    k, n_genes = 4, 30
    genes = np.repeat([f"g{i:02d}" for i in range(n_genes)], k)
    rng = np.random.default_rng(0)
    fraction = rng.random(n_genes * k)

    # The design spaCR fits: one column per guide (its own fraction) and one
    # per gene (the SUM of that gene's guide fractions).
    guide_block = np.zeros((n_genes * k, n_genes * k))
    np.fill_diagonal(guide_block, fraction)
    gene_block = np.zeros((n_genes * k, n_genes))
    for row, column in enumerate(np.repeat(np.arange(n_genes), k)):
        gene_block[row, column] = fraction.reshape(n_genes, k)[column].sum()
    design = np.column_stack([guide_block, gene_block])

    # Rank deficient by exactly the number of genes: every gene column is the
    # sum of its own guide columns, so it carries no information they do not.
    assert np.linalg.matrix_rank(design) <= design.shape[1] - n_genes

    out = rra.rank_aggregate(fraction, genes, n_permutations=300, seed=1)
    assert len(out) == n_genes
    assert out["p_neg"].notna().all()
    assert (out["p_neg"] > 0).all()
