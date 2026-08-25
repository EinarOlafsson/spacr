"""What the guide permutation analysis refuses, and how it labels a volcano.

Every refusal here is driven with a real long table -- one bad number in an
otherwise valid frame -- because the point of each guard is to catch that
number before 200,000 permutations are spent on it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import guide_permutation as gp


def _long_table(wells=8, guides=("g1", "g2", "g3"), seed=0):
    """A valid long guide table: two plates, one fraction per well/guide."""
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(wells):
        well = f"p{index // 4 + 1}_r1_c{index + 1}"
        plate = f"plate{index // 4 + 1}"
        phenotype = float(rng.normal())
        for order, guide in enumerate(guides):
            rows.append({
                "prc": well,
                "grna": guide,
                "fraction": float(rng.uniform(0.05, 0.9)) + 0.01 * order,
                "plateID": plate,
                "pathogen_rate": phenotype,
                "cell_count": float(200 + index * 7),
                "batch": "A" if index % 2 else "B",
            })
    return pd.DataFrame(rows)


def _prepared(**kwargs):
    return gp.prepare_long_guide_data(
        _long_table(**kwargs), "pathogen_rate",
        nuisance_columns=["cell_count"])


# ---------------------------------------------------------------------------
# Support thresholds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["4", None, 3.5])
def test_a_support_threshold_that_is_not_a_count_is_refused(bad):
    with pytest.raises(TypeError, match="positive integer"):
        gp._normalise_thresholds(bad)


@pytest.mark.parametrize("bad", [[], [0], [2, -1]])
def test_a_support_threshold_below_one_is_refused(bad):
    with pytest.raises(ValueError, match="at least one positive integer"):
        gp._normalise_thresholds(bad)


def test_repeated_thresholds_collapse_to_one_family_each():
    assert gp._normalise_thresholds([3, 1, 3, 2]) == (1, 2, 3)
    assert gp._normalise_thresholds(4) == (4,)


# ---------------------------------------------------------------------------
# Preparing the long table
# ---------------------------------------------------------------------------

def test_a_missing_well_identifier_is_refused_before_any_pivot():
    table = _long_table()
    table.loc[0, "prc"] = np.nan
    with pytest.raises(ValueError, match="must not contain missing values"):
        gp.prepare_long_guide_data(table, "pathogen_rate")


def test_an_infinite_fraction_is_refused():
    table = _long_table()
    table.loc[2, "fraction"] = np.inf
    with pytest.raises(ValueError, match="fractions must be finite"):
        gp.prepare_long_guide_data(table, "pathogen_rate")


def test_a_negative_fraction_is_refused():
    table = _long_table()
    table.loc[3, "fraction"] = -0.2
    with pytest.raises(ValueError, match="non-negative"):
        gp.prepare_long_guide_data(table, "pathogen_rate")


def test_an_infinite_phenotype_is_refused_for_the_well_it_belongs_to():
    table = _long_table()
    table.loc[table["prc"] == table["prc"].iloc[0], "pathogen_rate"] = np.inf
    with pytest.raises(ValueError, match="must be finite for every analyzed"):
        gp.prepare_long_guide_data(table, "pathogen_rate")


# ---------------------------------------------------------------------------
# The nuisance design
# ---------------------------------------------------------------------------

def test_the_block_column_is_not_added_twice_when_named_as_a_nuisance():
    _fractions, outcomes, _meta = _prepared()
    once = gp._nuisance_design(outcomes, "plateID", [])
    twice = gp._nuisance_design(outcomes, "plateID", ["plateID"])
    assert twice.shape == once.shape
    assert np.array_equal(twice, once)


def test_a_numeric_nuisance_adds_one_column_and_must_be_finite():
    _fractions, outcomes, _meta = _prepared()
    plain = gp._nuisance_design(outcomes, "plateID", [])
    with_count = gp._nuisance_design(outcomes, "plateID", ["cell_count"])
    assert with_count.shape[1] == plain.shape[1] + 1

    outcomes = outcomes.copy()
    outcomes.loc[outcomes.index[0], "cell_count"] = np.inf
    with pytest.raises(ValueError, match="must be finite"):
        gp._nuisance_design(outcomes, "plateID", ["cell_count"])


def test_a_nuisance_that_repeats_the_block_is_refused_as_rank_deficient():
    """A second spelling of the same split adds no information and no rank."""
    _fractions, outcomes, _meta = _prepared()
    outcomes = outcomes.copy()
    outcomes["site"] = outcomes["plateID"].map(
        {"plate1": "north", "plate2": "south"})
    with pytest.raises(ValueError, match="rank deficient"):
        gp._nuisance_design(outcomes, "plateID", ["site"])


def test_a_block_column_that_is_absent_says_so():
    _fractions, outcomes, _meta = _prepared()
    with pytest.raises(ValueError, match="absent from outcomes"):
        gp._nuisance_design(outcomes, "no_such_block", [])


# ---------------------------------------------------------------------------
# The test itself
# ---------------------------------------------------------------------------

def _run(fractions, outcomes, **kwargs):
    kwargs.setdefault("n_permutations", 20)
    kwargs.setdefault("min_wells", 1)
    kwargs.setdefault("batch_size", 10)
    return gp.guide_freedman_lane_test(
        fractions, outcomes, "pathogen_rate", **kwargs)


def test_a_statistic_nobody_implements_is_refused():
    fractions, outcomes, _meta = _prepared()
    with pytest.raises(ValueError, match="must be 'pearson' or 'rank'"):
        _run(fractions, outcomes, statistic="kendall")


def test_the_permutation_budget_must_be_at_least_one():
    fractions, outcomes, _meta = _prepared()
    with pytest.raises(ValueError, match="n_permutations must be at least 1"):
        _run(fractions, outcomes, n_permutations=0)


def test_the_batch_size_must_be_at_least_one():
    fractions, outcomes, _meta = _prepared()
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        _run(fractions, outcomes, batch_size=0)


def test_a_negative_presence_threshold_is_refused():
    fractions, outcomes, _meta = _prepared()
    with pytest.raises(ValueError, match="presence_threshold must be non-neg"):
        _run(fractions, outcomes, presence_threshold=-0.1)


def test_an_outcome_column_that_is_absent_says_so():
    fractions, outcomes, _meta = _prepared()
    with pytest.raises(ValueError, match="'missing_phenotype' is absent"):
        gp.guide_freedman_lane_test(fractions, outcomes, "missing_phenotype",
                                    n_permutations=20, min_wells=1)


def test_wells_in_a_different_order_are_realigned_not_rejected():
    fractions, outcomes, _meta = _prepared()
    shuffled = outcomes.iloc[::-1]
    result = _run(fractions, shuffled)
    straight = _run(fractions, outcomes)
    assert np.allclose(
        result["standardized_marginal_effect"].to_numpy(),
        straight["standardized_marginal_effect"].to_numpy())


def test_wells_that_do_not_match_at_all_are_refused():
    fractions, outcomes, _meta = _prepared()
    renamed = outcomes.copy()
    renamed.index = [f"other_{name}" for name in renamed.index]
    with pytest.raises(ValueError, match="do not align"):
        _run(fractions, renamed)


def test_a_negative_fraction_reaching_the_test_is_refused():
    fractions, outcomes, _meta = _prepared()
    fractions = fractions.copy()
    fractions.iloc[0, 0] = -1.0
    with pytest.raises(ValueError, match="finite and non-negative"):
        _run(fractions, outcomes)


def test_a_threshold_no_guide_can_meet_says_what_it_asked_for():
    fractions, outcomes, _meta = _prepared()
    with pytest.raises(ValueError, match="No guide occurs in at least 99"):
        _run(fractions, outcomes, min_wells=99)


def test_an_infinite_phenotype_reaching_the_test_is_refused():
    fractions, outcomes, _meta = _prepared()
    outcomes = outcomes.copy()
    outcomes.loc[outcomes.index[0], "pathogen_rate"] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        _run(fractions, outcomes)


def test_a_phenotype_the_blocks_explain_entirely_has_nothing_left_to_test():
    """One value per plate is the block effect itself: no residual, no test."""
    fractions, outcomes, _meta = _prepared()
    outcomes = outcomes.copy()
    outcomes["pathogen_rate"] = outcomes["plateID"].map(
        {"plate1": 0.2, "plate2": 0.9}).astype(float)
    with pytest.raises(ValueError, match="no residual variation"):
        gp.guide_freedman_lane_test(
            fractions, outcomes, "pathogen_rate", n_permutations=20,
            min_wells=1, nuisance_columns=[])


def test_a_guide_with_the_same_fraction_everywhere_is_named_not_divided_by(
        ):
    fractions, outcomes, _meta = _prepared()
    fractions = fractions.copy()
    fractions["g2"] = 0.5
    with pytest.raises(ValueError, match=r"no residual variation.*g2"):
        _run(fractions, outcomes, nuisance_columns=[])


def test_guide_metadata_given_as_a_column_is_joined_by_name(
        ):
    fractions, outcomes, meta = _prepared()
    metadata = meta.reset_index()
    metadata["gene"] = ["GENE_A", "GENE_B", "GENE_C"]
    result = _run(fractions, outcomes, guide_metadata=metadata)
    assert set(result["gene"]) == {"GENE_A", "GENE_B", "GENE_C"}
    assert (result.loc[result["guide"] == "g1", "gene"] == "GENE_A").all()


def test_the_rank_statistic_is_a_different_number_from_pearson():
    fractions, outcomes, _meta = _prepared()
    pearson = _run(fractions, outcomes, statistic="pearson")
    ranked = _run(fractions, outcomes, statistic="rank")
    assert not np.allclose(
        pearson["standardized_marginal_effect"].to_numpy(),
        ranked["standardized_marginal_effect"].to_numpy())


# ---------------------------------------------------------------------------
# The volcano
# ---------------------------------------------------------------------------

def _results(guides=("g1", "g2", "g3", "g4")):
    return pd.DataFrame({
        "guide": list(guides),
        "outcome": "pathogen_rate",
        "minimum_wells_threshold": 1,
        "standardized_marginal_effect": [0.4, -0.3, 0.2, -0.1],
        "permutation_p_value": [0.001, 0.002, 0.4, 0.5],
        "adjusted_p_value": [0.004, 0.008, 0.6, 0.7],
        "significant": [True, True, False, False],
        "multiple_testing_method": "fdr_bh",
        "alpha": 0.05,
    })


def test_labelled_guides_are_annotated_and_absent_ones_are_skipped(tmp_path):
    """Four labels, three of which exist: the fourth must not raise."""
    import matplotlib

    matplotlib.use("Agg")
    written = gp.plot_guide_permutation_volcano(
        _results(), outcome="pathogen_rate", minimum_wells=1,
        save_path=tmp_path / "volcano.png",
        label_guides={"g1": "GENE_A g1", "g2": "GENE_B g2",
                      "g3": "GENE_C g3", "g9": "not in the table"},
        effect_threshold=0.25,
    )
    assert written.exists()
    assert written.stat().st_size > 0


def test_a_volcano_for_a_family_with_no_rows_says_so(tmp_path):
    with pytest.raises(ValueError, match="No rows for outcome"):
        gp.plot_guide_permutation_volcano(
            _results(), outcome="pathogen_rate", minimum_wells=7,
            save_path=tmp_path / "volcano.png")
