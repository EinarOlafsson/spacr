"""The reported p-value counted objects when the design supplies wells.

`plot_proportion_stacked_bars` computed one chi-squared over object counts
and reported it at every level, while the `level` tooltip said outright that
"the reported statistics always treat the well as the unit of replication".
Measured across 1800 objects in 3 plates x 3 wells, object, well and plate
returned chi2 = 0.5794 and p = 4.4655e-01 -- byte-identical.

Objects in a well are not independent: they share a treatment, a
transfection, an imaging session and a monolayer. A chi-squared over 20,000
objects with three wells per condition answers "were these 20,000 objects
drawn from one distribution", which nobody asked, and its p-value is smaller
than the experiment supports by orders of magnitude. That is
pseudo-replication, and it is the most common way an image screen overstates
a result.

Two tests now sit beside the old number, both driven by `level`: a
comparison of the per-unit PROPORTIONS, and a binomial model with standard
errors clustered on the unit. The old number is kept, not replaced -- every
published figure came from it.
"""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from spacr.plot import (proportion_mixed_model, proportion_test_by_unit,
                        proportions_per_unit)

pytestmark = pytest.mark.filterwarnings("ignore")


def screen(*, n_per_condition, wells_per_condition, fractions, spread, seed=0):
    """A plate where the per-well fraction is known by construction."""
    rng = np.random.default_rng(seed)
    rows = []
    for condition, base in fractions.items():
        for well in range(wells_per_condition):
            fraction = base + (spread if well == 0 else
                               -spread if well == 1 else 0.0)
            for _ in range(n_per_condition // wells_per_condition):
                rows.append({
                    "condition": condition,
                    "prc": f"p1_r1_c{well}_{condition}",
                    "plateID": "p1",
                    "cls": int(rng.random() < fraction),
                })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def loud_wells():
    """Huge n, tiny effect, large well-to-well spread.

    The acceptance case from the instruction: the object test should be
    wildly significant and the well test should not be. If they agree here,
    the well test is not actually using wells.
    """
    return screen(n_per_condition=20000, wells_per_condition=3,
                  fractions={"nc": 0.50, "pc": 0.52}, spread=0.06)


# ---------------------------------------------------------------------------
# the per-unit proportions
# ---------------------------------------------------------------------------

def test_each_well_contributes_one_row_per_condition(loud_wells):
    table = proportions_per_unit(loud_wells, "condition", "cls", "prc")
    assert len(table) == loud_wells["prc"].nunique() == 6


def test_the_proportions_sum_to_one_within_a_well(loud_wells):
    table = proportions_per_unit(loud_wells, "condition", "cls", "prc")
    bins = [c for c in table.columns if c not in ("condition", "prc")]
    assert np.allclose(table[bins].sum(axis=1), 1.0)


def test_a_wells_proportion_is_its_own_not_the_conditions(loud_wells):
    """The spread has to survive aggregation, or the test cannot see it."""
    table = proportions_per_unit(loud_wells, "condition", "cls", "prc")
    within_nc = table.loc[table["condition"] == "nc", 1]
    assert within_nc.std() > 0.01, "the per-well spread was averaged away"


# ---------------------------------------------------------------------------
# the acceptance case
# ---------------------------------------------------------------------------

def test_the_object_test_is_significant_on_data_the_design_cannot_support(
        loud_wells):
    """Not a claim that the object test is wrong -- a claim about its n."""
    from scipy.stats import chi2_contingency

    counts = (loud_wells.groupby(["condition", "cls"], observed=True).size()
              .unstack(fill_value=0))
    # Six wells, a 2-point difference in fraction, and a well-to-well spread
    # three times that difference -- an experiment that supports nothing.
    # The object test returns p ~ 1e-6 on it.
    assert chi2_contingency(counts)[1] < 1e-5


def test_the_well_test_is_not(loud_wells):
    result = proportion_test_by_unit(loud_wells, "condition", "cls", "prc")
    assert (result["p_value"] > 0.05).all(), (
        "three wells per condition with this spread cannot support a claim; "
        "if this passes at p<0.05 the test is not using wells")


def test_the_mixed_model_agrees_with_the_wells_not_the_objects(loud_wells):
    result = proportion_mixed_model(loud_wells, "condition", "cls", "prc")
    assert result["p_value"].notna().all()
    assert (result["p_value"] > 0.05).all()


def test_n_is_the_number_of_wells_and_the_table_says_so(loud_wells):
    for result in (proportion_test_by_unit(loud_wells, "condition", "cls",
                                           "prc"),
                   proportion_mixed_model(loud_wells, "condition", "cls",
                                          "prc")):
        assert (result["unit"] == "prc").all()
        assert (result["n"] == 6).all()


# ---------------------------------------------------------------------------
# a real effect is still found
# ---------------------------------------------------------------------------

def test_a_difference_bigger_than_the_well_spread_is_detected():
    """Otherwise the fix would just be a test that never fires."""
    df = screen(n_per_condition=6000, wells_per_condition=4,
                fractions={"nc": 0.20, "pc": 0.80}, spread=0.03, seed=7)

    proportions = proportion_test_by_unit(df, "condition", "cls", "prc")
    assert (proportions["p_value"] < 0.05).all()

    model = proportion_mixed_model(df, "condition", "cls", "prc")
    assert (model["p_value"] < 0.05).all()


def test_the_test_chosen_is_named_in_the_results(loud_wells):
    result = proportion_test_by_unit(loud_wells, "condition", "cls", "prc")
    assert result["test"].str.contains("per-prc proportions").all()
    assert result["test"].str.contains(
        "T-test|Mann-Whitney|ANOVA|Kruskal").all()


# ---------------------------------------------------------------------------
# too little data is said, not guessed
# ---------------------------------------------------------------------------

def test_one_well_per_condition_reports_that_rather_than_a_p_value():
    """Two numbers cannot be compared, and inventing a p-value is worse than
    saying so."""
    df = screen(n_per_condition=200, wells_per_condition=1,
                fractions={"nc": 0.3, "pc": 0.7}, spread=0.0)

    result = proportion_test_by_unit(df, "condition", "cls", "prc")
    assert result["p_value"].isna().all()
    assert result["test"].str.contains("too few units").all()


def test_three_conditions_use_a_three_group_test():
    df = screen(n_per_condition=3000, wells_per_condition=4,
                fractions={"a": 0.2, "b": 0.5, "c": 0.8}, spread=0.02, seed=3)
    result = proportion_test_by_unit(df, "condition", "cls", "prc")
    assert result["test"].str.contains("ANOVA|Kruskal").all()
    assert (result["p_value"] < 0.05).all()


def test_the_plate_is_a_usable_unit_too():
    df = pd.concat([
        screen(n_per_condition=2000, wells_per_condition=2,
               fractions={"nc": 0.3, "pc": 0.7}, spread=0.02, seed=s)
        .assign(plateID=f"p{s}") for s in (1, 2, 3)], ignore_index=True)

    result = proportion_test_by_unit(df, "condition", "cls", "plateID")
    assert (result["unit"] == "plateID").all()
    assert (result["n"] == 3).all()
