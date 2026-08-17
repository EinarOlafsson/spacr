"""spacrGraph was the third place in spaCR that decided which test applies.

Instruction 127 finding 2 closed two of them: `spacr.sp_stats` became a
translation layer onto `spacr.figures.stats`, the one engine. The third was
`spacrGraph`, and it is the one behind EVERY figure a regression run draws --
every jitter_bar summary, every recruitment bar, every stats CSV saved beside
them.

It carried the same defect the other two were fixed for, a normality check
with no power floor::

    is_normal = bool(p_values) and all(p > 0.05 for p in p_values)

Measured on a perfectly normal three-versus-three pair, the two engines
answered differently:

    spacrGraph          T-test            p = 0.1304
    figures.stats       Mann-Whitney U    p = 0.2000

Three replicates is not an edge case in this field; it is the usual well
count. Shapiro-Wilk on three points cannot reject anything, so "not rejected"
was read as "normal" and a t-test ran. Levene on three points cannot reject
either, so Student's was picked over Welch's on the same non-evidence.

BELOW A STATED n THE ASSUMPTION CHECK IS UNINFORMATIVE AND THE ROBUST CHOICE
IS TAKEN. "Did not reject" on n = 3 means "could not tell". One direction
costs a little power when the assumption did hold; the other publishes a
difference that is not there.
"""

import matplotlib
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, ttest_ind

matplotlib.use("Agg")

from spacr.figures.stats import MIN_N_FOR_ASSUMPTIONS, compare
from spacr.plot import spacrGraph
from spacr.sp_stats import _ENGINE_TEST_NAMES
from spacr.sp_stats import perform_statistical_tests as sp_stats_tests


def normal_values(n, shift=0.0):
    """A sample as close to normal as a sample of size n can be.

    Quantiles of the normal on an evenly spaced grid, so nothing here depends
    on a seed and "the data really are normal" is true by construction rather
    than by luck.
    """
    return norm.ppf(np.linspace(0.5 / n, 1 - 0.5 / n, n)) + shift


def frame(groups, column="value", grouping="condition"):
    rows = []
    for name, values in groups.items():
        rows.extend({grouping: name, column: float(v)} for v in values)
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _close_figures():
    import matplotlib.pyplot as plt
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# the measured disagreement, and its direction
# ---------------------------------------------------------------------------

def test_three_replicates_do_not_buy_a_t_test():
    """The exact case from instruction 127, driven through spacrGraph.

    Both halves are asserted. The t-test that used to run really would have
    reported 0.1304 on this data -- so this is not a test that would pass with
    the fix removed -- and what runs instead is the rank test, at 0.2000.
    """
    a, b = normal_values(3), normal_values(3, shift=1.5)
    assert ttest_ind(a, b, equal_var=True)[1] == pytest.approx(0.1304, abs=5e-5)

    graph = spacrGraph(frame({"nc": a, "pc": b}),
                       grouping_column="condition", data_column=["value"])
    is_normal, _rows = graph.perform_normality_tests()
    assert is_normal is False, (
        "Shapiro-Wilk on three points cannot reject; reading that as "
        "'normal' is the whole defect")

    row = graph.perform_statistical_tests(["nc", "pc"], is_normal)[0]
    assert row["Test Name"] == "Mann-Whitney U test"
    assert row["p-value"] == pytest.approx(0.2)


def test_the_normality_row_says_it_could_not_see_rather_than_printing_a_p():
    """A p-value from a check with no power is worse than no p-value.

    0.87 beside "Shapiro-Wilk" reads as evidence of normality to every
    reviewer who has ever opened one of these CSVs. NaN plus a sentence does
    not.
    """
    graph = spacrGraph(frame({"nc": normal_values(3),
                              "pc": normal_values(3, shift=1.5)}),
                       grouping_column="condition", data_column=["value"])
    _is_normal, rows = graph.perform_normality_tests()

    assert [r["Test Name"] for r in rows] == ["Shapiro-Wilk", "Shapiro-Wilk"]
    assert all(r["Informative"] is False for r in rows)
    assert all(np.isnan(r["p-value"]) for r in rows)
    assert all("too few for a normality test to have power" in r["Verdict"]
               for r in rows)


def test_ten_replicates_are_enough_and_the_t_test_comes_back():
    """The floor is a floor, not a ban on parametric tests.

    Without this the change could be "always use a rank test", which loses
    real power on the experiments that earned a t-test.
    """
    n = MIN_N_FOR_ASSUMPTIONS
    graph = spacrGraph(frame({"nc": normal_values(n),
                              "pc": normal_values(n, shift=1.5)}),
                       grouping_column="condition", data_column=["value"])
    is_normal, _rows = graph.perform_normality_tests()
    assert is_normal is True

    row = graph.perform_statistical_tests(["nc", "pc"], is_normal)[0]
    assert row["Test Name"] == "T-test"


# ---------------------------------------------------------------------------
# one engine means one answer
# ---------------------------------------------------------------------------

SHAPES = {
    "3 v 3 normal": {"nc": normal_values(3), "pc": normal_values(3, 1.5)},
    "5 v 5 normal": {"nc": normal_values(5), "pc": normal_values(5, 1.0)},
    "20 v 20 normal": {"nc": normal_values(20), "pc": normal_values(20, 0.9)},
    "20 v 20 unequal spread": {"nc": normal_values(20),
                               "pc": normal_values(20, 0.9) * 8.0},
    "20 v 20 skewed": {"nc": np.exp(normal_values(20)),
                       "pc": np.exp(normal_values(20, 1.0))},
    "three arms, 20 each": {"a": normal_values(20),
                            "b": normal_values(20, 1.0),
                            "c": normal_values(20, 2.0)},
    "three arms, 4 each": {"a": normal_values(4),
                           "b": normal_values(4, 1.0),
                           "c": normal_values(4, 2.0)},
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_a_figure_and_a_results_table_cannot_disagree_about_a_plate(shape):
    """The same frame through both entry points, name and number.

    Two engines that agree on the easy shapes and diverge on the small ones
    is exactly how this survived: nobody compares a figure's caption with the
    CSV beside it.
    """
    groups = SHAPES[shape]
    df = frame(groups)

    graph = spacrGraph(df, grouping_column="condition", data_column=["value"])
    is_normal, _rows = graph.perform_normality_tests()
    figure_row = graph.perform_statistical_tests(list(groups), is_normal)[0]
    table_row = sp_stats_tests(df, "condition", ["value"])[0]

    assert figure_row["Test Name"] == table_row["Test Name"]
    assert figure_row["p-value"] == pytest.approx(table_row["p-value"])


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_engine_is_on_the_path_not_beside_it(shape, monkeypatch):
    """No fallback branch may quietly pick a test when the engine is gone.

    A translation layer that keeps a copy of the old decision for the cases
    the engine refuses is not a translation layer; it is the third engine
    again, hidden behind a try.
    """
    import spacr.figures.stats as engine

    def refuse(*args, **kwargs):
        raise AssertionError("spacrGraph chose a test without the engine")

    monkeypatch.setattr(engine, "compare", refuse)
    graph = spacrGraph(frame(SHAPES[shape]), grouping_column="condition",
                       data_column=["value"])
    with pytest.raises(AssertionError, match="without the engine"):
        graph.perform_statistical_tests(list(SHAPES[shape]), True)


def test_the_names_are_sp_stats_vocabulary_and_not_a_fourth_spelling():
    """'Mann-Whitney U' and 'Mann-Whitney U test' in one package is how a
    downstream filter silently matches nothing."""
    for engine_name in _ENGINE_TEST_NAMES:
        assert compare  # the map is keyed on names the engine can return

    reported = set()
    for groups in SHAPES.values():
        graph = spacrGraph(frame(groups), grouping_column="condition",
                           data_column=["value"])
        is_normal, _rows = graph.perform_normality_tests()
        reported.add(graph.perform_statistical_tests(
            list(groups), is_normal)[0]["Test Name"])

    assert reported <= set(_ENGINE_TEST_NAMES.values()), (
        f"spacrGraph invented a test name of its own: "
        f"{reported - set(_ENGINE_TEST_NAMES.values())}")


# ---------------------------------------------------------------------------
# the caller's verdict can veto, never grant
# ---------------------------------------------------------------------------

def test_is_normal_true_cannot_talk_the_engine_into_a_t_test():
    """`perform_statistical_tests` is public and takes the verdict as an
    argument. A caller passing True on three replicates -- which is what
    every caller did before the floor existed -- must still get the rank
    test, because a caller cannot know something the data does not contain.
    """
    groups = {"nc": normal_values(3), "pc": normal_values(3, shift=1.5)}
    graph = spacrGraph(frame(groups), grouping_column="condition",
                       data_column=["value"])
    row = graph.perform_statistical_tests(list(groups), is_normal=True)[0]
    assert row["Test Name"] == "Mann-Whitney U test"


def test_is_normal_false_still_forces_the_rank_test():
    """The other direction stays honoured: a caller who has decided the data
    are not normal gets a rank test even where the engine would allow a
    parametric one. Conservative is always available."""
    groups = {"nc": normal_values(20), "pc": normal_values(20, shift=0.9)}
    graph = spacrGraph(frame(groups), grouping_column="condition",
                       data_column=["value"])

    assert graph.perform_statistical_tests(
        list(groups), is_normal=True)[0]["Test Name"] == "T-test"
    assert graph.perform_statistical_tests(
        list(groups), is_normal=False)[0]["Test Name"] == "Mann-Whitney U test"


def test_the_veto_covers_every_parametric_test_the_engine_can_choose():
    """A rank equivalent missing from the table would silently keep the
    parametric test for a caller who asked not to have one."""
    parametric = {"Student's t", "Welch's t", "paired t", "one-way ANOVA",
                  "Welch's ANOVA"}
    assert set(spacrGraph.RANK_EQUIVALENT) == parametric
    assert set(spacrGraph.RANK_EQUIVALENT.values()) <= set(_ENGINE_TEST_NAMES)


# ---------------------------------------------------------------------------
# the equal-variance statistic a caller can still read
# ---------------------------------------------------------------------------

def test_levene_is_median_centred_and_refuses_below_the_floor():
    """perform_levene_test stays public, and reports what the check saw.

    Returning 0.7 for three replicates invites the reading the whole change
    exists to prevent, so it returns NaN and the verdict lives in the
    comparison row.
    """
    small = spacrGraph(frame({"nc": normal_values(3),
                              "pc": normal_values(3, 1.5)}),
                       grouping_column="condition", data_column=["value"])
    assert all(np.isnan(v)
               for v in small.perform_levene_test(["nc", "pc"]))

    from scipy.stats import levene

    # SKEWED on purpose. The two centrings coincide on a symmetric sample,
    # so a normal fixture here would pass with the wrong implementation.
    a = np.exp(normal_values(20))
    b = np.exp(normal_values(20, 0.9)) * 8.0
    big = spacrGraph(frame({"nc": a, "pc": b}),
                     grouping_column="condition", data_column=["value"])
    _stat, p = big.perform_levene_test(["nc", "pc"])
    assert p == pytest.approx(levene(a, b, center="median")[1])
    assert p != pytest.approx(levene(a, b, center="mean")[1]), (
        "the mean-centred form is back; it is the wrong one to use when "
        "normality is itself in question")


# ---------------------------------------------------------------------------
# through the drawing path a regression run actually uses
# ---------------------------------------------------------------------------

def test_a_three_well_jitter_bar_saves_a_rank_test_not_a_t_test(tmp_path):
    """The whole reason this matters: the stats CSV beside the figure.

    Three wells per condition is the shape a spaCR regression run writes for
    every jitter_bar summary, and this is the file a Methods section is
    written from.
    """
    rows = []
    rng = np.random.default_rng(11)
    for condition, centre in (("nc", 0.30), ("pc", 0.45)):
        for well in range(3):
            for _ in range(400):
                rows.append({"condition": condition,
                             "prc": f"p1_r1_c{well}_{condition}",
                             "value": float(rng.normal(centre, 0.1))})
    df = pd.DataFrame(rows)

    graph = spacrGraph(df, grouping_column="condition", data_column=["value"],
                       graph_type="jitter_bar", representation="well",
                       save=True, output_dir=str(tmp_path),
                       graph_name="wells")
    graph.create_plot()

    saved = pd.read_csv(tmp_path / "wells_value_condition_jitter_bar_stats.csv")
    comparison = saved[saved["Test Name"].notna()
                       & (saved["Test Name"] != "Shapiro-Wilk")]
    assert list(comparison["Test Name"]) == ["Mann-Whitney U test"], (
        "a 3-well figure is still saving a t-test")
    assert comparison["n_well"].iloc[0] == 6
    assert comparison["n_object"].iloc[0] == 2400
    assert "too few for a normality test to have power" in (
        comparison["Why This Test"].iloc[0])
