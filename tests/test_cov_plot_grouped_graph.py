"""Branch coverage for ``spacr.plot.create_grouped_plot`` and ``spacr.plot.spacrGraph``.

Everything here is CPU-only, offline and deterministic: group data is built
from normal quantiles (``norm.ppf`` on an evenly spaced grid) so the
Shapiro-Wilk normality verdicts are fixed rather than sampled, and the
log-transform of the same grid gives a reproducibly *non*-normal group.

Which test applies is decided by ``spacr.figures.stats`` since 2026-08-17
(instruction 127 finding 2); ``spacrGraph`` is a translation layer onto it.
Assertions that moved with that change carry the date and the reason.

Four genuine defects found while writing these tests are pinned with
``xfail(strict=True)`` asserting the CORRECT behaviour:

* ``spacrGraph.remove_outliers_from_plot`` drops every row (Series & DataFrame
  boolean mix), not just the outliers.
* ``spacrGraph.perform_statistical_tests`` reads ``'T'`` out of
  ``pingouin.wilcoxon`` output, which is named ``'W-val'``.
* Dunn's post-hoc ``n_well`` uses ``len(self.df[col] == b)`` (a mask length)
  instead of ``len(self.df[self.df[col] == b])``.
* Multi-data-column plots pass ``order=self.order`` (group names) while the
  x-axis is the ``'Combined Group'`` column, so nothing is drawn.

The tests are parametrised by contract. A row is a helper that does the setup
and returns what it observed, next to the value the contract promises; the
helpers keep their secondary assertions.
"""
from __future__ import annotations

import inspect
import re

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("seaborn")
pytest.importorskip("statsmodels")

import seaborn as sns  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures / deterministic data builders
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figures():
    """Never let Agg figures accumulate across tests."""
    import matplotlib.pyplot as plt
    yield
    plt.close("all")


def _normal_values(n, shift=0.0):
    """Deterministic, essentially perfectly normal sample of size ``n``."""
    from scipy.stats import norm
    return norm.ppf(np.linspace(0.5 / n, 1 - 0.5 / n, n)) + shift


def _skewed_values(n, scale=1.0):
    """Deterministic log-normal sample (normaltest p < 1e-4)."""
    return np.exp(_normal_values(n)) * scale


def _frame(groups, values_per_group, extra_cols=None):
    """Build a tidy DataFrame: one row per value with a 'grp' and a 'prc'."""
    rows = []
    for g, vals in zip(groups, values_per_group):
        for i, v in enumerate(vals):
            rec = {"grp": g, "v1": float(v),
                   "prc": f"plate{1 + i % 2}_r{1 + i % 3}_c{1 + i % 4}"}
            rows.append(rec)
    df = pd.DataFrame(rows)
    if extra_cols:
        for name, arr in extra_cols.items():
            df[name] = arr
    return df


def normal_df(groups=("a", "b", "c"), n=20):
    return _frame(groups, [_normal_values(n, shift=10.0 + 2 * i)
                           for i in range(len(groups))])


def skewed_df(groups=("a", "b", "c"), n=20):
    return _frame(groups, [_skewed_values(n, scale=1.0 + i)
                           for i in range(len(groups))])


def multi_col_df(groups=("a", "b", "c"), n=12):
    """Two positive data columns, needed for the multi-data-column code path."""
    df = _frame(groups, [_normal_values(n, shift=10.0 + 2 * i)
                         for i in range(len(groups))])
    df["v2"] = df["v1"] * 1.1 + 0.5
    return df


def line_df(groups=("p1", "p2", "p3"), n=8):
    rows = []
    for gi, g in enumerate(groups):
        for e in range(n):
            rows.append({"grp": g, "epoch": float(e + 1),
                         "acc": 0.5 + 0.04 * e + 0.01 * gi})
    return pd.DataFrame(rows)


def _call(request, helper):
    """Run a case helper, giving it the fixtures named in its signature."""
    names = inspect.signature(helper).parameters
    return helper(**{n: request.getfixturevalue(n) for n in names})


def _bind(helper, *args):
    """A fixture-free case that calls ``helper`` with fixed arguments."""
    def case():
        return helper(*args)
    return case


def _refused(exception, pattern, call):
    """Run ``call`` expecting ``exception``; return the matched fragment."""
    with pytest.raises(exception, match=pattern) as refusal:
        call()
    return re.search(pattern, str(refusal.value)).group(0)


# ===========================================================================
# create_grouped_plot
# ===========================================================================

def _case_a_grouped_plot_does_not_restyle_the_rest_of_the_session():
    """One grouped plot must not decide how every later figure looks.

    ``sns.set(style="whitegrid")`` writes a whole seaborn theme -- grid,
    background, fonts, colour cycle -- into matplotlib's process-wide
    rcParams. Drawing one grouped plot therefore restyled every figure the
    session drew after it, on every other screen, over whatever the user had
    chosen in figure preferences.

    The plot must still come out on a white grid, so both halves are asserted.
    """
    import matplotlib

    from spacr.plot import create_grouped_plot

    before = {key: repr(value) for key, value in matplotlib.rcParams.items()}

    fig, _results = create_grouped_plot(
        normal_df(groups=("a", "b"), n=20), grouping_column="grp",
        data_column="v1", graph_type="bar", save=False)

    # It drew itself on the seaborn whitegrid...
    has_grid = bool(fig.get_axes()[0].xaxis.get_gridlines())
    # ...without leaving the theme behind for the next figure.
    after = {key: repr(value) for key, value in matplotlib.rcParams.items()}
    changed = sorted(key for key in after if before.get(key) != after[key])
    return (has_grid, changed)


def _case_create_grouped_plot_two_normal_groups_uses_ttest_and_explicit_order():
    """order= branch + 2-group normal branch (T-test)."""
    from spacr.plot import create_grouped_plot

    df = normal_df(groups=("b", "a"), n=20)
    fig, results = create_grouped_plot(
        df, grouping_column="grp", data_column="v1",
        graph_type="bar", order=["a", "b"], save=False)

    assert fig is not None
    names = set(results["Test Name"])
    # exactly one pairwise comparison for two groups
    comparisons = list(results.loc[results["Test Name"] == "T-test",
                                   "Comparison"])
    # the explicit order drives the x-axis even though 'b' comes first in df
    ax = fig.get_axes()[0]
    return (names, comparisons, [t.get_text() for t in ax.get_xticklabels()])


def _case_create_grouped_plot_two_skewed_groups_uses_mann_whitney():
    """2-group non-normal branch (Mann-Whitney U)."""
    from spacr.plot import create_grouped_plot

    df = skewed_df(groups=("a", "b"), n=20)
    _fig, results = create_grouped_plot(
        df, grouping_column="grp", data_column="v1", graph_type="bar")

    pw = results[results["Test Name"] == "Mann-Whitney U test"]
    assert 0.0 <= float(pw["p-value"].iloc[0]) <= 1.0
    return ("Mann-Whitney U test" in set(results["Test Name"]),
            "T-test" in set(results["Test Name"]), len(pw))


def _case_create_grouped_plot_three_skewed_groups_use_a_pairwise_rank_test():
    """CHANGED 2026-08-17: the three rows said 'Kruskal-Wallis test'.

    They were `kruskal(data1, data2)` -- Kruskal-Wallis across TWO groups,
    which is a Mann-Whitney U in a chi-squared approximation, filed under a
    name that told a reader an omnibus test had run. The choice is the one
    engine's now and each pair is named as the two-group test it is.
    """
    from spacr.plot import create_grouped_plot

    df = skewed_df(groups=("a", "b", "c"), n=20)
    _fig, results = create_grouped_plot(
        df, grouping_column="grp", data_column="v1", graph_type="bar")

    # 3 groups -> 3 pairwise comparisons
    # Tukey post-hoc is only added for the normal case
    return (int((results["Test Name"] == "Mann-Whitney U test").sum()),
            "Tukey HSD Post-hoc" in set(results["Test Name"]))


def _case_create_grouped_plot_three_normal_groups_adds_tukey_posthoc():
    """CHANGED 2026-08-17: the three pairwise rows said 'One-way ANOVA'.

    A LABEL CHANGE, not an answer change: `f_oneway` on two groups returns
    F = t squared and therefore the identical p-value, which is asserted
    below so the rename cannot hide a moved number. The Tukey rows are
    untouched -- they are the genuine across-group post-hoc.
    """
    from scipy.stats import ttest_ind
    from spacr.plot import create_grouped_plot

    df = normal_df(groups=("a", "b", "c"), n=20)
    _fig, results = create_grouped_plot(
        df, grouping_column="grp", data_column="v1", graph_type="bar")

    pair = results[results["Comparison"] == "a vs b"]
    expected = ttest_ind(df.loc[df["grp"] == "a", "v1"],
                         df.loc[df["grp"] == "b", "v1"], equal_var=True)[1]
    assert float(pair["p-value"].iloc[0]) == pytest.approx(expected)
    return (int((results["Test Name"] == "T-test").sum()),
            int((results["Test Name"] == "Tukey HSD Post-hoc").sum()))


def _case_create_grouped_plot_sem_errorbars_custom_colors_and_ylim():
    """sem error-bar branch + explicit colors branch + y_lim branch."""
    import matplotlib.pyplot as plt
    from spacr.plot import create_grouped_plot

    df = normal_df(groups=("a", "b", "c"), n=20)
    colors = ["#ff0000", "#00ff00", "#0000ff"]
    fig, results = create_grouped_plot(
        df, grouping_column="grp", data_column="v1", graph_type="bar",
        colors=colors, error_bar_type="sem", y_lim=[0, 30])

    ax = fig.get_axes()[0]
    # three bars, one per group, painted with the requested colours
    bars = [p for p in ax.patches if isinstance(p, plt.Rectangle)]
    return (ax.get_ylim(), not results.empty, len(bars))


def _case_create_grouped_plot_rejects_unknown_error_bar_type():
    from spacr.plot import create_grouped_plot

    df = normal_df(groups=("a", "b"), n=20)
    return _refused(
        ValueError, "Invalid error_bar_type",
        lambda: create_grouped_plot(df, grouping_column="grp",
                                    data_column="v1", graph_type="bar",
                                    error_bar_type="iqr"))


def _case_create_grouped_plot_other_graph_types(graph_type):
    """The violin / jitter / box / jitter_box rendering branches."""
    from spacr.plot import create_grouped_plot

    df = normal_df(groups=("a", "b", "c"), n=20)
    fig, results = create_grouped_plot(
        df, grouping_column="grp", data_column="v1", graph_type=graph_type)

    ax = fig.get_axes()[0]
    # every renderer must have put *something* on the axes
    drew = len(ax.collections) + len(ax.patches) + len(ax.lines) > 0
    # normality (3) + pairwise (3) + tukey (3)
    return (drew, set(t.get_text() for t in ax.get_xticklabels()),
            len(results))


def _case_create_grouped_plot_saves_plot_and_stats(tmp_path):
    from spacr.plot import create_grouped_plot

    df = normal_df(groups=("a", "b"), n=20)
    _fig, results = create_grouped_plot(
        df, grouping_column="grp", data_column="v1", graph_type="bar",
        output_dir=str(tmp_path), save=True)

    # The extension follows the figure-format preference now.
    n_plots = len(list(tmp_path.glob("grouped_plot.*")))
    csv = tmp_path / "test_results.csv"
    assert csv.is_file()
    on_disk = pd.read_csv(csv)
    return (n_plots, len(on_disk) == len(results), list(on_disk.columns))


# ===========================================================================
# spacrGraph — theme helpers
# ===========================================================================

def _case_set_reordered_theme_without_order_returns_palette_untouched(monkeypatch):
    """order=None branch + show_theme branch of _set_reordered_theme."""
    import matplotlib.pyplot as plt
    from spacr.plot import spacrGraph

    calls = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: calls.append(1))

    g = spacrGraph(normal_df(("a", "b"), n=20), "grp", "v1")
    palette = g._set_reordered_theme(theme="deep", order=None, n_colors=5,
                                     show_theme=True)

    assert calls == [1]  # show_theme actually rendered the palette
    return list(palette)


def _case_set_theme_reorders_palette():
    from spacr.plot import spacrGraph

    g = spacrGraph(normal_df(("a", "b"), n=20), "grp", "v1", theme="deep")
    return list(g.sns_palette[:7])


# ===========================================================================
# spacrGraph.preprocess_data
# ===========================================================================

def _case_preprocess_object_representation_does_not_aggregate():
    from spacr.plot import spacrGraph

    df = normal_df(("a", "b"), n=20)
    g = spacrGraph(df, "grp", "v1", representation="object")
    assert len(g.df) == len(df)  # one row per object, no group-by
    return (len(g.df), list(g.df["grp"].cat.categories))


def _case_preprocess_well_representation_aggregates_per_prc():
    from spacr.plot import spacrGraph

    df = normal_df(("a", "b"), n=20)
    g = spacrGraph(df, "grp", "v1", representation="well")
    assert len(g.df) == len(df.groupby(["prc", "grp"], observed=True))
    assert len(g.df) < len(df)
    return (sorted(g.df.columns), len(g.df))


def _case_preprocess_plate_representation_splits_prc():
    from spacr.plot import spacrGraph

    df = normal_df(("a", "b"), n=20)
    g = spacrGraph(df, "grp", "v1", representation="plate")
    # prc encodes 2 plates x 2 groups
    return (sorted(g.df.columns), len(g.df), sorted(set(g.df["plateID"])))


def _case_preprocess_plate_representation_grouping_on_plate_id():
    """grouping_column == 'plateID' collapses group_cols to a single column."""
    from spacr.plot import spacrGraph

    df = normal_df(("a", "b"), n=20)
    df["plateID"] = ["plate1" if i % 2 else "plate2" for i in range(len(df))]
    g = spacrGraph(df, "plateID", "v1", representation="plate")
    return (list(g.df.columns), len(g.df))


def _case_preprocess_plate_representation_without_prc_raises_keyerror():
    from spacr.plot import spacrGraph

    df = normal_df(("a", "b"), n=20).drop(columns=["prc"])
    return _refused(KeyError, "cannot split from 'prc'",
                    lambda: spacrGraph(df, "grp", "v1",
                                       representation="plate"))


def _case_preprocess_unknown_representation_raises_valueerror():
    from spacr.plot import spacrGraph

    return _refused(ValueError, "Unknown representation: sample",
                    lambda: spacrGraph(normal_df(("a", "b"), n=20), "grp",
                                       "v1", representation="sample"))


def _case_preprocess_falls_back_to_sorted_categories_when_order_empty():
    """All-NaN grouping column -> self.order == [] -> default-category branch."""
    from pandas.api.types import CategoricalDtype
    from spacr.plot import spacrGraph

    df = pd.DataFrame({"grp": [np.nan] * 6, "v1": np.arange(6, dtype=float)})
    g = spacrGraph(df, "grp", "v1")

    # every row is dropped by dropna
    return (g.order, len(g.df), isinstance(g.df["grp"].dtype, CategoricalDtype),
            list(g.df["grp"].cat.categories))


# ===========================================================================
# spacrGraph.remove_outliers_from_plot  (BUG)
# ===========================================================================

def _case_remove_outliers_from_plot_keeps_inliers():
    from spacr.plot import spacrGraph

    # 11 evenly spaced inliers per group (no natural 1.5*IQR outlier) + 1 spike
    base = list(np.linspace(1.0, 11.0, 11))
    df = _frame(("a", "b"), [base + [1000.0], base + [2000.0]])
    g = spacrGraph(df, "grp", "v1", representation="object")

    filtered = g.remove_outliers_from_plot()

    assert len(filtered) == len(df) - 2          # only the two spikes go
    return (len(filtered), float(filtered["v1"].max()),
            sorted(set(filtered["grp"])))


def _case_create_plot_with_remove_outliers_still_plots():
    from spacr.plot import spacrGraph

    base = list(np.linspace(1.0, 11.0, 11))
    df = _frame(("a", "b"), [base + [1000.0], [v + 2 for v in base] + [2000.0]])
    g = spacrGraph(df, "grp", "v1", representation="object", remove_outliers=True)
    g.create_plot()

    assert g.get_figure() is not None
    assert len(g.df) == len(df) - 2
    return len(g.df)


# ===========================================================================
# spacrGraph.perform_normality_tests
# ===========================================================================

def _case_perform_normality_tests_reports_shapiro_and_its_power(capsys):
    """CHANGED 2026-08-17 (instruction 127 finding 2, Einar).

    The check itself moved: `spacrGraph` no longer runs D'Agostino above n=8
    and Shapiro below it, it asks `spacr.figures.stats.check_normality`, the
    one engine, which is Shapiro-Wilk against a Bonferroni threshold across
    the groups. Two assertions here reversed, and both reversed because the
    ANSWER changed rather than the label:

    * ``is_normal`` was True. Groups of 10, 5 and 2 cannot establish
      normality -- five points give Shapiro no power to reject anything --
      and the old code read "not rejected" as "normal".
    * the n=5 group's p-value was a number. It is NaN now, with a sentence
      saying the check could not see, because a plausible-looking p beside
      "Shapiro-Wilk" is what made the defect invisible.
    """
    from spacr.plot import spacrGraph

    df = _frame(("big", "small", "tiny"),
                [_normal_values(10, 10.0), _normal_values(5, 20.0), [1.0, 2.0]])
    g = spacrGraph(df, "grp", "v1", representation="object")

    is_normal, results = g.perform_normality_tests()
    by_group = {r["Comparison"].split(" for ")[1].split(" on ")[0]: r for r in results}

    return {
        "big": (by_group["big"]["Test Name"], by_group["big"]["n"],
                by_group["big"]["Informative"]),
        "small": (by_group["small"]["Test Name"], by_group["small"]["n"],
                  by_group["small"]["Informative"],
                  bool(np.isnan(by_group["small"]["p-value"]))),
        "tiny": (by_group["tiny"]["Test Name"],
                 by_group["tiny"]["Test Statistic"],
                 by_group["tiny"]["p-value"]),
        # five replicates cannot license a parametric test
        "is_normal": is_normal,
        "said_skipping_tiny": ("Skipping normality test for group 'tiny'"
                               in capsys.readouterr().out),
    }


def _case_perform_normality_tests_flags_skewed_groups():
    from spacr.plot import spacrGraph

    df = skewed_df(("a", "b"), n=20)
    g = spacrGraph(df, "grp", "v1", representation="object")
    is_normal, results = g.perform_normality_tests()

    # LABEL ONLY: the same rejection, now reported under the name of the test
    # the one engine actually ran.
    return (is_normal, all(r["Test Name"] == "Shapiro-Wilk" for r in results),
            all(r["p-value"] < 0.05 for r in results))


def _case_degenerate_groups_are_refused_rather_than_named():
    """One value per well cannot establish normality or equal variance.

    CHANGED 2026-08-17: the comparison used to be reported as
    'Mann-Whitney U test' with a NaN p. Naming a test that never ran is the
    same defect as claiming Student's when Welch's was used, so the engine's
    refusal is reported instead, with its reason.
    """
    import warnings
    from spacr.plot import spacrGraph

    df = _frame(("a", "b"), [[1.0], [2.0]])
    graph = spacrGraph(df, "grp", "v1", representation="object")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        is_normal, _results = graph.perform_normality_tests()
        levene_stat, levene_p = graph.perform_levene_test(
            graph.df["grp"].unique())
        tests = graph.perform_statistical_tests(
            graph.df["grp"].unique(), is_normal=is_normal)

    return (is_normal, bool(np.isnan(levene_stat) and np.isnan(levene_p)),
            tests[0]["Test Name"], bool(np.isnan(tests[0]["p-value"])),
            "cannot be tested" in tests[0]["Why This Test"],
            len([warning for warning in caught
                 if issubclass(warning.category, RuntimeWarning)]))


def _case_constant_groups_do_not_license_a_t_test():
    """CHANGED 2026-08-17: was 'T-test' with a NaN statistic.

    Five wells that all read exactly the same number carry no spread, so the
    normality check has nothing to describe and the engine takes the rank
    branch. p = 1.0 on two identical arms is a true statement about them; a
    NaN under the heading 'T-test' was a test name with no test behind it.

    The half that must NOT change is the reason this test was written: scipy
    emits a RuntimeWarning on a degenerate t-test, and a plot call that
    prints one into a user's console is how the case was found.
    """
    import warnings
    from spacr.plot import spacrGraph

    df = _frame(("a", "b"), [[1.0] * 5, [1.0] * 5])
    graph = spacrGraph(df, "grp", "v1", representation="object")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = graph.perform_statistical_tests(
            graph.df["grp"].unique(), is_normal=True)

    return (results[0]["Test Name"], results[0]["p-value"],
            len([warning for warning in caught
                 if issubclass(warning.category, RuntimeWarning)]))


# ===========================================================================
# spacrGraph.perform_levene_test / perform_statistical_tests
# ===========================================================================

def _case_perform_levene_test_returns_stat_and_p():
    from spacr.plot import spacrGraph

    df = normal_df(("a", "b"), n=20)
    g = spacrGraph(df, "grp", "v1", representation="object")
    stat, p = g.perform_levene_test(g.df["grp"].unique())

    # identical (shifted) samples -> identical spread -> no variance difference
    return (bool(np.isfinite(stat)), 0.0 <= p <= 1.0, bool(p > 0.05))


def _case_paired_ttest_branch():
    from spacr.plot import spacrGraph

    df = normal_df(("a", "b"), n=20)
    # Avoid a zero-variance paired difference: it has no estimable standard
    # error and therefore is not a valid t-test fixture.
    in_b = df["grp"] == "b"
    df.loc[in_b, "v1"] += np.linspace(-0.1, 0.1, in_b.sum())
    g = spacrGraph(df, "grp", "v1", representation="object", paired=True)
    res = g.perform_statistical_tests(g.df["grp"].unique(), is_normal=True)

    # b is shifted +2 from a
    return (len(res), res[0]["Test Name"], res[0]["Column"],
            res[0]["n_object"], bool(res[0]["p-value"] < 0.05))


def _case_paired_differences_with_no_spread_are_refused():
    """CHANGED 2026-08-17: was reported as 'Paired T-test' with a NaN p.

    `normal_df` shifts group b by a constant, so every matched difference is
    the same number and the standard error of the difference is zero. Left to
    itself `spacr.figures.stats.compare(paired=True)` hands that to
    `ttest_rel`, which returns t = -inf and p = 0.0 with a RuntimeWarning --
    the strongest claim the software can make, off an input that says
    nothing. spacrGraph refuses on the engine's behalf; the engine has no
    guard of its own for the paired case and should grow one.
    """
    import warnings
    from spacr.plot import spacrGraph

    graph = spacrGraph(
        normal_df(("a", "b"), n=20), "grp", "v1",
        representation="object", paired=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = graph.perform_statistical_tests(
            graph.df["grp"].unique(), is_normal=True)[0]

    return (result["Test Name"], bool(np.isnan(result["Test Statistic"])),
            bool(np.isnan(result["p-value"])),
            "no spread" in result["Why This Test"],
            len([warning for warning in caught
                 if issubclass(warning.category, RuntimeWarning)]))


def _case_paired_wilcoxon_branch():
    from spacr.plot import spacrGraph

    df = skewed_df(("a", "b"), n=20)
    g = spacrGraph(df, "grp", "v1", representation="object", paired=True)
    res = g.perform_statistical_tests(g.df["grp"].unique(), is_normal=False)

    return (len(res), res[0]["Test Name"],
            bool(np.isfinite(float(res[0]["Test Statistic"]))),
            0.0 <= float(res[0]["p-value"]) <= 1.0)


def _case_unpaired_mannwhitney_branch():
    from spacr.plot import spacrGraph

    df = skewed_df(("a", "b"), n=20)
    g = spacrGraph(df, "grp", "v1", representation="object")
    res = g.perform_statistical_tests(g.df["grp"].unique(), is_normal=False)

    return (res[0]["Test Name"], res[0]["n_object"], res[0]["n_well"])


def _case_anova_and_kruskal_branches_for_three_groups():
    from spacr.plot import spacrGraph

    g = spacrGraph(normal_df(("a", "b", "c"), n=20), "grp", "v1",
                   representation="object")
    groups = g.df["grp"].unique()

    anova = g.perform_statistical_tests(groups, is_normal=True)
    kruskal = g.perform_statistical_tests(groups, is_normal=False)

    # three well-separated groups -> both omnibus tests are significant
    return ((anova[0]["Test Name"], kruskal[0]["Test Name"]),
            (bool(anova[0]["p-value"] < 0.01),
             bool(kruskal[0]["p-value"] < 0.01)))


def _case_statistical_tests_run_once_per_data_column():
    from spacr.plot import spacrGraph

    g = spacrGraph(multi_col_df(("a", "b"), n=12), "grp", ["v1", "v2"],
                   representation="object")
    res = g.perform_statistical_tests(g.df["grp"].unique(), is_normal=True)

    return ([r["Column"] for r in res],
            all(r["Test Name"] == "T-test" for r in res))


# ===========================================================================
# spacrGraph.perform_posthoc_tests
# ===========================================================================

def _case_posthoc_tukey_for_normal_multi_group():
    from spacr.plot import spacrGraph

    g = spacrGraph(normal_df(("a", "b", "c"), n=20), "grp", "v1",
                   representation="object")
    res = g.perform_posthoc_tests(is_normal=True,
                                  unique_groups=g.df["grp"].unique())

    return ([r["Comparison"] for r in res],
            all(r["Test Name"] == "Tukey HSD Post-hoc" for r in res),
            all(r["Test Statistic"] is None for r in res),
            all(r["n_object"] == 40 and r["n_well"] == 40 for r in res))


def _case_posthoc_dunn_for_non_normal_multi_group(capsys):
    from spacr.plot import spacrGraph

    g = spacrGraph(skewed_df(("a", "b", "c"), n=20), "grp", "v1",
                   representation="object")
    res = g.perform_posthoc_tests(is_normal=False,
                                  unique_groups=g.df["grp"].unique())

    # 3 groups -> 3 comparisons, 20 points/group -> 'holm'
    return ("performing_dunns" in capsys.readouterr().out,
            [r["Comparison"] for r in res],
            all(r["Test Name"] == "Dunn's Post-hoc" for r in res),
            all(r["p_adjust_method"] == "holm" for r in res),
            all(0.0 <= r["p-value"] <= 1.0 for r in res),
            all(r["n_object"] == 40 for r in res))


def _case_posthoc_dunn_n_well_counts_only_the_two_groups():
    from spacr.plot import spacrGraph

    g = spacrGraph(skewed_df(("a", "b", "c"), n=20), "grp", "v1",
                   representation="object")
    res = g.perform_posthoc_tests(is_normal=False,
                                  unique_groups=g.df["grp"].unique())

    return [r["n_well"] for r in res]


def _case_posthoc_returns_empty_without_all_to_all():
    from spacr.plot import spacrGraph

    g = spacrGraph(normal_df(("a", "b", "c"), n=20), "grp", "v1",
                   representation="object", all_to_all=False,
                   compare_group="a")
    return (g.perform_posthoc_tests(True, g.df["grp"].unique()),
            g.perform_posthoc_tests(False, g.df["grp"].unique()))


def _case_posthoc_returns_empty_for_two_groups():
    from spacr.plot import spacrGraph

    g = spacrGraph(normal_df(("a", "b"), n=20), "grp", "v1",
                   representation="object")
    return g.perform_posthoc_tests(True, g.df["grp"].unique())


# ===========================================================================
# spacrGraph.create_plot — one test per graph type
# ===========================================================================

def _case_create_plot_single_data_column_graph_types(graph_type):
    from spacr.plot import spacrGraph

    g = spacrGraph(normal_df(("a", "b", "c"), n=20), "grp", "v1",
                   graph_type=graph_type, representation="object")
    g.create_plot()

    fig = g.get_figure()
    assert fig is not None
    ax = fig.get_axes()[0]
    drew = len(ax.collections) + len(ax.patches) + len(ax.lines) > 0
    # stats survived onto results_df: normality + omnibus + tukey
    # (LABEL ONLY: the normality rows say Shapiro-Wilk since 2026-08-17,
    # because that is the test the one engine runs.)
    res = g.get_results()
    # the violin renderer relabels the y-axis 'Value' after setting the column;
    # the x label is cleared for non-line graphs;
    # _standerdize_figure_format enforces a square >=10 inch canvas
    return (drew, ax.get_ylabel(), ax.get_xlabel(),
            tuple(fig.get_size_inches()),
            set(res["Test Name"]) >= {"Shapiro-Wilk", "One-way ANOVA"},
            len(res))


def _case_create_plot_log_scales(graph_type):
    from spacr.plot import spacrGraph

    g = spacrGraph(normal_df(("a", "b", "c"), n=20), "grp", "v1",
                   graph_type=graph_type, representation="object",
                   log_y=True, log_x=True)
    g.create_plot()

    ax = g.get_figure().get_axes()[0]
    return (ax.get_yscale(), ax.get_xscale())


def _case_create_plot_reuses_supplied_axes():
    import matplotlib.pyplot as plt
    from spacr.plot import spacrGraph

    fig, ax = plt.subplots()
    g = spacrGraph(normal_df(("a", "b"), n=20), "grp", "v1",
                   representation="object")
    g.create_plot(ax=ax)

    return (g.fig is fig, len(ax.patches) > 0)


def _case_create_plot_rejects_unknown_graph_type():
    from spacr.plot import spacrGraph

    g = spacrGraph(normal_df(("a", "b"), n=20), "grp", "v1",
                   graph_type="pie", representation="object")
    return _refused(ValueError, "Unknown graph type: pie", g.create_plot)


def _case_create_plot_y_lim_single_element_sets_only_bottom():
    from spacr.plot import spacrGraph

    g = spacrGraph(normal_df(("a", "b", "c"), n=20), "grp", "v1",
                   representation="object", y_lim=[2.0])
    g.create_plot()
    bottom, top = g.get_figure().get_axes()[0].get_ylim()

    return (bottom, top > 2.0)


def _case_create_plot_y_lim_two_elements():
    from spacr.plot import spacrGraph

    g = spacrGraph(normal_df(("a", "b", "c"), n=20), "grp", "v1",
                   representation="object", y_lim=[1.0, 50.0])
    g.create_plot()
    return g.get_figure().get_axes()[0].get_ylim()


def _case_create_plot_multi_data_column_builds_symbol_table():
    from spacr.plot import spacrGraph

    g = spacrGraph(multi_col_df(("a", "b", "c"), n=12), "grp", ["v1", "v2"],
                   graph_type="bar", representation="object")
    g.create_plot()

    assert "Combined Group" in g.df_melted.columns

    fig = g.get_figure()
    ax = fig.get_axes()[0]
    # x ticks are removed and replaced by the +/- symbol table
    texts = [t.get_text() for t in ax.texts]
    # texts[:3] are the row labels; "+" and "-" are the symbol rows; and an
    # extra axes was added below the plot to host the legend table
    return {
        "hue": g.hue,
        "jitter_bar_dodge": g.jitter_bar_dodge,
        "combined_groups": list(g.summary_df["Combined Group"]),
        "xticks": list(ax.get_xticks()),
        "row_labels": texts[:3],
        "has_symbols": "+" in texts and "-" in texts,
        "n_axes": len(fig.axes),
    }


def _case_create_plot_multi_data_column_jitter_bar_places_symbols_per_position():
    from spacr.plot import spacrGraph

    g = spacrGraph(multi_col_df(("a", "b", "c"), n=12), "grp", ["v1", "v2"],
                   graph_type="jitter_bar", representation="object")
    g.create_plot()

    ax = g.get_figure().get_axes()[0]
    texts = [t.get_text() for t in ax.texts]
    # 3 row labels, then one 3-cell column per jitter x-position
    return (texts[:3], (len(texts) - 3) % 3, len(texts) > 3,
            texts.count("+") == texts.count("-"))


def _case_create_plot_multi_data_column_draws_one_bar_per_combined_group():
    import matplotlib.pyplot as plt
    from spacr.plot import spacrGraph

    g = spacrGraph(multi_col_df(("a", "b", "c"), n=12), "grp", ["v1", "v2"],
                   graph_type="bar", representation="object")
    g.create_plot()

    ax = g.get_figure().get_axes()[0]
    bars = [p for p in ax.patches if isinstance(p, plt.Rectangle)]
    # 3 groups x 2 columns
    return len(bars)


def _case_create_plot_multi_data_column_graph_types(graph_type):
    from spacr.plot import spacrGraph

    g = spacrGraph(multi_col_df(("a", "b"), n=12), "grp", ["v1", "v2"],
                   graph_type=graph_type, representation="object")
    g.create_plot()
    assert g.get_figure() is not None
    ax = g.get_figure().get_axes()[0]
    return (ax.get_ylabel(), [t.get_text() for t in ax.texts][:3])


# ===========================================================================
# spacrGraph line graphs
# ===========================================================================

def _case_create_plot_line_graph():
    from spacr.plot import spacrGraph

    g = spacrGraph(line_df(), "grp", ["epoch", "acc"], graph_type="line",
                   representation="object")
    g.create_plot()

    fig = g.get_figure()
    ax = fig.get_axes()[0]
    # one line per plate group
    return (ax.get_xlabel(), ax.get_ylabel(), len(ax.lines) >= 3,
            (g.fig_height, g.fig_width), len(g.summary_df))


def _case_create_plot_line_graph_log_transforms_columns():
    from spacr.plot import spacrGraph

    df = line_df(groups=("p1", "p2"), n=8)
    g = spacrGraph(df.copy(), "grp", ["epoch", "acc"], graph_type="line",
                   representation="object", log_y=True, log_x=True)
    g.create_plot()

    # the log transform is applied in place to self.df
    return (bool(np.allclose(np.sort(g.df["epoch"].unique()),
                             np.log10(np.arange(1, 9, dtype=float)))),
            bool(np.allclose(g.df["acc"].to_numpy(),
                             np.log10(df["acc"].to_numpy()))))


def _case_create_line_graph_raises_for_missing_column():
    import matplotlib.pyplot as plt
    from spacr.plot import spacrGraph

    g = spacrGraph(line_df(), "grp", ["epoch", "acc"], graph_type="line",
                   representation="object")
    g.df = g.df.drop(columns=["acc"])
    _fig, ax = plt.subplots()
    return _refused(ValueError, "Column 'acc' not found",
                    lambda: g._create_line_graph(ax))


def _case_create_plot_line_std_area():
    from spacr.plot import spacrGraph

    g = spacrGraph(line_df(), "grp", ["epoch", "acc"], graph_type="line_std",
                   representation="object")
    g.create_plot()

    ax = g.get_figure().get_axes()[0]
    # one row per epoch
    observed = (list(g.summary_df.columns), len(g.summary_df),
                ax.get_xlabel(), ax.get_ylabel())
    # fill_between adds a band spanning mean +/- std of the accuracies
    band = ax.collections[-1]
    verts = band.get_paths()[0].vertices
    lo = (g.summary_df["mean_acc"] - g.summary_df["std_mean_acc"]).min()
    hi = (g.summary_df["mean_acc"] + g.summary_df["std_mean_acc"]).max()
    assert verts[:, 1].min() == pytest.approx(lo)
    assert verts[:, 1].max() == pytest.approx(hi)
    return observed


def _case_create_plot_line_std_log_transforms():
    from spacr.plot import spacrGraph

    df = line_df(groups=("p1", "p2"), n=8)
    g = spacrGraph(df.copy(), "grp", ["epoch", "acc"], graph_type="line_std",
                   representation="object", log_y=True, log_x=True)
    g.create_plot()

    # log10 of accuracies < 1
    return (bool(np.allclose(np.sort(g.summary_df["epoch"].to_numpy()),
                             np.log10(np.arange(1, 9, dtype=float)))),
            bool((g.summary_df["mean_acc"] < 0).all()))


# ===========================================================================
# spacrGraph._standerdize_figure_format
# ===========================================================================

def _case_standerdize_figure_format_skips_line_graphs(capsys):
    import matplotlib.pyplot as plt
    from spacr.plot import spacrGraph

    g = spacrGraph(normal_df(("a", "b"), n=20), "grp", "v1",
                   representation="object")
    fig, ax = plt.subplots(figsize=(3, 3))
    returned = g._standerdize_figure_format(ax=ax, num_groups=2,
                                            graph_type="line")

    return (returned,
            "Skipping layout adjustment for line graphs."
            in capsys.readouterr().out,
            tuple(fig.get_size_inches()))      # untouched


def _case_standerdize_figure_format_resizes_box_artists():
    """The box/violin branch resizes every Axes-level artist."""
    import matplotlib.pyplot as plt
    from matplotlib.artist import Artist
    from spacr.plot import spacrGraph

    class _WidthArtist(Artist):
        def __init__(self):
            super().__init__()
            self.width = None

        def set_width(self, w):
            self.width = w

        def draw(self, renderer):
            return None

    g = spacrGraph(normal_df(("a", "b", "c"), n=20), "grp", "v1",
                   representation="object")
    _fig, ax = plt.subplots()
    artist = _WidthArtist()
    ax.add_artist(artist)
    assert len(ax.artists) == 1

    g._standerdize_figure_format(ax=ax, num_groups=3, graph_type="box")

    return (artist.width, ax.get_xlim())


def _case_standerdize_figure_format_shrinks_bars_and_moves_legend():
    import matplotlib.pyplot as plt
    from spacr.plot import spacrGraph

    g = spacrGraph(normal_df(("a", "b", "c", "d"), n=20), "grp", "v1",
                   representation="object")
    _fig, ax = plt.subplots()
    bar = ax.bar([0, 1, 2, 3], [1, 2, 3, 4], width=0.8)[0]
    ax.plot([0, 1], [0, 1], label="dummy")
    ax.legend()

    g._standerdize_figure_format(ax=ax, num_groups=4, graph_type="bar")

    return (bar.get_width(), ax.get_legend() is not None)


def _case_standerdize_figure_format_shifts_jitter_collections():
    import matplotlib.pyplot as plt
    from spacr.plot import spacrGraph

    g = spacrGraph(normal_df(("a", "b", "c"), n=20), "grp", "v1",
                   representation="object")
    _fig, ax = plt.subplots()
    coll = ax.scatter([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])
    before = coll.get_offsets().copy()

    g._standerdize_figure_format(ax=ax, num_groups=3, graph_type="jitter")

    after = coll.get_offsets()
    return ((after[:, 0] - np.asarray(before)[:, 0]).tolist(),
            sorted(set(float(size) for size in coll.get_sizes())))


# ===========================================================================
# spacrGraph._save_results
# ===========================================================================

def _case_save_results_writes_figure_stats_and_data(tmp_path, capsys):
    from spacr.plot import spacrGraph

    out = tmp_path / "results"
    g = spacrGraph(normal_df(("a", "b", "c"), n=20), "grp", "v1",
                   graph_type="bar", representation="object",
                   save=True, output_dir=str(out), graph_name="mygraph")
    g.create_plot()

    stem = "mygraph_v1_grp_bar"
    assert (out / f"{stem}.pdf").is_file()
    assert (out / f"{stem}.pdf").stat().st_size > 0

    stats = pd.read_csv(out / f"{stem}_stats.csv")
    assert len(stats) == len(g.get_results())
    data = pd.read_csv(out / f"{stem}_data.csv")
    summary = pd.read_csv(out / f"{stem}_summary.csv")

    printed = capsys.readouterr().out
    return (g.results_name, len(data), list(summary["grp"]),
            "Plot  ->" in printed and "Stats ->" in printed
            and "Data ->" in printed)


def _case_save_results_without_summary_df(tmp_path):
    """_save_results tolerates being called before any renderer set summary_df."""
    import matplotlib.pyplot as plt
    from spacr.plot import spacrGraph

    out = tmp_path / "nosummary"
    g = spacrGraph(normal_df(("a", "b"), n=20), "grp", "v1",
                   representation="object", output_dir=str(out),
                   graph_name="bare")
    g.fig = plt.figure()
    g.results_df = pd.DataFrame([{"Comparison": "a vs b", "p-value": 0.1}])
    g._save_results()

    # THE OBJECT'S OWN NAME, not a hard-coded one. The stem ends in the
    # graph type, so spelling it out here made this a test of what the
    # DEFAULT graph type happens to be -- and it failed the moment that
    # default moved from a bar to a box with jitter, which is a statistical
    # correction and not a change to `_save_results` at all.
    stem = g.results_name
    return (stem.startswith("bare_v1_grp_"),
            (out / f"{stem}.pdf").is_file(),
            (out / f"{stem}_stats.csv").is_file(),
            (out / f"{stem}_summary.csv").exists())


def _case_a_p_value_that_is_not_a_number_has_no_answer():
    """A results row can carry a value no float can be made of.

    A skipped test writes ``None``; a refused one can write a string, and a
    column read back from CSV can hold anything the file did. ``None`` is the
    honest answer for all of them, and it is what stops a significance bar
    being drawn over a comparison that was never made.

    The alternative is worse than it sounds: ``float("n/a")`` raises inside
    the plotting call, so a single unparseable cell takes down the whole
    figure rather than omitting one bracket.
    """
    from spacr.plot import _finite_p_value

    return [_finite_p_value(value)
            for value in (None, "not a number", [0.01], {})]


def _case_a_non_finite_p_value_has_no_answer_either():
    """``nan`` is what a refused test writes, and it is not a small number.

    ``nan < 0.05`` is False, so a NaN would silently read as "not
    significant" rather than as "not tested" -- the same figure, with a
    comparison quietly downgraded instead of omitted.
    """
    import numpy as np

    from spacr.plot import _finite_p_value

    return [_finite_p_value(value)
            for value in (float("nan"), np.nan, float("inf"), float("-inf"))]


def _case_a_real_p_value_comes_back_as_a_float():
    """Otherwise the four refusals above would pass on a function that only
    ever returns None."""
    from spacr.plot import _finite_p_value

    answers = [_finite_p_value(value) for value in (0.01, "0.04", 0, 1)]
    return (answers, type(answers[-1]).__name__)


def _case_significance_marker_boundaries():
    """All conventional p-value bands, including the strongest, are stable."""
    from spacr.plot import _significance_marker

    return [_significance_marker(p) for p in (0.001, 0.01, 0.05, 0.051)]


# ===========================================================================
# the tests: one per contract, rows are (helper, expected)
# ===========================================================================

_DEEP = sns.color_palette("deep", 100)

_GROUPED_PLOT_ROWS = [
    pytest.param(
        _case_a_grouped_plot_does_not_restyle_the_rest_of_the_session,
        (True, []), id="grouped-plot-leaves-no-seaborn-theme-behind"),
    pytest.param(
        _case_create_grouped_plot_two_normal_groups_uses_ttest_and_explicit_order,
        ({"Normality test", "T-test"}, ["b vs a"], ["a", "b"]),
        id="two-normal-groups-ttest-and-explicit-order"),
    pytest.param(
        _case_create_grouped_plot_two_skewed_groups_uses_mann_whitney,
        (True, False, 1), id="two-skewed-groups-mann-whitney"),
    pytest.param(
        _case_create_grouped_plot_three_skewed_groups_use_a_pairwise_rank_test,
        (3, False), id="three-skewed-groups-pairwise-rank-no-tukey"),
    pytest.param(
        _case_create_grouped_plot_three_normal_groups_adds_tukey_posthoc,
        (3, 3), id="three-normal-groups-pairwise-ttest-plus-tukey"),
    pytest.param(
        _case_create_grouped_plot_sem_errorbars_custom_colors_and_ylim,
        ((0.0, 30.0), True, 3), id="sem-errorbars-colors-and-ylim"),
    pytest.param(
        _case_create_grouped_plot_rejects_unknown_error_bar_type,
        "Invalid error_bar_type", id="unknown-error-bar-type-refused-by-name"),
    pytest.param(
        _case_create_grouped_plot_saves_plot_and_stats,
        (1, True, ["Comparison", "Test Statistic", "p-value", "Test Name"]),
        id="saves-plot-and-stats-csv"),
] + [
    pytest.param(_bind(_case_create_grouped_plot_other_graph_types, kind),
                 (True, {"a", "b", "c"}, 9), id=f"{kind}-draws-with-stats-rows")
    for kind in ("violin", "jitter", "box", "jitter_box")
]


@pytest.mark.parametrize("helper, expected", _GROUPED_PLOT_ROWS)
def test_create_grouped_plot_chooses_tests_draws_and_saves(
        request, helper, expected):
    observed = _call(request, helper)
    assert observed == expected


_SETUP_ROWS = [
    pytest.param(
        _case_set_reordered_theme_without_order_returns_palette_untouched,
        list(sns.color_palette("deep", 5)),
        id="theme-without-order-returns-palette-untouched"),
    pytest.param(
        _case_set_theme_reorders_palette,
        [_DEEP[i] for i in [7, 9, 4, 0, 3, 6, 2]],
        id="theme-with-default-order-reorders-palette"),
    pytest.param(
        _case_preprocess_object_representation_does_not_aggregate,
        (40, ["a", "b"]), id="object-representation-does-not-aggregate"),
    pytest.param(
        _case_preprocess_well_representation_aggregates_per_prc,
        (["grp", "prc", "v1"], 24), id="well-representation-per-prc"),
    pytest.param(
        _case_preprocess_plate_representation_splits_prc,
        (["grp", "plateID", "v1"], 4, ["plate1", "plate2"]),
        id="plate-representation-splits-prc"),
    pytest.param(
        _case_preprocess_plate_representation_grouping_on_plate_id,
        (["plateID", "v1"], 2), id="plate-representation-grouped-on-plate-id"),
    pytest.param(
        _case_preprocess_plate_representation_without_prc_raises_keyerror,
        "cannot split from 'prc'", id="plate-without-prc-refused-by-name"),
    pytest.param(
        _case_preprocess_unknown_representation_raises_valueerror,
        "Unknown representation: sample",
        id="unknown-representation-refused-by-name"),
    pytest.param(
        _case_preprocess_falls_back_to_sorted_categories_when_order_empty,
        ([], 0, True, []), id="all-nan-grouping-gives-empty-categorical"),
    pytest.param(
        _case_remove_outliers_from_plot_keeps_inliers,
        (22, 11.0, ["a", "b"]), id="remove-outliers-keeps-inliers"),
    pytest.param(
        _case_create_plot_with_remove_outliers_still_plots,
        22, id="create-plot-with-remove-outliers-still-plots"),
]


@pytest.mark.parametrize("helper, expected", _SETUP_ROWS)
def test_spacrgraph_theme_preprocessing_and_outlier_removal(
        request, helper, expected):
    observed = _call(request, helper)
    assert observed == expected


_TEST_ROWS = [
    pytest.param(
        _case_perform_normality_tests_reports_shapiro_and_its_power,
        {"big": ("Shapiro-Wilk", 10, True),
         "small": ("Shapiro-Wilk", 5, False, True),
         "tiny": ("Skipped", None, None),
         "is_normal": False, "said_skipping_tiny": True},
        id="normality-reports-shapiro-and-its-power"),
    pytest.param(_case_perform_normality_tests_flags_skewed_groups,
                 (False, True, True), id="normality-flags-skewed-groups"),
    pytest.param(_case_degenerate_groups_are_refused_rather_than_named,
                 (False, True, "not testable", True, True, 0),
                 id="degenerate-groups-refused-not-named"),
    pytest.param(_case_constant_groups_do_not_license_a_t_test,
                 ("Mann-Whitney U test", pytest.approx(1.0), 0),
                 id="constant-groups-do-not-license-a-t-test"),
    pytest.param(_case_perform_levene_test_returns_stat_and_p,
                 (True, True, True), id="levene-returns-stat-and-p"),
    pytest.param(_case_paired_ttest_branch,
                 (1, "Paired T-test", "v1", 40, True), id="paired-ttest"),
    pytest.param(_case_paired_differences_with_no_spread_are_refused,
                 ("not testable", True, True, True, 0),
                 id="paired-differences-with-no-spread-refused"),
    pytest.param(_case_paired_wilcoxon_branch,
                 (1, "Paired Wilcoxon test", True, True),
                 id="paired-wilcoxon"),
    pytest.param(_case_unpaired_mannwhitney_branch,
                 ("Mann-Whitney U test", 40, 40), id="unpaired-mann-whitney"),
    pytest.param(_case_anova_and_kruskal_branches_for_three_groups,
                 (("One-way ANOVA", "Kruskal-Wallis test"), (True, True)),
                 id="three-groups-anova-and-kruskal"),
    pytest.param(_case_statistical_tests_run_once_per_data_column,
                 (["v1", "v2"], True), id="one-test-per-data-column"),
    pytest.param(_case_posthoc_tukey_for_normal_multi_group,
                 (["a vs b", "a vs c", "b vs c"], True, True, True),
                 id="posthoc-tukey-for-normal-groups"),
    pytest.param(_case_posthoc_dunn_for_non_normal_multi_group,
                 (True, ["a vs b", "a vs c", "b vs c"], True, True, True, True),
                 id="posthoc-dunn-for-non-normal-groups"),
    pytest.param(_case_posthoc_dunn_n_well_counts_only_the_two_groups,
                 [40, 40, 40], id="posthoc-dunn-n-well-counts-two-groups"),
    pytest.param(_case_posthoc_returns_empty_without_all_to_all,
                 ([], []), id="posthoc-empty-without-all-to-all"),
    pytest.param(_case_posthoc_returns_empty_for_two_groups,
                 [], id="posthoc-empty-for-two-groups"),
    pytest.param(_case_significance_marker_boundaries,
                 ["***", "**", "*", "ns"], id="significance-marker-bands"),
    pytest.param(_case_a_p_value_that_is_not_a_number_has_no_answer,
                 [None, None, None, None],
                 id="p-value-not-a-number-has-no-answer"),
    pytest.param(_case_a_non_finite_p_value_has_no_answer_either,
                 [None, None, None, None],
                 id="non-finite-p-value-has-no-answer"),
    pytest.param(_case_a_real_p_value_comes_back_as_a_float,
                 ([0.01, 0.04, 0.0, 1.0], "float"),
                 id="real-p-value-comes-back-as-a-float"),
]


@pytest.mark.parametrize("helper, expected", _TEST_ROWS)
def test_statistical_tests_are_chosen_and_reported_honestly(
        request, helper, expected):
    observed = _call(request, helper)
    assert observed == expected


_ALL_SINGLE = ("bar", "jitter", "box", "violin", "jitter_box", "jitter_bar")

_CREATE_PLOT_ROWS = [
    pytest.param(
        _bind(_case_create_plot_single_data_column_graph_types, kind),
        (True, "Value" if kind == "violin" else "v1", "", (10.0, 10.0), True,
         7),
        id=f"{kind}-single-column-draws-with-stats")
    for kind in _ALL_SINGLE
] + [
    pytest.param(_bind(_case_create_plot_log_scales, kind), ("log", "log"),
                 id=f"{kind}-takes-log-scales")
    for kind in _ALL_SINGLE
] + [
    pytest.param(_case_create_plot_reuses_supplied_axes, (True, True),
                 id="reuses-supplied-axes"),
    pytest.param(_case_create_plot_rejects_unknown_graph_type,
                 "Unknown graph type: pie", id="unknown-graph-type-refused"),
    pytest.param(_case_create_plot_y_lim_single_element_sets_only_bottom,
                 (2.0, True), id="y-lim-single-element-sets-bottom"),
    pytest.param(_case_create_plot_y_lim_two_elements, (1.0, 50.0),
                 id="y-lim-two-elements"),
    pytest.param(
        _case_create_plot_multi_data_column_builds_symbol_table,
        {"hue": "Data Column", "jitter_bar_dodge": True,
         "combined_groups": ["a - v1", "a - v2", "b - v1", "b - v2",
                             "c - v1", "c - v2"],
         "xticks": [], "row_labels": ["grp", "v1", "v2"],
         "has_symbols": True, "n_axes": 2},
        id="multi-column-bar-builds-symbol-table"),
    pytest.param(
        _case_create_plot_multi_data_column_jitter_bar_places_symbols_per_position,
        (["grp", "v1", "v2"], 0, True, True),
        id="multi-column-jitter-bar-symbols-per-position"),
    pytest.param(
        _case_create_plot_multi_data_column_draws_one_bar_per_combined_group,
        6, id="multi-column-bar-one-bar-per-combined-group"),
    pytest.param(
        _bind(_case_create_plot_multi_data_column_graph_types, "violin"),
        ("Value", ["grp", "v1", "v2"]), id="multi-column-violin"),
    pytest.param(
        _bind(_case_create_plot_multi_data_column_graph_types, "jitter"),
        ("Value", ["grp", "v1", "v2"]), id="multi-column-jitter"),
    pytest.param(
        _bind(_case_create_plot_multi_data_column_graph_types, "box"),
        ("Value", ["grp", "v1", "v2"]), id="snap-multi-column-box"),
    pytest.param(
        _bind(_case_create_plot_multi_data_column_graph_types, "jitter_box"),
        ("Value", ["grp", "v1", "v2"]), id="snap-multi-column-jitter-box"),
]


@pytest.mark.parametrize("helper, expected", _CREATE_PLOT_ROWS)
def test_create_plot_draws_each_graph_type_with_its_stats(
        request, helper, expected):
    observed = _call(request, helper)
    assert observed == expected


_LINE_LAYOUT_SAVE_ROWS = [
    pytest.param(_case_create_plot_line_graph,
                 ("epoch", "acc", True, (10, 10), 24), id="line-graph"),
    pytest.param(_case_create_plot_line_graph_log_transforms_columns,
                 (True, True), id="line-graph-log-transforms-columns"),
    pytest.param(_case_create_line_graph_raises_for_missing_column,
                 "Column 'acc' not found",
                 id="line-graph-missing-column-refused"),
    pytest.param(_case_create_plot_line_std_area,
                 (["epoch", "mean_acc", "std_mean_acc"], 8, "epoch", "acc"),
                 id="line-std-area"),
    pytest.param(_case_create_plot_line_std_log_transforms, (True, True),
                 id="line-std-log-transforms"),
    pytest.param(_case_standerdize_figure_format_skips_line_graphs,
                 (None, True, (3.0, 3.0)), id="layout-skips-line-graphs"),
    pytest.param(_case_standerdize_figure_format_resizes_box_artists,
                 (pytest.approx(0.125), (-0.5, 2.5)),
                 id="layout-resizes-box-artists"),
    pytest.param(_case_standerdize_figure_format_shrinks_bars_and_moves_legend,
                 (pytest.approx(0.09375), True),
                 id="layout-shrinks-bars-keeps-legend"),
    pytest.param(_case_standerdize_figure_format_shifts_jitter_collections,
                 ([pytest.approx(1 / 60)] * 3, [200.0]),
                 id="layout-shifts-jitter-collections"),
    pytest.param(_case_save_results_writes_figure_stats_and_data,
                 ("mygraph_v1_grp_bar", 60, ["a", "b", "c"], True),
                 id="save-results-writes-figure-stats-and-data"),
    pytest.param(_case_save_results_without_summary_df,
                 (True, True, True, False),
                 id="save-results-without-summary-df"),
]


@pytest.mark.parametrize("helper, expected", _LINE_LAYOUT_SAVE_ROWS)
def test_line_graphs_layout_and_saved_results(request, helper, expected):
    observed = _call(request, helper)
    assert observed == expected
