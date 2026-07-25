"""Branch coverage for ``spacr.plot.create_grouped_plot`` and ``spacr.plot.spacrGraph``.

Everything here is CPU-only, offline and deterministic: group data is built
from normal quantiles (``norm.ppf`` on an evenly spaced grid) so the
D'Agostino / Shapiro normality verdicts are fixed rather than sampled, and
the log-transform of the same grid gives a reproducibly *non*-normal group.

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
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("seaborn")
pytest.importorskip("statsmodels")


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


# ===========================================================================
# create_grouped_plot
# ===========================================================================

def test_create_grouped_plot_two_normal_groups_uses_ttest_and_explicit_order():
    """order= branch + 2-group normal branch (T-test)."""
    from spacr.plot import create_grouped_plot

    df = normal_df(groups=("b", "a"), n=20)
    fig, results = create_grouped_plot(
        df, grouping_column="grp", data_column="v1",
        graph_type="bar", order=["a", "b"], save=False)

    assert fig is not None
    names = set(results["Test Name"])
    assert names == {"Normality test", "T-test"}
    # exactly one pairwise comparison for two groups
    assert list(results.loc[results["Test Name"] == "T-test", "Comparison"]) == ["b vs a"]
    # the explicit order drives the x-axis even though 'b' comes first in df
    ax = fig.get_axes()[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b"]


def test_create_grouped_plot_two_skewed_groups_uses_mann_whitney():
    """2-group non-normal branch (Mann-Whitney U)."""
    from spacr.plot import create_grouped_plot

    df = skewed_df(groups=("a", "b"), n=20)
    _fig, results = create_grouped_plot(
        df, grouping_column="grp", data_column="v1", graph_type="bar")

    assert "Mann-Whitney U test" in set(results["Test Name"])
    assert "T-test" not in set(results["Test Name"])
    pw = results[results["Test Name"] == "Mann-Whitney U test"]
    assert len(pw) == 1
    assert 0.0 <= float(pw["p-value"].iloc[0]) <= 1.0


def test_create_grouped_plot_three_skewed_groups_uses_kruskal():
    """>2-group non-normal branch (Kruskal-Wallis) + no Tukey post-hoc."""
    from spacr.plot import create_grouped_plot

    df = skewed_df(groups=("a", "b", "c"), n=20)
    _fig, results = create_grouped_plot(
        df, grouping_column="grp", data_column="v1", graph_type="bar")

    assert "Kruskal-Wallis test" in set(results["Test Name"])
    # 3 groups -> 3 pairwise comparisons
    assert (results["Test Name"] == "Kruskal-Wallis test").sum() == 3
    # Tukey post-hoc is only added for the normal case
    assert "Tukey HSD Post-hoc" not in set(results["Test Name"])


def test_create_grouped_plot_three_normal_groups_adds_tukey_posthoc():
    from spacr.plot import create_grouped_plot

    df = normal_df(groups=("a", "b", "c"), n=20)
    _fig, results = create_grouped_plot(
        df, grouping_column="grp", data_column="v1", graph_type="bar")

    assert (results["Test Name"] == "One-way ANOVA").sum() == 3
    assert (results["Test Name"] == "Tukey HSD Post-hoc").sum() == 3


def test_create_grouped_plot_sem_errorbars_custom_colors_and_ylim():
    """sem error-bar branch + explicit colors branch + y_lim branch."""
    import matplotlib.pyplot as plt
    from spacr.plot import create_grouped_plot

    df = normal_df(groups=("a", "b", "c"), n=20)
    colors = ["#ff0000", "#00ff00", "#0000ff"]
    fig, results = create_grouped_plot(
        df, grouping_column="grp", data_column="v1", graph_type="bar",
        colors=colors, error_bar_type="sem", y_lim=[0, 30])

    ax = fig.get_axes()[0]
    assert ax.get_ylim() == (0.0, 30.0)
    assert not results.empty
    # three bars, one per group, painted with the requested colours
    bars = [p for p in ax.patches if isinstance(p, plt.Rectangle)]
    assert len(bars) == 3


def test_create_grouped_plot_rejects_unknown_error_bar_type():
    from spacr.plot import create_grouped_plot

    df = normal_df(groups=("a", "b"), n=20)
    with pytest.raises(ValueError, match="Invalid error_bar_type"):
        create_grouped_plot(df, grouping_column="grp", data_column="v1",
                            graph_type="bar", error_bar_type="iqr")


@pytest.mark.parametrize("graph_type", ["violin", "jitter", "box", "jitter_box"])
def test_create_grouped_plot_other_graph_types(graph_type):
    """The violin / jitter / box / jitter_box rendering branches."""
    from spacr.plot import create_grouped_plot

    df = normal_df(groups=("a", "b", "c"), n=20)
    fig, results = create_grouped_plot(
        df, grouping_column="grp", data_column="v1", graph_type=graph_type)

    ax = fig.get_axes()[0]
    # every renderer must have put *something* on the axes
    assert len(ax.collections) + len(ax.patches) + len(ax.lines) > 0
    assert set(t.get_text() for t in ax.get_xticklabels()) == {"a", "b", "c"}
    assert len(results) == 3 + 3 + 3  # normality + pairwise + tukey


def test_create_grouped_plot_saves_plot_and_stats(tmp_path):
    from spacr.plot import create_grouped_plot

    df = normal_df(groups=("a", "b"), n=20)
    _fig, results = create_grouped_plot(
        df, grouping_column="grp", data_column="v1", graph_type="bar",
        output_dir=str(tmp_path), save=True)

    assert (tmp_path / "grouped_plot.png").is_file()
    csv = tmp_path / "test_results.csv"
    assert csv.is_file()
    on_disk = pd.read_csv(csv)
    assert len(on_disk) == len(results)
    assert list(on_disk.columns) == ["Comparison", "Test Statistic", "p-value", "Test Name"]


# ===========================================================================
# spacrGraph — theme helpers
# ===========================================================================

def test_set_reordered_theme_without_order_returns_palette_untouched(monkeypatch):
    """order=None branch + show_theme branch of _set_reordered_theme."""
    import seaborn as sns
    import matplotlib.pyplot as plt
    from spacr.plot import spacrGraph

    calls = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: calls.append(1))

    g = spacrGraph(normal_df(("a", "b"), n=20), "grp", "v1")
    palette = g._set_reordered_theme(theme="deep", order=None, n_colors=5,
                                     show_theme=True)

    assert palette == list(sns.color_palette("deep", 5))
    assert calls == [1]  # show_theme actually rendered the palette


def test_set_theme_reorders_palette():
    import seaborn as sns
    from spacr.plot import spacrGraph

    g = spacrGraph(normal_df(("a", "b"), n=20), "grp", "v1", theme="deep")
    base = sns.color_palette("deep", 100)
    assert g.sns_palette[:7] == [base[i] for i in [7, 9, 4, 0, 3, 6, 2]]


# ===========================================================================
# spacrGraph.preprocess_data
# ===========================================================================

def test_preprocess_object_representation_does_not_aggregate():
    from spacr.plot import spacrGraph

    df = normal_df(("a", "b"), n=20)
    g = spacrGraph(df, "grp", "v1", representation="object")
    assert len(g.df) == len(df)  # one row per object, no group-by
    assert list(g.df["grp"].cat.categories) == ["a", "b"]


def test_preprocess_well_representation_aggregates_per_prc():
    from spacr.plot import spacrGraph

    df = normal_df(("a", "b"), n=20)
    g = spacrGraph(df, "grp", "v1", representation="well")
    assert set(g.df.columns) == {"prc", "grp", "v1"}
    assert len(g.df) == len(df.groupby(["prc", "grp"], observed=True))
    assert len(g.df) < len(df)


def test_preprocess_plate_representation_splits_prc():
    from spacr.plot import spacrGraph

    df = normal_df(("a", "b"), n=20)
    g = spacrGraph(df, "grp", "v1", representation="plate")
    assert set(g.df.columns) == {"plateID", "grp", "v1"}
    # prc encodes 2 plates x 2 groups
    assert len(g.df) == 4
    assert set(g.df["plateID"]) == {"plate1", "plate2"}


def test_preprocess_plate_representation_grouping_on_plate_id():
    """grouping_column == 'plateID' collapses group_cols to a single column."""
    from spacr.plot import spacrGraph

    df = normal_df(("a", "b"), n=20)
    df["plateID"] = ["plate1" if i % 2 else "plate2" for i in range(len(df))]
    g = spacrGraph(df, "plateID", "v1", representation="plate")
    assert list(g.df.columns) == ["plateID", "v1"]
    assert len(g.df) == 2


def test_preprocess_plate_representation_without_prc_raises_keyerror():
    from spacr.plot import spacrGraph

    df = normal_df(("a", "b"), n=20).drop(columns=["prc"])
    with pytest.raises(KeyError, match="cannot split from 'prc'"):
        spacrGraph(df, "grp", "v1", representation="plate")


def test_preprocess_unknown_representation_raises_valueerror():
    from spacr.plot import spacrGraph

    with pytest.raises(ValueError, match="Unknown representation: sample"):
        spacrGraph(normal_df(("a", "b"), n=20), "grp", "v1",
                   representation="sample")


def test_preprocess_falls_back_to_sorted_categories_when_order_empty():
    """All-NaN grouping column -> self.order == [] -> default-category branch."""
    from pandas.api.types import CategoricalDtype
    from spacr.plot import spacrGraph

    df = pd.DataFrame({"grp": [np.nan] * 6, "v1": np.arange(6, dtype=float)})
    g = spacrGraph(df, "grp", "v1")

    assert g.order == []
    assert len(g.df) == 0                       # every row dropped by dropna
    assert isinstance(g.df["grp"].dtype, CategoricalDtype)
    assert list(g.df["grp"].cat.categories) == []


# ===========================================================================
# spacrGraph.remove_outliers_from_plot  (BUG)
# ===========================================================================

@pytest.mark.xfail(strict=True, reason=(
    "BUG: remove_outliers_from_plot ANDs a Series mask with a DataFrame mask "
    "(self.data_column is a list), so the resulting index covers every row and "
    "drop() empties the frame instead of removing only the outliers."))
def test_remove_outliers_from_plot_keeps_inliers():
    from spacr.plot import spacrGraph

    # 11 evenly spaced inliers per group (no natural 1.5*IQR outlier) + 1 spike
    base = list(np.linspace(1.0, 11.0, 11))
    df = _frame(("a", "b"), [base + [1000.0], base + [2000.0]])
    g = spacrGraph(df, "grp", "v1", representation="object")

    filtered = g.remove_outliers_from_plot()

    assert len(filtered) == len(df) - 2          # only the two spikes go
    assert filtered["v1"].max() <= 11.0
    assert set(filtered["grp"]) == {"a", "b"}


@pytest.mark.xfail(strict=True, reason=(
    "BUG: create_plot(remove_outliers=True) empties self.df (see "
    "remove_outliers_from_plot) and then Levene's test raises "
    "'Must enter at least two input sample vectors.'"))
def test_create_plot_with_remove_outliers_still_plots():
    from spacr.plot import spacrGraph

    base = list(np.linspace(1.0, 11.0, 11))
    df = _frame(("a", "b"), [base + [1000.0], [v + 2 for v in base] + [2000.0]])
    g = spacrGraph(df, "grp", "v1", representation="object", remove_outliers=True)
    g.create_plot()

    assert g.get_figure() is not None
    assert len(g.df) == len(df) - 2


# ===========================================================================
# spacrGraph.perform_normality_tests
# ===========================================================================

def test_perform_normality_tests_picks_test_by_sample_size(capsys):
    """n>=8 -> D'Agostino, 3<=n<8 -> Shapiro, n<3 -> skipped."""
    from spacr.plot import spacrGraph

    df = _frame(("big", "small", "tiny"),
                [_normal_values(10, 10.0), _normal_values(5, 20.0), [1.0, 2.0]])
    g = spacrGraph(df, "grp", "v1", representation="object")

    is_normal, results = g.perform_normality_tests()
    by_group = {r["Comparison"].split(" for ")[1].split(" on ")[0]: r for r in results}

    assert by_group["big"]["Test Name"] == "D'Agostino-Pearson test"
    assert by_group["big"]["n"] == 10
    assert by_group["small"]["Test Name"] == "Shapiro-Wilk test"
    assert by_group["small"]["n"] == 5
    assert by_group["tiny"]["Test Name"] == "Skipped"
    assert by_group["tiny"]["Test Statistic"] is None
    assert by_group["tiny"]["p-value"] is None
    assert is_normal is True
    assert "Skipping normality test for group 'tiny'" in capsys.readouterr().out


def test_perform_normality_tests_flags_skewed_groups():
    from spacr.plot import spacrGraph

    df = skewed_df(("a", "b"), n=20)
    g = spacrGraph(df, "grp", "v1", representation="object")
    is_normal, results = g.perform_normality_tests()

    assert is_normal is False
    assert all(r["Test Name"] == "D'Agostino-Pearson test" for r in results)
    assert all(r["p-value"] < 0.05 for r in results)


# ===========================================================================
# spacrGraph.perform_levene_test / perform_statistical_tests
# ===========================================================================

def test_perform_levene_test_returns_stat_and_p():
    from spacr.plot import spacrGraph

    df = normal_df(("a", "b"), n=20)
    g = spacrGraph(df, "grp", "v1", representation="object")
    stat, p = g.perform_levene_test(g.df["grp"].unique())

    assert np.isfinite(stat)
    assert 0.0 <= p <= 1.0
    # identical (shifted) samples -> identical spread -> no variance difference
    assert p > 0.05


def test_paired_ttest_branch():
    from spacr.plot import spacrGraph

    df = normal_df(("a", "b"), n=20)
    g = spacrGraph(df, "grp", "v1", representation="object", paired=True)
    res = g.perform_statistical_tests(g.df["grp"].unique(), is_normal=True)

    assert len(res) == 1
    assert res[0]["Test Name"] == "Paired T-test"
    assert res[0]["Column"] == "v1"
    assert res[0]["n_object"] == 40
    assert res[0]["p-value"] < 0.05  # b is shifted +2 from a


@pytest.mark.xfail(strict=True, reason=(
    "BUG: perform_statistical_tests reads pg.wilcoxon(...)[['T','p-val']] but "
    "pingouin names the statistic 'W-val', so the paired non-normal branch "
    "raises KeyError: \"['T'] not in index\"."))
def test_paired_wilcoxon_branch():
    from spacr.plot import spacrGraph

    df = skewed_df(("a", "b"), n=20)
    g = spacrGraph(df, "grp", "v1", representation="object", paired=True)
    res = g.perform_statistical_tests(g.df["grp"].unique(), is_normal=False)

    assert len(res) == 1
    assert res[0]["Test Name"] == "Paired Wilcoxon test"
    assert np.isfinite(float(res[0]["Test Statistic"]))
    assert 0.0 <= float(res[0]["p-value"]) <= 1.0


def test_unpaired_mannwhitney_branch():
    from spacr.plot import spacrGraph

    df = skewed_df(("a", "b"), n=20)
    g = spacrGraph(df, "grp", "v1", representation="object")
    res = g.perform_statistical_tests(g.df["grp"].unique(), is_normal=False)

    assert res[0]["Test Name"] == "Mann-Whitney U test"
    assert res[0]["n_object"] == 40
    assert res[0]["n_well"] == 40


def test_anova_and_kruskal_branches_for_three_groups():
    from spacr.plot import spacrGraph

    g = spacrGraph(normal_df(("a", "b", "c"), n=20), "grp", "v1",
                   representation="object")
    groups = g.df["grp"].unique()

    anova = g.perform_statistical_tests(groups, is_normal=True)
    kruskal = g.perform_statistical_tests(groups, is_normal=False)

    assert anova[0]["Test Name"] == "One-way ANOVA"
    assert kruskal[0]["Test Name"] == "Kruskal-Wallis test"
    # three well-separated groups -> both omnibus tests are significant
    assert anova[0]["p-value"] < 0.01
    assert kruskal[0]["p-value"] < 0.01


def test_statistical_tests_run_once_per_data_column():
    from spacr.plot import spacrGraph

    g = spacrGraph(multi_col_df(("a", "b"), n=12), "grp", ["v1", "v2"],
                   representation="object")
    res = g.perform_statistical_tests(g.df["grp"].unique(), is_normal=True)

    assert [r["Column"] for r in res] == ["v1", "v2"]
    assert all(r["Test Name"] == "T-test" for r in res)


# ===========================================================================
# spacrGraph.perform_posthoc_tests
# ===========================================================================

def test_posthoc_tukey_for_normal_multi_group():
    from spacr.plot import spacrGraph

    g = spacrGraph(normal_df(("a", "b", "c"), n=20), "grp", "v1",
                   representation="object")
    res = g.perform_posthoc_tests(is_normal=True,
                                  unique_groups=g.df["grp"].unique())

    assert [r["Comparison"] for r in res] == ["a vs b", "a vs c", "b vs c"]
    assert all(r["Test Name"] == "Tukey HSD Post-hoc" for r in res)
    assert all(r["Test Statistic"] is None for r in res)
    assert all(r["n_object"] == 40 and r["n_well"] == 40 for r in res)


def test_posthoc_dunn_for_non_normal_multi_group(capsys):
    from spacr.plot import spacrGraph

    g = spacrGraph(skewed_df(("a", "b", "c"), n=20), "grp", "v1",
                   representation="object")
    res = g.perform_posthoc_tests(is_normal=False,
                                  unique_groups=g.df["grp"].unique())

    assert "performing_dunns" in capsys.readouterr().out
    assert [r["Comparison"] for r in res] == ["a vs b", "a vs c", "b vs c"]
    assert all(r["Test Name"] == "Dunn's Post-hoc" for r in res)
    # 3 groups -> 3 comparisons, 20 points/group -> 'holm'
    assert all(r["p_adjust_method"] == "holm" for r in res)
    assert all(0.0 <= r["p-value"] <= 1.0 for r in res)
    assert all(r["n_object"] == 40 for r in res)


@pytest.mark.xfail(strict=True, reason=(
    "BUG: Dunn's post-hoc computes n_well as len(self.df[grouping] == b) — the "
    "length of a boolean mask (== len(df)) — instead of the number of rows in "
    "group b, so n_well is inflated to count(a) + len(df)."))
def test_posthoc_dunn_n_well_counts_only_the_two_groups():
    from spacr.plot import spacrGraph

    g = spacrGraph(skewed_df(("a", "b", "c"), n=20), "grp", "v1",
                   representation="object")
    res = g.perform_posthoc_tests(is_normal=False,
                                  unique_groups=g.df["grp"].unique())

    assert all(r["n_well"] == 40 for r in res), [r["n_well"] for r in res]


def test_posthoc_returns_empty_without_all_to_all():
    from spacr.plot import spacrGraph

    g = spacrGraph(normal_df(("a", "b", "c"), n=20), "grp", "v1",
                   representation="object", all_to_all=False,
                   compare_group="a")
    assert g.perform_posthoc_tests(True, g.df["grp"].unique()) == []
    assert g.perform_posthoc_tests(False, g.df["grp"].unique()) == []


def test_posthoc_returns_empty_for_two_groups():
    from spacr.plot import spacrGraph

    g = spacrGraph(normal_df(("a", "b"), n=20), "grp", "v1",
                   representation="object")
    assert g.perform_posthoc_tests(True, g.df["grp"].unique()) == []


# ===========================================================================
# spacrGraph.create_plot — one test per graph type
# ===========================================================================

@pytest.mark.parametrize("graph_type", ["bar", "jitter", "box", "violin",
                                        "jitter_box", "jitter_bar"])
def test_create_plot_single_data_column_graph_types(graph_type):
    from spacr.plot import spacrGraph

    g = spacrGraph(normal_df(("a", "b", "c"), n=20), "grp", "v1",
                   graph_type=graph_type, representation="object")
    g.create_plot()

    fig = g.get_figure()
    assert fig is not None
    ax = fig.get_axes()[0]
    assert len(ax.collections) + len(ax.patches) + len(ax.lines) > 0
    # the violin renderer relabels the y-axis 'Value' after setting the column
    assert ax.get_ylabel() == ("Value" if graph_type == "violin" else "v1")
    assert ax.get_xlabel() == ""             # cleared for non-line graphs
    # _standerdize_figure_format enforces a square >=10 inch canvas
    assert tuple(fig.get_size_inches()) == (10.0, 10.0)
    # stats survived onto results_df: normality + omnibus + tukey
    res = g.get_results()
    assert set(res["Test Name"]) >= {"D'Agostino-Pearson test", "One-way ANOVA"}
    assert len(res) == 3 + 1 + 3


@pytest.mark.parametrize("graph_type", ["bar", "jitter", "box", "violin",
                                        "jitter_box", "jitter_bar"])
def test_create_plot_log_scales(graph_type):
    from spacr.plot import spacrGraph

    g = spacrGraph(normal_df(("a", "b", "c"), n=20), "grp", "v1",
                   graph_type=graph_type, representation="object",
                   log_y=True, log_x=True)
    g.create_plot()

    ax = g.get_figure().get_axes()[0]
    assert ax.get_yscale() == "log"
    assert ax.get_xscale() == "log"


def test_create_plot_reuses_supplied_axes():
    import matplotlib.pyplot as plt
    from spacr.plot import spacrGraph

    fig, ax = plt.subplots()
    g = spacrGraph(normal_df(("a", "b"), n=20), "grp", "v1",
                   representation="object")
    g.create_plot(ax=ax)

    assert g.fig is fig
    assert len(ax.patches) > 0


def test_create_plot_rejects_unknown_graph_type():
    from spacr.plot import spacrGraph

    g = spacrGraph(normal_df(("a", "b"), n=20), "grp", "v1",
                   graph_type="pie", representation="object")
    with pytest.raises(ValueError, match="Unknown graph type: pie"):
        g.create_plot()


def test_create_plot_y_lim_single_element_sets_only_bottom():
    from spacr.plot import spacrGraph

    g = spacrGraph(normal_df(("a", "b", "c"), n=20), "grp", "v1",
                   representation="object", y_lim=[2.0])
    g.create_plot()
    bottom, top = g.get_figure().get_axes()[0].get_ylim()

    assert bottom == 2.0
    assert top > 2.0


def test_create_plot_y_lim_two_elements():
    from spacr.plot import spacrGraph

    g = spacrGraph(normal_df(("a", "b", "c"), n=20), "grp", "v1",
                   representation="object", y_lim=[1.0, 50.0])
    g.create_plot()
    assert g.get_figure().get_axes()[0].get_ylim() == (1.0, 50.0)


def test_create_plot_multi_data_column_builds_symbol_table():
    from spacr.plot import spacrGraph

    g = spacrGraph(multi_col_df(("a", "b", "c"), n=12), "grp", ["v1", "v2"],
                   graph_type="bar", representation="object")
    g.create_plot()

    assert g.hue == "Data Column"
    assert g.jitter_bar_dodge is True
    assert "Combined Group" in g.df_melted.columns
    assert list(g.summary_df["Combined Group"]) == [
        "a - v1", "a - v2", "b - v1", "b - v2", "c - v1", "c - v2"]

    fig = g.get_figure()
    ax = fig.get_axes()[0]
    # x ticks are removed and replaced by the +/- symbol table
    assert list(ax.get_xticks()) == []
    texts = [t.get_text() for t in ax.texts]
    assert texts[:3] == ["grp", "v1", "v2"]      # row labels
    assert "+" in texts and "-" in texts          # symbol rows
    # an extra axes was added below the plot to host the legend table
    assert len(fig.axes) == 2


def test_create_plot_multi_data_column_jitter_bar_places_symbols_per_position():
    from spacr.plot import spacrGraph

    g = spacrGraph(multi_col_df(("a", "b", "c"), n=12), "grp", ["v1", "v2"],
                   graph_type="jitter_bar", representation="object")
    g.create_plot()

    ax = g.get_figure().get_axes()[0]
    texts = [t.get_text() for t in ax.texts]
    # 3 row labels, then one 3-cell column per jitter x-position
    assert texts[:3] == ["grp", "v1", "v2"]
    assert (len(texts) - 3) % 3 == 0
    assert len(texts) > 3
    assert texts.count("+") == texts.count("-")


@pytest.mark.xfail(strict=True, reason=(
    "BUG: multi-data-column plots draw x='Combined Group' but pass "
    "order=self.order (the raw group names), so seaborn matches no category "
    "and renders an empty plot (bar) or crashes (box/jitter_box)."))
def test_create_plot_multi_data_column_draws_one_bar_per_combined_group():
    import matplotlib.pyplot as plt
    from spacr.plot import spacrGraph

    g = spacrGraph(multi_col_df(("a", "b", "c"), n=12), "grp", ["v1", "v2"],
                   graph_type="bar", representation="object")
    g.create_plot()

    ax = g.get_figure().get_axes()[0]
    bars = [p for p in ax.patches if isinstance(p, plt.Rectangle)]
    assert len(bars) == 6, f"expected 3 groups x 2 columns, got {len(bars)}"


@pytest.mark.xfail(strict=True, reason=(
    "BUG: multi-data-column box plot passes order=self.order while x is the "
    "'Combined Group' column, so seaborn draws no boxes and dies with "
    "UnboundLocalError: local variable 'boxprops' referenced before assignment.")
)
def test_create_plot_multi_data_column_box():
    from spacr.plot import spacrGraph

    g = spacrGraph(multi_col_df(("a", "b", "c"), n=12), "grp", ["v1", "v2"],
                   graph_type="box", representation="object")
    g.create_plot()
    assert g.get_figure() is not None


@pytest.mark.xfail(strict=True, reason=(
    "BUG: same 'Combined Group' vs order mismatch kills the multi-column "
    "jitter_box plot with UnboundLocalError on seaborn's boxprops."))
def test_create_plot_multi_data_column_jitter_box():
    from spacr.plot import spacrGraph

    g = spacrGraph(multi_col_df(("a", "b", "c"), n=12), "grp", ["v1", "v2"],
                   graph_type="jitter_box", representation="object")
    g.create_plot()
    assert g.get_figure() is not None


def test_create_plot_multi_data_column_violin_and_jitter():
    from spacr.plot import spacrGraph

    for graph_type in ("violin", "jitter"):
        g = spacrGraph(multi_col_df(("a", "b"), n=12), "grp", ["v1", "v2"],
                       graph_type=graph_type, representation="object")
        g.create_plot()
        ax = g.get_figure().get_axes()[0]
        assert ax.get_ylabel() == "Value"
        assert [t.get_text() for t in ax.texts][:3] == ["grp", "v1", "v2"]


# ===========================================================================
# spacrGraph line graphs
# ===========================================================================

def test_create_plot_line_graph():
    from spacr.plot import spacrGraph

    g = spacrGraph(line_df(), "grp", ["epoch", "acc"], graph_type="line",
                   representation="object")
    g.create_plot()

    fig = g.get_figure()
    ax = fig.get_axes()[0]
    assert ax.get_xlabel() == "epoch"
    assert ax.get_ylabel() == "acc"
    assert len(ax.lines) >= 3           # one line per plate group
    assert g.fig_height == 10 and g.fig_width == 10
    assert len(g.summary_df) == 24


def test_create_plot_line_graph_log_transforms_columns():
    from spacr.plot import spacrGraph

    df = line_df(groups=("p1", "p2"), n=8)
    g = spacrGraph(df.copy(), "grp", ["epoch", "acc"], graph_type="line",
                   representation="object", log_y=True, log_x=True)
    g.create_plot()

    # the log transform is applied in place to self.df
    assert np.allclose(np.sort(g.df["epoch"].unique()),
                       np.log10(np.arange(1, 9, dtype=float)))
    assert np.allclose(g.df["acc"].to_numpy(),
                       np.log10(df["acc"].to_numpy()))


def test_create_line_graph_raises_for_missing_column():
    import matplotlib.pyplot as plt
    from spacr.plot import spacrGraph

    g = spacrGraph(line_df(), "grp", ["epoch", "acc"], graph_type="line",
                   representation="object")
    g.df = g.df.drop(columns=["acc"])
    _fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="Column 'acc' not found"):
        g._create_line_graph(ax)


def test_create_plot_line_std_area():
    from spacr.plot import spacrGraph

    g = spacrGraph(line_df(), "grp", ["epoch", "acc"], graph_type="line_std",
                   representation="object")
    g.create_plot()

    ax = g.get_figure().get_axes()[0]
    assert list(g.summary_df.columns) == ["epoch", "mean_acc", "std_mean_acc"]
    assert len(g.summary_df) == 8            # one row per epoch
    assert ax.get_xlabel() == "epoch"
    assert ax.get_ylabel() == "acc"
    # fill_between adds a band spanning mean +/- std of the accuracies
    band = ax.collections[-1]
    verts = band.get_paths()[0].vertices
    lo = (g.summary_df["mean_acc"] - g.summary_df["std_mean_acc"]).min()
    hi = (g.summary_df["mean_acc"] + g.summary_df["std_mean_acc"]).max()
    assert verts[:, 1].min() == pytest.approx(lo)
    assert verts[:, 1].max() == pytest.approx(hi)


def test_create_plot_line_std_log_transforms():
    from spacr.plot import spacrGraph

    df = line_df(groups=("p1", "p2"), n=8)
    g = spacrGraph(df.copy(), "grp", ["epoch", "acc"], graph_type="line_std",
                   representation="object", log_y=True, log_x=True)
    g.create_plot()

    assert np.allclose(np.sort(g.summary_df["epoch"].to_numpy()),
                       np.log10(np.arange(1, 9, dtype=float)))
    assert (g.summary_df["mean_acc"] < 0).all()   # log10 of accuracies < 1


# ===========================================================================
# spacrGraph._standerdize_figure_format
# ===========================================================================

def test_standerdize_figure_format_skips_line_graphs(capsys):
    import matplotlib.pyplot as plt
    from spacr.plot import spacrGraph

    g = spacrGraph(normal_df(("a", "b"), n=20), "grp", "v1",
                   representation="object")
    fig, ax = plt.subplots(figsize=(3, 3))
    assert g._standerdize_figure_format(ax=ax, num_groups=2,
                                        graph_type="line") is None

    assert "Skipping layout adjustment for line graphs." in capsys.readouterr().out
    assert tuple(fig.get_size_inches()) == (3.0, 3.0)   # untouched


def test_standerdize_figure_format_resizes_box_artists():
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

    assert artist.width == pytest.approx(min(0.8, 1.5 / 3) / 4)
    assert ax.get_xlim() == (-0.5, 2.5)


def test_standerdize_figure_format_shrinks_bars_and_moves_legend():
    import matplotlib.pyplot as plt
    from spacr.plot import spacrGraph

    g = spacrGraph(normal_df(("a", "b", "c", "d"), n=20), "grp", "v1",
                   representation="object")
    _fig, ax = plt.subplots()
    bar = ax.bar([0, 1, 2, 3], [1, 2, 3, 4], width=0.8)[0]
    ax.plot([0, 1], [0, 1], label="dummy")
    ax.legend()

    g._standerdize_figure_format(ax=ax, num_groups=4, graph_type="bar")

    expected = min(0.8, 1.5 / 4) / 4
    assert bar.get_width() == pytest.approx(expected)
    assert ax.get_legend() is not None


def test_standerdize_figure_format_shifts_jitter_collections():
    import matplotlib.pyplot as plt
    from spacr.plot import spacrGraph

    g = spacrGraph(normal_df(("a", "b", "c"), n=20), "grp", "v1",
                   representation="object")
    _fig, ax = plt.subplots()
    coll = ax.scatter([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])
    before = coll.get_offsets().copy()

    g._standerdize_figure_format(ax=ax, num_groups=3, graph_type="jitter")

    after = coll.get_offsets()
    shift = min(0.1, 0.2 / 3) / 4
    assert np.allclose(after[:, 0], np.asarray(before)[:, 0] + shift)
    assert np.allclose(coll.get_sizes(), max(50 / 3, 200))


# ===========================================================================
# spacrGraph._save_results
# ===========================================================================

def test_save_results_writes_figure_stats_and_data(tmp_path, capsys):
    from spacr.plot import spacrGraph

    out = tmp_path / "results"
    g = spacrGraph(normal_df(("a", "b", "c"), n=20), "grp", "v1",
                   graph_type="bar", representation="object",
                   save=True, output_dir=str(out), graph_name="mygraph")
    g.create_plot()

    stem = "mygraph_v1_grp_bar"
    assert g.results_name == stem
    assert (out / f"{stem}.pdf").is_file()
    assert (out / f"{stem}.pdf").stat().st_size > 0

    stats = pd.read_csv(out / f"{stem}_stats.csv")
    assert len(stats) == len(g.get_results())
    data = pd.read_csv(out / f"{stem}_data.csv")
    assert len(data) == 60
    summary = pd.read_csv(out / f"{stem}_summary.csv")
    assert list(summary["grp"]) == ["a", "b", "c"]

    printed = capsys.readouterr().out
    assert "Plot  ->" in printed and "Stats ->" in printed and "Data ->" in printed


def test_save_results_without_summary_df(tmp_path):
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

    stem = "bare_v1_grp_bar"
    assert (out / f"{stem}.pdf").is_file()
    assert (out / f"{stem}_stats.csv").is_file()
    assert not (out / f"{stem}_summary.csv").exists()
