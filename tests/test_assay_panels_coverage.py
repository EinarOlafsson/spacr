"""Edge paths of spacr.assay_panels: unparseable p values, untestable tables,
missing columns, caller palettes and colours, and the drawing paths without
statistics."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from spacr import assay_panels as ap  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _groups(values_by_group):
    rows = [(g, v) for g, vals in values_by_group.items() for v in vals]
    return pd.DataFrame(rows, columns=["group", "value"])


# --------------------------------------------------------------------------- formatting
def test_stars_and_fmt_p_on_values_that_are_not_numbers():
    assert ap.stars("abc") == "n/a" and ap.stars(None) == "n/a"
    assert ap.fmt_p("abc") == "p = n/a" and ap.fmt_p(object()) == "p = n/a"
    assert ap.stars(float("nan")) == "n/a" and ap.fmt_p(float("inf")) == "p = n/a"
    assert ap.stars(0.03) == "*" and ap.fmt_p(0.03) == "p = 0.03"


def test_condition_order_keeps_one_copy_of_repeated_values():
    assert ap.condition_order(["b", "a", "b", None, "a", np.nan]) == ["a", "b"]


def test_display_label_matches_a_text_key_for_a_numeric_value():
    assert ap.display_label(3, {"3": "three"}) == "three"
    assert ap.display_label(4, {"3": "three"}) == "4"


# --------------------------------------------------------------------------- bar tests
def test_a_failing_post_hoc_test_reports_not_tested(monkeypatch):
    import scikit_posthocs

    def _boom(*args, **kwargs):
        raise ValueError("no Dunn today")

    monkeypatch.setattr(scikit_posthocs, "posthoc_dunn", _boom)
    # a strongly skewed group makes Shapiro-Wilk fail -> Kruskal-Wallis + Dunn
    skewed = [1.0] * 9 + [50.0]
    df = _groups({"a": skewed, "b": [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]})
    out = ap.select_and_run_test(df, "value", "group", ["a", "b"])
    test = out["test"].iloc[0]
    assert test["test"] == "not tested"
    assert test["note"] == "test failed: no Dunn today"
    assert not test["all_groups_normal"]
    assert out["posthoc"].empty
    assert out["descriptive"]["n"].tolist() == [10, 10]


# --------------------------------------------------------------------------- proportion tests
def test_cramers_v_and_small_tables_give_nan_results():
    assert np.isnan(ap._cramers_v(np.array([[0, 0], [0, 0]])))
    assert np.isnan(ap._cramers_v(np.array([[3, 4]])))
    chi = ap._chi2_test(np.array([[5, 0], [7, 0]]))
    assert chi["test"] == "chi-square test of independence"
    assert np.isnan(chi["p_value"]) and chi["n"] == 12 and chi["n_resamples"] == 0
    fisher = ap._fisher_test(np.array([[0, 0], [3, 4]]))
    assert fisher["test"] == "Fisher-Freeman-Halton exact test (Monte Carlo)"
    assert np.isnan(fisher["statistic"]) and fisher["n"] == 7


def test_proportion_tests_reject_an_unknown_method():
    counts = pd.DataFrame([[1, 2]], index=["a"], columns=["x", "y"])
    with pytest.raises(ValueError, match="method must be 'chi2' or 'fisher', got 'g'"):
        ap.proportion_tests(counts, ["a"], method="g")


def test_proportion_tests_with_only_the_reference_has_no_pairwise_rows():
    counts = pd.DataFrame([[1, 2]], index=["a"], columns=["x", "y"])
    omni, pair = ap.proportion_tests(counts, ["a", "absent"])
    assert omni["comparison"] == "a" and np.isnan(omni["p_value"])
    assert omni["significance"] == "n/a"
    assert pair.empty and "p_holm" in pair.columns
    omni, pair = ap.proportion_tests(counts, ["absent"])
    assert omni["test"] == "chi2" and omni["n"] == 0 and pair.empty


def test_pairwise_rows_without_a_finite_p_keep_holm_missing():
    # the reference row is empty, so every pairwise table has one row left
    counts = pd.DataFrame([[0, 0], [10, 5], [5, 10]], index=["ref", "b", "c"],
                          columns=["x", "y"])
    omni, pair = ap.proportion_tests(counts, ["ref", "b", "c"])
    assert np.isfinite(omni["p_value"])
    assert pair["group_2"].tolist() == ["b", "c"]
    assert pair["p_holm"].isna().all()
    assert pair["significance"].tolist() == ["n/a", "n/a"]


# --------------------------------------------------------------------------- aggregation
def test_aggregate_to_unit_names_the_missing_column():
    df = pd.DataFrame({"plateID": ["p"], "rowID": ["r1"], "columnID": ["c1"], "v": [1.0]})
    with pytest.raises(KeyError, match="column 'g' is not in the table"):
        ap.aggregate_to_unit(df, "v", ["g"])
    with pytest.raises(KeyError, match="column 'rep' is not in the table"):
        ap.aggregate_to_unit(df, ["v"], [], replicate_column="rep")


# --------------------------------------------------------------------------- drawing
def test_set_xticks_uses_a_rotation_given_by_the_caller():
    fig, ax = plt.subplots()
    ap._set_xticks(ax, ["a", "b"], None, None, rotation=90)
    labels = ax.get_xticklabels()
    assert [t.get_text() for t in labels] == ["a", "b"]
    assert labels[0].get_rotation() == 90 and labels[0].get_ha() == "right"
    assert ax.get_xlim() == pytest.approx((-0.6, 1.6))


def test_bar_panel_without_stats_with_a_palette_list_and_an_empty_group():
    df = _groups({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0]})
    fig, ax = plt.subplots()
    ap.draw_bar_panel(ax, df, "value", "group", ["a", "b", "c"], title="plain",
                      palette=["#111111", "#222222"])
    assert ax.get_title() == "plain"
    colors = [matplotlib.colors.to_hex(p.get_facecolor()) for p in ax.patches]
    # the list palette cycles over the order: c reuses the first colour
    assert colors == ["#111111", "#222222", "#111111"]
    # one scatter collection per group with data; none for c
    assert len(ax.collections) == 1 + 2  # error bar + two groups of dots
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b", "c"]


def test_stacked_panel_with_given_colours_and_no_statistics():
    frac = pd.DataFrame({"g": ["a", "a", "b"], "x": [0.5, 0.7, 0.2], "y": [0.5, 0.3, 0.8]})
    fig, ax = plt.subplots()
    ap.draw_stacked_panel(ax, frac, "g", ["a", "b"], ["x", "y"], title="T",
                          colors=["#ff0000", "#00ff00"], legend=False)
    assert ax.get_title() == "T"
    assert ax.get_legend() is None
    faces = {matplotlib.colors.to_hex(p.get_facecolor()) for p in ax.patches}
    assert faces == {"#ff0000", "#00ff00"}
    assert not [t for t in ax.texts if "*" in t.get_text() or t.get_text() == "ns"]


def test_stacked_panel_skips_pairwise_rows_for_groups_not_drawn():
    frac = pd.DataFrame({"g": ["a", "b"], "x": [0.5, 0.2], "y": [0.5, 0.8]})
    pairwise = pd.DataFrame({"group_2": ["b", "gone"], "significance": ["**", "ns"]})
    fig, ax = plt.subplots()
    ap.draw_stacked_panel(ax, frac, "g", ["a", "b"], ["x", "y"], pairwise=pairwise,
                          omnibus={"p_value": 0.5, "cramers_v": float("nan")},
                          method_label="chi-square")
    texts = [t.get_text() for t in ax.texts]
    assert "**" in texts and "ns" not in texts
    # a missing Cramer V is left out of the title
    assert ax.get_title() == "chi-square omnibus: p = 0.5"


def test_bin_colors_return_the_callers_colours_as_a_list():
    assert ap._bin_colors(3, ("r", "g")) == ["r", "g"]
    assert len(ap._bin_colors(0)) == 1


# --------------------------------------------------------------------------- saving
def test_save_panel_writes_extra_tables_and_figure_carries_a_title(tmp_path):
    df = _groups({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    extra = pd.DataFrame({"note": ["kept"]})
    spec = ap.bar_spec(df, "value", "group", ["a", "b"], extra_tables={"filter_log": extra})
    folder = ap.save_panel(str(tmp_path), "P03", "extra", spec)
    assert spec["folder"] == folder
    written = pd.read_csv(os.path.join(folder, "filter_log.csv"))
    assert written["note"].tolist() == ["kept"]
    paths = ap.assemble_figure({"P03": spec}, ["P03"], 3, str(tmp_path / "F2.pdf"),
                               title="Figure 2")
    assert paths == [str(tmp_path / "F2.pdf"), str(tmp_path / "F2.png")]
    assert all(os.path.getsize(p) > 0 for p in paths)


def test_bar_panel_takes_a_palette_mapping_and_greys_unlisted_groups():
    df = _groups({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    fig, ax = plt.subplots()
    ap.draw_bar_panel(ax, df, "value", "group", ["a", "b"], palette={"a": "#123456"})
    colors = [matplotlib.colors.to_hex(p.get_facecolor()) for p in ax.patches]
    assert colors == ["#123456", "#888888"]
