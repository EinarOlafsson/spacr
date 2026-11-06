"""Tests for spacr.assay_panels and the panel / filter options of
analyze_recruitment and analyze_replication."""
import os
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from spacr import assay_panels as ap  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


def _normal(mean, sd, n):
    """Exactly normal-shaped sample (normal quantiles), so Shapiro-Wilk passes."""
    from scipy import stats
    return mean + sd * stats.norm.ppf((np.arange(n) + 0.5) / n)


def _groups(values_by_group):
    rows = [(g, v) for g, vals in values_by_group.items() for v in vals]
    return pd.DataFrame(rows, columns=["group", "value"])


# --------------------------------------------------------------------------- test selection
def test_normal_equal_variance_selects_anova_and_tukey():
    df = _groups({"a": _normal(10, 1, 12), "b": _normal(13, 1, 12),
                  "c": _normal(10.5, 1.1, 12)})
    out = ap.select_and_run_test(df, "value", "group", ["a", "b", "c"])
    test = out["test"].iloc[0]
    assert test["test"] == "one-way ANOVA"
    assert test["all_groups_normal"] and test["equal_variance"]
    assert test["p_value"] < 0.001
    assert set(out["posthoc"]["posthoc"]) == {"Tukey HSD"}
    assert len(out["posthoc"]) == 3
    ab = out["posthoc"].query("group_1 == 'a' and group_2 == 'b'").iloc[0]
    assert ab["p_adj"] < 0.001 and ab["significance"] != "ns"
    assert list(out["descriptive"]["group"]) == ["a", "b", "c"]
    assert (out["assumptions"]["check"] == "Shapiro-Wilk normality").sum() == 3


def test_normal_unequal_variance_selects_welch_and_games_howell():
    df = _groups({"a": _normal(10, 0.1, 15), "b": _normal(11, 5, 15),
                  "c": _normal(10, 0.1, 15)})
    out = ap.select_and_run_test(df, "value", "group", ["a", "b", "c"])
    test = out["test"].iloc[0]
    assert test["test"] == "Welch ANOVA"
    assert test["all_groups_normal"] and not test["equal_variance"]
    assert set(out["posthoc"]["posthoc"]) == {"Games-Howell"}


@pytest.mark.parametrize("sds, expected", [
    ((1.0, 1.0, 1.1), ("one-way ANOVA", "p-tukey")),
    ((0.1, 5.0, 0.1), ("Welch ANOVA", "pval")),
])
def test_the_parametric_tests_match_pingouin(sds, expected):
    """spaCR's own ANOVA, Welch ANOVA, Tukey HSD and Games-Howell agree with
    pingouin's, which is only a development dependency, to rounding error."""
    pg = pytest.importorskip("pingouin")
    df = _groups({"b10": _normal(10, sds[0], 9), "a2": _normal(12, sds[1], 7),
                  "c": _normal(10.5, sds[2], 11)})
    out = ap.select_and_run_test(df, "value", "group", ["c", "b10", "a2"])
    test, post = out["test"].iloc[0], out["posthoc"]
    name, p_column = expected
    assert test["test"] == name
    if name == "one-way ANOVA":
        ref = pg.anova(data=df, dv="value", between="group")
        ref_post = pg.pairwise_tukey(data=df, dv="value", between="group")
    else:
        ref = pg.welch_anova(data=df, dv="value", between="group")
        ref_post = pg.pairwise_gameshowell(data=df, dv="value", between="group")
    for ours, theirs in (("statistic", "F"), ("df2", "ddof2"),
                         ("effect_size", "np2")):
        assert test[ours] == pytest.approx(float(ref[theirs].iloc[0]), rel=1e-9)
    p_ref = ref["p-unc"] if "p-unc" in ref else ref["p_unc"]
    assert test["p_value"] == pytest.approx(float(p_ref.iloc[0]), rel=1e-9)
    assert list(zip(post["group_1"], post["group_2"])) == list(
        zip(ref_post["A"], ref_post["B"]))
    np.testing.assert_allclose(post["statistic"].to_numpy(float),
                               ref_post["T"].to_numpy(float), rtol=1e-9)
    ref_p = ref_post[p_column] if p_column in ref_post else ref_post[
        p_column.replace("-", "_")]
    np.testing.assert_allclose(post["p_adj"].to_numpy(float),
                               ref_p.to_numpy(float), rtol=1e-7, atol=1e-12)


def test_a_constant_group_falls_back_from_welch_to_kruskal_wallis():
    df = _groups({"a": _normal(10, 3, 12), "b": [5.0] * 12})
    out = ap.select_and_run_test(df, "value", "group", ["a", "b"])
    test = out["test"].iloc[0]
    assert test["test"] == "Kruskal-Wallis"
    assert "Welch ANOVA undefined" in test["note"]


def test_non_normal_selects_kruskal_wallis_and_dunn():
    rng = np.random.default_rng(3)
    df = _groups({"a": rng.exponential(1, 30) ** 3, "b": rng.exponential(3, 30) ** 3})
    out = ap.select_and_run_test(df, "value", "group", ["a", "b"])
    test = out["test"].iloc[0]
    assert test["test"] == "Kruskal-Wallis"
    assert not test["all_groups_normal"]
    assert out["posthoc"].iloc[0]["posthoc"] == "Dunn test (Holm)"
    assert len(out["posthoc"]) == 1


def test_a_group_without_data_is_reported_and_left_out():
    df = _groups({"a": [1.0, 1.2, 0.9], "b": [2.0, 2.1, 1.9]})
    out = ap.select_and_run_test(df, "value", "group", ["a", "missing", "b"])
    desc = out["descriptive"].set_index("group")
    assert desc.loc["missing", "n"] == 0
    assert out["test"].iloc[0]["n_groups"] == 2
    one = ap.select_and_run_test(df, "value", "group", ["a"])
    assert one["test"].iloc[0]["test"] == "not tested"


# --------------------------------------------------------------------------- proportions
def test_chi_square_and_fisher_ffh_on_a_shifted_distribution():
    counts = pd.DataFrame([[60, 30, 10], [20, 40, 40], [58, 32, 10]],
                          index=["WT", "mut", "comp"], columns=["1", "2", "4"])
    omni, pair = ap.proportion_tests(counts, ["WT", "mut", "comp"], method="chi2")
    assert omni["test"] == "chi-square test of independence" and omni["p_value"] < 1e-6
    assert omni["dof"] == 4 and 0 < omni["cramers_v"] < 1
    assert list(pair["group_2"]) == ["mut", "comp"]
    assert pair.set_index("group_2").loc["comp", "significance"] == "ns"
    omni_f, pair_f = ap.proportion_tests(counts, ["WT", "mut", "comp"], method="fisher",
                                         n_resamples=2000, seed=0)
    assert omni_f["test"].startswith("Fisher-Freeman-Halton")
    assert omni_f["p_value"] == pytest.approx(1 / 2001)
    assert pair_f.set_index("group_2").loc["mut", "p_holm"] < 0.01
    # seeded Monte Carlo is reproducible
    again, _ = ap.proportion_tests(counts, ["WT", "mut", "comp"], method="fisher",
                                   n_resamples=2000, seed=0)
    assert again["p_value"] == omni_f["p_value"]


def test_two_by_two_uses_the_fisher_exact_test():
    from scipy import stats
    counts = pd.DataFrame([[12, 3], [4, 10]], index=["a", "b"], columns=["x", "y"])
    omni, pair = ap.proportion_tests(counts, ["a", "b"], reference="a", method="fisher")
    assert omni["test"] == "Fisher exact test"
    assert omni["p_value"] == pytest.approx(stats.fisher_exact(counts.to_numpy()).pvalue)
    assert pair.iloc[0]["test"] == "Fisher exact test"


def test_empty_bins_are_dropped_before_testing():
    counts = pd.DataFrame([[10, 0, 5], [4, 0, 9]], index=["a", "b"], columns=["1", "2", "4"])
    omni, _ = ap.proportion_tests(counts, ["a", "b"], method="chi2")
    assert omni["dof"] == 1


# --------------------------------------------------------------------------- order / labels
def test_condition_order_puts_preferred_first_then_natural_sort():
    values = ["r10", "r2", "WT", "r1", None, "mut"]
    assert ap.condition_order(values) == ["mut", "r1", "r2", "r10", "WT"]
    assert ap.condition_order(values, preferred=["WT", "absent", "r10"]) == [
        "WT", "r10", "mut", "r1", "r2"]


def test_display_label_uses_labels_then_spaces():
    assert ap.display_label("HeLa_WT") == "HeLa WT"
    assert ap.display_label("HeLa_WT", {"HeLa_WT": "WT host"}) == "WT host"


# --------------------------------------------------------------------------- aggregation
def _objects():
    rows = []
    for rep in ("rep1", "rep2"):
        for cond, base in (("ctrl", 1.0), ("mut", 2.0)):
            for w, col in enumerate(("c1", "c2")):
                for i in range(3):
                    rows.append(dict(plateID="p1", rowID=f"r{1 if rep == 'rep1' else 2}",
                                     columnID=f"{col}_{cond}", rep=rep, condition=cond,
                                     value=base + w + i * 0.1))
    return pd.DataFrame(rows)


def test_aggregate_to_unit_well_then_replicate():
    df = _objects()
    wells = ap.aggregate_to_unit(df, "value", ["condition"])
    assert len(wells) == 8 and set(wells["n_objects"]) == {3}
    reps = ap.aggregate_to_unit(df, "value", ["condition"], replicate_column="rep")
    assert len(reps) == 4 and set(reps["n_wells"]) == {2}
    ctrl = reps[reps["condition"] == "ctrl"]["value"]
    assert np.allclose(ctrl, 1.0 + 0.5 + 0.1)


def test_fractions_per_unit_sum_to_one():
    df = pd.DataFrame(dict(plateID="p", rowID="r1", columnID=["c1"] * 4 + ["c2"] * 4,
                           g=["a"] * 4 + ["b"] * 4, bin=list("xxyz") + list("yyyz")))
    frac = ap.fractions_per_unit(df, "bin", ["x", "y", "z"], ["g"])
    assert np.allclose(frac[["x", "y", "z"]].sum(axis=1), 1)
    assert frac.set_index("g").loc["a", "x"] == 0.5


# --------------------------------------------------------------------------- drawing / saving
def test_save_panel_writes_the_bar_folder_layout(tmp_path):
    rng = np.random.default_rng(0)
    df = _groups({"WT": rng.normal(2, .2, 4), "dGRA8": rng.normal(1.4, .2, 4)})
    spec = ap.bar_spec(df, "value", "group", ["WT", "dGRA8"], title="t",
                       annotations={"GRA8": {"WT": "+", "dGRA8": "-"},
                                    "ALG2": {"WT": "+", "dGRA8": "+"}},
                       labels={"dGRA8": "ΔGRA8"}, ymin=1)
    folder = ap.save_panel(str(tmp_path), "P01", "recruitment WT/dGRA8", spec)
    assert os.path.basename(folder) == "P01_recruitment_WT_dGRA8"
    for name in ("P01_recruitment_WT_dGRA8.pdf", "P01_recruitment_WT_dGRA8.png",
                 "data_replicates.csv", "stats_test.csv", "stats_assumptions.csv",
                 "stats_posthoc.csv", "stats_descriptive.csv"):
        assert os.path.exists(os.path.join(folder, name)), name


def test_annotations_render_aligned_under_the_axis():
    df = _groups({"a": [1.0, 1.1, 1.2], "b": [2.0, 2.2, 2.1], "c": [1.5, 1.6, 1.4]})
    stats = ap.select_and_run_test(df, "value", "group", ["a", "b", "c"])
    fig, ax = plt.subplots()
    ap.draw_bar_panel(ax, df, "value", "group", ["a", "b", "c"], stats=stats,
                      annotations={"drug": {"a": "-", "b": "+", "c": "+"}})
    fig.canvas.draw()
    texts = {t.get_text(): t for t in ax.texts}
    assert "drug" in texts and "+" in texts
    ticks = [t.get_text() for t in ax.get_xticklabels()]
    assert ticks == ["a", "b", "c"]


def test_save_stacked_panel_and_assemble_figure(tmp_path):
    counts = pd.DataFrame([[30, 20, 10], [10, 20, 30]], index=["WT", "mut"],
                          columns=["1", "2", "4"])
    frac = pd.DataFrame(dict(g=["WT", "WT", "mut", "mut"], **{
        "1": [.5, .5, .2, .1], "2": [.3, .35, .3, .35], "4": [.2, .15, .5, .55]}))
    stacked = ap.stacked_spec(frac, counts, "g", ["WT", "mut"], ["1", "2", "4"],
                              method="chi2", legend_title="parasites per vacuole")
    folder = ap.save_panel(str(tmp_path), "P02", "bins", stacked)
    for name in ("P02_bins.pdf", "contingency_table_counts.csv", "stats_test.csv",
                 "stats_descriptive.csv", "data_replicates.csv"):
        assert os.path.exists(os.path.join(folder, name)), name
    test = pd.read_csv(os.path.join(folder, "stats_test.csv"))
    assert list(test["scope"]) == ["omnibus", "pairwise vs WT"]
    bar = ap.bar_spec(_groups({"WT": [1, 2, 3], "mut": [4, 5, 6]}), "value", "group",
                      ["WT", "mut"])
    written = ap.assemble_figure({"P01": bar, "P02": stacked}, ["P01", "P02"], 2,
                                 str(tmp_path / "figures" / "F1"))
    assert all(os.path.exists(p) for p in written) and len(written) == 2
    with pytest.raises(KeyError):
        ap.assemble_figure({"P01": bar}, ["P09"], 1, str(tmp_path / "x"))


# --------------------------------------------------------------------------- well density
def test_well_density_sums_fields(tmp_path):
    from spacr.recruitment_confounders import well_density, apply_well_density
    db = tmp_path / "m.db"
    with sqlite3.connect(db) as con:
        con.execute("create table cell (plateID, rowID, columnID, fieldID, cell_id)")
        con.execute("create table pathogen (plateID, rowID, columnID, fieldID, cell_id)")
        for fld, n in (("1", 3), ("2", 5)):
            for i in range(n):
                con.execute("insert into cell values (?,?,?,?,?)", ("P%5F1", "r1", "c1", fld, i + 1))
            con.execute("insert into pathogen values (?,?,?,?,?)", ("P%5F1", "r1", "c1", fld, 1))
    w = well_density(str(db))
    assert w.loc[0, "n_cells_well"] == 8 and w.loc[0, "n_pathogens_well"] == 2
    assert w.loc[0, "pathogens_per_cell_well"] == pytest.approx(0.25)
    objs = pd.DataFrame(dict(plateID=["P_1", "P_1"], rowID="r1", columnID=["c1", "c9"]))
    out = apply_well_density(objs, w)
    assert out.loc[0, "n_cells_well"] == 8 and np.isnan(out.loc[1, "n_cells_well"])


# --------------------------------------------------------------------------- analyze_recruitment
def _recruitment_frame():
    from tests.test_cov_submodules_recruitment_plaques import _merged_recruitment_df
    wells = [(r, c) for r in ("r1", "r2", "r4", "r5") for c in ("c1", "c2", "c4", "c5")]
    df = _merged_recruitment_df(wells=wells, n_cells=6)
    rng = np.random.default_rng(0)
    df["pathogen_channel_2_mean_intensity"] = rng.normal(400, 40, len(df)) + \
        np.where(df["rowID"].isin(["r4", "r5"]), 120, 0)
    df["pathogen_prcfo_count"] = np.tile([1, 1, 2, 1, 3, 1], len(df) // 6)
    df["pathogen_equivalent_diameter_area"] = 12.0
    df["pathogen_distance_to_field_edge"] = np.tile([100, 100, 100, 3, 100, 100], len(df) // 6)
    return df


def _recruitment_db(src, df):
    with sqlite3.connect(src / "measurements" / "measurements.db") as con:
        con.execute("create table cell (plateID, rowID, columnID, fieldID, cell_id)")
        con.execute("create table pathogen (plateID, rowID, columnID, fieldID, cell_id)")
        for k, ((rid, cid), _) in enumerate(df.groupby(["rowID", "columnID"])):
            for i in range(20 + k):
                con.execute("insert into cell values (?,?,?,?,?)", ("plate1", rid, cid, "1", i + 1))
                con.execute("insert into pathogen values (?,?,?,?,?)", ("plate1", rid, cid, "1", i + 1))


def test_analyze_recruitment_panels_controls_and_filters(tmp_path):
    from tests.test_cov_submodules_recruitment_plaques import _base_settings, _Recorder
    import spacr.io as SIO
    import spacr.plot as SPLT
    from spacr.submodules import analyze_recruitment

    df = _recruitment_frame()
    src = tmp_path / "plate"
    (src / "measurements").mkdir(parents=True)
    _recruitment_db(src, df)
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(SIO, "_read_and_merge_data", lambda locs, tables, **k: (df.copy(), []))
        for name in ("plot_image_mask_overlay", "_plot_recruitment"):
            mp.setattr(SPLT, name, _Recorder())
        settings = _base_settings(
            src, plot_control=True, channel_dims=[0, 3], channel_of_interest=2,
            pathogen_plate_metadata=[["c1", "c2"], ["c4", "c5"]],
            treatment_plate_metadata=[["r1", "r2"], ["r4", "r5"]],
            replicate_column="pathogen",
            condition_order=["lovastatin"], condition_labels={"cm": "CM"},
            panels=[{"name": "treatment_effect", "group": "treatment",
                     "annotations": {"statin": {"cm": "-", "lovastatin": "+"}}},
                    {"name": "cell_area", "group": "treatment", "value": "cell_area",
                     "where": {"pathogen": "pathogen_1"}}],
            figures={"F1": {"panels": ["P01", "P02", "C01"], "ncol": 3}},
            max_vacuoles_per_cell=2, exclude_edge_objects=True,
            confounder_filter={"cell_area": [0, 1004.5],
                               "n_cells_well": {"quantiles": [0.1, 1.0], "level": "well"}})
        cells, wells = analyze_recruitment(settings)
    finally:
        mp.undo()

    res = src / "results"
    summary = pd.read_csv(res / "filter_summary.csv")
    steps = list(summary["step"])
    assert steps[0].startswith("spaCR output")
    assert "max_vacuoles_per_cell 2" in steps
    assert any(s.startswith("exclude_edge_objects") for s in steps)
    assert "confounder_filter cell_area" in steps and "confounder_filter n_cells_well" in steps
    removed = summary.set_index("step")["objects_removed"]
    assert removed["max_vacuoles_per_cell 2"] == 16  # one in six rows has 3 vacuoles
    assert removed["exclude_edge_objects (pathogen cut by the field edge)"] == 16
    assert removed["confounder_filter n_cells_well"] > 0  # lowest-density wells dropped
    assert summary["objects_remaining"].iloc[-1] == len(cells)
    kept = pd.read_csv(res / "filter_kept_per_condition.csv")
    assert {"objects_before_filters", "objects_after_filters", "fraction_kept"} <= set(kept.columns)
    assert (cells["pathogen_prcfo_count"] <= 2).all()

    p1 = res / "panels" / "P01_treatment_effect"
    data = pd.read_csv(p1 / "data_replicates.csv")
    # unit = replicate (pathogen) mean of well means; lovastatin first (condition_order)
    assert set(data["treatment"]) == {"cm", "lovastatin"}
    assert len(data) == 4 and "n_wells" in data.columns
    test = pd.read_csv(p1 / "stats_test.csv").iloc[0]
    assert test["test"] in {"one-way ANOVA", "Welch ANOVA", "Kruskal-Wallis"}
    desc = pd.read_csv(p1 / "stats_descriptive.csv")
    assert list(desc["treatment"]) == ["lovastatin", "cm"]
    assert (res / "panels" / "P02_cell_area" / "P02_cell_area.png").exists()
    assert (res / "figures" / "F1.pdf").exists() and (res / "figures" / "F1.png").exists()
    controls = res / "panels" / "controls"
    names = sorted(p.name for p in controls.iterdir())
    # channels 0, 3 and the channel of interest 2: 3 x 4 compartments + 1 + 3 x 5 ratios;
    # the replicate (pathogen) is part of the condition, so wells are the control unit
    assert len(pd.read_csv(controls / names[0] / "data_replicates.csv")) == cells["prc"].nunique()
    assert len(names) == 12 + 1 + 15
    assert any("channel_3_cytoplasm_mean_intensity" in n for n in names)
    assert (res / "figures" / "controls_channel_2_intensity.png").exists()


def test_analyze_recruitment_defaults_write_no_new_files(tmp_path):
    from tests.test_cov_submodules_recruitment_plaques import _base_settings, _Recorder
    import spacr.io as SIO
    import spacr.plot as SPLT
    from spacr.submodules import analyze_recruitment

    df = _recruitment_frame()
    src = tmp_path / "plate"
    src.mkdir()
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(SIO, "_read_and_merge_data", lambda locs, tables, **k: (df.copy(), []))
        for name in ("plot_image_mask_overlay", "_plot_recruitment"):
            mp.setattr(SPLT, name, _Recorder())
        cells, _ = analyze_recruitment(_base_settings(src, well_qc=False, object_qc=False))
    finally:
        mp.undo()
    assert len(cells) == len(df)
    assert sorted(os.listdir(src / "results")) == ["cells.csv", "wells.csv"]


def test_analyze_recruitment_qc_defaults_write_qc_tables(tmp_path):
    from tests.test_cov_submodules_recruitment_plaques import _base_settings, _Recorder
    import spacr.io as SIO
    import spacr.plot as SPLT
    from spacr.submodules import analyze_recruitment

    df = _recruitment_frame()
    bright = df["rowID"] == "r1"
    df.loc[bright & (df["columnID"] == "c1"), "cell_channel_0_mean_intensity"] = 5000.0
    rng = np.random.default_rng(1)
    df["cell_channel_0_mean_intensity"] = df["cell_channel_0_mean_intensity"] + rng.normal(0, 2, len(df))
    src = tmp_path / "plate"
    src.mkdir()
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(SIO, "_read_and_merge_data", lambda locs, tables, **k: (df.copy(), []))
        for name in ("plot_image_mask_overlay", "_plot_recruitment"):
            mp.setattr(SPLT, name, _Recorder())
        cells, _ = analyze_recruitment(_base_settings(src, channel_dims=[0]))
    finally:
        mp.undo()
    res = src / "results"
    assert {"well_qc.csv", "object_qc_summary.csv", "filter_summary.csv"} <= set(os.listdir(res))
    qc = pd.read_csv(res / "well_qc.csv")
    bad = qc[~qc["well_qc_pass"]]
    assert list(zip(bad["rowID"], bad["columnID"])) == [("r1", "c1")]
    assert "cell_ch0_intensity_median" in bad["well_qc_reason"].iloc[0]
    assert not ((cells["rowID"] == "r1") & (cells["columnID"] == "c1")).any()
    steps = list(pd.read_csv(res / "filter_summary.csv")["step"])
    assert steps[1].startswith("well_qc") and steps[2].startswith("object_qc")


# --------------------------------------------------------------------------- analyze_replication
def _replication_src(tmp_path):
    from tests.test_replication_assay import write_db
    spec = []
    for row in ("r1", "r2"):
        for column, counts in (("c1", [1, 2, 2, 4, 4, 4, 8, 2]), ("c2", [1, 1, 1, 2, 2, 4, 1, 1])):
            for k, n in enumerate(counts):
                spec.append({"row": row, "column": column, "field": f"f{k % 2}",
                             "cell": k + 1, "n": n})
    src = write_db(tmp_path / "plate1", spec)
    db = os.path.join(src, "measurements", "measurements.db")
    with sqlite3.connect(db) as con:
        p = pd.read_sql("select * from pathogen", con)
        p["pathogen_distance_to_field_edge"] = 200.0
        p.loc[p["object_label"] == 1, "pathogen_distance_to_field_edge"] = 1.0
        p.to_sql("pathogen", con, index=False, if_exists="replace")
    return src


def test_analyze_replication_direct_count_panels_and_filters(tmp_path):
    from spacr.submodules import analyze_replication
    from tests.test_replication_assay import settings_for

    src = _replication_src(tmp_path)
    out = analyze_replication(settings_for(
        src, treatments=["rep1", "rep2"], treatment_plate_metadata=[["r1"], ["r2"]],
        replicate_column="treatment",
        panels=[{"name": "drug_vs_dmso", "group": "pathogen", "order": ["dmso", "drug"],
                 "annotations": {"drug": {"dmso": "-", "drug": "+"}}, "n_resamples": 500}],
        figures={"F2": {"panels": ["P01", "P02", "P03"], "ncol": 3}},
        exclude_edge_objects=True, max_area_quantile=0.99, field_qc=True,
        field_qc_min_objects=1))
    res = os.path.join(src, "results", "analyze_replication")
    summary = pd.read_csv(os.path.join(res, "filter_summary.csv"))
    steps = list(summary["step"])
    assert steps[1].startswith("well_qc") and steps[2].startswith("object_qc")   # QC runs first
    assert [s for s in steps if not s.startswith(("well_qc", "object_qc"))][:3] == [
        "vacuoles (after parasite filters)", "field_qc", "exclude_edge_objects (cut by the field edge)"]
    assert os.path.exists(os.path.join(res, "well_qc.csv"))
    assert summary.set_index("step").loc[
        "exclude_edge_objects (cut by the field edge)", "vacuoles_removed"] == 1
    assert os.path.exists(os.path.join(res, "field_qc.csv"))
    assert os.path.exists(os.path.join(res, "filter_kept_per_condition.csv"))
    names = sorted(os.listdir(os.path.join(res, "panels")))
    assert names == ["P01_drug_vs_dmso_bins_chi_square", "P02_drug_vs_dmso_bins_fisher",
                     "P03_drug_vs_dmso_mean_parasites_per_vacuole"]
    chi = os.path.join(res, "panels", names[0])
    table = pd.read_csv(os.path.join(chi, "contingency_table_counts.csv"), index_col=0)
    assert list(table.index) == ["dmso", "drug"]
    assert table.to_numpy().sum() == len(out["vacuoles"])
    frac = pd.read_csv(os.path.join(chi, "data_replicates.csv"))
    assert len(frac) == 4  # 2 conditions x 2 replicates
    fisher = pd.read_csv(os.path.join(res, "panels", names[1], "stats_test.csv"))
    assert fisher.loc[0, "test"].startswith("Fisher")
    bar = pd.read_csv(os.path.join(res, "panels", names[2], "data_replicates.csv"))
    assert set(bar.columns) >= {"pathogen", "treatment", "n_parasites", "n_wells"}
    assert os.path.exists(os.path.join(res, "figures", "F2.pdf"))


def test_analyze_replication_size_proxy_panels(tmp_path):
    from tests.test_cov_submodules_endodyogeny import (
        _write_endodyogeny_db, AREAS_C1, AREAS_C2, _settings)
    from spacr.submodules import analyze_replication

    root = tmp_path / "plate1"
    db = _write_endodyogeny_db(root, {"c1": AREAS_C1, "c2": AREAS_C2})
    with sqlite3.connect(db) as con:
        p = pd.read_sql("select * from pathogen", con)
        p["pathogen_equivalent_diameter_area"] = 2 * np.sqrt(p["pathogen_area"] / np.pi)
        p["pathogen_distance_to_field_edge"] = 500.0
        p.loc[0, "pathogen_distance_to_field_edge"] = 2.0
        p.to_sql("pathogen", con, index=False, if_exists="replace")
    out = analyze_replication(_settings(
        str(root), replication_method="size_proxy", panels=[{"name": "nc_vs_pc"}],
        exclude_edge_objects=True, max_area_quantile=0.95))
    res = os.path.join(str(root), "results", "analyze_replication")
    summary = pd.read_csv(os.path.join(res, "filter_summary.csv"))
    removed = summary.set_index("step")["objects_removed"]
    assert removed["exclude_edge_objects (cut by the field edge)"] == 1
    assert removed["max_area_quantile 0.95"] >= 1
    names = sorted(os.listdir(os.path.join(res, "panels")))
    assert names == ["P01_nc_vs_pc_bins_chi_square", "P02_nc_vs_pc_bins_fisher",
                     "P03_nc_vs_pc_mean_vacuole_area"]
    desc = pd.read_csv(os.path.join(res, "panels", names[2], "stats_descriptive.csv"))
    assert list(desc["condition"]) == ["Hela_nc", "Hela_pc"]
    assert out["replication_method"] == "size_proxy"


def test_replication_bin_labels_prefer_user_then_area_default():
    from spacr.submodules import _replication_bin_labels, _volume_bin_as_area
    bins = ['11180.34-22360.68', '>357770.88']
    default = {b: _volume_bin_as_area(b) for b in bins}
    assert default == {'11180.34-22360.68': '500-794', '>357770.88': '>5040'}
    assert _replication_bin_labels(bins, None, default) == ['500-794', '>5040']
    assert _replication_bin_labels(bins, ['~1', '>=8'], default) == ['~1', '>=8']
    assert _replication_bin_labels(bins, {'>357770.88': 'big'}, default) == ['500-794', 'big']


def _write_field(path, background, objects):
    """Merged array: one image channel + one mask channel; objects = [(label, value, slice)]."""
    img = np.full((40, 40), background, dtype=np.uint16)
    mask = np.zeros((40, 40), dtype=np.uint16)
    for label, value, (ys, xs) in objects:
        img[ys, xs] = value
        mask[ys, xs] = label
    np.save(path, np.stack([img, img // 2, mask], axis=-1))


def test_object_quality_filters_use_each_field_background(tmp_path):
    from spacr.submodules import _object_quality_filters, _FilterLog
    merged = tmp_path / 'merged'
    merged.mkdir()
    # field A background 100, field B background 300: the same raw 400 passes in A, fails in B
    _write_field(merged / 'A.npy', 100, [(1, 400, (slice(0, 8), slice(0, 8)))])
    _write_field(merged / 'B.npy', 300, [(1, 400, (slice(0, 8), slice(0, 8)))])
    frame = pd.DataFrame({'file_name': ['A', 'B', 'A'], 'plateID': ['p'] * 3,
                          'pathogen_channel_0_mean_intensity': [400.0, 400.0, 5000.0],
                          'pathogen_solidity': [0.97, 0.97, 0.80]},
                         index=['A_o1', 'B_o1', 'A_o2'])
    settings = {'background_filters': [{'channel': 0, 'min_ratio': 1.5, 'max_ratio': 20}],
                'min_solidity': 0.9}
    log = _FilterLog(frame)
    out = _object_quality_filters(frame, settings, 'pathogen', str(tmp_path), log, 'x')
    assert list(out.index) == ['A_o1']                 # B_o1: 400/300 < 1.5; A_o2: 50x and low solidity
    assert out['ch0_field_background'].iloc[0] == 100
    assert (tmp_path / 'results' / 'x' / 'field_background.csv').exists()
    steps = log.frame()['step'].tolist()
    assert any('background_filter' in s for s in steps) and any('min_solidity' in s for s in steps)


def test_object_quality_filters_max_quantile_per_plate(tmp_path):
    from spacr.submodules import _object_quality_filters, _FilterLog
    merged = tmp_path / 'merged'
    merged.mkdir()
    _write_field(merged / 'A.npy', 100, [(1, 200, (slice(0, 8), slice(0, 8)))])
    frame = pd.DataFrame({'file_name': ['A'] * 10, 'plateID': ['p'] * 10,
                          'pathogen_channel_0_percentile_25': np.arange(1, 11) * 100.0})
    settings = {'background_filters': {'channel': 0, 'statistic': 'percentile_25', 'max_quantile': 0.8}}
    out = _object_quality_filters(frame, settings, 'pathogen', str(tmp_path), _FilterLog(frame), 'x')
    assert len(out) == 8


def test_object_quality_filters_prefer_the_image_table(tmp_path):
    import sqlite3
    from spacr.submodules import _object_quality_filters, _FilterLog
    (tmp_path / 'measurements').mkdir()
    with sqlite3.connect(tmp_path / 'measurements' / 'measurements.db') as con:
        pd.DataFrame({'file_name': ['A'], 'plateID': ['p'], 'rowID': ['r1'], 'columnID': ['c1'],
                      'fieldID': ['f1'], 'channel_0_background_median': [100.0]}).to_sql('image', con)
    frame = pd.DataFrame({'file_name': ['A', 'A'], 'pathogen_channel_0_mean_intensity': [120.0, 400.0]})
    out = _object_quality_filters(frame, {'background_filters': {'channel': 0, 'min_ratio': 1.5}},
                                  'pathogen', str(tmp_path), _FilterLog(frame), 'x')
    assert out['pathogen_channel_0_mean_intensity'].tolist() == [400.0]   # no merged arrays needed
