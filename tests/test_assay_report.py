"""Tests for spacr.assay_report and the report option of analyze_recruitment."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from spacr import assay_report as ar  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


def _frame(n_cond=3, objects=("cell", "nucleus", "cytoplasm", "pathogen"), channels=(0, 1),
           wells_per_cond=3, cells=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_cond):
        for w in range(wells_per_cond):
            for i in range(cells):
                rec = dict(plateID="p1", rowID=f"r{c + 1}", columnID=f"c{w + 1}",
                           condition=f"cond_{c + 1:02d}", replicate=f"rep{w % 3 + 1}",
                           recruitment=1.2 + 0.1 * c + rng.normal(0, 0.05),
                           pathogen_prcfo_count=1 + i % 2, nucleus_prcfo_count=1)
                for o in objects:
                    rec[f"{o}_area"] = 100.0
                    for ch in channels:
                        rec[f"{o}_channel_{ch}_mean_intensity"] = 100 + 10 * ch + rng.normal(0, 3)
                rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- layout
def test_few_conditions_fit_one_page_and_many_paginate():
    def report_pages(n_cond, max_per_row=12):
        df = _frame(n_cond=n_cond)
        order = sorted(df["condition"].unique())
        items = []
        for k in range(4):
            spec = ar.bar_spec(ar.aggregate_to_unit(df, "recruitment", ["condition"]), "recruitment",
                               "condition", order)
            items += ar.spec_items(f"R{k + 1:02d}", spec, max_per_row)
        return ar.paginate([ar.ReportBlock("recruitment", items)]), items

    pages, items = report_pages(3)
    assert len(pages) == 1 and all(it.width == "half" for it in items)
    pages, items = report_pages(40)
    assert len(pages) > 1
    # 40 conditions at most 12 per row -> 4 balanced chunks of 10 per panel, full width
    assert len(items) == 16 and all(it.width == "full" for it in items)
    assert items[1].label == "R01 (continued)"
    for page in pages:   # nothing outside the A4 page
        for pl in page:
            x, y, w, h = pl["rect_mm"]
            assert x >= 0 and x + w <= ar.A4_MM[0] + 1e-6 and y + h <= ar.A4_MM[1] - ar.MARGIN_MM + 1e-6


def test_a_block_that_does_not_fit_starts_a_new_page():
    def item(h):
        return ar.ReportItem("R", "R", lambda fig, rect: None, "full", h)
    pages = ar.paginate([ar.ReportBlock("a", [item(150)]), ar.ReportBlock("b", [item(150)])])
    assert len(pages) == 2
    assert pages[1][0]["kind"] == "heading" and pages[1][0]["text"] == "b"


# --------------------------------------------------------------------------- object types and channels
def test_generic_object_types_and_any_number_of_channels(tmp_path):
    objs = ("cell", "nucleus", "pathogen", "organelle_a")
    df = _frame(objects=objs, channels=(0, 1, 2, 5))
    assert ar.object_types(df) == ["cell", "nucleus", "pathogen", "organelle_a"]
    long = ar.well_intensity_long(df, "condition", 5)
    assert set(long["object_type"]) == set(objs)
    assert len(long) == 4 * 3 * 3                      # object types x conditions x wells
    order = sorted(df["condition"].unique())
    spec = ar.grouped_spec(long, "condition", order, list(objs), title="channel 5")
    stats = ar.stats_long(spec)
    assert set(stats.loc[stats["table"] == "test", "object_type"]) == set(objs)
    folder = ar.save_spec_panel(str(tmp_path), "R07", "channel_5_all_objects", spec)
    assert sorted(os.listdir(folder)) == ["R07_channel_5_all_objects.pdf",
                                          "R07_channel_5_all_objects.png", "data.csv", "stats.csv"]
    assert len(pd.read_csv(os.path.join(folder, "data.csv"))) == len(long)
    counts = ar.well_counts(df, "condition", "pathogen")
    assert counts["count_source"].iloc[0] == "pathogen_prcfo_count"
    assert counts["objects_per_well"].iloc[0] == 6     # 4 cells holding 1, 2, 1, 2 vacuoles
    assert ar.well_counts(df, "condition", "cell")["objects_per_well"].iloc[0] == 4
    assert ar.well_counts(df, "condition", "organelle_a")["count_source"].iloc[0] == "organelle_a_area"


# --------------------------------------------------------------------------- tiles
def _field(path, n_cells=4, size=200, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(100, 4000, (size, size, 4)).astype(np.uint16)
    cell = np.zeros((size, size), np.uint16)
    nuc = np.zeros_like(cell)
    pat = np.zeros_like(cell)
    for i in range(n_cells):
        y, x = 50 + 100 * (i // 2), 50 + 100 * (i % 2)
        cell[y - 30:y + 30, x - 30:x + 30] = i + 1
        nuc[y - 8:y + 8, x - 20:x - 4] = i + 1
        pat[y - 6:y + 6, x + 4:x + 16] = i + 1
        img[y - 6:y + 6, x + 4:x + 16, 2] = 30000
    arr = np.concatenate([img, np.stack([cell, nuc, pat], -1)], axis=-1)
    np.save(path, arr)
    return arr


def test_tiles_three_versions_16_bit_and_raw_equal_the_source(tmp_path):
    import tifffile
    import cv2
    merged = tmp_path / "merged"
    merged.mkdir()
    src = _field(merged / "f1.npy")
    df = pd.DataFrame(dict(condition=["a"] * 4, file_name="f1", object_label=[f"o{i}" for i in range(1, 5)],
                           pathogen_distance_to_field_edge=[50.0] * 4))
    out = tmp_path / "tiles"
    res = ar.build_tiles(df, "condition", ["a"], str(merged), {"cell": 4, "nucleus": 5, "pathogen": 6},
                         {2: "red", 1: "green", 0: "blue"}, str(out), n=3, size=64, seed=1, mono_channel=2)
    table = res["table"]
    assert len(table) == 3
    for _, r in table.iterrows():
        raw = tifffile.imread(out / r["raw_tiff"])
        assert raw.dtype == np.uint16
        np.testing.assert_array_equal(np.moveaxis(raw, 0, -1), src[r.y0:r.y1, r.x0:r.x1, :])
        for v in ("merged", "mono", "outlines"):
            png = cv2.imread(str(out / r[f"{v}_png"]), cv2.IMREAD_UNCHANGED)
            assert png.dtype == np.uint16 and png.shape == (64, 64, 3)
            tif = tifffile.imread(str(out / r[f"{v}_png"]).replace(".png", ".tif"))
            np.testing.assert_array_equal(tif, png[..., ::-1])   # PNG is BGR, TIFF is RGB
        mono = cv2.imread(str(out / r["mono_png"]), cv2.IMREAD_UNCHANGED)
        assert (mono[..., 0] == mono[..., 1]).all() and (mono[..., 1] == mono[..., 2]).all()  # grey
        outl = cv2.imread(str(out / r["outlines_png"]), cv2.IMREAD_UNCHANGED)
        assert (outl != mono).any() and not (outl[..., 0] == outl[..., 2]).all()  # coloured outlines
        # centred on the pathogen object of the host cell, which lies fully inside the crop
        pat = src[..., 6] == int(r["mask_label"])
        ys, xs = np.nonzero(pat)
        assert abs((r.y0 + r.y1) / 2 - ys.mean()) <= 1 and abs((r.x0 + r.x1) / 2 - xs.mean()) <= 1
        assert ys.min() >= r.y0 and ys.max() < r.y1 and xs.min() >= r.x0 and xs.max() < r.x1
    for v in ("merged", "mono", "outlines"):
        mont = cv2.imread(str(out / v / "montage16.png"), cv2.IMREAD_UNCHANGED)
        assert mont.dtype == np.uint16 and mont.shape == (3 * 64 + 2 * 1, 64, 3)   # 1 px hairlines
    # 2nd-98th percentile of the crops of the figure, per channel
    crops = [np.moveaxis(tifffile.imread(out / f), 0, -1) for f in table["raw_tiff"]]
    px = np.concatenate([c[..., 2].ravel() for c in crops]).astype(float)
    assert res["ranges"][2] == pytest.approx(tuple(np.percentile(px, [2, 98])))


def test_same_seed_same_tiles():
    df = pd.DataFrame(dict(condition="a", object_label=[f"o{i}" for i in range(20)]))
    a = ar.select_tiles(df, "condition", ["a"], 5, seed=3)["a"].index[:5].tolist()
    b = ar.select_tiles(df, "condition", ["a"], 5, seed=3)["a"].index[:5].tolist()
    assert a == b


# --------------------------------------------------------------------------- legend
def test_legend_mentions_filters_and_tests():
    summary = pd.DataFrame(dict(step=["spaCR output (after object filters)", "field_qc",
                                      "max_vacuoles_per_cell 1"],
                                objects_remaining=[100, 90, 70], objects_removed=[0, 10, 20],
                                note=["", "", ""]))
    text = ar.filter_sentences(summary)
    assert "field_qc: 10 removed, 90 remaining" in text and "max_vacuoles_per_cell 1" in text
    assert "identical" in ar.filter_sentences(None)
    sent = ar.describe_test(dict(test="Kruskal-Wallis", p_value=0.001, significance="**"))
    assert "Kruskal-Wallis" in sent and "Shapiro-Wilk" in sent


# --------------------------------------------------------------------------- end to end
def test_analyze_recruitment_report_end_to_end(tmp_path):
    from tests.test_cov_submodules_recruitment_plaques import _base_settings, _Recorder
    from tests.test_assay_panels import _recruitment_frame, _recruitment_db
    import spacr.io as SIO
    import spacr.plot as SPLT
    from spacr.submodules import analyze_recruitment

    df = _recruitment_frame()
    df["pathogen_distance_to_field_edge"] = 60.0
    src = tmp_path / "plate"
    (src / "measurements").mkdir(parents=True)
    merged = src / "merged"
    merged.mkdir()
    for k, name in enumerate(df["file_name"].unique()):
        _field(merged / f"{name}.npy", n_cells=4, seed=k)
    df = df[df["object_label"].isin(["o1", "o2", "o3", "o4"])].copy()   # labels present in the masks
    _recruitment_db(src, df)
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(SIO, "_read_and_merge_data", lambda locs, tables, **k: (df.copy(), []))
        for name in ("plot_image_mask_overlay", "_plot_recruitment"):
            mp.setattr(SPLT, name, _Recorder())
        settings = _base_settings(
            src, plot_control=False, channel_dims=[0, 1, 3], channel_of_interest=2,
            pathogen_plate_metadata=[["c1", "c2"], ["c4", "c5"]],
            treatment_plate_metadata=[["r1", "r2"], ["r4", "r5"]],
            replicate_column="pathogen", max_vacuoles_per_cell=2, exclude_edge_objects=True,
            report=True, report_tiles_per_condition=3, report_tile_size=48)
        cells, _ = analyze_recruitment(settings)
    finally:
        mp.undo()
    rep = src / "results" / "report"
    assert (rep / "recruitment_report.pdf").exists() and (rep / "recruitment_report.png").exists()
    assert (rep / "recruitment_report_figure_filtered.pdf").exists()
    assert (rep / "recruitment_report_figure_unfiltered.pdf").exists()
    legend = (rep / "figure_legend.txt").read_text()
    assert "FIGURE: FILTERED DATA" in legend and "FIGURE: UNFILTERED DATA" in legend
    assert "max_vacuoles_per_cell 2" in legend and "exclude_edge_objects" in legend
    assert any(t in legend for t in ("one-way ANOVA", "Welch ANOVA", "Kruskal-Wallis"))
    assert "2nd and 98th percentile" in legend and "seed 0" in legend
    allp = sorted(os.listdir(rep / "panels"))
    assert allp[:2] == ["H01_heatmap_recruitment_unfiltered", "H02_heatmap_recruitment_filtered"]
    names = [p for p in allp if p.startswith("R")]
    # filtered figure first: tiles, representative, well + replicate graphs (with crops), controls
    assert names[:5] == ["R01_tiles_filtered",
                         "R02_recruitment_filtered_well", "R02_recruitment_filtered_well_crops",
                         "R03_recruitment_filtered_replicate", "R03_recruitment_filtered_replicate_crops"]
    assert "R04_cell_channel_0_intensity_filtered" in names
    assert not any("representative_per_condition" in p for p in names)
    tiles = rep / "panels" / "R01_tiles_filtered"
    for v in ("merged", "mono", "outlines"):
        assert (tiles / v / "montage16.png").exists() and (tiles / v / f"R01_tiles_filtered_{v}.pdf").exists()
    t = pd.read_csv(tiles / "data.csv")
    # the replicate (pathogen) is part of the condition: two report conditions (cm, lovastatin)
    assert t["report_condition"].nunique() == 2 and len(t) == 6
    graphs = [p for p in names if p.startswith(("R02", "R03")) and not p.endswith("_crops")]
    for p in graphs:
        st = pd.read_csv(rep / "panels" / p / "stats.csv")
        assert "test" in set(st["table"])                        # main graphs keep their tests
    unf = [p for p in names if p.endswith("recruitment_unfiltered_well")][0]
    filt = "R02_recruitment_filtered_well"
    assert pd.read_csv(rep / "panels" / unf / "data.csv")["n_objects"].sum() == len(df)
    assert pd.read_csv(rep / "panels" / filt / "data.csv")["n_objects"].sum() == len(cells)
    reps = pd.read_csv(rep / "panels" / "R03_recruitment_filtered_replicate" / "data.csv")
    assert len(reps) == 4 and "n_wells" in reps.columns
    ctrl = [p for p in names if "intensity" in p or "objects" in p or "area" in p or "per_host" in p]
    # one panel per object type x channel: 4 object types x channels 0, 1, 3 and 2
    assert sum(p.endswith("_intensity_unfiltered") for p in ctrl) == 16
    assert sum(p.endswith("_intensity_filtered") for p in ctrl) == 16
    assert sum(p.endswith("_area_filtered") for p in ctrl) == 4
    before = pd.read_csv(rep / "panels" / [p for p in ctrl if "cell_objects_per_well_unfiltered" in p][0]
                         / "data.csv")
    assert set(before["count_source"]) == {"measurements.db table cell"}
    after = pd.read_csv(rep / "panels" / [p for p in ctrl if "pathogen_objects_per_well_filtered" in p][0]
                        / "data.csv")
    assert set(after["count_source"]) == {"pathogen_prcfo_count"}
    # controls are tested again, one test per panel
    for p in ctrl:
        st = pd.read_csv(rep / "panels" / p / "stats.csv")
        assert "test" in set(st["table"])
    assert "Control graphs are tested the same way" in legend
    # forms of recruitment: well and replicate graphs per variant, both versions
    variants = [p for p in names if "_variant_" in p]
    assert any(p.endswith("variant_recruitment_filtered_well") for p in variants)
    assert any(p.endswith("variant_pathogen_channel_2_cytoplasm_outside_mean_ratio_unfiltered_replicate")
               for p in variants)
    assert (rep / "recruitment_report_figure_variants_filtered.pdf").exists()
    assert "outside ring" in legend and "FORMS OF" in legend
    # filtration steps: overall and per condition, % removed per step
    fsteps = [p for p in names if p.endswith("filtration_steps")]
    assert len(fsteps) == 1
    fs = pd.read_csv(rep / "panels" / fsteps[0] / "data.csv")
    allrows = fs[fs["group"] == "all"].sort_values("step_index")
    assert allrows["remaining"].iloc[0] == len(df) and allrows["remaining"].iloc[-1] == len(cells)
    assert {"percent_of_start", "percent_removed_by_step"} <= set(fs.columns)
    assert set(fs["group"]) - {"all"} == set(t["report_condition"])


def test_tiles_skip_cells_without_a_vacuole_or_with_one_at_the_field_edge():
    arr = np.zeros((100, 100, 3), np.uint16)
    arr[10:40, 10:40, 1] = 1          # cell 1: no pathogen
    arr[50:90, 0:40, 1] = 2           # cell 2: pathogen touching the field edge
    arr[60:70, 0:8, 2] = 5
    arr[50:90, 50:90, 1] = 3          # cell 3: pathogen well inside
    arr[65:75, 65:75, 2] = 7
    assert ar.extract_tile(arr, 1, 1, 32, center_plane=2) is None
    assert ar.extract_tile(arr, 1, 2, 32, center_plane=2) is None
    raw, box, _ = ar.extract_tile(arr, 1, 3, 32, center_plane=2)
    assert box == (54, 86, 54, 86)


def test_db_counts_use_every_object_table(tmp_path):
    import sqlite3
    db = tmp_path / "measurements.db"
    with sqlite3.connect(db) as con:
        for t, n in (("cell", 3), ("cell_1", 9), ("nucleus", 5), ("cytoplasm", 3),
                     ("organelle_a", 2), ("png_list", 1)):
            con.execute(f"create table {t} (plateID, rowID, columnID, fieldID, object_label)")
            for i in range(n):
                con.execute(f"insert into {t} values ('p%5F1', 'r1', 'c1', '1', {i})")
    assert ar.db_object_tables(str(db)) == ["cell", "nucleus", "organelle_a"]
    frame = pd.DataFrame(dict(plateID=["p_1", "p_1"], rowID="r1", columnID=["c1", "c2"], condition=["a", "b"]))
    counts = ar.db_well_counts(str(db), frame, "condition")
    got = counts.set_index(["object_type", "columnID"])["objects_per_well"]
    assert got[("cell", "c1")] == 3 and got[("nucleus", "c1")] == 5 and got[("nucleus", "c2")] == 0
    assert set(counts["count_source"]) == {"measurements.db table cell", "measurements.db table nucleus",
                                          "measurements.db table organelle_a"}


def _report_inputs(tmp_path, n_cond, channels=(0, 1, 2, 3)):
    df = _frame(n_cond=n_cond, channels=channels, wells_per_cond=4, cells=4)
    df["file_name"] = "f_" + df["rowID"] + "_" + df["columnID"]
    df["object_label"] = [f"o{i % 4 + 1}" for i in range(len(df))]
    df["pathogen_distance_to_field_edge"] = 60.0
    merged = tmp_path / "merged"
    merged.mkdir(exist_ok=True)
    for k, name in enumerate(df["file_name"].unique()):
        _field(merged / f"{name}.npy", n_cells=4, seed=k)
    order = sorted(df["condition"].unique())
    labels = {g: f"Me49GRA8comp{i}, HeLa ALG2" for i, g in enumerate(order)}
    tiles = dict(merged_dir=str(merged), mask_dims={"cell": 4, "nucleus": 5, "pathogen": 6},
                 colour_map={2: "red", 1: "green", 0: "blue"}, n=5, size=48, seed=0)
    return df, order, labels, tiles


def test_two_figures_each_with_its_legend_on_the_figure(tmp_path):
    df, order, labels, tiles = _report_inputs(tmp_path, 6)
    res = ar.build_assay_report({"unfiltered": df, "filtered": df}, str(tmp_path / "report"),
                                "recruitment", "condition", order, [0, 1, 2, 3], labels=labels,
                                replicate_column="replicate", ratio=True, tiles=tiles, crops={"n": 3},
                                heatmaps=False)
    assert set(res["figures"]) == {"filtered", "unfiltered"}
    assert res["n_legend_pages"] == 0                     # the legend is printed on each figure
    assert res["n_pages"] == res["figure_pages"]["filtered"] + res["figure_pages"]["unfiltered"]


def test_many_conditions_paginate(tmp_path):
    df, order, labels, tiles = _report_inputs(tmp_path, 30)
    res = ar.build_assay_report({"unfiltered": df, "filtered": df}, str(tmp_path / "report"),
                                "recruitment", "condition", order, [0, 1, 2, 3], labels=labels,
                                replicate_column="replicate", ratio=True, tiles=None, heatmaps=False)
    assert res["figure_pages"]["filtered"] > 1


def _crop_inputs(tmp_path):
    df, order, labels, tiles = _report_inputs(tmp_path, 3)
    return df, order, labels, tiles


def test_representative_crops_off_writes_nothing(tmp_path):
    df, order, labels, tiles = _crop_inputs(tmp_path)
    ar.build_assay_report({"unfiltered": df, "filtered": df}, str(tmp_path / "rep"), "recruitment",
                          "condition", order, [0, 1], labels=labels, tiles=tiles, crops=None, heatmaps=False)
    names = os.listdir(tmp_path / "rep" / "panels")
    assert not any(n.endswith("_crops") or "representative_per_condition" in n for n in names)


def test_representative_crops_grid_and_figure(tmp_path):
    import cv2
    import tifffile
    df, order, labels, tiles = _crop_inputs(tmp_path)
    ar.build_assay_report({"unfiltered": df, "filtered": df}, str(tmp_path / "rep"), "recruitment",
                          "condition", order, [0, 1], labels=labels, tiles=tiles, crops={"n": 3},
                          heatmaps=False)
    panels = tmp_path / "rep" / "panels"
    assert not any("representative_per_condition" in p.name for p in panels.iterdir())
    grid = panels / "R02_recruitment_filtered_well_crops"
    g = pd.read_csv(grid / "data.csv")
    assert g.groupby("condition").size().tolist() == [3] * len(order)
    assert (grid / "R02_recruitment_filtered_well_crops.pdf").exists()
    assert {"recruitment", "group_median"} <= set(g.columns)
    for _, r in g.drop_duplicates("condition").iterrows():
        sub = df[df["condition"] == r["condition"]]
        assert abs(r["group_median"] - sub["recruitment"].median()) < 1e-9
    for v in ("merged", "mono", "outlines"):
        mont = cv2.imread(str(grid / v / "montage16.png"), cv2.IMREAD_UNCHANGED)
        assert mont.dtype == np.uint16
        assert (grid / v / "R02_recruitment_filtered_well_crops_{}.png".format(v)).exists()
    raw = tifffile.imread(grid / g["raw_tiff"].iloc[0])
    assert raw.dtype == np.uint16
    # mono version: channel 2 scaled to its 2nd-98th percentile over the crops of this grid
    crops = [np.moveaxis(tifffile.imread(grid / f), 0, -1) for f in g["raw_tiff"]]
    px = np.concatenate([c[..., 2].ravel() for c in crops]).astype(float)
    lo, hi = np.percentile(px, [2, 98])
    mono = np.concatenate([cv2.imread(str(grid / f), cv2.IMREAD_UNCHANGED)[..., 0].ravel()
                           for f in g["mono_png"]]).astype(float)
    expect = np.clip((px - lo) / (hi - lo), 0, 1) * 65535
    assert np.max(np.abs(mono - expect)) <= 1.0


def test_axis_labels_or_matrix():
    from spacr.assay_panels import axis_layout, condition_parts, string_parts, MINUS
    strains = ["Me49WT", "Me49dGRA8", "Me49GRA8comp"]
    # three strains in one host: one variable varies -> labels
    parts = {s: condition_parts({"host_cells": s, "treatment": "HeLa_WT"}) for s in strains}
    a = axis_layout(strains, parts, "auto", labels={"HeLa_WT": "HeLa WT"})
    assert not a["matrix"] and a["labels"]["Me49dGRA8"] == "Me49dGRA8" and a["constant"] == ["HeLa WT"]
    # strains x hosts: matrix with one row per value
    conds = [(s, h) for s in strains for h in ("HeLa_WT", "HeLa_ALG2")]
    order = [f"{s}_{h}" for s, h in conds]
    parts = {f"{s}_{h}": condition_parts({"host_cells": s, "treatment": h}) for s, h in conds}
    a = axis_layout(order, parts, "auto", labels={"HeLa_WT": "HeLa WT", "HeLa_ALG2": "HeLa ALG2"})
    assert a["matrix"] and list(a["annotations"]) == ["Me49WT:", "Me49dGRA8:", "Me49GRA8comp:",
                                                      "HeLa WT:", "HeLa ALG2:"]
    assert a["annotations"]["HeLa WT:"][order[0]] == "+" and a["annotations"]["HeLa WT:"][order[1]] == MINUS
    # string-only conditions: two tokens vary -> matrix
    a = axis_layout(["A_x", "A_y", "B_x"], {g: string_parts(g) for g in ["A_x", "A_y", "B_x"]}, "auto")
    assert a["matrix"] and set(a["annotations"]) == {"A:", "B:", "x:", "y:"}
    # explicit styles
    assert not axis_layout(order, parts, "labels")["matrix"]
    assert axis_layout(strains, {s: condition_parts({"host_cells": s}) for s in strains}, "matrix")["matrix"]


def test_condition_factors_split_background_and_genotype():
    from spacr.assay_panels import axis_layout, condition_parts
    factors = {"Me49WT": ["Me49", "WT"], "Me49dGRA8": ["Me49", "dGRA8"], "Me49GRA8comp": ["Me49", "Comp"],
               "RHWT": ["RH", "WT"], "RHdGRA8": ["RH", "dGRA8"], "RHGRA8comp": ["RH", "Comp"]}
    strains = list(factors)
    order = [f"{s}_{h}" for s in strains for h in ("HeLa_WT", "HeLa_ALG2")]
    parts = {f"{s}_{h}": condition_parts({"host_cells": s, "treatment": h}, factors)
             for s in strains for h in ("HeLa_WT", "HeLa_ALG2")}
    labels = {"HeLa_WT": "HeLa WT", "HeLa_ALG2": "HeLa ALG2"}
    a = axis_layout(order, parts, "auto", factor_order=["HeLa WT", "HeLa ALG2", "RH", "Me49", "WT", "dGRA8",
                                                         "Comp"], labels=labels)
    assert list(a["annotations"]) == ["HeLa WT:", "HeLa ALG2:", "RH:", "Me49:", "WT:", "dGRA8:", "Comp:"]
    assert a["annotations"]["Comp:"]["Me49GRA8comp_HeLa_WT"] == "+"
    assert a["annotations"]["Me49:"]["RHWT_HeLa_WT"] != "+"
    # Me49 only, one host: only the genotype varies -> labels; Me49 and the host go to the legend
    me49 = [s for s in strains if s.startswith("Me49")]
    p2 = {s: condition_parts({"host_cells": s, "treatment": "HeLa_WT"}, factors) for s in me49}
    a = axis_layout(me49, p2, "auto", labels=labels)
    assert not a["matrix"] and [a["labels"][s] for s in me49] == ["WT", "dGRA8", "Comp"]
    assert set(a["constant"]) == {"Me49", "HeLa WT"}


def _post(pairs):
    return pd.DataFrame([dict(group_1=a, group_2=b, p_adj=p, significance="*") for a, b, p in pairs])


def test_brackets_one_factor_pairs_packed_and_capped():
    from spacr.assay_panels import condition_parts, select_bracket_pairs, pack_brackets
    strains, hosts = ["S1", "S2", "S3"], ["H1", "H2"]
    order = [f"{s}_{h}" for s in strains for h in hosts]
    parts = {f"{s}_{h}": condition_parts({"host_cells": s, "treatment": h}) for s in strains for h in hosts}
    import itertools
    post = _post([(a, b, 1e-6) for a, b in itertools.combinations(order, 2)])   # all 15 significant
    chosen, hidden = select_bracket_pairs(post, order, parts, "one_factor", 12)
    # one factor: same strain other host (3) + same host other strain (2 x 3) = 9
    assert len(chosen) == 9 and hidden == 0
    assert all(sum(dict(parts[a])[v] != dict(parts[b])[v] for v in ("host_cells", "treatment")) == 1
               for a, b in zip(chosen["group_1"], chosen["group_2"]))
    assert len(select_bracket_pairs(post, order, parts, "all", 12)[0]) == 12
    assert select_bracket_pairs(post, order, parts, "all", 12)[1] == 3
    # no factor structure -> every significant pair
    one_var = {g: [("x", g)] for g in order}
    assert len(select_bracket_pairs(post, order, one_var, "one_factor", 20)[0]) == 15
    explicit = select_bracket_pairs(post, order, parts, [[order[0], order[5]]], 12)[0]
    assert len(explicit) == 1
    # packing: neighbours share level 0, a span covering them sits higher
    lv = pack_brackets(_post([(order[0], order[1], .01), (order[2], order[3], .01),
                              (order[0], order[3], .01)]), order)
    assert lv == [0, 0, 1]


def test_bar_panel_draws_one_factor_brackets_beyond_four(tmp_path):
    import itertools
    from spacr.assay_panels import bar_spec, condition_parts, draw_bar_panel
    strains, hosts = ["S1", "S2", "S3"], ["H1", "H2"]
    order = [f"{s}_{h}" for s in strains for h in hosts]
    rows = [dict(c=g, v=10.0 * (k + 1) + d) for k, g in enumerate(order) for d in (-.1, 0, .1, .05)]
    spec = bar_spec(pd.DataFrame(rows), "v", "c", order)
    spec["stats"]["posthoc"] = _post([(a, b, 1e-6) for a, b in itertools.combinations(order, 2)])
    parts = {f"{s}_{h}": condition_parts({"s": s, "h": h}) for s in strains for h in hosts}
    fig, ax = plt.subplots()
    draw_bar_panel(ax, spec["data"], "v", "c", order, stats=spec["stats"], parts=parts)
    stars = [t for t in ax.texts if t.get_text() == "*"]
    assert len(stars) == 9                                   # more than 4: all one-factor pairs drawn
    assert ax.get_ylim()[1] > max(t.get_position()[1] for t in stars)


def test_matrix_drawn_once_under_image_versions(tmp_path):
    df, order, labels, tiles = _report_inputs(tmp_path, 3)
    from spacr.assay_panels import axis_layout, condition_parts
    parts = {g: condition_parts({"a": g.split("_")[0], "b": "x" if i % 2 else "y"}) for i, g in enumerate(order)}
    axis = axis_layout(order, parts, "matrix")
    tr = ar.build_tiles(df, "condition", order, tiles["merged_dir"], tiles["mask_dims"], tiles["colour_map"],
                        str(tmp_path / "t"), n=2, size=48, mono_channel=2)
    for side_width in (190.0, 40.0):                       # side by side, and stacked
        draw, h, _ = ar.crop_drawer(tr, order, 2, 48, ar.COMPACT, axis, None, width_mm=side_width)
        fig = ar._new_figure(side_width + 10, h + 10)
        draw(fig, (0.02, 0.02, 0.96, 0.96))
        names = [t.get_text() for t in fig.texts if t.get_text().endswith(":")]
        assert sorted(names) == sorted(axis["annotations"])       # each row name once
        plus = [t for t in fig.texts if t.get_text() in ("+", "−")]
        assert len(plus) == len(axis["annotations"]) * len(order)   # one matrix, not three


def test_outlines_are_vector_in_figures_and_upsampled_in_files(tmp_path):
    import cv2
    from matplotlib.collections import LineCollection
    df, order, labels, tiles = _report_inputs(tmp_path, 3)
    out = tmp_path / "t"
    tr = ar.build_tiles(df, "condition", order, tiles["merged_dir"], tiles["mask_dims"], tiles["colour_map"],
                        str(out), n=2, size=48, mono_channel=2)
    name = tr["table"]["outlines_png"].iloc[0]
    native = cv2.imread(str(out / name), cv2.IMREAD_UNCHANGED)
    x4 = cv2.imread(str(out / name.replace(".png", "_x4.png")), cv2.IMREAD_UNCHANGED)
    assert native.shape == (48, 48, 3) and x4.shape == (192, 192, 3) and x4.dtype == np.uint16
    assert (out / name.replace(".png", "_x4.tif")).exists() and (out / "outlines" / "montage16_x4.png").exists()
    assert len(np.unique(x4)) > len(np.unique(native))           # anti-aliased, smooth at 4x
    draw, h, _ = ar.crop_drawer(tr, order, 2, 48, ar.COMPACT, None, labels, width_mm=180)
    fig = ar._new_figure(190, h + 10)
    draw(fig, (0.02, 0.02, 0.96, 0.96))
    lines = [c for ax in fig.axes for c in ax.collections if isinstance(c, LineCollection)]
    assert len(lines) == 1 and len(lines[0].get_segments()) >= 3 * 2
    assert lines[0].get_alpha() == pytest.approx(0.6)
    assert len(fig.axes[2].images) == 1                            # grey image under the vector lines


def test_grouped_object_colours_distinct_and_no_yellow():
    from matplotlib.colors import to_rgb
    cols = ar.object_colours(["cell", "nucleus", "cytoplasm", "pathogen", "organelle_a"])
    assert len(set(cols.values())) == 5
    for c in cols.values():
        r, g, b = to_rgb(c)
        assert not (r > 0.7 and g > 0.6 and b < 0.4)               # no yellow


def test_constant_factors_are_printed_on_the_panel(tmp_path):
    from spacr.submodules import _apply_axis
    from spacr.assay_panels import bar_spec, draw_panel
    strains = ["Me49WT", "Me49dGRA8", "Me49GRA8comp"]
    frame = pd.DataFrame(dict(host_cells=strains * 4, treatment="HeLa_WT", pathogen="rep_1",
                              v=np.arange(12.0), plateID="p", rowID="r1",
                              columnID=[f"c{i}" for i in range(12)]))
    settings = {"condition_factors": {"Me49WT": ["Me49", "WT"], "Me49dGRA8": ["Me49", "dGRA8"],
                                      "Me49GRA8comp": ["Me49", "Comp"]},
                "condition_labels": {"HeLa_WT": "HeLa WT"}, "x_axis_style": "auto"}
    spec = bar_spec(frame, "v", "host_cells", strains)
    spec = _apply_axis(spec, frame, "host_cells", strains, {}, settings)
    assert spec["labels"] == {"Me49WT": "WT", "Me49dGRA8": "dGRA8", "Me49GRA8comp": "Comp"}
    assert spec["context"] == "Me49"
    fig, ax = plt.subplots()
    draw_panel(ax, spec)
    assert ax.get_xlabel() == "Me49"
    # condition column: the host is constant too and is named on the panel
    frame["condition"] = frame["host_cells"] + "_" + frame["treatment"]
    conds = [f"{s}_HeLa_WT" for s in strains]
    spec = _apply_axis(bar_spec(frame, "v", "condition", conds), frame, "condition", conds, {}, settings)
    assert {"Me49", "HeLa WT"} <= set(spec["context"].split(" · "))


def test_crop_grid_names_the_constant_factors(tmp_path):
    from spacr.assay_panels import axis_layout, condition_parts
    df, order, labels, tiles = _report_inputs(tmp_path, 3)
    parts = {g: condition_parts({"strain": g, "host": "HeLa_WT"}) for g in order}
    axis = axis_layout(order, parts, "auto", labels={"HeLa_WT": "HeLa WT"})
    tr = ar.build_tiles(df, "condition", order, tiles["merged_dir"], tiles["mask_dims"], tiles["colour_map"],
                        str(tmp_path / "t"), n=1, size=48, mono_channel=2)
    draw, h, _ = ar.crop_drawer(tr, order, 1, 48, ar.COMPACT, axis, None, width_mm=180)
    fig = ar._new_figure(190, h + 10)
    draw(fig, (0.02, 0.02, 0.96, 0.96))
    assert "HeLa WT" in [t.get_text() for t in fig.texts]
