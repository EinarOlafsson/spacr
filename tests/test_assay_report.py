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


def test_tiles_are_16_bit_and_raw_tiles_equal_the_source(tmp_path):
    import tifffile
    import cv2
    merged = tmp_path / "merged"
    merged.mkdir()
    src = _field(merged / "f1.npy")
    df = pd.DataFrame(dict(condition=["a"] * 4, file_name="f1", object_label=[f"o{i}" for i in range(1, 5)],
                           pathogen_distance_to_field_edge=[50.0] * 4))
    out = tmp_path / "tiles"
    res = ar.build_tiles(df, "condition", ["a"], str(merged), {"cell": 4, "nucleus": 5, "pathogen": 6},
                         {2: "red", 1: "green", 0: "blue"}, str(out), n=3, size=64, seed=1)
    table = res["table"]
    assert len(table) == 3
    for _, r in table.iterrows():
        raw = tifffile.imread(out / "tiles" / r["raw_tiff"])
        assert raw.dtype == np.uint16
        np.testing.assert_array_equal(np.moveaxis(raw, 0, -1), src[r.y0:r.y1, r.x0:r.x1, :])
        png = cv2.imread(str(out / "tiles" / r["rgb16_png"]), cv2.IMREAD_UNCHANGED)
        assert png.dtype == np.uint16 and png.shape == (64, 64, 3)
        tif = tifffile.imread(str(out / "tiles" / r["rgb16_png"]).replace(".png", ".tif"))
        np.testing.assert_array_equal(tif, png[..., ::-1])   # PNG is BGR, TIFF is RGB
        # centred on the pathogen object of the host cell, which lies fully inside the crop
        pat = src[..., 6] == int(r["mask_label"])
        ys, xs = np.nonzero(pat)
        assert abs((r.y0 + r.y1) / 2 - ys.mean()) <= 1 and abs((r.x0 + r.x1) / 2 - xs.mean()) <= 1
        assert ys.min() >= r.y0 and ys.max() < r.y1 and xs.min() >= r.x0 and xs.max() < r.x1
    mont = cv2.imread(str(out / "montage_rgb16.png"), cv2.IMREAD_UNCHANGED)
    assert mont.dtype == np.uint16 and mont.shape == (3 * 64 + 2 * 4, 64, 3)
    assert set(res["ranges"]) == {0, 1, 2}


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
    legend = (rep / "figure_legend.txt").read_text()
    assert "max_vacuoles_per_cell 2" in legend and "exclude_edge_objects" in legend
    assert any(t in legend for t in ("one-way ANOVA", "Welch ANOVA", "Kruskal-Wallis"))
    assert "replicate" in legend and "seed 0" in legend
    allp = sorted(os.listdir(rep / "panels"))
    assert "R02_representative_per_condition" in allp
    assert "R03_recruitment_unfiltered_well_crops" in allp
    panels = sorted(p for p in allp if p.startswith("R") and not p.endswith("_crops")
                    and p != "R02_representative_per_condition")
    heat = sorted(p for p in os.listdir(rep / "panels") if p.startswith("H"))
    assert "H01_heatmap_recruitment_unfiltered" in heat and len(heat) % 2 == 0
    assert set(os.listdir(rep / "panels" / heat[0])) == {f"{heat[0]}.pdf", f"{heat[0]}.png", "data.csv"}
    assert panels[0] == "R01_representative_tiles"
    assert panels[1:5] == ["R03_recruitment_unfiltered_well", "R04_recruitment_unfiltered_replicate",
                           "R05_recruitment_filtered_well", "R06_recruitment_filtered_replicate"]
    for p in panels:
        files = set(os.listdir(rep / "panels" / p))
        assert {f"{p}.pdf", f"{p}.png", "data.csv", "stats.csv"} <= files
    tiles = pd.read_csv(rep / "panels" / panels[0] / "data.csv")
    # the replicate (pathogen) is part of the condition: two report conditions (cm, lovastatin)
    assert tiles["report_condition"].nunique() == 2 and len(tiles) == 6
    assert set(tiles["prcfo"] if "prcfo" in tiles else tiles["object_label"]) <= set(cells["object_label"])
    unf = pd.read_csv(rep / "panels" / panels[1] / "data.csv")
    filt = pd.read_csv(rep / "panels" / panels[3] / "data.csv")
    assert unf["n_objects"].sum() == len(df) and filt["n_objects"].sum() == len(cells)
    reps = pd.read_csv(rep / "panels" / panels[2] / "data.csv")
    assert len(reps) == 4 and "n_wells" in reps.columns
    # controls: channels 0, 1, 3 and 2, one grouped panel each, and counts per object type, twice
    ctrl = [p for p in panels[5:]]
    assert sum("all_objects_unfiltered" in p for p in ctrl) == 4
    assert sum("all_objects_filtered" in p for p in ctrl) == 4
    assert any("pathogen_objects_per_well_filtered" in p for p in ctrl)
    before = pd.read_csv(rep / "panels" / [p for p in ctrl if "cell_objects_per_well_unfiltered" in p][0]
                         / "data.csv")
    assert set(before["count_source"]) == {"measurements.db table cell"}
    after = pd.read_csv(rep / "panels" / [p for p in ctrl if "pathogen_objects_per_well_filtered" in p][0]
                        / "data.csv")
    assert set(after["count_source"]) == {"pathogen_prcfo_count"}
    assert "measurements.db" in legend
    stats = pd.read_csv(rep / "panels" / [p for p in ctrl if "channel_2_all_objects_filtered" in p][0]
                        / "stats.csv")
    assert set(stats.loc[stats["table"] == "test", "object_type"]) == {"cell", "nucleus",
                                                                        "cytoplasm", "pathogen"}


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


def test_compact_layout_fits_the_standard_figure_on_one_page(tmp_path):
    # 6 conditions, 4 channels, 3 counted object types (cell, nucleus, pathogen), tiles
    df, order, labels, tiles = _report_inputs(tmp_path, 6)
    res = ar.build_assay_report({"unfiltered": df, "filtered": df}, str(tmp_path / "report"),
                                "recruitment", "condition", order, [0, 1, 2, 3], labels=labels,
                                replicate_column="replicate", ratio=True, tiles=tiles, layout="compact")
    assert res["n_main_pages"] == 1 and res["n_legend_pages"] >= 1
    assert res["n_pages"] == res["n_figure_pages"] + res["n_legend_pages"]
    assert res["n_figure_pages"] > 1                          # plate heatmaps follow as an appendix
    counts = [p for p in res["panels"].values() if "objects_per_well" in p]
    assert len(counts) == 6                                   # 3 types, before and after


def test_compact_layout_paginates_many_conditions(tmp_path):
    df, order, labels, tiles = _report_inputs(tmp_path, 30)
    res = ar.build_assay_report({"unfiltered": df, "filtered": df}, str(tmp_path / "report"),
                                "recruitment", "condition", order, [0, 1, 2, 3], labels=labels,
                                replicate_column="replicate", ratio=True, tiles=None, layout="compact")
    assert res["n_main_pages"] > 1
    sp = ar.build_assay_report({"unfiltered": df, "filtered": df}, str(tmp_path / "report2"),
                               "recruitment", "condition", order[:6], [0, 1, 2, 3], labels=labels,
                               replicate_column="replicate", ratio=True, layout="spacious")
    assert sp["n_main_pages"] > 1                             # spacious keeps the roomy pages


def _crop_inputs(tmp_path):
    df, order, labels, tiles = _report_inputs(tmp_path, 3)
    return df, order, labels, tiles


def test_representative_crops_off_writes_nothing(tmp_path):
    df, order, labels, tiles = _crop_inputs(tmp_path)
    res = ar.build_assay_report({"unfiltered": df, "filtered": df}, str(tmp_path / "rep"), "recruitment",
                                "condition", order, [0, 1], labels=labels, tiles=tiles, crops=None,
                                heatmaps=False)
    names = os.listdir(tmp_path / "rep" / "panels")
    assert not any(n.endswith("_crops") or "representative_per_condition" in n for n in names)
    assert res["n_pages"] >= 1


def test_representative_crops_grid_and_figure(tmp_path):
    import cv2
    import tifffile
    df, order, labels, tiles = _crop_inputs(tmp_path)
    ar.build_assay_report({"unfiltered": df, "filtered": df}, str(tmp_path / "rep"), "recruitment",
                          "condition", order, [0, 1], labels=labels, tiles=tiles, crops={"n": 3},
                          heatmaps=False)
    panels = tmp_path / "rep" / "panels"
    rep = panels / "R02_representative_per_condition"
    table = pd.read_csv(rep / "data.csv")
    assert len(table) == len(order) and {"recruitment", "group_median"} <= set(table.columns)
    # the chosen object is the one nearest its condition median among those that can be cropped
    for _, r in table.iterrows():
        sub = df[df["condition"] == r["condition"]]
        assert abs(r["group_median"] - sub["recruitment"].median()) < 1e-9
    grid = panels / "R03_recruitment_unfiltered_well_crops"
    g = pd.read_csv(grid / "data.csv")
    assert g.groupby("condition").size().tolist() == [3] * len(order)
    assert (grid / "R03_recruitment_unfiltered_well_crops.pdf").exists()
    mont = cv2.imread(str(grid / "montage16.png"), cv2.IMREAD_UNCHANGED)
    assert mont.dtype == np.uint16
    raw = tifffile.imread(grid / "crops" / g["raw_tiff"].iloc[0])
    assert raw.dtype == np.uint16
    # 1st-99th percentile scaling over the crops of the figure: about 1 % of the
    # pixels of a channel sit at or above full scale
    crops = [np.moveaxis(tifffile.imread(grid / "crops" / f), 0, -1) for f in g["raw_tiff"]]
    px = np.concatenate([c[..., 2].ravel() for c in crops]).astype(float)
    lo, hi = np.percentile(px, [1, 99])
    rgb = [cv2.imread(str(grid / "crops" / f), cv2.IMREAD_UNCHANGED)[..., ::-1] for f in g["rgb16_png"]]
    red = np.concatenate([x[..., 0].ravel() for x in rgb]).astype(float)
    expect = np.clip((px - lo) / (hi - lo), 0, 1) * 65535
    inside = np.concatenate([np.ones(c.shape[:2], bool).ravel() for c in crops])
    # every pixel off the drawn outlines (about 8 % here) matches the 1-99 % scaling
    assert np.mean(np.abs(red[inside] - expect[inside]) <= 2) > 0.9
