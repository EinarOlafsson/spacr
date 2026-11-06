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
                           cell_distance_to_field_edge=[50.0] * 4))
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
    df["cell_distance_to_field_edge"] = 60.0
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
    panels = sorted(os.listdir(rep / "panels"))
    assert panels[0] == "R01_representative_tiles"
    assert panels[1:5] == ["R02_recruitment_unfiltered_well", "R03_recruitment_unfiltered_replicate",
                           "R04_recruitment_filtered_well", "R05_recruitment_filtered_replicate"]
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
    stats = pd.read_csv(rep / "panels" / [p for p in ctrl if "channel_2_all_objects_filtered" in p][0]
                        / "stats.csv")
    assert set(stats.loc[stats["table"] == "test", "object_type"]) == {"cell", "nucleus",
                                                                        "cytoplasm", "pathogen"}
