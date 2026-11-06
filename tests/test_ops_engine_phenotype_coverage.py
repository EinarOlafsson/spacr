"""The OPS phenotype phases on small planted fields, and what they refuse.

Fields are written to disk the way the phenotype acquisition names them, and
Cellpose is replaced by a threshold model, because the engine's contract with
a segmenter is "a label image per field". Everything else is the shipped
code: reading, segmenting, pairing, measuring, resuming, merging and
reducing to guides and genes.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

tifffile = pytest.importorskip("tifffile")
pytest.importorskip("scipy")
pytest.importorskip("pyarrow")

from scipy import ndimage

from spacr import ops_engine
from spacr.ops_store import read_table, write_table

PLATE = "P1"
SIZE = 64
CENTRES = ((16, 16), (16, 46), (46, 20), (44, 46))


class _ThresholdModel:
    """A stand-in for Cellpose: objects are the bright regions of channel 0."""

    def __init__(self, grow=0):
        self.grow = grow
        self.calls = []

    def eval(self, image, batch_size=8, resample=True, channels=None,
             channel_axis=None, z_axis=None, normalize=True, rescale=None,
             diameter=None, flow_threshold=0.4, cellprob_threshold=0.0,
             do_3D=False, anisotropy=None, flow3D_smooth=0,
             stitch_threshold=0.0, min_size=15, max_size_fraction=0.4,
             niter=None, augment=False, tile_overlap=0.1, bsize=None,
             compute_masks=True, progress=None):
        """Label the last channel above half its maximum, grown by ``grow``.

        The installed cellpose's eval parameters are written out, not taken
        as ``**kwargs``, so an argument cellpose removed fails here too.
        """
        self.calls.append({"shape": np.shape(image), "batch": batch_size,
                           "diameter": diameter, "axis": channel_axis})
        plane = np.asarray(image, float)
        if plane.ndim == 3:
            plane = plane[..., -1]
        labels, _n = ndimage.label(plane > plane.max() / 2.0)
        if self.grow:
            from skimage.segmentation import expand_labels

            labels = expand_labels(labels, self.grow)
        return labels, None, None


def _field(seed=0):
    """A DAPI plane of four discs and a cell plane of wider discs around them."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    dapi = rng.normal(10, 1, (SIZE, SIZE))
    body = rng.normal(10, 1, (SIZE, SIZE))
    for cy, cx in CENTRES:
        distance = np.hypot(yy - cy, xx - cx)
        dapi[distance < 4] += 200
        body[distance < 9] += 120
    return dapi.astype(np.uint16), body.astype(np.uint16)


def _write_fields(folder, well="A1", sites=(0, 1)):
    """Phenotype tiles, one file per channel per site, named as acquired."""
    folder.mkdir(parents=True, exist_ok=True)
    for site in sites:
        dapi, body = _field(site)
        tifffile.imwrite(folder / f"20X_DAPI-AF750_{well}_DAPI_Site-{site}.tif",
                         dapi)
        tifffile.imwrite(folder / f"20X_DAPI-AF750_{well}_AF750_Site-{site}.tif",
                         body)
    return folder


def _placed(db, well="A1", sites=(0, 1), transforms=True):
    """``ops_phenotype`` rows for the sites, with or without transforms."""
    frame = pd.DataFrame([{
        "plate": PLATE, "well": well, "site": site,
        "centre_y": 1000.0 + 100 * site, "centre_x": 1000.0,
        "sbs_site": 0, "is_anchor": int(site == 0)} for site in sites])
    if transforms:
        frame = ops_engine._ph_transform_columns(frame, 0.25, 0.0,
                                                 (SIZE, SIZE))
    write_table(db, "ops_phenotype", frame)
    return frame


@pytest.fixture
def models(monkeypatch):
    """The threshold model in place of Cellpose, one per model setting."""
    built = {}

    def fake(settings, gpu, key="cellpose_model"):
        built[key] = _ThresholdModel(grow=0 if "nucleus" in key else 5)
        return built[key]

    monkeypatch.setattr(ops_engine, "_cellpose_model", fake)
    return built


SETTINGS = {"ops_ph_channels": "DAPI,AF750", "ops_ph_downsample": 1,
            "ops_ph_texture": False, "n_workers": 1}


# --- small helpers -----------------------------------------------------------

def test_site_settings_name_ranges_lists_and_nothing():
    assert ops_engine._ph_sites(None) is None
    assert ops_engine._ph_sites("") is None
    assert ops_engine._ph_sites([3, "4"]) == {3, 4}
    assert ops_engine._ph_sites("520-523, 555,,") == {520, 521, 522, 523, 555}


def test_channels_default_to_the_reference_plate_and_strip_blanks():
    assert ops_engine._ph_channels({}) == ("DAPI", "GFP", "A594", "AF750")
    assert ops_engine._ph_channels({"ops_ph_channels": " DAPI , ,GFP"}) == (
        "DAPI", "GFP")


def test_downscale_averages_blocks_and_upscale_restores_the_shape():
    plane = np.arange(25, dtype=float).reshape(5, 5)
    assert ops_engine._ph_downscale(plane, 1) is plane
    small = ops_engine._ph_downscale(plane, 2)
    assert small.shape == (3, 3)
    assert small[0, 0] == pytest.approx(np.mean([0, 1, 5, 6]))
    mask = np.array([[1, 2], [3, 0]])
    big = ops_engine._ph_upscale(mask, 2, (3, 4))
    assert big.shape == (3, 4) and big.dtype == np.int32
    assert big[0].tolist() == [1, 1, 2, 2] and big[2].tolist() == [3, 3, 0, 0]
    assert ops_engine._ph_upscale(mask, 1, (2, 2)).tolist() == mask.tolist()


# --- reading a field ---------------------------------------------------------

def test_a_field_reads_planes_of_a_stack_and_projects_focal_planes(tmp_path):
    stack = np.zeros((3, 2, 8, 8), np.uint16)  # z, channel, y, x
    stack[1, 0] = 5
    stack[2, 1] = 9
    path = tmp_path / "20X_DAPI-GFP_A1_DAPI-GFP_Site-0.tif"
    tifffile.imwrite(path, stack, metadata={"axes": "ZCYX"})
    unreadable = []
    field = ops_engine._ph_read_field({"DAPI-GFP": str(path)},
                                      ("DAPI", "GFP"), unreadable)
    assert unreadable == []
    assert field.shape == (8, 8, 2) and field.dtype == np.float32
    assert field[..., 0].max() == 5 and field[..., 1].max() == 9


def test_a_field_missing_a_channel_or_a_plane_or_a_file_is_reported(tmp_path):
    plane = tmp_path / "dapi.tif"
    tifffile.imwrite(plane, np.ones((8, 8), np.uint16))
    unreadable = []
    assert ops_engine._ph_read_field({"DAPI": str(plane)}, ("DAPI", "GFP"),
                                     unreadable) is None
    assert unreadable[-1][1] == "no GFP plane"
    two = tmp_path / "two.tif"
    tifffile.imwrite(two, np.ones((2, 8, 8), np.uint16),
                     metadata={"axes": "CYX"})
    assert ops_engine._ph_read_field({"A-B-DAPI": str(two)}, ("DAPI",),
                                     unreadable) is None
    assert unreadable[-1] == (str(two), "has no plane 2")
    broken = tmp_path / "broken.tif"
    broken.write_bytes(b"not a tiff at all")
    assert ops_engine._ph_read_field({"DAPI": str(broken)}, ("DAPI",),
                                     unreadable) is None
    assert unreadable[-1][0] == str(broken)
    assert len(unreadable) == 3


# --- segmenting a field ------------------------------------------------------

def test_segmentation_pairs_each_nucleus_with_its_cell():
    dapi, body = _field()
    stack = np.stack([dapi, body], axis=-1).astype(np.float32)
    nucleus, cell = _ThresholdModel(), _ThresholdModel(grow=5)
    nuclei, cells, counts = ops_engine._ph_segment_field(
        stack, ("DAPI", "AF750"), (nucleus, cell),
        {"ops_ph_downsample": 2, "ops_ph_batch_size": 3,
         "ops_ph_nucleus_diameter": 20, "ops_ph_cell_diameter": 40})
    assert counts["paired"] == len(CENTRES)
    assert nuclei.shape == cells.shape == (SIZE, SIZE)
    for label in range(1, len(CENTRES) + 1):
        assert np.all(cells[nuclei == label] == label)
    assert nucleus.calls[0]["shape"] == (SIZE // 2, SIZE // 2)
    assert nucleus.calls[0]["diameter"] == pytest.approx(10.0)
    assert cell.calls[0]["shape"] == (SIZE // 2, SIZE // 2, 2)
    assert cell.calls[0]["diameter"] == pytest.approx(20.0)
    assert cell.calls[0]["batch"] == 3
    assert cell.calls[0]["axis"] == -1 and nucleus.calls[0]["axis"] is None


def test_segmentation_refuses_channels_without_the_cell_stain():
    stack = np.zeros((8, 8, 2), np.float32)
    with pytest.raises(ValueError, match="cell channel"):
        ops_engine._ph_segment_field(stack, ("DAPI", "GFP"),
                                     (_ThresholdModel(), _ThresholdModel()),
                                     {})


# --- the field phases --------------------------------------------------------

def test_fields_are_segmented_measured_and_resumed(tmp_path, models):
    tiles = _write_fields(tmp_path / "pheno")
    db = str(tmp_path / "measurements.db")
    _placed(db, transforms=False)
    (tmp_path / "A1").mkdir()
    (tmp_path / "A1" / "ops_report.json").write_text(json.dumps(
        {"phenotype": {"expected_scale": 0.25}}))
    index = ops_engine._index_tiles(str(tiles), "uncycled")
    report = ops_engine._ph_fields(db, PLATE, "A1", index["A1"], SETTINGS,
                                   False, str(tmp_path), segment=True,
                                   measure=True)
    assert report["fields"] == 2
    assert report["ph_segment"]["segmented"] == 2
    assert report["ph_segment"]["objects"]["paired"] == 2 * len(CENTRES)
    assert report["ph_measure"]["measured"] == 2
    assert report["ph_measure"]["cells"] == 2 * len(CENTRES)
    assert report["ph_measure"]["features"] > 0
    assert report["unreadable"] == []
    assert set(models) == {"ops_ph_nucleus_model"}  # same model for both
    table = pd.read_parquet(tmp_path / "ops_phenotype_cells_A1.parquet")
    assert sorted(table["site"].unique()) == [0, 1]
    assert (table["plate"] == PLATE).all()
    assert (tmp_path / "A1" / "phenotype_masks" / "site-1.npz").is_file()
    migrated = read_table(db, "ops_phenotype")
    assert migrated["field_height"].tolist() == [SIZE, SIZE]
    assert migrated["scale"].tolist() == [pytest.approx(0.25)] * 2

    again = ops_engine._ph_fields(db, PLATE, "A1", index["A1"],
                                  {**SETTINGS, "ops_ph_sites": "1"}, False,
                                  str(tmp_path), segment=True, measure=True)
    assert again["fields"] == 1
    assert again["ph_segment"]["segmented"] == 0
    assert again["ph_segment"]["resumed"] == 1
    assert again["ph_segment"]["segment_seconds_per_field"] is None
    assert again["ph_measure"]["measured"] == 0
    assert again["ph_measure"]["resumed"] == 1
    assert again["ph_measure"]["measure_seconds_per_field"] is None


def test_a_separate_cell_model_is_built_and_an_unreadable_field_is_skipped(
        tmp_path, models):
    tiles = _write_fields(tmp_path / "pheno", sites=(0,))
    (tiles / "20X_DAPI-AF750_A1_DAPI_Site-1.tif").write_bytes(b"broken")
    (tiles / "20X_DAPI-AF750_A1_AF750_Site-1.tif").write_bytes(b"broken")
    db = str(tmp_path / "measurements.db")
    _placed(db)
    index = ops_engine._index_tiles(str(tiles), "uncycled")
    report = ops_engine._ph_fields(
        db, PLATE, "A1", index["A1"],
        {**SETTINGS, "ops_ph_cell_model": "cyto3"}, False, str(tmp_path),
        segment=True, measure=False)
    assert set(models) == {"ops_ph_nucleus_model", "ops_ph_cell_model"}
    assert report["ph_segment"]["segmented"] == 1
    assert "ph_measure" not in report
    assert len(report["unreadable"]) == 1
    assert report["unreadable"][0][0].endswith("Site-1.tif")


def test_measuring_without_masks_writes_an_empty_table(tmp_path, models):
    tiles = _write_fields(tmp_path / "pheno", sites=(0,))
    db = str(tmp_path / "measurements.db")
    _placed(db, sites=(0,))
    index = ops_engine._index_tiles(str(tiles), "uncycled")
    report = ops_engine._ph_fields(db, PLATE, "A1", index["A1"], SETTINGS,
                                   False, str(tmp_path), segment=False,
                                   measure=True)
    assert report["ph_measure"]["measured"] == 0
    assert report["ph_measure"]["cells"] == 0
    table = pd.read_parquet(tmp_path / "ops_phenotype_cells_A1.parquet")
    assert len(table) == 0 and {"plate", "well", "site"} <= set(table.columns)


def test_the_measure_worker_reports_a_field_it_cannot_read(tmp_path):
    result = ops_engine._ph_measure_field({
        "site": 7, "files": {"DAPI": str(tmp_path / "gone.tif")},
        "channels": ("DAPI",), "mask_path": "", "cell_path": "",
        "texture": False})
    assert result["site"] == 7 and result["cells"] is None
    assert result["unreadable"][0][0] == str(tmp_path / "gone.tif")


def test_fields_need_placed_rows(tmp_path):
    with pytest.raises(ValueError, match="no ops_phenotype rows"):
        ops_engine._ph_fields(str(tmp_path / "measurements.db"), PLATE, "A1",
                              {}, SETTINGS, False, str(tmp_path),
                              segment=True, measure=True)


# --- run_ops over the phenotype phases ----------------------------------------

def test_run_ops_segments_and_measures_from_the_phenotype_source(tmp_path,
                                                                 models):
    tiles = _write_fields(tmp_path / "pheno")
    db = str(tmp_path / "measurements.db")
    frame = _placed(db)
    other = frame.assign(well="B2")
    write_table(db, "ops_phenotype", pd.concat([frame, other]))
    (tmp_path / "A1").mkdir()
    (tmp_path / "A1" / "ops_report.json").write_text(json.dumps(
        {"phenotype": {"expected_scale": 0.25}, "kept": True}))
    out = ops_engine.run_ops(
        {**SETTINGS, "dst_root": str(tmp_path), "plate": PLATE,
         "ops_gpu": False, "phenotype_source": str(tiles)},
        phases=("ph_segment", "ph_measure"))
    assert sorted(out["wells"]) == ["A1", "B2"]
    a1 = out["wells"]["A1"]
    assert a1["kept"] is True
    assert a1["ph_fields"]["ph_measure"]["cells"] == 2 * len(CENTRES)
    assert "no phenotype tiles for well B2" in \
        out["wells"]["B2"]["ph_fields"]["skipped"]
    saved = json.loads((tmp_path / "A1" / "ops_report.json").read_text())
    assert saved["ph_fields"]["fields"] == 2


def test_run_ops_refuses_phenotype_runs_it_cannot_start(tmp_path):
    with pytest.raises(ValueError, match="dst_root"):
        ops_engine.run_ops({}, phases=("merge",))
    with pytest.raises(ValueError, match="no ops_phenotype rows"):
        ops_engine.run_ops({"dst_root": str(tmp_path), "plate": PLATE},
                           phases=("merge",))
    db = str(tmp_path / "measurements.db")
    _placed(db)
    with pytest.raises(ValueError, match="folder of phenotype tiles"):
        ops_engine.run_ops({"dst_root": str(tmp_path), "plate": PLATE,
                            "phenotype_source": str(tmp_path / "nowhere")},
                           wells=["a1"], phases=("ph_segment",))
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="is named like"):
        ops_engine.run_ops({"dst_root": str(tmp_path), "plate": PLATE,
                            "phenotype_source": str(tmp_path / "empty")},
                           phases=("ph_measure",))


def test_placed_wells_is_empty_without_the_table(tmp_path):
    db = str(tmp_path / "measurements.db")
    assert ops_engine._placed_wells(db, PLATE) == []
    _placed(db)
    assert ops_engine._placed_wells(db, PLATE) == ["A1"]
    assert ops_engine._placed_wells(db, "other") == []


# --- transforms ---------------------------------------------------------------

def test_transforms_need_placed_rows(tmp_path):
    with pytest.raises(ValueError, match="run the phenotype phase first"):
        ops_engine._ph_transforms(str(tmp_path / "measurements.db"), PLATE,
                                  "A1", str(tmp_path))


def test_transforms_complete_from_the_expected_scale(tmp_path):
    db = str(tmp_path / "measurements.db")
    _placed(db, transforms=False)
    (tmp_path / "A1").mkdir()
    (tmp_path / "A1" / "ops_report.json").write_text(json.dumps(
        {"phenotype": {"expected_scale": 0.5, "anchors": [{"site": 0}]}}))
    migrated = ops_engine._ph_transforms(db, PLATE, "A1", str(tmp_path),
                                         field_shape=(SIZE, SIZE))
    assert migrated["scale"].tolist() == [0.5, 0.5]
    assert migrated["rotation_deg"].tolist() == [0.0, 0.0]


def test_transforms_complete_from_a_partly_filled_frame(tmp_path):
    db = str(tmp_path / "measurements.db")
    frame = _placed(db, transforms=True)
    frame.loc[1, "offset_y"] = np.nan
    frame.loc[0, "rotation_deg"] = 2.0
    frame.loc[1, "rotation_deg"] = 2.0
    write_table(db, "ops_phenotype", frame)
    migrated = ops_engine._ph_transforms(db, PLATE, "A1", str(tmp_path))
    assert migrated["offset_y"].notna().all()
    assert migrated["rotation_deg"].tolist() == [2.0, 2.0]
    assert migrated["field_height"].tolist() == [SIZE, SIZE]


def test_transforms_refuse_rows_without_a_field_size(tmp_path):
    db = str(tmp_path / "measurements.db")
    _placed(db, transforms=False)
    (tmp_path / "A1").mkdir()
    (tmp_path / "A1" / "ops_report.json").write_text(json.dumps(
        {"phenotype": {"expected_scale": 0.25}}))
    with pytest.raises(ValueError, match="no field size"):
        ops_engine._ph_transforms(db, PLATE, "A1", str(tmp_path))


# --- merge ---------------------------------------------------------------------

def _cells_table(tmp_path, sites=(0,)):
    """A per-cell table whose nuclei sit at planted field positions."""
    rows = []
    for site in sites:
        for label, (cy, cx) in enumerate(CENTRES, start=1):
            rows.append({"plate": PLATE, "well": "A1", "site": site,
                         "label": label, "centroid_row": float(cy),
                         "centroid_col": float(cx),
                         "nucleus_area": 50.0 + label,
                         "nucleus_dapi_mean_intensity": 100.0 + label})
    pd.DataFrame(rows).to_parquet(tmp_path / "ops_phenotype_cells_A1.parquet")


def test_merge_refuses_a_well_without_cells_or_nuclei(tmp_path):
    db = str(tmp_path / "measurements.db")
    _placed(db)
    with pytest.raises(ValueError, match="ph_measure phase first"):
        ops_engine._ph_merge(db, PLATE, "A1", {}, str(tmp_path), False)
    _cells_table(tmp_path)
    with pytest.raises(ValueError, match="objects phase first"):
        ops_engine._ph_merge(db, PLATE, "A1", {}, str(tmp_path), False)


def test_merge_matches_the_placed_field_and_skips_an_unplaced_one(tmp_path):
    db = str(tmp_path / "measurements.db")
    placed = _placed(db, sites=(0,))
    _cells_table(tmp_path, sites=(0, 9))
    row = placed.iloc[0]
    from spacr.ops_merge import field_to_well, field_transform

    transform = field_transform(row.scale, row.rotation_deg,
                                (row.centre_y, row.centre_x), (SIZE, SIZE))
    targets = field_to_well(np.array(CENTRES, float), transform)
    write_table(db, "ops_objects", pd.DataFrame({
        "plate": PLATE, "well": "A1", "object_id": [11, 12, 13, 14],
        "centroid_y": targets[:, 0], "centroid_x": targets[:, 1]}))
    report = ops_engine._ph_merge(db, PLATE, "A1", {"ops_merge_threshold": 2},
                                  str(tmp_path), False)
    assert report["cells"] == 8 and report["fields"] == 2
    assert report["merged"] == 4
    assert report["threshold_px"] == 2.0
    merged = read_table(db, "ops_merge")
    assert sorted(merged["object_id"]) == [11, 12, 13, 14]
    assert (merged["site"] == 0).all()
    stored = read_table(db, "ops_phenotype")
    assert "refined_inliers" in stored.columns


# --- design and aggregate -------------------------------------------------------

def test_a_design_needs_guide_and_gene_columns_and_drops_shared_prefixes(
        tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"x": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="needs an sgRNA"):
        ops_engine._ph_design(str(bad), 4)
    good = tmp_path / "good.csv"
    pd.DataFrame({"guide": ["AAAAC", "AAAAG", "CCCCA", "CCCCT"],
                  "gene": ["G1", "G2", "G3", "G3"]}).to_csv(good, index=False)
    table, ambiguous = ops_engine._ph_design(str(good), 4)
    assert ambiguous == 1
    assert table["prefix"].tolist() == ["CCCC"]
    assert table["gene"].tolist() == ["G3"]


def _aggregate_inputs(tmp_path, n=120, seed=0):
    """A merged, guide-called well: controls and one gene with larger nuclei."""
    rng = np.random.default_rng(seed)
    db = str(tmp_path / "measurements.db")
    guides = ["AAAA", "CCCC", "GGGG", "TTTT"]
    genes = ["nontargeting", "nontargeting_b", "GENE1", "GENE1"]
    ids = np.arange(n) + 1
    guide_of = [guides[i % 4] for i in range(n)]
    write_table(db, "ops_merge", pd.DataFrame({
        "plate": PLATE, "well": "A1", "site": 0, "label": ids,
        "object_id": ids, "distance": 0.5, "refined": 1}))
    write_table(db, "ops_barcodes", pd.DataFrame({
        "plate": PLATE, "well": "A1", "object_id": ids, "barcode": "x",
        "mapped_guide": guide_of}))
    pd.DataFrame({"sgRNA": [g + "TTTT" for g in guides],
                  "gene_symbol": genes}).to_csv(tmp_path / "design.csv",
                                                index=False)
    shift = np.array([3.0 if guide_of[i] in ("GGGG", "TTTT") else 0.0
                      for i in range(n)])
    cells = pd.DataFrame({
        "plate": PLATE, "well": "A1", "site": 0, "label": ids,
        "nucleus_area": rng.normal(100, 5, n) + 5 * shift,
        "nucleus_hoechst_mean_intensity": rng.normal(500, 20, n),
        "cell_gfp_mean_intensity": rng.normal(50, 3, n)})
    cells.to_parquet(tmp_path / "ops_phenotype_cells_A1.parquet")
    return db


def test_aggregate_tests_genes_against_control_guides(tmp_path):
    db = _aggregate_inputs(tmp_path)
    report = ops_engine._ph_aggregate(
        db, PLATE, ["A1"],
        {"ops_design": str(tmp_path / "design.csv"), "ops_bootstrap": 200,
         "ops_ph_channels": "DAPI,GFP"}, str(tmp_path))
    assert report["guides"] == 4 and report["genes"] == 3
    assert report["control_cells"] == 60
    assert report["bootstrap"] == 200
    assert report["bootstrap_features"] == ["nucleus_area",
                                            "nucleus_hoechst_mean_intensity",
                                            "cell_gfp_mean_intensity"]
    # The DAPI column is absent, so the first nuclear intensity is used.
    assert report["mitotic_rule"]["dna"] == "nucleus_hoechst_mean_intensity"
    genes = pd.read_parquet(report["outputs"]["genes_parquet"]).set_index(
        "gene")
    assert genes.loc["GENE1", "nucleus_area"] > 2.0
    assert genes.loc["GENE1", "nucleus_area_pvalue"] < 0.05


def test_aggregate_without_a_nuclear_intensity_calls_nothing_mitotic(tmp_path):
    db = _aggregate_inputs(tmp_path)
    cells = pd.read_parquet(tmp_path / "ops_phenotype_cells_A1.parquet")
    cells.drop(columns=["nucleus_hoechst_mean_intensity"]).to_parquet(
        tmp_path / "ops_phenotype_cells_A1.parquet")
    report = ops_engine._ph_aggregate(
        db, PLATE, ["A1"], {"ops_design": str(tmp_path / "design.csv"),
                            "ops_ph_channels": "DAPI"}, str(tmp_path))
    assert report["mitotic_rule"]["dna"] == ""
    assert report["mitotic_fraction_all"] == 0.0
    assert report["bootstrap_features"] == []


def test_aggregate_refuses_missing_inputs(tmp_path):
    db = str(tmp_path / "measurements.db")
    with pytest.raises(ValueError, match="ops_design"):
        ops_engine._ph_aggregate(db, PLATE, ["A1"], {}, str(tmp_path))
    design = tmp_path / "design.csv"
    pd.DataFrame({"sgRNA": ["AAAA"], "gene": ["G"]}).to_csv(design,
                                                           index=False)
    with pytest.raises(ValueError, match="ops_merge and ops_barcodes"):
        ops_engine._ph_aggregate(db, PLATE, ["A1"],
                                 {"ops_design": str(design)}, str(tmp_path))
    _aggregate_inputs(tmp_path)
    (tmp_path / "ops_phenotype_cells_A1.parquet").unlink()
    with pytest.raises(ValueError, match="no phenotype cell table"):
        ops_engine._ph_aggregate(db, PLATE, ["A1"],
                                 {"ops_design": str(design)}, str(tmp_path))
