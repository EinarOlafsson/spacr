"""The OPS phenotype run from one settings file (item 871, alpha).

A synthetic two-field well is segmented by a threshold stand-in for
Cellpose, measured on the CPU, merged onto planted sequencing nuclei and
aggregated; then ``ph_merged``, ``ph_tables`` and ``ph_crops`` are checked
against what was planted: one merged file per field with the layout record,
one table row per owned cell keyed by ``prcfo``, and crops that read back
from their shard offsets as the cells they were cut from. The plate A
equivalence is recorded with the item, not here.
"""
from __future__ import annotations

import json
import os
import sqlite3
import tarfile

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")
pytest.importorskip("pyarrow")
tifffile = pytest.importorskip("tifffile")

from scipy import ndimage

from spacr import ops_engine
from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call
from spacr.ops_merge import field_to_well
from spacr.ops_store import read_table, write_table

PLATE = "P_1"
SIZE = 64
CENTRES = ((16, 16), (16, 46), (46, 20), (44, 46))
GUIDES = ("AAAA", "CCCC", "GGGG", "TTTT")
GENES = ("nontargeting", "nontargeting", "GENE1", "GENE2")


class _ThresholdModel:
    """Objects are the bright regions of the last channel, grown by ``grow``."""

    def __init__(self, grow=0):
        self.grow = grow

    def eval(self, image, batch_size=8, resample=True, channels=None,
             channel_axis=MISSING_CHANNEL_AXIS, z_axis=None, normalize=True,
             rescale=None, diameter=None, flow_threshold=0.4,
             cellprob_threshold=0.0, do_3D=False, anisotropy=None,
             flow3D_smooth=0, stitch_threshold=0.0, min_size=15,
             max_size_fraction=0.4, niter=None, augment=False,
             tile_overlap=0.1, bsize=None, compute_masks=True, progress=None):
        """Label the last channel above half its maximum.

        The call is checked as Cellpose would check it; the nucleus plane is
        passed without ``channel_axis``, the two-channel cell stack with it.
        """
        check_cellpose_eval_call(image, channel_axis, require_channel_axis=False)
        from skimage.segmentation import expand_labels

        plane = np.asarray(image, float)
        if plane.ndim == 3:
            plane = plane[..., -1]
        labels, _n = ndimage.label(plane > plane.max() / 2.0)
        return (expand_labels(labels, self.grow) if self.grow else labels,
                None, None)


@pytest.fixture
def models(monkeypatch):
    """The threshold model in place of Cellpose."""
    def fake(settings, gpu, key="cellpose_model"):
        """A nucleus model or a grown cell model, by setting name."""
        return _ThresholdModel(grow=0 if "nucleus" in key else 5)

    monkeypatch.setattr(ops_engine, "_cellpose_model", fake)


def _field(site):
    """DAPI discs and wider cell-stain discs; site 1 is shifted 2 px down."""
    rng = np.random.default_rng(site)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    planes = {name: rng.normal(10, 1, (SIZE, SIZE)) for name in
              ("DAPI", "GFP", "AF750")}
    for index, (cy, cx) in enumerate(CENTRES):
        distance = np.hypot(yy - cy, xx - cx)
        planes["DAPI"][distance < 4] += 200
        planes["AF750"][distance < 9] += 120
        planes["GFP"][distance < 9] += 30 * (index + 1)
    return {k: v.astype(np.uint16) for k, v in planes.items()}


def _write_tiles(folder, sites=(0, 1)):
    """One file per channel per site, named as acquired."""
    folder.mkdir(parents=True, exist_ok=True)
    for site in sites:
        for name, plane in _field(site).items():
            tifffile.imwrite(
                folder / f"20X_DAPI-GFP-AF750_A1_{name}_Site-{site}.tif", plane)
    return folder


def _placed(db, sites=(0, 1)):
    """``ops_phenotype`` rows: fields 100 well px apart at scale 0.25."""
    frame = pd.DataFrame([{
        "plate": PLATE, "well": "A1", "site": site,
        "centre_y": 1000.0 + 100 * site, "centre_x": 1000.0, "sbs_site": 0,
        "is_anchor": int(site == 0)} for site in sites])
    frame = ops_engine._ph_transform_columns(frame, 0.25, 0.0, (SIZE, SIZE))
    write_table(db, "ops_phenotype", frame)
    return frame


SETTINGS = {"plate": PLATE, "ops_ph_channels": "DAPI,GFP,AF750",
            "ops_ph_downsample": 1, "ops_ph_texture": False, "n_workers": 1,
            "ops_gpu": False, "ops_ph_cell_model": "cyto3"}


def _sequencing(db, tmp_path, cells):
    """One sequencing nucleus under every phenotype cell, and its guide."""
    placed = read_table(db, "ops_phenotype")
    rows, barcodes = [], []
    object_id = 0
    for record in placed.itertuples():
        seed = (record.scale, np.eye(2), np.array([record.offset_y,
                                                   record.offset_x]))
        mine = cells[cells["site"] == record.site].sort_values("label")
        points = field_to_well(mine[["centroid_row", "centroid_col"]]
                               .to_numpy(float), seed)
        for (y, x), label in zip(points, mine["label"]):
            object_id += 1
            rows.append({"plate": PLATE, "well": "A1",
                         "object_id": object_id, "centroid_y": y,
                         "centroid_x": x})
            barcodes.append({"plate": PLATE, "well": "A1",
                             "object_id": object_id, "barcode": "x",
                             "mapped_guide": GUIDES[(int(label) - 1) % 4]})
    write_table(db, "ops_objects", pd.DataFrame(rows))
    write_table(db, "ops_barcodes", pd.DataFrame(barcodes))
    pd.DataFrame({"sgRNA": [g + "TTTT" for g in GUIDES],
                  "gene_symbol": GENES}).to_csv(tmp_path / "design.csv",
                                                index=False)


def _run(tmp_path, phases, **extra):
    """``run_ops`` over the synthetic well with the run's settings."""
    settings = {**SETTINGS, "dst_root": str(tmp_path / "run"),
                "phenotype_source": str(tmp_path / "tiles"),
                "ops_design": str(tmp_path / "design.csv"), **extra}
    return ops_engine.run_ops(settings, phases=phases)


@pytest.fixture
def measured(tmp_path, models):
    """A well segmented, measured and given sequencing nuclei."""
    _write_tiles(tmp_path / "tiles")
    run = tmp_path / "run"
    run.mkdir()
    db = str(run / "measurements.db")
    _placed(db)
    _run(tmp_path, ("ph_segment", "ph_measure"))
    cells = pd.read_parquet(ops_engine._ph_cells_path(str(run), "A1"))
    _sequencing(db, tmp_path, cells)
    return tmp_path


FULL = ("ph_masks", "merge", "ph_merged", "aggregate", "ph_tables",
        "ph_crops")


def test_a_settings_file_run_writes_merged_tables_and_crops(measured):
    out = _run(measured, FULL, ops_merged_root=str(measured / "nas"),
               ops_crop_control_fraction=0.0, ops_crop_masks=True,
               ops_ph_measure_gpu=False)
    run, nas = measured / "run", measured / "nas"
    layout = ops_engine._ph_merged_layout(str(nas))
    assert layout["layout"] == "ops_merged_v1"
    assert layout["channels"] == ["DAPI", "GFP", "AF750"]
    assert ops_engine._ph_merged_layout(str(run)) is None
    report = out["wells"]["A1"]["ph_merged"]
    assert report["written"] == 2 and report["resumed"] == 0
    field = ops_engine._ph_read_merged(
        ops_engine._ph_merged_path(str(nas), PLATE, "A1", 0))
    raw = _field(0)
    assert field["channels"].dtype == np.uint16
    assert field["channels"].shape == (3, SIZE, SIZE)
    assert np.array_equal(field["channels"][0], raw["DAPI"])
    with np.load(run / "A1" / "phenotype_masks" / "site-0.npz") as saved:
        assert np.array_equal(field["cells"], saved["cells"])
        assert np.array_equal(field["nuclei"], saved["nuclei"])
    assert np.array_equal(field["cytoplasm"],
                          np.where(field["nuclei"] > 0, 0, field["cells"]))
    assert field["owned"][1:].tolist() == [1, 1, 1, 1]
    meta = field["meta"]
    assert (meta["plate"], meta["well"], meta["site"]) == (PLATE, "A1", 0)
    assert meta["transform"]["scale"] == 0.25
    assert meta["image"]["cell_count"] == 4

    tables = out["ph_tables"]
    assert tables["rows"]["cell"] == 8 and tables["rows"]["image"] == 2
    with sqlite3.connect(tables["db"]) as conn:
        cell = pd.DataFrame(conn.execute(
            "SELECT prcfo, object_id, sgRNA, gene, owned, fieldID, "
            "cell_area FROM cell ORDER BY prcfo").fetchall(),
            columns=["prcfo", "object_id", "sgRNA", "gene", "owned",
                     "fieldID", "cell_area"])
        indexes = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'")}
        image = conn.execute("SELECT prcf, cell_count FROM image "
                             "ORDER BY prcf").fetchall()
    assert cell["prcfo"].iloc[0] == "P_1_r1_c1_f0_o1"
    assert cell["object_id"].is_unique and cell["owned"].eq(1).all()
    assert set(cell["gene"]) == {"nontargeting", "GENE1", "GENE2"}
    assert {"ix_cell_prcfo", "ix_cell_field", "ix_nucleus_prcfo"} <= indexes
    assert image == [("P_1_r1_c1_f0", 4), ("P_1_r1_c1_f1", 4)]
    joined = cell.merge(pd.read_parquet(
        out["aggregate"]["outputs"]["cells_parquet"]).assign(
        prcfo=lambda f: "P_1_r1_c1_f" + f["site"].astype(str) + "_o"
        + f["label"].astype(str)), on="prcfo")
    assert (joined["gene_x"] == joined["gene_y"]).all()

    crops = out["ph_crops"]["wells"]["A1"]
    assert crops["crops"] == 8 and crops["reasons"] == {"linked": 8}
    shard = nas / "data" / "cell" / PLATE / "A1" / "shard-0000.tar"
    index = pd.read_parquet(str(shard)[:-4] + ".index.parquet")
    with open(shard, "rb") as handle:
        blob = handle.read()
    assert all(blob[o:o + 4] in (b"II*\x00", b"MM\x00*")
               for o in index["offset_data"])
    with tarfile.open(shard) as tar:
        assert sorted(tar.getnames()) == sorted(index["member"])
    first = index.iloc[0]
    image, mask = ops_engine._ph_read_crop(str(shard), first["offset_data"],
                                           first["size"])
    assert image.shape == (3, 96, 96) and image.dtype == np.uint16
    with np.load(run / "A1" / "phenotype_masks" /
                 f"site-{first['site']}.npz") as saved:
        cells = saved["cells"].astype(np.int32)
        nuclei = saved["nuclei"].astype(np.int32)
    stack = np.stack([_field(first["site"])[c] for c in
                      ("DAPI", "GFP", "AF750")], -1)
    expected, expected_mask, _centre, _edge = ops_engine._ph_crop_cell(
        stack, cells, nuclei, ndimage.find_objects(cells)[first["label"] - 1],
        int(first["label"]), 96, True)
    assert np.array_equal(image, expected)
    assert np.array_equal(mask, expected_mask)
    assert set(np.unique(mask)) == {0, 1, 2}
    assert (image[:, mask == 0] == 0).all()


def test_a_rerun_resumes_merged_fields_and_crops(measured):
    _run(measured, FULL)
    again = _run(measured, ("ph_merged", "ph_crops"))
    assert again["wells"]["A1"]["ph_merged"]["resumed"] == 2
    assert again["wells"]["A1"]["ph_merged"]["written"] == 0
    assert again["ph_crops"]["wells"]["A1"]["resumed"] is True
    with pytest.raises(ValueError, match="holds parts made with"):
        _run(measured, ("ph_merged",), ops_merged_dtype="float32")
    with pytest.raises(ValueError, match="uint16 or float32"):
        _run(measured, ("ph_merged",), ops_merged_dtype="int8")


def test_float32_merged_fields_keep_the_flat_corrected_values(measured):
    run = measured / "run"
    folder = run / "flat"
    folder.mkdir()
    flats = {}
    for name in ("DAPI", "GFP", "AF750"):
        flat = np.linspace(0.5, 1.5, 16).reshape(4, 4).astype(np.float32)
        np.save(folder / f"flat_{name}.npy", flat)
        flats[name] = flat
    from spacr.ops_flatfield import correct_field, flat_planes, load_flat_maps

    info = {"method": {c: {"offset": 5.0} for c in flats},
            "maps": {c: f"flat_{c}.npy" for c in flats}}
    (folder / "flat_info.json").write_text(json.dumps(info))
    maps, offsets, _info = load_flat_maps(str(folder), list(flats))
    out = _run(measured, ("ph_merged",), ops_flatfield_maps=str(folder),
               ops_merged_dtype="float32")
    assert out["wells"]["A1"]["ph_merged"]["flatfield_md5"]
    field = ops_engine._ph_read_merged(ops_engine._ph_merged_path(
        str(run), PLATE, "A1", 1))
    stack = np.stack([_field(1)[c] for c in ("DAPI", "GFP", "AF750")], -1)
    planes = flat_planes(maps, (SIZE, SIZE))
    expected = correct_field(stack, planes, offsets, ("DAPI", "GFP", "AF750"))
    assert field["channels"].dtype == np.float32
    assert np.array_equal(np.moveaxis(field["channels"], 0, -1), expected)


def test_tables_above_the_threshold_go_to_parquet(measured):
    out = _run(measured, ("ph_masks", "merge", "aggregate", "ph_tables"),
               ops_tables_parquet_threshold=3)
    tables = out["ph_tables"]
    assert tables["wide_to_parquet"] and tables["parquet_files"] == 3
    with sqlite3.connect(tables["db"]) as conn:
        columns = [r[1] for r in conn.execute("PRAGMA table_info(cell)")]
    assert columns == list(ops_engine._PH_TABLE_KEYS[:6]) + list(
        ops_engine._PH_TABLE_KEYS[6:])
    wide = pd.read_parquet(measured / "run" / "measurements" / "cell" /
                           f"{PLATE}_A1.parquet")
    assert len(wide) == 8 and "cell_area" in wide.columns


def test_crops_per_guide_follow_the_seeded_priority(measured):
    out = _run(measured, ("ph_masks", "merge", "ph_crops"),
               ops_crop_per_guide=1, ops_crop_all_nt=False,
               ops_crop_control_fraction=0.0)
    reasons = out["ph_crops"]["wells"]["A1"]["reasons"]
    assert reasons == {"guide_sample": 2}
    called = pd.DataFrame({"object_id": [1, 2, 3, 4, 5, 6],
                           "mapped_guide": ["GGGG", "GGGG", "TTTT", "TTTT",
                                            "AAAA", ""]})
    design = pd.DataFrame({"prefix": list(GUIDES), "sgRNA": list(GUIDES),
                           "gene": list(GENES)})
    first = ops_engine._ph_crop_targets(PLATE, "A1", called, design,
                                        {"ops_crop_per_guide": 1})
    assert first == ops_engine._ph_crop_targets(
        PLATE, "A1", called, design, {"ops_crop_per_guide": 1})
    assert sorted(first.values()) == ["control_guide", "guide_sample",
                                      "guide_sample"]
    assert first[5] == "control_guide"
    assert ops_engine._ph_crop_targets(PLATE, "A1", called, design, {}) is None


def test_a_random_control_share_is_a_stable_hash(measured):
    out = _run(measured, ("ph_masks", "merge", "ph_crops"),
               ops_crop_per_guide=1, ops_crop_all_nt=False,
               ops_crop_control_fraction=1.0)
    reasons = out["ph_crops"]["wells"]["A1"]["reasons"]
    assert reasons == {"guide_sample": 2, "random": 6}
    value = ops_engine._ph_hash_fraction(PLATE, "A1", 0, 1)
    assert 0.0 <= value < 1.0
    assert value == ops_engine._ph_hash_fraction(PLATE, "A1", 0, 1)


def test_no_folder_holds_more_files_than_the_cap(tmp_path):
    folders = [ops_engine._ph_shard_folder(str(tmp_path), PLATE, "A1", n, 4)
               for n in range(7)]
    assert folders[:2] == [str(tmp_path / "data" / "cell" / PLATE / "A1")] * 2
    assert folders[2].endswith("part-0001") and folders[6].endswith("part-0003")
    members = [(f"m{n}.tif", bytes([n]) * (n + 3), {"label": n})
               for n in range(5)]
    path = str(tmp_path / "shard-0000.tar")
    index = ops_engine._ph_write_shard(path, members)
    with open(path, "rb") as handle:
        blob = handle.read()
    for row in index.itertuples():
        assert blob[row.offset_data:row.offset_data + row.size] == \
            bytes([row.label]) * (row.label + 3)


def test_the_second_acquisition_is_read_shifted_from_its_own_file(tmp_path):
    rng = np.random.default_rng(3)
    base = ndimage.gaussian_filter(rng.random((80, 80)), 2) * 4000
    first = base.astype(np.uint16)
    moved = np.roll(base, (-3, 2), axis=(0, 1)).astype(np.uint16)
    stack_a = np.stack([first, first + 1])
    stack_b = np.stack([moved, moved + 2])
    a = tmp_path / "20X_x_A1_DAPI-A594_Site-0.tif"
    b = tmp_path / "20X_x_A1_A594-AF750_Site-0.tif"
    tifffile.imwrite(a, stack_a)
    tifffile.imwrite(b, stack_b)
    files = {"DAPI-A594": str(a), "A594-AF750": str(b)}
    primary, second = ops_engine._ph_split_files(files, ["AF750"])
    assert list(primary) == ["DAPI-A594"] and list(second) == ["A594-AF750"]
    assert ops_engine._ph_secondary_pair(files, ["AF750"]) == ("A594", "A594")
    shift = ops_engine._ph_register_secondary(files, ["AF750"], 64, [])
    assert (shift["dy"], shift["dx"]) == (3, -2)
    stack = ops_engine._ph_read_field(
        files, ("DAPI", "A594", "AF750"), [],
        {"channels": ["AF750"], "dy": 3, "dx": -2})
    assert np.array_equal(stack[..., 1], first + 1)
    expected = ops_engine._ph_shift_plane(moved + 2, 3, -2)
    assert np.array_equal(stack[..., 2], expected.astype(np.float32))
    assert np.array_equal(stack[3:, :-2, 2], (first + 2)[3:, :-2])
    assert ops_engine._ph_secondary_pair(
        {"DAPI-GFP": str(a), "A594-AF750": str(b)}, ["A594", "AF750"]) == (
        "DAPI", "A594")
    with pytest.raises(ValueError, match="not two acquisitions"):
        ops_engine._ph_secondary_pair({"DAPI-GFP": str(a)}, ["AF750"])


def test_shifts_are_stored_and_an_outlier_takes_the_well_median(tmp_path,
                                                               monkeypatch):
    db = str(tmp_path / "measurements.db")
    _placed(db, sites=range(8))
    shifts = {s: (20, -30) for s in range(8)}
    shifts[6] = (60, 10)
    monkeypatch.setattr(
        ops_engine, "_ph_register_secondary",
        lambda files, channels, window, unreadable: {
            "dy": shifts[files["site"]][0], "dx": shifts[files["site"]][1],
            "raw_dy": float(shifts[files["site"]][0]),
            "raw_dx": float(shifts[files["site"]][1]), "error": 0.0,
            "pair": "A594/A594"})
    index = {1: {s: {"site": s} for s in range(8)}}
    settings = {"ops_ph_secondary_channels": "AF750",
                "ops_ph_secondary_registration": "phase_correlation"}
    report = ops_engine._ph_secondary_shifts(db, PLATE, "A1", index, settings)
    assert report["registered"] == 8 and report["fallbacks"] == 1
    rows = read_table(db, "ops_phenotype").set_index("site")
    assert rows.loc[6, ["secondary_dy", "secondary_dx"]].tolist() == [20, -30]
    assert rows.loc[6, "secondary_fallback"] == 1
    assert rows.loc[6, "secondary_raw_dy"] == 60.0
    assert ops_engine._ph_secondary_shifts(db, PLATE, "A1", index,
                                           settings) == {"resumed": 8}
    by_site = ops_engine._ph_secondary_by_site(db, PLATE, "A1", settings)
    assert by_site[0] == {"channels": ["AF750"], "dy": 20, "dx": -30}
    assert ops_engine._ph_secondary_by_site(db, PLATE, "A1", {}) == {}
    assert ops_engine._ph_secondary_shifts(
        db, PLATE, "A1", index, {"ops_ph_secondary_channels": "AF750"}) == {
        "skipped": "no second acquisition to register"}
    with pytest.raises(ValueError, match="phase_correlation"):
        ops_engine._ph_secondary({"ops_ph_secondary_channels": "AF750",
                                  "ops_ph_secondary_registration": "x"})
    with pytest.raises(ValueError, match="well_median"):
        ops_engine._ph_secondary({"ops_ph_secondary_channels": "AF750",
                                  "ops_ph_secondary_fallback": "x"})


def test_crop_channels_must_be_measured_channels(measured):
    with pytest.raises(ValueError, match="ops_crop_channels"):
        _run(measured, ("ph_masks", "merge", "ph_crops"),
             ops_crop_channels="CY5")


def test_measure_gpu_reaches_the_measurement(monkeypatch, tmp_path):
    seen = {}

    def fake(nuclei, cells, stack, channels, **kwargs):
        """Record the keywords and return an empty table."""
        seen.update(kwargs)
        return pd.DataFrame({"label": []})

    monkeypatch.setattr("spacr.ops_sample.measure_phenotype_field", fake)
    tiles = _write_tiles(tmp_path / "tiles", sites=(0,))
    masks = tmp_path / "m.npz"
    np.savez(masks, nuclei=np.zeros((SIZE, SIZE), np.uint32),
             cells=np.zeros((SIZE, SIZE), np.uint32))
    files = {name: str(tiles / f"20X_DAPI-GFP-AF750_A1_{name}_Site-0.tif")
             for name in ("DAPI", "GFP", "AF750")}
    ops_engine._ph_measure_field({
        "site": 0, "files": files, "channels": ("DAPI",),
        "mask_path": str(masks), "cell_path": str(tmp_path / "c.parquet"),
        "texture": False, "measure_gpu": True})
    assert seen["gpu"] is True


def test_crops_can_divide_by_flats_scaled_to_mean_one(measured):
    run = measured / "run"
    folder = run / "flat"
    folder.mkdir()
    channels = ("DAPI", "GFP", "AF750")
    for name in channels:
        np.save(folder / f"flat_{name}.npy",
                np.linspace(0.5, 1.0, 16).reshape(4, 4).astype(np.float32))
    (folder / "flat_info.json").write_text(json.dumps(
        {"method": {c: {"offset": 5.0} for c in channels}}))
    from spacr.ops_flatfield import correct_field, flat_planes, load_flat_maps

    maps, offsets, _info = load_flat_maps(str(folder), list(channels))
    out = _run(measured, ("ph_masks", "merge", "ph_crops"),
               ops_flatfield_maps=str(folder), ops_crop_flat_mean_one=True,
               ops_crop_control_fraction=0.0)
    assert out["ph_crops"]["options"]["flat_mean_one"] is True
    shard = run / "data" / "cell" / PLATE / "A1" / "shard-0000.tar"
    first = pd.read_parquet(str(shard)[:-4] + ".index.parquet").iloc[0]
    image, _mask = ops_engine._ph_read_crop(str(shard), first["offset_data"],
                                            first["size"])
    planes = {c: (p.astype(np.float64) / p.astype(np.float64).mean()
                  ).astype(np.float32)
              for c, p in flat_planes(maps, (SIZE, SIZE)).items()}
    stack = np.stack([_field(first["site"])[c] for c in channels], -1)
    stack = np.clip(np.rint(correct_field(stack, planes, offsets, channels)),
                    0, 65535).astype(np.uint16)
    with np.load(run / "A1" / "phenotype_masks" /
                 f"site-{first['site']}.npz") as saved:
        cells = saved["cells"].astype(np.int32)
        nuclei = saved["nuclei"].astype(np.int32)
    expected = ops_engine._ph_crop_cell(
        stack, cells, nuclei, ndimage.find_objects(cells)[first["label"] - 1],
        int(first["label"]), 96, False)[0]
    assert np.array_equal(image, expected)
