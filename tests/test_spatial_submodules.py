"""Spatial Mask Generation and Spatial Map Barcodes (item 534), end to end.

The fixtures are small slides written in the platforms' own layouts: a
Xenium bundle (tiled, pyramidal morphology OME-TIFF, ``transcripts.parquet``
with ``cell_id``, ``experiment.xenium``) and a Visium HD ``outs`` folder (a
striped RGB H&E TIFF, ``binned_outputs/square_002um`` with a 10x
feature-barcode ``.h5``, ``tissue_positions.parquet`` and
``scalefactors_json.json``). Nuclei sit on a grid, so every count asserted
can be worked out by hand.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from spacr import ops_engine as oe
from spacr.tabular import database_tables, read_table
from tests.cellpose_api_contract import assert_declares_installed_eval_signature
from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call

pytest.importorskip("pyarrow")

PIXEL = 0.5
SHAPE = (300, 420)
PITCH = 50
RADIUS = 7
EXPAND_UM = 5.0
INFECTED_EVERY = 3


def _centres():
    """Nucleus centres (row, column) on the grid, 6 x 8 of them."""
    return [(PITCH // 2 + r * PITCH, PITCH // 2 + c * PITCH)
            for r in range(SHAPE[0] // PITCH) for c in range(SHAPE[1] // PITCH)]


def _nuclei():
    """The planted nucleus labels, numbered in raster order from 1."""
    labels = np.zeros(SHAPE, np.int32)
    rows, cols = np.ogrid[:SHAPE[0], :SHAPE[1]]
    for label, (r, c) in enumerate(_centres(), start=1):
        labels[(rows - r) ** 2 + (cols - c) ** 2 <= RADIUS ** 2] = label
    return labels


def _infected(index):
    """Whether nucleus ``index`` (from 0) carries the pathogen."""
    return index % INFECTED_EVERY == 0


def _pyramid(path, image, tile=64):
    """Write a tiled two-level pyramidal OME-TIFF."""
    import tifffile

    with tifffile.TiffWriter(path, ome=True) as handle:
        handle.write(image, tile=(tile, tile), subifds=1,
                     metadata={"axes": "YX"})
        handle.write(image[::2, ::2], tile=(tile, tile), subfiletype=1)


def xenium_slide(root):
    """A Xenium bundle: DAPI disks, GENE_A in every nucleus, GENE_B and two
    CHIKV probe reads in every third, and reads in the background."""
    root = str(root)
    os.makedirs(os.path.join(root, "morphology_focus"), exist_ok=True)
    nuclei = _nuclei()
    image = np.where(nuclei > 0, 2000, 100).astype(np.uint16)
    _pyramid(os.path.join(root, "morphology_focus",
                          "morphology_focus_0000.ome.tif"), image)
    rng = np.random.default_rng(1)
    rows = []
    for index, (r, c) in enumerate(_centres()):
        plan = {"GENE_A": 6, "GENE_B": 5 if _infected(index) else 0,
                "CHIKV_sg": 2 if _infected(index) else 0}
        for gene, count in plan.items():
            for _ in range(count):
                dy, dx = rng.uniform(-3, 3, 2)
                rows.append((gene, (c + dx) * PIXEL, (r + dy) * PIXEL, 30.0,
                             f"cell{index + 1}"))
    rows += [("GENE_A", 2.0 * PIXEL, 2.0 * PIXEL, 30.0, "UNASSIGNED"),
             ("GENE_A", 5.0 * PIXEL, 5.0 * PIXEL, 5.0, "UNASSIGNED"),
             ("NegControlProbe_1", 5.0 * PIXEL, 5.0 * PIXEL, 30.0,
              "UNASSIGNED")]
    frame = pd.DataFrame(rows, columns=["feature_name", "x_location",
                                        "y_location", "qv", "cell_id"])
    frame["z_location"] = 1.0
    frame.to_parquet(os.path.join(root, "transcripts.parquet"))
    with open(os.path.join(root, "experiment.xenium"), "w") as handle:
        json.dump({"pixel_size": PIXEL}, handle)
    return root


def _write_h5(path, barcodes, genes, dense):
    """A 10x feature-barcode matrix, barcodes by genes."""
    h5py = pytest.importorskip("h5py")
    from scipy import sparse

    csc = sparse.csc_matrix(np.asarray(dense).T)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("matrix")
        group["barcodes"] = np.array(barcodes, dtype="S")
        group["data"] = csc.data
        group["indices"] = csc.indices
        group["indptr"] = csc.indptr
        group["shape"] = np.array(csc.shape)
        features = group.create_group("features")
        features["name"] = np.array(genes, dtype="S")
        features["id"] = np.array(genes, dtype="S")
        features["feature_type"] = np.array(["Gene Expression"] * len(genes),
                                            dtype="S")


def visium_hd_slide(root, bin_px=4):
    """A Visium HD outs folder: an RGB H&E TIFF with purple nuclei and 2 um
    bins on a 4 px grid, GENE_A one UMI in every bin inside a nucleus."""
    import tifffile

    outs = os.path.join(str(root), "outs")
    folder = os.path.join(outs, "binned_outputs", "square_002um")
    os.makedirs(os.path.join(folder, "spatial"), exist_ok=True)
    nuclei = _nuclei()
    image = np.empty(SHAPE + (3,), np.uint8)
    image[:] = (235, 200, 220)
    image[nuclei > 0] = (90, 50, 140)
    tifffile.imwrite(os.path.join(outs, "slide_tissue_image.tif"), image,
                     photometric="rgb", rowsperstrip=16)
    ys, xs = np.mgrid[bin_px // 2:SHAPE[0]:bin_px, bin_px // 2:SHAPE[1]:bin_px]
    xs, ys = xs.ravel().astype(float), ys.ravel().astype(float)
    barcodes = [f"s_002um_{i:05d}-1" for i in range(len(xs))]
    inside = nuclei[ys.astype(int), xs.astype(int)]
    dense = np.zeros((len(xs), 3), np.int64)
    dense[:, 0] = inside > 0
    dense[:, 1] = [1 if label and _infected(label - 1) else 0
                   for label in inside]
    dense[:, 2] = 1
    _write_h5(os.path.join(folder, "filtered_feature_bc_matrix.h5"),
              barcodes, ["GENE_A", "GENE_B", "GENE_BG"], dense)
    pd.DataFrame({"barcode": barcodes, "in_tissue": 1,
                  "array_row": (ys // bin_px).astype(int),
                  "array_col": (xs // bin_px).astype(int),
                  "pxl_row_in_fullres": ys, "pxl_col_in_fullres": xs}
                 ).to_parquet(os.path.join(folder, "spatial",
                                           "tissue_positions.parquet"))
    with open(os.path.join(folder, "spatial", "scalefactors_json.json"),
              "w") as handle:
        json.dump({"microns_per_pixel": PIXEL,
                   "spot_diameter_fullres": 2.0 / PIXEL,
                   "tissue_hires_scalef": 0.25,
                   "tissue_lowres_scalef": 0.025}, handle)
    return outs, int(dense[:, 0].sum())


def _plant(image):
    """An injected segmenter that returns the planted nuclei of a window,
    found by thresholding the stain image (inverted grey for colour)."""
    from skimage.measure import label

    if image.ndim == 3:
        image = 255.0 - image.astype(float).mean(axis=-1)
    return label(image > np.percentile(image, 50) if image.max() > image.min()
                 else np.zeros(image.shape, bool)).astype(np.int32)


def _mask_settings(source, dst, **over):
    settings = {"spatial_source": str(source), "spatial_dst": str(dst),
                "spatial_sample": "slide one", "spatial_tile_size": 128,
                "spatial_tile_overlap": 48, "spatial_segmenter": "threshold",
                "spatial_cell_expansion_um": EXPAND_UM}
    settings.update(over)
    return settings


@pytest.fixture(scope="module")
def xenium_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("xenium")
    source = xenium_slide(root / "bundle")
    dst = root / "fields"
    report = oe.run_spatial_masks(_mask_settings(source, dst))
    return {"source": source, "dst": str(dst), "report": report}


def _measure(report):
    from spacr.measure import measure_crop
    from spacr.settings import get_measure_crop_settings

    settings = get_measure_crop_settings({})
    measure = report["measure"]
    settings.update({
        "src": measure["src"], "channels": measure["channels"],
        "cell_mask_dim": measure["cell_mask_dim"],
        "nucleus_mask_dim": measure["nucleus_mask_dim"],
        "pathogen_mask_dim": None, "cell_min_size": 0, "nucleus_min_size": 0,
        "pathogen_min_size": 0, "cytoplasm_min_size": 0, "cytoplasm": False,
        "save_png": False, "save_arrays": False, "plot": False,
        "verbose": False, "n_jobs": 1})
    measure_crop(settings)
    return os.path.join(os.path.dirname(measure["src"]), "measurements",
                        "measurements.db")


def test_every_nucleus_is_kept_once_and_whole_across_the_seams(xenium_run):
    report = xenium_run["report"]
    dst = xenium_run["dst"]
    assert report["windows"] > 4
    assert report["objects"] == len(_centres())
    assert report["unseen_objects"] == 0
    db = report["db"]
    objects = read_table(db, table="spatial_objects", canonicalise=False,
                         report=None)
    fields = read_table(db, table="spatial_fields", canonicalise=False,
                        report=None)
    assert len(objects) == len(_centres())
    assert objects["prcfo"].is_unique
    found = np.sort(objects[["centroid_y", "centroid_x"]].round().to_numpy(),
                    axis=0)
    assert np.array_equal(found, np.sort(np.array(_centres(), float), axis=0))
    grown = RADIUS + EXPAND_UM / PIXEL
    expected = np.pi * grown ** 2
    for _i, field in fields.iterrows():
        merged = np.load(os.path.join(dst, "merged",
                                      f"{field['file_name']}.npy"))
        assert merged.dtype == np.uint16 and merged.shape[-1] == 3
        cells, nuclei = merged[..., 1], merged[..., 2]
        labels = np.unique(cells[cells > 0])
        assert list(labels) == list(range(1, field["objects"] + 1))
        assert set(np.unique(nuclei[nuclei > 0])) <= set(labels)
        for label in labels:
            area = int((cells == label).sum())
            assert abs(area - expected) / expected < 0.15
    layout = json.load(open(os.path.join(dst, "merged",
                                         ".spacr_plane_layout.json")))
    assert layout["mask_dims"] == {"cell": 1, "nucleus": 2}
    assert not os.path.exists(os.path.join(dst, "spatial", "_windows"))
    stored = json.load(open(os.path.join(dst, "spatial_mask_report.json")))
    assert stored["fields"] == len(fields)


def test_measure_measures_the_fields_like_mask_generation_output(xenium_run):
    db = _measure(xenium_run["report"])
    tables = database_tables(db)
    assert {"cell", "nucleus"} <= set(tables)
    cell = oe._st_measured(db, "cell")
    objects = read_table(db, table="spatial_objects", canonicalise=False,
                         report=None)
    assert len(cell) == len(_centres())
    assert set(cell.index) == set(objects["prcfo"])
    xenium_run["measured"] = db


def test_xenium_reads_are_counted_into_the_cells_and_join_measure(xenium_run):
    if "measured" not in xenium_run:
        xenium_run["measured"] = _measure(xenium_run["report"])
    summary = oe.run_spatial_map({
        "spatial_source": xenium_run["source"],
        "spatial_masks_dir": xenium_run["dst"],
        "spatial_pathogen_genes": "CHIKV*",
        "spatial_min_pathogen_transcripts": 2, "spatial_gene": "GENE_B",
        "spatial_anndata": False})
    db = summary["db"]
    wide = read_table(db, table="cell_expression", canonicalise=False,
                      report=None)
    cell = oe._st_measured(db, "cell")
    joined = wide.merge(cell, left_on="prcfo", right_index=True,
                        how="inner")
    assert len(joined) == len(_centres()) == len(wide)
    objects = read_table(db, table="spatial_objects", canonicalise=False,
                         report=None)
    placed = wide.merge(objects, on="prcfo")
    for _i, row in placed.iterrows():
        index = _centres().index((round(row["centroid_y"]),
                                  round(row["centroid_x"])))
        assert row["expr_GENE_A"] == 6
        assert row["expr_GENE_B"] == (5 if _infected(index) else 0)
    infected = sum(_infected(i) for i in range(len(_centres())))
    assert summary["infection"]["infected"] == infected
    assert summary["infection"]["rule"] == "pathogen_genes"
    assert summary["xenium_assigned_in_spacr_cell"] == 1.0
    assert summary["points"] == 6 * 48 + (5 + 2) * infected + 1
    assert summary["points_in_fields"] == summary["points"]
    assert summary["joined_measure_rows"] == len(_centres())
    assert os.path.exists(summary["files"]["overlay"])
    assert os.path.exists(summary["files"]["infected_vs_uninfected"])
    assert os.path.exists(summary["files"]["distance_trend"])
    long = read_table(db, table="nucleus_expression_long", canonicalise=False,
                      report=None)
    assert set(long["gene"]) <= {"GENE_A", "GENE_B", "CHIKV_sg"}
    again = oe.run_spatial_map({
        "spatial_source": xenium_run["source"],
        "spatial_masks_dir": xenium_run["dst"], "spatial_objects": "cell",
        "spatial_anndata": False})
    rerun = read_table(again["db"], table="cell_expression",
                       canonicalise=False, report=None)
    assert len(rerun) == len(_centres())
    assert "infection" not in again


def test_visium_hd_bins_are_registered_through_the_full_resolution_image(
        tmp_path):
    outs, umis_in_nuclei = visium_hd_slide(tmp_path / "hd")
    dst = tmp_path / "fields"
    report = oe.run_spatial_masks(_mask_settings(
        outs, dst, spatial_cell_expansion_um=0.0, spatial_level=3))
    assert report["platform"] == "visium_hd"
    assert report["channels"] == 3 and report["level"] == 0
    assert report["objects"] == len(_centres())
    db = _measure(report)
    summary = oe.run_spatial_map({
        "spatial_source": outs, "spatial_masks_dir": str(dst),
        "spatial_bin_um": 2, "spatial_objects": "cell",
        "spatial_gene": "GENE_A", "spatial_anndata": False})
    assert summary["platform"] == "visium_hd"
    wide = read_table(db, table="cell_expression", canonicalise=False,
                      report=None)
    assert len(wide.merge(oe._st_measured(db, "cell"), left_on="prcfo",
                          right_index=True)) == len(_centres())
    shared = wide["expr_GENE_A"].sum() / umis_in_nuclei
    assert 0.8 < shared <= 1.0
    assert (wide["expr_GENE_A"] > 0).all()
    assert "cell_spot_coverage" in database_tables(db)
    assert 0.0 < summary["umi_fraction_assigned"] < 1.0


def test_um_and_pixel_coordinates_round_trip_through_a_field():
    xy_um = np.array([[10.0, 20.0], [33.3, 7.25]])
    full = xy_um / PIXEL
    field = {"scale_x": 2.0, "scale_y": 2.0, "x0": 5, "y0": 3}
    local = oe._st_field_frame(full, field)
    back = (local + [field["x0"], field["y0"]]) * 2.0 * PIXEL
    assert np.allclose(back, xy_um)


def test_scalefactors_take_full_resolution_bins_to_hires(tmp_path):
    outs, _umis = visium_hd_slide(tmp_path / "hd")
    bundle = oe._st_read_bundle(outs, bin_um=2)
    full, radius = oe._st_slide_xy(bundle)
    hires, hires_radius, _microns = oe._st_platform_xy(bundle, "hires", 0)
    assert np.allclose(hires / 0.25, full)
    assert radius == pytest.approx(2.0) and hires_radius == pytest.approx(0.5)


def test_a_window_and_level_segment_only_that_part(tmp_path):
    source = xenium_slide(tmp_path / "bundle")
    report = oe.run_spatial_masks(_mask_settings(
        source, tmp_path / "fields", spatial_level=1,
        spatial_window="0, 0, 100, 75", spatial_window_unit="um",
        spatial_tile_size=96, spatial_tile_overlap=32,
        spatial_cell_expansion_um=2.0), segment=_plant)
    assert report["level"] == 1
    assert report["region_px"] == [0, 0, 100, 75]
    assert report["segmenter"] == "custom"
    fields = read_table(report["db"], table="spatial_fields",
                        canonicalise=False, report=None)
    assert set(fields["scale_x"]) == {2.0}
    assert report["objects"] == sum(
        1 for r, c in _centres() if r < 150 - RADIUS and c < 200 - RADIUS)


def test_cellpose_is_driven_through_the_ops_model(tmp_path, monkeypatch):
    calls = []

    class _Model:
        def eval(self, x, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None, normalize=True,
                 rescale=None, diameter=None, flow_threshold=0.4,
                 cellprob_threshold=0.0, do_3D=False, anisotropy=None,
                 flow3D_smooth=0, stitch_threshold=0.0, min_size=15,
                 max_size_fraction=0.4, niter=None, augment=False,
                 tile_overlap=0.1, bsize=None, compute_masks=True,
                 progress=None):
            check_cellpose_eval_call(x, channel_axis)
            calls.append({"batch_size": batch_size, "diameter": diameter,
                          "channel_axis": channel_axis})
            return _plant(x), [None], None

    assert_declares_installed_eval_signature(_Model.eval)
    monkeypatch.setattr(oe, "_cellpose_model",
                        lambda settings, gpu: calls.append(
                            (settings, gpu)) or _Model())
    outs, _umis = visium_hd_slide(tmp_path / "hd")
    report = oe.run_spatial_masks(_mask_settings(
        outs, tmp_path / "fields", spatial_segmenter="cellpose",
        spatial_stain="rgb", spatial_diameter=14,
        spatial_cell_expansion_um=0.0))
    assert report["objects"] == len(_centres())
    assert calls[0] == ({"cellpose_model": "cpsam"}, False)
    assert calls[1] == {"batch_size": 16, "diameter": 14.0,
                        "channel_axis": -1}


def test_a_rerun_replaces_the_slide_and_keeps_others(tmp_path):
    source = xenium_slide(tmp_path / "bundle")
    dst = tmp_path / "fields"
    oe.run_spatial_masks(_mask_settings(source, dst))
    oe.run_spatial_masks(_mask_settings(source, dst,
                                        spatial_sample="slide_two"))
    oe.run_spatial_masks(_mask_settings(source, dst))
    db = os.path.join(dst, "measurements", "measurements.db")
    objects = read_table(db, table="spatial_objects", canonicalise=False,
                         report=None)
    assert sorted(objects["plateID"].unique()) == ["slide one", "slide_two"]
    assert len(objects) == 2 * len(_centres())



@pytest.mark.parametrize("keep_other_slide", [False, True])
def test_empty_rerun_retires_only_its_slide_rows(tmp_path, keep_other_slide):
    import hashlib
    import pathlib
    import sqlite3

    source = xenium_slide(tmp_path / "bundle")
    dst = tmp_path / "fields"
    settings = _mask_settings(source, dst)
    oe.run_spatial_masks(settings)
    if keep_other_slide:
        oe.run_spatial_masks(_mask_settings(source, dst,
                                            spatial_sample="slide_two"))
    db = str(dst / "measurements" / "measurements.db")
    tables = ("spatial_fields", "spatial_objects")
    before = {table: read_table(db, table=table, canonicalise=False,
                               report=None) for table in tables}
    with sqlite3.connect(db) as connection:
        schemas = {table: connection.execute(
            f'PRAGMA table_info("{table}")').fetchall() for table in tables}
    own_stems = set(before["spatial_fields"].loc[
        before["spatial_fields"]["plateID"] == "slide one", "file_name"])
    other_stems = set(before["spatial_fields"]["file_name"]) - own_stems
    assert own_stems
    assert bool(other_stems) == keep_other_slide
    other_files = {str(path.relative_to(dst)): hashlib.sha256(
        path.read_bytes()).hexdigest() for path in dst.rglob("*.npy")
                   if path.stem in other_stems}
    assert bool(other_files) == keep_other_slide
    input_files = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in pathlib.Path(source).rglob("*") if path.is_file()}

    report = oe.run_spatial_masks(
        settings, segment=lambda image: np.zeros(image.shape, np.int32))
    assert report["fields"] == report["objects"] == 0
    for table in tables:
        actual = read_table(db, table=table, canonicalise=False, report=None)
        expected = before[table].loc[
            before[table]["plateID"] != "slide one"].reset_index(drop=True)
        assert list(actual.columns) == list(expected.columns)
        assert len(actual) == len(expected)
        if keep_other_slide:
            pd.testing.assert_frame_equal(actual, expected)
        with sqlite3.connect(db) as connection:
            assert connection.execute(
                f'PRAGMA table_info("{table}")').fetchall() == schemas[table]
    assert not any(path.stem in own_stems for path in dst.rglob("*.npy"))
    assert {name: hashlib.sha256((dst / name).read_bytes()).hexdigest()
            for name in other_files} == other_files
    fields = read_table(db, table="spatial_fields", canonicalise=False,
                        report=None)
    assert all((dst / "merged" / f"{name}.npy").is_file()
               for name in fields["file_name"])
    before_repeat = pathlib.Path(db).read_bytes()
    oe.run_spatial_masks(
        settings, segment=lambda image: np.zeros(image.shape, np.int32))
    assert pathlib.Path(db).read_bytes() == before_repeat
    assert {name: hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()
            for name in input_files} == input_files


def test_bad_inputs_are_refused_with_the_reason(tmp_path):
    with pytest.raises(ValueError, match="spatial_source"):
        oe.run_spatial_masks({"spatial_source": str(tmp_path / "none")})
    source = xenium_slide(tmp_path / "bundle")
    with pytest.raises(ValueError, match="smaller than"):
        oe.run_spatial_masks(_mask_settings(source, tmp_path / "f",
                                            spatial_tile_overlap=128))
    with pytest.raises(ValueError, match="spatial_segmenter"):
        oe.run_spatial_masks(_mask_settings(source, tmp_path / "f",
                                            spatial_segmenter="magic"))
    with pytest.raises(ValueError, match="spatial_masks_dir"):
        oe.run_spatial_map({"spatial_source": source})
    with pytest.raises(ValueError, match="spatial_source"):
        oe.run_spatial_map({"spatial_masks_dir": str(tmp_path)})
    with pytest.raises(ValueError, match="no spatial_fields"):
        oe.run_spatial_map({"spatial_source": source,
                            "spatial_masks_dir": str(tmp_path)})
    oe.run_spatial_masks(_mask_settings(source, tmp_path / "f"))
    with pytest.raises(ValueError, match="no pathogen plane"):
        oe.run_spatial_map({"spatial_source": source,
                            "spatial_masks_dir": str(tmp_path / "f"),
                            "spatial_objects": "pathogen"})
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "transcripts.parquet").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="morphology"):
        oe._st_slide(str(empty))
    hd = tmp_path / "hd" / "outs"
    (hd / "binned_outputs").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="microscope image"):
        oe._st_slide(str(hd))
    import tifffile
    tifffile.imwrite(str(hd / "image.tif"), np.zeros((8, 8, 3), np.uint8))
    with pytest.raises(FileNotFoundError, match="microns_per_pixel"):
        oe._st_slide(str(hd))


def test_the_slide_reader_takes_single_images_and_pngs(tmp_path):
    import tifffile
    from matplotlib import image as mpimage

    bundle = tmp_path / "x"
    bundle.mkdir()
    tifffile.imwrite(str(bundle / "morphology.ome.tif"),
                     np.zeros((3, 16, 16), np.uint16))
    slide = oe._st_slide(str(bundle), "xenium")
    assert slide["channels"][0].endswith("morphology.ome.tif")
    assert slide["microns_per_pixel"] == pytest.approx(0.2125)
    png = tmp_path / "a.png"
    mpimage.imsave(str(png), np.zeros((10, 12, 3)))
    geometry = oe._st_level_geometry(str(png), 2)
    assert geometry["level"] == 0 and geometry["shape"] == (10, 12)
    visium = tmp_path / "v"
    (visium / "spatial").mkdir(parents=True)
    tifffile.imwrite(str(visium / "full.tif"), np.zeros((8, 8, 3), np.uint8))
    with open(visium / "spatial" / "scalefactors_json.json", "w") as handle:
        json.dump({"spot_diameter_fullres": 110.0}, handle)
    assert oe._st_slide(str(visium))["microns_per_pixel"] == pytest.approx(0.5)


def test_stain_choices_give_the_image_the_segmenter_sees():
    colour = np.zeros((4, 4, 3), np.uint8)
    colour[:2] = (90, 50, 140)
    colour[2:] = (235, 200, 220)
    hematoxylin = oe._st_stain(colour, "auto", True)
    assert hematoxylin[:2].mean() > hematoxylin[2:].mean()
    gray = oe._st_stain(colour, "gray", True)
    assert gray[:2].mean() > gray[2:].mean()
    assert oe._st_stain(colour, "rgb", True).shape == (4, 4, 3)
    assert oe._st_stain(np.ones((4, 4, 2)), "auto", False).shape == (4, 4)
    assert oe._st_threshold_nuclei(np.zeros((20, 20)), 1.0, 6).max() == 0


def test_blank_windows_are_skipped_and_a_moved_bundle_still_maps(tmp_path):
    import shutil
    import tifffile

    source = xenium_slide(tmp_path / "bundle")
    focus = os.path.join(source, "morphology_focus",
                         "morphology_focus_0000.ome.tif")
    image = tifffile.imread(focus)
    image[:, 200:] = 100
    _pyramid(focus, image)
    dst = tmp_path / "fields"
    report = oe.run_spatial_masks(_mask_settings(source, dst))
    assert report["empty_windows"] >= 2
    kept = [c for c in _centres() if c[1] < 200 - RADIUS]
    assert report["objects"] == len(kept)
    moved = str(tmp_path / "moved")
    shutil.copytree(source, moved)
    summary = oe.run_spatial_map({"spatial_source": moved,
                                  "spatial_masks_dir": str(dst),
                                  "spatial_objects": "cell",
                                  "spatial_anndata": False})
    assert summary["objects"]["cell"]["objects"] == len(kept)
    assert "joined_measure_rows" not in summary


def test_a_segmenter_that_finds_nothing_writes_no_fields(tmp_path):
    source = xenium_slide(tmp_path / "bundle")
    report = oe.run_spatial_masks(
        _mask_settings(source, tmp_path / "fields"),
        segment=lambda image: np.zeros(image.shape, np.int32))
    assert report["objects"] == 0 and report["fields"] == 0
    assert report["empty_windows"] == report["windows"]


def test_measure_tables_are_keyed_like_io_joins_them(tmp_path):
    from spacr.tabular import write_database

    db = str(tmp_path / "m.db")
    assert oe._st_measured(db, "cell") is None
    write_database(pd.DataFrame({"prcf": ["p_r1_c1_f1"] * 2,
                                 "object_label": [1, 2], "cell_area": [5, 6]}),
                   db, "cell", if_exists="replace")
    write_database(pd.DataFrame({"x": [1]}), db, "nucleus",
                   if_exists="replace")
    measured = oe._st_measured(db, "cell")
    assert list(measured.index) == ["p_r1_c1_f1_o1", "p_r1_c1_f1_o2"]
    assert oe._st_measured(db, "nucleus") is None


def test_the_spatial_settings_register_once():
    from spacr import ops_settings
    from spacr.settings import categories, tooltips

    assert ops_settings.register_spatial() is False
    assert ops_settings.register_spatial(replace=True) is True
    for key in ops_settings.SPATIAL_TYPES:
        assert len(tooltips[key].split()) >= 6
    keys = set(ops_settings.spatial_mask_defaults()) | set(
        ops_settings.spatial_map_defaults())
    filed = {k for name in ("Spatial input", "Spatial region & tiling",
                            "Spatial segmentation", "Spatial mapping",
                            "Spatial infection") for k in categories[name]}
    assert keys == filed == set(ops_settings.SPATIAL_TYPES)


def test_a_gene_name_shared_by_two_features_gets_its_own_column(tmp_path):
    from scipy import sparse

    db = str(tmp_path / "m.db")
    oe._st_write_object_counts(
        db, "cell", ["p_r1_c1_f1_o1"], ["p_r1_c1_f1"], [1],
        np.array(["Pakap", "Pakap", "Actb"], dtype=object),
        sparse.csr_matrix(np.array([[1.0, 2.0, 3.0]])), "visium_hd")
    wide = read_table(db, table="cell_expression", canonicalise=False,
                      report=None)
    assert wide.loc[0, "expr_Pakap"] == 1 and wide.loc[0, "expr_Pakap_2"] == 2
