"""Visium and Xenium reads registered to an image and assigned to objects.

The fixtures are written in the platforms' own file formats (a 10x
feature-barcode ``.h5``, ``tissue_positions.csv``, ``scalefactors_json.json``,
``transcripts.parquet``, ``experiment.xenium`` and a morphology OME-TIFF),
small enough that every number asserted can be worked out by hand.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from spacr import ops_engine as oe
from spacr.tabular import read_table

pytest.importorskip("pyarrow")

PIXEL = 0.5
GRID = (4, 5)
PITCH = 40
INFECTED = {(r, c) for r in range(GRID[0]) for c in range(GRID[1])
            if (r + c) % 2 == 0}


def _disk(mask, centre, radius, label):
    rows, cols = np.ogrid[:mask.shape[0], :mask.shape[1]]
    inside = (rows - centre[0]) ** 2 + (cols - centre[1]) ** 2 <= radius ** 2
    mask[inside] = label


def _masks():
    """Cell, nucleus and pathogen masks on a 160x200 px frame."""
    shape = (GRID[0] * PITCH, GRID[1] * PITCH)
    cell = np.zeros(shape, np.int32)
    nucleus = np.zeros(shape, np.int32)
    pathogen = np.zeros(shape, np.int32)
    parasite = 0
    for r in range(GRID[0]):
        for c in range(GRID[1]):
            label = r * GRID[1] + c + 1
            centre = (PITCH // 2 + r * PITCH, PITCH // 2 + c * PITCH)
            _disk(cell, centre, 15, label)
            _disk(nucleus, centre, 6, label)
            if (r, c) in INFECTED:
                parasite += 1
                _disk(pathogen, (centre[0], centre[1] + 9), 3, parasite)
    return cell, nucleus, pathogen


def xenium_bundle(root):
    """A Xenium bundle whose GENE_UP is made in infected cells only."""
    import tifffile

    root = str(root)
    os.makedirs(os.path.join(root, "morphology_focus"), exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for r in range(GRID[0]):
        for c in range(GRID[1]):
            centre = np.array([PITCH // 2 + c * PITCH, PITCH // 2 + r * PITCH])
            plan = {"GENE_FLAT": 10,
                    "GENE_UP": 25 if (r, c) in INFECTED else 1}
            for gene, count in plan.items():
                for _ in range(count):
                    x, y = centre + rng.uniform(-4, 4, 2)
                    rows.append((gene, x * PIXEL, y * PIXEL, 30.0))
    rows += [("NegControlProbe_A", 20 * PIXEL, 20 * PIXEL, 30.0),
             ("GENE_FLAT", 20 * PIXEL, 20 * PIXEL, 5.0)]
    frame = pd.DataFrame(rows, columns=["feature_name", "x_location",
                                        "y_location", "qv"])
    frame["z_location"] = 1.0
    frame["cell_id"] = "UNASSIGNED"
    frame.to_parquet(os.path.join(root, "transcripts.parquet"))
    with open(os.path.join(root, "experiment.xenium"), "w") as handle:
        json.dump({"pixel_size": PIXEL}, handle)
    cell, _nucleus, _pathogen = _masks()
    tifffile.imwrite(
        os.path.join(root, "morphology_focus", "morphology_focus_0000.ome.tif"),
        (cell > 0).astype(np.uint16) * 1000)
    return root


def visium_bundle(root):
    """A Visium outs folder with four spots on a 200x200 px hires image."""
    h5py = pytest.importorskip("h5py")
    from matplotlib import image as mpimage
    from scipy import sparse

    root = str(root)
    spatial = os.path.join(root, "spatial")
    os.makedirs(spatial, exist_ok=True)
    barcodes = ["AAA-1", "CCC-1", "GGG-1", "TTT-1"]
    genes = ["GeneA", "GeneB", "CD3_TotalSeq"]
    dense = np.array([[10, 0, 5], [4, 2, 5], [0, 8, 5], [6, 6, 5]])
    csc = sparse.csc_matrix(dense.T)
    with h5py.File(os.path.join(root, "filtered_feature_bc_matrix.h5"),
                   "w") as handle:
        group = handle.create_group("matrix")
        group["barcodes"] = np.array(barcodes, dtype="S")
        group["data"] = csc.data
        group["indices"] = csc.indices
        group["indptr"] = csc.indptr
        group["shape"] = np.array(csc.shape)
        features = group.create_group("features")
        features["name"] = np.array(genes, dtype="S")
        features["id"] = np.array(genes, dtype="S")
        features["feature_type"] = np.array(
            ["Gene Expression", "Gene Expression", "Antibody Capture"],
            dtype="S")
    pd.DataFrame({
        "barcode": barcodes[::-1], "in_tissue": 1,
        "array_row": [0, 0, 1, 1], "array_col": [0, 1, 0, 1],
        "pxl_row_in_fullres": [300, 300, 100, 100],
        "pxl_col_in_fullres": [300, 100, 300, 100],
    }).to_csv(os.path.join(spatial, "tissue_positions.csv"), index=False)
    with open(os.path.join(spatial, "scalefactors_json.json"), "w") as handle:
        json.dump({"tissue_hires_scalef": 0.5, "tissue_lowres_scalef": 0.1,
                   "spot_diameter_fullres": 40.0}, handle)
    mpimage.imsave(os.path.join(spatial, "tissue_hires_image.png"),
                   np.full((200, 200, 3), 0.8))
    return root


def test_the_xenium_reader_keeps_gene_reads_of_enough_quality(tmp_path):
    bundle = oe._st_read_bundle(xenium_bundle(tmp_path / "x"))
    assert bundle["platform"] == "xenium"
    assert list(bundle["genes"]) == ["GENE_FLAT", "GENE_UP"]
    assert bundle["dropped"] == 2
    assert bundle["pixel_size"] == PIXEL


def test_the_visium_reader_orders_positions_by_the_matrix(tmp_path):
    bundle = oe._st_read_bundle(visium_bundle(tmp_path / "v"))
    assert bundle["platform"] == "visium"
    assert list(bundle["genes"]) == ["GeneA", "GeneB"]
    assert bundle["counts"].toarray().tolist() == [[10, 0], [4, 2], [0, 8],
                                                   [6, 6]]
    assert bundle["obs"]["x_full"].tolist() == [100, 300, 100, 300]
    xy, radius, microns = oe._st_platform_xy(bundle, "hires")
    assert xy[0].tolist() == [50.0, 50.0] and radius == 10.0
    assert microns == pytest.approx(55.0 / 40.0 / 0.5)


def test_xenium_transcripts_land_in_the_cell_they_were_placed_in(tmp_path):
    bundle = oe._st_read_bundle(xenium_bundle(tmp_path / "x"))
    registered = oe._st_register(bundle, {})
    cell, _nucleus, _pathogen = _masks()
    labels, matrix, _ = oe._st_object_counts(bundle, registered["xy"], cell)
    assert len(labels) == GRID[0] * GRID[1]
    up = list(bundle["genes"]).index("GENE_UP")
    first = matrix[list(labels).index(1)].toarray().ravel()
    assert first[up] == 25 and first[1 - up] == 10


def test_a_spot_half_on_an_object_gives_it_half_its_counts():
    bundle = {"platform": "visium", "genes": np.array(["g"]),
              "counts": __import__("scipy").sparse.csr_matrix([[8.0]]),
              "spot_shape": "circle"}
    mask = np.zeros((41, 41), np.int64)
    mask[:, :20] = 7
    labels, matrix, coverage = oe._st_object_counts(
        bundle, np.array([[20.0, 20.0]]), mask, radius=10.0)
    assert labels.tolist() == [7]
    assert coverage["fraction"].iloc[0] == pytest.approx(0.475, abs=0.03)
    assert matrix.toarray()[0, 0] == pytest.approx(8 * coverage["fraction"].iloc[0])


def test_affine_from_landmarks_and_from_image_content():
    from skimage.transform import AffineTransform, warp

    true = AffineTransform(scale=(1.2, 1.2), rotation=0.1,
                           translation=(15, -5))
    source = np.array([[10, 10], [150, 20], [60, 140], [120, 120]], float)
    matrix, rms = oe._st_fit_affine(source, true(source))
    assert rms < 1e-6 and np.allclose(matrix, true.params)
    rng = np.random.default_rng(3)
    moving = np.zeros((200, 200))
    for _ in range(40):
        r, c = rng.integers(10, 190, 2)
        moving[r - 3:r + 4, c - 3:c + 4] = rng.uniform(0.3, 1.0)
    shifted = np.roll(np.roll(moving, 12, axis=0), -7, axis=1)
    found, info = oe._st_register_intensity(moving, shifted)
    point = oe._st_apply_affine(found, np.array([[100.0, 100.0]]))[0]
    assert np.allclose(point, [93, 112], atol=1.0), info


def test_the_whole_run_writes_counts_and_finds_the_infection_gene(tmp_path):
    folder = xenium_bundle(tmp_path / "x")
    cell, nucleus, pathogen = _masks()
    paths = {}
    for kind, mask in (("cell", cell), ("nucleus", nucleus),
                       ("pathogen", pathogen)):
        paths[kind] = str(tmp_path / f"plate1_r1_c1_f1_{kind}.npy")
        np.save(paths[kind], mask)
    db = str(tmp_path / "measurements" / "measurements.db")
    request = {"folder": folder, "masks": paths, "db": db,
               "prcf": "plate1_r1_c1_f1", "gene": "GENE_UP",
               "anndata": False}
    summary = oe._st_run(request)
    assert summary["objects"]["cell"]["objects"] == 20
    assert summary["objects"]["cell"]["assigned_fraction"] == 1.0
    wide = read_table(db, table="cell_expression", report=None)
    assert wide["prcfo"].iloc[0].startswith("plate1_r1_c1_f1_o")
    assert set(wide["expr_total"]) >= {11.0, 35.0}
    compared = read_table(summary["files"]["infected_vs_uninfected"])
    up = compared.set_index("gene").loc["GENE_UP"]
    assert up["q_value"] < 0.01 and up["log2_fold_change"] > 3
    assert summary["infected_vs_uninfected"]["n"] == {"infected": 10,
                                                      "uninfected": 10}
    trend = read_table(summary["files"]["distance_trend"])
    assert trend.set_index("gene").loc["GENE_UP", "spearman_rho"] < -0.5
    regions = read_table(summary["files"]["region_summary"])
    assert set(regions["region"]) == {"infected", "uninfected"}
    assert os.path.exists(summary["files"]["overlay"])
    oe._st_run(request)
    again = read_table(db, table="cell_expression_long", report=None)
    assert again.groupby(["prcfo", "gene"]).size().max() == 1


def test_visium_spots_are_shared_out_and_recorded(tmp_path):
    folder = visium_bundle(tmp_path / "v")
    mask = np.zeros((200, 200), np.int32)
    mask[40:60, 40:50] = 1
    mask[140:160, 140:160] = 2
    np.save(tmp_path / "img.npy", mask)
    db = str(tmp_path / "m.db")
    summary = oe._st_run({"folder": folder, "masks": {
        "cell": str(tmp_path / "img.npy")}, "db": db, "anndata": False})
    coverage = read_table(db, table="cell_spot_coverage", report=None)
    assert sorted(coverage["barcode"]) == ["AAA-1", "TTT-1"]
    assert summary["objects"]["cell"]["spots_covering"] == 2
    wide = read_table(db, table="cell_expression", report=None).set_index(
        "object_label")
    share = coverage.set_index("object_label").loc[1, "fraction"]
    assert wide.loc[1, "expr_GeneA"] == pytest.approx(10 * share)


def test_a_mask_of_another_frame_is_refused(tmp_path):
    folder = xenium_bundle(tmp_path / "x")
    np.save(tmp_path / "small.npy", np.ones((10, 10), np.int32))
    with pytest.raises(ValueError, match="registered to"):
        oe._st_run({"folder": folder, "db": str(tmp_path / "m.db"),
                    "masks": {"cell": str(tmp_path / "small.npy")}})


def test_anndata_export_carries_positions_and_measurements(tmp_path):
    anndata = pytest.importorskip("anndata")
    folder = xenium_bundle(tmp_path / "x")
    cell, _nucleus, pathogen = _masks()
    np.save(tmp_path / "c.npy", cell)
    np.save(tmp_path / "p.npy", pathogen)
    summary = oe._st_run({"folder": folder, "db": str(tmp_path / "m.db"),
                          "masks": {"cell": str(tmp_path / "c.npy"),
                                    "pathogen": str(tmp_path / "p.npy")}})
    adata = anndata.read_h5ad(summary["files"]["anndata"])
    assert adata.shape == (20, 2)
    assert adata.obsm["spatial"].shape == (20, 2)
    assert adata.obs["infected"].sum() == 10
