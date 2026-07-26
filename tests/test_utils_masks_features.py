"""CPU coverage for spacr.utils: cell-mask adjustment, clustering/embedding
helpers and the feature-analysis block.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import tifffile

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)


# ---------------------------------------------------------------------------
# synthetic mask helpers
# ---------------------------------------------------------------------------

def _two_cells(h=40, w=40):
    """Two adjacent cells; cell 1 has a nucleus, cell 2 does not."""
    cell = np.zeros((h, w), np.uint16)
    cell[5:20, 5:35] = 1
    cell[20:35, 5:35] = 2
    nuc = np.zeros((h, w), np.uint16)
    nuc[8:14, 10:16] = 1          # only inside cell 1
    return cell, nuc


def test_merge_cells_without_nucleus_relabels():
    from spacr.utils import _merge_cells_without_nucleus
    cell, nuc = _two_cells()
    out = _merge_cells_without_nucleus(cell.copy(), nuc)
    arr = out[0] if isinstance(out, tuple) else out
    # the nucleus-less cell must no longer keep its own distinct id
    labels = set(np.unique(arr)) - {0}
    assert labels, "all cells were erased"
    assert 2 not in labels or len(labels) < 2


def test_merge_cells_without_nucleus_all_have_nuclei():
    from spacr.utils import _merge_cells_without_nucleus
    cell, nuc = _two_cells()
    nuc[24:30, 10:16] = 2          # give cell 2 a nucleus too
    out = _merge_cells_without_nucleus(cell.copy(), nuc)
    arr = out[0] if isinstance(out, tuple) else out
    assert len(set(np.unique(arr)) - {0}) == 2


def test_merge_cells_based_on_parasite_overlap():
    from spacr.utils import _merge_cells_based_on_parasite_overlap
    cell, nuc = _two_cells()
    para = np.zeros_like(cell)
    para[18:23, 12:18] = 1          # straddles both cells
    org = np.zeros_like(cell)
    out = _merge_cells_based_on_parasite_overlap(
        para, cell.copy(), nuc, org, overlap_threshold=5, perimeter_threshold=30)
    assert out is not None
    arr = out[0] if isinstance(out, tuple) else out
    assert isinstance(arr, np.ndarray)


def test_process_mask_file_adjust_cell(tmp_path):
    from spacr.utils import process_mask_file_adjust_cell
    cell, nuc = _two_cells()
    para = np.zeros_like(cell); para[18:23, 12:18] = 1
    folders = {}
    for name, arr in (("parasite", para), ("cell", cell), ("nuclei", nuc)):
        d = tmp_path / name; d.mkdir()
        np.save(d / "f1.npy", arr)
        folders[name] = str(d)
    elapsed = process_mask_file_adjust_cell(
        "f1.npy", folders["parasite"], folders["cell"], folders["nuclei"])
    assert elapsed is None or elapsed >= 0
    # cell mask was rewritten in place
    assert np.load(os.path.join(folders["cell"], "f1.npy")).ndim == 2


def test_adjust_cell_masks_folder(tmp_path):
    from spacr.utils import adjust_cell_masks
    cell, nuc = _two_cells()
    para = np.zeros_like(cell); para[18:23, 12:18] = 1
    folders = {}
    for name, arr in (("parasite", para), ("cell", cell), ("nuclei", nuc)):
        d = tmp_path / name; d.mkdir()
        for i in range(2):
            np.save(d / f"f{i}.npy", arr)
        folders[name] = str(d)
    adjust_cell_masks(folders["parasite"], folders["cell"], folders["nuclei"],
                      organelle_folder=None, n_jobs=1)
    assert np.load(os.path.join(folders["cell"], "f0.npy")).ndim == 2


# ---------------------------------------------------------------------------
# clustering / embedding helpers
# ---------------------------------------------------------------------------

def _numeric(n=60, d=6, rng=None):
    rng = rng or np.random.default_rng(0)
    a = rng.normal(0, 1, (n // 2, d))
    b = rng.normal(5, 1, (n - n // 2, d))
    return np.vstack([a, b])


def test_reduction_and_clustering_umap_dbscan():
    pytest.importorskip("umap")
    from spacr.utils import reduction_and_clustering
    emb, labels, reducer = reduction_and_clustering(
        _numeric(), n_neighbors=5, min_dist=0.1, metric="euclidean",
        eps=0.9, min_samples=3, clustering="dbscan",
        reduction_method="umap", verbose=False, n_jobs=1)
    assert emb.shape[0] == 60 and emb.shape[1] == 2
    assert len(labels) == 60


def test_reduction_and_clustering_tsne_kmeans():
    from spacr.utils import reduction_and_clustering
    emb, labels, _ = reduction_and_clustering(
        _numeric(), n_neighbors=5, min_dist=0.1, metric="euclidean",
        eps=0.9, min_samples=3, clustering="kmeans",
        reduction_method="tsne", verbose=True, n_jobs=1)
    assert emb.shape[0] == 60
    assert len(set(labels)) >= 1


def test_reduction_and_clustering_rejects_unknown_method():
    """Only umap/tsne are implemented — anything else must raise clearly."""
    from spacr.utils import reduction_and_clustering
    with pytest.raises(ValueError, match="Unsupported reduction method"):
        reduction_and_clustering(
            _numeric(), n_neighbors=5, min_dist=0.1, metric="euclidean",
            eps=0.9, min_samples=3, clustering="kmeans",
            reduction_method="pca", verbose=False, n_jobs=1)


def test_reduction_method_tooltip_matches_implementation():
    """The settings tooltip must not advertise unsupported methods."""
    from spacr.settings import tooltips
    assert "pca" not in tooltips["reduction_method"].lower()


def test_search_reduction_and_clustering():
    pytest.importorskip("umap")
    from spacr.utils import search_reduction_and_clustering
    out = search_reduction_and_clustering(
        _numeric(), n_neighbors=5, min_dist=0.1, metric="euclidean",
        eps=0.9, min_samples=3, clustering="dbscan",
        reduction_method="umap", verbose=False,
        reduction_param={"n_neighbors": 5}, n_jobs=1)
    assert out is not None


def test_remove_noise_drops_minus_one():
    from spacr.utils import remove_noise
    emb = np.random.default_rng(0).normal(size=(10, 2))
    labels = np.array([-1, 0, 0, 1, 1, -1, 0, 1, 1, 0])
    e2, l2 = remove_noise(emb, labels)
    assert len(e2) == 8 and -1 not in set(l2)


def test_generate_colors():
    from spacr.utils import generate_colors
    c1 = generate_colors(4, black_background=True)
    c2 = generate_colors(4, black_background=False)
    assert len(c1) >= 4 and len(c2) >= 4


def test_check_overlap_and_non_overlapping_position():
    from spacr.utils import check_overlap, find_non_overlapping_position
    positions = [(0.0, 0.0), (1.0, 1.0)]
    assert check_overlap((0.0, 0.0), positions, threshold=0.5) is True
    assert check_overlap((10.0, 10.0), positions, threshold=0.5) is False
    x, y = find_non_overlapping_position(0.0, 0.0, positions, threshold=0.5)
    assert not check_overlap((x, y), positions, threshold=0.5)


# ---------------------------------------------------------------------------
# feature analysis
# ---------------------------------------------------------------------------

def _feature_df(n=40, rng=None):
    rng = rng or np.random.default_rng(1)
    df = pd.DataFrame({
        "cell_channel_0_mean_intensity": rng.normal(10, 2, n),
        "cell_channel_1_mean_intensity": rng.normal(20, 2, n),
        "cell_channel_2_mean_intensity": rng.normal(30, 2, n),
        "cell_area": rng.normal(500, 50, n),
        "cell_perimeter": rng.normal(90, 5, n),
        "constant_col": np.ones(n),                       # zero variance
        "cluster": [0] * (n // 2) + [1] * (n - n // 2),
    })
    df["dup_of_area"] = df["cell_area"] * 1.0             # perfectly correlated
    return df


def test_filter_dataframe_features_channel():
    from spacr.utils import filter_dataframe_features
    df = _feature_df()
    out = filter_dataframe_features(df.drop(columns=["cluster"]),
                                    channel_of_interest=1, verbose=True)
    frame = out[0] if isinstance(out, tuple) else out
    assert frame.shape[1] >= 1


def test_filter_dataframe_features_morphology():
    from spacr.utils import filter_dataframe_features
    df = _feature_df().drop(columns=["cluster"])
    out = filter_dataframe_features(df, channel_of_interest="morphology")
    frame = out[0] if isinstance(out, tuple) else out
    assert frame is not None


def test_filter_columns():
    from spacr.utils import filter_columns
    df = _feature_df().drop(columns=["cluster"])
    out = filter_columns(df, filter_by="channel_0")
    assert out.shape[1] >= 1


def test_check_normality():
    from spacr.utils import check_normality
    rng = np.random.default_rng(0)
    assert check_normality(pd.Series(rng.normal(size=60))) in (True, False)


def test_random_forest_feature_importance():
    from spacr.utils import random_forest_feature_importance
    out = random_forest_feature_importance(_feature_df(), cluster_col="cluster")
    assert isinstance(out, pd.DataFrame) and len(out) > 0


def test_perform_statistical_tests():
    from spacr.utils import perform_statistical_tests
    out = perform_statistical_tests(_feature_df(), cluster_col="cluster")
    assert out is not None


def test_cluster_feature_analysis():
    from spacr.utils import cluster_feature_analysis
    out = cluster_feature_analysis(_feature_df(), cluster_col="cluster")
    assert isinstance(out, pd.DataFrame)


# ---------------------------------------------------------------------------
# small path / db helpers
# ---------------------------------------------------------------------------

def test_get_db_paths_and_sequencing_paths(tmp_path):
    from spacr.utils import get_db_paths, get_sequencing_paths
    meas = tmp_path / "measurements"; meas.mkdir()
    (meas / "measurements.db").write_bytes(b"")
    dbs = get_db_paths([str(tmp_path)])
    assert dbs and dbs[0].endswith("measurements.db")
    seq = tmp_path / "sequencing"; seq.mkdir()
    (seq / "sequencing.db").write_bytes(b"")
    out = get_sequencing_paths([str(tmp_path)])
    assert isinstance(out, list)


def test_merge_dataframes():
    from spacr.utils import merge_dataframes
    df = pd.DataFrame({"prcfo": ["a", "b"], "v": [1, 2]})
    ip = pd.DataFrame({"prcfo": ["a", "b"], "png_path": ["/x/a.png", "/x/b.png"]})
    out = merge_dataframes(df, ip, verbose=True)
    assert "png_path" in out.columns


def test_process_vision_results():
    """`path` holds a bare tar member name in spacr's crop-PNG convention
    ``<plate>_<well>_<field>_<object>.png`` (see _map_wells / _map_wells_png).

    The old fixture used ``/x/plate1_A01_f1_o1.png``: the directory prefix
    leaked into plateID and the ``f1`` field could not be int-parsed, so
    fieldID silently became ``f0``. The swallowed skip hid the whole thing.
    """
    from spacr.utils import process_vision_results
    df = pd.DataFrame({
        "path": ["plate1_A01_1_1.png", "plate1_A01_1_2.png"],
        "pred": [0.2, 0.8],
    })
    out = process_vision_results(df, threshold=0.5)
    assert {"plateID", "rowID", "columnID", "fieldID", "prc",
            "cv_predictions"} <= set(out.columns)
    assert out["plateID"].tolist() == ["plate1", "plate1"]
    assert out["rowID"].tolist() == ["r1", "r1"]
    assert out["columnID"].tolist() == ["c1", "c1"]
    assert out["fieldID"].tolist() == ["f1", "f1"]
    assert out["prc"].tolist() == ["plate1_r1_c1"] * 2
    assert out["object"].tolist() == ["1", "2"]
    # pred 0.2 < 0.5 <= 0.8
    assert out["cv_predictions"].tolist() == [0, 1]
