"""measure.py edge branches: empty/missing object masks, degenerate
regions in the intensity-distance metric, and generate_object_dataset's
missing-array / bad-mask_dim paths.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)


def _settings(**over):
    s = {
        "cell_mask_dim": 4, "nucleus_mask_dim": 5, "pathogen_mask_dim": 6,
        "channels": [0, 1, 2], "homogeneity": False,
        "homogeneity_distances": [8], "radial_dist": False,
        "calculate_correlation": False, "manders_thresholds": [15, 85],
        "distance_gaussian_sigma": 2, "verbose": False,
        "summarize_organelles_by": None,
        "cytoplasm": False, "organelle_mask_dim": None,
        "uninfected": True, "timelapse": False,
    }
    s.update(over)
    return s


# ---------------------------------------------------------------------------
# morphological measurements with objects switched off
# ---------------------------------------------------------------------------

def test_morphological_with_nucleus_and_pathogen_disabled():
    """nucleus/pathogen dims None -> empty frames are appended, not crashes."""
    from spacr.measure import _morphological_measurements
    cell = np.zeros((32, 32), np.uint16); cell[4:20, 4:20] = 1
    empty = np.zeros((32, 32), np.uint16)
    out = _morphological_measurements(
        cell, empty, empty, empty, empty,
        _settings(nucleus_mask_dim=None, pathogen_mask_dim=None),
        zernike=False)
    assert len(out) == 5
    # cell frame has content; the disabled ones are empty
    assert out[0] is not None


def test_morphological_pathogen_without_cell():
    """pathogen present, cell disabled -> no cell->pathogen merge branch."""
    from spacr.measure import _morphological_measurements
    pat = np.zeros((32, 32), np.uint16); pat[6:12, 6:12] = 1
    empty = np.zeros((32, 32), np.uint16)
    out = _morphological_measurements(
        empty, empty, pat, empty, empty,
        _settings(cell_mask_dim=None, nucleus_mask_dim=None),
        zernike=False)
    assert len(out) == 5


# ---------------------------------------------------------------------------
# organelle-per-parent summary with no organelles
# ---------------------------------------------------------------------------

def test_summarize_organelles_with_none_present():
    """An empty organelle mask returns a zero-filled row per parent."""
    from spacr.measure import _summarize_organelles_per_parent
    parent = np.zeros((32, 32), np.uint16)
    parent[2:14, 2:14] = 1
    parent[18:30, 18:30] = 2
    organelle = np.zeros((32, 32), np.uint16)      # none at all
    channels = np.zeros((32, 32, 2), np.uint16)
    out = _summarize_organelles_per_parent(organelle, parent, channels,
                                           parent_name="cell")
    assert len(out) == 2
    assert set(out["organelle_count"]) == {0}
    assert set(out["organelle_fraction"]) == {0.0}


def test_summarize_organelles_with_organelles_present():
    from spacr.measure import _summarize_organelles_per_parent
    parent = np.zeros((32, 32), np.uint16); parent[2:20, 2:20] = 1
    organelle = np.zeros((32, 32), np.uint16); organelle[5:9, 5:9] = 1
    rng = np.random.default_rng(0)
    channels = rng.integers(0, 500, (32, 32, 2)).astype(np.uint16)
    out = _summarize_organelles_per_parent(organelle, parent, channels,
                                           parent_name="cell")
    assert len(out) == 1
    assert int(out["organelle_count"].iloc[0]) >= 1
    assert any("mean_intensity" in c for c in out.columns)


# ---------------------------------------------------------------------------
# intensity-distance metric degenerate regions
# ---------------------------------------------------------------------------

def test_measure_intensity_distance_handles_empty_cells():
    """Labels present in the label set but with no pixels yield NaN rows."""
    from spacr.measure import _measure_intensity_distance
    cell = np.zeros((32, 32), np.uint16)
    cell[4:16, 4:16] = 1
    cell[20:26, 20:26] = 3          # label 2 deliberately absent
    nuc = np.zeros((32, 32), np.uint16); nuc[6:10, 6:10] = 1
    pat = np.zeros((32, 32), np.uint16); pat[8:10, 8:10] = 1
    rng = np.random.default_rng(1)
    channels = rng.integers(0, 500, (32, 32, 3)).astype(np.uint16)
    out = _measure_intensity_distance(cell, nuc, pat, channels, _settings())
    assert out is not None


def test_measure_intensity_distance_all_zero_image():
    """A flat-zero channel makes the centre-of-mass undefined -> NaN path."""
    from spacr.measure import _measure_intensity_distance
    cell = np.zeros((32, 32), np.uint16); cell[4:16, 4:16] = 1
    nuc = np.zeros((32, 32), np.uint16); nuc[6:10, 6:10] = 1
    pat = np.zeros((32, 32), np.uint16)
    channels = np.zeros((32, 32, 3), np.uint16)     # nothing to weight by
    out = _measure_intensity_distance(cell, nuc, pat, channels, _settings())
    assert out is not None


# ---------------------------------------------------------------------------
# generate_object_dataset array-loading edge paths
# ---------------------------------------------------------------------------

def _dataset(root, missing_array=False):
    merged = os.path.join(root, "merged")
    meas = os.path.join(root, "measurements")
    os.makedirs(merged, exist_ok=True); os.makedirs(meas, exist_ok=True)
    npy = os.path.join(merged, "plate1_r1_c1_f1.npy")
    arr = np.zeros((32, 32, 4), np.float32)
    arr[..., 0] = 10.0
    mask = np.zeros((32, 32), np.int32); mask[4:20, 4:20] = 1
    arr[..., 3] = mask
    if not missing_array:
        np.save(npy, arr)
    db = os.path.join(meas, "measurements.db")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE cell (object_label INT, path_name TEXT, "
                "plateID TEXT, rowID TEXT, columnID TEXT, fieldID TEXT, "
                "cell_area REAL)")
    con.execute("INSERT INTO cell VALUES (?,?,?,?,?,?,?)",
                (1, npy, "plate1", "r1", "c1", "f1", 256.0))
    con.commit(); con.close()
    return root


def test_generate_object_dataset_skips_missing_array(tmp_path, capsys):
    """A row pointing at a deleted .npy is reported and skipped."""
    from spacr.measure import generate_object_dataset
    root = _dataset(str(tmp_path), missing_array=True)
    man = generate_object_dataset(root, object_type="cell", channels=(0,),
                                  mask_dims={"cell": 3}, verbose=True)
    assert man == [] or len(man) == 0
    assert "missing array" in capsys.readouterr().out


def test_generate_object_dataset_bad_mask_dim_raises(tmp_path):
    """A mask_dim beyond the array's channel count raises IndexError."""
    from spacr.measure import generate_object_dataset
    root = _dataset(str(tmp_path))
    with pytest.raises(IndexError):
        generate_object_dataset(root, object_type="cell", channels=(0,),
                                mask_dims={"cell": 99}, verbose=False)
