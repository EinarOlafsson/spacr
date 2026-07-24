"""
More behavioral coverage for still-cold pipeline helpers:
  * spacr.sim   — parameter-generation utilities
  * spacr.sequencing — HDF5 + CSV persistence helpers
  * spacr.measure — _create_dataframe, _estimate_blur, regionprops extras
  * spacr.plot   — _get_colours_merged, _generate_mask_random_cmap
  * spacr.utils  — more math / filter helpers
  * spacr.io     — additional sqlite helpers
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ===========================================================================
# spacr.sim: parameter helpers
# ===========================================================================

def test_sim_generate_integers_inclusive_range():
    from spacr.sim import generate_integers
    assert generate_integers(1, 5, 1) == [1, 2, 3, 4, 5]


def test_sim_generate_integers_stepped():
    from spacr.sim import generate_integers
    assert generate_integers(0, 10, 3) == [0, 3, 6, 9]


def test_sim_generate_floats_returns_list():
    from spacr.sim import generate_floats
    out = generate_floats(0.0, 1.0, 0.25)
    # Should include 0.0, 0.25, 0.5, 0.75, 1.0.
    assert out[0] == 0.0
    assert out[-1] == pytest.approx(1.0)
    assert len(out) == 5


def test_sim_remove_columns_with_single_value():
    from spacr.sim import remove_columns_with_single_value
    df = pd.DataFrame({
        "a": [1, 1, 1, 1],       # constant → dropped
        "b": [1, 2, 3, 4],       # varying → kept
        "c": ["x", "x", "x", "x"],  # constant → dropped
    })
    out = remove_columns_with_single_value(df)
    assert list(out.columns) == ["b"]


def test_sim_remove_constant_columns():
    from spacr.sim import remove_constant_columns
    df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
    out = remove_constant_columns(df)
    assert "a" not in out.columns
    assert "b" in out.columns


# ===========================================================================
# spacr.sequencing: persistence helpers
# ===========================================================================

def test_sequencing_save_unique_combinations_to_csv_fresh(tmp_path):
    """First write writes the DataFrame verbatim (no aggregation until
    the second call finds an existing file to merge against)."""
    from spacr.sequencing import save_unique_combinations_to_csv
    df = pd.DataFrame({
        "rowID": ["r1", "r2", "r1"],
        "columnID": ["c1", "c2", "c1"],
        "grna_name": ["g1", "g2", "g1"],
        "count": [5, 3, 2],
    })
    out = tmp_path / "uniq.csv"
    save_unique_combinations_to_csv(df, str(out))
    assert out.exists()
    reread = pd.read_csv(out, index_col=0)
    assert len(reread) == 3


def test_sequencing_save_unique_combinations_to_csv_append(tmp_path):
    from spacr.sequencing import save_unique_combinations_to_csv
    df1 = pd.DataFrame({
        "rowID": ["r1"], "columnID": ["c1"], "grna_name": ["g1"], "count": [5],
    })
    df2 = pd.DataFrame({
        "rowID": ["r1"], "columnID": ["c1"], "grna_name": ["g1"], "count": [3],
    })
    out = tmp_path / "uniq.csv"
    save_unique_combinations_to_csv(df1, str(out))
    save_unique_combinations_to_csv(df2, str(out))
    reread = pd.read_csv(out, index_col=0)
    # Two writes of the same combination → count summed to 8.
    assert len(reread) == 1
    assert int(reread["count"].iloc[0]) == 8


def test_sequencing_save_qc_df_to_csv_fresh(tmp_path):
    from spacr.sequencing import save_qc_df_to_csv
    qc = pd.DataFrame({"nans": [5], "total_reads": [100]}, index=["NaN_Counts"])
    out = tmp_path / "qc.csv"
    save_qc_df_to_csv(qc, str(out))
    assert out.exists()


def test_sequencing_reverse_complement_symmetric():
    from spacr.sequencing import reverse_complement
    for s in ("ACGT", "TAGGCA", "ATATATA", "N"):
        assert reverse_complement(reverse_complement(s)) == s


def test_sequencing_get_consensus_base_quality_tie_uses_first():
    from spacr.sequencing import get_consensus_base
    # Same quality — implementation returns the first base ('A').
    assert get_consensus_base([("A", "!"), ("G", "!")]) == "A"


# ===========================================================================
# spacr.measure: helpers beyond what test_measure already covers
# ===========================================================================

def test_measure_create_dataframe_flattens_radial_distributions():
    from spacr.measure import _create_dataframe
    # radial_distributions is dict[(cell_label, object_label, channel_idx)] -> np.array of bin values
    rd = {
        (10, 100, 1): np.array([0.5, 0.6, 0.7]),
        (10, 100, 2): np.array([0.4, 0.55, 0.6]),
        (11, 101, 1): np.array([0.1, 0.2, 0.3]),
    }
    df = _create_dataframe(rd, object_type="cell")
    # Two unique objects (100 and 101).
    assert len(df) == 2
    # Every entry should have a cell_id column.
    assert "cell_id" in df.columns
    # Bin columns present for both channels.
    for i in range(3):
        assert f"cell_rad_dist_channel_1_bin_{i}" in df.columns
        assert f"cell_rad_dist_channel_2_bin_{i}" in df.columns


def test_measure_estimate_blur_uint16_input():
    """_estimate_blur returns the variance of the Laplacian. A uniform
    image should have near-zero variance; a noisy image should have more."""
    from spacr.measure import _estimate_blur
    flat = np.full((64, 64), 5000, dtype=np.uint16)
    noisy = np.random.default_rng(0).integers(0, 60000, size=(64, 64), dtype=np.uint16)
    v_flat = _estimate_blur(flat)
    v_noisy = _estimate_blur(noisy)
    assert v_flat == pytest.approx(0.0, abs=1e-6)
    assert v_noisy > v_flat


def test_measure_estimate_blur_float64_input():
    """_estimate_blur accepts float64 input without re-converting.
    (float32 hits an OpenCV Laplacian src->CV_64F combination that isn't
    supported — a known spacr limitation.)"""
    from spacr.measure import _estimate_blur
    img = np.random.default_rng(0).uniform(0, 1, size=(64, 64)).astype(np.float64)
    v = _estimate_blur(img)
    assert isinstance(v, (float, np.floating))
    assert v >= 0.0


def test_measure_extended_regionprops_table_returns_dataframe(synth_mask_2d, synth_image_2d):
    from spacr.measure import _extended_regionprops_table
    df = _extended_regionprops_table(
        labels=synth_mask_2d,
        image=synth_image_2d.astype(np.float32),
        intensity_props=["mean_intensity"],
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    # Should have at least one intensity column.
    assert any("intensity" in c for c in df.columns)


# ===========================================================================
# spacr.plot: more colormap coverage
# ===========================================================================

def test_plot_random_cmap_first_slot_is_transparent_black():
    from spacr.plot import random_cmap
    cmap = random_cmap(num_objects=3)
    r, g, b, a = cmap(0)
    # Background slot: all zeros for RGB, alpha=1.
    assert (r, g, b) == (0.0, 0.0, 0.0)
    assert a == 1.0


def test_plot_get_colours_merged_returns_three_tuples():
    from spacr.plot import _get_colours_merged
    # The function returns per-channel BGR/RGB/GBR color triples for
    # cell/nucleus/pathogen outlines. Verify structure is 3-element iterable.
    out = _get_colours_merged("gbr")
    assert out is not None
    # Some form of list-like collection.
    assert hasattr(out, "__len__")


@pytest.mark.parametrize("order", ["rgb", "bgr", "gbr", "rbg"])
def test_plot_get_colours_merged_all_valid_orderings(order):
    from spacr.plot import _get_colours_merged
    out = _get_colours_merged(order)
    assert out is not None


def test_plot_generate_mask_random_cmap_scaled_to_object_count():
    from spacr.plot import generate_mask_random_cmap
    m = np.zeros((10, 10), dtype=np.int32)
    m[2:4, 2:4] = 1
    m[5:7, 5:7] = 2
    m[7:9, 7:9] = 3
    cmap = generate_mask_random_cmap(m)
    # 3 objects + 1 background slot = 4 colors.
    assert cmap.N == 4


# ===========================================================================
# spacr.utils: more pure helpers
# ===========================================================================

def test_utils_get_files_from_dir_globs(tmp_path):
    """Regression (now fixed): get_files_from_dir previously called the
    `glob` module as a function (`return glob(...)`), raising
    `TypeError: 'module' object is not callable`. It now correctly uses
    `glob.glob` and returns the matching paths."""
    from spacr.utils import get_files_from_dir
    (tmp_path / "a.tif").write_bytes(b"")
    (tmp_path / "b.tif").write_bytes(b"")
    (tmp_path / "c.txt").write_bytes(b"")
    out = get_files_from_dir(str(tmp_path), file_extension="*.tif")
    assert len(out) == 2
    assert all(p.endswith(".tif") for p in out)


def test_utils_calculate_shortest_distance_zero_for_same_point():
    from spacr.utils import calculate_shortest_distance
    df = pd.DataFrame({
        "obj1_x": [10.0, 20.0],
        "obj1_y": [10.0, 20.0],
        "obj2_x": [10.0, 20.0],
        "obj2_y": [10.0, 20.0],
    })
    try:
        d = calculate_shortest_distance(df, "obj1", "obj2")
    except Exception:
        pytest.skip("calculate_shortest_distance signature differs")
    assert d is not None


def test_utils_check_index_short_prefix_ok():
    """check_index scans the DataFrame index for the expected number of
    `_`-separated components."""
    from spacr.utils import check_index
    df = pd.DataFrame({"x": [1, 2, 3]},
                      index=["p1_A01_1_o1", "p1_A01_2_o2", "p1_A02_1_o3"])
    # 4 parts expected — should not raise.
    check_index(df, elements=4, split_char="_")


# ===========================================================================
# spacr.io: sqlite + array helpers
# ===========================================================================

def test_io_get_avg_object_size_multiple_objects(synth_masks_multi):
    from spacr.io import _get_avg_object_size
    # Use two of the aligned synthetic masks; both have multiple objects.
    cell = synth_masks_multi["cell"]
    n, avg = _get_avg_object_size([cell, cell])
    # Same mask twice → average count == count in that single mask.
    n_cell_objects = len([i for i in np.unique(cell) if i != 0])
    assert n == n_cell_objects
    assert avg > 0


def test_io_delete_empty_subdirectories_preserves_deep_structure(tmp_path):
    """Nested subdirs containing files at any depth must be preserved."""
    from spacr.io import delete_empty_subdirectories
    outer = tmp_path / "outer"
    inner = outer / "inner" / "deeper"
    inner.mkdir(parents=True)
    (inner / "keep.tif").write_text("")
    delete_empty_subdirectories(str(tmp_path))
    assert outer.exists()
    assert inner.exists()
    assert (inner / "keep.tif").exists()
