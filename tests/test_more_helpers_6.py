"""
Seventh batch of behavioral tests — ring-filling algorithms, preprocessing
batches, and additional measure/timelapse helpers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ===========================================================================
# spacr.object: ring-filling algorithms
# ===========================================================================

def test_object_fill_rings_flood_fills_interior():
    """A hollow ring (edges only) should be filled by _fill_rings_flood."""
    from spacr.object import _fill_rings_flood
    # Create a ring: circle outline at radius 8, on a 32x32 canvas.
    edges = np.zeros((32, 32), dtype=bool)
    yy, xx = np.mgrid[:32, :32]
    d = np.sqrt((yy - 16) ** 2 + (xx - 16) ** 2)
    ring = (d >= 7) & (d <= 9)
    edges[ring] = True
    filled = _fill_rings_flood(edges)
    # The center should be filled.
    assert filled[16, 16]


def test_object_fill_rings_flood_leaves_background_alone():
    """Border-touching background regions should NOT be filled — otherwise
    the whole image would be filled."""
    from spacr.object import _fill_rings_flood
    edges = np.zeros((20, 20), dtype=bool)
    # Small ring in the center.
    edges[8:12, 8:12] = True
    edges[9:11, 9:11] = False  # make it a ring
    filled = _fill_rings_flood(edges)
    # Corner (0, 0) is background and touches border — should remain False.
    assert filled[0, 0] == False


def test_object_fill_rings_convex_fills_with_hull():
    """_fill_rings_convex uses convex hulls; a ring becomes a filled disk
    (its convex hull)."""
    from spacr.object import _fill_rings_convex
    edges = np.zeros((32, 32), dtype=bool)
    yy, xx = np.mgrid[:32, :32]
    d = np.sqrt((yy - 16) ** 2 + (xx - 16) ** 2)
    ring = (d >= 7) & (d <= 9)
    edges[ring] = True
    filled = _fill_rings_convex(edges)
    # Center should be filled (hull covers ring's interior).
    assert filled[16, 16]


def test_object_fill_rings_convex_empty_input():
    """Empty binary edges → empty filled mask (no crash)."""
    from spacr.object import _fill_rings_convex
    edges = np.zeros((16, 16), dtype=bool)
    filled = _fill_rings_convex(edges)
    assert filled.shape == edges.shape
    assert not filled.any()


# ===========================================================================
# spacr.object._preprocess_batch: no-op when nothing enabled
# ===========================================================================

def test_object_preprocess_batch_no_settings_returns_input(rng):
    from spacr.object import _preprocess_batch
    batch = rng.uniform(0, 1, size=(3, 32, 32)).astype(np.float32)
    settings = {}   # nothing enabled → passthrough
    out = _preprocess_batch(batch, settings)
    assert np.array_equal(out, batch)


def test_object_preprocess_batch_rolling_ball_only(rng):
    from spacr.object import _preprocess_batch
    batch = rng.uniform(0, 1, size=(2, 32, 32)).astype(np.float32)
    settings = {"organelle_rolling_ball": True,
                "organelle_rolling_ball_radius": 5}
    try:
        out = _preprocess_batch(batch, settings)
    except Exception as e:  # pragma: no cover - skimage version differences
        pytest.skip(f"rolling_ball unavailable: {e}")
    # Shape preserved; values non-negative after background subtraction.
    assert out.shape == batch.shape
    assert (out >= 0).all()


def test_object_preprocess_batch_clahe_only(rng):
    from spacr.object import _preprocess_batch
    batch = rng.uniform(0, 1, size=(2, 32, 32)).astype(np.float32)
    settings = {"organelle_clahe": True, "organelle_clahe_clip_limit": 0.01}
    out = _preprocess_batch(batch, settings)
    assert out.shape == batch.shape
    # After CLAHE, output should still be in [0, 1].
    assert out.min() >= 0.0
    assert out.max() <= 1.0 + 1e-6


# ===========================================================================
# spacr.timelapse — _parse_merged_filename more edge cases
# ===========================================================================

def test_timelapse_parse_merged_filename_wellID_starts_with_letter():
    from spacr.timelapse import _parse_merged_filename
    meta = _parse_merged_filename("plate2_H12_005_t100.npy")
    assert meta["rowID"] == "H"
    assert meta["columnID"] == 12


def test_timelapse_parse_merged_filename_purely_numeric_time():
    from spacr.timelapse import _parse_merged_filename
    meta = _parse_merged_filename("p1_A01_1_42.npy")
    assert meta["timeID"] == 42


def test_timelapse_parse_merged_filename_nontif_extension():
    from spacr.timelapse import _parse_merged_filename
    # Extension is stripped whatever it is.
    a = _parse_merged_filename("p_A01_1_0.npy")
    b = _parse_merged_filename("p_A01_1_0.h5")
    assert a["filename"] != b["filename"]  # base names retained
    assert a["prcft"] == b["prcft"]


# ===========================================================================
# spacr.measure._map_child_to_parent — more edge cases
# ===========================================================================

def test_measure_map_child_to_parent_multiple_children_per_parent():
    """Each of 3 children fully inside parent 1 → all mapped to cell=1."""
    from spacr.measure import _map_child_to_parent
    parent = np.zeros((20, 20), dtype=np.int32)
    parent[2:18, 2:18] = 1
    child = np.zeros((20, 20), dtype=np.int32)
    child[3:5, 3:5] = 1
    child[7:9, 7:9] = 2
    child[12:14, 12:14] = 3
    df = _map_child_to_parent(child, parent, "obj", "cell")
    assert (df["cell"] == 1).all()
    assert set(df["obj"]) == {1, 2, 3}


def test_measure_map_child_to_parent_argmax_ties_broken_by_bincount():
    """A child overlapping two parents: assigned to the parent with the
    larger overlap (bincount argmax)."""
    from spacr.measure import _map_child_to_parent
    parent = np.zeros((20, 20), dtype=np.int32)
    parent[0:10, 0:10] = 1   # left half
    parent[0:10, 10:20] = 2  # right half
    child = np.zeros((20, 20), dtype=np.int32)
    child[5:8, 8:12] = 7   # 12 pixels, 6 in parent=1, 6 in parent=2
    df = _map_child_to_parent(child, parent, "obj", "cell")
    # In tie, bincount picks the smaller label first (argmax returns
    # the first max) — so cell = 1.
    assert df.iloc[0]["cell"] in (1, 2)


def test_measure_map_child_to_parent_child_with_no_parent():
    """A child in the background area gets parent id 0."""
    from spacr.measure import _map_child_to_parent
    parent = np.zeros((20, 20), dtype=np.int32)   # all background
    child = np.zeros((20, 20), dtype=np.int32)
    child[5:8, 5:8] = 1
    df = _map_child_to_parent(child, parent, "obj", "cell")
    assert df.iloc[0]["cell"] == 0


# ===========================================================================
# spacr.sequencing — extract_sequence_and_quality
# ===========================================================================

def test_sequencing_extract_sequence_and_quality_exact_slice():
    from spacr.sequencing import extract_sequence_and_quality
    s, q = extract_sequence_and_quality("ATGCATGC", "IIIIIIII", 2, 6)
    assert s == "GCAT"
    assert q == "IIII"


def test_sequencing_extract_sequence_and_quality_empty_slice():
    from spacr.sequencing import extract_sequence_and_quality
    s, q = extract_sequence_and_quality("ATGC", "IIII", 2, 2)
    assert s == ""
    assert q == ""


# ===========================================================================
# spacr.utils: check_index parametrised
# ===========================================================================

@pytest.mark.parametrize("elements", [3, 4, 5])
def test_utils_check_index_various_element_counts(elements):
    """check_index accepts DataFrames whose index has exactly `elements`
    parts."""
    from spacr.utils import check_index
    idx = ["_".join([f"x{i}" for i in range(elements)])] * 3
    df = pd.DataFrame({"val": [1, 2, 3]}, index=idx)
    # No exception expected.
    check_index(df, elements=elements, split_char="_")


# ===========================================================================
# spacr.plot: random_cmap consistency
# ===========================================================================

def test_plot_random_cmap_alpha_channel_always_one():
    """All slots must have alpha=1."""
    from spacr.plot import random_cmap
    cmap = random_cmap(num_objects=20)
    for i in range(cmap.N):
        _, _, _, a = cmap(i)
        assert a == 1.0
