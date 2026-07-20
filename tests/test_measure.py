"""
Tests for spacr.measure — measurement primitives on synthetic masks.

Full pipelines (measure_crop) hit sqlite + heavy image I/O and are
covered by higher-level integration tests, not here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import spacr.measure as M


# ---------------------------------------------------------------------------
# get_components — parent→child mapping via label overlap
# ---------------------------------------------------------------------------

def test_get_components_empty_masks():
    empty = np.zeros((32, 32), dtype=np.int32)
    nucleus_df, pathogen_df = M.get_components(empty, empty, empty)
    assert isinstance(nucleus_df, pd.DataFrame)
    assert isinstance(pathogen_df, pd.DataFrame)
    assert len(nucleus_df) == 0
    assert len(pathogen_df) == 0


def test_get_components_maps_to_correct_parents(synth_masks_multi):
    """Every nucleus/pathogen id in the returned mapping should belong to the
    cell it was drawn inside."""
    cell = synth_masks_multi["cell"]
    nucleus = synth_masks_multi["nucleus"]
    pathogen = synth_masks_multi["pathogen"]

    nucleus_df, pathogen_df = M.get_components(cell, nucleus, pathogen)

    # Columns are named as documented in the docstring.
    assert list(nucleus_df.columns) == ["cell_id", "nucleus"]
    assert list(pathogen_df.columns) == ["cell_id", "pathogen"]

    # Every cell id present in either df is a real cell id in the mask.
    real_cell_ids = set(int(x) for x in np.unique(cell) if x != 0)
    for cid in nucleus_df["cell_id"].dropna().astype(int):
        assert cid in real_cell_ids
    for cid in pathogen_df["cell_id"].dropna().astype(int):
        assert cid in real_cell_ids

    # Every child id present is a real object id in its own mask.
    real_nuc = set(int(x) for x in np.unique(nucleus) if x != 0)
    for nid in nucleus_df["nucleus"].dropna().astype(int):
        assert nid in real_nuc
    real_pth = set(int(x) for x in np.unique(pathogen) if x != 0)
    for pid in pathogen_df["pathogen"].dropna().astype(int):
        assert pid in real_pth


def test_get_components_nucleus_inside_cell_relationship(synth_masks_multi):
    """A nucleus mapped to cell_id C must actually overlap cell C in the mask."""
    cell = synth_masks_multi["cell"]
    nucleus = synth_masks_multi["nucleus"]
    nucleus_df, _ = M.get_components(cell, nucleus, np.zeros_like(cell))
    nucleus_df = nucleus_df.dropna()
    for row in nucleus_df.itertuples():
        cid = int(row.cell_id)
        nid = int(row.nucleus)
        # The overlap between "cell == cid" and "nucleus == nid" must be > 0
        overlap = ((cell == cid) & (nucleus == nid)).sum()
        assert overlap > 0, (
            f"reported nucleus {nid} in cell {cid} but they don't overlap on the mask"
        )


# ---------------------------------------------------------------------------
# _map_child_to_parent — one row per child, argmax-overlap parent
# ---------------------------------------------------------------------------

def test_map_child_to_parent_returns_one_row_per_child(synth_masks_multi):
    parent = synth_masks_multi["cell"]
    child = synth_masks_multi["nucleus"]
    n_children = len([i for i in np.unique(child) if i != 0])
    out = M._map_child_to_parent(child, parent, child_name="nucleus", parent_name="cell")
    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == {"nucleus", "cell"}
    assert len(out) == n_children


def test_map_child_to_parent_handles_orphaned_children():
    """A child in an area with zero-labelled parent gets parent id 0."""
    parent = np.zeros((32, 32), dtype=np.int32)
    child = np.zeros((32, 32), dtype=np.int32)
    child[5:10, 5:10] = 1
    out = M._map_child_to_parent(child, parent, "obj", "cell")
    assert list(out["cell"]) == [0]


# ---------------------------------------------------------------------------
# _periphery_intensity — reports one row per region
# ---------------------------------------------------------------------------

def test_periphery_intensity_row_per_region(synth_mask_2d, synth_image_2d):
    stats = M._periphery_intensity(synth_mask_2d, synth_image_2d)
    n_regions = len([i for i in np.unique(synth_mask_2d) if i != 0])
    assert len(stats) == n_regions
    # First entry of each tuple is the region label.
    for row in stats:
        assert row[0] in np.unique(synth_mask_2d)


def test_periphery_intensity_returns_9_tuple(synth_mask_2d, synth_image_2d):
    stats = M._periphery_intensity(synth_mask_2d, synth_image_2d)
    for row in stats:
        # label + mean + 7 percentile stats = 9 items
        assert len(row) == 9


# ---------------------------------------------------------------------------
# _calculate_homogeneity — texture GLCM smoke
# ---------------------------------------------------------------------------

def test_calculate_homogeneity_shape(synth_mask_2d, synth_image_2d):
    df = M._calculate_homogeneity(synth_mask_2d, synth_image_2d, distances=[2, 4])
    n_regions = len([i for i in np.unique(synth_mask_2d) if i != 0])
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["homogeneity_distance_2", "homogeneity_distance_4"]
    assert len(df) == n_regions
    # GLCM homogeneity is in [0, 1].
    for col in df.columns:
        assert ((df[col] >= 0) & (df[col] <= 1)).all()


def test_calculate_homogeneity_uses_default_distances(synth_mask_2d, synth_image_2d):
    df = M._calculate_homogeneity(synth_mask_2d, synth_image_2d)
    # Docstring documents [2,4,8,16,32,64] as the default.
    assert list(df.columns) == [f"homogeneity_distance_{d}" for d in (2, 4, 8, 16, 32, 64)]


# ---------------------------------------------------------------------------
# _calculate_zernike — sanity when mask has zero regions vs. some regions
# ---------------------------------------------------------------------------

def test_calculate_zernike_returns_input_when_mask_empty():
    mask = np.zeros((32, 32), dtype=np.int32)
    df = pd.DataFrame({"a": [1, 2, 3]})
    out = M._calculate_zernike(mask, df, degree=4)
    # No regions → nothing to append → identical DataFrame.
    pd.testing.assert_frame_equal(out.reset_index(drop=True), df.reset_index(drop=True))


def test_calculate_zernike_appends_columns_when_regions_present(synth_mask_2d):
    n_regions = len([i for i in np.unique(synth_mask_2d) if i != 0])
    df = pd.DataFrame({"cell_id": range(n_regions)})
    out = M._calculate_zernike(synth_mask_2d.astype(np.int32), df, degree=4)
    zernike_cols = [c for c in out.columns if c.startswith("zernike_")]
    assert len(zernike_cols) > 0
    assert len(out) == n_regions
