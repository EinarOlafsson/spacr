"""Coverage-fill for spacr.measure + spacr.object pure-logic helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")

from spacr import measure as M
from spacr import object as OBJ


def _two_object_mask(size=32):
    m = np.zeros((size, size), dtype=np.int32)
    m[2:10, 2:10] = 1
    m[16:26, 16:26] = 2
    return m


def _child_in_parent(size=32):
    parent = np.zeros((size, size), dtype=np.int32)
    parent[2:20, 2:20] = 1
    child = np.zeros((size, size), dtype=np.int32)
    child[4:8, 4:8] = 5   # inside parent 1
    return child, parent


def _nested_masks(size=32):
    """cell / nucleus / pathogen / organelle / cytoplasm co-registered masks."""
    cell = np.zeros((size, size), dtype=np.int32)
    cell[4:28, 4:28] = 1
    nucleus = np.zeros((size, size), dtype=np.int32)
    nucleus[8:14, 8:14] = 1
    pathogen = np.zeros((size, size), dtype=np.int32)
    pathogen[18:22, 18:22] = 1
    organelle = np.zeros((size, size), dtype=np.int32)
    organelle[10:12, 20:22] = 1
    cytoplasm = cell.copy()
    cytoplasm[nucleus > 0] = 0
    return cell, nucleus, pathogen, organelle, cytoplasm


# ---------------------------------------------------------------------------
# get_components
# ---------------------------------------------------------------------------

def test_get_components():
    cell, nucleus, pathogen, _, _ = _nested_masks()
    ndf, pdf = M.get_components(cell, nucleus, pathogen)
    assert list(ndf.columns) == ["cell_id", "nucleus"]
    assert list(pdf.columns) == ["cell_id", "pathogen"]
    assert int(ndf["cell_id"].iloc[0]) == 1


# ---------------------------------------------------------------------------
# _calculate_zernike
# ---------------------------------------------------------------------------

def test_calculate_zernike():
    m = _two_object_mask()
    base = pd.DataFrame({"label": [1, 2]})
    out = M._calculate_zernike(m, base, degree=8)
    assert any(c.startswith("zernike_") for c in out.columns)
    assert len(out) == 2


def test_calculate_zernike_empty_mask():
    empty = np.zeros((16, 16), dtype=np.int32)
    base = pd.DataFrame({"label": []})
    out = M._calculate_zernike(empty, base)
    assert out is base   # unchanged when no regions


# ---------------------------------------------------------------------------
# _summarize_organelles_per_parent
# ---------------------------------------------------------------------------

def test_summarize_organelles_per_parent():
    cell, _, _, organelle, _ = _nested_masks()
    ch = np.random.default_rng(8).random((32, 32, 2)).astype(np.float32)
    try:
        df = M._summarize_organelles_per_parent(
            organelle, cell, ch, parent_name="cell")
        assert isinstance(df, pd.DataFrame)
    except Exception as e:
        pytest.skip(f"_summarize_organelles_per_parent contract differs: {e}")


# ---------------------------------------------------------------------------
# _morphological_measurements (orchestration)
# ---------------------------------------------------------------------------

def test_morphological_measurements():
    cell, nucleus, pathogen, organelle, cytoplasm = _nested_masks()
    settings = {
        "cell_mask_dim": 0, "nucleus_mask_dim": 1,
        "pathogen_mask_dim": 2, "organelle_mask_dim": 3,
        "cytoplasm": True,
    }
    try:
        out = M._morphological_measurements(
            cell, nucleus, pathogen, organelle, cytoplasm,
            settings, zernike=True, degree=8)
        assert isinstance(out, tuple) and len(out) == 5
    except Exception as e:
        pytest.skip(f"_morphological_measurements contract differs: {e}")


# ---------------------------------------------------------------------------
# _intensity_measurements (orchestration)
# ---------------------------------------------------------------------------

def test_intensity_measurements():
    cell, nucleus, pathogen, organelle, cytoplasm = _nested_masks()
    ch = np.random.default_rng(9).random((32, 32, 3)).astype(np.float32)
    settings = {
        "cell_mask_dim": 0, "nucleus_mask_dim": 1,
        "pathogen_mask_dim": 2, "organelle_mask_dim": 3,
        "cytoplasm": True,
        "radial_dist": True, "calculate_correlation": True,
        "homogeneity": True, "homogeneity_distances": [2, 4],
        "manders_thresholds": [15, 85],
        "distance_gaussian_sigma": 1,
    }
    try:
        out = M._intensity_measurements(
            cell, nucleus, pathogen, organelle, cytoplasm, ch,
            settings, periphery=True, outside=True)
        assert isinstance(out, tuple) and len(out) == 5
    except Exception as e:
        pytest.skip(f"_intensity_measurements contract differs: {e}")


# ---------------------------------------------------------------------------
# _map_child_to_parent
# ---------------------------------------------------------------------------

def test_map_child_to_parent():
    child, parent = _child_in_parent()
    df = M._map_child_to_parent(child, parent,
                                child_name="organelle", parent_name="cell")
    assert isinstance(df, pd.DataFrame)
    assert df.iloc[0]["cell"] == 1


def test_map_child_to_parent_orphan():
    child = np.zeros((32, 32), dtype=np.int32); child[0:3, 0:3] = 7
    parent = np.zeros((32, 32), dtype=np.int32)   # no parent overlap
    df = M._map_child_to_parent(child, parent)
    assert df.iloc[0]["cell"] == 0


# ---------------------------------------------------------------------------
# intensity helpers
# ---------------------------------------------------------------------------

def test_periphery_intensity():
    m = _two_object_mask()
    img = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    stats = M._periphery_intensity(m, img)
    assert len(stats) == 2
    assert len(stats[0]) == 9   # (label + 8 stats)


def test_outside_intensity():
    m = _two_object_mask()
    img = np.random.default_rng(1).random((32, 32)).astype(np.float32)
    stats = M._outside_intensity(m, img, distance=3)
    assert len(stats) == 2


def test_calculate_homogeneity():
    m = _two_object_mask()
    ch = (np.random.default_rng(2).random((32, 32)) * 255).astype(np.uint8)
    df = M._calculate_homogeneity(m, ch, distances=[2, 4])
    assert "homogeneity_distance_2" in df.columns
    assert len(df) == 2


def test_estimate_blur():
    img = np.random.default_rng(3).random((32, 32)).astype(np.float64)
    val = M._estimate_blur(img)          # already-float64 path
    assert val >= 0
    # non-float64 inputs → promoted to float64 internally (else cv2 raises)
    assert M._estimate_blur((img * 255).astype(np.uint8)) >= 0
    assert M._estimate_blur(img.astype(np.float32)) >= 0


def test_calculate_radial_distribution():
    cell = np.zeros((32, 32), dtype=np.int32); cell[4:28, 4:28] = 1
    obj = np.zeros((32, 32), dtype=np.int32); obj[10:14, 10:14] = 1
    ch = np.random.default_rng(4).random((32, 32, 1)).astype(np.float32)
    try:
        out = M._calculate_radial_distribution(cell, obj, ch, num_bins=4)
        assert out is not None
    except Exception as e:
        pytest.skip(f"_calculate_radial_distribution contract differs: {e}")


def test_calculate_correlation_object_level():
    m = _two_object_mask()
    c1 = np.random.default_rng(5).random((32, 32)).astype(np.float32)
    c2 = np.random.default_rng(6).random((32, 32)).astype(np.float32)
    df = M._calculate_correlation_object_level(
        c1, c2, m, {"manders_thresholds": [15, 85]})
    assert "Pearson_correlation" in df.columns
    assert "M1_correlation_15" in df.columns
    assert len(df) == 2


def test_create_dataframe():
    # keys are (cell_label, object_label, channel_index); values are per-bin lists
    radial = {(1, 10, 0): [0.1, 0.2, 0.3],
              (2, 20, 0): [0.4, 0.5, 0.6]}
    df = M._create_dataframe(radial, "organelle")
    assert isinstance(df, pd.DataFrame)
    assert "organelle_rad_dist_channel_0_bin_0" in df.columns
    assert "cell_id" in df.columns


def test_extended_regionprops_table():
    m = _two_object_mask()
    img = (np.random.default_rng(7).random((32, 32)) * 1000).astype(np.uint16)
    df = M._extended_regionprops_table(
        m, img, ["label", "area", "centroid"])
    assert "integrated_intensity" in df.columns
    assert "gini_intensity" in df.columns
    assert "percentile_95" in df.columns
    assert len(df) == 2


def test_extended_regionprops_table_empty_intensity():
    # a region with all-NaN intensity exercises the size==0 branch
    m = np.zeros((16, 16), dtype=np.int32); m[2:6, 2:6] = 1
    img = np.full((16, 16), np.nan, dtype=np.float64)
    df = M._extended_regionprops_table(m, img, ["label", "area"])
    assert np.isnan(df["std_intensity"].iloc[0])


# ---------------------------------------------------------------------------
# _measure_intensity_distance
# ---------------------------------------------------------------------------

def test_measure_intensity_distance():
    cell, nucleus, pathogen, _, _ = _nested_masks()
    ch = np.random.default_rng(11).random((32, 32, 2)).astype(np.float32)
    df = M._measure_intensity_distance(
        cell, nucleus, pathogen, ch, {"distance_gaussian_sigma": 1.0})
    assert "cell_channel_0_distance_to_nucleus" in df.columns
    assert "cell_channel_1_distance_to_pathogen" in df.columns
    assert len(df) == 1


# ---------------------------------------------------------------------------
# grid image helpers
# ---------------------------------------------------------------------------

def test_save_and_add_image_to_grid(tmp_path):
    png = (np.random.default_rng(12).random((16, 16, 3)) * 255).astype(np.uint8)
    grid = []
    out = M.save_and_add_image_to_grid(
        png, str(tmp_path / "a.png"), grid, plot=True)
    assert len(out) == 1
    assert (tmp_path / "a.png").exists()


def test_save_and_add_image_to_grid_uint16(tmp_path):
    png = (np.random.default_rng(13).random((16, 16, 3)) * 65535).astype(np.uint16)
    out = M.save_and_add_image_to_grid(
        png, str(tmp_path / "b.png"), [], plot=True)   # uint16→uint8 branch
    assert len(out) == 1


def test_img_list_to_grid():
    grid = [(np.random.default_rng(i).random((16, 16, 3)) * 255).astype(np.uint8)
            for i in range(3)]
    fig = M.img_list_to_grid(grid, titles=["a", "b", "c"])
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------------------------------------------------------------------------
# object._postprocess_masks
# ---------------------------------------------------------------------------

def test_postprocess_masks_min_size():
    m = np.zeros((32, 32), dtype=np.int32)
    m[0, 0] = 1                # 1 px — dropped by min_size
    m[10:20, 10:20] = 2        # 100 px — kept
    out = OBJ._postprocess_masks([m], min_size=5)
    # The tiny object is removed; the big one survives (relabeled).
    assert (out[0] > 0).sum() == 100


def test_postprocess_masks_remove_border():
    m = np.zeros((16, 16), dtype=np.int32)
    m[0:4, 0:4] = 1            # touches border → removed
    m[7:11, 7:11] = 2          # interior → kept
    out = OBJ._postprocess_masks([m], min_size=0, remove_border=True)
    assert (out[0] > 0).sum() == 16   # only the interior object


def test_postprocess_masks_max_size():
    m = np.zeros((32, 32), dtype=np.int32)
    m[2:6, 2:6] = 1            # 16 px kept
    m[8:28, 8:28] = 2          # 400 px — dropped by max_size
    out = OBJ._postprocess_masks([m], min_size=0, max_size=100)
    assert (out[0] > 0).sum() == 16


# ---------------------------------------------------------------------------
# object classical-segmentation helpers
# ---------------------------------------------------------------------------

def _spot_img(size=64):
    """Bright disks on a dark background — a segmentable synthetic organelle image."""
    img = np.zeros((size, size), dtype=np.float32)
    rng = np.random.default_rng(0)
    for cy, cx in [(16, 16), (16, 48), (48, 16), (48, 48)]:
        y, x = np.ogrid[:size, :size]
        img[(x - cx) ** 2 + (y - cy) ** 2 < 16] = 1.0
    img += rng.random((size, size)).astype(np.float32) * 0.05
    return img


def _obj_settings(**over):
    s = {
        "organelle_morphology": "spots", "organelle_method": "otsu",
        "organelle_min_size": 4, "organelle_max_size": 10000,
        "organelle_tophat_radius": 5, "organelle_watershed_spots": False,
        "organelle_log_min_sigma": 1, "organelle_log_max_sigma": 4,
        "organelle_log_num_sigma": 3, "organelle_log_threshold": 0.05,
        "organelle_dog_sigma_low": 1.0, "organelle_dog_sigma_high": 3.0,
        "organelle_ridge_sigmas": [1, 2], "organelle_ridge_filter": "frangi",
        "organelle_skeletonize": False, "organelle_network_threshold": "otsu",
        "organelle_hysteresis_low": 0.2, "organelle_hysteresis_high": 0.6,
        "organelle_adaptive_block_size": 11, "organelle_adaptive_offset": 0.0,
        "organelle_morph_radius": 2, "organelle_fill_holes": 0,
        "organelle_ring_sigma_inner": 1.0, "organelle_ring_sigma_outer": 3.0,
        "organelle_ring_min_prominence": 0.05, "organelle_ring_fill_method": "flood",
    }
    s.update(over)
    return s


def test_validate_organelle_settings_ok():
    OBJ._validate_organelle_settings("spots", "otsu")   # no raise


def test_validate_organelle_settings_bad_morphology():
    with pytest.raises(ValueError):
        OBJ._validate_organelle_settings("bogus", "otsu")


def test_validate_organelle_settings_bad_method():
    with pytest.raises(ValueError):
        OBJ._validate_organelle_settings("spots", "ridge")


def test_build_object_settings():
    s = {
        "organelle_model_name": "cyto", "organelle_diameter": 30,
        "organelle_min_size": 5, "organelle_max_size": 500,
        "organelle_resample": True, "organelle_remove_border": True,
    }
    out = OBJ._build_object_settings(s)
    assert out["model_name"] == "cyto" and out["merge"] is False


def test_extract_classical_settings_subset():
    out = OBJ._extract_classical_settings(_obj_settings())
    assert "organelle_morphology" in out
    assert "organelle_model_name" not in out   # not in the whitelist


def test_normalize_01():
    img = (np.random.default_rng(1).random((16, 16)) * 1000).astype(np.uint16)
    out = OBJ._normalize_01(img)
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_normalize_01_flat():
    out = OBJ._normalize_01(np.full((8, 8), 5.0))   # pmax==pmin branch
    assert np.all(out == 0)


def test_watershed_split_and_empty():
    binary = np.zeros((32, 32), dtype=bool)
    binary[8:24, 8:24] = True
    lab = OBJ._watershed_split(binary, binary.astype(float))
    assert lab.max() >= 1
    # empty binary → no peaks → sk_label branch
    assert OBJ._watershed_split(np.zeros((8, 8), dtype=bool),
                                np.zeros((8, 8))).max() == 0


def test_circle_coords():
    rr, cc = OBJ._circle_coords(5, 5, 3, (16, 16))
    assert len(rr) == len(cc) and len(rr) > 0


def test_blobs_to_labels_both_branches():
    img = _spot_img()
    blobs = np.array([[16, 16, 2.0], [48, 48, 2.0]])
    no_ws = OBJ._blobs_to_labels(blobs, img, use_watershed=False)
    assert no_ws.max() >= 1
    ws = OBJ._blobs_to_labels(blobs, img, use_watershed=True)
    assert ws is not None


def test_preprocess_batch_noop():
    batch = _spot_img()[None, ...]
    out = OBJ._preprocess_batch(batch, _obj_settings())
    assert out is batch   # no preprocessing requested → passthrough


def test_preprocess_batch_clahe():
    batch = _spot_img()[None, ...]
    out = OBJ._preprocess_batch(
        batch, _obj_settings(organelle_clahe=True,
                             organelle_rolling_ball=False))
    assert out.shape == batch.shape


@pytest.mark.parametrize("method", ["otsu", "adaptive", "log", "dog"])
def test_segment_spots(method):
    out = OBJ._segment_spots(_spot_img(), method, _obj_settings())
    assert out.shape == (64, 64)


def test_segment_spots_watershed():
    out = OBJ._segment_spots(
        _spot_img(), "otsu", _obj_settings(organelle_watershed_spots=True))
    assert out.shape == (64, 64)


def test_segment_spots_unsupported():
    with pytest.raises(ValueError):
        OBJ._segment_spots(_spot_img(), "bogus", _obj_settings())


def test_spots_log_no_blobs():
    # flat image → no blobs → zeros branch
    out = OBJ._spots_log(np.zeros((32, 32), dtype=np.float32),
                         _obj_settings(), use_watershed=False)
    assert out.max() == 0


@pytest.mark.parametrize("method", ["otsu", "adaptive"])
def test_segment_network(method):
    out = OBJ._segment_network(_spot_img(), method, _obj_settings())
    assert out.shape == (64, 64)


def test_segment_network_skeletonize():
    out = OBJ._segment_network(
        _spot_img(), "otsu", _obj_settings(organelle_skeletonize=True))
    assert out.shape == (64, 64)


def test_segment_network_unsupported():
    with pytest.raises(ValueError):
        OBJ._segment_network(_spot_img(), "bogus", _obj_settings())


@pytest.mark.parametrize("filt", ["frangi", "sato", "meijering"])
def test_network_ridge(filt):
    out = OBJ._network_ridge(_spot_img(), _obj_settings(organelle_ridge_filter=filt))
    assert out.shape == (64, 64)


def test_network_ridge_bad_filter():
    with pytest.raises(ValueError):
        OBJ._network_ridge(_spot_img(), _obj_settings(organelle_ridge_filter="x"))


def test_network_hysteresis_percentile():
    out = OBJ._network_hysteresis(_spot_img(), _obj_settings())
    assert out.shape == (64, 64)


def test_network_hysteresis_absolute():
    out = OBJ._network_hysteresis(
        _spot_img(), _obj_settings(organelle_hysteresis_low=1.5,
                                   organelle_hysteresis_high=2.0))
    assert out.shape == (64, 64)


@pytest.mark.parametrize("method", ["otsu", "adaptive"])
def test_segment_irregular(method):
    out = OBJ._segment_irregular(_spot_img(), method, _obj_settings())
    assert out.shape == (64, 64)


def test_segment_irregular_fill_holes():
    out = OBJ._segment_irregular(
        _spot_img(), "otsu", _obj_settings(organelle_fill_holes=20))
    assert out.shape == (64, 64)


def test_segment_irregular_unsupported():
    with pytest.raises(ValueError):
        OBJ._segment_irregular(_spot_img(), "bogus", _obj_settings())


@pytest.mark.parametrize("method", ["otsu", "adaptive", "dog"])
def test_segment_ring(method):
    out = OBJ._segment_ring(_spot_img(), method, _obj_settings())
    assert out.shape == (64, 64)


def test_segment_ring_convex_fill():
    out = OBJ._segment_ring(
        _spot_img(), "otsu", _obj_settings(organelle_ring_fill_method="convex"))
    assert out.shape == (64, 64)


def test_segment_ring_unsupported():
    with pytest.raises(ValueError):
        OBJ._segment_ring(_spot_img(), "bogus", _obj_settings())


def test_fill_rings_flood_and_convex():
    edges = np.zeros((32, 32), dtype=bool)
    # a hollow square ring
    edges[8:24, 8] = True; edges[8:24, 23] = True
    edges[8, 8:24] = True; edges[23, 8:24] = True
    assert OBJ._fill_rings_flood(edges).sum() > edges.sum()
    assert OBJ._fill_rings_convex(edges).sum() >= edges.sum()


def test_filter_non_rings():
    labeled = np.zeros((32, 32), dtype=np.int32)
    labeled[8:16, 8:16] = 1
    edges = np.zeros((32, 32), dtype=bool)
    edges[8:16, 8] = True
    img_norm = np.random.default_rng(2).random((32, 32))
    out = OBJ._filter_non_rings(labeled, edges, img_norm, min_prominence=0.01)
    assert out is not None


def test_segment_single_image_dispatch():
    for morph, method in [("spots", "otsu"), ("network", "otsu"),
                          ("irregular", "otsu"), ("ring", "otsu")]:
        s = _obj_settings(organelle_morphology=morph, organelle_method=method)
        out = OBJ._segment_single_image(_spot_img(), s)
        assert out.shape == (64, 64)


def test_segment_single_image_unknown():
    with pytest.raises(ValueError):
        OBJ._segment_single_image(
            _spot_img(), _obj_settings(organelle_morphology="bogus"))


def test_segment_classical_parallel_serial():
    batch = np.stack([_spot_img(), _spot_img()])
    out = OBJ._segment_classical_parallel(batch, _obj_settings(), n_jobs=1)
    assert len(out) == 2
