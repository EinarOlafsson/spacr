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
    pytest.importorskip(
        "mahotas",
        reason="numerical Zernike descriptors require the optional spacr[zernike] extra",
    )
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
    df = M._summarize_organelles_per_parent(
        organelle, cell, ch, parent_name="cell")
    assert isinstance(df, pd.DataFrame)


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
    out = M._morphological_measurements(
        cell, nucleus, pathogen, organelle, cytoplasm,
        settings, zernike=False, degree=8)
    assert isinstance(out, tuple) and len(out) == 5


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
    out = M._intensity_measurements(
        cell, nucleus, pathogen, organelle, cytoplasm, ch,
        settings, periphery=True, outside=True)
    assert isinstance(out, tuple) and len(out) == 5


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


def test_calculate_homogeneity_caps_an_offset_wider_than_the_region():
    """An offset wider than a region is read at the widest one that fits (792)."""
    m = np.zeros((12, 12), dtype=np.int32)
    m[2:8, 3:9] = 1
    ch = np.arange(m.size, dtype=np.uint8).reshape(m.shape)

    df = M._calculate_homogeneity(m, ch, distances=[2, 5, 6, 32])

    assert np.isfinite(df.loc[0, "homogeneity_distance_2"])
    assert df.loc[0, "homogeneity_distance_6"] == df.loc[0, "homogeneity_distance_5"]
    assert df.loc[0, "homogeneity_distance_32"] == df.loc[0, "homogeneity_distance_5"]


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
    out = M._calculate_radial_distribution(cell, obj, ch, num_bins=4)
    # one entry per (cell_label, object_label, channel_index), each a bin vector
    assert list(out) == [(1, 1, 0)]
    assert out[(1, 1, 0)].shape == (4,)
    assert np.isfinite(out[(1, 1, 0)]).all()


def test_calculate_correlation_object_level():
    """One Manders definition, and it is the standards-compliant one.

    The `M1_correlation_<t>` / `M2_correlation_<t>` pair was removed on
    2026-09-02 (instruction 337): it was never Manders' coefficient, it was
    ~99% self-redundant, and it shipped ON by default beside the correct
    columns under names that did not say which definition produced them.

    `manders_thresholds` is still passed here on purpose -- a settings dict
    from before the change still carries it, and the function must ignore it
    rather than fail on it.
    """
    m = _two_object_mask()
    c1 = np.random.default_rng(5).random((32, 32)).astype(np.float32)
    c2 = np.random.default_rng(6).random((32, 32)).astype(np.float32)
    df = M._calculate_correlation_object_level(
        c1, c2, m, {"manders_thresholds": [15, 85]})
    assert "Pearson_correlation" in df.columns
    assert len(df) == 2

    for column in ("manders_m1", "manders_m2", "manders_overlap_coefficient"):
        assert column in df.columns, column
    assert not [c for c in df.columns if c.startswith(("M1_correlation",
                                                       "M2_correlation"))]


def test_the_manders_coefficients_need_no_setting_at_all():
    """The trio is unconditional: an empty settings dict still produces it.

    `corrected_manders` used to gate these and is retired. Driving the
    function with NOTHING in the dict is what proves the gate is gone rather
    than defaulted to True somewhere.
    """
    m = _two_object_mask()
    c1 = np.random.default_rng(5).random((32, 32)).astype(np.float32)
    c2 = np.random.default_rng(6).random((32, 32)).astype(np.float32)
    df = M._calculate_correlation_object_level(c1, c2, m, {})
    assert "manders_overlap_coefficient" in df.columns


def test_the_retired_manders_switch_is_reported_as_retired():
    """A settings CSV naming `corrected_manders` says so, not "unknown"."""
    from spacr.validate import RETIRED_SETTINGS

    assert "corrected_manders" in RETIRED_SETTINGS
    assert RETIRED_SETTINGS["corrected_manders"] == ""


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


def test_img_list_to_grid_keeps_rgb_channel_order():
    red = np.zeros((8, 8, 3), dtype=np.uint8)
    red[..., 0] = 255
    fig = M.img_list_to_grid([red])
    shown = np.asarray(fig.axes[0].images[0].get_array())
    assert shown[0, 0].tolist() == [255, 0, 0]
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------------------------------------------------------------------------
# object classical-segmentation helpers
# ---------------------------------------------------------------------------


def test_build_object_settings():
    s = {
        "organelle_model_name": "cyto", "organelle_diameter": 30,
        "organelle_min_area": 5, "organelle_max_area": 500,
        "organelle_resample": True, "organelle_remove_border": True,
    }
    out = OBJ._build_object_settings(s)
    assert out["model_name"] == "cyto" and out["merge"] is False


# ---------------------------------------------------------------------------
# measure.py DB / IO tail helpers
# ---------------------------------------------------------------------------

def test_get_object_counts(tmp_path):
    import sqlite3
    db_dir = tmp_path / "measurements"
    db_dir.mkdir()
    with sqlite3.connect(str(db_dir / "measurements.db")) as conn:
        pd.DataFrame({
            "count_type": ["cell", "cell", "nucleus"],
            "file_name": ["a", "b", "a"],
            "object_count": [10, 20, 5],
        }).to_sql("object_counts", conn, index=False)
    out = M.get_object_counts(str(tmp_path))
    cell = out[out["count_type"] == "cell"].iloc[0]
    assert cell["total_object_count"] == 30
    assert cell["avg_object_count_per_file_name"] == 15


def test_generate_cellpose_train_set(tmp_path):
    import tifffile
    src = tmp_path / "exp1"
    (src / "masks").mkdir(parents=True)
    # mask with 6 objects (>= min_objects=5) → copied
    big = np.zeros((16, 16), dtype=np.uint16)
    for i in range(1, 7):
        big[i, i] = i
    tifffile.imwrite(str(src / "masks" / "m1.tif"), big)
    tifffile.imwrite(str(src / "m1.tif"), big)   # the "image" alongside
    # mask with 1 object → skipped
    small = np.zeros((16, 16), dtype=np.uint16); small[0, 0] = 1
    tifffile.imwrite(str(src / "masks" / "m2.tif"), small)
    tifffile.imwrite(str(src / "m2.tif"), small)

    dst = tmp_path / "trainset"
    M.generate_cellpose_train_set([str(src)], str(dst), min_objects=5)
    copied = list((dst / "masks").glob("*.tif"))
    assert len(copied) == 1
    assert copied[0].name == "exp1_m1.tif"


def test_process_meassure_crop_results(tmp_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(); ax.plot([0, 1], [0, 1])
    partial = [
        None,                                    # None-skip branch
        (0, 1.2, 3, {"cell__overlay": fig}),     # fig-save branch
    ]
    settings = {"src": str(tmp_path / "run" / "data")}
    (tmp_path / "run").mkdir()
    with pytest.deprecated_call(match="process_measure_crop_results"):
        M.process_meassure_crop_results(partial, settings)
    assert list((tmp_path / "run" / "results").rglob("*.pdf"))


def test_process_measure_crop_results(tmp_path):
    settings = {"src": str(tmp_path / "merged")}
    M.process_measure_crop_results([None], settings)
    assert not (tmp_path / "results").exists()


# ---------------------------------------------------------------------------
# _analyze_cytoskeleton (regression: skeleton_branch_analysis did not exist)
# ---------------------------------------------------------------------------

def test_analyze_cytoskeleton():
    size = 48
    # a filamentous object with a branch, on one channel of a 3-channel array
    arr = np.zeros((size, size, 3), dtype=np.float32)
    rng = np.random.default_rng(3)
    # bright cross-shaped filament (has a branch point at the centre)
    arr[20:28, 6:42, 1] = 200 + rng.random((8, 36)) * 50
    arr[6:42, 20:28, 1] = 200 + rng.random((36, 8)) * 50
    mask = np.zeros((size, size), dtype=np.int32)
    mask[4:44, 4:44] = 1
    df = M._analyze_cytoskeleton(arr, mask, channel=1)
    assert "skeleton_length" in df.columns
    assert "skeleton_branch_points" in df.columns
    assert len(df) == 1
    assert df["skeleton_branch_points"].iloc[0] >= 0
