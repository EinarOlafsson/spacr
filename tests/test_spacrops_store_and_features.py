"""spacr.spacrops — feature cache, construction, axis handling, feature/matching.

These tests exercise the pieces the stitcher stands on: the disk-backed
feature store (LRU, locking, corruption), TIFF axis normalisation, keypoint
extraction and descriptor matching.  Values are asserted against a known
ground truth wherever one exists.
"""
from __future__ import annotations

import os
import sys
import threading
import types

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
import tifffile

from spacr.spacrops import _DiskFeatureStore, spacrStitcher
from tests.spacrops_synth import blob_canvas, crop, write_cyx


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _feat(i: int, k: int = 3):
    return {
        "ds8": np.full((4, 4), i, dtype=np.uint8),
        "pts": (np.arange(2 * k, dtype=np.float32) + i).reshape(k, 2),
        "desc": np.full((k, 32), i % 251, dtype=np.uint8),
        "Hds": 4 + i, "Wds": 5 + i, "H": 40 + i, "W": 50 + i,
    }


def _stitcher(tmp_path, **kw):
    kw.setdefault("outdir", str(tmp_path / "out"))
    kw.setdefault("save_qc", False)
    kw.setdefault("save_stitched_default", False)
    kw.setdefault("feature_cache_mode", "ram")
    return spacrStitcher(**kw)


# ===========================================================================
# _DiskFeatureStore
# ===========================================================================

def test_store_roundtrip_preserves_every_field_and_dtype(tmp_path):
    store = _DiskFeatureStore(str(tmp_path / "cache"), max_ram_items=8)
    f = _feat(7)
    store.put("/img/a.tif", f)
    store._ram.clear()                      # force the disk read path
    got = store.get("/img/a.tif")

    assert np.array_equal(got["ds8"], f["ds8"])
    assert got["ds8"].dtype == np.uint8
    assert np.array_equal(got["pts"], f["pts"])
    assert got["pts"].dtype == np.float32
    assert np.array_equal(got["desc"], f["desc"])
    # scalars round-trip as plain python ints, not 0-d arrays
    assert (got["Hds"], got["Wds"], got["H"], got["W"]) == (11, 12, 47, 57)
    assert all(isinstance(got[k], int) for k in ("Hds", "Wds", "H", "W"))


def test_store_writes_one_npz_per_distinct_path(tmp_path):
    root = tmp_path / "cache"
    store = _DiskFeatureStore(str(root), max_ram_items=2)
    for i in range(5):
        store.put(f"/img/{i}.tif", _feat(i))
    files = sorted(p for p in os.listdir(root) if p.endswith(".npz"))
    assert len(files) == 5
    # the file name is the truncated sha1 of the absolute path
    assert os.path.basename(store._npz_path("/img/0.tif")) in files
    # re-putting the same path overwrites rather than adding a file
    store.put("/img/0.tif", _feat(99))
    assert len([p for p in os.listdir(root) if p.endswith(".npz")]) == 5


def test_store_lru_evicts_least_recently_used_not_oldest_inserted(tmp_path):
    store = _DiskFeatureStore(str(tmp_path / "c"), max_ram_items=2)
    store.put("/a.tif", _feat(1))
    store.put("/b.tif", _feat(2))
    # touch /a.tif so /b.tif becomes the least-recently-used entry
    assert store.get("/a.tif") is not None
    store.put("/c.tif", _feat(3))

    assert set(store._ram.keys()) == {"/a.tif", "/c.tif"}
    assert list(store._ram.keys())[-1] == "/c.tif"   # most recent at the tail
    # the evicted entry is still recoverable from disk with its exact payload
    b = store.get("/b.tif")
    assert int(b["ds8"][0, 0]) == 2 and b["H"] == 42


def test_store_get_reinserts_from_disk_and_respects_the_cap(tmp_path):
    store = _DiskFeatureStore(str(tmp_path / "c"), max_ram_items=1)
    store.put("/a.tif", _feat(1))
    store.put("/b.tif", _feat(2))            # evicts /a.tif
    assert list(store._ram.keys()) == ["/b.tif"]

    got = store.get("/a.tif")                # disk fallback re-inserts
    assert int(got["ds8"][0, 0]) == 1
    assert list(store._ram.keys()) == ["/a.tif"]   # cap still 1


def test_store_miss_returns_none_without_creating_a_file(tmp_path):
    root = tmp_path / "c"
    store = _DiskFeatureStore(str(root))
    assert store.get("/never/seen.tif") is None
    assert [p for p in os.listdir(root) if p.endswith(".npz")] == []


def test_store_corrupt_npz_is_treated_as_a_miss_and_removed(tmp_path):
    """A truncated NPZ (run killed mid-write) must not poison the cache."""
    store = _DiskFeatureStore(str(tmp_path / "c"), verbose=True)
    path = "/img/x.tif"
    npz = store._npz_path(path)
    with open(npz, "wb") as fh:
        fh.write(b"PK\x03\x04 not really a zip")

    assert store.get(path) is None           # miss, not an exception
    assert not os.path.exists(npz)           # poisoned entry deleted
    # and the store is usable again for that same path
    store.put(path, _feat(5))
    assert int(store.get(path)["ds8"][0, 0]) == 5


def test_store_truncated_npz_is_also_treated_as_a_miss(tmp_path):
    store = _DiskFeatureStore(str(tmp_path / "c"))
    path = "/img/y.tif"
    store.put(path, _feat(4))
    store._ram.clear()
    npz = store._npz_path(path)
    data = open(npz, "rb").read()
    with open(npz, "wb") as fh:              # chop the archive in half
        fh.write(data[: len(data) // 2])
    assert store.get(path) is None


def test_store_is_thread_safe_under_concurrent_put_and_get(tmp_path):
    store = _DiskFeatureStore(str(tmp_path / "c"), max_ram_items=4)
    n_threads, per_thread = 8, 12
    errors = []

    def worker(t):
        try:
            for i in range(per_thread):
                key = f"/t{t}/{i}.tif"
                store.put(key, _feat(t * per_thread + i))
                got = store.get(key)
                assert got is not None
                assert int(got["ds8"][0, 0]) == (t * per_thread + i) % 256
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    # the RAM LRU never exceeded its cap despite concurrent inserts
    assert len(store._ram) <= 4
    # every entry survived on disk
    for t in range(n_threads):
        for i in range(per_thread):
            assert store.get(f"/t{t}/{i}.tif") is not None


def test_store_key_for_path_normalises_relative_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _DiskFeatureStore._key_for_path("sub/x.tif") == \
        _DiskFeatureStore._key_for_path(str(tmp_path / "sub" / "x.tif"))
    assert _DiskFeatureStore._key_for_path("/a.tif") != _DiskFeatureStore._key_for_path("/b.tif")


# ===========================================================================
# spacrStitcher construction / validation
# ===========================================================================

def test_init_rejects_unknown_detector(tmp_path):
    with pytest.raises(ValueError, match="detector must be"):
        spacrStitcher(detector="AKAZE", outdir=str(tmp_path / "o"))


def test_init_rejects_unknown_feature_cache_mode(tmp_path):
    with pytest.raises(ValueError, match="feature_cache_mode"):
        spacrStitcher(feature_cache_mode="s3", outdir=str(tmp_path / "o"))


def test_init_rejects_axes_without_y_and_x(tmp_path):
    with pytest.raises(ValueError, match="must include 'Y' and 'X'"):
        spacrStitcher(arr_axes="CZT", outdir=str(tmp_path / "o"))


def test_init_disk_mode_creates_cache_dir_and_store(tmp_path):
    st = spacrStitcher(outdir=str(tmp_path / "o"), feature_cache_mode="disk")
    assert st.feature_cache_dir == os.path.join(st.outdir, "feat_cache")
    assert isinstance(st._store, _DiskFeatureStore)
    assert os.path.isdir(st.feature_cache_dir)

    custom = str(tmp_path / "elsewhere")
    st2 = spacrStitcher(outdir=str(tmp_path / "o2"), feature_cache_mode="disk",
                        feature_cache_dir=custom, max_ram_features=3)
    assert st2._store.max_ram == 3
    assert os.path.isdir(custom)


def test_init_ram_mode_has_no_disk_store(tmp_path):
    st = spacrStitcher(outdir=str(tmp_path / "o"), feature_cache_mode="RAM")
    assert st._store is None
    assert st.feature_cache_mode == "ram"
    assert st._feat_cache == {}


def test_init_normalises_scalar_options(tmp_path):
    st = spacrStitcher(outdir=str(tmp_path / "o"), max_keypoints=None,
                       downsample=0.25, canny=[10, 20], n_workers_features=3,
                       arr_axes="czyx", z_index=2, t_index=1, mip=True)
    assert st.max_keypoints is None
    assert st.canny == (10, 20)
    assert st.n_workers_features == 3
    assert st.arr_axes == "CZYX"
    assert (st.z_index, st.t_index, st.mip) == (2, 1, True)
    assert st.outdir == os.path.abspath(str(tmp_path / "o")) and os.path.isdir(st.outdir)


def test_init_default_feature_workers_is_half_the_cpus(tmp_path):
    st = spacrStitcher(outdir=str(tmp_path / "o"))
    assert st.n_workers_features == max(1, os.cpu_count() // 2)


@pytest.mark.skipif(not hasattr(__import__("cv2"), "SIFT_create"),
                    reason="opencv build without SIFT")
def test_init_sift_selects_flann(tmp_path):
    st = spacrStitcher(detector="sift", outdir=str(tmp_path / "o"))
    assert st.detector == "SIFT" and st._use_flann is True
    assert hasattr(st, "_flann")


def test_ensure_dir_is_idempotent(tmp_path):
    p = str(tmp_path / "deep" / "nested")
    assert spacrStitcher._ensure_dir(p) == os.path.abspath(p)
    assert spacrStitcher._ensure_dir(p) == os.path.abspath(p)   # already exists
    assert os.path.isdir(p)


# ===========================================================================
# masks
# ===========================================================================

def _bimodal(seed=17, H=64, W=64, box=(20, 44)):
    """Noisy background with a bright square: a non-degenerate Otsu histogram."""
    rng = np.random.default_rng(seed)
    img = (rng.random((H, W)) * 40).astype(np.uint8)
    img[box[0]:box[1], box[0]:box[1]] = 220
    img[0:6, 0:6] = 0                       # a guaranteed-dark corner
    return img


def test_foreground_mask_otsu_marks_the_bright_object_not_the_background(tmp_path):
    """Regression: cv2.threshold returns (level, image) - unpacking it the
    other way round made this mask select the background instead."""
    st = _stitcher(tmp_path)
    img = _bimodal()
    m = st._foreground_mask(img)
    assert m.dtype == bool
    assert m[20:44, 20:44].all()                  # the whole bright square
    assert not m[0:6, 0:6].any()                  # and none of the dark corner
    # the mask is the object (576 px), not its ~3500 px complement
    assert 24 * 24 <= m.sum() < 700


def test_foreground_mask_dilate_grows_the_mask(tmp_path):
    img = _bimodal(box=(28, 36))
    base = _stitcher(tmp_path)._foreground_mask(img).sum()
    grown = _stitcher(tmp_path, dilate_ksize=5)._foreground_mask(img).sum()
    assert grown > base


def test_foreground_mask_blur_is_applied_before_threshold(tmp_path):
    rng = np.random.default_rng(3)
    img = (rng.random((48, 48)) * 40).astype(np.uint8)
    img[18:30, 18:30] = 220
    img[5, 5] = 255                       # single hot pixel in the background
    sharp = _stitcher(tmp_path)._foreground_mask(img)
    blurred = _stitcher(tmp_path, blur_sigma=2.0)._foreground_mask(img)
    assert sharp[5, 5]                    # survives without blurring
    assert not blurred[5, 5]              # smoothed away by the Gaussian
    assert blurred[24, 24] and sharp[24, 24]


def test_foreground_and_outline_masks_are_empty_when_source_is_none(tmp_path):
    st = _stitcher(tmp_path, outline_source="NONE")
    img = np.full((16, 16), 255, np.uint8)
    assert st._foreground_mask(img).sum() == 0
    assert st._outline_mask(img).sum() == 0
    assert st._foreground_mask(img).dtype == bool


def test_outline_mask_traces_the_object_border(tmp_path):
    st = _stitcher(tmp_path, blur_sigma=1.5)
    img = _bimodal(box=(20, 44))
    edges = st._outline_mask(img)
    assert edges.dtype == bool
    # ~ the perimeter of a 24x24 square, and nothing in the background
    assert 80 <= edges.sum() <= 110
    ys, xs = np.nonzero(edges)
    assert ys.min() >= 18 and ys.max() <= 45
    assert xs.min() >= 18 and xs.max() <= 45
    assert not edges[32, 32]                 # hollow, not filled


def test_outline_mask_line_thickness_widens_the_edges(tmp_path):
    img = _bimodal(box=(20, 44))
    thin = _stitcher(tmp_path, blur_sigma=1.5)._outline_mask(img).sum()
    thick = _stitcher(tmp_path, blur_sigma=1.5, line_thickness=5)._outline_mask(img).sum()
    assert thin > 0
    assert thick > thin


def test_outline_mask_dilate_ksize_grows_before_canny(tmp_path):
    img = _bimodal(box=(28, 36))
    plain = _stitcher(tmp_path, blur_sigma=1.5)._outline_mask(img)
    dil = _stitcher(tmp_path, blur_sigma=1.5, dilate_ksize=7)._outline_mask(img)
    assert np.nonzero(plain)[0].size > 0
    assert np.nonzero(dil)[0].max() > np.nonzero(plain)[0].max()


def _install_fake_cellpose(monkeypatch, mask, n_returns=3):
    """Stand in for the Cellpose 4 API.

    This used to fake ``models.Cellpose`` -- the pre-SAM wrapper class -- with
    a four-value ``eval``, which is exactly the Cellpose-3 call spacrops.py
    was making. The mock and the code agreed with each other and with nothing
    else, so these tests passed green while ``outline_source='cellpose'``
    raised ``AttributeError: module 'cellpose.models' has no attribute
    'Cellpose'`` against every installed Cellpose 4. Cellpose 4 ships
    ``CellposeModel`` only, and its ``eval`` returns three values.
    """
    # spacrops resolves the model name through spacr.utils, which imports
    # cellpose at module level. Load the REAL spacr.utils before the fake goes
    # into sys.modules -- in a run it is imported long before any stitching.
    import spacr.utils  # noqa: F401

    cellpose = types.ModuleType("cellpose")
    models = types.ModuleType("cellpose.models")
    seen = {}

    class CellposeModel:
        def __init__(self, **kwargs):
            seen["init_kwargs"] = kwargs

        def eval(self, x, **kw):
            seen["x_max"] = float(np.max(x))
            seen["kw"] = kw
            if n_returns == 4:
                return mask, None, None, 30.0
            return mask, None, None

    models.CellposeModel = CellposeModel
    cellpose.models = models
    monkeypatch.setitem(sys.modules, "cellpose", cellpose)
    monkeypatch.setitem(sys.modules, "cellpose.models", models)
    return seen


def test_foreground_mask_cellpose_uses_the_model_labels(tmp_path, monkeypatch):
    labels = np.zeros((16, 16), np.int32)
    labels[4:8, 4:8] = 3
    seen = _install_fake_cellpose(monkeypatch, labels)
    st = _stitcher(tmp_path, outline_source="cellpose")
    img = np.full((16, 16), 255, np.uint8)

    m = st._foreground_mask(img)
    assert m.dtype == bool and m.sum() == 16
    assert m[5, 5] and not m[0, 0]
    # cpsam, not 'nuclei': Cellpose 4 ships one model, and model_type= is
    # accepted-and-ignored, so passing it named weights that never loaded.
    assert seen["init_kwargs"]["pretrained_model"] == "cpsam"
    assert "model_type" not in seen["init_kwargs"]
    # eval(channels=) was dropped in v4.0.1+; diameter still does something.
    assert "channels" not in seen["kw"]
    assert seen["kw"]["diameter"] is None
    assert seen["x_max"] == pytest.approx(1.0)     # scaled to 0..1 before eval


def test_foreground_mask_cellpose_accepts_a_four_value_eval(tmp_path, monkeypatch):
    """Cellpose 3 returned a fourth value, diams; the unpack must tolerate it."""
    labels = np.zeros((16, 16), np.int32)
    labels[4:8, 4:8] = 3
    _install_fake_cellpose(monkeypatch, labels, n_returns=4)
    st = _stitcher(tmp_path, outline_source="cellpose")
    assert st._foreground_mask(np.full((16, 16), 255, np.uint8)).sum() == 16


def test_outline_mask_cellpose_outlines_the_model_labels(tmp_path, monkeypatch):
    labels = np.zeros((40, 40), np.int32)
    labels[10:30, 10:30] = 1
    _install_fake_cellpose(monkeypatch, labels)
    st = _stitcher(tmp_path, outline_source="cellpose")
    edges = st._outline_mask(np.full((40, 40), 128, np.uint8))
    assert edges.sum() > 0 and not edges[20, 20]


def test_cellpose_model_is_built_once_per_stitcher(tmp_path, monkeypatch):
    """It used to be constructed inside the per-tile mask helpers.

    cpsam is a 1.2 GB checkpoint; rebuilding it per image in a well is the
    difference between a stitch that finishes and one that does not.
    """
    built = []
    labels = np.zeros((16, 16), np.int32)
    labels[4:8, 4:8] = 1
    seen = _install_fake_cellpose(monkeypatch, labels)
    real_models = sys.modules["cellpose.models"]
    original = real_models.CellposeModel

    class Counting(original):
        def __init__(self, **kwargs):
            built.append(kwargs)
            super().__init__(**kwargs)

    real_models.CellposeModel = Counting
    st = _stitcher(tmp_path, outline_source="cellpose")
    img = np.full((16, 16), 255, np.uint8)
    for _ in range(3):
        st._foreground_mask(img)
    st._outline_mask(img)
    assert len(built) == 1
    assert seen["init_kwargs"]["pretrained_model"] == "cpsam"


def test_masks_raise_a_clear_error_when_cellpose_is_missing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "cellpose", None)
    st = _stitcher(tmp_path, outline_source="cellpose")
    img = np.zeros((8, 8), np.uint8)
    with pytest.raises(RuntimeError, match="requires `cellpose` installed"):
        st._foreground_mask(img)
    with pytest.raises(RuntimeError, match="requires `cellpose` installed"):
        st._outline_mask(img)


# ===========================================================================
# normalisation helpers
# ===========================================================================

def test_norm01_maps_to_unit_range_linearly():
    out = spacrStitcher._norm01(np.array([[0.0, 5.0], [10.0, 20.0]]))
    assert out.dtype == np.float32
    assert out.tolist() == [[0.0, 0.25], [0.5, 1.0]]


def test_to_uint8_scales_min_to_zero_and_max_to_255():
    out = spacrStitcher._to_uint8(np.array([[100.0, 200.0], [300.0, 500.0]]))
    assert out.tolist() == [[0, 63], [127, 255]]


def test_edge_zncc_is_high_for_identical_and_low_for_unrelated_images():
    rng = np.random.default_rng(11)
    a = (rng.random((64, 64)) * 255).astype(np.float32)
    b = (rng.random((64, 64)) * 255).astype(np.float32)
    assert spacrStitcher._edge_zncc(a, a) == pytest.approx(1.0, abs=1e-3)
    assert abs(spacrStitcher._edge_zncc(a, b)) < 0.2


def test_edge_zncc_mask_restricts_the_comparison():
    rng = np.random.default_rng(21)
    a = (rng.random((64, 64)) * 255).astype(np.float32)
    b = a.copy()
    b[40:] = (rng.random((24, 64)) * 255).astype(np.float32)   # bottom differs
    top = np.zeros((64, 64), bool)
    top[:32] = True                       # stays clear of the Sobel footprint
    # identical where the mask looks, so the masked score is perfect...
    assert spacrStitcher._edge_zncc(a, b, mask=top) == pytest.approx(1.0, abs=1e-3)
    # ...while the unmasked score is dragged down by the differing region
    assert spacrStitcher._edge_zncc(a, b) < 0.9


def test_edge_zncc_returns_zero_for_a_tiny_mask():
    a = np.zeros((16, 16), np.float32)
    mask = np.zeros((16, 16), bool)
    mask[:2, :2] = True                   # 4 px < the 25 px minimum overlap
    assert spacrStitcher._edge_zncc(a, a, mask=mask) == 0.0


def test_affine_to_3x3_appends_the_homogeneous_row():
    M = np.array([[2.0, 0.5, 5.0], [-0.5, 2.0, 7.0]])
    A = spacrStitcher._affine_to_3x3(M)
    assert A.dtype == np.float32
    assert A[2].tolist() == [0.0, 0.0, 1.0]
    assert np.allclose(A[:2], M)


def test_invert_affine_round_trips_a_rotation_plus_translation():
    th = np.deg2rad(20.0)
    M = np.array([[np.cos(th), -np.sin(th), 11.0],
                  [np.sin(th), np.cos(th), -4.0]], np.float32)
    Mi = spacrStitcher._invert_affine(M)
    A = spacrStitcher._affine_to_3x3(M) @ spacrStitcher._affine_to_3x3(Mi)
    assert np.allclose(A, np.eye(3), atol=1e-4)
    pt = np.array([[[3.0, 9.0]]], np.float32)
    import cv2
    back = cv2.transform(cv2.transform(pt, M), Mi)
    assert np.allclose(back.ravel(), [3.0, 9.0], atol=1e-3)


def test_affine_from_row_builds_a_rotation_scale_translation():
    M = spacrStitcher._affine_from_row(
        {"dx_px_full": "3.5", "dy_px_full": "-2.5", "theta_deg": "90", "scale": "2"})
    assert np.allclose(M[:, :2], [[0.0, -2.0], [2.0, 0.0]], atol=1e-5)
    assert M[:, 2].tolist() == [3.5, -2.5]


def test_closest_rotation_removes_scale_and_shear():
    A = np.array([[3.0, 0.0], [0.0, 2.0]], np.float32)
    R = spacrStitcher._closest_rotation(A)
    assert np.allclose(R @ R.T, np.eye(2), atol=1e-5)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-5)


def test_closest_rotation_flips_a_reflection_to_a_proper_rotation():
    A = np.array([[1.0, 0.0], [0.0, -1.0]], np.float32)   # det = -1
    R = spacrStitcher._closest_rotation(A)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-5)


# ===========================================================================
# axis guessing / normalisation
# ===========================================================================

@pytest.mark.parametrize("shape,expected", [
    ((256, 256), "YX"),
    ((3, 256, 256), "CYX"),
    ((40, 256, 256), "ZYX"),
    ((256, 256, 4), "YXC"),
    ((256, 256, 40), "YXZ"),
    ((4, 4, 4), "CYX"),                     # nothing looks like an image plane
    ((3, 40, 256, 256), "CZYX"),
    ((40, 3, 256, 256), "ZCYX"),
    ((3, 4, 256, 256), "CZYX"),
    ((40, 50, 256, 256), "CZYX"),
    ((2, 3, 4, 5), "TCYX"),
    ((2, 3, 4, 256, 256), "TCZYX"),
    ((2, 40, 4, 256, 256), "TZCYX"),
    ((2, 3, 4, 5, 6), "TZCYX"),
    ((2, 3, 4, 5, 6, 7), "CZYX"),           # 6-D fallback
])
def test_guess_axes_from_shape(shape, expected):
    assert spacrStitcher._guess_axes_from_shape(shape) == expected


def test_normalize_to_yx_selects_channel_and_time(tmp_path):
    st = _stitcher(tmp_path, arr_axes="TCYX", t_index=1)
    arr = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    out = st._normalize_to_yx(arr, ch=2)
    assert out.shape == (4, 5)
    assert np.array_equal(out, arr[1, 2])


def test_normalize_to_yx_drops_surplus_axis_labels(tmp_path):
    st = _stitcher(tmp_path, arr_axes="TCYX")
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = st._normalize_to_yx(arr, ch=0)      # T and C dropped, YX kept
    assert np.array_equal(out, arr)


def test_normalize_to_yx_drops_from_the_left_when_only_zyx_remain(tmp_path):
    st = _stitcher(tmp_path, arr_axes="ZYX")
    arr = np.arange(20, dtype=np.float32).reshape(4, 5)
    # Z cannot be removed by the T/C pass, so the leftmost label is popped
    assert np.array_equal(st._normalize_to_yx(arr, ch=0), arr)


def test_normalize_to_yx_pads_missing_leading_axes(tmp_path):
    st = _stitcher(tmp_path, arr_axes="YX", t_index=1)
    arr = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    # a T axis is inserted at the front and sliced with t_index
    assert np.array_equal(st._normalize_to_yx(arr, ch=0), arr[1])

    st2 = _stitcher(tmp_path, arr_axes="YX", t_index=0, z_index=1)
    arr4 = np.arange(2 * 2 * 3 * 4, dtype=np.float32).reshape(2, 2, 3, 4)
    # each pad goes to the front, so T then C gives "CTYX"
    assert np.array_equal(st2._normalize_to_yx(arr4, ch=1), arr4[1, 0])

    st3 = _stitcher(tmp_path, arr_axes="YX", t_index=0, z_index=1)
    arr5 = np.arange(2 * 2 * 2 * 3 * 4, dtype=np.float32).reshape(2, 2, 2, 3, 4)
    # pads T, C then Z -> "ZCTYX"
    assert np.array_equal(st3._normalize_to_yx(arr5, ch=1), arr5[1, 1, 0])


def test_normalize_to_yx_mip_maximum_projects_z(tmp_path):
    st = _stitcher(tmp_path, arr_axes="ZYX", mip=True)
    rng = np.random.default_rng(5)
    arr = rng.random((6, 200, 200)).astype(np.float32)
    out = st._normalize_to_yx(arr, ch=0)
    assert np.allclose(out, arr.max(axis=0))


def test_normalize_to_yx_mip_projects_a_trailing_z_axis(tmp_path):
    st = _stitcher(tmp_path, arr_axes="YXZ", mip=True)
    rng = np.random.default_rng(6)
    arr = rng.random((200, 200, 5)).astype(np.float32)
    assert np.allclose(st._normalize_to_yx(arr, ch=0), arr.max(axis=2))


def test_normalize_to_yx_z_index_picks_one_slice(tmp_path):
    st = _stitcher(tmp_path, arr_axes="ZYX", mip=False, z_index=3)
    arr = np.arange(5 * 4 * 6, dtype=np.float32).reshape(5, 4, 6)
    assert np.array_equal(st._normalize_to_yx(arr, ch=0), arr[3])


def test_normalize_to_yx_uses_the_axes_hint_when_auto(tmp_path):
    st = _stitcher(tmp_path, arr_axes="AUTO")
    arr = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    assert np.array_equal(st._normalize_to_yx(arr, ch=2, axes_hint="cyx"), arr[2])


def test_normalize_to_yx_guesses_when_no_hint_is_given(tmp_path):
    st = _stitcher(tmp_path, arr_axes="AUTO")
    arr = np.arange(3 * 256 * 256, dtype=np.float32).reshape(3, 256, 256)
    assert np.array_equal(st._normalize_to_yx(arr, ch=1), arr[1])


def test_normalize_to_yx_squeezes_singletons(tmp_path):
    st = _stitcher(tmp_path, arr_axes="TCZYX", squeeze_singleton=True)
    # (T=1, C=1, Z=1, 3, 4) -> the leading singletons vanish, YX survives
    arr = np.arange(12, dtype=np.float32).reshape(1, 1, 1, 3, 4)
    assert st._normalize_to_yx(arr, ch=0).shape == (3, 4)
    # a degenerate 1-D result is rejected rather than silently returned
    flat = np.arange(4, dtype=np.float32).reshape(1, 1, 1, 1, 4)
    with pytest.raises(ValueError, match="Expected 2D YX"):
        st._normalize_to_yx(flat, ch=0)


def test_normalize_to_yx_keeps_singletons_when_squeeze_is_off(tmp_path):
    st = _stitcher(tmp_path, arr_axes="TCZYX", squeeze_singleton=False)
    arr = np.arange(12, dtype=np.float32).reshape(1, 1, 1, 3, 4)
    assert st._normalize_to_yx(arr, ch=0).shape == (3, 4)


def test_normalize_to_yx_drops_a_stray_small_axis(tmp_path):
    """Defensive guard: a duplicated spatial label leaves a 3-D result."""
    st = _stitcher(tmp_path, arr_axes="YXX")
    arr = np.arange(200 * 200 * 3, dtype=np.float32).reshape(200, 200, 3)
    out = st._normalize_to_yx(arr, ch=0)
    assert out.shape == (200, 200)
    assert np.array_equal(out, arr[:, :, 0])


def test_normalize_to_yx_raises_when_the_result_is_not_2d(tmp_path):
    st = _stitcher(tmp_path, arr_axes="YXX")
    arr = np.zeros((200, 200, 200), np.float32)
    with pytest.raises(ValueError, match="Expected 2D YX"):
        st._normalize_to_yx(arr, ch=0)


# ===========================================================================
# _read_plane / channel counting
# ===========================================================================

def test_read_plane_reads_a_plain_2d_tiff(tmp_path):
    st = _stitcher(tmp_path)
    arr = (np.arange(64, dtype=np.uint16).reshape(8, 8) * 7)
    p = str(tmp_path / "flat.tif")
    tifffile.imwrite(p, arr)
    out = st._read_plane(p)
    assert out.dtype == np.float32
    assert np.array_equal(out, arr.astype(np.float32))


def test_read_plane_selects_the_requested_channel_of_a_cyx_tiff(tmp_path):
    st = _stitcher(tmp_path)
    planes = [np.full((32, 32), v, np.uint16) for v in (11, 22, 33)]
    p = str(tmp_path / "cyx.tif")
    write_cyx(p, planes)
    for c, v in enumerate((11, 22, 33)):
        assert st._read_plane(p, ch=c).mean() == pytest.approx(v)


def test_read_plane_mip_projects_a_z_stack(tmp_path):
    st = _stitcher(tmp_path, mip=True)
    rng = np.random.default_rng(2)
    arr = (rng.random((5, 200, 200)) * 1000).astype(np.uint16)
    p = str(tmp_path / "z.tif")
    tifffile.imwrite(p, arr, metadata={"axes": "ZYX"})
    assert np.array_equal(st._read_plane(p), arr.max(axis=0).astype(np.float32))


def test_read_plane_z_index_without_mip(tmp_path):
    st = _stitcher(tmp_path, mip=False, z_index=2)
    arr = np.stack([np.full((200, 200), i, np.uint16) for i in range(4)])
    p = str(tmp_path / "z2.tif")
    tifffile.imwrite(p, arr, metadata={"axes": "ZYX"})
    assert st._read_plane(p).mean() == pytest.approx(2.0)


def test_read_plane_on_a_metadata_less_stack_returns_the_first_plane(tmp_path):
    """tifffile labels a bare 3-D write 'SYX'; only Y/X survive filtering."""
    st = _stitcher(tmp_path)
    arr = np.stack([np.full((200, 200), i, np.uint16) for i in range(3)])
    p = str(tmp_path / "bare.tif")
    tifffile.imwrite(p, arr)
    assert st._read_plane(p, ch=2).mean() == pytest.approx(0.0)
    # ...and the module consistently reports it as single-channel
    assert spacrStitcher._get_channel_count_tif(p) == 1


def test_get_channel_count_tif_reads_the_axes_metadata(tmp_path):
    p = str(tmp_path / "c4.tif")
    write_cyx(p, [np.zeros((16, 16), np.uint16)] * 4)
    assert spacrStitcher._get_channel_count_tif(p) == 4

    p2 = str(tmp_path / "flat.tif")
    tifffile.imwrite(p2, np.zeros((16, 16), np.uint16))
    assert spacrStitcher._get_channel_count_tif(p2) == 1


def test_read_all_channels_cyx_stacks_every_channel(tmp_path):
    st = _stitcher(tmp_path)
    planes = [np.full((16, 16), v, np.uint16) for v in (5, 6)]
    p = str(tmp_path / "two.tif")
    write_cyx(p, planes)
    out = st._read_all_channels_cyx(p)
    assert out.shape == (2, 16, 16) and out.dtype == np.float32
    assert out[0].mean() == pytest.approx(5) and out[1].mean() == pytest.approx(6)


# ===========================================================================
# metadata parsing / grouping
# ===========================================================================

def test_parse_meta_extracts_all_four_fields(tmp_path):
    st = _stitcher(tmp_path)
    meta = st._parse_meta("/data/10X_c2_B12_r01f03_Site-7.tif")
    assert meta == {"well": "B12", "site": 7, "chan": 2, "mag": "10X"}


def test_parse_meta_falls_back_when_the_regex_misses(tmp_path):
    st = _stitcher(tmp_path)
    meta = st._parse_meta("/data/scan_D4_Site_9.png")     # not a .tif -> regex misses
    assert meta["well"] == "D4" and meta["site"] == 9
    assert meta["chan"] is None and meta["mag"] is None


def test_parse_meta_fallback_well_pattern_is_case_insensitive(tmp_path):
    """Documents a known wart: the lenient fallback accepts a lowercase
    letter, so a channel token such as ``_c3_`` is read as well 'C3'.
    See the report accompanying these tests."""
    st = _stitcher(tmp_path)
    meta = st._parse_meta("/data/scan_c3_Site_2.png")
    assert meta["chan"] == 3 and meta["site"] == 2
    assert meta["well"] == "C3"           # not a real well - the token was '_c3_'


def test_parse_meta_returns_all_none_for_an_opaque_name(tmp_path):
    st = _stitcher(tmp_path)
    assert st._parse_meta("/data/scan.tif") == \
        {"well": None, "site": None, "chan": None, "mag": None}


def test_set_meta_regex_accepts_string_and_compiled_patterns(tmp_path):
    import re
    st = _stitcher(tmp_path)
    st.set_meta_regex(r"w(?P<well>[A-H]\d+)_s(?P<site>\d+)\.tif$")
    assert st._parse_meta("/x/wA3_s12.tif")["site"] == 12
    st.set_meta_regex(re.compile(r"(?P<well>[A-H]\d+)__(?P<site>\d+)"))
    assert st._parse_meta("/x/B2__4.tif")["well"] == "B2"


def test_group_by_well_buckets_and_sorts_by_site(tmp_path):
    st = _stitcher(tmp_path)
    paths = ["/d/10X_c1_A1_Site-3.tif", "/d/10X_c1_A1_Site-1.tif",
             "/d/10X_c1_B2_Site-2.tif", "/d/unparseable.tif"]
    groups = st._group_by_well(paths)
    assert set(groups) == {"A1", "B2", "UNK"}
    assert [os.path.basename(p) for p in groups["A1"]] == \
        ["10X_c1_A1_Site-1.tif", "10X_c1_A1_Site-3.tif"]
    assert groups["UNK"] == ["/d/unparseable.tif"]


def test_pairs_by_site_window_only_links_neighbouring_sites(tmp_path):
    st = _stitcher(tmp_path)
    files = [f"/d/10X_c1_A1_Site-{i}.tif" for i in (1, 2, 3, 4)]
    pairs = st._pairs_by_site_window(files, max_site_gap=1)
    assert len(pairs) == 3
    sites = {(st._parse_meta(a)["site"], st._parse_meta(b)["site"]) for a, b in pairs}
    assert sites == {(1, 2), (2, 3), (3, 4)}

    pairs2 = st._pairs_by_site_window(files, max_site_gap=2)
    assert len(pairs2) == 5     # +(1,3) and (2,4)


def test_pairs_by_site_window_falls_back_to_index_when_site_is_missing(tmp_path):
    st = _stitcher(tmp_path)
    files = ["/d/a.tif", "/d/b.tif", "/d/c.tif"]      # no parseable site
    pairs = st._pairs_by_site_window(files, max_site_gap=1)
    assert pairs == [("/d/a.tif", "/d/b.tif"), ("/d/b.tif", "/d/c.tif")]


def test_list_tifs_respects_extensions_and_recursion(tmp_path):
    root = tmp_path / "imgs"
    (root / "sub").mkdir(parents=True)
    for rel in ("a.tif", "b.TIFF", "c.png", "sub/d.tif"):
        (root / rel).write_bytes(b"x")
    flat = spacrStitcher._list_tifs(str(root), False, (".tif", ".tiff"))
    assert sorted(os.path.basename(p) for p in flat) == ["a.tif", "b.TIFF"]
    deep = spacrStitcher._list_tifs(str(root), True, (".tif",))
    assert sorted(os.path.basename(p) for p in deep) == ["a.tif", "d.tif"]


# ===========================================================================
# feature extraction / caching
# ===========================================================================

def _tiles(tmp_path, n=2, tile=384, step=150):
    canvas = blob_canvas(seed=1)
    d = tmp_path / "tiles"
    d.mkdir(exist_ok=True)
    paths = []
    for i in range(n):
        p = str(d / f"10X_c1_A1_Site-{i + 1}.tif")
        tifffile.imwrite(p, crop(canvas, 100, 100 + i * step, tile))
        paths.append(p)
    return paths


def test_compute_features_one_downsamples_and_finds_keypoints(tmp_path):
    st = _stitcher(tmp_path, downsample=0.5)
    p = _tiles(tmp_path, 1)[0]
    f = st._compute_features_one(p, 0)
    assert (int(f["H"]), int(f["W"])) == (384, 384)
    assert (int(f["Hds"]), int(f["Wds"])) == (192, 192)
    assert f["ds8"].shape == (192, 192) and f["ds8"].dtype == np.uint8
    assert f["pts"].shape[0] >= 20 and f["pts"].shape[1] == 2
    assert f["desc"].shape[0] == f["pts"].shape[0]
    assert f["pts"].max() <= 192


def test_compute_features_one_treats_non_positive_downsample_as_full_res(tmp_path):
    st = _stitcher(tmp_path, downsample=0.0)
    p = _tiles(tmp_path, 1)[0]
    f = st._compute_features_one(p, 0)
    assert (int(f["Hds"]), int(f["Wds"])) == (384, 384)


def test_compute_features_one_caps_keypoints(tmp_path):
    st = _stitcher(tmp_path, downsample=0.5, max_keypoints=15)
    f = st._compute_features_one(_tiles(tmp_path, 1)[0], 0)
    assert f["pts"].shape[0] == 15 and f["desc"].shape[0] == 15


def test_detect_and_describe_returns_empty_arrays_on_a_blank_image(tmp_path):
    st = _stitcher(tmp_path)
    pts, desc = st._detect_and_describe(np.zeros((64, 64), np.uint8))
    assert pts.shape == (0, 2) and pts.dtype == np.float32
    assert desc.shape == (0, 32) and desc.dtype == np.uint8


@pytest.mark.skipif(not hasattr(__import__("cv2"), "SIFT_create"),
                    reason="opencv build without SIFT")
def test_detect_and_describe_empty_descriptor_width_matches_sift(tmp_path):
    st = _stitcher(tmp_path, detector="SIFT")
    pts, desc = st._detect_and_describe(np.zeros((64, 64), np.uint8))
    assert desc.shape == (0, 128) and desc.dtype == np.float32


def test_prepare_features_disk_mode_writes_one_npz_per_image(tmp_path, capsys):
    cache = str(tmp_path / "cache")
    st = spacrStitcher(outdir=str(tmp_path / "o"), downsample=0.5, save_qc=False,
                       feature_cache_mode="disk", feature_cache_dir=cache, verbose=True)
    paths = _tiles(tmp_path, 3)
    st.prepare_features(paths, 0, num_workers=2)
    assert len([f for f in os.listdir(cache) if f.endswith(".npz")]) == 3
    assert "computing features for 3 images" in capsys.readouterr().out

    # second call recognises everything is already on disk
    st.prepare_features(paths, 0, num_workers=2)
    assert "nothing to compute" in capsys.readouterr().out


def test_prepare_features_ram_mode_fills_the_in_memory_cache(tmp_path):
    st = _stitcher(tmp_path, downsample=0.5, feature_cache_mode="ram")
    paths = _tiles(tmp_path, 2)
    st.prepare_features(paths, 0)
    assert sorted(st._feat_cache) == sorted(paths)
    assert st._feat_cache[paths[0]]["pts"].shape[0] > 0
    # a second call is a no-op that leaves the cached objects identical
    first = st._feat_cache[paths[0]]
    st.prepare_features(paths, 0)
    assert st._feat_cache[paths[0]] is first


def test_prepare_features_defaults_to_the_configured_worker_count(tmp_path):
    st = _stitcher(tmp_path, downsample=0.5, n_workers_features=1)
    paths = _tiles(tmp_path, 2)
    st.prepare_features(paths, 0)        # num_workers=None -> self.n_workers_features
    assert len(st._feat_cache) == 2


def test_prepare_features_reports_and_skips_a_broken_image(tmp_path, capsys):
    """One unreadable tile must be reported loudly and skipped, not abort the
    whole batch and not be cached."""
    st = _stitcher(tmp_path, downsample=0.5)
    good = _tiles(tmp_path, 1)[0]
    bad = str(tmp_path / "broken.tif")
    with open(bad, "wb") as fh:
        fh.write(b"definitely not a tiff")

    st.prepare_features([good, bad], 0, num_workers=2)

    printed = capsys.readouterr().out
    assert "WARNING" in printed and "broken.tif" in printed
    assert bad not in st._feat_cache             # not cached
    assert st._feat_cache[good]["pts"].shape[0] > 0   # the good one still went through


def test_prepare_features_disk_mode_also_survives_a_broken_image(tmp_path, capsys):
    cache = str(tmp_path / "cache")
    st = spacrStitcher(outdir=str(tmp_path / "o"), downsample=0.5, save_qc=False,
                       feature_cache_mode="disk", feature_cache_dir=cache)
    good = _tiles(tmp_path, 1)[0]
    bad = str(tmp_path / "broken.tif")
    with open(bad, "wb") as fh:
        fh.write(b"nope")
    st.prepare_features([good, bad], 0, num_workers=2)
    assert "WARNING" in capsys.readouterr().out
    assert len([f for f in os.listdir(cache) if f.endswith(".npz")]) == 1
    assert st._store.get(bad) is None


def test_get_features_computes_once_and_then_hits_the_cache(tmp_path):
    st = _stitcher(tmp_path, downsample=0.5, feature_cache_mode="ram")
    p = _tiles(tmp_path, 1)[0]
    f1 = st._get_features(p, 0)
    f2 = st._get_features(p, 0)
    assert f1 is f2                       # served from RAM the second time
    assert f1["pts"].shape[0] > 0


def test_get_features_disk_mode_persists_across_stitcher_instances(tmp_path):
    cache = str(tmp_path / "cache")
    p = _tiles(tmp_path, 1)[0]
    st = spacrStitcher(outdir=str(tmp_path / "o"), downsample=0.5, save_qc=False,
                       feature_cache_mode="disk", feature_cache_dir=cache)
    f1 = st._get_features(p, 0)
    st2 = spacrStitcher(outdir=str(tmp_path / "o"), downsample=0.5, save_qc=False,
                        feature_cache_mode="disk", feature_cache_dir=cache)
    f2 = st2._get_features(p, 0)          # never computed by st2, read from disk
    assert np.array_equal(f1["pts"], f2["pts"])
    assert np.array_equal(f1["ds8"], f2["ds8"])


# ===========================================================================
# matching / RANSAC
# ===========================================================================

def test_match_returns_empty_when_either_side_has_too_few_points(tmp_path):
    st = _stitcher(tmp_path)
    small = {"pts": np.zeros((2, 2), np.float32), "desc": np.zeros((2, 32), np.uint8)}
    big = {"pts": np.zeros((9, 2), np.float32), "desc": np.zeros((9, 32), np.uint8)}
    a, b = st._match(small, big)
    assert a.shape == (0, 2) and b.shape == (0, 2)
    a, b = st._match(big, small)
    assert a.shape == (0, 2) and b.shape == (0, 2)


def test_match_pairs_identical_orb_descriptors(tmp_path):
    st = _stitcher(tmp_path)
    rng = np.random.default_rng(4)
    desc = rng.integers(0, 256, (6, 32), dtype=np.uint8)
    pts = rng.random((6, 2)).astype(np.float32) * 100
    perm = np.array([3, 1, 0, 5, 4, 2])
    fA = {"pts": pts, "desc": desc}
    fB = {"pts": pts[perm], "desc": desc[perm]}
    pA, pB = st._match(fA, fB)
    assert pA.shape[0] == 6
    # every matched pair points at the same underlying keypoint
    assert np.allclose(pA, pB, atol=1e-5)


@pytest.mark.skipif(not hasattr(__import__("cv2"), "SIFT_create"),
                    reason="opencv build without SIFT")
def test_match_sift_ratio_test_rejects_ambiguous_descriptors(tmp_path):
    st = _stitcher(tmp_path, detector="SIFT")
    desc = np.ones((6, 128), np.float32)      # every distance identical
    pts = np.arange(12, dtype=np.float32).reshape(6, 2)
    pA, pB = st._match({"pts": pts, "desc": desc}, {"pts": pts, "desc": desc})
    assert pA.shape == (0, 2) and pB.shape == (0, 2)


@pytest.mark.skipif(not hasattr(__import__("cv2"), "SIFT_create"),
                    reason="opencv build without SIFT")
def test_match_sift_keeps_discriminative_descriptors(tmp_path):
    st = _stitcher(tmp_path, detector="SIFT")
    rng = np.random.default_rng(8)
    desc = (rng.random((8, 128)).astype(np.float32) * 100)
    pts = rng.random((8, 2)).astype(np.float32) * 50
    pA, pB = st._match({"pts": pts, "desc": desc}, {"pts": pts, "desc": desc})
    assert pA.shape[0] >= 4
    assert np.allclose(pA, pB, atol=1e-5)


def test_affine_from_pts_recovers_a_known_translation():
    rng = np.random.default_rng(9)
    ptsB = (rng.random((40, 2)).astype(np.float32) * 200)
    ptsA = ptsB + np.array([12.5, -7.25], np.float32)
    M, mask, ratio = spacrStitcher._affine_from_pts(ptsA, ptsB, 3.0)
    assert np.allclose(M[:, :2], np.eye(2), atol=1e-3)
    assert M[0, 2] == pytest.approx(12.5, abs=1e-2)
    assert M[1, 2] == pytest.approx(-7.25, abs=1e-2)
    assert ratio == pytest.approx(1.0)
    assert mask.dtype == bool and mask.all()


def test_affine_from_pts_recovers_rotation_and_scale():
    rng = np.random.default_rng(10)
    ptsB = (rng.random((60, 2)).astype(np.float32) * 200)
    th = np.deg2rad(15.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], np.float32) * 1.5
    ptsA = (ptsB @ R.T) + np.array([4.0, 9.0], np.float32)
    M, mask, ratio = spacrStitcher._affine_from_pts(ptsA, ptsB, 2.0)
    assert np.allclose(M[:, :2], R, atol=1e-2)
    assert ratio > 0.9


def test_affine_from_pts_ignores_outliers():
    rng = np.random.default_rng(12)
    ptsB = (rng.random((60, 2)).astype(np.float32) * 300)
    ptsA = ptsB + np.array([20.0, 0.0], np.float32)
    ptsA[:12] = rng.random((12, 2)).astype(np.float32) * 300     # 20% garbage
    M, mask, ratio = spacrStitcher._affine_from_pts(ptsA, ptsB, 2.0)
    assert M[0, 2] == pytest.approx(20.0, abs=0.5)
    assert 0.7 <= ratio <= 0.85
    assert not mask[:12].all()


def test_affine_from_pts_needs_at_least_four_points():
    pts = np.zeros((3, 2), np.float32)
    assert spacrStitcher._affine_from_pts(pts, pts, 3.0) == (None, None, 0.0)


def test_affine_from_pts_returns_none_for_degenerate_points():
    pts = np.zeros((8, 2), np.float32)       # all points coincide
    M, mask, ratio = spacrStitcher._affine_from_pts(pts, pts, 3.0)
    assert M is None and mask is None and ratio == 0.0


# ===========================================================================
# elbow threshold / plotting
# ===========================================================================

def test_auto_elbow_threshold_edge_cases():
    assert spacrStitcher._auto_elbow_threshold([]) == 0.0
    assert spacrStitcher._auto_elbow_threshold([0.42]) == 0.42
    assert spacrStitcher._auto_elbow_threshold([0.1, 0.9]) == 0.9     # n==2 -> s[1]


def test_auto_elbow_threshold_finds_the_knee_of_an_l_shaped_curve():
    scores = [0.01, 0.02, 0.03, 0.04, 0.05, 0.8, 0.82, 0.85]
    thr = spacrStitcher._auto_elbow_threshold(scores)
    assert thr == pytest.approx(0.05)     # last point before the jump


def test_plot_sorted_scores_writes_a_png(tmp_path):
    st = _stitcher(tmp_path)
    out = str(tmp_path / "scores.png")
    st._plot_sorted_scores([0.1, 0.5, 0.9], 0.5, out)
    assert os.path.getsize(out) > 0
