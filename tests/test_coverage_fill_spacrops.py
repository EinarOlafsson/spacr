"""Coverage-fill for spacr.spacrops stitcher static/pure helpers."""
from __future__ import annotations

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")

from spacr.spacrops import spacrStitcher as S, _DiskFeatureStore


# ---------------------------------------------------------------------------
# normalization / dtype helpers
# ---------------------------------------------------------------------------

def test_norm01():
    out = S._norm01(np.array([[10.0, 20.0], [30.0, 40.0]]))
    assert out.min() == 0.0 and out.max() == 1.0
    # flat image → zeros
    flat = S._norm01(np.full((4, 4), 5.0))
    assert np.all(flat == 0)


def test_to_uint8():
    out = S._to_uint8(np.array([[0.0, 1.0], [2.0, 4.0]]))
    assert out.dtype == np.uint8 and out.max() == 255
    assert np.all(S._to_uint8(np.full((3, 3), 2.0)) == 0)   # flat → zeros


# ---------------------------------------------------------------------------
# affine helpers
# ---------------------------------------------------------------------------

def test_affine_to_3x3():
    M = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 7.0]])
    A = S._affine_to_3x3(M)
    assert A.shape == (3, 3) and A[2, 2] == 1.0 and A[0, 2] == 5.0


def test_invert_affine():
    # translation-only affine → inverse negates translation
    M = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 7.0]], dtype=np.float32)
    Mi = S._invert_affine(M)
    assert Mi[0, 2] == pytest.approx(-5.0, abs=1e-3)
    assert Mi[1, 2] == pytest.approx(-7.0, abs=1e-3)


def test_affine_from_row():
    row = {"dx_px_full": 10.0, "dy_px_full": -4.0,
           "theta_deg": 0.0, "scale": 1.0}
    M = S._affine_from_row(row)
    assert M.shape == (2, 3)
    assert M[0, 2] == 10.0 and M[1, 2] == -4.0
    assert M[0, 0] == pytest.approx(1.0)


def test_closest_rotation():
    # a near-identity matrix → closest rotation is ~identity
    A = np.array([[1.01, 0.0], [0.0, 0.99]], dtype=np.float32)
    R = S._closest_rotation(A)
    assert R.shape == (2, 2)
    # rotation matrices have determinant ~1
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# axis-guessing helpers
# ---------------------------------------------------------------------------

def test_is_large_dim():
    assert S._is_large_dim(256) is True and S._is_large_dim(3) is False


@pytest.mark.parametrize("shape,expected", [
    ((256, 256), "YX"),
    ((3, 256, 256), "CYX"),      # small leading axis + big YX
    ((256, 256, 3), "YXC"),      # big YX + small trailing
    ((20, 256, 256), "ZYX"),     # large leading axis
])
def test_guess_axes_from_shape(shape, expected):
    assert S._guess_axes_from_shape(shape) == expected


# ---------------------------------------------------------------------------
# _direction_bin
# ---------------------------------------------------------------------------

def test_direction_bin():
    assert S._direction_bin(10.0, 0.0) == "R"
    assert S._direction_bin(0.0, 10.0) == "U"
    assert S._direction_bin(-10.0, 0.0) == "L"
    assert S._direction_bin(0.0, -10.0) == "D"
    # too diagonal → None
    assert S._direction_bin(10.0, 10.0) is None


# ---------------------------------------------------------------------------
# _auto_elbow_threshold
# ---------------------------------------------------------------------------

def test_auto_elbow_threshold():
    assert S._auto_elbow_threshold([]) == 0.0
    assert S._auto_elbow_threshold([0.5]) == 0.5
    # an L-shaped score set has a clear knee
    scores = [0.1, 0.12, 0.13, 0.15, 0.9, 0.95]
    thr = S._auto_elbow_threshold(scores)
    assert 0.1 <= thr <= 0.95


# ---------------------------------------------------------------------------
# _edge_zncc
# ---------------------------------------------------------------------------

def test_edge_zncc_identical():
    rng = np.random.default_rng(0)
    a = (rng.random((32, 32)) * 255).astype(np.float32)
    # identical images → high positive correlation
    z = S._edge_zncc(a, a.copy())
    assert z > 0.9


def test_edge_zncc_masked_insufficient():
    a = np.zeros((16, 16), dtype=np.float32)
    mask = np.zeros((16, 16), dtype=bool)
    mask[0, 0] = True   # <25 pixels → 0.0
    assert S._edge_zncc(a, a, mask=mask) == 0.0


# ---------------------------------------------------------------------------
# _DiskFeatureStore
# ---------------------------------------------------------------------------

def test_disk_feature_store_key():
    k = _DiskFeatureStore._key_for_path("/some/path/img.tif")
    assert isinstance(k, str) and len(k) == 16


def test_disk_feature_store_put_get(tmp_path):
    store = _DiskFeatureStore(str(tmp_path), max_ram_items=4)
    feat = {
        "ds8": np.zeros((8, 8), dtype=np.uint8),
        "pts": np.array([[1.0, 2.0]], dtype=np.float32),
        "desc": np.array([[0.1, 0.2]], dtype=np.float32),
        "Hds": 8, "Wds": 8, "H": 64, "W": 64,
    }
    store.put("/x/img.tif", feat)
    got = store.get("/x/img.tif")
    assert got is not None and "pts" in got
    # a never-stored path → None
    assert store.get("/x/missing.tif") is None
