"""The CuPy branches of the OPS primitives, run against a NumPy stand-in.

CI has no CUDA, so the real CuPy never loads there and these branches went
unmeasured. The stand-in below exposes exactly the calls the backend makes
and nothing else, so a branch that reached for a CuPy call this file does
not model fails here rather than being waved through.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from scipy.ndimage import maximum_filter as scipy_filter

import spacr.ops_accel as accel


@pytest.fixture
def fake_cupy(monkeypatch):
    calls = []
    namespace = types.SimpleNamespace(
        asarray=lambda value: (calls.append("asarray"), np.asarray(value))[1],
        asnumpy=lambda value: (calls.append("asnumpy"), np.asarray(value))[1],
        argmin=np.argmin,
        take_along_axis=np.take_along_axis,
        linalg=types.SimpleNamespace(norm=np.linalg.norm),
    )
    ndimage = types.ModuleType("cupyx.scipy.ndimage")
    ndimage.maximum_filter = scipy_filter
    for name, module in (("cupyx", types.ModuleType("cupyx")),
                         ("cupyx.scipy", types.ModuleType("cupyx.scipy")),
                         ("cupyx.scipy.ndimage", ndimage)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(accel, "_cupy", lambda gpu=True: namespace)
    return calls


@pytest.fixture
def no_cupy(monkeypatch):
    monkeypatch.setattr(accel, "_cupy", lambda gpu=True: None)


def test_the_cupy_windowed_maximum_matches_the_cpu_answer(fake_cupy):
    field = np.random.default_rng(2).random((31, 23)).astype(np.float32)
    got = accel.maximum_filter(field, 5, backend="cupy")
    assert np.array_equal(got, accel.maximum_filter(field, 5, backend="numpy"))
    assert "asarray" in fake_cupy and "asnumpy" in fake_cupy


def test_the_cupy_matmul_matches_the_cpu_answer(fake_cupy):
    rng = np.random.default_rng(3)
    left = rng.random((8, 4)).astype(np.float32)
    right = rng.random((4, 5)).astype(np.float32)
    assert np.allclose(accel.matmul(left, right, backend="cupy"), left @ right)
    assert "asnumpy" in fake_cupy


def test_the_cupy_nearest_neighbour_is_chunked_and_matches_the_cpu(fake_cupy):
    rng = np.random.default_rng(4)
    source = rng.random((25, 2)).astype(np.float32) * 50
    target = rng.random((11, 2)).astype(np.float32) * 50
    want = accel.nearest_neighbours(source, target, backend="numpy")
    got = accel.nearest_neighbours(source, target, backend="cupy", chunk=7)
    assert np.array_equal(got[0], want[0])
    assert np.allclose(got[1], want[1], atol=1e-4)
    assert fake_cupy.count("asnumpy") >= 2 * 4


@pytest.mark.parametrize("call", [
    lambda: accel.maximum_filter(np.zeros((9, 9)), 3, backend="cupy"),
    lambda: accel.matmul(np.eye(2), np.eye(2), backend="cupy"),
    lambda: accel.nearest_neighbours(np.zeros((3, 2)), np.ones((2, 2)),
                                     backend="cupy"),
])
def test_a_named_cupy_that_is_not_usable_is_raised_not_swallowed(no_cupy, call):
    with pytest.raises(RuntimeError, match="cupy is not usable"):
        call()
