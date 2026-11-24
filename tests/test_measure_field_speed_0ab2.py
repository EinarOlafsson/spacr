"""Item 0ab, second part: the compiled per-object kernels of the shared measurement core.

Each kernel in :mod:`spacr.measure` that replaced a per-object library call
or a chain of whole-field array passes is checked here against a plain
reference: the per-object NumPy and SciPy calls it replaced, the bincounts
it fuses (to the bit, where it adds in the same order), or a brute-force
definition. Values that are summed in another order must agree within
float32 rounding (the maintainer's rule for development changes), with
NaN in the same places and the dtype each column had.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr import measure as M


def _blobs(seed, shape=(90, 100), n=14, dtype=np.float32):
    from skimage.segmentation import expand_labels

    rng = np.random.default_rng(seed)
    labels = np.zeros(shape, np.int32)
    for k in range(1, n + 1):
        labels[rng.integers(4, shape[0] - 4), rng.integers(4, shape[1] - 4)] = k
    labels = expand_labels(labels, 6)
    image = (rng.random(shape + (3,)) * 100).astype(dtype)
    return labels, image


def _close(got, want, *, rel=1e-6, sd=1e-7):
    got = np.asarray(got, dtype=float)
    want = np.asarray(want, dtype=float)
    assert np.array_equal(np.isnan(got), np.isnan(want))
    keep = ~np.isnan(want)
    if keep.any():
        spread = np.std(want[keep])
        assert np.all(np.abs(got[keep] - want[keep])
                      <= rel * np.abs(want[keep]) + sd * spread + 1e-300)


def test_the_texture_kernels_give_the_same_table_twice_and_on_any_thread_count(monkeypatch):
    labels, image = _blobs(7)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    one = M.measure_texture(labels, image)
    again = M.measure_texture(labels, image)
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    three = M.measure_texture(labels, image)
    for other in (again, three):
        assert np.array_equal(one.to_numpy(), other.to_numpy(), equal_nan=True)
