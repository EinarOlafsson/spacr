"""Coverage-fill batch 5 for spacr.utils stack-filter / regex helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")

from spacr import utils as U


def _stack(size=24):
    """3-channel stack: cell(0), nucleus(1), pathogen(2)."""
    cell = np.zeros((size, size), dtype=np.int32)
    cell[2:10, 2:10] = 1       # cell 1 (infected)
    cell[14:22, 14:22] = 2     # cell 2 (uninfected)
    nucleus = np.zeros((size, size), dtype=np.int32)
    nucleus[3:6, 3:6] = 1
    nucleus[15:18, 15:18] = 2
    pathogen = np.zeros((size, size), dtype=np.int32)
    pathogen[6:9, 6:9] = 1     # only inside cell 1
    return np.stack([cell, nucleus, pathogen], axis=-1)


def test_remove_noninfected():
    stack = _stack()
    out = U._remove_noninfected(stack.copy(), cell_dim=0, nucleus_dim=1,
                                pathogen_dim=2)
    cell_out = out[:, :, 0]
    # cell 2 (no pathogen) removed, cell 1 kept
    assert 1 in np.unique(cell_out) and 2 not in np.unique(cell_out)


def test_remove_noninfected_none_dims():
    stack = _stack()
    out = U._remove_noninfected(stack.copy(), cell_dim=None, nucleus_dim=None,
                               pathogen_dim=None)
    assert out.shape == stack.shape


def test_remove_outside_objects():
    stack = _stack()
    # add a pathogen with no overlapping cell
    stack[20:22, 0:2, 2] = 5
    out = U._remove_outside_objects(stack.copy(), cell_dim=0, nucleus_dim=1,
                                    pathogen_dim=2)
    assert 5 not in np.unique(out[:, :, 2])


def test_remove_outside_objects_no_cell_dim():
    stack = _stack()
    out = U._remove_outside_objects(stack.copy(), cell_dim=None,
                                    nucleus_dim=1, pathogen_dim=2)
    assert np.array_equal(out, stack)


def test_remove_multiobject_cells():
    # 4-channel stack: mask(0), cell(1), nucleus(2), pathogen(3)
    size = 24
    mask = np.zeros((size, size), dtype=np.int32); mask[2:12, 2:12] = 1
    cell = mask.copy()
    nucleus = np.zeros((size, size), dtype=np.int32); nucleus[3:5, 3:5] = 1
    pathogen = np.zeros((size, size), dtype=np.int32)
    pathogen[3:5, 3:5] = 1
    pathogen[8:10, 8:10] = 2   # two objects in the one cell → removed
    stack = np.stack([mask, cell, nucleus, pathogen], axis=-1)
    out = U._remove_multiobject_cells(
        stack.copy(), mask_dim=0, cell_dim=1, nucleus_dim=2,
        pathogen_dim=3, object_dim=3)
    assert 1 not in np.unique(out[:, :, 1])


def test_merge_touching_objects():
    m = np.zeros((16, 16), dtype=np.int32)
    m[4:12, 4:9] = 1
    m[4:12, 9:12] = 2   # adjacent to label 1
    out = U.merge_touching_objects(m.copy(), threshold=0.1)
    assert out is not None


# ---------------------------------------------------------------------------
# _object_filter
# ---------------------------------------------------------------------------

def test_object_filter():
    df = pd.DataFrame({
        "cell_area": [50, 200, 500, 1000],
        "cell_channel_1_mean_intensity": [10, 100, 1000, 5000],
    })
    out = U._object_filter(
        df, object_type="cell", size_range=[100, 800],
        intensity_range=[50, 2000], mask_chans=[0, 1], mask_chan=1)
    assert (out["cell_area"] > 100).all() and (out["cell_area"] < 800).all()


def test_object_filter_none_ranges():
    df = pd.DataFrame({"cell_area": [1, 2, 3]})
    out = U._object_filter(
        df, object_type="cell", size_range=None,
        intensity_range=None, mask_chans=[0], mask_chan=0)
    assert len(out) == 3


# ---------------------------------------------------------------------------
# _get_regex
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mtype", ["cellvoyager", "cq1", "auto"])
def test_get_regex_builtins(mtype):
    rx = U._get_regex(mtype, img_format="tif")
    assert "chanID" in rx


def test_get_regex_custom():
    rx = U._get_regex("custom", img_format="tif",
                      custom_regex=r"(?P<plateID>.*)")
    assert "plateID" in rx
