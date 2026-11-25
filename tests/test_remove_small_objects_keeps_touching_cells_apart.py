"""Make Masks' remove-small drops by size and keeps every instance label."""
import numpy as np

from spacr.qt import mask_engine as engine


def test_touching_cells_keep_their_own_labels():
    mask = np.zeros((12, 12), dtype=np.uint16)
    mask[1:7, 1:4] = 5
    mask[1:7, 4:7] = 9
    mask[9:11, 9:11] = 3
    out = engine.remove_small_objects(mask, min_area=10)
    assert sorted(np.unique(out).tolist()) == [0, 5, 9]
    assert (out[1:7, 1:4] == 5).all() and (out[1:7, 4:7] == 9).all()
    assert not (out == 3).any()


def test_a_binary_mask_still_drops_small_components():
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[0:2, 0:2] = 255
    mask[10:20, 10:20] = 255
    assert int((engine.remove_small_objects(mask, 10) > 0).sum()) == 100
