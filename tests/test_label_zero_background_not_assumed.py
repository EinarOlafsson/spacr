"""Regression tests: label enumeration must not assume the first unique value is 0.

``np.unique(mask)[1:]`` silently drops the smallest real label when a mask has
no background pixel (every pixel labelled). The fix excludes 0 explicitly. Each
test uses a fully labelled mask (no 0) and asserts every label is handled, and
a mask with background to show the result is unchanged.
"""
import matplotlib

matplotlib.use("Agg")

import numpy as np

from spacr import measure, plot, timelapse, utils


def _full(shape=(4, 4)):
    mask = np.ones(shape, dtype=int)
    mask[:, 2:] = 2
    return mask


def _with_bg():
    mask = np.zeros((6, 6), dtype=int)
    mask[0:2, 0:2] = 1
    mask[3:5, 3:5] = 2
    return mask


def test_link_by_iou_keeps_smallest_label_without_background():
    assert sorted(timelapse.link_by_iou(_full(), _full())) == [(1, 1), (2, 2)]


def test_link_by_iou_unchanged_with_background():
    assert sorted(timelapse.link_by_iou(_with_bg(), _with_bg())) == [(1, 1), (2, 2)]


def test_track_by_iou_tracks_every_label_without_background():
    masks = np.stack([_full(), _full()])
    df = timelapse._track_by_iou(masks)
    assert set(df[df.frame == 0].original_label) == {1, 2}
    assert set(df[df.frame == 1].original_label) == {1, 2}
    assert df.track_id.nunique() == 2


def test_track_by_iou_single_label_without_background():
    a = np.full((4, 4), 5)
    b = np.full((4, 4), 3)
    df = timelapse._track_by_iou(np.stack([a, b]))
    assert set(df.original_label) == {3, 5}


def test_track_by_iou_unchanged_with_background():
    df = timelapse._track_by_iou(np.stack([_with_bg(), _with_bg()]))
    assert set(df.original_label) == {1, 2}
    assert df.track_id.nunique() == 2


def test_remove_noninfected_removes_lowest_cell_without_background():
    stack = np.zeros((4, 4, 3), dtype=int)
    stack[:, :, 0] = _full()
    stack[:, 2:, 2] = 7
    out = utils._remove_noninfected(stack.copy(), 0, 1, 2)
    assert (out[:, :2, 0] == 0).all()
    assert (out[:, 2:, 0] == 2).all()


def test_remove_noninfected_unchanged_with_background():
    stack = np.zeros((6, 6, 3), dtype=int)
    stack[:, :, 0] = _with_bg()
    stack[3:5, 3:5, 2] = 9
    out = utils._remove_noninfected(stack.copy(), 0, 1, 2)
    assert (out[0:2, 0:2, 0] == 0).all()
    assert (out[3:5, 3:5, 0] == 2).all()


def test_remove_outside_objects_removes_lowest_pathogen_without_background():
    stack = np.zeros((4, 4, 3), dtype=int)
    stack[:, 2:, 0] = 1
    stack[:, :, 2] = _full()
    out = utils._remove_outside_objects(stack.copy(), 0, 1, 2)
    assert (out[:, :2, 2] == 0).all()
    assert (out[:, 2:, 2] == 2).all()


def test_remove_outside_objects_unchanged_with_background():
    stack = np.zeros((6, 6, 3), dtype=int)
    stack[3:5, 3:5, 0] = 1
    stack[:, :, 2] = _with_bg()
    out = utils._remove_outside_objects(stack.copy(), 0, 1, 2)
    assert (out[0:2, 0:2, 2] == 0).all()
    assert (out[3:5, 3:5, 2] == 2).all()


def test_remove_multiobject_cells_checks_lowest_cell_without_background():
    stack = np.zeros((4, 4, 2), dtype=int)
    stack[:, :, 0] = 1
    stack[0, 0, 1] = 1
    stack[3, 3, 1] = 2
    out = utils._remove_multiobject_cells(stack.copy(), 0, 0, None, None, 1)
    assert (out[:, :, 0] == 0).all()


def test_remove_multiobject_cells_unchanged_with_background():
    stack = np.zeros((6, 6, 2), dtype=int)
    stack[:, :, 0] = _with_bg()
    stack[0, 0, 1] = 1
    stack[1, 1, 1] = 2
    out = utils._remove_multiobject_cells(stack.copy(), 0, 0, None, None, 1)
    assert (out[0:2, 0:2, 0] == 0).all()
    assert (out[3:5, 3:5, 0] == 2).all()


def test_relabel_parent_covers_lowest_child_without_background():
    child = np.ones((4, 5), dtype=int)
    child[:, 3:] = 2
    parent = np.ones((4, 5), dtype=int)
    parent[:, 2] = 0
    new_parent, _ = utils._relabel_parent_with_child_labels(parent, child.copy())
    assert (new_parent[:, :2] == 1).all()
    assert (new_parent[:, 3:] == 2).all()


def test_relabel_parent_unchanged_with_background():
    child = _with_bg()
    parent = (child > 0).astype(int)
    new_parent, _ = utils._relabel_parent_with_child_labels(parent, child.copy())
    assert (new_parent[0:2, 0:2] == 1).all()
    assert (new_parent[3:5, 3:5] == 2).all()
    assert (new_parent[2, :] == 0).all()


def test_periphery_intensity_reports_lowest_label_without_background():
    image = np.arange(16, dtype=float).reshape(4, 4)
    labels = [r[0] for r in measure._periphery_intensity(_full(), image)]
    assert labels == [1, 2]


def test_periphery_intensity_unchanged_with_background():
    image = np.arange(36, dtype=float).reshape(6, 6)
    labels = [r[0] for r in measure._periphery_intensity(_with_bg(), image)]
    assert labels == [1, 2]


def test_object_correlation_reports_lowest_label_without_background():
    rng = np.random.default_rng(0)
    a = rng.random((4, 4))
    b = rng.random((4, 4))
    out = measure._calculate_correlation_object_level(a, b, _full(), {})
    assert sorted(out["label_correlation"]) == [1, 2]


def test_object_correlation_unchanged_with_background():
    rng = np.random.default_rng(0)
    a = rng.random((6, 6))
    b = rng.random((6, 6))
    out = measure._calculate_correlation_object_level(a, b, _with_bg(), {})
    assert sorted(out["label_correlation"]) == [1, 2]


def _texts(mask):
    import matplotlib.pyplot as plt

    image = np.random.default_rng(1).random(mask.shape + (1,))
    stack = mask[:, :, None]
    fig = plot._plot_merged_plot(False, image, stack, [0], 2, None, [mask],
                                 "viridis", ["red"], True)
    texts = sorted(t.get_text() for ax in fig.axes for t in ax.texts)
    plt.close(fig)
    return texts


def test_merged_plot_numbers_lowest_object_without_background():
    assert _texts(_full()) == ["1", "2"]


def test_merged_plot_unchanged_with_background():
    assert _texts(_with_bg()) == ["1", "2"]
