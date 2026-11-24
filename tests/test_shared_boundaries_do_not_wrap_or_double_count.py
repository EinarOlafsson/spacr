"""Contacts between labels are counted once and never across the field edge."""
import numpy as np

from spacr.utils import _compute_shared_boundaries, _merge_by_perimeter


def test_a_contact_is_counted_once_per_touching_pixel_pair():
    labels = np.zeros((6, 6), dtype=np.uint16)
    labels[1:5, 1:3] = 1
    labels[1:5, 3:5] = 2
    assert _compute_shared_boundaries(labels) == {(1, 2): 4}


def test_labels_on_opposite_edges_do_not_touch():
    labels = np.zeros((5, 8), dtype=np.uint16)
    labels[:, 0] = 1
    labels[:, 7] = 2
    assert _compute_shared_boundaries(labels) == {}
    labels = np.zeros((8, 5), dtype=np.uint16)
    labels[0, :] = 1
    labels[7, :] = 2
    assert _compute_shared_boundaries(labels) == {}


def test_edge_labels_are_not_merged_by_wrapping():
    labels = np.zeros((10, 10), dtype=np.uint16)
    labels[2:8, 0:2] = 1
    labels[2:8, 8:10] = 2
    parent = {}
    _merge_by_perimeter(labels, 0.1, parent)
    assert not parent or all(k == v for k, v in parent.items())
