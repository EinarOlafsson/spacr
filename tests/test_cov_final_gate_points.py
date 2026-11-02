"""Which outline anchors are new, and the Cluster button's hand-off."""

import numpy as np

from spacr.qt.widgets.gate_editor import GateEditorPanel, _distinct_from


def test_every_candidate_is_new_when_nothing_exists_yet():
    points = [[0, 0], [1, 0], [0, 1]]
    assert _distinct_from(points, np.empty((0, 2))).tolist() == [True] * 3


def test_a_candidate_on_an_existing_point_is_not_new():
    keep = _distinct_from([[0, 0], [5, 5]], [[0, 0]])
    assert keep.tolist() == [False, True]


def test_a_repeated_candidate_is_kept_once_across_blocks():
    points = [[1, 1], [2, 2], [1, 1], [3, 3], [2, 2]]
    keep = _distinct_from(points, np.empty((0, 2)), block=2)
    assert keep.tolist() == [True, True, False, True, False]


def test_repeats_of_an_existing_point_are_all_refused():
    keep = _distinct_from([[0, 0], [7, 7], [7, 7]], [[7, 7]], block=2)
    assert keep.tolist() == [True, False, False]


def test_the_cluster_button_asks_before_running_on_a_worker():
    asked = []

    class Stub:
        def run_cluster_in_background(self, ask=False):
            asked.append(ask)

    GateEditorPanel._on_cluster_clicked(Stub(), False)
    assert asked == [True]
