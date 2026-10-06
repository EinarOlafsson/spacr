"""Random AP is the expectation over finite complete rankings."""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from spacr.embeddings import _retrieval_scorecard


def _enumerated_ap(candidates, positives):
    values = []
    for positions in combinations(range(1, candidates + 1), positives):
        values.append(sum(hit / rank for hit, rank in enumerate(positions, 1))
                      / positives)
    return sum(values) / len(values)


@pytest.mark.parametrize("candidates", range(2, 8))
def test_chance_map_matches_every_possible_small_class_size(candidates):
    for positives in range(1, candidates):
        names = ["a"] * (positives + 1) + ["b"] * (candidates - positives)
        keys = list(map(str, range(len(names))))
        frame = pd.DataFrame(np.eye(len(names)), index=keys)
        card = _retrieval_scorecard(frame, dict(zip(keys, names)))
        counts = [names.count(name) for name in names]
        expected = [_enumerated_ap(candidates, count - 1)
                    for count in counts if count > 1]
        assert card["chance_map"] == pytest.approx(
            sum(expected) / len(expected), abs=1e-12)
        assert card["chance_precision"] == pytest.approx(
            sum(count - 1 for count in counts) / (len(names) * candidates))


def test_four_crops_distinguish_random_ap_from_random_precision():
    keys = ["a0", "a1", "b0", "b1"]
    card = _retrieval_scorecard(pd.DataFrame(np.eye(4), index=keys),
                               dict(zip(keys, ["a", "a", "b", "b"])))
    assert card["chance_map"] == pytest.approx(11 / 18)
    assert card["chance_precision"] == pytest.approx(1 / 3)


def test_imbalanced_classes_weight_queries_and_exclude_singleton_ap():
    names = ["a"] * 3 + ["b"] * 2 + ["singleton"]
    keys = list(map(str, range(6)))
    card = _retrieval_scorecard(pd.DataFrame(np.eye(6), index=keys),
                               dict(zip(keys, names)))
    expected = (3 * _enumerated_ap(5, 2) + 2 * _enumerated_ap(5, 1)) / 5
    assert card["chance_map"] == pytest.approx(expected, abs=1e-12)
    assert card["chance_precision"] == pytest.approx(8 / 30)


def test_only_singleton_classes_preserve_undefined_ap():
    with pytest.warns(RuntimeWarning):
        card = _retrieval_scorecard(pd.DataFrame(np.eye(2), index=["a", "b"]),
                                   {"a": "a", "b": "b"})
    assert np.isnan(card["map"]) and np.isnan(card["chance_map"])
    assert card["chance_precision"] == 0
