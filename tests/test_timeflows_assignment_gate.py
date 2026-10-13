"""A rejected temporal edge must not displace a valid identity link."""
import math

import numpy as np
import pytest

from spacr.timeflows_model import link_by_timeflows


def scene(points=((21.5, 12.5), (21.5, 7.5))):
    source = np.zeros((32, 36), dtype=np.int64)
    source[4:8, 4:8] = 7
    source[4:8, 14:18] = 9000000001
    target = np.zeros_like(source)
    target[20:24, 10:14] = 71
    target[20:24, 20:24] = 81
    yy, xx = np.indices(source.shape)
    vector = np.zeros((2,) + source.shape, dtype=np.float64)
    diameter = 8 / math.sqrt(math.pi)
    for label, (y, x) in zip((7, 9000000001), points):
        selected = source == label
        vector[0, selected] = (y - yy[selected]) / diameter
        vector[1, selected] = (x - xx[selected]) / diameter
    return source, target, {"vector": vector, "successor": np.ones(source.shape)}


def test_an_impossible_edge_cannot_steal_the_closest_valid_match():
    source, target, prediction = scene()
    assert link_by_timeflows(source, target, prediction) == {7: 71}


def test_a_source_with_no_admissible_target_does_not_change_other_links():
    source, target, prediction = scene(points=((21.5, 11.5), (21.5, -100)))
    assert link_by_timeflows(source, target, prediction) == {7: 71}


def test_global_matching_can_use_two_admissible_edges():
    source, target, prediction = scene(points=((21.5, 12.5), (21.5, 20.5)))
    assert link_by_timeflows(source, target, prediction) == {7: 71, 9000000001: 81}


def test_successor_threshold_is_inclusive_and_can_abstain():
    source, target, prediction = scene()
    prediction["successor"][source == 7] = 0.5
    assert link_by_timeflows(source, target, prediction) == {7: 71}
    prediction["successor"][source == 7] = 0.49
    assert link_by_timeflows(source, target, prediction) == {9000000001: 71}


def test_zero_distance_accepts_exact_centres_without_forcing_other_links():
    source, _, prediction = scene()
    target = np.zeros_like(source)
    target[source == 7] = 71
    target[20:24, 20:24] = 81
    prediction["vector"][:] = 0
    assert link_by_timeflows(source, target, prediction, max_distance=0) == {7: 71}


@pytest.mark.parametrize("name,value", [
    ("max_distance", -1), ("max_distance", np.nan), ("max_distance", np.inf),
    ("min_successor", -0.1), ("min_successor", 1.1),
    ("min_successor", np.nan), ("min_successor", np.inf),
])
def test_invalid_linking_thresholds_raise_before_assignment(name, value):
    with pytest.raises(ValueError, match=name):
        link_by_timeflows(*scene(), **{name: value})


def test_background_predictions_do_not_participate_in_linking():
    source, target, prediction = scene(points=((21.5, 11.5), (21.5, 21.5)))
    prediction["vector"][:, source == 0] = np.nan
    prediction["successor"][source == 0] = np.nan
    assert link_by_timeflows(source, target, prediction) == {7: 71, 9000000001: 81}


@pytest.mark.parametrize("channel,value", [("vector", np.nan), ("vector", np.inf),
                                         ("successor", np.nan), ("successor", -0.1),
                                         ("successor", 1.1)])
def test_invalid_foreground_prediction_is_reported(channel, value):
    source, target, prediction = scene()
    if channel == "vector":
        prediction[channel][:, source == 7] = value
    else:
        prediction[channel][source == 7] = value
    with pytest.raises(ValueError, match="prediction"):
        link_by_timeflows(source, target, prediction)


@pytest.mark.parametrize("channel", ["vector", "successor"])
def test_prediction_shape_must_match_the_source_frame(channel):
    source, target, prediction = scene()
    prediction[channel] = prediction[channel][..., :-1]
    with pytest.raises(ValueError, match="shape"):
        link_by_timeflows(source, target, prediction)


def test_distance_boundary_is_inclusive():
    source, target, prediction = scene()
    prediction["successor"][source != 7] = 0
    cutoff = 1 / (2 * math.sqrt(16 / math.pi))
    assert link_by_timeflows(source, target, prediction,
                             max_distance=cutoff) == {7: 71}
    assert link_by_timeflows(source, target, prediction,
                             max_distance=np.nextafter(cutoff, 0)) == {}


def test_unrepresentable_predicted_centre_is_reported():
    source, target, prediction = scene()
    prediction["vector"][:, source == 7] = np.finfo(np.float64).max
    with pytest.raises(ValueError, match="prediction.*centre"):
        link_by_timeflows(source, target, prediction)


@pytest.mark.parametrize("nsource,ntarget", [(1, 3), (3, 1), (2, 3), (3, 2), (3, 3)])
def test_assignment_cost_matches_exhaustive_partial_matchings(nsource, ntarget):
    from itertools import product

    rng = np.random.default_rng(142)
    source = np.zeros((80, 80), np.int64)
    target = np.zeros_like(source)
    source_ids = [7 + i * 1000000000 for i in range(nsource)]
    target_ids = [71 + i * 10 for i in range(ntarget)]
    for i, label in enumerate(source_ids):
        source[2 + i * 6:4 + i * 6, 2:4] = label
    centres = [(60.5, 4.5 + j * 5) for j in range(ntarget)]
    for j, label in enumerate(target_ids):
        target[60:62, 4 + j * 5:6 + j * 5] = label
    diameter = 4 / math.sqrt(math.pi)
    yy, xx = np.indices(source.shape)
    choices = [choice for choice in product(range(-1, ntarget), repeat=nsource)
               if len([j for j in choice if j >= 0]) == len({j for j in choice if j >= 0})]
    for _ in range(12):
        points = np.array([centres[int(rng.integers(ntarget))] for _ in source_ids])
        points += rng.normal(0, 2, size=points.shape)
        vector = np.zeros((2,) + source.shape)
        for label, (y, x) in zip(source_ids, points):
            inside = source == label
            vector[0, inside] = (y - yy[inside]) / diameter
            vector[1, inside] = (x - xx[inside]) / diameter
        distances = [[math.dist(p, c) / diameter for c in centres] for p in points]
        candidates = []
        for choice in choices:
            if all(j < 0 or distances[i][j] <= 1 for i, j in enumerate(choice)):
                candidates.append(sum(1 if j < 0 else distances[i][j]
                                      for i, j in enumerate(choice)))
        links = link_by_timeflows(source, target,
                                 {"vector": vector, "successor": np.ones(source.shape)})
        assert len(set(links.values())) == len(links)
        actual = nsource - len(links)
        for old, new in links.items():
            distance = distances[source_ids.index(old)][target_ids.index(new)]
            assert distance <= 1 + 1e-12
            actual += distance
        assert actual == pytest.approx(min(candidates), abs=1e-10)
