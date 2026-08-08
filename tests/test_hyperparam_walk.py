"""The N-dimensional Walk.

These defend the property that made the generalisation safe to make: the
old 2-by-2 search is not a special case in the code, it is two numeric axes
at resolution 2. If that stops being true, the four corners these tests
count will stop matching and the 174 tests that describe the old search
will start describing something else.
"""

import pytest

from spacr.hyperparam import (
    MAX_WALK_CANDIDATES_PER_ROUND, UMAP_WALK_PARAMETERS, WalkAxis,
    local_direction_search, umap_walk_axes, walk_neighbourhood, walk_search,
)


class TestWalkAxis:
    """One direction, and the range it is allowed to move in."""

    def test_resolution_two_is_the_classic_plus_minus_pair(self):
        """Resolution 2 omits the centre, which is what makes it a MOVE.

        The centre is where the walk already is. Scoring it again costs a
        fit and cannot change the answer.
        """
        axis = WalkAxis("x", step=1.0)
        assert axis.values_around(10.0) == [9.0, 11.0]

    def test_odd_resolution_puts_the_centre_back(self):
        axis = WalkAxis("x", step=1.0, resolution=3)
        assert axis.values_around(10.0) == [9.0, 10.0, 11.0]

    def test_higher_resolution_reaches_further_out(self):
        axis = WalkAxis("x", step=1.0, resolution=5)
        assert axis.values_around(10.0) == [8.0, 9.0, 10.0, 11.0, 12.0]

    def test_clamping_collapses_duplicates_at_a_boundary(self):
        """min_dist at 0 offers ONE move, not two identical ones.

        Both -step and +step exist, but -step clamps back onto the centre.
        Returning it twice would spend a fit re-scoring where we already
        are, at exactly the moment the search is short of directions.
        """
        axis = WalkAxis("min_dist", step=0.05, minimum=0.0, maximum=1.0)
        assert axis.values_around(0.0) == [0.0, 0.05]

    def test_integer_axes_never_emit_a_float(self):
        axis = WalkAxis("n_neighbors", step=1.0, minimum=2.0, integer=True)
        values = axis.values_around(15)
        assert values == [14, 16]
        assert all(isinstance(v, int) for v in values)

    def test_categorical_axis_rotates_from_the_current_value(self):
        """A categorical axis has no direction, so it offers the NEXT
        choices in declared order. Successive rounds from a moving centre
        therefore cover the alphabet instead of retrying the same pair."""
        axis = WalkAxis("init", choices=("spectral", "random", "pca"),
                        resolution=2)
        assert axis.values_around("spectral") == ["spectral", "random"]
        assert axis.values_around("random") == ["random", "pca"]

    def test_a_value_outside_the_alphabet_comes_back_inside(self):
        axis = WalkAxis("init", choices=("spectral", "random"))
        assert axis.clamp("nonsense") == "spectral"

    @pytest.mark.parametrize("kwargs, fragment", [
        ({"name": "x"}, "either a step"),
        ({"name": "x", "step": 0}, "never leaves the centre"),
        ({"name": "x", "step": 1, "resolution": 1}, "not being searched"),
        ({"name": "x", "choices": ("only",)}, "at least two"),
        ({"name": "x", "step": 1, "minimum": 5, "maximum": 1}, "above maximum"),
    ])
    def test_a_useless_axis_is_refused_at_construction(self, kwargs, fragment):
        with pytest.raises(ValueError, match=fragment):
            WalkAxis(**kwargs)


class TestNeighbourhood:

    def test_two_axes_at_resolution_two_are_the_original_four_corners(self):
        """The regression that matters. The 2-by-2 search is this."""
        axes = [WalkAxis("n_neighbors", step=1.0, minimum=2.0, integer=True),
                WalkAxis("min_dist", step=0.05, minimum=0.0, maximum=1.0)]
        centre = {"n_neighbors": 15, "min_dist": 0.1}
        moves, full = walk_neighbourhood(axes, centre)
        assert full is True
        assert sorted((m["n_neighbors"], m["min_dist"]) for m in moves) == [
            (14, 0.05), (14, 0.15), (16, 0.05), (16, 0.15)]

    def test_the_centre_is_never_a_candidate(self):
        axes = [WalkAxis("a", step=1.0, resolution=3),
                WalkAxis("b", step=1.0, resolution=3)]
        moves, _ = walk_neighbourhood(axes, {"a": 0.0, "b": 0.0})
        assert {"a": 0.0, "b": 0.0} not in moves
        assert len(moves) == 8  # 3x3 minus the centre

    def test_the_product_grows_with_the_axis_count(self):
        axes = [WalkAxis(n, step=1.0) for n in ("a", "b", "c")]
        moves, full = walk_neighbourhood(axes, {"a": 0, "b": 0, "c": 0})
        assert full is True and len(moves) == 8  # 2^3

    def test_past_the_cap_a_round_varies_one_axis_at_a_time(self):
        """Ten axes is 1024 fits for ONE step. The fallback is linear.

        The flag is what the caller uses to SAY so -- a search that
        quietly looked at 20 of 1024 configurations while reporting a
        neighbourhood walk would be lying about its own coverage.
        """
        names = [f"p{i}" for i in range(10)]
        axes = [WalkAxis(n, step=1.0) for n in names]
        centre = {n: 0.0 for n in names}
        moves, full = walk_neighbourhood(axes, centre)
        assert full is False
        assert len(moves) == 20  # 10 axes x 2 values, not 2^10
        for move in moves:
            differing = [n for n in names if move[n] != centre[n]]
            assert len(differing) == 1

    def test_the_cap_is_where_the_switch_happens(self):
        axes = [WalkAxis(f"p{i}", step=1.0) for i in range(5)]
        centre = {f"p{i}": 0.0 for i in range(5)}
        assert walk_neighbourhood(axes, centre, max_candidates=32)[1] is True
        assert walk_neighbourhood(axes, centre, max_candidates=31)[1] is False
        assert MAX_WALK_CANDIDATES_PER_ROUND >= 32  # keeps 5 axes factorial

    def test_no_axes_is_an_empty_neighbourhood_not_a_crash(self):
        assert walk_neighbourhood([], {"a": 1}) == ([], True)


class TestWalkSearch:

    def _rosenbrock_like(self, target):
        """Score peaks at ``target``; higher is better."""
        def fit(params):
            return -sum((params[k] - v) ** 2 for k, v in target.items())
        return fit

    def test_the_walk_climbs_in_three_dimensions(self):
        target = {"a": 5.0, "b": 5.0, "c": 5.0}
        axes = [WalkAxis(n, step=1.0) for n in "abc"]
        result = walk_search(self._rosenbrock_like(target),
                             {"a": 0.0, "b": 0.0, "c": 0.0}, axes,
                             n_trials=20)
        assert result.ok
        for name in "abc":
            assert result.best.params[name] == pytest.approx(5.0, abs=1.0)

    def test_frozen_parameters_reach_the_fit_untouched(self):
        seen = []

        def fit(params):
            seen.append(dict(params))
            return -abs(params["a"])

        walk_search(fit, {"a": 3.0, "held": "constant"},
                    [WalkAxis("a", step=1.0)], n_trials=3)
        assert seen and all(p["held"] == "constant" for p in seen)

    def test_the_starting_point_is_a_centre_and_is_never_fitted(self):
        seen = []

        def fit(params):
            seen.append(params["a"])
            return -abs(params["a"] - 100)

        walk_search(fit, {"a": 0.0}, [WalkAxis("a", step=1.0)], n_trials=1)
        assert 0.0 not in seen

    def test_it_stops_when_a_round_stops_improving(self):
        """Two rounds on a flat landscape, not one.

        The FIRST round always moves: there is no best score yet, so the
        gain is infinite by definition and any result beats nothing. Only
        the second round has something to fail to improve on. This is the
        two-axis search's behaviour unchanged, and the reason a flat
        objective costs two rounds of fits rather than one.
        """
        calls = []

        def fit(params):
            calls.append(1)
            return 1.0  # flat: no round after the first can improve

        result = walk_search(fit, {"a": 0.0}, [WalkAxis("a", step=1.0)],
                             n_trials=50)
        assert len(calls) == 4  # two rounds of two, then the floor stops it
        assert any("not more than" in n for n in result.notes)

    def test_the_fallback_is_recorded_in_the_notes(self):
        """No silent caps: a reduced round says it was reduced."""
        names = [f"p{i}" for i in range(10)]
        result = walk_search(
            lambda p: -sum(p[n] ** 2 for n in names),
            {n: 0.0 for n in names},
            [WalkAxis(n, step=1.0) for n in names], n_trials=1)
        assert any("varies ONE axis at a time" in n for n in result.notes)

    def test_the_axes_are_described_in_the_notes(self):
        result = walk_search(lambda p: -p["a"] ** 2, {"a": 1.0},
                             [WalkAxis("a", step=0.5)], n_trials=1)
        assert any("Walk over 1 parameter(s)" in n and "step 0.5" in n
                   for n in result.notes)

    @pytest.mark.parametrize("axes, start, fragment", [
        ([], {"a": 1}, "at least one axis"),
        ([WalkAxis("a", step=1)], {}, "missing"),
        ([WalkAxis("a", step=1), WalkAxis("a", step=2)], {"a": 1},
         "listed twice"),
    ])
    def test_an_unsearchable_space_is_refused(self, axes, start, fragment):
        with pytest.raises(ValueError, match=fragment):
            walk_search(lambda p: 1.0, start, axes)


class TestBackwardsCompatibility:
    """local_direction_search must keep behaving exactly as it did."""

    def test_it_still_scores_four_corners_per_round(self):
        seen = []

        def fit(params):
            seen.append((params["n_neighbors"], params["min_dist"]))
            return 1.0  # flat, so exactly one round runs

        local_direction_search(fit, {"n_neighbors": 15, "min_dist": 0.1},
                               n_trials=1)
        assert sorted(seen) == [(14, 0.05), (14, 0.15),
                                (16, 0.05), (16, 0.15)]

    def test_it_still_clamps_n_neighbors_to_two(self):
        seen = []
        local_direction_search(
            lambda p: seen.append(p["n_neighbors"]) or 1.0,
            {"n_neighbors": 2, "min_dist": 0.1}, n_trials=1)
        assert min(seen) >= 2


class TestUmapWalkAxes:

    def test_the_default_space_is_the_two_it_has_always_been(self):
        axes = umap_walk_axes({"n_neighbors": 15, "min_dist": 0.1})
        assert [a.name for a in axes] == ["n_neighbors", "min_dist"]

    def test_every_named_parameter_can_be_an_axis(self):
        start = {"n_neighbors": 15, "min_dist": 0.1, "n_components": 2,
                 "metric": "euclidean", "spread": 1.0,
                 "set_op_mix_ratio": 1.0, "local_connectivity": 1,
                 "repulsion_strength": 1.0, "negative_sample_rate": 5,
                 "init": "spectral"}
        axes = umap_walk_axes(start, parameters=sorted(UMAP_WALK_PARAMETERS))
        assert len(axes) == len(UMAP_WALK_PARAMETERS) == 10

    def test_umaps_own_ranges_are_carried_by_the_axis(self):
        """A walk generates values nobody typed, so out-of-range has to be
        impossible here rather than caught inside the fit."""
        axes = {a.name: a for a in umap_walk_axes(
            {"min_dist": 0.1, "set_op_mix_ratio": 0.5, "n_neighbors": 15},
            parameters=["min_dist", "set_op_mix_ratio", "n_neighbors"])}
        assert axes["min_dist"].clamp(-1) == 0.0
        assert axes["min_dist"].clamp(9) == 1.0
        assert axes["set_op_mix_ratio"].clamp(2) == 1.0
        assert axes["n_neighbors"].clamp(0) == 2

    def test_the_metric_axis_offers_the_installed_metrics(self):
        axes = umap_walk_axes({"metric": "euclidean"}, parameters=["metric"])
        assert "euclidean" in axes[0].choices and len(axes[0].choices) > 5

    def test_n_neighbors_max_reaches_the_axis(self):
        axes = umap_walk_axes({"n_neighbors": 15}, parameters=["n_neighbors"],
                              n_neighbors_max=20)
        assert axes[0].clamp(999) == 20

    def test_a_parameter_umap_does_not_have_is_refused_by_name(self):
        with pytest.raises(ValueError, match="Not a searchable"):
            umap_walk_axes({"nonsense": 1}, parameters=["nonsense"])

    def test_a_missing_starting_value_is_refused(self):
        with pytest.raises(ValueError, match="missing"):
            umap_walk_axes({"n_neighbors": 15}, parameters=["spread"])
