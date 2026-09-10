"""The three-way neighbourhood split, and the confound it removes.

Instruction 388, step 1. The package measures each cell alone, so an
uninfected cell touching an infected one and an uninfected cell on the far
side of the well are currently the same row. That is not only a missing
column: pooling bystanders into "uninfected" makes the control a mixture,
inflates its variance and shrinks the effect size of every infection
comparison the package produces.

THE ACCEPTANCE TEST THE INSTRUCTION NAMES IS THE FIRST ONE HERE -- a well
with no parasites must report no bystanders, and it is "the first thing to
check". It holds by construction rather than by special case: with nothing
infected, every distance is infinity, and infinity is not within any finite
reach.
"""

import numpy as np
import pytest

from spacr.bystanders import (DEFAULT_REACH_IN_DIAMETERS, STATUSES, classify,
                              counts, distance_to_infected,
                              reach_from_diameter)


def _row_of_cells(n=5, size=10, gap=0):
    """`n` square cells in a row, each `size` across, `gap` apart."""
    pitch = size + gap
    mask = np.zeros((size + 2, n * pitch + 2), dtype=np.int32)
    for i in range(n):
        x = 1 + i * pitch
        mask[1:1 + size, x:x + size] = i + 1
    return mask


class TestAWellWithNoParasitesHasNoBystanders:
    """The instruction's own first check, and it must not be a special case."""

    def test_nothing_infected_makes_every_cell_distal(self):
        mask = _row_of_cells()
        frame = classify(mask, [], reach=100.0)
        assert set(frame["neighbourhood"]) == {"distal"}
        assert counts(frame) == {"infected": 0, "bystander": 0, "distal": 5}

    def test_and_the_distances_are_infinite_rather_than_zero(self):
        """Zero would make every cell a bystander of nothing.

        Infinity is the only value that keeps `distance <= reach` false for
        every finite reach, which is why the classifier needs no special
        case for an uninfected well.
        """
        frame = distance_to_infected(_row_of_cells(), [])
        assert np.isinf(frame["distance_to_infected"]).all()

    def test_an_empty_image_answers_with_no_rows(self):
        frame = classify(np.zeros((6, 6), dtype=np.int32), [], reach=5.0)
        assert len(frame) == 0
        assert counts(frame) == {"infected": 0, "bystander": 0, "distal": 0}


class TestTheSplitItself:

    def test_a_touching_cell_is_a_bystander_and_a_far_one_is_not(self):
        mask = _row_of_cells(n=5, size=10, gap=0)
        # Cell 1 is infected. Cell 2 touches it; cell 5 is four cells away.
        frame = classify(mask, [1], reach=1.0).set_index("label")
        assert frame.loc[1, "neighbourhood"] == "infected"
        assert frame.loc[2, "neighbourhood"] == "bystander"
        assert frame.loc[5, "neighbourhood"] == "distal"

    def test_an_abutting_cell_is_one_pixel_away_not_zero(self):
        """THE OBVIOUS READING IS WRONG AND THIS PINS THE RIGHT ONE.

        `surface_distance_transform`'s docstring says two touching objects
        come out at 0, and that is true of the case it describes: reading
        the field at a point INSIDE the other object, which is what an
        overlapping or contained object gives. Two disjoint labels that
        merely share a border do not overlap -- the neighbour's own nearest
        pixel is one step away, so its minimum is 1.

        Only the infected cell itself reads 0. A reach expressed in cell
        diameters dwarfs one pixel either way, so this changes no
        classification; it is pinned because the first version of this
        module asserted 0 and was wrong.
        """
        frame = distance_to_infected(_row_of_cells(gap=0), [1]).set_index("label")
        assert frame.loc[1, "distance_to_infected"] == 0.0
        assert frame.loc[2, "distance_to_infected"] == 1.0
        assert frame.loc[3, "distance_to_infected"] > 1.0

    def test_the_reach_is_what_moves_the_boundary(self):
        mask = _row_of_cells(n=5, size=10, gap=4)
        near = classify(mask, [1], reach=5.0)
        far = classify(mask, [1], reach=25.0)
        assert (near["neighbourhood"] == "bystander").sum() < \
               (far["neighbourhood"] == "bystander").sum()

    def test_an_infected_cell_is_never_relabelled_a_bystander(self):
        """Two infected neighbours must both stay infected."""
        frame = classify(_row_of_cells(), [1, 2], reach=100.0).set_index("label")
        assert frame.loc[1, "neighbourhood"] == "infected"
        assert frame.loc[2, "neighbourhood"] == "infected"


class TestTurningItOff:
    """A non-positive reach cannot invent bystanders."""

    @pytest.mark.parametrize("reach", [0.0, -1.0])
    def test_no_reach_means_no_bystanders(self, reach):
        frame = classify(_row_of_cells(), [1], reach=reach)
        assert "bystander" not in set(frame["neighbourhood"])
        assert (frame["neighbourhood"] == "infected").sum() == 1

    @pytest.mark.parametrize("reach", ["wide", None, float("nan")])
    def test_a_reach_that_is_not_a_number_falls_back_to_off(self, reach):
        """A mis-parsed setting must not silently widen the split."""
        frame = classify(_row_of_cells(), [1], reach=reach)
        assert "bystander" not in set(frame["neighbourhood"])


class TestTheReachIsDerivedFromTheData:
    """A multiple of measured cell diameter, not a hard-coded micron count."""

    def test_one_diameter_by_default(self):
        assert DEFAULT_REACH_IN_DIAMETERS == 1.0
        assert reach_from_diameter(30.0) == 30.0

    def test_it_scales_with_the_cells(self):
        assert reach_from_diameter(60.0) == 2 * reach_from_diameter(30.0)

    @pytest.mark.parametrize("bad", [None, "large", float("nan"), float("inf")])
    def test_a_diameter_it_cannot_read_reaches_nothing(self, bad):
        """Zero, not a guess: an unmeasurable plate gets no split rather
        than a split on an invented number."""
        assert reach_from_diameter(bad) == 0.0

    def test_a_negative_diameter_cannot_produce_a_negative_reach(self):
        assert reach_from_diameter(-10.0) == 0.0


class TestTheCountsAreAlwaysComplete:

    def test_every_status_is_present_even_at_zero(self):
        frame = classify(_row_of_cells(), [1], reach=1.0)
        assert set(counts(frame)) == set(STATUSES)

    def test_the_counts_add_up_to_the_cells(self):
        mask = _row_of_cells(n=7, gap=2)
        frame = classify(mask, [1, 4], reach=3.0)
        assert sum(counts(frame).values()) == 7


# --- step 2: how infected is the neighbourhood, not just the nearest cell ---

from spacr.bystanders import neighbourhood  # noqa: E402


class TestTheNeighbourhoodIsCountedNotAssumed:
    """`neighbours_counted` is a column because k is a request, not a fact."""

    def test_a_sparse_field_reports_what_it_could_find(self):
        """Three cells cannot supply six neighbours each.

        A fraction divided by an assumed k would be wrong in exactly the
        sparse fields where a bystander analysis matters most.
        """
        frame = neighbourhood(_row_of_cells(n=3), [1], k=6)
        assert (frame["neighbours_counted"] == 2).all()
        assert frame["local_infected_fraction"].max() <= 1.0

    def test_a_full_field_supplies_the_k_that_was_asked_for(self):
        frame = neighbourhood(_row_of_cells(n=10), [1], k=3)
        assert (frame["neighbours_counted"] == 3).all()

    def test_one_cell_alone_has_no_neighbourhood(self):
        frame = neighbourhood(_row_of_cells(n=1), [1], k=6)
        assert len(frame) == 1
        assert frame["neighbours_counted"].iloc[0] == 0
        assert np.isnan(frame["local_infected_fraction"].iloc[0])


class TestTheCellIsNotItsOwnNeighbour:

    def test_an_isolated_infected_cell_does_not_count_itself(self):
        """Otherwise every infected cell would report a non-zero fraction
        purely by being infected, and the column would measure nothing."""
        frame = neighbourhood(_row_of_cells(n=6), [1], k=2).set_index("label")
        # Cell 6 is at the far end; neither of its two nearest is cell 1.
        assert frame.loc[6, "neighbours_infected"] == 0
        assert frame.loc[6, "local_infected_fraction"] == 0.0

    def test_an_infected_cell_still_gets_a_fraction(self):
        """"This infected cell sits among uninfected ones" is a different
        observation from "this one sits in a cluster"."""
        alone = neighbourhood(_row_of_cells(n=6), [1], k=2).set_index("label")
        clustered = neighbourhood(_row_of_cells(n=6), [1, 2, 3],
                                  k=2).set_index("label")
        assert alone.loc[1, "local_infected_fraction"] == 0.0
        assert clustered.loc[2, "local_infected_fraction"] > 0.0


class TestItAnswersTheQuestionClassifyCannot:

    def test_the_middle_of_a_focus_scores_above_its_edge(self):
        """Both cells are bystanders; they are not in the same place.

        This is the whole reason step 2 exists on top of step 1.
        """
        mask = _row_of_cells(n=9)
        frame = neighbourhood(mask, [4, 5, 6], k=2).set_index("label")
        middle, edge = frame.loc[5], frame.loc[8]
        assert middle["local_infected_fraction"] > edge["local_infected_fraction"]

    def test_nothing_infected_is_a_field_of_zeros_not_of_nans(self):
        frame = neighbourhood(_row_of_cells(n=5), [], k=2)
        assert (frame["neighbours_infected"] == 0).all()
        assert (frame["local_infected_fraction"] == 0.0).all()


class TestSpacingIsRespectedAndNeverFatal:

    @pytest.mark.parametrize("spacing", [None, 1.0, (1.0, 1.0), "coarse",
                                         (0.0, 1.0), (1.0, 2.0, 3.0)])
    def test_any_spacing_still_answers(self, spacing):
        """An anisotropic stack with a mis-typed spacing gets neighbours in
        pixels -- what the caller had before -- rather than no answer."""
        frame = neighbourhood(_row_of_cells(n=5), [1], k=2, spacing=spacing)
        assert len(frame) == 5
        assert frame["neighbours_counted"].min() >= 1


# --- step 4: infected vs bystander vs distal, on a measured feature --------

import pandas as pd  # noqa: E402

from spacr.bystanders import compare  # noqa: E402


class TestTheComparisonTheItemIsFor:

    @staticmethod
    def _measured():
        return pd.DataFrame({
            "label": [1, 2, 3, 4, 5, 6],
            "neighbourhood": ["infected", "infected", "bystander",
                              "bystander", "distal", "distal"],
            "area": [100.0, 110.0, 150.0, 160.0, 200.0, 210.0],
        })

    def test_it_separates_the_three_groups(self):
        out = compare(self._measured(), "area").set_index("neighbourhood")
        assert out.loc["infected", "mean"] == pytest.approx(105.0)
        assert out.loc["bystander", "mean"] == pytest.approx(155.0)
        assert out.loc["distal", "mean"] == pytest.approx(205.0)

    def test_this_is_the_confound_it_removes(self):
        """Pooling bystanders into "uninfected" inflates the control.

        The same six cells, read the way the package reads them today,
        give an uninfected group whose spread is far wider than either of
        the two populations inside it -- which is the mechanism by which
        every infection effect size shrinks.
        """
        rows = self._measured()
        split = compare(rows, "area").set_index("neighbourhood")
        pooled = rows[rows["neighbourhood"] != "infected"]["area"].std(ddof=1)
        assert pooled > split.loc["bystander", "std"]
        assert pooled > split.loc["distal", "std"]

    def test_every_group_is_a_row_even_when_empty(self):
        rows = self._measured()
        rows = rows[rows["neighbourhood"] != "bystander"]
        out = compare(rows, "area")
        assert list(out["neighbourhood"]) == list(STATUSES)
        assert int(out.loc[out["neighbourhood"] == "bystander", "n"].iloc[0]) == 0

    def test_a_single_cell_has_no_spread_rather_than_zero_spread(self):
        """Zero would read as a perfectly consistent group."""
        rows = pd.DataFrame({"neighbourhood": ["bystander"], "area": [5.0]})
        out = compare(rows, "area").set_index("neighbourhood")
        assert out.loc["bystander", "n"] == 1
        assert np.isnan(out.loc["bystander", "std"])

    def test_a_missing_feature_answers_with_the_shape_and_no_numbers(self):
        out = compare(self._measured(), "not_a_column")
        assert list(out["neighbourhood"]) == list(STATUSES)
        assert (out["n"] == 0).all()

    def test_unreadable_values_are_dropped_not_counted(self):
        rows = pd.DataFrame({"neighbourhood": ["distal"] * 3,
                             "area": [1.0, "wide", None]})
        out = compare(rows, "area").set_index("neighbourhood")
        assert out.loc["distal", "n"] == 1
