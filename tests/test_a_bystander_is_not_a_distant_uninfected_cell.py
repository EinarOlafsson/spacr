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
