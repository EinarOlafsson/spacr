"""Ripley's K, judged the only way it can be: against patterns we made.

THE TEST IS A SIMULATION, NOT A PLATE, and instruction 388 says why: real
data has no ground truth about its own clustering, so a K validated on it
is a K nobody can debug. Every pattern below is generated with a known
answer -- random, clustered, or on a grid -- inside a real mask, and the
statistic has to agree.

THE SEEDS ARE FIXED. A min/max envelope over 99 simulations is an exact
test at 2%, which means one run in fifty of a genuinely random pattern
leaves it somewhere; with a free seed this file would fail for real about
that often and teach everyone to re-run it.
"""

from __future__ import annotations

import numpy as np
import pytest

from spacr.point_patterns import (DEFAULT_SIMULATIONS, MAX_RADIUS_FRACTION,
                                  csr_envelope, ripley_k, ripley_test,
                                  suggest_radii)

FIELD = (200, 200)
RADII = np.array([4.0, 8.0, 12.0, 16.0, 20.0])


def _whole_field(shape=FIELD):
    return np.ones(shape, dtype=bool)


def _disc(shape=FIELD, radius=90):
    """A window that is emphatically not a rectangle."""
    rows, cols = np.ogrid[:shape[0], :shape[1]]
    centre = (shape[0] / 2, shape[1] / 2)
    return ((rows - centre[0]) ** 2 + (cols - centre[1]) ** 2) < radius ** 2


def _random_points(mask, n, seed):
    inside = np.argwhere(mask)
    rng = np.random.default_rng(seed)
    return inside[rng.choice(len(inside), size=n, replace=False)]


def _clustered_points(mask, n, seed, *, foci=6, sigma=4.0):
    """n points in a few tight clumps, all inside the mask."""
    inside = np.argwhere(mask)
    rng = np.random.default_rng(seed)
    centres = inside[rng.choice(len(inside), size=foci, replace=False)]
    allowed = set(map(tuple, inside.tolist()))
    out = []
    while len(out) < n:
        centre = centres[rng.integers(0, foci)]
        point = np.round(centre + rng.normal(0, sigma, 2)).astype(int)
        if tuple(point) in allowed and tuple(point) not in set(map(tuple, out)):
            out.append(point)
    return np.array(out)


def _grid_points(mask, step=12):
    """A lattice: the most dispersed arrangement there is."""
    rows, cols = np.where(mask)
    grid = [(r, c) for r in range(rows.min(), rows.max(), step)
            for c in range(cols.min(), cols.max(), step) if mask[r, c]]
    return np.array(grid)


class TestTheKnownPatternsComeBackKnown:

    def test_a_random_pattern_stays_inside_its_envelope(self):
        """The claim the whole statistic rests on."""
        mask = _whole_field()
        points = _random_points(mask, 250, seed=11)
        got = ripley_test(points, mask, RADII, seed=3)
        assert not got["clustered"].any(), got[["r", "l_minus_r", "hi"]]
        assert not got["dispersed"].any(), got[["r", "l_minus_r", "lo"]]

    def test_a_clustered_pattern_leaves_it_upwards(self):
        mask = _whole_field()
        points = _clustered_points(mask, 250, seed=11)
        got = ripley_test(points, mask, RADII, seed=3)
        assert got["clustered"].all()
        assert not got["dispersed"].any()

    def test_a_lattice_leaves_it_downwards(self):
        """Dispersion is a real answer, not just the absence of clustering."""
        mask = _whole_field()
        points = _grid_points(mask, step=12)
        got = ripley_test(points, mask, RADII, seed=3)
        assert got.loc[got["r"] < 12, "dispersed"].all()

    def test_a_lattice_overshoots_at_its_own_spacing(self):
        """And that is the lattice, not a fault in K.

        Below the spacing a grid has no neighbours at all, so K sits under
        the envelope. AT the spacing four neighbours arrive at once, and a
        step function crossing lambda*pi*r^2 has to overshoot it. K
        oscillates around the null for a regular pattern; only the small-r
        deficit is a stable read of dispersion, which is why the test
        above asks for r below the spacing.

        This test exists so that the overshoot is a recorded property
        rather than the thing that makes someone "fix" the estimator.
        """
        mask = _whole_field()
        got = ripley_test(_grid_points(mask, step=12), mask,
                          np.array([8.0, 12.0]), seed=3)
        assert got["dispersed"].iloc[0]
        assert got["clustered"].iloc[1]

    def test_the_answer_does_not_depend_on_the_window_being_a_rectangle(self):
        mask = _disc()
        random = ripley_test(_random_points(mask, 200, seed=5), mask, RADII,
                             seed=7)
        clumped = ripley_test(_clustered_points(mask, 200, seed=5), mask,
                              RADII, seed=7)
        assert not random["clustered"].any()
        assert clumped["clustered"].all()


class TestEdgeCorrectionIsThePartThatDecidesTheAnswer:

    def test_without_it_a_random_pattern_reports_dispersion(self):
        """The artefact instruction 388 names, reproduced.

        Uncorrected, points near the boundary find fewer neighbours than
        they have, because part of their neighbourhood is outside the
        image. The deficit grows with r and reads as dispersion that the
        pattern does not have.
        """
        mask = _whole_field()
        points = _random_points(mask, 250, seed=11)
        naive = ripley_k(points, mask, RADII, correction="none")
        assert (naive["l_minus_r"] < 0).all()
        assert naive["l_minus_r"].iloc[-1] < naive["l_minus_r"].iloc[0]

    def test_with_it_the_same_pattern_sits_at_zero(self):
        mask = _whole_field()
        points = _random_points(mask, 250, seed=11)
        fixed = ripley_k(points, mask, RADII)
        naive = ripley_k(points, mask, RADII, correction="none")
        assert abs(fixed["l_minus_r"].iloc[-1]) < abs(
            naive["l_minus_r"].iloc[-1])
        assert abs(fixed["l_minus_r"]).max() < 1.5

    def test_the_frame_edge_counts_as_a_boundary(self):
        """A mask filling the image is cut off by the image.

        Without the pad in `_distance_to_boundary` the transform finds no
        background to measure to, every pixel is scored as deep inside,
        and no point is ever dropped however large r gets.
        """
        mask = _whole_field()
        points = _random_points(mask, 250, seed=11)
        got = ripley_k(points, mask, RADII)
        counts = got["n_contributing"].to_numpy()
        assert counts[0] > counts[-1]
        assert counts[-1] < len(points)

    def test_the_correction_shrinks_the_evidence_and_says_so(self):
        mask = _whole_field()
        points = _random_points(mask, 250, seed=11)
        got = ripley_k(points, mask, RADII)
        assert list(got["n_contributing"]) == sorted(
            got["n_contributing"], reverse=True)
        assert (ripley_k(points, mask, RADII, correction="none")
                ["n_contributing"] == len(points)).all()

    def test_a_radius_no_point_survives_is_nan_not_zero(self):
        mask = _whole_field((40, 40))
        points = _random_points(mask, 30, seed=2)
        got = ripley_k(points, mask, [30.0])
        assert got["n_contributing"].iloc[0] == 0
        assert np.isnan(got["k"].iloc[0])


class TestTheNullRespectsTheMask:

    def test_the_envelope_is_scattered_in_the_mask_not_its_box(self):
        """A disc null and a square null are not the same null.

        The disc's area is under 64% of its bounding box, so a null that
        used the box would sit at a different density and the envelope
        would be visibly different.
        """
        disc, box = _disc(), _whole_field()
        in_disc = csr_envelope(disc, 200, RADII, simulations=15, seed=1)
        in_box = csr_envelope(box, 200, RADII, simulations=15, seed=1)
        assert not np.allclose(in_disc["mean"], in_box["mean"])

    def test_the_null_holds_no_duplicated_points(self):
        """A repeated point is a pair at distance zero.

        Sampling with replacement would put clustering into the null
        itself, raising the envelope at precisely the small radii the
        observation is judged at, and hiding real foci.
        """
        mask = _whole_field()
        loose = csr_envelope(mask, 250, [2.0], simulations=40, seed=4)
        assert loose["hi"].iloc[0] < 1.0

    def test_more_points_than_pixels_still_answers(self):
        tiny = np.zeros((20, 20), bool)
        tiny[5:8, 5:8] = True
        got = csr_envelope(tiny, 50, [2.0], simulations=5, seed=1)
        assert len(got) == 1

    def test_the_envelope_brackets_its_own_mean(self):
        mask = _whole_field()
        got = csr_envelope(mask, 200, RADII, simulations=20, seed=6)
        assert (got["lo"] <= got["mean"]).all()
        assert (got["mean"] <= got["hi"]).all()


class TestItRefusesRatherThanGuesses:

    def test_a_point_outside_the_window_is_an_error(self):
        """Silently dropping it is the same bias with the evidence gone."""
        mask = _disc()
        with pytest.raises(ValueError, match="outside the mask"):
            ripley_k(np.array([[1, 1]]), mask, RADII)

    def test_a_point_off_the_image_is_the_same_error(self):
        with pytest.raises(ValueError, match="outside the mask"):
            ripley_k(np.array([[500, 500]]), _whole_field(), RADII)

    def test_a_volume_is_refused_because_l_is_planar(self):
        with pytest.raises(ValueError, match="planar"):
            ripley_k(np.zeros((0, 2)), np.ones((8, 8, 8), bool), RADII)

    def test_a_zero_radius_is_refused_with_the_reason(self):
        with pytest.raises(ValueError, match="carries no information"):
            ripley_k(np.zeros((0, 2)), _whole_field(), [0.0, 4.0])

    def test_no_radii_is_refused(self):
        with pytest.raises(ValueError, match="no radii"):
            ripley_k(np.zeros((0, 2)), _whole_field(), [])

    def test_one_point_has_no_pattern_and_does_not_crash(self):
        got = ripley_k(np.array([[100, 100]]), _whole_field(), RADII)
        assert got["k"].isna().all()
        assert (got["n_contributing"] == 1).all()

    def test_an_empty_well_is_nan_rather_than_zero(self):
        """Zero would read as a measured absence of clustering."""
        got = ripley_k(np.zeros((0, 2)), _whole_field(), RADII)
        assert got["l_minus_r"].isna().all()


class TestTheRadiiAndTheUnits:

    def test_the_suggested_radii_stop_a_quarter_of_the_way_across(self):
        got = suggest_radii(np.ones((400, 200), bool))
        assert got[-1] == pytest.approx(MAX_RADIUS_FRACTION * 200)
        assert got[0] > 0

    def test_they_are_lengths_when_spacing_is_given(self):
        got = suggest_radii(np.ones((400, 200), bool), spacing=0.5)
        assert got[-1] == pytest.approx(MAX_RADIUS_FRACTION * 100)

    def test_spacing_changes_the_numbers_not_the_verdict(self):
        """Half the pixel size, half the length: the same pattern.

        L - r is in image units, so it halves with the spacing; whether
        the well is clustered does not depend on the objective.
        """
        mask = _whole_field()
        points = _clustered_points(mask, 250, seed=11)
        pixels = ripley_k(points, mask, RADII)
        microns = ripley_k(points, mask, RADII / 2.0, spacing=0.5)
        assert np.allclose(microns["l_minus_r"] * 2.0, pixels["l_minus_r"])

    def test_an_unusable_spacing_falls_back_to_pixels(self):
        mask = _whole_field()
        points = _random_points(mask, 100, seed=1)
        assert np.allclose(ripley_k(points, mask, RADII, spacing=0)["k"],
                           ripley_k(points, mask, RADII)["k"])

    def test_the_default_simulation_count_gives_a_round_test_level(self):
        assert 2.0 / (DEFAULT_SIMULATIONS + 1) == pytest.approx(0.02)


class TestTheShapeOfTheAnswer:

    def test_the_test_reports_both_sides_and_the_envelope(self):
        mask = _whole_field()
        got = ripley_test(_random_points(mask, 150, seed=8), mask, RADII,
                          simulations=19, seed=2)
        assert list(got.columns) == ["r", "k", "l", "l_minus_r",
                                     "n_contributing", "lo", "hi", "mean",
                                     "clustered", "dispersed"]

    def test_it_picks_its_own_radii_when_asked_for_none(self):
        mask = _whole_field()
        got = ripley_test(_random_points(mask, 150, seed=8), mask,
                          simulations=5, seed=2)
        assert len(got) == 20
        assert got["r"].iloc[-1] == pytest.approx(MAX_RADIUS_FRACTION * 200)

    def test_l_is_the_square_root_of_k_over_pi(self):
        mask = _whole_field()
        got = ripley_k(_random_points(mask, 150, seed=8), mask, RADII)
        assert np.allclose(got["l"], np.sqrt(got["k"] / np.pi))

    def test_the_same_seed_gives_the_same_envelope(self):
        mask = _whole_field()
        one = csr_envelope(mask, 100, RADII, simulations=9, seed=42)
        two = csr_envelope(mask, 100, RADII, simulations=9, seed=42)
        assert np.allclose(one["hi"], two["hi"])
