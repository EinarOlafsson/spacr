"""Instruction 372, Phase A4: the phenotype acquisition put on the SBS one.

THE TESTS ARE ABOUT REFUSING, as much as about aligning. `align_by_triangles`
alone returns a transform for every pair of point sets it is ever handed --
three coincidentally similar triangles exist in any two sets of a few hundred
points -- so the first eleven real phenotype/sequencing pairs it was tried on
all "aligned", at 2-9 % agreement, and none of them overlapped. An alignment
that cannot say no is not an alignment.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.ops_phenotype import (Alignment, MATCH_RADIUS_PX, align_phenotype_to_sbs,
                                 phenotype_site_map, refine_similarity,
                                 seed_by_scaled_pairs,
                                 similarity_from_correspondences)


def _field(count=400, extent=1480.0, seed=0):
    """A field of points standing in for nuclei."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, extent, size=(count, 2))


def _move(points, scale, degrees, shift, jitter=0.0, seed=1):
    angle = np.radians(degrees)
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    moved = scale * (rotation @ np.asarray(points, float).T).T + np.asarray(shift)
    if jitter:
        moved = moved + np.random.default_rng(seed).normal(0.0, jitter,
                                                           moved.shape)
    return moved


# ---------------------------------------------------------------------------
# The closed-form solve
# ---------------------------------------------------------------------------

class TestSolvingASimilarity:

    def test_it_recovers_what_it_was_given(self):
        source = _field(60)
        target = _move(source, 0.25, 7.0, (13.0, -21.0))
        scale, rotation, translation = similarity_from_correspondences(
            source, target)
        assert scale == pytest.approx(0.25, rel=1e-9)
        assert np.degrees(np.arctan2(rotation[1, 0],
                                     rotation[0, 0])) == pytest.approx(7.0)
        assert translation == pytest.approx(np.array([13.0, -21.0]))

    def test_it_refuses_a_reflection(self):
        """Two acquisitions of one well differ by a rigid motion and a
        magnification, never by a mirror.

        Allowing one lets a bad correspondence set produce a transform that
        fits beautifully and means nothing, so the solve is constrained to
        proper rotations and a mirrored target is answered with the best
        NON-mirrored fit rather than an exact mirrored one.
        """
        source = _field(60)
        mirrored = source.copy()
        mirrored[:, 1] = -mirrored[:, 1]
        scale, rotation, _translation = similarity_from_correspondences(
            source, mirrored)
        assert np.linalg.det(rotation) == pytest.approx(1.0), (
            "the solve returned a reflection")

    def test_too_few_points_is_none_rather_than_a_guess(self):
        assert similarity_from_correspondences(np.zeros((1, 2)),
                                               np.zeros((1, 2))) is None

    def test_a_degenerate_source_is_none(self):
        """Every source point at one place fixes no scale at all."""
        same = np.zeros((8, 2))
        assert similarity_from_correspondences(same, _field(8)) is None


# ---------------------------------------------------------------------------
# The refinement, which is what makes a poor seed usable
# ---------------------------------------------------------------------------

class TestRefiningFromASeed:

    def test_a_slightly_wrong_seed_is_pulled_onto_the_answer(self):
        """A seed near the answer converges onto it.

        NEAR MEANS NEAR IN EVERY PARAMETER, and the rotation is the tight
        one. Across a 370 px target field a 2 degree error displaces the
        far corner by 13 px, which is already outside the widest pass of
        the schedule -- so a seed with the right rotation and a 1 % scale
        error converges, and one with the right scale and no rotation does
        not. That is why the two-point seed, which fixes the rotation
        exactly from a correspondence, is what feeds this in practice.
        """
        source = _field(300, seed=3)
        angle = np.radians(2.0)
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        target = _move(source, 0.25, 2.0, (9.0, 4.0))
        refined = refine_similarity(
            source, target, (0.2525, rotation, np.array([9.0, 4.0])))
        assert refined is not None
        assert refined[0] == pytest.approx(0.25, rel=1e-3)
        assert refined[3].shape[0] > 250, (
            "most of the field should agree once it has converged")

    def test_a_badly_wrong_seed_is_NOT_corrected_and_that_is_why(self):
        """THE LIMIT OF REFINEMENT, pinned so nobody relies on more.

        `align_by_triangles` returned 0.36 against a true 0.25 on sparse
        correspondence -- a 44 % error. Refinement cannot recover from
        that, and this test says so rather than leaving it to be
        rediscovered: a scale that wrong finds a small self-consistent
        cluster of points near its own fixed point and converges happily
        onto it.

        That is the entire reason `seed_by_scaled_pairs` exists. The scale
        is not something A4 has to estimate; it is the ratio of the two
        acquisitions' tile pitches, and the layout knows it.
        """
        source = _field(300, seed=3)
        target = _move(source, 0.25, 2.0, (9.0, 4.0))
        refined = refine_similarity(source, target,
                                    (0.36, np.eye(2), np.array([9.0, 4.0])))
        assert refined is None or abs(refined[0] - 0.25) > 0.01

    def test_the_last_pass_is_at_the_radius_that_was_asked_for(self):
        """The schedule starts wide so a poor seed has something to solve
        from, and ends at the caller's radius so what comes back is
        measured the way the caller asked."""
        source = _field(200, seed=4)
        target = _move(source, 0.25, 1.0, (3.0, 3.0))
        refined = refine_similarity(source, target, (0.25, np.eye(2),
                                                     np.array([3.0, 3.0])),
                                    radius=2.0)
        assert refined is not None
        distances = refined[5]
        assert distances.size and float(distances.max()) <= 2.0

    def test_a_seed_that_matches_nothing_gives_nothing(self):
        source = _field(50, seed=5)
        target = _field(50, seed=6) + 1e6
        assert refine_similarity(source, target,
                                 (1.0, np.eye(2), np.zeros(2))) is None


# ---------------------------------------------------------------------------
# The two-point seed, which is where the known scale earns its keep
# ---------------------------------------------------------------------------

class TestSeedingWithTheScaleKnown:

    def test_it_finds_a_transform_it_can_be_given(self):
        source = _field(250, seed=7)
        target = _move(source, 0.25, 4.0, (11.0, -6.0))
        seed = seed_by_scaled_pairs(source, target, 0.25)
        assert seed is not None
        scale, rotation, translation = seed
        assert scale == pytest.approx(0.25)
        moved = scale * (rotation @ source.T).T + translation
        # A SEED, NOT AN ANSWER. Two points fix the transform exactly at
        # those two points and approximately elsewhere; what it has to be
        # is in the right basin, which the refinement then closes. The
        # tolerance here is a fraction of the field, not a pixel count.
        extent = float(max(target[:, 0].ptp(), target[:, 1].ptp()))
        assert np.abs(moved - target).max() < 0.15 * extent

    def test_a_rotation_outside_the_bound_is_never_proposed(self):
        """The physical constraint, as a test.

        The plate does not turn between acquisitions, and without this the
        search finds symmetric coincidences that score well: 194 agreeing
        nuclei, 148 degrees out, 381 px wrong.
        """
        source = _field(250, seed=8)
        target = _move(source, 0.25, 150.0, (5.0, 5.0))
        seed = seed_by_scaled_pairs(source, target, 0.25, max_rotation=15.0)
        # It may still return the best coincidence it could find inside the
        # bound -- what it must never do is return the 150 degree answer,
        # however well that one scores.
        if seed is not None:
            _scale, rotation, _translation = seed
            degrees = np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))
            assert abs(degrees) <= 15.0

    def test_it_is_reproducible(self):
        source = _field(200, seed=9)
        target = _move(source, 0.25, 3.0, (7.0, 2.0))
        first = seed_by_scaled_pairs(source, target, 0.25, seed=42)
        second = seed_by_scaled_pairs(source, target, 0.25, seed=42)
        assert first is not None and second is not None
        assert first[0] == second[0]
        assert np.allclose(first[1], second[1])
        assert np.allclose(first[2], second[2])


# ---------------------------------------------------------------------------
# The whole thing, and what it refuses
# ---------------------------------------------------------------------------

class TestAligningAField:

    def test_it_aligns_a_field_it_should(self):
        source = _field(400, seed=10)
        target = _move(source, 0.2543, 1.5, (17.0, -9.0), jitter=0.4)
        fit = align_phenotype_to_sbs(source, target, expected_scale=0.2543)
        assert isinstance(fit, Alignment)
        assert fit.scale == pytest.approx(0.2543, rel=0.02)
        assert abs(fit.degrees - 1.5) < 1.0
        assert fit.inliers > 300
        assert fit.residual_px < 1.5

    def test_it_refuses_two_fields_that_have_nothing_to_do_with_each_other(self):
        """The assertion the first eleven real pairs would have failed.

        Handed a phenotype field and a sequencing field from opposite sides
        of the well, `align_by_triangles` proposed a transform every time.
        """
        source = _field(400, seed=11)
        target = _field(400, seed=12, extent=380.0)
        assert align_phenotype_to_sbs(source, target,
                                      expected_scale=0.2543) is None

    def test_it_refuses_a_scale_the_optics_cannot_produce(self):
        """The magnification ratio is a constant, not a measurement."""
        source = _field(400, seed=13)
        target = _move(source, 0.5, 1.0, (4.0, 4.0))
        assert align_phenotype_to_sbs(source, target,
                                      expected_scale=0.2543) is None

    def test_the_transform_it_returns_is_the_one_it_applies(self):
        """`apply` exists so no caller has to reconstruct the convention.

        Getting it the wrong way round is not hypothetical -- it cost
        372's PART 11-C a canvas one pitch per column too large.
        """
        source = _field(400, seed=14)
        target = _move(source, 0.2543, 1.0, (5.0, 5.0))
        fit = align_phenotype_to_sbs(source, target, expected_scale=0.2543)
        assert fit is not None
        assert np.abs(fit.apply(source) - target).max() < 1.0

    def test_too_few_points_is_refused_rather_than_fitted(self):
        assert align_phenotype_to_sbs(_field(2), _field(2)) is None


# ---------------------------------------------------------------------------
# The site correspondence, which is a layout question
# ---------------------------------------------------------------------------

class TestWhichSequencingTileCoversWhich:

    def test_the_measured_acquisition_maps(self):
        """1281 phenotype fields and 333 sequencing fields, both round.

        41 columns at radius 20.15 against 21 at radius 10.25: the same
        well at twice the tile density, and 41 = 2 * 21 - 1.
        """
        mapping = phenotype_site_map(1281, 333)
        assert len(mapping) > 1200
        assert set(mapping.values()) <= set(range(333))

    def test_the_centre_maps_to_the_centre(self):
        from spacr.ops_layout import round_well_layout

        phenotype = round_well_layout(1281)
        sbs = round_well_layout(333)
        mapping = phenotype_site_map(1281, 333)
        middle = phenotype.site(*phenotype.centre)
        assert middle is not None
        assert mapping[middle] == sbs.site(*sbs.centre)

    def test_a_count_that_is_not_a_round_well_says_so(self):
        with pytest.raises(ValueError):
            phenotype_site_map(1000, 333)
