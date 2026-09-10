"""Phase A4: putting the phenotype acquisition on the sequencing one.

WHAT THIS STEP IS. The phenotype images are 2960x2960 at 20X and the
sequencing images are 1480x1480 at 10X, of the same round well. Phase A4 of
instruction 372 asks for one transform per phenotype field into the
sequencing frame, aligned on the nuclear stain, "and do not resample one
into the other -- aligned, not averaged".

WHY IT IS NOT A PIXEL PROBLEM. Across a magnification change and two
different acquisitions a pixel neighbourhood means nothing, but a nucleus is
a nucleus at both scales. So the alignment runs on NUCLEAR CENTROIDS, and
:func:`spacr.ops_merge.align_by_triangles` proposes the transform from their
arrangement alone.

WHY THAT PROPOSAL IS NOT THE ANSWER. Measured on the real plate, the
triangle consensus is biased and the bias grows as the correspondence
thins. Synthetic controls built from 593 real nuclei, transform known
exactly:

    what was asked for              scale returned    inliers
    identity                            0.9878          56 %
    scale 0.25                          0.2469         100 %
    scale 0.25, 1 px of jitter          0.2065          53 %
    scale 0.25, half the points kept    0.2916          23 %
    scale 0.25, a third kept            0.3596          15 %

A 1.2 % scale error is 18 px of displacement across a 1480 px field, which
is why an EXACT copy of a point set scores 56 % rather than 100. And the
last two rows are the shape of every real pair this module was first tried
on -- two tiles that partially overlap, imaged twice, are exactly "some of
the points, moved a little".

SO THE CONSENSUS IS A SEED AND NOT A RESULT. This module takes it, finds the
correspondences it implies, and RE-SOLVES the similarity from those
correspondences in closed form; then does it again on the correspondences
the better transform implies. That is the same argument
:func:`align_by_triangles` already makes for its translation -- "not a
refinement, it is the difference between working and not" -- applied to the
scale and the rotation as well.

AND IT REPORTS HOW MANY NUCLEI AGREE, which the triangle consensus does not.
`align_by_triangles` returned a transform for every pair it was ever handed,
including phenotype tiles and sequencing tiles from opposite sides of the
well, because three coincidentally similar triangles can always be found in
two sets of a few hundred points. An alignment with no inlier count is not
evidence, and PART 6-A of 372 records the same lesson about placements: a
count of things attempted is not a measure of anything succeeding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Alignment",
    "align_phenotype_to_sbs",
    "nuclear_points",
    "phenotype_site_map",
    "refine_similarity",
    "seed_by_scaled_pairs",
    "similarity_from_correspondences",
]

#: How close a moved nucleus must land to one in the target frame to count
#: as the same nucleus, in TARGET pixels.
#:
#: THREE, NOT SIX, and the difference is the whole discrimination. Six is a
#: nucleus's own radius at 10X, which sounds like the honest scale for "the
#: same object" and is far too generous for deciding whether two fields
#: overlap: the target points sit about 21 px apart, so a 6 px disc covers
#: a quarter of the field and a quarter of ANY transform's points land in
#: one. Halving the radius quarters that -- and the residual of a genuine
#: alignment is under 1 px, so nothing real is refused by the change.
MATCH_RADIUS_PX: float = 3.0

#: How many refinement rounds. Two is not a guess: the seed is far enough
#: out that round one changes the correspondence set substantially and round
#: two changes it hardly at all, which is where iterating stops paying.
REFINE_ROUNDS: int = 2

#: How far the two acquisitions may be rotated from each other, in degrees.
#:
#: A PHYSICAL CONSTRAINT, NOT A TUNING KNOB, and it is the one that makes
#: this step possible at all. The plate sits in the same holder for both
#: acquisitions; the stage translates and the objective changes, and
#: neither turns the specimen. A few degrees covers the holder's own slop.
#:
#: Without it the search finds SYMMETRIC FALSE SOLUTIONS and they score
#: well. Measured on 593 real nuclei with the transform known exactly: a
#: solution 148 degrees out collected 194 agreeing nuclei at a median
#: residual of 3.9 px while being 381 px wrong. A dense, roughly uniform
#: point field is full of coincidences -- at 0.25 scale the target spacing
#: is about 21 px, so a 6 px radius catches a quarter of the points by
#: chance whatever the transform is. Inlier count alone cannot tell that
#: apart from an answer; the physics can.
MAX_ROTATION_DEGREES: float = 15.0

#: How far the recovered scale may sit from the expected one, as a
#: fraction, when the acquisition supplies an expected scale.
#:
#: The magnification ratio is an optical constant, not a measurement. The
#: same experiment as above: a solution at scale 0.2069 against a true 0.25
#: gathered 319 agreeing nuclei -- MORE than the correct answer -- because
#: a transform that shrinks the field crowds the points and manufactures
#: coincidences. Scoring by inliers rewards exactly that, so the scale has
#: to be bounded rather than scored.
SCALE_TOLERANCE: float = 0.10

#: The fewest agreeing nuclei an alignment may rest on.
#:
#: NOT A CONFIDENCE, A FLOOR. Below this the answer is one small cluster of
#: points agreeing with itself, which is what a coincidental triangle match
#: looks like after refinement too.
MIN_INLIERS: int = 40

#: How many times the chance agreement rate the real one must beat.
#:
#: A FIXED INLIER COUNT IS THE WRONG INSTRUMENT and this is what replaces
#: it. Whether 45 agreeing nuclei is a lot depends entirely on how crowded
#: the target field is: at 0.25 scale the target points sit about 21 px
#: apart, so a 6 px radius catches a quarter of anything thrown at it, and
#: an unrelated point cloud scored 45 inliers at a median residual of
#: 3.6 px -- indistinguishable, by count, from a real alignment of a
#: sparse field.
#:
#: So the count is compared to what it would be BY CHANCE, computed from
#: the target's own density and the match radius. Three times chance is a
#: wide margin and still refuses every unrelated pair measured.
MIN_CHANCE_RATIO: float = 3.0


@dataclass(frozen=True)
class Alignment:
    """One phenotype field placed in the sequencing frame.

    :param scale: target pixels per source pixel.
    :param rotation: the 2x2 rotation, source frame to target frame.
    :param translation: the offset, applied after scale and rotation.
    :param inliers: how many nuclei agree, at :data:`MATCH_RADIUS_PX`.
    :param residual_px: the median distance between agreeing nuclei, in
        target pixels. The number to read when asking how good it is;
        `inliers` says whether to believe it at all.
    :param points: how many source nuclei were offered.
    """

    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    inliers: int
    residual_px: float
    points: int

    @property
    def degrees(self) -> float:
        """The rotation in degrees, for a person reading a table."""
        return float(np.degrees(np.arctan2(self.rotation[1, 0],
                                           self.rotation[0, 0])))

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Move ``(N, 2)`` source coordinates into the target frame.

        The stored transform, applied -- so a caller never has to
        reconstruct the convention and get it the wrong way round, which is
        what cost 372's PART 11-C a canvas one pitch per column too large.
        """
        source = np.asarray(points, float)
        return self.scale * (self.rotation @ source.T).T + self.translation


def nuclear_points(image, *, sigma: float = 3.0, min_distance: int = 8,
                   top: int = 600, percentile: float = 99.0) -> np.ndarray:
    """Nuclear centres from a nuclear stain, without segmenting anything.

    :param image: one 2-D plane of a nuclear channel.
    :param sigma: Gaussian smoothing before peak finding; a nucleus is a
        blob, and smoothing at its own scale is what stops one nucleus
        answering as three.
    :param min_distance: the closest two accepted peaks may be, in pixels.
    :param top: keep at most this many, brightest first. The triangle
        matcher is quadratic in the point count and a field's brightest few
        hundred nuclei are as informative as all of them.
    :param percentile: the intensity floor, as a percentile of the smoothed
        image. Relative rather than absolute, because the two acquisitions
        this module compares were taken at different exposures.
    :returns: ``(N, 2)`` row/column centres.

    A4 DOES NOT NEED MASKS. It needs points that mean the same thing in two
    frames, and a smoothed local maximum on a nuclear stain is that at a
    fraction of a segmentation's cost. Phase B segments, once, on the
    composite; this is not that step and must not become it.
    """
    from scipy import ndimage
    from skimage.feature import peak_local_max

    plane = np.asarray(image, np.float32)
    smooth = ndimage.gaussian_filter(plane, sigma)
    peaks = peak_local_max(smooth, min_distance=int(min_distance),
                           threshold_abs=float(np.percentile(smooth,
                                                             percentile)))
    if peaks.shape[0] > top:
        brightest = np.argsort(smooth[peaks[:, 0], peaks[:, 1]])[::-1][:top]
        peaks = peaks[brightest]
    return peaks.astype(float)


def similarity_from_correspondences(source: np.ndarray, target: np.ndarray):
    """The best similarity taking ``source`` onto ``target``, in closed form.

    :param source: ``(N, 2)`` points.
    :param target: ``(N, 2)`` points, in the same order.
    :returns: ``(scale, rotation, translation)``, or None for fewer than
        two points or a degenerate source.

    Umeyama's solution: the least-squares scale, rotation and translation
    over every correspondence at once. Reflections are refused -- two
    acquisitions of one well differ by a rigid motion and a magnification,
    never by a mirror, and allowing one lets a bad correspondence set
    produce a transform that fits beautifully and means nothing.
    """
    src = np.asarray(source, float)
    dst = np.asarray(target, float)
    if src.shape != dst.shape or src.shape[0] < 2:
        return None
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centred = src - src_mean
    dst_centred = dst - dst_mean
    variance = float((src_centred ** 2).sum() / src.shape[0])
    if not np.isfinite(variance) or variance <= 0.0:
        return None
    covariance = (dst_centred.T @ src_centred) / src.shape[0]
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0.0:
        correction[1, 1] = -1.0
    rotation = u @ correction @ vt
    scale = float((singular * np.diag(correction)).sum() / variance)
    if not np.isfinite(scale) or scale <= 0.0:
        return None
    translation = dst_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation


def _chance_inliers(source_count: int, target: np.ndarray,
                    radius: float) -> float:
    """How many agreements an arbitrary transform would collect.

    :param source_count: how many points are being moved.
    :param target: the ``(M, 2)`` points being moved onto.
    :param radius: the agreement radius.
    :returns: the expected number of coincidental agreements.

    The target points occupy an area; a moved point agrees by chance if it
    lands within ``radius`` of any of them, which is that many discs over
    that area. Computed from the target's own extent rather than assumed,
    because a field of 600 nuclei and a field of 60 are different problems
    and the same inlier count means different things in each.
    """
    points = np.asarray(target, float)
    if points.shape[0] < 3 or source_count < 1:
        return 0.0
    height = float(points[:, 0].max() - points[:, 0].min())
    width = float(points[:, 1].max() - points[:, 1].min())
    area = max(1.0, height * width)
    covered = min(1.0, points.shape[0] * np.pi * radius * radius / area)
    return float(source_count) * covered


def _correspondences(source: np.ndarray, target: np.ndarray, scale: float,
                     rotation: np.ndarray, translation: np.ndarray,
                     radius: float):
    """Source and target rows that land within ``radius`` of each other."""
    from scipy.spatial import cKDTree

    moved = scale * (rotation @ np.asarray(source, float).T).T + translation
    distance, index = cKDTree(np.asarray(target, float)).query(moved, k=1)
    close = distance <= radius
    return (np.asarray(source, float)[close],
            np.asarray(target, float)[index[close]],
            distance[close])


def refine_similarity(source: np.ndarray, target: np.ndarray, seed,
                      *, radius: float = MATCH_RADIUS_PX,
                      rounds: int = REFINE_ROUNDS):
    """Re-solve a seed transform from the correspondences it implies.

    :param source: ``(N, 2)`` points in the frame being moved.
    :param target: ``(M, 2)`` points in the frame being moved to.
    :param seed: ``(scale, rotation, translation)`` to start from.
    :param radius: how close a moved point must land to count as a match.
    :param rounds: how many times to re-solve.
    :returns: ``(scale, rotation, translation, source_in, target_in,
        distances)``, or None when a round finds too few correspondences to
        solve from.

    THE SEED IS ALLOWED TO BE POOR. That is the point: the triangle
    consensus is biased by up to 44 % in scale on sparse data, and a bias
    that large still puts most of the field within a few pixels of the
    truth near the centre of rotation, which is enough correspondence to
    solve properly from.
    """
    scale, rotation, translation = seed
    result = None
    # WIDE FIRST, THEN TIGHT. The seed is poor by construction, so the
    # first pass has to accept correspondences a long way out or there is
    # nothing to re-solve from; the last pass has to accept only close ones
    # or it re-solves from coincidences. One radius cannot be both, and
    # using the scoring radius throughout refused every noisy case that
    # this schedule now recovers. The final pass is always at `radius`, so
    # what comes back is measured at the radius the caller asked for.
    widths = [radius * 4.0, radius * 2.0]
    schedule = widths + [radius] * max(1, int(rounds))
    for width in schedule:
        src_in, dst_in, _distance = _correspondences(
            source, target, scale, rotation, translation, width)
        if src_in.shape[0] < 2:
            return result
        solved = similarity_from_correspondences(src_in, dst_in)
        if solved is None:
            return result
        scale, rotation, translation = solved
        src_in, dst_in, distance = _correspondences(
            source, target, scale, rotation, translation, radius)
        result = (scale, rotation, translation, src_in, dst_in, distance)
    return result


def align_phenotype_to_sbs(phenotype_points: np.ndarray,
                           sbs_points: np.ndarray, *,
                           expected_scale: Optional[float] = None,
                           tolerance: float = 0.02,
                           radius: float = MATCH_RADIUS_PX,
                           rounds: int = REFINE_ROUNDS,
                           min_inliers: int = MIN_INLIERS,
                           max_rotation: float = MAX_ROTATION_DEGREES,
                           scale_tolerance: float = SCALE_TOLERANCE,
                           min_chance_ratio: float = MIN_CHANCE_RATIO
                           ) -> Optional[Alignment]:
    """One phenotype field's transform into the sequencing frame.

    :param phenotype_points: ``(N, 2)`` nuclear centres in the phenotype
        field.
    :param sbs_points: ``(M, 2)`` nuclear centres in the sequencing field.
    :param expected_scale: target pixels per source pixel, when the
        acquisition says. Given, a second seed is tried with the scale
        fixed -- see :func:`seed_by_scaled_pairs` for why that matters
        more than it sounds.
    :param tolerance: triangle-shape tolerance for the consensus seed.
    :param radius: how close a moved nucleus must land to agree.
    :param rounds: refinement rounds.
    :param min_inliers: the fewest agreeing nuclei to return anything at
        all.
    :param max_rotation: the largest rotation the two acquisitions may
        differ by, in degrees.
    :param scale_tolerance: how far the answer's scale may sit from
        ``expected_scale``, as a fraction. Ignored without one.
    :param min_chance_ratio: how many times the chance agreement rate the
        real one must beat. See :data:`MIN_CHANCE_RATIO`.
    :returns: the :class:`Alignment`, or None -- which is what "these two
        fields do not overlap" looks like, and it has to be sayable.

    TWO SEEDS, AND THE ONE WITH MORE NUCLEI BEHIND IT WINS. Refinement
    converges to whatever basin its seed is in, so trying one seed and
    refining it is a decision made before any evidence is in. Scoring both
    on the same measure -- how many nuclei agree -- is not.

    BUT INLIERS ONLY DECIDE BETWEEN CANDIDATES THAT ARE PHYSICALLY
    POSSIBLE. Scored on inliers alone, a wrong answer wins: a solution
    148 degrees out gathered 194 nuclei, and one at scale 0.2069 against a
    true 0.25 gathered 319 -- more than the truth -- because shrinking the
    field crowds the points. Both are excluded here before they are
    scored, by the rotation and scale bounds, and both bounds are facts
    about the microscope rather than thresholds anybody tuned.
    """
    from .ops_merge import align_by_triangles

    source = np.asarray(phenotype_points, float)
    target = np.asarray(sbs_points, float)
    if source.shape[0] < 3 or target.shape[0] < 3:
        return None

    seeds = []
    consensus = align_by_triangles(source, target, tolerance=tolerance)
    if consensus is not None:
        seeds.append(consensus)
    if expected_scale:
        paired = seed_by_scaled_pairs(source, target, float(expected_scale),
                                      radius=radius,
                                      max_rotation=max_rotation)
        if paired is not None:
            seeds.append(paired)
    if not seeds:
        return None

    best = None
    for seed in seeds:
        refined = refine_similarity(source, target, seed, radius=radius,
                                    rounds=rounds)
        if refined is None:
            continue
        scale, rotation, translation, src_in, _dst_in, distance = refined
        degrees = float(np.degrees(np.arctan2(rotation[1, 0],
                                              rotation[0, 0])))
        if abs(degrees) > float(max_rotation):
            continue
        if expected_scale and abs(scale - float(expected_scale)) > (
                float(expected_scale) * float(scale_tolerance)):
            continue
        if best is None or src_in.shape[0] > best[3].shape[0]:
            best = (scale, rotation, translation, src_in, distance)
    if best is None:
        return None
    scale, rotation, translation, src_in, distance = best
    if src_in.shape[0] < int(min_inliers):
        return None
    if src_in.shape[0] < min_chance_ratio * _chance_inliers(
            source.shape[0], target, radius):
        return None
    return Alignment(scale=float(scale), rotation=np.asarray(rotation, float),
                     translation=np.asarray(translation, float),
                     inliers=int(src_in.shape[0]),
                     residual_px=float(np.median(distance))
                     if distance.size else float("inf"),
                     points=int(source.shape[0]))


#: How many two-point correspondences to try when the scale is known.
#:
#: Each trial is a KD-tree query over a few hundred points, so the whole
#: search costs a fraction of a second. Three hundred is where the best
#: score stopped improving on the real fields this was measured against.
PAIR_TRIALS: int = 300

#: How far a target pair's separation may be from the expected one, as a
#: fraction. Two acquisitions of one well differ by a magnification and a
#: rigid motion, so a correspondence's separation is fixed by the scale --
#: this is the tolerance on the stage and the centroid, not on the optics.
PAIR_TOLERANCE: float = 0.04


def seed_by_scaled_pairs(source: np.ndarray, target: np.ndarray,
                         scale: float, *, radius: float = MATCH_RADIUS_PX,
                         trials: int = PAIR_TRIALS,
                         tolerance: float = PAIR_TOLERANCE,
                         max_rotation: float = MAX_ROTATION_DEGREES,
                         seed: int = 0):
    """A seed transform from two-point correspondences, with the scale known.

    :param source: ``(N, 2)`` points to be moved.
    :param target: ``(M, 2)`` points to move to.
    :param scale: target pixels per source pixel, from the acquisition.
    :param radius: how close a moved point must land to agree.
    :param trials: how many correspondences to try.
    :param tolerance: fractional slack on a pair's separation.
    :param max_rotation: the largest rotation to consider, in degrees. See
        :data:`MAX_ROTATION_DEGREES` -- this is what stops the search
        returning a symmetric coincidence with a good-looking score.
    :param seed: the random seed, so a run is reproducible.
    :returns: ``(scale, rotation, translation)`` for the best-scoring
        correspondence, or None if none scored above two points.

    WHY THIS EXISTS RATHER THAN THE TRIANGLE CONSENSUS ALONE. The consensus
    estimates the scale, and on sparse or noisy correspondence it estimates
    it badly -- 0.36 where the truth was 0.25, measured. Refinement cannot
    recover from that: a transform that wrong finds a small self-consistent
    cluster of points near its own fixed point and converges happily onto
    it. The scale, though, is not something A4 has to estimate. It is the
    ratio of the two acquisitions' tile pitches, and the layout knows it.

    So this fixes the scale and searches only rotation and translation,
    which two matched points determine exactly. A source pair separated by
    ``d`` must land on a target pair separated by ``scale * d``, and that
    single constraint is enough to make random sampling cheap: for a
    sampled target point, only its neighbours at the right distance can be
    the partner.

    SCORED BY INLIERS, NOT BY RESIDUAL. A transform that maps four points
    onto four points perfectly beats nothing; one that maps two hundred
    within a nucleus's radius is the answer.
    """
    from scipy.spatial import cKDTree

    src = np.asarray(source, float)
    dst = np.asarray(target, float)
    if src.shape[0] < 2 or dst.shape[0] < 2:
        return None
    tree = cKDTree(dst)
    rng = np.random.default_rng(seed)
    best = None
    best_score = 1
    # A SEED IS SCORED LOOSELY, and this is not the same number the answer
    # is scored with. Two points fix a transform exactly at those two
    # points and approximately everywhere else, and the error grows with
    # the distance from them -- so scoring a seed at the final radius asks
    # it to already be the answer. It only has to be in the right basin.
    seed_radius = radius * 4.0
    # AND A SEED NEEDS A BASELINE. A pair of nuclei 10 px apart fixes the
    # rotation to within the centroid noise, which is to say not at all;
    # the same noise on a pair a third of the field apart is a fraction of
    # a degree. Short pairs are also where the separation test stops
    # discriminating, because almost any target pair passes it.
    extent = float(max(src[:, 0].ptp(), src[:, 1].ptp()))
    min_span = 0.2 * extent
    for _ in range(max(1, int(trials))):
        i, j = rng.choice(src.shape[0], size=2, replace=False)
        span = float(np.hypot(*(src[j] - src[i])))
        if span < min_span:
            continue
        want = scale * span
        k = int(rng.integers(dst.shape[0]))
        near = tree.query_ball_point(dst[k], want * (1.0 + tolerance))
        for l in near:
            if l == k:
                continue
            got = float(np.hypot(*(dst[l] - dst[k])))
            if abs(got - want) > want * tolerance:
                continue
            # Two points fix the rotation: the angle between the source
            # separation and the target separation.
            #
            # BOTH ANGLES ARE TAKEN THE SAME WAY ROUND, and it has to be
            # the way the rotation matrix below reads. These are (row,
            # column) points, so a separation is (dy, dx) and the angle
            # that matches `[[cos, -sin], [sin, cos]]` acting on (row,
            # column) is `arctan2(dy, dx)`. Written as `arctan2(dx, dy)`
            # the two angles are each the complement of the right one, so
            # their difference comes out NEGATED -- a seed rotated the
            # wrong way, which the refinement then has to climb out of.
            source_angle = np.arctan2(*(src[j] - src[i]))
            target_angle = np.arctan2(*(dst[l] - dst[k]))
            angle = target_angle - source_angle
            angle = (angle + np.pi) % (2.0 * np.pi) - np.pi
            if abs(np.degrees(angle)) > max_rotation:
                continue
            rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])
            translation = dst[k] - scale * (rotation @ src[i])
            moved = scale * (rotation @ src.T).T + translation
            distance, _ = tree.query(moved, k=1)
            score = int(np.sum(distance <= seed_radius))
            if score > best_score:
                best_score = score
                best = (float(scale), rotation, translation)
    return best


def phenotype_site_map(phenotype_sites: int, sbs_sites: int
                       ) -> Dict[int, int]:
    """Which sequencing tile covers each phenotype tile.

    :param phenotype_sites: how many fields the phenotype acquisition holds.
    :param sbs_sites: how many the sequencing acquisition holds.
    :returns: ``{phenotype site -> sequencing site}``, omitting any
        phenotype tile whose sequencing position is outside that circle.
    :raises ValueError: when either count is not a round well, which
        :func:`spacr.ops_layout.round_well_layout` decides.

    IT IS A LAYOUT QUESTION, NOT AN IMAGE ONE, and answering it by image
    was how the first three attempts at A4 spent an afternoon: phenotype
    site N was matched against sequencing site N, which is a different part
    of the well, and every pair came back at 2-9 % inliers. The layout says
    it in closed form. The measured acquisition fits 41 columns at radius
    20.15 against the sequencing well's 21 at 10.25 -- the same round well
    at twice the tile density, and 41 = 2 * 21 - 1 -- so a phenotype grid
    position halves about the centre.
    """
    from .ops_layout import round_well_layout

    phenotype = round_well_layout(int(phenotype_sites))
    sbs = round_well_layout(int(sbs_sites))
    ratio = (phenotype.columns - 1) / max(1, (sbs.columns - 1))
    p_col, p_row = phenotype.centre
    s_col, s_row = sbs.centre
    mapping: Dict[int, int] = {}
    for site in range(phenotype.site_count):
        column, row = phenotype.position(site)
        target_column = int(round(s_col + (column - p_col) / ratio))
        target_row = int(round(s_row + (row - p_row) / ratio))
        found = sbs.site(target_column, target_row)
        if found is not None:
            mapping[site] = found
    return mapping
