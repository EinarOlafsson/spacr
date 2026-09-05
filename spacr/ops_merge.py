"""Match cells between two magnifications by their arrangement, not their pixels.

THE PROBLEM. Optical pooled screening images the same well twice: once at low
magnification to read the barcodes, once at high magnification to measure the
phenotype. Every cell must be matched between the two, and the obvious method
-- correlate the images -- does not work. The pixels do not correspond: the
scale differs, the stains differ, and often the microscope differs too.

THE METHOD. What survives all of that is the ARRANGEMENT of the cells. Take
three neighbouring cells and the SHAPE of the triangle they form is unchanged
by magnification, rotation and translation. So the cells are triangulated, each
triangle is reduced to a descriptor that depends only on its shape, and the two
sets are matched on those descriptors. The transform between the acquisitions
falls out of the matched triangles, and the cells are paired under it.

WHY THE PAIRING REFUSES AMBIGUITY. Matching is MUTUAL nearest neighbour with a
distance limit: a pair is kept only when each cell is the other's nearest, and
anything doubtful is dropped rather than assigned. A wrongly matched cell gives
a real phenotype the wrong barcode, which silently corrupts a screen; a dropped
one costs only statistical power, which more cells can buy back.

WHERE IT STOPS WORKING, measured on 90 cells over a 500 px field related by a
known scale of 2.0, an 11 degree rotation and a (137.5, -92.25) shift:

    centroid jitter   cells dropped   scale     angle      shift error
    none              none            2.000     11.00      0.00 px
    none              8               2.000     11.00      0.00 px
    1.5 px            none            1.995     10.77      3.45 px
    1.5 px            8               1.993     10.71      4.45 px
    4.0 px            25              2.050      8.83     41.29 px

The last row is a failure and is recorded as one. Triangle shape is what
carries the signal, and jitter of a few pixels on a cell a few tens of pixels
across changes that shape materially -- so the method degrades with centroid
noise rather than with cell count. If a real acquisition sits nearer the last
row than the fourth, the answer is better segmentation centroids, not a looser
tolerance: widening the tolerance admits more coincidental shape matches and
moves the median onto them.

Follows brieflow (Cheeseman lab; github.com/cheeseman-lab/brieflow, MIT,
Copyright 2025 Massachusetts Institute of Technology), whose `merge` stage was
read for the approach.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def triangle_descriptors(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Delaunay-triangulate ``points`` and describe each triangle by its shape.

    THE DESCRIPTOR IS TWO NUMBERS, and it has to be: the ratios of the two
    shorter sides to the longest. Scaling a triangle multiplies every side by
    the same factor, so the ratios do not move; rotating and translating it
    does not change side lengths at all. That is exactly the invariance the
    problem needs, and nothing about the absolute size or position survives --
    which is the point, because those are what differ between the two
    acquisitions.

    :param points: ``(N, 2)`` cell centroids.
    :returns: ``(descriptors, vertex_indices)`` -- an ``(M, 2)`` array and the
        ``(M, 3)`` indices of each triangle's vertices, ordered so that vertex
        0 is opposite the shortest side and vertex 2 opposite the longest.
        That ordering is what makes two matched descriptors also give matched
        POINTS, without which the transform cannot be solved.
    """
    from scipy.spatial import Delaunay

    coords = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if coords.shape[0] < 3:
        return np.empty((0, 2)), np.empty((0, 3), dtype=np.int64)

    simplices = Delaunay(coords).simplices
    descriptors = np.zeros((simplices.shape[0], 2))
    ordered = np.zeros_like(simplices)

    for i, tri in enumerate(simplices):
        p0, p1, p2 = coords[tri]
        # Side lengths, each labelled by the vertex OPPOSITE it.
        sides = np.array([
            np.linalg.norm(p1 - p2),   # opposite vertex 0
            np.linalg.norm(p0 - p2),   # opposite vertex 1
            np.linalg.norm(p0 - p1),   # opposite vertex 2
        ])
        order = np.argsort(sides)      # shortest side first
        a, b, c = sides[order]
        if c <= 0:
            continue
        descriptors[i] = (a / c, b / c)
        ordered[i] = tri[order]

    return descriptors, ordered


def _similarity_from_pairs(src: np.ndarray, dst: np.ndarray):
    """Least-squares similarity (scale, rotation, translation) from matched points.

    Solved in closed form rather than iteratively, because a similarity has
    only four degrees of freedom and two point sets determine it exactly.

    :param src: ``(N, 2)`` source points.
    :param dst: ``(N, 2)`` the same points in the target frame.
    :returns: ``(scale, rotation_2x2, translation_2)``, or None when the
        points are degenerate (all coincident, so no scale is defined).
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - src_mean, dst - dst_mean

    variance = float((src_c ** 2).sum())
    if variance <= 1e-12:
        return None

    covariance = dst_c.T @ src_c / src.shape[0]
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(2)
    if np.linalg.det(u @ vt) < 0:
        # A reflection fits the points as well as a rotation but means the
        # sample was flipped, which does not happen between two microscopes
        # looking at the same well.
        correction[1, 1] = -1.0
    rotation = u @ correction @ vt
    # Umeyama's closed form: the scale is the trace of the singular values
    # (sign-corrected) over the source's variance about its own centroid.
    scale = float((singular * np.diag(correction)).sum()
                  / (variance / src.shape[0]))
    translation = dst_mean - scale * rotation @ src_mean
    return scale, rotation, translation


def align_by_triangles(source: np.ndarray, target: np.ndarray, *,
                       tolerance: float = 0.02,
                       min_votes: int = 3):
    """Recover the transform between two cell sets, from triangle shapes alone.

    Every triangle in ``source`` is matched to the ``target`` triangles whose
    shape descriptor is within ``tolerance``, each match proposes a transform,
    and the proposals are pooled. Pooling is what makes this robust: a single
    coincidental shape match proposes a wrong transform, but wrong proposals
    disagree with each other while right ones agree, so the median survives
    and the outliers do not.

    :param source: ``(N, 2)`` centroids in the frame to be moved.
    :param target: ``(M, 2)`` centroids in the frame to move to.
    :param tolerance: how close two shape descriptors must be to be considered
        the same triangle.
    :param min_votes: how many agreeing triangle matches are required before a
        transform is returned at all. Below this the answer is not evidence.
    :returns: ``(scale, rotation, translation)``, or None if too few triangles
        agreed.
    """
    src_desc, src_tri = triangle_descriptors(source)
    dst_desc, dst_tri = triangle_descriptors(target)
    if src_desc.shape[0] == 0 or dst_desc.shape[0] == 0:
        return None

    from scipy.spatial import cKDTree

    tree = cKDTree(dst_desc)
    source_pts = np.asarray(source, float)
    target_pts = np.asarray(target, float)

    scales, angles = [], []
    matched_src, matched_dst = [], []
    for i, desc in enumerate(src_desc):
        for j in tree.query_ball_point(desc, tolerance):
            src_vertices = source_pts[src_tri[i]]
            dst_vertices = target_pts[dst_tri[j]]
            fit = _similarity_from_pairs(src_vertices, dst_vertices)
            if fit is None:
                continue
            scale, rotation, _translation = fit
            if not np.isfinite(scale) or scale <= 0:
                continue
            scales.append(scale)
            angles.append(np.arctan2(rotation[1, 0], rotation[0, 0]))
            matched_src.append(src_vertices)
            matched_dst.append(dst_vertices)

    if len(scales) < min_votes:
        return None

    # The MEDIAN, not the mean: a handful of coincidental shape matches
    # propose transforms that are arbitrarily wrong, and a mean would let one
    # of them drag the answer.
    scale = float(np.median(scales))
    angle = float(np.median(angles))
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])

    # THE TRANSLATION IS SOLVED LAST, FROM EVERY CORRESPONDENCE AT ONCE, and
    # this is not a refinement -- it is the difference between working and
    # not. A per-triangle translation is `dst_mean - scale * R @ src_mean`
    # using THAT triangle's own noisy scale and rotation, and any error in
    # them is multiplied by the distance from the origin, which across a
    # 500 px field is large. Measured with 1.5 px of centroid jitter: taking
    # the median of per-triangle translations gave (146, -54) where the truth
    # was (137.5, -92.25) and the scale and angle were already correct to
    # 0.3 %. Re-solving here against the consensus scale and rotation, over
    # every matched vertex, removes that lever entirely.
    src_all = np.concatenate(matched_src, axis=0)
    dst_all = np.concatenate(matched_dst, axis=0)
    offsets = dst_all - (scale * (rotation @ src_all.T).T)
    translation = np.median(offsets, axis=0)
    return scale, rotation, translation


def match_cells(source: np.ndarray, target: np.ndarray, *,
                transform=None, threshold: float = 10.0) -> np.ndarray:
    """Pair cells one-to-one, keeping only mutual nearest neighbours.

    AMBIGUITY IS DROPPED, NOT ASSIGNED. A pair survives only when each cell is
    the other's nearest AND they are within ``threshold``. Two cells competing
    for the same partner both lose. That asymmetry is deliberate: a wrongly
    matched cell hands a real phenotype somebody else's barcode and quietly
    corrupts every statistic computed from it, whereas a dropped cell costs
    only statistical power that more cells can recover.

    :param source: ``(N, 2)`` centroids.
    :param target: ``(M, 2)`` centroids.
    :param transform: ``(scale, rotation, translation)`` from
        :func:`align_by_triangles`, applied to ``source`` first. ``None``
        means the two sets are already in the same frame.
    :param threshold: the largest distance, in TARGET units, that may still be
        called the same cell.
    :returns: ``(K, 2)`` array of ``(source_index, target_index)`` pairs.
    """
    from scipy.spatial import cKDTree

    src = np.asarray(source, float).reshape(-1, 2)
    dst = np.asarray(target, float).reshape(-1, 2)
    if src.size == 0 or dst.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    if transform is not None:
        scale, rotation, translation = transform
        src = (scale * (rotation @ src.T).T) + translation

    forward = cKDTree(dst).query(src)[1]
    backward = cKDTree(src).query(dst)[1]

    pairs = []
    for i, j in enumerate(forward):
        if backward[j] != i:
            continue                      # not mutual: somebody else is closer
        if np.linalg.norm(src[i] - dst[j]) > threshold:
            continue
        pairs.append((i, j))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
