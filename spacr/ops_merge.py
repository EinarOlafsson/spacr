"""Match cells between two magnifications by their arrangement, not their pixels.

Optical pooled screening images the same well twice: once at low magnification
to read the barcodes, once at high magnification to measure the phenotype.
Every cell must be matched between the two, and the obvious method -- correlate
the images -- does not work. The pixels do not correspond: the scale differs,
the stains differ, and often the microscope differs too.

What is preserved is the arrangement of the cells. Take three neighbouring
cells and the shape of the triangle they form is unchanged by magnification,
rotation and translation. So the cells are triangulated, each triangle is
reduced to a descriptor that depends only on its shape, and the two sets are
matched on those descriptors. The transform between the acquisitions falls out
of the matched triangles, and the cells are paired under it.

Matching is mutual nearest neighbour with a distance limit: a pair is kept only
when each cell is the other's nearest, and anything doubtful is dropped rather
than assigned. A wrongly matched cell gives a real phenotype the wrong barcode,
which silently corrupts a screen; a dropped one costs only statistical power,
which more cells can buy back.

Accuracy, measured on 90 cells over a 500 px field related by a known scale of
2.0, an 11 degree rotation and a (137.5, -92.25) shift:

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

from typing import Mapping, Sequence, Tuple

import numpy as np


def triangle_descriptors(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Delaunay-triangulate ``points`` and describe each triangle by its shape.

    The descriptor is two numbers: the ratios of the two shorter sides to the
    longest. Scaling a triangle multiplies every side by the same factor, so
    the ratios do not move; rotating and translating it does not change side
    lengths at all. The descriptor therefore ignores absolute size and
    position, which are what differ between the two acquisitions.

    :param points: ``(N, 2)`` cell centroids.
    :returns: ``(descriptors, vertex_indices)`` -- an ``(M, 2)`` array and the
        ``(M, 3)`` indices of each triangle's vertices, ordered so that vertex
        0 is opposite the shortest side and vertex 2 opposite the longest. That
        ordering is what makes two matched descriptors also give matched
        points, without which the transform cannot be solved.
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
        sides = np.array([
            np.linalg.norm(p1 - p2),
            np.linalg.norm(p0 - p2),
            np.linalg.norm(p0 - p1),
        ])
        order = np.argsort(sides)
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
        correction[1, 1] = -1.0
    rotation = u @ correction @ vt
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

    scale = float(np.median(scales))
    angle = float(np.median(angles))
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])

    src_all = np.concatenate(matched_src, axis=0)
    dst_all = np.concatenate(matched_dst, axis=0)
    offsets = dst_all - (scale * (rotation @ src_all.T).T)
    translation = np.median(offsets, axis=0)
    return scale, rotation, translation


def match_cells(source: np.ndarray, target: np.ndarray, *,
                transform=None, threshold: float = 10.0,
                gpu: bool = True) -> np.ndarray:
    """Pair cells one-to-one, keeping only mutual nearest neighbours.

    A pair survives only when each cell is the other's nearest and they are
    within ``threshold``. When several cells choose the same partner, only
    that partner's nearest cell can survive.

    :param source: ``(N, 2)`` centroids.
    :param target: ``(M, 2)`` centroids.
    :param transform: ``(scale, rotation, translation)`` from
        :func:`align_by_triangles`, applied to ``source`` first. ``None``
        means the two sets are already in the same frame.
    :param threshold: the largest distance, in target units, that may still be
        called the same cell.
    :param gpu: run the two nearest-neighbour searches on a GPU when one is
        usable. A well carries tens of thousands of nuclei on each side and the
        search is done twice, once in each direction;
        `spacr.ops_accel.nearest_neighbours` processes source points in chunks
        against the full target set.
    :returns: ``(K, 2)`` array of ``(source_index, target_index)`` pairs.
    """
    from .ops_accel import nearest_neighbours

    src = np.asarray(source, float).reshape(-1, 2)
    dst = np.asarray(target, float).reshape(-1, 2)
    if src.size == 0 or dst.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    if transform is not None:
        scale, rotation, translation = transform
        src = (scale * (rotation @ src.T).T) + translation

    forward, gaps = nearest_neighbours(src, dst, gpu=gpu)
    backward, _gaps = nearest_neighbours(dst, src, gpu=gpu)

    pairs = []
    for i, j in enumerate(forward):
        if backward[j] != i:
            continue
        if np.linalg.norm(src[i] - dst[j]) > threshold:
            continue
        pairs.append((i, j))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def field_transform(scale: float, degrees: float, centre, field_shape
                    ) -> Tuple[float, np.ndarray, np.ndarray]:
    """The similarity taking a phenotype field's pixels onto the well frame.

    :param scale: well-frame pixels per field pixel; 0.25 on the reference
        plate, where a 2,960 px 20X field lands on 1,480 px 10X tiles.
    :param degrees: the field's rotation in the well frame.
    :param centre: ``(y, x)`` of the field's centre in the well frame, as
        ``ops_phenotype.centre_y/centre_x`` stores it.
    :param field_shape: ``(height, width)`` of the field in its own pixels.
    :returns: ``(scale, rotation, translation)``, the form
        :func:`match_cells` takes, mapping ``(row, column)`` field points to
        ``(y, x)`` well points.

    The centre is what the phenotype phase predicted for the field's middle
    pixel, so the translation is whatever puts ``field_shape / 2`` there
    once the field is scaled and rotated.
    """
    angle = np.radians(float(degrees))
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    middle = np.array([float(field_shape[0]) / 2.0,
                       float(field_shape[1]) / 2.0])
    translation = (np.asarray(centre, float)
                   - float(scale) * rotation @ middle)
    return float(scale), rotation, translation


def field_to_well(points, transform) -> np.ndarray:
    """Field ``(row, column)`` points moved into the well frame.

    :param points: ``(N, 2)`` points in field pixels.
    :param transform: ``(scale, rotation, translation)``.
    :returns: ``(N, 2)`` well-frame ``(y, x)``.
    """
    scale, rotation, translation = transform
    src = np.asarray(points, float).reshape(-1, 2)
    return float(scale) * (np.asarray(rotation, float) @ src.T).T + np.asarray(
        translation, float)


def refine_field_transform(points, targets, seed, *, radius: float = 3.0,
                           min_inliers: int = 20,
                           min_fraction: float = 0.2,
                           max_scale_change: float = 0.02,
                           max_degrees_change: float = 1.0):
    """Re-solve one field's transform on its own nuclei, or keep the seed.

    :param points: ``(N, 2)`` nucleus centroids of the field, field pixels.
    :param targets: ``(M, 2)`` sequencing-nucleus centroids near the field,
        well pixels.
    :param seed: the stored ``(scale, rotation, translation)``.
    :param radius: how close a moved nucleus must land to agree, well px.
    :param min_inliers: the fewest agreeing nuclei to accept a re-solve.
    :param min_fraction: the smallest share of ``points`` that must agree.
    :param max_scale_change: how far the re-solved scale may move from the
        seed's, as a fraction.
    :param max_degrees_change: how far its rotation may move, in degrees.
    :returns: ``(transform, inliers, residual_px, refined)``; ``refined`` is
        False when the seed was kept.

    THE SEED IS A RASTER PREDICTION for every field that was not an anchor,
    and a few well pixels of error in it is a few nuclear radii at 10X.
    Re-solving on the field's own segmented nuclei removes that error; the
    bounds stop a crowded field from pulling the answer into a wrong basin,
    which is the failure :func:`spacr.ops_phenotype.align_phenotype_to_sbs`
    records for unconstrained scores.
    """
    from .ops_phenotype import refine_similarity

    src = np.asarray(points, float).reshape(-1, 2)
    dst = np.asarray(targets, float).reshape(-1, 2)
    seed = (float(seed[0]), np.asarray(seed[1], float),
            np.asarray(seed[2], float))
    if src.shape[0] < 3 or dst.shape[0] < 3:
        return seed, 0, float("nan"), False
    found = refine_similarity(src, dst, seed, radius=float(radius))
    if found is None:
        return seed, 0, float("nan"), False
    scale, rotation, translation, src_in, _dst_in, distance = found
    seed_degrees = np.degrees(np.arctan2(seed[1][1, 0], seed[1][0, 0]))
    degrees = np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))
    agree = int(src_in.shape[0])
    if (agree < int(min_inliers) or agree < float(min_fraction) * src.shape[0]
            or abs(scale - seed[0]) > abs(seed[0]) * float(max_scale_change)
            or abs(degrees - seed_degrees) > float(max_degrees_change)):
        return seed, agree, float("nan"), False
    residual = float(np.median(distance)) if distance.size else float("nan")
    return (float(scale), np.asarray(rotation, float),
            np.asarray(translation, float)), agree, residual, True


def merge_field(points, targets, transform, *, threshold: float = 4.0,
                gpu: bool = False) -> np.ndarray:
    """Pair one field's nuclei with sequencing nuclei, one to one.

    :param points: ``(N, 2)`` field nucleus centroids, field pixels.
    :param targets: ``(M, 2)`` sequencing nucleus centroids, well pixels.
    :param transform: the field's ``(scale, rotation, translation)``.
    :param threshold: the largest distance still called the same nucleus,
        in well pixels.
    :param gpu: passed to :func:`match_cells`.
    :returns: ``(K, 3)`` float rows of ``(point_index, target_index,
        distance)``.
    """
    src = np.asarray(points, float).reshape(-1, 2)
    dst = np.asarray(targets, float).reshape(-1, 2)
    pairs = match_cells(src, dst, transform=transform,
                        threshold=float(threshold), gpu=gpu)
    if pairs.size == 0:
        return np.empty((0, 3), float)
    moved = field_to_well(src[pairs[:, 0]], transform)
    distance = np.linalg.norm(moved - dst[pairs[:, 1]], axis=1)
    return np.column_stack([pairs.astype(float), distance])


def deduplicate_matches(frame, *, key: str = "object_id",
                        by: str = "distance"):
    """One observation per sequencing nucleus where fields overlap.

    :param frame: merge rows with ``key`` and ``by`` columns.
    :param key: the column that must be unique afterwards.
    :param by: the column whose smallest value wins.
    :returns: ``(kept, dropped)`` -- the frame with one row per ``key``,
        and how many rows were dropped.

    A nucleus imaged in two overlapping phenotype fields is two
    observations of one cell. Counting both would weight that cell twice in
    its guide's median, so the closer match is kept; ties go to the first
    row, which keeps the result independent of nothing but the input order.
    """
    if frame is None or len(frame) == 0:
        return frame, 0
    ordered = frame.sort_values([by], kind="mergesort")
    kept = ordered.drop_duplicates(subset=[key], keep="first")
    return kept.sort_index(), int(len(frame) - len(kept))


def field_footprint(transform, field_shape) -> Tuple[np.ndarray, np.ndarray]:
    """The well-frame box a phenotype field covers.

    :param transform: the field's ``(scale, rotation, translation)``.
    :param field_shape: ``(height, width)`` of the field in its own pixels.
    :returns: ``(low, high)``, each ``(y, x)`` in well pixels: the smallest
        and largest coordinates of the field's four corners.
    """
    height, width = float(field_shape[0]), float(field_shape[1])
    corners = field_to_well(np.array([[0.0, 0.0], [0.0, width],
                                      [height, 0.0], [height, width]]),
                            transform)
    return corners.min(axis=0), corners.max(axis=0)


def footprint_depth(points, low, high) -> np.ndarray:
    """How far each well-frame point lies inside a field's footprint.

    :param points: ``(N, 2)`` well-frame ``(y, x)``.
    :param low: the footprint's smallest ``(y, x)``.
    :param high: its largest ``(y, x)``.
    :returns: ``(N,)`` distances in well pixels to the nearest footprint
        edge, negative outside.

    THE DEPTH DECIDES WHICH FIELD OWNS A CELL. Where fields overlap, the
    field in which a cell lies deepest images it furthest from any edge, so
    that copy is the one least likely to be cut by the field's border.
    """
    p = np.asarray(points, float).reshape(-1, 2)
    return np.minimum(np.minimum(p[:, 0] - low[0], high[0] - p[:, 0]),
                      np.minimum(p[:, 1] - low[1], high[1] - p[:, 1]))


def overlapping_fields(footprints) -> dict:
    """The fields whose footprints overlap each field's.

    :param footprints: ``{site: (low, high)}`` from :func:`field_footprint`.
    :returns: ``{site: [other sites]}``, in the order of ``footprints``.
    """
    sites = list(footprints)
    out = {}
    for site in sites:
        low, high = footprints[site]
        out[site] = [other for other in sites if other != site and np.all(
            np.minimum(high, footprints[other][1])
            > np.maximum(low, footprints[other][0]))]
    return out


def deepest_field_matches(frame, *, key: str = "object_id",
                          depth: str = "sbs_depth_px",
                          by: str = "distance"):
    """One observation per sequencing nucleus: the copy deepest in its field.

    :param frame: merge rows with ``key``, ``depth`` and ``by`` columns.
    :param key: the column that must be unique afterwards.
    :param depth: how deep the sequencing nucleus lies in the row's field,
        from :func:`footprint_depth`; the largest wins.
    :param by: breaks ties in ``depth``; the smallest wins.
    :returns: ``(kept, dropped)`` -- the frame with one row per ``key``, in
        the order the winning rows had, and how many rows were dropped.

    CORE OWNERSHIP BY THE SEQUENCING CENTROID (A1b). A nucleus imaged in two
    overlapping phenotype fields is kept from the field in which its
    sequencing centroid lies deepest, not from the closer match
    (:func:`deduplicate_matches`). The closer match is often the margin
    copy, and a margin copy is more often cut by the field's edge: on plate
    A of Funk et al. 2022 the closer match kept a different copy for 14.4
    percent of linked cells. Equal depths go to the closer match, then to
    the first row.
    """
    if frame is None or len(frame) == 0:
        return frame, 0
    ordered = frame.sort_values([depth, by], ascending=[False, True],
                                kind="mergesort")
    kept = ordered.drop_duplicates(subset=[key], keep="first")
    return kept, int(len(frame) - len(kept))


def _one_blas_thread():
    """A context that runs its block on one BLAS thread.

    :returns: a ``threadpoolctl`` limit, or a null context when
        ``threadpoolctl`` is not installed.

    Threaded BLAS may change the last bits of a least-squares solve with
    the thread count, so the affine fits run on one thread and a rerun on
    any machine stores the same transforms (as
    :func:`spacr.ops_solve.solve_placements`).
    """
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=1, user_api="blas")
    except ImportError:
        import contextlib
        return contextlib.nullcontext()


def affine_polish(points, targets, matrix, offset, *, radius: float = 3.0,
                  passes: int = 3, min_pairs: int = 20, tree=None):
    """Re-fit a 6-parameter affine on mutual nearest pairs, trimmed.

    :param points: ``(N, 2)`` field points, field pixels.
    :param targets: ``(M, 2)`` sequencing nuclei, well pixels.
    :param matrix: the starting ``2 x 2`` linear part.
    :param offset: the starting translation, ``(y, x)``.
    :param radius: a moved point and its nearest target pair only within
        this many well pixels.
    :param passes: how many times the pairs are found and the affine
        re-fitted.
    :param min_pairs: the fewest pairs a pass may fit on.
    :param tree: a ``cKDTree`` of ``targets``, built when None.
    :returns: ``(matrix, offset, pairs)``, or None when a pass finds fewer
        than ``min_pairs`` pairs.

    Each pass moves the points with the current affine, pairs a point with
    its nearest target within ``radius`` when that target's nearest moved
    point is the same point (mutual nearest), drops pairs further apart
    than ``max(3 x median, 0.5)`` px and solves the affine by least squares
    on the rest.
    """
    from scipy.spatial import cKDTree

    src = np.asarray(points, float).reshape(-1, 2)
    dst = np.asarray(targets, float).reshape(-1, 2)
    if tree is None:
        tree = cKDTree(dst)
    matrix = np.asarray(matrix, float)
    offset = np.asarray(offset, float)
    count = 0
    for _ in range(int(passes)):
        moved = src @ matrix.T + offset
        distance, j = tree.query(moved, distance_upper_bound=float(radius))
        found = np.isfinite(distance)
        i = np.flatnonzero(found)
        j = j[found]
        back = cKDTree(moved).query(dst[j])[1]
        mutual = back == i
        i, j = i[mutual], j[mutual]
        if len(i) < int(min_pairs):
            return None
        r = np.hypot(*(moved[i] - dst[j]).T)
        keep = r <= max(3 * np.median(r), 0.5)
        i, j = i[keep], j[keep]
        design = np.c_[src[i], np.ones(len(i))]
        solution = np.linalg.lstsq(design, dst[j], rcond=None)[0]
        matrix, offset = solution[:2].T, solution[2]
        count = len(i)
    return matrix, offset, count


def affine_within_bounds(matrix, seed, *, max_scale_change: float = 0.02,
                         max_degrees_change: float = 1.0,
                         max_anisotropy: float = 0.02) -> bool:
    """Whether a fitted affine stays near the field's seed.

    :param matrix: the ``2 x 2`` linear part of the affine.
    :param seed: the field's stored ``(scale, rotation, translation)``.
    :param max_scale_change: how far the mean singular value may move from
        the seed's scale, as a fraction.
    :param max_degrees_change: how far the rotation of its polar factor
        may move from the seed's, in degrees.
    :param max_anisotropy: the largest ratio of the two singular values,
        minus one.
    :returns: True when every bound holds and the affine does not mirror.

    The scale and rotation bounds are those of
    :func:`refine_field_transform`; the anisotropy bound stops a crowded
    corner from shearing the field.
    """
    a = np.asarray(matrix, float)
    scale0 = float(seed[0])
    rotation0 = np.asarray(seed[1], float)
    u, singular, vt = np.linalg.svd(a)
    rotation = u @ vt
    degrees = abs(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])
                             - np.arctan2(rotation0[1, 0], rotation0[0, 0])))
    return bool(abs(singular.mean() - scale0) <= max_scale_change * scale0
                and degrees <= max_degrees_change
                and singular[0] / singular[1] - 1 <= max_anisotropy
                and np.linalg.det(a) > 0)


def tile_affine_transform(points, targets, start, seed, tile_centres, *,
                          refined: bool = True, min_nuclei: int = 200,
                          radius: float = 3.0, passes: int = 3):
    """One affine per field and one per sequencing tile group of the field.

    :param points: ``(N, 2)`` field nucleus centroids, field pixels.
    :param targets: ``(M, 2)`` sequencing nuclei near the field, well
        pixels.
    :param start: the field's re-solved ``(scale, rotation, translation)``
        from :func:`refine_field_transform`.
    :param seed: the field's stored ``(scale, rotation, translation)``;
        every fit must stay within :func:`affine_within_bounds` of it.
    :param tile_centres: ``(T, 2)`` centres of the sequencing tiles, well
        pixels. A field point belongs to the group of the tile whose centre
        is nearest to it under the field affine.
    :param refined: whether ``start`` is a re-solve; when it is the seed
        nothing is fitted.
    :param min_nuclei: the fewest field nuclei a tile group needs for its
        own affine; smaller groups keep the field affine.
    :param radius: the pairing radius of :func:`affine_polish`, well px.
    :param passes: the trimmed passes of :func:`affine_polish`.
    :returns: ``(field, groups)``: ``field`` is the field affine as
        ``(1.0, matrix, offset)``, a transform :func:`field_to_well` takes,
        and ``groups`` maps a tile index to ``(1.0, matrix, offset, nuclei)``.

    WHY PER TILE. The sequencing mosaic is stitched from tiles placed one
    by one, so the residual of a phenotype field changes where it crosses a
    tile seam; a single similarity per field leaves that step in. On Funk
    et al. 2022 plates A-D (well A1) the share of nuclei within 2 px of
    their sequencing nucleus rose from 94.7-95.8 % to 97.0-97.5 % with no
    failed field. The field affine starts from ``start`` and falls back to
    it when its fit fails or leaves the bounds; a group fit that fails or
    leaves the bounds keeps the field affine. The solves run on one BLAS
    thread, so the result does not depend on the thread count.
    """
    from scipy.spatial import cKDTree

    src = np.asarray(points, float).reshape(-1, 2)
    dst = np.asarray(targets, float).reshape(-1, 2)
    matrix = float(start[0]) * np.asarray(start[1], float)
    offset = np.asarray(start[2], float)
    groups = {}
    if not refined or src.shape[0] < 3 or dst.shape[0] < 3:
        return (1.0, matrix, offset), groups
    tree = cKDTree(dst)
    with _one_blas_thread():
        fit = affine_polish(src, dst, matrix, offset, radius=radius,
                            passes=passes, tree=tree)
        if fit is not None and affine_within_bounds(fit[0], seed):
            matrix, offset = fit[0], fit[1]
        centres = np.asarray(tile_centres, float).reshape(-1, 2)
        if centres.shape[0] == 0:
            return (1.0, matrix, offset), groups
        _, own = cKDTree(centres).query(src @ matrix.T + offset)
        for tile in np.unique(own):
            members = np.flatnonzero(own == tile)
            if len(members) < int(min_nuclei):
                continue
            fit = affine_polish(src[members], dst, matrix, offset,
                                radius=radius, passes=passes, tree=tree)
            if fit is not None and affine_within_bounds(fit[0], seed):
                groups[int(tile)] = (1.0, fit[0], fit[1], int(len(members)))
    return (1.0, matrix, offset), groups


def tile_affine_to_well(points, field, groups, tile_centres) -> np.ndarray:
    """Field points moved into the well frame by their tile group's affine.

    :param points: ``(N, 2)`` field points.
    :param field: the field affine from :func:`tile_affine_transform`.
    :param groups: its tile groups, ``{tile: (1.0, matrix, offset, ...)}``.
    :param tile_centres: the same ``(T, 2)`` tile centres the fit used.
    :returns: ``(N, 2)`` well-frame ``(y, x)``. A point takes the affine of
        the tile whose centre is nearest to it under the field affine, and
        the field affine when that tile has none.
    """
    from scipy.spatial import cKDTree

    src = np.asarray(points, float).reshape(-1, 2)
    moved = src @ np.asarray(field[1], float).T + np.asarray(field[2], float)
    if groups and src.shape[0]:
        centres = np.asarray(tile_centres, float).reshape(-1, 2)
        _, own = cKDTree(centres).query(moved)
        for tile, transform in groups.items():
            members = own == int(tile)
            moved[members] = (src[members] @ np.asarray(transform[1], float).T
                              + np.asarray(transform[2], float))
    return moved


def _triangle_votes(source, target, *, tolerance: float = 0.02,
                    scale=None, scale_tolerance: float = 0.05,
                    radius: float = 3.0, max_pairs: int = 2_000_000):
    """The triangle-hash consensus between two cell sets, with its votes.

    The same descriptors as :func:`align_by_triangles`
    (:func:`triangle_descriptors`), solved for every descriptor match at
    once: each matched triangle pair proposes a similarity, the proposals
    are pooled where they send the centre of ``source``, and the
    similarity is re-solved on the vertices of the triangle pairs that
    agree with the pool. A triangle pair agrees -- one vote -- when all
    three of its source vertices land within ``radius`` of their target
    vertices under the consensus, so a pair of unrelated point sets, which
    still produces descriptor matches, scores few votes rather than many.

    :param source: ``(N, 2)`` points to be moved.
    :param target: ``(M, 2)`` points to move them to.
    :param tolerance: descriptor distance still called the same triangle.
    :param scale: the expected scale, or None; proposals further than
        ``scale_tolerance`` (relative, in log scale) are not counted.
    :param scale_tolerance: see ``scale``.
    :param radius: vertex agreement, in target units.
    :param max_pairs: the most descriptor matches solved; above it every
        k-th match is kept, deterministically.
    :returns: ``(transform, votes)``; ``transform`` is ``(scale, rotation,
        translation)`` or None when no proposal survived.
    """
    from scipy.spatial import cKDTree

    src = np.asarray(source, float).reshape(-1, 2)
    dst = np.asarray(target, float).reshape(-1, 2)
    if src.shape[0] < 3 or dst.shape[0] < 3:
        return None, 0
    src_desc, src_tri = triangle_descriptors(src)
    dst_desc, dst_tri = triangle_descriptors(dst)
    keep_src = src_desc[:, 1] > 0
    keep_dst = dst_desc[:, 1] > 0
    src_desc, src_tri = src_desc[keep_src], src_tri[keep_src]
    dst_desc, dst_tri = dst_desc[keep_dst], dst_tri[keep_dst]
    if src_desc.shape[0] == 0 or dst_desc.shape[0] == 0:
        return None, 0
    hits = cKDTree(dst_desc).query_ball_point(src_desc, float(tolerance))
    counts = np.fromiter((len(h) for h in hits), np.int64, len(hits))
    if counts.sum() == 0:
        return None, 0
    i = np.repeat(np.arange(len(hits)), counts)
    j = np.fromiter((k for h in hits for k in h), np.int64, int(counts.sum()))
    if i.size > int(max_pairs):
        step = int(np.ceil(i.size / float(max_pairs)))
        i, j = i[::step], j[::step]
    s = src[src_tri[i]]
    d = dst[dst_tri[j]]
    s_c = s[..., 0] + 1j * s[..., 1]
    d_c = d[..., 0] + 1j * d[..., 1]
    s_mean = s_c.mean(axis=1)
    d_mean = d_c.mean(axis=1)
    s0 = s_c - s_mean[:, None]
    d0 = d_c - d_mean[:, None]
    norm = (np.abs(s0) ** 2).sum(axis=1)
    usable = norm > 1e-12
    z = np.zeros(len(norm), complex)
    z[usable] = (d0[usable] * np.conj(s0[usable])).sum(axis=1) / norm[usable]
    scales = np.abs(z)
    usable &= np.isfinite(scales) & (scales > 0)
    if scale is not None:
        usable &= np.abs(np.log(np.where(usable, scales, 1.0)
                                / float(scale))) <= float(scale_tolerance)
    if not usable.any():
        return None, 0
    z, s, d = z[usable], s_c[usable], d_c[usable]
    reference = complex(src[:, 0].mean(), src[:, 1].mean())
    lands = (d.mean(axis=1) + z * (reference - s.mean(axis=1)))
    cell = 8.0 * float(radius)
    keys = np.column_stack([np.floor(lands.real / cell),
                            np.floor(lands.imag / cell)]).astype(np.int64)
    found, sizes = np.unique(keys, axis=0, return_counts=True)
    best = found[int(np.argmax(sizes))]
    near = np.all(np.abs(keys - best) <= 1, axis=1)
    zz = z[near]
    scale_mid = float(np.exp(np.median(np.log(np.abs(zz)))))
    angle_mid = float(np.angle(np.exp(1j * np.angle(zz)).sum()))
    mid = scale_mid * np.exp(1j * angle_mid)
    land = complex(np.median(lands[near].real), np.median(lands[near].imag))
    shift = land - mid * reference
    votes = 0
    fit = None
    for _round in range(2):
        moved = mid * s + shift
        agree = np.abs(moved - d).max(axis=1) <= float(radius)
        votes = int(agree.sum())
        if votes == 0:
            return None, 0
        pairs_src = np.column_stack([s[agree].real.ravel(),
                                     s[agree].imag.ravel()])
        pairs_dst = np.column_stack([d[agree].real.ravel(),
                                     d[agree].imag.ravel()])
        fit = _similarity_from_pairs(pairs_src, pairs_dst)
        if fit is None:
            return None, 0
        rotation = np.asarray(fit[1], float)
        mid = float(fit[0]) * complex(rotation[0, 0], rotation[1, 0])
        shift = complex(float(fit[2][0]), float(fit[2][1]))
    moved = mid * s + shift
    votes = int((np.abs(moved - d).max(axis=1) <= float(radius)).sum())
    if votes == 0:
        return None, 0
    return (float(fit[0]), np.asarray(fit[1], float),
            np.asarray(fit[2], float)), votes


def _compose(outer, inner):
    """The similarity ``outer`` after ``inner``, as one ``(s, R, t)``."""
    scale = float(outer[0]) * float(inner[0])
    rotation = np.asarray(outer[1], float) @ np.asarray(inner[1], float)
    translation = (float(outer[0]) * np.asarray(outer[1], float)
                   @ np.asarray(inner[2], float) + np.asarray(outer[2], float))
    return scale, rotation, translation


def _weak_pair(votes: int, fraction: float, min_votes: int,
               min_fraction: float) -> bool:
    """Whether one tile pair fails the fallback rule (item 962)."""
    return int(votes) < int(min_votes) or not (
        float(fraction) >= float(min_fraction))


def _choose_route(weak, fields: int, share: float) -> Tuple[str, str]:
    """``('tile' | 'stitch', reason)`` for one well under the fallback rule.

    :param weak: how many tile pairs are weak.
    :param fields: how many tile pairs were tried.
    :param share: the share of weak pairs above which the well falls back.
    """
    if fields <= 0:
        return "stitch", "no tile pair could be tried"
    ratio = float(weak) / float(fields)
    if ratio > float(share):
        return "stitch", (f"{int(weak)} of {int(fields)} tile pairs weak "
                          f"({ratio:.3f} > {float(share):.3f})")
    return "tile", (f"{int(weak)} of {int(fields)} tile pairs weak "
                    f"({ratio:.3f} <= {float(share):.3f})")


def _well_similarity(source, target, *, scale: float = 1.0,
                     regions: int = 9, region_radius: float = 600.0,
                     margin: float = 200.0, radius: float = 3.0,
                     min_votes: int = 3):
    """One similarity between two well mosaics, from sampled regions.

    A whole well holds too many cells to triangulate and hash at once, so
    ``regions`` discs of ``region_radius`` are taken on a grid over the
    source mosaic, each is hashed against the target cells within
    ``region_radius + margin`` of the same place (:func:`_triangle_votes`),
    and the regions whose transforms agree are kept. A region's own fit
    pins its translation well but its scale and rotation only to a fraction
    of a percent, which is several pixels across a well; so the cells each
    agreeing region pairs (:func:`match_cells` within ``radius``) are
    pooled and one similarity is solved on all of them by least squares,
    then re-solved once on the pairs that similarity gives.

    :param source: ``(N, 2)`` source mosaic points, target pixels.
    :param target: ``(M, 2)`` target mosaic points.
    :param scale: the expected scale between the mosaics (1 once both are
        in target pixels).
    :param regions: how many regions to try (a square grid).
    :param region_radius: the source disc radius, target px.
    :param margin: how far the stage may be off, target px.
    :param radius: pairing distance for the votes and the pooled pairs.
    :param min_votes: the fewest votes for a region to count.
    :returns: ``(transform, votes, regions_used)``; the identity with 0
        votes and 0 regions when no region agreed.
    """
    from scipy.spatial import cKDTree

    src = np.asarray(source, float).reshape(-1, 2)
    dst = np.asarray(target, float).reshape(-1, 2)
    identity = (1.0, np.eye(2), np.zeros(2))
    if src.shape[0] < 3 or dst.shape[0] < 3:
        return identity, 0, 0
    side = max(1, int(np.ceil(np.sqrt(int(regions)))))
    low, high = src.min(axis=0), src.max(axis=0)
    edges_y = np.linspace(low[0], high[0], side + 1)
    edges_x = np.linspace(low[1], high[1], side + 1)
    src_tree, dst_tree = cKDTree(src), cKDTree(dst)
    found = []
    for a in range(side):
        for b in range(side):
            centre = np.array([(edges_y[a] + edges_y[a + 1]) / 2.0,
                               (edges_x[b] + edges_x[b + 1]) / 2.0])
            mine = np.sort(src_tree.query_ball_point(
                centre, float(region_radius))).astype(np.int64)
            if len(mine) < 3:
                continue
            theirs = np.sort(dst_tree.query_ball_point(
                centre, float(region_radius) + float(margin))).astype(np.int64)
            if len(theirs) < 3:
                continue
            fit, votes = _triangle_votes(src[mine], dst[theirs], scale=scale,
                                         radius=radius)
            if fit is not None and votes >= int(min_votes):
                found.append((fit, votes, mine, theirs))
    if not found:
        return identity, 0, 0
    shifts = np.array([f[0][2] for f in found])
    middle = np.median(shifts, axis=0)
    agree = [f for f in found
             if np.hypot(*(np.asarray(f[0][2]) - middle)) <= 4.0 * radius
             * max(1.0, float(margin) / 50.0)]
    if not agree:
        agree = found
    fit = None
    for round_ in range(2):
        pairs_src, pairs_dst = [], []
        for region_fit, _votes, mine, theirs in agree:
            use = region_fit if round_ == 0 else fit
            pairs = match_cells(src[mine], dst[theirs], transform=use,
                                threshold=float(radius), gpu=False)
            if len(pairs):
                pairs_src.append(src[mine][pairs[:, 0]])
                pairs_dst.append(dst[theirs][pairs[:, 1]])
        if not pairs_src:
            break
        solved = _similarity_from_pairs(np.vstack(pairs_src),
                                        np.vstack(pairs_dst))
        if solved is None:
            break
        fit = (float(solved[0]), np.asarray(solved[1], float),
               np.asarray(solved[2], float))
    if fit is None:
        return identity, 0, 0
    return fit, int(sum(f[1] for f in agree)), int(len(agree))


#: SEAM OWNERSHIP. Neighbouring fields of a tiled acquisition overlap, so a
#: cell in the overlap is segmented once per field that sees it. The
#: functions below are the one place that decides which copy is the cell:
#: :func:`_match_copies_by_centroid` finds the copies from their nucleus
#: centroids in stitched coordinates and :func:`_assign_field_owners` names
#: one owner per physical cell: the field whose centre is nearest the cell
#: centroid among the fields that hold it whole, ties to the lower field,
#: then the lower label. A copy cut by its field edge never owns a cell.
#: ``'largest'`` (the largest copy, item 874) is deprecated and only
#: reproduces runs made before settings schema 963. The owner rules, the
#: first the default; the others are deprecated.
_FIELD_OWNER_RULES = ("nearest_centre", "largest")

#: The rules kept only to reproduce outputs of earlier runs.
_DEPRECATED_FIELD_OWNER_RULES = ("largest",)


def _field_owner_rule(value) -> str:
    """``value`` as one of :data:`_FIELD_OWNER_RULES`; None is the default.

    :param value: a rule name, or None.
    :returns: the rule, stripped and in lower case.
    :raises ValueError: naming the rules, for any other value.
    """
    from .qt.i18n import tr

    rule = str(value or _FIELD_OWNER_RULES[0]).strip().lower()
    if rule not in _FIELD_OWNER_RULES:
        raise ValueError(tr(
            "the field owner rule must be one of {rules}; got {value}",
            rules=list(_FIELD_OWNER_RULES), value=rule))
    return rule


def _field_centre(transform, shape) -> np.ndarray:
    """The well ``(y, x)`` of the centre of a field.

    :param transform: the field ``(scale, rotation, offset)``; a point moves
        to the well as ``scale * rotation @ p + offset``.
    :param shape: the field ``(height, width)`` in its own pixels.
    """
    scale, rotation, offset = transform
    middle = np.array([float(shape[0]) / 2.0, float(shape[1]) / 2.0])
    return scale * (np.asarray(rotation, float) @ middle) + np.asarray(
        offset, float)


def _mutual_pairs(points_a, points_b, radius: float):
    """``(i, j, distance)`` where ``a[i]`` and ``b[j]`` are each the nearest
    of the other within ``radius``."""
    from scipy.spatial import cKDTree

    empty = (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, float))
    if not len(points_a) or not len(points_b):
        return empty
    near_b, index_b = cKDTree(points_b).query(points_a, k=1,
                                              distance_upper_bound=radius)
    near_a, index_a = cKDTree(points_a).query(points_b, k=1,
                                              distance_upper_bound=radius)
    hit = np.flatnonzero(np.isfinite(near_b))
    hit = hit[index_a[index_b[hit]] == hit]
    return hit, index_b[hit].astype(np.int64), near_b[hit]


def _copy_components(count: int, first, second, sites, rank=None):
    """Physical cells from the copy edges, never two copies of one field in one.

    The connected components of the undirected copy graph, except that an
    edge which would put two copies of the same field (two distinct cells
    of one field image) into one component is refused. Edges are taken in
    ``rank`` order, so each copy keeps its nearest (or best scoring)
    partners and a chain of near matches cannot merge two cells of one
    field.

    :param count: the number of copies.
    :param first: the first copy of each edge (row index).
    :param second: the second copy of each edge.
    :param sites: the field of every copy.
    :param rank: the order the edges are taken in (best first), or None
        for the given order.
    :returns: ``(component, refused)``: a component number per copy,
        numbered as :func:`scipy.sparse.csgraph.connected_components`
        numbers them (by the smallest copy in each), and the number of
        edges refused.
    """
    first = np.asarray(first, np.int64)
    second = np.asarray(second, np.int64)
    sites = np.asarray(sites, np.int64)
    parent = np.arange(count, dtype=np.int64)
    fields = {}

    def root(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return int(node)

    refused = 0
    order = range(len(first)) if rank is None else np.asarray(rank, np.int64)
    for edge in order:
        a, b = root(int(first[edge])), root(int(second[edge]))
        if a == b:
            continue
        here = fields.get(a, {int(sites[a])})
        there = fields.get(b, {int(sites[b])})
        if here & there:
            refused += 1
            continue
        if len(here) < len(there):
            a, b, here, there = b, a, there, here
        parent[b] = a
        fields[a] = here | there
        fields.pop(b, None)
    roots = np.array([root(i) for i in range(count)], np.int64)
    _unique, first_seen, component = np.unique(roots, return_index=True,
                                               return_inverse=True)
    renumber = np.empty(len(first_seen), np.int64)
    renumber[np.argsort(first_seen, kind="stable")] = np.arange(
        len(first_seen))
    return renumber[component.reshape(-1)], refused


def _match_copies_by_centroid(cells, neighbours: Mapping[int, Sequence[int]],
                             radius: float, cut_radius: float = None):
    """Edges between the copies of one cell in overlapping fields.

    :param cells: one row per copy with ``site``, ``label``, ``well_y``,
        ``well_x`` (the nucleus centroid in stitched coordinates) and
        ``whole``.
    :param neighbours: ``{site: [sites whose footprint overlaps]}``.
    :param radius: the largest centroid distance, in well pixels, of two
        whole copies of one cell.
    :param cut_radius: the same for a copy cut by its field edge, whose
        centroid moves with the part that is missing; 1.5 times
        ``radius`` when None.
    :returns: ``site_a, label_a, site_b, label_b, distance``. Whole copies
        are paired per pair of overlapping fields when each is the nearest
        whole copy of the other within ``radius``. A cut copy is joined to
        the one nearest whole copy of an overlapping field within
        ``cut_radius``, unless that cell already has a whole copy in the
        field of the cut copy, so it can never join two cells; cut copies
        with no whole copy near are paired with each other as whole copies
        are.
        Only whole copies can own a cell; the cut edges let a table count
        each physical cell once, owned or not.
    """
    import pandas as pd
    from scipy.spatial import cKDTree

    columns = ["site_a", "label_a", "site_b", "label_b", "distance"]
    cut_radius = 1.5 * float(radius) if cut_radius is None else float(
        cut_radius)
    whole = np.asarray(cells["whole"], bool)
    points = cells[["well_y", "well_x"]].to_numpy(float).reshape(-1, 2)
    sites = cells["site"].to_numpy(np.int64)
    labels = cells["label"].to_numpy(np.int64)
    rows = {int(s): np.flatnonzero(sites == s) for s in np.unique(sites)}
    found = []

    pairs = [(a, b) for a in sorted(rows) for b in sorted(
        int(s) for s in neighbours.get(a, ())) if b > a and b in rows]
    for a, b in pairs:
        first, second = rows[a][whole[rows[a]]], rows[b][whole[rows[b]]]
        i, j, distance = _mutual_pairs(points[first], points[second], radius)
        if i.size:
            found.append((first[i], second[j], distance))
    if found:
        head = np.concatenate([f for f, _s, _d in found])
        tail = np.concatenate([t for _f, t, _d in found])
        gap = np.concatenate([d for _f, _s, d in found])
    else:
        head = tail = np.empty(0, np.int64)
        gap = np.empty(0, float)
    group, _refused = _copy_components(len(cells), head, tail, sites,
                                       np.argsort(gap, kind="stable"))
    group_sites: dict = {}
    for index in np.flatnonzero(whole):
        group_sites.setdefault(int(group[index]), set()).add(int(sites[index]))
    lone = []
    for site, members in sorted(rows.items()):
        cut = members[~whole[members]]
        if not cut.size:
            continue
        best = np.full(cut.size, np.inf)
        partner = np.full(cut.size, -1, np.int64)
        for other in sorted(int(s) for s in neighbours.get(site, ())):
            if other not in rows:
                continue
            target = rows[other][whole[rows[other]]]
            if not target.size:
                continue
            near, index = cKDTree(points[target]).query(
                points[cut], k=1, distance_upper_bound=cut_radius)
            better = near < best
            best[better] = near[better]
            partner[better] = target[index[better]]
        joined = partner >= 0
        joined[joined] = [site not in group_sites[int(group[t])]
                          for t in partner[joined]]
        if joined.any():
            found.append((cut[joined], partner[joined], best[joined]))
        lone.append(cut[~joined])
    lone = np.concatenate(lone) if lone else np.empty(0, np.int64)
    alone = np.zeros(len(cells), bool)
    alone[lone] = True
    for a, b in pairs:
        first, second = rows[a][alone[rows[a]]], rows[b][alone[rows[b]]]
        i, j, distance = _mutual_pairs(points[first], points[second],
                                       cut_radius)
        if i.size:
            found.append((first[i], second[j], distance))
    if not found:
        return pd.DataFrame({c: pd.Series(dtype=float if c == "distance"
                                          else np.int64) for c in columns})
    first = np.concatenate([f for f, _s, _d in found])
    second = np.concatenate([s for _f, s, _d in found])
    return pd.DataFrame({
        "site_a": sites[first], "label_a": labels[first],
        "site_b": sites[second], "label_b": labels[second],
        "distance": np.concatenate([d for _f, _s, d in found])})[columns]


def _assign_field_owners(cells, edges, *, rule: str = "nearest_centre",
                        centres: Mapping[int, Sequence[float]] = None):
    """Physical cells and the one field that owns each.

    :param cells: one row per copy: ``site``, ``label``; ``area`` for
        ``'largest'``; ``whole``, ``well_y`` and ``well_x`` for
        ``'nearest_centre'``.
    :param edges: ``site_a, label_a, site_b, label_b``: two copies of one
        cell (:func:`_match_copies_by_centroid`, or mask overlaps).
    :param rule: one of :data:`_FIELD_OWNER_RULES`.
    :param centres: ``{site: (y, x)}``, the well centre of each field
        (:func:`_field_centre`), for ``'nearest_centre'``.
    :returns: ``cells`` with ``component`` (the connected component of the
        undirected edge graph, never holding two copies of one field:
        :func:`_copy_components`, edges taken by ``distance`` ascending or
        ``score`` descending when the edges have one; ``attrs["refused_edges"]``
        counts the edges refused), ``n_copies``, ``owner_field``,
        ``owner_label`` (both missing when no copy can own the cell),
        ``is_owner``, ``owned_area_share`` (when ``area`` is given) and,
        for ``'nearest_centre'``, ``centre_distance_px``.
    """
    import pandas as pd

    rule = _field_owner_rule(rule)
    cells = cells.reset_index(drop=True).copy()
    index = pd.Series(np.arange(len(cells)), index=pd.MultiIndex.from_arrays(
        [cells["site"].astype(np.int64), cells["label"].astype(np.int64)]))
    if len(edges):
        first = index.reindex(pd.MultiIndex.from_arrays(
            [edges["site_a"].astype(np.int64),
             edges["label_a"].astype(np.int64)])).to_numpy(float)
        second = index.reindex(pd.MultiIndex.from_arrays(
            [edges["site_b"].astype(np.int64),
             edges["label_b"].astype(np.int64)])).to_numpy(float)
        known = ~(np.isnan(first) | np.isnan(second))
        first, second = first[known].astype(np.int64), second[known].astype(
            np.int64)
        if "distance" in edges:
            key = edges["distance"].to_numpy(float)[known]
        elif "score" in edges:
            key = -edges["score"].to_numpy(float)[known]
        else:
            key = np.zeros(len(first))
    else:
        first = second = np.empty(0, np.int64)
        key = np.empty(0, float)
    count = len(cells)
    component, refused = _copy_components(
        count, first, second, cells["site"].to_numpy(np.int64),
        np.argsort(key, kind="stable"))
    cells["component"] = component
    cells["n_copies"] = cells.groupby("component")["label"].transform("size")
    rank = cells[["component", "site", "label"]].copy()
    eligible = np.ones(count, bool)
    if rule == "nearest_centre":
        centre = np.array([centres[int(s)] for s in cells["site"]],
                          float).reshape(-1, 2)
        distance = np.hypot(cells["well_y"].to_numpy(float) - centre[:, 0],
                            cells["well_x"].to_numpy(float) - centre[:, 1])
        cells["centre_distance_px"] = distance
        rank["key"] = distance
        eligible = np.asarray(cells["whole"], bool)
    else:
        rank["key"] = -cells["area"].to_numpy(float)
    order = rank[eligible].sort_values(["component", "key", "site", "label"],
                                       kind="mergesort")
    owners = order.drop_duplicates("component")[["component", "site",
                                                 "label"]]
    owners.columns = ["component", "owner_field", "owner_label"]
    cells = cells.merge(owners, on="component", how="left")
    cells["is_owner"] = ((cells["site"] == cells["owner_field"])
                         & (cells["label"] == cells["owner_label"]))
    if "area" in cells:
        owner_area = cells.loc[cells["is_owner"]].set_index("component")[
            "area"]
        cells["owned_area_share"] = cells["area"] / cells["component"].map(
            owner_area)
    cells.attrs["refused_edges"] = int(refused)
    return cells


def _owner_counts(cells) -> Tuple[int, int, int]:
    """``(physical cells, owned cells, cells without an owner)`` in a table.

    :param cells: a table of :func:`_assign_field_owners`.
    :returns: the three counts.
    """
    components = cells.drop_duplicates("component")
    owned = int(cells["is_owner"].sum())
    return int(len(components)), owned, int(len(components) - owned)
