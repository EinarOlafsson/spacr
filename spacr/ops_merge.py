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

from typing import Tuple

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
