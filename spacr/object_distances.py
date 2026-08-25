"""Every distance worth measuring, within and between segmented objects.

What existed before this module: within ONE object type, centroid-to-
centroid neighbour distances over a KD-tree; and between types, a single
family measuring from a CHANNEL'S intensity centre of mass to the nearest
nucleus or pathogen surface. So there was no centre-to-centre between
types, no centre-to-perimeter in either direction, no perimeter-to-
perimeter, nothing about local maxima, and nothing about where an object
sits inside its parent.

THE WHOLE MODULE RESTS ON ONE TRICK. A Euclidean distance transform of a
type's mask answers "how far is this point from the nearest object of that
type" for EVERY point in the field at once. So the cost is O(field) per
object type, not O(objects squared), and every number below is a lookup
into a transform that was computed once:

    centre -> nearest surface      dt_b[centroid_a]
    surface -> surface             min of dt_b over a's boundary pixels
    a local maximum -> anything     dt_b[peak]

That is what makes it affordable to ask for all of it.

WHAT "DISTANCE" MEANS HERE, because three different numbers get called it:

    centre_to_centre     between the two centroids. Big for two large
                         objects that are touching.
    centre_to_surface    from a's centroid to the nearest point on ANY b.
                         Asymmetric: a's centre to b's edge is not b's
                         centre to a's edge, so both are emitted.
    surface_to_surface   closest point to closest point. ZERO when they
                         touch, and it is the number a biologist means by
                         "how far apart are they".

NO OBJECT-TYPE PREFIX ON THE COLUMNS. `measure` prefixes every measurement
family with the object it belongs to, so a column called
`distance_to_own_boundary` here reaches the database as
`cell_distance_to_own_boundary`. Naming it `cell_...` here produced
`cell_cell_...`, which is the shape of every doubled-prefix bug.

NEVER NaN WHERE A NUMBER IS MEANINGFUL. An object with no partner of the
other type is ``inf`` -- genuinely infinitely far, which is a fact -- and
NaN is reserved for "not measured".
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

#: How many local maxima are kept per object, per channel.
#:
#: A cap, not a target. An object whose channel is flat has hundreds of
#: equal peaks and none of them means anything; twenty is past the point
#: where a twenty-first changes any summary derived from them.
MAX_PEAKS_PER_OBJECT = 20

#: Minimum separation between two local maxima, in pixels.
#:
#: Without it `peak_local_max` returns a plateau's every pixel, so one
#: bright blob becomes fifty "maxima" and `maxima_count` measures the
#: blob's area rather than how many bright spots there are.
PEAK_MIN_DISTANCE = 3


def _as_spacing(spacing, ndim: int) -> Optional[Tuple[float, ...]]:
    """Voxel size as a per-axis tuple, or None for pixel units."""
    if spacing is None:
        return None
    if np.isscalar(spacing):
        return tuple(float(spacing) for _ in range(ndim))
    values = tuple(float(v) for v in spacing)
    return values if len(values) == ndim else None


def surface_distance_transform(mask, spacing=None):
    """Distance from every point to the nearest object surface in ``mask``.

    ZERO INSIDE AN OBJECT. `distance_transform_edt` measures distance to the
    nearest ZERO, so it is run on the INVERTED mask: the result is 0 on any
    labelled pixel and grows outward. That is what makes a lookup at another
    object's centroid mean "distance to the nearest surface of this type",
    and what makes two touching objects come out at 0.
    """
    from scipy.ndimage import distance_transform_edt

    binary = np.asarray(mask) > 0
    if not binary.any():
        return np.full(binary.shape, np.inf, dtype=np.float32)
    kwargs = {}
    step = _as_spacing(spacing, binary.ndim)
    if step is not None:
        kwargs["sampling"] = step
    return distance_transform_edt(~binary, **kwargs).astype(np.float32)


def interior_distance_transform(mask, spacing=None):
    """Distance from every point INSIDE an object to that object's boundary.

    The complement of :func:`surface_distance_transform`: run on the mask
    itself, so it is 0 outside and peaks at each object's deepest point.
    Read at a centroid it says how far the centre is from its own rim,
    which is what makes a relative radial position possible.
    """
    from scipy.ndimage import distance_transform_edt

    binary = np.asarray(mask) > 0
    if not binary.any():
        return np.zeros(binary.shape, dtype=np.float32)
    kwargs = {}
    step = _as_spacing(spacing, binary.ndim)
    if step is not None:
        kwargs["sampling"] = step
    return distance_transform_edt(binary, **kwargs).astype(np.float32)


def _centroids(mask) -> Tuple[np.ndarray, np.ndarray]:
    """``(labels, centroids)`` for ``mask``, centroids in array order."""
    from skimage.measure import regionprops_table

    labelled = np.asarray(mask)
    if not labelled.any():
        return np.empty(0, dtype=np.int64), np.empty((0, labelled.ndim))
    props = regionprops_table(labelled, properties=("label", "centroid"))
    labels = np.asarray(props["label"], dtype=np.int64)
    axes = [props[f"centroid-{i}"] for i in range(labelled.ndim)]
    return labels, np.column_stack([np.asarray(a, dtype=float) for a in axes])


def _sample(field, points) -> np.ndarray:
    """``field`` at each point, rounded to the containing voxel.

    A point outside the field is ``inf`` rather than clipped to the edge:
    clipping would invent a distance measured from somewhere the object is
    not.
    """
    if len(points) == 0:
        return np.empty(0, dtype=float)
    index = np.round(np.asarray(points)).astype(int)
    inside = np.ones(len(index), dtype=bool)
    for axis, size in enumerate(field.shape):
        inside &= (index[:, axis] >= 0) & (index[:, axis] < size)
    out = np.full(len(index), np.inf, dtype=float)
    if inside.any():
        picked = tuple(index[inside, axis] for axis in range(index.shape[1]))
        out[inside] = np.asarray(field)[picked]
    return out


def _boundary_pixels(mask) -> Dict[int, Tuple[np.ndarray, ...]]:
    """Label -> the coordinates of that object's boundary pixels.

    The boundary, not the whole object: the closest point of a to b is on
    a's surface by definition, so scanning the interior would cost more and
    find the same minimum.
    """
    from skimage.segmentation import find_boundaries

    labelled = np.asarray(mask)
    edges = find_boundaries(labelled, mode="inner")
    coords = np.nonzero(edges)
    if not len(coords[0]):
        return {}
    values = labelled[coords]
    order = np.argsort(values, kind="stable")
    values = values[order]
    coords = tuple(axis[order] for axis in coords)
    out: Dict[int, Tuple[np.ndarray, ...]] = {}
    starts = np.searchsorted(values, np.unique(values), side="left")
    ends = np.searchsorted(values, np.unique(values), side="right")
    for label, start, end in zip(np.unique(values), starts, ends):
        out[int(label)] = tuple(axis[start:end] for axis in coords)
    return out


def _min_over_boundary(field, boundary) -> float:
    """The smallest value of ``field`` anywhere on one object's boundary."""
    if boundary is None or not len(boundary[0]):
        return float("inf")
    return float(np.min(np.asarray(field)[boundary]))


def local_maxima(image, mask, label: int) -> np.ndarray:
    """Coordinates of the intensity peaks inside one object.

    :returns: an ``(n, ndim)`` array, possibly empty.
    """
    from skimage.feature import peak_local_max

    inside = np.asarray(mask) == label
    if not inside.any():
        return np.empty((0, np.asarray(mask).ndim))
    try:
        peaks = peak_local_max(
            np.asarray(image, dtype=float), labels=inside.astype(np.int32),
            min_distance=PEAK_MIN_DISTANCE, num_peaks=MAX_PEAKS_PER_OBJECT,
            exclude_border=False)
    except Exception:                                        # noqa: BLE001
        LOG.debug("no local maxima for label %s", label, exc_info=True)
        return np.empty((0, np.asarray(mask).ndim))
    return np.asarray(peaks, dtype=float)


def _pairwise_spread(points, spacing=None) -> float:
    """Mean distance between every pair of ``points``. 0 for fewer than two.

    What it is for: two bright spots at opposite ends of a cell and two
    sitting on top of each other have the same count. This tells them
    apart.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return 0.0
    step = _as_spacing(spacing, points.shape[1])
    scaled = points * np.asarray(step) if step else points
    from scipy.spatial.distance import pdist

    return float(np.mean(pdist(scaled)))


def between_object_types(masks: Dict[str, "np.ndarray"], *,
                         primary: str, spacing=None) -> pd.DataFrame:
    """Distances from every ``primary`` object to every other object type.

    :param masks: object type -> label image, all the same shape.
    :param primary: the type whose objects are the rows.
    :param spacing: voxel size, so the numbers carry physical units.
    :returns: one row per primary object, keyed on ``label``.

    THREE NUMBERS PER PAIR OF TYPES, because they answer three different
    questions -- see the module docstring. Plus where the object sits inside
    itself and how close it is to the edge of the field, which is what says
    an object is clipped.
    """
    labelled = np.asarray(masks[primary])
    labels, centroids = _centroids(labelled)
    frame = pd.DataFrame({"label": labels})
    if not len(labels):
        return frame

    boundaries = _boundary_pixels(labelled)
    own_interior = interior_distance_transform(labelled, spacing)

    # WHERE IT SITS IN ITSELF. The interior transform at the centroid is the
    # centre's distance to its own rim; over the object's deepest point it
    # is a shape-free 0-to-1 position that compares across sizes.
    own = _sample(own_interior, centroids)
    frame["distance_to_own_boundary"] = own
    deepest = np.array([
        float(np.max(own_interior[boundaries.get(int(l), ((),))[0].size and
                                  (labelled == l)])) if (labelled == l).any()
        else np.nan for l in labels], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["relative_radial_position"] = np.where(
            deepest > 0, 1.0 - (own / deepest), np.nan)

    # HOW CLOSE TO THE EDGE OF THE FIELD. An object that touches it is
    # clipped, and every measurement of it is of a fragment.
    edge = np.full(len(labels), np.inf, dtype=float)
    for axis, size in enumerate(labelled.shape):
        here = np.minimum(centroids[:, axis], size - 1 - centroids[:, axis])
        edge = np.minimum(edge, here)
    frame["distance_to_field_edge"] = edge

    for other, other_mask in masks.items():
        if other == primary:
            continue
        other_mask = np.asarray(other_mask)
        if other_mask.shape != labelled.shape:
            continue
        to_other = surface_distance_transform(other_mask, spacing)
        frame[f"centre_to_{other}_surface"] = _sample(to_other,
                                                                centroids)
        frame[f"surface_to_{other}_surface"] = [
            _min_over_boundary(to_other, boundaries.get(int(l)))
            for l in labels]

        # CENTRE TO CENTRE, to the NEAREST object of the other type. The
        # full N x M matrix is neither cheap nor something a one-row-per-
        # object table can hold; the nearest is both.
        other_labels, other_centroids = _centroids(other_mask)
        if len(other_centroids):
            from scipy.spatial import cKDTree

            step = _as_spacing(spacing, labelled.ndim)
            scale = np.asarray(step) if step else 1.0
            tree = cKDTree(other_centroids * scale)
            nearest, _idx = tree.query(centroids * scale, k=1)
            frame[f"centre_to_nearest_{other}_centre"] = nearest
        else:
            frame[f"centre_to_nearest_{other}_centre"] = np.inf

        # OVERLAP, which is the answer when the distance is zero.
        overlap = []
        for label in labels:
            inside = labelled == label
            area = int(inside.sum())
            overlap.append(float((other_mask[inside] > 0).sum()) / area
                           if area else np.nan)
        frame[f"{other}_overlap_fraction"] = overlap
    return frame


def maxima_distances(masks: Dict[str, "np.ndarray"], images, *,
                     primary: str, channels: Sequence[int] = (),
                     spacing=None) -> pd.DataFrame:
    """Where each object's intensity peaks are, and what they are near.

    :param images: the field as ``(..., channel)``.
    :param channels: which channels to find maxima in. Empty means all.
    :returns: one row per primary object, keyed on ``label``.
    """
    labelled = np.asarray(masks[primary])
    labels, centroids = _centroids(labelled)
    frame = pd.DataFrame({"label": labels})
    if not len(labels):
        return frame

    stack = np.asarray(images)
    if stack.ndim == labelled.ndim:
        stack = stack[..., None]
    wanted = list(channels) if len(channels) else list(range(stack.shape[-1]))

    own_interior = interior_distance_transform(labelled, spacing)
    others = {name: surface_distance_transform(np.asarray(mask), spacing)
              for name, mask in masks.items()
              if name != primary
              and np.asarray(mask).shape == labelled.shape}

    for channel in wanted:
        if channel >= stack.shape[-1]:
            continue
        plane = stack[..., channel]
        counts, spreads = [], []
        to_own = {"min": [], "mean": []}
        to_centre = {"min": [], "mean": []}
        to_other = {name: {"min": [], "mean": []} for name in others}
        for label, centre in zip(labels, centroids):
            peaks = local_maxima(plane, labelled, int(label))
            counts.append(len(peaks))
            spreads.append(_pairwise_spread(peaks, spacing))
            if not len(peaks):
                # NaN, NOT ZERO. An object with no peak has no distance
                # from one; zero would read as "the peak is right here",
                # which is the opposite of what happened. The count column
                # beside it says why the row is empty.
                for holder in (to_own, to_centre, *to_other.values()):
                    holder["min"].append(np.nan)
                    holder["mean"].append(np.nan)
                continue
            own = _sample(own_interior, peaks)
            to_own["min"].append(float(np.min(own)))
            to_own["mean"].append(float(np.mean(own)))
            step = _as_spacing(spacing, peaks.shape[1])
            scale = np.asarray(step) if step else 1.0
            radial = np.linalg.norm((peaks - centre) * scale, axis=1)
            to_centre["min"].append(float(np.min(radial)))
            to_centre["mean"].append(float(np.mean(radial)))
            for name, field in others.items():
                values = _sample(field, peaks)
                to_other[name]["min"].append(float(np.min(values)))
                to_other[name]["mean"].append(float(np.mean(values)))

        stem = f"channel_{channel}_maxima"
        frame[f"{stem}_count"] = counts
        frame[f"{stem}_spread"] = spreads
        frame[f"{stem}_to_own_boundary_min"] = to_own["min"]
        frame[f"{stem}_to_own_boundary_mean"] = to_own["mean"]
        frame[f"{stem}_to_centre_min"] = to_centre["min"]
        frame[f"{stem}_to_centre_mean"] = to_centre["mean"]
        for name, holder in to_other.items():
            frame[f"{stem}_to_{name}_surface_min"] = holder["min"]
            frame[f"{stem}_to_{name}_surface_mean"] = holder["mean"]
    return frame


def intensity_centre_offset(mask, images, *, primary: str,
                            channels: Sequence[int] = (),
                            spacing=None) -> pd.DataFrame:
    """How far each channel's intensity centre sits from the geometric one.

    POLARISATION IN ONE NUMBER. A uniformly stained object has an offset of
    about zero; one whose signal is all at one end does not, and no
    intensity summary says so.
    """
    from skimage.measure import regionprops_table

    labelled = np.asarray(mask)
    labels, centroids = _centroids(labelled)
    frame = pd.DataFrame({"label": labels})
    if not len(labels):
        return frame

    stack = np.asarray(images)
    if stack.ndim == labelled.ndim:
        stack = stack[..., None]
    wanted = list(channels) if len(channels) else list(range(stack.shape[-1]))
    step = _as_spacing(spacing, labelled.ndim)
    scale = np.asarray(step) if step else 1.0

    for channel in wanted:
        if channel >= stack.shape[-1]:
            continue
        props = regionprops_table(
            labelled, intensity_image=stack[..., channel],
            properties=("label", "centroid_weighted"))
        order = {int(l): i for i, l in enumerate(props["label"])}
        weighted = np.column_stack([
            np.asarray(props[f"centroid_weighted-{i}"], dtype=float)
            for i in range(labelled.ndim)])
        picked = np.array([weighted[order[int(l)]] if int(l) in order
                           else [np.nan] * labelled.ndim for l in labels])
        offset = np.linalg.norm((picked - centroids) * scale, axis=1)
        frame[f"channel_{channel}_intensity_centre_offset"] = offset
    return frame


def object_distances(masks: Dict[str, "np.ndarray"], images=None, *,
                     primary: str, channels: Sequence[int] = (),
                     spacing=None, maxima: bool = True) -> pd.DataFrame:
    """Every distance this module measures, for one object type.

    The one call the measure pipeline makes. Joined on ``label`` so it
    widens the object's row like any other measurement family.

    :param masks: object type -> label image.
    :param images: the field, for the intensity-derived families. None
        skips them.
    :param maxima: whether to find local maxima. The most expensive part.
    """
    frame = between_object_types(masks, primary=primary, spacing=spacing)
    if images is None:
        return frame
    for extra in (intensity_centre_offset(masks[primary], images,
                                          primary=primary, channels=channels,
                                          spacing=spacing),
                  maxima_distances(masks, images, primary=primary,
                                   channels=channels, spacing=spacing)
                  if maxima else None):
        if extra is not None and len(extra.columns) > 1:
            frame = frame.merge(extra, on="label", how="left",
                                validate="one_to_one")
    return frame
