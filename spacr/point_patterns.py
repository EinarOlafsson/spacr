"""Is this well clustered, dispersed, or neither -- against the right null.

WHAT THE PACKAGE COULD NOT ASK. :mod:`spacr.object_distances` says how far
one object is from another, and :mod:`spacr.bystanders` says whether a cell
has an infected neighbour. Neither answers the question a focus of
infection actually poses: are the infected cells in this well arranged more
tightly than chance would put them? That is Ripley's K, and it is the
statistic that turns "these two cells are close" into "this well is
clustered".

THE NULL IS THE WHOLE STATISTIC. K is meaningless without something to
compare it against, and the comparison has to respect the shape the cells
were allowed to occupy. A well is not a rectangle, a field with a torn
monolayer is not a rectangle, and a null that scatters points over the
bounding box will call every mask with a concave edge "clustered" -- the
points are crowded into the tissue because there is nowhere else to be.
Everything here takes a mask and both the estimator and the simulated null
stay inside it.

EDGE CORRECTION IS NOT OPTIONAL AND IT IS THE PART THAT GETS SKIPPED. A
point near the boundary has part of its neighbourhood outside the window,
so it finds fewer neighbours than it has; uncorrected, that deficit reads
as dispersion, and it is worst at exactly the radii anyone cares about.
:func:`ripley_k` applies the reduced-sample (border) correction by
default: at radius r, only points at least r inside the boundary
contribute. It costs precision at large r -- honestly, by reporting how
many points were left -- and it is exact for any window shape, which the
analytic isotropic corrections are not.

THE FRAME EDGE IS A BOUNDARY TOO. The distance-to-boundary field is
computed on a mask padded with one row of background all round, so a mask
that runs to the edge of the image is treated as cut off there, because it
is. Without the pad, cells at the top of the image would be scored as
though the monolayer continued past it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: How far out K is worth computing, as a fraction of the window's shorter
#: side.
#:
#: The reduced-sample correction throws away every point within r of the
#: boundary, so past about a quarter of the window there are too few
#: contributing points for the estimate to mean anything -- and the
#: envelope widens to swallow any real signal anyway. This is the
#: conventional stopping rule and :func:`suggest_radii` applies it.
MAX_RADIUS_FRACTION = 0.25

#: Simulations behind a default envelope.
#:
#: 99, NOT 100, AND THE ODD NUMBER IS THE POINT. A min/max envelope over
#: ``m`` simulations is an exact Monte Carlo test at level ``2 / (m + 1)``,
#: which is 2% at 99. Round numbers give awkward levels.
DEFAULT_SIMULATIONS = 99


def suggest_radii(mask, *, spacing=None, count: int = 20) -> np.ndarray:
    """Radii worth computing K at, for this window.

    :param mask: boolean array of where a point was allowed to be.
    :param spacing: pixel size, so the radii come back as real lengths.
    :param count: how many radii.
    :returns: ``count`` radii, evenly spaced, ending at
        :data:`MAX_RADIUS_FRACTION` of the window's shorter side.

    NOT STARTING AT ZERO. K(0) is 0 by construction and carries no
    information; including it only puts a point where every curve agrees.
    """
    window = np.asarray(mask, dtype=bool)
    scale = _as_spacing(spacing, window.ndim)
    sides = [size * step for size, step in zip(window.shape, scale)]
    far = MAX_RADIUS_FRACTION * min(sides)
    return np.linspace(far / count, far, int(count))


def ripley_k(points, mask, radii, *, spacing=None,
             correction: str = "border") -> pd.DataFrame:
    """Ripley's K and L for one point pattern in one window.

    :param points: ``(n, 2)`` centroids in array (row, column) order, as
        :func:`spacr.object_distances._centroids` returns them.
    :param mask: boolean array, true where a point could have been.
    :param radii: the radii to evaluate, in the same units as ``spacing``.
    :param spacing: pixel size; ``None`` means the radii are in pixels.
    :param correction: ``"border"`` for the reduced-sample correction, or
        ``"none"`` to see what the correction is worth.
    :returns: ``r``, ``k``, ``l``, ``l_minus_r`` and ``n_contributing``.

    ``l_minus_r`` IS THE COLUMN TO READ. K grows as r squared, so two
    curves are impossible to compare by eye; L is its variance-stabilised
    square root, and under complete spatial randomness L(r) = r exactly, so
    L(r) - r is a flat line at zero whatever the density. Above zero is
    clustering, below is dispersion, and the size of the departure is in
    the units of the image.

    ``n_contributing`` IS NOT DECORATION. It is the number of points the
    border correction kept at that radius, and when it falls into single
    figures the row is arithmetic, not evidence.
    """
    window = np.asarray(mask, dtype=bool)
    if window.ndim != 2:
        raise ValueError("ripley_k is planar: L = sqrt(K / pi) is a 2-D "
                         f"identity and this mask has {window.ndim} axes")
    radii = _as_radii(radii)
    scale = _as_spacing(spacing, 2)
    area = float(window.sum()) * float(np.prod(scale))
    located = _points_in_window(points, window)

    n = len(located)
    frame = pd.DataFrame({"r": radii})
    if n < 2 or area <= 0:
        # ONE POINT HAS NO PATTERN. NaN rather than zero: zero is a
        # measured absence of clustering and this is the absence of a
        # measurement.
        frame["k"] = np.nan
        frame["l"] = np.nan
        frame["l_minus_r"] = np.nan
        frame["n_contributing"] = n
        return frame

    from scipy.spatial import cKDTree

    metric = located * scale
    tree = cKDTree(metric)
    edge = _distance_to_boundary(window, scale)
    at_point = edge[located[:, 0].astype(int), located[:, 1].astype(int)]

    values, contributing = [], []
    for r in radii:
        keep = (at_point >= r) if correction == "border" else np.ones(n, bool)
        kept = int(keep.sum())
        contributing.append(kept)
        if not kept:
            values.append(np.nan)
            continue
        # `- 1` FOR THE POINT ITSELF, which every ball contains.
        counts = tree.query_ball_point(metric[keep], r, return_length=True)
        pairs = float(np.sum(counts) - kept)
        # lambda-hat is n / A from ALL the points -- the density is a
        # property of the window, not of the subset the correction kept.
        values.append(area * pairs / (float(n) * kept))

    k = np.asarray(values, dtype=float)
    with np.errstate(invalid="ignore"):
        l = np.sqrt(k / np.pi)
    frame["k"] = k
    frame["l"] = l
    frame["l_minus_r"] = l - radii
    frame["n_contributing"] = contributing
    return frame


def csr_envelope(mask, n_points: int, radii, *, spacing=None,
                 simulations: int = DEFAULT_SIMULATIONS, seed: int = 0,
                 correction: str = "border") -> pd.DataFrame:
    """The spread of L - r for ``n_points`` scattered at random in ``mask``.

    :param mask: the same window the observed pattern was measured in.
    :param n_points: how many points to scatter -- the observed count.
    :param radii: the same radii.
    :param spacing: the same pixel size.
    :param simulations: how many random patterns.
    :param seed: fixed, so an envelope is reproducible.
    :param correction: the same correction as the observation used.
    :returns: ``r``, ``lo``, ``hi`` and ``mean`` of ``l_minus_r``.

    THE POINTS ARE SCATTERED IN THE MASK, NOT IN ITS BOUNDING BOX. That is
    the entire reason this function exists rather than a lookup of the
    analytic CSR variance: the analytic result assumes a rectangle, and a
    well is a disc with holes in it.

    SAME CORRECTION ON BOTH SIDES. An observation corrected against an
    envelope that was not -- or the reverse -- compares two different
    estimators and the difference between them will look like biology.
    """
    window = np.asarray(mask, dtype=bool)
    radii = _as_radii(radii)
    inside = np.argwhere(window)
    if not len(inside) or n_points < 2:
        return pd.DataFrame({"r": radii, "lo": np.nan, "hi": np.nan,
                             "mean": np.nan})

    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(int(simulations)):
        # WITHOUT REPLACEMENT. Two cells cannot share a centroid pixel, and
        # a duplicated point is a pair at distance zero -- which would put
        # clustering into the null itself and raise the envelope at exactly
        # the small radii the observation is being judged at.
        chosen = inside[rng.choice(len(inside), size=int(n_points),
                                   replace=int(n_points) > len(inside))]
        runs.append(ripley_k(chosen, window, radii, spacing=spacing,
                             correction=correction)["l_minus_r"].to_numpy())
    stack = np.vstack(runs)
    with np.errstate(invalid="ignore"):
        return pd.DataFrame({
            "r": radii,
            # MIN AND MAX, NOT A QUANTILE. This is what makes the envelope
            # an exact test at 2 / (simulations + 1); a percentile of a
            # small sample has no such guarantee.
            "lo": np.nanmin(stack, axis=0),
            "hi": np.nanmax(stack, axis=0),
            "mean": np.nanmean(stack, axis=0),
        })


def ripley_test(points, mask, radii=None, *, spacing=None,
                simulations: int = DEFAULT_SIMULATIONS, seed: int = 0,
                correction: str = "border") -> pd.DataFrame:
    """Observed K against a null that respects the mask.

    :param points: the observed centroids.
    :param mask: the window.
    :param radii: ``None`` asks :func:`suggest_radii`.
    :param spacing: pixel size.
    :param simulations: how many random patterns behind the envelope.
    :param seed: fixed, so the answer is reproducible.
    :param correction: applied to both the observation and the null.
    :returns: the columns of :func:`ripley_k`, plus ``lo``, ``hi``,
        ``clustered`` and ``dispersed``.

    ``clustered`` AND ``dispersed`` ARE PER RADIUS AND THAT IS NOT ONE
    TEST. Twenty radii give twenty chances to leave a 2% envelope, so a
    single excursion is close to meaningless; a run of them at adjacent
    radii is the thing to believe. The columns are deliberately named for
    what they are -- an excursion at this r -- rather than "significant".
    """
    window = np.asarray(mask, dtype=bool)
    if radii is None:
        radii = suggest_radii(window, spacing=spacing)
    observed = ripley_k(points, window, radii, spacing=spacing,
                        correction=correction)
    envelope = csr_envelope(window, len(_points_in_window(points, window)),
                            observed["r"].to_numpy(), spacing=spacing,
                            simulations=simulations, seed=seed,
                            correction=correction)
    merged = observed.merge(envelope, on="r")
    merged["clustered"] = merged["l_minus_r"] > merged["hi"]
    merged["dispersed"] = merged["l_minus_r"] < merged["lo"]
    return merged


# --- the parts that decide the answer ---------------------------------------


def _distance_to_boundary(window: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """How far each pixel of ``window`` is from leaving it.

    THE PAD IS THE WHOLE FUNCTION. `distance_transform_edt` measures the
    distance to the nearest zero *in the array*, so a mask that fills the
    image has no zeros to find and every pixel is reported as infinitely
    interior. One row of background all round makes the frame edge count
    as the boundary it is, and the crop puts the field back on the
    original pixels.
    """
    from scipy.ndimage import distance_transform_edt

    padded = np.pad(window, 1, mode="constant", constant_values=False)
    field = distance_transform_edt(padded, sampling=tuple(scale))
    return field[1:-1, 1:-1]


def _points_in_window(points, window: np.ndarray) -> np.ndarray:
    """``points`` as integer array indices, all of them inside ``window``.

    A POINT OUTSIDE THE WINDOW IS AN ERROR, NOT A ROW TO DROP. The density
    the estimator divides by is the count over the window's area, so a
    point that was never in the window makes every K on the plate wrong by
    a factor nobody would see. Dropping it silently is the same bug with
    the evidence removed.
    """
    located = np.asarray(points, dtype=float)
    if not located.size:
        return np.empty((0, 2), dtype=int)
    located = np.atleast_2d(located)
    if located.shape[1] != 2:
        raise ValueError("points must be (n, 2) in row, column order; "
                         f"got {located.shape[1]} columns")
    index = np.round(located).astype(int)
    inside = np.ones(len(index), dtype=bool)
    for axis, size in enumerate(window.shape):
        inside &= (index[:, axis] >= 0) & (index[:, axis] < size)
    inside[inside] &= window[index[inside, 0], index[inside, 1]]
    if not inside.all():
        raise ValueError(
            f"{int((~inside).sum())} of {len(index)} points lie outside the "
            "mask; K is normalised by the mask's area, so a point that was "
            "never in the window would bias every radius")
    return index


def _as_radii(radii) -> np.ndarray:
    values = np.atleast_1d(np.asarray(radii, dtype=float))
    if not values.size:
        raise ValueError("no radii to evaluate")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("radii must be finite and positive; K(0) is 0 by "
                         "construction and carries no information")
    return values


def _as_spacing(spacing, ndim: int) -> np.ndarray:
    """Per-axis pixel size, ones when there is nothing usable."""
    if spacing is None:
        return np.ones(ndim, dtype=float)
    values = np.atleast_1d(np.asarray(spacing, dtype=float)).ravel()
    if values.size == 1:
        values = np.repeat(values, ndim)
    if values.size != ndim or not np.all(np.isfinite(values)) or \
            np.any(values <= 0):
        return np.ones(ndim, dtype=float)
    return values.astype(float)
