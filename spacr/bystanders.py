"""Label uninfected cells as bystanders or distal by distance to infection.

spaCR measures each cell on its own, but an infected cell can change the
phenotype of its neighbours. Without a neighbourhood column, an uninfected
cell touching an infected one and an uninfected cell on the far side of
the well are indistinguishable rows.

This has two consequences. A bystander phenotype cannot be detected,
because no column separates a bystander from a distant uninfected cell.
In addition, if bystanders are phenotypically shifted and pooled into
"uninfected", the control population is a mixture: its variance is
inflated and the effect size of every infection comparison is reduced.

Distances are not recomputed here. :mod:`spacr.object_distances` computes
a Euclidean distance transform per object type once, at O(image pixels)
rather than O(pairs), and this module reads that field and applies a
threshold and a label.

The threshold is a length, not a pixel count. It is expressed as a
multiple of the measured cell diameter, so it has the same meaning on a
20x and a 63x acquisition and on a plate whose cells are larger. A fixed
micron value would suit one dataset and not the next.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .object_distances import _centroids, surface_distance_transform

#: What a cell's neighbourhood is called, in the order a reader meets them.
#:
#: THREE STATES AND NOT TWO. "infected" and "uninfected" is the split the
#: package has; the whole point of this module is that the second of those
#: is two populations. `distal` is the honest name for what "uninfected"
#: used to mean and now does not.
STATUSES: tuple = ("infected", "bystander", "distal")

#: How far an infected cell's influence is assumed to reach, as a multiple
#: of the measured cell diameter.
#:
#: ONE CELL DIAMETER, WHICH IS THE MOST DEFENSIBLE DEFAULT AND NOT A
#: MEASURED ONE. A cell whose surface is within one diameter of an infected
#: cell's surface is, at most, one cell away from it. That is a statement
#: about adjacency rather than about diffusion, and it is the claim this
#: module can actually support: the reach of a secreted effector is a
#: quantity somebody would have to measure per effector, per medium, per
#: time point.
#:
#: It is a SETTING for that reason. The default has to be honest about
#: being a convention; the number that matters is the user's.
DEFAULT_REACH_IN_DIAMETERS: float = 1.0


def _median_cell_diameter(cell_mask, spacing=None) -> float:
    """The plate's own cell size, in the units ``spacing`` is given in.

    :param cell_mask: the cell label image.
    :param spacing: voxel size; ``None`` means pixels.
    :returns: the median equivalent diameter, or ``0.0`` when nothing
        measurable is present.

    PRIVATE, AND THAT IS DELIBERATE: it takes nothing off the documented
    API surface and adds nothing to it, so wiring the reach up as a setting
    does not move the symbol count a third time in one night.

    BORDER-TOUCHING CELLS ARE DROPPED. A cell clipped by the edge of the
    field has an area that understates its size, and a median pulled down
    by clipped fragments would give a reach shorter than a real cell --
    which is the one error that silently turns every bystander distal.
    Same rule as :func:`spacr.diameter._region_diameters`.

    THE MEDIAN AND NOT THE MEAN, because a fused segmentation is one
    enormous object and a shattered one is many tiny ones; both are
    ordinary on a real plate and both move a mean.
    """
    labels = np.asarray(cell_mask)
    if labels.ndim != 2 or not labels.any():
        return 0.0
    counts = np.bincount(labels.ravel())
    edge = np.unique(np.concatenate([labels[0, :], labels[-1, :],
                                     labels[:, 0], labels[:, -1]]))
    keep = np.ones(counts.size, dtype=bool)
    keep[0] = False
    keep[edge[edge < counts.size]] = False
    areas = counts[keep]
    areas = areas[areas > 0]
    if not areas.size:
        return 0.0
    scale = _as_neighbour_scale(spacing, 2)
    pixel_area = float(scale[0]) * float(scale[1])
    return float(np.median(2.0 * np.sqrt(areas * pixel_area / np.pi)))


def reach_from_diameter(cell_diameter: float,
                        diameters: float = DEFAULT_REACH_IN_DIAMETERS
                        ) -> float:
    """The bystander reach in the units ``cell_diameter`` is measured in.

    :param cell_diameter: the plate's measured cell diameter, from
        :mod:`spacr.diameter`.
    :param diameters: how many diameters count as neighbouring.
    :returns: a distance in the same units, never negative.

    Kept as its own function so the conversion is visible in one place and
    a caller can see that the reach is derived rather than chosen.
    """
    try:
        size = float(cell_diameter)
        scale = float(diameters)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(size) or not np.isfinite(scale):
        return 0.0
    return max(0.0, size * scale)


def distance_to_infected(cell_mask, infected_labels: Iterable[int], *,
                         spacing=None) -> pd.DataFrame:
    """Distance from each cell to the nearest infected cell's surface.

    :param cell_mask: the cell label image.
    :param infected_labels: labels of the cells carrying a pathogen.
    :param spacing: voxel size, so the distances carry physical units.
    :returns: ``label`` and ``distance_to_infected``, one row per cell.

    An infected cell reads 0 and its immediate neighbour reads one pixel
    step, not 0. The field is zero on every infected pixel and grows
    outward, so a cell that shares a border with an infected one has its
    nearest pixel one step away: the minimum over that cell is 1. Only a
    cell whose own pixels are infected reads 0.

    `surface_distance_transform`'s docstring says "two touching objects
    come out at 0", and that is true of the case it describes -- reading
    the field at a point INSIDE the other object, which is what an
    overlapping or contained object gives. Two disjoint labels that merely
    abut do not overlap, so they do not. A threshold expressed in cell
    diameters is far larger than one pixel either way, so a value of 1 for
    an adjacent cell is expected.

    Every distance is ``inf`` when the image has no infected cell, rather
    than 0 or NaN: infinity keeps `distance <= reach` False for every
    finite reach.
    """
    labelled = np.asarray(cell_mask)
    labels = np.unique(labelled)
    labels = labels[labels != 0]
    if not labels.size:
        return pd.DataFrame({"label": np.array([], dtype=labelled.dtype),
                             "distance_to_infected": np.array([], dtype=float)})

    wanted = {int(v) for v in infected_labels}
    infected = np.isin(labelled, list(wanted)) if wanted else np.zeros_like(
        labelled, dtype=bool)
    if not infected.any():
        return pd.DataFrame({
            "label": labels,
            "distance_to_infected": np.full(labels.size, np.inf, dtype=float)})

    field = surface_distance_transform(infected, spacing=spacing)
    from scipy.ndimage import labeled_comprehension

    smallest = labeled_comprehension(
        field, labelled, labels, np.min, np.float64, np.inf)
    return pd.DataFrame({"label": labels,
                         "distance_to_infected": np.asarray(smallest,
                                                            dtype=float)})


def classify(cell_mask, infected_labels: Iterable[int], *, reach: float,
             spacing=None) -> pd.DataFrame:
    """Label every cell infected, bystander or distal.

    :param cell_mask: the cell label image.
    :param infected_labels: labels of the cells carrying a pathogen.
    :param reach: how close an uninfected cell must be to count as a
        bystander, in the same units as ``spacing``. See
        :func:`reach_from_diameter`.
    :param spacing: voxel size.
    :returns: ``label``, ``distance_to_infected`` and ``neighbourhood``.

    A non-positive reach makes every uninfected cell distal. This turns the
    bystander split off without a second setting, and a reach that is not
    a finite number is treated the same way.

    A well with no parasites reports no bystanders. This follows from
    `distance_to_infected` returning infinity there, not from a special
    case.
    """
    frame = distance_to_infected(cell_mask, infected_labels, spacing=spacing)
    wanted = {int(v) for v in infected_labels}
    try:
        limit = float(reach)
    except (TypeError, ValueError):
        limit = 0.0
    if not np.isfinite(limit):
        limit = 0.0

    status = np.full(len(frame), "distal", dtype=object)
    if len(frame):
        is_infected = frame["label"].isin(wanted).to_numpy()
        near = (frame["distance_to_infected"].to_numpy() <= limit) & (limit > 0)
        status[near] = "bystander"
        status[is_infected] = "infected"
    frame["neighbourhood"] = status
    return frame


def counts(frame: pd.DataFrame) -> Dict[str, int]:
    """Count cells in each neighbourhood state, including empty states.

    :param frame: the output of :func:`classify`.
    :returns: every name in :data:`STATUSES`, zero where absent.

    Every key is always present, so a caller comparing wells receives 0 for
    an absent state rather than a ``KeyError``.
    """
    seen = frame["neighbourhood"].value_counts().to_dict() if len(frame) else {}
    return {name: int(seen.get(name, 0)) for name in STATUSES}


def neighbourhood(cell_mask, infected_labels: Iterable[int], *, k: int = 6,
                  spacing=None) -> pd.DataFrame:
    """Fraction of each cell's ``k`` nearest neighbours that are infected.

    :param cell_mask: the cell label image.
    :param infected_labels: labels of the cells carrying a pathogen.
    :param k: how many neighbours to count, excluding the cell itself.
    :param spacing: voxel size, so "nearest" means nearest in real space
        rather than in pixels on an anisotropic stack.
    :returns: ``label``, ``neighbours_counted``, ``neighbours_infected``
        and ``local_infected_fraction``.

    :func:`classify` applies a threshold to one distance. This function
    measures the infected share of the neighbourhood, which separates a
    bystander at the edge of an infection focus from one in its middle.
    Neighbours are found among cell centroids.

    A cell is never counted as its own neighbour, and infected cells also
    receive a fraction, so an infected cell among uninfected ones can be
    distinguished from one inside a cluster.

    `neighbours_counted` is reported rather than assumed to be ``k``. A
    field with fewer than ``k + 1`` cells supplies fewer than ``k``
    neighbours, and the column gives the denominator actually used. Fields
    with fewer than two cells return ``NaN`` fractions.
    """
    from scipy.spatial import cKDTree

    labelled = np.asarray(cell_mask)
    labels, centroids = _centroids(labelled)
    empty = pd.DataFrame({
        "label": np.asarray(labels, dtype=np.int64),
        "neighbours_counted": np.zeros(len(labels), dtype=int),
        "neighbours_infected": np.zeros(len(labels), dtype=int),
        "local_infected_fraction": np.full(len(labels), np.nan)})
    if len(labels) < 2:
        return empty

    scale = _as_neighbour_scale(spacing, centroids.shape[1])
    points = centroids * scale
    wanted = {int(v) for v in infected_labels}
    infected = np.isin(labels, list(wanted)) if wanted else np.zeros(
        len(labels), dtype=bool)

    take = min(int(k) + 1, len(labels)) if int(k) > 0 else 1
    tree = cKDTree(points)
    _distance, index = tree.query(points, k=take)
    index = np.atleast_2d(index)
    if take == 1:
        return empty

    neighbours = index[:, 1:]
    counted = np.full(len(labels), neighbours.shape[1], dtype=int)
    infected_counts = infected[neighbours].sum(axis=1).astype(int)
    frame = pd.DataFrame({
        "label": np.asarray(labels, dtype=np.int64),
        "neighbours_counted": counted,
        "neighbours_infected": infected_counts,
        "local_infected_fraction": infected_counts / counted})
    return frame


def _as_neighbour_scale(spacing, ndim: int) -> np.ndarray:
    """Per-axis multipliers that turn centroid indices into real lengths.

    :param spacing: a scalar, a per-axis sequence, or None.
    :param ndim: how many axes the image has.
    :returns: an array of length ``ndim``, all ones when there is nothing
        usable to apply.

    ONES ON ANYTHING UNREADABLE rather than a raise. An anisotropic stack
    with a mis-typed spacing should give neighbours in pixels -- which is
    what the caller had before -- not no answer at all.
    """
    if spacing is None:
        return np.ones(ndim)
    try:
        values = np.asarray(spacing, dtype=float).ravel()
    except (TypeError, ValueError):
        return np.ones(ndim)
    if values.size == 1:
        values = np.repeat(values, ndim)
    if values.size != ndim or not np.all(np.isfinite(values)) or \
            np.any(values <= 0):
        return np.ones(ndim)
    return values


def compare(frame: pd.DataFrame, feature: str, *,
            statuses: Sequence[str] = STATUSES) -> pd.DataFrame:
    """Summarise one measured feature per neighbourhood state.

    :param frame: measurements carrying a ``neighbourhood`` column, as
        :func:`classify` produces it, joined to the feature table.
    :param feature: the column to summarise.
    :param statuses: which neighbourhoods to report, in order.
    :returns: ``neighbourhood``, ``n``, ``mean``, ``std`` and ``median``.

    Splitting "uninfected" into bystander and distal rows lets a bystander
    effect appear as a difference between groups instead of only as a
    wider control distribution.

    Every requested status is a row, with ``n = 0`` and ``NaN`` statistics
    where the frame has none, so results from different plates have the
    same rows. A missing ``feature`` or ``neighbourhood`` column also
    returns these empty rows.

    No test statistic is computed. The choice of test depends on the
    design (pairing, nesting of wells, the null hypothesis) and is left to
    the caller; this function returns the per-group summaries.
    """
    wanted = [str(name) for name in statuses]
    if feature not in frame.columns or "neighbourhood" not in frame.columns:
        return pd.DataFrame({"neighbourhood": wanted,
                             "n": [0] * len(wanted),
                             "mean": [np.nan] * len(wanted),
                             "std": [np.nan] * len(wanted),
                             "median": [np.nan] * len(wanted)})
    rows = []
    for name in wanted:
        values = pd.to_numeric(
            frame.loc[frame["neighbourhood"] == name, feature],
            errors="coerce").dropna()
        rows.append({
            "neighbourhood": name,
            "n": int(values.size),
            "mean": float(values.mean()) if values.size else np.nan,
            "std": float(values.std(ddof=1)) if values.size > 1 else np.nan,
            "median": float(values.median()) if values.size else np.nan,
        })
    return pd.DataFrame(rows)
