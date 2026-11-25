"""Which field owns a cell imaged in two or more overlapping fields.

Neighbouring fields of a tiled acquisition overlap, so a cell in the overlap
is segmented once per field that sees it. Counting every copy weights that
cell two or more times in its profile and in its gene. This module is the one
place that decides which copy is the cell:

* :func:`match_copies_by_centroid` finds the copies of one cell from their
  nucleus centroids in stitched (well) coordinates: whole copies in two
  fields whose centroids are mutual nearest neighbours within a radius, and
  each cut copy joined to the nearest whole copy;
* :func:`assign_field_owners` groups the copies into physical cells (the
  connected components of the copy edges) and names one owner per cell.

The rule (``'nearest_centre'``, the default): a cell belongs to the field
whose centre is nearest the cell centroid, among the fields that hold it
whole; ties go to the lower field index, then the lower label. A copy cut by
its field edge never owns a cell, so a cell cut by the edge of every field
that holds it has no owner. Every cell with a whole copy has exactly one.

``'largest'`` (the largest copy, whole or cut, ties to the lower field, then
label) is deprecated. It reproduces outputs of spaCR runs made before
settings schema 963, and new runs should not use it.
"""
from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import numpy as np

#: The owner rules, the first the default; the others are deprecated.
FIELD_OWNER_RULES = ("nearest_centre", "largest")

#: The rules kept only to reproduce outputs of earlier runs.
DEPRECATED_FIELD_OWNER_RULES = ("largest",)


def field_owner_rule(value) -> str:
    """``value`` as one of :data:`FIELD_OWNER_RULES`; None is the default.

    :raises ValueError: naming the rules, for any other value.
    """
    from .qt.i18n import tr

    rule = str(value or FIELD_OWNER_RULES[0]).strip().lower()
    if rule not in FIELD_OWNER_RULES:
        raise ValueError(tr(
            "the field owner rule must be one of {rules}; got {value}",
            rules=list(FIELD_OWNER_RULES), value=rule))
    return rule


def field_centre(transform, shape) -> np.ndarray:
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


def match_copies_by_centroid(cells, neighbours: Mapping[int, Sequence[int]],
                             radius: float, cut_radius: float = None):
    """Edges between the copies of one cell in overlapping fields.

    :param cells: one row per copy with ``site``, ``label``, ``well_y``,
        ``well_x`` (the nucleus centroid in stitched coordinates) and
        ``whole``.
    :param neighbours: ``{site: [sites whose footprint overlaps]}``.
    :param radius: the largest centroid distance, in well pixels, of two
        whole copies of one cell.
    :param cut_radius: the same for a copy cut by its field edge, whose
        centroid moves with the part that is missing; three times
        ``radius`` when None.
    :returns: ``site_a, label_a, site_b, label_b, distance``. Whole copies
        are paired per pair of overlapping fields when each is the nearest
        whole copy of the other within ``radius``. A cut copy is joined to
        the one nearest whole copy of an overlapping field within
        ``cut_radius``, so it can never join two cells; cut copies with no
        whole copy near are paired with each other as whole copies are.
        Only whole copies can own a cell; the cut edges let a table count
        each physical cell once, owned or not.
    """
    import pandas as pd
    from scipy.spatial import cKDTree

    columns = ["site_a", "label_a", "site_b", "label_b", "distance"]
    cut_radius = 3.0 * float(radius) if cut_radius is None else float(
        cut_radius)
    whole = np.asarray(cells["whole"], bool)
    points = cells[["well_y", "well_x"]].to_numpy(float).reshape(-1, 2)
    sites = cells["site"].to_numpy(np.int64)
    labels = cells["label"].to_numpy(np.int64)
    rows = {int(s): np.flatnonzero(sites == s) for s in np.unique(sites)}
    found = []

    def keep(first, second, distance):
        found.append(pd.DataFrame({
            "site_a": sites[first], "label_a": labels[first],
            "site_b": sites[second], "label_b": labels[second],
            "distance": distance}))

    pairs = [(a, b) for a in sorted(rows) for b in sorted(
        int(s) for s in neighbours.get(a, ())) if b > a and b in rows]
    for a, b in pairs:
        first, second = rows[a][whole[rows[a]]], rows[b][whole[rows[b]]]
        i, j, distance = _mutual_pairs(points[first], points[second], radius)
        if i.size:
            keep(first[i], second[j], distance)
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
        if joined.any():
            keep(cut[joined], partner[joined], best[joined])
        lone.append(cut[~joined])
    lone = np.concatenate(lone) if lone else np.empty(0, np.int64)
    alone = np.zeros(len(cells), bool)
    alone[lone] = True
    for a, b in pairs:
        first, second = rows[a][alone[rows[a]]], rows[b][alone[rows[b]]]
        i, j, distance = _mutual_pairs(points[first], points[second],
                                       cut_radius)
        if i.size:
            keep(first[i], second[j], distance)
    if not found:
        return pd.DataFrame({c: pd.Series(dtype=float if c == "distance"
                                          else np.int64) for c in columns})
    return pd.concat(found, ignore_index=True)[columns]


def assign_field_owners(cells, edges, *, rule: str = "nearest_centre",
                        centres: Mapping[int, Sequence[float]] = None):
    """Physical cells and the one field that owns each.

    :param cells: one row per copy: ``site``, ``label``; ``area`` for
        ``'largest'``; ``whole``, ``well_y`` and ``well_x`` for
        ``'nearest_centre'``.
    :param edges: ``site_a, label_a, site_b, label_b``: two copies of one
        cell (:func:`match_copies_by_centroid`, or mask overlaps).
    :param rule: one of :data:`FIELD_OWNER_RULES`.
    :param centres: ``{site: (y, x)}``, the well centre of each field
        (:func:`field_centre`), for ``'nearest_centre'``.
    :returns: ``cells`` with ``component`` (the connected component of the
        undirected edge graph), ``n_copies``, ``owner_field``,
        ``owner_label`` (both missing when no copy can own the cell),
        ``is_owner``, ``owned_area_share`` (when ``area`` is given) and,
        for ``'nearest_centre'``, ``centre_distance_px``.
    """
    import pandas as pd
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    rule = field_owner_rule(rule)
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
    else:
        first = second = np.empty(0, np.int64)
    count = len(cells)
    graph = coo_matrix((np.ones(len(first)), (first, second)),
                       shape=(count, count))
    _n, component = connected_components(graph, directed=False)
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
    return cells


def owner_counts(cells) -> Tuple[int, int, int]:
    """``(physical cells, owned cells, cells without an owner)`` in a table
    of :func:`assign_field_owners`."""
    components = cells.drop_duplicates("component")
    owned = int(cells["is_owner"].sum())
    return int(len(components)), owned, int(len(components) - owned)
