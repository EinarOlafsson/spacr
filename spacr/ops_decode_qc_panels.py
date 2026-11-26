"""Two opt-in genotype QC panels of the decode, read from the stored field cache (items 1042, 1043).

Both are exact ports of the genotype QC v2 analyses of the paper session
(``qc2_arm2.py`` and ``qc2_arm3.py`` with ``qc2_common.py`` and ``qc2_sets.py``):

* the bleed panel (item 1042): the share of the minority reads of a cell
  whose guide is the top guide of its second-nearest nucleus, against a
  distance-matched null, by distance from the neighbour boundary and by spot
  position, for all pairs, with sister pairs excluded and for sisters alone;
* the depth curve (item 1043): silver cells, whose own reads are the
  reads of the guide only, keep ``k`` of those reads; the field is voted again
  and the recovery, the wrong-guide rate and the no-call rate are reported
  per ``k``, with the silver guides permuted between cells as the null.

Nothing here reads an image. The harness of the paper read four things the
decode does not store, and this port replaces them as follows: the tile size
comes from ``ops_geometry`` (or, without it, from the extent of the cache);
the calls of the neighbours come from voting the cache again under the
rule and thresholds of the run; mitotic hand labels and the fixed split-read
reference are not in the decode, so sisters are found by geometry alone and
the silver set has no split-read agreement condition.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

#: Bins of the gap from a read to the boundary of the neighbour nucleus, in pixels.
GAP_BINS = [-np.inf, 0, 1, 3, 6, 10, np.inf]
GAP_NAMES = ["<=0", "0-1", "1-3", "3-6", "6-10", ">10"]

#: Bins of the gap from a read to the boundary of its own nucleus (spot position).
POSITION_BINS = [-np.inf, 0, 3, np.inf]
POSITION_NAMES = ["centre", "border", "outside"]

#: Draws of the distance-matched null, and the least distance in pixels of
#: the nucleus a null pair borrows its top guide from.
NULL_DRAWS = 500
NULL_MIN_DISTANCE = 30.0

#: Read-to-nucleus gap that keeps a read in the bleed pairs, and the gap
#: within which reads name the top guide of a nucleus, and the reads that takes.
OWNED_GAP = 10.0
TOP_GAP = 3.0
TOP_MIN_READS = 2

#: Sister geometry: centres closer than this many mean diameters, areas
#: within this share of the larger.
SISTER_DIAMETERS = 1.5
SISTER_AREA = 0.2

#: Silver set: the reads of the top guide at ``CORE_GAP`` or less, at least
#: ``ANCHOR_MIN_READS`` to anchor and ``SILVER_MIN_READS`` to be silver, the
#: top guide holding ``ANCHOR_SHARE`` of the reads at ``TOP_GAP``; a silver
#: cell lies ``EDGE_PX`` from the tile border and has no neighbour within
#: ``CONFLICT_GAP`` of its boundary that calls another guide.
CORE_GAP = 3.0
ANCHOR_MIN_READS = 4
SILVER_MIN_READS = 5
ANCHOR_SHARE = 0.8
EDGE_PX = 50.0
CONFLICT_GAP = 6.0
CONFLICT_SEARCH_PX = 40.0

#: Depth curve: reads kept per silver cell, draws of each, the distance
#: between the silver cells voted together, the window of peaks voted, and
#: how close a fallback pixel must lie to a removed peak to go with it.
DEPTH_LEVELS = (0, 1, 2, 3, 4, "all")
DEPTH_DRAWS = 5
GROUP_SEPARATION_PX = 40.0
VOTE_WINDOW_PX = 60.0
PIXEL_REMOVAL_PX = 2.0
DEPTH_NULL_DRAWS = 500

#: Tile-bootstrap replicates of every interval.
BOOTSTRAP_REPS = 2000

_NOT_PER_PEAK = ("owned_ids", "owned_areas", "cycles", "nuc_ids",
                 "nuc_centroids", "nuc_areas")


def _interval(values) -> List[float]:
    """The 2.5 and 97.5 percentiles of a set of replicates.

    :param values: the replicates.
    :returns: ``[low, high]``.
    """
    values = np.asarray(values, float)
    return [float(np.nanquantile(values, .025)), float(np.nanquantile(values, .975))]


def tile_bootstrap(count: int, seed: int, reps: int = BOOTSTRAP_REPS) -> np.ndarray:
    """Bootstrap weights over tiles.

    :param count: the number of tiles.
    :param seed: the seed.
    :param reps: the replicates.
    :returns: ``(reps, count)`` how often each tile is drawn in each replicate.
    """
    draws = np.random.default_rng(seed).integers(0, count, (reps, count))
    weights = np.zeros((reps, count))
    np.add.at(weights, (np.repeat(np.arange(reps), count), draws.ravel()), 1)
    return weights


def universe_index(library: Sequence[str], decoys: Sequence[str]):
    """The guides and decoys in one sorted index.

    :param library: the guide barcodes.
    :param decoys: the decoy barcodes.
    :returns: ``(names, index, is_decoy)``: the sorted barcodes, barcode to
        position, and the decoy flag per position.
    """
    names = sorted(set(library) | set(decoys))
    decoy_set = set(decoys)
    return (names, {g: i for i, g in enumerate(names)},
            np.array([g in decoy_set for g in names], bool))


def exact_matches(z: Mapping[str, Any], index: Mapping[str, int], rule: Mapping[str, Any],
                  library: Sequence[str], crosstalk: str) -> np.ndarray:
    """The guide or decoy each cached peak reads exactly, called as the decode calls reads.

    :param z: the field cache.
    :param index: barcode to position from :func:`universe_index`.
    :param rule: the decode rule (library calibration and its parameters).
    :param library: the guide barcodes.
    :param crosstalk: the unmixing method.
    :returns: the position of the barcode each peak reads, -1 for none.
    """
    from .ops_decode import _call

    values = np.asarray(z["values"], np.float32)
    _, codes, _, _ = _call(values, rule, library, crosstalk)
    return np.array([index.get(s, -1) for s in codes], np.int64)


def _digitise(values: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """The bin of each value between the inner edges.

    :param values: the values.
    :param edges: the bin edges, outer edges infinite.
    :returns: bin positions from 0.
    """
    return np.digitize(values, edges[1:-1])


def bleed_field(z: Mapping[str, Any], exact: np.ndarray, is_decoy: np.ndarray,
                threshold: float, rng: np.random.Generator,
                draws: int = NULL_DRAWS):
    """The bleed pairs of one field and their distance-matched null (item 1042).

    A read counts when its strength exceeds the threshold, its nucleus owns
    it, it reads a library guide exactly (decoys excluded) and lies within
    10 px of that nucleus. The top guide of a nucleus is the guide with most
    such reads within 3 px, kept at 2 reads or more. A minority read is a
    read of a topped nucleus whose guide is not its top; its pair is that
    read with its second-nearest nucleus, and it matches when its guide is
    the top guide of that nucleus. The null gives each pair the top guide of
    a random topped nucleus of the field more than 30 px from the read
    own.

    :param z: the field cache.
    :param exact: from :func:`exact_matches`.
    :param is_decoy: the decoy flag per barcode position.
    :param threshold: the read threshold on the spot strength.
    :param rng: the generator of the null draws.
    :param draws: the null draws.
    :returns: ``(pairs, null)``: a DataFrame with one row per pair (``gbin``,
        ``pbin``, ``sister``, ``has_topj``, ``match``) and a ``(draws, pairs)``
        boolean matrix of null matches; ``(None, None)`` without pairs.
    """
    import pandas as pd

    gap = np.asarray(z["gap"])
    g0, g1 = gap[:, 0], gap[:, 1]
    selected = ((np.asarray(z["strength"]) > threshold) & np.asarray(z["owned0"])
                & (exact >= 0) & ~is_decoy[np.maximum(exact, 0)] & (g0 <= OWNED_GAP))
    reads = pd.DataFrame({"i": np.asarray(z["id0"])[selected],
                          "j": np.asarray(z["id1"])[selected],
                          "g": exact[selected], "g0": g0[selected], "g1": g1[selected]})
    top = (reads[reads.g0 <= TOP_GAP].groupby(["i", "g"]).size().rename("n")
           .reset_index().sort_values("n", ascending=False).drop_duplicates("i"))
    top = top[top.n >= TOP_MIN_READS].set_index("i").g
    ids = np.asarray(z["nuc_ids"])
    centres = np.asarray(z["nuc_centroids"], float)
    areas = np.asarray(z["nuc_areas"], float)
    position = pd.Series(np.arange(len(ids)), index=ids)
    reads = reads[reads.i.isin(top.index) & (reads.j >= 0)]
    reads = reads[reads.g != top.reindex(reads.i).to_numpy()]
    if not len(reads):
        return None, None
    own = position.reindex(reads.i).to_numpy()
    other = position.reindex(reads.j).to_numpy()
    known = ~(np.isnan(own) | np.isnan(other))
    reads = reads[known]
    own = own[known].astype(int)
    other = other[known].astype(int)
    if not len(reads):
        return None, None
    distance = np.hypot(*(centres[own] - centres[other]).T)
    diameter = 2 * np.sqrt((areas[own] + areas[other]) / 2 / np.pi)
    sister = (distance < SISTER_DIAMETERS * diameter) & (
        np.abs(areas[own] - areas[other]) / np.maximum(areas[own], areas[other]) <= SISTER_AREA)
    top_other = top.reindex(reads.j).fillna(-1).to_numpy().astype(int)
    pairs = pd.DataFrame({
        "gbin": _digitise(reads.g1.to_numpy(), GAP_BINS),
        "pbin": _digitise(reads.g0.to_numpy(), POSITION_BINS),
        "sister": sister, "has_topj": top_other >= 0,
        "match": reads.g.to_numpy() == top_other})
    topped = top.index.to_numpy()
    topped_position = position.reindex(topped).to_numpy().astype(int)
    topped_guide = top.to_numpy()
    guide = reads.g.to_numpy()
    null = np.zeros((draws, len(reads)), bool)
    for draw in range(draws):
        pick = rng.integers(0, len(topped), len(reads))
        far = np.hypot(*(centres[topped_position[pick]] - centres[own]).T) > NULL_MIN_DISTANCE
        for _ in range(3):
            bad = ~far
            if not bad.any():
                break
            pick[bad] = rng.integers(0, len(topped), bad.sum())
            far = np.hypot(*(centres[topped_position[pick]] - centres[own]).T) > NULL_MIN_DISTANCE
        null[draw] = (topped_guide[pick] == guide) & far
    return pairs, null


def bleed_summary(pairs, null: np.ndarray, tiles: np.ndarray, seed: int):
    """The bleed rates, overall and by distance and spot position.

    :param pairs: the pair table of all fields, with a ``tile`` column.
    :param null: the null matches of all fields side by side.
    :param tiles: the tile of each pair, as a position from 0.
    :param seed: the seed of the tile bootstrap.
    :returns: a DataFrame with a row per group (``all``, ``sisters_excluded``,
        ``sisters_alone``), dimension (``overall``, ``read_to_neighbour_gap``,
        ``spot_position``) and bin: the pairs, the observed match rate, the
        null mean and its 95 % range, and the bleed (observed minus null) with
        tile-bootstrap intervals, all in percent.
    """
    import pandas as pd

    count = int(tiles.max()) + 1
    weights = tile_bootstrap(count, seed)
    match = pairs.match.to_numpy()

    def per_tile(mask, values):
        out = np.zeros(count)
        np.add.at(out, tiles[mask], values[mask].astype(float))
        return out

    def rate(mask):
        n = per_tile(mask, np.ones(len(pairs)))
        m = per_tile(mask, match)
        observed = m.sum() / max(n.sum(), 1)
        boot = (weights @ m) / np.maximum(weights @ n, 1)
        draws = null[:, mask].mean(1) if mask.any() else np.zeros(len(null))
        mean = float(draws.mean())
        return {"pairs": int(n.sum()), "observed_pct": 100 * observed,
                "observed_lo": 100 * _interval(boot)[0], "observed_hi": 100 * _interval(boot)[1],
                "null_mean_pct": 100 * mean,
                "null_lo": 100 * _interval(draws)[0], "null_hi": 100 * _interval(draws)[1],
                "bleed_pct": 100 * (observed - mean),
                "bleed_lo": 100 * _interval(boot - mean)[0],
                "bleed_hi": 100 * _interval(boot - mean)[1]}

    base = pairs.has_topj.to_numpy()
    sister = pairs.sister.to_numpy()
    rows = []
    for group, mask in (("all", base), ("sisters_excluded", base & ~sister),
                        ("sisters_alone", base & sister)):
        rows.append({"group": group, "dimension": "overall", "bin": "all", **rate(mask)})
        for b, name in enumerate(GAP_NAMES):
            rows.append({"group": group, "dimension": "read_to_neighbour_gap", "bin": name,
                         **rate(mask & (pairs.gbin.to_numpy() == b))})
        for b, name in enumerate(POSITION_NAMES):
            rows.append({"group": group, "dimension": "spot_position", "bin": name,
                         **rate(mask & (pairs.pbin.to_numpy() == b))})
    return pd.DataFrame(rows)


def silver_field(z: Mapping[str, Any], exact: np.ndarray, is_decoy: np.ndarray,
                 calls: Mapping[int, int], tile_shape: Tuple[float, float],
                 threshold: float):
    """The anchor and silver cells of one field.

    An anchor is an owned nucleus with at least 4 reads of one library guide
    within 3 px, that guide holding 80 % of the reads of any guide or decoy
    within 3 px, 50 px from the tile border, and with no neighbour within
    6 px of its boundary that the run calls as another guide. A silver cell
    is an anchor with at least 5 reads of its guide, no read of any other
    guide or decoy within 3 px, and every read of its guide central (within
    3 px).

    :param z: the field cache.
    :param exact: from :func:`exact_matches`.
    :param is_decoy: the decoy flag per barcode position.
    :param calls: nucleus -> barcode position of the call of the run.
    :param tile_shape: ``(height, width)`` of the tile.
    :param threshold: the read threshold on the spot strength.
    :returns: a DataFrame (``object_id``, ``guide``, ``n_g1``, ``share``,
        ``anchor``, ``silver``), possibly empty.
    """
    import pandas as pd
    from scipy.spatial import cKDTree

    empty = pd.DataFrame({"object_id": np.zeros(0, np.int64), "guide": np.zeros(0, np.int64),
                          "n_g1": np.zeros(0, np.int64), "share": np.zeros(0),
                          "anchor": np.zeros(0, bool), "silver": np.zeros(0, bool)})
    ids = np.asarray(z["nuc_ids"])
    centres = np.asarray(z["nuc_centroids"], float)
    radius = np.sqrt(np.asarray(z["nuc_areas"], float) / np.pi)
    selected = (np.asarray(z["strength"]) > threshold) & np.asarray(z["owned0"]) & (exact >= 0)
    g0 = np.asarray(z["gap"])[:, 0]
    reads = pd.DataFrame({"o": np.asarray(z["id0"])[selected], "g": exact[selected],
                          "gap": g0[selected]})
    near = reads[reads.gap <= TOP_GAP]
    core = reads[reads.gap <= CORE_GAP]
    library_core = core[~is_decoy[core.g.to_numpy()]]
    if not len(library_core):
        return empty
    top = (library_core.groupby(["o", "g"]).size().rename("n").reset_index()
           .sort_values("n", ascending=False).drop_duplicates("o"))
    total = near.groupby("o").size()
    top = top[top.n >= ANCHOR_MIN_READS].set_index("o")
    if not len(top):
        return empty
    top["tot3"] = total.reindex(top.index).fillna(0)
    joined = near.merge(top[["g"]].rename(columns={"g": "tg"}), left_on="o", right_index=True)
    top["n_g3"] = joined[joined.g == joined.tg].groupby("o").size().reindex(top.index).fillna(0)
    top["other3"] = joined[joined.g != joined.tg].groupby("o").size().reindex(top.index).fillna(0)
    top["share"] = top.n_g3 / top.tot3
    top = top[top.share >= ANCHOR_SHARE]
    position = {o: i for i, o in enumerate(ids.tolist())}
    height, width = tile_shape
    tree = cKDTree(centres)
    keep = []
    for o, row in top.iterrows():
        i = position[o]
        y, x = centres[i]
        edge = min(y, x, height - y, width - x) < EDGE_PX
        neighbours = [j for j in tree.query_ball_point(centres[i], CONFLICT_SEARCH_PX)
                      if j != i and np.hypot(*(centres[j] - centres[i])) - radius[i] - radius[j]
                      <= CONFLICT_GAP]
        conflict = any(calls.get(int(ids[j]), -1) >= 0 and calls.get(int(ids[j]), -1) != row.g
                       for j in neighbours)
        keep.append((not edge) and (not conflict))
    top = top[np.array(keep, bool)]
    if not len(top):
        return empty
    own_core = joined[joined.o.isin(top.index) & (joined.g == joined.tg)]
    central = (own_core.groupby("o").gap.max() <= CORE_GAP).reindex(top.index).fillna(False)
    silver = (top.other3 == 0) & central & (top.n >= SILVER_MIN_READS)
    return pd.DataFrame({"object_id": top.index.to_numpy(), "guide": top.g.to_numpy(),
                         "n_g1": top.n.to_numpy(), "share": top.share.to_numpy(),
                         "anchor": True, "silver": silver.to_numpy()})


def _vote(evidence, rule, library, matcher, decoy_set, crosstalk) -> Dict[int, dict]:
    """The calls of one field under a rule.

    :param evidence: the field evidence.
    :param rule: from :func:`spacr.ops_decode._cached_rule`.
    :param library: the guide barcodes.
    :param matcher: from :func:`spacr.ops_decode.matcher_for`.
    :param decoy_set: the decoy set from the same call.
    :param crosstalk: the unmixing method.
    :returns: object -> call, as :func:`spacr.ops_decode.field_calls` returns.
    """
    from .ops_decode import field_calls

    return field_calls(evidence, rule, library, matcher, decoy_set,
                       threshold=rule["threshold"], footprint=rule["footprint"],
                       max_distance=rule["max_distance"],
                       min_fraction=rule["min_fraction"],
                       single_read_quality=rule["single_read_quality"],
                       crosstalk=crosstalk)["calls"]


def depth_field(z: Mapping[str, Any], exact: np.ndarray, silver, names: Sequence[str],
                rule, library, matcher, decoy_set, crosstalk: str, site: int, seed: int,
                levels=DEPTH_LEVELS, draws: int = DEPTH_DRAWS):
    """The depth curve of the silver cells of one field (item 1043).

    Silver cells are split greedily into groups at least 40 px apart, so the
    reads removed for one cell cannot enter the vote of another. Only peaks
    within 60 px of the silver cells of a group are kept. A silver cell keeps ``k``
    of its own-guide reads, drawn without replacement (its reads are the
    peaks assigned to it whose base call matches its silver guide exactly),
    and every other peak stays. Fallback pixels within 2 px of a removed peak
    go with it. The group is voted again under the rule of the run.

    :param z: the field cache.
    :param exact: from :func:`exact_matches`.
    :param silver: the silver cells of the field (``object_id``, ``guide``).
    :param names: the barcode of each position.
    :param rule: from :func:`spacr.ops_decode._cached_rule`.
    :param library: the guide barcodes.
    :param matcher: from :func:`spacr.ops_decode.matcher_for`.
    :param decoy_set: the decoy set from the same call.
    :param crosstalk: the unmixing method.
    :param site: the field.
    :param seed: the seed.
    :param levels: the reads kept, ``"all"`` keeps every read.
    :param draws: draws of each level but 0 and ``"all"``.
    :returns: a DataFrame with a row per cell, level, draw: ``object_id``,
        ``k``, ``draw``, ``call`` (the barcode, -1 for no call) and ``silver_guide``.
    """
    import pandas as pd
    from scipy.spatial import cKDTree

    ids = np.asarray(z["nuc_ids"])
    centres = np.asarray(z["nuc_centroids"], float)
    position = {o: i for i, o in enumerate(ids.tolist())}
    guide_of = dict(zip(silver.object_id.tolist(), silver.guide.tolist()))
    groups: List[List[int]] = []
    for o in silver.object_id.tolist():
        centre = centres[position[o]]
        for group in groups:
            if all(np.hypot(*(centres[position[x]] - centre)) >= GROUP_SEPARATION_PX for x in group):
                group.append(o)
                break
        else:
            groups.append([o])
    id0 = np.asarray(z["id0"])
    own_reads = {o: np.flatnonzero((id0 == o) & (exact == guide_of[o])) for o in guide_of}
    index = {g: i for i, g in enumerate(names)}
    rng = np.random.default_rng([seed, 3, site])
    peaks = np.asarray(z["peaks"], float)
    pixel_yx = np.asarray(z["pix_yx"], float) if "pix_yx" in z else np.zeros((0, 2))
    total = len(peaks)
    rows = []
    for group in groups:
        window = cKDTree(centres[[position[o] for o in group]]).query(peaks)[0] > VOTE_WINDOW_PX
        for k in levels:
            for draw in (range(draws) if k not in (0, "all") else [0]):
                drop = window.copy()
                for o in group:
                    reads = own_reads[o]
                    if k == "all":
                        continue
                    kept = rng.choice(reads, min(k, len(reads)), replace=False) if k else []
                    drop[np.setdiff1d(reads, kept)] = True
                removed = peaks[drop & ~window]
                if len(removed) and len(pixel_yx):
                    pixel_drop = cKDTree(removed).query(pixel_yx)[0] <= PIXEL_REMOVAL_PX
                else:
                    pixel_drop = np.zeros(len(pixel_yx), bool)
                evidence = {}
                for key in z.keys():
                    value = z[key]
                    if key.startswith("pix_"):
                        evidence[key] = value[~pixel_drop]
                    elif (hasattr(value, "shape") and value.shape[:1] == (total,)
                          and key not in _NOT_PER_PEAK):
                        evidence[key] = value[~drop]
                    else:
                        evidence[key] = value
                calls = _vote(evidence, rule, library, matcher, decoy_set, crosstalk)
                for o in group:
                    hit = calls.get(o)
                    rows.append((o, str(k), draw, index.get(hit["guide"], -2) if hit else -1))
    frame = pd.DataFrame(rows, columns=["object_id", "k", "draw", "call"])
    frame["silver_guide"] = frame.object_id.map(guide_of)
    return frame


def depth_summary(cells, seed: int, null_draws: int = DEPTH_NULL_DRAWS):
    """The recovery, wrong-guide and no-call rates per reads kept.

    :param cells: the rows of :func:`depth_field` of every field, with a
        ``tile`` column of positions from 0.
    :param seed: the seed of the bootstrap and the permutation null.
    :param null_draws: permutations of the silver guides within a tile.
    :returns: ``(table, null)``: a DataFrame per level ``k`` with the cells,
        the recovered, wrong-guide and no-call percentages and their
        tile-bootstrap intervals, and the permutation null of the ``all`` level
        as ``{"mean_pct", "lo", "hi"}``.
    """
    import pandas as pd

    count = int(cells.tile.max()) + 1
    weights = tile_bootstrap(count, seed)
    cells = cells.assign(rec=cells.call == cells.silver_guide)
    cells["wrong"] = (cells.call >= 0) & ~cells.rec
    cells["none"] = cells.call < 0
    rows = []
    for k, part in cells.groupby("k", sort=False):
        tile = part.tile.to_numpy()
        n = np.bincount(tile, minlength=count).astype(float)
        row = {"k": k, "n": int(len(part)), "cells": int(part.object_id.nunique())}
        for name in ("rec", "wrong", "none"):
            v = np.bincount(tile, weights=part[name].to_numpy().astype(float), minlength=count)
            interval = _interval((weights @ v) / np.maximum(weights @ n, 1))
            row[f"{name}_pct"] = 100 * v.sum() / max(n.sum(), 1)
            row[f"{name}_lo"], row[f"{name}_hi"] = 100 * interval[0], 100 * interval[1]
        rows.append(row)
    full = cells[cells.k == "all"]
    values = []
    if len(full):
        groups = [np.flatnonzero(full.tile.to_numpy() == t) for t in range(count)]
        call = full.call.to_numpy()
        silver = full.silver_guide.to_numpy()
        for draw in range(null_draws):
            rng = np.random.default_rng([seed, 33, draw])
            permuted = silver.copy()
            for index in groups:
                if len(index):
                    permuted[index] = rng.permutation(silver[index])
            values.append(100 * float((call == permuted).mean()))
    if values:
        interval = _interval(values)
        null = {"mean_pct": float(np.mean(values)), "lo": interval[0], "hi": interval[1]}
    else:
        null = {"mean_pct": float("nan"), "lo": float("nan"), "hi": float("nan")}
    return pd.DataFrame(rows), null


def bleed_figure(summary, path: str, title: str = "") -> str:
    """Draw the bleed panel: observed match rate against the null, by distance and position.

    :param summary: from :func:`bleed_summary`.
    :param path: the PNG to write.
    :param title: a heading.
    :returns: the path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [6, 3]})
    for ax, dim, names in ((axes[0], "read_to_neighbour_gap", GAP_NAMES),
                           (axes[1], "spot_position", POSITION_NAMES)):
        for group, colour, shift in (("sisters_excluded", "#2a6f97", -0.18), ("all", "#8d99ae", 0.18)):
            part = summary[(summary.group == group) & (summary.dimension == dim)].set_index("bin")
            part = part.reindex(names)
            x = np.arange(len(names)) + shift
            ax.bar(x, part.observed_pct, 0.32, color=colour, alpha=0.6, label=f"observed, {group.replace('_', ' ')}")
            ax.errorbar(x, part.observed_pct, yerr=[part.observed_pct - part.observed_lo,
                                                    part.observed_hi - part.observed_pct],
                        fmt="none", ecolor="black", lw=0.6)
            ax.plot(x, part.null_mean_pct, "_", color="black", ms=9, mew=1.2,
                    label="ring-matched null" if group == "all" else None)
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, fontsize=7)
        ax.tick_params(labelsize=7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_xlabel("read to neighbour boundary (px)", fontsize=8)
    axes[1].set_xlabel("spot position in own cell", fontsize=8)
    axes[0].set_ylabel("reads matching the neighbour guide (%)", fontsize=8)
    axes[0].legend(fontsize=6, frameon=False)
    if title:
        fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def depth_figure(table, null: Mapping[str, float], path: str, title: str = "") -> str:
    """Draw the depth curve: recovery and wrong-guide rate against the reads kept.

    :param table: from :func:`depth_summary` (one well).
    :param null: the permutation null of the same call.
    :param path: the PNG to write.
    :param title: a heading.
    :returns: the path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = [k for k in table.k.tolist()]
    x = np.arange(len(levels))
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for name, colour, label in (("rec", "#2a6f97", "recovered"), ("wrong", "#c1121f", "wrong guide"),
                                ("none", "#8d99ae", "no call")):
        ax.plot(x, table[f"{name}_pct"], "-o", color=colour, ms=3, lw=1, alpha=0.8, label=label)
        ax.fill_between(x, table[f"{name}_lo"], table[f"{name}_hi"], color=colour, alpha=0.15, lw=0)
    if np.isfinite(null.get("mean_pct", np.nan)):
        ax.axhline(null["mean_pct"], color="black", lw=0.6, ls=":", label="permuted null")
    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("own-guide reads kept per silver cell", fontsize=8)
    ax.set_ylabel("silver cells (%)", fontsize=8)
    ax.legend(fontsize=6, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _tile_shapes(db: str, plate: str, well: str) -> Dict[int, Tuple[float, float]]:
    """The tile size of each field from ``ops_geometry``.

    :param db: the measurements database.
    :param plate: the plate.
    :param well: the well.
    :returns: site -> ``(height, width)``; empty when the table is absent.
    """
    from .ops_store import read_table

    try:
        frame = read_table(db, "ops_geometry")
    except Exception:
        return {}
    frame = frame[(frame["plate"].astype(str) == str(plate)) & (frame["well"].astype(str) == str(well))]
    if "cycle" in frame.columns and len(frame):
        frame = frame[frame["cycle"] == frame["cycle"].min()]
    return {int(r.site): (float(r.tile_height), float(r.tile_width)) for r in frame.itertuples()}


def _extent(z: Mapping[str, Any]) -> Tuple[float, float]:
    """A tile size estimated from the extent of the cache, for a run without ``ops_geometry``.

    :param z: the field cache.
    :returns: ``(height, width)``, the largest coordinate of a peak, pixel or nucleus plus one.
    """
    parts = [np.asarray(z[k], float) for k in ("peaks", "pix_yx", "nuc_centroids")
             if k in z and len(z[k])]
    top = np.max(np.vstack(parts), axis=0) + 1
    return float(top[0]), float(top[1])


def _well_thresholds(db: str, well: str, settings: Mapping[str, Any]):
    """The thresholds a well was decoded at, and the fields its auto thresholds were tuned on.

    :param db: the measurements database.
    :param well: the well.
    :param settings: the resolved settings.
    :returns: ``(read_threshold, single_read_quality, pixel_quality, tune_sites)``.
    """
    from .ops_decode import results_folder

    path = os.path.join(results_folder(db), f"decode_thresholds_{well}.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            chosen = json.load(handle)
        return (float(chosen["read_threshold"]), float(chosen["single_read_quality"]),
                float(chosen["pixel_quality"]), [int(s) for s in chosen.get("tune_sites", [])])
    return (float(settings["ops_read_threshold"]), float(settings["ops_single_read_quality"]),
            float(settings["ops_pixel_quality"]), [])


def qc_panels(fields: Sequence[Mapping[str, Any]], rule: Mapping[str, Any],
              library: Sequence[str], decoys: Sequence[str], crosstalk: str, *,
              seed: int, bleed: bool, depth: bool, tile_shapes: Mapping[int, Tuple[float, float]],
              tune_sites: Sequence[int] = (), null_draws: int = NULL_DRAWS,
              depth_levels=DEPTH_LEVELS, depth_draws: int = DEPTH_DRAWS,
              depth_null_draws: int = DEPTH_NULL_DRAWS) -> Dict[str, Any]:
    """Both panels of one well, from its stored field caches.

    :param fields: ``{"site", "z"}`` per field, ``z`` the opened cache.
    :param rule: from :func:`spacr.ops_decode._cached_rule`.
    :param library: the guide barcodes.
    :param decoys: the decoy barcodes.
    :param crosstalk: the unmixing method.
    :param seed: ``ops_decode_qc_seed``.
    :param bleed: compute the bleed panel.
    :param depth: compute the depth curve.
    :param tile_shapes: site -> ``(height, width)``.
    :param tune_sites: fields the thresholds were tuned on; the depth curve
        scores the others, when there are others.
    :param null_draws: draws of the bleed null.
    :param depth_levels: the reads kept in the depth curve.
    :param depth_draws: draws per level.
    :param depth_null_draws: permutations of the depth null.
    :returns: ``bleed_pairs``, ``bleed_summary``, ``silver``, ``depth_cells``,
        ``depth_summary`` and ``depth_null`` as computed.
    """
    import pandas as pd

    from .ops_decode import matcher_for

    names, index, is_decoy = universe_index(library, decoys)
    matcher, decoy_set = matcher_for(library, decoys)
    out: Dict[str, Any] = {}
    pair_frames, null_parts, silver_frames, depth_frames = [], [], [], []
    scored = [f for f in fields if int(f["site"]) not in set(tune_sites)] or list(fields)
    scored_sites = {int(f["site"]) for f in scored}
    sites = sorted(int(f["site"]) for f in fields)
    tile_of = {s: i for i, s in enumerate(sites)}
    for field in fields:
        site, z = int(field["site"]), field["z"]
        if hasattr(z, "files"):
            z = {key: z[key] for key in z.files}
        exact = exact_matches(z, index, rule, library, crosstalk)
        if bleed:
            rng = np.random.default_rng([seed, site])
            pairs, null = bleed_field(z, exact, is_decoy, rule["threshold"], rng, null_draws)
            if pairs is not None:
                pairs["site"] = site
                pairs["tile"] = tile_of[site]
                pair_frames.append(pairs)
                null_parts.append(null)
        if depth:
            calls = {int(o): index.get(c["guide"], -1)
                     for o, c in _vote(z, rule, library, matcher, decoy_set, crosstalk).items()}
            shape = tile_shapes.get(site) or _extent(z)
            sets = silver_field(z, exact, is_decoy, calls, shape, rule["threshold"])
            if len(sets):
                sets["site"] = site
                silver_frames.append(sets)
            chosen = sets[sets.silver]
            if site in scored_sites and len(chosen):
                cells = depth_field(z, exact, chosen, names, rule, library, matcher, decoy_set,
                                    crosstalk, site, seed, depth_levels, depth_draws)
                cells["site"] = site
                cells["tile"] = tile_of[site]
                depth_frames.append(cells)
    if bleed and pair_frames:
        pairs = pd.concat(pair_frames, ignore_index=True)
        null = np.concatenate(null_parts, axis=1)
        pairs["tile"] = pairs.tile.map({t: i for i, t in enumerate(sorted(set(pairs.tile)))})
        out["bleed_pairs"] = pairs
        out["bleed_summary"] = bleed_summary(pairs, null, pairs.tile.to_numpy(), seed)
    if depth:
        if silver_frames:
            out["silver"] = pd.concat(silver_frames, ignore_index=True)
        if depth_frames:
            cells = pd.concat(depth_frames, ignore_index=True)
            cells["tile"] = cells.tile.map({t: i for i, t in enumerate(sorted(set(cells.tile)))})
            out["depth_cells"] = cells
            out["depth_summary"], out["depth_null"] = depth_summary(cells, seed, depth_null_draws)
    return out


def run_qc_panels(db: str, plate: str, wells: Sequence[str], settings: Mapping[str, Any],
                  library: Sequence[str], folder: str, fields_of_well=None) -> Dict[str, Any]:
    """The opt-in QC panels of a plate, written under ``results/ops_decode_qc``.

    Runs the bleed panel when ``ops_decode_qc_bleed_panel`` is on and the
    depth curve when ``ops_decode_qc_depth_curve`` is on, per well, from the
    stored field caches under the rule and thresholds the well was decoded
    at, seeded with ``ops_decode_qc_seed``. Writes ``bleed_panel_<well>.parquet``,
    ``bleed_pairs_<well>.parquet`` and ``bleed_panel_<well>.png``;
    ``depth_curve_<well>.parquet``, ``depth_curve_cells_<well>.parquet``,
    ``silver_cells_<well>.parquet`` and ``depth_curve_<well>.png``.

    :param db: the measurements database.
    :param plate: the plate.
    :param wells: the wells decoded.
    :param settings: the resolved settings.
    :param library: the guide barcodes.
    :param folder: ``results/ops_decode_qc``.
    :param fields_of_well: ``well -> [{"site", "z"}]``; None opens the caches.
    :returns: ``{well: {name: path}}`` of the files written.
    :raises ValueError: when a well has no stored field cache.
    """
    from .ops_decode import _cached_fields, _cached_rule, _flag_or, _qc_number

    bleed = _flag_or(settings, "ops_decode_qc_bleed_panel")
    depth = _flag_or(settings, "ops_decode_qc_depth_curve")
    destination = os.path.dirname(os.path.abspath(db))
    seed = _qc_number(settings, "ops_decode_qc_seed", 0, int)
    report: Dict[str, Any] = {}
    for well in sorted(wells):
        if fields_of_well is not None:
            fields = fields_of_well[well]
        else:
            fields = [{"site": f["site"], "z": f["z"]}
                      for f in _cached_fields(destination, plate, [well])]
        if not fields:
            raise ValueError("the decode QC panels need the field evidence the decode stores "
                             "with ops_store_field_cache on; decode again with it")
        read_threshold, single, pixel, tune = _well_thresholds(db, well, settings)
        rule, lib, decoys, crosstalk = _cached_rule(
            settings, library, read_threshold=read_threshold,
            single_read_quality=single, pixel_quality=pixel)
        result = qc_panels(fields, rule, lib, decoys, crosstalk, seed=seed, bleed=bleed,
                           depth=depth, tile_shapes=_tile_shapes(db, plate, well),
                           tune_sites=tune)
        paths: Dict[str, str] = {}
        title = f"{plate} {well}"
        if "bleed_summary" in result:
            paths["bleed_panel"] = os.path.join(folder, f"bleed_panel_{well}.parquet")
            result["bleed_summary"].to_parquet(paths["bleed_panel"], index=False)
            paths["bleed_pairs"] = os.path.join(folder, f"bleed_pairs_{well}.parquet")
            result["bleed_pairs"].to_parquet(paths["bleed_pairs"], index=False)
            paths["bleed_figure"] = bleed_figure(
                result["bleed_summary"], os.path.join(folder, f"bleed_panel_{well}.png"), title)
        if "silver" in result:
            paths["silver_cells"] = os.path.join(folder, f"silver_cells_{well}.parquet")
            result["silver"].to_parquet(paths["silver_cells"], index=False)
        if "depth_summary" in result:
            paths["depth_curve"] = os.path.join(folder, f"depth_curve_{well}.parquet")
            result["depth_summary"].assign(
                null_recovery_pct=result["depth_null"]["mean_pct"],
                null_recovery_lo=result["depth_null"]["lo"],
                null_recovery_hi=result["depth_null"]["hi"]).to_parquet(
                    paths["depth_curve"], index=False)
            paths["depth_cells"] = os.path.join(folder, f"depth_curve_cells_{well}.parquet")
            result["depth_cells"].to_parquet(paths["depth_cells"], index=False)
            paths["depth_figure"] = depth_figure(
                result["depth_summary"], result["depth_null"],
                os.path.join(folder, f"depth_curve_{well}.png"), title)
        report[well] = paths
    return report
