"""Max-projection clustering over feature views of an OPS gene profile (item 992, alpha).

Some phenotypes are broad and show on many columns, others are thin and show only on the few
columns that carry them. Each view (a named subset of the profile's columns) is embedded, whitened
on the non-targeting genes and turned into calibrated gene-pair scores: every pair similarity
becomes a normal score against that view's own guide-permutation null, pooled within bins of the
two genes' cell counts. The projection keeps, for every pair, its strongest calibrated score over
the views, and a joint permutation null of that maximum (Westfall-Young) gives each pair a p value.
The pairs then form a graph that Leiden clusters, and every cluster is labelled by the views its
edges came from.

The null distributions are kept as counts on a fixed grid, so memory does not grow with the number
of permutations; every draw is seeded from ``ops_maxproj_seed`` and its index, in a fixed order,
so the results do not depend on the worker count.
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache, partial
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SIMILARITY_GRID = np.linspace(-1.0, 1.0, 4001)
SCORE_GRID = np.linspace(-12.0, 12.0, 4801)
MAXPROJ_GRAPHS = ("knn", "phate")
MAXPROJ_SIMILARITIES = ("cosine", "shrunk")
DEFAULT_SEED = 20261119
DEFAULT_RESOLUTIONS = tuple(float(r) for r in np.logspace(np.log10(0.05), np.log10(50.0), 20))
THIN_COLUMNS = 100
LISTED_SHARE = 0.2


def _packed_indices(n, a, b):
    """Locate unordered, distinct gene pairs in row-ordered upper-triangle storage."""
    low, high = np.minimum(a, b), np.maximum(a, b)
    return n * low - low * (low + 1) // 2 + high - low - 1

def _pair_values(values, n, a, b):
    """Read the same gene pairs from square or packed storage."""
    values = np.asarray(values)
    return values[a, b] if values.ndim == 2 else values[_packed_indices(n, a, b)]

def _pair_row(values, n, row, diagonal):
    """Materialise one symmetric pair row, keeping the complete matrix unallocated."""
    columns = np.arange(n)
    selected = columns != row
    result = np.empty(n, dtype=values.dtype)
    result[selected] = _pair_values(values, n, row, columns[selected])
    result[row] = diagonal
    return result

def _projected_strength_rows(pvalues, n, edge_p, valid):
    """Yield the existing symmetric threshold-and-best graph strengths one row at a time."""
    best = np.full(n, -1, int)
    for row in range(n):
        probabilities = _pair_row(pvalues, n, row, 1)
        eligible = np.ones(n, bool) if valid is None else _pair_row(valid, n, row, False)
        eligible[row] = False
        strength = -np.log10(np.clip(probabilities, 1e-300, 1.0))
        if eligible.any():
            best[row] = int(np.argmax(np.where(eligible, strength, -np.inf)))
    for row in range(n):
        probabilities = _pair_row(pvalues, n, row, 1)
        eligible = np.ones(n, bool) if valid is None else _pair_row(valid, n, row, False)
        eligible[row] = False
        keep = (probabilities < edge_p) & eligible
        if best[row] >= 0:
            keep[best[row]] = True
        keep |= best == row
        strength = -np.log10(np.clip(probabilities, 1e-300, 1.0))
        yield np.where(keep & eligible, strength, -np.inf)


def _pair_rows(n):
    """Yield a gene index and the corresponding packed upper-triangle slice."""
    offset = 0
    for row in range(n - 1):
        stop = offset + n - row - 1
        yield row, slice(offset, stop)
        offset = stop

@lru_cache(maxsize=16)
def _covered_pairs(n, covered_bytes, packed):
    """Cache an immutable pair mask for one exact gene-coverage vector."""
    covered = np.frombuffer(covered_bytes, dtype=bool, count=n)
    if packed:
        result = np.empty(n * (n - 1) // 2, bool)
        for row, span in _pair_rows(n):
            result[span] = covered[row] & covered[row + 1:]
    else:
        result = covered[:, None] & covered[None, :]
    result.flags.writeable = False
    return result

def _coverage_pairs(covered, packed):
    """Return the pair mask in the requested layout, without global pair indices."""
    covered = np.asarray(covered, bool)
    return _covered_pairs(len(covered), covered.tobytes(), packed)

def _joint_coverage(coverage, packed):
    """Return pairs sharing at least one covered view, or no restriction."""
    if coverage is None:
        return None
    result = _coverage_pairs(coverage[:, 0], packed).copy()
    for column in range(1, coverage.shape[1]):
        result |= _coverage_pairs(coverage[:, column], packed)
    return result

def _view_arrays(rows, control, variance, seed, similarity, covered, dtype, max_missing):
    """Embed covered rows with the shared PCA owner and compute similarity scales."""
    from .ops_reference import embed_gene_profiles
    from .ops_tvn import tvn_on_controls

    values = np.asarray(rows, float)
    selected = np.flatnonzero(covered)
    values = values if covered.all() else values[selected]
    embedded = embed_gene_profiles(values, variance, seed, max_missing=max_missing)
    controls = None if control is None else np.asarray(control, bool)[selected]
    if controls is not None and controls.sum() > embedded.shape[1]:
        embedded, _ = tvn_on_controls(embedded, controls)
    embedded = np.asarray(embedded, dtype=dtype)
    norm = (embedded ** 2).sum(axis=1)
    if similarity == "shrunk":
        norm = norm + float(norm.mean() if controls is None else norm[controls].mean())
    scale = np.sqrt(np.clip(norm, 1e-24, None))
    return selected, embedded, scale

def _column_rows(rows, covered, seed, view_index):
    """Shuffle each column independently among covered genes in one view and draw."""
    selected = np.arange(len(rows)) if covered is None else np.flatnonzero(covered)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(view_index)]))
    result = rows.copy()
    for column in range(rows.shape[1]):
        result[selected, column] = rows[rng.permutation(selected), column]
    return result

def _iter_view_rows(base, seed, draw):
    """Yield view rows and independent column seeds, preserving the guide seed order."""
    (rows, groups, guide_rows, cells, targeting, control, views_ix, variance, kind,
     null_mode, coverage, dtype, packed, block_size, max_missing) = base
    drawn = (permuted_rows(rows, groups, guide_rows, cells, targeting, seed)
             if draw and null_mode == "guides" else rows)
    for index, columns in enumerate(views_ix):
        covered = None if coverage is None else coverage[:, index]
        values = drawn[:, columns]
        view_seed = seed
        if null_mode == "columns":
            if draw:
                values = _column_rows(values, covered, seed, index)
            view_seed = int(np.random.SeedSequence([int(seed), index]).generate_state(1)[0])
        yield index, values, covered, view_seed

def _projection(args, tables, *, draw, keep_scores):
    """Share observed and null view calibration, streaming packed maxima."""
    base = args[:-3]
    seed, pb, nb = args[-3:]
    control, variance, kind = base[5], base[7], base[8]
    dtype, packed, block_size, max_missing = base[11:]
    retained = []

    def scores():
        """Yield each calibrated view and retain only requested observed scores."""
        for index, rows, covered, view_seed in _iter_view_rows(base, seed, draw):
            value = calibrate_view(view_similarity(
                rows, control, variance, view_seed, kind, coverage=covered,
                dtype=dtype, packed=packed, block_size=block_size,
                max_missing=max_missing), tables[index], pb)
            if keep_scores:
                retained.append(value)
            yield value

    coverage = base[10]
    valid = (None if coverage is None else
             (_coverage_pairs(coverage[:, index], packed) for index in range(len(base[6]))))
    projected, view = max_projection(scores(), valid=valid)
    return retained, projected, view


def read_views(path: str, columns: Sequence[str]) -> Tuple[List[Dict[str, Any]], str]:
    """The views of a views file, checked against the profile's columns.

    :param path: a JSON list of ``{"name", "block", "columns"}`` objects.
    :param columns: the measurement columns of the gene table.
    :returns: ``(views, md5)``; every view keeps its columns in the file's order and its
        ``block`` (the view's name when the file gives none).
    :raises ValueError: when the file holds no view, a view has no column, two views share a
        name, or a view names a column the gene table lacks.
    """
    with open(path, "rb") as handle:
        raw = handle.read()
    md5 = hashlib.md5(raw).hexdigest()
    entries = json.loads(raw.decode("utf-8"))
    if isinstance(entries, Mapping):
        entries = entries.get("views", [])
    if not entries:
        raise ValueError(f"the views file {path} holds no view")
    known = set(map(str, columns))
    views, names = [], set()
    for entry in entries:
        name = str(entry["name"])
        cols = [str(c) for c in entry.get("columns", [])]
        if not cols:
            raise ValueError(f"view {name!r} has no column")
        if name in names:
            raise ValueError(f"two views are named {name!r}")
        missing = [c for c in cols if c not in known]
        if missing:
            raise ValueError(f"view {name!r} names columns the gene table lacks: {missing[:5]}")
        names.add(name)
        views.append({"name": name, "block": str(entry.get("block") or name), "columns": cols})
    return views, md5


def cell_bins(n_cells: np.ndarray, bins: int) -> np.ndarray:
    """Each gene's cell-count quantile bin, 0 to ``bins - 1``.

    :param n_cells: cells per gene.
    :param bins: the number of quantile bins.
    :returns: integer bins; ties fall in the lower bin.
    """
    n_cells = np.asarray(n_cells, float)
    if bins <= 1:
        return np.zeros(len(n_cells), int)
    edges = np.quantile(n_cells, np.linspace(0, 1, bins + 1)[1:-1])
    return np.searchsorted(edges, n_cells, side="right").astype(int)


def pair_bins(gene_bins: np.ndarray, bins: int, *, packed: bool = False) -> np.ndarray:
    """Return unordered gene-bin pairs in a square or packed upper-triangle array.

    :param gene_bins: from :func:`cell_bins`.
    :param bins: the number of gene bins.
    :param packed: return row-ordered pairs ``i < j`` without diagonal entries.
    :returns: ``min(b_i, b_j) * bins + max(b_i, b_j)`` for every pair.
    """
    if packed:
        result = np.empty(len(gene_bins) * (len(gene_bins) - 1) // 2,
                          dtype=np.min_scalar_type(bins * bins - 1))
        for row, span in _pair_rows(len(gene_bins)):
            lo = np.minimum(gene_bins[row], gene_bins[row + 1:])
            hi = np.maximum(gene_bins[row], gene_bins[row + 1:])
            result[span] = lo * bins + hi
        return result
    lo = np.minimum(gene_bins[:, None], gene_bins[None, :])
    hi = np.maximum(gene_bins[:, None], gene_bins[None, :])
    return lo * bins + hi


def view_similarity(rows: np.ndarray, control: Optional[np.ndarray], variance: float, seed: int,
                    similarity: str = "cosine", *, coverage=None, dtype="float64",
                    packed: bool = False, block_size: int = 1024,
                    max_missing: float = 0.05) -> np.ndarray:
    """Compute gene-pair similarities within one covered feature view.

    The existing :func:`spacr.ops_reference.embed_gene_profiles` standardises
    and embeds the covered rows. Controls enable the existing gene-level TVN
    when their count exceeds the component count. None skips TVN. The shrunk
    similarity is ``x . y / sqrt((|x|^2 + c) (|y|^2 + c))``; without controls,
    ``c`` is the mean squared norm over all covered rows.

    :param rows: genes by the view columns.
    :param control: a gene-control mask, or None for no control normalisation.
    :param variance: the PCA variance kept.
    :param seed: the PCA seed.
    :param similarity: ``'cosine'`` or ``'shrunk'``.
    :param coverage: a gene mask; uncovered pairs receive negative infinity.
    :param dtype: ``'float64'`` or the opt-in ``'float32'`` computation.
    :param packed: return row-ordered upper-triangle pairs without the diagonal.
    :param block_size: the maximum row count of a packed similarity tile.
    :param max_missing: drop columns at or above this missing fraction.
    :returns: a square or packed similarity array; covered values lie in [-1, 1].
    """
    n = len(rows)
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("max-projection dtype must be float32 or float64")
    if similarity not in MAXPROJ_SIMILARITIES:
        raise ValueError("max-projection similarity must be cosine or shrunk")
    if not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("max-projection block_size must be a positive integer")
    covered = np.ones(n, bool) if coverage is None else np.asarray(coverage, bool)
    if covered.shape != (n,):
        raise ValueError("view coverage must have one boolean per gene")
    if control is not None and np.asarray(control).shape != (n,):
        raise ValueError("view controls must have one boolean per gene")
    shape = (n * (n - 1) // 2,) if packed else (n, n)
    if covered.sum() < 2:
        return np.full(shape, -np.inf, dtype=dtype)
    selected, embedded, scale = _view_arrays(
        rows, control, variance, seed, similarity, covered, dtype, max_missing)
    if not packed:
        similarity_values = np.clip((embedded @ embedded.T) / np.outer(scale, scale), -1.0, 1.0)
        if covered.all():
            return similarity_values
        result = np.full(shape, -np.inf, dtype=dtype)
        result[np.ix_(selected, selected)] = similarity_values
        return result
    result = np.full(shape, -np.inf, dtype=dtype)
    for row_start in range(0, len(selected), block_size):
        row_stop = min(row_start + block_size, len(selected))
        for col_start in range(row_start, len(selected), block_size):
            col_stop = min(col_start + block_size, len(selected))
            block = np.clip(
                (embedded[row_start:row_stop] @ embedded[col_start:col_stop].T)
                / np.outer(scale[row_start:row_stop], scale[col_start:col_stop]), -1.0, 1.0)
            columns = selected[col_start:col_stop]
            for local, gene in enumerate(selected[row_start:row_stop]):
                after = columns > gene
                destination = _packed_indices(n, gene, columns[after])
                result[destination] = block[local, after]
    return result


def permuted_rows(rows: np.ndarray, groups: Sequence[np.ndarray], guide_rows: np.ndarray,
                  guide_cells: np.ndarray, targeting: np.ndarray, seed: int) -> np.ndarray:
    """One guide-level label permutation of a gene profile.

    The guides of the targeting genes are shuffled between those genes (each keeps its guide
    count; the non-targeting rows and their guides stay fixed) and every targeting row is
    rebuilt by the cell-weighted guide mean of its new guides
    (:func:`spacr.ops_clusters._guide_shift`).

    :param rows: the gene profile.
    :param groups: each row's guide indices (:func:`spacr.ops_clusters._guide_groups`).
    :param guide_rows: one row per guide.
    :param guide_cells: each guide's cell count.
    :param targeting: True for the rows whose guides are shuffled.
    :param seed: the generator seed.
    :returns: the permuted rows, a new array.
    """
    from .ops_clusters import _guide_shift

    pool = np.concatenate([groups[i] for i in np.flatnonzero(targeting) if len(groups[i])]
                          or [np.array([], int)])
    shuffled = np.random.default_rng(seed).permutation(pool)
    drawn, at = [], 0
    for i, ix in enumerate(groups):
        if targeting[i] and len(ix):
            drawn.append(shuffled[at:at + len(ix)])
            at += len(ix)
        else:
            drawn.append(ix)
    return _guide_shift(rows, groups, drawn, guide_rows, guide_cells)


def _packed_grid_indices(values, grid):
    """Match clipped searchsorted bins on a declared uniform grid with bounded work arrays.

    :param values: one packed float32 vector.
    :param grid: the declared similarity or normal-score grid.
    :returns: right-sided bin indices, clipped to the last histogram interval.
        Estimates are corrected against actual floating-point grid boundaries;
        endpoints, infinities and NaNs match the existing searchsorted result.
    """
    result = np.empty(len(values), np.int32)
    last = len(grid) - 2
    scale = (len(grid) - 1) / (grid[-1] - grid[0])
    for start in range(0, len(values), 1_000_000):
        stop = min(start + 1_000_000, len(values))
        selected = np.asarray(values[start:stop], dtype=float)
        selected = np.nan_to_num(selected, nan=grid[-1], posinf=grid[-1], neginf=grid[0])
        selected = np.clip(selected, grid[0], grid[-1])
        index = np.clip(np.floor((selected - grid[0]) * scale), 0, last).astype(np.int32)
        while True:
            lower = (index > 0) & (selected < grid[index])
            upper = (index < last) & (selected >= grid[index + 1])
            if not (lower.any() or upper.any()):
                break
            index[lower] -= 1
            index[upper] += 1
        result[start:stop] = index
    return result


def _counts(values: np.ndarray, bins_of_pairs: np.ndarray, n_bins: int, grid: np.ndarray,
            valid=None) -> np.ndarray:
    """Count eligible upper-triangle values on a grid in either storage layout."""
    if values.ndim == 2:
        iu = np.triu_indices(values.shape[0], 1)
        v, b = values[iu], bins_of_pairs[iu]
        mask = None if valid is None else valid[iu]
    else:
        v, b, mask = values, bins_of_pairs, valid
    if mask is not None:
        v, b = v[mask], b[mask]
    index = (_packed_grid_indices(v, grid)
             if values.ndim == 1 and values.dtype == np.float32
             and (grid is SIMILARITY_GRID or grid is SCORE_GRID) else
             np.clip(np.searchsorted(grid, v, side="right") - 1, 0, len(grid) - 2))
    if values.ndim == 2 and valid is None:
        out = np.zeros((n_bins, len(grid) - 1), np.int64)
        np.add.at(out, (b, index), 1)
        return out
    flat = np.asarray(b, np.int64) * (len(grid) - 1) + index
    return np.bincount(flat, minlength=n_bins * (len(grid) - 1)).reshape(n_bins, len(grid) - 1)


def calibrate_view(similarity: np.ndarray, null_counts: np.ndarray,
                   bins_of_pairs: np.ndarray) -> np.ndarray:
    """Convert square or packed similarities to calibrated upper-tail normal scores.

    The empirical tail uses ``(1 + nulls at or above it) / (1 + nulls)``
    within each unordered pair bin. Uncovered negative-infinity similarities
    remain negative infinity, so they cannot win a projection.

    :param similarity: from :func:`view_similarity`.
    :param null_counts: per-bin counts on :data:`SIMILARITY_GRID`.
    :param bins_of_pairs: from :func:`pair_bins`, in the same storage layout.
    :returns: calibrated scores with the input precision and storage layout;
        a square result has zero on the diagonal.
    """
    from scipy.stats import norm

    upper = np.cumsum(null_counts[:, ::-1], axis=1)[:, ::-1]
    totals = null_counts.sum(axis=1)
    if similarity.ndim == 1 and similarity.dtype == np.float32:
        index = _packed_grid_indices(similarity, SIMILARITY_GRID)
        probabilities = (1.0 + upper) / (1.0 + totals[:, None])
        lookup = -norm.ppf(np.clip(probabilities, 1e-300, 1.0))
        score = lookup[bins_of_pairs, index]
    else:
        index = np.clip(np.searchsorted(SIMILARITY_GRID, similarity, side="right") - 1, 0,
                        len(SIMILARITY_GRID) - 2)
        above = upper[bins_of_pairs, index]
        p = (1.0 + above) / (1.0 + totals[bins_of_pairs])
        score = -norm.ppf(np.clip(p, 1e-300, 1.0))
    if score.ndim == 2:
        np.fill_diagonal(score, 0.0)
    return score.astype(similarity.dtype, copy=False)


def max_projection(scores: Sequence[np.ndarray], *, valid=None) -> Tuple[np.ndarray, np.ndarray]:
    """Return the strongest calibrated pair score and its first eligible view.

    :param scores: score arrays in view order, square or packed.
    :param valid: optional pair masks in the same order and storage layout.
    :returns: ``(M, view)``; ties select the lower eligible view index.
        Pairs without any eligible view have negative-infinity M and view -1.
        Packed inputs are consumed one view at a time without stacking them.
    """
    iterator = iter(scores)
    first = next(iterator)
    if first.ndim == 2 and valid is None:
        stack = np.stack([first, *iterator])
        view = np.argmax(stack, axis=0)
        return np.take_along_axis(stack, view[None], 0)[0], view
    from itertools import chain

    projected = np.full_like(first, -np.inf)
    view = np.full(first.shape, -1, np.int32)
    masks = None if valid is None else iter(valid)
    for index, score in enumerate(chain((first,), iterator)):
        allowed = True if masks is None else next(masks)
        better = allowed & ((view < 0) | (score > projected)
                            | (np.isnan(score) & ~np.isnan(projected)))
        projected[better] = score[better]
        view[better] = index
    return projected, view


def max_null_pvalues(projected: np.ndarray, null_counts: np.ndarray,
                     bins_of_pairs: np.ndarray) -> np.ndarray:
    """Return max-null pair p values in square or packed storage.

    :param projected: M from :func:`max_projection`.
    :param null_counts: per-bin max-null counts on :data:`SCORE_GRID`.
    :param bins_of_pairs: pair bins in the same layout.
    :returns: upper-tail p values with the input precision; diagonal square
        entries and pairs without a covered view have p equal to one.
    """
    upper = np.cumsum(null_counts[:, ::-1], axis=1)[:, ::-1]
    totals = null_counts.sum(axis=1)
    index = np.clip(np.searchsorted(SCORE_GRID, projected, side="right") - 1, 0,
                    len(SCORE_GRID) - 2)
    p = (1.0 + upper[bins_of_pairs, index]) / (1.0 + totals[bins_of_pairs])
    if p.ndim == 2:
        np.fill_diagonal(p, 1.0)
    return np.clip(p, 0.0, 1.0).astype(projected.dtype, copy=False)


def projected_graph(pvalues: np.ndarray, knn: int, edge_p: float, *,
                    n_genes: Optional[int] = None, valid=None):
    """Build the gene graph of the calibrated projected pairs.

    Each gene keeps its ``knn`` strongest partners by ``-log10 p`` among
    pairs below ``edge_p`` and its single best partner, through the shared
    :func:`spacr.ops_clusters._knn_graph` owner. Packed input streams one row
    at a time. An explicit validity mask excludes unsupported pairs, leaving
    genes with no supported partner isolated. The original square call keeps
    its existing graph and tie order.

    :param pvalues: square or packed values from :func:`max_null_pvalues`.
    :param knn: partners per gene.
    :param edge_p: the p value an edge must stay below.
    :param n_genes: required gene count for packed input.
    :param valid: optional boolean pair mask in the same layout; use
        ``result['view'] >= 0`` to retain only pairs sharing a covered view.
    :returns: an undirected igraph graph weighted by ``-log10 p``.
    """
    from .ops_clusters import _knn_graph

    if np.ndim(pvalues) == 1 or valid is not None:
        pvalues = np.asarray(pvalues)
        if pvalues.ndim == 1:
            if not isinstance(n_genes, (int, np.integer)) or n_genes < 2:
                raise ValueError("packed graph probabilities require n_genes of at least two")
            n = int(n_genes)
            if pvalues.shape != (n * (n - 1) // 2,):
                raise ValueError("packed graph probabilities must contain every unordered gene pair")
        else:
            n = len(pvalues)
            if pvalues.shape != (n, n):
                raise ValueError("graph probabilities must be a square matrix or a packed vector")
        if valid is not None:
            valid = np.asarray(valid, bool)
            if valid.shape != pvalues.shape:
                raise ValueError("graph validity must have the same layout as its probabilities")
        return _knn_graph(_projected_strength_rows(pvalues, n, edge_p, valid), knn, n_rows=n)
    strength = -np.log10(np.clip(pvalues, 1e-300, 1.0))
    keep = pvalues < edge_p
    best = np.argmax(np.where(np.eye(len(strength), dtype=bool), -np.inf, strength), axis=1)
    keep[np.arange(len(best)), best] = True
    keep = keep | keep.T
    masked = np.where(keep, strength, -np.inf)
    return _knn_graph(masked, knn)



def label_clusters_by_view(labels: np.ndarray, graph, view: np.ndarray,
                           views: Sequence[Mapping[str, Any]], scores: Sequence[np.ndarray],
                           null_counts: np.ndarray, n_columns: int, edge_p: float = 0.05
                           ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise within-cluster edges by argmax view and broad, thin or mixed label.

    A view is broad when its block is ``W`` (the whole profile) or a ``P1`` block holding at
    least a quarter of the ``n_columns`` columns, and thin when it has at most
    :data:`THIN_COLUMNS` columns. A cluster is ``broad`` when its broad views carry at least
    half of its within-cluster edges, else ``thin`` when its thin views do, else ``mixed``.

    :param labels: one cluster per gene.
    :param graph: from :func:`projected_graph`.
    :param view: square or packed argmax views from :func:`max_projection`;
        edges with view -1 have no measured support and are excluded.
    :param views: the views, as from :func:`read_views`.
    :param scores: calibrated score matrices or packed vectors from :func:`calibrate_view`.
    :param null_counts: the max null's counts (all pair bins pooled for the threshold).
    :param n_columns: the number of profile measurement columns.
    :param edge_p: the p value the max-null threshold is read at.
    :returns: ``(shares, kinds)``: one row per cluster and view with its ``block``, ``share``
        of the edges, ``mean_score`` (the mean calibrated score of the within-cluster edges in
        that view) and ``max_null_score`` (the max-null score at ``edge_p``); and one row per
        cluster with its ``edges``, ``label`` and ``views`` (every view carrying at least
        :data:`LISTED_SHARE`, ``;``-joined).
    """
    sizes = np.array([len(v["columns"]) for v in views])
    blocks = [str(v["block"]) for v in views]
    broad = np.array([b == "W" or (b.startswith("P1") and size * 4 >= n_columns)
                      for b, size in zip(blocks, sizes)])
    thin = sizes <= THIN_COLUMNS
    pooled = np.asarray(null_counts).sum(axis=0)
    upper = np.cumsum(pooled[::-1])[::-1] / max(pooled.sum(), 1)
    threshold = float(SCORE_GRID[min(int(np.argmax(upper <= edge_p)), len(SCORE_GRID) - 1)])
    rows, kinds = [], []
    edges = np.array(graph.get_edgelist(), int).reshape(-1, 2)
    edge_views = _pair_values(view, len(labels), edges[:, 0], edges[:, 1])
    for cluster in np.unique(labels):
        inside = ((labels[edges[:, 0]] == cluster) & (labels[edges[:, 1]] == cluster)
                  & (edge_views >= 0))
        a, b = edges[inside, 0], edges[inside, 1]
        n = int(inside.sum())
        shares = np.bincount(edge_views[inside], minlength=len(views)) / max(n, 1)
        for v, share, score in zip(views, shares, scores):
            rows.append({"cluster": int(cluster), "view": v["name"], "block": v["block"],
                         "share": float(share),
                         "mean_score": float(_pair_values(score, len(labels), a, b).mean()) if n else float("nan"),
                         "max_null_score": threshold})
        if n and shares[broad].sum() >= 0.5:
            kind = "broad"
        elif n and shares[thin].sum() >= 0.5:
            kind = "thin"
        else:
            kind = "mixed"
        listed = [v["name"] for v, share in zip(views, shares) if n and share >= LISTED_SHARE]
        kinds.append({"cluster": int(cluster), "edges": n, "label": kind,
                      "views": ";".join(listed)})
    return pd.DataFrame(rows), pd.DataFrame(kinds)


def _get(settings: Mapping[str, Any], key: str, default: Any) -> Any:
    """One setting, its default when unset."""
    return default if settings.get(key) is None else settings[key]


def _options(settings: Mapping[str, Any]) -> SimpleNamespace:
    """The ``ops_maxproj_*`` settings with their defaults, checked."""
    get = partial(_get, settings)
    options = SimpleNamespace(
        rows=settings.get("ops_maxproj_rows") or None,
        guides=settings.get("ops_maxproj_guides") or None,
        views=settings.get("ops_maxproj_views") or None,
        variance=float(get("ops_maxproj_pca_variance", 0.99)),
        bins=int(get("ops_maxproj_bins", 3)),
        calibration=int(get("ops_maxproj_calibration_permutations", 50)),
        null=int(get("ops_maxproj_null_permutations", 200)),
        seed=int(get("ops_maxproj_seed", DEFAULT_SEED)),
        graph=str(get("ops_maxproj_graph", "knn")).lower(),
        similarity=str(get("ops_maxproj_similarity", "cosine")).lower(),
        knn=int(get("ops_maxproj_knn", 10)),
        edge_p=float(get("ops_maxproj_edge_p", 0.05)),
        resolutions=[float(r) for r in (settings.get("ops_maxproj_resolutions")
                                        or DEFAULT_RESOLUTIONS)],
        leiden_seeds=[int(s) for s in get("ops_maxproj_leiden_seeds", [42, 1, 2])],
        k=None if settings.get("ops_maxproj_k") in (None, "") else int(settings["ops_maxproj_k"]),
        consensus=bool(get("ops_maxproj_consensus", False)),
        save_views=bool(get("ops_maxproj_save_views", True)),
        workers=max(1, int(get("ops_maxproj_workers", 1))),
        n_boot=int(get("ops_consensus_bootstraps", 10)),
        prefix=str(settings.get("ops_control_prefix") or "nontargeting"))
    if not options.views:
        raise ValueError("ops_maxproj_views is required when ops_maxproj_enabled is True")
    if options.graph not in MAXPROJ_GRAPHS:
        raise ValueError(f"ops_maxproj_graph must be one of {list(MAXPROJ_GRAPHS)}; "
                         f"got {options.graph!r}")
    if options.similarity not in MAXPROJ_SIMILARITIES:
        raise ValueError(f"ops_maxproj_similarity must be one of {list(MAXPROJ_SIMILARITIES)}; "
                         f"got {options.similarity!r}")
    if not 0 < options.variance <= 1:
        raise ValueError("ops_maxproj_pca_variance must be in (0, 1]")
    if options.bins < 1 or options.calibration < 1 or options.null < 1 or options.knn < 1:
        raise ValueError("ops_maxproj_bins, the permutation counts and ops_maxproj_knn must be "
                         "at least 1")
    if not 0 < options.edge_p <= 1:
        raise ValueError("ops_maxproj_edge_p must be in (0, 1]")
    return options


def _tables(options: SimpleNamespace, genes: Optional[pd.DataFrame],
            guides: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """The gene and guide tables in named measurement columns."""
    from .tabular import read_table

    gene_table = (read_table(str(options.rows), canonicalise=False, report=None)
                  if options.rows else genes)
    guide_table = (read_table(str(options.guides), canonicalise=False, report=None)
                   if options.guides else guides)
    if gene_table is None or guide_table is None:
        raise ValueError("cluster_maxproj needs gene and guide rows: run the profile phase or set "
                         "ops_maxproj_rows and ops_maxproj_guides")
    for name, table in (("gene", gene_table), ("guide", guide_table)):
        missing = [c for c in ("gene", "n_cells") if c not in table.columns]
        if missing:
            raise ValueError(f"the {name} table lacks {missing}")
    return gene_table.reset_index(drop=True), guide_table.reset_index(drop=True)


def _view_job(args):
    """Count one calibration draw over all views with its declared null and coverage."""
    base = args[:-3]
    seed, pb, nb = args[-3:]
    control, variance, kind = base[5], base[7], base[8]
    dtype, packed, block_size, max_missing = base[11:]
    result = []
    for index, rows, covered, view_seed in _iter_view_rows(base, seed, True):
        similarities = view_similarity(rows, control, variance, view_seed, kind,
                                       coverage=covered, dtype=dtype, packed=packed,
                                       block_size=block_size, max_missing=max_missing)
        valid = None if covered is None else _coverage_pairs(covered, packed)
        result.append(_counts(similarities, pb, nb, SIMILARITY_GRID, valid))
    return result


def _max_job(args):
    """Count one joint max-null draw, including eligible negative-infinity scores."""
    work, tables = args[:-1], args[-1]
    projected = _projection(work, tables, draw=True, keep_scores=False)[1]
    valid = _joint_coverage(work[10], work[12])
    return _counts(projected, work[-2], work[-1], SCORE_GRID, valid)


def _map(job, tasks: List[tuple], workers: int) -> List[Any]:
    """``job`` over the tasks in task order, in ``workers`` processes."""
    if workers <= 1:
        return [job(t) for t in tasks]
    from .resource_log import _parallel_process_executor

    with _parallel_process_executor(workers) as pool:
        return list(pool.map(job, tasks))


def maxproj_scores(rows: np.ndarray, names: Sequence[str], n_cells=None,
                   guide_rows=None, guide_genes=None, guide_cells=None,
                   views_ix=None, *, variance: float = 0.99,
                   similarity: str = "cosine", bins: int = 3,
                   calibration: int = 50, null: int = 200, seed: int = DEFAULT_SEED,
                   prefix: str = "nontargeting", workers: int = 1, report=None,
                   null_mode: str = "guides", control="prefix", coverage=None,
                   dtype="float64", packed: bool = False, block_size: int = 1024,
                   max_missing=None) -> Dict[str, Any]:
    """Calibrate feature views and compute their max projection and max-null p values.

    Defaults retain the signed OPS cosine, guide-permutation null and square
    float64 results. The opt-in column null independently shuffles every column
    among covered genes of each view, with independent seeds for each view and
    draw. A row-label permutation is not used as a gene-level null.

    :param rows: genes by named measurement columns.
    :param names: gene names, in row order.
    :param n_cells: cell counts for pair bins, or None to use view-coverage counts.
    :param guide_rows: guide profiles for the guide null; None for the column null.
    :param guide_genes: gene names of guide rows, or None for the column null.
    :param guide_cells: guide cell counts, or None for the column null.
    :param views_ix: measurement-column indices of every view, in fixed order.
    :param variance: the per-view PCA variance kept.
    :param similarity: ``'cosine'`` or ``'shrunk'``.
    :param bins: quantile bins per gene, giving bins squared unordered-pair bins.
    :param calibration: per-view null draws, seeded ``seed + p``.
    :param null: fresh joint max-null draws, seeded ``seed + 100000 + p``.
    :param seed: the base seed; column seeds derive from that draw seed and view index.
    :param prefix: the default non-targeting prefix.
    :param workers: worker processes; results retain the fixed draw order.
    :param report: optional progress-line callback.
    :param null_mode: ``'guides'`` (the OPS default) or opt-in ``'columns'``.
    :param control: ``'prefix'`` derives the OPS control mask; a boolean gene mask
        supplies controls; None skips TVN and fits shrunk c over all covered rows.
        Column mode with no matching prefix also uses all covered rows for c.
    :param coverage: genes by views boolean mask; uncovered pairs cannot win a view.
    :param dtype: ``'float64'`` or opt-in ``'float32'`` similarity and result storage.
    :param packed: return row-ordered pairs ``i < j`` in vectors, avoiding dense
        view stacks; returned metadata includes ``packed`` and ``n_genes``.
    :param block_size: row count of packed similarity tiles, at most this squared.
    :param max_missing: embedding column-missingness cutoff; None keeps the OPS
        0.05 cutoff in guide mode and all nonempty columns in column mode.
    :returns: ``scores``, ``projected``, ``view``, ``p``, ``q``, ``pair_bins``,
        ``calibration_counts``, ``null_counts`` and ``control``. Pairs sharing no
        covered view have score negative infinity, view -1 and p/q equal to one.
        Coverage restricts the BH family to pairs sharing a measured view.
    """
    from .multiple_testing import adjust_p_values
    from .ops_clusters import _guide_groups, _is_control_row

    if null_mode not in ("guides", "columns"):
        raise ValueError("max-projection null_mode must be guides or columns")
    if calibration < 1 or null < 1 or bins < 1:
        raise ValueError("max-projection calibration, null and bins must be positive")
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("max-projection dtype must be float32 or float64")
    say = report or (lambda text: None)
    rows = np.asarray(rows, float)
    names = np.asarray(names).astype(str)
    if rows.ndim != 2 or len(names) != len(rows) or len(rows) < 2:
        raise ValueError("max-projection needs at least two gene rows and matching names")
    if views_ix is None or len(views_ix) == 0:
        raise ValueError("max-projection needs at least one view")
    views_ix = [np.asarray(columns, int) for columns in views_ix]
    if any(columns.ndim != 1 or len(columns) == 0 or (columns < 0).any()
           or (columns >= rows.shape[1]).any() for columns in views_ix):
        raise ValueError("max-projection views must contain valid measurement-column indices")
    if coverage is not None:
        coverage = np.asarray(coverage, bool)
        if coverage.shape != (len(rows), len(views_ix)):
            raise ValueError("max-projection coverage must have shape genes by views")
    if isinstance(control, str):
        if control != "prefix":
            raise ValueError("max-projection control must be prefix, None or a boolean mask")
        control = _is_control_row(names, prefix)
        if null_mode == "columns" and not control.any():
            control = None
    elif control is not None:
        control = np.asarray(control, bool)
        if control.shape != (len(rows),):
            raise ValueError("max-projection controls must have one boolean per gene")
    control_rows = np.zeros(len(rows), bool) if control is None else control
    targeting = ~control_rows
    if null_mode == "guides":
        if guide_rows is None or guide_genes is None or guide_cells is None:
            raise ValueError("the guide null needs guide rows, gene names and cell counts")
        groups = _guide_groups(names, guide_genes, control_rows)
        guide_rows = np.asarray(guide_rows, float)
        cells = np.asarray(guide_cells, float)
    else:
        groups, cells = None, None
    if n_cells is None:
        n_cells = (np.full(len(rows), len(views_ix)) if coverage is None
                   else coverage.sum(axis=1))
    if np.asarray(n_cells).shape != (len(rows),):
        raise ValueError("max-projection cell counts must have one value per gene")
    max_missing = (0.05 if null_mode == "guides" else 1.0) if max_missing is None else max_missing
    gb = cell_bins(np.asarray(n_cells, float), bins)
    pb = pair_bins(gb, bins, packed=packed)
    nb = bins * bins
    base = (rows, groups, guide_rows, cells, targeting, control, views_ix, variance,
            similarity, null_mode, coverage, dtype, packed, block_size, max_missing)
    jobs = [(*base, seed + p, pb, nb) for p in range(calibration)]
    counts = _map(_view_job, jobs, workers)
    tables = [sum(c[v] for c in counts) for v in range(len(views_ix))]
    say(f"cluster_maxproj: calibrated {len(views_ix)} views on {calibration} permutations")
    scores, projected, view = _projection((*base, seed, pb, nb), tables,
                                         draw=False, keep_scores=True)
    jobs = [(*base, seed + 100000 + p, pb, nb, tables) for p in range(null)]
    null_counts = sum(_map(_max_job, jobs, workers))
    p = max_null_pvalues(projected, null_counts, pb)
    valid = _joint_coverage(coverage, packed)
    q = np.ones_like(p)
    if packed:
        selected = np.ones(len(p), bool) if valid is None else valid
        q[selected] = adjust_p_values(p[selected], "fdr_bh")[0]
    else:
        iu = np.triu_indices(len(names), 1)
        selected = np.ones(len(iu[0]), bool) if valid is None else valid[iu]
        adjusted = np.ones(len(iu[0]), dtype=p.dtype)
        adjusted[selected] = adjust_p_values(p[iu][selected], "fdr_bh")[0]
        q[iu] = adjusted
        q[(iu[1], iu[0])] = adjusted
    say(f"cluster_maxproj: max null from {null} joint permutations")
    result = {"scores": scores, "projected": projected, "view": view, "p": p, "q": q,
              "pair_bins": pb, "calibration_counts": tables, "null_counts": null_counts,
              "control": control_rows}
    if packed:
        result.update(packed=True, n_genes=len(names))
    if null_mode != "guides":
        result["null_mode"] = null_mode
    return result


def projected_rows_graph(rows: np.ndarray, control: np.ndarray, views_ix: Sequence[np.ndarray],
                         tables: Sequence[np.ndarray], null_counts: np.ndarray,
                         bins_of_pairs: np.ndarray, *, variance: float, similarity: str,
                         seed: int, knn: int, edge_p: float):
    """The projected graph of a new set of rows under fixed calibration and null tables.

    The consensus hook of :func:`spacr.ops_clusters.guide_bootstrap_consensus`: each guide
    bootstrap is scored against the tables of the base run, so only the rows change.

    :param rows: genes by columns.
    :param control: True for the non-targeting rows.
    :param views_ix: each view's column indices.
    :param tables: each view's calibration counts.
    :param null_counts: the max null's counts.
    :param bins_of_pairs: from :func:`pair_bins`.
    :param variance: the per-view PCA variance kept.
    :param similarity: see :func:`view_similarity`.
    :param seed: the PCA seed of the base run.
    :param knn: partners per gene.
    :param edge_p: the edge p value.
    :returns: the graph of :func:`projected_graph`.
    """
    scores = [calibrate_view(view_similarity(rows[:, ix], control, variance, seed, similarity), t,
                             bins_of_pairs) for ix, t in zip(views_ix, tables)]
    projected, _ = max_projection(scores)
    return projected_graph(max_null_pvalues(projected, null_counts, bins_of_pairs), knn, edge_p)


def _graph_of(base: Mapping[str, Any], rows: np.ndarray, run_seed: int):
    """The consensus graph builder: :func:`projected_rows_graph` on the base run's tables."""
    return projected_rows_graph(rows, **base)


def _figure(shares: pd.DataFrame, path: str) -> str:
    """Stacked bars: each cluster's within-cluster edges by view block."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from .figures.bundle import _register_figure_data
    from .figures.style import figure_style, resolve_label_ground
    from .plot import _checked_savefig

    table = shares.groupby(["cluster", "block"], as_index=False)["share"].sum()
    wide = table.pivot(index="cluster", columns="block", values="share").fillna(0.0)
    palette = ["#0A77B8", "#0A9878", "#283868", "#78AED2", "#C878A8", "#F0A830", "#8C8C8C"]
    with figure_style("print"):
        figure = Figure(figsize=(max(3.0, 0.18 * len(wide) + 1.5), 2.8))
        ax = figure.subplots()
        bottom = np.zeros(len(wide))
        for i, block in enumerate(wide.columns):
            ax.bar(range(len(wide)), wide[block].to_numpy(), bottom=bottom, width=0.6,
                   color=palette[i % len(palette)], alpha=0.6, label=str(block))
            bottom += wide[block].to_numpy()
        ax.set_xticks(range(len(wide)))
        ax.set_xticklabels([str(c) for c in wide.index], fontsize=8, rotation=90)
        ax.set_xlabel("cluster", fontsize=8)
        ax.set_ylabel("within-cluster edges (share)", fontsize=8)
        ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        _register_figure_data(figure, lambda: table, x="cluster", y="share", hue="block",
                              kind="bar")
        figure.tight_layout()
        _checked_savefig(figure, path, dpi=150, bbox_inches="tight",
                         facecolor=resolve_label_ground("print"), metadata={"Software": None})
    return path


def cluster_maxproj_phase(settings: Mapping[str, Any], genes: Optional[pd.DataFrame],
                          guides: Optional[pd.DataFrame], folder: str, report=None
                          ) -> Dict[str, Any]:
    """The ``cluster_maxproj`` phase of the OPS profile path (item 992, alpha).

    Reads the views file (``ops_maxproj_views``), scores every gene pair in every view
    (``ops_maxproj_similarity``, :func:`view_similarity`) against
    that view's guide-permutation null (:func:`maxproj_scores`), keeps each pair's strongest
    score over the views, tests it against the joint max-statistic null, builds the projected
    graph (:func:`projected_graph`, or PHATE on the projected scores with
    ``ops_maxproj_graph`` ``'phate'``), runs Leiden over ``ops_maxproj_resolutions`` and
    ``ops_maxproj_leiden_seeds`` plus the partition matched to the target count
    (``ops_maxproj_k``, else :func:`spacr.ops_clusters.target_cluster_count`), and labels every
    cluster by the views its edges came from (:func:`label_clusters_by_view`).

    :param settings: read for every ``ops_maxproj_*`` setting, ``ops_control_prefix`` and
        ``ops_consensus_bootstraps`` (with ``ops_maxproj_consensus``).
    :param genes: the profile phase's gene table in named measurement columns, or None.
    :param guides: the profile phase's guide table, or None.
    :param folder: ``dst_root``; outputs go to its ``results/ops_maxproj``.
    :param report: called with progress lines.
    :returns: ``{"folder", "outputs", "k", "clusters"}``.
    :raises ValueError: when the views file is missing or names unknown columns, the tables lack
        ``gene`` or ``n_cells``, or a setting is out of range.
    :raises ImportError: when the ``clustering`` extra is not installed.
    """
    import time

    from .ops_clusters import (fused_phate_graph, leiden_partitions, match_resolution,
                               target_cluster_count)
    from .tabular import write_table
    from .version import __version__ as spacr_version

    say = report or (lambda text: None)
    started = time.perf_counter()
    options = _options(settings)
    gene_table, guide_table = _tables(options, genes, guides)
    columns = [c for c in gene_table.columns if c not in ("gene", "n_cells")
               and pd.api.types.is_numeric_dtype(gene_table[c])]
    views, views_md5 = read_views(str(options.views), columns)
    index = {c: i for i, c in enumerate(columns)}
    views_ix = [np.array([index[c] for c in v["columns"]], int) for v in views]
    missing = [c for c in columns if c not in guide_table.columns]
    if missing:
        raise ValueError(f"the guide table lacks the gene table's columns {missing[:5]}")
    names = gene_table["gene"].astype(str).to_numpy()
    result = maxproj_scores(gene_table[columns].to_numpy(float), names,
                            gene_table["n_cells"].to_numpy(float),
                            guide_table[columns].to_numpy(float),
                            guide_table["gene"].astype(str).to_numpy(),
                            guide_table["n_cells"].to_numpy(float), views_ix,
                            variance=options.variance, similarity=options.similarity,
                            bins=options.bins,
                            calibration=options.calibration, null=options.null,
                            seed=options.seed, prefix=options.prefix, workers=options.workers,
                            report=say)
    targeting = ~result["control"]
    if options.graph == "phate":
        graph = fused_phate_graph(result["projected"], options.seed, knn=options.knn)
    else:
        graph = projected_graph(result["p"], options.knn, options.edge_p)
    grid = leiden_partitions(graph, options.resolutions, options.leiden_seeds)
    k = options.k or target_cluster_count(int(targeting.sum()))
    labels, resolution = match_resolution(graph, int(k), int(options.leiden_seeds[0]), targeting)
    if options.consensus:
        from .ops_clusters import guide_bootstrap_consensus, guide_bootstrap_profiles

        rows = gene_table[columns].to_numpy(float)
        draws = [guide_bootstrap_profiles(rows, names, guide_table[columns].to_numpy(float),
                                          guide_table["gene"].astype(str).to_numpy(),
                                          guide_table["n_cells"].to_numpy(float),
                                          options.seed + 200000 + b,
                                          control_prefix=options.prefix)
                 for b in range(options.n_boot)]
        base = {"control": result["control"], "views_ix": views_ix,
                "tables": result["calibration_counts"], "null_counts": result["null_counts"],
                "bins_of_pairs": result["pair_bins"], "variance": options.variance,
                "similarity": options.similarity,
                "seed": options.seed, "knn": options.knn, "edge_p": options.edge_p}
        consensus = guide_bootstrap_consensus(draws, options.n_boot, int(k), targeting,
                                              seed=100, rows=rows, report=say,
                                              graph_of=partial(_graph_of, base))
        labels = np.asarray(consensus["labels"])
    shares, kinds = label_clusters_by_view(labels, graph, result["view"], views, result["scores"],
                                           result["null_counts"], len(columns), options.edge_p)
    out_dir = os.path.join(folder, "results", "ops_maxproj")
    os.makedirs(out_dir, exist_ok=True)
    outputs: Dict[str, str] = {}
    outputs["views"] = write_table(pd.DataFrame(
        [{"view": v["name"], "block": v["block"], "columns": len(v["columns"]),
          "md5": hashlib.md5(json.dumps(v["columns"]).encode()).hexdigest()} for v in views]),
        os.path.join(out_dir, "views.csv"), canonicalise=False)
    nb = options.bins * options.bins
    outputs["calibration_bins"] = write_table(pd.DataFrame(
        [{"view": v["name"], "pair_bin": b, "null_pairs": int(t[b].sum())}
         for v, t in zip(views, result["calibration_counts"]) for b in range(nb)]),
        os.path.join(out_dir, "calibration_bins.parquet"), canonicalise=False)
    iu = np.triu_indices(len(names), 1)
    outputs["pairs_projected"] = write_table(pd.DataFrame(
        {"gene_a": names[iu[0]], "gene_b": names[iu[1]], "M": result["projected"][iu],
         "view": [views[v]["name"] for v in result["view"][iu]], "p": result["p"][iu],
         "q": result["q"][iu]}), os.path.join(out_dir, "pairs_projected.parquet"),
        canonicalise=False)
    if options.save_views:
        path = os.path.join(out_dir, "per_view_scores.npz")
        np.savez_compressed(path, genes=names, views=np.array([v["name"] for v in views]),
                            scores=np.stack([s[iu] for s in result["scores"]]).astype(np.float32))
        outputs["per_view_scores"] = path
    outputs["leiden_grid"] = write_table(pd.DataFrame(
        [{"resolution": r, "seed": s, "clusters": int(len(np.unique(lab[targeting])))}
         for (s, r), lab in grid.items()]), os.path.join(out_dir, "leiden_grid.csv"),
        canonicalise=False)
    kind = dict(zip(kinds["cluster"], kinds["label"]))
    listed = dict(zip(kinds["cluster"], kinds["views"]))
    outputs["maxproj_partition"] = write_table(pd.DataFrame(
        {"gene": names, "cluster": labels, "label": [kind[c] for c in labels],
         "views": [listed[c] for c in labels], "targeting": targeting}), os.path.join(out_dir, "maxproj_partition.csv"),
        canonicalise=False)
    outputs["cluster_views"] = write_table(shares, os.path.join(out_dir, "cluster_views.csv"),
                                           canonicalise=False)
    outputs["maxproj_figure"] = _figure(shares, os.path.join(out_dir, "maxproj.png"))
    share = np.bincount(result["view"][iu], minlength=len(views)) / max(len(iu[0]), 1)
    v_eff = float(np.exp(-(share[share > 0] * np.log(share[share > 0])).sum()))
    record = {"settings": {key: settings.get(key) for key in sorted(settings)
                           if str(key).startswith("ops_maxproj_")},
              "views_md5": views_md5, "spacr_version": spacr_version, "V": len(views),
              "V_eff": v_eff, "k": int(k), "resolution": float(resolution),
              "clusters": int(len(np.unique(labels[targeting]))), "genes": int(targeting.sum()),
              "rows": int(len(names)), "seconds": round(time.perf_counter() - started, 3)}
    path = os.path.join(out_dir, "maxproj_summary.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1, default=str, sort_keys=True)
    outputs["maxproj_summary"] = path
    say(f"cluster_maxproj: {record['clusters']} clusters (target {k}) over {len(views)} views")
    return {"folder": out_dir, "outputs": outputs, "k": int(k), "clusters": record["clusters"]}
