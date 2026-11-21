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
from functools import partial
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SIMILARITY_GRID = np.linspace(-1.0, 1.0, 4001)
SCORE_GRID = np.linspace(-12.0, 12.0, 4801)
MAXPROJ_GRAPHS = ("knn", "phate")
MAXPROJ_SIMILARITIES = ("shrunk", "cosine")
DEFAULT_SEED = 20261119
DEFAULT_RESOLUTIONS = tuple(float(r) for r in np.logspace(np.log10(0.05), np.log10(50.0), 20))
THIN_SHARE = 0.6


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


def pair_bins(gene_bins: np.ndarray, bins: int) -> np.ndarray:
    """The pair bin of every gene pair: the two genes' bins, unordered.

    :param gene_bins: from :func:`cell_bins`.
    :param bins: the number of gene bins.
    :returns: a symmetric integer matrix, ``min(b_i, b_j) * bins + max(b_i, b_j)``.
    """
    lo = np.minimum(gene_bins[:, None], gene_bins[None, :])
    hi = np.maximum(gene_bins[:, None], gene_bins[None, :])
    return lo * bins + hi


def view_similarity(rows: np.ndarray, control: np.ndarray, variance: float, seed: int,
                    similarity: str = "shrunk") -> np.ndarray:
    """The similarity of every gene pair within one view.

    The view's columns are embedded as the evaluate phase embeds a profile
    (:func:`spacr.ops_reference.embed_gene_profiles`: standardise, PCA keeping ``variance``),
    then whitened on the non-targeting genes in the PCA space
    (:func:`spacr.ops_tvn.tvn_on_controls`, one batch). ``'cosine'`` is the cosine of the
    whitened rows. ``'shrunk'`` is ``x . y / sqrt((|x|^2 + c) (|y|^2 + c))`` with ``c`` the mean
    squared norm of the non-targeting rows (the component count once whitened): a gene no
    stronger than the controls stays near zero, which plain cosine cannot do in a view of few
    columns, where unrelated genes often point the same way.

    :param rows: genes by the view's columns.
    :param control: True for the non-targeting rows.
    :param variance: the PCA variance kept.
    :param seed: the PCA seed.
    :param similarity: ``'shrunk'`` or ``'cosine'``.
    :returns: a symmetric genes-by-genes matrix in [-1, 1].
    """
    from .ops_reference import embed_gene_profiles
    from .ops_tvn import tvn_on_controls

    embedded = embed_gene_profiles(np.asarray(rows, float), variance, seed)
    if np.asarray(control, bool).sum() > embedded.shape[1]:
        embedded, _ = tvn_on_controls(embedded, np.asarray(control, bool))
    norm = (embedded ** 2).sum(axis=1)
    if similarity == "shrunk":
        norm = norm + float(norm[np.asarray(control, bool)].mean())
    scale = np.sqrt(np.clip(norm, 1e-24, None))
    return np.clip((embedded @ embedded.T) / np.outer(scale, scale), -1.0, 1.0)


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


def _counts(values: np.ndarray, bins_of_pairs: np.ndarray, n_bins: int, grid: np.ndarray
            ) -> np.ndarray:
    """Counts of the upper-triangle values on a grid, one row per pair bin."""
    iu = np.triu_indices(values.shape[0], 1)
    v, b = values[iu], bins_of_pairs[iu]
    index = np.clip(np.searchsorted(grid, v, side="right") - 1, 0, len(grid) - 2)
    out = np.zeros((n_bins, len(grid) - 1), np.int64)
    np.add.at(out, (b, index), 1)
    return out


def calibrate_view(similarity: np.ndarray, null_counts: np.ndarray,
                   bins_of_pairs: np.ndarray) -> np.ndarray:
    """A view's pair similarities as normal scores against its own permutation null.

    Each similarity's upper-tail probability in the pooled null of its pair bin, with the
    usual +1 correction, ``(1 + nulls at or above it) / (1 + nulls)``, is turned into the normal
    score ``-Phi^-1(p)``; a high similarity gets a high score.

    :param similarity: from :func:`view_similarity`.
    :param null_counts: the view's null counts per pair bin on :data:`SIMILARITY_GRID`.
    :param bins_of_pairs: from :func:`pair_bins`.
    :returns: a symmetric score matrix, zero on the diagonal.
    """
    from scipy.stats import norm

    upper = np.cumsum(null_counts[:, ::-1], axis=1)[:, ::-1]
    totals = null_counts.sum(axis=1)
    index = np.clip(np.searchsorted(SIMILARITY_GRID, similarity, side="right") - 1, 0,
                    len(SIMILARITY_GRID) - 2)
    above = upper[bins_of_pairs, index]
    p = (1.0 + above) / (1.0 + totals[bins_of_pairs])
    score = -norm.ppf(np.clip(p, 1e-300, 1.0))
    np.fill_diagonal(score, 0.0)
    return score


def max_projection(scores: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """The strongest calibrated score of every pair over the views, and the view it came from.

    :param scores: one score matrix per view, in the views file's order.
    :returns: ``(M, view)``; ties go to the lower view index.
    """
    stack = np.stack(scores)
    view = np.argmax(stack, axis=0)
    return np.take_along_axis(stack, view[None], 0)[0], view


def max_null_pvalues(projected: np.ndarray, null_counts: np.ndarray,
                     bins_of_pairs: np.ndarray) -> np.ndarray:
    """Pair p values of the projected scores against the pooled max-statistic null of their bin.

    :param projected: ``M`` from :func:`max_projection`.
    :param null_counts: counts of the null ``M`` per pair bin on :data:`SCORE_GRID`.
    :param bins_of_pairs: from :func:`pair_bins`.
    :returns: ``(1 + null maxima at or above M) / (1 + null maxima)``; one on the diagonal.
    """
    upper = np.cumsum(null_counts[:, ::-1], axis=1)[:, ::-1]
    totals = null_counts.sum(axis=1)
    index = np.clip(np.searchsorted(SCORE_GRID, projected, side="right") - 1, 0,
                    len(SCORE_GRID) - 2)
    p = (1.0 + upper[bins_of_pairs, index]) / (1.0 + totals[bins_of_pairs])
    np.fill_diagonal(p, 1.0)
    return np.clip(p, 0.0, 1.0)


def projected_graph(pvalues: np.ndarray, knn: int, edge_p: float):
    """The gene graph of the projected pairs.

    Each gene keeps its ``knn`` strongest partners by ``-log10 p`` among the pairs below
    ``edge_p``, and always its single best partner, through
    :func:`spacr.ops_clusters._knn_graph`.

    :param pvalues: from :func:`max_null_pvalues`.
    :param knn: partners per gene.
    :param edge_p: the p value an edge must stay below.
    :returns: an undirected igraph graph weighted by ``-log10 p``.
    """
    from .ops_clusters import _knn_graph

    strength = -np.log10(np.clip(pvalues, 1e-300, 1.0))
    keep = pvalues < edge_p
    best = np.argmax(np.where(np.eye(len(strength), dtype=bool), -np.inf, strength), axis=1)
    keep[np.arange(len(best)), best] = True
    keep = keep | keep.T
    masked = np.where(keep, strength, -np.inf)
    return _knn_graph(masked, knn)


def label_clusters_by_view(labels: np.ndarray, graph, view: np.ndarray,
                           views: Sequence[Mapping[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """The share of each cluster's within-cluster edges by argmax view and view block.

    A cluster is ``thin`` when one view carries at least :data:`THIN_SHARE` of its edges and
    that view has fewer columns than the median view, ``broad`` when that view has as many
    columns as the median view or more, and ``mixed`` otherwise.

    :param labels: one cluster per gene.
    :param graph: from :func:`projected_graph`.
    :param view: the argmax view matrix from :func:`max_projection`.
    :param views: the views, as from :func:`read_views`.
    :returns: ``(shares, kinds)``: one row per cluster and view with ``share`` and ``block``,
        and one row per cluster with its ``label`` and edge count.
    """
    sizes = np.array([len(v["columns"]) for v in views])
    median = float(np.median(sizes))
    rows, kinds = [], []
    edges = np.array(graph.get_edgelist(), int).reshape(-1, 2)
    for cluster in np.unique(labels):
        inside = (labels[edges[:, 0]] == cluster) & (labels[edges[:, 1]] == cluster)
        from_view = view[edges[inside, 0], edges[inside, 1]]
        n = int(inside.sum())
        shares = np.bincount(from_view, minlength=len(views)) / max(n, 1)
        for v, share in zip(views, shares):
            rows.append({"cluster": int(cluster), "view": v["name"], "block": v["block"],
                         "share": float(share)})
        top = int(np.argmax(shares)) if n else 0
        if n and shares[top] >= THIN_SHARE:
            kind = "thin" if sizes[top] < median else "broad"
        else:
            kind = "mixed"
        kinds.append({"cluster": int(cluster), "edges": n, "label": kind,
                      "top_view": views[top]["name"] if n else ""})
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
        similarity=str(get("ops_maxproj_similarity", "shrunk")).lower(),
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
    """One permutation of every view: the null similarities' grid counts (a worker task)."""
    rows, groups, guide_rows, cells, targeting, control, views_ix, variance, kind, seed, pb, nb = args
    permuted = permuted_rows(rows, groups, guide_rows, cells, targeting, seed)
    return [_counts(view_similarity(permuted[:, ix], control, variance, seed, kind), pb, nb,
                    SIMILARITY_GRID) for ix in views_ix]


def _max_job(args):
    """One joint permutation: the null maxima's grid counts (a worker task)."""
    (rows, groups, guide_rows, cells, targeting, control, views_ix, variance, kind, seed, pb, nb,
     tables) = args
    permuted = permuted_rows(rows, groups, guide_rows, cells, targeting, seed)
    scores = [calibrate_view(view_similarity(permuted[:, ix], control, variance, seed, kind), t, pb)
              for ix, t in zip(views_ix, tables)]
    projected, _ = max_projection(scores)
    return _counts(projected, pb, nb, SCORE_GRID)


def _map(job, tasks: List[tuple], workers: int) -> List[Any]:
    """``job`` over the tasks in task order, in ``workers`` processes."""
    if workers <= 1:
        return [job(t) for t in tasks]
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(job, tasks))


def maxproj_scores(rows: np.ndarray, names: Sequence[str], n_cells: np.ndarray,
                   guide_rows: np.ndarray, guide_genes: Sequence[str], guide_cells: np.ndarray,
                   views_ix: Sequence[np.ndarray], *, variance: float = 0.99,
                   similarity: str = "shrunk", bins: int = 3,
                   calibration: int = 50, null: int = 200, seed: int = DEFAULT_SEED,
                   prefix: str = "nontargeting", workers: int = 1, report=None
                   ) -> Dict[str, Any]:
    """The calibrated scores of every view, their max projection and its max-null p values.

    :param rows: genes by columns (named measurement columns, controls included).
    :param names: the gene names; guides are matched on them.
    :param n_cells: cells per gene (the pair bins).
    :param guide_rows: one row per guide, in the profile's columns.
    :param guide_genes: each guide's gene.
    :param guide_cells: each guide's cell count.
    :param views_ix: each view's column indices.
    :param variance: the per-view PCA variance kept.
    :param similarity: see :func:`view_similarity`.
    :param bins: cell-count quantile bins per gene.
    :param calibration: permutations of each view's null; permutation ``p`` is seeded
        ``seed + p``.
    :param null: joint permutations of the max null, seeded ``seed + 100000 + p``.
    :param seed: the base seed.
    :param prefix: the control prefix.
    :param workers: processes; the results do not depend on it.
    :param report: called with progress lines.
    :returns: ``{"scores", "projected", "view", "p", "q", "pair_bins", "calibration_counts",
        "null_counts", "control"}``.
    """
    from .multiple_testing import adjust_p_values
    from .ops_clusters import _guide_groups, _is_control_row

    say = report or (lambda text: None)
    rows = np.asarray(rows, float)
    names = np.asarray(names).astype(str)
    control = _is_control_row(names, prefix)
    targeting = ~control
    groups = _guide_groups(names, guide_genes, control)
    guide_rows = np.asarray(guide_rows, float)
    cells = np.asarray(guide_cells, float)
    gb = cell_bins(np.asarray(n_cells, float), bins)
    pb = pair_bins(gb, bins)
    nb = bins * bins
    base = (rows, groups, guide_rows, cells, targeting, control, list(views_ix), variance,
            similarity)
    jobs = [(*base, seed + p, pb, nb) for p in range(calibration)]
    counts = _map(_view_job, jobs, workers)
    tables = [sum(c[v] for c in counts) for v in range(len(views_ix))]
    say(f"cluster_maxproj: calibrated {len(views_ix)} views on {calibration} permutations")
    scores = [calibrate_view(view_similarity(rows[:, ix], control, variance, seed, similarity), t,
                             pb) for ix, t in zip(views_ix, tables)]
    projected, view = max_projection(scores)
    jobs = [(*base, seed + 100000 + p, pb, nb, tables) for p in range(null)]
    null_counts = sum(_map(_max_job, jobs, workers))
    p = max_null_pvalues(projected, null_counts, pb)
    iu = np.triu_indices(len(names), 1)
    q = np.ones_like(p)
    adjusted, _ = adjust_p_values(p[iu], "fdr_bh")
    q[iu] = adjusted
    q[(iu[1], iu[0])] = adjusted
    say(f"cluster_maxproj: max null from {null} joint permutations")
    return {"scores": scores, "projected": projected, "view": view, "p": p, "q": q,
            "pair_bins": pb, "calibration_counts": tables, "null_counts": null_counts,
            "control": control}


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
    shares, kinds = label_clusters_by_view(labels, graph, result["view"], views)
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
    outputs["maxproj_partition"] = write_table(pd.DataFrame(
        {"gene": names, "cluster": labels, "label": [kind[c] for c in labels],
         "targeting": targeting}), os.path.join(out_dir, "maxproj_partition.csv"),
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
