"""Gene clusters of an optical pooled screen and their benchmarks.

Gene profiles are embedded (standardised, PCA to 99 % of the variance),
turned into a PHATE diffusion graph (k = 10 nearest neighbours, cosine
distance) and split into clusters with Leiden community detection. A sweep of
Leiden resolutions finds the one whose cluster count is closest to a target.
This is the clustering Funk et al. 2022 and Brieflow (Cheeseman lab,
``workflow/lib/cluster``, MIT licence) report; the code here is a clean
reimplementation of that published method, not a copy.

Every clustering is scored the way Brieflow scores it:

* **CORUM % and KEGG %**, the share of clusters with at least one enriched
  group: a two-sided Fisher exact test for every group in every cluster,
  Benjamini-Hochberg within the cluster, enriched at FDR < 0.05. The
  background is every clustered row, non-targeting rows included. Two
  variants are reported: ``may2025`` counts every member of a group
  (Brieflow's release of May 2025, which reproduces Funk et al.'s 24.3 % and
  13.5 %) and ``head`` only the members in the background (Brieflow's
  current code).
* **STRING precision and recall** of the gene pairs that share a cluster.
* **STRING AUPRC**: every gene pair ranked by the cosine of the two
  profiles, which needs no clustering at all.
* **ARI** against a reference clustering, such as Funk et al.'s.

Controls: the same pipeline on PCA columns permuted independently (the
shuffled-PC control), k-means on the same components at the same cluster
count (a baseline that shows how much the enrichment share depends on
cluster sizes alone), and a gene-label shuffle for the AUPRC.

:func:`consensus_clustering` repeats the clustering on bootstrap samples of
genes and of the guides within each gene, with its own seed and resolution
each time; :func:`co_membership` turns the runs into the probability that
two genes share a cluster.

PHATE (``phate``), Leiden (``leidenalg``) and ``igraph`` are optional: they
are the ``clustering`` extra, ``pip install "spacr[clustering]"``. Only the
PHATE + Leiden steps need them. The benchmark files (CORUM, KEGG and STRING
tables in Brieflow's format) are the user's own: spaCR does not ship them.
"""
from __future__ import annotations

import os
from functools import lru_cache, partial
from importlib import import_module
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

#: The message when the clustering extra is missing.
CLUSTERING_MISSING_MESSAGE = (
    "PHATE + Leiden clustering needs the optional packages phate, leidenalg "
    "and igraph. Install them with:  pip install \"spacr[clustering]\"")

#: The Leiden resolutions the plate A pilot swept.
DEFAULT_RESOLUTIONS = (5, 10, 12, 15, 16, 17, 18, 20, 22)


def _optional(module: str):
    """Import one of the clustering extra's modules, or explain how to get it."""
    try:
        return import_module(module)
    except ImportError as exc:
        raise ImportError(CLUSTERING_MISSING_MESSAGE) from exc


def _is_control(names: Sequence[str], prefix: str) -> np.ndarray:
    """True for every name that starts with the control prefix (any case)."""
    return pd.Series(np.asarray(names, dtype=object)).astype(str).str.lower().str.startswith(
        prefix.lower()).to_numpy()


def read_benchmark(path: str) -> pd.DataFrame:
    """Read a benchmark table in Brieflow's format.

    :param path: a tab-separated file. Group tables (CORUM, KEGG) have
        ``gene_name`` and ``group`` columns; the STRING pair table has
        ``gene_name`` and ``pair``.
    :returns: the table.
    :raises ValueError: when neither ``group`` nor ``pair`` is present.
    """
    from .tabular import read_table

    table = read_table(path, sep="\t", report=None)
    if "gene_name" not in table or not ({"group", "pair"} & set(table.columns)):
        raise ValueError(
            f"{path} is not a benchmark table: it needs gene_name and either "
            f"group or pair columns; it has {list(table.columns)}")
    return table


def benchmark_rows(genes: pd.DataFrame, guides: pd.DataFrame,
                   features: Sequence[str], *, min_cells: int = 10,
                   control_prefix: str = "nontargeting") -> pd.DataFrame:
    """The rows Brieflow clusters: targeting genes plus every control guide.

    :param genes: the gene table (``gene``, ``n_cells`` and ``features``).
    :param guides: the guide table (``guide``, ``gene``, ``n_cells``, ...).
    :param features: the profile columns.
    :param min_cells: rows with this many cells or fewer are left out.
    :param control_prefix: the control gene prefix.
    :returns: ``gene, n_cells, <features>``; each control guide is its own
        row, named ``<prefix>_<guide>``.
    """
    nt = guides[_is_control(guides["gene"], control_prefix)].copy()
    nt["gene"] = f"{control_prefix}_" + nt["guide"].astype(str)
    targeting = genes[~_is_control(genes["gene"], control_prefix)]
    columns = ["gene", "n_cells", *features]
    table = pd.concat([targeting[columns], nt[columns]], ignore_index=True)
    return table[table["n_cells"] > min_cells].reset_index(drop=True)


def embed_profiles(frame: pd.DataFrame, features: Sequence[str], *,
                   variance: float = 0.99, seed: int = 0
                   ) -> Tuple[np.ndarray, int, int]:
    """Standardise the profiles and keep the PCs holding ``variance``.

    :param frame: one row per gene.
    :param features: the profile columns.
    :param variance: the share of variance kept.
    :param seed: the PCA seed.
    :returns: ``(components, n_components, n_features_used)``. Columns with
        5 % or more missing values are dropped, other gaps take the column
        median, and constant columns are dropped.
    """
    from sklearn.decomposition import PCA

    x = frame[list(features)].to_numpy(float)
    x = np.where(np.isfinite(x), x, np.nan)
    keep = np.isnan(x).mean(axis=0) < 0.05
    x = x[:, keep]
    x = np.where(np.isnan(x), np.nanmedian(x, axis=0), x)
    spread = x.std(axis=0)
    x = (x[:, spread > 0] - x[:, spread > 0].mean(axis=0)) / spread[spread > 0]
    pca = PCA(n_components=variance, random_state=seed).fit(x)
    return pca.transform(x), int(pca.n_components_), int(keep.sum())


def shuffle_columns(components: np.ndarray, seed: Any = 0) -> np.ndarray:
    """Permute every column independently: the shuffled-PC control.

    :param components: one row per gene.
    :param seed: the permutation seed, or a ``numpy.random.Generator`` to
        draw from (it is advanced).
    :returns: a copy whose columns keep their values but lose their rows.
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    out = components.copy()
    for j in range(out.shape[1]):
        out[:, j] = rng.permutation(out[:, j])
    return out


def phate_leiden(components: np.ndarray, resolutions: Iterable[float], *,
                 knn: int = 10, metric: str = "cosine", seed: int = 42,
                 n_jobs: int = 1) -> Dict[float, np.ndarray]:
    """Leiden clusters of the PHATE diffusion graph, one per resolution.

    :param components: one row per gene.
    :param resolutions: Leiden resolution parameters.
    :param knn: PHATE's nearest neighbours.
    :param metric: PHATE's distance.
    :param seed: the PHATE and Leiden seed.
    :param n_jobs: PHATE's worker count.
    :returns: ``{resolution: labels}``.
    :raises ImportError: when the ``clustering`` extra is not installed.

    The graph is PHATE's diffusion operator made symmetric, ``(P + P.T) / 2``;
    Leiden optimises the RB configuration model on it with
    ``n_iterations=-1`` (until it stops improving).
    """
    phate = _optional("phate")
    igraph = _optional("igraph")
    leidenalg = _optional("leidenalg")
    model = phate.PHATE(random_state=seed, n_jobs=n_jobs, knn=knn, knn_dist=metric,
                        verbose=False)
    model.fit(components)
    w = model.graph.diff_op
    w = (w + w.T) / 2
    if hasattr(w, "tocoo"):
        w = w.tocoo()
    else:
        from scipy.sparse import coo_matrix
        w = coo_matrix(w)
    upper = w.row < w.col
    graph = igraph.Graph(n=components.shape[0], edges=list(zip(w.row[upper], w.col[upper])),
                         directed=False)
    graph.es["weight"] = w.data[upper].tolist()
    out = {}
    for resolution in resolutions:
        partition = leidenalg.find_partition(
            graph, leidenalg.RBConfigurationVertexPartition, weights="weight",
            n_iterations=-1, seed=seed, resolution_parameter=float(resolution))
        out[resolution] = np.array(partition.membership)
    return out


def _fisher_p(a: int, k: int, n: int, *, n_background: int) -> float:
    """The Fisher exact p-value of one cluster against one group.

    :param a: genes both in the cluster and in the group.
    :param k: genes in the group.
    :param n: genes in the cluster.
    :param n_background: genes in the background.
    :returns: the two-sided Fisher exact p-value of the 2x2 table.
    """
    from scipy.stats import fisher_exact

    b, c = k - a, n - a
    return fisher_exact([[a, b], [c, max(n_background - (a + b + c), 0)]])[1]


def group_enrichment(groups: pd.DataFrame, genes: Sequence[str], clusters: np.ndarray, *,
                     restrict: bool = True, control_prefix: str = "nontargeting"
                     ) -> Tuple[float, pd.DataFrame]:
    """The share of clusters enriched for at least one group.

    :param groups: ``gene_name, group`` rows (CORUM or KEGG).
    :param genes: the clustered row names, controls included.
    :param clusters: one label per row.
    :param restrict: True keeps only the group members in the background
        (Brieflow's current ``head`` code); False counts every member
        (Brieflow's May 2025 release, the ``may2025`` variant).
    :param control_prefix: control rows are background, never cluster
        members.
    :returns: ``(share, table)``; the table has one row per cluster with
        its size, the number of enriched groups and up to ten of them.
    """
    from statsmodels.stats.multitest import multipletests

    background = set(genes)
    keep = ~_is_control(genes, control_prefix)
    members = {g: (set(d.gene_name) & background if restrict else set(d.gene_name))
               for g, d in groups.groupby("group")}
    members = {g: s for g, s in members.items() if s}
    names = list(members)
    sizes = np.array([len(members[n]) for n in names])
    n_background = len(background)

    fisher = lru_cache(maxsize=None)(partial(_fisher_p, n_background=n_background))

    table = []
    frame = pd.DataFrame({"gene": np.asarray(genes)[keep], "cluster": np.asarray(clusters)[keep]})
    for cluster, d in frame.groupby("cluster"):
        in_cluster = set(d.gene)
        n = len(in_cluster)
        overlap = [len(in_cluster & members[name]) for name in names]
        p = np.array([fisher(int(a), int(k), n) for a, k in zip(overlap, sizes)])
        q = multipletests(p, method="fdr_bh")[1] if len(p) else p
        enriched = [names[i] for i in np.flatnonzero(q < 0.05)]
        table.append({"cluster": int(cluster), "n_genes": n, "n_enriched": len(enriched),
                      "enriched": "; ".join(enriched[:10])})
    table = pd.DataFrame(table, columns=["cluster", "n_genes", "n_enriched", "enriched"])
    return (float((table.n_enriched > 0).mean()) if len(table) else 0.0), table


def string_positives(pairs: pd.DataFrame) -> Set[Tuple[str, str]]:
    """The STRING gene pairs, each as an ordered ``(a, b)`` with ``a < b``.

    :param pairs: ``gene_name, pair`` rows of the STRING benchmark.
    :returns: the set of pairs.
    """
    positives: Set[Tuple[str, str]] = set()
    for _, d in pairs.groupby("pair"):
        u = sorted(d.gene_name.unique())
        positives |= {(u[i], u[j]) for i in range(len(u)) for j in range(i + 1, len(u))}
    return positives


def string_cluster_pairs(positives: Set[Tuple[str, str]], genes: Sequence[str],
                         clusters: np.ndarray, *, control_prefix: str = "nontargeting"
                         ) -> Dict[str, float]:
    """STRING precision and recall of the pairs that share a cluster.

    :param positives: :func:`string_positives`.
    :param genes: the clustered row names.
    :param clusters: one label per row.
    :param control_prefix: control rows are left out.
    :returns: ``precision, recall, f1, tp, fp, fn``. Recall counts only the
        STRING pairs with both genes clustered.
    """
    keep = ~_is_control(genes, control_prefix)
    g = np.asarray(genes)[keep]
    c = np.asarray(clusters)[keep]
    tp = fp = 0
    for cluster in np.unique(c):
        m = sorted(g[c == cluster])
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                if (m[i], m[j]) in positives:
                    tp += 1
                else:
                    fp += 1
    gene_cluster = dict(zip(g, c))
    fn = sum(1 for a, b in positives
             if a in gene_cluster and b in gene_cluster and gene_cluster[a] != gene_cluster[b])
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "tp": tp, "fp": fp, "fn": fn}


def string_auprc(components: np.ndarray, genes: Sequence[str],
                 positives: Set[Tuple[str, str]], *, shuffle_seed: Optional[int] = None,
                 control_prefix: str = "nontargeting", curve_points: int = 400
                 ) -> Tuple[float, float, pd.DataFrame]:
    """Area under the STRING precision-recall curve of profile cosines.

    :param components: one row per gene.
    :param genes: the row names; control rows are left out.
    :param positives: :func:`string_positives`.
    :param shuffle_seed: shuffle the gene labels with this seed first (the
        control); None scores the real labels.
    :param control_prefix: the control prefix.
    :param curve_points: points kept on the returned curve.
    :returns: ``(auprc, base_rate, curve)``. Every gene pair is ranked by
        the cosine of its two profiles; the base rate is the share of pairs
        that are STRING pairs, which a random ranking scores.
    """
    from sklearn.metrics import average_precision_score

    keep = ~_is_control(genes, control_prefix)
    x = np.asarray(components)[keep]
    g = np.asarray(genes)[keep]
    if shuffle_seed is not None:
        g = np.random.default_rng(shuffle_seed).permutation(g)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    upper = np.triu_indices(len(g), 1)
    scores = (x @ x.T)[upper]
    a, b = g[upper[0]], g[upper[1]]
    labels = np.fromiter(((min(p, q), max(p, q)) in positives for p, q in zip(a, b)), bool,
                         len(scores))
    labels = labels[np.argsort(-scores, kind="stable")]
    tp = np.cumsum(labels)
    rank = np.arange(1, len(labels) + 1)
    ap = float(average_precision_score(labels, -np.arange(len(labels))))
    pick = np.unique(np.r_[np.geomspace(1, len(labels), curve_points).astype(int) - 1,
                           len(labels) - 1])
    curve = pd.DataFrame({"rank": rank[pick], "precision": (tp / rank)[pick],
                          "recall": (tp / max(labels.sum(), 1))[pick]})
    return ap, float(labels.mean()), curve


def score_clusters(name: str, genes: Sequence[str], clusters: np.ndarray, *,
                   corum: pd.DataFrame, kegg: pd.DataFrame,
                   positives: Set[Tuple[str, str]],
                   control_prefix: str = "nontargeting") -> Dict[str, Any]:
    """One benchmark row: cluster count, CORUM %, KEGG % and STRING P/R.

    :param name: the row's label.
    :param genes: the clustered row names.
    :param clusters: one label per row.
    :param corum: CORUM groups (:func:`read_benchmark`).
    :param kegg: KEGG groups.
    :param positives: :func:`string_positives`.
    :param control_prefix: the control prefix.
    :returns: ``profile, clusters, genes, CORUM_pct_enriched_head,
        CORUM_pct_enriched_may2025, KEGG_..., STRING_precision,
        STRING_recall, STRING_f1, STRING_tp, STRING_fp, STRING_fn``.
    """
    clusters = np.asarray(clusters)
    keep = ~_is_control(genes, control_prefix)
    row: Dict[str, Any] = {"profile": name, "clusters": int(len(np.unique(clusters[keep]))),
                           "genes": int(keep.sum())}
    for label, groups in (("CORUM", corum), ("KEGG", kegg)):
        for restrict, tag in ((True, "head"), (False, "may2025")):
            share, _ = group_enrichment(groups, genes, clusters, restrict=restrict,
                                        control_prefix=control_prefix)
            row[f"{label}_pct_enriched_{tag}"] = round(100 * share, 1)
    pairs = string_cluster_pairs(positives, genes, clusters, control_prefix=control_prefix)
    row.update({f"STRING_{k}": (round(v, 4) if isinstance(v, float) else v)
                for k, v in pairs.items()})
    return row


def closest_resolution(counts: Mapping[float, int], target: int) -> float:
    """The resolution whose cluster count is closest to ``target``.

    :param counts: ``{resolution: cluster count}``.
    :param target: the wanted cluster count.
    :returns: the resolution; ties go to the one listed first.
    """
    return min(counts, key=lambda r: abs(counts[r] - target))


def adjusted_rand_vs(reference: pd.DataFrame, genes: Sequence[str], clusters: np.ndarray, *,
                     null_draws: int = 200, seed: Any = 0) -> Dict[str, float]:
    """Adjusted Rand index of a clustering against a reference clustering.

    :param reference: ``gene, cluster`` rows, such as Funk et al.'s
        published interphase clusters.
    :param genes: the clustered row names.
    :param clusters: one label per row.
    :param null_draws: label shuffles for the control.
    :param seed: the shuffle seed, or a ``numpy.random.Generator``.
    :returns: ``ari``, ``ari_shuffled_p95`` and ``genes`` (the genes in both).
    """
    from sklearn.metrics import adjusted_rand_score

    merged = pd.DataFrame({"gene": genes, "c": clusters}).merge(
        reference[["gene", "cluster"]], on="gene")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    ari = adjusted_rand_score(merged["cluster"], merged["c"])
    null = [adjusted_rand_score(merged["cluster"], rng.permutation(merged["c"]))
            for _ in range(null_draws)]
    return {"ari": float(ari), "ari_shuffled_p95": float(np.percentile(null, 95)) if null else 0.0,
            "genes": int(len(merged))}


def benchmark_clusters(table: pd.DataFrame, features: Sequence[str], *,
                       corum: pd.DataFrame, kegg: pd.DataFrame, string: pd.DataFrame,
                       resolutions: Sequence[float] = DEFAULT_RESOLUTIONS,
                       target: int = 251, seed: int = 42,
                       reference: Optional[pd.DataFrame] = None,
                       baselines: Sequence[int] = (222, 251), label: str = "spaCR",
                       control_prefix: str = "nontargeting", n_jobs: int = 1,
                       shuffles: int = 5) -> Dict[str, Any]:
    """The plate A pilot's cluster benchmark, from one profile table.

    :param table: :func:`benchmark_rows` output.
    :param features: the profile columns.
    :param corum: CORUM groups.
    :param kegg: KEGG groups.
    :param string: the STRING pair table.
    :param resolutions: the Leiden resolution sweep.
    :param target: the cluster count the headline resolution is chosen for.
    :param seed: the PHATE and Leiden seed.
    :param reference: an optional ``gene, cluster`` reference clustering,
        scored on the same genes and used for the ARI.
    :param baselines: extra k-means cluster counts besides the headline
        count; counts that are not below the number of rows are skipped.
    :param label: the profile name in the table.
    :param control_prefix: the control prefix.
    :param n_jobs: PHATE's worker count.
    :param shuffles: gene-label shuffles for the AUPRC control.
    :returns: ``{"table", "summary", "assignments", "curve", "components"}``:
        the benchmark rows (every resolution, its shuffled-PC control, the
        k-means baseline and its control, the reference), a summary, the
        cluster labels per resolution, the STRING precision-recall curve
        and the embedded components.
    :raises ImportError: when the ``clustering`` extra is not installed.
    """
    from sklearn.cluster import KMeans

    components, n_components, n_features = embed_profiles(table, features)
    names = table["gene"].to_numpy()
    positives = string_positives(string)
    real = phate_leiden(components, resolutions, seed=seed, n_jobs=n_jobs)
    rng = np.random.default_rng(0)
    shuffled = shuffle_columns(components, seed=rng)
    control = phate_leiden(shuffled, resolutions, seed=seed, n_jobs=n_jobs)
    rows = []
    kwargs = {"corum": corum, "kegg": kegg, "positives": positives,
              "control_prefix": control_prefix}
    targeting = ~_is_control(names, control_prefix)
    for r in resolutions:
        rows.append(dict(score_clusters(label, names, real[r], **kwargs), resolution=r,
                         method="PHATE+Leiden"))
        rows.append(dict(score_clusters(f"{label} shuffled PCs (control)", names, control[r],
                                        **kwargs), resolution=r, method="PHATE+Leiden"))
    counts = {r: int(len(np.unique(real[r][targeting]))) for r in resolutions}
    best = closest_resolution(counts, target)
    for k in sorted(k for k in {counts[best], *baselines} if 1 < k < len(table)):
        km = KMeans(k, n_init=4, random_state=0).fit_predict(components)
        rows.append(dict(score_clusters(f"{label} k-means baseline", names, km, **kwargs),
                         resolution=None, method=f"k-means k={k}"))
        km = KMeans(k, n_init=2, random_state=0).fit_predict(shuffled)
        rows.append(dict(score_clusters(f"{label} k-means shuffled PCs (control)", names, km,
                                        **kwargs), resolution=None, method=f"k-means k={k}"))
    summary: Dict[str, Any] = {"rows_clustered": int(len(table)), "genes": int(targeting.sum()),
                               "pcs": n_components, "features_used": n_features,
                               "cluster_counts": counts, "headline_resolution": best,
                               "seed": seed}
    if reference is not None:
        ref = reference[reference["gene"].isin(set(names))]
        rows.append(dict(score_clusters("reference clusters (same genes)",
                                        ref["gene"].to_numpy(), ref["cluster"].to_numpy(),
                                        **kwargs), resolution=None, method="reference"))
        ari = adjusted_rand_vs(reference, names, real[best], seed=rng)
        summary.update({"reference_ARI": round(ari["ari"], 4),
                        "reference_ARI_shuffled_p95": round(ari["ari_shuffled_p95"], 4)})
    ap, base, curve = string_auprc(components, names, positives, control_prefix=control_prefix)
    shuffled_ap = [string_auprc(components, names, positives, shuffle_seed=s,
                                control_prefix=control_prefix)[0] for s in range(shuffles)]
    summary.update({"string_AUPRC_cosine": round(ap, 5),
                    "string_AUPRC_shuffled_mean": round(float(np.mean(shuffled_ap)), 5)
                    if shuffled_ap else None,
                    "string_pair_base_rate": round(base, 5)})
    assignments = pd.DataFrame({"gene": names, "n_cells": table["n_cells"].to_numpy()})
    for r in resolutions:
        assignments[f"leiden_r{r}"] = real[r]
    out = pd.DataFrame(rows)
    out.insert(0, "arm", label)
    return {"table": out, "summary": summary, "assignments": assignments, "curve": curve,
            "components": components}


def consensus_rows(genes: pd.DataFrame, guides: pd.DataFrame, columns: Sequence[str], *,
                   rng: Optional[np.random.Generator] = None, min_cells: int = 10,
                   control_prefix: str = "nontargeting", pseudo_size: int = 4,
                   pseudo_seed: int = 0) -> pd.DataFrame:
    """Gene profiles for consensus clustering, optionally with guides resampled.

    :param genes: the gene table; genes with ``min_cells`` cells or fewer
        are left out.
    :param guides: the guide table with ``columns``.
    :param columns: the profile columns.
    :param rng: when given, each gene's guides are drawn with replacement
        before the median is taken (the guide bootstrap).
    :param min_cells: the gene cell-count floor.
    :param control_prefix: the control prefix.
    :param pseudo_size: control guides are grouped at random into
        pseudo-genes of this many guides, named ``NTpseudo_<k>``, so the
        controls are clustered at the same resolution as the genes.
    :param pseudo_seed: the seed of that grouping.
    :returns: ``gene, n_cells, <columns>``, one row per gene and
        pseudo-gene, sorted by name.
    """
    g = guides[guides["n_cells"] > 0].copy()
    nt = g[_is_control(g["gene"], control_prefix)].reset_index(drop=True)
    perm = np.random.default_rng(pseudo_seed).permutation(len(nt))
    nt["gene"] = ["NTpseudo_%02d" % (k // pseudo_size) for k in np.argsort(perm)]
    nt = nt[nt["gene"].map(nt["gene"].value_counts()) == pseudo_size]
    targeting = g[~_is_control(g["gene"], control_prefix)]
    targeting = targeting[targeting["gene"].isin(set(genes[genes["n_cells"] > min_cells]["gene"]))]
    names, profiles, cells = [], [], []
    for gene, d in pd.concat([targeting, nt], ignore_index=True).groupby("gene", sort=True):
        x = d[list(columns)].to_numpy(float)
        if rng is not None:
            x = x[rng.integers(0, len(x), len(x))]
        names.append(gene)
        profiles.append(np.median(x, axis=0))
        cells.append(int(d["n_cells"].sum()))
    out = pd.DataFrame(np.vstack(profiles), columns=list(columns))
    out.insert(0, "gene", names)
    out.insert(1, "n_cells", cells)
    return out


def consensus_run(genes: pd.DataFrame, guides: pd.DataFrame, columns: Sequence[str],
                  run: int, *, resolutions: Sequence[float], base_seed: int = 1000,
                  subsample: float = 0.9, **row_options) -> Tuple[float, int, np.ndarray]:
    """One consensus clustering run.

    :param genes: the gene table.
    :param guides: the guide table.
    :param columns: the profile columns.
    :param run: the run number; its generator is seeded ``base_seed + run``.
    :param resolutions: the sweep; run ``k`` uses ``resolutions[k % len]``.
    :param base_seed: see ``run``.
    :param subsample: the share of rows clustered, drawn without
        replacement (PHATE cannot take duplicated rows).
    :param row_options: passed to :func:`consensus_rows`.
    :returns: ``(resolution, clusters, labels)``; ``labels`` has one entry per
        row of :func:`consensus_rows` without resampling, -1 for rows left
        out of this run.
    :raises ImportError: when the ``clustering`` extra is not installed.
    """
    rng = np.random.default_rng(base_seed + run)
    table = consensus_rows(genes, guides, columns, rng=rng, **row_options)
    n = len(table)
    sub = np.sort(rng.choice(n, int(subsample * n), replace=False))
    components, _, _ = embed_profiles(table.iloc[sub].reset_index(drop=True), columns)
    resolution = resolutions[run % len(resolutions)]
    labels = phate_leiden(components, [resolution], seed=int(rng.integers(1, 10 ** 6)))[resolution]
    full = np.full(n, -1, np.int32)
    full[sub] = labels
    return resolution, int(len(np.unique(labels))), full


def consensus_clustering(genes: pd.DataFrame, guides: pd.DataFrame, columns: Sequence[str], *,
                         runs: int = 100,
                         resolutions: Sequence[float] = (3, 5, 8, 10, 13, 17, 22, 26),
                         base_seed: int = 1000, subsample: float = 0.9,
                         **row_options) -> Dict[str, Any]:
    """Consensus clustering: many runs on bootstrap samples of genes and guides.

    :param genes: the gene table.
    :param guides: the guide table.
    :param columns: the profile columns.
    :param runs: the number of runs.
    :param resolutions: the resolution sweep the runs cycle through.
    :param base_seed: run ``k`` is seeded ``base_seed + k``.
    :param subsample: the share of rows in each run.
    :param row_options: passed to :func:`consensus_rows`.
    :returns: ``{"genes", "labels", "runs"}``: the row names, a
        ``runs x rows`` label matrix (-1 = left out of that run) and every
        run's resolution and cluster count. Pass ``labels`` to
        :func:`co_membership`.
    :raises ImportError: when the ``clustering`` extra is not installed.

    Every run differs from the others in three ways, so the co-membership
    reflects the sampling of genes, of guides and of the algorithm: a
    ``subsample`` share of the rows, each gene's guides drawn with
    replacement, and its own PHATE and Leiden seed and resolution.
    """
    reference = consensus_rows(genes, guides, columns, **row_options)
    labels, meta = [], []
    for k in range(int(runs)):
        resolution, clusters, full = consensus_run(
            genes, guides, columns, k, resolutions=resolutions, base_seed=base_seed,
            subsample=subsample, **row_options)
        labels.append(full)
        meta.append({"run": k, "resolution": resolution, "clusters": clusters})
    return {"genes": reference["gene"].to_numpy(),
            "labels": np.vstack(labels) if labels else np.zeros((0, len(reference)), np.int32),
            "runs": meta}


def co_membership(labels: np.ndarray) -> np.ndarray:
    """How often two rows share a cluster when both were clustered.

    :param labels: a ``runs x rows`` matrix from
        :func:`consensus_clustering`; -1 marks a row left out of a run.
    :returns: a symmetric ``rows x rows`` matrix of probabilities, NaN for a
        pair never clustered together, 1 on the diagonal of every row that
        was clustered at least once.
    """
    labels = np.asarray(labels)
    n = labels.shape[1]
    together = np.zeros((n, n), np.float64)
    both = np.zeros((n, n), np.float64)
    for run in labels:
        present = run >= 0
        p = present.astype(np.float64)
        both += np.outer(p, p)
        for cluster in np.unique(run[present]):
            m = (run == cluster).astype(np.float64)
            together += np.outer(m, m)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(both > 0, together / np.where(both > 0, both, 1), np.nan)


def _pick_column(table: pd.DataFrame, path: str, given: Optional[str],
                 options: Sequence[str], what: str) -> str:
    """The column of a reference table to use for ``what``.

    :param table: the reference table.
    :param path: its file, named in the error.
    :param given: the column the caller named, or None to search ``options``.
    :param options: the column names tried in order when ``given`` is None.
    :param what: what the column holds, ``gene`` or ``cluster``.
    :returns: the column name.
    :raises ValueError: when ``given`` is absent or no option is present.
    """
    if given:
        if given not in table:
            raise ValueError(f"{path} has no {what} column {given!r}")
        return given
    for option in options:
        if option in table:
            return option
    raise ValueError(f"{path} has no {what} column; looked for {list(options)}")


def reference_clusters(path: str, *, gene: Optional[str] = None,
                       cluster: Optional[str] = None) -> pd.DataFrame:
    """Read a reference clustering, such as Funk et al.'s published table.

    :param path: a CSV, TSV or Excel file.
    :param gene: the gene column; found among ``gene``, ``gene_symbol``
        and ``gene_name`` when None.
    :param cluster: the cluster column; found among ``cluster`` and
        ``interphase_cluster`` when None.
    :returns: ``gene, cluster`` rows with a cluster, the cluster as int.
    :raises ValueError: when a column cannot be found.
    """
    from .tabular import read_table, table_format

    sniffed = table_format(path) != "csv" or str(path).lower().endswith(".txt")
    options = {} if sniffed else {"low_memory": False}
    table = read_table(path, report=None, **options)

    g = _pick_column(table, path, gene, ("gene", "gene_symbol", "gene_name"), "gene")
    c = _pick_column(table, path, cluster, ("cluster", "interphase_cluster"), "cluster")
    out = table[[g, c]].dropna().rename(columns={g: "gene", c: "cluster"})
    out["cluster"] = out["cluster"].astype(int)
    return out.reset_index(drop=True)


def _resolutions(value: Any) -> List[float]:
    """Resolutions from a setting: a list or a comma-separated string."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return list(DEFAULT_RESOLUTIONS)
    items = value.split(",") if isinstance(value, str) else list(value)
    out = []
    for item in items:
        number = float(item)
        out.append(int(number) if number == int(number) else number)
    return out


def _aggregate_benchmark(guides: pd.DataFrame, genes: pd.DataFrame, columns: Sequence[str],
                         settings: Mapping[str, Any], destination: str, stem: str,
                         control_prefix: str = "nontargeting") -> Dict[str, Any]:
    """The cluster benchmark and consensus runs of the OPS aggregate phase.

    :param guides: the guide table of the profile benchmarked.
    :param genes: its gene table.
    :param columns: its profile columns.
    :param settings: read for ``ops_benchmark_corum``,
        ``ops_benchmark_kegg``, ``ops_benchmark_string``,
        ``ops_reference_clusters``, ``ops_cluster_target``,
        ``ops_cluster_resolutions``, ``ops_cluster_seed`` and
        ``ops_consensus_runs``.
    :param destination: the output folder.
    :param stem: the output name stem, ``<plate>[tag]``.
    :param control_prefix: the control prefix.
    :returns: the benchmark summary with the paths written.
    :raises ValueError: when a benchmark file is not given or not found.
    """
    import json
    import os

    from .tabular import write_table

    paths = {}
    for name in ("corum", "kegg", "string"):
        path = settings.get(f"ops_benchmark_{name}")
        if not path or not os.path.isfile(str(path)):
            raise ValueError(
                f"ops_cluster_benchmark needs ops_benchmark_{name}, a {name.upper()} table "
                f"in Brieflow's format; got {path!r}")
        paths[name] = str(path)
    reference = settings.get("ops_reference_clusters") or None
    ref = reference_clusters(str(reference)) if reference else None
    table = benchmark_rows(genes, guides, columns, control_prefix=control_prefix)
    out = benchmark_clusters(
        table, columns, corum=read_benchmark(paths["corum"]), kegg=read_benchmark(paths["kegg"]),
        string=read_benchmark(paths["string"]),
        resolutions=_resolutions(settings.get("ops_cluster_resolutions")),
        target=int(settings.get("ops_cluster_target") or 251),
        seed=int(42 if settings.get("ops_cluster_seed") is None else settings["ops_cluster_seed"]),
        reference=ref, control_prefix=control_prefix)
    if ref is not None:
        whole = score_clusters("reference clusters (all reference genes)",
                               ref["gene"].to_numpy(), ref["cluster"].to_numpy(),
                               corum=read_benchmark(paths["corum"]),
                               kegg=read_benchmark(paths["kegg"]),
                               positives=string_positives(read_benchmark(paths["string"])),
                               control_prefix=control_prefix)
        out["table"] = pd.concat([out["table"], pd.DataFrame(
            [dict(whole, arm=out["table"]["arm"].iloc[0], resolution=None,
                  method="reference")])], ignore_index=True)
    written = {}
    for name, frame in (("cluster_benchmark", out["table"]),
                        ("cluster_assignments", out["assignments"]),
                        ("string_pr_curve", out["curve"])):
        target = os.path.join(destination, f"ops_{name}_{stem}.csv")
        write_table(frame, target)
        written[name] = target
    summary = {**out["summary"], "benchmark_files": paths, "reference_clusters": reference,
               "outputs": written}
    runs = int(settings.get("ops_consensus_runs") or 0)
    if runs > 0:
        consensus = consensus_clustering(genes, guides, columns, runs=runs,
                                         control_prefix=control_prefix)
        target = os.path.join(destination, f"ops_consensus_partitions_{stem}.npz")
        np.savez_compressed(target, labels=consensus["labels"],
                            genes=consensus["genes"].astype(str))
        written["consensus_partitions"] = target
        summary["consensus_runs"] = consensus["runs"]
    with open(os.path.join(destination, f"ops_cluster_summary_{stem}.json"), "w",
              encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1, default=str)
    return summary


#: The MitoCarta sub-compartments the retrieval benchmark scores: each label
#: and the ``MitoCarta3.0_MitoPathways`` text that puts a gene in it.
MITOCARTA_SUBCOMPARTMENTS: Dict[str, Tuple[str, ...]] = {
    "OXPHOS": ("OXPHOS",),
    "mitoribosome": ("Mitochondrial ribosome",),
    "ATP synthase (complex V)": ("Complex V",),
    "membrane organisation (dynamics, cristae, phospholipid)":
        ("Mitochondrial dynamics", "Cristae", "Phospholipid"),
}

#: The MitoCarta 3.0 sheet that lists the human genes.
MITOCARTA_SHEET = "A Human MitoCarta3.0"


def _read_any(path: str, sheet: Optional[str] = None) -> pd.DataFrame:
    """Read a table exactly as written; ``sheet`` picks an Excel sheet."""
    from .tabular import read_table

    options = {"sheet_name": sheet} if sheet else {}
    if str(path).lower().endswith((".csv", ".csv.gz")):
        options["low_memory"] = False
    return read_table(str(path), canonicalise=False, report=None, **options)


def read_mitocarta(path: str, sheet: str = MITOCARTA_SHEET,
                   subcompartments: Mapping[str, Sequence[str]] = MITOCARTA_SUBCOMPARTMENTS
                   ) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """The MitoCarta genes and the genes of each sub-compartment.

    :param path: MitoCarta 3.0 (``Human.MitoCarta3.0.xls``), or any table with
        ``Symbol`` and ``MitoCarta3.0_MitoPathways`` columns.
    :param sheet: the Excel sheet; ignored for other formats.
    :param subcompartments: ``{label: pathway texts}``; a gene whose pathway
        field contains any of the texts is in that sub-compartment.
    :returns: ``(genes, {label: genes})``.
    """
    excel = str(path).lower().endswith((".xls", ".xlsx"))
    table = _read_any(path, sheet if excel else None)
    table = table[["Symbol", "MitoCarta3.0_MitoPathways"]].fillna("").astype(str)
    out = {}
    for label, texts in subcompartments.items():
        hit = table["MitoCarta3.0_MitoPathways"].map(lambda s, t=tuple(texts): any(x in s for x in t))
        out[label] = set(table.Symbol[hit])
    return set(table.Symbol), out


def mitocarta_enrichment(assignments: Mapping[str, Any], mitocarta: Set[str]) -> Dict[str, Any]:
    """MitoCarta recovery of one clustering.

    :param assignments: ``{gene: cluster}`` of the targeting genes; they are
        the background.
    :param mitocarta: the MitoCarta genes.
    :returns: the gene and MitoCarta counts, the cluster count, the clusters
        enriched for MitoCarta genes (one-sided Fisher, BH q < 0.05), the
        MitoCarta genes in them, their purity and the share of MitoCarta
        recovered.
    """
    from scipy.stats import fisher_exact
    from statsmodels.stats.multitest import multipletests

    g = pd.Series(assignments)
    background = set(g.index)
    mito = background & mitocarta
    rows = []
    for cluster, members in g.groupby(g):
        genes = set(members.index)
        a = len(genes & mito)
        p = fisher_exact([[a, len(genes) - a], [len(mito) - a, len(background) - len(genes) - len(mito) + a]],
                         alternative="greater")[1]
        rows.append((cluster, len(genes), a, p))
    table = pd.DataFrame(rows, columns=["cluster", "size", "mito", "p"])
    table["q"] = multipletests(table.p, method="fdr_bh")[1]
    enriched = table[table.q < 0.05]
    return {"genes": len(background), "mitocarta_genes": len(mito), "clusters": int(len(table)),
            "mito_enriched_clusters": int(len(enriched)),
            "mito_genes_in_enriched": int(enriched.mito.sum()),
            "purity_in_enriched": float(enriched.mito.sum() / max(enriched["size"].sum(), 1)),
            "share_of_mitocarta_recovered": float(enriched.mito.sum() / max(len(mito), 1))}


def _profile_cosines(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """The row names and float32 cosine matrix of a profile file (``names`` with ``pcs`` or ``S``)."""
    with np.load(path, allow_pickle=True) as stored:
        names = stored["names"]
        if "pcs" in stored:
            x = np.asarray(stored["pcs"], float)
            x = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
            return names, (x @ x.T).astype(np.float32)
        return names, stored["S"].astype(np.float32)


def _symbols(names: Sequence[Any]) -> np.ndarray:
    """Gene symbols of profile rows; ``"A&B"`` gives ``"A"``."""
    from .ops_reference import gene_symbol

    return np.array([gene_symbol(x) for x in names])


def _controls(names: Sequence[Any]) -> np.ndarray:
    """True for control guides and pseudo-genes."""
    from .ops_reference import is_control

    return is_control(names)


def subcompartment_retrieval(label: str, names: np.ndarray, sims: np.ndarray, mitocarta: Set[str],
                             subcompartments: Mapping[str, Set[str]]) -> List[Dict[str, Any]]:
    """Retrieval mAP within MitoCarta, one row per sub-compartment.

    :param label: the arm label of the rows.
    :param names: the profile row names.
    :param sims: their similarity matrix; its diagonal is set to ``-inf``.
    :param mitocarta: the MitoCarta genes.
    :param subcompartments: ``{label: genes}`` of :func:`read_mitocarta`.
    :returns: per sub-compartment with a query: queries, mAP and its
        bootstrap interval (500 resamples, seed 0), the random expectation
        and the candidate count. The queries are the sub-compartment's
        genes, the candidates every MitoCarta gene of the arm, the positives
        the other genes of the same sub-compartment.
    """
    from .sp_stats import _expected_ap, _pair_average_precision, bootstrap_ci

    np.fill_diagonal(sims, -np.inf)
    sym = _symbols(names)
    univ = np.isin(sym, list(mitocarta)) & ~_controls(names)
    rows = []
    for name, genes in subcompartments.items():
        inc = np.isin(sym, list(genes))
        positives = inc[:, None] & inc[None, :]
        np.fill_diagonal(positives, False)
        ap, npos, ntot = _pair_average_precision(sims, positives & univ[None, :] & univ[:, None], univ)
        q = univ & inc & np.isfinite(ap)
        frame = pd.DataFrame({"ap": ap[q], "expected_ap": [_expected_ap(a, b - a)
                                                           for a, b in zip(npos[q], ntot[q])]})
        if len(frame):
            lo, hi = bootstrap_ci(frame.ap, 500)
            rows.append({"arm": label, "sub_compartment": name, "queries": int(len(frame)),
                         "mAP": float(frame.ap.mean()), "mAP_lo": lo, "mAP_hi": hi,
                         "expected_random": float(frame.expected_ap.mean()),
                         "candidates (MitoCarta genes)": int(univ.sum())})
    return rows


def _pair_comembership(labels: np.ndarray, rows: np.ndarray) -> float:
    """The share of runs in which two of ``rows`` share a cluster, over every pair clustered together."""
    sub = labels[:, rows]
    total = count = 0.0
    for a in range(len(rows)):
        for b in range(a + 1, len(rows)):
            ok = (sub[:, a] >= 0) & (sub[:, b] >= 0)
            total += ok.sum()
            count += (sub[ok, a] == sub[ok, b]).sum()
    return count / max(total, 1)


def module_coherence(consensus_dir: str, genes: Iterable[str], rng: np.random.Generator,
                     n_null: int = 200) -> Optional[Dict[str, Any]]:
    """How tightly an external module holds together in a consensus atlas.

    :param consensus_dir: a consensus folder of the modules phase
        (``rows_reference.parquet``, ``consensus_partitions.npz``,
        ``reference.npz``).
    :param genes: the module's genes; taken in sorted order.
    :param rng: the generator of the null draws (advanced).
    :param n_null: fitness-matched random gene sets of the same size.
    :returns: None without partitions; ``{"present"}`` alone for fewer than
        three genes; else the mean pairwise co-membership over the runs,
        the null mean, 99th percentile and percentile, whether it is above
        that 99th percentile, and the largest share of the module in one
        ~250-cluster of the reference clustering.
    """
    from .ops_modules import MODULE_RESOLUTIONS
    from .tabular import read_table

    partitions = os.path.join(consensus_dir, "consensus_partitions.npz")
    if not os.path.isfile(partitions):
        return None
    rows = read_table(os.path.join(consensus_dir, "rows_reference.parquet"), canonicalise=False, report=None)
    with np.load(partitions, allow_pickle=True) as stored:
        labels = stored["labels"]
    key = f"r{MODULE_RESOLUTIONS['~250']}"
    with np.load(os.path.join(consensus_dir, "reference.npz")) as stored:
        r250 = stored[key] if key in stored else None
    ix = {g: i for i, g in enumerate(rows.gene)}
    idx = np.array([ix[g] for g in sorted(genes) if g in ix])
    if len(idx) < 3:
        return {"present": int(len(idx))}
    targeting = ~rows.gene.str.startswith("NTpseudo").to_numpy()
    dec = np.full(len(rows), -1)
    dec[targeting] = pd.qcut(rows.n_cells[targeting].rank(method="first"), 10, labels=False).to_numpy()
    observed = _pair_comembership(labels, idx)
    null = np.array([_pair_comembership(labels, np.array(
        [rng.choice(np.flatnonzero((dec == dec[i]) & targeting)) for i in idx])) for _ in range(int(n_null))])
    out = {"present": int(len(idx)), "comembership": float(observed), "null_mean": float(null.mean()),
           "null_p99": float(np.quantile(null, 0.99)), "percentile": float((null < observed).mean()),
           "recovered (above null p99)": bool(observed > np.quantile(null, 0.99))}
    if r250 is not None:
        out["largest_fraction_in_one_~250_cluster"] = float(
            pd.Series(r250[idx]).value_counts().iloc[0] / len(idx))
    return out


def _external_clusters(spec: Mapping[str, Any]) -> pd.DataFrame:
    """One external cluster table: ``table`` (and ``sheet``), its ``gene`` and ``cluster`` columns."""
    table = _read_any(spec["table"], spec.get("sheet"))
    gene, cluster = spec.get("gene", "gene_symbol"), spec.get("cluster", "cluster_id")
    for column in (gene, cluster):
        if column not in table:
            raise ValueError(f"{spec['table']} has no column {column!r}")
    return table.dropna(subset=[cluster])


def _arm_assignments(path: str, target: int) -> Tuple[Dict[str, Any], int]:
    """``{symbol: cluster}`` of an arm's targeting genes at the resolution nearest ``target`` clusters."""
    from .tabular import read_table

    table = read_table(str(path), canonicalise=False, report=None)
    counts = {c[len("leiden_r"):]: int(table[c].nunique()) for c in table.columns
              if str(c).startswith("leiden_r")}
    if not counts:
        raise ValueError(f"{path} has no leiden_r<resolution> column")
    r = min(counts, key=lambda k: abs(counts[k] - target))
    table = table[~_controls(table.gene)]
    return dict(zip(_symbols(table.gene), table[f"leiden_r{r}"])), int(float(r))


def mitocarta_benchmark(arms: Sequence[Mapping[str, Any]], mitocarta: Set[str],
                        subcompartments: Mapping[str, Set[str]], external: Mapping[str, Any], *,
                        seed: int = 0) -> Dict[str, pd.DataFrame]:
    """MitoCarta and external-module benchmark of several clusterings.

    :param arms: one mapping per arm, in report order: ``label``, and any of
        ``assignments`` (a cluster table with ``gene`` and
        ``leiden_r<resolution>`` columns, :func:`benchmark_clusters`),
        ``profiles`` (an ``.npz`` with ``names`` and ``pcs`` or a similarity
        ``S``), ``atlas`` (a consensus folder of the modules phase) and
        ``own_genes_only`` (score this arm on its own genes only, as for an
        arm restricted to active genes).
    :param mitocarta: the MitoCarta genes.
    :param subcompartments: ``{label: genes}`` of :func:`read_mitocarta`.
    :param external: the external modules, as read from
        ``ops_eval_external_modules``: ``modules`` (``name``, ``table``,
        ``sheet``, ``gene``, ``cluster``, ``clusters``: the external
        sub-modules), ``published`` (a list of published clusterings, each
        ``label``, ``table``, ``sheet``, ``gene``, ``cluster`` and
        ``score_modules``), ``named_clusters`` (a list, each ``key``,
        ``title``, ``short``, ``table``, ``sheet``, ``gene``, ``cluster``,
        ``id``, ``members_flag``, ``members_name``, ``partners``,
        ``partners_name``, ``probe``) and ``target_clusters`` (250).
    :param seed: the seed of every null draw, shared in order.
    :returns: ``{"A", "B", "C", "D_<key>", "comparison"}``: MitoCarta
        recovery, sub-compartment retrieval, the external sub-modules and
        each named cluster, then one headline row per clustering.
    """
    rng = np.random.default_rng(seed)
    target = int(external.get("target_clusters", 250))
    tag = f"~{target}"
    modules = external["modules"]
    module_table = _external_clusters(modules)
    module_genes = {int(c): set(g[modules.get("gene", "gene_symbol")])
                    for c, g in module_table.groupby(modules.get("cluster", "cluster_id"))}
    module_ids = [int(c) for c in modules.get("clusters", sorted(module_genes))]
    module_name = str(modules.get("name", "external"))
    sets: Dict[str, Dict[str, Any]] = {}
    scored_modules = []
    for published in external.get("published", []):
        table = _external_clusters(published)
        g, c = published.get("gene", "gene_symbol"), published.get("cluster", "cluster_id")
        sets[published["label"]] = dict(zip(table[g], table[c].astype(int)))
        if published.get("score_modules", True):
            scored_modules.append(published["label"])
    resolution: Dict[str, int] = {}
    own_only = set()
    for arm in arms:
        if arm.get("assignments"):
            sets[arm["label"]], resolution[arm["label"]] = _arm_assignments(arm["assignments"], target)
        if arm.get("own_genes_only"):
            own_only.add(arm["label"])
    shared = [k for k in sets if k not in own_only]
    common = set.intersection(*[set(sets[k]) for k in shared]) if shared else set()
    rows_a = []
    for label, assignments in sets.items():
        rows_a.append({"set": label, "scope": "own genes", "leiden_resolution": resolution.get(label),
                       **mitocarta_enrichment(assignments, mitocarta)})
        if label in shared:
            rows_a.append({"set": label, "scope": f"common genes ({len(common)})",
                           "leiden_resolution": resolution.get(label),
                           **mitocarta_enrichment({g: c for g, c in assignments.items() if g in common},
                                                  mitocarta)})
    rows_b = []
    for arm in arms:
        if arm.get("profiles"):
            names, sims = _profile_cosines(arm["profiles"])
            rows_b += subcompartment_retrieval(arm["label"], names, sims, mitocarta, subcompartments)
    column = f"{module_name.lower()}_cluster"
    largest = f"largest_fraction_in_one_arm_cluster_{tag}"
    rows_c = []
    for arm in arms:
        assignments = sets.get(arm["label"])
        for c in module_ids:
            genes = module_genes[c]
            row = {"arm": arm["label"], column: c, "module_genes": len(genes)}
            if assignments:
                present = [x for x in sorted(genes) if x in assignments]
                row["present_in_arm"] = len(present)
                row[largest] = float(pd.Series([assignments[x] for x in present]).value_counts().iloc[0]
                                     / len(present)) if present else np.nan
            if arm.get("atlas"):
                coherence = module_coherence(str(arm["atlas"]), genes, rng)
                if coherence:
                    row.update({f"atlas {k}": v for k, v in coherence.items()})
            rows_c.append(row)
    for label in scored_modules:
        assignments = sets[label]
        for c in module_ids:
            present = [x for x in sorted(module_genes[c]) if x in assignments]
            rows_c.append({"arm": label, column: c, "module_genes": len(module_genes[c]),
                           "present_in_arm": len(present),
                           largest: float(pd.Series([assignments[x] for x in present]).value_counts().iloc[0]
                                          / len(present)) if present else np.nan})
    out = {"A": pd.DataFrame(rows_a), "B": pd.DataFrame(rows_b), "C": pd.DataFrame(rows_c)}
    for named in external.get("named_clusters", []):
        out[f"D_{named['key']}"] = _named_cluster_rows(named, arms, sets, rng, tag)
    out["comparison"] = _mito_comparison(sets, out["A"], out["B"], out["C"], module_name,
                                         len(module_ids), largest)
    return out


def _named_cluster_rows(named: Mapping[str, Any], arms: Sequence[Mapping[str, Any]],
                        sets: Mapping[str, Mapping[str, Any]], rng: np.random.Generator,
                        tag: str) -> pd.DataFrame:
    """One row per arm: where a named external cluster's members and its partner genes land.

    :param named: the named cluster (see :func:`mitocarta_benchmark`).
    :param arms: the arms.
    :param sets: ``{label: {gene: cluster}}`` of every clustering.
    :param rng: the generator of the random-gene null (advanced).
    :param tag: the cluster-count tag of the column names, ``~250``.
    :returns: the table.
    """
    table = _external_clusters(named)
    gene = named.get("gene", "gene_symbol")
    cluster = table[table[named.get("cluster", "cluster_id")] == named["id"]]
    flag = named.get("members_flag")
    members = set(cluster[gene][cluster[flag].astype(bool)]) if flag else set(cluster[gene])
    every = set(cluster[gene])
    partners = set(named["partners"])
    probe = named.get("probe")
    key, short, title = named["key"], str(named.get("short", named["id"])), named.get("title", named["key"])
    m_name, p_name = named.get("members_name", "members"), named.get("partners_name", "partners")
    rows = []
    for arm in arms:
        assignments = sets.get(arm["label"])
        row = {"arm": arm["label"], f"{key}_{m_name.lower()}_members": len(members)}
        if assignments:
            clusters = {assignments[g] for g in partners if g in assignments}
            row[f"{p_name} clusters ({tag})"] = len(clusters)
            row[f"{short} {m_name} members sharing a cluster with the {p_name}"] = int(
                sum(assignments.get(g) in clusters for g in members if g in assignments))
            if probe and probe in assignments:
                same = {g for g, c in assignments.items() if c == assignments[probe]}
                row[f"{probe} cluster size"] = len(same)
                row[f"{probe} cluster members from {title}"] = ", ".join(sorted(same & every))
                row[f"{probe} cluster contains {p_name}"] = ", ".join(sorted(same & partners))
        if arm.get("profiles"):
            names, sims = _profile_cosines(arm["profiles"])
            sym = _symbols(names)
            ia = np.flatnonzero(np.isin(sym, list(partners)))
            io = np.flatnonzero(np.isin(sym, list(members)))
            if len(ia) and len(io):
                observed = float(sims[np.ix_(ia, io)].mean())
                targeting = np.flatnonzero(~_controls(names))
                null = [float(sims[np.ix_(rng.choice(targeting, len(ia), replace=False), io)].mean())
                        for _ in range(500)]
                row.update({f"cosine {p_name} vs {short} {m_name}": observed,
                            f"random genes vs {short} {m_name} (mean)": float(np.mean(null)),
                            "random p99": float(np.quantile(null, 0.99))})
        rows.append(row)
    return pd.DataFrame(rows)


def _mito_comparison(sets, table_a, table_b, table_c, module_name, n_modules, largest) -> pd.DataFrame:
    """The headline row of every clustering: recovery, sub-compartment mAP and sub-module coherence.

    :param sets: ``{label: {gene: cluster}}``, in report order.
    :param table_a: the MitoCarta recovery table.
    :param table_b: the sub-compartment retrieval table.
    :param table_c: the sub-module table.
    :param module_name: the external modules' name.
    :param n_modules: how many sub-modules were scored.
    :param largest: the largest-fraction column of ``table_c``.
    :returns: one row per clustering.
    """
    common = table_a[table_a.scope.str.startswith("common")].set_index("set") if len(table_a) else pd.DataFrame()
    pivot = (table_b.pivot_table(index="arm", columns="sub_compartment", values="mAP")
             if len(table_b) else pd.DataFrame())
    coherent = "atlas recovered (above null p99)"
    rows = []
    for label in sets:
        row: Dict[str, Any] = {"set": label}
        if label in common.index:
            row.update({"MitoCarta genes in mito-enriched clusters (common genes)":
                        common.loc[label, "mito_genes_in_enriched"],
                        "purity": common.loc[label, "purity_in_enriched"],
                        "mito-enriched clusters": common.loc[label, "mito_enriched_clusters"]})
        if label in pivot.index:
            row.update({f"sub-compartment mAP: {k}": v for k, v in pivot.loc[label].items()})
        c = table_c[table_c.arm == label] if len(table_c) else table_c
        if coherent in c:
            row[f"{module_name} sub-modules coherent in atlas (of {n_modules})"] = int(
                c[coherent].eq(True).sum())
        if largest in c:
            row["sub-modules: mean largest fraction in one cluster"] = float(c[largest].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def run_mitocarta_benchmark(settings: Mapping[str, Any],
                            report: Optional[Any] = None) -> Dict[str, Any]:
    """The OPS ``mitocarta`` phase: :func:`mitocarta_benchmark` from settings.

    Writes ``mito_A_mitocarta_recovery.csv``,
    ``mito_B_subcompartment_retrieval.csv``, ``mito_C_submodules.csv``, one
    ``mito_D_<key>.csv`` per named cluster and ``mito_comparison.csv`` to
    ``<dst_root>/ops_eval/mitocarta``.

    :param settings: read here: ``ops_eval_mitocarta`` (the MitoCarta 3.0
        table), ``ops_eval_external_modules`` (a JSON file with the external
        modules, published clusterings and named clusters, see
        :func:`mitocarta_benchmark`), ``ops_eval_arms`` (the arms, a list of
        mappings or a JSON file holding one) and ``dst_root``.
    :param report: called with one line when done.
    :returns: ``{"folder", "outputs", "tables"}``.
    :raises ValueError: when a setting is missing.
    """
    import json

    from .tabular import write_table

    for key in ("ops_eval_mitocarta", "ops_eval_external_modules", "ops_eval_arms", "dst_root"):
        if not settings.get(key):
            raise ValueError(f"the mitocarta phase needs {key}")
    with open(str(settings["ops_eval_external_modules"]), encoding="utf-8") as handle:
        external = json.load(handle)
    arms = settings["ops_eval_arms"]
    if isinstance(arms, str):
        with open(arms, encoding="utf-8") as handle:
            arms = json.load(handle)
    mitocarta, subcompartments = read_mitocarta(str(settings["ops_eval_mitocarta"]))
    tables = mitocarta_benchmark(arms, mitocarta, subcompartments, external)
    folder = os.path.join(str(settings["dst_root"]), "ops_eval", "mitocarta")
    os.makedirs(folder, exist_ok=True)
    names = {"A": "mito_A_mitocarta_recovery.csv", "B": "mito_B_subcompartment_retrieval.csv",
             "C": "mito_C_submodules.csv", "comparison": "mito_comparison.csv"}
    outputs = {}
    for key, table in tables.items():
        name = names.get(key, f"mito_{key}.csv")
        outputs[key] = write_table(table, os.path.join(folder, name), canonicalise=False)
    if report:
        report(f"mitocarta: {len(tables['comparison'])} clusterings scored")
    return {"folder": folder, "outputs": outputs, "tables": tables}


#: The Leiden resolution grids of the resolution sweep (item 916), per graph.
SWEEP_RESOLUTIONS: Dict[str, Tuple[float, ...]] = {
    "phate": (1, 2, 3, 5, 7, 8, 10, 12, 13, 15, 18, 22, 26, 32, 40),
    "knn": (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 13, 16),
}

#: The PHATE and Leiden seeds of the resolution sweep.
SWEEP_SEEDS: Tuple[int, ...] = (42, 1, 2)

#: The seed of the sweep's shuffled-label controls, drawn in loop order.
SWEEP_CONTROL_SEED = 20261109

#: The metrics the sweep summary averages over seeds.
SWEEP_METRICS: Tuple[str, ...] = (
    "clusters", "STRING_precision", "STRING_recall", "STRING_f1",
    "CORUM_pct_enriched_may2025", "KEGG_pct_enriched_may2025",
    "complexes_recovered_pct", "CORUM_partner_in_cluster_pct")


def gene_graph(rows: np.ndarray, kind: str = "phate", k: int = 12, seed: int = 42, *,
               knn: int = 10, n_jobs: int = 2):
    """The gene graph a Leiden sweep clusters.

    :param rows: one profile row per gene (controls included).
    :param kind: ``'phate'``, PHATE's diffusion operator (``knn`` cosine
        neighbours) made symmetric, ``(P + P.T) / 2``; or ``'knn'``, the
        union of every row's ``k`` nearest rows by cosine, each edge weighted
        by its cosine clipped at 1e-6.
    :param k: the neighbours of the ``'knn'`` graph.
    :param seed: PHATE's seed; the ``'knn'`` graph has none.
    :param knn: PHATE's neighbours.
    :param n_jobs: PHATE's worker count.
    :returns: an undirected, weighted ``igraph.Graph``, one vertex per row.
    :raises ImportError: when the ``clustering`` extra is not installed.
    :raises ValueError: for an unknown ``kind``.
    """
    if kind not in ("phate", "knn"):
        raise ValueError(f"the gene graph must be 'phate' or 'knn'; got {kind!r}")
    igraph = _optional("igraph")
    rows = np.asarray(rows, float)
    if kind == "phate":
        phate = _optional("phate")
        model = phate.PHATE(random_state=seed, n_jobs=n_jobs, knn=knn, knn_dist="cosine",
                            verbose=False)
        model.fit(rows)
        w = model.graph.diff_op
        w = (w + w.T) / 2
        if hasattr(w, "tocoo"):
            w = w.tocoo()
        else:
            from scipy.sparse import coo_matrix
            w = coo_matrix(w)
        upper = w.row < w.col
        graph = igraph.Graph(n=rows.shape[0], edges=list(zip(w.row[upper], w.col[upper])),
                             directed=False)
        graph.es["weight"] = w.data[upper].tolist()
        return graph
    unit = rows / np.clip(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12, None)
    similarity = unit @ unit.T
    np.fill_diagonal(similarity, -np.inf)
    nearest = np.argpartition(-similarity, k, axis=1)[:, :k]
    edges: Dict[Tuple[int, int], float] = {}
    for i in range(len(unit)):
        for j in nearest[i]:
            a, b = (i, int(j)) if i < j else (int(j), i)
            edges[(a, b)] = max(float(similarity[a, b]), 1e-6)
    graph = igraph.Graph(n=len(unit), edges=list(edges), directed=False)
    graph.es["weight"] = list(edges.values())
    return graph


def leiden_partitions(graph, resolutions: Iterable[float], seeds: Iterable[int]
                      ) -> Dict[Tuple[int, float], np.ndarray]:
    """Leiden partitions of one graph for every seed and resolution.

    :param graph: from :func:`gene_graph`.
    :param resolutions: Leiden resolution parameters.
    :param seeds: Leiden seeds.
    :returns: ``{(seed, resolution): labels}`` in seed-then-resolution order;
        each partition optimises the RB configuration model with
        ``n_iterations=-1`` (until it stops improving).
    :raises ImportError: when the ``clustering`` extra is not installed.
    """
    leidenalg = _optional("leidenalg")
    out: Dict[Tuple[int, float], np.ndarray] = {}
    for seed in seeds:
        for resolution in resolutions:
            partition = leidenalg.find_partition(
                graph, leidenalg.RBConfigurationVertexPartition, weights="weight",
                n_iterations=-1, seed=int(seed), resolution_parameter=float(resolution))
            out[(int(seed), resolution)] = np.array(partition.membership)
    return out


def complex_metrics(genes: Sequence[str], labels: np.ndarray,
                    groups: Mapping[str, Sequence[str]], halves: Mapping[str, int]
                    ) -> Dict[str, Any]:
    """Complex recovery and partner-in-cluster share of one clustering.

    :param genes: the scored genes.
    :param labels: one cluster label per gene.
    :param groups: ``{complex: members}`` (CORUM).
    :param halves: ``{complex: 0 or 1}``; complexes of half 1 (REPORT) are
        scored for recovery.
    :returns: ``REPORT_complexes`` (REPORT complexes with at least three
        scored members), ``complexes_recovered_pct`` (those with at least
        half their scored members in one cluster) and
        ``CORUM_partner_in_cluster_pct`` (genes with a CORUM partner among
        the scored genes, of any half, that share its cluster).
    """
    labels = np.asarray(labels)
    index = {g: i for i, g in enumerate(genes)}
    recovered: List[bool] = []
    has = np.zeros(len(index), bool)
    hit = np.zeros(len(index), bool)
    for name, members in groups.items():
        idx = np.array([index[x] for x in members if x in index], dtype=int)
        if len(idx) >= 2:
            has[idx] = True
            values, counts = np.unique(labels[idx], return_counts=True)
            hit[idx[np.isin(labels[idx], values[counts >= 2])]] = True
        if len(idx) >= 3 and halves.get(name) == 1:
            recovered.append(np.unique(labels[idx], return_counts=True)[1].max() / len(idx) >= 0.5)
    return {"REPORT_complexes": len(recovered),
            "complexes_recovered_pct": 100 * float(np.mean(recovered)) if recovered else float("nan"),
            "CORUM_partner_in_cluster_pct": 100 * float(hit[has].mean()) if has.any() else float("nan")}


def sweep_references(reference_dir: Any) -> Dict[str, Any]:
    """The reference tables a resolution sweep scores against.

    :param reference_dir: the folder (or folders) holding Brieflow's
        ``corum_group_benchmark.tsv``, ``kegg_group_benchmark.tsv`` and
        ``string_pair_benchmark.tsv``.
    :returns: ``corum`` and ``kegg`` (group tables), ``positives`` (the
        STRING pairs), ``groups`` (CORUM complexes) and ``halves`` (each
        complex's half, :func:`spacr.ops_reference.group_halves`).
    """
    from . import ops_reference

    def table(name: str) -> pd.DataFrame:
        """One benchmark table by its reference name."""
        return read_benchmark(ops_reference.reference_file(name, reference_dir))

    groups = ops_reference.reference_groups("CORUM", reference_dir)
    return {"corum": table("CORUM"), "kegg": table("KEGG"),
            "positives": string_positives(table("STRING_brieflow")),
            "groups": groups, "halves": ops_reference.group_halves(groups)}


def score_partition(name: str, genes: Sequence[str], labels: np.ndarray,
                    refs: Mapping[str, Any]) -> Dict[str, Any]:
    """One sweep row: :func:`score_clusters` plus :func:`complex_metrics`.

    :param name: the row's label.
    :param genes: the scored genes.
    :param labels: their cluster labels.
    :param refs: from :func:`sweep_references`.
    :returns: the row; ``STRING_f1`` is computed from the rounded precision
        and recall, as the sweep reports them.
    """
    row = score_clusters(name, genes, np.asarray(labels), corum=refs["corum"], kegg=refs["kegg"],
                         positives=refs["positives"])
    p, r = row["STRING_precision"], row["STRING_recall"]
    row["STRING_f1"] = 2 * p * r / (p + r) if p + r else 0.0
    row.update(complex_metrics(list(genes), labels, refs["groups"], refs["halves"]))
    return row


def score_partitions(names: Sequence[str], partitions: Mapping[Tuple[str, int, float], np.ndarray],
                     refs: Mapping[str, Any], shuffles: int = 1, seed: int = SWEEP_CONTROL_SEED, *,
                     genes: Optional[Sequence[str]] = None, label: str = "spaCR",
                     report=None) -> pd.DataFrame:
    """Score every partition and its shuffled-label controls.

    :param names: the clustered row names, in the partitions' row order.
    :param partitions: ``{(graph, seed, resolution): labels}``, scored in
        this order.
    :param refs: from :func:`sweep_references`.
    :param shuffles: shuffled-label controls per partition: the scored
        genes' labels permuted, cluster sizes kept.
    :param seed: the controls' seed; one generator draws them in loop order.
    :param genes: the genes scored; every non-control row when None.
    :param label: the ``profile`` of every row.
    :param report: called with one progress line per partition.
    :returns: one row per partition (``control`` ``'real'``) and per
        control (``'shuffled gene labels'``), with ``graph``, ``seed`` and
        ``resolution``.
    :raises KeyError: when a gene of ``genes`` is not a row.
    """
    names = np.asarray(names).astype(str)
    if genes is None:
        genes = names[~_is_control(names, "nontargeting")]
    genes = np.asarray(genes).astype(str)
    position = pd.Series(np.arange(len(names)), index=names).loc[genes].to_numpy()
    rng = np.random.default_rng(seed)
    rows = []
    for (graph, run_seed, resolution), labels in partitions.items():
        scored = np.asarray(labels)[position]
        key = {"graph": graph, "seed": int(run_seed), "resolution": float(resolution)}
        real = dict(score_partition(label, genes, scored, refs), **key, control="real")
        rows.append(real)
        for _ in range(int(shuffles)):
            rows.append(dict(score_partition(label, genes, scored[rng.permutation(len(scored))], refs),
                             **key, control="shuffled gene labels"))
        if report is not None:
            report(f"{graph} seed {run_seed} r {resolution}: {real['clusters']} clusters, "
                   f"CORUM {real['CORUM_pct_enriched_may2025']} %, "
                   f"STRING F1 {round(real['STRING_f1'], 4)}")
    return pd.DataFrame(rows)


def sweep_summary(rows: pd.DataFrame, target: int = 227) -> pd.DataFrame:
    """Mean, minimum and maximum of every sweep metric over the seeds.

    :param rows: from :func:`score_partitions`.
    :param target: the cluster count to flag.
    :returns: one row per graph, resolution and control, with
        ``<metric> mean|min|max`` columns and ``nearest_<target>`` True on
        each graph's real resolution whose mean cluster count is nearest
        ``target`` (the first on a tie).
    """
    summary = rows.groupby(["graph", "resolution", "control"])[list(SWEEP_METRICS)].agg(
        ["mean", "min", "max"])
    summary.columns = [f"{a} {b}" for a, b in summary.columns]
    summary = summary.reset_index()
    flag = f"nearest_{int(target)}"
    summary[flag] = np.nan
    summary[flag] = summary[flag].astype(object)
    for graph in summary["graph"].unique():
        real = summary[(summary["graph"] == graph) & (summary["control"] == "real")]
        if len(real):
            near = real.iloc[(real["clusters mean"] - target).abs().argsort()[:1]]["resolution"].iloc[0]
            summary.loc[(summary["graph"] == graph) & (summary["resolution"] == near), flag] = True
    return summary


#: The seed of the published comparators' shuffled-label controls, drawn
#: in comparator order.
PUBLISHED_CONTROL_SEED = 20261110


def published_clusters(spec: Mapping[str, Any], control_prefix: str = "nontargeting") -> Dict[str, int]:
    """One published clustering as ``{gene: cluster}``.

    :param spec: ``table`` (CSV, TSV, Excel or Parquet), optional ``sheet``
        (an Excel sheet), ``gene`` (default ``gene_symbol``) and
        ``cluster`` (default ``cluster_id``); Brieflow's supplementary
        workbook is ``sheet`` ``'Interphase Genes'``, and Funk et al.'s
        table has ``cluster`` ``'interphase_cluster'``.
    :param control_prefix: rows whose gene starts with it are left out.
    :returns: every gene with a cluster; a gene listed twice keeps its last
        cluster.
    :raises ValueError: when a column is missing.
    """
    table = _external_clusters(spec)
    gene, cluster = spec.get("gene", "gene_symbol"), spec.get("cluster", "cluster_id")
    table = table[[gene, cluster]].dropna()
    table = table[~table[gene].astype(str).str.lower().str.startswith(control_prefix.lower())]
    return dict(zip(table[gene], table[cluster].astype(int)))


def score_published(published: Mapping[str, Mapping[str, int]], genes: Sequence[str],
                    refs: Mapping[str, Any], shuffles: int = 1,
                    seed: int = PUBLISHED_CONTROL_SEED) -> pd.DataFrame:
    """Score published clusterings exactly as the sweep scores its partitions.

    :param published: ``{name: {gene: cluster}}``.
    :param genes: the sweep's scored genes; each comparator is scored on
        those it clusters, in this order.
    :param refs: from :func:`sweep_references`.
    :param shuffles: shuffled-label controls per comparator, cluster sizes
        kept, as the sweep's.
    :param seed: the controls' seed; one generator draws them in order.
    :returns: one ``'real'`` row per comparator and its ``'shuffled gene
        labels'`` rows, with the sweep's columns, ``graph`` ``'published'``
        and no seed or resolution.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for name, clusters in published.items():
        scored = np.array([g for g in genes if g in clusters])
        labels = np.array([clusters[g] for g in scored])
        key = {"graph": "published", "seed": np.nan, "resolution": np.nan}
        rows.append(dict(score_partition(name, scored, labels, refs), **key, control="real"))
        for _ in range(int(shuffles)):
            rows.append(dict(score_partition(name, scored, labels[rng.permutation(len(labels))], refs),
                             **key, control="shuffled gene labels"))
    return pd.DataFrame(rows)


def published_record(rows: pd.DataFrame, published: Sequence[Mapping[str, Any]], genes: Sequence[str],
                     summary: pd.DataFrame, target: int, shuffles: int, seed: int) -> Dict[str, Any]:
    """What the published comparator rows match and what they do not.

    :param rows: from :func:`score_published`.
    :param published: the comparator specs (``ops_cluster_published``).
    :param genes: the sweep's scored genes.
    :param summary: from :func:`sweep_summary`, for the matched cluster count.
    :param target: the cluster count the summary flags.
    :param shuffles: shuffled-label controls per comparator.
    :param seed: their seed.
    :returns: the legend and, per comparator, its source, genes scored and
        missing, clusters, and the sweep resolution nearest its cluster
        count on each graph.
    """
    real = summary[summary["control"] == "real"]
    out = []
    for spec, row in zip(published, rows[rows["control"] == "real"].to_dict("records")):
        nearest = {}
        for graph, d in real.groupby("graph"):
            pick = d.iloc[(d["clusters mean"] - row["clusters"]).abs().argsort()[:1]]
            nearest[str(graph)] = {"resolution": float(pick["resolution"].iloc[0]),
                                   "clusters_mean": float(pick["clusters mean"].iloc[0])}
        out.append({"name": row["profile"], "table": str(spec.get("table")),
                    "sheet": spec.get("sheet"), "clusters": int(row["clusters"]),
                    "genes_scored": int(row["genes"]), "genes_missing": int(len(genes) - row["genes"]),
                    "nearest_sweep_resolution": nearest})
    return {
        "legend": (
            "Published comparator rows (graph 'published', data_matched False) score the authors' own "
            "published gene clusters. Matched: the same scored gene list (genes the publication did not "
            "cluster are missing, counted per row), the same CORUM, KEGG and STRING files and the same "
            "scoring code as every sweep row, and the same control (gene labels shuffled with cluster "
            f"sizes kept, {int(shuffles)} per comparator, numpy default_rng({int(seed)})). Not matched: "
            "the data and the clustering: the published partitions come from the authors' own profiles "
            "and pipeline, not from this run's cells, and their cluster counts are the authors' choice. "
            "Compare a comparator with the sweep at a matched cluster count (nearest_sweep_resolution, "
            f"or the rows flagged nearest_{int(target)})."),
        "comparators": out}


def _sweep_figure(summary: pd.DataFrame, path: str, published: Optional[pd.DataFrame] = None) -> str:
    """Each sweep metric against the cluster count, real and shuffled.

    :param summary: from :func:`sweep_summary`.
    :param path: the PNG written.
    :param published: rows of :func:`score_published`, drawn as points at
        their own cluster counts, real and shuffled.
    :returns: ``path``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (("CORUM_pct_enriched_may2025", "CORUM clusters enriched (%)"),
              ("KEGG_pct_enriched_may2025", "KEGG clusters enriched (%)"),
              ("STRING_f1", "STRING pair F1"),
              ("complexes_recovered_pct", "CORUM complexes in one cluster (%)"))
    colours = {"real": "#1f6f8b", "shuffled gene labels": "#9aa5b1"}
    figure, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.4))
    for ax, (metric, title) in zip(axes, panels):
        for (graph, control), d in summary.groupby(["graph", "control"]):
            d = d.sort_values("clusters mean")
            ax.plot(d["clusters mean"], d[f"{metric} mean"], marker="o", ms=3,
                    color=colours.get(control, "#333333"),
                    linestyle="-" if graph == "phate" else "--",
                    label=f"{graph}, {control}")
        if published is not None:
            for k, ((name, control), d) in enumerate(published.groupby(["profile", "control"], sort=False)):
                real = control == "real"
                ax.scatter(d["clusters"], d[metric], s=28 if real else 14,
                           marker=("s", "^", "D", "v")[k // 2 % 4],
                           color="#0b2545" if real else "#9aa5b1", alpha=0.6, zorder=3,
                           label=f"{name}, {control} (published partition, data not matched)")
        ax.set_xscale("log")
        ax.set_xlabel("clusters (mean over seeds)")
        ax.set_title(title, fontsize=9)
    axes[-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def _sweep_rows_from_table(table: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Row names and values of a ready profile table: ``gene`` plus numbers."""
    columns = [c for c in table.columns if c not in ("gene", "n_cells")
               and pd.api.types.is_numeric_dtype(table[c])]
    return table["gene"].astype(str).to_numpy(), table[columns].to_numpy(float)


def _sweep_setting_list(value: Any, default: Sequence[Any], cast) -> List[Any]:
    """A list setting given as a list or a comma-separated string."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return list(default)
    items = value.split(",") if isinstance(value, str) else list(value)
    return [cast(item) for item in items]


def _published_specs(value: Any) -> List[Dict[str, Any]]:
    """The ``ops_cluster_published`` comparators: a list of specs, or the path of a JSON list.

    :raises ValueError: when a spec has no ``name`` or ``table`` or two share a name.
    """
    import json

    if not value:
        return []
    if isinstance(value, str):
        with open(value, encoding="utf-8") as handle:
            value = json.load(handle)
    specs = [dict(spec) for spec in value]
    names = [str(spec.get("name") or "") for spec in specs]
    if any(not name for name in names) or len(set(names)) < len(names) or not all(
            spec.get("table") for spec in specs):
        raise ValueError("every ops_cluster_published entry needs a table and a name of its own")
    return specs


def cluster_sweep_phase(settings: Mapping[str, Any], genes: Optional[pd.DataFrame], folder: str,
                        report=None) -> Dict[str, Any]:
    """The ``cluster_sweep`` phase of the OPS profile path (item 916).

    :param settings: read for ``ops_cluster_rows`` (a table of ready
        profile rows, ``gene`` plus numeric columns used as given; the
        profile phase's gene rows, embedded as the evaluate phase embeds
        them, when empty), ``ops_cluster_graph`` (``'phate'``, ``'knn'`` or
        both), ``ops_cluster_knn``, ``ops_cluster_sweep_resolutions``
        (``{graph: grid}``; :data:`SWEEP_RESOLUTIONS` for a graph not
        named), ``ops_cluster_seeds``, ``ops_cluster_target_clusters``,
        ``ops_cluster_shuffles``, ``ops_cluster_reference_dir`` (falling
        back to ``ops_reference_dir``), ``ops_cluster_gene_list`` (a table
        with a ``gene`` column; every targeting row when empty),
        ``ops_profile_min_cells``, ``ops_control_prefix`` and
        ``ops_cluster_published`` (published clusterings scored as
        comparators, :func:`published_clusters` specs each with a
        ``name``, or the path of a JSON list of them).
    :param genes: the profile phase's gene table, or None.
    :param folder: ``dst_root``; outputs go to its
        ``results/ops_clusters``.
    :param report: called with progress lines.
    :returns: ``{"folder", "outputs", "rows", "graphs"}``.
    :raises ValueError: when no rows or no reference folder are given, a
        graph is unknown, or a published comparator has no name of its own.
    :raises ImportError: when the ``clustering`` extra is not installed.
    """
    import json
    import os

    from . import ops_reference
    from .tabular import read_table, write_table

    say = report or (lambda text: None)
    prefix = str(settings.get("ops_control_prefix") or "nontargeting")
    graphs = _sweep_setting_list(settings.get("ops_cluster_graph"), ("phate", "knn"),
                                 lambda v: str(v).strip().lower())
    unknown = [g for g in graphs if g not in SWEEP_RESOLUTIONS]
    if unknown:
        raise ValueError(f"ops_cluster_graph must name phate and/or knn; got {unknown}")
    reference_dir = settings.get("ops_cluster_reference_dir") or settings.get("ops_reference_dir")
    if not reference_dir:
        raise ValueError("cluster_sweep needs ops_cluster_reference_dir (or ops_reference_dir), the folder "
                         "holding Brieflow's CORUM, KEGG and STRING benchmark tables")
    source = settings.get("ops_cluster_rows")
    if source:
        names, rows = _sweep_rows_from_table(read_table(str(source), canonicalise=False, report=None))
    elif genes is not None:
        kept = genes[genes["n_cells"] > int(settings.get("ops_profile_min_cells", 10))]
        columns = [c for c in kept.columns if c.startswith(("PC_", "w_"))]
        names = kept["gene"].astype(str).to_numpy()
        rows = ops_reference.embed_gene_profiles(kept[columns].to_numpy(float))
    else:
        raise ValueError("cluster_sweep needs gene rows: run the profile phase first or set ops_cluster_rows")
    k = int(settings.get("ops_cluster_knn") or 12)
    grids = dict(settings.get("ops_cluster_sweep_resolutions") or {})
    seeds = _sweep_setting_list(settings.get("ops_cluster_seeds"), SWEEP_SEEDS, int)
    target = int(settings.get("ops_cluster_target_clusters") or 227)
    shuffles = int(1 if settings.get("ops_cluster_shuffles") is None else settings["ops_cluster_shuffles"])
    gene_list = settings.get("ops_cluster_gene_list")
    scored = (read_table(str(gene_list), canonicalise=False, report=None)["gene"].astype(str).to_numpy()
              if gene_list else names[~_is_control(names, prefix)])
    refs = sweep_references(reference_dir)
    out_dir = os.path.join(folder, "results", "ops_clusters")
    os.makedirs(out_dir, exist_ok=True)
    outputs: Dict[str, str] = {}
    partitions: Dict[Tuple[str, int, float], np.ndarray] = {}
    tags = []
    for kind in graphs:
        tag = "phate" if kind == "phate" else f"knn{k}"
        tags.append(tag)
        grid = list(grids.get(kind) or grids.get(tag) or SWEEP_RESOLUTIONS[kind])
        for seed in seeds:
            graph = gene_graph(rows, kind, k=k, seed=seed)
            assignments = pd.DataFrame({"gene": names})
            for (run_seed, resolution), labels in leiden_partitions(graph, grid, [seed]).items():
                partitions[(tag, run_seed, resolution)] = labels
                assignments[f"leiden_r{resolution}"] = labels
            outputs[f"assignments_{tag}_s{seed}"] = write_table(
                assignments, os.path.join(out_dir, f"assignments_{tag}_s{seed}.csv.gz"), canonicalise=False)
            say(f"cluster_sweep: {tag} seed {seed}: {len(grid)} resolutions")
    table = score_partitions(names, partitions, refs, shuffles, genes=scored, report=say)
    summary = sweep_summary(table, target)
    outputs["sweep_rows"] = write_table(table, os.path.join(out_dir, "sweep_rows.csv"), canonicalise=False)
    outputs["sweep_summary"] = write_table(summary, os.path.join(out_dir, "sweep_summary.csv"),
                                           canonicalise=False)
    specs = _published_specs(settings.get("ops_cluster_published"))
    published_rows = None
    if specs:
        published = {str(spec["name"]): published_clusters(spec, prefix) for spec in specs}
        published_rows = score_published(published, scored, refs, shuffles)
        combined = pd.concat([table.assign(data_matched=True), published_rows.assign(data_matched=False)],
                             ignore_index=True)
        outputs["sweep_rows_with_published"] = write_table(
            combined, os.path.join(out_dir, "sweep_rows_with_published.csv"), canonicalise=False)
        record = published_record(published_rows, specs, scored, summary, target, shuffles,
                                  PUBLISHED_CONTROL_SEED)
        path = os.path.join(out_dir, "published_comparators.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=1)
        outputs["published_comparators"] = path
        say(f"cluster_sweep: {len(specs)} published comparators scored on the same genes")
    outputs["figure"] = _sweep_figure(summary, os.path.join(out_dir, "sweep.png"), published_rows)
    return {"folder": out_dir, "outputs": outputs, "rows": int(len(names)), "graphs": tags}
