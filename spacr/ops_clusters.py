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

:func:`guide_bootstrap_consensus` (item 928) clusters guide bootstraps of
the gene profile with PHATE + Leiden at a matched cluster count and cuts
their co-clustering matrix into one partition in which every gene has a
cluster. :func:`paired_replicate_test` (item 929) compares two clusterings
over paired guide-bootstrap replicates, with an exact sign-flip p-value and
a two-level (replicates x genes) interval, after the jitter-null
:func:`calibration_gate` has checked its false-positive rate.

PHATE (``phate``), Leiden (``leidenalg``) and ``igraph`` are optional: they
are the ``clustering`` extra, ``pip install "spacr[clustering]"``. Only the
PHATE + Leiden steps need them. The benchmark files (CORUM, KEGG and STRING
tables in Brieflow's format) are the user's own: spaCR does not ship them.
"""
from __future__ import annotations

import os
from functools import lru_cache, partial
from importlib import import_module
from types import SimpleNamespace
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

    from .figures.bundle import _register_figure_data
    from .figures.style import figure_style, resolve_label_ground
    from .plot import _checked_savefig

    panels = (("CORUM_pct_enriched_may2025", "CORUM clusters enriched (%)"),
              ("KEGG_pct_enriched_may2025", "KEGG clusters enriched (%)"),
              ("STRING_f1", "STRING pair F1"),
              ("complexes_recovered_pct", "CORUM complexes in one cluster (%)"))
    colours = {"real": "#1f6f8b", "shuffled gene labels": "#9aa5b1"}
    with figure_style("print"):
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
        _register_figure_data(figure, summary, x="clusters mean", y=f"{panels[0][0]} mean", hue="control", kind="line")
        figure.tight_layout()
        _checked_savefig(figure, path, dpi=150, bbox_inches="tight", facecolor=resolve_label_ground("print"))
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


#: Bootstrap ``b`` of consensus replicate ``r`` draws its guides with seed
#: ``ops_consensus_draw_seed + CONSENSUS_REPLICATE_STRIDE * r + b``.
CONSENSUS_REPLICATE_STRIDE = 20

#: Bootstrap ``b`` of a consensus run with clustering seed ``s`` builds its
#: PHATE graph and Leiden partition with seed ``s * CONSENSUS_SEED_STRIDE + b``.
CONSENSUS_SEED_STRIDE = 100

#: Brieflow's published clustering: 209 clusters over its 2,647 active
#: genes. A gene set of ``n`` genes is cut into ``round(n * 209 / 2647)``.
CONSENSUS_K_RATIO = (209, 2647)

#: The replicate count whose calibration gate passed (Design D, 2026-11-11):
#: changing it needs a new gate record.
GATED_REPLICATES = 20

#: The calibrated test's metrics: unit key and output name.
CALIBRATED_METRICS: Tuple[Tuple[str, str], ...] = (
    ("cx", "complexes"), ("pr", "STRING pairs"), ("gm", "STRING pairs, per-gene capped"))


def _is_control_row(names: Sequence[str], prefix: str) -> np.ndarray:
    """Control rows: the control prefix or a non-targeting pseudo-gene."""
    lowered = pd.Series(np.asarray(names, dtype=object)).astype(str).str.lower()
    return lowered.str.startswith((prefix.lower(), "ntpseudo")).to_numpy()


def match_resolution(graph, k: int, seed: int, mask: Optional[np.ndarray] = None, *,
                     lo: float = 1e-3, hi: float = 1e3, iters: int = 22) -> Tuple[np.ndarray, float]:
    """The Leiden partition whose cluster count is nearest a target.

    A bisection on the resolution, on a log scale: each step runs
    :func:`leiden_partitions` once at the geometric mean of the bounds.

    :param graph: from :func:`gene_graph`.
    :param k: the target cluster count.
    :param seed: the Leiden seed of every step.
    :param mask: the rows counted (the targeting genes); every row when None.
    :param lo: the lowest resolution tried.
    :param hi: the highest resolution tried.
    :param iters: the most steps; the search stops early at exactly ``k``.
    :returns: ``(labels, resolution)`` of the step nearest ``k`` (the first
        on a tie).
    :raises ImportError: when the ``clustering`` extra is not installed.
    """
    mask = np.ones(graph.vcount(), bool) if mask is None else np.asarray(mask, bool)
    best = None
    for _ in range(int(iters)):
        mid = np.sqrt(lo * hi)
        labels = leiden_partitions(graph, [mid], [seed])[(int(seed), mid)]
        count = len(np.unique(labels[mask]))
        if best is None or abs(count - k) < abs(best[2] - k):
            best = (labels, mid, count)
        if count == k:
            break
        if count < k:
            lo = mid
        else:
            hi = mid
    return best[0], float(best[1])


def _unit_rows(rows: np.ndarray) -> np.ndarray:
    """Rows scaled to unit length."""
    return rows / np.clip(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12, None)


def _split_merge_to_k(unit: np.ndarray, labels: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Split the largest cluster (cosine 2-means) or merge the smallest into its nearest centroid, to ``k``."""
    labels = pd.factorize(labels)[0]
    while True:
        values, counts = np.unique(labels, return_counts=True)
        if len(values) == k:
            return pd.factorize(labels)[0]
        if len(values) < k:
            idx = np.flatnonzero(labels == values[np.argmax(counts)])
            centres = unit[rng.choice(idx, 2, replace=False)]
            for _ in range(20):
                side = np.argmax(unit[idx] @ centres.T, axis=1)
                if side.min() == side.max():
                    side = np.arange(len(idx)) % 2
                centres = _unit_rows(np.vstack([unit[idx[side == j]].mean(0) for j in (0, 1)]))
            labels[idx[side == 1]] = labels.max() + 1
        else:
            small = values[np.argmin(counts)]
            centres = _unit_rows(np.vstack([unit[labels == j].mean(0) for j in values]))
            similarity = centres @ centres[values == small][0]
            similarity[values == small] = -np.inf
            labels[labels == small] = values[np.argmax(similarity)]
            labels = pd.factorize(labels)[0]


def match_count(rows: np.ndarray, labels: np.ndarray, k: int, mask: Optional[np.ndarray] = None,
                seed: int = 0, tolerance: float = 0.1) -> Tuple[np.ndarray, bool]:
    """Bring a partition to the target cluster count when it is too far off.

    :param rows: the clustered rows, one per label.
    :param labels: the partition.
    :param k: the target count, over the ``mask`` rows.
    :param mask: the rows counted (the targeting genes); every row when None.
    :param seed: the seed of the 2-means splits.
    :param tolerance: a count within ``k * tolerance`` of ``k`` is kept.
    :returns: ``(labels, adjusted)``: labels renumbered from 0 in order of
        first appearance. When adjusted, the ``mask`` rows are split (the
        largest cluster by cosine 2-means) or merged (the smallest into its
        nearest centroid) to exactly ``k``; a control row keeps its cluster
        when no targeting row shared it, else it follows the most common new
        label of that cluster's targeting rows.
    """
    labels = np.asarray(labels)
    mask = np.ones(len(labels), bool) if mask is None else np.asarray(mask, bool)
    count = len(np.unique(labels[mask]))
    if abs(count - k) <= tolerance * k:
        return pd.factorize(labels)[0], False
    targeting = _split_merge_to_k(_unit_rows(np.asarray(rows, float)[mask]), labels[mask].copy(), int(k),
                                  np.random.default_rng(seed))
    out = np.empty(len(labels), int)
    out[mask] = targeting
    following = targeting.max() + 1
    old = pd.Series(labels[mask]).to_numpy()
    for i in np.flatnonzero(~mask):
        same = old == labels[i]
        out[i] = np.bincount(targeting[same]).argmax() if same.any() else following + int(labels[i])
    return pd.factorize(out)[0], True


def guide_bootstrap_profiles(rows: np.ndarray, names: Sequence[str], guide_rows: np.ndarray,
                             guide_genes: Sequence[str], guide_cells: Sequence[float], seed: int, *,
                             control_prefix: str = "nontargeting") -> np.ndarray:
    """One guide bootstrap of a gene profile table.

    Every targeting gene's guides are drawn with replacement, and its row is
    shifted by the change of the cell-weighted guide mean:
    ``row + (mean of the drawn guides - mean of all its guides)``, each mean
    weighted by the guides' cell counts. This rebuilds the profile from the
    drawn guides without aggregating the cells again. Control rows (one
    barcode each) and genes without guide rows are not resampled.

    :param rows: the gene profile, one row per name.
    :param names: the row names; guides are matched on them.
    :param guide_rows: one row per guide, in the profile's columns.
    :param guide_genes: each guide's gene.
    :param guide_cells: each guide's cell count.
    :param seed: the generator seed; genes draw in row order.
    :param control_prefix: rows named with it (or ``NTpseudo``) are controls.
    :returns: the bootstrap rows, a new array.
    """
    names = np.asarray(names).astype(str)
    guide_rows = np.asarray(guide_rows, float)
    cells = np.asarray(guide_cells, float)
    members: Dict[str, List[int]] = {}
    for i, gene in enumerate(np.asarray(guide_genes).astype(str)):
        members.setdefault(gene, []).append(i)
    control = _is_control_row(names, control_prefix)
    empty = np.array([], int)
    groups = [np.array(members[g], int) if g in members and not control[i] else empty
              for i, g in enumerate(names)]
    rng = np.random.default_rng(seed)
    drawn = [rng.choice(ix, len(ix), replace=True) if len(ix) else ix for ix in groups]
    shift = np.zeros((len(names), guide_rows.shape[1]))
    for i, (ix, bx) in enumerate(zip(groups, drawn)):
        if len(ix):
            shift[i] = ((cells[bx] @ guide_rows[bx]) / cells[bx].sum()
                        - (cells[ix] @ guide_rows[ix]) / cells[ix].sum())
    return np.asarray(rows, float) + shift


def coclustering_matrix(partitions: Sequence[np.ndarray]) -> np.ndarray:
    """The share of partitions in which two rows share a cluster.

    :param partitions: label vectors over the same rows; -1 marks a row left
        out of that partition.
    :returns: a symmetric single-precision ``rows x rows`` matrix in [0, 1]
        with ones on the diagonal (every row present at least once). With
        no row left out it is the co-occurrence count divided by the number
        of partitions; otherwise each pair is divided by the partitions
        holding both rows, and a pair never held together is 0.
    :raises ValueError: when no partition is given.
    """
    partitions = [np.asarray(p) for p in partitions]
    if not partitions:
        raise ValueError("coclustering_matrix needs at least one partition")
    n = len(partitions[0])
    together = np.zeros((n, n), np.float32)
    missing = any((p < 0).any() for p in partitions)
    both = np.zeros((n, n), np.float32) if missing else None
    for labels in partitions:
        present = labels >= 0
        together += (labels[:, None] == labels[None, :]) & present[:, None]
        if missing:
            both += present[:, None] & present[None, :]
    if not missing:
        together /= len(partitions)
        return together
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(both > 0, together / np.where(both > 0, both, 1), 0).astype(np.float32)


def consensus_partition(cooc: np.ndarray, k: int, mask: Optional[np.ndarray] = None, *,
                        method: str = "average", seed: int = 0, min_cooc: float = 0.0
                        ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """The consensus partition of a co-clustering matrix, at a target count.

    :param cooc: from :func:`coclustering_matrix`.
    :param k: the target cluster count over the ``mask`` rows.
    :param mask: the rows counted (the targeting genes); every row when None.
    :param method: ``'average'``, average linkage on ``1 - cooc`` cut into
        ``t`` clusters with ``t`` bisected over [2, 4k] to the count nearest
        ``k`` (Monti et al. 2003); or ``'leiden'``, Leiden on the graph of
        the pairs whose co-clustering exceeds ``min_cooc`` (weight = the
        co-clustering), its resolution matched to ``k`` by
        :func:`match_resolution`.
    :param seed: the Leiden seed (``'leiden'`` only).
    :param min_cooc: the edge floor (``'leiden'`` only).
    :returns: ``(labels, info)``; every row has a cluster. ``info`` holds the
        ``cut`` (average) or ``resolution`` (leiden) and ``clusters``.
    :raises ValueError: for an unknown method.
    """
    cooc = np.asarray(cooc)
    mask = np.ones(len(cooc), bool) if mask is None else np.asarray(mask, bool)
    if method == "average":
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        distance = 1 - cooc
        np.fill_diagonal(distance, 0)
        tree = linkage(squareform(distance, checks=False), "average")
        lo, hi, best = 2, 4 * int(k), None
        while lo <= hi:
            mid = (lo + hi) // 2
            labels = fcluster(tree, mid, "maxclust")
            count = len(np.unique(labels[mask]))
            if best is None or abs(count - k) < abs(best[1] - k):
                best = (labels, count, mid)
            if count == k:
                break
            if count < k:
                lo = mid + 1
            else:
                hi = mid - 1
        return best[0], {"cut": int(best[2]), "clusters": int(best[1])}
    if method == "leiden":
        igraph = _optional("igraph")
        i, j = np.nonzero(np.triu(cooc > min_cooc, 1))
        graph = igraph.Graph(n=len(cooc), edges=list(zip(i.tolist(), j.tolist())), directed=False)
        graph.es["weight"] = cooc[i, j].astype(float).tolist()
        labels, resolution = match_resolution(graph, int(k), int(seed), mask)
        return labels, {"resolution": resolution, "clusters": int(len(np.unique(labels[mask])))}
    raise ValueError(f"the consensus method must be 'average' or 'leiden'; got {method!r}")


def consensus_stability(cooc: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Each row's mean co-clustering with the other rows of its own cluster.

    :param cooc: from :func:`coclustering_matrix`.
    :param labels: the consensus partition.
    :returns: one value per row in [0, 1]; 1 for a row alone in its cluster.
    """
    labels = np.asarray(pd.factorize(np.asarray(labels))[0])
    out = np.ones(len(labels))
    for cluster in np.unique(labels):
        idx = np.flatnonzero(labels == cluster)
        if len(idx) > 1:
            out[idx] = (cooc[np.ix_(idx, idx)].sum(1) - 1) / (len(idx) - 1)
    return out


def guide_bootstrap_consensus(draw, n_boot: int, k: int, mask: Optional[np.ndarray] = None, *,
                              seed: int = 100, knn: int = 10, method: str = "average",
                              min_cooc: float = 0.0, rows: Optional[np.ndarray] = None,
                              match: bool = True, force_genes: bool = True, min_stability: float = 0.5,
                              n_jobs: int = 1, report=None) -> Dict[str, Any]:
    """Guide-bootstrap consensus clustering (item 928).

    Bootstrap ``b`` clusters ``draw(b)`` (a guide bootstrap of the profile,
    :func:`guide_bootstrap_profiles`) with PHATE (``knn`` cosine
    neighbours, :func:`gene_graph`) and Leiden at the target count
    (:func:`match_resolution`), both seeded ``seed * 100 + b``. The
    co-clustering matrix of the ``n_boot`` partitions
    (:func:`coclustering_matrix`) is cut into the consensus partition
    (:func:`consensus_partition`), which :func:`match_count` then brings
    within 10 % of ``k`` on ``rows`` with ``seed``.

    :param draw: ``draw(b) -> rows`` for ``b`` from 0 to ``n_boot - 1``, or
        a sequence of row arrays; every draw has the same rows in the same
        order.
    :param n_boot: the number of bootstraps.
    :param k: the target cluster count over the ``mask`` rows.
    :param mask: the rows counted (the targeting genes); every row when None.
    :param seed: the clustering seed.
    :param knn: PHATE's neighbours.
    :param method: see :func:`consensus_partition`.
    :param min_cooc: see :func:`consensus_partition`.
    :param rows: the rows :func:`match_count` splits and merges on; the
        first draw when None.
    :param match: False skips :func:`match_count`.
    :param force_genes: True assigns every gene to a cluster. False makes
        each row whose :func:`consensus_stability` is below
        ``min_stability`` a singleton (counted as unassigned) and skips
        :func:`match_count`.
    :param min_stability: see ``force_genes``.
    :param n_jobs: PHATE's worker count.
    :param report: called with one progress line per bootstrap.
    :returns: ``{"labels", "cooc", "stability", "cut_labels", "runs",
        "adjusted", "unassigned", "cut"}``: the final labels (from 0), the
        co-clustering matrix, each row's stability, the labels of the cut
        before any adjustment, one record per bootstrap (``bootstrap``,
        ``seed``, ``resolution``, ``clusters``, ``seconds``), whether the
        count was adjusted, the unassigned rows (a boolean array) and the
        consensus cut's own record.
    :raises ValueError: for fewer than one bootstrap.
    :raises ImportError: when the ``clustering`` extra is not installed.
    """
    import time

    if int(n_boot) < 1:
        raise ValueError(f"guide_bootstrap_consensus needs at least one bootstrap; got {n_boot}")
    fetch = draw if callable(draw) else draw.__getitem__
    partitions, runs, first = [], [], None
    for b in range(int(n_boot)):
        started = time.perf_counter()
        boot = np.asarray(fetch(b), float)
        first = boot if first is None else first
        run_seed = int(seed) * CONSENSUS_SEED_STRIDE + b
        graph = gene_graph(boot, "phate", seed=run_seed, knn=int(knn), n_jobs=n_jobs)
        boot_mask = np.ones(len(boot), bool) if mask is None else np.asarray(mask, bool)
        labels, resolution = match_resolution(graph, int(k), run_seed, boot_mask)
        partitions.append(np.asarray(labels))
        runs.append({"bootstrap": b, "seed": run_seed, "resolution": resolution,
                     "clusters": int(len(np.unique(labels[boot_mask]))),
                     "seconds": round(time.perf_counter() - started, 3)})
        if report is not None:
            report(f"consensus bootstrap {b + 1}/{n_boot}: {runs[-1]['clusters']} clusters")
    cooc = coclustering_matrix(partitions)
    cut, info = consensus_partition(cooc, int(k), mask, method=method, seed=int(seed), min_cooc=min_cooc)
    stability = consensus_stability(cooc, cut)
    unassigned = np.zeros(len(cut), bool)
    adjusted = False
    if not force_genes:
        labels = np.asarray(pd.factorize(cut)[0])
        unassigned = stability < min_stability
        labels[unassigned] = labels.max() + 1 + np.arange(int(unassigned.sum()))
    elif match:
        labels, adjusted = match_count(first if rows is None else rows, cut, int(k), mask, int(seed))
    else:
        labels = pd.factorize(cut)[0]
    return {"labels": np.asarray(labels), "cooc": cooc, "stability": stability, "cut_labels": np.asarray(cut),
            "runs": runs, "adjusted": bool(adjusted), "unassigned": unassigned, "cut": info}


def single_partition(rows: np.ndarray, k: int, mask: Optional[np.ndarray] = None, *, seed: int = 100,
                     knn: int = 10, n_jobs: int = 1) -> Tuple[np.ndarray, Dict[str, Any]]:
    """One PHATE + Leiden partition at a target count: the consensus comparator.

    :param rows: the profile rows.
    :param k: the target cluster count over the ``mask`` rows.
    :param mask: the rows counted; every row when None.
    :param seed: the PHATE and Leiden seed (and :func:`match_count`'s).
    :param knn: PHATE's neighbours.
    :param n_jobs: PHATE's worker count.
    :returns: ``(labels, info)``: :func:`match_resolution` on
        :func:`gene_graph`, then :func:`match_count`; ``info`` holds the
        ``resolution``, ``clusters`` and ``adjusted``.
    :raises ImportError: when the ``clustering`` extra is not installed.
    """
    rows = np.asarray(rows, float)
    mask = np.ones(len(rows), bool) if mask is None else np.asarray(mask, bool)
    graph = gene_graph(rows, "phate", seed=int(seed), knn=int(knn), n_jobs=n_jobs)
    labels, resolution = match_resolution(graph, int(k), int(seed), mask)
    labels, adjusted = match_count(rows, labels, int(k), mask, int(seed))
    return labels, {"resolution": resolution, "clusters": int(len(np.unique(labels[mask]))),
                    "adjusted": bool(adjusted)}


def target_cluster_count(n_genes: int, ratio: Sequence[int] = CONSENSUS_K_RATIO) -> int:
    """The cluster count matched to Brieflow's for a gene set.

    :param n_genes: the targeting genes clustered.
    :param ratio: ``(clusters, genes)`` of the reference clustering.
    :returns: ``round(n_genes * clusters / genes)``.
    """
    return int(round(int(n_genes) * int(ratio[0]) / int(ratio[1])))


def calibrated_reference(genes: Sequence[str], refs: Mapping[str, Any], half: Optional[int] = 1
                         ) -> Dict[str, Any]:
    """The units the calibrated test scores on one gene set and reference half.

    :param genes: the scored genes, in the order of the labels scored.
    :param refs: from :func:`sweep_references` (``groups``, ``halves`` and
        ``positives``).
    :param half: 0 (SELECT) or 1 (REPORT): the complexes of that half and
        the STRING pairs of that half
        (:func:`spacr.ops_reference.pair_half`); None keeps every pair and
        the REPORT complexes.
    :returns: ``{"genes", "n", "complexes", "cx_members", "pi", "pj",
        "pgenes", "pair_index"}``: CORUM complexes with at least three of
        their members among ``genes`` (name and member rows), the STRING
        pairs as sorted row indices, the genes with at least one pair and,
        for each, the indices of its pairs.
    """
    from .ops_reference import pair_half

    genes = np.asarray(genes).astype(str)
    index = {g: i for i, g in enumerate(genes)}
    wanted = 1 if half is None else int(half)
    complexes = []
    for name, members in refs["groups"].items():
        idx = np.array([index[x] for x in members if x in index], int)
        if len(idx) >= 3 and refs["halves"].get(name) == wanted:
            complexes.append((name, idx))
    pairs = {(a, b) for a, b in refs["positives"] if a in index and b in index
             and (half is None or pair_half(a, b) == half)}
    pp = np.array(sorted((index[a], index[b]) for a, b in pairs), int).reshape(-1, 2)
    by_gene: Dict[int, List[int]] = {}
    for unit, (i, j) in enumerate(zip(pp[:, 0], pp[:, 1])):
        by_gene.setdefault(int(i), []).append(unit)
        by_gene.setdefault(int(j), []).append(unit)
    pgenes = np.array(sorted(by_gene), int)
    return {"genes": genes, "n": len(genes), "complexes": complexes,
            "cx_members": [idx for _, idx in complexes], "pi": pp[:, 0], "pj": pp[:, 1],
            "pgenes": pgenes, "pair_index": [np.array(by_gene[g]) for g in pgenes]}


def _sizes(labels: np.ndarray) -> Tuple[int, ...]:
    """Cluster sizes, largest first."""
    return tuple(sorted(np.unique(labels, return_counts=True)[1].tolist(), reverse=True))


def _recovered(labels: np.ndarray, reference: Mapping[str, Any]) -> np.ndarray:
    """Per complex: at least half its members in one cluster."""
    return np.array([np.unique(labels[idx], return_counts=True)[1].max() / len(idx) >= 0.5
                     for idx in reference["cx_members"]], bool)


def size_matched_null(sizes: Sequence[int], reference: Mapping[str, Any], nperm: int = 200
                      ) -> Dict[str, Any]:
    """The size-matched null of the calibrated test's units.

    Gene labels are permuted with the exact cluster sizes kept; the seed is
    taken from the size vector, so the same sizes give the same null.

    :param sizes: cluster sizes, largest first (as a partition's sizes).
    :param reference: from :func:`calibrated_reference`.
    :param nperm: the permutations.
    :returns: ``complex_prob`` (each complex's recovery probability) and
        ``pair_prob`` (the probability two genes share a cluster, exact).
    """
    import hashlib

    sizes = tuple(int(s) for s in sizes)
    sz = np.asarray(sizes)
    labels = np.repeat(np.arange(len(sz)), sz)
    rng = np.random.default_rng(int(hashlib.md5(str(sizes).encode()).hexdigest()[:8], 16))
    recovered = [_recovered(labels[rng.permutation(len(labels))], reference) for _ in range(int(nperm))]
    n = sz.sum()
    return {"complex_prob": np.mean(recovered, 0) if recovered else np.zeros(len(reference["cx_members"])),
            "pair_prob": float((sz * (sz - 1.0)).sum() / (n * (n - 1.0)))}


def partition_units(labels: np.ndarray, reference: Mapping[str, Any],
                    nulls: Optional[Dict[Tuple[int, ...], Dict[str, Any]]] = None, nperm: int = 200
                    ) -> Dict[str, np.ndarray]:
    """Per-unit excess of one partition over the size-matched null.

    :param labels: one label per gene of ``reference``.
    :param reference: from :func:`calibrated_reference`.
    :param nulls: a cache ``{sizes: null}`` filled as needed.
    :param nperm: the null's permutations.
    :returns: ``cx`` (each complex: recovered minus its null probability),
        ``pr`` (each STRING pair: in one cluster minus the null
        co-membership probability), ``gm`` (each gene with a pair: the mean
        of its pairs' ``pr``, the per-gene-capped metric) and the raw values
        ``cx_raw``, ``pr_raw``, ``gm_raw``.
    """
    labels = np.asarray(labels)
    nulls = {} if nulls is None else nulls
    sizes = _sizes(labels)
    if sizes not in nulls:
        nulls[sizes] = size_matched_null(sizes, reference, nperm)
    null = nulls[sizes]
    recovered = _recovered(labels, reference).astype(float)
    together = (labels[reference["pi"]] == labels[reference["pj"]]).astype(float)
    pr = together - null["pair_prob"]
    return {"cx": recovered - null["complex_prob"], "pr": pr,
            "gm": np.array([pr[ix].mean() for ix in reference["pair_index"]]),
            "cx_raw": recovered, "pr_raw": together,
            "gm_raw": np.array([together[ix].mean() for ix in reference["pair_index"]])}


def stack_units(units: Sequence[Mapping[str, np.ndarray]], raw: bool = False) -> Dict[str, np.ndarray]:
    """Replicates by units arrays of :func:`partition_units` results.

    :param units: one result per replicate, in replicate order.
    :param raw: True stacks the raw values under the same keys.
    :returns: ``{"cx", "pr", "gm"}``, each ``replicates x units``.
    """
    suffix = "_raw" if raw else ""
    return {key: np.vstack([u[key + suffix] for u in units]) for key, _ in CALIBRATED_METRICS}


@lru_cache(maxsize=4)
def _sign_patterns(n: int) -> np.ndarray:
    """Every sign pattern of ``n`` replicates, single precision."""
    import itertools

    return np.array(list(itertools.product([1.0, -1.0], repeat=n)), dtype=np.float32)


def signflip_exact(d: Sequence[float], one_sided: bool = False) -> float:
    """Exact paired sign-flip p-value over replicate differences.

    :param d: one difference per replicate.
    :param one_sided: True tests ``mean(d) > 0``; False is two-sided.
    :returns: the share of the ``2^R`` sign patterns whose mean is at least
        as extreme as the observed mean (every pattern enumerated for
        ``R <= 20``; for more replicates 200,000 random patterns, seed 0);
        1.0 when every difference is 0.
    """
    d = np.asarray(d, float)
    n = len(d)
    observed = d.mean()
    if np.allclose(d, 0):
        return 1.0
    if n <= 20:
        signs = _sign_patterns(n)
    else:
        signs = np.random.default_rng(0).choice([-1.0, 1.0], size=(200000, n))
    means = (signs @ d.astype(np.float32)) / n
    if one_sided:
        return float(np.mean(means >= observed - 1e-12))
    return float(np.mean(np.abs(means) >= abs(observed) - 1e-12))


def two_level_ci(differences: np.ndarray, kind: str, reference: Mapping[str, Any], *, draws: int = 2000,
                 seed: int = 0, level: float = 0.95) -> Tuple[float, float]:
    """The two-level bootstrap interval of a paired effect.

    Each draw resamples the replicates with replacement and the genes with
    replacement (a block bootstrap over genes): a pair is weighted by the
    product of its genes' multiplicities, a complex by the mean
    multiplicity of its members, a gene (the capped metric) by its own.

    :param differences: ``replicates x units`` per-unit differences.
    :param kind: ``'cx'``, ``'pr'`` or ``'gm'``.
    :param reference: from :func:`calibrated_reference`.
    :param draws: the bootstrap draws.
    :param seed: the generator seed.
    :param level: the interval's coverage.
    :returns: ``(low, high)`` as fractions (not points).
    """
    rng = np.random.default_rng(seed)
    n = reference["n"]
    differences = np.atleast_2d(differences)
    replicates = differences.shape[0]
    out = []
    for _ in range(int(draws)):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        effect = differences[rng.integers(0, replicates, replicates)].mean(0)
        if kind == "pr":
            weight = w[reference["pi"]] * w[reference["pj"]]
        elif kind == "cx":
            weight = np.array([w[m].mean() for m in reference["cx_members"]])
        else:
            weight = w[reference["pgenes"]]
        out.append((weight * effect).sum() / max(weight.sum(), 1e-12))
    tail = 100 * (1 - level) / 2
    return float(np.percentile(out, tail)), float(np.percentile(out, 100 - tail))


def paired_replicate_test(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray],
                          reference: Mapping[str, Any], *, one_sided: bool = False,
                          labels: Optional[Mapping[str, Any]] = None, seed: int = 0) -> List[Dict[str, Any]]:
    """The calibrated paired-replicate test of two clusterings (item 929).

    :param a: :func:`stack_units` of the arm's replicates.
    :param b: :func:`stack_units` of the comparator's replicates, paired by
        replicate row.
    :param reference: from :func:`calibrated_reference`.
    :param one_sided: True tests ``a > b``.
    :param labels: extra columns of every row.
    :param seed: the interval's seed.
    :returns: one row per metric (complexes, STRING pairs, per-gene-capped
        pairs): ``a_excess_pts`` and ``b_excess_pts`` (mean excess over the
        size-matched null, points), ``diff_pts`` (the mean over replicates
        of the difference), its 95 % two-level interval ``ci_low`` /
        ``ci_high`` (:func:`two_level_ci`), ``sd_rep_diff_pts`` and the
        exact sign-flip ``p`` over the replicate differences
        (:func:`signflip_exact`).
    """
    rows = []
    for kind, name in CALIBRATED_METRICS:
        x, y = np.asarray(a[kind], float), np.asarray(b[kind], float)
        per_replicate = x.mean(1) - y.mean(1)
        low, high = two_level_ci(x - y, kind, reference, seed=seed)
        rows.append(dict(unit=name, n_units=x.shape[1], replicates=x.shape[0], a_excess_pts=100 * x.mean(),
                         b_excess_pts=100 * y.mean(), diff_pts=100 * per_replicate.mean(),
                         ci_low=100 * low, ci_high=100 * high,
                         sd_rep_diff_pts=100 * per_replicate.std(ddof=1),
                         p=signflip_exact(per_replicate, one_sided),
                         sided="one (a > b)" if one_sided else "two", **dict(labels or {})))
    return rows


def jitter_rows(rows: np.ndarray, copy: int, size: float = 1.7e-3, seed_base: int = 50000) -> np.ndarray:
    """A jitter copy of profile rows: the calibration gate's null.

    :param rows: the rows (one replicate draw).
    :param copy: the copy number; its noise is seeded ``seed_base + copy``
        and so is the same in every replicate.
    :param size: the noise norm relative to each row's norm.
    :param seed_base: see ``copy``.
    :returns: ``rows`` plus Gaussian noise whose norm is ``size`` times each
        row's norm.
    """
    rows = np.asarray(rows, float)
    noise = np.random.default_rng(int(seed_base) + int(copy)).standard_normal(rows.shape)
    noise = noise / np.linalg.norm(noise, axis=1, keepdims=True) * np.linalg.norm(rows, axis=1, keepdims=True) * size
    return rows + noise


def calibration_gate(base: Mapping[str, np.ndarray], copies: Sequence[Mapping[str, np.ndarray]],
                     reference: Mapping[str, Any], *, alpha: float = 0.05, limit: float = 0.05,
                     labels: Optional[Mapping[str, Any]] = None) -> Tuple[pd.DataFrame, Dict[str, float], bool]:
    """The jitter-null calibration gate of the paired-replicate test.

    Each jitter copy's replicates are tested against the base profile's
    with :func:`paired_replicate_test` (seed = the copy number); a jitter
    copy differs from the base by noise only, so every rejection is a false
    positive.

    :param base: :func:`stack_units` of the base profile.
    :param copies: :func:`stack_units` of each jitter copy, in copy order.
    :param reference: from :func:`calibrated_reference`.
    :param alpha: the test level.
    :param limit: the largest false-positive share that passes.
    :param labels: extra columns of every row.
    :returns: ``(table, false_positive_share, passed)``: one row per copy and
        metric, the share of copies with ``p < alpha`` per metric, and
        whether complexes and STRING pairs are both at or under ``limit``.
    """
    rows = []
    for j, units in enumerate(copies):
        rows += paired_replicate_test(units, base, reference, labels=dict(dict(labels or {}), copy=j), seed=j)
    table = pd.DataFrame(rows)
    share = table.groupby("unit").p.apply(lambda p: float((p < alpha).mean())).to_dict()
    passed = all(share.get(unit, 1.0) <= limit + 1e-9 for unit in ("complexes", "STRING pairs"))
    return table, share, passed


def noise_floor_margin(gate: pd.DataFrame, by: Sequence[str] = ("unit",), quantile: float = 0.95,
                       alpha: float = 0.05, limit: float = 0.05) -> pd.DataFrame:
    """The noise-floor margin of a gate table, with its cross-validated check.

    delta = the ``quantile`` of ``|diff_pts|`` over the jitter copies. The
    check fits delta on the first half of the copies and counts the copies
    of the second half with ``p < alpha`` and ``|diff| > delta``, then the
    reverse; the two shares are averaged.

    :param gate: :func:`calibration_gate`'s table (``copy``, ``p``,
        ``diff_pts`` and the ``by`` columns).
    :param by: the columns a margin is computed per (setup, cluster count,
        metric).
    :param quantile: see above.
    :param alpha: the test level.
    :param limit: the largest false-positive share that passes.
    :returns: one row per group: ``delta_pts``,
        ``raw_false_positive_share``, ``margin_rule_false_positive_share_cv``
        and ``passed``.
    """
    rows = []
    for key, group in gate.groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        absd = group["diff_pts"].abs()
        half = (group["copy"].max() + 1) // 2
        rates = []
        for fit, test in ((group["copy"] < half, group["copy"] >= half),
                          (group["copy"] >= half, group["copy"] < half)):
            delta = absd[fit].quantile(quantile)
            rates.append(float(((group.p[test] < alpha) & (absd[test] > delta)).mean()))
        rows.append(dict(zip(by, key), delta_pts=float(absd.quantile(quantile)),
                         raw_false_positive_share=float((group.p < alpha).mean()),
                         margin_rule_false_positive_share_cv=float(np.mean(rates)),
                         passed=bool(np.mean(rates) <= limit + 1e-9)))
    return pd.DataFrame(rows)


def _holm(p: Sequence[float]) -> np.ndarray:
    """Holm-adjusted p-values, in input order."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def call_effects(tests: pd.DataFrame, margins: Optional[Mapping[str, float]] = None, *, alpha: float = 0.05,
                 gated: Optional[bool] = None) -> pd.DataFrame:
    """Holm q values and the call of every metric of one comparison family.

    :param tests: :func:`paired_replicate_test` rows of one family.
    :param margins: ``{metric: delta_pts}``; a call then also needs
        ``|diff_pts| > delta`` (the noise-floor rule). None calls on q alone.
    :param alpha: the level of q.
    :param gated: True when the setup's gate passed, False when it failed,
        None when it was not run; written as ``gate``.
    :returns: the rows with ``q`` (Holm), ``margin_pts``, ``call``
        (``'higher'``, ``'lower'`` or ``'no detectable effect'``) and
        ``gate`` (``'passed'``, ``'failed: margin rule'`` or ``'not gated'``).
    """
    out = tests.copy()
    out["q"] = _holm(out["p"].to_numpy()) if len(out) else []
    margins = dict(margins or {})
    out["margin_pts"] = [float(margins.get(unit, 0.0)) for unit in out["unit"]]
    out["call"] = ["higher" if q < alpha and d > m else "lower" if q < alpha and d < -m else "no detectable effect"
                   for q, d, m in zip(out["q"], out["diff_pts"], out["margin_pts"])]
    out["gate"] = "not gated" if gated is None else ("passed" if gated else "failed: margin rule")
    return out


def _guide_inputs(settings: Mapping[str, Any], genes: Optional[pd.DataFrame], guides: Optional[pd.DataFrame]
                  ) -> Dict[str, Any]:
    """The profile rows, their guide rows and the rows clustered, for the consensus and bench phases.

    Ready rows (``ops_cluster_rows``) are clustered as given, with their
    guides from ``ops_consensus_guides`` (a table with ``gene``,
    ``n_cells`` and the same numeric columns). Otherwise the profile
    phase's gene and guide tables are put in the Hybrid space
    (:func:`spacr.ops_reference.hybrid_profile_rows`: the gene-level
    embedding the evaluate phase uses, fitted once on the gene rows, with
    the guide rows sent through the same transform), rounded to single
    precision (the precision of the profile tables and of the plate A rows
    Design F clustered; on plate A one replicate in 20 changes its
    partition without the rounding), and every draw moves the genes within
    that space.

    :raises ValueError: when no rows or no guide rows are available, or a
        listed gene is not a row.
    """
    from . import ops_reference
    from .tabular import read_table

    prefix = str(settings.get("ops_control_prefix") or "nontargeting")
    source = settings.get("ops_cluster_rows")
    guide_source = settings.get("ops_consensus_guides")
    if source:
        table = read_table(str(source), canonicalise=False, report=None)
        names, rows = _sweep_rows_from_table(table)
        columns = [c for c in table.columns if c not in ("gene", "n_cells")
                   and pd.api.types.is_numeric_dtype(table[c])]
    elif genes is not None:
        kept = genes[genes["n_cells"] > int(settings.get("ops_profile_min_cells", 10))]
        columns = [c for c in kept.columns if c.startswith(("PC_", "w_"))]
        if guide_source:
            guides = read_table(str(guide_source), canonicalise=False, report=None)
        if guides is None:
            raise ValueError("guide-bootstrap consensus needs the guide rows: set ops_consensus_guides (gene, "
                             "n_cells and the profile's columns) or run the profile phase first")
        hybrid, guides = ops_reference.hybrid_profile_rows(kept.reset_index(drop=True), guides, columns)
        columns = [c for c in hybrid.columns if c.startswith("E_")]
        names = hybrid["gene"].astype(str).to_numpy()
        rows = hybrid[columns].to_numpy(np.float32).astype(float)
        guides = guides.assign(**{c: guides[c].to_numpy(np.float32).astype(float) for c in columns})
        guide_source = None
    else:
        raise ValueError("the consensus and bench phases need gene rows: run the profile phase first or set "
                         "ops_cluster_rows")
    if guide_source:
        guides = read_table(str(guide_source), canonicalise=False, report=None)
    if guides is None:
        raise ValueError("guide-bootstrap consensus needs the guide rows: set ops_consensus_guides (gene, n_cells "
                         "and the profile's columns) or run the profile phase first")
    missing = [c for c in [*columns, "gene", "n_cells"] if c not in guides.columns]
    if missing:
        raise ValueError(f"the guide table lacks the profile's columns {missing[:5]}")
    gene_list = settings.get("ops_cluster_gene_list")
    control = _is_control_row(names, prefix)
    if gene_list:
        scored = read_table(str(gene_list), canonicalise=False, report=None)["gene"].astype(str).tolist()
        unknown = sorted(set(scored) - set(names))
        if unknown:
            raise ValueError(f"{len(unknown)} listed genes are not profile rows, such as {unknown[:3]}")
    else:
        scored = names[~control].tolist()
    order = scored + names[control].tolist()
    position = pd.Series(np.arange(len(names)), index=names).loc[order].to_numpy()
    mask = np.zeros(len(order), bool)
    mask[:len(scored)] = True
    return {"names": names, "rows": rows, "guide_rows": guides[columns].to_numpy(float),
            "guide_genes": guides["gene"].astype(str).to_numpy(),
            "guide_cells": guides["n_cells"].to_numpy(float), "position": position, "order": np.array(order),
            "mask": mask, "prefix": prefix}


def _draw(data: Mapping[str, Any], seed: Optional[int], copy: Optional[int] = None, noise: float = 1.7e-3
          ) -> np.ndarray:
    """One guide bootstrap (``seed``; the base rows when None) of the clustered rows, optionally jittered."""
    rows = data["rows"] if seed is None else guide_bootstrap_profiles(
        data["rows"], data["names"], data["guide_rows"], data["guide_genes"], data["guide_cells"], int(seed),
        control_prefix=data["prefix"])
    rows = rows[data["position"]]
    if copy is not None:
        rows = jitter_rows(rows, copy, noise)
    return rows


def _consensus_options(settings: Mapping[str, Any], n_genes: int) -> SimpleNamespace:
    """The ``ops_consensus_*`` settings with their defaults."""
    method = str(settings.get("ops_consensus_method") or "average").lower()
    if method not in ("average", "leiden"):
        raise ValueError(f"ops_consensus_method must be 'average' or 'leiden'; got {method!r}")
    ratio = settings.get("ops_consensus_k_ratio") or CONSENSUS_K_RATIO
    k = settings.get("ops_consensus_k")
    return SimpleNamespace(**{"n_boot": int(settings.get("ops_consensus_bootstraps") or 10),
            "seed": int(100 if settings.get("ops_consensus_seed") is None else settings["ops_consensus_seed"]),
            "draw_seed": int(8000 if settings.get("ops_consensus_draw_seed") is None
                             else settings["ops_consensus_draw_seed"]),
            "knn": int(settings.get("ops_consensus_knn") or 10), "method": method,
            "min_cooc": float(settings.get("ops_consensus_min_cooc") or 0.0),
            "k": int(k) if k else target_cluster_count(n_genes, ratio),
            "force_genes": bool(True if settings.get("ops_consensus_force_genes") is None
                                else settings["ops_consensus_force_genes"]),
            "min_stability": float(0.5 if settings.get("ops_consensus_min_stability") is None
                                   else settings["ops_consensus_min_stability"]),
            "match": bool(True if settings.get("ops_consensus_match_count") is None
                          else settings["ops_consensus_match_count"])})


def _run_consensus(data: Mapping[str, Any], options: SimpleNamespace, seed: int, draw_seed: int,
                   rows: np.ndarray, copy: Optional[int] = None, noise: float = 1.7e-3, report=None
                   ) -> Dict[str, Any]:
    """:func:`guide_bootstrap_consensus` of one replicate."""
    return guide_bootstrap_consensus(
        lambda b: _draw(data, draw_seed + b, copy, noise), options.n_boot, options.k, data["mask"],
        seed=seed, knn=options.knn, method=options.method, min_cooc=options.min_cooc, rows=rows,
        match=options.match, force_genes=options.force_genes, min_stability=options.min_stability,
        report=report)


def _consensus_figure(cooc: np.ndarray, labels: np.ndarray, path: str) -> str:
    """The co-clustering matrix with rows and columns ordered by consensus cluster."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from .figures.bundle import _register_figure_data
    from .figures.style import figure_style, resolve_label_ground
    from .plot import _checked_savefig

    order = np.argsort(labels, kind="stable")
    shown = cooc[np.ix_(order, order)]
    with figure_style("print"):
        figure = Figure(figsize=(6, 5.4))
        ax = figure.subplots()
        image = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"{len(labels)} rows, ordered by consensus cluster")
        figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="share of bootstraps in one cluster")
        _register_figure_data(figure, pd.DataFrame(shown), kind="heatmap", matrix=True)
        figure.tight_layout()
        _checked_savefig(figure, path, dpi=150, bbox_inches="tight", facecolor=resolve_label_ground("print"))
    return path


def cluster_consensus_phase(settings: Mapping[str, Any], genes: Optional[pd.DataFrame],
                            guides: Optional[pd.DataFrame], folder: str, report=None) -> Dict[str, Any]:
    """The ``cluster_consensus`` phase of the OPS profile path (item 928, alpha).

    Clusters ``ops_consensus_bootstraps`` guide bootstraps of the gene
    profile (bootstrap ``b`` draws its guides with seed
    ``ops_consensus_draw_seed + b`` and clusters with PHATE + Leiden at the
    target count, seed ``ops_consensus_seed * 100 + b``) and cuts their
    co-clustering matrix into the consensus partition
    (:func:`guide_bootstrap_consensus`); the count is then matched on the
    profile's own rows with seed ``ops_consensus_seed``.

    :param settings: read for ``ops_cluster_rows`` and
        ``ops_consensus_guides`` (ready rows and their guide rows; when
        empty, the profile phase's gene and guide tables in the Hybrid
        space of :func:`spacr.ops_reference.hybrid_profile_rows`), ``ops_cluster_gene_list`` (the
        targeting genes clustered, in order; every targeting row when
        empty; control rows are always clustered and never counted),
        ``ops_consensus_bootstraps``, ``ops_consensus_seed``,
        ``ops_consensus_draw_seed``, ``ops_consensus_knn``,
        ``ops_consensus_k`` (the target count; ``round(genes *
        ops_consensus_k_ratio)`` when empty), ``ops_consensus_k_ratio``,
        ``ops_consensus_method``, ``ops_consensus_min_cooc``,
        ``ops_consensus_force_genes``, ``ops_consensus_min_stability``,
        ``ops_consensus_match_count``, ``ops_profile_min_cells`` and
        ``ops_control_prefix``.
    :param genes: the profile phase's gene table, or None.
    :param guides: the profile phase's guide table, or None.
    :param folder: ``dst_root``; outputs go to its ``results/ops_clusters``.
    :param report: called with progress lines.
    :returns: ``{"folder", "outputs", "k", "clusters", "adjusted"}``.
    :raises ValueError: when no rows or guide rows are given, or a setting is
        out of range.
    :raises ImportError: when the ``clustering`` extra is not installed.
    """
    import json

    from .tabular import write_table

    say = report or (lambda text: None)
    data = _guide_inputs(settings, genes, guides)
    options = _consensus_options(settings, int(data["mask"].sum()))
    out_dir = os.path.join(folder, "results", "ops_clusters")
    os.makedirs(out_dir, exist_ok=True)
    result = _run_consensus(data, options, options.seed, options.draw_seed, _draw(data, None), report=say)
    labels, mask = result["labels"], data["mask"]
    outputs: Dict[str, str] = {}
    path = os.path.join(out_dir, "consensus_cooc.npz")
    np.savez_compressed(path, cooc=result["cooc"], genes=data["order"].astype(str), targeting=mask)
    outputs["consensus_cooc"] = path
    partition = pd.DataFrame({"gene": data["order"], "cluster": labels, "stability": result["stability"],
                              "unassigned": result["unassigned"], "targeting": mask})
    outputs["consensus_partition"] = write_table(partition, os.path.join(out_dir, "consensus_partition.csv"),
                                                 canonicalise=False)
    outputs["consensus_runs"] = write_table(pd.DataFrame(result["runs"]),
                                            os.path.join(out_dir, "consensus_runs.csv"), canonicalise=False)
    outputs["consensus_figure"] = _consensus_figure(result["cooc"], labels,
                                                    os.path.join(out_dir, "consensus.png"))
    record = {"k": options.k, "clusters": int(len(np.unique(labels[mask]))), "adjusted": result["adjusted"],
              "cut": result["cut"], "options": vars(options), "rows": int(len(labels)), "genes": int(mask.sum())}
    path = os.path.join(out_dir, "consensus_summary.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1, default=str)
    outputs["consensus_summary"] = path
    say(f"cluster_consensus: {record['clusters']} clusters (target {options.k}) from "
        f"{options.n_boot} guide bootstraps")
    return {"folder": out_dir, "outputs": outputs, "k": options.k, "clusters": record["clusters"],
            "adjusted": result["adjusted"]}


def _bench_options(settings: Mapping[str, Any]) -> SimpleNamespace:
    """The ``ops_bench_*`` settings with their defaults, R checked against the gate record."""
    def get(key, default):
        """One setting, its default when unset."""
        return default if settings.get(key) is None else settings[key]

    options = SimpleNamespace(**{"replicates": int(get("ops_bench_replicates", GATED_REPLICATES)),
               "replicate_seed": int(get("ops_bench_replicate_seed", 7000)),
               "cluster_seed": int(get("ops_bench_cluster_seed", 100)),
               "gate_copies": int(get("ops_bench_gate_copies", 40)),
               "gate_noise": float(get("ops_bench_gate_noise", 1.7e-3)),
               "alpha": float(get("ops_bench_gate_alpha", 0.05)),
               "limit": float(get("ops_bench_gate_limit", 0.05)),
               "margin_quantile": float(get("ops_bench_margin_quantile", 0.95)),
               "gate_record": settings.get("ops_bench_gate_record") or None,
               "half": None if settings.get("ops_bench_half", 1) is None else int(settings.get("ops_bench_half", 1)),
               "nperm": int(get("ops_bench_null_permutations", 200)),
               "gate_arm": str(get("ops_bench_gate_arm", "consensus")).lower()})
    if options.gate_arm not in ("consensus", "phate_leiden"):
        raise ValueError(f"ops_bench_gate_arm must be 'consensus' or 'phate_leiden'; got {options.gate_arm!r}")
    if options.replicates < 2:
        raise ValueError("ops_bench_replicates must be at least 2")
    if options.replicates != GATED_REPLICATES and options.gate_copies < 1 and not options.gate_record:
        raise ValueError(
            f"ops_bench_replicates={options.replicates} has no gate record: R = {GATED_REPLICATES} is the gated "
            "value, and another R needs a new calibration gate (ops_bench_gate_copies > 0) or "
            "ops_bench_gate_record")
    return options


def _gate_record(path: str, options: SimpleNamespace, arm: str) -> Optional[pd.DataFrame]:
    """A saved gate table that matches this run's R, jitter size and setup, or None."""
    from .tabular import read_table

    table = read_table(str(path), canonicalise=False, report=None)
    same = ((table["replicates"] == options.replicates) & (table["arm"] == arm)
            & np.isclose(table["jitter"].astype(float), options.gate_noise))
    if not same.any():
        raise ValueError(f"ops_bench_gate_record {path} has no gate of {arm} at R = {options.replicates} and "
                         f"jitter {options.gate_noise}: run the gate (ops_bench_gate_copies > 0)")
    return table[same].reset_index(drop=True)


def _bench_figure(units: Mapping[str, Mapping[str, np.ndarray]], path: str) -> str:
    """Bars (mean) and jittered replicate points of each arm's excess, per metric."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from .figures.bundle import _register_figure_data
    from .figures.style import figure_style, resolve_label_ground
    from .plot import _checked_savefig

    colours = {"consensus": "#1b6f8a", "phate_leiden": "#23395d"}
    with figure_style("print"):
        figure = Figure(figsize=(3.2 * len(CALIBRATED_METRICS), 3.4))
        axes = figure.subplots(1, len(CALIBRATED_METRICS))
        rng = np.random.default_rng(0)
        for ax, (kind, name) in zip(axes, CALIBRATED_METRICS):
            for x, arm in enumerate(units):
                values = 100 * units[arm][kind].mean(1)
                ax.bar(x, values.mean(), color=colours.get(arm, "#4a90b8"), alpha=0.6, width=0.6)
                ax.scatter(x + rng.uniform(-0.15, 0.15, len(values)), values, s=10, color="#0d1b2a", alpha=0.6)
            ax.set_xticks(range(len(units)))
            ax.set_xticklabels([a.replace("_", " + ").replace("phate", "PHATE").replace("leiden", "Leiden")
                                for a in units], fontsize=8)
            ax.set_title(name, fontsize=9)
            ax.set_ylabel("excess over size-matched null (points)", fontsize=8)
        _register_figure_data(figure, lambda: pd.DataFrame(
            [{"arm": arm, "metric": name, "excess": float(v)} for kind, name in CALIBRATED_METRICS
             for arm in units for v in 100 * units[arm][kind].mean(1)]),
            x="arm", y="excess", hue="metric", kind="bar")
        figure.tight_layout()
        _checked_savefig(figure, path, dpi=150, bbox_inches="tight", facecolor=resolve_label_ground("print"))
    return path


def cluster_bench_test_phase(settings: Mapping[str, Any], genes: Optional[pd.DataFrame],
                             guides: Optional[pd.DataFrame], folder: str, report=None) -> Dict[str, Any]:
    """The ``cluster_bench_test`` phase: consensus vs PHATE + Leiden under the calibrated test (item 929, alpha).

    Replicate ``r`` of both arms clusters the guide bootstrap seeded
    ``ops_bench_replicate_seed + r`` with clustering seed
    ``ops_bench_cluster_seed + r``: the single PHATE + Leiden run
    (:func:`single_partition`) and the consensus
    (:func:`guide_bootstrap_consensus`, inner bootstrap ``b`` drawn with
    seed ``ops_consensus_draw_seed + 20 r + b``). Each partition of the
    scored genes is turned into per-unit excess over the size-matched null
    (:func:`partition_units`) on the ``ops_bench_half`` reference half, and
    the arms are compared by :func:`paired_replicate_test`. Before the test
    is used, the gate (:func:`calibration_gate`) tests
    ``ops_bench_gate_copies`` jitter copies of the gated arm against it;
    when it fails, every call also needs ``|effect|`` above the noise-floor
    margin (:func:`noise_floor_margin`), and a metric whose margin rule
    still fails its cross-validated check is marked not calibrated.

    :param settings: read for the inputs of :func:`cluster_consensus_phase`
        and its ``ops_consensus_*`` settings, ``ops_cluster_reference_dir``
        (falling back to ``ops_reference_dir``), ``ops_bench_replicates``
        (R; a value other than 20 needs a gate in this run or a matching
        ``ops_bench_gate_record``), ``ops_bench_replicate_seed``,
        ``ops_bench_cluster_seed``, ``ops_bench_gate_copies``,
        ``ops_bench_gate_noise``, ``ops_bench_gate_alpha``,
        ``ops_bench_gate_limit``, ``ops_bench_margin_quantile``,
        ``ops_bench_gate_record``, ``ops_bench_gate_arm``,
        ``ops_bench_half`` (None scores every STRING pair) and
        ``ops_bench_null_permutations``.
    :param genes: the profile phase's gene table, or None.
    :param guides: the profile phase's guide table, or None.
    :param folder: ``dst_root``; outputs go to its ``results/ops_clusters``.
    :param report: called with progress lines.
    :returns: ``{"folder", "outputs", "gate_passed", "calls"}``.
    :raises ValueError: for missing inputs, a changed R without a gate
        record, or an unknown gated arm.
    :raises ImportError: when the ``clustering`` extra is not installed.
    """
    import json

    from .tabular import write_table

    say = report or (lambda text: None)
    options = _bench_options(settings)
    data = _guide_inputs(settings, genes, guides)
    consensus = _consensus_options(settings, int(data["mask"].sum()))
    reference_dir = settings.get("ops_cluster_reference_dir") or settings.get("ops_reference_dir")
    if not reference_dir:
        raise ValueError("cluster_bench_test needs ops_cluster_reference_dir (or ops_reference_dir), the folder "
                         "holding Brieflow's CORUM, KEGG and STRING benchmark tables")
    mask, k = data["mask"], consensus.k
    reference = calibrated_reference(data["order"][mask], sweep_references(reference_dir), options.half)
    nulls: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    out_dir = os.path.join(folder, "results", "ops_clusters")
    os.makedirs(out_dir, exist_ok=True)

    def replicate(arm: str, r: int, copy: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """One replicate partition of one arm (optionally of jitter copy ``copy``)."""
        rows = _draw(data, options.replicate_seed + r, copy, options.gate_noise)
        seed = options.cluster_seed + r
        if arm == "phate_leiden":
            return single_partition(rows, k, mask, seed=seed, knn=consensus.knn)
        result = _run_consensus(data, consensus, seed, consensus.draw_seed + CONSENSUS_REPLICATE_STRIDE * r,
                                rows, copy, options.gate_noise)
        return result["labels"], {"clusters": int(len(np.unique(result["labels"][mask]))),
                                  "adjusted": result["adjusted"]}

    labels: Dict[str, List[np.ndarray]] = {"consensus": [], "phate_leiden": []}
    records = []
    for r in range(options.replicates):
        for arm in labels:
            part, info = replicate(arm, r)
            labels[arm].append(np.asarray(part)[mask])
            records.append({"arm": arm, "replicate": r, **info})
        say(f"cluster_bench_test: replicate {r + 1}/{options.replicates}")
    units = {arm: stack_units([partition_units(lab, reference, nulls, options.nperm) for lab in parts])
             for arm, parts in labels.items()}
    outputs: Dict[str, str] = {}
    path = os.path.join(out_dir, "bench_partitions.npz")
    np.savez_compressed(path, genes=data["order"][mask].astype(str),
                        **{arm: np.vstack(parts) for arm, parts in labels.items()})
    outputs["bench_partitions"] = path
    outputs["bench_replicates"] = write_table(pd.DataFrame(records), os.path.join(out_dir, "bench_replicates.csv"),
                                              canonicalise=False)
    gate_arm = options.gate_arm
    level = {"alpha": options.alpha, "limit": options.limit}
    gate, share, passed = None, None, None
    if options.gate_copies > 0:
        copies = []
        for j in range(options.gate_copies):
            parts = [np.asarray(replicate(gate_arm, r, j)[0])[mask] for r in range(options.replicates)]
            copies.append(stack_units([partition_units(lab, reference, nulls, options.nperm) for lab in parts]))
            say(f"cluster_bench_test: gate copy {j + 1}/{options.gate_copies}")
        gate, share, passed = calibration_gate(units[gate_arm], copies, reference, **level,
                                               labels={"arm": gate_arm, "jitter": options.gate_noise})
        outputs["calibration_gate"] = write_table(gate, os.path.join(out_dir, "calibration_gate.csv"),
                                                  canonicalise=False)
    elif options.gate_record:
        gate = _gate_record(options.gate_record, options, gate_arm)
        share = gate.groupby("unit").p.apply(lambda p: float((p < options.alpha).mean())).to_dict()
        passed = all(share.get(u, 1.0) <= options.limit + 1e-9 for u in ("complexes", "STRING pairs"))
    tests = pd.DataFrame(paired_replicate_test(units["consensus"], units["phate_leiden"], reference,
                                               labels={"arm": "consensus", "comparator": "phate_leiden",
                                                       "k": k, "half": options.half}))
    margins, calibrated = None, {}
    if gate is not None and not passed:
        margin = noise_floor_margin(gate, ("unit",), options.margin_quantile, options.alpha, options.limit)
        outputs["calibration_margin"] = write_table(margin, os.path.join(out_dir, "calibration_margin.csv"),
                                                    canonicalise=False)
        margins = dict(zip(margin["unit"], margin["delta_pts"]))
        calibrated = dict(zip(margin["unit"], margin["passed"]))
    stats = call_effects(tests, margins, gated=passed, **{"alpha": options.alpha})
    stats["calibrated"] = [bool(calibrated.get(u, passed is not False)) for u in stats["unit"]]
    outputs["test_stats"] = write_table(stats, os.path.join(out_dir, "test_stats.csv"), canonicalise=False)
    outputs["bench_figure"] = _bench_figure(units, os.path.join(out_dir, "bench_test.png"))
    record = {"options": vars(options), "consensus": vars(consensus), "k": k, "genes": int(mask.sum()),
              "gate_passed": passed, "gate_false_positive_share": share,
              "calls": dict(zip(stats["unit"], stats["call"])), "outputs": outputs}
    path = os.path.join(out_dir, "bench_test_summary.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1, default=str)
    outputs["bench_test_summary"] = path
    say(f"cluster_bench_test: {record['calls']}")
    return {"folder": out_dir, "outputs": outputs, "gate_passed": passed, "calls": record["calls"]}
