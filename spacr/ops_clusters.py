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


def _sweep_figure(summary: pd.DataFrame, path: str) -> str:
    """Each sweep metric against the cluster count, real and shuffled.

    :param summary: from :func:`sweep_summary`.
    :param path: the PNG written.
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
        ``ops_profile_min_cells`` and ``ops_control_prefix``.
    :param genes: the profile phase's gene table, or None.
    :param folder: ``dst_root``; outputs go to its
        ``results/ops_clusters``.
    :param report: called with progress lines.
    :returns: ``{"folder", "outputs", "rows", "graphs"}``.
    :raises ValueError: when no rows or no reference folder are given, or a
        graph is unknown.
    :raises ImportError: when the ``clustering`` extra is not installed.
    """
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
    outputs["figure"] = _sweep_figure(summary, os.path.join(out_dir, "sweep.png"))
    return {"folder": out_dir, "outputs": outputs, "rows": int(len(names)), "graphs": tags}
