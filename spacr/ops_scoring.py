"""Shared community-standard scoring of OPS gene profiles and clusterings (item 913).

Gene-level results of different analyses of one screen (spaCR, Brieflow,
the published Funk et al. 2022 tables) are only comparable when the same
scoring code is applied to the same genes. This module applies the field's
two reference packages, unchanged, to every analysis at once:

* copairs (Kalinin et al. 2025): phenotypic ACTIVITY, the mAP of each gene's
  guides retrieving each other against the non-targeting guides, and
  phenotypic CONSISTENCY, the mAP of each CORUM complex and of each gene's
  reference partners;
* efaar_benchmarking (Celik et al. 2024): RECALL of known gene relationships
  in the outer tails of all gene-pair cosines.

What each input can be scored on is stated, never dropped: activity needs
guide-level profiles, profile-level consistency and EFAAR need one profile
per gene, and an analysis that published only clusters is scored on its
clusters, as one-hot rows. Every metric comes with its controls: shuffled
gene labels, non-targeting pseudo-genes, and non-targeting pseudo-complexes.

The packages are an optional extra (``spacr[scoring]``), pinned and never
vendored; copairs' null cache goes to the run folder.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: The pinned versions the reference tables were made with.
SCORING_PACKAGES = {"copairs": "0.6.0", "efaar_benchmarking": "1.0"}

#: The column copairs reads the reference index from.
_REF = "Metadata_Reference_Index"

#: The metrics of the scorability table, in order.
SCORING_METRICS = ("activity", "consistency_complexes", "consistency_pairs", "efaar")


def _require(name: str):
    """Import an optional scoring package or say how to install it."""
    import importlib

    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise ImportError(f"OPS shared scoring needs {name} {SCORING_PACKAGES.get(name.split('.')[0], '')}; "
                          "install it with pip install 'spacr[scoring]'") from error


def is_nt(names: Sequence[Any]) -> np.ndarray:
    """True for non-targeting rows and the pseudo-genes made from them.

    :param names: gene or row names.
    :returns: a boolean array; names starting with ``nontargeting`` or
        ``ntpseudo`` in any case.
    """
    return pd.Series(list(names)).astype(str).str.lower().str.startswith(("nontargeting", "ntpseudo")).to_numpy()


def onehot_rows(labels: Sequence[Any], seed: int = 0, eps: float = 1e-3) -> np.ndarray:
    """Cluster labels as one-hot rows plus a little Gaussian noise.

    :param labels: one cluster label per gene.
    :param seed: the noise seed of :func:`numpy.random.default_rng`.
    :param eps: the noise sd; 0 gives exact one-hot rows, whose cosine is
        exactly 1 within and 0 between clusters.
    :returns: float32 rows. With noise, cosine is about 1 within and 0
        between clusters and ties are broken at random, so copairs scores
        are averaged over several seeds.
    """
    uniques, inverse = np.unique(np.asarray(labels), return_inverse=True)
    rows = np.zeros((len(inverse), len(uniques)), np.float32)
    rows[np.arange(len(inverse)), inverse] = 1.0
    rng = np.random.default_rng(seed)
    return rows + rng.normal(0, eps, rows.shape).astype(np.float32)


def _features(table: pd.DataFrame, skip: Sequence[str]) -> List[str]:
    """The numeric profile columns of a table, in file order."""
    return [c for c in table.columns if c not in skip and pd.api.types.is_numeric_dtype(table[c])]


def load_scoring_inputs(inputs: Mapping[str, Mapping[str, str]], genes: Optional[Sequence[str]] = None
                        ) -> Dict[str, Any]:
    """Read every analysis to score and restrict it to the shared genes.

    :param inputs: ``{name: {"genes": path, "guides": path, "clusters":
        path}}``, any subset per name. A gene table has ``gene`` and numeric
        profile columns (non-targeting rows named ``nontargeting*`` are kept
        as controls); a guide table has ``gene`` and numeric columns
        (``guide`` and ``n_cells`` are not features); a cluster table has
        ``gene`` and ``cluster``.
    :param genes: the shared genes in order; the sorted intersection of the
        targeting genes of every input when None.
    :returns: ``{"genes", "profiles": {name: (rows, nt_names)},
        "guides": {name: (rows, gene)}, "clusters": {name: labels}}``;
        profile rows are the shared genes in order then the control rows in
        file order, guide rows those of shared genes and controls in file
        order.
    :raises ValueError: when an input lacks a shared gene or a column.
    """
    from .tabular import read_table

    tables: Dict[str, Dict[str, pd.DataFrame]] = {}
    for name, parts in inputs.items():
        tables[name] = {kind: read_table(path, canonicalise=False, report=None)
                        for kind, path in dict(parts).items() if path}
        for kind, table in tables[name].items():
            need = ["gene", "cluster"] if kind == "clusters" else ["gene"]
            if any(c not in table.columns for c in need):
                raise ValueError(f"{name} {kind}: the table needs columns {need}")
    if genes is None:
        sets = [set(t.gene.astype(str)[~is_nt(t.gene)]) for parts in tables.values() for t in parts.values()]
        genes = sorted(set.intersection(*sets)) if sets else []
    genes = [str(g) for g in genes]
    shared = set(genes)
    out: Dict[str, Any] = {"genes": np.array(genes), "profiles": {}, "guides": {}, "clusters": {}}
    for name, parts in tables.items():
        if "genes" in parts:
            table = parts["genes"].drop_duplicates("gene")
            table["gene"] = table.gene.astype(str)
            missing = shared - set(table.gene)
            if missing:
                raise ValueError(f"{name} genes: {len(missing)} shared genes have no profile")
            cols = _features(table, ("gene", "n_cells"))
            indexed = table.set_index("gene")
            nt = table.gene[is_nt(table.gene)].to_numpy()
            rows = np.vstack([indexed.loc[genes, cols].to_numpy(np.float32),
                              indexed.loc[nt, cols].to_numpy(np.float32)])
            out["profiles"][name] = (rows, nt)
        if "guides" in parts:
            table = parts["guides"]
            gene = table.gene.astype(str).to_numpy()
            keep = np.isin(gene, genes) | is_nt(gene)
            cols = _features(table, ("gene", "guide", "n_cells"))
            out["guides"][name] = (table.loc[keep, cols].to_numpy(np.float32), gene[keep])
        if "clusters" in parts:
            table = parts["clusters"].drop_duplicates("gene")
            table["gene"] = table.gene.astype(str)
            missing = shared - set(table.gene)
            if missing:
                raise ValueError(f"{name} clusters: {len(missing)} shared genes have no cluster")
            out["clusters"][name] = table.set_index("gene").loc[genes, "cluster"].to_numpy()
    return out


def scorability_table(data: Mapping[str, Any], names: Sequence[str]) -> pd.DataFrame:
    """Which input can be scored on which metric, and why not.

    :param data: :func:`load_scoring_inputs` output.
    :param names: the input names.
    :returns: one row per input, level and metric with ``scorable``
        ("yes" or "not scorable") and ``reason``.
    """
    rows = []
    for name in names:
        has = {"guides": name in data["guides"], "profile": name in data["profiles"],
               "clusters": name in data["clusters"]}
        for level in ("profile", "clusters"):
            if level == "clusters" and not has["clusters"]:
                continue
            for metric in SCORING_METRICS:
                if metric == "activity":
                    ok, why = (has["guides"], "" if has["guides"] else "no guide-level profiles")
                    if level == "clusters":
                        ok, why = False, "clusters only: activity needs guide-level profiles"
                elif level == "profile":
                    ok = has["profile"]
                    why = "" if ok else ("clusters only: no profile per gene" if has["clusters"]
                                         else "no profile per gene")
                else:
                    ok, why = True, ""
                rows.append({"input": name, "level": level, "metric": metric,
                             "scorable": "yes" if ok else "not scorable", "reason": why})
    return pd.DataFrame(rows)


def activity_map(features: np.ndarray, gene: Sequence[str], is_control: np.ndarray, tag: str, *,
                 null_size: int = 10000, workers: int = 4, cache_dir: Optional[str] = None
                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """copairs phenotypic activity of every gene's guides against the controls.

    :param features: one row per guide.
    :param gene: the gene label of each row (``"NT"`` for controls).
    :param is_control: True for the control rows.
    :param tag: the ``set`` the rows are labelled with.
    :param null_size: permutations of the null per configuration.
    :param workers: copairs worker threads.
    :param cache_dir: copairs' null cache.
    :returns: ``(ap, map)``: per-guide AP and per-gene mAP with p, BH q and
        the flags at 0.05; ``assign_reference_index`` on the controls,
        ``pos_sameby=[gene, reference]``, ``neg_diffby=[ctrl]``, cosine,
        ``mean_average_precision(seed=0)``.
    """
    cmap = _require("copairs.map")
    matching = _require("copairs.matching")
    meta = pd.DataFrame({"gene": list(gene), "ctrl": np.asarray(is_control, bool)})
    meta = matching.assign_reference_index(meta, meta.index[meta.ctrl], reference_col=_REF)
    ap = cmap.average_precision(meta, features, pos_sameby=["gene", _REF], pos_diffby=[], neg_sameby=[],
                                neg_diffby=["ctrl"], batch_size=20000, progress_bar=False)
    ap = ap[~ap.ctrl]
    result = cmap.mean_average_precision(ap, ["gene"], null_size=null_size, threshold=0.05, seed=0,
                                         progress_bar=False, max_workers=workers, cache_dir=cache_dir)
    result["set"] = tag
    return pd.DataFrame(ap.drop(columns=["ctrl", _REF], errors="ignore")).assign(set=tag), result


def score_activity(features: np.ndarray, gene: Sequence[str], *, null_size: int = 10000,
                   workers: int = 4, cache_dir: Optional[str] = None, label: str = "genes",
                   nt_seeds: Sequence[int] = (1000, 1001, 1002, 1003, 1004), shuffle_seed: int = 7
                   ) -> pd.DataFrame:
    """Activity of real genes with its two controls, as one table.

    :param features: one row per guide.
    :param gene: each guide's gene; controls start with ``nontargeting``.
    :param null_size: see :func:`activity_map`.
    :param workers: see :func:`activity_map`.
    :param cache_dir: see :func:`activity_map`.
    :param label: the set name of the real genes.
    :param nt_seeds: one NT pseudo-gene control per seed: half the control
        guides, permuted, in fours, scored against the other half.
    :param shuffle_seed: the targeting guides' gene labels permuted (group
        sizes kept) and scored against the same controls.
    :returns: the per-gene mAP rows of every set.
    """
    gene = np.asarray(gene).astype(str)
    nt = is_nt(gene)
    rows = [activity_map(features, np.where(nt, "NT", gene), nt, label, null_size=null_size,
                         workers=workers, cache_dir=cache_dir)[1]]
    nt_index = np.flatnonzero(nt)
    for k, seed in enumerate(nt_seeds):
        perm = np.random.default_rng(seed).permutation(nt_index)
        take = len(perm) // 2 // 4 * 4
        decoys, controls = perm[:take], perm[take:]
        labels = np.array(["NT"] * len(gene), object)
        labels[decoys] = [f"ntpseudo_{k}_{i // 4}" for i in range(take)]
        chosen = np.r_[decoys, controls]
        rows.append(activity_map(features[chosen], labels[chosen], np.isin(chosen, controls),
                                 f"NT pseudo-genes s{k}", null_size=null_size, workers=workers,
                                 cache_dir=cache_dir)[1])
    labels = np.where(nt, "NT", gene).astype(object)
    targeting = np.flatnonzero(~nt)
    labels[targeting] = labels[targeting][np.random.default_rng(shuffle_seed).permutation(len(targeting))]
    rows.append(activity_map(features, labels, nt, "shuffled gene labels", null_size=null_size,
                             workers=workers, cache_dir=cache_dir)[1])
    return pd.concat(rows, ignore_index=True)


def reference_complexes(genes: Sequence[str], reference_dir: Any, *, half: str = "REPORT",
                        min_members: int = 3) -> Dict[str, List[int]]:
    """CORUM complexes of one selection half with enough shared genes.

    :param genes: the shared genes in row order.
    :param reference_dir: the folder of the references
        (:mod:`spacr.ops_reference`).
    :param half: ``'SELECT'`` or ``'REPORT'``.
    :param min_members: the fewest member rows a complex needs.
    :returns: ``{complex: sorted row indices}``.
    """
    from .ops_reference import gene_symbol, group_halves, reference_groups

    groups = reference_groups("CORUM", reference_dir)
    halves = group_halves(groups)
    wanted = {"SELECT": 0, "REPORT": 1}[half.upper()]
    index = {gene_symbol(g): i for i, g in enumerate(genes)}
    out = {}
    for name, members in groups.items():
        if halves[name] != wanted:
            continue
        rows = sorted({index[m] for m in members if m in index})
        if len(rows) >= min_members:
            out[name] = rows
    return out


def complex_map(features: np.ndarray, complexes: Mapping[str, Sequence[int]], tag: str, seed: int = 0, *,
                null_size: int = 10000, workers: int = 4, cache_dir: Optional[str] = None) -> pd.DataFrame:
    """copairs multilabel consistency mAP of every complex.

    :param features: one row per gene (and control rows).
    :param complexes: ``{complex: row indices}``.
    :param tag: the representation name (``rep``).
    :param seed: the null seed.
    :param null_size: see :func:`activity_map`.
    :param workers: see :func:`activity_map`.
    :param cache_dir: see :func:`activity_map`.
    :returns: per complex the mAP, p, q, flags, ``rep`` and ``n_members``.
    """
    cmap = _require("copairs.map")
    multilabel = _require("copairs.map.multilabel")
    labels: List[List[str]] = [[] for _ in range(len(features))]
    for name, rows in complexes.items():
        for i in rows:
            labels[i].append(name)
    rows = np.array([i for i in range(len(features)) if labels[i]])
    meta = pd.DataFrame({"complexes": [labels[i] for i in rows]})
    ap = multilabel.average_precision(meta, features[rows].astype(np.float32), pos_sameby=["complexes"],
                                      pos_diffby=[], neg_sameby=[], neg_diffby=["complexes"],
                                      multilabel_col="complexes", progress_bar=False)
    result = cmap.mean_average_precision(ap, ["complexes"], null_size=null_size, threshold=0.05, seed=seed,
                                         progress_bar=False, max_workers=workers, cache_dir=cache_dir)
    result = result.rename(columns={"complexes": "complex"})
    result["rep"] = tag
    result["n_members"] = result.complex.map({c: len(v) for c, v in complexes.items()})
    return pd.DataFrame(result)


def pair_ap(features: np.ndarray, positives: np.ndarray, tag: str, seed: int = 0, *,
            null_size: int = 10000, cache_dir: Optional[str] = None) -> pd.DataFrame:
    """Per-gene AP of retrieving its reference partners, with copairs' kernels.

    :param features: one row per gene.
    :param positives: the boolean partner matrix.
    :param tag: the representation name.
    :param seed: the null seed.
    :param null_size: null draws per ``(positives, total)`` configuration.
    :param cache_dir: copairs' null cache.
    :returns: per query gene ``gene_idx``, ``ap``, ``n_pos``, ``n_total``,
        ``random_ap``, ``p`` and BH ``q``. Queries and candidates are the
        genes with at least one partner; ties keep row order.
    """
    compute = _require("copairs.compute")
    from statsmodels.stats.multitest import multipletests

    candidates = np.flatnonzero(positives.any(1))
    x = features[candidates].astype(np.float64)
    x /= np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    sims = (x @ x.T).astype(np.float32)
    sub = positives[np.ix_(candidates, candidates)]
    relevance, counts = [], []
    for i in range(len(candidates)):
        others = np.r_[0:i, i + 1:len(candidates)]
        order = others[np.argsort(-sims[i, others], kind="stable")]
        relevance.append(sub[i, order])
        counts.append(len(others))
    ap, configs = compute.ap_contiguous(np.concatenate(relevance).astype(np.uint8), np.array(counts))
    uniques, inverse = np.unique(configs, axis=0, return_inverse=True)
    nulls = compute.get_null_dists(uniques, null_size, seed, cache_dir=cache_dir, progress_bar=False)
    nulls.sort(axis=1)
    p = np.array([(null_size - np.searchsorted(nulls[j], a) + 1) / (null_size + 1)
                  for a, j in zip(ap, inverse.ravel())])
    random_ap = np.array([u / t for u, t in configs])
    q = multipletests(p, method="fdr_bh")[1]
    return pd.DataFrame({"gene_idx": candidates, "ap": ap, "n_pos": configs[:, 0], "n_total": configs[:, 1],
                         "random_ap": random_ap, "p": p, "q": q, "rep": tag})


def _representations(data: Mapping[str, Any], cluster_seeds: Sequence[int], jitter: float
                     ) -> Dict[str, List[Tuple[int, np.ndarray]]]:
    """``{rep: [(seed, rows)]}`` on the shared genes for copairs."""
    n = len(data["genes"])
    reps: Dict[str, List[Tuple[int, np.ndarray]]] = {}
    for name, (rows, _nt) in data["profiles"].items():
        reps[f"{name} profile"] = [(0, rows[:n])]
    for name, labels in data["clusters"].items():
        reps[f"{name} clusters"] = [(s, onehot_rows(labels, s, jitter)) for s in cluster_seeds]
    return reps


def score_consistency(data: Mapping[str, Any], reference_dir: Any, *, references: Sequence[str],
                      min_members: int = 3, null_size: int = 10000, workers: int = 4,
                      cluster_seeds: Sequence[int] = (0, 1, 2), jitter: float = 1e-3,
                      shuffle_seed: int = 7, complex_seeds: Sequence[int] = (500, 501, 502),
                      pair_seeds: Sequence[int] = (900, 901, 902), cache_dir: Optional[str] = None
                      ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Complex and partner consistency of every representation, with controls.

    :param data: :func:`load_scoring_inputs` output.
    :param reference_dir: the reference folder.
    :param references: the pair references of the per-gene retrieval, such
        as ``CORUM@REPORT`` and ``STRING>=0.7@REPORT``.
    :param min_members: the fewest shared genes of a REPORT-half complex.
    :param null_size: see :func:`activity_map`.
    :param workers: see :func:`activity_map`.
    :param cluster_seeds: noise seeds of the one-hot cluster rows.
    :param jitter: the noise sd of the cluster rows.
    :param shuffle_seed: the gene-label permutation of the shuffled control.
    :param complex_seeds: seeds of the NT pseudo-complexes (control rows
        grouped like the real complexes' sizes).
    :param pair_seeds: seeds of the NT pseudo-genes of partner retrieval (a
        control row takes a real gene's partner list).
    :param cache_dir: copairs' null cache.
    :returns: ``(complex_map, pairs_per_gene, setup)``.
    """
    from .ops_reference import positive_matrix, reference_pairs

    genes = data["genes"]
    n = len(genes)
    reps = _representations(data, cluster_seeds, jitter)
    setup: Dict[str, Any] = {"null_size": null_size, "min_members": min_members,
                             "seeds_clusters": list(cluster_seeds)}
    complexes = reference_complexes(genes, reference_dir, min_members=min_members)
    setup["complexes_REPORT_ge3"] = len(complexes)
    rows = []
    for tag, items in reps.items():
        for seed, X in items:
            rows.append(complex_map(X, complexes, tag, seed, null_size=null_size, workers=workers,
                                    cache_dir=cache_dir).assign(seed=seed, control="none"))
        X = items[0][1]
        rows.append(complex_map(X[np.random.default_rng(shuffle_seed).permutation(n)], complexes, tag,
                                null_size=null_size, workers=workers, cache_dir=cache_dir).assign(
            seed=0, control="shuffled gene labels"))
    sizes = np.array(sorted(len(v) for v in complexes.values()))
    for name, (Xa, _nt) in data["profiles"].items():
        for k, seed in enumerate(cluster_seeds):
            rng = np.random.default_rng(complex_seeds[k])
            nt_rows = rng.permutation(np.arange(n, len(Xa)))
            decoys, pos = {}, 0
            for j, size in enumerate(rng.permutation(sizes)):
                if pos + size > len(nt_rows):
                    break
                decoys[f"decoy_{j}"] = list(nt_rows[pos:pos + size])
                pos += size
            result = complex_map(Xa, {**complexes, **decoys}, f"{name} profile", seed, null_size=null_size,
                                 workers=workers, cache_dir=cache_dir)
            rows.append(result[result.complex.str.startswith("decoy_")].assign(
                seed=seed, control="NT pseudo-complexes"))
    complex_table = pd.concat(rows, ignore_index=True)
    rows = []
    for ref in references:
        positives = positive_matrix(genes, reference_pairs(ref, reference_dir))
        setup[f"{ref}_genes_with_partner"] = int(positives.any(1).sum())
        setup[f"{ref}_pairs"] = int(positives.sum() // 2)
        for tag, items in reps.items():
            for seed, X in items:
                rows.append(pair_ap(X, positives, tag, seed, null_size=null_size, cache_dir=cache_dir).assign(
                    seed=seed, control="none", reference=ref))
            X = items[0][1]
            rows.append(pair_ap(X[np.random.default_rng(shuffle_seed).permutation(n)], positives, tag,
                                null_size=null_size, cache_dir=cache_dir).assign(
                seed=0, control="shuffled gene labels", reference=ref))
        for name, (Xa, _nt) in data["profiles"].items():
            for k, seed in enumerate(cluster_seeds):
                rng = np.random.default_rng(pair_seeds[k])
                has = np.flatnonzero(positives.any(1))
                pick = rng.choice(has, size=min(len(has), len(Xa) - n), replace=False)
                nt_rows = rng.permutation(np.arange(n, len(Xa)))[:len(pick)]
                X = np.vstack([Xa[:n], Xa[nt_rows]])
                grown = np.zeros((len(X), len(X)), bool)
                grown[:n, :n] = positives
                for g, r in zip(pick, range(n, len(X))):
                    grown[r, :n] = positives[g]
                    grown[:n, r] = positives[g]
                result = pair_ap(X, grown, f"{name} profile", seed, null_size=null_size, cache_dir=cache_dir)
                rows.append(result[result.gene_idx >= n].assign(seed=seed, control="NT pseudo-genes",
                                                                reference=ref))
    pairs = pd.concat(rows, ignore_index=True)
    pairs["gene"] = [genes[i] if i < n else f"ntpseudo_{i}" for i in pairs.gene_idx]
    return complex_table, pairs, setup


def pair_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    """Partner retrieval per reference, representation, control and seed.

    :param pairs: :func:`score_consistency`'s per-gene table.
    :returns: genes, mAP, random mAP and the share of genes at q < 0.05.
    """
    return pairs.groupby(["reference", "rep", "control", "seed"]).agg(
        genes=("ap", "size"), mAP=("ap", "mean"), random_mAP=("random_ap", "mean"),
        frac_q_lt_0p05=("q", lambda v: float((v < 0.05).mean()))).reset_index()


def _cosine(X: np.ndarray) -> np.ndarray:
    """All-pairs cosine of the rows, float64."""
    x = X.astype(np.float64)
    x /= np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    return x @ x.T


def efaar_tails(X: np.ndarray, genes: Sequence[str], source: str) -> Dict[str, np.ndarray]:
    """Per known pair of one EFAAR source: in the outer tails at each threshold.

    :param X: one row per gene.
    :param genes: the gene names.
    :param source: an EFAAR benchmark source.
    :returns: ``{"lo_hi": bool per pair}``, EFAAR's ``compute_recall`` kept
        per pair, for the paired tests.
    """
    benchmarking = _require("efaar_benchmarking.benchmarking")
    constants = _require("efaar_benchmarking.constants")
    sims = _cosine(X)
    null = np.sort(sims[np.triu_indices(len(sims), 1)])
    index = {g: i for i, g in enumerate(genes)}
    rels = benchmarking.get_benchmark_relationships(constants.BENCHMARK_DATA_DIR, source)
    rels = rels[rels.entity1.isin(index) & rels.entity2.isin(index)]
    q = sims[rels.entity1.map(index).to_numpy(), rels.entity2.map(index).to_numpy()]
    left = np.searchsorted(null, q, side="left") / len(null)
    right = np.searchsorted(null, q, side="right") / len(null)
    return {f"{lo}_{hi}": (right <= lo) | (left >= hi) for lo, hi in constants.RECALL_PERC_THRS}


def _mcnemar(a: np.ndarray, b: np.ndarray) -> Tuple[int, int, float]:
    """Discordant counts and the exact McNemar (binomial) p."""
    from scipy.stats import binomtest

    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    return n10, n01, (binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0)


def score_efaar(data: Mapping[str, Any], *, shuffle_seeds: Sequence[int] = (70, 71, 72, 73, 74),
                nt_seeds: Sequence[int] = (300, 301, 302)) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """EFAAR known-relationship recall of every representation, with controls.

    :param data: :func:`load_scoring_inputs` output.
    :param shuffle_seeds: one shuffled-label run per seed.
    :param nt_seeds: one NT pseudo-gene control per seed: each known pair
        ``(a, b)`` becomes ``(control row, b)``, against the same all-gene
        null (profiles only).
    :returns: ``(recall_all, recall_summary, mcnemar)``. Clusters are exact
        one-hot rows, so their high tail is exactly the co-clustered pairs;
        compare clusters with clusters only. McNemar pairs every two
        profiles and every two clusterings on the same known pairs.
    """
    benchmarking = _require("efaar_benchmarking.benchmarking")
    constants = _require("efaar_benchmarking.constants")
    from sklearn.utils import Bunch

    genes = [str(g) for g in data["genes"]]
    n = len(genes)

    def bunch(X):
        """The EFAAR input of one representation."""
        return Bunch(features=pd.DataFrame(X), metadata=pd.DataFrame({"gene": genes}))

    reps = {f"{k} profile": rows[:n] for k, (rows, _nt) in data["profiles"].items()}
    reps.update({f"{k} clusters": onehot_rows(v, 0, eps=0.0) for k, v in data["clusters"].items()})
    rows, per_pair = [], {}
    for tag, X in reps.items():
        result = benchmarking.known_relationship_benchmark(bunch(X), "gene", log_stats=False)
        rows.append(result.assign(rep=tag, seed=0, control="none"))
        for source in constants.BENCHMARK_SOURCES:
            for thr, hit in efaar_tails(X, genes, source).items():
                per_pair[(tag, source, thr)] = hit
        for k, seed in enumerate(shuffle_seeds):
            perm = np.random.default_rng(seed).permutation(n)
            result = benchmarking.known_relationship_benchmark(bunch(X[perm]), "gene")
            rows.append(result.assign(rep=tag, seed=k, control="shuffled gene labels"))
    for name, (Xa, _nt) in data["profiles"].items():
        Xn = Xa.astype(np.float64)
        Xn = Xn / np.clip(np.linalg.norm(Xn, axis=1, keepdims=True), 1e-12, None)
        sims = Xn[:n] @ Xn[:n].T
        null = np.sort(sims[np.triu_indices(n, 1)])
        index = {g: i for i, g in enumerate(genes)}
        for k, seed in enumerate(nt_seeds):
            rng = np.random.default_rng(seed)
            record = {"rep": f"{name} profile", "seed": k, "control": "NT pseudo-genes"}
            for source in constants.BENCHMARK_SOURCES:
                rel = benchmarking.get_benchmark_relationships(constants.BENCHMARK_DATA_DIR, source)
                rel = rel[rel.entity1.isin(index) & rel.entity2.isin(index)]
                b = rel.entity2.map(index).to_numpy()
                controls = rng.integers(n, len(Xa), len(b))
                q = (Xn[controls] * Xn[b]).sum(1)
                left = np.searchsorted(null, q, side="left") / len(null)
                right = np.searchsorted(null, q, side="right") / len(null)
                row = {"source": source, "null_distribution_size": len(null), "query_distribution_size": len(q)}
                for lo, hi in constants.RECALL_PERC_THRS:
                    row[f"recall_{lo}_{hi}"] = float(((right <= lo) | (left >= hi)).mean())
                rows.append(pd.DataFrame([{**row, **record}]))
    recall = pd.concat(rows, ignore_index=True)
    summary = recall.groupby(["rep", "control", "source"]).agg(
        pairs=("query_distribution_size", "first"),
        recall_0p05_0p95=("recall_0.05_0.95", "mean"), recall_0p05_0p95_sd=("recall_0.05_0.95", "std"),
        recall_0p1_0p9=("recall_0.1_0.9", "mean"), recall_0p1_0p9_sd=("recall_0.1_0.9", "std"),
        seeds=("seed", "nunique")).reset_index()
    profiles = [r for r in reps if r.endswith(" profile")]
    clusters = [r for r in reps if r.endswith(" clusters")]
    pairs = ([(profiles[i], profiles[j]) for i in range(len(profiles)) for j in range(i + 1, len(profiles))]
             + [(clusters[i], clusters[j]) for i in range(len(clusters)) for j in range(i + 1, len(clusters))])
    tests = []
    for source in constants.BENCHMARK_SOURCES:
        for thr in [f"{lo}_{hi}" for lo, hi in constants.RECALL_PERC_THRS]:
            for a, b in pairs:
                va, vb = per_pair[(a, source, thr)], per_pair[(b, source, thr)]
                n10, n01, p = _mcnemar(va, vb)
                tests.append({"source": source, "threshold": thr, "a": a, "b": b, "recall_a": float(va.mean()),
                              "recall_b": float(vb.mean()), "a_only": n10, "b_only": n01, "p_mcnemar": p,
                              "pairs": len(va)})
    return recall, summary, pd.DataFrame(tests)


def _holm(pvalues: Mapping[str, float]) -> Dict[str, float]:
    """Holm-corrected p within one panel."""
    from statsmodels.stats.multitest import multipletests

    keys = list(pvalues)
    q = multipletests([pvalues[k] for k in keys], method="holm")[1] if keys else []
    return dict(zip(keys, map(float, q)))


def summary_and_stats(data: Mapping[str, Any], activity: Optional[pd.DataFrame], complexes: pd.DataFrame,
                      pairs: pd.DataFrame, efaar: pd.DataFrame, references: Sequence[str],
                      mcnemar: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """The main summary table and the paired tests, Holm within each panel.

    :param data: :func:`load_scoring_inputs` output.
    :param activity: the activity table, or None when no input has guides.
    :param complexes: the complex table.
    :param pairs: the per-gene partner table.
    :param efaar: the EFAAR recall table.
    :param references: the partner references.
    :param mcnemar: :func:`score_efaar`'s paired tests; with them the
        10 % / 90 % profile tests get their own panel, Holm over the sources.
    :returns: ``(summary, stats)``. Activity: Fisher exact of active counts
        against each control. Complexes: exact McNemar over the same
        complexes (seed 0). Partners: paired Wilcoxon over the same genes.
        EFAAR: exact McNemar over the same known pairs (5 % / 95 % tails),
        per source for profiles and on the union of sources for clusters.
    """
    from scipy.stats import fisher_exact, wilcoxon

    benchmarking = _require("efaar_benchmarking.benchmarking")
    constants = _require("efaar_benchmarking.constants")
    genes = [str(g) for g in data["genes"]]
    n = len(genes)
    sources = constants.BENCHMARK_SOURCES
    stats: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    guide_inputs = list(data["guides"])
    if activity is not None and len(activity):
        for name in guide_inputs:
            part = activity[activity.input == name] if "input" in activity else activity
            real = part[part.set == f"{name} genes"] if (part.set == f"{name} genes").any() else \
                part[~part.set.str.startswith(("NT pseudo", "shuffled"))]
            shuffled = part[part.set == "shuffled gene labels"]
            nt = part[part.set.str.startswith("NT pseudo")]

            def table(d):
                """Active and inactive counts of one set."""
                return [d.below_corrected_p.sum(), (~d.below_corrected_p.astype(bool)).sum()]

            p = {f"{name} vs shuffled": fisher_exact([table(real), table(shuffled)])[1],
                 f"{name} vs NT pseudo-genes": fisher_exact([table(real), table(nt)])[1]}
            stats["activity"] = {"p": p, "q_holm": _holm(p), "test": "Fisher exact on active / not active counts"}
            for label, d in ((name, real), ("Shuffled gene labels", shuffled), ("NT pseudo-genes", nt)):
                rows.append({"metric": "Phenotypic activity: genes active (q<0.05), %",
                             "level": "guides vs NT guides", "pipeline": label,
                             "value": 100 * d.below_corrected_p.mean(), "n": len(d),
                             "mAP_mean": d.mean_average_precision.mean(), "p_lt_0.05_pct": 100 * d.below_p.mean()})
    for name in sorted(set(data["profiles"]) | set(data["clusters"])):
        if name not in data["guides"]:
            rows.append({"metric": "Phenotypic activity: genes active (q<0.05), %", "level": "guides vs NT guides",
                         "pipeline": name, "value": np.nan, "n": 0,
                         "note": "not scorable: no guide-level profiles published"})
    profiles = [f"{k} profile" for k in data["profiles"]]
    clusters = [f"{k} clusters" for k in data["clusters"]]

    def fraction(rep, control="none"):
        """Share of complexes at q < 0.05 per seed, and the rows."""
        d = complexes[(complexes.rep == rep) & (complexes.control == control)]
        return d.groupby("seed").below_corrected_p.mean() * 100, d

    def flags(rep, control="none"):
        """Seed-0 significance per complex, sorted by complex."""
        d = complexes[(complexes.rep == rep) & (complexes.control == control) & (complexes.seed == 0)]
        return d.set_index("complex").sort_index().below_corrected_p.astype(bool)

    for rep in profiles + clusters:
        per_seed, d = fraction(rep)
        shuffled, _ = fraction(rep, "shuffled gene labels")
        rows.append({"metric": "Consistency: CORUM complexes retrieved (q<0.05), %", "level": rep.split()[-1],
                     "pipeline": rep.rsplit(" ", 1)[0], "value": per_seed.mean(),
                     "n": int(d.groupby("seed").size().iloc[0]), "mAP_mean": d.mean_average_precision.mean(),
                     "shuffled_value": shuffled.mean()})
        if rep in profiles:
            rows[-1]["NT_pseudo_value"] = fraction(rep, "NT pseudo-complexes")[0].mean()
    for tag, reps in (("complex_profile", profiles), ("complex_clusters", clusters)):
        p = {}
        for i in range(len(reps)):
            for j in range(i + 1, len(reps)):
                a = flags(reps[i])
                p[f"{reps[i]} vs {reps[j]}"] = _mcnemar(a, flags(reps[j]).loc[a.index])[2]
        for r in reps:
            a = flags(r)
            p[f"{r} vs shuffled"] = _mcnemar(a, flags(r, "shuffled gene labels").loc[a.index])[2]
        stats[tag] = {"p": p, "q_holm": _holm(p), "test": "exact McNemar over the same CORUM complexes (seed 0)"}

    def per_gene(rep, ref, control="none"):
        """Seed-averaged AP per gene of one representation and reference."""
        d = pairs[(pairs.rep == rep) & (pairs.reference == ref) & (pairs.control == control)]
        return d.groupby("gene").agg(ap=("ap", "mean"), random_ap=("random_ap", "first"),
                                     sig=("q", lambda v: float((v < 0.05).mean())))

    for tag, reps in (("retrieval_profile", profiles), ("retrieval_clusters", clusters)):
        p = {}
        for ref in references:
            for i in range(len(reps)):
                for j in range(i + 1, len(reps)):
                    a, b = per_gene(reps[i], ref), per_gene(reps[j], ref)
                    p[f"{ref}: {reps[i]} vs {reps[j]}"] = float(wilcoxon(a.ap, b.loc[a.index].ap).pvalue)
            for r in reps:
                a = per_gene(r, ref)
                rows.append({"metric": f"Consistency: retrieval mAP of {ref.split('@')[0]} partners",
                             "level": r.split()[-1], "pipeline": r.rsplit(" ", 1)[0], "value": a.ap.mean(),
                             "n": len(a), "random_mAP": a.random_ap.mean(),
                             "fold_over_random": a.ap.mean() / a.random_ap.mean(),
                             "genes_q_lt_0.05_pct": 100 * a.sig.mean(),
                             "shuffled_value": per_gene(r, ref, "shuffled gene labels").ap.mean()})
                if r in profiles:
                    d = pairs[(pairs.rep == r) & (pairs.reference == ref) & (pairs.control == "NT pseudo-genes")]
                    rows[-1]["NT_pseudo_value"] = d.ap.mean()
                    rows[-1]["NT_pseudo_fold"] = d.ap.mean() / d.random_ap.mean()
                    rows[-1]["NT_pseudo_q_lt_0.05_pct"] = 100 * (d.q < 0.05).mean()
        stats[tag] = {"p": p, "q_holm": _holm(p),
                      "test": "paired Wilcoxon signed-rank over the same genes (per-gene AP)"}
    index = {g: i for i, g in enumerate(genes)}
    reps_x = {f"{k} profile": rows_[:n] for k, (rows_, _nt) in data["profiles"].items()}
    reps_x.update({f"{k} clusters": onehot_rows(v, 0, eps=0.0) for k, v in data["clusters"].items()})
    tails: Dict[str, Dict[str, np.ndarray]] = {}
    for rep, X in reps_x.items():
        sims = _cosine(X)
        null = np.sort(sims[np.triu_indices(n, 1)])
        known, stacked = {}, []
        for source in sources:
            rel = benchmarking.get_benchmark_relationships(constants.BENCHMARK_DATA_DIR, source)
            rel = rel[rel.entity1.isin(index) & rel.entity2.isin(index)]
            a, b = rel.entity1.map(index).to_numpy(), rel.entity2.map(index).to_numpy()
            known[source] = (np.minimum(a, b), np.maximum(a, b))
            stacked.append(np.c_[np.minimum(a, b), np.maximum(a, b)])
        union = np.unique(np.vstack(stacked), axis=0)
        known["union"] = (union[:, 0], union[:, 1])
        tails[rep] = {}
        for key, (a, b) in known.items():
            q = sims[a, b]
            left = np.searchsorted(null, q, side="left") / len(null)
            right = np.searchsorted(null, q, side="right") / len(null)
            tails[rep][key] = (right <= 0.05) | (left >= 0.95)
    for tag, reps, keys in (("efaar_profile", profiles, sources), ("efaar_clusters", clusters, ["union"])):
        p = {}
        for key in keys:
            for i in range(len(reps)):
                for j in range(i + 1, len(reps)):
                    p[f"{key}: {reps[i]} vs {reps[j]}"] = _mcnemar(tails[reps[i]][key], tails[reps[j]][key])[2]
        stats[tag] = {"p": p, "q_holm": _holm(p),
                      "test": "exact McNemar over the same known pairs (5 % / 95 % tails)",
                      "union_pairs": int(len(tails[reps[0]]["union"])) if reps else 0}
    if mcnemar is not None and len(profiles) > 1:
        for i in range(len(profiles)):
            for j in range(i + 1, len(profiles)):
                part = mcnemar[(mcnemar.threshold == "0.1_0.9") & (mcnemar.a == profiles[i])
                               & (mcnemar.b == profiles[j])].set_index("source")
                p = {src: float(part.loc[src, "p_mcnemar"]) for src in sources if src in part.index}
                tag = "efaar_profile_10pct" if len(profiles) == 2 else \
                    f"efaar_profile_10pct: {profiles[i]} vs {profiles[j]}"
                stats[tag] = {"p": p, "q_holm": _holm(p),
                              "test": "exact McNemar over the same known pairs (10 % / 90 % tails)"}
    for r in profiles + clusters:
        d = efaar[(efaar.rep == r) & (efaar.control == "none")].groupby("source")["recall_0.05_0.95"].mean()
        shuffled = efaar[(efaar.rep == r) & (efaar.control == "shuffled gene labels")].groupby(
            "source")["recall_0.05_0.95"].mean()
        row = {"metric": "EFAAR recall, 5 %/95 % tails (mean of 5 sources)", "level": r.split()[-1],
               "pipeline": r.rsplit(" ", 1)[0], "value": d.mean(), "n": 5, "shuffled_value": shuffled.mean(),
               "random_expectation": 0.10, "union_recall": float(tails[r]["union"].mean()),
               **{f"recall_{s}": d[s] for s in sources}}
        if r in profiles:
            row["NT_pseudo_value"] = efaar[(efaar.rep == r) & (efaar.control == "NT pseudo-genes")][
                "recall_0.05_0.95"].mean()
        rows.append(row)
    return pd.DataFrame(rows), stats


def _figures(summary: pd.DataFrame, stats: Mapping[str, Any], folder: str) -> List[str]:
    """Two bar figures (profiles, clusters) with jittered points and Holm brackets."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = ["#2a9d8f", "#1f3a5f", "#4a90d9", "#9cc3e6"]
    grey = "#8d99ae"
    rng = np.random.default_rng(0)
    written = []
    for level, key in (("profile", "complex_profile"), ("clusters", "complex_clusters")):
        part = summary[(summary.level == level) & summary.metric.str.startswith(("Consistency", "EFAAR"))]
        if not len(part):
            continue
        metrics = list(dict.fromkeys(part.metric))
        fig, axes = plt.subplots(1, len(metrics), figsize=(3.0 * len(metrics), 3.0), squeeze=False)
        for ax, metric in zip(axes[0], metrics):
            d = part[part.metric == metric].reset_index(drop=True)
            for i, row in d.iterrows():
                colour = palette[i % len(palette)]
                ax.bar(i, row.value, 0.7, color=colour, alpha=0.6, edgecolor=colour)
                ax.scatter(i + rng.uniform(-0.15, 0.15), row.value, s=6, color=colour, alpha=0.6,
                           edgecolor="black", linewidth=0.3)
            shuffled = d.get("shuffled_value")
            if shuffled is not None:
                ax.bar(len(d), float(np.nanmean(shuffled)), 0.7, color=grey, alpha=0.6, edgecolor=grey)
            ax.set_xticks(range(len(d) + 1), list(d.pipeline) + ["Shuffled"], fontsize=7)
            ax.set_title(metric, fontsize=7)
            top = float(np.nanmax(d.value)) if len(d) else 1.0
            q = dict(stats.get(key, {}).get("q_holm", {}))
            for k, (pair, value) in enumerate(q.items()):
                if " vs shuffled" in pair:
                    continue
                stars = "***" if value < 0.001 else "**" if value < 0.01 else "*" if value < 0.05 else "ns"
                ax.text(0.5, top * (1.05 + 0.08 * k), f"{pair}: {stars}", fontsize=5.5, ha="left")
            ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        path = os.path.join(folder, f"fig_shared_{level}.png")
        fig.savefig(path, dpi=200, facecolor="white")
        plt.close(fig)
        written.append(path)
    return written


#: The profile-benchmark metrics of item 923, run by :func:`score_profile_benchmarks`.
BENCHMARK_METRICS = ("corum_within_r", "string_auprc", "hits_mw_ztpm")

#: The phase-score metrics of item 913, run when ``ops_scoring_metrics`` is empty.
SHARED_METRICS = ("activity", "consistency", "efaar")

#: PERISCOPE's compartments (Ramezani et al. 2025, Figure 4 notebook): a
#: feature belongs to every compartment whose name it contains, in any case.
PERISCOPE_COMPARTMENTS: Dict[str, str] = {"Mito": "mito", "ConA": "cona", "WGA": "wga",
                                          "DAPI": "dapi", "Phalloidin": "phalloidin"}

#: Where PERISCOPE's threshold search starts: compartment counts, whole profile.
PERISCOPE_NULL_START: Dict[str, int] = {"compartment": 200, "Sum": 500}

#: The column names a gene can go by in a profile or p-value table.
GENE_COLUMNS = ("gene", "Gene", "Metadata_Foci_Barcode_MatchedTo_GeneCode", "Metadata_gene_symbol",
                "Metadata_Gene")

#: The correlation bins of PERISCOPE's STRING display (Figure 4D).
STRING_BINS = (-1.0, -0.66, -0.33, 0.0, 0.33, 0.66, 1.0)

#: More genes than this are not scored on STRING AUPRC (all gene pairs are held at once).
STRING_MAX_GENES = 6000


def gene_column(table: pd.DataFrame) -> str:
    """The column of a table that names the gene.

    :param table: a profile, guide or p-value table.
    :returns: the first of :data:`GENE_COLUMNS` the table has.
    :raises ValueError: when it has none of them.
    """
    for name in GENE_COLUMNS:
        if name in table.columns:
            return name
    raise ValueError(f"the table names no gene column; expected one of {list(GENE_COLUMNS)}")


def profile_frame(table: pd.DataFrame) -> pd.DataFrame:
    """A profile table as numeric features indexed by gene, controls dropped.

    :param table: one row per gene with a gene column (:data:`GENE_COLUMNS`)
        and numeric features; ``Metadata_*`` and ``n_cells`` are not features.
    :returns: the targeting genes' rows, first row per gene, in file order.
    """
    gene = gene_column(table)
    frame = table.drop_duplicates(gene)
    frame = frame[~is_nt(frame[gene])]
    cols = [c for c in _features(frame, (gene, "n_cells")) if not str(c).startswith("Metadata_")]
    return frame.set_index(frame[gene].astype(str))[cols].astype(float)


def compartment_features(features: Sequence[str], compartments: Any = "periscope") -> Dict[str, List[str]]:
    """Which features belong to which compartment.

    :param features: the feature names.
    :param compartments: ``"periscope"`` (:data:`PERISCOPE_COMPARTMENTS`) or a
        ``{name: substring}`` mapping, matched in any case.
    :returns: ``{compartment: features}`` in feature order; one feature may
        sit in several compartments, as in the reference.
    """
    table = PERISCOPE_COMPARTMENTS if compartments in (None, "periscope") else dict(compartments)
    return {name: [f for f in features if str(part).lower() in str(f).lower()] for name, part in table.items()}


def mann_whitney_table(guides: pd.DataFrame, *, control: str = "nontargeting",
                       features: Optional[Sequence[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per gene and feature, a Mann-Whitney U test of the gene's rows against the controls'.

    The reference rule (PERISCOPE ``per_feature_hit_calling.ipynb``): the
    unit is a guide-level profile row; each gene's rows are tested against
    all non-targeting rows with :func:`scipy.stats.mannwhitneyu` (control
    rows first, two-sided, default method), one test per feature.

    :param guides: guide-level rows with a gene column and numeric features.
    :param control: the gene name of the control rows.
    :param features: the features tested; every numeric non-metadata column
        when None.
    :returns: ``(u, p)``, each indexed by gene in order of first appearance,
        one column per feature; U is the control sample's statistic.
    :raises ValueError: when there are no control rows.
    """
    from scipy.stats import mannwhitneyu

    gene = gene_column(guides)
    names = guides[gene].astype(str)
    if features is None:
        features = [c for c in _features(guides, (gene, "n_cells")) if not str(c).startswith("Metadata_")]
    values = guides[list(features)].to_numpy(float)
    nt = values[(names == control).to_numpy()]
    if not len(nt):
        raise ValueError(f"no {control!r} rows to test against")
    genes = [g for g in pd.unique(names) if g != control]
    u_rows, p_rows = [], []
    for g in genes:
        test = mannwhitneyu(nt, values[(names == g).to_numpy()], axis=0)
        u_rows.append(np.asarray(test.statistic, float))
        p_rows.append(np.asarray(test.pvalue, float))
    index = pd.Index(genes, name="Gene")
    return (pd.DataFrame(u_rows, index=index, columns=list(features)),
            pd.DataFrame(p_rows, index=index, columns=list(features)))


def significant_feature_counts(pvalues: pd.DataFrame, compartments: Any = "periscope",
                               feature_p: float = 0.001) -> pd.DataFrame:
    """Per gene, the number of features below the p threshold, per compartment and in all.

    :param pvalues: gene by feature p-values, indexed by gene; a
        ``sig_gene_count`` row (the reference file's per-feature tally) is
        dropped.
    :param compartments: as :func:`compartment_features`.
    :param feature_p: a feature counts when its p is strictly below this.
    :returns: one column per compartment and ``Sum``, integer counts.
    """
    frame = pvalues.drop(index=[g for g in ("sig_gene_count",) if g in pvalues.index])
    significant = frame.to_numpy(float) < float(feature_p)
    columns = list(frame.columns)
    out = {}
    for name, feats in compartment_features(columns, compartments).items():
        where = [columns.index(f) for f in feats]
        out[name] = significant[:, where].sum(axis=1)
    out["Sum"] = significant.sum(axis=1)
    return pd.DataFrame(out, index=frame.index.astype(str))


def null_threshold(scores: Sequence[float], level: float, start: int) -> int:
    """PERISCOPE's score threshold from the null genes' scores at one FDR level.

    From ``start`` downwards, the first count ``c`` that more than
    ``int(n / (100 / (100 * level)))`` null genes exceed sets the threshold
    to ``c + 1``; a gene is a hit when its score is above it. This is the
    reference's ``FDR_thresholds`` loop, kept exactly.

    :param scores: the null (zero-TPM) genes' scores.
    :param level: the FDR level, e.g. 0.05.
    :param start: where the search starts (200 for a compartment, 500 for
        the whole profile in the reference).
    :returns: the threshold.
    :raises ValueError: when there are no null genes.
    """
    values = np.asarray(scores, float)
    if not len(values):
        raise ValueError("the hit null has no genes")
    allowed = int(len(values) / (100 / round(100 * float(level), 10)))
    count, c, threshold = 0, int(start), int(start) + 1
    while count <= allowed and c >= -1:
        count = int((values > c).sum())
        threshold = c + 1
        c -= 1
    return threshold


def expression_null(genes: Sequence[str], expression: Any, cell_line: Optional[str] = None
                    ) -> Tuple[List[str], List[str]]:
    """The zero-TPM (null) genes and the expressed genes among the scored genes.

    :param genes: the scored genes.
    :param expression: a table with ``Gene`` and one column per cell line
        (DepMap log2(TPM+1); 0 is not expressed), or a mapping with
        ``zero_tpm`` and ``expressed_genes`` lists (the reference's
        ``<line>_CCLE_expression_summary.json``).
    :param cell_line: the expression column, e.g. ``"A549"``.
    :returns: ``(zero_tpm, expressed)``: zero-TPM genes sorted, expressed
        genes in ``genes`` order, controls left out.
    :raises ValueError: when the cell line is not a column.
    """
    genes = [str(g) for g in genes if not is_nt([g])[0]]
    if isinstance(expression, Mapping):
        zero = set(map(str, expression["zero_tpm"])) & set(genes)
        if "expressed_genes" in expression:
            expressed = set(map(str, expression["expressed_genes"]))
            return sorted(zero), [g for g in genes if g in expressed]
    else:
        gene = gene_column(expression)
        if cell_line not in expression.columns:
            raise ValueError(f"ops_hits_cell_line {cell_line!r} is not a column of the expression table")
        zero = set(expression.loc[expression[cell_line] == 0, gene].astype(str)) & set(genes)
    return sorted(zero), [g for g in genes if g not in zero]


def call_hits(pvalues: pd.DataFrame, *, zero_tpm: Sequence[str], expressed: Sequence[str],
              compartments: Any = "periscope", feature_p: float = 0.001,
              fdr_levels: Sequence[float] = (0.05, 0.01),
              starts: Optional[Mapping[str, int]] = None) -> Dict[str, Any]:
    """Hit genes by PERISCOPE's per-feature Mann-Whitney rule with a zero-TPM null.

    A gene's score is its count of features with p below ``feature_p``, in
    each compartment and in all (:func:`significant_feature_counts`). The
    null is the scores of genes not expressed in the cell line; per FDR
    level each score gets a threshold (:func:`null_threshold`). An
    expressed gene is a WHOLE-CELL hit when its whole-profile count is
    above that threshold, and a COMPARTMENT-SPECIFIC hit when it is not but
    some compartment's count is above its own.

    :param pvalues: gene by feature p-values, indexed by gene.
    :param zero_tpm: the null genes.
    :param expressed: the genes that can be hits.
    :param compartments: as :func:`compartment_features`.
    :param feature_p: the per-feature threshold.
    :param fdr_levels: the FDR levels.
    :param starts: the threshold search starts, ``{"compartment", "Sum"}``;
        :data:`PERISCOPE_NULL_START` when None.
    :returns: ``{"scores", "thresholds", "null", "summary"}``: scores per
        gene with ``expressed``, ``zero_tpm`` and per level
        ``whole_cell_<pct>``, ``compartment_<pct>`` and ``hit_<pct>``; the
        thresholds per level; the null genes' scores; counts per level.
    """
    starts = dict(PERISCOPE_NULL_START if starts is None else starts)
    scores = significant_feature_counts(pvalues, compartments, feature_p)
    names = [c for c in scores.columns if c != "Sum"]
    zero, keep = set(map(str, zero_tpm)), set(map(str, expressed))
    null = scores.loc[[g for g in scores.index if g in zero]]
    out = scores.copy()
    out["expressed"] = out.index.isin(list(keep))
    out["zero_tpm"] = out.index.isin(list(zero))
    thresholds, summary = {}, {}
    for level in fdr_levels:
        pct = f"{round(100 * float(level), 6):g}"
        limits = {c: null_threshold(null[c], level, starts["Sum" if c == "Sum" else "compartment"])
                  for c in names + ["Sum"]}
        whole = out["expressed"] & (out["Sum"] > limits["Sum"])
        part = (out["expressed"] & (out["Sum"] <= limits["Sum"])
                & np.any([out[c] > limits[c] for c in names], axis=0))
        out[f"whole_cell_{pct}"] = whole
        out[f"compartment_{pct}"] = part
        out[f"hit_{pct}"] = whole | part
        thresholds[pct] = limits
        summary[pct] = {"whole_cell": int(whole.sum()), "compartment_specific": int(part.sum()),
                        "hits": int((whole | part).sum()),
                        "per_compartment": {c: int((out["expressed"] & (out["Sum"] <= limits["Sum"])
                                                    & (out[c] > limits[c])).sum()) for c in names}}
    summary["null_genes"] = int(len(null))
    summary["expressed_genes"] = int(out["expressed"].sum())
    return {"scores": out, "thresholds": thresholds, "null": null, "summary": summary}


def overlap_with_reference(hits: Sequence[str], reference: Sequence[str], *,
                           cell_counts: Optional[Mapping[str, float]] = None, min_cells: int = 20
                           ) -> Dict[str, Any]:
    """Jaccard and recall of a published hit list, on genes with enough cells.

    :param hits: the genes called here.
    :param reference: the published hit genes.
    :param cell_counts: cells per gene on the scored plate; no restriction
        when None.
    :param min_cells: genes with fewer cells are left out of both lists.
    :returns: ``hits``, ``reference``, ``shared``, ``jaccard``, ``recall``
        (of the reference), ``precision`` and ``restricted`` (whether the
        cell-count rule was applied).
    """
    mine, theirs = set(map(str, hits)), set(map(str, reference))
    if cell_counts is not None:
        enough = {str(g) for g, n in cell_counts.items() if n >= min_cells}
        mine, theirs = mine & enough, theirs & enough
    shared = mine & theirs
    union = mine | theirs
    return {"hits": len(mine), "reference": len(theirs), "shared": len(shared),
            "jaccard": len(shared) / len(union) if union else float("nan"),
            "recall": len(shared) / len(theirs) if theirs else float("nan"),
            "precision": len(shared) / len(mine) if mine else float("nan"),
            "restricted": cell_counts is not None}


def pca_rows(profile: pd.DataFrame, variance: Optional[float] = 0.9) -> pd.DataFrame:
    """Gene rows projected on the principal components closest to a variance share.

    The reference keeps the number of components whose cumulative
    explained variance is closest to ``variance``. spaCR uses the exact
    (full) SVD, so the result is deterministic; the reference's
    randomized solver differs from it in the fourth decimal of the median r.

    :param profile: genes by features.
    :param variance: the share, or None to keep the features.
    :returns: genes by components (or the profile unchanged).
    """
    if not variance:
        return profile
    from sklearn.decomposition import PCA

    ratios = PCA(svd_solver="full").fit(profile.to_numpy(float)).explained_variance_ratio_
    keep = int(np.argmin(np.abs(float(variance) - np.cumsum(ratios)))) + 1
    rows = PCA(n_components=keep, svd_solver="full").fit_transform(profile.to_numpy(float))
    return pd.DataFrame(rows, index=profile.index)


def read_corum(path: str) -> List[Tuple[str, List[str]]]:
    """CORUM complexes as ``(name, subunits)`` from the ``humanComplexes.txt`` export.

    :param path: the tab-separated CORUM file with ``ComplexName`` and
        ``subunits(Gene name)`` (members separated by ``;``).
    :returns: one entry per row, members in file order.
    """
    from .tabular import read_table

    table = read_table(path, canonicalise=False, report=None, sep="\t")
    name = "ComplexName" if "ComplexName" in table.columns else table.columns[0]
    return [(str(n), str(s).split(";")) for n, s in zip(table[name], table["subunits(Gene name)"])
            if isinstance(s, str)]


def corum_within_r(rows: pd.DataFrame, complexes: Sequence[Tuple[str, Sequence[str]]], *,
                   coverage: float = 0.333, seed: int = 0) -> Dict[str, Any]:
    """Pearson r of gene pairs within CORUM complexes, against random pairs.

    The reference rule (PERISCOPE Figure 4C): a complex counts when more
    than ``coverage`` of its listed subunits are scored genes; every
    distinct pair of its scored members is a within-complex pair, and the
    summary is the median r over the unique pairs pooled across complexes.
    The background is the median over all pairs of scored genes. The
    controls are the same number of random gene pairs and the same pairs
    after shuffling the gene labels of the rows (both with ``seed``).

    :param rows: genes by features (e.g. :func:`pca_rows`).
    :param complexes: :func:`read_corum` output.
    :param coverage: the share of a complex's subunits that must be scored.
    :param seed: the controls' :func:`numpy.random.default_rng` seed.
    :returns: ``{"summary", "complexes"}``: the summary rows (``observed``,
        ``all_pairs``, ``random_pairs``, ``shuffled_labels``) with ``median_r``
        and ``n_pairs``; one row per counted complex with its scored members,
        pairs and median r.
    """
    genes = [str(g) for g in rows.index]
    where = {g: i for i, g in enumerate(genes)}
    r = np.corrcoef(rows.to_numpy(float))
    pairs, per_complex = set(), []
    for name, members in complexes:
        found = [m for m in members if m in where]
        if not found or len(found) / len(members) <= float(coverage):
            continue
        mine = sorted({tuple(sorted((where[a], where[b]))) for a in found for b in found if a != b})
        pairs.update(mine)
        per_complex.append({"complex": name, "subunits": len(members), "scored": len(set(found)),
                            "pairs": len(mine),
                            "median_r": float(np.median([r[i, j] for i, j in mine])) if mine else float("nan")})
    pairs = sorted(pairs)
    rng = np.random.default_rng(int(seed))
    upper = np.triu_indices(len(genes), 1)
    summary = [{"control": "observed", "median_r": float(np.median([r[i, j] for i, j in pairs]))
                if pairs else float("nan"), "n_pairs": len(pairs)},
               {"control": "all_pairs", "median_r": float(np.median(r[upper])) if len(genes) > 1
                else float("nan"), "n_pairs": int(len(upper[0]))}]
    if pairs:
        pick = rng.choice(len(upper[0]), size=min(len(pairs), len(upper[0])), replace=False)
        summary.append({"control": "random_pairs", "median_r": float(np.median(r[upper[0][pick], upper[1][pick]])),
                        "n_pairs": int(len(pick))})
        order = rng.permutation(len(genes))
        summary.append({"control": "shuffled_labels",
                        "median_r": float(np.median([r[order[i], order[j]] for i, j in pairs])),
                        "n_pairs": len(pairs)})
    return {"summary": pd.DataFrame(summary), "complexes": pd.DataFrame(per_complex)}


def _string_positive(rows: pd.DataFrame, string: pd.DataFrame, threshold: float) -> set:
    """Unordered index pairs of scored genes with a STRING edge at a confidence."""
    where = {str(g): i for i, g in enumerate(rows.index)}
    scale = 1000.0 if float(string["combined_score"].max()) > 1.0 else 1.0
    edges = string[string["combined_score"] >= float(threshold) * scale]
    first = edges["protein1"].astype(str).map(where)
    second = edges["protein2"].astype(str).map(where)
    ok = first.notna() & second.notna() & (first != second)
    return {tuple(sorted((int(a), int(b)))) for a, b in zip(first[ok], second[ok])}


def string_auprc(rows: pd.DataFrame, string: pd.DataFrame, thresholds: Sequence[float] = (0.4, 0.7, 0.9),
                 *, seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Gene pairs ranked by cosine similarity against STRING edges, with the r-bin display.

    :param rows: genes by features.
    :param string: STRING links with ``protein1``, ``protein2`` (gene
        symbols) and ``combined_score`` (0-1000 or 0-1).
    :param thresholds: STRING confidences; pairs at or above one are its
        positives, every other pair of scored genes a negative.
    :param seed: the shuffled-label control's seed.
    :returns: ``(auprc, bins)``: per threshold and control (``observed``,
        ``shuffled_labels``) the average precision, the random baseline
        (positive share) and the counts; per Pearson-r bin
        (:data:`STRING_BINS`, as the reference's Figure 4D) the STRING links
        among scored genes with their mean and median combined score.
    :raises ValueError: when more than :data:`STRING_MAX_GENES` genes are given.
    """
    from sklearn.metrics import average_precision_score

    if len(rows) > STRING_MAX_GENES:
        raise ValueError(f"STRING AUPRC holds every gene pair at once; {len(rows)} genes is more than "
                         f"{STRING_MAX_GENES}, so score a gene subset (ops_scoring_benchmark_genes)")
    X = rows.to_numpy(float)
    upper = np.triu_indices(len(X), 1)
    order = np.random.default_rng(int(seed)).permutation(len(X))
    out = []
    for control, perm in (("observed", np.arange(len(X))), ("shuffled_labels", order)):
        cos = _cosine(X[perm])[upper]
        for threshold in thresholds:
            positive = _string_positive(rows, string, threshold)
            labels = np.zeros(len(cos), bool)
            if positive:
                a, b = np.array(sorted(positive)).T
                n = len(X)
                labels[(a * n - a * (a + 1) // 2 + (b - a - 1)).astype(np.int64)] = True
            out.append({"control": control, "threshold": float(threshold), "pairs": int(len(cos)),
                        "positives": int(labels.sum()),
                        "auprc": float(average_precision_score(labels, cos)) if labels.any() else float("nan"),
                        "baseline": float(labels.mean()) if len(labels) else float("nan")})
    r = np.corrcoef(X)
    where = {str(g): i for i, g in enumerate(rows.index)}
    links = string[string["protein1"].astype(str).isin(where) & string["protein2"].astype(str).isin(where)]
    values = np.array([r[where[str(a)], where[str(b)]] for a, b in zip(links["protein1"], links["protein2"])])
    edges = np.asarray(STRING_BINS)
    bin_of = np.clip(np.searchsorted(edges, values, side="left") - 1, 0, len(edges) - 2) if len(values) else values
    bins = []
    for k in range(len(edges) - 1):
        inside = links["combined_score"].to_numpy(float)[bin_of == k] if len(values) else np.array([])
        bins.append({"bin": f"{edges[k]:g} to {edges[k + 1]:g}", "links": int(len(inside)),
                     "mean_score": float(inside.mean()) if len(inside) else float("nan"),
                     "median_score": float(np.median(inside)) if len(inside) else float("nan")})
    return pd.DataFrame(out), pd.DataFrame(bins)


def _sha256(path: str) -> str:
    """The SHA-256 of a file, to record a reference version."""
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _reference_genes(entries: Any) -> List[str]:
    """The genes of a published hit list: paths, each optionally ``path::sheet``."""
    from .tabular import read_table

    genes: List[str] = []
    for entry in ([entries] if isinstance(entries, str) else list(entries)):
        path, _, sheet = str(entry).partition("::")
        kwargs = {"sheet_name": sheet} if sheet else {}
        table = read_table(path, canonicalise=False, report=None, **kwargs)
        genes.extend(table[gene_column(table)].astype(str).tolist())
    return genes


def score_profile_benchmarks(settings: Mapping[str, Any], folder: str,
                             report: Optional[Callable[[str], None]] = None) -> Dict[str, str]:
    """CORUM within-complex r, STRING AUPRC and the zero-TPM hit rule on every input (item 923).

    The same code scores every input of ``ops_scoring_inputs``: a reference
    pipeline's deposited profiles and spaCR's. ``hits_mw_ztpm`` reads an
    input's ``pvalues`` table (gene by feature, as deposited) or, without
    one, tests its ``guides`` rows (:func:`mann_whitney_table`), then calls
    hits (:func:`call_hits`); ``corum_within_r`` and ``string_auprc`` score
    its ``genes`` profile on the genes ``ops_scoring_benchmark_genes``
    names. What an input lacks is written to ``benchmark_scorable.csv``.

    :param settings: the resolved settings: ``ops_scoring_inputs``,
        ``ops_scoring_metrics``, ``ops_scoring_corum_file``,
        ``ops_scoring_string_file``, ``ops_scoring_string_thresholds``,
        ``ops_scoring_corum_coverage``, ``ops_scoring_pca_variance``,
        ``ops_scoring_benchmark_genes``, ``ops_scoring_benchmark_seed`` and
        the ``ops_hits_*`` keys.
    :param folder: the output folder.
    :param report: called with progress lines.
    :returns: the written files by name.
    :raises ValueError: when a metric's reference file is missing.
    """
    from .tabular import read_table, write_table

    say = report or (lambda text: None)
    metrics = [m for m in (settings.get("ops_scoring_metrics") or []) if m in BENCHMARK_METRICS]
    inputs = dict(settings.get("ops_scoring_inputs") or {})
    seed = int(settings.get("ops_scoring_benchmark_seed") or 0)
    levels = [float(x) for x in (settings.get("ops_hits_fdr") or [0.05, 0.01])]
    compartments = settings.get("ops_hits_compartments") or "periscope"
    outputs: Dict[str, str] = {}
    scorable, summary, hit_summary, references = [], [], {}, {}

    def write(frame, name):
        """Write one output table and remember it."""
        outputs[name] = write_table(frame, os.path.join(folder, name), canonicalise=False)

    def missing(name, metric, why):
        """Record that an input cannot be scored on a metric."""
        scorable.append({"input": name, "metric": metric, "scorable": "not scorable", "reason": why})

    hits_of: Dict[str, List[str]] = {}
    if "hits_mw_ztpm" in metrics:
        table = settings.get("ops_hits_expression_table")
        if not table:
            raise ValueError("hits_mw_ztpm needs ops_hits_expression_table (DepMap TPM table or the "
                             "expression summary JSON)")
        if str(table).endswith(".json"):
            with open(str(table), encoding="utf-8") as handle:
                expression: Any = json.load(handle)
        else:
            expression = read_table(str(table), canonicalise=False, report=None)
        references["expression"] = {"path": str(table), "sha256": _sha256(str(table)),
                                    "cell_line": settings.get("ops_hits_cell_line")}
        for name, parts in inputs.items():
            parts = dict(parts)
            if parts.get("pvalues"):
                raw = read_table(parts["pvalues"], canonicalise=False, report=None)
                pvalues = raw.set_index(raw[gene_column(raw)].astype(str)).drop(columns=[gene_column(raw)])
                source = "pvalues"
            elif parts.get("guides"):
                guides = read_table(parts["guides"], canonicalise=False, report=None)
                u, pvalues = mann_whitney_table(guides)
                write(u.reset_index(), f"mann_whitney_u_{name}.csv.gz")
                write(pvalues.reset_index(), f"mann_whitney_p_{name}.csv.gz")
                source = "guides"
            else:
                missing(name, "hits_mw_ztpm", "no guide-level profiles or per-feature p-values")
                continue
            genes = [g for g in pvalues.index if g != "sig_gene_count"]
            zero, expressed = expression_null(genes, expression, settings.get("ops_hits_cell_line"))
            called = call_hits(pvalues, zero_tpm=zero, expressed=expressed, compartments=compartments,
                               feature_p=float(settings.get("ops_hits_feature_p") or 0.001), fdr_levels=levels)
            scores = called["scores"].rename_axis("gene").reset_index()
            write(scores, f"hits_{name}.csv")
            write(called["null"].rename_axis("gene").reset_index(), f"hit_null_{name}.csv")
            record = {"source": source, "thresholds": called["thresholds"], **called["summary"]}
            strict = f"{round(100 * min(levels), 6):g}"
            hits_of[name] = scores.loc[scores[f"hit_{strict}"], "gene"].tolist()
            if settings.get("ops_hits_reference_list"):
                counts = None
                if parts.get("cell_counts"):
                    cells = read_table(parts["cell_counts"], canonicalise=False, report=None)
                    column = next(c for c in ("cells", "n_cells", "count") if c in cells.columns)
                    counts = dict(zip(cells[gene_column(cells)].astype(str), cells[column]))
                reference = _reference_genes(settings["ops_hits_reference_list"])
                record["reference_overlap"] = {
                    pct: overlap_with_reference(scores.loc[scores[f"hit_{pct}"], "gene"], reference,
                                                cell_counts=counts,
                                                min_cells=int(settings.get("ops_hits_min_cells") or 20))
                    for pct in called["thresholds"]}
            hit_summary[name] = record
            for pct, values in called["summary"].items():
                if isinstance(values, dict):
                    summary.append({"input": name, "metric": "hits_mw_ztpm", "control": f"fdr_{pct}",
                                    "value": values["hits"], "n": called["summary"]["expressed_genes"]})
            say(f"hits_mw_ztpm {name}: " + ", ".join(f"{p}% FDR {v['hits']}" for p, v in called["summary"].items()
                                                    if isinstance(v, dict)))
    profiles: Dict[str, pd.DataFrame] = {}
    if {"corum_within_r", "string_auprc"} & set(metrics):
        for name, parts in inputs.items():
            if dict(parts).get("genes"):
                profiles[name] = profile_frame(read_table(dict(parts)["genes"], canonicalise=False, report=None))
            else:
                for metric in ("corum_within_r", "string_auprc"):
                    if metric in metrics:
                        missing(name, metric, "no profile per gene")
        choice = settings.get("ops_scoring_benchmark_genes") or "all"
        if choice == "all" and profiles:
            shared = sorted(set.intersection(*[set(p.index) for p in profiles.values()]))
            chosen = {name: shared for name in profiles}
        elif choice == "hits":
            chosen = {name: [g for g in hits_of.get(name, []) if g in profiles[name].index] for name in profiles}
            for name in [n for n in profiles if n not in hits_of]:
                missing(name, "corum_within_r", "ops_scoring_benchmark_genes 'hits' needs hits_mw_ztpm hits")
                chosen.pop(name)
        else:
            listed = choice if isinstance(choice, (list, tuple)) else read_table(
                str(choice), canonicalise=False, report=None).pipe(lambda t: t[gene_column(t)].astype(str).tolist())
            chosen = {name: [g for g in listed if g in profiles[name].index] for name in profiles}
        variance = settings.get("ops_scoring_pca_variance", 0.9)
        corum = string = None
        if "corum_within_r" in metrics:
            path = settings.get("ops_scoring_corum_file")
            if not path:
                raise ValueError("corum_within_r needs ops_scoring_corum_file (CORUM humanComplexes.txt)")
            corum = read_corum(str(path))
            references["corum"] = {"path": str(path), "sha256": _sha256(str(path))}
        if "string_auprc" in metrics:
            path = settings.get("ops_scoring_string_file")
            if not path:
                raise ValueError("string_auprc needs ops_scoring_string_file (protein1, protein2, combined_score)")
            string = read_table(str(path), canonicalise=False, report=None)
            references["string"] = {"path": str(path), "sha256": _sha256(str(path))}
        complex_rows, auprc_rows, bin_rows = [], [], []
        for name, genes in chosen.items():
            rows = pca_rows(profiles[name].loc[genes], variance)
            if corum is not None:
                result = corum_within_r(rows, corum, coverage=float(settings.get("ops_scoring_corum_coverage", 0.333)),
                                        seed=seed)
                for row in result["summary"].to_dict("records"):
                    summary.append({"input": name, "metric": "corum_within_r", "control": row["control"],
                                    "value": row["median_r"], "n": row["n_pairs"], "genes": len(genes)})
                complex_rows.append(result["complexes"].assign(input=name))
            if string is not None:
                if len(rows) > STRING_MAX_GENES:
                    missing(name, "string_auprc", f"{len(rows)} genes; more than {STRING_MAX_GENES}")
                    continue
                auprc, bins = string_auprc(rows, string, settings.get("ops_scoring_string_thresholds")
                                           or [0.4, 0.7, 0.9], seed=seed)
                for row in auprc.to_dict("records"):
                    summary.append({"input": name, "metric": f"string_auprc>={row['threshold']:g}",
                                    "control": row["control"], "value": row["auprc"], "n": row["pairs"],
                                    "genes": len(genes), "baseline": row["baseline"]})
                auprc_rows.append(auprc.assign(input=name))
                bin_rows.append(bins.assign(input=name))
            say(f"profile benchmarks {name}: {len(genes)} genes")
        if complex_rows:
            write(pd.concat(complex_rows, ignore_index=True), "corum_complexes.csv")
        if auprc_rows:
            write(pd.concat(auprc_rows, ignore_index=True), "string_auprc.csv")
            write(pd.concat(bin_rows, ignore_index=True), "string_bins.csv")
    write(pd.DataFrame(summary), "benchmark_summary.csv")
    write(pd.DataFrame(scorable, columns=["input", "metric", "scorable", "reason"]), "benchmark_scorable.csv")
    setup = {"metrics": metrics, "references": references, "fdr_levels": levels, "seed": seed,
             "benchmark_genes": settings.get("ops_scoring_benchmark_genes") or "all",
             "pca_variance": settings.get("ops_scoring_pca_variance", 0.9),
             "matching": ("every input is scored by the same functions with the same reference files, "
                          "thresholds, FDR levels, PCA rule and control seed; with benchmark genes 'all' "
                          "on the genes all inputs share, with 'hits' on each input's own hits at the "
                          "strictest FDR level (the published rule, so the gene sets differ)")}
    for name, record in (("hits_summary.json", hit_summary), ("benchmark_setup.json", setup)):
        if name == "hits_summary.json" and not hit_summary:
            continue
        with open(os.path.join(folder, name), "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=1, default=float)
        outputs[name] = os.path.join(folder, name)
    return outputs


def score_ops_profiles(settings: Mapping[str, Any], report: Optional[Callable[[str], None]] = None
                       ) -> Dict[str, Any]:
    """Score OPS gene profiles and clusterings with copairs and EFAAR (item 913).

    :param settings: read for ``ops_scoring_inputs`` (``{name: {"genes",
        "guides", "clusters", "pvalues", "cell_counts": path}}``; the last
        two serve item 923's hit rule), ``ops_scoring_metrics`` (empty runs
        the item-913 metrics ``activity``, ``consistency`` and ``efaar``;
        listing ``corum_within_r``, ``string_auprc`` or ``hits_mw_ztpm``
        adds :func:`score_profile_benchmarks`, and a list of only those
        skips the copairs and EFAAR scoring), ``ops_scoring_genes`` (a table with
        ``gene``, or a list; the intersection of the inputs when empty),
        ``ops_scoring_null_size``, ``ops_scoring_min_complex``,
        ``ops_scoring_references``, ``ops_scoring_cluster_jitter``,
        ``ops_scoring_cluster_seeds``, ``ops_scoring_controls`` (False
        skips the shuffled and NT controls), ``ops_scoring_workers``,
        ``ops_scoring_figures``, ``ops_reference_dir`` and ``dst_root``.
    :param report: called with progress lines.
    :returns: ``{"folder", "outputs"}``; everything is written to
        ``<dst_root>/ops_scoring``: ``scorable.csv``, ``activity_map.csv``,
        ``consistency_complex_map.csv``, ``consistency_pairs_pergene.csv.gz``,
        ``consistency_pairs_summary_by_seed.csv``, ``efaar_recall_all.csv``,
        ``efaar_recall_summary.csv``, ``efaar_mcnemar.csv``,
        ``summary_main.csv``, ``stats.json``, ``setup.json`` and two figures.
    :raises ValueError: for missing inputs, ``dst_root`` or references.
    """
    from .ops_settings import ops_defaults
    from .tabular import read_table, write_table

    settings = ops_defaults(dict(settings))
    say = report or (lambda text: None)
    inputs = settings.get("ops_scoring_inputs")
    if not inputs:
        raise ValueError("OPS shared scoring needs ops_scoring_inputs, {name: {genes, guides, clusters}}")
    if not settings.get("dst_root"):
        raise ValueError("OPS shared scoring needs dst_root, the folder its results go to")
    wanted = list(settings.get("ops_scoring_metrics") or SHARED_METRICS)
    unknown = [m for m in wanted if m not in SHARED_METRICS + BENCHMARK_METRICS]
    if unknown:
        raise ValueError(f"ops_scoring_metrics: unknown {unknown}; choose from "
                         f"{list(SHARED_METRICS + BENCHMARK_METRICS)}")
    folder = os.path.join(str(settings["dst_root"]), "ops_scoring")
    os.makedirs(folder, exist_ok=True)
    benchmark: Dict[str, str] = {}
    if any(m in BENCHMARK_METRICS for m in wanted):
        benchmark = score_profile_benchmarks({**settings, "ops_scoring_metrics": wanted}, folder, report=say)
    if not any(m in SHARED_METRICS for m in wanted):
        return {"folder": folder, "outputs": benchmark}
    genes = settings.get("ops_scoring_genes")
    if isinstance(genes, str):
        genes = read_table(genes, canonicalise=False, report=None).gene.astype(str).tolist()
    data = load_scoring_inputs({name: {k: v for k, v in dict(parts).items()
                                       if k in ("genes", "guides", "clusters")}
                                for name, parts in inputs.items()}, genes)
    cache = os.path.join(folder, "copairs_null")
    null = int(settings.get("ops_scoring_null_size") or 10000)
    workers = int(settings.get("ops_scoring_workers") or 4)
    references = list(settings.get("ops_scoring_references") or ["CORUM@REPORT", "STRING>=0.7@REPORT"])
    seeds = [int(s) for s in (settings.get("ops_scoring_cluster_seeds") or [0, 1, 2])]
    jitter = float(settings.get("ops_scoring_cluster_jitter", 1e-3))
    outputs: Dict[str, str] = dict(benchmark)

    def write(frame, name):
        """Write one output table and remember it."""
        outputs[name] = write_table(frame, os.path.join(folder, name), canonicalise=False)

    write(scorability_table(data, list(inputs)), "scorable.csv")
    say(f"scoring {len(data['genes'])} shared genes of {list(inputs)}")
    activity = None
    if data["guides"]:
        parts = []
        for name, (X, gene) in data["guides"].items():
            parts.append(score_activity(X, gene, null_size=null, workers=workers, cache_dir=cache,
                                        label=f"{name} genes").assign(input=name))
        activity = pd.concat(parts, ignore_index=True)
        write(activity, "activity_map.csv")
        say(f"activity: {int(activity.below_corrected_p.sum())} active rows")
    complexes, pairs, setup = score_consistency(
        data, settings.get("ops_reference_dir"), references=references,
        min_members=int(settings.get("ops_scoring_min_complex") or 3), null_size=null, workers=workers,
        cluster_seeds=seeds, jitter=jitter, cache_dir=cache)
    write(complexes, "consistency_complex_map.csv")
    write(pairs, "consistency_pairs_pergene.csv.gz")
    write(pair_summary(pairs), "consistency_pairs_summary_by_seed.csv")
    say(f"consistency: {setup['complexes_REPORT_ge3']} complexes")
    recall, recall_summary, tests = score_efaar(data)
    write(recall, "efaar_recall_all.csv")
    write(recall_summary, "efaar_recall_summary.csv")
    write(tests, "efaar_mcnemar.csv")
    summary, stats = summary_and_stats(data, activity, complexes, pairs, recall, references, tests)
    write(summary, "summary_main.csv")
    setup.update({"genes": int(len(data["genes"])), "inputs": {k: dict(v) for k, v in inputs.items()},
                  "references": references, "packages": _versions(),
                  "matching": ("every input is scored on the same shared genes, with the same copairs and "
                               "EFAAR code, reference halves, thresholds, null sizes and control rows; an "
                               "input that lacks a level is listed in scorable.csv instead of being compared")})
    for name, record in (("stats.json", stats), ("setup.json", setup)):
        with open(os.path.join(folder, name), "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=1, default=float)
        outputs[name] = os.path.join(folder, name)
    if settings.get("ops_scoring_figures", True):
        for path in _figures(summary, stats, folder):
            outputs[os.path.basename(path)] = path
    return {"folder": folder, "outputs": outputs}


def _versions() -> Dict[str, str]:
    """Installed versions of the two scoring packages."""
    from importlib import metadata

    out = {}
    for name in SCORING_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out
