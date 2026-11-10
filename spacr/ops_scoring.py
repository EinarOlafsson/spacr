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


def score_ops_profiles(settings: Mapping[str, Any], report: Optional[Callable[[str], None]] = None
                       ) -> Dict[str, Any]:
    """Score OPS gene profiles and clusterings with copairs and EFAAR (item 913).

    :param settings: read for ``ops_scoring_inputs`` (``{name: {"genes",
        "guides", "clusters": path}}``), ``ops_scoring_genes`` (a table with
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
    genes = settings.get("ops_scoring_genes")
    if isinstance(genes, str):
        genes = read_table(genes, canonicalise=False, report=None).gene.astype(str).tolist()
    data = load_scoring_inputs(inputs, genes)
    folder = os.path.join(str(settings["dst_root"]), "ops_scoring")
    os.makedirs(folder, exist_ok=True)
    cache = os.path.join(folder, "copairs_null")
    null = int(settings.get("ops_scoring_null_size") or 10000)
    workers = int(settings.get("ops_scoring_workers") or 4)
    references = list(settings.get("ops_scoring_references") or ["CORUM@REPORT", "STRING>=0.7@REPORT"])
    seeds = [int(s) for s in (settings.get("ops_scoring_cluster_seeds") or [0, 1, 2])]
    jitter = float(settings.get("ops_scoring_cluster_jitter", 1e-3))
    outputs: Dict[str, str] = {}

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
