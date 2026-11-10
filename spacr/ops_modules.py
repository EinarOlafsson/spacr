"""Gene modules of an optical pooled screen: consensus clusters with a known core.

A module is a cluster of the reference clustering that is anchored by known
biology (a CORUM complex, a KEGG pathway or a GO-slim term enriched in it),
together with the genes the screen places next to that core although no
resource links them to it. The atlas is built in three steps.

1. **Partitions.** The gene profiles (targeting genes with more than ten
   cells, plus the control guides grouped at random into pseudo-genes of
   four, :func:`spacr.ops_clusters.consensus_rows`) are clustered once with
   PHATE + Leiden (seed 42) at the resolutions of :data:`MODULE_RESOLUTIONS`
   (about 70, 250 and 500 clusters), and again ``ops_modules_runs`` times on
   resampled rows and guides (:func:`spacr.ops_clusters.consensus_run`).
   By default the profiles are standardised and reduced to the components
   holding 99 % of the variance first. With
   ``ops_modules_precomputed_embedding`` the profile columns are used as
   they are, which is what a weighted profile (contrastive weighting) needs:
   standardising it again would undo the weighting.
2. **Anchoring.** In every reference cluster, a one-sided Fisher test for
   every group of 3 to 500 clustered genes, Benjamini-Hochberg within the
   cluster; the best group with q < 0.05 and three or more cluster genes is
   the anchor and its cluster genes are the known core.
3. **Gates.** A gene not linked to the core (no STRING v12 link of 0.4 or
   more, no BioGRID physical interaction, no shared enriched group) is a
   stable new member when its co-membership with the core over the
   consensus runs is above 0.5 and above the 99th percentile of both the
   control pseudo-genes and 500 fitness-matched random genes, its mean cosine
   to the core is above that of 1,000 fitness-matched genes (p < 0.01), at
   least three of its four guides point at the core, and its guides carry no
   specificity flag.

Controls: the control pseudo-genes pass through the same gates (empirical
FDR), and the gene labels are shuffled three times among the targeting rows
and the anchoring and gates recomputed on the same partitions.

This is the plate A pilot's module atlas; every seed is the pilot's, so a
settings run gives the pilot's partitions and tables. PHATE and Leiden come
with the ``clustering`` extra, ``pip install "spacr[clustering]"``.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

#: The reference resolutions, keyed by the cluster count they aim at.
MODULE_RESOLUTIONS: Dict[str, int] = {"~70": 3, "~250": 10, "~500": 26}

#: The resolutions the consensus runs cycle through.
CONSENSUS_SWEEP: Tuple[int, ...] = (3, 5, 8, 10, 13, 17, 22, 26)

#: The files the atlas reads from ``ops_reference_dir``.
MODULE_REFERENCE_FILES: Dict[str, str] = {
    "CORUM": "corum_group_benchmark.tsv",
    "KEGG": "kegg_group_benchmark.tsv",
    "GOslim": "go_slim_groups.parquet",
    "STRING": "string_v12_ge400.parquet",
    "BioGRID": "biogrid_physical.parquet",
    "specificity": "gene_specificity.csv",
}

#: The columns of ``module_members.csv``, in order.
MEMBER_COLUMNS: Tuple[str, ...] = (
    "module", "resolution", "res_label", "cluster", "anchor", "gene", "is_nt_pseudo", "in_cluster",
    "comembership", "comembership_threshold", "soft_cosine", "soft_z", "soft_p", "guide_support",
    "guides", "persistence_of_3", "driver_sign_agreement", "member_top_features", "specificity_flag",
    "fitness_decile", "stable_new_member", "hard_miss")

#: The gates, as the summary states them.
GATES_TEXT = ("comembership > max(0.5, NT pseudo-gene p99, p99 of 500 fitness-matched random genes); "
              "soft_p < 0.01; guide_support >= 3 of 4; no specificity flag; no STRING>=0.4 / BioGRID / "
              "enriched-group link to the core")


def precomputed_embedding(frame: pd.DataFrame, features: Sequence[str], variance: float = 0.99,
                          seed: int = 0) -> Tuple[np.ndarray, int, int]:
    """The profile columns as they are, in place of the standardise + PCA embedding.

    :param frame: one row per gene.
    :param features: the profile columns.
    :param variance: ignored; kept so the call matches
        :func:`spacr.ops_clusters.embed_profiles`.
    :param seed: ignored, as ``variance``.
    :returns: ``(values, n_columns, n_columns)``.
    """
    return frame[list(features)].to_numpy(float), len(features), len(features)


def _embedder(precomputed: bool) -> Callable[..., Tuple[np.ndarray, int, int]]:
    """The embedding the partitions use: standardise + PCA, or the columns as they are."""
    if precomputed:
        return precomputed_embedding
    from .ops_clusters import embed_profiles

    return embed_profiles


def profile_columns(guides: pd.DataFrame) -> List[str]:
    """The profile columns of a guide table: its ``PC_`` columns, else every numeric one.

    :param guides: the guide table.
    :returns: the column names, in table order.
    """
    pcs = [c for c in guides.columns if str(c).startswith("PC_")]
    if pcs:
        return pcs
    meta = {"gene", "sgRNA", "n_cells", "n_guides", "guide", "barcode"}
    return [c for c in guides.columns
            if c not in meta and pd.api.types.is_numeric_dtype(guides[c])]


def reference_partitions(rows: pd.DataFrame, columns: Sequence[str], *,
                         resolutions: Mapping[str, float] = MODULE_RESOLUTIONS, seed: int = 42,
                         precomputed: bool = False) -> Dict[str, Any]:
    """The reference clustering: one PHATE + Leiden run at every module resolution.

    :param rows: the rows of :func:`spacr.ops_clusters.consensus_rows`.
    :param columns: the profile columns.
    :param resolutions: ``{label: resolution}``.
    :param seed: the PHATE and Leiden seed.
    :param precomputed: use the columns as they are (see
        :func:`precomputed_embedding`).
    :returns: ``{"pcs", "labels", "info"}``: the embedding as float32, the
        labels per resolution and ``{"rows", "pcs", "clusters",
        "resolutions"}``.
    :raises ImportError: when the ``clustering`` extra is not installed.
    """
    from .ops_clusters import phate_leiden

    pcs, n_pcs, _ = _embedder(precomputed)(rows, list(columns))
    labels = phate_leiden(pcs, list(resolutions.values()), seed=seed)
    info = {"rows": int(len(rows)), "pcs": int(n_pcs),
            "clusters": {k: int(len(np.unique(labels[r]))) for k, r in resolutions.items()},
            "resolutions": dict(resolutions)}
    return {"pcs": np.asarray(pcs).astype(np.float32), "labels": labels, "info": info}


def _consensus_run(genes: pd.DataFrame, guides: pd.DataFrame, columns: Sequence[str], run: int, *,
                   precomputed: bool, resolutions: Sequence[float] = CONSENSUS_SWEEP,
                   base_seed: int = 1000, subsample: float = 0.9,
                   **row_options) -> Tuple[float, int, np.ndarray]:
    """One consensus run, :func:`spacr.ops_clusters.consensus_run` with the embedding of this module.

    :param genes: the gene table.
    :param guides: the guide table.
    :param columns: the profile columns.
    :param run: the run number; its generator is seeded ``base_seed + run``.
    :param precomputed: use the columns as they are.
    :param resolutions: the sweep; run ``k`` uses ``resolutions[k % len]``.
    :param base_seed: see ``run``.
    :param subsample: the share of rows clustered, drawn without replacement.
    :param row_options: passed to :func:`spacr.ops_clusters.consensus_rows`.
    :returns: ``(resolution, clusters, labels)``, -1 for rows left out.
    """
    from . import ops_clusters

    rng = np.random.default_rng(base_seed + run)
    table = ops_clusters.consensus_rows(genes, guides, columns, rng=rng, **row_options)
    n = len(table)
    sub = np.sort(rng.choice(n, int(subsample * n), replace=False))
    components, _, _ = _embedder(precomputed)(table.iloc[sub].reset_index(drop=True), list(columns))
    resolution = resolutions[run % len(resolutions)]
    labels = ops_clusters.phate_leiden(components, [resolution],
                                       seed=int(rng.integers(1, 10 ** 6)))[resolution]
    full = np.full(n, -1, np.int32)
    full[sub] = labels
    return resolution, int(len(np.unique(labels))), full


def consensus_partitions(genes: pd.DataFrame, guides: pd.DataFrame, columns: Sequence[str], *,
                         runs: int = 100, precomputed: bool = False,
                         folder: Optional[str] = None,
                         report: Optional[Callable[[str], None]] = None,
                         **options) -> Dict[str, Any]:
    """The consensus runs, one after another, kept per run so a stopped run resumes.

    :param genes: the gene table.
    :param guides: the guide table.
    :param columns: the profile columns.
    :param runs: the number of runs.
    :param precomputed: use the columns as they are in every run.
    :param folder: when given, run ``k`` is written to ``run_<k>.npz``
        there and an existing file is read instead of rerun.
    :param report: called with one line per run.
    :param options: the sweep (``resolutions``, :data:`CONSENSUS_SWEEP`),
        ``base_seed``, ``subsample`` and the options of
        :func:`spacr.ops_clusters.consensus_rows`, as for
        :func:`spacr.ops_clusters.consensus_run`.
    :returns: ``{"labels", "runs"}``: the ``runs x rows`` label matrix
        (-1 = left out of that run) and each run's resolution and cluster
        count.
    :raises ImportError: when the ``clustering`` extra is not installed.
    """
    say = report or (lambda text: None)
    labels, meta = [], []
    for k in range(int(runs)):
        path = os.path.join(folder, f"run_{k:03d}.npz") if folder else None
        if path and os.path.isfile(path):
            with np.load(path) as stored:
                full = stored["labels"]
                resolution, clusters = int(stored["resolution"]), int(stored["clusters"])
        else:
            resolution, clusters, full = _consensus_run(genes, guides, columns, k,
                                                        precomputed=precomputed, **options)
            if path:
                np.savez(path, labels=full, resolution=resolution, clusters=clusters)
            say(json.dumps({"run": k, "resolution": resolution, "clusters": clusters}))
        labels.append(full)
        meta.append({"resolution": int(resolution), "clusters": int(clusters)})
    width = len(labels[0]) if labels else 0
    return {"labels": np.vstack(labels) if labels else np.zeros((0, width), np.int32),
            "runs": meta}


def module_groups(background: Set[str], sources: Sequence[Tuple[str, pd.DataFrame]], *,
                  min_size: int = 3, max_size: int = 500) -> Dict[str, Set[str]]:
    """The annotation groups an anchor is chosen from.

    :param background: the clustered targeting genes.
    :param sources: ``(name, table)`` pairs, each table with ``group`` and
        ``gene_name`` columns, in the order the groups are tested.
    :param min_size: the fewest background genes a group keeps.
    :param max_size: the most.
    :returns: ``{"<name>: <group>": genes}``, the groups sorted by name
        within each source.
    """
    out: Dict[str, Set[str]] = {}
    for source, table in sources:
        for group, rows in table.groupby("group"):
            members = set(rows.gene_name) & background
            if min_size <= len(members) <= max_size:
                out[f"{source}: {group}"] = members
    return out


def anchor_clusters(clusters: np.ndarray, names: np.ndarray, controls: np.ndarray,
                    groups: Mapping[str, Set[str]], n_background: int) -> Dict[int, Dict[str, Any]]:
    """The anchor of every cluster: its best enriched group.

    :param clusters: one label per row.
    :param names: the row names.
    :param controls: True for the control pseudo-gene rows, which are never
        members.
    :param groups: the groups of :func:`module_groups`.
    :param n_background: the number of clustered targeting genes.
    :returns: ``{cluster: {"anchor", "anchor_q", "core", "enriched",
        "members"}}`` for every cluster of three or more targeting genes
        with a group at BH q < 0.05 and three or more members in it. Every
        group enters the correction; one with fewer than two cluster genes
        is given p = 1.
    """
    from scipy.stats import fisher_exact
    from statsmodels.stats.multitest import multipletests

    out: Dict[int, Dict[str, Any]] = {}
    group_names = list(groups)
    for cluster in np.unique(clusters[~controls]):
        members = set(names[(clusters == cluster) & ~controls])
        if len(members) < 3:
            continue
        overlap = [(g, len(members & groups[g])) for g in group_names]
        p = [fisher_exact([[k, len(groups[g]) - k],
                           [len(members) - k, n_background - len(groups[g]) - len(members) + k]],
                          alternative="greater")[1] if k >= 2 else 1.0 for g, k in overlap]
        q = multipletests(p, method="fdr_bh")[1]
        enriched = [(g, k, pi, qi) for (g, k), pi, qi in zip(overlap, p, q) if qi < 0.05 and k >= 3]
        if enriched:
            enriched.sort(key=lambda x: (x[3], x[2]))
            out[int(cluster)] = {"anchor": enriched[0][0], "anchor_q": enriched[0][3],
                                 "core": sorted(members & groups[enriched[0][0]]),
                                 "enriched": [e[0] for e in enriched], "members": sorted(members)}
    return out


def _comembership(labels: np.ndarray, core: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """How often each row shares a cluster with the core's majority cluster.

    :param labels: the ``runs x rows`` consensus labels.
    :param core: the core's row indices.
    :param rows: the rows scored.
    :returns: one share per row over the runs in which the row and at least
        two core genes were clustered; NaN for a row never counted.
    """
    num = np.zeros(len(rows))
    den = np.zeros(len(rows))
    for run in labels:
        core_labels = run[core]
        core_labels = core_labels[core_labels >= 0]
        if len(core_labels) < 2:
            continue
        values, counts = np.unique(core_labels, return_counts=True)
        majority = values[counts.argmax()]
        these = run[rows]
        present = these >= 0
        den += present
        num += present & (these == majority)
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def _neighbours(a: Sequence[str], b: Sequence[str]) -> Dict[str, Set[str]]:
    """Each gene's partners in an undirected edge list."""
    out: Dict[str, Set[str]] = {}
    for x, y in zip(a, b):
        out.setdefault(x, set()).add(y)
        out.setdefault(y, set()).add(x)
    return out


class _AtlasInputs:
    """Everything the gates read, prepared once for the real and the shuffled labels.

    :param rows: the reference rows (``gene``, ``n_cells``, profiles).
    :param pcs: the reference embedding, one row per row of ``rows``.
    :param reference: ``{resolution: labels}`` of the reference clustering.
    :param labels: the consensus label matrix.
    :param guides: the guide profile table (``gene``, ``n_cells``, profiles).
    :param driver_genes: the robust-z gene table the driving features come from.
    :param driver_guides: the robust-z guide table (its control guides fill
        in for a row without a gene profile).
    :param groups: the anchor groups of :func:`module_groups`.
    :param string: STRING links of 0.4 or more (``a``, ``b``,
        ``combined_score``).
    :param biogrid: BioGRID physical pairs (``a``, ``b``).
    :param specificity: per-gene ``gene`` and ``specificity_flag``.
    :param rng: the generator every random draw of the atlas comes from.
    :param resolutions: ``{label: resolution}``.
    :param control_prefix: the control guide prefix.
    """

    def __init__(self, rows, pcs, reference, labels, guides, driver_genes, driver_guides, groups,
                 string, biogrid, specificity, rng, resolutions, control_prefix):
        """Prepare the inputs; see the class for the parameters."""
        self.names = rows.gene.to_numpy()
        self.n = len(self.names)
        self.controls = pd.Series(self.names).str.startswith("NTpseudo").to_numpy()
        self.reference = reference
        self.resolutions = dict(resolutions)
        pcs = np.asarray(pcs, np.float32).astype(float)
        self.unit = pcs / np.linalg.norm(pcs, axis=1, keepdims=True)
        self.labels = labels
        background = set(self.names[~self.controls])
        self.groups = groups
        self.n_background = len(background)
        s4 = string[string.a.isin(background) & string.b.isin(background)]
        self.nb4 = _neighbours(s4.a, s4.b)
        s7 = s4[s4.combined_score >= 700]
        self.e7 = set(zip(s7.a, s7.b))
        self.dens7 = len(self.e7) / (self.n_background * (self.n_background - 1) / 2)
        self.nbb = _neighbours(biogrid.a, biogrid.b)
        spec = specificity.set_index("gene")
        flagged = set(spec[spec.specificity_flag.astype(bool)].index)
        for g in list(flagged):
            flagged |= set(str(g).split("&"))
        self.flagged = flagged
        nc = rows.n_cells.to_numpy(float)
        dec = np.full(self.n, -1)
        dec[~self.controls] = pd.qcut(pd.Series(nc[~self.controls]).rank(method="first"), 10,
                                      labels=False).to_numpy()
        self.dec = dec
        self.by_dec = {d: np.flatnonzero(dec == d) for d in range(10)}
        columns = profile_columns(guides)
        g = guides[guides.n_cells > 0]
        gx = np.array(g[columns].to_numpy(float), copy=True)
        gx /= np.linalg.norm(gx, axis=1, keepdims=True)
        self.gx = gx
        g_of: Dict[str, List[int]] = {}
        for k, name in enumerate(g.gene):
            g_of.setdefault(name, []).append(k)
        nt = g[g.gene.str.lower().str.startswith(control_prefix.lower())].reset_index()
        perm = np.random.default_rng(0).permutation(len(nt))
        nt["pseudo"] = ["NTpseudo_%02d" % (k // 4) for k in np.argsort(perm)]
        position = {v: i for i, v in enumerate(g.index)}
        for pseudo, d in nt.groupby("pseudo"):
            g_of[pseudo] = d["index"].map(position).tolist()
        self.g_of = g_of
        tg = np.vstack([np.median(gx[g_of[x]], 0) if x in g_of else np.zeros(len(columns))
                        for x in self.names])
        tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-9)
        self.tg = tg
        self.rng = rng
        tg_i, tg_j = rng.integers(0, len(gx), 40000), rng.integers(0, len(gx), 40000)
        gn = g.gene.to_numpy()
        ok = gn[tg_i] != gn[tg_j]
        self.guide_p99 = float(np.percentile(np.einsum("ij,ij->i", gx[tg_i[ok]], gx[tg_j[ok]]), 99))
        z = driver_genes.set_index("gene")
        self.zf = [c for c in z.columns if c not in ("n_guides", "n_cells") and z[c].notna().mean() > 0.95]
        self.z = z
        self.ntz = driver_guides[driver_guides.gene.str.lower().str.startswith(
            control_prefix.lower())][self.zf].median()


def _module_rows(inp: _AtlasInputs, names_now: np.ndarray, key: str
                 ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Anchoring and gates for one labelling of the rows, the real or a shuffled one.

    :param inp: the prepared inputs; its generator is advanced.
    :param names_now: the row names under this labelling.
    :param key: the plate key the module ids start with.
    :returns: ``(members, modules)``: one row per candidate and module, and
        one record per anchored module.
    """
    ix_now = {g: i for i, g in enumerate(names_now)}
    names, n, rng = inp.names, inp.n, inp.rng
    res_rows, mods = [], []
    for rlab, r in inp.resolutions.items():
        cl = inp.reference[r]
        anchors = anchor_clusters(cl, names_now, inp.controls, inp.groups, inp.n_background)
        for c, a in anchors.items():
            mid = f"{key}_r{r}_c{c}"
            core = [g for g in a["core"] if g in ix_now]
            cidx = np.array([ix_now[g] for g in core])
            sims = inp.unit @ inp.unit[cidx].T
            self_mask = np.isin(np.arange(n), cidx)
            soft = (sims.sum(1) - self_mask * 1.0) / (len(cidx) - self_mask)
            null_all, nullm, nulls = {}, np.zeros(10), np.ones(10)
            for d_, idx in inp.by_dec.items():
                idx = idx[~np.isin(idx, cidx)]
                sample = soft[rng.choice(idx, min(1000, len(idx)), replace=False)]
                null_all[d_] = np.sort(sample)
                nullm[d_], nulls[d_] = sample.mean(), sample.std() + 1e-9
            dd = np.where(inp.dec >= 0, inp.dec, 4)
            sz = (soft - nullm[dd]) / nulls[dd]
            sp = np.array([(len(null_all[d_]) - np.searchsorted(null_all[d_], s_, side="left") + 1)
                           / (len(null_all[d_]) + 1) for d_, s_ in zip(dd, soft)])
            linked = set(core)
            for g in core:
                linked |= inp.nb4.get(g, set()) | inp.nbb.get(g, set())
            for gname in a["enriched"]:
                linked |= inp.groups[gname]
            cand = np.flatnonzero((sz > 2) | (cl == c))
            cand = [i for i in cand if names_now[i] not in linked]
            if not cand:
                mods.append({**a, "module": mid, "resolution": r, "res_label": rlab, "cluster": c,
                             "core": core})
                continue
            cm = _comembership(inp.labels, cidx, np.array(cand))
            cm_nt = _comembership(inp.labels, cidx, np.flatnonzero(inp.controls))
            dmix = np.array([inp.dec[i] for i in cand if inp.dec[i] >= 0] or [4])
            rnd = np.concatenate([rng.choice(inp.by_dec[d_], int(np.ceil(500 * (dmix == d_).mean())))
                                  for d_ in np.unique(dmix)])
            rnd = rnd[~np.isin(rnd, cidx)]
            cm_rnd = _comembership(inp.labels, cidx, rnd)
            thr = max(0.5, np.nanpercentile(cm_nt, 99), np.nanpercentile(cm_rnd, 99))
            zc = inp.z.reindex(core)[inp.zf].mean()
            top = zc.abs().sort_values(ascending=False).index[:8]
            cen_t = inp.tg[cidx].mean(0)
            cen_t /= np.linalg.norm(cen_t)
            modes = {r2: pd.Series(inp.reference[r2][cidx]).mode().iloc[0]
                     for r2 in inp.resolutions.values()}
            for i, cmv in zip(cand, cm):
                g = names_now[i]
                gi = inp.g_of.get(names[i], [])
                gsup = int(((inp.gx[gi] @ cen_t) > inp.guide_p99).sum()) if gi else 0
                need = 3 if len(gi) >= 4 else max(1, len(gi) - 1)
                zrow = inp.z.reindex([names[i]])[top].iloc[0] if names[i] in inp.z.index else inp.ntz[top]
                agree = float((np.sign(zrow) == np.sign(zc[top])).mean())
                pers = int(sum(inp.reference[r2][i] == modes[r2] for r2 in inp.resolutions.values()))
                is_nt = bool(inp.controls[i])
                spec_flag = (names[i] in inp.flagged) and not is_nt
                stable = (cmv > thr) and (sp[i] < 0.01) and (gsup >= need) and not spec_flag
                res_rows.append({"module": mid, "resolution": r, "res_label": rlab, "cluster": c,
                                 "anchor": a["anchor"], "gene": g, "is_nt_pseudo": is_nt,
                                 "in_cluster": bool(cl[i] == c), "comembership": round(float(cmv), 3),
                                 "comembership_threshold": round(float(thr), 3),
                                 "soft_cosine": round(float(soft[i]), 3), "soft_z": round(float(sz[i]), 2),
                                 "soft_p": float(sp[i]), "guide_support": gsup, "guides": len(gi),
                                 "persistence_of_3": pers, "driver_sign_agreement": round(agree, 2),
                                 "member_top_features": "; ".join(f"{f} {zrow[f]:+.2f}" for f in top[:5]),
                                 "specificity_flag": spec_flag, "fitness_decile": int(inp.dec[i]),
                                 "stable_new_member": bool(stable),
                                 "hard_miss": bool(stable and cl[i] != c)})
            members = a["members"]
            mods.append({**a, "module": mid, "resolution": r, "res_label": rlab, "cluster": c, "core": core,
                         "drivers": "; ".join(f"{f} {zc[f]:+.2f}" for f in top),
                         "string07_edges_obs": int(sum((min(x, y), max(x, y)) in inp.e7
                                                       for k_, x in enumerate(members)
                                                       for y in members[k_ + 1:])),
                         "string07_density_background": inp.dens7})
    return pd.DataFrame(res_rows, columns=list(MEMBER_COLUMNS)), mods


def module_atlas(rows: pd.DataFrame, pcs: np.ndarray, reference: Mapping[float, np.ndarray],
                 labels: np.ndarray, guides: pd.DataFrame, driver_genes: pd.DataFrame,
                 driver_guides: pd.DataFrame, groups: Mapping[str, Set[str]], string: pd.DataFrame,
                 biogrid: pd.DataFrame, specificity: pd.DataFrame, *, key: str = "A",
                 resolutions: Mapping[str, float] = MODULE_RESOLUTIONS, seed: int = 0,
                 shuffles: int = 3, control_prefix: str = "nontargeting") -> Dict[str, Any]:
    """The module atlas: anchored modules, their candidate members, gates and controls.

    :param rows: the reference rows of :func:`spacr.ops_clusters.consensus_rows`
        (``gene``, ``n_cells`` and the profiles).
    :param pcs: the reference embedding of :func:`reference_partitions`.
    :param reference: ``{resolution: labels}`` of the reference clustering.
    :param labels: the ``runs x rows`` consensus labels.
    :param guides: the guide profiles in the clustered space (``gene``,
        ``n_cells`` and the profile columns).
    :param driver_genes: the robust-z gene table the driving features are
        read from (``gene`` and features).
    :param driver_guides: the robust-z guide table.
    :param groups: the anchor groups of :func:`module_groups`.
    :param string: STRING links of 0.4 or more: ``a``, ``b``,
        ``combined_score``.
    :param biogrid: BioGRID physical pairs: ``a``, ``b``.
    :param specificity: ``gene`` and ``specificity_flag``.
    :param key: the plate key the module ids start with.
    :param resolutions: ``{label: resolution}`` of the reference clustering.
    :param seed: the seed of the atlas' own random draws.
    :param shuffles: label shuffles of the control, seeded 100, 101, ...
    :param control_prefix: the control guide prefix.
    :returns: ``{"members", "atlas", "summary"}``: the candidate table
        (``module_members.csv``), one row per anchored module
        (``module_atlas.csv``) and the summary (``module_summary.json``).
    """
    rng = np.random.default_rng(seed)
    inp = _AtlasInputs(rows, pcs, reference, labels, guides, driver_genes, driver_guides, groups,
                       string, biogrid, specificity, rng, resolutions, control_prefix)
    names, isnt = inp.names, inp.controls
    members, mods = _module_rows(inp, names, key)
    is_nt = members.is_nt_pseudo.astype(bool)
    tgt = members[~is_nt]
    ntm = members[is_nt]
    calls = tgt[tgt.stable_new_member.astype(bool)]
    n_nt = int(isnt.sum())
    nt_rate = ntm.groupby("gene").stable_new_member.any().reindex(names[isnt], fill_value=False).mean()
    fdr_nt = nt_rate * (~isnt).sum() / max(calls.gene.nunique(), 1)
    shuffled = []
    for k in range(int(shuffles)):
        nm = names.copy()
        t_ = np.flatnonzero(~isnt)
        nm[t_] = np.random.default_rng(100 + k).permutation(nm[t_])
        m_, md_ = _module_rows(inp, nm, key)
        st = m_[(~m_.is_nt_pseudo) & m_.stable_new_member] if len(m_) else m_
        shuffled.append({"anchored_modules": len(md_),
                         "modules_with_stable_new_member": int(st.module.nunique()) if len(st) else 0,
                         "stable_new_member_genes": int(st.gene.nunique()) if len(st) else 0})
    atlas = []
    for m in mods:
        mm = calls[calls.module == m["module"]]
        size = len(m["members"])
        atlas.append({"module": m["module"], "resolution": m["resolution"], "clusters_label": m["res_label"],
                      "anchor": m["anchor"], "anchor_q": m["anchor_q"], "known_core": " ".join(m["core"]),
                      "n_core": len(m["core"]), "cluster_size": size,
                      "other_enriched_groups": " | ".join(m["enriched"][1:6]),
                      "string07_edges_in_cluster": m.get("string07_edges_obs"),
                      "string07_expected": round(m.get("string07_density_background", inp.dens7)
                                                 * size * (size - 1) / 2, 1),
                      "new_members": " ".join(mm.gene), "n_new": len(mm),
                      "new_members_hard_miss": " ".join(mm[mm.hard_miss].gene),
                      "driving_features_core": m.get("drivers", ""),
                      "replicate_status": "pending B-D"})
    atlas = pd.DataFrame(atlas)
    summary = {"key": key, "rows": inp.n, "nt_pseudo_genes": n_nt,
               "anchored_modules": {lab: int((atlas.clusters_label == lab).sum()) if len(atlas) else 0
                                    for lab in resolutions},
               "modules_with_stable_new_members": int((atlas.n_new > 0).sum()) if len(atlas) else 0,
               "stable_new_member_calls": int(len(calls)), "stable_new_member_genes": int(calls.gene.nunique()),
               "hard_miss_calls": int(calls.hard_miss.sum()),
               "nt_pseudo_genes_called (control)": int(ntm[ntm.stable_new_member.astype(bool)].gene.nunique()),
               "empirical_FDR_vs_NT_pseudo_genes": round(float(fdr_nt), 4),
               "shuffled_labels (control)": shuffled,
               "empirical_FDR_vs_shuffled_labels": round(float(
                   np.mean([x["stable_new_member_genes"] for x in shuffled]) / max(calls.gene.nunique(), 1)), 4)
               if shuffled else None,
               "guide_cosine_p99_cross_gene": inp.guide_p99,
               "gates": GATES_TEXT}
    return {"members": members, "atlas": atlas, "summary": summary}


def _reference_path(name: str, reference_dir: Any) -> str:
    """The path of one atlas reference file, from the first folder of ``ops_reference_dir`` that has it."""
    from .ops_reference import reference_dirs

    folders = reference_dirs(reference_dir)
    for folder in folders:
        path = os.path.join(folder, MODULE_REFERENCE_FILES[name])
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"the module atlas needs {MODULE_REFERENCE_FILES[name]} in one of {list(folders)}")


def _read(path: str, **options) -> pd.DataFrame:
    """Read a table exactly as written."""
    from .tabular import read_table

    return read_table(path, canonicalise=False, report=None, **options)


def _profile_folder(settings: Mapping[str, Any]) -> str:
    """The folder of the profile tables: ``ops_modules_profiles``, else the profile path's run folder."""
    given = settings.get("ops_modules_profiles")
    if given:
        return str(given)
    if not settings.get("dst_root"):
        raise ValueError("the modules phase needs ops_modules_profiles or dst_root")
    cells = settings.get("ops_profile_cells")
    if settings.get("ops_profile_run"):
        run = str(settings["ops_profile_run"])
    elif isinstance(cells, Mapping):
        run = "".join(sorted(map(str, cells)))
    else:
        run = str(settings.get("plate") or "A")
    return os.path.join(str(settings["dst_root"]), "ops_profile", run)


def _driver_tables(value: Any, fallback: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """The robust-z gene and guide tables: a folder, a two-item list of paths, or the profile folder."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError("ops_modules_driver_profiles takes a folder or [genes, guides] paths")
        genes, guides = map(str, value)
    else:
        folder = str(value) if value else fallback
        genes, guides = os.path.join(folder, "genes.parquet"), os.path.join(folder, "guides.parquet")
    return _read(genes), _read(guides)


def run_module_phase(settings: Mapping[str, Any],
                     report: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """The OPS ``modules`` phase: reference + consensus partitions, then the module atlas.

    Reads ``genes.parquet`` and ``guides.parquet`` from
    ``ops_modules_profiles`` (the profile path's run folder,
    ``<dst_root>/ops_profile/<run>``, when empty) and writes
    ``modules/consensus/`` (``rows_reference.parquet``, ``reference.npz``,
    ``reference.json``, ``run_<k>.npz``, ``consensus_partitions.npz``,
    ``consensus_runs.json``) and ``modules/atlas/`` (``module_members.csv``,
    ``module_atlas.csv``, ``module_summary.json``) beside them. Partitions
    already written are read back, so a stopped run resumes.

    :param settings: the OPS settings; missing keys take the defaults of
        :func:`spacr.ops_settings.ops_defaults`. Read here:
        ``ops_modules_runs``, ``ops_modules_precomputed_embedding``,
        ``ops_modules_profiles``, ``ops_modules_driver_profiles`` (the
        robust-z tables the driving features come from; the profile folder
        when empty), ``ops_reference_dir`` (the CORUM and KEGG group
        benchmarks, ``go_slim_groups.parquet``,
        ``string_v12_ge400.parquet``, ``biogrid_physical.parquet`` and
        ``gene_specificity.csv``), ``ops_control_prefix`` and ``plate``.
    :param report: called with progress lines.
    :returns: ``{"folder", "outputs", "summary"}``.
    :raises ValueError: when ``ops_modules_runs`` is below 1.
    :raises FileNotFoundError: when a profile or reference table is missing.
    :raises ImportError: when the ``clustering`` extra is not installed.
    """
    from .ops_clusters import consensus_rows
    from .ops_settings import ops_defaults
    from .tabular import write_table

    settings = ops_defaults(dict(settings))
    say = report or (lambda text: None)
    runs = int(settings.get("ops_modules_runs") or 0)
    if runs < 1:
        raise ValueError(f"ops_modules_runs must be 1 or more for the modules phase; got {runs}")
    precomputed = bool(settings.get("ops_modules_precomputed_embedding"))
    prefix = str(settings.get("ops_control_prefix") or "nontargeting")
    key = str(settings.get("plate") or "A")
    source = _profile_folder(settings)
    for name in ("genes.parquet", "guides.parquet"):
        if not os.path.isfile(os.path.join(source, name)):
            raise FileNotFoundError(f"the modules phase needs {os.path.join(source, name)}")
    genes = _read(os.path.join(source, "genes.parquet"))
    guides = _read(os.path.join(source, "guides.parquet"))
    columns = profile_columns(guides)
    consensus_dir = os.path.join(source, "modules", "consensus")
    atlas_dir = os.path.join(source, "modules", "atlas")
    os.makedirs(consensus_dir, exist_ok=True)
    os.makedirs(atlas_dir, exist_ok=True)
    outputs: Dict[str, str] = {}
    rows = consensus_rows(genes, guides, columns, control_prefix=prefix)
    outputs["rows_reference"] = write_table(rows, os.path.join(consensus_dir, "rows_reference.parquet"),
                                            canonicalise=False)
    reference_npz = os.path.join(consensus_dir, "reference.npz")
    if not os.path.isfile(reference_npz):
        ref = reference_partitions(rows, columns, precomputed=precomputed)
        np.savez(reference_npz, pcs=ref["pcs"], **{f"r{r}": v for r, v in ref["labels"].items()})
        with open(os.path.join(consensus_dir, "reference.json"), "w", encoding="utf-8") as handle:
            json.dump(ref["info"], handle, indent=1)
        say(json.dumps(ref["info"]))
    outputs["reference"] = reference_npz
    consensus = consensus_partitions(genes, guides, columns, runs=runs, precomputed=precomputed,
                                     folder=consensus_dir, report=say, control_prefix=prefix)
    partitions = os.path.join(consensus_dir, "consensus_partitions.npz")
    np.savez_compressed(partitions, labels=consensus["labels"], genes=rows.gene.to_numpy())
    with open(os.path.join(consensus_dir, "consensus_runs.json"), "w", encoding="utf-8") as handle:
        json.dump(consensus["runs"], handle, indent=1)
    outputs["consensus_partitions"] = partitions
    with np.load(reference_npz) as stored:
        pcs = stored["pcs"]
        reference = {r: stored[f"r{r}"] for r in MODULE_RESOLUTIONS.values()}
    reference_dir = settings.get("ops_reference_dir")
    background = set(rows.gene[~rows.gene.str.startswith("NTpseudo")])
    groups = module_groups(background, [
        ("CORUM", _read(_reference_path("CORUM", reference_dir), sep="\t")),
        ("KEGG", _read(_reference_path("KEGG", reference_dir), sep="\t")),
        ("GOslim", _read(_reference_path("GOslim", reference_dir)))])
    string = _read(_reference_path("STRING", reference_dir), columns=["a", "b", "combined_score"])
    biogrid = _read(_reference_path("BioGRID", reference_dir), columns=["a", "b"])
    specificity = _read(_reference_path("specificity", reference_dir))
    driver_genes, driver_guides = _driver_tables(settings.get("ops_modules_driver_profiles"), source)
    out = module_atlas(rows, pcs, reference, consensus["labels"], guides, driver_genes, driver_guides,
                       groups, string, biogrid, specificity, key=key, control_prefix=prefix)
    for name in ("members", "atlas"):
        outputs[f"module_{name}"] = write_table(out[name], os.path.join(atlas_dir, f"module_{name}.csv"),
                                                canonicalise=False)
    target = os.path.join(atlas_dir, "module_summary.json")
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(out["summary"], handle, indent=1, default=str)
    outputs["module_summary"] = target
    say(f"modules {key}: {sum(out['summary']['anchored_modules'].values())} anchored modules, "
        f"{out['summary']['stable_new_member_genes']} stable new member genes")
    return {"folder": os.path.join(source, "modules"), "outputs": outputs, "summary": out["summary"]}
