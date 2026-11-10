"""Gene classes of a pooled screen, gene-grouped folds and the class-prediction benchmark.

A gene class is a set of screened genes that share an annotation: a kinase
group, family or subfamily (KinHub, Manning et al. 2002), a CORUM complex, a
KEGG pathway, a GO-slim biological process, or a direction of one of Funk et
al. 2022's per-gene feature scores. :func:`build_gene_classes` builds the
class table from versioned source files (md5 recorded), merges or drops
near-duplicate classes, and splits the genes into folds with multi-label
iterative stratification (Sechidis et al. 2011, :func:`grouped_multilabel_folds`),
so every class has members in every fold and no gene is in two folds.

:func:`train_gene_class_baseline` fits one logistic regression per class on a
gene-level feature table, fold by fold, with optional decoy heads (random
gene sets of the classes' sizes, the null of the discovery calls) and a
shuffled-class control. :func:`evaluate_gene_class_run` scores the
out-of-fold predictions: per-class AUROC and average precision with
bootstrap intervals, specificity against phenotype-positive and
overlapping-class non-members, and the two discovery lists (known members
with a new image phenotype; predicted new functions) at BH q < 0.05 against
the decoy heads.

With ``ml_model='image'`` the classes are learned from single-cell crops
instead, by an image model (``train_arch``) trained fold by fold with
:mod:`spacr.gene_class_images`, and scored by the same evaluation.

:func:`run_gene_classes` runs the whole chain from a settings file. Every
seed is the reference's: folds 0, control pseudo-genes 0, decoys 99, label
shuffle 1, bootstrap 0, each offset by ``train_seed``.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .gene_class_images import ARCHITECTURE_CATEGORY, TRAIN_ARCH_TOOLTIP, TRAIN_PROTOCOL_TOOLTIP

#: Key the settings of this module are registered under.
APP_KEY = 'gene_classes'

#: The order sources win a near-duplicate tie in.
SOURCE_PRIORITY: Dict[str, int] = {"kinase": 0, "corum": 1, "kegg": 2, "goslim": 3, "funk": 4}

#: The KEGG BRITE top levels kept by ``gene_class_kegg_filter``.
KEGG_KEEP: Tuple[str, ...] = ("Metabolism", "Genetic Information Processing",
                              "Environmental Information Processing", "Cellular Processes")

#: The source files :func:`build_gene_classes` reads, by key.
SOURCE_KEYS: Tuple[str, ...] = ("screen", "kinases", "corum", "kegg_link", "kegg_list",
                                "kegg_hierarchy", "goslim_annotations", "goslim_obo", "design")

_DEFAULTS: Dict[str, Any] = {
    'gene_class_sources': '',
    'gene_class_gating': False,
    'gene_class_merge': True,
    'gene_class_merge_jaccard': 0.5,
    'gene_class_kegg_filter': True,
    'gene_class_min_train': 5,
    'gene_class_max_members': 400,
    'gene_class_gated_from': '',
    'train_label_source': 'gene_classes',
    'train_gene_classes': '',
    'train_split': 'grouped_multilabel',
    'train_split_group': 'gene',
    'train_folds': 5,
    'train_decoy_heads': 300,
    'train_shuffle_control': False,
    'train_seed': 0,
    'ml_model': 'logistic_regression',
    'train_arch': 'resnet18',
    'train_input_size': 224,
    'train_balanced_sampling': 'class_gene',
    'train_image_dataset': 'ops',
    'train_crops': '',
    'train_crop_index': '',
    'train_genotype': '',
    'train_crop_scales': '',
    'train_classes': '',
    'train_protocol': 'NSCB',
    'train_steps': 1500,
    'train_batch': 64,
    'train_features': '',
    'train_feature_suffix': '',
    'train_exclude_sources': '',
    'gene_class_eval_min_cells': 20,
    'gene_class_eval_boot': 1000,
    'gene_class_dst': '',
}

_EXPECTED_TYPES: Dict[str, Any] = {key: type(value) for key, value in _DEFAULTS.items()}
_EXPECTED_TYPES['gene_class_merge_jaccard'] = (float, int)

_TOOLTIPS: Dict[str, str] = {
    'gene_class_sources': "(str) - JSON file naming the annotation files the gene classes are built from: screen (the per-gene screen table with gene_symbol, gene_id, uniprot_accession and the per-feature _pvalue and _score columns), kinases (KinHub), corum, kegg_link, kegg_list, kegg_hierarchy (BRITE br08901), goslim_annotations (BioMart GO-slim), goslim_obo and design (the library design, for the non-targeting guides). Blank uses the class tables of train_gene_classes as they are. Default blank.",
    'gene_class_gating': "(bool) - Train each class only on its members with a screen phenotype (two or more features at BH q < 0.01); the other members are left out of the training, not used as negatives. Off trains on every screened member. Default False.",
    'gene_class_merge': "(bool) - Merge two classes whose members overlap by gene_class_merge_jaccard or more into the class of the higher-priority source (kinase, CORUM, KEGG, GO-slim, Funk). Off drops the lower-priority class instead. Default True.",
    'gene_class_merge_jaccard': "(float) - The Jaccard overlap at which two classes count as the same class and are merged or dropped. Lower merges more classes into fewer, larger ones. Default 0.5.",
    'gene_class_kegg_filter': "(bool) - Keep only KEGG maps under Metabolism, Genetic Information Processing, Environmental Information Processing and Cellular Processes; disease and organismal maps reuse the same signalling core and make hub genes. Default True.",
    'gene_class_min_train': "(int) - The fewest training members a class needs to be kept. Higher keeps fewer classes, each with more positives per fold. Default 5.",
    'gene_class_max_members': "(int) - The most screened members a class may have; larger classes, such as broad pathways, are left out. Higher keeps more broad classes. Default 400.",
    'gene_class_gated_from': "(str) - A class-table folder to take the classes and folds from, keeping only the classes with gene_class_min_train phenotype-positive members: the gated arm of an ungated table. Blank builds the classes from gene_class_sources. Default blank.",
    'train_label_source': "(str) - Where the training labels come from: 'gene_classes', the classes of a pooled screen's genes. Default 'gene_classes'.",
    'train_gene_classes': "(str) - Folder with the class tables (class_table.csv, class_members.csv, gene_folds.csv, nt_pseudogenes.csv, gene_universe.csv). When gene_class_sources is set they are built here first. Default blank.",
    'train_split': "(str) - How genes are split into folds: 'grouped_multilabel', iterative stratification over every class with each gene in one fold only. Default 'grouped_multilabel'.",
    'train_split_group': "(str) - What a fold keeps together: 'gene', so no gene is in a training and a test fold. Default 'gene'.",
    'train_folds': "(int) - How many folds the genes are split into; each fold is held out once while the others train. More folds train on more genes per model and take longer. Default 5.",
    'train_decoy_heads': "(int) - Random gene sets, drawn with the sizes of the real classes, trained and scored beside them; they are the null that a discovery call must beat. 0 trains none and makes no discovery calls. Default 300.",
    'train_shuffle_control': "(bool) - Shuffle the class labels among the targeting genes before training: the control that shows what chance alone gives. Default False.",
    'train_seed': "(int) - Added to every seed of the run (folds, decoy heads, label shuffle, bootstrap), so 0 gives the reference run. Default 0.",
    'ml_model': "(str) - The model: 'logistic_regression', one standardised L2 logistic regression per class on the gene feature table with balanced class weights, or 'image', an image model (train_arch) trained on single-cell crops with one output per class. Default 'logistic_regression'.",
    'train_arch': TRAIN_ARCH_TOOLTIP,
    'train_input_size': "(int) - Size the crops are resized to before an image model, on the device. Default 224.",
    'train_balanced_sampling': "(str) - How training crops are drawn: 'class_gene' (OPS: half the batch from genes in a class, class then gene then cell uniform; half from genes in no class and the control pseudo-genes), 'class' (uniform over classes; BBBC021) or 'none' (uniform over cells). Default 'class_gene'.",
    'train_image_dataset': "(str) - The crops of an image model: 'ops' (cell-masked TIFF crops in tar archives with byte-offset index tables, labels from the gene classes) or 'bbbc021' (a crop array with its index, labels the MoA). Default 'ops'.",
    'train_crops': "(str) - The crops of an image model: comma-separated folders of crop tar archives (ops) or the crop array .npy (bbbc021). Default blank.",
    'train_crop_index': "(str) - BBBC021: the crop index table, one row per crop, with plate, well, compound, concentration, moa, week and one scale column per channel. Default blank.",
    'train_genotype': "(str) - OPS: the per-cell genotype table (plate, well, site, label, sgRNA, gene). Default blank.",
    'train_crop_scales': "(str) - OPS: JSON file with each plate's per-channel scale (the 99th percentile of cell pixels). Default blank.",
    'train_classes': "(str) - OPS: comma-separated class ids an image model trains on; blank trains every class of the table. Default blank.",
    'train_protocol': TRAIN_PROTOCOL_TOOLTIP,
    'train_steps': "(int) - Training steps per fold of an image model. Default 1500.",
    'train_batch': "(int) - Crops per training step of an image model; on a GPU that runs out of memory the micro-batch halves and the batch is kept by accumulation. Default 64.",
    'train_features': "(str) - Gene-level feature table, one row per gene. Its gene column is gene or gene_symbol. Default blank.",
    'train_feature_suffix': "(str) - Use only the feature columns whose names end with this text, such as _score; blank uses every numeric column. Default blank.",
    'train_exclude_sources': "(str) - Comma-separated class sources left out, such as funk when the features are Funk's own scores and the funk classes would be circular. Default blank.",
    'gene_class_eval_min_cells': "(int) - The fewest cells a gene needs to be scored in the evaluation. Higher leaves out genes with few cells, whose scores are noisier. Default 20.",
    'gene_class_eval_boot': "(int) - Bootstrap resamples of the per-class AUROC and AP intervals. More resamples give steadier intervals and take longer. Default 1000.",
    'gene_class_dst': "(str) - Folder the run is written to: config.json, decoys.csv, one fold_<k> folder per fold and eval with per_class.csv, summary.json, discovery.csv, discovery_a.csv and discovery_b.csv. Default blank, which writes to results/gene_classes beside the class tables.",
}

_CATEGORIES: Dict[str, List[str]] = {
    ARCHITECTURE_CATEGORY: ['train_arch', 'train_protocol'],
    'Gene classes': ['gene_class_sources', 'gene_class_gating', 'gene_class_merge',
                     'gene_class_merge_jaccard', 'gene_class_kegg_filter', 'gene_class_min_train',
                     'gene_class_max_members', 'gene_class_gated_from'],
    'Gene-class training': ['train_label_source', 'train_gene_classes', 'train_split',
                            'train_split_group', 'train_folds', 'train_decoy_heads',
                            'train_shuffle_control', 'train_seed', 'ml_model',
                            'train_input_size', 'train_balanced_sampling', 'train_image_dataset',
                            'train_crops', 'train_crop_index', 'train_genotype', 'train_crop_scales',
                            'train_classes', 'train_steps', 'train_batch', 'train_features',
                            'train_feature_suffix', 'train_exclude_sources',
                            'gene_class_eval_min_cells', 'gene_class_eval_boot', 'gene_class_dst'],
}

_DESCRIPTION = (
    "Build gene classes from annotation files, split the genes into folds that keep every gene "
    "in one fold, train one classifier per class with decoy heads and a shuffled control, and "
    "score the held-out genes."
)


def gene_class_defaults(settings: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The settings of :func:`run_gene_classes`, filled in with their defaults.

    :param settings: settings given so far; a key already present is kept.
    :returns: a new dict with every key of this module.
    """
    merged = dict(settings or {})
    for key, value in _DEFAULTS.items():
        merged.setdefault(key, value)
    return merged


def _register() -> None:
    """Register this module's settings through the defaults seam, once per process."""
    from .settings import has_registered_defaults, register_defaults

    if has_registered_defaults(APP_KEY):
        return
    register_defaults(APP_KEY, gene_class_defaults, expected_types=_EXPECTED_TYPES,
                      tooltips=_TOOLTIPS, categories=_CATEGORIES, description=_DESCRIPTION)


_register()


def _read(path: Any, **options) -> pd.DataFrame:
    """Read a table exactly as written."""
    from .tabular import read_table

    return read_table(str(path), canonicalise=False, report=None, **options)


def _write(frame: pd.DataFrame, path: str, **options) -> str:
    """Write a table exactly as given."""
    from .tabular import write_table

    return write_table(frame, path, canonicalise=False, **options)


def _text(path: Any) -> str:
    """The text of a small annotation file."""
    with open(str(path), encoding="utf-8") as handle:
        return handle.read()


def file_md5(path: Any) -> str:
    """The md5 of a file, read in blocks.

    :param path: the file.
    :returns: the hex digest.
    """
    digest = hashlib.md5()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bh(p: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg q values; NaN p values stay NaN and are not counted.

    :param p: p values.
    :returns: the q values, capped at 1.
    """
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    m = np.isfinite(p)
    if m.sum() == 0:
        return out
    pv = p[m]
    order = np.argsort(pv)
    ranked = pv[order] * m.sum() / (np.arange(m.sum()) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty_like(pv)
    q[order] = np.minimum(ranked, 1)
    out[m] = q
    return out


def screen_table(path: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """The screen's gene table with its phenotype call, and the per-feature q values.

    :param path: the per-gene screen table (Funk et al. 2022's
        ``vesuvius_genes``): ``gene_symbol``, ``gene_id``,
        ``uniprot_accession`` and ``<feature>_pvalue`` / ``<feature>_score``
        columns.
    :returns: ``(screen, q)``. ``screen`` gains ``gene``,
        ``n_sig_features_q01``, ``min_q``, ``phenotype_funk`` (two or more
        features at BH q < 0.01; p values outside [0, 1] are missing) and
        ``uniprot_list``; ``q`` has ``gene``, ``<feature>_q`` and
        ``<feature>_sign``.
    """
    d = _read(path)
    d["gene"] = d["gene_symbol"].astype(str)
    pcols = [c for c in d.columns if c.endswith("_pvalue")]
    p = d[pcols].to_numpy(float)
    p[(p < 0) | (p > 1)] = np.nan
    q = np.column_stack([bh(p[:, j]) for j in range(p.shape[1])])
    d["n_sig_features_q01"] = (q < 0.01).sum(1)
    d["min_q"] = np.nanmin(np.where(np.isfinite(q), q, 1.0), 1)
    d["phenotype_funk"] = d["n_sig_features_q01"] >= 2
    qdf = pd.DataFrame(q, columns=[c.replace("_pvalue", "_q") for c in pcols])
    qdf.insert(0, "gene", d["gene"])
    for c in pcols:
        s = c.replace("_pvalue", "_score")
        if s in d:
            qdf[c.replace("_pvalue", "_sign")] = np.sign(d[s].to_numpy(float))
    d["uniprot_list"] = [ast.literal_eval(u) if isinstance(u, str) and u.startswith("[") else []
                         for u in d["uniprot_accession"]]
    return d, qdf


def kinase_classes(screen: pd.DataFrame, kinases: pd.DataFrame
                   ) -> Tuple[Dict[Tuple[str, str], Set[str]], pd.DataFrame]:
    """Kinase classes: all kinases, and each KinHub group, family and subfamily.

    :param screen: the table of :func:`screen_table`.
    :param kinases: KinHub's table (third column the HGNC name; ``Group``,
        ``Family``, ``SubFamily``, ``UniprotID``); a kinase is matched to a
        screened gene by UniProt accession, else by HGNC name.
    :returns: ``({("kinase", name): genes}, screened kinases)``.
    """
    by_uni = {}
    for g, accessions in zip(screen["gene"], screen["uniprot_list"]):
        for u in accessions:
            by_uni[u] = g
    sym = set(screen["gene"])
    hgnc = kinases.columns[2]
    rows = []
    for uni, name, group, family, subfamily in zip(kinases["UniprotID"], kinases[hgnc], kinases["Group"],
                                                   kinases["Family"], kinases["SubFamily"]):
        g = by_uni.get(str(uni)) or (name if name in sym else None)
        if g is None:
            continue
        rows.append((g, group, family, subfamily))
    kk = pd.DataFrame(rows, columns=["gene", "group", "family", "subfamily"]).drop_duplicates()
    out: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for r in kk.itertuples():
        out[("kinase", "all kinases")].add(r.gene)
        out[("kinase", f"group {r.group}")].add(r.gene)
        if isinstance(r.family, str):
            out[("kinase", f"family {r.group}/{r.family}")].add(r.gene)
        if isinstance(r.subfamily, str):
            out[("kinase", f"subfamily {r.group}/{r.family}/{r.subfamily}")].add(r.gene)
    return out, kk


def corum_classes(symbols: Set[str], corum: pd.DataFrame) -> Dict[Tuple[str, str], Set[str]]:
    """CORUM complexes as classes (Brieflow's ``group`` / ``gene_name`` table).

    :param symbols: the screened genes.
    :param corum: the complex table.
    :returns: ``{("corum", complex): genes}``.
    """
    out: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for g, grp in zip(corum["gene_name"], corum["group"]):
        if g in symbols:
            out[("corum", grp)].add(g)
    return out


def kegg_categories(hierarchy: Any) -> Dict[str, Tuple[str, str]]:
    """KEGG map number -> (BRITE top level, sub-category), from br08901.

    :param hierarchy: the ``br08901`` ``.keg`` file.
    :returns: the mapping.
    """
    cat, top, sub = {}, None, None
    for line in _text(hierarchy).splitlines():
        if line.startswith("A"):
            top = line[1:].strip()
        elif line.startswith("B"):
            sub = line[1:].strip()
        elif line.startswith("C"):
            cat[line[1:].split()[0]] = (top, sub)
    return cat


def kegg_classes(screen: pd.DataFrame, link: Any, names: Any) -> Dict[Tuple[str, str], Set[str]]:
    """KEGG pathways as classes, global and overview maps (``hsa011*``, ``hsa01100``) left out.

    :param screen: the table of :func:`screen_table` (``gene_id`` is the Entrez id).
    :param link: KEGG REST ``link/pathway/hsa`` (two columns, no header).
    :param names: KEGG REST ``list/pathway/hsa``.
    :returns: ``{("kegg", "<map> <name>"): genes}``.
    """
    links = _read(link, header=None, names=["g", "p"])
    listed = _read(names, header=None, names=["p", "name"])
    listed["p"] = "path:" + listed["p"]
    nm = dict(zip(listed["p"], listed["name"].str.replace(" - Homo sapiens (human)", "", regex=False)))
    by_id = dict(zip(screen["gene_id"].astype(str), screen["gene"]))
    out: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for g, p in zip(links["g"], links["p"]):
        gene = by_id.get(g.split(":")[1])
        if gene and not p.startswith("path:hsa011") and p != "path:hsa01100":
            out[("kegg", f"{p.split(':')[1]} {nm.get(p, '')}")].add(gene)
    return out


def filter_kegg(raw: Mapping[Tuple[str, str], Set[str]], hierarchy: Any,
                keep: Sequence[str] = KEGG_KEEP) -> Tuple[Dict[Tuple[str, str], Set[str]], pd.DataFrame]:
    """Drop the KEGG maps outside the kept BRITE top levels.

    :param raw: every class.
    :param hierarchy: the ``br08901`` ``.keg`` file.
    :param keep: the top levels kept.
    :returns: ``(kept classes, dropped maps)``; the dropped table has
        ``pathway``, ``brite_A``, ``brite_B`` and ``n_screened_members``.
    """
    cat = kegg_categories(hierarchy)
    kept, dropped = {}, []
    for (src, name), members in raw.items():
        if src != "kegg":
            kept[(src, name)] = members
            continue
        top, sub = cat.get(name.split()[0].replace("hsa", ""), ("unknown", "unknown"))
        if top in keep:
            kept[(src, name)] = members
        else:
            dropped.append({"pathway": name, "brite_A": top, "brite_B": sub, "n_screened_members": len(members)})
    return kept, pd.DataFrame(dropped)


def goslim_classes(screen: pd.DataFrame, annotations: Any, obo: Any) -> Dict[Tuple[str, str], Set[str]]:
    """GO-slim biological processes as classes.

    :param screen: the table of :func:`screen_table`.
    :param annotations: BioMart's GO-slim export (Ensembl id, name, Entrez
        id, GO id, description).
    :param obo: ``goslim_generic.obo``, for each term's namespace.
    :returns: ``{("goslim", "<GO id> <description>"): genes}``.
    """
    b = _read(annotations, sep="\t", dtype=str)
    b.columns = ["ens", "name", "entrez", "go", "desc"]
    ns, cur = {}, None
    for line in _text(obo).splitlines():
        if line.startswith("id: GO:"):
            cur = line[4:].strip()
        elif line.startswith("namespace:") and cur:
            ns[cur] = line.split(":", 1)[1].strip()
    by_id = dict(zip(screen["gene_id"].astype(str), screen["gene"]))
    sym = set(screen["gene"])
    out: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for r in b.dropna(subset=["go"]).itertuples():
        if ns.get(r.go) != "biological_process":
            continue
        gene = by_id.get(str(r.entrez).split(".")[0]) or (r.name if r.name in sym else None)
        if gene:
            out[("goslim", f"{r.go} {r.desc}")].add(gene)
    return out


def funk_classes(screen: pd.DataFrame, q: pd.DataFrame, clusters: bool = False
                 ) -> Dict[Tuple[str, str], Set[str]]:
    """Feature-direction classes: the genes high (or low) on one screen feature at q < 0.01.

    :param screen: the table of :func:`screen_table`.
    :param q: its q-value table.
    :param clusters: also add the screen's own interphase and mitotic clusters.
    :returns: ``{("funk", name): genes}``.
    """
    out: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for col, tag in (("interphase_cluster", "interphase cluster"), ("mitotic_cluster", "mitotic cluster")):
        for g, c in zip(screen["gene"], screen[col]):
            if clusters and np.isfinite(c):
                out[("funk", f"{tag} {int(c)}")].add(g)
    for qc in [c for c in q.columns if c.endswith("_q")]:
        base = qc[:-2]
        sign = q.get(base + "_sign")
        if sign is None:
            continue
        for direction, s in (("high", 1), ("low", -1)):
            m = (q[qc] < 0.01) & (sign == s)
            out[("funk", f"{direction} {base}")] = set(q.loc[m, "gene"])
    return out


def grouped_multilabel_folds(Y: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    """Multi-label iterative stratification of genes into folds (Sechidis et al. 2011).

    Each gene goes to one fold, so a fold split by these labels is grouped by
    gene. The rarest remaining class is placed first, each of its genes into
    the fold that most lacks that class.

    :param Y: a ``genes x classes`` 0/1 matrix.
    :param k: the number of folds.
    :param seed: the seed of the tie-breaking draws.
    :returns: one fold per gene, ``0 .. k-1``.
    """
    rng = np.random.default_rng(seed)
    n = len(Y)
    fold = -np.ones(n, int)
    desired = np.full(k, n / k)
    desired_c = np.tile(Y.sum(0) / k, (k, 1)).astype(float)
    left = np.ones(n, bool)
    while left.any():
        counts = Y[left].sum(0)
        if counts.max() == 0:
            idx = np.flatnonzero(left)
            rng.shuffle(idx)
            for i in idx:
                f = int(np.argmax(desired + rng.random(k) * 1e-6))
                fold[i] = f
                desired[f] -= 1
            break
        c = int(np.argmin(np.where(counts > 0, counts, np.inf)))
        idx = np.flatnonzero(left & (Y[:, c] > 0))
        rng.shuffle(idx)
        for i in idx:
            score = desired_c[:, c] + rng.random(k) * 1e-6
            best = np.flatnonzero(score == score.max())
            f = int(best[np.argmax(desired[best])])
            fold[i] = f
            left[i] = False
            desired[f] -= 1
            desired_c[f] -= Y[i]
    return fold


def nt_pseudogenes(design: Any, folds: int = 5, seed: int = 0, size: int = 4,
                   control: str = "nontargeting") -> pd.DataFrame:
    """The non-targeting guides grouped at random into pseudo-genes, each in one fold.

    :param design: the library design table (``gene_symbol``, ``sgRNA``).
    :param folds: the number of folds.
    :param seed: the seed of the grouping and the fold draw.
    :param size: guides per pseudo-gene.
    :param control: the gene symbol of the control guides.
    :returns: ``sgRNA, pseudogene, fold``; pseudo-genes are ``NT_<k>``.
    """
    d = _read(design)
    nt = sorted(d.loc[d["gene_symbol"] == control, "sgRNA"].unique())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(nt))
    out = pd.DataFrame({"sgRNA": np.array(nt)[perm]})
    out["pseudogene"] = [f"NT_{i // size:03d}" for i in range(len(out))]
    pg = out["pseudogene"].unique()
    pf = dict(zip(pg, rng.permutation(np.arange(len(pg)) % folds)))
    out["fold"] = out["pseudogene"].map(pf)
    return out


def _select_classes(raw, symbols, phenotype, *, gating, merge, jaccard, min_train, max_members):
    """Size limits and the near-duplicate rule, as :func:`build_gene_classes` documents them.

    :returns: ``(candidates, merges)``: every candidate with its status, and
        one row per merged class.
    """
    rows = []
    for (src, name), members in raw.items():
        members = set(members) & symbols
        train = members & phenotype if gating else set(members)
        rows.append({"source": src, "name": name, "members": members, "trainable": train,
                     "n_members_screened": len(members), "n_trainable": len(train)})
    cand = pd.DataFrame(rows)
    keep = (cand["n_trainable"] >= min_train) & (cand["n_members_screened"] <= max_members)
    keep &= ~((cand["source"] == "corum") & (cand["n_members_screened"] < 5))
    cand["status"] = np.where(keep, "candidate", "too_small_or_large")
    cand["prio"] = cand["source"].map(SOURCE_PRIORITY)
    cand["absorbed"] = ""
    order = cand[keep].sort_values(["prio", "n_trainable"]).index
    kept, merges = [], []
    key = "members" if merge else "trainable"
    for i in order:
        t = cand.at[i, key]
        dup = None
        for j in kept:
            u = cand.at[j, key]
            if len(t & u) / len(t | u) >= jaccard:
                dup = j
                break
        if dup is None:
            kept.append(i)
            cand.at[i, "status"] = "kept"
        elif merge:
            jac = len(t & cand.at[dup, key]) / len(t | cand.at[dup, key])
            merges.append({"into_source": cand.at[dup, "source"], "into_name": cand.at[dup, "name"],
                           "absorbed_source": cand.at[i, "source"], "absorbed_name": cand.at[i, "name"],
                           "jaccard": round(jac, 3), "absorbed_size": len(t),
                           "into_size_before": len(cand.at[dup, key])})
            cand.at[dup, "members"] = cand.at[dup, "members"] | cand.at[i, "members"]
            cand.at[dup, "trainable"] = cand.at[dup, "trainable"] | cand.at[i, "trainable"]
            cand.at[dup, "absorbed"] = (cand.at[dup, "absorbed"] + " | " if cand.at[dup, "absorbed"] else "") + \
                f"{cand.at[i, 'source']}:{cand.at[i, 'name']}"
            cand.at[i, "status"] = f"merged_into:{cand.at[dup, 'source']}:{cand.at[dup, 'name']}"
        else:
            cand.at[i, "status"] = f"duplicate_of:{cand.at[dup, 'source']}:{cand.at[dup, 'name']}"
    cand["n_members_screened"] = cand["members"].map(len)
    cand["n_trainable"] = cand["trainable"].map(len)
    cand["n_absorbed"] = cand["absorbed"].map(lambda a: len(a.split(" | ")) if a else 0)
    cand = cand.sort_values(["prio", "name"]).reset_index(drop=True)
    k = cand["status"] == "kept"
    cand.loc[k, "class_id"] = [f"C{i:03d}" for i in range(int(k.sum()))]
    return cand, merges


def build_gene_classes(sources: Mapping[str, Any], settings: Mapping[str, Any], out_dir: str
                       ) -> Dict[str, Any]:
    """Build the class tables and gene folds from versioned annotation files.

    Every annotated screened gene is a member of its class; the trainable
    members are all of them, or with ``gene_class_gating`` only those with a
    screen phenotype. A class is kept with ``gene_class_min_train`` or more
    trainable members and at most ``gene_class_max_members`` members (a
    CORUM complex needs five members). Classes are then visited by source
    priority (kinase, CORUM, KEGG, GO-slim, Funk) and size; one whose
    members (or trainable members, without ``gene_class_merge``) overlap a
    kept class by ``gene_class_merge_jaccard`` is merged into it or dropped.

    :param sources: the files of :data:`SOURCE_KEYS`.
    :param settings: read here: ``gene_class_gating``, ``gene_class_merge``,
        ``gene_class_merge_jaccard``, ``gene_class_kegg_filter``,
        ``gene_class_min_train``, ``gene_class_max_members``,
        ``train_folds`` and ``train_seed``.
    :param out_dir: where the tables go: ``class_table.csv``,
        ``class_table_all_candidates.csv``, ``class_members.csv``,
        ``merge_table.csv`` (with merging), ``kegg_dropped_pathways.csv``
        (with the KEGG filter), ``kinase_screened.csv``,
        ``gene_universe.csv``, ``gene_folds.csv``,
        ``class_trainable_per_fold.csv``, ``nt_pseudogenes.csv``,
        ``classes_per_gene.csv``, ``class_summary.json`` and
        ``sources.json`` (each source's path and md5).
    :returns: the class summary.
    :raises KeyError: when a source file is missing from ``sources``.
    """
    settings = gene_class_defaults(settings)
    missing = [k for k in SOURCE_KEYS if not sources.get(k)]
    if missing:
        raise KeyError(f"gene_class_sources is missing {missing}")
    os.makedirs(out_dir, exist_ok=True)
    gating = bool(settings["gene_class_gating"])
    merge = bool(settings["gene_class_merge"])
    folds_k = int(settings["train_folds"])
    seed = int(settings["train_seed"])
    screen, q = screen_table(sources["screen"])
    sym = set(screen["gene"])
    pheno_col = "phenotype_funk"
    pheno = set(screen.loc[screen[pheno_col], "gene"])
    genes = screen["gene"].tolist()
    gi = {g: i for i, g in enumerate(genes)}
    raw: Dict[Tuple[str, str], Set[str]] = {}
    kin, kk = kinase_classes(screen, _read(sources["kinases"]))
    for part in (kin, corum_classes(sym, _read(sources["corum"], sep="\t")),
                 kegg_classes(screen, sources["kegg_link"], sources["kegg_list"]),
                 goslim_classes(screen, sources["goslim_annotations"], sources["goslim_obo"]),
                 funk_classes(screen, q)):
        raw.update(part)
    _write(kk, os.path.join(out_dir, "kinase_screened.csv"))
    if settings["gene_class_kegg_filter"]:
        raw = {k: set(v) & sym for k, v in raw.items()}
        raw, dropped = filter_kegg(raw, sources["kegg_hierarchy"])
        _write(dropped.sort_values(["brite_A", "pathway"]), os.path.join(out_dir, "kegg_dropped_pathways.csv"))
    cand, merges = _select_classes(raw, sym, pheno, gating=gating, merge=merge,
                                   jaccard=float(settings["gene_class_merge_jaccard"]),
                                   min_train=int(settings["gene_class_min_train"]),
                                   max_members=int(settings["gene_class_max_members"]))
    k = cand["status"] == "kept"
    rule = pheno_col if gating else "none (all screened members train)"
    table = cand.drop(columns=["members", "trainable", "prio"])
    table["phenotype_rule"] = rule
    _write(table, os.path.join(out_dir, "class_table_all_candidates.csv"))
    kept_t = cand[k].copy()
    kept_t["n_phenotype_funk"] = kept_t["members"].map(lambda m: len(m & pheno))
    _write(kept_t.assign(phenotype_rule=rule)[
        ["class_id", "source", "name", "n_members_screened", "n_trainable", "n_phenotype_funk", "n_absorbed",
         "absorbed", "phenotype_rule"]], os.path.join(out_dir, "class_table.csv"))
    mem = [(r.class_id, g, g in r.trainable, g in pheno) for r in kept_t.itertuples() for g in sorted(r.members)]
    _write(pd.DataFrame(mem, columns=["class_id", "gene", "trainable", "phenotype_funk"]),
           os.path.join(out_dir, "class_members.csv"))
    if merge:
        mt = pd.DataFrame(merges)
        cid = dict(zip(zip(kept_t["source"], kept_t["name"]), kept_t["class_id"]))
        if len(mt):
            mt.insert(0, "into_class_id", [cid.get((a, b)) for a, b in zip(mt.into_source, mt.into_name)])
        _write(mt, os.path.join(out_dir, "merge_table.csv"))
    Y = np.zeros((len(genes), len(kept_t)), np.int8)
    for j, r in enumerate(kept_t.itertuples()):
        for g in r.trainable:
            Y[gi[g], j] = 1
    fold = grouped_multilabel_folds(Y, folds_k, seed=seed)
    uni = screen[["gene", "gene_id", "phenotype_funk", "n_sig_features_q01", "min_q"]].copy()
    uni["n_classes_trainable"] = Y.sum(1)
    uni["fold"] = fold
    _write(uni, os.path.join(out_dir, "gene_universe.csv"))
    _write(uni[["gene", "fold"]], os.path.join(out_dir, "gene_folds.csv"))
    per_fold = pd.DataFrame(Y).groupby(fold).sum().T
    per_fold.index = kept_t["class_id"].tolist()
    _write(per_fold, os.path.join(out_dir, "class_trainable_per_fold.csv"), index=True)
    ntdf = nt_pseudogenes(sources["design"], folds_k, seed=seed)
    _write(ntdf, os.path.join(out_dir, "nt_pseudogenes.csv"))
    cpg = pd.Series(Y.sum(1), index=genes)
    _write(cpg.rename("n_classes").to_frame(), os.path.join(out_dir, "classes_per_gene.csv"), index=True,
           index_label="gene")
    summary = {"screen_genes": len(genes), "phenotype_rule": rule,
               "classes_per_gene": {"mean_in_any": float(cpg[cpg > 0].mean()),
                                    "median_in_any": float(cpg[cpg > 0].median()),
                                    "p98": float(np.percentile(cpg, 98)), "max": int(cpg.max()),
                                    "genes_gt20": int((cpg > 20).sum()), "frac_gt20": float((cpg > 20).mean()),
                                    "top10": cpg.sort_values(ascending=False).head(10).to_dict()},
               "kegg_filter": bool(settings["gene_class_kegg_filter"]), "funk_phenotype_positive": len(pheno),
               "candidates": int(len(cand)), "kept": int(k.sum()),
               "kept_by_source": kept_t["source"].value_counts().to_dict(),
               "merge_or_duplicate_jaccard": settings["gene_class_merge_jaccard"],
               "merge_mode": "merge (union)" if merge else "drop",
               "merged_or_dropped": int(cand["status"].str.startswith(("duplicate", "merged")).sum()),
               "trainable_genes": int((Y.sum(1) > 0).sum()),
               "kinase_classes": kept_t.loc[kept_t.source == "kinase", "name"].tolist(),
               "nt_guides": len(ntdf), "nt_pseudogenes": int(ntdf["pseudogene"].nunique()),
               "classes_with_empty_fold": int((per_fold.min(1) == 0).sum())}
    _dump(summary, os.path.join(out_dir, "class_summary.json"))
    _dump({key: {"path": str(sources[key]), "md5": file_md5(sources[key])} for key in SOURCE_KEYS},
          os.path.join(out_dir, "sources.json"))
    return summary


def gate_gene_classes(source_dir: str, screen: Any, out_dir: str, min_train: int = 5) -> Dict[str, Any]:
    """The gated arm of a class table: the same classes and folds, phenotype-positive members train.

    :param source_dir: the ungated class tables.
    :param screen: the per-gene screen table of :func:`screen_table`.
    :param out_dir: where the gated tables go.
    :param min_train: the fewest phenotype-positive members a class keeps.
    :returns: the summary written to ``class_summary.json``.
    """
    os.makedirs(out_dir, exist_ok=True)
    table, _ = screen_table(screen)
    pheno = set(table.loc[table["phenotype_funk"], "gene"])
    ct = _read(os.path.join(source_dir, "class_table.csv"))
    cm = _read(os.path.join(source_dir, "class_members.csv"))
    cm["trainable"] = cm["gene"].isin(pheno)
    n_tr = cm.groupby("class_id")["trainable"].sum()
    ct["n_trainable"] = ct["class_id"].map(n_tr).fillna(0).astype(int)
    keep = ct[ct["n_trainable"] >= int(min_train)]
    _write(keep.assign(phenotype_rule="phenotype_funk"), os.path.join(out_dir, "class_table.csv"))
    _write(cm[cm["class_id"].isin(keep["class_id"])], os.path.join(out_dir, "class_members.csv"))
    for name in ("gene_folds.csv", "nt_pseudogenes.csv", "gene_universe.csv", "merge_table.csv"):
        path = os.path.join(source_dir, name)
        if os.path.isfile(path):
            with open(path, "rb") as src, open(os.path.join(out_dir, name), "wb") as dst:
                dst.write(src.read())
    summary = {"arm": "gated (supplement)", "classes_from": str(source_dir), "phenotype_rule": "phenotype_funk",
               "kept": int(len(keep)), "dropped_below_min_train": int(len(ct) - len(keep)),
               "kept_by_source": keep["source"].value_counts().to_dict(),
               "trainable_genes": int(cm[cm["class_id"].isin(keep["class_id"]) & cm["trainable"]]["gene"].nunique())}
    _dump(summary, os.path.join(out_dir, "class_summary.json"))
    return summary


def _dump(record: Mapping[str, Any], path: str) -> None:
    """Write a JSON record with one-space indents."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(record, indent=1, default=str))


def _feature_table(path: Any, suffix: str) -> pd.DataFrame:
    """One row per gene, its numeric (or ``suffix``) columns, gaps filled with the column median."""
    d = _read(path)
    gene = "gene" if "gene" in d else "gene_symbol"
    X = d.set_index(gene)
    X = X[[c for c in X.columns if c.endswith(suffix)]] if suffix else X.select_dtypes("number")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.loc[:, X.isna().mean() < 0.05].fillna(X.median())


def train_gene_class_baseline(features: pd.DataFrame, tables: str, out_dir: str, *,
                              decoys: int = 0, shuffle: bool = False, exclude_sources: Sequence[str] = (),
                              seed: int = 0, report: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Per-class logistic regression on a gene feature table, fold by fold.

    Per fold the features are standardised on the training genes; for every
    class an L2 logistic regression (C = 1, balanced class weights) is fitted
    on its trainable members against the non-members (members that do not
    train are left out) and scores the held-out genes. A class with fewer
    than two training positives in a fold is not scored there.

    :param features: one row per gene (index) and its features.
    :param tables: the class-table folder.
    :param out_dir: the run folder: ``config.json``, ``decoys.csv`` and
        ``fold_<k>/gene_scores.parquet`` (genes x classes, ``n_cells`` 999).
    :param decoys: decoy heads: random gene sets with sizes drawn from the
        real classes, members drawn from the genes in any class (seed
        ``99 + seed``).
    :param shuffle: shuffle the labels among the targeting genes (seed
        ``1 + seed``).
    :param exclude_sources: class sources left out.
    :param seed: added to the decoy and shuffle seeds.
    :param report: called with one line per fold.
    :returns: the run configuration.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    say = report or (lambda text: None)
    cm = _read(os.path.join(tables, "class_members.csv"))
    ct = _read(os.path.join(tables, "class_table.csv")).set_index("class_id")
    classes = [c for c in ct.index if ct.at[c, "source"] not in set(exclude_sources)]
    folds = _read(os.path.join(tables, "gene_folds.csv"))
    nt = _read(os.path.join(tables, "nt_pseudogenes.csv"))
    fold = dict(zip(folds.gene, folds.fold))
    fold.update(dict(zip(nt.pseudogene, nt.fold)))
    X = features
    genes = np.array(X.index)
    f = np.array([fold.get(g, -1) for g in genes])
    Y = pd.DataFrame(0, index=genes, columns=classes, dtype=np.int8)
    U = pd.DataFrame(False, index=genes, columns=classes)
    for r in cm[cm.class_id.isin(classes)].itertuples():
        if r.gene in Y.index:
            if r.trainable:
                Y.at[r.gene, r.class_id] = 1
            else:
                U.at[r.gene, r.class_id] = True
    os.makedirs(out_dir, exist_ok=True)
    if decoys:
        rng = np.random.default_rng(99 + seed)
        pool = [g for g in genes if not str(g).startswith("NT_") and Y.loc[g].sum() > 0]
        sizes = rng.choice(Y.sum(0).to_numpy().astype(int), decoys)
        heads, columns = [], {}
        for j, size in enumerate(sizes):
            c = f"DECOY{j:03d}"
            col = pd.Series(0, index=genes, dtype=np.int8)
            for g in rng.choice(pool, int(size), replace=False):
                col.at[g] = 1
                heads.append((c, g))
            columns[c] = col
            classes.append(c)
        Y = pd.concat([Y, pd.DataFrame(columns)], axis=1)
        U = pd.concat([U, pd.DataFrame(False, index=genes, columns=list(columns))], axis=1)
        _write(pd.DataFrame(heads, columns=["class_id", "gene"]), os.path.join(out_dir, "decoys.csv"))
    if shuffle:
        rng = np.random.default_rng(1 + seed)
        real = ~pd.Index(genes).astype(str).str.startswith("NT_")
        src = rng.permutation(np.flatnonzero(real))
        Y.iloc[np.flatnonzero(real)] = Y.iloc[src].to_numpy()
        U.iloc[np.flatnonzero(real)] = U.iloc[src].to_numpy()
    config = {"baseline": "logistic", "n_features": int(X.shape[1]), "shuffle_labels": bool(shuffle),
              "tables": str(tables), "decoys": int(decoys), "classes": classes}
    _dump(config, os.path.join(out_dir, "config.json"))
    for k in sorted(set(f) - {-1}):
        tr, te = f != k, f == k
        tr &= f != -1
        sc = StandardScaler().fit(X.values[tr])
        Xtr, Xte = sc.transform(X.values[tr]), sc.transform(X.values[te])
        scores = {}
        for c in classes:
            y = Y[c].values[tr]
            keep = ~U[c].values[tr]
            if y[keep].sum() < 2:
                scores[c] = np.full(int(te.sum()), np.nan)
                continue
            model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(Xtr[keep], y[keep])
            scores[c] = model.predict_proba(Xte)[:, 1]
        S = pd.DataFrame(scores, index=genes[te], columns=classes, dtype=float)
        S["n_cells"] = 999
        os.makedirs(os.path.join(out_dir, f"fold_{k:02d}"), exist_ok=True)
        S.to_parquet(os.path.join(out_dir, f"fold_{k:02d}", "gene_scores.parquet"))
        say(f"fold {k}: {int(te.sum())} genes")
    return config


def fast_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUROC with ties counted half, equal to ``sklearn.metrics.roc_auc_score``.

    :param pos: scores of the positives.
    :param neg: scores of the negatives.
    :returns: the AUROC.
    """
    ns = np.sort(neg)
    lo = np.searchsorted(ns, pos, side="left")
    hi = np.searchsorted(ns, pos, side="right")
    return float((lo + 0.5 * (hi - lo)).sum() / (len(pos) * len(neg)))


def fast_ap(pos: np.ndarray, neg: np.ndarray) -> float:
    """Average precision of the positives ranked among the negatives.

    :param pos: scores of the positives.
    :param neg: scores of the negatives.
    :returns: ``sklearn.metrics.average_precision_score``.
    """
    from sklearn.metrics import average_precision_score

    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(average_precision_score(y, np.r_[pos, neg]))


def boot_ci(pos: np.ndarray, neg: np.ndarray, B: int, rng: np.random.Generator
            ) -> Tuple[np.ndarray, np.ndarray]:
    """Bootstrap 95 % intervals of the AUROC and AP, positives and negatives resampled apart.

    :param pos: scores of the positives.
    :param neg: scores of the negatives.
    :param B: resamples.
    :param rng: the generator (advanced).
    :returns: ``(auroc [lo, hi], ap [lo, hi])``.
    """
    a, p = [], []
    for _ in range(int(B)):
        ps, ns = rng.choice(pos, len(pos)), rng.choice(neg, len(neg))
        a.append(fast_auc(ps, ns))
        p.append(fast_ap(ps, ns))
    return np.percentile(a, [2.5, 97.5]), np.percentile(p, [2.5, 97.5])


def _class_rows(G, classes, cm, ct, decoy_sets, funk_pos, boot, rng):
    """One evaluation row per class and decoy head; see :func:`evaluate_gene_class_run`."""
    cset = {c: set(g) for c, g in cm.groupby("class_id")["gene"]}
    g2c = cm.groupby("gene")["class_id"].apply(set).to_dict()
    rows = []
    for c in classes:
        s = G[c].dropna()
        if c.startswith("DECOY"):
            members = train = decoy_sets.get(c, set())
            src, name = "decoy", c
        else:
            m = cm[cm["class_id"] == c]
            members, train = set(m["gene"]), set(m.loc[m["trainable"], "gene"])
            src, name = ct.at[c, "source"], ct.at[c, "name"]
        pos = s[s.index.isin(train)].to_numpy()
        neg_all = s[~s.index.isin(members)].to_numpy()
        neg_nt = s[s.index.str.startswith("NT_")].to_numpy()
        r = {"class_id": c, "source": src, "name": name, "n_pos": len(pos), "n_neg": len(neg_all),
             "n_nt": len(neg_nt), "prevalence": len(pos) / max(len(pos) + len(neg_all), 1)}
        if len(pos) >= 2 and len(neg_all) >= 2:
            r["auroc"], r["ap"] = fast_auc(pos, neg_all), fast_ap(pos, neg_all)
            (r["auroc_lo"], r["auroc_hi"]), (r["ap_lo"], r["ap_hi"]) = boot_ci(pos, neg_all, boot, rng)
            if len(neg_nt) >= 2:
                r["auroc_vs_nt"] = fast_auc(pos, neg_nt)
            neg_ph = s[~s.index.isin(members) & s.index.isin(funk_pos)].to_numpy()
            if len(neg_ph) >= 2:
                r["auroc_vs_pheno"], r["ap_vs_pheno"] = fast_auc(pos, neg_ph), fast_ap(pos, neg_ph)
            if c in cset:
                over = set().union(*(cset[o] for g in cset[c] for o in g2c.get(g, ()) if o != c)) \
                    if cset[c] else set()
                neg_ov = s[s.index.isin(over - members)].to_numpy()
                r["n_neg_overlap"] = len(neg_ov)
                if len(neg_ov) >= 2:
                    r["auroc_vs_overlap"] = fast_auc(pos, neg_ov)
                uniq = [g for g in train if all(o == c for o in g2c.get(g, {c}))]
                pu = s[s.index.isin(uniq)].to_numpy()
                r["n_unique_members"] = len(pu)
                if len(pu) >= 2:
                    r["auroc_unique_members"] = fast_auc(pu, neg_all)
            for tag, sel in (("funkpos", s.index.isin(members) & s.index.isin(funk_pos)),
                             ("funkneg", s.index.isin(members) & ~s.index.isin(funk_pos))):
                pp = s[sel].to_numpy()
                r[f"n_{tag}"] = len(pp)
                if len(pp) >= 2:
                    r[f"auroc_{tag}"], r[f"ap_{tag}"] = fast_auc(pp, neg_all), fast_ap(pp, neg_all)
        rows.append(r)
    return pd.DataFrame(rows), cset, g2c


def _discovery(G, classes, P, cm, ct, cset, g2c, decoy_sets, funk_pos):
    """The discovery calls against the decoy heads; see :func:`evaluate_gene_class_run`."""
    Z = (G[classes] - G[classes].mean()) / G[classes].std()
    dsize = {c: len(v) for c, v in decoy_sets.items()}
    Pi = P.set_index("class_id")
    empty = pd.Series(dtype=float)
    spec_ok = {c: bool((Pi.at[c, "auroc_lo"] > 0.5) and (Pi.get("auroc_vs_pheno", empty).get(c, 0) > 0.5)
                       and (Pi.get("auroc_vs_overlap", empty).get(c, 0) > 0.5))
               if c in Pi.index and "auroc_lo" in Pi and pd.notna(Pi.at[c, "auroc_lo"]) else False
               for c in classes}
    disc = []
    if dsize:
        realc = [c for c in classes if not c.startswith("DECOY")]
        Mem = pd.DataFrame(False, index=Z.index, columns=realc)
        for c in realc:
            Mem.loc[Mem.index.isin(cset.get(c, set())), c] = True
        fpos = Z.index.isin(funk_pos)
        notnt = ~Z.index.str.startswith("NT_")
        for c in realc:
            sz = int((cm["class_id"] == c).sum())
            md = [d for d, n in dsize.items() if sz / 2 <= n <= sz * 2 and d in Z] or [d for d in dsize if d in Z]
            null = np.sort(Z[md].to_numpy().ravel())
            null = null[np.isfinite(null)]
            z = Z[c].to_numpy()
            ok = np.isfinite(z) & notnt
            if len(md) >= 100:
                Zd = Z[md].to_numpy()
                pv = (1 + (Zd >= z[:, None]).sum(1)) / (1 + np.isfinite(Zd).sum(1))
            else:
                pv = (1 + len(null) - np.searchsorted(null, np.nan_to_num(z, nan=-np.inf), side="left")) \
                    / (1 + len(null))
            mem = Mem[c].to_numpy()
            over = sorted({o for g in cset.get(c, ()) for o in g2c.get(g, ()) if o != c and o in Mem.columns})
            zo = Z[over].where(~Mem[over]).max(1).to_numpy() if over else np.full(len(z), -np.inf)
            zo = np.where(np.isfinite(zo), zo, -np.inf)
            sel_a = ok & mem & ~fpos
            sel_b = ok & ~mem
            spec_b = bool(spec_ok.get(c)) & (z > zo)
            for sel, typ, spec in ((sel_a, "a_member_new_phenotype", np.ones(len(z), bool)),
                                   (sel_b, "b_new_function", spec_b)):
                for k in np.flatnonzero(sel):
                    disc.append((typ, c, ct.at[c, "source"], ct.at[c, "name"], Z.index[k], bool(fpos[k]),
                                 float(G[c].iloc[k]) if Z.index[k] in G.index else np.nan, float(z[k]),
                                 float(pv[k]), len(md), bool(spec[k])))
    D = pd.DataFrame(disc, columns=["type", "class_id", "source", "name", "gene", "funk_positive", "score", "z",
                                    "p_vs_decoys", "n_matched_decoys", "passes_specificity"])
    if len(D):
        from statsmodels.stats.multitest import multipletests

        D["q"] = np.nan
        for _, gidx in D.groupby("type").groups.items():
            D.loc[gidx, "q"] = multipletests(D.loc[gidx, "p_vs_decoys"], method="fdr_bh")[1]
        D = D[(D["q"] < 0.05) & D["passes_specificity"]].sort_values(["type", "q", "z"],
                                                                      ascending=[True, True, False])
    return D, dsize


def evaluate_gene_class_run(run: str, *, min_cells: int = 20, boot: int = 1000, seed: int = 0,
                            tag: str = "") -> Dict[str, Any]:
    """Score a gene-class run out of fold: per class, specificity, discovery and a summary.

    Per class (and decoy head) the positives are the held-out trainable
    members and the negatives every scored non-member; AUROC and AP with
    bootstrap intervals, AUROC against control pseudo-genes, against
    phenotype-positive non-members and against members of overlapping
    classes, for members unique to the class, and for members with and
    without a screen phenotype. A class is specific when its AUROC interval
    is above 0.5 and both specificity AUROCs are above 0.5. Discovery: each
    head is z-scored over genes; with 100 or more size-matched decoy heads a
    gene's p value is its rank on the decoy heads (the gene-wise null), else
    against the pooled decoy z; BH within each type: (a) members without a
    screen phenotype that score as members (a new image phenotype), (b)
    non-members of a specific class whose best class it is (a predicted new
    function).

    :param run: a run folder of :func:`train_gene_class_baseline`.
    :param min_cells: the fewest cells a gene needs to be scored.
    :param boot: bootstrap resamples.
    :param seed: the bootstrap seed.
    :param tag: the run name in the summary; the folder name when blank.
    :returns: the summary; ``eval/`` gets ``per_class.csv``,
        ``summary.json``, ``gene_class_scores.parquet``, ``discovery.csv``,
        ``discovery_a.csv`` and ``discovery_b.csv``.
    """
    parts = [pd.read_parquet(os.path.join(run, d, "gene_scores.parquet"))
             for d in sorted(os.listdir(run)) if d.startswith("fold_")]
    G = pd.concat(parts)
    G = G[G["n_cells"] >= min_cells]
    classes = [c for c in G.columns if c != "n_cells"]
    with open(os.path.join(run, "config.json"), encoding="utf-8") as handle:
        cfg = json.load(handle)
    TB = cfg["tables"]
    cm = _read(os.path.join(TB, "class_members.csv"))
    ct = _read(os.path.join(TB, "class_table.csv")).set_index("class_id")
    out = os.path.join(run, "eval")
    os.makedirs(out, exist_ok=True)
    G.to_parquet(os.path.join(out, "gene_class_scores.parquet"))
    is_nt = G.index.str.startswith("NT_")
    uni = _read(os.path.join(TB, "gene_universe.csv"))
    funk_pos = set(uni.loc[uni["phenotype_funk"].astype(bool), "gene"])
    rng = np.random.default_rng(seed)
    decoy_sets = {}
    if os.path.isfile(os.path.join(run, "decoys.csv")):
        dd = _read(os.path.join(run, "decoys.csv"))
        decoy_sets = {c: set(g) for c, g in dd.groupby("class_id")["gene"]}
    P, cset, g2c = _class_rows(G, classes, cm, ct, decoy_sets, funk_pos, boot, rng)
    _write(P, os.path.join(out, "per_class.csv"))
    D, dsize = _discovery(G, classes, P, cm, ct, cset, g2c, decoy_sets, funk_pos)
    _write(D, os.path.join(out, "discovery.csv"))
    for typ, name in (("a_member_new_phenotype", "discovery_a.csv"), ("b_new_function", "discovery_b.csv")):
        _write(D[D["type"] == typ], os.path.join(out, name))
    n_a = int((D["type"] == "a_member_new_phenotype").sum()) if len(D) else 0
    n_b = int((D["type"] == "b_new_function").sum()) if len(D) else 0
    real = P[P.source != "decoy"].dropna(subset=["auroc"]) if "auroc" in P else P.iloc[:0]
    dec = P[P.source == "decoy"].dropna(subset=["auroc"]) if "auroc" in P else P.iloc[:0]

    def med(col: str) -> Optional[float]:
        """The median of a column over the real classes, or None."""
        return float(real[col].median()) if col in real and real[col].notna().any() else None

    summary = {"tag": tag or os.path.basename(os.path.normpath(run)), "tables": str(TB),
               "genes_scored": int(len(G)), "nt_pseudogenes": int(is_nt.sum()),
               "classes_scored": int(len(real)), "median_auroc": med("auroc"), "median_ap": med("ap"),
               "median_auroc_vs_pheno": med("auroc_vs_pheno"),
               "median_auroc_vs_overlap": med("auroc_vs_overlap"),
               "median_auroc_unique_members": med("auroc_unique_members"),
               "classes_specific": int(((real.get("auroc_lo", 0) > 0.5) & (real.get("auroc_vs_pheno", 0) > 0.5)
                                        & (real.get("auroc_vs_overlap", 0) > 0.5)).sum()),
               "median_auroc_funkpos_members": med("auroc_funkpos"),
               "median_auroc_funkneg_members": med("auroc_funkneg"),
               "median_ap_funkpos_members": med("ap_funkpos"), "median_ap_funkneg_members": med("ap_funkneg"),
               "median_ap_over_prevalence": float((real.ap / real.prevalence).median()) if len(real) else None,
               "frac_classes_auroc_lo_gt_0.5": float((real.auroc_lo > 0.5).mean()) if len(real) else None,
               "by_source_median_auroc": real.groupby("source").auroc.median().to_dict() if len(real) else {},
               "by_source_median_auroc_funkneg": real.groupby("source")["auroc_funkneg"].median().to_dict()
               if "auroc_funkneg" in real else None,
               "decoy_median_auroc": float(dec.auroc.median()) if len(dec) else None,
               "decoy_frac_auroc_lo_gt_0.5": float((dec.auroc_lo > 0.5).mean()) if len(dec) else None,
               "discovery_null": "gene-wise (>= 100 matched decoys)" if dsize and len(dsize) >= 100
               else "pooled decoy z (< 100 decoys; liberal)",
               "discovery_a_member_new_phenotype_pairs": n_a,
               "discovery_a_genes": int(D.loc[D.type == "a_member_new_phenotype", "gene"].nunique()) if len(D) else 0,
               "discovery_b_new_function_pairs": n_b,
               "discovery_b_genes": int(D.loc[D.type == "b_new_function", "gene"].nunique()) if len(D) else 0}
    _dump(summary, os.path.join(out, "summary.json"))
    return summary


def run_gene_classes(settings: Mapping[str, Any],
                     report: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Gene classes from settings: build (or gate) the class tables, train the baseline, evaluate.

    :param settings: this module's settings (:func:`gene_class_defaults`).
        ``gene_class_sources`` (a JSON file of :data:`SOURCE_KEYS`) builds
        the class tables into ``train_gene_classes`` first;
        ``gene_class_gated_from`` gates an existing table instead.
    :param report: called with progress lines.
    :returns: ``{"tables", "run", "classes", "summary"}``.
    :raises ValueError: for an unsupported label source, split, model or
        group, or a missing ``train_gene_classes`` or ``train_features``.
    """
    s = gene_class_defaults(settings)
    say = report or (lambda text: None)
    if s["train_label_source"] != "gene_classes":
        raise ValueError(f"train_label_source must be 'gene_classes'; got {s['train_label_source']!r}")
    if s["train_split"] != "grouped_multilabel" or s["train_split_group"] != "gene":
        raise ValueError("train_split must be 'grouped_multilabel' with train_split_group 'gene'")
    if s["ml_model"] == "image":
        from .gene_class_images import run_gene_class_images

        if s["train_image_dataset"] == "bbbc021":
            images = run_gene_class_images(s, log=say)
            return {"tables": None, "run": images["config"]["out"], "classes": None, "summary": images["summary"]}
    elif s["ml_model"] != "logistic_regression":
        raise ValueError(f"ml_model must be 'logistic_regression' or 'image'; got {s['ml_model']!r}")
    tables = str(s["train_gene_classes"] or "")
    if not tables:
        raise ValueError("train_gene_classes, the class-table folder, is empty")
    built = None
    if s["gene_class_gated_from"]:
        with open(str(s["gene_class_sources"]), encoding="utf-8") as handle:
            screen = json.load(handle)["screen"]
        built = gate_gene_classes(str(s["gene_class_gated_from"]), screen, tables,
                                  int(s["gene_class_min_train"]))
    elif s["gene_class_sources"]:
        with open(str(s["gene_class_sources"]), encoding="utf-8") as handle:
            built = build_gene_classes(json.load(handle), s, tables)
    if built is not None:
        say(f"gene classes: {built['kept']} kept in {tables}")
    dst = str(s["gene_class_dst"] or os.path.join(tables, "results", "gene_classes"))
    seed = int(s["train_seed"])
    if s["ml_model"] == "image":
        from .gene_class_images import run_gene_class_images

        run_gene_class_images(dict(s, gene_class_dst=dst), log=say)
        summary = evaluate_gene_class_run(dst, min_cells=int(s["gene_class_eval_min_cells"]),
                                          boot=int(s["gene_class_eval_boot"]), seed=seed)
        say(f"gene classes: median AUROC {summary['median_auroc']} over {summary['classes_scored']} classes")
        return {"tables": tables, "run": dst, "classes": built, "summary": summary}
    if not s["train_features"]:
        raise ValueError("train_features, the gene feature table, is empty")
    exclude = [x.strip() for x in str(s["train_exclude_sources"]).split(",") if x.strip()]
    train_gene_class_baseline(_feature_table(s["train_features"], str(s["train_feature_suffix"])), tables, dst,
                              decoys=int(s["train_decoy_heads"]), shuffle=bool(s["train_shuffle_control"]),
                              exclude_sources=exclude, seed=seed, report=say)
    summary = evaluate_gene_class_run(dst, min_cells=int(s["gene_class_eval_min_cells"]),
                                      boot=int(s["gene_class_eval_boot"]), seed=seed)
    say(f"gene classes: median AUROC {summary['median_auroc']} over {summary['classes_scored']} classes")
    return {"tables": tables, "run": dst, "classes": built, "summary": summary}
