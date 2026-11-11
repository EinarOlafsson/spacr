"""The ``ops_profiles`` source of Image UMAP: OPS gene maps (item 918).

:func:`spacr.core.generate_image_umap` draws per-object measurements and
clusters its 2-D embedding. For OPS gene and guide profiles that is wrong:
the clusters every benchmark scores are made in the full profile space, and
a gene is not an image. With ``umap_source='ops_profiles'`` the same entry
calls :func:`run_ops_image_umap`, which

* lays the gene rows (and optionally the guide rows) out with umap-learn, for
  display only (:func:`umap_layout`);
* clusters the full profiles: a cosine kNN or PHATE graph and Leiden through
  :mod:`spacr.ops_clusters`, or a partition read from a file
  (:func:`profile_clusters`);
* builds a hierarchy of the clusters by linkage of their centroid profiles
  (:func:`cluster_hierarchy`);
* labels clusters by enrichment against versioned gene sets, BH across all
  cluster x term pairs, with a shuffled-label control
  (:func:`enrich_clusters`);
* picks, per cluster or top-level group, the medoid gene and that gene's
  cells closest to its profile (:func:`medoid_cells`) and shows their raw
  crops at ONE display range per channel shared by every crop
  (:func:`shared_display_range`), on a ring around the map whose slots are
  assigned by angle so leader lines do not cross (:func:`layout_crop_ring`);
* records the library versions that produced the layout
  (``umap_provenance.json``), because UMAP coordinates change across
  umap-learn / pynndescent / numba versions even with the same seed, while
  clusters, hierarchy, labels and medoids do not depend on the layout.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: The settings of the ``ops_profiles`` source and their defaults.
OPS_UMAP_DEFAULTS: Dict[str, Any] = {
    "umap_source": "measurements",
    "ops_umap_profiles": None,
    "ops_umap_guide_profiles": None,
    "ops_umap_guides": "none",
    "ops_umap_graph": "knn",
    "ops_umap_knn": 12,
    "ops_umap_resolution": 12,
    "ops_umap_leiden_seed": 42,
    "ops_umap_partition": None,
    "ops_umap_hierarchy_method": "average",
    "ops_umap_hierarchy_levels": [8, 30],
    "ops_umap_label_sources": ["CORUM@REPORT", "GO_BP"],
    "ops_umap_reference_dir": None,
    "ops_umap_enrichment_test": "hypergeom",
    "ops_umap_label_q": 0.05,
    "ops_umap_cell_embedding": None,
    "ops_umap_crop_level": "top",
    "ops_umap_cells_per_cluster": 3,
    "ops_umap_crop_px": 96,
    "ops_umap_crop_channels": None,
    "ops_umap_display_percentiles": [0.5, 99.5],
    "ops_umap_display_range": None,
    "ops_umap_min_cell_px": 200,
    "ops_umap_stability_seeds": [42, 1, 2, 3, 4],
    "ops_umap_shuffle_seed": 7,
    "ops_umap_dst": None,
}

#: UMAP's reference values for this source, used where the caller set none.
OPS_UMAP_LAYOUT_DEFAULTS: Dict[str, Any] = {"n_neighbors": 15, "min_dist": 0.3,
                                            "metric": "cosine", "random_seed": 42}

#: Packages whose versions ``umap_provenance.json`` records.
PROVENANCE_PACKAGES = ("umap", "pynndescent", "numba", "sklearn", "scipy", "numpy",
                       "leidenalg", "igraph")

#: The note every layout carries.
LAYOUT_NOTE = ("UMAP layout, for display; clusters are computed on the full profiles. "
               "UMAP layouts differ across umap-learn / pynndescent / numba versions even "
               "with the same seed; clusters, hierarchy, labels and medoids do not depend "
               "on the layout.")

#: Columns of a cell table that are addresses, not embedding dimensions.
CELL_ADDRESS_COLUMNS = ("cell_id", "gene", "guide", "well", "site", "field", "plate",
                        "centroid_y", "centroid_x", "png_path", "crop_path", "field_path",
                        "cell_px", "area", "edge_cut")

#: Prefixes of non-targeting rows.
CONTROL_PREFIXES = ("nontargeting", "ntpseudo")


def ops_umap_settings(settings: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The ``ops_profiles`` settings with defaults filled in.

    :param settings: the caller's settings.
    :returns: a new dict; UMAP keys take :data:`OPS_UMAP_LAYOUT_DEFAULTS`
        where the caller set none.
    """
    out = dict(OPS_UMAP_DEFAULTS)
    out.update(OPS_UMAP_LAYOUT_DEFAULTS)
    out.update({k: v for k, v in dict(settings or {}).items() if v is not None})
    return out


def is_control(names: Sequence[Any]) -> np.ndarray:
    """True for non-targeting rows.

    :param names: row names.
    :returns: a boolean array.
    """
    return np.array([str(n).lower().startswith(CONTROL_PREFIXES) for n in names], bool)


def read_profiles(path: str, kind: str = "gene") -> Tuple[np.ndarray, np.ndarray]:
    """Profile rows from an npz (``names``, ``pcs``) or a table.

    :param path: ``.npz``, or a table :func:`spacr.tabular.read_table` reads
        with a ``gene`` (or ``guide``) column and numeric feature columns.
    :param kind: the name column of a table, ``'gene'`` or ``'guide'``.
    :returns: ``(names, rows)`` with rows as float64.
    :raises ValueError: when a table has no ``kind`` column.
    """
    if str(path).endswith(".npz"):
        with np.load(path, allow_pickle=False) as data:
            return np.asarray(data["names"]).astype(str), np.asarray(data["pcs"], np.float64)
    from .tabular import read_table

    table = read_table(path, canonicalise=False, report=None)
    if kind not in table:
        raise ValueError(f"{path} has no {kind!r} column")
    features = [c for c in table.columns if c != kind and pd.api.types.is_numeric_dtype(table[c])]
    return table[kind].astype(str).to_numpy(), table[features].to_numpy(np.float64)


def library_versions() -> Dict[str, Optional[str]]:
    """The installed versions of :data:`PROVENANCE_PACKAGES`.

    :returns: ``{package: version}``, ``None`` for one not installed.
    """
    import importlib

    out: Dict[str, Optional[str]] = {}
    for name in PROVENANCE_PACKAGES:
        try:
            out[name] = str(getattr(importlib.import_module(name), "__version__", "unknown"))
        except ImportError:
            out[name] = None
    return out


def umap_layout(rows: np.ndarray, *, n_neighbors: int = 15, min_dist: float = 0.3,
                metric: str = "cosine", random_state: int = 42,
                extra: Optional[np.ndarray] = None, mode: str = "none"
                ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """The display layout of profile rows with umap-learn.

    :param rows: gene rows.
    :param n_neighbors: UMAP neighbours.
    :param min_dist: UMAP minimum distance.
    :param metric: UMAP metric.
    :param random_state: UMAP seed; with a seed umap-learn runs on one thread.
    :param extra: guide rows to place too.
    :param mode: ``'joint'`` fits genes and guides together, ``'project'``
        fits the genes and transforms the guides, ``'none'`` ignores ``extra``.
    :returns: ``(gene_xy, guide_xy)``, float32; ``guide_xy`` is ``None``
        without guides.
    :raises ValueError: for an unknown mode.
    """
    import umap

    if mode not in ("none", "joint", "project"):
        raise ValueError(f"ops_umap_guides must be 'none', 'joint' or 'project'; got {mode!r}")
    model = umap.UMAP(n_neighbors=int(n_neighbors), min_dist=float(min_dist), metric=metric,
                      random_state=int(random_state), n_jobs=1)
    if extra is None or mode == "none":
        return model.fit_transform(rows).astype(np.float32), None
    if mode == "joint":
        both = model.fit_transform(np.vstack([rows, extra])).astype(np.float32)
        return both[:len(rows)], both[len(rows):]
    genes = model.fit_transform(rows).astype(np.float32)
    return genes, model.transform(extra).astype(np.float32)


def profile_clusters(names: Sequence[str], rows: np.ndarray, settings: Mapping[str, Any]
                     ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Clusters of the full profiles, never of the layout.

    :param names: gene names, one per row.
    :param rows: the full profile rows.
    :param settings: ``ops_umap_partition`` (``'file:column'``, with a
        ``gene`` column) wins; else ``ops_umap_graph`` (``'knn'`` with
        ``ops_umap_knn`` neighbours, or ``'phate'``) through
        :func:`spacr.ops_clusters.gene_graph`, then Leiden at
        ``ops_umap_resolution`` and ``ops_umap_leiden_seed`` through
        :func:`spacr.ops_clusters.leiden_partitions`.
    :returns: ``(labels, record)``; ``record`` says where the partition came
        from.
    :raises ValueError: when the partition file lacks genes that are mapped.
    """
    given = settings.get("ops_umap_partition")
    if given:
        from .tabular import read_table

        path, _, column = str(given).rpartition(":")
        table = read_table(path, canonicalise=False, report=None)
        lookup = dict(zip(table["gene"].astype(str), table[column]))
        missing = [n for n in names if n not in lookup]
        if missing:
            raise ValueError(f"{len(missing)} mapped genes have no {column!r} in {path}: "
                             f"{missing[:5]}")
        return (np.array([int(lookup[n]) for n in names]),
                {"source": "file", "file": path, "column": column})
    from .ops_clusters import gene_graph, leiden_partitions

    seed = int(settings.get("ops_umap_leiden_seed", 42))
    resolution = float(settings.get("ops_umap_resolution", 12))
    graph = gene_graph(rows, kind=str(settings.get("ops_umap_graph") or "knn"),
                       k=int(settings.get("ops_umap_knn") or 12), seed=seed)
    labels = leiden_partitions(graph, [resolution], [seed])[(seed, resolution)]
    return np.asarray(labels, int), {"source": "leiden", "graph": settings.get("ops_umap_graph"),
                                     "k": settings.get("ops_umap_knn"), "seed": seed,
                                     "resolution": resolution}


def cluster_centroids(rows: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The mean profile of every cluster.

    :param rows: profile rows.
    :param labels: one cluster per row.
    :returns: ``(clusters, centroids)``, clusters sorted.
    """
    clusters = np.unique(labels)
    return clusters, np.vstack([rows[labels == c].mean(axis=0) for c in clusters])


def cluster_hierarchy(rows: np.ndarray, labels: np.ndarray, *, method: str = "average",
                      levels: Sequence[int] = (8, 30)) -> Tuple[pd.DataFrame, np.ndarray]:
    """Linkage of cluster centroid profiles (cosine), cut at group counts.

    :param rows: profile rows (controls excluded by the caller if wanted).
    :param labels: one cluster per row.
    :param method: :func:`scipy.cluster.hierarchy.linkage` method.
    :param levels: numbers of top-level groups to cut at.
    :returns: ``(table, linkage)``; the table has ``cluster``, ``level``
        (the group count asked for), ``parent`` (the group at that level),
        ``height`` (the linkage height at which the cluster first merges)
        and ``n_genes``.
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    clusters, centroids = cluster_centroids(rows, labels)
    sizes = {c: int((labels == c).sum()) for c in clusters}
    if len(clusters) < 2:
        frame = pd.DataFrame([{"cluster": int(c), "level": int(lv), "parent": 1, "height": 0.0,
                               "n_genes": sizes[c]} for lv in levels for c in clusters])
        return frame, np.zeros((0, 4))
    z = linkage(centroids, method=method, metric="cosine")
    first = np.zeros(len(clusters))
    for step in range(len(z) - 1, -1, -1):
        for side in (0, 1):
            leaf = int(z[step, side])
            if leaf < len(clusters):
                first[leaf] = z[step, 2]
    out = []
    for level in levels:
        parents = fcluster(z, t=int(level), criterion="maxclust")
        out += [{"cluster": int(c), "level": int(level), "parent": int(p),
                 "height": float(h), "n_genes": sizes[c]}
                for c, p, h in zip(clusters, parents, first)]
    return pd.DataFrame(out), z


def label_sets(sources: Sequence[str], reference_dir: Any) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """The gene sets clusters are labelled with, and their versions.

    :param sources: ``'CORUM'`` / ``'KEGG'`` (every group), ``'CORUM@REPORT'``
        or ``'@SELECT'`` (that half of the groups, as
        :func:`spacr.ops_reference.group_halves` splits them), ``'NAME'``
        (``NAME.tsv`` in a reference folder) or ``'NAME=path'``; a file has
        ``gene_name`` and ``group`` columns.
    :param reference_dir: the reference folder(s).
    :returns: ``(sets, record)``; ``sets`` has ``source``, ``term``,
        ``gene``; ``record`` maps each source to its file and md5, or to
        ``'missing'`` when no file was found (the source is then skipped and
        the legend says so).
    """
    from .ops_reference import (REFERENCE_FILES, file_md5, group_halves, reference_dirs,
                                reference_file, reference_groups)
    from .tabular import read_table

    frames, record = [], {}
    for source in sources:
        name, _, path = str(source).partition("=")
        base, _, half = name.partition("@")
        try:
            if path:
                pass
            elif base in REFERENCE_FILES:
                path = reference_file(base, reference_dir)
            else:
                candidates = [os.path.join(d, f"{base}.tsv") for d in reference_dirs(reference_dir)]
                path = next((c for c in candidates if os.path.isfile(c)), "")
                if not path:
                    raise FileNotFoundError(base)
        except (FileNotFoundError, ValueError):
            record[name] = "missing"
            continue
        if base in ("CORUM", "KEGG") and not str(source).count("="):
            groups = reference_groups(base, reference_dir)
        else:
            table = read_table(path, canonicalise=False, report=None,
                               **({"sep": "\t"} if path.endswith(".tsv") else {}))
            table = table.dropna(subset=["gene_name", "group"])
            groups = {g: sorted(set(d.gene_name.astype(str))) for g, d in table.groupby("group")}
        if half:
            halves = group_halves(groups)
            groups = {g: m for g, m in groups.items() if halves[g] == {"SELECT": 0, "REPORT": 1}[half]}
        record[name] = {"file": path, "md5": file_md5(path), "terms": len(groups)}
        frames.append(pd.DataFrame([(name, g, m) for g, ms in groups.items() for m in ms],
                                   columns=["source", "term", "gene"]))
    sets = (pd.concat(frames, ignore_index=True) if frames
            else pd.DataFrame(columns=["source", "term", "gene"]))
    return sets, record


def _bh(p: np.ndarray, m: Optional[int] = None) -> np.ndarray:
    """Benjamini-Hochberg q-values over ``m`` tests (the untested ones count as p = 1)."""
    p = np.asarray(p, float)
    if not p.size:
        return p
    m = max(int(m or p.size), p.size)
    order = np.argsort(p, kind="stable")
    ranked = p[order] * m / np.arange(1, p.size + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def enrich_clusters(genes: Sequence[str], labels: np.ndarray, sets: pd.DataFrame, *,
                    test: str = "hypergeom", min_overlap: int = 2, q: float = 0.05
                    ) -> Tuple[pd.DataFrame, Dict[int, Tuple[str, float]]]:
    """Enrichment of every cluster for every gene set, and each cluster's label.

    The background is every clustered gene except non-targeting rows; set
    members outside it are dropped. Pairs with fewer than ``min_overlap``
    shared genes are not tested and enter the correction as p = 1: q-values
    are Benjamini-Hochberg across ALL cluster x term pairs (every cluster
    times every set with a member in the background), so filtering on
    overlap does not shrink the correction.

    :param genes: clustered gene names.
    :param labels: one cluster per gene.
    :param sets: ``source``, ``term``, ``gene`` rows.
    :param test: ``'hypergeom'`` (one-sided, over-representation) or
        ``'fisher_two_sided'`` (Brieflow's benchmark test).
    :param min_overlap: the fewest shared genes tested.
    :param q: the q-value a label needs.
    :returns: ``(table, labels)``; the table has ``cluster``, ``source``,
        ``term``, ``overlap``, ``size``, ``cluster_size``, ``p``, ``q``,
        ``label`` (True for the term drawn); ``labels`` maps a cluster to
        ``(term, q)``, or ``('unlabelled', nan)``.
    :raises ValueError: for an unknown test.
    """
    from scipy.stats import fisher_exact, hypergeom

    if test not in ("hypergeom", "fisher_two_sided"):
        raise ValueError(f"ops_umap_enrichment_test must be 'hypergeom' or 'fisher_two_sided'; "
                         f"got {test!r}")
    genes = np.asarray(genes).astype(str)
    keep = ~is_control(genes)
    genes, labels = genes[keep], np.asarray(labels)[keep]
    background = set(genes)
    n_bg = len(background)
    members = {}
    for (source, term), d in sets.groupby(["source", "term"], sort=True):
        inside = set(d.gene.astype(str)) & background
        if inside:
            members[(source, term)] = inside
    rows = []
    for cluster in np.unique(labels):
        in_cluster = set(genes[labels == cluster])
        n = len(in_cluster)
        for (source, term), inside in members.items():
            a = len(in_cluster & inside)
            if a < min_overlap:
                continue
            k = len(inside)
            if test == "hypergeom":
                p = float(hypergeom.sf(a - 1, n_bg, k, n))
            else:
                p = float(fisher_exact([[a, k - a], [n - a, n_bg - k - n + a]])[1])
            rows.append({"cluster": int(cluster), "source": source, "term": term, "overlap": a,
                         "size": k, "cluster_size": n, "p": p})
    table = pd.DataFrame(rows, columns=["cluster", "source", "term", "overlap", "size",
                                        "cluster_size", "p"])
    table["q"] = _bh(table["p"].to_numpy(), len(np.unique(labels)) * len(members))
    table = table.sort_values(["cluster", "q", "p", "source", "term"], kind="stable")
    table["label"] = False
    chosen: Dict[int, Tuple[str, float]] = {int(c): ("unlabelled", float("nan"))
                                            for c in np.unique(labels)}
    best = table[table.q < q].groupby("cluster", sort=True).head(1)
    table.loc[best.index, "label"] = True
    for row in best.itertuples():
        chosen[int(row.cluster)] = (str(row.term), float(row.q))
    return table.reset_index(drop=True), chosen


def shuffled_label_control(genes: Sequence[str], labels: np.ndarray, sets: pd.DataFrame, *,
                           seed: int = 7, **kwargs: Any) -> int:
    """How many clusters get a label when the gene labels are shuffled.

    :param genes: clustered gene names.
    :param labels: one cluster per gene; the permutation keeps cluster sizes.
    :param sets: as for :func:`enrich_clusters`.
    :param seed: the permutation seed.
    :param kwargs: passed to :func:`enrich_clusters`.
    :returns: the number of labelled clusters under the shuffle.
    """
    genes = np.asarray(genes).astype(str)
    targeting = ~is_control(genes)
    shuffled = genes.copy()
    shuffled[targeting] = np.random.default_rng(seed).permutation(genes[targeting])
    _, chosen = enrich_clusters(shuffled, labels, sets, **kwargs)
    return sum(1 for term, _ in chosen.values() if term != "unlabelled")


def _cosine_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine distances between the rows of ``a`` and the rows of ``b``."""
    na = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    nb = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return 1.0 - na @ nb.T


def medoid_gene(names: Sequence[str], rows: np.ndarray) -> str:
    """The gene with the smallest summed cosine distance to the others.

    :param names: the group's gene names.
    :param rows: their profiles.
    :returns: the medoid; ties go to the alphabetically first name.
    """
    names = np.asarray(names).astype(str)
    total = _cosine_distances(rows, rows).sum(axis=1)
    order = np.lexsort((names, np.round(total, 12)))
    return str(names[order[0]])


def read_cell_table(path: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """The per-cell table: addresses and the cells' profile-space rows.

    :param path: a table with ``cell_id``, ``gene``, crop address columns
        (:data:`CELL_ADDRESS_COLUMNS`) and numeric embedding columns.
    :returns: ``(addresses, rows)``.
    :raises ValueError: without ``cell_id`` or ``gene``.
    """
    from .tabular import read_table

    table = read_table(path, canonicalise=False, report=None)
    for column in ("cell_id", "gene"):
        if column not in table:
            raise ValueError(f"{path} has no {column!r} column")
    dims = [c for c in table.columns
            if c not in CELL_ADDRESS_COLUMNS and pd.api.types.is_numeric_dtype(table[c])]
    return table[[c for c in table.columns if c not in dims]].copy(), table[dims].to_numpy(np.float64)


def medoid_cells(groups: Mapping[Any, Sequence[str]], names: Sequence[str], rows: np.ndarray,
                 cells: pd.DataFrame, cell_rows: np.ndarray, *, n: int = 3,
                 min_cell_px: int = 0, usable: Optional[Callable[[pd.Series], bool]] = None
                 ) -> pd.DataFrame:
    """Each group's medoid gene and that gene's cells closest to its profile.

    :param groups: group id to member gene names.
    :param names: profile row names.
    :param rows: profile rows (the space the cell rows live in).
    :param cells: the per-cell addresses, with ``cell_id`` and ``gene``.
    :param cell_rows: the cells' rows, same dimensions as ``rows``.
    :param n: cells per group.
    :param min_cell_px: skip cells whose ``cell_px`` (or ``area``) is smaller.
    :param usable: a further test per cell, such as "its crop lies inside
        the field"; cells failing it are skipped.
    :returns: one row per group: ``group``, ``medoid_gene``, ``cell_ids``
        (``;``-joined), ``distances``.
    :raises ValueError: when the cell rows' dimensions differ from the
        profile rows'.
    """
    rows = np.asarray(rows, float)
    if cell_rows.shape[1] != rows.shape[1]:
        raise ValueError(f"the cell rows have {cell_rows.shape[1]} dimensions, the profiles "
                         f"{rows.shape[1]}")
    index = {name: i for i, name in enumerate(np.asarray(names).astype(str))}
    size = cells["cell_px"] if "cell_px" in cells else cells.get("area")
    out = []
    for group in sorted(groups):
        members = [g for g in groups[group] if g in index]
        if not members:
            continue
        gene = medoid_gene(members, rows[[index[g] for g in members]])
        mine = np.flatnonzero(cells["gene"].astype(str).to_numpy() == gene)
        if size is not None:
            mine = mine[np.asarray(size)[mine] >= min_cell_px]
        if "edge_cut" in cells:
            mine = mine[~cells["edge_cut"].to_numpy(bool)[mine]]
        distance = _cosine_distances(cell_rows[mine], rows[[index[gene]]])[:, 0]
        ids = cells["cell_id"].astype(str).to_numpy()[mine]
        order = np.lexsort((ids, np.round(distance, 12)))
        picked, dists = [], []
        for k in order:
            if usable is not None and not usable(cells.iloc[mine[k]]):
                continue
            picked.append(ids[k])
            dists.append(float(distance[k]))
            if len(picked) == n:
                break
        out.append({"group": group, "medoid_gene": gene, "cell_ids": ";".join(picked),
                    "distances": ";".join(f"{d:.6g}" for d in dists)})
    return pd.DataFrame(out, columns=["group", "medoid_gene", "cell_ids", "distances"])


def read_crop(cell: Mapping[str, Any], crop_px: int) -> Optional[np.ndarray]:
    """One raw ``(crop_px, crop_px, C)`` crop, or None when it is cut off.

    :param cell: a cell row with ``field_path`` (an ``(H, W, C)`` ``.npy``)
        and ``centroid_y`` / ``centroid_x``, or ``crop_path`` / ``png_path``
        (a crop centred on the cell, cut to ``crop_px`` around its centre).
    :param crop_px: the crop side.
    :returns: the stored intensities, never stretched; ``None`` when the
        window leaves the field or the crop is smaller than ``crop_px``.
    """
    half = crop_px // 2
    field_path = cell.get("field_path")
    if isinstance(field_path, str) and field_path:
        field = np.load(field_path, mmap_mode="r")
        y, x = int(round(float(cell["centroid_y"]))), int(round(float(cell["centroid_x"])))
        y0, x0 = y - half, x - half
        if y0 < 0 or x0 < 0 or y0 + crop_px > field.shape[0] or x0 + crop_px > field.shape[1]:
            return None
        crop = np.asarray(field[y0:y0 + crop_px, x0:x0 + crop_px])
    else:
        path = cell.get("crop_path") or cell.get("png_path")
        if str(path).endswith(".npy"):
            crop = np.load(path)
        else:
            from skimage.io import imread

            crop = imread(path)
        if crop.shape[0] < crop_px or crop.shape[1] < crop_px:
            return None
        y0, x0 = (crop.shape[0] - crop_px) // 2, (crop.shape[1] - crop_px) // 2
        crop = crop[y0:y0 + crop_px, x0:x0 + crop_px]
    return crop[..., None] if crop.ndim == 2 else crop


def shared_display_range(crops: Sequence[np.ndarray], channels: Sequence[int], *,
                         percentiles: Sequence[float] = (0.5, 99.5),
                         fixed: Optional[Mapping[Any, Sequence[float]]] = None
                         ) -> Dict[int, Tuple[float, float]]:
    """One display range per channel, shared by every crop shown.

    :param crops: the raw crops of the figure.
    :param channels: channel indices shown.
    :param percentiles: low and high percentiles of the pooled pixels of all
        crops.
    :param fixed: channel to ``(low, high)``; wins over the percentiles.
    :returns: ``{channel: (low, high)}``.
    """
    out = {}
    for c in channels:
        given = (fixed or {}).get(c, (fixed or {}).get(str(c)))
        if given is not None:
            out[int(c)] = (float(given[0]), float(given[1]))
            continue
        pooled = (np.concatenate([np.asarray(k)[..., c].ravel() for k in crops])
                  if crops else np.zeros(1))
        lo, hi = np.percentile(pooled.astype(np.float64), [percentiles[0], percentiles[1]])
        out[int(c)] = (float(lo), float(hi))
    return out


def composite(crop: np.ndarray, colours: Mapping[int, str],
              ranges: Mapping[int, Tuple[float, float]]) -> np.ndarray:
    """An RGB view of a raw crop at the shared ranges.

    :param crop: ``(H, W, C)`` raw crop.
    :param colours: channel to a matplotlib colour.
    :param ranges: channel to ``(low, high)``.
    :returns: ``(H, W, 3)`` float in ``[0, 1]``.
    """
    from matplotlib.colors import to_rgb

    rgb = np.zeros(crop.shape[:2] + (3,))
    for c, colour in colours.items():
        lo, hi = ranges[c]
        scaled = np.clip((crop[..., c].astype(np.float64) - lo) / max(hi - lo, 1e-12), 0, 1)
        rgb += scaled[..., None] * np.asarray(to_rgb(colour))
    return np.clip(rgb, 0, 1)


def layout_crop_ring(points: Mapping[Any, Tuple[float, float]], centre: Tuple[float, float],
                     n_slots: Optional[int] = None) -> Dict[Any, int]:
    """Assign each group to a slot on a ring around the map by angle.

    Slots are evenly spaced, slot 0 at angle 0 (east), counter-clockwise.
    The assignment minimises the summed angular distance between each
    group's medoid point (seen from ``centre``) and its slot (Hungarian
    assignment), so leader lines keep the groups' angular order and do not
    cross.

    :param points: group to its medoid point ``(x, y)`` on the map.
    :param centre: the map centre ``(x, y)``.
    :param n_slots: slots on the ring; at least one per group.
    :returns: ``{group: slot}``.
    """
    from scipy.optimize import linear_sum_assignment

    keys = sorted(points, key=str)
    if not keys:
        return {}
    n_slots = max(int(n_slots or 0), len(keys))
    angles = np.array([np.arctan2(points[k][1] - centre[1], points[k][0] - centre[0])
                       for k in keys])
    slots = 2 * np.pi * np.arange(n_slots) / n_slots
    gap = np.abs(angles[:, None] - slots[None, :]) % (2 * np.pi)
    cost = np.minimum(gap, 2 * np.pi - gap)
    rows, cols = linear_sum_assignment(cost)
    return {keys[r]: int(c) for r, c in zip(rows, cols)}


def layout_stability(rows: np.ndarray, labels: np.ndarray, settings: Mapping[str, Any],
                     seeds: Sequence[int], k: int = 15) -> pd.DataFrame:
    """How faithful the layout is, over several UMAP seeds.

    :param rows: gene rows.
    :param labels: their clusters.
    :param settings: the UMAP keys.
    :param seeds: UMAP seeds.
    :param k: the neighbours compared.
    :returns: one row per seed: ``trustworthiness`` (cosine, ``k``) and the
        share of each gene's ``k`` nearest neighbours that are in its own
        cluster in 2-D (``same_cluster_2d``) and in the profile space
        (``same_cluster_profile``).
    """
    from sklearn.manifold import trustworthiness
    from sklearn.neighbors import NearestNeighbors

    k = min(int(k), len(rows) - 1)
    profile = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(rows)
    near_p = profile.kneighbors(rows, return_distance=False)[:, 1:]
    out = []
    for seed in seeds:
        xy, _ = umap_layout(rows, n_neighbors=settings["n_neighbors"], min_dist=settings["min_dist"],
                            metric=settings["metric"], random_state=int(seed))
        near_2d = NearestNeighbors(n_neighbors=k + 1).fit(xy).kneighbors(
            xy, return_distance=False)[:, 1:]
        out.append({"seed": int(seed),
                    "trustworthiness": float(trustworthiness(rows, xy, n_neighbors=k,
                                                             metric="cosine")),
                    "same_cluster_2d": float((labels[near_2d] == labels[:, None]).mean()),
                    "same_cluster_profile": float((labels[near_p] == labels[:, None]).mean())})
    return pd.DataFrame(out)


def _crop_channels(value: Any) -> Dict[int, str]:
    """``ops_umap_crop_channels`` as ``{channel index: colour}``."""
    if isinstance(value, Mapping):
        return {int(k): str(v) for k, v in value.items()}
    return {int(k): str(v) for k, v in value}


def _figure(coords: pd.DataFrame, hierarchy: pd.DataFrame, z: np.ndarray, top_level: int,
            panels: Mapping[Any, Dict[str, Any]], slots: Mapping[Any, int], n_slots: int,
            legend: str, dark: bool, per_cluster: bool):
    """Draw the map, hulls, crop ring with leader lines, and dendrogram."""
    import textwrap

    from matplotlib.figure import Figure
    from matplotlib.patches import ConnectionPatch, Polygon
    from scipy.cluster.hierarchy import dendrogram
    from scipy.spatial import ConvexHull

    from .utils import generate_colors

    fg, bg = ("#e6edf3", "#0d1b2a") if dark else ("#0d1b2a", "#ffffff")
    fig = Figure(figsize=(17, 11), facecolor=bg)
    ax = fig.add_axes([0.2, 0.24, 0.42, 0.5], facecolor=bg)
    genes = coords[coords.kind == "gene"]
    clusters = sorted(genes.loc[~genes.control, "cluster"].unique())
    palette = generate_colors(len(clusters), dark)
    colour = {c: tuple(palette[i]) for i, c in enumerate(clusters)}
    ctrl = genes[genes.control]
    ax.scatter(ctrl.x, ctrl.y, s=6, c="#8a96a3", alpha=0.6, linewidths=0)
    for c in clusters:
        d = genes[(genes.cluster == c) & ~genes.control]
        ax.scatter(d.x, d.y, s=8, color=colour[c], alpha=0.8, linewidths=0)
    guides = coords[coords.kind == "guide"]
    if len(guides):
        ax.scatter(guides.x, guides.y, s=3, c=fg, alpha=0.25, marker="x", linewidths=0.5)
    parent = hierarchy[hierarchy.level == top_level].set_index("cluster")["parent"]
    for group in sorted(parent.unique()):
        members = parent.index[parent == group]
        pts = genes.loc[genes.cluster.isin(members) & ~genes.control, ["x", "y"]].to_numpy()
        if len(pts) >= 3 and np.ptp(pts, axis=0).min() > 0:
            hull = ConvexHull(pts)
            ax.add_patch(Polygon(pts[hull.vertices], closed=True, fill=False, lw=0.6,
                                 ec=fg, alpha=0.35))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(fg)
    ax.set_title("UMAP layout, for display; clusters are computed on the full profiles",
                 color=fg, fontsize=9)
    for group, panel in panels.items():
        angle = 2 * np.pi * slots[group] / n_slots
        cx, cy = 0.41 + 0.3 * np.cos(angle), 0.49 + 0.4 * np.sin(angle)
        title = "\n".join(textwrap.wrap(panel["title"], 30)[:3])
        line = colour.get(group, fg) if per_cluster else fg
        side = np.cos(angle)
        align = "left" if side > 1e-6 else "right" if side < -1e-6 else "center"
        if panel["images"]:
            pax = fig.add_axes([cx - 0.07, cy - 0.035, 0.14, 0.07])
            pax.imshow(np.concatenate(panel["images"], axis=1))
            pax.set_xticks([])
            pax.set_yticks([])
            for spine in pax.spines.values():
                spine.set_color(line)
            pax.set_title(title, color=fg, fontsize=6)
            target, coords = (0.5, 0.5), pax.transAxes
        else:
            fig.text(cx, cy, title, color=fg, fontsize=6, ha=align, va="center")
            target, coords = (cx, cy), fig.transFigure
        ax.add_artist(ConnectionPatch(xyA=panel["point"], coordsA=ax.transData,
                                      xyB=target, coordsB=coords,
                                      color=line, lw=0.6, alpha=0.8))
    if len(z):
        dax = fig.add_axes([0.83, 0.30, 0.14, 0.4], facecolor=bg)
        names = [f"C{c}" for c in sorted(hierarchy.cluster.unique())]
        dendrogram(z, ax=dax, orientation="left", labels=names if len(names) <= 40 else None,
                   truncate_mode="lastp" if len(names) > 40 else None, p=40,
                   color_threshold=0, above_threshold_color=fg)
        dax.tick_params(colors=fg, labelsize=4)
        dax.set_title("cluster hierarchy (cosine, centroids)", color=fg, fontsize=7)
    fig.text(0.02, 0.02, legend, color=fg, fontsize=6, wrap=True, va="bottom")
    return fig


def run_ops_image_umap(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """The ``ops_profiles`` source of Image UMAP, end to end.

    Writes under ``ops_umap_dst`` (default: the profile file's folder plus
    ``ops_umap``): ``umap_coords.csv``, ``clusters.csv``,
    ``cluster_hierarchy.csv``, ``cluster_enrichment.csv``,
    ``enrichment_control.json``, ``layout_stability.csv``,
    ``cluster_medoids.csv``, ``crops/`` (raw ``.npy`` and PNG at the shared
    range), ``crops_display.json``, ``ops_umap.pdf/png/svg`` and
    ``umap_provenance.json``.

    :param settings: Image UMAP settings with ``umap_source='ops_profiles'``;
        ``ops_umap_profiles`` is required, ``ops_umap_crop_channels`` is
        required when ``ops_umap_cell_embedding`` is given.
    :returns: the output paths and summary counts.
    :raises ValueError: without ``ops_umap_profiles``, or with cells but no
        crop channels.
    """
    from .ops_reference import file_md5
    from .tabular import write_table

    s = ops_umap_settings(settings)
    if not s.get("ops_umap_profiles"):
        raise ValueError("umap_source='ops_profiles' needs ops_umap_profiles")
    if s.get("ops_umap_cell_embedding") and not s.get("ops_umap_crop_channels"):
        raise ValueError("ops_umap_crop_channels is required when ops_umap_cell_embedding is set")
    dst = s.get("ops_umap_dst") or os.path.join(
        os.path.dirname(os.path.abspath(s["ops_umap_profiles"])), "ops_umap")
    os.makedirs(os.path.join(dst, "crops"), exist_ok=True)
    names, rows = read_profiles(s["ops_umap_profiles"], "gene")
    guide_names, guide_rows = (read_profiles(s["ops_umap_guide_profiles"], "guide")
                               if s.get("ops_umap_guide_profiles") else (None, None))
    control = is_control(names)

    labels, cluster_record = profile_clusters(names, rows, s)
    xy, guide_xy = umap_layout(rows, n_neighbors=s["n_neighbors"], min_dist=s["min_dist"],
                               metric=s["metric"], random_state=int(s["random_seed"]),
                               extra=guide_rows, mode=str(s["ops_umap_guides"]))
    coords = pd.DataFrame({"name": names, "kind": "gene", "x": xy[:, 0], "y": xy[:, 1],
                           "cluster": labels, "control": control})
    if guide_xy is not None:
        coords = pd.concat([coords, pd.DataFrame({
            "name": guide_names, "kind": "guide", "x": guide_xy[:, 0], "y": guide_xy[:, 1],
            "cluster": -1, "control": is_control(guide_names)})], ignore_index=True)
    write_table(coords, os.path.join(dst, "umap_coords.csv"), canonicalise=False)
    write_table(pd.DataFrame({"gene": names, "cluster": labels, "control": control}),
                os.path.join(dst, "clusters.csv"), canonicalise=False)

    levels = [int(v) for v in s["ops_umap_hierarchy_levels"]]
    hierarchy, z = cluster_hierarchy(rows[~control], labels[~control],
                                     method=s["ops_umap_hierarchy_method"], levels=levels)
    write_table(hierarchy, os.path.join(dst, "cluster_hierarchy.csv"), canonicalise=False)

    sets, set_record = label_sets(s["ops_umap_label_sources"],
                                  s.get("ops_umap_reference_dir") or s.get("ops_reference_dir"))
    enrich_kwargs = {"test": s["ops_umap_enrichment_test"], "q": float(s["ops_umap_label_q"])}
    enrichment, chosen = enrich_clusters(names, labels, sets, **enrich_kwargs)
    write_table(enrichment, os.path.join(dst, "cluster_enrichment.csv"), canonicalise=False)
    real = sum(1 for term, _ in chosen.values() if term != "unlabelled")
    shuffled = shuffled_label_control(names, labels, sets, seed=int(s["ops_umap_shuffle_seed"]),
                                      **enrich_kwargs)
    control_record = {"labelled_clusters": real, "labelled_clusters_shuffled": shuffled,
                      "clusters": int(len(chosen)), "shuffle_seed": int(s["ops_umap_shuffle_seed"])}
    with open(os.path.join(dst, "enrichment_control.json"), "w", encoding="utf-8") as handle:
        json.dump(control_record, handle, indent=2, sort_keys=True)

    stability = layout_stability(rows, labels, s, [int(v) for v in s["ops_umap_stability_seeds"]])
    write_table(stability, os.path.join(dst, "layout_stability.csv"), canonicalise=False)

    top = min(levels)
    targeting = names[~control]
    if s["ops_umap_crop_level"] == "top":
        parent = hierarchy[hierarchy.level == top].set_index("cluster")["parent"].to_dict()
        groups: Dict[Any, List[str]] = {}
        for gene, c in zip(targeting, labels[~control]):
            groups.setdefault(int(parent[c]), []).append(str(gene))
        group_of = {int(c): int(p) for c, p in parent.items()}
    else:
        groups = {}
        for gene, c in zip(targeting, labels[~control]):
            groups.setdefault(int(c), []).append(str(gene))
        group_of = {int(c): int(c) for c in groups}
    point = {}
    for group, members in groups.items():
        idx = np.flatnonzero(np.isin(names, members))
        medoid = medoid_gene(names[idx], rows[idx])
        point[group] = tuple(xy[int(np.flatnonzero(names == medoid)[0])])

    panels: Dict[Any, Dict[str, Any]] = {}
    display: Dict[str, Any] = {}
    medoids = pd.DataFrame(columns=["group", "medoid_gene", "cell_ids", "distances"])
    if s.get("ops_umap_cell_embedding"):
        cells, cell_rows = read_cell_table(s["ops_umap_cell_embedding"])
        crop_px = int(s["ops_umap_crop_px"])
        medoids = medoid_cells(groups, names, rows, cells, cell_rows,
                               n=int(s["ops_umap_cells_per_cluster"]),
                               min_cell_px=int(s["ops_umap_min_cell_px"]),
                               usable=lambda cell: read_crop(cell, crop_px) is not None)
        by_id = cells.assign(cell_id=cells["cell_id"].astype(str)).set_index("cell_id")
        colours = _crop_channels(s["ops_umap_crop_channels"])
        raw: Dict[Any, List[Tuple[str, np.ndarray]]] = {}
        for row in medoids.itertuples():
            ids = [i for i in str(row.cell_ids).split(";") if i]
            raw[row.group] = [(i, read_crop(by_id.loc[i], crop_px)) for i in ids]
        pooled = [crop for items in raw.values() for _, crop in items]
        ranges = shared_display_range(pooled, sorted(colours),
                                      percentiles=s["ops_umap_display_percentiles"],
                                      fixed=s.get("ops_umap_display_range"))
        display = {"channels": {str(c): {"colour": colours[c], "low": ranges[c][0],
                                         "high": ranges[c][1]} for c in sorted(colours)},
                   "percentiles": list(s["ops_umap_display_percentiles"]),
                   "fixed": s.get("ops_umap_display_range"), "crops_pooled": len(pooled),
                   "crop_px": crop_px, "note": "raw crops, no per-crop stretch; one range per "
                                               "channel shared by every crop shown"}
        with open(os.path.join(dst, "crops_display.json"), "w", encoding="utf-8") as handle:
            json.dump(display, handle, indent=2, sort_keys=True)
        from matplotlib.image import imsave

        for group, items in raw.items():
            images = []
            for cell_id, crop in items:
                stem = os.path.join(dst, "crops", f"g{group}_{cell_id}")
                np.save(stem + ".npy", crop)
                view = composite(crop, colours, ranges)
                imsave(stem + ".png", view)
                images.append(view)
            panels[group] = {"images": images}
        medoids["cell_addresses"] = [
            ";".join(str(by_id.loc[i].get("field_path") or by_id.loc[i].get("crop_path")
                         or by_id.loc[i].get("png_path")) for i in str(ids).split(";") if i)
            for ids in medoids["cell_ids"]]
    else:
        panels = {g: {"images": []} for g in groups}
    write_table(medoids, os.path.join(dst, "cluster_medoids.csv"), canonicalise=False)

    for group in panels:
        members = sorted(c for c, g in group_of.items() if g == group)
        best = sorted(((chosen[c][1], chosen[c][0]) for c in members
                       if chosen[c][0] != "unlabelled"))
        name = f"{best[0][1]} (q={best[0][0]:.2g})" if best else "unlabelled"
        panels[group].update({"title": f"{'group' if s['ops_umap_crop_level'] == 'top' else 'C'}"
                                       f"{group}: {name}", "point": point[group]})
    low, span = xy.min(axis=0), np.ptp(xy, axis=0) + 1e-12
    unit = {g: tuple((np.asarray(v) - low) / span) for g, v in point.items()}
    slots = layout_crop_ring(unit, (0.5, 0.5), n_slots=max(len(point), 8))
    missing = [k for k, v in set_record.items() if v == "missing"]
    legend = (f"{LAYOUT_NOTE} Clusters: {cluster_record}. Labels: best term with BH q < "
              f"{s['ops_umap_label_q']} ({s['ops_umap_enrichment_test']}); {real} of "
              f"{len(chosen)} clusters labelled, {shuffled} with shuffled gene labels "
              f"(seed {s['ops_umap_shuffle_seed']}, same cluster sizes)."
              + (f" Sets not found: {missing}." if missing else "")
              + (" Crop display ranges: " + "; ".join(
                  f"ch{c} {v['low']:.4g}-{v['high']:.4g}" for c, v in display["channels"].items())
                 if display else "") + " Non-targeting rows grey, excluded from enrichment.")
    fig = _figure(coords, hierarchy, z, top, panels, slots, max(len(point), 8), legend,
                  dark=bool(s.get("black_background", False)),
                  per_cluster=s["ops_umap_crop_level"] != "top")
    import matplotlib

    with matplotlib.rc_context({"svg.hashsalt": "spacr-ops-umap"}):
        for ext in ("pdf", "png", "svg"):
            fig.savefig(os.path.join(dst, f"ops_umap.{ext}"), dpi=200,
                        facecolor=fig.get_facecolor(),
                        metadata={"CreationDate": None} if ext == "pdf" else
                        ({"Date": None} if ext == "svg" else None))

    inputs = {k: s.get(k) for k in ("ops_umap_profiles", "ops_umap_guide_profiles",
                                    "ops_umap_cell_embedding") if s.get(k)}
    if s.get("ops_umap_partition"):
        inputs["ops_umap_partition"] = cluster_record["file"]
    provenance = {
        "settings": {k: v for k, v in s.items() if k.startswith(("ops_umap", "umap_source"))
                     or k in OPS_UMAP_LAYOUT_DEFAULTS},
        "input_md5": {k: file_md5(v) for k, v in inputs.items()},
        "random_state": int(s["random_seed"]), "n_jobs": 1,
        "versions": library_versions(), "backend": "umap-learn (CPU)",
        "clusters": cluster_record, "label_sets": set_record,
        "enrichment_control": control_record, "note": LAYOUT_NOTE,
    }
    with open(os.path.join(dst, "umap_provenance.json"), "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True, default=str)
    return {"dst": dst, "clusters": int(len(chosen)), "labelled": real,
            "labelled_shuffled": shuffled, "groups": len(groups)}
