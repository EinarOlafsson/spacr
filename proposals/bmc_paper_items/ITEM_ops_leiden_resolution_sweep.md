# Item spec: Leiden clustering of OPS gene profiles with a resolution sweep, scored against reference sets with shuffled-cluster controls

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first. Related:
  - `ITEM_ops_profile_pipeline.md` (the gene-profile rows this clusters), `ITEM_ops_reference_sets.md` (CORUM / KEGG / STRING sets), `ITEM_ops_modules_precomputed.md` (module atlas, which also clusters).
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs identically on the validation data. Every seed is fixed as in the reference. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.

## Why
spaCR has no graph community clustering of gene profiles. Funk et al. 2022 (Cell) and Brieflow (Di Bernardo et al., bioRxiv 2025) cluster OPS gene profiles with PHATE + Leiden; Brieflow reports 227 interphase clusters at resolution 12 and scores them by STRING pair F1 and CORUM / KEGG enrichment (interphase STRING F1 0.098 vs Funk 0.067; CORUM 33.0 % vs 23.9 %; KEGG 18.1 % vs 13.5 %). Every cluster metric moves with the number of clusters (precision up and recall down with resolution, % clusters enriched down), so a fair comparison needs the whole resolution curve, a matched cluster count and a shuffled-cluster control at every resolution.

## Reference implementation
- Graphs:
  - `phate`: `phate.PHATE(random_state=seed, n_jobs=2, knn=10, knn_dist="cosine")`, diffusion operator `graph.diff_op`, symmetrised (W + Wᵀ)/2, upper triangle as an undirected weighted igraph;
  - `knn`: symmetric cosine kNN graph (k = 12), union of each row's k nearest rows, edge weight = cosine clipped at 1e-6.
- Leiden: `leidenalg.find_partition(g, RBConfigurationVertexPartition, weights="weight", n_iterations=-1, seed=seed, resolution_parameter=r)`; grids phate r ∈ {1, 2, 3, 5, 7, 8, 10, 12, 13, 15, 18, 22, 26, 32, 40}, knn r ∈ {0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 13, 16}; seeds 42, 1, 2.
- Every row is clustered (non-targeting rows included, as Brieflow); scores use the targeting genes of a fixed gene list.
- Scoring = Brieflow's benchmark code unchanged (two-sided Fisher per cluster × group, BH within cluster, enriched at FDR < 0.05; "head" = groups restricted to the clustered genes, "may2025" = unrestricted; STRING pair precision / recall from co-membership) plus STRING F1 = 2PR/(P+R), % CORUM complexes (≥ 3 members, report half) with ≥ 50 % of members in one cluster, and % genes with a CORUM partner in their cluster.
- Control per partition: gene labels permuted (cluster sizes kept), `np.random.default_rng(20261109)` in loop order.
- Per graph, flag the resolution whose mean cluster count is nearest a target (default 227).

## What to build
New `spacr.ops_clusters` (alpha): `gene_graph(rows, kind, k, seed)`, `leiden_partitions(graph, resolutions, seeds)`, `score_partitions(names, partitions, refs, shuffles, seed)`; a settings-file phase `cluster_sweep` after the profile phase. `leidenalg`, `igraph`, `phate` are optional extras, imported inside the functions, with an error that names the extra when missing.

## Settings (alpha)
`ops_cluster_graph` ('phate' | 'knn'), `ops_cluster_knn` (12), `ops_cluster_resolutions` (the grids above), `ops_cluster_seeds` ([42, 1, 2]), `ops_cluster_target_clusters` (227), `ops_cluster_shuffles` (1), `ops_cluster_reference_dir`, `ops_cluster_gene_list` (optional).

## Outputs
`results/ops_clusters/sweep_rows.csv` (graph, seed, resolution, control, clusters, genes, CORUM/KEGG % enriched head and may2025, STRING precision/recall/F1/tp/fp/fn, complexes recovered %, partner-in-cluster %), `sweep_summary.csv` (mean/min/max over seeds, nearest-target flag), `assignments_<graph>_s<seed>.csv.gz`, one figure (metric vs clusters, real vs shuffled, published comparator points).

## Validation data and acceptance
- **Data:** the final Funk 2022 plate A gene-profile rows and the fixed shared gene list held with the reference outputs.
- **Acceptance:** identical partitions for every (graph, seed, resolution) and identical `sweep_rows.csv` (integers exact, floats ≤ 1e-12), with phate 2.0.0, leidenalg 0.12.0, igraph 1.0.0, scikit-learn 1.9.1 pinned; a version change re-baselines the reference with the versions recorded.
- **Tests (CPU):** a planted-block similarity graph recovers its blocks at the matching resolution; shuffled labels give near-zero enrichment; a missing optional dependency raises the named error.

## Literature
- Traag, Waltman & van Eck 2019 (Sci Rep), Leiden algorithm; Moon et al. 2019 (Nat Biotechnol), PHATE.
- Funk et al. 2022 (Cell); Di Bernardo et al. 2025 (bioRxiv, Brieflow).

## Effort
- **Effort:** S–M.
