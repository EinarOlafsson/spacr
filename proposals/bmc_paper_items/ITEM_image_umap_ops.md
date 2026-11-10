# Item spec: Image UMAP for OPS gene and guide profiles: display-only map, clusters and a cluster hierarchy from the full-profile kNN/Leiden graph drawn on it, medoid-cell raw crops around the map with leader lines, enrichment labels with q-values, recorded library versions

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first. Extends spaCR's Image UMAP (`spacr.core.generate_image_umap`, `spacr.settings.set_default_umap_image_settings`, `spacr.umap_search.UmapRecipe`, `spacr.utils.plot_embedding` / `plot_umap_images`); it does not add a second UMAP module. Related:
  - `ITEM_ops_leiden_resolution_sweep.md` (`ops_clusters`, kNN / PHATE graphs and Leiden): this item calls it and does not re-implement Leiden;
  - `ITEM_ops_reference_sets.md` (versioned CORUM / GO sets), `ITEM_ops_profile_pipeline.md` (gene and guide profiles), `ITEM_ops_phenotype_run_v2.md` (raw crops / merged fields the crops come from);
  - 869 (AI Console cluster annotator): a different, optional label source.
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs identically on the validation data: same partitions and hierarchy, same enrichment table, same medoid cells and crops, and (with the recorded umap-learn / pynndescent / numba versions) the same UMAP coordinates. Every seed is fixed. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.

## Why
OPS gene maps are usually drawn outside the analysis tool: `umap.UMAP(n_neighbors=15, min_dist=0.3, metric="cosine", random_state=42)` on the gene profile PCs, coloured by a Leiden partition made elsewhere. spaCR's Image UMAP works on per-object `measurements.db` rows; it clusters **on the 2-D embedding** (`clustering` = dbscan / kmeans in `reduction_and_clustering`; HDBSCAN in `umap_search.cluster_embedding`) and places random or per-cluster thumbnails on the map. For OPS gene profiles this is wrong in two ways:
1. clusters must come from the full profiles (the space every benchmark scores), not from UMAP coordinates, which distort distances and depend on library versions;
2. a gene is not an image: the representative picture of a cluster must be chosen by profile distance from the cells of its genes, and shown as raw crops with one shared, documented display range, not per-crop stretched thumbnails.
There is also no cluster hierarchy, no enrichment-based label, and the output does not record which umap-learn / pynndescent produced the layout.

## Reference implementation
- **Map:** `umap.UMAP(n_neighbors=15, min_dist=0.3, metric="cosine", random_state=42).fit_transform(gene_pcs)` → `gene, x, y, cluster`.
- **Clusters:** symmetric cosine kNN or PHATE graph → `leidenalg.find_partition(RBConfigurationVertexPartition, seed, resolution)`, as in `ITEM_ops_leiden_resolution_sweep.md`; 20 reference partitions of the final gene profile (seeds 42, 1–4 × resolutions 12, 13, 15, 18).
- **Enrichment:** Brieflow's benchmark `group_enrichment` (two-sided Fisher per cluster × group, BH within cluster) and the CORUM / GO sets of `ITEM_ops_reference_sets.md`.
- **Crops:** none yet (new); `spacr.crops` reads merged fields and sharded crops.

## What to build
In `generate_image_umap` (one entry, new `umap_source='ops_profiles'`):
1. **Inputs:** a gene profile file and optionally a guide profile file (`ops_umap_profiles`, `ops_umap_guide_profiles`; npz `names`/`pcs` or parquet `gene`/`guide` + feature columns), a per-cell table with the cells' profile-space rows (`ops_umap_cell_embedding`, e.g. the aligned per-cell embedding kept by the profile recipe) and crop addresses (well, site/field, centroid, or `png_path`).
2. **Map (display only):** UMAP of gene rows (and, if given, guide rows fitted jointly or projected with `transform`, setting `ops_umap_guides`), `n_neighbors`, `min_dist`, `metric`, `random_state`. Written as `umap_coords.csv`. The UI and the figure say "UMAP layout, for display; clusters are computed on the full profiles".
3. **Clusters from the full profiles:** kNN (cosine, k) or PHATE graph → Leiden (`RBConfigurationVertexPartition`, resolution, seed) through `ops_clusters`, or a partition column read from a file (`ops_umap_partition`). Never from UMAP coordinates for this source (the existing embedding clustering stays for per-object UMAPs).
4. **Hierarchy:** average-linkage (cosine) on the cluster centroid profiles in the full profile space (`scipy.cluster.hierarchy.linkage`, method set by `ops_umap_hierarchy_method`), cut at `ops_umap_hierarchy_levels`; optionally a multi-resolution Leiden containment tree (each fine cluster assigned to the coarse cluster holding most of its genes, with the containment fraction reported). Drawn on the UMAP as hulls / contours per top-level group and as a dendrogram panel; written as `cluster_hierarchy.csv` (cluster, parent, level, height, containment).
5. **Labels from enrichment:** per cluster, one-sided hypergeometric (or Fisher, the same test the benchmark uses; `ops_umap_enrichment_test`) against CORUM and GO BP (versioned, `ops_umap_label_sources`), background = all clustered genes, minimum overlap 2, BH across all cluster × term pairs; label = best term with q < `ops_umap_label_q` (0.05), else "unlabelled". Table `cluster_enrichment.csv` (cluster, source, term, overlap, size, p, q, label flag). The label drawn on the map carries its q-value.
6. **Representative crops:** per cluster (or per top-level group, `ops_umap_crop_level`): the medoid gene (minimum summed cosine distance to the cluster's genes in profile space), then that gene's `ops_umap_cells_per_cluster` (3) cells closest to the gene profile in the per-cell embedding (medoid cells; ties by cell id; cells with crops cut by the field edge or below `ops_umap_min_cell_px` skipped). Crops are raw (no per-crop stretch), same size (`ops_umap_crop_px`), channels and colours from `ops_umap_crop_channels`; ONE display range per channel shared by every crop in the figure (percentiles `ops_umap_display_percentiles` (0.5, 99.5) of the pooled shown crops, or fixed values), written to `crops_display.json` and printed in the legend text.
7. **Layout:** panels on a ring / border of slots around the map; each cluster assigned to a slot by the angle of its UMAP medoid point around the map centre (Hungarian assignment on angular distance), so leader lines (medoid point → panel) do not cross; leader lines thin, in the cluster colour. Theme-aware (light and dark), spaCR palette.
8. **Outputs:** `umap_coords.csv`, `clusters.csv`, `cluster_hierarchy.csv`, `cluster_enrichment.csv`, `cluster_medoids.csv` (cluster, medoid gene, cell ids, crop addresses), `crops/` (raw arrays + PNGs at the shared range), `crops_display.json`, `ops_umap.pdf/png/svg`, `umap_provenance.json`.
9. **Reproducibility (`umap_provenance.json`):** settings, input md5s, `random_state`, umap-learn, pynndescent, numba, scikit-learn, leidenalg, igraph versions, backend (CPU umap-learn; cuML maps are a different recipe as in `UmapRecipe`), and the note: "UMAP layouts differ across umap-learn / pynndescent / numba versions even with the same seed; clusters, hierarchy, labels and medoids do not depend on the layout." `n_jobs` is 1 whenever `random_state` is set (umap-learn forces this).

## Settings (alpha; added to `set_default_umap_image_settings`)
| Key | Default | Meaning |
|---|---|---|
| `umap_source` | `'measurements'` (today) | `'ops_profiles'` adds this item |
| `ops_umap_profiles` | None | gene profile file |
| `ops_umap_guide_profiles` | None | guide profile file |
| `ops_umap_guides` | `'none'` | `'joint'` or `'project'` |
| `n_neighbors`, `min_dist`, `metric`, `random_seed` | 15, 0.3, `'cosine'`, 42 for this source | UMAP (reference values) |
| `ops_umap_graph` | `'knn'` | or `'phate'`; k = `ops_umap_knn` (12) |
| `ops_umap_resolution` | 12 | Leiden resolution |
| `ops_umap_leiden_seed` | 42 | Leiden seed |
| `ops_umap_partition` | None | read partition column `file:column` instead |
| `ops_umap_hierarchy_method` | `'average'` | linkage on cluster centroids (cosine) |
| `ops_umap_hierarchy_levels` | [8, 30] | top-level group counts drawn |
| `ops_umap_label_sources` | ['CORUM@REPORT', 'GO_BP'] | versioned sets |
| `ops_umap_enrichment_test` | `'hypergeom'` | or `'fisher_two_sided'` (benchmark) |
| `ops_umap_label_q` | 0.05 | BH threshold for a label |
| `ops_umap_cell_embedding` | None | per-cell profile-space rows + crop addresses |
| `ops_umap_crop_level` | `'top'` | `'cluster'` or `'top'` (hierarchy level) |
| `ops_umap_cells_per_cluster` | 3 | medoid cells per panel |
| `ops_umap_crop_px` | 96 | crop side |
| `ops_umap_crop_channels` | None (required) | channel → colour |
| `ops_umap_display_percentiles` | (0.5, 99.5) | shared range per channel, pooled over shown crops |
| `ops_umap_min_cell_px` | 200 | skip tiny / edge-cut cells |

## Validation data and tolerances
- **Data:** Funk et al. 2022 plate A final gene profile (5,299 rows × 71 dims) with its 20 reference partitions; the plate A per-cell aligned embedding from the profile recipe; plate A merged fields / raw crops.
- **Partitions:** Leiden with seed 42, resolution 12 on the same graph gives the reference `s42_r12` column exactly.
- **Map:** the reference UMAP settings on the same rows give identical coordinates (float32) with the recorded versions (environment of record: umap-learn 0.5.12, pynndescent 0.5.13; confirm and pin). With other versions coordinates are not compared; everything else must still be identical.
- **Medoids and crops:** medoid cells and crop bytes identical across reruns.
- **Enrichment:** with the same reference-set versions, `cluster_enrichment.csv` identical (p, q, labels).
- **Tests (CPU, synthetic):** planted gene groups → the profile-graph clusters recover them while clustering a deliberately distorted 2-D layout does not; a planted "complex" gets its label with q < 0.05 and the shuffled control does not; medoid selection returns the planted central gene; the shared display range is the same number in every panel and is written to `crops_display.json`; two runs with the same seed give byte-identical outputs; changing only the UMAP seed changes `umap_coords.csv` but not clusters, hierarchy, enrichment or medoids.

## Controls
- **Shuffled gene labels** (cluster sizes kept; seed 7): number of clusters that get an enrichment label by chance, reported beside the real count.
- **NT / non-targeting rows** shown on the map in grey and excluded from enrichment.
- **Layout stability:** 5 UMAP seeds; the cluster-membership agreement of nearest neighbours in 2-D vs the profile graph (trustworthiness, k = 15) reported, so readers see how faithful the layout is.

## Catalogue and locales
New documented callables (the `ops_profiles` path of `generate_image_umap`, `cluster_hierarchy`, `enrich_clusters`, `medoid_cells`, `layout_crop_ring`) ship with arrival contracts and 9-locale records in the same commit; every new setting gets a tooltip and a GUI field under the Image UMAP alpha section.

## Literature
- McInnes L, Healy J, Melville J (2018) UMAP. arXiv:1802.03426.
- Dong W, Moses C, Li K (2011) Efficient k-nearest neighbor graph construction (NN-descent; pynndescent). WWW 2011.
- Traag VA, Waltman L, van Eck NJ (2019) From Louvain to Leiden. Sci Rep 9:5233.
- Kobak D, Linderman GC (2021) Initialization is critical for preserving global data structure in both t-SNE and UMAP. Nat Biotechnol 39:156–157.
- Chari T, Pachter L (2023) The specious art of single-cell genomics. PLoS Comput Biol 19:e1011288 (2-D distances are not profile distances).
- Funk L et al. (2022) Cell 185:4634 (gene maps of the screen); Tsitsiridis G et al. (2023) CORUM 4.0, Nucleic Acids Res 51:D539; Gene Ontology Consortium (2023) Genetics 224:iyad031.

## Effort
- **Effort:** M (map and clusters reuse existing code; new: hierarchy, enrichment labels, medoid crops, ring layout, provenance).
