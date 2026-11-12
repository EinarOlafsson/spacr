# Item spec: interactive OPS cluster viewer (new spaCR module)

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first.
- **Builds on:** the volcano explorer (item 265) and the object filter explorer (item 822) for interaction patterns. Related: `ITEM_ops_cluster_bench.md`, `ITEM_ops_cluster_annotator.md`, `ITEM_ops_reference_sets.md`.
- **Rule:** every view is reproducible from a settings file (spacr-run or GUI Import settings), giving identical exported figures and tables. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes (exploration figure; Brieflow ships a comparable viewer).

## Why
Users need to explore gene clusters interactively: find a gene, see its neighbours, see what a cluster looks like in the cells, and check how clusterings agree, without writing code.

## Behaviour
- **Map panel:** a 2-D gene map (t-SNE / UMAP / PHATE / PCA, with trustworthiness shown) coloured by cluster. Hover shows the gene, cluster and stability; lasso selects. Search box for genes and complexes.
- **Overlays:** CORUM complexes, STRING edges for the selection, user gene lists, and non-targeting controls.
- **Cluster panel:**
  - member list with per-gene activity and q;
  - cluster stability;
  - enrichment against reference sets (with the size-matched null);
  - mean-feature heatmap against non-targeting controls;
  - representative cell crops per gene from the screen's crop store (raw crops, normalised for display only, with the display range stated);
  - spaCR AI annotation, if run.
- **Compare panel:** two clusterings side by side (for example spaCR vs an imported Brieflow or Funk table), with a correspondence (Sankey or contingency) view and per-complex kept-together status.
- **Export:** selected-cluster tables, figures (PNG/SVG, spaCR plain theme) and a settings file that reproduces the view.
- **Scale:** 5,000 genes and 300 clusters stay interactive. Crops load lazily.

## Website counterpart (later phase)
- The same viewer is published on the spaCR website as a static, browser-only page per published screen. No server-side compute; data are precomputed JSON/Parquet tiles exported by the headless export API.
- Shared data contract: the desktop module and the website read the same exported bundle (map coordinates, clusters, stability, enrichment, mean-feature tables, crop thumbnails, optional annotations), so a view on the website matches the desktop export.
- Website features: map, search, cluster panel, complex overlays, compare panel and downloads. Crops are served as small display thumbnails, with the display range stated.
- First published bundle: the Funk 2022 HeLa reanalysis (spaCR clusters, with Brieflow and Funk published clusters for comparison).
- Built after the desktop module, as its own item; it follows the site build pipeline (site-src).

## What to build
- `spacr.ops_cluster_viewer` (Qt module plus a headless export API).
- Importers for spaCR cluster tables, Brieflow `phate_leiden` output and Funk published cluster CSVs.
- Tests: load / export round trip, identical exports from settings, the importers and a headless render.
