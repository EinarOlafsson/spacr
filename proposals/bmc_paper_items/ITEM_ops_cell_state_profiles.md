# Item spec: cell-state stratified OPS gene profiles (interphase-only and mitotic-only, each through the full profile recipe) and a per-gene mitotic index test

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first. Related:
  - 856 (trained mitotic classifier, `spacr.ops_mitotic`): required first;
  - `ITEM_ops_cell_cycle_profiles.md` (per-phase centring inside one embedding, [G1 | S-G2 | M] concatenation): different method; this item runs the whole recipe separately per cell class, as Brieflow does;
  - `ITEM_ops_profile_pipeline.md`, `ITEM_ops_gene_level_aggregation.md`, `ITEM_ops_contrastive_weighting.md`, `ITEM_ops_guide_efficacy_blend.md` (the recipe steps reused per class).
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs identically on the validation data. Every seed is fixed as in the reference. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.

## Why
Funk et al. 2022 and Brieflow analyse interphase and mitotic cells separately. Brieflow's cell-state classifier (XGBoost, top 50 features, ~800 labelled cells, 96.3 % accuracy) feeds separate interphase (227 clusters) and mitotic (222 clusters) profiles, filtered, aligned and aggregated per class. spaCR drops mitotic cells with a DNA-intensity / nuclear-area rule; it has a trained classifier (856) but no profile path that uses it per class, and no per-gene mitotic index test against non-targeting cells.

## Reference implementation
1. **Calls:** `load_mitotic_classifier` + `predict_mitotic` (bundle threshold) per cell, row-aligned with the cell table.
2. **Recipe per class** (class = classifier interphase, classifier mitotic, or all cells), Brieflow's aggregation code unchanged: per well `perturbation_filter` → `missing_values_filter(0.05, 0.01, impute=True)` → `LocalOutlierFactor(contamination 0.01)` on intensity means → `prepare_alignment_data`; PCA(0.99) fitted on a 100,000-cell sample (`np.random.seed(0)`) of `centerscale_by_batch`; `centerscale_by_batch` → PCA → `tvn_on_controls`; median per gene (cell_count > 10) and per guide; the per-cell aligned embedding is kept.
3. **Final weighting per class:** S02 contrastive weighting + X04 guide-efficacy blend on that class's gene and guide tables (gene universe restricted to genes with > 10 cells in the class). On the interphase tables of the shipped profile this reproduces the shipped rows to 1e-14.
4. **Mitotic index test:** per gene, classifier-mitotic share of its cells; two-sided exact binomial test against the pooled NT rate, BH; NT guides as the control; Spearman ρ with a published per-gene mitotic index and a 200-shuffle control (seed 0).

## What to build
`ops_engine._ph_aggregate`: a `cell_state` grouping before the profile phase, one profile set per state; `spacr.ops_mitotic.mitotic_index_table` (alpha).

## Settings (alpha)
`ops_profile_cell_state` ('interphase_rule' (today) | 'interphase_classifier' | 'mitotic_classifier' | 'all', or a list), `ops_mitotic_source` ('rule' | 'classifier'; shared with the cell-cycle item), `ops_mitotic_classifier_path`, `ops_mitotic_index_test` (bool), `ops_mitotic_index_min_cells` (50).

## Outputs
Per state `genes_<state>.parquet`, `guides_<state>.parquet`, `rows_<state>.npz`; `mitotic_index_per_gene.csv` (cells, mitotic, index, z, p, q), `mitotic_index_nt_guides.csv`, `mitotic_index_summary.json`.

## Validation data and acceptance
- **Data:** Funk 2022 plate A per-cell tables and the 856 plate A classifier, with the reference per-state tables and rows.
- **Acceptance:** identical per-cell calls, per-state gene/guide tables (float32) and rows, and identical mitotic-index tables.
- **Tests (CPU):** a synthetic two-state population gives separate profiles whose state-specific gene effect appears only in its state; the index test calls a planted high-mitosis gene and not NT guides.

## Literature
- Funk et al. 2022 (Cell); Di Bernardo et al. 2025 (bioRxiv, Brieflow).

## Effort
- **Effort:** M (after 856 and the profile-pipeline items).
