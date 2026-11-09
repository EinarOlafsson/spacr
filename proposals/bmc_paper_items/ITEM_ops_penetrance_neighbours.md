# Item spec: OPS penetrance, escapers and neighbour effects

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first.
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs identically on the validation data. Every seed is fixed as in the reference. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.
- **Paths:** R = `/nas_mnt/data/sync/papers/BMCspacr/reanalysis`. Source inventory: `R/SPACR_PROVENANCE.md` (port P11). The detailed feature list is `R/funk_plateA_full/strategies/PORTING.md`, rows P1, P4–P8; this spec makes those rows one buildable item. The cell-cycle-resolved profiles are a separate spec (`ITEM_ops_cell_cycle_profiles.md`).

## Why
Per-gene penetrance (the fraction of a gene's cells that carry its phenotype), escaper detection (a sub-population that looks unperturbed), phase-coupled tests, and the effect of a knockout on its unperturbed neighbours are computed by custom scripts only.

## Reference implementation
`R/funk_plateA_full/strategies/scripts/`:
- **`s1c_penetrance.py`:**
  - classifier cross-fitted by field against NT cells;
  - scores calibrated on the NT percentile;
  - Storey estimator (Storey 2002) with a field-bootstrap CI;
  - controls: NT pseudo-genes and global label permutation.
- **`s1c_escapers.py`, `s1c_read_support.py`:**
  - mixture with a fixed NT component vs a unimodal shift; parametric simulations; Ashman's D (Ashman et al. 1994);
  - guide heterogeneity, phase coupling, genotype-error check.
- **`phase_cells.py`** (calls `spacr.ops_mitotic.predict_mitotic`), **`s2a_phase.py`** (`NULL=targeting`), **`s2a_report.py`**.
- **`neighbour_pairs.py`:** a mask-based neighbour table; radius 50 px, grid 2 px.
- **`s2b_neighbours.py`, `s2b_report.py`, `s2b_category_matched.py`:**
  - receivers whose own gene is inactive or NT;
  - FWL (Frisch–Waugh–Lovell) residualisation within field on density and position;
  - field-clustered variance;
  - nulls: within-field shuffle, clone-preserving shuffle, NT pseudo-genes.
- **`spill_*.py`:** spill-over check; `spill_remeasure.py` calls `ops_sample.measure_phenotype_field`.
- Tables in `strategies/tables/`.

## What to build
In `spacr.ops_engine` (aggregate phase) and `spacr.gene_measurement_compare` (alpha). Settings names are taken from strategies/PORTING.md:
- `ops_penetrance` (False), `ops_penetrance_folds` (5), `ops_penetrance_nt_pool` (20000), `ops_penetrance_lambda` (0.5);
- `ops_escapers` (False), `ops_escaper_dbic` (10), `ops_escaper_min_sep` (2);
- `ops_phase_tests` (False), `ops_phase_null` ('targeting' | 'nt'), using `ops_mitotic_model` (item 856);
- `ops_neighbour_radius_px` (50), `ops_neighbour_grid_px` (2) (neighbour table, also usable by `_spatial_measurements`);
- `ops_neighbour_effects` (False), `ops_neighbour_receivers` ('inactive' | 'nt'), `ops_neighbour_null_reps` (5).

Outputs go to `results/ops_penetrance/` and `results/ops_neighbours/`, with the same table names and columns as `strategies/tables/`.

## Validation data and acceptance
- **Data:** plate A (the same cell table and genotype rule as the reference). Reference tables are in `R/funk_plateA_full/strategies/tables/`.
- **Acceptance:** identical output to the reference tables at `R/funk_plateA_full/strategies/tables/`: with the same seeds, per-gene penetrance estimates, CIs, escaper calls and neighbour statistics are identical (CPU). The NT pseudo-gene and permutation controls are reproduced exactly.

## Effort
- **Effort:** M–L (penetrance/escapers M; neighbours M).
