# Item spec: OPS yield table with a read stage, and phenotype–genotype match rates in tile-overlap strips vs cores, per copy and per physical cell

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first. Related:
  - `ITEM_ops_run_report_funnel_counts.md` (the `report` phase and `funnel_per_well.csv`): **this item extends it**; build it first;
  - `ITEM_ops_cross_field_ownership.md` (mask-based ownership components): required for the per-physical-cell unit.
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs identically on the validation data. Every seed is fixed as in the reference. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.

## Why
Brieflow reports a yield table for the Funk screen (74.4 M cells, 79.2 % with a barcode, 57.0 % unique gene, 76.4 % phenotype recovery) and states that unmatched cells sit in tile overlaps (97 % matched in non-overlapping regions vs 53 % in overlapping regions). spaCR's funnel has no "nuclei with ≥ 1 read" stage and no per-zone match report. A per-copy strip rate mixes lost cells with duplicate copies of cells matched elsewhere, so the report must give both units, per segmented copy and per physical cell.

## What to build (in the `report` phase)
1. `funnel_per_well.csv` gains `sbs_nuclei_with_read` and its %: nuclei with ≥ 1 spot above the read threshold attributed to their territory, counted in the decode phase and stored in its per-site stats.
2. **Zones:** per field transform (refined when available), a cell centroid is in an OVERLAP STRIP when it maps inside ≥ 1 other acquired field (inverse transform inside [0, H) × [0, W)), else CORE; distance to the nearest field edge in 100 px bins.
3. `overlap_match_by_zone.csv`: well, unit ('copy' | 'physical_cell'), zone, cells, matched, %; copy matched = linked; physical cell matched = any copy linked, zone = owner copy's zone. `overlap_match_by_edge_distance.csv`; `overlap_match_per_field.csv` (fields with ≥ 50 cells per zone).
4. Optional external comparison: `ops_report_external_cells` (field, label, y, x, matched) assigned to the nearest spaCR nucleus centroid in the same field within `ops_report_external_max_px` (8), giving per-copy and per-physical-cell rates on the same physical cells with exact McNemar counts.

## Settings (alpha)
`ops_report_read_stage` (bool), `ops_report_overlap_zones` (bool), `ops_report_edge_bin_px` (100), `ops_report_external_cells` (null), `ops_report_external_max_px` (8.0).

## Validation data and acceptance
- **Data:** Funk 2022 plate A well A1: mask-pass tables, links, field transforms, ownership table and decode stats, with the reference outputs.
- **Acceptance:** identical counts per zone, unit and bin (integers exact) and identical read-stage counts.
- **Tests (CPU):** two overlapping synthetic fields with one cell copied in both: per copy the strip rate is 50 %, per physical cell 100 %.

## Literature
- Di Bernardo et al. 2025 (bioRxiv, Brieflow); Funk et al. 2022 (Cell).

## Effort
- **Effort:** S–M (after the funnel and ownership items).
