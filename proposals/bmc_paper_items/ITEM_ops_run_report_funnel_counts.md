# Item spec: OPS run report: per-well yield funnel and per-guide counts against a published reference

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first.
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs identically on the validation data. Every seed is fixed as in the reference. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.
- **Paths:** R = `/nas_mnt/data/sync/papers/BMCspacr/reanalysis`. Source inventory: `R/SPACR_PROVENANCE.md` (port P4).

## Why
The per-well cell-yield funnel and the per-guide count agreement with a published count table come from pilot scripts. No spaCR run writes these tables. Two count levels exist and must never be mixed:
- SBS level (decoded nuclei), from `finish_plate_final.py`;
- after phenotype (linked, owned cells), from `qc_profiles.py`.

## Reference implementation
- **Funnel:** `R/funk_plateA_full/scripts/aggregate_final.py` (stage counts) → `R/funk_plateA_full/scripts/figures.py::funnel` → `R/funk_plateA_full/csv/funnel_per_well.csv` and `funnel_shipped_per_well.csv`; also `/mnt/firecuda2/ops_decode_yield/src_ruleA_S2_20261107/scripts/funnel_final.py`.
- **Counts vs published:** `/mnt/firecuda2/ops_decode_yield/src_ruleA_S2_20261107/scripts/finish_plate_final.py` (Spearman ρ per guide and per gene, with spaCR counts scaled to the reference table's pooled total; shuffled-label control, 95th percentile of abs(ρ)) → `R/decode_yield_2026-11-04/plates/plateA_final/per_guide_counts_spacr_plateA_final_vs_funk.csv`. The after-phenotype variant is `R/funk_plateA_full/scripts/qc_profiles.py` → `csv/counts_per_guide_final_vs_funk.csv`.

## What to build
A new phase `report` at the end of `spacr.ops_engine.run_ops` (alpha). It writes `<dst_root>/results/ops_report/`:
1. **`funnel_per_well.csv`:** columns plate, well, segmented, whole_and_owned, linked, with_library_guide, interphase, decoy_assigned, dropped_failed_fields, dropped_nonfinite. Every count is read from the tables the earlier phases wrote, never recomputed differently.
2. **`counts_vs_reference.csv` + `counts_vs_reference.json`:**
   - `ops_reference_counts`: a path to a published per-guide count table (columns guide, gene, count), e.g. the Funk et al. 2022 supplementary counts.
   - `ops_reference_counts_level`: `sbs` (decoded nuclei) | `phenotype` (linked, owned cells); default `sbs`. Both levels can be listed.
   - `ops_reference_shuffles` (1000) and `ops_reference_seed` (0) control the shuffled-label control. Report the 95th percentile of abs(ρ).
   - The output records which level made each ρ, so two levels cannot be mixed downstream.

## Validation data and acceptance
Acceptance: identical output, or within the stated tolerance, to the reference outputs at the paths below.
- **Plate A funnel:** `funnel_per_well.csv` equal, integer for integer, to `R/funk_plateA_full/csv/funnel_per_well.csv`. The pilot's guide-stage count was rounded and predates the item-861 drop; the exact integers in that CSV are the target, and the item should record the 861 difference.
- **Counts vs published:** ρ per guide / per gene equal to the two reference CSVs above to 1e-12, at both levels. The shuffled 95th percentile is equal with the same seed: re-run the reference with seed 0 if it used another seed, and record that in the item.
- **Tests:** synthetic counts with a known ρ; the level is recorded in the JSON.

## Effort
- **Effort:** S.
