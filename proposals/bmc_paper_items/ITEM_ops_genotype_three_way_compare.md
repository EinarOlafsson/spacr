# Item spec: per-cell genotype comparison against external call tables

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first.
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs identically on the validation data. Every seed is fixed as in the reference. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.
- **Paths:** R = `/nas_mnt/data/sync/papers/BMCspacr/reanalysis`. Source inventory: `R/SPACR_PROVENANCE.md` (port P7).

## Why
spaCR has no way to match its per-nucleus genotype calls to per-cell call tables produced by other OPS pipelines and to report agreement categories with a decoy FDR per category. The existing matching and category logic is custom, and its spaCR arm is a re-implementation (`decode_spacr.py`) that does not import spaCR. The external tools themselves are not ported; their call tables are inputs.

## Reference implementation
`R/ops_3way_percell/code/`:
- **spaCR arm:** `decode_spacr.py` over the cache from `run_spacr_cache.sh`;
- **matching / categories:** the compare scripts in the same folder. Also the older `/mnt/wd4tb/ops_3way/compare`, archived in `R/ops_overlap/code_archive`.
- The external call tables are already produced in the same folder; the scripts that produced them are not part of this item.

## What to build
1. **spaCR arm:** none new. Run the 842 decode (spec `ITEM_ops_decode_rule_a_s2.md`) on the same 25 fields per plate, from a settings file.
2. **`spacr.ops_engine` phase `compare_calls` (alpha).** It reads external per-cell call tables and matches them to spaCR nuclei. Settings:
   - `ops_compare_tables` (dict name → path; columns: x, y in the SBS frame, or a nucleus id, plus guide);
   - `ops_compare_match_px` (as in the reference);
   - `ops_compare_fields` (list, or a sample spec with a seed).
3. **Outputs:** `results/ops_compare/categories.csv`:
   - categories all-agree, two-agree, spaCR-only, other-only, decoy-assigned;
   - each with a decoy FDR per category and table;
   - per-table assigned % and FDR.

## Validation data and acceptance
- **Data:** the 25 fields per plate in `R/ops_3way_percell` (field list in the code), with the external call tables already produced there.
- **Acceptance** (identical to the reference outputs in `R/ops_3way_percell`):
  - identical matched-nucleus count;
  - identical category counts;
  - spaCR arm calls identical to `decode_spacr.py` output. Any difference is a bug in one of the two, and must be resolved and written into the item.

## Effort
- **Effort:** M.
