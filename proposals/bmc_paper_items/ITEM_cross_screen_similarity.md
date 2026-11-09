# Item spec: cross-screen gene similarity (OPS ↔ JUMP)

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first.
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs identically on the validation data. Every seed is fixed as in the reference. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.
- **Paths:** R = `/nas_mnt/data/sync/papers/BMCspacr/reanalysis`. Source inventory: `R/SPACR_PROVENANCE.md` (port P12).

## Why
Comparing gene-gene similarity of the same genes across two screens (here an OPS screen in HeLa and the JUMP CRISPR/ORF consensus profiles, cpg0016) is custom code. Only a CPJUMP1 pilot script touches `spacr.sp_stats`.

## Reference implementation
`R/ops_jump_cross/scripts/`:
- `ojx_data.py` (loading and gene matching of OPS gene profiles and JUMP cpg0016 consensus profiles);
- `ojx_cpg0016.py` (Mantel test, permutations; Mantel 1967);
- `ojx_cpg0016_pairs.py` (top-pair STRING enrichment);
- `ojx_cpg0016_precise.py` (999 permutations);
- the named-pair test scripts in the same folder (pair-level tests with NT and strength nulls);
- `ojx_sensitivity.py`;
- `jump_compare_frozen.py` (CPJUMP1 pilot, `sp_stats`).

Source data: `R/ops_jump_cross/source_data/` (`cpg0016_final`, `cpg0016_interpretable`, `headline_by_jump_features.csv`, and the named-pair folder).

## What to build
`spacr.sp_stats.cross_screen_similarity(profiles_a, profiles_b, settings)`, exposed through the Measure profiling entry as an alpha analysis. Settings:
- `cross_screen_profiles_a`, `cross_screen_profiles_b` (paths to gene-level profile tables);
- `cross_screen_gene_key` ('gene');
- `cross_screen_metric` ('cosine');
- `cross_screen_permutations` (999);
- `cross_screen_seed` (0);
- `cross_screen_top_pairs` (as in the reference);
- `cross_screen_string` (path; versioned, md5 recorded);
- `cross_screen_pairs` (an optional list of named gene pairs to test).

Outputs:
- `results/cross_screen/summary.json` (Mantel r, p, controls);
- `pairs.csv` (rank agreement, STRING share);
- `named_pairs.csv`.

## Validation data and acceptance
- **Data:** the reference inputs in `R/ops_jump_cross/source_data/cpg0016_final` plus the plate A OPS gene profiles the reference used.
- **Acceptance:** identical output to the reference outputs in `R/ops_jump_cross/source_data/`: identical Mantel r and permutation p (same seed, 999 permutations), identical top-pair tables, and an identical named-pair result for the pair list the reference tested (read it from the named-pair folder).
- When the OPS profiles are later regenerated from the phenotype settings run (`ITEM_ops_phenotype_run_v2.md`), the analysis is re-run from settings.

## Effort
- **Effort:** M.
