# Item spec: Brieflow cell-to-gene profile recipe as a selectable option in spaCR's OPS gene-profile builder

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first.
- **Extends:** the OPS gene-profile builder (per-cell features to per-gene profile). Related: `ITEM_ops_cluster_bench.md` (the tests that decide the default), `ITEM_ops_consensus_clustering.md`, `ITEM_ops_feature_families.md`.
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs identically on the validation data. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes (development analysis of the OPS gene profile).

## Why
spaCR's current gene profile (per-well scaling, contrastive Wiener weighting, guide-efficacy blend) was compared on the same cells with the cell-to-gene recipe of Brieflow (Di Bernardo et al., bioRxiv 2025, doi 10.1101/2025.05.26.656231). In development benchmarks with the calibrated tests of the cluster benchmark spec, swapping single steps showed that one step carries most of the gain: aggregating cells to a gene with a 10 % trimmed mean instead of a median or a mean. Making the whole Brieflow recipe selectable, with the trimmed mean as a separate switch, lets users reproduce either pipeline and lets the default be chosen on held-out evidence.

## Behaviour
A new profile recipe option `ops_profile_recipe` with values `spacr` (current behaviour; stays the default until the acceptance below is met) and `brieflow`.

**Brieflow recipe (steps in order):**
1. Cell filters: perturbation filter; missing-value filter (a feature is dropped above 5 % missing, a cell above 1 %) with KNN imputation of the remainder; local-outlier-factor filter at 1 % on mean intensities.
2. Per well: standard scaling, then PCA keeping 99 % of variance, fitted on a 100,000-cell sample (seed 0).
3. Typical variation normalisation on non-targeting (NT) cells, epsilon 0: global NT standard scaling, PCA on NT cells, per-well NT standard scaling, then CORAL alignment with ridge 0.5.
4. Gene rows from cells (aggregation option below). NT rows are built per barcode; guide rows are built the same way per guide. Rows with 10 cells or fewer are dropped.

**Options (each a separate setting, usable with either recipe):**
- `ops_profile_aggregate`: `median` | `mean` | `trimmed_mean`. The Brieflow recipe's own default is `median`. The trimmed mean is the 10 % two-sided trimmed mean (`trim_fraction` 0.1, `scipy.stats.trim_mean` semantics). In development benchmarks it was the only single-step change that raised complexes and STRING pairs on the selection half (+2.1 and +1.2 points over spaCR's current profile; +1.95 points of pairs over the Brieflow recipe with the median), and it was kept in every later chain.
- `ops_profile_nt_centering`: `none` | `per_well` | `per_field`. Per-field NT centring subtracts, from every cell, the mean of the NT cells of the same field of view (fields with too few NT cells fall back to the well). Off by default; it is evaluated as an option, not assumed to help.
- `ops_profile_phase_aware`: `off` | `on`. Cell-cycle phase (from the existing phase calls) is used when aggregating: cells are trimmed within phase and phase fractions can be kept as extra columns. Off by default. In development benchmarks the phase-aware variant gave a small gain in one chain (pairs +1.9 points over the Brieflow recipe, q 0.047) and a loss in another, so it stays an option until confirmed.
- `ops_profile_embedding`: the downstream gene-row embedding (drop columns with >= 5 % missing, median impute, standardise, PCA 99 %), identical for all recipes so that profiles are compared under one clustering stage.

## What to build
In the OPS profile module (alpha): `build_gene_profile(cells, recipe, aggregate, nt_centering, phase_aware, settings)`; a recipe registry (`spacr`, `brieflow`); `aggregate_cells(values, method, trim_fraction)`; the per-field NT-centring step; the phase-aware step. Settings-file keys for each option; every choice is written to the run record. The Brieflow recipe is implemented in spaCR code (no runtime dependency on Brieflow); Di Bernardo et al. 2025 is cited in the docstring and the documentation.

## Settings (alpha)
`ops_profile_recipe` ('spacr' | 'brieflow'), `ops_profile_aggregate`, `ops_profile_trim_fraction` (0.1), `ops_profile_nt_centering`, `ops_profile_phase_aware`, `ops_profile_min_cells` (10), `ops_profile_seed` (0).

## Status of the default
Candidate default after held-out confirmation. The default changes only if, with the calibrated cluster benchmark, the recipe is not worse than the current profile on the report half and held-out genes (complexes and STRING pairs, Holm-adjusted, beyond the noise-floor margin) and the result replicates on a second plate. Until then the option is available and the default stays `spacr`. Development benchmark so far: the Brieflow recipe with the trimmed mean was about +1.9 points of STRING pairs over the current profile on the report half and on a second plate, and not distinguishable on complexes; a combined profile did not beat it on pairs. These are development numbers, not the confirmation.

## Outputs
In `results/ops_profiles/`: the gene-profile table, the guide-profile table, `profile_recipe.json` (every option and seed), and a short QC table (cells kept per filter, rows kept).

## Acceptance and tests
- **Reproduction:** with pinned versions, a settings-file run reproduces the profile table identically (integers exact, floats <= 1e-12) and the same downstream partitions for the same seeds.
- **Tests (CPU):**
  - `trimmed_mean` equals `scipy.stats.trim_mean(x, 0.1, axis=0)` per gene; `median` and `mean` match numpy;
  - on planted data with heavy-tailed cell outliers, the trimmed mean recovers the planted gene profile with lower error than the mean;
  - per-field NT centring removes a planted field offset and keeps a planted gene effect;
  - phase-aware aggregation removes a planted phase-imbalance confound;
  - NT rows are one per barcode; rows with <= 10 cells are dropped;
  - shuffling gene labels gives an excess of about 0 on the cluster benchmark;
  - each option is recorded in `profile_recipe.json`, and a re-run from that file is identical.
- **Held-out acceptance (decides the default):** the comparison above, run through the cluster benchmark spec, on held-out genes and a second plate.

## Literature
- Di Bernardo et al. 2025 (bioRxiv), doi 10.1101/2025.05.26.656231 (Brieflow); Funk et al. 2022 (Cell).

## Effort
- **Effort:** M.
