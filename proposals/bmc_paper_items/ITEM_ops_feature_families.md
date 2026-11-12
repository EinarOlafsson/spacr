# Item spec: per-cell measurement family audit against Brieflow's feature set (spaCR measurement module)

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first.
- **Extends:** spaCR's per-cell measurement module. Related: `ITEM_ops_profile_recipe.md`, `ITEM_ops_cluster_bench.md`.
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs identically on the validation data. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes (OPS gene-profile quality).
- **Status:** pending results. A per-channel feature-family ledger is in progress. This item records the audit design and the rule for adding measurements. No family is added until the ledger shows it improves cluster quality.

## Why
On the same matched cells, spaCR's per-cell features gave a lower cluster quality than Brieflow's features (development benchmark, calibrated tests: STRING pairs about 1.4 to 1.8 points lower at 221 and 466 clusters). The cells were not the cause: at equal cell counts the two pipelines gave no detectable difference. Which feature families explain the gap is not known, and guessing would add many columns that do not help.

## Behaviour (the audit)
1. **Family map.** Every per-cell feature (spaCR's about 350 columns and Brieflow's) is assigned a family and, where it has one, a channel. Channels are matched between the two pipelines by the correlation of matched-cell mean intensities. Families: intensity statistics; edge intensity; intensity location and moments; radial distribution; texture (Haralick / GLCM; parameter-free threshold adjacency statistics as its own family); granularity and foci; colocalisation; shape and size; neighbours.
2. **Ledger.** For each family and each channel, on the matched cells and under one identical profile pipeline (column filter, standardise, PCA 99 %, typical variation normalisation, 10 % trimmed-mean gene rows, same embedding and clustering stage):
   - leave-one-family-out from the Brieflow feature set;
   - add-one-family to the spaCR feature set;
   - spaCR union Brieflow, with and without correlation filtering at |r| 0.95;
   - count-matched random-feature controls.
3. **Calls.** The calibrated tests of the cluster benchmark spec (R = 20 paired replicates, exact sign-flip, two-level interval, noise-floor margins). A family "explains the gap" if removing it from Brieflow lowers quality and adding it to spaCR raises quality, both beyond the margin and beyond the matched random control.

## What to build (conditional on the ledger)
- `feature_family_map()` returning the family and channel table for spaCR's measurements, saved with the run.
- For each family that wins the ledger: the measurement function (documented with how Brieflow computes it, cited), added to spaCR's measurement module behind a setting, with the family name in the output column prefix.
- A settings key `measure_families` listing the enabled families. The default stays as now until a family is confirmed on held-out genes and a second plate.

## Outputs
`feature_family_map.csv`; the ledger table (one row per arm: family, channel, comparison, effect, interval, p, q, margin, call); the list of families added.

## Acceptance and tests
- **Pending results:** acceptance for a given family is the ledger call above, then held-out and second-plate confirmation.
- **Tests (CPU):**
  - the family map covers every measured column exactly once;
  - a planted informative feature family is found by add-one and leave-one-out, and a planted noise family is not;
  - the random control gives about zero effect;
  - each added feature matches a reference implementation on synthetic images within tolerance.

## Literature
- Di Bernardo et al. 2025 (bioRxiv), doi 10.1101/2025.05.26.656231 (Brieflow); Haralick et al. 1973 (IEEE Trans Syst Man Cybern), texture; Hamilton et al. 2007 (BMC Bioinformatics), threshold adjacency statistics; Manders et al. 1993 (J Microsc), colocalisation; Funk et al. 2022 (Cell). The citation of each new measurement is added as it is implemented.

## Effort
- **Effort:** S for the audit; M for each added family.
