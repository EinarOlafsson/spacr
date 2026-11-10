# Item spec: profile benchmark extensions for a second OPS screen: CORUM within-complex correlation, STRING AUPRC, and the PERISCOPE per-feature Mann-Whitney / zero-TPM hit rule, run with the same code on reference and spaCR profiles

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first.
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs IDENTICALLY on the validation data. Every seed is fixed. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes (PERISCOPE profile-quality rows; the same code re-scores Funk).
- **Overlap, by design:** this item **extends** `ITEM_ops_shared_scoring.md` (`spacr.ops_scoring.score_ops_profiles`: copairs activity and consistency mAP, EFAAR recall, shuffled and NT controls, "not scorable" rows). mAP via copairs is NOT re-specified here; it comes from `ITEM_ops_shared_scoring.md`. Pair AUPRC reuses `ITEM_ops_specificity_metrics.md` if it has landed. This item adds three metrics as new `ops_scoring_metrics` entries and a hit-calling step.

## Why
PERISCOPE (Ramezani et al. 2025) reports profile quality as CORUM within-complex correlation (median r 0.39 HeLa DMEM, 0.27 HPLM; 0 for random pairs) and STRING correlation by confidence, and calls hits with a per-feature rule: a Mann-Whitney test of each gene against non-targeting controls per feature at p = 0.001, gene score = number of significant features, null from zero-TPM (non-expressed) genes at their 95th / 99th percentile (5 % / 1 % FDR). A549 at 5 % FDR: 3,659 hits (1,339 whole-cell + 2,320 compartment-specific). The comparison is fair only if the same code scores their per-plate profiles and spaCR's, with the published hit list reproduced from their profiles as the control row.

## Reference implementation
- Their code: `2022_PERISCOPE` repository (the 2022_PERISCOPE repository: `Hit_Calling/` notebook, `Profile_Aggregation/`, `common_files/` with `CORUM_humanComplexes.txt` (accessed 2022-12-15), STRING v11.5, `CCLE_expression_A549_HeLa.csv` from DepMap Public 21Q4 (`CCLE_expression.csv`, A549 = ACH-000681, HeLa = ACH-001086)). The hit-calling notebook input is the guide-level `guide_normalized_feature_select_median_merged` profile (`..._CP186___ALLWELLS.csv.gz` for A549). **First step of this item: read the notebook and pin exactly** the unit of the test (guide-level rows of a gene vs NT guide rows, per feature), the zero-TPM definition (TPM = 0 in the 21Q4 log2(TPM+1) table), the compartment split (whole-cell vs compartment-specific features by name prefix), and the percentile rule.
- Deposited outputs: `A549_significant_features_mann_whitney_{p,u}_values.csv.gz` (per gene × feature), Supplementary Table 2 hit lists at 5 % and 1 % FDR, per-plate HeLa compartment / whole-cell hit tables (`Extended_Data_3/outputs/`).
- spaCR side: profiles from `ITEM_ops_profile_pipeline.md` / `ITEM_ops_phenotype_run_v2.md`; scoring host `spacr.ops_scoring`.

## What to build
New `ops_scoring_metrics` entries in `spacr.ops_scoring`:
1. **`corum_within_r`:** per CORUM complex with ≥ `ops_scoring_min_complex` members among the scored genes, the median Pearson r over member pairs (profile rows); summary = median over complexes; control = random gene pairs matched for count (seed). Complex version pinned (`ops_scoring_references`).
2. **`string_auprc`:** gene pairs ranked by cosine similarity, positives = STRING edges ≥ confidence threshold(s) (`ops_scoring_string_thresholds`, [0.4, 0.7, 0.9]); AUPRC and its random baseline; per-threshold curve of mean r by STRING confidence bin (their display).
3. **`hits_mw_ztpm`** (`spacr.ops_scoring.call_hits`): per gene × feature two-sided Mann-Whitney U (gene unit vs NT unit as pinned from the notebook) at `ops_hits_feature_p` (0.001); gene score = count of significant features (separately whole-cell and per compartment); null = the scores of zero-TPM genes (`ops_hits_expression_table`, `ops_hits_cell_line`); hits at the `ops_hits_fdr` levels (0.05 → 95th percentile, 0.01 → 99th). Outputs `hits.csv` (gene, score per compartment, hit flags), `hit_null.csv`, `hits_summary.json`.
4. **Overlap with a published list:** Jaccard and recall of a reference hit list restricted to genes with ≥ `ops_hits_min_cells` (20) cells on the scored plate.
5. All three accept the "not scorable" rows of (e.g. a clusters-only input cannot be scored for r or hits).

## Settings (alpha; added to those of `ITEM_ops_shared_scoring.md`)
| Key | Default | Meaning |
|---|---|---|
| `ops_scoring_metrics` | list | add `'corum_within_r'`, `'string_auprc'`, `'hits_mw_ztpm'` |
| `ops_scoring_string_thresholds` | [0.4, 0.7, 0.9] | STRING confidence |
| `ops_hits_feature_p` | 0.001 | per-feature threshold |
| `ops_hits_fdr` | [0.05, 0.01] | percentile levels of the zero-TPM null |
| `ops_hits_expression_table` | None (required for hits) | DepMap/CCLE TPM table, version recorded |
| `ops_hits_cell_line` | None | e.g. `'A549'` |
| `ops_hits_compartments` | `'periscope'` | feature-name → compartment map |
| `ops_hits_min_cells` | 20 | for overlap with a reference list |
| `ops_hits_reference_list` | None | published hit list for overlap |

## Validation data and acceptance
- **Data:** the deposited Tier 0 files (CP186A gene and gene-normalized per-plate profiles, the A549 gene-aggregated feature-selected profile, the Mann-Whitney p and U tables, single-cell metadata, `Barcodes.csv`); the 2022_PERISCOPE repository (Table S2, `common_files`, NGS counts); per-plate guide and single-cell profiles of CP186A (13.1 GB).
- **Acceptance:**
  - **control row:** from their deposited Mann-Whitney p-value table, `call_hits` reproduces their A549 Table S2 hit lists exactly (3,659 at 5 % FDR: 1,339 whole-cell + 2,320 compartment-specific; and the 1 % list);
  - from their guide-level profile (the notebook's input), the recomputed U and p equal the deposited tables (float tolerance 1e-12 relative), then the same hit lists;
  - their CP186A per-plate profile scored with `corum_within_r`, `string_auprc` and the mAP gives the reference rows; spaCR's CP186A profile is scored with the identical call;
  - the Funk plate A rows of are unchanged when the new metrics are added (no side effect).
- **Tests (CPU):** synthetic genes with planted feature shifts vs NT rows and a set of "zero-TPM" genes without effect → planted genes called, null genes at the expected rate; complexes with planted correlation → within-r above the random-pair control.

## Controls
- **Their own profiles** scored by the same code (reference row; must reproduce their published hits).
- **Shuffled gene labels** (seeds of `ITEM_ops_shared_scoring.md`) for r, AUPRC and hits.
- **Random pairs** for within-complex r; **zero-TPM genes** as the hit null (by construction); **NT pseudo-genes** for hits.

## Catalogue and locales
New documented callables (`call_hits`, the two metric functions) ship with arrival contracts and 9-locale records in the same commit; each setting has a tooltip.

## Literature
- Ramezani M et al. (2025) Nat Methods 22:621 (hit rule, CORUM r).
- Kalinin AA et al. (2025) Nat Commun 16:5181 (copairs mAP; via `ITEM_ops_shared_scoring.md`).
- Tsitsiridis G et al. (2023) CORUM 4.0, Nucleic Acids Res 51:D539; Szklarczyk D et al. (2021) STRING v11.5, Nucleic Acids Res 49:D605.
- DepMap / CCLE: Ghandi M et al. (2019) Nature 569:503 (expression; DepMap Public 21Q4).
- Mann HB, Whitney DR (1947) Ann Math Stat 18:50.

## Effort
- **Effort:** S–M (three metrics and one hit caller in an existing module; most time is pinning the notebook's exact rule).
