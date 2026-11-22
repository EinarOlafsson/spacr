# Item spec: field centring and an external mitotic call for Measure's DNA-content cell-cycle fit

- **Lifecycle:** future (alpha), behind "Show alpha features"; the default (off) leaves Measure unchanged.
- **Status:** implemented 2026-11-20 on branch `claude-995-impl` (feature file `features/feature_list/995_measure_dna_field_centring.txt`). Identity (a): 5,532,572 cells, 0 differ, md5 bece7f3e5a97f78fb346727540c365ff equals the frozen reference. Proof: `BMCspacr/reanalysis/item995_proof/`. The reference files are archived in `/nas_mnt/data/bmc_reanalysis/cellcycle_validation_1119_reference/`.
- **Number:** take the next free item number. Check `features/feature_list/` and the claims files first.
- **Rule:** a spaCR settings-file run must reproduce the reference calls IDENTICALLY: the same G1 / S / G2 / subG1 / >4N / M label for every nucleus. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly (written against f04495e887).
- **Paper-critical:** yes. Maintainer decision 2026-11-20: Funk et al. plate A G1/S/G2 calls use this path.
- **Readiness:** "Per-cell cell-cycle phase classifier" stays Under development until the identity acceptance below passes. Until then, phase shares are used only as relative shifts against NT.
- **Related:**
  - `ITEM_ops_cell_cycle_profiles.md`: the OPS aggregate uses `gene_measurement_compare.cell_cycle_phases`, a G1 vs S/G2 split at a fixed 0.30 log2.
  - Item 856 (`spacr.ops_mitotic`): the mitotic call.
  - Q2 (2a696122ea): population normalisation.

## Why
`spacr.measure._fit_dna_content` (a Dean-Jett-Fox EM: G1 and G2 Gaussians with a shared CV, uniform S, a flat outlier term) and `_gate_dna_content` fit one histogram per plate (and time point).

On Funk et al. plate A (HeLa, 5.53 M nuclei), DAPI scaling differs between fields: the SD of the field G1 mode is 0.04–0.05 log2 in A1–A2 and 0.09–0.13 in A3 and B1–B3. A pooled, uncentred fit therefore broadens both peaks:
- Measure as it runs today puts G2/G1 at its 2.4 bound and gives 0 % G2;
- it agrees with the reference calls for 82.9 % of cells.

The same EM on field-centred DNA:
- fits every well with G2/G1 1.945–1.955 and agrees 95.4 % with the Design H least-squares fit;
- on simulated mixtures, its fitted weights are unbiased (≤ 0.4 point) and its per-cell calls are less biased than the Design H argmax calls (G2 +0.1 point vs +2.7 points).

The maintainer adopted this path.

## Reference implementation (identity target)
- Script: `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/cellcycle_validation_1119/scripts/cc_measure_fc.py` (functions `field_offsets`, `call_well`).
- Frozen reference calls: `/mnt/wd4tb/scratch_cellcycle_1119/reference_measfc_calls.parquet` (well, site, label, phase_measfc), with an md5 of the joined labels in `.../cellcycle_validation_1119/tables/e5c_measfc.json`.
- Copy both to the NAS archive before the scratch folder is cleaned.

## What to build
1. **Field centring** in `_phases_by_measurements` (setting `cell_cycle_field_centre`, default False):
   - x = log2(max(background-subtracted DNA content, 1)).
   - For every field (`plateID, rowID, columnID, fieldID`; `well, site` in OPS tables) with at least `cell_cycle_field_min_cells` (default 50) non-mitotic nuclei:
     - m_f = `spacr.gene_measurement_compare.g1_mode(x of those nuclei)` (120 bins between the 0.5th and 99.5th percentiles, Gaussian smoothing 2 bins);
     - offset_f = m_f − the median of m_f over the fields of the same fit group.
   - Fields with fewer cells get offset 0.
   - centred content = 2 ** (x − offset_f).
   - Write `field_offset_log2` and `dna_content_centred` per nucleus.
2. **Fit group** (setting `cell_cycle_fit_group`: `'plate'` (current) or `'well'`): one `_fit_dna_content` per group, on that group's non-mitotic nuclei only.
3. **External mitotic call** (setting `cell_cycle_mitotic_column`, default empty):
   - a boolean column of the nucleus/cell table (for example item 856's `predict_mitotic` output);
   - these nuclei are left out of the field modes and the fit, and are called `M`;
   - the condensation rule (`cell_cycle_mitotic_ratio`) is not applied when this column is given.
4. **Per-cell call rule (unchanged functions):**
   - `_gate_dna_content(centred content, fit)` with the fitted crossing gates (or `cell_cycle_gates`) gives G1 / S / G2;
   - content outside 4 peak SDs gives subG1 / >4N;
   - M from item 3 overrides.
   - The confidence and posteriors stay as now.
5. **OPS:** the aggregate phase gets `ops_profile_cell_cycle_method: 'measure_fc'`, which calls items 1–4 on the cell tables (`_dna_integrated`, `well`, `site`, the 856 call), replacing `cell_cycle_phases` when set.
6. **Outputs:** the existing `cell_cycle` / `cell_cycle_well` tables, plus `field_offset_log2`, and per fit group the fitted g1, g2, CV, weights, gates and the number of fields shifted.

## Settings (alpha)
| Key | Default | Meaning |
|---|---|---|
| `cell_cycle_field_centre` | False | centre every field on its G1 mode before the fit |
| `cell_cycle_field_min_cells` | 50 | non-mitotic nuclei a field needs to be shifted |
| `cell_cycle_fit_group` | `'plate'` | `'plate'` or `'well'` |
| `cell_cycle_mitotic_column` | `''` | boolean external mitotic call; empty keeps the condensation rule |
| `ops_profile_cell_cycle_method` | `''` | `'measure_fc'` uses this path in the OPS aggregate |

## Validation data and acceptance
- **Data:** Funk et al. plate A, 6 wells, 5,532,572 cells.
  - DNA: `analysis_plateA_flatv2/cells/raw_<well>.parquet` `nucleus_dapi_integrated_intensity`.
  - Mitotic call: the deployed 856 model `funk_plateA_full/models/mitotic_classifier_plateA.joblib` on the `spacr_equiv_plateA` features. Its calls equal `m856` in `/mnt/wd4tb/scratch_cellcycle_1119/cells_phases.parquet`.
- **Acceptance:**
  - (a) Identity: with `cell_cycle_field_centre=True`, `cell_cycle_fit_group='well'` and `cell_cycle_mitotic_column=<856 call>`, every cell's label equals the frozen reference. Report the md5 of the joined labels; zero cells may differ.
  - (b) Off by default: with `cell_cycle_field_centre=False`, the outputs equal current nightly byte for byte on the existing cell-cycle tests.
  - (c) The settings-file rerun is identical.
- **Tests (CPU, synthetic, small):**
  - Fields with planted log2 offsets (SD 0.10) and known G1/S/G2 shares: the centred fit recovers G2/G1 within 0.05 of 2.0, and the fitted weights within 1 point of truth at n = 100,000.
  - Without centring, the same data broadens the peaks; assert that the CV is larger.
  - A field below `cell_cycle_field_min_cells` gets offset 0.
  - Mitotic rows are excluded from the modes and the fit and are called M.
  - Every field with the same offset gives the same calls as no centring.
  - `fit_group='well'` fits once per well.

## Known limits (reported, not blockers)
- The per-cell calls of any two-peak model are biased by peak overlap: cells in late S are called G2 and early-S cells G1. Share estimates should use the fitted weights. Per-gene comparisons are relative shifts against NT, made with the same calls.
- HeLa (HPV18 E6/E7) lacks functional RB/p53, so the G1-arrest controls are weak by biology.
