# Item spec: decode rule (a) + S2 as OPS engine settings

- **Lifecycle:** future (alpha).
- **Number:** this is the build of item **842** (`features/feature_list/842_ops_decode_strategies_are_settings.txt`), restricted to the arms listed here. The other 842 arms can follow.
- **Rule:** a `spacr-run ops` settings run must reproduce the reference genotype calls identically. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.
- **Source inventory:** `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/SPACR_PROVENANCE.md` (port P2).

## Why
The decode rule (a) + S2 is not in the spaCR engine.
- It comes from scripts over a `sample_fields` cache, plus `refine_patch.py`, which monkey-patches `spacr.ops_cycles` (MANIFEST.json: "not yet in the engine").
- A separate re-implementation exists in `ops_3way_percell/code/decode_spacr.py`; it does not import spaCR.

## Reference implementation
Frozen source: `/mnt/firecuda2/ops_decode_yield/src_ruleA_S2_20261107/`, revision r3. Copy: `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/decode_yield_2026-11-04/code_archive/src_ruleA_S2_20261107.tar.gz`; the tar predates r3, and `PATCHES_2026-11-07.diff` + MANIFEST.json `revisions` hold r3.
- `scripts/sample_fields.py` (`--refine --pixels 97`): per-field SBS read extraction through `ops_cycles.align_field`, `sbs_mean_target`, `ops_engine._read_plane`, `ops_sbs.find_peaks`, `extract_bases`, `estimate_read_locations`.
- `scripts/refine_patch.py` (`_spot_shifts`, `apply`): S2 = per-channel registration refined on the reads (spots_refine).
- `scripts/decode_rules_well.py` (`brieflow`, `pixels`, `fallback`) and `decode_rulea_well.py`: rule (a) = S1b + S6 + QW primary.
  - read threshold 315;
  - runner-up 1-mismatch margin < 0.1;
  - quality-weighted vote, single-read quality ≥ 0.06.
- `scripts/n3_pixels.py::pixel_calls`, `n3_combo.py`: the N3 pixel fallback (≥ 2 exact pixels, top guide ≥ 0.6, pixel quality ≥ 0.1, pixel percentile 97).
- `scripts/s2_rulea.py`, `finish_plate_final.py`, `validate_wells.py`: decoy FDR with library + 4× decoys, decoys dropped and counted, FDR = 100 × 1.268/4 × decoys/real (`decoy_calibration.json`).
- Skip-unplaced-site rule: decode only the SBS sites present in `ops_geometry`, and log the skipped ones (`plates/plateB_final/skipped_sites.json`).
- Driver: `scripts/run_final.sh` → `mirror_well.py` → `run_manifest_mirror.sh`.

## What to build (hooks as named in 842)
Use 842's setting names. The defaults below reproduce rule (a) + S2:

| Setting | Value of record | Hook |
|---|---|---|
| `ops_channel_registration` | `spots_refine` (companions `ops_registration_refine_px` 3, `ops_spot_min_peak` 20, `ops_spot_overrule_share` 0.5) | `ops_cycles.align_field` (replace the monkey-patch with a real code path) |
| `ops_read_threshold` | 315 | spots |
| `ops_crosstalk` | `argmax_median` | `ops_sbs.called_bases` |
| `ops_decode_library_calibration` | True (passes 2, min labels 30) | base calling |
| `ops_read_mismatches` | 0 | matching |
| `ops_decode_runner_up` / `ops_runner_up_margin` | True / 0.1 | matching |
| `ops_decode_impute_missing_cycles` | True | matching |
| `ops_assign_by_cell_territory` | True (`ops_footprint` 10, `ops_tie` 1.0) | `ops_sbs.attribute_reads` |
| `ops_cell_rule` | `mapped` | `ops_engine._decode` |
| `ops_decode_quality_weighted` / `ops_vote_weight_floor` | True / 0.05 | cell rule |
| `ops_min_fraction` | 0.6 | cell rule |
| `ops_single_read_quality` | 0.06 | cell rule |
| `ops_decode_pixel_fallback` | True (`ops_pixel_percentile` 97, `ops_pixel_radius` 10, `ops_pixel_min_pixels` 2, `ops_pixel_min_fraction` 0.6, `ops_pixel_quality` 0.1) | fallback |
| `ops_decoy_library_k` / `ops_fdr_scale` | 4 / 1.268 | every decode writes `decode_fdr_<well>.json` |
| `ops_decode_skip_unplaced_sites` (new) | True | decode only sites in `ops_geometry`; log skipped sites in the FDR report |

Outputs:
- the existing `ops_barcodes` / genotype tables;
- a per-field stats table with the columns of `*.field_stats_all_rules.csv` (assigned, decoy-assigned, per rule);
- a per-well FDR report.

## Validation data and acceptance
Acceptance: identical output, or within the tolerance stated per check, to the reference outputs at the paths below.
1. **176-field subset** (`/mnt/firecuda2/ops_decode_yield/subset_v1.1.csv` + `subset_v1.1_extra.csv`): assigned fraction and FDR equal to the row "final" of `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/decode_yield_2026-11-04/results_v3/final/master_table_decode_strategies.csv`.
2. **Fields A1/74 and A1/190:** cache arrays and calls bit-identical to `/mnt/firecuda2/ops_decode_yield/repro_check/` (r2 = r3 identity already shown).
3. **Plate A and plate B, full:**
   - per-nucleus genotype calls exact (guide, n_reads, fraction) against `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/decode_yield_2026-11-04/plates/plateA_final/decoded/` and `plateB_final/decoded/`;
   - per-well assigned % and decoy counts exact, and the plate-level assigned % and FDR equal to the per-plate summaries in the same `plates/plate{A,B}_final/` folders;
   - the quality scores atol 1e-9 (CPU deterministic);
   - the skipped sites equal `plates/plateB_final/skipped_sites.json` and are logged.
4. **Tests (CPU):** the 842 ratchet for each setting; a synthetic field where each arm changes the outcome; the decoy-FDR report key; the skip-unplaced-site log.

## Effort
- **Effort:** L.
