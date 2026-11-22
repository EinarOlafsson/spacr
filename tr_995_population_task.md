# 995 population diagnostics translation task

Branch: `ws-offload-codex-work-995-population`; base: `007ff7a7074c37f1f0592409de47b895322837d0`.
This stacks the completed 990s6nj report fix on train 10 (`299179d953`). Land the
prerequisite first. Scope follows the owner answer at 09:00 on 2026-11-22:
**diagnostics only; no production correction, refit, tuning or gate change.**

Home owns sv, de, es, zh_CN, pt, hi, ko, is and fr. All exact English bodies,
hashes, source locations, canonical admission and test evidence are in
`tr_995_population_sources.json`. No translation records, catalogs or pins are
authored in this branch.

## Report behaviour and scientific boundary

The new S6 block is labelled **diagnostic, not used for the verdict**. For each
well it compares three explicitly labelled populations against the same fixed
fit: the selected fit mask before finite-positive filtering and trimming, the
actual trimmed values passed to the fitter, and the production scored population.
Rows include selected and finite counts, actual finite DNA minimum and maximum,
the actual seed-relative trim bounds, R² and G2/G1. Counts remain exact integers.
Pooled fallback fits report their actual pooled input rather than an inferred
local-well population. Missing legacy fit-mask or trim records say **not retained**;
their sizes and scores remain unavailable. The audit never reconstructs a seed,
refits a model or changes a production verdict.

The existing all-finite S6 population and 0.90 criterion remain unchanged.
`self_checks`, `_fit_quality`, `failed_stages` and `final_stages` are exact source
matches to the base. Fitter arithmetic ASTs are identical after removing only
the three diagnostic capture branches and the new optional sink. Source-bound
legacy tests verify fitted fields, output frames and CSV bytes exactly, including
the production S6 CSV. No settings, schema or production cell-stage values change.

Capture defaults to off in the shared Measure functions. OPS enables it for the
report. It retains two float64 population copies, up to roughly 90 MB for a
5.5-million-cell fit; pooled records share the same arrays across fallback wells.
No raw images, published per-cell inputs or paper checkpoints are loaded for
this task. Synthetic tests establish implementation behaviour and do not identify
the physical cause of the CP186A fit failure.

## Runtime source locations

The shipped scanner admits all 21 new call locations.
The canonical runtime inventory gains 18 identities;
existing shared identities are reused. None is removed. Canonical API text is
unchanged: all 14,961 bodies equal the exact base extraction.

- `spacr/ops_cell_cycle.py:1290` — `_report_html`: Fit mask
- `spacr/ops_cell_cycle.py:1290` — `_report_html`: Trimmed fitting range
- `spacr/ops_cell_cycle.py:1291` — `_report_html`: Production scored population
- `spacr/ops_cell_cycle.py:1293` — `_report_html`: Well
- `spacr/ops_cell_cycle.py:1293` — `_report_html`: Pooled wells
- `spacr/ops_cell_cycle.py:1295` — `_report_html`: available
- `spacr/ops_cell_cycle.py:1295` — `_report_html`: not retained
- `spacr/ops_cell_cycle.py:1297` — `_report_html`: Well
- `spacr/ops_cell_cycle.py:1297` — `_report_html`: Population
- `spacr/ops_cell_cycle.py:1297` — `_report_html`: Fit scope
- `spacr/ops_cell_cycle.py:1298` — `_report_html`: Selected cells
- `spacr/ops_cell_cycle.py:1298` — `_report_html`: Finite cells
- `spacr/ops_cell_cycle.py:1299` — `_report_html`: Minimum DNA
- `spacr/ops_cell_cycle.py:1299` — `_report_html`: Maximum DNA
- `spacr/ops_cell_cycle.py:1300` — `_report_html`: Fit R2
- `spacr/ops_cell_cycle.py:1300` — `_report_html`: G2/G1 ratio
- `spacr/ops_cell_cycle.py:1300` — `_report_html`: Availability
- `spacr/ops_cell_cycle.py:1301` — `_report_html`: Trim lower bound
- `spacr/ops_cell_cycle.py:1301` — `_report_html`: Trim upper bound
- `spacr/ops_cell_cycle.py:1303` — `_report_html`: S6 populations: diagnostic, not used for the verdict
- `spacr/ops_cell_cycle.py:1304` — `_report_html`: Each row scores the same saved fit on the labelled population; no refit is performed. R^2 uses 100 bins between the finite population 0.5th and 99.5th percentiles, with the existing finite-cell density scaling. The production verdict uses all finite scored cells.

## Changed and new documentation

- `spacr/measure.py:7229` — `_fit_dna_content` (private).
- `spacr/measure.py:7560` — `_field_centred_dna_phases` (private).
- `spacr/measure.py:7676` — `_cell_table_phases_measure_fc` (private).
- `spacr/ops_cell_cycle.py:590` — `_dna_phases` (private).
- `spacr/ops_cell_cycle.py:625` — `_s6_population_diagnostics` (private).
- `spacr/ops_cell_cycle.py:1258` — `_report_html` (private).
- `tests/test_ops_s6_population_diagnostics_995.py:1` — `<module>` (test).
- `tests/test_ops_s6_population_diagnostics_995.py:20` — `_legacy_pipeline` (test).
- `tests/test_ops_s6_population_diagnostics_995.py:29` — `_inputs` (test).
- `tests/test_ops_s6_population_diagnostics_995.py:50` — `test_optional_capture_preserves_the_source_bound_legacy_pipeline_exactly` (test).
- `tests/test_ops_s6_population_diagnostics_995.py:79` — `test_pooled_fallback_records_the_actual_pooled_fit_population` (test).
- `tests/test_ops_s6_population_diagnostics_995.py:94` — `test_diagnostics_and_report_keep_production_s6_bytes_without_refitting` (test).
- `tests/test_ops_s6_population_diagnostics_995.py:109` — `test_diagnostics_and_report_keep_production_s6_bytes_without_refitting.forbid_refit` (test).
- `tests/test_ops_s6_population_diagnostics_995.py:134` — `test_legacy_fits_report_unavailable_populations_without_guessing` (test).
- `tests/test_ops_s6_population_diagnostics_995.py:151` — `test_diagnostic_report_retains_exact_large_population_sizes` (test).
- `tests/test_cell_cycle_phase_three_ways.py:412` — `test_the_torch_route_trains_with_classify_and_reuses_its_model` (test).

These are private and test bodies, separate from canonical API text. The legacy
code fixture is a hash-bound copy of the pre-change shared fitter and field
pipeline, for numerical comparison; it introduces no translated API source.

## Validation

| Group | Passed | Skipped |
| --- | ---: | ---: |
| core | 20 | 0 |
| shared-hidden | 50 | 1 |
| ops-hidden | 30 | 0 |
| source | 157 | 0 |
| shared-visible | 50 | 1 |

The same single dataset-dependent test skips in both visibility runs: the local
plate1 database lacks `nucleus_channel_0_integrated_intensity`. The new core tests
cover default and captured legacy byte identity, exact selected/trimmed inputs,
pooled fallback, no refitting during reporting, missing provenance and unrounded
population counts. Existing shared DNA, OPS end-to-end and 990 not-judged report
tests pass. All six mandatory source ratchets pass.

The visible run exposed an inherited CPU-only Torch fixture that changed CUDA
visibility without forcing the cached accelerator selector to CPU. The identical
failure was reproduced on the unchanged prerequisite (49 pass, one fail and the
same dataset skip). The bounded fixture correction sets `SPACR_DEVICE=cpu` and
uses the existing `_reset_device_cache` helper; all 105 assertions in its file
remain identical. Both final visibility runs pass. No production accelerator or
inference code changes are included. Baseline and initial failure logs are retained.

Runs use scheduler P2, nice 19, two workers and 12 GB. GPU-touching consumer files
run with CUDA hidden and with CUDA visible under a 2 GB VRAM allocation.
Machine-local logs, runtime snapshots and proof JSON:
`/home/carruthers/Documents/repo/spacr-codex-recovery-20261121/995-population/`.
