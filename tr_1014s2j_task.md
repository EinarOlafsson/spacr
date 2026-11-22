# 1014s2j translation and numerical confirmation task

Branch: `ws-offload-codex-work-1014s2j`; base: `299179d9530ac618d82f11cc823f3ad8abd1c001` (train 10).
Home owns the nine-language records: sv, de, es, zh_CN, pt, hi, ko, is, fr.
All exact English bodies, old/new hashes, canonical API blocks and layouts are in
`tr_1014s2j_sources.json`. No translations or generated catalogs are included here.

## Changed documentation

One public body changes: `spacr.ops_sbs.never_barcodes`, nine canonical blocks.
Its description now states that the builder completes each sorted guide before the
next guide, using at most 200 total attempts per guide. Its signature and defaults
are unchanged. The canonical API source count stays 14,961. The private builder
docstring and new test docstrings are inventoried separately below. Runtime
translation call ASTs are identical to the base; no report captions change.

- `spacr/ops_sbs.py:875` — `_never_barcodes_cached` (private).
- `spacr/ops_sbs.py:931` — `never_barcodes` (canonical API).
- `tests/test_ops_never_s2j_parity_1014.py:1` — `<module>` (test).
- `tests/test_ops_never_s2j_parity_1014.py:19` — `_s2j_change` (test).
- `tests/test_ops_never_s2j_parity_1014.py:36` — `_current_s2j_records` (test).
- `tests/test_ops_never_s2j_parity_1014.py:50` — `_current_s2j_api` (test).
- `tests/test_ops_never_s2j_parity_1014.py:61` — `test_synthetic_acceptance_order_matches_the_paper_builder` (test).
- `tests/test_ops_never_s2j_parity_1014.py:70` — `test_distance_two_keeps_empty_results_for_guides_too_short_to_mutate` (test).
- `tests/test_ops_never_s2j_parity_1014.py:80` — `test_s2j_doc_revision_composes_without_altering_unrelated_api_snapshots` (test).
- `tests/test_ops_never_s2j_parity_1014.py:102` — `s2j_data` (test).
- `tests/test_ops_never_s2j_parity_1014.py:129` — `_barcode_bytes` (test).
- `tests/test_ops_never_s2j_parity_1014.py:134` — `_set_parity` (test).
- `tests/test_ops_never_s2j_parity_1014.py:156` — `test_full_s2j_never_sets_and_production_decoys_are_exact` (test).
- `tests/test_ops_never_s2j_parity_1014.py:190` — `_reference_estimator` (test).
- `tests/test_ops_never_s2j_parity_1014.py:213` — `test_reference_estimator_and_spacr_reproduce_the_s2j_scale_points` (test).

The historical 1014 arrival/source JSON files remain immutable. A new exact-body
revision and composed adapters preserve all 192 assertions in the existing 1014
test file. The protected API consumer receives only the one line approved by the
owner at 08:07; all 493 of its assertions remain.

## Confirmed reference parity

This confirms and supersedes the earlier 1014 PRELIMINARY status for the two S2J
scale points. The readonly reference is paper commit `999052f`; all 15 declared
reference files and all six declared input files retain their SHA-256 values.
No production barcode sets or per-field inputs are committed. The compact evidence
contains counts, hashes and scalar estimates only. Each passed full-data test
asserts generated row order and generated CSV byte hash against the reference;
the table below reports those passed assertions and their reference hashes.
Funk D retains production decoy acceptance order; PERISCOPE D uses the sorted
order of the production prefix-decoy owner.

| Library | Set | Rows | Exact order and SHA-256 | SHA-256 |
| --- | --- | ---: | --- | --- |
| funk_full | D | 81780 | yes | `139e2c422a725f08ffe94355dd14763c1c0177de4810851e091c439843f7f9ac` |
| funk_full | N1 | 81764 | yes | `1ecd5be96684732236908b6e57d49db8bc344f4cede8c474d10e3d28d8af3094` |
| funk_full | N2 | 81780 | yes | `a0deefb0f06a6cbed457afe596e1263245ab06b5516ca942732771577c9df9a6` |
| peri_full | D | 323448 | yes | `5dceaece771738622e1b02bc3166305b3fd951a08e68597eb5af1fac4905c951` |
| peri_full | N1 | 323318 | yes | `e0616993f39ed30927be7809b3edcfd9dcb4613b0b7a6ad931439e45b8a4c158` |
| peri_full | N2 | 323448 | yes | `26de7be59405080bc8d9e0dd6100ddac4f73d3c0e5727b4474faf6b436fd1d7b` |

For b=1.0, the exact AST-extracted `post.est` and existing spaCR `never_scale`
both return **1.3556586979213574** for Funk (333 fields) and
**1.96468558629205** for PERISCOPE (85 fields), absolute difference zero.
These are the N2-only point estimates, k=4 and ci=False. The top-level reference
scripts are never run and no outside paths are followed. The optional raw
PERISCOPE barcode table was not transferred; the explicitly approved production
key export is used. The reference folder inventory is checked before and after
the full-data tests.

The original builder differed at index 1 because it visited guides round by round.
For a three-guide synthetic example, N1 was `GGTGCAAATGC` versus reference
`GATGGCGTGAC`; N2 was `GATGCAAAAAC` versus `GCTGGCGTTCC`. Completing one guide
at a time with the reference draw sequence fixes both. Only the existing builder
changes numerically; production decoys, estimator and decoder remain unchanged.
The previous empty N2 result for guides shorter than two bases is preserved.

## Validation

- Full reference, builder and decoder-consumer tests: **105 passed**, no skips,
  386.25 seconds, train 10 base above.
- Six mandatory source ratchets: **157 passed**.
- Three native API gates: **3 passed**, zero production-module imports.
- Exact public API delta: one revised body, nine blocks, no added or removed keys.
- CUDA hidden, nice 19, scheduler P1, two workers, 16 GB for real parity.

Machine-local logs and integrity receipts:
`/home/carruthers/Documents/repo/spacr-codex-recovery-20261121/1014s2j/`.
