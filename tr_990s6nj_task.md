# 990s6nj translation task

Branch: `ws-offload-codex-work-990s6nj`; base: `299179d9530ac618d82f11cc823f3ad8abd1c001`.
Home owns sv, de, es, zh_CN, pt, hi, ko, is and fr. Exact English bodies,
hashes, admission evidence and synthetic results are in `tr_990s6nj_sources.json`.

## Runtime source

`spacr/ops_cell_cycle.py:1235` in `_report_html`, through `tr()`:

    not judged (S6 R^2 {r2} < {threshold})

Preserve both placeholders. The values are formatted to two decimals before
translation. The shipped static UI scanner admits this one new source. No runtime
source is removed. Canonical API bodies are unchanged: the full dictionary of
14,961 sources equals the exact extraction of the base.

A failed finite S6 R² below 0.90 adds the reason to its report row and to three
explicit paragraphs for G1, S and G2. The input checks, their boolean verdicts,
counts, collapse decisions and other check rows remain unchanged. A ratio-only
failure retains its existing explanation. A good fit retains the normal report.
The acceptance threshold remains exactly 0.90, including the rounding boundary.

## Other changed or new documentation

- `spacr/ops_cell_cycle.py:1217` — `_report_html` (private).
- `tests/test_ops_cell_cycle_s6_not_judged_990.py:1` — `<module>` (test).
- `tests/test_ops_cell_cycle_s6_not_judged_990.py:15` — `_well_checks` (test).
- `tests/test_ops_cell_cycle_s6_not_judged_990.py:34` — `test_s6_report_explains_unjudged_phases_and_preserves_other_verdicts` (test).
- `tests/test_ops_cell_cycle_s6_not_judged_990.py:67` — `test_ratio_only_s6_failure_keeps_the_existing_report_verdict` (test).
- `tests/test_ops_cell_cycle_s6_not_judged_990.py:80` — `test_report_rounding_does_not_move_the_s6_acceptance_boundary` (test).

These private and test bodies are inventoried separately from the canonical API.
Only `_report_html` changes in production; every other top-level source AST in
`ops_cell_cycle.py` is identical to the base. No settings or classifier changes
are included. No translation records, catalogs or pins are authored here.

## Validation

- **65 passed**: five new report cases and existing cell-cycle, queue and follow-up
  consumers; no skips.
- **157 passed**: all six mandatory source ratchets; no skips.
- Full canonical API dictionary equality and actual runtime scanner admission.
- Actual synthetic histogram scores: poor R² **-414.020968163**, failed;
  good R² **0.995782203485**, passed. Both use a fixed known two-peak mixture;
  these are test fixtures and make no biological accuracy claim.
- Report examples and logs are under
  `/home/carruthers/Documents/repo/spacr-codex-recovery-20261121/990s6nj/`.
- CPU only, CUDA hidden, nice 19, scheduler P2, two pytest workers, 12 GB.
