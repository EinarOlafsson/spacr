# Timelapse pretrained-video boundary coverage, CPU 2026-10-07

The single final focused run passes 51 cases in 11.80 seconds under 4 GiB:
21 new cases and 30 existing event-backend/provenance cases. The new tests cover
invalid fusion-vector shape/batch/absence, crop and checkpoint-cache integrity,
empty clips, fit routing and persistence, every saved provenance component,
checkpoint/channel selection, unknown encoders, and saved-head/runtime overrides.
They use a real small PyTorch CPU fusion head and deterministic encoder/training/
scoring doubles. This is not real pretrained inference, GPU model acceptance,
a full-suite run, or source-current documentation/API acceptance.

`spacr/timelapse.py` has SHA256
`954c51e8f5903705204d25ccc258070607c488fa4cd30758f4ad041ddf374cc4`;
it is byte-identical to hosted e1f54c80650f60942ba6c98d4e999b36a3fa2846.
The extracted hosted and focused coverage reports have identical universes:
4535 statements and 1616 branch destinations, despite coverage versions
7.16.2 and 7.15.4. Their strict union covers all 21 hosted missing statements
and 17 of 18 missing destinations. The remaining [9323, 9328] guard destination
is inside the unchanged original allowance of one statement and one destination.
No source, exclusion, baseline or ceiling changes were made. The eager worker
builder already has nonempty input; no impossible empty worker structure was
fabricated to eliminate the remaining allowed branch.

`receipt.json` records exact source, test and original full-report hashes;
`focused-tests.log` is the final actual raw run. The coverage payloads retain
only this module, not the full hosted report or repeated function/class maps.
The parent owns the original complete hosted report and the six other repaired
module receipts. This archive does not duplicate or recount those phases.

From the repository root, reproduce the focused command in `receipt.json`
(use any supported spaCR Python environment and a writable scratch JSON target).
Verify the archived evidence without executing tests:

```
tools/run_capped.sh 4G python features/data/43_timelapse_video_boundaries_cpu_2026-10-07/verify_union.py
```

The verifier checks every archived payload, unchanged source against its actual
Git blob, test hashes, the complete unchanged baseline file, equal coverage
universes, and exact union destinations. The intermediate 50-case run is
superseded; it is not added to the final 51-case count.
