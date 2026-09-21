# Portuguese runtime recovery, 2026-09-21

The first 80 records from the preserved draft were compared with their full
English sources: 71 were corrected and nine retained. Three related action
labels were also reviewed so the embedding help and its buttons consistently
say “Vetorizar”. Portuguese now has 424 distinct source-bound reviewed strings.

The corrections preserve the scientific meaning of Cellpose flow thresholds,
model/backend selection, track filtering and smoothing, OPS sequencing reads,
guide-library column names, raster overlap, Otsu thresholds and hole filling.
They also restore omitted logging instructions and accurately describe what
error reports send outside the machine. Paths, parameters, format fields,
markup and protected names remain checked by the existing gates.

Evidence is in `reviewed/runtime/pt/2026-09-21-runtime-first-slice.json` and
`reviewed/runtime/pt/2026-09-21-action-labels.json`. The review loader and the
catalog audit were run after applying the records. This is AI-assisted
technical review, not native-speaker approval.

This is a partial recovery. The rest of the 311-record draft, newer live
application captions and API translations remain open under item 411. The
full Portuguese runtime audit still fails on the remaining catalog debt.
No audit rule or baseline was relaxed, and no GPU translation job was started.
