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

A second source-bound review adds 73 records: 52 corrected and 21 retained,
bringing the reviewed total to 497 distinct sources. It covers Otsu thresholds,
watershed and object morphology, image downloads, model setup and interface
messages. The existing format/protected-token gates and contextualization
checks pass for every new record. Two duplicate OPS records were already
covered by the first slice. The inversion warning is explicitly deferred:
its English source still describes separate display/detection switches while
item 435 documents unified inversion. No translation can resolve that source
contradiction; the application owner is notified in the shared coordination
file. Evidence: `reviewed/runtime/pt/2026-09-21-runtime-second-slice.json`.
