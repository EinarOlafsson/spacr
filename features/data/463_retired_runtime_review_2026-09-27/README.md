# Item 463 source replacement, 2026-09-27

These nine files preserve the complete previous review inputs byte for byte.
Only the tooltip record with source SHA256
`507d4d3281c7bca17f60d89a601d9e36305c25bb0ef5435258a9553c62b7c39e`
was retired from the active `docs/i18n/reviewed/runtime/<locale>/2026-09-21-synthetic-invasion.json`
files. Their still-current download-progress caption remains active.

The old tooltip described drawn masks. Item 463 now publishes Cellpose-SAM
segmented masks of the same synthetic images, with drawn-object ground truth
kept separately. Reusing the old translation with a new source hash would
misrepresent that change. New translations of the actual current source are
being drafted and technically reviewed by Codex in the runtime delta pass;
this retirement does not claim that those translations are already installed.

Publication and content verification are recorded in
`../463_segmented_publication_2026-09-27.json`. New source:

> Download about {size} MB of SYNTHETIC test data: two-colour images drawn by spaCR, segmented by Mask with Cellpose-SAM, then measured by Measure. No real cells were imaged. A staining-control column and two conditions have a known share of invaded parasites; ground truth records the drawn objects, not the segmented masks. Settings are filled in, so Run is the next step. Cached after the first download.

Review classification: AI technical review (Codex), no native-speaker signoff.
