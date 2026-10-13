# German runtime catalog refresh — 2026-09-21

The current German runtime catalog passes the complete runtime audit for
982 settings, 194 categories, 3,796 UI strings and 68 module summaries.
The English source manifest was refreshed from the running source extractor.

The local OPUS English–German model generated 311 distinct repair candidates
on CPU, with CUDA disabled, four threads and a 6-GiB hard memory limit.
Every changed row was compared with its English source. Technical review
rewrote 240 distinct candidates, including scientific terminology, filenames,
undo/default semantics, time labels and translation of “crop” as image crops.
This is source-by-source agent review, not a claim of native-speaker approval.

The resulting 314 table/key records are saved in
`reviewed/runtime/de/2026-09-21-runtime-refresh.json`. Three source strings
occur in both category and UI tables. Each record stores its exact English
source, SHA-256 and accepted translation; the normal reviewed-record loader,
format-field, protected-token, script and semantic gates remain unchanged.
All 3,826 current runtime review records across the nine languages still bind
to current sources. The German full runtime audit passes independently of
other languages.

Swedish and French count assertions also now account explicitly for the eight
Make Masks readouts added by the preceding tutorial checkpoint. Their new
source sets do not overlap earlier records: subtracting them restores the
previous 319 Swedish and 310 French distinct-source counts.

Other locale repairs remain drafts pending review. The complete multilingual
runtime audit, API catalog debt and nested-helper work remain open. No GPU
job was started, and no coverage threshold or audit rule was relaxed.

## Integration boundary

The audit above was measured on `608e5e6bf` plus this refresh. During
integration, `ee6a608e6` added 701 secondary-organelle background switches.
Repeating the audit after that change reports 701 missing German labels and
701 missing tooltips, plus the corresponding English manifest/hash debt.
The reviewed 314-row repair remains valid; the integrated full audit is red.
The extractor currently recognizes dynamic organelle prefixes, but these
switches put the organelle role at the end of the setting key. That adapter
work and the new wording are the next translation slice, not silently
absorbed into the earlier passing result.
