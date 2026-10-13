# Portuguese runtime recovery, 2026-09-21

All 308 distinct English sources in the preserved 311-record draft have been
reviewed. Of these, 306 are applied: 236 corrected translations and 70 retained
translations. Three duplicate OPS records reuse their first review. Two
inversion messages remain deferred because the live English descriptions
contradict one another. Three embedding action labels were also reviewed.
Portuguese now has 650 distinct source-bound reviewed strings.

Corrections preserve Cellpose flow thresholds, model selection, OPS sequencing
reads and guide-library columns, Otsu thresholds and local windows, watershed,
object morphology, measurement arrays and filenames. The erosion warning now
retains “twice the distance”; the local-window help correctly identifies even
numbers; “Train data” describes training data, not railway trains. Installer
commands, credentials, report-sharing behavior and download placeholders retain
their original meanings and protected syntax.

A narrow normalization correction recognizes the exact installer-closing
message as a GUI screen. Before this change it rewrote the correct Portuguese
“tela” to “triagem” (screening). A regression check exercises normalization
without reviewed overrides and preserves the distinct CRISPR-screening sense.
The existing syntax, placeholders, protected-token and semantic gates remain.

Evidence is in the four `reviewed/runtime/pt/2026-09-21-runtime-*-slice.json`
files and `2026-09-21-action-labels.json`. The review loader, contextualization
checks and complete Portuguese runtime audit were run. This is AI-assisted
technical review, not native-speaker approval. Forty-two obsolete catalog keys
were removed during the first slice.

The deferred messages are the separate display/detection inversion warning
and the inversion tooltip's claim that hover values remain original, which
contradicts the current status message. Item 435 and the shared coordination
file notify the application owner. These sources require reconciliation and
then translation rebinding; the translation pass does not alter detection.

Newer runtime text and API translations remain open under item 411. The full
locale audit still reports that debt. No baseline or audit rule was relaxed,
and no GPU translation job was started.
