# French runtime recovery — 21 September 2026

The French runtime audit passed at the 46d11afaf application-source boundary:
985 settings, 194 categories, 3,824 UI sources and 68 module summaries.
Later nightly source additions require their own audit and translation pass.

Four source-bound review files add 308 distinct scientific and UI sources.
Of the preserved machine drafts, 244 were corrected and 64 retained after
technical review. Three OPS descriptions shared by categories and UI are
applied consistently from the same reviewed source. Five additional action
labels make the displayed controls agree with their help text. There are now
670 distinct reviewed French runtime sources; the previous 357 are preserved
and counted separately by the existing regression test.

Corrections cover Cellpose models and thresholds, OPS sequencing and barcode
assignment, trajectory filtering, Otsu segmentation, erosion and dilation,
mask measurement, sound controls, installations and public issue reporting.
Python placeholders, commands, identifiers and protected quoted controls are
preserved. Examples of repaired meaning include “training data” translated as
train transport, reversed logging instructions and missing morphology undo
steps. “Embed” now reads “Vectoriser” in both its action and its instructions.

Removed 42 catalog keys absent from the current English sources: two setting
labels, two tooltips, two category descriptions and 36 UI strings. No live
source or translation gate was removed. The four slice files and action-label
file are under `docs/i18n/reviewed/runtime/fr/`; all records carry exact source
hashes. This is AI-assisted technical review, not native-speaker approval.

The other runtime locales and the API catalog debt remain part of item 411.
No API coverage baseline or completeness threshold was relaxed.
