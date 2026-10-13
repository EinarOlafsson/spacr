# Simplified Chinese runtime recovery, 2026-09-21

The preserved draft contains 311 records covering 308 distinct English
sources. Review now covers 306 distinct sources: 272 required correction
and 34 were retained. Three duplicate records reuse those reviews. Two
contradictory English inversion descriptions remain deferred to the
application owner, as recorded in items 435 and 325. Simplified Chinese now
has 949 distinct source-bound reviewed strings. This completes review of
the usable preserved draft, not the full runtime or API catalogs.

This slice corrects scientific setting labels and help for Cellpose models
and flow thresholds, OPS image-based sequencing, regression annotations,
tracking and report sharing. It restores omitted defaults, numeric limits,
channel order, overlap constraints and the consequences of changing each
setting. Guide-library columns, model identifiers, filenames, settings keys
and the distinction between sequencing images and FASTQ remain explicit.

The draft's public-library, shopping, railway and stage-performance senses
are corrected to barcode libraries, storing reads, model training and
microscope stage drift. Two untranslated setting descriptions are translated.
The report-sharing descriptions retain both the data sent and the user's
choice of automatic or previewed submission.

The second slice covers OPS setup, embedding inputs, image-table controls,
mask filtering and sound preferences. The Measure table's “Add field” means
adding an image field, not a database column. Music terminology is separated
from image-analysis terminology, and the DEBUG description retains the actual
timing measurements and the distinction from function-call tracing.

Evidence: the eight slice JSON files under
`reviewed/runtime/zh_CN/2026-09-21-runtime-*-slice.json`.
All 306 rows pass the existing source, placeholder, protected-token, language
and terminology validation before application. No gate is relaxed. This is
AI-assisted technical review, not native-speaker approval or full catalog
completion. The remaining catalog audit debt is reported separately.

The later slices restore omitted segmentation controls and consequences:
Gaussian smoothing, Otsu and watershed behavior, minimum area, shrinking
objects, normalization, the live magnifier and model-specific thresholds.
They also distinguish uploads from downloads, training from railway travel,
and image fields from database fields. Model-sharing and installation help
retains consent, credential handling, checksum limitations, and the risk of
changing the installed torch version. File-grid cells remain table cells:
an exact-source normalization correction preserves that meaning while a
regression test retains biological-cell correction in scientific contexts.

Forty-two obsolete catalog keys were removed: two setting labels, two
setting tooltips, two category entries and 36 UI strings. This pruning uses
the current English source inventory and does not remove source-bound review
history. The earlier 96-diagnostic audit predates the second slice and this
pruning; the full catalog remains open.

The post-pruning full Chinese runtime audit exited 1 with 83 diagnostics at
application source 9e2f5335a after the first two slices. Missing entries,
remaining unreviewed drafts and format-field errors are still explicit debt.
Seven focused source-review, scientific-token, Chinese-normalization and
catalog-syntax checks passed before the second slice; all second-slice rows
also passed the same pre-application source and content gates.

After all eight slices, the full Chinese audit exits 1 with 40 diagnostics
at application source eed895c2f plus these reviewed translations. The missing
runtime inventory is now 20 setting labels, 20 tooltips and 110 UI entries;
one protected-literal/hash mismatch, a changed reviewed PDF label and
remaining format-field errors are still explicit. No audit gate was relaxed.
Twenty-eight focused catalog-syntax, terminology and narration checks pass;
the separate data/capture/catalog-preservation checks had 39 passes and one
stale mastering expectation, subsequently corrected and verified.

After that draft checkpoint, 33 additional source-bound UI repairs restore
format fields in download, plaque, annotation and sign-in messages. These
include numeric format specifiers such as `{area:.0f}`. All candidates pass
the existing syntax/content gates with their original placeholder order.
The reviewed Chinese total is now 982; the 949 count above describes only
the completed preserved-draft checkpoint. Evidence:
`reviewed/runtime/zh_CN/2026-09-21-format-repairs.json`.

Two further current-source reviews cover “Updating {name}…” and the
plaque_model tooltip, restoring the actual toxoplasma_plaque_v2 default,
cpsam_plaque_r5/v5 provenance, and Cellpose 3 requirement for the historical
bundled model. The reviewed total is 984. The PDF label uses the builder's
existing approved identity directly; no identical-English review exception
was added. All review-loader gates remain unchanged.

The final Chinese runtime audit at application source 73063d7b3 plus these
reviews has five diagnostics: 20 missing labels, 20 missing tooltips, 98
missing UI entries, their 138 missing source hashes, and 20 fallback tooltips
without Chinese text. No format-field, wrong-hash, protected-literal or
reviewed-target mismatch remains in this audit. Current runtime coverage is
6094/6232 with zero orphan keys. The remaining missing rows and API catalogs
keep the full translation gate open.
