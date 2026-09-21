# Simplified Chinese runtime recovery, 2026-09-21

The preserved draft contains 311 records covering 308 distinct English
sources. The first 80 records are now reviewed and applied against their
current source hashes: 76 required correction and four were retained.
Simplified Chinese now has 723 distinct source-bound reviewed strings.
The remaining draft is open.

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

Evidence: the first and second slice JSON files under
`reviewed/runtime/zh_CN/2026-09-21-runtime-*-slice.json`.
All 80 rows pass the existing source, placeholder, protected-token, language
and terminology validation before application. No gate is relaxed. This is
AI-assisted technical review, not native-speaker approval or full catalog
completion. The remaining catalog audit debt is reported separately.

Forty-two obsolete catalog keys were removed: two setting labels, two
setting tooltips, two category entries and 36 UI strings. This pruning uses
the current English source inventory and does not remove source-bound review
history. The earlier 96-diagnostic audit predates the second slice and this
pruning; the full catalog remains open.
