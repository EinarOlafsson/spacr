# Issue 146: diameter-zero validation source triage

The reported rejection is supported by source: CSV numeric zero remains zero,
and the shared diameter rule rejects it before the pipeline runs. Doctor uses
the same rule. Six declared keys match the generic condition; they do not all
share an agreed automatic-size contract.

Current CPSAM passes actual None as native scale. Legacy Cellpose 3 has a
different default and a dedicated size-model flag that intentionally produces
zero downstream. Diagnostic magnification-derived metadata is separate from
the value passed to the model. A blanket zero-to-None conversion would conflate
these meanings and could affect QC or deprecated settings.

Blank CSV and JSON null already represent None; textual None and the CLI
int-only override boundary expose separate inconsistencies. Existing source
contracts explicitly expect zero rejection. The receipt lists owner decisions
and future scoped cases without bypassing validation or prescribing model tuning.
Root checked the exact issue snapshot and 15 source/test bindings. No CLI,
model, GPU, test, issue comment, source or package change was performed.
Home owns final compatibility semantics and implementation after the freeze.
