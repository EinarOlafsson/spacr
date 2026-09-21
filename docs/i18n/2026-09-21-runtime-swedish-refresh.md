# Swedish runtime refresh and new assay captions

The Swedish follow-on adds 272 reviewed UI/category records representing 269
distinct current English sources. Three OPS descriptions occur in both source
tables and share the same translation. The review corrected 203 source-level
machine drafts, including lost object-size warnings, incorrect threshold
semantics, translated model identifiers and paths, and misleading UI actions.
The other 66 source translations were read and retained.

Three superseded ten-field captions in the draft inventory were excluded;
their variable-count replacements were already reviewed in the preceding
checkpoint. Forty-two obsolete Swedish catalog keys were removed: two labels,
two tooltips, two category descriptions and 36 UI strings. None exists in the
current English inventory. No current source or review requirement was removed.

The seven new assay-download and plate-template captions introduced by
`c0558cabf` and `7c4ac005a` now have 63 reviewed, source-bound records across all
nine runtime languages. Quantities, dataset identifiers and `{detail}` are
preserved. This closes the seven-row integration debt recorded in the preceding
checkpoint.

The complete German and Swedish runtime audits pass at source revision
`e29ac7783`: 985 setting tooltips, 194 categories, 3803 UI strings and 68 modules
per language. Review-count tests retain earlier counts by subtracting the exact
new source sets; they check every new record against its current source and
unchanged translation gates. This does not claim native-speaker review.

Other runtime languages and the API catalogs still have work remaining. Their
generated drafts are excluded from this checkpoint except for the 63 reviewed
captions described above. No translation gate or coverage baseline was relaxed.

Validation: the two finalized Swedish/French review checks pass (63.91 seconds).
The other eight selected syntax/reporter checks passed in the preceding run;
that run loaded the earlier count assertions and correctly rejected the two
new seven-source additions before their explicit count updates were applied.
