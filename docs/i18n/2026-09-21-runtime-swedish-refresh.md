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

Integration note: `1eadae49e` subsequently adds five preview-refresh/plaque
captions. The English manifest includes them; their nine-language translations
remain follow-on debt. The complete green audits above were measured before
that source addition. No existing reviewed source was renamed by this update.

Preview follow-on: the five captions from `1eadae49e` now have 45 reviewed
records across all nine runtime languages in `2026-09-21-preview-refresh.json`.
The full German and Swedish runtime audits pass with 3808 UI strings each.
All ten selected syntax/review-reporter checks pass (257.45 seconds). The
single warning records another process changing real QSettings while the
test process itself remained correctly isolated.

Integration of `85a3ae1cc` adds five Make Masks normalization captions. The
English manifest is refreshed to 3813 UI sources; translations for these five
remain open. The full German/Swedish green result above applies to the prior
3808-source manifest. No earlier reviewed runtime source was renamed in this
incoming change.

The five normalization sources now have 45 reviewed translations across all
nine languages. The full German/Swedish runtime audits pass at 3813 UI
sources, and all ten selected source/syntax/reporter tests pass (204.52 s).
This verification precedes the Import sources in `e4fd5f31e`.

After integration of `e4fd5f31e`, all 4467 reviewed runtime records retain
their exact current source and hash bindings. English now has 3822 UI sources;
nine new Import captions still need translations. The preceding full de/sv
audit remains evidence for the earlier 3813-source snapshot.
