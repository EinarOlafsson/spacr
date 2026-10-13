# Swedish settings and variable-count dataset captions

This checkpoint adds 39 reviewed Swedish scientific-setting records, correcting
35 generated translations. Corrections preserve equations, threshold semantics,
model names, genomic annotation identifiers, credential redaction, OPS metadata
and data-export warnings. The remaining Swedish machine drafts are excluded.

The dataset-sample update in `715687246` changed three English captions. Three
obsolete records were removed from the German runtime refresh (runtime records
have no retirement flag), and 27 replacement records cover all nine languages.
The download caption retains ordered `{count}` and `{name}` fields. Descriptions
now describe a sample without assuming every model provides ten fields.

The source-bound runtime syntax and review-reporter checks pass: 10 tests in
136.09 seconds. German's complete runtime catalog audit also passes, covering
985 setting tooltips, 194 categories, 3796 UI strings and 68 modules at the
measured source revision. Broader API and remaining runtime translation debt
is still open. No audit gate or coverage baseline was relaxed.

Integration note: rebasing onto `7c4ac005a` added seven UI sources for assay
test-data downloads and plate templates. The English manifest is refreshed;
German now reports seven missing rows (including the new `{detail}` failure
caption). This is additional translation debt, not a regression in the 66
reviewed records above. The earlier green full audit applies before those
incoming source changes. Source-binding checks are rerun on the integrated tree.
