# Import and synthetic Invasion captions — 21 September 2026

All nine runtime languages now have source-bound reviewed translations of the
nine Import captions and two synthetic Invasion captions: 99 records total.
Spanish's eleven records were published in `52d0c03f9`; this checkpoint adds
the other 88. Only those UI entries and their hashes changed in the remaining
eight catalogs. No model inference was used for these reviewed captions.

The synthetic-data notice retains its distinction between fields drawn by
spaCR and fields acquired by imaging or produced by segmentation. Named regex
groups, placeholders, microscope formats and Format Converter remain intact.

All new records pass the ordinary review loader's source/hash, syntax,
semantic and target-language gates. Full German and Swedish runtime audits
pass at 985 settings, 194 categories, 3824 UI sources and 68 modules, matching
the full Spanish audit at the same source boundary. The existing source tests
pass (three tests); their historical count arithmetic remains intact by
subtracting the eleven newly evidenced sources. Swedish now has 674 distinct
reviewed sources and French has 357. This does not assert that the six other
complete runtime catalogs pass their audits.

The API inventory was independently measured against `52d0c03f9`: 11041
current public docstrings, 10539 catalog symbols, 504 additions, two removals
and 114 changed existing sources. The exact symbol lists are recorded in
`features/data/411_api_debt_2026-09-21.json`. Neither the API catalogs nor the
extractor's debt test baseline was changed. Generation, review and full docs
acceptance remain open.

The complete nine-locale runtime audit still exits 1 with 369 diagnostics
(the first 200 are printed, followed by 169 more). Those diagnostics describe
missing/stale entries and translation defects in the other six locales; they
are not a count of unique bad sentences. The live COVERAGE and REVIEW_SCOPE
tables were regenerated after their existing reporter tests exposed stale
counts. The feature index was also regenerated to include already committed
items 464–467; no new item number was allocated.

After report regeneration, all four existing reporter tests pass (127.44 s),
and the generated instruction-index check passes. Together with the three
source/syntax tests, this gives seven passing checks for the new evidence and
reporting contracts; it does not turn the remaining full audits green.
