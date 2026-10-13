# Runtime translation delta, 2026-09-27

AI technical review (Codex), no native-speaker signoff.

All nine locales have source-bound reviewed translations for 140 shared
sources. Hindi additionally has direct translations of 35 historical
captions (IDs 821–855), following content and technical review by two Codex
agents. These are ordinary microscopy/UI captions. No external translator
or its previously rejecting filter was invoked; earlier fallback history is
retained as dated evidence.

The original 129-source delta was extended by 11 real prose captions that
canonical extraction had missed: foundation-model registry labels, spatial
mask labels, the Detect platform choice, spatial-transcriptomics toggle
help, and the cloud URI placeholder. The extractor now includes those
sources. Visium, Visium HD and Xenium remain exact product identities;
Open belongs to the compact catalog. URI examples, model identifiers,
placeholders, defaults and scientific qualifications are preserved.

The root Codex agent independently reviewed the original 129 German and
French translations and all 35 historical Hindi translations. Another Codex
agent independently reviewed the original 129 Icelandic translations;
its wording corrections were incorporated. The supplemental 11-source set
and other locales have authoring-agent AI technical review. None of these
reviews is native-speaker signoff or listening review.

All 1,295 locale/source cases passed source-currentness, contextualization,
protected-literal, placeholder/format, HTML, accelerator, newline and script
gates with zero hard errors and zero length warnings. Nine Spotiflow cases
are canonical identities; the other 1,286 cases were atomically written as
review records. The strict canonical loader accepted the complete existing
and new review collections in all nine locales. No model inference ran.

The old item463 tooltip was retired only where its source had been replaced;
complete original review files and replacement provenance remain under
`features/data/463_retired_runtime_review_2026-09-27/`.

The external inventory was measured against commit
`6ba9eba0f308fc468c1e934c5035967b5f8bbee9`, reproducing the previous ratchet
fingerprint exactly. The full difference is +813/-37 record identities:
1,194 setting labels, 1,216 help entries, 229 categories, 6,014 UI captions,
and 72 module summaries. All 76 awaiting captions have a known owner.
The exact identities are retained in
`features/data/316_runtime_inventory_delta_2026-09-27.json`.

English and all nine locale catalogs have now been regenerated against the
expanded inventory: all 8,361 unique sources resolved without inference.
The full nine-locale canonical audit passed, including all 35 former Hindi
fallbacks. The measured ratchet pins were advanced only after that pass,
and all 76 resolved captions were removed from the awaiting set. All 54 named
runtime checks passed with the advisory translation plugin disabled: 51 in
the first run and the three corrected cohort/context assertions in the
focused rerun. Ten API-dependent cases and combined reports await the API
lane. The receipt is
`features/data/316_runtime_codex_delta_2026-09-27.json`.
