# Documentation and tutorial continuation

The maintainer requested maximum useful completion with the remaining token
budget before they bump to 1.5.1.0 and push main. Do not block a release on
incomplete translations. Keep registered English fallbacks. Do not claim
private drafts, old captures or unfinished narration as reviewed completion.

Work only in `/mnt/firecuda2/codex/repo/spacr`, branch nightly. Read the END of
`325_two_sessions_one_repo_working_protocol.temp` before acting. Claude owns
GPU jobs; the maintainer owns the version bump and main promotion. All Python, including probes and pytest, must
use `tools/run_capped.sh`; probes need private HOME/XDG_CONFIG_HOME. No whole
suite. Stage explicit paths, rebase before pushing, preserve other work.

## Current English sources

The English API manifest contains 11,524 rendered entries; the Help index is
current. Existing parameter-description debt remains. The shared module map
is `spacr/resources/module_workflows.json`. Four new guides are linked there:
82 Toxoplasma, 83 Plasmodium, 84 Candida, 85 Host–Pathogen. Generated API and
workflow pages use this same map.

57 of the original 81 scripts have practical walkthrough rewrites; four more
organism guides were authored. See the exact per-lesson audit in
`tools/tutorials/evidence/2026-09-23-user-walkthrough-review.json`. Rewritten
source does not imply new native screenshots or publication. 716 older
translation review records are currently incompatible with their English
sources; they remain registered work, not accepted translations.

The one unavailable catalog entry is **71_investigate_hit**. OPS76 already has
an introductory/alignment lesson, but the complete current OPS recording and
native Train recording remain queued with Claude. Do not call OPS the sole
ComingSoon entry.

## Media workspace

`W=/mnt/firecuda2/codex/workflow-authoring-20260922`
`U=$W/user-walkthrough-stage`

The last verified deployed baseline is `release-candidate-append-8vaimbg8`,
published by run35951320497 at nightly8b8458b14. Live API/catalog bytes exactly
match the committed sources. Real browser playback and chapter seeks pass in
both channels. Receipt:411_live_current_documentation_2026-09-24.json.
Do not upload it again.

The organism candidate is `release-candidate-append-_h__k19q`, manifest SHA256
497ba62c9e87f1cee9daf617cdc1d5fb54f9412660967fe03ea15d8398923b0d.
It contains 85 routes, 84 playable entries, 4 new guides, and refreshes 22,24,25,
29,30,31. All 84 playable checks and 14 localized unavailable views passed.
The media upload uses branch `candidate-organisms-20260924-hk19q` and tag
`tutorials-organisms-20260924-hk19q`; check the publication receipt and
`organism-candidate-upload.log` before any retry. Never upload to main or reuse
an existing media branch. It was accepted and pushed at61688f2d6: all5250
hosted files SHA256matched revision931a7e95feac3730cde47f02d6a76f805aa59038,
and all84 hosted playback cases plus14 placeholder views passed. Do not
re-upload it. Pipeline79/80 links changed without changing prose,
translations, audio or timing; rebinding evidence records exact equivalence.

Additional staged practical lessons:
- 32 Align & Stitch and 35 Converter: `align-converter-media-rerun.log`.
- 34 Database,36 Import,37 Batch,38 Distributed Jobs,39 Evaluation,40 History:
  `utilities-media.log`.
- 42 Curate,43 Illumination,44 Data Manager,45 Project Browser,46 Napari,
  47 Barcode QC: `data-tools-media.log`.
- 50 Run Compare,51 Control Charts,52 Pipeline Graph,53 Profiler,54 QC,
  55 Image Scatter: `analysis-views-media.log`.

All20 additional lessons finished their narration, master, rendition and
Chromium checks. Their combined candidate is
`release-candidate-append-8vaimbg8`, built against the exact organism baseline.
All84 local playback cases and14 placeholder views passed, and both mutation
guards observed red. Upload/readback is in `data-candidate-upload.log` using
branch `candidate-data-tools-20260924-8vaimbg8` and tag
`tutorials-data-tools-20260924-8vaimbg8`. Inspect receipts before any retry;
never overwrite or reuse an existing media branch.

The data candidate is now accepted:3290/3290 hosted files SHA256matched
revision948fd55c6d13ee107b8bce22c26957178e7415dd (3,742,683,994bytes).
All84 hosted playback cases and14 placeholder-language views passed; the
checkpoint hold is lifted. It contains1603 source-compatible narration tracks.
47 of the53 rewritten existing lessons now have matching prepared media;
the six listed capture gaps remain. Four new organism guides are also ready.

Four later rewrites are now accepted for publication:49Methods&Results,57LayerViewer,58GraphBuilder
and62FeatureDictionary. Their original capture hashes and visual sequences
are retained, with shorter practical narration. Rendering/check logs are
`graph-reporting-media.log` and `layer-dictionary-media.log`. The new downloadable
Layer_Viewer_real_image_mask.zip contains the exact image/mask TIFFs from the
recording; archive readback and original hashes pass. Update publication only
after the complete media pipeline passes.

Docs run35948635800 built both branches successfully but failed assembly:
1,081,843,213bytes exceeded the950MiB Pages budget. The publisher now uses
immutable full-resolution video URLs only when local renditions match the
manifest and the corresponding hosted recordings have complete readback
evidence. Other media remains locally deduplicated.
This avoids dropping either documentation channel or weakening the size limit.
The real main build plus current nightly site assembles to682,878,218bytes
(651MiB); all15 focused publisher checks pass. The source trees retain their
local videos; only disposable Pages artifacts use the hosted recordings.

New example downloads: `Align_Stitch_nine_tiles.zip` (nine original pixel crops,
row-major3x3,overlap0.25,reference channel1) and
`Control_Charts_SYNTHETIC_campaign.csv` (synthetic fixture values, LF newlines).
They are in the committed tutorial examples directory and linked by their
respective current lesson sources.

## Checks and evidence

309 data-tool checks,66 analysis-view checks,42 workflow/review checks and
40 incremental candidate tests pass. Native Host–Pathogen receipt records
164 cells,97 vacuoles,2 wells; unknown parasite counts stay unknown. Native
Model Zoo CPU receipt records149 labels from three unchanged uint16 fields.
These are execution examples, not biological accuracy claims. Source evidence
is in `tools/tutorials/evidence/2026-09-24-*`.
The final strict English Sphinx build passed with11,520 API entries;73 actual
rendered module tutorial links preserve both channels. English runtime/API
audits pass. Translation incompatibilities remain report-only; per-locale
issue counts are in `features/data/411_release_translation_register_2026-09-24.json`.

Frozen API translation drafting runs in a separate checkout at3ee2a1809.
Eight languages have complete private drafts. Korean completed11,387documents
in `api-full-draft-20260923-3ee2a1809-ko.json`; Icelandic follows. These drafts
contain known semantic errors and are NOT approved translations. Do not
promote a full machine-generated catalog. Review and rebind to current source.

Six restoration control labels and the updated PSF/restoration stage-order
caption now have source-bound translations in all nine runtime languages.
All nine actual runtime lookups and both Swedish/French review-cohort checks
pass. English fallback is allowed. The settings-flow page has also
been regenerated for the new Model Zoo normalization reader.

Six more restoration choices/status strings now have source-bound translations
in all nine languages, including the ready/error format placeholders. Eighteen
formatted runtime lookups and both review-cohort checks pass. The subsequent
readiness/help batches complete all20 newly added restoration UI sources in
all nine languages. The shared Compare label is corrected in Portuguese and
Icelandic, with a command form in German. All189 runtime lookups and source
hash bindings pass; receipt:411_restoration_runtime_complete_2026-09-24.json.
The restoration tutorial and animation remain owed.

The four later videos completed their per-lesson checks. Their private
candidate is `release-candidate-append-6kuath0f`, built against8vaimbg8.
All84 local playback cases and14 placeholder-language views pass; both
placeholder mutation guards observed red. Upload/readback is in
`four-guide-candidate-upload.log`, branch
`candidate-practical-guides-20260924-6kuath0f`, tag
`tutorials-practical-guides-20260924-6kuath0f`. Inspect the receipt before any
retry. The candidate has1407 valid narration tracks. Remaining translated
tracks stay registered pending review and synthesis.

All remaining API inventory guards are reconciled at403a5d279:9524callables,
11520API entries and2290preexisting required-parameter omissions. The exact
subtraction proof and four passing focused guards preserve the prior boundary.
Claude's full test workflows35944548693 and35952253721 are protected and must
be allowed to finish across pushes. No current full-CI success is claimed.

No new agents were spawned. Preserve the unrelated untracked
`tools/tutorials/authoring/project/` directory.

## Latest accepted batch

Candidate6kuath0f passed all84 hosted playback checks and14 localized unavailable
views. All2898 files SHA256matched immutable revision
3295e3d747743e67517bd0a13a1a80fef74a5619 (3,395,194,044bytes). Hold lifted;
1407 source-compatible narration tracks. Do not re-upload. This brings prepared
matching media to51of57 rewritten existing lessons, plus four new guides.
Last directly verified live checkpoint remains8b8458b14 until HTTP readback.

The publisher now derives the browser player cache key from its transformed
contents, including the immutable media map.16 tests pass, including changing
only a nightly video while main stays identical. Restoration readiness and
long tooltips complete the20new UI sources in all9runtime languages;189actual
lookups/source bindings and both review-cohort checks pass. The restoration
tutorial and animation remain open.

Claude's GC/mask validation changes add two rendered API entries (11522total)
and update four existing API documents. The public callable boundary remains
9524 because these helpers are private. Exact receipt:
411_gc_mask_inventory_2026-09-24.json. Four inventory/source guards pass.

Final strict English Sphinx build passed for the GC/mask API refresh; log
`gc-mask-sphinx.log`, receipt411_gc_mask_inventory_2026-09-24.json. All source
changes are pushed through32e760991; docs run35955044338 is building both
channels. This is deployment pending, not a failed English/translation gate.

The subsequent lightweight Motility parser adds two rendered API entries
(11524total) and updates group_merged_files. Four source/inventory guards
pass; Help is regenerated. The locale register includes this exact delta.
The strict Sphinx receipt covers the immediately preceding11522-entry build;
the publication workflow rebuilds current sources. Receipt:411_motility_lightweight_api_2026-09-24.json.
