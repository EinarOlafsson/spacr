# Documentation and tutorial continuation

The maintainer requested maximum useful completion with the remaining token
budget before they bump to 1.5.1.0 and push main. Do not block a release on
incomplete translations. Keep registered English fallbacks. Do not claim
private drafts, old captures or unfinished narration as reviewed completion.

Work only in `/mnt/firecuda2/codex/repo/spacr`, branch nightly. Read the END of
`325_two_sessions_one_repo_working_protocol.temp` before acting. Claude owns
release promotion and GPU jobs. All Python, including probes and pytest, must
use `tools/run_capped.sh`; probes need private HOME/XDG_CONFIG_HOME. No whole
suite. Stage explicit paths, rebase before pushing, preserve other work.

## Current English sources

The English API manifest contains 11,519 public entries; the Help index is
current. Existing parameter-description debt remains. The shared module map
is `spacr/resources/module_workflows.json`. Four new guides are linked there:
82 Toxoplasma, 83 Plasmodium, 84 Candida, 85 Host–Pathogen. Generated API and
workflow pages use this same map.

53 of the original 81 scripts have practical walkthrough rewrites; four more
organism guides were authored. See the exact per-lesson audit in
`tools/tutorials/evidence/2026-09-23-user-walkthrough-review.json`. Rewritten
source does not imply new native screenshots or publication. 664 older
translation review records are currently incompatible with their English
sources; they remain registered work, not accepted translations.

The one unavailable catalog entry is **71_investigate_hit**. OPS76 already has
an introductory/alignment lesson, but the complete current OPS recording and
native Train recording remain queued with Claude. Do not call OPS the sole
ComingSoon entry.

## Media workspace

`W=/mnt/firecuda2/codex/workflow-authoring-20260922`
`U=$W/user-walkthrough-stage`

The last verified deployed baseline is `release-candidate-append-xw4j5buw`
(Agreement/Invasion/Replication). Do not upload it again.

The next candidate is `release-candidate-append-_h__k19q`, manifest SHA256
497ba62c9e87f1cee9daf617cdc1d5fb54f9412660967fe03ea15d8398923b0d.
It contains 85 routes, 84 playable entries, 4 new guides, and refreshes 22,24,25,
29,30,31. All 84 playable checks and 14 localized unavailable views passed.
The media upload uses branch `candidate-organisms-20260924-hk19q` and tag
`tutorials-organisms-20260924-hk19q`; check the publication receipt and
`organism-candidate-upload.log` before any retry. Never upload to main or reuse
an existing media branch. Pipeline79/80 links changed without changing prose,
translations, audio or timing; rebinding evidence records exact equivalence.

Additional staged practical lessons:
- 32 Align & Stitch and 35 Converter: `align-converter-media-rerun.log`.
- 34 Database,36 Import,37 Batch,38 Distributed Jobs,39 Evaluation,40 History:
  `utilities-media.log`.
- 42 Curate,43 Illumination,44 Data Manager,45 Project Browser,46 Napari,
  47 Barcode QC: `data-tools-media.log`.
- 50 Run Compare,51 Control Charts,52 Pipeline Graph,53 Profiler,54 QC,
  55 Image Scatter: `analysis-views-media.log`.

Each renderer performs source-bound English narration checks, creates the
master and 1440p rendition, and checks both in Chromium. Inspect final status;
an output file alone is not completion. After the first candidate is published,
build one refresh candidate for the ready lessons against that exact baseline.
Do not overwrite or reuse the prior candidate. Publish through the existing
upload/readback/pages/browser/checkpoint/record tools.

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

Frozen API translation drafting runs in a separate checkout at3ee2a1809.
Seven languages have complete private drafts. Korean continues in
`api-full-draft-20260923-ko-continuation.log`, then Icelandic. These drafts
contain known semantic errors and are NOT approved translations. Do not
promote a full machine-generated catalog. Review and rebind to current source.

No new agents were spawned. Preserve the unrelated untracked
`tools/tutorials/authoring/project/` directory.
