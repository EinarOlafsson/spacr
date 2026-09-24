# Documentation and tutorial continuation

The maintainer requested useful completion within the remaining token budget,
then will bump to 1.5.1.0 and push main. Do not perform that promotion here.
Incompatible translations are registered and fall back to English; they do
not block publication. Do not call drafts or pending media complete.

## Working rules

Use `/mnt/firecuda2/codex/repo/spacr`, branch nightly. Read the END of
`325_two_sessions_one_repo_working_protocol.temp` first. Claude owns GPU jobs.
All Python/pytest uses `tools/run_capped.sh`; private HOME/XDG_CONFIG_HOME for
probes, named test files, offscreen Qt. No whole suite. Explicit staging and
rebase before push. Preserve unrelated `tools/tutorials/authoring/project/`.
No new agents were spawned. Protected full CI35952253721 remains unclaimed;
older35944548693 finished/failure. Never cancel these protected runs.

## Current source and publication

Pushed checkpoint605aeb67a includes accepted table media192d0356b, wrapping
Toggle API7c2692ec3 and documentation fixes266203de0. Docs run35959737649
successfully built and deployed both channels; direct live readback passed.

English API:11,527 entries;9527public callables. Help includes Toggle's new
heightForWidth/minimumSizeHint methods. Conversion/stream docstrings are
current through738fb293b. The display-lifetime helper from728031e9b is also
included, with four passing source/inventory guards and current English audits.
Receipt:411_screen_lifetime_inventory_2026-09-24.json. English API/runtime audits pass. Strict Sphinx
passed atfc47d0194 before the two incoming docstring-only changes. Receipts:
411_wrapping_toggle_inventory_2026-09-24.json and
411_conversion_stream_api_2026-09-24.json. Remaining parameter-description
debt is unchanged; English publication does not imply all prose is complete.

The shared map is `spacr/resources/module_workflows.json`. API/workflow pages
and tutorials use it. New guides82Toxoplasma,83Plasmodium,84Candida and
85Host–Pathogen are linked. Setup order is GitHub,PyPI,Conda,installers, followed
by Home/orientation/pipelines and modules.

Last verified LIVE: I/605aeb67a, run35959737649. English API11526 and catalog
bytes match the commit; browser playback/chapter seeks passed for59,60,61,63
and main Mask. Both served player cache keys match their content hashes.
Receipt:358_live_table_guides_2026-09-24.json. Source/translation checkpoint
a79a8039d is pushed and awaits its later deployment; its media remain I.

## Accepted media: never re-upload

Workspace `W=/mnt/firecuda2/codex/workflow-authoring-20260922`,
stage `U=$W/user-walkthrough-stage`. Original captures are under
`/mnt/firecuda2/Claude/toxoplasma_projects/tutorials/refresh_2026-09-09`.

Previously accepted candidate I: `$U/release-candidate-append-hpih30qn`.
Four refreshed lessons59AnnData,60PCA,61Tabulate,63SmallMultiples.
2506/2506 hosted files SHA256match immutable revision
077895c4c53658d9af5a0b08cc38d814775c25f4;2,950,025,020bytes;1211compatible
narration tracks;85routes/84ready. All84hosted playback cases and14localized
unavailable views pass; mutation guards observed red; hold lifted. Eight
focused publication guards pass. Logs:table-candidate-{upload,pages}.log.
Do not upload candidate I again. Its predecessor H/6kuath0f is also accepted
at3295e3d747743e67517bd0a13a1a80fef74a5619. Earlier candidates are historical.

## Current remaining work

69of81existing scripts have practical walkthrough rewrites, plus four new
organism guides. Latest eight source rewrites:48HitList,56Lineage,64GateEditor,
65FeatureExplorer,66Outliers,67ExperimentDesign,68PowerDesign,69DoseResponse.
Their capture sequences/hashes match the originals. Matching English media
is rendering/checking in exploration-media.log, design-dose-media.log and
outliers-power-media.log. Inspect terminal results; do not publish new prose
with old audio. Prior prepared matching media covers55rewritten existing
lessons; last verified live also covers55. Audit:
`tools/tutorials/evidence/2026-09-23-user-walkthrough-review.json`.

Twelve sources remain for review/rewrite:33PlateViewer,70ExplainCV,
71InvestigateHit,72VolcanoExplorer,73ParameterSweep,74ImportImages,
75RegressionDiagnostics,77Embeddings,78Screens,79InputsOutputs,
80ImagePathways,81SequencingPathways.78–81are generated: edit the shared map
or generator, not only their JSON.71is the sole unavailable catalog lesson.
Six fresh-capture gaps remain:02Conda,03PyPI,04installers,17Timelapse,19Train,
76OPS. OPS has introductory/alignment media; full native OPS and Train remain
queued with Claude. Restoration tutorial/animation work remains owed.

AnnData's normal writer is repaired by b2dac1bc3/9863c8b7f. The new written
guide `docs/source/anndata_export.rst` uses the normal GUI/headless entry point;
its example exports/reopens2341x261 with28missing values and unchanged database
bytes. Evidence:2026-09-24-anndata-fixed-guide.json. The older video still uses
the supplied helper and needs a fresh normal-export capture; its historical
failure receipt is retained. The guide explicitly distinguishes that older
recording from the current exporter and explains preservation on failed writes.

## Translations and checks

All20new restoration captions/help sources have source-bound translations in
all9runtime languages;189runtime lookups and review cohorts pass. Nine updated
score-threshold tooltips are now source-bound and their runtime lookups pass;
review-cohort subtraction accounts for the exact single-source addition.

Locale debt is registered in411_release_translation_register_2026-09-24.json.
Most full API/runtime catalogs and matching tutorial narration still require
review. Frozen API drafting at3ee2a1809 has finished for all9languages, with11387
documents each. Source hashes and private file hashes are recorded in
411_private_api_drafts_2026-09-24.json. Icelandic retains138unresolved blocks.
Known semantic errors remain; all drafts need review/rebinding to current
source. They are not approved translations: never promote them wholesale.

Documentation-owned CI failures were reconciled:85routes/1080scenes, actual
organism lessons, current objectives, restored help rows, PSF GIF review and
README checkout size. Full tutorial audit:13passed,2locale tests report-only.
PSF GIF reviewed all18frames and87assets markedGOOD. Checkout1607MiB excludes
Git history;5.8GBwas the historical full-clone download. All9README translations
preserve the historical shallow-clone filter comparison. See
288_documentation_ci_repairs_2026-09-24.json and328_checkout_size_2026-09-24.json.

The Pages publisher uses verified immutable media to retain both channels
within its size limit. It hashes the transformed player to invalidate browser
caches. Seventeen publisher tests pass. Do not weaken size or hash guards.

Latest accepted candidate J: release-candidate-append-l1r815zg. Eight refreshed
walkthroughs48,56,64,65,66,67,68,69; all1722hosted files SHA256match
5fef816b732a14df2654c5eda9610baa071a080c (2,123,675,799bytes).819compatible
narration tracks; all84hosted playback and14localized unavailable views pass.
Hold lifted. Do not re-upload. The first readback hit HTTP503; recovery read
back the same immutable revision with four workers, then tagged it. Logs:
exploration-candidate-recovered-upload.log and exploration-candidate-pages.log.
Matching prepared media now covers63of69rewritten existing lessons, plus four
new organism guides. Last directly verified live remains I/605aeb67a until
this newer batch deploys.12scripts and seven fresh-capture follow-ups remain,
including AnnData's repaired normal exporter.872translation reviews are stale.

Strict English Sphinx with11527APIentries and the current AnnData guide passes
(anndata-current-guide-sphinx-followup.log). The guide's rendered API link and
index link resolve. Eight final candidate/inventory/README checks pass. Guide
receipt411_anndata_write_api_2026-09-24.json records the successful build.
