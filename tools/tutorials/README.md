# Tutorial refresh — 9 September 2026

This lane owns tutorials only. Application changes belong to the other sessions.
The working repository is `/mnt/firecuda2/codex/repo/spacr`, branch `nightly`.

The authoring workspace remains
`/mnt/firecuda2/Claude/toxoplasma_projects/tutorials`. Its `web/` and `catalog/`
files feed the publisher; the documentation copy is derived output, not the
place to make player or narration edits. `snapshot_sources.py` checkpoints the
small authoring inputs here so commits protect them as well as the published
output. It does not copy recordings, model weights, credentials, or datasets.

The user confirmed these requirements on 9 September:

- Preserve all 50 current voices, eight spoken languages, and existing
  caption-only languages.
- Record one English GUI master per lesson and reuse it for every language.
- Preserve the current player, visual treatment, and approved pronunciations.
- Show the current GUI, real downloadable datasets, folded actions, API, and
  supported acceleration paths. Retain useful specialist lessons and link them
  from their current hosts.
- Commit and push coherent checkpoints regularly. Do not publish to `main` or
  replace live remote media while the refreshed set is incomplete.
- Run substantial commands through `tools/run_memory_guarded.py --limit-gib 100`
  and limit thread counts. Do not overlap another session's full coverage run.
  The 100-GiB threshold is approximately 107 GB, below the maintainer's latest
  110-GB ceiling. It watches total system usage and stops only the guarded job.
- No further questions while the user is away; proceed with these choices.

## Checkpoints

### Maintainer-approved completion scope — 11 September

The maintainer explicitly approved **Coming soon screens for unfinished
workflows**, superseding the earlier no-placeholder rule for this release.
The candidate therefore contains 71 complete tutorial packages and five
unavailable screens: Map Barcodes, Model Compare, Model Zoo, Investigate Hit,
and OPS. These screens are not successful workflow demonstrations and do not
close the underlying application/data defects. OPS is under Mask.

`coming_soon.py` supplies text in all fourteen catalog languages. It only
transforms release copies; held recordings and original catalogs are preserved.
Screens have no narration/video, cannot be marked complete, and do not count
as available tutorials. The fifty-voice selection remains intact for all 71
ready lessons. Parent routing is generated from the live registry, including
for unchanged lessons whose media and narration are retained byte-for-byte.

`build_release_candidate.py` creates a new private candidate, rechecks the
existing full-library evidence and copies verified media without uploading it.
`verify_release_candidate.py <candidate>` checks the candidate in Chromium,
including all five screens in fourteen languages, mobile text visibility,
each ready English video/audio pair, seeking, and ready/unavailable transitions.
`checkpoint_release_candidate.py <candidate>` saves its exact player, fourteen
catalogs, route metadata, browser evidence and all media hashes under
`tools/tutorials/release_candidate/`. The maintainer subsequently requested all
remaining work pushed: `--include-web-media` includes web videos, posters, fonts
and examples. Narration and 4K media remain in the private candidate/original
workspace pending separate media-host upload approval; they are not in Git.

The candidate is an **offline preview**: its narration and 4K roots are relative
to its sibling `media_host/`. Do not copy that index directly into live docs or
run the legacy publisher against the old authoring catalogs. The version-update
and live-publication hold remains in force; release requires a separately
approved deployment with final hosted roots and live-byte verification.

1. Preserve and reconcile authoring/published sources; measure the live registry.
2. Capture the current Home/navigation and rebuild each runtime Core lesson using
   real test-data controls, a bounded operation, and inspectable output.
3. Refresh changed specialist routes and add missing lessons, without deleting
   useful old lessons. Audit API and acceleration explanations against source.
4. Translate changed scenes, generate all voices from frozen scripts, and render
   each silent master once. Validate pronunciation, audio, and caption timing.
5. Stage a coherent publishable set; check links, mobile playback, media hashes,
   and size limits. Report any deployment hold separately from completed media.

An inventory or passing structural test is not a completed tutorial. Only
validated recordings, narration, captions, and links close a lesson.

### Final library and web-copy checkpoints

`verify_library_checkpoint.py --held 12_map_barcodes 21_model_compare
22_model_zoo 71_investigate_hit --output <private-report.json>` reconciles
every other staged lesson, including the explicit whole-media retention path.
It checks final bytes against existing audio/browser reports, all fourteen
catalogs, and the current navigation registry. Listed holds and missing routes
(including OPS) remain visible; passing the checked subset is not whole-library
or publication approval. No model, synthesis or full-suite coverage run occurs.

`stage_web_renditions.py --all-verified` consumes the private
`library-checkpoint-2026-09-11.json` and uses the existing publisher encoder to
prepare 1440p copies under `web-renditions/`, never in the live docs/media tree.
Use the 100-GiB memory guard. Encoding is serial with two encoder/filter threads;
frame counts, every presentation timestamp and full decoding are checked.
Matching completed copies are reused on restart. Plate Viewer's moved-only
1440p file is copied byte-for-byte from the existing docs asset, not re-encoded.

`verify_staged_lesson.py --lesson <id> --web-rendition` tests the actual private
1440p bytes in the unchanged player, with independent `browser-web/` evidence.
For whole-media retention also pass `--retained-media`. It checks the loaded
video hash and decoded dimensions, narration/scene synchronization, links and
mobile layout without replacing the original 4K browser reports. This does not
replace visual review, listening approval, or the whole-set publication hold.

### Visual-only refresh with retained narration

`retain_narration.py --lesson 34_database` compares the selected lesson in all
14 staged, published-source and original authoring catalogs before copying any
audio. It requires all 50 original audio/timing pairs to match their identities,
narration and media hashes, refuses differing existing destinations, and rechecks
every copied byte. Original files remain untouched. Use `--retained-narration`
on both audio verification and final matrix reconciliation for this explicit
path; the normal newly rendered path still requires current synthesis inputs
and source-pinned translation reviews. Retention is not a new native-language,
listening or renderer-runtime certification.

`stage_lesson.py --focus-map lessons/34_database.focus.json` binds source-pinned
visual-only highlights without changing an accurate narration catalog. Playback
requires exactly the authored cross-links in chapters and transcripts, including
an empty set when the retained lesson defines none.

### Whole-media retention for a moved, unchanged specialist lesson

`verify_retained_media.py --lesson 33_plate_viewer` verifies all original
catalogs/audio/timings, fully decodes the 4K master and 50 tracks, and refuses
any staged production override. It writes evidence only, not new media.
`verify_staged_lesson.py --retained-media` then checks the real player against
those original bytes and the current generated parent route. Current-GUI
behavior still needs a separate real-data check: `capture_refresh.py --module
plate_view` uses Graph Builder's actual fold, threaded queries, minimum-count
restoration, well clicks and a genuine export picker. It is a retention audit,
not a replacement recording or a new human-language/listening sign-off.

## Navigation clarification — 9 September

The user requests two module sections: **Main modules** (the Home tiles) and
**Submodules**, grouped beneath their actual parent-module headings. A tutorial
whose module only moved is relocated without regenerating its video, narration,
or captions. Rewrite only content that changed. Preserve the existing lesson
IDs, media paths, deep links, and watch history when reorganizing the library.
Setup/API lessons remain accessible; Help-only utilities must be identified as
Help workflows rather than assigned an invented Home parent.

### Navigation checkpoint

`build_navigation.py` now derives the hierarchy from `tiled_apps()`,
`home_bands()` and `folded_children()`: 21 Home tiles, with all 73 existing
lesson identities assigned exactly once. The navigation-only publisher leaves
catalogs, captions, voices and media untouched. Browser checks cover desktop,
390-pixel mobile layout, parent search, retained deep links and Spanish labels;
they do **not** constitute playback or translation acceptance.

The full tutorial-route gate remains open for missing lessons. Import Images
and Regression Diagnostics need production; OPS is explicitly deferred because
its real workflow has not been validated (325, 9 September). No placeholder OPS
lesson or weakened route gate is a substitute for that validation.

## Expanded demonstration requirement — 9 September

The maintainer requests a real-data demonstration for **every module with a
downloadable test dataset**, not only Core modules. For each such lesson:

- Load the registered example through its visible UI control and identify the
  dataset and what question the example addresses.
- Enable plotting where supported, complete a bounded real operation, and show
  the resulting figures and console output. Merely pointing at settings or
  naming an output file does not satisfy the demonstration.
- Explain the spaCR AI controls and how assistance relates to the workflow.
  Do not imply that a response was obtained when no provider was used, expose
  credentials, or send private datasets just to make a recording.
- Demonstrate the live preview wherever one exists. In Mask, change filtering
  controls and model options, rerun as required by the real UI, and show the
  actual before/after segmentation or object counts. Distinguish display-only
  controls from filters/settings that affect a saved batch run.

The "move without remaking" exception remains for internally unchanged lessons
that only moved. Audit an existing lesson against the demonstration requirements
before deciding its current media can be retained. Do not silently treat a new
real-data or live-preview requirement as satisfied by relocating its page.

## Recording and staging commands

Use the private refresh environment under the authoring workspace, never its
old broken `.venv` symlink or an in-place dependency change to the app environment.
`requirements-refresh.txt` records the additional renderer/browser dependencies.

`capture_refresh.py --module mask --download --preview --preview-variants
--platform xcb` runs the actual UI, using a private example-data bind mount and
private preferences/logs. Launch it under `xvfb-run -a -s '-screen 0 3840x2160x24'`
and the memory guard. The real 4K screen is important: an offscreen platform's
small reported display can make dialogs open too narrow even when the main
window is 4K. Loading example settings replaces a screen; the recorder must
reacquire the visible screen before driving its preview.

`stage_lesson.py lessons/05_home.json --capture-module home` validates capture
hashes, focus geometry and specialist links before staging English. Set
`SPACR_TUTORIAL_WORKSPACE` to the refresh directory when calling the external
`render_all_voices.py`; model weights stay shared but narration outputs do not
overwrite the published collection. `verify_staged_lesson.py` checks the actual
player against staged files with HTTP byte ranges, including chapter seeking,
scene links and mobile layout. It does not claim human listening acceptance.

Translation drafts remain in `catalog-drafts`, separate from renderer inputs.
The first whole-scene NLLB pass dropped complete sentences and mistranslated
image crops as cultures; it was stopped and rejected. The replacement translates
sentence chunks and reconstructs each scene, but still requires semantic review
before any translated narration is generated. English text and the existing
published multilingual catalogs remain untouched by draft generation.

`apply_translation_review.py` promotes only explicit, English-hash-pinned
editorial corrections to staging, retaining other lessons. The review records
distinguish editorial checking from native-speaker and listening sign-off.

The first Plot-enabled Mask batch exposed a real partial overlay output: the
plotting loop selected `.spacr_plane_layout.json` as an image. Nine figures and
a successful GUI completion signal do not make that a complete demonstration.
`capture_acceptance.py` now rejects the console's partial-artifact markers, and
staging refuses that run. The application owners have the source-level report;
the tutorial lane does not patch the app to obtain a successful recording.

### Import Images recording

`capture_refresh.py --module import_images --platform xcb` opens the real
Import host and clicks its Import Images fold. It uses eight byte-identical
copies from Mask's downloaded microscopy example, in two distinct wells and
fields. The raw acquisition's varying `A` token is unresolved by the current
importer: the recording shows that refusal before switching to explicitly
prepared copies that omit only that token. The `C` channel, well and field are
retained and checked against the proposal; this is not a claim that raw naming
works automatically. Neither original images nor application code are edited.

The accepted recording demonstrates Scan, Save plan, Load plan and Import via
their actual buttons. File selections must emit acceptance, not merely leave
an older plan visible after cancellation. Output checks require eight independent
copies, unchanged source hashes, matching output hashes and unchanged well,
field and channel identities. The two input-selection guard tests were each
observed failing against in-memory mutations before committing the recorder.

The recorder uses Qt's real non-native file dialogs so an external desktop
portal cannot put the file chooser outside the private Xvfb recording.

### Japanese runtime dependency

The isolated environment's `unidic` package initially lacked its dictionary.
The existing `/home/olafsson/anaconda3/lib/python3.12/site-packages/unidic/dicdir`
was reused via a link inside the refresh environment. No app dependency or
pronunciation rule changed, and no second dictionary download was required.

### Regression and Diagnostics recording

`capture_refresh.py --module regression --download --run --ai-controls
--platform xcb` downloads the four actual example score/count pairs. A fresh
output root is mandatory: the default form can contain an unrelated saved
project path. The bounded demonstration uses 199 permutations, a two-well
minimum and both guide and gene results. Its p-value floor is 0.005, so this
is not a final discovery-quality analysis. PNG is selected in private figure
preferences; this Regression route produces figures without a separate Plot
setting. No spaCR AI prompt is submitted.

`capture_diagnostics.py --project <private-regression-project>` opens the real
Regression -> Diagnostics button and the generated PNGs in the system viewer.
It launches a private Xvfb display and session bus with private XDG configuration,
data, cache and runtime directories. The recorded file-manager/image-viewer
windows are actual system applications, not fabricated spaCR panels. It reads
the existing diagnostic summary and checks all three expected figures first;
it does not recompute the analysis or turn statistical warnings into passes.

A first guide-only example produced valid guide results but triggered a
downstream Hit List worker error requiring gene coefficients. The capture gate
now rejects that late worker error even after a main-pipeline success signal;
the new test was observed red before implementing the guard. The both-level
repeat completed without that error (434 guide and 325 gene results). This is
a tutorial configuration change, not an application fix for guide-only runs.
