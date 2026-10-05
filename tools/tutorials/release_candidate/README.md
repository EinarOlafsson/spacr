# Tutorial publication checkpoint — 5 October 2026

The current checkpoint is **`release-candidate-append-y1yawu76`**, refreshing
Home and Alpha features on 5 October 2026. Its media is pinned to the Hugging Face
dataset commit **`290e9e70174e13add3df25329e937ca52549e119`**. The
[publication receipt](publication-receipt.json) records the upload and full
byte readback; [hosted playback checks](published-media-browser-checks.json)
cover all 85 ready routes. The other 83 lessons are preserved from wave 4.
Nightly website deployment and its readback are tracked separately.

| Published contents | Count |
| --- | ---: |
| Ready lessons and navigation routes | 85 |
| English scenes | 1,192 |
| Catalog languages | 14 |
| Spoken languages | 8 |
| Published voices per lesson | 27 |
| Published narration tracks | 2,295 |

The renderer supports 50 voices. This checkpoint publishes the verified
27-voice selection, including one English voice; it does not contain 50
tracks for each lesson. Voices are offered only for source-bound, AI-reviewed
translations. Other catalog languages use English audio. The maintainer's
voice and phone sign-offs and native-speaker-review waiver are recorded in
item 358; AI technical review does not imply native-speaker listening review.

Wave 4 refreshed Mask, Import, and Import Images and added the two permitted
alpha lessons: **86, Alpha: organism modules**, and **87, Alpha: experimental
features**. Lessons 83 and 84 were withdrawn before this checkpoint. All other
lessons record with both alpha preferences off. Every GUI recording starts
fresh without restored sessions or drafts; the policy is in
[capture_policy.py](../capture_policy.py).

The current Home/Alpha refresh has 22 and 28 scenes respectively, including
the genuine waiting Live plate card in the alpha lesson. All 54 narration
tracks pass current-source acceptance. Local path roots in nine captured
frames were replaced using the normal fitted-text redactor; the original
captures and initial media remain preserved. Replacement masters, web copies,
language/caption playback and OCR sweeps pass before publication.

The deployed tutorial tree is
[`docs/source/_extra/tutorials`](../../../docs/source/_extra/tutorials).
Nightly and main documentation publish independently; a nightly push updates
the nightly site, while main changes at promotion. The player pins an
immutable media revision, rather than following the dataset's `main` branch.
Previously published revisions remain available for rollback.

## Current refresh work

The source-current audit on 5 October identifies newer authoring content for
Home, Measure, Batch Runner, Distributed Jobs, and Alpha features. Those
changes require the normal capture, staging, narration, caption, rendering,
and playback checks before they replace published media. Lesson 87 also
needs the newly added Live plate scene. The existing source-bound translations
are preserved, and the new scene has translations in all 13 non-English
catalog languages. Authoring changes alone are not publication evidence.

Run the audit from this checkout, using the existing spaCR Python environment:

```bash
CUDA_VISIBLE_DEVICES='' tools/run_capped.sh 8G python \
  tools/tutorials/audit_user_walkthroughs.py
```

The auditor compares every authored lesson with deployed nightly and main
catalogs, including the alpha lessons and newly numbered lessons. Its report
is [the walkthrough evidence](../evidence/2026-09-23-user-walkthrough-review.json).
That report detects text drift; it cannot establish capture quality or audio
synchronization. `--output` can place a diagnostic report in scratch.

The macOS updater scene was omitted at the maintainer's accepted scope because
no native capture host was available. It was not recorded or published. The
[installer guide](../../../docs/source/installer_guide.rst) remains its
available documentation.

## Validate and publish a later candidate

`checkpoint.json` identifies the publication and its manifest hashes. Its
`private_candidate` is the original producer's filesystem location, not a
portable path or a promise that the complete candidate exists on this host.
A complete candidate includes `web/`, `media_host/`, and its release manifest;
this Git directory preserves the publication metadata and web checkpoint.

1. Capture the current application into an isolated workspace. Stage each
   changed lesson with `stage_lesson.py`; retain unrelated published lessons.
2. Generate only source-bound reviewed language tracks. Run the audio,
   caption, visual, and complete-candidate checks against the actual artifacts.
3. Verify the complete candidate with `verify_release_candidate.py`. The
   existing Python environment must include Playwright and its Chromium
   browser. `--serve` offers a local preview without publishing.
4. Use `publish_release_candidate.py upload` with a new versioned branch and
   tag. Read back all uploaded bytes, then stage Pages with the verified
   immutable commit. Publish the deployment catalogs, media roots, navigation,
   bundled tutorial index, and cache keys together.
5. Verify the exact staged and deployed player, including chapter seeking,
   native captions, language changes, and audio/video synchronization. Update
   the receipt and this checkpoint after those checks pass.

Example validation, with the actual complete candidate path substituted:

```bash
CUDA_VISIBLE_DEVICES='' tools/run_capped.sh 8G python \
  tools/tutorials/verify_release_candidate.py /path/to/complete-candidate
```

The publisher's `upload`, `readback`, `pages`, `record`, and `resume-receipt`
subcommands separate these checkpoints. Inspect `--help` for the selected
subcommand before use. Never replace a published media pin with unverified
staging output. Historical acceptance remains evidence for its named revision;
a green source test alone does not prove that newer media has been published.
