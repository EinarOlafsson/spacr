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
- Run substantial commands through `tools/run_memory_guarded.py --limit-gib 110`
  and limit thread counts. Do not overlap another session's full coverage run.
- No further questions while the user is away; proceed with these choices.

## Checkpoints

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
