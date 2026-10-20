# Lesson 20 opening: capture and GUI timing evidence

Source app/recorder commit: `29552bae7`; app files match nightly `9692fd146`.
The two recorder fixes before this run are local held commits `467b2bda9`
(deferred Apply worker start) and `8f01f9a6b` (clear stale settings search).

Full run r28: 10/11 strict smoothness passes. Opening navigation: 26.96
distinct fps. Its one permitted retry: 26.92. Neither reaches the 27.0
near-miss floor; neither was accepted. All other clips were 28.64–29.77.
The clip gate labels the failure capture loss, but the GUI evidence needs
correction before relying on that label.

| Event | Full run | Retry |
|---|---:|---:|
| Make Masks click | 4.05 s | 4.08 s |
| Workbench fold click | 8.29 s | 8.45 s |
| Delayed GUI sample after fold click | 1.216 s | 1.099 s |
| Distinct frames in second 9 | 6 | 4 |
| Apply tab click | 13.49 s | 13.29 s |

`HostRuntime.field_sample` sums cumulative counters of currently visible
AmbientWidgets. Membership changes at navigation, making the counter regress
(2000 to 1895, and 1841 to 1749). The gate interpolates those samples and
does not classify a delayed interval as a stall if the next counter advanced.
WS is correcting telemetry in an isolated tools-only worktree, with the fps
limits unchanged. This diagnostic correction does not fix the app slowdown.

Home app follow-up: cold Workbench construction on the GUI thread. Stall
stacks name `FoldedModulePanel.__init__`, `CellposeWorkbenchScreen`,
`preview_registry.install`, `collapsible_splitter.insertWidget`, and
`flow.minimumSize`. Please retain the actual navigation actions and address
the construction/layout pause. A new app base requires a whole-lesson retake.

Lesson 20 has not entered audio, translation, rendering or publication. Four
narration scenes also need small wording corrections to match the in-clip
actions; those are drafted locally. The original scientific acceptance
receipts remain unchanged at accepted=false. Independent reference checks for
both runs passed at 18:18:33 EDT through the standard GPU guard: all three
saved masks match the direct reference exactly, and the three figure arrays
match their source images and outputs. Preview matches the raw-image default
reference and differs from Apply output. This checks input/output attribution;
it does not establish segmentation accuracy or approve either opening clip.
The two `verification_run_retry.json` receipts and `independent_reference.json`
results are included with hashes. Their dependency hashes still match.

The first isolated CUDA verifier failed with error 304 because its wrapper
omitted the writable `/proc` bind already used by the capture launcher. Its
original failure receipt is preserved locally. The successful retry used that
bind; no package or environment change was needed. No new recording is queued.

The evidence is copied byte-for-byte at each snapshot; the manifest records
original paths and hashes. The reference retry helper added only an empty
`accepted_near_miss` list to the local retry summary after its first archive.
Both summary versions are retained, and their verdict is still failed.
No gate threshold, result flag or recording was changed.

## Corrected telemetry replay

The tools-only correction is committed locally as
`6dfed0613c1b88de62f7e28021197c0e0fe1fb18` in
`/media/carruthers/mnt3/claude/spacr-capture-telemetry-20261013` and is held until
Home releases the tools/tests restriction. It tracks stable widget identities
and observed paint deltas, records delayed GUI sampler delivery, and keeps
unknown paint rates separate from capture loss. The fps and pointer limits
are unchanged. Fifteen focused telemetry cases and eleven owning gate cases
passed under CPU-only, offscreen, 8 GB validation. The included test log is an
explicitly labelled transcription of tool-output summaries, not a new run.

The corrected gate was then run on both existing native opening videos,
serially under the same 8 GB CPU cap. Both still fail: 26.96 and 26.92 distinct
fps, with app-freeze seconds 4, 8 and 9. Their changing legacy counters cannot
establish the remaining per-second app paint rates, so those seconds are
unlabelled rather than capture loss. Neither result can enter the capture-loss
near-miss exception. The two `opening_telemetry_replay.json` files are separate
diagnostic results with code and input hashes; the original gates remain
unchanged. Video measurements, pointer results, top-bar checks and thresholds
match the original gates exactly.

Home acknowledged the app opening issue in `9af7f3dc1` and queued its fix
behind CI/release. Lesson 20 remains unpublished and requires a fresh whole
lesson recording once the app fix is available. No further retake is queued.
