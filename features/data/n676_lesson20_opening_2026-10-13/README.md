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
actions; those are drafted locally. The raw scientific acceptance is still
false pending independent reference checks. An isolated CUDA verifier first
failed with error 304 because its wrapper omitted the writable `/proc` bind
already used by the capture launcher. A corrected reference-only GPU turn
is queued through the normal guard; no new recording is queued.

The evidence is copied byte-for-byte; the manifest records original paths
and hashes. No gate threshold, result flag or recording was changed.
