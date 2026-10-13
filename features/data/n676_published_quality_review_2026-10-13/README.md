# Published capture review and held capture tooling

Source tracing covers 61 unique selected app clips / 64 uses in published
04/05/07/08. Twelve static installer/reference scenes are outside this sampler
review. Native/video/receipt/poster bindings match, and all 63 available staged
video copies match their selected original clips. Candidate catalog prose and
scene content match the staged lessons; differences are only enriched
narration_voices metadata. This metadata audit alone is not a full quality pass.

Four selected source clips were then decoded and replayed through the corrected
gate. Picture-change, pointer, top-bar measurements and numeric limits remained
unchanged. All four fail; their original passing gate receipts are preserved.

| Published use | Distinct fps | Failed seconds | Evidence scope |
| --- | ---: | --- | --- |
| 05 Home, scene index 32, Spaceout field | 28.48 | 22, 23 | GUI sampler scene-clock gap 22.240–22.846 s, supported by the original stall stack |
| 05 Home, scene index 27, Preferences Apply | 28.14 | 4 | Legacy aggregate paint estimate; no dense sampler gap |
| 07 Mask, scene index 11, batch settings | 28.95 | 55 | Legacy aggregate paint estimate; no dense sampler gap |
| 08 Measure, scene index 4, all channels | 28.76 | 16 | Legacy aggregate paint estimate; no dense sampler gap |

The 606 ms Spaceout gap is a GUI sampler/scene-clock interval; the replay
records a 0.367 s sample-to-video offset. The separate watchdog reports a
0.38 s GUI stall at its recorded clip second 22.63. These measurements have
different scopes and are preserved separately. The Spaceout stack is tied to
Preferences Apply / apply_preferences_to_app /
console_panel.apply_zoom / preferences.get_font_scale in the recorded GUI
stack, with background numba warming. It needs app investigation and a fresh
whole-lesson capture on the final app base. For the other three, historical
aggregate counters do not prove a physical GUI pause. They emit freeze labels
and therefore cannot certify acceptance under the consistent gate; fresh
stable-widget telemetry is needed. Do not describe them as measured GUI stalls.

One strict passing 07 organize_popup clip has slow seconds 28/33/38 without
freeze or GUI sampler delay. This review does not change the strict average-fps
policy for slow seconds. The two selected near-floor approvals (05 appearance
animation and 08 second field) have no dense sampler gap or counter regression;
this audit does not renew their full quality certification. Source-run receipt
rows also include omitted/replaced scenes, so selected published media was
traced through staged visuals and frame origins rather than assumed from all
receipt rows.

The publication fact stays unchanged. The remake goal remains all 84 existing
routes: 80 have not yet been remade, and 05/07/08 need repair or reassessment.
No route was unpublished, no original gate/receipt/video was edited, and no
new recording, audio, translation, render or upload was started. Lesson 20
remains next, followed by 43/24/26; its failed opening still waits for the Home
app repair and release done.

The held integration tree is
`/media/carruthers/mnt3/claude/spacr-capture-ready-20261013`, commit
f7b88b7ceeeed9632c5ad7cecb0960d6cab66c6d. It combines stable-widget telemetry,
bound lesson 20 staging acceptance, queue prerequisite checks with an explicit
launcher option, and repository-owned scoped scientific acceptance/staging
helpers. Commit cf3bd48a6 makes every emitted app-freeze diagnostic fail even
when average fps passes; it retains the legacy diagnostic scope and changes
no numeric threshold. Twenty-seven telemetry/gate cases pass, including a
modern stable-widget low-paint second without sampler delay.

Combined owning tests pass **165 cases** under capped CPU/offscreen execution.
Separate helper and queue runs passed 69 and 14 cases. Included raw logs bind
these results; they do not certify recording quality. All tool/test changes
are committed locally and held until Home reports release done. This handoff
push changes features only. The capture worktree stays frozen at 29552bae7 to
retain the existing reference source bindings.

A separate bound read-only app investigation is now included in
`lesson05_preferences_apply_readonly_20261013.json`. It reviews the recorded
0cde90 app files, which match the frozen CH copies; the recorded driver blob
is reviewed separately because the current driver differs. Apply uses the
shared QSettings block throughout this call, so the stack does not establish
repeated INI reads as a cause. Every unchanged console zoom still restyles
retained output documents. This is confirmed repeated work with unmeasured
cost. Concurrent cold Numba entry-point discovery is an unmeasured hypothesis.
The existing smooth test warms Apply before measuring it. Home should compare
cold first Apply and Save separately, then unchanged and actual-scale changes,
including pending sliced work and both shown and hidden consoles. No app fix
or performance acceptance is claimed by this source review.
