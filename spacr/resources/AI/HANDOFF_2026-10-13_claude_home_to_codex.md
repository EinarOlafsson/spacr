# Handoff: Claude (Home master session) -> Codex, 2026-10-13 ~15:40 ET

Claude ran out of weekly tokens. This file is the complete state. Read AGENTS.md
and features/HANDOFF.md ("CURRENT HANDOFF FOR CODEX AND OTHER AGENTS") first; the
rules there (worktrees, push only `git push origin HEAD:nightly`, no stash/reset in
the shared checkout, no Co-Authored-By/AI trailer, no `#` comments in spacr/,
tr() for strings, regenerate artefacts with their tools, never raise a ratchet,
tests with `CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen tools/run_capped.sh 6G
python -m pytest -n 2`, RAM < 110 GB, no GPU at Home, never touch Timepose/livecell
jobs) all still apply. While Timepose training runs at Home, keep at most 2 heavy
agents/test runs at once and wait if `free -g` shows < 8 GB available.

## 1. Goal: release spaCR 1.5.1.5

Plan (maintainer-approved): one green full `tests` CI run on nightly, then bump to
1.5.1.5 through the release workflow, then publish the notes:
`gh release edit v1.5.1.5 --notes-file docs/notes/release_notes_1.5.1.5.md`, then
`python tools/build_release_notes.py`, then after the bump run
`python tools/build_readme_images.py` (release.py writes the installer block in its
old form). Remind the maintainer to make the GHCR spacr-sif package public.

Dispatch CI with `gh workflow run tests.yml --ref nightly` (the workflow ignores
docs/** and features/**). Push runs and dispatch runs are in different concurrency
groups; cancel duplicates with `gh run cancel`.

### Last CI run: 38444864697 on nightly 4c26bb3a8 -> FAILED. Failures:

1. `tests/qt/test_mask_screen_field_frame_costs_near_home_20261012.py::test_a_mask_field_frame_reads_no_preference_store`
   and both tests in `tests/qt/test_a_screen_built_while_torch_imports_uses_the_window_backdrop_20261013.py`:
   "1.0 preference reads per field frame". The CI stack names the cause:
   `console_panel.py:258 _CopyGlyphButton.paintEvent -> active_palette() -> resolve_effective_theme -> get_theme`.
   **FIX READY, NOT PUSHED**: branch `paint-palette`, commit `b07389e36`
   (worktree /mnt/firecuda2/Claude/spacr-worktrees/paint-palette): that paintEvent
   uses `theme._painting_palette()` (cached per application palette). 596 related
   tests passed locally. Other paintEvents also call active_palette() (shortcuts,
   dock, activity_spinner, workflow_diagram, align, profiler, make_masks:7714,
   image_scatter, plate_view) but tests monkeypatch `<module>.active_palette`
   there, so only change them with their tests.
2. `tests/qt/test_cov_r5_app_screen.py::TestTheSheetBuilderIsFoundEitherWay::test_a_screen_whose_package_cannot_be_resolved_still_draws`
   ERROR ModuleNotFoundError (new; likely from 6f38f9f8f's `_painting_theme` import path in AppScreen paint). Not investigated.
3. `tests/qt/test_home_navigation_acceptance.py::test_visible_home_tile_opens_real_screen[mask]`:
   a `spacr.qt.first_run._TourOverlay` is on top (assertion lists it). Not investigated
   (maybe the first-run tour now shows in that test, or the "widget on top" check from
   742bfc9de needs to ignore the tour overlay).
4. `tests/qt/test_spaceout_field_holds_its_budget_at_1440p_20261012.py::test_a_colour_wave_frame_stays_within_a_few_plain_frames_at_1440p`
   (budget was moved to 6.0 in 4f7abea96; still flaky on CI runners). Look at the log value.
5. Coverage gate "every module at 90% and none loses coverage" and "Release gate"
   (aggregate). Check the coverage job log for which modules dropped.
Job logs: /mnt/wd4tb/scratch/ci4c26/*.log (previous rounds: /mnt/wd4tb/scratch/ci921/, ci3864/, ci4874/).

Next step: fix 2-5 in a worktree, cherry-pick b07389e36, push once, dispatch CI,
repeat until green, then bump.

## 2. Work committed LOCALLY, held until after the release (push after the bump)

| Branch / worktree | Commit(s) | What | Notes |
|---|---|---|---|
| `makemasks-paint`, /mnt/firecuda2/Claude/spacr-worktrees/makemasks-paint | aea9783d8, ddbc1a518, a9336827e, 71cba5ffc + UNCOMMITTED edits to make_masks.py and its test | Lesson 14 Make Masks: duplicate screen field removed (60->30 flushes/s), canvas redraw ~400->30 ms (byte-identical LUT blend), Wand/numbering/spacr.io off the GUI thread, YOLO driver waits for the requested field; 71cba5ffc fixes offscreen segfaults (GC on GUI thread in driver, painters end before paint returns) and recomposes only changed labels | The agent hit the token limit mid-work: review/finish or discard the uncommitted diff, re-run tests/qt/test_make_masks_lesson14_paint_and_stalls_20261013.py and the make_masks suites, stress the lesson driver (/mnt/wd4tb/scratch/mm14/tools/h_editor.py) for crashes. 3 new strings need translations (Wand still filling / failed / region not applied). |
| `feat-697-brush-labels`, /mnt/firecuda2/Claude/spacr-worktrees/697-brush-labels | 59e76dd36 (on top of a9336827e) | Feature 697: each disconnected brush stroke is a new label; Shift+drag with the Brush keeps the label (Alt still pans) | Rebase onto final makemasks-paint. Add/add conflict on features/new/697_*.txt: take this branch's version. 3 new strings to translate; the pan row in the shortcut panel needs rewording by the translation lane. Lesson 14 brush/pan clips need re-recording. |
| `feat-696-auto-diameter`, /mnt/firecuda2/Claude/spacr-worktrees/696-auto-diameter | 7a93f9216 | Feature 696: magnification sets default diameter on every path incl. Cellpose-SAM; 0 == "auto" (stored "auto"); "px" suffix removed; journal records diameter used | ~30 translation tests fail ("stale reviewed runtime source") until the WS translation lane translates the changed English (diameter tooltips, "Diameter (px)", grid refusal sentence, MaskConfig/Run/follow_the_form docstrings). Push together with or right before those translations. Behaviour change: blank diameter on Cellpose-SAM is no longer native scale. |

## 3. Queue after the release (maintainer's order)

1. Push the three local branches above (with WS translations).
2. README text boxes (the SVG panels): each panel image links to its text version;
   if a panel already links somewhere, the panel text must be the START of the
   linked page (maintainer's choice "Box text = page's start"). Tools:
   tools/build_readme_images.py, packaging/generate_readme_visuals.py; copy README
   outputs to main as done in e0bedd153 (README.rst, docs/i18n/readme,
   docs/readme_images, docs/readme_assets, docs/models/cards).
3. Feature 699 (features/new/699_*.txt): Submit to cluster reads ~/.ssh config +
   keyboard-interactive Duo/Okta second factor, one approval per session.
4. Feature 700 (features/new/700_*.txt): anonymised error reports through a bot
   account; the token lives ONLY in a relay (HF Space / Cloudflare Worker), never
   in spaCR; on by default with first-run disclosure. Maintainer must create the
   bot account (GitHub names can't contain "_", e.g. spacr-first-responder) and a
   fine-grained Issues:write token.
5. Feature 698 (features/new/698_*.txt): web mask generator on the spaCR site
   (site-src), one page per model class, BOTH GPU options (HF Space ZeroGPU and
   in-browser WebGPU), random HF test image, ZIP download (image, mask, flows,
   cell probability, per-object CSV, README).

## 4. Workstation (WS) session "Model cards redesign" (tutorials/translations lane)

Reachable via SendMessage over Remote Control (bridge:session_0121wwkMTTws3DL6jVK536Ce), else via features/HANDOFF.md.
- Lesson 05: republished r23 (spaCR field, popup_backdrop=off, narration-clock fix
  23ab6d74e in the web player, onset checks). Done.
- Lessons 07, 08: re-recording on the latest nightly; publications in progress
  (07 r26 data already on nightly). Then 04 (both app scenes on GNOME host), then
  14 (waits for makemasks-paint), 20, 43, 24, 26...
- WS holds a tools/ commit 9be053b86 (narration onset/boundary guards) to push after
  "release done".
- Exit SIGSEGV on WS (lesson 08 Templates, GPU moderngl/EGL): a540d62b5 retires
  shading threads before exit; WS to confirm rc 0. spacr.log renderer line on WS:
  `data_art_impulse_lens animation renderer: GPU (moderngl over egl, NVIDIA GeForce RTX 3090 Ti/PCIe/SSE2)`.
- Docs-lane follow-ups: Preferences "Dynamic animation" tooltip and
  AmbientWidget.stop docstring still say the animation pauses for CPU jobs (it now
  slows to 30 fps); translations for 696/697/makemasks strings.

## 5. Maintainer to-dos
- Remove leftover worktree /mnt/firecuda2/Claude/spacr-worktrees/measure-gil-base (auto mode refused).
- ParaScape name on PyPI (`parascape` 0.0.1.0 + trusted publisher); conda-forge pyfixest PR #35101 review; plaque curation -> WS trains plaque v5.
- After release: make GHCR spacr-sif public.

## 6. Today's landed work (for orientation; all on nightly)
Freeze/crash fixes for lessons 05/07/08 (02080e57a, 71a7563af, 5cc78fa29,
1ef114a12, eff787098, 1a2bc3367, a412cd805, fc94e6ead, 15d5f0524, eb5568eb0,
0ceda623f, 4d0927011, 5662574eb, b6abb2854, a540d62b5, 2a22c6efc), CI rounds
(742bfc9de, c921b5dc5, 4c26bb3a8), README phone/desktop redesign (c1e0691ea; main
e0bedd153), features 696-700 added (2cabd4916, c65595df7, bcd75905b).
