# Handoff, 2026-09-22 — Claude to the next session

Written when the Claude session that did the work below ran out of context.
Everything here is on `origin/nightly` at `d5981528c` unless it says otherwise.
Read this, then `features/325_two_sessions_one_repo_working_protocol.temp`
from the END (the last three blocks are the Codex lane, the fresh-session
orientation, and this work), then `features/00_INDEX.txt`.

## Released

**spaCR 1.5.0.9** is on PyPI, with the GitHub release, installers for the
three platforms and the container images. `main` carries it. The release runs
before it failed twice, both causes fixed with tests (a packaging check that
read prose as code, and a deck restamp that needed Pillow in a step that has
none). `packaging/release.py bump <version>` does the bump and restamps the
README deck's title slide; pushing `setup.py` to `main` starts the release
workflow.

## What is RUNNING right now

| Thing | State | Where |
|---|---|---|
| Item 473, Make Masks detection methods | an agent is building it | worktree `scratchpad/wt-473`, branch `make-masks-methods-0922` (the only agent branch left on purpose) |
| Item 372, full-plate OPS run | wells A1 and A3 done, B1 started 07:12 | driver `tools/run_ops_plate.py`, results under `/mnt/wd4tb/spacr_testdata/ops_plate_run/`, GPU busy |

If the 473 agent is gone, its branch holds whatever it had; rebase it on
nightly before trusting it. The OPS run writes per-well JSON and a log; item
372 wants the per-well table appended to its file when it finishes, and B2's
low library-exact rate (0.666 against ~0.72) explained if the data allows.

## Finished today, for context

* **Item 471 (fits a laptop) — DONE.** GUI scale 10–200 % applied LIVE (a
  scaling layer over Qt's setters, `spacr/qt/gui_scale.py`), every scale
  change asks Keep or Revert with a 15 s auto-revert, each live preview has
  its own slider, Ctrl+Alt+0 resets. The shell and ~200 figures/tables/sections
  collapse and drag through one primitive,
  `spacr/qt/widgets/collapsible_splitter.py` (`FoldSection`, `add_section`,
  `add_pane(mode=EDGE)`, `lock_folded_to_bottom`); a folded thing locks to the
  BOTTOM. The Live switch rides on the preview card. The handle is a dark grey
  tab with a stroked chevron that turns accent when hovered or held.
* **Items 284/380 — not done, much faster.** Closed settings categories are
  built when opened and prebuilt in idle slices; Classify's worst stall
  877 → 697 ms, first open of a category 55–92 ms → ~25 ms. What remains is
  the build and styling of what IS visible, and quiet-machine, macOS and
  Windows numbers.
* **Item 470 — crops cut.** 585,311 ground-truth crops
  (`/mnt/wd4tb/af3/projects/cross_channel_models/ground_truth_crops/`), one
  per label, verified against the labels and each plate's png_list. A seeded
  1,000-crop sample per object type is ready to annotate under
  `annotation_sample/<type>` (Annotate source, annotation column `real`).
  WAITING ON THE MAINTAINER to judge them; then a real/not-real classifier.
* **Item 469 — done** (figure reader in its own environment, checked for real).
  **Item 446 — done** (Cellpose 3 real run: cyto2 F1 0.549 against stock
  Cellpose-SAM 0.491). **Item 407 — done** (magnifier's last four points).
  **Item 424 — measured and built** (duplicates, conflict flag, scale, PDF
  text layer); a person must still check the lawn-strip figures.
* **Item 426 — v1 trained, not promoted.** Timeflows v1 ties the IoU stitcher
  on the held-out movies because those cells barely move; it needs held-out
  movies with small fast cells to show anything. Model and data:
  `/mnt/wd4tb/af3/projects/live_cell/timeflows/`.
* **Model Zoo:** SpotNet (DeepCell) added as a spot-detection backend — NON-
  COMMERCIAL licence, needs `DEEPCELL_ACCESS_TOKEN`, installs only on Python
  3.7–3.10, and its `trackpy`/`deepcell` pins were needed because
  deepcell-spots pins neither. Backends now explain where they are chosen
  instead of greying "Use this model" silently.

## Open items, not started

* **472** (GitHub #130, "what to do after installation"): CODEX owns the
  tutorials, walkthroughs and API prose from one map,
  `spacr/resources/module_workflows.json` (Codex announced that name). Claude's
  part 5 is DONE: a module that cannot open now says so, and Home has
  "Start a sample project…" which reads that map when it exists
  (`spacr/qt/widgets/sample_project.py`). Still owed there: the chosen
  pathway's walkthrough one click away, once the walkthroughs exist.
* **473** (in progress, above): the six remaining organelle detection methods
  in Make Masks plus a pre-detection chain (contrast, background, denoise,
  sharpen, morphology, split) and a raw-vs-enhanced compare. NOT to be wired
  into the Mask module without asking the maintainer.
* **474** (future): one page per organism — Toxoplasma, Plasmodium, Candida —
  each with a description, links and eight tiles, "Coming soon" on the unbuilt
  ones. The proposed module lists are in the item for the maintainer to confirm.
* **475** (future): where SpotNet would earn its place (OPS spot detection),
  measured against spaCR's own detector.
* **Known, pre-existing:** the Regression figures card can overlap the Console
  heading at 768 px tall (its 560 px minimum).

## Waiting on the maintainer

The 470 annotation; 424's figure checks; the README deck rebuild (built and
checked in the scratchpad, NOT published — he stopped it); 370/404/405
curation, 449, 434, 444/435, 453 (needs sudo), 425, 59, and 416 on his own
machine.

## Rules that cost time to learn

Every python/pytest through `tools/run_capped.sh <cap> …`, never the whole
suite, `QT_QPA_PLATFORM=offscreen` for Qt; probes use their OWN scratch HOME
and XDG_CONFIG_HOME; GPU jobs only after 20 idle minutes, one at a time; NAS
through `tools/nas_guard.sh`; no `#` comments in `spacr/**`; user-visible
strings through `tr()` and no catalog edits (that is the translation lane);
stage explicit paths, rebase before every push, commit as the maintainer with
no AI attribution.

## Branches

`main` and `nightly` only, at the maintainer's instruction (2026-09-22). The
eight old `wip/*` remote branches were tagged `archive/wip-<name>` before
deletion, so their commits are still reachable; 174 local branches and 92
stale worktrees were removed.
