# Claude takeover — 2026-10-31

This is the current coordination handoff, requested by the maintainer. Read `spacr/resources/AI/AGENTS.md`, this file, then the relevant current item files. Older HANDOFF.md entries are historical context, not authoritative current branch or CI state. Continue the full Progress List goal: finish every actionable item, get CI green and release 1.5.1.7. Do not claim release or whole-item completion from partial tests.

## Published state and immediate priority

- Main: `92ff3609c4d5db136d9c793828652f60dfcf22d2`, ordinary push succeeded. This corrected release candidate integrates shared CI/native Qt fixes; its parent is `371485fef234311fac0316586575d7ffe5262281`.
- Nightly at handoff preparation: `1c0fd2281c40c6de604a6ba0ae3458f6dde2d9ad`. See the final checkpoint below for the final Nightly tip.
- Protected shared checkout: `/mnt/firecuda2/Claude/repo/spacr`. Only fetch and `git merge --ff-only origin/nightly` here. Preserve the untracked `handoff-note/` worktree; no stash/reset/clean/commits here.
- Root's owned Nightly integration worktree: `/mnt/wd4tb/spacr-worktrees/codex-protocols-coordinator-20261031`, branch `fix/import-safety-coverage-and-icons-20261031`.
- Main owned worktree: `/mnt/wd4tb/spacr-worktrees/readiness-optional-coverage-20261031`, branch `release/cancellable-probe-20261031`, clean published `92ff3609c4`.
- Main's API inputs remain exactly compatible with the actual 13,641-owner/e889 manifest; canonical UI is 9,974 entries and consumer reads are unchanged. Source/artifact closure passed; no generated rebuild was needed. Preserve Main's existing translation-review pin `10eab...` and API pin `e889...`; Nightly has a different `c596...` review successor. Read complete hashes from `tools/media_revision.json` rather than inventing/truncating pins.
- N739 explicitly excludes its new Protocols implementation from 1.5.1.7. New Protocols/Navigator UI/service work is on Nightly, not this Main candidate. Do not silently change that release scope or claim Main CI validates Nightly-only code.

## Live release automation — DO NOT DUPLICATE

Verified processes at handoff preparation:

1. Observer PID **3642648**: `/home/olafsson/anaconda3/bin/python /nas_mnt/data/sync/codex-resume-20261026/final-ci-observer/observe.py --scheduled-next`.
2. Strict release guard PID **220981**, script `/mnt/wd4tb/scratch/ci-cache-report-20261030/kickoff_approved_release.py`, source **92ff3609c4d5db136d9c793828652f60dfcf22d2**, version **1.5.1.7**. It has a twelve-hour deadline and waits on the observer cache, then does fresh source/workflow/tag/PyPI checks before ONE strict release dispatch. It never enables `allow_unfinished_ci`.

Mandatory Main runs:

- Tests: **41207514822**
- Docs: **41207514812**
- Compatibility: **41207514752**

Read `/nas_mnt/data/sync/codex-resume-20261026/final-ci-observer/latest-summary.json` and `latest-<run-id>.json`. Job failures are cached as `job-<job-id>.log`. Targets are in `observer-targets.json`; update this existing observer when a source changes, rather than launching another monitor or forcing cycles.

Guard state/log: `/mnt/wd4tb/scratch/ci-cache-report-20261030/kickoff-92ff3609c4d5db136d9c793828652f60dfcf22d2.{json,log}`. Check actual `/proc/<pid>/cmdline`, not state files alone. At launch, dispatch/watcher flags were absent (false); no release POST or watcher had occurred. Never repeat an ambiguous dispatch. If a mandatory check fails or Main changes, inspect/stop this exact guard before preparing one replacement for a qualified source. Old source371 guard4130309 was stopped; never restart it.

Approved release notes: `/mnt/wd4tb/scratch/release-1517-user-notes-corrected-ci.txt`, SHA256 `438d31b4e3fd4fec354d0e7d5835bf2b0f06f3d4190ecc990acc3c4ddc0f3f29`. Maintainer perspective, no 'at the maintainer’s request'. Guarded watcher `/mnt/wd4tb/scratch/ci-cache-report-20261030/watch_approved_release.py`, SHA256 `6f021e7b9b3b625291a81e403c7e0f4d60f9020038d4e78cc6907966b305acbf`.

The automatic push-triggered release run **41207514979** completed as a no-op; that is NOT proof of publication. Actual completion requires verified release/tag/PyPI 1.5.1.7 and appropriate installer status. The existing guard/watcher handles one real dispatch and installer collection without rebuilding or moving tags.

## What is already delivered — do not redo

N733 Navigator and N739 Protocols are whole-item DONE on Nightly. Their completion table was printed. Current user confirms the Parascape icon is perfect: retain it unchanged.

Nightly includes:

- Actual left module Dock opening; stationary centered composer; unclipped animated cards; inline view/model lifetime repair.
- Transparent white ParaScape vector for dark mode, packaged and shared by dark icon consumers; original/light artwork and license unchanged.
- P control aligned and scaled with AI/Browse, circle only on hover/down; normal foreground guidance/links, gray hover feedback, uniform composer font scaling.
- Protocol reader Black/White/Transparent options, Open Sans Regular, genuine checklist controls and readable PDF footers; native browser and first/dense/last PDF pages reviewed. Do not claim transparent PDF page alpha.
- Required nine-language captions/guides/catalogs and exact source-bound arrival proofs; normal tools were used, passing results retained.
- Production hosted relay footer/navigation guidance and token cap1,200 deployed once: worker deployment `d9f404dd-041b-4b78-874e-8ccf8d704b35`, 100% version `b27c5981-9b3b-4dec-9527-91853a4acd0f`, source service tree `41479100b44fb9c947288576d14c93fcbaec8888`. The real existing app helper made one `/health` GET at13:33:16Z: HTTP200, ok=true, public version0.2.5. No fresh live inference was performed; no further deployment/inference is needed.

Qualified shared fixes now on Main92ff and Nightly1c0fd:

- Exclude collection-only pytest children from executed-outcome receipts.
- Reuse one fresh canonical snapshot across nine strict caption-review validators.
- Bind translation/API/README test expectations to real current reviewed content, with original hashes, rendering assertions and ratchets preserved.
- Invalidate pending Make Masks loads before channel sorting moves source files; controlled native race passes with image/mask bytes and merged arrays exact.
- Guard per-window styling against masked Qt methods; reuse existing dialog hooks.
- Plaque stale-result fixture waits for decoded input AND actual loader-thread retirement. All three cases pass with original stale/cancel assertions. This fixes the actual failed Qt5 shard on old Main371.
- Serial runner is importable by real spawned workers, uses a private Openbox display, and preserves test order/selection, caps and deadlines.
- Native startup/activation/scientific/reporting fixtures retain original behavior assertions.
- Exhaustive nine-catalog API browser PASS54.98s and README accessibility PASS7.81s. Their Main/Nightly source/catalog bindings are exact, so no repeat or rebuild was needed.

## Remaining CI work and exact evidence

The old displayed badge was100466 total/97145 passing, from source `b3389477718d9ffa10f6a025b9d6343317afd568`, run41114409302. Exact census:142 failed-category nodes (89 assertions+53 errors),786 skipped-category (including41 exact XFAILs),2393 notrun-category nodes that also passed in other recorded sessions. Do not call all3321 actual failures.101/142 were an updater cascade already fixed in current source. Evidence under `/mnt/wd4tb/scratch/ci-cache-report-20261030/b338-*`.

Uninterrupted serial run **41193296770**, Main371, job **124514387563**, is terminal FAILURE: it reached21% of38,362 tests, recorded **37 actual failures**, then hit the unchanged10.8GiB RSS guard. It was not OOM-killed.35/37 received qualified source/fixture repairs; the two remaining causes below are not solved. Local artifact is `/mnt/wd4tb/scratch/ci-cache-report-20261030/serial-371/`. Do not rerun the whole local suite or claim the standalone passes explain serial failures.

1. **Borderless menu activation**: owning case `tests/qt/test_695_startup_sequence_and_borderless_window.py::test_the_menu_switch_keeps_a_normal_window_visible_and_placed`. Original product source was restored byte-exact after failed listener/queue experiments; do not ship them. Strong final evidence: after genuine XTest input and desired QAction dispatch, the reframed window has ACTUAL native focus (XGetInputFocus and `_NET_ACTIVE_WINDOW` equal the real targetXID; QGuiApplication.focusWindow matches and QWindow.isActive=true), while QApplication.activeWindow=None and QWidget.isActiveWindow=false for5s. Original geometry and normalGeometry remain70,90,1300,800. Thus investigate QtWidgets activation bookkeeping across native reframe, not an assumed Openbox focus refusal. See `/mnt/wd4tb/scratch/borderless-native-activation-20261031/focus-snapshot.log` and agent's final scratch handoff. No fake-activeWindow test waiver, arbitrary retry/timer or weakened deadline is qualified.
2. **Overview comparison**: historical TypeError followed a missing comparison signature; isolated test passes but does not reveal cause. New failure-only test diagnostics preserve actual measurement status, sanitized worker errors, read-only fov count/schema and request/generation state. Original assertions/waits unchanged. Qualified eaa9a74b64 (PASS12.62s), integrated as03ee31544f. Not a product fix or historical-cause claim.
3. **Serial RSS**: widget-count drops do not prove cleanup fixes RSS. Existing file_end measurement precedes the next test's DeferredDelete flush; prior high widget counts can be transient. Qualified diagnostic bc0c230c12 (PASS0.94s), integrated94d5a37323, records primitive widget/top-level class counts and currentRSS immediately after the EXISTING setup flush, once per file. No extra flush/deletion/GC/malloc-trim/cap increase. Use the next exact-source hosted serial run to find actual retention, rather than repeating the entire suite locally. Log `/tmp/spacr-postflush-phase-final-20261031.log`.
4. **Older hosted allocator abort**: Nightly0808 run41192988161 Coverage11 job124513698142 aborted during pytest-qt post-call events after `test_qc_field_browser_ratchet_edges_20261024.py::test_wizard_restores_saved_sources_and_reports_a_real_worker_refusal`. Four controlled same-worker cases pass; cause is unproven. Do not claim a fix, infer native cause from Python stacks, or repeat the previously tool-rejected GDB/core task. Keep it distinct from Main's Plaque failure and serial RSS.
5. **Compact caption ownership**: four generated UI strings belong in compact `_ROWS`: Mask folder and three long AI-preference captions. The CI agent prepared exact36 existing reviewed targets; use `/mnt/wd4tb/scratch/ci-cache-report-20261030/compact-four-reviewed-targets.json`, row order sv,de,es,zh_CN,pt,hi,ko,is,fr. Worktree `/mnt/wd4tb/spacr-worktrees/ci-compact-caption-ownership-20261031`. Extend the existing normal producer for guarded computed compact pins, preserve old rows, then normally archive only those four generated records per locale and regenerate/audit nine runtimes/arrival proof. No manual count/SHA, ceiling raise, generated merge, fabricated translation or adoption of Nightly's pin on Main. Final checkpoint below records what was completed before handoff.

## Progress List and other ownership

Canonical `features/FEATURE_LIST.md` and numbered `features/feature_list/` files. Preserve IDs/history and new/future lifecycle. Whole-item completion alone triggers the full progress table: Item | Percent done | Time left | Owner | Description. Newly completed ✅100% first; omit older100% items. Implementation + reasonable focused validation + normal artifacts + publication qualify delivery; no new human acceptance gate.

Remaining broader items include N43/F288 CI+coverage; F370/F404 GPU rescoring; N441 vendor/data coverage; N449/F470 ground-truth annotations; F534 VisiumHD; F536 wound accuracy (not a release blocker); F538 scientific calibration; F559 independent QC labels; F560 public benchmark provenance; F561 workstation benchmark; N676 Work-owned tutorials; N678 broader factual docs/API review; N710 media/history prerequisites; F711 final Timepose v2 (not ready); N729 Mac-owned work; N740/N741 record-only; N745 original cluster cause. Consult dated notes; do not invent data, acceptance or completion.

N678 Pipeline Graph follow-up was prepared during handoff: actual Copy Graphviz only copies DOT to QApplication.clipboard, writes no figures. Correct false figures metadata using a truthful transient clipboard output while retaining nonempty-output validation. English/source peer accepted; final checkpoint records commit/drafts. Do not start competing runtime producers while compact4 owns the lease. Preserve Work-owned tutorial banks and authored lessons; prior Lineage changes exposed preexisting stale tutorial-map pins and were not permission to rebind them blindly.

Single latest GitHub issue lookup returned **0 open issues**. If new issues are solved, explain cause, fix commits, branches/release status and validation BEFORE closing them. Do not repeatedly poll an empty list.

## Working rules and environment

- No Home GPU, no livecell/cellposeTIME jobs. Chrome/graphics processes NEVER count as compute or block the GPU queue; inspect each actual scheduler, including old worktrees, before any new GPU run.
- Reuse retained context/agents; no repeated broad scoping or invented tasks. Freeze/commit one item end to end. Normal generators once against final source; rerun only affected failures. Never weaken tests/raise ratchets.
- New owned worktrees `/mnt/wd4tb/spacr-worktrees/`; scratch `/mnt/wd4tb/scratch/`. No new product modules, no `#` comments under spacr/, user strings through tr(). Descriptive commits without item-number prefixes/suffixes or AI trailers.
- Normal pushes Nightly only; current release Main exception is authorized. Never force-push. Shared checkout only fast-forward.
- CPU tests: `CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen tools/run_capped.sh 8G python -m pytest`, at most-n4; related tests only. Native checks use isolated Xvfb/Openbox, never the user's desktop.
- Real native Qt environment: `/mnt/wd4tb/scratch/qt-pdf-native-firstuse-20261022/qt6112-env/bin/python` WITH `PYTHONPATH=/mnt/wd4tb/scratch/uninstall-slot-qt6113-20261027`; actual runtime is Qt/PySide6 **6.11.3**, despite the base directory name. Preserve any other necessary PYTHONPATH entries.
- The Codex agents were asked to finish/checkpoint current work and stop starting new scopes for this handoff. Background observer/strict release automation are intentionally retained. Recheck their actual live state before taking ownership; do not assume old PIDs are still alive.

## Final checkpoint

The final Nightly handoff commit contains the qualified diagnostic commits **03ee31544f** and **94d5a37323** plus this document. Obtain its exact tip from `git ls-remote origin refs/heads/nightly`; Main remains92ff. These diagnostic-only additions do NOT solve the historical comparison failure or RSS growth. No extra product/API/catalog changes and no local full-suite replay.

**Compact4 checkpoint is NOT ready to integrate/publish:** clean commit `f92aa41efc45ea18822a01c9e9f1a8ef609efd3c`, worktree `/mnt/wd4tb/spacr-worktrees/ci-compact-caption-ownership-20261031`, base1da. Five producer guard controls pass1.25s and peer review accepts the guarded tool design. However, the REAL normal pin refresh correctly REFUSED: discovery yields287 versus old pin284, and no subset of the four approved additions restores the old fingerprint. Pins remain UNWRITTEN, archives untouched, no runtime/catalog producer is running; the complete compact closure is NOT qualified. Investigate the old compact pin's real source delta, last changed in76dcb45330, rather than manually repinning. Existing literal rows preserve all36 reviewed c596 targets and all old row bytes. Preserve this exact source checkpoint and use the CI agent's scratch handoff/proofs.

**Pipeline Graph checkpoint is source/drafts only:** clean `ee0a235e108144cb4869c3ec5a7e5f8aefabd61b`, `/mnt/wd4tb/spacr-worktrees/codex-n678-pipeline-clipboard-20261031`, base1c0fd. Only canonical workflow JSON and N678 note are committed. Draft18 runtime+9 guide targets: `/mnt/wd4tb/scratch/n678-pipeline-clipboard-20261031/english-and-nine-drafts.json`. English/source peer accepted. No generator, external review, runtime or guide closure has run. Do NOT publish it as delivered; normal tools follow compact baseline reconciliation and exclusive producer ownership. See the owner's scratch handoff for exact commands and retained tutorial-consumer limitations.

**Borderless investigation is frozen without a fix:** `/mnt/wd4tb/scratch/borderless-native-activation-20261031/HANDOFF.txt`. Final corrected snapshot finished with test failure9.25s and private Xvfb/Openbox cleanup complete; no owned experiment is still running. Product app.py and owning test were restored exactly to their pre-experiment HEAD. Failed actuator patch is only scratch `unqualified-input-fixture.patch`; failed Expose/queue product prototypes were removed. Only an uncommitted N43 note remains in the investigation worktree; preserve or deliberately reconcile that note, do not mistake it for product work. The public Qt/X11 focus snapshot is the concrete new diagnostic evidence, not a passing test.

**No Codex generator/test runner should be assumed live from agent messages.** Current workers were asked to finish/checkpoint and stop new scopes; only the explicitly retained observer/strict release automation continue. Recheck the live release state and newly triggered Nightly runs at takeover. The older automatic Nightly1da serial run41206297109 was preserved because no replacement existed at that observation; inspect before cancelling or dispatching anything.
