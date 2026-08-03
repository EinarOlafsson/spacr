# Execution lists

73 items. **List 0 runs alone first**, then Lists 1–6 run in parallel,
5 agents each = 30 concurrent, which is the stated ceiling.

Rules this partition obeys:

- **A dependency lives in the same list as its dependent, above it.** The
  only exception is List 0, which is a prerequisite for everything and
  therefore runs to completion before any other list starts.
- **One list owns each contended file.** Where two lists wanted the same
  file, either they were merged (Lists 3+4 of the earlier draft, which both
  wanted `annotate.py`) or List 0 adds a hook so neither has to edit it
  (`app.py`, `settings.py`, `theme.py`, `measure.py`).
- **Order within a list is dependency order.** Items with no dependency on
  each other may run concurrently up to the 5-agent cap.

Legend: `C/N/U/V/L/S` = the 2026-08-02 selection · `B` = JMP/Napari list ·
`Z` = backlog carried from earlier this session.

---

## List 0 — Foundations · 5 items · runs alone

Nothing else starts until this list is green. It is all plumbing, no
features, and it exists so Lists 1–6 never touch the same file.

| # | Item | Delivers |
|---|---|---|
| 0.1 | Registration seams | `APPS` becomes a registry with `register_app()`; modules ship their own settings defaults; `register_widget_qss()` hook. **Frees `app.py`, `settings.py`, `theme.py`.** Two new sections so 7 new modules don't blow `MAX_APPS_PER_SECTION = 13` |
| 0.2 | `V1` Linked-selection contract | Finish the publish/subscribe API every view consumes. Also defines `open_objects(keys, reason)`, the slot Lists 2 and 3 call instead of editing each other's screens |
| 0.3 | `L1` `L2` Ports + artifact registry | `spacr/ports.py`, `spacr/artifacts.py`. Provenance for every output; query API published **before** implementation so Lists 1–6 code against it |
| 0.4 | `S7` `S5` `S9` Run context | `spacr/runctx.py` — one run id through every log line; a global seed that reaches numpy/torch/cellpose/sklearn; `on_error: stop\|skip\|retry`. Reaches into the science modules, so it goes alone |
| 0.5 | `measure.py` extension hooks | A preprocessing hook and a region-filter hook. **Frees `measure.py`**, which List 3 (ROI→Measure) and List 4 (illumination) both need |

Order: 0.1 ∥ 0.2 → 0.3 → 0.4 → 0.5.

---

## List 1 — Provenance, projects & reporting · 12 items

Owns: `artifacts.py` consumers, `core.py` run completion, new project-level
screens, `spacr/report.py`.

| # | Item |
|---|---|
| 1.1 | `L3` Auto-chaining — Measure defaults to where Mask wrote, Classify to where Measure wrote |
| 1.2 | `L6` Stale-output detection — mark downstream results stale when an upstream setting changed |
| 1.3 | `L5` "Continue to next step" — a finished run offers its successor, pre-filled |
| 1.4 | `N4` Project browser — every project: stage, size, last run, what's stale *(needs 1.2)* |
| 1.5 | `N7` Run comparison — settings diff, count diff, hit-list diff. `settings_diff.py` exists at 36% |
| 1.6 | `N8` Data manager — archive, prune intermediates, disk use per project |
| 1.7 | `L4` Pipeline graph view — the DAG of what produced what *(needs 1.1, 1.2)* |
| 1.8 | `B19` Macro recorder — every GUI run emits the equivalent Python script |
| 1.9 | `Z3` Verify PDF figure output actually works |
| 1.10 | `Z13` AnnData export |
| 1.11 | `B4` Prediction profiler — interrogate a fitted model interactively |
| 1.12 | `N6` **Methods + results exporter (AI)** — structured run digest in, prose out. The model never sees raw data; every number comes from the digest *(needs 1.8, and 0.3 + 0.4)* |

---

## List 2 — Tables, plots & the data platform · 10 items

Owns: `qt/widgets/` plotting, `spacr/plot.py` additions. The JMP half.

| # | Item |
|---|---|
| 2.1 | `V7` **Graph Builder** — drag columns onto x / y / colour / facet. Everything below builds on its axis + facet engine |
| 2.2 | `V8` PCA / multivariate platform + loadings biplot |
| 2.3 | `B6` Tabulate / pivot builder |
| 2.4 | `B7` Column formula editor |
| 2.5 | `V5` Small multiples / trellis — shared axes, two-way faceting, empty panels drawn empty, per-panel n *(needs 2.1)* |
| 2.6 | `V2` Gate editor — draw thresholds on a histogram or 2-D scatter; the gate becomes a filter *(needs 2.1)* |
| 2.7 | `V4` Feature explorer — per-feature distributions split by class, ranked by separation *(needs 2.1, 2.2)* |
| 2.8 | `B8` Robust outlier detection |
| 2.9 | `B10` Dose–response / EC50 fitting |
| 2.10 | `B9` Control charts across a campaign |

---

## List 3 — Images, annotation & classification · 13 items

Owns: `qt/screens/annotate.py`, `deep_spacr.py`, `classifier_evaluation.py`,
the new layer viewer. The napari half, merged with the classify loop because
both wanted `annotate.py`.

| # | Item |
|---|---|
| 3.1 | `B11` **Layer model viewer** — image / labels / points / shapes as stacked layers. 3.2–3.6 are all layers on it |
| 3.2 | `B14` ROI / shapes layer honoured by Measure *(needs 3.1, and 0.5)* |
| 3.3 | `B13` Points layer for manual counting *(needs 3.1)* |
| 3.4 | `B12` `C7` Interactive label brush + **timelapse track curation** — join, split, delete tracks by hand *(needs 3.1)* |
| 3.5 | `B15` Orthogonal views + dimension sliders *(needs 3.1)* |
| 3.6 | `B16` Synchronised comparison grid *(needs 3.1)* |
| 3.7 | `C10` Per-class accuracy — computed at `deep_spacr.py:370`, currently thrown away |
| 3.8 | `C9` Model cards — dataset, class balance, split rule, held-out metrics, spaCR version, beside the weights |
| 3.9 | `C5` Annotation coverage — per class, per well, per plate |
| 3.10 | `C4` **Closed active-learning loop** — retrain from inside Annotate, re-rank the queue, learning curve, stopping rule, round provenance. The queue itself already exists at `annotate.py:728` *(needs 3.7, 3.8)* |
| 3.11 | `C8` Confusion-driven relabelling — click a matrix cell, get those crops sorted by confidence; high-confidence errors and low-confidence errors listed separately *(needs 3.9, and 0.2)* |
| 3.12 | `V3` Image-linked scatter — hover a point see the crop, click to open it *(needs 3.1, and 0.2)* |
| 3.13 | `V9` Lineage / relationship view — cell → nucleus → pathogen *(needs 0.2)* |

---

## List 4 — Science: illumination, sequencing, power, design · 8 items

Owns: `measure.py` science path, `sequencing.py`, `power_*.py`, new
`illumination.py`.

| # | Item |
|---|---|
| 4.1 | `N2` **Illumination correction / flat-field** — the only item that changes your numbers rather than your view; root cause of the edge effects `plate_view` detects *(needs 0.5)* |
| 4.2 | `C1` Measure QC banner — `seg_qc`'s verdict shown when you open Measure. **Informs, does not block** |
| 4.3 | `Z11` Diameter estimator wiring |
| 4.4 | `C6` Barcode QC — reads per well, collision rate, unmapped fraction. Sweep driven by **target gRNAs per well**: you state the target, the module derives the threshold and sweeps around it |
| 4.5 | `N1` **Power / Design GUI** — `power_simulate.py` and `power_model.py` are written and tested; this is the app surface |
| 4.6 | `Z14` spaCRPower gaps — sequencing error, and well dropout from too few imaged cells *(needs 4.5)* |
| 4.7 | `N3` Experiment designer — plate layout, controls, replicates, exported for the pipeline to read *(needs 4.5, 4.6)* |
| 4.8 | `N5` QC dashboard — aggregates `seg_qc`, `plate_qc`, agreement, leakage, dtype into one verdict *(needs 4.1, 4.2, 4.4)* |

---

## List 5 — GUI shell & visual backlog · 15 items

Owns: `qt/theme.py` consumers, `settings_model.py`, `preferences.py`, the
live-preview panels. Everything you flagged visually, plus the UX items.

| # | Item |
|---|---|
| 5.1 | `Z1` Zoom must reach tab text, the right-hand home panel, tooltips, and text buttons (Live, AI) |
| 5.2 | `Z7` Tooltip animations — animation right of text, text top-aligned, text box the same width as the square, content auto-zoomed to 70–80% (measured median today: 63.9%; 72 of 94 below 70%), plus an off switch |
| 5.3 | `Z2` Tooltips for every settings category, shown under the run/stop panel |
| 5.4 | `Z8` Field-fade gradient — 0→100% left to right, accelerating, outlines included, container not text; on by default with a preference to disable |
| 5.5 | `Z9` Container styling batch — Align & Stitch "press Plan" region, Plate Viewer "press Render" container, Model Compare A/B, Training Runs area, Classifier Evaluation area + tabs, Run History tabs + container |
| 5.6 | `Z10` System panel — RAM, GPU and VRAM bar regions are not subject to opacity; CPU is |
| 5.7 | `Z4` Live view — FOV and channel dropdowns left of "choose image", styled like the Live button; "choose image" restyled to match; in mask, measure and the rest |
| 5.8 | `Z5` Mask outline colour stuck green — automatic should be random |
| 5.9 | `Z6` DNA rain — settings behind the DNA button, random-colour option |
| 5.10 | `Z12` Cellpose model list pulled from the API rather than hard-coded |
| 5.11 | `U1` **Settings search + progressive disclosure** — 585 keys, 205 on mask alone, no search today |
| 5.12 | `U2` Recipes — save a named settings bundle, reuse and share it |
| 5.13 | `C3` Feature dictionary panel — `feature_dict.py` exists and is export-only |
| 5.14 | `U4` Per-module first-run walkthrough, re-runnable |
| 5.15 | `U5` Keyboard shortcut overlay (`?`) + command palette for settings |

---

## List 6 — Stability, testing & debuggability · 10 items

Owns: `qt/bridge.py`, `tests/`, new `doctor.py` and `crashreport.py`.
Fully isolated from Lists 1–5 — this list could start during List 0.

| # | Item |
|---|---|
| 6.1 | `S11` **Kill the flake sources** first — cross-test QSettings pollution, the `test_report_screen` flake. A suite with known-red tests is one nobody reads, and 6.5/6.8 are meaningless until it is green |
| 6.2 | `S1` Qt worker teardown — the 137-thread live-lock and the shard SIGSEGV, one investigation, starting at `bridge.py` |
| 6.3 | `Z15` `F34` residual on the non-resume path in `utils._merge_and_save_to_database` (currently a strict xfail) |
| 6.4 | `S2` `spacr doctor` — environment, deps, GPU, CUDA, database integrity, common misconfigurations |
| 6.5 | `S3` Golden-output regression — a tiny fixture dataset end to end with checked-in numbers *(needs 6.1)* |
| 6.6 | `S4` Property-based tests for key parsing — every `prc` bug this session would have been caught by one |
| 6.7 | `S8` DB contract tests — `DB_CONTRACT_AUDIT.jsonl` found real bugs and was never turned into tests |
| 6.8 | `S10` Mutation testing on the science core *(needs 6.1)* |
| 6.9 | `S12` Perf regression guard — frame cost, measure throughput, memory ceiling as asserted numbers |
| 6.10 | `S6` Crash reporter — log, settings, versions, last run in one attachable file *(needs 6.2, 6.4)* |

---

## Adding items during execution

Routing rule, applied in this order:

1. **Dependency closure** — if the new item depends on something, it goes in
   that item's list, positioned below it. If it depends on items in two
   different lists, it goes in the list holding the *later* dependency and
   waits.
2. **File ownership** — otherwise it goes to the list that already owns the
   file it will edit. This is what actually prevents conflicts.
3. **Load** — only if 1 and 2 leave it free does it go to the shortest list.

Position within a list: above anything that will depend on it, below
anything it depends on, otherwise at the end. A bug found during execution
goes to the top of its list, not the end.

---

## Not in any list

- **Tutorials v2** — Codex owns tutorials in a separate session. The design
  doc (`proposals/TUTORIALS_V2.md`) is written and committed; implementation
  is deliberately out of scope here.
- **Not selected:** `U6` undo for destructive actions, `U7` explain-this-run,
  `U9` units + provenance, `V6` plate over time, `L7` project manifest.
  `U6` is worth revisiting — F34/F35 exist precisely because deletes are
  irreversible.
