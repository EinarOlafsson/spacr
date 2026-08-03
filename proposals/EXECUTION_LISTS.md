# Execution lists

99 items — the 2026-08-02 selection, the JMP/Napari list, the open audit
items, and everything asked for in chat that is not yet built.

**Lists 0, 5 and 6 start immediately.** Lists 1–4 wait for List 0.

Legend: `C/N/U/V/L/S` = the 2026-08-02 selection · `B` = JMP/Napari ·
`A/F` = audit feature/stability item · `Z` = chat backlog.
**L0** column: ✓ = blocked on List 0, — = not blocked.

---

## List 0 — Foundations · 5 items · runs first

Pure plumbing. Exists so Lists 1–4 never edit the same file.

| # | Item | Delivers |
|---|---|---|
| 0.1 | Registration seams | `APPS` → a registry with `register_app()`; modules ship their own settings defaults; `register_widget_qss()` hook. **Frees `app.py`, `settings.py`, `theme.py`.** Two new sections — 7 new modules would blow `MAX_APPS_PER_SECTION = 13` |
| 0.2 | `V1` Linked-selection contract | Finishes publish/subscribe; defines `open_objects(keys, reason)`, the slot Lists 2 and 3 call instead of editing each other's screens |
| 0.3 | `L1` `L2` Ports + artifact registry | `spacr/ports.py`, `spacr/artifacts.py`. Provenance per output; API published before implementation |
| 0.4 | `S7` `S5` `S9` Run context | `spacr/runctx.py` — one run id per log line; a seed reaching numpy/torch/cellpose/sklearn; `on_error: stop\|skip\|retry` |
| 0.5 | `measure.py` extension hooks | Preprocessing + region-filter hooks. **Frees `measure.py`** for List 3 (ROI→Measure) and List 4 (illumination) |
| 0.6 | Wire the existing views to `LinkedView` | UMAP, db_browser, and plate_view (which has bespoke `_link` wiring predating the mixin). Added after 0.2 landed — the contract is only worth what consumes it *(0.2)* |

Order: 0.1 ∥ 0.2 ∥ 0.3 ∥ 0.5, then 0.6 after 0.2, then 0.4 last (it touches
what the others touch).

**0.2 landed** as `88cbabe7` — both modules at 100% statement *and* branch
coverage, 90 tests green. The contract other lists code against:
`open_objects(keys, *, reason, kind, source, timelapse, context)` on the
caller side, `register_object_opener(kind, fn)` on the destination side, and
a `LinkedView` mixin with built-in echo suppression. Opening deliberately
does **not** move the shared selection — that would wipe the lasso the
request came from.

---

## List 1 — Provenance, projects, I/O & reporting · 15 items

| # | Item | L0 |
|---|---|---|
| 1.1 | `L3` Auto-chaining — Measure defaults to where Mask wrote, Classify to where Measure wrote | ✓ |
| 1.2 | `L6` Stale-output detection — mark downstream stale when an upstream setting changed | ✓ |
| 1.3 | `L5` "Continue to next step" — a finished run offers its successor, pre-filled | ✓ |
| 1.4 | `N4` Project browser — stage, size, last run, what's stale *(1.2)* | ✓ |
| 1.5 | `N7` `A20` Run comparison — settings, count and hit-list diffs. `settings_diff.py` is at 36% | ✓ |
| 1.6 | `N8` Data manager — archive, prune intermediates, disk use per project | ✓ |
| 1.7 | `L4` Pipeline graph view — the DAG of what produced what *(1.1, 1.2)* | ✓ |
| 1.8 | `B19` Macro recorder — every GUI run emits the equivalent Python script | ✓ |
| 1.9 | `Z3` Verify PDF figure output actually works | — |
| 1.10 | `A9` `Z13` AnnData / scanpy export | — |
| 1.11 | `A8` OME-Zarr read and write | ✓ |
| 1.12 | `A17` OMERO import / export | ✓ |
| 1.13 | `A19` A real hit-list deliverable | ✓ |
| 1.14 | `B4` Prediction profiler | ✓ |
| 1.15 | `N6` **Methods + results exporter (AI)** — structured run digest in, prose out; the model never sees raw data, every number comes from the digest *(1.8, 0.3, 0.4)* | ✓ |

## List 2 — Tables, plots & the data platform · 10 items

| # | Item | L0 |
|---|---|---|
| 2.1 | `V7` `B3` **Graph Builder** — drag columns onto x / y / colour / facet. Everything below uses its axis + facet engine | ✓ |
| 2.2 | `V8` `B5` PCA / multivariate + loadings biplot | ✓ |
| 2.3 | `B6` Tabulate / pivot builder | ✓ |
| 2.4 | `B7` Column formula editor | ✓ |
| 2.5 | `V5` Small multiples / trellis — shared axes, two-way faceting, empty panels drawn empty, per-panel n *(2.1)* | ✓ |
| 2.6 | `V2` Gate editor — draw thresholds on a histogram or 2-D scatter; the gate becomes a filter *(2.1)* | ✓ |
| 2.7 | `V4` Feature explorer — distributions split by class, ranked by separation *(2.1, 2.2)* | ✓ |
| 2.8 | `B8` Robust outlier detection | ✓ |
| 2.9 | `B10` Dose–response / EC50 fitting | ✓ |
| 2.10 | `B9` Control charts across a campaign | ✓ |

## List 3 — Images, annotation & classification · 14 items

Owns `annotate.py`, `deep_spacr.py`, `classifier_evaluation.py`.

| # | Item | L0 |
|---|---|---|
| 3.1 | `B11` **Layer model viewer** — image / labels / points / shapes as stacked layers. 3.2–3.6 are layers on it | ✓ |
| 3.2 | `B14` ROI / shapes layer honoured by Measure *(3.1, 0.5)* | ✓ |
| 3.3 | `B13` Points layer for manual counting *(3.1)* | ✓ |
| 3.4 | `B12` `C7` Interactive label brush + **timelapse track curation** — join, split, delete tracks by hand *(3.1)* | ✓ |
| 3.5 | `B15` Orthogonal views + dimension sliders *(3.1)* | ✓ |
| 3.6 | `B16` Synchronised comparison grid *(3.1)* | ✓ |
| 3.7 | `A18` Napari bridge for mask correction *(3.1)* | ✓ |
| 3.8 | `C10` `A6` Per-class accuracy — computed at `deep_spacr.py:370`, thrown away | — |
| 3.9 | `C9` `A14` Model cards — dataset, class balance, split rule, held-out metrics, version, beside the weights | — |
| 3.10 | `C5` Annotation coverage — per class, per well, per plate | — |
| 3.11 | `C4` `B18` **Closed active-learning loop** — retrain inside Annotate, re-rank, learning curve, stopping rule, round provenance. The queue exists at `annotate.py:728` *(3.8, 3.9)* | — |
| 3.12 | `C8` Confusion-driven relabelling — click a matrix cell, get those crops by confidence; high-confidence and low-confidence errors listed apart *(3.10, 0.2)* | ✓ |
| 3.13 | `V3` Image-linked scatter — hover a point see the crop, click to open *(3.1, 0.2)* | ✓ |
| 3.14 | `V9` `B20` Lineage view — cell → nucleus → pathogen *(0.2)* | ✓ |

## List 4 — Science: illumination, sequencing, power, design · 11 items

| # | Item | L0 |
|---|---|---|
| 4.1 | `N2` `B17` **Illumination correction / flat-field** — the only item that changes your numbers rather than your view; root cause of the edge effects `plate_view` detects *(0.5)* | ✓ |
| 4.2 | `C1` Measure QC banner — `seg_qc`'s verdict shown on opening Measure. **Informs, does not block** | — |
| 4.3 | `A3` `Z11` Diameter estimator wiring | — |
| 4.4 | `C6` Barcode QC — reads per well, collision rate, unmapped fraction. Sweep driven by **target gRNAs per well**: you state the target, the module derives the threshold and sweeps around it | — |
| 4.5 | Verify every regression type works with sane settings — mixed model, logistic, hinge, lasso, GLM, quantile, beta | — |
| 4.6 | `A12` Empirical-Bayes batch correction (ComBat) beside `center` / `zscore` | — |
| 4.7 | `N1` `A1` **Power / Design GUI** — `power_simulate.py` and `power_model.py` are written and tested; this is the app surface | ✓ |
| 4.8 | `Z14` spaCRPower gaps — sequencing error, and well dropout from too few imaged cells *(4.7)* | ✓ |
| 4.9 | `N3` Experiment designer — plate layout, controls, replicates, exported for the pipeline *(4.7, 4.8)* | ✓ |
| 4.10 | `A15` 3-D segmentation end to end | ✓ |
| 4.11 | `N5` QC dashboard — aggregates `seg_qc`, `plate_qc`, agreement, leakage, dtype into one verdict *(4.1, 4.2, 4.4)* | ✓ |

## List 5 — GUI shell & visual backlog · 22 items · starts now

| # | Item | L0 |
|---|---|---|
| 5.1 | `Z7` **Tooltip animations** — animation to the **right** of the text, text top-aligned, text box the same width as the square, content auto-zoomed to 70–80% (measured median 63.9%; 72 of 94 below 70%), plus an off switch | — |
| 5.2 | `Z6` **DNA rain settings** — move them behind a DNA button beside the AI button in map_barcodes; add the random-colour option | — |
| 5.3 | `Z4` Live view — FOV and channel dropdowns left of "choose image", styled like the Live button; "choose image" restyled to match; in mask, measure and the rest | — |
| 5.4 | `Z5` Mask outline colour stuck green — automatic should be random | — |
| 5.5 | `Z10` System panel — RAM, GPU and VRAM bar regions are not subject to opacity; CPU is | — |
| 5.6 | `Z2` Tooltips for every settings category, shown under the run/stop panel | — |
| 5.7 | `Z8` Field-fade gradient — 0→100% left to right, accelerating, outlines included, container not text; on by default, preference to disable | — |
| 5.8 | `Z1` Zoom must reach tab text, the right-hand home panel, tooltips, and text buttons (Live, AI) | — |
| 5.9 | `Z9` Container styling batch — Align & Stitch "press Plan", Plate Viewer "press Render", Model Compare A/B, Training Runs area, Classifier Evaluation area + tabs, Run History tabs + container | — |
| 5.10 | `Z12` `A5` Cellpose model list read from the API, not hard-coded | — |
| 5.11 | Silence the cellpose `Sparse invariant checks` warning at launch | — |
| 5.12 | `F9` **Repair the demo set** | — |
| 5.13 | `F10` Wire the three missing apps into the home generators | — |
| 5.14 | `A10` Fixed-alphabet multi-select for `train_channels` | — |
| 5.15 | `A11` Live Preview everywhere it helps | — |
| 5.16 | `A4` Curated settings layouts for the 25 uncovered apps | — |
| 5.17 | `A16` Promote or retire the 15 alpha apps | — |
| 5.18 | `U1` `A2` **Settings search + progressive disclosure** — 585 keys, 205 on mask alone, no search today | — |
| 5.19 | `U2` Recipes — save a named settings bundle, reuse and share it | — |
| 5.20 | `C3` Feature dictionary panel — `feature_dict.py` exists and is export-only | — |
| 5.21 | `U4` Per-module first-run walkthrough, re-runnable | — |
| 5.22 | `U5` Keyboard shortcut overlay (`?`) + command palette | — |

## List 6 — Stability, testing & debuggability · 22 items · starts now

| # | Item | L0 |
|---|---|---|
| 6.1 | `S11` **Kill the flake sources first** — cross-test QSettings pollution, the `test_report_screen` flake. 6.10/6.13 are meaningless until green | — |
| 6.2 | `S1` Qt worker teardown — the 137-thread live-lock and the shard SIGSEGV, one investigation, from `bridge.py` | — |
| 6.3 | `F20` Remove the 6 remaining hardcoded `/home/carruthers/...` paths from the suite | — |
| 6.4 | `F13` Fix `test_image_umap_end_to_end`, which has never run | — |
| 6.5 | `S2` `spacr doctor` — environment, deps, GPU, CUDA, database integrity, common misconfigurations | — |
| 6.6 | `F3` Give `.merge()` a key contract — **31 of 76 call sites still have no `validate=`** | — |
| 6.7 | `Z15` `F34` residual on the non-resume path in `utils._merge_and_save_to_database` (strict xfail today) | — |
| 6.8 | `F18` Fix the sqlite retry arithmetic | — |
| 6.9 | `F11` Assert something in the **83** assertion-free tests | — |
| 6.10 | `S3` Golden-output regression — a tiny fixture dataset end to end, numbers checked in *(6.1)* | — |
| 6.11 | `S4` Property-based tests for key parsing — every `prc` bug this session would have been caught by one | — |
| 6.12 | `S8` DB contract tests — `DB_CONTRACT_AUDIT.jsonl` found real bugs and was never turned into tests | — |
| 6.13 | `S10` Mutation testing on the science core *(6.1)* | — |
| 6.14 | `S12` Perf regression guard — frame cost, measure throughput, memory ceiling as asserted numbers | — |
| 6.15 | `F12` Tighten the 8 remaining `**kwargs` Cellpose mocks | — |
| 6.16 | `F14` Ban the broad-`except`→`skip` shape in CI | — |
| 6.17 | `F15` Audit the 141 `except Exception: pass` sites in product code | — |
| 6.18 | `F16` `settings.py` to 100% — 240 units, the biggest single win | — |
| 6.19 | `F17` `sequencing.py` to 100% — 84 units on a real science path | — |
| 6.20 | **Every module to 100%** (excluding the Tk modules) — the standing goal | — |
| 6.21 | `S6` Crash reporter — log, settings, versions, last run in one attachable file *(6.2, 6.5)* | — |
| 6.22 | `F19` Get the docs site under the Pages ceiling; stop nightly republishing | — |

---

## Adding items during execution

1. **Dependency closure** — joins its dependency's list, below it. Depends on
   two lists → joins the one holding the *later* dependency.
2. **File ownership** — otherwise, the list that already owns the file.
3. **Load** — only if 1 and 2 leave it free.

Position: below what it needs, above what needs it, else the end. **A bug
found during execution goes to the top of its list.**

## Not in any list

Tutorials (Codex owns them). Not selected: `U6` undo for destructive actions,
`U7` explain-this-run, `U9` units + provenance, `V6` plate over time, `L7`
project manifest. `U6` is worth revisiting — F34/F35 exist precisely because
deletes are irreversible.
