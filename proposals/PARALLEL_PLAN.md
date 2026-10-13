# Parallel execution plan — 46 selected items

Selection made 2026-08-02. This document is the work breakdown: **what runs
concurrently, what must not, and why.**

The binding constraint is **file ownership**, not agent count. Two agents
editing `spacr/qt/app.py` at the same time is a merge conflict; two agents
editing `spacr/measure.py` at the same time is a merge conflict *and* a
science bug. So the plan is organised around who owns which file, and the
first wave exists mostly to create seams so the second wave can go wide.

---

## The contended files

Measured, not guessed:

| File | Lines | Who wants it |
|---|---|---|
| `spacr/settings.py` | 3,844 | every new feature adds defaults |
| `spacr/qt/theme.py` | 2,671 | every new widget wants styling |
| `spacr/qt/screens/settings_model.py` | 2,411 | U1, plus every new setting key |
| `spacr/qt/app.py` | 2,019 | 7 new screens want an `APPS` row |
| `spacr/measure.py` | 3,459 | N2, S5, S9 |
| `spacr/deep_spacr.py` | 3,082 | C9, C10, C4, S5 |
| `spacr/qt/screens/annotate.py` | — | C4, C5, V3, C8 |

`APPS` (app.py:208) is a flat list literal with `MAX_APPS_PER_SECTION = 13`.
Seven new modules land in it. That is a guaranteed conflict *and* a capacity
problem, so it gets fixed in Wave 0 before anyone needs it.

---

## Wave 0 — substrate (3 agents, must land before Wave 1)

Nothing else starts until these three merge. They are small, they are
plumbing, and every later track builds on them.

### W0-A — Typed ports + artifact registry `L1, L2`
**New files only:** `spacr/ports.py`, `spacr/artifacts.py`
**Touches:** `spacr/core.py` (registration hook at run completion)

- Each module declares what it consumes and produces (`ports.py`)
- Every output registers with provenance: producing module, settings hash,
  spaCR version, input artifact ids, timestamp, path (`artifacts.py`)
- Registry is a SQLite table in the project root, not a pickle
- Query API: `by_kind`, `by_project`, `latest`, `downstream_of`, `is_stale`

**Deliverable contract other tracks depend on** — publish these signatures
first, before implementing, so Wave 1 can code against them.

### W0-B — Run context: logging, seed, error policy `S7, S5, S9`
**New file:** `spacr/runctx.py`
**Touches broadly:** `settings.py`, `core.py`, `utils.py`, `measure.py`,
`deep_spacr.py`, `sequencing.py`, `ml.py`

This is the one track that reaches into the science modules, which is
exactly why it goes first and alone. Wave 1 tracks D/E/F edit those same
files and must not race it.

- `S7` — one run id, threaded through every log line, queryable
- `S5` — a global seed that actually reaches numpy, torch, cellpose, sklearn,
  and the samplers; today only `deep_spacr` and `sim` read it
- `S9` — `on_error: stop | skip | retry` as a first-class setting, default
  `stop`, honoured at every batch boundary

### W0-C — Screen registry seam
**Touches:** `spacr/qt/app.py` only

- `APPS` becomes a registry with `register_app(...)`; screens register from
  their own module
- Two new sections so the seven new modules do not blow `MAX_APPS_PER_SECTION`
- Settings-defaults seam so a module ships its own defaults instead of
  appending to `settings.py`
- Theme seam: a `register_widget_qss(...)` hook so new widgets style
  themselves without editing `theme.py`

**After W0-C, `app.py`, `settings.py` and `theme.py` stop being contended.**
That is the entire point of Wave 0.

---

## Wave 1 — wide (8–10 agents, disjoint file sets)

| Track | Items | Owns |
|---|---|---|
| **T1** Registry consumers | N4, N7, N8, L3, L5, L6 | `qt/screens/project_browser.py`, `run_compare.py`, `data_manager.py`; reads `artifacts.py` |
| **T2** Selection consumers | V1 wiring, V3, C8 | `qt/linked_selection.py` consumers, `qt/widgets/crop_grid.py`, `qt/screens/classifier_evaluation.py` |
| **T3** Plot platforms | V7, V8, V2, V5 | `qt/widgets/graph_builder.py`, `plot_pca.py`, `gate_editor.py`, `trellis.py` |
| **T4** Design apps | N1, N3 | `qt/screens/power.py`, `experiment_designer.py`; `power_simulate.py`, `power_model.py` already exist |
| **T5** Image science | N2 | new `spacr/illumination.py` + `measure.py` hook |
| **T6** Annotate & classify | C4, C5, C7, C9, C10 | `qt/screens/annotate.py`, `deep_spacr.py`, `timelapse.py`, `active_learning.py` |
| **T7** Sequencing & QC | C6, C1 | `sequencing.py`, `qt/screens/map_barcodes.py`, `qt/screens/measure.py` |
| **T8** UX shell | U1, U2, U4, U5, C3 | `settings_model.py`, `preferences.py`, `first_run.py`, `feature_dict.py` |
| **T9** Runtime stability | S1, S2, S6 | `qt/bridge.py`, new `spacr/doctor.py`, `spacr/crashreport.py` |
| **T10** Test infrastructure | S3, S4, S8, S10, S11, S12 | `tests/` only — fully isolated, can start at Wave 0 |

### Cross-track contracts (declared in Wave 0, honoured in Wave 1)

Two tracks want `annotate.py`. Rather than share it:

- **T6 owns `annotate.py` outright.**
- T6 exposes `open_objects(keys, *, reason)` — a slot that takes object keys
  and shows exactly those crops.
- T2's V3 (click a scatter point → open the crop) and C8 (click a confusion
  cell → open those cells) both call it. Neither edits the file.

Same pattern for plots: **T3 owns the plot widgets**, T2 consumes them
through `LinkedSelection` only.

---

## Wave 2 — integration (needs ≥2 Wave-1 outputs)

These cannot start early because they aggregate other tracks' work.

| Item | Waits on |
|---|---|
| **L4** pipeline graph view | L1+L2 actually in use by ≥3 real modules |
| **N5** QC dashboard | C1, C6, N2 — it aggregates their verdicts |
| **N6** methods + results exporter (AI) | S7 run logs, L2 registry, every module's settings |
| **V4** feature explorer | V1 + T3's plot infrastructure |
| **V9** lineage view | V1 + the relationship columns `measure` writes |

### N6 note — AI-backed, two outputs

Confirmed with the user: methods **and** results sections, both AI-generated
from structured input. The AI never sees raw data — it receives a structured
run digest (modules run, parameters, n, versions, statistics already computed)
and writes prose. Numbers come from the digest, never from the model, so the
figures in the text are the figures in the run.

---

## Concurrency ceiling

- **Wave 0:** 3 agents (+T10 in parallel, it is isolated) = 4
- **Wave 1:** 10 agents, disjoint by construction
- **Wave 2:** 5 agents

The ceiling is set by Wave 0. Skipping it and running 10 agents immediately
would produce ten conflicting edits to `app.py`, `settings.py` and
`theme.py`, and one race on `measure.py` between S5/S9 and N2 that would be
very hard to see in review.

---

## Not selected

Recorded so they are not silently lost: U6 (undo for destructive actions),
U7 (explain this run), U9 (units + provenance), V6 (plate over time), L7
(project manifest). From the JMP/Napari list: 4, 6, 7, 8, 9, 10, 11, 13, 14,
15, 16, 19.

Two worth revisiting: **U6**, because F34/F35 exist precisely because deletes
are irreversible; and **B19** (macro recorder), because it is most of N6's
input for free.
