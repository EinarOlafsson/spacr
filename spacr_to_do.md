# spacr_to_do

This is the ordered engineering backlog for `spacr-codex`. Tasks are arranged
in dependency waves; within each wave, shorter work comes first. Effort is a
rough implementation-and-test estimate, not elapsed calendar time.

Messages beginning with `#spacr_to_do` are added here and positioned by effort
and dependencies.

Status: `queued`, `in progress`, `blocked`, or `complete`.

## Wave 0 — completed prerequisites

| ID | Status | Effort | Task | Commit |
|---|---|---:|---|---|
| STAB-01 | complete | 2–4 h | Fix the permitted `umap-learn 0.5.6` / newer scikit-learn incompatibility. | `b7d2bbe` |
| STAB-02 | complete | 2–4 h | Detect incompatible UMAP dependencies and provide an actionable error. | `b7d2bbe` |
| STAB-03 | complete | 2–4 h | Make lazy UMAP loading resettable and test-order independent. | `b7d2bbe` |

## Wave 1 — short independent hardening

| Order | ID | Status | Effort | Task | Depends on |
|---:|---|---|---:|---|---|
| 1 | ARCH-08 | complete | 1–2 h | Add correctly spelled `interpret_vision_model` APIs while retaining compatibility aliases. | — |
| 2 | ARCH-06 | complete | 1–2 h | Remove workstation-specific defaults from settings. | — |
| 3 | STAB-05 | complete | 2–4 h | Validate count data and minimum sample sizes before fitting Poisson GLMs. | — |
| 4 | ARCH-07 | complete | 2–4 h | Consolidate the three `MEASUREMENT_STAMP_COLUMNS` definitions. | — |
| 5 | DATA-05 | complete | 2–4 h | Coalesce missing measurement stamps from the right table when the left value is null. | ARCH-07 |

## Wave 2 — focused correctness and CI

| Order | ID | Status | Effort | Task | Depends on |
|---:|---|---|---:|---|---|
| 6 | DATA-01 | complete | 0.5–1 d | Resolve the V1/V2 segmentation parity xfail. | — |
| 7 | DATA-04 | complete | 0.5–1 d | Reject or explicitly reconcile conflicting acquisition metadata. | ARCH-07 |
| 8 | TEST-02 | complete | 0.5–1 d | Treat selected pandas and scikit-learn deprecations as CI errors. | — |
| 9 | TEST-03 | complete | 1–2 d | Set TIFF photometric and planar configuration consistently. | — |
| 10 | STAB-04 | complete | 1–2 d | Test minimum and newest supported dependency versions in CI. | STAB-01 |
| 11 | UI-01 | complete | 1–2 d | Add persisted preferences to show or hide Alpha and Beta modules and settings; both visibility options default to enabled. | — |
| 12 | DATA-03 | complete | 1–2 d | Add merge cardinality validation where table contracts require it. | ARCH-07 |
| 13 | TEST-04 | complete | 2–4 d | Split fast, integration, slow, GPU, network, NAS, and Qt CI jobs while retaining automatic resource detection. | STAB-04 |
| 14 | TEST-05 | queued | 2–4 d | Add linting, typing, complexity checks, and a zero-unexpected-failure release gate. | TEST-04 |

## Wave 3 — schema and persistence foundation

| Order | ID | Status | Effort | Task | Depends on |
|---:|---|---|---:|---|---|
| 15 | DATA-02 | queued | 3–5 d | Define an enforceable canonical schema for cell, cytoplasm, nucleus, and pathogen tables. | ARCH-07, DATA-05 |
| 16 | DATA-10 | queued | 2–4 d | Separate provenance columns from numerical model features using the canonical schema. | DATA-02 |
| 17 | DATA-06 | queued | 4–7 d | Add database schema versions and an explicit migration framework. | DATA-02 |
| 18 | DATA-07 | queued | 4–7 d | Make multi-table database writes atomic and transactional. | DATA-02, DATA-06 |
| 19 | DATA-08 | queued | 3–5 d | Store schema version, exact features, dependencies, and preprocessing configuration with each model artifact. | DATA-02, DATA-06, DATA-10 |
| 20 | DATA-09 | queued | 1–2 wk | Add systematic validation at every module boundary. | DATA-02, DATA-06 |

## Wave 4 — cross-cutting maintenance

| Order | ID | Status | Effort | Task | Depends on |
|---:|---|---|---:|---|---|
| 21 | TEST-01 | queued | 2–4 d | Gate figure `.show()` calls for clean headless execution. | TEST-02 |
| 22 | ARCH-04 | queued | 1–2 wk | Separate plotting from analysis in `plot.py` and `submodules.py`. | TEST-01 |
| 23 | ARCH-05 | queued | 1–2 wk | Replace large settings dictionaries with typed configuration models. | DATA-02, DATA-09 |
| 24 | ARCH-09 | queued | 2–4 wk | Replace direct printing with structured logging and injectable progress callbacks. | DATA-09 |

## Wave 5 — large module decomposition

| Order | ID | Status | Effort | Task | Depends on |
|---:|---|---|---:|---|---|
| 25 | ARCH-03 | queued | 2–3 wk | Split `io.py` into database, images, exports, datasets, and migrations. | DATA-06, DATA-07 |
| 26 | ARCH-02 | queued | 2–3 wk | Split `timelapse.py` into tracking, QC, visualization, and analysis modules. | DATA-09, ARCH-09 |
| 27 | ARCH-01 | queued | 3–5 wk | Split `utils.py` into focused packages with compatibility imports. | ARCH-04, ARCH-05, ARCH-07, ARCH-09 |
