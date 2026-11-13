# Item spec: OPS live mode: try each step's settings on the smallest subset that exercises it, see the result, write the settings back

- **Lifecycle:** future (alpha), behind "Show alpha features", as every new feature is. One `ALPHA_FEATURES[<this item>]` entry covers:
  - the `ops_live_*` settings;
  - the OPS live-preview card;
  - the "Apply to run settings" action.
- **Number:** 935 (reserved 2026-11-13).
  - 752 (OPS engine, GUI panel);
  - the Live Preview registry (`spacr/qt/preview_registry.py`);
  - the OPS and Align & Stitch example data (`spacr/qt/ops_stitch_demo.py`);
  - 912, 921, 928–932;
  - 106 (inapplicable settings greyed out).
- **Overlap with an existing spec:** none. I searched `proposals/` and `features/feature_list/` on origin/nightly and the NAS `spacr_items/`. The only "live preview" is the existing per-module registry, which has no OPS entry.
- **Rule:**
  - A live run of a step on a subset gives the same per-object output as the full run on the same tiles or fields, with the same settings and the same upstream inputs (exceptions stated per step).
  - Live outputs never enter the production `measurements.db` and are never paper data (READINESS rule).
  - Every new documented callable ships with its arrival contract and nine-locale records in the same commit.
- **Code basis:** origin/nightly (46f259ea23).
- **Paper-critical:** no.

## Problem and goal
An OPS run takes hours per well. Today a user who wants to know whether a stitch overlap, a decode threshold or a segmentation diameter is right has two options: run whole wells, or hand-cut a sample. The bundled example (`ops_stitch_demo.py`) is two SBS sites and nine stitch tiles from one plate. It cannot be the user's own data, and its docstring says why it cannot exercise the phenotype alignment.

**Goal:** for each OPS step (stitch, align, genotype, phenotype, cluster analysis), the user:
1. picks a well;
2. lets spaCR pick the smallest subset of **their own** data that still exercises every feature of that step;
3. runs only that step on it in seconds to a few minutes;
4. sees the step's result;
5. changes settings and reruns;
6. finally writes the chosen settings into the full-run settings file.

## Current state (verified on origin/nightly)
- **Engine.**
  - `spacr.ops_engine.run_ops(settings, *, wells, phases, library)` (`ops_engine.py:7828`). Phases include `stitch`, `phenotype`, `objects`, `decode` and the alpha `ph_*`, `merge`, `aggregate`, `profile`, `cluster_*` and `score` phases.
  - A phase left out reads its inputs from `measurements.db`, so phases can be run one at a time.
- **Subsets already possible.**
  - `ops_ph_sites` (str or None, `ops_settings.py:58`) restricts the phenotype sites.
  - `ops_stitch.stitch_well(tiles, layout, *, sites=...)` (`ops_stitch.py:190`) places a subset of sites against the full well layout.
  - `ops_sample` measures single fields.
  - Decode works per field (`ops_cycles.align_field`, `ops_decode`).
- **Constraints recorded in code.**
  - The engine takes a well's field count as its highest site number plus one, and fits the round well to that count. A subset without the last site therefore gets a wrong layout unless the layout comes from the full index (`ops_stitch_demo.py` docstring).
  - "The phenotype phase aligns anchors spread over the whole well, which a two-field sample cannot supply" (same docstring).
  - The per-field refinements (`ops_merge.refine_field_transform`, `tile_affine_transform`; item 930) start from a stored well-level seed.
- **FDR.**
  - Decoy-based false-discovery estimates use `ops_fdr_scale` (1.268) and `ops_decoy_library_k` (0) (`ops_settings.py:175-176`). The FDR estimate is about `scale × decoy calls / real calls`.
  - The Funk plate A library plus decoys is 102,225 sequences (S2 speed card).
- **GUI.**
  - Live Preview exists for Mask, Measure, Timelapse, Motility, host-pathogen and plaque, through `PREVIEWS` in `spacr/qt/preview_registry.py`, with `PreviewSpec(builder, fill, ...)`.
  - The OPS panel has no preview.
  - `align.default_settings` has `preview_only`, which plans without writing.
- **Speed today** (S1 and S2 speed cards, one core, loaded machine):
  - placement is about 33–56 CPU s per 333-tile well, plus 150–320 s of reading;
  - decode was about 100 CPU s per FOV before item 932.

## Proposed design

### Minimal subset per step
Each step has a default automatic selection, `ops_live_auto=True`, and the user can override it. The selection is deterministic (seeded) and recorded with the result.

| Step | Default subset | Why this is the minimum that exercises the step | What is not exercised (shown to the user) |
|---|---|---|---|
| **Stitch** (`stitch`) | A **3 × 3 block** of adjacent cycle-1 SBS tiles at the well centre (9 tiles, 12 edges, 4 interior seams meeting at one point), **plus a 2 × 2 block on the rim** of the round well (tiles with missing neighbours) | Covers both edge directions; the global least-squares solve with redundant edges (a loop); the acceptance rule including a rejected or absent edge at the rim; the windowed feathered composition (931) across a 4-way seam junction. The layout is fitted from the **full** site index, not the subset | Long-range drift across the well; the canvas-size check against `expected_canvas` (not meaningful on a block) |
| **Align** (`merge` with `ops_align_model`, items 930 and 921) | The phenotype fields whose footprint overlaps a **2 × 2 block** of SBS tiles. At about 3.85 phenotype fields per SBS tile (1,281 / 333 on Funk plate A) that is about 10–16 fields | Every field crosses at least one SBS tile boundary, so the per-tile-group affine has more than one group per field. The mutual-nearest matching, trimming and bounds checks are exercised | The **well-level similarity seed**. With a full run, the stored seed is reused. Without one, live align first fits the seed from a sparse spread of 12 fields across the well (DAPI only, nuclei points only) and says so; that costs one read of 12 fields |
| **Genotype** (`decode`, items 873, 932) | **Sized by the FDR target**, not a fixed count. Start with 4 SBS tiles from the stitch block's neighbourhood; add tiles until the expected decoy count reaches `ops_live_min_decoys` (10), or `ops_live_max_sbs_tiles` (24) is hit | Calls, reads, cell rule and per-well decoy threshold all run on any tile. **A handful of tiles is not enough for a meaningful FDR at the paper's target.** At FDR 0.1 % and scale 1.268, 10 expected decoy calls need about 12,700 assigned cells (10 / (0.001 / 1.268)). At a target of 1 % they need about 1,300. The cells per tile are measured on the first tiles and the count is extended from them. If the cap is hit first, the FDR is shown as an upper bound with its Poisson 95 % CI, marked "not resolved at this size" | Per-well decoy threshold tuning (the maintainer's 2026-11-13 rule) needs a well's worth of cells. Live mode shows the threshold the subset would pick and how far it is from the stored full-well value, if one exists |
| **Phenotype** (`ph_flatfield` / `ph_segment` / `ph_measure` / `ph_masks`) | **4 fields**: one at the well centre, one corner-heavy field (largest flat-field fall-off), one on the well rim (partial coverage), and the field with the highest nucleus density from a cheap DAPI count | Exercises segmentation at both densities, field-edge flags, the flat-field correction where it matters (the corners), measurement, and ownership flags. With `ops_ph_large_site`, the tiling of item 921 on one site. The flat-field map is **read** from the plate's existing map; it is not rebuilt, because 912's estimator needs the whole plate | Building the flat-field map; per-plate statistics; cross-field ownership beyond the chosen fields |
| **Cluster analysis** (`profile`, `cluster_*`, items 928, 929, 916) | A **small gene set**: by default 40 genes, made of 4–6 complete CORUM complexes (at least 3 members each, present in the screen), 10 random genes as a control, and the non-targeting rows. They come from the stored **cell tables of a finished run**; live mode never produces cell tables for clustering | Profile building (TVN, PCA), Leiden at the chosen resolution, consensus (928) at R = 5 instead of 20, and the complex-recovery count | **Statistics are not meaningful at this size.** The calibrated test (929), p values, CIs and FDRs are **not computed** in live mode; the card says why. Retrieval counts are shown as "x of y complexes recovered", with the random-gene control beside them, and labelled illustrative |

### What is shown per step (one card each, all on screen in under one minute where possible)
- **Stitch:**
  - the tile outlines on the composed block, each edge coloured accepted or rejected, with its peak ratio;
  - per-tile residuals;
  - a zoom on the 4-way seam junction with a seam-step score (the S1 (c) metric);
  - time per edge and per window.
- **Align:**
  - SBS nuclei over transformed phenotype nuclei per field;
  - the tile-group map;
  - the within-2-px share per field next to the similarity-only share;
  - fields that fell back to similarity, with the reason.
- **Genotype:**
  - per-tile calls on the DAPI and base images;
  - read-quality histograms;
  - cells genotyped and the share of cells genotyped;
  - FDR with its Poisson CI (or the "not resolved" banner);
  - the decoy threshold;
  - time per FOV, with the registration precision and backend (932).
- **Phenotype:**
  - mask overlays for the 4 fields;
  - raw vs flat-field-corrected corner crops;
  - per-field summary rows (cells, median area, edge-flag share);
  - time per field;
  - GPU or CPU path.
- **Cluster:**
  - embedding plot coloured by cluster and by complex;
  - complex-recovery table with the random-gene control;
  - a banner: "illustrative; no statistics at n = 40 genes".

Every card shows the settings that differ from the full-run settings file, and the time each step took.

### Settings (alpha; read only by the live runner and never by `run_ops` on a full run)
| Key | Type | Default | Meaning |
|---|---|---|---|
| `ops_live_step` | str | `'stitch'` | one of `stitch`, `align`, `genotype`, `phenotype`, `cluster` |
| `ops_live_well` | str or None | None | None is the first well in the index |
| `ops_live_auto` | bool | True | automatic subset as in the table |
| `ops_live_sites` | str or None | None | explicit SBS sites (stitch, genotype) when `ops_live_auto` is False |
| `ops_live_ph_sites` | str or None | None | explicit phenotype fields (align, phenotype) |
| `ops_live_block` | int | 3 | stitch block edge, in tiles |
| `ops_live_rim_block` | bool | True | add the rim block |
| `ops_live_fdr_target` | float | 0.001 | FDR the genotype subset is sized for |
| `ops_live_min_decoys` | int | 10 | expected decoy calls before the FDR counts as resolved |
| `ops_live_max_sbs_tiles` | int | 24 | cap on genotype tiles |
| `ops_live_ph_fields` | int | 4 | phenotype fields |
| `ops_live_genes` | str or None | None | explicit gene list for cluster; None means auto |
| `ops_live_n_genes` | int | 40 | auto gene-set size |
| `ops_live_consensus_reps` | int | 5 | consensus replicates in live cluster |
| `ops_live_dst` | str or None | None | None is `<dst_root>/live/` |
| `ops_live_seed` | int | 0 | subset selection seed |
| `ops_live_tile_cache` | bool | True | keep the subset's tiles in a local cache between reruns (tiles are read from the NAS once) |

### Public callables (new module `spacr/ops_live.py`)
- `select_live_subset(settings, step) -> LiveSubset`: sites and fields, the reason for each, and what is not exercised.
- `run_live_step(settings, step, *, subset=None) -> LiveResult`: tables, figure paths, timings, the settings used, and the subset.
- `live_settings_diff(live_result, settings_path) -> dict`: the keys whose value differs from the file.
- `write_live_settings(live_result, settings_path, *, keys=None, backup=True) -> str`:
  - writes the changed step settings back;
  - never writes `ops_live_*` keys;
  - writes a timestamped backup first;
  - records `live_provenance` (step, subset, result hash, spaCR version) in the file's metadata block.

`run_live_step` calls the engine's own phase functions (`stitch_well`, `_ph_*`, decode, `run_profile_phases`) on the subset. There is no copy of engine logic in `ops_live.py`. A test asserts that `ops_live` imports no engine private function that is missing from an explicit allow-list, and that the allow-list functions are the ones `run_ops` calls.

### Output layout
- Everything goes under `<dst_root>/live/<step>/<YYYYmmdd-HHMMSS>/`: `live.db` (same schema as `measurements.db`), `figures/`, `settings_used.json`, `subset.json`, `timing.json`.
- Live runs never write to `<dst_root>/measurements.db`. Old live folders are pruned to the last `ops_live_keep` (10) per step.

### GUI
- `PREVIEWS['ops'] = PreviewSpec(builder='spacr.qt.widgets.ops_live_preview:build_ops_live_card', fill='spacr.qt.widgets.ops_live_preview:fill_ops_live_card', title='Live mode')`. The OPS panel gets the standard preview toggle; no edits to `AppScreen` are needed.
- The card has five step tabs. Each tab has:
  - a "Pick subset" row (auto, or a site picker on the well map);
  - "Run step";
  - the result card;
  - "Apply to run settings".
- "Apply" shows the diff (old → new per key) and asks before writing.
- A step whose upstream inputs are missing (for example, align without a stitch) offers to run the upstream step on the same subset first.
- The tile cache and the live folder are shown with their size. One "Clear live data" action deletes them.
- Runs go through the existing worker and cancellation machinery, and the GPU only through the existing admission (`ops_gpu_admission`) when it is on.

## Acceptance tests (CPU unless marked; `CUDA_VISIBLE_DEVICES=''`, at most 4 test workers)
1. **Subset selection.** On the synthetic 333-site round-well index and on the Funk plate A index:
   - the stitch subset is a 3 × 3 adjacent block plus a 2 × 2 rim block;
   - the layout is the full-well layout (site count 333, not the subset's highest site + 1);
   - selection is identical for the same seed.
2. **Stitch identity.** On plate A well A1, the subset's integer edge shifts, peak ratios and acceptances are identical to the full run's for the same edges.
   - Placements are **not** expected to equal the full-well solve, because the least-squares solve is global.
   - The test instead checks that placements equal `solve_placements` over the subset's own edges (≤ 1e-9 px).
   - The composed window equals `compose_window` on the same placements byte for byte.
3. **Align identity.** Given the stored well seed, the per-field transforms, tile groups and moved nuclei for the live fields are identical to the full-run `ops_phenotype` rows for those fields (item 930's step-6 tables). Without a stored seed, the 12-field seed is within 2 % scale and 1° rotation of the full seed on plate A.
4. **Genotype identity.** Per-cell calls, reads and qualities on the live tiles are identical to the full run's for the same tiles, at the same threshold.
   - Subset sizing: with a planted cell count per tile, the tile count equals ceil(min_decoys × scale / (target × cells per tile)), capped.
   - The "not resolved" banner appears exactly when the cap is hit.
5. **Phenotype identity.** Masks and measurement rows of the 4 fields are identical to the full run's (exact, CPU; the GPU path is checked under `tools/gpu_turn.sh` only).
6. **Cluster isolation.**
   - No p value, CI or FDR column appears in live cluster output.
   - The complex-recovery count on a planted synthetic profile set (4 planted complexes) recovers the planted complexes, and the random-gene control recovers none.
7. **Write-back.**
   - `write_live_settings` changes only the step's keys.
   - It never writes `ops_live_*`.
   - It writes a backup.
   - The rewritten file loads and validates.
   - A full `run_ops` from it differs from the original file's run only in the changed keys (dry-run settings comparison).
8. **No production writes.** The production `measurements.db` mtime and md5 are unchanged after every live step.
9. **Time vs the current path** (marked slow; plate A, one pinned core, tiles in the live cache). Pre-set targets, reported with the full-path time for the same step:
   - stitch ≤ 10 s (9 + 4 tiles, after the stitch speed card);
   - align ≤ 15 s with a stored seed;
   - genotype ≤ 60 s per 4 tiles after item 932;
   - phenotype ≤ 120 s for 4 fields on GPU, ≤ 10 min on CPU;
   - cluster ≤ 60 s.
   Misses are reported and do not block the alpha release.
10. **GUI** (offscreen Qt): the OPS live card registers, each tab builds lazily, and "Apply" shows the diff and writes only on confirmation.

## Translations / catalog obligations
- Every public callable of `ops_live.py`, `LiveSubset` and `LiveResult` ship in the same commit with:
  - their arrival contracts;
  - nine-locale records;
  - the regenerated help index, API catalogs, setting consumer map and settings flow.
- Every `ops_live_*` tooltip, card label, banner ("not resolved at this size", "illustrative; no statistics") and dialog string has nine-locale records.
- Per the tutorials rule, the alpha-features tutorial lesson is the only lesson that may show the card.

## Dependencies on items 928–932
- **930** (per-tile affine): the align step's main subject. Its subset is defined by 930's tile groups.
- **931** (windowed feathered composition, settings schema version): the stitch card shows 931's seams. `write_live_settings` must write the schema version 931 introduces.
- **932** (SBS registration precision and backend): the genotype time target assumes 932. The card shows `ops_registration_precision` and `ops_registration_backend`.
- **928 / 929** (consensus, calibrated test): live cluster runs 928 at low R and **never** 929 (no statistics at this size).
- Also 921 (large-site tiling in the phenotype step) and 912 (the flat-field map is read, not built).

## Alpha status
Alpha. The whole feature sits under one `ALPHA_FEATURES` entry. With alpha off, the OPS panel shows no live toggle.

## Maintainer decisions (2026-11-13, AskUserQuestion in the master session)
Live genotype shows FDR at a looser 1 % target (the paper target stays on full wells); decoy library k = 0 for identity with full runs; a 12-field sparse align seed is allowed and labelled; cluster live mode also accepts a published profile table; 'Apply to run settings' is blocked while a full run of the same settings file is in progress.

## Open questions for the maintainer (answered above)
1. Genotype FDR sizing: is 10 expected decoys the right floor, giving about 12,700 assigned cells at FDR 0.1 %, which may be more than a "handful" of tiles? Or should live genotype show the FDR only at a looser target (1 %), with the paper target left to full wells?
2. Can `ops_decoy_library_k > 0` (a larger decoy library) be used in live mode to get more decoy draws from the same cells? That would need the FDR scale recalibrated for the larger library. Is it worth it, or should live mode stay at k = 0 for identity with full runs?
3. Without a finished run there is no well seed for align. Is a 12-field sparse seed acceptable, or should live align require a finished stitch-and-seed of the well?
4. Cluster live mode needs cell tables from a finished run. Should it also accept a published profile table (for example, the Funk et al. 2022 gene table) so users can try clustering settings before running their own plate?
5. Should "Apply to run settings" be allowed while a full run of the same settings file is in progress, or be blocked until it finishes?
