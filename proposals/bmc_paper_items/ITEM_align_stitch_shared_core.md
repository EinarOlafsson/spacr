# Item spec: Align & Stitch adopts the OPS stitching and alignment code through one shared core (no duplicate registration, solve or composition)

- **Lifecycle:** future (alpha), behind "Show alpha features".
  - The new engine is selected by `align_engine='core'` (alpha).
  - The default stays `'legacy'` until the identity and accuracy proofs pass and the maintainer flips it. That flip is a separate decision row in 931's default-change table.
- **Number:** 936 (reserved 2026-11-13).
  - 930 (per-tile affine, `ops_merge`);
  - 931 (windowed feathered composition, `ops_compose`; worktree `/mnt/wd4tb/worktrees/compose-feather-1113`, not yet on nightly);
  - 932 (FFT engine, float32 / GPU registration, `ops_cycles` / `ops_register`; worktree `/mnt/wd4tb/worktrees/sbs-speed-1113`, not yet on nightly);
  - the S1 stitch speed card (`BMCspacr/reanalysis/proof_stage1_stage2/DESIGN_PROPOSAL_S1_stitch_speed_DRAFT.md`, signed off 2026-11-13, not yet run);
  - the stitch determinism commit 5797246603 (one BLAS thread in the solve, on nightly);
  - 921 (large-site tiling).
- **Overlap with an existing spec:** none in `proposals/bmc_paper_items/`. The stitch speed card is a design card for OPS only. This spec depends on its result and does not repeat it.
- **Rule:**
  - With `align_engine='legacy'`, `align_folder` output is byte-identical to today: stack, `align_coordinates` table, and plan text.
  - With `align_engine='core'` and the OPS-equivalent settings, `align_folder` gives the same integer edge shifts, acceptances and placements (≤ 1e-9 px) as `ops_stitch.stitch_well` on the same tiles, and the same composed pixels as `ops_compose.compose_window`.
  - OPS outputs do not change when OPS moves onto the core.
  - Every new documented callable ships with its arrival contract and nine-locale records in the same commit.
- **Code basis:** origin/nightly (46f259ea23), plus the 931 and 932 branches once they land.
- **Paper-critical:** no for the OPS paper. Yes for the platform claim that the general module uses the OPS-proven code.

## Problem and goal
spaCR has **three** tile-stitching implementations:
- the general Align & Stitch module (`spacr/align.py`);
- the import-time field stitcher (`spacr/image_stitch.py`);
- the OPS chain (`ops_register` → `ops_solve` → `ops_stitch` → `ops_compose`, plus `ops_merge` for cross-acquisition alignment and `ops_cycles` for cycle registration).

The OPS chain is the one that has been proven against reference data: the S1 proof cards, 930's 97 % within 2 px, 931's seam result, and the deterministic solve. It is getting the speed work (932 and the stitch speed card). The general module does not benefit from any of it, and every fix has to be made more than once.

**Goal:**
1. One registration, solve and composition core, used by OPS and by Align & Stitch.
2. Align & Stitch gains:
   - overlap-strip registration with peak-ratio acceptance against a layout prediction;
   - the deterministic global least-squares solve;
   - windowed feathered composition (931);
   - per-tile-group affine alignment of a second acquisition (930);
   - the float32 / in-place registration speed-ups, once the stitch speed card confirms them.
3. No function is implemented twice.

## Current state (verified on origin/nightly)

### General: `spacr/align.py` (2,574 lines; GUI app key `align`, `spacr/qt/bridge.py:2160` → `align_folder`)
- **Input.** `scan_tiles` (l. 1046) reads headers only, of `.npy` or TIFF. It takes a grid, an overlap fraction (default 0.10) and an order (`row-major`, `column-major`, `snake-row`, `snake-column`). `group_tiles` (l. 1179) groups tiles per well.
- **Registration.** `_register_pair` (l. 1250):
  - reads the nominal overlap strips through `_TileReader` (memory-mapped windows);
  - standardises them;
  - runs `skimage.registration.phase_cross_correlation` with `upsample_factor=10` (sub-pixel), normalisation 'phase' and None;
  - scores each candidate by the NCC of the shifted strips (`_score_shift`) and accepts above `min_confidence` (0.30);
  - applies an optional `max_shift`.
  - A tile with no surviving pair keeps its nominal position (`method='nominal'`).
- **Solve.** `solve_positions` (l. 1378) is a weighted least squares over the pair graph, plus weak anchor equations to the nominal positions (`DEFAULT_ANCHOR_WEIGHT` 1e-3). It returns residuals. The BLAS thread count is not pinned.
- **Composition.**
  - `write_stack` (l. 1994) writes horizontal bands under `max_buffer_bytes` (64 MB).
  - `blend` is `'feather' | 'average' | 'none'`. Feather is a linear ramp `_ramp` (l. 1857), clipped to `[_WEIGHT_FLOOR=1e-3, 1]`, over a width from `_feather_width` (l. 1756): the median overlap span, at most half a tile.
  - Writers are `stream` and `memmap`. Channel-last stacks, all channels share one placement.
- **Output.** The `align_coordinates` table in `measurements.db` (`save_coordinates`, l. 2282), keyed by `plateID` / `rowID` / `columnID` / `fieldID`.
- **Settings.** `default_settings` (l. 2448):
  - inputs: `src`, `dst`, `db_path`, `grid`, `overlap`, `order`, `recursive`;
  - registration: `reference_channel`, `min_confidence`, `neighbour_radius`, `upsample`, `min_overlap_px`, `max_shift`;
  - composition: `blend`, `feather`, `subpixel`, `writer`, `max_buffer_bytes`, `band_rows`;
  - run control: `save_stack`, `overwrite`, `preview_only`, `group_by_well`.

### Import-time stitcher: `spacr/image_stitch.py` (648 lines)
- `plan_mosaic` and `stitch_tiles` put tiles of one field back together at import.
- The layout comes from OME stage positions (`read_stage_positions`) when they exist. Otherwise it scores every admissible displacement by NCC, and chooses the arrangement (row-major or serpentine) by correlation.
- If nothing correlates, it butt-joins the tiles and marks the result `assumed`.
- It has its own `Placement` and `Mosaic` classes.

### OPS
- **`ops_register.py`** (522 lines):
  - `phase_correlate` (l. 301): numpy, cupy or torch backend (`available_backends`, l. 213); `unwrap` (l. 151); peak/mean ratio; rejection of a peak at (0, 0).
  - `register_edge` (l. 402): a 30 % strip (`STRIP_FRACTION`), acceptance against the layout-predicted overlap within `tolerance`, plus a separate `skew` (`SKEW_PX`, 9 px measured).
  - `register_pairs` (l. 487).
  - Integer shifts.
- **`ops_solve.solve_placements`** (l. 30): unweighted least squares over the accepted edges. The origin is pinned at the lowest site of each connected component; there are no anchors. Since 5797246603 it runs on one BLAS thread, so reruns are byte-identical.
- **`ops_stitch.stitch_well`** (l. 190): runs register → solve over a `WellLayout` (`ops_layout`, round wells, `round_well_layout`). It returns `StitchedWell` with `expected_canvas` as an arithmetic check. Output is coordinates, never a full canvas.
- **`ops_compose`** (180 lines):
  - `windows_over` (l. 69) gives overlapping windows (default 4,096 px, overlap 256).
  - `compose_window` (l. 104) is an average on nightly.
  - **931** (worktree, not on nightly) adds `blend='feather'` as the default, weighting each tile by its distance to its own nearest edge, plus 1, in float64. The setting is `ops_compose_blend`. Result: on Funk et al. 2022 the feathered seams matched an unstitched tile cut at the same place within 1 point (95 % CI), and the averaged seams were 11–12 points worse on plates C and D.
- **`ops_merge`**: phenotype → SBS alignment.
  - `align_by_triangles`, `refine_field_transform` (l. 278), `affine_polish` (l. 473), `affine_within_bounds` (l. 526).
  - **930**: `tile_affine_transform` (l. 558) and `tile_affine_to_well` (l. 623). Settings: `ops_align_model` ('similarity' | 'tile_affine'), `ops_align_tile_min_nuclei` (200), `ops_align_match_radius` (3.0), `ops_align_trim_passes` (3). Result: 97.0–97.5 % within 2 px on plates A–D.
- **932** (worktree, step 6 pending): `ops_cycles._FFTEngine` caches each reference's forward transform. `ops_registration_precision` is 'float32' (scipy.fft, the new default) or 'float64' (identical to before), and `ops_registration_backend` is 'cpu' or 'gpu'. It adds `ops_register._read_peak`, shared with `phase_correlate`.
- **Stitch speed card** (signed off, not run). Candidates: (1) scipy.fft float32 with `workers=1`; (3) in-place cross-power normalisation and a one-pass peak ratio; (4) parallel edges; (6) read each tile once. The identity gate is identical integer shifts and acceptances, and placements within ≤ 1e-9 px.

### Differences between general and OPS
| Aspect | `align.py` | OPS | Consequence for the core |
|---|---|---|---|
| Registration | skimage `phase_cross_correlation`, sub-pixel (upsample 10), two normalisations | own FFT, integer, numpy/cupy/torch, unwrap, (0,0) rejection; float32 and GPU via 932 | one `phase_correlate`; sub-pixel refinement as an option on top (`upsample > 1`) |
| Acceptance | NCC of the shifted strips ≥ 0.30, optional `max_shift` | peak/mean ratio ≥ `MIN_PEAK_RATIO`, plus layout-predicted overlap within `tolerance` and `skew` | both scores computed. `acceptance` setting: 'ncc', 'peak_ratio' or 'both' |
| Strip | the nominal overlap window | 30 % of the tile edge (wider than the 14 % overlap; measured better) | `strip_fraction` option; None means the nominal overlap |
| Layout | rectangular grid plus scan order | `WellLayout` (round well, measured pitch and skew) | the core takes a pair list and expected offsets; layouts stay with their callers |
| Solve | weighted, with nominal anchors, BLAS threads not pinned | unweighted, origin pinned per component, one BLAS thread | one `solve_positions(edges, n, *, weights=None, anchors=None, anchor_weight, gauge='anchor' or 'lowest_site')` on one BLAS thread |
| Composition | band writer, linear ramp of width ≈ overlap, floor 1e-3 | windows; 931 feather by distance-to-edge + 1, float64 | one `feather_weights(tile_shape, mode, width)` used by both writers; band and window writers stay as two drivers of the same weights |
| Determinism | not pinned | pinned (5797246603) | pinned in the core |
| Cross-acquisition alignment | none | 930 per-tile affine; `ops_ph_secondary_registration` | offered in Align & Stitch as a second-acquisition mode (alpha) |

## Proposed design

### Shared core: `spacr/stitch_core.py` (no Qt, no torch at import; torch and cupy only inside the GPU backend, as `ops_register` does now)
| Function | Moved from | Callers after the move |
|---|---|---|
| `phase_correlate`, `unwrap`, `available_backends`, `_read_peak`, `FFTEngine` (932) | `ops_register`, `ops_cycles` | `ops_register` (re-export), `ops_cycles`, `align` |
| `register_strip_pair(first, second, axis, *, strip_fraction, expected, tolerance, skew, acceptance, min_peak_ratio, min_ncc, upsample, precision, backend)` | `ops_register.register_edge` and `align._register_pair` (merged) | `ops_stitch`, `align`, `image_stitch` (optional) |
| `solve_positions` | `ops_solve.solve_placements` and `align.solve_positions` (merged) | both |
| `feather_weights`, `compose_window`, `windows_over` | `ops_compose` and `align._ramp` | `ops_compose` (re-export), `align.write_stack` |
| `affine_polish`, `affine_within_bounds`, `tile_affine_transform`, `tile_affine_to_well`, `refine_field_transform` | `ops_merge` | `ops_merge` (re-export), `align` (second-acquisition mode) |

The old module names stay as re-exports, so every OPS import, test and pickled reference keeps working. A test asserts identity of the objects (`ops_register.phase_correlate is stitch_core.phase_correlate`, and so on).

### Merged semantics, with the defaults that reproduce each caller exactly
- **`register_strip_pair`.** OPS calls it with `acceptance='peak_ratio'`, `strip_fraction=0.30`, `upsample=1` and integer output. `align` (legacy-equivalent) calls it with `acceptance='ncc'`, `strip_fraction=None`, `upsample=10` and the skimage normalisations.
  - The legacy skimage path is kept **only** under `align_engine='legacy'`. The core engine does not call skimage.
  - Once the core default is approved, the skimage path is retired, with a deprecation note in the release notes.
- **`solve_positions`.** `gauge='lowest_site'` with no anchors reproduces OPS; `gauge='anchor'` with `anchor_weight=1e-3` reproduces `align`. Both run under one BLAS thread. `align`'s numbers can then move in the last float digits relative to today; that is what `legacy` is for.
- **`feather_weights`.**
  - `mode='distance'` is 931's distance-to-edge + 1, over the whole tile.
  - `mode='ramp'` is `align`'s linear ramp with `width` and the floor.
  - Both writers accumulate in float64 under `'distance'`, as 931 does.

### Align & Stitch settings (new; alpha; added to `align.default_settings`)
| Key | Type | Default | Meaning |
|---|---|---|---|
| `align_engine` | str | `'legacy'` | `'core'` uses `stitch_core` |
| `registration_precision` | str | `'float64'` | `'float32'` once the stitch speed card confirms identity (then the default becomes float32 under `core`) |
| `registration_backend` | str | `'cpu'` | `'gpu'` with automatic CPU fallback (as in 932) |
| `acceptance` | str | `'ncc'` | `'peak_ratio'` or `'both'` |
| `min_peak_ratio` | float | `ops_register.MIN_PEAK_RATIO` | used by `peak_ratio` and `both` |
| `strip_fraction` | float or None | None | None means the nominal overlap; 0.30 is OPS |
| `expected_overlap_px` | int or None | None | from `overlap` × tile size when None |
| `tolerance_px` | int or None | None | how far along the raster a shift may land from the prediction |
| `skew_px` | int or None | None | how far across it may land |
| `solve_gauge` | str | `'anchor'` | or `'lowest_site'` |
| `blend` | str | `'feather'` (unchanged) | adds `'feather_distance'` (931) |
| `compose` | str | `'bands'` | `'windows'` writes window tiles (`windows_over`), as OPS segmentation wants |
| `compose_window_px` | int | 4096 | window edge |
| `compose_window_overlap` | int | 256 | window overlap |
| `second_acquisition` | str or None | None | folder of a second acquisition (another magnification or another round) to align onto the stitched reference |
| `second_model` | str | `'similarity'` | or `'tile_affine'` (930) |
| `second_points_channel` | int | 0 | nuclear channel used for nuclei points in both acquisitions |
| `second_tile_min_nuclei` | int | 200 | as `ops_align_tile_min_nuclei` |
| `second_match_radius` | float | 3.0 | as `ops_align_match_radius` |
| `second_trim_passes` | int | 3 | as `ops_align_trim_passes` |

The OPS keys stay. One mapping, `OPS_TO_ALIGN_KEYS`, documents the equivalence, and a test keeps it complete.

### Public callables
- `spacr.stitch_core.*`: the functions in the table above.
- `spacr.align.align_second_acquisition(settings) -> DataFrame`: the per-field transforms, in the same columns as `ops_phenotype`'s `align_*`.
- `align_folder` keeps its signature.

### GUI
- The Align & Stitch screen gains the keys above under alpha. Keys that do not apply are greyed out (for example, `skew_px` when `acceptance='ncc'`).
- The plan preview (`preview_only`) shows per-edge peak ratio and NCC side by side.
- The second-acquisition mode adds a results card: within-2-px share per field, and the tile-group map.

### `image_stitch.py`
- It keeps its arrangement search and stage-position reading; that is a different problem: unknown order, one field.
- Its pair scoring and solve move to `register_strip_pair(acceptance='ncc')` and `solve_positions` **only if** its existing tests stay identical. Otherwise it is left as is and listed as the one documented exception (open question 3).

## Acceptance tests (CPU unless marked; `CUDA_VISIBLE_DEVICES=''`, at most 4 test workers)
1. **Legacy identity.** On the Align & Stitch example (9 Funk tiles, snake-column, 213 px overlap) and the synthetic 10 × 10 grid, `align_folder` with `align_engine='legacy'` writes the same stack bytes, the same `align_coordinates` rows and the same plan text as today.
2. **OPS unchanged by the move.**
   - `stitch_well` on plates A and B, well A1, gives identical integer edge shifts and acceptances (624 of 624). Placements are within ≤ 1e-9 px of production `ops_geometry`. The canvas is 26,855 × 26,865 (A) and 26,856 × 26,865 (B).
   - `compose_window` (931) gives identical bytes.
   - The 930 step-6 tables are identical.
   - The 932 tests pass.
3. **Core = OPS through Align & Stitch.** On the same 333 tiles of A1, `align_folder(align_engine='core', acceptance='peak_ratio', strip_fraction=0.30, solve_gauge='lowest_site', blend='feather_distance', compose='windows', ...)` gives:
   - the same integer edge shifts and acceptances as `stitch_well`;
   - placements within ≤ 1e-9 px;
   - windows identical to `compose_window`.
4. **Negative control.** The 50 non-adjacent pairs of `placement_arms.py` are rejected (0 of 50) under the core with `peak_ratio`, and under `both`.
5. **Accuracy on non-OPS data.** On the synthetic grid with known shifts (sub-pixel ground truth), and on one non-OPS tiled dataset (proposed: a BBBC or spaCR example tiled acquisition):
   - report placement error against the truth for `legacy` and `core`;
   - pre-set criterion: core error ≤ legacy error + 0.5 px at the 95th percentile;
   - seam score (931's metric) for `feather` vs `feather_distance`.
6. **Second-acquisition mode.** On plate A well A1, `align_second_acquisition` with `second_model='tile_affine'` gives moved nuclei identical to `ops_merge` (930 step-6: 1,271 fields, max difference 0.0 px).
7. **Determinism.** Two core runs give byte-identical coordinates, under different `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` values.
8. **Shared objects.** Every name re-exported from `ops_register`, `ops_solve`, `ops_compose` and `ops_merge` is the `stitch_core` object. A grep ratchet fails if a second `def phase_correlate`, `def solve_positions` or `def compose_window` appears under `spacr/`.
9. **Time vs the current path** (marked slow; 1 pinned core, BLAS threads 1, tiles preloaded; 3 interleaved rounds; plates A and B well A1). Report CPU s per well for:
   - `align` legacy;
   - `align` core in float64;
   - `align` core in float32, after the speed card;
   - OPS `stitch_well`.
   Pre-set: core float64 ≤ legacy (the sub-pixel skimage path upsamples and is expected to be slower). Core float32 within 5 % of `stitch_well` (the same code). Report wall time for parallel edges separately.

## Translations / catalog obligations
- Every `stitch_core` function (moved docstrings plus new ones) and `align_second_acquisition` ship in the same commit with:
  - their arrival contracts;
  - nine-locale records;
  - the regenerated help index, API catalogs, setting consumer map and settings flow.
- Every new Align & Stitch tooltip gets nine-locale records.
- Moved docstrings keep their existing translations; the catalog keys move with the qualified name.
- Warm the API catalogs first.
- Run the pre-handover ratchets (raw-reader, nested-function, documented-callable, hash-comment).

## Dependencies on items 928–932
- **930:** the affine functions move into the core; the second-acquisition mode exposes them. 930's tests must pass unchanged.
- **931:** must land on nightly first. `feather_weights('distance')` is its weight function, and its settings-schema version and default-change table get the `align_engine` row.
- **932:** must land first. `FFTEngine` and `_read_peak` move into the core, and its precision and backend settings are mirrored as `registration_precision` and `registration_backend`.
- **Stitch speed card:** float32 / in-place for **stitch** registration is adopted only after the card's identity gate passes (identical integer shifts on A, B, C and D A1). Until then the core default stays float64.
- 928 and 929: none.

## Alpha status
Alpha. `align_engine='core'` and all new keys sit under one `ALPHA_FEATURES` entry. The default stays `legacy`. Making `core` the default is a separate maintainer decision after tests 3, 5 and 9.

## Open questions for the maintainer
1. Acceptance for general users: OPS uses peak ratio against a layout prediction; `align` uses NCC of the shifted strips. Should the core default for Align & Stitch be `'both'` (safer, slightly slower), or the OPS rule?
2. Sub-pixel: OPS shifts are integers, while `align` refines to 1/10 px. Is sub-pixel worth keeping for general users? It changes composed pixels by interpolation. Or should the core be integer-only, as OPS is?
3. Should `image_stitch.py` (import-time, unknown arrangement) also move onto the core, or stay a documented exception?
4. Is a second-acquisition (cross-magnification) mode wanted in Align & Stitch at all, or should 930 stay OPS-only?
5. Once the core is proven, should the `legacy` engine be removed (one implementation), or kept for reproducing earlier runs?
