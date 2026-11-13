# Item spec: the OPS corner-true flat-field (nuclear-median map, local log-quadratic fit, block-centre lookup) in spaCR's general illumination correction

- **Lifecycle:** future (alpha), behind "Show alpha features". Every new key and choice goes under one new `ALPHA_FEATURES[<this item>]` entry. The existing `illumination_*` defaults do not change.
- **Number:** 933 (reserved 2026-11-13).
- **Overlap with an existing spec:** this request is the unbuilt half of 912. Under "What to build", point 1 of `ITEM_ops_flatfield_v2.md` put the estimator "in `spacr.illumination` ... next to `'polynomial'` and `'smooth'`". The 912 implementation shipped it in `spacr/ops_flatfield.py` as `ops_flatfield_*` keys only. Its feature record says the keys were renamed "because the OPS run already owns ops_flatfield_maps and Measure's illumination_dark is a float". 912 is closed (DONE), so this is filed as a new item and not as an addendum. If the coordinator prefers an addendum, everything below can be appended to `ITEM_ops_flatfield_v2.md` as "Part 2: the general module".
- **Rule:**
  - With the new estimator, a spaCR settings-file run gives the same map bytes (md5) as the OPS path for the same nucleus table and the same options.
  - Every existing estimator gives byte-identical models and corrected pixels to today.
  - Every new documented callable ships with its arrival contract and nine-locale records in the same commit.
- **Code basis:** origin/nightly (46f259ea23 at the time of writing).
- **Paper-critical:** no for the OPS paper, which already uses the OPS path. Yes for the claim that spaCR's general modules carry the OPS improvements.

## Problem and goal
Microscope fields are darker at the edges and corners than at the centre. spaCR's general illumination correction (Measure, Mask generation and the pipeline v2 stages) fits a smooth polynomial or a Gaussian to the per-pixel median of many fields. That median is mostly background between cells, and the polynomial cannot follow a steep corner fall, so the corners stay under-corrected.

The OPS path (`spacr/ops_flatfield.py`, item 912) builds the map from the nuclei themselves instead:
- the median nuclear DNA intensity per 16 px block;
- a count-weighted local quadratic fit of the log;
- an explicit camera offset;
- a lookup at block centres.

It keeps intensity flat into the field corners. The maintainer's figure for the corner deviation is **0.06 for spaCR vs 0.46 for Brieflow**.

**Goal:** any spaCR user with a nuclear stain can choose this estimator in the general illumination correction. It then reaches Measure, the later-mask segmentation stages and pipeline v2, with no second implementation of the map maths.

## Current state (verified on origin/nightly)

### General module: `spacr/illumination.py` (2,861 lines)
- **Model.**
  - `IlluminationField` (l. 235) stores a full-resolution `(C, Y, X)` `flatfield`, normalised to **mean 1**, plus a per-channel `dark` and `estimator` (`'polynomial' | 'smooth'`, or `'harmony'` / `'vendor image'` for vendor profiles).
  - `IlluminationModel` (l. 428) groups the fields; `load_illumination_model` is at l. 553.
- **Correction.**
  - The formula is `corrected = (observed - dark) / flatfield`. The offset is **not added back**.
  - `IlluminationCorrector` (l. 1336) is a measure preprocessing hook (`spacr.measure_hooks.register_preprocessing_hook`). It reaches workers through `MODEL_ENV_VAR` and `SPACR_MEASURE_HOOKS` (`enable_illumination_correction`, l. 2187).
- **Estimator.**
  - `estimate_illumination(src, channels, *, per_plate, estimator, degree, max_fields, grid, dark)` is at l. 779 and accepts only `'polynomial'` and `'smooth'` (l. 825).
  - The input is a per-pixel median of binned merged fields (`_bin_image`, `_relative_profile`).
  - `_smooth_surface` (l. 709) interpolates back with a `(factor - 1)/2` centre convention. 912 flagged this as a half-pixel difference from the reference `16 i + 8` convention.
- **Settings.**
  - `illumination_settings` (l. 2710), registered through `register_defaults` (l. 2824). The keys are:
    - `illumination_correction` (False)
    - `illumination_model` ('')
    - `illumination_estimator` ('polynomial')
    - `illumination_degree` (4)
    - `illumination_per_plate` (True)
    - `illumination_max_fields` (50)
    - `illumination_dark` (0.0, float)
    - `illumination_on_missing` ('error')
    - `illumination_qc` (True)
    - `illumination_vendor_profile` ('')
    - `illumination_vendor_channel_map` ('')
- **Entry points.**
  - `prepare_illumination_model` (l. 2550)
  - `prepare_illumination_correction` (l. 2636, Measure)
  - `prepare_segmentation_illumination` (l. 2673, mask generation)
  - `illumination_qc` (l. 2352: position-intensity slope, radial profile `_radial_profile` l. 2526)
- **Users outside OPS.**
  - `spacr/measure.py:12335`: `measure_crop` calls `prepare_illumination_correction` and `validate_measurement_illumination_inputs`.
  - `spacr/io.py:4392` and `:4506`: the mask-generation resume and `prepare_segmentation_illumination`.
  - `spacr/core.py:640`: the segmentation resume.
  - `spacr/pipeline_v2.py:1718`
  - `spacr/embedding_profiles.py:256`: `load_illumination_model` for embeddings.
  - `spacr/qt/bridge.py:2125`: the GUI "Illumination" route.
  - `spacr/settings.py:1459`, `:2210`: settings registration.

### OPS module: `spacr/ops_flatfield.py` (1,429 lines; item 912)
- Pixel-median maps: `build_flat_maps` (l. 97), `flat_planes`, `correct_field` (exact), `mask_pass` and `correct_features` (analytic). The model is `observed = b + flat·(true − b)`, so `corrected = (observed − b)/flat + b`.
- Nuclear-median maps:
  - `nuclear_median_grid` (l. 530)
  - `normalise_flat` (l. 567: centre-normalised, floor 0.10)
  - `lookup_positions` (l. 588)
  - `_local_fit` (l. 605)
  - `fit_nuclear_flat` (l. 638; fits `NUCLEAR_FITS = ("local_quadratic_log", "gaussian", "local_linear")`)
  - `flat_to_field` (l. 721; lookups `FLAT_LOOKUPS = ("block_centre", "zoom_endpoints")`, grids `FLAT_MAP_GRIDS`)
  - `build_nuclear_flat` (l. 815), `write_nuclear_flat` (l. 897), `record_flat_md5` (l. 964)
  - `validate_nuclear_flat` (l. 1103: held-out exact and analytic re-measure, radial table with 2,000 field bootstraps)
  - `run_flat_phases` (l. 1319; OPS phases `flat_nuclear`, `flat_validate`)
- Settings: `ops_flatfield_*`, 24 keys (`spacr/ops_settings.py:76-103`). The offset `ops_flatfield_dark` is per plate and **required**; it is never 0.

### Differences that block a plain hand-over
1. **Normalisation.** General maps have mean 1; OPS nuclear maps have centre 1 (the central 18 × 18 cells). A centre-normalised map raises every corrected value by the map's mean-to-centre ratio.
2. **Offset.** The general path subtracts `dark` and does not add it back. OPS adds `b` back. The two give different absolute intensities for the same map.
3. **Map resolution.** The general model stores a full-resolution plane. OPS stores a 185 × 185 block map plus its lookup convention.
4. **Input.** OPS needs a per-nucleus table: centroid plus nuclear mean of the DNA channel with no flat applied. The general path has no nuclei at segmentation time. Nuclei exist after mask generation (the `nucleus` table in `measurements.db`, or the nucleus masks in `masks/`).
5. **Offset type.** `illumination_dark` is a single float; OPS needs one value per plate (a dict, or the plate's `flat_info.json`).
6. **Scope.** The OPS nuclear map exists for the DNA channel only; the other OPS channels use the pixel-median map. The general estimator setting is one value for all channels.

## Proposed design

### Shared core, with no duplicate maths
- Move the estimator-independent nuclear-map functions out of `ops_flatfield.py` into a neutral module, `spacr/flatfield_core.py`:
  - `nuclear_median_grid`, `fit_nuclear_flat`, `normalise_flat`, `lookup_positions`, `flat_to_field`, `holdout_fields`, `radial_table`, `field_nucleus_means`;
  - the constants `NUCLEAR_FITS`, `FLAT_LOOKUPS`, `FLAT_MAP_GRIDS`, `RADIAL_EDGES`.
- `ops_flatfield.py` re-exports them under the same names, so OPS callers and pickled references do not change.
- `spacr.illumination` imports from `flatfield_core` and never from `ops_*`.
- Test: the functions in `ops_flatfield` are the same objects as in `flatfield_core` (`is`).

### Estimator
- `estimate_illumination(..., estimator='nuclear_median', nuclei=<DataFrame | path | None>, ...)`.
- When `nuclei` is None, the table is read from the run's own nucleus masks:
  - the `nucleus` table of `measurements/measurements.db` (centroid plus `nucleus_channel_<k>_mean_intensity`, measured with illumination off);
  - failing that, `masks/` plus the raw stack, via `field_nucleus_means`.
- If the nucleus table was itself measured with a correction applied, the call is refused, and the error names the column.
- **Per-channel estimators.** `illumination_estimator` accepts a dict as well as a string: `{"0": "nuclear_median", "1": "polynomial", ...}`. The nuclear channel gets the nuclear map. For the other channels the default is the **existing** estimator (polynomial), as in OPS.
- **New model type** in `IlluminationField`. New attributes: `block_px`, `lookup`, `map_grid`, `centre_offset`, `normalisation ('mean' | 'centre')`, `offset_restore (bool)`, and `block_map` (the stored 185 × 185 map).
  - `flatfield` is built lazily through `flat_to_field` at the field shape, so memory stays one map per channel.
  - The model file gets a format version, and an old model loads unchanged.
- **Correction.** `IlluminationCorrector` applies `(observed − b)/F + b` when `offset_restore` is True, which is the default for `nuclear_median` and matches OPS. Otherwise it applies today's formula. The new mode reuses the integer rounding and rescaling rules already in the corrector.

### Settings (all alpha; registered in `illumination_settings`)
| Key | Type | Default | Meaning |
|---|---|---|---|
| `illumination_estimator` | str or dict | `'polynomial'` (unchanged) | adds `'nuclear_median'`; a dict sets one estimator per channel |
| `illumination_nuclear_channel` | int or None | None | the DNA channel; required by `nuclear_median` |
| `illumination_nuclei` | str or None | None | path to a per-nucleus table; None reads the run's nucleus masks or table |
| `illumination_dark_per_plate` | dict or str or None | None | plate → offset, or a path to a JSON / `flat_info.json`. `nuclear_median` refuses to run without it, or without a non-zero `illumination_dark`; 0 is not accepted, as in 912 |
| `illumination_block_px` | int | 16 | map cell in field px |
| `illumination_fit` | str | `'local_quadratic_log'` | or `'gaussian'`, `'local_linear'` |
| `illumination_sigma_px` | float | 32.0 | Gaussian weight σ in field px |
| `illumination_ridge` | float or None | None | None takes the per-fit default (1e-6 quadratic, 1e-9 linear) |
| `illumination_floor` | float | 0.10 | lowest map value |
| `illumination_centre_cells` | int | 18 | the central square the map is normalised to |
| `illumination_lookup` | str | `'block_centre'` | or `'zoom_endpoints'`, which reads maps made the earlier way |
| `illumination_map_grid` | str | `'block_centre'` | or `'lookup'` |
| `illumination_centre_offset` | float | 8.0 | field-px offset of a block's centre |
| `illumination_holdout_n` | int | 24 | held-out fields for validation |
| `illumination_seed` | int | 0 | held-out field draw |
| `illumination_validate` | bool | False | run the held-out radial validation and write its table beside the model |

The `ops_flatfield_*` keys stay. The OPS engine translates them into these keys through one mapping (`OPS_TO_ILLUMINATION_KEYS`), which a test keeps complete.

### Public callables
- `spacr.illumination.estimate_illumination` (extended)
- `spacr.illumination.validate_illumination(settings) -> DataFrame`: the radial table and the corner-deviation summary. It wraps `validate_nuclear_flat`.
- `spacr.flatfield_core.*`: the moved functions, with their docstrings carried over and their arrival contracts moved with them.

### Stage reach
- **Measure.** Masks exist, so `nuclear_median` works directly.
- **Mask generation.** The first nucleus segmentation cannot use a map built from nuclei it has not found yet. `prepare_segmentation_illumination` therefore:
  - refuses `nuclear_median` for the stage that segments the nuclei, with a message pointing to `illumination_for_later_masks`;
  - allows it for the masks made after the nuclei (cell and pathogen).
- **Pipeline v2.** Same rule as mask generation.
- **Embeddings.** `load_illumination_model` handles the new model type.

### GUI
- The Illumination settings group gains the keys above, all under the alpha gate. Keys that do not apply are greyed out unless the estimator includes `nuclear_median` (item 106 behaviour).
- The QC figure (`illumination_qc`) gains a radial panel: median corrected intensity per radius ring relative to the centre, with bootstrap CIs and the corner-deviation number in the title.
- No new screen.

## Acceptance tests (CPU, `CUDA_VISIBLE_DEVICES=''`, at most 4 test workers)
1. **Default identity.** For each existing estimator (polynomial, smooth, vendor), the model `.npz` bytes and the corrected pixels on the bundled example plate are byte-identical before and after.
2. **Identity with OPS.** For the plate A and plate B nucleus tables and offsets of 912, `estimate_illumination(estimator='nuclear_median', illumination_map_grid='lookup')` writes block maps with the 912 md5s:
   - plate A: `f0dbfb9da3e6ab3c1e38a63485491146`
   - plate B: `088f3fa027f58bfb3e44fac7ed05e997`
   The default grid gives the OPS default map byte for byte.
3. **Corrected pixel identity.** On the 24 held-out plate A fields, the corrected DNA plane from `IlluminationCorrector` equals `ops_flatfield.correct_field` with the same map, in float32 (`np.array_equal`).
4. **Measure identity.** `measure_crop` on the same fields gives nucleus DNA mean intensities equal, within 1e-6 relative, to the exact OPS re-measure (`validate_nuclear_flat`, exact arm).
5. **Corner check on non-OPS data.** On one non-OPS dataset with a nuclear stain (proposed: the bundled spaCR example plate, plus one BBBC021 plate; see the open questions):
   - the outermost-ring deviation of corrected nuclear intensity is reported for `polynomial` and `nuclear_median`;
   - pre-set criterion: `nuclear_median` ≤ `polynomial` in every ring with ≥ 100 nuclei, upper 95 % CI;
   - a shuffled-position control (nucleus positions permuted within a field before the map is fitted) must give a flat map, with deviation ≤ 0.02 from 1 everywhere.
6. **Refusals.**
   - `nuclear_median` with no offset, or with offset 0, raises.
   - A nucleus table measured with correction on raises.
   - `nuclear_median` on the nucleus segmentation stage raises with the message pointing to `illumination_for_later_masks`.
7. **Shared core.** `ops_flatfield.fit_nuclear_flat is flatfield_core.fit_nuclear_flat`, and so on for every moved name. OPS 912 tests pass unchanged.
8. **Time vs the current path** (marked slow, not in the default run). On the same plate and machine, report the estimate time of `polynomial` (50 fields) against `nuclear_median` (the existing nucleus table; the image re-read is not counted when the table exists). Pre-set expectation: `nuclear_median` ≤ 2× `polynomial`, because it reads no images. Report the times and the ratio; do not assert seconds.

## Translations / catalog obligations
- New documented callables (`validate_illumination`) and every moved `flatfield_core` function ship in the same commit as the code, with:
  - their arrival contracts;
  - nine-locale records (de, es, fr, hi, is, ko, pt, sv, zh_CN);
  - the regenerated help index, API catalogs and setting consumer map.
- Every new tooltip gets nine-locale records.
- Run `build_documentation_i18n.py --assemble` before the test run (warm catalogs).

## Dependencies on items 928–932
- None functionally.
- 931's settings-schema version and default-change table must list the new keys. They are new, so the table needs no rows; the schema version still increments.

## Alpha status
Alpha. All new keys go under one `ALPHA_FEATURES` entry; the default is unchanged.

## Maintainer decisions (2026-11-13, AskUserQuestion in the master session)
Spec defaults: centre-normalised maps (identical to OPS); offset restore only with nuclear_median; no non-DNA cell-median map yet (later item if wanted); validation on the bundled example plate plus one BBBC021 plate; refuse instead of a two-pass mode.

## Open questions for the maintainer (answered above)
1. **Normalisation.** Centre-normalised maps raise corrected intensities relative to today's mean-normalised maps. Should general nuclear maps be centre-normalised (identical to OPS) or mean-normalised (same scale as the other general estimators)? The spec assumes centre, for OPS identity.
2. **Offset restore.** Should `(I − b)/F + b` become available for the existing estimators too (a separate `illumination_restore_offset` key), or only come with `nuclear_median`?
3. **Non-OPS nuclear channels.** Is a nuclear map also wanted for non-DNA channels? A cytoplasmic stain has no "nucleus median", but a per-cell median map could be built the same way from cell masks.
4. **Validation dataset outside OPS.** Is the bundled example plate plus one BBBC021 plate acceptable for the corner check?
5. **Segmentation stage.** Should a two-pass mode be offered (segment nuclei uncorrected, build the nuclear map, re-segment corrected), or is the refusal enough?
