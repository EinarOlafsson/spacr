# Item 1006: OPS `ph_flatfield` builds the DNA map with the nuclear-median fit (item 912) as an option

> **Extended by the maintainer (2026-11-21).** The fit is chosen per channel, not for DNA only:
> `ops_flatfield_channel_fit` maps a channel to `pixel_median`, `polynomial`, `nuclear_median`
> or `cell_median` (unset: DNA `polynomial`, the rest `pixel_median`, today's bytes). The
> object-median channels are pending after `ph_flatfield` and built by the plate phase
> `ph_flatfield_objects`; `ops_flatfield_object_fields` replaces `ops_flatfield_dna_fields`.
> `ops_flatfield` is now `'exact'` by default (schema 952, DEFAULT_CHANGES row `'none'`).
> The winners of the flat-field accuracy report become the per-channel defaults later.
> Names below that mention DNA describe the original single-channel design.


- **Lifecycle:** future (alpha), behind "Show alpha features" (ALPHA_FEATURES[1006]); the default stays `polynomial`.
- **Number:** 1006. Checked: proposals on origin/nightly c9fddfc769 end at 1003; 1004 is the OPS Live tile-affine captions lane (ws-offload-codex-tr-1004); 1005 is taken by another lane (coordinator, 2026-11-21).
- **Related:** 912 (`spacr.ops_flatfield.build_nuclear_flat`, `write_nuclear_flat`, the `flat_nuclear` / `flat_validate` phases), 933 (the same map in the general illumination correction), the final Funk runs (`ops_flatfield` `exact`, signed 2026-11-21).
- **Base:** origin/nightly. **Paper-critical:** yes. It must land before the final-run pilot, or the pilot runs with a prebuilt map folder (see "Interim").
- **Rule:** a spaCR settings-file run (spacr-run, or GUI Import settings + Run) must reproduce the reference outputs identically on the validation data. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.

## Why (measured, pre-set metric)

The maintainer's rule (2026-11-21): "spaCR should use what is most accurate".

Today OPS `ph_flatfield` builds the DNA (DAPI) map with `build_flat_maps(..., polynomial_channels=(DAPI,))`:
- pixel median of 204 raw fields;
- Gaussian smoothing, sigma 6 map px;
- 4th-order 2-D polynomial;
- floor 0.25.

On plate A this gives md5 6aebe237 (spacr_equiv_852). The nuclear-median map of item 912 (md5 f0dbfb9d, the map the fs3b runs used) is only reachable through the separate `flat_nuclear` phase. That phase needs an already measured, uncorrected nucleus table, so it does not fit the `ph_flatfield → ph_segment → ph_measure (exact)` order of a single run.

The design card `reanalysis/experiment_designs/20261121_dapi_flatfield_accuracy.md` was fixed before the results. Result: `reanalysis/dapi_flatfield_1121/REPORT.md`. Setup: Funk plate A well A1, 1,526,716 non-edge nuclei, 1,271 fields, 1,000 field bootstraps. Same nuclei and raw values in every arm; only the DAPI map differs. The nuclear-median map was built with every A1 field left out.

| Metric (lower = more accurate) | Uncorrected | Polynomial (today) | Nuclear median, A1 left out | Polynomial − nuclear (95 % CI) |
|---|---|---|---|---|
| CV of the G1 peak over 8 × 8 field positions (primary) | 0.124 | 0.059 | 0.027 | 0.033 [0.026, 0.036] |
| Position R² of G1 log2 DNA (64-bin factor) | 0.085 | 0.026 | 0.0014 | 0.025 [0.024, 0.026] |

The other metrics are in the report. The full plate A confirmation is in the report.

## What to build

1. **Setting** `ops_flatfield_dna_fit`: `'polynomial'` (default, today's bytes) or `'nuclear_median'` (alpha). It applies to `_NUCLEAR`, the DNA channel `ph_flatfield` already treats separately. Other channels keep the pixel-median Gaussian maps, byte for byte.
2. **Phase order with `nuclear_median`.** `ph_flatfield` writes the non-DNA maps as today and records the DNA map as pending in `flat_info.json`. A new plate phase, `ph_flatfield_dna`, runs after `ph_segment` of the flat-field wells and before `ph_measure`:
   - It reads the raw DNA channel of the chosen fields through the phenotype stager (no NAS hot path).
   - It takes the field's `ph_segment` nucleus masks and computes, per nucleus, the raw DNA mean, the centroid and the edge flag (`field_nucleus_means` exists).
   - It builds the map with `build_nuclear_flat` (fit `local_quadratic_log`, sigma 32 px, ridge 1e-6, floor 0.10, centre 18 × 18 cells, block-centre grid and lookup unless `ops_flatfield_map_grid` / `ops_flatfield_lookup` say otherwise). The offset comes from `ops_flatfield_dark`, else the plate's DNA offset that `ph_flatfield` measured, never 0.
   - It writes the map into the same maps folder (`write_nuclear_flat` semantics: never overwrite, md5 sidecar, `md5_record`).
   - `ph_measure` (`exact` or `analytic`) refuses to start while the DNA map is pending, and the message names `ph_flatfield_dna`.
3. **Fields.** `ops_flatfield_dna_fields` (default `all`: every placed field of the wells run; or an integer N drawn with seed 0; or a `[[well, site], ...]` file). The 912 reference used every plate A nucleus (about 10 M), so `all` is the default. The phase's cost (one DNA-channel read per field + a mask read) is measured and recorded in the report. Held-out fields (`ops_flatfield_holdout_fields` / `_n` 24, seed 0) are left out of the build and re-measured by `flat_validate`, which runs on the same folder unchanged.
4. **Prebuilt maps:** unchanged. A folder with a complete `flat_info.json` in `ops_flatfield_maps` is reused and never rebuilt.
5. **Report:** `ph_flatfield_dna` writes the nucleus count, the nuclei per map cell (min / median / corner), the corner mean, the % at floor and the md5 to the run report and the run record.
6. **GUI:** the OPS settings view shows `ops_flatfield_dna_fit` under Flat-field, as an alpha choice with a tooltip in 9 locales. No other screen changes.

## Interim (if this item has not landed before the final-run pilot)

Point `ops_flatfield_maps` at a spaCR-built folder: `flat_nuclear` (item 912, `ops_flatfield_map_grid` `lookup`), with the plate's pixel maps copied in by `write_nuclear_flat(base=...)`. For plate A that is md5 f0dbfb9d; for plate B, 088f3fa0. Plate C has no nuclear map yet and needs its `flat_nuclear` build first. `ph_flatfield` then reuses the folder.

## Acceptance

- **Default unchanged:** with `ops_flatfield_dna_fit` `polynomial`, every map byte-identical to today: plate A DAPI 6aebe23775a9307f28d2802fd2c6ae79, GFP 7964951a, A594 d4e5c640, AF750 a61f9299.
- **Reproduction:** with `nuclear_median`, `ops_flatfield_map_grid` `lookup`, `ops_flatfield_lookup` `zoom_endpoints` and the 912 reference nucleus table (`ops_flat_v2/out/nuclei_plateA.parquet`, the 24 held-out fields), the DNA map equals f0dbfb9da3e6ab3c1e38a63485491146. Through the new phase, with the run's own masks: every held-out ring with ≥ 100 cells within 0.03 of 1.0 and within 0.01 of the 912 reference ring values (plates A and B).
- **Accuracy:** on plate A A1, the G1-peak position CV (the card's M1, `reanalysis/dapi_flatfield_1121/scripts/dapi_flat_accuracy.py` on the run's tables) lies within the reference CI [0.0257, 0.0279].
- **Order and refusals:** `ph_measure` before `ph_flatfield_dna` refuses; a changed DNA map refuses resume (md5 record); a missing offset refuses.
- **Tests (CPU):**
  - A synthetic plate with a steep vignette and nuclei of known DNA content on a background: the `nuclear_median` run recovers the corners within 1 %, and the polynomial run does not.
  - Default-path byte identity.
  - The phase-order refusal.

## Not in scope
Exact vs analytic correction (unchanged), and the non-DNA channels (pixel-median maps are byte-identical between spaCR and the fs3b runs).
