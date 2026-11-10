# Item spec (912): DNA flat-field map from per-nucleus intensities (nuclear median per map cell, count-weighted local quadratic fit of the log), offset handling, block-centre lookup, held-out exact re-measure validation, per-plate maps with md5 records

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** 912. Check `features/feature_list/` and the claims files first. Related:
  - `ITEM_ops_phenotype_run_v2.md`: its flat-field step must take the map this item builds;
  - 809–813 (OPS phenotype run).
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs identically on the validation data: the same map bytes (md5), the same per-nucleus gains and corrected values, the same radial table. Every seed is fixed as in the reference. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.

## Why
A DNA flat-field map built the usual way (pixel-wise median of raw fields, wide Gaussian smoothing, a 4th-order polynomial, a 4× gain limit) left the field corners of an OPS phenotype acquisition at about half of the centre after correction, measured by exact re-measurement on held-out fields. Four causes were measured one at a time:
1. the pixel median of a DNA image is mostly background between nuclei and falls less steeply than nuclear signal;
2. wide smoothing and a global polynomial cannot follow the steep corner fall;
3. the gain limit cuts the deepest corner;
4. the lookup offset: `scipy.ndimage.zoom(map, 2960/185, order=1)` puts map pixel i at field pixel i·2959/184 (zoom endpoints), not at the centre of the 16 px block it was measured on (16 i + 8). The corner block lands up to 8 px further out, where the fall is steepest.

spaCR's `spacr.illumination.estimate_illumination` has only `'polynomial'` and `'smooth'` estimators over per-pixel medians of binned fields; neither estimates from nuclei, neither fits locally in log space, and the OPS map path reads maps with the zoom lookup.

## Reference implementation
1. **Per-nucleus table:** per nucleus `well, site, label, nucleus_cy, nucleus_cx`, nuclear DNA mean (run value, no flat), all wells of the plate.
2. **Held out:** 24 re-measure fields (4 per well), never used to build.
3. **Map cells:** H = 185 cells of 16 × 16 field px; cell index `clip(cy // 16, 0, 184)`, same for x. Per cell: median of the training nuclei's nuclear DNA mean and their count; empty cells get weight 0.
4. **Offset:** b = the plate's DNA camera offset (from the plate's flat-field info; measured per plate).
5. **Fit:** per map cell, weighted least squares of `log(clip(median − b, 1, None))` on the 6 quadratic terms (1, x, y, x², xy, y²) in map-pixel coordinates; weights = count × Gaussian (σ = 2 map px = 32 field px, `scipy.ndimage.gaussian_filter(mode="constant")` on every moment); ridge 1e-6 on the 6 × 6 system.
6. **Evaluate:** at the field positions the zoom lookup assigns to map pixel i, `t = (i·2959/184 − 8)/16` in block units, with the coefficients of the nearest block (`rint(t)`); value `exp(·) + b`.
7. **Normalise:** `clip((value − b)/(centre − b), 0.10, None)`, centre = mean of the central 18 × 18 cells; stored float32, 185 × 185.
8. **Losing arms** (kept as regression cases): `old` (pixel median of 204 fields, Gaussian σ 6 map px, 4th-order polynomial, floor 0.25), `pix_s6`, `pix_s2` (pixel median of 160 fields, Gaussian only, floor 0.10), `nuc_s1`, `nuc_s3` (nuclear median, count-weighted normalised Gaussian σ 1 / 3, nearest-valid fill), `nuc_ll1`, `nuc_ll2` (count-weighted local plane), `nuc_lq2_centres` (the chosen fit evaluated at block centres 16 i + 8 instead of the lookup grid).
9. **Validation:** per held-out nucleus, the analytic gain g = mean of 1/F over the nucleus mask (corrected `(v − b)·g + b`) and the exact corrected mean of `(I − b)/F + b` on the raw DNA image (F = the map zoomed to 2,960 px, order 1), re-measured through `spacr.ops_sample.measure_phenotype_field` (texture off). Radius bins [0, 500, 1000, 1400, 1600, 1800, 1900, 2000, 2100] px from the field centre; per bin the median over the 0–500 px median; 95 % CI from 2,000 bootstraps over fields. Sanity rows: CPU raw mean vs run table r = 1.0000000; old-map gains recomputed equal the run's gains.
10. **md5 record:** the map md5 is written on first use; a run whose maps changed afterwards refuses.

## What to build
1. **Estimator** in `spacr.illumination`: `estimator='nuclear_median'` on a per-nucleus table (centroid + nuclear mean of the channel, no flat applied), next to `'polynomial'` and `'smooth'`. The table comes from the run's own nucleus masks (OPS raw intensities; Measure `nucleus` table), so no second image pass is needed.
2. **Fits:** `'local_quadratic_log'` (default for this estimator), `'gaussian'`, `'local_linear'` (losing arms, kept for regression and supplement).
3. **Offset:** b is subtracted before the log and before normalising, and added back on correction (exact `(I − b)/F + b`; analytic `(v − b)·g + b`). A missing offset is an error for this estimator, not 0.
4. **Lookup alignment:** one helper that maps a map to field pixels with map pixel i at `block·i + centre_offset` (default 16 i + 8, the reference convention; the existing `_smooth_surface` uses `(factor − 1)/2` = 7.5, a half-pixel difference that must be pinned, not inherited). The `'zoom_endpoints'` lookup is kept as a legacy mode with the map grid built to match it (`illumination_map_grid='lookup'`), so tables made with the reference maps reproduce bit for bit. New maps default to block-centre evaluation and block-centre lookup.
5. **Per-plate maps** (`per_plate=True`), one file per plate and channel, with a JSON sidecar: offset, floor, held-out fields, training-nucleus count, centre value, corner mean, % at floor, estimator, fit, σ, lookup, package versions.
6. **md5 record** in the run folder (`illumination_md5.json`); a resume or re-measure with a different map refuses with both md5s in the message.
7. **Held-out validation** (`validate_illumination(settings)`): analytic and exact corrected values per held-out nucleus, the radial table with field bootstraps, the largest ring deviation (bins ≥ 100 cells) and the "beyond 1,800 px" median.

## Settings (alpha)
| Key | Default | Meaning |
|---|---|---|
| `illumination_estimator` | `'polynomial'` (today) | `'nuclear_median'` adds this item |
| `illumination_nuclear_channel` | None (required for `nuclear_median`) | the DNA channel index |
| `illumination_block_px` | 16 | map cell size in field px |
| `illumination_fit` | `'local_quadratic_log'` | or `'gaussian'`, `'local_linear'` |
| `illumination_sigma_px` | 32 | Gaussian weight σ in field px (= 2 map px) |
| `illumination_ridge` | 1e-6 | ridge on the local normal equations |
| `illumination_floor` | 0.10 | lowest map value |
| `illumination_dark` | None (required) | camera offset per plate (dict plate → counts) |
| `illumination_centre_cells` | 18 | central square (map cells) the map is normalised to |
| `illumination_lookup` | `'block_centre'` | or `'zoom_endpoints'` (legacy) |
| `illumination_map_grid` | `'block_centre'` | or `'lookup'` (evaluate at the zoom positions) |
| `illumination_centre_offset` | 8.0 | field-px offset of a block's centre |
| `illumination_holdout_fields` | None | explicit list `well:site`; else `illumination_holdout_n` (24) + `illumination_seed` (0) |
| `illumination_radial_edges` | [0, 500, 1000, 1400, 1600, 1800, 1900, 2000, 2100] | radius bins (px) |
| `illumination_bootstrap` | 2000 | field bootstraps for the radial CIs |
| `illumination_min_bin_cells` | 100 | bins below this are not scored |

## Validation data and acceptance
- **Data:** Funk et al. 2022 plates A and B (phenotype DNA channel, run nucleus masks, per-nucleus tables, the 24 held-out fields per plate).
- **Map bytes (md5, float32 185 × 185):** plate A `f0dbfb9da3e6ab3c1e38a63485491146`; plate B `088f3fa027f58bfb3e44fac7ed05e997`; the earlier plate A map `6aebe23775a9307f28d2802fd2c6ae79` must stay untouched. Regression arms: `nuc_s1` 8936c3b4dec88ebfa8c2312dac23deb3, `nuc_s3` c6a41754717b1f954f03c51f1dabbd10, `nuc_ll1` 2433c49fd10c2d79a55010c65cdf923d, `nuc_ll2` db59aafeb690731f946344306f080dae, `nuc_lq2_centres` b64d551ef7763b0b36a3f221755272f0, `pix_s2` 4bc056b7b0c1d8128f625669a0abcb71, `pix_s6` 2804586cb10c99a6a27a98dce4ea98d7.
- **Plate A held-out rings** (exact re-measure, median / 0–500 px), identical to 4 decimals:

  | Ring (px) | Earlier map | Rebuilt map (95 % CI) |
  |---|---|---|
  | 500–1,000 | 0.9518 | 0.9960 (0.9760–1.0132) |
  | 1,000–1,400 | 0.9585 | 1.0033 (0.9848–1.0202) |
  | 1,400–1,600 | 1.0271 | 1.0111 (0.9952–1.0265) |
  | 1,600–1,800 | 0.9320 | 1.0109 (0.9912–1.0348) |
  | 1,800–1,900 | 0.6856 | 1.0247 (0.9958–1.0634) |
  | 1,900–2,000 | 0.5093 | 1.0010 (0.9650–1.0285) |

  Largest ring deviation 0.0247 (analytic 0.028).
- **Plate B held-out:** largest ring deviation 0.0229; corner rings 1,800–1,900 / 1,900–2,000 = 1.021 / 1.009.
- **Regression cases** (exact, 1,800–1,900 / 1,900–2,000 px): old 0.6856 / 0.5093; pix_s6 0.8060 / 0.6417; pix_s2 0.8789 / 0.7565; nuc_s3 0.9364 / 0.8073; nuc_s1 0.9667 / 0.9180; nuc_ll1 0.9665 / 0.9237; nuc_ll2 0.9639 / 0.9208; nuc_lq2_centres 0.9674 / 0.9279.
- **Downstream:** rebuilt per-cell tables keep the same rows; every non-DNA feature and every DNA feature the correction does not touch is bit-identical; nuclear and cell mean DNA change by the new/old gain ratio within 1e-7.
- **New default (block-centre fit + block-centre lookup):** not bit-identical to the reference maps by construction; acceptance = every held-out ring ≥ 100 cells within 0.03 of 1.0 on both plates and within 0.01 of the reference ring value.
- **Tests (CPU):** synthetic fields with a known steep vignette and nuclei of known brightness on a background → the nuclear-median map recovers the corners within 1 % while the pixel-median map does not; a map read with the zoom offset fails the corner ring and the block-centre lookup passes; a changed map refuses resume via the md5 record.

## Controls
- **As measured** (no correction) and **earlier map** rows in every radial table.
- **Held-out fields** never enter the build (asserted disjoint).
- **Exact vs analytic** correction on the same nuclei (agree within 0.01 per ring).
- **Earlier-map gains** recomputed by the new code equal the run's stored gains.

## Literature
- Singh et al. 2014, J Microsc 256:231 (illumination correction for high-throughput microscopy).
- Peng et al. 2017, Nat Commun 8:14836 (BaSiC).
- Smith et al. 2015, Nat Methods 12:404 (CIDRE).
- Cleveland & Devlin 1988, J Am Stat Assoc 83:596 (local polynomial regression).
- Funk et al. 2022, Cell 185:4634 (the screen).

## Effort
- **Effort:** M (estimator, fit, lookup helper, validation command; the per-nucleus table already exists).
