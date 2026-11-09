# Item spec: cross-field ownership of OPS phenotype cells

Lifecycle: future (alpha). Suggested number: the next free item number
(nightly ends at 868; the coordinator reports 851-869 taken, so check
`features/feature_list/` and the claims files before numbering; 870 if 869
is taken). Related: 752 (OPS phenotype path), 813 (OPS v2), 852 (core
ownership by the SBS centroid), the overlap study
`/nas_mnt/data/sync/papers/BMCspacr/reanalysis/ops_overlap/REPORT.md`.

**Reference implementation:**
`/mnt/wd4tb/bmc_reanalysis/funk_phenotype/ownership_A1/explore_cross_field_ownership.py`
(exploration script, not spaCR code). Its A1 outputs are in the same folder.

## Problem

Funk et al. 2022 phenotype fields are 2,960 px wide on a pitch of about
2,530 px, so neighbouring fields overlap by about 430 px. spaCR
keeps edge cells (`remove_border_objects` off) and flags
`touches_field_edge`. A cell in an overlap is segmented in two or more
fields, often whole in one field and cut in the other. Item 852 picks one
copy per SBS nucleus (deepest field), but it only acts on linked cells and
on the SBS centroid; it does not identify copies by their masks, and it
does not exclude unlinked or cut copies from crop export.

## Settings (all alpha; register under `ALPHA_FEATURES[752]` in `spacr/settings.py`)

| Name | Type | Default | Meaning |
|---|---|---|---|
| `ops_cross_field_ownership` | bool | `False` | Run the ownership pass and drop non-owner copies from aggregation and crop export. Off: outputs identical to today. |
| `ops_cross_field_metric` | str, one of `'overlap_fraction'`, `'iou'` | `'overlap_fraction'` | Score of two copies: intersection / smaller mask area, or intersection over union. |
| `ops_cross_field_min_overlap` | float in (0, 1] | `0.5` | Smallest score that calls two masks the same cell. |
| `ops_cross_field_policy` | str, one of `'one_per_cell'`, `'owner_only'`, `'flag_only'` | `'one_per_cell'` | How the columns are used downstream; see step 9. `'flag_only'` writes the columns and changes nothing else. |

Add each to `spacr/ops_settings.py` (defaults dict, types dict, the OPS
setting list), the settings catalogue and `docs/setting_consumers.json`, as
`ops_ownership` is wired today. Settings-file runs only, through
`spacr.ops_engine.run_ops` (no script).

## Phase

New phase `ph_ownership` in `_PH_PHASES`, after `ph_masks` and before
`merge`:

```
_PH_PHASES = ("ph_flatfield", "ph_segment", "ph_measure", "ph_masks",
              "ph_ownership", "merge", "aggregate")
```

It runs only when named in `phases` (as the other `ph_*` phases) and only
when `ops_cross_field_ownership` is True (otherwise it reports
`{"skipped": "ops_cross_field_ownership is off"}`). It reads
`<dst_root>/<well>/phenotype_masks/site-<n>.npz` (`cells` label image),
`ops_maskpass_<well>.parquet` (`cell_px`, `touches_field_edge`) and the
well's `ops_phenotype` rows through `_ph_transforms`. It must not write to
`ops_phenotype` or any existing table.

## Algorithm (must match the reference script exactly)

1. **Transform per field.** From `ops_phenotype`: when `refined == 1` use
   `refined_scale`, `refined_rotation_deg`, `(refined_offset_y,
   refined_offset_x)`; else `scale`, `rotation_deg`, `(offset_y, offset_x)`.
   `T(p) = s * R(theta) @ p + t` for field `(row, col)` pixels, i.e.
   `spacr.ops_merge.field_to_well` with `R = [[cos, -sin], [sin, cos]]`.
   Inverse: `p = R^T (w - t) / s`. Field size from `field_height/width`.
   Use only sites that have both a label image and a mask-pass row set.
2. **Candidate pairs.** Footprint = min/max of the four mapped corners
   (`spacr.ops_merge.field_footprint`); a pair (a, b), a < b by site,
   is a candidate when the boxes overlap with positive area
   (`spacr.ops_merge.overlapping_fields`).
3. **Intersections.** Map b's four corners into a's pixel frame, take their
   bounding box (floor - 1 / ceil + 1, clipped to a). For every pixel of a in
   that box with label > 0, map it to well, then to b's frame, round to the
   nearest pixel (`np.rint`), drop pixels outside b, read b's label. The
   joint histogram of `(label_a, label_b)` over pixels where both are > 0 is
   the intersection, in field-a pixels.
4. **Score.** `area` = each copy's full pixel count in its own field
   (`cell_px`). `overlap_fraction = inter / min(area_a, area_b)`;
   `iou = inter / (area_a + area_b - inter)`.
5. **Edges.** Within each field pair keep `(label_a, label_b)` only when
   each is the other's best partner (largest `inter`; ties to the smaller
   label) and `score >= ops_cross_field_min_overlap`.
6. **Copies.** The connected components of the edge graph over all
   `(site, label)` nodes (scipy `connected_components`, undirected) are the
   physical cells; each cell without an edge is its own component.
7. **Owner.** The copy with the largest `area`; ties go to the smallest
   `site`, then the smallest `label`.
8. **Columns.** Written per segmented cell to
   `<dst_root>/ops_cross_field_<well>.parquet` (new file, `write_table`):

| Column | Type | Meaning |
|---|---|---|
| `plate`, `well`, `site`, `label` | | key, as in `ops_maskpass_<well>` |
| `area` | float | `cell_px` |
| `touches_field_edge` | bool | from the mask pass |
| `component` | int | physical-cell id within the well (scipy component index) |
| `n_copies` | int | copies in the component (1 = no other copy) |
| `owner_field` | int | site of the owner copy |
| `owner_label` | int | label of the owner copy |
| `is_owner` | bool | this copy is the owner (True for single-copy cells) |
| `owned_area_share` | float | `area / owner's area` (1.0 for owners) |

   Also write the edges (`site_a, site_b, label_a, label_b, inter, score`)
   to `ops_cross_field_edges_<well>.parquet` and a report entry
   (`fields`, `field_pairs`, `edges`, `cells`, `physical_cells`,
   `cells_with_copies`, `non_owner_copies`, `components_with_two_copies_in_one_field`,
   `seconds`) in `ops_report.json`.
9. **Consumers.** With the setting on:
   - **Crop export** skips every non-owner copy (all policies but
     `'flag_only'`): one crop per physical cell, from the largest copy.
   - **Aggregate, `'one_per_cell'` (default).** After the existing merge
     and genotype filtering, keep one genotyped row per physical cell
     (`component`): the one with the largest `area`, ties to the smallest
     `site`, then `label`. Rows of cells not in the ownership table keep
     themselves. Report `dropped_extra_copies`.
   - **Aggregate, `'owner_only'`.** Drop every row with `is_owner == False`.
     Report `dropped_non_owner_copies`. WARNING from the A1 run below: on
     the current links this throws away many genotyped cells,
     because the merge's closer-match dedup often linked the non-owner
     copy and the owner was never linked. Only use it together with
     restricting the `merge` phase's candidates to owners (so the owner is
     the copy that gets linked); that variant is not in the reference
     script and has no acceptance numbers yet.
   With the setting off nothing reads the file and every output is
   byte-identical to today's.

Memory: hold label images in an LRU cache (the script keeps 160 fields,
about 5.6 GB of uint32 2,960 x 2,960 arrays); stay under the run's cap.
Parallel workers over field pairs are allowed if results are concatenated in
pair order (the components and owners do not depend on order).

## Tests (CPU only; `CUDA_VISIBLE_DEVICES=''`, pytest `-n 2` at most)

Synthetic fields with identity rotation and scale 1 (the reference script's
`--selftest` builds exactly these; port them):

1. **Whole vs cut.** Fields at x offsets 0 and 70 (100 x 100 px); a radius
   12 disk at well (50, 92) is cut by field 0's edge and whole in field 1:
   `n_copies == 2` for both, owner is the field-1 copy, the cut copy has
   `0 < owned_area_share < 1`.
2. **No-overlap cell.** A disk only in field 0: `n_copies == 1`,
   `is_owner`, `owned_area_share == 1.0`.
3. **Three-way overlap.** Fields at (0, 0), (0, 60), (60, 60); a radius 10
   disk at well (80, 80), cut at well x 85 in field 0 and at well y 85 in
   field 1, whole in field 2: one component of 3, the field-2 copy owns it.
4. **Tie.** The same whole disk in sites 5 and 9: owner is site 5.
5. **No overlapping fields.** Fields 500 px apart: zero pairs, zero
   edges, every cell its own owner.
6. **Threshold.** A small rectangle lying in a neighbour's disk scores
   IoU about 0.34: no edge at `metric='iou', min_overlap=0.5`.
7. **Setting off.** `run_ops` with `ops_cross_field_ownership=False`
   produces identical merge and aggregate outputs to a run without the
   phase. With it on and `'one_per_cell'`, two genotyped copies of one cell
   become one row (the larger); with `'owner_only'` non-owners are absent;
   with `'flag_only'` the aggregate is unchanged.
8. **Settings-file entry.** A settings dict through `run_ops(...,
   phases=("ph_ownership",))` on a tiny synthetic run folder writes
   `ops_cross_field_<well>.parquet`.

Plus the usual: catalogue + arrival-contract records and all 9 locale
records for every new documented callable in the same commit; ratchets
`test_the_raw_reader_count_only_goes_down`,
`test_nested_functions_are_documented`,
`test_every_callable_in_the_package_is_documented`,
`tests/qt/test_alpha_features_gate.py`, `tests/test_every_ops_setting_is_declared.py`.

## Validation data (read-only)

- Label images: `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_phenotype/work/plateA/A1/phenotype_masks/site-<n>.npz` (`cells`, `nuclei`; 1,271 fields).
  Copy them to a run folder on NVMe as `<dst_root>/A1/phenotype_masks/` before the run.
- Mask-pass tables: `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/analysis_plateA/masks/A1/site-<n>.parquet` (`label`, `cell_px`, `touches_field_edge`).
- Transforms: `ops_phenotype` (well A1) in `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/work/plateA/measurements.db`. Open it read-only or work on a copy: `_ph_transforms` writes back when it migrates (the A1 rows already carry the columns, so no migration should happen, but do not point a run at this file).
- Links: `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/work/plateA/A1/link_A1.parquet`.
- Genotyped cells: `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/analysis_plateA/cells/final/cells_A1.parquet` (`site, label, object_id, sgRNA, gene`).

## Acceptance criterion

On plate A well A1 with defaults (`overlap_fraction`, 0.5), spaCR's
`ops_cross_field_A1.parquet` equals the reference
`ownership_A1_overlap_fraction_0.5.parquet` row for row on
`site, label, n_copies, owner_field, owner_label, is_owner` (exact) and
`owned_area_share` (to 1e-12), after sorting both by `(site, label)`;
`component` ids may differ but must induce the same partition. The edge
table must equal `ownership_edges_A1_overlap_fraction_0.5.parquet` on
`site_a, site_b, label_a, label_b, inter` (exact). The A1 numbers it must
reproduce are in `summary_A1_overlap_fraction_0.5.json` next to the
script (see "A1 numbers" below). With `ops_cross_field_policy='one_per_cell'`
the aggregate input on A1 (starting from
`analysis_plateA/cells/final/cells_A1.parquet`) must hold the row count given in the summary JSON, and
the per-guide and per-gene counts must equal
`cells_per_sgRNA_A1_overlap_fraction_0.5_one_per_cell.csv` and
`cells_per_gene_A1_overlap_fraction_0.5_one_per_cell.csv` (column
`n_after`) exactly.

## Reference numbers

All A1 reference counts (fields, copies, physical cells, links, policy effects) are in
`/mnt/wd4tb/bmc_reanalysis/funk_phenotype/ownership_A1/summary_A1_overlap_fraction_0.5.json`.
Default policy must be `'one_per_cell'`: on A1 a blanket drop of non-owner copies removes many
genotyped cells whose larger twin was never linked; mask ownership is still needed for crop export.
