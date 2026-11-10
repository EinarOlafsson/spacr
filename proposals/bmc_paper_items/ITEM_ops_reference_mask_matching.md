# Item spec: cross-pipeline cell matching by mask overlap: link spaCR cells to a reference pipeline's label masks, carry over its per-cell gene calls, mask agreement (F1 at IoU 0.5, median IoU) and per-cell call concordance

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first.
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs IDENTICALLY (same matches, same tables). Ties are broken deterministically (below). Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes (PERISCOPE and Feldman comparison tables: mask agreement, per-cell concordance).
- Related: **extends** `ITEM_ops_genotype_three_way_compare.md` (`ops_engine` phase `compare_calls`, which matches external call tables to spaCR nuclei by centroid distance). This item adds a mask-overlap matching mode and reference-ID mapping to the same phase; it does not add a second compare module. Also `ITEM_ops_large_site_segmentation.md` (reports the F1 computed here), `ITEM_ops_spacer_prefix_decode.md` (spaCR calls compared here).

## Why
PERISCOPE deposits label masks (`Plate_<plate>_Well_<well>_Site_<S>_{Nuclei,Cells,Cytoplasm}_Objects.tiff`, label = CellProfiler ObjectNumber) and per-cell barcode calls (`single_cell_metadata`: `Metadata_Foci_site` like `CP186A-Well3-75`, `Metadata_Cells_ObjectNumber`, `Metadata_Foci_Barcode_MatchedTo_Barcode`, `_GeneCode`, `Metadata_Foci_Cell_Quality`) on the same pixels spaCR analyses in track P1. Feldman 2019 deposits `cells.tif` / `nuclei.tif` per tile and per-read calls (`reads.csv`). Comparing per cell needs a cell-to-cell link by mask overlap; centroid matching is not enough when segmentations differ in shape and cells touch.

## What to build
In `compare_calls`, `ops_compare_match='mask_overlap'`:
1. **Reference masks:** `ops_compare_reference_masks` = a path template with `{plate}`, `{well}`, `{site}`/`{tile}` and `{object}` (`Nuclei`, `Cells`), read as label images on the analysed grid (shape must equal spaCR's; refuse otherwise unless a scale factor is set).
2. **Matching per site:** contingency of spaCR labels × reference labels (sparse, one pass over pixels); for each spaCR object, the reference object with the largest overlap (ties: higher IoU, then lower reference label); IoU recorded. One-to-one matching for F1 by Hungarian assignment on IoU ≥ `ops_compare_iou` (0.5), as standard instance-segmentation F1.
3. **Reference IDs → calls:** `ops_compare_reference_calls` = table + column map (`site` key parser, object-number column, barcode, gene, quality). PERISCOPE map: site from `Metadata_Foci_site` (`<plate>-<well>-<site>`), object = `Metadata_Cells_ObjectNumber`, barcode = `Metadata_Foci_Barcode_MatchedTo_Barcode`, gene = `Metadata_Foci_Barcode_MatchedTo_GeneCode`, quality = `Metadata_Foci_Cell_Quality`. Feldman map: per-cell call made from `reads.csv` (`cell` column; top barcode by read count, ties → no call; verify the column names on the download) — the rule is pinned in the settings.
4. **Outputs:** `match_cells.parquet` (spaCR id, reference id, overlap px, IoU, matched flag), `mask_agreement.csv` per site and well (n spaCR, n reference, TP/FP/FN at IoU 0.5, F1, median IoU of matches; edge-excluded objects listed separately), `call_concordance.csv` (both called / same gene / same guide / spaCR only / reference only / disagree, per quality class), with the shuffled control.

## Settings (alpha)
| Key | Default | Meaning |
|---|---|---|
| `ops_compare_match` | `'centroid'` | `'mask_overlap'` adds this item |
| `ops_compare_reference_masks` | None | path template |
| `ops_compare_reference_objects` | `['Cells', 'Nuclei']` | objects compared |
| `ops_compare_iou` | 0.5 | F1 threshold |
| `ops_compare_reference_calls` | None | table path |
| `ops_compare_call_columns` | PERISCOPE map above | dict |
| `ops_compare_edge_policy` | `'exclude'` | drop objects touching the crop edge in both |
| `ops_compare_control_seed` | 0 | shuffled-call control |

## Validation data and acceptance
- **Data:** CP186A Well1 CellProfiler masks (85 sites × 3 objects) and the deposited `A549_single_cell_metadata_CP186A_ALLWELLS.csv.gz` (Well1 196,153 retained cells); spaCR Well1 masks and calls from the P1 run. Feldman: experiment C `cells.tif`, `nuclei.tif`, `reads.csv` for wells A1, B1.
- **Acceptance:**
  - every reference cell ID in the Well1 single-cell metadata resolves to a label present in its site's `Cells_Objects.tiff` (report the count that do not);
  - self-match control: reference masks matched to themselves give F1 = 1, IoU = 1 and 100 % concordance;
  - spaCR vs reference F1 at IoU 0.5, median IoU and per-cell gene concordance reported for Well1 (feeds the PERISCOPE comparison table); rerun from the saved settings gives identical tables.
- **Tests (CPU):** synthetic label pairs with known IoUs (split, merge, shift) → expected F1 and matches; tie-break order; a site key parser for `CP186A-Well3-75`.

## Controls
- **Self-match** (above).
- **Shuffled reference calls** within a site (seed 0): the concordance expected by chance.
- **Centroid matching** on the same cells, to show where the two modes differ.

## Catalogue and locales
New documented callables (mask contingency, overlap matcher, call mapper) ship with arrival contracts and 9-locale records in the same commit; each setting has a tooltip.

## Literature
- Caicedo JC et al. (2019) Evaluation of deep learning strategies for nucleus segmentation in fluorescence images. Cytometry A 95:952–965 (F1 at IoU thresholds).
- Ramezani M et al. (2025) Nat Methods 22:621; Feldman D et al. (2019) Cell 179:787.

## Effort
- **Effort:** S–M (extends the `compare_calls` phase; the contingency matcher is the new part).
