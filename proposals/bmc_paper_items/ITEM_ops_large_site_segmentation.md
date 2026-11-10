# Item spec: large cropped sites (5,500² px × 5 channels): tiled segmentation with overlap and stitched labels, chunked measurement under a per-worker RAM cap, crop-edge cell policy, and a Phalloidin + DNA cell-segmentation preset

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first.
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs IDENTICALLY: the same labels per site (given the same model, device class and deterministic flags), the same per-cell tables. Every seed is fixed. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes (PERISCOPE track P1).
- Related: `ITEM_cpg_cropped_reader.md` (input), `ITEM_ops_cross_field_ownership.md` (mask-based ownership; reused for the `'own'` edge policy), `ITEM_ops_reference_mask_matching.md` (computes the mask F1 this item reports), `ITEM_ops_phenotype_run_v2.md` (measurement tables).

## Why
PERISCOPE's analysis inputs are 100 crops per well, each 5,500 × 5,500 px (checked on CP186A Well1) × 5 painting channels, uint16: about 300 MB per site raw, over 1 GB once converted to float32 with Cellpose flows and probability maps, and more during measurement of 3,973-feature-class profiles. spaCR's OPS phenotype path was sized for Funk fields (2,960² px). Without tiling and chunking a worker can exceed the machine's RAM rules, and crops abut without overlap, so cells are cut at every crop edge.

PERISCOPE segments nuclei with CellProfiler `IdentifyPrimaryObjects` on DNA (diameter 10–80 px) and cells by watershed on Phalloidin (`9_Analysis.cppipe`); spaCR has no preset that segments cells from an actin stain, and its defaults assume a cytoplasm/cell-body marker.

## What to build
1. **Tiled segmentation** (`spacr.ops_phenotype` or the nightly Make Masks path): tiles of `seg_tile_px` (2,048) with `seg_tile_overlap_px` (≥ 2 × the largest expected cell diameter, default 256); Cellpose run per tile; labels stitched by IoU of objects in the overlap (keep the copy whose centroid is farther from its tile edge; relabel consecutively); deterministic tile order. Small sites (≤ one tile) take the existing path unchanged (bit-identical to today).
2. **Chunked measurement:** per-object features computed per tile band with a halo that contains every object whose centroid lies in the band; each object measured once, on the full object (no cut objects).
3. **RAM cap:** `seg_max_ram_gb` per worker (24 by default for this preset); the planner picks tile size and worker count from it and refuses to start when the cap cannot be met; peak RSS per site written to the run log (spaCR `resource_log`).
4. **Crop-edge policy** `seg_edge_policy`: `'exclude'` (drop objects touching the crop border, the default for P1), `'keep'`, or `'own'` (mask-based ownership across neighbouring crops when their placement is known, `ITEM_ops_cross_field_ownership.md`). The count of excluded objects per site is reported.
5. **Phalloidin preset** `seg_preset='cell_painting_phalloidin'`:
   - nuclei: Cellpose nuclei model on DNA, diameter estimated per plate within `[10, 80]` px (CellProfiler's range), or fixed `nucleus_diameter`;
   - cells: Cellpose cyto model (spaCR default `cpsam`; `cyto3` as an arm) on `[Phalloidin, DNA]`, seeded/matched to nuclei (one nucleus per cell, cells without a nucleus dropped, as the OPS path does);
   - channel indices come from the reader's pinned order (`ITEM_cpg_cropped_reader.md`);
   - SBS linking at the reader's grid (P1 needs no 0.5× rescale because painting and SBS crops share one grid; P2 uses the 20X→10X factor 0.5).
6. **Agreement report:** per site, objects segmented (spaCR vs their `Cells_Objects.tiff` / `Nuclei_Objects.tiff`, non-edge objects only), and the mask F1 at IoU 0.5 and median IoU from `ITEM_ops_reference_mask_matching.md`. Reported, not assumed.

## Settings (alpha)
| Key | Default | Meaning |
|---|---|---|
| `seg_tile_px` | 2048 | tile side (0 = no tiling, today) |
| `seg_tile_overlap_px` | 256 | overlap between tiles |
| `seg_stitch_iou` | 0.5 | overlap IoU to merge copies |
| `seg_max_ram_gb` | 24 | per-worker cap |
| `seg_edge_policy` | `'exclude'` | `'keep'`, `'own'` |
| `seg_preset` | None | `'cell_painting_phalloidin'` |
| `nucleus_diameter` | None (estimate in [10, 80]) | px |
| `cell_model` | `'cpsam'` | or `'cyto3'` |
| `cell_channels` | `['CorrPhalloidin', 'CorrDNA']` | preset channels |
| `seg_report_reference` | None | folder of reference label masks |

## Validation data and acceptance
- **Data:** CP186A Well1 (100 cropped sites; CellProfiler masks `segmentation_masks/Plate_CP186A_Well_Well1_Site_<S>_{Nuclei,Cells,Cytoplasm}_Objects.tiff` for 85 sites, uint16 label images of the same 5,500² grid).
- **Acceptance:**
  - Well1 runs end to end within `seg_max_ram_gb` per worker (peak RSS logged per site);
  - tiled vs untiled segmentation of one site small enough to fit untiled: identical labels after stitching for objects away from tile seams, and ≥ 99 % object IoU ≥ 0.9 at seams;
  - per-site non-edge cell counts within ± 10 % of CellProfiler's; sites outside are listed with the reason, not hidden;
  - mask F1 at IoU 0.5 and median IoU against their cells and nuclei are reported per site and per well (agreement, not ground truth);
  - a rerun from the saved settings gives identical labels and tables (CPU: bit-identical; GPU: identical given the recorded device class and deterministic flags; otherwise reported as a difference count).
- **Tests (CPU):** synthetic 6,000² image with known discs crossing tile seams → every disc once, correct area; edge policy counts; RAM planner refuses an impossible cap.

## Controls
- **CellProfiler masks** (their pipeline, same pixels) as the reference row.
- **Untiled run** on a cropped 2,048² region as the tiling control.
- **cyto3 vs cpsam** arm for the preset.

## GPU and compute
Cellpose on GPU; expected 15–35 min per well (scaled from Funk; check on Well1). CPU tests only for CI.

## Catalogue and locales
New documented callables (tiled segmenter, stitcher, chunked measurer, RAM planner, preset) ship with arrival contracts and 9-locale records in the same commit; the preset appears in the GUI preset list with a tooltip.

## Literature
- Stringer C et al. (2021) Cellpose. Nat Methods 18:100–106; Pachitariu M et al. (2025) Cellpose-SAM, bioRxiv.
- Bray M-A et al. (2016) Cell Painting. Nat Protoc 11:1757–1774.
- Ramezani M et al. (2025) Nat Methods 22:621 (PERISCOPE segmentation).

## Effort
- **Effort:** M (tiling + stitching + chunked measurement are the work; the preset is settings).
