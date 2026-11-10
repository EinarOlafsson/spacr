# Item spec: multipoint ND2 reader for raw OPS acquisitions (one site per file, 5 channels): parse well / site / sequence and channels from the name or the ND2 metadata, 20X phenotype and 10X SBS cycle folders, stage positions for stitching and registration

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first.
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs IDENTICALLY (same arrays, same index, same stage positions). Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly (spaCR already imports ND2 in `convert.py` / `io` for Import; this item adds the OPS multipoint layout and stage metadata — extend that reader, do not add a second ND2 dependency).
- **Paper-critical:** no (track P2 is optional; P1 is the main benchmark).
- Related: `ITEM_cpg_cropped_reader.md` (P1 reader; same channel-map settings), `ITEM_ops_spacer_prefix_decode.md`, `ITEM_ops_large_site_segmentation.md`, `ITEM_ops_flatfield_v2.md` (P2 needs spaCR's own illumination correction).

## Why
PERISCOPE raw images (`images/<batch>/images/<plate>/20X_CP_<plate>/` and `10X_cNN-SBS-NN/`, including a lower-case `10x_c2` folder) are Nikon ND2 files, one site per file, 5 channels, named like `Well1_Point1_0000_ChannelDAPI,Cy3,A594,Cy5,Cy7_Seq0000.nd2` (phenotype channels `DAPI,GFP,A594,Cy5,750`; about 1,480² px, 22.3 MB). A549 CP186A has 1,364 phenotype sites and 320 SBS sites per cycle per well. Track P2 runs spaCR's illumination correction, stitching, cycle registration and 20X→10X registration from these files; that needs the stage positions stored in the ND2 metadata.

## What to build
1. **Layout** `ops_layout='nd2_multipoint'`: index from names (`Well<N>`, `Point<site>` / sequence number `Seq<NNNN>`, channel list after `Channel`) with the cycle and magnification from the parent folder (`20X_CP_<plate>`, `10X_c<NN>-SBS-<NN>`, case-insensitive); channel names cross-checked against the ND2 metadata (mismatch → error naming both).
2. **Loader:** `(Y, X, C)` per file in the pinned channel order; dtype preserved; lazy (memory-mapped where the ND2 library allows).
3. **Stage positions:** per site x, y (µm) and pixel size from the ND2 metadata, written to `nd2_positions.csv`; used as the initial layout for `ops_stitch` / `ops_layout` and for 20X↔10X registration (scale from the pixel sizes, not assumed).
4. **Completeness:** expected sites per well (`nd2_expected_sites`, optional) checked; missing files listed.

## Settings (alpha)
| Key | Default | Meaning |
|---|---|---|
| `ops_layout` | `'tiles'` | `'nd2_multipoint'` adds this item |
| `nd2_root` | None | plate folder |
| `nd2_phenotype_folder` | `'20X_CP_{plate}'` | template |
| `nd2_sbs_folder` | `'10X_c{cycle:02d}-SBS-{cycle:02d}'` | template, case-insensitive |
| `nd2_phenotype_channels` | `['DAPI','GFP','A594','Cy5','750']` | pinned order |
| `nd2_sbs_channels` | `['DAPI','Cy3','A594','Cy5','Cy7']` | DAPI + G, T, A, C |
| `nd2_expected_sites` | None | e.g. `{'phenotype': 1364, 'sbs': 320}` |

## Validation data and acceptance
- **Data:** raw CP186A Well1 (116 GB: 30 GB painting + 86 GB SBS; ).
- **Acceptance:** reads CP186A Well1 phenotype and all 12 cycles; site counts 1,364 phenotype and 320 per cycle; arrays equal the ND2 library's own read byte for byte; stage positions exported; rerun gives identical index and positions.
- **Tests (CPU):** name parser over the S3 listing (the cpg0021 S3 key listing, no pixels needed) → every raw ND2 key indexed, including `10x_c2`; a small synthetic ND2 (or a fixture from the ND2 library's test data) for the loader and metadata.

## Controls
- Byte identity against the ND2 library; stage positions against the CPG `images_corrected_stitched_10X` previews (sites land where the preview shows them).

## Catalogue and locales
New documented callables ship with arrival contracts and 9-locale records in the same commit; each setting has a tooltip.

## Literature
- Ramezani M et al. (2025) Nat Methods 22:621; Weisbart E et al. (2024) Nat Methods 21:1775.

## Effort
- **Effort:** S–M (parser and metadata; the ND2 dependency already exists).
