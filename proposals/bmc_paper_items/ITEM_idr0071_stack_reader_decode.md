# Item spec: IDR idr0071 (Feldman 2019) tile-stack input and its 9-of-10-cycle CROPseq decode: read `10X_c<N>-SBS-<N>_<well>_Tile-<t>.sbs.tif` (1024² × 5) and `10X_c0-DAPI-p65ab_<well>_Tile-<t>.phenotype.tif`, translation-only phenotype↔SBS registration, spacer-prefix decode with cycle 6 masked

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first.
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs IDENTICALLY (same arrays, same reads and calls). Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes (Feldman experiment C support benchmark).
- Required first: `ITEM_ops_spacer_prefix_decode.md` (library mode `spacer_prefix`, `ops_cycle_mask`). Related: `ITEM_translocation_gene_score.md` (phenotype readout), `ITEM_ops_reference_mask_matching.md` (comparison with their masks and reads).

## Why
idr0071 (CC BY 4.0) stores raw SBS as one 5-channel TIFF per cycle and tile, `input/10X_c<N>-SBS-<N>/10X_c<N>-SBS-<N>_<well>_Tile-<t>.sbs.tif` (1024 × 1024 × 5: DAPI + 4 bases), and the phenotype as `input/10X_c0-DAPI-p65ab/10X_c0-DAPI-p65ab_<well>_Tile-<t>.phenotype.tif` (DAPI, p65). Experiment C (A549) has cycles c1–c5 and c7–c10 (nine cycles; **c6 is absent**, checked in the IDR FTP listing), 6 wells × 533 tiles, plus `input/well_tile_list.csv`. spaCR's `_TILE_PATTERN` expects `10X_c<N>_<well>_<channel>_Site-<n>` with one channel per file; it cannot read multi-channel per-cycle stacks with `-SBS-N` acquisition names and `Tile-` indices. CROPseq barcodes are the first 12 nt of the spacer (library `CROPseq_NFkB_library.csv` (ops repository): `barcode, sgRNA, gene_symbol`, 5,738 rows, 950 genes + NT); with c6 missing and 10 cycles read, the usable read is positions 1–5 and 7–10. Phenotype and SBS are both 10X on one tile grid, so registration is a translation.

## Reference implementation
- Feldman `ops` package (`ops.firesnake` Snakemake pipeline, `Snakefile_experimentC` on IDR, a few kB — download it with the data): channel order of the SBS stack, base order, crosstalk compensation, `THRESHOLD_READS`, read → cell assignment; their outputs per tile `process/<well>_Tile-<t>.{bases,reads}.csv`, `cells.tif`, `nuclei.tif`, `phenotype_aligned.tif`. Pin the base order and thresholds from that Snakefile (expected DAPI, G, T, A, C; verify).

## What to build
1. **Layout** `ops_layout='idr_stack'`: index from names only (`well`, `tile`, `cycle`, `modality`), cycle from `c<N>`; channels inside each stack named by `idr_sbs_channels` (pinned, e.g. `['DAPI', 'G', 'T', 'A', 'C']`) and `idr_phenotype_channels` (`['DAPI', 'p65']`); `well_tile_list.csv` used to check completeness; tiles not stitched (`ops_stitch=False`).
2. **Missing cycles:** cycles in `ops_sbs_cycles` (1–10) without files become masked (`ops_cycle_mask`), never an error, and are listed in the run report; any other missing file is an error.
3. **Registration:** per tile, cycle-to-cycle and phenotype-to-SBS translation on DAPI (phase correlation; existing `ops_register` primitives), no scale; shifts written per tile.
4. **Decode:** `ITEM_ops_spacer_prefix_decode.md` with `ops_library` = the CROPseq library (`ops_library_sequence_col='sgRNA'`), `ops_prefix_length=10`, `ops_cycle_mask=[6]` (9 informative bases), collisions reported (prefixes that become identical when position 6 is masked are ambiguous).
5. **Experiment A** (lentiGuide-BC barcodes needing a barcode → sgRNA map) is out of scope; the reader must still parse its names.

## Settings (alpha)
| Key | Default | Meaning |
|---|---|---|
| `ops_layout` | `'tiles'` | `'idr_stack'` adds this item |
| `idr_root` | None | experiment folder (`experimentC/`) |
| `idr_wells` | None (all) | e.g. `['A1', 'B1']` |
| `idr_sbs_channels` | `['DAPI','G','T','A','C']` (verify) | order inside each SBS stack |
| `idr_phenotype_channels` | `['DAPI','p65']` | |
| `ops_sbs_cycles` | 10 | cycles of the protocol |
| `ops_cycle_mask` | auto from missing files | here [6] |
| `ops_library_mode` | `'spacer_prefix'` | from the decode item |
| `ops_prefix_length` | 10 | |

## Validation data and acceptance
- **Data:** experiment C wells A1, B1 (533 tiles × 9 SBS cycles + phenotype each; their `reads.csv`, `cells.tif`, `nuclei.tif`); IDR crawl and manifest the IDR FTP listing, `Cell_IDR_files.csv.gz`; library `CROPseq_NFkB_library.csv` (ops repository).
- **Acceptance:**
  - loads A1 and B1: 533 tiles × 9 cycles each, cycle 6 reported as masked; arrays equal `tifffile.imread` byte for byte;
  - mapping rate: % of cells with ≥ 1 exact read on the same tiles, spaCR vs theirs from `reads.csv` on their `cells.tif` (paper: > 77 % of transduced cells), reported with the random-barcode control;
  - per-cell guide concordance with their calls via `ITEM_ops_reference_mask_matching.md`;
  - rerun from the saved settings gives identical reads and calls.
- **Tests (CPU):** synthetic tree with c6 missing → mask [6]; a stack with the wrong channel count → error; name parser on all six experiments' patterns from the crawl (no pixels needed).

## Controls
- **Random-barcode decode** and **out-of-library** rate (from the decode item).
- **Their deposited reads** as the reference row.

## Catalogue and locales
New documented callables (the `idr_stack` index builder and loader) ship with arrival contracts and 9-locale records in the same commit; each setting has a tooltip.

## Literature
- Feldman D et al. (2019) Cell 179:787–799; Williams E et al. (2017) The Image Data Resource. Nat Methods 14:775–781.

## Effort
- **Effort:** S (name parser, stack loader, masked-cycle wiring; decode comes from the prefix-decode item).
