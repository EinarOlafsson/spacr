# Item spec: Cell Painting Gallery cropped-site reader (PERISCOPE `images_corrected_cropped` layout → one multichannel phenotype field and one SBS cycle stack per site)

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first (871–908 paper lanes, 912 flat-field map).
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs IDENTICALLY: the same arrays (bytes) per site and channel, the same index table. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes (second OPS benchmark, PERISCOPE A549 plate CP186A, track P1 = their illumination-corrected, aligned, cropped pixels).
- Related: `ITEM_ops_large_site_segmentation.md` (sites are 5,500² px), `ITEM_ops_spacer_prefix_decode.md` (reads the SBS stack this item builds), `ITEM_ops_reference_mask_matching.md` (reads the CellProfiler masks next to these sites).

## Why
PERISCOPE (Ramezani et al. 2025, cpg0021) publishes its analysis inputs as one TIFF per channel and site: `images_corrected_cropped/<plate>_Well<N>/<Channel>/<Channel>_Site_<S>.tiff`. Painting channels are `CorrDNA, CorrER, CorrMito, CorrPhalloidin, CorrWGA`; SBS channels are `Cycle<CC>_{A,C,G,T}` for cycles 01–12 plus `Cycle01_DAPI`. spaCR's OPS engine indexes tiles by `_TILE_PATTERN` / `_UNCYCLED_TILE_PATTERN` (`ops_engine._match_tile`: magnification, well `A1`-style, `Site-N`), and Import's regex inference expects plate/well/field in the file name. Neither reads channel-per-folder layouts with base-named channels, `WellN` wells or plate names in the parent folder. Track P1 runs spaCR segmentation, decode, features and profiles on the identical pixels CellProfiler analysed, so the reader must not resample or rescale anything it does not have to.

Checked on CP186A Well1: 54 channel folders × 100 sites; `CorrDNA_Site_10.tiff`, `Cycle01_A_Site_10.tiff` and `Cycle01_DAPI_Site_10.tiff` are all 5,500 × 5,500 uint16, LZW-compressed; the painting and SBS crops share one pixel grid (not about 4.9k² as sometimes stated).

## Reference implementation
None in spaCR; the layout is defined by the S3 listing (the cpg0021 S3 key listing) and the PCPIP pipelines (`9_Analysis.cppipe`, A549: LoadData of `Corr*` and `Cycle*` images per site). The reference arrays are the files themselves (`tifffile.imread`).

## What to build
1. **Layout parser** `spacr.io_layouts.cpg_cropped` (or the nightly equivalent of a layout registry): from a root `images_corrected_cropped/` (or one `<plate>_Well<N>` folder) build an index table `plate, well, site, modality ('phenotype'|'sbs'), cycle, channel, base, path, shape, dtype` from paths only (no pixel reads), with plate and well from the folder name (`CP186A_Well1` → plate `CP186A`, well `Well1`, and a normalised well id `W1` for spaCR tables), site from `_Site_<S>`.
2. **Channel maps pinned in the settings file:** `cpg_phenotype_channels` (ordered list) and `cpg_sbs_bases` (ordered list, default `['G', 'T', 'A', 'C']`, the MiSeq order spaCR's decode reads as `ops_base_channels` CY3, A594, CY5, CY7), `cpg_sbs_dapi` (`Cycle01_DAPI`, cycle 1 only). The reader never infers order from folder listing order.
3. **Loader:** per site, a `(Y, X, C)` phenotype array in the pinned order and a `(cycles, Y, X, 4)` SBS array (+ the cycle-1 DAPI plane), exact dtype, no rescaling. If a painting and an SBS crop of the same site differ in shape, refuse unless `cpg_sbs_resample` names a method (`'none'` default, `'nearest'`, `'linear'`) and record the factor; P1 must run with `'none'`.
4. **Completeness check** per well: every site has every pinned channel and every cycle; missing files are listed, never silently skipped (`cpg_missing='error'|'skip_site'`).
5. **Engine hook:** the OPS engine accepts this index in place of its tile index (`ops_layout='cpg_cropped'`), with no stitching and no cycle registration (`ops_register=False` for P1: the crops are already aligned), and writes the same `merged/` per site as `ITEM_ops_phenotype_run_v2.md` (channels stacked, masks added later).
6. **Provenance:** `cpg_index.csv` (one row per file, with size and md5 when `cpg_md5=True`) and the pinned channel maps in the run folder.

## Settings (alpha)
| Key | Default | Meaning |
|---|---|---|
| `ops_layout` | `'tiles'` (today) | `'cpg_cropped'` adds this item |
| `cpg_root` | None | `images_corrected_cropped/` or one `<plate>_Well<N>/` |
| `cpg_wells` | None (all) | e.g. `['Well1']` |
| `cpg_sites` | None (all) | list or range |
| `cpg_phenotype_channels` | `['CorrDNA','CorrPhalloidin','CorrMito','CorrER','CorrWGA']` | stacked order |
| `cpg_sbs_cycles` | 12 | cycles present |
| `cpg_sbs_bases` | `['G','T','A','C']` | base order per cycle |
| `cpg_sbs_dapi` | `'Cycle01_DAPI'` | SBS nuclear channel |
| `cpg_sbs_resample` | `'none'` | refuse on grid mismatch |
| `cpg_missing` | `'error'` | or `'skip_site'` |
| `cpg_md5` | False | md5 per file in the index |

## Validation data and acceptance
- **Data:** PERISCOPE cpg0021 `images_corrected_cropped/CP186A_Well1/` (5,400 files). Wells 2–6 next.
- **Acceptance:**
  - all 100 sites of CP186A_Well1 load; 5 painting channels and 12 × 4 + 1 SBS planes per site;
  - every loaded plane equals `tifffile.imread` of its file byte for byte (md5 of the array);
  - index rows = 5,400 for Well1; channel order in the run folder equals the settings file;
  - a rerun from the saved settings gives identical arrays and index.
- **Tests (CPU):** a synthetic tree with two wells, three sites, shuffled folder creation order → identical stacks; a missing `Cycle07_C` file → error naming it; a shape mismatch → refusal unless resampling is set; a well folder name without `_Well` → clear error.

## Controls
- Pixel identity against the source files (above); no other control applies to a reader.

## Catalogue and locales
New documented callables (`cpg_cropped` index builder, loader) ship with arrival contracts and 9-locale records in the same commit; each setting has a tooltip.

## Literature
- Ramezani M et al. (2025) A genome-wide atlas of human cell morphology. Nat Methods 22:621–633 (PERISCOPE).
- Weisbart E et al. (2024) Cell Painting Gallery: an open resource for image-based profiling. Nat Methods 21:1775–1777.

## Effort
- **Effort:** S (path parser, loader, completeness check, engine hook).
