# Item 983: OME-Zarr 0.5 images (Zarr v3, sharded) for merged, stitched and mask images, with readers that accept both formats

- **Number:** 983. Approved by the maintainer on 2026-11-18, around 10:45.
- **Priority:** medium. **Alpha.**
- **Order:** code starts after 974 lands.
- **Scope:** 983 only.
  - The crop store and GUI readers are 984 (in 976).
  - The converter for old projects is 985.
  - Switching the default is 986, which needs the maintainer.
- **Gap:** spaCR writes its merged field stacks (`merged/*.npy`), its stitched OPS well mosaics and its masks (`*.npy`, `*.npz`, PNG) as one file per field and object class.
  - A plate gives tens of thousands of small files, carrying no channel names, no pixel size and no plate layout.
  - Other tools cannot open them: napari, Fiji, OMERO and the BioImage Archive all read OME-Zarr.

## Design card
- **Question:**
  - Can spaCR write its images as OME-Zarr 0.5 (Zarr v3, sharded), so that every spaCR reader and outside viewers open them?
  - Are measurements, masks and profiles exactly what they are with npy?
  - What does it cost in file count, disk and time?
- **Data and why:**
  - Funk et al. 2022 plate A well A1 test tier (`ops_testdata_funk2022_A1_core`): the OPS merged, stitched and phenotype paths on a fixture that runs in minutes.
  - One Mask Generation / Measure example plate (`spacr-download` example data): the npy/PNG path of the main app.
- **Controls:**
  - The setting off (`npy`) against nightly: byte-identical.
  - npy against OME-Zarr on the same run: the same array values.
  - A settings-file rerun on the OME-Zarr output against the npy run: the same tables.
- **Statistics:** exact equality, no tolerance. Speed and disk are reported as median and range over 3 repeats.
- **Pre-set criterion:**
  - (a) Off is byte-identical to nightly.
  - (b) On, every pixel, label and dtype equals npy.
  - (c) A rerun reading OME-Zarr gives identical measurement tables, masks and profiles.
  - (d) napari (napari-ome-zarr) and Fiji (OME-Zarr reader) open a written plate with channel names and pixel size.
  - File count and disk are reported, not gated.
- **Compute:** CPU only, about 1 h on the test tier. Heavy steps run under the master flock.

## Spec
1. **Setting:** `image_format`.
   - Shared by Mask Generation, Measure and OPS (`ops_image_format` aliases it in the OPS file).
   - Values: `'npy'` (default; today's files exactly) and `'ome-zarr'` (alpha, ALPHA_FEATURES[983]).
   - No schema bump; the default is unchanged.
2. **Layout: OME-Zarr 0.5 HCS (high-content screening).**
   - One store per plate, `<run>/images.ome.zarr`, laid out plate → row → column (well) → field.
   - Plate and well metadata carry the well names and field order that spaCR's file names hold today.
   - **Merged field stacks:** one multiscale image per field, axes `c, y, x` (`t`/`z` only when present).
   - **Channel names:** `omero.channels[].label` from the run's channel settings (`ops_ph_channels`, the Mask Generation channel names), so readers no longer guess them.
   - **Pixel size:** in `coordinateTransformations` (scale), from the acquisition metadata when known; otherwise 1 and `unit` left empty, and the report says so.
   - **Masks:** written as OME-Zarr `labels`, one label image per object class under the field (`labels/nucleus`, `labels/cell`, `labels/pathogen`, `labels/organelle*`, `labels/cytoplasm`). Integer dtype as today; `image-label` metadata, with colours left out.
   - **Stitched OPS well mosaics:** one image per well and acquisition under the well group (`stitched/<acquisition>`).
3. **Pyramids:** only for stitched wells, and for fields above 4,096 px on a side. Downsampling is mean for intensities and mode/nearest for labels. Ordinary fields get a single level, so no extra disk.
4. **Chunks and shards: aligned to the worker split.**
   - Parallel writers must never share a chunk; Zarr v3 shards are the unit one writer owns.
   - Shard = one field (merged stacks and masks), or one spaCR window/tile of the stitched mosaic (the `ops_window_*` / tile grid).
   - Chunks inside a shard are 512 × 512 × all channels.
   - Writers write whole shards and nothing else. A run with N mask or measure workers maps fields to workers by the same split as today, so no two workers touch one shard.
   - Writes are atomic per shard (temporary name + rename), as `_save_array_atomic` is for npy.
   - Codec: zstd, level 3. Lossless only; no lossy codecs.
5. **Readers: every spaCR image reader accepts both formats.**
   - One shared function `spacr.image_store.read_image(path_or_key, ...)` and its label counterpart. They return the same arrays (same dtype, axes and values) from npy/npz/PNG or from an OME-Zarr field/label.
   - The existing loaders (`io` merged and mask loaders, the OPS phenotype and merged readers, Measure, crops) route through them.
   - Format detection: by store presence (`.zarr` + `zarr.json`), never by a setting, so a project written either way opens with either setting.
6. **Versions:**
   - `zarr >= 3.1, < 4` (Zarr v3 format with sharding) and `ome-zarr` pinned to the version tested here (0.21.x on the build machine).
   - Both stay optional, under `spacr[zarr]`. `'ome-zarr'` refuses with an install hint when they are missing.
   - The versions are recorded in the run's settings stamp.
7. **Interoperability:**
   - The test opens the written plate with the ome-zarr-py reader and validates it with the OME-NGFF JSON schema (`ome-zarr` validator).
   - The proof shows napari (napari-ome-zarr) and Fiji (OME-Zarr reader / BigDataViewer) opening it, with screenshots in both themes.
8. **Docs and translations:** labels, tooltips and docstrings in 9 languages; arrival contracts and catalogue records in the same commit.

## Proof check (pre-set)
- **Off identity:** `image_format='npy'`.
  - Funk A1 OPS phenotype path and one Mask Generation/Measure example: every output identical to nightly.
  - Arrays are compared by md5 of their bytes; tables by md5.
- **On equality:** `image_format='ome-zarr'`, same runs.
  - Every merged stack, mask and stitched mosaic read back equals the npy one exactly (values, dtype, shape).
- **Settings-file rerun:** rerun measurement and profiling from the saved settings file on the OME-Zarr output.
  - Measurement tables, masks and gene profiles identical to the npy run.
- **Cost table:**

  | Format | Files | Disk | Write s (median/range) | Read s (median/range) |
  |---|---|---|---|---|
  | npy/PNG (control row, today) | | | | |
  | OME-Zarr | | | | |

  Published comparator: n/a in paper (Funk et al. 2022 and Brieflow publish no file-count, disk or timing figures for their image stores).
- **Viewers:** napari and Fiji open the plate, with channel names and pixel size shown.
- **Report:** `BMCspacr/reanalysis/item983_proof/REPORT.md`, with tables, controls and a glossary.
