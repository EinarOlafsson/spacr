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

## Maintainer decisions (2026-11-18, about 11:30)
These bind 983 and 984.
1. **One writer per file.** Per-field (FOV) workers never share a shard.
   - **Local runs:** a few writer processes each own a set of wells. FOV workers send them compressed crop or label chunks through shared memory. There are no shared files and no merge step.
   - **Cloud or multi-machine runs:** each worker writes its own parts, registers them in the run's DuckDB catalog, and they are readable at once. A compaction step rewrites them into per-well shards afterwards.
   - The same rule holds for masks and labels written into the plate OME-Zarr.
2. **Object relationships** become per-cell columns in the crop index, next to each crop's locator: has a nucleus, infected or uninfected, pathogen count, class. Zarr attributes name those columns. Readers filter by query, not by folder. PNG (Legacy) mode still writes today's folders.
3. **Readers** (the annotation app's Load and Stream, Image-UMAP, Classify and the others) offer PNG (Legacy) and OME-Zarr (Default).
   - On an old project, OME-Zarr falls back to PNG with one log note.
   - Both the old and the new folder layouts are readable.
4. **Formats:**
   - Mask Generation writes OME-Zarr, TIFF or npy.
   - Measure crops write OME-Zarr, PNG (Legacy), TIFF or npy.
   - In OME-Zarr mode, masks go in as labels inside the stack's own OME-Zarr, with no merged copy.
5. **Related:** item 987 removes the Cellpose batch .npz. Its v1 half, 987b, keeps the batch plan servable from Zarr chunks. Read 987's design note when it lands, and keep the chunk layout here compatible with it.
6. **Limits:** keep to OME-Zarr 0.5 with sharding. Defaults switch only at item 986, after asking the maintainer.

## Spec
1. **Setting:** `image_format`.
   - Shared by Mask Generation, Measure and OPS (`ops_image_format` aliases it in the OPS file).
   - Mask Generation values: `'npy'` (default; today's files exactly), `'tiff'` and `'ome-zarr'`.
   - Measure crop values (984): `'png'` (PNG (Legacy), today's default), `'tiff'`, `'npy'` and `'ome-zarr'`.
   - New values are alpha, in ALPHA_FEATURES[983].
   - No schema bump; defaults stay unchanged until 986.
2. **Layout: OME-Zarr 0.5 HCS (high-content screening).**
   - One store per plate, `<run>/images.ome.zarr`, laid out plate → row → column (well) → field.
   - Plate and well metadata carry the well names and field order that spaCR's file names hold today.
   - **Merged field stacks:** one multiscale image per field, axes `c, y, x` (`t`/`z` only when present).
   - **Channel names:** `omero.channels[].label` from the run's channel settings (`ops_ph_channels`, the Mask Generation channel names), so readers no longer guess them.
   - **Pixel size:** in `coordinateTransformations` (scale), from the acquisition metadata when known; otherwise 1 and `unit` left empty, and the report says so.
   - **Masks:** written as OME-Zarr `labels` inside the stack's own OME-Zarr, with no merged copy. There is one label image per object class under the field (`labels/nucleus`, `labels/cell`, `labels/pathogen`, `labels/organelle*`, `labels/cytoplasm`). The integer dtype stays as today; `image-label` metadata is written with colours left out.
   - **Stitched OPS well mosaics:** one image per well and acquisition under the well group (`stitched/<acquisition>`).
3. **Pyramids:** only for stitched wells, and for fields above 4,096 px on a side. Downsampling is mean for intensities and mode/nearest for labels. Ordinary fields get a single level, so no extra disk.
4. **Chunks, shards and the writer model (decision 1).**
   - Zarr v3 shards are the unit one writer owns.
     - A shard is one field (merged stacks and labels), or one spaCR window/tile of the stitched mosaic (the `ops_window_*` / tile grid).
     - Chunks inside a shard are 512 × 512 × all channels.
     - Codec: zstd, level 3. Lossless only; no lossy codecs.
   - **Local runs:**
     - K writer processes each own a disjoint set of wells. K is small (2–4 by default) and set by disk bandwidth.
     - FOV workers compress their chunks and pass them to the owning writer through shared memory: a bounded pool of `multiprocessing.shared_memory` blocks plus a pipe of (well, field, level, chunk) descriptors.
     - Each writer assembles whole shards and writes them atomically (temporary name + rename).
     - No two processes ever open one shard for writing, and nothing is merged afterwards.
   - **Cloud or multi-machine runs:**
     - Each worker writes complete per-worker part stores.
     - Every part is registered in the run's DuckDB catalog (path, well, fields, levels, checksum) and is readable immediately.
     - A `compact` step rewrites a well's parts into its per-well shards and swaps the catalog rows atomically.
   - **Crashes:** a writer crash leaves only whole shards or whole parts. Resume rewrites the missing ones.
5. **Readers: every spaCR image reader accepts every format.** This section is the 983 core; the GUI readers are 984.
   - One shared function, `spacr.image_store.read_image(path_or_key, ...)`, and its label counterpart. They return the same arrays (same dtype, axes and values) from npy/npz/TIFF/PNG or from an OME-Zarr field/label.
   - The existing loaders route through them: the `io` merged and mask loaders, the OPS phenotype and merged readers, Measure and crops.
   - **Format detection:** by what is on disk (`.zarr` + `zarr.json`), never by a setting, so a project written either way opens with either setting.
   - **Fallback (decision 3):** when OME-Zarr is asked for and the project has none, readers fall back to the npy/PNG files with one log note.
   - Both the old and the new folder layouts are readable.
6. **Versions:**
   - `zarr >= 3.1, < 4` (Zarr v3 format with sharding) and `ome-zarr` pinned to the version tested here (0.21.x on the build machine).
   - Both stay optional, under `spacr[zarr]`. `'ome-zarr'` refuses with an install hint when they are missing.
   - The versions are recorded in the run's settings stamp.
7. **Interoperability:**
   - The test opens the written plate with the ome-zarr-py reader and validates it with the OME-NGFF JSON schema (`ome-zarr` validator).
   - The proof shows napari (napari-ome-zarr) and Fiji (OME-Zarr reader / BigDataViewer) opening it, with screenshots in both themes.
8. **Docs and translations:** labels, tooltips and docstrings in 9 languages; arrival contracts and catalogue records in the same commit.

## Item 984 (for the 974 lane): the crop store and GUI readers on OME-Zarr
- **Scope:** Measure crops and every crop reader, built on the 983 store and writer model. The same alpha rules apply, and no default changes before 986.
- **Writing crops:**
  - `image_format` for Measure crops is one of `'ome-zarr'`, `'png'` (PNG (Legacy), today's default), `'tiff'` and `'npy'`.
  - In OME-Zarr mode, each well gets one sharded crop array. Crops are a fixed size, channels last as spaCR crops are today.
  - The crop arrays are written by the decision-1 writers:
    - local runs send chunks through shared memory to the writers that own each well;
    - cloud runs write per-worker parts registered in the DuckDB catalog and compact them afterwards.
  - PNG (Legacy) writes today's folders unchanged.
- **Crop index (decision 2):**
  - One row per crop. The locator is plate, well, field and the crop's index in the well array.
  - Next to it go the per-cell relationship columns: has a nucleus, infected or uninfected, pathogen count, class, and the object IDs (cell, nucleus, pathogen, organelle).
  - The crop array's Zarr attributes name the index table and these columns, so a reader knows what it can filter on.
  - Readers select crops by query on the index (for example, infected cells of class 2 in well B03), not by folder. PNG (Legacy) keeps today's folder-per-class layout.
- **GUI and app readers (decision 3):**
  - The annotation app (Load and Stream), Image-UMAP, Classify and every other crop reader offer OME-Zarr (Default) and PNG (Legacy).
  - Until 986 lands, OME-Zarr stays alpha and PNG stays the shipped default.
  - On an old project, OME-Zarr falls back to PNG with one log note.
  - Both the old and the new folder layouts are readable.
- **Proof (984):**
  - The PNG (Legacy) path, with the feature off, is byte-identical to nightly.
  - OME-Zarr crops read back pixel-identical to the PNG crops.
  - The crop index reproduces today's folder membership exactly, in counts per class and per infection state. Control: a query on a shuffled index must not match.
  - Annotation, Image-UMAP and Classify give identical results from either source.
  - File count, disk and read/write time are measured against PNG, with a control row; the published comparator is "n/a in paper".
- **Related:** 985 converts old projects; 986 switches the defaults (maintainer); 987 and 987b (the Cellpose batch .npz removal, with the batch plan served from Zarr chunks).

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
