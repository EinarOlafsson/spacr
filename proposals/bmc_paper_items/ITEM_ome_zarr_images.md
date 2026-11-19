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
- **Compute:** CPU only, about 1 h on the test tier. Heavy steps run through `heavy_slot.sh` (load policy of 2026-11-18, 14:40: up to 4 master slots, load1 < 30, RAM available >= 25 GB, pytest -n 4).

## Maintainer decisions (2026-11-18, about 11:30)
These bind 983 and 984.
1. **One writer per file.** Per-field (FOV) workers never share a shard.
   - **Local runs (corrected by the master on 2026-11-18; no shared memory):**
     - FOV workers return their results through the process pool's normal result pipe: per-object table rows with the relationship columns, plus a raw uint16 crop array `(n, C, H, W)`.
     - The parent is the only writer. It buffers crops per open well and per object type, and writes whole shards only, because a partial shard write means read-modify-write.
     - There are no shared files and no merge step.
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
   - **Local runs: the parent is the only writer. No shared memory.**
     - **Why not shared memory:** `multiprocessing.shared_memory` behaves differently on each platform.
       - Linux `/dev/shm` is 64 MB in Docker.
       - macOS limits segment names to 31 characters.
       - Windows has no unlink, and a segment vanishes on its last close.
       - Before Python 3.13 the resource tracker can unlink other processes' segments.
     - **Transport:** FOV workers return their results through the pool's normal result pipe:
       - label arrays, or the per-object table rows with relationship columns;
       - a raw uint16 crop array `(n, C, H, W)`.
     - **Buffering:** the parent buffers per open well and per object type, and writes whole shards only. A partial shard write would mean read-modify-write.
     - **Locators:** the parent assigns `(plate, well, array, row)` as it writes, then writes the table rows and the key table to the store.
     - **Commit:** a FOV is marked committed in the catalog only after all of its writes are done. FOVs are committed in plan order, with a small reorder buffer, so reruns are byte-identical.
     - **Workers** process whole FOVs, all object types together. Writers split by object type only if one writer becomes the bottleneck.
     - **Bounded RAM:** FOVs are queued well by well, so RAM is at most open wells × shard size.
     - **Chosen sizes:**
       - Field images and labels: one shard per field. A 2,960 px Funk field with 4 uint16 channels is about 70 MB, in 512 × 512 × C chunks.
       - Crops (984): ragged pools, `crops_per_shard` 4096 by default (16 chunks of 256 crops), lowered automatically on low RAM with a log line.
       - The parent keeps at most 2 open wells per object type, under the run's RAM budget.
       - The proof reports the measured peak buffer.
     - **Writes:** each shard is written to a temporary name, the file is closed, then renamed (Windows needs the close first).
     - **Writer idle time:** the proof measures the parent writer's idle time against worker time, to show the single writer is not the bottleneck.
   - **Cross-platform rules:**
     - Worker functions are top-level and spawn-safe.
     - Tests force the `'spawn'` start method on Linux, and a second test uses `'forkserver'`.
     - Store keys are short, for the 260-character Windows path limit.
     - Names are unique regardless of case, for macOS and Windows filesystems.
   - **Resume:** a rerun processes only FOVs not marked committed. Shards of uncommitted FOVs are ignored and overwritten.
   - **Cloud or multi-machine runs (the fallback when one parent cannot reach all workers):**
     - Each worker writes complete per-worker part arrays.
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

## Item 984 (for the 974 lane): the crop store, the crop table and the readers
- **Scope:**
  - Measure crops and every crop reader, built on the 983 store and writer model.
  - The same alpha rules apply, and no default changes before 986.
  - Decided by the maintainer on 2026-11-18, around 11:30 and around 12:00.
- **Crop shape: ragged, spaCR's own crop format.**
  - **Box:** each crop is its object's bounding box plus a margin (setting `crop_margin_px`), stored ragged, not resized.
  - **Pixel pool:** each well gets one flat pixel pool per object type, chunked and sharded, and the crop table gives each crop's `offset`, `height` and `width` into it.
  - **Mask crop:** the object's own mask crop is stored beside it in a mask pool (`mask_offset`).
  - **Object only or with context:** a view and loader option. Zeroing outside the object at write time is a separate setting (`crop_zero_outside`).
  - **Resizing:** CV loaders and Annotate resize deterministically at load.
  - **Not standard OME-Zarr:** viewers cannot open the pools directly. `spacr.crop_store.export_crops(...)` writes plain OME-Zarr images, TIFF or PNG for outside tools.
- **Where crops go:**
  - One store per well, `crops/plate=<P>/well=<W>.zarr`, as item 974's store already writes (master, 2026-11-19, replacing the earlier `crops/<plate>.zarr`). Writers own wells, so no two processes ever write the same container. The `merged/` folder keeps its name.
  - Cell, nucleus and pathogen pools are separate arrays/files.
  - Old `data/` PNG layouts stay readable.
- **Shards:**
  - `crops_per_shard` defaults to 4096, in 16 chunks of 256 crops.
  - It is lowered automatically when RAM is low, with one log line.
  - This replaces the 1,024 used in the first draft of this spec.
- **Writer:** the 983 model.
  - Workers process whole FOVs, all object types together.
  - The parent writes whole shards and commits FOVs in plan order, with a small reorder buffer, so reruns are byte-identical.
  - Writers split by object type only if one writer becomes the bottleneck.
- **Crop table:** `ssCell_Crops`, replacing `png_list`. Readers still accept old `png_list` tables.
  - **Identity:** object_id, object_type, plate, row, column, well, field, timepoint, object_label.
  - **Locator:** crop_store (path relative to the project), crop_array, crop_index, offset, height, width, mask_offset.
  - **Source box:** bbox_y0, bbox_x0, bbox_y1, bbox_x1, margin_px, clipped_at_edge.
  - **Relationships:** parent_cell_id, nucleus_id, has_nucleus, n_nuclei, n_pathogens, infected, one n_organelles column per organelle class, annotation, class, class_score.
  - **Legacy:** png_path, filled only in PNG mode.
  - **Per-array attributes or the catalog** hold the channels, pixel size, dtype, compression, spaCR version, settings hash and the list of relationship columns.
  - Readers select crops by query on these columns (for example, infected cells of class 2 in well B03), not by folder. PNG (Legacy) keeps today's folder-per-class layout.
- **Readers:** the annotation app (Load and Stream), Image-UMAP, Classify and every other crop reader.
  - **Sources:** OME-Zarr (Default once 986 lands; alpha until then) and PNG (Legacy).
  - **Fallback:** on an old project, OME-Zarr falls back to PNG with one log note. Old and new layouts are both readable.
  - **Annotate takes any number of channels.** Each channel has a role:
    - Image: colour, display range, gamma, additive blend, more than 3 channels allowed.
    - Outline: colour and width, drawn from a mask channel or a threshold.
  - Presets are saved per project. PNG (Legacy) keeps R, G, B.
- **Server, workstation and laptop:**
  - The number of workers is set automatically from cores and RAM, and a RAM budget caps the writer buffers.
  - On NFS and SMB shares: temporary name, close, then rename, with the catalog written last.
  - A small nightly cross-OS CI job runs on `windows-latest` and `macos-latest` for the crop-store, pool and Annotate-reader tests.
- **Proof (984):**
  - **Off identity:** with the feature off, the PNG (Legacy) path is byte-identical to nightly.
  - **Pixels:** ragged crops read back pixel-identical to the bounding box plus margin cut from the source images. Resized views equal the PNG crops after the same deterministic resize.
  - **Index membership:** `ssCell_Crops` reproduces today's folder membership exactly, in counts per class and per infection state. Control: a query on a shuffled index must not match.
  - **Reruns:** byte-identical, because of the plan-order commit.
  - **Readers:** Annotate, Image-UMAP and Classify give identical results from either source.
  - **Annotate:** works with more than 3 channels and with outline roles.
  - **Cost:** writer idle time against worker time, the peak buffer against the RAM budget, and file count, disk and read/write time against PNG, with a control row. The published comparator is "n/a in paper".
  - **Platforms:** the spawn and forkserver tests pass, and the nightly Windows and macOS job is green.
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
- **Writer and platforms:**
  - Writer idle time against worker time, and the peak parent buffer against open wells × shard size.
  - The spawn and forkserver tests pass.
  - Resume after a forced stop reruns only uncommitted FOVs and gives byte-identical arrays.
- **Viewers:** napari and Fiji open the plate, with channel names and pixel size shown.
- **Report:** `BMCspacr/reanalysis/item983_proof/REPORT.md`, with tables, controls and a glossary.
