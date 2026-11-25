# Item spec 1015: cell crops on OME-Zarr NGFF 0.6 (crop views over the field store, and 0.6 crop atlases) (alpha)

- **Lifecycle:** future (alpha), behind "Show alpha features". The default stays the current zarr-v3 crop pool (`layout='pool'`) until the maintainer flips it.
- **Number:** 1015. 1008, 1009, 1010 and 1011 are used on other spec branches. Confirm against origin/nightly and the claims files before building.
- **Base:** origin/nightly 19a95cd799. Item 1007 lane A (`claude-1007-impl-1121`, 7b086319ea) lands first: this item calls its 0.6 writer and mappers.
- **Evidence:** `BMCspacr/reanalysis/crop_store_ngff06_1121/REPORT.md`, with every number, its spread and its scripts. The benchmark used plate A, well A1, sites 1000-1002, and 3,131 whole cells; md5-identical in every layout. Figure: paper repo `supplementary/figure_S13`.
- **Rule:** same operation, same code.
  - Crops are written only through `CropPoolWriter` and read only through `read_crop` / `read_crops`.
  - NGFF metadata, zarr arrays and shards are written only through `spacr.ome_zarr`.
  - Field images are read only through `spacr.image_store` / `spacr.ome_zarr`.
  - No second writer or reader anywhere.
- **Pixel rule:** a crop read in any layout is byte-identical to `cut_crops` + `_ph_crop_box` on the same field: uint16 pixels, and a uint8 mask with 0 for background, 1 for the cell and 2 for its nucleus. Every test checks this with md5.

## Measured basis (NVMe, cold cache, 1 thread; REPORT has HDD and spreads)
| Layout | Extra disk / cell | Random load s / 1,000 | Sequential s / 1,000 | Write s / 1,000 | Files / well (extrap.) |
|---|---|---|---|---|---|
| PNG files (legacy, ref.) | 81.5 kB | 1.6 | 2.3 | 3.8 | 4.5 M |
| spaCR pool, 4,096 / shard (current) | 44.8 kB | 100 (spaCR read_crop) / 107 (open once) | 1.1 | 4.3 | 1,474 |
| 0.6 views (field store as the final run writes it) | 0.23 kB (table) | 412 | 1.0 | 0 (fields written anyway) | 0 |
| 0.6 views, fields tiled 256 px | 0.23 kB (table) | 3.9 | 2.3 | 0 | 0 |
| 0.6 crop atlas, chunk 128 (spaCR shard writer) | 84.0 kB | 1.2 | 0.7 | 5.4 | 10,461 |

- The pool is the smallest stored copy. Its zstd stream over 4,096 crops also removes the context that neighbouring crops share; small independent chunks cannot do that.
- A random read from the pool decodes a whole chunk of about 200 crops (`chunk = total // 16`), which costs 0.1 s per crop.
- No standard OME-Zarr reader opens a pool (REPORT, "Standard readers").

## What NGFF 0.6 allows (quotes in REPORT)
- 0.6 has no collection or table concept. RFC-8 (Collections) is "R1 (Under review)", and no tables RFC exists.
- Several images in one store only work through the transitional `bioformats2raw.layout` with a `series` list.
- An image has at most one channel or custom axis, plus optional time and 2-3 space axes. A crop stack `(n, c, y, x)` would therefore have to misuse the time axis.
- A multiscales-level transformation can only reach systems of its own image or of a child labels group, so a per-crop transform back to the field needs one image per crop. The benchmark measured that layout at 109 kB per cell, 10.5 M files per well and 62 s per 1,000 writes.
- Labels must be integer and have the image's levels.

## Design
1. **`layout='views'` (proposed for OPS once T10 passes).**
   - No crop pixels are written. The crop rows point to the field: `crop_store` is the plate store and field key, plus `bbox_y0/x0/y1/x1`, `object_label` and `mask_source`, which is the labels names (`cells`, `nuclei`).
   - The bbox is in the field's own array coordinates, the 0.6 intrinsic system. No transform is needed.
   - `read_crops` groups rows by field. It reads each field and its labels once, through `image_store`, then cuts every crop with `cut_crops` and `_ph_crop_box`, so it is the same code as the writer path.
   - `read_crop` reads the bbox region.
   - Fast random reads need tiled field chunks. A new `PlateWriter(field_chunks=...)` setting takes `'field'` (current) or `256`, sharded per field. The cost is measured: sequential crop reads go from 1.0 to 2.3 s per 1,000 through zarr-python. Lane A's shard-index reader should remove most of that, and test T12 measures it before any default changes.
   - Preconditions, checked at run start and refused with the reason otherwise:
     - `ops_ph_image_format 'ome-zarr'`;
     - `ops_crop_flatfield false`;
     - the stored field is the stack that `_ph_crop_field` cuts from: raw, rounded, secondary channels already shifted.
2. **`layout='atlas'` (materialised 0.6 crops, for sharing and random-order training).**
   - Per well and class: `crops/plate=<P>/well=<W>.zarr`.
     - The root has `ome: {version 0.6, bioformats2raw.layout: 3}`.
     - `OME/zarr.json` has a `series` key listing `cell/<k>`.
     - `cell/` is the class group: `spacr` attrs, `rows.jsonl`, `spacr_commit.json`.
     - `cell/<k>` is one 0.6 image per shard of `crops_per_shard` crops: axes c,y,x, width 4,096, chunks (C, 128, 128), shards of 16 chunk rows, zstd 3, empty chunks not written.
     - `cell/<k>/labels/mask` holds the uint8 masks.
   - Crops are shelf-packed on the chunk grid: each starts on a chunk corner, so a random read decodes only its own chunks.
   - Rows gain `atlas_y0` and `atlas_x0`, and `offset` is None. The atlas-to-field translation is `bbox - atlas` and stays in the table, since 0.6 has no per-crop transform inside one image.
   - Write every shard whole with lane A's direct shard writer. The atlas took 4.3 s per 1,000 crops written this way and 24 s through zarr-python's partial writes, with the same bytes on disk. Read through lane A's shard-index reader: 1.2 s per 1,000 random reads against 20-35 s through zarr-python slicing.
3. **`layout='pool'`** is unchanged. An optional follow-up: `crop_pool_chunk_crops`, which sets the pool chunk to N crops instead of `total // 16`, would cut its random cost. It is not measured here.
4. **Tables:** 0.6 defines none, so the crop table stays in `rows.jsonl` and the measurements database. No anndata until a tables RFC is adopted.

## Lanes and files
- **Lane A (`spacr/ome_zarr.py`; the 1007 lane A author):**
  - `create_ome_zarr(path, shape, dtype, *, axes, spacing, chunks, shards, compressor, compression_level, fill_value, ngff_version, name, channel_names, labels=False) -> OmeZarrArrayWriter`.
  - `OmeZarrArrayWriter.write_shard(key, block)`: one whole shard, with zstd inner chunks in C order, the index at the end, crc32c, and empty chunks marked. This is the `sharding_indexed` byte layout.
  - `OmeZarrArrayWriter.finish(omero_sample=None)`: writes the metadata through the existing `multiscale_to_ngff06` path.
  - `ShardIndexReader(path).read_region(starts, stops)`: reads each shard's index once, then preads and decodes only the needed chunks.
  - `write_label_attrs(...)`, moved from `image_store.PlateWriter.write_field` so both callers share it, and `write_series_root(root, series, ngff_version)`.
  - `write_ome_zarr` is rebuilt on these and stays byte-identical to its golden files.
- **Lane B (`spacr/crop_store.py`):**
  - `CropPoolWriter(..., layout='pool'|'atlas'|'views', atlas_chunk=128, atlas_width=4096)`. `_flush` branches into `_flush_pool` (unchanged), `_flush_atlas` (`_pack_on_chunk_grid` plus lane A) and `_flush_views` (rows only).
  - The same `commit`, `read_commit` and `discard_uncommitted`. For atlases, `discard_uncommitted` removes uncommitted `cell/<k>` and rewrites the series list.
  - `read_crop`, `read_crops` and `_crop_from_row` branch on the row's layout; `pool_attributes` records the layout. A class never mixes layouts.
  - `convert_crop_layout(project, plate, well, cls, to)` converts with an md5 check per crop.
- **Lane C (callers and settings):**
  - `spacr/ops_engine.py`: `_ph_crops` / `_ph_write_crop_pool` pass the layout. For views, `_ph_crop_field` computes boxes and labels without copying pixels, and the run-start precondition check goes here.
  - `spacr/image_store.py`: `PlateWriter(field_chunks=...)` and `read_field_region(...)`.
  - `spacr/measure.py`: the crop sink passes the layout.
  - `spacr/ome_zarr_convert.py`: the converter's `CropPoolWriter` call passes the layout.
  - `spacr/settings.py` and `spacr/settings_spec.py`: `crop_store_layout` and `ops_crop_store_layout` ('pool' default, 'views', 'atlas'), `crop_atlas_chunk` (128) and `ops_ph_field_chunks` ('field', 256). All alpha. Strings go to the Codex translation inbox. A DEFAULT_CHANGES row is needed only when a default flips.
- **Lane D (`tests/test_crop_store_ngff06.py`):** T1 to T12, CPU only, synthetic and small, except T10.

## Tests
- **T1:** Atlas round trip. Random sizes from 1 to 300 px, 4 channels, uint8 masks 0/1/2, edge-clipped crops included. `read_crop` and `read_crops` are md5-identical to the input.
- **T2:** Packing. No two crops overlap, every crop starts on a chunk corner, and each chunk belongs to at most one crop.
- **T3:** 0.6 conformance. `read_ome_zarr` opens every atlas `cell/<k>` as 0.6 with axes c,y,x and labels `mask`. The root has `bioformats2raw.layout`, and `OME/zarr.json` lists exactly the committed images.
- **T4:** The direct shard writer and zarr-python's writer give the same decoded array, and zarr-python reads the direct writer's shards.
- **T5:** `ShardIndexReader.read_region` equals zarr slicing on random boxes, including the edges and empty chunks.
- **T6:** Resume. A stopped writer leaves uncommitted images, `discard_uncommitted` cleans them up, and the rerun is byte-identical.
- **T7:** Layout guard. Adding atlas crops to a pool class, or the reverse, is refused with the reason.
- **T8:** The `write_ome_zarr` golden files are unchanged after the lane A split (0.4, 0.5 and 0.6).
- **T9:** Views round trip. Crops read from a 0.6 PlateWriter field store are md5-identical to `cut_crops` + `_ph_crop_box`, for both `field_chunks` options.
- **T10:** OPS `ph_crops` on the e2e test tier with `ops_crop_store_layout='views'` and with `'atlas'` gives the same crops (md5) as `'pool'`. `_ShardCrops` reads agree. The run-start precondition refuses a flat-fielded crop setting.
- **T11:** A settings-file rerun gives identical stores (metadata and chunk bytes).
- **T12:** Benchmark gate before any default flip: rerun `reanalysis/crop_store_ngff06_1121/scripts` on the built code. Views with tiled fields must keep whole-field reads within 1.5x of `'field'` chunks (lane A reader). Random crop reads must be at least 10x faster than the pool.

## Open points for the maintainer
- Whether the one-time final Funk runs write views (no crop pixels, saves 67 GB per well against the pool) or pools now and convert later. `convert_crop_layout` is lossless and md5-checked.
- The tiled field chunks for random access, against whole-field read speed (T12).
- The default flip from pool, and the atlas chunk size (128 measured best for reads, at the same disk use).
