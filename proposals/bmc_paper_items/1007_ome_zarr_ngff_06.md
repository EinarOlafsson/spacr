# Item spec: OME-Zarr NGFF 0.6 support (read, opt-in write, registration as coordinate transformations, lossless 0.5 to 0.6 converter)

- **Lifecycle:** future (alpha), behind "Show alpha features"; the default (off) leaves every reader and writer unchanged (0.5 written, 0.4 and 0.5 read).
- **Number:** 1007 (1003-1006 are used on other spec branches; confirm against origin/nightly and the claims files before building).
- **Rule:** the pixel data of a store must never change in a conversion or a rewrite: the chunk bytes are identical (md5), only metadata moves. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly (written against c9fddfc769): `spacr/ome_zarr.py` (`SUPPORTED_NGFF_VERSIONS = ("0.4", "0.5")`, `_open_ome_zarr`, `write_ome_zarr`), `spacr/ome_zarr_convert.py` (`_verify_field` checks `ngff_version == "0.5"`).
- **Paper-critical: YES (correction 2026-11-21).** Maintainer decision 2026-11-21 (later the same day): the one-time final Funk runs wait until spaCR writes NGFF 0.6 natively, so the final runs write 0.6. `ome_zarr_ngff_version` must therefore be settable to `'0.6'` for the OPS phenotype plate stores (merged field images and their label masks; `ops_ph_image_format 'ome-zarr'`). The crop pools (`ops_crop_image_format 'ome-zarr'`) are spaCR's own zarr-format 3 pools, not NGFF images, and carry no `ome` metadata, so the version does not apply to them.
- **BUILD GATE: MET (correction 2026-11-21).** 0.6 is the current official release (released 2026-09-14), verified on ngff.openmicroscopy.org on 2026-11-21. The reader accepts 0.6 unconditionally; the writer writes it when asked.
- **Default:** `'0.5'` stays the spaCR default. Flipping it to `'0.6'` is a maintainer decision (see "Open points").
- **Lanes (2026-11-21):** A = reader, writer, plate store and OPS wiring (`spacr/ome_zarr.py`, `spacr/image_store.py`, the `spacr/ops_engine.py` PlateWriter call sites, the `ome_zarr_ngff_version` setting); B = converter (`spacr/ome_zarr_convert.py`, T8-T10); C = registration (`spacr/ome_zarr_registration.py`, T7). B and C branch off A's head and do not edit `spacr/ome_zarr.py`. Lane A's API for them is listed under "Lane A API".
- **Related:** items 919 and 998 (SBS to phenotype shifts), `ITEM_ops_placement_seed_file.md` (tile placement), item 936 (align and stitch shared core), the OME-Zarr export and convert work in `ome_zarr_convert.py`.

## Spec status (checked 2026-11-21)
The brief described 0.6 as release candidate rc0 (about July 2026). The pages fetched on 2026-11-21 say otherwise; the maintainer must reconcile this before the gate is called open.
- `https://ngff.openmicroscopy.org/specifications/index.html`: "The current released version of the OME-Zarr specification is 0.6." The dev version is 0.9.dev2 (unreleased).
- `.../specifications/0.6/version_history.html`: "0.6 - 2026-09-14"; earlier labels were 0.6dev1 (2025-11-18) to 0.6.dev4 (2026-04-22). No rc label found.
- RFC-5 (coordinate systems and transformations): status S3 Implementation, associated version 0.6. Adopted in 0.6.
- RFC-4 (anatomical orientation): S1 Accepted, but the `orientation` axis field appears only in the dev spec 0.9.dev2, NOT in 0.6. Orientation support is therefore a separate, later part of this item (see "Later: orientation").
- Re-fetch the 0.6 page at build time and diff against the table below; the field lists here come from the published tables and a fetch summary.
- **Re-fetched 2026-11-21 at build time** (0.6/index.html and 0.6/version_history.html). The index says "The current released version of the OME-Zarr specification is 0.6"; changelog "0.6 - 2026-09-14". The diffs against the table are in "Spec diffs found at build time". The table is left as written; where they differ, the diffs win.

## Spec diffs found at build time (2026-11-21)
1. **Multiscale-level `coordinateTransformations` change meaning.**
   - In 0.4 and 0.5 they apply to every level and are composed into the level scale.
   - In 0.6 they map the intrinsic system to another named system. One side MUST name the intrinsic system. The other side names a system of the same multiscales, or of a child labels group (by `name` + `path`).
   - They are NOT folded into the level spacing.
   - For lane B: a 0.5 store with multiscale-level transforms has no lossless 0.6 form. Refuse it; folding the transforms in would change the file.
2. **Dataset endpoints.**
   - Input is `{"path": <dataset path>}` and SHOULD omit `name`.
   - Output is `{"name": <intrinsic>}` and SHOULD omit `path`. The output name MUST be the same at every level.
   - Inner transforms of a `sequence` carry no `input`/`output`. Only the outer wrapper object does.
3. **`byDimension` items** have the form `{"transformation": {...}, "inputAxes": [int], "outputAxes": [int]}`. The nested key is singular (`transformation`), and the axes are integer indices, not names.
4. **`scale` and `translation` are JSON-only.** 0.6.dev3 says "Zarr parameter storage no longer allowed for scale/translation". Only `affine` and `rotation` may use `path`, which points to a plain 2D Zarr array:
   - affine: shape M x (N+1), without the homogeneous last row;
   - rotation: shape N x N.
5. **`inverseOf` was removed** in 0.6.dev3. It is not in the table. The reader refuses it and gives the reason.
6. **`mapAxis`** is an integer array: output axis i is input axis `mapAxis[i]`.
7. **`projectAxis`** takes `droppedInputs` and `createdOutputs`, both integer indices. Created outputs are 0. It is part of 0.6: the index links to the dev docs, but the section is on the 0.6 page.
8. **Other transform types and axes.**
   - `bijection` has `forward` and `inverse`.
   - `displacements` and `coordinates` take `path` and an optional `interpolation`. The path points to a normal multiscale group with its own `ome.multiscales`.
   - Displacement and coordinate axes need `type` and `discrete`.
9. **Axis fields.**
   - `name` MUST be present; `type` and `unit` SHOULD; `discrete` and `longName` MAY.
   - `type` adds `array`, `coordinate` and `displacement` to space, time and channel.
10. **multiscales keys.**
    - `coordinateSystems` and `datasets` are required. `coordinateTransformations` is optional. `name`, `type` and `metadata` SHOULD be present.
    - There is no `axes` key.
    - An image has 2 to 5 axes, 2 or 3 of them of type space.
11. **The intrinsic system has no mandated name.** The spec's complete example uses `"intrinsic"`; other examples use `"physical"`. spaCR writes `"physical"` (this spec's choice) and reads any name.
12. **Transform arrays** SHOULD live in a group named `coordinateTransformations` beside the images, as the table says.
13. **Plate, well and labels.**
    - `version` is removed from `plate`, `well` and `labels`. spaCR also drops it from `image-label`.
    - 0.6.dev3 relaxed the characters allowed in well image paths (`a-z A-Z 0-9 - _ .`). spaCR's keys already fit.
14. **`arrayCoordinateSystems` is removed.** Pixel coordinates are a unitless named system (the spec's example names it `"array"`), linked by a scale.
15. **Reader conformance** is as the table says. In addition, a reader SHOULD warn on transforms it cannot apply. The transform graph MUST be connected; spaCR does not check this.
16. **Dev track and orientation.** The index page labels the dev track 0.9.dev2. RFC-4 `orientation` is still not part of 0.6.

## Lane A API (for lanes B and C; all in `spacr/ome_zarr.py` unless stated)
- **Constants.**
  - `SUPPORTED_NGFF_VERSIONS = ("0.4", "0.5", "0.6")`
  - `WRITABLE_NGFF_VERSIONS`
  - `NGFF06_INTRINSIC_SYSTEM = "physical"`
  - `NGFF06_MATRIX_TRANSFORMS`, `NGFF06_FIELD_TRANSFORMS`
- **Mappers:** `multiscale_to_ngff06(entry_05, *, system="physical") -> dict` and `multiscale_to_ngff05(entry_06) -> dict`.
  - Pure JSON mappers of one `multiscales` entry. They keep key order, and `multiscale_to_ngff05(multiscale_to_ngff06(e)) == e`.
  - Each raises `OmeZarrError` naming the key that has no lossless form.
  - To 0.6: a `version` inside the entry, multiscale-level `coordinateTransformations`, a level list other than `[scale]` or `[scale, translation]`, or extra keys inside a level transform.
  - Back to 0.5: more than one coordinate system, multiscale-level transforms, or other level transform types.
  - Lane B maps plate, well, labels and image-label itself: it drops `version` there, since `ome.version` carries it.
- **`transformation_matrix(obj, ndim, *, root=None) -> ndarray | None`.**
  - Returns the homogeneous `(M+1, ndim+1)` matrix of any matrix type, including `affine` and `rotation` stored by `path` (read relative to `root`).
  - For `displacements` and `coordinates`, it returns `None` and warns once.
- **Writer:** `write_ome_zarr(..., ngff_version="0.6", coordinate_systems=[...], coordinate_transformations=[...])`.
  - Adds extra named systems after `"physical"`, plus multiscale-level transforms.
  - Both are checked before any chunk is written. 0.6 only.
- **Reader fields.**
  - `OmeZarrImage.coordinate_systems` holds every system as written; `OmeZarrImage.transformations` holds the multiscale-level transforms as written. Both are empty for 0.4 and 0.5.
  - `Axis.discrete` and `Axis.long_name` are read from 0.6 files only.
- **`spacr.image_store.PlateWriter(..., ngff_version="0.5" | "0.6")`.** One version per store; a store started in the other version is refused.
- **Setting:** `ome_zarr_ngff_version` (OPS, alpha, default `'0.5'`), read by `spacr.ops_engine._ph_ngff_version`.

## Why
- Stores that carry the OPS registration (SBS to phenotype shifts, tile placement) keep it in spaCR sidecar tables. 0.6 can carry it inside the store as named coordinate systems and transformations, readable by other OME-Zarr tools.
- Once other tools write 0.6, spaCR must read it; a reader that refuses 0.6 blocks data exchange.
- The maintainer keeps the choice: 0.5 stays the default until he switches.

## What changes from 0.5 to 0.6 (what the code must handle)
| Area | 0.5 | 0.6 |
|---|---|---|
| Version key | `attributes.ome.version = "0.5"` | `"0.6"`; `version` no longer appears in `multiscales`, `plate`, `well`, `labels` metadata |
| Coordinate systems | `axes` in each multiscale | `coordinateSystems` list: each has a unique non-empty `name` and `axes`; datasets map `{"path": ...}` to `{"name": ...}` |
| Transform endpoints | `input` / `output` string or object | always an object `{"name": str, "path": str}` (breaking) |
| `byDimension` | `input_axes` / `output_axes` | `inputAxes` / `outputAxes` (breaking) |
| `arrayCoordinateSystems` | present | removed; dimensionless transforms explained instead (breaking) |
| New transform | none | `projectAxis` (`droppedInputs`, `createdOutputs`) |
| Transform types | scale, translation | `identity`, `mapAxis`, `projectAxis`, `translation`, `scale`, `affine` (list or `path`), `rotation` (list or `path`), `sequence`, `displacements`, `coordinates`, `bijection`, `byDimension` |
| `multiscales > datasets` | scale (+ translation) | one `scale`, `identity` or `sequence` of scale then translation; input `{"path"}`, output `{"name": "physical"}` |
| Axis fields | name, type, unit | adds `discrete`, `longName` (displacement and coordinate axes: `type` and `discrete` required) |
| Reader conformance | n/a | MUST parse `identity`, `scale`, `translation` and sequences of them; SHOULD parse `mapAxis`, `affine`, `rotation`, `byDimension` |
| Transform arrays | n/a | stored in a Zarr group beside the images, called `coordinateTransformations` |

## What to build
1. **Read 0.6** (`spacr.ome_zarr`):
   - Add `"0.6"` to `SUPPORTED_NGFF_VERSIONS` when the gate opens; `_open_ome_zarr` reads `ome.version` only (no per-multiscale `version`).
   - Parse `coordinateSystems` and the `{name, path}` endpoints. Resolve the dataset transform to spacing and offset as now (`spacing_from_axes` unchanged for `scale` and `translation`).
   - Support `identity`, `scale`, `translation`, `sequence` of those (MUST), plus `affine`, `rotation`, `mapAxis`, `byDimension` (SHOULD). Any other type: keep it in the metadata object, read the pixels, and warn once with the transform type; never guess.
   - Axis `discrete` and `longName` kept in the Axis record. 0.4 and 0.5 reads stay byte for byte as today.
2. **Write 0.6 behind a setting** `ome_zarr_ngff_version` (`'0.5'` default, `'0.6'` allowed; `'0.4'` stays as now). `write_ome_zarr` emits the 0.6 layout (version only at `ome.version`, `coordinateSystems`, `{name, path}` endpoints). 0.5 output is unchanged and tested against the current golden files.
3. **Registration as coordinate transformations** (setting `ome_zarr_store_registration`, default False; requires 0.6):
   - Per field image: a named coordinate system per frame, for example `phenotype` (the intrinsic one, `physical`), `sbs`, `tile_global`.
   - SBS to phenotype shifts (items 919, 998): a `translation` (or `sequence` of scale and translation) with input `{"name": "phenotype"}`, output `{"name": "sbs"}`, in `multiscales > coordinateTransformations`.
   - Tile affines: an `affine` list per tile, mapping the tile to `tile_global`; large or shared matrices as arrays in the `coordinateTransformations` group referenced by `path`.
   - The stored values are the numbers the spaCR tables hold; a reader returns them equal to the tables to the last bit (float64 JSON round trip). The sidecar tables stay the source of truth until the maintainer says otherwise.
4. **Converter** `spacr.ome_zarr_convert.upgrade_ngff(src, dst, to='0.6')` (GUI: Convert tab, alpha), and the reverse `to='0.5'` where lossless:
   - With a new `dst`, chunk files are hard-linked (copied if the filesystem refuses) and only `zarr.json` metadata is rewritten.
   - Maps 0.5 to 0.6: removes the per-multiscale version, builds `coordinateSystems` from `axes`, rewrites dataset transforms to the `{path}` to `{name}` form, rewrites labels, plate and well metadata. Anything it cannot map is refused with the reason, never dropped.
   - Built-in verification: md5 of every chunk file in source and destination is identical, and the metadata round trip 0.5 to 0.6 to 0.5 returns the original JSON (key order aside). A report lists arrays, chunks, the md5 of the chunk list and the verdict.
   - Resume, atomic write and lock behave like the existing convert (`_atomic_json`, `_lock`); `_verify_field` accepts the target version.
5. **Alpha, not default:** "Show alpha features" controls the 0.6 settings and the converter button. The reader needs no switch (reading a valid 0.6 store is always on once the gate opens).

## Later: orientation (RFC-4)
`orientation` (`{"type": "anatomical", "value": ...}`, spatial axes only, no implicit default, absent equals null) is in 0.9.dev2, not in 0.6. Do not write it in 0.6 stores. When a final release contains it: the reader keeps the field on the Axis record and the writer omits it unless `ome_zarr_axis_orientation` is set (microscopy images normally have none; writers SHOULD omit). Separate follow-up item then.

## Settings (alpha)
| Key | Default | Meaning |
|---|---|---|
| `ome_zarr_ngff_version` | `'0.5'` | version written by `write_ome_zarr` and the exporters: `'0.5'` or `'0.6'` |
| `ome_zarr_store_registration` | False | store SBS to phenotype shifts and tile affines as coordinate transformations (needs `'0.6'`) |
| `ome_zarr_upgrade_link` | True | the converter hard-links chunk files instead of copying when possible |

## Tests (CPU, synthetic, small)
- **T1** Read a hand-built 0.6 store (2D CYX and 3D ZYX) with `scale` + `translation`: spacing, offset and pixels equal the 0.5 equivalent.
- **T2** Read 0.4 and 0.5 fixtures: outputs byte for byte equal to current nightly (off by default, ratchet).
- **T3** `write_ome_zarr(..., ome_zarr_ngff_version='0.6')`: version only at `ome.version`, none in `multiscales`; `{name, path}` endpoints; our reader reads it back identically.
- **T4** Default write is still 0.5 and equals the current golden metadata.
- **T5** Transform conformance: `identity`, `scale`, `translation`, `sequence`, `affine`, `rotation`, `mapAxis`, `byDimension` (`inputAxes` / `outputAxes`) parse; an unknown type warns once and the pixels still read.
- **T6** Old spelling: a 0.6 file with string endpoints or `input_axes` gives a clear error naming the key.
- **T7** Registration: stored shifts and tile affines read back exactly equal to the source tables; the sequence order is translation after scale.
- **T8** Converter 0.5 to 0.6: md5 of every chunk file identical; metadata round trip returns the original; hard-link and copy modes give the same verdict.
- **T9** Converter refusals: an unmappable 0.5 attribute stops with the reason, writes nothing and leaves `src` untouched; an interrupted run resumes.
- **T10** Labels, plate and well groups convert and read (version key removed, the rest unchanged).
- **T11** Settings-file rerun: the same settings file gives identical stores (metadata and chunks).
- **T12** Gate guard: while the gate is closed, `'0.6'` is refused with a clear message; after it opens the same test asserts acceptance.

## Arrival contract and catalogue
New documented callables (`upgrade_ngff`, any public transform parser) get the arrival contract and 9-locale records in their own commit. Alpha settings go into the settings schema with descriptions; no new tutorial (alpha tutorials: only the two dedicated lessons).

## Size
M: reader and writer about 400 lines, converter about 300, registration storage about 200, plus tests T1-T12 and docs. 2 to 3 agent lanes (reader/writer, converter, registration) after the gate; the reader lands first because all lanes touch `ome_zarr.py`.

## Open points for the maintainer
- ~~Confirm the gate~~: met. 0.6 was released 2026-09-14 (verified 2026-11-21).
- When to flip the default from 0.5 to 0.6.
  - The final Funk runs set `ome_zarr_ngff_version '0.6'` explicitly in their settings file, so they do not need the flip.
  - Flipping the default would change the stores of every new OPS run and needs a DEFAULT_CHANGES row (schema 953 is reserved for 1007). It stays the maintainer's decision.
- Whether the sidecar tables stay the source of truth for registration, or the store becomes it.


## Lane A crop-benchmark correction: NGFF 0.6 reader interoperability (verified 2026-11-21)

The validated independent metadata reader is **`ome-zarr-models==1.8.1`**.
`open_ome_zarr(path, version="0.6")` accepts the spaCR plate, well, image,
and label metadata. Use the spaCR OME-Zarr reader for pixel arrays.

**`ome-zarr==0.21.0` does not declare NGFF 0.6 support.** Its
[format registry, lines 24–32](https://github.com/ome/ome-zarr-py/blob/v0.21.0/ome_zarr/format.py#L24-L32)
ends at 0.5. Its [multiscale reader, lines 279–287](https://github.com/ome/ome-zarr-py/blob/v0.21.0/ome_zarr/reader.py#L279-L287)
reads the old per-multiscale version and axes keys. Their absence in a valid
0.6 store selects the [five-axis fallback, lines 79–81](https://github.com/ome/ome-zarr-py/blob/v0.21.0/ome_zarr/axes.py#L79-L81),
which causes the reported axis-4 error when assembling a three-dimensional
plate field. Adding legacy axes to a 0.6 multiscale is not the correction.
The interoperability test records the installed reader version and skips a
format that the library does not support; a future reader that declares 0.6
must open the plate and return identical pixels.

New plate fields use the existing shared OME-Zarr chunk policy: one channel
and at most 256 × 256 spatial pixels per unsharded chunk, with Blosc LZ4 and
byte shuffle. A uint16 tile is 128 KiB. The tile edge aligns with common
64- and 128-pixel cell crops. Images and masks retain their exact pixel
bytes and dtype; the Zarr array metadata records the chunk shape. Existing
committed fields are read and resumed without rewriting their chunks.


### Synthetic crop benchmark (2026-11-21)

A four-channel 1994 × 1994 uint16 field, 1,000 identical random 64 × 64
crops, seed 20261121, local storage after one warm-up read. The comparison
used one CPU thread, one Blosc thread, and Zarr async concurrency one.

| Layout | Chunk shape (C, Y, X) | 1,000 crops |
|---|---|---|
| Previous PlateWriter | 1 × 1994 × 1994 | 7.811 s |
| Shared default tiles | 1 × 256 × 256 | 1.936 s |

This synthetic run was 4.04× faster. Full pixel bytes, dtype,
random crop positions and crop checksums were identical. This is a local,
warm-cache comparison; it does not reproduce the paper-session storage
benchmark. The regression test allows 30 seconds for 1,000 crops on its
smaller synthetic field.
