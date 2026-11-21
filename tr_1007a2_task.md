# Translation handover: item 1007 lane A crop/interoperability fix

Branch: `ws-offload-codex-work-1007a-fix`. Based on C candidate `c1d432ed0c6990f9cd24e188181340995acafb6b` (lane A base `e2a1ace77c`). C publication must precede A-fix publication. WORK_STATUS.md identifies the eventual final commit.

Home owns reviewed records and generated catalog arrivals. Required locales: sv, de, es, zh_CN, pt, hi, ko, is, fr. This body applies to the A-fix branch; C retains its earlier body and needs matching records there.

Inventory: one changed canonical API block; zero added runtime messages, settings, or public signature changes. The canonical extractor and AST comparison verified this inventory.

## Changed canonical API block

`spacr/image_store.py:90` and unchanged constructor summary at `spacr/image_store.py:118` — `spacr.image_store.PlateWriter`; API records required.

Previous C body SHA-256: `926a60609ce4c36837643d89dd9e83ba82a33bd0a321d4cd23157f6a53cc0d20`

Current body SHA-256: `c887438cad237760ae0db873342f4bfb46047cbf9ceb493c0644d126bac95463`

```text
The only writer of one plate's OME-Zarr store (item 983).

:param root: the folder that holds ``<plate>.ome.zarr``.
:param plate: the plate name.
:param channel_names: the image channels' names, written to ``omero``.
:param spacing: a :class:`spacr.layers.Spacing` for the pixel size, or
    None to record pixels with no physical unit.
:param ngff_version: ``'0.5'`` (the default) or ``'0.6'`` (item 1007):
    the OME-Zarr version of every group in the store. One store holds
    one version, so a store started in the other one is refused.
:param registration: optional field-number to OPS registration-row mapping.
    Requires version 0.6 and pixel coordinates (spacing=None). Existing
    stores must use the same registration mode.

Fields are written whole with :meth:`write_field`; the plate and well
metadata are rewritten from the ledger after each field, so a reader
opening the store mid-run sees only committed fields. Image and label
arrays share the OME-Zarr default chunk policy: one channel and up to
256 x 256 spatial pixels per chunk, clipped to the array shape. These
128 KiB uint16 tiles align with 64- and 128-pixel cell crops and bound
the amount decoded for a region read. Chunks remain unsharded and use
Blosc LZ4 with byte shuffle.
Open (or start) the plate store and read its commit ledger.
```

Binding: `tests/data/release_contracts/1007a2_ome_zarr_plate_chunk_api_changes_2026-11-21.json`, hash-chained after the retained 983, 1007 A and 1007 C bodies. Exact canonical verification: `1007a2-sources.json` in the recovery directory.

## Private and test docstrings: no records

The obsolete private `_field_chunks` helper and its docstring were removed. The writer now reuses `spacr.ome_zarr._default_chunks`; it does not introduce a second chunk-selection implementation.

- `tests/test_ome_zarr_plate_interop_chunks_1007.py:1` `<module>` — private test/helper; no records. NGFF 0.6 plate interoperability and crop-friendly shared chunk policy.
- `tests/test_ome_zarr_plate_interop_chunks_1007.py:15` `field_data` — private test/helper; no records. A two-channel field and integer masks spanning several spatial chunks.
- `tests/test_ome_zarr_plate_interop_chunks_1007.py:23` `_same_pixel_bytes` — private test/helper; no records. Compare dtype, shape and every pixel bit, including signed zero.
- `tests/test_ome_zarr_plate_interop_chunks_1007.py:30` `test_ngff06_plate_well_image_and_labels_validate_with_models` — private test/helper; no records. The pinned independent 0.6 model reader accepts every HCS group type.
- `tests/test_ome_zarr_plate_interop_chunks_1007.py:48` `test_plate_opens_with_a_supporting_ome_zarr_reader` — private test/helper; no records. Record the reader version and skip only a missing declared format capability.
- `tests/test_ome_zarr_plate_interop_chunks_1007.py:70` `test_tiled_plate_pixels_match_whole_field_reference_bytes` — private test/helper; no records. Spatial tiling preserves image and mask bytes at every pyramid level.
- `tests/test_ome_zarr_plate_interop_chunks_1007.py:104` `test_one_thousand_random_cell_crops_have_a_generous_time_bound` — private test/helper; no records. A thousand 64-pixel crops avoid the former whole-field decoding cost.
- `tests/test_documentation_i18n.py:457` `_current_1007_ome_zarr_api` — private test/helper; no records. Bind the 1007 PlateWriter revisions while retaining the item 983 fixture.

1007 (2026-11-21) documents ``PlateWriter(ngff_version=...)``. The 983
fixture keeps its original text; the change file pins that text by hash
and carries the current docstring.

## Documentation prose and fixed identifiers

Preserve the pinned validated reader `ome-zarr-models==1.8.1`, the unsupported reader version `ome-zarr==0.21.0`, source links to tag `v0.21.0`, axis names, and numerical chunk/crop dimensions. The public constructor signature is unchanged.

- `docs/notes/spacr/ome_zarr.md:233` — documentation prose; reader/version pin, source-line references and measured chunk comparison. Include with the associated documentation translation.
- `proposals/bmc_paper_items/1007_ome_zarr_ngff_06.md:160` — documentation prose; reader/version pin, source-line references and measured chunk comparison. Include with the associated documentation translation.
- `proposals/bmc_paper_items/1007_ome_zarr_ngff_06.md:185` — documentation prose; reader/version pin, source-line references and measured chunk comparison. Include with the associated documentation translation.

The synthetic benchmark used identical pixels/crop positions and recorded a 7.811 s to 1.936 s change for 1,000 crops. Raw version/source/identity reports are retained in `spacr-codex-recovery-20261121/1007a-chunk-benchmark/`.
