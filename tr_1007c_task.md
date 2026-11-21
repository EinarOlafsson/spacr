# Translation handover: item 1007 lane C

Branch: `ws-offload-codex-work-1007c`; base: `e2a1ace77c537e4d3783f57055214040a9272741`. WORK_STATUS.md supplies the final commit SHA. The v73 inbox approves the registration writer hook and mechanical docstring arrival binding.

Home owns reviewed translations and generated catalog arrivals. Required locales: sv, de, es, zh_CN, pt, hi, ko, is, fr. This packet contains exact English source bindings, not reviewed locale evidence.

Inventory: 13 runtime/setting sources (11 UI messages and two setting sources), plus three canonical API blocks (two new blocks and the changed PlateWriter block). Preserve identifiers, y/x order, coordinate direction, and all format placeholders.

## Canonical API blocks

`spacr/ome_zarr_registration.py:1` — `spacr.ome_zarr_registration`; canonical API, records required.

SHA-256: `8fe3215f29d1656b57d7efb7332fff3683f83e6c9956c269555083a87cdf2df7`

```text
Store OPS field registration tables as OME-Zarr 0.6 transformations.
```

`spacr/ome_zarr_registration.py:67` — `spacr.ome_zarr_registration.registration_metadata`; canonical API, records required.

SHA-256: `bfc3b666ec15ae7230821a86445d1518f6050f618e7017c95c2dc5b19b853b6a`

```text
Build OME-Zarr writer metadata from one OPS phenotype table row.

Coordinates are field pixels in ``y, x`` order. The intrinsic ``physical``
system must therefore use unit spacing, as merged OPS field stores do.
The applied ``dy/dx`` rows from items 919/998 map painting pixels to
SBS pixels directly; logged source shifts are never negated again.
Placement maps field pixels to SBS well pixels: scale, rotation, then
translation. OPS selects the refined placement only when ``refined == 1``;
otherwise the stored seed is used, even when diagnostic refined values exist.
Secondary shifts map the moving secondary frame to the primary field;
applied shifts are used, rather than the diagnostic raw estimates.

Each tile-affine fit maps this field to a well frame estimated from one
SBS tile. Its distinct ``tile_global_<site>`` system identifies that fit;
the fits are alternatives and are never composed with each other. The
global affine is also retained as ``tile_global``, including when the
tile-group list is empty. Numeric affine entries and shifts are copied
without fitting or rounding. Sidecar
tables remain the source of truth. No image pixels are read or changed.

:param record: one 919/998 applied shift row (``dy/dx``), or one
    ``ops_phenotype`` row, including placement, optional
    ``secondary_dy/dx`` and ``align_tile_transforms`` (the existing JSON
    list of ``[site, count, a11, a12, a21, a22, ty, tx]`` rows).
:param axes: image axis names, containing exactly one ``y`` and ``x``;
    other axes retain identity coordinates.
:param ngff_version: must be ``"0.6"`` when registration is requested.
:returns: ``coordinate_systems`` and ``coordinate_transformations``
    keyword arguments for :func:`spacr.ome_zarr.write_ome_zarr`.
:raises OmeZarrError: on another version, invalid axes, incomplete table
    groups, duplicate tile sites or non-finite transformation parameters.
```

`spacr/image_store.py:96`, `spacr/image_store.py:121` — `spacr.image_store.PlateWriter`; canonical API, records required.

SHA-256: `926a60609ce4c36837643d89dd9e83ba82a33bd0a321d4cd23157f6a53cc0d20`

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
opening the store mid-run sees only committed fields. Each image channel
and label image uses one unsharded whole-field chunk, compressed with
Blosc LZ4 and byte shuffle.
Open (or start) the plate store and read its commit ledger.
```

## Runtime and setting sources

`spacr/ome_zarr_registration.py:132` — `ui/Active registration needs a complete finite scale, rotation and offset; repair the selected OPS placement in the registration table before exporting.`; records required.

SHA-256: `22c58a47f0f1e86295262369b3e8f22b69c9ac41f02d9ea5c08ab104b0a4359f`

```text
Active registration needs a complete finite scale, rotation and offset; repair the selected OPS placement in the registration table before exporting.
```

`spacr/ome_zarr_registration.py:108` — `ui/Registration axes must have unique names including y and x; check the exported image axes.`; records required.

SHA-256: `ca4394a81cb554c2f3178e1aa5f4e03511caeef29718754b14ac25683271a3ec`

```text
Registration axes must have unique names including y and x; check the exported image axes.
```

`spacr/ome_zarr_registration.py:22` — `ui/Registration column {column} must contain a finite number; check the OPS registration table before exporting.`; records required.

SHA-256: `b6a22673a560f6705166dcb1ab9b0a283c324b918949f9d1ac3ea4d3382b3c14`

```text
Registration column {column} must contain a finite number; check the OPS registration table before exporting.
```

`spacr/ome_zarr_registration.py:139` — `ui/Registration contains both placement and dy/dx shifts; export one source table row at a time.`; records required.

SHA-256: `0a86361d0e5d58b7ebafa06645033034c81950b272ec9a4a410856c89f6af739`

```text
Registration contains both placement and dy/dx shifts; export one source table row at a time.
```

`spacr/ome_zarr_registration.py:102` — `ui/Registration metadata requires OME-Zarr 0.6; set ome_zarr_ngff_version to '0.6' or disable ome_zarr_store_registration.`; records required.

SHA-256: `8e44670e6ee33d125b6a3b7662a2eba4d3622e8b3856fd8bca17cc7ada86a12e`

```text
Registration metadata requires OME-Zarr 0.6; set ome_zarr_ngff_version to '0.6' or disable ome_zarr_store_registration.
```

`spacr/ome_zarr_registration.py:41` — `ui/Registration needs all of {columns}; complete the OPS registration table before exporting.`; records required.

SHA-256: `a748227fafefb51fb9d70cae29b9346e1c78bc61b3557a5344f60cf640a35cf0`

```text
Registration needs all of {columns}; complete the OPS registration table before exporting.
```

`spacr/ome_zarr_registration.py:168`, `spacr/ome_zarr_registration.py:173`, `spacr/ome_zarr_registration.py:179` — `ui/Tile registration must be a JSON list of eight-value rows; check align_tile_transforms in the OPS table.`; records required.

SHA-256: `ef86f9f811ff11cc42c8043f153ddaa531b383c852c064b689ce1fbf375fe587`

```text
Tile registration must be a JSON list of eight-value rows; check align_tile_transforms in the OPS table.
```

`spacr/ome_zarr_registration.py:184` — `ui/Tile registration sites must be unique integers; check align_tile_transforms in the OPS table.`; records required.

SHA-256: `b2b4a3122912c96fc402c000a0a320f419040170f2b8996f38751d3ae6c35210`

```text
Tile registration sites must be unique integers; check align_tile_transforms in the OPS table.
```

`spacr/ops_settings.py:1754` — `setting_labels/ome_zarr_store_registration`; records required.

SHA-256: `453510eec04c621197f426116939975ced92cc5bbc46461857030ca4eeedfe68`

```text
Ome zarr store registration
```

`spacr/ops_settings.py:1755` — `setting_tooltips/ome_zarr_store_registration`; records required.

SHA-256: `6c0e8086434f61282076493aa9675f68fedf17a6b007e168b73009e00ddb9588`

```text
Store the applied phenotype-to-SBS placement, secondary channel shifts and tile affine fits as named coordinate transformations in merged OME-Zarr field images. Requires ome_zarr_ngff_version 0.6. Pixels are unchanged; OPS sidecar tables remain the source of truth. Default False. Alpha.
```

`spacr/image_store.py:215` — `ui/Field {field} has no registration table row; complete the OPS registration before exporting.`; records required.

SHA-256: `211843aa931379c1cb6f804cece3aa33ba4c1b796eaa93bc968138e599630ea5`

```text
Field {field} has no registration table row; complete the OPS registration before exporting.
```

`spacr/image_store.py:138` — `ui/Stored registration requires OME-Zarr 0.6 and pixel coordinates; use ngff_version='0.6' and spacing=None, or disable registration.`; records required.

SHA-256: `3f03269ba74cd7d52410cab2a7aea45424b2f6f43a9f46e2c51cc6aae5ed0288`

```text
Stored registration requires OME-Zarr 0.6 and pixel coordinates; use ngff_version='0.6' and spacing=None, or disable registration.
```

`spacr/image_store.py:163` — `ui/This plate store uses another registration mode; keep ome_zarr_store_registration unchanged or write to a new folder.`; records required.

SHA-256: `dad754be692feed6aee20165a522dee12653b13b0478118e420f43a0327de5e4`

```text
This plate store uses another registration mode; keep ome_zarr_store_registration unchanged or write to a new folder.
```

## Source contracts and coordinate semantics

Canonical extractor verification: `1007c-sources-with-hook.json` in the recovery directory. The English runtime fixture is `tests/data/release_contracts/1007c_ome_zarr_registration_runtime_arrivals_2026-11-21.json`. New API blocks use `1007c_ome_zarr_registration_api_arrivals_2026-11-21.json`; the PlateWriter revision uses `1007c_ome_zarr_registration_api_changes_2026-11-21.json`, hash-chained after the retained 1007 A and 983 fixtures.

The engine passes the FULL original `by_site` registration row, including the bare `refined` flag and `align_*` columns. The filtered persistence payload omits those fields and cannot replace it. The shared OPS selector uses refined placement only when `refined == 1`. Applied dy/dx values retain their moving-to-reference direction. The global affine and each alternative tile affine are retained separately, with float64 values preserved.

Registration requires NGFF 0.6 and pixel coordinates (spacing=None). Image and label groups receive the same frames in their respective axis dimensions. Validation precedes field replacement. A committed store keeps one registration mode; the default disabled path retains its ledger representation and pixel chunks.

## Private and test docstrings: no records

- `spacr/ome_zarr_registration.py:14` `_number` — private helper or test; no records. Copy one finite table number without decimal rounding.
- `spacr/ome_zarr_registration.py:29` `_missing` — private helper or test; no records. Recognize empty optional cells in OPS dataframe records.
- `spacr/ome_zarr_registration.py:34` `_values` — private helper or test; no records. Read a complete optional group; refuse partially populated table rows.
- `spacr/ome_zarr_registration.py:49` `_vector` — private helper or test; no records. Embed a row/column vector while leaving the other axes unchanged.
- `spacr/ome_zarr_registration.py:56` `_affine` — private helper or test; no records. Embed an exact two-dimensional table affine in the image axes.
- `tests/test_ome_zarr_registration_1007.py:1` `<module>` — private helper or test; no records. T7: exact OPS registration metadata, coordinate direction and chunk identity.
- `tests/test_ome_zarr_registration_1007.py:15` `_bits` — private helper or test; no records. Expose float64 bits, including the sign bit of zero.
- `tests/test_ome_zarr_registration_1007.py:20` `_chunk_hashes` — private helper or test; no records. Hash actual chunk files independently of their metadata.
- `tests/test_ome_zarr_registration_1007.py:26` `test_t7_shifts_and_tile_affines_survive_real_store_round_trip` — private helper or test; no records. Source float64 entries survive JSON storage and the lane A reader exactly.
- `tests/test_ome_zarr_registration_1007.py:65` `test_scale_rotation_translation_map_field_pixels_to_sbs` — private helper or test; no records. Lane A composes the same forward map as the OPS placement table.
- `tests/test_ome_zarr_registration_1007.py:84` `test_secondary_uses_applied_shift_with_the_moving_to_primary_direction` — private helper or test; no records. Diagnostic raw estimates cannot replace the applied secondary shift.
- `tests/test_ome_zarr_registration_1007.py:98` `test_refined_placement_precedes_seed_without_composing_them` — private helper or test; no records. Only an accepted refinement replaces the seed, matching the OPS selector.
- `tests/test_ome_zarr_registration_1007.py:108` `test_one_tile_affine_embeds_without_changing_other_axes` — private helper or test; no records. All six affine table values retain their row/column order.
- `tests/test_ome_zarr_registration_1007.py:126` `test_invalid_registration_is_refused_before_writing` — private helper or test; no records. Bad table metadata fails with an actionable registration error.
- `tests/test_ome_zarr_registration_1007.py:133` `test_ambiguous_or_missing_spatial_axes_are_refused` — private helper or test; no records. A transform cannot silently use another spatial axis order.
- `tests/test_ome_zarr_registration_1007.py:139` `test_registration_requires_opt_in_version_06` — private helper or test; no records. A version mismatch tells the user which settings to change.
- `tests/test_ome_zarr_registration_1007.py:145` `test_empty_row_has_no_invented_registration` — private helper or test; no records. Absent source tables produce no additional coordinate systems or fits.
- `tests/test_ome_zarr_registration_1007.py:150` `test_nullable_dataframe_columns_do_not_invent_registration` — private helper or test; no records. Optional all-null table groups remain absent after dataframe conversion.
- `tests/test_ome_zarr_registration_1007.py:160` `test_t7_applied_919_998_shift_rows_round_trip_without_sign_changes` — private helper or test; no records. Use the applied dy/dx table columns, rather than the opposite logged sign.
- `tests/test_ome_zarr_registration_1007.py:175` `test_ambiguous_shift_and_placement_tables_are_refused` — private helper or test; no records. Independent source fits cannot be silently composed or discarded.
- `tests/test_ome_zarr_registration_1007.py:184` `test_inactive_refinement_diagnostics_do_not_override_or_break_the_seed` — private helper or test; no records. Partial or invalid rejected diagnostics remain unused by OPS and metadata.
- `tests/test_ome_zarr_registration_1007.py:194` `test_invalid_active_refinement_is_refused_without_using_the_seed` — private helper or test; no records. An accepted but incomplete or non-finite placement cannot silently fall back.
- `tests/test_ome_zarr_registration_1007.py:202` `test_registration_runtime_sources_match_their_arrival_contract` — private helper or test; no records. Every registration diagnostic and setting source has an exact English binding.
- `tests/test_ome_zarr_registration_1007.py:232` `test_global_affine_survives_empty_or_populated_tile_groups` — private helper or test; no records. The stored global fallback and optional tile fit each retain their exact values.
- `tests/test_ome_zarr_registration_1007.py:256` `test_plate_registration_and_labels_keep_the_same_chunks` — private helper or test; no records. The optional HCS hook adds matching image/label frames without pixel changes.
- `tests/test_ome_zarr_registration_1007.py:282` `test_plate_registration_refuses_a_version_mismatch_before_creation` — private helper or test; no records. Registration cannot silently disappear into a default 0.5 plate store.
- `tests/test_ome_zarr_registration_1007.py:293` `test_resumed_plate_cannot_silently_change_registration_mode` — private helper or test; no records. A completed field cannot be skipped under changed registration options.
- `tests/test_ome_zarr_registration_1007.py:308` `test_bad_registration_cannot_delete_a_committed_field` — private helper or test; no records. Table validation happens before the existing field is removed.
- `tests/test_ome_zarr_registration_1007.py:324` `test_registration_rejects_non_pixel_spacing_before_creation` — private helper or test; no records. Pixel-coordinate tables cannot be attached to micrometre coordinates directly.
- `tests/test_docstring_correctness.py:3870` `_validated_ome_zarr_registration_1007c_callables` — private helper or test; no records. Validate the exact new registration API before projecting the old pin.
- `tests/test_documentation_i18n.py:457` `_current_1007_ome_zarr_api` — private helper or test; no records. Bind the 1007 A/C PlateWriter revisions while retaining the item 983 fixture.

1007 (2026-11-21) documents ``PlateWriter(ngff_version=...)``. The 983
fixture keeps its original text; the change file pins that text by hash
and carries the current docstring.
