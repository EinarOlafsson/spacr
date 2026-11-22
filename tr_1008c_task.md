# Item 1008 Part C: per-cell cytoskeleton translation handover

Source branch: `ws-offload-codex-work-1008c`.
Base source: `e2a1ace77c537e4d3783f57055214040a9272741`.
The implementation commit containing this handover is reported with the final task status.
Source file: `spacr/measure.py`; SHA256 `cdf19ea56da054d4783b71bc6943286111289ca502ca626ca372ab738f670bd5`.

This task adds one public API callable, two distinct runtime templates and one hidden compatibility setting with a label and tooltip. Translation records and catalog rebuilds remain deferred to Home under WORK_TASKS v74. No records or catalogs were fabricated.

## Public API source

`spacr.measure.measure_cytoskeleton` at `spacr/measure.py:801`.
Admission: canonical API, verified by `public_docstrings()` and the exact arrival fixture.
SHA256: `6a4b67e1595eeb306dd37f0abf2503367d6d629d36f0482b72590d05ce34c259`.

```rst
Measure skeleton length and branch points in each cell and selected channel.

:param labels: Two-dimensional cell label mask; zero is background.
:param image: Intensity image with channels last, or a single 2-D plane.
:param channels: Channel indices, names, or a comma-separated selection.
    ``None`` selects every channel. Repeated selections are measured once.
:param channel_names: One name per image channel, required when selecting
    channels by name. Name matching ignores case and surrounding spaces.
:returns: One row per positive cell label in ascending order, with
    ``label``, ``channel_<c>_skeleton_length`` and
    ``channel_<c>_skeleton_branch_points`` for each selected index.
    Background-only and single-positive-pixel objects receive zeros.
    The caller adds the ``cell_`` prefix and uses only the cell mask.
:raises ValueError: When a selected channel is absent or invalid.

Each object is measured by :func:`_analyze_cytoskeleton` in its bounding
box enlarged by 24 pixels and clipped to the field. Other objects are
zeroed by the same helper before local thresholding.
```

Canonical translation blocks:

Block 0: SHA256 `2b9a9acb0390574e62f952a2e20e27b64011e22ce4664e9a9acb152d9c4c6b84`.

```rst
Measure skeleton length and branch points in each cell and selected channel.
```

Block 1: SHA256 `abde3f9d8a6047023c28899946c5e737e828f67b2431de6203dc3fb9bc87e35c`.

```rst
Two-dimensional cell label mask; zero is background.
```

Block 2: SHA256 `96cf63c50f99965e0c01482896df4ba95803a16b8ac5a006acbe1837b12616bc`.

```rst
Intensity image with channels last, or a single 2-D plane.
```

Block 3: SHA256 `ea113809d3bde0fec976ded9800863cc954f675501de580e16e89a9d89ad2b65`.

```rst
Channel indices, names, or a comma-separated selection. ``None`` selects every channel. Repeated selections are measured once.
```

Block 4: SHA256 `ca5580ff03e3e1a214bea2abb1b14d5c2d421eeb8a8fe851a1379a507ea16dc5`.

```rst
One name per image channel, required when selecting channels by name. Name matching ignores case and surrounding spaces.
```

Block 5: SHA256 `40892176fc864e77000b47e3cc4edbffbc446279e5a42b15171f6aa3931a7d03`.

```rst
One row per positive cell label in ascending order, with ``label``, ``channel_<c>_skeleton_length`` and ``channel_<c>_skeleton_branch_points`` for each selected index. Background-only and single-positive-pixel objects receive zeros. The caller adds the ``cell_`` prefix and uses only the cell mask.
```

Block 6: SHA256 `7da0a1efb73d4782a8405b3bf293c88db8218467c3f44e10cfea47d7d4233738`.

```rst
When a selected channel is absent or invalid.
```

Block 7: SHA256 `fdcb63845bd555f4cf777d4c48bc40a8042066d19f40273e1f379dc8a1977e0d`.

```rst
Each object is measured by :func:`_analyze_cytoskeleton` in its bounding box enlarged by 24 pixels and clipped to the field. Other objects are zeroed by the same helper before local thresholding.
```

## Runtime sources

1. Locations: `spacr/measure.py:856`, `spacr/measure.py:852`.
Admission: canonical `ui`, verified by `canonical_sources()`.
SHA256: `b818e527c1cf714b3f91b927ebe5280a39054b6b92b2b42e12d22e532f775f4e`.

```text
Cytoskeleton channel {channel!r} is not available. Choose an image channel index or a name from channel_names.
```

2. Locations: `spacr/measure.py:831`.
Admission: canonical `ui`, verified by `canonical_sources()`.
SHA256: `fc71195927bc4f63d4c2d32bbf2531e6bc760b534d931543bc56288ec719d828`.

```text
Cytoskeleton channel_names must give one distinct name per image channel.
```

3. Locations: `spacr/settings.py:964`.
Admission: canonical `setting_labels`, verified by `canonical_sources()`.
SHA256: `35b526668725250e42ed5448c5158f5a993bffa67d21e7e31c60ca9797ebf797`.

```text
Cytoskeleton object support
```

4. Locations: `spacr/settings.py:1013`.
Admission: canonical `setting_tooltips`, verified by `canonical_sources()`.
SHA256: `c8fc806ef8018c603d8e552ca639b20fdb1a16cdda70cb861777dee148ecfff6`.

```text
Restrict cytoskeleton skeleton pixels to the object mask. Settings saved before schema 958 retain False to reproduce earlier counts. Default True.
```

## Internal and test documentation

No records for private, nested or test callables. Existing source admission rules exclude these docstrings from the public API catalog.

- `_analyze_cytoskeleton`: `spacr/measure.py:740`; none: private API.

- `_create_dataframe`: `spacr/measure.py:2121`; none: private API.

- `_image_fibres`: `spacr/measure.py:17949`; none: private API.

- `_image_level_features`: `spacr/measure.py:18036`; none: private API.

- `spacr.settings._fold_older_defaults`: `spacr/settings.py:2625`; none: private API.

- `spacr.qt.widgets.qc_field_browser._overview_measurement`: `spacr/qt/widgets/qc_field_browser.py:3459`; none: private API.
- `_legacy_dataframe`: `tests/test_cytoskeleton_radial_1008.py:13`; no records, test callable.
- `test_radial_dataframe_preserves_values_columns_identity_and_order`: `tests/test_cytoskeleton_radial_1008.py:30`; no records, test callable.
- `test_radial_dataframe_matches_the_934_fixture`: `tests/test_cytoskeleton_radial_1008.py:36`; no records, test callable.
- `_skeleton_objects`: `tests/test_cytoskeleton_radial_1008.py:47`; no records, test callable.
- `test_skeleton_matches_whole_field_per_object_and_channel`: `tests/test_cytoskeleton_radial_1008.py:61`; no records, test callable.
- `test_skeleton_channel_names_indices_and_all_channels_agree`: `tests/test_cytoskeleton_radial_1008.py:76`; no records, test callable.
- `test_skeleton_unknown_name_uses_translated_actionable_message`: `tests/test_cytoskeleton_radial_1008.py:87`; no records, test callable.
- `test_skeleton_invalid_channel_index_stops`: `tests/test_cytoskeleton_radial_1008.py:106`; no records, test callable.
- `test_skeleton_empty_and_single_signal_objects_keep_their_rows`: `tests/test_cytoskeleton_radial_1008.py:113`; no records, test callable.
- `test_skeleton_public_arrival_contract_equals_source`: `tests/test_cytoskeleton_radial_1008.py:128`; no records, test callable.
- `test_corrected_skeleton_exact_crop_parity_in_ninety_cases`: `tests/test_cytoskeleton_radial_1008.py:142`; no records, test callable.
- `test_skeleton_never_skeletonizes_pixels_outside_the_object`: `tests/test_cytoskeleton_radial_1008.py:176`; no records, test callable.
- `translated`: `tests/test_cytoskeleton_radial_1008.py:93`; no records, test callable.
- `recorded`: `tests/test_cytoskeleton_radial_1008.py:185`; no records, test callable.
- `test_public_docstrings_matches_reviewed_visible_coverage`: `tests/test_api_i18n_extractor.py:239`; no records, test callable.
- `test_documentation_api_catalog_inventory_and_hashes_are_current`: `tests/test_documentation_i18n.py:3585`; no records, test callable.
- `_validated_cytoskeleton_1008c_callables`: `tests/test_docstring_correctness.py:3865`; no records, test callable.
- `test_public_callable_inventory_is_source_derived_not_docstring_derived`: `tests/test_docstring_correctness.py:4100`; no records, test callable.
- `test_callable_boundary_is_cross_checked_with_i18n_extractor`: `tests/test_docstring_correctness.py:6401`; no records, test callable.
- `_saved_overview_settings`: `tests/test_cytoskeleton_migration_1008.py:11`; no records, test callable.
- `test_old_saved_overview_files_keep_legacy_counts`: `tests/test_cytoskeleton_migration_1008.py:27`; no records, test callable.
- `test_current_files_and_explicit_choices_use_the_requested_semantics`: `tests/test_cytoskeleton_migration_1008.py:37`; no records, test callable.
- `test_saved_settings_reach_the_actual_overview_measurement`: `tests/test_cytoskeleton_migration_1008.py:48`; no records, test callable.
- `test_hidden_compatibility_value_survives_overview_import_and_collect`: `tests/test_cytoskeleton_migration_1008.py:76`; no records, test callable.
- `test_schema958_runtime_arrivals_are_exact_canonical_sources`: `tests/test_cytoskeleton_migration_1008.py:92`; no records, test callable.
- `raw_planes`: `tests/test_cytoskeleton_migration_1008.py:63`; no records, test callable.

## Reproducibility and release text

Maintainer option A is approved in WORK_TASKS v77. Schema 958 records the corrected object-support default. The hidden Overview compatibility setting keeps the old value False for saved files before 958; new files use True. The new per-cell API always uses corrected object support. One shared helper implements both modes.

`docs/source/reproducibility_manifests.rst:178`; SHA256 `23987065f9a364649bf804ea09560a07bdcff871afb6dc7218aba44bfae7b17f`.

```text
- 958 (item 1008): ``cytoskeleton_object_support`` is now ``True``.
  Measure no longer counts cytoskeleton skeleton pixels outside the object.
  Older Overview settings retain ``False`` and reproduce their whole-image
  fibre counts. New per-cell cytoskeleton measurements use the corrected
  object support and 24-pixel padded crops.
```

`docs/notes/release_notes_1.5.1.8.md:37`; SHA256 `6900fe5362dde0c23cc7c087550541a2b3490bf0847ae8edf2c0f9c8444a9d47`.

```text
- Measure no longer counts cytoskeleton skeleton pixels outside the object.
```


## Validation evidence

The focused scientific and migration suite passed 182 tests (exit 0), including 90 exact crop/whole-field comparisons, the outside-object regression, actual saved-file migration, old and current Overview counts, hidden import/collect, and P8. Final functional, source/API and ratchet results are reported with the final task status. Canonical blocks and hashes are retained at `/home/carruthers/Documents/repo/spacr-codex-recovery-20261121/1008c/sources.json`.

Final validation: scientific 182 passed, corrected source/migration 12 passed, existing Measure/Overview 132 passed, all exit 0. Full items 1,260 passed, OPS 1,240 passed, settings 614 passed, all actual exits 0.

Full ratchets retain their actual exit 1 and original accepted=false report: 4,070 passed, 76 skipped, 3 report-only incompatibilities, and exactly two deferred missing-Home-tooltip failures. Both failures name only `cytoskeleton_object_support` as absent from the generated English tooltip catalog. The parent independently validated these under the v74 pilot deferral. No catalog, global pin, awaiting list or workflow was changed.

Exactly 30 catalog nodes were deselected in the actual full runs; the two following failures were selected and failed. They were not deselected:

- `tests.qt.test_external_i18n_catalogs::test_runtime_source_inventory_is_complete_before_optional_module_imports`
- `tests.qt.test_external_i18n_catalogs::test_runtime_source_inventory_is_stable_after_runctx_import`

The 30 actual deselections and every raw outcome are in `tr_1008c_sources.json` under `validation`. The raw generic-runner reports remain unchanged at `/home/carruthers/Documents/repo/spacr-codex-recovery-20261121/1008c/v77-full-functional/pilot-gate-result.json` and `/home/carruthers/Documents/repo/spacr-codex-recovery-20261121/1008c/v77-full-ratchets/pilot-gate-result.json`.
