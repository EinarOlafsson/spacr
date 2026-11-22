# Pipeline performance rebase after pilot landing

Branch: `ws-offload-codex-work-pipeperf2`.
Base: `b9af1a272ba6f01a3aa8f6fd732e45c4cd2f0f3e` (nightly after pilot and train 9).
Source: `40c345affd06f0275c4bd6006bb76778273d9666`, including `8d1a2fd6ad` and `371d7234f6`.

The current static UI extractor supplies the entire per-file body. The cache
refactor changes its source-read expression and six outer-loop exits only.
All current paths and trailing owner callbacks remain. This includes the NGFF 0.6 owners in ome_zarr.py, image_store.py and ops_engine.py; the never-barcode owners in ops_sbs.py and ops_decode.py; and every current flat-field, painting, crop and segmentation-input owner. The complete diff from
train 5 also includes classification pixel owners, crop readers, CropPoolWriter,
engine crop owners, measure_crop and _read_measure_field, beyond the examples
listed in the assignment. The new owner callbacks participate in cache
invalidation.

No application source or generated catalog changes are included. Exact
before/after source snapshots preserve all 16,619 runtime records
and 14,961 API bodies, with zero new, revised or retired application
sources. No translations are authored. The companion JSON lists every changed
development docstring and its location.

Validation: 311 distinct tests passed, including all 157
required source ratchets and 154 focused extractor/cache tests.
Both runtime audits returned zero. The same physical current-pilot checkout
and 10,963 identical original inputs
produced 30 byte-identical data outputs,
including all 20 language catalogs, arrivals, reports, source inventories,
API history and audit diagnostics. The separate assembly cache stamp changes
only the input fingerprint that binds it to generator source; both raw hashes
are retained and the invalidation rule is unchanged.

The complete table and measured timings are in
[the output proof](reports/pipeperf2_current_output_identity_20261122.md).
[The source proof](reports/pipeperf2_current_source_preservation_20261122.json)
records the full current-loop and parser preservation checks. All operations
ran through sched at P1 with four threads and a 16 GiB RAM grant; timings are
single measurements on the shared workstation. The existing API freshness
count, 14906/14961 per locale, remains identical in the coverage reports.

## Development documentation

- `tools/build_documentation_i18n.py:1633` `_api_memo`.
- `tools/build_documentation_i18n.py:1643` `_api_memo.__enter__`.
- `tools/build_documentation_i18n.py:1650` `_api_memo.__exit__`.
- `tools/build_documentation_i18n.py:1657` `_memoized`.
- `tools/build_documentation_i18n.py:1690` `_derived_api_translation_source`.
- `tools/build_documentation_i18n.py:4304` `_public_docstrings_inputs`.
- `tools/build_documentation_i18n.py:4328` `public_docstrings`.
- `tools/build_documentation_i18n.py:4342` `_public_docstrings_uncached`.
- `tools/build_documentation_i18n.py:6156` `_assemble_api_catalog`.
- `tools/build_documentation_i18n.py:6301` `_reviewed_api_block_translations`.
- `tools/build_documentation_i18n.py:6377` `_reviewed_api_block_valid`.
- `tools/build_documentation_i18n.py:6391` `_reviewed_api_block_valid_uncached`.
- `tools/build_i18n_catalogs.py:4941` `_static_ui_rules`.
- `tools/build_i18n_catalogs.py:4963` `_static_ui_sources_of_file`.
- `tools/build_i18n_catalogs.py:8497` `_audit_language_failures`.
- `tools/build_i18n_catalogs.py:8509` `_audit_language_failures.fields`.
- `tools/build_i18n_catalogs.py:8519` `_audit_language_failures.html_tags`.
- `tools/build_i18n_catalogs.py:8785` `audit`.
- `tools/i18n_caption_arrivals.py:54` `pinned_external_sources`.
- `tools/nested_helper_docs.py:331` `_walk_with_scopes`.
- `tools/i18n_language_pool.py:1` `<module>`.
- `tools/i18n_language_pool.py:30` `Outcome`.
- `tools/i18n_language_pool.py:40` `Outcome.unwrap`.
- `tools/i18n_language_pool.py:50` `language_workers`.
- `tools/i18n_language_pool.py:62` `_outcome`.
- `tools/i18n_language_pool.py:80` `map_languages`.
- `tools/i18n_language_pool.py:106` `review_scope_counts`.
- `tools/i18n_language_pool.py:122` `runtime_review_translations`.
- `tools/i18n_language_pool.py:137` `catalog_audit_failures`.
- `tools/i18n_language_pool.py:149` `source_bound_review_snapshot`.
- `tests/test_pipeperf_current_source_owners.py:1` `<module>`.
- `tests/test_pipeperf_current_source_owners.py:15` `test_current_owner_callback_changes_invalidate_the_file_caption_cache`.
- `tests/test_pipeperf_current_source_owners.py:25` `test_current_owner_callback_changes_invalidate_the_file_caption_cache.changed`.
