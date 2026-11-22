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

No application source or generated catalog changes are included. Runtime and
public API identities will be checked by exact before/after extraction during
the required output proof. No translations are authored. The companion JSON
lists the exact changed development documentation below.

Validation checkpoint: all 157 required source ratchets passed in 155.82 seconds;
154 current caption, callback-cache, nested-helper and API-extractor tests passed
in 203.57 seconds. The full current-base catalog/output byte and timing proof is
in progress. This branch is not FINAL.

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
