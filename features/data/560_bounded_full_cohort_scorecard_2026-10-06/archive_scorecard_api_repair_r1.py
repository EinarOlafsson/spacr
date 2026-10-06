from pathlib import Path
import gzip
import hashlib
import json
import shutil

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
acceptance = json.loads((scratch / 'scorecard-api-repair-full-cohort-r1.json').read_text())
preservation = json.loads((scratch / 'scorecard-catalog-preservation-proof-r1.json').read_text())
assert acceptance['normal_repaired_full_17074_cell_scorecard_two_threads_passed']
assert len(preservation['API']) == len(preservation['runtime']) == 10
assert all(r['all_complete_records_preserved'] == 13181 for r in preservation['API'].values())
assert all(r['current_rows'] == 9857 and r['all_other_complete_rows_and_hashes_preserved'] == 9856
           for r in preservation['runtime'].values())
assert 'verified API catalogs: languages=9 symbols=13181' in (scratch / 'scorecard-api-strict-audit-r1.log').read_text()
assert 'build succeeded.' in (scratch / 'scorecard-docs-all-r1.log').read_text()
assert '11 passed' in (scratch / 'scorecard-docstring-tests-r1.log').read_text()
assert '3 passed' in (scratch / 'cell-dino-foundation-picker-tests-r1.log').read_text()
assert 'verified' in (scratch / 'cell-dino-runtime-normal-refresh-r1.log').read_text().lower()
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
for path, sha in acceptance['current_source_sha256'].items():
    assert digest(Path(path)) == sha
destination = Path('features/data/560_bounded_full_cohort_scorecard_2026-10-06')
destination.mkdir(exist_ok=False)
for name in ('scorecard-api-repair-full-cohort-r1.json', 'scorecard-catalog-preservation-proof-r1.json',
             'scorecard-api-runtime-source-delta-r1.json', 'cell-dino-runtime-review-input-proof-r1.json',
             'verify_repaired_full_cohort_scorecard_r1.py', 'check_scorecard_catalog_preservation_r1.py',
             'refresh_cell_dino_runtime_review_r1.py', 'preserve_cell_dino_review_order_r1.py',
             Path(__file__).name):
    shutil.copyfile(scratch / name, destination / name)
for name in ('scorecard-api-repair-full-cohort-r1.log', 'scorecard-api-strict-audit-r1.log',
             'cell-dino-runtime-review-admission-r1.log', 'cell-dino-runtime-review-admission-r2.log',
             'cell-dino-runtime-normal-refresh-r1.log', 'scorecard-docs-all-r1.log',
             'scorecard-docstring-tests-r1.log', 'cell-dino-foundation-picker-tests-r1.log',
             'scorecard-consumer-map-r1.log', 'scorecard-settings-flow-r1.log',
             'scorecard-catalog-preservation-r1.log'):
    (destination / (name + '.gz')).write_bytes(gzip.compress((scratch / name).read_bytes(), mtime=0))
receipt = {
    'item': 560, 'coordinated_source_parent': 'ad7ffb18c4292b25dbf960fa2c85fd60626d763b',
    'normal_two_thread_full_17074_cell_scorecard_terminal_exit_code': 0,
    'full_original_GPU_feature_cohort_and_human_labels_preserved': True,
    'all_prior_full_dense_metrics_match_atol_1e_12': True,
    'normal_model_zoo_encoder_entry': acceptance['normal_model_zoo_encoder_entry'],
    'elapsed_seconds': acceptance['elapsed_seconds'],
    'peak_process_RSS_KiB': acceptance['peak_process_RSS_KiB'],
    'hard_memory_cap_GiB': 4, 'scorecard_query_block_rows': 256,
    'full_rankings_self_exclusion_stable_ties_singleton_handling_and_signatures_preserved': True,
    'no_subsampled_cohort_or_top_k_mAP_approximation': True,
    'bare_numpy_large_product_library_failure_not_claimed_fixed': True,
    'focused_scorecard_foundation_and_zoo_tests_observed_terminal_exit_code': 0,
    'focused_scorecard_foundation_and_zoo_tests_passed': 22,
    'focused_zoo_weight_test_skipped': 1,
    'skip_reason': 'Existing test reports pretrained weights are not cached on this machine. Main environment lacks timm; the real full-cohort verifier checks actual cached weights in the private encoder environment.',
    'focused_test_output_provenance': 'Exact terminal tool handle 53583; no reconstructed raw log claimed',
    'docstring_tests_passed': 11, 'foundation_picker_tests_passed': 3,
    'current_Cell_DINO_unsupported_caption_in_nine_languages': True,
    'no_Cell_DINO_checkpoint_loader_or_GPU_support_claim': True,
    'Cell_DINO_primary_source_evidence': '560_Cell_DINO_upstream_availability_2026-10-06_r2.json',
    'normal_nine_language_runtime_build_and_strict_API_audit_passed': True,
    'complete_unchanged_API_records_each_ten_catalogs': 13181,
    'all_other_runtime_rows_and_hashes_preserved_each_ten_catalogs': 9856,
    'runtime_replaced_source_rows_each_catalog': 1,
    'strict_full_all_language_Sphinx_build_passed': True,
    'normal_consumer_map_and_settings_flow_regenerated': True,
    'Mask_Measure_Home_Conda_media_unchanged': True,
    'new_checkpoint_actual_nightly_deployment_verified': False,
    'review_method': 'Direct Codex AI technical review; no native-speaker signoff',
    'earlier_review_admission_failure_retained': 'r1 used an unsupported helper keyword before changing any inputs; r2 succeeds',
    'current_source_sha256': acceptance['current_source_sha256'],
    'no_new_GPU_job_dependency_environment_change_or_protected_job_interference': True,
    'Home_retains_CPU_CI_Qt_and_remaining_embedding_feature_source_ownership': True,
    'artifacts': {str(path): {'sha256': digest(path), 'bytes': path.stat().st_size}
                  for path in sorted(destination.iterdir())},
}
Path('features/data/560_bounded_full_cohort_scorecard_2026-10-06.json').write_text(json.dumps(receipt, indent=2) + '\n')
note = '\n2026-10-06 workstation bounded full-cohort scorecard repair acceptance: the coordinated existing _retrieval_scorecard now ranks 256 query rows at a time against every eligible original crop. The normal two-thread _scored_encoder_entry completes all 17,074 original human-labelled DINOv2 cells under the unchanged 4 GiB cap in 23.2965 seconds, with peak process RSS 880,100 KiB. Every original full dense one-thread kNN/mAP/precision/chance metric matches to 1e-12; full rankings, self exclusion, stable ties, singleton handling and signature are preserved. This resolves the observed application scorecard crash without sampling or top-k AP approximation, and does not claim to repair the underlying bare NumPy large-product failure. Twenty-two focused foundation/scorecard/zoo tests pass with one existing main-environment weight-cache skip; the actual full-cohort verifier independently hashes the real cached weights. Eleven docstring and three focused picker tests pass. Cell-DINO refusal and picker tooltip now truthfully say unsupported by this version and point to the official checkpoint request, with the tooltip regenerated normally in all nine languages. No checkpoint loader or Cell-DINO GPU support is claimed. The two changed helper docstrings are private: normal English generation and strict all-nine API audit retain all 13,181 complete API records byte-for-byte in every one of ten catalogs. Exactly one runtime source is replaced in each catalog; all other 9,856 complete rows and source hashes remain identical, and all unrelated reviewed inputs/order are preserved. The normal consumer map/settings flow and full strict all-language Sphinx build pass. Receipt 560_bounded_full_cohort_scorecard_2026-10-06.json archives source hashes, real full-cohort result, complete-record proofs, terminal logs, generation and replay scripts, including the failed private review-admission attempt. Current source hashes supersede the prior source freeze only for the two scoped embeddings files; historical GPU receipts remain valid for their exact earlier source. Mask 07, corrected Measure 08, Home 05 and Conda 02 media are unchanged. Actual deployment of this new source checkpoint remains pending. Home may integrate this tested source checkpoint, retaining checkpoint loading, remaining embedding GUI/pipeline/screen-wide feature source and final CPU CI/coverage/serial Qt; workstation retains GPU/API/docs/translations/tutorials. No protected GPU process was touched, no dependency changed and no guard weakened.\n'
note += 'Item 560 remains OPEN for the remaining foundation-backbone human-label comparisons and official Cell-DINO loading. Item 615 remains OPEN for its remaining requested work; this receipt closes the scoped scorecard/caption refresh only.\n'
for name in ('features/325_two_sessions_one_repo_working_protocol.temp',
             'features/future/560_foundation_model_embeddings.txt',
             'features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt'):
    with Path(name).open('a') as stream:
        stream.write(note)
print('PASS: bounded full-cohort API scorecard and normal nine-language documentation acceptance archived.', flush=True)
