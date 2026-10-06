from pathlib import Path
import gzip
import hashlib
import json
import re
import shutil
import subprocess

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
def read(name):
    return json.loads((scratch / name).read_text())
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def passed_count(name):
    text = (scratch / name).read_text()
    assert not re.search(r'\d+ failed', text)
    match = re.search(r'(\d+) passed', text)
    assert match
    return int(match[1])
replay = read('foundation-api-full-cohort-replay-r3.json')
preservation = read('foundation-api-catalog-preservation-r3.json')
guides = read('foundation-puncta-guide-preservation-r1.json')
review = read('foundation-api-reviewed-inputs-r3.json')
generation = read('foundation-current-API-generation-r1.json')
assert replay['passed'] and len(replay['comparisons']) == 12
assert replay['source_sha256'] == generation['source_sha256'] == digest('spacr/embeddings.py')
assert len(preservation['API']) == len(preservation['runtime']) == 10
assert len(guides) == 9 and all(row['prior_complete_messages_exact'] == 2979 for row in guides.values())
assert all(len(row['API']) == 7 for row in review['languages'].values())
for suffix in ('A', 'B', 'C'):
    assert 'verified API catalogs: languages=3 symbols=13181' in (scratch / ('foundation-final-API-audit-' + suffix + '-r1.log')).read_text()
assert 'build succeeded.' in (scratch / 'foundation-api-docs-all-r2.log').read_text()
guide_text = (scratch / 'foundation-guides-render-r1.log').read_text()
assert all(language + ': sphinx exit 0' in guide_text for language in guides)
browser = read('foundation-local-browser-r3/acceptance.json')
assert browser['passed'] and len(browser['source_current_API_panels']) == 9
assert len(browser['source_current_puncta_guide_pages']) == 9
counts = {name: passed_count(name) for name in (
    'foundation-final-source-focused-tests-r1.log', 'foundation-final-docstring-guards-r1.log',
    'foundation-catalog-scoped-tests-r3.log', 'foundation-readme-presentation-tests-r2.log',
    'foundation-help-search-tests-r1.log', 'foundation-guide-tests-r1.log',
)}
destination = Path('features/data/560_foundation_provenance_and_exact_chance_API_2026-10-06')
destination.mkdir(exist_ok=False)
json_names = (
    'foundation-api-source-delta-r1.json', 'foundation-api-full-cohort-replay-r1.json',
    'foundation-api-full-cohort-replay-r2.json', 'foundation-api-full-cohort-replay-r3.json',
    'foundation-api-reviewed-inputs-r1.json', 'foundation-api-reviewed-inputs-r2.json',
    'foundation-api-reviewed-inputs-r3.json', 'foundation-api-catalog-preservation-r2.json',
    'foundation-api-catalog-preservation-r3.json', 'foundation-runtime-preservation-r1.json',
    'foundation-current-API-generation-r1.json', 'current-readme-measurement-r1.json',
    'foundation-guide-audit-r1.json', 'foundation-puncta-guide-import-r1.json',
    'foundation-puncta-guide-preservation-r1.json',
)
script_names = (
    'run_foundation_api_tool_r1.py', 'run_foundation_api_tool_r2.py',
    'prepare_foundation_api_refresh_r1.py', 'verify_foundation_api_full_cohorts_r1.py',
    'verify_foundation_api_full_cohorts_r2.py', 'verify_foundation_api_full_cohorts_r3.py',
    'review_foundation_api_translations_r1.py', 'refine_foundation_reviewed_inputs_r1.py',
    'review_complete_encoder_description_r1.py', 'review_complete_encoder_description_r2.py',
    'generate_current_encoder_API_r1.py', 'check_foundation_catalog_preservation_r2.py',
    'check_foundation_catalog_preservation_r3.py', 'refresh_current_readme_evidence_r1.py',
    'prepare_puncta_guide_worklists_r1.py', 'import_puncta_guide_reviews_r1.py',
    'check_puncta_guide_preservation_r1.py', 'verify_foundation_local_browser_r1.py',
    'verify_foundation_local_browser_r2.py', 'verify_foundation_local_browser_r3.py', 'build_foundation_api_docs_r1.py',
    'build_foundation_api_docs_r2.py', Path(__file__).name,
)
log_names = (
    'foundation-api-preparation-r1.log', 'foundation-api-focused-tests-r1.log',
    'foundation-api-focused-tests-r2.log', 'foundation-api-full-cohort-replay-r1.log',
    'foundation-api-full-cohort-replay-r2.log', 'foundation-api-full-cohort-replay-r3.log',
    'foundation-api-reviewed-inputs-r1.log', 'foundation-api-reviewed-inputs-r2.log',
    'foundation-api-English-source-generation-r1.log', 'foundation-api-English-source-generation-r2.log',
    'foundation-reviewed-input-semantic-refinement-r1.log',
    'foundation-API-normal-refresh-r1.log', 'foundation-API-normal-refresh-r2.log',
    'foundation-API-normal-refresh-r3.log', 'foundation-current-API-generation-r1.log',
    'foundation-final-API-audit-A-r1.log', 'foundation-final-API-audit-B-r1.log',
    'foundation-final-API-audit-C-r1.log', 'foundation-runtime-normal-refresh-r1.log',
    'foundation-runtime-integrated-audit-r1.log', 'foundation-runtime-scoped-tests-r1.log',
    'foundation-api-complete-description-review-r1.log', 'foundation-api-complete-description-review-r2.log',
    'foundation-integrated-api-tests-r1.log', 'foundation-final-source-focused-tests-r1.log',
    'foundation-docstring-guards-r1.log', 'foundation-docstring-guards-r2.log',
    'foundation-final-docstring-guards-r1.log', 'foundation-consumer-map-integrated-r1.log',
    'foundation-settings-flow-integrated-r1.log', 'foundation-api-catalog-preservation-r2.log',
    'foundation-api-catalog-preservation-r3.log', 'foundation-catalog-scoped-tests-r1.log',
    'foundation-catalog-scoped-tests-r2.log', 'foundation-catalog-scoped-tests-r3.log',
    'foundation-readme-editorial-r1.log', 'foundation-readme-editorial-r2.log',
    'current-readme-measurement-r1.log', 'foundation-readme-normal-refresh-r1.log',
    'foundation-readme-presentation-tests-r2.log', 'foundation-help-search-generation-r1.log',
    'foundation-help-search-tests-r1.log', 'foundation-make-masks-reference-check-r1.log',
    'foundation-make-masks-reference-generation-r1.log', 'foundation-guide-extract-r1.log',
    'foundation-guide-update-r1.log', 'foundation-puncta-guide-worklists-r1.log',
    'foundation-puncta-guide-import-r1.log', 'foundation-puncta-guide-preservation-r1.log',
    'foundation-guide-audit-r1.log', 'foundation-guide-tests-r1.log',
    'foundation-api-docs-all-r1.log', 'foundation-api-docs-all-r2.log',
    'foundation-guides-render-r1.log', 'foundation-local-browser-r1.log',
    'foundation-local-browser-r2.log', 'foundation-local-browser-r3.log', 'foundation-instruction-index-r1.log',
)
for name in json_names + script_names:
    shutil.copyfile(scratch / name, destination / name)
for name in log_names:
    raw = (scratch / name).read_bytes()
    target = destination / (name + '.gz')
    target.write_bytes(gzip.compress(raw, mtime=0))
    assert gzip.decompress(target.read_bytes()) == raw
for folder in ('foundation-puncta-guide-worklists-r1', 'foundation-local-browser-r3'):
    shutil.copytree(scratch / folder, destination / folder)
for label, path in (
    ('original-embeddings.py.gz', scratch / 'foundation-api-refresh-baseline-r1/embeddings.py'),
    ('final-embeddings.py.gz', Path('spacr/embeddings.py')),
):
    (destination / label).write_bytes(gzip.compress(path.read_bytes(), mtime=0))
receipt = {
    'item': 560, 'source_original_parent': '11d3a451b9fc49fb822213859c040b3a090298d6',
    'local_commit_at_verification': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'source_sha256': replay['source_sha256'], 'root_existing_function_repairs':
        ['_weights_on_disk', 'encoder_entry', '_retrieval_scorecard'],
    'verification_scope': 'Owned three-function repair integrated with df5b9bb7f strict timm sizing, before the later Home four-channel SubCell source checkpoint. The source digest is authoritative; later source needs its own refresh and acceptance.',
    'later_Home_four_channel_source_pending_integration_at_this_receipt': '3ce39ac7bde621dffc34c82f18779ce1a21a61e0',
    'Home_timm_resize_retained_exact_to_df5b9bb7f': True,
    'all_twelve_current_source_full_cohorts_and_six_actual_checkpoint_digests_pass': True,
    'all_measured_scores_and_chance_precision_preserved_atol_1e_12': True,
    'exact_finite_random_ranking_AP_corrected': True,
    'four_crop_random_AP': '11/18', 'four_crop_random_precision': '1/3',
    'normal_all_nine_generation_functions_and_all_nine_full_normal_CLI_audits_pass': True,
    'only_one_public_API_entry_changed_each_catalog': 'spacr.embeddings.encoder_entry',
    'current_API_records': 13181, 'all_other_complete_API_records_exact': 13180,
    'all_seven_encoder_description_blocks_directly_technically_reviewed_each_language': True,
    'prior_complete_runtime_rows_and_hashes_exact': 9857, 'new_runtime_notes': 3,
    'current_runtime_rows': 9860, 'previous_complete_guide_messages_exact': 2979,
    'new_puncta_guide_reference_messages': 12, 'current_guide_messages': 2991,
    'all_nine_guide_audits_coverage_100_percent_stale_invalid_unlabelled_zero': True,
    'all_nine_readmes_normally_rebuilt_and_original_prose_ceiling_preserved': True,
    'dated_checkout_claim_binary_MB': 1710,
    'only_redundant_installer_description_and_duplicate_feature_guide_sentence_trimmed': True,
    'obsolete_omit_Qt_source_archived_and_standard_Qt_headless_execution_translated': True,
    'removed_Make_Masks_heading_review_cannot_reactivate_from_embedded_mentions': True,
    'all_old_reviewed_API_runtime_files_and_order_exact': preservation['reviewed_files'],
    'Help_search_all_four_YOLO_helpers_and_current_theme_descriptions_regenerated': True,
    'Make_Masks_reference_all_six_puncta_parameters_normally_regenerated': True,
    'strict_complete_Sphinx_build_and_all_nine_guide_builds_passed': True,
    'nine_actual_local_browser_panels_all_seven_current_blocks_exact': True,
    'nine_actual_local_browser_Make_Masks_pages_all_twelve_new_reference_messages_exact': True,
    'checks_passed': counts, 'inventory_guard_repair': 'b39c55531',
    'no_old_guard_pin_or_debt_coverage_editorial_ceiling_raised': True,
    'all_Mask_Measure_Home_Conda_and_other_tutorial_source_media_preserved': True,
    'retained_nonaccepted_attempts': {
        'old_API_generation_r1_r2_r3': 'All terminated by owner with SIGTERM/rc143. r1 switched from full repair to normal reuse; r2 integrated Home source; r3 completed nine writes but its audit was stopped for the browser-discovered complete-description refinement. None is represented as strict acceptance.',
        'complete_description_review_r1': 'Protected literal order gate rejected Chinese return text before any file write; reviewed r2 preserves source literal order in Chinese, Korean and Hindi.',
        'local_browser_r1': 'Private verifier compared literal RST role markup with rendered text; actual role display is correct. r2 checks rendered roles and all seven reviewed blocks.',
        'local_browser_r2': 'All nine encoder panels passed. The added guide check used a main element selector that the actual Furo theme does not provide; r3 uses its actual article role=main and checks the same twelve complete reviewed messages in each guide.',
        'early_README_checks': 'Temporary missing navigation sentence was restored; localized checkout facts awaited the normal rebuild. Final 32 presentation/install checks pass.',
        'catalog_r1': 'Obsolete Make Masks heading evidence incorrectly reactivated from prose mentions. Source-bound retirement now passes all final catalog checks without changing the test.',
        'Make_Masks_reference_check_r1': 'Six source puncta parameters were absent from the generated include; normal regeneration and nine guide translations repair it.',
        'historical_README_measurement_r1_editorial_field': 'The numerical checkout measurement remains valid. Its temporary module-navigation removal was reversed before acceptance; only the redundant installer paragraph and duplicate feature-guide sentence are absent in the final README.',
        'initial_guard': '144 pass/one inherited NightTheme sound_key pin mismatch, proven and repaired independently in b39c55531; original logs remain archived.',
        'raw_tracebacks': 'Wrapper failure tracebacks outside its Tee are not reconstructed as raw log evidence.',
    },
    'actual_publication_readback_of_new_checkpoint_pending': True,
    'no_new_GPU_inference_or_classifier_fit_or_native_speaker_host_or_global_CI_claim': True,
}
receipt['artifacts'] = {str(path): {'sha256': digest(path), 'bytes': path.stat().st_size}
    for path in sorted(destination.rglob('*')) if path.is_file()}
Path(str(destination) + '.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
note = '\n2026-10-06 workstation final local encoder/documentation acceptance: the existing provenance and exact finite random-AP repairs pass all twelve original full expert fluorescence cohorts on the final application source, with unchanged measured scores/chance precision at 1e-12 and all six actual checkpoint digests. Encoder description now explains provenance, optional retrieval metrics and compatible channel policies without internal instruction numbers; all seven blocks are directly technically reviewed in all nine languages. The normal API generation functions and all nine full normal CLI audits pass; the other 13,180 complete API records, all 9,857 prior runtime rows/hashes and every older reviewed API/runtime file/order remain exact. Current runtime total is 9,860. Current README is below the original 1,750-word prose bound, honestly states its dated 1,710 MB tracked checkout, and all nine READMEs are normally rebuilt with correct standard Qt/headless execution guidance. Obsolete omit-Qt wording and the removed Make Masks heading review are retained as archival evidence, not active assertions. Help search is normally regenerated for all four YOLO helpers and current theme summaries. Six omitted puncta parameters are generated into the Make Masks reference; all nine guides preserve every prior 2,979 complete message and add twelve exact reviewed reference messages, with 2,991/2,991, zero stale/invalid/unlabelled. Strict complete Sphinx and all nine translated guide builds pass, with all seven blocks rendered exactly in nine actual browser panels. Receipt 560_foundation_provenance_and_exact_chance_API_2026-10-06.json retains final/current and failed/nonaccepted source-bound diagnostics and proofs. Mask07 and corrected Measure08/Home05/Conda02 plus every tutorial source/media byte are unchanged. Home retains feature/model integration and CPU CI/coverage/serial Qt; workstation retains all GPU/API/docs/translations/tutorials. No global green, official Cell-DINO/four-channel integration, native-speaker/host or new GPU inference is claimed. Actual new-checkpoint publication/readback remains pending; items560/615 remain OPEN for the remaining scope.\n'
note = note.replace('on the final application source', 'on the exact source SHA ' + replay['source_sha256'] + ', integrated with df5b9bb7f before the later four-channel source')
note += '\n2026-10-06 workstation next-source ownership: Home checkpoint 3ce39ac7b adds native-size four-channel SubCell and explicit alpha channel mapping. It is not covered by this earlier source receipt and is the next private integration/API/runtime/documentation scope. Workstation retains its GPU acceptance. The public 349,009,018-byte all_channels_ViT-ProtS-Pool.pth and three pinned authors reference files have been independently reacquired; all four SHA256 values exactly match Home CPU evidence. Acquisition is preparation only, not four-stain biological or CUDA acceptance. Current protected livecell/cellposeTIME jobs are untouched.\n'
for path in ('features/325_two_sessions_one_repo_working_protocol.temp',
             'features/future/560_foundation_model_embeddings.txt',
             'features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt'):
    with Path(path).open('a') as handle:
        handle.write(note)
print('PASS final source encoder API, README, Help and puncta guide acceptance archived', flush=True)
