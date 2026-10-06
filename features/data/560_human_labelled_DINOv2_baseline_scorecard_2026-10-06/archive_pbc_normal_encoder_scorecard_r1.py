from pathlib import Path
import gzip
import hashlib
import importlib.metadata
import json
import shutil

import numpy as np

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '565-pbc-retrieval-r4'
target = root / 'normal-human-label-encoder-scorecard-r3'
actual = json.loads((target / 'acceptance.json').read_text())
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert actual['normal_full_human_labelled_DINOv2_baseline_scorecard_and_model_zoo_entry_complete']
assert actual['normal_retrieval_scorecard'] == actual['independent_complete_blocked_rank_AP_and_vote_replay_exact']
assert all(value == '1' for value in actual['CPU_thread_settings'].values())
entry = actual['normal_model_zoo_encoder_entry']
assert digest(entry['path']) == entry['sha256']
for path, sha in actual['source_sha256'].items():
    assert digest(path) == sha
for path, record in actual['artifacts'].items():
    assert digest(path) == record['sha256'] and Path(path).stat().st_size == record['bytes']
with np.load(target / 'all-human-retrieval-AP-and-precision.npz', allow_pickle=False) as data:
    assert len(data['keys']) == len(set(data['keys'])) == 17074
    assert ((data['average_precision'] >= 0) & (data['average_precision'] <= 1)).all()
    assert ((data['precision_at_10'] >= 0) & (data['precision_at_10'] <= 1)).all()
    np.testing.assert_allclose(data['average_precision'].mean(), actual['normal_retrieval_scorecard']['map'], rtol=0, atol=1e-12)
    np.testing.assert_allclose(data['precision_at_10'].mean(), actual['normal_retrieval_scorecard']['precision_at_k'], rtol=0, atol=1e-12)
out = Path('features/data/560_human_labelled_DINOv2_baseline_scorecard_2026-10-06')
out.mkdir(exist_ok=False)
sources = list(target.iterdir()) + [scratch / f'measure_pbc_normal_encoder_scorecard_r{i}.py' for i in (1, 2, 3)] + [scratch / f'pbc-normal-encoder-scorecard-r{i}.log' for i in (1, 2, 3)] + [scratch / 'probe_pbc_full_matrix_BLAS_r1.py', scratch / 'pbc-full-matrix-BLAS-one-thread-r1.log', scratch / 'pbc-full-matrix-BLAS-two-thread-r1.log', Path(__file__)]
for source in sources:
    destination = out / source.name
    if source.suffix == '.log':
        destination = destination.with_name(destination.name + '.gz')
        destination.write_bytes(gzip.compress(source.read_bytes(), mtime=0))
        assert gzip.decompress(destination.read_bytes()) == source.read_bytes()
    else:
        shutil.copyfile(source, destination)
assert 'PASS actual full-size float64 dense product' in (scratch / 'pbc-full-matrix-BLAS-one-thread-r1.log').read_text()
assert 'Fatal Python error: Segmentation fault' in (scratch / 'pbc-full-matrix-BLAS-two-thread-r1.log').read_text()
assert 'Fatal Python error: Segmentation fault' in (scratch / 'pbc-normal-encoder-scorecard-r2.log').read_text()
receipt = {'item': 560, 'normal_full_expert_labelled_DINOv2_baseline_encoder_scorecard_complete_on_one_CPU_thread': True, 'original_unique_unambiguous_cells': 17074, 'dimensions': 384, 'normal_kNN_mAP_precision_chance': actual['normal_retrieval_scorecard'], 'independent_complete_blocked_rank_AP_and_vote_replay_exact': True, 'normal_model_zoo_entry_with_actual_cached_weight_SHA256': entry, 'normal_five_stratified_crop_fold_linear_classifier_diagnostic': actual['normal_linear_classifier_scorecard'], 'no_patient_well_plate_held_out_or_clinical_accuracy_claim': True, 'original_full_cohort_not_reduced_or_resampled': True, 'earlier_two_thread_private_scorecard_runs_terminal_rc139_retained': True, 'same_size_bare_numpy_product_two_threads_rc139_one_thread_rc0': True, 'actual_local_numpy_version': importlib.metadata.version('numpy'), 'local_OpenBLAS_0_3_31_dev_build_configuration_archived': True, 'one_thread_workaround_not_a_claim_of_two_thread_problem_resolved': True, 'Home_CPU_source_environment_owner_should_investigate_dense_scorecard_BLAS_failure_before_broad_acceptance': True, 'all_source_inputs_GPU_feature_and_model_hashes_exact': True, 'no_new_GPU_inference_application_source_or_production_environment_change': True, 'foundation_model_comparisons_Cell_DINO_and_GUI_pipeline_screen_integration_remain_separate': True, 'artifacts': {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(out.iterdir())}}
out.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')
note = '''
2026-10-06 workstation complete expert-labelled DINOv2 baseline scorecard: all 17,074 original unique unambiguous PBC cells and their accepted actual CUDA features run through normal _scored_encoder_entry. kNN accuracy is 0.9279020733278669, mAP 0.4630365058482782 and precision@10 0.8568876654562493, with mAP/precision chance 0.14397477263482286. Independent blocked ranking over the full unchanged cohort reproduces every aggregate kNN/AP/vote/chance metric exactly and retains all original per-cell AP/precision values. The normal Model Zoo entry attaches the scorecard and hashes the actual cached 88,240,510-byte pretrained weights. Normal five-fold stratified crop logistic-regression diagnostic gives accuracy 0.9681974918537998, SD 0.0034559488664303163, versus majority chance 0.19491624692514936; scaling/fitting are inside each training fold. Patient identifiers are unavailable, so this is within-cohort crop performance, not a patient/plate/well holdout or clinical accuracy claim. Receipt 560_human_labelled_DINOv2_baseline_scorecard_2026-10-06.json archives actual normal scorecard, full independent per-cell replay, entry, raw logs and scripts. Source/input/model/GPU feature hashes remain exact; there is no further GPU inference or source/environment change.

The first two ordinary two-thread attempts terminate rc=139 at _retrieval_scorecard's full float64 product. An independent bare NumPy ones-matrix diagnostic reproduces rc=139 at the same 17074x384 product with two BLAS threads, while the same full-size diagnostic and full normal scorecard pass with one thread. Actual NumPy/OpenBLAS0.3.31.dev configuration and all failed/passed logs are retained. The cohort and scientific parameters are unchanged, and no memory ceiling or guard is raised. One-thread success is a workaround, not resolution of the default two-thread failure. Home owns its CPU/source/environment investigation and should make the normal full-cohort scorecard robust before broader application acceptance. This establishes the genuine human-labelled DINOv2 comparison baseline; the foundation-model comparisons and remaining Cell-DINO/GUI/pipeline/screen-wide source work remain open. All GPU/API/docs/tutorial/translation work stays workstation-owned; protected jobs remain untouched.
'''
for path in (Path('features/future/560_foundation_model_embeddings.txt'), Path('features/future/565_similarity_search_find_cells_like_this.txt'), Path('features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt'), Path('features/325_two_sessions_one_repo_working_protocol.temp')):
    with path.open('a') as stream:
        stream.write(note)
print('PASS full human-label DINOv2 baseline normal scorecard/classifier and independent replay archived, with original local two-thread BLAS crash and one-thread workaround retained.', flush=True)
