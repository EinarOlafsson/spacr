from pathlib import Path
import gzip
import hashlib
import json
import shutil

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '560-human-fluorescence-primary-r1'
encoded = root / 'full-fluorescence-foundation-CUDA-r1'
destination = Path('features/data/560_full_human_fluorescence_CUDA_2026-10-06')
destination.mkdir(exist_ok=False)
def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            sha.update(block)
    return sha.hexdigest()
plan_path = root / 'frozen-full-fluorescence-foundation-plan-r1.json'
plan = json.loads(plan_path.read_text())
execution = json.loads((encoded / 'complete-execution.json').read_text())
assert execution['passed'] and len(execution['comparisons']) == 12
assert digest(plan_path) == execution['plan_sha256'] == 'cf7321646d184fb475c243349c87066cb2ec16bb244242d7700288964e409746'
assert digest('spacr/embeddings.py') == plan['application_source_sha256']
label = '560-full-human-fluorescence-foundations-20261006-r1'
events = [line for line in (Path.home() / '.spacr/gpu/log').read_text().splitlines() if '[' + label + ']' in line]
assert len(events) == 4 and 'queued' in events[0] and '360s idle' in events[1] and 'START:' in events[2] and 'FINISH rc=0' in events[3]
(destination / 'exact-normal-GPU-turn-lifecycle.json').write_text(json.dumps({'exact_label': label, 'exact_four_original_lifecycle_lines': events, 'normal_idle_seconds': 360, 'normal_handoff_seconds': 600, 'actual_terminal_return_code': 0}, indent=2) + '\n')
originals = {}
for comparison in execution['comparisons']:
    cohort = plan['cohorts'][comparison['cohort']]
    preparation = plan['models'][comparison['backbone']]['receipt']
    assert comparison['ordered_rows'] == cohort['rows'] and comparison['rows'] == len(cohort['rows'])
    assert comparison['complete_ordered_model_state_sha256'] == preparation['complete_ordered_model_state_sha256']
    assert comparison['all_actual_model_parameters_on_CUDA']
    assert sum(comparison['input_stack_shapes'].values()) == (comparison['rows'] + 15) // 16
    for profile in comparison['first_last_actual_CUDA_profiles']:
        path = Path(profile['trace_path'])
        assert digest(path) == profile['sha256']
        trace = json.loads(path.read_text())
        assert sum(event.get('cat') == 'kernel' for event in trace['traceEvents']) == profile['CUDA_kernel_events'] > 0
    original = Path(comparison['full_features']['path'])
    assert digest(original) == comparison['full_features']['sha256']
    originals[original.name] = comparison['full_features']
    print('PASS original full GPU features, keys, labels, model state and both CUDA traces', comparison['backbone'], comparison['cohort'], flush=True)
paths = sorted(encoded.iterdir()) + [root / 'full-fluorescence-foundation-CUDA-r1.log', plan_path]
for path in paths:
    assert path.is_file()
    target = destination / (path.name + '.gz')
    with path.open('rb') as source, target.open('wb') as sink:
        with gzip.GzipFile(fileobj=sink, mode='wb', mtime=0, filename='', compresslevel=6) as compressor:
            shutil.copyfileobj(source, compressor, length=1024 * 1024)
    reopened = hashlib.sha256()
    with gzip.open(target, 'rb') as stream:
        while block := stream.read(1024 * 1024):
            reopened.update(block)
    assert reopened.hexdigest() == digest(path)
    assert target.stat().st_size < 100_000_000
    print('PASS all original bytes losslessly archived and replayed', path.name, target.stat().st_size, flush=True)
for name in ('run_full_fluorescence_foundation_CUDA_r1.py', 'archive_full_fluorescence_foundation_CUDA_r1.py'):
    shutil.copyfile(scratch / name, destination / name)
assert (destination / 'run_full_fluorescence_foundation_CUDA_r1.py').read_bytes() == Path('features/data/560_full_fluorescence_foundation_preparation_2026-10-06/run_full_fluorescence_foundation_CUDA_r1.py').read_bytes()
artifacts = {str(path): {'sha256': digest(path), 'bytes': path.stat().st_size} for path in sorted(destination.iterdir())}
receipt = {'item': 560, 'all_twelve_real_normal_CUDA_encoder_comparisons_completed_on_all_original_eligible_expert_crops': True,
           'six_normal_backbones': list(plan['models']), 'cohort_sizes': {name: len(value['rows']) for name, value in plan['cohorts'].items()},
           'actual_normal_GPU_turn_terminal_rc': 0, 'normal_GPU_idle_and_handoff_rules_preserved': True,
           'actual_GPU': execution['actual_CUDA_device'], 'frozen_plan_sha256': execution['plan_sha256'],
           'application_source_sha256': plan['application_source_sha256'], 'all_actual_loaded_model_states_and_checkpoint_bytes_match_CPU_freeze': True,
           'actual_first_and_last_CUDA_kernel_profiles': 24,
           'all_original_feature_keys_labels_full_matrices_and_traces_losslessly_preserved': True,
           'original_full_feature_files': originals, 'SubCell_DAPI_is_DNA_only_zero_protein_ablation': True,
           'SubCell_yeast_uses_genuine_nuclear_reference_and_GFP_planes': True,
           'CPU_full_normal_scorecards_classifiers_and_independent_rank_AP_replay_pending': True,
           'Model_Zoo_provenance_and_exact_random_AP_chance_API_repairs_remain_open': True,
           'no_Cell_DINO_four_channel_SubCell_clinical_plate_holdout_or_manual_mask_accuracy_claim': True,
           'no_application_source_change_or_protected_job_touched': True, 'artifacts': artifacts}
Path(str(destination) + '.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('PASS COMPLETE actual all-twelve full-cohort normal CUDA acceptance and independently replayed original-byte archival', len(artifacts), 'artifacts', flush=True)
