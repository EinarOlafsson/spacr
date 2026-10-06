from pathlib import Path
import gzip
import hashlib
import json
import shutil
import sys
import zipfile

import numpy as np

sys.meta_path[:] = [f for f in sys.meta_path if not getattr(f, '__module__', '').startswith('__editable__')]
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, 'tools')
from benchmark_segmentation_strategies import OMNIPOSE_IMAGES, score_field

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '553-omnipose-gpu-r1'
read = lambda p: json.loads(p.read_text())
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
cpu = read(root / 'cpu/acceptance.json')
gpu = read(root / 'cuda/acceptance.json')
assert cpu['accepted_execution'] and gpu['accepted_execution']
assert cpu['device'] == 'cpu' and gpu['device'] == 'cuda'
for key in ('inputs', 'upstream_commit', 'models', 'wrapper_sha256', 'benchmark_script_sha256', 'application_backend_sha256', 'benchmark_driver_sha256'):
    assert cpu[key] == gpu[key], key
assert digest(Path('spacr/_segmentation_backends.py')) == gpu['application_backend_sha256']
assert digest(Path('tools/benchmark_segmentation_strategies.py')) == gpu['benchmark_driver_sha256']
assert digest(scratch / 'benchmark_omnipose_current.py') == gpu['benchmark_script_sha256']
assert digest(scratch / 'omnipose_worker_profile.py') == gpu['wrapper_sha256']
assert len(cpu['profiles']) == len(gpu['profiles']) == len(cpu['scored_fields']) == len(gpu['scored_fields']) == 7
assert len(gpu['inputs']) == 14
for name, expected in gpu['inputs'].items():
    assert digest(root / 'omnipose_test' / name) == expected
for path, expected in gpu['models'].items():
    assert digest(Path(path)) == expected
fields = []
for index, (c, g, cp, gp) in enumerate(zip(cpu['scored_fields'], gpu['scored_fields'], cpu['profiles'], gpu['profiles'])):
    assert cp['cuda_events'] == 0 and cp['actual_worker_reply_device'] == 'cpu'
    assert gp['cuda_kernel_events'] > 0 and gp['actual_worker_reply_device'] == 'cuda'
    for key in ('model', 'params', 'input_file_sha256', 'source_sha256', 'torch_version'):
        assert cp[key] == gp[key], (index, key)
    assert c['reference_sha256'] == g['reference_sha256']
    arrays = []
    for row in (c, g):
        for key in ('prediction', 'reference'):
            assert digest(Path(row[key])) == row[key + '_sha256']
        truth, pred = (np.load(row[key], allow_pickle=False) for key in ('reference', 'prediction'))
        assert score_field(truth, pred) == row['scores']
        arrays.append((truth, pred))
    assert np.array_equal(arrays[0][0], arrays[1][0])
    c_mask, g_mask = arrays[0][1], arrays[1][1]
    assert c_mask.shape == g_mask.shape
    fields.append({'field': Path(OMNIPOSE_IMAGES[index]).stem,
                   'cpu_objects': c['scores']['n_pred'], 'cuda_objects': g['scores']['n_pred'],
                   'exact_label_array_equal': bool(np.array_equal(c_mask, g_mask)),
                   'foreground_pixel_agreement': float(np.mean((c_mask > 0) == (g_mask > 0))),
                   'cpu_to_cuda_instance_match': score_field(c_mask, g_mask),
                   'reference_identical': True, 'model_parameters_identical': True,
                   'cuda_kernel_events': gp['cuda_kernel_events']})
for run in (cpu, gpu):
    for tag in ('50', '75'):
        for prefix in ('tp', 'fp', 'fn'):
            assert run['pooled'][prefix + tag] == sum(f['scores'][prefix + tag] for f in run['scored_fields'])
    assert run['pooled']['n_truth'] == 621
report = {'item': 553, 'normal_seven_field_cuda_mask_execution_accepted': True,
          'terminal_turn_exit_code': 0, 'turn_label': '553-omnipose-current-cuda-20261006-r1',
          'same_actual_inputs_model_source_and_parameters': True,
          'fresh_cpu_and_cuda_scored_outputs_independently_recomputed': True,
          'exact_cpu_cuda_label_parity': all(f['exact_label_array_equal'] for f in fields),
          'cpu_pooled': cpu['pooled'], 'cuda_pooled': gpu['pooled'], 'fields': fields,
          'comparison_scope': 'Actual CPU/CUDA instance agreement; differences are retained and no numerical parity threshold is invented.',
          'reference_labels_from_upstream_Omnipose_not_independent_human_truth': True,
          'benchmark_derives_reference_diameter_but_adapter_declares_it_unsupported': True,
          'three_dimensional_path_validated': False, 'broader_model_comparisons_validated': False}
dest = Path('features/data/553_omnipose_gpu_acceptance_2026-10-06')
dest.mkdir(exist_ok=True)
for name in ('benchmark_omnipose_current.py', 'omnipose_worker_profile.py', Path(__file__).name):
    shutil.copyfile(scratch / name, dest / name)
for name, source in (('cpu-acceptance.json', root / 'cpu/acceptance.json'), ('cuda-acceptance.json', root / 'cuda/acceptance.json')):
    shutil.copyfile(source, dest / name)
(dest / 'omnipose-current-cuda-r1.log.gz').write_bytes(gzip.compress((scratch / 'omnipose-current-cuda-r1.log').read_bytes(), mtime=0))
with zipfile.ZipFile(dest / 'actual-cpu-cuda-outputs.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for run in ('cpu', 'cuda'):
        for folder in ('profiling', 'scored-labels', 'results'):
            for path in sorted((root / run / folder).rglob('*')):
                if path.is_file():
                    archive.write(path, str(path.relative_to(root)))
report['artifacts'] = {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(dest.iterdir()) if p.is_file()}
Path('features/data/553_omnipose_gpu_acceptance_2026-10-06.json').write_text(json.dumps(report, indent=2) + '\n')
note = '\n2026-10-06 workstation Omnipose real GPU execution acceptance: turn 553-omnipose-current-cuda-20261006-r1 is terminal rc=0. Normal seven-field Mask inference uses the same fourteen pinned upstream input files, private model files, application/driver source and worker parameters as the fresh CPU baseline; each actual worker request records 2,313-2,555 CUDA kernel events. Every original reference score is independently recomputed from retained actual masks. CPU and CUDA both produce 644 objects, and every one matches at IoU 0.75; three complete label arrays are exact and four have retained pixel/label differences, so exact overall parity is false. Both runs have pooled F1@0.5=0.9518/F1@0.75=0.9265 against 621 upstream Omnipose-produced reference objects. This establishes actual execution and reference agreement, not independent biological accuracy or a throughput claim. The adapter explicitly reports diameter unsupported. Receipt 553_omnipose_gpu_acceptance_2026-10-06.json archives paired outputs, complete positive CUDA profiles, model/source hashes and terminal raw turn. 3-D plant and broader model comparisons remain open. Home retains CPU CI/coverage/Qt/source ownership; all GPU work stays workstation-owned and protected livecell/cellposeTIME jobs remain untouched.\n'
for path in ('features/future/553_omnipose_backend.txt', 'features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt', 'features/325_two_sessions_one_repo_working_protocol.temp'):
    with Path(path).open('a') as stream:
        stream.write(note)
print(json.dumps({k: report[k] for k in ('normal_seven_field_cuda_mask_execution_accepted', 'exact_cpu_cuda_label_parity', 'cpu_pooled', 'cuda_pooled', 'fields')}, indent=2), flush=True)
