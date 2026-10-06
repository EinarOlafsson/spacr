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
from benchmark_segmentation_strategies import score_field

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '552-instanseg-gpu-r1'
read = lambda p: json.loads(p.read_text())
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
cpu = read(root / 'cpu-r2/acceptance.json')
gpu = read(root / 'cuda/acceptance.json')
assert cpu['accepted_execution'] and gpu['accepted_execution']
assert cpu['device'] == 'cpu' and gpu['device'] == 'cuda'
for key in ('inputs', 'models', 'wrapper_sha256', 'benchmark_script_sha256', 'application_backend_sha256', 'benchmark_driver_sha256'):
    assert cpu[key] == gpu[key], key
assert digest(Path('spacr/_segmentation_backends.py')) == gpu['application_backend_sha256']
assert digest(Path('tools/benchmark_segmentation_strategies.py')) == gpu['benchmark_driver_sha256']
assert digest(scratch / 'benchmark_instanseg_current.py') == gpu['benchmark_script_sha256']
assert digest(scratch / 'instanseg_worker_profile.py') == gpu['wrapper_sha256']
assert len(cpu['profiles']) == len(gpu['profiles']) == len(cpu['scored_fields']) == len(gpu['scored_fields']) == 4
fields = []
for index, (c, g, cp, gp) in enumerate(zip(cpu['scored_fields'], gpu['scored_fields'], cpu['profiles'], gpu['profiles']), 1):
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
    fields.append({'field': Path(gpu['inputs'][index - 1]['path']).stem,
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
    assert run['pooled']['n_truth'] == 223
report = {'item': 552, 'normal_four_field_cuda_mask_execution_accepted': True,
          'terminal_turn_exit_code': 0, 'turn_label': '552-instanseg-current-cuda-20261006-r1',
          'same_actual_inputs_model_source_and_parameters': True,
          'fresh_cpu_and_cuda_scored_outputs_independently_recomputed': True,
          'exact_cpu_cuda_label_parity': all(f['exact_label_array_equal'] for f in fields),
          'cpu_pooled': cpu['pooled'], 'cuda_pooled': gpu['pooled'], 'fields': fields,
          'comparison_scope': 'Actual CPU/CUDA instance agreement; differences are retained and no numerical parity threshold is invented.',
          'reference_labels_from_Cellpose_SAM_not_independent_human_truth': True,
          'reference_label_diameters_used_for_reproducible_execution': True,
          'brightfield_model_validated': False, 'multiplex_gpu_validated': False}
dest = Path('features/data/552_instanseg_gpu_acceptance_2026-10-06')
dest.mkdir(exist_ok=True)
for name in ('benchmark_instanseg_current.py', 'instanseg_worker_profile.py', Path(__file__).name):
    shutil.copyfile(scratch / name, dest / name)
for name, source in (('cpu-acceptance.json', root / 'cpu-r2/acceptance.json'), ('cuda-acceptance.json', root / 'cuda/acceptance.json')):
    shutil.copyfile(source, dest / name)
(dest / 'instanseg-current-cuda-r1.log.gz').write_bytes(gzip.compress((scratch / 'instanseg-current-cuda-r1.log').read_bytes(), mtime=0))
with zipfile.ZipFile(dest / 'actual-cpu-cuda-outputs.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for run in ('cpu-r2', 'cuda'):
        for folder in ('profiling', 'scored-labels', 'results'):
            for path in sorted((root / run / folder).rglob('*')):
                if path.is_file():
                    archive.write(path, str(path.relative_to(root)))
report['artifacts'] = {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(dest.iterdir()) if p.is_file()}
Path('features/data/552_instanseg_gpu_acceptance_2026-10-06.json').write_text(json.dumps(report, indent=2) + '\n')
note = '\n2026-10-06 workstation InstanSeg real GPU execution acceptance: normal turn 552-instanseg-current-cuda-20261006-r1 is terminal rc=0. The same four acquired nucleus fields, private model files, application/driver source and actual worker parameters as the fresh CPU baseline pass normal Mask inference, with 1,601/1,528/1,437/1,563 actual CUDA kernel events. Exact scored masks and references are retained and every original reference score is independently recomputed. CUDA produces 208 objects versus 209 on CPU; exact label parity is false. All 208 CUDA objects match CPU instances at IoU 0.75, with per-field matched mean IoU 0.9989-0.9997; one CPU-only object and all pixel differences remain in the archive. Against saved Cellpose-SAM reference labels, CUDA pooled F1 is 0.9188/0.8677 at IoU 0.5/0.75, versus CPU 0.9167/0.8657. These are reference-derived diameter execution checks, not independent biological accuracy or a throughput claim. Receipt 552_instanseg_gpu_acceptance_2026-10-06.json archives paired actual outputs, complete profiles, source/model hashes and the terminal raw turn. Brightfield model and multiplex GPU validation remain OPEN. Home retains CPU CI/coverage/Qt/source ownership; GPU work stays workstation-owned and protected livecell/cellposeTIME jobs remain untouched.\n'
for path in ('features/future/552_instanseg_backend.txt', 'features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt', 'features/325_two_sessions_one_repo_working_protocol.temp'):
    with Path(path).open('a') as stream:
        stream.write(note)
print(json.dumps({k: report[k] for k in ('normal_four_field_cuda_mask_execution_accepted', 'exact_cpu_cuda_label_parity', 'cpu_pooled', 'cuda_pooled', 'fields')}, indent=2), flush=True)
