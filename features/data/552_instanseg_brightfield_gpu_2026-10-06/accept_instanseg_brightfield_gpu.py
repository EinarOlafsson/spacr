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
root = scratch / '552-instanseg-brightfield-r2'
read = lambda p: json.loads(p.read_text())
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
cpu = read(root / 'cpu-r2/acceptance.json')
gpu = read(root / 'cuda/acceptance.json')
assert cpu['accepted_execution'] and gpu['accepted_execution']
assert cpu['device'] == 'cpu' and gpu['device'] == 'cuda'
for key in ['source', 'actual_original_RGB_shape', 'crop_or_simulated_data_used',
            'source_TIFF_channels_hashes', 'models', 'benchmark_sha256', 'worker_sha256', 'source_hashes']:
    assert cpu[key] == gpu[key], key
for path, expected in gpu['source_hashes'].items():
    assert digest(path) == expected
for path, expected in gpu['models'].items():
    assert digest(path) == expected
assert digest(root / 'HE_example.tif') == gpu['source']['source_sha256']
assert digest(scratch / 'benchmark_instanseg_brightfield.py') == gpu['benchmark_sha256']
assert digest(scratch / 'instanseg_multiplex_worker.py') == gpu['worker_sha256']
assert len(cpu['cases']) == len(gpu['cases']) == 2
rows = []
for c, g in zip(cpu['cases'], gpu['cases']):
    assert c['route'] == g['route']
    for case, device in [(c, 'cpu'), (g, 'cuda')]:
        profile = case['profile']
        assert profile['actual_worker_reply_device'] == device
        assert (profile['cuda_kernel_events'] > 0) if device == 'cuda' else (profile['cuda_events'] == 0)
        for row in profile['inputs'] + profile['outputs']:
            assert digest(row['path']) == row['sha256']
        assert digest(case['labels']) == case['labels_sha256']
        labels = np.load(case['labels'], allow_pickle=False)
        assert int(len(np.unique(labels[labels > 0]))) == case['objects']
    for key in ['model', 'params', 'source_sha256', 'torch_version', 'cuda_runtime']:
        assert c['profile'][key] == g['profile'][key], key
    assert c['profile']['inputs'][0]['sha256'] == g['profile']['inputs'][0]['sha256']
    compared = {}
    for name, paths in [('original_worker_output', [c['profile']['outputs'][0]['path'], g['profile']['outputs'][0]['path']]),
                        ('final_normal_route_labels', [c['labels'], g['labels']])]:
        left, right = [np.load(path, allow_pickle=False) for path in paths]
        assert left.shape == right.shape == (1484, 1380)
        compared[name] = {'exact_label_array_equal': bool(np.array_equal(left, right)),
                         'foreground_pixel_agreement': float(np.mean((left > 0) == (right > 0))),
                         'cpu_to_cuda_instance_agreement_not_human_accuracy': score_field(left, right)}
    rows.append({'route': c['route'], 'cpu_objects': c['objects'], 'cuda_objects': g['objects'],
                 'cuda_kernel_events': g['profile']['cuda_kernel_events'], **compared})
    print('PASS:', c['route'], 'input/model/source/device contract; actual CPU/CUDA agreement', rows[-1], flush=True)
log = (scratch / 'instanseg-brightfield-cuda-r1.log').read_text()
assert '[552-instanseg-brightfield-cuda-20261006-r1] FINISH rc=0' in log
dest = Path('features/data/552_instanseg_brightfield_gpu_2026-10-06')
dest.mkdir(exist_ok=False)
for name in ['benchmark_instanseg_brightfield.py', 'instanseg_multiplex_worker.py', Path(__file__).name]:
    shutil.copyfile(scratch / name, dest / name)
for run in ['cpu-r2', 'cuda']:
    shutil.copyfile(root / run / 'acceptance.json', dest / (run + '-acceptance.json'))
for name in ['instanseg-brightfield-cpu-r2.log', 'instanseg-brightfield-cuda-r1.log']:
    (dest / (name + '.gz')).write_bytes(gzip.compress((scratch / name).read_bytes(), mtime=0))
paths = [root / 'HE_example.tif', root / 'source-provenance.json', root / 'upstream-README.md']
for run in ['cpu-r2', 'cuda']:
    folder = root / run
    paths.extend(p for p in sorted((folder / 'profiling').rglob('*')) if p.is_file())
    paths.extend(Path(case['labels']) for case in read(folder / 'acceptance.json')['cases'])
    paths.extend(p for p in sorted((folder / 'independent-channel-controls').rglob('*.npz')))
    paths.extend(p for p in sorted((folder / 'normal-mask-plate/settings').glob('*')) if p.is_file())
with zipfile.ZipFile(dest / 'original-input-controls-profiles-and-paired-labels.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(set(paths)):
        archive.write(path, str(path.relative_to(root)))
report = {'item': 552, 'original_full_field_brightfield_real_GPU_execution_accepted': True,
    'terminal_GPU_returncode': 0, 'turn_label': '552-instanseg-brightfield-cuda-20261006-r1',
    'original_RGB_shape': [1484, 1380, 3], 'input_models_source_parameters_and_worker_contract_exact': True,
    'normal_RGB_ingest_verified_by_independent_single_channel_controls': True,
    'routes': rows, 'same_route_CPU_CUDA_exact_label_arrays': all(row['final_normal_route_labels']['exact_label_array_equal'] for row in rows),
    'CPU_CUDA_count_and_pixel_differences_explicitly_retained': True,
    'no_crop_simulation_independent_human_accuracy_or_throughput_claim': True,
    'Make_Masks_and_normal_Mask_route_outputs_not_claimed_identical': True,
    'artifacts': {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(dest.iterdir())}}
Path('features/data/552_instanseg_brightfield_gpu_2026-10-06.json').write_text(json.dumps(report, indent=2) + '\n')
counts = [(row['route'], row['cpu_objects'], row['cuda_objects'], row['cuda_kernel_events'],
           row['final_normal_route_labels']['cpu_to_cuda_instance_agreement_not_human_accuracy']['iou_mean']) for row in rows]
note = '\n2026-10-06 workstation InstanSeg full brightfield GPU acceptance: normal turn 552-instanseg-brightfield-cuda-20261006-r1 is terminal rc=0. The original acquired 1484x1380 RGB HE_example.tif runs through actual Make Masks and normal Mask ingest/segmentation with positive profiled CUDA kernels in both workers. Same original input/model/source/worker parameters and every independently normalized RGB channel match the accepted CPU controls. Per-route CPU count, CUDA count, CUDA kernels and matched mean IoU: ' + repr(counts) + '. Exact CPU/CUDA label arrays are not equal and the count/pixel/object differences are retained; the two differently preprocessed routes are not claimed equivalent. This is full-field execution and CPU/CUDA agreement, not independent human brightfield accuracy or throughput. Receipt 552_instanseg_brightfield_gpu_2026-10-06.json archives original TIFF, paired raw/final labels, requests/profiles, channel controls and terminal logs. Together with prior single-channel/multiplex receipts this closes the assigned InstanSeg GPU execution checks at the recorded acquired-data scope; potential new 3D applications are not claimed. No GPU job remains from this turn. Home retains CPU/Qt/CI/source ownership and protected livecell/cellposeTIME jobs remain untouched.\n'
for path in ['features/future/552_instanseg_backend.txt',
             'features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt',
             'features/325_two_sessions_one_repo_working_protocol.temp']:
    with Path(path).open('a') as stream:
        stream.write(note)
print('PASS: original full-field brightfield CUDA contract and explicit CPU/GPU agreement archived.', flush=True)
