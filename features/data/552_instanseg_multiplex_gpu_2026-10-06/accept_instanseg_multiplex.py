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
root = scratch / '552-instanseg-multiplex-r1'
read = lambda p: json.loads(p.read_text())
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
cpu = read(root / 'cpu-r3/acceptance.json')
gpu = read(root / 'cuda/acceptance.json')
assert cpu['accepted_execution'] and gpu['accepted_execution']
assert cpu['device'] == 'cpu' and gpu['device'] == 'cuda'
for key in ['source', 'explicitly_derived_crop_yx', 'intensity_planes',
            'independent_normalized_reference_sha256', 'fixed_diameter_pixels', 'models',
            'benchmark_script_sha256', 'worker_script_sha256', 'source_hashes']:
    assert cpu[key] == gpu[key], key
for path, expected in gpu['source_hashes'].items():
    assert digest(Path(path)) == expected
for path, expected in gpu['models'].items():
    assert digest(Path(path)) == expected
assert digest(Path(gpu['source']['path'])) == gpu['source']['sha256']
assert digest(scratch / 'benchmark_instanseg_multiplex.py') == gpu['benchmark_script_sha256']
assert digest(scratch / 'instanseg_multiplex_worker.py') == gpu['worker_script_sha256']
assert len(cpu['cases']) == len(gpu['cases']) == 2
fields = []
for c, g in zip(cpu['cases'], gpu['cases']):
    assert c['case'] == g['case']
    assert c['selected_acquired_intensity_planes'] == g['selected_acquired_intensity_planes']
    assert c['source_TIFF_hashes'] == g['source_TIFF_hashes']
    for case, device in [(c, 'cpu'), (g, 'cuda')]:
        assert case['normal_ingest_and_worker_input_exact'] and not case['generated_mask_planes_sent']
        profile = case['profile']
        assert profile['actual_worker_reply_device'] == device
        assert (profile['cuda_kernel_events'] > 0) if device == 'cuda' else (profile['cuda_events'] == 0)
        for row in profile['inputs'] + profile['outputs']:
            assert digest(Path(row['path'])) == row['sha256']
        assert digest(Path(case['mask'])) == case['mask_sha256']
        labels = np.load(case['mask'], allow_pickle=False)
        assert int(len(np.unique(labels[labels > 0]))) == case['objects_after_normal_pipeline_filters']
    for key in ['model', 'params', 'source_sha256', 'torch_version', 'cuda_runtime']:
        assert c['profile'][key] == g['profile'][key], key
    assert c['profile']['inputs'][0]['sha256'] == g['profile']['inputs'][0]['sha256']
    compared = {}
    for name, paths in [('original_worker_output', [c['profile']['outputs'][0]['path'], g['profile']['outputs'][0]['path']]),
                        ('normal_filtered_pipeline_output', [c['mask'], g['mask']])]:
        left, right = [np.load(path, allow_pickle=False) for path in paths]
        assert left.shape == right.shape == (384, 384)
        compared[name] = {'exact_label_array_equal': bool(np.array_equal(left, right)),
                          'foreground_pixel_agreement': float(np.mean((left > 0) == (right > 0))),
                          'cpu_to_cuda_instance_match': score_field(left, right)}
    fields.append({'case': c['case'], 'selected_acquired_intensity_planes': c['selected_acquired_intensity_planes'],
                   'actual_worker_input_shape': g['profile']['inputs'][0]['shape'],
                   'cuda_kernel_events': g['profile']['cuda_kernel_events'], **compared})
log = (scratch / 'instanseg-multiplex-cuda-r1.log').read_text()
assert '[552-instanseg-multiplex-cuda-20261006-r1] FINISH rc=0' in log
report = {'item': 552, 'one_and_three_channel_normal_mask_ingest_and_cuda_inference_accepted': True,
          'terminal_turn_returncode': 0, 'turn_label': '552-instanseg-multiplex-cuda-20261006-r1',
          'same_acquired_intensity_inputs_models_source_and_worker_parameters': True,
          'normal_preprocessing_independently_verified_with_matching_channel_roles': True,
          'generated_reference_mask_plane_5_never_sent': True,
          'explicit_acquired_crop_yx': gpu['explicitly_derived_crop_yx'], 'fields': fields,
          'exact_final_cpu_cuda_label_parity': all(f['normal_filtered_pipeline_output']['exact_label_array_equal'] for f in fields),
          'one_crop_execution_and_cpu_cuda_agreement_not_independent_biological_accuracy': True,
          'no_original_failed_cpu_verifier_attempt_accepted': True,
          'brightfield_acceptance_pending': True}
dest = Path('features/data/552_instanseg_multiplex_gpu_2026-10-06')
dest.mkdir(exist_ok=False)
for name in ['benchmark_instanseg_multiplex.py', 'instanseg_multiplex_worker.py', Path(__file__).name]:
    shutil.copyfile(scratch / name, dest / name)
for run in ['cpu-r3', 'cuda']:
    shutil.copyfile(root / run / 'acceptance.json', dest / (run + '-acceptance.json'))
for name in ['instanseg-multiplex-cpu-r1.log', 'instanseg-multiplex-cpu-r2.log',
             'instanseg-multiplex-cpu-r3.log', 'instanseg-multiplex-cuda-r1.log']:
    (dest / (name + '.gz')).write_bytes(gzip.compress((scratch / name).read_bytes(), mtime=0))
with zipfile.ZipFile(dest / 'actual-inputs-controls-profiles-and-paired-labels.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for run in ['cpu-r3', 'cuda']:
        for path in sorted((root / run).rglob('*')):
            if path.is_file():
                archive.write(path, str(path.relative_to(root)))
report['artifacts'] = {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(dest.iterdir())}
Path('features/data/552_instanseg_multiplex_gpu_2026-10-06.json').write_text(json.dumps(report, indent=2) + '\n')
counts = [(f['case'], f['cuda_kernel_events'], f['normal_filtered_pipeline_output']['cpu_to_cuda_instance_match']) for f in fields]
note = '\n2026-10-06 workstation InstanSeg multiplex GPU acceptance: terminal turn 552-instanseg-multiplex-cuda-20261006-r1 finishes rc=0. Actual normal Mask ingest/inference is accepted for one and three selected acquired intensity channels from an explicitly declared 384x384 crop; generated reference mask plane 5 is excluded. Three independent single-channel ingests preserve the same nucleus/cell/pathogen roles and verify every normalized selected plane and worker request exactly. CPU and GPU retain identical acquired inputs, model/source/worker hashes and parameters; both original worker outputs and final filtered labels are compared without inventing a parity threshold. Counts and actual positive CUDA kernel profiles: ' + repr(counts) + '. This is one-crop execution/channel-contract and CPU/CUDA agreement evidence, not independent multiplex biological accuracy. CPU r1/r2 verifier failures remain failed evidence (raw-vs-normalized and mismatched-role reference assumptions); the fresh corrected CPU r3 is accepted. Receipt 552_instanseg_multiplex_gpu_2026-10-06.json retains input crops, channel controls, original and final paired labels, full profiles and raw terminal logs. Brightfield remains open. Home retains CPU/Qt/CI source ownership; all GPU tasks remain workstation-owned and protected livecell/cellposeTIME jobs are untouched.\n'
for path in ['features/future/552_instanseg_backend.txt',
             'features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt',
             'features/325_two_sessions_one_repo_working_protocol.temp']:
    with Path(path).open('a') as stream:
        stream.write(note)
print(json.dumps({key: report[key] for key in ['one_and_three_channel_normal_mask_ingest_and_cuda_inference_accepted',
                                              'exact_final_cpu_cuda_label_parity', 'fields']}, indent=2), flush=True)
