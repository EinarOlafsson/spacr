from pathlib import Path
import gzip
import hashlib
import json
import shutil
import zipfile

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '552-instanseg-gpu-r1'
dest = Path('features/data/552_instanseg_gpu_preparation_2026-10-06')
dest.mkdir(exist_ok=True)
read = lambda p: json.loads(p.read_text())
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
cpu = read(root / 'cpu-r2/acceptance.json')
assert cpu['accepted_execution'] and cpu['pooled']['fields'] == 4 and cpu['pooled']['n_truth'] == 223
assert len(cpu['scored_fields']) == len(cpu['profiles']) == 4
assert cpu['benchmark_script_sha256'] == digest(scratch / 'benchmark_instanseg_current.py')
assert all(profile['cuda_events'] == 0 for profile in cpu['profiles'])
assert all(digest(Path(path)) == expected for path, expected in cpu['models'].items())
for path in (root / 'installation.json', root / 'cpu-r2/acceptance.json'):
    shutil.copyfile(path, dest / ('cpu-acceptance.json' if path.name == 'acceptance.json' else path.name))
for name in ('prepare_instanseg_gpu_environment.py', 'benchmark_instanseg_current.py', 'instanseg_worker_profile.py', Path(__file__).name):
    shutil.copyfile(scratch / name, dest / name)
for name in ('instanseg-gpu-environment-r1.log', 'instanseg-current-cpu-r1.log', 'instanseg-current-cpu-r2.log'):
    (dest / (name + '.gz')).write_bytes(gzip.compress((scratch / name).read_bytes(), mtime=0))
with zipfile.ZipFile(dest / 'fresh-cpu-reference.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for folder in ('profiling', 'scored-labels', 'results'):
        for path in sorted((root / 'cpu-r2' / folder).rglob('*')):
            if path.is_file():
                archive.write(path, str(path.relative_to(root / 'cpu-r2')))
receipt = {'item': 552, 'normal_isolated_installer_complete': True,
           'installer_selected_default_torch_index': 'https://download.pytorch.org/whl/cu128',
           'actual_private_torch': '2.11.0+cu128', 'cpu_baseline_normal_mask_pipeline_passed': True,
           'acquired_fields': 4, 'saved_reference_objects': 223, 'predicted_objects': 209,
           'cpu_f1_50': cpu['pooled']['f1_50'], 'cpu_f1_75': cpu['pooled']['f1_75'],
           'reference_labels_from_Cellpose_SAM_not_independent_human_truth': True,
           'reference_label_diameters_used_for_reproducible_execution': True,
           'actual_cpu_outputs_retained_for_cuda_comparison': True,
           'cuda_inference_accepted': False,
           'queued_label': '552-instanseg-current-cuda-20261006-r1',
           'queue_scope': 'Normal four-field Mask pipeline, isolated real worker, actual torch.profiler CUDA events; no model double.',
           'brightfield_model_validated': False, 'multiplex_gpu_validated': False,
           'artifacts': {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(dest.iterdir()) if p.is_file()}}
Path('features/data/552_instanseg_gpu_preparation_2026-10-06.json').write_text(json.dumps(receipt, indent=2) + '\n')
note = '\n2026-10-06 workstation InstanSeg GPU preparation: the normal isolated installer selects the same cu128 wheel family as spaCR and completes on private torch 2.11.0+cu128/InstanSeg 0.1.1, with CUDA hidden during preparation. A fresh normal four-field nucleus Mask pipeline reproduces 209 predictions against 223 saved Cellpose-SAM reference objects, F1@0.5=0.9167 and F1@0.75=0.8657. Exact final scored masks, references, four real CPU worker profiles and model/source hashes are retained for comparison. The first CPU run is valid execution evidence but its normal driver cleans temporary masks; a second fresh run now captures the scored outputs without altering them. This is reproducible execution, not independent biological accuracy; the normal benchmark uses reference-label diameters. GPU turn 552-instanseg-current-cuda-20261006-r1 is queued through tools/gpu_turn.sh, real HOME, unchanged six-minute idle/ten-minute handoff and a 24 GiB cap. It will run the same normal pipeline with actual worker CUDA kernel evidence; no inference acceptance is claimed yet. Brightfield and multiplex GPU validation remain separate. Receipt 552_instanseg_gpu_preparation_2026-10-06.json archives completed preparation and exact CPU outputs. Home retains CPU CI/coverage/Qt/source ownership; protected livecell/cellposeTIME jobs stay untouched.\n'
for path in ('features/future/552_instanseg_backend.txt',
             'features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt',
             'features/325_two_sessions_one_repo_working_protocol.temp'):
    with Path(path).open('a') as stream:
        stream.write(note)
print('PASS: archived normal fresh InstanSeg CPU/installation preparation; queued GPU acceptance remains open.', flush=True)
