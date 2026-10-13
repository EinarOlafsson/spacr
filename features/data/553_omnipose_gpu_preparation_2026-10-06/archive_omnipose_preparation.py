from pathlib import Path
import gzip
import hashlib
import json
import shutil
import zipfile

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '553-omnipose-gpu-r1'
dest = Path('features/data/553_omnipose_gpu_preparation_2026-10-06')
dest.mkdir(exist_ok=True)
read = lambda p: json.loads(p.read_text())
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
cpu = read(root / 'cpu/acceptance.json')
assert cpu['accepted_execution'] and cpu['pooled']['fields'] == 7 and cpu['pooled']['n_truth'] == 621
assert len(cpu['scored_fields']) == len(cpu['profiles']) == 7
assert cpu['benchmark_script_sha256'] == digest(scratch / 'benchmark_omnipose_current.py')
assert all(profile['cuda_events'] == 0 for profile in cpu['profiles'])
assert all(digest(Path(path)) == expected for path, expected in cpu['models'].items())
for path in (root / 'installation.json', root / 'cpu/acceptance.json'):
    shutil.copyfile(path, dest / ('cpu-acceptance.json' if path.name == 'acceptance.json' else path.name))
for name in ('prepare_omnipose_gpu_environment.py', 'benchmark_omnipose_current.py', 'omnipose_worker_profile.py', Path(__file__).name):
    shutil.copyfile(scratch / name, dest / name)
for name in ('omnipose-gpu-environment-r1.log', 'omnipose-current-cpu-r1.log'):
    (dest / (name + '.gz')).write_bytes(gzip.compress((scratch / name).read_bytes(), mtime=0))
with zipfile.ZipFile(dest / 'fresh-cpu-reference.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for folder in ('profiling', 'scored-labels', 'results'):
        for path in sorted((root / 'cpu' / folder).rglob('*')):
            if path.is_file():
                archive.write(path, str(path.relative_to(root / 'cpu')))
    for path in sorted((root / 'omnipose_test').rglob('*')):
        if path.is_file():
            archive.write(path, str(path.relative_to(root)))
receipt = {'item': 553, 'normal_isolated_installer_complete': True,
           'installer_selected_default_torch_index': 'https://download.pytorch.org/whl/cu128',
           'actual_private_torch': '2.11.0+cu128', 'actual_private_python': '3.12.14',
           'actual_Omnipose': '1.1.4', 'cpu_baseline_normal_mask_pipeline_passed': True,
           'acquired_fields': 7, 'saved_reference_objects': 621, 'predicted_objects': cpu['pooled']['n_pred'],
           'cpu_f1_50': cpu['pooled']['f1_50'], 'cpu_f1_75': cpu['pooled']['f1_75'],
           'reference_labels_from_upstream_Omnipose_not_independent_human_truth': True,
           'benchmark_derives_reference_diameter_but_adapter_declares_it_unsupported': True,
           'actual_cpu_outputs_and_all_14_pinned_upstream_files_retained': True,
           'cuda_inference_accepted': False,
           'queued_label': '553-omnipose-current-cuda-20261006-r1',
           'queue_scope': 'Same normal seven-field Mask pipeline, isolated real worker, actual torch.profiler CUDA evidence; no model double.',
           'three_dimensional_path_validated': False,
           'artifacts': {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(dest.iterdir()) if p.is_file()}}
Path('features/data/553_omnipose_gpu_preparation_2026-10-06.json').write_text(json.dumps(receipt, indent=2) + '\n')
note = '\n2026-10-06 workstation Omnipose GPU preparation: normal isolated installation selects supported Python 3.12.14, Omnipose 1.1.4 and private torch 2.11.0+cu128 with CUDA hidden. The same seven fields at pinned upstream commit 2adc9aaa reproduce the previous CPU comparison: 644 predictions against 621 upstream Omnipose reference objects, pooled F1@0.5=0.9518/F1@0.75=0.9265. All fourteen original input files, final scored masks, references, seven CPU profiles and model/source hashes are retained. The benchmark derives reference diameters, but the actual Omnipose adapter explicitly reports diameter unsupported; no honoured diameter setting is claimed. Upstream model-produced masks are not independent biological truth. Receipt 553_omnipose_gpu_preparation_2026-10-06.json archives completed normal installation and fresh CPU evidence. GPU label 553-omnipose-current-cuda-20261006-r1 now runs through the normal scheduler, real HOME, 24 GiB cap and unchanged six-minute idle/ten-minute handoff rules; inference/comparison acceptance remains separate until terminal. 3-D plant and broader GPU model comparisons remain open. Home retains CPU CI/coverage/Qt/source ownership and protected livecell/cellposeTIME jobs remain untouched.\n'
for path in ('features/future/553_omnipose_backend.txt', 'features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt', 'features/325_two_sessions_one_repo_working_protocol.temp'):
    with Path(path).open('a') as stream:
        stream.write(note)
print('PASS: normal fresh Omnipose CPU/installation preparation archived; CUDA acceptance remains open.', flush=True)
