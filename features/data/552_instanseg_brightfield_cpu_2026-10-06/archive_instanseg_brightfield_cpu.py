from pathlib import Path
import gzip
import hashlib
import json
import shutil
import zipfile

import numpy as np

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '552-instanseg-brightfield-r2'
read = lambda p: json.loads(p.read_text())
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
cpu = read(root / 'cpu-r2/acceptance.json')
assert cpu['accepted_execution'] and cpu['device'] == 'cpu'
assert cpu['actual_original_RGB_shape'] == [1484, 1380, 3]
assert digest(root / 'HE_example.tif') == cpu['source']['source_sha256']
for path, expected in cpu['models'].items():
    assert digest(Path(path)) == expected
for case in cpu['cases']:
    assert case['profile']['cuda_events'] == 0
    assert case['profile']['actual_worker_reply_device'] == 'cpu'
    assert digest(Path(case['labels'])) == case['labels_sha256']
    for row in case['profile']['inputs'] + case['profile']['outputs']:
        assert digest(Path(row['path'])) == row['sha256']
    labels = np.load(case['labels'], allow_pickle=False)
    assert len(np.unique(labels[labels > 0])) == case['objects']
dest = Path('features/data/552_instanseg_brightfield_cpu_2026-10-06')
dest.mkdir(exist_ok=False)
for name in ['fetch_instanseg_brightfield_example.py', 'benchmark_instanseg_brightfield.py', Path(__file__).name]:
    shutil.copyfile(scratch / name, dest / name)
shutil.copyfile(root / 'cpu-r2/acceptance.json', dest / 'cpu-acceptance.json')
for name in ['instanseg-brightfield-source-r1.log', 'instanseg-brightfield-source-r2.log',
             'instanseg-brightfield-cpu-r1.log', 'instanseg-brightfield-cpu-r2.log']:
    (dest / (name + '.gz')).write_bytes(gzip.compress((scratch / name).read_bytes(), mtime=0))
paths = [root / name for name in ['HE_example.tif', 'source-provenance.json', 'upstream-README.md',
                                 'upstream-head.json', 'upstream-tree.json']]
paths.extend(p for p in sorted((root / 'cpu-r2/profiling').rglob('*')) if p.is_file())
paths.extend(Path(case['labels']) for case in cpu['cases'])
paths.extend(p for p in sorted((root / 'cpu-r2/independent-channel-controls').rglob('*.npz')))
paths.extend(sorted((root / 'cpu-r2/normal-mask-plate/settings').glob('*')))
with zipfile.ZipFile(dest / 'original-brightfield-input-controls-and-cpu-labels.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(set(paths)):
        archive.write(path, str(path.relative_to(root)))
report = {'item': 552, 'actual_brightfield_CPU_execution_accepted': True, 'terminal_CPU_returncode': 0,
          'upstream_commit': cpu['source']['upstream_commit'], 'original_RGB_shape': cpu['actual_original_RGB_shape'],
          'source_original_bytes_and_git_blob_verified': True, 'no_cropping_or_simulated_images': True,
          'routes_and_actual_counts': {case['route']: case['objects'] for case in cpu['cases']},
          'normal_mask_ingest_all_RGB_intensity_planes_independently_verified': True,
          'RGB_intensity_slot_labels_do_not_claim_cell_or_pathogen_analysis': True,
          'native_model_pixel_size_requested_without_reference_derived_diameter': True,
          'no_independent_biological_accuracy_claimed': True,
          'prior_source_path_and_unconfigured_RGB_slot_fixture_failures_retained': True,
          'GPU_acceptance_pending': True,
          'next_normal_GPU_label_planned_after_installation_narration': '552-instanseg-brightfield-cuda-20261006-r1',
          'artifacts': {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(dest.iterdir())}}
Path('features/data/552_instanseg_brightfield_cpu_2026-10-06.json').write_text(json.dumps(report, indent=2) + '\n')
note = '\n2026-10-06 workstation InstanSeg brightfield CPU preparation: acquired upstream HE_example.tif is pinned to commit 414fbc9e8039c1d049cc27871d66409084ded21c and verified by original git blob/SHA256, with no crop or simulated pixels. Its full 1484x1380 RGB field now runs through actual Make Masks load/detect (5,728 objects) and normal Mask ingest/segmentation (5,871 objects), at the model native size with no reference-derived diameter. Independent single-channel controls verify every normalized RGB plane and the actual worker request. The RGB planes occupy three configured intensity slots only; their legacy cell/pathogen slot names do not claim biological RGB identities or additional biological analyses. CPU r1 is retained as a failed fixture attempt because unassigned planes were correctly excluded by normal preprocessing; corrected CPU r2 supplies accepted evidence. No independent human labels or brightfield accuracy claim is made. Receipt 552_instanseg_brightfield_cpu_2026-10-06.json retains original image/provenance, models/source hashes, controls, original/final CPU labels and complete logs. Current active GPU turn remains 615-current-installation-narration-20261006-r1; the next planned brightfield turn is 552-instanseg-brightfield-cuda-20261006-r1 only after it finishes and the normal handoff gap. No duplicate GPU turn is launched; protected livecell/cellposeTIME jobs remain untouched and Home retains CPU/Qt/CI/source ownership.\n'
for path in ['features/future/552_instanseg_backend.txt',
             'features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt',
             'features/325_two_sessions_one_repo_working_protocol.temp']:
    with Path(path).open('a') as stream:
        stream.write(note)
print('PASS: full original brightfield CPU routes/inputs/outputs archived; CUDA acceptance remains pending', flush=True)
