from pathlib import Path
import gzip
import hashlib
import json
import shutil

import numpy as np
import tifffile

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '578-robustness-plate-r1'
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan = json.loads((root / 'plan.json').read_text())
shutil.copyfile(root / 'plan.json', root / 'historical-pre-relocation-plan.json')
shutil.copyfile(root / 'GPU-freeze.json', root / 'historical-r1-GPU-freeze.json')
log = (scratch / 'robustness-plate-gpu-r1.log').read_text()
assert '[578-full-field-robustness-20261006-r1] FINISH rc=1' in log
assert 'FileNotFoundError' in log
assert not (root / 'cuda-r1').exists()
for row in plan['inputs']:
    old = Path(row['export'])
    path = old.parent / 'orig' / old.name
    assert not old.exists() and digest(path) == row['export_sha256']
    original = np.load(row['original_path'], allow_pickle=False)[..., 0]
    np.testing.assert_array_equal(tifffile.imread(path), original)
    stack = root / 'plate/stack' / Path(row['original_path']).name
    np.testing.assert_array_equal(np.load(stack, allow_pickle=False)[..., 0], original)
    row['pre_ingest_export_path'] = str(old)
    row['export'] = str(path)
plan['normal_ingest_relocated_original_TIFFs_to_orig_verified_byte_and_pixel_exact'] = True
plan['prior_GPU_attempt_failed_before_model_loading_or_inference'] = True
(root / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
old_benchmark = scratch / 'benchmark_robustness_plate_gpu.py'
new_benchmark = scratch / 'benchmark_robustness_plate_gpu_r2.py'
source = old_benchmark.read_text()
assert source.count("target = root / 'cuda-r1'") == 1
new_benchmark.write_text(source.replace("target = root / 'cuda-r1'", "target = root / 'cuda-r2'"))
model = Path.home() / '.cellpose/models/cpsam'
freeze = {'model_path': str(model), 'model_sha256': digest(model),
          'plan_sha256': digest(root / 'plan.json'), 'benchmark_sha256': digest(new_benchmark)}
(root / 'GPU-freeze.json').write_text(json.dumps(freeze, indent=2) + '\n')
dest = Path('features/data/578_full_field_GPU_relocated_inputs_2026-10-06')
dest.mkdir(exist_ok=False)
for path in (root / 'plan.json', root / 'GPU-freeze.json', root / 'historical-pre-relocation-plan.json',
             root / 'historical-r1-GPU-freeze.json', new_benchmark, Path(__file__)):
    shutil.copyfile(path, dest / path.name)
(dest / 'robustness-plate-gpu-r1.log.gz').write_bytes(gzip.compress(log.encode(), mtime=0))
report = {'item': 578, 'relocated_native_TIFF_paths_byte_and_pixel_verified': 16,
          'normal_stack_pixels_original_exact': True,
          'actual_GPU_execution_pending': True, 'prior_turn_terminal_returncode': 1,
          'prior_turn_stopped_before_model_loading_or_inference': True,
          'normal_GPU_retry_label': '578-full-field-robustness-20261006-r2',
          'source_grid_inputs_and_model_scientific_parameters_preserved': True,
          'artifacts': {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(dest.iterdir())}}
Path('features/data/578_full_field_GPU_relocated_inputs_2026-10-06.json').write_text(json.dumps(report, indent=2) + '\n')
note = '\n2026-10-06 workstation robustness preflight correction: normal GPU turn r1 is terminal rc=1 before model loading or inference. Normal Mask ingest had moved the staged TIFFs into plate/orig, while the private plan still named their pre-ingest locations. Every relocated original now matches its prior SHA256 and acquired nucleus pixels; every normal stack plane is also exact. The original plan/freeze and failed raw log are retained. Corrected frozen retry 578-full-field-robustness-20261006-r2 keeps the same sixteen full fields, model and eight-point grid, with a separate cuda-r2 output folder and unchanged normal idle/handoff rules. Receipt 578_full_field_GPU_relocated_inputs_2026-10-06.json records the path correction; no application defect, model inference or fragile verdict is claimed from r1. Home retains CPU/Qt/CI/application source, and protected jobs remain untouched.\n'
for path in ('features/future/578_segmentation_robustness_report.txt',
             'features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt',
             'features/325_two_sessions_one_repo_working_protocol.temp'):
    with Path(path).open('a') as stream:
        stream.write(note)
print('PASS: sixteen normal relocated originals and stacks preserve exact acquired pixels; separate frozen GPU retry prepared.', flush=True)
