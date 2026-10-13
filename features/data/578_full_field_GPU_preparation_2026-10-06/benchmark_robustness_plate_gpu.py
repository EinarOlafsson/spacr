from pathlib import Path
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

checkout = Path.cwd().resolve()
sys.meta_path[:] = [f for f in sys.meta_path if not getattr(f, '__module__', '').startswith('__editable__')]
sys.path.insert(0, str(checkout))
import spacr
from spacr import object as objects
from spacr.seg_qc import _robustness_grid, _score_robustness

assert Path(spacr.__file__).resolve().is_relative_to(checkout)
assert os.environ.get('SPACR_DEVICE') == 'cuda' and torch.cuda.is_available()
holder = dict(part.split('=', 1) for part in (Path.home() / '.spacr/gpu/holder').read_text().split())
ancestor = os.getpid()
while ancestor and ancestor != int(holder['pid']):
    ancestor = int(next(line.split(':', 1)[1] for line in Path(f'/proc/{ancestor}/status').read_text().splitlines() if line.startswith('PPid:')))
assert ancestor == int(holder['pid'])
scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '578-robustness-plate-r1'
plan = json.loads((root / 'plan.json').read_text())
recipe = json.loads((root / 'recipe.json').read_text())
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert plan['prepared'] and plan['planned_real_GPU_calls'] == 128
assert digest(root / 'recipe.json') == plan['recipe_sha256']
for path, expected in plan['source_sha256'].items():
    assert digest(path) == expected
for row in plan['inputs']:
    assert digest(row['original_path']) == row['original_sha256']
    assert digest(row['export']) == row['export_sha256']
target = root / 'cuda-r1'
target.mkdir(exist_ok=False)
fields = objects._robustness_sample(str(root / 'plate/masks'), recipe, 'nucleus')
grid = _robustness_grid(recipe, 'nucleus')
assert len(fields) == 16 and grid == plan['grid']
for name, image in fields:
    np.testing.assert_array_equal(image, np.load(root / (name + '-normalized-intensity.npy'), allow_pickle=False))
calls = []
original = objects._robustness_segment

def recorded_segment(model, object_settings, object_type, image, point):
    index = len(calls)
    field_index, grid_index = divmod(index, len(grid))
    name, expected_image = fields[field_index]
    assert str(model.device).startswith('cuda') and object_type == 'nucleus'
    assert point == grid[grid_index]
    np.testing.assert_array_equal(image, expected_image)
    started = time.monotonic()
    if index in (0, 127):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
            labels = original(model, object_settings, object_type, image, point)
            torch.cuda.synchronize()
        profile.export_chrome_trace(str(target / f'call-{index:03d}-profile.json'))
        kernels = sum(1 for event in profile.events() if event.device_type == torch.autograd.DeviceType.CUDA)
        assert kernels > 0
    else:
        labels = original(model, object_settings, object_type, image, point)
        torch.cuda.synchronize()
        kernels = None
    assert labels.shape == (1994, 1994) and labels.dtype.kind in 'iu'
    path = target / f'call-{index:03d}-labels.npz'
    np.savez_compressed(path, labels=labels)
    calls.append({'index': index, 'field': name, 'grid_index': grid_index, 'point': point,
                  'image_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                  'labels_path': str(path), 'labels_sha256': digest(path),
                  'objects': int(len(np.unique(labels[labels > 0]))),
                  'model_device': str(model.device), 'cuda_kernel_events': kernels,
                  'seconds': time.monotonic() - started})
    print('actual normal GPU segmentation', index + 1, '/128', name, point['parameter'], point['value'], 'objects', calls[-1]['objects'], flush=True)
    return labels

objects._robustness_segment = recorded_segment
try:
    report = objects._run_robustness_report(str(root / 'plate/masks'), recipe, 'nucleus')
finally:
    objects._robustness_segment = original
assert report is not None and len(calls) == 128 and len(report) == len(grid)
replay_index = 0

def replay(image, point):
    global replay_index
    call = calls[replay_index]
    assert point == call['point'] and hashlib.sha256(image.tobytes()).hexdigest() == call['image_sha256']
    assert digest(call['labels_path']) == call['labels_sha256']
    replay_index += 1
    with np.load(call['labels_path'], allow_pickle=False) as data:
        return data['labels']

per_field, recomputed = _score_robustness(fields, replay, grid, recipe['robustness_tolerance'])
assert replay_index == 128
pd.testing.assert_frame_equal(report, recomputed)
summary_path = root / 'plate/qc/segmentation_robustness_nucleus.csv'
field_path = root / 'plate/qc/segmentation_robustness_nucleus_fields.csv'
pdf_path = root / 'plate/qc/segmentation_robustness_nucleus.pdf'
pd.testing.assert_frame_equal(pd.read_csv(summary_path).fillna({'reason': ''}), report, check_dtype=False)
pd.testing.assert_frame_equal(pd.read_csv(field_path), per_field, check_dtype=False)
assert pdf_path.stat().st_size > 1000
baseline = report[report.parameter == 'baseline'].iloc[0]
fragile = report[(report.parameter == 'cellprob_threshold') & (report.value == '6')].iloc[0]
assert not bool(baseline.fragile) and bool(fragile.fragile)
for path, expected in plan['source_sha256'].items():
    assert digest(path) == expected
models = {str(p): digest(p) for p in (Path.home() / '.cellpose/models').glob('cpsam')}
assert len(models) == 1
acceptance = {'GPU_execution_accepted': True, 'source_plan_sha256': digest(root / 'plan.json'),
              'benchmark_sha256': digest(__file__), 'source_sha256': plan['source_sha256'],
              'models_sha256': models, 'fields': 16, 'full_original_field_shape': [1994, 1994],
              'calls': calls, 'grid': grid,
              'normal_report_recomputed_exactly_from_saved_original_GPU_labels': True,
              'normal_summary_field_csv_and_pdf_verified': True,
              'deliberately_fragile_threshold_6_flagged': bool(fragile.fragile),
              'baseline_flagged_fragile': bool(baseline.fragile),
              'summary': report.to_dict(orient='records'),
              'scope': plan['scope'],
              'independent_ground_truth_full_screen_throughput_or_other_backend_acceptance_claimed': False,
              'artifacts': {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in (summary_path, field_path, pdf_path)}}
(target / 'acceptance.json').write_text(json.dumps(acceptance, indent=2) + '\n')
print('PASS: all 128 actual normal CUDA grid calls, first/last positive kernel profiles, saved-label report replay and deliberate fragile threshold verified.', flush=True)
