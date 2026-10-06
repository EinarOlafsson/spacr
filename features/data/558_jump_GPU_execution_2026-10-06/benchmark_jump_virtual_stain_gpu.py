from pathlib import Path
import hashlib
import inspect
import json
import os
import sys
import time

import numpy as np
import torch

checkout = Path.cwd().resolve()
sys.meta_path[:] = [f for f in sys.meta_path if not getattr(f, '__module__', '').startswith('__editable__')]
sys.path.insert(0, str(checkout))
import spacr
from spacr import deep_spacr as deep
from spacr.tabular import write_table

assert Path(spacr.__file__).resolve().is_relative_to(checkout)
assert os.environ.get('SPACR_DEVICE') == 'cuda' and torch.cuda.is_available()
holder = dict(part.split('=', 1) for part in (Path.home() / '.spacr/gpu/holder').read_text().split())
ancestor = os.getpid()
while ancestor and ancestor != int(holder['pid']):
    ancestor = int(next(line.split(':', 1)[1] for line in Path(f'/proc/{ancestor}/status').read_text().splitlines() if line.startswith('PPid:')))
assert ancestor == int(holder['pid'])
root = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/558-jump-current-r1')
read = lambda p: json.loads(Path(p).read_text())
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan = read(root / 'plan.json')
freeze = read(root / 'GPU-freeze.json')
assert plan['prepared'] and freeze['benchmark_sha256'] == digest(__file__)
assert freeze['plan_sha256'] == digest(root / 'plan.json')
assert digest(freeze['cpsam_path']) == freeze['cpsam_sha256']
for path, expected in plan['source_sha256'].items():
    assert digest(path) == expected
for row in plan['original_downloads']:
    assert digest(row['path']) == row['sha256']
for row in plan['training'] + plan['held_out']:
    assert digest(row['field_path']) == row['field_sha256']
assert len(plan['training']) == 48 and len(plan['held_out']) == 20
assert {r['plate'] for r in plan['training']}.isdisjoint({r['plate'] for r in plan['held_out']})
target = root / 'cuda-r1'
target.mkdir(exist_ok=False)
progress = []
training_profile = {}
original_forward = deep._VirtualStainUNet.forward

def profile_first_forward(model, x):
    assert next(model.parameters()).device.type == 'cuda' and x.device.type == 'cuda'
    if training_profile:
        return original_forward(model, x)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
        output = original_forward(model, x)
        torch.cuda.synchronize()
    kernels = sum(event.device_type == torch.autograd.DeviceType.CUDA for event in profile.events())
    assert kernels > 0
    profile.export_chrome_trace(str(target / 'actual-first-training-forward-profile.json'))
    training_profile.update(cuda_kernel_events=kernels, input_shape=list(x.shape), model_device=str(next(model.parameters()).device))
    return output

def on_epoch(epoch, loss):
    progress.append({'epoch': epoch, 'loss': loss})
    print('actual GPU training', epoch, '/20', 'loss', loss, flush=True)

parameters = dict(plan['training_parameters'], batch_size=16, lr=1e-3, model_type='unet')
deep._VirtualStainUNet.forward = profile_first_forward
started = time.monotonic()
try:
    fitted = deep._train_virtual_stain([np.load(r['field_path'], allow_pickle=False) for r in plan['training']], progress=on_epoch, **parameters)
finally:
    deep._VirtualStainUNet.forward = original_forward
assert len(progress) == len(fitted['losses']) == 20
assert next(fitted['model'].parameters()).device.type == 'cuda'
model_path = target / 'virtual_stain_c0.pt'
deep._save_virtual_stain(fitted, model_path)
loaded = deep._load_virtual_stain(model_path, device='cuda')
assert loaded['losses'] == fitted['losses']
for name, value in fitted['model'].state_dict().items():
    assert torch.equal(value, loaded['model'].state_dict()[name])
training_seconds = time.monotonic() - started
fields = [np.load(r['field_path'], allow_pickle=False) for r in plan['held_out']]
names = [Path(r['field_path']).stem for r in plan['held_out']]
for name, field in zip(names, fields):
    pred = deep._predict_virtual_stain(loaded, field)
    assert pred.shape == (1080, 1080) and np.isfinite(pred).all()
    np.save(target / (name + '-prediction.npy'), pred)
segment = deep._vs_default_segment(device='cuda')
assert segment.name == 'cellpose'
cellpose_model = inspect.getclosurevars(segment).nonlocals['model']
assert str(cellpose_model.device).startswith('cuda')
calls = []

def recorded_segment(plane):
    index = len(calls)
    assert str(cellpose_model.device).startswith('cuda')
    if index in (0, 59):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
            labels = segment(plane)
            torch.cuda.synchronize()
        kernels = sum(event.device_type == torch.autograd.DeviceType.CUDA for event in profile.events())
        assert kernels > 0
        profile.export_chrome_trace(str(target / f'actual-cellpose-call-{index:02d}-profile.json'))
    else:
        labels = segment(plane)
        kernels = None
    assert labels.shape == (1080, 1080) and labels.dtype.kind in 'iu'
    field, kind = divmod(index, 3)
    label_path = target / f'{names[field]}-{("real", "predicted", "input_baseline")[kind]}-labels.npz'
    np.savez_compressed(label_path, labels=labels)
    calls.append({'field': names[field], 'kind': ('real', 'predicted', 'input_baseline')[kind],
                  'plane_sha256': hashlib.sha256(np.asarray(plane).tobytes()).hexdigest(),
                  'labels_path': str(label_path), 'labels_sha256': digest(label_path),
                  'objects': int(len(np.unique(labels[labels > 0]))),
                  'cuda_kernel_events': kernels, 'model_device': str(cellpose_model.device)})
    print('actual GPU Cellpose scorecard', index + 1, '/60', names[field], calls[-1]['kind'], 'objects', calls[-1]['objects'], flush=True)
    return labels

scores = deep._virtual_stain_scorecard(loaded, fields, names, recorded_segment)
assert len(calls) == 60 and len(scores) == 40
write_table(scores, target / 'actual-Cellpose-SAM-scorecard.csv')
assert digest(freeze['cpsam_path']) == freeze['cpsam_sha256']
for path, expected in plan['source_sha256'].items():
    assert digest(path) == expected
acceptance = {'actual_GPU_training_and_scorecard_complete': True,
              'training_fields': 48, 'held_out_fields': 20, 'plate_disjoint_split': True,
              'training_parameters': parameters, 'training_progress': progress,
              'training_seconds': training_seconds, 'training_profile': training_profile,
              'trained_model': str(model_path), 'trained_model_sha256': digest(model_path),
              'save_load_model_state_exact': True,
              'trained_model_inference_on_CUDA': True,
              'same_current_normal_Cellpose_default_segmenter_on_all_three_arms': True,
              'actual_Cellpose_calls': calls, 'scorecard': scores.to_dict(orient='records'),
              'summary_mean_per_field': scores.groupby('kind')[['pearson', 'ssim', 'f1_50', 'f1_75']].mean().to_dict(orient='index'),
              'source_plan_sha256': digest(root / 'plan.json'), 'benchmark_sha256': digest(__file__),
              'source_sha256': plan['source_sha256'], 'cpsam_sha256': freeze['cpsam_sha256'],
              'independent_saved_label_rescoring_pending': True,
              'biological_accuracy_reference': 'Real stain Cellpose-SAM predictions are reference, not human gold',
              'no_claim_to_reuse_historical_model_or_new_cell_line_or_four_biological_replicates': True}
(target / 'acceptance.json').write_text(json.dumps(acceptance, indent=2) + '\n')
print('COMPLETE: actual normal GPU training and Cellpose scorecard; independent saved-label verification remains separate.', acceptance['summary_mean_per_field'], flush=True)
