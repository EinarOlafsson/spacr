import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=('cpu', 'gpu'), required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--cpu-baseline', type=Path)
args = parser.parse_args()
sys.path.insert(0, str(Path.cwd()))
from spacr import _segmentation_backends as backend
from spacr.zstack import plan_from_settings
import tifffile

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '551-stardist'
environment = root / 'backends/stardist'
wrapper = scratch / 'stardist_worker_profile_r3.py'
package = environment / 'lib/python3.13/site-packages/stardist'
plant = Path('/media/carruthers/mnt3/spacr-f551-3d-20261002')
if args.device == 'cpu':
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and args.cpu_baseline is None
else:
    assert args.cpu_baseline and os.environ.get('CUDA_VISIBLE_DEVICES') != ''
    holder = dict(part.split('=', 1) for part in (Path.home() / '.spacr/gpu/holder').read_text().split())
    ancestor = os.getpid()
    while ancestor and ancestor != int(holder['pid']):
        status = Path(f'/proc/{ancestor}/status').read_text()
        ancestor = int(next(line.split(':', 1)[1] for line in status.splitlines() if line.startswith('PPid:')))
    assert ancestor == int(holder['pid']), 'GPU acceptance requires the actual shared scheduler'
assert not args.output.exists()
args.output.mkdir()
os.environ['SPACR_STARDIST_PROFILE_ROOT'] = str(args.output / 'profiling')
image = package / 'data/images/img2d.tif'
truth = package / 'data/images/mask2d.tif'
volume = plant / 'test_input.npy'
cases = [{'name': 'official_2d_nuclei', 'path': image, 'image': tifffile.imread(image), 'truth': truth}]
arrays = sorted((scratch / 'tutorial-measure-run-r4-current/example_data/plate1/merged').glob('*.npy'))[:4]
assert len(arrays) == 4
for path in arrays:
    plane = np.load(path, allow_pickle=False)[..., 0]
    assert plane.shape == (1994, 1994) and np.isfinite(plane).all()
    cases.append({'name': path.stem, 'path': path, 'image': plane, 'truth': None})
v = np.load(volume, allow_pickle=False)
assert v.shape == (1, 66, 116, 116, 1)
cases.append({'name': 'biological_plant_3d', 'path': volume, 'image': v[0, ..., 0], 'truth': None})
gpu_operations = set()
all_worker_lines = []
def audit(label, line):
    all_worker_lines.append(line)
    if ('Conv2D' in line or 'Conv3D' in line) and '/device:GPU:0' in line:
        gpu_operations.add(line.strip())
stop_listening = backend._listen_to_workers(audit)
worker = backend._WorkerProcess('stardist', str(environment), worker=str(wrapper))
backend._WORKERS['stardist'] = worker
assert worker.hello['device'] == args.device
model_2d = backend._load_backend('stardist', model_name='stardist:2D_versatile_fluo', device=args.device, root=root / 'backends')
model_3d = backend._load_backend('stardist', model_name='stardist:' + str(plant / 'plant-nuclei-3d'),
                               device=args.device, root=root / 'backends',
                               z_plan=plan_from_settings({'z_stack': True, 'z_segmentation_mode': 'volumetric', 'z_axis': 0, 'anisotropy': 2}))

def predict(case):
    if case['name'] == 'biological_plant_3d':
        return model_3d.eval(case['image'], do_3D=True, z_axis=0, anisotropy=2,
                             channel_axis=None, min_size=15)[0]
    return model_2d.eval([case['image']], channel_axis=None, min_size=0)[0][0]

try:
    warmup = {}
    for case in (cases[0], cases[-1]):
        started = time.perf_counter()
        predict(case)
        warmup[case['name']] = time.perf_counter() - started
    source_paths = [Path(__file__).resolve(), wrapper, Path(backend.__file__).resolve(),
                    environment / 'spacr-backend.json', truth, *[case['path'] for case in cases],
                    *sorted((plant / 'plant-nuclei-3d').glob('*')),
                    *sorted((environment / 'keras').rglob('*')), *sorted(package.rglob('*.py'))]
    source_paths = sorted(set(path for path in source_paths if path.is_file()))
    hashes = {str(path): {'sha256': digest(path), 'bytes': path.stat().st_size} for path in source_paths}
    baseline = json.loads((args.cpu_baseline / 'benchmark.json').read_text()) if args.cpu_baseline else None
    if baseline:
        assert baseline['accepted'] and baseline['source_files'] == hashes
    records = []
    for case in cases:
        started = time.perf_counter()
        labels = predict(case)
        seconds = time.perf_counter() - started
        assert labels.shape == case['image'].shape and labels.dtype.kind in 'iu' and np.any(labels > 0)
        path = args.output / (case['name'] + '-labels.npy')
        np.save(path, labels, allow_pickle=False)
        record = {'case': case['name'], 'shape': list(labels.shape), 'objects': len(np.unique(labels)) - 1,
                  'foreground': int(np.count_nonzero(labels)), 'labels_sha256': digest(path),
                  'seconds': seconds, 'independent_truth': str(case['truth']) if case['truth'] else None}
        if labels.ndim == 3:
            record['occupied_z_planes'] = int(np.count_nonzero(np.any(labels > 0, axis=(1, 2))))
        if baseline:
            old = np.load(args.cpu_baseline / path.name, allow_pickle=False)
            intersection = np.count_nonzero((old > 0) & (labels > 0))
            union = np.count_nonzero((old > 0) | (labels > 0))
            record['cpu_gpu_foreground_iou'] = float(intersection / union)
            record['cpu_gpu_labels_bit_identical'] = bool(np.array_equal(old, labels))
            assert record['cpu_gpu_foreground_iou'] >= 0.99
        records.append(record)
        print(args.device, record, flush=True)
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                         '--format=csv,noheader'], text=True).strip()
    if args.device == 'gpu':
        assert str(worker._proc.pid) in [line.split(',')[0].strip() for line in processes.splitlines()]
        print('Convolution placement before closing:', sorted(gpu_operations), flush=True)
        worker.close()
        print('Convolution placement after closing:', sorted(gpu_operations), flush=True)
        assert list((args.output / 'profiling/trace').rglob('*.xplane.pb'))
        assert list((args.output / 'profiling/hlo').glob('*gpu_after_optimizations.txt'))
    assert all(digest(path) == record['sha256'] for path, record in hashes.items())
    receipt = {'schema': 1, 'accepted': args.device == 'cpu', 'device': args.device,
               'scope': 'Fresh normally installed StarDist backend, official annotated biological 2D image, four original acquired DNA fields, and real biological plant 3D volume. 3D is model/example execution and CPU/GPU parity, not independent held-out accuracy.',
               'worker_hello': worker.hello, 'observed_gpu_processes': processes,
               'observed_gpu_convolution_placement': sorted(gpu_operations),
               'gpu_execution_verification_pending': args.device == 'gpu',
               'warmup_excluded': warmup, 'cases': records, 'source_files': hashes,
               'original_files_unchanged': True, 'annotation_scoring_pending': True}
    write(args.output / ('benchmark.json' if args.device == 'cpu' else 'inference-pending-gpu-verification.json'), receipt)
finally:
    worker.close()
    stop_listening()
    (args.output / 'worker-device-placement.log').write_text(''.join(all_worker_lines))
    (args.output / 'captured-gpu-convolution-lines.json').write_text(json.dumps(sorted(gpu_operations), indent=2) + '\n')
