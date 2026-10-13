import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import tifffile

sys.path.insert(0, str(Path.cwd()))
from spacr import _segmentation_backends as backend
from spacr.qt.detect_chain import Chain, prepare
from spacr.scorecard import match_objects

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=('cpu-baseline', 'gpu-n2v'), required=True)
args = parser.parse_args()
scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '557-gowt1'
plan_path = root / 'prepared-r1/plan.json'
diameter_path = root / 'training-diameter-r1.json'
plan = json.loads(plan_path.read_text())
size = json.loads(diameter_path.read_text())
assert plan['accepted_preparation'] and size['accepted']
assert len(plan['training']) == 20 and len(plan['evaluation']) == 4
assert size['policy'].startswith('Median of normal Cellpose3 nuclei size-model estimates')
parameters = {'channel_axis': None, 'diameter': size['diameter_px'], 'normalize': True,
              'flow_threshold': 0.4, 'cellprob_threshold': 0.0, 'min_size': 15,
              'resample': True, 'batch_size': 8, 'augment': False}
target = root / ('cpu-baseline-r1' if args.mode == 'cpu-baseline' else 'gpu-n2v-r1')
assert not target.exists()
target.mkdir()

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

if args.mode == 'cpu-baseline':
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
else:
    assert os.environ.get('CUDA_VISIBLE_DEVICES') != ''
    holder = dict(part.split('=', 1) for part in (Path.home() / '.spacr/gpu/holder').read_text().split())
    ancestor = os.getpid()
    while ancestor and ancestor != int(holder['pid']):
        status = Path(f'/proc/{ancestor}/status').read_text()
        ancestor = int(next(line.split(':', 1)[1] for line in status.splitlines() if line.startswith('PPid:')))
    assert ancestor == int(holder['pid'])

model = backend._load_backend('cellpose3', model_name='cellpose3:nuclei',
                              device='cpu', root=root / 'backends')
worker_lines = []
stop_listening = backend._listen_to_workers(lambda label, line: worker_lines.append(label + ': ' + line))

def predict(image):
    masks = model.eval([image], **parameters)[0]
    assert len(masks) == 1 and masks[0].shape == image.shape
    assert masks[0].dtype.kind in 'iu'
    return masks[0]

def source_inventory():
    files = [Path(__file__).resolve(), plan_path, diameter_path,
             Path(backend.__file__), Path('spacr/qt/detect_chain.py'),
             Path('spacr/point_spread.py'), Path('spacr/scorecard.py')]
    for row in plan['training'] + plan['evaluation']:
        for key, value in row.items():
            if key.endswith('_sha256') and key[:-7] in row:
                source = Path(row[key[:-7]])
                assert source.is_file() and digest(source) == value
                files.append(source)
        files.append(Path(row['exact_pixel_export']))
        assert digest(row['exact_pixel_export']) == row['export_sha256']
    for name, python in (('cellpose3', 'python3.10'), ('careamics', 'python3.13')):
        environment = root / 'backends' / name
        files.append(environment / 'spacr-backend.json')
        package = environment / 'lib' / python / 'site-packages' / ('cellpose' if name == 'cellpose3' else 'careamics')
        files += list(package.rglob('*.py'))
    files += [path for path in (root / 'backends/cellpose3/models').rglob('*') if path.is_file()]
    return {str(path.resolve()): digest(path) for path in sorted(set(files)) if path.is_file()}

def score(case, labels):
    truth = tifffile.imread(case['segmentation'])
    assert truth.shape == labels.shape and len(np.unique(truth)) - 1 == 24
    present = np.unique(labels[50:-50, 50:-50])
    retained = labels.copy()
    retained[~np.isin(retained, present)] = 0
    scores = {}
    for threshold in (0.5, 0.75):
        matched = match_objects(truth, retained, threshold)
        assert matched.n_truth == 24
        scores[str(threshold)] = {'tp': matched.true_positives, 'fp': matched.false_positives,
                                 'fn': matched.false_negatives, 'f1': matched.f1,
                                 'precision': matched.precision, 'recall': matched.recall,
                                 'n_truth': matched.n_truth, 'n_pred': matched.n_pred}
    return {'scores': scores, 'whole_field_predictions': int(np.count_nonzero(np.unique(labels))),
            'predictions_intersecting_foi': int(np.count_nonzero(np.unique(retained))),
            'ignored_wholly_exterior_predictions': int(np.count_nonzero(np.unique(labels))) - int(np.count_nonzero(np.unique(retained)))}

def pooled(records):
    result = {}
    for arm in sorted({row['arm'] for row in records}):
        fields = [row for row in records if row['arm'] == arm]
        assert len(fields) == 4
        result[arm] = {}
        for threshold in ('0.5', '0.75'):
            totals = {key: sum(row['scores'][threshold][key] for row in fields) for key in ('tp', 'fp', 'fn')}
            totals['f1'] = 2 * totals['tp'] / (2 * totals['tp'] + totals['fp'] + totals['fn'])
            result[arm][threshold] = totals
    return result

try:
    first = np.load(plan['evaluation'][0]['exact_pixel_export'], allow_pickle=False).astype(np.float32)
    start = time.perf_counter()
    predict(first)
    warmup = {'segmentation_cpu_seconds': time.perf_counter() - start}
    restoration = None
    if args.mode == 'cpu-baseline':
        restoration = backend._restoration_plan('denoise_nuclei', size['diameter_px'],
                                               root=root / 'backends', device='cpu')
        start = time.perf_counter()
        backend._restore_plane(first, restoration)
        warmup['restoration_cpu_seconds'] = time.perf_counter() - start
    frozen = source_inventory()
    training = None
    if args.mode == 'gpu-n2v':
        baseline = json.loads((root / 'cpu-baseline-r1/benchmark.json').read_text())
        assert baseline['accepted'] and baseline['source_input_and_model_sha256'] == frozen
        assert baseline['segmentation_parameters'] == parameters
        training_images = [np.load(row['exact_pixel_export'], allow_pickle=False) for row in plan['training']]
        checkpoint = target / 'n2v2-final-epoch20-seed0.ckpt'
        assert plan['predeclared_n2v']['epochs'] == 20 and plan['predeclared_n2v']['seed'] == 0
        training = backend._n2v_train(training_images, checkpoint, epochs=20, seed=0,
                                      device='cuda', root=root / 'backends')
        assert training['device'] == 'cuda' and training['epochs'] == 20 and training['planes'] == 20
        gpu_worker = backend._n2v_worker(root / 'backends')
        assert gpu_worker.hello['device'] == 'cuda'
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True)
        assert str(gpu_worker._proc.pid) in [line.split(',')[0].strip() for line in processes.splitlines()]
        training['observed_gpu_processes'] = processes.strip()
        training['worker_hello'] = gpu_worker.hello
        training['checkpoint_sha256'] = digest(checkpoint)
        write(target / 'training.json', training)
        start = time.perf_counter()
        backend._n2v_denoise(first, checkpoint, device='cuda', root=root / 'backends')
        warmup['n2v_gpu_inference_seconds'] = time.perf_counter() - start
    arms = ('raw', 'nlm', 'cellpose3_denoise_nuclei') if args.mode == 'cpu-baseline' else ('n2v2',)
    records = []
    for arm in arms:
        for case in plan['evaluation']:
            original = np.load(case['exact_pixel_export'], allow_pickle=False)
            image = original.astype(np.float32)
            start = time.perf_counter()
            provenance = {}
            if arm == 'nlm':
                image = prepare(image, Chain(denoise='nlm', denoise_strength=1.0), strict=True)
            elif arm == 'cellpose3_denoise_nuclei':
                image, provenance = backend._restore_plane(image, restoration)
            elif arm == 'n2v2':
                image = backend._n2v_denoise(image, checkpoint, device='cuda', root=root / 'backends')
                provenance = {'checkpoint_sha256': training['checkpoint_sha256'], 'device': 'cuda'}
            denoising_seconds = time.perf_counter() - start
            assert image.shape == original.shape and np.isfinite(image).all()
            start = time.perf_counter()
            labels = predict(image)
            segmentation_seconds = time.perf_counter() - start
            prefix = arm + '-01-t' + str(case['frame']).zfill(3)
            image_output = target / (prefix + '-image.npy')
            mask_output = target / (prefix + '-labels.npy')
            np.save(image_output, image, allow_pickle=False)
            np.save(mask_output, labels, allow_pickle=False)
            row = {'arm': arm, 'sequence': '01', 'frame': case['frame'],
                   'denoising_seconds': denoising_seconds, 'segmentation_cpu_seconds': segmentation_seconds,
                   'intensity_sha256': digest(image_output), 'labels_sha256': digest(mask_output),
                   'provenance': provenance, **score(case, labels)}
            records.append(row)
            print(row, flush=True)
    assert source_inventory() == frozen
    receipt = {'schema': 1, 'accepted': True,
               'scope': 'Additional real GOWT1 held-out-sequence denoising comparison through normal isolated adapters; not calibrated low-light evidence or universal improvement.',
               'mode': args.mode, 'segmentation_backend': 'cellpose3:nuclei', 'segmentation_device': 'cpu',
               'segmentation_parameters': parameters, 'diameter_selection': size['policy'],
               'scoring_policy': plan['scoring_policy'], 'matcher': 'spacr.scorecard.match_objects',
               'training': training, 'warmup_excluded': warmup, 'cases': records, 'pooled': pooled(records),
               'source_input_and_model_sha256': frozen, 'original_inputs_labels_and_models_unchanged': True,
               'limitations': plan['limitations']}
    write(target / 'benchmark.json', receipt)
    print('Accepted', args.mode, 'pooled scores:', receipt['pooled'], flush=True)
finally:
    for worker in list(backend._WORKERS.values()):
        worker.close()
    stop_listening()
    (target / 'worker-lines.log').write_text(''.join(worker_lines))
