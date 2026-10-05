import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np
import tifffile

from spacr.scorecard import match_objects

repo = Path.cwd()
scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '557-gowt1'
output = root / 'gpu-n2v-r1'
cpu = json.loads((root / 'cpu-baseline-r1/benchmark.json').read_text())
gpu = json.loads((output / 'benchmark.json').read_text())
plan = json.loads((root / 'prepared-r1/plan.json').read_text())
training = json.loads((output / 'training.json').read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

assert cpu['accepted'] and gpu['accepted']
assert gpu['mode'] == 'gpu-n2v'
assert gpu['original_inputs_labels_and_models_unchanged']
assert cpu['source_input_and_model_sha256'] == gpu['source_input_and_model_sha256']
assert cpu['segmentation_parameters'] == gpu['segmentation_parameters']
for path, expected in gpu['source_input_and_model_sha256'].items():
    assert sha(path) == expected, path
assert training == gpu['training']
assert training['device'] == training['worker_hello']['device'] == 'cuda'
assert training['epochs'] == 20 and training['planes'] == 20 and training['seed'] == 0
assert len(training['train_loss']) == len(training['val_loss']) == 20
assert training['observed_gpu_processes']
assert sha(training['checkpoint']) == training['checkpoint_sha256']
assert len(gpu['cases']) == len(plan['evaluation']) == 4
assert {row['frame'] for row in gpu['cases']} == {21, 31, 33, 39}
verification = []
for case in gpu['cases']:
    source = next(row for row in plan['evaluation'] if row['frame'] == case['frame'])
    prefix = f"n2v2-01-t{case['frame']:03d}"
    image_path = output / (prefix + '-image.npy')
    labels_path = output / (prefix + '-labels.npy')
    assert sha(image_path) == case['intensity_sha256']
    assert sha(labels_path) == case['labels_sha256']
    image = np.load(image_path, allow_pickle=False)
    labels = np.load(labels_path, allow_pickle=False)
    truth = tifffile.imread(source['segmentation'])
    assert image.shape == labels.shape == truth.shape
    assert np.isfinite(image).all() and labels.dtype.kind in 'iu'
    present = np.unique(labels[50:-50, 50:-50])
    retained = labels.copy()
    retained[~np.isin(retained, present)] = 0
    for threshold in ('0.5', '0.75'):
        scored = match_objects(truth, retained, float(threshold))
        expected = case['scores'][threshold]
        assert (scored.true_positives, scored.false_positives, scored.false_negatives) == (
            expected['tp'], expected['fp'], expected['fn'])
        assert scored.f1 == expected['f1'] and scored.n_truth == 24
    verification.append({'frame': case['frame'], 'output_hashes_and_independent_rescoring_passed': True})
for threshold in ('0.5', '0.75'):
    pooled = gpu['pooled']['n2v2'][threshold]
    for key in ('tp', 'fp', 'fn'):
        assert pooled[key] == sum(row['scores'][threshold][key] for row in gpu['cases'])
    assert pooled['f1'] == 2 * pooled['tp'] / (2 * pooled['tp'] + pooled['fp'] + pooled['fn'])
log = scratch / '557-gowt1-gpu-n2v-r1.log'
assert '[557-gowt1-normal-n2v-20epochs-r1] FINISH rc=0' in log.read_text()
artifact_dir = repo / 'features/data/557_gowt1_gpu_acceptance_2026-10-05'
assert not artifact_dir.exists()
artifact_dir.mkdir()
copies = {
    'gpu-benchmark.json': output / 'benchmark.json',
    'training.json': output / 'training.json',
    'prepared-plan.json': root / 'prepared-r1/plan.json',
    'training-diameter.json': root / 'training-diameter-r1.json',
    'benchmark_gowt1_denoising.py': scratch / 'benchmark_gowt1_denoising.py',
    'archive_gowt1_gpu_r1.py': Path(__file__),
    'n2v2-final-epoch20-seed0.ckpt': Path(training['checkpoint']),
}
for name, path in copies.items():
    (artifact_dir / name).write_bytes(path.read_bytes())
for name, path in {'gpu-turn.log.gz': log, 'worker-lines.log.gz': output / 'worker-lines.log'}.items():
    (artifact_dir / name).write_bytes(gzip.compress(path.read_bytes(), mtime=0))
with zipfile.ZipFile(artifact_dir / 'paired-outputs.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(output.glob('n2v2-*-*.npy')):
        archive.write(path, path.name)
with zipfile.ZipFile(artifact_dir / 'training-logs.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted((output / 'csv_logs').rglob('*')):
        if path.is_file():
            archive.write(path, str(path.relative_to(output)))
with zipfile.ZipFile(artifact_dir / 'source-snapshot.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for relative in ('spacr/_segmentation_backends.py', 'spacr/qt/detect_chain.py',
                     'spacr/point_spread.py', 'spacr/scorecard.py'):
        archive.write(repo / relative, relative)
receipt = {
    'schema': 1, 'item': 'F557', 'date': '2026-10-05',
    'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'accepted_gpu_training_and_inference': True,
    'second_calibrated_low_light_criterion_met': False,
    'scope': gpu['scope'],
    'comparison': {'cpu': cpu['pooled'], 'gpu_n2v': gpu['pooled']},
    'training': training,
    'segmentation_parameters': gpu['segmentation_parameters'],
    'segmentation_device': 'cpu',
    'held_out_frames_independently_rescored': verification,
    'source_input_and_model_sha256': gpu['source_input_and_model_sha256'],
    'source_input_and_model_hashes_current_cpu_and_gpu_identical': True,
    'checkpoint_selection': 'Final epoch 20, seed 0, predeclared before GPU training. No held-out selection or parameter tuning.',
    'limitations': gpu['limitations'],
    'interpretation': 'N2V2 improves the pooled score on four GOWT1 evaluation frames under the frozen setup. The frames are not four biological replicates; this is not calibrated low-light validation, official CTC scoring or a universal denoising benefit. No production default changes.',
    'artifacts': {str(path.relative_to(repo)): {'bytes': path.stat().st_size, 'sha256': sha(path)}
                  for path in sorted(artifact_dir.iterdir())},
}
destination = repo / 'features/data/557_gowt1_gpu_acceptance_2026-10-05.json'
assert not destination.exists()
destination.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'accepted': True, 'frames': len(verification), 'gpu_pooled': gpu['pooled'],
                  'artifact_bytes': sum(row['bytes'] for row in receipt['artifacts'].values())}))
