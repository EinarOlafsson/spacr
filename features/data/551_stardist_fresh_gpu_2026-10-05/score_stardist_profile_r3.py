import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile
from stardist.matching import matching

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '551-stardist'
cpu = root / 'cpu-r3'
gpu = root / 'gpu-r3'
records = {device: json.loads((folder / 'benchmark.json').read_text())
           for device, folder in (('cpu', cpu), ('gpu', gpu))}
assert all(record['accepted'] for record in records.values())
assert records['cpu']['source_files'] == records['gpu']['source_files']

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def score(truth, prediction, threshold):
    return {key: value.item() if isinstance(value, np.generic) else value
            for key, value in matching(truth, prediction, thresh=threshold)._asdict().items()}

cases = []
for old, new in zip(records['cpu']['cases'], records['gpu']['cases']):
    assert old['case'] == new['case']
    paths = [folder / (old['case'] + '-labels.npy') for folder in (cpu, gpu)]
    assert [digest(path) for path in paths] == [old['labels_sha256'], new['labels_sha256']]
    a, b = [np.load(path, allow_pickle=False) for path in paths]
    result = {'case': old['case'],
              'scope': 'CPU/GPU object agreement; CPU predictions are a runtime reference, not independent biological truth.',
              'cpu_gpu_object_matching': {str(t): score(a, b, t) for t in (0.5, 0.75)}}
    if old['independent_truth']:
        truth_path = Path(old['independent_truth'])
        assert digest(truth_path) == records['cpu']['source_files'][str(truth_path)]['sha256']
        truth = tifffile.imread(truth_path)
        result['official_2d_independent_annotation_scores'] = {
            device: {str(t): score(truth, mask, t) for t in (0.5, 0.75)}
            for device, mask in (('cpu', a), ('gpu', b))}
        result['distinct_truth_objects'] = int(np.count_nonzero(np.unique(truth)))
        result['scope'] += ' This 2D case additionally has official independent annotations.'
    cases.append(result)
assert len(cases) == 6
proof = {'schema': 1, 'accepted': True,
         'scope': 'Official annotated 2D accuracy plus paired CPU/GPU mask-object agreement for six real inputs. Four acquired DNA fields and the biological 3D plant volume have no independent annotations.',
         'cases': cases, 'cpu_benchmark_sha256': digest(cpu / 'benchmark.json'),
         'gpu_benchmark_sha256': digest(gpu / 'benchmark.json')}
target = root / 'fresh-installed-scoring.json'
assert not target.exists()
target.write_text(json.dumps(proof, indent=2) + '\n')
print(json.dumps(proof, indent=2))
