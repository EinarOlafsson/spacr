import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
args = parser.parse_args()
scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '553-omnipose-gpu-r1'
output = root / args.device
assert not output.exists()
if args.device == 'cpu':
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
else:
    assert os.environ.get('CUDA_VISIBLE_DEVICES') != ''
    holder = dict(part.split('=', 1) for part in (Path.home() / '.spacr/gpu/holder').read_text().split())
    ancestor = os.getpid()
    while ancestor and ancestor != int(holder['pid']):
        ancestor = int(next(line.split(':', 1)[1] for line in Path(f'/proc/{ancestor}/status').read_text().splitlines() if line.startswith('PPid:')))
    assert ancestor == int(holder['pid'])
    assert json.loads((root / 'cpu/acceptance.json').read_text())['accepted_execution']
sys.meta_path[:] = [finder for finder in sys.meta_path if not getattr(finder, '__module__', '').startswith('__editable__')]
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, 'tools')
from spacr import _segmentation_backends as backend
import benchmark_segmentation_strategies as benchmark

digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert benchmark.OMNIPOSE_COMMIT == '2adc9aaaacee84107ecae230b00283706f27d157'
data = benchmark.fetch_omnipose_tests(root / 'omnipose_test')
inputs = {str(p.relative_to(data)): digest(p) for p in sorted(data.rglob('*')) if p.is_file()}
assert len(inputs) == 14
if args.device == 'cuda':
    assert inputs == json.loads((root / 'cpu/acceptance.json').read_text())['inputs']
os.environ['SPACR_BACKENDS_DIR'] = str(root / 'backends')
os.environ['SPACR_DEVICE'] = args.device
os.environ['SPACR_OMNIPOSE_PROFILE_ROOT'] = str(output / 'profiling')
wrapper = scratch / 'omnipose_worker_profile.py'
environment = root / 'backends/omnipose'
backend._WORKERS['omnipose'] = backend._WorkerProcess('omnipose', str(environment), worker=str(wrapper))
original_score = benchmark.score_field
scored_fields = []
def capture_scoring(truth, predicted):
    folder = output / 'scored-labels'
    folder.mkdir(exist_ok=True)
    number = len(scored_fields) + 1
    label_path = folder / f'field-{number:02d}-prediction.npy'
    truth_path = folder / f'field-{number:02d}-reference.npy'
    np.save(label_path, predicted, allow_pickle=False)
    np.save(truth_path, truth, allow_pickle=False)
    result = original_score(truth, predicted)
    scored_fields.append({'prediction': str(label_path), 'prediction_sha256': digest(label_path),
                          'reference': str(truth_path), 'reference_sha256': digest(truth_path), 'scores': result})
    return result
benchmark.score_field = capture_scoring
try:
    code = benchmark.main(['--device', args.device, '--out', str(output), '--datasets', 'bacteria_omnipose',
                           '--strategies', 'omnipose_phase', '--plan', 'models', '--threads', '2', '--no-fetch'])
    assert code == 0
    result = json.loads((output / 'results/bacteria_omnipose__omnipose_phase.json').read_text())
    assert result['pooled']['fields'] == 7 and result['pooled']['n_truth'] == 621
    assert result.get('failure') is None
    profiles = [json.loads(path.read_text()) for path in sorted((output / 'profiling').glob('*.json'))]
    assert len(profiles) == len(scored_fields) == 7
    if args.device == 'cuda':
        assert all(p['cuda_kernel_events'] > 0 and p['actual_worker_reply_device'].startswith('cuda') for p in profiles)
    else:
        assert all(p['cuda_events'] == 0 and p['actual_worker_reply_device'] == 'cpu' for p in profiles)
    assert inputs == {str(p.relative_to(data)): digest(p) for p in sorted(data.rglob('*')) if p.is_file()}
    model_files = {str(path): digest(path) for path in sorted((environment / 'models').rglob('*')) if path.is_file()}
    assert model_files
    report = {'accepted_execution': True, 'device': args.device, 'profiles': profiles, 'pooled': result['pooled'],
              'inputs': inputs, 'upstream_commit': benchmark.OMNIPOSE_COMMIT, 'models': model_files,
              'wrapper_sha256': digest(wrapper), 'scored_fields': scored_fields,
              'benchmark_script_sha256': digest(__file__),
              'application_backend_sha256': digest('spacr/_segmentation_backends.py'),
              'benchmark_driver_sha256': digest('tools/benchmark_segmentation_strategies.py'),
              'reference_labels_from_upstream_Omnipose_not_independent_human_truth': True,
              'diameter_derived_from_reference_labels_for_reproducible_execution': True,
              'cpu_gpu_label_parity_verified': False, 'three_dimensional_path_validated': False}
    (output / 'acceptance.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: normal seven-field Omnipose Mask pipeline and actual worker profiling:', args.device, flush=True)
finally:
    benchmark.score_field = original_score
    backend._shutdown_workers()
