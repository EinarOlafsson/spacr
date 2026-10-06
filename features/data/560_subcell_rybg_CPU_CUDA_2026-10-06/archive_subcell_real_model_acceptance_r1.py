from pathlib import Path
import gzip
import hashlib
import json
import shutil
import numpy as np

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
base = scratch / '560-subcell-rybg-preparation-r1'
out = Path('features/data/560_subcell_rybg_CPU_CUDA_2026-10-06')
out.mkdir(exist_ok=False)
sha = lambda value: hashlib.sha256(value).hexdigest()
source_sha = sha(Path('spacr/embeddings.py').read_bytes())
reports = {}
for lane in ('CPU-mapping-parity-r2', 'CUDA-mapping-parity-r1'):
    folder = base / lane
    report = json.loads((folder / 'acceptance.json').read_text())
    assert report['passed'] and report['application_source_sha256'] == source_sha
    destination = out / lane
    destination.mkdir()
    preserved = {}
    for name, row in report['files'].items():
        path = folder / name
        raw = path.read_bytes()
        assert len(raw) == row['bytes'] and sha(raw) == row['sha256']
        packed = destination / (name + '.gz')
        packed.write_bytes(gzip.compress(raw, mtime=0))
        assert gzip.decompress(packed.read_bytes()) == raw
        preserved[name] = {'original_bytes': len(raw), 'original_sha256': sha(raw), 'packed_sha256': sha(packed.read_bytes()), 'retained': str(packed)}
    for name, comparison in report['comparisons'].items():
        original = np.load(folder / (name + '-original.npy'), allow_pickle=False)
        features = np.load(folder / (name + '-spacr-features.npy'), allow_pickle=False)
        authors = np.load(folder / (name + '-authors-features.npy'), allow_pickle=False)
        assert features.shape == authors.shape == (1, 1536)
        assert np.isfinite(features).all() and np.isfinite(authors).all()
        delta = np.abs(features - authors)
        assert float(delta.max()) == comparison['max_abs_difference'] < 1e-5
        assert float(delta.mean()) == comparison['mean_abs_difference']
        assert list(original.shape) == comparison['original_shape']
        if lane.startswith('CUDA'):
            trace = json.loads((folder / (name + '-CUDA-trace.json')).read_text())
            kernels = [event for event in trace['traceEvents'] if event.get('cat') == 'kernel' and event.get('dur', 0) > 0]
            assert kernels and report['profiles'][name]['positive_CUDA_events'] > 0
            cpu_original = base / 'CPU-mapping-parity-r2' / (name + '-original.npy')
            assert cpu_original.read_bytes() == (folder / (name + '-original.npy')).read_bytes()
    shutil.copy2(folder / 'acceptance.json', destination / 'acceptance.json')
    reports[lane] = {'receipt': str(destination / 'acceptance.json'), 'raw_files_losslessly_retained': preserved,
                    'maximum_author_difference': max(row['max_abs_difference'] for row in report['comparisons'].values())}
cpu = json.loads((base / 'CPU-mapping-parity-r2/acceptance.json').read_text())
gpu = json.loads((base / 'CUDA-mapping-parity-r1/acceptance.json').read_text())
assert cpu['complete_ordered_model_state_sha256'] == gpu['complete_ordered_model_state_sha256']
for name in ('prepare_subcell_rybg_real_model_parity_r1.py', 'prepare_subcell_rybg_real_model_parity_r2.py',
             'run_subcell_rybg_real_model_CUDA_r1.py', 'subcell-rybg-real-model-CPU-parity-r1.log',
             'subcell-rybg-real-model-CPU-parity-r2.log', 'verify_subcell_rybg_acquisition_r1.py'):
    shutil.copy2(scratch / name, out / name)
for name in ('acquisition.json', 'CUDA-mapping-parity-r1.log'):
    shutil.copy2(base / name, out / name)
gpu_log = (Path.home() / '.spacr/gpu/log').read_text()
rows = [row for row in gpu_log.splitlines() if '[560-subcell-rybg-author-parity-20261006-r1]' in row]
assert len(rows) == 4 and '360s idle and 600s' in rows[1] and 'FINISH rc=0' in rows[-1]
(out / 'closed-normal-gpu-turn.log').write_text('\n'.join(rows) + '\n')
packed_source = out / 'embeddings.py.gz'
packed_source.write_bytes(gzip.compress(Path('spacr/embeddings.py').read_bytes(), mtime=0))
report = {'scope': 'Real strict-loaded official four-channel SubCell CPU and CUDA mapping/normalization/native-geometry parity against pinned original authors model on five deterministic synthetic fixtures.',
          'passed': True, 'application_source_sha256': source_sha, 'integrated_Home_commit': '3ce39ac7b',
          'normal_gpu_turn_closed_rc': 0, 'normal_idle_seconds': 360, 'normal_gap_seconds': 600,
          'all_five_inputs_and_ordered_model_state_identical_between_CPU_and_CUDA': True,
          'checkpoint_sha256': gpu['checkpoint_sha256'], 'checkpoint_bytes': 349009018,
          'predeclared_maximum_absolute_difference_guard': 1e-5, 'reports': reports,
          'retained_failed_CPU_r1': 'Verification failed after app features were produced: private verifier used uppercase UNKNOWN instead of model_zoo.UNKNOWN. CPU r2 also corrects NumPy reference channel indexing before author parity. Application inference code was unchanged. The wrapper log excludes the uncaught traceback after finally; it is not reconstructed as a raw log.',
          'not_claimed': ['Four-stain expert human-label biological accuracy', 'Official Cell-DINO loading', 'Whole item 560 completion', 'Final CI or deployed documentation acceptance'],
          'archive_script_sha256': sha(Path(__file__).read_bytes())}
Path('features/data/560_subcell_rybg_CPU_CUDA_2026-10-06.json').write_text(json.dumps(report, indent=2) + '\n')
shutil.copy2(__file__, out / Path(__file__).name)
print('PASS independent full-output rescoring, actual CUDA traces, closed normal scheduler lifecycle and lossless archive', flush=True)
