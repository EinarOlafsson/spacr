"""Run the actual CPU Mask generator with a model double and scratch loader variants."""
import argparse
import ast
import contextlib
import gc
import hashlib
import io
import json
import mmap
import os
import shutil
import sqlite3
import sys
import threading
import time
import types
from pathlib import Path

from probe import ROOT, residency

REPO = Path(os.environ['SPACR_PROOF_REPO'])

sys.path.insert(0, str(REPO))
import numpy as np
import spacr.object as O
import spacr.io as IO
import spacr.utils as U

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['before', 'after'])
parser.add_argument('--plan', choices=['t', 'z'], default='t')
args = parser.parse_args()
active = []
stages = []
evaluations = []
stop_sample = threading.Event()
samples = []


def cgroup_memory():
    relative = Path('/proc/self/cgroup').read_text().strip().split('::')[-1]
    directory = Path('/sys/fs/cgroup') / relative.lstrip('/')
    stat = {}
    for line in (directory / 'memory.stat').read_text().splitlines():
        key, value = line.split()
        if key in ['anon', 'file', 'file_mapped', 'file_dirty', 'kernel']:
            stat[key] = int(value)
    return {'current': int((directory / 'memory.current').read_text()), **stat}


def snapshot(stage):
    state = {'stage': stage, 'residency_kib': residency(), 'cgroup_bytes': cgroup_memory()}
    stages.append(state)
    return state


def sample_memory():
    while not stop_sample.wait(.01):
        state = residency()
        samples.append({'rss_kib': state['Rss'], 'anon_kib': state['Anonymous'],
                        'cgroup_current_bytes': cgroup_memory()['current']})


class Model:
    """CPU-only deterministic native-volume model substitute, retaining no inputs."""
    def __init__(self, **kwargs):
        self.arguments = kwargs

    def eval(self, x, **kwargs):
        assert not isinstance(x, list) and x.ndim == 4
        assert x.flags.writeable
        volume = np.ascontiguousarray(x)
        evaluations.append({'shape': list(x.shape), 'dtype': x.dtype.str,
                            'input_sha256': hashlib.sha256(memoryview(volume)).hexdigest(),
                            'kwargs': kwargs})
        snapshot('model_eval')
        return (x[..., 0] > .55).astype(np.uint16), None, None


O.cp_models = types.SimpleNamespace(CellposeModel=Model)
original_prepare = U.prepare_batch_for_segmentation


def recorded_prepare(batch):
    result = original_prepare(batch)
    snapshot('prepared_owned_batch')
    return result


U.prepare_batch_for_segmentation = recorded_prepare
original_save = IO._save_array_atomic


def recorded_save(path, array, **kwargs):
    original_save(path, array, **kwargs)
    snapshot('saved_mask')


IO._save_array_atomic = recorded_save
text = (REPO / 'spacr/object.py').read_text()
generator = O.generate_cellpose_masks_sam
if args.mode == 'before':
    text = (ROOT / 'before_object.py').read_text()
    original = ast.parse(text)
    function = next(node for node in original.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'generate_cellpose_masks_sam')
    namespace = dict(O.__dict__)
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<frozen-original-mask-generator>', 'exec'), namespace)
    generator = namespace[function.name]
working = ROOT / ('run_' + args.plan + '_' + args.mode)
if working.exists():
    shutil.rmtree(working)
src = working / 'stack'
src.mkdir(parents=True)
shutil.copyfile(ROOT / 'large.npz', src / 'large.npz')
settings = {'src': str(src), 'cell_channel': 0, 'nucleus_channel': 1,
            'pathogen_channel': None, 'magnification': 20, 'batch_size': 2,
            'verbose': False, 'plot': False, 'save': True, 'timelapse': False,
            'n_jobs': 1, 'seg_qc': 'off', 'z_segmentation_mode': 'volumetric',
            'anisotropy': 2.0}
if args.plan == 't':
    settings.update(t_stack=True, t_axis_order='TZYX')
else:
    settings.update(z_stack=True, z_axis=0)
completed = []


def batch_done(path):
    assert not list(src.rglob('.spacr-native-mask-*'))
    completed.append(Path(path).name)
    snapshot('archive_completed_callback')


snapshot('baseline_after_imports')
thread = threading.Thread(target=sample_memory, daemon=True)
thread.start()
started = time.perf_counter()
try:
    with contextlib.redirect_stdout(io.StringIO()) as log:
        generator(str(src), settings, 'cell', on_batch_done=batch_done, run_qc=False)
finally:
    stop_sample.set()
    thread.join()
elapsed = time.perf_counter() - started
snapshot('after_generator')
gc.collect()
snapshot('after_gc')
mask_files = sorted((src / 'cell_mask_stack').glob('*.npy'))
mask_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in mask_files}
with sqlite3.connect(working / 'measurements/measurements.db') as database:
    rows = database.execute('SELECT file_name,count_type,object_count FROM object_counts ORDER BY file_name,count_type').fetchall()
result = {'mode': args.mode, 'plan': args.plan, 'batch_size': settings['batch_size'],
          'model': 'deterministic CPU substitute; no actual Cellpose compute/memory claim',
          'source_repo': str(REPO),
          'source_sha256': hashlib.sha256(text.encode()).hexdigest(),
          'archive_sha256': hashlib.sha256((ROOT / 'large.npz').read_bytes()).hexdigest(),
          'pipeline_seconds': elapsed, 'vmhwm_kib': int(Path('/proc/self/status').read_text().split('VmHWM:')[1].split()[0]),
          'stages': stages, 'sampled_peak_rss_kib': max(sample['rss_kib'] for sample in samples),
          'sampled_peak_anon_kib': max(sample['anon_kib'] for sample in samples),
          'sampled_peak_cgroup_current_bytes': max(sample['cgroup_current_bytes'] for sample in samples),
          'evaluations': evaluations, 'mask_file_sha256': mask_hashes, 'database_rows': rows,
          'completion': completed, 'pipeline_stdout': log.getvalue(),
          'private_workspace_clean': not list(src.rglob('.spacr-native-mask-*'))}
(ROOT / ('production_' + args.plan + '_' + args.mode + '.json')).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({key: value for key, value in result.items() if key not in ['stages', 'evaluations', 'pipeline_stdout']}, indent=2))
shutil.rmtree(working)
