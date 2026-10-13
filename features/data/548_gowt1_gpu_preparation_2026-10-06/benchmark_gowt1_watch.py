import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import tifffile
import torch

assert os.environ.get('CUDA_VISIBLE_DEVICES') != ''
holder = dict(part.split('=', 1) for part in
              (Path.home() / '.spacr/gpu/holder').read_text().split())
ancestor = os.getpid()
while ancestor and ancestor != int(holder['pid']):
    status = Path(f'/proc/{ancestor}/status').read_text()
    ancestor = int(next(line.split(':', 1)[1] for line in status.splitlines()
                        if line.startswith('PPid:')))
assert ancestor == int(holder['pid'])
assert torch.cuda.is_available()
checkout = Path.cwd().resolve()
sys.meta_path[:] = [finder for finder in sys.meta_path
                   if not getattr(finder, '__module__', '').startswith('__editable__')]
sys.path.insert(0, str(checkout))
import spacr
from spacr import core, convert
from spacr.measure import measure_crop
from spacr.scorecard import match_objects

assert Path(spacr.__file__).resolve().is_relative_to(checkout)
root = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/548-gowt1-watch-r1')
plan = json.loads((root / 'plan.json').read_text())
assert plan['accepted_preparation'] and plan['waited_with_no_analysis_or_database']
target = root / 'gpu-benchmark-r1'
assert not target.exists()
target.mkdir()

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')

for name, expected in plan['application_source_sha256'].items():
    assert digest(checkout / name) == expected
tracking_source = checkout / 'spacr/timelapse.py'
tracking_hash = hashlib.sha256(subprocess.check_output(
    ['git', 'show', plan['application_source_commit'] + ':spacr/timelapse.py'])).hexdigest()
assert digest(tracking_source) == tracking_hash
for record in plan['sources']:
    assert digest(record['path']) == record['sha256']
assert digest(plan['derived_stack']['path']) == plan['derived_stack']['sha256']
converted = root / 'converted'
assert digest(converted / convert.MAP_FILENAME) == plan['conversion_map_sha256']
checkpoint = Path.home() / '.cellpose/models/cpsam'
checkpoint_hash = digest(checkpoint)
batch = target / 'batch'
batch.mkdir()
for row, source in zip(plan['conversion_rows'], plan['sources']):
    incoming = converted / row['target']
    np.testing.assert_array_equal(tifffile.imread(incoming), tifffile.imread(source['path']))
    shutil.copy2(incoming, batch / incoming.name)
mask = plan['mask_recipe']
measure = plan['measure_recipe']

def profiled(label, action):
    torch.manual_seed(0)
    np.random.seed(0)
    started = time.perf_counter()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profile:
        result = action()
        torch.cuda.synchronize()
    events = [event for event in profile.events()
              if event.device_type == torch.autograd.DeviceType.CUDA]
    kernels = [event for event in events if not any(
        word in event.name.lower() for word in ['memcpy', 'memset'])]
    report = {'phase': label, 'seconds': time.perf_counter() - started,
              'cuda_events': len(events), 'cuda_kernel_events': len(kernels),
              'kernel_names': sorted({event.name for event in kernels}),
              'cuda_device': torch.cuda.get_device_name(0)}
    write(target / (label + '-cuda-execution.json'), report)
    print(label, report['seconds'], 'seconds;', len(kernels), 'actual CUDA kernel events', flush=True)
    return result, report

_, batch_profile = profiled('batch', lambda: core.preprocess_generate_masks(dict(mask, src=str(batch))))
assert batch_profile['cuda_kernel_events'] > 0
measure_crop(dict(measure, src=str(batch / 'merged')))
watched = root / 'partial-watch'
assert len(list(watched.glob('*.tif'))) == 3
last = plan['conversion_rows'][-1]['target']
shutil.copy2(converted / last, watched / (last + '.part'))
os.replace(watched / (last + '.part'), watched / last)
recipe = root / 'measure.json'
assert json.loads(recipe.read_text()) == measure
watch = dict(mask, src=str(watched), watch_folder=True, watch_pipeline='mask_measure',
             watch_measure_settings=str(recipe), watch_settle_seconds=0.1,
             watch_poll_seconds=0.05, watch_idle_minutes=1 / 60)
original_merge = core._watch_merge_database

def interrupted_collection(*args, **kwargs):
    raise OSError('Private acceptance fault: collection interrupted after actual GPU analysis')

core._watch_merge_database = interrupted_collection
try:
    interrupted, watch_profile = profiled('watch-analysis', lambda: core.preprocess_generate_masks(watch))
finally:
    core._watch_merge_database = original_merge
assert watch_profile['cuda_kernel_events'] > 0
assert len(interrupted['failed']) == 1 and not interrupted['done']
field = watched / 'spacr_watch/fields' / interrupted['failed'][0]
staged = {str(path.relative_to(field)): (digest(path), path.stat().st_mtime_ns)
          for folder in ['merged', 'tracks', 'measurements']
          for path in (field / folder).glob('*') if path.is_file()}
assert staged and any(name.startswith('tracks/') for name in staged)
resumed, resume_profile = profiled('collection-resume', lambda: core.preprocess_generate_masks(watch))
assert resumed['done'] == interrupted['failed'] and not resumed['failed'] and not resumed['incomplete']
assert resume_profile['cuda_events'] == 0
assert staged == {name: (digest(field / name), (field / name).stat().st_mtime_ns) for name in staged}
combined = watched / 'spacr_watch'
names = sorted(path.name for path in (batch / 'merged').glob('*.npy'))
assert len(names) == 4
assert names == sorted(path.name for path in (combined / 'merged').glob('*.npy'))
for name in names:
    np.testing.assert_array_equal(np.load(batch / 'merged' / name), np.load(combined / 'merged' / name))

def scientific_rows(database):
    with sqlite3.connect(database) as connection:
        columns = [row[1] for row in connection.execute('PRAGMA table_info(nucleus)')]
        assert columns
        kept = [column for column in columns if 'path' not in column.lower() and column != 'file_name']
        selected = ', '.join('"' + name.replace('"', '""') + '"' for name in kept)
        records = connection.execute('SELECT ' + selected + ' FROM nucleus').fetchall()
        frames = connection.execute('SELECT DISTINCT prcf FROM nucleus ORDER BY prcf').fetchall()
    return kept, sorted(records, key=repr), frames

batch_rows = scientific_rows(batch / 'measurements/measurements.db')
watch_rows = scientific_rows(combined / 'measurements/measurements.db')
assert batch_rows == watch_rows and len(batch_rows[1]) > 0
assert batch_rows[2] == [(f'plate1_r1_c1_f1_t{index}',) for index in range(1, 5)]
tracks = sorted((batch / 'tracks').glob('trackpy_tracks_nucleus_*.csv'))
assert len(tracks) == 1
track = tracks[0]
for folder in [field, combined]:
    assert digest(track) == digest(folder / 'tracks' / track.name)
table = pd.read_csv(track)
assert sorted(table['frame'].unique().tolist()) == [0, 1, 2, 3]
assert (table.groupby('track_id')['frame'].nunique() == 4).any()
restart, restart_profile = profiled('completed-restart', lambda: core.preprocess_generate_masks(watch))
assert restart['done'] == resumed['done'] and not restart['failed'] and not restart['incomplete']
assert restart_profile['cuda_events'] == 0
assert digest(checkpoint) == checkpoint_hash
for name, expected in plan['application_source_sha256'].items():
    assert digest(checkout / name) == expected
assert digest(tracking_source) == tracking_hash
report = {'accepted': True, 'scope': 'Real CUDA fixed-map four-frame batch/watch parity at batch_size=1; derived acquired-data series.',
          'plan_sha256': digest(root / 'plan.json'), 'script_sha256': digest(__file__),
          'watched_source': str(watched),
          'source_sha256': {**plan['application_source_sha256'], 'spacr/timelapse.py': tracking_hash},
          'cpsam_sha256': checkpoint_hash,
          'torch_version': torch.__version__, 'cuda_runtime': torch.version.cuda,
          'profiles': [batch_profile, watch_profile, resume_profile, restart_profile],
          'exact_merged_frames': names, 'nucleus_measurement_columns': batch_rows[0],
          'exact_nucleus_measurement_rows': len(batch_rows[1]),
          'measurement_frame_identities': batch_rows[2],
          'exact_track_sha256': digest(track), 'track_rows': len(table),
          'partial_arrivals_waited': True, 'collection_interruption_was_real_analysis_with_injected_io_error': True,
          'collection_resume_without_inference': True, 'completed_restart_without_inference': True,
          'quality_scoring_complete': False, 'native_vendor_T_gt_1': False,
          'raw_vendor_completion_complete': False, 'pooled_batch_size_gt_1': False}
write(target / 'benchmark.json', report)
print('PASS: actual CUDA analysis, four exact batch/watch merged frames, all nucleus rows and track CSVs; interrupted collection and completed restart do not repeat inference.', flush=True)
