import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import tifffile

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
checkout = Path.cwd().resolve()
sys.meta_path[:] = [finder for finder in sys.meta_path
                   if not getattr(finder, '__module__', '').startswith('__editable__')]
sys.path.insert(0, str(checkout))
import spacr
from spacr import convert, core

assert Path(spacr.__file__).resolve().is_relative_to(checkout)
scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
dataset = scratch / '557-gowt1/Fluo-N2DH-GOWT1'
target = scratch / '548-gowt1-watch-r1'
assert not target.exists()
target.mkdir()

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')

sources = [dataset / '01' / f't{frame:03d}.tif' for frame in range(21, 25)]
images = [tifffile.imread(path) for path in sources]
assert all(image.shape == (1024, 1024) and image.dtype == np.uint8 for image in images)
stack = np.stack(images)
raw = target / 'raw/plate1/A01'
raw.mkdir(parents=True)
export = raw / 'field01_C1.tif'
tifffile.imwrite(export, stack, metadata={'axes': 'TYX'}, photometric='minisblack')
with tifffile.TiffFile(export) as image:
    assert image.series[0].axes == 'TYX'
np.testing.assert_array_equal(tifffile.imread(export), stack)
converted = target / 'converted'
result = convert.convert_folder({'src': str(target / 'raw'), 'dst': str(converted),
                                 'z_handling': 'keep', 'preview_rows': 0})
assert result.is_complete
rows = sorted(result.rows(), key=lambda row: int(row['t']))
assert len(rows) == 4 and {int(row['t']) for row in rows} == {1, 2, 3, 4}
assert {int(row['z']) for row in rows} == {1}
assert {int(row['channel']) for row in rows} == {1}
for row, image in zip(rows, images):
    np.testing.assert_array_equal(tifffile.imread(converted / row['target']), image)
    assert row['well'] == 'A01'

diameter_path = scratch / '557-gowt1/training-diameter-r1.json'
diameter = json.loads(diameter_path.read_text())
assert diameter['accepted']
mask = {'metadata_type': 'cellvoyager', 'custom_regex': None,
        'channels': [0], 'nucleus_channel': 0, 'cell_channel': None,
        'pathogen_channel': None, 'organelle_channel': None,
        'preprocess': True, 'masks': True, 'plot': False, 'verbose': False,
        'test_mode': False, 'timelapse': True, 'n_jobs': 1,
        'adjust_cells': False, 'consolidate': False, 'save': True,
        'batch_size': 1, 'randomize': False, 'normalize': True,
        'segmentation_backend': 'cellpose', 'nucleus_model_name': 'cpsam',
        'nucleus_diameter': diameter['diameter_px'],
        'timelapse_objects': ['nucleus'], 'timelapse_mode': 'iou',
        'timelapse_displacement': 20, 'timelapse_remove_transient': False,
        'seg_qc': 'off', 'robustness_report': False}
measure = {'timelapse': True, 'channels': [0], 'nucleus_chann_dim': 0,
           'nucleus_mask_dim': 1, 'cell_mask_dim': None, 'pathogen_mask_dim': None,
           'organelle_mask_dim': None, 'cell_chann_dim': None,
           'pathogen_chann_dim': None, 'organelle_chann_dim': None,
           'nucleus_min_size': 0, 'save_png': False, 'save_arrays': False,
           'plot': False, 'save_measurements': True, 'n_jobs': 1,
           'verbose': False, 'radial_dist': False, 'homogeneity': False,
           'calculate_correlation': False, 'experiment': 'GOWT1-live-watch',
           'measure_gpu': False}
recipe = target / 'measure.json'
write(recipe, measure)
watched = target / 'partial-watch'
watched.mkdir()
shutil.copy2(converted / convert.MAP_FILENAME, watched / convert.MAP_FILENAME)
for row in rows[:3]:
    shutil.copy2(converted / row['target'], watched / row['target'])
watch = dict(mask, src=str(watched), watch_folder=True,
             watch_pipeline='mask_measure', watch_measure_settings=str(recipe),
             watch_settle_seconds=0.1, watch_poll_seconds=0.05,
             watch_idle_minutes=1 / 60)
partial = core.preprocess_generate_masks(watch)
assert not partial['done'] and not partial['failed'] and len(partial['incomplete']) == 1
ledger = json.loads(Path(partial['ledger']).read_text())
assert not ledger['fields']
assert not list((watched / 'spacr_watch').rglob('*.npy'))
assert not list((watched / 'spacr_watch').rglob('measurements.db'))
truth = dataset / '01_GT/SEG/man_seg021.tif'
assert np.count_nonzero(np.unique(tifffile.imread(truth))) == 24
source_files = ['spacr/core.py', 'spacr/convert.py', 'spacr/io.py',
                'spacr/object.py', 'spacr/measure.py', 'spacr/settings.py',
                'spacr/utils.py', 'spacr/_segmentation_backends.py',
                'spacr/accelerator.py', 'spacr/tracking.py']
source_hashes = {name: digest(checkout / name) for name in source_files
                 if (checkout / name).is_file()}
plan = {'accepted_preparation': True, 'application_source_commit':
        subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'application_source_sha256': source_hashes,
        'script_sha256': digest(__file__), 'dataset': 'CTC Fluo-N2DH-GOWT1 sequence 01',
        'source_dataset_receipt': 'features/data/557_gowt1_preparation_2026-10-05.json',
        'input_policy': 'Exact acquired t021-t024 pixels, stacked into a derived TYX TIFF; no native vendor T>1 claim.',
        'sources': [{'path': str(path), 'sha256': digest(path), 't': number + 1}
                    for number, path in enumerate(sources)],
        'derived_stack': {'path': str(export), 'sha256': digest(export),
                          'axes': 'TYX', 'shape': list(stack.shape), 'dtype': str(stack.dtype)},
        'conversion_map_sha256': digest(converted / convert.MAP_FILENAME),
        'conversion_rows': rows, 'mask_recipe': mask, 'measure_recipe': measure,
        'diameter_receipt_sha256': digest(diameter_path),
        'partial_arrival_result': partial, 'waited_with_no_analysis_or_database': True,
        'quality_reference': {'path': str(truth), 'sha256': digest(truth),
                              'gold_objects': 24, 'frame': 1,
                              'policy': 'Use the prior 50-pixel field-of-interest policy; functional parity and quality are separate.'},
        'next_acceptance': ['real CUDA batch and watch inference',
                            'four exact merged arrays and complete nucleus measurement rows',
                            'exact batch/field/combined IoU track CSVs',
                            'completed restart without model inference',
                            'interrupted collection resumes without model inference'],
        'gpu_inference_complete': False, 'batch_watch_parity_complete': False,
        'scope_remaining': ['native vendor T>1', 'raw/vendor acquisition completion',
                            'pooled batch_size>1 parity']}
write(target / 'plan.json', plan)
print('PASS: four acquired GOWT1 frames preserve every pixel through normal Convert; three arrivals wait with no analysis output. Real GPU batch/watch acceptance remains pending.', flush=True)
