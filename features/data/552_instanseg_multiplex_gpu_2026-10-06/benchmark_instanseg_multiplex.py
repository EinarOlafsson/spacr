import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import tifffile

parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
parser.add_argument('--run-name')
args = parser.parse_args()
scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '552-instanseg-multiplex-r1'
root.mkdir(exist_ok=True)
output = root / (args.run_name or args.device)
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
    assert json.loads((root / 'cpu-r3/acceptance.json').read_text())['accepted_execution']
output.mkdir()
sys.meta_path[:] = [f for f in sys.meta_path if not getattr(f, '__module__', '').startswith('__editable__')]
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, 'tools')
from spacr import _segmentation_backends as backend
from spacr.core import preprocess_generate_masks
from spacr.object import generate_cellpose_masks_sam
from spacr.qt.synthetic import cellvoyager_filename
from benchmark_segmentation_strategies import base_settings

digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs = json.loads(Path('features/data/552_instanseg_cpu_acceptance_2026-10-02.json').read_text())['benchmark']['input_files']
original = next(row for row in inputs if Path(row['path']).stem == 'plate1_E01_9_1')
source = Path(original['path'])
assert digest(source) == original['sha256']
acquired = np.load(source, allow_pickle=False)
assert acquired.ndim == 3 and acquired.shape[-1] >= 6
crop = acquired[:384, :384, :3].copy()
assert crop.shape == (384, 384, 3)
np.save(output / 'acquired-intensity-crop.npy', crop, allow_pickle=False)
environment = scratch / '552-instanseg-gpu-r1/backends/instanseg'
os.environ['SPACR_BACKENDS_DIR'] = str(environment.parent)
os.environ['SPACR_DEVICE'] = args.device
os.environ['SPACR_MULTIPLEX_PROFILE_ROOT'] = str(output / 'profiling')
wrapper = scratch / 'instanseg_multiplex_worker.py'
backend._WORKERS['instanseg'] = backend._WorkerProcess('instanseg', str(environment), worker=str(wrapper))
cases = []
channel_references = []
try:
    for channel in range(3):
        plate = output / 'independent-channel-controls' / str(channel)
        plate.mkdir(parents=True)
        path = plate / cellvoyager_filename(plate='plate1', well='E01', field=9, chan=0)
        tifffile.imwrite(path, crop[..., channel])
        assert np.array_equal(tifffile.imread(path), acquired[:384, :384, channel])
        reference_settings = base_settings(plate, {'channels': [0], 'masks': False,
                                                  'nucleus_channel': 0 if channel == 0 else None,
                                                  'cell_channel': 0 if channel == 1 else None,
                                                  'pathogen_channel': 0 if channel == 2 else None})
        preprocess_generate_masks(reference_settings)
        archives = sorted((plate / 'masks').glob('*.npz'))
        assert len(archives) == 1
        with np.load(archives[0], allow_pickle=False) as archive:
            plane = archive['data'][0, ..., 0].copy()
        assert plane.shape == (384, 384) and np.isfinite(plane).all()
        channel_references.append(plane)
    normalized_reference = np.stack(channel_references, axis=-1)
    np.save(output / 'independently-preprocessed-intensity-planes.npy', normalized_reference, allow_pickle=False)
    for name, channels in (('one_channel', [0]), ('three_channels', [0, 1, 2])):
        plate = output / name
        plate.mkdir()
        raw = {}
        for channel in channels:
            filename = cellvoyager_filename(plate='plate1', well='E01', field=9, chan=channel)
            path = plate / filename
            tifffile.imwrite(path, crop[..., channel])
            assert np.array_equal(tifffile.imread(path), acquired[:384, :384, channel])
            raw[filename] = digest(path)
        settings = base_settings(plate, {'channels': channels, 'masks': False,
                                        'nucleus_channel': 0, 'cell_channel': 1 if len(channels) == 3 else None,
                                        'pathogen_channel': 2 if len(channels) == 3 else None,
                                        'nucleus_model_name': 'instanseg:fluorescence_nuclei_and_cells#nuclei',
                                        'nucleus_diameter': 86, 'nucleus_min_size': 0})
        preprocess_generate_masks(settings)
        archives = sorted((plate / 'masks').glob('*.npz'))
        assert len(archives) == 1
        with np.load(archives[0], allow_pickle=False) as archive:
            batch = archive['data'].copy()
        assert batch.shape == (1, 384, 384, len(channels))
        assert np.array_equal(batch[0], normalized_reference[..., channels])
        expected = batch[0].astype(np.float32)
        if expected.max() > 1:
            expected = expected / expected.max()
        generate_cellpose_masks_sam(str(plate / 'masks'), settings, 'nucleus')
        profiles = [json.loads(path.read_text()) for path in sorted((output / 'profiling').glob('segment-*.json'))]
        assert len(profiles) == len(cases) + 1
        profile = profiles[-1]
        assert len(profile['inputs']) == len(profile['outputs']) == 1
        actual = np.load(profile['inputs'][0]['path'], allow_pickle=False)
        assert actual.shape == expected.shape and np.array_equal(actual, expected)
        if args.device == 'cuda':
            assert profile['cuda_kernel_events'] > 0 and profile['actual_worker_reply_device'] == 'cuda'
        else:
            assert profile['cuda_events'] == 0 and profile['actual_worker_reply_device'] == 'cpu'
        masks = sorted((plate / 'masks/nucleus_mask_stack').glob('*.npy'))
        assert len(masks) == 1
        labels = np.load(masks[0], allow_pickle=False)
        assert labels.shape == (384, 384) and np.issubdtype(labels.dtype, np.integer)
        cases.append({'case': name, 'selected_acquired_intensity_planes': channels,
                      'normal_ingest_and_worker_input_exact': True, 'generated_mask_planes_sent': False,
                      'source_TIFF_hashes': raw, 'profile': profile, 'mask': str(masks[0]),
                      'mask_sha256': digest(masks[0]), 'objects_after_normal_pipeline_filters': int(len(np.unique(labels[labels > 0])))})
    assert digest(source) == original['sha256']
    report = {'accepted_execution': True, 'device': args.device, 'source': original,
              'explicitly_derived_crop_yx': [0, 384, 0, 384], 'intensity_planes': [0, 1, 2],
              'only_acquired_intensity_planes_sent_not_reference_mask_plane_5': True,
              'source_unchanged_and_all_normalized_channels_independently_verified': True, 'cases': cases,
              'preprocessing_controls': 'Three actual separate one-channel ingests retaining the same nucleus/cell/pathogen role and its normal normalization/background parameters; exact comparison with selected planes of the normal one/three-channel ingests.',
              'independent_normalized_reference_sha256': digest(output / 'independently-preprocessed-intensity-planes.npy'),
              'fixed_diameter_pixels': 86, 'no_independent_multiplex_ground_truth_or_accuracy_claim': True,
              'models': {str(p): digest(p) for p in sorted((environment / 'instanseg_models').rglob('*')) if p.is_file()},
              'benchmark_script_sha256': digest(__file__), 'worker_script_sha256': digest(wrapper),
              'source_hashes': {p: digest(p) for p in ('spacr/core.py', 'spacr/object.py', 'spacr/utils.py', 'spacr/_segmentation_backends.py')}}
    (output / 'acceptance.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: actual one/three-channel InstanSeg preprocessing, selection and Mask inference:', args.device, flush=True)
finally:
    backend._shutdown_workers()
