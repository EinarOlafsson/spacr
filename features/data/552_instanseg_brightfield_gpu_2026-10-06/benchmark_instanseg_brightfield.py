import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import tifffile

parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
parser.add_argument('--run-name')
args = parser.parse_args()
scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '552-instanseg-brightfield-r2'
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
    assert json.loads((root / 'cpu-r2/acceptance.json').read_text())['accepted_execution']
output.mkdir()
sys.meta_path[:] = [f for f in sys.meta_path if not getattr(f, '__module__', '').startswith('__editable__')]
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, 'tools')
from spacr import _segmentation_backends as backend
from spacr.core import preprocess_generate_masks
from spacr.object import generate_cellpose_masks_sam
from spacr.qt.synthetic import cellvoyager_filename
from spacr.qt.screens.make_masks import load_cellpose_model, cellpose_detect
from benchmark_segmentation_strategies import base_settings

digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
provenance = json.loads((root / 'source-provenance.json').read_text())
source = root / 'HE_example.tif'
assert digest(source) == provenance['source_sha256']
image = tifffile.imread(source)
assert image.ndim == 3 and image.shape[-1] == 3 and image.dtype == np.uint8
environment = scratch / '552-instanseg-gpu-r1/backends/instanseg'
os.environ['SPACR_BACKENDS_DIR'] = str(environment.parent)
os.environ['SPACR_DEVICE'] = args.device
os.environ['SPACR_MULTIPLEX_PROFILE_ROOT'] = str(output / 'profiling')
wrapper = scratch / 'instanseg_multiplex_worker.py'
backend._WORKERS['instanseg'] = backend._WorkerProcess('instanseg', str(environment), worker=str(wrapper))
model_name = 'instanseg:brightfield_nuclei#nuclei'
try:
    model = load_cellpose_model(model_name)
    labels, probability, flow = cellpose_detect(image, model, diameter=0, min_size=0)
    assert labels.shape == image.shape[:2] and np.issubdtype(labels.dtype, np.integer)
    assert probability is None and flow is None
    np.save(output / 'make-masks-labels.npy', labels, allow_pickle=False)
    plate = output / 'normal-mask-plate'
    plate.mkdir()
    raw_hashes = {}
    for channel in range(3):
        path = plate / cellvoyager_filename(plate='plate1', well='A01', field=1, chan=channel)
        tifffile.imwrite(path, image[..., channel])
        assert np.array_equal(tifffile.imread(path), image[..., channel])
        raw_hashes[path.name] = digest(path)
    settings = base_settings(plate, {'channels': [0, 1, 2], 'masks': False, 'nucleus_channel': 0,
        'cell_channel': 1, 'pathogen_channel': 2, 'remove_background_pathogen': False,
        'pathogen_background': 100, 'pathogen_signal_to_noise': 10, 'nucleus_model_name': model_name,
        'nucleus_diameter': 0, 'nucleus_min_size': 0})
    preprocess_generate_masks(settings)
    archives = list((plate / 'masks').glob('*.npz'))
    assert len(archives) == 1
    with np.load(archives[0], allow_pickle=False) as archive:
        normalized = archive['data'].copy()
    assert normalized.shape == (1,) + image.shape
    references = []
    for channel in range(3):
        control = output / 'independent-channel-controls' / str(channel)
        control.mkdir(parents=True)
        path = control / cellvoyager_filename(plate='plate1', well='A01', field=1, chan=0)
        tifffile.imwrite(path, image[..., channel])
        control_settings = base_settings(control, {'channels': [0], 'masks': False,
            'nucleus_channel': 0, 'cell_channel': None, 'pathogen_channel': None})
        preprocess_generate_masks(control_settings)
        paths = list((control / 'masks').glob('*.npz'))
        assert len(paths) == 1
        with np.load(paths[0], allow_pickle=False) as archive:
            references.append(archive['data'][0, ..., 0].copy())
    assert np.array_equal(normalized[0], np.stack(references, axis=-1))
    generate_cellpose_masks_sam(str(plate / 'masks'), settings, 'nucleus')
    masks = list((plate / 'masks/nucleus_mask_stack').glob('*.npy'))
    assert len(masks) == 1
    profiles = [json.loads(p.read_text()) for p in sorted((output / 'profiling').glob('segment-*.json'))]
    assert len(profiles) == 2
    expected = normalized[0].astype(np.float32)
    if expected.max() > 1:
        expected /= expected.max()
    assert np.array_equal(np.load(profiles[0]['inputs'][0]['path'], allow_pickle=False), image)
    assert np.array_equal(np.load(profiles[1]['inputs'][0]['path'], allow_pickle=False), expected)
    cases = []
    for name, mask, profile in [('make_masks', output / 'make-masks-labels.npy', profiles[0]),
                                ('normal_mask_generation', masks[0], profiles[1])]:
        actual = np.load(mask, allow_pickle=False)
        count = int(len(np.unique(actual[actual > 0])))
        assert count > 0
        assert profile['model'] == 'brightfield_nuclei'
        assert profile['actual_worker_reply_device'] == args.device
        if args.device == 'cuda':
            assert profile['cuda_kernel_events'] > 0
        else:
            assert profile['cuda_events'] == 0
        cases.append({'route': name, 'profile': profile, 'labels': str(mask),
                      'labels_sha256': digest(mask), 'objects': count})
    assert digest(source) == provenance['source_sha256']
    model_root = environment / 'instanseg_models/brightfield_nuclei'
    model_files = {str(p): digest(p) for p in sorted(model_root.rglob('*')) if p.is_file()}
    assert model_files
    result = {'accepted_execution': True, 'device': args.device, 'source': provenance,
        'actual_original_RGB_shape': list(image.shape), 'crop_or_simulated_data_used': False,
        'both_current_Make_Masks_and_normal_Mask_generation_paths': True,
        'normal_preprocessing_independently_verified_per_channel': True,
        'RGB_planes_routed_as_three_intensity_slots_only_nucleus_inference_requested': True,
        'cell_pathogen_slot_names_do_not_claim_biological_RGB_channel_identity': True,
        'original_RGB_and_normal_selected_RGB_worker_inputs_exact': True,
        'source_TIFF_channels_hashes': raw_hashes, 'cases': cases, 'models': model_files,
        'diameter_0_uses_model_native_pixel_size_not_reference_derived': True,
        'no_independent_human_reference_labels_or_biological_accuracy_claimed': True,
        'benchmark_sha256': digest(__file__), 'worker_sha256': digest(wrapper),
        'source_hashes': {p: digest(p) for p in ['spacr/core.py', 'spacr/object.py', 'spacr/io.py',
            'spacr/utils.py', 'spacr/_segmentation_backends.py', 'spacr/qt/screens/make_masks.py']}}
    (output / 'acceptance.json').write_text(json.dumps(result, indent=2) + '\n')
    print('PASS: original acquired upstream brightfield example in actual Make Masks and normal Mask pipeline:', args.device,
          [(row['route'], row['objects']) for row in cases], flush=True)
finally:
    backend._shutdown_workers()
