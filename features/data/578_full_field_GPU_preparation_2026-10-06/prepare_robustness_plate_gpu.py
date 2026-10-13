from pathlib import Path
import hashlib
import json
import os
import sys

import numpy as np
import tifffile

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
checkout = Path.cwd().resolve()
sys.meta_path[:] = [f for f in sys.meta_path if not getattr(f, '__module__', '').startswith('__editable__')]
sys.path.insert(0, str(checkout))
import spacr
from spacr import core, object as objects, settings as defaults
from spacr.qt.synthetic import cellvoyager_filename
from spacr.utils import prepare_batch_for_segmentation
from spacr.seg_qc import _robustness_grid

assert Path(spacr.__file__).resolve().is_relative_to(checkout)
scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '578-robustness-plate-r1'
assert not root.exists()
root.mkdir()
source = Path('/media/carruthers/mnt3/claude/toxoplasma_projects/tutorials/refresh_2026-09-09/example_data/plate1/merged')
inputs = sorted(source.glob('*.npy'))
assert len(inputs) == 16
plate = root / 'plate'
plate.mkdir()
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
records = []
for path in inputs:
    image = np.load(path, allow_pickle=False)
    assert image.ndim == 3 and image.shape[:2] == (1994, 1994) and image.shape[-1] >= 6
    tokens = path.stem.split('_')
    well, field = tokens[1], int(tokens[2])
    export = plate / cellvoyager_filename(plate='plate1', well=well, field=field, chan=0)
    assert not export.exists()
    tifffile.imwrite(export, image[..., 0], photometric='minisblack')
    np.testing.assert_array_equal(tifffile.imread(export), image[..., 0])
    records.append({'original_path': str(path), 'original_sha256': digest(path),
                    'export': str(export), 'export_sha256': digest(export),
                    'selected_intensity_channel': 0, 'well': well, 'field': field,
                    'generated_reference_mask_planes_excluded': True})
recipe = defaults.set_default_settings_preprocess_generate_masks(dict(
    src=str(plate), metadata_type='cellvoyager', channels=[0], nucleus_channel=0,
    cell_channel=None, pathogen_channel=None, organelle_channel=None,
    preprocess=True, masks=False, plot=False, verbose=False, n_jobs=1,
    test_mode=False, randomize=False, batch_size=1, normalize=True,
    nucleus_model_name='cpsam', nucleus_diameter=None, nucleus_min_size=10,
    nucleus_flow_threshold=0.4, nucleus_cellprob_threshold=0.0,
    segmentation_backend='cellpose', robustness_report=True,
    robustness_fields=16, robustness_crop=0, robustness_tolerance=0.2,
    robustness_diameter_factors=[0.75, 1.25], robustness_flow_thresholds=[0.2, 0.6],
    robustness_cellprob_thresholds=[-2.0, 6.0], robustness_enhancement=True,
    seg_qc='off'))
core.preprocess_generate_masks(recipe)
sample = objects._robustness_sample(str(plate / 'masks'), recipe, 'nucleus')
assert len(sample) == 16 and all(image.shape == (1994, 1994, 1) for _, image in sample)
arrays = {}
for path in sorted((plate / 'masks').glob('*.npz')):
    with np.load(path, allow_pickle=False) as batch:
        for name, image in zip(batch['filenames'], batch['data']):
            assert str(name) not in arrays and image.shape == (1994, 1994, 1)
            arrays[str(name)] = prepare_batch_for_segmentation(image[None])[0]
assert set(name for name, _ in sample) == set(arrays)
for name, image in sample:
    np.testing.assert_array_equal(image, arrays[name])
    np.save(root / (name + '-normalized-intensity.npy'), image)
grid = _robustness_grid(recipe, 'nucleus')
assert len(grid) == 8 and grid[0]['parameter'] == 'baseline'
assert any(p['parameter'] == 'cellprob_threshold' and p['cellprob_threshold'] == 6 for p in grid)
(root / 'recipe.json').write_text(json.dumps(recipe, indent=2, default=str) + '\n')
sources = ['spacr/core.py', 'spacr/object.py', 'spacr/seg_qc.py', 'spacr/settings.py',
           'spacr/utils.py', 'spacr/accelerator.py', 'spacr/qt/detect_chain.py',
           'spacr/figures/style.py', 'spacr/plot.py', 'spacr/tabular.py']
plan = {'prepared': True, 'inputs': records, 'field_count': 16, 'wells': sorted({r['well'] for r in records}),
        'whole_1994_by_1994_fields_no_crop_or_simulated_pixels': True,
        'scope': 'All sixteen fields of the acquired four-well example plate, not a complete production screen',
        'normal_preprocess_and_independent_normalized_pixel_readback_passed': True,
        'grid': grid, 'planned_real_GPU_calls': 16 * len(grid),
        'recipe_sha256': digest(root / 'recipe.json'),
        'source_sha256': {p: digest(p) for p in sources},
        'prepared_script_sha256': digest(__file__),
        'GPU_execution_or_fragility_result_claimed': False}
(root / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
print('PASS: all sixteen full acquired fields prepared through normal Mask ingest, exact selected-channel pixels and complete normal sampling verified; 128 real GPU grid calls remain pending.', flush=True)
