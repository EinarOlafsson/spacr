"""Generate source-bound interoperability files from the existing awkward mask."""
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import roifile
import tifffile

from spacr import mask_io


ROOT = Path('/mnt/wd4tb/scratch/spacr-completion/worktree')
OUT = Path('/mnt/wd4tb/scratch/spacr-completion/545-native-interop-preparation')
fixture_source = ROOT / 'tests/test_545_roi_export_import.py'
tree = ast.parse(fixture_source.read_text())
function = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == '_awkward_mask')
namespace = {'np': np}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(fixture_source), 'exec'), namespace)
cells = namespace['_awkward_mask']()
nuclei = np.where(cells == 7, 2, 0).astype(np.uint16)
masks = {'cell': cells, 'nucleus': nuclei}
classes = {'cell': {3: 'infected', 9: 'dividing'}}
yy, xx = np.indices(cells.shape)
image = (20 + xx + yy + 60 * (cells != 0)).astype(np.uint8)
tifffile.imwrite(OUT / 'field.tif', image)
for kind, mask in masks.items():
    tifffile.imwrite(OUT / f'expected-{kind}.tif', mask)
for fmt, suffix in [('geojson', '.geojson'), ('imagej', '.zip'), ('coco', '.json')]:
    target = OUT / ('spacr-field' + suffix)
    mask_io.export_rois(masks, target, fmt, classes=classes, file_name='field.tif')
    back, got_classes = mask_io.import_rois(target, cells.shape, fmt, with_classes=True)
    assert all(np.array_equal(back[kind], mask) for kind, mask in masks.items())
    assert got_classes['cell'][3] == 'infected'
    assert got_classes['cell'][9] == 'dividing'
rois = {roi.name: roi for roi in roifile.roiread(OUT / 'spacr-field.zip')}
objects = []
for kind, mask in masks.items():
    for number in sorted(int(value) for value in np.unique(mask) if value):
        rows, cols = np.where(mask == number)
        name = f'{kind}-{number}'
        roi = rois[name]
        objects.append({
            'name': name, 'object_type': kind, 'object_id': number,
            'class': classes.get(kind, {}).get(number, kind),
            'bounds_xywh': [int(cols.min()), int(rows.min()),
                            int(cols.max() - cols.min() + 1),
                            int(rows.max() - rows.min() + 1)],
            'area_pixels': int(len(rows)),
            'native_imagej_selection_type': 9 if roi.shape_roi_size else 4,
        })
receipt = {
    'schema': 'spacr.545.gui_interop_fixture.v1',
    'fixture_is_synthetic': True,
    'fixture_function': 'tests/test_545_roi_export_import.py::_awkward_mask',
    'shape_yx': list(cells.shape), 'expected_roi_count': len(objects),
    'objects': objects, 'spacr_round_trips_exact': ['geojson', 'imagej', 'coco'],
    'python': sys.version, 'roifile_version': importlib.metadata.version('roifile'),
    'tifffile_version': importlib.metadata.version('tifffile'),
    'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in [fixture_source, ROOT / 'spacr/mask_io.py']},
}
(OUT / 'fixture-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'roi_count': len(objects), 'shape': list(cells.shape),
                  'round_trips': receipt['spacr_round_trips_exact']}))
