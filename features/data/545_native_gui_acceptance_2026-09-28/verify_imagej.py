"""Verify outputs emitted by the actual Fiji-launched ImageJ GUI application."""
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import tifffile

from spacr import mask_io


ROOT = Path('/mnt/wd4tb/scratch/spacr-completion/worktree')
OUT = Path('/mnt/wd4tb/scratch/spacr-completion/545-native-interop-preparation')
fixture = json.loads((OUT / 'fixture-receipt.json').read_text())
status = dict(line.split('=', 1) for line in (OUT / 'imagej-native-status.txt').read_text().splitlines())
assert int((OUT / 'imagej-exit.txt').read_text()) == 0
assert status == {'version': '1.54p', 'count': '8', 'batch_mode': 'false'}
with (OUT / 'imagej-native-objects.csv').open() as stream:
    observed = list(csv.DictReader(stream))
expected = {row['name']: row for row in fixture['objects']}
assert len(observed) == fixture['expected_roi_count'] == 8
assert {row['name'] for row in observed} == set(expected)
objects = []
for row in observed:
    target = expected[row['name']]
    bounds = [int(row[key]) for key in ['x', 'y', 'width', 'height']]
    assert bounds == target['bounds_xywh'], (row['name'], bounds)
    assert float(row['area']) == target['area_pixels'], row
    assert int(row['selection_type']) == target['native_imagej_selection_type'], row
    assert row['class'] == target['class'], row
    assert int(row['object_id']) == target['object_id'], row
    assert row['object_type'] == target['object_type'], row
    objects.append({**target, 'native_observation': row})
native_back, native_classes = mask_io.import_rois(
    OUT / 'imagej-resaved-roiset.zip', tuple(fixture['shape_yx']), 'imagej', with_classes=True)
rasters = []
for kind in ['cell', 'nucleus']:
    wanted = tifffile.imread(OUT / f'expected-{kind}.tif')
    actual = tifffile.imread(OUT / f'imagej-rasterized-{kind}.tif')
    assert actual.shape == wanted.shape
    assert actual.dtype == wanted.dtype == np.dtype('uint16')
    difference = int(np.count_nonzero(actual != wanted))
    assert difference == 0, (kind, difference)
    assert np.array_equal(native_back[kind], wanted), kind
    for row in objects:
        if row['object_type'] == kind:
            assert native_classes[kind][row['object_id']] == row['class']
    rasters.append({'object_type': kind, 'shape_yx': list(actual.shape),
                    'dtype': str(actual.dtype), 'different_pixels': difference,
                    'max_object_id': int(actual.max())})
windows = (OUT / 'imagej-xwindows.txt').read_text()
assert '"ROI Manager"' in windows
assert '"ImageJ"' in windows
assert 'spaCR 545 source image - real ImageJ GUI (600%)' in windows
assert 'Macro Error' not in windows
assert (OUT / 'imagej-gui.png').stat().st_size > 1000
for line in (OUT / 'source-and-native-before.sha256').read_text().splitlines():
    expected_hash, name = line.split('  ', 1)
    source = Path(name) if Path(name).is_absolute() else ROOT / name
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_hash, name
cgroup = next(line.split(':', 2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines()
              if line.startswith('0::'))
cap = Path('/sys/fs/cgroup' + cgroup)
assert int((cap / 'memory.max').read_text()) == 4294967296
assert int((cap / 'memory.swap.max').read_text()) == 0
assert all(os.environ.get(key) == '' for key in ['CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'])
artifact_names = [
    'field.tif', 'spacr-field.geojson', 'spacr-field.zip', 'spacr-field.json',
    'expected-cell.tif', 'expected-nucleus.tif', 'fixture-receipt.json',
    'prepare_fixture.py', 'run-imagej-gui.sh', 'imagej-interop.ijm', 'verify_imagej.py',
    'imagej-native-objects.csv', 'imagej-resaved-roiset.zip',
    'imagej-rasterized-cell.tif', 'imagej-rasterized-nucleus.tif',
    'imagej-native-status.txt', 'imagej-native.log', 'imagej-runner.log',
    'imagej-xwindows.txt', 'imagej-gui.png', 'imagej-cgroup.txt',
    'source-and-native-before.sha256', 'source-revision.txt',
]
receipt = {
    'schema': 'spacr.545.native_imagej_interoperability.v1',
    'source_revision': (OUT / 'source-revision.txt').read_text().strip(),
    'source_sha256': fixture['source_sha256'],
    'passed': True,
    'scope': 'Actual Fiji launcher --ij1 GUI loads spaCR RoiSet and emits native object geometry, exact raster masks and a re-importable saved RoiSet',
    'native_application': {'launcher': '/home/olafsson/Fiji.app/ImageJ-linux64',
                           'mode': '--ij1 (real ImageJ GUI from installed Fiji)',
                           'observed_imagej_version': status['version'],
                           'bundled_jre': 'Zulu 8u452',
                           'ij2_gui_exercised': False, 'headless_or_batch': False},
    'fixture_is_synthetic': True,
    'native_roi_count': len(observed), 'native_objects': objects,
    'native_rasterization': rasters,
    'native_resaved_roiset_spacr_roundtrip_exact': True,
    'native_resaved_classes_preserved': True,
    'native_windows_observed': ['ImageJ', 'ROI Manager', 'spaCR 545 source image - real ImageJ GUI (600%)'],
    'screenshot_review': {'path': 'imagej-gui.png', 'reviewed': True,
                          'observed': 'Actual ImageJ toolbar and 600% source image with labelled outlines, hole/island and multipart objects. ROI Manager is partly behind the image; its full eight-row native observations are retained in CSV.'},
    'guards': {'gui_cgroup_receipt': 'imagej-cgroup.txt', 'memory_max_bytes': 4294967296,
               'swap_max_bytes': 0, 'java_heap_max_mb': 512, 'private_xvfb': True,
               'private_home_xdg_and_java_prefs': True, 'software_rendering': True,
               'hidden_gpu': True, 'thread_limit': 2, 'source_and_native_hashes_unchanged': True,
               'verification_cgroup': cgroup},
    'attempts': [{'directory': 'imagej-attempt1', 'exit': 1,
                  'note': 'Native object outputs were produced, then harness used an invalid zoom menu label; retained without GUI completion claim'},
                 {'directory': '.', 'session': 14407, 'exit': 0,
                  'note': 'Correct documented In [+] display command; native GUI/screenshot completed and application exited normally'}],
    'qupath_gui': 'Independent task owned by cellpose_contracts; not claimed by this receipt',
    'whole_item_done': False,
    'human_or_native_speaker_signoff': False,
    'omissions': ['QuPath GUI acceptance is separate and pending this receipt',
                  'This exercises installed Fiji launcher in ImageJ1 GUI mode; it does not claim ImageJ2 GUI testing',
                  'The acceptance image/masks are the existing synthetic awkward fixture, not a scientific sample or training run',
                  'No repository/ledger changes or acceptance of unrelated feature requirements'],
    'reference_docs': ['https://imagej.net/ij/developer/macro/functions.html',
                       'https://imagej.net/ij/docs/user-guide-A4booklet.pdf'],
    'artifacts': {name: {'bytes': (OUT / name).stat().st_size,
                          'sha256': hashlib.sha256((OUT / name).read_bytes()).hexdigest()}
                  for name in artifact_names},
}
(OUT / 'imagej-verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'passed': True, 'native_roi_count': len(observed),
                  'native_rasterization': rasters, 'native_resaved_roiset_exact': True}))
