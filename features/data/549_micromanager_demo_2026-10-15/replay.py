"""Drive spaCR's actual queue/acquisition using native MMCore demo devices."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--adapters', type=Path, required=True)
parser.add_argument('--config', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = args.root.resolve()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=False)
os.chdir(output)
sys.path.insert(0, str(root))
import numpy as np
import pandas as pd
import tifffile
from spacr import core, tabular
from pycromanager import start_headless, stop_headless
from mmpycorex import launcher
import pymmcore

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

allowed = {'DHub': 'DHub', 'Camera': 'DCam', 'XY': 'DXYStage', 'Z': 'DStage'}
devices = [line.split(',')[1:] for line in args.config.read_text().splitlines()
           if line.startswith('Device,')]
assert devices and all(library == 'DemoCamera' and allowed.get(label) == device
                       for label, library, device in devices)
assert {row[0] for row in devices} == set(allowed)
assert not launcher._PYMMCORES and not launcher._JAVA_HEADLESS_SUBPROCESSES
assert Path(core.__file__).resolve() == root / 'spacr/core.py'
report = {
    'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
    'source_file_sha256': sha(root / 'spacr/core.py'),
    'pid': os.getpid(),
    'backend': 'pycromanager Python headless, native pymmcore C++ MMCore',
    'versions': {name: importlib.metadata.version(name) for name in
                 ('pycromanager', 'mmpycorex', 'pymmcore', 'pymmcore-plus', 'numpy', 'tifffile')},
    'config_sha256': sha(args.config),
    'adapter_sha256': sha(args.adapters / 'libmmgr_dal_DemoCamera.so.0'),
    'declared_devices': devices,
    'fixture_only_events': True,
    'hardware_or_external_server_used': False,
    'passed': False,
}
(output / 'owned-pid.txt').write_text(str(os.getpid()) + '\n')
scope = None
try:
    print('START owned native MMCore DemoCamera backend', os.getpid(), flush=True)
    start_headless(str(args.adapters), str(args.config), python_backend=True,
                   buffer_size_mb=16, core_log_path=str(output / 'mmcore.log'))
    assert len(launcher._PYMMCORES) == 1 and not launcher._JAVA_HEADLESS_SUBPROCESSES
    settings = {
        'microscope_driver': 'pycromanager',
        'microscope_event_query': 'cell_area > 600',
        'microscope_event_table': 'cell',
        'microscope_max_events': 2,
        'microscope_timepoints': 2,
        'microscope_interval_seconds': 0,
        'microscope_stage_transform': [0.5, 0.0, 0.0, -0.5],
    }
    scope = core._microscope_open(settings, str(output))
    assert scope is launcher._PYMMCORES[0]
    assert isinstance(scope, pymmcore.CMMCore)
    loaded = list(scope.get_loaded_devices())
    report['loaded_devices'] = [
        {'label': label, 'library': scope.get_device_library(label), 'device': scope.get_device_name(label)}
        for label in loaded if label != 'Core'
    ]
    assert {r['label'] for r in report['loaded_devices']} == set(allowed)
    assert all(r['library'] == 'DemoCamera' and r['device'] == allowed[r['label']]
               for r in report['loaded_devices'])
    report['native_version_info'] = scope.get_version_info()
    report['native_api_version_info'] = scope.get_api_version_info()
    report['loaded_native_library_maps'] = [line for line in Path('/proc/self/maps').read_text().splitlines()
                                          if 'libmmgr_dal_DemoCamera.so' in line]
    assert report['loaded_native_library_maps']
    assert scope.get_xy_stage_device() == 'XY' and scope.get_focus_device() == 'Z'
    scope.set_roi(0, 0, 64, 64)
    scope.set_exposure(1.0)

    positions = pd.DataFrame([{'field': 'demo_A01', 'x': 1000.0, 'y': 2000.0, 'z': 7.5}])
    position_path = output / 'positions.csv'
    tabular.write_table(positions, position_path, canonicalise=False)
    settings['microscope_positions'] = str(position_path)
    field = output / 'field'
    (field / 'merged').mkdir(parents=True)
    (field / 'measurements').mkdir()
    np.save(field / 'merged/demo.npy', np.zeros((64, 64, 3), dtype=np.uint16))
    measured = pd.DataFrame([
        {'object_label': 1, 'cell_area': 100, 'cell_channel_0_centroid_weighted-0': 31.5, 'cell_channel_0_centroid_weighted-1': 31.5},
        {'object_label': 2, 'cell_area': 672, 'cell_channel_0_centroid_weighted-0': 45.5, 'cell_channel_0_centroid_weighted-1': 43.5},
        {'object_label': 3, 'cell_area': 900, 'cell_channel_0_centroid_weighted-0': 17.5, 'cell_channel_0_centroid_weighted-1': 17.5},
    ])
    tabular.write_database(measured, field / 'measurements/measurements.db', 'cell')
    work = output / 'spacr_watch'
    work.mkdir()
    context = {
        'settings': settings, 'positions': core._microscope_positions(settings),
        'matrix': core._microscope_matrix(settings), 'microscope': scope,
        'work': str(work), 'ledger_path': str(work / 'watch_ledger.json'),
        'ledger': {'fields': {'demo_A01': {'finished': 1}, 'demo_missing': {'finished': 2}}},
    }
    assert core._microscope_queue('demo_A01', str(field), context) == 2
    queued = context['ledger']['fields']['demo_A01']['events']
    assert [e['object'] for e in queued] == ['2', '3']
    assert [e['stage'] for e in queued] == [[1006.0, 1993.0, 7.5], [993.0, 2007.0, 7.5]]
    assert all(e['status'] == 'queued' for e in queued)
    assert core._microscope_queue('demo_missing', str(field), context) == 0
    before = json.loads(Path(context['ledger_path']).read_text())
    (output / 'queued-ledger.json').write_text(json.dumps(before, indent=2) + '\n')
    assert all(e['status'] == 'no_position' for e in before['fields']['demo_missing']['events'])
    print('PASS fixture events transformed and durably queued', flush=True)
    context['ledger'] = before
    assert core._microscope_drain(context) == 2
    after = json.loads(Path(context['ledger_path']).read_text())
    acquired = after['fields']['demo_A01']['events']
    assert all(e['status'] == 'acquired' and len(e['files']) == 2 for e in acquired)
    assert all(e['status'] == 'no_position' for e in after['fields']['demo_missing']['events'])
    actual_stage = [scope.get_x_position(), scope.get_y_position(), scope.get_position()]
    assert actual_stage == [993.0, 2007.0, 7.5], actual_stage
    report['final_native_stage_um'] = actual_stage
    report['images'] = []
    for event in acquired:
        for filename in event['files']:
            path = work / 'reimaged' / filename
            pixels = tifffile.imread(path)
            with tifffile.TiffFile(path) as tiff:
                metadata = json.loads(tiff.pages[0].description)
            assert pixels.shape == (64, 64) and pixels.dtype == np.uint16
            assert pixels.max() > pixels.min()
            assert metadata['stage_x_um'] == event['stage'][0]
            assert metadata['stage_y_um'] == event['stage'][1]
            report['images'].append({'file': str(path.relative_to(output)), 'sha256': sha(path),
                                     'shape': list(pixels.shape), 'dtype': str(pixels.dtype),
                                     'minimum': int(pixels.min()), 'maximum': int(pixels.max()),
                                     'stage_um': event['stage']})
    np.testing.assert_array_equal(tifffile.imread(work / 'reimaged' / acquired[-1]['files'][-1]),
                                  np.asarray(scope.get_image()).reshape(64, 64))
    assert len(report['images']) == 4
    assert core._microscope_drain(context) == 0
    report['acquired_events'] = acquired
    report['missing_positions_not_acquired'] = after['fields']['demo_missing']['events']
    report['repeat_drain_acquisitions'] = 0
    report['passed'] = True
    print('PASS native DemoCamera:2 queued positions,4 real native TIFFs,XYZ and ledger verified', flush=True)
except Exception:
    report['error'] = traceback.format_exc()
    print(report['error'], flush=True)
finally:
    stop_headless()
    report['cleanup_python_cores'] = len(launcher._PYMMCORES)
    report['cleanup_java_processes'] = len(launcher._JAVA_HEADLESS_SUBPROCESSES)
    report['cleanup_loaded_devices'] = list(scope.get_loaded_devices()) if scope is not None else []
    assert not launcher._PYMMCORES and not launcher._JAVA_HEADLESS_SUBPROCESSES
    (output / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print('CLEANUP owned native backend stopped', flush=True)
raise SystemExit(0 if report['passed'] else 1)
