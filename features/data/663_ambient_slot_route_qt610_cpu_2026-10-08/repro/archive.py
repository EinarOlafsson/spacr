import gzip
import hashlib
import json
import subprocess
from pathlib import Path

root = Path.cwd()
scratch = Path('/mnt/wd4tb/scratch/ambient-slot-610-20261008')
out = root / 'features/data/663_ambient_slot_route_qt610_cpu_2026-10-08'
out.mkdir(parents=True, exist_ok=True)
paths = ['spacr/qt/widgets/ambient.py', 'spacr/qt/preferences.py', 'spacr/qt/app.py',
         'spacr/qt/widgets/glass.py', 'spacr/qt/screens/app_screen.py']
bindings = {}


def write(name, data):
    destination = out / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)


for path in paths:
    data = (root / path).read_bytes()
    bindings[path] = hashlib.sha256(data).hexdigest()
    write('source/' + path + '.gz', gzip.compress(data, mtime=0))
for name in ['probe-setup.log', 'probe-final.log', 'probe-accepted.log']:
    write('logs/' + name + '.gz', gzip.compress((scratch / name).read_bytes(), mtime=0))
for name in ['probe.py', 'archive.py']:
    write('repro/' + name, (scratch / name).read_bytes())
report = json.loads((scratch / 'receipt.json').read_text())
report['capture_limits'] = {'max_qt_messages': 256, 'captured_qt_messages': len(report['messages']),
                            'qt_capture_truncated': len(report['messages']) == 256,
                            'producer_tracking': 'Nine producers sampled at the 17 explicit stage snapshots; intermediate theme-switch producers are not exhaustively enumerated.'}
write('probe-receipt.json', (json.dumps(report, indent=2) + '\n').encode())
receipt = {
    'feature': 'N663', 'scope': 'Missing AmbientWidget slot route; bounded negative reproduction, no production repair.',
    'checkpoint': report['checkpoint'], 'current_bindings': bindings,
    'command': "CUDA_VISIBLE_DEVICES='' SPACR_DEVICE=cpu QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg PYTHONPATH=. tools/run_capped.sh 4G /home/olafsson/anaconda3/bin/python /mnt/wd4tb/scratch/ambient-slot-610-20261008/probe.py > /mnt/wd4tb/scratch/ambient-slot-610-20261008/probe-accepted.log 2>&1",
    'environment': {'python': report['python'], 'executable': report['executable'], 'Qt': report['Qt'],
                    'numpy': report['numpy'], 'cap': '4G', 'CUDA_VISIBLE_DEVICES': '', 'QT_QPA_PLATFORM': 'offscreen',
                    'preferences': 'Temporary INI file; private home/network/log directories; Spaceout explicitly enabled for the offered spinn theme.'},
    'accepted_probe': {'exit_code': 0, 'stage_snapshots': len(report['records']), 'seconds_after_setup_to_shutdown': report['records'][-1]['seconds'],
                       'preferences_apply_keep_cycles': 3, 'independent_popup_open_close_cycles': 3,
                       'hide_restart_cycles': 3, 'density': [1, 2, 3], 'detail': [1, 2, 1],
                       'navigation': ['measure', 'make_masks', '__home__'], 'python_exceptions': 0,
                       'missing_slot_or_wrapper_messages': 0, 'qt_message_count': len(report['messages']),
                       'qt_capture_truncated': False, 'slot_index_at_each_sample': 33,
                       'native_window_destroyed': True, 'remaining_ambient_widgets': 0,
                       'sampled_producers': 9, 'sampled_producers_still_alive': 0},
    'static_review': [
        'AmbientWidget binds QTimer.timeout to the ordinary bound _on_tick method; all sampled real Qt6.10 wrappers/metaobjects retain AmbientWidget and a valid dynamic slot.',
        'Destroyed retirement captures only the producer box, not the widget. Stop retires workers and timer; independent popup replacement stops and deletes the old backdrop.',
        'The existing late-caption ChildAdded observer defers the watched host and does not retain event.child() during child construction, preserving the earlier wrapper-poisoning repair.',
        'No evidence warrants a speculative @Slot decorator, timer disconnect, event-filter or lifetime change.',
    ],
    'historical_probe_harnesses': [
        {'log': 'logs/probe-setup.log.gz', 'exit_code': 1, 'outcome': 'Initial setup selected Spaceout-only spinn in ordinary mode and was correctly refused before GUI creation; not an application failure or accepted lifecycle run.'},
        {'log': 'logs/probe-final.log.gz', 'exit_code': 1, 'outcome': 'The first full lifecycle run reached native shutdown without missing-slot warnings, then the diagnostic called nonexistent _FrameProducer.join. Final harness removes that artificial join and observes natural retirement; not a product defect.'},
    ],
    'limits': [
        'Historical AttributeError: Slot AmbientWidget:: not found was not reproduced; its original exact PID/event sequence/source was unavailable. Cause remains unresolved.',
        'No causal relationship to historical painter warnings or hosted Make Masks native faults is established.',
        'This is one bounded source/version-specific CPU offscreen run, not broad stress, user native-display, whole Qt, native4K FPS or GPU acceptance.',
        'Nine sampled producer references retired; intermediate replacement workers were not exhaustively tracked. Zero ambient widgets remained after natural GUI deletion.',
        'Qt messages were captured without application suppression; 17 messages were only the known offscreen size-hint/font notes, below the 256-message bound.',
        'Accepted Home64 source is frozen here. This packet does not change or claim acceptance of the separately frozen workstation publication bundle.',
    ],
}
write('receipt.json', (json.dumps(receipt, indent=2) + '\n').encode())
write('README.txt', b'''N663 Qt6.10 AmbientWidget missing-slot route: bounded negative evidence

No production source was changed. The real user base interpreter is Python
3.12.4 / PySide6 6.10.0 / NumPy1.26.4, not the separate spaCR environment or
Qt6.12 overlay. The exact accepted Home64 ambient source is frozen and bound.

One accepted CPU/offscreen process exercised actual Home, three Preferences
Apply/Keep cycles, spinn/field/blobs switches, independent popup open/close,
hide/restart, Measure/Make Masks/Home navigation and native destruction. Its
17 snapshots retained AmbientWidget wrappers and a valid _on_tick slot. No
Python exception or missing-slot/wrapper warning occurred. Zero ambient
widgets and no live sampled producer references remained at shutdown.

The historical symptom remains unresolved. This negative is not a fix, a
native crash explanation, native4K performance, GPU or whole-suite acceptance.
All limitations and two original diagnostic setup/retirement mistakes are
recorded explicitly. Existing painter cleanup was not retested or modified.

Run python verify.py --git to check payloads and current relevant Git source
bindings. --frozen checks only this immutable historical checkpoint if app
source has legitimately moved. No original agent commit object is required.
''')
write('verify.py', b'''import argparse,gzip,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--git',action='store_true');p.add_argument('--frozen',action='store_true');args=p.parse_args()
root=Path(__file__).resolve().parent;repo=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=root,text=True).strip());relative=root.relative_to(repo)
def read(path):
    local=root/path
    if local.exists():return local.read_bytes()
    if args.git:return subprocess.check_output(['git','show','HEAD:'+str(relative/path)],cwd=repo)
    raise FileNotFoundError(local)
m=json.loads(read('manifest.json'));r=json.loads(read('receipt.json'));probe=json.loads(read('probe-receipt.json'))
for row in m['payloads']:
    data=read(row['path']);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],row['path']
for path,digest in r['current_bindings'].items():
    assert hashlib.sha256(gzip.decompress(read('source/'+path+'.gz'))).hexdigest()==digest,path
    if not args.frozen:
        data=subprocess.check_output(['git','show','HEAD:'+path],cwd=repo) if args.git else (repo/path).read_bytes();assert hashlib.sha256(data).hexdigest()==digest,path
assert probe['Qt']=='6.10.0' and not probe['exceptions'] and not probe['capture_limits']['qt_capture_truncated']
assert probe['alive_producers_after']==0 and probe['remaining_ambient_widgets']==0 and probe['window_native_destroyed']
assert all(w['tick_index']==33 and w['meta_class']=='AmbientWidget' for row in probe['records'] for w in row['ambient'])
print('Verified',len(m['payloads']),'payloads,',len(r['current_bindings']),'frozen source bindings;', 'current bindings checked' if not args.frozen else 'historical checkpoint only')
''')
payloads = []
for path in sorted(out.rglob('*')):
    if path.is_file() and path.name != 'manifest.json':
        data = path.read_bytes()
        payloads.append({'path': str(path.relative_to(out)), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
write('manifest.json', (json.dumps({'payloads': payloads}, indent=2) + '\n').encode())
print(json.dumps({'payloads': len(payloads), 'bytes': sum(row['bytes'] for row in payloads)}))
