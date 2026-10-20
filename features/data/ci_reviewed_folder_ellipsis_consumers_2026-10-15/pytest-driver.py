import json
import os
import sys
from pathlib import Path

import PySide6
from PySide6.QtCore import qVersion

os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
root = Path(__file__).parent
mode = sys.argv[1]
cgroup = Path('/proc/self/cgroup').read_text().strip().split('::', 1)[1]
cgroup_root = Path('/sys/fs/cgroup') / cgroup.lstrip('/')
receipt = {'pid': os.getpid(), 'cgroup': cgroup, 'memory_max': (cgroup_root / 'memory.max').read_text().strip(),
           'memory_swap_max': (cgroup_root / 'memory.swap.max').read_text().strip(),
           'cpus_allowed': sorted(os.sched_getaffinity(0)), 'pyside6': PySide6.__version__, 'qt': qVersion()}
(root / (mode + '-startup.json')).write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
sys.path.insert(0, os.getcwd())
import pytest
targets = [
 'tests/test_i18n_reviewed_ui.py::test_reviewed_ui_vocabulary_is_complete_and_pinned',
 'tests/qt/test_reviewed_folder_ellipsis_consumers.py::test_reviewed_folder_ellipsis_uses_real_persisted_language',
]
import hashlib
pins = json.loads((root / 'before-input-pins.json').read_text())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == pin for p, pin in pins.items())

blocked = []
def deny_network(event, args):
    if event == 'socket.getaddrinfo' or (event == 'socket.connect' and isinstance(args[1], tuple)):
        blocked.append(event)
        raise RuntimeError('External network forbidden in fixture verification')
sys.addaudithook(deny_network)
try:
    result = pytest.main(['-p', 'pytestqt.plugin', *targets, '-q', '--tb=short',
        '--junitxml=' + str(root / (mode + '.xml')), '--basetemp=' + str(root / (mode + '-tmp')),
        '-o', 'cache_dir=' + str(root / 'pytest-cache')])
finally:
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == pin for p, pin in pins.items())
    (root / 'after-input-pins.json').write_text(json.dumps({p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in pins}, indent=2) + '\n')
    module = sys.modules.get('i18n_reviewed_ui')
    assert module is not None and Path(module.__file__).resolve() == Path.cwd() / 'tools/i18n_reviewed_ui.py'
    (root / 'actual-authoritative-module.json').write_text(json.dumps({'name': module.__name__, 'path': str(Path(module.__file__).resolve()), 'sha256': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}, indent=2) + '\n')
    (root / (mode + '-network.json')).write_text(json.dumps({'blocked_network_events': blocked}) + '\n')
raise SystemExit(result)
