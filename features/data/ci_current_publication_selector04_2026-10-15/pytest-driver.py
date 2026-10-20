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
 'tests/test_docs_media_budget.py::test_every_lesson_keeps_its_video_and_its_poster',
 'tests/test_docs_media_budget.py::test_narration_is_the_stable_mobile_clock',
]
import hashlib
pins = json.loads((root / 'protected-pins.json').read_text())
pins = {p: pin for p, pin in pins.items() if Path(p).is_file()}
pins['tests/test_docs_media_budget.py'] = hashlib.sha256(Path('tests/test_docs_media_budget.py').read_bytes()).hexdigest()
assert pins['tests/test_docs_media_budget.py'] == '5e98ea3160606fa5459063afe9d8da64ecbac85ff5269439181c3a49e5b5b256'
(root / 'actual-input-pins.json').write_text(json.dumps(pins, indent=2) + '\n')

blocked = []
def deny_network(event, args):
    if event == 'socket.getaddrinfo' or (event == 'socket.connect' and isinstance(args[1], tuple)):
        blocked.append(event)
        raise RuntimeError('External network forbidden in fixture verification')
sys.addaudithook(deny_network)
try:
    result = pytest.main([*targets, '-q', '--tb=short',
        '--junitxml=' + str(root / (mode + '.xml')), '--basetemp=' + str(root / (mode + '-tmp')),
        '-o', 'cache_dir=' + str(root / 'pytest-cache')])
finally:
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == pin for p, pin in pins.items())
    (root / (mode + '-network.json')).write_text(json.dumps({'blocked_network_events': blocked}) + '\n')
raise SystemExit(result)
