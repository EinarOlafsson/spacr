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
    'tests/qt/test_the_preview_uses_the_model_the_run_will_use.py::test_a_model_the_user_picked_survives_an_object_switch',
    'tests/qt/test_the_preview_uses_the_model_the_run_will_use.py::test_the_preview_offers_only_cellpose_models',
]
import socket
attempts = []
def refuse_network(*args, **kwargs):
    attempts.append('blocked')
    raise RuntimeError('Offline CI fixture reproduction forbids network')
socket.socket.connect = refuse_network
socket.socket.connect_ex = refuse_network
socket.create_connection = refuse_network
socket.getaddrinfo = refuse_network
result = pytest.main(['-p', 'pytestqt.plugin', *targets, '-q', '--tb=short',
    '--junitxml=' + str(root / (mode + '.xml')), '--basetemp=' + str(root / (mode + '-tmp')),
    '-o', 'cache_dir=' + str(root / 'pytest-cache')])
(root / (mode + '-network.json')).write_text(json.dumps({'network_blocked': True, 'blocked_attempts': len(attempts)}) + '\n')
raise SystemExit(result)
