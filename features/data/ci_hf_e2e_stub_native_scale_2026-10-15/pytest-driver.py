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
import socket
attempts = []
def refuse_network(*args, **kwargs):
    attempts.append('blocked')
    raise RuntimeError('Offline fixture check forbids network')
socket.socket.connect = refuse_network
socket.socket.connect_ex = refuse_network
socket.create_connection = refuse_network
socket.getaddrinfo = refuse_network
import pytest
target = 'tests/test_hf_e2e_stub_diameter_contract.py'
result = pytest.main(['-p', 'pytestqt.plugin', target, '-q', '--tb=short',
    '--junitxml=' + str(root / (mode + '.xml')), '--basetemp=' + str(root / (mode + '-tmp')),
    '-o', 'cache_dir=' + str(root / 'pytest-cache')])
(root / (mode + '-runtime.json')).write_text(json.dumps({'model_modules_loaded': sorted(name for name in sys.modules if name.split('.')[0] in {'cellpose', 'ultralytics', 'onnxruntime'}), 'network_attempts_blocked': len(attempts)}) + '\n')
raise SystemExit(result)
