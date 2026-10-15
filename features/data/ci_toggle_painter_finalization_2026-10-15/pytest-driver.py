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
target = 'tests/qt/test_toggle_painter_finalization.py'
if mode == 'baseline':
    target += '::test_painter_is_ended_even_when_a_paint_stage_raises[focus_rect]'
raise SystemExit(pytest.main(['-p', 'pytestqt.plugin', target, '-q', '--tb=short',
    '--junitxml=' + str(root / (mode + '.xml')), '--basetemp=' + str(root / (mode + '-tmp')),
    '-o', 'cache_dir=' + str(root / 'pytest-cache')]))
