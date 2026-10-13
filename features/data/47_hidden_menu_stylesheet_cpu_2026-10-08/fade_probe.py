import cProfile
import json
import sys
import time
from pathlib import Path
import pytest
from spacr.qt.widgets import ambient
counts = {}
for cls, name in ((ambient.AmbientWidget, '_on_tick'), (ambient._BufferedEngine, 'shade')):
    original = getattr(cls, name)
    def wrapped(*args, _original=original, _name=name, **kwargs):
        started = time.perf_counter()
        try:
            return _original(*args, **kwargs)
        finally:
            item = counts.setdefault(_name, [0, 0.0])
            item[0] += 1
            item[1] += time.perf_counter() - started
    setattr(cls, name, wrapped)
class Probe:
    def __init__(self):
        self.rows = []
    def pytest_runtest_logreport(self, report):
        self.rows.append({'nodeid':report.nodeid,'phase':report.when,'duration':report.duration,'outcome':report.outcome})
probe = Probe()
profile = cProfile.Profile()
profile.enable()
code = pytest.main(['-q', '-p', 'no:randomly', '--durations=12', 'tests/qt/test_field_fade.py'], plugins=[probe])
profile.disable()
root = Path('/mnt/wd4tb/scratch/serial-field-fade-20261008')
profile.dump_stats(str(root / 'field-fade.pstats'))
(root / 'fresh.json').write_text(json.dumps({'returncode':code,'ambient_counts':counts,'reports':probe.rows},indent=2))
sys.exit(code)
