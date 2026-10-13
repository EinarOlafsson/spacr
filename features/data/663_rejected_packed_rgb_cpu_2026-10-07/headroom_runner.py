"""Run edge contracts against the explicit private renderer source."""
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest
import spacr.qt.widgets

root = Path(__file__).resolve().parent
path = root / 'after.py'
assert hashlib.sha256(path.read_bytes()).hexdigest() == '3b9f1cbc2920d64d59542be100e24bbc8c27f1d83b0e47423d811f829cd8dff3'
spec = importlib.util.spec_from_file_location('spacr.qt.widgets._packed_candidate', path)
ambient = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ambient
spec.loader.exec_module(ambient)
spacr.qt.widgets.ambient = ambient
sys.modules['spacr.qt.widgets.ambient'] = ambient
print('Frozen tested source', ambient.__file__, hashlib.sha256(path.read_bytes()).hexdigest(), flush=True)
raise SystemExit(pytest.main(['-q', '-p', 'no:randomly', str(root / 'test_fungal_packed_rgb_addition.py')]))
