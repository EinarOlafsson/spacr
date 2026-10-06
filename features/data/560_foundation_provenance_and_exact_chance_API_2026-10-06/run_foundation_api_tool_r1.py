from pathlib import Path
import os
import runpy
import sys

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
root = Path.cwd().resolve()
sys.meta_path[:] = [f for f in sys.meta_path
                   if '__editable__' not in (getattr(f, '__module__', '') or type(f).__module__)]
sys.path.insert(0, str(root))
import spacr
assert Path(spacr.__file__).resolve().parent == root / 'spacr'
log = Path(sys.argv[1])
assert not log.exists(), log
script = sys.argv[2]
sys.argv = sys.argv[2:]

class Tee:
    def __init__(self, original, handle):
        self.original, self.handle = original, handle
    def write(self, text):
        self.original.write(text)
        self.handle.write(text)
        self.handle.flush()
    def flush(self):
        self.original.flush()
        self.handle.flush()

with log.open('w') as handle:
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(stdout, handle), Tee(stderr, handle)
    try:
        print('Actual imported source:', spacr.__file__, flush=True)
        if script == 'pytest':
            import pytest
            raise SystemExit(pytest.main(sys.argv[1:]))
        sys.path.insert(0, str(root / 'tools'))
        runpy.run_path(script, run_name='__main__')
    finally:
        sys.stdout, sys.stderr = stdout, stderr
