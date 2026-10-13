"""Replay the frozen native field-loader ownership experiment."""
import gzip
import runpy
import sys
from pathlib import Path
from spacr.qt.screens import make_masks as module

archive = Path(__file__).resolve().parent
phase = sys.argv[1]
assert phase in {'before', 'after'}
source = archive / (phase + '_make_masks.py.gz')
import tempfile
stage = tempfile.TemporaryDirectory(prefix='spacr-field-owner-replay-', dir='/mnt/wd4tb/scratch')
plain = Path(stage.name) / (phase + '_make_masks.py')
code = gzip.decompress(source.read_bytes())
plain.write_bytes(code)
module.__file__ = str(plain)
exec(compile(code, str(source), 'exec'), module.__dict__)
runpy.run_path(str(archive / 'native_probe.py'), run_name='__main__')
