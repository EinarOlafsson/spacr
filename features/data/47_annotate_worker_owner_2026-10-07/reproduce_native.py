"""Replay the frozen Linux retrain-worker native ownership proof."""
import gzip
import runpy
import sys
import tempfile
from pathlib import Path
from spacr.qt.screens import annotate as module

archive = Path(__file__).resolve().parent
phase = sys.argv[1]
assert phase in {'before', 'after'}
source = archive / (phase + '_annotate.py.gz')
stage = tempfile.TemporaryDirectory(prefix='spacr-annotate-owner-', dir='/mnt/wd4tb/scratch')
plain = Path(stage.name) / (phase + '_annotate.py')
code = gzip.decompress(source.read_bytes())
plain.write_bytes(code)
module.__file__ = str(plain)
exec(compile(code, str(plain), 'exec'), module.__dict__)
runpy.run_path(str(archive / 'native_probe.py'), run_name='__main__')
