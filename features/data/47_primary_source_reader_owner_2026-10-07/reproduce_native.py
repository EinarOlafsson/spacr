"""Replay a frozen primary-selector source through its real native lifecycle."""
import runpy
import sys
from pathlib import Path
from spacr.qt.widgets import primary_mask_selector as module

archive = Path(__file__).resolve().parent
phase = sys.argv[1]
assert phase in {'before', 'after'}
source = archive / (phase + '_primary_mask_selector.py')
module.__file__ = str(source)
exec(compile(source.read_bytes(), str(source), 'exec'), module.__dict__)
runpy.run_path(str(archive / 'native_probe.py'), run_name='__main__')
