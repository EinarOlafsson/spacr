"""Run one preserved scratch probe against decompressed frozen renderers."""
import argparse
import gzip
import importlib.util
import runpy
import sys
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('phase', choices=['unrestricted', 'cached', 'admission'])
parser.add_argument('--scratch-root', type=Path, default=Path('/mnt/wd4tb/scratch'))
args = parser.parse_args()
archive = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='fungal-stroker-replay-', dir=args.scratch_root) as directory:
    stage = Path(directory)
    for name in ('before', 'candidate'):
        (stage / (name + '.py')).write_bytes(gzip.decompress((archive / (name + '_ambient.py.gz')).read_bytes()))
    if args.phase == 'unrestricted':
        import spacr.qt.widgets
        spec = importlib.util.spec_from_file_location('spacr.qt.widgets.ambient', stage / 'before.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        spacr.qt.widgets.ambient = module
        script = stage / 'probe.py'
        script.write_bytes((archive / 'unrestricted_probe.py').read_bytes())
        runpy.run_path(str(script), run_name='__main__')
    elif args.phase == 'cached':
        script = stage / 'cached_probe.py'
        script.write_bytes((archive / 'cached_probe.py').read_bytes())
        runpy.run_path(str(script), run_name='__main__')
    else:
        script = stage / 'admission_probe.py'
        source = (archive / 'admission_probe.py').read_text()
        old = '/mnt/wd4tb/scratch/fungal-stroker-parity-20261007'
        assert source.count(old) == 1
        script.write_text(source.replace(old, str(stage)))
        runpy.run_path(str(script), run_name='__main__')
    for receipt in stage.glob('*.json'):
        print(receipt.name, receipt.read_text())
