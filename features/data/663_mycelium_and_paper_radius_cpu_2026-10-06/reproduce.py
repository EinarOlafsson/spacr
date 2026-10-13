"""Replay an archived CPU proof against explicit immutable renderer sources."""
import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['frames', 'paper', 'worker'])
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--scratch', type=Path, required=True)
args = parser.parse_args()
archive = Path(__file__).resolve().parent
manifest = json.loads((archive / 'manifest.json').read_text())
for name, record in manifest['files'].items():
    assert hashlib.sha256((archive / name).read_bytes()).hexdigest() == record['sha256'], name
destination = args.scratch.resolve() / args.mode
destination.mkdir(parents=True, exist_ok=True)
mapping = {'frames': ('before', 'after', 'render.py'),
           'paper': ('before', 'after', 'paper_center_probe.py'),
           'worker': ('before', 'after', 'worker_probe.py')}
before, after, script = mapping[args.mode]
for label, output in [(before, 'before.py'), (after, 'after.py')]:
    data = gzip.decompress((archive / (label + '.py.gz')).read_bytes())
    assert hashlib.sha256(data).hexdigest() == manifest['sources'][label]['source_sha256']
    (destination / output).write_bytes(data)
shutil.copy2(archive / script, destination / script)
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', QT_QPA_PLATFORM='offscreen',
           SPACR_REPRO_REPO=str(args.repo.resolve()))
base = [str(args.repo.resolve() / 'tools/run_capped.sh'), '4G', sys.executable,
        str(destination / script)]
if args.mode == 'worker':
    env.update(QT_QPA_PLATFORM='xcb', QT_OPENGL='software', LIBGL_ALWAYS_SOFTWARE='1')
    for variant in ['before', 'after']:
        subprocess.run(['xvfb-run', '-a', '-s',
                        '-screen 0 3840x2160x24 -nolisten tcp -noreset',
                        *base, 'data_art_fungal_growth', variant], env=env, cwd=args.repo, check=True)
else:
    subprocess.run(base, env=env, cwd=args.repo, check=True)
