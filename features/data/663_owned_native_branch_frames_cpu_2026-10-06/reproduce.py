"""Replay exact native branch-frame proof or actual CPU worker comparisons."""

import argparse
import gzip
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['exact', 'worker', 'allocation'])
parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
parser.add_argument('--output', type=Path)
args = parser.parse_args()
archive = Path(__file__).resolve().parent
if args.output is None:
    output = Path(tempfile.mkdtemp(prefix='owned-branch-proof-', dir=os.environ.get('TMPDIR')))
else:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
for name in ['before.py', 'after.py']:
    (output / name).write_bytes(gzip.decompress((archive / (name + '.gz')).read_bytes()))
for name in ['exact_proof.py', 'worker_probe.py', 'run_workers.py', 'allocation_probe.py']:
    source = gzip.decompress((archive / (name + '.gz')).read_bytes()).decode()
    source = re.sub(r"Path\('/mnt/wd4tb/spacr-worktrees/codex-cpu-final-20261006'\)",
                    'Path(' + repr(str(args.repo.resolve())) + ')', source)
    source = re.sub(r"sys.path.insert\(0,\s*'/mnt/wd4tb/spacr-worktrees/codex-cpu-final-20261006'\)",
                    'sys.path.insert(0, ' + repr(str(args.repo.resolve())) + ')', source)
    (output / name).write_text(source)
script = {'exact': 'exact_proof.py', 'worker': 'run_workers.py',
          'allocation': 'allocation_probe.py'}[args.mode]
print('Replay output:', output, flush=True)
environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', QT_QPA_PLATFORM='offscreen')
subprocess.run([sys.executable, str(output / script)], cwd=args.repo,
               env=environment, check=True)
