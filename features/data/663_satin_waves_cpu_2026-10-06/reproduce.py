"""Replay frozen accepted CPU renderers into a separate scratch directory."""

import argparse
import gzip
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=('worker', 'native'))
parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
parser.add_argument('--output', type=Path)
args = parser.parse_args()
archive = Path(__file__).resolve().parent
if args.output is None:
    output = Path(tempfile.mkdtemp(prefix='satin-accepted-', dir=os.environ.get('TMPDIR')))
else:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
for archived, expanded in [('baseline_renderer.py.gz', 'before.py'),
                           ('accepted_renderer.py.gz', 'after.py')]:
    (output / expanded).write_bytes(gzip.decompress((archive / archived).read_bytes()))
if args.mode == 'worker':
    source = gzip.decompress((archive / 'original_worker_probe.py.gz').read_bytes()).decode()
    source = re.sub(r'^ROOT = Path\(.+\)$', 'ROOT = Path(' + repr(str(output)) + ')',
                    source, count=1, flags=re.MULTILINE)
else:
    source = gzip.decompress((archive / 'original_native_proof.py.gz').read_bytes()).decode()
    source = re.sub(r'^root=Path\(.+\)$', 'root=Path(' + repr(str(output)) + ')',
                    source, count=1, flags=re.MULTILINE)
    source = source.replace("path=root/'satin-strip-worker/after.py'", "path=root/'after.py'")
source = re.sub(r"sys.path.insert\(0, '[^']+'\)",
                'sys.path.insert(0, ' + repr(str(args.repo.resolve())) + ')', source)
script = output / ('replay_' + args.mode + '.py')
script.write_text(source)
print('Replay output:', output, flush=True)
environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', QT_QPA_PLATFORM='offscreen')
subprocess.run([sys.executable, str(script)], cwd=args.repo, env=environment, check=True)
