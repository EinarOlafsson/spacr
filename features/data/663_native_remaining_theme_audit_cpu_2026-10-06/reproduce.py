"""Replay the frozen remaining-theme CPU audit in a fresh scratch directory."""

import argparse
import gzip
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
parser.add_argument('--output', type=Path)
args = parser.parse_args()
archive = Path(__file__).resolve().parent
if args.output is None:
    output = Path(tempfile.mkdtemp(prefix='native-theme-audit-', dir=os.environ.get('TMPDIR')))
else:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
frozen = archive.parent / '663_satin_waves_cpu_2026-10-06/integrated_renderer.py.gz'
(output / 'ambient_frozen_9ad.py').write_bytes(gzip.decompress(frozen.read_bytes()))
for name in ['probe.py', 'run.py']:
    source = gzip.decompress((archive / (name + '.gz')).read_bytes()).decode()
    source = re.sub(r"Path\('/mnt/wd4tb/spacr-worktrees/codex-cpu-final-20261006'\)",
                    'Path(' + repr(str(args.repo.resolve())) + ')', source)
    (output / name).write_text(source)
print('Replay output:', output, flush=True)
subprocess.run([sys.executable, str(output / 'run.py')], cwd=args.repo, check=True)
