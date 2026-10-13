"""Replay immutable packed-word proofs in an empty caller-owned scratch folder."""
import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    """Restore verified renderer snapshots and run one capped CPU proof."""
    parser = argparse.ArgumentParser()
    parser.add_argument('repo', type=Path)
    parser.add_argument('scratch', type=Path)
    parser.add_argument('mode', choices=['startup', 'primitive', 'native', 'stages', 'abba', 'gc', 'worker'])
    parser.add_argument('--variant', choices=['before', 'after'], default='after')
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    if args.scratch.exists() and any(args.scratch.iterdir()):
        parser.error('scratch must be empty to preserve existing receipts')
    args.scratch.mkdir(parents=True, exist_ok=True)
    hashes = json.loads((source / 'acceptance.json').read_text())['source_sha256']
    for label in ['before', 'after']:
        payload = gzip.decompress((source / (label + '_ambient.py.gz')).read_bytes())
        assert hashlib.sha256(payload).hexdigest() == hashes[label]
        (args.scratch / (label + '.py')).write_bytes(payload)
    probe = {'startup': 'startup_probe.py', 'primitive': 'primitive_proof.py',
             'native': 'native_proof.py', 'stages': 'balanced_stages.py',
             'abba': 'balanced_worker.py', 'gc': 'gc_probe.py',
             'worker': 'worker_probe.py'}[args.mode]
    shutil.copy2(source / probe, args.scratch / probe)
    command = [sys.executable, str(args.scratch / probe)]
    if args.mode == 'worker':
        command += ['data_art_genetic_advection', args.variant, '--palette', 'random']
    elif args.mode not in ('abba', 'gc'):
        command += [str(args.repo.resolve())]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='',
                       SPACR_PROOF_REPO=str(args.repo.resolve()))
    subprocess.run(command, cwd=args.repo, env=environment, check=True)


if __name__ == '__main__':
    main()
