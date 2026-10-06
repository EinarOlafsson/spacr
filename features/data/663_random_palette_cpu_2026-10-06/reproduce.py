"""Restore immutable CPU proof sources and replay without app modifications."""
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
    """Run one bounded proof inside the caller's capped CPU environment."""
    parser = argparse.ArgumentParser()
    parser.add_argument('repo', type=Path)
    parser.add_argument('scratch', type=Path)
    parser.add_argument('mode', choices=['startup', 'native', 'worker', 'integrated_worker', 'optimized_native', 'optimized_worker', 'optimized_startup'])
    parser.add_argument('--theme', default='data_art_point_atlas')
    parser.add_argument('--palette', default='random')
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    args.scratch.mkdir(parents=True, exist_ok=True)
    hashes = json.loads((source / 'native_receipt.json').read_text())['source_sha256']
    checkpoints = json.loads((source / 'source_checkpoints.json').read_text())
    for label in ['before', 'after']:
        payload = gzip.decompress((source / (label + '_ambient.py.gz')).read_bytes())
        assert hashlib.sha256(payload).hexdigest() == hashes[label]
        (args.scratch / (label + '.py')).write_bytes(payload)
    if args.mode == 'integrated_worker' or args.mode.startswith('optimized_'):
        payload = gzip.decompress((source / 'integrated_ambient.py.gz').read_bytes())
        assert hashlib.sha256(payload).hexdigest() == checkpoints['integrated_ambient.py.gz']
        if args.mode.startswith('optimized_'):
            (args.scratch / 'before.py').write_bytes(payload)
            filename = 'optimized_individual_ambient.py.gz' if args.mode == 'optimized_startup' else 'optimized_integrated_ambient.py.gz'
            payload = gzip.decompress((source / filename).read_bytes())
            assert hashlib.sha256(payload).hexdigest() == checkpoints[filename]
        (args.scratch / 'after.py').write_bytes(payload)
    probe = {'startup': 'startup_probe.py', 'native': 'native_proof.py',
             'worker': 'worker_probe.py', 'integrated_worker': 'integrated_worker_probe.py',
             'optimized_worker': 'integrated_worker_probe.py',
             'optimized_native': 'optimized_native_proof.py',
             'optimized_startup': 'startup_probe.py'}[args.mode]
    shutil.copy2(source / probe, args.scratch / probe)
    command = [sys.executable, str(args.scratch / probe)]
    command += [args.theme, 'after'] if 'worker' in args.mode else [str(args.repo.resolve())]
    if args.mode in ('integrated_worker', 'optimized_worker'):
        command += ['--palette', args.palette]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='',
                       SPACR_PROOF_REPO=str(args.repo.resolve()))
    subprocess.run(command, cwd=args.repo, env=environment, check=True)


if __name__ == '__main__':
    main()
