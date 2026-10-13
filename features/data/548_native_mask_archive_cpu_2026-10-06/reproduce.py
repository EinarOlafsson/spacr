"""Reproduce the bounded native Mask archive proof outside the repository."""
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
    """Prepare a private scratch directory and run sequential CPU-only probes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--unsafe-before', action='store_true',
                        help='Also reproduce the rejected explicit-close SIGSEGV in a capped subprocess.')
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    repo, scratch = args.repo.resolve(), args.scratch.resolve()
    if not scratch.is_relative_to(Path('/mnt/wd4tb/scratch')):
        raise ValueError('Proof workspace must be under /mnt/wd4tb/scratch.')
    scratch.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((source / 'receipt.json').read_text())
    if hashlib.sha256((repo / 'spacr/object.py').read_bytes()).hexdigest() != receipt['accepted_object_sha256']:
        raise ValueError('This reproduction requires the exact accepted object.py source.')
    for name in ['probe.py', 'production_probe.py', 'ast_parity.py', 'traceback_probe.py']:
        shutil.copy2(source / name, scratch / name)
    for name in ['before_object.py', 'explicit_close_object.py']:
        with gzip.open(source / (name + '.gz'), 'rb') as archive:
            (scratch / name).write_bytes(archive.read())
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'QT_QPA_PLATFORM': 'offscreen',
           'SPACR_PROOF_DIR': str(scratch), 'SPACR_PROOF_REPO': str(repo)}
    env.pop('SPACR_TRACE_BEFORE', None)
    cap = [str(repo / 'tools/run_capped.sh'), '4G', sys.executable]

    def run(name, *arguments, expected=0, extra=None):
        """Capture one capped subprocess without admitting unexpected failures."""
        with (scratch / (name + '.log')).open('w') as log:
            result = subprocess.run(cap + list(arguments), cwd=repo,
                                    env={**env, **(extra or {})}, stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
        if result.returncode != expected:
            raise RuntimeError(f'{name} exited {result.returncode}; expected {expected}.')

    run('fixture', str(scratch / 'probe.py'), 'fixture')
    for plan in ['t', 'z']:
        for mode in ['before', 'after']:
            run(f'production_{plan}_{mode}', str(scratch / 'production_probe.py'), mode,
                '--plan', plan)
    run('ast-parity', str(scratch / 'ast_parity.py'))
    run('traceback-after', str(scratch / 'traceback_probe.py'))
    if args.unsafe_before:
        run('traceback-before', str(scratch / 'traceback_probe.py'), expected=139,
            extra={'SPACR_TRACE_BEFORE': '1'})
    run('focused-tests', '-m', 'pytest', '-q', '-p', 'no:randomly',
        'tests/test_native_mask_archive_memory_f548.py',
        'tests/test_watch_native_t_series_f548.py::test_native_series_resume_excludes_only_regular_reserved_mask_workspace',
        '--basetemp=' + str(scratch / 'pytest'), '--cov=spacr.object', '--cov=spacr.core',
        '--cov-branch', '--cov-report=json:' + str(scratch / 'coverage.json'), '--cov-report=')
    for plan in ['t', 'z']:
        pair = [json.loads((scratch / f'production_{plan}_{mode}.json').read_text())
                for mode in ['before', 'after']]
        for key in receipt['pairs'][plan]['exact_comparison_keys']:
            if pair[0][key] != pair[1][key]:
                raise AssertionError(f'{plan} differs at {key}.')
    print('Exact CPU scientific/output parity and ownership checks passed:', scratch)


if __name__ == '__main__':
    main()
