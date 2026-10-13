#!/usr/bin/env python3
"""Verify conda-forge in a fresh prefix, keeping its registry/cache private.

Run under run_memory_guarded.py. The public channel may lag PyPI. Plan and
transaction failures stay recorded; successful resolution alone is not an
installation, import, GUI or hardware acceptance result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen

from stage_lesson import DEFAULT_STAGE, write


def accepted_plan(plan, free_bytes):
    if plan.get('success') is not True or not plan.get('actions', {}).get('LINK'):
        raise ValueError('Conda did not resolve a complete installable environment')
    packages = plan['actions']['LINK']
    spacr = [row for row in packages if row.get('name') == 'spacr']
    if len(spacr) != 1 or not str(spacr[0].get('channel', '')).startswith('conda-forge'):
        raise ValueError('The plan must install exactly one spaCR from conda-forge')
    size = sum(row.get('size', 0) for row in plan['actions'].get('FETCH', []))
    if size > 20 * 1024**3 or free_bytes < max(40 * 1024**3, size * 5):
        raise ValueError('The plan exceeds the bounded tutorial download/disk budget')
    return dict(packages=len(packages), download_bytes=size, version=spacr[0]['version'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=DEFAULT_STAGE)
    parser.add_argument('--conda', type=Path, default=Path('/home/olafsson/anaconda3/bin/conda'))
    args = parser.parse_args()
    stage = args.stage.resolve()
    if shutil.disk_usage(stage).free < 40 * 1024**3:
        raise ValueError('Keep at least 40 GiB of free installation headroom')
    parent = stage / 'installation_runs'
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='conda-forge-', dir=parent))
    prefix = root / 'env'
    registry = root / 'conda-registry'
    registry.mkdir()
    receipt = dict(route='conda-forge, fresh private prefix', folder=str(root),
                   accepted=False, steps=[], published=False, source_checkout_installed=False)
    write(root / 'receipt.json', receipt)
    print(f'Private Conda check: {root}', flush=True)
    with urlopen('https://api.anaconda.org/package/conda-forge/spacr', timeout=30) as response:
        payload = response.read()
    info = json.loads(payload)
    version = info['latest_version']
    if not version or not all(c.isdecimal() or c == '.' for c in version):
        raise ValueError('Unexpected conda-forge release version')
    receipt['channel'] = dict(version=version, url='https://api.anaconda.org/package/conda-forge/spacr',
                              response_sha256=hashlib.sha256(payload).hexdigest())
    write(root / 'channel-metadata.json', info)
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'CONDA_PREFIX',
                'CONDA_DEFAULT_ENV', 'CONDA_SUBDIR', 'CONDA_OVERRIDE_CUDA'):
        env.pop(key, None)
    env.update(CONDARC=os.devnull, CONDA_PKGS_DIRS=str(stage / 'installation_conda_cache'),
               CONDA_ENVS_PATH=str(root / 'envs'), CONDA_AUTO_UPDATE_CONDA='false',
               CONDA_REMOTE_CONNECT_TIMEOUT_SECS='30', CONDA_REMOTE_READ_TIMEOUT_SECS='90',
               CONDA_REMOTE_MAX_RETRIES='1', CONDA_FETCH_THREADS='2',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
               NUMEXPR_NUM_THREADS='2', QT_QPA_PLATFORM='offscreen', USE_TF='0',
               PYTHONNOUSERSITE='1')
    for key, name in (('XDG_CONFIG_HOME', 'config'), ('XDG_CACHE_HOME', 'cache'),
                      ('XDG_DATA_HOME', 'data')):
        folder = root / name
        folder.mkdir()
        env[key] = str(folder)
    wrapper = ['bwrap', '--die-with-parent', '--bind', '/', '/', '--dev-bind', '/dev', '/dev',
               '--bind', str(registry), str(Path.home() / '.conda'), '--']

    def run(name, command, timeout):
        row = dict(name=name, command=command, completed=False, returncode=None)
        receipt['steps'].append(row)
        write(root / 'receipt.json', receipt)
        start = time.monotonic()
        log = root / (name + '.log')
        errors = root / (name + '.stderr.log')
        print('Starting ' + name, flush=True)
        try:
            with log.open('w') as stdout, errors.open('w') as stderr:
                result = subprocess.run(wrapper + command, cwd=root, env=env,
                                        stdout=stdout, stderr=stderr, timeout=timeout)
            row.update(completed=True, returncode=result.returncode)
        except subprocess.TimeoutExpired:
            row['failure'] = 'timeout'
            raise
        finally:
            row['elapsed_seconds'] = round(time.monotonic() - start, 3)
            row['log_sha256'] = hashlib.sha256(log.read_bytes()).hexdigest()
            row['stderr_sha256'] = hashlib.sha256(errors.read_bytes()).hexdigest()
            write(root / 'receipt.json', receipt)
        print(f'{name}: exit {result.returncode}, {row["elapsed_seconds"]}s', flush=True)
        if result.returncode:
            print(log.read_text(errors='replace')[-4000:] + errors.read_text(errors='replace')[-2000:], flush=True)
            raise RuntimeError(f'{name} failed; do not claim this route is verified')
        return log

    create = [str(args.conda), 'create', '--prefix', str(prefix), '--override-channels',
              '-c', 'conda-forge', '--strict-channel-priority', '--no-default-packages',
              '--solver', 'libmamba', '--json', '--yes', 'python=3.12', f'spacr={version}']
    plan_path = run('01_solve', create + ['--dry-run'], 900)
    receipt['plan'] = accepted_plan(json.loads(plan_path.read_text()), shutil.disk_usage(stage).free)
    if receipt['plan']['version'] != version:
        raise ValueError('The solver changed the requested release')
    write(root / 'receipt.json', receipt)
    print('Plan: ' + json.dumps(receipt['plan']), flush=True)
    run('02_install', create, 2400)
    env['PATH'] = str(prefix / 'bin') + os.pathsep + env['PATH']
    program = ('import pathlib,sys,json,spacr,PySide6; '
               'assert pathlib.Path(spacr.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix)); '
               f'assert spacr.__version__ == {version!r}; '
               'print(json.dumps(dict(python=sys.executable,spacr=spacr.__version__, '
               'qt=PySide6.__version__,package=spacr.__file__,prefix=sys.prefix),indent=2))')
    run('03_import', [str(prefix / 'bin/python'), '-I', '-c', program], 180)
    run('04_doctor', [str(prefix / 'bin/spacr-doctor'), '--json', '--no-gpu-probe'], 240)
    run('05_packages', [str(args.conda), 'list', '--prefix', str(prefix), '--json'], 120)
    receipt['accepted'] = True
    receipt['scope'] = 'Linux conda-forge install, package/Qt import and doctor; no GUI, GPU allocation or other-platform claim'
    write(root / 'receipt.json', receipt)
    print('Conda installation checks accepted: ' + str(root), flush=True)


if __name__ == '__main__':
    main()
