#!/usr/bin/env python3
"""Exercise the public pip route in a new, private tutorial environment.

This installs no editable checkout, reuses no system site-packages, and never
updates or removes an existing environment. Run under run_memory_guarded.py.
Logs and failed environments are retained as evidence, not marked successful.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=DEFAULT_STAGE)
    args = parser.parse_args()
    stage = args.stage.resolve()
    if shutil.disk_usage(stage).free < 40 * 1024**3:
        raise RuntimeError('A fresh scientific installation needs 40 GiB free headroom')
    parent = stage / 'installation_runs'
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='public-pypi-', dir=parent))
    receipt = {'route': 'Public PyPI, clean Python venv, no system packages',
               'folder': str(root), 'steps': [], 'accepted': False,
               'source_checkout_installed': False, 'published': False}
    write(root / 'receipt.json', receipt)
    print(f'Private installation: {root}', flush=True)
    with urlopen('https://pypi.org/pypi/spacr/json', timeout=30) as response:
        payload = response.read()
    metadata = json.loads(payload)
    version = metadata['info']['version']
    if not version or not all(c.isdecimal() or c == '.' for c in version):
        raise ValueError('Expected a numeric public release version')
    receipt['pypi'] = {'url': 'https://pypi.org/pypi/spacr/json',
                       'response_sha256': hashlib.sha256(payload).hexdigest(),
                       'version': version,
                       'requires_python': metadata['info']['requires_python'],
                       'files': [{key: item[key] for key in
                                  ('filename', 'url', 'size', 'digests', 'upload_time_iso_8601')}
                                 for item in metadata['urls']]}
    write(root / 'pypi-metadata.json', metadata)
    venv = root / 'venv'
    python = str(venv / 'bin/python')
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'CONDA_PREFIX',
                'PIP_INDEX_URL', 'PIP_EXTRA_INDEX_URL', 'PIP_TARGET',
                'PIP_PREFIX', 'PIP_USER', 'PIP_CONSTRAINT'):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE='1', PIP_CONFIG_FILE=os.devnull,
               PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_PROGRESS_BAR='off',
               PIP_DEFAULT_TIMEOUT='60', PIP_RETRIES='1',
               PIP_CACHE_DIR=str(stage / 'installation_pip_cache'),
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
               MAX_JOBS='2', CARGO_BUILD_JOBS='2', USE_TF='0',
               QT_QPA_PLATFORM='offscreen')
    for key, name in (('XDG_CONFIG_HOME', 'config'), ('XDG_CACHE_HOME', 'cache'),
                      ('XDG_DATA_HOME', 'data')):
        folder = root / name
        folder.mkdir()
        env[key] = str(folder)

    def run(name, command, timeout):
        start = time.monotonic()
        log = root / f'{name}.log'
        row = {'name': name, 'command': command, 'log': str(log),
               'returncode': None, 'completed': False}
        receipt['steps'].append(row)
        write(root / 'receipt.json', receipt)
        print(f'Starting {name}', flush=True)
        with log.open('w') as output:
            try:
                result = subprocess.run(command, cwd=root, env=env,
                                        stdout=output, stderr=subprocess.STDOUT,
                                        timeout=timeout)
                row['returncode'] = result.returncode
                row['completed'] = True
            except subprocess.TimeoutExpired:
                row['failure'] = 'timeout'
                raise
            finally:
                row['elapsed_seconds'] = round(time.monotonic() - start, 3)
                write(root / 'receipt.json', receipt)
        row['log_sha256'] = hashlib.sha256(log.read_bytes()).hexdigest()
        write(root / 'receipt.json', receipt)
        print(f'{name}: exit {result.returncode}, {row["elapsed_seconds"]}s', flush=True)
        if result.returncode:
            print(log.read_text(errors='replace')[-5000:], flush=True)
            raise RuntimeError(f'{name} failed; evidence retained in {root}')

    run('01_create', [sys.executable, '-m', 'venv', str(venv)], 90)
    env['PATH'] = str(venv / 'bin') + os.pathsep + env['PATH']
    env['VIRTUAL_ENV'] = str(venv)
    env['PIP_REQUIRE_VIRTUALENV'] = '1'
    run('02_upgrade_pip', [python, '-m', 'pip', 'install', '--index-url',
                          'https://pypi.org/simple', '--upgrade', 'pip'], 180)
    run('03_install', [python, '-m', 'pip', 'install', '--index-url',
                      'https://pypi.org/simple', '--report', str(root / 'install-report.json'),
                      f'spacr=={version}'], 2400)
    run('04_dependency_check', [python, '-m', 'pip', 'check'], 90)
    program = (
        'import json,sys,pathlib,importlib.metadata as m,spacr,PySide6,torch; '
        f'assert m.version("spacr")=={version!r}; '
        'assert pathlib.Path(spacr.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix)); '
        'print(json.dumps(dict(python=sys.executable,prefix=sys.prefix,spacr=spacr.__version__, '
        'location=spacr.__file__,qt=PySide6.__version__,torch=torch.__version__, '
        'cuda_available=torch.cuda.is_available()),indent=2))'
    )
    run('05_import', [python, '-I', '-c', program], 180)
    run('06_doctor', [str(venv / 'bin/spacr-doctor'), '--json', '--no-gpu-probe'], 240)
    receipt['accepted'] = True
    receipt['scope'] = 'Linux installation, pip consistency, real package imports and doctor only; no GUI or other-platform success claimed'
    write(root / 'receipt.json', receipt)
    print(f'Installation checks accepted: {root}', flush=True)


if __name__ == '__main__':
    main()
