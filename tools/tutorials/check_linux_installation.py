#!/usr/bin/env python3
"""Run the verified public Linux installer entirely under tutorial storage.

Existing system GUI libraries are reused, never installed or upgraded. All
launcher, desktop entry, runtime, cache, profile and log paths stay private.
The actual installer selects its default backend; no GPU result is fabricated.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from stage_lesson import DEFAULT_STAGE, read, write


def verified_installer(stage, evidence):
    evidence = evidence.resolve()
    if not evidence.is_relative_to(stage / 'installation_runs'):
        raise ValueError('Use the private release-verification receipt')
    receipt = read(evidence)
    target = Path(receipt['installer']).resolve()
    if (receipt.get('accepted') is not True or not target.is_relative_to(evidence.parent)
            or hashlib.sha256(target.read_bytes()).hexdigest() != receipt.get('sha256')):
        raise ValueError('Only the exact checksum-verified installer can run')
    return target, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=DEFAULT_STAGE)
    parser.add_argument('--verified-release', type=Path, required=True)
    args = parser.parse_args()
    stage = args.stage.resolve()
    installer, verified = verified_installer(stage, args.verified_release)
    if shutil.disk_usage(stage).free < 40 * 1024**3:
        raise ValueError('Keep 40 GiB free before an isolated installer run')
    root = Path(tempfile.mkdtemp(prefix='linux-runtime-', dir=stage / 'installation_runs'))
    runtime = root / 'runtime'
    state = root / 'app-state'
    state.mkdir()
    receipt = dict(route='Public Linux online installer, default auto backend', folder=str(root),
                   accepted=False, steps=[], installer_sha256=verified['sha256'],
                   published=False, installer_tag=verified['tag'],
                   system_dependencies_installed=False, gui_launched=False,
                   consent_choices_supplied=False, existing_environment_changed=False)
    write(root / 'receipt.json', receipt)
    print('Private Linux installer run: ' + str(root), flush=True)
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'CONDA_PREFIX',
                'SPACR_TORCH_BACKEND', 'SPACR_PACKAGE_SPEC', 'SPACR_INSTALL_DRY_RUN',
                'SPACR_LAUNCHER_DIR', 'UV_INDEX_URL', 'UV_EXTRA_INDEX_URL',
                'PIP_INDEX_URL', 'PIP_EXTRA_INDEX_URL'):
        env.pop(key, None)
    env.update(SPACR_INSTALL_LANGUAGE='en', PYTHONNOUSERSITE='1',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
               UV_CONCURRENT_DOWNLOADS='2', UV_CONCURRENT_INSTALLS='2',
               UV_CONCURRENT_BUILDS='1', MAX_JOBS='2', CARGO_BUILD_JOBS='2',
               USE_TF='0', QT_QPA_PLATFORM='offscreen')
    for key, name in (('XDG_CONFIG_HOME', 'config'), ('XDG_DATA_HOME', 'data'),
                      ('XDG_CACHE_HOME', 'cache'), ('SPACR_LOG_DIR', 'logs'),
                      ('MPLCONFIGDIR', 'mpl')):
        path = root / name
        path.mkdir()
        env[key] = str(path)
    wrapper = ['bwrap', '--die-with-parent', '--bind', '/', '/', '--dev-bind', '/dev', '/dev',
               '--bind', str(state), str(Path.home() / '.spacr'), '--']

    def run(name, command, timeout):
        start = time.monotonic()
        log = root / (name + '.log')
        row = dict(name=name, command=command, completed=False, returncode=None)
        receipt['steps'].append(row)
        write(root / 'receipt.json', receipt)
        print('Starting ' + name, flush=True)
        try:
            with log.open('w') as output:
                result = subprocess.run(wrapper + command, cwd=root, env=env, stdout=output,
                                        stderr=subprocess.STDOUT, timeout=timeout)
            row.update(completed=True, returncode=result.returncode)
        except subprocess.TimeoutExpired:
            row['failure'] = 'timeout'
            raise
        finally:
            row['elapsed_seconds'] = round(time.monotonic() - start, 3)
            row['log_sha256'] = hashlib.sha256(log.read_bytes()).hexdigest()
            write(root / 'receipt.json', receipt)
        print(f'{name}: exit {result.returncode}, {row["elapsed_seconds"]}s', flush=True)
        if result.returncode:
            print(log.read_text(errors='replace')[-5000:], flush=True)
            raise RuntimeError(f'{name} failed; do not claim installation success')

    run('01_installer', ['bash', str(installer), '--install-root', str(runtime),
                        '--launcher-dir', str(root / 'bin'), '--skip-system-deps', '--no-launch'], 2400)
    python = runtime / 'venv/bin/python'
    env['PATH'] = str(root / 'bin') + os.pathsep + str(python.parent) + os.pathsep + env['PATH']
    program = ('import json,pathlib,sys,spacr,PySide6,torch; '
               'assert pathlib.Path(spacr.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix)); '
               f'assert spacr.__version__ == {verified["tag"][1:]!r}; '
               'print(json.dumps(dict(version=spacr.__version__,qt=PySide6.__version__, '
               'torch=torch.__version__,package=spacr.__file__,prefix=sys.prefix),indent=2))')
    run('02_import', [str(python), '-I', '-c', program], 180)
    run('03_doctor', [str(python.parent / 'spacr-doctor'), '--json', '--no-gpu-probe'], 240)
    receipt['install_profile'] = read(runtime / 'install-profile.json')
    receipt['accepted'] = True
    receipt['scope'] = 'Linux installer, real imports and doctor; no GUI, GPU throughput, Windows or macOS acceptance'
    write(root / 'receipt.json', receipt)
    print('Private Linux installation accepted: ' + str(root), flush=True)


if __name__ == '__main__':
    main()
