"""Record the real AnnData failure and explicit API workaround in GNOME Terminal.

Only private copies and new exports are written. The original GUI failure
remains documented separately; this does not substitute a success for it.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

from capture_cli import accepted_command, capture_terminal
from stage_lesson import DEFAULT_STAGE, REPO, read, write

NAME = 'anndata_api_terminal'


def terminal_driver(stage):
    from capture_database import prepare_database_copy, require_unchanged_source
    from anndata_evidence import read_source, verify_file

    captures = stage / 'captures' / NAME
    work = Path(tempfile.mkdtemp(prefix='anndata-native-api-', dir=stage))
    database = work / 'measurements.db'
    source = stage / 'annotate_fresh/example_data/plate1/measurements/measurements.db'
    manifest = prepare_database_copy(source, database,
        expected_sha256='7b18161f0161d39b3ecedf92cfb0ccf9fee2328980da8e43167555a8f6fd27cd')
    helper = Path(__file__).with_name('anndata_missing_metadata_example.py')
    shutil.copy2(helper, work / helper.name)
    reference = read_source(database)
    failure = ("from spacr.anndata_export import export_anndata; "
               "export_anndata('measurements.db', 'unmodified_failure.h5ad', verbose=False)")
    commands = [
        ('00_version', ['python', '-c', 'import spacr,anndata; print("spaCR",spacr.__version__); print("AnnData",anndata.__version__)'], 0, 'spaCR'),
        ('01_unmodified_failure', ['python', '-c', failure], 1, 'non-string objects'),
        ('02_explicit_workaround', ['python', '-c', 'import inspect; from anndata_missing_metadata_example import encode_empty_metadata; print(inspect.getsource(encode_empty_metadata))'], 0, 'pd.Categorical(series)'),
    ]
    exports = [('', 'keep'), ('cell', 'keep'), ('cell', 'mean'),
               ('cell', 'drop_features'), ('cell', 'drop_objects'), ('nucleus', 'keep')]
    for index, (single, policy) in enumerate(exports, 3):
        command = ['python', helper.name, '--source', 'measurements.db', '--out',
                   f'results/{single or "joined"}_{policy}.h5ad', '--nan-policy', policy]
        if single:
            command += ['--single-table', single]
        commands.append((f'{index:02d}_{single or "joined"}_{policy}', command, 0,
                         'Original database unchanged: True'))
    commands.append(('09_saved_shape', ['python', '-c',
        'import anndata,numpy as np; a=anndata.read_h5ad("results/joined_keep.h5ad"); '
        'print("Saved joined shape:",a.shape); print("Missing X values:",int(np.isnan(a.X).sum())); '
        'print("Metadata-only workaround:",a.uns["tutorial_missing_metadata_encoding"]); '
        'print("Images are NOT embedded; retain the source image files.")'], 0, '(2341, 1136)'))
    outcomes, verified = [], []
    try:
        for scene, command, expected_code, expected_text in commands:
            print('\033[2J\033[3J\033[H', end='', flush=True)
            print('$ ' + shlex.join(command), flush=True)
            result = subprocess.run(command, cwd=work, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
            print(result.stdout, end='', flush=True)
            print(f'\nExit status: {result.returncode}', flush=True)
            passed = accepted_command(result, expected_code, expected_text)
            if scene.startswith(tuple(f'{index:02d}_' for index in range(3, 9))) and passed:
                index = int(scene[:2]) - 3
                single, policy = exports[index]
                target = work / f'results/{single or "joined"}_{policy}.h5ad'
                record = {'single_table': single, 'nan_policy': policy,
                          'settings': {'anndata_nan_policy': policy, 'anndata_compute_umap': False}}
                check = verify_file(target, reference, record, database)
                verified.append(check)
                print('Independent SQLite comparison: PASS', flush=True)
            outcomes.append({'scene': scene, 'command': command, 'returncode': result.returncode,
                             'output': result.stdout, 'accepted': passed,
                             'expected_failure': expected_code != 0})
            write(captures / 'commands.json', outcomes)
            write(captures / 'terminal_ready.json', {'scene': scene, 'accepted': passed})
            input('\nPress Enter to continue the recording. ')
            if not passed:
                return 1
        require_unchanged_source(source, manifest['source_bundle'])
        if hashlib.sha256(database.read_bytes()).hexdigest() != manifest['database_sha256']:
            raise ValueError('Private source database changed')
        if len(verified) != 6:
            raise ValueError('The actual terminal did not verify all six exports')
        write(captures / 'scientific_acceptance.json', {
            'accepted': True, 'scope': 'Observed unmodified failure followed by explicit API workaround',
            'source': manifest, 'private_directory': str(work), 'source_unchanged': True,
            'helper_sha256': hashlib.sha256(helper.read_bytes()).hexdigest(),
            'exports': verified, 'gui_defect_fixed': False, 'published': False})
        return 0
    finally:
        require_unchanged_source(source, manifest['source_bundle'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=DEFAULT_STAGE)
    parser.add_argument('--inside', action='store_true')
    parser.add_argument('--terminal-driver', action='store_true')
    args = parser.parse_args(); stage = args.stage.resolve()
    if args.terminal_driver:
        return terminal_driver(stage)
    if args.inside:
        return capture_terminal(stage, driver=Path(__file__).resolve(), capture_name=NAME,
            window_title='spaCR AnnData API workaround', expected_scenes=10,
            module='anndata_export', pipeline_requested=True)
    if (stage / 'captures' / NAME / 'provenance.json').exists():
        raise FileExistsError('Preserve the existing AnnData terminal capture')
    env = dict(os.environ); root = stage / 'desktop' / NAME
    for key, name in [('XDG_CONFIG_HOME', 'config'), ('XDG_DATA_HOME', 'data'),
                      ('XDG_CACHE_HOME', 'cache'), ('XDG_RUNTIME_DIR', 'runtime')]:
        path = root / name; path.mkdir(parents=True, exist_ok=True, mode=0o700)
        env[key] = str(path)
    env.update(SPACR_TUTORIAL_PRIVATE_DESKTOP='1', GIO_USE_VFS='local',
        GSETTINGS_BACKEND='memory', GTK_USE_PORTAL='0', XDG_CURRENT_DESKTOP='SPACR_TUTORIAL',
        NO_AT_BRIDGE='1', QT_QPA_PLATFORM='xcb',
        PATH=str(Path(sys.executable).parent) + os.pathsep + env['PATH'], PYTHONPATH=str(REPO),
        OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    return subprocess.run(['xvfb-run', '-a', '-s', '-screen 0 3840x2160x24',
        'dbus-run-session', '--', sys.executable, str(Path(__file__).resolve()),
        '--stage', str(stage), '--inside'], env=env, timeout=650).returncode


if __name__ == '__main__':
    raise SystemExit(main())
