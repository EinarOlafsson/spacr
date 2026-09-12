"""Run and record the actual Embeddings API and saved figures on a private desktop."""
import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

from capture_cli import accepted_command, capture_terminal
from embeddings_example import digest
from stage_lesson import DEFAULT_STAGE, REPO, read, write

NAME = 'embeddings_1507_api'


def terminal_driver(stage):
    from capture_diagnostics import PrivateDesktop
    captures = stage / 'captures' / NAME
    work = Path(tempfile.mkdtemp(prefix='embeddings-recorded-', dir=stage))
    helper = Path(__file__).with_name('embeddings_example.py')
    shutil.copy2(helper, work / helper.name)
    source = stage / 'annotate_fresh/example_data/plate1/data/single_nucleus/uninfected/plate1_E02/cell_png'
    (work / 'crops').symlink_to(source, target_is_directory=True)
    commands = [
        ('00_version', ['python', '-c',
         'import spacr,timm; print("spaCR",spacr.__version__); print("timm",timm.__version__); '
         'print("Optional install: python -m pip install spacr[embeddings]==1.5.0.7"); '
         'print("Default ResNet18: pretrained on ImageNet, not a cell-trained encoder.")'], 0, '1.5.0.7'),
        ('01_inputs', ['python', '-c',
         'from embeddings_example import load_crops; a,rows=load_crops("crops"); '
         'print("Real downloaded crops:",a.shape); print("No resizing; channels last."); '
         '[print(i+1,r["name"]) for i,r in enumerate(rows)]'], 0, '(16, 224, 224, 3)'),
        ('02_per_channel', ['python', helper.name, '--crops', 'crops', '--output', 'per_channel'],
         0, 'Vectors: (16, 1536)'),
        ('03_project', ['python', helper.name, '--crops', 'crops', '--output', 'project', '--policy', 'project'],
         0, 'Vectors: (16, 512)'),
        ('04_provenance', ['python', '-c',
         'import json; '
         'a=json.load(open("per_channel/run.json")); b=json.load(open("project/run.json")); '
         'assert a["sources"]==b["sources"]; '
         '[print(k,a[k]) for k in ("shape","weights_sha256","spec","verification")]; '
         'print("Same 16 unchanged crops:",a["sources"]==b["sources"]); '
         'print("Filenames identify this example; validate database join keys separately.")'],
         0, 'Same 16 unchanged crops: True'),
    ]
    outcomes = []
    for scene, command, expected, text in commands:
        print('\033[2J\033[3J\033[H', end='', flush=True)
        print('$ ' + shlex.join(command), flush=True)
        result = subprocess.run(command, cwd=work, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=120)
        print(result.stdout, end='', flush=True)
        print(f'\nExit status: {result.returncode}', flush=True)
        passed = accepted_command(result, expected, text)
        outcomes.append({'scene': scene, 'command': command, 'returncode': result.returncode,
                         'output': result.stdout, 'accepted': passed})
        write(captures / 'commands.json', outcomes)
        write(captures / 'terminal_ready.json', {'scene': scene, 'accepted': passed})
        input('\nPress Enter to continue the recording. ')
        if not passed:
            return 1
    desktop = PrivateDesktop(stage)
    figures = []
    try:
        for scene, relative in [('05_input_crops', 'per_channel/input_crops.png'),
                                ('06_per_channel_pca', 'per_channel/pca.png'),
                                ('07_project_pca', 'project/pca.png')]:
            target = work / relative
            image = subprocess.Popen(['eog', '--new-instance', str(target)],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            try:
                wid, title = desktop.find(target.name, time.sleep)
                desktop.show(wid)
                time.sleep(1.5)
                figures.append({'scene': scene, 'path': str(target), 'sha256': digest(target),
                                'viewer': 'eog', 'window_title': title})
                write(captures / 'figures.json', figures)
                write(captures / 'terminal_ready.json', {'scene': scene, 'accepted': True})
                # The recorder focuses this driver terminal when sending Return.
                input('\nPress Enter after recording the actual image viewer. ')
            finally:
                image.terminate()
                image.wait(timeout=5)
    finally:
        desktop.close()
    runs = [read(work / policy / 'run.json') for policy in ('per_channel', 'project')]
    if (runs[0]['sources'] != runs[1]['sources'] or any(not r['accepted'] for r in runs)
            or any(r['helper_sha256'] != digest(helper) for r in runs)):
        raise ValueError('Recorded run source or helper differs')
    for policy, report in zip(('per_channel', 'project'), runs):
        for name, sha in report['artifacts'].items():
            if digest(work / policy / name) != sha:
                raise ValueError('Recorded API output changed')
        for record in report['sources']:
            if digest(source / record['name']) != record['sha256']:
                raise ValueError('Original crop changed')
    write(captures / 'scientific_acceptance.json', {
        'accepted': True, 'scope': 'Two real API runs and actual saved-figure viewers',
        'runs': runs, 'figures': figures, 'private_work': str(work),
        'source_unchanged': True, 'helper_sha256': digest(helper),
        'gui_workflow_completed': False, 'crops_injected': False, 'published': False})
    return 0


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
                                window_title='spaCR Embeddings API', expected_scenes=8,
                                module='embeddings', pipeline_requested=True,
                                refocus_terminal_after_capture=True)
    if (stage / 'captures' / NAME).exists():
        raise FileExistsError('Preserve the existing API capture')
    env = dict(os.environ); root = stage / 'desktop' / NAME
    for key, name in [('XDG_CONFIG_HOME', 'config'), ('XDG_DATA_HOME', 'data'),
                      ('XDG_CACHE_HOME', 'cache'), ('XDG_RUNTIME_DIR', 'runtime')]:
        path = root / name; path.mkdir(parents=True, exist_ok=True, mode=0o700)
        env[key] = str(path)
    # An isolated desktop must not hide the already pinned encoder cache.
    env['HF_HOME'] = str(Path.home() / '.cache/huggingface')
    env.update(SPACR_TUTORIAL_PRIVATE_DESKTOP='1', GIO_USE_VFS='local',
               GSETTINGS_BACKEND='memory', GTK_USE_PORTAL='0', XDG_CURRENT_DESKTOP='SPACR_TUTORIAL',
               NO_AT_BRIDGE='1', QT_QPA_PLATFORM='xcb', HF_HUB_OFFLINE='1',
               PATH=str(Path(sys.executable).parent) + os.pathsep + env['PATH'],
               PYTHONPATH=str(REPO), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    return subprocess.run(['xvfb-run', '-a', '-s', '-screen 0 3840x2160x24',
        'dbus-run-session', '--', sys.executable, str(Path(__file__).resolve()),
        '--stage', str(stage), '--inside'], env=env, timeout=650).returncode


if __name__ == '__main__':
    raise SystemExit(main())
