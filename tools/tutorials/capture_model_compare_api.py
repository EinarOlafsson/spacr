"""Record the real pure mask-comparison API, not injected GUI results."""
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
from ops_geometry_example import digest
from stage_lesson import DEFAULT_STAGE, REPO, read, write

NAME = 'model_compare_1507_api'


def terminal_driver(stage):
    from capture_diagnostics import PrivateDesktop
    captures = stage / 'captures' / NAME
    work = Path(tempfile.mkdtemp(prefix='model-compare-recorded-', dir=stage))
    helper = Path(__file__).with_name('model_compare_example.py')
    for path in (helper, helper.with_name('ops_geometry_example.py')):
        shutil.copy2(path, work / path.name)
    (work / 'inputs').symlink_to(stage / 'model_compare_1507_inputs', target_is_directory=True)
    commands = [
        ('00_inputs', ['python', '-c',
         'import tifffile,numpy as np; '
         'from model_compare_example import EXPECTED; from ops_geometry_example import digest; '
         'from pathlib import Path; '
         'assert all(digest(Path("inputs")/n)==h for n,h in EXPECTED.items()); '
         'print("Exact saved inputs from the Apply tutorial: cell_pair_02."); '
         'print("Image",tifffile.imread("inputs/image.tif").shape); '
         'print("A: batch preprocessing. B: live-preview preprocessing."); '
         'print("Same cpsam model, not ground truth and not a held-out accuracy test.")'], 0, '(512, 512)'),
        ('01_current_gui_limit', ['python', '-c',
         'import inspect; from importlib.metadata import version; '
         'from cellpose.models import CellposeModel; from spacr.model_compare import ModelConfig; '
         's=inspect.signature(CellposeModel.eval); k=ModelConfig().eval_kwargs(); '
         's.bind(None,x=[],**{n:v for n,v in k.items() if n in s.parameters}); '
         'print("Cellpose",version("cellpose")); print("Supported-key binding passes."); '
         'print("Current Model Compare defaults contain unsupported arguments:",[n for n in k if n not in s.parameters]); '
         's.bind(None,x=[],**k)'], 1, "unexpected keyword argument 'invert'"),
        ('02_actual_comparison', ['python', helper.name, '--source', 'inputs', '--output', 'comparison'],
         0, 'Matched 91 pairs'),
        ('03_outputs', ['python', '-c',
         'import json; r=json.load(open("comparison/run.json")); '
         '[print(k,v) for k,v in r["independent_checks"].items()]; '
         '[print(k,r[k]) for k in ("same_mask_control_passed","ground_truth_used",'
         '"accuracy_validated","different_model_weights_compared","gui_workflow_completed")]; '
         'print("Saved CSV and figure describe agreement, not a winner.")'],
         0, 'Saved CSV and figure describe agreement'),
    ]
    outcomes = []
    for scene, command, code, expected in commands:
        print('\033[2J\033[3J\033[H', end='', flush=True)
        print('$ ' + shlex.join(command), flush=True)
        result = subprocess.run(command, cwd=work, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=90)
        print(result.stdout, end='', flush=True)
        print(f'\nExit status: {result.returncode}', flush=True)
        passed = accepted_command(result, code, expected)
        outcomes.append({'scene': scene, 'command': command, 'returncode': result.returncode,
                         'output': result.stdout, 'accepted': passed})
        write(captures / 'commands.json', outcomes)
        write(captures / 'terminal_ready.json', {'scene': scene, 'accepted': passed})
        input('\nPress Enter to continue the recording. ')
        if not passed:
            return 1
    target = work / 'comparison/comparison.png'
    desktop = PrivateDesktop(stage)
    viewer = subprocess.Popen(['eog', '--new-instance', str(target)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        wid, title = desktop.find(target.name, time.sleep)
        desktop.show(wid); time.sleep(1.5)
        figure = {'scene': '04_actual_comparison_figure', 'path': str(target),
                  'sha256': digest(target), 'viewer': 'eog', 'window_title': title}
        write(captures / 'figures.json', [figure])
        write(captures / 'terminal_ready.json', {'scene': figure['scene'], 'accepted': True})
        input('\nPress Enter after recording the actual image viewer. ')
    finally:
        viewer.terminate(); viewer.wait(timeout=5); desktop.close()
    report = read(work / 'comparison/run.json')
    if not report['accepted'] or report['helper_sha256'] != digest(helper):
        raise ValueError('Recorded helper differs from the verified API example')
    for name, expected in report['artifacts'].items():
        if digest(work / 'comparison' / name) != expected:
            raise ValueError('A recorded output changed')
    write(captures / 'scientific_acceptance.json', {
        'accepted': True, 'scope': 'Pure compare_masks API with genuine saved segmentations',
        'run': report, 'figure': figure, 'private_work': str(work), 'helper_sha256': digest(helper),
        'gui_workflow_completed': False, 'inference_performed': False,
        'results_injected': False, 'accuracy_validated': False, 'published': False})
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
                                window_title='spaCR Model Compare API', expected_scenes=5,
                                module='model_compare', pipeline_requested=True,
                                refocus_terminal_after_capture=True, terminal_zoom=2.8)
    if (stage / 'captures' / NAME).exists():
        raise FileExistsError('Preserve the previous API capture')
    env = dict(os.environ)
    for key, name in [('XDG_CONFIG_HOME', 'config'), ('XDG_DATA_HOME', 'data'),
                      ('XDG_CACHE_HOME', 'cache'), ('XDG_RUNTIME_DIR', 'runtime')]:
        path = stage / 'desktop' / NAME / name
        path.mkdir(parents=True, exist_ok=True, mode=0o700); env[key] = str(path)
    env.update(SPACR_TUTORIAL_PRIVATE_DESKTOP='1', GIO_USE_VFS='local',
               GSETTINGS_BACKEND='memory', GTK_USE_PORTAL='0', XDG_CURRENT_DESKTOP='SPACR_TUTORIAL',
               NO_AT_BRIDGE='1', QT_QPA_PLATFORM='xcb',
               PATH=str(Path(sys.executable).parent) + os.pathsep + env['PATH'],
               PYTHONPATH=str(REPO), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    return subprocess.run(['xvfb-run', '-a', '-s', '-screen 0 3840x2160x24',
        'dbus-run-session', '--', sys.executable, str(Path(__file__).resolve()),
        '--stage', str(stage), '--inside'], env=env, timeout=500).returncode


if __name__ == '__main__':
    raise SystemExit(main())
