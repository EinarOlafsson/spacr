"""Record four real OPS tiles through the bounded geometry API, with its figure."""
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

NAME = 'ops_1507_geometry_api_v2'
SOURCE = Path('/nas_mnt/data/ops/OpticalPooledScreens_data/screenA/20200202_6W-LaC024A/sequencing/images/input/c1')


def terminal_driver(stage):
    from capture_diagnostics import PrivateDesktop
    captures = stage / 'captures' / NAME
    work = Path(tempfile.mkdtemp(prefix='ops-geometry-recorded-', dir=stage))
    helper = Path(__file__).with_name('ops_geometry_example.py')
    shutil.copy2(helper, work / helper.name)
    (work / 'cycle1').symlink_to(SOURCE, target_is_directory=True)
    commands = [
        ('00_scope', ['python', '-c',
         'import spacr; print("spaCR",spacr.__version__); '
         'print("Four real cycle-1 DAPI tiles: registration and composition only."); '
         'print("CPU example. No segmentation, barcode decoding, or full-well run."); '
         'print("The current OPS GUI Run uses a different, legacy engine.")'], 0, 'registration and composition only'),
        ('01_inputs', ['python', '-c',
         'from pathlib import Path; import tifffile; '
         'from ops_geometry_example import SITES; from spacr.ops_layout import round_well_layout; '
         'layout=round_well_layout(333); '
         'print("Retain the original 333-site layout and site indices."); '
         '[(print("Site",s,"position",layout.position(s),"shape",'
         'tifffile.imread(next(Path("cycle1").glob(f"*_Site-{s}.tif"))).shape)) for s in SITES]; '
         'print("Plane 0 is DAPI. Source files are read only; output goes elsewhere.")'], 0, '(5, 1480, 1480)'),
        ('02_geometry', ['python', helper.name, '--source', 'cycle1', '--output', 'patch'],
         0, '4/4 placed, 4/4 edges'),
        ('03_independent_checks', ['python', '-c',
         'import json; r=json.load(open("patch/run.json")); '
         'print("Independent pixel correlation: aligned versus wrong 17-pixel offsets"); '
         '[print(c["sites"],"aligned",round(c["aligned_pearson"],3),'
         '"controls",[round(v,3) for v in c["offset_17px_controls"]]) '
         'for c in r["independent_overlap_checks"]]; '
         'print("Canvas:",r["canvas"],"; four tiles, not an entire well."); '
         'print("Coverage zero marks missing acquisition pixels, not biological background.")'],
         0, 'Canvas: [2756, 2756]'),
        ('04_provenance', ['python', '-c',
         'import json; from pathlib import Path; from ops_geometry_example import digest; '
         'r=json.load(open("patch/run.json")); '
         'assert all(digest(Path("cycle1")/s["name"])==s["sha256"] for s in r["sources"]); '
         'assert all(digest(Path("patch")/n)==h for n,h in r["artifacts"].items()); '
         'print("All four source hashes unchanged; saved output hashes verified."); '
         '[print(k,r[k]) for k in ("backends","full_pipeline_completed",'
         '"gui_pipeline_completed","segmentation_or_decoding_performed")]; '
         'print("Never average sequencing cycles; this example composes spatial neighbors in one plane.")'],
         0, 'source hashes unchanged'),
    ]
    outcomes = []
    for scene, command, expected, expected_text in commands:
        print('\033[2J\033[3J\033[H', end='', flush=True)
        print('$ ' + shlex.join(command), flush=True)
        result = subprocess.run(command, cwd=work, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=180)
        print(result.stdout, end='', flush=True)
        print(f'\nExit status: {result.returncode}', flush=True)
        passed = accepted_command(result, expected, expected_text)
        outcomes.append({'scene': scene, 'command': command, 'returncode': result.returncode,
                         'output': result.stdout, 'accepted': passed})
        write(captures / 'commands.json', outcomes)
        write(captures / 'terminal_ready.json', {'scene': scene, 'accepted': passed})
        input('\nPress Enter to continue the recording. ')
        if not passed:
            return 1
    target = work / 'patch/geometry_and_coverage.png'
    desktop = PrivateDesktop(stage)
    viewer = subprocess.Popen(['eog', '--new-instance', str(target)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        wid, title = desktop.find(target.name, time.sleep)
        desktop.show(wid)
        time.sleep(1.5)
        figure = {'scene': '05_geometry_figure', 'path': str(target),
                  'sha256': digest(target), 'viewer': 'eog', 'window_title': title}
        write(captures / 'figures.json', [figure])
        write(captures / 'terminal_ready.json', {'scene': figure['scene'], 'accepted': True})
        input('\nPress Enter after recording the actual image viewer. ')
    finally:
        viewer.terminate()
        viewer.wait(timeout=5)
        desktop.close()
    report = read(work / 'patch/run.json')
    if not report['accepted'] or report['helper_sha256'] != digest(helper):
        raise ValueError('Recorded run does not match the verified helper')
    for name, sha in report['artifacts'].items():
        if digest(work / 'patch' / name) != sha:
            raise ValueError('Recorded output changed')
    for record in report['sources']:
        if digest(SOURCE / record['name']) != record['sha256']:
            raise ValueError('An original acquisition file changed')
    write(captures / 'scientific_acceptance.json', {
        'accepted': True, 'scope': 'Four real tiles through geometry/composition APIs only',
        'run': report, 'figure': figure, 'private_work': str(work),
        'source_unchanged': True, 'helper_sha256': digest(helper),
        'gui_workflow_completed': False, 'inputs_injected': False,
        'segmentation_or_decoding_performed': False, 'published': False})
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
                                window_title='spaCR OPS geometry API', expected_scenes=6,
                                module='ops', pipeline_requested=True,
                                refocus_terminal_after_capture=True, terminal_zoom=2.8)
    if (stage / 'captures' / NAME).exists():
        raise FileExistsError('Preserve the existing API capture')
    env = dict(os.environ); root = stage / 'desktop' / NAME
    for key, name in [('XDG_CONFIG_HOME', 'config'), ('XDG_DATA_HOME', 'data'),
                      ('XDG_CACHE_HOME', 'cache'), ('XDG_RUNTIME_DIR', 'runtime')]:
        path = root / name; path.mkdir(parents=True, exist_ok=True, mode=0o700)
        env[key] = str(path)
    env.update(SPACR_TUTORIAL_PRIVATE_DESKTOP='1', GIO_USE_VFS='local',
               GSETTINGS_BACKEND='memory', GTK_USE_PORTAL='0', XDG_CURRENT_DESKTOP='SPACR_TUTORIAL',
               NO_AT_BRIDGE='1', QT_QPA_PLATFORM='xcb',
               PATH=str(Path(sys.executable).parent) + os.pathsep + env['PATH'],
               PYTHONPATH=str(REPO), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    return subprocess.run(['xvfb-run', '-a', '-s', '-screen 0 3840x2160x24',
        'dbus-run-session', '--', sys.executable, str(Path(__file__).resolve()),
        '--stage', str(stage), '--inside'], env=env, timeout=600).returncode


if __name__ == '__main__':
    raise SystemExit(main())
