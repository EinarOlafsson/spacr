"""Record the actual two-entry barcode-set API and saved GUI count inspection."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from capture_cli import accepted_command, capture_terminal
from map_barcodes_data import digest
from stage_lesson import REPO, read, write

NAME = 'map_barcodes_api'


def terminal_driver(stage):
    from capture_diagnostics import PrivateDesktop
    captures = stage / 'captures' / NAME
    helper = Path(__file__).with_name('map_barcodes_example.py')
    output = stage / 'map_api_recorded_output'
    command = ['python', str(helper), '--stage', str(stage), '--output', str(output)]
    print('$ ' + shlex.join(command), flush=True)
    result = subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=120)
    print(result.stdout, end='', flush=True)
    passed = accepted_command(result, 0, 'Every saved count reconciles.')
    write(captures / 'commands.json', [{'scene': '01_actual_two_barcode_api',
          'command': command, 'returncode': result.returncode,
          'output': result.stdout, 'accepted': passed}])
    write(captures / 'terminal_ready.json', {'scene': '01_actual_two_barcode_api', 'accepted': passed})
    input('\nPress Enter to continue the recording. ')
    if not passed:
        return 1
    desktop = PrivateDesktop(stage)
    figure = output / 'mapped_read_depth.png'
    viewer = subprocess.Popen(['eog', '--new-instance', str(figure)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        wid, title = desktop.find(figure.name, time.sleep)
        desktop.show(wid); time.sleep(1)
        write(captures / 'figures.json', [{'path': str(figure), 'sha256': digest(figure),
                                          'window_title': title, 'viewer': 'eog'}])
        write(captures / 'terminal_ready.json', {'scene': '02_real_count_figure', 'accepted': True})
        input('\nPress Enter after recording the real figure viewer. ')
    finally:
        viewer.terminate(); viewer.wait(timeout=5); desktop.close()
    proof = read(output / 'run.json')
    if proof['helper_sha256'] != digest(helper) or not proof['accepted']:
        raise ValueError('The actual helper did not produce verified output')
    write(captures / 'scientific_acceptance.json', proof)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--inside', action='store_true')
    parser.add_argument('--terminal-driver', action='store_true')
    args = parser.parse_args(); stage = args.stage.resolve()
    if args.terminal_driver:
        return terminal_driver(stage)
    if args.inside:
        return capture_terminal(stage, driver=Path(__file__).resolve(), capture_name=NAME,
                window_title='spaCR Map Barcodes API', expected_scenes=2, module='map_barcodes',
                pipeline_requested=True, refocus_terminal_after_capture=True, terminal_zoom=2.4)
    if (stage / 'captures' / NAME).exists():
        raise FileExistsError('Preserve the previous API recording')
    env = dict(os.environ)
    for key, name in [('XDG_CONFIG_HOME','config'), ('XDG_DATA_HOME','data'),
                      ('XDG_CACHE_HOME','cache'), ('XDG_RUNTIME_DIR','runtime')]:
        path = stage / 'desktop' / NAME / name
        path.mkdir(parents=True, exist_ok=True, mode=0o700); env[key] = str(path)
    env.update(SPACR_TUTORIAL_PRIVATE_DESKTOP='1', GIO_USE_VFS='local',
               GSETTINGS_BACKEND='memory', GTK_USE_PORTAL='0', XDG_CURRENT_DESKTOP='SPACR_TUTORIAL',
               NO_AT_BRIDGE='1', QT_QPA_PLATFORM='xcb',
               PATH=str(Path(sys.executable).parent) + os.pathsep + env['PATH'],
               PYTHONPATH=str(REPO), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    return subprocess.run(['xvfb-run','-a','-s','-screen 0 3840x2160x24','dbus-run-session','--',
        sys.executable, str(Path(__file__).resolve()), '--stage', str(stage), '--inside'],
        env=env, timeout=400).returncode


if __name__ == '__main__':
    raise SystemExit(main())
