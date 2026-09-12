#!/usr/bin/env python3
"""Record real headless commands in a terminal on an isolated X11 desktop.

The default API driver only lists, describes, inspects and dry-runs commands.
Explicit workflow drivers may export to private tutorial directories. The
terminal executes the commands; its output is never painted into a mock UI.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from stage_lesson import DEFAULT_STAGE, REPO, read, write


def accepted_command(result, expected_code, expected_text):
    """A refusal can be the intended result; status alone is not evidence."""
    return result.returncode == expected_code and expected_text in result.stdout


def terminal_driver(stage):
    work = stage / 'api_workflow'
    captures = stage / 'captures/api_terminal'
    work.mkdir(parents=True, exist_ok=True)
    captures.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stage / 'captures/regression_release/batch_settings.json',
                 work / 'regression.settings.json')
    commands = [
        ('01_version', ['python', '-c', 'import spacr; print(spacr.__version__)'], 0, '1.5.'),
        ('02_list', ['spacr-run', '--list'], 0, 'classify_merged'),
        ('03_describe', ['spacr-run', '--describe', 'regression'], 0, 'perform_regression'),
        ('04_gui_only', ['spacr-run', 'annotate'], 2, 'no batch equivalent'),
        ('05_report_api', ['python', '-c', 'import inspect; from spacr.report import build_report; print(inspect.signature(build_report))'], 0, 'src'),
        ('06_dry_run', ['spacr-run', 'regression', '--settings', 'regression.settings.json', '--dry-run'], 0, 'was not called'),
    ]
    outcomes = []
    for scene, command, expected_code, expected_text in commands:
        print('\033[2J\033[3J\033[H', end='', flush=True)
        print('$ ' + shlex.join(command), flush=True)
        result = subprocess.run(command, cwd=work, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=90)
        print(result.stdout, end='', flush=True)
        print(f'\nExit status: {result.returncode}', flush=True)
        passed = accepted_command(result, expected_code, expected_text)
        outcomes.append({'scene': scene, 'command': command,
                         'returncode': result.returncode, 'output': result.stdout,
                         'accepted': passed})
        write(captures / 'commands.json', outcomes)
        write(captures / 'terminal_ready.json', {'scene': scene, 'accepted': passed})
        input('\nPress Enter to continue the recording. ')
        if not passed:
            return 1
    return 0


def capture_terminal(stage, *, driver=None, capture_name='api_terminal',
                     window_title='spaCR Python API', expected_scenes=6,
                     module='api', pipeline_requested=False,
                     refocus_terminal_after_capture=False, terminal_zoom=1.6):
    if (Path(capture_name).name != capture_name or capture_name in {'', '.', '..'}
            or expected_scenes < 1 or not 0.5 <= terminal_zoom <= 4):
        raise ValueError('A terminal workflow needs a simple name and positive scene count')
    from PySide6.QtWidgets import QApplication
    from capture_diagnostics import PrivateDesktop

    app = QApplication([])
    captures = stage / 'captures' / capture_name
    captures.mkdir(parents=True, exist_ok=True)
    write(captures / 'provenance.json', {'completed_capture': False})
    write(captures / 'terminal_ready.json', {'scene': None})
    terminal = subprocess.Popen([
        'gnome-terminal', '--wait', '--hide-menubar', '--title=' + window_title,
        f'--zoom={terminal_zoom}', '--', sys.executable, str(driver or Path(__file__).resolve()),
        '--stage', str(stage), '--terminal-driver'])

    def settle(seconds=0.6):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            app.processEvents()
            time.sleep(0.02)

    desktop = PrivateDesktop(stage)
    xtest = ctypes.CDLL('libXtst.so.6')
    xtest.XTestFakeKeyEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint,
                                      ctypes.c_int, ctypes.c_ulong]
    desktop.x.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    desktop.x.XKeysymToKeycode.restype = ctypes.c_ubyte
    desktop.x.XSetInputFocus.argtypes = [ctypes.c_void_p, ctypes.c_ulong,
                                        ctypes.c_int, ctypes.c_ulong]

    def key(symbol, shift=False):
        code = desktop.x.XKeysymToKeycode(desktop.display, symbol)
        shift_code = desktop.x.XKeysymToKeycode(desktop.display, 0xffe1)
        if shift:
            xtest.XTestFakeKeyEvent(desktop.display, shift_code, 1, 0)
        xtest.XTestFakeKeyEvent(desktop.display, code, 1, 0)
        xtest.XTestFakeKeyEvent(desktop.display, code, 0, 0)
        if shift:
            xtest.XTestFakeKeyEvent(desktop.display, shift_code, 0, 0)
        desktop.x.XFlush(desktop.display)

    frames = {}
    try:
        wid, title = desktop.find(window_title, settle)
        desktop.show(wid)
        desktop.x.XSetInputFocus(desktop.display, wid, 1, 0)
        desktop.x.XFlush(desktop.display)
        seen = set()
        deadline = time.monotonic() + 600
        while terminal.poll() is None:
            if time.monotonic() > deadline:
                raise TimeoutError('The bounded terminal recording exceeded its limit')
            ready = read(captures / 'terminal_ready.json')
            scene = ready.get('scene')
            if scene and scene not in seen:
                settle(1)
                if module == 'api' and scene == '02_list':
                    for _ in range(12):
                        key(0xff55, shift=True)  # Actual terminal scrollback, Shift+PageUp.
                    settle()
                image = app.primaryScreen().grabWindow(0)
                if (image.width(), image.height()) != (3840, 2160):
                    raise RuntimeError('The terminal desktop is not native 4K')
                path = captures / (scene + '.png')
                if not image.save(str(path), 'PNG'):
                    raise RuntimeError('The real terminal could not be captured')
                frames[scene] = {'image': path.name,
                                 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                 'buttons': [], 'terminal_title': title}
                write(captures / 'frames.json', frames)
                print('captured ' + module + '/' + scene, flush=True)
                if not ready['accepted']:
                    raise RuntimeError('The actual command did not meet its expected outcome')
                seen.add(scene)
                if refocus_terminal_after_capture:
                    # An explicit figure-viewer driver may place a real viewer
                    # above the terminal. Capture it first, then send Return
                    # only to this private terminal, not to the user's desktop.
                    desktop.x.XSetInputFocus(desktop.display, wid, 1, 0)
                    desktop.x.XFlush(desktop.display)
                key(0xff0d)  # Return advances only our private terminal driver.
            settle(0.2)
        if terminal.returncode != 0 or len(seen) != expected_scenes:
            raise RuntimeError('The terminal recording did not finish every real command')
        write(captures / 'provenance.json', {
            'completed_capture': True, 'module': module,
            'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            'version': __import__('spacr').__version__,
            'app_source_modified': False, 'actual_system_terminal': True,
            'pipeline_execution_requested': bool(pipeline_requested),
            'private_display': os.environ['DISPLAY']})
    finally:
        if terminal.poll() is None:
            terminal.terminate()
            try:
                terminal.wait(timeout=5)
            except subprocess.TimeoutExpired:
                terminal.kill()
                terminal.wait(timeout=5)
        desktop.close()
        app.quit()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=DEFAULT_STAGE)
    parser.add_argument('--inside', action='store_true')
    parser.add_argument('--terminal-driver', action='store_true')
    args = parser.parse_args()
    stage = args.stage.resolve()
    if args.terminal_driver:
        return terminal_driver(stage)
    if args.inside:
        return capture_terminal(stage)
    env = dict(os.environ)
    root = stage / 'desktop/api'
    for key, name in [('XDG_CONFIG_HOME', 'config'), ('XDG_DATA_HOME', 'data'),
                      ('XDG_CACHE_HOME', 'cache'), ('XDG_RUNTIME_DIR', 'runtime')]:
        path = root / name
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        env[key] = str(path)
    env.update(SPACR_TUTORIAL_PRIVATE_DESKTOP='1', GIO_USE_VFS='local',
               GSETTINGS_BACKEND='memory', GTK_USE_PORTAL='0',
               XDG_CURRENT_DESKTOP='SPACR_TUTORIAL', NO_AT_BRIDGE='1',
               QT_QPA_PLATFORM='xcb', PATH=str(Path(sys.executable).parent) + os.pathsep + env['PATH'],
               PYTHONPATH=str(REPO))
    return subprocess.run([
        'xvfb-run', '-a', '-s', '-screen 0 3840x2160x24', 'dbus-run-session', '--',
        'bwrap', '--die-with-parent', '--bind', '/', '/', '--dev-bind', '/dev', '/dev',
        '--bind', str(stage / 'example_data'), str(Path.home() / '.cache/spacr/example_data'),
        '--', sys.executable, str(Path(__file__).resolve()), '--stage', str(stage), '--inside',
    ], env=env, timeout=650).returncode


if __name__ == '__main__':
    raise SystemExit(main())
