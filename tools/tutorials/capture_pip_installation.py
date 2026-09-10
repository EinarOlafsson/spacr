#!/usr/bin/env python3
"""Record real verification commands and the installed package's GUI.

The previously completed clean-install receipt is mandatory. This recorder
does not install an editable checkout, replay invented output, or claim to
record an installation that has already finished. Setup/tour are explicitly
skipped for the pip launch, in a private profile on a private display. The
older Conda release lacks that flag and the newer preload preference; its
normal launcher and default preload behaviour are recorded without patching.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

from stage_lesson import DEFAULT_STAGE, REPO, read, write


def installation(stage, folder, route='pip'):
    root = folder.resolve()
    if not root.is_relative_to(stage / 'installation_runs'):
        raise ValueError('Only a private tutorial installation can be recorded')
    result = read(root / 'receipt.json')
    expected, count, relative = {
        'pip': (6, 'six', 'venv'),
        'conda': (5, 'five', 'env'),
        'linux_installer': (3, 'three', 'runtime/venv'),
    }[route]
    if (not result.get('accepted') or len(result.get('steps', [])) != expected
            or any(row.get('returncode') != 0 or not row.get('completed')
                   for row in result['steps'])):
        raise ValueError(f'The real installation must have {count} successful checks')
    return root, root / relative, result


def installed_version(prior, route):
    if route == 'linux_installer':
        return prior['installer_tag'].removeprefix('v')
    return prior['pypi' if route == 'pip' else 'channel']['version']


def commands(route='pip', version='1.5.0.5', prefix=None):
    result = [
        ('01_python_version', ['python', '--version'], 'Python 3.12.'),
        ('02_pip_environment', ['python', '-m', 'pip', '--version'], 'venv/lib/python3.12/site-packages/pip'),
        ('03_installed_versions', ['python', '-c',
         'import sys,spacr,PySide6; print("Python:", sys.executable); '
         'print("spaCR:", spacr.__version__); print("Qt:", PySide6.__version__); '
         'print("Package:", spacr.__file__)'], 'spaCR: ' + version),
        ('04_dependency_check', ['python', '-m', 'pip', 'check'], 'No broken requirements found.'),
        ('05_doctor', ['spacr-doctor', '--no-gpu-probe'], 'pylibCZIrw'),
    ]
    if route == 'conda':
        result[1] = ('02_conda_environment', ['/home/olafsson/anaconda3/bin/conda',
                     'list', '--prefix', str(prefix), 'spacr'], 'conda-forge')
        result[3] = ('04_environment_prefix', ['python', '-c',
                     'import sys; print(sys.prefix)'], str(prefix))
        result[4] = ('05_doctor', ['spacr-doctor'], 'pylibCZIrw')
    elif route == 'linux_installer':
        profile = Path(prefix).parent / 'install-profile.json'
        launcher = Path(prefix).parents[1] / 'bin/spacr'
        result[1] = ('02_installer_backend', ['python', '-c',
                     'import json; from pathlib import Path; '
                     f'print(json.dumps(json.loads(Path({str(profile)!r}).read_text()), indent=2))'],
                     '"requested_backend": "auto"')
        result[3] = ('04_installed_launcher', ['python', '-c',
                     'from pathlib import Path; '
                     f'p=Path({str(launcher)!r}); print(p); print(p.read_text())'],
                     str(Path(prefix) / 'bin/python'))
    return result


def terminal_driver(stage, root, capture, route='pip', version='1.5.0.5', prefix=None):
    outcomes = []
    for name, command, expected in commands(route, version, prefix):
        print('\033[2J\033[3J\033[H', end='', flush=True)
        print('$ ' + shlex.join(command), flush=True)
        result = subprocess.run(command, cwd=root, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=120)
        print(result.stdout, end='', flush=True)
        print(f'\nExit status: {result.returncode}', flush=True)
        accepted = result.returncode == 0 and expected in result.stdout
        outcomes.append(dict(scene=name, command=command, output=result.stdout,
                             returncode=result.returncode, accepted=accepted))
        write(capture / 'commands.json', outcomes)
        write(capture / 'terminal_ready.json', dict(scene=name, accepted=accepted))
        input('\nPress Enter to continue the recording. ')
        if not accepted:
            return 1
    return 0


def close_window(desktop, wid):
    """Send the same WM_DELETE_WINDOW protocol as a window-manager close."""
    class Data(ctypes.Union):
        _fields_ = [('b', ctypes.c_char * 20), ('s', ctypes.c_short * 10),
                    ('l', ctypes.c_long * 5)]

    class Client(ctypes.Structure):
        _fields_ = [('type', ctypes.c_int), ('serial', ctypes.c_ulong),
                    ('send_event', ctypes.c_int), ('display', ctypes.c_void_p),
                    ('window', ctypes.c_ulong), ('message_type', ctypes.c_ulong),
                    ('format', ctypes.c_int), ('data', Data)]

    class Event(ctypes.Union):
        _fields_ = [('client', Client), ('pad', ctypes.c_long * 24)]

    desktop.x.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    desktop.x.XInternAtom.restype = ctypes.c_ulong
    desktop.x.XSendEvent.argtypes = [ctypes.c_void_p, ctypes.c_ulong,
                                    ctypes.c_int, ctypes.c_long, ctypes.POINTER(Event)]
    event = Event()
    event.client.type = 33
    event.client.display = desktop.display
    event.client.window = wid
    event.client.message_type = desktop.x.XInternAtom(desktop.display, b'WM_PROTOCOLS', 0)
    event.client.format = 32
    event.client.data.l[0] = desktop.x.XInternAtom(desktop.display, b'WM_DELETE_WINDOW', 0)
    desktop.x.XSendEvent(desktop.display, wid, False, 0, ctypes.byref(event))
    desktop.x.XFlush(desktop.display)


def visible_app_window(tree):
    """Ignore Qt's identically named 1x1 application-group helper window."""
    matches = [(int(wid, 16), int(width) * int(height)) for wid, name, width, height in
               re.findall(r'(0x[0-9a-f]+) "([^"]+)".*? (\d+)x(\d+)[+-]\d+', tree)
               if name == 'spaCR' and int(width) >= 640 and int(height) >= 400]
    if not matches:
        raise ValueError('No visible spaCR application window exists yet')
    return max(matches, key=lambda item: item[1])[0]


def privacy_window_owned(title, properties, gui_pid):
    expected = 'spaCR privacy and optional account setup'
    return (title in {expected, expected + ' — spaCR'} and re.search(
        r'^_NET_WM_PID\(CARDINAL\) =\s*' + str(gui_pid) + r'\s*$',
        properties, re.MULTILINE) is not None)


def dialog_dismissed(returncode, info, tree, wid):
    # Qt may destroy the rejected dialog instead of retaining a hidden window.
    return ((returncode == 0 and 'Map State: IsUnMapped' in info)
            or (returncode == 1 and re.search(
                r'^\s*' + re.escape(hex(wid)) + r'\s+', tree, re.MULTILINE) is None))


def record(stage, root, venv, prior, capture, route='pip'):
    from PySide6.QtWidgets import QApplication
    from capture_diagnostics import PrivateDesktop

    capture.mkdir(parents=True)
    version = installed_version(prior, route)
    provenance = dict(completed_capture=False, module=route + '_install',
                      commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                      actual_system_terminal=True, installation_recorded_live=False,
                      verification_recorded_live=True, application_source_modified=False,
                      private_display=os.environ['DISPLAY'], installed_environment=str(venv),
                      setup_skipped_explicitly=route != 'conda', first_run_tour_marked_seen=True,
                      format_scope=f'Linux, fresh {route} package; not other platforms',
                      doctor_gpu_allocation_probe_requested=route == 'conda')
    write(capture / 'provenance.json', provenance)
    identity = subprocess.check_output([str(venv / 'bin/python'), '-I', '-c',
        'import pathlib,sys,json,spacr; p=pathlib.Path(spacr.__file__).resolve(); '
        'assert p.is_relative_to(pathlib.Path(sys.prefix)); '
        'print(json.dumps(dict(version=spacr.__version__,package=str(p),prefix=sys.prefix)))'],
        cwd=root, text=True, timeout=90)
    import json
    provenance['installed_identity'] = json.loads(identity)
    if provenance['installed_identity']['version'] != version:
        raise RuntimeError('The installed package changed since the successful preflight')
    # Real preference APIs, only in this private profile. No GUI methods,
    # package metadata, consent answers, or pipeline callbacks are replaced.
    preferences = (
        'from spacr.qt.first_run import mark_tour_seen; '
        'from spacr.qt.preferences import set_theme,set_font_scale; '
        'mark_tour_seen(); set_theme("dark"); set_font_scale(1.5)')
    if route != 'conda':
        preferences += '; from spacr.qt.preferences import set_preload_policy; set_preload_policy("on_demand")'
    subprocess.run([str(venv / 'bin/python'), '-I', '-c', preferences],
        cwd=root, check=True, timeout=90)
    provenance['recording_preferences'] = dict(theme='dark', font_scale=1.5,
                                              preload='unmodified release default' if route == 'conda' else 'on_demand')
    app = QApplication([])
    desktop = PrivateDesktop(stage)
    frames = {}
    children = []

    def settle(seconds=0.3):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            app.processEvents()
            time.sleep(.02)

    def snapshot(name):
        pixmap = app.primaryScreen().grabWindow(0)
        if (pixmap.width(), pixmap.height()) != (3840, 2160):
            raise RuntimeError('The actual private display must be native 4K')
        path = capture / (name + '.png')
        if not pixmap.save(str(path), 'PNG'):
            raise RuntimeError('Could not save the actual display')
        frames[name] = dict(image=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), buttons=[])
        write(capture / 'frames.json', frames)
        print('Captured ' + route + '/' + name, flush=True)

    try:
        write(capture / 'terminal_ready.json', dict(scene=None))
        terminal_title = f'spaCR {route} verification'
        terminal = subprocess.Popen(['gnome-terminal', '--wait', '--hide-menubar',
            '--title=' + terminal_title, '--zoom=1.8', '--', sys.executable,
            str(Path(__file__).resolve()), '--stage', str(stage), '--installation', str(root),
            '--capture-name', capture.name, '--route', route, '--terminal-driver'], cwd=root)
        children.append(terminal)
        wid, _ = desktop.find(terminal_title, settle)
        desktop.show(wid)
        desktop.x.XSetInputFocus.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        desktop.x.XSetInputFocus(desktop.display, wid, 1, 0)
        desktop.x.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        desktop.x.XKeysymToKeycode.restype = ctypes.c_ubyte
        xtest = ctypes.CDLL('libXtst.so.6')
        xtest.XTestFakeKeyEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]
        seen = set()
        deadline = time.monotonic() + 300
        while terminal.poll() is None:
            if time.monotonic() > deadline:
                raise TimeoutError('Installed-package terminal exceeded five minutes')
            ready = read(capture / 'terminal_ready.json')
            name = ready.get('scene')
            if name and name not in seen:
                settle(1)
                snapshot(name)
                if not ready['accepted']:
                    raise RuntimeError('A real command did not produce its required result')
                seen.add(name)
                code = desktop.x.XKeysymToKeycode(desktop.display, 0xff0d)
                xtest.XTestFakeKeyEvent(desktop.display, code, 1, 0)
                xtest.XTestFakeKeyEvent(desktop.display, code, 0, 0)
                desktop.x.XFlush(desktop.display)
            settle()
        if terminal.returncode != 0 or len(seen) != len(commands(route, version, venv)):
            raise RuntimeError('All five real verification commands must succeed')
        with (capture / 'installed_gui.log').open('w') as log:
            # The older channel package does not implement --no-setup.
            launcher = root / 'bin/spacr' if route == 'linux_installer' else venv / 'bin/spacr'
            command = [str(launcher)] + ([] if route == 'conda' else ['--no-setup'])
            gui = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            children.append(gui)
            deadline = time.monotonic() + 120
            while True:
                tree = subprocess.check_output(['xwininfo', '-root', '-tree'], text=True)
                try:
                    wid = visible_app_window(tree)
                    break
                except ValueError:
                    if gui.poll() is not None or time.monotonic() > deadline:
                        raise
                    settle()
            provenance['native_window_tree'] = tree
            title = 'spaCR'
            owner = subprocess.check_output(['xprop', '-id', hex(wid), '_NET_WM_PID'], text=True)
            if not re.search(r'=\s*' + str(gui.pid) + r'\s*$', owner):
                raise RuntimeError('The visible window is not owned by our installed GUI process')
            desktop.x.XMoveResizeWindow(desktop.display, wid, 0, 0, 3840, 2160)
            desktop.x.XMapRaised(desktop.display, wid)
            desktop.x.XFlush(desktop.display)
            settle(5)
            geometry = subprocess.check_output(['xwininfo', '-id', hex(wid)], text=True)
            if 'Width: 3840' not in geometry or 'Height: 2160' not in geometry:
                raise RuntimeError('The installed application did not reach the requested 4K window size')
            provenance['native_window_geometry'] = geometry
            if gui.poll() is not None:
                raise RuntimeError('Installed GUI exited before the native screenshot')
            if route == 'linux_installer':
                # --no-setup skips account setup, not the installer's separate
                # unanswered privacy dialog. Record it, then use its native
                # reject/Escape action: no consent or account is enabled.
                expected_title = 'spaCR privacy and optional account setup'
                dialog_wid, dialog_title = desktop.find(expected_title, settle)
                dialog_owner = subprocess.check_output(
                    ['xprop', '-id', hex(dialog_wid), '_NET_WM_PID', 'WM_TRANSIENT_FOR'], text=True)
                write(capture / 'privacy_window_identity.json', dict(
                    title=dialog_title, properties=dialog_owner, gui_pid=gui.pid,
                    main_window=hex(wid), dialog_window=hex(dialog_wid)))
                if not privacy_window_owned(dialog_title, dialog_owner, gui.pid):
                    raise RuntimeError('Privacy prompt is not our installed application dialog')
                desktop.x.XMapRaised(desktop.display, dialog_wid)
                desktop.x.XSetInputFocus(desktop.display, dialog_wid, 1, 0)
                desktop.x.XFlush(desktop.display)
                settle(.5)
                snapshot('07_privacy_keep_off')
                code = desktop.x.XKeysymToKeycode(desktop.display, 0xff1b)
                xtest.XTestFakeKeyEvent(desktop.display, code, 1, 0)
                xtest.XTestFakeKeyEvent(desktop.display, code, 0, 0)
                desktop.x.XFlush(desktop.display)
                settle(2)
                after = subprocess.run(['xwininfo', '-id', hex(dialog_wid)], text=True,
                                       capture_output=True)
                after_tree = subprocess.check_output(['xwininfo', '-root', '-tree'], text=True)
                if not dialog_dismissed(after.returncode, after.stdout, after_tree, dialog_wid):
                    raise RuntimeError('The native privacy rejection did not dismiss the dialog')
                write(capture / 'privacy_rejection.json', dict(
                    returncode=after.returncode, window_info=after.stdout,
                    window_error=after.stderr, remaining_tree=after_tree))
                provenance['privacy_dialog'] = dict(title=dialog_title,
                    owner_pid=gui.pid, native_reject_key='Escape', shown_before_rejection=True,
                    hidden_after_rejection=True, optional_choices_enabled=False)
            snapshot('06_installed_home')
            close_window(desktop, wid)
            deadline = time.monotonic() + 45
            while gui.poll() is None and time.monotonic() < deadline:
                settle()
            if gui.poll() != 0:
                snapshot('99_gui_shutdown_failure')
                raise RuntimeError('Installed GUI did not close normally with exit zero')
            provenance['gui'] = dict(command=command, title=title, returncode=gui.returncode,
                                     normal_window_close=True, analysis_started=False)
        provenance['completed_capture'] = True
        write(capture / 'provenance.json', provenance)
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
        desktop.close()
        app.quit()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=DEFAULT_STAGE)
    parser.add_argument('--installation', type=Path, required=True)
    parser.add_argument('--route', choices=('pip', 'conda', 'linux_installer'), default='pip')
    parser.add_argument('--capture-name', default='pip_installed_verification')
    parser.add_argument('--inside', action='store_true')
    parser.add_argument('--terminal-driver', action='store_true')
    args = parser.parse_args()
    stage = args.stage.resolve()
    root, venv, prior = installation(stage, args.installation, args.route)
    if Path(args.capture_name).name != args.capture_name or args.capture_name in {'.', '..'}:
        parser.error('--capture-name must be one directory name')
    capture = stage / 'captures' / args.capture_name
    if args.terminal_driver:
        version = installed_version(prior, args.route)
        return terminal_driver(stage, root, capture, args.route, version, venv)
    if capture.exists():
        raise FileExistsError('Choose a new capture name; earlier evidence is retained')
    if args.inside:
        return record(stage, root, venv, prior, capture, args.route)
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'CONDA_PREFIX', 'QT_PLUGIN_PATH',
                'QT_QPA_PLATFORM_PLUGIN_PATH', 'SPACR_NO_SETUP', 'VIRTUAL_ENV', 'CONDA_DEFAULT_ENV'):
        env.pop(key, None)
    private = stage / 'desktop' / args.capture_name
    for key, name in [('XDG_CONFIG_HOME', 'config'), ('XDG_DATA_HOME', 'data'),
                      ('XDG_CACHE_HOME', 'cache'), ('XDG_RUNTIME_DIR', 'runtime')]:
        folder = private / name
        folder.mkdir(parents=True, mode=0o700)
        env[key] = str(folder)
    state = private / 'spacr-state'
    state.mkdir()
    conda_registry = private / 'conda-registry'
    conda_registry.mkdir()
    env.update(SPACR_TUTORIAL_PRIVATE_DESKTOP='1', GIO_USE_VFS='local',
               GSETTINGS_BACKEND='memory', GTK_USE_PORTAL='0', NO_AT_BRIDGE='1',
               XDG_CURRENT_DESKTOP='SPACR_TUTORIAL', QT_QPA_PLATFORM='xcb',
               QT_SCALE_FACTOR='1', QT_AUTO_SCREEN_SCALE_FACTOR='0', QT_FONT_DPI='96',
               SPACR_LANGUAGE='en', PYTHONNOUSERSITE='1',
               PATH=str(venv / 'bin') + os.pathsep + env['PATH'],
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', USE_TF='0',
               SPACR_LOG_DIR=str(private / 'logs'), MPLCONFIGDIR=str(private / 'mpl'))
    if args.route != 'conda':
        env['VIRTUAL_ENV'] = str(venv)
    else:
        env.update(CONDA_PREFIX=str(venv), CONDA_DEFAULT_ENV=str(venv), CONDARC=os.devnull)
    return subprocess.run(['xvfb-run', '-a', '-s', '-screen 0 3840x2160x24',
        'dbus-run-session', '--', 'bwrap', '--die-with-parent', '--bind', '/', '/',
        '--dev-bind', '/dev', '/dev', '--bind', str(state), str(Path.home() / '.spacr'),
        '--bind', str(conda_registry), str(Path.home() / '.conda'),
        '--', sys.executable, str(Path(__file__).resolve()), '--stage', str(stage),
        '--installation', str(root), '--capture-name', args.capture_name, '--route', args.route, '--inside'],
        env=env, cwd=root, timeout=480).returncode


if __name__ == '__main__':
    raise SystemExit(main())
