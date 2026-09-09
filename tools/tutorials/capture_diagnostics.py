"""Record the genuine Diagnostics fold and its saved figures on a private X11 desktop.

The application button opens its normal file manager. Existing diagnostic PNGs
are viewed in the system image viewer, not injected into spaCR as invented GUI
panels. Run under a private dbus-run-session as well as Xvfb.
"""
from __future__ import annotations

import csv
import argparse
import ctypes
import hashlib
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


class PrivateDesktop:
    def __init__(self, stage):
        if os.environ.get('SPACR_TUTORIAL_PRIVATE_DESKTOP') != '1':
            raise RuntimeError('Use a dedicated dbus-run-session and Xvfb for external viewers')
        display = os.environ.get('DISPLAY', '')
        if not display.startswith(':') or display in {':0', ':0.0', ':1', ':1.0'}:
            raise RuntimeError('Never automate a personal desktop session')
        for key in ('XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_CACHE_HOME', 'XDG_RUNTIME_DIR'):
            if not os.environ.get(key) or not Path(os.environ[key]).resolve().is_relative_to(stage):
                raise RuntimeError(f'{key} must be isolated before launching the private desktop bus')
        self.x = ctypes.CDLL('libX11.so.6')
        self.x.XOpenDisplay.argtypes = [ctypes.c_char_p]
        self.x.XOpenDisplay.restype = ctypes.c_void_p
        self.x.XCloseDisplay.argtypes = [ctypes.c_void_p]
        self.x.XFlush.argtypes = [ctypes.c_void_p]
        for name in ('XMapRaised', 'XLowerWindow'):
            getattr(self.x, name).argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        self.x.XMoveResizeWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong,
                                          ctypes.c_int, ctypes.c_int,
                                          ctypes.c_uint, ctypes.c_uint]
        self.display = self.x.XOpenDisplay(None)
        if not self.display:
            raise RuntimeError('The recording X display could not be opened')

    def find(self, title, settle):
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            tree = subprocess.check_output(['xwininfo', '-root', '-tree'], text=True)
            matches = [(int(wid, 16), name) for wid, name in
                       re.findall(r'(0x[0-9a-f]+) "([^"]+)"', tree)
                       if title in name]
            if matches:
                return matches[-1]
            settle(0.2)
        raise RuntimeError(f'The system viewer did not show {title!r} on the private display')

    def show(self, wid):
        self.x.XMoveResizeWindow(self.display, wid, 200, 100, 3440, 1920)
        self.x.XMapRaised(self.display, wid)
        self.x.XFlush(self.display)

    def lower(self, wid):
        self.x.XLowerWindow(self.display, wid)
        self.x.XFlush(self.display)

    def close(self):
        self.x.XCloseDisplay(self.display)


def record_diagnostics(window, screen, stage, project, captures, capture, settle, write_json):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from spacr.qt.screens.regression import DiagnosticsOpener
    from spacr.qt.widgets.fold_strip import FoldButton

    project = project.resolve()
    if not project.is_relative_to(stage / 'regression_runs') or not project.is_dir():
        raise RuntimeError('Only a completed project in the private tutorial workspace is permitted')
    if not screen._settings_model.set_value_for_key('src', str(project)):
        raise RuntimeError('Could not select the real project output root')
    settle()
    opener = DiagnosticsOpener(screen)
    folder = opener._folder()
    if folder is None or not Path(folder).resolve().is_relative_to(project):
        raise RuntimeError('The actual Diagnostics route does not resolve to this example')
    folder = Path(folder)
    names = ['design_identifiability.png', 'design_diagnostics.png', 'inference_diagnostics.png']
    if not all((folder / name).is_file() for name in names):
        raise RuntimeError('The real run did not write every panel this lesson demonstrates')
    with (folder / 'diagnostic_summary.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    if any(row['metric'].endswith('_error') for row in rows):
        raise RuntimeError('The diagnostics report contains a failed panel')
    write_json(captures / 'diagnostics.json', {
        'project': str(project), 'folder': str(folder), 'verdict': opener.verdict(),
        'summary': rows, 'residuals_available': (folder / 'residual_diagnostics.png').exists(),
        'panels': [{'name': name, 'sha256': hashlib.sha256((folder / name).read_bytes()).hexdigest()}
                   for name in names], 'panels_computed_by_recorder': False})
    buttons = [button for button in screen.findChildren(FoldButton)
               if button.isVisible() and button.app_key == 'regression_diagnostics']
    if len(buttons) != 1 or not buttons[0].isEnabled():
        raise RuntimeError('Regression must expose its actual Diagnostics button')
    desktop = PrivateDesktop(stage)
    try:
        QTest.mouseMove(buttons[0])
        settle(1.5)
        capture('02_diagnostics_button')
        # The actual callback opens the existing folder, without computing a fit.
        QTest.mouseClick(buttons[0], Qt.LeftButton)
        wid, title = desktop.find('diagnostics', settle)
        desktop.show(wid)
        settle(1.5)
        capture('03_diagnostics_folder', desktop=True)
        desktop.lower(wid)
        viewers = []
        for number, name in enumerate(names, 4):
            viewer = subprocess.Popen(['eog', '--new-instance', str(folder / name)],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                figure_wid, figure_title = desktop.find(name, settle)
                desktop.show(figure_wid)
                settle(1.5)
                capture(f'{number:02d}_{Path(name).stem}', desktop=True)
                viewers.append({'image': name, 'window_title': figure_title})
            finally:
                if viewer.poll() is None:
                    viewer.terminate()
                    try:
                        viewer.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        viewer.kill()
                        viewer.wait(timeout=5)
        desktop.x.XMapRaised(desktop.display, int(window.winId()))
        desktop.x.XFlush(desktop.display)
        settle()
        capture('07_back_to_regression')
        write_json(captures / 'desktop_acceptance.json', {
            'accepted': True, 'folder_opened_by_actual_diagnostics_button': True,
            'file_manager_title': title, 'system_viewer_panels': viewers,
            'application_source_modified': False, 'private_display': os.environ['DISPLAY'],
            'new_regression_fitted_by_button': False})
    finally:
        desktop.close()


def main():
    """Launch external viewers with private settings, caches, runtime and bus."""
    from stage_lesson import DEFAULT_STAGE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--stage', type=Path, default=DEFAULT_STAGE)
    args = parser.parse_args()
    stage = args.stage.resolve()
    env = dict(os.environ)
    desktop_root = stage / 'desktop/diagnostics'
    for key, name in [('XDG_CONFIG_HOME', 'config'), ('XDG_DATA_HOME', 'data'),
                      ('XDG_CACHE_HOME', 'cache')]:
        path = desktop_root / name
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    runtime_root = desktop_root / 'runtime'
    runtime_root.mkdir(parents=True, exist_ok=True)
    env['XDG_RUNTIME_DIR'] = tempfile.mkdtemp(prefix='capture-', dir=runtime_root)
    env.update(SPACR_TUTORIAL_PRIVATE_DESKTOP='1', GIO_USE_VFS='local',
               GVFS_DISABLE_FUSE='1', GSETTINGS_BACKEND='memory',
               GTK_USE_PORTAL='0', QT_QPA_PLATFORMTHEME='',
               XDG_CURRENT_DESKTOP='SPACR_TUTORIAL', NO_AT_BRIDGE='1',
               GDK_SCALE='2', GDK_DPI_SCALE='1')
    command = ['xvfb-run', '-a', '-s', '-screen 0 3840x2160x24',
               'dbus-run-session', '--', sys.executable,
               str(Path(__file__).with_name('capture_refresh.py')),
               '--module', 'regression_diagnostics', '--stage', str(stage),
               '--diagnostics-from', str(args.project.resolve()), '--platform', 'xcb']
    return subprocess.run(command, env=env, timeout=240, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
