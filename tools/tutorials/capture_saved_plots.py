"""Show unchanged, locally saved PNG plots in the real system image viewer.

This never inserts an image into spaCR's Figures card or changes application
code. A private X11 display and D-Bus session are mandatory.
"""
import hashlib
import os
import re
from pathlib import Path
import subprocess


def check_saved_file_run(run):
    """Accept only a completed worker and the specifically diagnosed src error.

The stricter in-app figure acceptance remains separate and unchanged.
Independent CSV checks and actual viewer capture are additionally required.
"""
    outcome = run['outcome']
    if not outcome['finished'] or not outcome['ok'] or outcome['errors']:
        raise ValueError('The actual worker did not finish successfully')
    errors = [line.strip() for line in run['settings_errors']]
    if errors not in ([], ['[settings] ERROR [src]: src is missing from the settings.']):
        raise ValueError('Unexpected settings errors cannot be accepted by this workaround')


def _inspection_display(stage):
    display = os.environ.get('DISPLAY', '')
    marker = os.environ.get('SPACR_CAPTURE_HOST', '')
    runtime_name = os.environ.get('XDG_RUNTIME_DIR', '')
    wayland_name = os.environ.get('WAYLAND_DISPLAY', '')
    try:
        runtime = Path(runtime_name)
        socket = Path(wayland_name)
        if not socket.is_absolute():
            socket = runtime / socket
        owned = (bool(marker and runtime_name and wayland_name)
                 and Path(marker).samefile(Path(stage) / 'gnome-host')
                 and runtime.samefile(Path(stage) / 'gnome-host' / 'run')
                 and socket.resolve().parent.samefile(runtime)
                 and socket.is_socket() and socket.stat().st_uid == os.getuid())
    except OSError:
        owned = False
    if (not owned or not re.fullmatch(r':\d+(?:\.0)?', display)
            or display in {':0', ':0.0'}):
        raise RuntimeError("Grid inspection requires this stage's owned private Xwayland/Wayland runtime")
    return display


def _inspection_window(stage, pid, title, *, fullscreen=False):
    """Read the actual active Xwayland viewer; never place a window by code."""
    _inspection_display(stage)
    def query(*args):
        return subprocess.check_output(args, text=True, timeout=5)
    active = query('xprop', '-root', '_NET_ACTIVE_WINDOW')
    match = re.search(r'0x[0-9a-fA-F]+', active)
    if not match or int(match.group(), 16) == 0:
        raise RuntimeError('The actual viewer is not the active X window')
    wid = match.group()
    props = query('xprop', '-id', wid, '_NET_WM_PID', '_NET_WM_NAME', 'WM_NAME', '_NET_WM_STATE')
    owner = re.search(r'_NET_WM_PID\([^)]*\)\s*=\s*(\d+)', props)
    names = re.findall(r'(?:_NET_WM_NAME|WM_NAME)\([^)]*\)\s*=\s*"([^"\n]*)"', props)
    if not owner or int(owner.group(1)) != pid or not any(title in name for name in names):
        raise RuntimeError('The active window does not belong to this actual saved PNG viewer')
    stats = query('xwininfo', '-id', wid, '-stats')
    def number(label):
        found = re.search(r'(?m)^\s*' + re.escape(label) + r':\s*(-?\d+)\s*$', stats)
        if not found:
            raise RuntimeError('The actual viewer geometry is unavailable: ' + label)
        return int(found.group(1))
    if 'Map State: IsViewable' not in stats:
        raise RuntimeError('The actual viewer is not mapped')
    is_fullscreen = '_NET_WM_STATE_FULLSCREEN' in props
    if fullscreen and not is_fullscreen:
        raise RuntimeError('The actual viewer did not enter native fullscreen')
    return dict(window_id=wid, pid=pid, title=title, fullscreen=is_fullscreen,
                rect=[number('Absolute upper-left X'), number('Absolute upper-left Y'),
                      number('Width'), number('Height')], image_viewport_verified=False)


def _inspection_pan(dimensions, window, screen):
    """One left-edge inspection of this original four-occupied-slot grid."""
    x, y, width, height = window['rect']
    if (dimensions != [4800, 600] or window.get('fullscreen') is not True
            or window['rect'] != screen or not 2400 <= width < dimensions[0]
            or height < 900):
        raise ValueError('The actual fullscreen window must accommodate four cells and the full grid height')
    start = [x + round(width * .25), y + height // 2]
    end = [start[0] + (dimensions[0] - width) // 2, start[1]]
    if not x + 20 <= start[0] < end[0] < x + width - 20:
        raise ValueError('The single actual-size grid pan cannot stay within the measured window')
    return dict(start=start, end=end, duration=6.0,
                actual_size_key='1', image_viewport_verified=False,
                image_area_review='Actual native frames must confirm all four cells and labels')


def _inspect_activation_grid(stage, capture, viewer, record, settle):
    from PySide6.QtWidgets import QApplication
    import capture_ui
    if not capture_ui.live(capture):
        raise RuntimeError('Activation grid inspection is a native capture-host route only')
    before = _inspection_window(stage, viewer.pid, Path(record['path']).name)
    if before['fullscreen']:
        raise RuntimeError('Expected an ordinary viewer before the native F11 action')
    adapter = capture_ui._h().host_of(capture).adapter
    adapter.call('press', 'F11'); settle(1.0)
    adapter.call('press', '1'); settle(.6)
    measured = _inspection_window(stage, viewer.pid, Path(record['path']).name, fullscreen=True)
    screens = QApplication.screens()
    if len(screens) != 1:
        raise RuntimeError('Activation grid inspection requires the actual single capture monitor')
    geometry = screens[0].geometry()
    screen = [geometry.x(), geometry.y(), geometry.width(), geometry.height()]
    pan = _inspection_pan(record['dimensions'], measured, screen)
    adapter.call('drag', *pan['start'], *pan['end'], duration=pan['duration'])
    adapter.note('saved-grid-inspection', **pan)
    settle(.6)
    after = _inspection_window(stage, viewer.pid, Path(record['path']).name, fullscreen=True)
    if after['rect'] != measured['rect'] or after['window_id'] != measured['window_id']:
        raise RuntimeError('The measured actual viewer changed during image inspection')
    return dict(before=before, fullscreen=measured, after=after, pan=pan,
                png_modified=False, repeated_motion=False, native_frame_review_required=True)


def show_saved_plots(app, window, stage, capture, settle, paths, *,
                     inspect_activation_grid=False, scene_prefix=""):
    from PIL import Image
    from capture_diagnostics import PrivateDesktop

    if not 1 <= len(paths) <= 8:
        raise ValueError('Use one to eight already verified saved plots')
    checked = []
    for path in map(Path, paths):
        if path.is_symlink() or not path.resolve().is_relative_to(stage.resolve()) or path.suffix != '.png':
            raise ValueError('Only original PNG files within private tutorial staging are allowed')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as image:
            size = list(image.size)
            if image.format != 'PNG' or min(size) < 100:
                raise ValueError('Expected a nonempty PNG figure')
            image.verify()
        checked.append(dict(path=str(path), sha256=digest, dimensions=size))
    desktop = PrivateDesktop(stage)
    try:
        for index, record in enumerate(checked, 9):
            path = Path(record['path'])
            options = ['eog', '--new-instance', '--disable-gallery']
            viewer_env = None
            if inspect_activation_grid:
                # Only this opt-in viewer uses the host's queryable Xwayland display.
                options.append('--display=' + _inspection_display(stage))
                viewer_env = {**os.environ, 'GDK_BACKEND': 'x11'}
            viewer = subprocess.Popen([*options, str(path)], env=viewer_env,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                wid, title = desktop.find(path.name, settle)
                desktop.show(wid); settle(2)
                name = scene_prefix + f'{index:02d}_saved_{path.stem}'
                if inspect_activation_grid:
                    from capture_ui_drv_a import Rec
                    with Rec(capture, name, settle, desktop=True):
                        record['native_inspection'] = _inspect_activation_grid(stage, capture, viewer, record, settle)
                else:
                    capture(name, desktop=True)
                record.update(window_title=title, viewer='eog', actual_desktop_capture=True)
            finally:
                if viewer.poll() is None:
                    viewer.terminate()
                    try:
                        viewer.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        viewer.kill(); viewer.wait(timeout=5)
            if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
                raise ValueError('Viewing changed the saved plot')
        desktop.x.XMapRaised(desktop.display, int(window.winId()))
        desktop.x.XFlush(desktop.display); settle()
        capture(scene_prefix + '11_back_to_actual_gui')
    finally:
        desktop.close()
    return dict(accepted=True, saved_plots=checked, saved_files_unchanged=True,
                application_figures_fixed=False, scope='External system viewer, not spaCR Figures')
