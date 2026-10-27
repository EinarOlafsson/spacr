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


_PROC = Path('/proc')


def _inspection_process(stage, pid, record):
    """Bind the owned viewer process to the exact unchanged original PNG."""
    display = _inspection_display(stage)
    path = Path(record['path']).resolve()
    process = _PROC / str(pid)
    try:
        argv = [os.fsdecode(arg) for arg in (process / 'cmdline').read_bytes().split(b'\0') if arg]
        env = dict(item.split('=', 1) for item in (process / 'environ').read_text().split('\0') if '=' in item)
        owner = process.stat().st_uid
        bound = (owner == os.getuid() and str(path) in argv
                 and env.get('DISPLAY') == display
                 and Path(env.get('SPACR_CAPTURE_HOST', '')).samefile(Path(stage) / 'gnome-host')
                 and Path(env.get('XDG_RUNTIME_DIR', '')).samefile(Path(stage) / 'gnome-host' / 'run')
                 and env.get('WAYLAND_DISPLAY') == os.environ.get('WAYLAND_DISPLAY'))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError) as error:
        raise RuntimeError(f'Cannot bind actual viewer process: expected_pid={pid} png={str(path)!r}: {error}') from error
    if (not bound or Path(record['path']).is_symlink() or not path.is_relative_to(Path(stage).resolve())
            or digest != record['sha256']):
        raise RuntimeError(f'Viewer process/PNG mismatch: expected_pid={pid} png={str(path)!r} '
                           f'uid={owner} argv={argv!r} display={env.get("DISPLAY")!r} '
                           f'png_sha256={digest} expected_sha256={record["sha256"]}')
    return dict(pid=pid, uid=owner, png=str(path), png_sha256=digest, argv=argv,
                display=display, private_host_runtime_verified=True)


def _inspection_window(stage, pid, record, *, fullscreen=False, window_id=None, require_active=True):
    """Read the actual active Xwayland viewer; never place a window by code."""
    process = _inspection_process(stage, pid, record)
    title = Path(record['path']).name
    def query(*args):
        return subprocess.check_output(args, text=True, timeout=5)
    active = query('xprop', '-root', '_NET_ACTIVE_WINDOW')
    match = re.search(r'0x[0-9a-fA-F]+', active)
    active_id = match.group() if match else '0x0'
    wid = window_id or active_id
    if int(wid, 16) == 0:
        raise RuntimeError(f'No active viewer: expected_pid={pid} title={title!r} active={active!r}')
    props = query('xprop', '-id', wid, '_NET_WM_PID', '_NET_WM_NAME', 'WM_NAME', '_NET_WM_STATE')
    owner = re.search(r'_NET_WM_PID\([^)]*\)\s*=\s*(\d+)', props)
    names = re.findall(r'(?:_NET_WM_NAME|WM_NAME)\([^)]*\)\s*=\s*"([^"\n]*)"', props)
    if not owner or int(owner.group(1)) != pid or not names or not all(name == 'Image Viewer' or title in name for name in names):
        raise RuntimeError(f'Viewer ownership mismatch: expected_pid={pid} title={title!r} '
                           f'wid={wid} active={active_id} properties={props!r}')
    if require_active and int(active_id, 16) != int(wid, 16):
        raise RuntimeError(f'Owned viewer is not active: expected_pid={pid} title={title!r} '
                           f'wid={wid} active={active_id} properties={props!r}')
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
    return dict(window_id=wid, active_window_id=active_id, pid=pid, expected_png=process['png'],
                wm_titles=names, process=process, fullscreen=is_fullscreen,
                rect=[number('Absolute upper-left X'), number('Absolute upper-left Y'),
                      number('Width'), number('Height')], image_viewport_verified=False)


def _find_inspection_window(stage, pid, record):
    _inspection_display(stage)
    title = Path(record['path']).name
    clients = subprocess.check_output(['xprop', '-root', '_NET_CLIENT_LIST_STACKING', '_NET_CLIENT_LIST'],
                                      text=True, timeout=5)
    found, rejected = [], []
    for wid in dict.fromkeys(re.findall(r'0x[0-9a-fA-F]+', clients)):
        if int(wid, 16) == 0:
            continue
        try:
            found.append(_inspection_window(stage, pid, record, window_id=wid, require_active=False))
        except RuntimeError as error:
            rejected.append(str(error))
    if len(found) != 1:
        raise RuntimeError(f'Expected one owned PNG viewer: expected_pid={pid} title={title!r} '
                           f'clients={clients!r} matches={found!r} rejected={rejected!r}')
    return found[0]


def _inspection_pan(dimensions, window, screen):
    """One left-edge inspection of this original four-occupied-slot grid."""
    x, y, width, height = window['rect']
    if (dimensions != [4800, 600] or window.get('fullscreen') is not True
            or window['rect'] != screen or not 2400 <= width < dimensions[0]
            or height < 900):
        raise ValueError('The actual fullscreen window must accommodate four cells and the full grid height')
    thumb_width = width * width / dimensions[0]
    grabpoint = [x + width // 2, y + height - 4]
    start = [grabpoint[0], y + height - 80]
    end = [x + round(thumb_width / 2), start[1]]
    if not x + 20 <= end[0] < start[0] < x + width - 20:
        raise ValueError('The single actual-size grid pan cannot stay within the measured window')
    return dict(grabpoint=grabpoint, start=start, end=end, duration=6.0, kind='horizontal_scrollbar',
                thumb_width=thumb_width, expected_image_displacement=(dimensions[0] - width) / 2,
                actual_size_key='1', image_viewport_verified=False,
                image_area_review='Actual native frames must confirm all four cells and labels')


def _inspect_activation_grid(stage, capture, viewer, record, settle):
    from PySide6.QtWidgets import QApplication
    import capture_ui
    if not capture_ui.live(capture):
        raise RuntimeError('Activation grid inspection is a native capture-host route only')
    before = _find_inspection_window(stage, viewer.pid, record)
    if before['fullscreen']:
        raise RuntimeError('Expected an ordinary viewer before the native F11 action')
    adapter = capture_ui._h().host_of(capture).adapter
    x, y, width, height = before['rect']
    from PySide6.QtCore import QPoint
    screens = QApplication.screens()
    if (len(screens) != 1 or width < 300 or height < 100
            or not screens[0].geometry().contains(QPoint(x + width // 2, y + 16))):
        raise RuntimeError('The measured viewer titlebar is not reachable on the private monitor: ' + repr(before))
    adapter.call('click_at', x + width // 2, y + 16)
    settle(.4)
    focused = _inspection_window(stage, viewer.pid, record, window_id=before['window_id'])
    adapter.call('press', '#keysym:0xffc8'); settle(1.0)
    adapter.call('press', '1'); settle(.6)
    measured = _inspection_window(stage, viewer.pid, record, fullscreen=True, window_id=before['window_id'])
    screens = QApplication.screens()
    if len(screens) != 1:
        raise RuntimeError('Activation grid inspection requires the actual single capture monitor')
    geometry = screens[0].geometry()
    screen = [geometry.x(), geometry.y(), geometry.width(), geometry.height()]
    pan = _inspection_pan(record['dimensions'], measured, screen)
    adapter.call('move_to', *pan['grabpoint'])
    settle(.6)
    try:
        adapter.call('button_down', 'left')
        adapter.call('move_to', *pan['start'], duration=.35)
        adapter.call('move_to', *pan['end'], duration=pan['duration'])
    finally:
        adapter.call('button_up', 'left')
    adapter.note('saved-grid-inspection', **pan)
    settle(.6)
    after = _inspection_window(stage, viewer.pid, record, fullscreen=True, window_id=before['window_id'])
    if after['rect'] != measured['rect'] or after['window_id'] != measured['window_id']:
        raise RuntimeError('The measured actual viewer changed during image inspection')
    return dict(before=before, focused=focused, fullscreen=measured, after=after, pan=pan,
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
