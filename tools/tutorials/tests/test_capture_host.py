"""Pure parts of the headless GNOME capture host: pointer paths, refusals, sprites."""
from pathlib import Path
import math
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import human_input  # noqa: E402
from human_input import Human, min_jerk  # noqa: E402


class FakeHost:
    """Records warps instead of talking to mutter."""

    def __init__(self, width=2560, height=1440):
        self.width, self.height = width, height
        self.pointer = (100.0, 100.0)
        self.warps = []

    def warp(self, x, y):
        self.pointer = (float(x), float(y))
        self.warps.append(self.pointer)


@pytest.mark.parametrize('legacy_manager', [True, False])
def test_private_host_selects_a_video_only_session_manager(tmp_path, monkeypatch, legacy_manager):
    from types import SimpleNamespace
    import gnome_capture_host as host

    seen = []
    is_file = Path.is_file
    monkeypatch.setattr(Path, 'is_file', lambda path: legacy_manager
                        if str(path) == '/usr/bin/pipewire-media-session' else is_file(path))
    monkeypatch.setenv('DBUS_SESSION_BUS_ADDRESS', 'unix:path=/tmp/private-test-bus')
    monkeypatch.setattr(host, 'wait_for', lambda *args: None)
    monkeypatch.setattr(host.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(stdout='Linked with libwireplumber 0.4.17\n'))

    def start(argv, **kwargs):
        if '/usr/bin/gnome-shell' in argv:
            raise RuntimeError('stop before starting a compositor')
        seen.append((argv, kwargs['env']))
        return SimpleNamespace(poll=lambda: 0)

    monkeypatch.setattr(host.subprocess, 'Popen', start)
    args = SimpleNamespace(stage=tmp_path, cursor_theme='Yaru', cursor_size=32,
                           show_top_bar=False, compositor_gpu='all', size='2560x1440',
                           refresh=60, shell_env=[])
    with pytest.raises(RuntimeError, match='stop before starting a compositor'):
        host.inner(args)
    expected = '/usr/bin/pipewire-media-session' if legacy_manager else '/usr/bin/wireplumber'
    assert [argv[0] for argv, _env in seen] == ['/usr/bin/python3', '/usr/bin/pipewire', expected]
    manager_env = seen[-1][1]
    assert manager_env['XDG_CONFIG_HOME'] == str(tmp_path / 'config')
    assert manager_env['XDG_RUNTIME_DIR'] == str(tmp_path / 'run')
    main = (tmp_path / 'config/wireplumber/main.lua.d/89-video-only.lua').read_text()
    bluetooth = (tmp_path / 'config/wireplumber/bluetooth.lua.d/89-video-only.lua').read_text()
    for monitor in ('alsa_monitor', 'v4l2_monitor', 'libcamera_monitor'):
        assert f'{monitor}.enabled = false' in main
    for monitor in ('bluez_monitor', 'bluez_midi_monitor'):
        assert f'{monitor}.enabled = false' in bluetooth


@pytest.mark.parametrize('state', ['free', 'occupied', 'foreign_after_spawn'])
def test_software_native_host_owns_its_display_and_uses_cpu_rendering(tmp_path, monkeypatch, state):
    from types import SimpleNamespace
    import gnome_capture_host as host

    seen = []
    is_file, exists, read_text = Path.is_file, Path.exists, Path.read_text
    monkeypatch.setattr(Path, 'is_file', lambda path: True
                        if str(path) == '/usr/bin/pipewire-media-session' else is_file(path))
    monkeypatch.setattr(Path, 'exists', lambda path: state == 'occupied'
                        if str(path) in ('/tmp/.X11-unix/X96', '/tmp/.X96-lock') else exists(path))
    monkeypatch.setattr(Path, 'read_text', lambda path, *args, **kwargs:
                        ('77777' if state == 'foreign_after_spawn' else '12345')
                        if str(path) == '/tmp/.X96-lock' else read_text(path, *args, **kwargs))
    monkeypatch.setenv('DBUS_SESSION_BUS_ADDRESS', 'unix:path=/tmp/private-test-bus')
    monkeypatch.setattr(host, 'wait_for', lambda *args: None)
    monkeypatch.setattr(host.subprocess, 'run', lambda *args, **kwargs: None)

    def start(argv, **kwargs):
        if '/usr/bin/gnome-shell' in argv:
            assert '--x11' in argv and '--headless' not in argv and '--wayland' not in argv
            assert kwargs['env']['LIBGL_ALWAYS_SOFTWARE'] == '1'
            assert kwargs['env']['GALLIUM_DRIVER'] == 'llvmpipe'
            raise RuntimeError('stop before starting a compositor')
        seen.append((argv, kwargs['env']))
        return SimpleNamespace(pid=12345, poll=lambda: None, terminate=lambda: None,
                               wait=lambda timeout=None: 0)

    monkeypatch.setattr(host.subprocess, 'Popen', start)
    args = SimpleNamespace(stage=tmp_path, cursor_theme='Yaru', cursor_size=32,
                           show_top_bar=False, compositor_gpu='nvidia', size='2560x1440',
                           refresh=60, shell_env=[], software_x11=True, x11_display=96)
    message = {'free': 'stop before starting a compositor', 'occupied': 'already in use',
               'foreign_after_spawn': 'not owned by this host'}[state]
    with pytest.raises(RuntimeError, match=message):
        host.inner(args)
    servers = [(argv, env) for argv, env in seen if argv[0] == '/usr/bin/Xvfb']
    assert len(servers) == (0 if state == 'occupied' else 1)
    if servers:
        argv, env = servers[0]
        assert argv == ['/usr/bin/Xvfb', ':96', '-screen', '0', '2560x1440x24', '-ac', '-nolisten', 'tcp']
        assert env['DISPLAY'] == ':96' and env['XDG_SESSION_TYPE'] == 'x11'
        assert env['LP_NUM_THREADS'] == '2'


def test_min_jerk_starts_and_stops_at_rest():
    assert min_jerk(0.0) == 0.0 and min_jerk(1.0) == 1.0
    eps = 1e-4
    assert (min_jerk(eps) - min_jerk(0)) / eps < 1e-3          # zero start speed
    assert (min_jerk(1) - min_jerk(1 - eps)) / eps < 1e-3      # zero end speed
    samples = [min_jerk(i / 100) for i in range(101)]
    assert all(b >= a for a, b in zip(samples, samples[1:]))  # never goes back


@pytest.mark.parametrize('distance', [30, 300, 900, 2000])
def test_durations_grow_with_distance_and_respect_the_bounds(distance):
    human = Human(FakeHost())
    seconds = human.duration_for(distance)
    assert 0.25 <= seconds <= 3.6
    assert human.duration_for(distance * 2) >= seconds * 0.9


def steps_of(target, start=(100.0, 100.0)):
    host = FakeHost()
    host.pointer = start
    Human(host, hz=60).move_to(*target)
    assert host.warps[-1] == tuple(float(v) for v in target)
    return host, [math.dist(a, b) for a, b in zip(host.warps, host.warps[1:])]


def test_a_move_is_smooth_and_never_jumps():
    host, steps = steps_of((1000, 500))          # ~1080 px
    # Two 60 Hz steps make one 30 fps video frame: at most ~2 % of the width.
    assert 2 * max(steps) <= Human.peak_per_frame * host.width * 1.1
    # Smooth: no step more than twice its neighbours' mean during the glide.
    middle = steps[2:-3]
    assert all(s <= 2 * max(1.0, (a + b) / 2) for a, s, b in zip(middle, middle[1:], middle[2:]))


def test_even_a_full_diagonal_stays_under_five_percent_per_frame():
    host, steps = steps_of((2540, 1420), start=(10.0, 10.0))
    assert 2 * max(steps) <= 0.05 * host.width


def test_host_session_refuses_outside_the_private_host(monkeypatch):
    monkeypatch.delenv('SPACR_CAPTURE_HOST', raising=False)
    with pytest.raises(RuntimeError, match='gnome_capture_host'):
        human_input.HostSession()


def test_keysyms():
    assert human_input._keysym('a') == ord('a')
    assert human_input._keysym('\n') == 0xff0d
    assert human_input._keysym('é') == 0xe9
    assert human_input._keysym('→') == 0x01000000 + ord('→')


def test_xcursor_reader_on_the_installed_theme():
    measure = pytest.importorskip('measure_capture_host_proof')
    path = Path('/usr/share/icons/Yaru/cursors/left_ptr')
    if not path.exists():
        pytest.skip('Yaru cursor theme not installed')
    rgba, hotspot = measure.xcursor_image(path.resolve(), 32)
    assert rgba.shape == (32, 32, 4) and rgba.dtype == np.uint8
    assert 0 <= hotspot[0] < 32 and 0 <= hotspot[1] < 32
    assert (rgba[..., 3] == 255).sum() > 20


def test_a_glide_to_the_menu_bar_stays_under_the_gate_bound():
    # The 2026-10-12 smoke: ~1760 px from mid-screen to the spaCR menu. With
    # the old 1.5 s cap it peaked at ~73 px per 30 fps frame (2.9 %).
    host, steps = steps_of((40, 15), start=(1587.0, 835.0))
    assert 2 * max(steps) <= 0.03 * host.width


@pytest.mark.parametrize('major', [42, 46])
def test_private_chrome_extension_matches_the_installed_shell_module_format(tmp_path, major):
    import json
    import gnome_capture_host as host

    dirs = host.private_dirs(tmp_path, major)
    extension = dirs['data'] / 'gnome-shell/extensions' / host.CHROME_EXTENSION_UUID
    assert json.loads((extension / 'metadata.json').read_text())['shell-version'] == [str(major)]
    script = (extension / 'extension.js').read_text()
    if major >= 45:
        expected = host.CHROME_EXTENSION_JS.replace('const GLib = imports.gi.GLib;',
                                                    "import GLib from 'gi://GLib';")
        expected = expected.replace('const Main = imports.ui.main;',
                                    "import * as Main from 'resource:///org/gnome/shell/ui/main.js';")
        assert script == expected + '\nexport default class { enable() { enable(); } disable() { disable(); } }\n'
    else:
        assert script == host.CHROME_EXTENSION_JS


def test_nested_native_host_uses_real_monitor_mode_and_no_physical_devices(tmp_path, monkeypatch, request):
    from types import SimpleNamespace
    import gnome_capture_host as host

    import socket
    bus = socket.socket(socket.AF_UNIX)
    bus_path = tmp_path / 'session-bus'
    bus.bind(str(bus_path))
    request.addfinalizer(bus.close)
    seen = []
    exists, read_text = Path.exists, Path.read_text
    monkeypatch.setattr(Path, 'is_file', lambda path: True)
    monkeypatch.setattr(Path, 'exists', lambda path: False
                        if str(path) in ('/tmp/.X11-unix/X98', '/tmp/.X98-lock') else exists(path))
    monkeypatch.setattr(Path, 'read_text', lambda path, *args, **kwargs: '12345'
                        if str(path) == '/tmp/.X98-lock' else read_text(path, *args, **kwargs))
    monkeypatch.setenv('DBUS_SESSION_BUS_ADDRESS', 'unix:path=' + str(bus_path))
    monkeypatch.setattr(host, 'wait_for', lambda *args: None)
    monkeypatch.setattr(host.subprocess, 'run', lambda *args, **kwargs: None)

    def start(argv, **kwargs):
        seen.append((argv, kwargs['env']))
        if '/usr/bin/gnome-shell' in argv:
            assert argv[:5] == ['/usr/bin/bwrap', '--die-with-parent', '--ro-bind', '/', '/']
            assert '--dev' in argv and '--dev-bind' not in argv
            position = argv.index('--tmpfs')
            assert argv[position + 1] == '/tmp'
            assert ['--bind', str(tmp_path / 'run/x11'), '/tmp/.X11-unix'] == argv[position + 2:position + 5]
            assert ['--ro-bind', '/tmp/.X11-unix/X98', '/tmp/.X11-unix/X98'] == argv[position + 5:position + 8]
            assert ['--ro-bind', str(bus_path), str(bus_path)] == argv[position + 8:position + 11]
            position = argv.index('--bind')
            assert argv[position + 1:position + 3] == [str(tmp_path.parent), str(tmp_path.parent)]
            assert '--nested' in argv and '--wayland' in argv and '--no-x11' not in argv
            assert '--headless' not in argv and '--replace' not in argv
            values = kwargs['env']
            assert values['MUTTER_DEBUG_DUMMY_MODE_SPECS'] == '2560x1440@60'
            assert values['DISPLAY'] == ':98' and values['CUDA_VISIBLE_DEVICES'] == ''
            assert values['GALLIUM_DRIVER'] == 'llvmpipe'
            assert values['TMPDIR'] == str(tmp_path / 'run')
            assert (tmp_path / 'run').stat().st_mode & 0o777 == 0o700
            assert (tmp_path / 'run/x11').stat().st_mode & 0o1777 == 0o1777
            assert values['__EGL_VENDOR_LIBRARY_FILENAMES'] == '/usr/share/glvnd/egl_vendor.d/50_mesa.json'
            assert values['DBUS_SYSTEM_BUS_ADDRESS'] == 'unix:path=' + str(tmp_path / 'run/logind-proxy')
            raise RuntimeError('stop before starting a compositor')
        return SimpleNamespace(pid=12345, poll=lambda: None, terminate=lambda: None, wait=lambda timeout=None: 0)

    monkeypatch.setattr(host.subprocess, 'Popen', start)
    args = SimpleNamespace(stage=tmp_path, cursor_theme='Yaru', cursor_size=32,
                           show_top_bar=False, compositor_gpu='nvidia', size='2560x1440',
                           refresh=60, shell_env=[], software_nested=True, x11_display=98, capture_stage=tmp_path.parent)
    with pytest.raises(RuntimeError, match='stop before starting a compositor'):
        host.inner(args)
    proxy = next(argv for argv, _env in seen if argv[0] == '/usr/bin/xdg-dbus-proxy')
    assert proxy[:2] == ['/usr/bin/xdg-dbus-proxy', 'unix:path=/run/dbus/system_bus_socket']
    assert proxy[3:6] == ['--filter', '--log', '--see=org.freedesktop.login1']
    calls = {value for value in proxy if value.startswith('--call=')}
    expected_calls = {'--call=org.freedesktop.login1=org.freedesktop.login1.Manager.' + method + '@/org/freedesktop/login1'
                      for method in ('GetSession', 'GetSessionByPID', 'GetUser', 'GetUserByPID', 'GetSeat', 'ListSessions', 'ListUsers', 'ListSeats')}
    expected_calls |= {'--call=org.freedesktop.login1=' + method + '@' + path
                       for method in ('org.freedesktop.DBus.Properties.Get', 'org.freedesktop.DBus.Properties.GetAll', 'org.freedesktop.DBus.Introspectable.Introspect')
                       for path in ('/org/freedesktop/login1', '/org/freedesktop/login1/*')}
    assert calls == expected_calls
    assert len(proxy) == 23
    assert proxy[2] == str(tmp_path / 'run/logind-proxy')


def test_nested_xwayland_connection_ignores_other_displays_and_preserves_owned_auth(tmp_path, monkeypatch):
    import gnome_capture_host as host

    proc = tmp_path / 'proc'
    proc.mkdir()
    owned_auth = tmp_path / 'private-runtime/Xauthority'
    entries = [
        (901, 1, [b'/usr/bin/Xwayland', b':0', b'-auth', b'/personal/Xauthority']),
        (500, 1, [b'/usr/bin/bwrap']),
        (501, 500, [b'/usr/bin/gnome-shell', b'--nested']),
        (502, 501, [b'/usr/bin/Xwayland', b':123', b'-auth', bytes(owned_auth)]),
    ]
    for pid, parent, argv in entries:
        entry = proc / str(pid)
        entry.mkdir()
        (entry / 'stat').write_text(f'{pid} (native process) S {parent} 0 0\n')
        (entry / 'cmdline').write_bytes(b'\0'.join(argv) + b'\0')
    real_path = Path
    monkeypatch.setattr(host, 'Path', lambda value: proc if value == '/proc' else real_path(value))
    assert host.xwayland_connection(500) == {'DISPLAY': ':123', 'XAUTHORITY': str(owned_auth)}
    assert host.xwayland_display(500) == ':123'
    (proc / '502/cmdline').write_bytes(b'/usr/bin/Xwayland\0:123\0')
    assert host.xwayland_connection(500) == {'DISPLAY': ':123'}
    assert host.xwayland_display(500) == ':123'
    (proc / '502/cmdline').write_bytes(b'/usr/bin/not-an-X-server\0')
    assert host.xwayland_connection(500) is None


@pytest.mark.parametrize('warning', [None, 'invalid video buffer received', 'gst_buffer_peek_memory: assertion failed'])
def test_recording_rejects_invalid_buffers_even_after_clean_eos(tmp_path, warning):
    import subprocess
    import time

    output = tmp_path / 'preserved.mp4'
    output.write_bytes(b'preserved failed recording')
    log = output.with_suffix('.gst.log')
    host = human_input.HostSession.__new__(human_input.HostSession)
    host.output = output
    host.node_id = 38
    host.recording_started = time.monotonic()
    host._log = log.open('wb')
    script = ("import signal,sys,time; "
              "signal.signal(signal.SIGINT, lambda *_: sys.exit(0)); "
              f"print({warning!r}, flush=True); "
              "print('PLAYING', flush=True); time.sleep(30)")
    process = host.recorder = subprocess.Popen([sys.executable, '-c', script],
                                               stdout=host._log, stderr=host._log)
    try:
        deadline = time.monotonic() + 5
        while b'PLAYING' not in log.read_bytes():
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(.01)
        if warning is None:
            result = host.stop_recording()
            assert result['video'] == str(output)
            assert result['node_id'] == 38
        else:
            with pytest.raises(RuntimeError, match='produced invalid video buffers') as error:
                host.stop_recording()
            assert warning in str(error.value)
        assert process.returncode == 0
        assert host.recorder is None
        assert host._log.closed
        assert output.read_bytes() == b'preserved failed recording'
        assert 'PLAYING' in log.read_text()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        host._log.close()


@pytest.mark.parametrize('diagnostics', [False, True])
def test_recorder_diagnostics_are_fixed_and_explicit(tmp_path, monkeypatch, diagnostics):
    seen = {}

    def observe(argv, **kwargs):
        seen.update(argv=argv, env=kwargs['env'])
        raise RuntimeError('stop before launching recorder')

    monkeypatch.setenv('GST_DEBUG', '*:9')
    monkeypatch.setenv('GST_DEBUG_FILE', '/foreign/output.log')
    monkeypatch.setattr(human_input.subprocess, 'Popen', observe)
    host = human_input.HostSession.__new__(human_input.HostSession)
    host.node_id = 38
    try:
        with pytest.raises(RuntimeError, match='stop before launching recorder'):
            host.start_recording(tmp_path / 'clip.mp4', diagnostics=diagnostics)
        assert ('-v' in seen['argv']) is diagnostics
        assert 'GST_DEBUG_FILE' not in seen['env']
        if diagnostics:
            assert seen['env']['GST_DEBUG'] == ('2,pipewiresrc:6,pipewirepool:6,videofilter:6,default:6,'
                                                'GST_BUFFER:6,GST_MEMORY:6,bufferpool:6,videorate:6')
            assert seen['env']['GST_DEBUG_NO_COLOR'] == '1'
        else:
            assert not any(key.startswith('GST_') for key in seen['env'])
    finally:
        host._log.close()


def test_live_directory_picker_preserves_paths_that_need_no_menu_action(tmp_path):
    from types import SimpleNamespace
    from PySide6.QtCore import QDir
    from PySide6.QtWidgets import QApplication, QFileDialog
    from capture_model_zoo import _show_hidden_directory_components

    app = QApplication.instance() or QApplication([])
    for already_hidden in (False, True):
        target = tmp_path / ('.models' if already_hidden else 'models')
        target.mkdir()
        dialog = QFileDialog(None, 'Choose a directory', str(target))
        dialog.setFileMode(QFileDialog.Directory)
        if already_hidden:
            dialog.setFilter(dialog.filter() | QDir.Hidden)
        before = (dialog.directory().absolutePath(), dialog.selectedFiles(), dialog.filter())
        try:
            _show_hidden_directory_components(SimpleNamespace(host=object()), dialog, target)
            assert (dialog.directory().absolutePath(), dialog.selectedFiles(), dialog.filter()) == before
            assert QApplication.activePopupWidget() is None
            assert target.is_dir()
        finally:
            dialog.close()
            dialog.deleteLater()
            app.processEvents()
