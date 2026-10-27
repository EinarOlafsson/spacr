"""Run a command inside a private, headless GNOME Shell (Wayland + Xwayland).

The tutorial recording host (maintainer requirement, 2026-10-11): tutorials
must look like a screen capture of the real application with a human at the
mouse. A bare Xvfb has no compositor, so spaCR's rounded, translucent windows
(Preferences) rendered square and opaque, and nothing moved a real pointer.

This host starts, in a session bus of its own (``dbus-run-session``):

* ``pipewire`` + ``pipewire-media-session`` or WirePlumber 0.4, with ALSA,
  Bluetooth and camera monitors disabled so no audio/camera device is opened;
* ``gnome-shell --headless --wayland --virtual-monitor WxH@R`` -- the real
  GNOME compositor drawing to a virtual monitor, with its own Xwayland;

all with a private ``HOME``, ``XDG_*`` directories and ``XDG_RUNTIME_DIR``
under ``--stage``. The maintainer's desktop (``DISPLAY=:0``, his session bus,
his dconf and PipeWire) is never contacted: every daemon gets a clean
environment built here, not the caller's.

The command runs with ``DISPLAY`` (Xwayland), ``WAYLAND_DISPLAY`` and
``DBUS_SESSION_BUS_ADDRESS`` pointing into the private session, so
:mod:`screen_recorder` and :mod:`human_input` reach the private mutter.
On exit every process carrying this host's marker variable is terminated.

    python tools/tutorials/gnome_capture_host.py --stage /path/new-stage -- \
        python tools/tutorials/record_preferences_proof.py --out clip.mp4
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

MARKER = 'SPACR_CAPTURE_HOST'
SYSTEM_PATH = '/usr/local/bin:/usr/bin:/bin'
SHELL_SETTINGS = [
    # No blanking, locking or idle dimming during a recording.
    ('org.gnome.desktop.session', 'idle-delay', 'uint32 0'),
    ('org.gnome.desktop.screensaver', 'lock-enabled', 'false'),
    ('org.gnome.desktop.screensaver', 'idle-activation-enabled', 'false'),
    ('org.gnome.settings-daemon.plugins.power', 'idle-dim', 'false'),
    ('org.gnome.desktop.notifications', 'show-banners', 'false'),
    # No "Welcome to GNOME" tour dialog on a brand-new profile.
    ('org.gnome.shell', 'welcome-dialog-last-shown-version', "'999999'"),
    ('org.gnome.shell', 'disable-user-extensions', 'true'),
    ('org.gnome.desktop.interface', 'enable-animations', 'true'),
    ('org.gnome.desktop.interface', 'color-scheme', "'prefer-dark'"),
]
#: The GNOME top bar is not part of a spaCR tutorial. This private-profile
#: extension removes it (and its strut, so a maximised window takes the
#: whole monitor) and leaves the startup overview. Nothing else changes.
CHROME_EXTENSION_UUID = 'spacr-capture-chrome@spacr.tutorials'
CHROME_EXTENSION_METADATA = {
    'uuid': CHROME_EXTENSION_UUID, 'name': 'spaCR capture chrome',
    'description': 'Hide the top bar for tutorial screen captures (private capture host only).',
    'shell-version': ['42'], 'version': 1,
}
CHROME_EXTENSION_JS = """// spaCR tutorial capture host: no top bar, no startup overview.
const GLib = imports.gi.GLib;
const Main = imports.ui.main;
let startupId = 0;
let timeouts = [];

function hideChrome() {
    const box = Main.layoutManager.panelBox;
    try {
        Main.layoutManager.untrackChrome(box);
    } catch (e) {
        // already untracked
    }
    box.hide();
    if (Main.overview.visible)
        Main.overview.hide();
}

function init() {
}

function enable() {
    hideChrome();
    startupId = Main.layoutManager.connect('startup-complete', hideChrome);
    // The startup overview opens after startup-complete on some versions.
    for (const ms of [500, 1500, 3000])
        timeouts.push(GLib.timeout_add(GLib.PRIORITY_DEFAULT, ms, () => {
            hideChrome();
            return GLib.SOURCE_REMOVE;
        }));
}

function disable() {
    if (startupId)
        Main.layoutManager.disconnect(startupId);
    startupId = 0;
    timeouts.forEach(id => GLib.source_remove(id));
    timeouts = [];
    Main.layoutManager.panelBox.show();
}
"""
MEDIA_SESSION_CONF = """# spaCR capture host: link video streams only; never open audio/camera devices.
context.properties = { }
context.spa-libs = { }
context.modules = [
    { name = libpipewire-module-protocol-native }
    { name = libpipewire-module-client-node }
    { name = libpipewire-module-client-device }
    { name = libpipewire-module-adapter }
    { name = libpipewire-module-metadata }
    { name = libpipewire-module-session-manager }
]
session.modules = {
    default = [
        suspend-node
        policy-node
        metadata
    ]
}
"""


def private_dirs(stage: Path, shell_major: int = 42) -> dict:
    dirs = {name: stage / name for name in ('home', 'config', 'cache', 'data', 'state', 'run', 'logs')}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    dirs['run'].chmod(0o700)
    conf = dirs['config'] / 'pipewire' / 'media-session.d'
    conf.mkdir(parents=True, exist_ok=True)
    (conf / 'media-session.conf').write_text(MEDIA_SESSION_CONF)
    for name, monitors in (('main.lua.d', ('alsa_monitor', 'v4l2_monitor', 'libcamera_monitor')),
                           ('bluetooth.lua.d', ('bluez_monitor', 'bluez_midi_monitor'))):
        wireplumber = dirs['config'] / 'wireplumber' / name
        wireplumber.mkdir(parents=True, exist_ok=True)
        (wireplumber / '89-video-only.lua').write_text(
            ''.join(f'{monitor}.enabled = false\n' for monitor in monitors))
    extension = dirs['data'] / 'gnome-shell' / 'extensions' / CHROME_EXTENSION_UUID
    extension.mkdir(parents=True, exist_ok=True)
    metadata = {**CHROME_EXTENSION_METADATA, 'shell-version': [str(shell_major)]}
    script = CHROME_EXTENSION_JS
    if shell_major >= 45:
        script = script.replace('const GLib = imports.gi.GLib;', "import GLib from 'gi://GLib';")
        script = script.replace('const Main = imports.ui.main;',
                                "import * as Main from 'resource:///org/gnome/shell/ui/main.js';")
        script += '\nexport default class { enable() { enable(); } disable() { disable(); } }\n'
    (extension / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (extension / 'extension.js').write_text(script)
    return dirs


def chrome_settings(show_top_bar: bool) -> list:
    """gsettings that hide (default) or keep the GNOME top bar for a recording."""
    if show_top_bar:
        return [('org.gnome.shell', 'disable-user-extensions', 'true')]
    return [('org.gnome.shell', 'disable-user-extensions', 'false'),
            ('org.gnome.shell', 'disable-extension-version-validation', 'true'),
            ('org.gnome.shell', 'enabled-extensions', f"['{CHROME_EXTENSION_UUID}']"),
            # The pointer reaching the top-left corner must not open the overview.
            ('org.gnome.desktop.interface', 'enable-hot-corners', 'false')]


def daemon_env(stage: Path, dirs: dict) -> dict:
    """A clean environment: nothing from the caller's desktop session."""
    return {
        'PATH': SYSTEM_PATH, 'HOME': str(dirs['home']), 'USER': os.environ.get('USER', 'user'),
        'LOGNAME': os.environ.get('LOGNAME', 'user'), 'LANG': 'en_US.UTF-8',
        'XDG_RUNTIME_DIR': str(dirs['run']), 'XDG_CONFIG_HOME': str(dirs['config']),
        'XDG_CACHE_HOME': str(dirs['cache']), 'XDG_DATA_HOME': str(dirs['data']),
        'XDG_STATE_HOME': str(dirs['state']),
        'XDG_DATA_DIRS': '/usr/share/gnome:/usr/local/share:/usr/share',
        'XDG_CONFIG_DIRS': '/etc/xdg', 'XDG_SESSION_TYPE': 'wayland',
        'XDG_CURRENT_DESKTOP': 'GNOME', 'GSETTINGS_BACKEND': 'dconf',
        'NO_AT_BRIDGE': '1', 'GTK_A11Y': 'none', MARKER: str(stage),
        # Not a logind session id: gnome-shell must not adopt (and set lock
        # hints on) the maintainer's real login session.
        'XDG_SESSION_ID': 'spacr-capture-host',
    }


def marked_pids(stage: str) -> list[int]:
    """Every live process whose environment carries this host's marker."""
    found = []
    needle = f'{MARKER}={stage}'.encode() + b'\0'
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            if needle in (entry / 'environ').read_bytes() + b'\0':
                found.append(int(entry.name))
        except OSError:
            continue
    return found


def terminate_marked(stage: str, grace: float = 8.0) -> list[int]:
    """SIGTERM, then SIGKILL, every marked process; returns survivors (should be [])."""
    pids = marked_pids(stage)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline and marked_pids(stage):
        time.sleep(.2)
    for pid in marked_pids(stage):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(.5)
    return marked_pids(stage)


def wait_for(predicate, timeout: float, what: str, proc=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f'{what}: gnome-shell exited with {proc.returncode}')
        time.sleep(.1)
    raise TimeoutError(what)


def bus_has(name: str, env: dict) -> bool:
    result = subprocess.run(['/usr/bin/gdbus', 'call', '--session', '--dest', 'org.freedesktop.DBus',
                             '--object-path', '/org/freedesktop/DBus', '--method',
                             'org.freedesktop.DBus.NameHasOwner', name],
                            env=env, capture_output=True, text=True)
    return result.returncode == 0 and 'true' in result.stdout


def xwayland_display(shell_pid: int):
    """The ``:N`` our gnome-shell gave its Xwayland (a descendant; bwrap may sit between)."""
    parents, argvs = {}, {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / 'stat').read_text()
            parents[int(entry.name)] = int(stat.rsplit(')', 1)[1].split()[1])
            argvs[int(entry.name)] = (entry / 'cmdline').read_bytes().split(b'\0')
        except (OSError, ValueError, IndexError):
            continue
    for pid, argv in argvs.items():
        if not argv or not argv[0].endswith(b'Xwayland'):
            continue
        ancestor, hops = parents.get(pid), 0
        while ancestor and ancestor != shell_pid and hops < 6:
            ancestor, hops = parents.get(ancestor), hops + 1
        if ancestor != shell_pid:
            continue
        for arg in argv[1:]:
            if arg.startswith(b':') and arg[1:].isdigit():
                return arg.decode()
    return None



def logind_proxy_argv(socket: Path) -> list[str]:
    """Expose only the read-only session metadata needed by the private compositor."""
    name = 'org.freedesktop.login1'
    rules = ['--filter', '--log', '--see=' + name]
    for method in ('GetSession', 'GetSessionByPID', 'GetUser', 'GetUserByPID', 'GetSeat',
                   'ListSessions', 'ListUsers', 'ListSeats'):
        rules.append('--call=' + name + '=org.freedesktop.login1.Manager.' + method + '@/org/freedesktop/login1')
    for method in ('org.freedesktop.DBus.Properties.Get', 'org.freedesktop.DBus.Properties.GetAll',
                   'org.freedesktop.DBus.Introspectable.Introspect'):
        for path in ('/org/freedesktop/login1', '/org/freedesktop/login1/*'):
            rules.append('--call=' + name + '=' + method + '@' + path)
    for method in ('PrepareForSleep', 'SessionRemoved'):
        rules.append('--broadcast=' + name + '=org.freedesktop.login1.Manager.' + method + '@/org/freedesktop/login1')
    rules.append('--broadcast=' + name + '=org.freedesktop.DBus.Properties.PropertiesChanged@/org/freedesktop/login1/*')
    return ['/usr/bin/xdg-dbus-proxy', 'unix:path=/run/dbus/system_bus_socket', str(socket), *rules]


def resize_nested_viewport(display: str, shell_pid: int, size: str, env: dict):
    """Resize only the nested compositor window owned by this launched process tree."""
    owned, pending = {shell_pid}, [shell_pid]
    while pending:
        pid = pending.pop()
        try:
            children = map(int, Path(f'/proc/{pid}/task/{pid}/children').read_text().split())
            for child in children:
                if child not in owned:
                    owned.add(child)
                    pending.append(child)
        except FileNotFoundError:
            continue
    tree = subprocess.check_output(['/usr/bin/xwininfo', '-display', display, '-root', '-tree'], env=env, text=True)
    windows = []
    for identifier in dict.fromkeys(re.findall(r'^\s*(0x[0-9a-fA-F]+)\b', tree, re.MULTILINE)):
        properties = subprocess.check_output(['/usr/bin/xprop', '-display', display, '-id', identifier, '_NET_WM_PID'], env=env, text=True)
        match = re.search(r'=\s*(\d+)', properties)
        if not match or int(match[1]) not in owned:
            continue
        argv = Path(f'/proc/{match[1]}/cmdline').read_bytes().split(b'\0')
        if b'/usr/bin/gnome-shell' not in argv or b'--nested' not in argv:
            continue
        geometry = subprocess.check_output(['/usr/bin/xwininfo', '-display', display, '-id', identifier], env=env, text=True)
        width = int(re.search(r'Width:\s*(\d+)', geometry)[1])
        height = int(re.search(r'Height:\s*(\d+)', geometry)[1])
        windows.append((width * height, identifier))
    if not windows:
        return False
    windows.sort(reverse=True)
    if len(windows) > 1 and windows[0][0] == windows[1][0]:
        raise RuntimeError('The owned nested compositor window is ambiguous')
    identifier = windows[0][1]
    width, height = map(int, size.split('x'))
    x11 = ctypes.CDLL('libX11.so.6')
    x11.XOpenDisplay.argtypes, x11.XOpenDisplay.restype = [ctypes.c_char_p], ctypes.c_void_p
    x11.XResizeWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_uint, ctypes.c_uint]
    x11.XSync.argtypes = [ctypes.c_void_p, ctypes.c_int]
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    connection = x11.XOpenDisplay(display.encode())
    if not connection:
        raise RuntimeError('The owned nested compositor display is unavailable')
    try:
        x11.XResizeWindow(connection, int(identifier, 16), width, height)
        x11.XSync(connection, 0)
    finally:
        x11.XCloseDisplay(connection)
    geometry = subprocess.check_output(['/usr/bin/xwininfo', '-display', display, '-id', identifier], env=env, text=True)
    actual_width = int(re.search(r'Width:\s*(\d+)', geometry)[1])
    actual_height = int(re.search(r'Height:\s*(\d+)', geometry)[1])
    return (actual_width, actual_height) == (width, height)


def drm_nodes(vendor: str) -> list[str]:
    """/dev/dri node names (cardN, renderDN) whose PCI vendor id is ``vendor``."""
    found = []
    for node in sorted(Path('/dev/dri').iterdir()):
        try:
            if (Path('/sys/class/drm') / node.name / 'device/vendor').read_text().strip() == vendor:
                found.append(node.name)
        except OSError:
            continue
    return found


def inner(args) -> int:
    stage = args.stage
    dirs = private_dirs(stage, getattr(args, 'shell_major', 42))
    capture_stage = getattr(args, 'capture_stage', None)
    if capture_stage is not None:
        capture_stage = Path(capture_stage).resolve()
        if capture_stage != stage.parent or not capture_stage.is_dir():
            raise RuntimeError('The writable capture stage must be the existing direct parent of the private host stage')
    env = daemon_env(stage, dirs)
    env['DBUS_SESSION_BUS_ADDRESS'] = os.environ['DBUS_SESSION_BUS_ADDRESS']
    # X clients pick their pointer sprites themselves (a full GNOME session's
    # gsd-xsettings would tell them); without this Qt on Xwayland falls back
    # to the system default theme (DMZ-White, 24 px) while the shell uses ours.
    env.update(XCURSOR_THEME=args.cursor_theme, XCURSOR_SIZE=str(args.cursor_size))
    logs = dirs['logs']
    procs = []

    def start(name, argv, extra=None):
        log = (logs / f'{name}.log').open('wb')
        proc = subprocess.Popen(argv, env={**env, **(extra or {})}, stdout=log, stderr=log,
                                stdin=subprocess.DEVNULL)
        procs.append((name, proc))
        return proc

    rc = 1
    try:
        for schema, key, value in SHELL_SETTINGS + chrome_settings(args.show_top_bar) + [
                ('org.gnome.desktop.interface', 'cursor-theme', f"'{args.cursor_theme}'"),
                ('org.gnome.desktop.interface', 'cursor-size', str(args.cursor_size))]:
            subprocess.run(['/usr/bin/gsettings', 'set', schema, key, value], env=env,
                           check=False, capture_output=True)
        start('systemd-stub', ['/usr/bin/python3', '-I', str(Path(__file__).with_name('capture_host_systemd_stub.py'))])
        wait_for(lambda: bus_has('org.freedesktop.systemd1', env), 10, 'systemd stub on the private bus')
        start('pipewire', ['/usr/bin/pipewire'])
        wait_for(lambda: (dirs['run'] / 'pipewire-0').exists(), 10, 'pipewire socket')
        session_manager = ('/usr/bin/pipewire-media-session'
                           if Path('/usr/bin/pipewire-media-session').is_file() else '/usr/bin/wireplumber')
        if session_manager.endswith('/wireplumber'):
            version = subprocess.run([session_manager, '--version'], env=env,
                                     text=True, capture_output=True, check=True).stdout
            if 'Linked with libwireplumber 0.4.' not in version:
                raise RuntimeError('The video-only capture profile requires WirePlumber 0.4')
        start('media-session', [session_manager])
        software_x11 = getattr(args, 'software_x11', False)
        software_nested = getattr(args, 'software_nested', False)
        software = software_x11 or software_nested
        isolation = []
        shell_extra = dict(kv.split('=', 1) for kv in args.shell_env)
        shell_argv = ['/usr/bin/gnome-shell']
        if software:
            if args.x11_display < 2:
                raise RuntimeError('The software host requires a dedicated non-personal X11 display')
            display = f':{args.x11_display}'
            if (Path(f'/tmp/.X11-unix/X{args.x11_display}').exists()
                    or Path(f'/tmp/.X{args.x11_display}-lock').exists()):
                raise RuntimeError(f'The dedicated X11 display {display} is already in use')
            env.update(DISPLAY=display, XDG_SESSION_TYPE='x11',
                       LIBGL_ALWAYS_SOFTWARE='1', GALLIUM_DRIVER='llvmpipe', LP_NUM_THREADS='2')
            server = start('xvfb', ['/usr/bin/Xvfb', display, '-screen', '0',
                                    f'{args.size}x24', '-ac', '-nolisten', 'tcp'])
            wait_for(lambda: Path(f'/tmp/.X11-unix/X{args.x11_display}').exists(),
                     10, 'private Xvfb socket', server)
            if (server.poll() is not None
                    or Path(f'/tmp/.X{args.x11_display}-lock').read_text().strip() != str(server.pid)):
                raise RuntimeError(f'The dedicated X11 display {display} is not owned by this host')
            if software_nested:
                socket = dirs['run'] / 'logind-proxy'
                start('logind-proxy', logind_proxy_argv(socket))
                wait_for(socket.exists, 10, 'private read-only logind proxy')
                env.update(DBUS_SYSTEM_BUS_ADDRESS='unix:path=' + str(socket), XDG_SESSION_TYPE='wayland',
                           MESA_LOADER_DRIVER_OVERRIDE='llvmpipe',
                           __EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/50_mesa.json',
                           MUTTER_DEBUG_DUMMY_MODE_SPECS=f'{args.size}@{args.refresh}',
                           QT_OPENGL='software', QT_QUICK_BACKEND='software', CUDA_VISIBLE_DEVICES='',
                           GDK_BACKEND='x11', GIO_USE_VFS='local')
                isolation = ['/usr/bin/bwrap', '--die-with-parent', '--ro-bind', '/', '/',
                             '--bind', str(capture_stage or stage), str(capture_stage or stage),
                             '--dev', '/dev', '--proc', '/proc', '--']
                shell_argv = [*isolation, '/usr/bin/gnome-shell', '--nested', '--wayland', '--no-x11',
                              '--wayland-display', 'wayland-capture', '--sm-disable']
            else:
                shell_argv += ['--x11', '--replace', '--sm-disable']
            shell_extra.update(LIBGL_ALWAYS_SOFTWARE='1', GALLIUM_DRIVER='llvmpipe', LP_NUM_THREADS='2')
        elif args.compositor_gpu != 'all':
            # Hide the other GPU's DRM nodes from gnome-shell (and so from
            # its Xwayland) in a private mount namespace; nothing else changes.
            keep = drm_nodes({'amd': '0x1002', 'nvidia': '0x10de'}[args.compositor_gpu])
            if not keep:
                raise RuntimeError(f'No {args.compositor_gpu} DRM device found')
            binds = []
            for node in keep:
                binds += ['--dev-bind', f'/dev/dri/{node}', f'/dev/dri/{node}']
            shell_argv = ['/usr/bin/bwrap', '--dev-bind', '/', '/', '--tmpfs', '/dev/dri', *binds,
                          '--', '/usr/bin/gnome-shell']
        if not software:
            shell_argv += ['--headless', '--wayland', '--wayland-display', 'wayland-capture',
                           '--virtual-monitor', f'{args.size}@{args.refresh}', '--sm-disable']
        shell = start('gnome-shell', shell_argv, shell_extra)
        wait_for(lambda: bus_has('org.gnome.Mutter.ScreenCast', env), 60, 'mutter ScreenCast on the private bus', shell)
        wait_for(lambda: bus_has('org.gnome.Mutter.RemoteDesktop', env), 30, 'mutter RemoteDesktop', shell)
        if not software:
            display = wait_for(lambda: xwayland_display(shell.pid), 60, 'Xwayland display', shell)
        wait_for(lambda: Path(f'/tmp/.X11-unix/X{display[1:]}').exists(), 30, 'Xwayland socket', shell)
        if software_nested:
            wait_for(lambda: resize_nested_viewport(display, shell.pid, args.size, env),
                     15, 'owned nested compositor viewport', shell)
        # What gsd-xsettings does in a full session: X clients (Qt on xcb
        # reads Xcursor.* from RESOURCE_MANAGER, not XCURSOR_THEME) get the
        # same pointer theme and size as the shell.
        subprocess.run(['/usr/bin/xrdb', '-merge'], env={**env, 'DISPLAY': display},
                       input=f'Xcursor.theme: {args.cursor_theme}\nXcursor.size: {args.cursor_size}\n'
                             'Xft.dpi: 96\n', text=True, check=True)
        time.sleep(args.settle)
        host = {'DISPLAY': display,
                'DBUS_SESSION_BUS_ADDRESS': env['DBUS_SESSION_BUS_ADDRESS'],
                'XDG_RUNTIME_DIR': str(dirs['run']), 'XDG_SESSION_TYPE': 'x11' if software_x11 else 'wayland',
                'SPACR_CAPTURE_HOST_SIZE': args.size, 'SPACR_CAPTURE_HOST_REFRESH': str(args.refresh),
                'XCURSOR_THEME': args.cursor_theme, 'XCURSOR_SIZE': str(args.cursor_size),
                'SPACR_CAPTURE_HOST_TOP_BAR': 'shown' if args.show_top_bar else 'hidden',
                MARKER: str(stage)}
        if software_nested:
            host.update({key: env[key] for key in ('LIBGL_ALWAYS_SOFTWARE', 'GALLIUM_DRIVER', 'LP_NUM_THREADS',
                        'MESA_LOADER_DRIVER_OVERRIDE', '__EGL_VENDOR_LIBRARY_FILENAMES',
                        'QT_OPENGL', 'QT_QUICK_BACKEND', 'CUDA_VISIBLE_DEVICES', 'DBUS_SYSTEM_BUS_ADDRESS')})
            host.update(SPACR_CAPTURE_HOST_BACKEND='software-nested-wayland',
                        WAYLAND_DISPLAY='wayland-capture', XDG_SESSION_TYPE='wayland', GDK_BACKEND='wayland',
                        QT_QPA_PLATFORM='wayland')
        elif software_x11:
            host.update(LIBGL_ALWAYS_SOFTWARE='1', GALLIUM_DRIVER='llvmpipe', LP_NUM_THREADS='2',
                        SPACR_CAPTURE_HOST_BACKEND='software-x11')
        else:
            host['SPACR_CAPTURE_HOST_BACKEND'] = 'wayland'
            host['WAYLAND_DISPLAY'] = 'wayland-capture'
        (stage / 'host.json').write_text(json.dumps({**host, 'gnome_shell_pid': shell.pid}, indent=2) + '\n')
        command_env = dict(os.environ)
        for key in ('DBUS_SESSION_BUS_ADDRESS', 'DISPLAY', 'WAYLAND_DISPLAY', 'XDG_RUNTIME_DIR',
                    'XDG_SESSION_TYPE', 'XDG_CURRENT_DESKTOP', 'XDG_SESSION_DESKTOP',
                    'GIO_LAUNCHED_DESKTOP_FILE', 'GIO_LAUNCHED_DESKTOP_FILE_PID',
                    'SESSION_MANAGER', 'GNOME_TERMINAL_SCREEN', 'QT_IM_MODULE'):
            command_env.pop(key, None)
        command_env.update(host)
        command_env.update(args.outer_env)
        rc = subprocess.run([*isolation, *args.command], env=command_env).returncode
    finally:
        for name, proc in reversed(procs):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', type=Path, required=True, help='New private directory for this host')
    parser.add_argument('--size', default='2560x1440',
                        help='Virtual monitor. 3840x2160 works but mutter 42 cannot hand NVIDIA frames '
                             'to the screencast as DMA-bufs here, so every 4K frame is a synchronous '
                             'read-back: even a trivial full-screen animation kept only 85-90 %% '
                             'distinct frames (3200x1800: 77 %%), against 99 %% at 2560x1440.')
    parser.add_argument('--refresh', type=int, default=60)
    parser.add_argument('--cursor-theme', default='Yaru')
    parser.add_argument('--cursor-size', type=int, default=32)
    parser.add_argument('--settle', type=float, default=3.0, help='Seconds to let the shell finish starting')
    parser.add_argument('--compositor-gpu', choices=('all', 'amd', 'nvidia'), default='nvidia',
                        help='DRM devices gnome-shell and its Xwayland may see (default: the NVIDIA '
                             'card only; with the AMD card alone mutter cannot create its '
                             'high-priority context (EACCES) and stalls).')
    parser.add_argument('--capture-stage', type=Path,
                        help='Writable direct parent of the host stage for a nested software recorder')
    parser.add_argument('--software-nested', action='store_true',
                        help='Use an owned Xvfb with nested software Wayland GNOME and no physical GPU devices')
    parser.add_argument('--software-x11', action='store_true',
                        help='Use a dedicated Xvfb and software-rendered GNOME, retaining native ScreenCast and pointer APIs')
    parser.add_argument('--x11-display', type=int, default=96,
                        help='Unused dedicated X11 display number for --software-x11; personal displays 0 and 1 are refused')
    parser.add_argument('--xwayland-glamor', action='store_true',
                        help='Let Xwayland render with GLAMOR on the GPU. Off by default: Xwayland '
                             '22.1.1 + NVIDIA dies with "wl_drm: invalid format" as soon as an ARGB '
                             '(translucent) X window maps -- every spaCR glass dialog. Software '
                             'Xwayland hands mutter shm buffers; at 2560x1440 that keeps 99 %% of '
                             'frames distinct. spaCR\'s own GPU paths (CUDA, the headless-EGL field '
                             'renderer) do not go through Xwayland and are unaffected.')
    parser.add_argument('--show-top-bar', action='store_true',
                        help='Keep the GNOME top bar (default: a private extension hides it and its '
                             'strut, so a maximised window fills the monitor; hot corners are off)')
    parser.add_argument('--shell-env', action='append', default=[], metavar='KEY=VALUE',
                        help='Extra environment for gnome-shell and its Xwayland')
    parser.add_argument('--shell-major', type=int, default=42, help=argparse.SUPPRESS)
    parser.add_argument('--inner', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--outer-env', default='{}', help=argparse.SUPPRESS)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ['--']:
        args.command = args.command[1:]
    if args.software_x11 and args.software_nested:
        parser.error('Choose only one software host backend')
    if not args.command:
        parser.error('give a command to run inside the host')
    args.stage = args.stage.resolve()
    if args.capture_stage is not None:
        args.capture_stage = args.capture_stage.resolve()
        if not args.software_nested or args.capture_stage != args.stage.parent or not args.capture_stage.is_dir():
            parser.error('--capture-stage requires the existing direct parent of a --software-nested host stage')
    if args.inner:
        args.outer_env = json.loads(args.outer_env)
        return inner(args)
    args.stage.mkdir(parents=True, exist_ok=False)
    shell_version = subprocess.check_output(['/usr/bin/gnome-shell', '--version'], text=True)
    args.shell_major = int(shell_version.split()[2].split('.')[0])
    dirs = private_dirs(args.stage, args.shell_major)
    env = daemon_env(args.stage, dirs)
    # The command itself keeps the caller's interpreter, conda env and GPU
    # access; only display/bus variables are redirected (in ``inner``).
    outer = {'PATH': os.environ.get('PATH', SYSTEM_PATH), 'HOME': str(dirs['home'])}
    # Recorder inputs named by the caller (SPACR_TUTORIAL_* fixtures, e.g. the
    # synthetic plaque source) reach the command; nothing else of the desktop.
    outer.update({k: v for k, v in os.environ.items() if k.startswith('SPACR_TUTORIAL_')})
    rc = 1
    # `timeout`/kill sends SIGTERM: unwind through `finally` so cleanup runs.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        rc = subprocess.run([
            '/usr/bin/dbus-run-session', '--', sys.executable, str(Path(__file__).resolve()),
            '--inner', '--shell-major', str(args.shell_major),
            '--stage', str(args.stage), '--size', args.size, '--refresh', str(args.refresh),
            '--cursor-theme', args.cursor_theme, '--cursor-size', str(args.cursor_size),
            '--settle', str(args.settle), '--outer-env', json.dumps(outer),
            '--compositor-gpu', args.compositor_gpu,
            *(['--software-x11'] if args.software_x11 else []),
            *(['--software-nested'] if args.software_nested else []),
            *(['--capture-stage', str(args.capture_stage)] if args.capture_stage is not None else []),
            *(['--x11-display', str(args.x11_display)] if args.software_x11 or args.software_nested else []),
            *(['--show-top-bar'] if args.show_top_bar else []),
            *([] if args.xwayland_glamor else ['--shell-env=XWAYLAND_NO_GLAMOR=1']),
            *[f'--shell-env={kv}' for kv in args.shell_env], '--', *args.command,
        ], env=env).returncode
    finally:
        survivors = terminate_marked(str(args.stage))
        report = {'exit_code': rc, 'survivors_after_cleanup': survivors}
        (args.stage / 'cleanup.json').write_text(json.dumps(report, indent=2) + '\n')
        if survivors:
            print(f'capture host: processes survived cleanup: {survivors}', file=sys.stderr)
            rc = rc or 3
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
