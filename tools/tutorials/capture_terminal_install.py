#!/usr/bin/env python3
"""Record a real installation walkthrough in a private 4K terminal and GUI.

A real interactive bash runs in a pseudo-terminal shown by gnome-terminal on
a private Xvfb display. bubblewrap gives that shell a private home at the
neutral path /home/user, so no personal path reaches the screen. The recorder
types each command into the shell (bash echoes it like keyboard input),
answers the programs' own prompts, waits for the prompt to return and saves
the whole display. Every line on screen is the programs' own output: nothing
is replayed, edited or composed, and the full transcript is kept.

The installed application starts with its own defaults in the private home.
Its first-launch setup screen and tour are dismissed with their own Escape
action. CUDA devices are hidden from the whole session (no GPU work).

Routes, each in a fresh throwaway folder under <stage>/installation_runs:
  pip        standalone CPython 3.12 at /usr/local, a .venv and PyPI
  conda      a fresh Miniforge and the conda-forge package
  installer  the checksum-verified public Linux online installer

The installation folder is deleted afterwards unless --keep-installation.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib.request import Request, urlopen

from stage_lesson import DEFAULT_STAGE, REPO, read, write

TOOL = Path(__file__).resolve()
TERMINAL_TITLE = 'Terminal'
NEUTRAL_HOME = '/home/user'
VERSION_COMMAND = 'python -c "import spacr; print(spacr.__version__); print(spacr.__file__)"'
PROCEED = (r'Proceed \(\[y\]/n\)\?', 'y')
PBS_RELEASE = '20260924'
PBS_NAME = f'cpython-3.12.14+{PBS_RELEASE}-x86_64-unknown-linux-gnu-install_only.tar.gz'
MINIFORGE = 'https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh'
ANSI = re.compile(r'\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07|\x1b[()][A-Z0-9]|\r')


def installer_name(version):
    return f'spaCR-{version}-Linux-x86_64-Online.run'


def steps(route, version, phase):
    """The on-camera script for one phase.

    'install' runs with CUDA devices hidden and needs no GPU turn. 'gpu' shows
    the card to the programs (spacr-doctor's probe, the first launch, and the
    desktop installer's own CUDA check) and runs inside tools/gpu_turn.sh.
    A ('quiet', command) step runs before the screen is cleared, to reopen an
    environment in a new terminal.
    """
    if route == 'pip':
        if phase == 'install':
            return [
                ('run', 'python3 --version'), ('shot', '01_python_version'),
                ('clear',), ('run', 'python3 -m venv .venv'), ('run', 'ls .venv'),
                ('shot', '07_create_environment'),
                ('clear',), ('run', 'source .venv/bin/activate'), ('run', 'which python'),
                ('shot', '08_activate_environment'),
                ('clear',), ('run', 'python -m pip --version'), ('shot', '02_pip_environment'),
                ('clear',), ('run', 'python -m pip install --upgrade pip', 900),
                ('run', 'python -m pip install spacr', 5400), ('shot', '09_install_package'),
                ('clear',), ('run', VERSION_COMMAND, 300), ('shot', '03_installed_versions'),
                ('clear',), ('run', 'python -m pip check', 300), ('shot', '04_dependency_check'),
                ('clear',), ('run', 'python -m pip install --upgrade spacr', 1800),
                ('run', 'spacr --version', 300), ('run', 'python -m pip check', 300),
                ('shot', '11_update_intentionally'),
            ]
        return [
            ('quiet', 'source .venv/bin/activate'),
            ('clear',), ('run', 'spacr-doctor', 900), ('shot', '05_doctor'),
            ('clear',), ('gui', 'spacr', '10_launch_commands', '06_installed_home'),
        ]
    if route == 'conda':
        if phase == 'install':
            return [
                ('run', 'conda --version'), ('run', 'conda env list'), ('shot', '02_conda_environment'),
                ('clear',), ('run', 'conda create -n spacr-conda -c conda-forge python=3.12', 1800, [PROCEED]),
                ('shot', '07_create_conda'),
                ('clear',), ('run', 'conda activate spacr-conda'),
                ('run', 'python -c "import sys; print(sys.prefix)"'), ('shot', '08_activate_conda'),
                ('clear',), ('run', 'conda install -c conda-forge spacr', 5400, [PROCEED]),
                ('run', 'conda list spacr', 300), ('shot', '09_install_conda'),
                ('clear',), ('run', VERSION_COMMAND, 300), ('shot', '03_installed_versions'),
                ('clear',), ('run', 'conda update -c conda-forge spacr', 1800, [PROCEED]),
                ('run', 'conda list spacr', 300), ('shot', '10_current_release_choice'),
            ]
        return [
            ('quiet', 'conda activate spacr-conda'),
            ('clear',), ('run', 'spacr-doctor', 900), ('shot', '05_doctor'),
            ('clear',), ('gui', 'spacr', None, '06_installed_home'),
        ]
    if phase == 'install':
        raise ValueError('The desktop installer checks CUDA itself; record it in the gpu phase')
    name = installer_name(version)
    return [
        ('run', 'ls -l'), ('run', 'chmod +x ' + name), ('run', f'./{name} --help'),
        ('shot', '11_linux_commands'),
        ('clear',), ('run', f'./{name} --skip-system-deps', 5400,
                     [(r'report previews\? \[y/N\] ', 'n'),
                      (r'issue-report action\? \[y/N\] ', 'n'),
                      (r'on first launch\? \[y/N\] ', 'n')], '07_privacy_keep_off'),
        ('gui', None, None, '06_installed_home'),
        ('clear',), ('run', 'cat ~/.local/share/spacr/install-profile.json'),
        ('shot', '02_installer_backend'),
        ('clear',), ('run', 'spacr --version', 300), ('shot', '03_installed_versions'),
        ('clear',), ('run', '~/.local/share/spacr/venv/bin/spacr-doctor', 900), ('shot', '05_doctor'),
        ('clear',), ('run', 'ls ~/.local/share/spacr'),
        ('run', 'tail -n 6 ~/.local/share/spacr/install.log'), ('shot', '12_logs_and_versions'),
    ]


# --------------------------------------------------------------------------
# Inside the terminal: a real bash in a pseudo-terminal, typed into.

def typist(ctl):
    import fcntl
    import pty
    import select
    import signal
    import termios
    import tty

    pid, fd = pty.fork()
    if pid == 0:
        os.execv('/bin/bash', ['bash', '--noprofile', '--rcfile', str(ctl / 'bashrc'), '-i'])

    def sync(*_):
        try:
            fcntl.ioctl(fd, termios.TIOCSWINSZ, fcntl.ioctl(0, termios.TIOCGWINSZ, b'\0' * 8))
        except OSError:
            pass

    signal.signal(signal.SIGWINCH, sync)
    sync()
    try:
        tty.setraw(0)
    except termios.error:
        pass
    log = (ctl / 'transcript.log').open('ab', buffering=0)
    inbox, seen, pending, last = ctl / 'inbox', set(), b'', 0.0
    while True:
        try:
            ready, _, _ = select.select([fd], [], [], 0.01)
        except InterruptedError:
            continue
        if ready:
            try:
                data = os.read(fd, 65536)
            except OSError:
                break
            if not data:
                break
            view = memoryview(data)
            while view:
                view = view[os.write(1, view):]
            log.write(data)
        now = time.monotonic()
        if pending and now - last >= 0.018:
            os.write(fd, pending[:1])
            pending, last = pending[1:], now
        elif not pending:
            for item in sorted(inbox.glob('*.json')):
                if item.name not in seen:
                    seen.add(item.name)
                    pending = json.loads(item.read_text())['text'].encode()
                    break
    os.waitpid(pid, 0)
    return 0


# --------------------------------------------------------------------------
# The recorder, on the private display.

class Display:
    def __init__(self):
        self.x = ctypes.CDLL('libX11.so.6')
        self.x.XOpenDisplay.argtypes = [ctypes.c_char_p]
        self.x.XOpenDisplay.restype = ctypes.c_void_p
        self.x.XFlush.argtypes = [ctypes.c_void_p]
        for name in ('XMapRaised',):
            getattr(self.x, name).argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        self.x.XMoveResizeWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int,
                                             ctypes.c_int, ctypes.c_uint, ctypes.c_uint]
        self.x.XSetInputFocus.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        self.x.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        self.x.XKeysymToKeycode.restype = ctypes.c_ubyte
        self.display = self.x.XOpenDisplay(None)
        if not self.display:
            raise RuntimeError('The private X display could not be opened')
        self.xtest = ctypes.CDLL('libXtst.so.6')
        self.xtest.XTestFakeKeyEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]
        self.xtest.XTestFakeMotionEvent.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                                    ctypes.c_int, ctypes.c_ulong]

    def windows(self):
        tree = subprocess.check_output(['xwininfo', '-root', '-tree'], text=True)
        return [(int(wid, 16), name, int(w), int(h)) for wid, name, w, h in
                re.findall(r'(0x[0-9a-f]+) "([^"]*)".*? (\d+)x(\d+)[+-]\d+[+-]\d+', tree)]

    def place(self, wid, x, y, width, height):
        self.x.XMoveResizeWindow(self.display, wid, x, y, width, height)
        self.x.XMapRaised(self.display, wid)
        self.x.XFlush(self.display)

    def key(self, wid, keysym):
        self.x.XSetInputFocus(self.display, wid, 2, 0)
        self.x.XFlush(self.display)
        time.sleep(0.3)
        code = self.x.XKeysymToKeycode(self.display, keysym)
        self.xtest.XTestFakeKeyEvent(self.display, code, 1, 0)
        self.xtest.XTestFakeKeyEvent(self.display, code, 0, 0)
        self.x.XFlush(self.display)

    def park_pointer(self):
        self.xtest.XTestFakeMotionEvent(self.display, -1, 3839, 2159, 0)
        self.x.XFlush(self.display)


class Recorder:
    def __init__(self, ctl, capture, route):
        self.ctl, self.capture, self.route = ctl, capture, route
        self.sent = 0
        existing = capture / 'frames.json'
        self.frames = read(existing) if existing.exists() else {}
        self.commands = read(capture / 'commands.json') if (capture / 'commands.json').exists() else []
        self.evidence = (read(capture / 'evidence_frames.json')
                         if (capture / 'evidence_frames.json').exists() else [])
        self.display = Display()

    def prompts(self):
        path = self.ctl / 'prompts'
        return path.read_text().split() if path.exists() else []

    def transcript(self, offset=0):
        with (self.ctl / 'transcript.log').open('rb') as handle:
            handle.seek(offset)
            return handle.read()

    def size(self):
        return (self.ctl / 'transcript.log').stat().st_size

    def send(self, text):
        self.sent += 1
        temporary = self.ctl / 'inbox' / f'{self.sent:05d}.part'
        temporary.write_text(json.dumps({'text': text}))
        temporary.rename(temporary.with_suffix('.json'))
        # Wait until every character has been typed.
        time.sleep(0.2 + 0.02 * len(text))

    def wait_prompts(self, count, timeout):
        deadline = time.monotonic() + timeout
        while len(self.prompts()) < count:
            if time.monotonic() > deadline:
                raise TimeoutError(f'The shell prompt did not return within {timeout} s')
            time.sleep(0.5)

    def clear(self):
        self.send('\x0c')
        time.sleep(0.8)

    def run(self, command, timeout=180, answers=(), shot_after=None):
        before, offset, start = len(self.prompts()), self.size(), time.monotonic()
        self.send(command + '\r')
        answers, cursor = list(answers), 0
        deadline = start + timeout
        while len(self.prompts()) <= before:
            if time.monotonic() > deadline:
                raise TimeoutError(f'{command!r} exceeded {timeout} s')
            if answers:
                text = self.transcript(offset).decode('utf-8', 'replace')
                pattern, reply = answers[0]
                match = re.compile(pattern).search(text, cursor)
                if match:
                    time.sleep(1.2)
                    self.send(reply + '\r')
                    cursor = match.end()
                    answers.pop(0)
                    if not answers and shot_after:
                        time.sleep(2.5)
                        self.shot(shot_after, settle=0)
                    continue
            time.sleep(0.5)
        status = int(self.prompts()[-1])
        output = ANSI.sub('', self.transcript(offset).decode('utf-8', 'replace'))
        self.commands.append(dict(command=command, exit_status=status, output=output,
                                  seconds=round(time.monotonic() - start, 1),
                                  unanswered_prompts=[a[0] for a in answers]))
        write(self.capture / 'commands.json', self.commands)
        print(f'{command} -> {status} ({time.monotonic() - start:.0f} s)', flush=True)
        if status != 0:
            self.shot('99_failure', settle=0.5)
            raise RuntimeError(f'{command!r} exited {status}')
        return output

    def shot(self, name, settle=1.5, evidence=False):
        time.sleep(settle)
        self.display.park_pointer()
        path = self.capture / (name + '.png')
        subprocess.run(['import', '-window', 'root', '-depth', '8', str(path)], check=True, timeout=60)
        size = subprocess.check_output(['identify', '-format', '%wx%h', str(path)], text=True)
        if size != '3840x2160':
            raise RuntimeError(f'{name} is {size}, not native 4K')
        record = dict(image=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), buttons=[])
        if evidence:
            self.evidence.append(dict(record, name=name))
        else:
            self.frames[name] = record
            write(self.capture / 'frames.json', self.frames)
        print('Captured ' + name, flush=True)

    def app_windows(self):
        found = []
        for w in self.display.windows():
            # Qt appends the application name: "Set spaCR up — spaCR".
            if w[1] == TERMINAL_TITLE or w[2] < 300 or w[3] < 200 or 'spaCR' not in w[1]:
                continue
            state = subprocess.run(['xwininfo', '-id', hex(w[0])], text=True, capture_output=True)
            if 'Map State: IsViewable' in state.stdout:
                found.append(w)
        return found

    def gui(self, command, setup_frame, home_frame):
        before = len(self.prompts())
        if command:
            self.send(command + '\r')
        deadline = time.monotonic() + 300
        dismissed, main = [], None
        while main is None:
            if time.monotonic() > deadline:
                (self.capture / 'window_tree_at_timeout.txt').write_text(
                    subprocess.check_output(['xwininfo', '-root', '-tree'], text=True))
                self.shot('99_gui_timeout', settle=0.5, evidence=True)
                raise TimeoutError('The installed application did not open a window')
            if command and len(self.prompts()) > before:
                self.shot('99_gui_exit', settle=0.5, evidence=True)
                raise RuntimeError('The application exited before its window opened')
            found = self.app_windows()
            setup = [w for w in found if w[1].startswith('Set spaCR up') and w[0] not in dismissed]
            if setup:
                time.sleep(6)
                if setup_frame and not dismissed:
                    self.shot(setup_frame, settle=0)
                self.shot(f'gui_first_launch_{len(dismissed) + 1}', settle=0, evidence=True)
                self.display.key(setup[0][0], 0xff1b)
                dismissed.append(setup[0][0])
                time.sleep(3)
                continue
            mains = [w for w in found if not w[1].startswith('Set spaCR up') and w[2] >= 640]
            if mains and not setup:
                main = max(mains, key=lambda w: w[2] * w[3])
            time.sleep(0.5)
        wid = main[0]
        self.display.place(wid, 0, 0, 3840, 2160)
        time.sleep(15)
        self.shot('gui_home_before_escape', settle=0, evidence=True)
        self.display.key(wid, 0xff1b)
        time.sleep(4)
        self.display.place(wid, 0, 0, 3840, 2160)
        self.shot(home_frame, settle=2)
        from capture_pip_installation import close_window
        close_window(self.display, wid)
        deadline = time.monotonic() + 90
        while any(w[0] == wid for w in self.app_windows()):
            if time.monotonic() > deadline:
                self.shot('99_close_timeout', settle=0.5, evidence=True)
                raise TimeoutError('The application did not close from its window')
            time.sleep(0.5)
        if command:
            self.wait_prompts(before + 1, 90)
            status = int(self.prompts()[-1])
            self.commands.append(dict(command=command, exit_status=status, gui=True,
                                      setup_screens_dismissed=len(dismissed)))
            if status != 0:
                raise RuntimeError(f'{command!r} exited {status} after its window closed')
        else:
            self.commands.append(dict(command='(launched by installer)', gui=True,
                                      setup_screens_dismissed=len(dismissed)))
        write(self.capture / 'commands.json', self.commands)


def inside(args):
    run_dir, capture = args.installation.resolve(), args.capture.resolve()
    ctl = run_dir / ('ctl-' + args.phase)
    recorder = Recorder(ctl, capture, args.route)
    terminal = subprocess.Popen(['gnome-terminal', '--wait', '--hide-menubar', '--title=' + TERMINAL_TITLE,
                                 '--zoom=1.8', '--', *sandbox(run_dir, args.route, args.cwd, args.phase),
                                 '/usr/bin/python3', str(TOOL), '--typist', str(ctl)])
    try:
        deadline = time.monotonic() + 60
        while True:
            match = [w for w in recorder.display.windows() if w[1] == TERMINAL_TITLE and w[2] > 200]
            if match:
                break
            if time.monotonic() > deadline or terminal.poll() not in (None, 0):
                raise RuntimeError('The terminal did not open on the private display')
            time.sleep(0.3)
        recorder.display.place(match[0][0], 200, 100, 3440, 1920)
        recorder.wait_prompts(1, 60)
        time.sleep(2)
        recorder.clear()
        for step in steps(args.route, args.version, args.phase):
            kind = step[0]
            if kind in ('run', 'quiet'):
                recorder.run(step[1], *step[2:])
            elif kind == 'shot':
                recorder.shot(step[1])
            elif kind == 'clear':
                recorder.clear()
            elif kind == 'gui':
                recorder.gui(*step[1:])
        recorder.send('exit\r')
        terminal.wait(timeout=60)
    finally:
        write(capture / 'evidence_frames.json', recorder.evidence)
        if terminal.poll() is None:
            terminal.terminate()
    return 0


# --------------------------------------------------------------------------
# Outside: the throwaway installation folder and the private display.

def sandbox(run_dir, route, cwd, phase='install'):
    args = ['bwrap', '--die-with-parent', '--bind', '/', '/', '--dev-bind', '/dev', '/dev',
            '--tmpfs', '/home', '--bind', str(run_dir / 'home'), NEUTRAL_HOME,
            '--chdir', cwd, '--clearenv']
    if route == 'pip':
        args += ['--bind', str(run_dir / 'python'), '/usr/local']
    environment = dict(
        HOME=NEUTRAL_HOME, USER='user', LOGNAME='user', SHELL='/bin/bash', TERM='xterm-256color',
        LANG='en_US.UTF-8', LC_ALL='en_US.UTF-8',
        PATH=f'{NEUTRAL_HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin',
        TMPDIR=f'{NEUTRAL_HOME}/.cache/tmp', XDG_RUNTIME_DIR=str(run_dir / 'xdg-runtime'),
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4', UV_CONCURRENT_DOWNLOADS='4',
        UV_CONCURRENT_INSTALLS='4', UV_CONCURRENT_BUILDS='1', MAX_JOBS='2',
        QT_QPA_PLATFORM='xcb', QT_SCALE_FACTOR='2', NO_AT_BRIDGE='1', GSETTINGS_BACKEND='memory',
        PIP_NO_CACHE_DIR='1', SPACR_CTL=str(run_dir / ('ctl-' + phase)))
    if phase != 'gpu':
        environment['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('DISPLAY', 'XAUTHORITY', 'DBUS_SESSION_BUS_ADDRESS'):
        if os.environ.get(key):
            environment[key] = os.environ[key]
    for key, value in environment.items():
        args += ['--setenv', key, value]
    return args + ['--']


def fetch(url, limit=400 * 1024 ** 2):
    with urlopen(Request(url, headers={'User-Agent': 'spaCR-tutorial-capture'}), timeout=300) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError('Unexpectedly large download: ' + url)
    return data


def prepare(route, run_dir, version, receipt):
    home = run_dir / 'home'
    (home / '.cache/tmp').mkdir(parents=True)
    (run_dir / 'xdg-runtime').mkdir(mode=0o700)
    if route == 'pip':
        base = f'https://github.com/astral-sh/python-build-standalone/releases/download/{PBS_RELEASE}/'
        payload = fetch(base + PBS_NAME.replace('+', '%2B'))
        sums = fetch(base + 'SHA256SUMS').decode()
        expected = [line.split()[0] for line in sums.splitlines() if line.strip().endswith(PBS_NAME)]
        actual = hashlib.sha256(payload).hexdigest()
        if expected != [actual]:
            raise ValueError('The standalone Python download does not match its published checksum')
        archive = run_dir / PBS_NAME
        archive.write_bytes(payload)
        with tarfile.open(archive) as bundle:
            bundle.extractall(run_dir, filter='data')
        archive.unlink()
        receipt['python'] = dict(source=base + PBS_NAME, sha256=actual,
                                 note='Standalone CPython 3.12 visible as /usr/local/bin/python3')
        (home / 'spacr-project').mkdir()
        return f'{NEUTRAL_HOME}/spacr-project'
    if route == 'conda':
        payload = fetch(MINIFORGE)
        expected = fetch(MINIFORGE + '.sha256').decode().split()[0]
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise ValueError('The Miniforge download does not match its published checksum')
        (run_dir / 'Miniforge3.sh').write_bytes(payload)
        receipt['miniforge'] = dict(source=MINIFORGE, sha256=actual)
        for command in (['bash', str(run_dir / 'Miniforge3.sh'), '-b', '-p', f'{NEUTRAL_HOME}/miniforge3'],
                        [f'{NEUTRAL_HOME}/miniforge3/bin/conda', 'init', 'bash']):
            subprocess.run([*sandbox(run_dir, route, NEUTRAL_HOME), *command], check=True,
                           stdout=subprocess.DEVNULL, timeout=1800)
        (run_dir / 'Miniforge3.sh').unlink()
        return NEUTRAL_HOME
    from check_release_installer import checksum_for, release_asset
    tag = 'v' + version
    release = json.loads(fetch(f'https://api.github.com/repos/EinarOlafsson/spacr/releases/tags/{tag}'))
    name = installer_name(version)
    manifest = fetch(release_asset(release['assets'], 'SHA256SUMS.txt', tag)['browser_download_url'])
    payload = fetch(release_asset(release['assets'], name, tag)['browser_download_url'])
    actual = hashlib.sha256(payload).hexdigest()
    if actual != checksum_for(manifest.decode(), name):
        raise ValueError('The installer does not match its release checksum')
    (home / 'Downloads').mkdir()
    target = home / 'Downloads' / name
    target.write_bytes(payload)
    target.chmod(0o644)
    receipt['installer'] = dict(tag=tag, name=name, sha256=actual,
                                release_url=release['html_url'])
    return f'{NEUTRAL_HOME}/Downloads'


def identity(route, run_dir, cwd):
    python = {'pip': f'{NEUTRAL_HOME}/spacr-project/.venv/bin/python',
              'conda': f'{NEUTRAL_HOME}/miniforge3/envs/spacr-conda/bin/python',
              'installer': f'{NEUTRAL_HOME}/.local/share/spacr/venv/bin/python'}[route]
    program = ('import json,sys,spacr; print(json.dumps(dict(version=spacr.__version__, '
               'package=spacr.__file__, prefix=sys.prefix, python=sys.version.split()[0])))')
    output = subprocess.check_output([*sandbox(run_dir, route, cwd), python, '-I', '-c', program],
                                     text=True, timeout=300, env=dict(os.environ, QT_QPA_PLATFORM='offscreen'))
    return json.loads(output.strip().splitlines()[-1])


def remove(path):
    def writable(function, name, _info):
        os.chmod(name, 0o700)
        function(name)
    for folder, directories, _files in os.walk(path):
        for directory in directories:
            try:
                os.chmod(os.path.join(folder, directory), 0o700)
            except OSError:
                pass
    shutil.rmtree(path, onerror=writable)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', type=Path, default=DEFAULT_STAGE)
    parser.add_argument('--route', choices=('pip', 'conda', 'installer'))
    parser.add_argument('--version', default='1.5.1.0', help='Release expected from the route')
    parser.add_argument('--capture-name')
    parser.add_argument('--phase', choices=('install', 'gpu'), default='install',
                        help='install: CUDA hidden, no GPU turn. gpu: inside tools/gpu_turn.sh')
    parser.add_argument('--installation', type=Path,
                        help='Continue in the throwaway folder an install phase kept')
    parser.add_argument('--keep-installation', action='store_true')
    parser.add_argument('--typist', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--inside', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--capture', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--cwd', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.typist:
        return typist(args.typist)
    if args.inside:
        return inside(args)
    if not args.route or not args.capture_name:
        parser.error('--route and --capture-name are required')
    if Path(args.capture_name).name != args.capture_name or args.capture_name in {'.', '..'}:
        parser.error('--capture-name must be one directory name')
    stage = args.stage.resolve()
    capture = stage / 'captures' / args.capture_name
    runs = stage / 'installation_runs'
    if args.installation:
        run_dir = args.installation.resolve()
        if not run_dir.is_relative_to(runs) or not (run_dir / 'home').is_dir():
            raise ValueError('Continue only in a kept throwaway folder under installation_runs')
        provenance = read(capture / 'provenance.json')
        if provenance.get('installation_folder') != str(run_dir):
            raise ValueError('That folder belongs to a different capture')
        cwd = provenance['working_directory']
    else:
        if capture.exists():
            raise FileExistsError('Choose a new capture name; earlier evidence is retained')
        if shutil.disk_usage(stage).free < 40 * 1024 ** 3:
            raise ValueError('Keep at least 40 GiB free for a throwaway installation')
        runs.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix=f'{args.route}-walkthrough-', dir=runs))
        capture.mkdir(parents=True)
        provenance = dict(
            completed_capture=False, module=f'{args.route}_install_walkthrough',
            commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            route=args.route, expected_version=args.version, actual_system_terminal=True,
            interactive_shell_typed=True, installation_recorded_live=True,
            application_source_modified=False, application_preferences_modified=False,
            first_launch_dialogs='dismissed with their own Escape action', visible_home=NEUTRAL_HOME,
            phases={}, installation_folder=str(run_dir), installation_deleted=False)
        (run_dir / 'home').mkdir()
        cwd = prepare(args.route, run_dir, args.version, provenance)
        provenance['working_directory'] = cwd
    write(capture / 'provenance.json', provenance)
    print('Throwaway installation folder: ' + str(run_dir), flush=True)
    finished = False
    try:
        ctl = run_dir / ('ctl-' + args.phase)
        if ctl.exists():
            raise FileExistsError('This phase was already recorded in that folder')
        (ctl / 'inbox').mkdir(parents=True)
        (ctl / 'bashrc').write_text(
            "PS1='\\w \\$ '\nHISTFILE=\n[ -f ~/.bashrc ] && . ~/.bashrc\n"
            "alias ls='ls --color=auto'\n"
            "PROMPT_COMMAND='printf \"%s\\n\" \"$?\" >> \"$SPACR_CTL/prompts\"'\n")
        env = dict(os.environ)
        for key, name in (('XDG_CONFIG_HOME', 'config'), ('XDG_DATA_HOME', 'data'),
                          ('XDG_CACHE_HOME', 'cache'), ('XDG_RUNTIME_DIR', 'runtime')):
            folder = run_dir / ('desktop-' + args.phase) / name
            folder.mkdir(parents=True, mode=0o700)
            env[key] = str(folder)
        env.update(GSETTINGS_BACKEND='memory', NO_AT_BRIDGE='1', GTK_USE_PORTAL='0',
                   XDG_CURRENT_DESKTOP='SPACR_TUTORIAL')
        for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'CONDA_PREFIX'):
            env.pop(key, None)
        command = ['xvfb-run', '-a', '-s', '-screen 0 3840x2160x24', 'dbus-run-session', '--',
                   sys.executable, str(TOOL), '--inside', '--route', args.route, '--phase', args.phase,
                   '--version', args.version, '--installation', str(run_dir),
                   '--capture', str(capture), '--cwd', cwd]
        if args.phase == 'gpu':
            command = [str(REPO / 'tools/gpu_turn.sh'), f'358-install-{args.route}-capture', *command]
        started = time.time()
        result = subprocess.run(command, env=env, timeout=8 * 3600)
        shutil.copyfile(ctl / 'transcript.log', capture / f'transcript.{args.phase}.log')
        provenance['phases'][args.phase] = dict(returncode=result.returncode, cuda_visible=args.phase == 'gpu',
                                                gpu_turn=args.phase == 'gpu', started=started,
                                                seconds=round(time.time() - started))
        write(capture / 'provenance.json', provenance)
        if result.returncode:
            raise RuntimeError('The recording did not complete')
        provenance['installed_identity'] = identity(args.route, run_dir, cwd)
        if args.route != 'conda' and provenance['installed_identity']['version'] != args.version:
            raise RuntimeError('The installed release is not the expected version')
        provenance['frames'] = sorted(read(capture / 'frames.json'))
        needed = {'pip': {'install', 'gpu'}, 'conda': {'install', 'gpu'}, 'installer': {'gpu'}}[args.route]
        provenance['completed_capture'] = needed <= {k for k, v in provenance['phases'].items()
                                                      if v['returncode'] == 0}
        write(capture / 'provenance.json', provenance)
        finished = True
        print(f'Phase {args.phase} complete: {capture}', flush=True)
    finally:
        if not args.keep_installation:
            remove(run_dir)
            provenance['installation_deleted'] = True
            write(capture / 'provenance.json', provenance)
        elif not finished:
            print('Kept for inspection: ' + str(run_dir), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
