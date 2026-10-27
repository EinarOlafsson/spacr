"""Real pointer/keyboard input and GNOME screen recording for tutorial captures.

Use inside :mod:`gnome_capture_host` (a private headless GNOME Shell). One
:class:`HostSession` owns, on one D-Bus connection, a mutter RemoteDesktop
session bound to a ScreenCast of the virtual monitor:

* the ScreenCast stream is recorded by GStreamer (``pipewiresrc`` -> x264 ->
  MP4, constant 30 fps) with the **real cursor embedded by mutter**, and real
  window effects (rounded corners, translucency, shadows) composited by
  GNOME Shell;
* pointer and keyboard events go through mutter's virtual input devices --
  the same path a physical mouse takes -- so Qt receives genuine enter /
  hover / press / wheel events and the visible cursor is the one that moves.

Pointer motion is human-like: a minimum-jerk speed profile (smooth start and
stop), a slight arc, ~60 Hz steps and a duration that grows with distance
(Fitts-like, 250-900 ms). Nothing ever teleports while recording.

Recorder pattern (the Qt main thread just runs the app; a driver thread is
"the human")::

    host = HostSession()                    # needs the host's private bus
    host.start_recording(Path('clip.mp4'))
    human = Human(host)
    human.move_to(*gui(lambda: center_of(widget)))
    human.click()
    ...
    receipt = host.stop_recording()
    host.close()

``gui(fn)`` (see :class:`GuiCaller`) runs ``fn`` on the Qt main thread and
returns its result, so geometry is read where Qt owns it, even while a
modal dialog or a menu runs its own nested event loop.
"""
from __future__ import annotations

import json
import math
import os
import random
import signal
import subprocess
import threading
import time
from pathlib import Path

from jeepney import DBusAddress, MatchRule, Properties, message_bus, new_method_call
from jeepney.io.blocking import Proxy, open_dbus_connection

RD = DBusAddress('/org/gnome/Mutter/RemoteDesktop', bus_name='org.gnome.Mutter.RemoteDesktop',
                 interface='org.gnome.Mutter.RemoteDesktop')
SC = DBusAddress('/org/gnome/Mutter/ScreenCast', bus_name='org.gnome.Mutter.ScreenCast',
                 interface='org.gnome.Mutter.ScreenCast')
BTN = {'left': 0x110, 'right': 0x111, 'middle': 0x112}   # linux/input-event-codes.h
CURSOR_EMBEDDED = 1
KEYSYM = {'\n': 0xff0d, '\r': 0xff0d, '\t': 0xff09, '\b': 0xff08, 'Escape': 0xff1b,
          'Return': 0xff0d, 'Tab': 0xff09, 'BackSpace': 0xff08, 'Delete': 0xffff,
          'Left': 0xff51, 'Up': 0xff52, 'Right': 0xff53, 'Down': 0xff54,
          'Home': 0xff50, 'End': 0xff57, 'Page_Up': 0xff55, 'Page_Down': 0xff56,
          'Shift_L': 0xffe1, 'Control_L': 0xffe3, 'Alt_L': 0xffe9, 'Super_L': 0xffeb}
SYSTEM_GST = '/usr/bin/gst-launch-1.0'


def _keysym(char: str) -> int:
    if char in KEYSYM:
        return KEYSYM[char]
    if char.startswith('#keysym:0x'):
        return int(char.removeprefix('#keysym:'), 16)
    code = ord(char)
    return code if 0x20 <= code <= 0xff else 0x01000000 + code


class HostSession:
    """RemoteDesktop + ScreenCast of the whole virtual monitor, owned by one connection."""

    def __init__(self, cursor_mode: int = CURSOR_EMBEDDED):
        if not os.environ.get('SPACR_CAPTURE_HOST'):
            raise RuntimeError('HostSession must run inside tools/tutorials/gnome_capture_host.py '
                               '(refusing to drive a real desktop session)')
        self._lock = threading.RLock()
        self.conn = open_dbus_connection(bus='SESSION')
        self.width, self.height = (int(v) for v in os.environ['SPACR_CAPTURE_HOST_SIZE'].split('x'))
        self.rd_path = self._call(RD, 'CreateSession')[0]
        self.rd = DBusAddress(self.rd_path, bus_name=RD.bus_name,
                              interface='org.gnome.Mutter.RemoteDesktop.Session')
        session_id = self.conn.send_and_get_reply(
            Properties(self.rd).get('SessionId')).body[0][1]
        self.sc_path = self._call(SC, 'CreateSession', 'a{sv}',
                                  ({'remote-desktop-session-id': ('s', session_id)},))[0]
        self.sc = DBusAddress(self.sc_path, bus_name=SC.bus_name,
                              interface='org.gnome.Mutter.ScreenCast.Session')
        # '' = the primary (only) monitor: the headless virtual monitor.
        self.stream_path = self._call(self.sc, 'RecordMonitor', 'sa{sv}',
                                      ('', {'cursor-mode': ('u', cursor_mode)}))[0]
        rule = MatchRule(type='signal', interface='org.gnome.Mutter.ScreenCast.Stream',
                         member='PipeWireStreamAdded', path=self.stream_path)
        Proxy(message_bus, self.conn).AddMatch(rule)
        with self.conn.filter(rule) as queue:
            self._call(self.rd, 'Start')
            self.node_id = self.conn.recv_until_filtered(queue, timeout=15).body[0]
        self.recorder = None
        self.pointer = (self.width / 2, self.height / 2)
        self.warp(*self.pointer)

    def _call(self, address, method, signature=None, body=()):
        with self._lock:
            message = new_method_call(address, method, signature, body)
            reply = self.conn.send_and_get_reply(message, timeout=15)
        if reply.header.message_type.name == 'error':
            raise RuntimeError(f'{method}: {reply.body}')
        return reply.body

    # -- raw input (no easing; use Human for anything visible) ---------------
    def warp(self, x: float, y: float):
        x = min(max(float(x), 0.0), self.width - 1.0)
        y = min(max(float(y), 0.0), self.height - 1.0)
        self._call(self.rd, 'NotifyPointerMotionAbsolute', 'sdd', (self.stream_path, x, y))
        self.pointer = (x, y)

    def button(self, name: str, pressed: bool):
        self._call(self.rd, 'NotifyPointerButton', 'ib', (BTN[name], pressed))

    def wheel(self, steps: int):
        """One discrete wheel notch per call sign: +1 = down (towards the user)."""
        self._call(self.rd, 'NotifyPointerAxisDiscrete', 'ui', (0, int(steps)))

    def key(self, keysym: int, pressed: bool):
        self._call(self.rd, 'NotifyKeyboardKeysym', 'ub', (keysym, pressed))

    # -- recording -----------------------------------------------------------
    def start_recording(self, output: Path, fps: int = 30, quantizer: int = 18,
                        preset: str = 'superfast', threads: int = 12, source_fps: int = 60,
                        constant_rate: bool = True, nice: int = 5, diagnostics: bool = False) -> None:
        """Record the monitor to H.264 MP4 at a constant ``fps`` until stop_recording().

        mutter is asked for up to ``source_fps`` (its own frame clock), and
        ``videorate`` picks the newest frame for every output slot. Asking
        mutter for exactly 30 made its rate limiter skip frames that fell a
        little early, which ``videorate`` then filled with repeats.
        ``diagnostics`` enables fixed caps and buffer-mapping logs for a bounded investigation.
        """
        output = Path(output)
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        pipeline = [
            'pipewiresrc', f'path={self.node_id}', 'do-timestamp=true', 'always-copy=true',
            f'keepalive-time={int(1000 / fps)}', '!',
            f'video/x-raw,max-framerate={source_fps}/1', '!',
            'queue', 'max-size-buffers=8', 'max-size-time=0', 'max-size-bytes=0', '!',
            *(['videorate', '!', f'video/x-raw,framerate={fps}/1', '!'] if constant_rate else []),
            'videoconvert', 'n-threads=8', '!', 'video/x-raw,format=I420', '!',
            'queue', 'max-size-buffers=240', 'max-size-time=0', 'max-size-bytes=0', '!',
            'x264enc', f'speed-preset={preset}', 'pass=quant', f'quantizer={quantizer}',
            f'threads={threads}', f'key-int-max={fps * 2}', 'bframes=0', '!',
            'video/x-h264,profile=high,stream-format=avc', '!',
            'mp4mux', 'faststart=true', '!', 'filesink', f'location={output}',
        ]
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(('GST_', 'LD_', 'PYTHON', 'CONDA'))}
        env['PATH'] = '/usr/bin:/bin'
        if diagnostics:
            env['GST_DEBUG'] = ('2,pipewiresrc:6,pipewirepool:6,videofilter:6,default:6,'
                                'GST_BUFFER:6,GST_MEMORY:6,bufferpool:6')
            env['GST_DEBUG_NO_COLOR'] = '1'
        self._log = output.with_suffix('.gst.log').open('wb')
        # Niced: when the machine is busy the compositor (which draws the
        # frames and the pointer) must win the CPU over the encoder.
        self.recorder = subprocess.Popen(['/usr/bin/nice', '-n', str(nice), SYSTEM_GST, '-e', *(['-v'] if diagnostics else []), *pipeline], env=env,
                                         stdout=self._log, stderr=self._log, stdin=subprocess.DEVNULL)
        self.output = output
        self.recording_started = time.monotonic()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self.recorder.poll() is not None:
                raise RuntimeError('GStreamer recorder exited: ' + output.with_suffix('.gst.log').read_text())
            # mp4mux faststart writes the file only at EOS; the pipeline log
            # says when the live source is running.
            if b'PLAYING' in output.with_suffix('.gst.log').read_bytes():
                break
            time.sleep(.05)
        else:
            self.recorder.kill()
            self.recorder.wait()
            self.recorder = None
            raise RuntimeError('GStreamer recorder did not start (caps not negotiated?): '
                               + output.with_suffix('.gst.log').read_text())
        time.sleep(.3)

    def stop_recording(self) -> dict:
        if self.recorder is None:
            raise RuntimeError('Not recording')
        seconds = time.monotonic() - self.recording_started
        self.recorder.send_signal(signal.SIGINT)      # -e: EOS, so mp4mux finalises
        try:
            self.recorder.wait(timeout=120)
        finally:
            if self.recorder.poll() is None:
                self.recorder.kill()
                self.recorder.wait()
            self._log.close()
        rc, self.recorder = self.recorder.returncode, None
        if rc:
            raise RuntimeError(f'GStreamer recorder failed ({rc}): '
                               + self.output.with_suffix('.gst.log').read_text())
        log = self.output.with_suffix('.gst.log').read_text(errors='replace')
        if any(message in log for message in
               ('invalid video buffer received', 'gst_buffer_peek_memory')):
            raise RuntimeError('GStreamer recorder produced invalid video buffers: ' + log)
        return {'video': str(self.output), 'wall_seconds': seconds, 'node_id': self.node_id}

    def close(self):
        if self.recorder is not None:
            try:
                self.stop_recording()
            except Exception:
                pass
        try:
            self._call(self.rd, 'Stop')
        except Exception:
            pass
        self.conn.close()


def min_jerk(t: float) -> float:
    return t * t * t * (10 - 15 * t + 6 * t * t)


class Human:
    """Human-like use of a HostSession's real pointer and keyboard.

    Every visible pointer change is an eased path; ``log`` keeps a trace of
    (monotonic seconds, x, y, event) for verification against the video.
    """

    def __init__(self, host: HostSession, *, hz: float = 60.0, seed: int = 7):
        self.host = host
        self.hz = hz
        self.rng = random.Random(seed)
        self.log = []

    @property
    def position(self):
        return self.host.pointer

    def _note(self, event, call_ms=None):
        x, y = self.host.pointer
        self.log.append((time.monotonic(), round(x, 2), round(y, 2), event, call_ms))

    #: Peak pointer speed, as a fraction of the screen width per 30 fps video
    #: frame. A minimum-jerk move peaks at 1.875x its mean speed; a fast
    #: human flick reads as a jump in a 30 fps video above ~3 %. 0.014 keeps
    #: the step under the clip gate bound (3 % per frame) even when the
    #: compositor drops one frame, which doubles the step seen in the video.
    peak_per_frame = 0.011

    def duration_for(self, distance: float) -> float:
        """Fitts-like: ~250 ms for a nudge, ~900 ms across the screen.

        Long moves are stretched (up to 3.6 s; with 1.5 s a 1760 px glide to the menu bar
        peaked at 2.9 % of the width per frame and the clip gate flagged it) so the peak per-frame step
        stays under :attr:`peak_per_frame` of the screen width.
        """
        if distance < 1:
            return 0.0
        seconds = (0.18 + 0.11 * math.log2(1 + distance / 12.0)) * self.rng.uniform(0.92, 1.08)
        seconds = min(0.9, max(0.25, seconds))
        # 1.15: the arc and the duration jitter add up to ~15 % peak speed.
        speed_floor = 1.15 * 1.875 * distance / (self.peak_per_frame * self.host.width * 30.0)
        # Past the cap the profile flattens (cruise_share, at most 0.8: a peak
        # of 1.175x the mean); a glide longer than even that allows takes longer.
        cap = max(self.longest_glide_s, speed_floor * 1.175 / 1.875)
        return min(cap, max(seconds, speed_floor))

    #: The longest glide. A move that would need longer to stay under
    #: :attr:`peak_per_frame` keeps this duration and flattens its speed
    #: profile instead (:meth:`cruise_share`).
    longest_glide_s = 3.6

    def cruise_share(self, distance: float, duration: float) -> float:
        """How much of a glide follows a constant speed instead of minimum jerk.

        A minimum-jerk move peaks at 1.875x its mean speed. Above about
        1,200 px the 3.6 s cap made that peak 1.5x :attr:`peak_per_frame`
        (lesson 08, 2026-10-17: a 2,080 px glide at 42 px per frame, and one
        dropped frame put an 84 px step in the video). Mixing in a share
        ``a`` of a straight constant-speed path lowers the peak to
        ``(1 - a) * 1.875 + a`` times the mean, so the long glide stays
        under the limit without taking longer. Short moves keep ``a = 0``.

        :param distance: the length of the move in pixels.
        :param duration: its duration in seconds.
        :returns: the share ``a`` in ``[0, 0.8]``.
        """
        if distance <= 0 or duration <= 0:
            return 0.0
        allowed = self.peak_per_frame * self.host.width * 30.0 / 1.15
        ratio = allowed / (distance / duration)
        share = (1.875 - ratio) / 0.875
        return 0.0 if share < 0.02 else min(0.8, share)   # 0.02: rounding at the floor

    def move_to(self, x: float, y: float, duration: float | None = None):
        """Glide to (x, y) on a slight arc with a minimum-jerk speed profile."""
        x0, y0 = self.host.pointer
        dx, dy = x - x0, y - y0
        distance = math.hypot(dx, dy)
        duration = self.duration_for(distance) if duration is None else duration
        if distance < 0.5 or duration <= 0:
            self.host.warp(x, y)
            self._note('move')
            return
        # Bow the path sideways by 2-6 % of its length (wrist arc), never more than 60 px.
        bow = min(60.0, distance * self.rng.uniform(0.02, 0.06)) * self.rng.choice((-1, 1))
        nx, ny = -dy / distance, dx / distance
        steps = max(2, int(round(duration * self.hz)))
        cruise = self.cruise_share(distance, duration)
        begin = time.perf_counter()
        for i in range(1, steps + 1):
            target = begin + i / self.hz
            while (now := time.perf_counter()) < target:
                time.sleep(min(0.004, target - now))
            u = i / steps
            s = (1 - cruise) * min_jerk(u) + cruise * u
            arc = 4 * s * (1 - s) * bow
            sent = time.perf_counter()
            self.host.warp(x0 + dx * s + nx * arc, y0 + dy * s + ny * arc)
            self._note('move', round((time.perf_counter() - sent) * 1000, 2))
        self.host.warp(x, y)
        self._note('move')

    def pause(self, seconds: float):
        time.sleep(max(0.0, seconds))

    def click(self, button: str = 'left', hold: float | None = None):
        self.host.button(button, True)
        self._note(f'press-{button}')
        time.sleep(hold if hold is not None else self.rng.uniform(0.07, 0.11))
        self.host.button(button, False)
        self._note(f'release-{button}')

    def click_at(self, x: float, y: float, *, settle: float = 0.12, button: str = 'left'):
        self.move_to(x, y)
        time.sleep(settle * self.rng.uniform(0.8, 1.3))
        self.click(button)

    def drag(self, x0: float, y0: float, x1: float, y1: float, *, button: str = 'left',
             duration: float | None = None):
        self.move_to(x0, y0)
        time.sleep(0.12)
        self.host.button(button, True)
        self._note(f'press-{button}')
        time.sleep(0.12)
        distance = math.hypot(x1 - x0, y1 - y0)
        self.move_to(x1, y1, duration or max(0.45, self.duration_for(distance) * 1.4))
        time.sleep(0.1)
        self.host.button(button, False)
        self._note(f'release-{button}')

    def scroll(self, notches: int, *, interval: float = 0.11):
        """Turn the wheel ``notches`` times (+ = down), one notch at a time."""
        for _ in range(abs(int(notches))):
            self.host.wheel(1 if notches > 0 else -1)
            self._note('wheel-down' if notches > 0 else 'wheel-up')
            time.sleep(interval * self.rng.uniform(0.8, 1.25))

    def press(self, key: str):
        sym = _keysym(key)
        self.host.key(sym, True)
        time.sleep(self.rng.uniform(0.04, 0.07))
        self.host.key(sym, False)
        self._note(f'key-{key}')

    def key_down(self, key: str):
        """Hold a key (e.g. a modifier: ``Control_L``) until :meth:`key_up`."""
        self.host.key(_keysym(key), True)
        self._note(f'keydown-{key}')

    def key_up(self, key: str):
        self.host.key(_keysym(key), False)
        self._note(f'keyup-{key}')

    def button_down(self, button: str = 'left'):
        """Press and hold a button where the pointer is (a drag continues with move_to)."""
        self.host.button(button, True)
        self._note(f'press-{button}')

    def button_up(self, button: str = 'left'):
        self.host.button(button, False)
        self._note(f'release-{button}')

    def chord(self, modifiers, key: str):
        """Hold ``modifiers``, press ``key``, release all -- one request, so an app that
        opens a modal dialog on the chord never sees the modifiers still held."""
        for name in modifiers:
            self.key_down(name)
            time.sleep(self.rng.uniform(0.03, 0.05))
        try:
            self.press(key)
        finally:
            for name in reversed(list(modifiers)):
                time.sleep(self.rng.uniform(0.02, 0.04))
                self.key_up(name)

    def modified_click(self, modifiers, button: str = 'left'):
        for name in modifiers:
            self.key_down(name)
        try:
            time.sleep(0.05)
            self.click(button)
        finally:
            for name in reversed(list(modifiers)):
                self.key_up(name)

    def double_click(self, button: str = 'left'):
        self.click(button, hold=self.rng.uniform(0.04, 0.06))
        time.sleep(self.rng.uniform(0.07, 0.10))
        self.click(button, hold=self.rng.uniform(0.04, 0.06))

    def type_text(self, text: str, *, cps: float = 9.0):
        """Type at a human rate (``cps`` characters/second, with jitter)."""
        for char in text:
            self.press(char)
            time.sleep(max(0.02, self.rng.gauss(1.0 / cps, 0.25 / cps)))

    def save_log(self, path: Path, origin: float | None = None):
        origin = self.log[0][0] if origin is None and self.log else (origin or 0.0)
        Path(path).write_text(json.dumps(
            [{'t': round(t - origin, 4), 'x': x, 'y': y, 'event': e, 'call_ms': c}
             for t, x, y, e, c in self.log],
            indent=0) + '\n')


class HumanProcess:
    """The same API as :class:`Human` (plus recording), run in a separate process.

    USE THIS FROM INSIDE THE APPLICATION PROCESS. A driver thread in the
    app's own interpreter shares the GIL with the Qt main thread; while the
    animated field paints, the driver was starved for up to ~130 ms, so the
    pointer path arrived in bursts (17 % of 60 Hz steps were > 40 ms late).
    Here the pointer is moved by a child interpreter with nothing else to do,
    and the recorder only sends targets (screen coordinates) and waits.
    """

    def __init__(self, *, seed: int = 7, log=None):
        import sys
        self._reply_buffer = bytearray()
        self._reply_eof = False
        self._proc = subprocess.Popen(
            [sys.executable, '-u', str(Path(__file__).resolve()), '--serve', '--seed', str(seed)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
        hello = self._request('hello')
        self.width, self.height = hello['size']

    def _request(self, op, *args, **kwargs):
        self._proc.stdin.write(json.dumps({'op': op, 'args': args, 'kwargs': kwargs}) + '\n')
        self._proc.stdin.flush()
        return self.collect(op)

    def __getattr__(self, name):
        if name in _SERVED:
            return lambda *a, **k: self._request(name, *a, **k)
        raise AttributeError(name)

    def pause(self, seconds: float):
        time.sleep(max(0.0, seconds))

    # -- asynchronous use: the caller keeps its own event loop running -------
    def submit(self, op, *args, **kwargs):
        """Send a request without waiting; collect it with :meth:`collect`.

        The Qt main thread uses this so it can keep painting (and keep
        handling the events the moving pointer produces) while the separate
        process glides the pointer.
        """
        if op not in _SERVED:
            raise ValueError(f'not a served operation: {op}')
        self._proc.stdin.write(json.dumps({'op': op, 'args': args, 'kwargs': kwargs}) + '\n')
        self._proc.stdin.flush()

    def reply_ready(self, timeout: float | None = 0.0) -> bool:
        """A complete reply may already be buffered even when the pipe is empty."""
        import select
        if b'\n' in self._reply_buffer or self._reply_eof:
            return True
        if not select.select([self._proc.stdout], [], [], timeout)[0]:
            return False
        chunk = os.read(self._proc.stdout.fileno(), 65536)
        if chunk:
            self._reply_buffer.extend(chunk)
        else:
            self._reply_eof = True
        return b'\n' in self._reply_buffer or self._reply_eof

    def collect(self, op='request'):
        while not self.reply_ready(None):
            pass
        line, separator, remaining = self._reply_buffer.partition(b'\n')
        self._reply_buffer = bytearray(remaining) if separator else bytearray()
        if not line:
            raise RuntimeError(f'human_input server died during {op}')
        reply = json.loads(line)
        if 'error' in reply:
            raise RuntimeError(f'{op}: {reply["error"]}')
        return reply.get('value')

    def close(self):
        if self._proc.poll() is None:
            try:
                self._request('close')
            except Exception:
                pass
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()


_SERVED = {'move_to', 'click', 'click_at', 'drag', 'scroll', 'press', 'type_text', 'warp',
           'position', 'start_recording', 'stop_recording', 'save_log', 'key_down', 'key_up',
           'button_down', 'button_up', 'double_click', 'hello', 'trace_mark', 'trace_since',
           'chord', 'modified_click'}


def serve(seed: int = 7) -> int:
    """stdin/stdout JSON-lines server owning one HostSession and one Human."""
    import sys
    host = HostSession()
    human = Human(host, seed=seed)
    out = sys.stdout
    for line in sys.stdin:
        request = json.loads(line)
        op, args, kwargs = request['op'], request.get('args', []), request.get('kwargs', {})
        try:
            if op == 'hello':
                value = {'size': [host.width, host.height]}
            elif op == 'close':
                host.close()
                out.write(json.dumps({'value': None}) + '\n')
                out.flush()
                return 0
            elif op == 'warp':
                value = host.warp(*args)
                human._note('warp')
            elif op == 'position':
                value = list(host.pointer)
            elif op == 'trace_mark':
                # Monotonic clock of this process plus the trace length, so a
                # recorder can cut the pointer trace of one clip out of the log.
                value = {'t': time.monotonic(), 'index': len(human.log)}
            elif op == 'trace_since':
                value = [{'t': t, 'x': x, 'y': y, 'event': e, 'call_ms': c}
                         for t, x, y, e, c in human.log[int(args[0]):]]
            elif op == 'start_recording':
                value = host.start_recording(Path(args[0]), **kwargs)
            elif op == 'stop_recording':
                value = host.stop_recording()
            elif op == 'save_log':
                value = human.save_log(Path(args[0]), **kwargs)
            else:
                value = getattr(human, op)(*args, **kwargs)
            out.write(json.dumps({'value': value}) + '\n')
        except Exception as error:                     # noqa: BLE001
            out.write(json.dumps({'error': f'{type(error).__name__}: {error}'}) + '\n')
        out.flush()
    host.close()
    return 0


class GuiCaller:
    """Run callables on the Qt main thread from a driver thread and wait for the result.

    Construct it ON THE MAIN THREAD -- before ``QApplication`` exists is fine,
    which is how a recorder wraps an unmodified ``spacr.qt.run()`` (spaCR's
    thread guard reports any QObject constructed on another thread). Calls are
    queued slot invocations delivered by whichever event loop is spinning, so
    they work while a modal dialog or a menu runs a nested loop.
    """

    def __init__(self):
        from PySide6.QtCore import QObject, Qt, Signal, Slot
        class _Bridge(QObject):
            call = Signal(object)

            @Slot(object)
            def run(self, job):
                fn, box, done = job
                try:
                    box['value'] = fn()
                except BaseException as error:          # noqa: BLE001
                    box['error'] = error
                done.set()

        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError('Construct GuiCaller on the main (Qt GUI) thread')
        self._bridge = _Bridge()
        self._bridge.call.connect(self._bridge.run, Qt.QueuedConnection)

    def post(self, fn):
        """Queue ``fn`` on the Qt main thread and return at once (no result)."""
        self._bridge.call.emit((fn, {}, threading.Event()))

    def __call__(self, fn, timeout: float = 30.0):
        box, done = {}, threading.Event()
        self._bridge.call.emit((fn, box, done))
        if not done.wait(timeout):
            raise TimeoutError('The Qt main thread did not run the call')
        if 'error' in box:
            raise box['error']
        return box.get('value')


def widget_center(widget, dx: float = 0.0, dy: float = 0.0):
    """Screen coordinates of a widget's centre (xcb under Xwayland: global == compositor)."""
    from PySide6.QtCore import QPoint
    point = widget.mapToGlobal(QPoint(widget.width() // 2, widget.height() // 2))
    return point.x() + dx, point.y() + dy


def widget_rect(widget):
    from PySide6.QtCore import QPoint
    origin = widget.mapToGlobal(QPoint(0, 0))
    return origin.x(), origin.y(), widget.width(), widget.height()


if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='human_input JSON-lines server (see HumanProcess)')
    parser.add_argument('--serve', action='store_true', required=True)
    parser.add_argument('--seed', type=int, default=7)
    sys.exit(serve(parser.parse_args().seed))
