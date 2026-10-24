"""Where do frames get lost between the app and the GNOME capture-host recording?

Run INSIDE ``gnome_capture_host.py`` (one GPU turn)::

    python tools/tutorials/gnome_capture_host.py --stage NEW -- \\
        python tools/tutorials/capture_host_bench.py --out OUT --app spacr

``--app spacr`` starts the real application with the capture profile
(maximised Home, the field animating); ``--app anim`` a trivial full-screen
Qt window that repaints everything every frame (no spaCR cost). While it
runs, a driver thread tries each capture variant for ``--seconds``:

* ``deliver-*``: ``pipewiresrc -> fpsdisplaysink(fakesink)``: how many frames
  mutter's ScreenCast actually hands over, per cursor mode, nothing encoded;
* ``encode-*``: the recorder pipeline of :mod:`human_input` (or a variant)
  to a file, then its distinct-frame rate as the clip gate measures it.

For each variant it also records the frames the app painted per second and
the CPU used by gnome-shell, Xwayland and the encoder.  ``bench.json``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import traceback

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GST = '/usr/bin/gst-launch-1.0'


def gst_env():
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GST_', 'LD_', 'PYTHON', 'CONDA'))}
    env['PATH'] = '/usr/bin:/bin'
    return env


def proc_cpu(pids):
    """Total utime+stime clock ticks of ``pids`` (missing pids count 0)."""
    total = 0
    for pid in pids:
        try:
            fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            total += int(fields[11]) + int(fields[12])
        except (OSError, IndexError, ValueError):
            pass
    return total


def pids_named(name):
    out = []
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                if (entry / 'comm').read_text().strip() == name:
                    out.append(int(entry.name))
            except OSError:
                pass
    return out


ENCODERS = {
    'superfast': ['x264enc', 'speed-preset=superfast', 'pass=quant', 'quantizer=18', 'threads=12'],
    'ultrafast': ['x264enc', 'speed-preset=ultrafast', 'pass=quant', 'quantizer=18', 'threads=12'],
}


def variants(seconds):
    rows = []
    for cursor in (1, 0, 2):
        rows.append({'name': f'deliver-cursor{cursor}', 'cursor': cursor, 'kind': 'deliver', 'always_copy': True})
    rows.append({'name': 'deliver-cursor1-nocopy', 'cursor': 1, 'kind': 'deliver', 'always_copy': False})
    for enc in ('superfast', 'ultrafast'):
        rows.append({'name': f'encode-{enc}-cfr60', 'cursor': 1, 'kind': 'encode', 'encoder': enc, 'cfr': 60})
    rows.append({'name': 'encode-superfast-cfr30', 'cursor': 1, 'kind': 'encode', 'encoder': 'superfast', 'cfr': 30})
    rows.append({'name': 'encode-ultrafast-vfr', 'cursor': 1, 'kind': 'encode', 'encoder': 'ultrafast', 'cfr': None})
    for row in rows:
        row['seconds'] = seconds
    return rows


def pipeline(row, node, output):
    source = ['pipewiresrc', f'path={node}', 'do-timestamp=true',
              f'always-copy={"true" if row.get("always_copy", True) else "false"}', '!',
              'video/x-raw,max-framerate=60/1', '!']
    if row['kind'] == 'deliver':
        # fpsdisplaysink (gst-plugins-bad) is not installed: identity prints one
        # last-message line per buffer mutter hands over.
        return source + ['identity', 'silent=false', '!', 'fakesink', 'sync=false']
    cfr = row.get('cfr')
    return source + ['queue', 'max-size-buffers=8', 'max-size-time=0', 'max-size-bytes=0', '!',
                     *(['videorate', '!', f'video/x-raw,framerate={cfr}/1', '!'] if cfr else []),
                     'videoconvert', 'n-threads=8', '!', 'video/x-raw,format=I420', '!',
                     'queue', 'max-size-buffers=240', 'max-size-time=0', 'max-size-bytes=0', '!',
                     *ENCODERS[row['encoder']], 'key-int-max=120', 'bframes=0', '!',
                     'video/x-h264,profile=high,stream-format=avc', '!',
                     'mp4mux', '!', 'filesink', f'location={output}']


def run_variant(row, out: Path, field_fps):
    from human_input import HostSession
    host = HostSession(cursor_mode=row['cursor'])
    try:
        output = out / f'{row["name"]}.mp4'
        log = (out / f'{row["name"]}.gst.log').open('w')
        shell = [p for p in pids_named('gnome-shell')]
        xway = pids_named('Xwayland')
        cpu0 = {'gnome-shell': proc_cpu(shell), 'Xwayland': proc_cpu(xway)}
        f0 = field_fps()
        t0 = time.monotonic()
        proc = subprocess.Popen([GST, '-e', '-v', *pipeline(row, host.node_id, output)], env=gst_env(),
                                stdout=log, stderr=subprocess.STDOUT)
        time.sleep(row['seconds'])
        f1 = field_fps()
        cpu1 = {'gnome-shell': proc_cpu(shell), 'Xwayland': proc_cpu(xway)}
        gst_cpu = proc_cpu([proc.pid])
        proc.send_signal(2)
        proc.wait(timeout=60)
        wall = time.monotonic() - t0
        log.close()
        tick = os.sysconf('SC_CLK_TCK')
        result = {**row, 'wall_s': round(wall, 2),
                  'app_painted_fps': round((f1['frames'] - f0['frames']) / (f1['t'] - f0['t']), 2),
                  'cpu_percent': {k: round(100 * (cpu1[k] - cpu0[k]) / tick / row['seconds'], 1) for k in cpu0},
                  'gst_cpu_percent': round(100 * gst_cpu / tick / row['seconds'], 1)}
        text = (out / f'{row["name"]}.gst.log').read_text(errors='replace')
        if row['kind'] == 'deliver':
            buffers = len(re.findall(r'last-message = chain', text))
            result['delivered_buffers'] = buffers
            result['delivered_avg_fps'] = round(buffers / wall, 2)
        else:
            import capture_clip_gate as gate
            from measure_capture_host_proof import exact_distinct
            track = [{'found': False}] * 100000
            changed, _, _ = gate.picture_changes(output, track, level=10, pixels=4, cursor_box=0,
                                                 top_bar_height=32, top_bar_every=10 ** 9)
            fps = gate.frame_rate(output) if row.get('cfr') else None
            seconds = gate.duration(output)
            result.update(frames=len(changed), video_seconds=round(seconds, 2),
                          distinct_per_second=round(sum(changed[1:]) / seconds, 2),
                          stream_fps=fps,
                          exact_distinct_fraction=round(sum(exact_distinct(output)[1]) / max(1, len(changed)), 3))
        return result
    finally:
        host.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--app', choices=('spacr', 'anim'), default='spacr')
    parser.add_argument('--seconds', type=float, default=6.0)
    parser.add_argument('--only', help='comma-separated variant names')
    parser.add_argument('--prepare-only', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not os.environ.get('SPACR_CAPTURE_HOST'):
        parser.error('run inside gnome_capture_host.py')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.meta_path[:] = [f for f in sys.meta_path
                        if '__editable__' not in (getattr(f, '__module__', '') or type(f).__module__)]
    sys.path[:0] = [str(REPO), str(HERE)]
    width, height = (int(v) for v in os.environ['SPACR_CAPTURE_HOST_SIZE'].split('x'))
    profile = out / 'profile'
    for name in ('config', 'state', 'spacr-home', 'logs', 'mpl', 'cache'):
        (profile / name).mkdir(parents=True, exist_ok=True)
    os.environ.update(QT_QPA_PLATFORM='xcb', QT_SCALE_FACTOR='1', QT_AUTO_SCREEN_SCALE_FACTOR='0',
                      QT_FONT_DPI='96', SPACR_LANGUAGE='en', XDG_CONFIG_HOME=str(profile / 'config'),
                      XDG_STATE_HOME=str(profile / 'state'), SPACR_HOME=str(profile / 'spacr-home'),
                      SPACR_LOG_DIR=str(profile / 'logs'), MPLCONFIGDIR=str(profile / 'mpl'),
                      XDG_CACHE_HOME=str(profile / 'cache'), SPACR_TUTORIAL_CACHE_ISOLATED='1')
    from human_input import GuiCaller
    rows = variants(args.seconds)
    if args.only:
        keep = set(args.only.split(','))
        rows = [r for r in rows if r['name'] in keep]
    results = {'app': args.app, 'monitor': os.environ['SPACR_CAPTURE_HOST_SIZE'],
               'loadavg_start': os.getloadavg(), 'variants': []}
    state = {'widget': None, 'frames': 0}
    gui = GuiCaller()

    def driver():
        from PySide6.QtWidgets import QApplication
        try:
            deadline = time.monotonic() + 180
            while QApplication.instance() is None and time.monotonic() < deadline:
                time.sleep(0.1)
            if args.app == 'spacr':
                from spacr.qt import app as spacr_app
                from spacr.qt.widgets.ambient import AmbientWidget
                while True:
                    try:
                        window = gui(lambda: next((w for w in QApplication.topLevelWidgets()
                                                   if isinstance(w, spacr_app.MainWindow) and w.isVisible()), None),
                                     timeout=10)
                    except TimeoutError:
                        window = None
                    if window is not None or time.monotonic() > deadline:
                        break
                    time.sleep(0.3)
                gui(window.showMaximized)
                time.sleep(7)

                def field_fps():
                    return gui(lambda: {'t': time.monotonic(), 'frames': sum(
                        w.frames_painted for w in window.findChildren(AmbientWidget)
                        if w.isVisible() and not w.property('spacrPopupBackdrop'))})
                results['field_cap'] = gui(lambda: [w.fps() for w in window.findChildren(AmbientWidget)
                                                    if w.isVisible()])
            else:
                time.sleep(4)

                def field_fps():
                    return {'t': time.monotonic(), 'frames': state['frames']}
            for row in rows:
                print(f'[bench] {row["name"]}', flush=True)
                try:
                    results['variants'].append(run_variant(row, out, field_fps))
                except Exception:                  # noqa: BLE001
                    results['variants'].append({**row, 'error': traceback.format_exc()})
                (out / 'bench.json').write_text(json.dumps(results, indent=2, default=str) + '\n')
                time.sleep(1)
        except BaseException:                      # noqa: BLE001
            results['error'] = traceback.format_exc()
        finally:
            results['loadavg_end'] = os.getloadavg()
            (out / 'bench.json').write_text(json.dumps(results, indent=2, default=str) + '\n')
            gui.post(QApplication.quit)

    threading.Thread(target=driver, daemon=True).start()
    if args.app == 'spacr':
        rc = subprocess.run([sys.executable, str(HERE / 'capture_host_mode.py'), '--prepare-profile',
                             '--font-scale', str(round(1.5 * width / 3840, 3)), '--out', str(out / 'profile.json')],
                            env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen'}).returncode
        if rc:
            return rc
        import spacr.qt
        spacr.qt.run(['--fresh', '--no-setup'])
    else:
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtGui import QColor, QPainter
        from PySide6.QtWidgets import QApplication, QWidget

        class Anim(QWidget):
            def paintEvent(self, _):
                state['frames'] += 1
                p = QPainter(self)
                hue = (state['frames'] * 3) % 360
                p.fillRect(self.rect(), QColor.fromHsv(hue, 120, 90))
                for k in range(200):
                    x = (k * 97 + state['frames'] * 7) % self.width()
                    y = (k * 53 + state['frames'] * 3) % self.height()
                    p.fillRect(x, y, 6, 6, Qt.white)
                p.end()

        app = QApplication(['anim'])
        w = Anim()
        w.setWindowFlag(Qt.FramelessWindowHint)
        w.showMaximized()
        timer = QTimer()
        timer.setTimerType(Qt.PreciseTimer)
        timer.timeout.connect(w.update)
        timer.start(16)
        app.exec()
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
