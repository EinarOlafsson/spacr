import hashlib
import argparse
import gzip
import importlib.util
import json
import resource
import statistics
import sys
import time
from pathlib import Path
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
import spacr.qt.widgets

parser = argparse.ArgumentParser(description='Probe a verified frozen renderer.')
parser.add_argument('theme', choices=('fungal', 'thore', 'aurora'))
parser.add_argument('--renderer', type=Path, required=True)
parser.add_argument('--source-sha256', required=True)
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
payload = args.renderer.read_bytes()
if args.renderer.suffix == '.gz':
    payload = gzip.decompress(payload)
if hashlib.sha256(payload).hexdigest() != args.source_sha256:
    parser.error('renderer source does not match the required frozen SHA256')
ambient = importlib.util.module_from_spec(importlib.util.spec_from_loader(
    'spacr.qt.widgets._frozen_native_probe', loader=None))
sys.modules[ambient.__name__] = ambient
ambient.__file__ = str(args.renderer.resolve())
exec(compile(payload, ambient.__file__, 'exec'), ambient.__dict__)
args.output_dir.mkdir(parents=True, exist_ok=True)
name = args.theme
choices = {
    'fungal': ('data_art_fungal_growth', 'deepwater'),
    'thore': ('data_art_thore', 'midnight'),
    'aurora': ('aurora', 'borealis'),
}
theme, palette = choices[name]
app = QApplication([])
ambient.screen_pixels = lambda _widget=None: 3840 * 2160
paint_times = []
original_paint = ambient.AmbientWidget._paint_ambient
def timed_paint(self, event):
    at = time.monotonic()
    result = original_paint(self, event)
    paint_times.append((time.monotonic() - at) * 1000)
    return result
ambient.AmbientWidget._paint_ambient = timed_paint
widget = ambient.AmbientWidget(theme=theme, palette=palette,
                               background='#09121b', seed=7, fps=24,
                               blur=0.0, density=0.25, size=0.25,
                               resolution=1.0, gravity_radius=0.0)
widget.resize(3840, 2160)
widget.show()
beats, captures = [], []
start = time.monotonic()
last = [start]
heart = QTimer()
heart.setInterval(25)
def heartbeat():
    now = time.monotonic()
    beats.append((round(now-start,3), (now - last[0]) * 1000))
    last[0] = now
heart.timeout.connect(heartbeat)
heart.start()
def capture(tag):
    now = time.monotonic()
    tick = widget.engine.time
    shaded = widget.frames_shaded()
    painted = widget.frames_painted
    at = time.monotonic()
    image = widget.grab().toImage()
    grab_ms = (time.monotonic() - at) * 1000
    digest = hashlib.sha256(image.bits().tobytes()).hexdigest()
    if tag == 'high':
        image.save(str(args.output_dir / f'{name}-widget-4k.png'))
    captures.append({'phase': tag, 'elapsed_s': round(now-start,3),
                     'clock_s': round(tick,3), 'shaded': shaded,
                     'painted': painted, 'grab_ms': round(grab_ms,2),
                     'sha256': digest})
def high_controls():
    at = time.monotonic()
    widget.set_density(3.0)
    widget.set_size_scale(2.5)
    captures.append({'phase':'high_controls',
                     'setter_ms':round((time.monotonic()-at)*1000,2)})
def finish():
    capture('high')
    producer = widget._producer_box[0]
    result = {'theme':name,'source':ambient.__file__,
              'source_sha256': args.source_sha256,
              'size':[widget.width(),widget.height()],
              'buffer_size':list(widget.engine.buffer_size(3840,2160)),
              'frames_painted':widget.frames_painted,
              'frames_shaded':widget.frames_shaded(),
              'repeated_frames':widget.repeated_frames,
              'clock_s':round(widget.engine.time,3),
              'producer_alive_before_hide':producer.is_alive() if producer else False,
              'captures':captures,
              'heartbeat_n':len(beats),
              'heartbeat_median_ms':round(statistics.median(gap for _,gap in beats),2) if beats else None,
              'heartbeat_p95_ms':round(sorted(gap for _,gap in beats)[int(.95*(len(beats)-1))],2) if beats else None,
              'heartbeat_max_ms':round(max(gap for _,gap in beats),2) if beats else None,
              'heartbeat_largest':[(at,round(gap,2)) for at,gap in sorted(beats,key=lambda row:row[1],reverse=True)[:6]],
              'paint_median_ms':round(statistics.median(paint_times),2) if paint_times else None,
              'paint_p95_ms':round(sorted(paint_times)[int(.95*(len(paint_times)-1))],2) if paint_times else None,
              'paint_max_ms':round(max(paint_times),2) if paint_times else None,
              'rss_peak_mib':round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1)}
    widget.hide()
    app.processEvents()
    result['producer_alive_after_hide'] = widget.shading_thread_alive()
    result['timer_after_hide'] = widget.is_running()
    print(json.dumps(result,sort_keys=True))
    widget.close()
    app.quit()
QTimer.singleShot(550, lambda:capture('low'))
QTimer.singleShot(1000, high_controls)
QTimer.singleShot(2200, finish)
app.exec()
