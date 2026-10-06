"""Frozen native 4K widget producer, cold cost and steady CPU audit."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import resource
import statistics
import sys
import time
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = Path(os.environ['SPACR_PROOF_REPO']).resolve()
sys.path.insert(0, str(REPO))
from PySide6.QtCore import QEventLoop, QPoint, QSettings, Qt, QTimer
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QApplication, QWidget

from spacr.qt import preferences as prefs

importlib.import_module('spacr.qt.widgets')

parser = argparse.ArgumentParser()
parser.add_argument('theme')
parser.add_argument('--palette', default='random')
parser.add_argument('variant', choices=['before', 'after'])
args = parser.parse_args()
path = ROOT / (args.variant + '.py')
expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
spec = importlib.util.spec_from_file_location('spacr.qt.widgets._frozen_native_audit', path)
a = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = a
spec.loader.exec_module(a)
assert os.environ['QT_QPA_PLATFORM'] == 'xcb'
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
app = QApplication([])
app.setQuitOnLastWindowClosed(False)
store = QSettings(str(ROOT / (args.theme + '-' + args.variant + '-' + args.palette + '-prefs.ini')), QSettings.IniFormat)
prefs._settings = lambda: store
prefs._set_ambient_custom_colors(('#3b82f6', '#ff00ff'))


def wait(milliseconds):
    loop = QEventLoop()
    timer = QTimer()
    timer.setSingleShot(True)
    timer.setTimerType(Qt.PreciseTimer)
    timer.timeout.connect(loop.quit)
    timer.start(milliseconds)
    loop.exec()


def summary(values):
    if not values:
        return None
    ordered = sorted(values)
    return {'median_ms': statistics.median(values),
            'p95_ms': ordered[math.ceil(len(values) * .95) - 1],
            'max_ms': max(values), 'samples': len(values)}


class TimedWidget(a.AmbientWidget):
    def __init__(self, parent):
        self.paints = []
        super().__init__(parent, theme=args.theme, palette=args.palette,
                         background='#101418', seed=42, fps=24, blur=0,
                         speed=1, size=1, resolution=2, density=1, gravity_radius=.25)

    def paintEvent(self, event):
        started = time.perf_counter()
        super().paintEvent(event)
        self.paints.append((started, (time.perf_counter() - started) * 1000))


host = QWidget()
host.resize(3840, 2160)
beats = []
heart = QTimer()
heart.setTimerType(Qt.PreciseTimer)
heart.setInterval(10)
heart.timeout.connect(lambda: beats.append(time.perf_counter()))
heart.start()
show_started = time.perf_counter()
widget = TimedWidget(host)
widget.follow_parent()
host.show()
host.activateWindow()
app.setActiveWindow(host)
QCursor.setPos(widget.mapToGlobal(QPoint(1920, 1080)))
app.processEvents()
show_ms = (time.perf_counter() - show_started) * 1000
engine = widget.engine
assert a.screen_pixels(widget) == 3840 * 2160
assert engine.max_pixels == 3840 * 2160
if args.theme.startswith('data_art_'):
    assert engine.buffer_size(3840, 2160) == (3840, 2160)
shades, publications = [], []
original_shade = engine.shade
last_clock = [engine.time]


def shade(width, height):
    clock = engine.time
    started = time.perf_counter()
    image = original_shade(width, height)
    shades.append((started, (time.perf_counter() - started) * 1000, clock))
    last_clock[0] = clock
    return image


with widget._engine_lock:
    engine.shade = shade
producer = widget._producer_box[0]
original_publish = producer.publish


def publish(image):
    original_publish(image)
    publications.append((time.perf_counter(), last_clock[0]))


producer.publish = publish
cold_started = time.perf_counter()
wait(2000)
cold_elapsed = time.perf_counter() - cold_started
cold_gaps = [(b - c) * 1000 for c, b in pairwise(beats)]
cold = {'current_rss_kib': int(Path('/proc/self/status').read_text().split('VmRSS:')[1].split()[0]), 'widget_show_ms': show_ms, 'post_show_seconds': cold_elapsed,
        'post_show_published_frames': len(publications),
        'post_show_published_fps': len(publications) / cold_elapsed,
        'worker_shade': summary([v[1] for v in shades]),
        'heartbeat_10ms': summary(cold_gaps),
        'compile_ready': a._PACKED_SCATTER is not None,
        'compile_failed': a._PACKED_SCATTER_FAILED, 'colored_compile_ready': a._COLORED_SCATTER is not None, 'colored_compile_failed': a._COLORED_SCATTER_FAILED,
        'rss_peak_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
widget.set_time(95.0 if args.theme == 'data_art_fungal_growth' else 8.0)
wait(2200)
if args.theme == 'data_art_point_atlas':
    assert a._PACKED_SCATTER is not None and not a._PACKED_SCATTER_FAILED
if args.theme == 'data_art_thore':
    widget.set_time(16.5)
    wait(100)
assert engine.pointer is not None and abs(engine.pointer[0]-.5)<.01 and abs(engine.pointer[1]-.5)<.01
records = []
for repeat in range(1):
    shades.clear()
    publications.clear()
    beats.clear()
    widget.paints.clear()
    first_paint, first_repeat = widget.frames_painted, widget.repeated_frames
    first_clock = engine.time
    start = time.perf_counter()
    wait(5000)
    elapsed = time.perf_counter() - start
    copied_shades, copied_publications = list(shades), list(publications)
    gaps = [(b - c) * 1000 for c, b in pairwise(beats)]
    intervals = [(b[0] - c[0]) * 1000 for c, b in pairwise(copied_publications)]
    lightning_samples = [v for v in copied_shades
                         if args.theme == 'data_art_thore' and v[2] % engine._event_interval < .34]
    record = {'repeat': repeat + 1, 'steady_seconds': elapsed,
              'published_frames': len(copied_publications),
              'published_fps': len(copied_publications) / elapsed,
              'distinct_published_clocks': len({v[1] for v in copied_publications}),
              'first_clock': first_clock, 'clock_advance': engine.time - first_clock,
              'painted_fps': (widget.frames_painted - first_paint) / elapsed,
              'repeated_frame_objects': widget.repeated_frames - first_repeat,
              'worker_shade': summary([v[1] for v in copied_shades]),
              'gui_paint': summary([v[1] for v in widget.paints]),
              'heartbeat_10ms': summary(gaps), 'publication_intervals': summary(intervals),
              'lightning_worker_shades': summary([v[1] for v in lightning_samples]),
              'load_average': list(os.getloadavg()), 'current_rss_kib': int(Path('/proc/self/status').read_text().split('VmRSS:')[1].split()[0])}
    records.append(record)
    print(json.dumps(record), flush=True)
heart.stop()
image = widget.grab().toImage()
image.save(str(ROOT / (args.theme + '-' + args.variant + '-' + args.palette + '-3840.png')))
frame_hash = hashlib.sha256(image.bits().tobytes()).hexdigest()
if args.theme == 'data_art_thore':
    with widget._engine_lock:
        engine.set_time(16.97)
        flash = engine.shade(3840, 2160)
    flash.save(str(ROOT / (args.theme + '-' + args.variant + '-' + args.palette + '-lightning-3840.png')))
host.hide()
retired_immediately = not widget.shading_thread_alive()
wait(200)
retired_after_wait = not widget.shading_thread_alive()
widget.deleteLater()
host.deleteLater()
app.processEvents()
receipt = {'source_commit': 'b2aed3966c454702a23adbb459791fe32d0b4182',
           'source_sha256': expected_hash, 'source_path': str(path),
           'variant': args.variant, 'theme': args.theme, 'label': a.theme_label(args.theme),
           'viewport': [3840, 2160], 'screen_pixels': 3840 * 2160,
           'buffer': list(engine.buffer_size(3840, 2160)), 'max_pixels': engine.max_pixels,
           'controls': {'fps': 24, 'active_rate_cap': widget._rate(),
                        'resolution': engine.resolution, 'density': engine.density,
                        'size': engine.size, 'speed': engine.speed, 'blur': engine.blur,
                        'gravity_radius': .25, 'pointer': engine.pointer, 'seed': 42, 'background': '#101418',
                        'palette': args.palette, 'colors': [color.name() for color in engine.colors]},
           'qt_platform': os.environ['QT_QPA_PLATFORM'],
           'scope': 'actual software Xvfb native4K AmbientWidget/FrameProducer/QTimer; fresh Python process per theme; no inference or GPU; end grabs outside steady timing; requested controls unchanged',
           'cold': cold, 'records': records, 'frame_sha256': frame_hash,
           'worker_stopped_immediately_after_hide': retired_immediately,
           'worker_stopped_200ms_after_hide': retired_after_wait,
           'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
           'dependency_source_sha256': {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in [REPO/'spacr/qt/theme.py', REPO/'spacr/qt/preferences.py', REPO/'spacr/qt/gil_priority.py']}}
(ROOT / (args.theme + '-' + args.variant + '-' + args.palette + '.json')).write_text(json.dumps(receipt, indent=2) + '\n')
assert retired_after_wait
