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
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = Path(os.environ.get('SPACR_PROOF_REPO', '/mnt/wd4tb/spacr-worktrees/codex-fungal-sparse-cache-20261006'))
sys.path.insert(0, str(REPO))
from PySide6.QtCore import QEventLoop, QSettings, QTimer, Qt
from PySide6.QtWidgets import QApplication, QWidget
from spacr.qt import preferences as prefs
import spacr.qt.widgets

parser = argparse.ArgumentParser()
parser.add_argument('theme')
parser.add_argument('variant', choices=['before', 'after'])
parser.add_argument('--palette', choices=['spacr', 'random', 'custom'], default='spacr')
parser.add_argument('--background', default='#101418')
parser.add_argument('--run-id', default='first')
parser.add_argument('--profile', action='store_true')
parser.add_argument('--density', type=float, default=3)
parser.add_argument('--resolution', type=float, default=2)
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
                         background=args.background, seed=42, fps=24, blur=0,
                         speed=1, size=2.5, resolution=args.resolution, density=args.density, gravity_radius=0)

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
stages = {}
profiled_frames = []

def timed(name, function, *arguments):
    started = time.perf_counter()
    try:
        return function(*arguments)
    finally:
        stages[name] = stages.get(name, 0) + (time.perf_counter() - started) * 1000

if args.profile:
    from PySide6.QtGui import QPainter

    class ProfilePainter(QPainter):
        def drawPath(self, path):
            return timed("Qt.drawPath", super().drawPath, path)

        def setPen(self, pen):
            return timed("Qt.setPen", super().setPen, pen)

        def drawEllipse(self, *arguments):
            return timed("Qt.drawEllipse", super().drawEllipse, *arguments)

    a.QPainter = ProfilePainter
    with widget._engine_lock:
        for name in ("geometry", "_fungal_paths", "_paint_cached_fungal_paths", "_warm_fungal_raster", "_reuse_fungal_raster", "_paint_fungal_tips"):
            original = getattr(engine, name)
            setattr(engine, name, lambda *values, key=name, function=original: timed(key, function, *values))


def shade(width, height):
    clock = engine.time
    stages.clear()
    started = time.perf_counter()
    image = original_shade(width, height)
    shades.append((started, (time.perf_counter() - started) * 1000, clock))
    profiled_frames.append(dict(stages))
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
cold_gaps = [(b - c) * 1000 for c, b in zip(beats, beats[1:])]
cold = {'current_rss_kib': int(Path('/proc/self/status').read_text().split('VmRSS:')[1].split()[0]), 'widget_show_ms': show_ms, 'post_show_seconds': cold_elapsed,
        'post_show_published_frames': len(publications),
        'post_show_published_fps': len(publications) / cold_elapsed,
        'worker_shade': summary([v[1] for v in shades]),
        'heartbeat_10ms': summary(cold_gaps),
        'compile_ready': a._PACKED_SCATTER is not None,
        'compile_failed': a._PACKED_SCATTER_FAILED,
        'rss_peak_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
widget.set_time(95.0 if args.theme == 'data_art_fungal_growth' else 8.0)
wait(2200)
if args.theme == 'data_art_point_atlas':
    assert a._PACKED_SCATTER is not None and not a._PACKED_SCATTER_FAILED
if args.theme == 'data_art_thore':
    widget.set_time(16.5)
    wait(100)
records = []
for repeat in range(3 if args.theme == 'data_art_point_atlas' else 1):
    shades.clear()
    publications.clear()
    beats.clear()
    widget.paints.clear()
    profiled_frames.clear()
    first_paint, first_repeat = widget.frames_painted, widget.repeated_frames
    first_clock = engine.time
    start = time.perf_counter()
    wait(5000)
    elapsed = time.perf_counter() - start
    copied_shades, copied_publications = list(shades), list(publications)
    gaps = [(b - c) * 1000 for c, b in zip(beats, beats[1:])]
    intervals = [(b[0] - c[0]) * 1000 for c, b in zip(copied_publications, copied_publications[1:])]
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
              'inclusive_profile_stages': {name: summary([row.get(name, 0) for row in profiled_frames]) for name in set().union(*(row.keys() for row in profiled_frames))}, 'load_average': list(os.getloadavg()), 'current_rss_kib': int(Path('/proc/self/status').read_text().split('VmRSS:')[1].split()[0])}
    records.append(record)
    print(json.dumps(record), flush=True)
heart.stop()
image = widget.grab().toImage()

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
receipt = {'source_commit': 'packed-add-private-proof / Current scalar ambient ade8062f vs density-domain-only private candidate',
           'source_sha256': expected_hash, 'source_path': str(path),
           'variant': args.variant, 'theme': args.theme, 'label': a.theme_label(args.theme),
           'viewport': [3840, 2160], 'screen_pixels': 3840 * 2160,
           'buffer': list(engine.buffer_size(3840, 2160)), 'max_pixels': engine.max_pixels,
           'controls': {'fps': 24, 'active_rate_cap': widget._rate(),
                        'resolution': engine.resolution, 'density': engine.density,
                        'size': engine.size, 'speed': engine.speed, 'blur': engine.blur,
                        'gravity_radius': 0, 'seed': 42, 'background': args.background,
                        'palette': args.palette, 'colors': [c.name() for c in engine._colors]},
           'qt_platform': os.environ['QT_QPA_PLATFORM'],
           'profile_enabled': args.profile,
           'profile_scope': 'inclusive wrappers overlap and introduce overhead; timing is diagnostic, not an uninstrumented performance acceptance',
           'scope': 'actual software Xvfb native4K AmbientWidget/FrameProducer/QTimer; fresh Python process per theme; no inference or GPU; end grabs outside steady timing; requested controls unchanged',
           'cold': cold, 'records': records, 'frame_sha256': frame_hash,
           'worker_stopped_immediately_after_hide': retired_immediately,
           'worker_stopped_200ms_after_hide': retired_after_wait,
           'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
           'dependency_source_sha256': {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in [REPO/'spacr/qt/theme.py', REPO/'spacr/qt/preferences.py', REPO/'spacr/qt/gil_priority.py']}}
(ROOT / (args.theme + '-' + args.variant + '-' + args.palette + '-' + str(args.resolution) + '-' + str(args.density) + '-' + args.run_id + '.json')).write_text(json.dumps(receipt, indent=2) + '\n')
assert retired_after_wait
