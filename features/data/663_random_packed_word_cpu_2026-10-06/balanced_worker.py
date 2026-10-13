"""Measure warmed native widgets in ABBA order with real producer and GUI."""
import gc
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
sys.path.insert(0, os.environ['SPACR_PROOF_REPO'])
from PySide6.QtCore import QEventLoop, QPoint, Qt, QTimer
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QApplication, QWidget

importlib.import_module('spacr.qt.widgets')
app = QApplication([])
app.setQuitOnLastWindowClosed(False)
assert os.environ['QT_QPA_PLATFORM'] == 'xcb'
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
modules = {}
for label in ['before', 'after']:
    spec = importlib.util.spec_from_file_location('spacr.qt.widgets._abba_' + label,
                                                ROOT / (label + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module._warm_packed_scatter()
    assert module._COLORED_SCATTER is not None
    modules[label] = module


def wait(milliseconds):
    loop = QEventLoop()
    timer = QTimer()
    timer.setSingleShot(True)
    timer.setTimerType(Qt.PreciseTimer)
    timer.timeout.connect(loop.quit)
    timer.start(milliseconds)
    loop.exec()


def summary(values):
    ordered = sorted(values)
    return {'samples': len(values), 'median_ms': statistics.median(values),
            'p95_ms': ordered[math.ceil(len(values) * .95) - 1], 'max_ms': max(values)}


host = QWidget()
host.resize(3840, 2160)
host.show()
host.activateWindow()
app.setActiveWindow(host)
records = []
for index, label in enumerate(['before', 'after', 'after', 'before']):
    module = modules[label]
    paints = []

    class TimedWidget(module.AmbientWidget):
        def paintEvent(self, event):
            started = time.perf_counter()
            super().paintEvent(event)
            paints.append((time.perf_counter() - started) * 1000)

    widget = TimedWidget(host, theme='data_art_genetic_advection', palette='random',
                         background='#101418', seed=42, fps=24, blur=0,
                         speed=1, size=1, resolution=2, density=1, gravity_radius=.25)
    widget.follow_parent()
    widget.show()
    QCursor.setPos(widget.mapToGlobal(QPoint(1920, 1080)))
    app.processEvents()
    engine = widget.engine
    producer = widget._producer_box[0]
    shades, publications, beats = [], [], []
    original_shade, original_publish = engine.shade, producer.publish
    last_clock = [engine.time]

    def shade(width, height):
        clock = engine.time
        started = time.perf_counter()
        image = original_shade(width, height)
        shades.append((time.perf_counter() - started) * 1000)
        last_clock[0] = clock
        return image

    def publish(image):
        original_publish(image)
        publications.append((time.perf_counter(), last_clock[0]))

    with widget._engine_lock:
        engine.shade = shade
    producer.publish = publish
    widget.set_time(8)
    wait(2200)
    assert engine.buffer_size(3840, 2160) == (3840, 2160)
    assert engine.pointer is not None and abs(engine.pointer[0] - .5) < .01
    heart = QTimer()
    heart.setTimerType(Qt.PreciseTimer)
    heart.setInterval(10)
    heart.timeout.connect(lambda: beats.append(time.perf_counter()))
    shades.clear()
    publications.clear()
    paints.clear()
    first_clock = engine.time
    first_paints = widget.frames_painted
    heart.start()
    started = time.perf_counter()
    wait(5000)
    elapsed = time.perf_counter() - started
    heart.stop()
    records.append({'index': index, 'variant': label, 'steady_seconds': elapsed,
                    'published_fps': len(publications) / elapsed,
                    'published_frames': len(publications),
                    'distinct_clocks': len({v[1] for v in publications}),
                    'painted_fps': (widget.frames_painted - first_paints) / elapsed,
                    'clock_advance': engine.time - first_clock,
                    'worker_shade': summary(shades), 'gui_paint': summary(paints),
                    'publication_intervals': summary([(b[0] - a[0]) * 1000
                                                     for a, b in pairwise(publications)]),
                    'heartbeat_10ms': summary([(b - a) * 1000 for a, b in pairwise(beats)]),
                    'load_average': list(os.getloadavg())})
    widget.hide()
    wait(200)
    assert not producer.is_alive()
    records[-1]['worker_stopped_on_hide'] = True
    print(json.dumps(records[-1]), flush=True)
    widget.deleteLater()
    wait(10)
    del widget, engine, producer, original_shade, original_publish, shade, publish
    gc.collect()
host.close()
receipt = {'source_sha256': {label: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                             for label, module in modules.items()},
           'scope': 'actual warmed 3840x2160 software Xvfb GUI/producer, ABBA, 24 cap',
           'native_size': [3840, 2160], 'records': records,
           'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
(ROOT / 'balanced_worker.json').write_text(json.dumps(receipt, indent=2) + '\n')
