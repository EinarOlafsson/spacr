"""Replay a native AmbientWidget and producer cadence for one frozen renderer."""

import hashlib
import importlib
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
importlib.import_module('spacr.qt.widgets')

source = Path(os.environ['AMBIENT_SOURCE'])
spec = importlib.util.spec_from_file_location(
    'spacr.qt.widgets._advection_probe', source)
ambient = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ambient
spec.loader.exec_module(ambient)
app = QApplication.instance() or QApplication([])
app.setQuitOnLastWindowClosed(False)


def wait(milliseconds):
    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()


def summary(values):
    return {'median_ms': statistics.median(values),
            'p95_ms': sorted(values)[math.ceil(len(values) * .95) - 1],
            'samples': len(values)}


class TimedWidget(ambient.AmbientWidget):
    def __init__(self):
        self.paint_samples = []
        super().__init__(theme='data_art_genetic_advection', palette='spacr',
                         background='#101418', fps=24, seed=42, blur=0,
                         speed=1, size=1, resolution=2, density=1,
                         direction=ambient.DEFAULT_DRIFT_DIRECTION)

    def _follow_screen(self):
        with self._engine_lock:
            self._engine.set_max_pixels(max(1, self.width() * self.height()))

    def _data_art_pointer_for_tick(self):
        return (.5, .5)

    def paintEvent(self, event):
        started = time.perf_counter()
        super().paintEvent(event)
        self.paint_samples.append((time.perf_counter() - started) * 1000)


def probe(width, height):
    widget = TimedWidget()
    widget.resize(width, height)
    widget.show()
    app.setActiveWindow(widget)
    app.processEvents()
    widget.set_time(8.0)
    widget.set_gravity_radius(float(os.environ.get('RADIUS', '.25')))
    engine = widget._engine
    original_shade = engine.shade
    shade_samples = []
    published = []
    last_clock = [engine.time]

    def timed_shade(w, h):
        started = time.perf_counter()
        image = original_shade(w, h)
        shade_samples.append((time.perf_counter() - started) * 1000)
        last_clock[0] = engine.time
        return image

    with widget._engine_lock:
        engine.shade = timed_shade
    producer = widget._producer_box[0]
    original_publish = producer.publish

    def timed_publish(image):
        original_publish(image)
        published.append((time.perf_counter(), last_clock[0]))

    producer.publish = timed_publish
    heartbeats = []
    timer = QTimer()
    timer.setTimerType(Qt.PreciseTimer)
    timer.setInterval(10)
    timer.timeout.connect(lambda: heartbeats.append(time.perf_counter()))
    timer.start()
    wait(600)
    shade_samples.clear()
    published.clear()
    heartbeats.clear()
    widget.paint_samples.clear()
    before = widget.frames_painted
    repeated_before = widget.repeated_frames
    started = time.perf_counter()
    wait(3000)
    elapsed = time.perf_counter() - started
    timer.stop()
    gaps = [(b - a) * 1000 for a, b in zip(heartbeats, heartbeats[1:])]
    record = {
        'display': [width, height],
        'buffer': list(engine.buffer_size(width, height)),
        'radius': engine.gravity_radius,
        'elapsed_s': elapsed,
        'published_fps': len(published) / elapsed,
        'published': len(published),
        'distinct_clocks': len(set(clock for _, clock in published)),
        'painted_fps': (widget.frames_painted - before) / elapsed,
        'repeated_frames': widget.repeated_frames - repeated_before,
        'shade': summary(shade_samples),
        'paint': summary(widget.paint_samples),
        'heartbeat': summary(gaps),
    }
    widget.hide()
    record['stopped_after_hide'] = not widget.shading_thread_alive()
    widget.deleteLater()
    app.processEvents()
    return record


print(json.dumps({
    'source': str(source),
    'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'records': [probe(1920, 1080), probe(3840, 2160)],
}, indent=2), flush=True)
