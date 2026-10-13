"""Actual frozen renderer workers and offscreen GUI paints, baseline versus JIT."""

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

ROOT = Path('/mnt/wd4tb/scratch/theme-refinement-20261006/visual-revision/wave-flow-worker')
sys.path.insert(0, '/mnt/wd4tb/spacr-worktrees/codex-theme-clock-20261006')
from PySide6.QtCore import QEventLoop, QPoint, QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
import spacr.qt.widgets

modules = []
for label in ('before', 'after'):
    path = ROOT / (label + '.py')
    spec = importlib.util.spec_from_file_location('spacr.qt.widgets._numba_worker_' + label, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    modules.append(module)
compile_receipt = {}
_original_warm = modules[1]._SATIN_COMPILER._warm
def _record_warm():
    import threading
    started = time.perf_counter()
    _original_warm()
    compile_receipt.update(wall_ms=(time.perf_counter() - started) * 1000, thread=threading.current_thread().name, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
modules[1]._SATIN_COMPILER._warm = _record_warm
app = QApplication.instance() or QApplication([])
app.setQuitOnLastWindowClosed(False)


def wait(milliseconds):
    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()


def summary(values):
    if not values:
        return None
    return {'median_ms': statistics.median(values),
            'p95_ms': sorted(values)[math.ceil(len(values) * 0.95) - 1],
            'max_ms': max(values), 'samples': len(values)}


def measure(module, family, width, height, cold=False):
    class TimedWidget(module.AmbientWidget):
        def __init__(self):
            self.paint_samples = []
            self.pointer_epoch = time.perf_counter()
            super().__init__(theme='data_art_' + family, palette='spacr', background='#101418',
                             fps=24, seed=42, blur=0, speed=1, size=1,
                             resolution=2, density=1, direction=module.DEFAULT_DRIFT_DIRECTION)

        def _follow_screen(self):
            with self._engine_lock:
                self._engine.set_max_pixels(max(1, self.width() * self.height()))

        def _data_art_pointer_for_tick(self):
            if family != 'impulse_lens':
                return (0.5, 0.5)
            elapsed = time.perf_counter() - self.pointer_epoch
            return (0.5 + 0.12 * math.sin(elapsed * 0.6),
                    0.5 + 0.09 * math.cos(elapsed * 0.8))

        def paintEvent(self, event):
            started = time.perf_counter()
            super().paintEvent(event)
            self.paint_samples.append((started, (time.perf_counter() - started) * 1000))

    heartbeat_times = []
    heartbeat = QTimer()
    heartbeat.setTimerType(Qt.PreciseTimer)
    heartbeat.setInterval(10)
    heartbeat.timeout.connect(lambda: heartbeat_times.append(time.perf_counter()))
    heartbeat.start()
    epoch = time.perf_counter()
    widget = TimedWidget()
    widget.resize(width, height)
    widget.show()
    app.setActiveWindow(widget)
    app.processEvents()
    show_ms = (time.perf_counter() - epoch) * 1000
    if cold:
        wait(2200)
        deadline = time.perf_counter() + 10
        while (module is modules[1] and module._SATIN_COMPILER.kernel is None
               and not module._SATIN_COMPILER.failed and time.perf_counter() < deadline):
            wait(50)
        gaps = [(b - a) * 1000 for a, b in zip(heartbeat_times, heartbeat_times[1:])]
        receipt = {'scope': 'first native4K widget show, background CPU compile + exact NumPy fallback',
                   'widget_show_ms': show_ms, 'heartbeat': summary(gaps),
                   'compiler': dict(compile_receipt),
                   'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   'ready_after_probe': modules[1]._SATIN_COMPILER.kernel is not None,
                   'baseline': module is modules[0]}
        widget.hide()
        receipt['worker_stopped_after_hide'] = not widget.shading_thread_alive()
        widget.deleteLater()
        app.processEvents()
        heartbeat.stop()
        return receipt
    if hasattr(widget, 'set_gravity_radius'):
        widget.set_gravity_radius(1.0)
    widget.set_time(8.0)
    engine = widget._engine
    original_shade = engine.shade
    shade_records, published = [], []
    last_shade_clock = [engine.time]

    def timed_shade(w, h):
        clock = engine.time
        started = time.perf_counter()
        image = original_shade(w, h)
        shade_records.append((started, (time.perf_counter() - started) * 1000, clock))
        last_shade_clock[0] = clock
        return image

    with widget._engine_lock:
        engine.shade = timed_shade
    producer = widget._producer_box[0]
    original_publish = producer.publish

    def timed_publish(image):
        original_publish(image)
        published.append((time.perf_counter(), last_shade_clock[0]))

    producer.publish = timed_publish
    wait(900)
    shade_records.clear()
    published.clear()
    heartbeat_times.clear()
    widget.paint_samples.clear()
    start_painted, start_repeated = widget.frames_painted, widget.repeated_frames
    started = time.perf_counter()
    if family == 'impulse_lens':
        QTimer.singleShot(1000, lambda: QTest.mouseClick(widget, Qt.LeftButton,
                                                       pos=QPoint(width // 2, height // 2)))
    wait(3500)
    elapsed = time.perf_counter() - started
    heartbeat.stop()
    records = list(shade_records)
    frames = list(published)
    paints = list(widget.paint_samples)
    gaps = [(b - a) * 1000 for a, b in zip(heartbeat_times, heartbeat_times[1:])]
    intervals = [(b[0] - a[0]) * 1000 for a, b in zip(frames, frames[1:])]
    receipt = {'family': family, 'display': [width, height],
               'buffer': list(engine.buffer_size(width, height)), 'max_pixels': engine.max_pixels,
               'requested_fps': 24, 'actual_widget_rate_cap': widget._rate(),
               'elapsed_seconds': elapsed, 'published_frames': len(frames),
               'published_fps': len(frames) / elapsed,
               'distinct_published_animation_clocks': len({entry[1] for entry in frames}),
               'painted_frames': widget.frames_painted - start_painted,
               'painted_fps': (widget.frames_painted - start_painted) / elapsed,
               'repeated_frame_objects': widget.repeated_frames - start_repeated,
               'worker_shade_copy': summary([entry[1] for entry in records]),
               'gui_paint': summary([entry[1] for entry in paints]),
               'gui_10ms_heartbeat': summary(gaps), 'publication_intervals': summary(intervals),
               'load_average': list(os.getloadavg()),
               'lens_active_impulses_at_end': len(getattr(engine, '_gravity_impulses', []))}
    widget.hide()
    receipt['worker_stopped_after_hide'] = not widget.shading_thread_alive()
    widget.deleteLater()
    app.processEvents()
    return receipt


cold = []
for module in (modules[0], modules[1]):
    cold.append(measure(module, 'chromatin_ribbon', 3840, 2160, cold=True))
    print(json.dumps({'cold': cold[-1]}), flush=True)
assert modules[1]._SATIN_COMPILER.kernel is not None
records = []
for family in ('chromatin_ribbon', 'genetic_advection'):
    for width, height in ((1920, 1080), (3840, 2160)):
        order = (0, 1) if family != 'impulse_lens' else (1, 0)
        for slot in order:
            result = measure(modules[slot], family, width, height)
            result['mode'] = 'retained_rigid_folds_flow' if slot == 0 else 'travelling_waves_vortices'
            records.append(result)
            print(json.dumps(result), flush=True)
            (ROOT / 'worker_gui_perf.json').write_text(json.dumps({
                'renderer_sha256': hashlib.sha256((ROOT / 'before.py').read_bytes()).hexdigest(),
                'production_renderer_sha256': hashlib.sha256(Path(modules[1].__file__).read_bytes()).hexdigest(),
                'scope': 'actual timers/FrameProducer/offscreen GUI; native area override; enabled local pointer, satin waves and changing advection',
                'cold': cold, 'records': records,
                'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}, indent=2) + '\n')
