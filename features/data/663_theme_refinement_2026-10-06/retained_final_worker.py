import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path('/mnt/wd4tb/scratch/theme-refinement-20261006/retained_final')
REPO = Path('/mnt/wd4tb/spacr-worktrees/codex-theme-clock-20261006')
sys.path.insert(0, str(REPO))
from PySide6.QtCore import QEventLoop, QPoint, QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
import spacr.qt.widgets

spec = importlib.util.spec_from_file_location('spacr.qt.widgets._final_worker_review', ROOT / 'ambient.py')
ambient = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ambient
spec.loader.exec_module(ambient)
app = QApplication.instance() or QApplication([])
app.setQuitOnLastWindowClosed(False)
families = ('point_atlas', 'tissue_facets', 'chromatin_ribbon', 'genetic_advection',
            'impulse_lens', 'fungal_growth')
records = []


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


class TimedWidget(ambient.AmbientWidget):
    def __init__(self, family):
        self.paint_samples = []
        self.pointer_epoch = time.perf_counter()
        self.family = family
        super().__init__(theme='data_art_' + family, palette='spacr', background='#101418',
                         fps=24, seed=42, blur=0, speed=1, size=1,
                         resolution=2, density=1, direction=ambient.DEFAULT_DRIFT_DIRECTION)

    def _follow_screen(self):
        with self._engine_lock:
            self._engine.set_max_pixels(max(1, self.width() * self.height()))

    def _data_art_pointer_for_tick(self):
        if self.family != 'impulse_lens':
            return (0.5, 0.5)
        elapsed = time.perf_counter() - self.pointer_epoch
        return (0.5 + 0.12 * math.sin(elapsed * 0.6),
                0.5 + 0.09 * math.cos(elapsed * 0.8))

    def paintEvent(self, event):
        started = time.perf_counter()
        super().paintEvent(event)
        self.paint_samples.append((started, (time.perf_counter() - started) * 1000))


for family in families:
    for width, height in ((1920, 1080), (3840, 2160)):
        widget = TimedWidget(family)
        widget.resize(width, height)
        widget.show()
        app.setActiveWindow(widget)
        app.processEvents()
        widget.set_time(29.0 if family == 'fungal_growth' else 8.0)
        engine = widget._engine
        original_shade = engine.shade
        shade_records = []
        published = []
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
        heartbeat_times = []
        heartbeat = QTimer()
        heartbeat.setTimerType(Qt.PreciseTimer)
        heartbeat.setInterval(10)
        heartbeat.timeout.connect(lambda: heartbeat_times.append(time.perf_counter()))
        heartbeat.start()
        wait(600)
        shade_records.clear()
        published.clear()
        heartbeat_times.clear()
        widget.paint_samples.clear()
        start_painted = widget.frames_painted
        start_repeated = widget.repeated_frames
        started = time.perf_counter()
        if family == 'impulse_lens':
            QTimer.singleShot(1000, lambda: QTest.mouseClick(widget, Qt.LeftButton,
                                                            pos=QPoint(width // 2, height // 2)))
        wait(3000)
        finished = time.perf_counter()
        heartbeat.stop()
        elapsed = finished - started
        shade_snapshot = list(shade_records)
        published_snapshot = list(published)
        paints = list(widget.paint_samples)
        painted_count = widget.frames_painted - start_painted
        repeated_count = widget.repeated_frames - start_repeated
        heartbeat_gaps = [(b - a) * 1000 for a, b in zip(heartbeat_times, heartbeat_times[1:])]
        publish_gaps = [(b[0] - a[0]) * 1000 for a, b in zip(published_snapshot, published_snapshot[1:])]
        record = {'family': family, 'display': [width, height],
                  'buffer': list(engine.buffer_size(width, height)), 'max_pixels': engine.max_pixels,
                  'requested_fps': 24, 'actual_widget_rate_cap': widget._rate(),
                  'elapsed_seconds': elapsed, 'published_frames': len(published_snapshot),
                  'published_fps': len(published_snapshot) / elapsed,
                  'distinct_published_animation_clocks': len({entry[1] for entry in published_snapshot}),
                  'painted_frames': painted_count, 'painted_fps': painted_count / elapsed,
                  'repeated_frame_objects': repeated_count,
                  'worker_shade_copy': summary([entry[1] for entry in shade_snapshot]),
                  'gui_paint': summary([entry[1] for entry in paints]),
                  'gui_10ms_heartbeat': summary(heartbeat_gaps),
                  'publication_intervals': summary(publish_gaps),
                  'load_average': list(os.getloadavg()),
                  'lens_pending_clicks_at_end': len(widget._pending_art_impulses),
                  'lens_active_impulses_at_end': len(getattr(engine, '_gravity_impulses', []))}
        widget.hide()
        stopped = not widget.shading_thread_alive()
        record['worker_stopped_after_hide'] = stopped
        widget.deleteLater()
        app.processEvents()
        records.append(record)
        (ROOT / 'worker_gui_perf.json').write_text(json.dumps({
            'renderer_sha256': hashlib.sha256((ROOT / 'ambient.py').read_bytes()).hexdigest(),
            'imported_renderer': str(ambient.__file__), 'qt_platform': os.environ.get('QT_QPA_PLATFORM'),
            'scope': 'real AmbientWidget timer + FrameProducer + offscreen QWidget paints; source unchanged; synthetic local pointer and QTest click',
            'screen_budget_override': 'Offscreen QScreen size replaced by explicit native viewport pixel area only for this measurement',
            'records': records}, indent=2))
        print(json.dumps(record), flush=True)
