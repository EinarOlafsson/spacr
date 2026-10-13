import hashlib
import json
import math
import os
import resource
import statistics
import time
from pathlib import Path

from PySide6.QtCore import QEventLoop, QPoint, QTimer, Qt
from PySide6.QtGui import QCursor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget

from spacr.qt import preferences as prefs
import importlib.util
import sys
spec = importlib.util.spec_from_file_location('spacr.qt.widgets._owned_point_probe', '/mnt/wd4tb/scratch/root-evidence-20261006/direct-frame-worker/ambient-frozen.py')
ambient = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ambient
spec.loader.exec_module(ambient)
assert hashlib.sha256(Path(ambient.__file__).read_bytes()).hexdigest() == '2017f255047a9c7d29a766d98df3d2f50b2a5225348ab7d667dbb13b9be20cbe' 


OUT = Path('/mnt/wd4tb/scratch/root-evidence-20261006/direct-frame-worker/native_widget_result.json')
EXPECTED = Path('/mnt/wd4tb/scratch/root-evidence-20261006/direct-frame-worker/ambient-frozen.py')
assert Path(ambient.__file__).resolve() == EXPECTED.resolve()
assert os.environ.get('QT_QPA_PLATFORM') == 'xcb'


def wait(milliseconds):
    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()


def summary(values):
    if not values:
        return None
    ordered = sorted(values)
    return {'median_ms': statistics.median(values),
            'p95_ms': ordered[math.ceil(len(ordered) * 0.95) - 1],
            'max_ms': ordered[-1], 'samples': len(values)}


class TimedWidget(ambient.AmbientWidget):
    def __init__(self, host, theme, background, radius):
        self.paint_samples = []
        super().__init__(host, theme=theme, palette='custom',
                         background=background, seed=42, fps=24, blur=0.0,
                         speed=1.0, size=1.0, resolution=2.0, density=1.0,
                         gravity_radius=radius)

    def paintEvent(self, event):
        started = time.perf_counter()
        super().paintEvent(event)
        self.paint_samples.append((started, (time.perf_counter() - started) * 1000))


def measure(app, theme, background, radius, moving, clicking):
    host = QWidget()
    host.resize(3840, 2160)
    widget = TimedWidget(host, theme, background, radius)
    widget.follow_parent()
    host.show()
    host.activateWindow()
    app.setActiveWindow(host)
    app.processEvents()
    assert ambient.screen_pixels(widget) == 3840 * 2160
    assert widget.engine.max_pixels == 3840 * 2160
    widget.set_time(8.0)
    pointer_at = [None]
    mover = QTimer()
    mover.setInterval(70)
    mover.setTimerType(Qt.PreciseTimer)
    started_pointer = time.perf_counter()

    def move_pointer():
        elapsed = time.perf_counter() - started_pointer
        local = QPoint(round(host.width() * (0.50 + 0.18 * math.sin(elapsed * 1.7))),
                       round(host.height() * (0.50 + 0.14 * math.cos(elapsed * 1.3))))
        QCursor.setPos(host.mapToGlobal(local))
        pointer_at[0] = widget._data_art_pointer_for_tick()

    if moving:
        move_pointer()
        mover.timeout.connect(move_pointer)
        mover.start()
    else:
        QCursor.setPos(host.mapToGlobal(host.rect().center()))

    clicks = [0]
    clicker = QTimer()
    clicker.setInterval(750)

    def click():
        local = host.mapFromGlobal(QCursor.pos())
        QTest.mouseClick(host, Qt.LeftButton, pos=local)
        clicks[0] += 1

    if clicking:
        clicker.timeout.connect(click)
        clicker.start()

    engine = widget._engine
    original_shade = engine.shade
    shades = []
    publications = []
    last_clock = [engine.time]

    def timed_shade(width, height):
        clock = engine.time
        started = time.perf_counter()
        image = original_shade(width, height)
        shades.append((started, (time.perf_counter() - started) * 1000, clock))
        last_clock[0] = clock
        return image

    with widget._engine_lock:
        engine.shade = timed_shade
    producer = widget._producer_box[0]
    original_publish = producer.publish

    def timed_publish(image):
        original_publish(image)
        publications.append((time.perf_counter(), last_clock[0]))

    producer.publish = timed_publish
    beats = []
    heart = QTimer()
    heart.setTimerType(Qt.PreciseTimer)
    heart.setInterval(10)
    heart.timeout.connect(lambda: beats.append(time.perf_counter()))
    heart.start()
    wait(2500)
    assert widget.shading_thread_alive()
    if moving:
        assert pointer_at[0] is not None, (theme, background, radius)
    shades.clear()
    publications.clear()
    beats.clear()
    widget.paint_samples.clear()
    first_clock = engine.time
    first_painted = widget.frames_painted
    first_repeated = widget.repeated_frames
    started = time.perf_counter()
    wait(3500)
    elapsed = time.perf_counter() - started
    heart.stop()
    mover.stop()
    clicker.stop()
    shade_copy = list(shades)
    published_copy = list(publications)
    paints = list(widget.paint_samples)
    gaps = [(right - left) * 1000 for left, right in zip(beats, beats[1:])]
    intervals = [(right[0] - left[0]) * 1000
                 for left, right in zip(published_copy, published_copy[1:])]
    result = {'theme': theme, 'background': background, 'radius': radius,
              'moving_pointer': moving, 'clicking': clicking,
              'viewport': [widget.width(), widget.height()],
              'screen_pixels': ambient.screen_pixels(widget),
              'buffer': list(engine.buffer_size(widget.width(), widget.height())),
              'max_pixels': engine.max_pixels,
              'requested_fps': 24, 'active_rate_cap': widget._rate(),
              'steady_seconds': elapsed, 'published_frames': len(published_copy),
              'published_fps': len(published_copy) / elapsed,
              'distinct_published_clocks': len({clock for _, clock in published_copy}),
              'clock_advance': engine.time - first_clock,
              'painted_frames': widget.frames_painted - first_painted,
              'painted_fps': (widget.frames_painted - first_painted) / elapsed,
              'repeated_frame_objects': widget.repeated_frames - first_repeated,
              'worker_shade': summary([sample[1] for sample in shade_copy]),
              'gui_paint': summary([sample[1] for sample in paints]),
              'heartbeat_10ms': summary(gaps),
              'publication_intervals': summary(intervals),
              'click_events': clicks[0], 'last_local_pointer': pointer_at[0],
              'active_impulses': len(getattr(engine, '_gravity_impulses', [])),
              'pending_clicks': len(widget._pending_art_impulses)}
    image = widget.grab().toImage()
    result['final_grab_sha256'] = hashlib.sha256(image.bits().tobytes()).hexdigest()
    host.hide()
    result['worker_stopped_after_hide'] = not widget.shading_thread_alive()
    widget.deleteLater()
    host.deleteLater()
    app.processEvents()
    return result


def main():
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    store = prefs.QSettings(str(OUT.parent / 'probe-prefs.ini'), prefs.QSettings.IniFormat)
    prefs._settings = lambda: store
    prefs._set_ambient_custom_colors(('#3b82f6', '#ff00ff'))
    cases = (
        ('data_art_impulse_lens', '#101418', 0.0, False, False),
        ('data_art_impulse_lens', '#101418', 0.25, True, True),
        ('data_art_point_atlas', '#101418', 0.0, False, False),
        ('data_art_point_atlas', '#101418', 0.25, True, False),
        ('data_art_genetic_advection', '#101418', 0.0, False, False),
        ('data_art_genetic_advection', '#101418', 0.25, True, False),
        ('data_art_impulse_lens', '#f6f7f9', 0.25, True, True),
        ('data_art_point_atlas', '#f6f7f9', 0.25, True, False),
        ('data_art_genetic_advection', '#f6f7f9', 0.25, True, False),
    )
    records = []
    for case in cases:
        result = measure(app, *case)
        records.append(result)
        print(json.dumps(result), flush=True)
        OUT.write_text(json.dumps({
            'source_sha256': hashlib.sha256(EXPECTED.read_bytes()).hexdigest(),
            'source_path': str(EXPECTED), 'qt_platform': os.environ.get('QT_QPA_PLATFORM'),
            'scope': 'real AmbientWidget and FrameProducer on Xvfb 4K, 10 ms GUI heartbeat; end grab outside timing',
            'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            'records': records}, indent=2) + '\n')


if __name__ == '__main__':
    main()
