"""Source-bound native field parity and actual queued-input ownership probe."""
import gc
import hashlib
import importlib.util
import json
import pathlib
import resource
import subprocess
import sys
import time

import numpy as np
from PySide6.QtCore import QEvent, QEventLoop, QPoint, QPointF, Qt, QTimer
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget
from spacr.qt import preferences
from spacr.qt.widgets import ambient

ROOT = pathlib.Path.cwd()
OUT = pathlib.Path(__file__).parent
SOURCE = ROOT / 'spacr/qt/widgets/ambient.py'
assert pathlib.Path(ambient.__file__).resolve() == SOURCE.resolve()
BASE = subprocess.check_output(['git', 'show', 'ce58f867e4:spacr/qt/widgets/ambient.py'])
(OUT / 'ambient_before.py').write_bytes(BASE)
name = 'spacr.qt.widgets.ambient_before_field_grab'
spec = importlib.util.spec_from_file_location(name, OUT / 'ambient_before.py')
before = importlib.util.module_from_spec(spec)
sys.modules[name] = before
spec.loader.exec_module(before)
app = QApplication.instance() or QApplication([])
preferences._ambient_custom_colors = lambda: ('#fbba38', '#3cadfc')
for module in (before, ambient):
    module._ready_packed_scatter = lambda: None


def engine(module, palette='spacr', background='#101418', density=1):
    value = module.make_engine('data_art_impulse_lens', palette, background,
                               seed=42, blur=0, density=density, resolution=2)
    value.set_max_pixels(3840 * 2160)
    return value


def raw(image):
    assert (image.width(), image.height()) == (3840, 2160)
    values = np.frombuffer(image.constBits(), np.uint32)
    assert np.all(values >> 24 == 255)
    return bytes(image.constBits())


def wait(milliseconds):
    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()


pairs = []
for background in ('#101418', '#f4f4f0'):
    for palette, density in (('spacr', 1), ('random', 1), ('custom', 1), ('random', 3)):
        old, new = engine(before, palette, background, density), engine(ambient, palette, background, density)
        for value in (old, new):
            value.set_time(1.25)
            value.set_gravity_radius(.45)
            value.set_pointer((.6, .4))
            value._add_impulse((.45, .55))
        old_raw, new_raw = raw(old.shade(3840, 2160)), raw(new.shade(3840, 2160))
        assert old_raw == new_raw
        pairs.append({'background': background, 'palette': palette, 'density': density,
                      'sha256': hashlib.sha256(new_raw).hexdigest(), 'exact': True})
        del old, new, old_raw, new_raw

value = engine(ambient)
value.set_time(2)
idle = value.shade(3840, 2160)
value._set_field_grab(((.5, .5), (.14, -.07)))
value.advance(.5)
held = value.shade(3840, 2160)
reference = engine(ambient)
reference.set_time(value.time)
changed = int(np.count_nonzero(np.frombuffer(held.constBits(), np.uint32)
                               != np.frombuffer(reference.shade(3840, 2160).constBits(), np.uint32)))
assert changed > 10000
retained = raw(held)
value._set_field_grab(None)
assert raw(value.shade(3840, 2160)) == retained
value.advance(4)
returned = value.shade(3840, 2160)
reference.set_time(value.time)
assert raw(returned) == raw(reference.shade(3840, 2160))
assert raw(held) == retained
idle.save(str(OUT / 'field-idle-native.png'))
held.save(str(OUT / 'field-held-native.png'))
returned.save(str(OUT / 'field-returned-native.png'))
del value, reference, idle, held, returned, retained
gc.collect()

ambient.screen_pixels = lambda _: 3840 * 2160
class Screen(QWidget):
    pass
host = Screen()
host.resize(3840, 2160)
backdrop = ambient.install_ambient(host, theme='data_art_impulse_lens', palette='random',
                                  background='#101418', seed=42, gravity_radius=0,
                                  resolution=2, density=1, fps=24)
heartbeats = []
beat = QTimer()
beat.setInterval(10)
beat.timeout.connect(lambda: heartbeats.append(time.perf_counter()))
beat.start()
started = time.perf_counter()
host.show()
app.processEvents()
cold_show_ms = (time.perf_counter() - started) * 1000
wait(150)
producer = backdrop._producer_box[0]
assert producer is not None
first = producer.latest()
first_raw = raw(first)
started = time.perf_counter()
QTest.mousePress(host, Qt.LeftButton, Qt.NoModifier, QPoint(1800, 1000))
move = QMouseEvent(QEvent.MouseMove, QPointF(2100, 1100),
                   QPointF(host.mapToGlobal(QPoint(2100, 1100))), Qt.NoButton,
                   Qt.LeftButton, Qt.NoModifier)
QApplication.sendEvent(host, move)
print("input", host.size(), backdrop.size(), backdrop._field_grab, backdrop._art_input._grab_snapshot, flush=True)
input_ms = (time.perf_counter() - started) * 1000
wait(900)
with backdrop._engine_lock:
    held_offset = backdrop.engine._field_grab_offset
    print("held", backdrop._field_grab, held_offset, backdrop.engine._field_grab_held, backdrop.engine.time, flush=True)
    assert backdrop.engine._field_grab_held and np.linalg.norm(held_offset) > .01
QTest.mouseRelease(host, Qt.LeftButton, Qt.NoModifier, QPoint(2100, 1100))
assert backdrop._field_grab is None
wait(3500)
with backdrop._engine_lock:
    assert not backdrop.engine._field_grab_held
    assert backdrop.engine._field_grab_center is None
assert raw(first) == first_raw
raw(producer.latest())
frames = producer.frames_shaded
host.close()
assert not producer.is_alive()
assert not backdrop.shading_thread_alive()
assert backdrop._interaction_app is None and backdrop._field_grab is None
beat.stop()
gaps_ms = np.diff(heartbeats) * 1000
result = {'base_commit': 'ce58f867e4', 'source_before_sha256': hashlib.sha256(BASE).hexdigest(),
          'source_after_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          'import_path': ambient.__file__, 'parity_pairs': pairs, 'native_size': [3840,2160],
          'held_changed_pixels_same_clock': changed, 'release_same_clock_exact': True,
          'return_same_clock_exact': True, 'retained_frame_owned_alpha_ff': True,
          'actual_widget': {'palette': 'random', 'density':1, 'resolution':2,
                           'hover_gravity':0, 'requested_fps':24, 'cold_show_ms':cold_show_ms,
                           'press_move_delivery_ms':input_ms, 'held_offset':held_offset,
                           'heartbeat_count':len(heartbeats), 'heartbeat_p95_ms':float(np.percentile(gaps_ms,95)),
                           'heartbeat_max_ms':float(gaps_ms.max()), 'completed_frames':frames,
                           'worker_retired':not producer.is_alive()},
          'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          'limits':'Offscreen CPU fallback point scatter, no native-display/GPU/aesthetic or 24 FPS acceptance.'}
(OUT / 'native_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
