import faulthandler
import gc
import hashlib
import json
import sys
import threading
import time
import weakref
from pathlib import Path

faulthandler.enable(all_threads=True)
from PySide6 import __version__ as pyside_version
from PySide6.QtCore import QEvent, QTimer, QCoreApplication
from PySide6.QtWidgets import QApplication, QWidget
import numpy
from spacr.qt import gc_policy
from spacr.qt.widgets import ambient

root = Path(ambient.__file__).parents[3]
print('provenance', json.dumps({
    'python': sys.version.split()[0], 'pyside': pyside_version,
    'numpy': numpy.__version__, 'ambient_sha256': hashlib.sha256(Path(ambient.__file__).read_bytes()).hexdigest(),
    'gc_policy_sha256': hashlib.sha256(Path(gc_policy.__file__).read_bytes()).hexdigest(),
    'source_root': str(root), 'timer_ms': gc_policy.INTERVAL_MS,
}), flush=True)
app = QApplication([])
print('screen', app.primaryScreen().size().width(), app.primaryScreen().size().height(), flush=True)
assert gc_policy.install(app)
main_thread = threading.get_ident()
collections = []
destroyed_threads = []
weak = []
def on_gc(phase, info):
    if phase == 'stop':
        collections.append((threading.get_ident(), info['generation'], info['collected']))
gc.callbacks.append(on_gc)
window = QWidget()
window.resize(3840, 2160)
backdrop = ambient.AmbientWidget(window, theme='data_art_impulse_lens', palette='spacr',
                                 blur=0.0, speed=1.0, size=1.0,
                                 resolution=2.0, density=1.0, direction='up',
                                 gravity_radius=0.65)
backdrop.resize(window.size())
window.show()

def one_cycle():
    panel = QWidget()
    panel._cycle = [panel]
    child_timer = QTimer(panel)
    child_timer.start(50000)
    panel.destroyed.connect(lambda *_: destroyed_threads.append(threading.get_ident()))
    weak.append(weakref.ref(panel))

def churn():
    for _ in range(125):
        one_cycle()

churn_timer = QTimer()
churn_timer.setInterval(500)
churn_timer.timeout.connect(churn)
churn_timer.start()

def report(stage):
    status = Path('/proc/self/status').read_text()
    memory = {line.split(':')[0]: line.split(':')[1].strip() for line in status.splitlines()
              if line.startswith(('VmHWM:', 'RssAnon:'))}
    print(stage, json.dumps({
        'seconds': round(time.monotonic() - started, 2),
        'gc_events': len(collections),
        'gc_generations': [v[1] for v in collections],
        'gc_all_gui': all(v[0] == main_thread for v in collections),
        'destroyed': len(destroyed_threads),
        'destroyed_all_gui': all(tid == main_thread for tid in destroyed_threads),
        'weak_alive': sum(ref() is not None for ref in weak),
        'frames_shaded': backdrop.frames_shaded(),
        'worker_alive': backdrop.shading_thread_alive(),
        'memory': memory,
    }), flush=True)

started = time.monotonic()
QTimer.singleShot(10500, lambda: report('before_close'))
def close():
    churn_timer.stop()
    window.close()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    report('after_close')
    app.quit()
QTimer.singleShot(11500, close)
code = app.exec()
report('after_exec')
gc.callbacks.remove(on_gc)
assert gc_policy.uninstall()
assert len(collections) >= 2, 'No natural Qt-timer GC was observed'
assert all(tid == main_thread for tid, _, _ in collections)
assert len(destroyed_threads) > 0 and all(tid == main_thread for tid in destroyed_threads)
assert not backdrop.shading_thread_alive()
raise SystemExit(code)
