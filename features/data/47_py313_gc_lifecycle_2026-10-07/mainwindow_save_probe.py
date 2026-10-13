import faulthandler
import gc
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

faulthandler.enable(all_threads=True)
from PySide6 import __version__ as pyside_version
from PySide6.QtCore import QCoreApplication, QEvent, QSettings, QTimer
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QSlider
import numpy
from spacr.qt import gc_policy, preferences as prefs
from spacr.qt.app import MainWindow
from spacr.qt.widgets import ambient

base = Path('/mnt/wd4tb/scratch/qt-py313-gc-20261007')
store = QSettings(str(base / 'mainwindow-save.ini'), QSettings.IniFormat)
prefs._settings = lambda: store
try:
    from spacr.qt.first_run import mark_tour_seen
    mark_tour_seen()
except Exception:
    pass
app = QApplication([])
assert gc_policy.install(app)
main_thread = threading.get_ident()
collections = []
def on_gc(phase, info):
    if phase == 'stop':
        collections.append((threading.get_ident(), info['generation']))
gc.callbacks.append(on_gc)
print('provenance', json.dumps({
    'python': sys.version.split()[0], 'pyside': pyside_version,
    'numpy': numpy.__version__, 'screen': [app.primaryScreen().size().width(), app.primaryScreen().size().height()],
    'app_sha256': hashlib.sha256(Path(sys.modules['spacr.qt.app'].__file__).read_bytes()).hexdigest(),
    'preferences_sha256': hashlib.sha256(Path(prefs.__file__).read_bytes()).hexdigest(),
    'ambient_sha256': hashlib.sha256(Path(ambient.__file__).read_bytes()).hexdigest(),
    'gc_policy_sha256': hashlib.sha256(Path(gc_policy.__file__).read_bytes()).hexdigest(),
}), flush=True)
prefs.set_ambient_animation('data_art_impulse_lens')
prefs.set_ambient_enabled(True)
prefs.set_ambient_resolution(2.0)
prefs.set_refresh_news(False)
window = MainWindow()
window.resize(3840, 2160)
window.show()
state = {'saved': False}

def save_dialog():
    dialog = app.activeModalWidget()
    assert dialog is not None, 'Preferences dialog did not open'
    slider = dialog.findChild(QSlider, 'AmbientResolution')
    assert slider is not None, 'Detail slider missing'
    assert slider.value() == 200, slider.value()
    slider.setValue(100)
    print('save_detail', slider.value(), flush=True)
    button = dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save)
    assert button is not None
    button.click()
    state['saved'] = True

def open_dialog():
    QTimer.singleShot(250, save_dialog)
    window.show_preferences_on()
    print('after_save', json.dumps({'stored_detail': prefs.get_ambient_resolution(),
                                    'gc_events': len(collections)}), flush=True)

QTimer.singleShot(2800, open_dialog)
started = time.monotonic()
def finish():
    status = Path('/proc/self/status').read_text()
    memory = {line.split(':')[0]: line.split(':')[1].strip() for line in status.splitlines()
              if line.startswith(('VmHWM:', 'RssAnon:'))}
    print('before_close', json.dumps({'seconds': round(time.monotonic() - started, 2),
                                      'saved': state['saved'], 'detail': prefs.get_ambient_resolution(),
                                      'gc_events': len(collections),
                                      'all_gui': all(tid == main_thread for tid, _ in collections),
                                      'memory': memory}), flush=True)
    window.close()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    living = [w for w in app.allWidgets() if isinstance(w, ambient.AmbientWidget)]
    print('after_close', json.dumps({'widgets': len(app.allWidgets()),
                                     'ambient': len(living),
                                     'workers': sum(w.shading_thread_alive() for w in living)}), flush=True)
    app.quit()
QTimer.singleShot(10500, finish)
QTimer.singleShot(16000, app.quit)
code = app.exec()
print('after_exec', json.dumps({'code': code, 'saved': state['saved'],
                                'gc_events': len(collections),
                                'all_gui': all(tid == main_thread for tid, _ in collections)}), flush=True)
gc.callbacks.remove(on_gc)
assert gc_policy.uninstall()
assert state['saved'] and prefs.get_ambient_resolution() == 1.0
assert collections and all(tid == main_thread for tid, _ in collections)
raise SystemExit(code)
