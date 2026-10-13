from pathlib import Path
import threading
import tempfile
import hashlib
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from spacr.qt import preferences
from spacr.qt.widgets.ambient import AmbientWidget

app = QApplication.instance() or QApplication([])
store = Path(tempfile.mkdtemp()) / 'qt.ini'
preferences._settings = lambda: QSettings(str(store), QSettings.IniFormat)
preferences._SAFE_MODE = False
preferences.set_ambient_animation('blobs')
widget = AmbientWidget(theme='blobs', palette='spacr', seed=17)
widget.resize(640, 480)
widget.show()
for _ in range(20):
    QTest.qWait(10)
widget._timer.stop()
ready = threading.Event()
release = threading.Event()
def hold():
    with widget._engine_lock:
        ready.set()
        release.wait(5)
thread = threading.Thread(target=hold)
thread.start()
assert ready.wait(1)
before = widget.time()
widget._clock.start()
QTest.qWait(30)
widget._on_tick()
assert widget.time() == before
release.set()
thread.join(1)
for _ in range(50):
    QTest.qWait(10)
after = widget.time()
print({'source': str(Path.cwd()), 'ambient_sha256': hashlib.sha256((Path.cwd() / 'spacr/qt/widgets/ambient.py').read_bytes()).hexdigest(), 'before': before, 'after': after, 'GUI_timer_running': widget._timer.isActive()}, flush=True)
widget.set_animating(False)
widget.close()
assert after > before, 'Blobs shaded frames without advancing after a contended GUI tick'
