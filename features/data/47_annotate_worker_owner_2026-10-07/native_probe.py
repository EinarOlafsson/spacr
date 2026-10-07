"""Hold an actual Annotate retrain QThread across its existing drain deadline."""
import hashlib
import json
import resource
import threading
from pathlib import Path

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication
from shiboken6 import isValid

from spacr.qt import bridge
from spacr.qt.screens import annotate as module

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
app = QApplication([])
entered, release = threading.Event(), threading.Event()
def blocked(self):
    entered.set()
    assert release.wait(40)
module._RetrainWorker.run = blocked
screen = module.AnnotateScreen()
worker = module._RetrainWorker('/controlled/no-model.db', 'annotate', {}, parent=screen)
screen._retrain_worker = worker
worker.start()
assert entered.wait(2)
screen.close()
assert worker.isRunning()
print(json.dumps({'checkpoint':'close_returned', 'timeout_ms':module.CLOSE_DRAIN_MS,
                  'parent_is_screen':worker.parent() is screen,
                  'parked_threads':bridge.parked_thread_count(),
                  'source_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}), flush=True)
screen.deleteLater()
QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
assert not isValid(screen) and isValid(worker) and worker.isRunning()
print('annotate_deleted_retrain_still_running', flush=True)
release.set()
assert worker.wait(2000)
assert bridge.prune_parked_threads() == 0
worker.deleteLater()
QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
assert not isValid(worker)
print('retrain_stopped_cleanly', flush=True)
