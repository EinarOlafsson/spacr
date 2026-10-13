"""Hold a real image loader across screen close and native destruction."""
import hashlib
import json
import resource
import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication
from shiboken6 import isValid

from spacr.qt import bridge
from spacr.qt.screens import make_masks as module

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
app = QApplication([])
entered, release = threading.Event(), threading.Event()
def blocked(*args, **kwargs):
    entered.set()
    assert release.wait(25)
    return np.ones((8, 8), dtype=np.uint16), np.zeros((8, 8), dtype=np.uint16)
module.engine.load_image_and_mask = blocked
screen = module.MakeMasksScreen()
screen._start_background_load('/controlled', 'image.tif', screen._load_token)
assert entered.wait(2)
worker = screen._load_worker
assert worker.parent() is screen
screen.close()
assert worker.isRunning()
print(json.dumps({'checkpoint': 'close_returned', 'module_path': module.__file__,
                  'source_sha256': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                  'parent_is_screen': worker.parent() is screen,
                  'parked_threads': bridge.parked_thread_count()}), flush=True)
screen.deleteLater()
QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
assert not isValid(screen) and isValid(worker) and worker.isRunning()
print('screen_deleted_loader_still_running', flush=True)
release.set()
assert worker.wait(2000)
assert bridge.prune_parked_threads() == 0
worker.deleteLater()
QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
assert not isValid(worker)
print('loader_stopped_cleanly', flush=True)
