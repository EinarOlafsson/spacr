"""Controlled blocked-reader lifetime probe; disable local core files."""
import hashlib
import json
import resource
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication
from shiboken6 import isValid

from spacr.qt import bridge
from spacr.qt.widgets import primary_mask_selector as module

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
app = QApplication([])
entered, release = threading.Event(), threading.Event()
def blocked(**request):
    entered.set()
    assert release.wait(20)
    return SimpleNamespace(labels=np.ones((10, 10), dtype=np.uint16))
module.read_primary_source = blocked
selector = module.PrimaryMaskSelector()
selector.path.setText('/controlled/source')
selector.bind_field('/controlled/image.tif', (10, 10), '/controlled/output.tif')
assert entered.wait(2)
worker = selector._worker
selector.shutdown()
assert worker.isRunning()
print(json.dumps({'checkpoint':'shutdown_returned',
                  'module_path':module.__file__,
                  'source_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                  'parent_is_selector':worker.parent() is selector,
                  'parked_threads':bridge.parked_thread_count()}), flush=True)
selector.deleteLater()
QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
assert not isValid(selector)
print('selector_deleted_reader_still_running', flush=True)
release.set()
assert worker.wait(2000)
assert bridge.prune_parked_threads() == 0
worker.deleteLater()
QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
assert not isValid(worker)
print('reader_stopped_cleanly', flush=True)
