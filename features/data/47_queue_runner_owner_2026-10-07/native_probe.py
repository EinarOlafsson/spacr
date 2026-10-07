"""Exercise a cooperative queue pipeline across owner deletion."""
import hashlib
import json
import os
import resource
import tempfile
import threading
from pathlib import Path

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
os.environ['XDG_CONFIG_HOME'] = tempfile.mkdtemp(prefix='queue-owner-config-')
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication
from shiboken6 import isValid
from spacr.cancellation import checkpoint
from spacr.qt import bridge
from spacr.qt.plate_queue import PlateQueue, Status
from spacr.qt.screens import queue as module

app = QApplication([])
entered, release = threading.Event(), threading.Event()
called = []
def blocked(settings):
    called.append(settings['src'])
    entered.set()
    assert release.wait(25)
    checkpoint()
bridge.resolve_pipeline_entry = lambda _: blocked
queue = PlateQueue(path=Path(tempfile.mkdtemp())/'queue.json')
screen = module.QueueScreen(queue)
first = screen.add_item('mask', {'src': '/controlled/first'})
second = screen.add_item('mask', {'src': '/controlled/second'})
screen.start_runner()
assert entered.wait(2)
worker = screen._runner
assert worker.parent() is screen
screen.close()
assert worker.isRunning()
print(json.dumps({'checkpoint':'close_returned', 'source_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(), 'parent_is_screen':worker.parent() is screen, 'parked_threads':bridge.parked_thread_count()}), flush=True)
screen.deleteLater()
QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
assert not isValid(screen) and isValid(worker) and worker.isRunning()
print('owner_deleted_runner_alive', flush=True)
release.set()
assert worker.wait(2000)
assert bridge.prune_parked_threads() == 0
assert called == ['/controlled/first']
assert first.status == second.status == Status.QUEUED
worker.deleteLater()
QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
assert not isValid(worker)
print('cooperative_cancel_preserves_both_queue_items', flush=True)
