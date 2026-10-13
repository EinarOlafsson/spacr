"""Observe real small-mask QThread ownership under the installed GUI GC cadence."""
from __future__ import annotations

import gc
import json
import os
import tempfile
import weakref
from pathlib import Path

import numpy as np
import shiboken6
import tifffile
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from spacr.qt import gc_policy
from spacr.qt.widgets.primary_mask_selector import PrimaryMaskSelector


def emit(**record):
    print(json.dumps(record, sort_keys=True), flush=True)


def until(predicate, timeout_ms=10000):
    loop = QEventLoop()
    pulse = QTimer()
    pulse.setInterval(5)
    deadline = QTimer()
    deadline.setSingleShot(True)
    state = {"expired": False}

    def check():
        if predicate():
            loop.quit()

    def expire():
        state["expired"] = True
        loop.quit()

    pulse.timeout.connect(check)
    deadline.timeout.connect(expire)
    pulse.start()
    deadline.start(timeout_ms)
    check()
    if not predicate():
        loop.exec()
    pulse.stop()
    deadline.stop()
    if state["expired"] or not predicate():
        raise TimeoutError("small source did not finish")


def pause(milliseconds):
    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()


def main():
    app = QApplication([])
    assert gc_policy.install(app)
    emit(event="start", pid=os.getpid(), gui_gc=gc_policy.is_installed(),
         gc_enabled=gc.isenabled(), gc_threshold=gc.get_threshold())
    destroyed = []
    refs = []
    with tempfile.TemporaryDirectory(prefix="spacr-small-source-native-") as base:
        root = Path(base)
        primary = root / "primary"
        primary.mkdir()
        for name, label in (("one", 7), ("two", 900)):
            tifffile.imwrite(primary / f"{name}.tif",
                             np.full((32, 32), label, dtype=np.uint16))
        for cycle in range(6):
            selector = PrimaryMaskSelector()
            selector.path.setText(str(primary))
            for name, label in (("one", 7), ("two", 900)):
                selector.bind_field(root / f"{name}.tif", (32, 32),
                                    root / "output" / f"{name}.tif")
                worker = selector._worker
                assert worker is not None
                native = shiboken6.getCppPointer(worker)[0]
                token = f"{cycle}:{name}"
                worker.destroyed.connect(
                    lambda _object=None, key=token, pointer=native:
                    (destroyed.append(key), emit(event="destroyed", key=key,
                                                  native=hex(pointer))))
                refs.append((token, weakref.ref(worker), native))
                emit(event="started", key=token, native=hex(native),
                     object_name=worker.objectName(), running=worker.isRunning())
                del worker
                until(lambda: selector.snapshot is not None)
                assert int(selector.snapshot.labels[0, 0]) == label
                until(lambda: token in destroyed)
                emit(event="published", key=token,
                     worker_ref_alive=refs[-1][1]() is not None,
                     gc_count=gc.get_count())
            selector.shutdown()
            selector.close()
            selector.deleteLater()
            del selector
            pause(100)
        pause(2200)
    emit(event="end", started=len(refs), destroyed=len(destroyed),
         live_wrappers=sum(ref() is not None for _, ref, _ in refs),
         gc_count=gc.get_count(), gc_enabled=gc.isenabled())
    gc_policy.uninstall()
    app.quit()


if __name__ == "__main__":
    main()
