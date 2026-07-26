"""Stress harness for spacr.qt.bridge.make_thread — NOT collected by pytest.

It reproduces the segfault that came from double-owning the worker: Python
holds a reference AND `deleteLater` schedules a C++ delete, and the two race
when the deferred delete is flushed by the worker thread while the GUI thread
drops the last reference. gdb located it in
`QThread -> sendPostedEvents -> ~QObject -> Sbk_GetPyOverride`.

Measured on this machine, 800 jobs per run:

    worker.finished -> worker.deleteLater   3 crashes in 8   (original)
    thread.finished -> worker.deleteLater   2 crashes in 20  (not a fix:
                                            the worker's affinity is still the
                                            worker thread, so the delete still
                                            defers into a stopped loop)
    no worker.deleteLater at all            0 crashes in 20  (shipped)

It lives here rather than in the suite because it is a probabilistic crash
test: a single clean run proves nothing, and a crash takes the interpreter
down rather than failing an assertion. Run it in a loop when touching the
threading idiom:

    for i in $(seq 1 20); do
        QT_QPA_PLATFORM=offscreen STRESS_N=800 python tests/manual/qt_thread_stress.py \
            >/dev/null 2>&1 || echo "run $i CRASHED"
    done

Pass `disconnect` to simulate the per-screen workaround.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QObject, Signal, QTimer
from PySide6.QtWidgets import QApplication, QWidget

from spacr.qt.bridge import make_thread

DISCONNECT = "disconnect" in sys.argv
N = int(os.environ.get("STRESS_N", "300"))


class Host(QWidget):
    settled = Signal(int, bool)
    retired = Signal(int)

    def __init__(self):
        super().__init__()
        self.jobs = {}
        self.n_settled = 0
        self.n_retired = 0
        self.settled.connect(self._on_settled)
        self.retired.connect(self._on_retired)

    def start(self, jid):
        box = {}
        thread, worker = make_thread(lambda payload: payload.setdefault("x", 1), box)
        if DISCONNECT:
            worker.finished.disconnect(worker.deleteLater)
        self.jobs[jid] = (thread, worker)
        worker.finished.connect(lambda ok, j=jid: self.settled.emit(j, bool(ok)))
        thread.finished.connect(lambda j=jid: self.retired.emit(j))
        thread.start()

    def _on_settled(self, jid, ok):
        self.n_settled += 1

    def _on_retired(self, jid):
        self.jobs.pop(jid, None)
        self.n_retired += 1


app = QApplication([])
h = Host()
for i in range(N):
    h.start(i)
    loop = QEventLoop()
    QTimer.singleShot(0, loop.quit)
    loop.exec()

deadline = 20000
while h.n_retired < N and deadline > 0:
    loop = QEventLoop()
    QTimer.singleShot(5, loop.quit)
    loop.exec()
    deadline -= 5
print(f"settled={h.n_settled} retired={h.n_retired} jobs_left={len(h.jobs)} "
      f"disconnect={DISCONNECT}")
