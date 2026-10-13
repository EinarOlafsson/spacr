"""Spawned mask plots traverse the real analysis thread and GUI figure signal."""

import threading
from pathlib import Path

from PySide6.QtCore import QObject, Slot
from PySide6.QtGui import QImage

from spacr.qt import bridge
from tests.test_mask_worker_figures import _run


def test_spawned_mask_figures_reach_the_gui_receiver(qtbot, tmp_path):
    received, errors, ended = [], [], []
    gui_thread = threading.get_ident()

    class Receiver(QObject):
        @Slot(object, str)
        def receive(self, figure, path):
            picture = QImage(path)
            received.append((threading.get_ident(), figure.axes[0].get_title(),
                             not picture.isNull()))
            Path(path).unlink(missing_ok=True)

    receiver = Receiver()
    thread, worker = bridge.make_thread(lambda settings: _run(tmp_path), {},
                                        app_key='mask', journal=False)
    worker.figure_ready.connect(receiver.receive)
    worker.error.connect(errors.append)
    worker.finished.connect(ended.append)
    try:
        thread.start()
        qtbot.waitUntil(lambda: bool(ended), timeout=45000)
        qtbot.waitUntil(lambda: bridge.thread_has_stopped(thread), timeout=5000)
        qtbot.waitUntil(lambda: len(received) == 2 or bool(errors), timeout=5000)
        assert not errors
        assert ended == [True]
        assert len(received) == 2
        assert all(thread_id == gui_thread and image_ok
                   for thread_id, _, image_ok in received)
        assert [title.rsplit(' ', 1)[-1] for _, title, _ in received] == ['0', '1']
    finally:
        if not bridge.thread_has_stopped(thread):
            worker.cancel_token.cancel()
            thread.quit()
            thread.wait(30000)
