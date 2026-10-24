"""The recorder must never hold the GUI thread: a 2 s pointer move keeps the application painting at 50 Hz or more.

The human pointer is another process; the driver waits for its reply in ``QTestAdapter.call``, which pumps Qt events
in 4 ms slices. If that wait ever slept or spun without processing events, the application could not paint and the
clip gate would count an app freeze that is not the application's.
"""
import time

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture_host_adapter import QTestAdapter  # noqa: E402


class SlowHuman:
    """A pointer that takes ``seconds`` to answer a move, as a gliding human does."""

    def __init__(self, seconds):
        self.seconds = seconds
        self.ready_at = None
        self.ops = []

    def submit(self, op, *args, **kwargs):
        self.ops.append(op)
        self.ready_at = time.monotonic() + self.seconds

    def reply_ready(self, timeout=0.0):
        if timeout:
            time.sleep(timeout)
        return self.ready_at is not None and time.monotonic() >= self.ready_at

    def collect(self, op='request'):
        self.ready_at = None
        return [0.0, 0.0]


def test_a_two_second_pointer_move_keeps_the_gui_painting_at_fifty_hertz():
    app = QApplication.instance() or QApplication([])
    adapter = QTestAdapter(SlowHuman(2.0))
    ticks = []
    timer = QTimer()
    timer.setInterval(5)
    timer.timeout.connect(lambda: ticks.append(time.monotonic()))
    timer.start()
    started = time.monotonic()
    adapter.call('move_to', 100, 100)
    elapsed = time.monotonic() - started
    timer.stop()
    assert elapsed >= 1.9
    rate = len(ticks) / elapsed
    gaps = [b - a for a, b in zip(ticks, ticks[1:])]
    assert rate >= 50, f"the GUI thread ticked only {rate:.0f} times a second during a pointer move"
    assert max(gaps) < 0.1, f"the GUI thread stood still for {max(gaps):.2f} s inside the pointer wait"
