"""Observe a heartbeat strictly inside one installed native drawPath call."""
import hashlib
import json
from pathlib import Path
import resource
import sys
import threading
import time

import PySide6
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen

root = Path(__file__).resolve().parent
path = QPainterPath()
for index in range(10000):
    y = (index * 7919) % 2160
    path.moveTo(0.0, float(y))
    path.lineTo(3839.0, float((y + 357) % 2160))
image = QImage(3840, 2160, QImage.Format_RGB32)
image.fill(QColor("#101418"))
painter = QPainter(image)
painter.setRenderHint(QPainter.Antialiasing)
painter.setPen(QPen(QColor(72, 162, 180, 43), 2.5))
ticks = []
stop = threading.Event()
ready = threading.Event()


def heartbeat():
    ready.set()
    while not stop.is_set():
        ticks.append(time.perf_counter_ns())
        time.sleep(0.001)


thread = threading.Thread(target=heartbeat, name="probe-gil-heartbeat")
previous_interval = sys.getswitchinterval()
sys.setswitchinterval(1.0)
thread.start()
ready.wait()
time.sleep(0.02)
try:
    started = time.perf_counter_ns()
    painter.drawPath(path)
    finished = time.perf_counter_ns()
    control_start = time.perf_counter_ns()
    time.sleep(0.1)
    control_end = time.perf_counter_ns()
finally:
    painter.end()
    stop.set()
    thread.join(timeout=2.0)
    sys.setswitchinterval(previous_interval)
inside = [tick for tick in ticks if started + 10000000 < tick < finished - 10000000]
control = [tick for tick in ticks if control_start + 10000000 < tick < control_end - 10000000]
receipt = {
    "scope": "Single synthetic native 4K drawPath; GIL observation only, no renderer timing acceptance",
    "pyside_version": PySide6.__version__,
    "python": sys.version,
    "path_lines": 10000,
    "switch_interval_seconds": 1.0,
    "strict_interior_margin_ms": 10,
    "draw_ms": (finished - started) / 1000000,
    "heartbeat_ticks_inside_draw": len(inside),
    "heartbeat_ticks_inside_sleep_control": len(control),
    "thread_retired": not thread.is_alive(),
    "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(root / "gil_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt), flush=True)
