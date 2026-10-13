"""The spaCR field keeps moving while a run goes.

Lessons 07 (clip org_organize_popup) and 08 (clip batch_20_batch_settings)
failed the capture host's smoothness gate because the field stopped
publishing frames while a worker ran, although the GUI thread stayed free.
Measured offscreen on 2026-10-13 at 2560x1440 with the default field:

- any job holding the responsive-GUI claim (every JobRunner job, such as the
  Organize popup's "Sort by regex", user-visible or not) paced the field to
  ``_RUN_FPS`` = 4 fps, and a tick only every 250 ms noticed the job had
  ended: a 60 ms sort left a 240 ms hole in the field's frames;
- a CPU-heavy job (Measure) made the Dynamic animation policy stop the field
  for the whole run: 0 frames shaded in 68 s.

The shading thread was never starved: it sat in its wait. Both are pacing
decisions, so the fix is the pace: a run slows the field to ``_RUN_FPS``
(30, the capture rate) and a CPU-heavy job paces it instead of stopping it.

The frame-rate tests are self-calibrated: the rate the field reaches idle on
this machine bounds what is asked of it under load.
"""
from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtWidgets import QApplication, QWidget                  # noqa: E402

from spacr.qt import gil_priority                                    # noqa: E402
from spacr.qt.widgets import ambient                                 # noqa: E402

pytestmark = pytest.mark.qt

THEME = "data_art_impulse_lens"

#: The least a backdrop may drop to during a run and still read as moving:
#: the tutorial capture records at 30 fps and its smoothness gate counts
#: distinct frames.
SMOOTH_FPS = 24


class _CpuJob:
    """The Dynamic animation policy while a CPU-heavy job such as Measure runs."""

    enabled = True
    cpu_busy = True
    gpu_busy = False

    def pauses_animation(self):
        return True

    def blocks_gpu(self):
        return True


@pytest.fixture(autouse=True)
def _no_hub_left_behind(qapp):
    """Take down an application event hub these backdrops installed."""
    before = gil_priority._application_event_hub(qapp, create=False)
    yield
    after = gil_priority._application_event_hub(qapp, create=False)
    if after is not None and after is not before:
        qapp.removeEventFilter(after)
        setattr(qapp, gil_priority._HUB_ATTRIBUTE, before)
        after.deleteLater()


@pytest.fixture
def field(qtbot, monkeypatch):
    """A field backdrop on screen at 60 fps, shading on its own thread."""
    monkeypatch.setattr(ambient, "_RESOURCE_POLICY", None)
    host = QWidget()
    host.resize(480, 300)
    widget = ambient.AmbientWidget(host, theme=THEME, fps=60)
    widget._engine._graphics_backend = "cpu"
    widget.setGeometry(host.rect())
    qtbot.addWidget(host)
    host.show()
    qtbot.waitExposed(host)
    qtbot.waitUntil(lambda: widget.shading_thread_alive(), timeout=10000)
    yield widget
    widget.stop()


def _frames(widget, seconds, until=None):
    """Pump the event loop for ``seconds``, or until ``until()`` is true;
    frames shaded per second and the longest wait for a new one, in seconds."""
    app = QApplication.instance()
    stamps = []
    last = None
    count = 0
    end = time.perf_counter() + seconds
    started = time.perf_counter()
    while time.perf_counter() < end and not (until is not None and until()):
        app.processEvents()
        producer = widget._producer_box[0]
        shaded = producer.frames_shaded if producer is not None else None
        if shaded is not None and shaded != last:
            if last is not None:
                count += max(1, shaded - last) if shaded > last else shaded
            stamps.append(time.perf_counter())
            last = shaded
        time.sleep(0.002)
    finished = time.perf_counter()
    edges = [started] + stamps + [finished]
    longest = max(b - a for a, b in zip(edges, edges[1:]))
    return count / max(1e-6, finished - started), longest


def _busy_python(stop):
    """A pure-Python worker: the interpreter lock is what it holds."""
    total = 0
    while not stop.is_set():
        for value in range(2000):
            total += value * value


def test_the_run_pace_is_smooth():
    assert ambient._RUN_FPS >= SMOOTH_FPS


def test_a_cpu_heavy_job_paces_the_field_instead_of_stopping_it(
        field, monkeypatch):
    monkeypatch.setattr(ambient, "_RESOURCE_POLICY", _CpuJob())
    field._sync_run_state()
    assert field.is_running() and field.shading_thread_alive()
    assert field._rate() == ambient._RUN_FPS
    monkeypatch.setattr(ambient, "_RESOURCE_POLICY", None)
    field._sync_run_state()
    assert field._rate() == 60


def test_the_field_keeps_publishing_frames_during_a_cpu_heavy_python_run(
        field, monkeypatch):
    """Self-calibrated against the idle rate on the same machine."""
    idle, _ = _frames(field, 1.0)
    assert idle > 5, f"the field shades only {idle:.1f} fps idle here"
    stop = threading.Event()
    worker = threading.Thread(target=_busy_python, args=(stop,), daemon=True)
    gil_priority.claim()
    monkeypatch.setattr(ambient, "_RESOURCE_POLICY", _CpuJob())
    field._sync_run_state()
    worker.start()
    try:
        busy, longest = _frames(field, 2.0)
    finally:
        stop.set()
        worker.join(5)
        gil_priority.release()
        monkeypatch.setattr(ambient, "_RESOURCE_POLICY", None)
        field._sync_run_state()
    expected = min(idle, SMOOTH_FPS)
    assert busy >= 0.5 * expected, (
        f"{busy:.1f} fps during the run against {idle:.1f} idle")
    assert longest < max(0.4, 8.0 / expected), (
        f"the field went {longest * 1000:.0f} ms without a frame")


def test_a_short_job_runner_job_leaves_no_hole_in_the_field(field, qtbot):
    """The Organize popup's sort: a hidden JobRunner job of tens of ms."""
    from spacr.qt.job_runner import JobRunner

    idle, _ = _frames(field, 0.8)
    owner = QWidget()
    qtbot.addWidget(owner)
    runner = JobRunner(owner, app_key="organize images", user_visible=False)
    done = []

    def sort_like():
        total = 0
        deadline = time.perf_counter() + 0.8
        while time.perf_counter() < deadline:
            for value in range(2000):
                total += value * value
        return total

    runner.submit(sort_like, done.append)
    busy, longest = _frames(field, 10.0, until=lambda: bool(done))
    assert done
    runner.shutdown()
    expected = min(idle, SMOOTH_FPS)
    assert busy >= 0.5 * expected, (
        f"{busy:.1f} fps during the job against {idle:.1f} idle")
    assert longest < max(0.4, 8.0 / expected), (
        f"the field went {longest * 1000:.0f} ms without a frame")
