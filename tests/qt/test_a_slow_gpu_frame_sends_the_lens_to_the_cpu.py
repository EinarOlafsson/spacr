"""A GPU frame that holds the interpreter lock sends the backdrop lens to
the CPU.

A moderngl frame holds the interpreter lock from submission to readback, so
a card busy elsewhere (a screen recorder encoding, a run on the same card)
froze the GUI thread for 0.13-0.18 s on the tutorial capture host while the
impulse lens shaded on the GPU. ``_GraphicsBackend._graphics_frame`` times
each GPU frame after a renderer's first and shades on the CPU for
``_GRAPHICS_SLOW_BACKOFF_S`` when one holds the lock longer than
``_GRAPHICS_HOLD_BUDGET_S``.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtGui import QImage                                 # noqa: E402
from PySide6.QtWidgets import QWidget                            # noqa: E402

from spacr.qt.widgets import ambient                             # noqa: E402

pytestmark = pytest.mark.qt

THEME = "data_art_impulse_lens"


@pytest.fixture(autouse=True)
def _no_hub_left_behind(qapp):
    """Take down an application event hub these backdrops installed."""
    from spacr.qt import gil_priority

    before = gil_priority._application_event_hub(qapp, create=False)
    yield
    after = gil_priority._application_event_hub(qapp, create=False)
    if after is not None and after is not before:
        qapp.removeEventFilter(after)
        setattr(qapp, gil_priority._HUB_ATTRIBUTE, before)
        after.deleteLater()


class _GilHoldingRenderer:
    """A GPU renderer whose every frame holds the interpreter lock, as a
    moderngl readback does while the card is busy elsewhere."""

    def __init__(self, hold_s):
        import time

        count = 10 ** 6
        started = time.perf_counter()
        sum(range(count))
        self.loop = max(1, int(hold_s * count
                               / max(1e-6, time.perf_counter() - started)))
        self.calls = 0

    def _draw_subpixel(self, width, height, *args):
        self.calls += 1
        sum(range(self.loop))
        image = QImage(width, height, QImage.Format_RGB32)
        image.fill(0)
        return image


def test_a_slow_gpu_frame_sends_the_lens_to_the_cpu(monkeypatch):
    """The first frame compiles and is not judged; a slow second one backs
    the engine off the GPU, which it tries again after the backoff."""
    renderer = _GilHoldingRenderer(0.04)
    monkeypatch.setattr(ambient._GraphicsBackend, "_graphics_renderer",
                        lambda self, width, height, work=0: renderer)
    engine = ambient.AmbientWidget(theme=THEME, seed=2)._engine
    draw = ("_draw_subpixel", 64, 48)
    assert engine._graphics_frame(*draw) is not None
    assert engine._graphics_frame(*draw) is not None
    assert renderer.calls == 2
    assert engine._graphics_frame(*draw) is None, "backed off to the CPU"
    assert renderer.calls == 2
    monkeypatch.setattr(engine, "_graphics_slow_until", 0.0)
    fast = _GilHoldingRenderer(0.0)
    monkeypatch.setattr(ambient._GraphicsBackend, "_graphics_renderer",
                        lambda self, width, height, work=0: fast)
    for _ in range(4):
        assert engine._graphics_frame(*draw) is not None
    assert fast.calls == 4, "a fast GPU keeps the GPU"


def test_the_gui_never_waits_long_on_the_lens_during_impulses(qtbot,
                                                             monkeypatch):
    """With the impulse lens firing on a GPU whose frames hold the lock for
    150 ms, at most the first two frames (compile, then the judged one)
    stall the GUI thread; after them no heartbeat gap reaches 100 ms.

    Measured on Home's workstation, 1600x949, 8 s: without the backoff
    p95 153 ms, p99 180 ms, max 200 ms, 54 gaps over 60 ms; with it p95
    3.1 ms, p99 4.5 ms, max 33 ms, none over 60 ms.
    """
    import time

    from PySide6.QtCore import QTimer, Qt

    renderer = _GilHoldingRenderer(0.15)
    monkeypatch.setattr(ambient, "_RESOURCE_POLICY", None)
    monkeypatch.setattr(ambient._GraphicsBackend, "_graphics_renderer",
                        lambda self, width, height, work=0: renderer)
    host = QWidget()
    host.resize(800, 500)
    backdrop = ambient.AmbientWidget(host, theme=THEME, fps=60, seed=5)
    backdrop.follow_parent()
    backdrop._engine._graphics_backend = "gpu"
    qtbot.addWidget(host)
    host.show()
    qtbot.waitExposed(host)
    gaps, last = [], [time.perf_counter()]

    def beat():
        now = time.perf_counter()
        gaps.append(now - last[0])
        last[0] = now

    def impulse():
        backdrop._pending_art_impulses.append((0.3, 0.6))

    heartbeat = QTimer()
    heartbeat.setTimerType(Qt.PreciseTimer)
    heartbeat.timeout.connect(beat)
    heartbeat.start(2)
    impulses = QTimer()
    impulses.timeout.connect(impulse)
    impulses.start(200)
    try:
        qtbot.wait(3000)
    finally:
        heartbeat.stop()
        impulses.stop()
        backdrop.stop()
    long_gaps = [gap for gap in gaps if gap >= 0.1]
    assert len(long_gaps) <= 2, [round(gap, 3) for gap in long_gaps]
    assert renderer.calls <= 2, "the lens stayed on the GPU"
    assert backdrop.frames_painted > 30, "the field kept moving on the CPU"
