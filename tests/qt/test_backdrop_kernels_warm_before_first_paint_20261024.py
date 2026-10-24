"""The backdrop's compiled kernels are ready before the window paints.

Importing numba and loading a kernel holds the interpreter lock for 20-130 ms
at a time; done after the window was up, it starved the GUI thread for
0.13-0.17 s while the first field frames were shaded (tutorial captures,
lessons 56 and 85). Startup now warms them before the first show, and a
running field afterwards leaves the GUI thread its turn.
"""
from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("numba")

from spacr.qt.widgets import ambient


def _warm():
    assert ambient._start_kernel_warmup_before_first_paint()
    ready = ambient._finish_kernel_warmup_before_first_paint(timeout_s=240)
    if not ready or ambient._PACKED_SCATTER_FAILED:
        pytest.skip("the CPU kernels could not be compiled here")


def _longest_gap(seconds):
    """The longest wait between 2 ms naps, the GUI thread's view of a lock hold.

    The same loop with no shading gives the machine's own scheduling gap, so
    a loaded runner is not mistaken for a held interpreter lock.
    """
    longest, last = 0.0, time.perf_counter()
    end = last + seconds
    while last < end:
        time.sleep(0.002)
        now = time.perf_counter()
        longest, last = max(longest, now - last), now
    return longest


def test_startup_warms_the_kernels_before_the_first_show():
    from pathlib import Path

    source = Path(ambient.__file__).resolve().parents[1].joinpath("app.py").read_text(encoding="utf-8")
    start = source.index("_start_kernel_warmup_before_first_paint()")
    window = source.index("win = MainWindow(initial_app=initial_app)")
    finish = source.index("_finish_kernel_warmup_before_first_paint()")
    show = source.index("    _show_at_launch(win)\n")
    assert start < window < finish < show


def test_a_running_field_leaves_the_gui_thread_its_turn(qapp):
    _warm()
    width, height = 1280, 720
    engine = ambient.make_engine("data_art_impulse_lens", "spacr", "#101418",
                                 seed=42, resolution=1, blur=0)
    engine.set_max_pixels(width * height)
    engine.set_gravity_radius(0.35)
    engine.pointer = (0.4, 0.5)
    for index in range(12):
        engine._add_impulse((0.1 + 0.07 * index, 0.3 + 0.03 * index))
    engine.shade(width, height)
    baseline = _longest_gap(1.0)
    stop, shaded = threading.Event(), []

    def shade():
        while not stop.is_set():
            engine.advance(1 / 60)
            engine.shade(width, height)
            shaded.append(1)

    worker = threading.Thread(target=shade, daemon=True)
    worker.start()
    try:
        longest = _longest_gap(2.0)
    finally:
        stop.set()
        worker.join(5)
    assert len(shaded) > 10
    assert longest < 0.040 + baseline, (
        f"the GUI thread waited {longest * 1000:.0f} ms "
        f"(idle machine gap {baseline * 1000:.0f} ms)")
