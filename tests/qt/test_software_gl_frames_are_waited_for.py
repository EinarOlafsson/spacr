"""A software rasteriser's slow frame is waited for, a hardware frame is not.

CI's llvmpipe takes longer than the 0.5 s first-frame and 33 ms frame budgets
the graphics lane gives a real GPU (item 747), so growth's producer route
never drew there. The route that draws on a software rasteriser the product
refuses raises ``_GRAPHICS_WAIT_SCALE``; every real context keeps the budget.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import ambient  # noqa: E402


class _Lane:
    def __init__(self):
        self.timeouts = []

    def busy(self):
        return False

    def alive(self):
        return True

    def call(self, fn, timeout_s):
        self.timeouts.append(timeout_s)
        return False, None, None


def _first_and_later_waits(monkeypatch):
    lane = _Lane()
    engine = ambient.make_engine("data_art_fungal_growth", "spacr", "#191919", seed=3)
    engine._graphics_backend = "gpu"
    engine._graphics_lane = lane
    monkeypatch.setattr(engine, "_graphics_accepts", lambda *a, **k: True)
    engine._graphics_frame_on_lane("_draw_filaments", 64, 64, (), 0, None)
    engine._graphics_judged = True
    engine._graphics_slow_until = 0.0
    engine._graphics_frame_on_lane("_draw_filaments", 64, 64, (), 0, None)
    return lane.timeouts


def test_a_real_context_keeps_the_frame_budgets(qapp, monkeypatch):
    first, later = _first_and_later_waits(monkeypatch)
    assert first == pytest.approx(ambient._GRAPHICS_FIRST_FRAME_WAIT_S)
    assert later == pytest.approx(ambient._GRAPHICS_FRAME_WAIT_S)


def test_the_software_route_waits_the_scaled_budgets(qapp, monkeypatch):
    monkeypatch.setattr(ambient, "_GRAPHICS_WAIT_SCALE", 60.0)
    first, later = _first_and_later_waits(monkeypatch)
    assert first == pytest.approx(60.0 * ambient._GRAPHICS_FIRST_FRAME_WAIT_S)
    assert later == pytest.approx(60.0 * ambient._GRAPHICS_FRAME_WAIT_S)
