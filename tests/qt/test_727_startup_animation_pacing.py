"""Software pacing and paint-callback budgets, not compositor presentation."""
from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from spacr.qt import startup_benchmark as sb, timing
from spacr.qt.widgets import loading_screen as ls
from tests.qt.test_694_photomosaic_blow_away_startup_screen import (
    _played, cover, host,
)


FRAME_BUDGETS = {
    "longest_gap_ms": 100.0,
    "p99_refresh_multiple": 2.5,
    "dropped_share": 0.05,
    "mean_ms": 8.0,
}


@pytest.mark.parametrize("rate", [60.0, 120.0, 144.0])
def test_refresh_deadlines_do_not_accumulate_timer_rounding(cover, rate):
    _played(cover)
    now = [100.0]
    cover._pace_clock = lambda: now[0]
    cover._pace_to(SimpleNamespace(refreshRate=lambda: rate))
    step = 1.0 / rate
    for frame in range(1, 13):
        assert cover._next_due == pytest.approx(100.0 + frame * step)
        assert 0 <= cover._tick_timer.interval() <= math.ceil(1000.0 * step)
        now[0] = cover._next_due
        cover._schedule_tick()
    now[0] += step * 10
    cover._schedule_tick()
    assert cover._next_due == pytest.approx(now[0] + step)
    cover._finish()


def test_the_real_window_screen_signal_repaces_the_cover(cover, host, qtbot):
    host.show()
    qtbot.waitExposed(host)
    _played(cover)
    handle = cover._paced_window
    assert handle is not None
    cover._pace_to(SimpleNamespace(refreshRate=lambda: 144.0))
    assert cover._refresh_hz == 144.0
    handle.screenChanged.emit(handle.screen())
    assert cover._refresh_hz == ls._screen_refresh_hz(handle.screen())
    assert cover._frame_budget_ms() == pytest.approx(1000.0 / cover._refresh_hz)
    assert cover._tick_timer.isActive()
    cover._finish()


@pytest.mark.parametrize("rate", [60.0, 120.0, 144.0])
def test_adaptation_uses_refresh_shares_and_bounded_step_down(cover, rate):
    _played(cover)
    cover._pace_to(SimpleNamespace(refreshRate=lambda: rate))
    budget = 1000.0 / rate
    cover._adapt(budget * 0.9)
    assert cover._floor == pytest.approx(0.10)
    cover._adapt(budget * 0.6)
    assert cover._floor == pytest.approx(0.10)
    cover._adapt(budget * 0.5)
    assert cover._floor == pytest.approx(0.06)
    stages = []
    for _ in range(80):
        before = (cover._stride, cover._divisor)
        cover._adapt(cover._frame_budget_ms())
        after = (cover._stride, cover._divisor)
        if after != before:
            stages.append(after)
    assert stages[:2] == [(2, 1), (4, 1)]
    assert cover._stride == 4 and cover._divisor == int(rate // 30)
    assert rate / cover._divisor >= 30.0
    cover._finish()


def test_preparing_in_the_hold_avoids_tile_work_in_the_first_frame(cover, monkeypatch):
    from spacr.qt.gil_priority import background_imports_held

    assert not background_imports_held()
    cover.grab()
    cuts = []
    real_cut = cover._cut_tiles

    def cut(*args):
        cuts.append(cover._state)
        return real_cut(*args)

    monkeypatch.setattr(cover, "_cut_tiles", cut)
    assert cover.blow_away(force=True)
    assert background_imports_held()
    cover._prepare()
    assert cuts == ["holding"] and cover._prepared is not None
    cover.clock.now += ls._MOSAIC_MIN_HOLD_S
    cover._begin()
    assert cover._state == "playing" and cuts == ["holding"]
    assert background_imports_held()
    cover._finish()
    assert not background_imports_held()


def test_callback_cadence_counts_late_and_estimated_dropped_frames():
    report = ls._pacing_stats([0.0, 0.01, 0.03, 0.04], 10.0)
    assert report["late_frames"] == 1 and report["dropped_frames"] == 1
    assert report["dropped_share"] == pytest.approx(0.2)
    assert report["longest_gap_ms"] == pytest.approx(20.0)
    assert report["interval_p99_ms"] == pytest.approx(np.percentile([10, 20, 10], 99))
    assert ls._display_paced("cocoa")
    assert not any(ls._display_paced(platform) for platform in ("xcb", "wayland", "windows", "offscreen"))


@pytest.mark.parametrize("plain", [False, True])
def test_completed_stats_survive_cover_deletion_in_the_real_benchmark(
        cover, host, qtbot, qapp, tmp_path, monkeypatch, plain):
    monkeypatch.delenv(sb.HARD_TIMEOUT_ENV, raising=False)
    cover.grab()
    assert cover.blow_away(force=True, plain=plain)
    cover.clock.now += ls._MOSAIC_MIN_HOLD_S
    cover._begin()
    for offset in (0.1, 0.2, 0.3):
        cover.clock.now = cover._began_at + offset
        cover.grab()
    expected = cover.frame_stats()
    cover.clock.now = cover._began_at + cover._duration
    cover._tick()
    qtbot.waitUntil(lambda: not host.findChildren(ls.LoadingScreen))
    controller = sb.BenchmarkController(qapp, host, (), str(tmp_path / "benchmark.json"),
                                        live_keys=lambda: (), measure_preferences=False)
    try:
        assert controller._persist("registry sweep complete") == ""
        record = json.loads((tmp_path / "benchmark.json").read_text())["startup_animation"]
        assert record["frame_stats"] == expected
        assert record["completed"] is True
        assert record["mode"] == ("cross_fade" if plain else "tiles")
        assert record["sample_source"] == "paint_callbacks"
        assert record["presentation_verified"] is False
    finally:
        controller.timeout.stop()
        timing.unsubscribe_readiness(controller._ready)
        qapp.aboutToQuit.disconnect(controller._application_quit)
        controller.deleteLater()


def test_skipping_does_not_report_a_completed_zero_cost_animation(cover):
    cover.grab()
    assert cover.blow_away(force=True)
    cover.skip()
    record = cover.parentWidget()._startup_animation_stats
    assert record["completed"] is False and record["frame_stats"]["frames"] == 0
    assert record["presentation_verified"] is False


def test_real_software_replay_meets_the_paint_callback_budget(host, qtbot):
    host.show()
    screen = ls.LoadingScreen(parent=host)
    assert screen.show_mosaic()
    screen.show()
    assert screen.blow_away(force=True)
    try:
        qtbot.waitUntil(lambda: hasattr(host, "_startup_animation_stats"), timeout=10000)
        record = host._startup_animation_stats
        stats = record["frame_stats"]
        assert record["completed"] is True and stats["frames"] > 30
        assert record["sample_source"] == "paint_callbacks" and not record["presentation_verified"]
        assert stats["mean_ms"] < FRAME_BUDGETS["mean_ms"], stats
        assert stats["longest_gap_ms"] < FRAME_BUDGETS["longest_gap_ms"], stats
        assert stats["interval_p99_ms"] < stats["refresh_ms"] * FRAME_BUDGETS["p99_refresh_multiple"], stats
        assert stats["dropped_share"] < FRAME_BUDGETS["dropped_share"], stats
    finally:
        from shiboken6 import isValid
        if isValid(screen):
            screen._finish()
