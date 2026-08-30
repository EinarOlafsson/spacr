"""The startup benchmark observes an operable frame, not a constructor.

These are ordinary release-contract tests, deliberately outside ``test_cov``:
coverage work may add lines without weakening the first-paint acceptance seam.
"""
from __future__ import annotations

import inspect
import json
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QListWidget,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

import spacr.qt
from spacr.qt import timing
from spacr.qt.startup_benchmark import BenchmarkController, maybe_start


@pytest.fixture
def enabled_timing(monkeypatch):
    """One empty instrumentation session without changing the process env."""
    monkeypatch.setattr(timing, "ENABLED", True)
    monkeypatch.setattr(timing, "_START", time.perf_counter())
    monkeypatch.setattr(timing, "_EVENT_LOOP_STARTED_AT", None)
    monkeypatch.setattr(timing, "_SPANS", [])
    monkeypatch.setattr(timing, "_IMPORTS", [])
    monkeypatch.setattr(timing, "_STALLS", [])
    monkeypatch.setattr(timing, "_MARKS", [])
    monkeypatch.setattr(timing, "_READINESS", [])
    monkeypatch.setattr(timing, "_READY_CALLBACKS", [])
    monkeypatch.setattr(timing, "_ACTIVE_PROBES", [])
    monkeypatch.setattr(timing, "_LAST_GUI_BEAT_AT", None)
    monkeypatch.setattr(timing, "_GUI_WATCHDOG_ACTIVE", False)
    yield
    for probe in list(timing._ACTIVE_PROBES):
        try:
            probe._retire()
        except RuntimeError:
            pass


class _PaintedRoot(QWidget):
    def __init__(self):
        super().__init__()
        self.paint_count = 0

    def paintEvent(self, event):  # noqa: N802 - Qt naming
        super().paintEvent(event)
        self.paint_count += 1


class _PaintedButton(QPushButton):
    def __init__(self, text="usable"):
        super().__init__(text)
        self.paint_count = 0

    def paintEvent(self, event):  # noqa: N802 - Qt naming
        super().paintEvent(event)
        self.paint_count += 1


def _page():
    root = _PaintedRoot()
    layout = QVBoxLayout(root)
    button = _PaintedButton()
    layout.addWidget(button)
    return root, button


def test_readiness_waits_for_the_event_loop_and_completed_control_paint(
        qtbot, enabled_timing):
    root, button = _page()
    qtbot.addWidget(root)
    seen = []
    timing.subscribe_readiness(seen.append)
    timing.watch_interactive(
        root, "interactive Home", "__home__",
        started_at=timing.process_started_at(), budget_s=5.0)

    root.show()
    # processEvents can deliver paint, but production has not yet observed
    # the callback queued after QApplication.exec(). Constructor/show/paint
    # alone must therefore remain insufficient.
    qtbot.wait(30)
    assert root.paint_count and button.paint_count
    assert seen == []

    root_before = root.paint_count
    button_before = button.paint_count
    QTimer.singleShot(0, timing.event_loop_started)
    qtbot.waitUntil(lambda: len(seen) == 1, timeout=2000)

    entry = seen[0]
    assert entry["event_loop_started_at"] is not None
    assert entry["at"] >= entry["event_loop_started_at"]
    assert root.paint_count > root_before
    assert button.paint_count > button_before
    assert entry["root_painted"] is True
    assert entry["screen_tree_painted"] is True
    assert entry["painted_usable_controls"] == 1
    assert entry["within_budget"] is True


def test_a_painted_but_disabled_control_is_not_interactive(
        qtbot, enabled_timing):
    root, button = _page()
    button.setEnabled(False)
    qtbot.addWidget(root)
    seen = []
    timing.subscribe_readiness(seen.append)
    timing.watch_interactive(root, "interactive module", "probe")
    root.show()
    QTimer.singleShot(0, timing.event_loop_started)
    qtbot.wait(50)
    assert seen == []

    button.setEnabled(True)
    button.update()
    root.update()
    qtbot.waitUntil(lambda: len(seen) == 1, timeout=2000)
    assert seen[0]["painted_usable_controls"] == 1


@pytest.mark.parametrize("view_type", (QTableWidget, QListWidget))
def test_item_view_update_overloads_cannot_break_readiness(
        qtbot, enabled_timing, view_type):
    root = _PaintedRoot()
    layout = QVBoxLayout(root)
    view = view_type(root)
    layout.addWidget(view)
    qtbot.addWidget(root)
    seen = []
    timing.subscribe_readiness(seen.append)
    timing.watch_interactive(root, "interactive module", "item-view")

    root.show()
    QTimer.singleShot(0, timing.event_loop_started)
    qtbot.waitUntil(lambda: len(seen) == 1, timeout=2000)

    assert seen[0]["painted_usable_controls"] == 1
    assert seen[0]["controls"] == [view_type.__name__]


def test_public_run_begins_timing_before_qt_setup_app_import_and_registration():
    source = inspect.getsource(spacr.qt.run)
    begin = source.index("_timing.begin()")
    quiet = source.index("_quiet_gtk_accessibility()")
    app_import = source.index("from .app import launch")
    registration = source.index("register_self_registering_modules()")
    assert begin < quiet < app_import < registration


def test_parent_wall_clock_is_translated_to_the_child_monotonic_clock(
        enabled_timing, monkeypatch):
    monkeypatch.setenv("SPACR_TIMING_PROCESS_START", "998.75")
    monkeypatch.setattr(timing, "IMPORT_TIMING_ENABLED", False)
    monkeypatch.setattr(timing.time, "time", lambda: 1000.0)
    monkeypatch.setattr(timing.time, "perf_counter", lambda: 500.0)
    monkeypatch.setattr(timing.begin, "_done", False, raising=False)

    timing.begin()

    assert timing._START == 498.75
    assert timing._MARKS[-1]["detail"] == "benchmark process spawn"


def test_snapshot_exposes_the_release_budgets_and_peak_resources(
        enabled_timing, tmp_path):
    timing._STALLS.append({
        "at": 0.2, "late_ms": 500.0,
        "source": "event-loop watchdog", "thread": "MainThread",
    })
    state = timing.snapshot()
    assert state["budgets"] == {
        "home_ready_s": 5.0,
        "module_ready_s": 10.0,
        "max_event_loop_stall_ms": 500.0,
        "watchdog_record_floor_ms": 50.0,
    }
    assert state["stall_budget_met"] is False
    assert state["resources"]["peak_rss_mb"] > 0
    # A process that never imported Torch reports unknown; an earlier test
    # may have imported it without initializing CUDA, which honestly reports
    # zero instead. Neither path initializes CUDA merely to take a snapshot.
    assert state["resources"]["gpu"] in (
        {"allocated_mb": None, "peak_allocated_mb": None},
        {"allocated_mb": 0.0, "peak_allocated_mb": 0.0},
    )
    assert state["environment"]["pid"] > 0
    hardware = state["environment"]["hardware"]
    assert hardware["logical_cpu_count"] >= 1
    assert hardware["total_memory_mb"] > 0
    assert isinstance(hardware["displays"], list)

    path = tmp_path / "timing.json"
    assert timing.write_json(str(path)) == str(path)
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 1


def test_a_watchdog_gap_is_clipped_to_the_interaction_it_overlaps(
        enabled_timing):
    timing._STALLS.extend([
        {"started_at": 0.0, "at": 0.8, "late_ms": 800.0},
        {"started_at": 0.9, "at": 1.2, "late_ms": 300.0},
    ])

    rows = timing.stalls_between(0.6, 1.0)

    assert [round(row["overlap_ms"]) for row in rows] == [200, 100]
    assert [row["late_ms"] for row in rows] == [800.0, 300.0]


def test_benchmark_controller_uses_clicks_waits_for_paint_and_exits(
        qapp, qtbot, enabled_timing, tmp_path):
    shell = QWidget()
    layout = QVBoxLayout(shell)
    stack = QStackedWidget()
    layout.addWidget(stack)
    home, _home_button = _page()
    module, _module_button = _page()
    stack.addWidget(home)
    stack.addWidget(module)

    nav = QPushButton("Probe")
    nav.setProperty("navKey", "probe")

    def navigate():
        started = timing.interval_started("navigation", "probe")
        stack.setCurrentWidget(module)
        timing.watch_interactive(
            module, "interactive module", "probe", started_at=started,
            budget_s=timing.MODULE_BUDGET_S)

    nav.clicked.connect(navigate)
    window = SimpleNamespace(_sidebar=SimpleNamespace(_items=[nav]))
    output = tmp_path / "benchmark.json"
    controller = BenchmarkController(
        qapp, window, ("probe",), str(output), timeout_s=2.0)
    timing.watch_interactive(
        home, "interactive Home", "__home__",
        started_at=timing.process_started_at(), budget_s=timing.HOME_BUDGET_S)
    shell.show()
    QTimer.singleShot(0, timing.event_loop_started)

    qtbot.waitUntil(lambda: controller._finished, timeout=5000)
    assert output.is_file()
    artifact = json.loads(output.read_text(encoding="utf-8"))
    benchmark = artifact["benchmark"]
    assert benchmark["exit_reason"] == "registry sweep complete"
    assert benchmark["registry_keys"] == ["probe"]
    assert benchmark["measured_keys"] == ["probe"]
    assert benchmark["registry_matches_measurements"] is True
    assert benchmark["violations"] == []
    assert [row["detail"] for row in benchmark["results"]] == [
        "__home__", "probe"]
    assert all(row["painted_usable_controls"] >= 1
               for row in benchmark["results"])
    assert all(
        row["worst_overlapping_frame_interval_ms"]
        >= row["worst_event_loop_stall_ms"]
        for row in benchmark["results"]
    )
    assert controller._finished is True


def test_benchmark_waits_for_a_post_checkpoint_watchdog_beat(
        qapp, qtbot, enabled_timing, tmp_path, monkeypatch):
    window = SimpleNamespace(_sidebar=SimpleNamespace(_items=[]))
    controller = BenchmarkController(
        qapp, window, (), str(tmp_path / "barrier.json"), timeout_s=2.0)
    controller.timeout.stop()
    beats = iter((0.0, float("inf")))
    observed = []
    monkeypatch.setattr(timing, "last_gui_beat_at", lambda: next(beats))
    monkeypatch.setattr(controller, "_advance", lambda: observed.append(True))

    controller._advance_after_watchdog()

    qtbot.waitUntil(lambda: bool(observed), timeout=1000)
    assert observed == [True]
    controller._finished = True
    timing.unsubscribe_readiness(controller._ready)


def test_checkpoint_failure_is_contained_and_leaves_no_partial_artifact(
        qapp, enabled_timing, tmp_path, monkeypatch):
    output = tmp_path / "checkpoint.json"
    window = SimpleNamespace(_sidebar=SimpleNamespace(_items=[]))
    controller = BenchmarkController(
        qapp, window, (), str(output), timeout_s=2.0)
    controller.timeout.stop()

    def fail_snapshot(_reason):
        raise RuntimeError("snapshot failed")

    monkeypatch.setattr(controller, "_artifact", fail_snapshot)

    assert controller._persist("registry sweep in progress") == (
        "snapshot failed")
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []
    controller._finished = True
    timing.unsubscribe_readiness(controller._ready)


def test_production_benchmark_inventory_is_the_complete_live_registry(
        qapp, monkeypatch, tmp_path):
    spacr.qt.register_self_registering_modules()
    from spacr.qt.app import APPS

    output = tmp_path / "registry.json"
    monkeypatch.setenv("SPACR_BENCHMARK_JSON", str(output))
    window = SimpleNamespace(_sidebar=SimpleNamespace(_items=[]))
    controller = maybe_start(qapp, window)
    assert controller is not None
    assert controller.keys == tuple(row[0] for row in APPS)
    assert len(controller.keys) == len(set(controller.keys))
    assert len(controller.keys) >= 44
    controller.timeout.stop()
    timing.unsubscribe_readiness(controller._ready)


def test_a_registry_change_during_the_sweep_cannot_pass_exact_parity(
        qapp, enabled_timing, tmp_path):
    live = ["first", "second"]
    window = SimpleNamespace(_sidebar=SimpleNamespace(_items=[]))
    controller = BenchmarkController(
        qapp, window, tuple(live), str(tmp_path / "drift.json"),
        timeout_s=2.0, live_keys=lambda: tuple(live))
    controller.timeout.stop()
    live.append("late")

    artifact = controller._artifact("test")

    benchmark = artifact["benchmark"]
    assert benchmark["registry_keys"] == ["first", "second"]
    assert benchmark["final_registry_keys"] == ["first", "second", "late"]
    assert benchmark["registry_stable"] is False
    assert "the live registry changed during the benchmark sweep" in (
        benchmark["violations"])
    controller._finished = True
    timing.unsubscribe_readiness(controller._ready)


def test_readiness_arriving_after_a_timeout_cannot_skip_the_next_app(
        qapp, enabled_timing, tmp_path):
    window = SimpleNamespace(_sidebar=SimpleNamespace(_items=[]))
    controller = BenchmarkController(
        qapp, window, ("slow", "next"), str(tmp_path / "late.json"),
        timeout_s=2.0)
    controller.timeout.stop()
    controller.phase = "module"
    controller.current_key = "slow"
    controller.index = 0
    timing._START -= 1.0
    interval_end = timing.elapsed()
    controller._attempt_started_elapsed = interval_end - 0.6
    timing._STALLS.extend([
        {"started_at": interval_end - 1.4,
         "at": interval_end - 0.8, "late_ms": 600.0},
        {"started_at": interval_end - 0.6,
         "at": interval_end, "late_ms": 600.0},
    ])

    class LateProbe:
        report_name = "interactive module"
        report_detail = "slow"

        def _retire(self):
            timing._ACTIVE_PROBES.remove(self)

    probe = LateProbe()
    timing._ACTIVE_PROBES.append(probe)

    controller._record_error(
        "slow", "no painted usable state within 2.0 seconds",
        already_stopped=True)
    controller._ready({
        "name": "interactive module", "detail": "slow",
        "duration_s": 2.1,
    })

    assert controller.index == 1
    assert controller.current_key is None
    assert controller._pending is None
    assert [row["detail"] for row in controller.results] == ["slow"]
    assert controller.results[0]["worst_event_loop_stall_ms"] == pytest.approx(
        600.0)
    assert controller.results[0][
        "worst_overlapping_frame_interval_ms"] == 600.0
    assert controller.results[0]["event_loop_stall_budget_met"] is False
    assert probe not in timing._ACTIVE_PROBES
    controller._finished = True
    timing.unsubscribe_readiness(controller._ready)
