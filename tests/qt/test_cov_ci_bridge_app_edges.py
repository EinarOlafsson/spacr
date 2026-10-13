"""Recovery paths for registered jobs and idle screen preparation."""
from __future__ import annotations

import contextlib
import sys
import types

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QWidget  # noqa: E402

from spacr.qt import app, bridge  # noqa: E402

pytestmark = pytest.mark.qt


def test_an_external_job_can_be_retired_twice_without_evicting_a_peer(
        monkeypatch):
    registry = bridge.RunRegistry()
    monkeypatch.setattr(bridge, "_REGISTRY", registry)
    first = bridge._track_external_job("first")
    peer = bridge._track_external_job("peer")
    try:
        first.retire()
        first.retire()
        assert first not in registry.active()
        assert peer in registry.active()
    finally:
        peer.retire()


def test_a_worker_lookup_ignores_other_jobs_and_an_absent_worker(monkeypatch):
    wanted = object()
    peer = types.SimpleNamespace(worker=object())
    matching = types.SimpleNamespace(worker=wanted)
    active = [peer, matching]
    monkeypatch.setattr(bridge, "registry", lambda: types.SimpleNamespace(
        active=lambda: list(active)))

    assert bridge._registered_handle(wanted) is matching
    assert bridge._registered_handle(object()) is None
    assert bridge._registered_handle(None) is None


@pytest.mark.parametrize("headroom,snapshot,reserve,expected", [
    (True, None, 0, True),
    (False, None, 0, False),
    (False, (800, 1000), 300, True),
    (False, (2_000_000_000, 3_000_000_000), 100_000_000, False),
])
def test_idle_prewarm_respects_available_memory(
        monkeypatch, headroom, snapshot, reserve, expected):
    from spacr import resource_log
    from spacr.qt import memory_budget

    monkeypatch.setattr(memory_budget, "headroom_is_short",
                        lambda: headroom)
    monkeypatch.setattr(resource_log, "_ram_snapshot", lambda: snapshot)
    monkeypatch.setattr(resource_log, "_ram_reserve_bytes",
                        lambda _total: reserve)
    assert app._prewarm_memory_is_short() is expected


def test_idle_prewarm_tolerates_an_unreadable_memory_probe(monkeypatch):
    from spacr.qt import memory_budget

    def unreadable():
        raise OSError("memory counters vanished")

    monkeypatch.setattr(memory_budget, "headroom_is_short", unreadable)
    assert app._prewarm_memory_is_short() is False


def test_background_import_failures_do_not_skip_later_warmups(monkeypatch):
    from spacr import run_journal

    calls = []

    class InlineThread:
        def __init__(self, *, target, name, daemon):
            self.target = target
            assert name == "spacr-data-libraries" and daemon

        def start(self):
            self.target()

    def unavailable(_name):
        calls.append("import")
        raise ImportError("optional library is missing")

    def cannot_warm_artwork():
        calls.append("artwork")
        raise RuntimeError("artwork is unavailable")

    def cannot_warm_environment():
        calls.append("environment")
        raise RuntimeError("package metadata is unavailable")

    monkeypatch.setattr(app, "_DATA_LIBRARIES", ("missing-optional-module",))
    monkeypatch.setattr(app, "_DATA_LIBRARIES_STARTED", False)
    monkeypatch.setattr(app, "threading", types.SimpleNamespace(Thread=InlineThread))
    monkeypatch.setattr(app._importlib, "import_module", unavailable)
    monkeypatch.setattr(app._timing, "span", lambda *_args: contextlib.nullcontext())
    monkeypatch.setitem(sys.modules, "spacr.qt.widgets.organism_diagram",
                        types.SimpleNamespace(_warm_the_artwork=cannot_warm_artwork))
    monkeypatch.setattr(run_journal, "_warm_env_snapshot",
                        cannot_warm_environment)
    monkeypatch.setattr(app, "_freeze_what_survived",
                        lambda: calls.append("freeze"))

    assert isinstance(app._import_the_data_libraries_off_the_gui_thread(),
                      InlineThread)
    assert calls == ["import", "artwork", "environment", "freeze"]
    assert app._import_the_data_libraries_off_the_gui_thread() is None


def test_prewarm_skips_a_screen_that_the_user_already_opened(qapp):
    class Window(QWidget):
        def _prewarmable_keys(self):
            return {"mask", "queue"}

        def _prewarm_steps(self, key):
            return iter((key,))

    window = Window()
    window._screens = {"mask": object()}
    prewarm = app._ScreenPrewarm(window, ["mask", "queue"])
    try:
        assert prewarm._next_screen() is True
        assert prewarm.skipped == ["mask"]
        assert prewarm._key == "queue"
        assert list(prewarm.take_steps_for("queue")) == ["queue"]
        app.MainWindow._run_one_prewarm_step(None, prewarm)
        prewarm._running_step = True
        prewarm._slice()
        assert prewarm._timer.isActive() is False
    finally:
        prewarm.stop()
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_prewarm_waits_if_the_application_has_gone_away(monkeypatch):
    monkeypatch.setattr(app, "_time_monotonic", lambda: 100.0)
    monkeypatch.setattr(app, "QApplication", types.SimpleNamespace(
        instance=lambda: None))
    prewarm = types.SimpleNamespace(_last_input=0.0)
    assert app._ScreenPrewarm.busy_reason(prewarm) == "no application"


def test_a_deleted_screen_does_not_abort_idle_layout():
    requested = []

    class DeadScreen:
        def resize(self, size):
            requested.append(size)
            raise RuntimeError("C++ widget is gone")

    owner = types.SimpleNamespace(_stack=types.SimpleNamespace(size=lambda: (1, 1)))
    app.MainWindow._lay_out_unshown(owner, DeadScreen())
    assert requested == [(1, 1)]
