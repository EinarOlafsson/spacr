"""The frozen .dmg smoke never imports the numerical stack on two threads at once.

Release run 39355725517: the macOS smoke stayed ``running`` in its launch
phase till the 660 s watchdog. Its provenance check imports torch, NumPy and
spaCR on the GUI thread while the start-up warm imported pandas, SciPy,
Matplotlib, scikit-image and numba on its own thread, through PyInstaller's
frozen importer. In a frozen build or a smoke run the warm now leaves that
group to its first use, the smoke's first pass waits for the warm thread on
the event loop, the receipt names the phase before every step, and a stack
dump reaches the application log before the watchdog fires.
"""

import faulthandler
import sys
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from spacr.qt import app as app_module
from spacr.qt import startup_benchmark as benchmark
from spacr.qt.widgets import ambient

Controller = benchmark._DistributionSmokeController
HEAVY = ("spacr.settings_advisor", "pandas", "scipy.linalg", "scipy.stats",
         "matplotlib.figure", "matplotlib.font_manager", "skimage.data",
         "skimage.draw", "skimage.filters", "skimage.measure",
         "skimage.morphology")


@pytest.fixture
def recorded_warm(monkeypatch):
    """Run the real warm thread with every import and numba load recorded."""
    imported, numba = [], []
    monkeypatch.setattr(app_module, "_importlib", SimpleNamespace(
        import_module=imported.append))
    monkeypatch.setattr(ambient, "_import_numba", lambda: numba.append(1))
    monkeypatch.setattr(app_module, "_freeze_what_survived", lambda: None)
    monkeypatch.setattr(app_module, "_SETTINGS_PREWARM_THREAD", None)
    monkeypatch.setattr(app_module, "_HEAVY_WARM_GATE", threading.Event())
    return imported, numba


def test_a_normal_source_launch_still_warms_the_numerical_stack(monkeypatch):
    monkeypatch.delenv("SPACR_DISTRIBUTION_SMOKE", raising=False)
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert app_module._heavy_warm_allowed() is True


@pytest.mark.parametrize("smoke, frozen", [(True, False), (False, True), (True, True)])
def test_a_frozen_build_or_smoke_run_never_warms_the_numerical_stack(
        monkeypatch, smoke, frozen):
    if smoke:
        monkeypatch.setenv("SPACR_DISTRIBUTION_SMOKE", "1")
    else:
        monkeypatch.delenv("SPACR_DISTRIBUTION_SMOKE", raising=False)
    if frozen:
        monkeypatch.setattr(sys, "frozen", True, raising=False)
    else:
        monkeypatch.delattr(sys, "frozen", raising=False)
    assert app_module._heavy_warm_allowed() is False


def test_the_smoke_warm_finishes_at_once_without_the_heavy_group(
        monkeypatch, recorded_warm):
    imported, numba = recorded_warm
    monkeypatch.setenv("SPACR_DISTRIBUTION_SMOKE", "1")
    thread = app_module._start_settings_prewarm(wait_for_home=True)
    thread.join(10)
    assert not thread.is_alive()
    assert app_module._HEAVY_WARM_GATE.is_set()
    assert app_module._settings_prewarm_running() is False
    assert "spacr.settings" in imported
    assert not set(HEAVY) & set(imported)
    assert numba == []


def test_a_source_warm_still_imports_the_heavy_group(monkeypatch, recorded_warm):
    imported, numba = recorded_warm
    monkeypatch.delenv("SPACR_DISTRIBUTION_SMOKE", raising=False)
    monkeypatch.delattr(sys, "frozen", raising=False)
    thread = app_module._start_settings_prewarm(wait_for_home=False)
    thread.join(10)
    assert not thread.is_alive()
    assert set(HEAVY) <= set(imported)
    assert numba == [1]


def test_a_running_warm_is_seen_by_the_smoke(monkeypatch):
    release = threading.Event()
    thread = threading.Thread(target=release.wait, args=(10,), daemon=True)
    thread.start()
    monkeypatch.setattr(app_module, "_SETTINGS_PREWARM_THREAD", thread)
    fake = SimpleNamespace(record={})
    try:
        assert Controller._start_up_warm_running(fake) is True
        assert fake.record["waited_for_start_up_warm"] is True
    finally:
        release.set()
        thread.join(10)
    assert Controller._start_up_warm_running(SimpleNamespace(record={})) is False


def _ticking(phase, warm_running, elapsed=0.0):
    """A controller stand-in that records receipt writes and advances."""
    calls = []
    fake = SimpleNamespace(
        phase=phase, record={}, started=benchmark.time.monotonic() - elapsed,
        _write=lambda: calls.append(("write", dict(fake.record))),
        _start_up_warm_running=lambda: warm_running,
        _advance=lambda: calls.append(("advance", fake.phase)))
    return fake, calls


def test_the_first_pass_waits_on_the_event_loop_while_the_warm_imports():
    fake, calls = _ticking("launch", warm_running=True)
    Controller._tick(fake)
    assert calls == [("write", {"phase": "launch"})]
    Controller._tick(fake)
    assert calls == [("write", {"phase": "launch"})]


def test_the_first_pass_runs_once_the_warm_is_done():
    fake, calls = _ticking("launch", warm_running=False)
    Controller._tick(fake)
    assert calls == [("write", {"phase": "launch"}), ("advance", "launch")]


def test_a_warm_that_never_ends_still_meets_the_deadline():
    fake, calls = _ticking("launch", warm_running=True, elapsed=601)
    Controller._tick(fake)
    assert calls[-1] == ("advance", "launch")


def test_each_new_phase_reaches_the_receipt_before_its_step():
    fake, calls = _ticking("screen", warm_running=True)
    Controller._tick(fake)
    fake.phase = "configured"
    Controller._tick(fake)
    Controller._tick(fake)
    assert calls == [("write", {"phase": "screen"}), ("advance", "screen"),
                     ("write", {"phase": "configured"}), ("advance", "configured"),
                     ("advance", "configured")]


def test_the_smoke_arms_a_stack_dump_into_the_application_log(monkeypatch, tmp_path):
    armed, cancelled = [], []
    monkeypatch.setenv("SPACR_DISTRIBUTION_SMOKE", "1")
    monkeypatch.setenv("SPACR_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(faulthandler, "dump_traceback_later",
                        lambda seconds, exit, file: armed.append((seconds, exit, file.name)))
    monkeypatch.setattr(faulthandler, "cancel_dump_traceback_later",
                        lambda: cancelled.append(1))
    monkeypatch.setattr(benchmark, "_SMOKE_STACK_DUMP_ARMED", False)
    record = {}
    handle = benchmark._arm_the_smoke_stack_dump(record)
    try:
        from spacr.qt.verbose_logger import current_log_file

        log = str(current_log_file())
        assert armed == [(540, False, log)]
        assert record["stack_dump_log"] == log
        benchmark._cancel_the_smoke_stack_dump()
        benchmark._cancel_the_smoke_stack_dump()
        assert cancelled == [1]
    finally:
        handle.close()


def test_outside_a_smoke_run_no_stack_dump_is_armed(monkeypatch):
    armed = []
    monkeypatch.delenv("SPACR_DISTRIBUTION_SMOKE", raising=False)
    monkeypatch.setattr(faulthandler, "dump_traceback_later",
                        lambda *args, **kwargs: armed.append(args))
    record = {}
    assert benchmark._arm_the_smoke_stack_dump(record) is None
    assert armed == [] and record == {}
