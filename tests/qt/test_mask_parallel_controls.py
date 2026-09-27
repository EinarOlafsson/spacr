"""Item 493: GPU mask-batch controls, per-GPU progress and cluster hand-off."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtWidgets import QLabel, QWidget                     # noqa: E402

from spacr import _mask_workers as mw                             # noqa: E402
from spacr.qt.screens import settings_model as sm                 # noqa: E402

NOTE = sm._PENDING_NOTE_PROPERTY


@pytest.fixture(autouse=True)
def _isolated_gpu_probe(monkeypatch):
    mw._stop_mask_gpu_probe()
    monkeypatch.setattr(mw, "_CONTROL_GPU_COUNT", [])
    monkeypatch.setattr(mw, "_CONTROL_GPU_PROBE", None)
    yield
    mw._stop_mask_gpu_probe()


def _panel(qtbot, monkeypatch, count):
    monkeypatch.setattr(mw, "_mask_gpu_count_for_controls", lambda: count)
    owner = QWidget()
    qtbot.addWidget(owner)
    panel = sm.SettingsWidgets("mask", parent=owner)
    panel.build_sections()
    return panel, panel._widgets["mask_parallel"], panel._widgets["mask_gpu_indices"]


@pytest.mark.parametrize("count", [0, 1])
def test_without_two_gpus_both_controls_are_greyed_with_the_reason(qtbot, monkeypatch, count):
    panel, parallel, indices = _panel(qtbot, monkeypatch, count)
    panel._refresh_mask_gpu_enablement()
    for control in (parallel, indices):
        assert not control.isEnabled()
        note = str(control.property(NOTE))
        assert f"{count} found" in note and "Cluster Distribution" in note


def test_with_two_gpus_the_list_follows_the_switch(qtbot, monkeypatch):
    panel, parallel, indices = _panel(qtbot, monkeypatch, 2)
    panel._refresh_mask_gpu_enablement()
    assert parallel.isEnabled() and not parallel.property(NOTE)
    assert not indices.isEnabled()
    assert "only when mask_parallel is on" in str(indices.property(NOTE))
    panel.set_value_for_key("mask_parallel", True)
    assert indices.isEnabled() and not indices.property(NOTE)
    panel.set_value_for_key("mask_parallel", False)
    assert not indices.isEnabled()


def test_the_defaults_leave_the_feature_off(qtbot, monkeypatch):
    panel, parallel, indices = _panel(qtbot, monkeypatch, 2)
    assert panel._read_widget(parallel) is False
    assert (panel._read_widget(indices) or "") == ""
    from spacr.settings import set_default_settings_preprocess_generate_masks
    defaults = set_default_settings_preprocess_generate_masks({})
    assert defaults["mask_parallel"] is False and defaults["mask_gpu_indices"] == ""
    from spacr.artifacts import material_settings
    assert "mask_parallel" not in material_settings(defaults)


class _ProbePipe:
    ready = False
    closed = False

    def poll(self):
        return self.ready

    def recv(self):
        return 3

    def close(self):
        self.closed = True


class _ProbeProcess:
    started = False
    stopped = False
    closed = False

    def __init__(self):
        self.joins = []

    def start(self):
        self.started = True

    def is_alive(self):
        return not self.stopped

    def terminate(self):
        self.stopped = True

    def kill(self):
        self.stopped = True

    def join(self, timeout):
        self.joins.append(timeout)

    def close(self):
        assert self.stopped
        self.closed = True


def _fake_probe(monkeypatch):
    from types import SimpleNamespace

    reader, writer = _ProbePipe(), _ProbePipe()
    process = _ProbeProcess()
    launches = []

    def make_process(**kwargs):
        launches.append(kwargs)
        return process

    context = SimpleNamespace(Pipe=lambda **_: (reader, writer), Process=make_process)
    monkeypatch.setattr(mw.multiprocessing, "get_context", lambda mode: context)
    monkeypatch.setattr(mw, "_CONTROL_GPU_COUNT", [])
    monkeypatch.setattr(mw, "_CONTROL_GPU_PROBE", None)
    monkeypatch.setattr(mw, "_compatible_mask_gpus", lambda: pytest.fail("probe ran in GUI"))
    return reader, writer, process, launches


def test_the_gpu_count_is_read_once_without_waiting_for_the_child(monkeypatch):
    reader, writer, process, launches = _fake_probe(monkeypatch)
    assert mw._mask_gpu_count_for_controls() is None
    assert process.started and writer.closed and not reader.closed
    assert mw._mask_gpu_count_for_controls() is None
    assert len(launches) == 1
    assert launches[0]["target"] is mw._probe_mask_gpu_count
    reader.ready = True
    assert mw._mask_gpu_count_for_controls() == 3
    assert mw._mask_gpu_count_for_controls() == 3
    assert reader.closed and process.stopped and len(launches) == 1
    assert process.closed and mw._CONTROL_GPU_PROBE is None
    assert process.joins == [0]


@pytest.mark.parametrize("failure", ["exit", "timeout", "eof"])
def test_failed_gpu_discovery_finishes_without_blocking(monkeypatch, failure):
    reader, _, process, _ = _fake_probe(monkeypatch)
    assert mw._mask_gpu_count_for_controls() is None
    if failure == "exit":
        process.stopped = True
    elif failure == "timeout":
        started = mw._CONTROL_GPU_PROBE['started']
        monkeypatch.setattr(mw.time, "monotonic", lambda: started + 61)
    else:
        reader.ready = True

        def closed_pipe():
            raise EOFError

        reader.recv = closed_pipe
    assert mw._mask_gpu_count_for_controls() == 0
    assert reader.closed and process.stopped


def test_discovery_retains_the_child_until_termination_is_confirmed(monkeypatch):
    reader, _, process, _ = _fake_probe(monkeypatch)
    now = [100.0]
    signals = []
    monkeypatch.setattr(mw.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(process, "terminate", lambda: signals.append("terminate"))
    monkeypatch.setattr(process, "kill", lambda: signals.append("kill"))
    assert mw._mask_gpu_count_for_controls() is None
    reader.ready = True
    assert mw._mask_gpu_count_for_controls() == 3
    assert mw._CONTROL_GPU_PROBE is not None and not process.closed
    assert reader.closed and signals == ["terminate"]
    now[0] += 0.5
    assert mw._mask_gpu_count_for_controls() == 3
    assert signals == ["terminate"]
    now[0] += 0.6
    assert mw._mask_gpu_count_for_controls() == 3
    assert signals == ["terminate", "kill"]
    assert mw._CONTROL_GPU_PROBE is not None and not process.closed
    now[0] += 2
    assert mw._mask_gpu_count_for_controls() == 3
    assert signals == ["terminate", "kill"]
    process.stopped = True
    assert mw._mask_gpu_count_for_controls() == 3
    assert mw._CONTROL_GPU_PROBE is None and process.closed
    assert all(timeout == 0 for timeout in process.joins)


def test_closing_the_panel_does_not_cancel_discovery_timeout(qtbot, qapp, monkeypatch):
    reader, _, process, _ = _fake_probe(monkeypatch)
    owner = QWidget()
    panel = sm.SettingsWidgets("mask", parent=owner)
    panel.build_sections()
    assert mw._CONTROL_GPU_TIMER.parent() is qapp
    started = mw._CONTROL_GPU_PROBE['started']
    with qtbot.waitSignal(owner.destroyed):
        owner.deleteLater()
    monkeypatch.setattr(mw.time, "monotonic", lambda: started + 61)
    qtbot.waitUntil(lambda: mw._CONTROL_GPU_PROBE is None)
    assert mw._CONTROL_GPU_COUNT == [0]
    assert reader.closed and process.closed
    assert not mw._CONTROL_GPU_TIMER.isActive()


def test_application_exit_kills_and_reaps_pending_discovery(qapp, monkeypatch):
    reader, _, process, _ = _fake_probe(monkeypatch)
    signals = []
    monkeypatch.setattr(process, "terminate", lambda: signals.append("terminate"))
    assert mw._mask_gpu_count_for_controls() is None
    mw._watch_mask_gpu_probe()
    qapp.aboutToQuit.emit()
    assert signals == ["terminate"]
    assert reader.closed and process.closed
    assert process.joins == [0.2, 0.2, 0]
    assert mw._CONTROL_GPU_PROBE is None and mw._CONTROL_GPU_COUNT == [0]
    assert not mw._CONTROL_GPU_TIMER.isActive()


def test_child_counts_compatible_devices_and_closes_pipe(monkeypatch):
    from types import SimpleNamespace

    sent = []
    closed = []
    monkeypatch.setattr(mw, "_compatible_mask_gpus", lambda: (1, 2, 3))
    mw._probe_mask_gpu_count(SimpleNamespace(send=sent.append, close=lambda: closed.append(True)))
    assert sent == [3] and closed == [True]


def test_pending_gpu_discovery_refreshes_the_controls(qtbot, monkeypatch):
    panel, parallel, indices = _panel(qtbot, monkeypatch, None)
    assert not parallel.isEnabled() and not indices.isEnabled()
    assert "Checking compatible GPUs" in str(parallel.property(NOTE))
    monkeypatch.setattr(mw, "_mask_gpu_count_for_controls", lambda: 2)
    qtbot.waitUntil(parallel.isEnabled)
    assert not parallel.property(NOTE)
    panel.set_value_for_key("mask_parallel", True)
    assert indices.isEnabled()


def test_real_discovery_finishes_outside_the_gui_process():
    import os
    import subprocess
    import sys

    code = """
import sys
import time
from PySide6.QtWidgets import QApplication, QWidget
from spacr import _mask_workers
from spacr.qt.screens.settings_model import SettingsWidgets
app = QApplication([])
owner = QWidget()
panel = SettingsWidgets('mask', parent=owner)
panel.build_sections()
deadline = time.monotonic() + 45
while (not _mask_workers._CONTROL_GPU_COUNT or
       _mask_workers._CONTROL_GPU_PROBE is not None) and time.monotonic() < deadline:
    app.processEvents()
    time.sleep(.01)
assert _mask_workers._CONTROL_GPU_COUNT == [0], _mask_workers._CONTROL_GPU_COUNT
assert _mask_workers._CONTROL_GPU_PROBE is None
assert 'torch' not in sys.modules
assert not panel._widgets['mask_parallel'].isEnabled()
owner.deleteLater()
app.processEvents()
"""
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="",
                       ROCR_VISIBLE_DEVICES="", QT_QPA_PLATFORM="offscreen")
    result = subprocess.run([sys.executable, "-c", code], env=environment,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr[-3000:]


def test_per_gpu_progress_reaches_the_label(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    class Host:
        _gpu_progress = QLabel()

    host = Host()
    qtbot.addWidget(host._gpu_progress)
    AppScreen._show_mask_gpu_progress(host, "unrelated output\n")
    assert host._gpu_progress.text() == ""
    state = {"total_batches": 5, "completed_batches": ["a", "b"], "workers": {
        0: {"state": "running", "completed": 2, "total": 3},
        1: {"state": "failed", "completed": 0, "total": 2}}}
    AppScreen._show_mask_gpu_progress(host, "noise\n" + mw._progress_line("nucleus", state) + "\n")
    text = host._gpu_progress.text()
    assert "2/5 nucleus batches done, 1 failed" in text
    assert "GPU 0: 2/3 running" in text and "GPU 1: 0/2 failed" in text


def test_cluster_distribution_submits_the_allocated_gpus(qtbot, qt_theme_applied, tmp_path):
    from tests.qt.test_distributed_jobs_screen import Runner, _manager
    from spacr.qt.screens.distributed_jobs import DistributedJobsScreen
    from spacr.remote_execution import CommandResult

    manager = _manager(tmp_path, Runner(CommandResult(0, "c-1\n"), CommandResult(0, "c-2\n")))
    sent = []
    original = manager.submit
    manager.submit = lambda module, settings, profile: sent.append(dict(settings)) or original(
        module, settings, profile)
    screen = DistributedJobsScreen(manager=manager, threaded=False, auto_poll=False)
    qtbot.addWidget(screen)
    screen.configure_submission("measure", {"src": "/p"})
    assert not screen._allocated_gpus.isEnabled()
    screen.configure_submission("mask", {"src": "/p", "mask_gpu_indices": "3,4"})
    assert screen._allocated_gpus.isEnabled() and not screen._allocated_gpus.isChecked()
    screen.submit()
    assert "mask_parallel" not in sent[-1] and sent[-1]["mask_gpu_indices"] == "3,4"
    screen._allocated_gpus.setChecked(True)
    screen.submit()
    assert sent[-1]["mask_parallel"] is True and sent[-1]["mask_gpu_indices"] == ""
    assert "mask_parallel" not in screen._settings_snapshot
