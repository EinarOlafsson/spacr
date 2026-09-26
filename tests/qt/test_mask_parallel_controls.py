"""Item 493: GPU mask-batch controls, per-GPU progress and cluster hand-off."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtWidgets import QLabel                              # noqa: E402

from spacr import _mask_workers as mw                             # noqa: E402
from spacr.qt.screens import settings_model as sm                 # noqa: E402

NOTE = sm._PENDING_NOTE_PROPERTY


def _panel(monkeypatch, count):
    monkeypatch.setattr(mw, "_mask_gpu_count_for_controls", lambda: count)
    panel = sm.SettingsWidgets("mask")
    panel.build_sections()
    return panel, panel._widgets["mask_parallel"], panel._widgets["mask_gpu_indices"]


@pytest.mark.parametrize("count", [0, 1])
def test_without_two_gpus_both_controls_are_greyed_with_the_reason(qtbot, monkeypatch, count):
    panel, parallel, indices = _panel(monkeypatch, count)
    panel._refresh_mask_gpu_enablement()
    for control in (parallel, indices):
        assert not control.isEnabled()
        note = str(control.property(NOTE))
        assert f"{count} found" in note and "Cluster Distribution" in note


def test_with_two_gpus_the_list_follows_the_switch(qtbot, monkeypatch):
    panel, parallel, indices = _panel(monkeypatch, 2)
    panel._refresh_mask_gpu_enablement()
    assert parallel.isEnabled() and not parallel.property(NOTE)
    assert not indices.isEnabled()
    assert "only when mask_parallel is on" in str(indices.property(NOTE))
    panel.set_value_for_key("mask_parallel", True)
    assert indices.isEnabled() and not indices.property(NOTE)
    panel.set_value_for_key("mask_parallel", False)
    assert not indices.isEnabled()


def test_the_defaults_leave_the_feature_off(qtbot, monkeypatch):
    panel, parallel, indices = _panel(monkeypatch, 2)
    assert panel._read_widget(parallel) is False
    assert (panel._read_widget(indices) or "") == ""
    from spacr.settings import set_default_settings_preprocess_generate_masks
    defaults = set_default_settings_preprocess_generate_masks({})
    assert defaults["mask_parallel"] is False and defaults["mask_gpu_indices"] == ""
    from spacr.artifacts import material_settings
    assert "mask_parallel" not in material_settings(defaults)


def test_the_gpu_count_is_read_once_per_process(monkeypatch):
    seen = []
    monkeypatch.setattr(mw, "_CONTROL_GPU_COUNT", [])
    monkeypatch.setattr(mw, "_compatible_mask_gpus", lambda: seen.append(1) or (1, 2, 3))
    assert mw._mask_gpu_count_for_controls() == 3
    assert mw._mask_gpu_count_for_controls() == 3
    assert seen == [1]


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
