"""One action-strip GPU toggle controls main Image UMAP and its search."""
from __future__ import annotations

import pytest

from spacr.qt.screens.app_screen import AppScreen


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    widget = AppScreen("umap")
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def quiet(monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    seen = []
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: seen.append(a)))
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: seen.append(a)))
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.No))
    return seen


def _plan(monkeypatch, action, message="msg"):
    monkeypatch.setattr("spacr.gpu_reduce.install_plan",
                        lambda: {"action": action, "message": message})


def test_the_shared_toggle_says_gpu_and_sits_left_of_search(screen):
    row = screen._gpu_switch.parentWidget().layout()
    assert screen._gpu_switch.text() == "GPU"
    assert row.indexOf(screen._gpu_switch) < row.indexOf(screen._hp_switch)
    assert "DIFFERENT MAP" in screen._gpu_switch.toolTip()


def test_reducer_switch_greys_fields_and_their_visible_labels(screen):
    model = screen._settings_model
    tsne = model._widgets["tsne_perplexity"]
    umap = model._widgets["n_neighbors"]
    assert not tsne.isEnabled()
    assert not tsne._spacr_setting_label.isEnabled()
    model.set_value_for_key("reduction_method", "tsne")
    assert not umap.isEnabled()
    assert not umap._spacr_setting_label.isEnabled()
    assert tsne.isEnabled()
    assert tsne._spacr_setting_label.isEnabled()


def test_ready_backend_updates_search_and_main_setting(screen, monkeypatch):
    _plan(monkeypatch, "ready")
    screen._gpu_switch.setChecked(True)
    assert screen._gpu_switch.isChecked()
    assert screen._hyperparam.gpu_backend() == "cuml"
    assert screen._settings_model.collect()["gpu"] is True
    screen._gpu_switch.setChecked(False)
    assert screen._hyperparam.gpu_backend() == "cpu"
    assert screen._settings_model.collect()["gpu"] is False


@pytest.mark.parametrize("action", ["wrong_python", "no_device"])
def test_unavailable_gpu_cannot_leave_the_toggle_claiming_on(
        screen, monkeypatch, quiet, action):
    _plan(monkeypatch, action)
    screen._gpu_switch.setChecked(True)
    assert not screen._gpu_switch.isChecked()
    assert screen._hyperparam.gpu_backend() == "cpu"
    assert screen._settings_model.collect()["gpu"] is False
    assert quiet


def test_declining_install_changes_nothing(screen, monkeypatch, quiet):
    installed = []
    monkeypatch.setattr(
        screen._hyperparam, "_install_cuml", lambda: installed.append(True))
    _plan(monkeypatch, "install")
    screen._gpu_switch.setChecked(True)
    assert not screen._gpu_switch.isChecked()
    assert not installed


def test_install_completion_requires_restart(screen, monkeypatch, quiet):
    monkeypatch.setattr("subprocess.run", lambda *a, **k: None)
    screen._hyperparam._install_cuml()
    assert "restart" in screen._hyperparam._status.text().lower()


def test_failed_install_is_reported(screen, monkeypatch, quiet):
    def boom(*_args, **_kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("subprocess.run", boom)
    screen._hyperparam._install_cuml()
    assert "failed" in screen._hyperparam._status.text().lower()
