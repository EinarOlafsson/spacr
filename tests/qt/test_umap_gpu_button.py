"""The GPU button, left of the search button. Instruction 95.

    "the option to suppurt GPU acceleration via cuML should the to the left of
     the hyperparamiter search button and should simply say GPU. if pressed it
     should activate the GPU cuML version, if dependencies are not installed
     then it should check the python version and install if compatible other
     wise tell the user what they need to do."
"""
from __future__ import annotations

import pytest

from spacr.qt.screens.hyperparam import HyperparamPanel


@pytest.fixture
def panel(qtbot):
    widget = HyperparamPanel()
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


def test_the_button_says_just_gpu(panel):
    assert panel._gpu_btn.text() == "GPU"


def test_it_sits_left_of_the_search_button(panel):
    from PySide6.QtWidgets import QGridLayout

    # The buttons live in a QGridLayout nested inside the panel, so find the
    # grid that holds them rather than assuming the parent's own layout is it.
    grid = next(g for g in panel.findChildren(QGridLayout)
                if g.indexOf(panel._run_btn) >= 0)
    gpu_col = grid.getItemPosition(grid.indexOf(panel._gpu_btn))[1]
    run_col = grid.getItemPosition(grid.indexOf(panel._run_btn))[1]
    assert gpu_col < run_col, "GPU must be to the LEFT of Run search"


def test_it_is_a_state_and_not_an_action(panel):
    """It says which backend the NEXT search runs under."""
    assert panel._gpu_btn.isCheckable()


def test_the_backend_follows_the_button(panel, monkeypatch):
    _plan(monkeypatch, "ready")
    panel._gpu_btn.setChecked(True)
    assert panel.gpu_backend() == "cuml"
    panel._gpu_btn.setChecked(False)
    assert panel.gpu_backend() == "cpu"


def test_installed_but_unwanted_still_means_cpu(panel, monkeypatch):
    """cuML being importable is not the same as the user asking for it."""
    _plan(monkeypatch, "ready")
    panel._refresh_gpu_button()
    assert panel.gpu_backend() == "cpu"


def test_a_wrong_interpreter_says_what_is_needed(panel, monkeypatch, quiet):
    _plan(monkeypatch, "wrong_python", "make a 3.11 environment")
    panel._gpu_btn.setChecked(True)
    panel._on_gpu_clicked()
    assert not panel._gpu_btn.isChecked(), "the toggle must not claim it is on"
    assert quiet, "the user was told nothing"


def test_a_missing_device_is_not_offered_an_install(panel, monkeypatch, quiet):
    """Installing again cannot conjure a GPU."""
    installed = []
    monkeypatch.setattr(panel, "_install_cuml", lambda: installed.append(True))
    _plan(monkeypatch, "no_device")
    panel._on_gpu_clicked()
    assert not installed


def test_declining_the_install_installs_nothing(panel, monkeypatch, quiet):
    installed = []
    monkeypatch.setattr(panel, "_install_cuml", lambda: installed.append(True))
    _plan(monkeypatch, "install")
    panel._on_gpu_clicked()
    assert not installed


def test_a_ready_backend_just_turns_on(panel, monkeypatch, quiet):
    installed = []
    monkeypatch.setattr(panel, "_install_cuml", lambda: installed.append(True))
    _plan(monkeypatch, "ready", "cuML 25.2 on 1 device(s)")
    panel._gpu_btn.setChecked(True)
    panel._on_gpu_clicked()
    assert panel._gpu_btn.isChecked()
    assert not installed


def test_the_tooltip_carries_the_different_map_caveat(panel, monkeypatch):
    """A cuML map is a different map of the same data, not the same map
    faster -- so rows from the two backends are not comparable."""
    _plan(monkeypatch, "ready")
    panel._refresh_gpu_button()
    assert "DIFFERENT MAP" in panel._gpu_btn.toolTip()


def test_an_install_says_to_restart_rather_than_claiming_it_is_live(
        panel, monkeypatch, quiet):
    """pip can upgrade numpy and scipy underneath a process that has already
    imported them, and this one has."""
    monkeypatch.setattr("subprocess.run", lambda *a, **k: None)
    panel._install_cuml()
    assert "Restart" in panel._status.text() or "restart" in panel._status.text().lower()


def test_a_failed_install_says_so_and_shows_the_command(panel, monkeypatch,
                                                        quiet):
    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr("subprocess.run", boom)
    panel._install_cuml()
    assert "failed" in panel._status.text().lower()
