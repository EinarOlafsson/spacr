"""The Cellpose workbench page: its guards, and where the checkpoint comes from.

The screen itself is built for real -- both module pages, both settings
models -- and every failure is injected into the one call that can fail, so
the guard is exercised by the thing it guards against.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens import train_cellpose as tc  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture()
def workbench(qtbot):
    screen = tc.CellposeWorkbenchScreen()
    qtbot.addWidget(screen)
    return screen


# ---------------------------------------------------------------------------
# Which settings cross between the tabs
# ---------------------------------------------------------------------------

def test_the_carried_keys_come_from_the_propagation_map():
    keys = tc.carried_setting_keys()
    assert keys
    assert "model_name" not in keys
    assert "src" not in keys


def test_an_unreadable_propagation_map_carries_nothing(monkeypatch):
    """A broken registry costs the carry, not the screen."""
    import sys

    monkeypatch.setitem(sys.modules, "spacr.qt.preview_registry", None)
    assert tc.carried_setting_keys() == ()


def test_an_apply_key_the_map_does_not_know_carries_nothing(monkeypatch):
    from spacr.qt import preview_registry

    monkeypatch.setattr(preview_registry, "PREVIEWS", {})
    assert tc.carried_setting_keys() == ()


def test_a_seam_that_will_not_install_does_not_cost_the_screen(
        qtbot, monkeypatch):
    """Search strip, recipes button and preview each fail on their own."""
    import sys

    for name in ("spacr.qt.settings_search", "spacr.qt.recipes",
                 "spacr.qt.preview_registry"):
        monkeypatch.setitem(sys.modules, name, None)
    screen = tc.CellposeWorkbenchScreen()
    qtbot.addWidget(screen)
    assert len(screen._screens) == 2
    assert screen.active_app_key() == tc.TRAIN_KEY


# ---------------------------------------------------------------------------
# Which tab is being asked
# ---------------------------------------------------------------------------

def test_a_key_neither_tab_runs_has_no_screen(workbench):
    assert workbench.screen_for(tc.TRAIN_KEY) is workbench.train_screen
    assert workbench.screen_for(tc.APPLY_KEY) is workbench.apply_screen
    assert workbench.screen_for("cellpose_dreams") is None


def test_the_shared_tooling_gets_the_visible_tabs_settings(workbench):
    assert workbench._settings_model is workbench.train_screen._settings_model
    key, values = workbench.current_settings()
    assert key == tc.TRAIN_KEY
    assert isinstance(values, dict) and values

    workbench._tabs.setCurrentIndex(1)
    assert workbench._settings_model is workbench.apply_screen._settings_model
    key, values = workbench.current_settings()
    assert key == tc.APPLY_KEY


def test_an_empty_settings_dict_is_taken_nowhere(workbench):
    assert workbench.apply_settings_dict({}) == 0
    assert workbench.apply_settings_dict(None) == 0


def test_a_seed_lands_in_the_tab_that_has_the_keys(workbench):
    taken = workbench.apply_seed({"model_name": "my_model"})
    assert taken >= 1
    assert workbench.active_app_key() == tc.TRAIN_KEY


# ---------------------------------------------------------------------------
# Carrying between the tabs
# ---------------------------------------------------------------------------

def test_a_carry_that_raises_is_logged_not_thrown(workbench, monkeypatch):
    """Switching tab must still switch tab when the copy fails."""
    def explode(_source, _target):
        raise RuntimeError("settings model is gone")

    monkeypatch.setattr(workbench, "carry", explode)
    workbench._tabs.setCurrentIndex(1)
    assert workbench._tabs.currentIndex() == 1
    assert workbench.active_app_key() == tc.APPLY_KEY


def test_settings_that_cannot_be_read_carry_nothing(workbench, monkeypatch):
    source, target = workbench.train_screen, workbench.apply_screen

    def explode():
        raise RuntimeError("cannot collect")

    monkeypatch.setattr(source._settings_model, "collect", explode)
    monkeypatch.setattr(workbench, "carry_trained_model", lambda: "")
    assert workbench.carry(source, target) == {}


def test_a_header_with_no_instruction_label_is_not_an_error(workbench,
                                                            monkeypatch):
    monkeypatch.setattr(workbench._header, "instruction_label", None,
                        raising=False)
    workbench._sync_instruction()
    assert workbench._tabs.currentIndex() == 0


# ---------------------------------------------------------------------------
# The trained checkpoint
# ---------------------------------------------------------------------------

def test_a_training_form_that_cannot_be_read_offers_no_checkpoint(
        workbench, monkeypatch):
    def explode():
        raise RuntimeError("cannot collect")

    monkeypatch.setattr(workbench.train_screen._settings_model, "collect",
                        explode)
    assert workbench.trained_checkpoint() == ""


def test_without_a_source_or_a_name_there_is_no_checkpoint(workbench,
                                                           monkeypatch):
    monkeypatch.setattr(workbench.train_screen._settings_model, "collect",
                        lambda: {"src": "", "model_name": "m"})
    assert workbench.trained_checkpoint() == ""
    monkeypatch.setattr(workbench.train_screen._settings_model, "collect",
                        lambda: {"src": "/tmp", "model_name": ""})
    assert workbench.trained_checkpoint() == ""


def test_the_finished_model_wins_over_the_epoch_snapshots(workbench,
                                                          monkeypatch, tmp_path):
    folder = tmp_path.joinpath(*tc.CHECKPOINT_DIR)
    folder.mkdir(parents=True)
    finished = folder / "my_model_final"
    snapshot = folder / f"my_model_{tc.EPOCH_INFIX}5"
    for path in (snapshot, finished):
        path.write_bytes(b"weights")
    monkeypatch.setattr(
        workbench.train_screen._settings_model, "collect",
        lambda: {"src": str(tmp_path), "model_name": "my_model"})
    assert workbench.trained_checkpoint() == str(finished)


def test_a_checkpoint_whose_time_cannot_be_read_is_still_offered(
        workbench, monkeypatch, tmp_path):
    """The newest wins; when nothing has a time, the last name does."""
    folder = tmp_path.joinpath(*tc.CHECKPOINT_DIR)
    folder.mkdir(parents=True)
    for name in ("my_model_a", "my_model_b"):
        (folder / name).write_bytes(b"weights")
    monkeypatch.setattr(
        workbench.train_screen._settings_model, "collect",
        lambda: {"src": str(tmp_path), "model_name": "my_model"})

    def no_time(_path):
        raise OSError("stat is unavailable")

    monkeypatch.setattr(os.path, "getmtime", no_time)
    assert workbench.trained_checkpoint() == str(folder / "my_model_b")


def test_a_source_with_no_checkpoint_folder_offers_nothing(workbench,
                                                           monkeypatch,
                                                           tmp_path):
    monkeypatch.setattr(
        workbench.train_screen._settings_model, "collect",
        lambda: {"src": str(tmp_path), "model_name": "my_model"})
    assert workbench.trained_checkpoint() == ""
