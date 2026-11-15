"""The Cellpose workbench page: its guards, and where the checkpoint comes from.

The screen itself is built for real -- both module pages, both settings
models -- and every failure is injected into the one call that can fail, so
the guard is exercised by the thing it guards against.
"""
from __future__ import annotations


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
