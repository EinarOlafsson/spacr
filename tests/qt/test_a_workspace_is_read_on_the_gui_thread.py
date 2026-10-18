"""A run's workspace is read on the GUI thread, whoever saves it.

The run journal saves the workspace from the worker thread that ran the
pipeline, and it asked every registered panel for ``workspace_state()``
there: widgets read off the GUI thread, which caught Regression's results
panel mid-draw in lesson 13. A screen's registration now installs a
dispatcher, every section is read on the GUI thread, and the worker is
handed plain data.
"""
from __future__ import annotations

import threading

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

from spacr import workspace
from spacr.qt import bridge

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def _clean_registry():
    workspace.clear_providers()
    yield
    workspace.clear_providers()


class _Panel:
    """A provider that fails loudly if it is read off the GUI thread."""

    def __init__(self):
        self.read_on = []

    def workspace_state(self):
        on_gui = QThread.currentThread() is QApplication.instance().thread()
        self.read_on.append(on_gui)
        assert on_gui, "workspace_state() ran off the GUI thread"
        return {"level": "gene", "rows": [1, 2, 3]}


def _save_from_a_worker(run_dir):
    result = {}

    def work():
        result["path"] = workspace.save_for_run(
            run_dir, {"save_workspace": "reference"}, app_key="regression")

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    return worker, result


def test_a_worker_save_reads_every_section_on_the_gui_thread(
        qtbot, tmp_path):
    panel = _Panel()
    workspace.register("regression:volcano", lambda: panel)
    assert bridge._install_the_workspace_dispatcher()

    worker, result = _save_from_a_worker(tmp_path)
    qtbot.waitUntil(lambda: not worker.is_alive(), timeout=15000)

    assert panel.read_on == [True]
    doc = workspace.load(tmp_path)
    assert doc["sections"]["regression:volcano"] == {
        "level": "gene", "rows": [1, 2, 3]}
    assert "problems" not in doc
    assert result["path"] is not None


def test_a_screen_registration_installs_the_dispatcher(
        qtbot, qt_theme_applied, tmp_path, monkeypatch):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    read_on = []
    real = RegressionResultsPanel.workspace_state

    def checked(self):
        on_gui = QThread.currentThread() is QApplication.instance().thread()
        read_on.append(on_gui)
        assert on_gui, "workspace_state() ran off the GUI thread"
        return real(self)

    monkeypatch.setattr(RegressionResultsPanel, "workspace_state", checked)
    screen = AppScreen(app_key="regression")
    qtbot.addWidget(screen)
    screen.register_workspace()
    assert screen._results_panel is not None
    assert workspace._STATE_DISPATCHER is bridge._call_on_the_gui_thread

    worker, _result = _save_from_a_worker(tmp_path)
    qtbot.waitUntil(lambda: not worker.is_alive(), timeout=15000)

    assert read_on and all(read_on)
    doc = workspace.load(tmp_path)
    assert doc and "regression:regression" in doc.get("sections", {})


def test_a_gui_that_cannot_answer_costs_the_sections_not_the_run(
        qtbot, tmp_path, monkeypatch):
    panel = _Panel()
    workspace.register("regression:volcano", lambda: panel)
    bridge._install_the_workspace_dispatcher()
    monkeypatch.setattr(bridge, "_GUI_CALL_TIMEOUT_S", 0.05)

    worker, result = _save_from_a_worker(tmp_path)
    worker.join(5)

    assert not worker.is_alive()
    assert result.get("path") is None
    assert panel.read_on == []
    qtbot.wait(50)


def test_on_the_gui_thread_the_sections_are_read_in_place(tmp_path):
    panel = _Panel()
    workspace.register("regression:volcano", lambda: panel)
    bridge._install_the_workspace_dispatcher()

    sections, problems = workspace.section_states(workspace.providers())

    assert sections == {"regression:volcano": {"level": "gene",
                                               "rows": [1, 2, 3]}}
    assert problems == [] and panel.read_on == [True]
