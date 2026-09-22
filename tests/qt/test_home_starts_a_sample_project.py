"""Home offers a sample project of the kind the person has (GitHub #130)."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def test_the_pathways_come_from_the_shared_map_when_it_is_there(tmp_path):
    from spacr.qt.widgets import sample_project as sp

    assert [p["id"] for p in sp.pathways(tmp_path / "missing.json")] == [
        "spacr_screen", "high_content", "train_model"]

    written = tmp_path / "module_workflows.json"
    written.write_text(json.dumps({"pathways": [
        {"id": "ops", "title": "An optical pooled screen",
         "summary": "s", "modules": ["align", "regression"]}]}))
    (only,) = sp.pathways(written)
    assert only["id"] == "ops" and only["modules"][0] == "align"


def test_choosing_a_pathway_opens_its_first_module_with_example_data(qtbot):
    from spacr.qt.widgets import sample_project as sp
    from PySide6.QtWidgets import QDialog

    opened, started = [], []

    class Screen:
        _test_data_apply = object()

    def fake_exec(self):
        self.list.setCurrentRow(1)
        self.accept()
        return QDialog.Accepted

    original_exec = sp.SampleProjectDialog.exec
    sp.SampleProjectDialog.exec = fake_exec
    sp_load = sp.start_example
    sp.start_example = lambda screen: started.append(screen) or "test data"
    try:
        key = sp.offer_a_sample_project(
            None, lambda k: opened.append(k) or Screen())
    finally:
        sp.SampleProjectDialog.exec = original_exec
        sp.start_example = sp_load
    assert key == "mask" and opened == ["mask"], "the pathway's first module"
    assert started, "and its example data was asked for"


def test_cancelling_opens_nothing(qtbot):
    from spacr.qt.widgets import sample_project as sp
    from PySide6.QtWidgets import QDialog

    original = sp.SampleProjectDialog.exec
    sp.SampleProjectDialog.exec = lambda self: QDialog.Rejected
    try:
        assert sp.offer_a_sample_project(None, lambda k: None) == ""
    finally:
        sp.SampleProjectDialog.exec = original


def test_home_has_the_button_and_asks_the_window_for_it(qtbot):
    from spacr.qt.app import make_home_page

    page = make_home_page()
    qtbot.addWidget(page)
    asked = []
    page.sample_project_requested.connect(lambda: asked.append(True))
    page._sample_project_button.click()
    assert asked, "the button asks the window to offer a sample project"
