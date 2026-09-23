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


def test_bundled_map_supplies_every_home_pathway_and_folded_module_name():
    from spacr.qt.widgets import sample_project as sp

    data = json.loads(sp.MAP_FILE.read_text())
    offered = sp.pathways()
    assert [entry["id"] for entry in offered] == list(data["pathways"])
    for entry in offered:
        route = data["pathways"][entry["id"]]
        assert entry["home_app"] == route["home_app"]
        assert entry["modules"] == [step["module"] for step in route["steps"]]
        assert entry["module_names"] == [
            data["modules"][step["module"]]["name"] for step in route["steps"]
        ]
        assert entry["summary"] == route["steps"][0]["action"]


@pytest.mark.parametrize("language", ["sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr"])
def test_sample_project_pathways_are_translated(qtbot, monkeypatch, language):
    from spacr.qt import i18n
    from spacr.qt.widgets import sample_project as sp

    monkeypatch.setenv(i18n.ENV_LANGUAGE, language)
    dialog = sp.SampleProjectDialog()
    qtbot.addWidget(dialog)
    for index, entry in enumerate(sp.pathways()):
        dialog.list.setCurrentRow(index)
        assert dialog.list.item(index).text() == i18n.tr(entry["title"], language)
        assert dialog.summary.text() == i18n.tr(entry["summary"], language)
        assert dialog.list.item(index).text() != entry["title"]
        assert dialog.summary.text() != entry["summary"]
        for name in entry["module_names"]:
            assert i18n.tr(name, language) in dialog.steps.text()


def test_each_sample_keeps_its_home_walkthrough_one_click_away(qtbot, monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QDialog, QMainWindow, QWidget
    from spacr.qt.widgets import sample_project as sp

    window = QMainWindow()
    qtbot.addWidget(window)
    window.resize(1200, 900)
    window.setCentralWidget(QWidget())
    window.statusBar()
    navigated, started = [], []
    window._on_nav_selected = navigated.append
    window.show()
    monkeypatch.setattr(sp, "start_example", lambda screen: started.append(screen))
    screen = QWidget(window)
    for index, entry in enumerate(sp.pathways()):
        def choose(dialog):
            dialog.list.setCurrentRow(index)
            dialog.accept()
            return QDialog.Accepted

        monkeypatch.setattr(sp.SampleProjectDialog, "exec", choose)
        assert sp.offer_a_sample_project(window, lambda key: navigated.append(key) or screen) == entry["modules"][0]
        assert navigated[-1] == entry["modules"][0]
        assert started[-1] is screen
        button = window._sample_pathway_button
        assert button.property("workflowPathway") == entry["id"]
        assert button.isVisible()
        qtbot.mouseClick(button, Qt.LeftButton)
        assert navigated[-1] == "__home__"
        assert window._pathway_overlay._title_lbl.text() == "Home"
        qtbot.mouseClick(window._pathway_overlay._next_btn, Qt.LeftButton)
        assert navigated[-1] == entry["home_app"]
        window._pathway_overlay._skip()
    assert len(window.findChildren(type(button), "SamplePathwayWalkthrough")) == 1


def test_ops_sample_opens_mask_ops_page_and_loads_its_own_example(qtbot, monkeypatch):
    from PySide6.QtWidgets import QDialog
    from spacr.qt.app import MainWindow
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.screens.mask import ops_page
    from spacr.qt.widgets import sample_project as sp

    loaded = []
    monkeypatch.setattr(AppScreen, "load_the_ops_example", lambda self: loaded.append(self))
    window = MainWindow()
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.show()

    def choose(dialog):
        index = next(i for i, entry in enumerate(dialog._entries) if entry["id"] == "optical_screen")
        dialog.list.setCurrentRow(index)
        assert "Align" not in dialog.steps.text()
        dialog.accept()
        return QDialog.Accepted

    monkeypatch.setattr(sp.SampleProjectDialog, "exec", choose)
    assert window._start_a_sample_project() == "ops"
    host = window._screens["mask"]
    page = ops_page(host).page
    assert host._ops_switch.isChecked()
    assert page is not None and page.app_key == "ops"
    qtbot.waitUntil(page.isVisible)
    assert loaded == [page]
    assert "align" not in window._screens
    assert "ops" not in window._screens
    assert window._sample_pathway_button.property("workflowPathway") == "optical_screen"
    host._fold_pages.setCurrentIndex(0)
    window._start_a_sample_project()
    assert loaded == [page, page]
    assert ops_page(host).page is page
    assert host._fold_pages.currentWidget() is page


def test_saved_ops_settings_return_to_the_ops_form_not_its_mask_host(qtbot, monkeypatch, tmp_path):
    from spacr import restart_state
    from spacr.qt.app import MainWindow
    from spacr.qt.screens.mask import ops_page

    monkeypatch.setenv("SPACR_HOME", str(tmp_path))
    window = MainWindow()
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.show()
    seed = {"genotype_source": str(tmp_path / "cycles")}
    window._on_train_requested("ops", seed)
    page = ops_page(window._screens["mask"]).page
    assert page._settings_model.collect()["genotype_source"] == seed["genotype_source"]
    seed["genotype_source"] = str(tmp_path / "resumed-cycles")
    restart_state.save(module="ops", settings=seed)
    assert window.resume_after_restart() == "mask"
    assert page._settings_model.collect()["genotype_source"] == seed["genotype_source"]


def test_ops_sample_does_not_offer_generic_measurement_crops():
    from types import SimpleNamespace
    from spacr.qt.widgets.sample_project import start_example

    calls = []
    page = SimpleNamespace(app_key="ops", _test_data_apply=object(),
                           load_the_ops_example=lambda: calls.append("ops"),
                           _choose_the_test_data=lambda: calls.append("wrong"))
    assert start_example(page) == "test data"
    assert calls == ["ops"]


def test_walkthrough_button_fits_a_status_bar_created_before_font_scaling(qtbot):
    from PySide6.QtWidgets import QMainWindow
    from spacr.qt.widgets import sample_project as sp

    window = QMainWindow()
    qtbot.addWidget(window)
    window.statusBar().setFixedHeight(17)
    font = window.font()
    font.setPixelSize(22)
    window.setFont(font)
    window.resize(1200, 900)
    window.show()
    sp._offer_pathway_walkthrough(window, sp.pathways()[0])
    button = window._sample_pathway_button
    qtbot.waitUntil(lambda: button.height() >= button.sizeHint().height())
    assert button.height() >= button.fontMetrics().height()


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
