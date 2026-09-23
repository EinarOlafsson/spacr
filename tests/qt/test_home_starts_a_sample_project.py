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
    assert page._sample_project_button.text() == "Pipeline overviews"
    asked = []
    page.sample_project_requested.connect(lambda: asked.append(True))
    page._sample_project_button.click()
    assert asked, "the button asks the window to offer a sample project"


@pytest.mark.parametrize("contents", [
    "{", "null", "[]", '{"pathways": []}',
    '{"pathways": [{"title": "Incomplete"}]}',
    '{"pathways": {"broken": {"title": "Broken", "steps": '
    '[{"module": "absent", "action": "Open"}]}}, "modules": {}}',
])
def test_unusable_workflow_maps_keep_the_fallback_routes_available(tmp_path, contents):
    from spacr.qt.widgets import sample_project as sp

    path = tmp_path / "workflows.json"
    path.write_text(contents, encoding="utf-8")
    routes = sp.pathways(path)
    assert [route["id"] for route in routes] == [
        "spacr_screen", "high_content", "train_model"]
    assert all(route["title"] and route["modules"] for route in routes)


@pytest.mark.parametrize("kind, expected", [
    ("chooser", "chooser"), ("measurements", "test data"),
    ("unavailable", ""), ("noncallable_chooser", "test data"),
])
def test_example_loader_dispatches_once_without_falling_through(monkeypatch, kind, expected):
    from types import SimpleNamespace
    from spacr.qt.widgets import measurements_example
    from spacr.qt.widgets import sample_project as sp

    calls = []
    screen = SimpleNamespace()
    monkeypatch.setattr(measurements_example, "load_test_data",
                        lambda target: calls.append(("measurements", target)))
    if kind != "unavailable":
        screen._test_data_apply = object()
    if kind == "chooser":
        screen._choose_the_test_data = lambda: calls.append(("chooser", screen))
    elif kind == "noncallable_chooser":
        screen._choose_the_test_data = "not a loader"
    assert sp.start_example(screen) == expected
    assert calls == ([] if not expected else [
        ("chooser" if kind == "chooser" else "measurements", screen)])


def test_empty_pipeline_dialog_cannot_start_an_example(qtbot, monkeypatch):
    from PySide6.QtWidgets import QDialogButtonBox
    from spacr.qt.widgets import sample_project as sp

    dialog = sp.SampleProjectDialog(entries=[])
    qtbot.addWidget(dialog)
    chosen = []
    dialog.chosen.connect(chosen.append)
    assert dialog.selected() is None
    assert dialog.summary.text() == dialog.steps.text() == ""
    assert not dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok).isEnabled()
    dialog.accept()
    assert chosen == []

    def accept_empty(target):
        qtbot.addWidget(target)
        target.accept()
        return target.result()

    monkeypatch.setattr(sp.SampleProjectDialog, "exec", accept_empty)
    assert sp.offer_a_sample_project(None, chosen.append, entries=[]) == ""
    assert chosen == []


def test_unavailable_pipeline_destination_does_not_load_data_or_offer_walkthrough(qtbot, monkeypatch):
    from PySide6.QtWidgets import QMainWindow
    from spacr.qt.widgets import sample_project as sp

    window = QMainWindow()
    qtbot.addWidget(window)
    opened, loaded = [], []

    def accept_first(dialog):
        dialog.accept()
        return dialog.result()

    monkeypatch.setattr(sp.SampleProjectDialog, "exec", accept_first)
    monkeypatch.setattr(sp, "start_example", loaded.append)
    entry = sp.pathways()[0]
    assert sp.offer_a_sample_project(window, opened.append, [entry]) == entry["modules"][0]
    assert opened == [entry["modules"][0]]
    assert loaded == []
    assert not hasattr(window, "_sample_pathway_button")


def test_legacy_pipeline_without_walkthrough_does_not_add_a_dead_button(qtbot):
    from PySide6.QtWidgets import QMainWindow
    from spacr.qt.widgets import sample_project as sp

    window = QMainWindow()
    qtbot.addWidget(window)
    sp._offer_pathway_walkthrough(window, dict(sp.FALLBACK[0]))
    sp._offer_pathway_walkthrough(None, sp.pathways()[0])
    assert not hasattr(window, "_sample_pathway_button")


@pytest.mark.parametrize("contents", [None, "{", "null", "[]", "42", '{"modules": []}'])
def test_fallback_pipeline_diagrams_remain_selectable_when_graph_map_is_unreadable(qtbot, monkeypatch, tmp_path, contents):
    from PySide6.QtCore import Qt
    from spacr.qt.widgets import sample_project as sp
    from spacr.qt.widgets import workflow_diagram as wd

    path = tmp_path / "map.json"
    if contents is not None:
        path.write_text(contents, encoding="utf-8")
    with pytest.raises(OSError if contents is None else ValueError):
        wd.workflow_map(path)
    monkeypatch.setattr(sp, "workflow_map", lambda: wd.workflow_map(path))
    entries = [dict(route) for route in sp.FALLBACK]
    dialog = sp.SampleProjectDialog(entries=entries)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitExposed(dialog)
    dialog.list.scrollToItem(dialog.list.item(1))
    graph = dialog.diagrams[1]
    graph.fit_diagram()
    node = graph.nodes[entries[1]["modules"][0]]
    point = graph.mapFromScene(node.sceneBoundingRect().center())
    qtbot.mouseClick(graph.viewport(), Qt.LeftButton, pos=point)
    qtbot.waitUntil(lambda: dialog.selected()["id"] == entries[1]["id"])
    assert dialog.summary.text() == entries[1]["summary"]
    assert "Inputs" in dialog.details.toPlainText()
    assert "Outputs" in dialog.details.toPlainText()
    assert set(graph.nodes) == set(entries[1]["modules"])
    assert {(link["from"], link["to"]) for link in graph.links} == set(
        zip(entries[1]["modules"], entries[1]["modules"][1:]))
