"""Pathways use the shared contracts and navigate without running analyses."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QStackedWidget, QWidget
import pytest

from spacr.qt import walkthrough as W


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.navigated = []
        self._screens = {}
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)
        self.menuBar().addMenu("Help")
        self.resize(1200, 900)

    def _on_nav_selected(self, key):
        self.navigated.append(key)
        if key not in self._screens:
            screen = QWidget()
            screen.app_key = key
            screen._settings_model = object()
            self._screens[key] = screen
            self._stack.addWidget(screen)
        self._stack.setCurrentWidget(self._screens[key])


@pytest.mark.parametrize("key", list(W._workflow_map()["pathways"]))
def test_each_route_starts_at_home_and_each_next_opens_the_declared_host(qtbot, key):
    window = Window()
    qtbot.addWidget(window)
    window.show()
    W.install_window_hooks(window)
    data = W._workflow_map()
    route = data["pathways"][key]
    overlay = W.show_walkthrough(window, "pathway:" + key)
    assert window.navigated == ["__home__"]
    assert overlay._title_lbl.text() == "Home"
    for step in route["steps"]:
        qtbot.mouseClick(overlay._next_btn, Qt.LeftButton)
        module = data["modules"][step["module"]]
        assert window.navigated[-1] == (module["parent"] or module["home"])
        assert overlay._body_lbl.text() == step["action"]
        assert len(window.findChildren(W._TourOverlay)) == 1
    overlay._skip()
    assert window._pathway_walkthrough_active is False


def test_help_menu_exposes_all_pathways_and_escape_restores_module_tours(qtbot):
    window = Window()
    qtbot.addWidget(window)
    window.show()
    menu = W.install_help_menu(window)
    actions = {a.property("workflowPathway"): a for a in menu.actions()
               if a.property("workflowPathway")}
    assert set(actions) == set(W._workflow_map()["pathways"])
    actions["pooled_screen"].trigger()
    overlay = window._pathway_overlay
    qtbot.keyClick(overlay, Qt.Key_Escape)
    assert not window._pathway_walkthrough_active
    assert window.navigated == ["__home__"]


def test_unknown_pathway_does_not_change_the_window(qtbot):
    window = Window()
    qtbot.addWidget(window)
    with pytest.raises(KeyError):
        W.show_walkthrough(window, "pathway:missing")
    assert window.navigated == []
    assert not getattr(window, "_pathway_walkthrough_active", False)


def test_delayed_first_run_tour_cannot_cover_a_requested_pathway(qtbot):
    from spacr.qt.first_run import maybe_show_tour
    window = Window()
    qtbot.addWidget(window)
    window.show()
    old = maybe_show_tour(window, force=True)
    assert old is not None
    overlay = W.show_walkthrough(window, "pathway:screen_planning")
    assert not old.isVisible()
    assert maybe_show_tour(window, force=True) is None
    assert overlay.isVisible()
    overlay._skip()


def test_real_home_to_experiment_design_route(qtbot):
    from spacr.qt.app import MainWindow
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1280, 900)
    window.show()
    overlay = W.show_walkthrough(window, "pathway:screen_planning")
    assert overlay._title_lbl.text() == "Home"
    qtbot.mouseClick(overlay._next_btn, Qt.LeftButton)
    qtbot.waitUntil(lambda: "experiment_design" in window._screens, timeout=15000)
    assert window._stack.currentWidget() is window._screens["experiment_design"]
    assert "conditions, controls, replicates" in overlay._body_lbl.text()
    assert overlay.isVisible()
    assert window.rect().contains(overlay._card.geometry())
    from spacr.qt.first_run import maybe_show_tour
    assert maybe_show_tour(window, force=True) is None
    overlay._skip()
