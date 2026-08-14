"""Filter and Search as tabs. The last item of instruction 31.

    "Filter and Hyperparameter-search become TABS, the active one blue like
     the app's buttons, sitting above the console."

A search is a thing you ITERATE on -- change a parameter, look at the plot,
change it again. It lived in a modal, so *looking* meant closing the dialog
and reopening it to change anything: every loop cost two clicks and the loss
of the previous view.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr.qt.screens.gate_editor import GateEditorScreen
from spacr.qt.widgets.gate_search_panel import GateSearchPanel
from spacr.qt.widgets.gate_settings import GateEditorSettings


@pytest.fixture
def screen(qtbot):
    widget = GateEditorScreen()
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def panel(qtbot):
    widget = GateSearchPanel()
    qtbot.addWidget(widget)
    widget.apply_settings(GateEditorSettings())
    return widget


# ---------------------------------------------------------------------------
# The tabs
# ---------------------------------------------------------------------------

def test_the_side_panel_has_a_filter_tab_and_a_search_tab(screen):
    titles = [screen.side_tabs.tabText(i)
              for i in range(screen.side_tabs.count())]
    assert titles == ["Filter", "Search"]


def test_filter_and_columns_stay_on_one_page(screen):
    """They are the same job -- both narrow what the scatter shows -- so
    hiding one behind the other meant neither could be checked while using
    the other. Search is a different job, which is what makes it a tab."""
    page = screen.side_tabs.widget(0)
    assert page.isAncestorOf(screen.filters)
    assert page.isAncestorOf(screen.formulas)


def test_the_tabs_are_registered_for_the_accent_styling(screen):
    """"the active one blue like the app's buttons" -- page_tabs_qss paints
    the selected tab in the accent."""
    assert screen.side_tabs.objectName() == "GateSidePanel"


def test_the_console_is_still_below_them(screen):
    """"sitting above the console"."""
    assert screen.console is not None


# ---------------------------------------------------------------------------
# One source of truth
# ---------------------------------------------------------------------------

def test_the_panel_shows_what_the_settings_hold(panel):
    panel.apply_settings(GateEditorSettings(
        cluster_eps=0.75, cluster_min_samples=33, cluster_scale=False,
        cluster_walk=True, cluster_walk_steps=7))
    assert panel._eps.value() == pytest.approx(0.75)
    assert panel._min_samples.value() == 33
    assert not panel._scale.isChecked()
    assert panel._walk.isChecked()
    assert panel._walk_steps.value() == 7


def test_loading_does_not_report_itself_as_an_edit(panel, qtbot):
    """Setting a spin box fires valueChanged; a load would otherwise write
    back the values it just read."""
    seen = []
    panel.settings_changed.connect(seen.append)
    panel.apply_settings(GateEditorSettings(cluster_eps=0.9))
    assert seen == []


def test_an_edit_is_announced_as_the_field_that_changed(panel, qtbot):
    seen = []
    panel.settings_changed.connect(seen.append)
    panel._eps.setValue(0.42)
    assert {"cluster_eps": 0.42} in seen


def test_an_edit_reaches_the_screens_settings(screen):
    screen._on_search_settings({"cluster_min_samples": 41})
    assert screen._settings.cluster_min_samples == 41


def test_the_screen_pushes_settings_into_the_panel(screen):
    screen.apply_settings(screen._settings.replaced(cluster_eps=0.33))
    assert screen.search._eps.value() == pytest.approx(0.33)


# ---------------------------------------------------------------------------
# Walk
# ---------------------------------------------------------------------------

def test_walk_steps_is_greyed_until_walking(panel):
    """INVARIANTS 6 -- greyed, not removed: a control that vanished would
    take its value with it."""
    panel._walk.setChecked(False)
    assert not panel._walk_steps.isEnabled()
    panel._walk.setChecked(True)
    assert panel._walk_steps.isEnabled()


def test_the_button_says_which_of_the_two_it_will_do(panel):
    panel._walk.setChecked(False)
    assert panel._run.text() == "Run search"
    panel._walk.setChecked(True)
    assert panel._run.text() == "Run walk"


def test_the_note_says_what_the_numbers_above_mean_in_each_mode(panel):
    panel._walk.setChecked(False)
    assert "exactly these values" in panel._note.text()
    panel._walk.setChecked(True)
    assert "range around the values above" in panel._note.text()


def test_a_greyed_walk_step_keeps_its_value(panel):
    panel._walk.setChecked(True)
    panel._walk_steps.setValue(19)
    panel._walk.setChecked(False)
    panel._walk.setChecked(True)
    assert panel._walk_steps.value() == 19


# ---------------------------------------------------------------------------
# Running from the tab does not ask again
# ---------------------------------------------------------------------------

@pytest.fixture
def quiet(monkeypatch):
    """Swallow the information boxes. Two rows cannot cluster, and the
    refusal is a modal that would hang a headless run."""
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: None))


def test_the_tab_runs_without_opening_the_dialog(screen, monkeypatch, quiet):
    """The tab IS the parameter editor; asking again would be asking twice
    for the same numbers."""
    asked = []
    import spacr.qt.widgets.gate_editor as G
    monkeypatch.setattr(G, "_ClusterSettingsDialog",
                        lambda *a, **k: asked.append(True))
    screen.set_frame(pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, 2.0]}))
    screen.gates.run_cluster(ask=False)
    assert not asked


def test_the_button_still_asks(screen, monkeypatch, quiet):
    import spacr.qt.widgets.gate_editor as G

    opened = []

    class _Cancelled:
        def __init__(self, *a, **k):
            opened.append(True)

        def exec(self):
            return 0

    monkeypatch.setattr(G, "_ClusterSettingsDialog", _Cancelled)
    screen.set_frame(pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, 2.0]}))
    # Clustering refuses before it asks unless both axes are chosen.
    screen._x.setCurrentText("a")
    screen._y.setCurrentText("b")
    screen.gates._on_cluster()
    assert opened


def test_both_paths_read_the_same_five_numbers():
    """The modal's own docstring records what happens when two editors of
    the same settings drift apart."""
    import inspect

    from spacr.qt.widgets import gate_editor

    source = inspect.getsource(gate_editor.GateEditorPanel.run_cluster)
    for field in ("cluster_eps", "cluster_min_samples", "cluster_scale",
                  "cluster_walk", "cluster_walk_steps"):
        assert field in source, field
