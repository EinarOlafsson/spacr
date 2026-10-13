"""Production contracts for Classify's lazy, live FlowView footer."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from spacr.qt.screens import classify, map_barcodes
from spacr.qt.screens.app_screen import AppScreen

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _restore_trace_state():
    """Keep global FlowView state local to each integration contract."""

    from spacr.flowview import trace

    previous_collector = trace.get_collector()
    previous_enabled = trace.is_enabled()
    yield
    trace.enable(previous_collector)
    if not previous_enabled:
        trace.disable()


def _screen(qtbot, app_key: str = classify.HOST_KEY) -> AppScreen:
    screen = AppScreen(app_key=app_key)
    qtbot.addWidget(screen)
    return screen


def test_classify_mount_costs_no_flowview_import_until_the_fold_opens(tmp_path):
    """A fresh Classify first paint owns a header, not a graphics renderer."""

    script = r"""
import sys
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])
from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.screens import map_barcodes

assert not any(name.startswith("spacr.flowview") for name in sys.modules)
screen = AppScreen("classify_merged")
strip = map_barcodes.install_folds_on(screen)
section = screen._flowview_section
assert strip is not None
assert section.is_expanded() is False
assert section.panel() is None
assert not any(name.startswith("spacr.flowview") for name in sys.modules)
screen.close()
app.processEvents()
assert not any(name.startswith("spacr.flowview") for name in sys.modules)
"""
    environment = {
        **os.environ,
        "QT_QPA_PLATFORM": "offscreen",
        "SPACR_FLOWVIEW": "0",
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
    }
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_open_footer_enables_and_follows_the_live_classify_collector(
    qtbot,
    qapp: QApplication,
    monkeypatch,
):
    """Open, hide, replace-run, collapse and teardown all share one panel."""

    from spacr.flowview import trace
    from spacr.flowview.classify_blueprint import (
        CLASSIFY_NODE_IDS,
        classify_graph,
    )
    from spacr.flowview.collector import Collector
    from spacr.flowview.model import RunGraph

    previous_collector = trace.get_collector()
    previous_enabled = trace.is_enabled()
    empty = Collector(RunGraph("empty", 1.0, {}, [], "test", "digest"))
    trace.enable(empty)
    trace.disable()

    screen = _screen(qtbot)
    assert map_barcodes.install_folds_on(screen) is not None
    section = screen._flowview_section
    assert section is not None
    assert classify.install_flowview(screen) is section
    layout = screen._settings_content.layout()
    assert layout.indexOf(section) == layout.count() - 2
    assert layout.itemAt(layout.count() - 1).spacerItem() is not None
    assert section.is_expanded() is False
    assert section.panel() is None
    assert section._header.toolTip() == "Fold FlowView away, or open it again"

    screen.resize(1200, 800)
    screen.show()
    qapp.processEvents()
    section.set_expanded(True)
    qapp.processEvents()
    panel = section.panel()

    try:
        assert panel is not None
        assert trace.is_enabled() is True
        assert panel._collector is trace.get_collector()
        assert panel._follow_global_collector is True
        assert tuple(panel.graph.nodes) == CLASSIFY_NODE_IDS
        assert panel.timer.isActive() is True
        assert panel._embedded is True
        assert panel.title_label.isHidden() is True

        section.set_expanded(False)
        assert section.panel() is panel
        assert panel.timer.isActive() is False
        assert trace.is_enabled() is True

        section.set_expanded(True)
        assert section.panel() is panel
        assert panel.timer.isActive() is True
        screen.hide()
        qapp.processEvents()
        assert panel.timer.isActive() is False
        screen.show()
        qapp.processEvents()
        assert panel.timer.isActive() is True

        live_graph = classify_graph(
            screen._settings_model.collect(),
            run_id="live-classify-run",
        )
        live_collector = Collector(live_graph)
        trace.enable(live_collector)
        assert panel.refresh() is True
        assert panel._collector is live_collector
        assert panel.graph.run_id == "live-classify-run"

        shutdown_state = []
        real_shutdown = classify.LazyFlowViewSection.shutdown

        def tracked_shutdown(instance):
            real_shutdown(instance)
            if instance is section:
                shutdown_state.append(
                    (panel.timer.isActive(), bool(panel.scene.items()))
                )

        monkeypatch.setattr(
            classify.LazyFlowViewSection,
            "shutdown",
            tracked_shutdown,
        )
        assert screen.close() is True
        assert shutdown_state == [(False, False)]
    finally:
        trace.enable(previous_collector)
        if not previous_enabled:
            trace.disable()


def test_open_failure_is_recoverable_and_never_costs_the_fold_strip(
    qtbot,
    monkeypatch,
):
    """Both optional integration seams isolate a fault from Classify."""

    screen = _screen(qtbot)
    section = classify.install_flowview(screen)
    assert section is not None
    real_collector = classify.LazyFlowViewSection._collector_for_open_panel

    def broken_collector(_section):
        raise RuntimeError("collector unavailable")

    monkeypatch.setattr(
        classify.LazyFlowViewSection,
        "_collector_for_open_panel",
        broken_collector,
    )
    section.set_expanded(True)
    assert section.panel() is None
    assert section._error_label is not None
    assert section._error_label.text() == classify.FLOWVIEW_OPEN_ERROR

    section.set_expanded(False)
    monkeypatch.setattr(
        classify.LazyFlowViewSection,
        "_collector_for_open_panel",
        real_collector,
    )
    section.set_expanded(True)
    assert section.panel() is not None
    assert section._error_label is None
    section.shutdown()
    screen.close()

    no_panel = _screen(qtbot)

    def broken_install(_screen):
        raise RuntimeError("panel unavailable")

    monkeypatch.setattr(classify, "install_flowview", broken_install)
    assert classify.install_folds(no_panel) is not None
    assert getattr(no_panel, "_flowview_section", None) is None
    no_panel.close()


def test_flowview_box_is_in_the_exhaustive_theme_registry():
    """The settings footer cannot regress to an unstyled black container."""

    from spacr.qt import theme

    assert "spacr.qt.screens.classify" in theme.WIDGET_QSS_MODULES
    sheet = theme.stylesheet()
    assert "registered widget QSS: ClassifyFlowViewSection" in sheet
    assert "QWidget#ClassifyFlowViewSection" in sheet
    assert "QWidget#ClassifyFlowViewBody" in sheet


def test_non_classify_screen_gets_no_flowview_footer(qtbot):
    """The placement seam is Classify-only, as the approved first scope says."""

    screen = _screen(qtbot, "mask")

    assert classify.install_flowview(screen) is None
    assert getattr(screen, "_flowview_section", None) is None
    screen.close()
