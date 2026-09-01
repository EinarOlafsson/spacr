"""The FlowView panel stays usable when global trace state is unavailable."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from spacr.flowview import panel as panel_module
from spacr.flowview.collector import Collector
from spacr.flowview.model import RunGraph


def _collector() -> Collector:
    graph = RunGraph("coverage", 1.0, {}, [], "test", "digest")
    return Collector(graph)


def test_panel_follows_a_matching_global_and_survives_a_broken_lookup(
    qtbot, monkeypatch,
):
    """Both lookup outcomes retain the explicitly supplied collector."""
    collector = _collector()
    monkeypatch.setattr(panel_module, "get_collector", lambda: collector)
    following = panel_module.FlowViewPanel(collector, auto_start=False)
    qtbot.addWidget(following)
    assert following._collector is collector
    assert following._follow_global_collector is True

    def broken_lookup():
        raise RuntimeError("trace state is shutting down")

    monkeypatch.setattr(panel_module, "get_collector", broken_lookup)
    detached = panel_module.FlowViewPanel(collector, auto_start=False)
    qtbot.addWidget(detached)
    assert detached._collector is collector
    assert detached._follow_global_collector is False
