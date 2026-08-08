"""A filter set is as much an analysis decision as a gate, and gates save.

Gates have had Save/Load since the beginning; filters had not, so a set of
filters -- which rows this analysis is about -- had to be rebuilt by hand
every session.
"""

from __future__ import annotations

import json
import os
import tempfile

import pandas as pd
import pytest


@pytest.fixture()
def frame():
    return pd.DataFrame({
        "area": [1.0, 2.0, 3.0, 4.0, 5.0],
        "well": list("AABBC"),
    })


@pytest.fixture()
def panel(qtbot, frame):
    from spacr.qt.widgets.data_filter_panel import DataFilterPanel

    widget = DataFilterPanel()
    qtbot.addWidget(widget)
    widget.set_frame(frame)
    return widget


def test_a_filter_set_round_trips(panel, frame, qtbot):
    from spacr.qt.widgets.data_filter_panel import DataFilterPanel

    panel.add_column("area")
    panel.add_column("well")
    before = panel.state()

    path = os.path.join(tempfile.mkdtemp(), "filters.json")
    panel.save(path)

    other = DataFilterPanel()
    qtbot.addWidget(other)
    other.set_frame(frame)
    assert other.load(path) == []
    assert other.state() == before


def test_the_saved_file_is_json_a_human_can_read(panel, qtbot):
    """It is a record of an analysis decision, so it has to be legible."""
    panel.add_column("area")
    path = os.path.join(tempfile.mkdtemp(), "filters.json")
    panel.save(path)

    with open(path, encoding="utf-8") as handle:
        loaded = json.load(handle)
    assert loaded["version"] == 1
    assert loaded["filters"][0]["column"] == "area"


def test_a_column_this_table_does_not_have_is_reported(panel, qtbot):
    """Loading a set saved against another plate is an ordinary thing to do.

    What must not happen is it half-applying: a set that silently drops a
    filter selects the wrong rows while looking like it worked.
    """
    from spacr.qt.widgets.data_filter_panel import DataFilterPanel

    panel.add_column("area")
    panel.add_column("well")
    path = os.path.join(tempfile.mkdtemp(), "filters.json")
    panel.save(path)

    narrower = DataFilterPanel()
    qtbot.addWidget(narrower)
    narrower.set_frame(pd.DataFrame({"area": [1.0, 2.0]}))
    assert narrower.load(path) == ["well"]


def test_an_unknown_version_is_refused_not_guessed(panel, qtbot):
    with pytest.raises(ValueError, match="version"):
        panel.restore({"version": 99, "filters": []})


def test_restoring_replaces_rather_than_appends(panel, frame, qtbot):
    """Otherwise loading twice doubles every filter."""
    panel.add_column("area")
    path = os.path.join(tempfile.mkdtemp(), "filters.json")
    panel.save(path)

    panel.load(path)
    panel.load(path)
    assert len(panel.state()["filters"]) == 1


def test_the_screen_exposes_both_buttons(qtbot):
    from PySide6.QtWidgets import QPushButton

    from spacr.qt.screens.gate_editor import GateEditorScreen

    screen = GateEditorScreen()
    qtbot.addWidget(screen)
    texts = {b.text() for b in screen.findChildren(QPushButton)}
    assert "Save filters…" in texts
    assert "Load filters…" in texts
