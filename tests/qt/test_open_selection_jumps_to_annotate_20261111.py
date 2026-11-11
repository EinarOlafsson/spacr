"""Open selection in Annotate switches to Annotate, building it if needed.

Reported: choosing Open selection in Annotate from the right-click menu in
Graph Builder or Gate Editor did nothing until the user had clicked the
Annotate module once, and even then left the user on the plot. The main window
now registers a navigator with the shared link, so the request builds
Annotate, switches the window to it through the sidebar path, and delivers the
objects in one step.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from spacr.qt.linked_selection import (DEFAULT_OPEN_KIND, LinkedSelection,
                                       NoObjectOpener, linked_selection,
                                       object_database_context)
from spacr.selection import ObjectRequest


def _measurements() -> pd.DataFrame:
    return pd.DataFrame({
        "plateID": ["p1"] * 4,
        "rowID": ["r1", "r1", "r2", "r2"],
        "columnID": ["c1", "c2", "c1", "c2"],
        "fieldID": ["f1"] * 4,
        "object_label": [1, 2, 3, 4],
        "area": [10.0, 20.0, 30.0, 40.0],
    })


@pytest.fixture
def window(qapp, monkeypatch, tmp_path):
    monkeypatch.delenv("SPACR_PREWARM", raising=False)
    from spacr.qt import app as app_module

    win = app_module.MainWindow(initial_app="__home__")
    win.resize(1200, 800)
    win.show()
    qapp.processEvents()
    assert "annotate" not in win._screens
    yield win
    prewarm = getattr(win, "_screen_prewarm", None)
    if prewarm is not None:
        prewarm.stop()
    win.close()
    qapp.processEvents()
    linked_selection().unregister_object_navigator(
        DEFAULT_OPEN_KIND, win._bring_object_destination_forward)


@pytest.fixture
def table_db(tmp_path):
    path = tmp_path / "table.db"
    with sqlite3.connect(path) as db:
        _measurements().to_sql("cell", db, index=False)
    return str(path)


def _assert_on_annotate_with(window, keys):
    annotate = window._screens.get("annotate")
    assert annotate is not None, "Annotate was not built by the jump"
    assert window._stack.currentWidget() is annotate
    request = annotate._object_request
    assert request is not None, "the selection never reached Annotate"
    assert sorted(request.keys) == sorted(keys)


def test_graph_builder_jumps_to_annotate_never_opened(window, qtbot,
                                                      table_db):
    window.open_module("graph_builder")
    screen = window._screens["graph_builder"]
    screen._path = table_db
    frame = _measurements()
    screen.set_frame(frame)
    picked = frame.iloc[:2]
    screen.builder.canvas.link.select_frame(picked, source="graph_builder")
    selection = screen.builder.canvas.link.selection
    screen._to_annotate.setEnabled(True)

    screen._to_annotate.trigger()

    _assert_on_annotate_with(window, list(selection.keys))
    assert "open the Annotate screen once" not in screen._source.text()


def test_gate_editor_jumps_to_annotate_never_opened(window, qtbot, table_db):
    from spacr.qt.widgets.gate_spec import GateSet, ThresholdGate

    window.open_module("gate_editor")
    screen = window._screens["gate_editor"]
    screen._path = table_db
    screen.set_frame(_measurements())
    qtbot.waitUntil(lambda: screen.gates._frame is not None, timeout=10000)
    screen.gates.set_gates(
        GateSet().add(ThresholdGate(name="big", column="area", low=25)))
    screen.gates.tree.select("big")
    items = {label: (enabled, callback) for label, enabled, callback, _why
             in screen._annotation_menu_items()}
    enabled, callback = items["Open selection in Annotate"]
    assert enabled

    callback()

    gated = screen.gates._frame
    inside = gated.loc[gated["area"] > 25]
    from spacr.selection import as_key_index

    _assert_on_annotate_with(window, list(as_key_index(inside)))
    assert "Open selection in Annotate" in [
        row[0] for row in screen.graph_menu_items()]


def test_a_navigator_runs_before_the_opener_and_reports_a_failed_open():
    link = LinkedSelection()
    seen = []

    def navigator(kind):
        seen.append(kind)
        link.register_object_opener(kind, lambda request: request)

    assert not link.has_object_opener("annotate")
    link.register_object_navigator("annotate", navigator)
    assert link.has_object_opener("annotate")
    request = link.open_objects(["p1_r1_c1_f1_o1"], reason="test")
    assert seen == ["annotate"] and request.reason == "test"

    link.deliver_request(ObjectRequest(keys=["p1_r1_c1_f1_o1"], reason="quiet"))
    assert seen == ["annotate"]

    stuck = LinkedSelection()
    stuck.register_object_navigator("annotate", lambda kind: None)
    with pytest.raises(NoObjectOpener, match="did not open"):
        stuck.open_objects(["p1_r1_c1_f1_o1"], reason="test")
    assert stuck.unregister_object_navigator("annotate")
    assert not stuck.has_object_opener("annotate")


def test_annotate_opens_the_requests_database_when_nothing_is_open(
        qtbot, tmp_path, monkeypatch):
    from spacr.qt.screens import annotate as annotate_module

    plate = tmp_path / "plate"
    database = plate / "measurements" / "measurements.db"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"")
    request = ObjectRequest(keys=["p1_r1_c1_f1_o1"], reason="jump",
                            context=object_database_context(str(database)))
    assert annotate_module._request_source_folder(request) == str(plate)
    assert object_database_context(str(tmp_path / "x.csv")) == {}

    screen = annotate_module.AnnotateScreen()
    qtbot.addWidget(screen)
    opened = []
    monkeypatch.setattr(screen, "_open_source",
                        lambda src, then=None: opened.append(src))
    screen._settings.db_path = ""
    assert screen.open_object_request(request) is screen
    assert opened == [str(plate)]
