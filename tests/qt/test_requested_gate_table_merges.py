"""Both real screens consume, persist and validate the same derived tables."""
import json
import sqlite3

import pandas as pd
import pytest

from spacr.derived_tables import default_definition, sidecar_path
from spacr.qt.widgets.gate_spec import GateSet, ThresholdGate
from spacr.qt.widgets.graph_spec import GraphSpec


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "measurements.db")
    ident = dict(plateID=["p1"] * 3, rowID=["r1"] * 3, columnID=["c1"] * 3, fieldID=["f1"] * 3)
    with sqlite3.connect(path) as connection:
        pd.DataFrame({**ident, "object_label": [1, 2, 3], "area": [100., 200., 300.]}).to_sql("cell", connection, index=False)
        pd.DataFrame({**ident, "object_label": [7, 8, 9], "cell_id": [1, 1, 2],
                      "area": [2., 3., 7.], "mean_intensity": [5., 7., 11.]}).to_sql("pathogen", connection, index=False)
    return path


def make_screen(qtbot, kind):
    if kind == "graph":
        from spacr.qt.screens.graph_builder import GraphBuilderScreen
        widget = GraphBuilderScreen(threaded=False)
    else:
        from spacr.qt.screens.gate_editor import GateEditorScreen
        widget = GateEditorScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget




def test_named_merge_is_real_source_and_survives_refresh(qtbot, db):
    kind = "gate"
    screen = make_screen(qtbot, kind)
    screen.load_path(db, "cell")
    assert screen._merge_button.isEnabled()
    definition = default_definition(db, ["cell", "pathogen"], name="Cell with pathogens")
    screen.use_derived_table(definition)
    assert screen._table_picker.currentText() == "Cell with pathogens"
    assert screen._frame.pathogen_area.iloc[:2].tolist() == [5., 7.]
    screen.load_path(db, "cell")
    assert screen._table_picker.findText("Cell with pathogens") >= 0
    screen.load_path(db, "Cell with pathogens")
    assert screen._frame.pathogen_area.iloc[0] == 5
    if kind == "graph":
        screen.builder.set_spec(GraphSpec(kind="scatter", x="cell_area", y="pathogen_area"))
        assert screen.builder.canvas.figure().axes
    else:
        gate = ThresholdGate(name="high", column="pathogen_area", low=6)
        screen.gates.set_gates(GateSet([gate]))
        assert screen.gates.gates.mask(screen._frame, "high").tolist() == [False, True, False]

def test_external_data_without_image_provenance_still_plots_and_gates(qtbot, tmp_path):
    path = str(tmp_path / "external.db")
    with sqlite3.connect(path) as connection:
        pd.DataFrame({"sample": ["a", "b"], "reading": [10., 20.]}).to_sql("samples", connection, index=False)
        pd.DataFrame({"sample_ref": ["a", "a", "b"], "value": [1., 3., 7.]}).to_sql("events", connection, index=False)
    definition = default_definition(path, ["samples", "events"], base="samples", name="Custom data")
    definition.update(mode="custom", acknowledged=True, base_keys=["sample"])
    definition["joins"][0].update(left_keys=["sample"], right_keys=["sample_ref"], how="left")
    kind = "gate"
    screen = make_screen(qtbot, kind)
    screen.load_path(path, "samples")
    screen.use_derived_table(definition)
    assert "object_label" not in screen._frame
    if kind == "graph":
        assert not screen._to_annotate.isEnabled()
        screen.builder.set_spec(GraphSpec(kind="scatter", x="samples_reading", y="events_value"))
        assert screen.builder.canvas.figure().axes
    else:
        assert not screen._annotate.isEnabled()
        assert not screen._export.isEnabled()
        gate = ThresholdGate(name="larger", column="events_value", low=3)
        assert gate.mask(screen._frame).tolist() == [False, True]

def test_gate_save_reconstructs_definition_and_export_gates_on_full_merge(qtbot, db, tmp_path):
    screen = make_screen(qtbot, "gate")
    screen.load_path(db, "cell")
    definition = default_definition(db, ["cell", "pathogen"], name="Derived gate data")
    screen.use_derived_table(definition)
    screen.gates.set_gates(GateSet([ThresholdGate(name="high", column="pathogen_area", low=6)]))
    saved = str(tmp_path / "gates.json")
    screen.save_gates(saved)
    assert json.loads(open(saved).read())["merge_definition"]["name"] == "Derived gate data"
    sidecar_path(db).unlink()
    screen.load_path(db, "cell")
    assert screen.load_gates(saved)
    assert screen._table == "Derived gate data"
    written, failed = screen._write_gates(db, screen._table, screen.gates.gates)
    assert not failed
    assert written and written[0][1] == 1
    with sqlite3.connect(db) as connection:
        filtered = pd.read_sql_query('SELECT * FROM filters', connection)
    hits = filtered[filtered[written[0][0]] == 1]
    assert hits.object_label.tolist() == [2]
    assert hits.object_type.tolist() == ["cell"]


def test_derived_gate_export_retains_time_id_alias(qtbot, db):
    screen = make_screen(qtbot, "gate")
    with sqlite3.connect(db) as connection:
        for table in ("cell", "pathogen"):
            frame = pd.read_sql_query('SELECT * FROM ' + table, connection)
            frames = [frame.assign(time_id=1), frame.assign(time_id=2)]
            if table == "pathogen":
                frames[1]["area"] *= 10
            pd.concat(frames, ignore_index=True).to_sql(table, connection, index=False, if_exists="replace")
    screen.load_path(db, "cell")
    screen.use_derived_table(default_definition(db, ["cell", "pathogen"], name="Time merge"))
    gates = GateSet([ThresholdGate(name="later", column="pathogen_area", low=60)])
    written, failed = screen._write_gates(db, "Time merge", gates)
    assert not failed
    assert written[0][1] == 1
    with sqlite3.connect(db) as connection:
        filtered = pd.read_sql_query("SELECT * FROM filters", connection)
    marked = filtered[filtered[written[0][0]] == 1]
    assert marked.object_label.tolist() == [2]
    assert marked.timeID.tolist() == [2]
