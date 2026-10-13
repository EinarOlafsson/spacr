"""Both real screens consume, persist and validate the same derived tables."""
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
    kind = "graph"
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
    kind = "graph"
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

def test_chart_roundtrip_reconstructs_data_before_plotting(qtbot, db, tmp_path):
    screen = make_screen(qtbot, "graph")
    screen.load_path(db, "cell")
    screen.use_derived_table(default_definition(db, ["cell", "pathogen"], name="Plot data"))
    screen.builder.set_spec(GraphSpec(kind="scatter", x="cell_area", y="pathogen_area"))
    saved = str(tmp_path / "chart.json")
    screen.save_chart(saved)
    sidecar_path(db).unlink()
    screen.load_path(db, "cell")
    screen.load_chart(saved)
    assert screen._table_picker.currentText() == "Plot data"
    assert screen.builder.spec.y == "pathogen_area"
    assert screen._frame.pathogen_area.iloc[0] == 5
