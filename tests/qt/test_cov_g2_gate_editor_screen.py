"""Gate Editor screen: merged tables, refusals and lock bookkeeping."""
from __future__ import annotations

import json

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.derived_tables import default_definition  # noqa: E402
from spacr.qt.screens import gate_editor as ge  # noqa: E402
from tests.qt.test_requested_gate_table_merges import db, make_screen  # noqa: E402,F401


def _saved_merge(path, name="Derived gate data"):
    from spacr.derived_tables import execute, save_definition

    definition = default_definition(path, ["cell", "pathogen"], name=name)
    frame, _report = execute(path, definition)
    save_definition(path, frame.attrs.get("merge_definition", definition))
    return name


def test_a_saved_merge_is_capped_and_sampled(db):  # noqa: F811
    name = _saved_merge(db)
    assert len(ge.GateEditorScreen._read(db, name, 1.0, 2)) == 2
    assert len(ge.GateEditorScreen._read(db, name, 0.5, None)) == 2


def test_a_merge_without_provenance_cannot_export_or_annotate(qtbot, db):  # noqa: F811
    screen = make_screen(qtbot, "gate")
    screen._merge_definition = {"name": "merged"}
    screen._frame = pd.DataFrame({"a": [1]})
    screen.export_gates()
    assert "no verified image/object provenance" in screen._source.text()
    screen._source.setText("")
    screen.annotate_from_gates()
    assert "no verified image/object provenance" in screen._source.text()


def test_choosing_a_saved_merge_from_the_picker_opens_it(qtbot, db, monkeypatch):  # noqa: F811
    screen = make_screen(qtbot, "gate")
    screen.load_path(db, "cell")
    name = _saved_merge(db)
    opened = []
    monkeypatch.setattr(screen, "load_path",
                        lambda path, table=None: opened.append(table))
    screen._table_picker.addItem(name)
    screen._table_picker.setCurrentText(name)
    screen._on_table_added(0)
    assert opened and opened[-1] == name


def test_loading_a_merged_strategy_needs_its_database(qtbot, tmp_path):
    screen = make_screen(qtbot, "gate")
    strategy = tmp_path / "gates.json"
    strategy.write_text(json.dumps({"gates": [], "merge_definition": {
        "name": "m", "version": 1}}), encoding="utf-8")
    screen._path = ""
    assert screen.load_gates(str(strategy)) is False
    assert "source database" in screen._source.text()


def test_a_table_picked_while_a_merge_is_open_replaces_it(qtbot, db, monkeypatch):  # noqa: F811
    screen = make_screen(qtbot, "gate")
    screen.load_path(db, "cell")
    opened = []
    monkeypatch.setattr(screen, "load_path",
                        lambda path, table=None: opened.append(table))
    screen._merge_definition = {"name": "merged"}
    screen._table_picker.addItem("pathogen")
    screen._table_picker.setCurrentText("pathogen")
    screen._on_table_added(0)
    assert opened == ["pathogen"] and screen._tables == ["pathogen"]


def test_gates_the_canvas_refuses_keep_the_previous_strategy(qtbot, db,  # noqa: F811
                                                             monkeypatch, tmp_path):
    screen = make_screen(qtbot, "gate")
    screen.load_path(db, "cell")
    strategy = tmp_path / "gates.json"
    strategy.write_text(json.dumps({"gates": []}), encoding="utf-8")
    before = screen._gate_strategy_path

    def refuse(gates):
        raise ValueError("canvas refused")

    monkeypatch.setattr(screen.gates, "set_gates", refuse)
    assert screen.load_gates(str(strategy)) is False
    assert screen._gate_strategy_path == before


def test_a_merged_strategy_already_in_the_picker_is_not_added_twice(qtbot, db,  # noqa: F811
                                                                    tmp_path):
    screen = make_screen(qtbot, "gate")
    screen.load_path(db, "cell")
    definition = default_definition(db, ["cell", "pathogen"], name="merged gates")
    screen._table_picker.addItem("merged gates")
    strategy = tmp_path / "gates.json"
    strategy.write_text(json.dumps({"gates": [], "merge_definition": definition}), encoding="utf-8")
    screen.load_gates(str(strategy))
    names = [screen._table_picker.itemText(i)
             for i in range(screen._table_picker.count())]
    assert names.count("merged gates") == 1
