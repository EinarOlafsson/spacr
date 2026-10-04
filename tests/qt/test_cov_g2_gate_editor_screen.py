"""Gate Editor screen: merged tables, refusals and lock bookkeeping."""
from __future__ import annotations

import json

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.derived_tables import default_definition  # noqa: E402
from spacr.qt.screens import gate_editor as ge  # noqa: E402
from spacr.qt.widgets.gate_spec import GateSet, ThresholdGate  # noqa: E402
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


def test_a_derived_table_without_provenance_writes_no_gate(db, monkeypatch):  # noqa: F811
    import spacr.derived_tables as dt

    name = _saved_merge(db)
    real = dt.execute

    def no_provenance(path, definition):
        frame, report = real(path, definition)
        frame.attrs["image_provenance"] = False
        return frame, report

    monkeypatch.setattr(dt, "execute", no_provenance)
    gates = GateSet([ThresholdGate(name="g", column="pathogen_area", low=1)])
    written, failed = ge.GateEditorScreen._write_gates(db, name, gates)
    assert written == [] and failed[0][1] == "No verified image/object provenance"


def test_a_provenance_failure_after_writing_is_reported(db, monkeypatch):  # noqa: F811
    def broken(*a, **k):
        raise OSError("sidecar locked")

    monkeypatch.setattr(ge, "_record_gate_export_provenance", broken)
    monkeypatch.setattr(ge.GateEditorScreen, "_write_gates",
                        staticmethod(lambda path, table, gates: ([("g", 1)], [])))
    written, failed, _verdicts = ge.GateEditorScreen._write_gates_checked(
        db, "cell", GateSet([]), "")
    assert written and failed == [("export provenance", "sidecar locked")]


def test_unreadable_and_unrelated_locks_are_skipped(tmp_path):
    from spacr.run_journal import _locks_root

    root = _locks_root()
    (root / "broken.json").write_text("{not json")
    (root / "other.json").write_text(json.dumps({"gates": {"/x.json": {}}}))
    strategy = tmp_path / "gates.json"
    strategy.write_text(json.dumps({"gates": []}))
    assert ge._gate_export_lock_verdicts(str(strategy), GateSet([])) == []
    assert ge._gate_export_lock_verdicts("", GateSet([])) == []


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
        "name": "m", "version": 1}}))
    screen._path = ""
    assert screen.load_gates(str(strategy)) is False
    assert "source database" in screen._source.text()
