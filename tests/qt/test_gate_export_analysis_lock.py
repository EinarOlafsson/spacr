"""Live gate exports carry the existing preregistration verdict, including edits."""
import json
import sqlite3

import pandas as pd
import pytest

from spacr import run_journal as rj
from spacr.qt.screens.gate_editor import GateEditorScreen, _gate_export_lock_verdicts
from spacr.qt.widgets.gate_spec import GateSet, ThresholdGate


@pytest.fixture
def journal(tmp_path, monkeypatch):
    runs = tmp_path / "home" / "runs"
    runs.mkdir(parents=True)
    monkeypatch.setattr(rj, "runs_root", lambda: runs)
    now = {"time": "2026-10-01T10:00:00.000000Z"}
    monkeypatch.setattr(rj, "_utc_now", lambda: now["time"])
    return tmp_path, now


def gates(low=150):
    return GateSet([ThresholdGate(name="large", column="area", low=low)])


def database(folder):
    path = str(folder / "measurements.db")
    with sqlite3.connect(path) as db:
        pd.DataFrame(dict(plateID=["p1"] * 3, rowID=["r1"] * 3,
                          columnID=["c1"] * 3, fieldID=["f1"] * 3,
                          object_label=[1, 2, 3], area=[100., 200., 300.])).to_sql("cell", db, index=False)
    return path


def test_live_unsaved_edits_are_checked_and_successful_export_has_receipt(qtbot, journal):
    folder, now = journal
    path = database(folder)
    strategy = folder / "gates.json"
    screen = GateEditorScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.load_path(path, "cell")
    screen.gates.set_gates(gates())
    screen.save_gates(str(strategy))
    rj.lock_analysis({"src": str(folder)}, app_key="measure", gates=str(strategy))
    now["time"] = "2026-10-01T10:01:00.000000Z"
    screen.gates.set_gates(gates(250))
    screen.export_gates()
    assert "DEVIATION" in screen._source.text().upper()
    assert json.loads(strategy.read_text())["gates"][0]["low"] == 150
    with sqlite3.connect(path) as db:
        rows = db.execute("SELECT gate_column, receipt_json FROM filter_export_provenance").fetchall()
        filters = pd.read_sql_query("SELECT * FROM filters", db)
    assert len(rows) == 1
    column, payload = rows[0]
    receipt = json.loads(payload)
    assert receipt["gates"]["gates"][0]["low"] == 250
    assert receipt["analysis_locks"][0]["status"] == "deviation"
    assert receipt["analysis_locks"][0]["deviations"][0]["first_seen_utc"] == now["time"]
    assert filters.loc[filters[column] == 1, "object_label"].tolist() == [3]


def test_export_before_and_after_unblinding_preserves_first_seen_policy(journal):
    folder, now = journal
    path = database(folder)
    strategy = folder / "gates.json"
    gates().save(str(strategy))
    key = rj.start_blinding(["field1"], scope="annotate", src=str(folder))
    rj.lock_analysis({"src": str(folder)}, app_key="measure", gates=str(strategy))
    now["time"] = "2026-10-01T10:01:00.000000Z"
    before = GateEditorScreen._write_gates_checked(path, "cell", gates(200), str(strategy))
    assert before[2][0]["status"] == "deviation"
    now["time"] = "2026-10-01T10:02:00.000000Z"
    rj.unblind(key["key_id"], reason="completed scoring")
    now["time"] = "2026-10-01T10:03:00.000000Z"
    still_before = _gate_export_lock_verdicts(str(strategy), gates(200))
    assert still_before[0]["status"] == "deviation"
    changed = GateEditorScreen._write_gates_checked(path, "cell", gates(250), str(strategy))
    assert changed[2][0]["status"] == "post_hoc"
    with sqlite3.connect(path) as db:
        receipts = [json.loads(row[0]) for row in db.execute("SELECT receipt_json FROM filter_export_provenance")]
    assert [r["analysis_locks"][0]["status"] for r in receipts] == ["deviation", "post_hoc"]


def test_no_lock_never_claims_verified_and_failed_export_has_no_receipt(journal):
    folder, _now = journal
    path = database(folder)
    failed_gates = GateSet([ThresholdGate(name="missing", column="absent", low=0)])
    result = GateEditorScreen._write_gates_checked(path, "cell", failed_gates, "")
    assert not result[0] and result[1] and not result[2]
    with sqlite3.connect(path) as db:
        assert not db.execute("SELECT name FROM sqlite_master WHERE name='filter_export_provenance'").fetchall()
    result = GateEditorScreen._write_gates_checked(path, "cell", gates(), "")
    assert result[0] and not result[1] and not result[2]
    with sqlite3.connect(path) as db:
        receipt = json.loads(db.execute("SELECT receipt_json FROM filter_export_provenance").fetchone()[0])
    assert receipt["analysis_locks"] == []
    assert receipt["lock_coverage"] == "no_applicable_strategy_lock"


def test_tampered_lock_is_recorded_without_claiming_verification(journal):
    folder, _now = journal
    path = database(folder)
    strategy = folder / "gates.json"
    gates().save(str(strategy))
    record = rj.lock_analysis({"src": str(folder)}, app_key="measure", gates=str(strategy))
    lock_path = rj._locks_root() / (record["lock_id"] + ".json")
    altered = json.loads(lock_path.read_text())
    altered["locked_by"] = "edited"
    lock_path.write_text(json.dumps(altered))
    result = GateEditorScreen._write_gates_checked(path, "cell", gates(), str(strategy))
    assert result[2][0]["status"] == "tampered"


def test_loaded_strategy_is_used_for_live_export_check(qtbot, journal):
    folder, _now = journal
    path = database(folder)
    strategy = folder / "gates.json"
    gates().save(str(strategy))
    rj.lock_analysis({"src": str(folder)}, app_key="measure", gates=str(strategy))
    screen = GateEditorScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.load_path(path, "cell")
    assert screen.load_gates(str(strategy))
    screen.export_gates()
    assert "verified" in screen._source.text().lower()
