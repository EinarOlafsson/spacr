"""Protocols route a drug screen to Dose-Response and to Regression in drug-screen mode."""
import sqlite3

import pandas as pd


def test_protocols_offer_regression_for_drug_screens_in_drug_screen_mode(tmp_path):
    from spacr.qt.widgets import sample_project as sample

    db = tmp_path / "measurements.db"
    with sqlite3.connect(db) as connection:
        pd.DataFrame(dict(plateID=["plate1"], rowID=["r1"], columnID=["c1"],
                          fieldID=["f1"], intensity=[1.0])).to_sql("cell", connection, index=False)
    raw = db.read_bytes()
    for format_ in ("database", "database_crops"):
        result = sample._protocol_result("features", format_, dict(experiment="drug"), str(db))
        starts = [row[0] for row in result["actions"]]
        assert starts[:2] == ["dose_response", "regression"]
        handoff = sample._protocol_handoff(result, "regression")
        assert handoff["settings"]["screen_type"] == "drug_screen"
        assert handoff["settings"]["drug_data"] == str(db)
        assert handoff["settings"].get("src", "") == ""
        assert any("vehicle_control" in text for text in result["requirements"])
    screen = sample._protocol_result("features", "database",
                                     dict(experiment="screen", barcodes="mapped"), str(db))
    handoff = sample._protocol_handoff(screen, "regression")
    assert "screen_type" not in handoff["settings"]
    assert db.read_bytes() == raw
