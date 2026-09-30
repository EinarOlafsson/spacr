"""Plate maps filled from sample records by plate barcode, with mismatches.

The records come from a local CSV, from a fake LIMS fetcher, and from a
LIMS stand-in served on 127.0.0.1; no real LIMS service is contacted.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pandas as pd
import pytest

from spacr.measure import _run_plate_barcode_step
from spacr.plate_qc import (_link_plate_barcodes, _lims_url,
                            _parse_barcode_assignments, _plate_barcodes)
from spacr.tabular import read_table, write_table


def _imaged(tmp_path, plate="plate1", wells=((1, 1), (1, 2), (2, 1))):
    """A plate folder whose merged folder holds one field per well."""
    src = tmp_path / plate / "merged"
    src.mkdir(parents=True, exist_ok=True)
    for r, c in wells:
        np.save(src / f"{plate}_{chr(64 + r)}{c:02d}_1.npy", np.zeros((2, 2, 1)))
    return src


def _records(tmp_path, rows):
    path = tmp_path / "lims.csv"
    write_table(pd.DataFrame(rows), str(path))
    return str(path)


ROWS = [
    {"barcode": "BC001", "well": "A01", "strain": "RH", "compound": "DMSO",
     "concentration": 0, "passage": 12, "operator": "eo"},
    {"barcode": "BC001", "well": "A02", "strain": "RH", "compound": "pyr",
     "concentration": 1.5, "passage": 12, "operator": "eo"},
    {"barcode": "BC001", "well": "B01", "strain": "ME49", "compound": "pyr",
     "concentration": 3.0, "passage": 12, "operator": "eo"},
    {"barcode": "BC777", "well": "A01", "strain": "GT1", "compound": "x",
     "concentration": 1, "passage": 3, "operator": "zz"},
]


def test_a_plate_imported_with_a_barcode_gets_its_plate_map(tmp_path):
    src = _imaged(tmp_path)
    (tmp_path / "plate1" / "barcode.txt").write_text("BC001\n")
    settings = {"src": str(src), "plate_barcode_source":
                _records(tmp_path, ROWS), "plate_barcode_column": "barcode",
                "profiling_metadata": "", "viability_plate_map": "",
                "timelapse": False}
    plate_map, mismatches = _run_plate_barcode_step(settings)
    assert mismatches.empty
    assert plate_map["plateID"].tolist() == ["plate1"] * 3
    assert plate_map["rowID"].tolist() == ["r1", "r1", "r2"]
    assert plate_map["columnID"].tolist() == ["c1", "c2", "c1"]
    assert plate_map["strain"].tolist() == ["RH", "RH", "ME49"]
    assert plate_map["concentration"].tolist() == [0, 1.5, 3.0]
    assert set(plate_map["plate_barcode"]) == {"BC001"}
    written = str(tmp_path / "plate1" / "measurements" / "plate_map_lims.csv")
    assert settings["profiling_metadata"] == written
    assert settings["viability_plate_map"] == written
    again = read_table(written, report=None)
    assert again["operator"].tolist() == ["eo"] * 3


def test_every_kind_of_mismatch_is_reported(tmp_path, capsys):
    src = _imaged(tmp_path, wells=((1, 1), (1, 2), (3, 3)))
    _imaged(tmp_path, plate="plate2", wells=((1, 1),))
    for plate in ("plate2",):
        np.save(src / f"{plate}_A01_1.npy", np.zeros((2, 2, 1)))
    np.save(src / "plate3_A01_1.npy", np.zeros((2, 2, 1)))
    rows = ROWS + [
        {"barcode": "BC001", "well": "A02", "strain": "GT1",
         "compound": "pyr", "concentration": 1.5, "passage": 12,
         "operator": "eo"},
        {"barcode": "BC001", "well": "??", "strain": "RH", "compound": "x",
         "concentration": 1, "passage": 1, "operator": "eo"}]
    user_map = tmp_path / "mine.csv"
    write_table(pd.DataFrame({"well": ["A01"], "compound": ["DMSO"],
                              "concentration": [0.5]}), str(user_map))
    settings = {"src": str(src), "plate_barcode_source":
                _records(tmp_path, rows),
                "plate_barcodes": "plate1=BC001; plate2=BC001; plate3=BCX",
                "profiling_metadata": str(user_map),
                "viability_plate_map": "", "timelapse": False}
    plate_map, mismatches = _run_plate_barcode_step(settings)
    kinds = mismatches.groupby("kind")["well"].apply(sorted).to_dict()
    assert kinds["barcode_shared"] == ["", ""]
    assert kinds["barcode_not_found"] == [""]
    assert kinds["well_not_in_lims"] == ["C03"]
    assert kinds["well_not_imaged"] == ["A02", "B01", "B01"]
    assert kinds["conflicting_records"] == ["A02", "A02"]
    assert kinds["unreadable_well"] == ["??"]
    differs = mismatches[mismatches["kind"] == "plate_map_differs"]
    assert set(differs["well"]) == {"A01"}
    assert differs["detail"].str.startswith("concentration").all()
    assert settings["profiling_metadata"] == str(user_map)
    assert settings["viability_plate_map"].endswith("plate_map_lims.csv")
    out = capsys.readouterr().out
    assert "MISMATCH barcode_not_found: plate plate3 (barcode BCX)" in out
    report = read_table(str(tmp_path / "plate1" / "measurements"
                            / "plate_barcode_mismatches.csv"), report=None)
    assert len(report) == len(mismatches)


def test_barcodes_come_from_the_setting_then_the_file_then_the_name(tmp_path):
    src = _imaged(tmp_path)
    assert _plate_barcodes(str(src), ["plate1"]) == {"plate1": "plate1"}
    (tmp_path / "plate1" / "barcode.txt").write_text("plate1=BCF\n")
    assert _plate_barcodes(str(src), ["plate1", "p2"]) == {
        "plate1": "BCF", "p2": "p2"}
    assert _plate_barcodes(str(src), ["plate1"], {"plate1": "BCS"}) == {
        "plate1": "BCS"}
    assert _parse_barcode_assignments("a=1,\nb = 2") == {"a": "1", "b": "2"}
    with pytest.raises(ValueError, match="plateID=barcode"):
        _parse_barcode_assignments("justone")


def test_a_lims_service_is_asked_per_barcode_with_the_token(tmp_path,
                                                            monkeypatch):
    src = _imaged(tmp_path, wells=((1, 1),))
    monkeypatch.setenv("FAKE_LIMS_TOKEN", "s3cret")
    calls = []

    def fetch(url, headers):
        calls.append((url, headers))
        return json.dumps({"barcode": "BC 1", "strain": "RH",
                           "wells": [{"well": "A01", "compound": "pyr",
                                      "concentration": 2}]}).encode()

    plate_map, mismatches = _link_plate_barcodes(
        str(src), "https://lims.invalid/api/plates",
        barcodes={"plate1": "BC 1"}, token_env="FAKE_LIMS_TOKEN",
        fetch=fetch)
    assert calls[0][0] == "https://lims.invalid/api/plates?barcode=BC%201"
    assert calls[0][1]["Authorization"] == "Bearer s3cret"
    assert plate_map[["strain", "compound"]].values.tolist() == [["RH", "pyr"]]
    assert mismatches.empty
    assert _lims_url("http://x/p/{barcode}/wells?f=1", "A/B") == \
        "http://x/p/A%2FB/wells?f=1"
    with pytest.raises(ValueError, match="not JSON"):
        _link_plate_barcodes(str(src), "http://lims.invalid/",
                             fetch=lambda url, headers: b"<html>")


def test_a_local_lims_stand_in_over_http(tmp_path):
    src = _imaged(tmp_path, wells=((1, 1), (1, 2)))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps([{"plate_barcode": "plate1", "rowID": "A",
                                "columnID": 1, "operator": "eo"}]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            return None

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/plates/{{barcode}}"
        plate_map, mismatches = _link_plate_barcodes(
            str(src), url, barcode_column="plate_barcode")
    finally:
        server.shutdown()
        server.server_close()
    assert plate_map["operator"].tolist() == ["eo"]
    assert mismatches[["kind", "well"]].values.tolist() == [
        ["well_not_in_lims", "A02"]]


def test_an_unreachable_service_is_reported_and_the_run_goes_on(tmp_path,
                                                                capsys):
    src = _imaged(tmp_path, wells=((1, 1),))

    def down(url, headers):
        raise OSError("connection refused")

    settings = {"src": str(src), "plate_barcode_source": "http://lims.invalid",
                "profiling_metadata": "", "viability_plate_map": ""}
    assert _run_plate_barcode_step(settings, fetch=down) is None
    assert "could not be made: connection refused" in capsys.readouterr().out
    assert settings["profiling_metadata"] == ""
    records = _records(tmp_path, [{"code": "x", "well": "A01"}])
    settings["plate_barcode_source"] = records
    assert _run_plate_barcode_step(settings) is None
    assert "no barcode column" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Edges the coverage ratchet found untested (dispatch 36739819315)
# ---------------------------------------------------------------------------

def test_barcode_assignments_skip_blank_halves():
    assert _parse_barcode_assignments("plate1=BC1, plate2= ,=BC3") == {
        "plate1": "BC1"}


def test_only_readable_field_arrays_count_as_imaged_wells(tmp_path):
    from spacr.plate_qc import _imaged_wells

    src = _imaged(tmp_path, wells=((1, 1),))
    (src / "notes.txt").write_text("not an array")
    np.save(src / "not_a_field.npy", np.zeros(1))
    np.save(src / "plate1_A00_1.npy", np.zeros(1))
    wells = _imaged_wells(str(src))
    assert wells.values.tolist() == [["plate1", 1, 1]]


def test_a_barcode_file_for_several_plates_needs_plate_names(tmp_path):
    """One bare barcode is only meaningful for a folder of one plate."""
    src = _imaged(tmp_path)
    (tmp_path / "plate1" / "barcode.txt").write_text("BC001\n")
    assert _plate_barcodes(str(src), ["plate1", "plate2"]) == {
        "plate1": "plate1", "plate2": "plate2"}


@pytest.mark.parametrize("answer,match", [
    ({"plate": "BC1"}, "has no list of well records"),
    ("just text", "is not a list of well records"),
])
def test_a_lims_answer_without_well_records_is_refused(answer, match):
    from spacr.plate_qc import _lims_payload_records

    with pytest.raises(ValueError, match=match):
        _lims_payload_records(answer, "BC1", "barcode")


def test_non_record_entries_in_a_lims_answer_are_skipped():
    from spacr.plate_qc import _lims_payload_records

    records = _lims_payload_records([{"well": "A01"}, "noise", 7], "BC1",
                                    "barcode")
    assert records == [{"well": "A01", "barcode": "BC1"}]


def test_a_text_answer_is_read_as_json_and_blank_values_agree():
    from spacr.plate_qc import _lims_records, _same_value

    frame = _lims_records("https://lims.invalid/", ["BC1"],
                          fetch=lambda url, headers: '[{"well": "A01"}]')
    assert frame["well"].tolist() == ["A01"]
    assert _same_value(None, float("nan")) is True
    assert _same_value("", 3) is False


def test_a_folder_with_no_field_arrays_has_nothing_to_link(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No merged field arrays"):
        _link_plate_barcodes(str(empty), _records(tmp_path, ROWS))


def test_records_for_other_plates_only_leave_every_plate_unknown(tmp_path):
    src = _imaged(tmp_path, wells=((1, 1),))
    only_other = [row for row in ROWS if row["barcode"] == "BC777"]
    plate_map, mismatches = _link_plate_barcodes(
        str(src), _records(tmp_path, only_other), barcodes="plate1=BC001")
    assert plate_map.empty
    assert set(mismatches["kind"]) == {"barcode_not_found"}


def test_a_plate_map_that_shares_no_column_has_no_differences(tmp_path):
    from spacr.plate_qc import _plate_map_differences

    theirs = tmp_path / "theirs.csv"
    write_table(pd.DataFrame({"well": ["A01"], "treatment": ["x"]}),
                str(theirs))
    linked = pd.DataFrame({"plateID": ["plate1"], "_row": [1], "_col": [1],
                           "strain": ["RH"]})
    assert _plate_map_differences(linked, str(theirs), ["strain"],
                                  {"plate1": "BC001"}) == []
