"""Plate and import errors explain recovery while retaining read-only guards."""
from __future__ import annotations

import json
import sqlite3

import pandas as pd
import pytest

from spacr import foreign, plate_qc
from spacr.errors import ConfigurationError


@pytest.fixture
def plate_database(tmp_path):
    """Create a small database with one located table and one unlocated table."""
    path = tmp_path / "measurements.db"
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE cells (prc TEXT, value REAL)")
        connection.execute("INSERT INTO cells VALUES ('plate1_r1_c1', 2.0)")
        connection.execute("CREATE TABLE notes (value REAL)")
        connection.execute("INSERT INTO notes VALUES (3.0)")
        connection.commit()
    finally:
        connection.close()
    return path


def test_missing_plate_database_names_the_path_without_creating_it(tmp_path):
    """A missing database remains missing and the error explains how to select one."""
    path = tmp_path / "missing.db"
    with pytest.raises(FileNotFoundError) as caught:
        plate_qc._connect(str(path))
    assert str(path) in str(caught.value)
    assert "Select" in str(caught.value)
    assert not path.exists()


@pytest.mark.parametrize("problem, diagnostic, recovery", [
    ("table", "has no 'missing' table", "Reopen the database and select one of the listed tables"),
    ("column", "has no column 'missing'", "Reopen the database and select an available measurement column"),
    ("well_identity", "carries no well identifier", "Select a table containing well identifiers"),
])
def test_plate_table_refusal_keeps_the_database_and_names_recovery(
        plate_database, problem, diagnostic, recovery):
    """Missing schema elements remain refusals and never alter the database."""
    before = plate_database.read_bytes()
    with pytest.raises(ValueError) as caught:
        if problem == "table":
            plate_qc.table_columns(str(plate_database), "missing")
        else:
            plate_qc.load_plate_frame(
                str(plate_database), "cells" if problem == "column" else "notes",
                "missing" if problem == "column" else "value")
    assert diagnostic in str(caught.value)
    assert recovery in str(caught.value)
    assert plate_database.read_bytes() == before


@pytest.mark.parametrize("frame", [
    pd.DataFrame({"prc": ["unlocated"], "value": [2]}),
    pd.DataFrame({"value": [2]}),
])
def test_unusable_well_identities_are_not_replaced_with_guesses(frame):
    """Invalid or absent well identities keep the frame unchanged and name a repair."""
    before = frame.copy(deep=True)
    with pytest.raises(ValueError) as caught:
        plate_qc._identify_wells(frame)
    assert "well" in str(caught.value).lower()
    assert "source" in str(caught.value).lower()
    pd.testing.assert_frame_equal(frame, before)


def test_missing_barcode_column_names_the_column_without_changing_records():
    """Records without the requested barcode column are refused unchanged."""
    imaged = pd.DataFrame({"plateID": ["plate1"], "_row": [1], "_col": [1]})
    records = pd.DataFrame({"well": ["A01"], "compound": ["control"]})
    before = records.copy(deep=True)
    with pytest.raises(ValueError) as caught:
        plate_qc._link_barcode_wells(
            imaged, {"plate1": "BC1"}, records, barcode_column="sample_barcode")
    assert "sample_barcode" in str(caught.value)
    assert "well" in str(caught.value)
    assert "column" in str(caught.value)
    assert "Select the sample-records column that contains plate barcodes" in str(caught.value)
    pd.testing.assert_frame_equal(records, before)


@pytest.mark.parametrize("link, diagnostic", [
    ("https://other.test/page2", "original service origin"),
    (42, "URL or null"),
])
def test_invalid_lims_page_link_stops_before_another_request(link, diagnostic):
    """Unusable paging stays blocked without requesting a second page."""
    calls = []

    def fetch(url, headers):
        """Return a fixed local response and record the requested fixture URL."""
        calls.append(url)
        return json.dumps({"wells": [{"well": "A01"}], "next": link})

    with pytest.raises(ValueError) as caught:
        plate_qc._lims_records("https://example.test/plates/{barcode}", ["BC1"], fetch=fetch)
    assert diagnostic in str(caught.value)
    assert "LIMS" in str(caught.value)
    assert "Ask" in str(caught.value)
    assert calls == ["https://example.test/plates/BC1"]


@pytest.mark.parametrize("body, diagnostic, recovery", [
    ("not JSON", "not JSON", "Check the LIMS API address and authentication settings"),
    ('{"operator": "fixture"}', "has no list of well records", "Select the well-records API endpoint"),
    ('"not a record list"', "is not a list of well records", "Select the well-records API endpoint"),
])
def test_unusable_lims_response_keeps_the_barcode_and_returns_no_partial_rows(
        body, diagnostic, recovery):
    """Malformed response content remains an error with service recovery advice."""
    calls = []

    def fetch(url, headers):
        """Return the selected fixture response without any network request."""
        calls.append(url)
        return body

    with pytest.raises(ValueError) as caught:
        plate_qc._lims_records("https://example.test/plates/{barcode}", ["BC1"], fetch=fetch)
    assert "'BC1'" in str(caught.value)
    assert diagnostic in str(caught.value)
    assert "LIMS" in str(caught.value)
    assert recovery in str(caught.value)
    assert len(calls) == 1
    if body == "not JSON":
        assert caught.value.__cause__ is None
        assert caught.value.__suppress_context__


@pytest.mark.parametrize("limit, value, diagnostic", [
    ("_LIMS_MAX_PAGE_BYTES", 10, "byte limit"),
    ("_LIMS_MAX_TOTAL_BYTES", 10, "byte limit"),
    ("_LIMS_MAX_PAGES", 1, "page limit"),
    ("_LIMS_MAX_RECORDS", 1, "record limit"),
])
def test_lims_size_refusal_keeps_the_limit_and_explains_smaller_requests(
        monkeypatch, limit, value, diagnostic):
    """Bounded reads remain refusals with no silent truncation or extra requests."""
    monkeypatch.setattr(plate_qc, limit, value)
    calls = []

    def fetch(url, headers):
        """Supply a small local page that crosses the isolated test limit."""
        calls.append(url)
        return json.dumps({"wells": [{"well": "A01"}, {"well": "A02"}],
                           "next": f"?page={len(calls) + 1}"})

    with pytest.raises(ValueError) as caught:
        plate_qc._lims_records("https://example.test/plates/{barcode}", ["BC1"], fetch=fetch)
    assert diagnostic in str(caught.value)
    assert "LIMS" in str(caught.value)
    assert "Ask the LIMS administrator" in str(caught.value)
    assert "requested" in str(caught.value)
    assert getattr(plate_qc, limit) == value
    assert len(calls) == 1


def test_empty_foreign_column_map_explains_the_header_and_keeps_the_file(tmp_path):
    """An empty selected map is refused without populating an invented mapping."""
    path = tmp_path / "column_map.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ConfigurationError) as caught:
        foreign.load_column_map(str(path))
    assert str(path) in str(caught.value)
    assert "header" in str(caught.value)
    assert "Load a saved column-map CSV" in str(caught.value)
    assert path.read_bytes() == b""


def test_foreign_reader_failure_keeps_its_cause_and_original_file(tmp_path, monkeypatch):
    """A failed reader keeps its cause and explains how to check the selected table."""
    path = tmp_path / "measurements.csv"
    content = "well,value\nA01,2\n"
    path.write_text(content, encoding="utf-8")
    cause = OSError("fixture measurement read denied")

    def refuse_read(*args, **kwargs):
        """Refuse the isolated CSV read without opening or rewriting its contents."""
        raise cause

    monkeypatch.setattr(foreign.pd, "read_csv", refuse_read)
    with pytest.raises(ConfigurationError) as caught:
        foreign.read_measurements(str(path))
    assert caught.value.__cause__ is cause
    assert str(path) in str(caught.value)
    assert "Check" in str(caught.value)
    assert path.read_text(encoding="utf-8") == content


@pytest.mark.parametrize("problem, recovery", [
    ("conflicting", "return one consistent next-page URL"),
    ("cycle", "correct the repeating next-page link"),
])
def test_inconsistent_pagination_requires_a_service_fix_before_more_requests(problem, recovery):
    """Conflicting or repeated pages never return partial rows or repeat a fetch."""
    calls = []

    def fetch(url, headers):
        """Return an inconsistent local pagination fixture."""
        calls.append(url)
        payload = {"wells": [{"well": "A01"}]}
        if problem == "conflicting":
            payload.update(next="?page=2", next_url="?page=3")
        else:
            payload["next"] = "/plates/BC1"
        return json.dumps(payload)

    with pytest.raises(ValueError) as caught:
        plate_qc._lims_records("https://example.test/plates/{barcode}", ["BC1"], fetch=fetch)
    assert recovery in str(caught.value)
    assert calls == ["https://example.test/plates/BC1"]
