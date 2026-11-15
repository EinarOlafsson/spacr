"""The QC dashboard reading a project that is missing, broken, or clean.

Every reader here does a directory listing, a stat and a parse -- nothing is
recomputed -- so the interesting cases are all about what is on disk: a file
that will not parse, a database that will not open, a report with a field
missing. Each is built as a real file under ``tmp_path`` and read by the real
reader.
"""
from __future__ import annotations

import json
import os
import sqlite3

import pandas as pd
import pytest

from spacr.qt.widgets import qc_summary as qc


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The dashboard object


def test_a_card_that_is_not_there_is_none():
    """`card()` is how a screen asks for one panel; absence is an answer."""
    board = qc.Dashboard(cards=[qc.QCCard(key="units", title="Units")])

    assert board.card("units").title == "Units"
    assert board.card("leakage") is None


# ---------------------------------------------------------------------------
# Where a project root comes from


@pytest.mark.parametrize("reader", [
    qc._read_leakage, qc._read_units, qc._read_plate, qc._read_agreement,
])
def test_every_reader_reports_missing_without_a_project(reader):
    """Nothing checked is not the same as nothing wrong, and it says so."""
    card = reader("")

    assert card.verdict == "missing"
    assert card.how_to_produce


# ---------------------------------------------------------------------------
# The bounded walk


def test_the_walk_gives_up_rather_than_costing_more_than_the_check(
        tmp_path, monkeypatch):
    """A project folder can hold a hundred thousand crops.

    ``os.walk`` lists a folder in the file system's own order, so the test
    fixes the order: the file looked for comes after more files than the
    limit allows, and the walk must stop before reaching it.
    """
    crops = [f"crop_{index}.png" for index in range(6)]
    for crop in crops:
        (tmp_path / crop).write_bytes(b"")
    (tmp_path / "leakage.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        os, "walk",
        lambda root: iter([(str(tmp_path), [], crops + ["leakage.json"])]))

    path, mtime = qc._newest_under(str(tmp_path), "leakage.json", limit=2)

    assert (path, mtime) == ("", 0.0)

    path, mtime = qc._newest_under(str(tmp_path), "leakage.json", limit=7)

    assert path == str(tmp_path / "leakage.json") and mtime > 0.0


def test_a_file_that_cannot_be_stat_ed_is_skipped(tmp_path, monkeypatch):
    """A vanished or unreadable file is not a reason to fail the dashboard."""
    (tmp_path / "leakage.json").write_text("{}", encoding="utf-8")
    real_stat = os.stat

    def _refuse(path, *args, **kwargs):
        if str(path).endswith("leakage.json"):
            raise OSError("it went away")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", _refuse)

    assert qc._newest_under(str(tmp_path), "leakage.json") == ("", 0.0)


def test_a_database_that_cannot_be_stat_ed_is_still_found(tmp_path,
                                                          monkeypatch):
    """The path is what matters; the mtime is only used for staleness."""
    db = tmp_path / "measurements" / "measurements.db"
    db.parent.mkdir()
    db.write_bytes(b"")
    real_stat = os.stat
    seen = []

    def _vanishes_after_the_existence_check(path, *args, **kwargs):
        # `os.path.isfile` stats it first; the reader stats it again for the
        # mtime, and that is the call this test makes fail.
        if str(path).endswith("measurements.db"):
            seen.append(path)
            if len(seen) > 1:
                raise OSError("it went away")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", _vanishes_after_the_existence_check)

    assert qc._find_measurements_db(str(tmp_path)) == (str(db), 0.0)


# ---------------------------------------------------------------------------
# Flag explanations


def test_a_flag_with_no_guidance_is_still_named():
    """Dropping it silently would make the dashboard quieter than the truth."""
    scorecard = type("Card", (), {"field_qcs": [
        type("Field", (), {"flags": ["a_flag_nobody_documented"]})()]})()

    lines = qc._flag_explanations([scorecard])

    assert lines == ["a_flag_nobody_documented: no guidance is written for "
                     "this flag yet."]


def test_only_the_first_few_flags_are_listed_and_the_rest_are_counted():
    """A card that lists forty flags is a card nobody reads."""
    flags = [f"flag_{i}" for i in range(qc._MAX_FLAGS + 3)]
    scorecard = type("Card", (), {"field_qcs": [
        type("Field", (), {"flags": flags})()]})()

    lines = qc._flag_explanations([scorecard])

    assert len(lines) == qc._MAX_FLAGS + 1
    assert lines[-1] == "... and 3 more flag(s)."


# ---------------------------------------------------------------------------
# Leakage


# ---------------------------------------------------------------------------
# Units


def _db(path, tables):
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        for name, frame in tables.items():
            frame.to_sql(name, db, index=False)
    return path


# ---------------------------------------------------------------------------
# Plate effects


# ---------------------------------------------------------------------------
# Annotator agreement


# ---------------------------------------------------------------------------
# The whole dashboard


def _clean_project(tmp_path):
    """A project where every check on disk passes."""
    from spacr.measurement_schema import MEASUREMENT_STAMP_COLUMNS

    stamps = {name: ["one"] for name in MEASUREMENT_STAMP_COLUMNS}
    _db(tmp_path / "measurements" / "measurements.db",
        {"cell": pd.DataFrame({"cell_area": [1.0], **stamps})})
    _write_json(tmp_path / "leakage.json",
                {"reports": [{"group_by": "well", "passed": True}]})
    _write_json(tmp_path / "plate_qc.json",
                {"verdict": "ok", "headline": "No edge effect."})
    _write_json(tmp_path / "agreement.json", {"kappa": 0.9, "band": "almost"})
    return tmp_path


def test_a_project_where_everything_passes_says_so_once(tmp_path):
    """Five green cards read as one sentence, not as five."""
    root = _clean_project(tmp_path)
    ok_digest = type("Digest", (), {"verdict": "ok", "headline": "Masks fine.",
                                    "stale": False, "scorecards": (),
                                    "subhead": ""})()

    board = qc.read_dashboard(str(root),
                              segmentation_reader=lambda src: ok_digest)

    assert board.verdict == "ok"
    assert board.headline == "All 5 checks that have run are clean."
    assert not board.stale
    assert [card.key for card in board.cards] == [
        "segmentation", "units", "leakage", "plate", "agreement"]


def test_an_unchecked_project_names_every_check_that_never_ran(tmp_path):
    """"Nothing has been checked" is a real finding, not an all-clear."""
    missing = type("Digest", (), {"verdict": "missing", "headline": "",
                                  "stale": False, "scorecards": (),
                                  "subhead": ""})()

    board = qc.read_dashboard(str(tmp_path),
                              segmentation_reader=lambda src: missing)

    assert board.verdict == "missing"
    assert board.headline.startswith("Nothing has been checked yet:")
    assert "Annotator agreement" in board.headline
    assert "-> " in qc.format_dashboard(board)
