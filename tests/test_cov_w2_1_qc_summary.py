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


def test_a_list_of_folders_is_read_as_its_first(tmp_path):
    """Screens hand this whatever `src` they were configured with."""
    assert qc._project_root([str(tmp_path), "/elsewhere"]) == str(tmp_path)
    assert qc._project_root([]) == ""


def test_no_source_at_all_is_no_root():
    """An unconfigured screen must not walk the working directory."""
    assert qc._project_root(None) == ""
    assert qc._project_root("") == ""


def test_a_file_stands_for_the_folder_it_is_in(tmp_path):
    """A user who dropped measurements.db meant the project around it."""
    path = tmp_path / "measurements.db"
    path.write_bytes(b"")

    assert qc._project_root(str(path)) == str(tmp_path)


def test_a_path_that_does_not_exist_is_no_root(tmp_path):
    """A stale path from a saved settings file, and not a crash."""
    assert qc._project_root(str(tmp_path / "gone")) == ""


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


def test_the_walk_gives_up_rather_than_costing_more_than_the_check(tmp_path):
    """A project folder can hold a hundred thousand crops."""
    for index in range(6):
        (tmp_path / f"crop_{index}.png").write_bytes(b"")
    (tmp_path / "leakage.json").write_text("{}", encoding="utf-8")

    path, mtime = qc._newest_under(str(tmp_path), "leakage.json", limit=2)

    assert (path, mtime) == ("", 0.0)


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


def test_a_leakage_file_with_no_reports_says_so(tmp_path):
    """An empty audit is not a passing audit."""
    _write_json(tmp_path / "leakage.json", {"reports": []})

    card = qc._read_leakage(str(tmp_path))

    assert card.verdict == "missing"
    assert "holds no split reports" in card.headline


def test_a_failed_boundary_names_it_and_says_what_it_means(tmp_path):
    """The accuracy is then an upper bound, not an estimate."""
    _write_json(tmp_path / "leakage.json", {"reports": [
        {"group_by": "well", "passed": False},
        {"group_by": "plate", "passed": True}]})

    card = qc._read_leakage(str(tmp_path))

    assert card.verdict == "fail"
    assert "1 of 2 split boundaries leaked (well)" in card.headline
    assert any("upper bound" in line for line in card.detail)


def test_a_leakage_file_that_will_not_parse_is_an_error_card(tmp_path):
    """Not a crash, and not a silent 'missing' either."""
    (tmp_path / "leakage.json").write_text("{not json", encoding="utf-8")

    card = qc._read_leakage(str(tmp_path))

    assert card.verdict == "error"
    assert "Could not read leakage.json" in card.headline


# ---------------------------------------------------------------------------
# Units


def _db(path, tables):
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        for name, frame in tables.items():
            frame.to_sql(name, db, index=False)
    return path


def test_a_database_that_will_not_open_is_an_error_card(tmp_path):
    """An unreadable file is reported, not raised."""
    db = _db(tmp_path / "measurements" / "measurements.db",
             {"cell": pd.DataFrame({"area": [1.0]})})
    os.chmod(db, 0o000)
    try:
        card = qc._read_units(str(tmp_path))
    finally:
        os.chmod(db, 0o644)

    assert card.verdict == "error"
    assert "Could not open measurements.db" in card.headline


def test_a_database_with_no_object_tables_says_that(tmp_path):
    """Settings and the crop list are not measurements."""
    _db(tmp_path / "measurements" / "measurements.db", {
        "settings": pd.DataFrame({"key": ["a"], "value": ["b"]}),
        "png_list": pd.DataFrame({"png_path": ["/a.png"]})})

    card = qc._read_units(str(tmp_path))

    assert card.verdict == "missing"
    assert card.headline == "The measurements database holds no object tables."


def test_two_runs_in_one_table_is_a_failure_that_names_the_columns(tmp_path):
    """`cell_area` then means px^2 in some rows and um^3 in others."""
    from spacr.measurement_schema import MEASUREMENT_STAMP_COLUMNS

    stamps = {name: ["a", "b"] for name in MEASUREMENT_STAMP_COLUMNS}
    _db(tmp_path / "measurements" / "measurements.db",
        {"cell": pd.DataFrame({"cell_area": [1.0, 2.0], **stamps})})

    card = qc._read_units(str(tmp_path))

    assert card.verdict == "fail"
    assert "1 table(s) hold rows measured in different units (cell)" in \
        card.headline
    assert any("no query over the table can tell them apart" in line
               for line in card.detail)


def test_a_table_with_no_stamp_is_a_warning_not_a_failure(tmp_path):
    """Written before the stamp existed; re-measure or take it on faith."""
    _db(tmp_path / "measurements" / "measurements.db",
        {"cell": pd.DataFrame({"cell_area": [1.0, 2.0]})})

    card = qc._read_units(str(tmp_path))

    assert card.verdict == "warn"
    assert "carry no units stamp" in card.headline


# ---------------------------------------------------------------------------
# Plate effects


def test_a_plate_verdict_carries_its_detail_lines(tmp_path):
    """The payload's detail may be one string or a list of them."""
    _write_json(tmp_path / "plate_qc.json",
                {"verdict": "warn", "headline": "Edge wells are dimmer.",
                 "detail": ["row A is 30% dimmer", "column 1 too"]})

    card = qc._read_plate(str(tmp_path))

    assert card.verdict == "warn"
    assert card.detail == ["row A is 30% dimmer", "column 1 too"]


def test_a_plate_verdict_with_a_single_detail_string(tmp_path):
    """One sentence is the common case and must not be split into characters."""
    _write_json(tmp_path / "plate_qc.json", {"detail": "one sentence"})

    card = qc._read_plate(str(tmp_path))

    assert card.detail == ["one sentence"]
    assert card.headline == "Plate QC ran."


def test_a_plate_file_that_will_not_parse_is_an_error_card(tmp_path):
    """Reported on the card, where the user is already looking."""
    (tmp_path / "plate_qc.json").write_text("{not json", encoding="utf-8")

    card = qc._read_plate(str(tmp_path))

    assert card.verdict == "error"
    assert "Could not read plate_qc.json" in card.headline


# ---------------------------------------------------------------------------
# Annotator agreement


def test_an_agreement_file_that_will_not_parse_is_an_error_card(tmp_path):
    """Same posture as every other reader."""
    (tmp_path / "agreement.json").write_text("{not json", encoding="utf-8")

    card = qc._read_agreement(str(tmp_path))

    assert card.verdict == "error"
    assert "Could not read agreement.json" in card.headline


def test_an_agreement_report_with_no_kappa_says_so(tmp_path):
    """There is nothing to judge, which is different from judging it poor."""
    _write_json(tmp_path / "agreement.json", {"band": "substantial"})

    card = qc._read_agreement(str(tmp_path))

    assert card.verdict == "missing"
    assert card.headline == "The agreement report holds no kappa."


def test_a_kappa_that_is_not_a_number_is_an_error(tmp_path):
    """Formatting it would produce a card that reads as a real verdict."""
    _write_json(tmp_path / "agreement.json", {"kappa": "high"})

    card = qc._read_agreement(str(tmp_path))

    assert card.verdict == "error"
    assert "kappa is not a number" in card.headline


@pytest.mark.parametrize("kappa, verdict", [
    (0.81, "ok"), (0.55, "warn"), (0.2, "fail"),
])
def test_the_kappa_bands_are_the_conventional_ones(tmp_path, kappa, verdict):
    """And below 0.6 the card explains what the ceiling on accuracy is."""
    _write_json(tmp_path / "agreement.json",
                {"kappa": kappa, "band": "measured"})

    card = qc._read_agreement(str(tmp_path))

    assert card.verdict == verdict
    assert f"kappa = {kappa:.2f}" in card.headline
    assert bool(card.detail) is (kappa < 0.6)


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
