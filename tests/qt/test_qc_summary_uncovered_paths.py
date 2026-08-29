"""What the QC dashboard does when its own vocabulary is unavailable, and
when a verdict file parses but is not a verdict.

Two families live here. The readers borrow their names and their words from
other modules -- ``seg_qc.FLAG_GUIDANCE``, ``classifier_evaluation``'s file
names, the measurement stamp columns -- and each borrow is written to fall
back rather than fail. The fallbacks are exercised by removing the borrowed
name, which is what an older install or a half-imported package looks like
from inside the reader.

The second family is a file that is valid JSON and still not a mapping: a
list, a bare ``null``. Every reader here promises an error *card* rather than
an exception, because one unreadable artifact must not take the other four
verdicts off the screen with it.
"""
from __future__ import annotations

import json
import os
import sqlite3

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import qc_summary as qc  # noqa: E402


def _write_json(folder, name, payload_text):
    """A file that exists and holds exactly ``payload_text``."""
    os.makedirs(str(folder), exist_ok=True)
    path = os.path.join(str(folder), name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(payload_text)
    return path


class _FieldQC:
    def __init__(self, flags):
        self.flags = tuple(flags)


class _Scorecard:
    def __init__(self, path="", mtime=0.0, flags=()):
        self.path = path
        self.mtime = mtime
        self.field_qcs = (_FieldQC(flags),)


class _Digest:
    def __init__(self, scorecards):
        self.verdict = "warn"
        self.headline = "one plate has flagged fields"
        self.subhead = "3 of 40 fields flagged"
        self.stale = False
        self.scorecards = tuple(scorecards)


# ---------------------------------------------------------------------------
# The borrowed vocabulary is missing


def test_a_seg_qc_without_read_digest_is_an_error_card_not_a_crash(tmp_path,
                                                                   monkeypatch):
    """The reader is imported at call time, so its absence is a runtime
    condition like any other: it must land on the card, naming the failure."""
    from spacr import seg_qc

    monkeypatch.delattr(seg_qc, "read_digest")

    card = qc._read_segmentation(str(tmp_path))

    assert card.verdict == "error"
    assert card.headline.startswith(
        "Could not read the segmentation scorecards: ")
    assert "read_digest" in card.headline
    assert card.detail == []


def test_flags_go_unnamed_when_the_guidance_table_cannot_be_imported(
        tmp_path, monkeypatch):
    """`FLAG_GUIDANCE` is the only place those sentences are written. Without
    it the card keeps the digest's own subhead and says nothing about the
    individual flags rather than inventing a second vocabulary."""
    from spacr import seg_qc

    digest = _Digest([_Scorecard(path=str(tmp_path / "plate1.json"),
                                 mtime=1000.0, flags=("empty_field",))])

    with_guidance = qc._read_segmentation(str(tmp_path),
                                          reader=lambda src: digest)
    assert with_guidance.detail == ["3 of 40 fields flagged",
                                   seg_qc.explain_flag("empty_field").text()]

    monkeypatch.delattr(seg_qc, "FLAG_GUIDANCE")
    without_guidance = qc._read_segmentation(str(tmp_path),
                                             reader=lambda src: digest)

    assert without_guidance.detail == ["3 of 40 fields flagged"]
    assert without_guidance.verdict == "warn"
    assert without_guidance.headline == "one plate has flagged fields"


def test_the_leakage_reader_falls_back_to_the_conventional_file_name(
        tmp_path, monkeypatch):
    """The bundle's file names are a contract owned by
    ``classifier_evaluation``. If that contract cannot be read, the reader
    still knows what the file has always been called."""
    from spacr import classifier_evaluation

    _write_json(tmp_path, "leakage.json", json.dumps({"reports": [
        {"group_by": "well", "passed": True},
        {"group_by": "plate", "passed": True},
    ]}))
    monkeypatch.delattr(classifier_evaluation, "EVALUATION_FILES")

    card = qc._read_leakage(str(tmp_path))

    assert card.verdict == "ok"
    assert card.headline == (
        "All 2 split boundaries held; no related samples crossed a split.")
    assert os.path.basename(card.source) == "leakage.json"


def test_the_units_reader_falls_back_to_the_two_column_stamp(tmp_path,
                                                            monkeypatch):
    """The fallback stamp is ndim and units only. A table carrying just
    ``n_z`` is stamped under the full contract and unstamped under the
    fallback, so the verdict says which one the reader used."""
    from spacr import measurement_schema

    folder = tmp_path / "measurements"
    folder.mkdir()
    connection = sqlite3.connect(str(folder / "measurements.db"))
    try:
        connection.execute(
            'CREATE TABLE "cell" (object_label INTEGER, cell_area REAL, '
            "n_z INTEGER)")
        connection.execute('INSERT INTO "cell" VALUES (1, 120.0, 1)')
        connection.commit()
    finally:
        connection.close()

    stamped = qc._read_units(str(tmp_path))
    assert stamped.verdict == "ok"

    monkeypatch.delattr(measurement_schema, "MEASUREMENT_STAMP_COLUMNS")
    fell_back = qc._read_units(str(tmp_path))

    assert fell_back.verdict == "warn"
    assert fell_back.headline == (
        "1 table(s) carry no units stamp, so a 2-D and a 3-D run cannot be "
        "told apart in them.")


# ---------------------------------------------------------------------------
# Valid JSON that is not a verdict


@pytest.mark.parametrize("payload_text", ["[1, 2, 3]", "null", '"leaked"'])
def test_a_leakage_file_that_is_not_an_object_is_an_error_card(tmp_path,
                                                               payload_text):
    """`json.loads` succeeds on a list, a null and a bare string. None of
    them is a report, and reaching into one must not escape the reader."""
    _write_json(tmp_path, "leakage.json", payload_text)

    card = qc._read_leakage(str(tmp_path))

    assert card.verdict == "error"
    assert card.headline.startswith("Could not read leakage.json: ")


def test_a_plate_file_that_is_not_an_object_is_an_error_card(tmp_path):
    _write_json(tmp_path, "plate_qc.json", "[]")

    card = qc._read_plate(str(tmp_path))

    assert card.verdict == "error"
    assert card.headline.startswith("Could not read plate_qc.json: ")
    assert card.detail == []


def test_an_agreement_file_that_is_not_an_object_is_an_error_card(tmp_path):
    _write_json(tmp_path, "agreement.json", "null")

    card = qc._read_agreement(str(tmp_path))

    assert card.verdict == "error"
    assert card.headline.startswith("Could not read agreement.json: ")


def test_one_unreadable_artifact_does_not_take_the_other_verdicts_down(
        tmp_path):
    """The point of the dashboard is seeing five verdicts at once. A plate
    file holding a JSON list must cost exactly its own card."""
    _write_json(tmp_path, "plate_qc.json", "[]")
    _write_json(tmp_path, "agreement.json",
                json.dumps({"kappa": 0.81, "band": "almost perfect"}))

    dashboard = qc.read_dashboard(str(tmp_path),
                                  segmentation_reader=lambda src: None)

    assert [card.key for card in dashboard.cards] == [
        "segmentation", "units", "leakage", "plate", "agreement"]
    assert dashboard.card("plate").verdict == "error"
    assert dashboard.card("agreement").verdict == "ok"
    assert dashboard.card("agreement").headline == (
        "Annotator agreement kappa = 0.81 (almost perfect).")
    assert dashboard.verdict == "error"
    assert "plate_qc.json" in qc.format_dashboard(dashboard)


# ---------------------------------------------------------------------------
# Two candidates, one card


def test_a_flag_raised_by_several_fields_is_explained_once(tmp_path):
    """Duplicate flags are the ordinary case -- one bad channel flags forty
    fields the same way. The card carries the vocabulary, not the tally, so
    each flag contributes one line however often it was raised."""
    from spacr import seg_qc

    scorecard = _Scorecard()
    scorecard.field_qcs = (
        _FieldQC(("empty_field", "empty_field")),
        _FieldQC(("empty_field", "near_empty_field")),
    )

    lines = qc._flag_explanations([scorecard])

    assert lines == [seg_qc.explain_flag("empty_field").text(),
                     seg_qc.explain_flag("near_empty_field").text()]


def test_two_verdict_files_of_the_same_age_still_make_one_card(
        tmp_path, monkeypatch):
    """Two plate folders written by one run share a timestamp to the second.
    The walk keeps a single best rather than accumulating both, and a tie
    leaves the one already held."""
    written = "1700000000"
    payload = json.dumps({"verdict": "warn", "headline": "row A runs dim"})
    left = _write_json(tmp_path / "plate_1", "plate_qc.json", payload)
    right = _write_json(tmp_path / "plate_2", "plate_qc.json", payload)
    for path in (left, right):
        os.utime(path, (int(written), int(written)))

    # The walk order decides which of the two the reader meets first, so pin
    # it: a tie must leave the first one held, not swap to the last seen.
    real_walk = os.walk

    def in_a_fixed_order(top, *args, **kwargs):
        return sorted(real_walk(top, *args, **kwargs), key=lambda row: row[0])

    monkeypatch.setattr(qc.os, "walk", in_a_fixed_order)

    card = qc._read_plate(str(tmp_path))

    assert card.source == left
    assert card.mtime == float(written)
    assert card.verdict == "warn"
    assert card.headline == "row A runs dim"
