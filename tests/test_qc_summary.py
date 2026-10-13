"""The QC dashboard: does it read the verdicts, and does it refuse to guess?

Three things are asserted throughout. It must **read** rather than score --
the segmentation reader is handed a stub and the test checks it was called
rather than that masks were opened. A check that never ran must come back
``missing`` and never ``ok``, because those are the two readings that matter
and conflating them is how a fresh project looks clean. And a card whose
inputs are newer than it is must say so without being downgraded: it
describes the previous run accurately, and turning it into a failure would
hide which of the two is actually wrong.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.qc_summary import (  # noqa: E402
    VERDICT_ORDER, Dashboard, QCCard, format_dashboard, read_dashboard,
    worst_verdict,
)


def _write_measurements(root, rows, table="cell"):
    """A measurements database with the units stamp on every row."""
    folder = os.path.join(str(root), "measurements")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "measurements.db")
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            f'CREATE TABLE IF NOT EXISTS "{table}" '
            '(object_label INTEGER, cell_area REAL, measurement_ndim INTEGER, '
            'measurement_units TEXT, n_z INTEGER, voxel_size_z_um REAL, '
            'voxel_size_xy_um REAL)')
        connection.executemany(
            f'INSERT INTO "{table}" VALUES (?, ?, ?, ?, ?, ?, ?)', rows)
        connection.commit()
    finally:
        connection.close()
    return path


TWO_D_ROW = (1, 120.0, 2, "px", 1, None, None)
THREE_D_ROW = (1, 608.4, 3, "um", 5, 2.0, 0.65)


# -- the verdict lattice ----------------------------------------------------


def test_missing_sorts_between_ok_and_warn():
    """A check that never ran is a gap, not evidence of a problem. Ranking it
    with `fail` would make an untouched project look broken; ranking it with
    `ok` would make it look checked."""
    assert VERDICT_ORDER.index("ok") < VERDICT_ORDER.index("missing")
    assert VERDICT_ORDER.index("missing") < VERDICT_ORDER.index("warn")
    assert worst_verdict(["ok", "missing"]) == "missing"
    assert worst_verdict(["missing", "warn"]) == "warn"
    assert worst_verdict(["ok", "fail", "warn"]) == "fail"
    assert worst_verdict([]) == "missing"
    assert worst_verdict(["nonsense"]) == "missing"


def test_a_stale_card_is_labelled_not_downgraded():
    card = QCCard(key="segmentation", title="Segmentation", verdict="ok",
                  stale=True)
    assert card.verdict == "ok"
    assert card.display_verdict == "ok (out of date)"
    # A missing card has nothing to be out of date about.
    assert QCCard(key="x", title="x", verdict="missing",
                  stale=True).display_verdict == "missing"


def test_the_dashboard_is_advisory_by_construction():
    """There is no field here a caller is meant to gate a Run button on, and
    the constant exists to say so in code."""
    assert Dashboard().blocks_run is False


# -- an empty project -------------------------------------------------------


def test_an_empty_project_is_missing_everywhere_and_says_what_to_run(tmp_path):
    dashboard = read_dashboard(tmp_path)
    assert dashboard.verdict == "missing"
    assert dashboard.card("segmentation").verdict == "missing"
    assert dashboard.card("units").verdict == "missing"
    assert dashboard.card("leakage").verdict == "missing"
    for card in dashboard.cards:
        if card.verdict == "missing":
            assert card.how_to_produce, f"{card.key} says nothing to do"
    text = format_dashboard(dashboard)
    assert "MISSING" in text
    assert "->" in text


def test_the_cards_keep_a_fixed_order(tmp_path):
    """A dashboard that reshuffles between refreshes makes the user re-find
    the card they were reading."""
    first = [card.key for card in read_dashboard(tmp_path).cards]
    _write_measurements(tmp_path, [THREE_D_ROW])
    second = [card.key for card in read_dashboard(tmp_path).cards]
    assert first == second
    assert first == ["segmentation", "units", "leakage", "plate", "agreement"]


# -- segmentation: read, do not score --------------------------------------


def test_the_segmentation_card_reads_a_digest_and_never_scores(tmp_path):
    """The reader is injected and the test asserts it was the thing called.
    Anything that opened a mask stack here would be the bug."""
    calls = []

    class _Digest:
        verdict = "warn"
        headline = "3 fields look over-segmented on plate1."
        subhead = "48 fields scored, 3 flagged."
        stale = False
        scorecards = ()

    def _reader(src):
        calls.append(src)
        return _Digest()

    dashboard = read_dashboard(tmp_path, segmentation_reader=_reader)
    assert len(calls) == 1
    card = dashboard.card("segmentation")
    assert card.verdict == "warn"
    assert "over-segmented" in card.headline
    assert "48 fields scored" in card.detail[0]
    assert dashboard.verdict == "warn"


def test_a_stale_scorecard_reaches_the_card_as_out_of_date(tmp_path):
    """seg_qc.read_digest dates each card against its mask stack; the
    dashboard carries that through rather than re-deriving it."""

    class _Card:
        path = "/tmp/qc/segmentation_qc_cell.csv"
        mtime = 100.0
        field_qcs = ()

    class _Digest:
        verdict = "ok"
        headline = "Nothing flagged."
        subhead = ""
        stale = True
        scorecards = (_Card(),)

    dashboard = read_dashboard(tmp_path, segmentation_reader=lambda s: _Digest())
    card = dashboard.card("segmentation")
    assert card.stale is True
    assert card.verdict == "ok"
    assert card.display_verdict == "ok (out of date)"
    assert dashboard.stale is True
    assert "out of date" in format_dashboard(dashboard)


def test_flag_explanations_come_from_seg_qc_not_from_here(tmp_path):
    """One vocabulary. The sentence on the card must be the sentence
    FLAG_GUIDANCE holds, not a paraphrase that will drift from it."""
    from spacr.seg_qc import FLAG_GUIDANCE, explain_flag

    flag = sorted(FLAG_GUIDANCE)[0]

    class _FieldQC:
        flags = (flag,)

    class _Card:
        path = ""
        mtime = 0.0
        field_qcs = (_FieldQC(),)

    class _Digest:
        verdict = "warn"
        headline = "h"
        subhead = ""
        stale = False
        scorecards = (_Card(),)

    dashboard = read_dashboard(tmp_path, segmentation_reader=lambda s: _Digest())
    detail = "\n".join(dashboard.card("segmentation").detail)
    assert explain_flag(flag).text() in detail


def test_an_unknown_flag_is_named_rather_than_dropped(tmp_path):
    """A flag with no guidance still gets a line. Dropping it would make the
    dashboard quieter than the truth."""

    class _FieldQC:
        flags = ("a_flag_nobody_wrote_guidance_for",)

    class _Card:
        path = ""
        mtime = 0.0
        field_qcs = (_FieldQC(),)

    class _Digest:
        verdict = "warn"
        headline = "h"
        subhead = ""
        stale = False
        scorecards = (_Card(),)

    dashboard = read_dashboard(tmp_path, segmentation_reader=lambda s: _Digest())
    detail = "\n".join(dashboard.card("segmentation").detail)
    assert "a_flag_nobody_wrote_guidance_for" in detail


def test_a_reader_that_raises_becomes_an_error_card_not_a_crash(tmp_path):
    def _boom(_src):
        raise RuntimeError("the disk went away")

    dashboard = read_dashboard(tmp_path, segmentation_reader=_boom)
    card = dashboard.card("segmentation")
    assert card.verdict == "error"
    assert "the disk went away" in card.headline
    assert dashboard.verdict == "error"


# -- units / dtype ----------------------------------------------------------


def test_a_consistently_stamped_table_passes(tmp_path):
    _write_measurements(tmp_path, [THREE_D_ROW, THREE_D_ROW])
    card = read_dashboard(tmp_path).card("units")
    assert card.verdict == "ok"
    assert "consistent" in card.headline
    assert card.source.endswith("measurements.db")


def test_a_table_holding_2d_and_3d_rows_together_fails(tmp_path):
    """The known answer: cell_area is px^2 in one row and um^3 in the other,
    and every query over the table silently mixes them."""
    _write_measurements(tmp_path, [TWO_D_ROW, THREE_D_ROW])
    card = read_dashboard(tmp_path).card("units")
    assert card.verdict == "fail"
    assert "different units" in card.headline
    assert any("cell_area" in line for line in card.detail)
    assert read_dashboard(tmp_path).verdict == "fail"


def test_an_unstamped_table_is_a_warning_not_a_pass(tmp_path):
    folder = tmp_path / "measurements"
    folder.mkdir()
    connection = sqlite3.connect(str(folder / "measurements.db"))
    try:
        connection.execute("CREATE TABLE cell (object_label INTEGER, a REAL)")
        connection.execute("INSERT INTO cell VALUES (1, 2.0)")
        connection.commit()
    finally:
        connection.close()
    card = read_dashboard(tmp_path).card("units")
    assert card.verdict == "warn"
    assert "no units stamp" in card.headline


def test_bookkeeping_tables_are_not_mistaken_for_measurements(tmp_path):
    """`settings` and `run_status` have no stamp and never will."""
    _write_measurements(tmp_path, [THREE_D_ROW])
    connection = sqlite3.connect(
        str(tmp_path / "measurements" / "measurements.db"))
    try:
        connection.execute("CREATE TABLE settings (key TEXT, value TEXT)")
        connection.execute("CREATE TABLE run_status (state TEXT)")
        connection.commit()
    finally:
        connection.close()
    card = read_dashboard(tmp_path).card("units")
    assert card.verdict == "ok"


def test_a_corrupt_database_is_an_error_card(tmp_path):
    folder = tmp_path / "measurements"
    folder.mkdir()
    (folder / "measurements.db").write_text("this is not sqlite",
                                            encoding="utf-8")
    card = read_dashboard(tmp_path).card("units")
    assert card.verdict == "error"


# -- leakage ----------------------------------------------------------------


def test_a_clean_leakage_audit_passes(tmp_path):
    bundle = tmp_path / "results" / "evaluation"
    bundle.mkdir(parents=True)
    (bundle / "leakage.json").write_text(json.dumps({
        "reports": [{"group_by": "well", "passed": True},
                    {"group_by": "plate", "passed": True}]}), encoding="utf-8")
    card = read_dashboard(tmp_path).card("leakage")
    assert card.verdict == "ok"
    assert "2 split boundaries held" in card.headline


def test_a_leaking_split_fails_and_says_what_the_number_means(tmp_path):
    bundle = tmp_path / "results"
    bundle.mkdir()
    (bundle / "leakage.json").write_text(json.dumps({
        "reports": [{"group_by": "well", "passed": False},
                    {"group_by": "plate", "passed": True}]}), encoding="utf-8")
    card = read_dashboard(tmp_path).card("leakage")
    assert card.verdict == "fail"
    assert "well" in card.headline
    assert "upper bound" in " ".join(card.detail), (
        "the card must say what a leaked accuracy actually is")


def test_a_malformed_leakage_file_is_an_error_not_a_pass(tmp_path):
    (tmp_path / "leakage.json").write_text("{not json", encoding="utf-8")
    card = read_dashboard(tmp_path).card("leakage")
    assert card.verdict == "error"


# -- plate and agreement ----------------------------------------------------


def test_the_plate_card_says_why_there_is_nothing_to_read(tmp_path):
    """An honest 'missing': spacr.plate_qc computes on demand and persists
    nothing, so there is genuinely no artifact. Saying that is more useful
    than an empty card."""
    card = read_dashboard(tmp_path).card("plate")
    assert card.verdict == "missing"
    assert "does not persist" in card.headline


def test_a_written_plate_verdict_is_read(tmp_path):
    (tmp_path / "plate_qc.json").write_text(json.dumps({
        "verdict": "warn",
        "headline": "Column 1 runs 22% below the plate median.",
        "detail": ["Edge column, consistent with evaporation."]}),
        encoding="utf-8")
    card = read_dashboard(tmp_path).card("plate")
    assert card.verdict == "warn"
    assert "22%" in card.headline
    assert card.detail == ["Edge column, consistent with evaporation."]


def test_agreement_bands_follow_the_usual_reading(tmp_path):
    for kappa, expected in ((0.81, "ok"), (0.55, "warn"), (0.2, "fail")):
        (tmp_path / "agreement.json").write_text(
            json.dumps({"kappa": kappa, "band": "b"}), encoding="utf-8")
        card = read_dashboard(tmp_path).card("agreement")
        assert card.verdict == expected, kappa
        if expected != "ok":
            assert "ceiling" in " ".join(card.detail), (
                "a low kappa should say what it caps")


def test_a_non_numeric_kappa_is_an_error(tmp_path):
    (tmp_path / "agreement.json").write_text(
        json.dumps({"kappa": "high"}), encoding="utf-8")
    assert read_dashboard(tmp_path).card("agreement").verdict == "error"


# -- the whole thing --------------------------------------------------------


def test_the_worst_card_sets_the_verdict_and_the_headline(tmp_path):
    _write_measurements(tmp_path, [TWO_D_ROW, THREE_D_ROW])
    (tmp_path / "agreement.json").write_text(
        json.dumps({"kappa": 0.9, "band": "almost perfect"}), encoding="utf-8")
    dashboard = read_dashboard(tmp_path)
    assert dashboard.verdict == "fail"
    assert dashboard.headline == dashboard.card("units").headline


def test_reading_a_project_does_not_walk_the_image_folders(tmp_path):
    """The cost rule. A project folder holds tens of thousands of crops, and
    a dashboard that walks them is one nobody opens twice."""
    for name in ("stack", "merged", "masks", "datasets"):
        folder = tmp_path / name
        folder.mkdir()
        for index in range(50):
            (folder / f"{index}.npy").write_bytes(b"")
    (tmp_path / "masks" / "leakage.json").write_text(
        json.dumps({"reports": [{"group_by": "well", "passed": False}]}),
        encoding="utf-8")

    started = time.time()
    dashboard = read_dashboard(tmp_path)
    assert time.time() - started < 5.0
    # The pruned folders are not searched, so the file planted in masks/ is
    # not found -- which is the intended behaviour, not an accident.
    assert dashboard.card("leakage").verdict == "missing"


def test_format_dashboard_survives_an_empty_one():
    assert "MISSING" in format_dashboard(Dashboard())
