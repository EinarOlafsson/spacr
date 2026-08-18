"""The Measurements tab as a workflow: instruction 154 sections A-E.

WHAT EACH SECTION OF THESE TESTS IS GUARDING, because none of them is about
the widget:

A. THE FREEZE.  ``merge`` used to run inside the button's own click handler --
   four databases, 226,467 cell rows, three joined tables -- so Qt could not
   paint, could not show a spinner and could not accept a cancel until it
   returned.  **The application was not hung; it was working, and had no way
   to say so.**  The test that matters is the one that asks WHICH THREAD the
   join ran on, because a progress label that is never repainted looks exactly
   like a frozen one.

C. `file_name` AND `path_name` CANNOT TAKE A MEAN.  The pre-merge plan matched
   column NAMES against ``AGGREGATION_RULES`` and reported everything left
   over as "would take the default (mean)".  Measured: the merge itself asks
   the DTYPE (``aggregation_plan``), so a string takes ``first`` -- the merge
   was right and the sentence was wrong.  The `first` was silent, though, even
   where the identifier was NOT constant within the group, which is the
   ambiguity instruction 79 item 2 says must be named rather than resolved.

B. The count is the sentence; the list is the evidence.

D. A plate called `plate1` is shown as `plate1` -- and the doubled `pp` is in
   the DATABASE, not in the rendering, which makes it a key question and not a
   cosmetic one.

E. "Nothing to scan" asserted a cause it had not checked, WITH FOUR DATABASES
   LOADED.
"""
from __future__ import annotations

import os
import sqlite3
import threading

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


# --------------------------------------------------------------------------- #
#  A database in spaCR's own shape, carrying the two TEXT columns 154 C is about
# --------------------------------------------------------------------------- #

def _database(directory, plate, *, cells=3, pathogens=(1, 1, 2, 2),
              paths_differ=False, name="measurements.db"):
    """One plate's measurements.db, with `file_name` and `path_name`.

    ``paths_differ`` makes the two pathogens of cell 1 come off different
    images -- the ambiguity that must be refused rather than resolved by
    picking one.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(str(directory), name)
    identity = {"rowID": "r1", "columnID": "c1", "fieldID": "f1"}
    stem = f"{plate}_r1_c1_f1"

    cell = pd.DataFrame({
        "plateID": [plate] * cells,
        **{k: [v] * cells for k, v in identity.items()},
        "object_label": list(range(1, cells + 1)),
        "file_name": [f"{stem}.tif"] * cells,
        "path_name": [f"/data/{plate}/{stem}.tif"] * cells,
        "area": [100.0 * i for i in range(1, cells + 1)],
        "wobble": [1.0] * cells,
    })
    n = len(pathogens)
    files = [f"{stem}.tif"] * n
    paths = [f"/data/{plate}/{stem}.tif"] * n
    if paths_differ:
        paths[1] = f"/data/{plate}/{stem}_second.tif"
    pathogen = pd.DataFrame({
        "plateID": [plate] * n, **{k: [v] * n for k, v in identity.items()},
        "cell_id": list(pathogens),
        "object_label": [1, 2] * (n // 2),
        "file_name": files,
        "path_name": paths,
        "pathogen_area": [10.0, 30.0, 50.0, 70.0][:n],
        "pathogen_wobble": [1.0, 2.0, 3.0, 4.0][:n],
    })
    with sqlite3.connect(path) as db:
        cell.to_sql("cell", db, index=False)
        pathogen.to_sql("pathogen", db, index=False)
    return path


def _rows(paths, plates=None):
    plates = plates or [f"plate{i + 1}" for i in range(len(paths))]
    return [{"plate": plates[i], "score": "s.csv", "count": "c.csv",
             "database": path} for i, path in enumerate(paths)]


@pytest.fixture()
def two_plates(tmp_path):
    return [_database(tmp_path / "plate1", "plate1"),
            _database(tmp_path / "plate2", "plate2")]


def _panel(qtbot, paths, *, threaded=False, plates=None):
    from spacr.qt.widgets.measurement_scan_panel import DatabaseMergePanel

    widget = DatabaseMergePanel(lambda: _rows(paths, plates),
                                threaded=threaded)
    qtbot.addWidget(widget)
    return widget


# --------------------------------------------------------------------------- #
#  A. THE FREEZE
# --------------------------------------------------------------------------- #

def test_the_merge_runs_off_the_gui_thread(qtbot, two_plates, monkeypatch):
    """The whole of 154 A, and the only test of it that cannot be faked.

    A progress label that is never repainted is indistinguishable from a
    frozen window, so what is asserted is WHICH THREAD the join ran on. Before
    this change the answer was the GUI thread, because `merge_across_databases`
    was called from the button's `clicked` handler.
    """
    from spacr.qt.widgets import measurement_scan_panel as module

    gui_thread = threading.current_thread()
    ran_on = []
    real = module.merge_across_databases

    def spy(*args, **kwargs):
        ran_on.append(threading.current_thread())
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "merge_across_databases", spy)
    panel = _panel(qtbot, two_plates, threaded=True)
    finished = []
    panel.merge_finished.connect(finished.append)

    assert panel.start_merge() is True
    # The click handler has ALREADY returned while the join is still going.
    assert panel.is_merging() is True
    qtbot.waitUntil(lambda: bool(finished), timeout=30000)

    assert ran_on and gui_thread not in ran_on, ran_on
    assert finished[0] is not None and len(finished[0]) == 6


def test_the_window_can_say_it_is_working_and_take_a_cancel(qtbot, two_plates):
    """While the merge runs, Merge is disabled and Stop is not -- which is the
    application saying "I am working" rather than looking hung."""
    panel = _panel(qtbot, two_plates, threaded=True)
    finished = []
    panel.merge_finished.connect(finished.append)

    assert panel.merge_button.isEnabled() is True
    assert panel.cancel_button.isEnabled() is False
    panel.start_merge()
    assert panel.merge_button.isEnabled() is False
    assert panel.cancel_button.isEnabled() is True

    qtbot.waitUntil(lambda: bool(finished), timeout=30000)
    assert panel.merge_button.isEnabled() is True
    assert panel.cancel_button.isEnabled() is False


def test_the_merge_names_its_stage_and_counts_against_the_plans_own_total(
        qtbot, two_plates):
    """"say what stage it is on ... show rows processed against rows expected
    -- the plan ALREADY KNOWS the total, it prints it."" So the denominator is
    checked against the plan rather than against a number invented here."""
    from spacr.multi_database import describe_merge

    panel = _panel(qtbot, two_plates)
    stages = []
    panel.merge_progress.connect(lambda *a: stages.append(a))

    assert panel.merge() is not None

    expected = sum(describe_merge(two_plates, table).total_rows
                   for table in ("cell", "pathogen"))
    named = [stage for stage, _done, _total in stages]
    assert any("reading cell from plate1" == stage for stage in named), named
    assert any(stage.startswith("aggregating pathogen") for stage in named)
    assert any("joining pathogen onto cell" in stage for stage in named)
    totals = {total for _stage, _done, total in stages if total}
    assert totals == {expected}, (totals, expected)
    assert max(done for _s, done, _t in stages) == expected


def test_a_cancelled_merge_leaves_nothing_half_written(qtbot, two_plates):
    """"be cancellable, leaving nothing half-written." Nothing is written
    until the merge RETURNS, so the previous frame has to survive intact."""
    panel = _panel(qtbot, two_plates)
    first = panel.merge()
    assert first is not None

    def stop_at(stage, _done, _total):
        if stage.startswith("read cell"):
            panel._stop.set()

    panel.merge_progress.connect(stop_at)
    assert panel.merge() is None
    assert panel.frame is first
    assert "Stopped" in panel.report.toPlainText()
    assert "none of them were kept" in panel.report.toPlainText()


def test_stopping_a_running_merge_reports_it_as_stopped_not_as_refused(
        qtbot, two_plates):
    """A refusal is an ANSWER about the data; a cancel is the user changing
    their mind. Wording them the same puts "Refused" in front of somebody who
    pressed Stop."""
    panel = _panel(qtbot, two_plates, threaded=True)
    finished = []
    panel.merge_finished.connect(finished.append)

    panel.start_merge()
    assert panel.cancel_merge() is True

    assert finished == [None]
    assert panel.is_merging() is False
    assert "Refused" not in panel.report.toPlainText()
    assert "Stopped" in panel.report.toPlainText()
    assert "untouched" in panel.report.toPlainText()
    panel._jobs.shutdown()


def test_a_second_merge_click_does_not_start_a_second_join(qtbot, two_plates):
    """Two joins over the same databases racing to set `_frame` is a result
    whose provenance depends on which finished first."""
    panel = _panel(qtbot, two_plates, threaded=True)
    finished = []
    panel.merge_finished.connect(finished.append)

    assert panel.start_merge() is True
    assert panel.start_merge() is False

    qtbot.waitUntil(lambda: bool(finished), timeout=30000)


# --------------------------------------------------------------------------- #
#  C. file_name and path_name cannot take a mean
# --------------------------------------------------------------------------- #

def test_a_text_identifier_is_never_reported_as_taking_a_mean(qtbot,
                                                              two_plates):
    """The reported bug, exactly. Both defaulted lists BEGAN with `file_name,
    path_name`, and "would take the default (mean)" is not merely unhelpful
    for a string -- it is impossible, and it is not what happens."""
    panel = _panel(qtbot, two_plates)
    plan = panel.plan_text()

    default_lines = [line for line in plan.splitlines()
                     if "would take the default" in line]
    assert default_lines, plan
    for line in default_lines:
        assert "file_name" not in line, line
        assert "path_name" not in line, line
    assert "NUMERIC" in "\n".join(default_lines)
    # ...and they are named as what they are.
    assert "TEXT identifier(s)" in plan
    assert "text takes no mean" in plan


def test_the_no_rule_bucket_is_split_by_dtype_before_it_is_counted():
    """"a 'no rule matched' bucket that mixes 83 numeric texture features with
    2 filesystem paths is not one bucket." Qt-free, because the classification
    is a fact about the data rather than about the widget."""
    from spacr.plate_measurements import classify_default_columns

    columns = ["texture_1", "texture_2", "file_name", "path_name", "mystery"]
    kinds = {"texture_1": "numeric", "texture_2": "numeric",
             "file_name": "text", "path_name": "text", "mystery": "unknown"}

    buckets = classify_default_columns(columns, kinds)

    assert buckets["mean"] == ("texture_1", "texture_2")
    assert buckets["identifier"] == ("file_name", "path_name")
    # SAY WHAT A NUMBER CANNOT SAY: a column with no declared type is named
    # rather than folded into either answer.
    assert buckets["unknown"] == ("mystery",)


def test_a_column_a_rule_names_is_in_no_bucket_at_all():
    """`area` is SUMMED by rule. Reporting it as a fall-through would bury the
    columns nobody thought about among the ones that are fine."""
    from spacr.plate_measurements import classify_default_columns

    buckets = classify_default_columns(["pathogen_area", "file_name"],
                                       {"pathogen_area": "numeric",
                                        "file_name": "text"})

    assert buckets["mean"] == ()
    assert buckets["identifier"] == ("file_name",)


def test_an_identifier_the_user_overrode_is_theirs_and_is_not_reclassified():
    """The rules are right most of the time, and an explicit choice beats
    every one of them -- including this classification."""
    from spacr.plate_measurements import classify_default_columns

    buckets = classify_default_columns(["file_name"], {"file_name": "text"},
                                       overrides={"file_name": "first"})

    assert buckets == {"mean": (), "identifier": (), "unknown": ()}


def test_a_constant_text_identifier_is_carried_through_as_first(qtbot,
                                                               two_plates):
    """Two pathogens of one cell off the SAME image have one path_name
    between them, so carrying it invents nothing."""
    from spacr.merge_tables import TEXT_AGGREGATION

    panel = _panel(qtbot, two_plates)
    frame = panel.merge().set_index(["plateID", "object_label"])

    assert TEXT_AGGREGATION == "first"
    assert "pathogen_path_name" in frame.columns
    assert frame.loc[("plate1", 1), "pathogen_path_name"] == \
        "/data/plate1/plate1_r1_c1_f1.tif"
    assert "constant within every group" in panel.report.toPlainText()


def test_a_text_identifier_that_differs_within_a_cell_is_refused_and_named(
        qtbot, tmp_path):
    """Instruction 79 item 2, and 154 C: "a file name that differs across the
    cells being combined is a genuine ambiguity and picking one silently
    invents provenance". So the column is LEFT OUT and named -- and the other
    columns are not lost with it."""
    paths = [_database(tmp_path / "p1", "plate1", paths_differ=True),
             _database(tmp_path / "p2", "plate2")]
    panel = _panel(qtbot, paths)

    frame = panel.merge()

    assert "pathogen_path_name" not in frame.columns
    # The unambiguous one beside it is untouched, and so is every measurement.
    assert "pathogen_file_name" in frame.columns
    assert "pathogen_area" in frame.columns
    said = panel.report.toPlainText()
    assert "path_name is a text identifier that differs WITHIN" in said, said
    assert "invents provenance" in said
    assert "_second.tif" in said, "the example from the offending group"
    assert frame.attrs["refused_identifiers"]["pathogen"]["path_name"]["groups"]


def test_the_old_silent_pick_is_still_reachable_but_has_to_be_asked_for(
        tmp_path):
    """"Refuse rather than fall back silently where the fallback would be
    presented as the thing that was asked for." A caller who genuinely wants
    one of the values has to say so in writing."""
    from spacr.qt.widgets.measurement_scan_panel import merge_across_databases

    paths = [_database(tmp_path / "p1", "plate1", paths_differ=True)]

    refused = merge_across_databases(paths, ["cell", "pathogen"])
    picked = merge_across_databases(paths, ["cell", "pathogen"],
                                    on_ambiguous_identifier="first")

    assert "pathogen_path_name" not in refused.columns
    assert "pathogen_path_name" in picked.columns


def test_a_numeric_label_that_differs_across_children_is_not_an_ambiguity(
        qtbot, two_plates):
    """`object_label` also takes `first`, by the identity rule, and it is
    SUPPOSED to differ -- three pathogens in a cell are labelled 1, 2 and 3.
    Refusing it would delete the column the rules carry verbatim so a row can
    be traced back."""
    panel = _panel(qtbot, two_plates)

    frame = panel.merge()

    assert "pathogen_object_label" in frame.columns
    assert not frame.attrs["refused_identifiers"]


def test_an_unaggregated_anchor_column_is_not_examined_for_ambiguity(
        qtbot, two_plates):
    """The anchor is one row per cell, so there is no group to be ambiguous
    within and `cell_path_name` is simply the cell's own value."""
    panel = _panel(qtbot, two_plates)

    frame = panel.merge()

    assert "cell_path_name" in frame.columns
    assert "cell" not in (frame.attrs["refused_identifiers"] or {})


# --------------------------------------------------------------------------- #
#  B. The count is the sentence; the list is the evidence
# --------------------------------------------------------------------------- #

def test_the_box_carries_the_count_and_the_disclosure_carries_the_names(
        qtbot, two_plates):
    """"85 columns match no aggregation rule and would take the default
    (mean)" is the SENTENCE. The list is the EVIDENCE, and evidence goes
    behind a disclosure, not in front of the summary."""
    panel = _panel(qtbot, two_plates)
    panel.merge()

    summary = panel.report.toPlainText()
    evidence = panel.details.toPlainText()

    assert "NUMERIC column(s) matched no aggregation rule" in summary
    assert "pathogen_wobble" not in summary
    assert "pathogen_wobble" in evidence
    # Nothing is lost: the panel still SAYS it, one click away.
    assert "pathogen_wobble" in panel.statement()


def test_the_disclosure_starts_collapsed_and_opens(qtbot, two_plates):
    """A foldable that is open by default is a wall of text with a chevron."""
    panel = _panel(qtbot, two_plates)
    panel.merge()

    assert panel.evidence.is_expanded() is False
    panel.evidence.set_expanded(True)
    assert panel.evidence.is_expanded() is True


def test_the_plan_summary_is_shorter_than_the_plan_it_summarises(qtbot,
                                                                two_plates):
    """The measurement behind 154 B: the box is 190 pixels tall and the plan
    printed a hundred and seventy column names into it."""
    panel = _panel(qtbot, two_plates)

    assert len(panel.plan_summary()) < len(panel.plan_text())
    assert panel.plan_evidence()
    assert panel.plan_evidence() in panel.plan_text()


# --------------------------------------------------------------------------- #
#  D. The plate ids read `pplate1`
# --------------------------------------------------------------------------- #

def test_a_plate_stored_as_pplate1_is_shown_as_the_plate_is_called(qtbot,
                                                                   tmp_path):
    """Measured before deciding it was cosmetic: the panel prints exactly what
    is stored, so the doubling is in the DATABASE. It is shown as the plate is
    called, and the stored id is named beside it."""
    paths = [_database(tmp_path / "p1", "pplate1")]
    panel = _panel(qtbot, paths, plates=["plate1"])

    plan = panel.plan_text()

    assert "plates plate1" in plan, plan
    assert "stored as pplate1" in plan
    assert "shown as plate1" in plan


def test_the_doubled_prefix_is_named_as_a_key_risk_not_a_rendering_choice(
        qtbot, tmp_path):
    """The join INSIDE this merge is safe -- both sides read the same stored
    value out of the same file. The risk is the score CSV, which
    `utils.correct_metadata` has already normalised. That is the sentence."""
    paths = [_database(tmp_path / "p1", "pplate1")]
    panel = _panel(qtbot, paths, plates=["plate1"])

    plan = panel.plan_text()

    assert "correct_metadata" in plan
    assert "unaffected" in plan
    assert "will not meet a score file" in plan


def test_a_plate_that_is_already_canonical_says_nothing_at_all(qtbot,
                                                              two_plates):
    """A warning that fires on every project is a warning nobody reads."""
    panel = _panel(qtbot, two_plates)

    assert "stored as" not in panel.plan_text()


# --------------------------------------------------------------------------- #
#  E. "Nothing to scan" named a cause it had not checked
# --------------------------------------------------------------------------- #

def test_nothing_to_scan_names_which_half_is_missing(qtbot, two_plates):
    """The maintainer saw the old sentence WITH FOUR DATABASES LOADED. It
    named two things a well must carry and checked neither."""
    from spacr.qt.widgets.measurement_scan_panel import MeasurementScanPanel

    panel = MeasurementScanPanel(frame_provider=lambda: None,
                                 database_provider=lambda: _rows(two_plates),
                                 threaded=False)
    qtbot.addWidget(panel)

    assert panel.run_scan() is False
    said = panel._status.text()

    assert "THE GENE HALF IS MISSING" in said, said
    # And it is right about the half that IS there.
    assert "2 measurement database(s) are attached" in said
    assert "regression_data.csv" in said


def test_nothing_to_scan_says_when_the_measurements_are_merged_too(
        qtbot, two_plates):
    """A merged frame is a fact about what this tab is holding, and the old
    message could not mention it."""
    from spacr.qt.widgets.measurement_scan_panel import MeasurementScanPanel

    panel = MeasurementScanPanel(frame_provider=lambda: None,
                                 database_provider=lambda: _rows(two_plates),
                                 threaded=False)
    qtbot.addWidget(panel)
    panel.databases.merge()

    panel.run_scan()

    assert "merged into 6 cell rows" in panel._status.text()


def test_nothing_to_scan_distinguishes_no_provider_from_no_run(qtbot):
    """Two different absences with two different remedies. The old sentence
    gave one remedy for both, and it was the wrong one for this case."""
    from spacr.qt.widgets.measurement_scan_panel import MeasurementScanPanel

    panel = MeasurementScanPanel(threaded=False)
    qtbot.addWidget(panel)

    assert panel.run_scan() is False
    assert "No source of well-level data is wired" in panel._status.text()
    assert "no measurement database is attached" in panel._status.text()


def test_an_empty_run_frame_is_not_reported_as_a_missing_run(qtbot,
                                                             two_plates):
    """A run that produced a table with no rows is a different fact from a run
    that was never loaded."""
    from spacr.qt.widgets.measurement_scan_panel import MeasurementScanPanel

    panel = MeasurementScanPanel(frame_provider=lambda: pd.DataFrame(),
                                 database_provider=lambda: _rows(two_plates),
                                 threaded=False)
    qtbot.addWidget(panel)

    assert panel.run_scan() is False
    assert "has no rows" in panel._status.text()


def test_two_halves_that_do_not_join_name_the_key_and_one_example_each():
    """"if they do not join, name the key that failed and show one example
    from each side." Qt-free: it is a fact about two frames."""
    from spacr.qt.widgets.measurement_scan_panel import describe_key_overlap

    left = pd.DataFrame({"plateID": ["plate1"], "rowID": ["r1"],
                         "columnID": ["c1"]})
    right = pd.DataFrame({"prc": ["plate9_r4_c4"], "gene": ["g"]})

    said = describe_key_overlap("merged measurements", left, "loaded run",
                                right)

    assert "share no well" in said
    assert "'plate1_r1_c1'" in said
    assert "'plate9_r4_c4'" in said


def test_two_halves_that_differ_only_by_the_doubled_prefix_say_so():
    """The same wells under two spellings is a different remedy from two
    different experiments, and 154 D is where that difference comes from."""
    from spacr.qt.widgets.measurement_scan_panel import describe_key_overlap

    left = pd.DataFrame({"prc": ["pplate1_r1_c1"]})
    right = pd.DataFrame({"prc": ["plate1_r1_c1"]})

    said = describe_key_overlap("merged measurements", left, "loaded run",
                                right)

    assert "different plate ids" in said
    assert "correct_metadata" in said


def test_wells_that_do_meet_are_not_reported_as_a_join_problem():
    """Saying anything here would send the user the wrong way."""
    from spacr.qt.widgets.measurement_scan_panel import describe_key_overlap

    frame = pd.DataFrame({"prc": ["plate1_r1_c1"]})

    assert describe_key_overlap("a", frame, "b", frame) == ""


def test_a_frame_with_no_well_identity_is_named_as_such():
    """An absent key is itself the answer to "why did nothing join"."""
    from spacr.qt.widgets.measurement_scan_panel import (describe_key_overlap,
                                                         well_keys)

    assert well_keys(pd.DataFrame({"gene": ["g"]})) == ("", ())
    said = describe_key_overlap("merged measurements",
                                pd.DataFrame({"gene": ["g"]}),
                                "loaded run",
                                pd.DataFrame({"prc": ["plate1_r1_c1"]}))
    assert "carries no well identity" in said
