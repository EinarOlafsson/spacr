"""The measurements tab's refusals, and the sentences it says instead.

Steps 1-4 of this tab -- attach, choose, merge, regress -- are a workflow where
every step can be reached before the one before it is done. What is driven here
is what each control does when it is pressed too early, and what the panel says
when it cannot answer: a button that silently does nothing is the defect this
file's own comments keep returning to, and a merge that quietly guessed how to
combine a column produces a number that is wrong and looks fine.

Offscreen, CPU-only, offline.
"""
from __future__ import annotations

import os
import sqlite3

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from spacr.qt.widgets import measurement_scan_panel as msp   # noqa: E402

pytestmark = pytest.mark.qt


# --------------------------------------------------------------------------- #
#  Real sqlite databases, in spaCR's own shape
# --------------------------------------------------------------------------- #

def _database(directory, plate, *, extra=None, extra_type="REAL",
              extra_table="pathogen", stored_plate=None):
    """One plate's ``measurements.db`` with a cell and a pathogen table.

    ``extra`` adds one more column to ``extra_table``; ``extra_type`` is the
    SQLite declaration it is given, which is what the panel reads to say how a
    column will be combined. An empty declaration is a column SQLite has no
    type for. It goes on ``pathogen`` by default, because the panel only
    describes aggregation for the tables that are many rows per cell.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(str(directory), "measurements.db")
    identity = {"rowID": "r1", "columnID": "c1", "fieldID": "f1"}
    written = str(stored_plate or plate)

    cell = pd.DataFrame({
        "plateID": [written] * 3, **{k: [v] * 3 for k, v in identity.items()},
        "object_label": [1, 2, 3],
        "area": [100.0, 200.0, 300.0],
        "perimeter": [10.0, 20.0, 30.0],
    })
    pathogen = pd.DataFrame({
        "plateID": [written] * 4, **{k: [v] * 4 for k, v in identity.items()},
        "cell_id": [1, 1, 2, 2],
        "object_label": [1, 2, 1, 2],
        "pathogen_area": [10.0, 30.0, 50.0, 70.0],
    })
    with sqlite3.connect(path) as database:
        cell.to_sql("cell", database, index=False)
        pathogen.to_sql("pathogen", database, index=False)
        if extra:
            database.execute(
                f'ALTER TABLE "{extra_table}" ADD COLUMN "{extra}" '
                f'{extra_type}')
            database.execute(f'UPDATE "{extra_table}" SET "{extra}" = 1')
    return path


def _rows(paths, plates=None):
    """Input-table rows, in the shape the paired input table emits."""
    plates = plates or [f"plate{i + 1}" for i in range(len(paths))]
    return [{"plate": plates[i], "score": f"{plates[i]}_scores.csv",
             "count": f"{plates[i]}_counts.csv", "database": path}
            for i, path in enumerate(paths)]


@pytest.fixture()
def two_plates(tmp_path):
    return [_database(tmp_path / "plate1", "plate1"),
            _database(tmp_path / "plate2", "plate2")]


@pytest.fixture()
def merge_panel(qtbot, two_plates):
    widget = msp.DatabaseMergePanel(lambda: _rows(two_plates), threaded=False)
    qtbot.addWidget(widget)
    return widget


# --------------------------------------------------------------------------- #
#  The merge, called directly
# --------------------------------------------------------------------------- #

def test_an_unknown_ambiguous_identifier_policy_is_refused_by_name(two_plates):
    """The policy decides whether provenance is invented or left out.

    Silently falling back to one of the two would answer a question the caller
    did not ask, and the two answers differ in whether a text identifier that
    disagrees within a cell is carried at all. The message lists what is
    allowed.
    """
    with pytest.raises(msp.MergeError) as excinfo:
        msp.merge_across_databases(two_plates, ["cell"],
                                   on_ambiguous_identifier="whatever")

    message = str(excinfo.value)
    assert "on_ambiguous_identifier" in message
    assert "'whatever'" in message
    assert all(name in message for name in msp.AMBIGUOUS_IDENTIFIER_POLICIES)


def test_a_cancelled_merge_writes_nothing_and_says_the_old_one_is_intact(
        two_plates):
    """Stopping mid-merge must leave the previous result exactly where it was.

    The Stop button is offered while a four-database merge runs. What the user
    needs to know at that moment is that nothing half-written has replaced
    what they had, so the exception says it.
    """
    with pytest.raises(msp.MergeCancelled) as excinfo:
        msp.merge_across_databases(two_plates, ["cell", "pathogen"],
                                   cancelled=lambda: True)

    message = str(excinfo.value)
    assert "Nothing was written" in message
    assert "untouched" in message


def test_a_doubled_plate_prefix_is_repaired_on_read_so_nothing_is_said(
        tmp_path):
    """A database stamped ``pplate1`` reads as ``plate1``, with no caveat.

    THIS TEST WAS PINNED AS xfail(strict=True) ASSERTING THE OPPOSITE -- that
    ``plate_id_notes`` names both spellings -- on the grounds that it computes
    the odd ids from the already-normalised ``plates`` and is therefore silent
    on every database. The silence is real; the conclusion was wrong, and the
    assertion is turned round here.

    The doubling is collapsed as the database is READ, so the plan and the
    merged frame both say ``plate1`` and the measurement side meets a score
    CSV that ``correct_metadata`` has normalised. There is no mismatch left to
    warn about, and a note naming ``pplate1`` would be the panel crying wolf
    -- which is what
    ``test_the_measurements_tab_is_a_workflow.py::test_the_doubled_prefix_cannot_reach_a_join_key``
    asserts from the panel's side.

    What ``plate_id_notes`` is left as -- a tripwire for an id that reaches
    the plan unrepaired -- is pinned by the test below, so its silence here
    cannot be confused with dead code.
    """
    from spacr.multi_database import describe_merge

    path = _database(tmp_path / "odd", "plate1", stored_plate="pplate1")
    plan = describe_merge([path], "cell")

    assert plan.sources[0].plates == ("plate1",), \
        "the read repair collapses the doubling before the plan sees it"
    assert plan.sources[0].stored_plates == ("pplate1",), \
        "the stored spelling is still carried, it is simply not a warning"
    assert msp.plate_id_notes(plan) == []


def test_a_plate_id_that_reaches_the_plan_uncollapsed_is_named(tmp_path):
    """The tripwire speaks when the read repair has NOT been applied.

    ``plate_id_notes`` reads the ids the merge will key on. One that is still
    not canonical at that point is one nothing repaired, and it will meet no
    normalised score file -- silently, several steps later -- so the note has
    to name it and say where the doubling is.
    """
    import types

    source = types.SimpleNamespace(label="measurements",
                                   plates=("pplate1",),
                                   stored_plates=("pplate1",))
    plan = types.SimpleNamespace(sources=(source,))

    notes = msp.plate_id_notes(plan)

    assert len(notes) == 1
    # BOTH IDENTIFIERS AND WHAT THE DIFFERENCE COSTS, rather than one exact
    # phrase. This pinned "shown as plate1" and went on failing after the
    # note was reworded to say the same thing more fully -- the wording is
    # not the behaviour, and the behaviour is that a reader is told the
    # stored id, the canonical one, and that score CSVs will not join.
    assert "pplate1" in notes[0]
    assert "plate1" in notes[0]
    assert msp.PLATE_KEY in notes[0]
    assert "will not match" in notes[0]


def test_the_whole_report_is_the_summary_and_then_its_evidence(two_plates):
    """``merge_report`` is the one-string form for a log, a test or a script.

    The panel shows the two halves in two places; anything that reads them
    together needs them joined in that order, with no blank line when there is
    no evidence to show.
    """
    frame = msp.merge_across_databases(two_plates, ["cell", "pathogen"])

    report = msp.merge_report(frame)
    summary = msp.merge_summary(frame)
    evidence = msp.merge_evidence(frame)

    assert report.startswith(summary)
    if evidence:
        assert report == summary + "\n" + evidence
    else:
        assert report == summary


# --------------------------------------------------------------------------- #
#  One fit's settings
# --------------------------------------------------------------------------- #

def test_a_paired_row_given_as_a_sequence_still_becomes_a_score_and_a_count():
    """The input table's pairs arrive as sequences as well as mappings.

    Reading only the mapping form would drop every pair from a settings block
    that used the older shape, and the queue would fit against no count data
    at all -- which is a different regression with the same name.
    """
    base = {"paired_data": [("old_scores.csv", "counts.csv"), ("alone.csv",)]}

    settings = msp.column_run_settings(base, "pathogen_area", "/merged.csv")

    assert settings["paired_data"] == [
        {"score": "/merged.csv", "count": "counts.csv"},
        {"score": "/merged.csv", "count": ""},
    ]
    assert settings["score_data"] == ["/merged.csv"]
    assert settings["dependent_variable"] == "pathogen_area"
    assert base["paired_data"][0] == ("old_scores.csv", "counts.csv"), \
        "the caller's own settings are not mutated"


# --------------------------------------------------------------------------- #
#  Well keys and the overlap diagnosis
# --------------------------------------------------------------------------- #

def test_a_frame_with_no_columns_carries_no_well_identity():
    """"Which wells is this?" has to be answerable for an empty frame.

    The diagnosis runs on whatever the two sides are, including a frame a
    provider returned empty, and it is the thing that explains why nothing
    joined -- so it cannot be the thing that raises.
    """
    assert msp.well_keys(None) == ("", ())
    assert msp.well_keys(pd.DataFrame()) == ("", ())


def test_the_side_with_no_well_identity_is_the_one_that_is_named():
    """The sentence has to point at the side that is missing the key.

    Naming the wrong half sends the user to check a file that is fine. Both
    halves are checked, in order, so a run frame with no plate columns is
    named as clearly as a measurement frame with none.
    """
    with_wells = pd.DataFrame({"plateID": ["p1"], "rowID": ["r1"],
                               "columnID": ["c1"], "value": [1.0]})
    without = pd.DataFrame({"value": [1.0]})

    said = msp.describe_key_overlap("merged measurements", with_wells,
                                    "loaded run", without)

    assert "loaded run" in said
    assert "no well identity" in said
    assert "merged measurements" not in said


def test_the_default_fit_is_the_real_regression(monkeypatch):
    """The queue's default fit must reach ``spacr.ml.perform_regression``.

    It is imported inside the call on purpose -- importing statsmodels and
    torch while the first window is being built is seconds of startup -- and
    an import that had drifted would only show up when a user pressed Run.
    """
    from spacr import ml

    seen = {}

    def _record(settings):
        seen.update(settings)
        return {"ok": True}

    monkeypatch.setattr(ml, "perform_regression", _record)

    assert msp._perform_regression({"dependent_variable": "area"}) == {"ok": True}
    assert seen == {"dependent_variable": "area"}


# --------------------------------------------------------------------------- #
#  Step 1 to 3: what the panel says about where the user is
# --------------------------------------------------------------------------- #

def test_plate_rows_with_no_database_on_disk_are_counted_as_such(qtbot):
    """Rows without a file are still rows, and the step line has to say so.

    A plate with no database is legal -- it still runs in the regression --
    but "nothing attached yet" would be wrong and "2 databases attached"
    would be a lie.
    """
    widget = msp.DatabaseMergePanel(
        lambda: _rows(["/nowhere/a.db", "/nowhere/b.db"]), threaded=False)
    qtbot.addWidget(widget)

    states = widget.step_states()

    assert widget.paths() == ()
    assert states[1] == "2 plate row(s), none with a database on disk."


def test_offering_tables_with_none_ticked_asks_for_the_anchor(merge_panel):
    """Step 2 with nothing chosen is a step that cannot proceed, and it says why.

    Leaving the previous sentence up -- the one naming the chosen tables --
    would describe a merge the panel is no longer able to run.
    """
    for row in range(merge_panel.tables_list.count()):
        merge_panel.tables_list.item(row).setCheckState(msp.Qt.Unchecked)

    states = merge_panel.step_states()

    assert merge_panel.selected_tables() == ()
    assert "none chosen" in states[2]
    assert "pick at least the anchor" in states[2]


def test_the_step_one_heading_is_only_written_when_it_is_empty(qtbot):
    """``_fill_table``'s richer sentence must not be replaced by the short one.

    Step 1's heading names the rows that are missing a file; the state line
    only counts them. Overwriting the first with the second would be a step
    backwards on screen, so the shorter text is written only into an empty
    heading.
    """
    widget = msp.DatabaseMergePanel(lambda: [], threaded=False)
    qtbot.addWidget(widget)
    widget.heading.setText("")

    widget._refresh_steps()
    assert widget.heading.text() == widget.step_states()[1]

    widget.heading.setText("2 of 4 plate rows have a database.")
    widget._refresh_steps()
    assert widget.heading.text() == "2 of 4 plate rows have a database."


def test_a_destination_provider_that_raises_leaves_nowhere_to_write(qtbot,
                                                                    two_plates):
    """The destination comes from a live settings panel, which can be mid-edit.

    A merge that produced a frame and then raised while asking where to put it
    would lose the frame. No destination means no artefact, and the merge
    still happens.
    """
    def _explode():
        raise RuntimeError("the src field is not set")

    widget = msp.DatabaseMergePanel(lambda: _rows(two_plates), threaded=False,
                                    destination_provider=_explode)
    qtbot.addWidget(widget)

    assert widget._destination() == ""
    frame = widget.merge()
    assert len(frame)


# --------------------------------------------------------------------------- #
#  What a column will be combined into
# --------------------------------------------------------------------------- #

def test_a_column_with_no_rule_and_no_declared_type_is_named_as_unknowable(
        qtbot, tmp_path):
    """A column SQLite has no type for cannot be promised any treatment.

    An absent answer that reads as a definite one is the false assurance this
    panel is most careful about, so the column is listed under a sentence
    saying what cannot be stated rather than defaulted into the numeric pile.
    """
    path = _database(tmp_path / "p1", "plate1", extra="wobbler", extra_type="")
    widget = msp.DatabaseMergePanel(lambda: _rows([path]), threaded=False)
    qtbot.addWidget(widget)

    kinds = widget._column_kinds([path], "pathogen")
    assert kinds["wobbler"] == "unknown"

    text = widget.describe()
    assert "match no rule AND carry no declared type" in text
    assert "wobbler" in text


def test_a_column_typed_differently_in_two_databases_becomes_unknown(qtbot,
                                                                     tmp_path):
    """One column, two declared types, is a column with no promise attached.

    Reading the type from a single database would give a definite answer that
    is only true of half the files being merged.
    """
    first = _database(tmp_path / "p1", "plate1", extra="mixed",
                      extra_type="REAL")
    second = _database(tmp_path / "p2", "plate2", extra="mixed",
                       extra_type="TEXT")
    widget = msp.DatabaseMergePanel(lambda: _rows([first, second]),
                                    threaded=False)
    qtbot.addWidget(widget)

    assert widget._column_kinds([first], "pathogen")["mixed"] == "numeric"
    assert widget._column_kinds([first, second],
                                "pathogen")["mixed"] == "unknown"


def test_one_unreadable_database_does_not_lose_the_types_of_the_others(qtbot,
                                                                       tmp_path):
    """A file that will not open costs its own columns, not everyone's.

    The type map is what the panel's "what will happen" text is built from,
    and losing it entirely because one of four plates is on an unmounted
    share would blank the whole description.
    """
    good = _database(tmp_path / "p1", "plate1")
    widget = msp.DatabaseMergePanel(lambda: _rows([good]), threaded=False)
    qtbot.addWidget(widget)

    kinds = widget._column_kinds([str(tmp_path / "gone.db"), good], "cell")

    assert kinds["area"] == "numeric"
    assert kinds["perimeter"] == "numeric"


# --------------------------------------------------------------------------- #
#  Running and stopping the merge
# --------------------------------------------------------------------------- #

def test_stopping_a_merge_that_is_not_running_reports_that_there_was_none(
        merge_panel):
    """The Stop button is a widget that exists whether or not a merge does.

    Returning ``False`` rather than raising is what lets a host wire it up
    unconditionally.
    """
    assert merge_panel.cancel_merge() is False


# --------------------------------------------------------------------------- #
#  Step 4: the column queue
# --------------------------------------------------------------------------- #

@pytest.fixture()
def frame():
    return pd.DataFrame({
        "plateID": ["p1", "p1", "p2"], "rowID": ["r1", "r2", "r1"],
        "columnID": ["c1", "c1", "c1"],
        "area": [100.0, 200.0, 300.0],
    })


def test_a_queue_worker_that_fails_reports_it_and_re_arms_the_button(qtbot,
                                                                     frame):
    """A crash in the queue worker leaves the panel usable.

    Leaving ``_running`` set would disable Run for the rest of the session,
    so the failure both reports itself and puts the panel back.
    """
    widget = msp.ColumnRegressionPanel(frame_provider=lambda: frame,
                                       threaded=False, fit=lambda s: {})
    qtbot.addWidget(widget)
    widget._running = True

    widget._on_job_failed("MemoryError")

    assert widget._running is False
    assert "The queue did not finish" in widget.progress.text()
    assert "MemoryError" in widget.progress.text()


def test_closing_the_panel_stops_the_queue_it_reports_to(qtbot, frame):
    """A queue must not outlive the widget it emits into.

    Every fit reports through this widget's signals; one still running after
    the widget is gone is a slot call into a deleted object.
    """
    widget = msp.ColumnRegressionPanel(frame_provider=lambda: frame,
                                       threaded=False, fit=lambda s: {})
    qtbot.addWidget(widget)
    assert not widget._stop.is_set()

    widget.close()

    assert widget._stop.is_set()


# --------------------------------------------------------------------------- #
#  The tab as a whole
# --------------------------------------------------------------------------- #

def test_the_refusal_names_what_is_there_when_both_halves_look_present(
        qtbot, two_plates):
    """A frame with rows that still cannot be scanned gets the shortest answer.

    The other refusals each name a missing half. When neither half is missing
    the sentence falls back to what the tab is holding, which is the only
    thing left to say.
    """
    widget = msp.MeasurementScanPanel(
        frame_provider=lambda: pd.DataFrame({"gene": ["a"], "m": [1.0]}),
        database_provider=lambda: _rows(two_plates), threaded=False)
    qtbot.addWidget(widget)

    said = widget.why_nothing_to_scan(pd.DataFrame({"gene": ["a"]}))

    assert said.startswith("Nothing to scan. ")
    assert "measurement database(s) are attached" in said


