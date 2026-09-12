"""Any file into any slot: pairing and naming without a filename convention.

INSTRUCTION 392, verbatim: "in the regression model as it is now the user
must name their count and score csvs starting with the plate name. this is
impractical and the user should be able to drag and drop any file into any
slot regardlett of nameing. if they do share a name it can be used if the
files do not the name should be generated, first row plate 1 second row
plate 2 and so on".

THE ENGINE ALREADY DOES THIS. `load_regression_input_pairs` says so in its
own first line -- "resolve plate identity WITHOUT FILENAME GUESSES ... own
column, partner column, then pair-row order". It is the GUI's proposal step
that insisted, one layer above, and never got the memo.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from spacr.qt.widgets.file_list import suggest_file_pairs   # noqa: E402


def test_two_ordinary_names_that_share_nothing_make_one_row():
    """THE COMPLAINT ITSELF. `scores.csv` and `counts.csv` share no token, so
    the token pass matched neither -- and the user got TWO half-empty rows
    with nothing to say the files belonged together, rather than one row with
    a blank plate.
    """
    rows = suggest_file_pairs(["scores.csv"], ["counts.csv"])

    assert len(rows) == 1, (
        f"two files that belong together produced {len(rows)} rows: {rows}")
    assert rows[0]["score"] == "scores.csv"
    assert rows[0]["count"] == "counts.csv"


def test_a_row_paired_by_position_is_named_by_its_position():
    """`plate 1`, `plate 2`, in row order, which is what was asked for."""
    rows = suggest_file_pairs(["a_scores.csv", "b_scores.csv"],
                              ["x_counts.csv", "y_counts.csv"])

    assert [row["plate"] for row in rows] == ["plate 1", "plate 2"]
    assert [row["score"] for row in rows] == ["a_scores.csv", "b_scores.csv"]
    assert [row["count"] for row in rows] == ["x_counts.csv", "y_counts.csv"]


def test_a_shared_token_still_decides_and_beats_arrival_order():
    """THE HALF THAT MUST NOT REGRESS. A user who DID name their files
    carefully must not have that overridden by drop order, so the token pass
    keeps priority and position only takes what it leaves.

    The counts here are supplied in the OPPOSITE order to the scores, so
    position alone would cross them.
    """
    rows = suggest_file_pairs(["exp42_scores.csv", "exp7_scores.csv"],
                              ["exp7_counts.csv", "exp42_counts.csv"])

    assert rows[0]["count"] == "exp42_counts.csv", (
        "position overrode a filename token the user chose deliberately")
    assert rows[1]["count"] == "exp7_counts.csv"


def test_a_shared_token_that_is_not_a_plate_token_still_names_the_row():
    """THE NARROWER HALF, AND THE EASIER ONE TO MISS, because the pairing
    looks like it worked. `_plate_label` took the longest token beginning
    with the five letters ``plate``, so `exp42_scores.csv` and
    `exp42_counts.csv` paired correctly and were labelled with nothing.
    """
    rows = suggest_file_pairs(["exp42_scores.csv"], ["exp42_counts.csv"])

    assert rows[0]["plate"], (
        "a correctly paired row was left with a blank plate cell because its "
        "shared token does not begin with 'plate'")


def test_a_real_plate_name_is_parsed_rather_than_generated():
    """A generated label must never displace one the filenames supply."""
    rows = suggest_file_pairs(["plate3_scores.csv"], ["plate3_counts.csv"])

    assert rows[0]["plate"] == "plate3"


def test_a_row_that_is_not_a_pair_keeps_a_blank_plate():
    """A LABEL THAT READS AS A FACT IS WORSE THAN A BLANK CELL.

    "the files" in the request is the PAIR. A score still waiting for its
    partner has not been named by anything, and numbering it would assert
    exactly what step 3 of 392 warns against -- a default that reads as
    parsed. The blank cell is the honest state, and it is the one the user is
    being asked to resolve.
    """
    rows = suggest_file_pairs(["one_scores.csv"],
                              ["a_counts.csv", "b_counts.csv"])

    leftover = [row for row in rows if not row["score"]]
    assert leftover, "expected the unpaired count to get a row of its own"
    assert not leftover[0]["plate"], (
        "an unpaired count claimed a plate number it has no basis for")


def _write(tmp_path, name, header="grna,count\nA,1\n"):
    """One CSV on disk, because the widget reads headers to pick a side."""
    path = tmp_path / name
    path.write_text(header, encoding="utf-8")
    return str(path)


def test_a_plate_name_the_user_typed_survives_a_later_drop(qtbot, tmp_path):
    """A GENERATED LABEL IS A DEFAULT; A TYPED ONE IS A DECISION.

    `_repropose` rebuilds every row from the filenames whenever a file is
    added, which is right for a generated number -- a row that becomes the
    first row must be renumbered rather than left claiming `plate 2`. It is
    wrong for a name the user chose, and without this the user's own word for
    their plate is discarded the moment they drag in the next file.

    Told apart by re-proposing and comparing rather than by a flag: whatever
    the proposal would say for the current files is by definition not a
    decision.
    """
    from spacr.qt.widgets.file_list import PairedFileTableWidget

    widget = PairedFileTableWidget()
    qtbot.addWidget(widget)

    scores = _write(tmp_path, "scores.csv", "grna,score\nA,0.5\n")
    counts = _write(tmp_path, "counts.csv")
    widget.add_paths_for_side([scores], "score")
    widget.add_paths_for_side([counts], "count")

    rows = widget.get_value()
    assert len(rows) == 1 and rows[0]["plate"] == "plate 1", rows

    # The user names it.
    widget.table.item(0, 0).setText("Treated, day 3")
    assert widget.get_value()[0]["plate"] == "Treated, day 3"

    # And then drops in another plate's files, which re-proposes everything.
    widget.add_paths_for_side([_write(tmp_path, "more_scores.csv",
                                      "grna,score\nB,0.2\n")], "score")
    widget.add_paths_for_side([_write(tmp_path, "more_counts.csv")], "count")

    named = [row["plate"] for row in widget.get_value()]
    assert "Treated, day 3" in named, (
        "the name the user typed was discarded when the next file arrived: "
        f"{named}")


def test_a_generated_number_is_renumbered_rather_than_kept(qtbot, tmp_path):
    """THE OTHER HALF, AND THE REASON THE FIRST CANNOT JUST PIN EVERYTHING.

    A row labelled `plate 2` that ends up first after a deletion is worse
    than a blank cell, because it reads as a fact about a plate. So a
    generated number must NOT be preserved across a re-propose -- only a
    typed one.
    """
    from spacr.qt.widgets.file_list import PairedFileTableWidget

    widget = PairedFileTableWidget()
    qtbot.addWidget(widget)
    widget.add_paths_for_side([_write(tmp_path, "a_scores.csv",
                                      "grna,score\nA,0.5\n")], "score")
    widget.add_paths_for_side([_write(tmp_path, "x_counts.csv")], "count")
    widget.add_paths_for_side([_write(tmp_path, "b_scores.csv",
                                      "grna,score\nB,0.5\n")], "score")
    widget.add_paths_for_side([_write(tmp_path, "y_counts.csv")], "count")

    assert [row["plate"] for row in widget.get_value()] == \
        ["plate 1", "plate 2"], widget.get_value()
