"""Edge cases in the six table-shaped widgets, pinned one arc at a time.

Each of these is a branch a real user reaches and nothing in the suite had
yet driven:

* :mod:`spacr.qt.widgets.gate_console` -- a column whose name cannot be a
  Python name, and a column literally called ``df``, must not be injected
  into the expression scope (the second would shadow the frame itself); and
  sending an empty chat box must not clear it or say anything.
* :mod:`spacr.qt.widgets.metadata_table` -- recomputing the Filename cell of
  a row that has no Filename cell must leave the other rows correct instead
  of raising; and Apply writes the CSV whether or not a callback was given.
* :mod:`spacr.qt.widgets.database_set` -- blanks and duplicates are dropped
  when the set is assigned, and the chip row is rebuilt correctly even when
  it holds a spacer rather than only chips.
* :mod:`spacr.qt.widgets.plate_layout` -- a design that fills the plate is
  not told it has spare wells, and ``write_design`` writes the table it was
  handed rather than recomputing one.
* :mod:`spacr.qt.widgets.folding_summary` -- blank lines never become rows.
* :mod:`spacr.qt.widgets.row_exclusion` -- clicking one rule's x twice
  removes one rule, and a value read for one column is not poured into a row
  showing another.

Two arcs in ``folding_summary`` are unreachable; the proofs are at the foot
of this file rather than a test contorted to reach them.
"""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QTableWidget, QToolButton     # noqa: E402

from spacr.qt.ingest_preview import ROW_COLUMNS             # noqa: E402
from spacr.qt.widgets.database_set import DatabaseSetWidget  # noqa: E402
from spacr.qt.widgets.folding_summary import split_rows     # noqa: E402
from spacr.qt.widgets.gate_console import GateConsole       # noqa: E402
from spacr.qt.widgets.metadata_table import (               # noqa: E402
    MetadataTableDialog, MetadataTablePanel,
)
from spacr.qt.widgets.plate_layout import (                 # noqa: E402
    EDGE_USE, ROLE_NEGATIVE, ROLE_POSITIVE, ROLE_TREATMENT,
    Condition, PlateDesign, assign_wells, check_design, write_design,
)
from spacr.qt.widgets.row_exclusion import (                # noqa: E402
    RowExclusionEditor, _ExclusionRuleRow,
)
from spacr.qt.widgets.table_chip import TableChip           # noqa: E402

pytestmark = pytest.mark.qt


def _col(key):
    return ROW_COLUMNS.index(key)


def _row(**over):
    row = {"original": "img_001.czi", "plate": "plate1", "well": "plate1_A01",
           "field": 1, "channel": 1, "time": 1, "canonical": ""}
    row.update(over)
    return row


# ---------------------------------------------------------------------------
# gate_console
# ---------------------------------------------------------------------------

def test_a_column_that_cannot_be_a_name_is_reachable_only_through_df():
    """Columns are injected as bare names -- but only when that is safe.

    ``mean area`` is not an identifier and ``df`` is already the frame, so
    neither may be bound: binding the second would replace the table with one
    of its own columns and every later expression would quietly disagree with
    the plot. The identifier column next to them proves the injection that is
    being withheld really happens.
    """
    from spacr.qt.widgets.gate_console import evaluate

    frame = pd.DataFrame({"area": [1.0, 3.0],
                          "mean area": [10.0, 20.0],
                          "df": ["x", "y"]})

    # The safe column IS injected...
    assert evaluate("area.mean()", frame) == "2.0"
    # ...while `df` still names the frame, not the column called df.
    assert evaluate("df", frame) == "2 rows × 3 columns"
    # ...and the non-identifier column has no bare name at all.
    assert evaluate("mean area", frame).startswith("SyntaxError")
    # It is still reachable the long way round, so nothing was dropped.
    assert evaluate("df['mean area'].sum()", frame) == "30.0"


def test_sending_an_empty_chat_box_says_nothing_and_clears_nothing(qtbot):
    """Whitespace is not a question, so it must not be sent or wiped.

    ``send_chat`` clears the box only when ``ask`` accepted something; a box
    the user has not typed in yet keeps whatever is in it.
    """
    console = GateConsole()
    qtbot.addWidget(console)
    console.set_responder(lambda text: f"answered {text}")

    # The presence case: a real question is asked and the box is emptied.
    console.chat.setPlainText("how many objects?")
    console.send_chat()
    assert console.chat.toPlainText() == ""
    assert "answered how many objects?" in console.transcript()

    before = console.transcript()
    console.chat.setPlainText("   \n  ")
    console.send_chat()

    assert console.chat.toPlainText() == "   \n  "
    assert console.transcript() == before


# ---------------------------------------------------------------------------
# metadata_table
# ---------------------------------------------------------------------------

def test_a_row_with_no_filename_cell_does_not_stop_the_other_rows(qtbot):
    """The Filename cell is rebuilt live; a row missing one is skipped.

    The grid is a plain ``QTableWidget`` whose items a host can take out, and
    the recompute runs on every edit. Losing one cell must cost that one cell
    and nothing else.
    """
    panel = MetadataTablePanel([_row(original="a.czi"),
                                _row(original="b.czi")])
    qtbot.addWidget(panel)
    # The QTableWidget itself, via Qt's own child lookup: the arc only exists
    # for a table whose canonical item is gone, and the panel never removes
    # one itself.
    table = panel.findChild(QTableWidget)
    table.takeItem(0, _col("canonical"))

    table.item(0, _col("plate")).setText("plateX")
    table.item(1, _col("plate")).setText("plateY")

    read_back = panel.rows()
    assert read_back[0]["plate"] == "plateX"
    assert read_back[0]["canonical"] == ""          # the cell that was taken
    assert read_back[1]["plate"] == "plateY"
    assert read_back[1]["canonical"].startswith("plate1_A01")


def test_apply_writes_the_csv_with_or_without_a_callback(qtbot, tmp_path):
    """``on_apply`` is optional; the write is not."""
    plain = MetadataTableDialog([_row()], tmp_path / "plain.csv")
    qtbot.addWidget(plain)
    plain._apply_btn.click()

    assert plain.written_path is not None
    assert plain.written_path.is_file()

    # The presence case, so the absence above is about the callback and not
    # about the write having failed.
    seen = []
    watched = MetadataTableDialog([_row()], tmp_path / "watched.csv",
                                  on_apply=seen.append)
    qtbot.addWidget(watched)
    watched._apply_btn.click()

    assert seen == [watched.written_path]
    assert watched.written_path.read_text() == plain.written_path.read_text()


# ---------------------------------------------------------------------------
# database_set
# ---------------------------------------------------------------------------

def test_blanks_and_repeats_are_dropped_from_an_assigned_set(qtbot):
    """A settings CSV can hold the same folder twice, or an empty field."""
    widget = DatabaseSetWidget(mode="folder")
    qtbot.addWidget(widget)

    widget.set_value(["/data/plate1", "", "   ", "/data/plate1",
                      "/data/plate2"])

    assert widget.sources() == ["/data/plate1", "/data/plate2"]
    assert len(widget.findChildren(TableChip)) == 2


def test_the_chip_row_is_rebuilt_even_when_it_holds_a_spacer(qtbot):
    """Rebuilding takes layout ITEMS, only some of which own a widget.

    The chip row already ends in a stretch, and a host that lays anything
    else out beside the chips adds another item with no widget. Reaching for
    the private layout because that is the only place a non-widget item can
    come from -- the widget itself only ever inserts chips.
    """
    widget = DatabaseSetWidget(["/data/plate1", "/data/plate2"], mode="folder")
    qtbot.addWidget(widget)
    assert len(widget.findChildren(TableChip)) == 2
    widget._chips.insertStretch(0, 1)

    widget.set_value(["/data/plate3", "/data/plate4"])

    chips = [chip for chip in widget.findChildren(TableChip)
             if chip.parent() is widget]
    assert [chip.name for chip in chips] == ["plate3", "plate4"]
    assert widget.sources() == ["/data/plate3", "/data/plate4"]


# ---------------------------------------------------------------------------
# plate_layout
# ---------------------------------------------------------------------------

def _full_plate_conditions():
    return (Condition("drug_a", 4, ROLE_TREATMENT),
            Condition("dmso", 1, ROLE_NEGATIVE),
            Condition("puro", 1, ROLE_POSITIVE))


def test_a_plate_with_nothing_left_over_is_not_told_to_spend_it():
    """The spare-well note is advice about wells that exist to be spent."""
    full = PlateDesign(plate_id="plate1", plate_format=6,
                       conditions=_full_plate_conditions(),
                       edge_policy=EDGE_USE, seed=1)
    assert full.wells_available == full.wells_requested

    keys = {finding.key for finding in check_design(full)}
    assert "spare_wells" not in keys

    # One replicate fewer and the same plate does get the note, so the
    # absence above is the emptiness of the remainder and not a dead check.
    roomy = PlateDesign(plate_id="plate1", plate_format=6,
                        conditions=(Condition("drug_a", 3, ROLE_TREATMENT),
                                    Condition("dmso", 1, ROLE_NEGATIVE),
                                    Condition("puro", 1, ROLE_POSITIVE)),
                        edge_policy=EDGE_USE, seed=1)
    spare = [f for f in check_design(roomy) if f.key == "spare_wells"]
    assert len(spare) == 1
    assert "1 usable well" in spare[0].message


def test_write_design_writes_the_assignment_it_was_handed(tmp_path):
    """The table is a parameter so the plate map matches what was SHOWN.

    A screen has already assigned the wells by the time it saves; recomputing
    here would write a second random layout under the same plate id.
    """
    design = PlateDesign(plate_id="plate1", plate_format=6,
                         conditions=_full_plate_conditions(), seed=7)
    table = assign_wells(design)
    table = table.assign(shown_to_the_user=True)

    paths = write_design(design, tmp_path / "out", table=table)

    written = pd.read_csv(paths["plate_map"])
    assert "shown_to_the_user" in written.columns
    assert list(written["well"]) == list(table["well"])
    assert paths["plate_map"].is_file()


# ---------------------------------------------------------------------------
# folding_summary
# ---------------------------------------------------------------------------

def test_a_blank_line_between_rows_never_becomes_a_row():
    """Blank lines are dropped BEFORE the rows are parsed.

    A summary is written with blank lines between its blocks, and an empty
    ``("", "")`` row would render as an empty table line in the panel.
    """
    rows = split_rows("  effect          0.42\n\n   \n  p value         0.01\n")

    assert rows == [("effect", "0.42"), ("p value", "0.01")]


# ---------------------------------------------------------------------------
# row_exclusion
# ---------------------------------------------------------------------------

def _measurements_source(tmp_path):
    """A project folder holding one measurements.db, as a screen gets one."""
    folder = tmp_path / "measurements"
    folder.mkdir()
    frame = pd.DataFrame({
        "plateID": ["p1", "p1", "p2"],
        "columnID": ["c1", "c2", "c1"],
        "object_label": [1, 2, 3],
    })
    with sqlite3.connect(str(folder / "measurements.db")) as connection:
        frame.to_sql("cell", connection, index=False)
    return str(tmp_path)


def test_clicking_one_rules_close_mark_twice_removes_one_rule(qtbot):
    """The second click arrives at a row that is already gone.

    Nothing stops a user double-clicking the x, and the row's button lives on
    until Qt gets round to deleting it, so the signal fires twice. The second
    one must not take the neighbouring rule with it, nor leave the editor
    adding a blank replacement.
    """
    editor = RowExclusionEditor({"plateID": ["p1"], "columnID": ["c1"]},
                                threaded=False)
    qtbot.addWidget(editor)
    rows = editor.findChildren(_ExclusionRuleRow)
    assert [row.column.currentText() for row in rows] == ["plateID",
                                                          "columnID"]
    close_mark = rows[0].findChildren(QToolButton)[0]

    close_mark.click()
    remaining = editor.findChildren(_ExclusionRuleRow)
    assert [row.column.currentText() for row in remaining] == ["columnID"]

    close_mark.click()

    remaining = editor.findChildren(_ExclusionRuleRow)
    assert [row.column.currentText() for row in remaining] == ["columnID"]


def test_a_value_read_only_fills_the_rows_showing_that_column(qtbot, tmp_path):
    """One read is delivered while another row is showing another column.

    The reads are per column and arrive one at a time, so every delivery
    meets rows the payload says nothing about. Those rows keep waiting for
    their own read instead of being given somebody else's values.
    """
    source = _measurements_source(tmp_path)
    editor = RowExclusionEditor({"plateID": ["p1"], "columnID": ["c1"]},
                                threaded=False)
    qtbot.addWidget(editor)

    editor.set_source(source)

    rows = editor.findChildren(_ExclusionRuleRow)
    options = {row.column.currentText():
               [row.values.itemText(i) for i in range(row.values.count())]
               for row in rows}
    assert sorted(options["plateID"]) == ["p1", "p2"]
    assert sorted(options["columnID"]) == ["c1", "c2"]
    assert editor.get_value() == {"plateID": ["p1"], "columnID": ["c1"]}


# ---------------------------------------------------------------------------
# Two arcs in folding_summary that no input can take. Proofs, not pragmas.
#
# 1. ``split_rows`` line 107, ``elif line.strip():`` -> next iteration.
#    ``lines`` is built on line 83 as
#        [line for line in str(body or "").splitlines() if line.strip()]
#    so every element of the very list line 100 iterates has already passed
#    the identical ``line.strip()`` test. The ``elif`` can be reached (a line
#    shorter than the label column, with no row yet above it, falls through
#    both earlier tests -- the test above drives exactly that) but it cannot
#    be false. Its false arc is dead for the same reason the filter exists.
#
# 2. ``FoldingSummaryView._clear`` line 311, ``if keep is not None:`` ->
#    return. ``keep`` is ``getattr(self, "_actions", None)`` and
#    ``__init__`` assigns ``self._actions = QWidget(self._body)``
#    unconditionally at line 182, before ``self._text``/``self._sections``
#    exist at all. ``_clear`` has exactly one caller, ``_rebuild`` at line
#    436, and ``_rebuild`` has no caller inside ``__init__`` -- it is reached
#    only from ``setPlainText``/``clear``/``appendPlainText``, i.e. after the
#    constructor returned. ``FoldingSummaryView`` is instantiated in exactly
#    one place in spacr (``regression_results.py`` line 921) and is not
#    subclassed anywhere, so no caller can arrive with ``_actions`` unset.
# ---------------------------------------------------------------------------
