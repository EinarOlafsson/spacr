"""A setting that names ONE file gets one row, not eight.

Reported 2026-09-15: "the barcode references in map barcodes should be one
line or row each now they are large fields for some reason."

MEASURED on the live Map Barcodes screen before this: `grna_csv`, `row_csv`
and `column_csv` each had a sizeHint height of 240, against 30-33 for every
other field on that form. `single=True` already existed and was not enough --
it shrank the backing list from 96 to 48 and still left the list, the hint
and the button on three separate rows.

THE LIST IS STILL THE VALUE. `paths()` reads it, every mutation goes through
it, and the drop target and path probes are wired to it. Hiding it and
mirroring its one row into a read-only field changes what the user sees and
nothing about what the widget is, which is what these tests hold.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

#: A one-file setting must sit in the band every other control occupies.
#: Toggle is 22 and _ScalarEdit is 30 on the same form; 240 was the defect.
ONE_ROW_MAX = 40


def _widget(qtbot, single):
    from spacr.qt.widgets.file_list import FilePathListWidget

    widget = FilePathListWidget(value="", kind="csv", title="Choose",
                                single=single)
    qtbot.addWidget(widget)
    return widget


def test_a_single_file_setting_is_one_row_high(qtbot):
    """THE DEFECT, as a number."""
    widget = _widget(qtbot, single=True)
    height = widget.sizeHint().height()
    assert height <= ONE_ROW_MAX, (
        f"a setting that names one file is {height}px tall; every other "
        f"control on the same form is 22-33, and 240 was what was reported")


def test_a_multi_file_setting_is_still_a_list(qtbot):
    """THE HALF THAT MUST NOT REGRESS.

    Settings that take MANY files still need the list, the order and the
    reorder buttons. A fix that flattened both would take the ordering away
    from the settings that have an order.
    """
    widget = _widget(qtbot, single=False)
    assert widget.sizeHint().height() > ONE_ROW_MAX, (
        "a multi-file setting was flattened to one row, so its order and "
        "its selection have nowhere to live")


def test_the_value_still_round_trips(qtbot, tmp_path):
    """One row on screen, the same value underneath."""
    chosen = tmp_path / "barcodes_row.csv"
    chosen.write_text("a,b\n", encoding="utf-8")
    widget = _widget(qtbot, single=True)

    widget.set_value(str(chosen))
    assert widget.get_value() == str(chosen)
    assert widget.paths() == [str(chosen)]


def test_the_visible_field_shows_the_chosen_path(qtbot, tmp_path):
    """PROOF THE ROW IS NOT EMPTY FURNITURE.

    Shrinking the control while leaving it blank would pass the height test
    and tell the user nothing. The field has to carry the path the setting
    holds.
    """
    chosen = tmp_path / "barcodes_column.csv"
    chosen.write_text("a,b\n", encoding="utf-8")
    widget = _widget(qtbot, single=True)
    widget.set_value(str(chosen))

    assert widget._single_line is not None
    assert widget._single_line.text() == str(chosen)
    assert widget._single_line.isReadOnly(), (
        "the path field is editable, so a typed value would disagree with "
        "the list that is actually the setting")


def test_clearing_the_value_clears_the_row(qtbot, tmp_path):
    """The mirror follows removals, not only additions."""
    chosen = tmp_path / "x.csv"
    chosen.write_text("a\n", encoding="utf-8")
    widget = _widget(qtbot, single=True)
    widget.set_value(str(chosen))
    assert widget._single_line.text()

    widget.set_value("")
    assert widget._single_line.text() == "", (
        "the field kept a path the setting no longer holds")
