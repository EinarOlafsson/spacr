"""Reading and writing a column field, whatever kind of field it is.

A settings key that names columns is edited through three different widgets
-- a line edit, a combo box and a chip strip -- and the picker has to read
and write all of them. Losing a name here is silent: the field still looks
filled, and the run measures one column where the user asked for four.
"""
from __future__ import annotations

from typing import Any, List

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QComboBox, QLineEdit, QWidget

from spacr.qt.widgets.column_picker import (ColumnPickerDialog, field_values,
                                            set_field_values)


class _ChipStrip(QWidget):
    """A minimal stand-in for the settings screen's chip-strip list editor.

    The picker duck-types on ``get_value``/``set_value`` rather than
    importing the screen, so this is exactly what it looks for.
    """

    def __init__(self, value: Any = None, parent=None):
        super().__init__(parent)
        self._value = value

    def get_value(self):
        return self._value

    def set_value(self, value):
        self._value = value


def test_a_chip_strip_holding_one_name_reads_as_one_name(qtbot):
    """A scalar in a list editor comes back as a single-item list.

    The editor's stored value is whatever was last written to it, and a
    single-column setting stores a bare string. Returning it unwrapped would
    make the caller iterate its characters and offer a column per letter.
    """
    strip = _ChipStrip("cell_area")
    qtbot.addWidget(strip)

    assert field_values(strip) == ["cell_area"]
    assert field_values(_ChipStrip("   ")) == []
    assert field_values(_ChipStrip(None)) == []
    assert field_values(_ChipStrip(["a", " b ", ""])) == ["a", "b"]


def test_a_text_field_is_read_in_whichever_list_style_it_holds(qtbot):
    """Bare commas and a bracketed list both read as the same names.

    A settings CSV round-trips a list as ``['a', 'b']`` while a user types
    ``a, b``. Reading only one style drops every name but the first, and the
    field still looks correct on screen.
    """
    field = QLineEdit()
    qtbot.addWidget(field)

    assert field_values(field) == []

    field.setText("cell_area, nucleus_area")
    assert field_values(field) == ["cell_area", "nucleus_area"]

    field.setText("['cell_area', \"nucleus_area\"]")
    assert field_values(field) == ["cell_area", "nucleus_area"]


def test_writing_several_names_into_a_single_valued_field_keeps_them_all(
        qtbot):
    """A line edit given several names ends up holding all of them.

    Replacing a single-valued field with the first name and dropping the rest
    is the silent loss this exists to prevent: the user picked four columns
    and the run measures one.
    """
    field = QLineEdit()
    qtbot.addWidget(field)

    assert set_field_values(field, ["cell_area", "nucleus_area", "solidity"])

    written = field_values(field)
    assert written == ["cell_area", "nucleus_area", "solidity"]


def test_writing_nothing_writes_nothing_and_says_so(qtbot):
    """No field, or no names, returns False without touching anything.

    False is what tells a caller to fall back to asking. A silent True over
    an unchanged field would report a fill that never happened.
    """
    field = QLineEdit("cell_area")
    qtbot.addWidget(field)

    assert set_field_values(None, ["cell_area"]) is False
    assert set_field_values(field, []) is False
    assert set_field_values(field, ["  ", ""]) is False
    assert field.text() == "cell_area"


def test_the_picker_knows_whether_it_returns_one_column_or_several(qtbot):
    """``is_multi`` reports the mode the dialog was opened in.

    The caller decides between writing a string and writing a list from this
    answer alone; getting it wrong puts ``['cell_area']`` into a field that
    is read as a single name.
    """
    single = ColumnPickerDialog(db_path="", multi=False)
    qtbot.addWidget(single)
    multi = ColumnPickerDialog(db_path="", multi=True)
    qtbot.addWidget(multi)

    assert single.is_multi() is False
    assert multi.is_multi() is True


def test_selection_is_empty_before_the_column_tree_exists(qtbot):
    """Asked for a selection with no tree yet, the dialog says none.

    The selection is read from signal handlers that can fire while the
    dialog is still being assembled. An AttributeError there aborts the
    construction of a dialog the user just opened.
    """
    dialog = ColumnPickerDialog(db_path="", multi=True)
    qtbot.addWidget(dialog)
    del dialog._column_tree

    assert dialog._selected_column_names() == []
