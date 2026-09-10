"""Corners of the SQL column picker nothing else drives.

Four behaviours live here, each of them the *quiet* half of a pair the rest
of ``tests/qt/test_column_picker.py`` only exercises from its loud side:

* a double-click that arrives with no row under it must not close the dialog,
  while a displayed column whose stored name has whitespace remains pickable;
* :meth:`ColumnPickerDialog.select_columns` asked for names the table does
  not have -- it must come back empty rather than leave a stale highlight
  that ``chosen_columns`` would then hand to the run;
* :func:`set_field_text` writing a name a closed combo box already offers --
  it must reuse the entry instead of adding a second identical one;
* :func:`attach_column_picker` on a field that lives in spaCR's own
  wrapping ``FlowLayout``, whose ``replaceWidget`` cannot swap a widget.
"""
from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QDialog, QLineEdit, QVBoxLayout, QWidget

from spacr.qt.widgets.column_picker import (
    ACTION_USE,
    ColumnPickerButton,
    ColumnPickerDialog,
    attach_column_picker,
    find_existing,
    set_field_text,
)
from spacr.qt.widgets.flow import FlowLayout


@pytest.fixture
def measdb(tmp_path):
    """A real run folder with an ordinary table and an imported one.

    ``imported`` carries a column whose name begins with a space -- what a
    ``pandas.to_sql`` of a CSV whose header read ``" area"`` writes -- so the
    "this row cannot be accepted" path has a row that provokes it.
    """
    path = tmp_path / "run" / "measurements" / "measurements.db"
    path.parent.mkdir(parents=True)
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE png_list (png_path TEXT, plate TEXT, "
                "annotate INTEGER, test INTEGER)")
    con.execute('CREATE TABLE imported (" area" REAL, area_ok REAL)')
    con.execute("INSERT INTO png_list VALUES ('/d/0.png', 'plate1', 1, NULL)")
    con.commit()
    con.close()
    return str(path)


def _dialog(qtbot, db_path, **kw):
    dialog = ColumnPickerDialog(db_path=db_path, **kw)
    qtbot.addWidget(dialog)
    return dialog


def test_a_double_click_on_no_row_neither_types_nor_closes(qtbot, measdb):
    """An activation carrying no item must leave the dialog exactly as it was.

    ``itemDoubleClicked`` fires with a null item when the double-click lands
    on the empty space below the last row -- a click the user makes all the
    time while aiming at a column. If that were treated as a pick, the dialog
    would close and write whatever happened to be in the name box into the
    host field, so a stray double-click would silently choose a column.
    """
    dialog = _dialog(qtbot, measdb, table="png_list")
    dialog.set_name("annotate")
    assert dialog.action() == ACTION_USE
    assert dialog.is_accept_enabled() is True

    dialog._column_tree.itemDoubleClicked.emit(None, 0)

    assert dialog.name_edit().text() == "annotate"
    assert dialog.result() != QDialog.Accepted
    assert dialog.isVisible() is False

    # The same signal, this time with the row a real double-click carries:
    # it types the column and closes the dialog, which is what makes the
    # assertions above a statement about the empty click and not about a
    # handler that ignores everything.
    row = dialog._column_tree.findItems("test", Qt.MatchExactly, 0)[0]
    dialog._column_tree.itemDoubleClicked.emit(row, 0)

    assert dialog.name_edit().text() == "test"
    assert dialog.result() == QDialog.Accepted


def test_a_double_click_on_a_spaced_stored_column_accepts_that_column(
        qtbot, measdb):
    """A listed SQL name remains selectable even when its header has spaces.

    ``allow_new=False`` fields (the heatmap feature, the regression
    dependent) take an existing column only. Imported CSV headers can retain
    leading whitespace in SQLite; if the tree displays such a name, the
    verdict must recognize it and return the exact stored spelling.
    """
    dialog = _dialog(qtbot, measdb, table="imported", allow_new=False)
    assert dialog.column_names() == [" area", "area_ok"]

    stray = dialog._column_tree.findItems(" area", Qt.MatchExactly, 0)[0]
    dialog._column_tree.itemDoubleClicked.emit(stray, 0)

    assert dialog.name_edit().text() == " area"
    assert dialog.action() == ACTION_USE
    assert dialog.result() == QDialog.Accepted
    assert find_existing(" area", dialog.column_names()) == " area"
    assert find_existing("area", dialog.column_names()) == " area"
    assert find_existing("area", [" area", "area"]) == "area"


def test_asking_for_columns_the_table_lacks_selects_nothing(qtbot, measdb):
    """Names that are not in the table must clear the highlight, not keep it.

    ``select_columns`` is how a host restores the columns a settings key
    already holds when the picker reopens. After the user switches tables --
    or edits the setting by hand -- those names need not exist any more, and
    a run's ``exclude`` list would then be restored from a table that never
    had them. The stale highlight is the dangerous outcome: ``chosen_columns``
    reads the selection, so a leftover row would be added to the field the
    user never picked.
    """
    dialog = _dialog(qtbot, measdb, table="png_list", multi=True)
    dialog.set_name("annotate")

    missing = dialog.select_columns(["cell_area", "nucleus_area"])

    assert missing == []
    assert dialog._column_tree.selectedItems() == []
    assert dialog.name_edit().text() == "annotate"
    # The verdict still ran over the empty selection: one name, judged.
    assert dialog.chosen_columns() == ["annotate"]
    assert dialog.action() == ACTION_USE

    # Names the table does have are highlighted, the last one lands in the
    # name box, and all of them come back -- the case the empty one is the
    # counterpart of.
    found = dialog.select_columns(["annotate", "test"])

    assert found == ["annotate", "test"]
    assert dialog.name_edit().text() == "test"
    assert sorted(dialog.chosen_columns()) == ["annotate", "test"]
    assert len(dialog._column_tree.selectedItems()) == 2


def test_a_closed_combo_reuses_the_entry_it_already_offers(qtbot):
    """Writing a name a fixed combo already lists must not duplicate it.

    Several column settings are edited through a non-editable combo box
    filled from the module's own list of allowed values. Picking a name that
    is already in that list has to select the existing entry: appending a
    second identical row would grow the list by one every time the user
    opened the picker, and the duplicates would be indistinguishable on
    screen while comparing unequal by index everywhere else.
    """
    combo = QComboBox()
    combo.addItems(["annotate", "test"])
    qtbot.addWidget(combo)
    assert combo.isEditable() is False

    assert set_field_text(combo, "test") is True
    assert combo.currentText() == "test"
    assert combo.currentIndex() == 1
    assert combo.count() == 2
    assert [combo.itemText(i) for i in range(combo.count())] == ["annotate",
                                                                "test"]

    # A name the combo does not offer is added and then selected -- the same
    # call, one input away, is what makes the count above meaningful.
    assert set_field_text(combo, "annotate_v2") is True
    assert combo.count() == 3
    assert combo.currentText() == "annotate_v2"
    assert combo.itemText(2) == "annotate_v2"


def test_a_field_in_a_wrapping_row_keeps_its_slot_with_the_picker(qtbot):
    """A picker wrapper replaces the field in place inside ``FlowLayout``.

    ``attach_column_picker`` re-uses the field's own slot by calling
    ``QLayout.replaceWidget``. That works for the box and form layouts every
    settings Section builds, but spaCR's :class:`FlowLayout` is a Python
    ``QLayout`` subclass with no ``replaceAt``, so ``replaceWidget`` returns
    ``None``. The fallback must retain the exact slot: appending the wrapper
    would reorder a settings row even though the field became visible again.
    """
    host = QWidget()
    qtbot.addWidget(host)
    flow = FlowLayout(host)
    before = QLineEdit("before")
    field = QLineEdit()
    after = QLineEdit("after")
    flow.addWidget(before)
    flow.addWidget(field)
    flow.addWidget(after)
    assert flow.indexOf(field) == 1

    button = attach_column_picker(field, lambda: "", "png_list")

    assert isinstance(button, ColumnPickerButton)
    wrapper = field.parentWidget()
    assert wrapper.objectName() == "ColumnPickerRow"
    assert button.parentWidget() is wrapper
    assert wrapper.parentWidget() is host
    assert flow.count() == 3
    assert flow.indexOf(field) == -1
    assert flow.indexOf(wrapper) == 1
    assert [flow.itemAt(index).widget() for index in range(flow.count())] == [
        before,
        wrapper,
        after,
    ]
    host.resize(800, 100)
    host.show()
    qtbot.waitExposed(host)
    assert field.isVisible() and button.isVisible()
    assert before.geometry().x() < wrapper.geometry().x() < after.geometry().x()

    # A box layout -- what every Section actually uses -- does the swap, so
    # the fallback above must not disturb the ordinary replacement path.
    box_host = QWidget()
    qtbot.addWidget(box_host)
    box = QVBoxLayout(box_host)
    box_field = QLineEdit()
    box.addWidget(box_field)

    attach_column_picker(box_field, lambda: "", "png_list")

    box_wrapper = box_field.parentWidget()
    assert box_wrapper.objectName() == "ColumnPickerRow"
    assert box.count() == 1
    assert box.indexOf(box_wrapper) == 0
    assert box.indexOf(box_field) == -1
