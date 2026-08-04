"""'Exclude' takes any number of columns, typed or picked.

The reported complaint: "when trying to exclude columns in 'Exclude' it looks
like the user can only exclude one column at a time". It was true twice over.
``spacr.settings`` declares ``exclude`` as ``(str, None)``, so the settings
screen's deliberately-conservative ``list_shape_for`` sent it to a plain text
box; and the ``SQL`` button beside that box returned exactly one name per
press and *overwrote* whatever was already in the field.

Both halves are pinned here:

* the field is the same chip strip Classify (CV)'s ``classes`` uses --
  ``_ListEditor`` -- so a name typed becomes a chip to the right and can be
  removed on its own, and the value round-trips as a real list;
* the picker can hand back several columns from one visit, and they are
  *added* to the chips rather than replacing them;
* and excluding a column actually removes it from the fit, not just from the
  settings dict -- that last one runs the real ``filter_dataframe_features``.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from PySide6.QtWidgets import QDialog, QLineEdit

from spacr.qt.screens.settings_model import (
    EXCLUDE_LIST_KEYS,
    SettingsWidgets,
    _ListEditor,
)
from spacr.qt.widgets.column_picker import (
    ColumnPickerButton,
    ColumnPickerDialog,
    attach_column_picker,
    field_is_list,
    field_text,
    field_values,
    set_field_values,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CELL_COLUMNS = ("cell_area", "cell_perimeter", "cell_channel_0_mean_intensity",
                "nucleus_area")


@pytest.fixture
def measdb(tmp_path):
    """A real run folder with a measurements.db the picker can read."""
    root = tmp_path / "run"
    (root / "measurements").mkdir(parents=True)
    path = root / "measurements" / "measurements.db"
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE cell ("
                + ", ".join(f"{c} REAL" for c in CELL_COLUMNS) + ")")
    for i in range(5):
        con.execute("INSERT INTO cell VALUES (?,?,?,?)",
                    (100.0 + i, 40.0 + i, 0.5 + i, 20.0 + i))
    con.commit()
    con.close()
    return type("Db", (), {"path": str(path), "root": str(root)})()


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    """Nothing here may enter an event loop; the dialog runner is injected."""
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    def _boom(*_a, **_k):
        raise AssertionError("a modal dialog was opened")

    for name in ("about", "critical", "information", "question", "warning"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(_boom))
    monkeypatch.setattr(QDialog, "exec", _boom, raising=False)
    for name in ("getOpenFileName", "getSaveFileName", "getExistingDirectory"):
        monkeypatch.setattr(QFileDialog, name, staticmethod(_boom))
    yield


def _settled(qtbot, dialog, timeout=20000):
    qtbot.waitUntil(lambda: not dialog.is_busy(), timeout=timeout)
    return dialog


# ---------------------------------------------------------------------------
# The field is a chip strip
# ---------------------------------------------------------------------------

@pytest.mark.qt
@pytest.mark.parametrize("app_key", ["ml_analyze", "umap"])
def test_exclude_is_a_chip_strip_not_a_text_box(qtbot, app_key):
    widgets = SettingsWidgets(app_key)
    widgets.build_sections()

    field = widgets._widgets.get("exclude")
    assert field is not None, f"{app_key} did not render 'exclude'"
    assert isinstance(field, _ListEditor), (
        f"{app_key}'s Exclude is a {type(field).__name__}; it must be the "
        f"same chip strip Classify (CV)'s `classes` uses")


@pytest.mark.qt
def test_exclude_round_trips_several_columns(qtbot):
    widgets = SettingsWidgets("ml_analyze")
    widgets.build_sections()
    names = ["cell_area", "nucleus_area", "cell_channel_0_mean_intensity"]

    assert widgets.set_value_for_key("exclude", names)

    assert widgets.collect()["exclude"] == names


@pytest.mark.qt
def test_an_untouched_exclude_is_still_none(qtbot):
    """The default is None, and a field nobody touched must still say None --
    an empty list would change what every consumer sees."""
    widgets = SettingsWidgets("ml_analyze")
    widgets.build_sections()

    assert widgets.collect()["exclude"] is None


@pytest.mark.qt
def test_a_single_name_still_loads_from_an_old_settings_csv(qtbot):
    """Settings written before this took one bare name. It still loads."""
    widgets = SettingsWidgets("ml_analyze")
    widgets.build_sections()

    widgets.set_value_for_key("exclude", "cell_area")

    assert widgets.collect()["exclude"] == ["cell_area"]


@pytest.mark.qt
def test_a_chip_can_be_removed_on_its_own(qtbot):
    editor = _ListEditor(key="exclude", default=["a", "b", "c"],
                         allow_none=True, element_type=str)
    qtbot.addWidget(editor)

    strip = editor._strips[0]
    strip._remove_chip(strip._chips[1])

    assert editor.get_value() == ["a", "c"]


# ---------------------------------------------------------------------------
# The picker returns several at once
# ---------------------------------------------------------------------------

@pytest.mark.qt
def test_the_dialog_can_select_several_columns_at_once(qtbot, measdb):
    d = ColumnPickerDialog(db_path=measdb.path, table="cell", multi=True)
    qtbot.addWidget(d)

    found = d.select_columns(["cell_area", "nucleus_area"])

    assert found == ["cell_area", "nucleus_area"]
    assert d.chosen_columns() == ["cell_area", "nucleus_area"]
    assert d.is_accept_enabled()
    assert "2 columns selected" in d.status_text()


@pytest.mark.qt
def test_a_single_select_dialog_still_returns_exactly_one(qtbot, measdb):
    d = ColumnPickerDialog(db_path=measdb.path, table="cell")
    qtbot.addWidget(d)

    d.select_column("cell_area")

    assert d.chosen_columns() == ["cell_area"]
    assert d.chosen_column() == "cell_area"


@pytest.mark.qt
def test_a_typed_name_joins_the_selection_rather_than_replacing_it(
        qtbot, measdb):
    """That is how a column that does not exist yet is named in multi mode."""
    d = ColumnPickerDialog(db_path=measdb.path, table="cell", multi=True)
    qtbot.addWidget(d)

    d.select_columns(["cell_area", "nucleus_area"])
    d.set_name("a_new_one")

    assert d.chosen_columns() == ["cell_area", "nucleus_area", "a_new_one"]


@pytest.mark.qt
def test_the_button_writes_every_picked_column_into_the_chip_strip(
        qtbot, measdb):
    editor = _ListEditor(key="exclude", default=None, allow_none=True,
                         element_type=str)
    qtbot.addWidget(editor)
    button = ColumnPickerButton(lambda: measdb.path, table="cell",
                                field=editor)
    qtbot.addWidget(button)

    many = []
    button.picked_many.connect(many.append)
    button.set_dialog_runner(
        lambda d: (_settled(qtbot, d).select_columns(
            ["cell_area", "nucleus_area"]), QDialog.Accepted)[1])

    button.open_picker()

    assert editor.get_value() == ["cell_area", "nucleus_area"]
    assert many == [["cell_area", "nucleus_area"]]


@pytest.mark.qt
def test_a_second_visit_adds_to_the_chips_rather_than_replacing_them(
        qtbot, measdb):
    """The complaint's other half: the SQL button used to overwrite."""
    editor = _ListEditor(key="exclude", default=["cell_perimeter"],
                         allow_none=True, element_type=str)
    qtbot.addWidget(editor)
    button = ColumnPickerButton(lambda: measdb.path, table="cell",
                                field=editor)
    qtbot.addWidget(button)
    button.set_dialog_runner(
        lambda d: (_settled(qtbot, d).select_columns(
            ["cell_area", "nucleus_area"]), QDialog.Accepted)[1])

    button.open_picker()

    assert editor.get_value() == [
        "cell_perimeter", "cell_area", "nucleus_area"]


@pytest.mark.qt
def test_picking_the_same_column_twice_does_not_duplicate_a_chip(
        qtbot, measdb):
    editor = _ListEditor(key="exclude", default=["cell_area"],
                         allow_none=True, element_type=str)
    qtbot.addWidget(editor)
    button = ColumnPickerButton(lambda: measdb.path, table="cell",
                                field=editor)
    qtbot.addWidget(button)
    button.set_dialog_runner(
        lambda d: (_settled(qtbot, d).select_columns(["cell_area"]),
                   QDialog.Accepted)[1])

    button.open_picker()

    assert editor.get_value() == ["cell_area"]


@pytest.mark.qt
def test_cancelling_leaves_the_chips_alone(qtbot, measdb):
    editor = _ListEditor(key="exclude", default=["cell_area"],
                         allow_none=True, element_type=str)
    qtbot.addWidget(editor)
    button = ColumnPickerButton(lambda: measdb.path, table="cell",
                                field=editor)
    qtbot.addWidget(button)
    button.set_dialog_runner(
        lambda d: (_settled(qtbot, d).select_columns(["nucleus_area"]),
                   QDialog.Rejected)[1])

    assert button.open_picker() == ""
    assert editor.get_value() == ["cell_area"]


@pytest.mark.qt
def test_a_chip_strip_is_recognised_as_list_valued_even_when_empty(qtbot):
    """A text inspection cannot tell -- an empty chip strip reads '' -- and
    an empty Exclude is exactly when several columns are most wanted."""
    editor = _ListEditor(key="exclude", default=None, allow_none=True,
                         element_type=str)
    qtbot.addWidget(editor)

    assert field_is_list(editor)
    assert field_values(editor) == []
    assert not field_is_list(QLineEdit("cell_area"))


@pytest.mark.qt
def test_the_helpers_read_and_write_a_chip_strip(qtbot):
    editor = _ListEditor(key="exclude", default=["a"], allow_none=True,
                         element_type=str)
    qtbot.addWidget(editor)

    assert field_values(editor) == ["a"]
    assert field_text(editor) == "a"
    set_field_values(editor, ["b", "c"], append=True)
    assert field_values(editor) == ["a", "b", "c"]
    set_field_values(editor, ["z"], append=False)
    assert field_values(editor) == ["z"]


@pytest.mark.qt
def test_a_plain_text_field_still_gets_every_picked_name(qtbot, measdb):
    """Multi-select must not silently drop names on a field that is not a
    chip strip -- it keeps them all, in that field's own list style."""
    field = QLineEdit("")
    qtbot.addWidget(field)
    button = ColumnPickerButton(lambda: measdb.path, table="cell",
                                field=field, multi=True)
    qtbot.addWidget(button)
    button.set_dialog_runner(
        lambda d: (_settled(qtbot, d).select_columns(
            ["cell_area", "nucleus_area"]), QDialog.Accepted)[1])

    button.open_picker()

    assert field.text() == "cell_area, nucleus_area"


@pytest.mark.qt
def test_attaching_the_picker_to_the_chip_strip_keeps_it_usable(qtbot):
    """attach_column_picker rehomes the field into a row wrapper; the chip
    strip has to survive that, since it is what collect() reads."""
    from PySide6.QtWidgets import QFormLayout, QWidget

    host = QWidget()
    form = QFormLayout(host)
    editor = _ListEditor(key="exclude", default=["a"], allow_none=True,
                         element_type=str)
    form.addRow("Exclude", editor)
    qtbot.addWidget(host)

    button = attach_column_picker(editor, lambda: "", None)

    assert button.field is editor
    assert button.is_multi()
    assert editor.get_value() == ["a"]


# ---------------------------------------------------------------------------
# Excluding actually changes the fit
# ---------------------------------------------------------------------------

def test_excluded_columns_leave_the_feature_matrix_not_just_the_dict():
    """The end of the chain: a list from the chip strip has to reach the fit.

    ``filter_dataframe_features`` is the function ``ml_analysis`` calls with
    ``settings['exclude']``.
    """
    from spacr.utils import filter_dataframe_features

    frame = pd.DataFrame({
        "cell_channel_0_mean_intensity": [1.0, 2.0, 3.0, 4.0],
        "cell_channel_0_perimeter": [5.0, 6.0, 7.0, 9.0],
        "cell_channel_0_area": [9.0, 11.0, 13.0, 17.0],
        "cell_channel_0_solidity": [0.4, 0.7, 0.2, 0.9],
    })
    dropped = ["cell_channel_0_perimeter", "cell_channel_0_area"]

    filtered, features = filter_dataframe_features(
        frame, channel_of_interest=0, exclude=dropped,
        remove_low_variance_features=False,
        remove_highly_correlated_features=False)

    for name in dropped:
        assert name not in features
        assert name not in filtered.columns
    assert "cell_channel_0_mean_intensity" in features


def test_excluding_a_column_that_is_not_there_says_so():
    """A typo in Exclude must not be silently ignored -- silently keeping a
    feature the user asked to drop is the failure that has no symptom."""
    from spacr.utils import filter_dataframe_features

    frame = pd.DataFrame({"cell_channel_0_area": [1.0, 2.0]})

    with pytest.raises(ValueError, match="cell_channel_0_arae"):
        filter_dataframe_features(
            frame, channel_of_interest=0,
            exclude=["cell_channel_0_area", "cell_channel_0_arae"])


def test_exclude_list_keys_is_what_the_widget_switch_reads():
    assert "exclude" in EXCLUDE_LIST_KEYS
