import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication, QFileDialog

from spacr import columns
from spacr.qt.screens.settings_model import (
    SettingsWidgets,
    _CsvColumnField,
)
from spacr.qt.widgets.file_list import PairedFileTableWidget


@pytest.fixture
def measurements(tmp_path):
    path = tmp_path / "measurements.csv"
    pd.DataFrame({
        "prcfo": ["p1_r1_c1_f1_o1", "p1_r1_c1_f1_o2"],
        "plateID": [1, 1],
        "cell_mean_intensity": [1.5, 2.5],
        "cell_max_intensity": [3, 4],
    }).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def classified(tmp_path):
    path = tmp_path / "240101_set_model_result.csv"
    pd.DataFrame({"path": ["a", "b"], "area": [3.0, 4.0],
                  "pred": [0.1, 0.9]}).to_csv(path, index=False)
    return str(path)


def _field(qtbot, paths, default="pred", response=True):
    field = _CsvColumnField(key="dependent_variable", default=default,
                            paths=lambda: list(paths), response=response)
    qtbot.addWidget(field)
    return field


def _settled(field):
    field.refresh().join()
    QApplication.processEvents()


def _items(field):
    return [field.edit.itemText(i) for i in range(field.edit.count())]


def test_numeric_headers_skip_labels_and_identifiers(measurements, tmp_path):
    found = columns.numeric_headers([measurements, str(tmp_path / "gone.csv")])
    assert found == {measurements: ["cell_mean_intensity",
                                    "cell_max_intensity"]}


def test_numeric_headers_skip_unparseable_files(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_bytes(b"")
    assert columns.numeric_headers(str(bad)) == {}


def test_pred_missing_lists_only_files_without_pred(measurements, classified):
    assert columns.pred_missing([measurements, classified]) == [measurements]
    assert "Classify" in columns.pred_guidance()


def test_combo_is_filled_from_the_csv_and_keeps_pred_first(qtbot, classified):
    field = _field(qtbot, [classified])
    _settled(field)
    assert _items(field) == ["pred", "area"]
    assert field.get_value() == "pred"
    assert field.note.isHidden()
    assert field.edit.toolTip() == ""


def test_combo_lists_the_numeric_columns_of_a_measurement_table(
        qtbot, measurements):
    field = _field(qtbot, [measurements])
    _settled(field)
    assert _items(field) == ["cell_mean_intensity", "cell_max_intensity"]
    assert field.get_value() == "pred"


def test_note_appears_when_pred_is_absent_and_says_where_it_comes_from(
        qtbot, measurements):
    field = _field(qtbot, [measurements])
    _settled(field)
    text = field.note.text()
    assert not field.note.isHidden()
    assert "measurements.csv has no 'pred' column" in text
    assert "Classify" in text and "_result.csv" in text
    assert "cell_mean_intensity" in text
    assert field.edit.toolTip() == text
    assert field.get_value() == "pred"


def test_note_clears_when_a_listed_column_is_chosen(qtbot, measurements):
    field = _field(qtbot, [measurements])
    _settled(field)
    field.set_value("cell_mean_intensity")
    assert field.note.isHidden()


def test_a_saved_value_not_in_the_file_is_kept_and_flagged(
        qtbot, measurements):
    field = _field(qtbot, [measurements], default="old_name")
    _settled(field)
    assert field.get_value() == "old_name"
    assert "'old_name' is not a numeric column" in field.note.text()


def test_non_response_field_gets_no_note_and_all_columns(qtbot, measurements):
    field = _field(qtbot, [measurements], default="prcfo", response=False)
    _settled(field)
    assert "prcfo" in _items(field)
    assert field.note.isHidden()


def test_no_files_gives_no_note(qtbot):
    field = _field(qtbot, [])
    _settled(field)
    assert field.note.isHidden()
    assert field.edit.count() == 0


def test_a_stale_read_is_dropped(qtbot, measurements):
    field = _field(qtbot, [measurements])
    field._generation = 5
    field._apply_read({"generation": 4, "names": ["x"], "lacking": [],
                       "read": True})
    assert field.edit.count() == 0


def test_stored_tables_columns_join_the_choices(qtbot, tmp_path):
    from spacr import tabular

    store = str(tmp_path / "scores.db")
    tabular.write_database(pd.DataFrame({"pred": [0.2], "x": [1.0]}), store,
                        table="scores")
    field = _field(qtbot, [(store, "scores")])
    _settled(field)
    items = _items(field)
    assert items[0] == "pred"
    assert "x" in items


def test_showing_the_field_reads_once(qtbot, measurements):
    field = _field(qtbot, [measurements])
    field.show()
    field._thread.join()
    first = field._thread
    field.hide()
    field.show()
    assert field._thread is first


def test_combo_keeps_the_line_edit_contract(qtbot):
    field = _field(qtbot, [])
    field.set_value(None)
    assert field.get_value() is None
    field.setText("abc")
    assert field.text() == "abc"
    assert field.edit.text() == "abc"


def test_changing_the_input_files_refreshes_the_choices(qtbot, measurements):
    model = SettingsWidgets("regression")
    paired = PairedFileTableWidget(value=[{"score": measurements}])
    qtbot.addWidget(paired)
    field = _field(qtbot, [measurements])
    model._widgets = {"paired_data": paired, "dependent_variable": field,
                      "filter_column": object()}
    model._connect_column_fields()
    paired.value_changed.emit()
    field._thread.join()
    QApplication.processEvents()
    assert field.edit.count() == 2


def test_score_picker_offers_classify_outputs_first(qtbot, monkeypatch):
    widget = PairedFileTableWidget()
    qtbot.addWidget(widget)
    seen = {}

    def fake(parent, title, directory, filters):
        seen[title] = filters
        return [], ""

    monkeypatch.setattr(QFileDialog, "getOpenFileNames", staticmethod(fake))
    widget._pick("score")
    widget._pick("count")
    assert seen["Add score tables"].startswith("Classify and ML scores")
    assert "_result.csv" in seen["Add score tables"]
    assert seen["Add count tables"].startswith("Tables")
