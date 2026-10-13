import os

import pandas as pd
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QFileDialog, QPushButton, QWidget

from spacr import tabular
from spacr.qt import path_probe
from spacr.qt.screens.settings_model import (
    SettingsWidgets,
    _CsvColumnField,
    regression_design_scan,
)
from spacr.qt.widgets.file_list import PairedFileTableWidget
from spacr.qt.widgets.measurement_scan_panel import column_run_settings


def test_named_store_controls_follow_alpha_preference_without_losing_rows(
        qtbot, tmp_path, monkeypatch):
    from spacr.qt import preferences
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.settings import ALPHA_FEATURES

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    preferences._set_show_alpha_features(False)
    assert "ALPHA_FEATURES576" in ALPHA_FEATURES[576]["widgets"]

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    widget = screen._settings_model._widgets["paired_data"]
    widget.set_value([{"score": "scores.duckdb", "score_table": "scores",
                       "count": "counts.csv"}])
    buttons = widget.findChildren(QPushButton, "ALPHA_FEATURES576")
    assert len(buttons) == 2
    assert all(button.isHidden() for button in buttons)
    assert all(widget.table.isColumnHidden(column) for column in
               (widget.SCORE_TABLE_COLUMN, widget.COUNT_TABLE_COLUMN))
    assert widget.get_value()[0]["score_table"] == "scores"

    preferences._set_show_alpha_features(True)
    screen._refresh_alpha_visibility()
    assert all(not button.isHidden() for button in buttons)
    assert all(not widget.table.isColumnHidden(column) for column in
               (widget.SCORE_TABLE_COLUMN, widget.COUNT_TABLE_COLUMN))
    widget.table.cellWidget(0, widget.COUNT_TABLE_COLUMN).setText("counts")
    assert widget.get_value()[0]["count_table"] == "counts"

    preferences._set_show_alpha_features(False)
    screen._refresh_alpha_visibility()
    assert all(button.isHidden() for button in buttons)
    assert all(widget.table.isColumnHidden(column) for column in
               (widget.SCORE_TABLE_COLUMN, widget.COUNT_TABLE_COLUMN))
    assert widget.get_value()[0]["score_table"] == "scores"
    assert widget.get_value()[0]["count_table"] == "counts"


def test_named_pair_survives_edit_pairing_move_and_remove(qtbot):
    widget = PairedFileTableWidget(value=[{
        "plate": "P", "score": "/tmp/p_scores.duckdb",
        "score_table": "scores", "count": None,
    }])
    qtbot.addWidget(widget)
    assert widget.table.cellWidget(0, widget.SCORE_TABLE_COLUMN).objectName() == \
        "ALPHA_FEATURES576"
    widget.add_paths_for_side(["/tmp/p_counts.csv"], "count")
    assert widget.get_value()[0]["score_table"] == "scores"
    assert widget.get_value()[0]["count"] == "/tmp/p_counts.csv"

    widget.add_paths_for_side(["/tmp/q_scores.csv"], "score")
    widget.table.selectRow(0)
    widget._move(1)
    kept = next(row for row in widget.get_value()
                if row.get("score") == "/tmp/p_scores.duckdb")
    assert kept["score_table"] == "scores"
    assert kept["count"] == "/tmp/p_counts.csv"
    widget.table.selectRow(next(index for index, row in
                                enumerate(widget.get_value())
                                if row.get("score") == "/tmp/q_scores.csv"))
    widget._remove()
    assert len(widget.get_value()) == 1
    assert widget.get_value()[0]["score_table"] == "scores"


def test_two_named_tables_in_one_store_survive_a_new_arrival(qtbot):
    store = "/tmp/paired.duckdb"
    widget = PairedFileTableWidget(value=[
        {"plate": "P1", "score": store, "score_table": "score_one",
         "count": store, "count_table": "count_one"},
        {"plate": "P2", "score": store, "score_table": "score_two",
         "count": store, "count_table": "count_two"},
    ])
    qtbot.addWidget(widget)
    widget.add_paths_for_side(["/tmp/new_scores.csv"], "score")
    named = [row for row in widget.get_value() if row.get("score") == store]
    assert [(row["score_table"], row["count_table"]) for row in named] == [
        ("score_one", "count_one"), ("score_two", "count_two")]
    widget.table.selectRow(0)
    widget._move(1)
    named = [row for row in widget.get_value() if row.get("score") == store]
    assert {(row["plate"], row["score_table"], row["count_table"])
            for row in named} == {
                ("P1", "score_one", "count_one"),
                ("P2", "score_two", "count_two")}


def test_regression_form_collect_keeps_saved_table_names(qtbot):
    owner = QWidget()
    qtbot.addWidget(owner)
    model = SettingsWidgets("regression", parent=owner)
    model.build_sections()
    widget = model._widgets["paired_data"]
    widget.set_value([{
        "plate": "P", "score": "/tmp/data.duckdb",
        "score_table": "scores", "count": "/tmp/data.duckdb",
        "count_table": "counts",
    }])
    collected = model.collect()["paired_data"]
    assert collected[0]["score_table"] == "scores"
    assert collected[0]["count_table"] == "counts"


def test_store_folder_picker_requires_a_parquet_store(tmp_path, qtbot,
                                                      monkeypatch):
    wrong = tmp_path / "ordinary"
    store = tmp_path / "paired.parquetdb"
    wrong.mkdir()
    store.mkdir()
    widget = PairedFileTableWidget()
    qtbot.addWidget(widget)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        lambda *args: str(wrong))
    widget._pick_store_folder("score")
    assert widget.get_value() == []
    assert ".parquetdb" in widget.status.text()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        lambda *args: str(store))
    widget._pick_store_folder("score")
    assert widget.get_value()[0]["score"] == str(store)
    assert "Name the table for score 1" in widget.status.text()
    widget.table.cellWidget(0, widget.SCORE_TABLE_COLUMN).setText("scores")
    assert "Name the table" not in widget.status.text()


def test_explicit_score_store_picker_does_not_attach_measurements_db(
        tmp_path, qtbot, monkeypatch):
    store = tmp_path / "scores.db"
    store.touch()
    widget = PairedFileTableWidget()
    qtbot.addWidget(widget)
    monkeypatch.setattr(QFileDialog, "getOpenFileNames",
                        lambda *args: ([str(store)], ""))
    widget._pick("score")
    assert widget.get_value()[0]["score"] == str(store)
    assert widget.get_value()[0]["database"] is None
    assert "Name the table for score 1" in widget.status.text()


@pytest.mark.parametrize("backend", ["sqlite", "duckdb", "parquet"])
def test_named_backend_columns_and_design_scan_are_real_tables(
        tmp_path, qtbot, backend):
    if backend == "duckdb":
        pytest.importorskip("duckdb")
    if backend == "parquet":
        pytest.importorskip("pyarrow")
    extension = {"sqlite": ".db", "duckdb": ".duckdb",
                 "parquet": ".parquetdb"}[backend]
    store = tmp_path / f"paired{extension}"
    tabular.write_database(
        pd.DataFrame({"plateID": ["p1"], "phenotype": [1.5]}),
        str(store), "scores", if_exists="replace")
    tabular.write_database(
        pd.DataFrame({"plateID": ["p1"], "rowID": ["A"],
                      "columnID": [1], "grna": ["org_gene_guide"],
                      "count": [3]}),
        str(store), "counts", if_exists="replace")
    pair = {"plate": "p1", "score": str(store),
            "score_table": "scores", "count": str(store),
            "count_table": "counts"}
    widget = PairedFileTableWidget(value=[pair])
    qtbot.addWidget(widget)
    assert widget.get_value()[0]["score_table"] == "scores"
    assert widget.get_value()[0]["count_table"] == "counts"

    model = SettingsWidgets("regression")
    model._widgets = {"paired_data": widget}
    assert model._input_csv_paths(("score",)) == [(str(store), "scores")]
    context = model._plate_context(model._loaded_table_paths(
        {"paired_data": widget.get_value()}))
    assert context == {"plate_count": None, "has_plate_id": None}

    field = _CsvColumnField(key="dependent_variable", default="phenotype",
                            paths=lambda: model._input_csv_paths(("score",)))
    qtbot.addWidget(field)
    seen = []
    field.set_chooser(lambda choices, current: seen.extend(choices)
                      or "phenotype")
    assert field.pick() == "phenotype"
    assert "phenotype" in seen
    scan = regression_design_scan({"paired_data": widget.get_value()})
    assert scan["files"] == scan["rows"] == scan["genes"] == 1
    assert scan["note"] == ""


def test_generated_score_csv_clears_only_its_obsolete_table_name():
    base = {"paired_data": [{"score": "old.duckdb",
                             "score_table": "scores", "count": "old.db",
                             "count_table": "counts"}]}
    result = column_run_settings(base, "cell_area", "/data/generated.csv")
    assert result["paired_data"][0]["score"] == "/data/generated.csv"
    assert "score_table" not in result["paired_data"][0]
    assert result["paired_data"][0]["count_table"] == "counts"
    assert base["paired_data"][0]["score_table"] == "scores"


def test_owned_postgres_table_is_pickable_and_scannable(qtbot):
    dsn = os.environ.get("SPACR_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("requires the owned PostgreSQL test database")
    pytest.importorskip("psycopg")
    score_table = "gui_f576_scores"
    count_table = "gui_f576_counts"
    tabular.write_database(
        pd.DataFrame({"plateID": ["p1"], "phenotype": [2.0]}),
        dsn, score_table, if_exists="replace")
    tabular.write_database(
        pd.DataFrame({"plateID": ["p1"], "rowID": ["A"],
                      "columnID": [1], "grna": ["org_gene_guide"],
                      "count": [4]}),
        dsn, count_table, if_exists="replace")
    widget = PairedFileTableWidget(value=[{
        "plate": "p1", "score": dsn, "score_table": score_table,
        "count": dsn, "count_table": count_table,
    }])
    qtbot.addWidget(widget)
    model = SettingsWidgets("regression")
    model._widgets = {"paired_data": widget}
    field = _CsvColumnField(key="dependent_variable", default="phenotype",
                            paths=lambda: model._input_csv_paths(("score",)))
    qtbot.addWidget(field)
    field.set_chooser(lambda choices, current: "phenotype"
                      if "phenotype" in choices else None)
    assert field.pick() == "phenotype"
    assert regression_design_scan({"paired_data": widget.get_value()})[
        "rows"] == 1


def test_missing_named_sqlite_source_is_reported_without_creating_it(
        tmp_path, qtbot):
    missing = tmp_path / "missing.db"
    field = _CsvColumnField(
        key="dependent_variable", default="phenotype",
        paths=lambda: [(str(missing), "scores")])
    qtbot.addWidget(field)
    reports = []
    field.set_reporter(reports.append)
    assert field.pick() is None
    assert reports and str(missing) in reports[0]
    assert not missing.exists()
    assert SettingsWidgets._plate_context(
        [(0, str(missing), "scores")])["plate_count"] is None
    assert not missing.exists()


def test_plate_context_never_connects_to_named_store_while_editing(
        monkeypatch):
    def no_query(*args, **kwargs):
        raise AssertionError("editing a setting queried a database")

    monkeypatch.setattr(tabular, "table_columns", no_query)
    assert SettingsWidgets._plate_context([
        (0, "postgresql:///some_remote_database", "scores")]) == {
            "plate_count": None, "has_plate_id": None}


def test_typing_table_name_does_not_reprobe_attached_measurement_path(
        qtbot, monkeypatch):
    widget = PairedFileTableWidget(value=[{
        "score": "/tmp/score.duckdb",
        "database": "/tmp/unavailable-measurements.db",
    }])
    qtbot.addWidget(widget)
    assert "NOT ON DISK" in widget.status.text()

    def no_probe(*args, **kwargs):
        raise AssertionError("typing a table name probed the filesystem")

    monkeypatch.setattr(path_probe, "exists", no_probe)
    widget.table.cellWidget(0, widget.SCORE_TABLE_COLUMN).setText("scores")
    assert "NOT ON DISK" in widget.status.text()
    assert "Name the table" not in widget.status.text()


def test_parquet_column_header_never_reads_a_data_batch(
        tmp_path, monkeypatch):
    parquet = pytest.importorskip("pyarrow.parquet")
    store = tmp_path / "pair.parquetdb"
    tabular.write_database(pd.DataFrame({"phenotype": [1.5]}), str(store),
                           "scores", if_exists="replace")

    def no_batch(*args, **kwargs):
        raise AssertionError("column picker read Parquet data")

    monkeypatch.setattr(parquet, "ParquetFile", no_batch)
    assert "phenotype" in tabular.table_columns(str(store), table="scores")
    plain = tmp_path / "score.parquet"
    pd.DataFrame({"phenotype": [1.5]}).to_parquet(plain)
    monkeypatch.setattr(pd, "read_parquet", no_batch)
    assert "phenotype" in tabular.table_columns(str(plain))
    assert SettingsWidgets._plate_context([(0, str(plain))]) == {
        "plate_count": None, "has_plate_id": None}
