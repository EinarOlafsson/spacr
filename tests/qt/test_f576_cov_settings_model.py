"""Edge behaviour of the F576 store-aware code in ``settings_model``.

Covers the measurement-store migration dialog and its copy helper, the
store-aware regression design scan, the column picker over named tables and
the plate-count context of the paired regression inputs. Every modal is
replaced; no PostgreSQL server is contacted (locators are only parsed or the
tabular calls are faked).
"""
from __future__ import annotations

import os
import sqlite3
import sys

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd                                                # noqa: E402

from spacr import tabular                                          # noqa: E402
from spacr.qt.screens import settings_model                        # noqa: E402
from spacr.qt.screens.settings_model import (                      # noqa: E402
    SettingsWidgets,
    _CsvColumnField,
    regression_design_scan,
)

PG = "postgresql://user@db.invalid/measurements"


def _sqlite_store(path, tables=("cell",)):
    for name in tables:
        tabular.write_database(pd.DataFrame({"id": [1]}), path, name,
                               if_exists="replace", canonicalise=False)
    return path


class _Emitter:
    def __init__(self):
        self.calls = []

    def emit(self, value):
        self.calls.append(value)


class _Signals:
    def __init__(self):
        self.progress, self.copied, self.failed = (_Emitter(), _Emitter(),
                                                   _Emitter())


def test_design_scan_without_any_count_file_says_so():
    scan = regression_design_scan({"paired_data": [{"score": "s.csv"}]})
    assert scan["files"] == 0 and scan["rows"] == 0
    assert scan["note"] == "no count files in the settings"
    assert regression_design_scan(None)["note"] == (
        "no count files in the settings")


def test_design_scan_falls_back_to_legacy_counts_and_names_unread_sources(
        tmp_path):
    counts = tmp_path / "counts.csv"
    pd.DataFrame({"rowID": ["A", "B"], "columnID": [1, 2],
                  "grna": ["org_g1_a", "org_g2_b"]}).to_csv(counts,
                                                           index=False)
    missing = tmp_path / "missing.csv"
    scan = regression_design_scan({
        "paired_data": [{"score": "scores.csv"}],
        "count_data": [str(counts), str(missing)]})
    assert scan["files"] == 1 and scan["rows"] == 2
    assert scan["guides"] == 2 and scan["genes"] == 2 and scan["wells"] == 2
    assert f"could not read {missing} (" in scan["note"]


def test_design_scan_names_the_table_of_an_unreadable_named_store(tmp_path):
    store = tmp_path / "absent.parquetdb"
    scan = regression_design_scan({"paired_data": [
        {"count": str(store), "count_table": "counts"}]})
    assert scan["files"] == 0
    assert f"could not read {store}:counts (" in scan["note"]
    assert not store.exists()


def test_migration_inputs_accept_a_postgres_source_without_touching_it(
        tmp_path):
    target = tmp_path / "copy.db"
    assert settings_model._measurement_migration_inputs(PG, str(target)) == (
        PG, str(target), "sqlite")
    assert not target.exists()


def test_migration_inputs_refuse_the_same_postgres_store_and_a_missing_parent(
        tmp_path):
    with pytest.raises(ValueError, match="must be different stores"):
        settings_model._measurement_migration_inputs(PG, PG)
    source = _sqlite_store(tmp_path / "source.db")
    with pytest.raises(ValueError, match="parent folder does not exist"):
        settings_model._measurement_migration_inputs(
            str(source), str(tmp_path / "nowhere" / "copy.db"))
    assert not (tmp_path / "nowhere").exists()


def test_copy_refuses_a_source_without_tables(tmp_path):
    source = tmp_path / "empty.db"
    sqlite3.connect(source).close()
    source.touch()
    target = tmp_path / "copy.db"
    with pytest.raises(ValueError, match="no measurement tables"):
        settings_model._copy_measurement_store(str(source), str(target), None)
    assert not target.exists()


def test_copy_to_postgres_refuses_a_destination_with_tables_and_fills_an_empty_one(
        tmp_path, monkeypatch):
    source = _sqlite_store(tmp_path / "source.db")
    real_tables = tabular.database_tables
    remote = {"tables": ("existing",)}
    migrated = []

    def tables(db, **kwargs):
        if str(db).startswith("postgresql://"):
            return remote["tables"]
        return real_tables(db, **kwargs)

    def migrate(src, dst, *, tables, report):
        migrated.append((src, dst, tables))
        return tables

    monkeypatch.setattr(tabular, "database_tables", tables)
    monkeypatch.setattr(tabular, "_migrate_database", migrate)
    with pytest.raises(ValueError, match="already has tables"):
        settings_model._copy_measurement_store(str(source), PG, None)
    assert migrated == []

    remote["tables"] = ()
    assert settings_model._copy_measurement_store(str(source), PG,
                                                  None) == ("cell",)
    assert migrated == [(os.path.realpath(source), PG, ("cell",))]


def test_copy_refuses_a_destination_that_exists_after_validation(
        tmp_path, monkeypatch):
    source = _sqlite_store(tmp_path / "source.db")
    target = tmp_path / "taken.db"
    target.write_bytes(b"someone else's store")
    monkeypatch.setattr(settings_model, "_measurement_migration_inputs",
                        lambda s, t: (str(source), str(target), "sqlite"))
    with pytest.raises(ValueError, match="already exists"):
        settings_model._copy_measurement_store(str(source), str(target), None)
    assert target.read_bytes() == b"someone else's store"
    assert not list(tmp_path.glob(".spacr-migration-*"))


def test_copy_refuses_to_publish_a_staged_store_missing_a_table(
        tmp_path, monkeypatch):
    source = _sqlite_store(tmp_path / "source.db", ("cell", "nucleus"))
    target = tmp_path / "copy.db"

    def partial(_source, staged, *, tables, report):
        _sqlite_store(staged, ("cell",))
        return tables

    monkeypatch.setattr(tabular, "_migrate_database", partial)
    with pytest.raises(ValueError, match="missing a source table"):
        settings_model._copy_measurement_store(str(source), str(target), None)
    assert not target.exists()
    assert not list(tmp_path.glob(".spacr-migration-*"))


def _parquet_stage(monkeypatch, layout):
    """Fake a staged parquet store with ``layout`` {relative path: is_dir}."""
    monkeypatch.setattr(tabular, "database_tables",
                        lambda db, **kwargs: ("cell",))

    def stage(_source, staged, *, tables, report):
        os.mkdir(staged)
        for relative, is_dir in layout:
            path = os.path.join(staged, relative)
            if is_dir:
                os.mkdir(path)
            else:
                with open(path, "wb") as handle:
                    handle.write(b"part")
        return tables

    monkeypatch.setattr(tabular, "_migrate_database", stage)


def test_failed_parquet_publication_removes_every_placed_part(
        tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    source.write_bytes(b"")
    target = tmp_path / "copy.parquetdb"
    _parquet_stage(monkeypatch, [("meta.json", False), ("cell", True),
                                 ("cell/part-0.parquet", False)])
    native_link = os.link
    links = []

    def flaky_link(src, dst, *args, **kwargs):
        links.append(dst)
        if len(links) == 2:
            raise OSError("disk full")
        return native_link(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "link", flaky_link)
    with pytest.raises(OSError, match="disk full"):
        settings_model._copy_measurement_store(str(source), str(target), None)
    assert len(links) == 2
    assert not target.exists()
    assert not list(tmp_path.glob(".spacr-migration-*"))


def test_failed_parquet_publication_keeps_the_error_when_cleanup_cannot_rmdir(
        tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    source.write_bytes(b"")
    target = tmp_path / "copy.parquetdb"
    _parquet_stage(monkeypatch, [("part-0.parquet", False)])

    def no_link(*_args, **_kwargs):
        raise OSError("links unsupported")

    native_rmdir = os.rmdir

    def stubborn_rmdir(path, *args, **kwargs):
        if os.fspath(path) == str(target):
            raise OSError("busy")
        return native_rmdir(path, *args, **kwargs)

    monkeypatch.setattr(os, "link", no_link)
    monkeypatch.setattr(os, "rmdir", stubborn_rmdir)
    with pytest.raises(OSError, match="links unsupported"):
        settings_model._copy_measurement_store(str(source), str(target), None)
    assert target.is_dir() and not os.listdir(target)
    assert not list(tmp_path.glob(".spacr-migration-*"))


def test_worker_reports_only_the_kind_of_a_failed_copy(monkeypatch):
    def failing(_source, _target, report):
        report("Copied cell: 1 row(s) from " + PG)
        raise RuntimeError("password=secret")

    monkeypatch.setattr(settings_model, "_copy_measurement_store", failing)
    signals = _Signals()
    settings_model._run_measurement_migration(PG, "copy.db", "sqlite",
                                              signals)
    assert signals.progress.calls == ["table"]
    assert signals.failed.calls == ["sqlite"]
    assert signals.copied.calls == []


@pytest.fixture
def dialog(qtbot):
    widget = settings_model._MeasurementMigrationDialog("")
    qtbot.addWidget(widget)
    return widget


def _controls(dialog):
    return [dialog.source.isEnabled(), dialog.target.isEnabled(),
            dialog.start_button.isEnabled(), dialog.close_button.isEnabled()]


def test_dialog_start_shows_the_validation_error_without_asking(
        dialog, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QMessageBox

    asked = []
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *args: asked.append(args) or QMessageBox.Yes)
    dialog.start_button.click()
    assert dialog.log.toPlainText() == (
        "Both source and destination are required.")
    dialog.source.setText(str(tmp_path / "missing.db"))
    dialog.target.setText(str(tmp_path / "copy.db"))
    dialog.start_button.click()
    assert dialog.log.toPlainText() == (
        "The source measurement store does not exist.")
    assert asked == []
    assert all(_controls(dialog))


def test_dialog_declined_confirmation_starts_nothing(
        dialog, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QMessageBox

    source = _sqlite_store(tmp_path / "source.db")
    asked = []
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *args: asked.append(args) or QMessageBox.No)
    monkeypatch.setattr(settings_model, "_run_measurement_migration",
                        lambda *args: pytest.fail("copy started"))
    dialog.source.setText(str(source))
    dialog.target.setText(str(tmp_path / "copy.db"))
    dialog.start_button.click()
    assert len(asked) == 1
    assert dialog._running is False
    assert dialog.log.toPlainText() == ""
    assert all(_controls(dialog))
    assert not (tmp_path / "copy.db").exists()


def test_dialog_ignores_a_second_start_while_copying(dialog, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "question",
                        lambda *args: pytest.fail("asked again"))
    dialog._running = True
    dialog.log.setPlainText("Copying tables…")
    dialog._start()
    assert dialog.log.toPlainText() == "Copying tables…"


def test_dialog_progress_ignores_unknown_messages(dialog):
    dialog._on_progress("table")
    dialog._on_progress("Copied cell from " + PG)
    assert dialog.log.toPlainText() == "A table finished copying."


@pytest.mark.parametrize("kind, expected, absent", [
    ("postgres", "PostgreSQL destination may contain partial new tables",
     "no local destination"),
    ("sqlite", "no local destination was published", "PostgreSQL"),
])
def test_dialog_failure_message_follows_the_destination_kind(
        dialog, kind, expected, absent):
    dialog._running = True
    for widget in (dialog.source, dialog.target, dialog.start_button,
                   dialog.close_button):
        widget.setEnabled(False)
    dialog._on_failed(kind)
    text = dialog.log.toPlainText()
    assert text.startswith("Migration stopped. The source is unchanged")
    assert expected in text and absent not in text
    assert dialog._running is False
    assert all(_controls(dialog))


def test_dialog_reject_waits_for_a_running_copy(dialog):
    dialog.show()
    dialog._running = True
    dialog.reject()
    assert dialog.isVisible()
    dialog._running = False
    dialog.reject()
    assert not dialog.isVisible()


def test_column_picker_reports_unreadable_duckdb_without_the_driver(
        qtbot, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "duckdb", None)
    missing = tmp_path / "scores.duckdb"
    field = _CsvColumnField(key="dependent_variable",
                            paths=[(str(missing), "scores")])
    qtbot.addWidget(field)
    reports = []
    field.set_reporter(reports.append)
    field.set_chooser(lambda *args: pytest.fail("offered an empty list"))
    assert field.pick() is None
    assert len(reports) == 1
    assert reports[0].startswith("Could not read columns from")
    assert str(missing) in reports[0]
    assert not missing.exists()


def test_column_picker_reports_postgres_errors_without_the_driver(
        qtbot, monkeypatch):
    monkeypatch.setitem(sys.modules, "psycopg", None)

    def refused(path, table=None):
        raise ValueError("server refused")

    monkeypatch.setattr(tabular, "table_columns", refused)
    field = _CsvColumnField(key="dependent_variable",
                            paths=lambda: [(PG, "scores")])
    qtbot.addWidget(field)
    reports = []
    field.set_reporter(reports.append)
    assert field.pick() is None
    assert reports == [f"Could not read columns from {PG}: server refused"]


def test_column_picker_offers_each_column_of_two_tables_once(
        qtbot, monkeypatch):
    calls = []

    def columns(path, table=None):
        calls.append(table)
        return ["phenotype", "area"]

    monkeypatch.setattr(tabular, "table_columns", columns)
    field = _CsvColumnField(key="dependent_variable", default="area",
                            paths=[(PG, "a"), (PG, "b")])
    qtbot.addWidget(field)
    offered = []
    field.set_chooser(lambda choices, current: offered.append(
        (list(choices), current)) or "phenotype")
    assert field.pick() == "phenotype"
    assert calls == ["a", "b"]
    assert offered == [(["phenotype", "area"], "area")]
    assert field.get_value() == "phenotype"


def test_column_picker_without_inputs_explains_instead_of_choosing(qtbot):
    field = _CsvColumnField(key="dependent_variable", default="phenotype",
                            paths=lambda: [])
    qtbot.addWidget(field)
    reports = []
    field.set_reporter(reports.append)
    field.set_chooser(lambda *args: pytest.fail("offered an empty list"))
    assert field.pick() is None
    assert len(reports) == 1 and reports[0]
    assert "Could not read columns" not in reports[0]


def test_column_prompt_names_tables_and_suggests_a_close_column(qtbot):
    from spacr import columns

    field = _CsvColumnField(key="dependent_variable",
                            paths=[(PG, "scores")])
    qtbot.addWidget(field)
    choices = ["phenotype", "cell_area"]
    assert field._prompt(columns, choices, "phenotyp") == (
        "No column 'phenotyp' in the input tables. "
        "Did you mean 'phenotype'?")
    assert field._prompt(columns, choices, "zzzzqqqq") == (
        "No column 'zzzzqqqq' in the input tables.")
    assert field._prompt(columns, choices, "phenotype") == (
        "2 column(s) in the input tables:")


def test_input_sources_drop_a_repeated_path_and_table():
    model = SettingsWidgets("regression")
    model._widgets = {}
    model._defaults = {"paired_data": [
        {"score": "s.duckdb", "score_table": "t"},
        {"score": "s.duckdb", "score_table": "t"},
        {"score": "s.duckdb", "score_table": "u"},
        {"score": "plain.csv"},
        {"score": "plain.csv"},
    ]}
    assert model._input_csv_paths(("score",)) == [
        ("s.duckdb", "t"), ("s.duckdb", "u"), "plain.csv"]


def test_plate_context_reads_plate_ids_from_plain_csv_paths(tmp_path):
    first = tmp_path / "scores.csv"
    pd.DataFrame({"plateID": ["P1", "P1", ""],
                  "value": [1, 2, 3]}).to_csv(first, index=False)
    assert SettingsWidgets._plate_context([str(first), ""]) == {
        "plate_count": 1, "has_plate_id": True}


def test_plate_context_skips_csv_inputs_too_large_to_sniff(tmp_path):
    big = tmp_path / "big.csv"
    with open(big, "wb") as handle:
        handle.write(b"plateID,value\nP1,1\n")
        handle.truncate(5_000_001)
    assert SettingsWidgets._plate_context([str(big)]) == {
        "plate_count": None, "has_plate_id": None}
