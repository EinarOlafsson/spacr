"""The measurement backend on Measure is an alpha feature.

Everything built from the future-features list is hidden until Preferences ->
Show alpha features is turned on. For the measurement backend that is its two
settings on the Measure form, under their own "Measurement Backend α"
heading; a value saved while hidden still reaches the run.
"""
from __future__ import annotations

import os
import hashlib

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings                              # noqa: E402

KEY = "measurement_backend"
KEYS = (KEY, "measurement_backend_target")


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    """Preferences read and written through a throwaway INI file."""
    from spacr.qt import preferences

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


def test_the_setting_is_registered_under_its_item():
    from spacr.settings import ALPHA_FEATURES, _is_alpha, categories

    assert ALPHA_FEATURES[576] == {
        "settings": KEYS,
        "widgets": ("ALPHA_FEATURES576", "MeasurementMigrationAction")}
    assert tuple(categories["Measurement Backend α"]) == KEYS
    assert all(_is_alpha("settings", key) for key in KEYS)


def test_the_gate_hides_it_until_alpha_features_are_on(prefs):
    assert prefs._is_alpha_visible("settings", KEY) is False
    prefs._set_show_alpha_features(True)
    assert prefs._is_alpha_visible("settings", KEY) is True


def test_the_measure_row_follows_the_switch(qtbot, prefs):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus

    screen = AppScreen("measure")
    try:
        screen._open_the_heading_of(KEY)
        screen._refresh_alpha_visibility()
        assert not screen.setting_row_is_visible(KEY)
        assert screen._settings_model.set_value_for_key(KEY, "duckdb")
        assert screen._settings_model.collect()[KEY] == "duckdb"

        prefs._set_show_alpha_features(True)
        screen._refresh_alpha_visibility()
        assert screen.setting_row_is_visible(KEY)

        prefs._set_show_alpha_features(False)
        screen._refresh_alpha_visibility()
        assert not screen.setting_row_is_visible(KEY)
        assert screen._settings_model.collect()[KEY] == "duckdb"
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()


def test_migration_action_is_alpha_and_opens_an_explicit_copy_dialog(
        qtbot, prefs):
    from PySide6.QtWidgets import QWidget
    from spacr.qt.screens.settings_model import SettingsWidgets

    owner = QWidget()
    qtbot.addWidget(owner)
    model = SettingsWidgets("measure", parent=owner)
    model.build_sections()
    edit = model._widgets.get("measurement_backend_target")
    assert edit is not None
    action = next(item for item in edit.actions()
                  if item.objectName() == "MeasurementMigrationAction")
    assert not action.isVisible()
    prefs._set_show_alpha_features(True)
    prefs._apply_alpha_widgets(owner)
    assert action.isVisible()
    action.trigger()
    from spacr.qt.screens.settings_model import _LIVE_MEASUREMENT_MIGRATIONS
    assert len(_LIVE_MEASUREMENT_MIGRATIONS) == 1
    dialog = next(iter(_LIVE_MEASUREMENT_MIGRATIONS))
    assert dialog.objectName() == "MeasurementMigrationDialog"
    assert dialog.source.text() == ""
    dialog.close()
    assert not _LIVE_MEASUREMENT_MIGRATIONS


def test_migration_copies_to_a_new_store_without_changing_the_source(tmp_path):
    import pandas as pd
    from spacr import tabular
    from spacr.qt.screens.settings_model import _copy_measurement_store

    source, target = tmp_path / "measurements.db", tmp_path / "copy.db"
    frame = pd.DataFrame({"plate": ["P1", "P2"], "cell_area": [7.5, 9.5]})
    tabular.write_database(frame, source, "cell", if_exists="replace",
                           canonicalise=False)
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    progress = []
    assert _copy_measurement_store(str(source), str(target), progress.append) == ("cell",)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original
    assert tabular.database_tables(target) == ("cell",)
    pd.testing.assert_frame_equal(
        tabular.read_database(target, "cell", canonicalise=False,
                              migrate=False, report=None)[0], frame,
        check_dtype=False)
    assert len(progress) == 1 and "Copied cell: 2 row(s)" in progress[0]
    assert not list(tmp_path.glob(".spacr-migration-*"))


def test_migration_dialog_keeps_qt_responsive_and_reports_completion(
        qtbot, tmp_path, monkeypatch):
    import pandas as pd
    from PySide6.QtWidgets import QMessageBox
    from spacr import tabular
    from spacr.qt.screens.settings_model import _MeasurementMigrationDialog

    source, target = tmp_path / "source.db", tmp_path / "copied.db"
    tabular.write_database(pd.DataFrame({"id": [1, 2]}), source, "cell",
                           if_exists="replace", canonicalise=False)
    monkeypatch.setattr(QMessageBox, "question", lambda *_args: QMessageBox.Yes)
    dialog = _MeasurementMigrationDialog(str(target))
    qtbot.addWidget(dialog)
    dialog.source.setText(str(source))
    dialog.show()
    dialog.start_button.click()
    qtbot.waitUntil(lambda: "Copied 1 table(s)" in dialog.log.toPlainText(),
                    timeout=10000)
    assert dialog.close_button.isEnabled()
    assert tabular.database_tables(target) == ("cell",)
    dialog.close()


def test_migration_dialog_cannot_discard_a_running_copy_receiver(
        qtbot, tmp_path, monkeypatch):
    import threading
    import pandas as pd
    from PySide6.QtWidgets import QMessageBox
    from spacr import tabular
    from spacr.qt.screens import settings_model

    source, target = tmp_path / "source.db", tmp_path / "target.db"
    tabular.write_database(pd.DataFrame({"id": [1]}), source, "cell",
                           if_exists="replace", canonicalise=False)
    entered, release = threading.Event(), threading.Event()

    def waiting_copy(_source, _target, _report):
        entered.set()
        assert release.wait(3)
        return ("cell",)

    monkeypatch.setattr(settings_model, "_copy_measurement_store", waiting_copy)
    monkeypatch.setattr(QMessageBox, "question", lambda *_args: QMessageBox.Yes)
    dialog = settings_model._MeasurementMigrationDialog(str(target))
    qtbot.addWidget(dialog)
    dialog.source.setText(str(source))
    dialog.show()
    dialog.start_button.click()
    assert entered.wait(3)
    dialog.close()
    assert dialog.isVisible()
    assert not dialog.close_button.isEnabled()
    release.set()
    qtbot.waitUntil(lambda: dialog.close_button.isEnabled(), timeout=5000)
    assert "Copied 1 table(s)" in dialog.log.toPlainText()
    dialog.close()


def test_migration_supports_a_parquet_destination_and_the_reverse_copy(tmp_path):
    pytest.importorskip("pyarrow")
    import pandas as pd
    from spacr import tabular
    from spacr.qt.screens.settings_model import _copy_measurement_store

    source = tmp_path / "source.db"
    parquet = tmp_path / "columns.parquetdb"
    restored = tmp_path / "restored.db"
    frame = pd.DataFrame({"plate": ["P1", "P2"], "area": [5.5, 7.5]})
    tabular.write_database(frame, source, "cell", if_exists="replace",
                           canonicalise=False)
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    assert _copy_measurement_store(str(source), str(parquet), None) == ("cell",)
    assert _copy_measurement_store(str(parquet), str(restored), None) == ("cell",)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original
    pd.testing.assert_frame_equal(
        tabular.read_database(restored, "cell", canonicalise=False,
                              migrate=False, report=None)[0], frame,
        check_dtype=False)


def test_migration_supports_a_duckdb_destination_and_the_reverse_copy(tmp_path):
    pytest.importorskip("duckdb")
    import pandas as pd
    from spacr import tabular
    from spacr.qt.screens.settings_model import _copy_measurement_store

    source, duck, restored = (tmp_path / "source.db", tmp_path / "copy.duckdb",
                              tmp_path / "restored.db")
    frame = pd.DataFrame({"plate": ["P1", "P2"], "area": [5.5, 7.5]})
    tabular.write_database(frame, source, "cell", if_exists="replace",
                           canonicalise=False)
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    assert _copy_measurement_store(str(source), str(duck), None) == ("cell",)
    assert _copy_measurement_store(str(duck), str(restored), None) == ("cell",)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original
    pd.testing.assert_frame_equal(
        tabular.read_database(restored, "cell", canonicalise=False,
                              migrate=False, report=None)[0], frame,
        check_dtype=False)


def test_migration_refuses_missing_same_or_existing_store_and_cleans_failed_stage(
        tmp_path, monkeypatch):
    import pandas as pd
    from spacr import tabular
    from spacr.qt.screens import settings_model

    source, target = tmp_path / "source.db", tmp_path / "target.db"
    tabular.write_database(pd.DataFrame({"id": [1]}), source, "cell",
                           if_exists="replace", canonicalise=False)
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    for bad_source, bad_target in (("", str(target)),
                                   (str(tmp_path / "missing.db"), str(target)),
                                   (str(source), str(source)),
                                   (str(source), str(tmp_path / "target.csv"))):
        with pytest.raises(ValueError):
            settings_model._measurement_migration_inputs(bad_source,
                                                         bad_target)
    target.write_bytes(b"existing target")
    with pytest.raises(ValueError, match="already exists"):
        settings_model._copy_measurement_store(str(source), str(target), None)
    target.unlink()

    def partial_copy(_source, staged, *, tables, report):
        tabular.write_database(pd.DataFrame({"id": [2]}), staged, "cell",
                               if_exists="replace", canonicalise=False)
        raise RuntimeError("interrupted copy")

    monkeypatch.setattr(tabular, "_migrate_database", partial_copy)
    with pytest.raises(RuntimeError, match="interrupted"):
        settings_model._copy_measurement_store(str(source), str(target), None)
    assert not target.exists()
    assert not list(tmp_path.glob(".spacr-migration-*"))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original

    def concurrent_destination(_source, staged, *, tables, report):
        tabular.write_database(pd.DataFrame({"id": [2]}), staged, "cell",
                               if_exists="replace", canonicalise=False)
        target.write_bytes(b"another writer's target")
        return ("cell",)

    monkeypatch.setattr(tabular, "_migrate_database", concurrent_destination)
    with pytest.raises(ValueError, match="appeared during the copy"):
        settings_model._copy_measurement_store(str(source), str(target), None)
    assert target.read_bytes() == b"another writer's target"
    assert not list(tmp_path.glob(".spacr-migration-*"))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original


def test_parquet_publication_keeps_a_concurrent_destination(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    import pandas as pd
    from spacr import tabular
    from spacr.qt.screens.settings_model import _copy_measurement_store

    source, target = tmp_path / "source.db", tmp_path / "target.parquetdb"
    tabular.write_database(pd.DataFrame({"id": [1]}), source, "cell",
                           if_exists="replace", canonicalise=False)
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    native_mkdir = os.mkdir

    def competing_mkdir(path, *args, **kwargs):
        if os.fspath(path) == str(target):
            native_mkdir(path, *args, **kwargs)
            (target / "another-writer.txt").write_text("keep this")
            raise FileExistsError(path)
        return native_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(os, "mkdir", competing_mkdir)
    with pytest.raises(FileExistsError):
        _copy_measurement_store(str(source), str(target), None)
    assert (target / "another-writer.txt").read_text() == "keep this"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original
    assert not list(tmp_path.glob(".spacr-migration-*"))


def test_parquet_migration_refuses_a_table_that_escapes_its_folder(tmp_path):
    import sqlite3
    from spacr.qt.screens.settings_model import _copy_measurement_store

    source, target = tmp_path / "source.db", tmp_path / "target.parquetdb"
    with sqlite3.connect(source) as connection:
        connection.execute('CREATE TABLE "../outside" (id INTEGER)')
        connection.execute('INSERT INTO "../outside" VALUES (1)')
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Parquet folder"):
        _copy_measurement_store(str(source), str(target), None)
    assert not target.exists()
    assert not (tmp_path / "outside").exists()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original
