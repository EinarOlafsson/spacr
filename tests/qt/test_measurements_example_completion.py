"""Shared example loading restores controls and reports every ending."""
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from spacr import example_archives
from spacr.qt import hf_download
from spacr.qt.widgets import measurements_example as example


@pytest.fixture
def plate(tmp_path, monkeypatch):
    folder = tmp_path / "plate1"
    monkeypatch.setattr(example_archives, "example_plate_folder", lambda: folder)
    monkeypatch.setattr(hf_download, "download_annotate_example", Mock(
        side_effect=AssertionError("attempted a real download launcher")))
    return folder


def _cached_database(folder):
    database = folder / "measurements" / "measurements.db"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE cell (object_id INTEGER, cell_area REAL)")
        connection.execute("INSERT INTO cell VALUES (1, 25.0)")
    return database


@pytest.mark.parametrize("index", [None, 0])
def test_clicked_button_uses_the_shared_downloader_and_restores_after_success(
        qtbot, monkeypatch, plate, index):
    screen = QWidget()
    qtbot.addWidget(screen)
    layout = QHBoxLayout(screen)
    existing = QPushButton("Existing", screen)
    layout.addWidget(existing)
    applied, messages, pending = Mock(), [], []

    def download(parent, destination, on_done):
        assert parent is screen and destination == plate
        pending.append(on_done)

    monkeypatch.setattr(hf_download, "download_annotate_example", download)
    button = example.install_test_data_button(
        screen, layout, applied, say=messages.append, index=index)
    assert layout.itemAt(1 if index is None else 0).widget() is button
    assert button.objectName() == example.BUTTON_OBJECT_NAME
    button.click()
    assert not button.isEnabled()
    assert "Fetching" in button.text()
    assert plate.is_dir() and len(pending) == 1
    assert not applied.called
    database = _cached_database(plate)
    pending[0](plate, None)
    assert button.isEnabled() and button.text() == "Load test data…"
    applied.assert_called_once_with(plate, database)
    assert messages == []
    button.click()  # A second click must use the database already on disk.
    assert len(pending) == 1 and applied.call_count == 2


def test_cached_plate_can_be_returned_without_screen_controls(plate):
    database = _cached_database(plate)
    assert example.example_measurements_folder() == plate
    assert example.example_measurements_db() == database
    result = example.load_test_data(SimpleNamespace())
    assert result == {"folder": str(plate), "db": str(database)}
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT * FROM cell").fetchall() == [(1, 25.0)]


@pytest.mark.parametrize("message,expected", [("unsupported schema", "unsupported schema"), ("", "ValueError")])
def test_a_screen_that_cannot_open_the_cached_plate_reports_the_reason(
        plate, caplog, message, expected):
    database = _cached_database(plate)
    reports = []
    screen = SimpleNamespace(_test_data_apply=Mock(side_effect=ValueError(message)),
                             _test_data_say=reports.append)
    assert example.load_test_data(screen) == {}
    assert reports == [expected]
    assert "screen could not open" in caplog.text
    assert database.is_file()


@pytest.mark.parametrize("reporter_fails", [False, True])
def test_download_failure_without_controls_falls_back_to_the_log(
        plate, caplog, reporter_fails):
    reporter = Mock(side_effect=RuntimeError("closed status label")) if reporter_fails else None
    screen = SimpleNamespace(_test_data_say=reporter)
    pending = []
    result = example.load_test_data(screen, ask=lambda *args: pending.append(args[-1]))
    assert result == {} and len(pending) == 1
    with caplog.at_level("DEBUG", logger=example.__name__):
        pending[0](None, None)
    assert result == {}
    assert "could not be downloaded" in caplog.text and "unknown error" in caplog.text
    if reporter_fails:
        reporter.assert_called_once()
        assert "reporter failed" in caplog.text


def test_download_result_without_database_restores_button_and_does_not_apply(
        qtbot, plate):
    screen = QWidget()
    qtbot.addWidget(screen)
    applied, reports = Mock(), []
    button = example.install_test_data_button(screen, None, applied, say=reports.append)
    pending = []
    assert example.load_test_data(screen, ask=lambda *args: pending.append(args[-1])) == {}
    assert not button.isEnabled()
    pending[0](plate, "archive has no database")
    assert button.isEnabled() and button.text() == "Load test data…"
    applied.assert_not_called()
    assert reports == ["The test data could not be downloaded: archive has no database"]


def test_synchronous_download_returns_the_actual_handover(plate):
    applied = Mock()
    screen = SimpleNamespace(_test_data_apply=applied)

    def download(parent, folder, on_done):
        _cached_database(folder)
        on_done(folder, None)

    result = example.load_test_data(screen, ask=download)
    assert result == {"folder": str(plate), "db": str(plate / "measurements/measurements.db")}
    applied.assert_called_once_with(plate, plate / "measurements/measurements.db")
