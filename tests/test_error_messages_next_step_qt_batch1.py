"""Failed Qt operations retain the error and explain how to try again."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QFileDialog, QMessageBox

pytestmark = pytest.mark.qt


@pytest.fixture
def warning(monkeypatch):
    """Capture warnings in English without opening a modal dialog."""
    from spacr.qt import i18n

    captured = Mock()
    monkeypatch.setattr(QMessageBox, "warning", captured)
    monkeypatch.setattr(i18n, "current_language", Mock(return_value="en"))
    return captured


def test_invalid_template_explains_accepted_inputs(tmp_path, monkeypatch, warning):
    """A non-template JSON is refused before saving or refreshing templates."""
    from spacr.qt import recipes

    path = tmp_path / "unrelated.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", Mock(return_value=(str(path), "")))
    save = Mock()
    monkeypatch.setattr(recipes, "save_recipe", save)
    dialog = SimpleNamespace(_app_key="mask", reload=Mock())

    assert recipes.RecipeDialog._on_import(dialog) is None
    message = warning.call_args.args[2]
    assert "not a spaCR settings template" in message
    assert "template JSON or settings CSV for this module" in message
    save.assert_not_called()
    dialog.reload.assert_not_called()


def test_settings_import_keeps_error_and_explains_file_choice(monkeypatch, warning):
    """A failed settings read shows recovery advice without applying settings."""
    from spacr.qt.screens.app_screen import AppScreen

    monkeypatch.setattr(QFileDialog, "getOpenFileName", Mock(return_value=("settings.csv", "")))
    screen = SimpleNamespace(
        _load_settings_csv=Mock(side_effect=ValueError("missing Key column")),
        _in_its_own_era=Mock(), apply_settings_dict=Mock())

    assert AppScreen._on_import_settings(screen) is None
    message = warning.call_args.args[2]
    assert "missing Key column" in message
    assert "readable settings CSV exported from spaCR for this module" in message
    screen.apply_settings_dict.assert_not_called()


def test_queue_import_explains_plate_list_without_adding_items(monkeypatch, warning):
    """A failed queue import identifies the required src column and adds no plate."""
    from spacr.qt.screens import queue

    monkeypatch.setattr(QFileDialog, "getOpenFileName", Mock(return_value=("plates.csv", "")))
    monkeypatch.setattr(queue, "import_plates_from_csv", Mock(side_effect=OSError("read denied")))
    screen = SimpleNamespace(_queue=SimpleNamespace(add=Mock()), _refresh_table=Mock())

    assert queue.QueueScreen._on_import(screen) is None
    message = warning.call_args.args[2]
    assert "read denied" in message and "plates.csv" in message
    assert "src column containing the input folder for each plate" in message
    screen._queue.add.assert_not_called()
    screen._refresh_table.assert_not_called()


def test_picture_write_failure_explains_destination(tmp_path, monkeypatch, warning):
    """A rejected picture write keeps the selected suffix and returns no path."""
    from spacr.qt.widgets import picture_export

    path = tmp_path / "figure.png"
    monkeypatch.setattr(picture_export, "ask_where_to_save", Mock(return_value=str(path)))
    save = Mock(return_value=False)
    monkeypatch.setattr(picture_export, "save_picture", save)
    picture = object()

    assert picture_export.save_as(None, picture, "figure", ".pdf") == ""
    save.assert_called_once_with(picture, str(path.with_suffix(".pdf")))
    message = warning.call_args.args[2]
    assert "figure.pdf could not be written" in message
    assert "existing folder you can write to" in message
    assert "free disk space" in message


def test_model_download_failure_keeps_details_and_reports_persistent_mismatch(warning):
    """A checksum error retains the status update and explains when to report it."""
    from spacr.qt.widgets.model_zoo_picker import ModelZooPicker

    dialog = SimpleNamespace(_finish_download=Mock())
    detail = "checksum mismatch for model-A"

    assert ModelZooPicker._on_download_failed(dialog, detail) is None
    message = warning.call_args.args[2]
    assert detail in message
    assert "retry the download" in message
    assert "report the model name and this error" in message
    dialog._finish_download.assert_called_once_with(f"Download failed: {detail}")


def test_missing_style_file_explains_save_graph_style(tmp_path, monkeypatch, warning):
    """An unreadable style file leaves the existing preference controls alone."""
    from spacr.qt.widgets.figure_settings import FigureStylePreferences

    path = tmp_path / "missing-style.json"
    monkeypatch.setattr(QFileDialog, "getOpenFileName", Mock(return_value=(str(path), "")))
    panel = SimpleNamespace(apply_values=Mock())

    assert FigureStylePreferences._load_from_file(panel) is None
    message = warning.call_args.args[2]
    assert path.name in message
    assert "readable JSON file created with Save graph style" in message
    panel.apply_values.assert_not_called()
