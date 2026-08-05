"""Real drop-event coverage for the standalone modules requested by users.

These tests drive Qt's enter -> move -> drop event sequence. They deliberately
exercise the handler through :func:`install_dropzone`, rather than calling
``handler.apply`` directly, so a screen that merely *has* a policy but cannot
receive an OS drop is caught.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QMimeData, QPoint, QPointF, QUrl, Qt
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import QApplication, QWidget

from spacr.qt import dnd as dnd_module
from spacr.qt.dnd import install_dropzone
from spacr.qt.dnd_handlers import get_handler


def _drop(widget: QWidget, paths: list[Path]) -> QDropEvent:
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path)) for path in paths])
    QApplication.sendEvent(widget, QDragEnterEvent(
        QPoint(4, 4), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
    QApplication.sendEvent(widget, QDragMoveEvent(
        QPoint(4, 4), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
    event = QDropEvent(
        QPointF(4, 4), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(widget, event)
    return event


class _Owner(QWidget):
    """One recording owner exposing the small APIs used by tool handlers."""

    last_error = ""

    def __init__(self):
        super().__init__()
        self.calls = []

    def set_images(self, path):
        self.calls.append(("set_images", path))

    def apply_settings(self, settings):
        self.calls.append(("apply_settings", dict(settings)))

    def set_source(self, path):
        self.calls.append(("set_source", path))
        return True

    def scan(self, path=None):
        self.calls.append(("scan", path))
        return True

    def set_fields_source(self, path):
        self.calls.append(("set_fields_source", path))
        return True

    def set_database(self, path):
        self.calls.append(("set_database", path))
        return True

    def add_job(self, module, settings):
        self.calls.append(("add_job", module, settings))
        return True

    def add_item(self, module, settings):
        self.calls.append(("add_item", module, settings))


def _zone(qtbot, key):
    owner = _Owner()
    owner.app_key = key
    qtbot.addWidget(owner)
    install_dropzone(owner, get_handler(key), owner)
    owner.show()
    return owner


def test_requested_tool_modules_receive_real_drop_events(
        tmp_path, qtbot, qt_theme_applied):
    images = tmp_path / "images"
    images.mkdir()
    (images / "tile.tif").write_bytes(b"II*\x00")
    container = tmp_path / "plate.nd2"
    container.write_bytes(b"nd2")
    models = tmp_path / "models"
    models.mkdir()
    (models / "demo.CP_model").write_bytes(b"model")
    database = tmp_path / "measurements.db"
    database.write_bytes(b"SQLite format 3\x00")
    runs = tmp_path / "runs"
    runs.mkdir()
    report = tmp_path / "report"
    report.mkdir()
    settings_csv = tmp_path / "mask_settings.csv"
    settings_csv.write_text("Key,Value\nsrc,/plate\n")

    cases = [
        ("foreign", images, [("set_images", str(images))]),
        ("align", images, [("apply_settings", {"src": str(images)})]),
        ("convert", container, [("set_source", str(tmp_path))]),
        ("model_compare", images, [("set_source", str(images))]),
        ("model_zoo", models, [("scan", str(models))]),
        ("plate_view", database, [("set_database", str(database))]),
        ("agreement", database, [("set_database", str(database))]),
        ("train_compare", runs, [("scan", str(runs))]),
        ("report", report,
         [("set_source", str(report)), ("scan", None)]),
    ]
    for key, path, expected in cases:
        owner = _zone(qtbot, key)
        event = _drop(owner, [path])
        assert event.isAccepted(), key
        assert owner.calls == expected, (key, owner.calls)

    batch = _zone(qtbot, "batch")
    event = _drop(batch, [settings_csv])
    assert event.isAccepted()
    assert len(batch.calls) == 1
    assert batch.calls[0][:2] == ("add_job", "mask")


def test_plate_queue_drop_adds_multiple_plates_with_settings_at_once(
        tmp_path, qtbot, qt_theme_applied):
    plates = []
    for index in range(3):
        plate = tmp_path / f"plate_{index}"
        settings = plate / "settings"
        settings.mkdir(parents=True)
        (settings / "mask_settings.csv").write_text(
            "Key,Value\ncell_channel,2\n")
        plates.append(plate)

    owner = _zone(qtbot, "queue")
    event = _drop(owner, plates)

    assert event.isAccepted()
    added = [call for call in owner.calls if call[0] == "add_item"]
    assert len(added) == len(plates)
    assert [call[2]["src"] for call in added] == list(map(str, plates))


def test_format_converter_rejection_prints_reason_and_action_to_console(
        tmp_path, qtbot, qt_theme_applied, monkeypatch):
    class Console:
        def __init__(self):
            self.text = ""

        def append_error(self, text):
            self.text += text

        def _current_provider(self):
            return None

    class MessageBox:
        @staticmethod
        def information(*_args):
            return None

    bad = tmp_path / "notes.txt"
    bad.write_text("not microscopy")
    owner = _zone(qtbot, "convert")
    owner._console = Console()
    monkeypatch.setattr(dnd_module, "QMessageBox", MessageBox)

    event = _drop(owner, [bad])

    assert event.isAccepted()
    assert "[drop rejected]" in owner._console.text
    assert "Reason: Format Converter accepts" in owner._console.text
    assert "Suggestion:" in owner._console.text
