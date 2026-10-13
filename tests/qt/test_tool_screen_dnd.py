"""Drag/drop installation on standalone Tool and Results screens."""
from __future__ import annotations

import pytest


@pytest.mark.parametrize("module_name,class_name,handler_name", [
    ("foreign", "ForeignScreen", "ForeignProjectDropHandler"),
    ("align", "AlignScreen", "AlignDropHandler"),
    ("convert", "ConvertScreen", "ConvertDropHandler"),
    ("batch", "BatchScreen", "BatchDropHandler"),
    ("model_compare", "ModelCompareScreen", "ImageFieldsDropHandler"),
    ("model_zoo", "ModelZooScreen", "ModelZooDropHandler"),
    ("plate_view", "PlateViewScreen", "ResultsDatabaseDropHandler"),
    ("agreement", "AgreementScreen", "ResultsDatabaseDropHandler"),
    ("train_compare", "TrainCompareScreen", "TrainingRunsDropHandler"),
    ("report", "ReportScreen", "ReportDropHandler"),
])
def test_standalone_screen_installs_its_dropzone(
        module_name, class_name, handler_name, qtbot, qt_theme_applied):
    module = __import__(
        f"spacr.qt.screens.{module_name}", fromlist=[class_name])
    cls = getattr(module, class_name)
    screen = cls(threaded=False)
    qtbot.addWidget(screen)
    assert screen.acceptDrops()
    assert type(screen._dnd_handler).__name__ == handler_name


def test_plate_queue_installs_plate_settings_dropzone(
        tmp_path, qtbot, qt_theme_applied):
    from spacr.qt.plate_queue import PlateQueue
    from spacr.qt.screens.queue import QueueScreen

    screen = QueueScreen(queue=PlateQueue(path=tmp_path / "queue.json"))
    qtbot.addWidget(screen)
    assert screen.acceptDrops()
    assert type(screen._dnd_handler).__name__ == "PlateQueueDropHandler"
