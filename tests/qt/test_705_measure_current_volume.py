"""Current edited labels and additional image sets reach the 3D measurement worker."""
from pathlib import Path
import threading

import numpy as np
import pytest
import tifffile
from PySide6.QtCore import QMimeData, QPointF, Qt, QTimer, QUrl
from PySide6.QtGui import QDropEvent
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QLabel

from spacr.qt.screens.make_masks import MakeMasksScreen
from spacr.qt.widgets.organize_for_measure import MeasureVolumeDialog


def _volume(path):
    tifffile.imwrite(path, np.arange(5 * 12 * 14, dtype=np.uint16).reshape(5, 12, 14),
                     metadata={"axes": "ZYX"}, photometric="minisblack")
    return str(path)


def test_dialog_accepts_dropped_channels_and_pasted_image_sets(qtbot, tmp_path):
    current = _volume(tmp_path / "current.tif")
    channel = _volume(tmp_path / "extra channel.tif")
    added = _volume(tmp_path / "added.tif")
    mask = _volume(tmp_path / "mask.tif")
    dialog = MeasureVolumeDialog(current)
    qtbot.addWidget(dialog)
    assert dialog.table.cellWidget(0, 0).isReadOnly()
    assert dialog.table.cellWidget(0, 1).isReadOnly()
    field = dialog.table.cellWidget(0, 2)
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(channel)])
    drop = QDropEvent(QPointF(2, 2), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    field.dropEvent(drop)
    assert drop.isAccepted()
    dialog.add_row_button.click()
    QApplication.clipboard().setText(QUrl.fromLocalFile(added).toString())
    dialog.table.cellWidget(1, 0).paste()
    dialog.table.cellWidget(1, 1).setText(f'"{mask}"')
    dialog.add_channel_button.click()
    assert dialog.table.columnCount() == 5
    dialog.add_row_button.click()
    assert dialog.requests() == [
        {"source_path": current, "mask_path": None, "channel_paths": (channel,)},
        {"source_path": added, "mask_path": mask, "channel_paths": ()},
    ]
    dialog.table.setCurrentCell(1, 0)
    dialog.remove_row_button.click()
    assert len(dialog.requests()) == 1
    dialog.table.setCurrentCell(0, 0)
    dialog.remove_row_button.click()
    assert dialog.table.cellWidget(0, 0).text() == current


@pytest.mark.parametrize("problem", ["missing", "gap", "source"])
def test_dialog_retains_input_when_a_path_is_invalid(qtbot, tmp_path, problem):
    source = _volume(tmp_path / "source.tif")
    dialog = MeasureVolumeDialog(source)
    qtbot.addWidget(dialog)
    if problem == "missing":
        dialog.table.cellWidget(0, 2).setText(str(tmp_path / "missing.tif"))
    elif problem == "gap":
        dialog.table.cellWidget(0, 3).setText(source)
    else:
        dialog.add_row()
        dialog.table.cellWidget(1, 2).setText(source)
    dialog.accept()
    assert dialog.result() != QDialog.DialogCode.Accepted
    assert dialog.error.text()
    assert dialog.table.cellWidget(0, 0).text() == source


@pytest.fixture
def screen(qtbot, tmp_path):
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    assert not widget._act_measure_volume.isEnabled()
    source = _volume(tmp_path / "volume.tif")
    assert widget.open_volume(source) is widget._mask_stage
    widget._mask_stage.session.mask[1:4, 3:7, 4:8] = 7
    yield widget
    qtbot.waitUntil(lambda: not widget._volume_measure_running, timeout=10000)
    widget._mask_stage.dirty = False
    widget.close()


def test_measure_button_routes_volume_and_preserves_the_2d_menu(screen, monkeypatch):
    calls = []
    monkeypatch.setattr(screen, "_on_measure_volume", lambda: calls.append("3d"))
    monkeypatch.setattr(screen, "_open_organize", lambda *a, **kw: calls.append("2d"))
    assert screen._act_measure_volume.isEnabled()
    screen._btn_organize.click()
    screen._measure_menu.actions()[0].trigger()
    assert calls == ["3d", "2d"]


def test_measurement_snapshots_labels_and_keeps_qt_responsive(screen, qtbot, monkeypatch):
    import spacr.measure as measure

    started, release = threading.Event(), threading.Event()
    received = {}
    main_thread = threading.get_ident()

    def run(rows, *, object_type, progress):
        received.update(rows=rows, role=object_type, thread=threading.get_ident())
        started.set()
        assert release.wait(5)
        return {"db_paths": (str(Path(rows[0]["source_path"]).parent / "measurements.db"),),
                "images": 1, "objects": 1}

    monkeypatch.setattr(measure, "measure_volume_rows", run, raising=False)
    session = screen._mask_stage.session
    rows = [{"source_path": session.source_path, "channel_paths": ()}]
    ticks = []
    timer = QTimer(screen)
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(5)
    try:
        assert screen._start_volume_measurement(rows, "nucleus")
        qtbot.waitUntil(started.is_set)
        qtbot.waitUntil(lambda: bool(ticks))
        assert received["thread"] != main_thread
        assert received["role"] == "nucleus"
        snapshot = received["rows"][0]["snapshot"]
        assert not np.shares_memory(snapshot["mask"], session.mask)
        session.mask[:] = 0
        assert np.count_nonzero(snapshot["mask"]) == 48
        assert not screen._act_measure_volume.isEnabled()
        assert not screen._start_volume_measurement(rows)
        assert not screen.close()
    finally:
        release.set()
        timer.stop()
    qtbot.waitUntil(lambda: not screen._volume_measure_running)
    assert screen._act_measure_volume.isEnabled()
    assert "Measured 1 object(s) in 1" in screen._status_label.text()


def test_measurement_failure_is_visible_and_can_be_retried(screen, qtbot, monkeypatch):
    import spacr.measure as measure

    def fail(*args, **kwargs):
        raise ValueError("Channel dimensions do not match")

    monkeypatch.setattr(measure, "measure_volume_rows", fail, raising=False)
    rows = [{"source_path": screen._mask_stage.session.source_path, "channel_paths": ()}]
    assert screen._start_volume_measurement(rows)
    qtbot.waitUntil(lambda: not screen._volume_measure_running)
    assert "Channel dimensions do not match" in screen._status_label.text()
    assert screen._act_measure_volume.isEnabled()
    monkeypatch.setattr(measure, "measure_volume_rows", lambda *a, **kw: {
        "db_paths": (), "images": 1, "objects": 1})
    assert screen._start_volume_measurement(rows)
    qtbot.waitUntil(lambda: not screen._volume_measure_running)
    assert "Measured 1 object(s)" in screen._status_label.text()


def test_measure_popup_writes_current_and_added_images_to_the_real_database(
        screen, qtbot, monkeypatch, tmp_path):
    from spacr import tabular

    channel = _volume(tmp_path / "channel.tif")
    added = _volume(tmp_path / "added.tif")
    labels = np.zeros((5, 12, 14), dtype=np.uint16)
    labels[2:4, 4:9, 5:11] = 9
    mask = tmp_path / "added_labels.tif"
    tifffile.imwrite(mask, labels, metadata={"axes": "ZYX"}, photometric="minisblack")
    saved = mask.read_bytes()

    def fill(dialog):
        calibration = dialog.findChild(QLabel, "VolumeMeasureCalibration")
        assert "ZYX" in calibration.text() and "pixel" in calibration.text()
        dialog.table.cellWidget(0, 2).setText(channel)
        dialog.add_row_button.click()
        dialog.table.cellWidget(1, 0).setText(added)
        dialog.table.cellWidget(1, 1).setText(str(mask))
        dialog.table.cellWidget(1, 2).setText(channel)
        dialog.buttons.button(QDialogButtonBox.Save).click()
        return dialog.result() == QDialog.DialogCode.Accepted

    monkeypatch.setattr(screen, "_run_organize", fill)
    screen._act_measure_volume.trigger()
    qtbot.waitUntil(lambda: not screen._volume_measure_running, timeout=10000)
    db = tmp_path / "measurements.db"
    assert db.is_file(), screen._status_label.text()
    frame = tabular.read_database(db, "cell", canonicalise=False, migrate=False)[0]
    frame = frame.sort_values("object_label")
    assert frame.object_label.tolist() == [7, 9]
    assert frame.cell_volume_voxels.tolist() == [48, 60]
    assert frame.measurement_ndim.tolist() == [3, 3]
    assert frame.measurement_units.tolist() == ["px", "px"]
    assert "cell_channel_0_channel_1_Pearson_correlation" in frame.columns
    assert "Measured 2 object(s) in 2" in screen._status_label.text()
    assert mask.read_bytes() == saved
