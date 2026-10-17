"""Item 706: the 3D switch runs Otsu and Cellpose on the whole open volume."""
import threading

import numpy as np
import pytest
import tifffile
from scipy import ndimage

from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks
from spacr.qt.screens.make_masks import MakeMasksScreen


def _blobs(shape=(9, 24, 24)):
    """Two bright spheres spanning several slices, and a diagonal voxel chain."""
    z, y, x = np.indices(shape)
    volume = np.full(shape, 10, dtype=np.uint16)
    volume[(z - 4) ** 2 + (y - 6) ** 2 + (x - 6) ** 2 <= 9] = 200
    volume[(z - 4) ** 2 + (y - 17) ** 2 + (x - 17) ** 2 <= 9] = 200
    for step in range(5):
        volume[2 + step, 4 + step, 18 - step] = 200
    return volume


def test_threshold_labels_a_volume_with_26_connectivity():
    volume = _blobs()
    labels = engine._otsu_instances(volume, min_area=0, smoothing=0.0, fill_holes=False,
                                    split_touching=False, correction=1.0)
    assert labels.shape == volume.shape
    assert int(labels.max()) == 3
    chain = {int(labels[2 + step, 4 + step, 18 - step]) for step in range(5)}
    assert len(chain) == 1 and 0 not in chain
    sphere = labels == labels[4, 6, 6]
    assert np.flatnonzero(sphere.any(axis=(1, 2))).size > 1
    planewise = sum(ndimage.label(plane > 100)[1] for plane in volume)
    assert planewise > 3


def test_connected_instances_keeps_2d_numbering_and_min_area():
    binary = np.zeros((6, 6), bool)
    binary[0, 0] = binary[1, 1] = True
    binary[4:6, 4:6] = True
    binary[0, 5] = True
    labels = engine.connected_instances(binary, min_area=2)
    assert labels[0, 0] == labels[1, 1] == 1 and labels[4, 4] == 2 and labels[0, 5] == 0


def test_replace_labels_is_one_undo_step():
    session = engine.VolumeMaskSession(np.zeros((3, 4, 5), np.uint16),
                                       np.zeros((3, 4, 5), np.uint16))
    labels = np.zeros((3, 4, 5), np.uint16)
    labels[0:2, 1:3, 1:3] = 4
    assert session.replace_labels(labels) == 8
    session.undo()
    assert not session.mask.any()
    with pytest.raises(ValueError):
        session.replace_labels(np.zeros((2, 4, 5), np.uint16))


@pytest.fixture
def screen(qtbot, tmp_path):
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    assert not widget._detect_3d.isEnabled()
    path = tmp_path / "volume.tif"
    tifffile.imwrite(path, _blobs(), metadata={"axes": "ZYX"}, photometric="minisblack")
    assert widget.open_volume(str(path)) is widget._mask_stage
    yield widget
    widget._mask_stage.dirty = False
    widget.close()


def _rows(box):
    return {box.itemData(index): box.model().item(index).isEnabled()
            for index in range(box.count())}


def test_switch_gates_2d_only_methods_and_restores_them(screen):
    from spacr.qt import cpu_modes

    assert screen._detect_3d.isEnabled()
    screen._cp_model.addItem("Cellpose-DINO", "cellpose_dino:/models/x.pt")
    before = _rows(screen._mag_mode)
    screen._mag_mode.setCurrentIndex(screen._mag_mode.findData(cpu_modes.PROPAGATE))
    screen._detect_3d.setChecked(True)
    assert screen._mag_mode.currentData() == "otsu"
    rows = _rows(screen._mag_mode)
    assert rows["otsu"] and rows["li"] and rows[cpu_modes.MULTIOTSU] and rows["sauvola"]
    assert not rows[cpu_modes.PROPAGATE] and not rows[cpu_modes.PUNCTA]
    index = screen._mag_mode.findData(cpu_modes.PROPAGATE)
    assert "2D" in screen._mag_mode.itemData(index, make_masks.Qt.ToolTipRole)
    dino = screen._cp_model.findData("cellpose_dino:/models/x.pt")
    assert not screen._cp_model.model().item(dino).isEnabled()
    assert not screen._otsu_local.isEnabled()
    screen._detect_3d.setChecked(False)
    assert _rows(screen._mag_mode) == before
    assert screen._cp_model.model().item(dino).isEnabled()
    assert "2D" not in str(screen._mag_mode.itemData(index, make_masks.Qt.ToolTipRole))


def test_switch_is_off_and_disabled_without_a_volume(screen):
    screen._detect_3d.setChecked(True)
    screen._leave_volume()
    assert not screen._detect_3d.isEnabled()
    assert not screen._detect_3d.isChecked()
    assert not screen._detect_3d_on()
    assert "3D image" in screen._detect_3d.toolTip()


def test_3d_otsu_off_the_gui_thread_with_undo(screen, qtbot, monkeypatch):
    threads = []
    original = make_masks._detect_volume_snapshot

    def spy(request, models):
        threads.append(threading.get_ident())
        return original(request, models)

    monkeypatch.setattr(make_masks, "_detect_volume_snapshot", spy)
    screen._mag_mode.setCurrentIndex(screen._mag_mode.findData("otsu"))
    screen._min_area.setValue(0)
    screen._detect_3d.setChecked(True)
    session = screen._mask_stage.session
    screen._btn_otsu.click()
    qtbot.waitUntil(lambda: screen._detection_request is None, timeout=20000)
    assert threads and threads[0] != threading.get_ident()
    labels = session.mask
    assert int(np.count_nonzero(np.unique(labels))) >= 2
    sphere = labels == labels[4, 6, 6]
    assert labels[4, 6, 6] and np.flatnonzero(sphere.any(axis=(1, 2))).size > 1
    assert labels[4, 6, 6] != labels[4, 17, 17]
    assert screen._mask_stage.dirty
    screen._on_undo()
    assert not session.mask.any()
    screen._on_redo()
    assert session.mask[4, 6, 6]
    screen._mask_stage.dirty = False


class _FakeCellpose:
    def __init__(self):
        self.calls = []

    def eval(self, x, **kwargs):
        self.calls.append((np.asarray(x).shape, kwargs))
        labels = np.zeros(np.asarray(x).shape, np.int32)
        labels[2:7, 3:9, 3:9] = 1
        return labels, None, None


def test_cellpose_runs_do_3d_and_stitching(screen, qtbot):
    fake = _FakeCellpose()
    screen._cp_loaded["cpsam"] = fake
    index = screen._cp_model.findData("cpsam")
    if index < 0:
        screen._cp_model.addItem("cpsam", "cpsam")
        index = screen._cp_model.count() - 1
    screen._cp_model.setCurrentIndex(index)
    screen._detect_3d.setChecked(True)
    screen._btn_cellpose.click()
    qtbot.waitUntil(lambda: screen._detection_request is None, timeout=10000)
    shape, kwargs = fake.calls[-1]
    assert shape == (9, 24, 24)
    assert kwargs["do_3D"] is True and kwargs["z_axis"] == 0 and kwargs["channel_axis"] is None
    assert "stitch_threshold" not in kwargs
    session = screen._mask_stage.session
    assert int(session.mask.max()) == 1 and session.mask[4, 5, 5] == 1
    screen._combine_mode.setCurrentIndex(screen._combine_mode.findData("replace"))
    screen._cp_stitch.setValue(0.3)
    screen._btn_cellpose.click()
    qtbot.waitUntil(lambda: screen._detection_request is None, timeout=10000)
    shape, kwargs = fake.calls[-1]
    assert kwargs["do_3D"] is False and kwargs["stitch_threshold"] == pytest.approx(0.3)
    screen._mask_stage.dirty = False


def test_cellpose_refuses_a_2d_only_model_in_3d(screen):
    screen._cp_model.addItem("Cellpose-DINO", "cellpose_dino:/models/x.pt")
    screen._cp_model.setCurrentIndex(screen._cp_model.count() - 1)
    screen._detect_3d.setChecked(True)
    screen._btn_cellpose.click()
    assert screen._detection_request is None
    assert "2D fields only" in screen._status_label.text()
    assert make_masks._model_supports_volume("cpsam")
    assert not make_masks._model_supports_volume("cellpose_dino:/models/x.pt")
