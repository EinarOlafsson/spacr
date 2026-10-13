"""N692: drawing 3-D masks slice by slice in Make Masks, and the FIB-SEM menu.

No network and no GPU: volumes are synthetic, and the download is replaced by
one that writes the unpacked set.

2026-10-12: the separate 3D mask editor window is retired. A volume opens in
the Make Masks view itself (tests/qt/test_673_make_masks_3d_in_container.py);
the drawing, side-view, anchor and FIB-SEM tests below now drive that view.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from spacr.qt import make_masks_demo as demo
from spacr.qt import mask_engine as engine
from spacr.qt.screens.make_masks import MODE_BRUSH, MODE_ERASE, MakeMasksScreen
from spacr.tiff_io import write_tiff

ROOT = Path(__file__).resolve().parents[2]


def _square(shape, rows, columns):
    region = np.zeros(shape, dtype=bool)
    region[rows, columns] = True
    return region


def test_plain_stack_round_trip_keeps_shape_labels_and_voxel_size(tmp_path):
    image = np.random.default_rng(1).integers(0, 4000, (12, 20, 16), dtype=np.uint16)
    tifffile.imwrite(tmp_path / "stack.tif", image)
    session = engine.load_volume_image_and_mask(str(tmp_path), "stack.tif")
    assert session.axes == "ZYX" and session.mask.shape == image.shape
    assert session.plane("Z", 0)[0].shape == (20, 16)
    assert session.plane("Y", 0)[0].shape == (12, 16)
    assert session.plane("X", 0)[0].shape == (12, 20)
    session.paint_plane("Z", 3, 1, paint=_square((20, 16), slice(2, 6), slice(2, 6)))
    session.paint_plane("Y", 10, 2, paint=_square((12, 16), slice(5, 9), slice(10, 14)))
    path = engine.save_volume_mask(session)
    with tifffile.TiffFile(path) as handle:
        assert handle.series[0].axes == "ZYX"
        assert handle.series[0].shape == image.shape
        assert handle.shaped_metadata[0]["voxel_spacing"] == [1.0, 1.0, 1.0]
    reopened = engine.load_volume_image_and_mask(str(tmp_path), "stack.tif")
    assert np.array_equal(reopened.mask, session.mask)
    expected = np.zeros(image.shape, dtype=np.uint16)
    expected[3, 2:6, 2:6] = 1
    expected[5:9, 10, 10:14] = 2
    assert np.array_equal(reopened.mask, expected)


def test_ome_and_mrc_volumes_keep_their_physical_voxel_size(tmp_path):
    image = np.arange(6 * 8 * 10, dtype=np.uint16).reshape(6, 8, 10)
    tifffile.imwrite(tmp_path / "cells.ome.tif", image, ome=True,
                     metadata={"axes": "ZYX", "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5,
                               "PhysicalSizeZ": 2.0})
    session = engine.load_volume_image_and_mask(str(tmp_path), "cells.ome.tif")
    assert session.spacing == (2.0, 0.5, 0.5) and session.unit == "um"
    session.paint_plane("Z", 1, 1, paint=_square((8, 10), slice(0, 2), slice(0, 2)))
    with tifffile.TiffFile(engine.save_volume_mask(session)) as handle:
        assert handle.shaped_metadata[0]["voxel_spacing"] == [2.0, 0.5, 0.5]
        assert handle.shaped_metadata[0]["unit"] == "um"
        x = handle.pages[0].tags["XResolution"].value
        assert x[0] / x[1] == pytest.approx(2.0)
    mrcfile = pytest.importorskip("mrcfile")
    with mrcfile.new(str(tmp_path / "tomo.mrc")) as handle:
        handle.set_data(image.astype(np.float32))
        handle.voxel_size = (40.0, 40.0, 80.0)
    volume = engine.load_volume_image_and_mask(str(tmp_path), "tomo.mrc")
    assert volume.axes == "ZYX" and volume.image.shape == image.shape
    assert volume.spacing == (8.0, 4.0, 4.0) and volume.unit == "nm"
    with mrcfile.new(str(tmp_path / "flat.mrc")) as handle:
        handle.set_data(image[0].astype(np.float32))
    with pytest.raises(ValueError):
        engine.load_volume_image_and_mask(str(tmp_path), "flat.mrc")


def test_painting_protects_other_objects_and_undo_restores_voxels():
    session = engine.VolumeMaskSession(np.zeros((4, 6, 6), dtype=np.uint8),
                                       np.zeros((4, 6, 6), dtype=np.uint16))
    session.paint_plane("Z", 1, 1, paint=_square((6, 6), slice(0, 3), slice(0, 3)))
    changed = session.paint_plane("Z", 1, 2, paint=_square((6, 6), slice(0, 6), slice(0, 6)))
    assert changed == 36 - 9
    assert np.count_nonzero(session.mask == 1) == 9
    session.paint_plane("Z", 1, 2, erase=_square((6, 6), slice(0, 2), slice(0, 6)))
    assert np.count_nonzero(session.mask[1, :2]) == 0
    assert session.undo() and np.count_nonzero(session.mask[1, :2]) == 12
    assert session.redo() and np.count_nonzero(session.mask[1, :2]) == 0
    with pytest.raises(ValueError):
        session.paint_plane("Z", 9, 1, paint=np.ones((6, 6), bool))
    with pytest.raises(ValueError):
        session.paint_plane("Z", 0, 1, paint=np.ones((5, 6), bool))


def test_interpolation_fills_only_the_gap_between_drawn_slices():
    session = engine.VolumeMaskSession(np.zeros((9, 20, 20), dtype=np.uint8),
                                       np.zeros((9, 20, 20), dtype=np.uint16))
    square = _square((20, 20), slice(5, 15), slice(5, 15))
    session.paint_plane("Z", 1, 1, paint=square)
    session.paint_plane("Z", 7, 1, paint=square)
    session.mask[4, 0, 0] = 3
    assert session.interpolate(1, "Z") > 0
    for index in range(1, 8):
        assert np.array_equal(session.mask[index] == 1, square), index
    assert not session.mask[0].any() and not session.mask[8].any()
    assert session.mask[4, 0, 0] == 3


def test_assign_objects_splits_unconnected_parts_into_new_ids():
    mask = np.zeros((6, 10, 10), dtype=np.uint16)
    mask[0:3, 0:3, 0:3] = 1
    mask[4:6, 6:10, 6:10] = 1
    mask[3, 3, 3] = 1
    mask[0:2, 7:9, 0:2] = 2
    session = engine.VolumeMaskSession(np.zeros(mask.shape, dtype=np.uint8), mask)
    changed, objects = session.assign_objects()
    assert objects == 3
    assert changed == 28
    assert np.all(session.mask[4:6, 6:10, 6:10] == 1)
    assert np.all(session.mask[0:3, 0:3, 0:3] == 3) and session.mask[3, 3, 3] == 3
    assert np.all(session.mask[0:2, 7:9, 0:2] == 2)
    assert session.undo() and np.count_nonzero(session.mask == 3) == 0


@pytest.fixture
def screen(qtbot, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Discard)
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    widget.resize(1300, 900)
    widget.show()
    yield widget
    stage = widget._mask_stage
    qtbot.waitUntil(lambda: stage._worker.idle() and stage._mesher.idle(), timeout=5000)
    stage.dirty = False


@pytest.fixture
def editor(screen, tmp_path):
    image = np.random.default_rng(2).integers(0, 255, (10, 24, 24), dtype=np.uint8)
    tifffile.imwrite(tmp_path / "volume.tif", image)
    assert screen.open_volume(str(tmp_path / "volume.tif")) is screen._mask_stage
    for _ in range(4):
        QApplication.processEvents()
    return screen._mask_stage


def _stroke(canvas, qtbot, start, end):
    qtbot.mousePress(canvas, Qt.LeftButton, pos=start)
    qtbot.mouseMove(canvas, end)
    qtbot.mouseRelease(canvas, Qt.LeftButton, pos=end)
    QApplication.processEvents()


def test_the_volume_opens_on_its_middle_slice_and_paints_with_the_mouse(editor, screen, qtbot):
    canvas = screen._canvas
    assert editor.view == "default" and canvas.mode == MODE_BRUSH
    assert editor._slider.value() == 5
    middle = canvas.rect().center()
    _stroke(canvas, qtbot, middle - QPoint(20, 0), middle + QPoint(20, 0))
    drawn = editor.session.mask
    assert drawn[5].any() and not drawn[:5].any() and not drawn[6:].any()
    assert set(np.unique(drawn)) == {0, 1}
    assert editor._selected == 1 and editor.dirty
    assert editor._labels.currentData() == 1
    screen._mode_buttons[MODE_ERASE].click()
    assert canvas.mode == MODE_ERASE
    _stroke(canvas, qtbot, middle - QPoint(40, 0), middle + QPoint(40, 0))
    assert not editor.session.mask.any()
    screen._on_undo()
    assert editor.session.mask[5].any()


def test_drawing_on_side_views_interpolating_and_assigning(editor, screen, qtbot):
    editor.set_view("orthogonal")
    y = editor._cursor[1]
    pane = editor._xz
    assert pane.image.shape == (10, 24)
    before = pane.mask.copy()
    pane.mask[2:8, 3:6] = 255
    pane.stroke_finished.emit()
    assert np.array_equal(editor.session.mask[2:8, y, 3:6], np.full((6, 3), 255, dtype=np.uint16))
    assert np.array_equal(before, np.zeros_like(before))
    editor.set_view("default")
    editor._labels.setCurrentIndex(editor._labels.findData(0))
    for index in (1, 6):
        editor.set_slice(index)
        screen._canvas.mask[15:20, 15:20] = editor.draw_label()
        screen._canvas.stroke_finished.emit()
    assert editor._selected == 1
    editor._interpolate_button.click()
    assert all(np.count_nonzero(editor.session.mask[index] == 1) == 25 for index in range(1, 7))
    assert "Interpolated" in screen._status_label.text()
    editor.session.paint_plane("Z", 9, 1, paint=_square((24, 24), slice(0, 2), slice(0, 2)))
    editor._assign_button.click()
    assert set(np.unique(editor.session.mask)) == {0, 1, 2, 255}
    assert editor.save()
    path = Path(editor.session.mask_path)
    with tifffile.TiffFile(path) as handle:
        assert handle.series[0].shape == (10, 24, 24)
        assert handle.series[0].axes == "ZYX"


def test_true_3d_still_edits_full_voxels(editor, qtbot):
    from spacr.qt.widgets.gate_spec import BoxGate

    editor.set_view("true3d")
    editor._panel.canvas.gate_drawn.emit(
        BoxGate.from_limits("(unnamed)", ("X", "Y", "Z"), [(0, 2), (0, 2), (0, 2)]))
    qtbot.waitUntil(lambda: not editor.busy, timeout=5000)
    assert np.count_nonzero(editor.session.mask == 1) == 8
    editor.set_view("default")
    editor.set_slice(0)
    assert editor.canvas.mask[0:2, 0:2].all()


@pytest.fixture
def fibsem_set(tmp_path_factory):
    spec = importlib.util.spec_from_file_location(
        "build_fibsem_example_dataset", ROOT / "tools" / "build_fibsem_example_dataset.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    source = tmp_path_factory.mktemp("fibsem_source")
    image = np.random.default_rng(3).integers(0, 255, (200, 16, 16), dtype=np.uint8)
    labels = np.zeros(image.shape, dtype=np.uint8)
    labels[80:120, 2:8, 2:8] = 5
    tifffile.imwrite(source / tool.SOURCE_IMAGE, image)
    tifffile.imwrite(source / tool.SOURCE_LABELS, labels)
    dest = tmp_path_factory.mktemp("fibsem_built")
    tool.build(dest, source)
    return dest / tool.FOLDER


def test_the_menu_offers_the_volume_alone_and_with_masks(screen):
    actions = screen._test_data_actions
    assert actions["fibsem"].objectName() == "MakeMasksFibsemTestData"
    assert actions["fibsem_masks"].objectName() == "MakeMasksFibsemMasksTestData"
    assert "FIB-SEM" in actions["fibsem"].text() and "masks" in actions["fibsem_masks"].text()


def test_with_masks_opens_the_labels_for_correction_and_keeps_the_truth(screen, fibsem_set):
    truth = fibsem_set / "ground_truth_masks" / demo.FIBSEM_VOLUME
    published = truth.read_bytes()
    assert demo.load_the_fibsem_test_data(screen, with_masks=True, folder=fibsem_set)
    stage = screen._mask_stage
    assert screen._view_tabs.widget(0) is stage and stage.view_picker.isVisible()
    assert screen._canvas.image.shape == (16, 16)
    assert stage.session.mask.max() == 1
    assert stage.session.spacing == (15.0, 15.0, 15.0)
    assert Path(stage.session.mask_path).parent.name == "corrected_masks"
    stage.session.paint_plane("Z", 0, 1, paint=np.ones((16, 16), dtype=bool))
    stage.dirty = True
    assert stage.save()
    assert truth.read_bytes() == published
    assert demo.load_the_fibsem_test_data(screen, folder=fibsem_set)
    assert stage.session is not None and not stage.session.mask.any()
    assert Path(stage.session.mask_path).parent.name == "masks"


def test_a_missing_set_is_downloaded_then_opened(screen, fibsem_set, tmp_path):
    import shutil

    target = tmp_path / "cache" / "make_masks_fibsem"
    seen = {}

    def fake_download(parent, dest, on_done):
        seen["dest"] = dest
        assert not screen._btn_test_data.isEnabled()
        shutil.copytree(fibsem_set, dest)
        on_done(object(), "")

    assert not demo.load_the_fibsem_test_data(screen, with_masks=True, ask=fake_download,
                                              folder=target)
    assert seen["dest"] == target
    assert screen._btn_test_data.isEnabled()
    assert screen._mask_stage.session is not None
    assert screen._mask_stage.session.mask.max() == 1
    assert (target / "corrected_masks" / demo.FIBSEM_VOLUME).exists()


def test_the_download_worker_targets_the_fibsem_repository(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(demo, "download_toxo_mito_demo",
                        lambda parent, dest, on_done, **kwargs: seen.update(dest=dest, **kwargs))
    demo.download_fibsem_example(None, tmp_path, lambda *args: None)
    assert seen["worker_factory"].repo == demo.FIBSEM_EXAMPLE_REPO
    assert seen["worker_factory"].archive == "spacr-example-fibsem.tar"
    assert seen["dest"] == tmp_path and "FIB-SEM" in seen["title"]
    assert demo.fibsem_example_folder().name == "make_masks_fibsem"
