"""Make Masks' 3D mask editor, MitoNet and histogram window at their guards.

Each test drives a path the coverage gate named on ``make_masks.py`` after
N692 (the Draw slices tab, its view and slice controls, Interpolate, Assign,
``_open_volume_with``) and item 691 (MitoNet): a stroke, edit or save that
arrives while the editor is busy or closing, a stroke that changes nothing,
a shape the anchor tab cannot voxelize, a volume dialog that refuses to
close, a MitoNet run without a ticket or inversion, a model fetched this
session and a magnifier drag without a usable start point.

No GPU and no network. Waits are condition-based with generous timeouts so a
heavily loaded machine slows the tests down rather than failing them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
from shiboken6 import delete, isValid

from spacr import model_zoo as mz
from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm
from spacr.qt.screens.make_masks import MakeMasksScreen, _VolumeMaskEditor
from spacr.qt.widgets.gate_spec import BoxGate
from spacr.tiff_io import write_tiff

WAIT = 60000


def _settle(rounds=4):
    for _ in range(rounds):
        QApplication.processEvents()


def _square(shape, rows, columns):
    region = np.zeros(shape, dtype=bool)
    region[rows, columns] = True
    return region


@pytest.fixture
def discard(monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Discard)


def _close(widget, qtbot):
    if isValid(widget):
        widget._busy = False
        widget._closing = False
        widget._dirty = False
        worker = widget._worker
        widget.close()
        qtbot.waitUntil(worker.idle, timeout=WAIT)


@pytest.fixture
def slices(qtbot, tmp_path, discard):
    """The editor on its Draw slices tab, over a 10 x 24 x 24 stack."""
    image = np.random.default_rng(7).integers(0, 255, (10, 24, 24), dtype=np.uint8)
    tifffile.imwrite(tmp_path / "volume.tif", image)
    session = engine.load_volume_image_and_mask(str(tmp_path), "volume.tif")
    widget = _VolumeMaskEditor(session)
    qtbot.addWidget(widget)
    widget.resize(900, 700)
    widget.show()
    _settle()
    yield widget
    _close(widget, qtbot)


@pytest.fixture
def anchors(qtbot, tmp_path, discard):
    """The editor on its 3D anchors tab, over an 8^3 stack with real spacing."""
    image = np.arange(8 ** 3, dtype=np.uint16).reshape(8, 8, 8)
    write_tiff(tmp_path / "volume.tif", image,
               metadata={"axes": "ZYX", "voxel_spacing": [1., .5, .25], "unit": "um"})
    session = engine.load_volume_image_and_mask(str(tmp_path), "volume.tif")
    widget = _VolumeMaskEditor(session)
    qtbot.addWidget(widget)
    widget.show()
    widget._tabs.setCurrentWidget(widget._panel)
    _settle()
    yield widget
    _close(widget, qtbot)


def _paint_on_canvas(editor, rows, columns, value=255):
    editor._slice_view.mask[rows, columns] = value
    editor._slice_view.stroke_finished.emit()
    _settle(2)


def test_a_stroke_while_an_anchor_edit_runs_is_ignored(slices):
    slices._busy = True
    before = slices._session.mask.copy()
    _paint_on_canvas(slices, slice(2, 8), slice(2, 8))
    assert np.array_equal(slices._session.mask, before)
    assert not slices._session._undo and not slices._dirty


def test_a_stroke_that_changes_no_pixel_makes_no_undo_step(slices):
    notice = slices._notice.text()
    slices._slice_view.stroke_finished.emit()
    _settle(2)
    assert not slices._session._undo and not slices._dirty
    assert slices._notice.text() == notice


def test_painting_over_another_object_changes_nothing_and_keeps_new_object(slices):
    index = slices._slice.value()
    slices._session.paint_plane("Z", index, 1, paint=_square((24, 24), slice(4, 12), slice(4, 12)))
    slices._refresh_slice()
    assert slices._labels.currentData() == 0
    owned = slices._session.mask.copy()
    _paint_on_canvas(slices, slice(5, 10), slice(5, 10), value=200)
    assert np.array_equal(slices._session.mask, owned)
    assert not slices._dirty and slices._selected_label == 0
    assert "Updated 0 voxels" in slices._notice.text()


def test_interpolate_without_a_chosen_object_explains_itself(slices):
    slices._labels.setCurrentIndex(slices._labels.findData(0))
    slices._interpolate_button.click()
    assert "at least two slices" in slices._notice.text()
    assert not slices._session._undo


def test_busy_assign_undo_and_save_leave_the_volume_alone(slices):
    _paint_on_canvas(slices, slice(2, 6), slice(2, 6))
    _paint_on_canvas(slices, slice(14, 18), slice(14, 18))
    painted = slices._session.mask.copy()
    history = len(slices._session._undo)
    slices._busy = True
    slices._assign_button.click()
    slices._history(False)
    assert slices._save_mask() is False
    assert np.array_equal(slices._session.mask, painted)
    assert len(slices._session._undo) == history
    assert not Path(slices._session.mask_path).exists()
    slices._busy = False
    slices._closing = True
    slices._history(False)
    assert np.array_equal(slices._session.mask, painted)
    slices._closing = False


def test_undo_with_nothing_to_undo_keeps_the_editor_clean(slices):
    slices._history(False)
    slices._history(True)
    assert not slices._dirty and not slices._session.mask.any()


def test_a_save_after_the_source_changed_is_refused_and_stays_dirty(slices):
    _paint_on_canvas(slices, slice(2, 6), slice(2, 6))
    assert slices._dirty
    source = Path(slices._session.source_path)
    tifffile.imwrite(source, np.zeros((10, 24, 24), dtype=np.uint8))
    assert slices._save_mask() is False
    assert "source volume changed" in slices._notice.text()
    assert slices._dirty and not Path(slices._session.mask_path).exists()


def test_a_disconnected_object_cannot_open_on_the_anchor_tab(slices):
    index = slices._slice.value()
    slices._session.paint_plane("Z", index, 1, paint=_square((24, 24), slice(0, 2), slice(0, 2)))
    slices._session.paint_plane("Z", index, 1, paint=_square((24, 24), slice(2, 4), slice(2, 4)))
    slices._refresh()
    slices._labels.setCurrentIndex(slices._labels.findData(1))
    assert slices._selected_label == 1
    slices._tabs.setCurrentWidget(slices._panel)
    _settle()
    assert slices._selected_label == 0
    assert slices._notice.text()
    assert "1" not in slices._panel.gates.names


def _wait_idle(editor, qtbot):
    qtbot.waitUntil(lambda: not editor._busy, timeout=WAIT)
    _settle()


def test_an_anchor_change_with_no_shape_queues_nothing(anchors):
    anchors._geometry_changed()
    assert not anchors._busy and not anchors._session.mask.any()


def test_a_shape_outside_the_spatial_axes_is_refused(anchors, qtbot):
    gate = BoxGate.from_limits("(unnamed)", ("X", "Y", "Intensity"), [(0, 1), (0, 2), (0, 100)])
    anchors._panel.canvas.gate_drawn.emit(gate)
    _wait_idle(anchors, qtbot)
    assert "valid bounded spatial volume" in anchors._notice.text()
    assert not anchors._session.mask.any() and not anchors._session.geometries


def test_a_shape_drawn_in_z_y_x_order_fills_the_same_voxels(anchors, qtbot):
    gate = BoxGate.from_limits("(unnamed)", ("Z", "Y", "X"), [(0, 4), (0, 2), (0, 1)])
    anchors._panel.canvas.gate_drawn.emit(gate)
    qtbot.waitUntil(lambda: anchors._session.mask.any() and not anchors._busy, timeout=WAIT)
    expected = np.zeros((8, 8, 8), dtype=np.uint16)
    expected[:4, :4, :4] = 1
    assert np.array_equal(anchors._session.mask, expected)
    assert anchors._session.geometries[1].columns == ("X", "Y", "Z")


def test_an_unchanged_shape_is_not_voxelized_twice(anchors, qtbot):
    anchors._panel.canvas.gate_drawn.emit(
        BoxGate.from_limits("(unnamed)", ("X", "Y", "Z"), [(0, 1), (0, 2), (0, 4)]))
    qtbot.waitUntil(lambda: anchors._session.mask.any() and not anchors._busy, timeout=WAIT)
    _settle()
    history = len(anchors._session._undo)
    panel = anchors._panel
    assert (panel.canvas.active_gate or panel.tree.active_gate()) == "1"
    anchors._geometry_changed()
    assert not anchors._busy
    assert len(anchors._session._undo) == history


def test_a_result_arriving_while_closing_is_not_published(anchors):
    notice = anchors._notice.text()
    anchors._closing = True
    anchors._take_edit(((anchors._session, 1, None, anchors._cancel), 64, None))
    assert anchors._notice.text() == notice
    assert not anchors._dirty and anchors._selected_label == 0
    anchors._closing = False


@pytest.fixture
def screen(qtbot, discard):
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    yield widget
    dialog = widget._volume_dialog
    if dialog is not None and isValid(dialog):
        _close(dialog, qtbot)


def _stack(tmp_path, name="stack.tif"):
    image = np.random.default_rng(5).integers(0, 255, (6, 12, 12), dtype=np.uint8)
    tifffile.imwrite(tmp_path / name, image)
    return tmp_path / name


def test_open_volume_asks_for_a_file_and_a_cancelled_picker_opens_nothing(screen, tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        lambda *args, **kwargs: (asked.append(args), ("", ""))[1])
    assert screen.open_volume() is None
    assert asked and screen._volume_dialog is None
    path = _stack(tmp_path)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(path), ""))
    dialog = screen.open_volume()
    assert isinstance(dialog, _VolumeMaskEditor)
    assert dialog._session.image.shape == (6, 12, 12)


def test_a_flat_image_is_refused_on_the_status_line(screen, tmp_path):
    tifffile.imwrite(tmp_path / "flat.tif", np.zeros((12, 12), dtype=np.uint8))
    assert screen.open_volume(str(tmp_path / "flat.tif")) is None
    assert screen._volume_dialog is None
    assert screen._status_label.text()


def test_an_unsaved_volume_that_will_not_close_keeps_the_screen_open(screen, tmp_path, monkeypatch):
    dialog = screen.open_volume(str(_stack(tmp_path)))
    dialog._session.paint_plane("Z", 0, 1, paint=np.ones((12, 12), dtype=bool))
    dialog._dirty = True
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Cancel)
    other = _stack(tmp_path, "other.tif")
    assert screen.open_volume(str(other)) is dialog
    assert Path(dialog._session.source_path).name == "stack.tif"
    screen.show()
    assert screen.close() is False
    assert isValid(dialog) and screen._volume_dialog is dialog
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Discard)


def test_mitonet_without_inversion_or_ticket_gets_the_raw_crop(monkeypatch):
    seen = {}

    def fake(image, **kwargs):
        seen["image"] = np.asarray(image)
        seen.update(kwargs)
        kwargs["on_tile"](1, 2)
        kwargs["on_tile"](2, 2)
        return np.ones(np.asarray(image).shape, np.int32)

    monkeypatch.setattr(mz, "segment_mitonet", fake)
    crop = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    request = mm._MagnifierRequest(
        key=(), crop=crop, box=(0, 0, 8, 8), shape=(8, 8), mode="mitonet",
        sensitivity=1.0, bright=True, min_area=0, model_name="cpsam",
        diameter=30, colour=(0, 0, 0), invert=False, ticket=None)
    labels = mm._mitonet_segmenter(request)
    assert labels.shape == (8, 8)
    np.testing.assert_array_equal(seen["image"], crop)
    assert seen["min_size"] == 1


def test_the_histogram_window_resizes_after_its_layout_is_gone(qtbot):
    dialog = mm._OtsuHistogramDialog(np.array([1., 3., 2.]), np.array([0., 1., 2., 3.]),
                                     [1.5], "Otsu")
    qtbot.addWidget(dialog)
    dialog.show()
    _settle()
    minimum = dialog.minimumHeight()
    delete(dialog.layout())
    assert dialog.layout() is None
    dialog.resize(dialog.width() + 80, dialog.height() + 40)
    QApplication.sendPostedEvents(dialog, QEvent.Resize)
    _settle()
    assert dialog.minimumHeight() == minimum


def test_a_model_fetched_this_session_is_listed_by_where_it_landed(screen, tmp_path, monkeypatch):
    weights = tmp_path / "landed" / "zoo_cp"
    weights.parent.mkdir()
    weights.write_bytes(b"weights")
    entry = mz.ModelEntry(key="zoo_cp", name="zoo_cp", kind="cellpose", source="remote",
                          uri="file:///nowhere/zoo_cp", sha256="0" * 64, size_bytes=7)
    monkeypatch.setattr(mm, "_zoo_cellpose_models", lambda: [("zoo_cp", None, entry)])
    screen._cp_fetched["zoo_cp"] = str(weights)
    screen._fill_zoo_models()
    box = screen._cp_model
    row = box.findData(str(weights))
    assert row >= 0 and box.itemText(row) == "zoo_cp"
    assert box.itemData(row, mm._ZOO_PENDING_ROLE) is None


def _drag(screen, outcome):
    screen._magnifier.drag_ready.emit((outcome, True))
    _settle(2)


@pytest.fixture
def field(screen):
    image = np.random.default_rng(3).integers(0, 4000, (64, 64), dtype=np.uint16)
    mask = np.zeros((64, 64), dtype=np.int32)
    mask[2:14, 2:14] = 4
    screen._canvas.set_image_and_mask(image, mask)
    return screen


def _stroke_labels():
    labels = np.zeros((14, 14), dtype=np.int32)
    labels[1:13, 1:13] = 1
    return labels


def test_a_drag_outcome_without_provenance_pastes_a_new_object(field):
    from spacr.qt._magnifier_drag import _StrokeOutcome

    assert field._detect_min_area() <= 144
    _drag(field, _StrokeOutcome(_stroke_labels(), (30, 30), 1, 1, 1))
    mask = field._canvas.mask
    assert set(np.unique(mask[31:43, 31:43])) == {5}
    assert np.all(mask[2:14, 2:14] == 4)


def test_a_drag_starting_off_the_field_does_not_extend_an_object(field):
    from spacr.qt._magnifier_drag import _StrokeOutcome

    _drag(field, _StrokeOutcome(_stroke_labels(), (30, 30), 1, 1, 1,
                                provenance={"path": [[-3, 90], [31, 31]]}))
    mask = field._canvas.mask
    assert set(np.unique(mask[31:43, 31:43])) == {5}
    assert np.count_nonzero(mask == 4) == 144
