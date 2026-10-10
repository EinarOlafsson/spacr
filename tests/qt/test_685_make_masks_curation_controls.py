"""Item 685: Make Masks curation controls and the plaque model's heading.

* an incomplete dividing stroke is kept, and a second stroke finishes it;
  undo and save/reopen keep working;
* with the magnifier on, a right click deletes the object under it, in
  every scope and in Divide / Merge, while a right drag still merges;
* the editable ``index / total`` tally beside the Flows tab jumps, refuses
  bad input and follows navigation;
* "First unreviewed" opens the first field without Keep or Discard, read
  from the persisted CSV after reopening;
* the Toxoplasma plaque model is filed under spaCR exactly once, also when
  the shared catalogue carries the same checkpoint.
"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QTabBar

from spacr import model_zoo as zoo
from spacr.qt import mask_engine as engine
from spacr.qt.screens.make_masks import MODE_BRUSH, MODE_DIVIDE, MakeMasksScreen

CANVAS_W, CANVAS_H = 600, 400
IMG_N = 64
PIXMAP_N = 400
MARGIN_X = (CANVAS_W - PIXMAP_N) // 2


def canvas_xy(img_x, img_y):
    return (MARGIN_X + img_x * PIXMAP_N / IMG_N, img_y * PIXMAP_N / IMG_N)


def _evt(kind, x, y, buttons, button):
    pos = QPointF(float(x), float(y))
    return QMouseEvent(kind, pos, pos, button, buttons, Qt.NoModifier)


def gesture(widget, points, button=Qt.LeftButton):
    first = canvas_xy(*points[0])
    widget.mousePressEvent(_evt(QEvent.Type.MouseButtonPress, *first,
                                button, button))
    for point in points[1:]:
        widget.mouseMoveEvent(_evt(QEvent.Type.MouseMove, *canvas_xy(*point),
                                   button, Qt.NoButton))
    widget.mouseReleaseEvent(_evt(QEvent.Type.MouseButtonRelease,
                                  *canvas_xy(*points[-1]), Qt.NoButton,
                                  button))


def field_image():
    img = np.zeros((IMG_N, IMG_N), dtype=np.uint16)
    img[8:56, 8:56] = 20000
    return img


def merged_mask():
    mask = np.zeros((IMG_N, IMG_N), dtype=np.uint16)
    yy, xx = np.mgrid[0:IMG_N, 0:IMG_N]
    left = (xx - 20) ** 2 + (yy - 32) ** 2 <= 8 ** 2
    right = (xx - 40) ** 2 + (yy - 32) ** 2 <= 8 ** 2
    waist = (np.abs(yy - 32) <= 3) & (xx >= 20) & (xx <= 40)
    mask[left | right | waist] = 7
    mask[4:10, 4:10] = 3
    mask[54:60, 44:54] = 9
    return mask


def labels_of(mask):
    return sorted(int(v) for v in np.unique(mask) if v)


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    root = tmp_path / "field"
    root.mkdir()
    for index in range(4):
        imageio.imwrite(root / f"img_{index:02d}.tif", field_image())
    return root


@pytest.fixture
def screen(qtbot, qt_theme_applied, folder):
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    assert widget._open_folder(str(folder))
    widget._canvas.set_image_and_mask(field_image(), merged_mask())
    widget._history.clear()
    widget._history.push(widget._canvas.mask)
    widget._canvas.resize(CANVAS_W, CANVAS_H)
    widget._canvas.refresh()
    assert widget._canvas.pixmap().width() == PIXMAP_N
    widget._magnifier.segment = lambda request: np.zeros(
        request.crop.shape[:2] if request.crop is not None
        else request.shape, dtype=np.int32)
    yield widget
    widget._magnifier.set_enabled(False)
    widget._magnifier.close()


def test_engine_keeps_an_incomplete_cut_and_a_second_cut_finishes_it():
    mask = merged_mask()
    first, splits, cut = engine._divide_or_cut(mask, (30, 16), (30, 30))
    assert splits == [] and cut == [7]
    assert int(first[30, 30]) == 0 and int(mask[30, 30]) == 7 and labels_of(first) == [3, 7, 9]
    second, splits, cut = engine._divide_or_cut(first, (30, 48), (30, 30))
    assert len(splits) == 1 and cut == []
    assert int(second[32, 20]) != int(second[32, 40])
    assert labels_of(mask) == [3, 7, 9]
    unchanged, splits, cut = engine._divide_or_cut(mask, (30, 32), (30, 32))
    assert splits == cut == [] and np.array_equal(unchanged, mask)


def test_engine_never_deletes_an_object_a_cut_covers():
    mask = np.zeros((20, 20), dtype=np.uint16)
    mask[10, 9:11] = 4
    out, splits, cut = engine._divide_or_cut(mask, (5, 10), (15, 10))
    assert splits == cut == [] and np.array_equal(out, mask)


def test_two_strokes_divide_with_undo_and_save_reopen(screen, qtbot):
    before = screen._canvas.mask.copy()
    screen._set_mode(MODE_DIVIDE)
    gesture(screen._canvas, [(30, 16), (30, 30)])
    grooved = screen._canvas.mask.copy()
    assert not np.array_equal(grooved, before)
    assert labels_of(grooved) == [3, 7, 9]
    gesture(screen._canvas, [(30, 48), (30, 30)])
    divided = screen._canvas.mask.copy()
    assert len(labels_of(divided)) == 4
    assert [edit.kind for edit in screen._log.edits] == ["divide", "divide"]
    screen._on_undo()
    np.testing.assert_array_equal(screen._canvas.mask, grooved)
    screen._on_undo()
    np.testing.assert_array_equal(screen._canvas.mask, before)
    screen._on_redo()
    screen._on_redo()
    np.testing.assert_array_equal(screen._canvas.mask, divided)
    screen._on_save()
    folder = screen._folder
    reopened = MakeMasksScreen()
    qtbot.addWidget(reopened)
    assert reopened._open_folder(folder)
    saved = reopened._canvas.mask
    assert len(labels_of(saved)) == 4
    assert int(saved[30, 30]) == 0 and int(saved[34, 30]) == 0
    reopened.close()


@pytest.mark.parametrize("scope", ["region", "image"])
def test_right_click_with_the_magnifier_on_deletes_and_is_one_undo(screen, scope):
    screen._set_mode(MODE_BRUSH)
    screen._magnifier.set_scope(scope)
    screen._magnifier.set_enabled(True)
    before = screen._canvas.mask.copy()
    gesture(screen._canvas, [(6, 6)], button=Qt.RightButton)
    assert 3 not in labels_of(screen._canvas.mask)
    assert labels_of(screen._canvas.mask) == [7, 9]
    assert screen._log.edits[-1].kind in ("delete", "sweep_delete")
    screen._on_undo()
    np.testing.assert_array_equal(screen._canvas.mask, before)


@pytest.mark.parametrize("scope", ["region", "image"])
def test_divide_mode_right_click_deletes_but_right_drag_merges(screen, scope):
    screen._set_mode(MODE_DIVIDE)
    gesture(screen._canvas, [(30, 16), (30, 48)])
    divided = screen._canvas.mask.copy()
    left, right = int(divided[32, 20]), int(divided[32, 40])
    screen._magnifier.set_scope(scope)
    screen._magnifier.set_enabled(True)
    gesture(screen._canvas, [(6, 6)], button=Qt.RightButton)
    assert 3 not in labels_of(screen._canvas.mask)
    assert screen._log.edits[-1].kind == "delete"
    gesture(screen._canvas, [(20, 32), (40, 32)], button=Qt.RightButton)
    merged = screen._canvas.mask
    assert int(merged[32, 20]) == int(merged[32, 40]) == left
    assert right not in labels_of(merged)
    assert screen._log.edits[-1].kind == "merge"


def test_a_locked_lens_deletes_what_was_clicked(screen):
    screen._set_mode(MODE_BRUSH)
    screen._magnifier.set_scope("image")
    screen._magnifier.set_enabled(True)
    screen._magnifier.hover(QPointF(*canvas_xy(48, 56)))
    screen._magnifier.set_locked(True)
    gesture(screen._canvas, [(6, 6)], button=Qt.RightButton)
    assert labels_of(screen._canvas.mask) == [7, 9]


def test_auto_accept_does_not_put_a_deleted_object_back(screen):
    magnifier = screen._magnifier
    magnifier.set_enabled(True)
    magnifier.auto_accept = True
    magnifier._cursor = (6, 6)
    footprint = screen._canvas.mask == 3
    magnifier.hold_auto_accept(footprint)
    emitted = []
    magnifier.commit_ready.connect(emitted.append)
    magnifier._waiting.clear()
    magnifier._stroke = None
    assert magnifier._accept_proposed() is False
    assert magnifier._auto_hold is not None
    magnifier._cursor = (30, 32)
    magnifier._accept_proposed()
    assert magnifier._auto_hold is None
    assert emitted == []


def test_the_queue_tally_sits_on_the_flows_tab_and_jumps(screen):
    bar = screen._view_tabs.tabBar()
    assert bar.tabText(screen._tab_flow) == "Flows"
    assert bar.tabButton(screen._tab_flow, QTabBar.RightSide).objectName() \
        == "MakeMasksQueueTally"
    assert screen._tally_index.text() == "1"
    assert screen._tally_total.text() == "/4"
    for wanted in (4, 2, 1, 3):
        screen._tally_index.setText(str(wanted))
        screen._tally_index.returnPressed.emit()
        assert screen._current_index == wanted - 1
        assert screen._tally_index.text() == str(wanted)
    screen._on_next()
    assert screen._tally_index.text() == "4"


@pytest.mark.parametrize("text", ["", "abc", "0", "5", "-1", "2.5"])
def test_invalid_tally_input_leaves_the_image_alone(screen, text):
    screen._go_to_index(1)
    shown = screen._canvas.image
    screen._tally_index.setText(text)
    assert screen._on_tally_entered() is False
    assert screen._current_index == 1
    assert screen._canvas.image is shown
    assert screen._tally_index.text() == "2"


def test_a_jump_that_cannot_leave_the_field_stays(screen, monkeypatch):
    monkeypatch.setattr(screen, "_save_boxes_if_needed", lambda: False)
    screen._tally_index.setText("3")
    assert screen._on_tally_entered() is False
    assert screen._current_index == 0
    assert screen._tally_index.text() == "1"


def test_a_jump_records_no_verdict(screen):
    screen._tally_index.setText("3")
    assert screen._on_tally_entered()
    assert engine.read_curation(screen._folder) == {}


def test_an_empty_queue_shows_zero(screen):
    screen._image_files = []
    screen._current_index = 0
    screen._sync_queue_tally()
    assert screen._tally_index.text() == "0"
    assert screen._tally_total.text() == "/0"
    assert not screen._tally_index.isEnabled()
    assert screen._go_to_index(0) is False
    assert screen._on_first_unreviewed() is False


def test_first_unreviewed_sits_left_of_clear_all_objects(screen):
    row = screen._btn_clear.parentWidget().layout()
    assert screen._btn_first_unreviewed.objectName() == "MakeMasksFirstUnreviewed"
    assert row.indexOf(screen._btn_first_unreviewed) + 1 == row.indexOf(
        screen._btn_clear)


def test_first_unreviewed_reads_persisted_verdicts_after_reopening(
        screen, qtbot, folder):
    names = screen._image_files
    for index, keep in ((0, True), (1, False), (3, True)):
        path = str(folder / names[index])
        engine.record_curation(str(folder), path, path, 1, keep)
    reopened = MakeMasksScreen()
    qtbot.addWidget(reopened)
    assert reopened._open_folder(str(folder))
    assert reopened._on_first_unreviewed() is True
    assert reopened._current_index == 2
    assert reopened._tally_index.text() == "3"
    assert reopened._on_first_unreviewed() is False
    assert reopened._current_index == 2
    path = str(folder / names[2])
    engine.record_curation(str(folder), path, path, 1, False)
    reopened._go_to_index(0)
    before = engine.read_curation(str(folder))
    assert reopened._on_first_unreviewed() is False
    assert reopened._current_index == 0
    assert engine.read_curation(str(folder)) == before
    reopened.close()


def _plaque_record():
    return next(record for record in zoo.BUNDLED_REMOTE_MODELS
                if record["key"] == "toxoplasma_plaque_v3")


def test_the_shared_plaque_checkpoint_is_filed_under_spacr():
    shared = zoo._entry_from_mapping(_plaque_record(), source="shared")
    assert zoo.source_of(shared) == "spaCR"
    borrowed = zoo._entry_from_mapping(
        dict(_plaque_record(), sha256="0" * 64), source="community")
    assert zoo.source_of(borrowed) == "spaCR community"
    stranger = zoo._entry_from_mapping(
        {"key": "someone_v1", "name": "someone", "sha256": "1" * 64},
        source="shared")
    assert zoo.source_of(stranger) == "spaCR community"


def test_the_plaque_model_appears_once_under_spacr(monkeypatch):
    shared = zoo._entry_from_mapping(_plaque_record(), source="shared")
    monkeypatch.setattr(zoo, "shared_catalogue", lambda **_kw: (shared,))
    monkeypatch.setattr(zoo, "bioimageio_entries", lambda *a, **k: [])
    groups = zoo.group_by_source(zoo.catalogue(remote=True, block=False))
    where = [(heading, entry) for heading, rows in groups.items()
             for entry in rows if entry.key == "toxoplasma_plaque_v3"]
    assert [heading for heading, _ in where] == ["spaCR"]
    entry = where[0][1]
    assert entry.sha256 == _plaque_record()["sha256"]
    assert entry.uri == _plaque_record()["uri"]
    for key in ("toxoplasma_plaque_v1", "toxoplasma_plaque_v2"):
        assert [h for h, rows in groups.items()
                for e in rows if e.key == key] == ["spaCR"]


def _two_channel_field():
    image = np.zeros((IMG_N, IMG_N, 2), dtype=np.uint16)
    image[..., 0] = 100
    image[..., 1] = 7
    image[4:10, 4:10, 0] = 1000
    return image


def test_vvvv_export_writes_the_agreed_folder(tmp_path):
    import json

    from spacr import tabular

    labels = merged_mask()
    folder = engine._export_vvvv(
        tmp_path / "out", "img_00.tif", labels, _two_channel_field(),
        source_path="/data/img_00.tif", pixel_size=(0.5, 0.25),
        channel_names=["dapi"], classes={3: "nucleus"}, scores={3: 0.9})
    folder = Path(folder)
    assert folder == tmp_path / "out" / "img_00"
    assert sorted(p.name for p in folder.iterdir()) == [
        "img_00_labels.png", "img_00_objects.csv", "img_00_outlines.png",
        "manifest.json"]
    saved = imageio.imread(folder / "img_00_labels.png")
    assert saved.dtype == np.uint16
    np.testing.assert_array_equal(saved, labels)
    outlines = imageio.imread(folder / "img_00_outlines.png")
    assert outlines.shape == (IMG_N, IMG_N, 4)
    assert int(outlines[..., 3][labels == 0].max()) == 0
    assert int(outlines[4, 4, 3]) == 255 and int(outlines[6, 6, 3]) == 0
    table = tabular.read_table(folder / "img_00_objects.csv", canonicalise=False)
    assert list(table["label"]) == [3, 7, 9]
    row = table.set_index("label").loc[3]
    assert row["area"] == 36
    assert row["centroid_x"] == pytest.approx(7.0)
    assert row["centroid_y"] == pytest.approx(7.0)
    assert (row["bbox_x0"], row["bbox_y0"], row["bbox_x1"], row["bbox_y1"]) \
        == (4, 4, 10, 10)
    assert row["mean_intensity_dapi"] == pytest.approx(1000)
    assert row["mean_intensity_channel_2"] == pytest.approx(7)
    assert row["class"] == "nucleus" and row["score"] == pytest.approx(0.9)
    manifest = json.loads((folder / "manifest.json").read_text("utf-8"))
    assert manifest["format"] == engine.VVVV_EXPORT_FORMAT
    assert manifest["pixel_size"] == {"x": 0.5, "y": 0.25, "unit": "µm"}
    assert manifest["channel_names"] == ["dapi", "channel_2"]
    assert manifest["source_path"] == "/data/img_00.tif"
    assert manifest["spacr_version"]
    assert manifest["objects"] == 3
    assert manifest["files"]["labels"] == "img_00_labels.png"


def test_vvvv_export_without_image_or_objects(tmp_path):
    folder = Path(engine._export_vvvv(tmp_path, "empty.png",
                                     np.zeros((8, 8), dtype=np.uint16)))
    import json

    manifest = json.loads((folder / "manifest.json").read_text("utf-8"))
    assert manifest["objects"] == 0 and manifest["pixel_size"] is None
    with pytest.raises(ValueError):
        engine._export_vvvv(tmp_path, "big.png",
                           np.full((2, 2), 70000, dtype=np.int64))


def test_vvvv_export_replaces_files_whole(tmp_path, monkeypatch):
    first = merged_mask()
    folder = Path(engine._export_vvvv(tmp_path, "img.tif", first))
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    real = engine.imageio.imwrite

    def half_then_fail(path, *args, **kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(engine.imageio, "imwrite", half_then_fail)
    with pytest.raises(OSError):
        engine._export_vvvv(tmp_path, "img.tif", np.zeros_like(first))
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before
    monkeypatch.setattr(engine.imageio, "imwrite", real)
    changed = first.copy()
    changed[changed == 9] = 0
    engine._export_vvvv(tmp_path, "img.tif", changed)
    assert sorted(p.name for p in folder.iterdir()) == sorted(before)
    np.testing.assert_array_equal(
        imageio.imread(folder / "img_labels.png"), changed)


@pytest.fixture
def alpha(monkeypatch):
    from spacr.qt import preferences

    state = {"on": False}
    monkeypatch.setattr(preferences, "_get_show_alpha_features",
                        lambda: state["on"])
    return state


def test_vvvv_buttons_follow_the_alpha_switch(qtbot, qt_theme_applied, alpha):
    from PySide6.QtWidgets import QPushButton

    from spacr.qt.preferences import _apply_alpha_widgets
    from spacr.settings import ALPHA_FEATURES

    assert set(ALPHA_FEATURES[685]["widgets"]) == {
        "MakeMasksVvvvExport", "MakeMasksVvvvExportOnSave"}
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    buttons = [widget.findChild(QPushButton, name)
               for name in ALPHA_FEATURES[685]["widgets"]]
    assert all(button is not None and button.isHidden() for button in buttons)
    alpha["on"] = True
    _apply_alpha_widgets(widget)
    assert not any(button.isHidden() for button in buttons)
    alpha["on"] = False
    _apply_alpha_widgets(widget)
    assert all(button.isHidden() for button in buttons)


def test_export_on_save_follows_every_save(screen, alpha, tmp_path, monkeypatch):
    out = tmp_path / "vvvv"
    monkeypatch.setattr(type(screen), "_vvvv_export_root",
                        staticmethod(lambda: str(out)))
    screen._on_save()
    assert not out.exists()
    alpha["on"] = True
    screen._btn_vvvv_on_save.setChecked(True)
    screen._on_save()
    labels = out / "img_00" / "img_00_labels.png"
    assert labels.is_file()
    np.testing.assert_array_equal(imageio.imread(labels),
                                  engine.canonical_labels(screen._canvas.mask))
    screen._set_mode(MODE_DIVIDE)
    gesture(screen._canvas, [(30, 16), (30, 48)])
    screen._on_save()
    assert len(labels_of(imageio.imread(labels))) == 4
    screen._blind = {"codes": {}}
    assert screen._export_vvvv_on_save() is None
    screen._blind = None
    screen._btn_vvvv_on_save.setChecked(False)
