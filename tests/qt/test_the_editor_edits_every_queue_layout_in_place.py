"""``396`` -- the editor edits a sibling set and a ``_seg.npy`` set where they lie.

Until 2026-09-19 ``spacr-make-masks`` read three layouts and the editor
edited one. A sibling set -- ``images/`` beside ``masks/``, which is how the
curation sets in the field are laid out, the plaque queue among them -- and a
folder of Cellpose ``_seg.npy`` bundles were summarised and then refused, so
item 370 had to re-lay its PV queue out as nested before anybody could curate
it.

What is asserted here, on small fixtures of each layout:

* a SIBLING session reads each field's draft from ``<root>/masks`` and saves
  it back there. The failure it replaces is specific: opened on
  ``<root>/images`` the editor used to read and write ``<root>/images/masks``,
  so the draft on screen was empty and every save landed in a folder the set
  does not have. Both halves are checked, and so is the absence of
  ``images/masks`` afterwards;
* a SEG session shows the bundle's image and labels and writes the edited
  labels back INTO the bundle, keeping every other key it had, and bringing
  the two keys derived from the labels (``outlines``, ``ismanual``) up to
  date. The status row is keyed on the queue's stem, ``b_0``, not on the
  ``b_0_seg`` that ``splitext`` would give;
* recrop keeps each folder in its own layout: a sibling child's mask goes to
  ``<root>/masks``, a bundle's child is a bundle. A TIFF written into a folder
  of bundles would make it two layouts at once, which the queue refuses to
  open the next time;
* the gestures are the user's: the tool button is pressed, the canvas is
  clicked, Save and Next are pressed.

The engine half is tested without the screen where the claim is about a file
on disk rather than about a control.
"""
from __future__ import annotations

import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from spacr import cli_make_masks
from spacr.curation import LOG_SUFFIX
from spacr.curation_queue import (
    LAYOUT_SEG,
    LAYOUT_SIBLING,
    STATUS_FILENAME,
    build_queue,
    detect_layout,
    read_status,
)
from spacr.qt import mask_engine as engine
from spacr.qt.screens.make_masks import MakeMasksScreen

CANVAS_W, CANVAS_H = 600, 400
IMG_N = 256
PIXMAP_N = 400
MARGIN_X = (CANVAS_W - PIXMAP_N) // 2

#: Where object 1 of :func:`three_object_field` is, in image pixels.
OBJECT_ONE = (44, 44)
#: A recrop box round object 1 alone -- the recrop suite's BOX_A.
BOX_A = ((16, 16), (112, 112))


def three_object_field():
    """A 256x256 field and its draft: three well-separated 40 px objects."""
    image = np.zeros((IMG_N, IMG_N), dtype=np.uint16)
    mask = np.zeros((IMG_N, IMG_N), dtype=np.uint16)
    for value, (top, left) in enumerate(((24, 24), (176, 176), (96, 96)), 1):
        image[top:top + 40, left:left + 40] = 20000 + 10000 * value
        mask[top:top + 40, left:left + 40] = value
    return image, mask


@pytest.fixture
def sibling(tmp_path: Path) -> Path:
    """``<root>/images/s_0.tif, s_1.tif`` beside ``<root>/masks/`` drafts."""
    root = tmp_path / "sibling"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir(parents=True)
    image, mask = three_object_field()
    for name in ("s_0", "s_1"):
        imageio.imwrite(root / "images" / f"{name}.tif", image)
        imageio.imwrite(root / "masks" / f"{name}.tif", mask)
    return root


@pytest.fixture
def seg(tmp_path: Path) -> Path:
    """Two Cellpose bundles, each carrying keys the editor must not drop."""
    folder = tmp_path / "seg"
    folder.mkdir()
    image, mask = three_object_field()
    for name in ("b_0", "b_1"):
        np.save(folder / f"{name}_seg.npy", {
            "img": (image // 257).astype(np.uint8),
            "masks": mask,
            "outlines": np.zeros_like(mask),
            "ismanual": np.array([False, True, False]),
            "flows": [None, None, None],
            "filename": f"{name}.png",
            "diameter": 31.5,
        }, allow_pickle=True)
    return folder


@pytest.fixture
def no_handover():
    """Leave the terminal slot empty before and after, whatever happens."""
    cli_make_masks.hand_over(None)
    yield
    cli_make_masks.hand_over(None)


def _canvas_xy(img_x: float, img_y: float) -> tuple:
    return (MARGIN_X + img_x * PIXMAP_N / IMG_N, img_y * PIXMAP_N / IMG_N)


def _evt(kind, x, y, buttons=Qt.LeftButton, button=Qt.LeftButton):
    pos = QPointF(float(x), float(y))
    return QMouseEvent(kind, pos, pos, button, buttons, Qt.NoModifier)


def click(canvas, img_x: float, img_y: float) -> None:
    """Press and release on one image pixel."""
    x, y = _canvas_xy(img_x, img_y)
    canvas.mousePressEvent(_evt(QEvent.Type.MouseButtonPress, x, y))
    canvas.mouseReleaseEvent(_evt(QEvent.Type.MouseButtonRelease, x, y,
                                  buttons=Qt.NoButton))


def box_drag(canvas, p0, p1) -> None:
    """Press, drag and release a rectangle, in image coordinates."""
    start, end = _canvas_xy(*p0), _canvas_xy(*p1)
    canvas.mousePressEvent(_evt(QEvent.Type.MouseButtonPress, *start))
    canvas.mouseMoveEvent(_evt(QEvent.Type.MouseMove, *end,
                               buttons=Qt.LeftButton, button=Qt.NoButton))
    canvas.mouseReleaseEvent(_evt(QEvent.Type.MouseButtonRelease, *end,
                                  buttons=Qt.NoButton, button=Qt.LeftButton))


def handed_screen(qtbot, folder: Path) -> MakeMasksScreen:
    """What ``spacr-make-masks --folder <folder> --order name`` opens."""
    cli_make_masks.hand_over(build_queue(folder, order="name",
                                         cache_counts=False))
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    screen._canvas.setFixedSize(CANVAS_W, CANVAS_H)
    screen._canvas.refresh()
    return screen


def bundle(path: Path) -> dict:
    """Read a bundle back the way Cellpose does."""
    return np.load(path, allow_pickle=True).item()


# ===========================================================================
# The sibling layout, through the screen
# ===========================================================================

def test_a_sibling_session_shows_the_draft_the_set_already_has(
        qtbot, qt_theme_applied, no_handover, sibling):
    """The draft on screen is ``<root>/masks/s_0.tif``, not an empty mask."""
    screen = handed_screen(qtbot, sibling)

    assert screen._queue is not None
    assert screen._folder == str(sibling / "images")
    assert screen._image_files == ["s_0.tif", "s_1.tif"]
    _image, draft = three_object_field()
    assert np.array_equal(screen._canvas.mask, draft), (
        "the editor opened the sibling set with the wrong masks folder")
    assert "sorted by name" in screen._src_label.text()


def test_a_sibling_save_lands_in_the_sets_own_masks_folder(
        qtbot, qt_theme_applied, no_handover, sibling):
    """Erase an object, press Save: ``<root>/masks`` changes, nothing else."""
    screen = handed_screen(qtbot, sibling)
    screen._btn_del_obj.click()
    click(screen._canvas, *OBJECT_ONE)
    screen._btn_save.click()

    saved = imageio.imread(sibling / "masks" / "s_0.tif")
    assert 1 not in np.unique(saved), "the erased object is still on disk"
    assert {2, 3} <= set(np.unique(saved).tolist())
    assert not (sibling / "images" / "masks").exists(), (
        "the save made a second masks folder beneath the images")
    assert (sibling / "masks" / ("s_0.tif" + LOG_SUFFIX)).is_file()

    rows = read_status(sibling)
    assert rows["s_0"].state == "done"
    assert rows["s_0"].n_objects == 2
    assert (sibling / STATUS_FILENAME).is_file()
    assert [item.stem for item in build_queue(
        sibling, order="name", cache_counts=False).items] == ["s_1"]


def test_next_in_a_sibling_session_loads_the_next_draft_from_beside(
        qtbot, qt_theme_applied, no_handover, sibling):
    """Every field, not only the first, reads its draft from ``<root>/masks``."""
    labels = np.zeros((IMG_N, IMG_N), dtype=np.uint16)
    labels[200:240, 20:60] = 7
    imageio.imwrite(sibling / "masks" / "s_1.tif", labels)
    screen = handed_screen(qtbot, sibling)

    screen._btn_next.click()

    assert screen._image_files[screen._current_index] == "s_1.tif"
    assert np.array_equal(screen._canvas.mask, labels)


def test_a_sibling_recrop_writes_the_childs_mask_beside_the_images(
        qtbot, qt_theme_applied, no_handover, sibling):
    """The child's image joins ``images/``; its mask joins ``<root>/masks``."""
    screen = handed_screen(qtbot, sibling)
    screen._btn_recrop.click()
    box_drag(screen._canvas, *BOX_A)

    assert (sibling / "images" / "s_0__r00.tif").is_file()
    assert (sibling / "masks" / "s_0__r00.tif").is_file()
    assert not (sibling / "images" / "masks").exists()

    screen._btn_next.click()

    archive = sibling / "images" / engine.RECROP_ARCHIVE_DIRNAME
    assert (archive / "s_0.tif").is_file()
    assert (archive / "masks" / "s_0.tif").is_file(), (
        "the parent's mask was left in <root>/masks after the parent moved")
    assert not (sibling / "masks" / "s_0.tif").exists()
    layout = detect_layout(sibling)
    assert layout.kind == LAYOUT_SIBLING
    assert [item.stem for item in layout.items] == ["s_0__r00", "s_1"]


def test_masking_the_whole_folder_is_refused_in_a_sibling_session(
        qtbot, qt_theme_applied, no_handover, sibling, monkeypatch):
    """Apply writes ``<src>/masks``: here that is ``images/masks``, not the set's."""
    screen = handed_screen(qtbot, sibling)
    opened = []
    monkeypatch.setattr(screen, "open_folded",
                        lambda key: opened.append(key))

    assert screen.mask_whole_folder() is False
    assert opened == []
    assert "not started" in screen._status_label.text()


# ===========================================================================
# The seg layout, through the screen
# ===========================================================================

def test_a_seg_session_shows_the_bundles_image_and_labels(
        qtbot, qt_theme_applied, no_handover, seg):
    """Each field is a bundle, drawn from the ``img`` and ``masks`` it holds."""
    screen = handed_screen(qtbot, seg)

    assert screen._folder == str(seg)
    assert screen._image_files == ["b_0_seg.npy", "b_1_seg.npy"]
    _image, draft = three_object_field()
    assert np.array_equal(screen._canvas.mask, draft)
    assert screen._canvas.image is not None
    assert screen._canvas.image.shape == (IMG_N, IMG_N)


def test_a_seg_save_writes_the_labels_back_into_the_bundle(
        qtbot, qt_theme_applied, no_handover, seg):
    """Erase, Save: the bundle changes, every other key is kept."""
    screen = handed_screen(qtbot, seg)
    screen._btn_del_obj.click()
    click(screen._canvas, *OBJECT_ONE)
    screen._btn_save.click()

    saved = bundle(seg / "b_0_seg.npy")
    assert 1 not in np.unique(saved["masks"])
    assert {2, 3} <= set(np.unique(saved["masks"]).tolist())
    assert saved["filename"] == "b_0.png"
    assert saved["diameter"] == 31.5
    assert saved["flows"] == [None, None, None]
    assert saved["img"].dtype == np.uint8, "the embedded image was rewritten"
    outlined = set(np.unique(saved["outlines"]).tolist())
    assert outlined == {0, 2, 3}, "the outlines still describe the old masks"
    assert not (seg / "masks").exists(), "a TIFF masks folder was started"
    assert detect_layout(seg).kind == LAYOUT_SEG
    assert not list(seg.glob(".*.tmp")), "the atomic write left its temp file"

    rows = read_status(seg)
    assert set(rows) == {"b_0"}, (
        f"status keyed on {sorted(rows)}; the queue calls the field b_0")
    assert rows["b_0"].state == "done"
    assert [item.stem for item in build_queue(
        seg, order="name", cache_counts=False).items] == ["b_1"]


def test_a_seg_recrop_writes_a_bundle_and_keeps_the_folder_one_layout(
        qtbot, qt_theme_applied, no_handover, seg):
    """A child of a bundle is a bundle; the parent goes to the archive."""
    screen = handed_screen(qtbot, seg)
    screen._btn_recrop.click()
    box_drag(screen._canvas, *BOX_A)

    child = seg / "b_0__r00_seg.npy"
    assert child.is_file()
    assert screen._image_files[1] == "b_0__r00_seg.npy"
    written = bundle(child)
    assert set(np.unique(written["masks"]).tolist()) == {0, 1}
    assert written["img"].shape == written["masks"].shape
    assert written["filename"].startswith("recrop of b_0 [")
    assert not list(seg.glob("*.tif")), "a TIFF child made the folder nested"

    screen._btn_next.click()

    archive = seg / engine.RECROP_ARCHIVE_DIRNAME
    assert (archive / "b_0_seg.npy").is_file()
    assert not (seg / "b_0_seg.npy").exists()
    layout = detect_layout(seg)
    assert layout.kind == LAYOUT_SEG
    assert [item.stem for item in layout.items] == ["b_0__r00", "b_1"]
    assert screen._image_files[screen._current_index] == "b_0__r00_seg.npy"
    assert screen._canvas.mask is not None


def test_a_bundle_is_not_handed_to_a_module_that_opens_mask_files(
        qtbot, qt_theme_applied, no_handover, seg):
    """Curate and napari read image and mask FILES; a bundle is neither."""
    screen = handed_screen(qtbot, seg)

    assert screen._seed_mask_editor(object(), "curate") == {}
    assert "bundle" in screen._status_label.text()


# ===========================================================================
# What the session had to say stays on screen
# ===========================================================================

#: A side at which a bundle holding the image and its labels crosses the
#: screen's 8 MB background-load threshold, as every real bundle does: the
#: two curation sets measured on 2026-09-19 hold 12 to 17 MB bundles.
BIG_N = 2100
#: The notice a queue of two fields carries when the scores name only one.
UNSCORED = "1 of 2 field(s) have no probability"


def _big_bundle(folder: Path, name: str) -> Path:
    """A bundle big enough to load off the GUI thread.

    It holds one real object and one four-pixel speck, which a minimum-area
    bound of ten drops when the field loads -- and that drop writes its own
    sentence to the status line, the second thing a notice has to outlast.
    """
    image = np.zeros((BIG_N, BIG_N), dtype=np.uint16)
    labels = np.zeros((BIG_N, BIG_N), dtype=np.uint16)
    image[100:200, 100:200] = 30000
    labels[100:200, 100:200] = 1
    image[400:402, 400:402] = 30000
    labels[400:402, 400:402] = 2
    path = folder / f"{name}_seg.npy"
    np.save(path, {"img": image, "masks": labels,
                   "outlines": np.zeros_like(labels)}, allow_pickle=True)
    return path


def test_the_sessions_notices_are_on_the_status_line_when_it_opens(
        qtbot, qt_theme_applied, no_handover, seg):
    """A small first field: the field is named and so is the notice."""
    queue = build_queue(seg, order="prob", probs={"b_0": 0.9},
                        cache_counts=False)
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)

    assert screen.open_queue(queue)

    text = screen._status_label.text()
    assert not screen._loading
    assert "b_0_seg.npy" in text
    assert UNSCORED in text

    screen._btn_next.click()

    text = screen._status_label.text()
    assert "b_1_seg.npy" in text
    assert UNSCORED not in text, "the opening notice is repeated on every field"


def test_the_notices_outlast_a_first_field_that_loads_in_the_background(
        qtbot, qt_theme_applied, no_handover, tmp_path):
    """Review of 2026-09-19: a 35 MB bundle left only ``f0_seg.npy (1/2)``.

    Every real bundle is over the 8 MB line, so the field arrives from the
    loader thread AFTER :meth:`open_queue` has returned, and the status line
    it wrote used to replace the notice within a fraction of a second. The
    load-time size filter then writes over the field's own line as well.
    """
    folder = tmp_path / "big"
    folder.mkdir()
    big = _big_bundle(folder, "a_big")
    assert big.stat().st_size >= 8 * 1024 * 1024, (
        "the fixture no longer crosses the background-load threshold")
    image, mask = three_object_field()
    np.save(folder / "b_small_seg.npy", {"img": image, "masks": mask},
            allow_pickle=True)
    queue = build_queue(folder, order="prob", probs={"a_big": 0.9},
                        cache_counts=False)
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    screen._filter_min_area.setValue(10)

    assert screen.open_queue(queue)

    assert screen._loading, (
        "the first field loaded on the GUI thread; this test is about the "
        "other path")
    assert UNSCORED in screen._status_label.text()
    qtbot.waitUntil(lambda: screen._load_worker is None, timeout=30000)

    text = screen._status_label.text()
    assert screen._canvas.mask is not None
    assert screen._canvas.mask.shape == (BIG_N, BIG_N)
    assert "filter removed 1 object" in text, text
    assert UNSCORED in text, text


def test_a_folder_opened_before_the_notice_lands_does_not_inherit_it(
        qtbot, qt_theme_applied, no_handover, tmp_path, sibling):
    """The notice belongs to the session; Browse ends the session."""
    folder = tmp_path / "big"
    folder.mkdir()
    _big_bundle(folder, "a_big")
    image, mask = three_object_field()
    np.save(folder / "b_small_seg.npy", {"img": image, "masks": mask},
            allow_pickle=True)
    queue = build_queue(folder, order="prob", probs={"a_big": 0.9},
                        cache_counts=False)
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    assert screen.open_queue(queue)
    assert screen._loading

    assert screen._open_folder(str(sibling / "images"),
                               masks_dir=str(sibling / "masks"))
    qtbot.waitUntil(lambda: screen._load_worker is None, timeout=30000)

    assert screen._queue is None
    assert UNSCORED not in screen._status_label.text()


# ===========================================================================
# The real launch path, for the layouts that used to be refused
# ===========================================================================

@pytest.mark.parametrize("layout", ["sibling", "seg"])
def test_the_command_opens_the_window_on_either_layout(
        qtbot, qt_theme_applied, no_handover, request, layout):
    """``spacr-make-masks`` leaves the queue and builds the main window."""
    from spacr.qt.app import MainWindow

    folder = request.getfixturevalue(layout)
    cli_make_masks.hand_over(build_queue(folder, order="name",
                                         cache_counts=False))
    window = MainWindow(initial_app="make_masks")
    qtbot.addWidget(window)
    try:
        screen = window._screens.get("make_masks")
        assert screen is not None
        assert screen._queue is not None
        assert screen._queue.layout.kind == layout
        _image, draft = three_object_field()
        assert np.array_equal(screen._canvas.mask, draft)
    finally:
        window.close()


# ===========================================================================
# The engine, where the claim is a file on disk
# ===========================================================================

def test_without_the_masks_folder_a_sibling_draft_reads_as_empty(sibling):
    """The failure the refusal existed for, kept visible as a baseline."""
    images = str(sibling / "images")
    _image, wrong = engine.load_image_and_mask(images, "s_0.tif")
    _image, right = engine.load_image_and_mask(
        images, "s_0.tif", masks_dir=str(sibling / "masks"))

    assert not wrong.any()
    assert right.any()
    assert engine.mask_save_path(images, "s_0.tif",
                                 masks_dir=str(sibling / "masks")) == str(
        sibling / "masks" / "s_0.tif")


def test_a_bundle_prefers_its_original_found_by_name_beside_it(
        seg, monkeypatch):
    """``source_image`` is looked for BY NAME, never at its stored path.

    The stored path is absolute and from whichever machine staged the set;
    a stat on another machine's mount is the kind of call that hangs the GUI
    thread. So the stored directory must never be asked about at all.
    """
    original = np.full((IMG_N, IMG_N), 1234, dtype=np.uint16)
    imageio.imwrite(seg / "raw_b_0.tif", original)
    path = seg / "b_0_seg.npy"
    payload = bundle(path)
    payload["source_image"] = "/elsewhere/machine/raw_b_0.tif"
    np.save(path, payload, allow_pickle=True)
    asked = []
    real_isfile = os.path.isfile
    monkeypatch.setattr(os.path, "isfile",
                        lambda p: asked.append(str(p)) or real_isfile(p))

    image, _mask = engine.load_seg_bundle(str(path))

    assert np.all(image == 1234), "the embedded uint8 img was shown instead"
    assert not any(p.startswith("/elsewhere") for p in asked)


@pytest.mark.parametrize("where", ["new_originals", "training_data"])
def test_a_bundle_finds_its_original_where_the_external_tool_kept_them(
        seg, where):
    """``new_originals/`` and ``training_data/`` beside the queue come first.

    The external tool looked for the original by name there, then beside the
    bundle. Without the first two, a set whose originals travel in them was
    shown as its 8-bit embedded copy, where that tool showed the original.
    """
    imageio.imwrite(seg / "raw_b_0.tif",
                    np.full((IMG_N, IMG_N), 1234, dtype=np.uint16))
    (seg.parent / where).mkdir()
    imageio.imwrite(seg.parent / where / "raw_b_0.tif",
                    np.full((IMG_N, IMG_N), 4321, dtype=np.uint16))
    path = seg / "b_0_seg.npy"
    payload = bundle(path)
    payload["source_image"] = "/elsewhere/machine/raw_b_0.tif"
    np.save(path, payload, allow_pickle=True)

    image, _mask = engine.load_seg_bundle(str(path))

    assert np.all(image == 4321), f"the original in {where}/ was passed over"


def test_new_originals_is_searched_before_training_data(seg):
    """The external tool's order, so both tools show the same pixels."""
    for where, value in (("new_originals", 11), ("training_data", 22)):
        (seg.parent / where).mkdir()
        imageio.imwrite(seg.parent / where / "raw_b_0.tif",
                        np.full((IMG_N, IMG_N), value, dtype=np.uint16))
    path = seg / "b_0_seg.npy"
    payload = bundle(path)
    payload["source_image"] = "raw_b_0.tif"
    np.save(path, payload, allow_pickle=True)

    image, _mask = engine.load_seg_bundle(str(path))

    assert np.all(image == 11)


def test_a_bundle_with_no_img_uses_the_display_image_of_its_stem(seg):
    """The external tool wrote ``<stem>.png`` beside each bundle."""
    path = seg / "b_1_seg.npy"
    payload = bundle(path)
    del payload["img"]
    np.save(path, payload, allow_pickle=True)
    imageio.imwrite(seg / "b_1.png", np.full((IMG_N, IMG_N), 9, np.uint8))

    image, mask = engine.load_image_and_mask(str(seg), "b_1_seg.npy")

    assert image.shape == mask.shape == (IMG_N, IMG_N)
    assert mask.any()


def test_a_bundle_with_nothing_to_draw_on_says_so(seg):
    """No ``img``, no original, no display image: a sentence, not a crash."""
    path = seg / "b_1_seg.npy"
    payload = bundle(path)
    del payload["img"]
    np.save(path, payload, allow_pickle=True)

    with pytest.raises(ValueError, match="nothing to draw"):
        engine.load_seg_bundle(str(path))


def test_a_file_that_is_not_a_bundle_is_refused_by_name(tmp_path):
    """A plain array saved under the bundle suffix is not a bundle."""
    path = tmp_path / "x_seg.npy"
    np.save(path, np.zeros((4, 4)))

    with pytest.raises(ValueError, match="not a Cellpose _seg.npy bundle"):
        engine.read_seg_bundle(str(path))


def test_saving_a_bundle_keeps_its_keys_and_updates_the_derived_ones(seg):
    """``ismanual`` keeps its flags and grows; ``outlines`` is redrawn."""
    path = seg / "b_0_seg.npy"
    _image, mask = three_object_field()
    edited = mask.copy()
    edited[2:12, 200:210] = 4

    engine.save_mask(str(seg), "b_0_seg.npy", edited)

    saved = bundle(path)
    assert set(np.unique(saved["masks"]).tolist()) == {0, 1, 2, 3, 4}
    assert saved["ismanual"].tolist() == [False, True, False, True]
    assert saved["outlines"][24, 30] == 1, "top edge of object 1"
    assert saved["outlines"][40, 40] == 0, "inside object 1"
    assert saved["diameter"] == 31.5


def test_binary_outlines_stay_binary(seg):
    """An ``outlines`` entry of 0/1 is redrawn as 0/1, in its own type."""
    path = seg / "b_1_seg.npy"
    payload = bundle(path)
    payload["outlines"] = np.zeros((IMG_N, IMG_N), dtype=bool)
    payload["outlines"][0, 0] = True
    np.save(path, payload, allow_pickle=True)
    _image, mask = three_object_field()

    engine.save_seg_bundle(str(path), mask)

    outlines = bundle(path)["outlines"]
    assert outlines.dtype == bool
    assert outlines[24, 30] and not outlines[40, 40] and not outlines[0, 0]


def test_a_retired_bundle_takes_its_ledger_and_display_image_with_it(seg):
    """The external tool moved the bundle and its ``.png``; so does this."""
    (seg / "b_0.png").write_bytes(b"display copy")
    (seg / ("b_0_seg.npy" + LOG_SUFFIX)).write_text("{}", encoding="utf-8")

    record = engine.retire_recropped_original(
        str(seg), "b_0_seg.npy", children=["b_0__r00_seg.npy"],
        boxes=[(1, 2, 3, 4)])

    archive = seg / engine.RECROP_ARCHIVE_DIRNAME
    for name in ("b_0_seg.npy", "b_0_seg.npy" + LOG_SUFFIX, "b_0.png"):
        assert (archive / name).is_file(), name
        assert not (seg / name).exists(), name
    assert len(record["moved"]) == 3
    assert engine.restore_recropped_original(str(seg), "b_0_seg.npy")
    assert (seg / "b_0_seg.npy").is_file()


def test_the_field_stem_is_the_queues_stem():
    """``splitext`` would give ``well_seg``; the status row says ``well``."""
    assert engine.field_stem("well_seg.npy") == "well"
    assert engine.field_stem("/a/b/well.tif") == "well"
    assert engine.field_stem("well__r02_seg.npy") == "well__r02"
    assert engine.recrop_child_name("/nowhere", "well__r02_seg.npy") == (
        "well__r00_seg.npy")
