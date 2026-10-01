"""Item 419, points 1 to 3: the readout, the layout, and installable models.

The maintainer's words, 2026-09-16:

    "1. In make masks, hovering should show intensity and object area, etc.,
    e.g. in the top left corner.

    2. the settings categorys should be to the left of the image and there
    should be a slight distance between the image and the settings. and the
    text to the right of the Make Masks "correct a m..." which has a tooltip
    should be removed. and the settings toggle button should be directly to
    the right of the magnifier button.

    3. for the new models, these should be grayed out in the model zoo button
    and clickable to install them in the spacr environment. [...] these models
    should also be available in the mask modual in the same way"

Every test drives the screen the way a user reaches it -- a mouse move over
the canvas, a click on the Filter button, a row chosen in a box -- and reads
the result back off the built widgets. The installs run a real child process
and a real download (a ``file://`` model and a stand-in for ``pip``), so the
claim that the window keeps working while one runs is measured, not assumed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm
from tests.qt.test_model_install_runs_off_the_gui_thread import (
    _choose,
    _fake_backend_install,
    _fake_pip,
    no_backend_environments,  # noqa: F401 - a fixture, used by name
    no_backends,  # noqa: F401 - a fixture, used by name
)
from tests.qt.test_the_live_magnifier_segments_under_the_mouse import (
    CANVAS_H,
    CANVAS_W,
    PIXMAP_N,
    canvas_xy,
    coded_field,
    rect_mask,
)

#: The objects on the test field, ``id -> (x0, y0, x1, y1)``, exclusive ends.
#: Object 9 is written twice, as a large and a small piece, the shape a brush
#: stroke beside a real object leaves.
OBJECTS = {3: (4, 4, 14, 12), 5: (30, 20, 42, 36), 9: (50, 44, 60, 58)}
SPLIT_PIECE = (6, 50, 9, 53)


def _mouse(kind, img_x, img_y, button=Qt.NoButton, buttons=Qt.NoButton):
    pos = QPointF(*canvas_xy(img_x, img_y))
    return QMouseEvent(kind, pos, pos, button, buttons, Qt.NoModifier)


def hover(screen, img_x, img_y, buttons=Qt.NoButton):
    screen._canvas.mouseMoveEvent(
        _mouse(QEvent.Type.MouseMove, img_x, img_y, buttons=buttons))


@pytest.fixture
def labelled(tmp_path: Path) -> Path:
    """One coded field with three objects, one of them in two pieces."""
    folder = tmp_path / "labelled"
    (folder / "masks").mkdir(parents=True)
    imageio.imwrite(folder / "a.tif", coded_field())
    mask = rect_mask((64, 64), OBJECTS, dtype=np.uint16)
    x0, y0, x1, y1 = SPLIT_PIECE
    mask[y0:y1, x0:x1] = 9
    imageio.imwrite(folder / "masks" / "a.tif", mask)
    return folder


@pytest.fixture
def screen(qtbot, qt_theme_applied, labelled: Path):
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    assert made._open_folder(str(labelled))
    made._canvas.resize(CANVAS_W, CANVAS_H)
    made._canvas.refresh()
    assert made._canvas.pixmap().width() == PIXMAP_N, "geometry assumption broke"
    yield made
    made._magnifier.close()
    made.close_folded()


# ---------------------------------------------------------------------------
# 1. The readout
# ---------------------------------------------------------------------------

def _expected(object_id):
    """Area and mean of a rectangle of the coded field, worked out by hand."""
    x0, y0, x1, y1 = OBJECTS[object_id]
    ys, xs = np.mgrid[y0:y1, x0:x1]
    return (x1 - x0) * (y1 - y0), float((ys * 64 + xs).mean())


def test_hovering_an_object_reads_out_its_pixel_label_area_and_mean(screen):
    """The pixel under the mouse, and the object it belongs to."""
    hover(screen, 35, 25)
    readout = screen._canvas.readout
    area, mean = _expected(5)
    assert (readout.x, readout.y) == (35, 25)
    assert readout.intensity == 25 * 64 + 35, "not the raw pixel value"
    assert (readout.label, readout.area) == (5, area)
    assert readout.mean_intensity == pytest.approx(mean)

    text = screen._canvas.readout_text()
    assert "x 35, y 25" in text and str(25 * 64 + 35) in text
    assert "Object 5" in text and f"area {area} px" in text
    assert f"{mean:.2f}".rstrip("0").rstrip(".") in text

    hover(screen, 20, 40)
    assert screen._canvas.readout.label == 0, "background named an object"
    assert "Object" not in screen._canvas.readout_text()


def test_the_readout_is_painted_in_the_images_top_left_corner(
        screen, qtbot):
    """Read off the rendered canvas, not off the attribute."""
    canvas = screen._canvas
    screen.resize(1200, 800)
    screen.show()
    qtbot.waitExposed(screen)
    canvas.resize(CANVAS_W, CANVAS_H)
    canvas.refresh()
    hover(screen, 35, 25)
    rect = canvas.readout_rect()
    assert rect is not None
    left = (canvas.width() - PIXMAP_N) // 2
    assert left <= rect.left() < left + 12, "not at the image's left edge"
    assert 0 <= rect.top() < 12, "not at the image's top edge"

    shown = canvas.grab().toImage()
    canvas.update_readout(None)
    bare = canvas.grab().toImage()
    ratio = shown.devicePixelRatio()
    probe = rect.center()
    differs = [
        shown.pixel(int((probe.x() + dx) * ratio), int((probe.y() + dy) * ratio))
        != bare.pixel(int((probe.x() + dx) * ratio), int((probe.y() + dy) * ratio))
        for dx in range(-20, 21, 4) for dy in (-3, 0, 3)]
    assert any(differs), "nothing was painted where the readout belongs"


def test_the_readout_predicts_what_the_filter_removes(screen):
    """"so a user can predict what a filter will remove", through the button.

    A minimum intensity set to the mean the readout shows keeps the object; a
    hundredth more removes it, and the ledger names the id the readout gave.
    """
    hover(screen, 8, 8)
    readout = screen._canvas.readout
    assert readout.label == 3
    screen._filter_list.set_filter("intensity_mean", readout.mean_intensity)
    assert screen._filter_list.filters()[0]["min"] == pytest.approx(
        readout.mean_intensity, rel=1e-9), (
        "the test's mean needs no rounding to fit the box")
    screen._btn_filter.click()
    assert (screen._canvas.mask == 3).any(), "a bound AT the mean removed it"

    screen._filter_list.set_filter("intensity_mean", readout.mean_intensity + 0.01)
    screen._btn_filter.click()
    assert not (screen._canvas.mask == 3).any()
    removed = screen._log.edits[-1]
    assert removed.kind == "filter" and 3 in list(removed.target)


def test_the_readout_measures_exactly_what_the_filter_measures(screen):
    """Every object, compared with the regionprops call the filter makes."""
    from skimage.measure import regionprops

    lookup = engine.ObjectLookup(screen._canvas.mask, screen._canvas.image)
    labels = engine.canonical_labels(screen._canvas.mask)
    grey = np.asarray(screen._canvas.image, dtype=np.float32)
    regions = regionprops(labels.astype(np.int32), intensity_image=grey)
    assert len(regions) == 4
    for region in regions:
        assert lookup.measure(region.label) == (
            int(region.area), float(region.intensity_mean))


def test_a_split_label_is_read_out_under_the_id_the_filter_gives_it(screen):
    """The small piece of object 9 is its own object to the filter."""
    x0, y0, x1, y1 = SPLIT_PIECE
    hover(screen, x0 + 1, y0 + 1)
    readout = screen._canvas.readout
    canonical = engine.canonical_labels(screen._canvas.mask)
    assert readout.label == int(canonical[y0 + 1, x0 + 1])
    assert readout.label not in (0, 3, 5, 9)
    assert readout.area == (x1 - x0) * (y1 - y0)
    hover(screen, 55, 50)
    assert screen._canvas.readout.label == 9


def test_the_readout_follows_an_edit_without_the_mouse_moving(screen, qtbot):
    """An object erased under a resting mouse stops being reported."""
    hover(screen, 35, 25)
    assert screen._canvas.readout.label == 5
    screen._set_mode(mm.MODE_ERASE_OBJECT)
    press = _mouse(QEvent.Type.MouseButtonPress, 35, 25, Qt.LeftButton,
                   Qt.LeftButton)
    release = _mouse(QEvent.Type.MouseButtonRelease, 35, 25, Qt.LeftButton,
                     Qt.NoButton)
    screen._canvas.mousePressEvent(press)
    screen._canvas.mouseReleaseEvent(release)
    assert not (screen._canvas.mask == 5).any()
    qtbot.waitUntil(lambda: screen._canvas.readout.label == 0, timeout=3000)
    assert screen._canvas.readout.intensity == 25 * 64 + 35


def test_a_held_button_keeps_the_readout_to_the_pixel(screen):
    """A stroke changes the mask on every move; the object waits for release."""
    hover(screen, 35, 25, buttons=Qt.LeftButton)
    readout = screen._canvas.readout
    assert (readout.x, readout.y, readout.intensity) == (35, 25, 25 * 64 + 35)
    assert readout.label == 0 and readout.mean_intensity is None


def test_the_readout_goes_when_the_mouse_leaves(screen):
    hover(screen, 35, 25)
    screen._canvas.leaveEvent(QEvent(QEvent.Type.Leave))
    assert screen._canvas.readout is None
    assert screen._canvas.readout_text() == ""
    assert screen._canvas.readout_rect() is None


def test_the_readout_works_with_the_magnifier_on(screen):
    """The magnifier takes the mouse move; the readout still follows it."""
    screen._magnifier.segment = lambda request: np.zeros(
        request.crop.shape, np.int32)
    screen._btn_magnifier.setChecked(True)
    hover(screen, 35, 25)
    assert screen._canvas.readout.label == 5


# ---------------------------------------------------------------------------
# 2. The layout
# ---------------------------------------------------------------------------

def test_the_settings_are_left_of_the_image_with_a_gap(screen, qtbot):
    """Measured on a shown, laid-out screen."""
    screen.resize(1400, 900)
    screen.show()
    qtbot.waitExposed(screen)
    QApplication.processEvents()
    settings = screen._settings_scroll
    image = screen._view_tabs
    assert screen._body_splitter.indexOf(settings) == 0
    assert screen._body_splitter.indexOf(screen._view_pane) == 1
    assert screen._view_pane.isAncestorOf(image)
    right_of_settings = settings.mapTo(screen, settings.rect().topRight()).x()
    left_of_image = image.mapTo(screen, image.rect().topLeft()).x()
    gap = left_of_image - right_of_settings - 1
    assert gap >= 8, f"only {gap}px between the settings and the image"
    assert gap == screen._body_splitter.handleWidth() == mm.SETTINGS_GAP


def test_the_masthead_carries_no_sentence_beside_the_name(screen):
    """"the text to the right of the Make Masks ... should be removed"."""
    from PySide6.QtWidgets import QLabel

    header = screen._header
    assert header.description_label is None
    texts = [label.text() for label in header.findChildren(QLabel)]
    assert not any("Correct a mask" in text for text in texts), texts
    assert header.title_label.text() == mm.HEADER_TITLE


def test_the_settings_toggle_sits_directly_right_of_the_magnifier(
        screen, qtbot):
    row = screen._tool_pin_layout
    magnifier = row.indexOf(screen._btn_magnifier)
    assert magnifier >= 0
    assert row.indexOf(screen._btn_settings) == magnifier + 1

    screen.resize(1600, 900)
    screen.show()
    qtbot.waitExposed(screen)
    QApplication.processEvents()
    gap = (screen._btn_settings.geometry().left()
           - screen._btn_magnifier.geometry().right() - 1)
    assert gap == row.spacing(), f"{gap}px between Magnifier and Settings"


def test_an_action_added_later_does_not_come_between_the_pair(screen):
    from PySide6.QtWidgets import QPushButton

    row = screen._tool_pin_layout
    added = screen.add_toolbar_action(QPushButton("Another action"))
    assert row.indexOf(added) < 0
    assert screen._tool_row_layout.indexOf(added) >= 0
    assert row.indexOf(screen._btn_settings) == \
        row.indexOf(screen._btn_magnifier) + 1


def test_the_toggle_still_gives_the_image_the_settings_width(screen, qtbot):
    screen.resize(1400, 900)
    screen.show()
    qtbot.waitExposed(screen)
    qtbot.waitUntil(lambda: screen._canvas.width() > 1)
    before = screen._canvas.width()
    width = screen._body_splitter.sizes()[0]
    screen._btn_settings.setChecked(False)
    qtbot.waitUntil(lambda: screen._canvas.width() > before)
    assert screen._body_splitter.sizes()[0] == 0
    screen._btn_settings.setChecked(True)
    qtbot.waitUntil(lambda: screen._body_splitter.sizes()[0] > 0)
    assert abs(screen._body_splitter.sizes()[0] - width) <= 8


# ---------------------------------------------------------------------------
# 3. Models that are not installed: greyed, and installed by a click
#
# The maintainer asked on 2026-09-16 for a greyed row that installs itself
# "in the spacr environment"; on 2026-09-19, answering item 423, he said
# where instead: "Isolated env per backend!". So the Mode box still greys
# what is missing and still installs it when it is chosen, but through the
# Model Zoo's own dialog, into ~/.spacr/backends/<name>, and spaCR's own
# environment is not changed. The dialog itself -- its progress, its Cancel,
# and the event loop still running while pip does -- is driven in
# tests/qt/test_a_backend_installs_into_its_own_environment.py; here the
# Mode box's end of it is what is pressed.
#
# spacr.qt.model_install.PackageInstall is unchanged and still installs the
# Mask settings' segmentation_backend dropdown; that surface is item 419's
# and is tested in tests/qt/test_model_install_runs_off_the_gui_thread.py.
# ---------------------------------------------------------------------------

def test_a_missing_backend_installs_into_its_own_environment_and_is_selected(
        qtbot, qt_theme_applied, monkeypatch, no_backends,
        no_backend_environments):
    asked = _fake_backend_install(monkeypatch)
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    try:
        box = made._mag_mode
        row = box.findData("samcell")
        assert box.itemData(row, Qt.ForegroundRole) is not None, "not greyed"
        assert box.model().item(row).isEnabled(), "greyed must stay choosable"

        _choose(box, "samcell")

        assert asked == ["samcell"], "the Model Zoo's installer was not used"
        assert "samcell" not in made._mag_uninstalled
        assert box.currentData() == "samcell", (
            "an install that succeeded did not select the row")
        assert box.itemData(box.findData("samcell"),
                            Qt.ForegroundRole) is None
        assert box.itemData(box.findData("dinocell"),
                            Qt.ForegroundRole) is not None, (
            "installing one backend un-greyed another")
        assert "SAMCell is installed" in made._status_label.text()
    finally:
        made._magnifier.close()
        made.close_folded()


def test_installing_cellpose3_lights_all_four_of_its_models(
        qtbot, qt_theme_applied, monkeypatch, no_backends,
        no_backend_environments):
    """One environment carries all four models, so one install lights them."""
    asked = _fake_backend_install(monkeypatch)
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    try:
        _choose(made._mag_mode, "cellpose3:cyto2")
        assert asked == ["cellpose3"]
        for mode in ("cellpose3:cyto3", "cellpose3:cyto2",
                     "cellpose3:cyto", "cellpose3:nuclei"):
            assert mode not in made._mag_uninstalled, mode
        assert "samcell" in made._mag_uninstalled
        assert made._mag_mode.currentData() == "cellpose3:cyto2"
    finally:
        made._magnifier.close()
        made.close_folded()


def test_an_install_that_did_not_happen_leaves_the_box_working(
        qtbot, qt_theme_applied, monkeypatch, no_backends,
        no_backend_environments):
    """Cancelled, refused or failed -- all of them are the dialog saying no."""
    asked = _fake_backend_install(monkeypatch, answer=False)
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    try:
        _choose(made._mag_mode, "dinocell")
        assert asked == ["dinocell"]
        assert "dinocell" in made._mag_uninstalled
        assert made._mag_mode.currentData() == "otsu"
        assert made._mag_mode.itemData(made._mag_mode.findData("dinocell"),
                                       Qt.ForegroundRole) is not None
        _choose(made._mag_mode, "otsu")
        assert made._magnifier.mode == "otsu"
    finally:
        made._magnifier.close()
        made.close_folded()


def test_a_missing_mode_never_reaches_the_magnifier(
        qtbot, qt_theme_applied, monkeypatch, no_backends,
        no_backend_environments):
    """From Cellpose, a click on a missing SAMCell goes back to Cellpose.

    The box changes its row before ``activated`` fires; the magnifier must
    not be handed SAMCell in between, or it starts a load that can only fail.
    """
    _fake_backend_install(monkeypatch, answer=False)
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    seen = []
    real = made._magnifier.set_mode
    monkeypatch.setattr(made._magnifier, "set_mode",
                        lambda mode: seen.append(mode) or real(mode))
    try:
        _choose(made._mag_mode, "cellpose")
        _choose(made._mag_mode, "samcell")
        assert made._mag_mode.currentData() == "cellpose"
        assert made._magnifier.mode == "cellpose"
        assert "samcell" not in seen
    finally:
        made._magnifier.close()
        made.close_folded()


def test_nothing_in_the_mode_box_runs_pip_against_spacrs_own_environment(
        qtbot, qt_theme_applied, monkeypatch, no_backends,
        no_backend_environments):
    """The maintainer's 2026-09-19 answer, held down where it was broken.

    Choosing a greyed row used to run `pip install "spacr[<backend>]"`
    against the interpreter spaCR is running in.
    """
    started = _fake_pip(monkeypatch)
    _fake_backend_install(monkeypatch, answer=False)
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    try:
        for mode in ("samcell", "dinocell", "cellpose3:cyto3"):
            _choose(made._mag_mode, mode)
        assert started == [], f"pip was run against spaCR's own environment: {started}"
    finally:
        made._magnifier.close()
        made.close_folded()


@pytest.fixture(params=["toxo_nuclei_r2.cp_model", "toxo_nuclei_v1.cp_model"])
def zoo_model(request, tmp_path: Path, monkeypatch):
    """A zoo Cellpose model published as a ``file://`` URI with its digest.

    The second name ends in ``_v1``, which the zoo files without the suffix,
    so the file does not land under the entry's own name.
    """
    from spacr import model_zoo

    source = tmp_path / "published" / request.param
    source.parent.mkdir()
    source.write_bytes(b"not really weights, but bytes with a digest" * 4096)
    entry = model_zoo.ModelEntry(
        key=source.stem, name=source.name, kind="cellpose",
        source="remote", uri=source.as_uri(),
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        size_bytes=source.stat().st_size)
    folder = tmp_path / "downloads"

    def listed():
        here = folder / entry.name
        return [(entry.key, str(here) if here.is_file() else None, entry)]

    monkeypatch.setattr(mm, "_zoo_cellpose_models", listed)
    from spacr.qt.widgets import model_zoo_picker
    monkeypatch.setattr(model_zoo_picker, "remembered_model_dir",
                        lambda: str(folder))
    return entry, folder


def test_a_zoo_model_not_downloaded_is_greyed_and_downloads_on_click(
        qtbot, qt_theme_applied, monkeypatch, zoo_model):
    entry, folder = zoo_model
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    monkeypatch.setattr(made, "_confirm", lambda *a: True)
    try:
        box = made._cp_model
        row = next(i for i in range(box.count())
                   if box.itemData(i, mm._ZOO_PENDING_ROLE) is not None)
        assert box.itemData(row, Qt.ForegroundRole) is not None, "not greyed"
        assert box.model().item(row).isEnabled(), "greyed must stay choosable"
        before = box.currentData()

        box.setCurrentIndex(row)
        box.activated.emit(row)
        assert box.currentData() == before, (
            "the box rested on a model that has nothing to load")
        assert not made._cp_download_bar.isHidden()

        qtbot.waitUntil(lambda: made._cp_download is None, timeout=15_000)
        landed = [path for path in folder.iterdir()
                  if not path.name.startswith(".")]
        assert len(landed) == 1, made._status_label.text()
        landed = landed[0]
        assert landed.read_bytes() == Path(entry.uri[len("file://"):]
                                           ).read_bytes()
        assert box.currentData() is not None
        assert Path(box.currentData()).resolve() == landed.resolve()
        assert made._magnifier_context()["model_name"] == box.currentData()
        assert made._cp_download_bar.isHidden()
        assert all(box.itemData(i, mm._ZOO_PENDING_ROLE) is None
                   for i in range(box.count())), "still listed as missing"
    finally:
        made._magnifier.close()
        made.close_folded()


def test_the_model_box_never_rests_on_a_model_not_downloaded(
        qtbot, qt_theme_applied, zoo_model):
    """The keyboard and the wheel change the row without ``activated``."""
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    try:
        box = made._cp_model
        before = box.currentIndex()
        row = next(i for i in range(box.count())
                   if box.itemData(i, mm._ZOO_PENDING_ROLE) is not None)
        box.setCurrentIndex(row)
        assert box.currentIndex() == before
        assert made._cp_download is None, "a download started with no click"
    finally:
        made._magnifier.close()
        made.close_folded()


def test_a_download_that_fails_its_checksum_leaves_nothing_behind(
        qtbot, qt_theme_applied, monkeypatch, zoo_model):
    entry, folder = zoo_model
    from dataclasses import replace

    bad = replace(entry, sha256="0" * 64)
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    monkeypatch.setattr(made, "_confirm", lambda *a: True)
    try:
        before = made._cp_model.currentData()
        assert made.download_zoo_model(bad)
        qtbot.waitUntil(lambda: made._cp_download is None, timeout=15_000)
        assert not any(folder.iterdir()), "a failed download left a file"
        assert "Download failed" in made._status_label.text()
        assert made._cp_model.currentData() == before
    finally:
        made._magnifier.close()
        made.close_folded()
