"""Item 417, the magnifier's second round: its settings and its models.

The maintainer's words, after using item 407's magnifier: the box size's
maximum of 512 "should be able to be as high as the image is high/wide";
"holding shift and scrolling should increase or decrease the size setting";
the settings categories fold "like in the core applications"; with Cellpose
on, "all the cellpose models in the model zoo" and "the model zoo button";
the flow and cell-probability thresholds, and Otsu's threshold correction,
set in the Object detection category, are what the magnifier's detection
uses; and "other computer vision models like a live YOLO or DINOCell".

The canvas geometry and the coded-field stub are item 407's, imported from
its test module, so a pixel here means what it means there.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent

from spacr.qt.screens import make_masks as mm
from tests.qt.test_the_live_magnifier_segments_under_the_mouse import (
    CANVAS_H,
    CANVAS_W,
    IMG_N,
    PIXMAP_N,
    SIZE,
    CodedStub,
    canvas_xy,
    coded_field,
    fields,  # noqa: F401 - a fixture, used by name
    hover,
    screen,  # noqa: F401 - a fixture, used by name
    switch_on,
    wait_for_result,
)


def shift_wheel(screen, img_x, img_y, delta=120, *, modifiers=Qt.ShiftModifier,
                sideways=False):
    """Turn the wheel one notch over image pixel (img_x, img_y)."""
    pos = QPointF(*canvas_xy(img_x, img_y))
    angle = QPoint(delta, 0) if sideways else QPoint(0, delta)
    screen._canvas.wheelEvent(QWheelEvent(
        pos, pos, QPoint(0, 0), angle, Qt.NoButton, modifiers,
        Qt.NoScrollPhase, False))


# ---------------------------------------------------------------------------
# 1. The box can be as large as the image
# ---------------------------------------------------------------------------

@pytest.fixture
def wide_then_square(tmp_path: Path) -> Path:
    """Two fields: 40 x 700 (wider than the old 512 cap), then 64 x 64."""
    folder = tmp_path / "shapes"
    folder.mkdir()
    imageio.imwrite(folder / "a_wide.tif", np.ones((40, 700), np.uint16))
    imageio.imwrite(folder / "b_square.tif", coded_field())
    return folder


def test_the_size_reaches_the_open_images_longer_side_and_stops_there(
        qtbot, qt_theme_applied, wide_then_square):
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    try:
        size, magnifier = made._mag_size, made._magnifier
        assert (size.minimum(), size.maximum()) == mm._MAGNIFIER_SIZE_RANGE, (
            "with no field open the range is the old one")

        assert made._open_folder(str(wide_then_square))
        assert made._canvas.image.shape == (40, 700)
        assert (size.minimum(), size.maximum()) == (32, 700)
        size.setValue(700)
        assert size.value() == 700 and magnifier.size == 700, (
            "past the old 512 cap, up to the field's own width")
        size.setValue(5000)
        assert size.value() == 700 and magnifier.size == 700
        magnifier.set_size(9999)
        assert magnifier.size == 700, "the magnifier clamps on its own too"

        made._on_next()
        qtbot.waitUntil(lambda: made._canvas.image is not None
                        and made._canvas.image.shape == (IMG_N, IMG_N),
                        timeout=5_000)
        assert size.maximum() == IMG_N, "the range follows the field that opened"
        assert size.value() == IMG_N and magnifier.size == IMG_N, (
            "a size past the new field's side is clamped to it")
    finally:
        made._magnifier.close()
        made.close_folded()


def test_the_range_is_never_empty_on_a_field_smaller_than_the_floor():
    assert mm._magnifier_size_range((20, 12)) == (20, 20)
    assert mm._magnifier_size_range((1, 1)) == (1, 1)
    assert mm._magnifier_size_range((300, 2000, 3)) == (32, 2000)
    assert mm._magnifier_size_range(None) == mm._MAGNIFIER_SIZE_RANGE


# ---------------------------------------------------------------------------
# 8. Shift + wheel changes the size, never the view
# ---------------------------------------------------------------------------

def test_shift_wheel_changes_the_size_and_stops_at_the_images_size(
        qtbot, screen):
    switch_on(screen, CodedStub({}))
    hover(screen, 30, 30)
    magnifier = screen._magnifier
    zoom = magnifier.zoom
    assert magnifier.size == SIZE == 32

    shift_wheel(screen, 30, 30, 120)
    grown = magnifier.size
    assert grown > SIZE
    assert screen._mag_size.value() == grown, "the Size box follows the wheel"
    assert magnifier.zoom == zoom, "Shift + wheel is not the magnifier's zoom"
    assert not screen._canvas.is_zoomed(), "and it is not the view's zoom"

    for _ in range(40):
        shift_wheel(screen, 30, 30, 120)
    assert magnifier.size == IMG_N == screen._mag_size.value(), (
        "it stops at the image's own size")
    assert not screen._canvas.is_zoomed()

    shift_wheel(screen, 30, 30, -120)
    assert magnifier.size < IMG_N
    for _ in range(40):
        shift_wheel(screen, 30, 30, -120)
    assert magnifier.size == 32 == screen._mag_size.value(), (
        "and at the bottom of the range the other way")


def test_a_sideways_shift_notch_counts_and_plain_wheel_is_unchanged(
        qtbot, screen):
    """Some platforms send Shift + wheel as a horizontal scroll."""
    switch_on(screen, CodedStub({}))
    hover(screen, 30, 30)
    magnifier = screen._magnifier

    shift_wheel(screen, 30, 30, 120, sideways=True)
    assert magnifier.size > SIZE

    size, zoom = magnifier.size, magnifier.zoom
    shift_wheel(screen, 30, 30, 120, modifiers=Qt.NoModifier)
    assert magnifier.size == size, "a plain notch leaves the size alone"
    assert magnifier.zoom > zoom, "and still zooms the box, as in item 407"

    screen._btn_magnifier.setChecked(False)
    shift_wheel(screen, 30, 30, 120)
    assert magnifier.size == size
    assert screen._canvas.is_zoomed(), (
        "with the magnifier off the wheel is the view's, Shift or not")


def test_a_shift_wheel_step_is_proportional_and_at_least_a_pixel(
        qtbot, screen):
    magnifier = screen._magnifier
    screen._canvas.zoom_speed = 1.5
    magnifier.set_size(40)
    assert magnifier.wheel_size(True) == 60
    assert magnifier.wheel_size(False) == 32
    screen._canvas.zoom_speed = 1.001
    assert magnifier.wheel_size(True) == 33, "never a notch that does nothing"


# ---------------------------------------------------------------------------
# 2. Every settings category folds like a core application's
# ---------------------------------------------------------------------------

CATEGORIES = ("Brush", "Magic wand", "Display", "Filter",
              "Object operations", "Otsu", "Object detection",
              "Live magnifier")


def _categories(made):
    from spacr.qt.widgets.section import Section

    found = [w for w in made._settings_scroll.findChildren(Section)
             if w.parentWidget() is not None
             and not isinstance(w.parentWidget().parentWidget(), Section)]
    return {title: section for title, section in made._settings_categories
            if section in found}


def test_every_category_is_the_core_applications_folding_section(
        qtbot, qt_theme_applied):
    """The same widget, header and card a core module's settings use."""
    from spacr.qt.widgets import Card
    from spacr.qt.widgets.section import Section

    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    try:
        categories = _categories(made)
        assert tuple(categories) == CATEGORIES
        assert not made._settings_scroll.findChildren(Card), (
            "no unfoldable card is left on the panel")
        owners = {
            "Brush": made._brush_slider, "Magic wand": made._wand_pct,
            "Display": made._norm_hi,
            "Filter": made._filter_min_area,
            "Object operations": made._btn_otsu,
            "Otsu": made._otsu_correction,
            "Object detection": made._cp_flow,
            "Live magnifier": made._mag_size,
        }
        for title, section in categories.items():
            assert type(section) is Section
            assert section.objectName() == "SectionCard"
            assert section.header().objectName() == "SectionHeader"
            assert section.header().text() == title.upper()
            assert section.is_expanded(), f"{title} starts open"
            assert section.isAncestorOf(owners[title])
            assert owners[title].isVisibleTo(made._settings_scroll)

            section.header().click()
            assert not section.is_expanded()
            assert not owners[title].isVisibleTo(made._settings_scroll), (
                f"folding {title} hides what is in it")
            section.header().click()
            assert section.is_expanded()
            assert owners[title].isVisibleTo(made._settings_scroll)
    finally:
        made._magnifier.close()
        made.close_folded()


def test_what_is_folded_is_remembered_for_the_next_visit(
        qtbot, qt_theme_applied):
    first = mm.MakeMasksScreen()
    qtbot.addWidget(first)
    try:
        cats = _categories(first)
        cats["Magic wand"].header().click()
        cats["Live magnifier"].header().click()
    finally:
        first._magnifier.close()
        first.close_folded()

    second = mm.MakeMasksScreen()
    qtbot.addWidget(second)
    try:
        cats = _categories(second)
        shut = {title for title, section in cats.items()
                if not section.is_expanded()}
        assert shut == {"Magic wand", "Live magnifier"}
        assert not second._wand_pct.isVisibleTo(second._settings_scroll)

        cats["Magic wand"].set_expanded(True)
    finally:
        second._magnifier.close()
        second.close_folded()

    third = mm.MakeMasksScreen()
    qtbot.addWidget(third)
    try:
        shut = {title for title, section in _categories(third).items()
                if not section.is_expanded()}
        assert shut == {"Live magnifier"}, "unfolding is remembered as well"
    finally:
        third._magnifier.close()
        third.close_folded()


# ---------------------------------------------------------------------------
# 3. The model zoo's Cellpose models, and the Model zoo button
# ---------------------------------------------------------------------------

@pytest.fixture
def a_zoo(tmp_path, monkeypatch):
    """A zoo with a local Cellpose model, a downloaded remote one, a remote
    one not downloaded, and a detector that Cellpose cannot load."""
    from spacr import model_zoo
    from spacr.model_zoo import ModelEntry
    from spacr.qt.widgets import model_zoo_picker

    downloads = tmp_path / "downloads"
    downloads.mkdir()
    local = tmp_path / "lab_cells_v2.CP_model"
    local.write_bytes(b"weights")
    (downloads / "cpsam_plaque_r3").write_bytes(b"weights")
    entries = [
        ModelEntry(key="lab_cells_v2", name=local.name, kind="cellpose",
                   source="local", path=str(local)),
        ModelEntry(key="toxoplasma_plaque_v1", name="cpsam_plaque_r3",
                   kind="cellpose", source="remote"),
        ModelEntry(key="toxoplasma_pv_v1", name="cpsam_v2_toxo_r2",
                   kind="cellpose", source="remote"),
        ModelEntry(key="toxoplasma_well_detector_v1",
                   name="yolo_welldetect_v3.pt", kind="detector",
                   source="remote"),
    ]
    asked = []

    def catalogue(**kwargs):
        asked.append(kwargs)
        return list(entries)

    monkeypatch.setattr(model_zoo, "catalogue", catalogue)
    monkeypatch.setattr(model_zoo_picker, "remembered_model_dir",
                        lambda: str(downloads))
    return {"local": str(local), "downloaded":
            str(downloads / "cpsam_plaque_r3"), "asked": asked,
            "downloads": downloads}


def _rows(combo):
    return [(combo.itemText(i), combo.itemData(i),
             combo.model().item(i).isEnabled()) for i in range(combo.count())]


def test_the_model_list_is_cpsam_and_every_cellpose_model_in_the_zoo(
        qtbot, qt_theme_applied, a_zoo):
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    try:
        rows = _rows(made._cp_model)
        assert rows[0] == ("cpsam", "cpsam", True), "stock cpsam first"
        assert ("lab_cells_v2", a_zoo["local"], True) in rows
        assert ("toxoplasma_plaque_v1", a_zoo["downloaded"], True) in rows, (
            "a zoo model downloaded into the picker's folder is selectable")
        assert ("toxoplasma_pv_v1 (not downloaded)", None, True) in rows, (
            "one not downloaded is listed, with no path to load, and can be "
            "chosen -- choosing it downloads it (item 419 point 3)")
        pending = made._cp_model.findText("toxoplasma_pv_v1 (not downloaded)")
        assert made._cp_model.itemData(pending, Qt.ForegroundRole) is not None, (
            "and it is greyed")
        assert made._cp_model.itemData(
            pending, mm._ZOO_PENDING_ROLE).key == "toxoplasma_pv_v1"
        assert not any("yolo" in text or "well_detector" in text
                       for text, _data, _on in rows), (
            "the zoo's YOLO detector is not a Cellpose model")
        assert all(call.get("block") is False for call in a_zoo["asked"]), (
            "building the screen never waits on the network")
        assert made._cp_model.currentData() == "cpsam"
    finally:
        made._magnifier.close()
        made.close_folded()


def test_the_model_zoo_button_opens_the_cellpose_zoo_and_selects_the_pick(
        qtbot, qt_theme_applied, a_zoo, monkeypatch):
    from spacr.qt.widgets import model_zoo_picker

    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    try:
        button = made._cp_model_zoo_btn
        assert button.text() == "Model zoo…"
        row = made._cp_model.parentWidget()
        assert button.parentWidget() is row, "the button sits beside the list"
        assert row.layout().indexOf(made._cp_model) == 0
        assert row.layout().indexOf(button) == 1

        opened = []
        # The picker downloads the model the combo showed greyed out.
        fetched = a_zoo["downloads"] / "cpsam_v2_toxo_r2"

        def choose(parent=None, kinds=None):
            opened.append((parent, kinds))
            fetched.write_bytes(b"weights")
            return str(fetched)

        monkeypatch.setattr(model_zoo_picker, "choose_model", choose)
        button.click()
        assert opened == [(made, ("cellpose",))]
        assert made._cp_model.currentData() == str(fetched)
        assert made._cp_model.currentText() == "toxoplasma_pv_v1"
        assert ("toxoplasma_pv_v1 (not downloaded)", None, False) not in _rows(
            made._cp_model), "the greyed-out row became the downloaded model"
        assert made._magnifier_context()["model_name"] == str(fetched)

        elsewhere = a_zoo["downloads"].parent / "picked_elsewhere.CP_model"
        elsewhere.write_bytes(b"weights")
        monkeypatch.setattr(model_zoo_picker, "choose_model",
                            lambda parent=None, kinds=None: str(elsewhere))
        button.click()
        assert made._cp_model.currentData() == str(elsewhere)
        assert made._cp_model.currentText() == "picked_elsewhere.CP_model"

        monkeypatch.setattr(model_zoo_picker, "choose_model",
                            lambda parent=None, kinds=None: None)
        button.click()
        assert made._cp_model.currentData() == str(elsewhere), (
            "cancelling the picker changes nothing")
    finally:
        made._magnifier.close()
        made.close_folded()


def test_a_zoo_that_cannot_be_read_leaves_the_installed_cellpose(
        qtbot, qt_theme_applied, monkeypatch):
    from spacr import model_zoo

    def broken(**_kwargs):
        raise OSError("the zoo is on a disk that went away")

    monkeypatch.setattr(model_zoo, "catalogue", broken)
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    try:
        assert made._cp_model.findData("cpsam") >= 0
        assert not any(made._cp_model.itemData(i, mm._ZOO_ROLE)
                       for i in range(made._cp_model.count()))
    finally:
        made._magnifier.close()
        made.close_folded()


# ---------------------------------------------------------------------------
# 4 and 9. The Object detection category is what the magnifier's detection uses
# ---------------------------------------------------------------------------

class _Spy:
    """Stands in for :func:`make_masks.cellpose_detect`, with its signature."""

    def __init__(self):
        self.calls = []

    def __call__(self, image, model, *, diameter=0, normalize=True,
                 flow_threshold=mm.FLOW_THRESHOLD,
                 cellprob_threshold=mm.CELLPROB_THRESHOLD, min_size=0):
        self.calls.append((np.array(image, copy=True), model, dict(
            diameter=diameter, normalize=normalize,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold, min_size=min_size)))
        return np.zeros(np.asarray(image).shape[:2], np.int32), None, None


def test_the_spy_has_cellpose_detects_own_signature():
    import inspect

    real = inspect.signature(mm.cellpose_detect)
    spy = inspect.signature(_Spy.__call__)
    assert list(real.parameters) == list(spy.parameters)[1:]
    for name, parameter in real.parameters.items():
        assert spy.parameters[name].default == parameter.default
        assert spy.parameters[name].kind == parameter.kind


def _cellpose_on(screen):
    index = screen._mag_mode.findData("cellpose")
    if index >= 0:
        screen._mag_mode.setCurrentIndex(index)
    else:                   # no Cellpose on this machine: the mode, not the box
        screen._on_magnifier_mode("cellpose")
    assert screen._magnifier.mode == "cellpose"


def test_the_cellpose_sam_values_are_exactly_what_the_detection_is_passed(
        qtbot, screen, monkeypatch, tmp_path):
    """One source of truth: the category's boxes reach the model call as set.

    The box, the whole-image run and Object detection are spied on in
    turn, and all three are handed the same numbers.
    """
    from spacr.qt.widgets import model_zoo_picker

    checkpoint = tmp_path / "fine_tuned.CP_model"
    checkpoint.write_bytes(b"weights")
    monkeypatch.setattr(model_zoo_picker, "choose_model",
                        lambda parent=None, kinds=None: str(checkpoint))
    screen._cp_model_zoo_btn.click()
    model = object()
    screen._cp_loaded[str(checkpoint)] = model
    spy = _Spy()
    monkeypatch.setattr(mm, "cellpose_detect", spy)

    for control, value in ((screen._cp_flow, 0.85),
                           (screen._cp_cellprob, -2.5),
                           (screen._cp_diameter, 37)):
        assert control.value() != value
        control.setValue(value)
    screen._cp_normalize.setChecked(False)
    screen._min_area.setValue(5)
    screen._mag_sensitivity.setValue(3.0)
    wanted = dict(diameter=37, normalize=False, flow_threshold=0.85,
                  cellprob_threshold=-2.5, min_size=5)

    _cellpose_on(screen)
    assert not screen._mag_sensitivity.isEnabled(), (
        "Sensitivity is the Otsu mode's, so it is greyed out")
    screen._btn_magnifier.setChecked(True)
    hover(screen, 30, 30)
    wait_for_result(qtbot, screen)
    crop, used, kwargs = spy.calls[-1]
    assert used is model, "the model chosen in the category"
    assert kwargs == wanted
    np.testing.assert_array_equal(crop, screen._canvas.image[14:46, 14:46])

    screen._mag_scope.setCurrentIndex(screen._mag_scope.findData("image"))
    qtbot.waitUntil(lambda: screen._magnifier._image_result is not None,
                    timeout=10_000)
    whole, used, kwargs = spy.calls[-1]
    assert used is model and kwargs == wanted
    np.testing.assert_array_equal(whole, screen._canvas.image)

    before = len(spy.calls)
    screen._cp_flow.setValue(1.2)
    assert screen._magnifier._image_result is None, (
        "a threshold changed in the category discards the whole-image objects")
    qtbot.waitUntil(lambda: screen._magnifier._image_result is not None,
                    timeout=10_000)
    assert len(spy.calls) == before + 1
    assert spy.calls[-1][2] == dict(wanted, flow_threshold=1.2)

    screen._mag_scope.setCurrentIndex(screen._mag_scope.findData("region"))
    screen.run_cellpose()
    image, used, kwargs = spy.calls[-1]
    assert used is model
    assert kwargs == dict(wanted, flow_threshold=1.2), (
        "Object detection is handed the very same values")
    np.testing.assert_array_equal(image, screen._canvas.image)


def test_the_thresholds_live_in_the_object_detection_category(screen):
    """Item 419 point 5 renamed this category and moved Otsu's settings out.

    The Cellpose controls stay where 417 put them; the threshold correction
    went to the Otsu category of its own, with the five settings item 419
    added, because it is not a Cellpose setting and never was.
    """
    categories = dict(screen._settings_categories)
    cellpose = categories["Object detection"]
    for control in (screen._cp_model, screen._cp_model_zoo_btn,
                    screen._cp_flow, screen._cp_cellprob,
                    screen._cp_diameter, screen._cp_normalize):
        assert cellpose.isAncestorOf(control)
    otsu = categories["Otsu"]
    for control in (screen._otsu_correction, screen._otsu_smoothing,
                    screen._otsu_bright, screen._otsu_fill_holes,
                    screen._otsu_split, screen._otsu_exclude_border):
        assert otsu.isAncestorOf(control)
    assert not cellpose.isAncestorOf(screen._otsu_correction)
    assert screen._cp_flow.value() == pytest.approx(mm.FLOW_THRESHOLD)
    assert screen._cp_cellprob.value() == pytest.approx(mm.CELLPROB_THRESHOLD)
    # Cellpose's own GUI offers -6..6 and 0..3; both fit inside these.
    assert screen._cp_cellprob.minimum() <= -6 and screen._cp_cellprob.maximum() >= 6
    assert screen._cp_flow.minimum() == 0 and screen._cp_flow.maximum() >= 3
    assert not categories["Live magnifier"].isAncestorOf(screen._cp_flow), (
        "no second set of thresholds on the magnifier's own category")


def test_a_ledger_entry_names_the_settings_the_objects_were_found_with(
        qtbot, screen):
    switch_on(screen, CodedStub({7: (20, 20, 26, 25)}))
    screen._cp_flow.setValue(0.6)
    screen._cp_cellprob.setValue(1.5)
    screen._otsu_correction.setValue(1.25)
    hover(screen, 30, 30)
    wait_for_result(qtbot, screen)
    from tests.qt.test_the_live_magnifier_segments_under_the_mouse import click

    click(screen, 30, 30)
    detail = screen._log.edits[-1].detail
    assert detail["flow_threshold"] == pytest.approx(0.6)
    assert detail["cellprob_threshold"] == pytest.approx(1.5)
    assert detail["otsu_correction"] == pytest.approx(1.25)
    assert detail["model"] == "cpsam"


# ---------------------------------------------------------------------------
# 7. Otsu's threshold correction
# ---------------------------------------------------------------------------

def soft_blobs(n: int = IMG_N) -> np.ndarray:
    """Three disks with a 5 px ramp at the rim, so the cut sets their size.

    Plateau disks, not Gaussian blobs, because the Otsu mode only cuts
    at Otsu's level where a region is two clear populations. Measured before
    this was written: Gaussian blobs (sigma 5) fall short of that, are cut at
    the noise floor, and no correction moves them; these disks score 0.868
    on the whole field and 0.863 in the 32 px box round (16, 16), against the
    0.8 needed, and the rim ramp is what a correction moves the cut across.
    """
    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[0:n, 0:n]
    img = np.full((n, n), 1000.0)
    for cy, cx in ((16, 16), (16, 46), (46, 30)):
        radius = np.hypot(yy - cy, xx - cx)
        img += 3000.0 * np.clip((9.0 - radius) / 5.0, 0.0, 1.0)
    img += rng.normal(0, 40, img.shape)
    return np.clip(img, 0, 65535).astype(np.uint16)


def test_the_correction_moves_the_otsu_cut_and_1_is_otsu_itself():
    from spacr.qt import mask_engine as engine

    field = soft_blobs()
    plain = engine._classical_region_labels(field, min_area=10)
    same = engine._classical_region_labels(field, min_area=10, correction=1.0)
    strict = engine._classical_region_labels(field, min_area=10,
                                             correction=1.4)
    loose = engine._classical_region_labels(field, min_area=10,
                                            correction=0.7)
    np.testing.assert_array_equal(plain, same)
    assert plain.max() == strict.max() == loose.max() == 3
    assert (strict > 0).sum() < (plain > 0).sum() < (loose > 0).sum()


def test_the_correction_moves_otsu_detect_on_either_side(monkeypatch):
    from spacr.qt import mask_engine as engine

    field = soft_blobs()
    np.testing.assert_array_equal(
        engine._otsu_instances(field, min_area=4),
        engine.otsu_instances(field, min_area=4))
    area = {c: int((engine._otsu_instances(field, min_area=4,
                                           correction=c) > 0).sum())
            for c in (0.8, 1.0, 1.2)}
    assert area[1.2] < area[1.0] < area[0.8]

    dark = (5000 - field.astype(np.int32)).astype(np.uint16)
    np.testing.assert_array_equal(
        engine._otsu_instances(dark, bright=False, min_area=4),
        engine.otsu_instances(dark, bright=False, min_area=4))
    dark_area = {c: int((engine._otsu_instances(
        dark, bright=False, min_area=4, correction=c) > 0).sum())
        for c in (0.8, 1.0, 1.2)}
    assert dark_area[1.2] < dark_area[1.0] < dark_area[0.8], (
        "above 1 is stricter for dark objects too")

    with pytest.raises(ValueError, match="greater than 0"):
        engine._otsu_instances(field, correction=0)
    with pytest.raises(ValueError, match="empty"):
        engine._otsu_instances(np.zeros((0, 0), np.uint16), correction=1.5)


@pytest.fixture
def blob_screen(qtbot, qt_theme_applied, tmp_path):
    folder = tmp_path / "blobs"
    folder.mkdir()
    imageio.imwrite(folder / "a.tif", soft_blobs())
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    assert made._open_folder(str(folder))
    made._canvas.resize(CANVAS_W, CANVAS_H)
    made._canvas.refresh()
    assert made._canvas.pixmap().width() == PIXMAP_N
    made._min_area.setValue(4)
    made._mag_size.setValue(SIZE)
    yield made
    made._magnifier.close()
    made.close_folded()


def test_the_correction_set_in_the_category_changes_the_otsu_detect_mask(
        blob_screen):
    made = blob_screen
    made._combine_mode.setCurrentIndex(made._combine_mode.findData("replace"))
    made._on_detect_otsu()
    at_one = made._canvas.mask.copy()
    assert made._log.edits[-1].detail["otsu_correction"] == 1.0

    made._otsu_correction.setValue(1.3)
    made._on_detect_otsu()
    corrected = made._canvas.mask.copy()
    assert made._log.edits[-1].detail["otsu_correction"] == pytest.approx(1.3)
    assert 0 < (corrected > 0).sum() < (at_one > 0).sum()


def test_the_correction_set_in_the_category_changes_the_magnifiers_objects(
        qtbot, blob_screen):
    """The real Otsu mode, no stub: the box's objects shrink."""
    made = blob_screen
    magnifier = made._magnifier
    made._btn_magnifier.setChecked(True)
    assert magnifier.mode == "otsu"
    hover(made, 16, 16)
    wait_for_result(qtbot, made)
    at_one = int((magnifier._shown.labels > 0).sum())
    assert at_one > 0
    assert magnifier._shown.request.otsu_correction == 1.0

    made._otsu_correction.setValue(1.3)
    qtbot.waitUntil(lambda: magnifier._shown is not None
                    and magnifier._shown.request.otsu_correction == 1.3
                    and not magnifier.updating(), timeout=10_000)
    corrected = int((magnifier._shown.labels > 0).sum())
    assert 0 < corrected < at_one


# ---------------------------------------------------------------------------
# 10. Cellpose 3, DINOCell and SAMCell as magnifier detectors
# ---------------------------------------------------------------------------

#: Every backend mode, in the order the Mode box lists them (item 423 added
#: the four Cellpose 3 models ahead of DINOCell and SAMCell).
_BACKEND_MODES = ["cellpose3:cyto3", "cellpose3:cyto2", "cellpose3:cyto",
                  "cellpose3:nuclei", "dinocell", "samcell"]


def _modes(made):
    return [made._mag_mode.itemData(i) for i in range(made._mag_mode.count())]


def test_dinocell_and_samcell_are_offered_where_installed(
        qtbot, qt_theme_applied, monkeypatch):
    monkeypatch.setattr(mm, "find_spec",
                        lambda name: object() if name == "cellpose" else None)
    monkeypatch.setattr(mm, "_backend_ready", lambda mode: True)
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    try:
        assert _modes(made) == ["otsu", "cellpose", *_BACKEND_MODES]
        assert [made._mag_mode.itemText(i) for i in range(8)][2:] == [
            "Cellpose 3 · cyto3", "Cellpose 3 · cyto2", "Cellpose 3 · cyto",
            "Cellpose 3 · nuclei", "DINOCell", "SAMCell"]
        assert made._mag_uninstalled == set()
        assert all(note.isHidden()
                   for note in made._mag_install_notes.values())
        made._mag_mode.setCurrentIndex(made._mag_mode.findData("samcell"))
        assert made._magnifier.mode == "samcell"
        assert not made._mag_sensitivity.isEnabled()
    finally:
        made._magnifier.close()
        made.close_folded()


def test_where_not_installed_the_modes_are_greyed_and_offer_to_install(
        qtbot, qt_theme_applied, monkeypatch):
    """A model absent from the box teaches nobody that it exists.

    So both are always listed, greyed when their package is missing, and
    choosing one offers the install. Cancelling must leave the magnifier where
    it was -- a curious click cannot point it at a model that cannot load.
    """
    from PySide6.QtCore import Qt

    monkeypatch.setattr(mm, "find_spec",
                        lambda name: object() if name == "cellpose" else None)
    monkeypatch.setattr(mm, "_backend_ready", lambda mode: False)
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    try:
        assert _modes(made) == ["otsu", "cellpose", *_BACKEND_MODES]
        assert made._mag_uninstalled == set(_BACKEND_MODES)
        for mode in _BACKEND_MODES:
            index = made._mag_mode.findData(mode)
            assert made._mag_mode.itemData(index, Qt.ForegroundRole) is not None
            assert "not installed" in made._mag_mode.itemData(
                index, Qt.ToolTipRole)

        offered = []
        monkeypatch.setattr(type(made), "_offer_backend_install",
                            lambda self, mode: offered.append(mode) or False)
        made._mag_mode.setCurrentIndex(made._mag_mode.findData("otsu"))
        made._on_magnifier_mode_activated(made._mag_mode.findData("samcell"))
        assert offered == ["samcell"], "choosing a missing model did not offer it"
        assert made._mag_mode.currentData() == "otsu", (
            "cancelling the install left the box on a model that cannot load")
    finally:
        made._magnifier.close()
        made.close_folded()


def test_installing_cellpose3_from_the_mode_box_lights_all_four_models(
        qtbot, qt_theme_applied, monkeypatch):
    """Item 423: the install goes through the Model Zoo's own dialog -- off
    the GUI thread, into an environment of its own -- and one install makes
    every Cellpose 3 model usable, not only the one that was chosen."""
    from PySide6.QtCore import Qt

    from spacr.qt.widgets import model_zoo_picker

    monkeypatch.setattr(mm, "_backend_ready", lambda mode: False)
    answers = iter([False, True])
    asked = []
    monkeypatch.setattr(model_zoo_picker, "install_backend",
                        lambda parent, name: asked.append(name)
                        or next(answers))
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    try:
        assert made._offer_backend_install("cellpose3:cyto2") is False
        assert made._mag_uninstalled == set(_BACKEND_MODES)
        assert made._offer_backend_install("cellpose3:cyto2") is True
        assert asked == ["cellpose3", "cellpose3"]
        assert made._mag_uninstalled == {"dinocell", "samcell"}
        for mode in _BACKEND_MODES[:4]:
            index = made._mag_mode.findData(mode)
            assert made._mag_mode.itemData(index, Qt.ForegroundRole) is None
        index = made._mag_mode.findData("samcell")
        assert made._mag_mode.itemData(index, Qt.ForegroundRole) is not None
    finally:
        made._magnifier.close()
        made.close_folded()


def test_a_cellpose3_mode_loads_its_own_model_once(monkeypatch):
    from spacr import _segmentation_backends as backends

    built = []
    monkeypatch.setattr(backends, "_load_backend",
                        lambda name, **kw: built.append((name, kw)) or object())
    monkeypatch.setattr(mm, "_BACKEND_MODELS", {})
    first = mm._backend_model("cellpose3:nuclei")
    assert mm._backend_model("cellpose3:nuclei") is first
    mm._backend_model("dinocell")
    assert built == [("cellpose3", {"model_name": "nuclei"}),
                     ("dinocell", {"model_name": None})]


def test_a_cached_model_whose_environment_was_deleted_is_not_reused(
        tmp_path, monkeypatch):
    """Uninstall Cellpose 3 from the Model Zoo with Make Masks still open.

    The mode box's cache held the model for the life of the process, so the
    next hover asked it to segment, it started ``<env>/bin/python``, and the
    magnifier reported a missing FILE rather than a missing backend. The
    folder the model was built from is remembered and checked instead.
    """
    from spacr import _segmentation_backends as backends

    env = tmp_path / "backend-environments" / "cellpose3"
    python = Path(backends._env_python(str(env)))
    python.parent.mkdir(parents=True)
    python.write_text("")
    backends._write_marker(str(env), {"backend": "cellpose3"})

    built = []
    monkeypatch.setattr(backends, "_load_backend",
                        lambda name, **kw: built.append(name) or object())
    monkeypatch.setattr(mm, "_BACKEND_MODELS", {})
    monkeypatch.setattr(mm, "_BACKEND_MODEL_ENVS", {})

    first = mm._backend_model("cellpose3:cyto3")
    assert mm._BACKEND_MODEL_ENVS["cellpose3:cyto3"] == str(env)
    assert mm._backend_model("cellpose3:cyto3") is first, "rebuilt for nothing"

    shutil.rmtree(str(env))
    second = mm._backend_model("cellpose3:cyto3")
    assert second is not first, "a deleted environment's model was reused"
    assert built == ["cellpose3", "cellpose3"]


def test_a_model_built_without_an_environment_of_its_own_is_kept(monkeypatch):
    """A stand-in, or a backend an older spaCR put in spaCR's own
    environment, was never a folder to lose -- and must not be dropped."""
    from spacr import _segmentation_backends as backends

    monkeypatch.setattr(backends, "_load_backend", lambda name, **kw: object())
    monkeypatch.setattr(mm, "_BACKEND_MODELS", {})
    monkeypatch.setattr(mm, "_BACKEND_MODEL_ENVS", {})

    first = mm._backend_model("samcell")
    assert mm._BACKEND_MODEL_ENVS["samcell"] == ""
    assert mm._backend_model("samcell") is first


def test_the_mode_box_re_reads_the_backends_when_a_mode_is_chosen(
        qtbot, qt_theme_applied, monkeypatch):
    """The box used to learn this once, when the screen was built.

    Uninstalling Cellpose 3 from the Model Zoo with Make Masks open left its
    four modes un-greyed and out of ``_mag_uninstalled``, so choosing one
    offered no install and went to a backend that was gone. Installing one
    from the Model Zoo screen left the reverse: four greyed modes and an
    install dialog that returned at once.
    """
    from PySide6.QtCore import Qt

    here = {"ready": True}
    monkeypatch.setattr(mm, "_backend_ready", lambda mode: here["ready"])
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    try:
        assert made._mag_uninstalled == set()

        here["ready"] = False
        offered = []
        monkeypatch.setattr(type(made), "_offer_backend_install",
                            lambda self, mode: offered.append(mode) or False)
        made._mag_mode.setCurrentIndex(made._mag_mode.findData("otsu"))
        made._on_magnifier_mode_activated(
            made._mag_mode.findData("cellpose3:cyto3"))

        assert offered == ["cellpose3:cyto3"], (
            "a backend uninstalled while this screen was open was still used")
        assert made._mag_uninstalled == set(_BACKEND_MODES)
        index = made._mag_mode.findData("cellpose3:cyto3")
        assert made._mag_mode.itemData(index, Qt.ForegroundRole) is not None
        assert made._mag_mode.currentData() == "otsu"

        from spacr.qt.widgets import model_zoo_picker

        monkeypatch.undo()
        monkeypatch.setattr(mm, "_backend_ready", lambda mode: here["ready"])
        monkeypatch.setattr(model_zoo_picker, "install_backend",
                            lambda parent, name: offered.append(name) or True)
        made._mag_mode.setCurrentIndex(index)
        made._on_magnifier_mode_activated(index)
        assert offered == ["cellpose3:cyto3", "cellpose3"]
        assert made._mag_mode.currentData() == "cellpose3:cyto3", (
            "an install that succeeded did not select the row")
        assert made._mag_uninstalled == {"dinocell", "samcell"}, (
            "one environment carries all four Cellpose 3 models")

        here["ready"] = True
        made._on_magnifier_mode_activated(index)
        assert len(offered) == 2, (
            "an install was offered for a backend that is already here")
        assert made._mag_uninstalled == set()
        assert made._mag_mode.itemData(index, Qt.ForegroundRole) is None
    finally:
        made._magnifier.close()
        made.close_folded()


def test_a_mode_is_ready_when_its_backend_environment_is(tmp_path,
                                                        monkeypatch):
    from spacr import _segmentation_backends as backends

    assert not mm._backend_ready("cellpose3:cyto3")
    env = tmp_path / "backend-environments" / "cellpose3"
    python = Path(backends._env_python(str(env)))
    python.parent.mkdir(parents=True)
    python.write_text("")
    backends._write_marker(str(env), {"backend": "cellpose3"})
    assert mm._backend_ready("cellpose3:cyto")

    def _broken(name):
        raise OSError("the home folder went away")

    monkeypatch.setattr(backends, "_backend_state", _broken)
    assert not mm._backend_ready("samcell")
    assert mm._model_env("samcell") == "", (
        "a backend whose state cannot be read has no folder to watch")


def _stub_backend_class(backends):
    """A real ``_PlaneBackend`` subclass that labels one known rectangle."""
    class StubPlane(backends._PlaneBackend):
        built = []

        def __init__(self, device=None, **options):
            super().__init__(device="cpu")
            type(self).built.append(dict(options, device=device))
            self.seen = []

        def _segment_plane(self, image, cellprob_threshold=None):
            self.seen.append((image.dtype, image.shape, cellprob_threshold))
            labels = np.zeros(image.shape, np.int32)
            labels[4:10, 5:12] = 3
            return labels, [image, None, None, None]

    return StubPlane


def test_the_stand_in_backend_keeps_the_real_backends_signatures():
    import inspect

    from spacr import _segmentation_backends as backends

    stub = _stub_backend_class(backends)
    for real in (backends._DinoCellBackend, backends._SamCellBackend):
        assert (list(inspect.signature(real._segment_plane).parameters)
                == list(inspect.signature(stub._segment_plane).parameters))
        assert "device" in inspect.signature(real.__init__).parameters
    assert stub.eval is backends._PlaneBackend.eval, (
        "the batch call is the real one, not the stand-in's")


@pytest.mark.parametrize("mode", ["dinocell", "samcell"])
def test_a_backend_segments_the_box_through_the_real_backend_seam(
        qtbot, screen, monkeypatch, mode):
    """The real _load_backend, the real eval and the real cellpose_detect."""
    from spacr import _segmentation_backends as backends
    from tests.qt.test_the_live_magnifier_segments_under_the_mouse import click

    stub = _stub_backend_class(backends)
    monkeypatch.setitem(backends._BACKEND_CLASSES, mode, stub)
    monkeypatch.setattr(mm, "_BACKEND_MODELS", {})
    screen._cp_cellprob.setValue(-1.5)
    screen._on_magnifier_mode(mode)
    screen._btn_magnifier.setChecked(True)

    hover(screen, 30, 30)
    wait_for_result(qtbot, screen)
    shown = screen._magnifier._shown
    assert shown.mode == mode and shown.note == ""
    hover(screen, 31, 30)
    wait_for_result(qtbot, screen)
    assert len(stub.built) == 1, "the model is built once, not per move"
    model = mm._BACKEND_MODELS[mode]
    dtype, shape, cellprob = model.seen[0]
    assert dtype == np.uint8 and shape == (32, 32)
    assert cellprob == -1.5, "the Object detection cell probability reaches it"

    click(screen, 31, 30)
    expected = np.zeros((IMG_N, IMG_N), bool)
    box = screen._magnifier._shown.request.box
    expected[box[1] + 4:box[1] + 10, box[0] + 5:box[0] + 12] = True
    np.testing.assert_array_equal(screen._canvas.mask > 0, expected)
    assert screen._log.edits[-1].detail["mode"] == mode


@pytest.mark.parametrize("mode", ["dinocell", "samcell"])
def test_a_backend_that_is_not_installed_falls_back_and_says_how_to_install(
        qtbot, screen, monkeypatch, mode):
    import sys

    monkeypatch.setitem(sys.modules, mode, None)
    monkeypatch.setattr(mm, "_BACKEND_MODELS", {})
    screen._on_magnifier_mode(mode)
    screen._btn_magnifier.setChecked(True)
    hover(screen, 30, 30)
    wait_for_result(qtbot, screen)
    assert screen._magnifier._shown.mode == "otsu"
    status = screen._status_label.text()
    assert f"{mm._magnifier_mode_label(mode)} could not run" in status, (
        "item 407: the note names the mode the way the Mode box does")
    assert "Install it from the Model Zoo" in status
    assert mode not in mm._BACKEND_MODELS, "a failed build is not kept"
    assert screen._magnifier.build_request().mode == "otsu"
