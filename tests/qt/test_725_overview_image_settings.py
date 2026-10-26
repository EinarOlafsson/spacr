"""Round-5 Overview, M4: the right-click per-image settings panel.

Item 725, maintainer feedback 2026-10-25 round 5, point 4: "the user should be
able to right cklick on an image and change its settings, those settings
should include a levels histagram, brightness, contranst, saturation and so
on", and point 5: each channel can be treated individually.
"""
import json

import numpy as np
import pytest
import tifffile

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QMenu

from spacr.qt.widgets import qc_field_browser as browser


PATTERN = r"(?P<plateID>p\d+)-(?P<wellID>[A-Z]\d+)-f(?P<fieldID>\d+)-c(?P<chanID>\d+)\.tif"
WELLS = ("A01", "A02", "B01", "B02")
HEIGHT, WIDTH = 40, 64


def _raw(channel, index, field):
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    return (yy * 50 + xx * 20 + 1000 * channel + 13 * index + 5 * field).astype(np.uint16)


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    for index, well in enumerate(WELLS):
        for field in (1, 2):
            for channel in (1, 2):
                tifffile.imwrite(folder / f"p1-{well}-f{field}-c{channel}.tif", _raw(channel, index, field))
    return folder


def _runners(view):
    return (view._index_jobs, view._image_jobs, view._thumbnail_jobs, view._field_thumbnail_jobs,
            view._montage_jobs, view._export_jobs, view._metadata_jobs, view._measurement_jobs,
            view._native_jobs, view._image_stack_jobs, view._group_jobs, view._readout_jobs,
            view._levels_jobs)


def _retire(view, qtbot):
    view._cancel()
    qtbot.waitUntil(lambda: all(runner.active_jobs() == 0 for runner in _runners(view)), timeout=15000)
    for runner in _runners(view):
        runner.shutdown(timeout_ms=3000)
    view.close()


def _open(qtbot, source):
    browser._OVERVIEW_PLANE_CACHE.clear()
    browser._OVERVIEW_INFO_CACHE.clear()
    dialog = browser._FieldOverview(settings={
        "source": str(source), "source_scope": "folder", "custom_regex": PATTERN,
        "channels": (1, 2), "geometry": (2, 2), "representatives": 1})
    qtbot.addWidget(dialog, before_close_func=lambda widget: _retire(widget, qtbot))
    dialog.resize(1300, 950)
    dialog.show()
    dialog._scan()
    qtbot.waitUntil(lambda: bool(dialog._fields) and dialog._normalization_ready
                    and not dialog._index_jobs.is_busy(), timeout=15000)
    dialog.fov_axis.setCurrentIndex(dialog.fov_axis.findData(None))
    return dialog


@pytest.fixture
def view(qtbot, source):
    return _open(qtbot, source)


def _field(view, well, number="1"):
    return next(field for field in view._fields if field.well == well and field.field == number)


def _pixels(view, field, **extra):
    options = dict(view._display_kwargs(view=True))
    options.update(extra)
    return browser._overview_pixels(field, (1, 2), view._choices()[1], edge=None, **options)


def _panel(view, qtbot, well="A01"):
    view._open_image_settings(_field(view, well))
    panel = view._image_settings
    qtbot.waitUntil(lambda: panel.histogram.counts is not None, timeout=10000)
    return panel


def test_an_override_changes_only_its_own_image(view, qtbot):
    target, other, neighbour = _field(view, "A01"), _field(view, "A02"), _field(view, "A01", "2")
    before = {field: _pixels(view, field) for field in (target, other, neighbour)}
    panel = _panel(view, qtbot)
    panel.black.setValue(.4)
    qtbot.waitUntil(lambda: browser._overview_image_key(target) in view._image_overrides, timeout=5000)
    assert set(view._image_overrides) == {browser._overview_image_key(target)}
    assert not (_pixels(view, target) == before[target]).all()
    assert (_pixels(view, other) == before[other]).all()
    assert (_pixels(view, neighbour) == before[neighbour]).all()
    assert view._is_adjusted(target) and not view._is_adjusted(other)


def test_the_exact_level_mapping_of_black_white_gamma_brightness_and_contrast():
    plane = np.linspace(0, 1, 101, dtype=np.float32)
    settings = {"black": .2, "white": .6, "gamma": 2., "brightness": .1, "contrast": 1.5}
    expected = np.clip((np.clip((plane - .2) / .4, 0, 1) ** .5 - .5) * 1.5 + .5 + .1, 0, 1)
    assert np.allclose(browser._overview_override_plane(plane, settings), expected, atol=1e-6)
    assert np.array_equal(browser._overview_override_plane(plane, {"visible": False}), np.zeros_like(plane))
    middle = browser._overview_gamma_midpoint(.2, .6, 2.)
    assert browser._overview_override_plane(np.float32([middle]), {"black": .2, "white": .6, "gamma": 2.})[0] == pytest.approx(.5)
    assert browser._overview_gamma_from_midpoint(.2, .6, middle) == pytest.approx(2.)


def test_the_override_reaches_the_rendered_pixels_exactly(view):
    field = _field(view, "B01")
    key = browser._overview_image_key(field)
    raw = next(browser._overview_raw_planes(field, (1,), edge=None))[2].astype(np.float64)
    plain = (raw - raw.min()) / (raw.max() - raw.min())
    adjusted = browser._overview_pixels(field, (1,), (0., 100.), edge=None, image_overrides={
        key: {"channels": {"1": {"black": .25, "white": .75, "gamma": 1.}}}})[..., 0].astype(np.int64)
    expected = np.uint8(np.clip((plain - .25) / .5, 0, 1) * 255).astype(np.int64)
    assert np.abs(adjusted - expected).max() <= 1


def test_changing_one_channel_leaves_the_other_channels_pixels_unchanged(view):
    field = _field(view, "A02")
    key = browser._overview_image_key(field)
    for settings in ({"black": .3}, {"white": .5}, {"gamma": 3.}, {"brightness": .2}, {"contrast": 2.},
                     {"visible": False}):
        plain = dict(browser._overview_pixels(field, (1, 2), (1., 99.), edge=None, channel_views=True))
        changed = dict(browser._overview_pixels(field, (1, 2), (1., 99.), edge=None, channel_views=True,
                                                image_overrides={key: {"channels": {"1": settings}}}))
        assert not np.array_equal(plain[1], changed[1]), settings
        assert np.array_equal(plain[2], changed[2]), settings


def test_reset_returns_the_image_to_the_plate_settings_and_hides_its_badge(view, qtbot):
    view._well(view.wells.item(*view._well_positions[(view.plates.currentData(), "A01")]))
    qtbot.waitUntil(lambda: view.fields.count() > 0, timeout=10000)
    view.fields.setCurrentRow(0)
    view._field(view.fields.item(0))
    shown = view._main_view_field
    panel = _panel(view, qtbot)
    panel.show_field(shown, (1, 2))
    plain = _pixels(view, shown)
    panel.contrast.setValue(250)
    panel.saturation.setValue(40)
    qtbot.waitUntil(lambda: view._is_adjusted(shown), timeout=5000)
    assert view._image_adjusted_badge.isVisibleTo(view)
    panel.reset_button.click()
    assert view._image_overrides == {}
    assert not view._image_adjusted_badge.isVisibleTo(view)
    assert (_pixels(view, shown) == plain).all()
    assert panel.contrast.value() == 100 and panel.saturation.value() == 100


def test_apply_to_well_plate_and_all_copies_the_selected_or_every_channel(view, qtbot):
    panel = _panel(view, qtbot)
    panel.tabs.setCurrentIndex(1)
    panel.gamma.setValue(2.5)
    panel.tabs.setCurrentIndex(0)
    panel.black.setValue(.2)
    panel._push()
    source = browser._overview_image_key(_field(view, "A01"))
    same_well = browser._overview_image_key(_field(view, "A01", "2"))
    other_well = browser._overview_image_key(_field(view, "A02"))
    panel.scope.setCurrentIndex(panel.scope.findData("well"))
    panel.apply_channels.setCurrentIndex(panel.apply_channels.findData("selected"))
    panel.apply_button.click()
    assert view._image_overrides[same_well] == {"channels": {"1": {"black": .2}}}
    assert other_well not in view._image_overrides
    panel.apply_channels.setCurrentIndex(panel.apply_channels.findData("all"))
    panel.apply_button.click()
    assert view._image_overrides[same_well] == view._image_overrides[source]
    panel.scope.setCurrentIndex(panel.scope.findData("plate"))
    panel.apply_button.click()
    assert len(view._image_overrides) == len(view._fields)
    view._set_image_override(source, None)
    view._copy_image_override(_field(view, "A01"), "all", None)
    assert view._image_overrides == {}


def test_the_all_channels_link_is_off_by_default_and_edits_every_channel_when_on(view, qtbot):
    panel = _panel(view, qtbot)
    key = browser._overview_image_key(_field(view, "A01"))
    assert not panel.link.isChecked()
    panel.gamma.setValue(2.)
    panel._push()
    assert view._image_overrides[key]["channels"] == {"1": {"gamma": 2.}}
    panel.link.setChecked(True)
    panel.brightness.setValue(30)
    panel._push()
    channels = view._image_overrides[key]["channels"]
    assert channels["1"] == {"gamma": 2., "brightness": .3}
    assert channels["2"] == {"brightness": .3}


def test_the_histogram_handles_drag_the_levels(view, qtbot):
    panel = _panel(view, qtbot)
    histogram = panel.histogram
    histogram.resize(300, 100)
    assert histogram.counts.sum() > 0
    key = browser._overview_image_key(_field(view, "A01"))

    def drag(start, end):
        for kind, x in ((QEvent.MouseButtonPress, start), (QEvent.MouseMove, end), (QEvent.MouseButtonRelease, end)):
            point = QPointF(x, 50)
            QApplication.sendEvent(histogram, QMouseEvent(kind, point, histogram.mapToGlobal(point),
                                                          Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))

    drag(histogram._x(0.), histogram._x(.25))
    assert view._image_overrides[key]["channels"]["1"]["black"] == pytest.approx(.25, abs=.01)
    assert panel.black.value() == pytest.approx(.25, abs=.01)
    drag(histogram._x(1.), histogram._x(.75))
    assert view._image_overrides[key]["channels"]["1"]["white"] == pytest.approx(.75, abs=.01)
    middle = browser._overview_gamma_midpoint(histogram.black, histogram.white, histogram.gamma)
    drag(histogram._x(middle), histogram._x(.4))
    gamma = view._image_overrides[key]["channels"]["1"]["gamma"]
    assert gamma == pytest.approx(browser._overview_gamma_from_midpoint(.25, .75, .4), abs=.05)


def test_overrides_survive_saving_and_reopening(view, qtbot, source):
    panel = _panel(view, qtbot)
    panel.tabs.setCurrentIndex(1)
    panel.visible.setChecked(False)
    panel.saturation.setValue(150)
    panel.close()
    key = browser._overview_image_key(_field(view, "A01"))
    path = source / "results" / "overview_view.json"
    text = browser._overview_key_text(key)
    qtbot.waitUntil(lambda: path.is_file() and text in json.loads(path.read_text())["images"], timeout=10000)
    saved = json.loads(path.read_text())["images"][text]
    assert saved == {"channels": {"2": {"visible": False}}, "saturation": 1.5}
    reopened = _open(qtbot, source)
    qtbot.waitUntil(lambda: key in reopened._image_overrides, timeout=10000)
    assert reopened._image_overrides[key] == saved
    assert reopened._is_adjusted(_field(reopened, "A01"))


def test_right_clicking_an_image_or_a_tile_offers_its_settings(view, qtbot, monkeypatch):
    shown = []
    monkeypatch.setattr(QMenu, "popup", lambda menu, *_: shown.append([a.text() for a in menu.actions()]))
    view._well(view.wells.item(*view._well_positions[(view.plates.currentData(), "B02")]))
    qtbot.waitUntil(lambda: view.fields.count() > 0, timeout=10000)
    view.fields.setCurrentRow(0)
    view._field(view.fields.item(0))
    view._image_menu(view.view, QPoint(5, 5))
    assert "Image settings…" in shown[-1] and "Reset image settings" in shown[-1]
    rect = view.fields.visualItemRect(view.fields.item(0))
    view._field_menu(view.fields, rect.center())
    assert "Image settings…" in shown[-1]
    assert view.view.contextMenuPolicy() == Qt.CustomContextMenu


def test_overrides_and_the_view_state_keep_string_channel_keys():
    cleaned = browser._overview_clean_override({"channels": {1: {"black": 0., "white": .5}, 2: {"gamma": 1.}},
                                                "saturation": 1.})
    assert cleaned == {"channels": {"1": {"white": .5}}}
    assert browser._overview_clean_override({"channels": {1: {"visible": True}}}) is None
