"""Round-6 Overview: display, normalization and settings.

Item 725, maintainer feedback 2026-10-26 round 6: saturation is part of the
Intensity measurements; more colour modes with RGB, CMY and CMYK on black
and on white and a custom colour per channel; no Display category, with the
tile spacing beside the well view's size slider; Normalized channels typed
as "1,2,3,4"; settings that the normalization mode does not read greyed
out; an Advanced normalization category; an Apply button that redraws every
view, the open image included; the image settings window without a black
square and with draggable levels, and per-image Global normalization and
Global enhancement ticks; Measure before Export.
"""
import json

import numpy as np
import pytest
import tifffile

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QLineEdit

from spacr.qt.widgets import qc_field_browser as browser


PATTERN = r"(?P<plateID>p\d+)-(?P<wellID>[A-Z]\d+)-f(?P<fieldID>\d+)-c(?P<chanID>\d+)\.tif"
WELLS = ("A01", "A02", "B01", "B02")


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    rng = np.random.default_rng(6)
    for index, well in enumerate(WELLS):
        for field in (1, 2):
            for channel in (1, 2):
                pixels = (rng.integers(0, 200, (60, 80)) + np.arange(60 * 80).reshape(60, 80) % 3000
                          * channel + 50 * index).astype(np.uint16)
                tifffile.imwrite(folder / f"p1-{well}-f{field}-c{channel}.tif", pixels)
    return folder


def _runners(view):
    return (view._index_jobs, view._image_jobs, view._thumbnail_jobs, view._field_thumbnail_jobs,
            view._montage_jobs, view._export_jobs, view._metadata_jobs, view._measurement_jobs,
            view._native_jobs, view._image_stack_jobs, view._group_jobs, view._levels_jobs)


def _close(mask, qtbot):
    tab = getattr(mask, "_overview_tab", None)
    page = getattr(tab, "_overview_page", None)
    if page is not None and page.browser is not None:
        view = page.browser
        view._cancel()
        qtbot.waitUntil(lambda: all(runner.active_jobs() == 0 for runner in _runners(view)), timeout=15000)
    if page is not None:
        page.retire()
    from spacr.qt.bridge import prune_parked_threads
    qtbot.waitUntil(lambda: prune_parked_threads() == 0, timeout=15000)
    if tab is not None:
        tab.close()
    mask.close()


@pytest.fixture
def mask(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    browser._OVERVIEW_PLANE_CACHE.clear()
    browser._OVERVIEW_INFO_CACHE.clear()
    screen = AppScreen("mask")
    qtbot.addWidget(screen, before_close_func=lambda widget: _close(widget, qtbot))
    screen.resize(1500, 1000)
    screen.show()
    return screen


@pytest.fixture
def module(mask, qtbot, source):
    mask._settings_model.set_value_for_key("src", str(source))
    field = mask._settings_model._widgets.get("src")
    next(action for action in field.actions()
         if action.objectName() == "GenerateMasksFieldOverview").trigger()
    tab = mask._overview_tab
    wizard = tab._overview_page.wizard
    wizard.preset.setCurrentIndex(wizard.preset.count() - 1)
    wizard.regex.setText(PATTERN)
    wizard._scan()
    qtbot.waitUntil(lambda: wizard.next.isEnabled(), timeout=15000)
    wizard._next()
    page = tab._overview_page
    qtbot.waitUntil(lambda: page.browser is not None and len(page.browser._fields) == 2 * len(WELLS)
                    and page.browser._normalization_ready and len(page.browser._channel_buttons) == 2,
                    timeout=15000)
    return tab, page.browser


def _section(tab, title):
    return next(section for section in tab.rendered_settings_sections()
                if str(section.property("settingsCategorySource")) == title)


def _apply(tab):
    bar = tab._overview_page._apply_bar
    assert bar is not None and bar.apply_button.isEnabled()
    bar.apply_button.click()


# --- 1. Saturation belongs to Intensity -------------------------------------------------------


def test_saturation_is_measured_with_intensity_and_is_no_setting_of_its_own():
    from spacr import settings
    from spacr.measure import _image_feature_groups

    declarations = settings._overview_measurement_declarations()
    assert "overview_measure_saturation" not in declarations["defaults"]
    assert "overview_measure_saturation" not in declarations["categories"]["Intensity"]
    assert "saturation" in declarations["tooltips"]["overview_measure_intensity"]
    on = _image_feature_groups({"overview_measure_intensity": True})
    off = _image_feature_groups({"overview_measure_intensity": False})
    assert {"intensity", "saturation"} <= set(on)
    assert not {"intensity", "saturation"} & set(off)
    from spacr.validate import _check_retired_keys

    problems = _check_retired_keys({"overview_measure_saturation": False})
    assert len(problems) == 1 and "overview_measure_saturation" in " ".join(
        str(value) for value in vars(problems[0]).values())


# --- 2 and 3. Colour modes: presets, black and white backgrounds, custom ---------------------


def _planes(*values, shape=(2, 2)):
    return [np.full(shape, value, dtype=np.uint8) for value in values]


def _composite(values, mode, colours=None):
    channels = tuple(range(1, len(values) + 1))
    colours_, background, _inked = browser._overview_color_plan(channels, colours, mode)
    return browser._overview_composite(_planes(*values), colours_, background)[0, 0].tolist()


def test_the_colour_modes_offer_rgb_cmy_and_cmyk_on_black_and_on_white_presets_and_custom():
    from spacr.qt.screens.settings_model import _APP_COMBO_OPTIONS

    modes = [value for value, _caption in _APP_COMBO_OPTIONS["overview"]["overview_color_mode"]]
    assert modes == [mode for mode, _caption in browser._OVERVIEW_COLOR_MODE_CAPTIONS]
    assert {"additive", "rgb_white", "cmy_black", "cmy", "cmyk_black", "cmyk", "custom"} <= set(modes)
    assert len(modes) >= 9
    assert {background for _palette, background in browser._OVERVIEW_COLOR_MODES.values()} == {"black", "white"}
    from spacr.settings import OVERVIEW_DEFAULTS

    assert OVERVIEW_DEFAULTS["overview_color_mode"] == "additive"
    assert browser._OVERVIEW_COLOR_MODES["additive"] == ("rgb", "black")


@pytest.mark.parametrize("mode, index, expected", [
    ("additive", 0, [255, 0, 0]), ("additive", 1, [0, 255, 0]), ("additive", 2, [0, 0, 255]),
    ("rgb_white", 0, [255, 0, 0]), ("rgb_white", 1, [0, 255, 0]), ("rgb_white", 2, [0, 0, 255]),
    ("cmy_black", 0, [0, 255, 255]), ("cmy_black", 1, [255, 0, 255]), ("cmy_black", 2, [255, 255, 0]),
    ("cmy", 0, [0, 255, 255]), ("cmy", 1, [255, 0, 255]), ("cmy", 2, [255, 255, 0]),
])
def test_one_full_channel_shows_exactly_its_colour_on_either_background(mode, index, expected):
    values = [0, 0, 0]
    values[index] = 255
    assert _composite(values, mode) == expected


@pytest.mark.parametrize("mode, empty", [("additive", 0), ("cmy_black", 0), ("cmyk_black", 0),
                                         ("rgb_white", 255), ("cmy", 255), ("cmyk", 255)])
def test_no_signal_is_the_background(mode, empty):
    count = 4 if mode.startswith("cmyk") else 3
    assert _composite([0] * count, mode) == [empty] * 3


def test_full_overlap_is_white_on_black_and_black_on_white():
    assert _composite([255, 255, 255], "additive") == [255, 255, 255]
    assert _composite([255, 255, 255], "cmy_black") == [255, 255, 255]
    assert _composite([255, 255, 255], "rgb_white") == [0, 0, 0]
    assert _composite([255, 255, 255], "cmy") == [0, 0, 0]


def test_the_key_channel_is_black_ink_on_white_and_white_light_on_black():
    assert _composite([0, 0, 0, 255], "cmyk") == [0, 0, 0]
    assert _composite([0, 0, 0, 255], "cmyk_black") == [255, 255, 255]
    assert _composite([0, 0, 0, 128], "cmyk_black") == [128, 128, 128]


def test_rgb_on_white_overlaps_as_its_complement_mix():
    assert _composite([255, 255, 0], "rgb_white") == [0, 0, 0]
    assert _composite([128, 0, 0], "rgb_white") == [255, 127, 127]


def test_the_additive_default_is_unchanged(tmp_path):
    folder = tmp_path / "w"
    folder.mkdir()
    for channel in (1, 2):
        tifffile.imwrite(folder / f"ch{channel}.tif", (np.arange(64 * 64).reshape(64, 64) * channel).astype(np.uint16))
    field = browser._OverviewField("p1", "A01", "1", 1, "1", tuple(
        (channel, str(folder / f"ch{channel}.tif")) for channel in (1, 2)))
    shown = browser._overview_pixels(field, (1, 2), (1., 99.), edge=64)
    views = dict(browser._overview_pixels(field, (1, 2), (1., 99.), edge=64, channel_views=True))
    assert np.array_equal(shown[..., 0], views[1][..., 0])
    assert np.array_equal(shown[..., 1], views[2][..., 0])
    assert not shown[..., 2].any()
    single = browser._overview_pixels(field, (1,), (1., 99.), edge=64)
    assert np.array_equal(single[..., 0], single[..., 2])
    inverted = browser._overview_pixels(field, (1,), (1., 99.), edge=64, color_mode="rgb_white")
    assert np.array_equal(inverted, 255 - single)
    custom = browser._overview_pixels(field, (1, 2), (1., 99.), edge=64, color_mode="custom",
                                      channel_colors={"1": "#ffff00", "2": "#0000ff"})
    assert np.array_equal(custom[..., 0], views[1][..., 0])
    assert np.array_equal(custom[..., 1], views[1][..., 0])
    assert np.array_equal(custom[..., 2], views[2][..., 0])
    side = browser._overview_pixels(field, (1, 2), (1., 99.), edge=64, color_mode="greys")
    assert np.array_equal(side[..., 0], side[..., 1]) and np.array_equal(side[..., 1], side[..., 2])
    assert not np.array_equal(side, shown) and side[:, 32:].any() and side[:, :32].any()


def test_custom_colours_are_read_written_and_default_by_position():
    colours = browser._overview_custom_colors("1=#FF8000, 2=#00ff00, bad, 3=nothex")
    assert colours == {"1": "#ff8000", "2": "#00ff00"}
    assert browser._overview_custom_text(colours) == "1=#ff8000, 2=#00ff00"
    assert browser._overview_custom_color(colours, 4, 2) == browser._OVERVIEW_CUSTOM_DEFAULTS[2]
    plan, background, inked = browser._overview_color_plan((1, 2, 5, 6), colours, "custom")
    assert plan[0] == (1., 128 / 255., 0.) and background == "black" and not inked
    assert len(plan) == 4


def test_custom_colours_are_picked_per_channel_saved_and_applied(module, qtbot, monkeypatch):
    tab, view = module
    model = tab._settings_model
    from PySide6.QtGui import QColor
    from spacr.qt.widgets import colour_picker

    picker = next(field for _section, field, _widgets in tab._overview_page._rows
                  if isinstance(field, browser._OverviewCustomColors))
    qtbot.waitUntil(lambda: len(picker._buttons) == 2, timeout=15000)
    assert picker.property("settingKey") == "overview_custom_colours"
    assert not picker.isEnabled()
    model.set_value_for_key("overview_color_mode", "custom")
    assert picker.isEnabled()
    monkeypatch.setattr(colour_picker, "pick_colour", lambda *_args, **_kwargs: QColor("#ff8000"))
    picker._buttons["2"].click()
    assert model._read_value("overview_custom_colors") == "1=#ff0000, 2=#ff8000"
    assert view._color_mode == "additive"
    _apply(tab)
    assert view._color_mode == "custom"
    assert view._display_kwargs(view=True)["channel_colors"] == {"1": "#ff0000", "2": "#ff8000"}


# --- 4. No Display category; the spacing sits left of the size slider -------------------------


def test_there_is_no_display_category_and_the_spacing_is_a_field_left_of_the_size_slider(module, qtbot):
    tab, view = module
    model = tab._settings_model
    titles = {str(section.property("settingsCategorySource")) for section in tab.rendered_settings_sections()}
    assert "Display" not in titles
    for key in ("overview_tile_size", "overview_tile_spacing"):
        row = model._widgets[key]
        assert not row.isVisibleTo(tab)
    view._well(view.wells.item(*view._well_positions[(view.plates.currentData(), "A01")]))
    qtbot.waitUntil(lambda: view.fields.count() > 0, timeout=15000)
    panel = view._well_panels[-1]
    layout = panel.header_layout
    assert layout.indexOf(panel.tile_spacing) == layout.indexOf(panel.tile_slider) - 1
    assert panel.tile_spacing.accessibleName() == browser.tr("Overview tile spacing")
    panel.tile_spacing.setValue(12)
    assert view._tile_spacing == 12
    assert model._read_value("overview_tile_spacing") == 12
    assert view.fields.gridSize().width() >= view._tile_size + 12
    model.set_value_for_key("overview_tile_spacing", 3)
    assert view._tile_spacing == 3 and panel.tile_spacing.value() == 3


# --- 5. Normalized channels is a typed list ------------------------------------------------------


def test_the_channel_list_is_checked():
    check = browser._overview_channel_list_error
    assert check("all", (1, 2)) == "" and check("none", (1, 2)) == ""
    assert check("1,2", (1, 2)) == "" and check(" 2 ", (1, 2)) == "" and check([1, 2], (1, 2)) == ""
    assert check("1,2,3,4") == ""
    assert "3" in check("1,3", (1, 2)) and "1, 2" in check("1,3", (1, 2))
    assert check("1,x", (1, 2)) and check("1,,2", (1, 2)) and check("", (1, 2))
    assert check(True)


def test_normalized_channels_is_a_text_field_whose_bad_input_is_shown_and_never_applied(module, qtbot):
    tab, view = module
    model = tab._settings_model
    row = model._widgets["overview_normalize_channels"]
    edit = row if isinstance(row, QLineEdit) else row.findChildren(QLineEdit)[0]
    assert "overview_normalize_channels" not in tab._overview_page._hidden_keys
    assert "overview_normalize_channels" not in model.rows_the_screen_hides()
    assert not any(isinstance(field, browser._OverviewChannelToggles) and field._key == "overview_normalize_channels"
                   for _section, field, _widgets in tab._overview_page._rows)
    bar = tab._overview_page._apply_bar
    edit.setText("1,7")
    assert edit.property("overviewInvalid") is True
    assert "7" in bar.state.text() and not bar.apply_button.isEnabled()
    edit.editingFinished.emit()
    assert view._normalize_channels is None
    assert tab._overview_page._bridge.apply() is False
    assert view._normalize_channels is None and "7" in view.status.text()
    edit.setText("2")
    edit.editingFinished.emit()
    assert edit.property("overviewInvalid") is False and bar.apply_button.isEnabled()
    _apply(tab)
    assert view._normalize_channels == (2,)


# --- 6 and 7. Greyed-out normalization settings and the Advanced normalization category ---------


def test_normalization_keeps_three_settings_and_the_rest_is_advanced_normalization(module):
    tab, view = module
    from spacr.qt.screens.settings_model import _APP_CATEGORY_PARENTS, categories_for_app, get_categories

    layout = categories_for_app("overview", get_categories())
    assert layout["Normalization"] == ["overview_normalize", "overview_percentile_low", "overview_percentile_high"]
    assert {"overview_normalize_channels", "overview_normalization_scope", "overview_advanced_normalization",
            "overview_normalization_method", "overview_gamma", "overview_reference_epsilon"} <= set(
                layout["Advanced normalization"])
    assert _APP_CATEGORY_PARENTS["overview"]["Advanced normalization"] == "Image"
    advanced = _section(tab, "Advanced normalization")
    mounted = {field.property("settingKey") for section, field, _widgets in tab._overview_page._rows
               if section is advanced}
    assert {"overview_reference_group", "overview_channel_to_edit", "overview_normalization_scope"} <= mounted
    plain = _section(tab, "Normalization")
    assert not [field for section, field, _widgets in tab._overview_page._rows if section is plain]


def test_settings_the_normalization_mode_does_not_read_are_greyed_out_live(module):
    tab, _view = module
    model = tab._settings_model

    def enabled(key):
        return model._widgets[key].isEnabled()

    assert enabled("overview_percentile_low") and enabled("overview_normalize_channels")
    for key in ("overview_normalization_method", "overview_advanced_low", "overview_fixed_low",
                "overview_background", "overview_gamma", "overview_reference_channel"):
        assert not enabled(key), key
    model.set_value_for_key("overview_advanced_normalization", True)
    assert enabled("overview_normalization_method") and enabled("overview_advanced_low") and enabled("overview_gamma")
    assert not enabled("overview_fixed_low") and not enabled("overview_reference_channel")
    assert not enabled("overview_normalize_channels")
    model.set_value_for_key("overview_normalization_method", "manual")
    assert enabled("overview_fixed_low") and enabled("overview_fixed_high") and not enabled("overview_advanced_low")
    model.set_value_for_key("overview_normalization_method", "ratio")
    assert enabled("overview_reference_channel") and enabled("overview_reference_epsilon")
    model.set_value_for_key("overview_normalization_method", "difference")
    assert enabled("overview_reference_channel") and not enabled("overview_reference_epsilon")
    model.set_value_for_key("overview_advanced_normalization", False)
    model.set_value_for_key("overview_normalize", False)
    assert not enabled("overview_percentile_low") and not enabled("overview_percentile_high")
    assert "Overview normalize" in model._widgets["overview_percentile_low"].toolTip() or not enabled(
        "overview_percentile_low")
    model.set_value_for_key("overview_normalize", True)
    assert enabled("overview_percentile_low")


# --- 8. Apply ---------------------------------------------------------------------------------


def _pixmap_bytes(view):
    image = view._item.pixmap().toImage()
    return bytes(image.constBits())


def test_display_changes_wait_for_apply_and_the_button_shows_it(module):
    tab, view = module
    model = tab._settings_model
    bar = tab._overview_page._apply_bar
    assert bar.isVisible() and bar.parentWidget() is tab._settings_scroll
    assert tab._settings_scroll.viewportMargins().bottom() == bar.height()
    assert bar.apply_button.property("pending") is False
    model.set_value_for_key("overview_percentile_high", 60.)
    assert view.high.value() == pytest.approx(99.9)
    assert bar.apply_button.property("pending") is True
    assert bar.apply_button.text() == browser.tr("Apply changes")
    assert "Unapplied" in bar.state.text()
    model.set_value_for_key("overview_max_fields", "1")
    assert view._load_limits == (None, 1)
    model.set_value_for_key("overview_max_fields", "all")
    _apply(tab)
    assert view.high.value() == pytest.approx(60.)
    assert bar.apply_button.property("pending") is False and tab._overview_page._bridge.pending() == ()


def test_apply_redraws_an_image_that_is_already_open_and_every_other_view(module, qtbot):
    tab, view = module
    model = tab._settings_model
    field = next(field for field in view._fields if field.well == "A01")
    view._open_image_views([field])
    panel = view._image_views[-1]
    qtbot.waitUntil(lambda: panel.view._item is not None, timeout=15000)
    qtbot.waitUntil(lambda: not view._image_stack_jobs.is_busy(), timeout=15000)
    before, generation = _pixmap_bytes(panel.view), panel.generation
    well = view.wells.item(*view._well_positions[(view.plates.currentData(), "A01")])
    qtbot.waitUntil(lambda: well.data(Qt.DecorationRole) is not None, timeout=15000)
    display_generation = view._display_generation
    model.set_value_for_key("overview_percentile_high", 40.)
    model.set_value_for_key("enhance_gamma", .5)
    qtbot.wait(50)
    assert panel.generation == generation and view._enhancement is None
    _apply(tab)
    assert panel.generation > generation
    assert view._display_generation > display_generation and not view._thumbnail_cache
    qtbot.waitUntil(lambda: not view._image_stack_jobs.is_busy() and _pixmap_bytes(panel.view) != before,
                    timeout=15000)
    assert view._enhancement is not None and view._enhancement.gamma == .5
    assert view._display_kwargs(view=True)["enhancement"] is view._enhancement


def test_an_open_image_is_redrawn_when_the_advanced_normalization_is_ready(module, qtbot):
    tab, view = module
    model = tab._settings_model
    field = next(field for field in view._fields if field.well == "B01")
    view._open_image_views([field])
    panel = view._image_views[-1]
    qtbot.waitUntil(lambda: panel.view._item is not None and not view._image_stack_jobs.is_busy(), timeout=15000)
    model.set_value_for_key("overview_advanced_normalization", True)
    model.set_value_for_key("overview_normalization_method", "manual")
    model.set_value_for_key("overview_fixed_high", 100.)
    _apply(tab)
    qtbot.waitUntil(lambda: view._advanced_model is not None and view._normalization_ready, timeout=15000)
    generation = panel.generation
    loads = []
    original = view._load_image_view
    view._load_image_view = lambda owner: (loads.append(owner), original(owner))
    view._advanced_invalidate()
    qtbot.waitUntil(lambda: view._normalization_ready and panel in loads, timeout=15000)
    assert panel.generation > generation


# --- 9. Image settings window and the per-image global ticks ----------------------------------


def test_the_image_settings_window_has_no_square_behind_it_and_its_levels_take_drags(module, qtbot):
    tab, view = module
    from spacr.qt.widgets.glass import _owns_mouse_gesture, make_frameless

    field = view._fields[0]
    view._open_image_settings(field)
    panel = view._image_settings
    qtbot.waitUntil(panel.isVisible, timeout=5000)
    flags = panel.windowFlags()
    assert not flags & Qt.WindowType.Popup & ~Qt.WindowType.Window
    assert panel.windowType() != Qt.WindowType.Tool
    make_frameless(panel)
    assert panel.testAttribute(Qt.WA_TranslucentBackground)
    assert panel.windowType() == Qt.WindowType.Window
    assert _owns_mouse_gesture(panel.histogram, panel)
    assert not _owns_mouse_gesture(panel.caption, panel)
    histogram = panel.histogram
    histogram.resize(300, 100)
    start = panel.pos()
    x_white = histogram._x(histogram.white)
    target = histogram._x(.5)
    y = histogram.height() - 4
    app = QApplication.instance()
    for kind, x, buttons in ((QEvent.MouseButtonPress, x_white, Qt.LeftButton),
                             (QEvent.MouseMove, (x_white + target) / 2, Qt.LeftButton),
                             (QEvent.MouseMove, target, Qt.LeftButton),
                             (QEvent.MouseButtonRelease, target, Qt.NoButton)):
        local = QPointF(x, y)
        event = QMouseEvent(kind, local, histogram.mapToGlobal(local), Qt.LeftButton, buttons, Qt.NoModifier)
        app.sendEvent(histogram, event)
    assert panel.pos() == start
    assert histogram.white == pytest.approx(.5, abs=.02)
    panel.close()


def test_global_normalization_and_enhancement_ticks_are_per_image_and_saved(module, qtbot, source):
    tab, view = module
    field = next(field for field in view._fields if field.well == "A02")
    other = next(field for field in view._fields if field.well == "B02")
    key = browser._overview_image_key(field)
    view._open_image_views([field])
    panel = view._image_views[-1]
    qtbot.waitUntil(lambda: panel.view._item is not None, timeout=15000)
    view._image_menu(panel.view, QPoint(5, 5))
    menu = view._image_menu_shown
    actions = {action.objectName(): action for action in menu.actions() if action.objectName()}
    normalization = actions["OverviewImage_global_normalization"]
    enhancement = actions["OverviewImage_global_enhancement"]
    assert normalization.isCheckable() and normalization.isChecked() and normalization.text() == browser.tr(
        "Global normalization")
    assert enhancement.isCheckable() and enhancement.isChecked()
    menu.close()
    generation = panel.generation
    normalization.setChecked(False)
    assert view._image_overrides[key] == {"global_normalization": False}
    assert panel.generation > generation
    enhancement.setChecked(False)
    assert view._image_overrides[key] == {"global_normalization": False, "global_enhancement": False}
    assert browser._overview_image_key(other) not in view._image_overrides
    view._image_menu(panel.view, QPoint(5, 5))
    ticks = {action.objectName(): action.isChecked() for action in view._image_menu_shown.actions()
             if action.objectName()}
    assert ticks == {"OverviewImage_global_normalization": False, "OverviewImage_global_enhancement": False}
    view._image_menu_shown.close()
    path = browser._overview_view_state_path(view._metadata_source)
    qtbot.waitUntil(lambda: path.exists() and browser._overview_key_text(key) in json.loads(
        path.read_text()).get("images", {}), timeout=15000)
    saved = json.loads(path.read_text())["images"][browser._overview_key_text(key)]
    assert saved == {"global_normalization": False, "global_enhancement": False}
    normalization.setChecked(True)
    assert view._image_overrides[key] == {"global_enhancement": False}


def test_an_image_without_the_global_layers_skips_normalization_and_enhancement(tmp_path):
    folder = tmp_path / "w"
    folder.mkdir()
    pixels = (np.arange(64 * 64).reshape(64, 64) % 1000).astype(np.uint16)
    pixels[0, 0] = 60000
    tifffile.imwrite(folder / "ch1.tif", pixels)
    field = browser._OverviewField("p1", "A01", "1", 1, "1", ((1, str(folder / "ch1.tif")),))
    key = browser._overview_image_key(field)
    from spacr.qt.detect_chain import NO_CHAIN

    chain = NO_CHAIN._replace(gamma=.4)
    stretched = browser._overview_pixels(field, (1,), (1., 99.), edge=64)
    raw = browser._overview_pixels(field, (1,), None, edge=64)
    enhanced = browser._overview_pixels(field, (1,), (1., 99.), edge=64, enhancement=chain)
    limits = {key: {1: (0., 10.)}}
    skip_norm = {key: {"global_normalization": False}}
    skip_enh = {key: {"global_enhancement": False}}
    assert not np.array_equal(stretched, raw)
    assert np.array_equal(browser._overview_pixels(field, (1,), (1., 99.), edge=64, channel_limits=limits,
                                                   image_overrides=skip_norm), raw)
    assert np.array_equal(browser._overview_pixels(field, (1,), (1., 99.), edge=64, enhancement=chain,
                                                   image_overrides=skip_enh), stretched)
    assert not np.array_equal(enhanced, stretched)
    assert browser._overview_clean_override({"global_normalization": True, "global_enhancement": False}) == {
        "global_enhancement": False}


# --- 10. Measure before Export --------------------------------------------------------------------


def test_measure_comes_before_export(module):
    tab, view = module
    texts = [button.text() for button in view._module_action_buttons]
    assert texts.index(browser.tr("Export image-level measurements")) < texts.index(browser.tr("Export…"))
    from PySide6.QtWidgets import QLayout

    strip = next(layout for layout in tab._btn_overview_measure.parentWidget().findChildren(QLayout)
                 if layout.indexOf(tab._btn_overview_measure) >= 0)
    assert 0 <= strip.indexOf(tab._btn_overview_measure) < strip.indexOf(tab._btn_overview_pdfs)
    from spacr.qt.screens.settings_model import _overview_category_spec

    titles = [title for title, _keys in _overview_category_spec()]
    assert titles.index("Measurements") < titles.index("Export")
    assert titles[-1] == "Export"
