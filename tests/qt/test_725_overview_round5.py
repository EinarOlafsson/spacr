"""Round-5 Overview: collapsible plate rows and columns, categories after Next, wider metadata settings, CMY/CMYK.

Item 725, maintainer feedback 2026-10-25 round 5.
"""
import json
import sys

import numpy as np
import pytest
import tifffile

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QComboBox, QLabel, QPushButton

from spacr.qt.widgets import qc_field_browser as browser


PATTERN = r"(?P<plateID>p\d+)-(?P<wellID>[A-Z]\d+)-f(?P<fieldID>\d+)-c(?P<chanID>\d+)\.tif"
WELLS = ("A01", "A02", "A03", "B01", "B02", "B03")


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    for index, well in enumerate(WELLS):
        for field in (1, 2):
            for channel in (1, 2):
                pixels = (np.arange(60 * 80, dtype=np.uint16).reshape(60, 80) * (channel + 1)
                          + 100 * index + field)
                tifffile.imwrite(folder / f"p1-{well}-f{field}-c{channel}.tif", pixels)
    return folder


def _runners(view):
    return (view._index_jobs, view._image_jobs, view._thumbnail_jobs, view._field_thumbnail_jobs,
            view._montage_jobs, view._export_jobs, view._metadata_jobs, view._measurement_jobs,
            view._native_jobs, view._image_stack_jobs, view._group_jobs)


def _retire(view, qtbot):
    view._cancel()
    qtbot.waitUntil(lambda: all(runner.active_jobs() == 0 for runner in _runners(view)), timeout=15000)
    for runner in _runners(view):
        runner.shutdown(timeout_ms=3000)
    view.close()


@pytest.fixture
def view(qtbot, source):
    browser._OVERVIEW_PLANE_CACHE.clear()
    browser._OVERVIEW_INFO_CACHE.clear()
    dialog = browser._FieldOverview(settings={
        "source": str(source), "source_scope": "folder", "custom_regex": PATTERN,
        "channels": (1, 2), "geometry": (2, 3), "representatives": 1})
    qtbot.addWidget(dialog, before_close_func=lambda widget: _retire(widget, qtbot))
    dialog.resize(1300, 950)
    dialog.show()
    dialog._scan()
    qtbot.waitUntil(lambda: bool(dialog._fields) and dialog._normalization_ready
                    and not dialog._index_jobs.is_busy(), timeout=15000)
    dialog.fov_axis.setCurrentIndex(dialog.fov_axis.findData(None))
    return dialog


def _item(view, well):
    return view.wells.item(*view._well_positions[well])


def _columns(view):
    return [view.wells.columnViewportPosition(column) for column in range(view.wells.columnCount())]


def _rows(view):
    return [view.wells.rowViewportPosition(row) for row in range(view.wells.rowCount())]


def test_collapsing_a_column_brings_its_neighbours_together_and_keeps_selection_and_annotations(view, qtbot):
    view._annotations = {("p1", well): {"cell": "HeLa"} for well in WELLS}
    _item(view, "A01").setSelected(True)
    _item(view, "B03").setSelected(True)
    selected = set(view._selected_wells)
    annotations = {key: dict(value) for key, value in view._annotations.items()}
    assert selected == {("p1", "A01"), ("p1", "B03")}
    before = _columns(view)
    gap = before[2] - before[0]
    view._set_line_collapsed("column", 1, True)
    after = _columns(view)
    assert view.wells.columnWidth(1) == browser._OVERVIEW_COLLAPSED_STRIP
    assert after[2] - after[0] < gap
    assert after[2] - after[1] == browser._OVERVIEW_COLLAPSED_STRIP
    assert view.wells.horizontalHeaderItem(1).text() == ""
    assert "collapsed" in view.wells.horizontalHeaderItem(1).toolTip()
    assert set(view._selected_wells) == selected
    assert _item(view, "A01").isSelected() and _item(view, "B03").isSelected()
    assert view._annotations == annotations
    view._set_line_collapsed("column", 1, False)
    assert _columns(view) == before
    assert view.wells.horizontalHeaderItem(1).text() == "2"


def test_collapsing_a_row_brings_its_neighbours_together_and_a_click_on_its_strip_shows_it(view, qtbot):
    view.resize(1300, 950)
    _item(view, "A02").setSelected(True)
    view.wells.setRowCount(view.wells.rowCount())
    before = _rows(view)
    view._toggle_line("row", 0)
    assert view.wells.rowHeight(0) == browser._OVERVIEW_COLLAPSED_STRIP
    assert _rows(view)[1] < before[1]
    assert view.wells.verticalHeaderItem(0).text() == ""
    assert ("p1", "A02") in view._selected_wells
    view._clicked_well(_item(view, "A01"))
    assert view._collapsed_lines() == (set(), set())
    assert view._scope[1] is None
    assert _rows(view) == before
    assert view.wells.verticalHeaderItem(0).text() == "A"


def test_header_double_click_and_menus_collapse_and_show_lines(view):
    view.wells.horizontalHeader().sectionDoubleClicked.emit(0)
    assert view._collapsed_lines()[1] == {0}
    view.wells.horizontalHeader().sectionDoubleClicked.emit(0)
    assert view._collapsed_lines()[1] == set()
    header = view.wells.verticalHeader()
    menu = view._header_menu("row", QPoint(2, header.sectionViewportPosition(1) + 2))
    texts = [action.text() for action in menu.actions()]
    assert texts == ["Collapse row B", "Show all rows and columns"]
    assert not menu.actions()[1].isEnabled()
    menu.actions()[0].trigger()
    assert view._collapsed_lines()[0] == {1}
    menu.close()
    position = view.wells.visualItemRect(_item(view, "A03")).center()
    menu = view._well_menu(position)
    texts = [action.text() for action in menu.actions()]
    assert "Collapse row A" in texts and "Collapse column 3" in texts
    show_all = next(action for action in menu.actions() if action.text() == "Show all rows and columns")
    assert show_all.isEnabled()
    show_all.trigger()
    assert view._collapsed_lines() == (set(), set())
    menu.close()


def test_collapsed_lines_survive_a_refilter_and_are_kept_per_plate(view):
    view._set_line_collapsed("column", 2, True)
    view._filter()
    assert view.wells.columnWidth(2) == browser._OVERVIEW_COLLAPSED_STRIP
    assert view.wells.horizontalHeaderItem(2).text() == ""
    assert view._collapsed == {"p1": (set(), {2})}
    assert view._collapsed_lines() == (set(), {2})
    view._collapsed["p2"] = ({0}, set())
    view._filter()
    assert view._collapsed_lines() == (set(), {2})


def test_collapsed_wells_are_drawn_as_strips_without_thumbnail_reads(view, qtbot, monkeypatch):
    view._set_line_collapsed("row", 1, True)
    qtbot.waitUntil(lambda: not _item(view, "A01").icon().isNull(), timeout=20000)
    assert view.wells._overview_collapsed == (frozenset({1}), frozenset())
    assert all(_item(view, well).data(Qt.UserRole + 1) is None for well in ("B01", "B02", "B03"))
    picture = view.wells.viewport().grab().toImage()
    rect = view.wells.visualItemRect(_item(view, "B02"))
    strip = picture.pixelColor(rect.center())
    highlight = view.wells.palette().color(view.wells.palette().ColorRole.Highlight)
    assert abs(strip.hue() - highlight.hue()) < 30 or strip.saturation() > 40


INK = {"cyan": (0, 255, 255), "magenta": (255, 0, 255), "yellow": (255, 255, 0)}


def _print(values, mode="cmy", colours=None):
    planes = [np.full((2, 3), value, dtype=np.uint8) for value in values]
    inks = browser._overview_inks(tuple(range(1, len(values) + 1)), colours, mode)
    return browser._overview_subtractive(planes, inks)


@pytest.mark.parametrize("index, ink", [(0, "cyan"), (1, "magenta"), (2, "yellow")])
def test_one_full_channel_prints_exactly_its_ink_on_white(index, ink):
    values = [0, 0, 0]
    values[index] = 255
    rgb = _print(values)
    assert rgb.dtype == np.uint8
    assert (rgb == np.asarray(INK[ink], dtype=np.uint8)).all()


def test_no_signal_is_pure_white_and_full_cyan_magenta_yellow_is_black():
    assert (_print([0, 0, 0]) == 255).all()
    assert (_print([0, 0, 0, 0], "cmyk") == 255).all()
    assert (_print([255, 255, 255]) == 0).all()


def test_the_key_channel_darkens_every_component_alike_and_clips():
    assert (_print([0, 0, 0, 128], "cmyk") == 127).all()
    assert (_print([255, 0, 0, 128], "cmyk") == np.asarray((0, 127, 127), dtype=np.uint8)).all()
    assert (_print([255, 255, 0, 255], "cmyk") == 0).all()
    assert (_print([255, 255, 255, 255], "cmyk") == 0).all()
    overlap = _print([200, 200, 0])
    assert (overlap == np.asarray((55, 55, 255), dtype=np.uint8)).all()


def test_inks_follow_the_channel_colours_and_unused_inks_are_absent():
    inks = browser._overview_inks((1, 2), {2: "#00ffff", 1: "#ff0000"}, "cmy")
    assert inks == [browser._OVERVIEW_INKS["magenta"], browser._OVERVIEW_INKS["cyan"]]
    assert browser._overview_inks((1,), None, "cmyk") == [browser._OVERVIEW_INKS["cyan"]]
    assert browser._overview_inks((1, 2, 3, 4), None, "cmy") is None
    assert browser._overview_inks((1, 2), None, "additive") is None
    assert browser._overview_inks((1, 2), {1: "#000000"}, "cmyk")[0] == browser._OVERVIEW_INKS["black"]


def test_overview_pixels_prints_two_channels_on_white(view):
    field = view._fields[0]
    additive = browser._overview_pixels(field, (1, 2), (0., 100.), edge=64)
    printed = browser._overview_pixels(field, (1, 2), (0., 100.), edge=64, color_mode="cmy")
    assert additive.shape == printed.shape
    assert (additive[0, 0] == 0).all() and (printed[0, 0] == 255).all()
    assert (printed[-1, -1] == np.asarray((0, 0, 255), dtype=np.uint8)).all()


def test_switching_the_colour_mode_updates_the_plate_well_and_image_views(view, qtbot, monkeypatch):
    qtbot.waitUntil(lambda: all(not _item(view, well).icon().isNull() for well in WELLS), timeout=20000)
    view._well(_item(view, "A01"))
    view._well(_item(view, "B03"))
    qtbot.waitUntil(lambda: len(view._image_views) == 2
                    and all(panel.view._item is not None for panel in view._image_views), timeout=15000)
    calls = []
    real = browser._overview_pixels

    def recorded(*args, **kwargs):
        calls.append((sys._getframe(1).f_code.co_qualname, kwargs.get("color_mode")))
        return real(*args, **kwargs)

    monkeypatch.setattr(browser, "_overview_pixels", recorded)
    view._set_color_mode("cmyk")
    assert view._display_kwargs(view=True)["color_mode"] == "cmyk"

    def reached():
        callers = {caller for caller, mode in list(calls) if mode == "cmyk"}
        return (any("read_thumbnails" in caller for caller in callers)
                and any("_update_field_thumbnails" in caller for caller in callers)
                and any("_load_image_view" in caller for caller in callers))

    qtbot.waitUntil(reached, timeout=20000)
    qtbot.waitUntil(lambda: all("'color_mode': 'cmyk'" in str(_item(view, well).data(Qt.UserRole + 1))
                                for well in WELLS), timeout=20000)
    view._set_color_mode("unknown")
    assert view._color_mode == "additive"


def test_exports_record_the_colour_mode():
    lines = browser._overview_pdf_header((1, 2), (1., 99.), 0, None, "cmyk")
    assert any("CMYK inks on white" in line for line in lines)
    assert not any("Colour mode" in line for line in browser._overview_pdf_header((1,), (1., 99.), 0, None))


# --- The Overview tab: categories after Next, wider metadata settings, live colour mode ---


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

    screen = AppScreen("mask")
    qtbot.addWidget(screen, before_close_func=lambda widget: _close(widget, qtbot))
    screen.resize(1500, 1000)
    screen.show()
    return screen


def _open_tab(mask, source):
    mask._settings_model.set_value_for_key("src", str(source))
    field = mask._settings_model._widgets.get("src")
    next(action for action in field.actions()
         if action.objectName() == "GenerateMasksFieldOverview").trigger()
    return mask._overview_tab


def _next(tab, qtbot):
    wizard = tab._overview_page.wizard
    wizard.preset.setCurrentIndex(wizard.preset.count() - 1)
    wizard.regex.setText(PATTERN)
    wizard._scan()
    qtbot.waitUntil(lambda: wizard.next.isEnabled(), timeout=15000)
    wizard._next()
    page = tab._overview_page
    qtbot.waitUntil(lambda: page.browser is not None and len(page.browser._fields) == 2 * len(WELLS)
                    and page.browser._normalization_ready, timeout=15000)
    return page.browser


def _shown_titles(tab):
    return {str(section.property("settingsCategorySource")) for section in tab.rendered_settings_sections()
            if not section.isHidden()}


def test_categories_are_held_back_until_next(mask, qtbot, source):
    tab = _open_tab(mask, source)
    page = tab._overview_page
    qtbot.waitUntil(lambda: sum(tab._body_splitter.sizes()) > 0, timeout=5000)
    assert page.stack.currentWidget() is page.wizard
    assert _shown_titles(tab) == set()
    assert tab._body_splitter.is_collapsed("Settings")
    tab.refresh_maturity_visibility()
    assert _shown_titles(tab) == set()
    _next(tab, qtbot)
    qtbot.wait(50)
    assert {"Source", "Image", "Figure settings", "Comparisons", "Export", "Measurements"} <= _shown_titles(tab)
    assert not tab._body_splitter.is_collapsed("Settings")
    page._retire_browser()
    assert page.stack.currentWidget() is page.placeholder
    assert _shown_titles(tab) == {"Source"}


def test_the_metadata_settings_open_wide_with_nothing_cut_off(mask, qtbot, source):
    tab = _open_tab(mask, source)
    wizard = tab._overview_page.wizard
    split = wizard._wizard_content
    qtbot.waitUntil(lambda: sum(split.sizes()) > 0, timeout=5000)
    total = sum(split.sizes())
    assert total * 3 // 5 >= browser._OVERVIEW_SETTINGS_WIDTH
    assert split.sizes()[0] >= browser._OVERVIEW_SETTINGS_WIDTH
    qtbot.waitUntil(lambda: not wizard._settings_scroll.horizontalScrollBar().isVisible(), timeout=3000)
    content = wizard._settings_content
    for label in content.findChildren(QLabel):
        if label.isVisibleTo(content) and not label.wordWrap() and label.text():
            assert label.width() >= label.sizeHint().width(), label.text()
    for combo in content.findChildren(QComboBox):
        if combo.isVisibleTo(content) and combo.currentText():
            assert combo.fontMetrics().horizontalAdvance(combo.currentText()) + 24 <= combo.width(), combo.currentText()
    for button in content.findChildren(QPushButton):
        if button.isVisibleTo(content):
            assert button.width() >= button.sizeHint().width(), button.text()
    split.splitterMoved.emit(300, 1)
    split.setSizes([300, total - 300])
    wizard.resize(wizard.width() + 1, wizard.height())
    qtbot.wait(20)
    assert split.sizes()[0] < browser._OVERVIEW_SETTINGS_WIDTH


def test_the_colour_mode_setting_is_live_in_the_tab(mask, qtbot, source):
    tab = _open_tab(mask, source)
    view = _next(tab, qtbot)
    model = tab._settings_model
    assert model._widgets.get("overview_color_mode") is not None
    model.set_value_for_key("overview_color_mode", "cmy")
    assert view._color_mode == "cmy"
    model.set_value_for_key("overview_color_mode", "additive")
    assert view._color_mode == "additive"


# --- Shared interface: plate-qualified keys, saved view state, display layers ---


def _field(source, plate="p1", well="A01", field="1"):
    return browser._OverviewField(plate, well, "1", 1, field, (), field_id=field, source=source, plate_id=plate)


def test_plate_and_image_keys_keep_plates_from_different_sources_apart():
    first, second = _field("/data/a"), _field("/data/b")
    assert browser._overview_plate_key(first) == ("/data/a", "p1")
    assert browser._overview_image_key(first) != browser._overview_image_key(second)
    assert browser._overview_image_key(first) == ("/data/a", "p1", "A01", "1", "", "")
    legacy = browser._OverviewField("p1", "A01", "1", 1, "1", ())
    assert browser._overview_plate_key(legacy) == ("", "p1")
    import json
    key = browser._overview_image_key(first)
    assert tuple(json.loads(browser._overview_key_text(key))) == key


def test_the_view_state_saves_and_loads_and_refuses_a_newer_version(tmp_path):
    assert browser._overview_load_view_state(tmp_path) == browser._overview_empty_view_state()
    state = browser._overview_empty_view_state()
    state["plates"]['["/x", "p1"]'] = {"collapsed_rows": [1], "collapsed_columns": []}
    state["images"]['["/x", "p1", "A01", "1", "", ""]'] = {"channels": {"1": {"black": .1}}}
    path = browser._overview_save_view_state(tmp_path, state)
    assert path == tmp_path / "results" / "overview_view.json"
    loaded = browser._overview_load_view_state(tmp_path)
    assert loaded["version"] == browser._OVERVIEW_VIEW_STATE_VERSION
    assert loaded["plates"] == state["plates"] and loaded["images"] == state["images"]
    path.write_text('{"version": 99, "plates": {"a": {}}}', encoding="utf-8")
    assert browser._overview_load_view_state(tmp_path)["plates"] == {}
    path.write_text("not json", encoding="utf-8")
    assert browser._overview_load_view_state(tmp_path)["plates"] == {}


def test_collapsed_lines_are_saved_and_restored_on_reopen(view, qtbot, source):
    view._set_line_collapsed("row", 1, True)
    view._set_line_collapsed("column", 2, True)
    path = source / "results" / "overview_view.json"
    qtbot.waitUntil(lambda: path.is_file() and '"collapsed_columns": [\n        2' in path.read_text(), timeout=10000)
    reopened = browser._FieldOverview(settings={
        "source": str(source), "source_scope": "folder", "custom_regex": PATTERN,
        "channels": (1, 2), "geometry": (2, 3), "representatives": 1})
    qtbot.addWidget(reopened, before_close_func=lambda widget: _retire(widget, qtbot))
    reopened.resize(1300, 950)
    reopened.show()
    reopened._scan()
    qtbot.waitUntil(lambda: reopened._collapsed.get("p1") == ({1}, {2}), timeout=15000)
    qtbot.waitUntil(lambda: reopened.wells.columnWidth(2) == browser._OVERVIEW_COLLAPSED_STRIP, timeout=5000)
    saved = json.loads(path.read_text())
    assert [(entry["collapsed_rows"], entry["collapsed_columns"]) for entry in saved["plates"].values()] == [([1], [2])]
    assert reopened._collapsed["p1"] == ({1}, {2})
    assert reopened.wells.columnWidth(2) == browser._OVERVIEW_COLLAPSED_STRIP
    assert reopened.wells.rowHeight(1) == browser._OVERVIEW_COLLAPSED_STRIP
    assert reopened.wells.columnWidth(1) > browser._OVERVIEW_COLLAPSED_STRIP
    assert reopened.wells.rowHeight(0) > browser._OVERVIEW_COLLAPSED_STRIP


def test_display_layers_apply_global_then_scope_limits_then_the_image_override(view):
    field = view._fields[0]
    key = browser._overview_image_key(field)
    raw = dict((channel, values) for channel, _path, values in
               browser._overview_raw_planes(field, (1,), edge=None))[1].astype(np.float64)
    low, high = float(raw.min()), float(raw.max())
    limited = browser._overview_pixels(field, (1,), (0., 100.), edge=None,
                                       channel_limits={key: {1: (low, (low + high) / 2)}})
    expected = np.uint8(np.clip((raw - low) / ((high - low) / 2), 0, 1) * 255)
    assert np.abs(limited[..., 0].astype(int) - expected.astype(int)).max() <= 1
    override = {key: {"channels": {1: {"black": .25, "white": .75}}}}
    both = browser._overview_pixels(field, (1,), (0., 100.), edge=None,
                                    channel_limits={key: {1: (low, high)}}, image_overrides=override)
    stretched = np.clip((raw - low) / (high - low), 0, 1)
    expected = np.uint8(np.clip((stretched - .25) / .5, 0, 1) * 255)
    assert np.abs(both[..., 0].astype(int) - expected.astype(int)).max() <= 1
    other = view._fields[1]
    assert browser._overview_image_key(other) != key
    untouched = browser._overview_pixels(other, (1,), (0., 100.), edge=None, image_overrides=override)
    plain = browser._overview_pixels(other, (1,), (0., 100.), edge=None)
    assert (untouched == plain).all()


def test_the_shown_field_is_found_from_its_view(view, qtbot):
    view._well(_item(view, "A01"))
    view._well(_item(view, "B03"))
    qtbot.waitUntil(lambda: len(view._image_views) == 2, timeout=15000)
    panel = view._image_views[1]
    assert view._field_shown_in(panel.view) is panel.field
    assert view._plate_key() == (view._fields[0].source, "p1") and view._fields[0].source
    view._set_display_layers(image_overrides={browser._overview_image_key(panel.field): {"saturation": 0.}})
    assert "image_overrides" in view._display_kwargs(view=True)


def test_the_load_limits_are_source_settings_and_live_in_the_tab(mask, qtbot, source):
    tab = _open_tab(mask, source)
    view = _next(tab, qtbot)
    model = tab._settings_model
    from spacr.qt.screens.settings_model import categories_for_app, get_categories

    assert {"overview_max_wells", "overview_max_fields"} <= set(categories_for_app("overview", get_categories())["Source"])
    assert model._read_value("overview_max_wells") == "all"
    model.set_value_for_key("overview_max_wells", "2")
    assert view._load_limits == (2, None)
    assert {field.well for field in view._fields} == {"A01", "A02"}
    model.set_value_for_key("overview_max_fields", "1")
    assert view._load_limits == (2, 1) and len(view._fields) == 2
    model.set_value_for_key("overview_max_wells", "all")
    model.set_value_for_key("overview_max_fields", "all")
    assert len(view._fields) == 2 * len(WELLS)


def test_the_normalization_scope_is_one_choice_per_channel_in_the_tab(mask, qtbot, source):
    tab = _open_tab(mask, source)
    view = _next(tab, qtbot)
    model = tab._settings_model
    scopes = next(field for _section, field, _widgets in tab._overview_page._rows
                  if isinstance(field, browser._OverviewChannelScopes))
    qtbot.waitUntil(lambda: len(scopes._choices) == 2, timeout=15000)
    assert model._read_value("overview_normalization_scope") == "image"
    first, second = scopes._choices.values()
    first.setCurrentIndex(first.findData("plate"))
    assert model._read_value("overview_normalization_scope") == "1=plate, 2=image"
    assert view._normalization_scopes == {"1": "plate", "2": "image"}
    model.set_value_for_key("overview_normalization_scope", "well")
    assert first.currentData() == "well" and second.currentData() == "well"
    assert view._normalization_scopes == {"*": "well"}
