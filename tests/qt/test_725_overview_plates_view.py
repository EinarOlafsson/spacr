"""Overview Plates view: several loaded plates shown at once, side by side or stacked.

Item 725, 1.5.1.6: round 5 loaded several plates but showed one at a time,
because the plate table and the well-position map were keyed by well only.
They are keyed by plate and well now, so three plates that share their well
names are shown together without one plate's A01 meeting another's.
"""
import json

import numpy as np
import pytest
import tifffile

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QBoxLayout

from spacr.qt.widgets import qc_field_browser as browser


PATTERN = r"(?P<plateID>p\d+)-(?P<wellID>[A-Z]\d+)-f(?P<fieldID>\d+)-c(?P<chanID>\d+)\.tif"
PLATES = ("plateA", "plateB", "plateC")


def _plate(folder, wells, offset):
    folder.mkdir(parents=True, exist_ok=True)
    for index, well in enumerate(wells):
        for channel in (1, 2):
            pixels = np.full((24, 32), 100 * (index + 1) + offset + channel, dtype=np.uint16)
            tifffile.imwrite(folder / f"p1-{well}-f1-c{channel}.tif", pixels)
    return folder


@pytest.fixture
def plates(tmp_path):
    return (_plate(tmp_path / "plateA", ("A01", "A02", "B01"), 0),
            _plate(tmp_path / "plateB", ("A01", "A02", "B01", "B02"), 1000),
            _plate(tmp_path / "plateC", ("A01", "B03"), 2000))


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


def _open(qtbot, source, **settings):
    browser._OVERVIEW_PLANE_CACHE.clear()
    browser._OVERVIEW_INFO_CACHE.clear()
    view = browser._FieldOverview(settings={
        "source": source, "source_scope": "folder", "custom_regex": PATTERN,
        "channels": (1, 2), "geometry": (2, 3), "representatives": 1, **settings})
    qtbot.addWidget(view, before_close_func=lambda widget: _retire(widget, qtbot))
    view.resize(1500, 1000)
    view.show()
    view._scan()
    qtbot.waitUntil(lambda: bool(view._fields) and view._normalization_ready
                    and not view._index_jobs.is_busy(), timeout=15000)
    return view


@pytest.fixture
def view(qtbot, plates):
    return _open(qtbot, "; ".join(str(path) for path in plates))


def _item(view, plate, well):
    return view._plate_tables[plate].item(*view._well_positions[(plate, well)])


def _header(view, plate):
    return view._plate_blocks[view._plate_tables[plate]][1]


def _hover(view, plate, well):
    table = view._plate_tables[plate]
    point = QPointF(table.visualItemRect(_item(view, plate, well)).center())
    event = QMouseEvent(QEvent.MouseMove, point, table.viewport().mapToGlobal(point),
                        Qt.NoButton, Qt.NoButton, Qt.NoModifier)
    QApplication.sendEvent(table.viewport(), event)


@pytest.mark.parametrize("mode, direction, axis", [("side", QBoxLayout.LeftToRight, "x"),
                                                    ("stacked", QBoxLayout.TopToBottom, "y")])
def test_side_by_side_and_stacked_show_every_plate_under_its_name(view, qtbot, mode, direction, axis):
    assert view.plates_view.isEnabled()
    view.plates_view.setCurrentIndex(view.plates_view.findData(mode))
    assert view._plates_view == mode and tuple(view._plate_tables) == PLATES
    assert view._plates_layout.direction() == direction
    tables = list(view._plate_tables.values())
    assert len({id(table) for table in tables}) == 3 and view.wells is view._plate_tables[view.plates.currentData()]
    qtbot.waitUntil(lambda: all(table.isVisible() and _header(view, plate).isVisible()
                                for plate, table in view._plate_tables.items()), timeout=5000)
    assert [_header(view, plate).text() for plate in PLATES] == [f"Plate {plate}" for plate in PLATES]
    origins = [getattr(view._plate_blocks[table][0].geometry(), axis)() for table in tables]
    assert origins == sorted(origins) and len(set(origins)) == 3
    assert set(view._well_positions) == {("plateA", "A01"), ("plateA", "A02"), ("plateA", "B01"),
                                         ("plateB", "A01"), ("plateB", "A02"), ("plateB", "B01"), ("plateB", "B02"),
                                         ("plateC", "A01"), ("plateC", "B03")}
    for plate in PLATES:
        assert _item(view, plate, "A01").data(Qt.UserRole) == "A01"
        assert all(field.plate == plate for field in view._well_fields[(plate, "A01")])
    qtbot.waitUntil(lambda: all(not _item(view, plate, "A01").icon().isNull() for plate in PLATES), timeout=20000)
    keys = {plate: _item(view, plate, "A01").data(Qt.UserRole + 1)[0][0] for plate in PLATES}
    assert {key[0] for key in keys.values()} == set(PLATES)


def test_a_chosen_subset_is_shown_and_steps_stay_within_it(view):
    view._set_plates_view("side", ["plateA", "plateC"])
    assert tuple(view._plate_tables) == ("plateA", "plateC")
    assert not any(address[0] == "plateB" for address in view._well_positions)
    menu = view._fill_shown_plates_menu()
    ticks = {action.text(): action.isChecked() for action in menu.actions() if action.isCheckable()}
    assert ticks == {"All plates": False, "plateA": True, "plateB": False, "plateC": True}
    view.plates.setCurrentIndex(view.plates.findData("plateA"))
    view._plate_buttons[1].click()
    assert view.plates.currentData() == "plateC"
    view._plate_buttons[1].click()
    assert view.plates.currentData() == "plateA"
    view._toggle_shown_plate("plateC", False)
    assert tuple(view._plate_tables) == ("plateA",)
    view._toggle_shown_plate("plateA", False)
    assert tuple(view._plate_tables) == ("plateA",)
    view._show_every_plate()
    assert tuple(view._plate_tables) == PLATES and view._plates_subset is None


def test_selecting_a01_on_one_plate_never_selects_a01_on_another(view, qtbot):
    view._set_plates_view("side")
    _item(view, "plateB", "A01").setSelected(True)
    assert view._selected_wells == {("plateB", "A01")}
    assert not _item(view, "plateA", "A01").isSelected() and not _item(view, "plateC", "A01").isSelected()
    table = view._plate_tables["plateA"]
    rect = table.visualItemRect(_item(view, "plateA", "A02"))
    qtbot.mouseClick(table.viewport(), Qt.LeftButton, Qt.ControlModifier, rect.center())
    assert view._selected_wells == {("plateB", "A01"), ("plateA", "A02")}
    assert _item(view, "plateB", "A01").isSelected() and not _item(view, "plateB", "A02").isSelected()
    view._filter()
    assert _item(view, "plateB", "A01").isSelected() and _item(view, "plateA", "A02").isSelected()
    assert not _item(view, "plateA", "A01").isSelected()
    view._clear_selection()
    assert view._selected_wells == set()
    assert not any(table.selectedItems() for table in view._plate_tables.values())


def test_the_hover_readout_names_the_plate_under_the_cursor(view, qtbot):
    view._set_plates_view("stacked")
    for plate in ("plateC", "plateA", "plateB"):
        _hover(view, plate, "A01")
        qtbot.waitUntil(lambda plate=plate: view._pixel_readout.text().startswith(f"Plate {plate}"), timeout=5000)
        assert "Well A01" in view._pixel_readout.text()
        assert all(f"Plate {other}" not in view._pixel_readout.text() for other in PLATES if other != plate)


def test_collapse_is_kept_per_plate_and_fit_and_zoom_apply_to_every_plate(view, qtbot):
    view._set_plates_view("side")
    strip = browser._OVERVIEW_COLLAPSED_STRIP
    view._plate_tables["plateB"].verticalHeader().sectionDoubleClicked.emit(0)
    assert view._collapsed["plateB"] == ({0}, set())
    assert view._collapsed.get("plateA", (set(), set())) == (set(), set())
    assert view._plate_tables["plateB"].rowHeight(0) == strip
    assert view._plate_tables["plateA"].rowHeight(0) != strip
    assert view._plate_tables["plateC"].rowHeight(0) != strip
    assert view.plates.currentData() == "plateB"
    view._plate_toggle_line(view._plate_tables["plateC"], "column", 2)
    assert view._collapsed["plateC"] == (set(), {2}) and view._collapsed["plateB"] == ({0}, set())
    widths = {plate: table.columnWidth(0) for plate, table in view._plate_tables.items()}
    view._zoom_plate(1)
    assert all(table.columnWidth(0) > widths[plate] for plate, table in view._plate_tables.items())
    assert len({table.columnWidth(0) for table in view._plate_tables.values()}) == 1
    view._zoom_plate()
    for table in view._plate_tables.values():
        assert table.horizontalScrollBar().maximum() == 0 and table.verticalScrollBar().maximum() == 0
        assert sum(table.columnWidth(index) for index in range(table.columnCount())) <= table.viewport().width()


def test_not_loaded_wells_are_hatched_on_each_plate(qtbot, plates):
    view = _open(qtbot, "; ".join(str(path) for path in plates))
    view._set_load_limits("1", None)
    view._set_plates_view("stacked")
    hatched = {plate: sorted(table.item(row, column).text().split("\n")[0]
                             for row in range(table.rowCount()) for column in range(table.columnCount())
                             if table.item(row, column) is not None
                             and table.item(row, column).data(browser._OVERVIEW_NOT_LOADED_ROLE))
               for plate, table in view._plate_tables.items()}
    assert hatched == {"plateA": ["A02", "B01"], "plateB": ["A02", "B01", "B02"], "plateC": ["B03"]}


def test_a_cross_plate_selection_feeds_a_comparison_group_and_the_guided_export(view, qtbot):
    view._set_plates_view("side")
    _item(view, "plateA", "A01").setSelected(True)
    table = view._plate_tables["plateC"]
    rect = table.visualItemRect(_item(view, "plateC", "A01"))
    qtbot.mouseClick(table.viewport(), Qt.LeftButton, Qt.ControlModifier, rect.center())
    assert view._selected_wells == {("plateA", "A01"), ("plateC", "A01")}
    dialog = view._annotate_selected_wells()
    assert set(dialog.wells) == {("plateA", "A01"), ("plateC", "A01")}
    dialog.key.setEditText("set")
    dialog.value.setText("x")
    dialog.accept()
    snapshot = view._annotation_snapshot()
    assert snapshot[("plateA", "A01")]["set"] == snapshot[("plateC", "A01")]["set"] == "x"
    assert "set" not in snapshot.get(("plateB", "A01"), {})
    assert browser._overview_group_count(view._fields, snapshot, {"set": "x"}) == 2
    builder = view.group_builder
    assert builder._counter({"set": "x"}) == 2
    guided = view._open_guided_export()
    assert set(guided.chosen_wells()) == {("plateA", "A01"), ("plateC", "A01")}
    groups = guided._condition_options()
    assert set(groups["set: x"]) == {("plateA", "A01"), ("plateC", "A01")}
    guided.close()


def test_the_plates_view_survives_saving_and_reopening(view, qtbot, plates):
    view._set_plates_view("stacked", ["plateA", "plateC"])
    path = plates[0] / "results" / "overview_view.json"
    qtbot.waitUntil(lambda: path.is_file() and '"stacked"' in path.read_text(), timeout=10000)
    saved = json.loads(path.read_text())["view"]
    assert saved["plates_view"] == "stacked"
    assert [json.loads(text)[1] for text in saved["shown_plates"]] == ["p1", "p1"]
    reopened = _open(qtbot, "; ".join(str(path) for path in plates))
    qtbot.waitUntil(lambda: reopened._plates_view == "stacked", timeout=10000)
    assert tuple(reopened._plate_tables) == ("plateA", "plateC")
    assert reopened.plates_view.currentData() == "stacked"
    reopened._set_plates_view("single")
    qtbot.waitUntil(lambda: '"single"' in path.read_text(), timeout=10000)
    again = _open(qtbot, "; ".join(str(path) for path in plates))
    qtbot.waitUntil(lambda: again._view_state.get("view", {}).get("plates_view") == "single", timeout=10000)
    assert again._plates_view == "single" and len(again._plate_tables) == 1


def test_single_mode_is_unchanged(view, qtbot):
    original = view.wells
    assert view._plates_view == "single" and view.plates_view.currentData() == "single"
    assert tuple(view._plate_tables) == (view.plates.currentData(),)
    assert not any(header.isVisible() for _block, header in view._plate_blocks.values())
    for plate in PLATES:
        view.plates.setCurrentIndex(view.plates.findData(plate))
        assert view.wells is original and tuple(view._plate_tables) == (plate,)
        assert {address[0] for address in view._well_positions} == {plate}
    view._set_plates_view("side")
    assert len(view._plate_tables) == 3
    view._set_plates_view("single")
    assert len(view._plate_tables) == 1 and view.wells is view._plate_tables[view.plates.currentData()]
    assert sum(block.isVisible() for block, _header in view._plate_blocks.values()) == 1
    assert not any(header.isVisible() for _block, header in view._plate_blocks.values())


def test_plate_visibility_and_collapse_are_idempotent_and_preserve_loaded_selection(view, qtbot):
    before = tuple(field.identity for field in view._fields)
    view.plates_view.setCurrentIndex(view.plates_view.findData('side'))
    address = ('plateB', 'A01')
    _item(view, *address).setSelected(True)
    selected = set(view._selected_wells)
    assert address in selected
    assert view._set_plates_expanded(('plateB',), False)
    assert not view._set_plates_expanded(('plateB',), False)
    assert set(view._selected_wells) == selected
    assert view._set_plates_expanded(('plateB',), True)
    assert not view._set_plates_expanded(('plateB',), True)
    assert view._hide_every_plate(persist=False)
    assert not view._hide_every_plate(persist=False)
    assert view._plate_tables == {} and view._plates_subset == ()
    assert tuple(field.identity for field in view._fields) == before
    view._show_every_plate()
    qtbot.waitUntil(lambda: len(view._plate_tables) == 3)
    assert set(view._selected_wells) == selected
    assert tuple(field.identity for field in view._fields) == before


def test_opening_wells_closes_only_unrelated_stacked_images_and_ignores_unknown_addresses(view, qtbot):
    view.plates_view.setCurrentIndex(view.plates_view.findData('side'))
    before = tuple(field.identity for field in view._fields)
    assert view._open_wells([('missing', 'A01')]) == ()
    assert 'no image' in view.status.text()
    opened = [('plateA', 'A01'), ('plateB', 'A01')]
    assert view._open_wells(opened + opened) == tuple(opened)
    fields = [next(field for field in view._fields if (field.plate, field.well) == address)
              for address in opened]
    view._open_image_views(fields)
    qtbot.waitUntil(lambda: view._image_stack_jobs.active_jobs() == 0
                    and all(panel.view._item is not None for panel in view._image_views), timeout=15000)
    assert [(panel.field.plate, panel.field.well) for panel in view._image_views] == opened
    view._open_image_views(fields)
    assert len(view._image_views) == 2
    assert view._open_wells([('plateC', 'A01')]) == (('plateC', 'A01'),)
    assert view._image_views == []
    assert [(panel.field_list._overview_plate, panel.field_list._overview_well)
            for panel in view._well_panels] == [('plateC', 'A01')]
    assert tuple(field.identity for field in view._fields) == before
    view._selected_fields.clear()
    view._open_selected_images()
    assert 'Select fields' in view.status.text() and view._image_views == []


def test_removing_a_plate_drops_only_its_selection_and_cache_and_cannot_remove_the_last(view, qtbot):
    view.plates_view.setCurrentIndex(view.plates_view.findData('side'))
    removed = tuple(field for field in view._all_fields if field.plate == 'plateB')
    keep = tuple(field for field in view._all_fields if field.plate != 'plateB')
    field = removed[0]
    address = (field.plate, field.well)
    view._selected_wells.add(address)
    view._selected_well_axes[address] = {'time_index': 0}
    view._selected_well_options[address] = {'color_mode': 'additive'}
    view._selected_fields[field.identity] = field
    view._selected_field_options[field.identity] = {'color_mode': 'additive'}
    view._collapsed[field.plate] = True
    view._collapsed_plates.add(field.plate)
    view._plates_subset = ('plateA', 'plateB')
    view._cache[(field.identity, 'preview')] = np.zeros((2, 2, 3), dtype=np.uint8)
    view._thumbnail_cache[(field.identity, 'thumb')] = np.zeros((2, 2, 3), dtype=np.uint8)
    view._remove_plate('plateB')
    assert tuple(view._all_fields) == keep
    assert address not in view._selected_wells and address not in view._selected_well_axes
    assert field.identity not in view._selected_fields and field.identity not in view._selected_field_options
    assert not any(key[0] == field.identity for key in view._cache)
    assert not any(key[0] == field.identity for key in view._thumbnail_cache)
    assert 'plateB' not in view._collapsed and 'plateB' not in view._collapsed_plates
    plates_path = keep[0].source
    assert str(plates_path) in view.source.text()
    assert all(not path.endswith('plateB') for path in browser._overview_source_paths(view.source.text()))
    assert browser.Path(removed[0].files[0][1]).is_file()
    view._remove_plate('absent')
    assert tuple(view._all_fields) == keep
    view._remove_plate('plateC')
    last = tuple(view._all_fields)
    assert {field.plate for field in last} == {'plateA'}
    view._remove_plate('plateA')
    assert tuple(view._all_fields) == last and str(plates_path) in view.source.text()


def test_adding_the_same_source_is_inert_and_late_source_results_do_not_replace_navigation(view):
    original = view._fields
    submitted = view._index_jobs._next_id
    view._add_sources(view.source.text())
    assert view._index_jobs._next_id == submitted
    view._sources_added(['/not-applied'], tuple(list(original)), ((), {}))
    assert view._fields is original and '/not-applied' not in view.source.text()


def test_container_wheel_zoom_tracks_one_actual_well_and_keeps_all_plate_identities(view, qtbot):
    view.plates_view.setCurrentIndex(view.plates_view.findData('side'))
    table = view._plate_tables['plateB']
    before = tuple(field.identity for field in view._fields)
    position = table.visualItemRect(_item(view, 'plateB', 'A01')).center()
    width = view._plate_extents[table][0]
    view._wheel_zoom_container(table, position, 120)
    assert view._plate_extents[table][0] > width
    grown = view._plate_extents[table][0]
    view._wheel_zoom_container(table, position, -120)
    assert view._plate_extents[table][0] < grown
    assert tuple(field.identity for field in view._fields) == before
    assert set(view._plate_tables) == set(PLATES)


def _settle_plate_auxiliary_jobs(view, qtbot):
    runners = (*_runners(view), view._readout_jobs, view._levels_jobs)
    qtbot.waitUntil(lambda: all(runner.active_jobs() == 0 for runner in runners), timeout=15000)


def _stacked_plate_image(view, qtbot, plate='plateC'):
    field = next(field for field in view._fields if field.plate == plate and field.well == 'A01')
    view._open_image_views([field])
    panel = next(panel for panel in view._image_views if panel.field.identity == field.identity)
    _settle_plate_auxiliary_jobs(view, qtbot)
    assert panel.view._item is not None and not panel.view._item.pixmap().isNull()
    return panel


def _move_plate_image_cursor(image_view):
    point = image_view.viewportTransform().map(image_view._item.mapToScene(QPointF(1.5, 1.5)))
    event = QMouseEvent(QEvent.MouseMove, point, image_view.viewport().mapToGlobal(point),
                        Qt.NoButton, Qt.NoButton, Qt.NoModifier)
    QApplication.sendEvent(image_view.viewport(), event)
    return point


@pytest.mark.parametrize('decoration', ['null_icon', 'missing_shape', 'outside_decoration'])
def test_incomplete_well_decoration_never_invents_a_pixel_readout(view, qtbot, decoration):
    from PySide6.QtGui import QIcon

    view._set_plates_view('side')
    assert view._open_wells([('plateB', 'A01')]) == (('plateB', 'A01'),)
    _settle_plate_auxiliary_jobs(view, qtbot)
    fields = view._well_panels[0].field_list
    fields.doItemsLayout()
    item = fields.item(0)
    assert item is not None and not item.icon().isNull()
    original_icon, original_shape = item.icon(), item.data(Qt.UserRole + 3)
    point = QPointF(fields.visualItemRect(item).bottomRight())
    try:
        if decoration == 'null_icon':
            item.setIcon(QIcon())
        elif decoration == 'missing_shape':
            item.setData(Qt.UserRole + 3, None)
        assert view._readout_target(fields, point) is None
        view._readout_pending = fields, point
        view._pixel_readout.setText('old readout')
        view._update_pixel_readout()
        assert view._pixel_readout.text() == ''
        assert view._readout_loading is None
        fields.doItemsLayout()
        rectangle = fields.visualItemRect(item)
        assert rectangle.isValid() and rectangle.intersects(fields.viewport().rect())
        hit = rectangle.center()
        assert fields.itemAt(hit) is item
        assert view._field_shown_in(fields, hit) is item.data(Qt.UserRole)
        assert view._field_shown_in(fields, fields.viewport().rect().bottomRight()) is None
    finally:
        item.setIcon(original_icon)
        item.setData(Qt.UserRole + 3, original_shape)
        view._clear_pixel_readout()
        _settle_plate_auxiliary_jobs(view, qtbot)


@pytest.mark.parametrize('state', ['null_pixmap', 'singular_transform'])
def test_stacked_image_readout_refuses_unavailable_display_geometry(view, qtbot, state):
    from PySide6.QtGui import QPixmap, QTransform

    panel = _stacked_plate_image(view, qtbot)
    original = panel.view._item.pixmap()
    transform = panel.view.transform()
    try:
        if state == 'null_pixmap':
            panel.view._item.setPixmap(QPixmap())
        else:
            panel.view.setTransform(QTransform.fromScale(0, 0))
        assert view._readout_target(panel.view, QPointF(1, 1)) is None
        assert view._field_shown_in(panel.view) is panel.field
        assert view._readout_loading is None
    finally:
        panel.view._item.setPixmap(original)
        panel.view.setTransform(transform)
        view._clear_pixel_readout()
        _settle_plate_auxiliary_jobs(view, qtbot)


@pytest.mark.parametrize('missing_source', [False, True])
def test_stacked_raw_readout_reports_exact_plate_values_or_source_failure(view, qtbot, missing_source):
    from pathlib import Path

    panel = _stacked_plate_image(view, qtbot)
    displayed = panel.view._item.pixmap().toImage().copy()
    originals = {Path(path): Path(path).read_bytes() for _channel, path in panel.field.files}
    moved = {}
    try:
        if missing_source:
            for path in originals:
                backup = path.with_suffix('.held-tif')
                path.rename(backup)
                moved[path] = backup
            browser._OVERVIEW_PLANE_CACHE.clear()
            browser._OVERVIEW_INFO_CACHE.clear()
        _move_plate_image_cursor(panel.view)
        if missing_source:
            qtbot.waitUntil(lambda: 'Pixel values unavailable:' in view._pixel_readout.text(), timeout=10000)
        else:
            qtbot.waitUntil(lambda: 'Channel 1: 2101' in view._pixel_readout.text()
                            and 'Channel 2: 2102' in view._pixel_readout.text(), timeout=10000)
            assert view._pixel_readout.text().startswith('x 1, y 1')
        text = view._pixel_readout.text()
        assert 'Plate plateC' in text and 'Well A01' in text and 'Image 1' in text
        assert panel.view._item.pixmap().toImage() == displayed
    finally:
        view._clear_pixel_readout()
        _settle_plate_auxiliary_jobs(view, qtbot)
        for path, backup in moved.items():
            backup.rename(path)
        assert all(path.read_bytes() == content for path, content in originals.items())


def test_image_settings_reuse_ignores_stale_histograms_and_default_resets(view, qtbot):
    from copy import deepcopy

    first = next(field for field in view._fields if field.plate == 'plateA')
    second = next(field for field in view._fields if field.plate == 'plateB')
    original_fields = tuple(field.identity for field in view._fields)
    original_overrides = deepcopy(view._image_overrides)
    save_active = view._override_save_timer.isActive()
    view._set_image_override(browser._overview_image_key(first), None)
    view._copy_image_override(first, 'all')
    assert view._image_overrides == original_overrides
    assert view._override_save_timer.isActive() == save_active
    view._open_image_settings(first)
    panel = view._image_settings
    try:
        _settle_plate_auxiliary_jobs(view, qtbot)
        assert panel.key == browser._overview_image_key(first)
        view._open_image_settings(second)
        assert view._image_settings is panel and panel.key == browser._overview_image_key(second)
        _settle_plate_auxiliary_jobs(view, qtbot)
        counts = deepcopy(panel.histogram.counts)
        view._image_histograms_read(browser._overview_image_key(first), {1: np.array([999])})
        np.testing.assert_array_equal(panel.histogram.counts, counts)
        assert tuple(field.identity for field in view._fields) == original_fields
        assert view._image_overrides == original_overrides
    finally:
        panel.close()
        _settle_plate_auxiliary_jobs(view, qtbot)


def test_invalid_channel_settings_preserve_open_image_and_histogram_panel(view, qtbot):
    panel = _stacked_plate_image(view, qtbot)
    before = panel.view._item.pixmap().toImage().copy()
    original = view.channels.text()
    settings_panel = view._image_settings
    view.channels.setText('not a channel')
    try:
        view._open_image_settings(panel.field)
        view._load_image_view(panel)
        assert view._image_settings is settings_panel
        assert 'nonnegative channel indices' in view.status.text()
        assert panel.view._item.pixmap().toImage() == before
    finally:
        view.channels.setText(original)
        _settle_plate_auxiliary_jobs(view, qtbot)


class _HeldPlatePixels:
    """Delay one real detached-image read until its actual panel has closed."""

    def __init__(self, read, identity):
        from threading import Event

        self.read, self.identity = read, identity
        self.entered, self.release = Event(), Event()

    def __call__(self, field, *args, **kwargs):
        if field.identity == self.identity and not self.entered.is_set():
            self.entered.set()
            assert self.release.wait(10)
        return self.read(field, *args, **kwargs)


def test_late_stacked_image_read_cannot_replace_another_plate_after_close(view, qtbot, monkeypatch):
    first = _stacked_plate_image(view, qtbot, 'plateA')
    other = _stacked_plate_image(view, qtbot, 'plateB')
    displayed = other.view._item.pixmap().toImage().copy()
    read = _HeldPlatePixels(browser._overview_pixels, first.field.identity)
    monkeypatch.setattr(browser, '_overview_pixels', read)
    try:
        view._load_image_view(first)
        qtbot.waitUntil(read.entered.is_set, timeout=5000)
        view._readout_pending = first.view, QPointF(1, 1)
        view._close_image_view(first)
        view._close_image_view(first)
        assert first not in view._image_views and other in view._image_views
        assert view._readout_pending is None and view._pixel_readout.text() == ''
        read.release.set()
        _settle_plate_auxiliary_jobs(view, qtbot)
        assert view._image_views == [other]
        assert other.field.plate == 'plateB' and other.view._item.pixmap().toImage() == displayed
        view._image_panel.hide()
        view._close_image_view(other)
        assert view._image_views == [] and view._image_column.isHidden()
    finally:
        read.release.set()
        _settle_plate_auxiliary_jobs(view, qtbot)


def test_original_chain_refusal_falls_back_then_source_failure_keeps_last_pixels(view, qtbot):
    from pathlib import Path

    panel = _stacked_plate_image(view, qtbot)
    displayed = panel.view._item.pixmap().toImage().copy()
    originals = {Path(path): Path(path).read_bytes() for _channel, path in panel.field.files}
    moved = {}
    steps = {'psf_operation': 'convolve', 'psf_error': 'owned missing calibration'}
    view._original_steps = steps.copy
    try:
        view.preview_resolution.setCurrentIndex(view.preview_resolution.findData(None))
        qtbot.waitUntil(lambda: not view._quality_timer.isActive(), timeout=5000)
        _settle_plate_auxiliary_jobs(view, qtbot)
        assert view._quality_original()
        assert 'shown without PSF convolution' in view.status.text()
        assert 'owned missing calibration' in view.status.text()
        assert panel.view._item.pixmap().toImage() == displayed
        for path in originals:
            backup = path.with_suffix('.held-tif')
            path.rename(backup)
            moved[path] = backup
        browser._OVERVIEW_PLANE_CACHE.clear()
        browser._OVERVIEW_INFO_CACHE.clear()
        view._load_image_view(panel)
        _settle_plate_auxiliary_jobs(view, qtbot)
        assert 'Could not read' in view.status.text()
        assert panel.view._item.pixmap().toImage() == displayed
    finally:
        _settle_plate_auxiliary_jobs(view, qtbot)
        for path, backup in moved.items():
            backup.rename(path)
        view._original_steps = None
        assert all(path.read_bytes() == content for path, content in originals.items())


class _HeldPlateReadout:
    """Keep one genuine raw read in flight across cursor/source retirement."""

    def __init__(self, read):
        from threading import Event

        self.read = read
        self.entered, self.release = Event(), Event()
        self.fields = []

    def __call__(self, field, axes):
        self.fields.append(field.identity)
        self.entered.set()
        assert self.release.wait(10)
        return self.read(field, axes)


@pytest.mark.parametrize('retirement', ['leave', 'cancel_source'])
def test_pending_pixel_read_is_deduplicated_and_cannot_restore_retired_readout(
        view, qtbot, monkeypatch, retirement):
    panel = _stacked_plate_image(view, qtbot)
    read = _HeldPlateReadout(browser._overview_readout_planes)
    monkeypatch.setattr(browser, '_overview_readout_planes', read)
    before = panel.view._item.pixmap().toImage().copy()
    try:
        _move_plate_image_cursor(panel.view)
        qtbot.waitUntil(read.entered.is_set, timeout=5000)
        axes = view._axis_options()
        key = panel.field.identity, tuple(sorted(axes.items()))
        assert view._readout_loading == key
        view._load_readout_planes(panel.field, key, axes)
        assert view._readout_jobs.active_jobs() == 1
        view._clear_pixel_readout()
        if retirement == 'cancel_source':
            view._cancel()
        read.release.set()
        _settle_plate_auxiliary_jobs(view, qtbot)
        assert read.fields == [panel.field.identity]
        assert view._readout_pending is None and view._readout_loading is None
        assert view._pixel_readout.text() == ''
        if retirement == 'cancel_source':
            assert view._readout_planes is None and view._cancelled.is_set()
        else:
            assert view._readout_planes[0] == key
        assert panel.view._item.pixmap().toImage() == before
    finally:
        read.release.set()
        view._clear_pixel_readout()
        _settle_plate_auxiliary_jobs(view, qtbot)


@pytest.mark.parametrize('edit', ['invalid_channels', 'outside_image'])
def test_raw_pixel_readout_handles_incomplete_choices_and_blank_image_margins(view, qtbot, edit):
    panel = _stacked_plate_image(view, qtbot)
    original_channels = view.channels.text()
    try:
        _move_plate_image_cursor(panel.view)
        qtbot.waitUntil(lambda: 'Channel 1: 2101' in view._pixel_readout.text(), timeout=10000)
        if edit == 'invalid_channels':
            view.channels.setText('unfinished channel')
            view._update_pixel_readout()
            assert 'Channel 1: 2101' in view._pixel_readout.text()
            assert 'Channel 2: 2102' in view._pixel_readout.text()
            assert 'Plate plateC' in view._pixel_readout.text()
        else:
            image_bounds = panel.view.viewportTransform().mapRect(panel.view._item.sceneBoundingRect())
            assert image_bounds.top() > 1
            point = QPointF(image_bounds.center().x(), image_bounds.top() / 2)
            assert panel.view.viewport().rect().contains(point.toPoint())
            assert not image_bounds.contains(point)
            event = QMouseEvent(QEvent.MouseMove, point, panel.view.viewport().mapToGlobal(point),
                                Qt.NoButton, Qt.NoButton, Qt.NoModifier)
            QApplication.sendEvent(panel.view.viewport(), event)
            qtbot.waitUntil(lambda: view._pixel_readout.text() == '', timeout=5000)
        assert view._field_shown_in(panel.view) is panel.field
    finally:
        view.channels.setText(original_channels)
        view._clear_pixel_readout()
        _settle_plate_auxiliary_jobs(view, qtbot)


class _RecordedPlateHistograms:
    """Observe real worker histogram inputs while still computing every count."""

    def __init__(self, read):
        self.read, self.requests = read, []

    def __call__(self, field, channels, limits, options):
        from copy import deepcopy

        self.requests.append((field.identity, tuple(channels), limits, deepcopy(options)))
        return self.read(field, channels, limits, options)


@pytest.mark.parametrize('layer', ['global_normalization', 'global_enhancement'])
def test_image_layer_action_refreshes_only_its_matching_histogram_with_real_counts(
        view, qtbot, monkeypatch, layer):
    field = next(field for field in view._fields if field.plate == 'plateB')
    view._open_image_settings(field)
    panel = view._image_settings
    _settle_plate_auxiliary_jobs(view, qtbot)
    read = _RecordedPlateHistograms(browser._overview_display_histograms)
    monkeypatch.setattr(browser, '_overview_display_histograms', read)
    original_channels = view.channels.text()
    try:
        view._set_global_layer(field, layer, False)
        _settle_plate_auxiliary_jobs(view, qtbot)
        assert view._image_settings is panel and panel.key == browser._overview_image_key(field)
        assert read.requests and all(row[0] == field.identity for row in read.requests)
        assert read.requests[-1][3]['image_overrides'] == {panel.key: {layer: False}}
        assert all(int(counts.sum()) == 24 * 32 for counts in panel._histograms.values())
        assert set(panel._histograms) == set(panel.channels)
        view.channels.setText('incomplete')
        view._request_image_histograms(panel)
        _settle_plate_auxiliary_jobs(view, qtbot)
        assert read.requests[-1][2] is None
        assert all(int(counts.sum()) == 24 * 32 for counts in panel._histograms.values())
        view.channels.setText(original_channels)
        view._set_global_layer(field, layer, True)
        _settle_plate_auxiliary_jobs(view, qtbot)
        assert layer not in (view._image_overrides.get(panel.key) or {})
        assert 'image_overrides' not in read.requests[-1][3]
    finally:
        view.channels.setText(original_channels)
        panel.close()
        _settle_plate_auxiliary_jobs(view, qtbot)


def test_plain_stacked_preview_read_error_keeps_pixels_and_original_input_bytes(view, qtbot):
    from pathlib import Path

    panel = _stacked_plate_image(view, qtbot)
    assert view._original_chain(view._display_kwargs(view=True)) is None
    displayed = panel.view._item.pixmap().toImage().copy()
    originals = {Path(path): Path(path).read_bytes() for _channel, path in panel.field.files}
    moved = {}
    try:
        for path in originals:
            backup = path.with_suffix('.held-tif')
            path.rename(backup)
            moved[path] = backup
        browser._OVERVIEW_PLANE_CACHE.clear()
        browser._OVERVIEW_INFO_CACHE.clear()
        view._load_image_view(panel)
        _settle_plate_auxiliary_jobs(view, qtbot)
        assert 'Could not read' in view.status.text() and panel.field.caption in view.status.text()
        assert panel.view._item.pixmap().toImage() == displayed
    finally:
        _settle_plate_auxiliary_jobs(view, qtbot)
        for path, backup in moved.items():
            backup.rename(path)
        assert all(path.read_bytes() == content for path, content in originals.items())


def test_added_source_with_an_existing_basename_keeps_every_physical_plate_distinct(view, qtbot, tmp_path):
    before = {field.identity: field for field in view._all_fields}
    duplicate = _plate(tmp_path / 'another_project' / 'plateA', ('A01',), 3000)
    originals = {path: path.read_bytes() for path in duplicate.glob('*.tif')}
    view._add_sources(str(duplicate))
    qtbot.waitUntil(lambda: len(view._all_fields) == len(before) + 1
                    and view._normalization_ready and not view._index_jobs.is_busy(), timeout=15000)
    _settle_plate_auxiliary_jobs(view, qtbot)
    fields = {field.identity: field for field in view._all_fields}
    assert all(fields[identity] == field for identity, field in before.items())
    added = [field for identity, field in fields.items() if identity not in before]
    assert len(added) == 1 and added[0].plate == 'plateA (2)'
    assert added[0].source == str(duplicate) and added[0].well == 'A01'
    assert tuple(view.plates.itemData(index) for index in range(view.plates.count())) == (
        'plateA', 'plateA (2)', 'plateB', 'plateC')
    assert all(path.read_bytes() == content for path, content in originals.items())


def test_real_plate_header_action_collapses_only_its_owner(view, qtbot):
    view._set_plates_view('side')
    table = view._plate_tables['plateB']
    header = table.verticalHeader()
    from PySide6.QtCore import QPoint

    position = QPoint(max(1, header.width() // 2), header.sectionViewportPosition(0) + header.sectionSize(0) // 2)
    menu = view._plate_header_menu(table, 'row', position)
    try:
        assert view.plates.currentData() == 'plateB' and view.wells is table
        collapse = next(action for action in menu.actions() if action.text() == 'Collapse row A')
        collapse.trigger()
        assert view._collapsed['plateB'][0] == {0}
        assert view._collapsed.get('plateA', (set(), set())) == (set(), set())
        assert view._collapsed.get('plateC', (set(), set())) == (set(), set())
        assert table.rowHeight(0) == browser._OVERVIEW_COLLAPSED_STRIP
    finally:
        menu.close()
        _settle_plate_auxiliary_jobs(view, qtbot)


def test_stacked_image_limit_evicts_oldest_panels_without_conflating_wells(view, qtbot):
    fields = tuple(view._fields)
    assert len(fields) == 9 and browser._OVERVIEW_MAX_IMAGE_VIEWS == 6
    view._open_image_views(fields[:6])
    _settle_plate_auxiliary_jobs(view, qtbot)
    oldest_viewports = [panel.view.viewport() for panel in view._image_views[:3]]
    view._open_image_views(fields[6:])
    _settle_plate_auxiliary_jobs(view, qtbot)
    assert [panel.field.identity for panel in view._image_views] == [field.identity for field in fields[-6:]]
    assert all(viewport not in view._readout_viewports for viewport in oldest_viewports)
    assert all(panel.view._item is not None and not panel.view._item.pixmap().isNull()
               for panel in view._image_views)
    assert len({panel.field.identity for panel in view._image_views}) == 6


def test_original_steps_reload_real_main_and_stacked_images_without_changing_sources(view, qtbot):
    view._well(_item(view, view.plates.currentData(), 'A01'))
    qtbot.waitUntil(lambda: view.fields.count() > 0, timeout=10000)
    view.fields.setCurrentRow(0)
    view._field(view.fields.item(0))
    qtbot.waitUntil(lambda: view._individual and view.view._item is not None
                    and not view._image_pending, timeout=10000)
    panel = _stacked_plate_image(view, qtbot)
    view.preview_resolution.setCurrentIndex(view.preview_resolution.findData(None))
    qtbot.waitUntil(lambda: not view._quality_timer.isActive(), timeout=5000)
    _settle_plate_auxiliary_jobs(view, qtbot)
    main = view.view._item.pixmap().toImage().copy()
    stacked = panel.view._item.pixmap().toImage().copy()
    identity = view._main_view_field.identity
    view._original_steps_changed()
    _settle_plate_auxiliary_jobs(view, qtbot)
    assert view._quality_original() and view._individual
    assert view._main_view_field.identity == identity and panel.field.plate == 'plateC'
    assert view.view._item.pixmap().toImage() == main
    assert panel.view._item.pixmap().toImage() == stacked


def test_plate_area_menu_identifies_the_real_child_owner_and_hides_only_that_plate(view, qtbot):
    view._set_plates_view('side')
    table = view._plate_tables['plateB']
    header = _header(view, 'plateB')
    view._plate_scroll.ensureWidgetVisible(header)
    QApplication.processEvents()
    body = view._plate_panel.body
    point = header.mapTo(body, header.rect().center())
    assert body.rect().contains(point) and body.childAt(point) is not None
    before = tuple(field.identity for field in view._all_fields)
    menu = view._plate_area_menu(point)
    try:
        hide = next(action for action in menu.actions() if action.text() == 'Hide plate plateB')
        assert hide.isEnabled()
        hide.trigger()
        assert set(view._plate_tables) == {'plateA', 'plateC'}
        assert table not in view._plate_tables.values()
        assert tuple(field.identity for field in view._all_fields) == before
    finally:
        menu.close()
        _settle_plate_auxiliary_jobs(view, qtbot)


def test_remove_plate_toolbar_keeps_the_final_plate_and_every_original_file(view, qtbot):
    from pathlib import Path

    originals = {Path(path): Path(path).read_bytes()
                 for field in view._all_fields for _channel, path in field.files}
    remove = view._plate_buttons[3]
    view.plates.setCurrentIndex(view.plates.findData('plateB'))
    remove.click()
    _settle_plate_auxiliary_jobs(view, qtbot)
    assert {field.plate for field in view._all_fields} == {'plateA', 'plateC'}
    view.plates.setCurrentIndex(view.plates.findData('plateC'))
    remove.click()
    _settle_plate_auxiliary_jobs(view, qtbot)
    last = tuple(view._all_fields)
    assert {field.plate for field in last} == {'plateA'}
    assert view.plates.count() == 1 and not remove.isEnabled()
    remove.click()
    view._remove_shown_plate()
    assert tuple(view._all_fields) == last
    assert all(path.read_bytes() == content for path, content in originals.items())


def test_smaller_load_cap_drops_a_real_field_selection_but_keeps_the_indexed_source(view, qtbot):
    view._set_plates_view('side')
    assert view._open_wells([('plateA', 'A02')]) == (('plateA', 'A02'),)
    _settle_plate_auxiliary_jobs(view, qtbot)
    fields = view.fields
    fields.doItemsLayout()
    item = fields.item(0)
    field = item.data(Qt.UserRole)
    item.setSelected(True)
    assert field.well == 'A02' and field.identity in view._selected_fields
    assert field.identity in view._selected_field_options
    originals = tuple(view._all_fields)
    view._set_load_limits('1', None)
    _settle_plate_auxiliary_jobs(view, qtbot)
    assert view._all_fields == originals and len(view._all_fields) == 9
    assert len(view._fields) == 3 and all(field.well == 'A01' for field in view._fields)
    assert field.identity not in view._selected_fields
    assert field.identity not in view._selected_field_options
    assert all(browser.Path(path).exists() for _channel, path in field.files)


@pytest.mark.parametrize('change', ['color', 'enhancement'])
def test_main_image_display_changes_reload_the_selected_field_without_touching_raw_data(view, qtbot, change):
    from pathlib import Path
    from PySide6.QtTest import QSignalSpy
    from spacr.qt.detect_chain import NO_CHAIN

    view._well(_item(view, view.plates.currentData(), 'A01'))
    qtbot.waitUntil(lambda: view.fields.count() > 0, timeout=10000)
    view.fields.setCurrentRow(0)
    view._field(view.fields.item(0))
    qtbot.waitUntil(lambda: view._individual and view.view._item is not None
                    and not view._image_pending, timeout=10000)
    _settle_plate_auxiliary_jobs(view, qtbot)
    field = view._main_view_field
    originals = {Path(path): Path(path).read_bytes() for _channel, path in field.files}
    completed = QSignalSpy(view._image_jobs.job_finished)
    if change == 'color':
        view._set_color_mode('greys')
        assert view._color_mode == 'greys'
    else:
        chain = NO_CHAIN._replace(gamma=2.)
        view._set_enhancement(chain)
        assert view._enhancement == chain
    _settle_plate_auxiliary_jobs(view, qtbot)
    assert completed.count() >= 1 and not view._image_pending
    assert view._main_view_field is field and view._individual
    assert not view.view._item.pixmap().isNull()
    assert all(path.read_bytes() == content for path, content in originals.items())


@pytest.mark.parametrize('owner', ['well', 'field'])
def test_blank_context_menu_retains_navigation_without_inventing_an_image_selection(view, qtbot, owner):
    from PySide6.QtCore import QPoint

    view._set_plates_view('side')
    if owner == 'well':
        view._focus_plate_table(view._plate_tables['plateA'])
        view._zoom_plate()
        table = view.wells
        assert table.item(1, 2) is None
        point = QPoint(table.columnViewportPosition(2) + table.columnWidth(2) // 2,
                       table.rowViewportPosition(1) + table.rowHeight(1) // 2)
        assert table.viewport().rect().contains(point) and table.itemAt(point) is None
        menu = view._well_menu(point)
        target_caption = 'Annotate selected wells…'
    else:
        assert view._open_wells([('plateA', 'A01')]) == (('plateA', 'A01'),)
        _settle_plate_auxiliary_jobs(view, qtbot)
        fields = view.fields
        fields.doItemsLayout()
        point = fields.viewport().rect().bottomRight()
        assert view._tile_at(fields, point) is None
        view._field_menu(fields, point)
        menu = view._image_menu_shown
        target_caption = 'Open selected images'
    try:
        target = next(action for action in menu.actions() if action.text() == target_caption)
        assert not target.isEnabled()
        assert any(action.text() == 'Back to plate' for action in menu.actions())
        assert not view._selected_fields and not view._selected_wells
        assert not any(action.text() == 'Image settings…' for action in menu.actions())
    finally:
        menu.close()
        _settle_plate_auxiliary_jobs(view, qtbot)


class _OrderedPlateReadouts:
    """Let the previous field finish while a different real field is still reading."""

    def __init__(self, read, fields):
        from threading import Event

        self.read = read
        self.entered = {field.identity: Event() for field in fields}
        self.release = {field.identity: Event() for field in fields}

    def __call__(self, field, axes):
        self.entered[field.identity].set()
        assert self.release[field.identity].wait(10)
        return self.read(field, axes)


def test_previous_field_readout_completion_does_not_erase_the_current_loading_key(view, qtbot, monkeypatch):
    first = _stacked_plate_image(view, qtbot, 'plateA')
    second = _stacked_plate_image(view, qtbot, 'plateB')
    read = _OrderedPlateReadouts(browser._overview_readout_planes, [first.field, second.field])
    monkeypatch.setattr(browser, '_overview_readout_planes', read)
    try:
        _move_plate_image_cursor(first.view)
        qtbot.waitUntil(read.entered[first.field.identity].is_set, timeout=5000)
        _move_plate_image_cursor(second.view)
        qtbot.waitUntil(read.entered[second.field.identity].is_set, timeout=5000)
        axes = tuple(sorted(view._axis_options().items()))
        current_key = second.field.identity, axes
        assert view._readout_loading == current_key
        read.release[first.field.identity].set()
        qtbot.waitUntil(lambda: view._readout_planes is not None
                        and view._readout_planes[0] == (first.field.identity, axes), timeout=5000)
        assert view._readout_loading == current_key
        assert 'Plate plateB' in view._pixel_readout.text()
        assert 'reading pixel values' in view._pixel_readout.text()
        assert 'Channel 1: 101' not in view._pixel_readout.text()
        read.release[second.field.identity].set()
        qtbot.waitUntil(lambda: 'Channel 1: 1101' in view._pixel_readout.text()
                        and 'Channel 2: 1102' in view._pixel_readout.text(), timeout=10000)
        assert view._readout_planes[0] == current_key and view._readout_loading is None
    finally:
        for release in read.release.values():
            release.set()
        view._clear_pixel_readout()
        _settle_plate_auxiliary_jobs(view, qtbot)


def test_persisted_invalid_image_key_is_ignored_while_valid_image_and_plate_state_survive(view, qtbot):
    from copy import deepcopy

    field = view._fields[0]
    key = browser._overview_image_key(field)
    view._persist_view_state()
    _settle_plate_auxiliary_jobs(view, qtbot)
    path = browser._overview_view_state_path(view._metadata_source)
    state = json.loads(path.read_text())
    plates = deepcopy(state['plates'])
    options = deepcopy(state['options'])
    state['images'] = {browser._overview_key_text(key): {'brightness': .2},
                       'not a JSON image key': {'brightness': .9}}
    path.write_text(json.dumps(state))
    view._restore_view_state()
    _settle_plate_auxiliary_jobs(view, qtbot)
    assert view._image_overrides == {key: {'brightness': .2}}
    assert view._view_state['plates'] == plates and view._view_state['options'] == options
    assert tuple(sorted({field.plate for field in view._fields})) == PLATES


def test_not_loaded_well_context_request_preserves_existing_selection_and_navigation(view, qtbot):
    view._set_load_limits('1', None)
    view._set_plates_view('side')
    _settle_plate_auxiliary_jobs(view, qtbot)
    table = view._plate_tables['plateB']
    loaded = _item(view, 'plateB', 'A01')
    blocked = table.item(0, 1)
    assert blocked is not None and blocked.text().splitlines()[0] == 'A02'
    assert ('plateB', 'A02') not in view._well_positions
    assert blocked.data(browser._OVERVIEW_NOT_LOADED_ROLE)
    assert not (blocked.flags() & Qt.ItemIsSelectable)
    loaded.setSelected(True)
    selected = set(view._selected_wells)
    assert selected == {('plateB', 'A01')}
    before = tuple(field.identity for field in view._all_fields)
    view._plate_zoom_actions[0].trigger()
    point = table.visualItemRect(blocked).center()
    assert table.viewport().rect().contains(point) and table.itemAt(point) is blocked
    table.customContextMenuRequested.emit(point)
    menu = view._well_context_menu
    try:
        assert view.wells is table and view.plates.currentData() == 'plateB'
        assert view._selected_wells == selected and not blocked.isSelected()
        assert loaded.isSelected() and not view._selected_fields
        assert ('plateB', 'A02') not in view._well_fields
        assert not view._well_panels and not view._image_views
        assert all('A02' not in action.text() for action in menu.actions())
        annotate = next(action for action in menu.actions() if action.text() == 'Annotate selected wells…')
        assert annotate.isEnabled()
        assert any(action.text() == 'Back to plate' for action in menu.actions())
        assert tuple(field.identity for field in view._all_fields) == before and len(before) == 9
        assert set(view._plate_tables) == set(PLATES)
    finally:
        menu.close()
        _settle_plate_auxiliary_jobs(view, qtbot)


def test_reopening_overlapping_wells_retains_the_shared_plate_panels_and_pixels(view, qtbot):
    view._set_plates_view('side')
    assert view._open_wells([('plateA', 'A01'), ('plateB', 'A01')]) == (('plateA', 'A01'), ('plateB', 'A01'))
    _settle_plate_auxiliary_jobs(view, qtbot)
    wells = {panel.field_list._overview_plate: panel for panel in view._well_panels}
    images = {panel.field.plate: panel for panel in view._image_views}
    assert set(wells) == set(images) == {'plateA', 'plateB'}
    shared_well, shared_image = wells['plateB'], images['plateB']
    displayed = shared_image.view._item.pixmap().toImage().copy()
    original = tuple(field.identity for field in view._all_fields)
    assert view._open_wells([('plateB', 'A01'), ('plateC', 'A01')]) == (('plateB', 'A01'), ('plateC', 'A01'))
    _settle_plate_auxiliary_jobs(view, qtbot)
    assert images['plateA'] not in view._image_views
    assert wells['plateA'] is view._well_panel
    assert wells['plateA'].field_list._overview_plate == 'plateC'
    assert shared_well in view._well_panels and shared_image in view._image_views
    assert {panel.field_list._overview_plate for panel in view._well_panels} == {'plateB', 'plateC'}
    assert {panel.field.plate for panel in view._image_views} == {'plateB', 'plateC'}
    assert shared_image.view._item.pixmap().toImage() == displayed
    assert tuple(field.identity for field in view._all_fields) == original


def test_removing_a_child_plate_from_a_parent_source_keeps_root_and_excludes_only_that_child(qtbot, plates):
    root = plates[0].parent
    original = {path: path.read_bytes() for plate in plates for path in plate.glob('*.tif')}
    view = _open(qtbot, str(root))
    assert {field.plate for field in view._all_fields} == set(PLATES)
    kept = {field.identity for field in view._all_fields if field.plate != 'plateB'}
    view._remove_plate('plateB')
    _settle_plate_auxiliary_jobs(view, qtbot)
    assert view.source.text() == str(root)
    assert str(plates[1].resolve()) in view._overview_options['excluded_plates']
    assert {field.identity for field in view._all_fields} == kept
    view._scan()
    qtbot.waitUntil(lambda: len(view._all_fields) == 5 and view._normalization_ready
                    and not view._index_jobs.is_busy(), timeout=15000)
    _settle_plate_auxiliary_jobs(view, qtbot)
    assert view.source.text() == str(root) and {field.identity for field in view._all_fields} == kept
    assert {field.source for field in view._all_fields} == {str(plates[0]), str(plates[2])}
    assert all(path.read_bytes() == content for path, content in original.items())


def test_single_remaining_plate_does_not_step_or_change_selection_on_late_navigation(view, qtbot):
    from PySide6.QtTest import QSignalSpy

    view._remove_plate('plateB')
    view._remove_plate('plateC')
    _settle_plate_auxiliary_jobs(view, qtbot)
    assert view.plates.count() == 1 and view.plates.currentData() == 'plateA'
    _item(view, 'plateA', 'A01').setSelected(True)
    selected = set(view._selected_wells)
    fields = tuple(view._all_fields)
    changed = QSignalSpy(view.plates.currentIndexChanged)
    for button, delta in zip(view._plate_buttons[:2], (-1, 1)):
        assert not button.isEnabled()
        button.click()
        view._step_plate(delta)
    assert changed.count() == 0 and view.plates.currentData() == 'plateA'
    assert view._selected_wells == selected and selected == {('plateA', 'A01')}
    assert tuple(view._all_fields) == fields


def test_real_matched_invalid_well_is_refused_before_browser_fields_are_accepted(view, qtbot, plates):
    from PySide6.QtTest import QSignalSpy

    invalid = plates[0] / 'p1-A00-f9-c1.tif'
    pixels = np.full((24, 32), 4096, dtype=np.uint16)
    tifffile.imwrite(invalid, pixels)
    original = {path: path.read_bytes() for plate in plates for path in plate.glob('*.tif')}
    failures = QSignalSpy(view._index_jobs.job_failed)
    view._scan()
    qtbot.waitUntil(lambda: failures.count() == 1 and not view._index_jobs.is_busy(), timeout=15000)
    _settle_plate_auxiliary_jobs(view, qtbot)
    message = 'The filename regex must include a valid wellID for every image.'
    assert failures.at(0)[0].endswith(message)
    assert view.status.text().endswith(message)
    assert not view._all_fields and not view._fields
    assert view.fields.count() == 0 and view.view._item is None
    assert all(path.read_bytes() == content for path, content in original.items())
    np.testing.assert_array_equal(tifffile.imread(invalid), pixels)


def test_returning_from_one_field_to_its_montage_refuses_stale_image_readout_and_settings(view, qtbot):
    from pathlib import Path
    from PySide6.QtCore import QPoint
    from PySide6.QtTest import QSignalSpy

    item = _item(view, 'plateA', 'A01')
    view._well(item)
    qtbot.waitUntil(lambda: view.fields.count() > 0, timeout=10000)
    view.fields.setCurrentRow(0)
    qtbot.waitUntil(lambda: view._individual and view.view._item is not None
                    and not view.view._item.pixmap().isNull(), timeout=10000)
    _settle_plate_auxiliary_jobs(view, qtbot)
    field = view._main_view_field
    originals = {Path(path): Path(path).read_bytes() for _channel, path in field.files}
    _move_plate_image_cursor(view.view)
    qtbot.waitUntil(lambda: view._pixel_readout.text().startswith('x '), timeout=10000)
    view._well(_item(view, 'plateA', 'A01'))
    qtbot.waitUntil(lambda: not view._individual and view._displayed_montage is not None
                    and view.view._item is not None
                    and not view.view._item.pixmap().isNull(), timeout=10000)
    _settle_plate_auxiliary_jobs(view, qtbot)
    assert view._main_view_field is field
    displayed = view.view._item.pixmap().toImage().copy()
    navigation = (view._scope, view.plates.currentData(), view.fields.count(),
                  tuple(candidate.identity for candidate in view._fields))
    readouts = QSignalSpy(view._readout_jobs.job_finished)
    cursor_updates = QSignalSpy(view._readout_timer.timeout)
    _move_plate_image_cursor(view.view)
    qtbot.waitUntil(lambda: cursor_updates.count() >= 1)
    assert view._pixel_readout.text() == ''
    view.view.customContextMenuRequested.emit(QPoint(1, 1))
    assert view._image_menu_shown is None and view._image_settings is None
    assert view._readout_jobs.active_jobs() == 0 and readouts.count() == 0
    assert view.view._item.pixmap().toImage() == displayed
    assert (view._scope, view.plates.currentData(), view.fields.count(),
            tuple(candidate.identity for candidate in view._fields)) == navigation
    assert all(path.read_bytes() == data for path, data in originals.items())


def test_hovering_a_real_empty_plate_cell_clears_the_previous_well_readout(view, qtbot):
    from pathlib import Path
    from PySide6.QtCore import QPoint

    view._set_plates_view('side')
    _settle_plate_auxiliary_jobs(view, qtbot)
    _item(view, 'plateB', 'A01').setSelected(True)
    selected = set(view._selected_wells)
    current = view.plates.currentData()
    identities = tuple(field.identity for field in view._fields)
    originals = {Path(path): Path(path).read_bytes()
                 for field in view._fields for _channel, path in field.files}
    _hover(view, 'plateA', 'A01')
    qtbot.waitUntil(lambda: 'Plate plateA' in view._pixel_readout.text()
                    and 'Well A01' in view._pixel_readout.text())
    table = view._plate_tables['plateA']
    assert table.item(1, 2) is None
    point = QPoint(table.columnViewportPosition(2) + table.columnWidth(2) // 2,
                   table.rowViewportPosition(1) + table.rowHeight(1) // 2)
    assert table.viewport().rect().contains(point) and table.itemAt(point) is None
    local = QPointF(point)
    event = QMouseEvent(QEvent.MouseMove, local, table.viewport().mapToGlobal(point),
                        Qt.NoButton, Qt.NoButton, Qt.NoModifier)
    QApplication.sendEvent(table.viewport(), event)
    qtbot.waitUntil(lambda: view._pixel_readout.text() == '')
    assert view._selected_wells == selected
    assert view.plates.currentData() == current
    assert tuple(field.identity for field in view._fields) == identities
    assert _item(view, 'plateB', 'A01').isSelected()
    assert all(path.read_bytes() == data for path, data in originals.items())


def test_cancelling_before_queued_plate_hover_timeout_keeps_the_prior_readout(view, qtbot):
    from pathlib import Path
    from PySide6.QtTest import QSignalSpy

    view._set_plates_view('side')
    _settle_plate_auxiliary_jobs(view, qtbot)
    _item(view, 'plateC', 'A01').setSelected(True)
    selection = set(view._selected_wells)
    identities = tuple(field.identity for field in view._fields)
    originals = {Path(path): Path(path).read_bytes()
                 for field in view._fields for _channel, path in field.files}
    _hover(view, 'plateA', 'A01')
    qtbot.waitUntil(lambda: 'Plate plateA' in view._pixel_readout.text()
                    and 'Well A01' in view._pixel_readout.text())
    before = view._pixel_readout.text()
    timeouts = QSignalSpy(view._readout_timer.timeout)
    readouts = QSignalSpy(view._readout_jobs.job_finished)
    _hover(view, 'plateB', 'A01')
    assert view._readout_pending[0] is view._plate_tables['plateB']
    assert view._readout_timer.isActive() and timeouts.count() == 0
    cancel = next(button for button in view._module_action_buttons
                  if button.text() == 'Cancel loading or export')
    assert cancel.isEnabled()
    cancel.click()
    assert view._cancelled.is_set()
    qtbot.waitUntil(lambda: timeouts.count() >= 1)
    assert view._pixel_readout.text() == before
    assert view._readout_jobs.active_jobs() == 0 and readouts.count() == 0
    assert view._selected_wells == selection
    assert _item(view, 'plateC', 'A01').isSelected()
    assert tuple(field.identity for field in view._fields) == identities
    assert all(path.read_bytes() == data for path, data in originals.items())


def test_lazy_overview_plate_configuration_never_persists_without_a_source(qtbot, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    view = browser._FieldOverview()
    qtbot.addWidget(view, before_close_func=lambda widget: _retire(widget, qtbot))
    view.resize(1500, 1000)
    view.show()
    qtbot.waitExposed(view)
    assert view.source.text() == '' and view._metadata_source == ''
    assert not view._fields and view.plates.count() == 0
    assert view._set_plates_view('side')
    assert view.plates_view.currentData() == 'side'
    assert view._plates_layout.direction() == QBoxLayout.LeftToRight
    assert view._set_plates_view('stacked')
    assert view.plates_view.currentData() == 'stacked'
    assert view._plates_layout.direction() == QBoxLayout.TopToBottom
    assert view._hide_every_plate()
    assert not view._hide_every_plate()
    assert view._plates_subset == ()
    assert view.plates.count() == 0 and view.fields.count() == 0
    assert not view._plate_tables and view.wells.rowCount() == 0
    assert view._view_state_pending is None and not view._view_state_writing
    assert all(runner.active_jobs() == 0 and not runner.is_busy()
               for runner in (*_runners(view), view._readout_jobs, view._levels_jobs))
    assert list(tmp_path.iterdir()) == []
