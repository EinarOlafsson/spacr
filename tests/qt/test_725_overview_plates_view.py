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
