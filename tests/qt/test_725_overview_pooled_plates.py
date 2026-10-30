"""Overview plate container: any number of plates, shown or hidden, folded, zoomed and pooled.

Item 725, 1.5.1.6, maintainer 2026-10-26: plates are added without a limit,
each has a show/hide button, one container holds the shown plates (fold one
or all, zoom and fit them together), wells opened on several plates fill the
same well and image views, and every well label names its plate once more
than one plate is loaded.
"""
import json

import numpy as np
import pytest
import tifffile

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication

from spacr.qt.widgets import qc_field_browser as browser


PATTERN = r"(?P<plateID>p\d+)-(?P<wellID>[A-Z]\d+)-f(?P<fieldID>\d+)-c(?P<chanID>\d+)\.tif"
PLATES = tuple(f"run7_plate{number}" for number in range(1, 7))
WELLS = ("A01", "A02", "A03", "B01", "B02", "B03", "C01", "C02", "C03", "D01", "D02", "D03")


def _plate(folder, wells, offset):
    folder.mkdir(parents=True, exist_ok=True)
    for index, well in enumerate(wells):
        for channel in (1, 2):
            pixels = np.full((24, 32), 100 * (index + 1) + offset + channel, dtype=np.uint16)
            tifffile.imwrite(folder / f"p1-{well}-f1-c{channel}.tif", pixels)
    return folder


@pytest.fixture
def plates(tmp_path):
    folders = [_plate(tmp_path / name, WELLS, 1000 * index) for index, name in enumerate(PLATES[:-1])]
    folders.append(_plate(tmp_path / PLATES[-1], WELLS[1:], 5000))
    return tuple(folders)


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


def _open(qtbot, source, geometry=(4, 3)):
    browser._OVERVIEW_PLANE_CACHE.clear()
    browser._OVERVIEW_INFO_CACHE.clear()
    view = browser._FieldOverview(settings={
        "source": source, "source_scope": "folder", "custom_regex": PATTERN,
        "channels": (1, 2), "geometry": geometry, "representatives": 1})
    qtbot.addWidget(view, before_close_func=lambda widget: _retire(widget, qtbot))
    view.resize(1500, 1000)
    view.show()
    view._scan()
    qtbot.waitUntil(lambda: bool(view._fields) and view._normalization_ready
                    and not view._index_jobs.is_busy(), timeout=20000)
    return view


@pytest.fixture
def view(qtbot, plates):
    return _open(qtbot, "; ".join(str(path) for path in plates))


def _item(view, plate, well):
    return view._plate_tables[plate].item(*view._well_positions[(plate, well)])


def _block(view, plate):
    return view._plate_blocks[view._plate_tables[plate]][0]


def _plate_menu(view, plate=None, well="A01"):
    plate = plate or view.plates.currentData()
    table = view._plate_tables[plate]
    return view._plate_table_menu(table, table.visualItemRect(_item(view, plate, well)).center())


def _close(menu):
    try:
        menu.close()
    except RuntimeError:
        pass


def _choose(view, text, plate=None, well="A01"):
    menu = _plate_menu(view, plate, well)
    _actions(menu)[text].trigger()
    _close(menu)


def _menu_actions(view, plate=None, well="A01"):
    menu = _plate_menu(view, plate, well)
    actions = _actions(menu)
    _close(menu)
    return actions


def _actions(menu):
    found = {}
    for action in menu.actions():
        if action.menu() is not None:
            found.update({f"{action.text()} > {key}": value for key, value in _actions(action.menu()).items()})
        if action.text():
            found[action.text()] = action
    return found


def _sections(menu):
    return [action.text() for action in menu.actions() if action.property("overviewMenuSection")]


def _ctrl_click(qtbot, view, plate, well):
    table = view._plate_tables[plate]
    qtbot.mouseClick(table.viewport(), Qt.LeftButton, Qt.ControlModifier,
                     table.visualItemRect(_item(view, plate, well)).center())


def test_six_plates_with_overlapping_wells_share_one_container(view, qtbot):
    view._set_plates_view("stacked")
    assert tuple(view._plate_tables) == PLATES
    assert view._plate_scroll.isVisible() and view._plate_scroll.widget() is view._plates_host
    ticks = {key.split(" > ")[1]: action.isChecked() for key, action in _menu_actions(view).items()
             if key.startswith("Shown plates… > ") and action.isCheckable()}
    assert ticks == {"All plates": True, **{plate: True for plate in PLATES}}
    for plate in PLATES[:-1]:
        assert _item(view, plate, "A01").data(Qt.UserRole) == "A01"
        assert {field.plate for field in view._well_fields[(plate, "A01")]} == {plate}
    assert ("run7_plate6", "A01") not in view._well_positions
    assert len({id(view._plate_tables[plate]) for plate in PLATES}) == 6
    qtbot.waitUntil(lambda: not _item(view, PLATES[0], "A01").icon().isNull(), timeout=20000)


def test_zoomed_container_decodes_only_the_plates_in_view(view, qtbot):
    view._set_plates_view("stacked")
    for _step in range(4):
        view._zoom_plate(1)
    scroll = view._plate_scroll
    qtbot.waitUntil(lambda: scroll.verticalScrollBar().maximum() > 0, timeout=5000)
    qtbot.waitUntil(lambda: not _item(view, PLATES[0], "A01").icon().isNull(), timeout=20000)
    qtbot.wait(300)
    qtbot.waitUntil(lambda: not view._thumbnail_jobs.active_jobs(), timeout=20000)
    assert _item(view, PLATES[-1], "D03").icon().isNull()
    assert not any(address[0] == PLATES[-1] for address in view._installed_thumbnails)
    scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
    scroll.horizontalScrollBar().setValue(scroll.horizontalScrollBar().maximum())
    qtbot.waitUntil(lambda: not _item(view, PLATES[-1], "D03").icon().isNull(), timeout=20000)


def test_the_plate_menu_shows_and_hides_plates_and_hidden_plates_stay_loaded(view, qtbot):
    view._set_plates_view("side")
    _choose(view, "Shown plates… > run7_plate3")
    assert "run7_plate3" not in view._plate_tables and len(view._plate_tables) == 5
    assert any(field.plate == "run7_plate3" for field in view._fields)
    assert view.plates.findData("run7_plate3") >= 0
    assert not _menu_actions(view)["Shown plates… > run7_plate3"].isChecked()
    _choose(view, "Shown plates… > run7_plate3")
    assert tuple(view._plate_tables) == PLATES
    _choose(view, "Hide plate run7_plate4", "run7_plate4")
    assert "run7_plate4" not in view._plate_tables
    _choose(view, "Shown plates… > Hide all")
    assert view._plate_tables == {} and view._visible_plates() == ()
    assert view._plates_empty.isVisible()
    assert not any(block.isVisible() for block, _header in view._plate_blocks.values())
    assert len(view._fields) and view.plates.count() == 6
    menu = view._plate_area_menu(view._plates_empty.geometry().center())
    actions = _actions(menu)
    assert not any(action.isChecked() for key, action in actions.items()
                   if key.startswith("Shown plates… > run7_plate"))
    actions["Shown plates… > run7_plate5"].trigger()
    _close(menu)
    assert tuple(view._plate_tables) == ("run7_plate5",) and not view._plates_empty.isVisible()
    _choose(view, "Shown plates… > All plates", "run7_plate5")
    assert tuple(view._plate_tables) == PLATES and view._plates_subset is None


def test_navigation_is_in_the_plate_menu_with_shortcuts_and_no_buttons(view, qtbot):
    from PySide6.QtGui import QKeySequence
    from PySide6.QtWidgets import QPushButton

    view._set_plates_view("stacked")
    menu = _plate_menu(view, "run7_plate2")
    assert _sections(menu)[-3:] == ["Navigation", "Wells", "Plates"]
    actions = _actions(menu)
    for text in ("Back to plate", "Fit plate", "Zoom in", "Zoom out", "Show plate metadata",
                 "Plates view > Single", "Plates view > Side by side", "Plates view > Stacked",
                 "Shown plates… > All plates", "Shown plates… > Hide all", "Previous plate", "Next plate",
                 "Add plates…", "Remove plate", "Expand all", "Collapse all",
                 "Collapse plate run7_plate2", "Hide plate run7_plate2"):
        assert text in actions, text
    assert actions["Plates view > Stacked"].isChecked() and not actions["Plates view > Single"].isChecked()
    assert [actions[text].shortcut().toString(QKeySequence.PortableText)
            for text in ("Fit plate", "Zoom in", "Zoom out")] == ["Ctrl+0", "Ctrl+=", "Ctrl+-"]
    shown = [button.text() for button in view.findChildren(QPushButton) if button.isVisible()]
    for text in ("Fit plate", "Previous plate", "Next plate", "Add plates…", "Remove plate",
                 "Back to plate", "Previous fields", "Next fields", "Shown plates…"):
        assert text not in shown, text
    assert not view.plates_view.isVisible()
    actions["Plates view > Side by side"].trigger()
    assert view._plates_view == "side"
    actions["Next plate"].trigger()
    assert view.plates.currentData() == "run7_plate3"
    _close(menu)
    table = view._plate_tables["run7_plate1"]
    table.setFocus()
    extent = view._plate_cell_extent
    qtbot.keyClick(table, Qt.Key_Equal, Qt.ControlModifier)
    assert view._plate_cell_extent[0] > extent[0] and not view._plate_fit
    qtbot.keyClick(table, Qt.Key_Minus, Qt.ControlModifier)
    qtbot.keyClick(table, Qt.Key_Minus, Qt.ControlModifier)
    assert view._plate_cell_extent[0] < extent[0]
    qtbot.keyClick(table, Qt.Key_0, Qt.ControlModifier)
    assert view._plate_fit
    view._well(_item(view, "run7_plate1", "A02"))
    actions = _menu_actions(view, "run7_plate1")
    assert "Previous fields" in actions and "Next fields" in actions and "Compare selected fields" in actions
    fields = view._well_panels[0].field_list
    view._field_menu(fields, fields.visualItemRect(fields.item(0)).center())
    field_menu = view._image_menu_shown
    field_menu.close()
    assert {"Back to plate", "Previous fields", "Next fields"} <= set(_actions(field_menu))
    _choose(view, "Back to plate", "run7_plate1")
    assert view._scope[1] is None


def test_plate_metadata_can_be_switched_off_from_the_menu_and_is_saved(view, qtbot, plates):
    view._set_plates_view("side")
    action = _menu_actions(view)["Show plate metadata"]
    assert action.isCheckable() and action.isChecked()
    assert all(table._overview_metadata for table in view._plate_tables.values())
    action.trigger()
    assert not any(table._overview_metadata for table in view._plate_tables.values())
    assert not _menu_actions(view)["Show plate metadata"].isChecked()
    path = plates[0] / "results" / "overview_view.json"
    qtbot.waitUntil(lambda: path.is_file() and json.loads(path.read_text())["view"].get("plate_metadata") is False,
                    timeout=10000)
    reopened = _open(qtbot, "; ".join(str(path) for path in plates))
    qtbot.waitUntil(lambda: not reopened._plate_metadata_action.isChecked(), timeout=10000)
    assert not any(table._overview_metadata for table in reopened._plate_blocks)


def test_plates_fold_to_their_header_bar_one_at_a_time_or_all_at_once(view, qtbot):
    view._set_plates_view("stacked")
    block = _block(view, "run7_plate2")
    qtbot.waitUntil(lambda: block.bar.isVisible(), timeout=5000)
    block.arrow.click()
    assert view._collapsed_plates == {"run7_plate2"}
    assert not view._plate_tables["run7_plate2"].isVisible()
    QApplication.processEvents()
    assert block.height() <= block.bar.sizeHint().height() + 2
    assert all(view._plate_tables[plate].isVisible() for plate in PLATES if plate != "run7_plate2")
    _choose(view, "Collapse all", "run7_plate1")
    assert view._collapsed_plates == set(PLATES)
    assert not any(table.isVisible() for table in view._plate_tables.values())
    assert all(_block(view, plate).bar.isVisible() for plate in PLATES)
    menu = view._plate_area_menu(view._plate_panel.body.rect().center())
    _actions(menu)["Expand all"].trigger()
    _close(menu)
    assert view._collapsed_plates == set()
    assert all(table.isVisible() for table in view._plate_tables.values())
    _block(view, "run7_plate4").arrow.click()
    view._set_plates_view("side")
    QApplication.processEvents()
    folded = _block(view, "run7_plate4")
    assert folded.width() <= folded.bar.sizeHint().width() + 2
    assert not view._plate_tables["run7_plate4"].isVisible()
    view._set_plates_view("single")
    view.plates.setCurrentIndex(view.plates.findData("run7_plate4"))
    assert view.wells.isVisible() and not view._plate_scroll.isVisible()


def test_the_container_zooms_and_fits_every_plate_together(view, qtbot):
    view._set_plates_view("side")
    scroll = view._plate_scroll
    widths = {plate: table.columnWidth(0) for plate, table in view._plate_tables.items()}
    for _step in range(3):
        view._zoom_plate(1)
    assert len({table.columnWidth(0) for table in view._plate_tables.values()}) == 1
    assert all(table.columnWidth(0) > widths[plate] for plate, table in view._plate_tables.items())
    for table in view._plate_tables.values():
        assert table.horizontalScrollBar().maximum() == 0 and table.verticalScrollBar().maximum() == 0
        assert sum(table.columnWidth(index) for index in range(table.columnCount())) <= table.viewport().width()
    assert scroll.horizontalScrollBar().maximum() > 0
    view._zoom_plate()
    assert scroll.horizontalScrollBar().maximum() == 0 and scroll.verticalScrollBar().maximum() == 0
    for table in view._plate_tables.values():
        assert table.horizontalScrollBar().maximum() == 0 and table.verticalScrollBar().maximum() == 0
        assert sum(table.columnWidth(index) for index in range(table.columnCount())) <= table.viewport().width()
        assert table.mapTo(view._plate_scroll.viewport(), QPoint(table.width(), 0)).x() <= scroll.viewport().width()


def test_wells_of_three_plates_fill_one_well_view_with_plate_qualified_labels(view, qtbot):
    view._set_plates_view("side")
    _item(view, "run7_plate1", "A01").setSelected(True)
    _ctrl_click(qtbot, view, "run7_plate2", "A01")
    _ctrl_click(qtbot, view, "run7_plate3", "A01")
    assert view._selected_wells == {(plate, "A01") for plate in PLATES[:3]}
    opened = view._open_selected_wells()
    assert opened == tuple((plate, "A01") for plate in PLATES[:3])
    panels = view._well_panels
    assert [(panel.field_list._overview_plate, panel.field_list._overview_well) for panel in panels] == list(opened)
    titles = [panel.title.text() for panel in panels]
    assert titles == ["Well plate1 · A01", "Well plate2 · A01", "Well plate3 · A01"]
    for panel, plate in zip(panels, PLATES[:3]):
        assert panel.isVisible()
        records = [panel.field_list.item(index).data(Qt.UserRole) for index in range(panel.field_list.count())]
        assert records and {record.plate for record in records} == {plate}
        captions = [panel.field_list.item(index).data(Qt.UserRole + 2) for index in range(panel.field_list.count())]
        assert all(caption.startswith(f"{plate[5:]} · A01 · field") for caption in captions)
    qtbot.waitUntil(lambda: len(view._image_views) == 3, timeout=10000)
    assert [panel.title.text() for panel in view._image_views] == [
        "Well plate1 · A01, field 1", "Well plate2 · A01, field 1", "Well plate3 · A01, field 1"]
    assert {panel.field.plate for panel in view._image_views} == set(PLATES[:3])
    view._well_panels[0].field_list.itemClicked.emit(view._well_panels[0].field_list.item(0))
    assert view._scope == ("run7_plate1", "A01") and view.plates.currentData() == "run7_plate1"
    assert len(view._well_panels) == 3


def test_right_click_shows_a_well_from_every_plate(view, qtbot):
    view._set_plates_view("stacked")
    table = view._plate_tables["run7_plate2"]
    position = table.visualItemRect(_item(view, "run7_plate2", "A01")).center()
    menu = view._plate_table_menu(table, position)
    actions = {action.text(): action for action in menu.actions()}
    menu.close()
    action = actions["Show A01 from every plate"]
    assert action.isEnabled()
    action.trigger()
    expected = [(plate, "A01") for plate in PLATES[:-1]]
    assert [(panel.field_list._overview_plate, panel.field_list._overview_well)
            for panel in view._well_panels] == expected
    assert len({panel.title.text() for panel in view._well_panels}) == 5
    qtbot.waitUntil(lambda: len(view._image_views) == 5, timeout=10000)
    assert sorted(panel.field.plate for panel in view._image_views) == list(PLATES[:-1])
    assert view.status.text() == "A01 from 5 plates."
    view._toggle_shown_plate("run7_plate4", False)
    assert ("run7_plate4", "A01") not in [(panel.field_list._overview_plate, panel.field_list._overview_well)
                                          for panel in view._well_panels]
    assert "run7_plate4" not in {panel.field.plate for panel in view._image_views}
    assert len(view._well_panels) == 4


def test_double_clicking_wells_on_two_plates_keeps_both_in_the_well_view(view, qtbot):
    view._set_plates_view("side")
    view._well(_item(view, "run7_plate1", "B02"))
    view._well(_item(view, "run7_plate5", "B02"))
    assert [(panel.field_list._overview_plate, panel.field_list._overview_well)
            for panel in view._well_panels] == [("run7_plate1", "B02"), ("run7_plate5", "B02")]
    assert view._scope == ("run7_plate5", "B02")
    assert {field.plate for field in view._visible} == {"run7_plate5"}


def test_labels_stay_as_they_were_with_one_plate(qtbot, plates):
    view = _open(qtbot, str(plates[0]))
    assert view.plates.count() == 1
    plate = view.plates.currentData()
    view._well(_item(view, plate, "A01"))
    panel = view._well_panels[0]
    assert panel.title.text() == "Well A01"
    assert panel.field_list.item(0).data(Qt.UserRole + 2).startswith("A01 · field 1 ·")
    view._open_image_views([panel.field_list.item(0).data(Qt.UserRole)])
    assert view._image_views[0].title.text() == "Well A01, field 1"
    assert view._well_label(plate, "A01") == "A01"
    assert not view._plate_scroll.isVisible()


def test_short_plate_names_drop_shared_words_and_stay_distinct():
    short = browser._overview_short_plate_names
    assert short(["screen_2026_plate1", "screen_2026_plate2"]) == {
        "screen_2026_plate1": "plate1", "screen_2026_plate2": "plate2"}
    assert short(["plateA", "plateB"]) == {"plateA": "plateA", "plateB": "plateB"}
    assert short(["Plate 1", "Plate 1 (2)"]) == {"Plate 1": "Plate 1", "Plate 1 (2)": "Plate 1 (2)"}
    long = short(["CQ1_projection_sulfide_screen", "Other"])
    assert long["Other"] == "Other" and len(long["CQ1_projection_sulfide_screen"]) <= 14
    assert short(["only"]) == {"only": "only"}


def test_visibility_and_folds_survive_saving_and_reopening(view, qtbot, plates):
    view._set_plates_view("stacked")
    _choose(view, "Shown plates… > run7_plate3")
    _block(view, "run7_plate2").arrow.click()
    path = plates[0] / "results" / "overview_view.json"
    qtbot.waitUntil(lambda: path.is_file() and "collapsed_plates" in path.read_text()
                    and len(json.loads(path.read_text())["view"]["collapsed_plates"]) == 1, timeout=10000)
    saved = json.loads(path.read_text())["view"]
    assert saved["plates_view"] == "stacked" and len(saved["shown_plates"]) == 5
    assert saved["no_plates_shown"] is False
    reopened = _open(qtbot, "; ".join(str(path) for path in plates))
    qtbot.waitUntil(lambda: reopened._plates_view == "stacked" and reopened._collapsed_plates == {"run7_plate2"},
                    timeout=10000)
    assert "run7_plate3" not in reopened._plate_tables and len(reopened._plate_tables) == 5
    assert not _menu_actions(reopened)["Shown plates… > run7_plate3"].isChecked()
    assert not reopened._plate_tables["run7_plate2"].isVisible()
    assert reopened._plate_blocks[reopened._plate_tables["run7_plate2"]][0].arrow.isChecked()
    reopened._hide_every_plate()
    qtbot.waitUntil(lambda: json.loads(path.read_text())["view"]["no_plates_shown"] is True, timeout=10000)
    again = _open(qtbot, "; ".join(str(path) for path in plates))
    qtbot.waitUntil(lambda: again._plates_view == "stacked" and again._plates_subset == (), timeout=10000)
    assert again._plate_tables == {} and again._plates_empty.isVisible()


def test_hidden_pooled_plate_receipts_cannot_redirect_the_current_image(qtbot, tmp_path, monkeypatch):
    folders = (_plate(tmp_path / "first", ("A01",), 0),
               _plate(tmp_path / "second", ("A01",), 1000),
               _plate(tmp_path / "current", ("B02",), 2000))
    source_bytes = {str(path): path.read_bytes() for folder in folders for path in folder.glob("*.tif")}
    receipts = []
    original = browser._FieldOverview._plate_arrow_toggled

    def arrow_received(owner, table, collapsed):
        receipts.append((table, collapsed))
        return original(owner, table, collapsed)

    monkeypatch.setattr(browser._FieldOverview, "_plate_arrow_toggled", arrow_received)
    view = _open(qtbot, "; ".join(map(str, folders)), geometry=(2, 2))
    view._set_plates_view("stacked")
    table = view._plate_tables["first"]
    arrow = view._plate_blocks[table][0].arrow
    qtbot.mouseClick(arrow, Qt.LeftButton)
    assert receipts and receipts[-1][0] is table and receipts[-1][1] is True
    receipt = receipts[-1]
    qtbot.mouseClick(arrow, Qt.LeftButton)
    menu = _plate_menu(view, "first")
    compare = next(action for action in menu.actions() if action.text() == "Show A01 from every plate")
    assert compare.isEnabled()
    menu.close()
    for name in ("first", "second"):
        actions = _menu_actions(view, "current", "B02")
        actions["Shown plates… > " + name].trigger()
    assert tuple(view._plate_tables) == ("current",)
    assert table._overview_plate is None
    view._well(_item(view, "current", "B02"))
    qtbot.waitUntil(lambda: view.fields.count() == 1)
    qtbot.mouseClick(view.fields.viewport(), Qt.LeftButton,
                     pos=view.fields.visualItemRect(view.fields.item(0)).center())
    qtbot.waitUntil(lambda: view._main_view_field is not None and view.view._item is not None
                   and not view.view._item.pixmap().isNull())
    field = view._main_view_field
    selected = set(view._selected_wells)
    image = view.view._item.pixmap().toImage().copy()
    collapsed = set(view._collapsed_plates)
    original(view, *receipt)
    compare.trigger()
    assert view.status.text() == browser.tr("These wells have no image at the selected acquisition axes.")
    assert view._main_view_field is field and field.plate == "current" and field.well == "B02"
    assert view.view._item.pixmap().toImage() == image
    assert set(view._selected_wells) == selected and view._collapsed_plates == collapsed
    for name in ("first", "second"):
        _menu_actions(view, "current", "B02")["Shown plates… > " + name].trigger()
    fresh = view._plate_blocks[view._plate_tables["first"]][0].arrow
    qtbot.mouseClick(fresh, Qt.LeftButton)
    assert "first" in view._collapsed_plates
    qtbot.mouseClick(fresh, Qt.LeftButton)
    fresh_menu = _plate_menu(view, "first")
    fresh_compare = next(action for action in fresh_menu.actions() if action.text() == "Show A01 from every plate")
    fresh_compare.trigger()
    fresh_menu.close()
    qtbot.waitUntil(lambda: {panel.field.plate for panel in view._image_views} == {"first", "second"})
    assert {str(path): path.read_bytes() for folder in folders for path in folder.glob("*.tif")} == source_bytes
