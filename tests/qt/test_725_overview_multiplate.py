"""Round-5 Overview: several plates from one or more source paths.

Item 725, maintainer feedback 2026-10-25 round 5: "for the overvie i need to be
able to load several plates" and "the sourse should be able to be one path (one
plate), or one path with subdirectories that are plates, or several paths where
each is either a plate or a folder with plates."
"""
import os

import numpy as np
import pandas as pd
import pytest
import tifffile

from PySide6.QtCore import Qt

from spacr.qt.widgets import qc_field_browser as browser


PATTERN = r"(?P<plateID>p\d+)-(?P<wellID>[A-Z]\d+)-f(?P<fieldID>\d+)-c(?P<chanID>\d+)\.tif"


def _plate(folder, wells=("A01", "A02", "B01"), offset=0):
    folder.mkdir(parents=True, exist_ok=True)
    for index, well in enumerate(wells):
        for channel in (1, 2):
            pixels = np.full((24, 32), 100 * (index + 1) + offset + channel, dtype=np.uint16)
            tifffile.imwrite(folder / f"p1-{well}-f1-c{channel}.tif", pixels)
    return folder


# --- Plate detection ---


def test_one_path_with_images_is_one_plate(tmp_path):
    folder = _plate(tmp_path / "plateA")
    found = browser._overview_detect_plates(str(folder))
    assert [(plate["path"], plate["name"], plate["images"], plate["mixed"]) for plate in found["plates"]] == [
        (str(folder), "plateA", 6, False)]


def test_a_folder_of_plate_folders_skips_output_and_hidden_folders(tmp_path):
    parent = tmp_path / "screen"
    for name in ("plate1", "plate2", "plate10"):
        _plate(parent / name)
    _plate(parent / "masks")
    _plate(parent / "measurements")
    _plate(parent / ".cache")
    (parent / "empty").mkdir()
    found = browser._overview_detect_plates(str(parent))
    assert [plate["name"] for plate in found["plates"]] == ["plate1", "plate2", "plate10"]
    assert all(plate["source"] == str(parent) for plate in found["plates"])


def test_several_mixed_paths_and_duplicate_names(tmp_path):
    single = _plate(tmp_path / "a" / "plate1")
    parent = tmp_path / "b"
    _plate(parent / "plate1")
    _plate(parent / "plate2")
    found = browser._overview_detect_plates(f"{single}; {parent}")
    assert [plate["name"] for plate in found["plates"]] == ["a/plate1", "b/plate1", "plate2"]
    assert len({plate["path"] for plate in found["plates"]}) == 3
    assert browser._overview_source_paths([str(single), f"{parent} ; {single}"]) == [str(single), str(parent)]


def test_a_folder_with_images_and_plate_subfolders_is_flagged_and_can_use_its_subfolders(tmp_path):
    folder = _plate(tmp_path / "mixed")
    _plate(folder / "inner1")
    _plate(folder / "inner2")
    found = browser._overview_detect_plates(str(folder))
    assert [(plate["name"], plate["mixed"]) for plate in found["plates"]] == [("mixed", True)]
    found = browser._overview_detect_plates(str(folder), use_subfolders=[str(folder)])
    assert [plate["name"] for plate in found["plates"]] == ["inner1", "inner2"]


def test_an_empty_folder_gives_a_clear_message(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    found = browser._overview_detect_plates(str(empty))
    assert found["plates"] == [] and "No images were found in" in found["messages"][0]
    fields, detection = browser._overview_index_sources(str(empty), PATTERN)
    assert fields == () and "No images were found in" in detection["messages"][0]
    with pytest.raises(browser._OverviewValidationError, match="existing image folder"):
        browser._overview_index_sources("", PATTERN)
    with pytest.raises(browser._OverviewValidationError, match="existing image folder"):
        browser._overview_index_sources(str(empty / "missing"), PATTERN)


def test_indexing_two_plates_with_the_same_wells_keeps_them_apart(tmp_path):
    first, second = _plate(tmp_path / "plateA"), _plate(tmp_path / "plateB", offset=1000)
    fields, detection = browser._overview_index_sources(f"{first}; {second}", PATTERN, source_scope="folder")
    assert {field.plate for field in fields} == {"plateA", "plateB"}
    assert {(field.source, field.plate_id) for field in fields} == {(str(first.resolve()), "p1"),
                                                                    (str(second.resolve()), "p1")}
    a01 = [field for field in fields if field.well == "A01"]
    assert len(a01) == 2 and len({field.identity for field in a01}) == 2
    assert len({browser._overview_image_key(field) for field in a01}) == 2
    alone, _ = browser._overview_index_sources(str(first), PATTERN, source_scope="folder")
    assert {field.plate for field in alone} == {"p1"} and {field.source for field in alone} == {str(first.resolve())}


# --- The browser with several plates ---


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
def plates(tmp_path):
    return _plate(tmp_path / "plateA"), _plate(tmp_path / "plateB", wells=("A01", "A02", "B01", "B02"), offset=1000)


def _open(qtbot, source):
    browser._OVERVIEW_PLANE_CACHE.clear()
    browser._OVERVIEW_INFO_CACHE.clear()
    view = browser._FieldOverview(settings={
        "source": source, "source_scope": "folder", "custom_regex": PATTERN,
        "channels": (1, 2), "geometry": (2, 3), "representatives": 1})
    qtbot.addWidget(view, before_close_func=lambda widget: _retire(widget, qtbot))
    view.resize(1300, 950)
    view.show()
    view._scan()
    qtbot.waitUntil(lambda: bool(view._fields) and view._normalization_ready
                    and not view._index_jobs.is_busy(), timeout=15000)
    return view


@pytest.fixture
def view(qtbot, plates):
    return _open(qtbot, "; ".join(str(path) for path in plates))


def _show(view, plate):
    view.plates.setCurrentIndex(view.plates.findData(plate))


def test_each_plate_keeps_its_own_wells_and_annotations(view, qtbot, plates):
    assert [view.plates.itemData(index) for index in range(view.plates.count())] == ["plateA", "plateB"]
    assert view._metadata_source == str(plates[0])
    _show(view, "plateA")
    assert set(view._well_positions) == {"A01", "A02", "B01"}
    view._apply_annotation({("plateA", "A01")}, "cell", "HeLa")
    _show(view, "plateB")
    assert set(view._well_positions) == {"A01", "A02", "B01", "B02"}
    assert view._annotation_snapshot().get(("plateB", "A01"), {}).get("cell") is None
    assert view._annotation_snapshot()[("plateA", "A01")]["cell"] == "HeLa"
    assert all(field.plate == "plateB" for fields in view._well_fields.values() for field in fields)
    qtbot.waitUntil(lambda: not view.wells.item(*view._well_positions["A01"]).icon().isNull(), timeout=20000)
    key = view.wells.item(*view._well_positions["A01"]).data(Qt.UserRole + 1)
    assert key[0][0][0] == "plateB"


def test_plate_buttons_step_between_plates(view):
    previous, following, _add, remove = view._plate_buttons
    assert previous.isEnabled() and following.isEnabled() and remove.isEnabled()
    _show(view, "plateA")
    following.click()
    assert view.plates.currentData() == "plateB"
    following.click()
    assert view.plates.currentData() == "plateA"
    previous.click()
    assert view.plates.currentData() == "plateB"


def test_collapsed_lines_are_kept_per_plate(view):
    _show(view, "plateA")
    view._set_line_collapsed("row", 0, True)
    _show(view, "plateB")
    assert view._collapsed_lines() == (set(), set())
    assert view.wells.rowHeight(0) != browser._OVERVIEW_COLLAPSED_STRIP
    view._set_line_collapsed("column", 1, True)
    _show(view, "plateA")
    assert view._collapsed_lines() == ({0}, set())
    assert view.wells.rowHeight(0) == browser._OVERVIEW_COLLAPSED_STRIP
    assert view._collapsed["plateB"] == (set(), {1})


def test_a_comparison_group_can_span_plates(view):
    view._annotations = {("plateA", "A01"): {"set": "x"}, ("plateA", "A02"): {"set": "x"},
                         ("plateB", "A01"): {"set": "x"}, ("plateB", "B02"): {"set": "y"}}
    view._refresh_comparison_choices()
    columns = {view.comparison_dimensions.topLevelItem(index).text(0)
               for index in range(view.comparison_dimensions.topLevelItemCount())}
    assert {"plate", "set"} <= columns
    groups = [{"plate": "plateA", "set": "x"}, {"plate": "plateB", "set": "x"}]
    assert [browser._overview_group_count(view._fields, view._annotation_snapshot(), group) for group in groups] == [2, 1]
    frame = pd.DataFrame([dict(plate=field.plate, well=field.well, value=index,
                               **view._annotation_snapshot().get((field.plate, field.well), {}))
                          for index, field in enumerate(view._fields)])
    grouped = browser._overview_grouped_frame(frame, groups, ["plate A set x", "plate B set x"])
    assert set(grouped["_overview_group"]) == {"plate A set x", "plate B set x"}
    assert set(grouped.loc[grouped["_overview_group"] == "plate B set x", "plate"]) == {"plateB"}


def test_saving_and_reopening_the_first_path_restores_both_plates(view, qtbot, plates):
    view._set_line_collapsed("row", 1, True)
    path = plates[0] / "results" / "overview_view.json"
    qtbot.waitUntil(lambda: path.is_file() and str(plates[1]) in path.read_text(), timeout=10000)
    fields, info = browser._overview_wizard_probe(str(plates[0]), PATTERN, "auto", __import__("threading").Event(),
                                                  "folder")
    assert {field.plate for field in fields} == {"plateA", "plateB"}
    assert info["restored_sources"] == [str(plates[1])]
    reopened = _open(qtbot, "; ".join(info["sources"]))
    assert {field.plate for field in reopened._fields} == {"plateA", "plateB"}
    qtbot.waitUntil(lambda: any(rows for rows, _columns in reopened._collapsed.values()), timeout=10000)


def test_removing_a_plate_drops_its_caches_selection_and_collapse_state(view, qtbot, plates):
    _show(view, "plateB")
    qtbot.waitUntil(lambda: all(not view.wells.item(*position).icon().isNull()
                                for position in view._well_positions.values()), timeout=20000)
    view.wells.item(*view._well_positions["A01"]).setSelected(True)
    view._set_line_collapsed("row", 1, True)
    removed = {field.identity for field in view._fields if field.plate == "plateB"}
    assert ("plateB", "A01") in view._selected_wells
    assert any(set(key[0]) & removed for key in view._thumbnail_cache)
    view._remove_plate("plateB")
    assert {field.plate for field in view._fields} == {"plateA"}
    assert view.plates.count() == 1 and not view._plate_buttons[3].isEnabled()
    assert not any(address[0] == "plateB" for address in view._selected_wells)
    assert "plateB" not in view._collapsed
    assert not any(set(key[0]) & removed for key in view._thumbnail_cache)
    assert not any(isinstance(key, tuple) and key and key[0] in removed for key in view._cache)
    assert browser._overview_source_paths(view.source.text()) == [str(plates[0])]


def test_plates_can_be_added_to_an_open_overview(qtbot, plates):
    view = _open(qtbot, str(plates[0]))
    assert {field.plate for field in view._fields} == {"p1"}
    view._add_sources([str(plates[1])])
    qtbot.waitUntil(lambda: view.plates.count() == 2, timeout=15000)
    assert {field.plate for field in view._fields} == {"p1", "plateB"}
    assert browser._overview_source_paths(view.source.text()) == [str(plates[0]), str(plates[1])]


def test_the_metadata_step_lists_the_detected_plates_and_can_remove_one(qtbot, tmp_path):
    parent = tmp_path / "screen"
    for name in ("plate1", "plate2", "plate3"):
        _plate(parent / name)
    wizard = browser._OverviewMetadataWizard(str(parent), {"custom_regex": PATTERN})
    wizard._startup_timer.stop()
    qtbot.addWidget(wizard)
    wizard.scope.setCurrentIndex(wizard.scope.findData("folder"))
    wizard._scan()
    qtbot.waitUntil(lambda: wizard.plate_table.rowCount() == 3, timeout=15000)
    assert [wizard.plate_table.item(row, 0).text() for row in range(3)] == ["plate1", "plate2", "plate3"]
    assert [wizard.plate_table.item(row, 1).text() for row in range(3)] == ["6", "6", "6"]
    assert "3 plates found." in wizard.plate_summary.text()
    wizard.plate_table.item(0, 0).setText("Control plate")
    assert wizard._plate_names == {"plate1": "Control plate"}
    wizard.plate_table.selectRow(1)
    wizard._remove_selected_plates()
    qtbot.waitUntil(lambda: wizard.plate_table.rowCount() == 2, timeout=15000)
    qtbot.waitUntil(lambda: wizard.next.isEnabled(), timeout=15000)
    config = wizard.configuration()
    assert config["excluded_plates"] == (str(parent / "plate2"),)
    assert config["plate_names"] == {"plate1": "Control plate"}
    assert {field.plate for field in wizard._fields} == {"plate1", "plate3"}
    wizard.reject()


def test_annotations_follow_a_plate_from_one_source_to_several(qtbot, plates):
    alone = _open(qtbot, str(plates[0]))
    alone._apply_annotation({("p1", "A01")}, "cell", "HeLa")
    saved = plates[0] / "results" / "overview_annotations.json"
    qtbot.waitUntil(lambda: saved.is_file() and "plate_key" in saved.read_text(), timeout=10000)
    both = _open(qtbot, f"{plates[0]}; {plates[1]}")
    qtbot.waitUntil(lambda: both._annotation_snapshot().get(("plateA", "A01"), {}).get("cell") == "HeLa",
                    timeout=10000)
    assert ("plateB", "A01") not in both._annotation_snapshot()
