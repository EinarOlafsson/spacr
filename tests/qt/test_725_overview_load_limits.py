"""Round-5 Overview: Max wells per plate and Max fields per well.

Item 725, maintainer feedback 2026-10-25 round 5: "in sourse settings category the
user should be able to choose max number of wells and fields per well that are
loaded. wells is per plate".
"""
import numpy as np
import pytest
import tifffile

from PySide6.QtCore import Qt

from spacr.qt.widgets import qc_field_browser as browser


PATTERN = r"(?P<plateID>p\d+)-(?P<wellID>[A-Z]\d+)-f(?P<fieldID>\d+)-c(?P<chanID>\d+)\.tif"


def _plate(folder, wells, fields=(1, 2, 3)):
    folder.mkdir(parents=True, exist_ok=True)
    for index, well in enumerate(wells):
        for field in fields:
            for channel in (1, 2):
                tifffile.imwrite(folder / f"p1-{well}-f{field}-c{channel}.tif",
                                 np.full((16, 20), 100 * (index + 1) + 10 * field + channel, dtype=np.uint16))
    return folder


def _field(plate, well, field, time=""):
    return browser._OverviewField(plate, well, well[0], int(well[1:]), field, (), field_id=field, time_id=time)


def test_the_kept_wells_and_fields_do_not_depend_on_the_input_order():
    fields = [_field("p", well, field, time) for well in ("B01", "A10", "A02", "A01", "A03")
              for field in ("10", "2", "1", "3") for time in ("1", "2")]
    kept = browser._overview_limited_fields(fields, 2, 2)
    assert {field.well for field in kept} == {"A01", "A02"}
    assert {field.field_id for field in kept} == {"1", "2"}
    assert len(kept) == 2 * 2 * 2
    reversed_kept = browser._overview_limited_fields(list(reversed(fields)), 2, 2)
    assert {field.identity for field in reversed_kept} == {field.identity for field in kept}
    assert browser._overview_limited_fields(fields) == tuple(fields)
    assert [browser._overview_load_limit(value) for value in ("all", "", None, 0, "0", "3", 4, "x")] == [
        None, None, None, None, None, 3, 4, None]


def test_each_plate_is_capped_on_its_own():
    fields = ([_field("a", well, "1") for well in ("A01", "A02", "A03")]
              + [_field("b", well, "1") for well in ("A01", "B01", "B02", "C01", "C02")])
    kept = browser._overview_limited_fields(fields, 2)
    assert sorted((field.plate, field.well) for field in kept) == [
        ("a", "A01"), ("a", "A02"), ("b", "A01"), ("b", "B01")]


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
def view(qtbot, tmp_path):
    first = _plate(tmp_path / "plateA", ("A01", "A02", "B01"))
    second = _plate(tmp_path / "plateB", ("A01", "A02", "A03", "B01", "B02"))
    browser._OVERVIEW_PLANE_CACHE.clear()
    browser._OVERVIEW_INFO_CACHE.clear()
    dialog = browser._FieldOverview(settings={
        "source": f"{first}; {second}", "source_scope": "folder", "custom_regex": PATTERN,
        "channels": (1, 2), "geometry": (2, 3), "representatives": 1})
    qtbot.addWidget(dialog, before_close_func=lambda widget: _retire(widget, qtbot))
    dialog.resize(1300, 950)
    dialog.show()
    dialog._scan()
    qtbot.waitUntil(lambda: bool(dialog._fields) and dialog._normalization_ready
                    and not dialog._index_jobs.is_busy(), timeout=15000)
    dialog.fov_axis.setCurrentIndex(dialog.fov_axis.findData(None))
    return dialog


def _wells(view, plate):
    return sorted({field.well for field in view._fields if field.plate == plate})


def _not_loaded(view):
    return sorted(view.wells.item(row, column).text().split("\n")[0]
                  for row in range(view.wells.rowCount()) for column in range(view.wells.columnCount())
                  if view.wells.item(row, column) is not None
                  and view.wells.item(row, column).data(browser._OVERVIEW_NOT_LOADED_ROLE))


def test_two_plates_are_capped_independently_and_unloaded_wells_are_marked(view):
    assert len(view._fields) == len(view._all_fields) == (3 + 5) * 3
    view._set_load_limits("2", None)
    assert _wells(view, "plateA") == ["A01", "A02"] and _wells(view, "plateB") == ["A01", "A02"]
    assert len(view._all_fields) == (3 + 5) * 3
    view.plates.setCurrentIndex(view.plates.findData("plateA"))
    assert _not_loaded(view) == ["B01"]
    item = view.wells.item(1, 0)
    assert not item.flags() & Qt.ItemIsEnabled and "not loaded" in item.toolTip()
    assert set(view._well_positions) == {("plateA", "A01"), ("plateA", "A02")}
    view.plates.setCurrentIndex(view.plates.findData("plateB"))
    assert _not_loaded(view) == ["A03", "B01", "B02"]


def test_the_fields_per_well_cap(view):
    view._set_load_limits(None, 2)
    assert {field.field_id for field in view._fields} == {"1", "2"}
    assert len(view._fields) == (3 + 5) * 2
    assert _not_loaded(view) == []


def test_lowering_and_raising_a_cap_updates_the_caches(view, qtbot):
    view.plates.setCurrentIndex(view.plates.findData("plateA"))
    qtbot.waitUntil(lambda: all(not view.wells.item(*position).icon().isNull()
                                for position in view._well_positions.values()), timeout=20000)
    dropped = {field.identity for field in view._fields if field.plate == "plateA" and field.well != "A01"}
    assert any(set(key[0]) & dropped for key in view._thumbnail_cache)
    view._set_load_limits(1, None)
    assert not any(set(key[0]) & dropped for key in view._thumbnail_cache)
    assert _wells(view, "plateA") == ["A01"]
    view._set_load_limits("all", None)
    assert _wells(view, "plateA") == ["A01", "A02", "B01"] and _not_loaded(view) == []
    qtbot.waitUntil(lambda: all(not view.wells.item(*position).icon().isNull()
                                for position in view._well_positions.values()), timeout=20000)


def test_measurements_count_only_loaded_fields(view, qtbot, monkeypatch):
    measured = []
    monkeypatch.setattr(browser, "_overview_measurements_export",
                        lambda fields, *args, **kwargs: measured.append(tuple(fields)) or {
                            "cancelled": False, "errors": [], "path": "", "rows": 0})
    view._set_load_limits(1, 1)
    view.plates.setCurrentIndex(view.plates.findData("plateB"))
    view._measure()
    qtbot.waitUntil(lambda: bool(measured), timeout=15000)
    assert [(field.plate, field.well, field.field_id) for field in measured[0]] == [("plateB", "A01", "1")]
