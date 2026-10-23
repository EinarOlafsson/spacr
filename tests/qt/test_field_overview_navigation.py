"""Real source planes pin acquisition navigation and retained overview identities."""
import threading

import numpy as np
import pytest
import tifffile

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog

from spacr import tabular
from spacr.qt.widgets import qc_field_browser as module


PATTERN = (r"(?P<plateID>p\d+)-(?P<wellID>[A-Z]\d+)-f(?P<fieldID>\d+)"
           r"-t(?P<timeID>\d+)-z(?P<sliceID>\d+)-c(?P<chanID>\d+)\.tif")


@pytest.fixture
def source_planes(tmp_path):
    source = tmp_path / "images"
    source.mkdir()
    paths = []
    for well in ("A01", "B02"):
        for field in (1, 2, 10):
            for time in (1, 2, 10):
                for z in (1, 2):
                    if well == "B02" and field == 10 and time == 10:
                        continue
                    path = source / f"p1-{well}-f{field}-t{time}-z{z}-c1.tif"
                    pixels = (np.arange(1200, dtype=np.uint16).reshape(30, 40)
                              + field * 10 + time + z)
                    if well == "A01" and field == time == z == 1:
                        pixels = np.arange(1536 * 2048, dtype=np.uint16).reshape(1536, 2048)
                    tifffile.imwrite(path, pixels)
                    paths.append(path)
    return source, tuple(paths)


@pytest.fixture
def overview(qtbot, source_planes):
    source, paths = source_planes
    dialog = module._FieldOverview(settings={
        "source": str(source), "source_scope": "folder", "custom_regex": PATTERN,
        "channels": (1,), "geometry": (2, 3), "representatives": 1})
    qtbot.addWidget(dialog)
    dialog.show()
    dialog._scan()
    qtbot.waitUntil(lambda: bool(dialog._fields) and dialog._normalization_ready
                    and not dialog._index_jobs.is_busy(), timeout=10000)
    yield dialog
    dialog._cancel()
    for runner in (dialog._index_jobs, dialog._image_jobs, dialog._thumbnail_jobs,
                   dialog._field_thumbnail_jobs, dialog._montage_jobs, dialog._export_jobs):
        runner.shutdown(timeout_ms=3000)
    dialog.close()


def _choose(combo, value):
    index = combo.findData(value)
    assert index >= 0
    combo.setCurrentIndex(index)


def test_real_axes_keep_well_identity_and_refuse_missing_field_borrowing(overview, qtbot):
    dialog = overview
    assert (dialog.wells.rowCount(), dialog.wells.columnCount()) == (2, 3)
    assert [dialog.fov_axis.itemData(i) for i in range(dialog.fov_axis.count())] == [None, "1", "2", "10"]
    assert [dialog.time_axis.itemData(i) for i in range(dialog.time_axis.count())] == ["1", "2", "10"]
    plate = dialog.plates.currentData()
    position = dialog._well_positions["B02"]
    dialog._well(dialog.wells.item(*position))
    assert dialog._scope == (plate, "B02")
    _choose(dialog.fov_axis, "10")
    _choose(dialog.time_axis, "10")
    assert dialog._scope == (plate, "B02")
    assert dialog._visible == ()
    assert dialog.fields.count() == 0
    assert dialog.view._item is None
    _choose(dialog.time_axis, "2")
    assert dialog._visible
    assert all(field.well == "B02" and field.time_id == "2"
               and field.field_id == "10" for field in dialog._visible)
    _choose(dialog.z_axis, "2")
    assert all(field.z_id == "2" and field.well == "B02" for field in dialog._visible)
    assert dialog._scope == (plate, "B02")


def test_comparison_uses_explicit_source_selection_and_axis_wheel_is_not_zoom(overview, qtbot):
    dialog = overview
    dialog._well(dialog.wells.item(*dialog._well_positions["A01"]))
    first, second = dialog.fields.item(0), dialog.fields.item(1)
    first.setSelected(True)
    second.setSelected(True)
    identities = {first.data(Qt.UserRole).identity, second.data(Qt.UserRole).identity}
    dialog._compare()
    qtbot.waitUntil(lambda: not dialog._montage_jobs.is_busy()
                    and dialog.view._item is not None, timeout=10000)
    assert set(dialog._selected_fields) == identities
    signature = dialog._comparison_signature
    assert set(signature[0]) == identities
    _choose(dialog.wheel_axis, "time")
    previous = dialog.view.transform()
    dialog._step_axis(1)
    assert dialog.time_axis.currentData() == "2"
    assert dialog._scope[1] == "A01"
    assert dialog.view.transform() == previous
    assert set(dialog._selected_fields) == identities


def test_condition_tables_are_plate_qualified_and_never_modify_input(
        overview, source_planes, tmp_path, monkeypatch, qtbot):
    dialog = overview
    source, paths = source_planes
    original = {path: path.read_bytes() for path in paths}
    plate = dialog.plates.currentData()
    _choose(dialog.annotation_scope, "plate")
    dialog.annotation_text.setText("control")
    dialog._annotate()
    assert dialog._conditions == {(plate, "A01"): "control", (plate, "B02"): "control"}
    _choose(dialog.annotation_scope, "row")
    _choose(dialog.rows, "1")
    dialog.annotation_text.setText("treated")
    dialog._annotate()
    assert dialog._conditions[(plate, "A01")] == "treated"
    assert dialog._conditions[(plate, "B02")] == "control"
    output = tmp_path / "conditions.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(output), ""))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(output), ""))
    qtbot.waitUntil(lambda: not dialog._export_jobs.is_busy())
    dialog._save_annotations()
    qtbot.waitUntil(lambda: not dialog._export_jobs.is_busy(), timeout=10000)
    assert output.is_file()
    table = tabular.read_table(output, canonicalise=False)
    assert set(table.columns) == {"plate", "well", "condition"}
    assert set(table["plate"]) == {plate}
    expected = dict(dialog._conditions)
    dialog._conditions.clear()
    dialog._load_annotations()
    qtbot.waitUntil(lambda: not dialog._export_jobs.is_busy(), timeout=10000)
    assert dialog._conditions == expected
    table.loc[0, "plate"] = "foreign-source"
    tabular.write_table(table, output, canonicalise=False)
    dialog._load_annotations()
    qtbot.waitUntil(lambda: not dialog._export_jobs.is_busy(), timeout=10000)
    assert dialog._conditions == expected
    assert "outside this source" in dialog.status.text()
    assert {path: path.read_bytes() for path in paths} == original


def test_full_source_export_choices_remain_independent_of_preview_cache(overview):
    dialog = overview
    dialog._well(dialog.wells.item(*dialog._well_positions["A01"]))
    dialog.fields.setCurrentRow(0)
    field = dialog.fields.item(0).data(Qt.UserRole)
    dialog._individual = True
    _choose(dialog.export_scope, "field")
    assert dialog._export_selection() == (field,)
    assert dialog.pdf_quality.currentData() is None
    assert dialog.pdf_dpi.currentData() == 300
    _choose(dialog.export_scope, "plate")
    selected = dialog._export_selection()
    assert selected
    assert {record.well for record in selected} == {"A01", "B02"}
    assert all(record.plate == field.plate for record in selected)


def test_progressive_detail_is_bounded_and_native_images_own_their_pixels(
        overview, monkeypatch, qtbot, qapp):
    from PySide6.QtCore import QThread

    dialog = overview
    requests, shown = [], []
    original_read, original_show = module._overview_pixels, dialog._show_pixels

    def read(field, channels, limits, edge=1200, **options):
        requests.append((edge, threading.get_ident()))
        return original_read(field, channels, limits, edge=edge, **options)

    def show(pixels, preserve_view=False):
        shown.append((pixels.shape, QThread.currentThread()))
        return original_show(pixels, preserve_view)

    monkeypatch.setattr(module, "_overview_pixels", read)
    monkeypatch.setattr(dialog, "_show_pixels", show)
    dialog._well(dialog.wells.item(*dialog._well_positions["A01"]))
    dialog.fields.setCurrentRow(0)
    dialog._field(dialog.fields.item(0))
    qtbot.waitUntil(lambda: any(shape[1] == 1024 for shape, _ in shown)
                    and not dialog._image_jobs.is_busy(), timeout=10000)
    assert any(shape[1] == 256 for shape, _ in shown)
    assert all(thread == qapp.thread() for _, thread in shown)
    dialog.view.zoom_by(8)
    dialog._request_detail()
    qtbot.waitUntil(lambda: any(edge == 4096 for edge, _ in requests)
                    and not dialog._image_jobs.is_busy(), timeout=10000)
    assert max(edge for edge, _ in requests) <= 4096
    assert all(ident != threading.get_ident() for _, ident in requests)
    assert len(dialog._cache) <= 8
    before = dialog.view._item.pixmap().toImage().copy()
    cached = next(reversed(dialog._cache.values()))
    cached.fill(0)
    assert dialog.view._item.pixmap().toImage() == before
    assert dialog.view._user_zoomed


def test_small_physical_window_keeps_configuration_and_images_reachable(overview, qtbot):
    from PySide6.QtCore import QSize
    from spacr.qt.gui_scale import _resize_actual

    dialog = overview
    _resize_actual(dialog, QSize(760, 620))
    qtbot.waitUntil(lambda: dialog.size() == QSize(760, 620))
    assert dialog._controls_scroll.isVisible()
    assert dialog._splitter.isVisible()
    assert dialog._splitter.height() > 0
    for control in (dialog.source, dialog.time_axis, dialog.annotation_text,
                    dialog.pdf_quality, dialog.measurements):
        dialog._controls_scroll.ensureWidgetVisible(control)
        assert dialog._controls_scroll.viewport().rect().intersects(
            control.rect().translated(control.mapTo(dialog._controls_scroll.viewport(),
                                                   control.rect().topLeft())))
    assert dialog.size() == QSize(760, 620)
