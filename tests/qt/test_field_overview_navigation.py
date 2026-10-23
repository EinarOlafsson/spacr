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
    qtbot.addWidget(dialog, before_close_func=lambda widget: _retire_overview(widget, qtbot))
    dialog.show()
    dialog._scan()
    qtbot.waitUntil(lambda: bool(dialog._fields) and dialog._normalization_ready
                    and not dialog._index_jobs.is_busy(), timeout=10000)
    yield dialog


def _retire_overview(dialog, qtbot):
    dialog._cancel()
    runners = (dialog._index_jobs, dialog._image_jobs, dialog._thumbnail_jobs,
               dialog._field_thumbnail_jobs, dialog._montage_jobs, dialog._export_jobs)
    qtbot.waitUntil(lambda: all(runner.active_jobs() == 0 for runner in runners), timeout=10000)
    for runner in runners:
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
    assert not dialog._controls_scroll.isVisible()
    dialog.options_button.click()
    assert dialog._controls_scroll.isVisible()
    assert dialog._splitter.isVisible()
    assert dialog._splitter.height() > 0
    assert dialog.time_axis.isVisible()
    for control in (dialog.source, dialog.annotation_text,
                    dialog.pdf_quality, dialog.measurements):
        dialog._controls_scroll.ensureWidgetVisible(control)
        assert dialog._controls_scroll.viewport().rect().contains(
            control.rect().translated(control.mapTo(dialog._controls_scroll.viewport(),
                                                   control.rect().topLeft())))
    assert dialog.size() == QSize(760, 620)


def test_embedded_nonzero_axes_survive_header_discovery_and_reference_sampling(tmp_path, qtbot):
    source = tmp_path / "stack"
    source.mkdir()
    pixels = np.empty((3, 2, 4, 20, 24), dtype=np.uint16)
    plane = np.arange(480, dtype=np.uint16).reshape(20, 24)
    for time in range(3):
        for channel in range(2):
            for z in range(4):
                pixels[time, channel, z] = np.roll(plane, time + z, axis=1) + time * 1000 + z * 100 + channel * 20
    tifffile.imwrite(source / "p1-A01-f1.tif", pixels,
                     metadata={"axes": "TCZYX"}, photometric="minisblack")
    pattern = r"(?P<plateID>p\d+)-(?P<wellID>[A-Z]\d+)-f(?P<fieldID>\d+)\.tif"
    dialog = module._FieldOverview(settings={"source": str(source), "source_scope": "folder",
        "custom_regex": pattern, "channels": (1,), "geometry": (1, 1),
        "time_index": 1, "z_index": 2, "z_mode": "plane",
        "normalization": {"mode": "shared", "percentiles": (0., 100.)}})
    qtbot.addWidget(dialog)
    dialog.show()
    try:
        dialog._scan()
        qtbot.waitUntil(lambda: bool(dialog._fields) and dialog._normalization_ready
                        and not dialog._index_jobs.is_busy(), timeout=10000)
        assert dialog.time_axis.currentData() == 1
        assert dialog.z_axis.currentData() == 2
        expected = pixels[1, 1, 2]
        assert dialog._normalization_plan["bounds"][1] == (float(expected.min()), float(expected.max()))
        dialog._well(dialog.wells.item(*dialog._well_positions["A01"]))
        dialog.fields.setCurrentRow(0)
        dialog._field(dialog.fields.item(0))
        qtbot.waitUntil(lambda: dialog.view._item is not None
                        and not dialog._image_jobs.is_busy(), timeout=10000)
        image = dialog.view._item.pixmap().toImage()
        expected_value = int((float(expected[0, 0]) - expected.min())
                             / (expected.max() - expected.min()) * 255)
        assert image.pixelColor(0, 0).red() == expected_value
    finally:
        _retire_overview(dialog, qtbot)


def test_real_wizard_conditions_reach_reference_plan_and_browser(source_planes, qtbot):
    source, paths = source_planes
    wizard = module._OverviewMetadataWizard(str(source), {"custom_regex": PATTERN})
    qtbot.addWidget(wizard)
    wizard._scan()
    qtbot.waitUntil(lambda: wizard.next.isEnabled(), timeout=10000)
    for row in range(wizard.conditions.rowCount()):
        if wizard.conditions.item(row, 1).text() == "A01":
            wizard.conditions.item(row, 2).setText("control")
    _choose(wizard.mode, "reference_condition")
    _choose(wizard.reference, "control")
    configuration = wizard.configuration()
    expected = configuration["normalization"]["conditions"]
    assert expected
    dialog = module._FieldOverview(settings=configuration)
    qtbot.addWidget(dialog)
    dialog.show()
    try:
        dialog._scan()
        qtbot.waitUntil(lambda: bool(dialog._fields) and dialog._normalization_ready
                        and not dialog._index_jobs.is_busy(), timeout=10000)
        assert dialog._conditions == expected
        assert dialog._normalization_plan["mode"] == "reference_condition"
        assert dialog._normalization_plan["reference"] == "control"
        expected_count = sum(field.well == "A01" and field.time_id == dialog.time_axis.currentData()
                             and field.z_id == dialog.z_axis.currentData() for field in dialog._fields)
        assert dialog._normalization_plan["sample_fields"] == expected_count
    finally:
        _retire_overview(dialog, qtbot)
        wizard._runner.shutdown(timeout_ms=3000)
        wizard.close()


def test_cancelled_pending_selection_does_not_start_another_decoder(overview, monkeypatch, qtbot):
    dialog = overview
    entered, release = threading.Event(), threading.Event()
    original = module._overview_pixels
    requested = []

    def read(field, channels, limits, edge=1200, **options):
        if edge == 256:
            requested.append(field.identity)
            entered.set()
            assert release.wait(10)
        return original(field, channels, limits, edge=edge, **options)

    monkeypatch.setattr(module, "_overview_pixels", read)
    dialog._well(dialog.wells.item(*dialog._well_positions["A01"]))
    dialog.fields.setCurrentRow(0)
    dialog._field(dialog.fields.item(0))
    try:
        qtbot.waitUntil(entered.is_set)
        dialog.fields.setCurrentRow(1)
        dialog._field(dialog.fields.item(1))
        dialog._cancel()
        release.set()
        qtbot.waitUntil(lambda: dialog._image_jobs.active_jobs() == 0, timeout=10000)
        assert len(requested) == 1
        assert dialog.view._item is None
    finally:
        release.set()


def test_readable_plate_cells_paint_real_native_icons_and_translucent_hover(source_planes, qtbot):
    from PySide6.QtCore import QRect
    from PySide6.QtGui import QColor, QImage, QPainter
    from PySide6.QtWidgets import QStyle, QStyleOptionViewItem

    source, paths = source_planes
    dialog = module._FieldOverview(settings={"source": str(source), "source_scope": "folder",
        "custom_regex": PATTERN, "channels": (1,), "geometry": (16, 24)})
    qtbot.addWidget(dialog, before_close_func=lambda widget: _retire_overview(widget, qtbot))
    dialog.show()
    dialog._scan()
    qtbot.waitUntil(lambda: bool(dialog._fields) and dialog._normalization_ready, timeout=10000)
    item = dialog.wells.item(*dialog._well_positions["A01"])
    qtbot.waitUntil(lambda: not item.icon().isNull(), timeout=10000)
    while min(dialog._plate_cell_extent) < 64:
        dialog._zoom_plate(1)
    assert dialog.wells.rowHeight(0) >= 64
    assert dialog.wells.columnWidth(0) >= 64
    assert item.text() in item.toolTip()
    assert dialog._well_representatives["A01"].caption in item.toolTip()
    index = dialog.wells.indexFromItem(item)
    option = QStyleOptionViewItem()
    option.initFrom(dialog.wells)
    option.widget = dialog.wells
    option.rect = QRect(0, 0, dialog.wells.columnWidth(0), dialog.wells.rowHeight(0))
    option.state = QStyle.State_Enabled
    delegate = dialog.wells.itemDelegate()

    def paint(state):
        option.state = state
        image = QImage(option.rect.size(), QImage.Format_ARGB32_Premultiplied)
        image.fill(QColor(17, 19, 23))
        painter = QPainter(image)
        try:
            delegate.paint(painter, option, index)
        finally:
            painter.end()
        return image

    normal = paint(QStyle.State_Enabled)
    height = normal.height() - option.fontMetrics.height() - 12
    values = [normal.pixelColor(x, y).red() for x in range(5, normal.width() - 5)
              for y in range(5, height)]
    assert height > 16
    assert max(values) - min(values) > 120
    hover = paint(QStyle.State_Enabled | QStyle.State_MouseOver)
    before, after = normal.pixelColor(2, 2), hover.pixelColor(2, 2)
    assert after != before
    assert after.blue() - after.red() > before.blue() - before.red()
    assert after != QColor(35, 130, 245)


def test_incompatible_plate_geometry_refuses_without_an_expanded_grid_or_gui_error(
        source_planes, qtbot, monkeypatch):
    source, paths = source_planes
    dialog = module._FieldOverview(settings={"source": str(source), "source_scope": "folder",
        "custom_regex": PATTERN, "channels": (1,), "geometry": (1, 1)})
    qtbot.addWidget(dialog, before_close_func=lambda widget: _retire_overview(widget, qtbot))
    dialog.show()
    with qtbot.captureExceptions() as exceptions:
        dialog._scan()
        qtbot.waitUntil(lambda: bool(dialog._fields) and not dialog._index_jobs.is_busy(), timeout=10000)
        assert not dialog._geometry_valid
        assert not dialog._normalization_ready
        assert dialog.wells.rowCount() == dialog.wells.columnCount() == 0
        assert dialog.fields.count() == 0
        assert dialog.view._item is None
        assert dialog.status.text() == (
            "The selected geometry does not contain the indexed wells (2 rows, 2 columns required).")
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k:
                            pytest.fail("an invalid plate geometry must not export"))
        dialog._export()
    assert exceptions == []


@pytest.mark.parametrize("dimensions", [(1280, 820), (760, 620)])
def test_initial_whole_plate_fits_and_opening_is_separate_from_selection(source_planes, qtbot, dimensions):
    from PySide6.QtCore import QSize
    from spacr.qt.gui_scale import _resize_actual

    source, _ = source_planes
    dialog = module._FieldOverview(settings={"source": str(source), "source_scope": "folder",
        "custom_regex": PATTERN, "channels": (1,), "geometry": (16, 24)})
    qtbot.addWidget(dialog, before_close_func=lambda widget: _retire_overview(widget, qtbot))
    _resize_actual(dialog, QSize(*dimensions))
    dialog.show()
    dialog._scan()
    qtbot.waitUntil(lambda: bool(dialog._fields) and dialog._normalization_ready, timeout=10000)
    qtbot.waitUntil(lambda: all(dialog.wells.item(*position).data(Qt.UserRole + 1) is not None
        for position in dialog._well_positions.values()), timeout=10000)
    assert dialog.size() == QSize(*dimensions)
    assert not dialog._controls_scroll.isVisible()
    assert all(control.isVisible() for control in (dialog.time_axis, dialog.z_axis, dialog.fov_axis))
    assert not dialog.fields.isVisible() and not dialog.view.isVisible()
    viewport = dialog.wells.viewport().rect()
    for row, column in ((0, 0), (0, 23), (15, 0), (15, 23)):
        assert viewport.contains(dialog.wells.visualRect(dialog.wells.model().index(row, column)))
    assert dialog.wells.horizontalScrollBar().maximum() == 0
    assert dialog.wells.verticalScrollBar().maximum() == 0
    identities = {well: dialog.wells.item(*position).data(Qt.UserRole)
                  for well, position in dialog._well_positions.items()}
    item = dialog.wells.item(*dialog._well_positions["A01"])
    dialog.wells.setCurrentItem(item)
    assert not dialog.fields.isVisible()
    dialog._open_selected_well()
    assert dialog.fields.isVisible() and dialog.view.isVisible()
    assert dialog._scope[1] == "A01"
    dialog._back()
    qtbot.waitUntil(lambda: dialog.wells.verticalScrollBar().maximum() == 0)
    assert not dialog.fields.isVisible() and not dialog.view.isVisible()
    assert viewport.contains(dialog.wells.visualRect(dialog.wells.model().index(15, 23)))
    before = dialog._plate_cell_extent
    dialog._zoom_plate(1)
    assert dialog._plate_cell_extent[0] > before[0]
    assert identities == {well: dialog.wells.item(*position).data(Qt.UserRole)
                          for well, position in dialog._well_positions.items()}
    dialog._zoom_plate()
    assert dialog.wells.verticalScrollBar().maximum() == 0


def test_well_mean_context_uses_only_actual_view_representatives_and_axes(tmp_path, qtbot, monkeypatch):
    source = tmp_path / "well-means"
    source.mkdir()
    plane = np.arange(1200, dtype=np.uint16).reshape(30, 40)
    for plate in ("p1", "p2"):
        for well in ("A01", "B02"):
            for field in (1, 2, 10):
                for time in (1, 2):
                    tifffile.imwrite(source / f"{plate}-{well}-f{field}-t{time}-z1-c1.tif",
                        plane + field * 10 + time * 100 + (500 if well == "B02" else 0))
    actual_prepare = module._overview_prepare_normalization
    calls = []

    def prepare(fields, channels, options, read_planes, cancelled=None):
        calls.append((tuple(fields), dict(options)))
        return actual_prepare(fields, channels, options, read_planes, cancelled)

    monkeypatch.setattr(module, "_overview_prepare_normalization", prepare)
    dialog = module._FieldOverview(settings={"source": str(source), "source_scope": "folder",
        "custom_regex": PATTERN, "channels": (1,), "geometry": (2, 3), "representatives": 1,
        "normalization": {"mode": "reference_average", "reference_kind": "reference_well",
                          "reference": ("p1", "A01"), "percentiles": (0., 100.)}})
    qtbot.addWidget(dialog, before_close_func=lambda widget: _retire_overview(widget, qtbot))
    dialog.show()
    dialog._scan()
    qtbot.waitUntil(lambda: dialog._normalization_ready and dialog._normalization_context is not None,
                    timeout=10000)
    fields, options = calls[-1]
    assert {(f.plate, f.well, f.field_id, f.time_id, f.z_id) for f in fields} == {
        ("p1", "A01", "1", "1", "1"), ("p1", "B02", "1", "1", "1")}
    assert dialog._display_kwargs()["normalization_context"] is dialog._normalization_context
    previous = dialog._normalization_context
    _choose(dialog.fov_axis, "10")
    _choose(dialog.time_axis, "2")
    qtbot.waitUntil(lambda: dialog._normalization_ready
        and dialog._normalization_context is not previous and not dialog._index_jobs.is_busy(), timeout=10000)
    fields, options = calls[-1]
    assert {(f.plate, f.well, f.field_id, f.time_id, f.z_id) for f in fields} == {
        ("p1", "A01", "10", "2", "1"), ("p1", "B02", "10", "2", "1")}
    provenance = dialog._normalization_context.provenance()
    assert provenance["selection"]["field_id"] == "10"
    assert provenance["selection"]["filename_time"] == "2"
    assert provenance["selection"]["representatives"] == 1
    assert provenance["selection"]["generation"] == dialog._display_generation
    assert provenance["scope_plate"] == "p1"
    assert all(f.plate == "p1" and f.field_id == "10" for f in dialog._well_fields["A01"])
