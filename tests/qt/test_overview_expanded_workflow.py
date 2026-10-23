"""Expanded overview contracts retain the published wizard and source readers."""
import json
import sqlite3
import threading

import numpy as np
import pytest
import tifffile

from spacr.qt.widgets import qc_field_browser as browser


def _images(tmp_path):
    for plate in ("plate1", "plate2"):
        for well in ("A01", "B02"):
            for field in (1, 2):
                tifffile.imwrite(tmp_path / f"{plate}_{well}_F{field}.tif",
                    np.arange(120, dtype=np.uint16).reshape(10, 12) + field)
    return browser._overview_index(tmp_path)


def _browser(tmp_path, qtbot, fields=None, channels=(0,)):
    dialog = browser._FieldOverview(settings={"source": str(tmp_path), "channels": channels})
    def retire(widget):
        widget._cancel()
        widget._visual_timer.stop()
        runners = (widget._index_jobs, widget._image_jobs, widget._export_jobs,
            widget._metadata_jobs, widget._measurement_jobs, widget._native_jobs,
            widget._thumbnail_jobs, widget._montage_jobs, widget._field_thumbnail_jobs)
        qtbot.waitUntil(lambda: all(runner.active_jobs() == 0 for runner in runners))
        for runner in runners:
            runner.shutdown(timeout_ms=3000)
        from spacr.qt.bridge import prune_parked_threads
        qtbot.waitUntil(lambda: prune_parked_threads() == 0)
    qtbot.addWidget(dialog, before_close_func=retire)
    dialog._indexed(_images(tmp_path) if fields is None else fields)
    qtbot.waitUntil(lambda: dialog._normalization_ready)
    return dialog


def test_well_only_regex_groups_channels_and_live_preview_requires_valid_wells(tmp_path, qtbot):
    for channel in (1, 2):
        tifffile.imwrite(tmp_path / f"A01_alpha_C{channel}.tif", np.zeros((5, 7), np.uint16))
    wizard = browser._OverviewMetadataWizard(str(tmp_path))
    qtbot.addWidget(wizard)
    wizard._startup_timer.stop()
    wizard.regex.setText(r"(?P<wellID>A01)_alpha_C(?P<chanID>\d+)")
    wizard._custom_regex_changed()
    qtbot.waitUntil(lambda: wizard.next.isEnabled())
    field, = wizard._fields
    assert field.well == "A01" and not field.unresolved and len(field.files) == 2
    assert wizard.preview.item(0, 2).text() == "A01"
    wizard.regex.setText(r"(?P<fieldID>alpha)")
    wizard._custom_regex_changed()
    assert not wizard.next.isEnabled()
    qtbot.waitUntil(lambda: "wellID" in wizard.status.text())
    wizard.reject()


def test_missing_well_cannot_be_accepted_as_invented_coverslip(tmp_path, qtbot):
    tifffile.imwrite(tmp_path / "unassigned.tif", np.zeros((5, 7), np.uint16))
    wizard = browser._OverviewMetadataWizard(str(tmp_path))
    qtbot.addWidget(wizard)
    qtbot.waitUntil(lambda: wizard._info is not None)
    assert not wizard.next.isEnabled()
    assert wizard._fields[0].well == "" and wizard._fields[0].unresolved
    with pytest.raises(ValueError):
        wizard.configuration()
    wizard.reject()


def test_wizard_detects_1536_and_refuses_smaller_override(tmp_path, qtbot):
    tifffile.imwrite(tmp_path / "plate1_AF48_F1.tif", np.zeros((5, 7), np.uint16))
    wizard = browser._OverviewMetadataWizard(str(tmp_path))
    qtbot.addWidget(wizard)
    qtbot.waitUntil(lambda: wizard.next.isEnabled())
    assert wizard.plate_geometry.currentData() == (32, 48)
    wizard.plate_geometry.setCurrentIndex(next(index for index in range(wizard.plate_geometry.count()) if tuple(wizard.plate_geometry.itemData(index)) == (8, 12)))
    with pytest.raises(ValueError, match="geometry"):
        wizard.configuration()
    wizard.reject()


def test_arbitrary_metadata_persists_across_plates_and_reloads_without_losing_condition(tmp_path, qtbot):
    dialog = _browser(tmp_path, qtbot)
    dialog.wells.item(0, 0).setSelected(True)
    dialog.plates.setCurrentIndex(dialog.plates.findData("plate2"))
    dialog.wells.item(0, 0).setSelected(True)
    assert dialog._selected_wells == {("plate1", "A01"), ("plate2", "A01")}
    dialog.annotation_scope.setCurrentIndex(dialog.annotation_scope.findData("wells"))
    for key, value in (("condition", "drug"), ("cell_type", "U2OS"), ("dose", "10 nM")):
        index = dialog.annotation_key.findData(key)
        if index < 0:
            dialog.annotation_key.setEditText(key)
        else:
            dialog.annotation_key.setCurrentIndex(index)
        dialog.annotation_text.setText(value)
        dialog._annotate()
    qtbot.waitUntil(lambda: not dialog._annotation_write_running)
    payload = json.loads((tmp_path / "results" / "overview_annotations.json").read_text())
    assert len(payload["wells"]) == 2
    assert all(record["metadata"] == {"condition": "drug", "cell_type": "U2OS", "dose": "10 nM"} for record in payload["wells"])
    assert dialog.normalization_scope.findData("dose") >= 0
    dialog.use_selection.setChecked(True)
    assert len(dialog._export_selection()) == 4
    dialog.close()
    other = _browser(tmp_path, qtbot, browser._overview_index(tmp_path))
    qtbot.waitUntil(lambda: other._conditions.get(("plate1", "A01")) == "drug")
    assert other._annotations[("plate2", "A01")]["dose"] == "10 nM"
    other.close()


def test_final_metadata_edit_survives_closing_during_first_write(tmp_path, qtbot, monkeypatch):
    from spacr import tabular
    entered, release, saved = threading.Event(), threading.Event(), threading.Event()
    original = tabular._publish
    count = []
    def publish(path, writer):
        if str(path).endswith("overview_annotations.json"):
            count.append(path)
            if len(count) == 1:
                entered.set()
                assert release.wait(5)
        result = original(path, writer)
        if len(count) == 2:
            saved.set()
        return result
    dialog = _browser(tmp_path, qtbot)
    qtbot.waitUntil(lambda: not dialog._metadata_jobs.is_busy())
    monkeypatch.setattr(tabular, "_publish", publish)
    dialog.annotation_key.setEditText("dose")
    dialog.annotation_text.setText("first")
    dialog._annotate()
    qtbot.waitUntil(entered.is_set)
    dialog.annotation_text.setText("final")
    dialog._annotate()
    try:
        dialog.close()
        release.set()
        qtbot.waitUntil(saved.is_set)
        from spacr.qt.bridge import prune_parked_threads
        qtbot.waitUntil(lambda: prune_parked_threads() == 0)
        payload = json.loads((tmp_path / "results" / "overview_annotations.json").read_text())
        assert all(record["metadata"]["dose"] == "final" for record in payload["wells"])
        assert len(count) == 2
    finally:
        release.set()


def test_independent_channel_editor_preserves_global_and_other_channel_settings(tmp_path, qtbot):
    np.save(tmp_path / "plate1_A01_F1.npy", np.ones((5, 7, 3)))
    dialog = _browser(tmp_path, qtbot, browser._overview_index(tmp_path), (0, 1, 2))
    original = dialog._normalization_options()
    dialog.norm_channel.setCurrentIndex(dialog.norm_channel.findData(1))
    dialog.normalization_method.setCurrentIndex(dialog.normalization_method.findData("manual"))
    dialog.manual_low.setValue(0)
    dialog.manual_high.setValue(100)
    dialog.background.setValue(9)
    assert dialog._normalization_options() == original
    dialog._apply_channel_normalization()
    options = dialog._normalization_options()
    assert options["method"] == "percentile" and options["baseline"] == 0
    assert browser._overview_channel_normalization(options, 0)["baseline"] == 0
    assert browser._overview_channel_normalization(options, 2)["method"] == "percentile"
    assert browser._overview_channel_normalization(options, 1)["baseline"] == 9
    assert browser._overview_channel_normalization(options, 1)["method"] == "manual"
    dialog.norm_channel.setCurrentIndex(dialog.norm_channel.findData(2))
    assert dialog.normalization_method.currentData() == "percentile" and dialog.background.value() == 0
    dialog.norm_channel.setCurrentIndex(dialog.norm_channel.findData(1))
    assert dialog.normalization_method.currentData() == "manual" and dialog.background.value() == 9
    dialog._cancel()
    qtbot.waitUntil(lambda: dialog._index_jobs.active_jobs() == 0)
    dialog._index_jobs.shutdown(timeout_ms=3000)
    dialog.close()


@pytest.mark.parametrize("limits,mode", ((None, None), ((0, 100), None), ((0, 100), "image"), ((0, 100), "advanced")))
def test_native_crop_matches_whole_field_intensity_scale(tmp_path, limits, mode):
    values = np.arange(24 * 30, dtype=np.float32).reshape(24, 30)
    values[0, 0] = 10000
    np.save(tmp_path / "plate1_A01_F1.npy", values)
    field, = browser._overview_index(tmp_path)
    options = {}
    if mode == "image":
        options["normalization"] = browser._overview_normalization_plan((), (0,), dict(mode="image", percentiles=limits), None)
    elif mode == "advanced":
        options["advanced_normalization"] = browser._overview_advanced_normalization((field,), (0,), dict(method="minmax", scope="plate"))
    whole = browser._overview_pixels(field, (0,), limits, edge=None, **options)
    crop = browser._overview_pixels(field, (0,), limits, edge=None, region=(5, 12, 7, 19), **options)
    np.testing.assert_array_equal(crop, whole[5:12, 7:19])


def test_background_database_action_uses_raw_pixels_and_arbitrary_metadata(tmp_path, qtbot):
    dialog = _browser(tmp_path, qtbot)
    dialog._annotations[("plate1", "A01")] = {"dose": "10 nM"}
    dialog.workers.setValue(1)
    dialog._measure()
    qtbot.waitUntil(lambda: not dialog._measurement_jobs.is_busy())
    with sqlite3.connect(tmp_path / "measurements" / "measurements.db") as database:
        rows = database.execute("select annotations, channel_0_intensity_mean from fov where well='A01'").fetchall()
    assert len(rows) == 2
    assert all(json.loads(metadata)["dose"] == "10 nM" for metadata, mean in rows)
    assert sorted(mean for metadata, mean in rows) == [60.5, 61.5]
    dialog.close()


def test_fourth_channel_reaches_native_pixels_beyond_the_4096_preview_limit(tmp_path, qtbot):
    raw = np.zeros((1024, 5120, 4), np.uint16)
    raw[..., 3] = np.arange(1024 * 5120, dtype=np.uint16).reshape(1024, 5120)
    np.save(tmp_path / "plate1_A01_F1.npy", raw)
    dialog = _browser(tmp_path, qtbot, browser._overview_index(tmp_path), (0, 1, 2, 3))
    dialog.show()
    dialog._well(dialog.wells.item(0, 0))
    dialog.fields.setCurrentRow(0)
    dialog._field(dialog.fields.item(0))
    qtbot.waitUntil(lambda: dialog.view._item is not None and dialog.view._item.pixmap().width() == 1024
        and dialog.view.viewport().width() > 0 and not dialog._image_jobs.is_busy())
    dialog.view.zoom_by(32)
    dialog._request_detail()
    qtbot.waitUntil(lambda: "toggle up to three channels" in dialog.status.text())
    assert dialog._native_key is None
    dialog.options_button.click()
    for channel in (0, 1, 2):
        control = dialog._channel_buttons[channel]
        dialog._controls_scroll.ensureWidgetVisible(control)
        qtbot.mouseClick(control, browser.Qt.LeftButton)
    dialog.options_button.click()
    qtbot.waitUntil(lambda: dialog._choices()[0] == (3,) and dialog.view._item is not None
        and dialog.view._item.pixmap().width() == 1024 and not dialog._image_jobs.is_busy())
    dialog.view.zoom_by(32)
    dialog._request_detail()
    qtbot.waitUntil(lambda: dialog._native_key is not None and not dialog._image_jobs.is_busy(), timeout=10000)
    transform = dialog.view.transform()
    y0, y1, x0, x1 = dialog._native_key[2]
    assert 0 < x1 - x0 <= 2048 and 0 < y1 - y0 <= 2048
    assert dialog._native_item.pixmap().width() == x1 - x0
    assert dialog._native_item.pixmap().height() == y1 - y0
    assert dialog.view.transform() == transform
    assert dialog._source_shapes[dialog.fields.item(0).data(browser.Qt.UserRole).identity] == (1024, 5120)
    dialog.close()


def test_pending_metadata_load_cannot_overwrite_new_edits_or_restore_deletions(tmp_path, qtbot, monkeypatch):
    dialog = _browser(tmp_path, qtbot)
    qtbot.waitUntil(lambda: not dialog._metadata_jobs.is_busy())
    path = tmp_path / "results" / "overview_annotations.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"wells": [{"plate": "plate1", "well": "A01", "metadata": {"dose": "old", "drug": "deleted"}}]}))
    started, release = threading.Event(), threading.Event()
    original = type(path).read_text
    def delayed_read(current, *args, **kwargs):
        value = original(current, *args, **kwargs)
        if current == path:
            started.set()
            assert release.wait(5)
        return value
    monkeypatch.setattr(type(path), "read_text", delayed_read)
    dialog._restore_metadata()
    qtbot.waitUntil(started.is_set)
    dialog._annotations[("plate1", "A01")] = {"dose": "new"}
    dialog._persist_metadata()
    release.set()
    qtbot.waitUntil(lambda: not dialog._metadata_jobs.is_busy())
    assert dialog._annotation_snapshot()[("plate1", "A01")] == {"dose": "new"}
    assert json.loads(original(path))["wells"][0]["metadata"] == {"dose": "new"}
    dialog.close()


def test_advanced_controls_learn_shared_bounds_toggle_channels_and_group_custom_metadata(tmp_path, qtbot):
    raw = np.stack((np.arange(120).reshape(10, 12), np.arange(120).reshape(10, 12) + 50), axis=-1)
    for well, offset in (("A01", 0), ("B02", 100)):
        np.save(tmp_path / f"plate1_{well}_F1.npy", raw + offset)
    dialog = _browser(tmp_path, qtbot, browser._overview_index(tmp_path), (0, 1))
    dialog.normalization_scope.setCurrentIndex(dialog.normalization_scope.findData("plate"))
    dialog.normalization_method.setCurrentIndex(dialog.normalization_method.findData("minmax"))
    dialog.advanced.setChecked(True)
    qtbot.waitUntil(lambda: dialog._normalization_ready and dialog._advanced_model is not None)
    model = dialog._advanced_model
    assert model["provenance"]["sampling_axes"] == dict(time_index=0, z_mode="plane", z_index=0)
    assert model["provenance"]["sampled_fields"] == 2
    limits = next(iter(model["bounds"].values()))
    assert limits["0"]["low"] == 0 and limits["0"]["high"] == 219
    dialog.show()
    dialog.options_button.click()
    control = dialog._channel_buttons[1]
    dialog._controls_scroll.ensureWidgetVisible(control)
    qtbot.waitUntil(lambda: control.isVisible() and dialog._controls_scroll.viewport().rect().contains(
        control.rect().translated(control.mapTo(dialog._controls_scroll.viewport(), control.rect().topLeft()))))
    qtbot.mouseClick(control, browser.Qt.LeftButton)
    qtbot.waitUntil(lambda: dialog._normalization_ready and dialog._advanced_model is not model)
    assert dialog._choices()[0] == (0,)
    dialog._apply_channel_normalization()
    dialog._clear_channel_normalization()
    qtbot.waitUntil(lambda: dialog._normalization_ready)
    dialog.annotation_scope.setCurrentIndex(dialog.annotation_scope.findData("plate"))
    dialog.annotation_key.setEditText("batch")
    dialog.annotation_text.setText("replicate1")
    dialog._annotate()
    qtbot.waitUntil(lambda: dialog._normalization_ready)
    dialog.grouping.setCurrentIndex(dialog.grouping.findData("batch"))
    dialog.count.setValue(1)
    assert len(dialog._sample_fields(dialog._fields)) == 1
    dialog.random.setChecked(True)
    dialog.seed.setValue(17)
    assert dialog._sample_fields(dialog._fields) == dialog._sample_fields(tuple(reversed(dialog._fields)))
    dialog.normalization_scope.setCurrentIndex(dialog.normalization_scope.findData("image"))
    qtbot.waitUntil(lambda: dialog._normalization_ready)
    assert dialog._advanced_model["provenance"]["sampling_axes"] == "per_selected_image"
    dialog.close()


@pytest.mark.parametrize("records", (
    [{"plate": "foreign", "well": "A01", "metadata": {"dose": "bad"}}],
    [{"plate": "plate1", "well": "A01", "metadata": []}],
    [{"plate": "plate1", "well": "A01", "metadata": {}}] * 2))
def test_saved_metadata_rejects_foreign_duplicate_and_nonmapping_records(tmp_path, qtbot, records):
    path = tmp_path / "results" / "overview_annotations.json"
    path.parent.mkdir()
    path.write_text(json.dumps(dict(wells=records)))
    dialog = _browser(tmp_path, qtbot)
    qtbot.waitUntil(lambda: not dialog._metadata_jobs.is_busy())
    assert "outside this source" in dialog.status.text()
    assert not dialog._annotation_snapshot()
    dialog.close()


def test_failed_metadata_write_preserves_previous_file_and_allows_retry(tmp_path, qtbot, monkeypatch):
    from spacr import tabular
    dialog = _browser(tmp_path, qtbot)
    qtbot.waitUntil(lambda: not dialog._metadata_jobs.is_busy())
    path = tmp_path / "results" / "overview_annotations.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text('{"wells": []}')
    original = tabular._publish
    def fail(destination, write):
        if destination == str(path):
            raise OSError("test disk unavailable")
        return original(destination, write)
    monkeypatch.setattr(tabular, "_publish", fail)
    dialog._annotations[("plate1", "A01")] = {"dose": "new"}
    dialog._persist_metadata()
    qtbot.waitUntil(lambda: not dialog._metadata_jobs.is_busy())
    assert not dialog._annotation_write_running
    assert "test disk unavailable" in dialog.status.text()
    assert json.loads(path.read_text()) == {"wells": []}
    monkeypatch.setattr(tabular, "_publish", original)
    dialog._persist_metadata()
    qtbot.waitUntil(lambda: not dialog._metadata_jobs.is_busy())
    assert json.loads(path.read_text())["wells"][0]["metadata"] == {"dose": "new"}
    dialog.close()


def test_measurement_action_reports_invalid_selection_failure_busy_and_cancellation(tmp_path, qtbot, monkeypatch):
    dialog = _browser(tmp_path, qtbot)
    source = dialog._metadata_source
    dialog._metadata_source = ""
    dialog._measure()
    assert "folder" in dialog.status.text()
    dialog._metadata_source = source
    dialog.channels.setText("invalid")
    dialog._measure()
    assert "channel" in dialog.status.text().lower()
    dialog.channels.setText("0")
    dialog.export_scope.setCurrentIndex(dialog.export_scope.findData("selection"))
    dialog._measure()
    assert "Select fields" in dialog.status.text()
    dialog.wells.item(0, 0).setSelected(True)
    field = next(field for field in dialog._fields if field.plate == "plate1" and field.well == "A01")
    original = browser._overview_raw_planes
    entered, release = threading.Event(), threading.Event()
    def paused(target, channels, **options):
        values = tuple(original(target, channels, **options))
        if options.get("edge") is None and target.identity == field.identity:
            entered.set()
            assert release.wait(5)
        yield from values
    monkeypatch.setattr(browser, "_overview_raw_planes", paused)
    dialog.workers.setValue(1)
    dialog._measure()
    qtbot.waitUntil(entered.is_set)
    dialog._measure()
    assert "current measurement run" in dialog.status.text()
    dialog._cancel()
    release.set()
    qtbot.waitUntil(lambda: not dialog._measurement_jobs.is_busy())
    assert "Measurements cancelled" in dialog.status.text()
    assert not (tmp_path / "measurements" / "measurements.db").exists()
    monkeypatch.setattr(browser, "_overview_raw_planes", original)
    from dataclasses import replace
    broken = replace(field, files=(("", str(tmp_path / "missing.npy")),))
    dialog._fields = (broken,)
    dialog._visible = (broken,)
    dialog.export_scope.setCurrentIndex(dialog.export_scope.findData("view"))
    dialog._measure()
    qtbot.waitUntil(lambda: not dialog._measurement_jobs.is_busy())
    assert "Measurements were not published" in dialog.status.text()
    dialog.close()


@pytest.mark.parametrize("suffix", ("npy", "tif"))
def test_large_mapped_plane_allows_only_explicit_bounded_original_region(tmp_path, suffix):
    path = tmp_path / f"plate1_A01_F1.{suffix}"
    if suffix == "npy":
        image = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint16, shape=(8193, 8192))
    else:
        image = tifffile.memmap(path, shape=(8193, 8192), dtype=np.uint16, photometric="minisblack")
    image[4000:4008, 5000:5012] = np.arange(96).reshape(8, 12) + 100
    image.flush()
    del image
    field, = browser._overview_index(tmp_path)
    with pytest.raises(ValueError, match="128 MiB"):
        tuple(browser._overview_raw_planes(field, (0,)))
    plane = next(browser._overview_raw_planes(field, (0,), region=(4000, 4008, 5000, 5012)))[2]
    assert plane.shape == (8, 12) and isinstance(plane, np.memmap)
    np.testing.assert_array_equal(plane, np.arange(96).reshape(8, 12) + 100)
    pixels = browser._overview_pixels(field, (0,), None, edge=None,
        region=(4000, 4008, 5000, 5012), advanced_normalization=dict(method="manual", low=100, high=195))
    assert pixels.shape == (8, 12, 3)
    assert pixels[0, 0].tolist() == [0, 0, 0]
    assert pixels[-1, -1].tolist() == [255, 255, 255]


def test_actual_plate_click_and_drag_preserve_selection_and_open_only_plain_clicks(tmp_path, qtbot):
    from PySide6.QtCore import QEvent, QPointF
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication
    dialog = _browser(tmp_path, qtbot)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.wells.viewport().width() > 0)
    viewport = dialog.wells.viewport()
    start = dialog.wells.visualItemRect(dialog.wells.item(0, 0)).center()
    end = dialog.wells.visualItemRect(dialog.wells.item(1, 1)).center()
    qtbot.mousePress(viewport, browser.Qt.LeftButton, pos=start)
    move = QMouseEvent(QEvent.MouseMove, QPointF(end), QPointF(viewport.mapToGlobal(end)),
        browser.Qt.NoButton, browser.Qt.LeftButton, browser.Qt.NoModifier)
    QApplication.sendEvent(viewport, move)
    qtbot.mouseRelease(viewport, browser.Qt.LeftButton, pos=end)
    assert dialog._drag_selected and dialog._scope[1] is None
    assert ("plate1", "A01") in dialog._selected_wells
    qtbot.mouseClick(viewport, browser.Qt.LeftButton, pos=start)
    assert dialog._scope == ("plate1", "A01") and dialog.fields.isVisible()
    dialog.close()


def test_native_crop_rejects_old_field_and_old_camera_then_replaces_visible_detail(tmp_path, qtbot, monkeypatch):
    for field in (1, 2):
        np.save(tmp_path / f"plate1_A01_F{field}.npy", np.arange(1024 * 5120, dtype=np.uint16).reshape(1024, 5120) + field)
    dialog = _browser(tmp_path, qtbot, browser._overview_index(tmp_path))
    dialog.show()
    dialog._well(dialog.wells.item(0, 0))
    dialog.fields.setCurrentRow(0)
    dialog._field(dialog.fields.item(0))
    qtbot.waitUntil(lambda: dialog.view._item is not None and dialog.view._item.pixmap().width() == 1024
        and not dialog._image_jobs.is_busy())
    original = browser._overview_pixels
    entered, release = threading.Event(), threading.Event()
    pause = [True]
    def reader(*args, **kwargs):
        pixels = original(*args, **kwargs)
        if kwargs.get("region") is not None and pause[0]:
            entered.set()
            assert release.wait(5)
        return pixels
    monkeypatch.setattr(browser, "_overview_pixels", reader)
    dialog.view.zoom_by(32)
    dialog._request_detail()
    qtbot.waitUntil(entered.is_set)
    dialog._load_native_detail()
    assert dialog._native_pending
    dialog.fields.setCurrentRow(1)
    dialog._field(dialog.fields.item(1))
    qtbot.waitUntil(lambda: not dialog._image_jobs.is_busy())
    release.set()
    qtbot.waitUntil(lambda: not dialog._native_jobs.is_busy())
    assert dialog._native_key is None
    dialog.view.zoom_by(32)
    entered.clear()
    release.clear()
    dialog._request_detail()
    qtbot.waitUntil(entered.is_set)
    horizontal = dialog.view.horizontalScrollBar()
    horizontal.setValue(horizontal.value() + 30)
    dialog._load_native_detail()
    pause[0] = False
    release.set()
    qtbot.waitUntil(lambda: dialog._native_key is not None and not dialog._native_jobs.is_busy())
    first = dialog._native_key
    assert first[0] == dialog.fields.item(1).data(browser.Qt.UserRole).identity
    horizontal.setValue(horizontal.value() + 40)
    dialog._request_native_detail()
    qtbot.waitUntil(lambda: dialog._native_key != first and not dialog._native_jobs.is_busy())
    assert dialog._native_key[-1][1] == horizontal.value()
    submitted = dialog._native_jobs._next_id
    dialog._load_native_detail()
    assert dialog._native_jobs._next_id == submitted
    horizontal.setValue(horizontal.minimum())
    dialog.view.verticalScrollBar().setValue(dialog.view.verticalScrollBar().minimum())
    dialog._load_native_detail()
    assert dialog._native_jobs._next_id == submitted
    dialog.view.fit()
    dialog.view.zoom_by(.1)
    dialog._load_native_detail()
    assert dialog._native_jobs._next_id == submitted
    dialog._back()
    dialog._load_native_detail()
    dialog._request_detail()
    assert dialog.view._item is None
    dialog.close()


def test_empty_source_controls_and_export_guards_never_start_image_or_output_jobs(tmp_path, qtbot):
    dialog = _browser(tmp_path, qtbot, ())
    dialog.advanced.setChecked(True)
    dialog._prepare_display()
    dialog._request_detail()
    dialog._export()
    assert "Select fields" in dialog.status.text()
    dialog._metadata_source = ""
    dialog._persist_metadata()
    assert not dialog._annotation_write_running
    assert not dialog._image_jobs.is_busy() and not dialog._export_jobs.is_busy()
    dialog.close()


def test_loaded_source_requires_destination_identity_before_export_and_well_scope_requires_open_well(tmp_path, qtbot):
    dialog = _browser(tmp_path, qtbot)
    dialog.export_scope.setCurrentIndex(dialog.export_scope.findData("well"))
    assert dialog._export_selection() == ()
    dialog.export_scope.setCurrentIndex(dialog.export_scope.findData("plate"))
    dialog._metadata_source = ""
    dialog._export()
    assert "image folder" in dialog.status.text()
    assert not dialog._export_jobs.is_busy()
    dialog.close()


def test_large_multi_channel_and_projection_allocations_obey_decode_budget(tmp_path):
    path = tmp_path / "plate1_A01_F1.npy"
    image = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint16, shape=(6000, 6000, 3))
    image[3:7, 5:11] = 100
    image.flush()
    del image
    field, = browser._overview_index(tmp_path)
    with pytest.raises(ValueError, match="128 MiB"):
        browser._overview_pixels(field, (0, 1, 2), edge=None)
    pixels = browser._overview_pixels(field, (0, 1, 2), edge=None, region=(3, 7, 5, 11),
        advanced_normalization=dict(method="manual", low=0, high=100))
    assert pixels.shape == (4, 6, 3) and (pixels == 255).all()
    del field
    image = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint16, shape=(2, 8193, 8192, 1))
    del image
    field, = browser._overview_index(tmp_path)
    with pytest.raises(ValueError, match="128 MiB"):
        tuple(browser._overview_raw_planes(field, (0,), z_mode="mean"))
    plane = next(browser._overview_raw_planes(field, (0,), z_mode="mean", region=(3, 7, 5, 11)))[2]
    assert plane.shape == (4, 6) and not plane.any()


def test_well_only_regex_keeps_independent_channel_named_folders_without_explicit_channel_capture(tmp_path):
    for channel in (1, 2):
        folder = tmp_path / "experiment" / "A01" / f"channel{channel}"
        folder.mkdir(parents=True)
        tifffile.imwrite(folder / "A01_sample.tif", np.full((4, 6), channel, np.uint16))
    fields = browser._overview_index(tmp_path, r"(?P<wellID>A01)_sample")
    assert len(fields) == 2 and len({field.identity for field in fields}) == 2
    assert all(field.plate == str(tmp_path / "experiment") and field.well == "A01" for field in fields)
    assert sorted(next(browser._overview_raw_planes(field, (0,)))[2].mean() for field in fields) == [1, 2]


def test_folder_channels_without_field_capture_coalesce_only_recognized_acquisition_channels(tmp_path):
    for channel in (1, 2):
        folder = tmp_path / "plate1" / "A01" / f"channel{channel}"
        folder.mkdir(parents=True)
        tifffile.imwrite(folder / f"ch{channel}.tif", np.full((4, 6), channel, np.uint16))
    field, = browser._overview_index(tmp_path)
    assert field.well == "A01" and field.plate == "plate1" and not field.unresolved
    assert [key for key, path in field.files] == ["1", "2"]
    assert [plane.mean() for channel, path, plane in browser._overview_raw_planes(field, (1, 2))] == [1, 2]


def test_actual_browser_load_rejects_missing_well_and_accepts_corrected_source(tmp_path, qtbot):
    from PySide6.QtWidgets import QPushButton
    invalid = tmp_path / "invalid"
    invalid.mkdir()
    tifffile.imwrite(invalid / "unassigned.tif", np.ones((4, 6), np.uint16))
    dialog = _browser(tmp_path, qtbot, ())
    dialog.source.setText(str(invalid))
    load = next(button for button in dialog.findChildren(QPushButton) if button.text() == browser.tr("Load overview"))
    load.click()
    qtbot.waitUntil(lambda: not dialog._index_jobs.is_busy())
    assert not dialog._fields and "valid wellID for every image" in dialog.status.text()
    assert dialog.view._item is None and dialog.fields.count() == 0
    valid = tmp_path / "valid"
    valid.mkdir()
    tifffile.imwrite(valid / "plate1_A01_F1.tif", np.ones((4, 6), np.uint16))
    dialog.source.setText(str(valid))
    load.click()
    qtbot.waitUntil(lambda: dialog._normalization_ready and not dialog._index_jobs.is_busy())
    assert len(dialog._fields) == 1 and dialog._fields[0].well == "A01"
    dialog.close()


def test_condition_csv_refreshes_metadata_grouping_and_advanced_bounds(tmp_path, qtbot, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from spacr import tabular
    import pandas as pd
    dialog = _browser(tmp_path, qtbot)
    dialog.advanced.setChecked(True)
    qtbot.waitUntil(lambda: dialog._normalization_ready)
    path = tmp_path / "conditions.csv"
    tabular.write_table(pd.DataFrame([dict(plate="plate1", well="A01", condition="treated")]),
        str(path), canonicalise=False)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_a, **_k: (str(path), ""))
    previous = dialog._advanced_model
    dialog._load_annotations()
    qtbot.waitUntil(lambda: not dialog._export_jobs.is_busy() and dialog._normalization_ready
        and dialog._advanced_model is not previous)
    assert dialog._annotation_snapshot()[("plate1", "A01")]["condition"] == "treated"
    assert dialog.grouping.findData("condition") >= 0 and dialog.normalization_scope.findData("condition") >= 0
    dialog.normalization_scope.setCurrentIndex(dialog.normalization_scope.findData("condition"))
    qtbot.waitUntil(lambda: dialog._normalization_ready)
    assert dialog._advanced_model["provenance"]["sampled_fields"] == 8
    dialog.close()


def test_late_condition_csv_cannot_overwrite_new_condition_edits(tmp_path, qtbot, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from spacr import tabular
    import pandas as pd
    dialog = _browser(tmp_path, qtbot)
    qtbot.waitUntil(lambda: not dialog._metadata_jobs.is_busy())
    path = tmp_path / "conditions.csv"
    tabular.write_table(pd.DataFrame([dict(plate="plate1", well="A01", condition="old")]),
        str(path), canonicalise=False)
    original = tabular.read_table
    entered, release = threading.Event(), threading.Event()
    def delayed(source, **options):
        values = original(source, **options)
        if str(source) == str(path):
            entered.set()
            assert release.wait(5)
        return values
    monkeypatch.setattr(tabular, "read_table", delayed)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_a, **_k: (str(path), ""))
    dialog._load_annotations()
    qtbot.waitUntil(entered.is_set)
    dialog.annotation_scope.setCurrentIndex(dialog.annotation_scope.findData("plate"))
    dialog.annotation_text.setText("new")
    dialog._annotate()
    release.set()
    qtbot.waitUntil(lambda: not dialog._export_jobs.is_busy())
    assert dialog._conditions[("plate1", "A01")] == "new"
    dialog.close()


def test_changing_field_axis_keeps_individual_view_and_advanced_time_axis_relearns(tmp_path, qtbot):
    values = np.empty((2, 4, 6, 1), np.float32)
    values[0, ..., 0] = np.arange(24).reshape(4, 6)
    values[1, ..., 0] = np.arange(24).reshape(4, 6) + 100
    for field in (1, 2):
        tifffile.imwrite(tmp_path / f"plate1_A01_F{field}.tif", values[..., 0],
            metadata={"axes": "TYX"}, photometric="minisblack")
    dialog = _browser(tmp_path, qtbot, browser._overview_index(tmp_path))
    dialog.show()
    dialog._well(dialog.wells.item(0, 0))
    dialog.fields.setCurrentRow(0)
    dialog._field(dialog.fields.item(0))
    qtbot.waitUntil(lambda: dialog.view._item is not None and not dialog._image_jobs.is_busy())
    dialog.fov_axis.setCurrentIndex(dialog.fov_axis.findData("2"))
    qtbot.waitUntil(lambda: dialog.view._item is not None and not dialog._image_jobs.is_busy())
    assert dialog._individual and dialog.fields.item(0).data(browser.Qt.UserRole).field_id == "2"
    dialog.normalization_method.setCurrentIndex(dialog.normalization_method.findData("minmax"))
    dialog.normalization_scope.setCurrentIndex(dialog.normalization_scope.findData("plate"))
    dialog.advanced.setChecked(True)
    qtbot.waitUntil(lambda: dialog._normalization_ready)
    previous = dialog._advanced_model
    dialog.time_axis.setCurrentIndex(dialog.time_axis.findData(1))
    qtbot.waitUntil(lambda: dialog._normalization_ready and dialog._advanced_model is not previous)
    limits = next(iter(dialog._advanced_model["bounds"].values()))["0"]
    assert limits["low"] == 100 and limits["high"] == 123
    assert dialog._advanced_model["provenance"]["sampling_axes"]["time_index"] == 1
    dialog.close()
