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


def _browser(tmp_path, qtbot, fields=None, channels=(0,), options=None):
    settings = {"source": str(tmp_path), "channels": channels} | dict(options or {})
    dialog = browser._FieldOverview(settings=settings)
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


def test_compressed_metadata_wizard_keeps_next_text_scrollable_and_restores_footer(tmp_path, qtbot, qapp):
    from PySide6.QtCore import QSize, Qt
    from PySide6.QtGui import QRegion
    from PySide6.QtWidgets import QWidget, QScrollArea, QStyle, QStyleOptionButton
    from spacr.qt import app, gui_scale, preferences

    tifffile.imwrite(tmp_path / "plate1_A01_F1.tif", np.zeros((10, 12), np.uint16))
    old_gui = gui_scale.current_scale()
    old_saved_gui, old_font = preferences.get_gui_scale(), preferences.get_font_scale()
    old_sheet, old_app_font, old_palette = qapp.styleSheet(), qapp.font(), qapp.palette()
    host = wizard = None
    try:
        gui_scale.install_scaling_layer()
        gui_scale.apply_saved_gui_scale()
        app._load_bundled_fonts()
        assert app._use_open_sans(qapp) == "Open Sans"
        preferences.apply_preferences_to_app()
        host = QWidget()
        qtbot.addWidget(host)
        gui_scale.change_scales(host, gui=2., font=2., ask=False)
        wizard = browser._OverviewMetadataWizard(str(tmp_path), parent=host)
        wizard._settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        wizard.setGeometry(0, 0, 1356, 102)
        gui_scale._resize_actual(host, QSize(1356, 102))
        host.show()
        wizard.show()
        qtbot.waitUntil(wizard.next.isEnabled, timeout=20000)
        assert wizard.next.font().pixelSize() == 52
        from PySide6.QtGui import QFontInfo
        assert QFontInfo(wizard.next.font()).family() == "Open Sans"
        next_button = wizard.next
        events = []
        wizard.rejected.connect(lambda: events.append("rejected"))
        wizard.finished.connect(lambda result: events.append(result))
        for height in (102, 760, 102):
            gui_scale._resize_actual(host, QSize(1356, height))
            wizard.setGeometry(0, 0, 1356, height)
            qtbot.wait(50)
            assert QWidget.size(wizard) == QSize(1356, height)
            ancestor = next_button.parentWidget()
            while ancestor is not None and not isinstance(ancestor, QScrollArea):
                ancestor = ancestor.parentWidget()
            if height == 102:
                assert ancestor is not None, (next_button.geometry(), next_button.visibleRegion().boundingRect())
                assert ancestor is not wizard._wizard_scroll
                assert ancestor.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
                assert wizard._settings_scroll.horizontalScrollBar().isVisible()
                assert wizard._settings_scroll.parentWidget() is wizard._wizard_content
                ancestor.ensureWidgetVisible(next_button, 0, 0)
                qtbot.wait(50)
            else:
                assert ancestor is None
            option = QStyleOptionButton()
            option.initFrom(next_button)
            option.rect = next_button.rect()
            option.text = next_button.text()
            option.icon = next_button.icon()
            option.iconSize = next_button.iconSize()
            if next_button.isDefault():
                option.features |= QStyleOptionButton.DefaultButton
            if next_button.autoDefault():
                option.features |= QStyleOptionButton.AutoDefaultButton
            if next_button.isFlat():
                option.features |= QStyleOptionButton.Flat
            field = next_button.style().subElementRect(QStyle.SE_PushButtonContents,
                option, next_button)
            text = next_button.style().itemTextRect(next_button.fontMetrics(),
                field, int(Qt.AlignCenter | Qt.TextSingleLine),
                next_button.isEnabled(), next_button.text())
            assert field.contains(text)
            assert QRegion(text).subtracted(next_button.visibleRegion()).isEmpty(), (
                height, text, next_button.visibleRegion().boundingRect(),
                next_button.geometry(), wizard._compact_scroll.geometry(),
                wizard._compact_scroll.viewport().geometry(),
                wizard._compact_page.geometry(), wizard._action_bar.geometry(),
                wizard._wizard_shell.contentsMargins())
            assert wizard.next is next_button and wizard.next.isEnabled()
        floor = next_button.property("minimumSize").height()
        for font_scale, pixels in ((.1, 2), (2., 52)):
            gui_scale.change_scales(host, gui=2., font=font_scale, ask=False)
            qtbot.wait(50)
            assert next_button.font().pixelSize() == pixels
            current_floor = next_button.property("minimumSize").height()
            assert current_floor < floor if font_scale == .1 else current_floor == floor
            assert next_button.maximumHeight() == 16777215
        wizard.reject()
        assert events == ["rejected", browser.QDialog.Rejected]
        assert not wizard.isVisible() and wizard._cancelled.is_set()
    finally:
        try:
            if wizard is not None:
                wizard.reject()
                wizard._runner.shutdown(timeout_ms=3000)
        finally:
            try:
                if host is not None:
                    host.close()
            finally:
                try:
                    if host is not None:
                        gui_scale.change_scales(host, gui=old_saved_gui, font=old_font, ask=False)
                finally:
                    preferences.set_gui_scale(old_saved_gui)
                    preferences.set_font_scale(old_font)
                    gui_scale.set_gui_scale_live(old_gui)
                    qapp.setStyleSheet(old_sheet)
                    qapp.setFont(old_app_font)
                    qapp.setPalette(old_palette)


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


def test_large_viewport_native_crop_remains_bounded_in_both_spatial_dimensions(tmp_path, qtbot):
    from PySide6.QtCore import QSize
    from spacr.qt.gui_scale import _resize_actual
    path = tmp_path / "plate1_A01_F1.npy"
    raw = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(8192, 8192))
    raw[4000:4100, 4000:4100] = 255
    raw.flush()
    del raw
    dialog = _browser(tmp_path, qtbot, browser._overview_index(tmp_path))
    dialog.show()
    _resize_actual(dialog, QSize(3000, 2600))
    dialog._well(dialog.wells.item(0, 0))
    dialog.fields.setCurrentRow(0)
    dialog._field(dialog.fields.item(0))
    dialog._splitter.setSizes([100, 100, 2800])
    try:
        qtbot.waitUntil(lambda: dialog.view._item is not None and dialog.view._item.pixmap().width() == 1024
            and not dialog._image_jobs.is_busy())
    except Exception:
        pytest.fail(str(dict(status=dialog.status.text(), edge=dialog._preview_edge,
            pixels=dialog.view._item.pixmap().size() if dialog.view._item else None,
            jobs=dialog._image_jobs.active_jobs(), pending=dialog._image_pending,
            visible=dialog.view.viewport().size(), window=dialog.size())))
    qtbot.waitUntil(lambda: dialog.view.viewport().width() > 1500)
    dialog.view.zoom_by(3)
    dialog._request_detail()
    qtbot.waitUntil(lambda: dialog._native_key is not None and not dialog._native_jobs.is_busy(), timeout=10000)
    y0, y1, x0, x1 = dialog._native_key[2]
    assert y1 - y0 == x1 - x0 == 2048
    assert dialog._native_item.pixmap().size() == QSize(2048, 2048)
    dialog.close()


def test_saved_metadata_restore_relearns_advanced_model_and_preserves_current_condition(tmp_path, qtbot):
    dialog = _browser(tmp_path, qtbot, options={"conditions": {("plate1", "A01"): "current"}})
    qtbot.waitUntil(lambda: not dialog._metadata_jobs.is_busy())
    dialog.normalization_scope.setCurrentIndex(dialog.normalization_scope.findData("plate"))
    dialog.advanced.setChecked(True)
    qtbot.waitUntil(lambda: dialog._normalization_ready and dialog._advanced_model is not None)
    previous = dialog._advanced_model
    path = tmp_path / "results" / "overview_annotations.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(dict(wells=[dict(plate="plate1", well="A01",
        metadata=dict(condition="saved", dose="12 nM"))])))
    dialog._restore_metadata()
    qtbot.waitUntil(lambda: not dialog._metadata_jobs.is_busy() and dialog._normalization_ready
        and dialog._advanced_model is not previous)
    assert dialog._annotation_snapshot()[("plate1", "A01")] == dict(condition="current", dose="12 nM")
    assert dialog.normalization_scope.findData("dose") >= 0
    dialog.close()


def test_late_metadata_restore_preserves_current_field_without_selecting_new_exports(tmp_path, qtbot, monkeypatch):
    dialog = _browser(tmp_path, qtbot)
    qtbot.waitUntil(lambda: not dialog._metadata_jobs.is_busy())
    path = tmp_path / "results" / "overview_annotations.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(dict(wells=[dict(plate="plate1", well="A01", metadata=dict(dose="12 nM"))])))
    original = type(path).read_text
    entered, release = threading.Event(), threading.Event()
    def delayed(current, *args, **kwargs):
        value = original(current, *args, **kwargs)
        if current == path:
            entered.set()
            assert release.wait(5)
        return value
    monkeypatch.setattr(type(path), "read_text", delayed)
    dialog._restore_metadata()
    qtbot.waitUntil(entered.is_set)
    dialog.show()
    dialog._well(dialog.wells.item(0, 0))
    dialog.fields.setCurrentRow(1)
    current = dialog.fields.item(1).data(browser.Qt.UserRole)
    dialog.fields.clearSelection()
    assert not dialog._selected_fields
    dialog._field(dialog.fields.item(1))
    release.set()
    qtbot.waitUntil(lambda: not dialog._metadata_jobs.is_busy() and not dialog._image_jobs.is_busy()
        and dialog.view._item is not None)
    assert dialog.fields.currentItem().data(browser.Qt.UserRole).identity == current.identity
    assert not dialog._selected_fields and not dialog.fields.selectedItems()
    assert dialog._annotation_snapshot()[("plate1", "A01")]["dose"] == "12 nM"
    dialog.close()


def test_keyboard_field_navigation_displays_current_field_pixels(tmp_path, qtbot):
    dialog = _browser(tmp_path, qtbot)
    dialog.show()
    dialog._well(dialog.wells.item(0, 0))
    dialog.fields.setCurrentRow(0)
    qtbot.waitUntil(lambda: not dialog._image_jobs.is_busy() and dialog.view._item is not None)
    first = dialog.fields.currentItem().data(browser.Qt.UserRole)
    dialog.fields.setFocus()
    qtbot.keyClick(dialog.fields, browser.Qt.Key_Right)
    current = dialog.fields.currentItem().data(browser.Qt.UserRole)
    assert current.identity != first.identity
    qtbot.waitUntil(lambda: not dialog._image_jobs.is_busy() and dialog._preview_edge == 1024)
    assert dialog._displayed_field[0] == current.identity
    channels, limits = dialog._choices()
    expected = browser._overview_pixels(current, channels, limits, edge=1024, **dialog._display_kwargs())
    actual = dialog.view._item.pixmap().toImage()
    from PySide6.QtGui import QImage
    actual = actual.convertToFormat(QImage.Format_RGB888)
    pixels = np.frombuffer(actual.bits(), np.uint8).reshape(actual.height(), actual.bytesPerLine())
    assert np.array_equal(pixels[:, :actual.width() * 3].reshape(actual.height(), actual.width(), 3), expected)
    dialog.close()


def test_custom_annotation_deletion_and_target_scope_updates_keep_global_settings(tmp_path, qtbot):
    dialog = _browser(tmp_path, qtbot, channels=(0,))
    dialog.wells.item(0, 0).setSelected(True)
    dialog.annotation_key.setEditText("dose")
    dialog.annotation_text.setText("10 nM")
    dialog._annotate()
    assert dialog._annotation_snapshot()[("plate1", "A01")]["dose"] == "10 nM"
    dialog.annotation_text.clear()
    dialog._annotate()
    assert "dose" not in dialog._annotation_snapshot().get(("plate1", "A01"), {})
    dialog.norm_channel.setCurrentIndex(dialog.norm_channel.findData(0))
    dialog.normalization_scope.setCurrentIndex(dialog.normalization_scope.findData("well"))
    assert dialog._normalization_options()["scope"] == "well"
    dialog.representatives.setValue(2)
    assert dialog.representatives.value() == 2
    dialog.close()


def test_queued_native_camera_after_current_field_disappears_is_safe(tmp_path, qtbot):
    dialog = _browser(tmp_path, qtbot)
    dialog.show()
    dialog._well(dialog.wells.item(0, 0))
    dialog.fields.setCurrentRow(0)
    qtbot.waitUntil(lambda: not dialog._image_jobs.is_busy() and dialog.view._item is not None)
    dialog.fields.setCurrentRow(-1)
    dialog.view._user_zoomed = True
    dialog.view.zoom_by(3)
    dialog._request_detail()
    dialog._load_native_detail()
    assert dialog._native_jobs.active_jobs() == 0
    assert dialog._native_key is None
    dialog._open_selected_well()
    dialog.close()


def test_camera_before_new_field_header_finishes_does_not_borrow_previous_shape(tmp_path, qtbot, monkeypatch):
    dialog = _browser(tmp_path, qtbot)
    dialog.show()
    dialog._well(dialog.wells.item(0, 0))
    dialog.fields.setCurrentRow(0)
    qtbot.waitUntil(lambda: not dialog._image_jobs.is_busy() and dialog.view._item is not None)
    original = browser._overview_channel_info
    entered, release = threading.Event(), threading.Event()
    next_field = dialog.fields.item(1).data(browser.Qt.UserRole)
    def delayed(field):
        if field.identity == next_field.identity:
            entered.set()
            assert release.wait(5)
        return original(field)
    monkeypatch.setattr(browser, "_overview_channel_info", delayed)
    dialog.fields.setCurrentRow(1)
    qtbot.waitUntil(entered.is_set)
    assert next_field.identity not in dialog._source_shapes
    dialog._load_native_detail()
    assert dialog._native_jobs.active_jobs() == 0
    release.set()
    qtbot.waitUntil(lambda: not dialog._image_jobs.is_busy() and dialog.view._item is not None)
    dialog.close()


def test_wizard_closed_queued_scan_and_missing_geometry_cannot_open(tmp_path, qtbot):
    from PySide6.QtCore import QTimer
    _images(tmp_path)
    wizard = browser._OverviewMetadataWizard(str(tmp_path))
    qtbot.addWidget(wizard)
    qtbot.waitUntil(lambda: wizard.next.isEnabled())
    wizard.plate_geometry.setCurrentIndex(-1)
    with pytest.raises(ValueError, match="plate geometry"):
        wizard.configuration()
    QTimer.singleShot(0, wizard._scan)
    wizard.reject()
    qtbot.wait(5)
    assert not wizard._runner.is_busy()


def test_plate_grid_refuses_unbounded_physical_coordinate_without_allocating_cells(tmp_path, qtbot):
    from dataclasses import replace
    fields = _images(tmp_path)
    field = replace(fields[0], well="A513", row="1", column=513)
    dialog = _browser(tmp_path, qtbot, (field,))
    dialog._filter()
    assert "grid limit" in dialog.status.text()
    assert dialog.wells.columnCount() < 513
    dialog.close()


def test_ctrl_click_well_retains_plate_selection_instead_of_opening(tmp_path, qtbot):
    dialog = _browser(tmp_path, qtbot)
    dialog.show()
    item = dialog.wells.item(0, 0)
    qtbot.mouseClick(dialog.wells.viewport(), browser.Qt.LeftButton,
        browser.Qt.ControlModifier, pos=dialog.wells.visualItemRect(item).center())
    assert dialog._scope[1] is None and not dialog._individual
    assert ("plate1", "A01") in dialog._selected_wells
    dialog.close()


def test_empty_browser_axis_controls_and_zoom_mode_do_not_invent_fields(tmp_path, qtbot):
    dialog = _browser(tmp_path, qtbot, ())
    assert not dialog._fields and not dialog._visible
    dialog._axis_changed()
    dialog.wheel_axis.setCurrentIndex(dialog.wheel_axis.findData("zoom"))
    before = tuple(combo.currentIndex() for combo in (dialog.fov_axis, dialog.z_axis, dialog.time_axis))
    dialog._step_axis(1)
    assert tuple(combo.currentIndex() for combo in (dialog.fov_axis, dialog.z_axis, dialog.time_axis)) == before
    assert dialog.fields.count() == 0
    dialog.close()


def test_reference_to_absent_plate_refuses_new_source_without_silent_fallback(tmp_path, qtbot):
    dialog = _browser(tmp_path, qtbot)
    dialog._overview_options["normalization"] = dict(mode="reference_average",
        reference_kind="reference_well", reference=("missing-plate", "A01"))
    dialog._indexed(dialog._fields)
    qtbot.waitUntil(lambda: not dialog._index_jobs.is_busy())
    assert not dialog._normalization_ready
    assert "reference has no fields" in dialog.status.text()
    dialog.close()


def test_unreadable_selected_comparison_does_not_install_fabricated_pixels(tmp_path, qtbot):
    dialog = _browser(tmp_path, qtbot)
    dialog._well(dialog.wells.item(0, 0))
    dialog.fields.selectAll()
    selected = tuple(dialog._selected_fields.values())
    assert len(selected) == 2
    for field in selected:
        for _, path in field.files:
            from pathlib import Path
            Path(path).unlink()
    dialog._compare()
    qtbot.waitUntil(lambda: not dialog._montage_jobs.is_busy())
    assert dialog.view._item is not None
    assert "2 unreadable previews" in dialog.status.text()
    dialog.close()


def test_wizard_refuses_source_larger_than_every_standard_geometry(tmp_path, qtbot):
    tifffile.imwrite(tmp_path / "plate1_A513_F1.tif", np.zeros((4, 4), np.uint8))
    wizard = browser._OverviewMetadataWizard(str(tmp_path))
    qtbot.addWidget(wizard)
    qtbot.waitUntil(lambda: wizard._info is not None)
    with pytest.raises(ValueError, match="geometry"):
        wizard.configuration()
    wizard.reject()


def test_cancel_after_metadata_inference_does_not_inspect_an_image_header(tmp_path, monkeypatch):
    _images(tmp_path)
    cancelled = threading.Event()
    original = browser._overview_index
    calls = []
    def indexed(*args, **kwargs):
        fields = original(*args, **kwargs)
        calls.append(fields)
        if len(calls) == 2:
            cancelled.set()
        return fields
    monkeypatch.setattr(browser, "_overview_index", indexed)
    fields, info = browser._overview_wizard_probe(str(tmp_path), "", "auto", cancelled)
    assert len(calls) == 2 and fields and info is None


def test_nearest_nested_well_and_explicit_channel_named_files_keep_source_identity(tmp_path):
    nested = tmp_path / "plate1" / "A01" / "B02"
    nested.mkdir(parents=True)
    tifffile.imwrite(nested / "F1.tif", np.zeros((4, 4), np.uint8))
    nested_fields = browser._overview_index(tmp_path)
    assert len(nested_fields) == 1 and nested_fields[0].well == "B02"
    root = tmp_path / "raw"
    for channel in (1, 2):
        folder = root / f"C{channel}"
        folder.mkdir(parents=True)
        tifffile.imwrite(folder / f"A01_alpha_C{channel}.tif", np.zeros((4, 4), np.uint8))
    fields = browser._overview_index(root, r"(?P<wellID>A01)_alpha_C(?P<chanID>\d+)")
    assert len(fields) == 1 and len(fields[0].files) == 2
    assert fields[0].field.startswith("{channel}")


def test_late_thumbnail_for_filtered_well_never_replaces_visible_well(tmp_path, qtbot):
    dialog = _browser(tmp_path, qtbot)
    old_well = "A01"
    key = ("detached old thumbnail",)
    pixels = np.full((4, 4, 3), 255, np.uint8)
    dialog.rows.setCurrentIndex(dialog.rows.findData("2"))
    assert old_well not in dialog._well_positions
    dialog._set_thumbnail(old_well, key, pixels)
    assert old_well not in dialog._installed_thumbnails
    for row in range(dialog.wells.rowCount()):
        for column in range(dialog.wells.columnCount()):
            item = dialog.wells.item(row, column)
            assert item is None or item.data(browser.Qt.UserRole + 1) != key
    dialog.close()


def test_display_prepare_finishing_after_current_field_cleared_keeps_selection_empty(tmp_path, qtbot, monkeypatch):
    dialog = _browser(tmp_path, qtbot)
    dialog.show()
    dialog._well(dialog.wells.item(0, 0))
    dialog.fields.setCurrentRow(0)
    qtbot.waitUntil(lambda: not dialog._image_jobs.is_busy() and dialog.view._item is not None)
    original = browser._overview_channel_info
    entered, release = threading.Event(), threading.Event()
    def delayed(field):
        entered.set()
        assert release.wait(5)
        return original(field)
    monkeypatch.setattr(browser, "_overview_channel_info", delayed)
    dialog._prepare_display()
    qtbot.waitUntil(entered.is_set)
    dialog.fields.setCurrentRow(-1)
    dialog.fields.clearSelection()
    release.set()
    qtbot.waitUntil(lambda: not dialog._index_jobs.is_busy() and dialog._normalization_ready)
    assert dialog.fields.currentItem() is None and not dialog._selected_fields
    assert not dialog._image_jobs.is_busy()
    dialog.close()


def _workspace_host(tmp_path, qtbot):
    from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout
    host = QWidget()
    host.app_key = "mask"
    host._fold_page_title = "Mask Generation"
    layout = QVBoxLayout(host)
    layout.addWidget(QLabel("Host masthead"))
    body = QWidget(host)
    body.setObjectName("OriginalHostBody")
    layout.addWidget(body, 1)
    host.resize(960, 760)
    def retire(widget):
        workspace = getattr(widget, "_field_overview_workspace", None)
        if workspace is None:
            return
        workspace.close()
        wizard = widget._field_overview_wizard
        qtbot.waitUntil(lambda: wizard._runner.active_jobs() == 0)
        wizard._runner.shutdown(timeout_ms=3000)
        dialog = getattr(widget, "_field_overview", None)
        if dialog is not None:
            runners = (dialog._index_jobs, dialog._image_jobs, dialog._export_jobs,
                dialog._metadata_jobs, dialog._measurement_jobs, dialog._native_jobs,
                dialog._thumbnail_jobs, dialog._montage_jobs, dialog._field_thumbnail_jobs)
            qtbot.waitUntil(lambda: all(runner.active_jobs() == 0 for runner in runners))
            for runner in runners:
                runner.shutdown(timeout_ms=3000)
        from spacr.qt.bridge import prune_parked_threads
        qtbot.waitUntil(lambda: prune_parked_threads() == 0)
    qtbot.addWidget(host, before_close_func=retire)
    host.show()
    return host, body


def test_embedded_overview_expands_plate_keeps_options_separate_and_back_returns_host(tmp_path, qtbot):
    from PySide6.QtCore import QSize
    from spacr.qt.gui_scale import _resize_actual
    _images(tmp_path)
    host, body = _workspace_host(tmp_path, qtbot)
    wizard = browser._open_field_overview(host, dict(source=str(tmp_path), geometry=(2, 3)))
    workspace = host._field_overview_workspace
    pages = host._fold_pages
    assert pages.widget(0) is body and pages.currentWidget() is workspace
    assert not wizard.isWindow() and wizard.window() is host
    assert not workspace.back_button.isVisible() and not workspace.autoFillBackground()
    qtbot.waitUntil(wizard.next.isEnabled)
    qtbot.mouseClick(wizard.next, browser.Qt.LeftButton)
    dialog = host._field_overview
    qtbot.waitUntil(lambda: dialog._normalization_ready and not dialog._index_jobs.is_busy())
    assert workspace.stack.currentWidget() is dialog
    assert not dialog.isWindow() and dialog.window() is host
    qtbot.waitUntil(lambda: dialog.wells.viewport().width() > 500)
    before = dialog.wells.viewport().size()
    before_cell = dialog._plate_cell_extent
    _resize_actual(host, QSize(1440, 1040))
    qtbot.waitUntil(lambda: dialog.wells.viewport().width() > before.width() + 200)
    assert dialog._plate_cell_extent[0] > before_cell[0]
    plate_area = dialog.wells.viewport().size()
    qtbot.mouseClick(dialog.options_button, browser.Qt.LeftButton)
    assert dialog._options_window.isWindow() and dialog._options_window.isVisible()
    assert dialog._controls_scroll.window() is dialog._options_window
    assert dialog.wells.viewport().size() == plate_area
    dialog.gamma.setValue(1.7)
    dialog._options_window.close()
    assert not dialog.options_button.isChecked()
    qtbot.mouseClick(dialog.options_button, browser.Qt.LeftButton)
    assert dialog.gamma.value() == 1.7
    dialog._options_window.close()
    dialog._zoom_plate(1)
    manual_extent = dialog._plate_cell_extent
    _resize_actual(host, QSize(1540, 1140))
    qtbot.waitUntil(lambda: dialog.wells.viewport().width() > plate_area.width())
    assert dialog._plate_cell_extent == manual_extent
    dialog._well(dialog.wells.item(0, 0))
    qtbot.waitUntil(lambda: dialog._displayed_montage is not None and not dialog._montage_jobs.is_busy())
    montage = dialog.view._item.pixmap().toImage().copy()
    dialog.fields.setCurrentRow(0)
    qtbot.waitUntil(lambda: dialog._individual and dialog.view._item is not None
        and not dialog._image_jobs.is_busy() and dialog.view._item.pixmap().width() == 12)
    qtbot.mouseClick(workspace.back_button, browser.Qt.LeftButton)
    assert not dialog._individual and dialog._scope[1] == "A01"
    qtbot.waitUntil(lambda: dialog.view._item.pixmap().toImage() == montage
        and not dialog._montage_jobs.is_busy())
    qtbot.mouseClick(workspace.back_button, browser.Qt.LeftButton)
    assert dialog._scope[1] is None
    qtbot.mouseClick(workspace.back_button, browser.Qt.LeftButton)
    assert pages.currentWidget() is body and pages.indexOf(workspace) == -1
    same = browser._open_field_overview(host, dict(source=str(tmp_path)))
    assert same is wizard and host._field_overview is dialog
    assert pages.currentWidget() is workspace and workspace.stack.currentWidget() is dialog


@pytest.mark.parametrize("selector", ("pdf_quality", "pdf_dpi", "grouping",
                                    "annotation_scope", "annotation_key",
                                    "normalization_method", "normalization_scope"))
def test_pdf_quality_reserves_all_native_text_slots_when_fonts_change(qtbot, qapp, selector):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QStyle, QStyleOptionComboBox
    from spacr.qt import gui_scale, theme

    old_gui = gui_scale.current_scale()
    body = None
    try:
        gui_scale.install_scaling_layer()
        gui_scale.set_gui_scale_live(.9)
        body = browser._FieldOverview()
        qtbot.addWidget(body)
        combo = getattr(body, selector)
        selected = (combo.currentIndex(), combo.currentText(), combo.currentData())
        floors = []
        for scale, pixels in ((.9, 11), (2., 23), (.9, 11)):
            body.setStyleSheet(theme.stylesheet('dark', font_scale=scale))
            combo.ensurePolished()
            gui_scale._resize_actual(combo, combo.minimumSizeHint())
            assert combo.font().pixelSize() == pixels
            option = QStyleOptionComboBox()
            option.initFrom(combo)
            option.rect = combo.rect()
            option.frame, option.editable = combo.hasFrame(), combo.isEditable()
            option.currentText, option.currentIcon = combo.currentText(), combo.itemIcon(combo.currentIndex())
            option.iconSize = combo.iconSize()
            field = combo.style().subControlRect(QStyle.CC_ComboBox, option, QStyle.SC_ComboBoxEditField, combo)
            arrow = combo.style().subControlRect(QStyle.CC_ComboBox, option, QStyle.SC_ComboBoxArrow, combo)
            assert not field.isEmpty() and not arrow.isEmpty()
            assert combo.rect().contains(field) and combo.rect().contains(arrow)
            for index in range(combo.count()):
                text = combo.style().itemTextRect(combo.fontMetrics(), field,
                    int(Qt.AlignVCenter | Qt.AlignLeft | Qt.TextSingleLine), combo.isEnabled(), combo.itemText(index))
                assert combo.itemIcon(index).isNull() and field.contains(text), (scale, combo.rect(), field, text)
            floors.append(combo.minimumSize())
            assert (combo.currentIndex(), combo.currentText(), combo.currentData()) == selected
            assert combo.maximumWidth() == combo.maximumHeight() == 16777215
        assert floors[-1] == floors[0] and floors[1].height() > floors[0].height()
    finally:
        try:
            if body is not None:
                body._retire()
        finally:
            gui_scale.set_gui_scale_live(old_gui)


def test_advanced_background_pair_keeps_wrapped_caption_and_native_editor_whole(qtbot, qapp):
    from PySide6.QtGui import QTextDocument, QTextOption
    from PySide6.QtWidgets import QLabel, QLineEdit, QStyle, QStyleOptionFrame
    from spacr.qt import gui_scale, theme

    old_gui = gui_scale.current_scale()
    body = None
    try:
        gui_scale.install_scaling_layer()
        gui_scale.set_gui_scale_live(.9)
        body = browser._FieldOverview()
        qtbot.addWidget(body)
        pair = body.background.parentWidget()
        caption = pair.findChild(QLabel)
        assert caption is not None and caption.wordWrap()
        floors = []
        for scale, pixels in ((.9, 11), (2., 23), (.9, 11)):
            body.setStyleSheet(theme.stylesheet('dark', font_scale=scale))
            body._options_window.show()
            qtbot.waitUntil(lambda: body._options_window.windowHandle() is not None
                            and body._options_window.windowHandle().isExposed())
            qapp.processEvents()
            assert caption.font().pixelSize() == body.background.font().pixelSize() == pixels
            document = QTextDocument()
            document.setDefaultFont(caption.font())
            document.setDocumentMargin(0)
            option = document.defaultTextOption()
            option.setWrapMode(QTextOption.WordWrap)
            document.setDefaultTextOption(option)
            document.setPlainText(caption.text())
            area = caption.contentsRect().adjusted(caption.margin(), caption.margin(),
                                                   -caption.margin(), -caption.margin())
            document.setTextWidth(area.width())
            assert document.size().height() <= area.height() + 1 / caption.devicePixelRatioF(), (caption.geometry(), document.size())
            editor = body.background.findChild(QLineEdit)
            assert editor is not None
            frame = QStyleOptionFrame()
            frame.initFrom(editor)
            frame.rect = editor.rect()
            frame.lineWidth = editor.style().pixelMetric(QStyle.PM_DefaultFrameWidth, frame, editor) if editor.hasFrame() else 0
            frame.midLineWidth = 0
            field = editor.style().subElementRect(QStyle.SE_LineEditContents, frame, editor)
            margins = editor.textMargins()
            field = field.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
            assert field.height() >= editor.fontMetrics().height(), (editor.rect(), field, editor.fontMetrics().height())
            assert not caption.geometry().intersects(body.background.geometry())
            assert pair.maximumWidth() == 16777215
            floors.append(pair.sizeHint())
            assert body.background.value() == 0
        assert floors[-1] == floors[0]
    finally:
        try:
            if body is not None:
                body._retire()
        finally:
            gui_scale.set_gui_scale_live(old_gui)


def test_embedded_metadata_cancel_and_source_replacement_keep_original_module_body(tmp_path, qtbot):
    _images(tmp_path)
    host, body = _workspace_host(tmp_path, qtbot)
    wizard = browser._open_field_overview(host, dict(source=str(tmp_path)))
    first = host._field_overview_workspace
    qtbot.waitUntil(wizard.next.isEnabled)
    wizard.reject()
    assert host._fold_pages.currentWidget() is body
    same = browser._open_field_overview(host, dict(source=str(tmp_path)))
    assert same is not wizard and host._fold_pages.currentWidget() is host._field_overview_workspace
    qtbot.waitUntil(same.next.isEnabled)
    qtbot.mouseClick(same.next, browser.Qt.LeftButton)
    qtbot.waitUntil(lambda: host._field_overview._normalization_ready)
    assert host._field_overview._fields
    second_source = tmp_path / "second"
    second_source.mkdir()
    tifffile.imwrite(second_source / "plate2_A01_F1.tif", np.zeros((4, 4), np.uint8))
    second = browser._open_field_overview(host, dict(source=str(second_source)))
    qtbot.waitUntil(second.next.isEnabled)
    assert second is not wizard and not second.isWindow()
    assert host._fold_pages.widget(0) is body and host._fold_pages.count() == 2
    qtbot.mouseClick(host._field_overview_workspace.back_button, browser.Qt.LeftButton)
    assert host._fold_pages.currentWidget() is body
    assert browser._open_field_overview(host, dict(source=str(second_source))) is second
    assert host._fold_pages.currentWidget() is host._field_overview_workspace


def test_cancel_replacement_metadata_restores_original_validated_browser_after_multiple_sources(tmp_path, qtbot):
    from shiboken6 import isValid
    _images(tmp_path)
    host, body = _workspace_host(tmp_path, qtbot)
    wizard = browser._open_field_overview(host, dict(source=str(tmp_path)))
    qtbot.waitUntil(wizard.next.isEnabled)
    qtbot.mouseClick(wizard.next, browser.Qt.LeftButton)
    current, workspace = host._field_overview, host._field_overview_workspace
    qtbot.waitUntil(lambda: current._normalization_ready)
    token, fields = current._cancelled, current._fields
    replacements = []
    for name in ("replacement-two", "replacement-three"):
        source = tmp_path / name
        source.mkdir()
        tifffile.imwrite(source / "plate2_B02_F1.tif", np.zeros((4, 4), np.uint8))
        replacement = browser._open_field_overview(host, dict(source=str(source)))
        replacements.append(replacement)
        qtbot.waitUntil(replacement.next.isEnabled)
        assert isValid(current) and not token.is_set()
    assert replacements[0]._cancelled.is_set()
    replacements[1].reject()
    assert host._field_overview_workspace is workspace
    assert host._field_overview_wizard is wizard and host._field_overview is current
    assert host._fold_pages.currentWidget() is workspace
    assert current._fields == fields and not token.is_set()
    assert host._fold_pages.widget(0) is body


def test_closing_pending_replacement_retires_both_sources_without_restoring_old_page(tmp_path, qtbot):
    _images(tmp_path)
    host, body = _workspace_host(tmp_path, qtbot)
    wizard = browser._open_field_overview(host, dict(source=str(tmp_path)))
    qtbot.waitUntil(wizard.next.isEnabled)
    qtbot.mouseClick(wizard.next, browser.Qt.LeftButton)
    current = host._field_overview
    qtbot.waitUntil(lambda: current._normalization_ready)
    source = tmp_path / "replacement"
    source.mkdir()
    tifffile.imwrite(source / "plate2_B02_F1.tif", np.zeros((4, 4), np.uint8))
    replacement = browser._open_field_overview(host, dict(source=str(source)))
    workspace = host._field_overview_workspace
    qtbot.waitUntil(replacement.next.isEnabled)
    workspace.close()
    assert replacement._cancelled.is_set() and current._cancelled.is_set()
    assert workspace._previous_workspace is None
    assert host._field_overview_workspace is workspace
    assert host._fold_pages.currentWidget() is body


def test_field_thumbnails_decode_visible_rows_and_load_new_rows_on_scroll(tmp_path, qtbot):
    for number in range(1, 25):
        tifffile.imwrite(tmp_path / f"plate1_A01_F{number}.tif",
            np.arange(120, dtype=np.uint16).reshape(10, 12) + number)
    dialog = _browser(tmp_path, qtbot, browser._overview_index(tmp_path))
    dialog.show()
    dialog.resize(760, 620)
    dialog.fields.setFixedWidth(180)
    dialog._well(dialog.wells.item(0, 0))
    qtbot.waitUntil(lambda: dialog.fields.verticalScrollBar().maximum() > 0)
    first, last = dialog.fields.item(0), dialog.fields.item(23)
    qtbot.waitUntil(lambda: not first.icon().isNull() and not dialog._field_thumbnail_jobs.is_busy())
    assert last.icon().isNull()
    dialog.fields.scrollToItem(last)
    qtbot.waitUntil(lambda: not last.icon().isNull() and not dialog._field_thumbnail_jobs.is_busy())
    assert first.data(browser.Qt.UserRole).identity != last.data(browser.Qt.UserRole).identity
    dialog.close()


@pytest.mark.parametrize("host_key", ("mask", "make_masks"))
def test_actual_application_dialog_filters_keep_metadata_and_browser_in_their_host(tmp_path, host_key):
    import os
    import subprocess
    import sys
    tifffile.imwrite(tmp_path / "plate1_A01_F1.tif", np.arange(120, dtype=np.uint16).reshape(10, 12))
    tifffile.imwrite(tmp_path / "OTF_DC_DCAM#1_CAM1.tif", np.zeros((4, 4), np.uint16))
    code = r'''
import sys, time, faulthandler
started = time.monotonic()
def progress(message):
 """Flush a bounded workflow-stage breadcrumb to the captured child stream."""
 print('overview-host-stage %.3fs %s' % (time.monotonic() - started, message),
  file=sys.stderr, flush=True)
faulthandler.dump_traceback_later(25, file=sys.stderr)
progress('import Qt')
from PySide6.QtWidgets import QApplication, QDialog, QSpacerItem, QSizePolicy
from PySide6.QtCore import QTimer, Qt
from PySide6.QtTest import QTest
progress('import application')
from spacr.qt.app import MainWindow, install_the_dialog_filters
from spacr.qt import preferences
from spacr.qt.first_run import mark_tour_seen
from spacr.qt.theme import active_palette
from PySide6.QtGui import QColor, QPalette
from spacr.qt.widgets.qc_field_browser import _open_field_overview
from spacr.qt.screens.map_barcodes import host_pages
progress('create QApplication')
app = QApplication([])
app.setQuitOnLastWindowClosed(False)
mark_tour_seen()
preferences.set_theme('dark' if sys.argv[1] == 'mask' else 'light')
preferences.apply_preferences_to_app(app)
progress('create MainWindow')
window = MainWindow()
progress('MainWindow created')
window.resize(1180, 820)
window.show()
install_the_dialog_filters(app)
state = {}
def finish(code):
 """Stop this workflow's owned jobs and close the real window before exit."""
 if state.get('finishing'):
  return
 state['finishing'] = True
 progress('finish %s: cancel owned jobs' % code)
 timer.stop()
 host = state.get('host')
 owners = [state.get('wizard'), state.get('body'), state.get('replacement')]
 workspace = getattr(host, '_field_overview_workspace', None)
 runners = []
 for owner in owners:
  if owner is None:
   continue
  for name in ('_visual_timer', '_native_timer', '_readout_timer', '_startup_timer', '_rescan_timer', '_action_fit_timer'):
   owned_timer = getattr(owner, name, None)
   if owned_timer is not None:
    owned_timer.stop()
  cancelled = getattr(owner, '_cancelled', None)
  if cancelled is not None:
   cancelled.set()
  cancel = getattr(owner, '_cancel', None)
  if callable(cancel):
   cancel()
  for name in ('_runner', '_index_jobs', '_image_jobs', '_export_jobs', '_thumbnail_jobs',
    '_montage_jobs', '_field_thumbnail_jobs', '_metadata_jobs', '_native_jobs', '_measurement_jobs'):
   runner = getattr(owner, name, None)
   if runner is not None:
    runner.cancel()
    runners.append(runner)
 progress('finish: drain %s owned runners' % len(runners))
 for runner in runners:
  runner.shutdown(timeout_ms=3000)
 progress('finish: close workspace and window')
 if workspace is not None:
  workspace.close()
 window.close()
 progress('finish: leave event loop')
 app.exit(code)
def fail():
 import traceback
 traceback.print_exc()
 finish(1)
def open_empty():
 try:
  window.open_module(sys.argv[1])
  host = window._screens[sys.argv[1]]
  assert host_pages(host).currentIndex() == 0
  original_settings = host._settings_scroll.widget()
  if sys.argv[1] == 'mask':
   state['figure_spacer'] = QSpacerItem(9, 13, QSizePolicy.Minimum, QSizePolicy.Fixed)
   host._figures_card.body_layout.addItem(state['figure_spacer'])
  console = host._console if hasattr(host, '_console') else host._masks_console.console
  settings = {'custom_regex': r'(?P<plateID>plate\d+)_(?P<wellID>[A-Z]\d+)_F(?P<fieldID>\d+)\.tif$'}
  if sys.argv[1] == 'mask':
   settings['geometry'] = (16, 24)
  wizard = _open_field_overview(host, settings)
  state.update(host=host, wizard=wizard, stage=0, original_settings=original_settings, console=console, settings=settings)
 except Exception:
  fail()
def tick():
 try:
  if not state:
   return
  host, wizard = state['host'], state['wizard']
  if state['stage'] == 0:
   assert wizard.isVisible() and wizard.source.isVisible()
   assert not wizard.isWindow() and not isinstance(wizard, QDialog)
   assert not wizard.property('spacrGlassed') and wizard.window() is window
   assert wizard.preview.isVisible() and not host._field_overview_workspace.back_button.isVisible()
   if sys.argv[1] == 'mask':
    retained = host._overview_content_stack.widget(0).layout()
    assert any(retained.itemAt(index) is state['figure_spacer'] for index in range(retained.count()))
   assert host._settings_scroll.widget() is wizard._settings_content
   assert _open_field_overview(host, state['settings']) is wizard
   assert state['console'] is (host._console if hasattr(host, '_console') else host._masks_console.console)
   if hasattr(host, '_btn_run'):
    assert not host._btn_run.isVisible() and wizard.next.isVisible()
    assert not host._settings_search.isVisible()
   if not state.get('cancelled_initial'):
    progress('cancel initial metadata wizard')
    wizard._runner.shutdown(timeout_ms=3000)
    wizard.reject()
    assert host._settings_scroll.widget() is state['original_settings']
    state['wizard'] = _open_field_overview(host, state['settings'])
    assert state['wizard'] is not wizard
    state['cancelled_initial'] = True
    return
   wizard.source.setText(sys.argv[2])
   wizard._scan()
   state['stage'] = 1
  elif state['stage'] == 1 and wizard.next.isEnabled():
   if wizard._readout_timer.isActive():
    return
   assert wizard.axes.height() >= wizard.axes.heightForWidth(wizard.axes.width())
   assert wizard.status.height() >= wizard.status.heightForWidth(wizard.status.width())
   row = next(row for row in range(wizard.preview.rowCount()) if wizard.preview.item(row, 0).text().startswith('OTF_'))
   gray = QColor(active_palette()['fg_dim'])
   assert wizard.preview.item(row, 0).foreground().color() == gray
   assert gray != wizard.palette().color(QPalette.ColorGroup.Active, QPalette.ColorRole.Text)
   image = wizard.preview.viewport().grab().toImage()
   rect = wizard.preview.visualItemRect(wizard.preview.item(row, 0)).intersected(image.rect())
   assert any(pixel.alpha() > 0 and pixel.getRgb()[:3] == gray.getRgb()[:3]
    for pixel in (image.pixelColor(x, y) for y in range(rect.top(), rect.bottom())
     for x in range(rect.left(), rect.right())))
   QTest.mouseClick(wizard.next, Qt.LeftButton)
   state['stage'] = 2
  elif state['stage'] == 2:
   body = host._field_overview
   state['body'] = body
   if not body._normalization_ready:
    return
   assert body.isVisible() and not body.isWindow() and not isinstance(body, QDialog)
   assert not body.property('spacrGlassed') and body.window() is window
   assert body.preview_resolution.isVisible()
   assert host._settings_scroll.widget() is body._settings_content
   assert body._settings_content.property('spacrTransparent') is True
   assert host._field_overview_workspace.stack.property('spacrTransparent') is True
   if sys.argv[1] == 'mask':
    assert (body.wells.rowCount(), body.wells.columnCount()) == (16, 24)
    assert body.wells.viewport().height() >= 400
    assert body.wells.rowHeight(0) >= 22
   QTest.mouseClick(body.options_button, Qt.LeftButton)
   state['stage'] = 3
  elif state['stage'] == 3:
   body = host._field_overview
   assert isinstance(body._options_window, QDialog) and body._options_window.isWindow()
   assert body._options_window.isVisible()
   body._options_window.close()
   assert host._settings_scroll.widget() is body._settings_content
   replacement = _open_field_overview(host, {'source': sys.argv[2] + '/missing'})
   state['replacement'] = replacement
   assert host._field_overview_workspace.stack.currentWidget() is replacement
   replacement._runner.shutdown(timeout_ms=3000)
   replacement.reject()
   assert host._field_overview_workspace.stack.currentWidget() is body
   assert host._settings_scroll.widget() is body._settings_content
   host._field_overview_workspace._back()
   assert host._settings_scroll.widget() is state['original_settings']
   assert state['console'] is (host._console if hasattr(host, '_console') else host._masks_console.console)
   if hasattr(host, '_btn_run'):
    assert host._btn_run.isVisible() and host._settings_search.isVisible()
   assert _open_field_overview(host, {}) is wizard
   assert body.isVisible() and host._settings_scroll.widget() is body._settings_content
   finish(0)
   print('embedded widgets survive actual application dialog filters', flush=True)
 except Exception:
  fail()
QTimer.singleShot(0, open_empty)
timer = QTimer()
timer.timeout.connect(tick)
timer.start(50)
QTimer.singleShot(20000, lambda: finish(2))
sys.exit(app.exec())
'''
    environment = dict(os.environ, XDG_CONFIG_HOME=str(tmp_path / "config"),
        XDG_CACHE_HOME=str(tmp_path / "cache"), SPACR_NO_NETWORK="1", SPACR_PREWARM="0",
        SPACR_ISOLATED_PROFILE="1", SPACR_HOME=str(tmp_path / "profile"))
    try:
        result = subprocess.run([sys.executable, "-c", code, host_key, str(tmp_path)],
            env=environment, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired as error:
        stdout = error.output or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        pytest.fail(f"Embedded {host_key} exceeded the unchanged {error.timeout}s budget."
                    f"\nCaptured stdout:\n{stdout}\nCaptured stderr:\n{stderr}", pytrace=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "embedded widgets survive actual application dialog filters" in result.stdout


def test_preview_resolution_caps_plate_reads_and_progressive_field_pixels(tmp_path, qtbot, monkeypatch):
    raw = np.arange(1024 * 1024, dtype=np.uint16).reshape(1024, 1024)
    path = tmp_path / "plate1_A01_F1.npy"
    np.save(path, raw)
    before = path.stat().st_mtime_ns
    dialog = _browser(tmp_path, qtbot, browser._overview_index(tmp_path))
    calls = []
    original = browser._overview_pixels
    def pixels(*args, **kwargs):
        calls.append(kwargs.get("edge"))
        return original(*args, **kwargs)
    monkeypatch.setattr(browser, "_overview_pixels", pixels)
    dialog.preview_resolution.setCurrentIndex(dialog.preview_resolution.findData(128))
    dialog.show()
    qtbot.waitUntil(lambda: "A01" in dialog._installed_thumbnails and not dialog._thumbnail_jobs.is_busy())
    assert calls and all(edge <= 128 for edge in calls)
    dialog._well(dialog.wells.item(0, 0))
    qtbot.waitUntil(lambda: dialog._displayed_montage is not None)
    assert all(edge <= 128 for edge in calls)
    dialog.fields.setCurrentRow(0)
    qtbot.waitUntil(lambda: dialog.view._item is not None and not dialog._image_jobs.is_busy()
        and dialog.view._item.pixmap().width() == 128)
    calls.clear()
    dialog.preview_resolution.setCurrentIndex(dialog.preview_resolution.findData(512))
    qtbot.waitUntil(lambda: dialog.view._item is not None and not dialog._image_jobs.is_busy()
        and dialog.view._item.pixmap().width() == 512)
    assert 256 in calls and 512 in calls
    assert path.stat().st_mtime_ns == before
    np.testing.assert_array_equal(np.load(path), raw)
    dialog.close()


@pytest.mark.parametrize("compression,expected_decodes", ((None, 0), ("deflate", 1)))
def test_multichannel_tiff_thumbnail_decodes_only_one_shared_page_when_required(tmp_path, monkeypatch, compression, expected_decodes):
    raw = np.arange(256 * 384 * 3, dtype=np.uint16).reshape(256, 384, 3)
    path = tmp_path / "plate1_A01_F1.tif"
    tifffile.imwrite(path, raw, photometric="rgb", compression=compression)
    field, = browser._overview_index(tmp_path)
    calls = []
    original = tifffile.TiffPage.asarray
    def decoded(page, *args, **kwargs):
        calls.append(page.shape)
        return original(page, *args, **kwargs)
    monkeypatch.setattr(tifffile.TiffPage, "asarray", decoded)
    planes = tuple(browser._overview_raw_planes(field, (0, 1, 2), edge=48))
    for channel, _, plane in planes:
        np.testing.assert_array_equal(plane, raw[::8, ::8, channel])
    assert len(calls) == expected_decodes
    calls.clear()
    region = (23, 57, 31, 89)
    planes = tuple(browser._overview_raw_planes(field, (0, 1, 2), region=region))
    for channel, _, plane in planes:
        np.testing.assert_array_equal(plane, raw[23:57, 31:89, channel])
    assert len(calls) == expected_decodes


@pytest.mark.parametrize("cancel", ("escape", "close"))
def test_embedded_metadata_keyboard_and_close_emit_one_hidden_cancel(tmp_path, qtbot, cancel):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QDialog
    wizard = browser._OverviewMetadataWizard(str(tmp_path))
    wizard._startup_timer.stop()
    qtbot.addWidget(wizard)
    events = []
    wizard.rejected.connect(lambda: events.append(("rejected", wizard.isVisible())))
    wizard.finished.connect(lambda result: events.append((result, wizard.isVisible())))
    wizard.show()
    qtbot.keyClick(wizard, Qt.Key_F1)
    assert wizard.isVisible() and not events
    if cancel == "escape":
        qtbot.keyClick(wizard, Qt.Key_Escape)
    else:
        wizard.close()
    assert events == [("rejected", False), (QDialog.Rejected, False)]
    assert wizard.result() == QDialog.Rejected and wizard._cancelled.is_set()
    wizard.reject()
    assert len(events) == 2
    wizard._runner.shutdown(timeout_ms=3000)


@pytest.mark.parametrize("invalid_well", (False, True))
def test_cellvoyager_calibrations_are_gray_excluded_and_cannot_mask_invalid_matches(tmp_path, qtbot, invalid_well):
    from PySide6.QtGui import QColor
    from spacr.qt.theme import active_palette
    for channel in (1, 2, 3, 4):
        tifffile.imwrite(tmp_path / f"plate1_A01_T0001F001L01A01Z01C{channel:02}.tif",
            np.full((5, 7), channel, np.uint16))
    calibration = tmp_path / "OTF_DC_DCAM#1_CAM1.tif"
    tifffile.imwrite(calibration, np.zeros((4, 4), np.uint16))
    if invalid_well:
        tifffile.imwrite(tmp_path / "plate1_Z00_T0001F001L01A01Z01C01.tif", np.ones((5, 7), np.uint16))
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}
    wizard = browser._OverviewMetadataWizard(str(tmp_path))
    qtbot.addWidget(wizard)
    qtbot.waitUntil(lambda: wizard._info is not None)
    assert wizard.preset.currentData() == "cellvoyager"
    assert wizard._info["channels"] == (1, 2, 3, 4)
    assert wizard.next.isEnabled() is (not invalid_well)
    assert all(path != str(calibration) for field in wizard._fields for _, path in field.files)
    assert wizard.preview.columnCount() == 9
    rows = {wizard.preview.item(row, 0).text(): row for row in range(wizard.preview.rowCount())}
    row = rows[calibration.name]
    gray = QColor(active_palette()["fg_dim"])
    assert all(wizard.preview.item(row, column).foreground().color() == gray for column in range(9))
    assert all(wizard.preview.item(row, column).text() == "" for column in range(1, 9))
    row = rows["plate1_A01_T0001F001L01A01Z01C01.tif"]
    assert wizard.preview.item(row, 3).text() == "A"
    assert wizard.preview.item(row, 4).text() == "1"
    if invalid_well:
        assert "valid wellID" in wizard.status.text()
        with pytest.raises(ValueError):
            wizard.configuration()
    else:
        assert "1 unmatched filenames excluded" in wizard.status.text()
        accepted = []
        wizard.accepted.connect(lambda: accepted.append(wizard.configuration()))
        wizard.next.click()
        assert len(accepted) == 1
        assert accepted[0]["channels"] == (1, 2, 3)
    assert all(path.read_bytes() == raw for path, raw in before.items())
    wizard.reject()
    wizard._runner.shutdown(timeout_ms=3000)


def test_visible_thumbnail_batches_bound_handoffs_and_keep_per_well_errors(tmp_path, qtbot, monkeypatch):
    for row in "ABCD":
        for column in range(1, 7):
            np.save(tmp_path / f"plate1_{row}{column:02}_F1.npy",
                np.roll(np.arange(256, dtype=np.uint16).reshape(16, 16), column, axis=1))
    fields = browser._overview_index(tmp_path)
    dialog = _browser(tmp_path, qtbot, fields, options={"geometry": (4, 6)})
    (tmp_path / "plate1_A01_F1.npy").unlink()
    batches = []
    original = dialog._thumbnail_jobs.submit
    def submitted(read, done):
        def completed(results):
            batches.append(len(results))
            done(results)
        return original(read, completed)
    monkeypatch.setattr(dialog._thumbnail_jobs, "submit", submitted)
    dialog.show()
    qtbot.waitUntil(lambda: len(dialog._installed_thumbnails) == 23 and not dialog._thumbnail_jobs.is_busy())
    assert all(1 <= count <= 8 for count in batches) and max(batches) == 8
    assert len(batches) < 12
    assert dialog.wells.item(0, 0).icon().isNull()
    assert "A01" in dialog.status.text() and "Could not read" in dialog.status.text()
    assert all(not dialog.wells.item(row, column).icon().isNull()
        for row in range(4) for column in range(6) if (row, column) != (0, 0))
    for field in fields:
        key = dialog.wells.item(*dialog._well_positions[field.well]).data(browser.Qt.UserRole + 1)
        pixels = dialog._thumbnail_cache[key]
        if field.well != "A01":
            np.testing.assert_array_equal(pixels, browser._overview_pixels(field, (0,), (1., 99.9), edge=key[4]))
    dialog.close()


def test_cancelled_thumbnail_batch_stops_before_reading_the_next_well(tmp_path, qtbot, monkeypatch):
    for column in range(1, 13):
        np.save(tmp_path / f"plate1_A{column:02}_F1.npy", np.arange(64).reshape(8, 8))
    dialog = _browser(tmp_path, qtbot, browser._overview_index(tmp_path))
    started, release = threading.Event(), threading.Event()
    calls = []
    original = browser._overview_pixels
    def read(field, *args, **kwargs):
        calls.append(field.well)
        started.set()
        assert release.wait(5)
        return original(field, *args, **kwargs)
    monkeypatch.setattr(browser, "_overview_pixels", read)
    try:
        dialog.show()
        qtbot.waitUntil(started.is_set)
        dialog._cancel()
        release.set()
        qtbot.waitUntil(lambda: not dialog._thumbnail_jobs.is_busy())
        assert len(calls) == 1 and not dialog._installed_thumbnails
    finally:
        release.set()
        dialog.close()


def test_late_comparison_cannot_replace_a_field_opened_while_it_loads(tmp_path, qtbot, monkeypatch):
    dialog = _browser(tmp_path, qtbot)
    delivered = []
    submit = dialog._montage_jobs.submit
    def submitted(read, done):
        def completed(result):
            delivered.append(True)
            return done(result)
        return submit(read, completed)
    monkeypatch.setattr(dialog._montage_jobs, "submit", submitted)
    for index in (0, 1):
        dialog.fields.item(index).setSelected(True)
    started, release = threading.Event(), threading.Event()
    original = browser._overview_montage
    def montage(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(browser, "_overview_montage", montage)
    try:
        dialog._compare()
        qtbot.waitUntil(started.is_set)
        dialog.fields.setCurrentRow(0)
        qtbot.waitUntil(lambda: dialog.view._item is not None and not dialog._image_jobs.is_busy())
        before = dialog.view._item.pixmap().toImage().copy()
        release.set()
        qtbot.waitUntil(lambda: bool(delivered) and not dialog._montage_jobs.is_busy())
        assert dialog._individual and dialog._comparison_signature is None
        assert dialog.view._item.pixmap().toImage() == before
    finally:
        release.set()
        dialog.close()


def test_late_field_thumbnail_error_cannot_overwrite_another_plate_status(tmp_path, qtbot, monkeypatch):
    dialog = _browser(tmp_path, qtbot)
    delivered = []
    submit = dialog._field_thumbnail_jobs.submit
    def submitted(read, done):
        def completed(result):
            delivered.append(True)
            return done(result)
        return submit(read, completed)
    monkeypatch.setattr(dialog._field_thumbnail_jobs, "submit", submitted)
    started, release = threading.Event(), threading.Event()
    original = browser._overview_pixels
    def read(field, *args, **kwargs):
        if field.plate == "plate1" and kwargs.get("edge") == 128:
            started.set()
            assert release.wait(5)
            raise ValueError("previous plate thumbnail read failed")
        return original(field, *args, **kwargs)
    monkeypatch.setattr(browser, "_overview_pixels", read)
    try:
        dialog.show()
        dialog._well(dialog.wells.item(0, 0))
        qtbot.waitUntil(started.is_set)
        dialog.plates.setCurrentIndex(dialog.plates.findData("plate2"))
        release.set()
        qtbot.waitUntil(lambda: bool(delivered) and not dialog._field_thumbnail_jobs.is_busy())
        assert dialog.plates.currentData() == "plate2"
        assert "previous plate thumbnail read failed" not in dialog.status.text()
    finally:
        release.set()
        dialog.close()
