"""Real field images exercise the shared overview without a model or GPU."""
import json
import math
import threading

import numpy as np
import pytest
import tifffile

from spacr.qt.widgets import qc_field_browser as browser


@pytest.fixture
def plate_images(tmp_path):
    for plate in ("plate1", "plate2"):
        for well in ("A01", "B02"):
            for field in range(1, 4):
                folder = tmp_path / plate / well / f"field{field}"
                folder.mkdir(parents=True)
                image = np.arange(120, dtype=np.uint16).reshape(10, 12)
                for channel in (1, 2):
                    tifffile.imwrite(folder / f"ch{channel}.tif", image * channel)
    tifffile.imwrite(tmp_path / "unassigned.tif", np.zeros((6, 8), np.uint8))
    return tmp_path


def _ready_overview(dialog, fields, qtbot):
    dialog._indexed(fields)
    qtbot.waitUntil(lambda: dialog._normalization_ready and not dialog._index_jobs.is_busy(), timeout=10000)


def _retire_overview(dialog, qtbot):
    dialog._cancel()
    runners = (dialog._index_jobs, dialog._image_jobs, dialog._export_jobs,
               dialog._thumbnail_jobs, dialog._montage_jobs, dialog._field_thumbnail_jobs,
               dialog._metadata_jobs, dialog._measurement_jobs, dialog._native_jobs)
    qtbot.waitUntil(lambda: all(runner.active_jobs() == 0 for runner in runners), timeout=10000)
    for runner in runners:
        runner.shutdown(timeout_ms=3000)
    from spacr.qt.bridge import prune_parked_threads
    qtbot.waitUntil(lambda: prune_parked_threads() == 0)


def _retire_overview_owner(owner, qtbot):
    wizard = getattr(owner, "_field_overview_wizard", None)
    if wizard is not None:
        wizard._cancelled.set()
        wizard._runner.cancel()
        qtbot.waitUntil(lambda: wizard._runner.active_jobs() == 0, timeout=10000)
        wizard._runner.shutdown(timeout_ms=3000)
    dialog = getattr(owner, "_field_overview", None)
    if dialog is not None:
        _retire_overview(dialog, qtbot)


def _magnify_plate(dialog):
    while (dialog._plate_cell_extent or (0, 0))[0] < 180:
        dialog._zoom_plate(1)


def test_folder_index_never_decodes_or_invents_wells(plate_images, monkeypatch):
    monkeypatch.setattr(tifffile.TiffFile, "asarray", lambda *_: pytest.fail("metadata index decoded pixels"))
    fields = browser._overview_index(plate_images)
    assert len(fields) == 13
    assert sum(field.unresolved for field in fields) == 1
    resolved = [field for field in fields if not field.unresolved]
    assert {field.plate for field in resolved} == {"plate1", "plate2"}
    assert {(field.well, field.row, field.column) for field in resolved} == {("A01", "1", 1), ("B02", "2", 2)}
    assert all(len(field.files) == 2 for field in resolved)
    assert next(field for field in fields if field.unresolved).well == ""
    cancelled = threading.Event()
    cancelled.set()
    assert browser._overview_index(plate_images, cancelled=cancelled) == ()


def test_regex_and_multichannel_pixels_are_source_bound(tmp_path):
    image = np.stack([np.arange(120).reshape(10, 12), np.ones((10, 12)) * 5], axis=-1).astype(np.uint16)
    np.save(tmp_path / "p9-A03-site8.npy", image)
    pattern = r"(?P<plateID>p\d+)-(?P<wellID>[A-Z]\d+)-site(?P<fieldID>\d+)"
    field, = browser._overview_index(tmp_path, pattern)
    assert (field.plate, field.well, field.field) == ("p9", "A03", "8")
    rgb = browser._overview_pixels(field, (0, 1), (0, 100))
    assert rgb.shape == (10, 12, 3)
    assert rgb.dtype == np.uint8
    assert rgb[0, 0, 0] == 0 and rgb[-1, -1, 0] == 255
    assert not rgb[..., 1:].any()
    assert browser._overview_pixels(field, (0,), (10, 90))[1, 0, 0] != rgb[1, 0, 0]
    with pytest.raises(ValueError, match=r"Channel 2 is missing or ambiguous\. Available channels: 0, 1"):
        browser._overview_pixels(field, (2,))


def test_grouped_random_samples_keep_plates_distinct_and_reproducible(plate_images):
    fields = tuple(f for f in browser._overview_index(plate_images) if not f.unresolved)
    selected = browser._overview_sample(fields, 2, 37, "well", True)
    assert len(selected) == 8
    assert selected == browser._overview_sample(tuple(reversed(fields)), 2, 37, "well", True)
    assert len({f.identity for f in selected}) == 8
    assert len(browser._overview_sample(fields, 1, grouping="plate")) == 2
    assert len(browser._overview_sample(fields, 1, grouping="row")) == 4
    assert len(browser._overview_sample(fields, 1, grouping="column")) == 4


def test_navigation_is_metadata_only_and_decodes_one_selected_field(plate_images, qtbot, monkeypatch):
    dialog = browser._FieldOverview()
    qtbot.addWidget(dialog, before_close_func=lambda widget: _retire_overview(widget, qtbot))
    fields = browser._overview_index(plate_images)
    calls = []
    original = browser._overview_pixels
    def read(field, *args, **kwargs):
        calls.append(field.identity)
        return original(field, *args, **kwargs)
    monkeypatch.setattr(browser, "_overview_pixels", read)
    dialog._image_jobs._threaded = False
    dialog.channels.setText("1,2")
    _ready_overview(dialog, fields, qtbot)
    assert calls == []
    dialog.plates.setCurrentIndex(dialog.plates.findData("plate1"))
    item = dialog.wells.item(0, 0)
    dialog._well(item)
    assert len(dialog._visible) == 3
    assert all(field.plate == "plate1" and field.well == "A01" for field in dialog._visible)
    dialog.fields.setCurrentRow(0)
    dialog._field(dialog.fields.item(0))
    assert len(calls) == 2
    dialog._field(dialog.fields.item(0))
    assert len(calls) == 2
    dialog._back()
    assert len(dialog._visible) == 6
    assert dialog.plates.currentData() == "plate1"
    assert len(dialog._cache) == 2
    dialog.close()


def test_real_pdf_and_selection_provenance(plate_images, tmp_path, monkeypatch):
    from spacr import plot
    original_save = plot.save_figure
    registered = []

    def save_registered(figure, *args, **kwargs):
        registered.append((figure._spacr_image, figure._spacr_spec,
                           [np.asarray(image.get_array()) for axis in figure.axes
                            for image in axis.get_images()]))
        return original_save(figure, *args, **kwargs)

    monkeypatch.setattr(plot, "save_figure", save_registered)
    fields = tuple(f for f in browser._overview_index(plate_images) if not f.unresolved)
    selected = browser._overview_sample(fields, 1, 15, "well", True)
    output = tmp_path / "pdf"
    paths, errors = browser._overview_export(selected, output, (1, 2), (1, 99), 15)
    assert len(paths) == 1 and not errors
    assert (output / "overview_0001.pdf").read_bytes().startswith(b"%PDF-")
    provenance = json.loads((output / "overview_selection.json").read_text())
    assert provenance["seed"] == 15
    assert provenance["channels"] == [1, 2]
    assert [(row["plate"], row["well"], row["field"]) for row in provenance["fields"]] == [
        (field.plate, field.well, field.field) for field in selected]
    images, spec, drawn = registered[0]
    assert len(registered) == 1 and len(images) == len(drawn) == len(selected)
    for image, actual in zip(images, drawn):
        np.testing.assert_array_equal(image, actual)
    assert spec["kind"] == "montage"
    assert spec["fields"] == [{"plate": field.plate, "well": field.well,
                              "field": field.field, "files": field.files,
                              "field_id": field.field_id or field.field,
                              "time_id": field.time_id, "z_id": field.z_id,
                              "condition": "", "time_index": None, "z_mode": "plane",
                              "z_index": None, "annotations": {}}
                             for field in selected]
    assert spec["channels"] == (1, 2) and spec["percentiles"] == (1, 99)
    assert spec["seed"] == 15 and spec["image_panels"] == list(range(len(selected)))
    assert spec["preview_edge"] == 1000 and spec["original_resolution"] is False


def test_channel_guidance_belongs_to_its_setting_label(qtbot):
    from spacr.qt.screens.settings_model import _sibling_label_for
    dialog = browser._FieldOverview()
    qtbot.addWidget(dialog, before_close_func=lambda widget: _retire_overview(widget, qtbot))
    label = _sibling_label_for(dialog.channels)
    assert label is not None
    assert not dialog.channels.toolTip()
    assert label.toolTip() == browser.tr(
        "Use filename channel IDs for separate files; use zero-based indices for multichannel images.")
    assert label.property("settingHelpLabel")
    dialog.close()


def test_padded_filename_channels_decode_and_reject_duplicate_channels(tmp_path):
    for channel in ("01", "02"):
        tifffile.imwrite(tmp_path / f"plate1_A01_T0001F001L01C{channel}.tif",
                         np.arange(48, dtype=np.uint16).reshape(6, 8))
    field, = browser._overview_index(tmp_path)
    assert [channel for channel, _ in field.files] == ["1", "2"]
    assert browser._overview_pixels(field, (1, 2)).shape == (6, 8, 3)
    tifffile.imwrite(tmp_path / "plate1_A01_T0001F001L01C1.tif", np.zeros((6, 8), np.uint16))
    duplicate, = browser._overview_index(tmp_path)
    with pytest.raises(ValueError, match="ambiguous"):
        browser._overview_pixels(duplicate, (1,))


def test_reopening_cancelled_overview_keeps_navigation_and_decodes(plate_images, qtbot):
    from PySide6.QtWidgets import QWidget
    owner = QWidget()
    qtbot.addWidget(owner, before_close_func=lambda widget: _retire_overview_owner(widget, qtbot))
    wizard = browser._open_field_overview(owner, {"source": str(plate_images / "plate1")})
    wizard.scope.setCurrentIndex(wizard.scope.findData("recursive"))
    qtbot.waitUntil(wizard.next.isEnabled, timeout=10000)
    wizard.channels.setText("1,2")
    configuration = wizard.configuration()
    assert configuration["channels"] == (1, 2)
    qtbot.mouseClick(wizard.next, browser.Qt.LeftButton)
    assert hasattr(owner, "_field_overview"), wizard.status.text()
    dialog = owner._field_overview
    dialog._image_jobs._threaded = False
    qtbot.waitUntil(lambda: dialog._normalization_ready and not dialog._index_jobs.is_busy(), timeout=10000)
    assert dialog._fields == wizard._fields
    assert dialog.plates.currentData() == str(plate_images / "plate1")
    dialog._well(dialog.wells.item(0, 0))
    _retire_overview(dialog, qtbot)
    dialog.close()
    assert dialog._cancelled.is_set()
    reopened = owner._field_overview
    reopened.show()
    assert reopened is dialog
    assert reopened._scope == (str(plate_images / "plate1"), "A01")
    assert not reopened._cancelled.is_set()
    qtbot.waitUntil(lambda: reopened._normalization_ready and not reopened._index_jobs.is_busy(), timeout=10000)
    reopened.fields.setCurrentRow(0)
    reopened._field(reopened.fields.item(0))
    assert reopened.view._item is not None
    assert len(reopened._cache) == 2
    _retire_overview(reopened, qtbot)
    reopened.close()


def test_explicit_well_exports_all_fields_while_sample_mode_limits(plate_images, qtbot):
    dialog = browser._FieldOverview()
    def retire(widget):
        """Stop only this overview's queued visuals and owned workers before close."""
        widget._visual_timer.stop()
        _retire_overview(widget, qtbot)
    qtbot.addWidget(dialog, before_close_func=retire)
    dialog._indexed(browser._overview_index(plate_images))
    dialog.plates.setCurrentIndex(dialog.plates.findData("plate1"))
    dialog.use_selection.setChecked(True)
    dialog.wells.item(0, 0).setSelected(True)
    selected = dialog._export_selection()
    assert len(selected) == 3 and all(f.well == "A01" for f in selected)
    dialog.sample_selected.setChecked(True)
    dialog.random.setChecked(True)
    assert len(dialog._export_selection()) == 1
    dialog.sample_selected.setChecked(False)
    dialog.use_selection.setChecked(False)
    dialog.grouping.setCurrentIndex(dialog.grouping.findData("well"))
    assert len(dialog._export_selection()) == 2
    dialog.use_selection.setChecked(True)
    dialog.fields.item(0).setSelected(True)
    assert dialog._export_selection() == (dialog.fields.item(0).data(browser.Qt.UserRole),)
    retire(dialog)
    dialog.close()


def test_cancelled_export_preserves_completed_pages_and_refuses_overwrite(plate_images, tmp_path, monkeypatch):
    from spacr import plot
    fields = tuple(f for f in browser._overview_index(plate_images) if not f.unresolved)
    fields = fields + (fields[0],)
    cancelled = threading.Event()
    original = plot.save_figure
    def save(*args, **kwargs):
        result = original(*args, **kwargs)
        cancelled.set()
        return result
    monkeypatch.setattr(plot, "save_figure", save)
    folder = tmp_path / "partial"
    paths, errors = browser._overview_export(fields, folder, (1, 2), (1, 99), cancelled=cancelled)
    assert len(paths) == 1 and not errors
    manifest = folder / "overview_selection.json"
    before = manifest.read_bytes()
    partial = json.loads(before)
    assert len(partial["fields"]) == 12
    assert partial["cancelled"] is True
    assert partial["requested_field_count"] == 13
    assert partial["written_field_count"] == 12
    with pytest.raises(ValueError, match="already exist"):
        browser._overview_export(fields[:1], folder, (1,), None)
    assert manifest.read_bytes() == before
    assert len(list(folder.glob("*.pdf"))) == 1


def test_large_metadata_pages_and_rgb_cache_stay_bounded(plate_images, qtbot):
    from dataclasses import replace
    actual = next(f for f in browser._overview_index(plate_images) if not f.unresolved)
    fields = tuple(replace(actual, field=str(index), field_id=str(index)) for index in range(100))
    dialog = browser._FieldOverview()
    qtbot.addWidget(dialog, before_close_func=lambda widget: _retire_overview(widget, qtbot))
    dialog._image_jobs._threaded = False
    _ready_overview(dialog, fields, qtbot)
    assert dialog.fields.count() == 24
    assert not dialog._cache
    for index in range(20):
        dialog.fields.setCurrentRow(index)
        dialog._field(dialog.fields.item(index))
    assert len(dialog._cache) == 8
    assert all(value.shape == (10, 12, 3) for value in dialog._cache.values())
    dialog._turn(1)
    assert dialog._page == 1 and dialog.fields.count() == 24
    assert dialog.fields.item(0).data(browser.Qt.UserRole) == fields[24]
    dialog.use_selection.setChecked(True)
    assert len(dialog._export_selection()) == 1
    assert dialog._export_selection()[0] == fields[19]
    dialog._clear_selection()
    assert dialog._export_selection() == ()
    dialog.close()


def test_rapid_field_changes_coalesce_behind_one_real_image_read(plate_images, qtbot, monkeypatch):
    dialog = browser._FieldOverview()
    qtbot.addWidget(dialog, before_close_func=lambda widget: _retire_overview(widget, qtbot))
    fields = tuple(f for f in browser._overview_index(plate_images) if not f.unresolved)
    _ready_overview(dialog, fields, qtbot)
    started, release = threading.Event(), threading.Event()
    original = browser._overview_pixels
    calls = []
    def read(field, *args, **kwargs):
        calls.append(field.identity)
        pixels = original(field, *args, **kwargs)
        if len(calls) == 1:
            started.set()
            assert release.wait(5)
        return pixels
    monkeypatch.setattr(browser, "_overview_pixels", read)
    try:
        dialog.fields.setCurrentRow(0)
        dialog._field(dialog.fields.item(0))
        qtbot.waitUntil(started.is_set)
        dialog.fields.setCurrentRow(1)
        dialog._field(dialog.fields.item(1))
        dialog.fields.setCurrentRow(2)
        dialog._field(dialog.fields.item(2))
        assert len(calls) == 1
        release.set()
        qtbot.waitUntil(lambda: len(calls) == 3 and not dialog._image_jobs.is_busy())
        assert calls[-2:] == [dialog.fields.item(2).data(browser.Qt.UserRole).identity] * 2
        assert dialog.view._item is not None
    finally:
        release.set()
        dialog.close()



def test_arbitrary_plate_folder_names_never_merge_their_fields(tmp_path):
    for plate in ("untreated", "treated"):
        folder = tmp_path / plate / "A01" / "field1"
        folder.mkdir(parents=True)
        tifffile.imwrite(folder / "ch1.tif", np.zeros((5, 6), np.uint8))
    fields = browser._overview_index(tmp_path)
    assert len(fields) == 2
    assert {field.plate for field in fields} == {str(tmp_path / "untreated"), str(tmp_path / "treated")}
    assert all(not field.unresolved and len(field.files) == 1 for field in fields)


@pytest.mark.parametrize("module,object_name", [
    ("mask", "GenerateMasksFieldOverview"), ("make_masks", "MakeMasksFieldOverview"),
])
def test_both_mask_modules_open_the_same_lazy_overview(qtbot, monkeypatch, tmp_path, module, object_name):
    from PySide6.QtWidgets import QAbstractButton, QToolButton
    from spacr.qt.i18n import tr
    from spacr.qt.widgets import ambient
    monkeypatch.setattr(ambient, "_flow_graphics_possible", lambda: False)
    source_folder = tmp_path / "source"
    source_folder.mkdir()
    np.save(source_folder / "plate1_A01_F1.npy", np.stack([
        np.arange(120).reshape(10, 12)] * 4, axis=-1).astype(np.uint16))
    source_path = str(source_folder)
    if module == "mask":
        from spacr.qt.screens.app_screen import AppScreen
        screen = AppScreen("mask")
        source = screen._settings_model._widgets["src"]
        source.setText(source_path)
        expected = screen._settings_model.collect()
    else:
        from spacr.qt.screens.make_masks import MakeMasksScreen
        screen = MakeMasksScreen()
        screen._folder = source_path
        screen._canvas.norm_lo, screen._canvas.norm_hi = 3., 97.
        expected = {"src": source_path, "normalization_percentiles": (3., 97.)}
    qtbot.addWidget(screen)
    button = screen.findChild(QToolButton, object_name)
    assert button is not None
    assert len(screen.findChildren(QAbstractButton, object_name)) == 1
    assert not button.text() and not button.icon().isNull()
    assert button.accessibleName() == tr("Field-of-view overview")
    assert tr("Field-of-view overview") in button.toolTip()
    if module == "mask":
        assert button.parentWidget() is source
        actions = source.actions()
        folder = next(action for action in actions if action.objectName() == "SourceFolderBrowse")
        assert actions.index(button.defaultAction()) == actions.index(folder) + 1
        cloud = next(action for action in actions if action.objectName() == "CloudSourceBrowse")
        assert actions.index(cloud) == actions.index(button.defaultAction()) + 1
        assert not screen._actions_row.isAncestorOf(button)
    else:
        row = screen._io_group.layout().itemAt(1).layout()
        assert row.itemAt(row.indexOf(screen._btn_open) + 1).widget() is button
        assert not screen._view_controls.isAncestorOf(button)
    button.click()
    pages = screen._fold_pages
    assert pages.currentWidget() is screen._overview_tab and pages.indexOf(screen._overview_tab) == 1
    qtbot.addWidget(screen._overview_tab)
    wizard = screen._overview_tab._overview_page.wizard
    assert isinstance(wizard, browser._OverviewMetadataWizard)
    assert wizard.source.text() == source_path
    qtbot.waitUntil(wizard.next.isEnabled, timeout=10000)
    channels = expected.get("channels") or (0,)
    wizard.channels.setText(",".join(str(value) for value in channels))
    configuration = wizard.configuration()
    assert configuration["channels"] == tuple(channels)
    qtbot.mouseClick(wizard.next, browser.Qt.LeftButton)
    dialog = screen._overview_tab._overview_page.browser
    assert dialog is not None, wizard.status.text()
    assert isinstance(dialog, browser._FieldOverview)
    assert dialog.source.text() == source_path
    channels = expected.get("channels") or (0,)
    assert dialog.channels.text() == ",".join(str(value) for value in channels)
    assert dialog.normalise.isChecked() == bool(expected.get("normalize", True))
    assert (dialog.low.value(), dialog.high.value()) == tuple(expected.get("normalization_percentiles") or (1., 99.9))
    assert dialog._fields == wizard._fields and dialog._fields
    assert not dialog._cache
    qtbot.waitUntil(lambda: dialog._normalization_ready and not dialog._index_jobs.is_busy(), timeout=10000)
    _retire_overview(dialog, qtbot)
    dialog.close()


def test_source_change_discards_late_image_from_previous_generation(plate_images, tmp_path, qtbot, monkeypatch):
    dialog = browser._FieldOverview()
    qtbot.addWidget(dialog, before_close_func=lambda widget: _retire_overview(widget, qtbot))
    _ready_overview(dialog, tuple(f for f in browser._overview_index(plate_images) if not f.unresolved), qtbot)
    started, release = threading.Event(), threading.Event()
    original = browser._overview_pixels
    def read(*args, **kwargs):
        pixels = original(*args, **kwargs)
        started.set()
        assert release.wait(5)
        return pixels
    monkeypatch.setattr(browser, "_overview_pixels", read)
    next_source = tmp_path / "new_source"
    next_source.mkdir()
    tifffile.imwrite(next_source / "other_A01_F1.tif", np.ones((4, 5), np.uint8))
    try:
        dialog.fields.setCurrentRow(0)
        dialog._field(dialog.fields.item(0))
        qtbot.waitUntil(started.is_set)
        dialog.source.setText(str(next_source))
        dialog._scan()
        qtbot.waitUntil(lambda: bool(dialog._fields) and dialog._fields[0].plate == "other")
        release.set()
        qtbot.waitUntil(lambda: not dialog._image_jobs.is_busy())
        assert not dialog._cache
        assert dialog.view._item is None
        assert all(field.plate == "other" for field in dialog._fields)
    finally:
        release.set()
        dialog.close()


def test_unreadable_field_is_traceable_in_pdf_instead_of_dropping_selection(tmp_path):
    broken = tmp_path / "broken_A01_F1.tif"
    broken.write_bytes(b"not a TIFF")
    field, = browser._overview_index(tmp_path)
    paths, errors = browser._overview_export((field,), tmp_path / "errors", (0,), (1, 99))
    assert len(paths) == 1
    assert len(errors) == 1 and errors[0][0] == field.caption
    manifest = json.loads((tmp_path / "errors" / "overview_selection.json").read_text())
    assert manifest["fields"][0]["files"] == [["", str(broken)]]
    assert len(manifest["errors"]) == 1



def test_identical_filename_plate_ids_keep_physical_screen_scopes(tmp_path):
    for screen in ("screenA", "screenB"):
        for well in ("A01", "B02"):
            for channel in ("01", "02"):
                folder = tmp_path / screen / "plate1" / well / "field1" / f"ch{channel}"
                folder.mkdir(parents=True)
                tifffile.imwrite(folder / f"plate1_{well}_T0001F001L01C{channel}.tif",
                                 np.arange(30, dtype=np.uint16).reshape(5, 6))
    fields = browser._overview_index(tmp_path)
    assert len(fields) == 4
    assert {f.plate for f in fields} == {"screenA/plate1", "screenB/plate1"}
    assert all(len(f.files) == 2 for f in fields)
    assert len(browser._overview_sample(fields, 1, grouping="plate")) == 2
    assert all(browser._overview_pixels(f, (1, 2)).shape == (5, 6, 3) for f in fields)



def test_fourth_channel_changes_shared_preview_and_pdf_pixels(tmp_path):
    array = np.zeros((12, 16, 4), dtype=np.uint16)
    array[..., :3] = np.arange(192).reshape(12, 16, 1)
    path = tmp_path / "four_A01_F1.npy"
    np.save(path, array)
    field, = browser._overview_index(tmp_path)
    before = browser._overview_pixels(field, (0, 1, 2, 3), (0, 100), edge=120)
    array[..., 3] = np.arange(192).reshape(12, 16)
    np.save(path, array)
    after = browser._overview_pixels(field, (0, 1, 2, 3), (0, 100), edge=120)
    assert before.shape == after.shape == (120, 120, 3)
    np.testing.assert_array_equal(before[:60], after[:60])
    np.testing.assert_array_equal(before[60:, :60], after[60:, :60])
    assert not np.array_equal(before[60:, 60:], after[60:, 60:])



def test_overview_initial_choices_follow_supplied_display_settings(qtbot):
    dialog = browser._FieldOverview(settings={"channels": [0, 1, 2, 3],
        "normalization_percentiles": [5, 95], "normalize": False})
    qtbot.addWidget(dialog)
    assert dialog._choices() == ((0, 1, 2, 3), None)
    dialog.normalise.setChecked(True)
    assert dialog._choices() == ((0, 1, 2, 3), (5, 95))
    dialog.close()



def test_plate_previews_decode_only_visible_representatives(plate_images, qtbot, monkeypatch):
    from dataclasses import replace
    actual = next(f for f in browser._overview_index(plate_images) if not f.unresolved)
    fields = tuple(replace(actual, well=f"A{column:02d}", column=column) for column in range(1, 41))
    dialog = browser._FieldOverview()
    qtbot.addWidget(dialog, before_close_func=lambda widget: _retire_overview(widget, qtbot))
    calls = []
    original = browser._overview_pixels
    def read(field, *args, **kwargs):
        calls.append((field.well, kwargs.get("edge")))
        return original(field, *args, **kwargs)
    monkeypatch.setattr(browser, "_overview_pixels", read)
    _ready_overview(dialog, fields, qtbot)
    assert not calls
    dialog.show()
    _magnify_plate(dialog)
    qtbot.waitUntil(lambda: not dialog.wells.item(0, 0).icon().isNull())
    def visible():
        return {dialog.wells.item(0, column).data(browser.Qt.UserRole)
                for column in range(40)
                if dialog.wells.viewport().rect().intersects(dialog.wells.visualItemRect(dialog.wells.item(0, column)))}
    initial = visible()
    qtbot.waitUntil(lambda: all(not dialog.wells.item(0, int(well[1:]) - 1).icon().isNull() for well in initial))
    before = dialog._thumbnail_jobs._next_id
    for well in initial:
        item = dialog.wells.item(0, int(well[1:]) - 1)
        assert item.data(browser.Qt.UserRole + 1) in dialog._thumbnail_cache
    dialog._update_visuals()
    assert dialog._thumbnail_jobs._next_id == before
    assert set(well for well, _ in calls) <= initial
    ratio = browser.device_ratio(dialog.wells.viewport())
    expected = min(dialog.preview_resolution.currentData(),
                   max(64, math.ceil(max(dialog._plate_cell_extent) * ratio)))
    assert all(edge == expected for _, edge in calls)
    assert len(calls) < len(fields)
    dialog.wells.horizontalScrollBar().setValue(dialog.wells.horizontalScrollBar().maximum())
    qtbot.waitUntil(lambda: not dialog.wells.item(0, 39).icon().isNull())
    assert "A40" in {well for well, _ in calls}
    assert len(dialog._thumbnail_cache) <= 64
    dialog.close()


def test_well_montage_and_individual_zoom_share_real_pixels(plate_images, qtbot):
    dialog = browser._FieldOverview()
    qtbot.addWidget(dialog)
    dialog._indexed(browser._overview_index(plate_images))
    dialog.plates.setCurrentIndex(dialog.plates.findData("plate1"))
    dialog.show()
    dialog._well(dialog.wells.item(0, 0))
    qtbot.waitUntil(lambda: dialog._displayed_montage is not None)
    montage = dialog.view._item.pixmap().toImage()
    assert montage.width() == 3 * 256 and montage.height() == 280
    from PIL import Image
    expected = Image.fromarray(browser._overview_pixels(
        dialog._visible[0], (1, 2), (1., 99.9), edge=240)).resize((240, 200), Image.Resampling.BILINEAR)
    assert montage.pixelColor(100, 100).getRgb()[:3] == expected.getpixel((100, 76))
    dialog.fields.setCurrentRow(0)
    dialog._field(dialog.fields.item(0))
    qtbot.waitUntil(lambda: not dialog._image_jobs.is_busy())
    assert dialog.view._item.pixmap().width() == 12
    dialog._back()
    assert dialog._scope == ("plate1", None)
    assert dialog.view._item is None
    assert dialog.wells.item(0, 0).data(browser.Qt.UserRole) == "A01"
    dialog.close()


def test_well_aliases_use_one_canonical_grid_address(tmp_path):
    for well, channel in (("A1", 1), ("A01", 2)):
        tifffile.imwrite(tmp_path / f"plate_{well}_F1_C{channel}.tif",
                         np.arange(30, dtype=np.uint16).reshape(5, 6))
    field, = browser._overview_index(tmp_path)
    assert field.well == "A01" and len(field.files) == 2
    assert browser._overview_pixels(field, (1, 2)).shape == (5, 6, 3)


def test_source_change_rejects_an_old_visible_well_thumbnail(plate_images, tmp_path, qtbot, monkeypatch):
    dialog = browser._FieldOverview()
    qtbot.addWidget(dialog, before_close_func=lambda widget: _retire_overview(widget, qtbot))
    _ready_overview(dialog, browser._overview_index(plate_images), qtbot)
    dialog.plates.setCurrentIndex(dialog.plates.findData("plate1"))
    started, release = threading.Event(), threading.Event()
    original = browser._overview_pixels
    def read(field, *args, **kwargs):
        pixels = original(field, *args, **kwargs)
        if field.plate == "plate1":
            started.set()
            assert release.wait(5)
        return pixels
    monkeypatch.setattr(browser, "_overview_pixels", read)
    next_source = tmp_path / "replacement"
    next_source.mkdir()
    for channel in (1, 2):
        tifffile.imwrite(next_source / f"other_A01_F1_C{channel}.tif",
                         np.arange(20, dtype=np.uint16).reshape(4, 5))
    try:
        dialog.show()
        qtbot.waitUntil(started.is_set)
        dialog.source.setText(str(next_source))
        dialog._scan()
        qtbot.waitUntil(lambda: bool(dialog._fields) and dialog._fields[0].plate == "other")
        release.set()
        qtbot.waitUntil(lambda: not dialog.wells.item(0, 0).icon().isNull())
        assert all(key[0][0][0] == "other" for key in dialog._thumbnail_cache)
        assert dialog.wells.item(0, 0).data(browser.Qt.UserRole + 1)[0][0][0] == "other"
    finally:
        release.set()
        dialog.close()



def test_scrolling_many_wells_releases_offscreen_native_icons(plate_images, qtbot, monkeypatch):
    from dataclasses import replace
    actual = next(f for f in browser._overview_index(plate_images) if not f.unresolved)
    fields = tuple(replace(actual, well=f"A{column:02d}", column=column) for column in range(1, 81))
    dialog = browser._FieldOverview()
    qtbot.addWidget(dialog, before_close_func=lambda widget: _retire_overview(widget, qtbot))
    calls = set()
    original = browser._overview_pixels
    def read(field, *args, **kwargs):
        calls.add(field.well)
        return original(field, *args, **kwargs)
    monkeypatch.setattr(browser, "_overview_pixels", read)
    _ready_overview(dialog, fields, qtbot)
    dialog.show()
    _magnify_plate(dialog)
    try:
        for column in range(80):
            target = dialog.wells.item(0, column)
            dialog.wells.scrollToItem(target)
            qtbot.waitUntil(lambda: not target.icon().isNull())
            dialog._update_visuals()
            visible = {index for index in range(80) if dialog.wells.viewport().rect().intersects(
                dialog.wells.visualItemRect(dialog.wells.item(0, index)))}
            installed = {index for index in range(80) if not dialog.wells.item(0, index).icon().isNull()}
            assert installed <= visible
            assert len(dialog._thumbnail_cache) <= 64
        assert len(calls) > 64
        assert len(dialog._installed_thumbnails) < 64
        old = dialog.wells.item(0, 0)
        assert old.icon().isNull()
        dialog._set_thumbnail((dialog.plates.currentData(), "A01"), (fields[0].identity, (1, 2), (1., 99.9)),
                              original(fields[0], (1, 2), edge=160))
        assert old.icon().isNull()
    finally:
        dialog.close()



def test_changed_display_choices_reject_the_previous_inflight_image(plate_images, qtbot, monkeypatch):
    dialog = browser._FieldOverview()
    qtbot.addWidget(dialog, before_close_func=lambda widget: _retire_overview(widget, qtbot))
    _ready_overview(dialog, tuple(f for f in browser._overview_index(plate_images) if not f.unresolved), qtbot)
    started, release = threading.Event(), threading.Event()
    original = browser._overview_pixels
    calls, shown = [], []
    show = dialog._show_pixels
    def read(field, *args, **kwargs):
        calls.append(args)
        pixels = original(field, *args, **kwargs)
        if len(calls) == 1:
            started.set()
            assert release.wait(5)
        return pixels
    def display(pixels, preserve_view=False):
        shown.append(pixels.copy())
        show(pixels, preserve_view)
    monkeypatch.setattr(browser, "_overview_pixels", read)
    monkeypatch.setattr(dialog, "_show_pixels", display)
    try:
        dialog.fields.setCurrentRow(0)
        dialog._field(dialog.fields.item(0))
        qtbot.waitUntil(started.is_set)
        dialog.channels.setText("1")
        dialog._refresh_image()
        release.set()
        qtbot.waitUntil(lambda: len(calls) == 3 and not dialog._image_jobs.is_busy())
        assert len(shown) == 2
        np.testing.assert_array_equal(shown[0], shown[1])
        np.testing.assert_array_equal(shown[0][..., 0], shown[0][..., 2])
        assert shown[0][..., 2].any()
    finally:
        release.set()
        dialog.close()



def test_missing_channel_clears_previous_well_and_field_images(plate_images, qtbot):
    dialog = browser._FieldOverview()
    qtbot.addWidget(dialog)
    dialog._indexed(tuple(f for f in browser._overview_index(plate_images) if not f.unresolved))
    dialog.show()
    cell = dialog.wells.item(0, 0)
    qtbot.waitUntil(lambda: not cell.icon().isNull())
    dialog.fields.setCurrentRow(0)
    dialog._field(dialog.fields.item(0))
    qtbot.waitUntil(lambda: not dialog._image_jobs.is_busy())
    assert dialog.view._item is not None
    dialog.channels.setText("99")
    dialog._refresh_image()
    qtbot.waitUntil(lambda: cell.data(browser.Qt.UserRole + 1)[1] == (99,))
    qtbot.waitUntil(lambda: not dialog._image_jobs.is_busy())
    assert cell.icon().isNull()
    assert (dialog.plates.currentData(), "A01") not in dialog._installed_thumbnails
    assert dialog.view._item is None
    before = dialog._thumbnail_jobs._next_id
    dialog._update_visuals()
    assert dialog._thumbnail_jobs._next_id == before
    dialog.close()



def test_back_to_plate_rejects_a_late_individual_field_image(plate_images, qtbot, monkeypatch):
    dialog = browser._FieldOverview()
    qtbot.addWidget(dialog, before_close_func=lambda widget: _retire_overview(widget, qtbot))
    _ready_overview(dialog, tuple(f for f in browser._overview_index(plate_images) if not f.unresolved), qtbot)
    dialog._well(dialog.wells.item(0, 0))
    started, release = threading.Event(), threading.Event()
    original = browser._overview_pixels
    shown = []
    show = dialog._show_pixels
    def read(*args, **kwargs):
        pixels = original(*args, **kwargs)
        started.set()
        assert release.wait(5)
        return pixels
    def display(pixels, preserve_view=False):
        shown.append(pixels.copy())
        show(pixels, preserve_view)
    monkeypatch.setattr(browser, "_overview_pixels", read)
    monkeypatch.setattr(dialog, "_show_pixels", display)
    try:
        dialog.fields.setCurrentRow(0)
        dialog._field(dialog.fields.item(0))
        qtbot.waitUntil(started.is_set)
        dialog._back()
        dialog.fields.setCurrentRow(0)
        assert dialog.fields.currentItem().data(browser.Qt.UserRole).identity == dialog._visible[0].identity
        release.set()
        qtbot.waitUntil(lambda: not dialog._image_jobs.is_busy())
        assert dialog._scope == ("plate1", None)
        assert not shown and dialog.view._item is None
        assert len(dialog._cache) == 1
    finally:
        release.set()
        dialog.close()


def test_measurements_use_original_raw_pixels_and_whole_image_regionprops(tmp_path, monkeypatch):
    import pandas as pd
    from skimage.measure import regionprops_table
    import skimage.measure

    def supported_regionprops(*args, **kwargs):
        assert "intensity_std" not in kwargs.get("properties", ())
        return regionprops_table(*args, **kwargs)

    monkeypatch.setattr(skimage.measure, "regionprops_table", supported_regionprops)
    raw = np.zeros((1003, 1007), dtype=np.uint16)
    raw[1::2, 1::2] = 30000
    tifffile.imwrite(tmp_path / "plate1_A01_F1_C1.tif", raw)
    tifffile.imwrite(tmp_path / "plate1_A01_F1_C2.tif", np.full(raw.shape, 1234, np.uint16))
    field, = browser._overview_index(tmp_path)
    preview = browser._overview_pixels(field, (1,), (1, 99), edge=1000)
    assert preview.shape[:2] != raw.shape and not preview.any()
    output = tmp_path / "measured"
    paths, errors = browser._overview_export((field,), output, (1, 2), (1, 99), measurements=True)
    assert len(paths) == 1 and not errors
    actual = pd.read_csv(output / "overview_measurements.csv", float_precision="round_trip").iloc[0]
    geometry = regionprops_table(np.ones(raw.shape, np.uint8), properties=(
        "area", "bbox", "centroid", "axis_major_length", "axis_minor_length", "eccentricity", "extent", "perimeter"))
    for key, values in geometry.items():
        assert actual["roi_" + key.replace("-", "_")] == pytest.approx(values[0])
    assert (actual["height_px"], actual["width_px"]) == raw.shape
    assert actual["roi"] == "whole_image" and actual["geometry_units"] == "pixels"
    assert actual["intensity_units"] == "raw_stored_pixel_values"
    assert actual["channel_1_intensity_mean"] == float(raw.mean())
    assert actual["channel_1_intensity_std"] == pytest.approx(float(raw.std()))
    assert actual["channel_1_intensity_min"] == 0 and actual["channel_1_intensity_max"] == 30000
    assert actual["channel_2_intensity_mean"] == 1234
    assert actual["channel_1_finite_pixel_count"] == raw.size
    assert actual["channel_1_nonfinite_pixel_count"] == 0
    assert actual["channel_1_dtype"] == "uint16" and actual["status"] == "ok"
    assert actual["channel_1_source"] == str(tmp_path / "plate1_A01_F1_C1.tif")
    assert json.loads(actual["channels"]) == [1, 2]
    metadata = json.loads((output / "overview_selection.json").read_text())
    assert metadata["measurements"]["original_resolution"] is True
    assert metadata["measurements"]["written_field_count"] == 1
    assert "object_count" not in actual.index and "label" not in actual.index


def test_measurements_report_nonfinite_missing_and_unreadable_fields(tmp_path):
    import pandas as pd

    partial = np.array([[[1., 10.], [np.nan, 20.]], [[np.inf, 30.], [5., 40.]]])
    np.save(tmp_path / "plate1_A01_F1.npy", partial)
    np.save(tmp_path / "plate1_A01_F2.npy", np.full((2, 2, 2), np.nan))
    np.save(tmp_path / "plate1_A01_F3.npy", np.ones((2, 2)))
    (tmp_path / "plate1_A01_F4.tif").write_bytes(b"not a TIFF")
    fields = browser._overview_index(tmp_path)
    output = tmp_path / "errors"
    paths, errors = browser._overview_export(fields, output, (0, 1), (1, 99), measurements=True)
    assert len(paths) == 1 and len(errors) == 3
    actual = pd.read_csv(output / "overview_measurements.csv").set_index("field")
    assert len(actual) == len(fields) == 4
    assert actual.loc[1, "roi_area"] == 4
    assert actual.loc[1, "channel_0_finite_pixel_count"] == 2
    assert actual.loc[1, "channel_0_nonfinite_pixel_count"] == 2
    assert actual.loc[1, "channel_0_intensity_mean"] == 3
    assert actual.loc[1, "channel_0_intensity_std"] == 2
    assert actual.loc[1, "status"] == "ok"
    assert set(actual.loc[[2, 3, 4], "status"]) == {"error"}
    assert actual.loc[2, "channel_0_finite_pixel_count"] == 0
    assert "finite" in actual.loc[2, "error"]
    assert actual.loc[3, "channel_0_intensity_mean"] == 1
    assert "channel" in actual.loc[3, "error"].lower()
    assert actual.loc[4, "source_files"] and actual.loc[4, "error"]


def test_cancelled_measurements_do_not_publish_a_partial_csv(plate_images, tmp_path, monkeypatch):
    fields = tuple(f for f in browser._overview_index(plate_images) if not f.unresolved)[:2]
    cancelled = threading.Event()
    original = browser._overview_measurement
    measured = []
    def measure(*args, **kwargs):
        row = original(*args, **kwargs)
        measured.append(row)
        cancelled.set()
        return row
    monkeypatch.setattr(browser, "_overview_measurement", measure)
    output = tmp_path / "cancelled_measurements"
    paths, errors = browser._overview_export(fields, output, (1,), (1, 99),
                                            cancelled=cancelled, measurements=True)
    assert len(measured) == 1 and measured[0]["status"] == "ok"
    assert not paths and not errors
    assert not (output / "overview_measurements.csv").exists()
    metadata = json.loads((output / "overview_selection.json").read_text())
    assert metadata["cancelled"] is True and metadata["requested_field_count"] == 2
    assert metadata["measurements"]["path"] is None
    assert metadata["measurements"]["written_field_count"] == 0
    assert not list(output.glob(".*pending*"))


def test_measurement_csv_refuses_overwrite_and_cleans_failed_atomic_write(plate_images, tmp_path, monkeypatch):
    from spacr import tabular

    field = next(f for f in browser._overview_index(plate_images) if not f.unresolved)
    existing = tmp_path / "existing"
    existing.mkdir()
    previous = existing / "overview_measurements.csv"
    previous.write_bytes(b"original measurements\n")
    with pytest.raises(ValueError, match="already exist"):
        browser._overview_export((field,), existing, (1,), None, measurements=True)
    assert previous.read_bytes() == b"original measurements\n"
    assert not list(existing.glob("*.pdf"))
    fresh = tmp_path / "write_failure"
    fresh.mkdir()
    sentinel = fresh / "unrelated.csv"
    sentinel.write_bytes(b"preserve me\n")
    def fail(frame, path, **kwargs):
        from pathlib import Path
        Path(path).write_bytes(b"incomplete CSV")
        raise OSError("disk unavailable")
    monkeypatch.setattr(tabular, "write_table", fail)
    with pytest.raises(OSError, match="disk unavailable"):
        browser._overview_export((field,), fresh, (1,), None, measurements=True)
    assert sentinel.read_bytes() == b"preserve me\n"
    assert not (fresh / "overview_measurements.csv").exists()
    assert not list(fresh.glob(".*pending*"))


def test_measurements_follow_exact_ui_selection_in_source_results(
        plate_images, tmp_path, qtbot, monkeypatch):
    import pandas as pd
    from PySide6.QtWidgets import QFileDialog

    dialog = browser._FieldOverview(settings={"source": str(plate_images)})
    def retire(widget):
        """Retire this overview's actual asynchronous work before closing its owner."""
        widget._visual_timer.stop()
        _retire_overview(widget, qtbot)
    qtbot.addWidget(dialog, before_close_func=retire)
    dialog._export_jobs._threaded = False
    dialog.channels.setText("1,2")
    dialog._indexed(browser._overview_index(plate_images))
    dialog.plates.setCurrentIndex(dialog.plates.findData("plate1"))
    dialog._well(dialog.wells.item(0, 0))
    qtbot.waitUntil(lambda: dialog._normalization_ready and not dialog._index_jobs.is_busy(), timeout=10000)
    dialog.use_selection.setChecked(True)
    dialog.fields.item(1).setSelected(True)
    selected = dialog._export_selection()
    assert len(selected) == 1
    assert dialog.measurements.isChecked() is False
    dialog.measurements.setChecked(True)
    output = tmp_path / "selection"
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: pytest.fail("Results export must not ask for a destination"))
    dialog._export()
    output = next((plate_images / "results").glob("overview_*"))
    actual = pd.read_csv(output / "overview_measurements.csv", dtype={"field": str})
    assert list(zip(actual["plate"], actual["well"], actual["field"])) == [
        (field.plate, field.well, field.field) for field in selected]
    assert all(actual["channel_1_intensity_mean"] == 59.5)
    assert all(actual["channel_2_intensity_mean"] == 119)
    assert "overview_measurements.csv" in dialog.status.text()
    retire(dialog)
    dialog.close()


def test_cancellation_during_measurement_csv_write_discards_complete_pending_file(plate_images, tmp_path, monkeypatch):
    from pathlib import Path
    from spacr import tabular

    field = next(f for f in browser._overview_index(plate_images) if not f.unresolved)
    cancelled = threading.Event()
    original = tabular.write_table
    pending_paths = []
    def write_then_cancel(frame, path, **kwargs):
        original(frame, path, **kwargs)
        pending_paths.append(path)
        assert Path(path).is_file()
        cancelled.set()
    monkeypatch.setattr(tabular, "write_table", write_then_cancel)
    output = tmp_path / "cancelled_during_write"
    paths, errors = browser._overview_export((field,), output, (1,), None,
                                            cancelled=cancelled, measurements=True)
    assert len(paths) == 1 and not errors and len(pending_paths) == 1
    assert not (output / "overview_measurements.csv").exists()
    assert not any(Path(path).exists() for path in pending_paths)
    metadata = json.loads((output / "overview_selection.json").read_text())
    assert metadata["cancelled"] is True
    assert metadata["measurements"]["path"] is None
    assert metadata["measurements"]["written_field_count"] == 0


def test_visible_well_thumbnails_follow_native_ratio_and_reuse_source_cache(tmp_path, qtbot, monkeypatch):
    from PySide6.QtCore import QEvent, QSize
    from PySide6.QtWidgets import QApplication
    image = np.arange(480 * 640, dtype=np.uint16).reshape(480, 640)
    tifffile.imwrite(tmp_path / "plate1_A01_T0001F001L01C1.tif", image)
    fields = browser._overview_index(tmp_path)
    dialog = browser._FieldOverview(settings={"channels": [1]})
    qtbot.addWidget(dialog, before_close_func=lambda widget: _retire_overview(widget, qtbot))
    viewport = dialog.wells.viewport()
    ratio = [1.0]
    monkeypatch.setattr(viewport, "devicePixelRatioF", lambda: ratio[0])
    decoded = []
    original = browser._overview_pixels

    def read(field, *args, **kwargs):
        decoded.append(kwargs["edge"])
        return original(field, *args, **kwargs)

    monkeypatch.setattr(browser, "_overview_pixels", read)
    _ready_overview(dialog, fields, qtbot)
    dialog.show()
    item = dialog.wells.item(0, 0)
    try:
        for density in (1.0, 2.0, 1.0):
            ratio[0] = density
            QApplication.sendEvent(viewport, QEvent(QEvent.Type.DevicePixelRatioChange))
            qtbot.waitUntil(lambda: item.data(browser.Qt.UserRole + 1) is not None
                            and item.data(browser.Qt.UserRole + 1)[3] == density)
            size = dialog.wells.iconSize()
            picture = item.icon().pixmap(size, density)
            assert picture.devicePixelRatio() == density
            expected = QSize(image.shape[1], image.shape[0]).scaled(size, browser.Qt.KeepAspectRatio)
            # hidpi.scaled_for fits the picture into the device-pixel bounds,
            # so where the icon box does not match the image's aspect its
            # logical size lands within a pixel of the logical fit.
            bounds = QSize(round(size.width() * density), round(size.height() * density))
            physical = QSize(image.shape[1], image.shape[0]).scaled(bounds, browser.Qt.KeepAspectRatio)
            assert picture.size() == physical
            assert abs(picture.deviceIndependentSize().width() - expected.width()) <= 1
            assert abs(picture.deviceIndependentSize().height() - expected.height()) <= 1
        base = max(64, max(dialog._plate_cell_extent or (160, 160)))
        cap = dialog.preview_resolution.currentData()
        assert decoded == [min(cap, base), min(cap, base * 2)]
        assert len(dialog._thumbnail_cache) == 2
    finally:
        dialog.close()


def test_plate_cells_keep_identity_and_shared_numeric_sorting(plate_images, qtbot, qapp):
    from collections import Counter
    from dataclasses import replace
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QTableWidget
    from spacr.qt.widgets.sortable_table import SortableTableItem, install_sorting, table_item

    original = next(field for field in browser._overview_index(plate_images) if not field.unresolved)
    expected = {"A01": 2, "A02": 10, "A03": 3}
    fields = tuple(replace(original, well=well, row="1", column=column, field=str(index))
                   for column, (well, count) in enumerate(expected.items(), 1)
                   for index in range(count))
    dialog = browser._FieldOverview()
    qtbot.addWidget(dialog)
    dialog._indexed(fields)
    cells = [dialog.wells.item(0, column) for column in range(3)]
    for column, (well, count) in enumerate(expected.items()):
        item = cells[column]
        assert isinstance(item, SortableTableItem)
        assert item.text() == browser.tr("{well}\n{count} fields", well=well, count=count)
        assert item.data(browser.Qt.UserRole) == well
        assert not item.flags() & browser.Qt.ItemIsEditable
        assert dialog._well_positions[(dialog.plates.currentData(), well)] == (0, column)
    assert not dialog.wells.isSortingEnabled()
    snapshots = {id(item): (item.text(), item.data(browser.Qt.UserRole), item.flags())
                 for item in cells}
    counts = Counter(field.well for field in fields)
    table = QTableWidget(3, 2)
    qtbot.addWidget(table)
    table.setHorizontalHeaderLabels(["Well", "Fields"])
    for row, item in enumerate(cells):
        assert dialog.wells.takeItem(0, row) is item
        table.setItem(row, 0, item)
        table.setItem(row, 1, table_item(counts[item.data(browser.Qt.UserRole)]))
    install_sorting(table)
    table.show()
    qtbot.waitExposed(table)
    header = table.horizontalHeader()
    for order in (("A02", "A03", "A01"), ("A01", "A03", "A02"), tuple(expected)):
        qtbot.mouseClick(header.viewport(), browser.Qt.LeftButton, pos=QPoint(
            header.sectionViewportPosition(1) + header.sectionSize(1) // 2,
            header.height() // 2))
        qapp.processEvents()
        assert tuple(table.item(row, 0).data(browser.Qt.UserRole) for row in range(3)) == order
        for row in range(3):
            item = table.item(row, 0)
            assert (item.text(), item.data(browser.Qt.UserRole), item.flags()) == snapshots[id(item)]
            assert int(table.item(row, 1).text()) == counts[item.data(browser.Qt.UserRole)]
    dialog.close()
    table.close()


def test_montage_keeps_a_read_error_panel_and_the_next_real_field(plate_images):
    from PIL import Image
    fields = tuple(field for field in browser._overview_index(plate_images)
                   if field.plate == "plate1" and field.well == "A01")[:2]
    expected = browser._overview_pixels(fields[1], (1, 2), None, edge=240)
    expected = Image.fromarray(expected).resize((240, 200), Image.Resampling.BILINEAR)
    from pathlib import Path
    Path(fields[0].files[0][1]).write_bytes(b"unreadable after indexing")
    pixels, errors = browser._overview_montage(fields, (1, 2), None)
    assert pixels.shape == (280, 512, 3)
    assert len(errors) == 1 and errors[0][0] == fields[0].caption and errors[0][1]
    np.testing.assert_array_equal(pixels[24:224, 256:496], np.asarray(expected))
    assert np.any(pixels[30:80, :240, 0] > pixels[30:80, :240, 1])


def test_cancelled_real_montage_does_not_deliver_a_partial_page(plate_images, qtbot, monkeypatch):
    from PySide6.QtWidgets import QPushButton
    fields = tuple(field for field in browser._overview_index(plate_images)
                   if field.plate == "plate1" and field.well == "A01")
    dialog = browser._FieldOverview(settings={"channels": [1, 2]})
    qtbot.addWidget(dialog)
    started, release = threading.Event(), threading.Event()
    original = browser._overview_pixels

    def read(field, *args, **kwargs):
        pixels = original(field, *args, **kwargs)
        if kwargs.get("edge") == 240:
            started.set()
            assert release.wait(5)
        return pixels

    original_montage = browser._overview_montage

    def montage(*args, **kwargs):
        import sys
        previous = sys.gettrace()
        inherited = threading.gettrace() if previous is None else None
        if inherited is not None:
            sys.settrace(inherited)
        try:
            return original_montage(*args, **kwargs)
        finally:
            if inherited is not None:
                sys.settrace(previous)

    monkeypatch.setattr(browser, "_overview_pixels", read)
    monkeypatch.setattr(browser, "_overview_montage", montage)
    dialog._indexed(fields)
    dialog.show()
    try:
        dialog._well(dialog.wells.item(0, 0))
        qtbot.waitUntil(started.is_set)
        cancel = next(button for button in dialog.findChildren(QPushButton)
                      if button.text() == browser.tr("Cancel loading or export"))
        qtbot.mouseClick(cancel, browser.Qt.LeftButton)
        assert dialog._cancelled.is_set()
        release.set()
        qtbot.waitUntil(lambda: not dialog._montage_jobs.is_busy())
        assert dialog._displayed_montage is None
        assert dialog.view._item is None
        assert dialog.status.text() == browser.tr(
            "Cancellation requested; the current image read may finish.")
    finally:
        release.set()
        qtbot.waitUntil(lambda: not dialog._montage_jobs.is_busy())
        dialog.close()


def test_editable_export_validation_then_real_worker_and_gui_result(plate_images, tmp_path, qtbot, monkeypatch, qapp):
    from PySide6.QtCore import QThread
    from PySide6.QtWidgets import QFileDialog, QPushButton
    fields = tuple(field for field in browser._overview_index(plate_images)
                   if field.plate == "plate1" and field.well == "A01")
    dialog = browser._FieldOverview(settings={"source": str(plate_images), "channels": [1, 2]})
    def retire(widget):
        """Retire this overview's actual asynchronous work before closing its owner."""
        widget._visual_timer.stop()
        _retire_overview(widget, qtbot)
    qtbot.addWidget(dialog, before_close_func=retire)
    dialog._indexed(fields)
    qtbot.waitUntil(lambda: dialog._normalization_ready and not dialog._index_jobs.is_busy(), timeout=10000)
    dialog.use_selection.setChecked(False)
    dialog.show()
    dialog.options_button.click()
    dialog._controls_scroll.ensureWidgetVisible(dialog.channels)
    export = next(button for button in dialog.findChildren(QPushButton)
                  if button.text() == browser.tr("Contact sheet PDFs"))
    output = tmp_path / "real-worker"
    folders, workers, delivered = [], [], []
    started, release = threading.Event(), threading.Event()
    original = browser._overview_export

    def choose(*args, **kwargs):
        folders.append(str(output))
        return str(output)

    def write(*args, **kwargs):
        import sys
        previous = sys.gettrace()
        inherited = threading.gettrace() if previous is None else None
        if inherited is not None:
            sys.settrace(inherited)
        try:
            workers.append(QThread.currentThread())
            started.set()
            assert release.wait(5)
            return original(*args, **kwargs)
        finally:
            if inherited is not None:
                sys.settrace(previous)

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", choose)
    monkeypatch.setattr(browser, "_overview_export", write)
    dialog.channels.setText("not a channel")
    try:
        qtbot.mouseClick(export, browser.Qt.LeftButton)
        assert dialog._visual_context() is None
        assert dialog.status.text() and not folders and not workers
        assert not dialog._export_jobs.is_busy() and not output.exists()
        dialog.channels.setText("1,2")
        dialog.channels.setFocus()
        qtbot.keyClick(dialog.channels, browser.Qt.Key_Tab)
        qtbot.waitUntil(lambda: dialog._normalization_ready and not dialog._index_jobs.is_busy(), timeout=10000)
        dialog.status.setText("")
        dialog._export_jobs.job_finished.connect(lambda: delivered.append(QThread.currentThread()))
        qtbot.mouseClick(export, browser.Qt.LeftButton)
        qtbot.waitUntil(started.is_set)
        assert dialog._export_jobs.is_busy()
        qtbot.mouseClick(export, browser.Qt.LeftButton)
        assert dialog.status.text() == browser.tr("Wait for the current export to finish or cancel it.")
        release.set()
        qtbot.waitUntil(lambda: bool(delivered) and not dialog._export_jobs.is_busy())
        output = next((plate_images / "results").glob("overview_*"))
        assert len(workers) == 1 and not folders
        assert workers[0] != qapp.thread() and delivered == [qapp.thread()]
        assert (output / "overview_0001.pdf").read_bytes().startswith(b"%PDF-")
        selection = json.loads((output / "overview_selection.json").read_text())
        assert selection["written_field_count"] == len(dialog._export_selection())
        assert dialog.status.text() == browser.tr("Wrote {count} PDF pages; {errors} unreadable fields.",
                                                  count=1, errors=0)
    finally:
        release.set()
        qtbot.waitUntil(lambda: not dialog._export_jobs.is_busy())
        retire(dialog)
        dialog.close()


@pytest.mark.parametrize("rgb", (False, True), ids=("gray-png", "rgb-png"))
def test_png_overview_preserves_stored_pixels_in_whole_image_measurements(tmp_path, rgb):
    from PIL import Image
    plane = np.arange(48, dtype=np.uint8).reshape(6, 8)
    image = np.stack((plane, plane + 50, plane + 100), axis=-1) if rgb else plane
    path = tmp_path / "plate1_A01_F1.png"
    Image.fromarray(image).save(path)
    field, = browser._overview_index(tmp_path)
    channels = (0, 1, 2) if rgb else (0,)
    actual = tuple(browser._overview_raw_planes(field, channels))
    for channel, source, values in actual:
        assert source == str(path)
        np.testing.assert_array_equal(values, image[..., channel] if rgb else image)
    measurement = browser._overview_measurement(field, channels)
    assert measurement["status"] == "ok"
    assert measurement["height_px"] == 6 and measurement["width_px"] == 8
    assert measurement["roi_area"] == 48 and measurement["intensity_units"] == "raw_stored_pixel_values"
    for channel in channels:
        assert measurement[f"channel_{channel}_intensity_mean"] == 23.5 + (50 * channel if rgb else 0)
    expected = np.repeat(np.uint8(plane / 47 * 255)[..., None], 3, axis=2)
    np.testing.assert_array_equal(browser._overview_pixels(field, channels, None), expected)


@pytest.mark.parametrize("axes", ("CYX", "TYX"), ids=("explicit-channels", "unresolved-time"))
def test_tiff_axis_metadata_selects_channels_without_guessing_time(tmp_path, axes):
    image = np.stack([np.arange(35, dtype=np.uint16).reshape(5, 7) + 100 * index
                      for index in range(3)])
    path = tmp_path / "plate1_A01_F1.tif"
    tifffile.imwrite(path, image, photometric="minisblack", metadata={"axes": axes})
    field, = browser._overview_index(tmp_path)
    if axes == "TYX":
        with pytest.raises(ValueError, match=r"Channel 2 is missing or ambiguous\. Available channels: 0"):
            tuple(browser._overview_raw_planes(field, (2,)))
        with pytest.raises(ValueError, match="Choose explicit time and Z positions"):
            tuple(browser._overview_raw_planes(field, (0,)))
        record = browser._overview_measurement(field, (0,))
        assert record["status"] == "error" and "Choose explicit time and Z positions" in record["error"]
        assert "channel_2_intensity_mean" not in record
        channel, source, plane = next(browser._overview_raw_planes(field, (0,), time_index=2))
        assert channel == 0 and source == str(path)
        np.testing.assert_array_equal(plane, image[2])
    else:
        channel, source, plane = next(browser._overview_raw_planes(field, (2,)))
        assert channel == 2 and source == str(path)
        np.testing.assert_array_equal(plane, image[2])
        record = browser._overview_measurement(field, (2,))
        assert record["status"] == "ok" and record["channel_2_intensity_mean"] == 217
        assert record["height_px"] == 5 and record["width_px"] == 7


@pytest.mark.parametrize("problem", ("complex", "different-channel-shapes"))
def test_measurements_report_real_unsupported_pixels_and_shape_mismatch(tmp_path, problem):
    if problem == "complex":
        np.save(tmp_path / "plate1_A01_F1.npy", np.ones((4, 5), dtype=np.complex64) * (1 + 2j))
        channels = (0,)
        expected_error = "Measurements require real numeric image pixels."
    else:
        tifffile.imwrite(tmp_path / "plate1_A01_F1_C1.tif", np.arange(20, dtype=np.uint16).reshape(4, 5))
        tifffile.imwrite(tmp_path / "plate1_A01_F1_C2.tif", np.arange(12, dtype=np.uint16).reshape(3, 4))
        channels = (1, 2)
        expected_error = "Selected channels must have matching image sizes."
    field, = browser._overview_index(tmp_path)
    record = browser._overview_measurement(field, channels)
    assert record["status"] == "error" and record["error"] == browser.tr(expected_error)
    assert record["source_files"] == json.dumps(field.files)
    assert not any(key.startswith(f"channel_{channels[-1]}_intensity_") for key in record)
    if problem != "complex":
        assert record["channel_1_intensity_mean"] == 9.5
        with pytest.raises(ValueError, match="matching image sizes"):
            browser._overview_pixels(field, channels)


@pytest.mark.parametrize("format", ("tif", "png"), ids=("128-MiB-tiff", "32-Mpixel-png"))
def test_real_oversized_image_headers_refuse_pixel_decode(tmp_path, monkeypatch, format):
    from PIL import Image, PngImagePlugin
    path = tmp_path / f"plate1_A01_F1.{format}"
    if format == "tif":
        tifffile.imwrite(path, data=None, shape=(8193, 8192), dtype=np.uint16, photometric="minisblack")
        monkeypatch.setattr(tifffile.TiffPageSeries, "asarray", lambda *_a, **_k: pytest.fail("oversized TIFF decoded"))
    else:
        Image.new("L", (8193, 4096)).save(path)
        monkeypatch.setattr(PngImagePlugin.PngImageFile, "load", lambda *_a, **_k: pytest.fail("oversized PNG decoded"))
    field, = browser._overview_index(tmp_path)
    with pytest.raises(ValueError, match="preview decode limit"):
        tuple(browser._overview_raw_planes(field, (0,)))
    record = browser._overview_measurement(field, (0,))
    assert record["status"] == "error" and "preview decode limit" in record["error"]
    assert "height_px" not in record and "channel_0_intensity_mean" not in record


@pytest.mark.parametrize("invalid", ("negative-channel", "reversed-percentiles"))
def test_invalid_editable_display_choices_reach_field_and_export_guards(plate_images, qtbot, monkeypatch, invalid):
    from PySide6.QtWidgets import QFileDialog, QPushButton
    dialog = browser._FieldOverview(settings={"channels": [1]})
    def retire(widget):
        """Stop only this overview's queued visuals and owned workers before close."""
        widget._visual_timer.stop()
        _retire_overview(widget, qtbot)
    qtbot.addWidget(dialog, before_close_func=retire)
    dialog._indexed(tuple(field for field in browser._overview_index(plate_images)
                          if field.plate == "plate1" and field.well == "A01"))
    dialog.show()
    dialog._well(dialog.wells.item(0, 0))
    qtbot.waitUntil(lambda: dialog._displayed_montage is not None)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_a, **_k: pytest.fail("invalid choices opened a folder dialog"))
    if invalid == "negative-channel":
        dialog.channels.setText("-1")
        expected = browser.tr("Choose one or more nonnegative channel indices.")
    else:
        dialog.normalise.setChecked(True)
        dialog.low.setValue(90)
        dialog.high.setValue(10)
        expected = browser.tr("The lower percentile must be below the upper percentile.")
    try:
        item = dialog.fields.item(0)
        qtbot.mouseClick(dialog.fields.viewport(), browser.Qt.LeftButton,
                         pos=dialog.fields.visualItemRect(item).center())
        assert dialog.status.text() == expected
        export = next(button for button in dialog.findChildren(QPushButton)
                      if button.text() == browser.tr("Contact sheet PDFs"))
        qtbot.mouseClick(export, browser.Qt.LeftButton)
        assert dialog.status.text() == expected and not dialog._export_jobs.is_busy()
        assert dialog._visual_context() is None
    finally:
        retire(dialog)
        dialog.close()


def test_folder_browse_and_real_metadata_acceptance_reindex_without_moving_images(tmp_path, qtbot, monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QPushButton, QDialogButtonBox
    import sys
    source = tmp_path / "images"
    source.mkdir()
    path = source / "sample-p9-A03-f2-c1.tif"
    tifffile.imwrite(path, np.arange(30, dtype=np.uint16).reshape(5, 6))
    before = path.read_bytes()
    initial = tmp_path / "empty"
    initial.mkdir()
    dialog = browser._FieldOverview(settings={"src": [str(initial)], "channels": [1]})
    qtbot.addWidget(dialog, before_close_func=lambda widget: _retire_overview(widget, qtbot))
    dialog.show()
    dialog.options_button.click()
    original = browser._overview_index

    def index(*args, **kwargs):
        previous = sys.gettrace()
        inherited = threading.gettrace() if previous is None else None
        if inherited is not None:
            sys.settrace(inherited)
        try:
            return original(*args, **kwargs)
        finally:
            if inherited is not None:
                sys.settrace(previous)

    monkeypatch.setattr(browser, "_overview_index", index)
    metadata = next(button for button in dialog.findChildren(QPushButton)
                    if button.text() == browser.tr("Filename metadata…"))
    browse = next(button for button in dialog.findChildren(QPushButton)
                  if button.text() == browser.tr("Browse folder"))
    try:
        dialog._controls_scroll.ensureWidgetVisible(metadata)
        qtbot.mouseClick(metadata, browser.Qt.LeftButton)
        assert dialog.status.text() == browser.tr("Load the image index before opening filename metadata.")
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_a, **_k: "")
        dialog._controls_scroll.ensureWidgetVisible(browse)
        qtbot.mouseClick(browse, browser.Qt.LeftButton)
        assert dialog.source.text() == str(initial) and not dialog._fields
        dialog.regex.setText(r"sample-p9-(?P<wellID>[A-Z]\d+)-f\d+-c(?P<chanID>\d+)")
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_a, **_k: str(source))
        dialog._controls_scroll.ensureWidgetVisible(browse)
        qtbot.mouseClick(browse, browser.Qt.LeftButton)
        qtbot.waitUntil(lambda: not dialog._index_jobs.is_busy() and len(dialog._fields) == 1)
        dialog._controls_scroll.ensureWidgetVisible(metadata)
        qtbot.mouseClick(metadata, browser.Qt.LeftButton)
        workbench = dialog._metadata_dialog
        qtbot.waitExposed(workbench)
        pattern = r"sample-(?P<plateID>p\d+)-(?P<wellID>[A-Z]\d+)-f(?P<fieldID>\d+)-c(?P<chanID>\d+)"
        workbench.workbench.regex.setText(pattern)
        buttons = workbench.findChild(QDialogButtonBox)
        qtbot.mouseClick(buttons.button(QDialogButtonBox.Ok), browser.Qt.LeftButton)
        qtbot.waitUntil(lambda: not dialog._index_jobs.is_busy() and len(dialog._fields) == 1
                        and dialog._fields[0].well == "A03")
        assert dialog.regex.text() == pattern
        field, = dialog._fields
        assert (field.plate, field.well, field.field, field.files) == ("p9", "A03", "2", (("1", str(path)),))
        assert path.read_bytes() == before and tuple(source.iterdir()) == (path,)
    finally:
        dialog._cancel()
        dialog._visual_timer.stop()
        for runner in (dialog._index_jobs, dialog._image_jobs, dialog._export_jobs,
                       dialog._thumbnail_jobs, dialog._montage_jobs):
            runner.shutdown()
        dialog.close()


def test_reopening_well_mean_browser_rebuilds_its_cancelled_context(plate_images, qtbot):
    dialog = browser._FieldOverview(settings={"channels": (1, 2),
        "normalization": {"mode": "reference_average", "reference_kind": "reference_well",
                          "reference": ("plate1", "A01"), "percentiles": (1., 99.)}})
    qtbot.addWidget(dialog, before_close_func=lambda widget: _retire_overview(widget, qtbot))
    dialog.show()
    _ready_overview(dialog, browser._overview_index(plate_images), qtbot)
    old_context, old_cancelled = dialog._normalization_context, dialog._cancelled
    assert old_context is not None
    field = dialog._well_representatives[(dialog.plates.currentData(), "A01")]
    previous_generation = dialog._display_generation
    _retire_overview(dialog, qtbot)
    dialog.close()
    assert old_cancelled.is_set()
    dialog.show()
    qtbot.waitUntil(lambda: dialog._normalization_ready
        and dialog._normalization_context is not old_context, timeout=10000)
    assert dialog._cancelled is not old_cancelled and not dialog._cancelled.is_set()
    assert old_cancelled.is_set()
    assert dialog._display_generation > previous_generation
    current = dialog._display_kwargs()
    pixels = browser._overview_pixels(field, (1, 2), (1., 99.), **current)
    assert pixels.shape == (10, 12, 3) and pixels.any()
    assert dialog._scope == ("plate1", None)


def test_individual_tczyx_planes_follow_native_axis_wheels_without_borrowing_wells(tmp_path, qtbot):
    from PySide6.QtCore import QPoint, QPointF
    from PySide6.QtGui import QImage, QWheelEvent
    from PySide6.QtWidgets import QApplication, QPushButton

    source = tmp_path / "axis-planes"
    source.mkdir()
    arrays, files, originals = {}, {}, {}
    for well, field in (("A01", 1), ("A01", 2), ("B02", 1)):
        pixels = np.zeros((3, 2, 3, 16, 20), dtype=np.uint16)
        for time in range(3):
            for channel in range(2):
                for z in range(3):
                    row = (field * 2 + time + channel * 3 + (7 if well == "B02" else 0)) % 16
                    column = (field * 3 + z * 2 + channel * 4) % 20
                    pixels[time, channel, z, row, column] = 2000
                    pixels[time, channel, z, (row + 5) % 16, (column + 7) % 20] = 1000
        path = source / f"p1_{well}_F{field}.tif"
        tifffile.imwrite(path, pixels, metadata={"axes": "TCZYX"}, photometric="minisblack")
        arrays[well, field], files[well, field] = pixels, path
        originals[path] = path.read_bytes()
    pattern = r"(?P<plateID>p\d+)_(?P<wellID>[A-Z]\d+)_F(?P<fieldID>\d+)\.tif"
    dialog = browser._FieldOverview(settings={"source": str(source), "source_scope": "folder",
        "custom_regex": pattern, "channels": (0,), "geometry": (2, 2),
        "normalization_percentiles": (0., 100.)})
    qtbot.addWidget(dialog, before_close_func=lambda widget: _retire_overview(widget, qtbot))
    dialog.show()
    dialog._scan()
    qtbot.waitUntil(lambda: bool(dialog._fields) and dialog._normalization_ready
                    and not dialog._index_jobs.is_busy(), timeout=10000)
    dialog.fov_axis.setCurrentIndex(dialog.fov_axis.findData("1"))
    item = dialog.wells.item(*dialog._well_positions[(dialog.plates.currentData(), "A01")])
    dialog.wells.setCurrentItem(item)
    open_well = next(button for button in dialog.findChildren(QPushButton)
                     if button.text() == browser.tr("Open selected well"))
    qtbot.mouseClick(open_well, browser.Qt.LeftButton)
    first = dialog.fields.item(0)
    qtbot.mouseClick(dialog.fields.viewport(), browser.Qt.LeftButton,
                    pos=dialog.fields.visualItemRect(first).center())

    def assert_plane(well, field, time, z):
        raw = arrays[well, field][time, 0, z]
        gray = (raw.astype(np.float64) / 2000. * 255.).astype(np.uint8)
        expected = np.repeat(gray[..., None], 3, axis=2)
        picture = QImage(expected.data, 20, 16, expected.strides[0], QImage.Format_RGB888).copy()
        qtbot.waitUntil(lambda: dialog.view._item is not None and not dialog._image_jobs.is_busy()
            and dialog.view._item.pixmap().toImage().convertToFormat(QImage.Format_RGB888) == picture,
            timeout=10000)
        current = dialog.fields.currentItem().data(browser.Qt.UserRole)
        assert current.well == well and current.field_id == str(field)
        assert tuple(path for _, path in current.files) == (str(files[well, field]),)
        assert dialog._individual and dialog._scope == ("p1", well)
        assert (dialog.fov_axis.currentData(), dialog.time_axis.currentData(),
                dialog.z_axis.currentData()) == (str(field), time, z)

    def wheel(angle):
        viewport = dialog.view.viewport()
        position = viewport.rect().center()
        event = QWheelEvent(QPointF(position), QPointF(viewport.mapToGlobal(position)),
                            QPoint(), QPoint(0, angle), browser.Qt.NoButton,
                            browser.Qt.NoModifier, browser.Qt.NoScrollPhase, False)
        QApplication.sendEvent(viewport, event)
        assert event.isAccepted()

    assert_plane("A01", 1, 0, 0)
    for mode, expected in (("time", ("A01", 1, 1, 0)),
                           ("z", ("A01", 1, 1, 1)),
                           ("field", ("A01", 2, 1, 1))):
        dialog.wheel_axis.setCurrentIndex(dialog.wheel_axis.findData(mode))
        transform = dialog.view.transform()
        wheel(120)
        assert dialog.view.transform() == transform
        assert_plane(*expected)
    identity = dialog.fields.currentItem().data(browser.Qt.UserRole).identity
    axes = (dialog.fov_axis.currentData(), dialog.time_axis.currentData(), dialog.z_axis.currentData())
    transform = dialog.view.transform()
    wheel(0)
    assert dialog.fields.currentItem().data(browser.Qt.UserRole).identity == identity
    assert (dialog.fov_axis.currentData(), dialog.time_axis.currentData(), dialog.z_axis.currentData()) == axes
    assert dialog.view.transform() == transform
    item = dialog.wells.item(*dialog._well_positions[(dialog.plates.currentData(), "B02")])
    dialog.wells.setCurrentItem(item)
    qtbot.mouseClick(open_well, browser.Qt.LeftButton)
    assert dialog._scope == ("p1", "B02")
    assert dialog.fields.count() == 0 and dialog.view._item is None
    wheel(-120)
    assert dialog.fov_axis.currentData() == "1"
    first = dialog.fields.item(0)
    qtbot.mouseClick(dialog.fields.viewport(), browser.Qt.LeftButton,
                    pos=dialog.fields.visualItemRect(first).center())
    assert_plane("B02", 1, 1, 1)
    assert all(path.read_bytes() == before for path, before in originals.items())


def test_empty_source_preview_recovers_after_real_tiffs_are_added(tmp_path, qtbot):
    from PySide6.QtWidgets import QPushButton, QScrollArea, QWidget

    source = tmp_path / "initially-empty"
    source.mkdir()
    owner = QWidget()
    qtbot.addWidget(owner, before_close_func=lambda widget: _retire_overview_owner(widget, qtbot))
    wizard = browser._open_field_overview(owner, {"source": str(source)})
    empty_message = browser.tr("No readable image fields or available channels were found.")
    qtbot.waitUntil(lambda: not wizard._runner.is_busy() and wizard.status.text() == empty_message,
                    timeout=10000)
    preview = next(button for button in wizard.findChildren(QPushButton)
                   if button.text() == browser.tr("Preview metadata"))
    wizard.findChild(QScrollArea).ensureWidgetVisible(preview)
    qtbot.mouseClick(preview, browser.Qt.LeftButton)
    qtbot.waitUntil(lambda: not wizard._runner.is_busy() and wizard.status.text() == empty_message,
                    timeout=10000)
    assert wizard._fields == () and wizard._info is None
    assert wizard.preview.rowCount() == 0 and not wizard.next.isEnabled()
    assert not hasattr(owner, "_field_overview")

    originals = {}
    for channel in (1, 2):
        path = source / f"plate1_A01_T0001F001L01C{channel:02}.tif"
        pixels = np.arange(120, dtype=np.uint16).reshape(10, 12) * channel
        tifffile.imwrite(path, pixels)
        originals[path] = path.read_bytes()
    qtbot.mouseClick(preview, browser.Qt.LeftButton)
    qtbot.waitUntil(lambda: not wizard._runner.is_busy() and wizard.next.isEnabled(), timeout=10000)
    field, = wizard._fields
    assert (field.plate, field.well, field.field_id) == ("plate1", "A01", "001")
    assert field.files == tuple((str(channel), str(path)) for channel, path in
                               zip((1, 2), originals))
    assert wizard._info["channels"] == (1, 2)
    assert wizard.preview.rowCount() == 2
    assert wizard.configuration()["channels"] == (1, 2)
    assert wizard.status.text() != empty_message
    qtbot.mouseClick(wizard.next, browser.Qt.LeftButton)
    assert hasattr(owner, "_field_overview"), wizard.status.text()
    dialog = owner._field_overview
    qtbot.waitUntil(lambda: dialog._normalization_ready and not dialog._index_jobs.is_busy()
                    and bool(dialog._well_positions), timeout=10000)
    assert dialog._fields == (field,)
    assert dialog.source.text() == str(source)
    assert dialog.plates.currentData() == "plate1"
    item = dialog.wells.item(*dialog._well_positions[(dialog.plates.currentData(), "A01")])
    qtbot.waitUntil(lambda: not item.icon().isNull(), timeout=10000)
    assert all(path.read_bytes() == before for path, before in originals.items())


def test_rejecting_wizard_during_real_header_probe_retires_without_delivery(tmp_path, qtbot, monkeypatch):
    from PySide6.QtCore import QCoreApplication, QEvent, QThread
    from shiboken6 import isValid
    from spacr.qt.bridge import drain_thread
    from PySide6.QtWidgets import QDialog, QPushButton, QWidget

    source = tmp_path / "pending-header"
    source.mkdir()
    path = source / "plate1_A01_T0001F001L01C01.tif"
    tifffile.imwrite(path, np.arange(120, dtype=np.uint16).reshape(10, 12))
    before = path.read_bytes()
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    retired = threading.Event()
    owned_threads = []
    actual_headers = browser._overview_channel_info

    def gated_headers(field):
        thread = QThread.currentThread()
        owned_threads.append(thread)
        thread.finished.connect(retired.set, browser.Qt.DirectConnection)
        result = actual_headers(field)
        entered.set()
        if not release.wait(10):
            raise RuntimeError("The actual image header read was not released.")
        returned.set()
        return result

    monkeypatch.setattr(browser, "_overview_channel_info", gated_headers)
    owner = QWidget()
    qtbot.addWidget(owner, before_close_func=lambda widget: _retire_overview_owner(widget, qtbot))
    with qtbot.captureExceptions() as exceptions:
        wizard = browser._open_field_overview(owner, {"source": str(source), "channels": (1,)})
        try:
            qtbot.waitUntil(entered.is_set, timeout=10000)
            assert wizard._runner.is_busy() and not wizard.next.isEnabled()
            assert wizard._fields == () and wizard._info is None
            state = (wizard.status.text(), wizard.channels.text(), wizard.preview.rowCount())
            cancel = next(button for button in wizard.findChildren(QPushButton)
                          if button.text() == browser.tr("Cancel"))
            qtbot.mouseClick(cancel, browser.Qt.LeftButton)
            assert wizard.result() == QDialog.Rejected and not wizard.isVisible()
            assert wizard._finished and wizard._cancelled.is_set()
            assert not wizard._startup_timer.isActive()
            release.set()
            qtbot.waitUntil(returned.is_set, timeout=10000)
            qtbot.waitUntil(retired.is_set, timeout=10000)
            owned_thread, = owned_threads
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            qtbot.waitUntil(lambda: not isValid(owned_thread), timeout=10000)
            assert drain_thread(owned_thread, timeout_ms=0)
            qtbot.waitUntil(lambda: wizard._runner.active_jobs() == 0, timeout=10000)
            wizard._runner.shutdown(timeout_ms=3000)
            assert wizard._fields == () and wizard._info is None
            assert (wizard.status.text(), wizard.channels.text(), wizard.preview.rowCount()) == state
            assert not wizard.next.isEnabled() and not hasattr(owner, "_field_overview")
            assert not wizard._runner.is_busy()
            assert path.read_bytes() == before and tuple(source.iterdir()) == (path,)
        finally:
            release.set()
    assert exceptions == []


def test_combined_channel_budget_refuses_full_resolution_but_keeps_real_thumbnail(tmp_path):
    path = tmp_path / "plate1_A01_F1.tif"
    stored = tifffile.memmap(path, shape=(2, 4097, 8192), dtype=np.uint16,
                            photometric="minisblack", metadata={"axes": "CYX"})
    stored[0, 0, 0] = 1000
    stored[1, 256, 256] = 2000
    stored.flush()
    del stored

    before_stat = path.stat()
    field, = browser._overview_index(tmp_path)
    info = browser._overview_channel_info(field)
    assert info["channels"] == (0, 1) and info["files"][0]["axes"] == "CYX"
    assert info["files"][0]["shape"] == (2, 4097, 8192)
    original_planes = tuple(browser._overview_raw_planes(field, (0, 1)))
    assert all(plane.shape == (4097, 8192) and plane.dtype == np.uint16
               and plane.nbytes < 128 * 1024 ** 2 for _, _, plane in original_planes)
    assert sum(plane.nbytes for _, _, plane in original_planes) > 128 * 1024 ** 2
    assert original_planes[0][2][0, 0] == 1000
    assert original_planes[1][2][256, 256] == 2000
    del original_planes

    thumbnail = browser._overview_pixels(field, (0, 1), (0., 100.), edge=32)
    expected = np.zeros((17, 32, 3), dtype=np.uint8)
    expected[0, 0, 0] = 255
    expected[1, 1, 1] = 255
    np.testing.assert_array_equal(thumbnail, expected)
    with pytest.raises(browser._OverviewValidationError, match="128 MiB preview decode limit") as failure:
        browser._overview_pixels(field, (0, 1), (0., 100.), edge=None)
    traceback = failure.value.__traceback__
    while traceback.tb_next is not None:
        traceback = traceback.tb_next
    assert traceback.tb_frame.f_code is browser._overview_pixels.__code__
    assert traceback.tb_frame.f_locals["planes"] == []
    unchanged = tifffile.memmap(path, mode="r")
    assert unchanged.shape == (2, 4097, 8192) and unchanged.dtype == np.uint16
    assert unchanged[0, 0, 0] == 1000 and unchanged[1, 256, 256] == 2000
    assert unchanged[0, -1, -1] == 0 and unchanged[1, -1, -1] == 0
    del unchanged
    assert tuple(tmp_path.iterdir()) == (path,)
    assert (path.stat().st_size, path.stat().st_mtime_ns) == (before_stat.st_size, before_stat.st_mtime_ns)
