"""The 2026-10-24 overview layout: level containers, stacked wells, metadata toggles and load order."""
from pathlib import Path

import numpy as np
import pytest
import tifffile

from PySide6.QtCore import QPoint, Qt

from spacr.qt.widgets import qc_field_browser as browser


PATTERN = r"(?P<plateID>p\d+)-(?P<wellID>[A-Z]\d+)-f(?P<fieldID>\d+)-c(?P<chanID>\d+)\.tif"
WELLS = ("A01", "A02", "B01", "B03")


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    for index, well in enumerate(WELLS):
        for field in (1, 2, 3):
            pixels = (np.arange(600, dtype=np.uint16).reshape(20, 30) + 50 * index + field)
            tifffile.imwrite(folder / f"p1-{well}-f{field}-c1.tif", pixels)
    return folder


def _retire(dialog, qtbot):
    dialog._cancel()
    runners = (dialog._index_jobs, dialog._image_jobs, dialog._thumbnail_jobs,
               dialog._field_thumbnail_jobs, dialog._montage_jobs, dialog._export_jobs,
               dialog._metadata_jobs, dialog._measurement_jobs, dialog._native_jobs)
    qtbot.waitUntil(lambda: all(runner.active_jobs() == 0 for runner in runners), timeout=10000)
    for runner in runners:
        runner.shutdown(timeout_ms=3000)
    from spacr.qt.bridge import prune_parked_threads
    qtbot.waitUntil(lambda: prune_parked_threads() == 0)
    dialog.close()


def _overview(qtbot, source, representatives=1):
    dialog = browser._FieldOverview(settings={
        "source": str(source), "source_scope": "folder", "custom_regex": PATTERN,
        "channels": (1,), "geometry": (2, 3), "representatives": representatives})
    qtbot.addWidget(dialog, before_close_func=lambda widget: _retire(widget, qtbot))
    dialog.resize(1200, 900)
    dialog.show()
    dialog._scan()
    qtbot.waitUntil(lambda: bool(dialog._fields) and dialog._normalization_ready
                    and not dialog._index_jobs.is_busy(), timeout=10000)
    dialog.fov_axis.setCurrentIndex(dialog.fov_axis.findData(None))
    return dialog


def _item(dialog, well):
    return dialog.wells.item(*dialog._well_positions[(dialog.plates.currentData(), well)])


def test_plate_loads_one_image_per_well_before_any_second_representative(source, qtbot, monkeypatch):
    calls = []
    real_pixels, real_montage = browser._overview_pixels, browser._overview_montage

    def pixels(field, *args, **kwargs):
        calls.append(("one", field.well))
        return real_pixels(field, *args, **kwargs)

    def montage(fields, *args, **kwargs):
        calls.append(("all", fields[0].well))
        return real_montage(fields, *args, **kwargs)

    monkeypatch.setattr(browser, "_overview_pixels", pixels)
    monkeypatch.setattr(browser, "_overview_montage", montage)
    dialog = _overview(qtbot, source, representatives=3)
    qtbot.waitUntil(lambda: all(len((_item(dialog, well).data(Qt.UserRole + 1) or ((),))[0]) == 3
                                for well in WELLS), timeout=20000)
    first_full = next(index for index, call in enumerate(calls) if call[0] == "all")
    assert {well for kind, well in calls[:first_full] if kind == "one"} == set(WELLS)
    full_key = _item(dialog, "A01").data(Qt.UserRole + 1)
    assert len(full_key[0]) == 3
    full = {well: _item(dialog, well).data(Qt.UserRole + 1) for well in WELLS}
    calls.clear()
    for well in WELLS:
        _item(dialog, well).setData(Qt.UserRole + 1, None)
    dialog._update_visuals()
    assert {well: _item(dialog, well).data(Qt.UserRole + 1) for well in WELLS} == full
    assert not calls
    for key in full.values():
        dialog._thumbnail_cache.pop(key)
    for well in WELLS:
        _item(dialog, well).setData(Qt.UserRole + 1, None)
    qtbot.waitUntil(lambda: not dialog._thumbnail_jobs.is_busy())
    dialog._update_visuals()
    for well in WELLS:
        assert _item(dialog, well).data(Qt.UserRole + 1)[0] == (dialog._well_fields[(dialog.plates.currentData(), well)][0].identity,)
    qtbot.waitUntil(lambda: {well: _item(dialog, well).data(Qt.UserRole + 1) for well in WELLS} == full,
                    timeout=20000)
    assert {well for kind, well in calls if kind == "all"} == set(WELLS)
    assert calls[0][0] == "all"


def test_figure_settings_category_holds_the_former_figure_controls(source, qtbot):
    from spacr.qt.widgets.section import Section

    dialog = _overview(qtbot, source)
    sections = dialog._settings_content.findChildren(Section)
    titles = [section._title_source for section in sections]
    assert titles == ["Source", "Figure settings", "Display", "Annotations", "Comparisons", "Export"]
    figure = dialog._categories[browser.tr("Figure settings")]
    assert figure.is_expanded()
    for control in (dialog.plates, dialog.rows, dialog.columns, dialog.fov_axis, dialog.time_axis,
                    dialog.z_axis, dialog.z_mode, dialog.wheel_axis, dialog.preview_resolution,
                    dialog.options_button):
        assert figure.isAncestorOf(control)
    assert figure.isAncestorOf(dialog._essential_scroll)
    assert not dialog.isAncestorOf(dialog._essential_scroll) or dialog._options_window.isAncestorOf(
        dialog._essential_scroll)
    assert dialog._splitter.parentWidget() is dialog
    assert dialog.layout().indexOf(dialog._splitter) == 0
    dialog._fit_plate_allocation()
    scroll = dialog._essential_scroll
    assert scroll.property("minimumHeight") == scroll.property("maximumHeight")
    assert scroll.property("minimumHeight") >= scroll.widget().heightForWidth(scroll.viewport().width())


def test_levels_are_translucent_rounded_and_collapse_with_the_shared_arrow(source, qtbot):
    dialog = _overview(qtbot, source)
    panel = dialog._plate_panel
    assert panel.objectName() == "FieldOverviewPanel"
    assert panel.isAncestorOf(dialog.wells)
    qss = browser._overview_panel_qss(dict(theme="dark", border="#333333", **{
        "surface_alt": "#222222"}), 0.6)
    assert "border-radius" in qss and "rgba(" in qss
    assert panel.arrow.sizeHint().width() == 3 * panel.arrow.sizeHint().height()
    assert panel.arrow.minimumSizeHint() == panel.arrow.sizeHint()
    image = panel.arrow.grab().toImage()
    assert not image.isNull()
    qtbot.mouseClick(panel.arrow, Qt.LeftButton)
    assert panel.is_collapsed() and not dialog.wells.isVisible()
    assert panel.property("maximumHeight") == panel.sizeHint().height()
    panel.set_collapsed(False)
    assert not panel.is_collapsed() and dialog.wells.isVisible()
    assert panel.property("maximumHeight") == 16777215
    dialog._well(_item(dialog, "A01"))
    image_panel = dialog._image_panel
    assert image_panel.isVisible() and image_panel.isAncestorOf(dialog.view)
    image_panel.set_collapsed(True)
    assert not dialog.view.isVisible()
    assert image_panel.property("maximumWidth") == 16777215
    image_panel.set_collapsed(False)
    assert dialog.view.isVisible() and image_panel.property("maximumWidth") == 16777215


def test_several_well_views_stack_load_one_image_each_first_and_close(source, qtbot, monkeypatch):
    dialog = _overview(qtbot, source)
    dialog._well(_item(dialog, "A01"))
    first = dialog.fields
    assert first is dialog._well_panel.field_list and first.count() == 3
    dialog._well(_item(dialog, "B01"))
    second = dialog.fields
    assert second is not first and dialog._scope[1] == "B01"
    panels = dialog._well_panels
    assert [panel.field_list for panel in panels] == [first, second]
    assert [dialog._levels.indexOf(panel) for panel in panels] == [1, 2]
    assert all(panel.isVisible() for panel in panels)
    assert first.count() == 3 and second.count() == 3
    assert {first.item(0).data(Qt.UserRole).well, second.item(0).data(Qt.UserRole).well} == {"A01", "B01"}
    dialog._well(_item(dialog, "B01"))
    assert len(dialog._well_panels) == 2 and dialog.fields is second
    qtbot.waitUntil(lambda: all(lst.item(index).data(Qt.UserRole + 1) is not None
                                for lst in (first, second) for index in range(lst.count())), timeout=20000)
    qtbot.waitUntil(lambda: not dialog._montage_jobs.is_busy() and not dialog._field_thumbnail_jobs.is_busy(),
                    timeout=10000)
    picked = []
    real_pixels = browser._overview_pixels

    def pixels(field, *args, **kwargs):
        picked.append(field.well)
        return real_pixels(field, *args, **kwargs)

    monkeypatch.setattr(browser, "_overview_pixels", pixels)
    monkeypatch.setattr(dialog._field_thumbnail_jobs, "submit", lambda read, done: done(read()))
    dialog._comparison_signature = ("held",)
    for lst in (first, second):
        for index in range(lst.count()):
            lst.item(index).setData(Qt.UserRole + 1, None)
    for _ in range(4):
        dialog._update_visuals()
    assert sorted(picked[:2]) == ["A01", "B01"] and sorted(picked[2:4]) == ["A01", "B01"]
    dialog._comparison_signature = None
    monkeypatch.undo()
    dialog._field_in(first, first.item(1))
    assert dialog.fields is first and dialog._scope[1] == "A01" and dialog._individual
    first.setCurrentRow(0)
    assert dialog.fields is first
    second.setCurrentRow(2)
    assert dialog.fields is second and dialog._scope[1] == "B01"
    dialog._clear_selection()
    first.item(0).setSelected(True)
    second.item(1).setSelected(True)
    assert len(dialog._selected_fields) == 2
    dialog._clear_selection()
    assert not dialog._selected_fields and not first.selectedItems()
    dialog._close_well_panel(panels[1])
    assert dialog._well_panels == [dialog._well_panel] and dialog.fields is first
    assert dialog._scope[1] == "A01"
    dialog._well(_item(dialog, "A02"))
    dialog._close_well_panel(dialog._well_panels[0])
    assert dialog.fields is dialog._well_panels[0].field_list and dialog._scope[1] == "A02"
    dialog._well_panels[0].close_button.click()
    assert not dialog._well_panels and dialog._scope[1] is None
    assert not dialog._image_panel.isVisible() and dialog.fields is first
    for well in WELLS:
        dialog._well(_item(dialog, well))
    dialog._well(_item(dialog, "A01"))
    assert len(dialog._well_panels) == 4
    monkeypatch.setattr(browser, "_OVERVIEW_MAX_WELL_VIEWS", 4)
    dialog._well_panels[0].field_list._overview_well = "Z99"
    dialog._well(_item(dialog, "A01"))
    assert len(dialog._well_panels) == 4 and dialog.fields._overview_well == "A01"
    dialog.plates.addItem("p2", "p2")
    dialog.plates.setCurrentIndex(1)
    assert not dialog._well_panels and dialog._scope == ("p2", None)


def test_well_metadata_toggle_shows_captions_like_the_ai_toggles(source, qtbot):
    from spacr.qt.widgets.ai_toggle_label import AiToggleLabel

    dialog = _overview(qtbot, source)
    dialog._well(_item(dialog, "A01"))
    toggle = dialog._well_panel.metadata_toggle
    assert isinstance(toggle, AiToggleLabel) and toggle.text() == "metadata"
    assert not toggle.isChecked()
    assert all(not dialog.fields.item(index).text() for index in range(dialog.fields.count()))
    caption = dialog.fields.item(0).data(Qt.UserRole + 2)
    assert "A01" in caption and caption in dialog.fields.item(0).toolTip()
    qtbot.mouseClick(toggle, Qt.LeftButton)
    assert toggle.isChecked()
    assert dialog.fields.item(0).text() == caption
    dialog._filter()
    assert dialog.fields.item(0).text() == dialog.fields.item(0).data(Qt.UserRole + 2)
    qtbot.mouseClick(toggle, Qt.LeftButton)
    assert not dialog.fields.item(0).text()


def test_clicked_image_enlarges_right_with_its_metadata_toggle(source, qtbot):
    from spacr.qt.widgets.ai_toggle_label import AiToggleLabel

    dialog = _overview(qtbot, source)
    dialog._well(_item(dialog, "A01"))
    assert dialog._splitter.indexOf(dialog._image_column) == 1
    assert dialog._image_stack.indexOf(dialog._image_panel) == 0
    qtbot.waitUntil(lambda: dialog._displayed_montage is not None and not dialog._montage_jobs.is_busy(),
                    timeout=10000)
    assert dialog._montage_labels is False
    toggle = dialog.metadata_toggle
    assert isinstance(toggle, AiToggleLabel) and dialog._image_panel.isAncestorOf(toggle)
    toggle.setChecked(True)
    qtbot.waitUntil(lambda: dialog._montage_labels is True and not dialog._montage_jobs.is_busy(),
                    timeout=10000)
    assert not dialog._image_metadata.isVisible()
    item = dialog.fields.item(1)
    qtbot.mouseClick(dialog.fields.viewport(), Qt.LeftButton,
                     pos=dialog.fields.visualItemRect(item).center())
    qtbot.waitUntil(lambda: dialog._individual and dialog.view._item is not None
                    and not dialog._image_jobs.is_busy(), timeout=10000)
    field = item.data(Qt.UserRole)
    assert dialog._image_metadata.isVisible()
    assert field.caption in dialog._image_metadata.text()
    assert all(Path(path).name in dialog._image_metadata.text() for _channel, path in field.files)
    assert dialog._image_metadata.parentWidget() is dialog.view.viewport()
    assert dialog._image_metadata.pos() == QPoint(8, 8)
    toggle.setChecked(False)
    assert not dialog._image_metadata.isVisible()
    dialog._back()
    assert not dialog._image_panel.isVisible()


def test_right_click_annotates_the_selected_wells(source, qtbot):
    dialog = _overview(qtbot, source)
    item = _item(dialog, "A02")
    menu = dialog._well_menu(dialog.wells.visualItemRect(item).center())
    action = menu.actions()[0]
    assert action.text() == browser.tr("Annotate selected wells…") and action.isEnabled()
    assert dialog._selected_wells == {("p1", "A02")}
    menu.close()
    _item(dialog, "B01").setSelected(True)
    menu = dialog._well_menu(dialog.wells.visualItemRect(item).center())
    assert dialog._selected_wells == {("p1", "A02"), ("p1", "B01")}
    menu.close()
    prompt = dialog._annotate_selected_wells()
    assert prompt.wells == (("p1", "A02"), ("p1", "B01"))
    prompt.key.setCurrentIndex(prompt.key.findData("cell_type"))
    prompt.value.setText(" HeLa ")
    assert prompt.annotation() == ("cell_type", "HeLa")
    prompt.accept()
    assert dialog._annotations[("p1", "A02")]["cell_type"] == "HeLa"
    assert dialog._annotations[("p1", "B01")]["cell_type"] == "HeLa"
    prompt = dialog._annotate_selected_wells()
    assert prompt.key.findData("cell_type") >= 0
    prompt.key.setEditText("replicate_id")
    prompt.value.setText("2")
    assert prompt.annotation() == ("replicate_id", "2")
    prompt.accept()
    assert dialog._annotations[("p1", "B01")]["replicate_id"] == "2"
    prompt = dialog._annotate_selected_wells()
    assert prompt.key.findData("replicate_id") >= 0
    prompt.reject()
    dialog._clear_selection()
    menu = dialog._well_menu(QPoint(-50, -50))
    assert not menu.actions()[0].isEnabled()
    menu.close()
    assert dialog._annotate_selected_wells() is None


def test_comparison_category_offers_every_annotation_dimension_and_value(source, qtbot):
    from spacr.qt.widgets.section import Section

    dialog = _overview(qtbot, source)
    comparisons = dialog._categories[browser.tr("Comparisons")]
    assert isinstance(comparisons, Section) and comparisons.isAncestorOf(dialog.comparison_dimensions)
    assert comparisons.isAncestorOf(dialog.representative_method)
    assert [dialog.representative_method.itemData(index)
            for index in range(dialog.representative_method.count())] == [
        "median", "medoid", "pca_centroid", "measurement", "typical_in_focus"]
    dialog._compare_groups()
    assert dialog.status.text() == browser.tr("Tick the annotation values to compare.")
    dialog._show_representatives()
    assert dialog.status.text() == browser.tr("Tick the annotation values to compare.")
    for wells, key, value in (({("p1", "A01"), ("p1", "A02")}, "cell_type", "HeLa"),
                              ({("p1", "B01"), ("p1", "B03")}, "cell_type", "HFF"),
                              ({("p1", "A01"), ("p1", "B01")}, "replicate", "1"),
                              ({("p1", "A02"), ("p1", "B03")}, "replicate", "2")):
        dialog._apply_annotation(wells, key, value)
    tree = dialog.comparison_dimensions
    assert [tree.topLevelItem(index).text(0) for index in range(tree.topLevelItemCount())] == [
        "cell_type", "replicate"]
    assert [item.text(0) for item in dialog._comparison_items()] == ["HeLa", "HFF", "1", "2"]
    for item in dialog._comparison_items():
        if item.text(0) in ("HFF", "HeLa", "2"):
            item.setCheckState(0, Qt.Checked)
    assert dialog._comparison_selection() == (
        ["cell_type", "replicate"], {"cell_type": ["HeLa", "HFF"], "replicate": ["2"]}, None)
    dialog.comparison_within.setCurrentIndex(dialog.comparison_within.findData("replicate"))
    assert dialog._comparison_selection() == (["cell_type"], {"cell_type": ["HeLa", "HFF"]}, "replicate")
    dialog._apply_annotation({("p1", "A01")}, "pathogen", "WT")
    assert [item.text(0) for item in dialog._comparison_items() if item.checkState(0) == Qt.Checked] == [
        "HeLa", "HFF", "2"]
    assert dialog.comparison_within.currentData() == "replicate"
    with pytest.raises(browser._OverviewValidationError):
        browser._overview_comparison_frame(str(source))


def test_annotations_reach_the_image_table_and_drive_comparisons_and_representatives(source, qtbot):
    from spacr.measure import (_image_annotation_database, _read_image_annotations,
                               _well_image_annotations)

    dialog = _overview(qtbot, source)
    database = _image_annotation_database(str(source))
    for wells, key, value in ((("A01", "A02"), "cell_type", "HeLa"), (("B01", "B03"), "cell_type", "HFF"),
                              (("A01", "B01"), "replicate", "1"), (("A02", "B03"), "replicate", "2")):
        dialog._clear_selection()
        for well in wells:
            _item(dialog, well).setSelected(True)
        prompt = dialog._annotate_selected_wells()
        prompt.key.setCurrentIndex(prompt.key.findData(key))
        prompt.value.setText(value)
        prompt.accept()
        qtbot.waitUntil(lambda: dialog.status.text().startswith(
            browser.tr("Saved {name} for {count} images in {path} (table image_annotations).",
                       name=key, count=0, path=database).split(" ")[0]) and not dialog._metadata_jobs.is_busy(),
            timeout=10000)
    table = _read_image_annotations(database)
    assert len(table) == len(WELLS) * 3
    assert _well_image_annotations(database)[("p1", "A02")] == {"cell_type": "HeLa", "replicate": "2"}
    dialog._apply_annotation({("p1", "A01")}, "well", "x")
    assert dialog.status.text().startswith(browser.tr(
        "Annotation names use letters, digits and underscores, start with a letter and cannot name an image identity: {name}",
        name="well"))
    assert "well" not in dialog._annotations[("p1", "A01")]
    reopened = _overview(qtbot, source)
    qtbot.waitUntil(lambda: reopened._annotations.get(("p1", "B03"), {}).get("cell_type") == "HFF",
                    timeout=10000)
    for item in dialog._comparison_items():
        item.setCheckState(0, Qt.Checked)
    dialog._compare_groups()
    qtbot.waitUntil(lambda: not dialog._export_jobs.is_busy(), timeout=20000)
    assert dialog.status.text().startswith(browser.tr(
        "Export image-level measurements first; comparisons read them from {path}.", path=database)[:20])
    dialog._show_representatives()
    qtbot.waitUntil(lambda: not dialog._export_jobs.is_busy(), timeout=20000)
    dialog.workers.setValue(2)
    dialog._measure()
    qtbot.waitUntil(lambda: not dialog._measurement_jobs.is_busy(), timeout=60000)
    dialog.comparison_within.setCurrentIndex(dialog.comparison_within.findData("replicate"))
    dialog._compare_groups()
    qtbot.waitUntil(lambda: not dialog._export_jobs.is_busy(), timeout=60000)
    assert dialog._comparison_signature is not None, dialog.status.text()
    assert dialog._comparison_signature[0] == "groups", dialog.status.text()
    assert dialog._image_panel.isVisible() and not dialog.view._item.pixmap().isNull()
    assert dialog.comparison_measurement.count() > 1
    assert list((source / "results" / "overview_comparisons").glob("*_tests.csv"))
    dialog.comparison_measurement.setCurrentText("no_such_measurement")
    dialog._compare_groups()
    qtbot.waitUntil(lambda: not dialog._export_jobs.is_busy(), timeout=60000)
    assert dialog.status.text() == browser.tr("These annotation groups cannot be compared with {measurement}.",
                                              measurement="no_such_measurement")
    dialog.comparison_measurement.setCurrentText("")
    dialog.comparison_within.setCurrentIndex(0)
    for method in ("median", "measurement"):
        dialog.representative_method.setCurrentIndex(dialog.representative_method.findData(method))
        dialog._show_representatives()
        qtbot.waitUntil(lambda: not dialog._export_jobs.is_busy(), timeout=60000)
    assert dialog.status.text() == browser.tr(
        "No representative image can be chosen this way for these annotation values.")
    dialog.representative_method.setCurrentIndex(0)
    dialog._show_representatives()
    qtbot.waitUntil(lambda: dialog.status.text().startswith(
        browser.tr("Representative images: {classes}", classes="")) and not dialog._montage_jobs.is_busy(),
        timeout=60000)
    assert dialog.status.text().count("+") == 4
    assert dialog._comparison_signature is not None and len(dialog._comparison_signature[0]) == 4
    for item in dialog._comparison_items():
        item.setCheckState(0, Qt.Unchecked)
    dialog._apply_annotation({("p1", "A01")}, "cell_type", "")
    qtbot.waitUntil(lambda: not dialog._metadata_jobs.is_busy(), timeout=10000)
    assert "cell_type" not in _well_image_annotations(database).get(("p1", "A01"), {})


def test_annotation_dialog_counts_one_well_in_the_singular(qtbot):
    from PySide6.QtWidgets import QLabel

    def note(wells):
        dialog = browser._OverviewAnnotationDialog([], wells)
        qtbot.addWidget(dialog)
        return [label.text() for label in dialog.findChildren(QLabel)]

    assert browser.tr("One selected well") in note([("p1", "A1")])
    assert browser.tr("{count} selected wells", count=2) in note([("p1", "A1"), ("p1", "A2")])
