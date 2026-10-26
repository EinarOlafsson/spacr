"""Round-4 Overview: standard tooltips, channel options, settings width, navigation, annotation, groups.

Item 725, maintainer feedback 2026-10-25 round 4, points 1-6 and the UI side of 9 and 10.
"""
import numpy as np
import pytest
import tifffile

from PySide6.QtWidgets import QPushButton

from spacr.qt.widgets import qc_field_browser as browser


PATTERN = r"(?P<plateID>p\d+)-(?P<wellID>[A-Z]\d+)-f(?P<fieldID>\d+)-c(?P<chanID>\d+)\.tif"
WELLS = ("A01", "A02", "B01")


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    for index, well in enumerate(WELLS):
        for field in (1, 2):
            for channel in (1, 2):
                pixels = (np.arange(48 * 64, dtype=np.uint16).reshape(48, 64) * channel
                          + 100 * index + field)
                tifffile.imwrite(folder / f"p1-{well}-f{field}-c{channel}.tif", pixels)
    return folder


def _runners(view):
    return (view._index_jobs, view._image_jobs, view._thumbnail_jobs, view._field_thumbnail_jobs,
            view._montage_jobs, view._export_jobs, view._metadata_jobs, view._measurement_jobs,
            view._native_jobs, view._image_stack_jobs, view._group_jobs)


def _close(mask, qtbot):
    tab = getattr(mask, "_overview_tab", None)
    page = getattr(tab, "_overview_page", None)
    if page is not None and page.browser is not None:
        view = page.browser
        view._cancel()
        qtbot.waitUntil(lambda: all(runner.active_jobs() == 0 for runner in _runners(view)), timeout=15000)
        page.retire()
    from spacr.qt.bridge import prune_parked_threads
    qtbot.waitUntil(lambda: prune_parked_threads() == 0, timeout=15000)
    if tab is not None:
        tab.close()
    mask.close()


@pytest.fixture
def mask(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("mask")
    qtbot.addWidget(screen, before_close_func=lambda widget: _close(widget, qtbot))
    screen.resize(1500, 1000)
    screen.show()
    return screen


def _open_tab(mask, source):
    mask._settings_model.set_value_for_key("src", str(source))
    field = mask._settings_model._widgets.get("src")
    next(action for action in field.actions()
         if action.objectName() == "GenerateMasksFieldOverview").trigger()
    return mask._overview_tab


def _browse(mask, qtbot, source):
    tab = _open_tab(mask, source)
    wizard = tab._overview_page.wizard
    wizard.preset.setCurrentIndex(wizard.preset.count() - 1)
    wizard.regex.setText(PATTERN)
    wizard._scan()
    qtbot.waitUntil(lambda: wizard.next.isEnabled(), timeout=15000)
    wizard._next()
    page = tab._overview_page
    qtbot.waitUntil(lambda: page.browser is not None and len(page.browser._fields) == 2 * len(WELLS)
                    and page.browser._normalization_ready, timeout=15000)
    return tab, page.browser


def _row_shown(field):
    from PySide6.QtWidgets import QFormLayout
    from spacr.qt.settings_search import _row_is_visible

    node = field
    while node.parentWidget() is not None:
        parent = node.parentWidget()
        if isinstance(parent.layout(), QFormLayout) and parent.layout().getWidgetPosition(node)[0] >= 0:
            return _row_is_visible(parent, node)
        node = parent
    return not field.isHidden()


def test_every_mounted_row_has_the_standard_popup_with_its_api_link(mask, qtbot, source):
    from spacr.qt.screens.settings_model import _setting_label_for_field

    tab, _view = _browse(mask, qtbot, source)
    rows = tab._overview_page._rows
    assert rows
    for _section, field, _widgets in rows:
        key = field.property("settingKey")
        assert key and key.startswith("overview_")
        label = _setting_label_for_field(tab, field)
        assert label is not None, key
        html = str(label.property("apiTooltipHtml") or "")
        assert label.toolTip() == html and "<a href=" in html and html.rstrip().endswith("</a>"), key
        assert field.toolTip() == ""


def test_the_separate_channels_row_is_gone_and_overview_channels_drives_the_views(mask, qtbot, source):
    tab, view = _browse(mask, qtbot, source)
    fields = [field for _section, field, _widgets in tab._overview_page._rows]
    assert view._channel_host not in fields
    assert not any(field.property("settingKey") == "overview_channels" and field is not
                   tab._settings_model._widgets["overview_channels"] for field in fields)
    tab._settings_model.set_value_for_key("overview_channels", "2")
    assert view.channels.text() == "2"


@pytest.mark.parametrize("key", ["overview_enhance_channels"])
def test_channel_targets_are_one_switch_per_line(mask, qtbot, source, key):
    tab, view = _browse(mask, qtbot, source)
    model = tab._settings_model
    toggles = next(field for _section, field, _widgets in tab._overview_page._rows
                   if isinstance(field, browser._OverviewChannelToggles) and field._key == key)
    qtbot.waitUntil(lambda: len(toggles._toggles) == 2, timeout=15000)
    assert not _row_shown(model._widgets[key])
    first, second = toggles._toggles.values()
    assert first.isChecked() and second.isChecked()
    from PySide6.QtWidgets import QVBoxLayout

    assert isinstance(toggles.layout(), QVBoxLayout)
    assert toggles.layout().indexOf(first) == 0 and toggles.layout().indexOf(second) == 1
    assert first.text() == "Channel 1" and second.text() == "Channel 2"
    second.setChecked(False)
    assert model._read_value(key) == "1"
    first.setChecked(False)
    assert model._read_value(key) == "none"
    model.set_value_for_key(key, "all")
    assert first.isChecked() and second.isChecked()
    model.set_value_for_key(key, "2")
    assert not first.isChecked() and second.isChecked()
    tab._overview_page._retire_browser()
    assert _row_shown(model._widgets[key])


def test_settings_open_wide_with_the_tab_and_stay_open_after_next(mask, qtbot, source):
    tab = _open_tab(mask, source)
    body = tab._body_splitter
    qtbot.waitUntil(lambda: sum(body.sizes()) > 0, timeout=5000)
    split = tab._overview_page.wizard._wizard_content
    assert body.is_collapsed("Settings")
    assert split.sizes()[0] >= min(browser._OVERVIEW_SETTINGS_WIDTH, sum(split.sizes()) * 3 // 5)
    tab, _view = _browse(mask, qtbot, source)
    qtbot.wait(50)
    assert not body.is_collapsed("Settings")
    assert body.sizes()[0] >= min(browser._OVERVIEW_SETTINGS_WIDTH, sum(body.sizes()) // 2)


def test_navigation_is_in_the_plate_right_click_menu_not_a_button_bar(mask, qtbot, source):
    tab, view = _browse(mask, qtbot, source)
    captions = {str(field.property("settingKey")) for _section, field, _widgets in tab._overview_page._rows}
    assert "overview_plate_navigation" not in captions and "overview_fields" not in captions
    assert view._navigation_bar is None
    holder = view._levels.parentWidget()
    assert holder.layout().indexOf(view._levels) == 0 and holder.layout().count() == 1
    assert view._levels.indexOf(view._plate_panel) == 0
    shown = [button.text() for button in view.findChildren(QPushButton) if button.isVisible()]
    for text in ("Back to plate", "Fit plate", "−", "+", "Larger wells", "Smaller wells", "Previous plate",
                 "Next plate", "Add plates…", "Remove plate", "Shown plates…", "Previous fields", "Next fields",
                 "Compare selected fields", "Open selected well", "Open selected images"):
        assert text not in shown, text
    assert not view.plates_view.isVisible()
    actions = _menu_actions(view, "A01")
    texts = list(actions)
    assert texts.index("Navigation") < texts.index("Wells") < texts.index("Plates")
    fit, larger, smaller = actions["Fit plate"], actions["Zoom in"], actions["Zoom out"]
    fit.trigger()
    extent = view._plate_cell_extent
    larger.trigger()
    assert view._plate_cell_extent[0] > extent[0]
    smaller.trigger()
    smaller.trigger()
    assert view._plate_cell_extent[0] < extent[0]
    item = view.wells.item(*view._well_positions[(view.plates.currentData(), "A01")])
    view.wells.itemDoubleClicked.emit(item)
    qtbot.waitUntil(lambda: view._scope[1] == "A01", timeout=5000)
    actions = _menu_actions(view, "A01")
    assert {"Previous fields", "Next fields", "Compare selected fields"} <= set(actions)
    actions["Back to plate"].trigger()
    assert view._scope[1] is None


def _menu_actions(view, well):
    rect = view.wells.visualItemRect(view.wells.item(*view._well_positions[(view.plates.currentData(), well)]))
    menu = view._well_menu(rect.center())
    actions = {action.text(): action for action in menu.actions() if action.text()}
    menu.close()
    return actions


def test_annotation_lives_in_the_plate_menu_and_its_category_is_gone(mask, qtbot, source):
    tab, view = _browse(mask, qtbot, source)
    titles = {str(section.property("settingsCategorySource")) for section in tab._settings_sections}
    assert "Annotations" not in titles and "Comparisons" in titles
    assert "overview_annotation_scope" not in tab._settings_model._widgets
    keys = {str(field.property("settingKey")) for _section, field, _widgets in tab._overview_page._rows}
    assert not {"overview_metadata", "overview_value", "overview_annotation_actions"} & keys
    actions = _menu_actions(view, "A01")
    texts = list(actions)
    assert texts[:texts.index("Navigation")] == [
        "Annotate selected wells…", "Annotate row A…", "Annotate column 1…",
        "Annotate the whole plate…", "Collapse row A", "Collapse column 1",
        "Show all rows and columns", "Import annotation scheme…",
        "Save annotations…", "Load annotations…"]
    actions["Annotate row A…"].trigger()
    dialog = view._annotation_dialog
    assert set(dialog.wells) == {("p1", "A01"), ("p1", "A02")}
    dialog.key.setCurrentIndex(dialog.key.findData("drug"))
    dialog.value.setText("DMSO")
    dialog.accept()
    assert view._annotations[("p1", "A01")]["drug"] == "DMSO"
    assert view._annotations[("p1", "A02")]["drug"] == "DMSO"
    assert "drug" not in view._annotations.get(("p1", "B01"), {})
    _menu_actions(view, "B01")["Annotate the whole plate…"].trigger()
    assert set(view._annotation_dialog.wells) == {("p1", well) for well in WELLS}
    view._annotation_dialog.reject()


def test_a_well_annotation_scheme_imports_from_a_table_or_a_plate_layout(mask, qtbot, source, tmp_path,
                                                                          monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    from spacr.measure import _image_annotation_database, _read_image_annotations

    _tab, view = _browse(mask, qtbot, source)
    table = tmp_path / "scheme.csv"
    table.write_text("well,cell_type,drug\nA1,HeLa,DMSO\nA02,HeLa,Taxol\nB01,HFF,\n", encoding="utf-8")
    layout = tmp_path / "pathogen_type.csv"
    layout.write_text("row,1,2\nA,WT,Ku80\nB,WT,\n", encoding="utf-8")
    chosen = [str(table)]
    asked = []
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (chosen[0], ""))
    monkeypatch.setattr(QMessageBox, "question",
                        lambda parent, title, text, *args, **kwargs: asked.append(text) or QMessageBox.Yes)
    _menu_actions(view, "A01")["Import annotation scheme…"].trigger()
    qtbot.waitUntil(lambda: view._annotations.get(("p1", "B01"), {}).get("cell_type") == "HFF", timeout=30000)
    assert "3 wells" in asked[0] and "cell_type, drug" in asked[0] and "0 wells of the scheme" in asked[0]
    assert view._annotations[("p1", "A01")] == {"cell_type": "HeLa", "drug": "DMSO"}
    assert view._annotations[("p1", "A02")] == {"cell_type": "HeLa", "drug": "Taxol"}
    assert view._annotations[("p1", "B01")] == {"cell_type": "HFF"}
    qtbot.waitUntil(lambda: view.status.text().startswith("Imported") or not view._export_jobs.is_busy(),
                    timeout=30000)
    stored = _read_image_annotations(_image_annotation_database(view._metadata_source))
    assert set(stored.loc[stored["wellID"] == "B01", "cell_type"]) == {"HFF"}
    chosen[0] = str(layout)
    qtbot.waitUntil(lambda: not view._export_jobs.is_busy(), timeout=30000)
    view._import_annotation_scheme()
    qtbot.waitUntil(lambda: view._annotations[("p1", "B01")].get("pathogen_type") == "WT", timeout=30000)
    assert view._annotations[("p1", "A02")]["pathogen_type"] == "Ku80"
    foreign = tmp_path / "foreign.csv"
    foreign.write_text("well,drug\nH12,DMSO\n", encoding="utf-8")
    chosen[0] = str(foreign)
    qtbot.waitUntil(lambda: not view._export_jobs.is_busy(), timeout=30000)
    before = {address: dict(values) for address, values in view._annotations.items()}
    count = len(asked)
    view._import_annotation_scheme()
    qtbot.waitUntil(lambda: len(asked) > count, timeout=30000)
    assert "Annotate 0 wells" in asked[-1] and "1 wells of the scheme are not in this source" in asked[-1]
    qtbot.waitUntil(lambda: not view._export_jobs.is_busy(), timeout=30000)
    assert view._annotations == before


def test_object_masks_are_picked_from_a_folder_and_refused_with_the_reason(mask, qtbot, source, tmp_path,
                                                                           monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    tab, view = _browse(mask, qtbot, source)
    edit = tab._settings_model._widgets["overview_objects_path"]
    action = edit._spacr_folder_browse_action
    plain = tmp_path / "plain"
    plain.mkdir()
    chosen = [str(plain)]
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args, **kwargs: chosen[0])
    action.trigger()
    assert view.status.text().startswith("These object masks cannot be used:")
    assert "mask_stack" in view.status.text() and edit.text() == ""
    stack = tmp_path / "plate" / "masks" / "cell_mask_stack"
    stack.mkdir(parents=True)
    tifffile.imwrite(stack / "p1-A01-f1.tif", np.ones((48, 64), np.uint16))
    chosen[0] = str(tmp_path / "plate")
    action.trigger()
    assert edit.text() == str(tmp_path / "plate")
    assert tab._settings_model._read_value("overview_objects_path") == str(tmp_path / "plate")
    assert view.status.text() == "Object masks: 1 files of cell."


def test_overview_measurements_add_the_imported_object_record(monkeypatch, source):
    import spacr.measure as measure

    calls = []

    def record(planes, files, plate, well, field, settings=None):
        calls.append((plate, well, field, dict(settings or {})))
        return {"cell_objects_count": 7}

    monkeypatch.setattr(measure, "_image_object_record", record)
    fields = tuple(field for field in browser._overview_index(source, PATTERN) if not field.unresolved)
    row = browser._overview_measurement(fields[0], (1,), measurements={"overview_objects_path": "x"})
    assert row["cell_objects_count"] == 7 and calls[0][:2] == (fields[0].plate, fields[0].well)
    assert calls[0][3]["overview_objects_path"] == "x"


def test_comparison_groups_can_be_named(mask, qtbot, source, monkeypatch):
    import pandas as pd
    from PySide6.QtWidgets import QInputDialog

    _tab, view = _browse(mask, qtbot, source)
    view._annotations = {("p1", "A01"): {"cell_type": "HeLa"}, ("p1", "A02"): {"cell_type": "HeLa"},
                         ("p1", "B01"): {"cell_type": "HFF"}}
    view._refresh_comparison_choices()
    builder = view.group_builder
    assert builder.add_group({"cell_type": "HeLa"}) and builder.add_group({"cell_type": "HFF"})
    assert builder.names() == ["", ""]
    builder.groups.setCurrentRow(0)
    monkeypatch.setattr(QInputDialog, "getText", lambda *args, **kwargs: ("Human cervix", True))
    builder.buttons["name"].click()
    assert builder.names() == ["Human cervix", ""]
    assert builder.groups.item(0).text() == "Human cervix (HeLa) — 4 images"
    assert builder.definitions() == [{"cell_type": "HeLa"}, {"cell_type": "HFF"}]
    assert not builder.rename_group(row=1, name="Human cervix")
    assert not builder.rename_group(row=1, name="HeLa")
    builder.groups.itemDoubleClicked.emit(builder.groups.item(1))
    assert builder.names() == ["Human cervix", ""]
    assert builder.rename_group(row=1, name="Foreskin fibroblast")
    builder.refresh_counts()
    assert builder.groups.item(1).text() == "Foreskin fibroblast (HFF) — 2 images"
    frame = pd.DataFrame({"cell_type": ["HeLa", "HFF", "HeLa"], "value": [1, 2, 3]})
    grouped = browser._overview_grouped_frame(frame, builder.definitions(), builder.names())
    assert list(grouped["_overview_group"]) == ["Human cervix", "Human cervix", "Foreskin fibroblast"]
    assert builder.rename_group(row=0, name="")
    assert builder.groups.item(0).text() == "HeLa — 4 images"


def test_group_names_are_saved_restored_and_reach_the_comparison(mask, qtbot, source, monkeypatch):
    import spacr.sp_stats as sp_stats
    from spacr.measure import _image_annotation_database, _read_comparison_groups

    tab, view = _browse(mask, qtbot, source)
    view._annotations = {("p1", "A01"): {"cell_type": "HeLa"}, ("p1", "B01"): {"cell_type": "HFF"}}
    view._refresh_comparison_choices()
    builder = view.group_builder
    builder.add_group({"cell_type": "HeLa"})
    builder.add_group({"cell_type": "HFF"})
    builder.rename_group(row=0, name="Cervix")
    database = _image_annotation_database(view._metadata_source)
    qtbot.waitUntil(lambda: database.is_file() and _read_comparison_groups(database)[1] == ["Cervix", "HFF"],
                    timeout=30000)
    seen = {}

    def compare(*args, **kwargs):
        seen.update(kwargs)
        raise ValueError("stop here")

    monkeypatch.setattr(sp_stats, "_compare_image_groups", compare)
    monkeypatch.setattr(browser, "_overview_comparison_frame",
                        lambda source: __import__("pandas").DataFrame({"cell_type": ["HeLa"], "x": [1.0]}))
    view._compare_groups()
    qtbot.waitUntil(lambda: "group_names" in seen, timeout=30000)
    assert seen["group_names"] == ["Cervix", ""]
    page = tab._overview_page
    page._retire_browser()
    model = tab._settings_model
    page.run(model.collect())
    wizard = page.wizard
    wizard.preset.setCurrentIndex(wizard.preset.count() - 1)
    wizard.regex.setText(PATTERN)
    wizard._scan()
    qtbot.waitUntil(lambda: wizard.next.isEnabled(), timeout=15000)
    wizard._next()
    qtbot.waitUntil(lambda: page.browser is not None and page.browser.group_builder.groups.count() == 2,
                    timeout=30000)
    assert page.browser.group_builder.names() == ["Cervix", ""]
    assert page.browser.group_builder.definitions() == [{"cell_type": "HeLa"}, {"cell_type": "HFF"}]
