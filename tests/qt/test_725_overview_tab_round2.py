"""Round-2 Overview: a closable tab in Mask generation with module-style settings.

Covers the 2026-10-24 round-2 feedback of item 725: the hidden source icon
opens Overview in a tab beside Mask's own page; its settings are ordinary
category rows with names and tooltips; Image and Enhancement settings drive
the views; Original loads full resolution; the size slider sizes the tiles;
several fields open as stacked images; and layered condition groups run
through the image-comparison data layer.
"""
import numpy as np
import pandas as pd
import pytest
import tifffile

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QLineEdit, QPushButton, QWidget

from spacr.qt.widgets import qc_field_browser as browser


PATTERN = r"(?P<plateID>p\d+)-(?P<wellID>[A-Z]\d+)-f(?P<fieldID>\d+)-c(?P<chanID>\d+)\.tif"
WELLS = ("A01", "A02", "B01")


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    for index, well in enumerate(WELLS):
        for field in (1, 2):
            pixels = (np.arange(400 * 600, dtype=np.uint32).reshape(400, 600) % 4000
                      + 100 * index + field).astype(np.uint16)
            tifffile.imwrite(folder / f"p1-{well}-f{field}-c1.tif", pixels)
    return folder


def _runners(view):
    return (view._index_jobs, view._image_jobs, view._thumbnail_jobs, view._field_thumbnail_jobs,
            view._montage_jobs, view._export_jobs, view._metadata_jobs, view._measurement_jobs,
            view._native_jobs, view._image_stack_jobs)


def _close(mask, qtbot):
    tab = getattr(mask, "_overview_tab", None)
    page = getattr(tab, "_overview_page", None)
    if page is not None and page.browser is not None:
        view = page.browser
        view._cancel()
        qtbot.waitUntil(lambda: all(runner.active_jobs() == 0 for runner in _runners(view)), timeout=15000)
        for runner in _runners(view):
            runner.shutdown(timeout_ms=3000)
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
    screen.resize(1400, 1000)
    screen.show()
    return screen


def _icon(mask):
    field = mask._settings_model._widgets.get("src")
    assert isinstance(field, QLineEdit)
    return next(action for action in field.actions() if action.objectName() == "GenerateMasksFieldOverview")


def _open(mask, qtbot, source):
    mask._settings_model.set_value_for_key("src", str(source))
    _icon(mask).trigger()
    tab = mask._overview_tab
    wizard = tab._overview_page.wizard
    wizard.preset.setCurrentIndex(wizard.preset.count() - 1)
    wizard.regex.setText(PATTERN)
    wizard._scan()
    qtbot.waitUntil(lambda: wizard.next.isEnabled(), timeout=15000)
    wizard._next()
    page = tab._overview_page
    qtbot.waitUntil(lambda: page.browser is not None, timeout=15000)
    tab._settings_model.set_value_for_key("overview_max_fields", "all")
    qtbot.waitUntil(lambda: page.browser is not None and len(page.browser._fields) == 6
                    and page.browser._normalization_ready, timeout=15000)
    return tab, page.browser


def _open_well(view, qtbot, well):
    view._well(view.wells.item(*view._well_positions[(view.plates.currentData(), well)]))
    qtbot.waitUntil(lambda: view.fields.count() > 0, timeout=15000)
    return view._well_panels[-1]


def _sections(tab):
    return {str(section.property("settingsCategorySource")): section for section in tab._settings_sections}


def test_icon_opens_a_tab_mask_stays_one_click_away_and_the_icon_reopens_it(mask, qtbot, source):
    tab, _view = _open(mask, qtbot, source)
    pages = mask._fold_pages
    assert [pages.tabText(index) for index in range(pages.count())] == ["Mask Generation", "Overview"]
    assert pages.currentWidget() is tab
    mask_page = pages.widget(0)
    assert mask._settings_scroll.widget() is mask._settings_content
    pages.setCurrentIndex(0)
    assert pages.currentWidget() is mask_page and mask_page.isVisible() and not tab.isVisible()
    assert mask._settings_scroll.isVisible()
    pages.tabCloseRequested.emit(1)
    assert pages.count() == 1 and pages.currentWidget() is mask_page
    _icon(mask).trigger()
    assert pages.count() == 2 and pages.currentWidget() is tab
    assert mask._overview_tab is tab


def test_every_setting_row_has_a_name_and_a_tooltip(mask, qtbot, source):
    tab, _view = _open(mask, qtbot, source)
    model = tab._settings_model
    checked = 0
    for title, section in _sections(tab).items():
        if not section.is_expanded():
            section.set_expanded(True)
        form = section._form
        for row in range(form.rowCount()):
            label_item = form.itemAt(row, form.ItemRole.LabelRole)
            field_item = form.itemAt(row, form.ItemRole.FieldRole)
            if label_item is None or field_item is None or label_item.widget() is None:
                continue
            labels = [label for label in label_item.widget().findChildren(QLabel)] or [label_item.widget()]
            label = labels[0]
            assert label.text().strip(), (title, row)
            key = field_item.widget().property("settingKey") or label.property("settingKey")
            if key:
                assert model.tooltip_for(str(key)).strip(), (title, key)
            else:
                assert label.toolTip().strip(), (title, label.text())
            checked += 1
    assert checked >= 70


def test_export_container_grows_and_shrinks_with_its_grip(mask, qtbot, source):
    tab, _view = _open(mask, qtbot, source)
    export = _sections(tab)["Export"]
    export.set_expanded(True)
    grip = export._overview_height_grip
    assert isinstance(grip, browser._OverviewHeightGrip) and grip.isVisible()
    tab._settings_layout.activate()
    before = export.sizeHint().height()
    grip.set_height(export._body.sizeHint().height() + 300)
    assert export.sizeHint().height() >= before + 250
    grip.set_height(0)
    assert export.sizeHint().height() == before


def test_categories_are_drawn_exactly_like_mask_generation(mask, qtbot, source):
    from spacr.qt.widgets.section import Section

    tab, _view = _open(mask, qtbot, source)
    theirs = mask._settings_sections[0]
    for section in _sections(tab).values():
        assert type(section) is type(theirs) is Section
        assert section.objectName() == theirs.objectName() == "SectionCard"
        assert section.header().objectName() == theirs.header().objectName()
        assert section._body.objectName() == theirs._body.objectName()
    assert tab._settings_scroll.objectName() == mask._settings_scroll.objectName()
    assert _sections(tab)["Enhancement"].parentWidget() is not tab._settings_content


def test_size_slider_and_settings_size_the_well_images(mask, qtbot, source):
    tab, view = _open(mask, qtbot, source)
    panel = _open_well(view, qtbot, "A01")
    panel.tile_slider.setValue(240)
    assert view._tile_size == 240
    assert panel.field_list.iconSize().width() == 240
    assert panel.field_list.gridSize().width() == 240 + view._tile_spacing
    assert tab._settings_model._read_value("overview_tile_size") == 240
    tab._settings_model.set_value_for_key("overview_tile_size", 96)
    assert panel.tile_slider.value() == 96 and panel.field_list.iconSize().width() == 96
    tab._settings_model.set_value_for_key("overview_tile_spacing", 0)
    assert panel.field_list.gridSize().width() == 96


@pytest.mark.parametrize("key,control,value,expected", [
    ("overview_preview_resolution", "preview_resolution", "256", 256),
    ("overview_preview_resolution", "preview_resolution", "512", 512),
    ("overview_pdf_dpi", "pdf_dpi", "150", 150),
    ("overview_pdf_quality", "pdf_quality", "3200", 3200),
])
def test_every_offered_preview_and_pdf_value_reaches_its_control(
        mask, qtbot, source, key, control, value, expected):
    tab, view = _open(mask, qtbot, source)
    tab._settings_model.set_value_for_key(key, value)
    assert getattr(view, control).currentData() == expected


def test_the_top_quality_loads_full_resolution_in_the_image_and_well_views(mask, qtbot, source, monkeypatch):
    tab, view = _open(mask, qtbot, source)
    tab._settings_model.set_value_for_key("overview_preview_resolution", 128)
    assert view.preview_resolution.currentData() == 128
    edges = []
    real = browser._overview_pixels

    def recorded(field, *args, edge=1200, **kwargs):
        edges.append(edge)
        return real(field, *args, edge=edge, **kwargs)

    monkeypatch.setattr(browser, "_overview_pixels", recorded)
    _open_well(view, qtbot, "A01")
    view.fields.setCurrentRow(0)
    view._field_in(view.fields, view.fields.item(0))
    qtbot.waitUntil(lambda: view.view._item is not None, timeout=15000)
    assert max(view.view._item.pixmap().width(), view.view._item.pixmap().height()) <= 128
    header = view._image_panel.header_layout
    assert not hasattr(view, "original_toggle")
    assert header.indexOf(view.resolution_slider) == header.indexOf(view.metadata_toggle) - 1
    panel = view._well_panels[-1]
    assert not hasattr(panel, "original_toggle")
    edges.clear()
    view.resolution_slider.setValue(view.resolution_slider.maximum())
    assert view.preview_resolution.currentText() == "Original"
    qtbot.waitUntil(lambda: view.view._item is not None and view.view._item.pixmap().width() == 600,
                    timeout=15000)
    assert view.view._item.pixmap().height() == 400
    qtbot.waitUntil(lambda: None in edges, timeout=15000)
    qtbot.waitUntil(lambda: (panel.field_list.item(0).data(Qt.UserRole + 1) or (0,))[-1] is None, timeout=15000)
    assert tab._settings_model._read_value("overview_preview_resolution") == 0


def test_image_and_enhancement_settings_change_what_is_shown(mask, qtbot, source):
    tab, view = _open(mask, qtbot, source)
    model = tab._settings_model
    assert view._enhancement is None
    model.set_value_for_key("enhance_gamma", 0.5)
    assert view._enhancement is None
    tab._overview_page._apply_settings()
    assert view._enhancement is not None and view._enhancement.gamma == 0.5
    field = view._fields[0]
    plain = browser._overview_pixels(field, (1,), (1., 99.), edge=128)
    enhanced = browser._overview_pixels(field, (1,), (1., 99.), edge=128, enhancement=view._enhancement)
    assert enhanced.mean() > plain.mean()
    assert view._display_kwargs(view=True)["enhancement"] is view._enhancement
    assert "enhancement" not in view._display_kwargs()
    model.set_value_for_key("enhance_gamma", 1.0)
    tab._overview_page._apply_settings()
    assert view._enhancement is None
    assert "overview_original_on_open" not in model._widgets


def test_actions_container_holds_the_standard_module_actions(mask, qtbot, source):
    tab, _view = _open(mask, qtbot, source)
    texts = [button.text() for button in tab._actions_row.findChildren(QPushButton)]
    for wanted in ("Run", "Stop", "Import settings…", "Export settings…", "Export…",
                   "Export image-level measurements"):
        assert wanted in texts
    assert tab._actions_section.isVisible()


def test_measurement_settings_reach_the_export(mask, qtbot, source, monkeypatch):
    tab, view = _open(mask, qtbot, source)
    seen = {}

    def export(fields, source, channels, workers=1, cancelled=None, annotations=None, progress=None, **kwargs):
        seen.update(workers=workers, measurements=kwargs.get("measurements"), progress=progress)
        progress(1, len(fields))
        return {"cancelled": False, "errors": [], "count": len(fields), "path": "db"}

    monkeypatch.setattr(browser, "_overview_measurements_export", export)
    tab._settings_model.set_value_for_key("overview_measure_texture", False)
    tab._btn_overview_measure.click()
    qtbot.waitUntil(lambda: "measurements" in seen and not view._measurement_jobs.is_busy(), timeout=15000)
    assert seen["workers"] is None
    assert seen["measurements"]["overview_measure_texture"] is False
    assert "overview_measure_n_jobs" in seen["measurements"]


def test_fields_from_two_wells_open_as_stacked_images_and_close_one_at_a_time(mask, qtbot, source):
    _tab, view = _open(mask, qtbot, source)
    chosen = [field for field in view._fields if field.well in ("A01", "B01")][:3]
    assert {field.well for field in chosen} == {"A01", "B01"}
    view._selected_fields = {field.identity: field for field in chosen}
    view._open_selected_images()
    panels = list(view._image_views)
    assert [panel.field for panel in panels] == chosen
    assert all(view._image_stack.indexOf(panel) >= 0 for panel in panels)
    assert panels[0].title.text() == f"Well {chosen[0].well}, field {chosen[0].field}"
    qtbot.waitUntil(lambda: all(panel.view._item is not None for panel in panels), timeout=15000)
    view._set_tile_size(300)
    assert all(panel.view.minimumHeight() == 450 for panel in panels)
    assert not hasattr(view, "all_metadata_toggle") and not hasattr(view, "_image_stack_bar")
    panels[0].metadata_toggle.setChecked(True)
    assert panels[0].metadata_label.isVisibleTo(panels[0]) and not panels[2].metadata_toggle.isChecked()
    panels[1].close_button.click()
    assert view._image_views == [panels[0], panels[2]]
    assert view._image_stack.indexOf(panels[0]) >= 0 and view._image_stack.indexOf(panels[2]) >= 0
    view._open_image_views([field for field in view._fields] * 2)
    assert len(view._image_views) == browser._OVERVIEW_MAX_IMAGE_VIEWS


def test_three_layered_groups_reach_the_comparison_with_their_counts(mask, qtbot, source, monkeypatch, tmp_path):
    _tab, view = _open(mask, qtbot, source)
    view._annotations = {("p1", "A01"): {"cell": "HeLa", "pathogen": "WT", "treatment": "none"},
                         ("p1", "A02"): {"cell": "HeLa", "pathogen": "Ku80", "treatment": "none"},
                         ("p1", "B01"): {"cell": "HFF", "pathogen": "WT", "treatment": "drug"}}
    view._refresh_comparison_choices()
    builder = view.group_builder
    for definition in ({"cell": "HeLa", "pathogen": "WT"}, {"cell": "HeLa", "pathogen": "Ku80"},
                       {"pathogen": "WT", "treatment": "drug"}):
        for column, combo in builder._pickers.items():
            combo.setCurrentIndex(combo.findData(definition.get(column)))
        assert builder.add_group()
    assert not builder.add_group({"cell": "HeLa", "pathogen": "WT"})
    assert [builder.groups.item(index).text() for index in range(3)] == [
        "HeLa + WT — 2 images", "HeLa + Ku80 — 2 images", "WT + drug — 2 images"]
    builder.groups.setCurrentRow(2)
    builder.move_group(-1)
    assert builder.definitions()[1] == {"pathogen": "WT", "treatment": "drug"}
    builder.move_group(1)
    rows = []
    for field in view._fields:
        metadata = view._annotations[(field.plate, field.well)]
        rows.append(dict(plate=field.plate, well=field.well, field=field.field,
                         intensity=float(len(rows)), **metadata))
    frame = pd.DataFrame(rows)
    monkeypatch.setattr(browser, "_overview_comparison_frame", lambda _source: frame)
    view._metadata_source = str(tmp_path)
    view.comparison_measurement.setEditText("intensity")
    view._compare_groups()
    folder = tmp_path / "results" / "overview_comparisons"
    qtbot.waitUntil(lambda: any(folder.glob("*_summary.csv")) and not view._export_jobs.is_busy(), timeout=30000)
    summary = pd.read_csv(next(folder.glob("*_summary.csv")))
    assert list(summary["group"]) == ["HeLa + WT", "HeLa + Ku80", "WT + drug"]
    assert list(summary["images"]) == [2, 2, 2]


def test_presets_build_all_combinations_or_vary_one_column(mask, qtbot, source):
    assert browser._overview_all_combinations({"cell": ["HeLa", "HFF"], "pathogen": ["WT", "Ku80"]}) == [
        {"cell": "HeLa", "pathogen": "WT"}, {"cell": "HeLa", "pathogen": "Ku80"},
        {"cell": "HFF", "pathogen": "WT"}, {"cell": "HFF", "pathogen": "Ku80"}]
    assert browser._overview_vary_one("pathogen", ["WT", "Ku80"], {"cell": "HFF", "pathogen": "x"}) == [
        {"cell": "HFF", "pathogen": "WT"}, {"cell": "HFF", "pathogen": "Ku80"}]
    _tab, view = _open(mask, qtbot, source)
    view._annotations = {("p1", "A01"): {"cell": "HeLa", "pathogen": "WT"},
                         ("p1", "A02"): {"cell": "HeLa", "pathogen": "Ku80"},
                         ("p1", "B01"): {"cell": "HFF", "pathogen": "WT"}}
    view._refresh_comparison_choices()
    builder = view.group_builder
    builder.preset.setCurrentIndex(builder.preset.findData("vary"))
    builder.vary.setCurrentIndex(builder.vary.findData("pathogen"))
    builder._pickers["cell"].setCurrentIndex(builder._pickers["cell"].findData("HeLa"))
    assert builder.apply_preset() == 2
    assert builder.definitions() == [{"cell": "HeLa", "pathogen": "Ku80"}, {"cell": "HeLa", "pathogen": "WT"}]


def _record_enhancements(monkeypatch):
    calls = []
    real = browser._overview_pixels

    def recorded(field, *args, edge=1200, enhancement=None, **kwargs):
        calls.append((edge, enhancement))
        return real(field, *args, edge=edge, enhancement=enhancement, **kwargs)

    monkeypatch.setattr(browser, "_overview_pixels", recorded)
    return calls


def _open_original(view, qtbot):
    _open_well(view, qtbot, "A01")
    view.fields.setCurrentRow(0)
    view._field_in(view.fields, view.fields.item(0))
    view.resolution_slider.setValue(view.resolution_slider.maximum())


def test_psf_deconvolution_applies_to_original_images_only(mask, qtbot, source, monkeypatch):
    tab, view = _open(mask, qtbot, source)
    page = tab._overview_page
    enhancement = _sections(tab)["Enhancement"]
    assert enhancement.isAncestorOf(page.psf_controls) and enhancement.isAncestorOf(page.restoration_controls)
    psf = page.psf_controls
    psf.operation.setCurrentIndex(psf.operation.findData("deconvolve"))
    qtbot.waitUntil(lambda: psf._kernel is not None, timeout=20000)
    calls = _record_enhancements(monkeypatch)
    _open_original(view, qtbot)
    qtbot.waitUntil(lambda: any(edge is None and chain is not None and chain.psf is not None
                                for edge, chain in calls) and not view._image_jobs.is_busy(), timeout=30000)
    qtbot.waitUntil(lambda: "PSF deconvolution" in view.status.text(), timeout=30000)
    assert view.status.text() == "Original image shown with PSF deconvolution."
    assert view.view._item.pixmap().width() == 600
    full = [chain for edge, chain in calls if edge is None]
    assert any(chain is not None and chain.psf_operation == "deconvolve" for chain in full)
    assert all(chain is None or chain.psf_operation in ("deconvolve", "none") for chain in full)
    view.preview_resolution.setCurrentIndex(view.preview_resolution.findData(1024))
    qtbot.waitUntil(lambda: not view._quality_timer.isActive(), timeout=5000)
    calls.clear()
    view._set_tile_size(200)
    qtbot.waitUntil(lambda: any(edge is not None for edge, _chain in calls), timeout=20000)
    previews = [chain for edge, chain in calls if edge is not None]
    assert all(chain is None or chain.psf_operation == "none" for chain in previews)


def test_missing_restoration_model_refuses_cleanly_and_still_shows_the_original(mask, qtbot, source, monkeypatch):
    from spacr.qt.widgets import restoration_controls

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("Cellpose 3 is not installed")

    monkeypatch.setattr(restoration_controls, "_restoration_plan", unavailable)
    tab, view = _open(mask, qtbot, source)
    controls = tab._overview_page.restoration_controls
    controls.operation.setCurrentIndex(controls.operation.findData("denoise"))
    qtbot.waitUntil(lambda: "Restoration unavailable" in controls.status.text(), timeout=20000)
    assert "Cellpose 3 is not installed" in controls.status.text()
    _open_original(view, qtbot)
    qtbot.waitUntil(lambda: view.status.text().startswith("Original image shown without Cellpose 3 restoration"),
                    timeout=30000)
    assert "Cellpose 3 is not installed" in view.status.text()
    qtbot.waitUntil(lambda: view.view._item is not None and view.view._item.pixmap().width() == 600, timeout=20000)
