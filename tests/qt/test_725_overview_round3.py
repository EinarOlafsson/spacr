"""Round-3 Overview: fast persistent plate thumbnails, live Image settings and stacked comparison.

Item 725, maintainer feedback 2026-10-24 round 3, points 1-8 and the UI side of 10.
"""
import numpy as np
import pandas as pd
import pytest
import tifffile

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox

from spacr.qt.widgets import qc_field_browser as browser
from spacr.qt.widgets.toggle import Toggle


PATTERN = r"(?P<plateID>p\d+)-(?P<wellID>[A-Z]\d+)-f(?P<fieldID>\d+)-c(?P<chanID>\d+)\.tif"
WELLS = ("A01", "A02", "B01", "B03")


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    for index, well in enumerate(WELLS):
        for field in (1, 2):
            for channel in (1, 2):
                pixels = (np.arange(60 * 80, dtype=np.uint16).reshape(60, 80) * (channel + 1)
                          + 100 * index + field)
                tifffile.imwrite(folder / f"p1-{well}-f{field}-c{channel}.tif", pixels)
    return folder


def _retire(view, qtbot):
    view._cancel()
    runners = (view._index_jobs, view._image_jobs, view._thumbnail_jobs, view._field_thumbnail_jobs,
               view._montage_jobs, view._export_jobs, view._metadata_jobs, view._measurement_jobs,
               view._native_jobs, view._image_stack_jobs)
    qtbot.waitUntil(lambda: all(runner.active_jobs() == 0 for runner in runners), timeout=15000)
    for runner in runners:
        runner.shutdown(timeout_ms=3000)
    view.close()


@pytest.fixture
def view(qtbot, source):
    browser._OVERVIEW_PLANE_CACHE.clear()
    browser._OVERVIEW_INFO_CACHE.clear()
    dialog = browser._FieldOverview(settings={
        "source": str(source), "source_scope": "folder", "custom_regex": PATTERN,
        "channels": (1, 2), "geometry": (2, 3), "representatives": 1})
    qtbot.addWidget(dialog, before_close_func=lambda widget: _retire(widget, qtbot))
    dialog.resize(1300, 950)
    dialog.show()
    dialog._scan()
    qtbot.waitUntil(lambda: bool(dialog._fields) and dialog._normalization_ready
                    and not dialog._index_jobs.is_busy(), timeout=15000)
    dialog.fov_axis.setCurrentIndex(dialog.fov_axis.findData(None))
    return dialog


def _icons(view):
    return {well for well in WELLS
            if not view.wells.item(*view._well_positions[well]).icon().isNull()}


def _item(view, well):
    return view.wells.item(*view._well_positions[well])


def test_plate_keeps_one_image_per_well_while_a_well_and_an_image_are_open(view, qtbot):
    qtbot.waitUntil(lambda: _icons(view) == set(WELLS), timeout=20000)
    view._well(_item(view, "A01"))
    qtbot.waitUntil(lambda: view.fields.count() > 0, timeout=15000)
    view.fields.setCurrentRow(0)
    view._field_in(view.fields, view.fields.item(0))
    qtbot.waitUntil(lambda: view.view._item is not None, timeout=15000)
    qtbot.waitUntil(lambda: _icons(view) == set(WELLS), timeout=20000)
    assert view._scope[1] == "A01" and view._individual


def test_thumbnails_decode_in_parallel_batches_and_display_changes_reuse_decoded_planes(view, qtbot, monkeypatch):
    qtbot.waitUntil(lambda: _icons(view) == set(WELLS), timeout=20000)
    assert browser._OVERVIEW_THUMBNAIL_WORKERS >= 1 and browser._OVERVIEW_THUMBNAIL_BATCH >= 1
    reads = []
    real = browser._overview_raw_planes

    def counted(field, channels, **kwargs):
        reads.append((field.well, tuple(channels)))
        yield from real(field, channels, **kwargs)

    monkeypatch.setattr(browser, "_overview_raw_planes", counted)
    before = {well: _item(view, well).data(Qt.UserRole + 1) for well in WELLS}
    view.high.setValue(view.high.value() - 1)
    qtbot.waitUntil(lambda: all(_item(view, well).data(Qt.UserRole + 1) != before[well]
                                and not _item(view, well).icon().isNull() for well in WELLS), timeout=20000)
    assert reads == []


def test_the_first_thumbnail_request_is_a_single_well(view, qtbot, monkeypatch):
    qtbot.waitUntil(lambda: _icons(view) == set(WELLS) and not view._thumbnail_jobs.is_busy()
                    and not view._thumbnail_jobs.active_jobs(), timeout=20000)
    batches = []
    real = browser._FieldOverview._submit_plate_thumbnails

    def recorded(self, requests, *args):
        batches.append(len(requests))
        return real(self, requests, *args)

    monkeypatch.setattr(browser._FieldOverview, "_submit_plate_thumbnails", recorded)
    for well in WELLS:
        _item(view, well).setIcon(browser.QPixmap())
        _item(view, well).setData(Qt.UserRole + 1, None)
    view._installed_thumbnails.clear()
    view._thumbnail_cache.clear()
    view._update_visuals()
    assert batches and batches[0] == 1
    assert all(size <= browser._OVERVIEW_THUMBNAIL_BATCH for size in batches)
    qtbot.waitUntil(lambda: _icons(view) == set(WELLS), timeout=20000)


def test_channel_targets_limit_the_percentile_stretch_and_the_enhancement(view):
    from spacr.qt.detect_chain import NO_CHAIN

    field = view._fields[0]
    channels = ((0, (1, 99)))
    every = browser._overview_pixels(field, (1, 2), (20., 80.), edge=64, channel_views=True)
    only_first = browser._overview_pixels(field, (1, 2), (20., 80.), edge=64, channel_views=True,
                                          percentile_channels=(1,))
    assert np.array_equal(every[0][1], only_first[0][1])
    assert not np.array_equal(every[1][1], only_first[1][1])
    chain = NO_CHAIN._replace(gamma=0.4)
    enhanced = browser._overview_pixels(field, (1, 2), (1., 99.), edge=64, channel_views=True,
                                        enhancement=chain, enhancement_channels=(2,))
    plain = browser._overview_pixels(field, (1, 2), (1., 99.), edge=64, channel_views=True)
    assert np.array_equal(enhanced[0][1], plain[0][1])
    assert enhanced[1][1].mean() > plain[1][1].mean()
    assert browser._overview_channel_targets("all") is None
    assert browser._overview_channel_targets("1, 3") == (1, 3)
    view._set_channel_targets((2,), (1,))
    options = view._display_kwargs(view=True)
    assert options["enhancement_channels"] == (2,) and options["percentile_channels"] == (1,)
    assert "percentile_channels" not in view._display_kwargs()
    assert channels


def test_two_open_wells_stack_one_image_from_each(view, qtbot):
    view._well(_item(view, "A01"))
    view._well(_item(view, "B03"))
    qtbot.waitUntil(lambda: len(view._image_views) == 2, timeout=15000)
    assert {panel.field.well for panel in view._image_views} == {"A01", "B03"}
    qtbot.waitUntil(lambda: all(panel.view._item is not None for panel in view._image_views), timeout=15000)
    first, second = view._image_views
    first.view.scale(2, 2)
    first.view.horizontalScrollBar().setValue(first.view.horizontalScrollBar().maximum())
    view._sync_image_views(first.view)
    assert second.view.transform() == first.view.transform()
    assert second.view.horizontalScrollBar().value() == first.view.horizontalScrollBar().value()


def test_the_image_view_has_the_image_quality_slider(view):
    header = view._image_panel.header_layout
    slider = view.resolution_slider
    assert header.indexOf(slider) >= 0 and header.indexOf(slider) < header.indexOf(view.original_toggle)
    slider.setValue(0)
    assert view.preview_resolution.currentData() == 128
    view.preview_resolution.setCurrentIndex(view.preview_resolution.count() - 1)
    assert slider.value() == view.preview_resolution.count() - 1


def test_every_overview_boolean_is_the_spacr_toggle(view, tmp_path):
    plain = [box for box in view.findChildren(QCheckBox) if not isinstance(box, Toggle)]
    assert plain == []
    wizard = browser._OverviewMetadataWizard(str(tmp_path), {})
    wizard._startup_timer.stop()
    try:
        assert [box for box in wizard.findChildren(QCheckBox) if not isinstance(box, Toggle)] == []
    finally:
        wizard.reject()
        wizard.deleteLater()


def test_a_group_comparison_stacks_representative_images_beside_its_plot(view, qtbot, monkeypatch, tmp_path):
    view._annotations = {("p1", well): {"cell": "HeLa" if well.startswith("A") else "HFF"} for well in WELLS}
    view._refresh_comparison_choices()
    rows = [dict(plate=field.plate, well=field.well, field=field.field, channel_1_mean=float(index),
                 channel_1_focus=float(index % 3), **view._annotations[(field.plate, field.well)])
            for index, field in enumerate(view._fields)]
    monkeypatch.setattr(browser, "_overview_comparison_frame", lambda _source: pd.DataFrame(rows))
    view._metadata_source = str(tmp_path)
    for item in view._comparison_items():
        item.setCheckState(0, Qt.Checked)
    view.comparison_measurement.setEditText("channel_1_mean")
    view._compare_groups()
    qtbot.waitUntil(lambda: view._comparison_signature is not None and view._comparison_signature[0] == "groups"
                    and len(view._image_views) == 2 and not view._export_jobs.is_busy(), timeout=60000)
    assert view.view._item is not None
    assert {panel.field.well[0] for panel in view._image_views} == {"A", "B"}
    assert view.status.text().startswith("Comparison of channel_1_mean")
    table = view.comparison_table
    assert table.isVisible() and table.rowCount() == 1
    headers = [table.horizontalHeaderItem(index).text() for index in range(table.columnCount())]
    assert {"group_a", "group_b", "p_adjusted"} <= set(headers)


def test_image_settings_are_one_main_category_with_four_subcategories(qtbot, source):
    from spacr.qt.screens.settings_model import (_APP_CATEGORY_PARENTS, categories_for_app,
                                                 get_categories)

    layout = categories_for_app("overview", get_categories())
    parents = _APP_CATEGORY_PARENTS["overview"]
    for title in ("Channels", "Display", "Normalization", "Enhancement"):
        assert parents[title] == "Image"
    assert "overview_channels" in layout["Channels"]
    assert "overview_channels" not in layout["Source"]
    assert {"overview_normalize", "overview_normalize_channels", "overview_percentile_high"} <= set(layout["Normalization"])
    assert "overview_enhance_channels" in layout["Enhancement"]
    assert {"overview_tile_size", "overview_tile_spacing"} <= set(layout["Display"])
