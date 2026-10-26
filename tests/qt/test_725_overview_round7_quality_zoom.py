"""Round-7 Overview: one image-quality slider for every view, and zoom on the hovered image only.

Item 725, maintainer feedback 2026-10-26: "origional and the quality slider should be for the
plate as well and for the image view it only works when an event occurs. and a new image is
loaded, these settings should periodically. only the image the mouse is hovering should be
zoomed, currently if i zoom one picture they all zoom." and "you can remove the origional
button the slider max is the same thing, and you can remove the metadata button for the
overarching image view as each image gets one anyway".
"""
import numpy as np
import pytest
import tifffile

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QNativeGestureEvent, QPointingDevice, QWheelEvent
from PySide6.QtWidgets import QApplication

from spacr.qt.widgets import qc_field_browser as browser


PATTERN = r"(?P<plateID>p\d+)-(?P<wellID>[A-Z]\d+)-f(?P<fieldID>\d+)-c(?P<chanID>\d+)\.tif"
WELLS = ("A01", "A02", "B01", "B03")
HEIGHT, WIDTH = 300, 400


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    for index, well in enumerate(WELLS):
        for field in (1, 2):
            for channel in (1, 2):
                pixels = (np.arange(HEIGHT * WIDTH, dtype=np.uint16).reshape(HEIGHT, WIDTH) * (channel + 1)
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


def _item(view, well):
    return view.wells.item(*view._well_positions[(view.plates.currentData(), well)])


def _edge(pixmap):
    return max(pixmap.width(), pixmap.height())


def _set_quality(view, qtbot, value):
    view.preview_resolution.setCurrentIndex(view.preview_resolution.findData(value))
    qtbot.waitUntil(lambda: not view._quality_timer.isActive(), timeout=5000)


def _stacked(view, qtbot):
    view._well(_item(view, "A01"))
    view._well(_item(view, "B03"))
    qtbot.waitUntil(lambda: len(view._image_views) == 2, timeout=15000)
    qtbot.waitUntil(lambda: all(panel.view._item is not None for panel in view._image_views), timeout=15000)
    return view._image_views


def _wheel(view, up=True):
    viewport = view.viewport()
    local = QPointF(viewport.rect().center())
    event = QWheelEvent(local, QPointF(viewport.mapToGlobal(local.toPoint())), QPoint(0, 0),
                        QPoint(0, 120 if up else -120), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    QApplication.sendEvent(viewport, event)


def test_the_original_buttons_and_the_all_images_bar_are_gone(view):
    assert not hasattr(view, "original_toggle") and not hasattr(view, "all_original_toggle")
    assert not hasattr(view, "all_metadata_toggle") and not hasattr(view, "_image_stack_bar")
    assert not hasattr(browser, "_OVERVIEW_ORIGINAL_TOOLTIP")
    assert view.preview_resolution.itemData(view.preview_resolution.count() - 1) == 0
    assert view.preview_resolution.itemText(view.preview_resolution.count() - 1) == "Original"
    for slider in (view.resolution_slider, view.plate_resolution_slider):
        assert slider.maximum() == view.preview_resolution.count() - 1
    assert view._plate_panel.header_layout.indexOf(view.plate_resolution_slider) >= 0
    view.plate_resolution_slider.setValue(view.plate_resolution_slider.maximum())
    assert view.resolution_slider.value() == view.resolution_slider.maximum()
    assert view._quality_original()


def test_quality_changes_redraw_the_open_plate_well_and_image_views(view, qtbot):
    _set_quality(view, qtbot, 128)
    first, second = _stacked(view, qtbot)
    assert _edge(first.view._item.pixmap()) <= 128
    view.fields.setCurrentRow(0)
    view._field_in(view.fields, view.fields.item(0))
    qtbot.waitUntil(lambda: view.view._item is not None and _edge(view.view._item.pixmap()) <= 128, timeout=15000)
    tiles = [fields for fields in view._field_lists() if fields.isVisible() and fields.count()]
    assert tiles
    qtbot.waitUntil(lambda: all((fields.item(0).data(Qt.UserRole + 1) or (0,))[-1] is not None
                                for fields in tiles), timeout=15000)
    qtbot.waitUntil(lambda: (_item(view, "A02").data(Qt.UserRole + 1) or (0,) * 5)[4] is not None, timeout=20000)
    first.view.scale(2, 2)
    first.view._user_zoomed = True
    generations = [panel.generation for panel in (first, second)]
    view.resolution_slider.setValue(view.resolution_slider.maximum())
    assert view._quality_timer.isActive()
    assert [panel.generation for panel in (first, second)] == generations
    qtbot.waitUntil(lambda: all(_edge(panel.view._item.pixmap()) == WIDTH for panel in (first, second)),
                    timeout=20000)
    assert [panel.generation for panel in (first, second)] == [value + 1 for value in generations]
    assert first.view._user_zoomed and not second.view._user_zoomed
    qtbot.waitUntil(lambda: view.view._item is not None and _edge(view.view._item.pixmap()) == WIDTH,
                    timeout=20000)
    qtbot.waitUntil(lambda: all((fields.item(0).data(Qt.UserRole + 1) or (0,))[-1] is None
                                for fields in tiles), timeout=20000)
    qtbot.waitUntil(lambda: (_item(view, "A02").data(Qt.UserRole + 1) or (0,) * 5)[4] is None, timeout=20000)


def test_dragging_the_quality_slider_decodes_only_where_it_rests(view, qtbot):
    _set_quality(view, qtbot, 128)
    first, second = _stacked(view, qtbot)
    generations = [panel.generation for panel in (first, second)]
    for value in (256, 512, 1024, 2048):
        view.resolution_slider.setValue(view.preview_resolution.findData(value))
    qtbot.waitUntil(lambda: not view._quality_timer.isActive(), timeout=5000)
    assert [panel.generation for panel in (first, second)] == [value + 1 for value in generations]
    qtbot.waitUntil(lambda: all(_edge(panel.view._item.pixmap()) == WIDTH for panel in (first, second)),
                    timeout=20000)


def test_zooming_one_stacked_image_leaves_the_other_alone(view, qtbot):
    first, second = _stacked(view, qtbot)
    before = second.view.transform()
    scale = first.view.transform().m11()
    _wheel(first.view)
    assert first.view.transform().m11() > scale
    assert second.view.transform() == before and not second.view._user_zoomed
    first.view.horizontalScrollBar().setValue(first.view.horizontalScrollBar().maximum())
    assert second.view.transform() == before


def test_a_pinch_zooms_only_the_image_under_the_pointer(view, qtbot):
    first, second = _stacked(view, qtbot)
    before, scale = second.view.transform(), first.view.transform().m11()
    viewport = first.view.viewport()
    local = QPointF(viewport.rect().center())
    pinch = QNativeGestureEvent(Qt.ZoomNativeGesture, QPointingDevice.primaryPointingDevice(), 2, local, local,
                                QPointF(viewport.mapToGlobal(local.toPoint())), 0.25, QPointF(0, 0))
    QApplication.sendEvent(viewport, pinch)
    assert first.view.transform().m11() == pytest.approx(scale * 1.25)
    assert first.view._user_zoomed
    assert second.view.transform() == before


def test_link_zoom_is_an_optional_menu_tick_and_off_by_default(view, qtbot):
    first, second = _stacked(view, qtbot)
    assert view._link_zoom is False
    view._image_menu(first.view, QPoint(5, 5))
    menu = view._image_menu_shown
    link = next(action for action in menu.actions() if action.objectName() == "OverviewImage_link_zoom")
    assert link.isCheckable() and not link.isChecked()
    assert link.text() == "Link zoom" and link.toolTip()
    menu.close()
    _wheel(first.view)
    assert second.view.transform() != first.view.transform()
    link.setChecked(True)
    assert view._link_zoom is True
    assert second.view.transform() == first.view.transform()
    _wheel(second.view)
    assert first.view.transform() == second.view.transform()
    view._image_menu(first.view, QPoint(5, 5))
    again = view._image_menu_shown
    assert next(action for action in again.actions()
                if action.objectName() == "OverviewImage_link_zoom").isChecked()
    again.close()
    view._set_link_zoom(first.view, False)
    _wheel(first.view)
    assert second.view.transform() != first.view.transform()


def test_the_enlarged_image_menu_has_no_link_zoom(view, qtbot):
    view._well(_item(view, "A01"))
    qtbot.waitUntil(lambda: view.fields.count() > 0, timeout=15000)
    view.fields.setCurrentRow(0)
    view._field_in(view.fields, view.fields.item(0))
    qtbot.waitUntil(lambda: view.view._item is not None, timeout=15000)
    assert all(panel.view is not view.view for panel in view._image_views)
    view._image_menu(view.view, QPoint(5, 5))
    menu = view._image_menu_shown
    assert menu is not None
    assert all(action.objectName() != "OverviewImage_link_zoom" for action in menu.actions())
    menu.close()


def test_a_saved_original_switch_becomes_the_top_quality():
    from spacr.qt.screens.app_screen import _translate_legacy_setting_keys
    from spacr.settings import OVERVIEW_DEFAULTS, get_overview_default_settings
    from spacr.validate import _check_retired_keys

    assert "overview_original_on_open" not in OVERVIEW_DEFAULTS
    assert get_overview_default_settings({"overview_original_on_open": True})["overview_preview_resolution"] == 0
    kept = get_overview_default_settings({"overview_original_on_open": False, "overview_preview_resolution": 512})
    assert kept["overview_preview_resolution"] == 512 and "overview_original_on_open" not in kept
    assert _translate_legacy_setting_keys({"overview_original_on_open": "true"}) == {
        "overview_preview_resolution": 0}
    problems = _check_retired_keys({"overview_original_on_open": True})
    assert len(problems) == 1 and "overview_preview_resolution" in problems[0].message


@pytest.mark.parametrize("theme", ("light", "dark"))
def test_image_metadata_keeps_contrast_and_viewport_anchor_through_zoom_and_pan(view, qtbot, theme):
    from PySide6.QtGui import QPalette
    from PySide6.QtTest import QTest
    from spacr.qt import preferences
    from spacr.qt.theme import contrast_ratio

    app = QApplication.instance()
    old_theme = preferences.get_theme()
    try:
        preferences.set_theme(theme)
        preferences.apply_preferences_to_app(app)
        panels = _stacked(view, qtbot)
        qtbot.waitUntil(lambda: not view._image_stack_jobs.is_busy(), timeout=15000)
        panel = panels[0]
        assert panel.view._metadata_overlay is panel.metadata_label
        assert view.view._metadata_overlay is view._image_metadata
        source_pixels = panel.view._item.pixmap().toImage().copy()
        toggle = panel.metadata_toggle
        assert toggle.property("spacrContrastBacking") is True
        for selected in (False, True):
            toggle.setChecked(selected)
            foreground = toggle.palette().color(QPalette.WindowText).name()
            background = toggle.palette().color(QPalette.Window).name()
            assert contrast_ratio(foreground, background) >= 4.5
            assert background in toggle.styleSheet()
            assert panel.metadata_label.isVisible() == selected
        assert panel.metadata_label.pos() == QPoint(8, 8)
        for _ in range(3):
            _wheel(panel.view)
        assert panel.view._user_zoomed
        assert panel.metadata_label.pos() == QPoint(8, 8)
        viewport = panel.view.viewport()
        center = viewport.rect().center()
        before = (panel.view.horizontalScrollBar().value(), panel.view.verticalScrollBar().value())
        QTest.mousePress(viewport, Qt.LeftButton, pos=center)
        QTest.mouseMove(viewport, center + QPoint(40, 25), delay=30)
        QTest.mouseRelease(viewport, Qt.LeftButton, pos=center + QPoint(40, 25))
        after = (panel.view.horizontalScrollBar().value(), panel.view.verticalScrollBar().value())
        assert after != before
        assert panel.metadata_label.pos() == QPoint(8, 8)
        assert panel.metadata_label.parentWidget() is viewport
        assert panel.metadata_label.isVisible()
        assert panel.view._item.pixmap().toImage() == source_pixels
        toggle.setChecked(False)
        assert not panel.metadata_label.isVisible()
    finally:
        preferences.set_theme(old_theme)
        preferences.apply_preferences_to_app(app)
