"""A replaced or half-deleted Overview browser never touches a deleted control (GitHub #165, #166).

#165: a display refresh read ``time_axis`` after the combo box, mounted in
one of the Overview tab's setting rows, had been deleted with its row.
#166: a setting row edited after its browser was replaced called the old
bridge, which wrote into the old browser's deleted line edit.
"""
import gc

import numpy as np
import pytest
import tifffile
from PySide6.QtCore import QCoreApplication, QEvent
from shiboken6 import isValid

from spacr.qt.widgets import qc_field_browser as browser


PATTERN = r"(?P<plateID>p\d+)-(?P<wellID>[A-Z]\d+)-f(?P<fieldID>\d+)-c(?P<chanID>\d+)\.tif"


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    for index, well in enumerate(("A01", "A02")):
        tifffile.imwrite(folder / f"p1-{well}-f1-c1.tif",
                         (np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) + index))
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


def _browse(mask, qtbot, source):
    mask._settings_model.set_value_for_key("src", str(source))
    field = mask._settings_model._widgets.get("src")
    next(action for action in field.actions()
         if action.objectName() == "GenerateMasksFieldOverview").trigger()
    tab = mask._overview_tab
    wizard = tab._overview_page.wizard
    wizard.preset.setCurrentIndex(wizard.preset.count() - 1)
    wizard.regex.setText(PATTERN)
    wizard._scan()
    qtbot.waitUntil(lambda: wizard.next.isEnabled(), timeout=15000)
    wizard._next()
    page = tab._overview_page
    qtbot.waitUntil(lambda: page.browser is not None and len(page.browser._fields) == 2
                    and page.browser._normalization_ready, timeout=15000)
    return tab, page.browser


def _flush_deletes():
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    gc.collect()


def test_a_refresh_after_a_mounted_combo_is_deleted_reads_no_deleted_control(mask, qtbot, source, caplog):
    import logging
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QDoubleSpinBox

    caplog.set_level(logging.ERROR, logger="spacr")
    tab, view = _browse(mask, qtbot, source)
    originals = {path: path.read_bytes() for path in source.glob("*.tif")}
    identities = tuple(field.identity for field in view._fields)
    view._open_image_views([view._fields[0]])
    panel, = view._image_views
    qtbot.waitUntil(lambda: panel.view._item is not None and not view._image_stack_jobs.is_busy(),
                    timeout=15000)
    pixels = panel.view._item.pixmap().toImage().copy()
    section = next(section for section in tab.rendered_settings_sections()
                   if section.property("settingsCategorySource") == "Enhancement")
    section.set_expanded(True)
    gamma = tab._settings_model._widgets["enhance_gamma"]
    assert isinstance(gamma, QDoubleSpinBox)
    tab._settings_scroll.ensureWidgetVisible(gamma)
    assert gamma.isVisibleTo(tab)
    gamma.setFocus()
    qtbot.keyClick(gamma, Qt.Key_A, modifier=Qt.ControlModifier)
    qtbot.keyClicks(gamma, "0.75")
    qtbot.keyClick(gamma, Qt.Key_Tab)
    assert gamma.value() == .75
    page = tab._overview_page
    assert "enhance_gamma" in page._bridge.pending()
    assert view._enhancement is None
    apply = page._apply_bar.apply_button
    assert apply.isVisibleTo(tab) and apply.isEnabled()
    combo = view.time_axis
    combo.deleteLater()
    _flush_deletes()
    assert not isValid(combo)
    assert view._controls_lost
    assert isValid(apply) and apply.isVisibleTo(tab) and apply.isEnabled()
    qtbot.mouseClick(apply, Qt.LeftButton)
    assert page._bridge.pending() == ()
    assert view._enhancement.gamma == .75
    assert panel.view._item.pixmap().toImage() == pixels
    assert tuple(field.identity for field in view._fields) == identities
    assert view._metadata_source == str(source)
    assert all(path.read_bytes() == contents for path, contents in originals.items())
    view.normalise.toggle()
    view._refresh_image()
    view._reload_image_views()
    view._request_visuals()
    assert not view._visual_timer.isActive()
    fresh_tab, fresh = _browse(mask, qtbot, source)
    assert fresh_tab is tab and fresh is not view and not fresh._controls_lost
    assert isValid(fresh.time_axis) and fresh._metadata_source == str(source)
    assert tuple(field.identity for field in fresh._fields) == identities
    assert fresh._enhancement.gamma == .75
    fresh._open_image_views([fresh._fields[0]])
    fresh_panel, = fresh._image_views
    qtbot.waitUntil(lambda: fresh_panel.view._item is not None and not fresh._image_stack_jobs.is_busy(),
                    timeout=15000)
    assert fresh_panel.field.files == panel.field.files
    assert all(path.read_bytes() == contents for path, contents in originals.items())
    assert not [record for record in caplog.records
                if record.name.startswith("spacr") and record.levelno >= logging.ERROR]


def test_a_setting_edited_after_its_browser_was_replaced_reaches_no_deleted_control(mask, qtbot, source):
    tab, old = _browse(mask, qtbot, source)
    page = tab._overview_page
    old_bridge = page._bridge
    old_source, old_axis = old.source, old.time_axis
    page._retire_browser()
    _flush_deletes()
    assert not isValid(old_source) and not isValid(old_axis)
    assert old_bridge._detached and old_bridge._connections == []
    model = tab._settings_model
    model.set_value_for_key("src", str(source / "elsewhere"))
    model.set_value_for_key("overview_channels", "1")
    model.set_value_for_key("overview_tile_size", 96)
    model.set_value_for_key("src", str(source))
    page.run(model.collect())
    wizard = page.wizard
    wizard.preset.setCurrentIndex(wizard.preset.count() - 1)
    wizard.regex.setText(PATTERN)
    wizard._scan()
    qtbot.waitUntil(lambda: wizard.next.isEnabled(), timeout=15000)
    wizard._next()
    qtbot.waitUntil(lambda: page.browser is not None and page.browser._normalization_ready, timeout=15000)
    fresh = page.browser
    model.set_value_for_key("overview_tile_size", 120)
    assert fresh._tile_size == 120
    assert fresh.time_axis.parentWidget() is not None and not fresh._controls_lost


def test_a_bridge_detaches_itself_when_its_browser_is_destroyed(mask, qtbot, source):
    tab, view = _browse(mask, qtbot, source)
    bridge = tab._overview_page._bridge
    control = view.source
    control.deleteLater()
    _flush_deletes()
    tab._settings_model.set_value_for_key("src", str(source / "other"))
    assert bridge._detached and bridge._connections == []
