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


def test_a_refresh_after_a_mounted_combo_is_deleted_reads_no_deleted_control(mask, qtbot, source):
    _tab, view = _browse(mask, qtbot, source)
    view._open_image_views([view._fields[0]])
    combo = view.time_axis
    combo.deleteLater()
    _flush_deletes()
    assert not isValid(combo)
    assert view._controls_lost
    view.normalise.toggle()
    view._refresh_image()
    view._reload_image_views()
    view._request_visuals()
    assert not view._visual_timer.isActive()


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
