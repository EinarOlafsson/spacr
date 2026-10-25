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
