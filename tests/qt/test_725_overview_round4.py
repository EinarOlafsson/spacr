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


@pytest.mark.parametrize("key", ["overview_normalize_channels", "overview_enhance_channels"])
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
    assert not body.is_collapsed("Settings")
    assert body.sizes()[0] >= min(browser._OVERVIEW_SETTINGS_WIDTH, sum(body.sizes()) // 2)
    tab, _view = _browse(mask, qtbot, source)
    qtbot.wait(50)
    assert not body.is_collapsed("Settings")
    assert body.sizes()[0] >= min(browser._OVERVIEW_SETTINGS_WIDTH, sum(body.sizes()) // 2)
