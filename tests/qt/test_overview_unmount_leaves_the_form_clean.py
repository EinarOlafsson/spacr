"""Unmounting the Overview's controls warns nothing and leaves no row behind.

Retiring the browser moved each mounted control back under it before removing
its form row; the move already took a single control out of the form, so
``removeRow(widget)`` printed ``QFormLayout::takeRow: Invalid widget`` once per
such row and left the row's label in the form. The cycle is driven through a
finished load, a load still running, and a browser replaced by a new run,
keeping the GitHub #165/#166 guarantees: the browser's controls die with it,
and only with it.
"""
import gc

import numpy as np
import pytest
import tifffile
from PySide6.QtCore import QCoreApplication, QEvent, qInstallMessageHandler
from shiboken6 import isValid

from tests.qt.test_overview_deleted_controls_165_166 import PATTERN, _close


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    for index, well in enumerate(("A01", "A02")):
        tifffile.imwrite(folder / f"p1-{well}-f1-c1.tif",
                         (np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) + index))
    return folder


@pytest.fixture
def mask(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("mask")
    qtbot.addWidget(screen, before_close_func=lambda widget: _close(widget, qtbot))
    screen.resize(1400, 1000)
    screen.show()
    return screen


@pytest.fixture
def form_warnings():
    seen = []
    previous = qInstallMessageHandler(lambda _kind, _context, text: seen.append(text))
    try:
        yield seen
    finally:
        qInstallMessageHandler(previous)
    assert not [text for text in seen if "QFormLayout" in text]


def _open(mask, qtbot, source, wait=True):
    mask._settings_model.set_value_for_key("src", str(source))
    field = mask._settings_model._widgets.get("src")
    next(action for action in field.actions()
         if action.objectName() == "GenerateMasksFieldOverview").trigger()
    page = mask._overview_tab._overview_page
    wizard = page.wizard
    wizard.preset.setCurrentIndex(wizard.preset.count() - 1)
    wizard.regex.setText(PATTERN)
    wizard._scan()
    qtbot.waitUntil(lambda: wizard.next.isEnabled(), timeout=15000)
    wizard._next()
    assert page.browser is not None and page._rows
    if wait:
        qtbot.waitUntil(lambda: len(page.browser._fields) == 2 and page.browser._normalization_ready,
                        timeout=15000)
    return page


def _sections(page):
    sections, mounted = {}, {}
    for section, _field, _widgets in page._rows:
        sections[id(section)] = section
        mounted[id(section)] = mounted.get(id(section), 0) + 1
    return sections, mounted


def _idle(view, qtbot):
    view._cancel()
    qtbot.waitUntil(lambda: all(runner.active_jobs() == 0 for runner in (
        view._index_jobs, view._image_jobs, view._thumbnail_jobs, view._field_thumbnail_jobs,
        view._montage_jobs, view._export_jobs, view._metadata_jobs, view._measurement_jobs,
        view._native_jobs, view._image_stack_jobs)), timeout=15000)


def _gone(view, controls):
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    gc.collect()
    assert not isValid(view)
    assert not any(isValid(widget) for widget in controls)


def _unmount(page, qtbot):
    view = page.browser
    sections, mounted = _sections(page)
    before = {key: section._form.rowCount() for key, section in sections.items()}
    controls = [widget for _section, _field, widgets in page._rows for widget in widgets]
    _idle(view, qtbot)
    page._retire_browser()
    assert page._rows == []
    for key, section in sections.items():
        assert section._form.rowCount() == before[key] - mounted[key]
    assert all(isValid(widget) and widget.parentWidget() is view for widget in controls)
    _gone(view, controls)
    return {key: section._form.rowCount() for key, section in sections.items()}


def test_retiring_the_browser_warns_nothing_and_removes_every_mounted_row(
        mask, qtbot, source, form_warnings):
    page = _open(mask, qtbot, source)
    resting = _unmount(page, qtbot)
    for wait in (False, True, False):
        page = _open(mask, qtbot, source, wait=wait)
        assert _unmount(page, qtbot) == resting


def test_a_browser_replaced_by_a_new_run_warns_nothing_and_leaves_one_set_of_rows(
        mask, qtbot, source, form_warnings):
    page = _open(mask, qtbot, source)
    resting = _unmount(page, qtbot)
    page = _open(mask, qtbot, source, wait=False)
    old = page.browser
    controls = [widget for _section, _field, widgets in page._rows for widget in widgets]
    _idle(old, qtbot)
    page = _open(mask, qtbot, source)
    assert page.browser is not old
    _gone(old, controls)
    mounted = _sections(page)[1]
    assert _unmount(page, qtbot) == resting
    assert mounted == _sections(_open(mask, qtbot, source))[1]
