"""Controlled deferred real metadata receivers retain their current UI owner."""

import copy

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton
from shiboken6 import isValid

from tests.qt.test_725_overview_tab_round2 import (
    PATTERN, _open, _open_well, mask as mask, source as source,
)
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile

pytestmark = pytest.mark.qt


@pytest.fixture
def overview_preferences(private_ai_profile):
    preferences, _, _ = private_ai_profile
    preferences.set_reduce_motion(True)
    return preferences


@pytest.mark.parametrize('completion', ['cancel', 'accept'])
def test_delayed_metadata_delivery_cannot_hide_the_current_wizard(
        overview_preferences, mask, source, qtbot, monkeypatch, completion):
    tab, view = _open(mask, qtbot, source)
    page = tab._overview_page
    original_bytes = {path.name: path.read_bytes() for path in source.glob('*.tif')}
    _open_well(view, qtbot, 'A01')
    view.fields.setCurrentRow(0)
    qtbot.waitUntil(lambda: not view.fields.item(0).icon().isNull(), timeout=15000)
    picture = view.fields.item(0).icon().pixmap(96, 96).toImage().copy()
    selected = copy.deepcopy((view._selected_wells, view._selected_fields))
    export_settings = copy.deepcopy(tab._settings_model.collect())
    current_field = view.fields.currentItem().data(Qt.UserRole)

    receiver_name = '_wizard_closed' if completion == 'cancel' else '_show_browser'
    receive_finished = getattr(page, receiver_name)
    deferred = []
    monkeypatch.setattr(page, receiver_name, lambda owner: deferred.append(owner))
    tab._btn_run.click()
    old = page.wizard
    assert old is not None and page.stack.currentWidget() is old
    if completion == 'cancel':
        cancel = next(button for button in old.findChildren(QPushButton)
                      if button.text() == 'Cancel')
        cancel.click()
    else:
        old.preset.setCurrentIndex(old.preset.count() - 1)
        old.regex.setText(PATTERN)
        old._scan()
        qtbot.waitUntil(old.next.isEnabled, timeout=15000)
        old.next.click()
    assert deferred == [old] and old._finished
    tab._btn_run.click()
    current = page.wizard
    assert current is not None and current is not old and isValid(old)
    assert page.stack.currentWidget() is current
    receive_finished(deferred.pop())
    assert page.wizard is current
    assert page.stack.currentWidget() is current
    assert page.browser is view
    assert (view._selected_wells, view._selected_fields) == selected
    assert view.fields.currentItem().data(Qt.UserRole) == current_field
    assert view.fields.item(0).icon().pixmap(96, 96).toImage() == picture
    assert tab._settings_model.collect() == export_settings
    assert {path.name: path.read_bytes() for path in source.glob('*.tif')} == original_bytes

    monkeypatch.setattr(page, receiver_name, receive_finished)
    current_cancel = next(button for button in current.findChildren(QPushButton)
                          if button.text() == 'Cancel')
    current_cancel.click()
    assert page.wizard is None and page.stack.currentWidget() is view
    tab._btn_run.click()
    current = page.wizard
    assert current is not None and page.stack.currentWidget() is current
    current.preset.setCurrentIndex(current.preset.count() - 1)
    current.regex.setText(PATTERN)
    current._scan()
    qtbot.waitUntil(current.next.isEnabled, timeout=15000)
    current.next.click()
    qtbot.waitUntil(lambda: page.browser is not view and page.browser._normalization_ready,
                    timeout=15000)
    assert len(page.browser._fields) == 6
    assert {path.name: path.read_bytes() for path in source.glob('*.tif')} == original_bytes
