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


def test_retained_real_channel_edit_receiver_ignores_a_deleted_owner(
        overview_preferences, source, qtbot, monkeypatch):
    from PySide6.QtCore import QCoreApplication, QEvent
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widgets import qc_field_browser as browser
    from tests.qt.test_725_overview_tab_round2 import _close

    original_receiver = browser._OverviewModulePage._check_channel_list
    received = []

    def observe(page, *args):
        result = original_receiver(page, *args)
        if args and isinstance(args[0], str):
            received.append((page, args))
        return result

    monkeypatch.setattr(browser._OverviewModulePage, '_check_channel_list', observe)
    originals = {path.name: path.read_bytes() for path in source.glob('*.tif')}
    old_owner = AppScreen('mask')
    old_owner.resize(1400, 1000)
    old_owner.show()
    try:
        old_tab, _ = _open(old_owner, qtbot, source)
        old_page = old_tab._overview_page
        old_editor = old_page._channel_list_edit
        received.clear()
        old_editor.setText('1')
        old_page._apply_bar.apply_button.click()
        retained_page, retained_args = next((page, args) for page, args in received
                                            if page is old_page and args == ('1',))
        assert not old_editor.property('overviewInvalid')
        _close(old_owner, qtbot)
        old_owner.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        assert not isValid(old_owner) and not isValid(old_editor)

        fresh_owner = AppScreen('mask')
        qtbot.addWidget(fresh_owner, before_close_func=lambda widget: _close(widget, qtbot))
        fresh_owner.resize(1400, 1000)
        fresh_owner.show()
        fresh_tab, fresh_view = _open(fresh_owner, qtbot, source)
        fresh_page = fresh_tab._overview_page
        fresh_editor = fresh_page._channel_list_edit
        _open_well(fresh_view, qtbot, 'A01')
        fresh_view.fields.setCurrentRow(0)
        qtbot.waitUntil(lambda: not fresh_view.fields.item(0).icon().isNull(), timeout=15000)
        image = fresh_view.fields.item(0).icon().pixmap(96, 96).toImage().copy()
        selection = copy.deepcopy((fresh_view._selected_wells, fresh_view._selected_fields))
        settings = copy.deepcopy(fresh_tab._settings_model.collect())
        status = fresh_view.status.text()
        assert original_receiver(retained_page, *retained_args) == ''
        assert fresh_page.browser is fresh_view and fresh_page.stack.currentWidget() is fresh_view
        assert fresh_view.status.text() == status
        assert fresh_tab._settings_model.collect() == settings
        assert (fresh_view._selected_wells, fresh_view._selected_fields) == selection
        assert fresh_view.fields.item(0).icon().pixmap(96, 96).toImage() == image
        assert {path.name: path.read_bytes() for path in source.glob('*.tif')} == originals

        fresh_editor.setText('77')
        assert fresh_editor.property('overviewInvalid')
        assert not fresh_page._apply_bar.apply_button.isEnabled()
        fresh_editor.setText('1')
        assert not fresh_editor.property('overviewInvalid')
        fresh_page._apply_bar.apply_button.click()
        assert fresh_view._normalize_channels == (1,)
        assert {path.name: path.read_bytes() for path in source.glob('*.tif')} == originals
    finally:
        if isValid(old_owner):
            _close(old_owner, qtbot)
            old_owner.deleteLater()


def test_native_mask_editor_restores_its_panels_after_overview_options_and_replacement(
        overview_preferences, source, tmp_path, qtbot):
    from spacr.qt.screens.make_masks import MakeMasksScreen
    from spacr.qt.widgets import qc_field_browser as browser

    host = MakeMasksScreen()
    workspaces = []

    def close_owner(_widget):
        for workspace in reversed(workspaces):
            if not isValid(workspace):
                continue
            workspace._retire()
            for index in range(workspace.stack.count()):
                child = workspace.stack.widget(index)
                names = ('_index_jobs', '_image_jobs', '_export_jobs', '_metadata_jobs',
                         '_measurement_jobs', '_native_jobs', '_thumbnail_jobs',
                         '_montage_jobs', '_field_thumbnail_jobs')
                runners = [getattr(child, name) for name in names if hasattr(child, name)]
                if hasattr(child, '_runner'):
                    runners.append(child._runner)
                for runner in runners:
                    runner.shutdown(timeout_ms=3000)

    qtbot.addWidget(host, before_close_func=close_owner)
    host.resize(1400, 1000)
    host.show()
    original_settings = host._settings_scroll.widget()
    original_image = host._body_stack.currentWidget()
    original_policy = original_image.sizePolicy()
    source_bytes = {path.name: path.read_bytes() for path in source.glob('*.tif')}
    assert getattr(host, '_fold_pages', None) is None
    wizard = browser._open_field_overview(host, {'source': str(source), 'geometry': (2, 3)})
    workspace = host._field_overview_workspace
    workspaces.append(workspace)
    assert workspace._module_mounted and host._body_stack.currentWidget() is workspace
    assert host._settings_scroll.widget() is not original_settings
    wizard.preset.setCurrentIndex(wizard.preset.count() - 1)
    wizard.regex.setText(PATTERN)
    wizard._scan()
    qtbot.waitUntil(wizard.next.isEnabled, timeout=15000)
    wizard.next.click()
    view = host._field_overview
    qtbot.waitUntil(lambda: view._normalization_ready, timeout=15000)
    assert len(view._fields) == 6
    settings_content = view._settings_content
    assert host._settings_scroll.widget() is settings_content
    view.options_button.click()
    assert view._options_window.isVisible()
    assert view._controls_scroll.widget() is settings_content
    assert host._settings_scroll.widget() is None
    workspace.back_button.click()
    assert host._body_stack.currentWidget() is original_image
    assert host._settings_scroll.widget() is original_settings
    assert original_image.sizePolicy() == original_policy
    assert view._controls_scroll.widget() is settings_content
    assert not view._options_window.isVisible()

    assert browser._open_field_overview(host, {'source': str(source)}) is wizard
    assert host._field_overview_workspace is workspace
    assert workspace.stack.currentWidget() is view
    assert host._settings_scroll.widget() is settings_content
    replacement = tmp_path / 'replacement'
    replacement.mkdir()
    for name, payload in source_bytes.items():
        (replacement / name).write_bytes(payload)
    next_wizard = browser._open_field_overview(host, {'source': str(replacement)})
    replacement_workspace = host._field_overview_workspace
    workspaces.append(replacement_workspace)
    assert next_wizard is not wizard
    cancel = next(button for button in next_wizard.findChildren(QPushButton)
                  if button.text() == 'Cancel')
    cancel.click()
    assert host._field_overview_workspace is workspace
    assert workspace.stack.currentWidget() is view
    assert host._body_stack.currentWidget() is workspace
    assert host._settings_scroll.widget() is settings_content
    assert len(view._fields) == 6
    workspace.back_button.click()
    assert host._settings_scroll.widget() is original_settings
    assert host._body_stack.currentWidget() is original_image
    assert original_image.sizePolicy() == original_policy
    assert {path.name: path.read_bytes() for path in source.glob('*.tif')} == source_bytes
    assert {path.name: path.read_bytes() for path in replacement.glob('*.tif')} == source_bytes


def test_retained_apply_and_layout_receivers_leave_the_replacement_browser_owned(
        overview_preferences, source, qtbot, monkeypatch):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widgets import qc_field_browser as browser
    from tests.qt.test_725_overview_tab_round2 import _close

    apply_received, fit_received = [], []
    apply_original = browser._OverviewModulePage._apply_settings
    fit_original = browser._FieldOverview._fit_module_actions

    def apply(page, *args):
        apply_received.append((page, args))
        return apply_original(page, *args)

    def fit(view, *args):
        fit_received.append((view, args))
        return fit_original(view, *args)

    monkeypatch.setattr(browser._OverviewModulePage, '_apply_settings', apply)
    monkeypatch.setattr(browser._FieldOverview, '_fit_module_actions', fit)
    old_owner = AppScreen('mask')
    qtbot.addWidget(old_owner, before_close_func=lambda widget: _close(widget, qtbot))
    old_owner.resize(1400, 1000)
    old_owner.show()
    old_tab, old_view = _open(old_owner, qtbot, source)
    old_tab._settings_model.set_value_for_key('overview_percentile_low', 5)
    old_tab._overview_page._apply_bar.apply_button.click()
    qtbot.waitUntil(lambda: any(view is old_view for view, _ in fit_received), timeout=5000)
    old_page, apply_args = next((page, args) for page, args in apply_received
                                if page is old_tab._overview_page)
    fit_view, fit_args = next((view, args) for view, args in fit_received if view is old_view)
    assert old_view.low.value() == 5
    _close(old_owner, qtbot)
    assert old_page._bridge is None and old_view._module_actions_retired

    current_owner = AppScreen('mask')
    qtbot.addWidget(current_owner, before_close_func=lambda widget: _close(widget, qtbot))
    current_owner.resize(1400, 1000)
    current_owner.show()
    tab, view = _open(current_owner, qtbot, source)
    _open_well(view, qtbot, 'A01')
    view.fields.setCurrentRow(0)
    qtbot.waitUntil(lambda: not view.fields.item(0).icon().isNull(), timeout=15000)
    image = view.fields.item(0).icon().pixmap(96, 96).toImage().copy()
    selected = copy.deepcopy((view._selected_wells, view._selected_fields))
    settings = copy.deepcopy(tab._settings_model.collect())
    source_bytes = {path.name: path.read_bytes() for path in source.glob('*.tif')}
    apply_original(old_page, *apply_args)
    fit_original(fit_view, *fit_args)
    assert tab._overview_page.browser is view
    assert tab._overview_page.stack.currentWidget() is view
    assert tab._settings_model.collect() == settings
    assert (view._selected_wells, view._selected_fields) == selected
    assert view.fields.item(0).icon().pixmap(96, 96).toImage() == image
    tab._settings_model.set_value_for_key('overview_percentile_low', 7)
    tab._overview_page._apply_bar.apply_button.click()
    assert view.low.value() == 7
    assert {path.name: path.read_bytes() for path in source.glob('*.tif')} == source_bytes


def test_retained_image_menu_request_cannot_retarget_a_closed_stacked_image(
        overview_preferences, mask, source, qtbot, monkeypatch):
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QContextMenuEvent
    from PySide6.QtWidgets import QApplication
    from spacr.qt.widgets import qc_field_browser as browser

    received = []
    original_menu = browser._FieldOverview._image_menu

    def observe(owner, image_view, position):
        received.append((owner, image_view, QPoint(position)))
        return original_menu(owner, image_view, position)

    monkeypatch.setattr(browser._FieldOverview, '_image_menu', observe)
    tab, view = _open(mask, qtbot, source)
    _open_well(view, qtbot, 'A01')
    view.fields.item(0).setSelected(True)
    view.fields.item(1).setSelected(True)
    assert len(view._selected_fields) == 2
    view._open_selected_images()
    old, current = view._image_views
    qtbot.waitUntil(lambda: all(panel.view._item is not None for panel in (old, current)),
                    timeout=15000)

    def request_menu(image_view):
        viewport = image_view.viewport()
        point = viewport.rect().center()
        event = QContextMenuEvent(QContextMenuEvent.Mouse, point, viewport.mapToGlobal(point))
        QApplication.sendEvent(viewport, event)
        return view._image_menu_shown

    old_menu = request_menu(old.view)
    assert received[-1][1] is old.view
    assert old_menu is not None and old_menu.isVisible()
    assert any(action.text() == 'Image settings…' for action in old_menu.actions())
    old_owner, old_view, old_position = received[-1]
    old_menu.close()
    image = current.view._item.pixmap().toImage().copy()
    selected = copy.deepcopy((view._selected_wells, view._selected_fields))
    settings = copy.deepcopy(tab._settings_model.collect())
    original_bytes = {path.name: path.read_bytes() for path in source.glob('*.tif')}
    old.close_button.click()
    assert view._image_views == [current] and isValid(old_view)
    assert view._image_stack.indexOf(current) >= 0
    original_menu(old_owner, old_view, old_position)
    assert view._image_menu_shown is old_menu
    assert view._image_views == [current]
    assert current.view._item.pixmap().toImage() == image
    assert (view._selected_wells, view._selected_fields) == selected
    assert tab._settings_model.collect() == settings
    current_menu = request_menu(current.view)
    assert received[-1][1] is current.view
    assert current_menu is not old_menu and current_menu.isVisible()
    assert any(action.text() == 'Image settings…' for action in current_menu.actions())
    assert any(action.objectName() == 'OverviewImage_link_zoom'
               for action in current_menu.actions())
    current_menu.close()
    assert {path.name: path.read_bytes() for path in source.glob('*.tif')} == original_bytes
