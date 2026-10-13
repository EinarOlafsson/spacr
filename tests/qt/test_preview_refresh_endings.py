"""Refreshing rereads the form and handles absent, lazy or failed previews."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtWidgets import QWidget

from spacr.qt.widgets import preview_refresh as refresh


@pytest.mark.parametrize("value", [None, "", "path", "/path", "/path/to/src"])
def test_empty_and_placeholder_sources_report_without_loading(value):
    messages = []
    screen = SimpleNamespace(_settings_src_path=lambda: value,
                             _console=SimpleNamespace(append_stdout=messages.append))
    panel = SimpleNamespace(load_source_async=Mock())
    assert not refresh.reload_from_src(screen, panel)
    panel.load_source_async.assert_not_called()
    assert messages == ["Refresh: the source setting is empty.\n"]


@pytest.mark.parametrize("reader", [None, 42, Mock(side_effect=RuntimeError("form removed"))])
def test_missing_or_failed_form_reader_is_safe_without_a_console(reader):
    screen = SimpleNamespace(_settings_src_path=reader)
    assert refresh.current_src(screen) == ""
    assert not refresh.reload_from_src(screen, SimpleNamespace())


def test_missing_file_reports_the_actual_path_without_loading(tmp_path):
    path = tmp_path/'not-created'
    messages = []
    screen = SimpleNamespace(_settings_src_path=lambda: str(path),
                             _console=SimpleNamespace(append_stdout=messages.append))
    panel = SimpleNamespace(load_source_async=Mock())
    assert not refresh.reload_from_src(screen, panel)
    panel.load_source_async.assert_not_called()
    assert messages == [f"Refresh: {path} does not exist.\n"]


@pytest.mark.parametrize("name", refresh.LOADERS)
@pytest.mark.parametrize("started", [True, False])
def test_each_panel_loader_receives_the_current_trimmed_path(tmp_path, name, started):
    messages = []
    screen = SimpleNamespace(_settings_src_path=lambda: f"  {tmp_path}  ",
                             _console=SimpleNamespace(append_stdout=messages.append))
    loader = Mock(return_value=started)
    panel = SimpleNamespace(**{name: loader}, _auto_loaded_src="old")
    assert refresh.reload_from_src(screen, panel) is started
    loader.assert_called_once_with(str(tmp_path))
    assert panel._auto_loaded_src == str(tmp_path)
    assert messages == [f"Refresh: reloading the preview from {tmp_path}\n"]


def test_loader_priority_and_optional_automatic_load_marker(tmp_path):
    primary, alternate = Mock(return_value=True), Mock()
    panel = SimpleNamespace(load_source_async=primary, load_array_async=alternate)
    screen = SimpleNamespace(_settings_src_path=lambda: tmp_path,
                             _console=SimpleNamespace(append_stdout=None))
    assert refresh.reload_from_src(screen, panel)
    primary.assert_called_once_with(str(tmp_path))
    alternate.assert_not_called()
    assert not hasattr(panel, '_auto_loaded_src')


def test_a_panel_without_any_loader_does_not_claim_a_refresh(tmp_path):
    screen = SimpleNamespace(_settings_src_path=lambda: tmp_path)
    assert not refresh.reload_from_src(screen, SimpleNamespace(load_source_async=False))
    assert not refresh.reload_from_src(screen, None)


def test_refresh_button_resolves_lazy_panel_at_each_click(qtbot, tmp_path):
    card = QWidget()
    qtbot.addWidget(card)
    added = []
    card.add_title_action = added.append
    first = SimpleNamespace(load_array_async=Mock(return_value=True))
    second = SimpleNamespace(load_folder_async=Mock(return_value=True))
    panels = iter([first, second])
    screen = SimpleNamespace(_settings_src_path=lambda: tmp_path)
    getter = Mock(side_effect=lambda: next(panels))
    button = refresh.install_refresh_button(screen, card, None, panel_getter=getter)
    getter.assert_not_called()
    assert added == [button]
    button.click()
    screen._settings_src_path = lambda: str(tmp_path/'missing')
    button.click()
    assert getter.call_count == 2
    first.load_array_async.assert_called_once_with(str(tmp_path))
    second.load_folder_async.assert_not_called()
    assert refresh.install_refresh_button(screen, card, first) is button
    assert added == [button]


def test_cards_without_title_actions_are_left_alone():
    assert refresh.install_refresh_button(None, SimpleNamespace(), None) is None
    existing = object()
    card = SimpleNamespace(add_title_action=None, _refresh_button=existing)
    assert refresh.install_refresh_button(None, card, None) is existing
