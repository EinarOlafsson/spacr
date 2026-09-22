"""A failed form read or refresh must not strand the grid's next real edit."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtWidgets import QLineEdit

from spacr.qt.widgets.object_grid_binding import ObjectGridBinding
from spacr.qt.widgets.object_settings_grid import ObjectSettingsGrid


class Panel:
    def __init__(self, grid):
        self.values = {'cell_channel': 0, 'cell_diameter': 16,
                       'nucleus_channel': 1, 'nucleus_diameter': 12}
        self._widgets = {key: QLineEdit(str(value), grid) for key, value in self.values.items()}
        self.fail_reads = False
        self.writes = []

    def collect(self):
        if self.fail_reads:
            raise RuntimeError('form is rebuilding')
        return {key: int(self._widgets[key].text()) for key in self.values}

    def set_value_for_key(self, key, value):
        if key not in self.values:
            return False
        self.writes.append((key, value))
        self._widgets[key].setText(str(value))
        return True


@pytest.fixture
def bound(qtbot, qt_theme_applied):
    grid = ObjectSettingsGrid()
    qtbot.addWidget(grid)
    panel = Panel(grid)
    binding = ObjectGridBinding(grid, panel, grid)
    binding.seed()
    return grid, panel, binding


def test_failed_form_read_preserves_cells_and_next_signal_updates_them(bound):
    grid, panel, binding = bound
    panel.fail_reads = True
    panel._widgets['cell_diameter'].setText('24')
    assert grid.settings()['cell_diameter'] == 16
    assert binding._busy is False
    panel.fail_reads = False
    panel._widgets['cell_diameter'].setText('25')
    assert grid.settings()['cell_diameter'] == 25
    assert panel.writes == []


def test_missing_form_key_keeps_its_cell_while_other_cells_follow(bound):
    grid, panel, binding = bound
    del panel.values['cell_diameter']
    panel._widgets['cell_channel'].setText('2')
    assert grid.settings()['cell_diameter'] == 16
    assert grid.settings()['cell_channel'] == 2
    assert binding._busy is False


def test_failed_signal_connection_can_be_replaced_with_a_working_editor(bound):
    grid, panel, binding = bound
    key = 'cell_diameter'
    binding._followed.discard(id(panel._widgets[key]))
    rejected = Mock(side_effect=RuntimeError('editor was deleted'))
    panel._widgets[key] = SimpleNamespace(textChanged=SimpleNamespace(connect=rejected))
    assert binding.follow_the_form() == 0
    assert rejected.call_count == 1
    assert id(panel._widgets[key]) not in binding._followed
    panel._widgets[key] = QLineEdit('16', grid)
    assert binding.follow_the_form() == 1
    assert binding.follow_the_form() == 0
    panel._widgets[key].setText('31')
    assert grid.settings()[key] == 31


def test_an_editor_without_change_signals_does_not_prevent_other_connections(bound):
    grid, panel, binding = bound
    binding._followed.clear()
    panel._widgets['cell_diameter'] = object()
    assert binding.follow_the_form() == 3
    assert id(panel._widgets['cell_diameter']) not in binding._followed


def test_visibility_refresh_failure_does_not_lose_the_committed_channel(bound):
    grid, panel, binding = bound
    panel.refresh_object_visibility = Mock(side_effect=RuntimeError('view is closing'))
    assert grid.set_value('channel', 'cell', '2')
    assert panel.collect()['cell_channel'] == 2
    assert panel.refresh_object_visibility.call_count == 1
    assert binding._busy is False
    panel.refresh_object_visibility.side_effect = None
    assert grid.set_value('channel', 'cell', '3')
    assert panel.collect()['cell_channel'] == 3
    assert panel.refresh_object_visibility.call_count == 2


def test_a_form_without_visibility_refresh_still_accepts_channel_edits(bound):
    grid, panel, binding = bound
    assert not hasattr(panel, 'refresh_object_visibility')
    assert grid.set_value('channel', 'cell', '4')
    assert panel.collect()['cell_channel'] == 4
    assert binding._busy is False


def test_an_unreadable_seed_releases_the_guard_for_a_later_retry(bound):
    grid, panel, binding = bound
    panel.fail_reads = True
    with pytest.raises(RuntimeError, match='rebuilding'):
        binding.seed()
    assert binding._busy is False
    panel.fail_reads = False
    binding.seed()
    assert grid.settings()['cell_diameter'] == 16
