"""The object settings grid binding keeps working when its widgets refuse a connection or a write."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtWidgets import QLineEdit

from spacr.qt.widgets.object_grid_binding import ObjectGridBinding, _filters_of
from spacr.qt.widgets.object_settings_grid import ObjectSettingsGrid

pytestmark = pytest.mark.qt


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


def test_a_calculate_switch_that_refuses_the_connection_does_not_stop_the_other_widgets(bound):
    grid, panel, binding = bound
    refused = Mock(side_effect=RuntimeError('switch was deleted'))
    panel._widgets['calculate_diameters'] = SimpleNamespace(
        clicked=SimpleNamespace(connect=refused))
    binding._followed.clear()
    connected = binding.follow_the_form()
    assert refused.call_count == 1
    assert connected == 4
    assert id(panel._widgets['calculate_diameters']) not in binding._followed


def test_an_editor_whose_commit_cannot_be_followed_is_still_followed_by_its_text(bound, monkeypatch):
    from spacr.qt.widgets import object_grid_binding

    grid, panel, binding = bound
    refused = Mock(side_effect=RuntimeError('commit signal gone'))
    monkeypatch.setattr(
        object_grid_binding, '_text_editor_of',
        lambda widget: SimpleNamespace(editingFinished=SimpleNamespace(connect=refused)))
    binding._followed.clear()
    assert binding.follow_the_form() == 4
    assert refused.call_count == 4
    monkeypatch.undo()
    panel._widgets['cell_diameter'].setText('41')
    assert grid.settings()['cell_diameter'] == 41


def test_a_redraw_that_comes_due_mid_write_waits_for_the_write(bound):
    grid, panel, binding = bound
    binding._busy = True
    binding._redraw_the_columns()
    assert binding._redraw.isActive()
    assert binding._busy is True
    binding._redraw.stop()
    binding._busy = False
    binding._redraw_the_columns()
    assert not binding._redraw.isActive()
    assert binding._busy is False


def test_a_calculate_click_during_a_write_is_ignored(bound):
    grid, panel, binding = bound
    reads = []
    panel.collect = lambda: reads.append(1) or {}
    binding._busy = True
    binding._calculate_clicked()
    assert reads == [] and panel.writes == []
    assert binding._busy is True


def test_turning_calculation_on_clears_only_the_diameters_that_hold_a_value(bound, monkeypatch):
    grid, panel, binding = bound
    current = {'calculate_diameters': True, 'cell_diameter': 15,
               'nucleus_diameter': None, 'magnification': 20}
    written = []
    panel.collect = lambda: dict(current)
    panel.set_value_for_key = lambda key, value: written.append((key, value)) or True
    monkeypatch.setattr(binding, '_show_what_the_panel_holds', lambda: 0)
    binding._calculate_clicked()
    assert written == [('cell_diameter', None)]
    assert binding._busy is False


def test_diameters_the_form_refuses_are_not_counted_as_written(bound):
    grid, panel, binding = bound
    attempts = []

    def refuse_the_nucleus(key, value):
        attempts.append(key)
        return key == 'cell_diameter'

    panel.set_value_for_key = refuse_the_nucleus
    current = {'calculate_diameters': False, 'magnification': 20,
               'cell_diameter': None, 'nucleus_diameter': None}
    assert binding._hold_the_diameters_still(current) == 1
    assert attempts == ['cell_diameter', 'nucleus_diameter']


def test_write_through_skips_keys_the_grid_lacks_and_writes_the_form_refuses(bound, monkeypatch):
    grid, panel, binding = bound
    monkeypatch.setattr(grid, 'settings', lambda: {})
    assert binding.write_through() == {}
    assert panel.writes == []
    monkeypatch.setattr(grid, 'settings', lambda: {'cell_diameter': 99})
    monkeypatch.setattr(panel, 'set_value_for_key', lambda key, value: False)
    assert binding.write_through() == {}
    assert panel._widgets['cell_diameter'].text() == '16'


def test_only_a_changed_channel_regates_the_form_and_a_failing_regate_is_survived(bound):
    grid, panel, binding = bound
    refresh = Mock()
    panel.refresh_object_visibility = refresh
    binding._reconsider_which_objects_the_run_has({'cell_diameter': 3})
    assert refresh.call_count == 0
    binding._reconsider_which_objects_the_run_has({'cell_channel': 2})
    assert refresh.call_count == 1
    refresh.side_effect = RuntimeError('form is rebuilding')
    binding._reconsider_which_objects_the_run_has({'nucleus_channel': 2})
    assert refresh.call_count == 2
    del panel.refresh_object_visibility
    binding._reconsider_which_objects_the_run_has({'cell_channel': 2})
    assert binding._busy is False


def test_a_grid_without_the_switch_hook_still_shows_the_panels_values(bound, monkeypatch):
    grid, panel, binding = bound
    monkeypatch.setattr(grid, '_keep_the_switches', None)
    panel._widgets['cell_diameter'].setText('55')
    assert grid.settings()['cell_diameter'] == 55
    assert binding._show_what_the_panel_holds() == 0


def test_an_unparseable_filter_value_is_compared_as_it_is(monkeypatch):
    import spacr.qt.mask_engine as mask_engine

    def broken(_value):
        raise ValueError('not a filter mapping')

    monkeypatch.setattr(mask_engine, 'parse_object_filters', broken)
    assert _filters_of('cell: ???') == 'cell: ???'
