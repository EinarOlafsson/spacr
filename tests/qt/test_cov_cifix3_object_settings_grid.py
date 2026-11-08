"""The object settings grid's plane, check-state and follow-the-form helpers keep their edge-case answers."""
from __future__ import annotations

import pytest
from PySide6.QtCore import Qt

from spacr.qt.widgets import object_settings_grid as osg
from spacr.qt.widgets.object_settings_grid import ObjectSettingsGrid

pytestmark = pytest.mark.qt


@pytest.mark.parametrize('value,expected', [
    (None, False), (True, False), (False, False), (0, True), (2.5, True),
    ('', False), ('  ', False), ('None', False), ('none', False),
    ('3', True), ('1.5', True), ('abc', False)])
def test_a_channel_value_names_a_plane_only_when_it_is_a_number(value, expected):
    assert osg._names_a_plane(value) is expected


def test_a_filter_row_edits_the_object_filters_key_and_other_rows_their_own_key():
    assert osg._cell_key('cell', 'filter:area') == 'object_filters'
    assert osg._is_filter('filter:area') is True
    assert osg._cell_key('nucleus', 'diameter') == 'nucleus_diameter'


@pytest.mark.parametrize('value,expected', [
    (True, True), (False, False), (Qt.CheckState.Checked, True),
    (Qt.CheckState.Unchecked, False), (2, True), (0, False),
    ('yes', True), ('on', True), ('true', True), ('1', False), ('no', False), ('', False),
    (object(), False), (None, False)])
def test_a_check_state_is_read_from_a_bool_a_state_an_integer_or_a_word(value, expected):
    assert osg._checked(value) is expected


def test_an_object_without_a_switch_keeps_its_column_and_a_set_switch_shows_it():
    assert osg._object_is_in_the_run('cell', {}) is True
    assert osg._object_is_in_the_run('cytoplasm', {'cell_channel': 0}) is True
    assert osg._object_is_in_the_run('nucleus', {'nucleus_channel': 1}) is True
    assert osg._object_is_in_the_run('nucleus', {'nucleus_channel': None}) is False


@pytest.fixture
def grid(qtbot, qt_theme_applied):
    widget = ObjectSettingsGrid()
    qtbot.addWidget(widget)
    widget.set_settings({'cell_channel': 0, 'cell_diameter': None,
                         'nucleus_channel': 1, 'nucleus_diameter': None,
                         'magnification': 20, 'pixel_size_um': None,
                         'calculate_diameters': False, 'object_filters': {}})
    return widget


def test_a_new_pixel_size_redraws_the_diameter_row_once_and_a_repeat_changes_nothing(grid, qtbot):
    with qtbot.waitSignal(grid._model.dataChanged, timeout=1000):
        assert grid._follow_pixel_size(0.65) is True
    assert grid._model._pixel_size == 0.65
    assert grid._base['pixel_size_um'] == 0.65
    assert grid._follow_pixel_size(0.65) is False


def test_a_new_calculation_switch_redraws_the_diameter_row_once(grid, qtbot):
    with qtbot.waitSignal(grid._model.dataChanged, timeout=1000):
        assert grid._follow_calculation(True) is True
    assert grid._model._calculating is True
    assert grid._follow_calculation(True) is False


def test_following_the_form_without_a_diameter_row_redraws_nothing(grid, monkeypatch):
    monkeypatch.setattr(grid, 'questions', lambda: ['channel'])
    assert grid._follow_pixel_size(0.3) is False
    assert grid._follow_calculation(True) is False
    assert grid._follow_magnification(40) is False


def test_following_the_form_with_no_columns_still_reports_a_redraw(grid, monkeypatch):
    monkeypatch.setattr(grid._model, 'columnCount', lambda *a: 0)
    assert grid._follow_pixel_size(0.9) is True
    assert grid._follow_calculation(True) is True
    assert grid._follow_magnification(63) is True


def test_a_filter_row_added_to_the_grid_is_listed_and_a_repeat_is_refused(grid, qtbot):
    with qtbot.waitSignal(grid.settings_changed, timeout=1000):
        grid._added('area')
    assert 'area' in grid.filter_properties()
    assert grid.add_filter('area') is False
