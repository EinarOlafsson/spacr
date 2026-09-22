"""Crop-loading failures retain their cause and grayscale pages retain their pixels."""
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from spacr import crop_loader as loader
from spacr.crops import CropError


def _plan(source=None):
    return loader.CropPlan(query=loader.CropQuery(source='folder', path='unused'),
                           rows=('one', 'two'), matched=2, _source=source)


def test_a_plan_without_a_pixel_source_is_refused():
    with pytest.raises(loader.CropLoadError, match='no crop source'):
        loader.load_page(_plan(), 0, 1)


def test_an_empty_page_never_reads_the_pixel_source():
    source=SimpleNamespace(get_many=Mock())
    with pytest.raises(loader.CropLoadError, match='page 2:3.*empty'):
        loader.load_page(_plan(source), 2, 3)
    source.get_many.assert_not_called()


@pytest.mark.parametrize('problem', [CropError('bad mask'), OSError('missing pixels'), ValueError('bad image')])
def test_source_failures_keep_their_cause_and_page_context(problem):
    source=SimpleNamespace(kind='png', get_many=Mock(side_effect=problem))
    with pytest.raises(loader.CropLoadError, match='crops 0-2 could not be read') as error:
        loader.load_page(_plan(source), 0, 2)
    assert error.value.__cause__ is problem
    assert str(problem) in str(error.value)
    source.get_many.assert_called_once_with(('one', 'two'))


def test_two_dimensional_grayscale_pages_gain_only_a_channel_axis():
    first=np.array([[1, 2], [3, 4]], dtype=np.uint16)
    second=np.array([[5, 6], [7, 8]], dtype=np.uint16)
    source=SimpleNamespace(get_many=Mock(return_value=[first, second]))
    result, conformed=loader.load_page(_plan(source), 0, 2)
    assert result.shape == (2, 2, 2, 1) and result.dtype == np.uint16
    np.testing.assert_array_equal(result[0, ..., 0], first)
    np.testing.assert_array_equal(result[1, ..., 0], second)
    assert conformed == 0


def test_invalid_page_size_is_named_without_reading_pixels():
    with pytest.raises(loader.CropLoadError, match='page_size must be at least 1'):
        _plan().pages(-1)


def test_missing_database_and_folder_choices_are_distinguished():
    with pytest.raises(loader.CropLoadError, match='no .*chosen'):
        loader.object_classes('')
    with pytest.raises(loader.CropLoadError, match='no crop folder was chosen'):
        loader.plan_from_folder(loader.CropQuery(source='folder', path=''))


def test_an_empty_parent_directory_points_to_its_actual_child_folder(tmp_path):
    (tmp_path/'plate1_png').mkdir()
    (tmp_path/'.hidden').mkdir()
    plan=loader.plan_from_folder(loader.CropQuery(source='folder', path=str(tmp_path)))
    assert plan.is_empty
    assert 'plate1_png' in plan.empty_reason and 'Choose one of those' in plan.empty_reason
    assert '.hidden' not in plan.empty_reason


def test_a_folder_that_becomes_unreadable_during_planning_reports_its_path(tmp_path, monkeypatch):
    monkeypatch.setattr(loader.os, 'scandir', Mock(side_effect=[iter(()), PermissionError('denied')]))
    with pytest.raises(loader.CropLoadError, match='cannot list crop folder') as error:
        loader.plan_from_folder(loader.CropQuery(source='folder', path=str(tmp_path)))
    assert str(tmp_path) in str(error.value) and 'denied' in str(error.value)


def test_legacy_database_without_plate_metadata_has_no_plate_choices(tmp_path):
    database=tmp_path/'measurements.db'
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE png_list (cell_id INTEGER)')
    assert loader.plates(str(database)) == ()
    assert loader.object_classes(str(database)) == ()


def test_unavailable_sqlite_column_metadata_is_not_reported_as_a_valid_table():
    connection=sqlite3.connect(':memory:')
    connection.close()
    assert loader._table_columns(connection, 'png_list') == ()
